//! Bounded central-path candidate; original contact laws alone accept impulses.

mod equations;
mod linear;
use super::{BlockLayout, ConstraintBlock, Continuation, DynamicsFactor, dot};

struct Point {
    first: usize,
    size: usize,
    friction: f64,
    rolling: Option<f64>,
    bounds: Option<crate::ImpulseBounds>,
}
struct Problem<'a> {
    size: usize,
    rows: Vec<&'a Vec<f64>>,
    response: Vec<Vec<f64>>,
    targets: Vec<f64>,
    scales: Vec<f64>,
    units: Vec<f64>,
    points: Vec<Point>,
}

impl Problem<'_> {
    fn motion(&self, impulses: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.size];
        for (response, impulse) in self.response.iter().zip(impulses) {
            for (value, component) in result.iter_mut().zip(response) {
                *value += impulse * component;
            }
        }
        result
    }
}

#[derive(Clone, Copy)]
struct ArcRow<'a> {
    tangent: &'a [f64],
    rhs: f64,
    homogeneous: bool,
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(in crate::response) fn propose(
    factor: &DynamicsFactor,
    blocks: &[ConstraintBlock],
    layout: &[BlockLayout],
    modes: &[Vec<bool>],
    maximum_iterations: usize,
    tolerance: f64,
) -> Continuation {
    let mut result = Continuation::default();
    if factor.size == 0 || factor.size > 64 || maximum_iterations == 0 {
        return result;
    }
    let rows = blocks.iter().flat_map(|b| &b.jacobian).collect::<Vec<_>>();
    let mut problem = Problem {
        size: factor.size,
        rows,
        response: Vec::new(),
        targets: Vec::new(),
        scales: Vec::new(),
        units: Vec::new(),
        points: Vec::new(),
    };
    for ((block, info), modes) in blocks.iter().zip(layout).zip(modes) {
        problem.targets.extend(&block.target);
        problem
            .scales
            .extend(std::iter::repeat_n(info.scale, block.target.len()));
        let mut first = info.first;
        if block.contacts.is_empty() {
            for bounds in &block.bounds {
                problem.points.push(Point {
                    first,
                    size: 1,
                    friction: 0.0,
                    rolling: None,
                    bounds: Some(*bounds),
                });
                let width = bounds.maximum - bounds.minimum;
                problem.units.push(if width.is_finite() {
                    width.max(1e-6)
                } else {
                    100.0
                });
                first += 1;
            }
            continue;
        }
        for (contact, sliding) in block.contacts.iter().zip(modes) {
            let friction = if *sliding {
                contact.kinetic_coefficient
            } else {
                contact.static_coefficient
            };
            if friction <= 0.0
                || contact.rolling_length.is_some_and(|r| r <= 0.0)
                || block.bounds[first - info.first].maximum.is_finite()
            {
                return result;
            }
            problem.points.push(Point {
                first,
                size: contact.rows(),
                friction,
                rolling: contact.rolling_length,
                bounds: None,
            });
            problem.units.extend([100.0; 3]);
            if let Some(length) = contact.rolling_length {
                problem.units.extend([100.0 * length; 2]);
            }
            first += contact.rows();
        }
    }
    if problem.size + problem.points.len() + 1 > 128 {
        return result;
    }
    for row in &problem.rows {
        let mut response = (*row).clone();
        result.work.factor_solves += 1;
        if factor.solve(&mut response).is_err() {
            return result;
        }
        problem.response.push(response);
    }
    result.work.reduced_storage = problem.rows.len() * problem.size;
    let mut impulses = vec![0.0; problem.rows.len()];
    for point in &problem.points {
        impulses[point.first] = point.bounds.map_or(10.0, |bounds| {
            match (bounds.minimum.is_finite(), bounds.maximum.is_finite()) {
                (true, true) => 0.5 * bounds.minimum + 0.5 * bounds.maximum,
                (true, false) => bounds.minimum + 10.0,
                (false, true) => bounds.maximum - 10.0,
                (false, false) => 0.0,
            }
        });
    }
    let mut epsilon = 1.0_f64;
    for _ in 0..6 {
        result.stages += 1;
        #[cfg(test)]
        if std::env::var_os("MECHANIC_TRACE_CONTINUATION").is_some() {
            eprintln!(
                "central coarse epsilon={epsilon:e} iterations={}",
                result.iterations
            );
        }
        if !refine(
            &problem,
            &mut impulses,
            epsilon,
            false,
            (epsilon * 1e-5).max(1e-11),
            maximum_iterations,
            &mut result,
        ) {
            return result;
        }
        if consider(
            &problem,
            blocks,
            layout,
            modes,
            &impulses,
            tolerance,
            &mut result,
        ) {
            return result;
        }
        epsilon *= 0.1;
    }
    // The last solved central parameter is 1e-5. A bordered predictor/corrector
    // can follow folds that a monotone reduction would skip.
    epsilon *= 10.0;
    let mut z = impulses
        .iter()
        .zip(&problem.units)
        .map(|(x, u)| x / u)
        .collect::<Vec<_>>();
    z.push(epsilon.ln());
    let mut previous = vec![0.0; z.len()];
    *previous.last_mut().expect("parameter row") = -1.0;
    let mut step = 0.5_f64;
    for _ in 0..64 {
        result.stages += 1;
        #[cfg(test)]
        if std::env::var_os("MECHANIC_TRACE_CONTINUATION").is_some() {
            eprintln!(
                "central arc epsilon={epsilon:e} iterations={} step={step}",
                result.iterations
            );
        }
        let Some(at) = evaluate(&problem, &impulses, epsilon, false, &mut result) else {
            return result;
        };
        if result.iterations == maximum_iterations {
            return result;
        }
        result.iterations += 1;
        let Some(mut tangent) = linear::direction(
            &problem,
            &at,
            false,
            Some(ArcRow {
                tangent: &previous,
                rhs: 1.0,
                homogeneous: true,
            }),
            &mut result.work,
        ) else {
            return result;
        };
        let norm = magnitude(&tangent);
        if norm <= 0.0 {
            return result;
        }
        let sign = if dot(&tangent, &previous) < 0.0 {
            -1.0
        } else {
            1.0
        };
        for value in &mut tangent {
            *value *= sign / norm;
        }
        let mut accepted = None;
        for _ in 0..12 {
            let predicted = z
                .iter()
                .zip(&tangent)
                .map(|(value, direction)| value + step * direction)
                .collect::<Vec<_>>();
            if let Some((candidate, iterations)) = correct_arc(
                &problem,
                &predicted,
                &tangent,
                maximum_iterations,
                &mut result,
            ) {
                accepted = Some((candidate, iterations));
                break;
            }
            step *= 0.5;
        }
        let Some((candidate, corrections)) = accepted else {
            return result;
        };
        z = candidate;
        impulses = z
            .iter()
            .zip(&problem.units)
            .map(|(value, unit)| value * unit)
            .collect();
        epsilon = z[problem.rows.len()].exp();
        previous = tangent;
        if consider(
            &problem,
            blocks,
            layout,
            modes,
            &impulses,
            tolerance,
            &mut result,
        ) {
            return result;
        }
        if epsilon <= 1e-6 {
            break;
        }
        if corrections < 5 {
            step = (step * 1.5).min(1.0);
        }
    }
    epsilon = 1e-6;
    for _ in 0..15 {
        result.stages += 1;
        // Stable disk elimination remains defined outside a strict barrier disk;
        // only positive normals are required during this finishing search.
        refine(
            &problem,
            &mut impulses,
            epsilon,
            true,
            (epsilon * 1e-4).max(2e-15),
            maximum_iterations,
            &mut result,
        );
        #[cfg(test)]
        if std::env::var_os("MECHANIC_TRACE_CONTINUATION").is_some() {
            eprintln!("central epsilon={epsilon:e} work={}", result.iterations);
        }
        if consider(
            &problem,
            blocks,
            layout,
            modes,
            &impulses,
            tolerance,
            &mut result,
        ) {
            return result;
        }
        if result.iterations == maximum_iterations {
            return result;
        }
        epsilon *= 0.01;
    }
    result
}

fn evaluate(
    problem: &Problem<'_>,
    x: &[f64],
    epsilon: f64,
    stable: bool,
    result: &mut Continuation,
) -> Option<equations::Evaluation> {
    result.evaluations += 1;
    result.work.applications += 1;
    equations::evaluate(problem, x, epsilon, stable)
}

#[allow(clippy::too_many_arguments)]
fn refine(
    problem: &Problem<'_>,
    x: &mut Vec<f64>,
    epsilon: f64,
    stable: bool,
    threshold: f64,
    maximum_iterations: usize,
    result: &mut Continuation,
) -> bool {
    for _ in 0..32 {
        let Some(at) = evaluate(problem, x, epsilon, stable, result) else {
            return false;
        };
        if at.maximum < threshold {
            return true;
        }
        if result.iterations == maximum_iterations {
            return false;
        }
        result.iterations += 1;
        let Some(direction) = linear::direction(problem, &at, stable, None, &mut result.work)
        else {
            return false;
        };
        let mut best = None;
        let mut norm = at.norm;
        let mut fraction = 1.0;
        for _ in 0..32 {
            let candidate = x
                .iter()
                .zip(&direction)
                .map(|(value, direction)| value + fraction * direction)
                .collect::<Vec<_>>();
            if let Some(trial) = evaluate(problem, &candidate, epsilon, stable, result)
                && trial.norm < norm
            {
                norm = trial.norm;
                best = Some(candidate);
                if norm < at.norm * (1.0 - 1e-4 * fraction) {
                    break;
                }
            }
            fraction *= 0.5;
        }
        let Some(candidate) = best else {
            return false;
        };
        *x = candidate;
    }
    false
}

#[allow(clippy::too_many_lines)]
fn correct_arc(
    problem: &Problem<'_>,
    predicted: &[f64],
    tangent: &[f64],
    maximum_iterations: usize,
    result: &mut Continuation,
) -> Option<(Vec<f64>, usize)> {
    let mut z = predicted.to_vec();
    for iteration in 0..24 {
        let x = z
            .iter()
            .zip(&problem.units)
            .map(|(value, unit)| value * unit)
            .collect::<Vec<_>>();
        let epsilon = z[problem.rows.len()].exp();
        let at = evaluate(problem, &x, epsilon, false, result)?;
        let arc = z
            .iter()
            .zip(predicted)
            .zip(tangent)
            .map(|((value, initial), direction)| (value - initial) * direction)
            .sum::<f64>();
        if at.maximum < 1e-10 && arc.abs() < 1e-10 {
            return Some((z, iteration));
        }
        if result.iterations == maximum_iterations {
            return None;
        }
        result.iterations += 1;
        let direction = linear::direction(
            problem,
            &at,
            false,
            Some(ArcRow {
                tangent,
                rhs: -arc,
                homogeneous: false,
            }),
            &mut result.work,
        )?;
        // Row equilibration of the full equation is evaluated one coefficient at
        // a time. No contact-response matrix is stored above the 128-row limit.
        let mut scales = Vec::with_capacity(z.len());
        for local in &at.locals {
            for row in 0..local.size {
                let coupling = &local.velocity[row * problem.size..(row + 1) * problem.size];
                let mut largest = local.parameter[row].abs();
                for (column, response) in problem.response.iter().enumerate() {
                    let diagonal = if (local.first..local.first + local.size).contains(&column) {
                        local.matrix[row * local.size + column - local.first]
                    } else {
                        0.0
                    };
                    largest = largest
                        .max(((diagonal + dot(coupling, response)) * problem.units[column]).abs());
                }
                scales.push(largest);
            }
        }
        scales.push(tangent.iter().map(|v| v.abs()).fold(0.0, f64::max));
        let norm = scaled_norm(&at.residual, arc, &scales);
        let mut fraction = 1.0;
        let mut accepted = None;
        for _ in 0..32 {
            let candidate = z
                .iter()
                .zip(&direction)
                .map(|(value, direction)| value + fraction * direction)
                .collect::<Vec<_>>();
            if candidate[problem.rows.len()].abs() <= 100.0 {
                let x = candidate
                    .iter()
                    .zip(&problem.units)
                    .map(|(value, unit)| value * unit)
                    .collect::<Vec<_>>();
                if let Some(trial) = evaluate(
                    problem,
                    &x,
                    candidate[problem.rows.len()].exp(),
                    false,
                    result,
                ) {
                    let arc = candidate
                        .iter()
                        .zip(predicted)
                        .zip(tangent)
                        .map(|((value, initial), direction)| (value - initial) * direction)
                        .sum::<f64>();
                    if scaled_norm(&trial.residual, arc, &scales) < norm {
                        accepted = Some(candidate);
                        break;
                    }
                }
            }
            fraction *= 0.5;
        }
        z = accepted?;
    }
    None
}

fn magnitude(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |sum, v| sum.hypot(*v))
}
fn scaled_norm(values: &[f64], arc: f64, scales: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .chain(std::iter::once(arc))
        .zip(scales)
        .fold(0.0, |sum, (v, scale)| sum.hypot(v / scale))
}

#[allow(clippy::too_many_arguments)]
fn consider(
    problem: &Problem<'_>,
    blocks: &[ConstraintBlock],
    layout: &[BlockLayout],
    modes: &[Vec<bool>],
    x: &[f64],
    tolerance: f64,
    result: &mut Continuation,
) -> bool {
    // Both validation motions apply the cached generalized response. Include
    // them even when this candidate fails the original contact laws.
    result.work.applications += 2;
    let velocity = problem.motion(x);
    let mut candidate = x
        .iter()
        .zip(&problem.rows)
        .zip(&problem.targets)
        .zip(&problem.scales)
        .map(|(((value, row), target), scale)| value + (target - dot(row, &velocity)) / scale)
        .collect::<Vec<_>>();
    for ((block, info), mode) in blocks.iter().zip(layout).zip(modes) {
        super::super::super::project(
            block,
            mode,
            &mut candidate[info.first..info.first + block.target.len()],
        );
    }
    let velocity = problem.motion(&candidate);
    let changed = problem
        .rows
        .iter()
        .map(|row| dot(row, &velocity))
        .collect::<Vec<_>>();
    let Ok(residual) = super::super::super::projected_residual(
        blocks, layout, &changed, &candidate, modes, tolerance,
    ) else {
        return false;
    };
    if result.impulses.is_none() || residual < result.residual {
        result.impulses = Some(candidate);
        result.residual = residual;
    }
    residual <= tolerance
}
