//! Barrier equations for search only; no compliance enters physical acceptance.

use super::{Problem, dot};

pub(super) struct Local {
    pub first: usize,
    pub size: usize,
    pub matrix: Vec<f64>,
    pub velocity: Vec<f64>,
    pub parameter: Vec<f64>,
    pub rhs: Vec<f64>,
}

pub(super) struct Evaluation {
    pub locals: Vec<Local>,
    pub residual: Vec<f64>,
    pub norm: f64,
    pub maximum: f64,
}

#[allow(clippy::too_many_lines)]
pub(super) fn evaluate(
    problem: &Problem<'_>,
    x: &[f64],
    epsilon: f64,
    stable: bool,
) -> Option<Evaluation> {
    let velocity = problem.motion(x);
    let motion = problem
        .rows
        .iter()
        .zip(&problem.targets)
        .map(|(row, target)| dot(row, &velocity) - target)
        .collect::<Vec<_>>();
    let mut result = Evaluation {
        locals: Vec::new(),
        residual: Vec::new(),
        norm: 0.0,
        maximum: 0.0,
    };
    for point in &problem.points {
        let first = point.first;
        let size = point.size;
        let normal = x[first];
        if !normal.is_finite() || (point.bounds.is_none() && normal <= 0.0) {
            return None;
        }
        let mut local = Local {
            first,
            size,
            matrix: vec![0.0; size * size],
            velocity: vec![0.0; size * problem.size],
            parameter: vec![0.0; size],
            rhs: vec![0.0; size],
        };
        let mut residual = motion[first..first + size].to_vec();
        if let Some(bounds) = point.bounds {
            scalar(
                bounds,
                normal,
                epsilon,
                &mut residual[0],
                &mut local,
                problem.rows[first],
                stable.then_some(problem.scales[first]),
            )?;
        } else {
            residual[0] -= epsilon / normal;
            local.matrix[0] = epsilon / (normal * normal);
            local.velocity[..problem.size].copy_from_slice(problem.rows[first]);
            local.parameter[0] = -epsilon / normal;
            for (offset, coefficient) in
                std::iter::once((1, point.friction)).chain(point.rolling.map(|length| (3, length)))
            {
                let tangent = [x[first + offset], x[first + offset + 1]];
                let radius = coefficient * normal;
                if stable {
                    let slip = [motion[first + offset], motion[first + offset + 1]];
                    let squared = dot(&slip, &slip);
                    let root = epsilon.hypot(radius * squared.sqrt());
                    let denominator = epsilon + root;
                    let beta = radius * radius / denominator;
                    let derivative_normal = coefficient
                        * (2.0 * radius / denominator
                            - radius.powi(3) * squared / (root * denominator * denominator));
                    let derivative_slip = -radius.powi(4) / (root * denominator * denominator);
                    let scale = problem.scales[first];
                    let divisor = 1.0 + scale * beta;
                    let leading = scale / divisor;
                    for row in 0..2 {
                        let at = offset + row;
                        let multiplier =
                            scale * (slip[row] - scale * tangent[row]) / (divisor * divisor);
                        residual[at] = leading * (tangent[row] + beta * slip[row]);
                        local.matrix[at * size + at] = leading;
                        local.matrix[at * size] = multiplier * derivative_normal;
                        local.parameter[at] = multiplier * (-epsilon / root * beta);
                        for axis in 0..problem.size {
                            local.velocity[at * problem.size + axis] =
                                leading * beta * problem.rows[first + at][axis]
                                    + multiplier
                                        * derivative_slip
                                        * (slip[0] * problem.rows[first + offset][axis]
                                            + slip[1] * problem.rows[first + offset + 1][axis]);
                        }
                    }
                } else {
                    let length = tangent[0].hypot(tangent[1]);
                    let gap = (radius - length) * (radius + length);
                    if gap <= 0.0 {
                        return None;
                    }
                    for row in 0..2 {
                        let at = offset + row;
                        local.parameter[at] = 2.0 * epsilon * tangent[row] / gap;
                        residual[at] += local.parameter[at];
                        local.matrix[at * size] =
                            -4.0 * epsilon * coefficient * coefficient * normal * tangent[row]
                                / (gap * gap);
                        for column in 0..2 {
                            local.matrix[at * size + offset + column] = 2.0 * epsilon / gap
                                * f64::from(row == column)
                                + 4.0 * epsilon * tangent[row] * tangent[column] / (gap * gap);
                        }
                        local.velocity[at * problem.size..(at + 1) * problem.size]
                            .copy_from_slice(problem.rows[first + at]);
                    }
                }
            }
        }
        if residual
            .iter()
            .chain(&local.matrix)
            .chain(&local.velocity)
            .chain(&local.parameter)
            .any(|v| !v.is_finite())
        {
            return None;
        }
        for (rhs, value) in local.rhs.iter_mut().zip(&residual) {
            *rhs = -value;
            result.norm = result.norm.hypot(*value);
            result.maximum = result.maximum.max(value.abs());
        }
        result.residual.extend(residual);
        result.locals.push(local);
    }
    Some(result)
}

// Fixed bounds are exact authored constraints; no barrier interior exists there.
#[allow(clippy::float_cmp)]
fn scalar(
    bounds: crate::ImpulseBounds,
    impulse: f64,
    epsilon: f64,
    residual: &mut f64,
    local: &mut Local,
    row: &[f64],
    stable_scale: Option<f64>,
) -> Option<()> {
    if bounds.minimum == bounds.maximum {
        *residual = impulse - bounds.minimum;
        local.matrix[0] = 1.0;
        return Some(());
    }
    if let Some(scale) = stable_scale
        && bounds.minimum.is_finite()
        && bounds.maximum.is_finite()
    {
        // Invert the interval barrier in slack space. Subtracting an impulse
        // from an almost active finite bound otherwise loses the remaining
        // interior gap before the original velocity residual can converge.
        let half = 0.5 * (bounds.maximum - bounds.minimum);
        let center = 0.5 * bounds.minimum + 0.5 * bounds.maximum;
        let slack = *residual;
        let root = epsilon.hypot(half * slack);
        let denominator = epsilon + root;
        let beta = half * half / denominator;
        let divisor = 1.0 + scale * beta;
        let leading = scale / divisor;
        let multiplier = scale * (slack - scale * (impulse - center)) / (divisor * divisor);
        let derivative = -half.powi(4) * slack / (root * denominator * denominator);
        *residual = leading * (impulse - center + beta * slack);
        local.matrix[0] = leading;
        local.parameter[0] = multiplier * (-epsilon / root * beta);
        for (value, coefficient) in local.velocity.iter_mut().zip(row) {
            *value = (leading * beta + multiplier * derivative) * coefficient;
        }
        return Some(());
    }
    local.velocity.copy_from_slice(row);
    for (sign, gap) in [
        (1.0, impulse - bounds.minimum),
        (-1.0, bounds.maximum - impulse),
    ] {
        if gap <= 0.0 {
            return None;
        }
        if gap.is_finite() {
            *residual -= sign * epsilon / gap;
            local.matrix[0] += epsilon / (gap * gap);
            local.parameter[0] -= sign * epsilon / gap;
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturated_drive_remains_evaluable_when_its_barrier_gap_is_below_one_ulp() {
        let bounds = crate::ImpulseBounds {
            minimum: -0.001,
            maximum: 0.001,
        };
        let mut local = Local {
            first: 0,
            size: 1,
            matrix: vec![0.0],
            velocity: vec![0.0],
            parameter: vec![0.0],
            rhs: vec![0.0],
        };
        let mut residual = -1e-4;
        assert!(
            scalar(
                bounds,
                bounds.maximum,
                1e-30,
                &mut residual,
                &mut local,
                &[1.0],
                None
            )
            .is_none()
        );
        scalar(
            bounds,
            bounds.maximum,
            1e-30,
            &mut residual,
            &mut local,
            &[1.0],
            Some(1.0),
        )
        .unwrap();
        assert!(residual.abs() < 1e-12);
        assert!(
            local
                .matrix
                .iter()
                .chain(&local.velocity)
                .chain(&local.parameter)
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn drive_stop_bilateral_and_fixed_impulse_derivatives_match_perturbations() {
        for (minimum, maximum, impulse) in [
            (-2.0, 3.0, 0.25),
            (f64::NEG_INFINITY, 3.0, 1.0),
            (-2.0, f64::INFINITY, 0.0),
            (f64::NEG_INFINITY, f64::INFINITY, 0.25),
            (0.2, 0.2, 0.2),
        ] {
            let bounds = crate::ImpulseBounds { minimum, maximum };
            let run = |value, epsilon| {
                let mut local = Local {
                    first: 0,
                    size: 1,
                    matrix: vec![0.0],
                    velocity: vec![0.0],
                    parameter: vec![0.0],
                    rhs: vec![0.0],
                };
                let mut residual = 1.5 * value - 0.3;
                scalar(
                    bounds,
                    value,
                    epsilon,
                    &mut residual,
                    &mut local,
                    &[1.5],
                    Some(2.0),
                )
                .unwrap();
                (local, residual)
            };
            for epsilon in [0.1, 1e-6] {
                let (local, _) = run(impulse, epsilon);
                let step = 1e-7_f64;
                let derivative = (run(impulse + step, epsilon).1 - run(impulse - step, epsilon).1)
                    / (2.0 * step);
                assert!((derivative - local.matrix[0] - local.velocity[0]).abs() < 1e-8);
                let parameter = (run(impulse, epsilon * step.exp()).1
                    - run(impulse, epsilon * (-step).exp()).1)
                    / (2.0 * step);
                assert!((parameter - local.parameter[0]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn barrier_and_eliminated_disk_derivatives_match_independent_perturbations() {
        let rows = (0..5)
            .map(|axis| {
                (0..5)
                    .map(|column| f64::from(axis == column))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let problem = Problem {
            size: 5,
            rows: rows.iter().collect(),
            response: rows.clone(),
            targets: vec![1.0, 0.1, 0.2, -0.01, 0.02],
            scales: vec![2.0; 5],
            units: vec![1.0; 5],
            points: vec![super::super::Point {
                first: 0,
                size: 5,
                friction: 0.6,
                rolling: Some(0.02),
                bounds: None,
            }],
        };
        let impulses = [2.0, 0.2, -0.3, 0.01, -0.015];
        let step = 1e-7;
        for stable in [false, true] {
            for epsilon in [0.1, 1e-5, 1e-10] {
                let at = evaluate(&problem, &impulses, epsilon, stable).unwrap();
                let local = &at.locals[0];
                for column in 0..=5 {
                    let mut before = impulses;
                    let mut after = impulses;
                    let mut below = epsilon;
                    let mut above = epsilon;
                    if column < 5 {
                        before[column] -= step;
                        after[column] += step;
                    } else {
                        below *= (-step).exp();
                        above *= step.exp();
                    }
                    let before = evaluate(&problem, &before, below, stable).unwrap();
                    let after = evaluate(&problem, &after, above, stable).unwrap();
                    for row in 0..5 {
                        let measured = (after.residual[row] - before.residual[row]) / (2.0 * step);
                        let derivative = if column < 5 {
                            local.matrix[row * 5 + column] + local.velocity[row * 5 + column]
                        } else {
                            local.parameter[row]
                        };
                        assert!(
                            (measured - derivative).abs() <= 1e-7 * (1.0 + derivative.abs()),
                            "stable={stable} epsilon={epsilon} row={row} column={column} derivative={derivative} measured={measured}"
                        );
                    }
                }
            }
        }
    }
}
