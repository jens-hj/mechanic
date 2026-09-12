//! Bounded smoothing continuation proposes an impulse guess. It never substitutes
//! smoothed equations for the original contact residual or friction-mode proof.

use super::super::{BlockLayout, ConstraintBlock, DynamicsFactor, dot, response_motion};
use super::NewtonWork;
mod central;
mod full;
mod manifold;
mod mixed;
mod rank;
mod smoothing;
use smoothing::Evaluation;

#[derive(Default)]
pub(in crate::response) struct Continuation {
    pub impulses: Option<Vec<f64>>,
    pub residual: f64,
    pub iterations: usize,
    pub stages: usize,
    pub work: NewtonWork,
    pub evaluations: usize,
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(in crate::response) fn propose(
    factor: &DynamicsFactor,
    blocks: &[ConstraintBlock],
    layout: &[BlockLayout],
    modes: &[Vec<bool>],
    initial: &[f64],
    maximum_iterations: usize,
    tolerance: f64,
) -> Continuation {
    // Small machines search bounded scalar rows (stops and drives) in full
    // space with a smoothed clamp. Bilateral and fixed rows keep the bordered path.
    let row_count = blocks.iter().map(|block| block.target.len()).sum::<usize>();
    if blocks.iter().any(|block| {
        block.contacts.is_empty()
            && (row_count > full::MAXIMUM_FULL_ROWS
                || block.bounds.iter().any(|bounds| {
                    bounds.minimum >= bounds.maximum
                        || !(bounds.minimum.is_finite() || bounds.maximum.is_finite())
                }))
    }) {
        return central::propose(factor, blocks, layout, modes, maximum_iterations, tolerance);
    }
    let mut result = Continuation::default();
    if factor.size == 0
        || factor.size > 64
        || blocks.iter().any(|b| {
            b.contacts
                .iter()
                .scan(0, |row, c| {
                    let bounded = b.bounds[*row].maximum.is_finite();
                    *row += c.rows();
                    Some(bounded)
                })
                .any(|bounded| bounded)
        })
    {
        return result;
    }
    // Duplicated manifold rolling rows share one disk during the search only;
    // candidates are expanded and validated against the original rows.
    let manifolds = manifold::Manifolds::new(blocks, layout);
    let original_rows = blocks.iter().flat_map(|b| &b.jacobian).collect::<Vec<_>>();
    let rows = manifolds
        .blocks
        .iter()
        .flat_map(|b| &b.jacobian)
        .collect::<Vec<_>>();
    let scales = manifolds
        .blocks
        .iter()
        .zip(&manifolds.layout)
        .flat_map(|(b, info)| std::iter::repeat_n(info.scale, b.target.len()))
        .collect::<Vec<_>>();
    let mut impulses = manifolds.reduce(initial);
    let warm = initial.iter().any(|value| value.abs() > 0.0);
    let Some((_, mut motion)) = response_motion(factor, &rows, &impulses).ok() else {
        return result;
    };
    result.work.factor_solves += 1;
    let mut epsilon = blocks
        .iter()
        .flat_map(|b| &b.target)
        .map(|v| v.abs())
        .fold(tolerance, f64::max)
        * 0.01;
    epsilon = epsilon.max(tolerance * 0.1);
    // Small machines search over every contact row. Rank-revealing truncation
    // there follows the smoothed branch through support transitions, so an
    // unsettled stage may hand its iterate to the next parameter.
    let coupling = full::Coupling::new(factor, &rows, &scales, &mut result.work);
    let mut unsettled_streak = 0;
    let mut first_stage = true;
    for _ in 0..32 {
        result.stages += 1;
        let mut stage_converged = false;
        let mut direction_failed = false;
        // Each stage receives a bounded share of the shared direction budget.
        // Successful stages keep their original checks.
        for _ in 0..16 {
            result.evaluations += 1;
            let Some(at) = smoothing::evaluate(
                &manifolds.blocks,
                &manifolds.layout,
                modes,
                &manifolds.rolling,
                &impulses,
                &motion,
                epsilon,
            ) else {
                return result;
            };
            if at.maximum <= (epsilon * 0.01).max(tolerance * 0.01) {
                stage_converged = true;
                break;
            }
            if result.iterations == maximum_iterations {
                return result;
            }
            result.iterations += 1;
            let direction = match &coupling {
                Some(coupling) => full::direction(coupling, &at, &mut result.work),
                None => mixed::direction(factor, &rows, &scales, &at, &mut result.work),
            };
            let Some(direction) = direction else {
                direction_failed = true;
                break;
            };
            let mut fraction = 1.0;
            let mut accepted = false;
            #[cfg(test)]
            let starting_norm = at.norm;
            for _ in 0..32 {
                result.evaluations += 1;
                let candidate = impulses
                    .iter()
                    .zip(&direction)
                    .map(|(a, b)| a + fraction * b)
                    .collect::<Vec<_>>();
                result.work.factor_solves += 1;
                if let Ok((_, change)) = response_motion(factor, &rows, &candidate)
                    && let Some(trial) = smoothing::evaluate(
                        &manifolds.blocks,
                        &manifolds.layout,
                        modes,
                        &manifolds.rolling,
                        &candidate,
                        &change,
                        epsilon,
                    )
                    && trial.norm < at.norm * (1.0 - 1e-4 * fraction)
                {
                    impulses = candidate;
                    motion = change;
                    accepted = true;
                    break;
                }
                fraction *= 0.5;
            }
            #[cfg(test)]
            if std::env::var_os("MECHANIC_TRACE_CONTINUATION").is_some() {
                eprintln!(
                    "smooth step epsilon={epsilon:e} norm={starting_norm:e} maximum={:e} accepted={accepted} fraction={fraction:e} direction={:e}",
                    at.maximum,
                    direction
                        .iter()
                        .fold(0.0_f64, |largest, v| largest.max(v.abs()))
                );
            }
            if !accepted {
                break;
            }
        }
        // End continuation as soon as the original nonsmooth laws pass. This
        // avoids spending further search stages on an already valid candidate.
        let mut projected = impulses.clone();
        manifolds.project(&mut projected, modes);
        let candidate = manifolds.expand(&projected);
        result.work.factor_solves += 1;
        if let Ok((_, rows_changed)) = response_motion(factor, &original_rows, &candidate)
            && let Ok(residual) = super::super::projected_residual(
                blocks,
                layout,
                &rows_changed,
                &candidate,
                modes,
                tolerance,
            )
        {
            if result.impulses.is_none() || residual < result.residual {
                result.impulses = Some(candidate);
                result.residual = residual;
            }
            if residual <= tolerance {
                return result;
            }
        }
        #[cfg(test)]
        if std::env::var_os("MECHANIC_TRACE_CONTINUATION").is_some() {
            eprintln!(
                "smooth epsilon={epsilon:e} iterations={} converged={stage_converged} direction_failed={direction_failed} best={:e}",
                result.iterations, result.residual
            );
        }
        if std::mem::take(&mut first_stage)
            && warm
            && !stage_converged
            && !direction_failed
            && coupling.is_some()
        {
            // A stalled sweep iterate can sit where smoothed steps barely
            // progress. Restart the schedule once from zero impulses; the
            // restart shares the caller's fixed direction budget.
            impulses.fill(0.0);
            let Ok((_, change)) = response_motion(factor, &rows, &impulses) else {
                return result;
            };
            result.work.factor_solves += 1;
            motion = change;
            continue;
        }
        unsettled_streak = if stage_converged {
            0
        } else {
            unsettled_streak + 1
        };
        // One unsettled full-space stage may hand its iterate to the next
        // parameter. A failed linearization, a reduced search that cannot
        // truncate dependent support rows, or a second consecutive unsettled
        // stage does not follow a solution branch. Follow folds with a bordered
        // path instead, spending only the directions left in the budget.
        if !stage_converged && (direction_failed || coupling.is_none() || unsettled_streak >= 2) {
            let trial = central::propose(
                factor,
                blocks,
                layout,
                modes,
                maximum_iterations - result.iterations,
                tolerance,
            );
            result.iterations += trial.iterations;
            result.stages += trial.stages;
            result.evaluations += trial.evaluations;
            result.work.applications += trial.work.applications;
            result.work.factor_solves += trial.work.factor_solves;
            result.work.local_factorizations += trial.work.local_factorizations;
            result.work.generalized_factorizations += trial.work.generalized_factorizations;
            result.work.reduced_storage =
                result.work.reduced_storage.max(trial.work.reduced_storage);
            result.work.mixed_rows = result.work.mixed_rows.max(trial.work.mixed_rows);
            if trial.impulses.is_some()
                && (result.impulses.is_none() || trial.residual < result.residual)
            {
                result.impulses = trial.impulses;
                result.residual = trial.residual;
            }
            return result;
        }
        if epsilon <= tolerance * 0.1 {
            return result;
        }
        epsilon = (epsilon * 0.1).max(tolerance * 0.1);
    }
    result
}
