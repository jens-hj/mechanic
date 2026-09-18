//! Coupled impulse response through factor solves, with bounded contact storage.

mod articulated;
mod constraints;
mod factor;
mod newton;
pub(crate) mod normal;
mod projection;

use constraints::BlockLayout;
pub use constraints::{
    ConstraintBlock, ConstraintSolution, ContactFriction, DENSE_CONTACT_ROWS, ImpulseBounds,
};
pub use factor::{DynamicsFactor, DynamicsFactorization};
use projection::{
    block_delta, block_scale, dot, project, project_disk, projected_residual, promote_sliding,
    response_motion, stalled_contact, valid_contacts,
};

use crate::PhysicsError;

/// Solve `J H⁻¹ Jᵀ λ = target` subject to block bounds. Above 128 scalar rows,
/// responses use factor solves and linear scratch storage instead of a dense W.
///
/// # Errors
/// Rejects malformed constraints and inconsistent dependent bilateral rows.
pub fn solve_constraints(
    factor: &DynamicsFactor,
    blocks: &[ConstraintBlock],
    max_iterations: usize,
    tolerance: f64,
) -> Result<ConstraintSolution, PhysicsError> {
    let mut prepared = PreparedConstraints::new(factor, blocks)?;
    let targets = blocks
        .iter()
        .flat_map(|b| b.target.iter().copied())
        .collect::<Vec<_>>();
    let mut solution = prepared.solve(&targets, max_iterations, tolerance)?;
    solution.factor_solves += prepared.preparation_factor_solves();
    Ok(solution)
}

/// Pose-local contact/drive/stop response retained across nonlinear force updates.
/// Holds a borrow of its numerical factor, preventing use after refactorization.
/// Jacobians, bounds, and friction are fixed; targets may change between solves.
pub struct PreparedConstraints<'a> {
    factor: &'a DynamicsFactor,
    blocks: Vec<ConstraintBlock>,
    layout: Vec<BlockLayout>,
    response: Vec<f64>,
    preparation_solves: usize,
}

impl<'a> PreparedConstraints<'a> {
    /// Prepares coupled response once, retaining the 128-row dense storage bound.
    ///
    /// # Errors
    /// Rejects malformed blocks or invalid numerical response.
    pub fn new(
        factor: &'a DynamicsFactor,
        blocks: &[ConstraintBlock],
    ) -> Result<Self, PhysicsError> {
        let mut rows = Vec::new();
        let mut layout = Vec::new();
        for block in blocks {
            let count = block.jacobian.len();
            if count == 0
                || count > DENSE_CONTACT_ROWS
                || block.target.len() != count
                || block.bounds.len() != count
                || block
                    .jacobian
                    .iter()
                    .any(|row| row.len() != factor.size || row.iter().any(|x| !x.is_finite()))
                || block.target.iter().any(|x| !x.is_finite())
                || block.bounds.iter().any(|b| {
                    b.minimum.is_nan()
                        || b.maximum.is_nan()
                        || b.minimum > b.maximum
                        || b.minimum == f64::INFINITY
                        || b.maximum == f64::NEG_INFINITY
                })
                || !valid_contacts(block)
            {
                return Err(PhysicsError::InvalidConstraints);
            }
            layout.push(BlockLayout {
                first: rows.len(),
                diagonal: vec![0.0; count * count],
                scale: 0.0,
            });
            rows.extend(block.jacobian.iter());
        }
        let count = rows.len();
        let mut response = if count <= DENSE_CONTACT_ROWS {
            vec![0.0; count * count]
        } else {
            Vec::new()
        };
        let mut factor_solves = 0;
        let mut scratch = vec![0.0; factor.size];
        for (block, info) in blocks.iter().zip(&mut layout) {
            let size = block.jacobian.len();
            for column in 0..size {
                scratch.copy_from_slice(rows[info.first + column]);
                factor.solve(&mut scratch)?;
                factor_solves += 1;
                for row in 0..size {
                    info.diagonal[row * size + column] = dot(rows[info.first + row], &scratch);
                }
                if !response.is_empty() {
                    for row in 0..count {
                        response[row * count + info.first + column] = dot(rows[row], &scratch);
                    }
                }
            }
            info.scale = block_scale(&info.diagonal, size);
        }

        Ok(Self {
            factor,
            blocks: blocks.to_vec(),
            layout,
            response,
            preparation_solves: factor_solves,
        })
    }

    /// Inverse-dynamics applications performed during response preparation.
    pub fn preparation_factor_solves(&self) -> usize {
        self.preparation_solves
    }

    /// Solves with new flattened target row velocities. Returned factor-solve
    /// counts cover this solve only; preparation is reported separately.
    ///
    /// # Errors
    /// Rejects invalid targets/settings or inconsistent dependent equations.
    pub fn solve(
        &mut self,
        targets: &[f64],
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<ConstraintSolution, PhysicsError> {
        self.solve_from(targets, None, max_iterations, tolerance)
    }

    /// Solves from a supplied impulse estimate, projected into the current bounds.
    /// The caller must map contact identities/frames before reuse across geometry
    /// updates. Reusing a result within this immutable pose-local response needs
    /// no geometry remapping. All residuals are recomputed for the new targets.
    ///
    /// # Errors
    /// Rejects invalid targets/settings, malformed warm starts, or inconsistent rows.
    pub fn solve_from(
        &mut self,
        targets: &[f64],
        initial_impulses: Option<&[f64]>,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<ConstraintSolution, PhysicsError> {
        self.solve_with_contact_trial(targets, initial_impulses, max_iterations, tolerance, true)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "original residual validation remains after the bounded active-set trial"
    )]
    fn solve_with_contact_trial(
        &mut self,
        targets: &[f64],
        initial_impulses: Option<&[f64]>,
        max_iterations: usize,
        tolerance: f64,
        allow_contact_trial: bool,
    ) -> Result<ConstraintSolution, PhysicsError> {
        let count = self.blocks.iter().map(|b| b.target.len()).sum::<usize>();
        if targets.len() != count
            || targets.iter().any(|v| !v.is_finite())
            || initial_impulses.is_some_and(|values| {
                values.len() != count || values.iter().any(|value| !value.is_finite())
            })
            || max_iterations == 0
            || !tolerance.is_finite()
            || tolerance <= 0.0
        {
            return Err(PhysicsError::InvalidConstraints);
        }
        if count == 0 {
            return Ok(ConstraintSolution {
                newton_attempts: 0,
                newton_accepts: 0,
                newton_applications: 0,
                newton_local_factorizations: 0,
                newton_generalized_factorizations: 0,
                newton_reduced_storage: 0,
                newton_contact_factorizations: 0,
                newton_contact_storage: 0,
                newton_line_searches: 0,
                active_set_trials: 0,
                active_set_trial_rejections: 0,
                active_set_trial_matrix_storage: 0,
                warm_start_restarts: 0,
                continuation_trials: 0,
                continuation_iterations: 0,
                continuation_stages: 0,
                continuation_evaluations: 0,
                continuation_mixed_rows: 0,
                continuation_full_rows: 0,
                sliding: Vec::new(),
                friction_transitions: 0,
                velocity_change: vec![0.0; self.factor.size],
                impulses: Vec::new(),
                residual: 0.0,
                iterations: 0,
                converged: true,
                response_storage: 0,
                factor_solves: 0,
            });
        }
        let mut cursor = 0;
        for block in &mut self.blocks {
            let length = block.target.len();
            block
                .target
                .copy_from_slice(&targets[cursor..cursor + length]);
            cursor += length;
        }
        let factor = self.factor;
        let blocks = self.blocks.as_slice();
        let layout = &self.layout;
        let response = &self.response;
        let mut sliding = blocks
            .iter()
            .map(|block| {
                block
                    .contacts
                    .iter()
                    .map(|point| point.sliding)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut friction_transitions = 0;
        let mut pending_mode_solve = false;
        let rows = blocks
            .iter()
            .flat_map(|b| b.jacobian.iter())
            .collect::<Vec<_>>();
        let mut scratch = vec![0.0; factor.size];
        let mut factor_solves = 0;
        let mut impulses = initial_impulses.map_or_else(|| vec![0.0; count], <[f64]>::to_vec);
        let mut change = vec![0.0; factor.size];
        let mut row_change = vec![0.0; count];
        if initial_impulses.is_some() {
            for ((block, info), modes) in blocks.iter().zip(layout).zip(&sliding) {
                project(
                    block,
                    modes,
                    &mut impulses[info.first..info.first + block.jacobian.len()],
                );
            }
            if response.is_empty() {
                (change, row_change) = response_motion(factor, &rows, &impulses)?;
                factor_solves += 1;
            } else {
                for (out, row) in row_change.iter_mut().zip(response.chunks_exact(count)) {
                    *out = dot(row, &impulses);
                }
            }
        }
        let mut residual;
        let mut iterations = 0;
        let mut newton_attempts = 0;
        let mut newton_shift = 1e-3_f64;
        let mut newton_accepts = 0;
        let mut newton_applications = 0;
        let mut newton_local_factorizations = 0;
        let mut newton_generalized_factorizations = 0;
        let mut newton_reduced_storage = 0;
        let mut newton_contact_factorizations = 0;
        let mut newton_contact_storage = 0;
        let mut newton_line_searches = 0;
        let mut residual_history = std::collections::VecDeque::new();
        let mut active_set_trials = 0;
        let mut restarted = false;
        let mut active_set_trial_rejections = 0;
        let mut active_set_trial_matrix_storage = 0;
        let mut continuation_trials = 0;
        let mut continuation_iterations = 0;
        let mut continuation_stages = 0;
        let mut continuation_evaluations = 0;
        let mut continuation_mixed_rows = 0;
        let mut continuation_full_rows = 0;
        for _ in 0..max_iterations {
            if iterations == max_iterations {
                break;
            }
            pending_mode_solve = false;
            for ((block, info), modes) in blocks.iter().zip(layout).zip(&sliding) {
                let size = block.jacobian.len();
                let rhs = (0..size)
                    .map(|row| {
                        block.target[row]
                            - if response.is_empty() {
                                dot(rows[info.first + row], &change)
                            } else {
                                row_change[info.first + row]
                            }
                    })
                    .collect::<Vec<_>>();
                let delta = block_delta(block, &info.diagonal, info.scale, &rhs, tolerance)?;
                let mut candidate = (0..size)
                    .map(|row| impulses[info.first + row] + delta[row])
                    .collect::<Vec<_>>();
                project(block, modes, &mut candidate);
                scratch.fill(0.0);
                for row in 0..size {
                    let difference = candidate[row] - impulses[info.first + row];
                    impulses[info.first + row] = candidate[row];
                    if response.is_empty() {
                        for (out, &j) in scratch.iter_mut().zip(rows[info.first + row]) {
                            *out += j * difference;
                        }
                    } else {
                        for (index, out) in row_change.iter_mut().enumerate() {
                            *out += response[index * count + info.first + row] * difference;
                        }
                    }
                }
                if response.is_empty() {
                    factor.solve(&mut scratch)?;
                    factor_solves += 1;
                    for (out, &delta) in change.iter_mut().zip(&scratch) {
                        *out += delta;
                    }
                }
            }
            // Re-evaluate every block against the final iterate, not the value seen
            // before later blocks changed it. Raw residuals are nonzero at active bounds.
            if response.is_empty() {
                for (out, row) in row_change.iter_mut().zip(&rows) {
                    *out = dot(row, &change);
                }
            }
            residual =
                projected_residual(blocks, layout, &row_change, &impulses, &sliding, tolerance)?;
            iterations += 1;
            if residual > tolerance {
                // A bounded nonmonotone window lets small contact systems cross
                // friction active-set transitions. It never changes the final
                // unshifted residual required for convergence/publication.
                let acceptance_bound = if count <= DENSE_CONTACT_ROWS {
                    residual_history.push_back(residual);
                    if residual_history.len() > 16 {
                        residual_history.pop_front();
                    }
                    residual_history.iter().copied().fold(residual, f64::max)
                } else {
                    residual
                };
                let mut accepted = false;
                for attempt in 0..if count <= DENSE_CONTACT_ROWS { 2 } else { 1 } {
                    newton_attempts += 1;
                    let proposal = newton::direction(
                        factor,
                        blocks,
                        layout,
                        &impulses,
                        &row_change,
                        &sliding,
                        newton_shift,
                        attempt == 1,
                        |trial| {
                            if response.is_empty() {
                                factor_solves += 1;
                                response_motion(factor, &rows, trial)
                                    .ok()
                                    .map(|(_, rows)| rows)
                            } else {
                                Some(
                                    response
                                        .chunks_exact(count)
                                        .map(|row| dot(row, trial))
                                        .collect(),
                                )
                            }
                        },
                    );
                    factor_solves += proposal.factor_solves;
                    newton_contact_factorizations += proposal.contact_factorizations;
                    newton_contact_storage = newton_contact_storage.max(proposal.contact_storage);
                    newton_generalized_factorizations += proposal.generalized_factorizations;
                    newton_reduced_storage = newton_reduced_storage.max(proposal.reduced_storage);
                    newton_applications += proposal.applications;
                    newton_local_factorizations += proposal.local_factorizations;
                    if let Some(direction) = proposal.direction {
                        let mut fraction = 1.0;
                        for backtrack in 0..if count <= DENSE_CONTACT_ROWS { 32 } else { 8 } {
                            newton_line_searches += 1;
                            let mut candidate = impulses
                                .iter()
                                .zip(&direction)
                                .map(|(old, delta)| old + fraction * delta)
                                .collect::<Vec<_>>();
                            for ((block, info), modes) in blocks.iter().zip(layout).zip(&sliding) {
                                project(
                                    block,
                                    modes,
                                    &mut candidate[info.first..info.first + block.jacobian.len()],
                                );
                            }
                            let evaluated = if response.is_empty() {
                                factor_solves += 1;
                                response_motion(factor, &rows, &candidate).ok()
                            } else {
                                Some((
                                    Vec::new(),
                                    response
                                        .chunks_exact(count)
                                        .map(|row| dot(row, &candidate))
                                        .collect(),
                                ))
                            };
                            if let Some((candidate_change, candidate_rows)) = evaluated
                                && let Ok(proposed) = projected_residual(
                                    blocks,
                                    layout,
                                    &candidate_rows,
                                    &candidate,
                                    &sliding,
                                    tolerance,
                                )
                                && proposed < acceptance_bound * (1.0 - 1e-4 * fraction)
                            {
                                impulses = candidate;
                                row_change = candidate_rows;
                                if response.is_empty() {
                                    change = candidate_change;
                                }
                                residual = proposed;
                                newton_accepts += 1;
                                accepted = true;
                                newton_shift = if backtrack == 0 {
                                    (newton_shift * 0.5).max(1e-10)
                                } else {
                                    (newton_shift * 2.0).min(0.1)
                                };
                                break;
                            }
                            fraction *= 0.5;
                        }
                    }
                    if accepted {
                        break;
                    }
                }
                if !accepted {
                    newton_shift = (newton_shift * 10.0).min(0.1);
                }
            }
            if allow_contact_trial
                && initial_impulses.is_some()
                && !restarted
                && count <= DENSE_CONTACT_ROWS
                && iterations < max_iterations
                && residual > tolerance
                && residual_history.len() == 16
                && residual >= 0.9 * residual_history.front().copied().unwrap_or(f64::INFINITY)
            {
                // A warm iterate can delay reaching a friction/drive boundary.
                // Discard it once after a full stalled residual window. W and H
                // remain valid, and all spent work stays in the original budget.
                // Keep only friction transitions already proved by convergence;
                // restarting an impulse guess cannot itself establish breakaway.
                restarted = true;
                impulses.fill(0.0);
                change.fill(0.0);
                row_change.fill(0.0);
                newton_shift = 1e-3;
                residual_history.clear();
                continue;
            }
            if allow_contact_trial
                && active_set_trials == 0
                && count <= DENSE_CONTACT_ROWS
                && iterations < max_iterations
                && residual > tolerance
                && residual_history.len() == 16
                && residual >= 0.9 * residual_history.front().copied().unwrap_or(f64::INFINITY)
                && let Some((block, row)) = stalled_contact(blocks, layout, &row_change, &impulses)
            {
                // Solve one inactive-contact hypothesis using the existing W/H.
                // At most 32 remaining iterations are spent on this single trial;
                // both solves share the original iteration budget.
                // Only the final residual against ALL ORIGINAL constraints can
                // accept this hypothesis; no collision row is omitted from proof.
                let mut hypothesis = PreparedConstraints {
                    factor,
                    blocks: blocks.to_vec(),
                    layout: layout.clone(),
                    response: response.clone(),
                    preparation_solves: 0,
                };
                for (block, modes) in hypothesis.blocks.iter_mut().zip(&sliding) {
                    for (contact, &mode) in block.contacts.iter_mut().zip(modes) {
                        contact.sliding = mode;
                    }
                }
                hypothesis.blocks[block].bounds[row].maximum = 0.0;
                let candidate = hypothesis.solve_with_contact_trial(
                    targets,
                    Some(&impulses),
                    (max_iterations - iterations).min(32),
                    tolerance,
                    false,
                )?;
                active_set_trials += 1;
                active_set_trial_matrix_storage = response.len()
                    + layout.iter().map(|b| b.diagonal.len()).sum::<usize>()
                    + rows.iter().map(|r| r.len()).sum::<usize>();
                iterations += candidate.iterations;
                newton_attempts += candidate.newton_attempts;
                newton_accepts += candidate.newton_accepts;
                newton_applications += candidate.newton_applications;
                newton_local_factorizations += candidate.newton_local_factorizations;
                newton_generalized_factorizations += candidate.newton_generalized_factorizations;
                newton_contact_factorizations += candidate.newton_contact_factorizations;
                newton_contact_storage =
                    newton_contact_storage.max(candidate.newton_contact_storage);
                newton_reduced_storage =
                    newton_reduced_storage.max(candidate.newton_reduced_storage);
                newton_line_searches += candidate.newton_line_searches;
                factor_solves += candidate.factor_solves;
                friction_transitions += candidate.friction_transitions;
                let mut candidate_modes = sliding.clone();
                let mut modes = candidate.sliding.into_iter();
                for block in &mut candidate_modes {
                    for mode in block {
                        *mode = modes.next().expect("same contact layout");
                    }
                }
                let candidate_rows = rows
                    .iter()
                    .map(|row| dot(row, &candidate.velocity_change))
                    .collect::<Vec<_>>();
                let original_residual = projected_residual(
                    blocks,
                    layout,
                    &candidate_rows,
                    &candidate.impulses,
                    &candidate_modes,
                    tolerance,
                )?;
                if candidate.converged && original_residual <= tolerance {
                    pending_mode_solve = false;
                    sliding = candidate_modes;
                    impulses = candidate.impulses;
                    break;
                }
                // A wrong active-set hypothesis is discarded. Resume the original
                // iterate and modes using the remaining shared iteration budget.
                active_set_trial_rejections += 1;
            }
            // Small contact systems first retain the ordinary dense solve and
            // its inactive-row experiment. Once a complete residual window has
            // stalled, the same bounded continuation used above the dense limit
            // may propose a guess; it still consumes the original iteration cap.
            if (count > DENSE_CONTACT_ROWS
                || (allow_contact_trial
                    && residual_history.len() == 16
                    && residual
                        >= 0.9 * residual_history.front().copied().unwrap_or(f64::INFINITY)))
                && factor.size <= 64
                && continuation_trials == 0
                && iterations >= 16
                && iterations < max_iterations
                && residual > tolerance
                && blocks.iter().any(|block| !block.contacts.is_empty())
            {
                let trial = newton::continuation::propose(
                    factor,
                    blocks,
                    layout,
                    &sliding,
                    &impulses,
                    max_iterations - iterations,
                    tolerance,
                );
                continuation_trials += 1;
                continuation_iterations += trial.iterations;
                continuation_stages += trial.stages;
                continuation_evaluations += trial.evaluations;
                continuation_mixed_rows = continuation_mixed_rows.max(trial.work.mixed_rows);
                continuation_full_rows = continuation_full_rows.max(trial.work.full_rows);
                iterations += trial.iterations;
                factor_solves += trial.work.factor_solves;
                newton_applications += trial.work.applications;
                newton_local_factorizations += trial.work.local_factorizations;
                newton_generalized_factorizations += trial.work.generalized_factorizations;
                newton_reduced_storage = newton_reduced_storage.max(trial.work.reduced_storage);
                if let Some(mut candidate) = trial.impulses {
                    for ((block, info), modes) in blocks.iter().zip(layout).zip(&sliding) {
                        project(
                            block,
                            modes,
                            &mut candidate[info.first..info.first + block.target.len()],
                        );
                    }
                    factor_solves += 1;
                    let (candidate_change, candidate_rows) =
                        response_motion(factor, &rows, &candidate)?;
                    let original = projected_residual(
                        blocks,
                        layout,
                        &candidate_rows,
                        &candidate,
                        &sliding,
                        tolerance,
                    )?;
                    if original < residual {
                        impulses = candidate;
                        change = candidate_change;
                        row_change = candidate_rows;
                        residual = original;
                        residual_history.clear();
                        newton_shift = 1e-6;
                    }
                }
            }
            if residual <= tolerance {
                // First settle normal loads and the static cones. Switching on
                // an unfinished iterate can falsely declare breakaway merely
                // because support impulses have not yet accumulated.
                let changed = promote_sliding(
                    blocks,
                    layout,
                    &row_change,
                    &impulses,
                    &mut sliding,
                    tolerance,
                );
                friction_transitions += changed;
                pending_mode_solve = changed > 0;
                if changed > 0 {
                    residual_history.clear();
                }
                if changed == 0 {
                    break;
                }
            }
        }
        // Reconstruct body motion only after the repeated contact solve. The dense
        // route never propagates a contact impulse through the machine inside a sweep.
        for ((block, info), modes) in blocks.iter().zip(layout).zip(&sliding) {
            project(
                block,
                modes,
                &mut impulses[info.first..info.first + block.jacobian.len()],
            );
        }
        change.fill(0.0);
        for (row, &impulse) in rows.iter().zip(&impulses) {
            for (out, &j) in change.iter_mut().zip(*row) {
                *out += j * impulse;
            }
        }
        factor.solve(&mut change)?;
        factor_solves += 1;
        for (out, row) in row_change.iter_mut().zip(&rows) {
            *out = dot(row, &change);
        }
        residual = projected_residual(blocks, layout, &row_change, &impulses, &sliding, tolerance)?;
        Ok(ConstraintSolution {
            newton_attempts,
            newton_accepts,
            newton_applications,
            newton_local_factorizations,
            newton_generalized_factorizations,
            newton_reduced_storage,
            newton_contact_factorizations,
            newton_contact_storage,
            newton_line_searches,
            active_set_trials,
            active_set_trial_rejections,
            active_set_trial_matrix_storage,
            warm_start_restarts: usize::from(restarted),
            continuation_trials,
            continuation_iterations,
            continuation_stages,
            continuation_evaluations,
            continuation_mixed_rows,
            continuation_full_rows,
            sliding: sliding.into_iter().flatten().collect(),
            friction_transitions,
            velocity_change: change,
            impulses,
            residual,
            iterations,
            converged: residual <= tolerance && !pending_mode_solve,
            response_storage: response.len(),
            factor_solves,
        })
    }
}

#[cfg(test)]
mod tests;
