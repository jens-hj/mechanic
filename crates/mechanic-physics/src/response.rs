//! Coupled impulse response through factor solves, with bounded contact storage.

use crate::PhysicsError;
mod articulated;
mod newton;
pub(crate) mod normal;

/// Largest scalar constraint count for explicit contact-response storage.
pub const DENSE_CONTACT_ROWS: usize = 128;

/// Explicit numerical factor selection for matched CPU experiments.
/// Both choices solve the same effective dynamics; arithmetic ordering differs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DynamicsFactorization {
    /// Frozen dense reference. Kept authoritative until complete tick gates pass.
    #[default]
    DenseReference,
    /// Articulated tree factor with component-local impulse responses.
    Articulated,
}

impl DynamicsFactorization {
    pub(crate) fn factor(
        self,
        creation: &mechanic_core::CompiledCreation,
        model: &crate::MachineDynamics,
        coordinates: &[f64],
        diagonal: &[f64],
    ) -> Result<DynamicsFactor, PhysicsError> {
        match self {
            Self::DenseReference => model.factor(diagonal),
            Self::Articulated => {
                DynamicsFactor::articulated(creation, &model.poses, coordinates, diagonal)
            }
        }
    }
}

/// Positive-definite effective dynamics factored once per substep.
#[derive(Clone, Debug)]
pub struct DynamicsFactor {
    size: usize,
    storage: FactorStorage,
}

#[derive(Clone, Debug)]
enum FactorStorage {
    Dense(Vec<f64>),
    Articulated(articulated::ArticulatedFactor),
}

impl DynamicsFactor {
    /// Factors a symmetric row-major effective mass matrix. No inverse is formed.
    ///
    /// # Errors
    /// Rejects malformed, non-finite, asymmetric, or non-positive dynamics.
    pub fn new(matrix: &[f64], size: usize) -> Result<Self, PhysicsError> {
        if size.checked_mul(size) != Some(matrix.len()) || matrix.iter().any(|x| !x.is_finite()) {
            return Err(PhysicsError::InvalidDynamics);
        }
        let mut lower = vec![0.0; matrix.len()];
        for row in 0..size {
            for column in 0..=row {
                let a = matrix[row * size + column];
                let b = matrix[column * size + row];
                if (a - b).abs() > 1e-12 * a.abs().max(b.abs()).max(1.0) {
                    return Err(PhysicsError::InvalidDynamics);
                }
                let value = a
                    - (0..column)
                        .map(|k| lower[row * size + k] * lower[column * size + k])
                        .sum::<f64>();
                lower[row * size + column] = if row == column {
                    if value <= 0.0 || !value.is_finite() {
                        return Err(PhysicsError::InvalidDynamics);
                    }
                    value.sqrt()
                } else {
                    value / lower[column * size + column]
                };
            }
        }
        Ok(Self {
            size,
            storage: FactorStorage::Dense(lower),
        })
    }

    /// Factors tree dynamics directly, with linear body storage and work. Root
    /// rows use world linear/angular velocities; scalar joints use compiled order.
    /// Numerical factors belong to this exact pose and implicit diagonal. Loop
    /// equations remain separate constraint rows; this does not enforce them.
    ///
    /// # Errors
    /// Rejects invalid poses, diagonal rows, and non-positive effective inertia.
    pub fn articulated(
        creation: &mechanic_core::CompiledCreation,
        roots: &[crate::BodyPose],
        coordinates: &[f64],
        implicit_diagonal: &[f64],
    ) -> Result<Self, PhysicsError> {
        Ok(Self {
            size: creation.dynamics.elimination_parent.len(),
            storage: FactorStorage::Articulated(articulated::ArticulatedFactor::new(
                creation,
                roots,
                coordinates,
                implicit_diagonal,
            )?),
        })
    }

    pub(crate) fn articulated_from_poses(
        creation: &mechanic_core::CompiledCreation,
        poses: &[crate::BodyPose],
        diagonal: &[f64],
    ) -> Result<Self, PhysicsError> {
        Ok(Self {
            size: creation.dynamics.elimination_parent.len(),
            storage: FactorStorage::Articulated(articulated::ArticulatedFactor::from_poses(
                creation, poses, diagonal,
            )?),
        })
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        match &self.storage {
            FactorStorage::Dense(values) => values.capacity() * size_of::<f64>(),
            FactorStorage::Articulated(factor) => factor.retained_bytes(),
        }
    }

    pub(crate) fn refit_articulated(
        &mut self,
        creation: &mechanic_core::CompiledCreation,
        poses: &[crate::BodyPose],
        diagonal: &[f64],
    ) -> Result<(), PhysicsError> {
        match &mut self.storage {
            FactorStorage::Articulated(factor) => {
                factor.refit(creation, poses, diagonal)?;
                self.size = creation.dynamics.elimination_parent.len();
            }
            FactorStorage::Dense(_) => {
                *self = Self::articulated_from_poses(creation, poses, diagonal)?;
            }
        }
        Ok(())
    }

    pub(crate) fn solve_ranges(
        &self,
        values: &mut [f64],
        ranges: &[std::ops::Range<usize>],
    ) -> Result<(), PhysicsError> {
        if values.len() != self.size {
            return Err(PhysicsError::InvalidDynamics);
        }
        match &self.storage {
            FactorStorage::Articulated(factor) => factor.solve_ranges(values, ranges),
            FactorStorage::Dense(_) => {
                for (row, value) in values.iter_mut().enumerate() {
                    if !ranges.iter().any(|range| range.contains(&row)) {
                        *value = 0.0;
                    }
                }
                self.solve(values)
            }
        }
    }

    /// Applies inverse dynamics in-place to a generalized impulse.
    ///
    /// # Errors
    /// Rejects incorrect row counts or non-finite inputs/results.
    pub fn solve(&self, values: &mut [f64]) -> Result<(), PhysicsError> {
        if values.len() != self.size || values.iter().any(|x| !x.is_finite()) {
            return Err(PhysicsError::InvalidDynamics);
        }
        let lower = match &self.storage {
            FactorStorage::Dense(lower) => lower,
            FactorStorage::Articulated(factor) => return factor.solve(values),
        };
        for row in 0..self.size {
            let previous = (0..row)
                .map(|k| lower[row * self.size + k] * values[k])
                .sum::<f64>();
            values[row] = (values[row] - previous) / lower[row * self.size + row];
        }
        for row in (0..self.size).rev() {
            let next = (row + 1..self.size)
                .map(|k| lower[k * self.size + row] * values[k])
                .sum::<f64>();
            values[row] = (values[row] - next) / lower[row * self.size + row];
        }
        if values.iter().any(|x| !x.is_finite()) {
            return Err(PhysicsError::InvalidDynamics);
        }
        Ok(())
    }
}

/// Bounds on one scalar row. Bilateral rows use infinite limits.
#[derive(Clone, Copy, Debug)]
pub struct ImpulseBounds {
    /// Minimum allowed impulse.
    pub minimum: f64,
    /// Maximum allowed impulse.
    pub maximum: f64,
}

/// Coulomb/rolling law for one point in a coupled contact manifold.
#[derive(Clone, Copy, Debug)]
pub struct ContactFriction {
    /// Coefficient available when a contact can remain at rest.
    pub static_coefficient: f64,
    /// Coefficient while sliding, no greater than the static coefficient.
    pub kinetic_coefficient: f64,
    /// Whether the contact is already sliding at the beginning of this solve.
    pub sliding: bool,
    /// Optional rolling-resistance coefficient times effective radius, in metres.
    /// Adds two angular rows after the normal and two tangential rows.
    pub rolling_length: Option<f64>,
}

impl ContactFriction {
    fn rows(self) -> usize {
        if self.rolling_length.is_some() { 5 } else { 3 }
    }
}

/// One coupled contact manifold, drive, stop, or loop block.
#[derive(Clone, Debug)]
pub struct ConstraintBlock {
    /// Generalized Jacobian rows. All rows must match the dynamics dimension.
    pub jacobian: Vec<Vec<f64>>,
    /// Required change in row velocity before constraint impulses.
    pub target: Vec<f64>,
    /// Per-row impulse bounds.
    pub bounds: Vec<ImpulseBounds>,
    /// Ordered contact points, each with normal/tangent/tangent and optionally
    /// two rolling rows. Empty for drives, stops, and bilateral loop blocks.
    /// Normal lower bounds are zero; tangent/rolling bounds are infinite.
    pub contacts: Vec<ContactFriction>,
}

/// Result of a deterministic block projected solve.
#[derive(Clone, Debug)]
pub struct ConstraintSolution {
    /// Safeguarded Newton proposals, separate from block sweeps.
    pub newton_attempts: usize,
    /// Newton proposals accepted within the bounded physical-residual history.
    pub newton_accepts: usize,
    /// Newton/search operator applications, including cached dynamics responses.
    pub newton_applications: usize,
    /// Attempted 1/3/5-row factors (analytic projection blocks or pivoted LU).
    pub newton_local_factorizations: usize,
    /// Bounded reduced or mixed Newton matrix factorizations.
    pub newton_generalized_factorizations: usize,
    /// Peak scalar matrix storage for reduced searches, including rectangular
    /// factors when an implicit mixed system exceeds 128 equations.
    pub newton_reduced_storage: usize,
    /// Bounded dense contact-space Newton factorizations.
    pub newton_contact_factorizations: usize,
    /// Peak scalar storage of the dense contact Newton matrix.
    pub newton_contact_storage: usize,
    /// Candidate residual evaluations during bounded Newton backtracking.
    pub newton_line_searches: usize,
    /// Bounded inactive-contact hypotheses, validated against every original row.
    pub active_set_trials: usize,
    /// Hypotheses discarded because original constraints or subset solving failed.
    pub active_set_trial_rejections: usize,
    /// Extra scalar matrix storage for a hypothesis: W, local blocks and Jacobian.
    /// Excludes vectors/metadata; numerical H is borrowed, never copied.
    pub active_set_trial_matrix_storage: usize,
    /// Stalled supplied warm iterates discarded within the original sweep budget.
    pub warm_start_restarts: usize,
    /// Bounded smoothing searches; candidates must pass the original contact laws.
    pub continuation_trials: usize,
    /// Search Newton directions charged to the same total iteration budget.
    pub continuation_iterations: usize,
    /// Smoothing stages evaluated during candidate search.
    pub continuation_stages: usize,
    /// Smoothed residual evaluations, including bounded backtracking.
    pub continuation_evaluations: usize,
    /// Peak mixed search dimension, bounded to 128 including retained contact rows.
    pub continuation_mixed_rows: usize,
    /// Peak full-space search dimension. Small machines deliberately store this
    /// search matrix beyond the 128-row response bound, up to 256 contact rows.
    pub continuation_full_rows: usize,
    /// Final sliding state in block/contact-point order.
    pub sliding: Vec<bool>,
    /// Static contacts whose required impulse exceeded the static cone.
    pub friction_transitions: usize,
    /// Generalized velocity change; apply only after checking convergence.
    pub velocity_change: Vec<f64>,
    /// Scalar impulses in block/row order.
    pub impulses: Vec<f64>,
    /// Maximum projected velocity residual.
    pub residual: f64,
    /// Total completed block sweeps and continuation Newton directions.
    pub iterations: usize,
    /// Whether the residual reached the requested threshold.
    pub converged: bool,
    /// Scalar slots allocated for the explicit response matrix (zero above 128 rows).
    pub response_storage: usize,
    /// Inverse-dynamics applications, including response preparation and final motion.
    pub factor_solves: usize,
}

#[derive(Clone)]
struct BlockLayout {
    first: usize,
    diagonal: Vec<f64>,
    scale: f64,
}

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
    #[allow(clippy::too_many_lines)] // Keep sweep updates and final residual verification together.
    pub fn solve_from(
        &mut self,
        targets: &[f64],
        initial_impulses: Option<&[f64]>,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<ConstraintSolution, PhysicsError> {
        self.solve_with_contact_trial(targets, initial_impulses, max_iterations, tolerance, true)
    }

    #[allow(clippy::too_many_lines)] // Original residual validation remains after the bounded active-set trial.
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

fn response_motion(
    factor: &DynamicsFactor,
    rows: &[&Vec<f64>],
    impulses: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), PhysicsError> {
    let mut change = vec![0.0; factor.size];
    for (row, &impulse) in rows.iter().zip(impulses) {
        for (out, &j) in change.iter_mut().zip(*row) {
            *out += j * impulse;
        }
    }
    factor.solve(&mut change)?;
    let row_change = rows.iter().map(|row| dot(row, &change)).collect();
    Ok((change, row_change))
}

fn stalled_contact(
    blocks: &[ConstraintBlock],
    layout: &[BlockLayout],
    row_change: &[f64],
    impulses: &[f64],
) -> Option<(usize, usize)> {
    let mut selected: Option<(usize, usize, f64)> = None;
    for (index, (block, info)) in blocks.iter().zip(layout).enumerate() {
        let mut local = 0;
        for contact in &block.contacts {
            let row = info.first + local;
            // Positive slack with load violates normal complementarity. Ties
            // retain the first feature in the deterministic constraint ordering.
            // A positive slack below the convergence tolerance can still cause a
            // larger coupled violation elsewhere. It may propose an inactive row;
            // only the unchanged residual over all original rows may accept it.
            let slack = row_change[row] - block.target[local];
            if impulses[row] > 0.0
                && slack > 0.0
                && selected.is_none_or(|(_, _, previous)| slack > previous)
            {
                selected = Some((index, local, slack));
            }
            local += contact.rows();
        }
    }
    selected.map(|(block, row, _)| (block, row))
}

fn projected_residual(
    blocks: &[ConstraintBlock],
    layout: &[BlockLayout],
    row_change: &[f64],
    impulses: &[f64],
    sliding: &[Vec<bool>],
    tolerance: f64,
) -> Result<f64, PhysicsError> {
    let mut residual = 0.0_f64;
    for ((block, info), modes) in blocks.iter().zip(layout).zip(sliding) {
        let size = block.jacobian.len();
        let rhs = (0..size)
            .map(|row| block.target[row] - row_change[info.first + row])
            .collect::<Vec<_>>();
        let delta = block_delta(block, &info.diagonal, info.scale, &rhs, tolerance)?;
        let mut candidate = (0..size)
            .map(|row| impulses[info.first + row] + delta[row])
            .collect::<Vec<_>>();
        project(block, modes, &mut candidate);
        for row in 0..size {
            let error = if is_bilateral(block) {
                rhs[row]
            } else {
                info.scale * (candidate[row] - impulses[info.first + row])
            };
            if !error.is_finite() {
                return Err(PhysicsError::InvalidDynamics);
            }
            residual = residual.max(error.abs());
        }
    }
    Ok(residual)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn is_bilateral(block: &ConstraintBlock) -> bool {
    block.contacts.is_empty()
        && block
            .bounds
            .iter()
            .all(|bounds| bounds.minimum == f64::NEG_INFINITY && bounds.maximum == f64::INFINITY)
}

fn block_scale(matrix: &[f64], size: usize) -> f64 {
    matrix
        .chunks_exact(size)
        .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0, f64::max)
        .max(1e-30)
}

fn block_delta(
    block: &ConstraintBlock,
    matrix: &[f64],
    scale: f64,
    rhs: &[f64],
    tolerance: f64,
) -> Result<Vec<f64>, PhysicsError> {
    if is_bilateral(block) {
        return rank_solve(matrix, rhs, tolerance);
    }
    // Reuse the pose-local spectral-radius bound. Clipping an unconstrained
    // W^-1 solution is not a bounded block solve: off-diagonal coupling can
    // falsely report convergence. The contact projection is non-associated
    // Coulomb (normal clamp followed by disks), not an associated cone QP.
    Ok(rhs.iter().map(|value| value / scale).collect())
}

fn valid_contacts(block: &ConstraintBlock) -> bool {
    if block.contacts.is_empty() {
        return true;
    }
    let mut row = 0;
    for law in &block.contacts {
        if !law.static_coefficient.is_finite()
            || !law.kinetic_coefficient.is_finite()
            || law.kinetic_coefficient < 0.0
            || law.static_coefficient < law.kinetic_coefficient
            || law
                .rolling_length
                .is_some_and(|length| !length.is_finite() || length < 0.0)
        {
            return false;
        }
        let count = law.rows();
        let Some(bounds) = block.bounds.get(row..row + count) else {
            return false;
        };
        if bounds[0].minimum != 0.0
            || bounds[1..]
                .iter()
                .any(|b| b.minimum != f64::NEG_INFINITY || b.maximum != f64::INFINITY)
        {
            return false;
        }
        row += count;
    }
    row == block.jacobian.len()
}

fn promote_sliding(
    blocks: &[ConstraintBlock],
    layout: &[BlockLayout],
    row_change: &[f64],
    impulses: &[f64],
    modes: &mut [Vec<bool>],
    tolerance: f64,
) -> usize {
    let mut changed = 0;
    for ((block, info), modes) in blocks.iter().zip(layout).zip(modes) {
        let mut row = 0;
        for (law, sliding) in block.contacts.iter().zip(modes) {
            let remaining_u = block.target[row + 1] - row_change[info.first + row + 1];
            let remaining_v = block.target[row + 2] - row_change[info.first + row + 2];
            if !*sliding
                && impulses[info.first + row] > 0.0
                && remaining_u.hypot(remaining_v) > tolerance * 4.0
            {
                *sliding = true;
                changed += 1;
            }
            row += law.rows();
        }
    }
    changed
}

fn project(block: &ConstraintBlock, sliding: &[bool], values: &mut [f64]) {
    for (value, bounds) in values.iter_mut().zip(&block.bounds) {
        *value = value.clamp(bounds.minimum, bounds.maximum);
    }
    let mut row = 0;
    for (law, &sliding) in block.contacts.iter().zip(sliding) {
        let point = &mut values[row..row + law.rows()];
        let coefficient = if sliding {
            law.kinetic_coefficient
        } else {
            law.static_coefficient
        };
        let normal = point[0].max(0.0);
        project_disk(&mut point[1..3], coefficient * normal);
        if let Some(length) = law.rolling_length {
            project_disk(&mut point[3..5], length * normal);
        }
        row += law.rows();
    }
}

fn project_disk(pair: &mut [f64], radius: f64) {
    let length = pair[0].hypot(pair[1]);
    if length > radius {
        pair[0] *= radius / length;
        pair[1] *= radius / length;
    }
}

// Deterministic complete diagonal pivoting for a positive-semidefinite block.
// Dependent rows are retained and checked for consistency, never regularized.
fn rank_solve(matrix: &[f64], rhs: &[f64], tolerance: f64) -> Result<Vec<f64>, PhysicsError> {
    let n = rhs.len();
    if n == 1 {
        if matrix[0] > 0.0 {
            return Ok(vec![rhs[0] / matrix[0]]);
        }
        if rhs[0].abs() <= tolerance {
            return Ok(vec![0.0]);
        }
        return Err(PhysicsError::InconsistentConstraints);
    }
    let mut a = matrix.to_vec();
    let mut b = rhs.to_vec();
    let mut permutation = (0..n).collect::<Vec<_>>();
    let scale = (0..n).map(|i| a[i * n + i].abs()).fold(0.0, f64::max);
    let mut rank = n;
    for k in 0..n {
        let mut pivot = k;
        for i in k + 1..n {
            if a[i * n + i] > a[pivot * n + pivot] {
                pivot = i;
            }
        }
        if a[pivot * n + pivot] <= scale * 1e-12 {
            rank = k;
            if b[k..].iter().any(|x| x.abs() > tolerance) {
                return Err(PhysicsError::InconsistentConstraints);
            }
            break;
        }
        for j in 0..n {
            a.swap(k * n + j, pivot * n + j);
        }
        for i in 0..n {
            a.swap(i * n + k, i * n + pivot);
        }
        b.swap(k, pivot);
        permutation.swap(k, pivot);
        for i in k + 1..n {
            let multiplier = a[i * n + k] / a[k * n + k];
            for j in k + 1..n {
                a[i * n + j] -= multiplier * a[k * n + j];
            }
            b[i] -= multiplier * b[k];
            a[i * n + k] = 0.0;
        }
    }
    let mut solution = vec![0.0; n];
    for i in (0..rank).rev() {
        solution[i] = (b[i]
            - (i + 1..rank)
                .map(|j| a[i * n + j] * solution[j])
                .sum::<f64>())
            / a[i * n + i];
    }
    let mut output = vec![0.0; n];
    for i in 0..n {
        output[permutation[i]] = solution[i];
    }
    Ok(output)
}

#[cfg(test)]
mod tests;
