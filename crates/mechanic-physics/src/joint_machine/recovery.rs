//! Split finite-terrain position recovery. Scratch displacements never become velocity.

use super::{
    CompiledCreation, ConstraintBlock, CoordinateDrive, ImpulseBounds, JointTickDiagnostics,
    JointTickSettings, MachineDynamics, MachineState, PhysicsError, TerrainSubstep,
    advance_positions, bounds, validate_positions,
};
use crate::terrain_contacts::CONTACT_ACTIVATION_DISTANCE;

#[allow(clippy::too_many_arguments)] // Preserve the operation boundary in failure diagnostics.
pub(super) fn correct(
    creation: &CompiledCreation,
    drives: &[CoordinateDrive],
    state: &mut MachineState,
    settings: JointTickSettings,
    terrain: Option<&TerrainSubstep<'_>>,
    diagnostics: &mut JointTickDiagnostics,
) -> Result<(), PhysicsError> {
    correct_candidate(creation, drives, state, settings, terrain, diagnostics).inspect_err(|_| {
        diagnostics.failure_stage = Some(super::JointFailureStage::TerrainRecovery);
    })
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Explicit immutable scene and unpublished state.
fn correct_candidate(
    creation: &CompiledCreation,
    drives: &[CoordinateDrive],
    state: &mut MachineState,
    settings: JointTickSettings,
    terrain: Option<&TerrainSubstep<'_>>,
    diagnostics: &mut JointTickDiagnostics,
) -> Result<(), PhysicsError> {
    let Some(terrain) = terrain else {
        return Ok(());
    };
    for _ in 0..8 {
        diagnostics.terrain_recovery_queries += 1;
        let query =
            terrain
                .scene
                .recovery_contacts(terrain.geometry, &state.poses, terrain.origin)?;
        diagnostics.terrain_chunk_candidates += query.chunk_candidates;
        diagnostics.terrain_triangle_candidates += query.triangle_candidates;
        let depth = query
            .contacts
            .iter()
            .map(|point| point.depth)
            .fold(0.0, f64::max);
        if depth <= CONTACT_ACTIVATION_DISTANCE {
            return Ok(());
        }
        if depth > terrain.maximum_depth {
            return Err(PhysicsError::NotConverged);
        }
        let model = MachineDynamics::assemble(creation, &state.poses, &state.coordinates)?;
        diagnostics.dynamics_assemblies += 1;
        let size = state.velocities.len();
        let factor = settings.factorization.factor(
            creation,
            &model,
            &state.coordinates,
            &vec![0.0; size],
        )?;
        diagnostics.factorizations += 1;
        let mut blocks = Vec::with_capacity(query.contacts.len());
        let mut add = |jacobian, gap| {
            blocks.push(ConstraintBlock {
                jacobian: vec![jacobian],
                target: vec![gap],
                bounds: vec![ImpulseBounds {
                    minimum: 0.0,
                    maximum: f64::INFINITY,
                }],
                contacts: Vec::new(),
            });
        };
        for point in &query.contacts {
            add(
                model.point_row(point.body, point.body_point, point.normal)?,
                -point.separation,
            );
        }
        for (coordinate, &row) in creation.dynamics.coordinate_velocities.iter().enumerate() {
            let [lower, upper] = bounds(creation, drives, coordinate);
            for (sign, gap) in [
                (1.0, lower - state.coordinates[coordinate]),
                (-1.0, state.coordinates[coordinate] - upper),
            ] {
                if gap.is_finite() {
                    let mut jacobian = vec![0.0; size];
                    jacobian[row] = sign;
                    add(jacobian, gap);
                }
            }
        }
        diagnostics.terrain_recovery_rows += blocks.len();
        let seed = crate::response::normal::seed(
            &factor,
            &blocks,
            settings.constraint_iterations,
            CONTACT_ACTIVATION_DISTANCE * 0.1,
        )?;
        diagnostics.position_factor_solves += seed.factor_solves;
        diagnostics.recovery_active_set_pivots += seed.pivots;
        diagnostics.recovery_active_set_factorizations += seed.basis_factorizations;
        diagnostics.recovery_active_set_storage = diagnostics
            .recovery_active_set_storage
            .max(seed.basis_storage);
        let displacement = if let Some(impulses) = seed.impulses {
            let mut displacement = vec![0.0; size];
            for (block, lambda) in blocks.iter().zip(impulses) {
                for (out, j) in displacement.iter_mut().zip(&block.jacobian[0]) {
                    *out += j * lambda;
                }
            }
            diagnostics.position_factor_solves += 1;
            factor.solve(&mut displacement)?;
            displacement
        } else {
            let remaining = settings.constraint_iterations - seed.pivots;
            if remaining == 0 {
                return Err(PhysicsError::NotConverged);
            }
            let correction = crate::solve_constraints(
                &factor,
                &blocks,
                remaining,
                CONTACT_ACTIVATION_DISTANCE * 0.1,
            )?;
            diagnostics.position_factor_solves += correction.factor_solves;
            diagnostics.record_newton(&correction);
            if !correction.converged {
                return Err(PhysicsError::NotConverged);
            }
            correction.velocity_change
        };
        let mut candidate = state.clone();
        candidate.velocities = displacement;
        // A correction can meet a previously separated surface. Advance only
        // its certified prefix, then refresh the finite contacts and solve again.
        // This changes no physical velocity and consumes no physical time.
        let fraction = correction_prefix(creation, &candidate, terrain, diagnostics)?;
        advance_positions(creation, &mut candidate, fraction);
        candidate.velocities.clone_from(&state.velocities);
        candidate.poses =
            MachineDynamics::reconstruct_poses(creation, &candidate.poses, &candidate.coordinates)?;
        diagnostics.terrain_recovery_poses += 1;
        validate_positions(creation, drives, &candidate)?;
        *state = candidate;
        diagnostics.terrain_recovery_passes += 1;
    }
    Err(PhysicsError::NotConverged)
}

fn correction_prefix(
    creation: &CompiledCreation,
    state: &MachineState,
    terrain: &TerrainSubstep<'_>,
    diagnostics: &mut JointTickDiagnostics,
) -> Result<f64, PhysicsError> {
    let mut fraction = 1.0;
    for _ in 0..16 {
        match super::events::validate_path(
            creation,
            state,
            fraction,
            Some(terrain),
            true,
            None,
            diagnostics,
        )? {
            super::events::PathOutcome::Clear | super::events::PathOutcome::Activate(_) => {
                return Ok(fraction);
            }
            super::events::PathOutcome::Refine(hit) => {
                let next = fraction * hit.fraction;
                if next <= 0.0 || next >= fraction {
                    break;
                }
                fraction = next;
            }
        }
    }
    Err(PhysicsError::NotConverged)
}
