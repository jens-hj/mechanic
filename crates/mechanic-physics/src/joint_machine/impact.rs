//! Instantaneous coupled contact and joint-stop impact. No elapsed force time.

use super::{JointTickDiagnostics, JointTickSettings, bounds, stops};
use crate::{
    ConstraintBlock, ImpulseBounds, MachineDynamics, MachineState, PhysicsError,
    TerrainImpactConstraints,
};
use mechanic_core::{CompiledCreation, CoordinateDrive};

#[allow(clippy::too_many_arguments)] // Preserve the operation boundary in failure diagnostics.
pub(super) fn activate(
    creation: &CompiledCreation,
    model: &MachineDynamics,
    drives: &[CoordinateDrive],
    state: &mut MachineState,
    contacts: &TerrainImpactConstraints,
    settings: JointTickSettings,
    diagnostics: &mut JointTickDiagnostics,
) -> Result<bool, PhysicsError> {
    activate_candidate(
        creation,
        model,
        drives,
        state,
        contacts,
        settings,
        diagnostics,
    )
    .inspect_err(|_| {
        diagnostics.failure_stage = Some(super::JointFailureStage::Impact);
    })
}

#[allow(clippy::too_many_arguments)] // Explicit pose model, constraints, command bounds and transaction-local state.
fn activate_candidate(
    creation: &CompiledCreation,
    model: &MachineDynamics,
    drives: &[CoordinateDrive],
    state: &mut MachineState,
    contacts: &TerrainImpactConstraints,
    settings: JointTickSettings,
    diagnostics: &mut JointTickDiagnostics,
) -> Result<bool, PhysicsError> {
    let closing = contacts.blocks.iter().any(|block| {
        let mut first = 0;
        block.contacts.iter().any(|point| {
            let speed = block.jacobian[first]
                .iter()
                .zip(&state.velocities)
                .map(|(j, v)| j * v)
                .sum::<f64>();
            first += if point.rolling_length.is_some() { 5 } else { 3 };
            speed < -settings.tolerance
        })
    });
    if !closing && !stops::closing(creation, drives, state, settings.tolerance) {
        return Ok(false);
    }
    let mut blocks = contacts.blocks.clone();
    // Finite actuator/passive forces have zero impulse at zero elapsed time.
    // Active hard stops must still prevent contact from driving through a limit.
    for (coordinate, &row) in creation.dynamics.coordinate_velocities.iter().enumerate() {
        let [lower, upper] = bounds(creation, drives, coordinate);
        for (sign, active) in [
            (
                1.0,
                lower.is_finite()
                    && state.coordinates[coordinate] <= lower + stops::POSITION_TOLERANCE,
            ),
            (
                -1.0,
                upper.is_finite()
                    && state.coordinates[coordinate] >= upper - stops::POSITION_TOLERANCE,
            ),
        ] {
            if !active {
                continue;
            }
            let mut jacobian = vec![0.0; state.velocities.len()];
            jacobian[row] = sign;
            blocks.push(ConstraintBlock {
                jacobian: vec![jacobian],
                target: vec![-sign * state.velocities[row]],
                bounds: vec![ImpulseBounds {
                    minimum: 0.0,
                    maximum: f64::INFINITY,
                }],
                contacts: Vec::new(),
            });
        }
    }
    diagnostics.impact_attempts += 1;
    diagnostics.impact_rows_prepared +=
        blocks.iter().map(|block| block.target.len()).sum::<usize>();
    let factor = settings.factorization.factor(
        creation,
        model,
        &state.coordinates,
        &vec![0.0; state.velocities.len()],
    )?;
    diagnostics.factorizations += 1;
    let solution = crate::solve_constraints(
        &factor,
        &blocks,
        settings.constraint_iterations,
        settings.tolerance * 0.1,
    )?;
    diagnostics.impact_factor_solves += solution.factor_solves;
    diagnostics.impact_iterations += solution.iterations;
    diagnostics.record_newton(&solution);
    diagnostics.impact_velocity_residual =
        diagnostics.impact_velocity_residual.max(solution.residual);
    diagnostics.residual = diagnostics.residual.max(solution.residual);
    if !solution.converged {
        #[cfg(test)]
        capture_failed_impact(model, &blocks);
        return Err(PhysicsError::NotConverged);
    }
    let outgoing = state
        .velocities
        .iter()
        .zip(solution.velocity_change)
        .map(|(old, change)| old + change)
        .collect::<Vec<_>>();
    model.body_motions(&outgoing)?;
    state.velocities = outgoing;
    diagnostics.impact_events += 1;
    Ok(true)
}

#[cfg(test)]
fn capture_failed_impact(model: &MachineDynamics, blocks: &[ConstraintBlock]) {
    let Some(path) = std::env::var_os("MECHANIC_IMPACT_CAPTURE") else {
        return;
    };
    let rows = blocks
        .iter()
        .map(|block| {
            (
                &block.jacobian,
                &block.target,
                block
                    .bounds
                    .iter()
                    .map(|bound| (bound.minimum, bound.maximum))
                    .collect::<Vec<_>>(),
                block
                    .contacts
                    .iter()
                    .map(|contact| {
                        (
                            contact.static_coefficient,
                            contact.kinetic_coefficient,
                            contact.sliding,
                            contact.rolling_length,
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    std::fs::write(path, ron::to_string(&(&model.mass_matrix, rows)).unwrap()).unwrap();
}

#[cfg(test)]
mod tests;
