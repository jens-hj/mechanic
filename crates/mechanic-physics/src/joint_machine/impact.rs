//! Instantaneous coupled contact and joint-stop impact. No elapsed force time.

use super::{JointTickConfig, JointTickDiagnostics, bounds, stops};
use crate::{
    ConstraintBlock, ImpulseBounds, MachineDynamics, MachineState, PhysicsError,
    TerrainImpactConstraints,
};
use mechanic_core::{CompiledCreation, CoordinateDrive};

pub(super) fn activate(
    creation: &CompiledCreation,
    model: &MachineDynamics,
    drives: &[CoordinateDrive],
    state: &mut MachineState,
    contacts: &TerrainImpactConstraints,
    settings: JointTickConfig,
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

fn activate_candidate(
    creation: &CompiledCreation,
    model: &MachineDynamics,
    drives: &[CoordinateDrive],
    state: &mut MachineState,
    contacts: &TerrainImpactConstraints,
    settings: JointTickConfig,
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
    trace_normals("incoming", contacts, &state.velocities, None);
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
    trace_normals(
        "outgoing",
        contacts,
        &outgoing,
        Some((solution.residual, &solution.impulses)),
    );
    state.velocities = outgoing;
    diagnostics.impact_events += 1;
    Ok(true)
}

// Normal closing speeds of one impact's contact points, gated like
// `MECHANIC_TRACE_CONTINUATION`. Tracing both ends of the solve measures how much
// of an incoming manifold an admissible solution ejects from the activation
// window, which is what leaves a rolling support re-arriving every substep.
fn trace_normals(
    label: &str,
    contacts: &TerrainImpactConstraints,
    velocities: &[f64],
    solved: Option<(f64, &[f64])>,
) {
    if std::env::var_os("MECHANIC_TRACE_EVENTS").is_none() {
        return;
    }
    let mut normals = Vec::new();
    for block in &contacts.blocks {
        let mut first = 0;
        for point in &block.contacts {
            normals.push(
                block.jacobian[first]
                    .iter()
                    .zip(velocities)
                    .map(|(j, v)| j * v)
                    .sum::<f64>(),
            );
            first += if point.rolling_length.is_some() { 5 } else { 3 };
        }
    }
    match solved {
        None => println!(
            "event impact {label} blocks={} normals={normals:?}",
            contacts.blocks.len()
        ),
        Some((residual, impulses)) => println!(
            "event impact {label} normals={normals:?} residual={residual:e} impulses={impulses:?}"
        ),
    }
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
