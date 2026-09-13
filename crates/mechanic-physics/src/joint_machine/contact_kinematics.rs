//! Exact endpoint normal velocity for the force-dependent pose trial.

use super::{
    CompiledCreation, ConstraintBlock, DVec3, JointTickDiagnostics, MachineDynamics, MachineState,
    PhysicsError, advance_positions,
};

pub(super) struct ContactPoint {
    pub row: usize,
    pub body: usize,
    pub local_point: DVec3,
    pub normal: DVec3,
    // Opposing body and its local point; the row measures relative speed.
    pub other: Option<(usize, DVec3)>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn endpoint_targets(
    creation: &CompiledCreation,
    initial: &MachineState,
    candidate: &[f64],
    dt: f64,
    blocks: &[ConstraintBlock],
    desired: &[f64],
    points: &[ContactPoint],
    diagnostics: &mut JointTickDiagnostics,
) -> Result<Vec<f64>, PhysicsError> {
    let mut targets = desired.to_vec();
    if points.is_empty() {
        return Ok(targets);
    }
    let (poses, motions) = endpoint_motion(creation, initial, candidate, dt, diagnostics)?;
    let rows = blocks
        .iter()
        .flat_map(|block| &block.jacobian)
        .collect::<Vec<_>>();
    for point in points {
        let mut speed = point_speed(
            creation,
            &poses,
            &motions,
            point.body,
            point.local_point,
            point.normal,
        );
        if let Some((body, local)) = point.other {
            speed -= point_speed(creation, &poses, &motions, body, local, point.normal);
        }
        let initial_speed = rows[point.row]
            .iter()
            .zip(candidate)
            .map(|(j, v)| j * v)
            .sum::<f64>();
        targets[point.row] -= speed - initial_speed;
    }
    if targets.iter().any(|target| !target.is_finite()) {
        return Err(PhysicsError::InvalidDynamics);
    }
    Ok(targets)
}

pub(super) fn endpoint_motion(
    creation: &CompiledCreation,
    initial: &MachineState,
    candidate: &[f64],
    dt: f64,
    diagnostics: &mut JointTickDiagnostics,
) -> Result<(Vec<crate::BodyPose>, Vec<crate::SpatialMotion>), PhysicsError> {
    let mut end = initial.clone();
    for (v, next) in end.velocities.iter_mut().zip(candidate) {
        *v = 0.5 * (*v + next);
    }
    advance_positions(creation, &mut end, dt);
    let poses = MachineDynamics::reconstruct_poses(creation, &end.poses, &end.coordinates)?;
    diagnostics.contact_kinematic_poses += 1;
    let motions = MachineDynamics::reconstruct_motions(creation, &poses, candidate)?;
    diagnostics.contact_velocity_traversals += 1;
    Ok((poses, motions))
}

pub(super) fn point_speed(
    creation: &CompiledCreation,
    poses: &[crate::BodyPose],
    motions: &[crate::SpatialMotion],
    body: usize,
    local: DVec3,
    normal: DVec3,
) -> f64 {
    let arm = poses[body].rotation * (local - creation.dynamics.inertias[body].center.as_dvec3());
    normal.dot(motions[body].linear + motions[body].angular.cross(arm))
}
