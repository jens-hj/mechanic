//! World contact validation and rigid-frame publication for moving weld targets.

use bevy::prelude::*;
use mechanic_core::{CompiledCreation, ConstructionFrame, LocalCollider};
use mechanic_gpu::GpuTransform;

pub(crate) fn world_from_build(
    creation: &CompiledCreation,
    poses: &[GpuTransform],
    body: usize,
) -> Result<ConstructionFrame, String> {
    let initial = creation
        .compounds
        .get(body)
        .ok_or("Weld body is unavailable")?;
    let pose = poses.get(body).ok_or("Weld snapshot is incomplete")?;
    let rotation = Quat::from_array(pose.rotation) * initial.root_rotation.conjugate();
    ConstructionFrame::new(
        Vec3::from_slice(&pose.position[..3]) - rotation * initial.root_translation,
        rotation,
    )
    .map_err(|e| e.to_string())
}

pub(crate) type ConvexGeometry = mechanic_core::WeldCollider;

pub(crate) fn geometry(
    collider: &LocalCollider,
    pose: GpuTransform,
) -> Result<ConvexGeometry, String> {
    let transform = ConstructionFrame::new(
        Vec3::from_slice(&pose.position[..3]),
        Quat::from_array(pose.rotation),
    )
    .map_err(|e| e.to_string())?;
    Ok(ConvexGeometry::new(collider, transform))
}

/// Negative means a separating gap; positive means overlap along every SAT axis.
pub(crate) fn penetration(a: &ConvexGeometry, b: &ConvexGeometry) -> f32 {
    a.penetration(b)
}

#[cfg(test)]
#[path = "live_weld_tests.rs"]
mod tests;
