//! Conversions between published GPU transforms, Bevy transforms, and construction poses.

use crate::builder::faces::face_geometry_from_ref;
use crate::editor::build_actions::PlacedBearing;
use crate::simulation::state::AppSimulation;
use bevy::prelude::{Quat, Transform, Vec3, default};
use mechanic_core::{CompiledCreation, ConstructionGraph, FaceOwner, PartId};
use mechanic_gpu::GpuTransform;

pub(crate) fn transform_from_gpu(transform: GpuTransform) -> Transform {
    Transform {
        translation: Vec3::from_slice(&transform.position[..3]),
        rotation: Quat::from_array(transform.rotation).normalize(),
        ..default()
    }
}

/// Resolves authored part coordinates through the published body's pose delta.
pub(crate) fn simulation_part_pose(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    part: PartId,
) -> Option<(Vec3, Quat)> {
    let body = creation
        .part_to_compound
        .iter()
        .find_map(|&(candidate, body)| (candidate == part).then_some(body))?
        as usize;
    let initial = creation.compounds.get(body)?;
    let transform = transforms.get(body)?;
    let rotation = Quat::from_array(transform.rotation) * initial.root_rotation.conjugate();
    Some((
        Vec3::from_slice(&transform.position[..3])
            + rotation * (graph.part_position(part)? - initial.root_translation),
        rotation * graph.part_rotation(part)?,
    ))
}

/// Where one graph bearing sits in simulation space.
///
/// The published snapshot moves compounds, so a running mechanism's joint is
/// found through its compiled row rather than through its build pose.
pub(crate) fn simulation_bearing_pose(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    bearing: &mechanic_core::BearingSpec,
) -> Option<(Vec3, Vec3)> {
    let compiled = creation
        .bearings
        .iter()
        .find(|compiled| graph.bearing(compiled.source_bearing) == Some(bearing))?;
    Some(transform_bearing_pose(
        *transforms.get(compiled.compound_a as usize)?,
        compiled.local_anchor_a,
        compiled.local_axis_a,
    ))
}

pub(crate) fn simulation_placed_bearing_pose(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    bearing: PlacedBearing,
) -> Option<(Vec3, Vec3)> {
    let FaceOwner::Part(source_part) = bearing.source.owner else {
        return None;
    };
    let compound_index = creation
        .part_to_compound
        .iter()
        .find_map(|&(part, index)| (part == source_part).then_some(index))?;
    let initial = creation.compounds.get(compound_index as usize)?;
    let inverse_initial_rotation = initial.root_rotation.inverse();
    let local_anchor = inverse_initial_rotation * (bearing.anchor - initial.root_translation);
    let local_axis =
        inverse_initial_rotation * face_geometry_from_ref(bearing.source, Some(graph)).normal;
    Some(transform_bearing_pose(
        *transforms.get(compound_index as usize)?,
        local_anchor,
        local_axis,
    ))
}

pub(crate) fn live_placed_bearing_pose(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    bearing: PlacedBearing,
) -> Option<(Vec3, Vec3)> {
    if simulation.creation.is_some() {
        return simulation_placed_bearing_pose(
            &simulation.published_graph,
            simulation.creation.as_ref()?,
            &simulation.transforms,
            bearing,
        );
    }
    Some((
        bearing.anchor,
        face_geometry_from_ref(bearing.source, Some(graph)).normal,
    ))
}

pub(crate) fn transform_bearing_pose(
    transform: GpuTransform,
    local_anchor: Vec3,
    local_axis: Vec3,
) -> (Vec3, Vec3) {
    let translation = Vec3::new(
        transform.position[0],
        transform.position[1],
        transform.position[2],
    );
    let rotation = Quat::from_array(transform.rotation);
    (translation + rotation * local_anchor, rotation * local_axis)
}

/// A published transform from a position and rotation.
pub(crate) fn gpu_transform(position: Vec3, rotation: Quat) -> GpuTransform {
    GpuTransform {
        position: position.extend(0.0).to_array(),
        rotation: rotation.to_array(),
    }
}
