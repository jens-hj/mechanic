//! World contact validation and rigid-frame publication for moving weld targets.

use bevy::prelude::*;
use mechanic_core::{
    CompiledCreation, ConstructionFrame, ConstructionGraph, FaceOwner, LocalCollider, PartId,
};
use mechanic_gpu::GpuTransform;

use crate::builder;
use crate::simulation::state::AppSimulation;

const CONTACT_TOLERANCE: f32 = 0.001;
const MAX_LIVE_PENETRATION: f32 = 0.005;

/// Stages an in-place weld between two touching live bodies. Bodies of one
/// creation weld in their authored arrangement; a separate creation is
/// reframed to its current pose relative to the first.
pub(crate) fn stage(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    first: PartId,
    second: PartId,
) -> Result<ConstructionGraph, String> {
    let creation = simulation
        .creation
        .as_ref()
        .ok_or("Wait for live physics before welding")?;
    let live = simulation
        .live_state
        .as_ref()
        .ok_or("Wait for an authoritative physics snapshot before welding")?;
    let first_body = body_for(creation, first).ok_or("First weld target is no longer published")?;
    let second_body =
        body_for(creation, second).ok_or("Second weld target is no longer published")?;
    if first_body == second_body {
        return Err("Select two different rigid bodies".to_owned());
    }
    let world = creation
        .colliders
        .iter()
        .map(|collider| {
            let pose = live
                .transforms
                .get(collider.compound_index as usize)
                .ok_or("Weld snapshot is incomplete")?;
            geometry(collider, *pose)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let body_geometry = |body: usize| {
        creation
            .colliders
            .iter()
            .zip(&world)
            .filter(move |(collider, _)| collider.compound_index as usize == body)
            .map(|(_, geometry)| geometry)
    };
    let mut touching = false;
    for a in body_geometry(first_body) {
        for b in body_geometry(second_body) {
            let depth = penetration(a, b);
            if depth > MAX_LIVE_PENETRATION {
                return Err("Weld targets penetrate each other too deeply".to_owned());
            }
            touching |= depth >= -CONTACT_TOLERANCE;
        }
    }
    if !touching {
        return Err("The selected moving bodies do not touch".to_owned());
    }
    let first_component = graph
        .structural_component(first, [])
        .map_err(|e| e.to_string())?;
    if first_component.contains(second) {
        return builder::stage_weld_objects(graph, FaceOwner::Part(first), FaceOwner::Part(second))
            .map_err(|e| e.to_string());
    }
    let second_component = graph
        .structural_component(second, [])
        .map_err(|e| e.to_string())?;
    if second_component.touches_authored_ground() {
        return Err("Select the grounded creation first when welding".to_owned());
    }
    let relative = world_from_build(creation, &live.transforms, first_body)?
        .inverse()
        .compose(world_from_build(creation, &live.transforms, second_body)?);
    let mut staged = graph.clone();
    staged
        .reframe_parts(second_component.parts(), relative)
        .map_err(|e| e.to_string())?;
    let defaults = staged.compile().map_err(|e| e.to_string())?;
    let default_geometry = defaults
        .colliders
        .iter()
        .map(|collider| {
            let body = &defaults.compounds[collider.compound_index as usize];
            geometry(
                collider,
                GpuTransform {
                    position: body.root_translation.extend(0.0).to_array(),
                    rotation: body.root_rotation.to_array(),
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let component_geometry = |component: &mechanic_core::StructuralComponent| {
        defaults
            .colliders
            .iter()
            .zip(&default_geometry)
            .filter(|(collider, _)| component.contains(collider.source_part))
            .map(|(_, geometry)| geometry)
            .collect::<Vec<_>>()
    };
    let second_default = component_geometry(&second_component);
    for a in component_geometry(&first_component) {
        for &b in &second_default {
            if penetration(a, b) > CONTACT_TOLERANCE {
                return Err("The combined Garage default pose would intersect itself".to_owned());
            }
        }
    }
    builder::stage_weld_objects(&staged, FaceOwner::Part(first), FaceOwner::Part(second))
        .map_err(|e| e.to_string())
}

fn body_for(creation: &CompiledCreation, part: PartId) -> Option<usize> {
    creation
        .part_to_compound
        .iter()
        .find_map(|&(id, body)| (id == part).then_some(body as usize))
}

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
mod tests;
