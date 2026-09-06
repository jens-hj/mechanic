//! World contact validation and rigid-frame publication for moving weld targets.

use bevy::prelude::*;
use mechanic_core::{
    BuildCommand, ColliderShape, CompiledCreation, ConstructionFrame, ConstructionGraph, FaceOwner,
    LocalCollider, PartId, RigidLinkSpec,
};
use mechanic_gpu::GpuTransform;

use crate::{AppSimulation, builder};

const CONTACT_TOLERANCE: f32 = 0.001;
const MAX_LIVE_PENETRATION: f32 = 0.005;

/// Stages a weld against one authoritative live snapshot, preserving authored joint defaults.
#[allow(clippy::too_many_lines)] // Validate and publish one rigid weld transaction.
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
    let first_component = graph
        .structural_component(first, [])
        .map_err(|e| e.to_string())?;
    let second_component = graph
        .structural_component(second, [])
        .map_err(|e| e.to_string())?;
    let world = collider_geometry(creation, &live.transforms)?;
    let mut touching = false;
    for (_, a) in creation
        .colliders
        .iter()
        .zip(&world)
        .filter(|(c, _)| c.compound_index as usize == first_body)
    {
        for (_, b) in creation
            .colliders
            .iter()
            .zip(&world)
            .filter(|(c, _)| c.compound_index as usize == second_body)
        {
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
    if first_component.contains(second) {
        let mut staged =
            builder::stage_weld_objects(graph, FaceOwner::Part(first), FaceOwner::Part(second))
                .map_err(|e| e.to_string())?;
        staged
            .apply(BuildCommand::CancelPending)
            .map_err(|e| e.to_string())?;
        staged.compile().map_err(|e| e.to_string())?;
        return Ok(staged);
    }
    if second_component.touches_authored_ground() {
        return Err("Select the grounded creation first when welding".to_owned());
    }
    let first_motion = world_from_build(creation, &live.transforms, first_body)?;
    let second_motion = world_from_build(creation, &live.transforms, second_body)?;
    let relative = first_motion.inverse().compose(second_motion);
    let mut staged = graph.clone();
    staged
        .reframe_parts(second_component.parts(), relative)
        .map_err(|e| e.to_string())?;
    let defaults = staged.compile().map_err(|e| e.to_string())?;
    let poses = defaults
        .compounds
        .iter()
        .map(|body| GpuTransform {
            position: body.root_translation.extend(0.0).to_array(),
            rotation: body.root_rotation.to_array(),
        })
        .collect::<Vec<_>>();
    let default_geometry = collider_geometry(&defaults, &poses)?;
    for (_, a) in defaults
        .colliders
        .iter()
        .zip(&default_geometry)
        .filter(|(c, _)| first_component.contains(c.source_part))
    {
        for (_, b) in defaults
            .colliders
            .iter()
            .zip(&default_geometry)
            .filter(|(c, _)| second_component.contains(c.source_part))
        {
            if penetration(a, b) > CONTACT_TOLERANCE {
                return Err("The combined Garage default pose would intersect itself".to_owned());
            }
        }
    }
    staged =
        match builder::stage_weld_objects(&staged, FaceOwner::Part(first), FaceOwner::Part(second))
        {
            Ok(welded) => welded,
            Err(builder::PlacementError::ObjectsDoNotTouch) => {
                // SAT has proved body contact and a valid combined default pose.
                // Edge and arbitrary-angle contacts need rigid membership without a face pair.
                staged
                    .apply(BuildCommand::RigidLink(RigidLinkSpec { first, second }))
                    .map_err(|e| e.to_string())?;
                staged
            }
            Err(error) => return Err(error.to_string()),
        };
    staged
        .apply(BuildCommand::CancelPending)
        .map_err(|e| e.to_string())?;
    staged.compile().map_err(|e| e.to_string())?;
    Ok(staged)
}

fn body_for(creation: &CompiledCreation, part: PartId) -> Option<usize> {
    creation
        .part_to_compound
        .iter()
        .find_map(|&(id, body)| (id == part).then_some(body as usize))
}

fn world_from_build(
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

pub(crate) struct ConvexGeometry {
    pub(crate) vertices: Vec<Vec3>,
    pub(crate) normals: Vec<Vec3>,
    pub(crate) edges: Vec<Vec3>,
}

fn collider_geometry(
    creation: &CompiledCreation,
    poses: &[GpuTransform],
) -> Result<Vec<ConvexGeometry>, String> {
    creation
        .colliders
        .iter()
        .map(|collider| {
            let pose = poses
                .get(collider.compound_index as usize)
                .ok_or("Weld snapshot is incomplete")?;
            geometry(collider, *pose)
        })
        .collect()
}

pub(crate) fn geometry(
    collider: &LocalCollider,
    pose: GpuTransform,
) -> Result<ConvexGeometry, String> {
    let transform = ConstructionFrame::new(
        Vec3::from_slice(&pose.position[..3]),
        Quat::from_array(pose.rotation),
    )
    .map_err(|e| e.to_string())?;
    Ok(match &collider.shape {
        ColliderShape::Cuboid {
            local_rotation,
            half_extents,
        } => {
            let rotation = transform.rotation() * *local_rotation;
            let center = transform.point(collider.local_center);
            let axes = [rotation * Vec3::X, rotation * Vec3::Y, rotation * Vec3::Z];
            let mut vertices = Vec::with_capacity(8);
            for x in [-1.0, 1.0] {
                for y in [-1.0, 1.0] {
                    for z in [-1.0, 1.0] {
                        vertices.push(center + rotation * (*half_extents * Vec3::new(x, y, z)));
                    }
                }
            }
            ConvexGeometry {
                vertices,
                normals: axes.to_vec(),
                edges: axes.to_vec(),
            }
        }
        ColliderShape::Convex(shape) => ConvexGeometry {
            // Convex vertices already use the compound's origin, unlike cuboid centers.
            vertices: shape.vertices.iter().map(|&v| transform.point(v)).collect(),
            normals: shape
                .face_planes
                .iter()
                .map(|p| transform.vector(p.truncate()))
                .collect(),
            edges: shape
                .edge_directions
                .iter()
                .map(|&e| transform.vector(e))
                .collect(),
        },
    })
}

/// Negative means a separating gap; positive means overlap along every SAT axis.
pub(crate) fn penetration(a: &ConvexGeometry, b: &ConvexGeometry) -> f32 {
    let mut minimum = f32::INFINITY;
    for axis in a
        .normals
        .iter()
        .copied()
        .chain(b.normals.iter().copied())
        .chain(
            a.edges
                .iter()
                .flat_map(|&a| b.edges.iter().map(move |&b| a.cross(b))),
        )
    {
        let Some(axis) = axis.try_normalize() else {
            continue;
        };
        let interval = |vertices: &[Vec3]| {
            vertices
                .iter()
                .map(|v| v.dot(axis))
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), value| {
                    (low.min(value), high.max(value))
                })
        };
        let (a_low, a_high) = interval(&a.vertices);
        let (b_low, b_high) = interval(&b.vertices);
        minimum = minimum.min((a_high - b_low).min(b_high - a_low));
    }
    minimum
}

#[cfg(test)]
#[path = "live_weld_tests.rs"]
mod tests;
