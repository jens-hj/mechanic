//! Existing selection-mesh geometry placed in the current gesture's moving grid.

use std::collections::HashSet;

use bevy::{
    asset::RenderAssetUsages, mesh::Indices, prelude::*, render::render_resource::PrimitiveTopology,
};
use mechanic_core::{ConstructionGraph, PartId, SolidOwner};

use crate::live_edit::EditContext;
use crate::render::mesh::construction::BuildTransform;
use crate::simulation::state::AppSimulation;

/// Highlights the selected rigid body using each published body's current pose.
pub(crate) fn weld_preview_mesh(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    part: PartId,
    context: Option<EditContext>,
) -> Mesh {
    let parts = crate::builder::rigid_body_parts(graph, part);
    parts_preview_mesh(graph, simulation, &parts, context, 1.018)
}

/// Highlights an explicit selection, retaining its frames and published poses.
pub(crate) fn parts_preview_mesh(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    parts: &[PartId],
    context: Option<EditContext>,
    scale: f32,
) -> Mesh {
    let graph = graph.canonicalized();
    let into_context = context.map(|context| context.frame_to_world.inverse());
    let mut seen = HashSet::new();
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    for &member in parts {
        let owner = graph
            .region_of(member)
            .map_or(SolidOwner::Part(member), SolidOwner::Region);
        if !seen.insert(owner) {
            continue;
        }
        let start = positions.len();
        let evaluated =
            matches!(owner, SolidOwner::Region(_)) || graph.owner_has_shape_features(owner);
        if evaluated {
            let Ok(solid) = graph.evaluated_solid(owner) else {
                continue;
            };
            crate::render::mesh::construction::append_evaluated_solid(
                &solid,
                BuildTransform::IDENTITY,
                &mut positions,
                &mut normals,
                &mut Vec::new(),
                &mut Vec::new(),
                &mut indices,
            );
            let owner_frame = match owner {
                SolidOwner::Part(part) => graph.part_frame(part),
                SolidOwner::Region(region) => graph.region_frame(region),
            }
            .expect("highlight owner has an authored frame");
            let inverse = owner_frame.inverse();
            let (low, high) = positions[start..].iter().fold(
                (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                |(low, high), point| {
                    let point = inverse.point(Vec3::from_array(*point));
                    (low.min(point), high.max(point))
                },
            );
            let center = owner_frame.point((low + high) * 0.5);
            for point in &mut positions[start..] {
                *point = (center + (Vec3::from_array(*point) - center) * scale).to_array();
            }
        } else if let Some(spec) = graph.part(member) {
            crate::render::mesh::construction::append_part(
                *spec,
                scale,
                &mut positions,
                &mut normals,
                &mut indices,
            );
        }
        let placement = published_placement(simulation, member);
        let frame = (!evaluated).then(|| graph.part_frame(member)).flatten();
        for (point, normal) in positions[start..].iter_mut().zip(&mut normals[start..]) {
            let mut position = Vec3::from_array(*point);
            let mut direction = Vec3::from_array(*normal);
            if let Some(frame) = frame {
                position = frame.point(position);
                direction = frame.vector(direction);
            }
            position = placement.point(position);
            direction = placement.direction(direction);
            if let Some(frame) = into_context {
                position = frame.point(position);
                direction = frame.vector(direction);
            }
            *point = position.to_array();
            *normal = direction.to_array();
        }
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

fn published_placement(simulation: &AppSimulation, part: PartId) -> BuildTransform {
    let Some(creation) = simulation.creation.as_ref() else {
        return BuildTransform::IDENTITY;
    };
    let Some(body) = creation
        .part_to_compound
        .iter()
        .find_map(|&(member, body)| (member == part).then_some(body as usize))
    else {
        return BuildTransform::IDENTITY;
    };
    let Some(initial) = creation.compounds.get(body) else {
        return BuildTransform::IDENTITY;
    };
    let Some(pose) = simulation.transforms.get(body) else {
        return BuildTransform::IDENTITY;
    };
    BuildTransform {
        origin: initial.root_translation,
        rotation: Quat::from_array(pose.rotation) * initial.root_rotation.conjugate(),
        translation: Vec3::from_slice(&pose.position[..3]),
    }
}

#[cfg(test)]
mod tests;
