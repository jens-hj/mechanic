//! Meshes of compiled compounds while the simulation runs.

use crate::builder::face_geometry_from_ref;
use crate::editor::build_actions::{PlacedBearing, bearing_uses_socket};
use crate::pose::transform_bearing_pose;
use crate::render::mesh::bearing::append_bearing_cylinder;
use crate::render::mesh::construction::{
    BuildTransform, append_authored_cuboid, append_evaluated_solid, append_layered_part,
    append_region, append_textured_part, ordinary_materials,
};
use crate::render::mesh::pipe::{pipe_end_faces, pipe_texture_offsets, welded_pipe_ends};
use crate::{AuthoredPart, chroma};
use bevy::asset::RenderAssetUsages;
use bevy::mesh::Indices;
use bevy::prelude::{Mesh, Quat, Vec3};
use bevy::render::render_resource::PrimitiveTopology;
use mechanic_core::{CompiledCreation, ConstructionGraph, ConstructionMaterial, FaceOwner};
use mechanic_gpu::GpuTransform;

#[derive(Clone, Copy)]
pub(crate) enum SimulationMeshKind {
    Static,
    Dynamic,
}

#[cfg(test)]
pub(crate) fn combined_simulation_mesh(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    kind: SimulationMeshKind,
) -> Mesh {
    combined_simulation_mesh_filtered(graph, creation, transforms, kind, None, None)
}

pub(crate) fn combined_simulation_material_mesh(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    kind: SimulationMeshKind,
    material: ConstructionMaterial,
) -> Mesh {
    combined_simulation_mesh_filtered(graph, creation, transforms, kind, Some(material), None)
}

pub(crate) fn local_simulation_material_mesh(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    local_transforms: &[GpuTransform],
    compound: u32,
    material: ConstructionMaterial,
) -> Mesh {
    combined_simulation_mesh_filtered(
        graph,
        creation,
        local_transforms,
        SimulationMeshKind::Dynamic,
        Some(material),
        Some(compound),
    )
}

pub(crate) fn simulation_material_is_present(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    kind: SimulationMeshKind,
    material: ConstructionMaterial,
) -> bool {
    creation.part_to_compound.iter().any(|&(part, compound)| {
        let is_static = creation.compounds[compound as usize].is_static;
        let right_motion = match kind {
            SimulationMeshKind::Static => is_static,
            SimulationMeshKind::Dynamic => !is_static,
        };
        right_motion
            && graph
                .part(part)
                .copied()
                .is_some_and(|spec| ordinary_materials(spec).any(|candidate| candidate == material))
    })
}

pub(crate) fn simulation_material_is_present_for_compound(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    compound: u32,
    material: ConstructionMaterial,
) -> bool {
    creation.part_to_compound.iter().any(|&(part, body)| {
        body == compound
            && graph
                .part(part)
                .copied()
                .is_some_and(|spec| ordinary_materials(spec).any(|candidate| candidate == material))
    })
}

#[expect(clippy::too_many_lines)]
pub(crate) fn combined_simulation_mesh_filtered(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    kind: SimulationMeshKind,
    material: Option<ConstructionMaterial>,
    compound_filter: Option<u32>,
) -> Mesh {
    let pipe_texture_offsets = pipe_texture_offsets(graph);
    let welded_pipe_ends = welded_pipe_ends(graph);
    let tooth_phases = mechanic_core::gear_phases(graph);
    let parts = creation
        .part_to_compound
        .iter()
        .filter(|(part, compound_index)| {
            let spec = graph.part(*part).copied();
            if compound_filter.is_some_and(|wanted| wanted != *compound_index) {
                return false;
            }
            let is_static = creation.compounds[*compound_index as usize].is_static;
            let right_motion = match kind {
                SimulationMeshKind::Static => is_static,
                SimulationMeshKind::Dynamic => !is_static,
            };
            right_motion
                && spec.is_some_and(|spec| {
                    ordinary_materials(spec)
                        .any(|part_material| material.is_none_or(|wanted| wanted == part_material))
                })
        });
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();
    let mut drawn_regions: Vec<mechanic_core::RegionId> = Vec::new();
    for &(part, compound_index) in parts {
        let transform = transforms[compound_index as usize];
        let root_translation = Vec3::from_array(transform.position[..3].try_into().unwrap());
        let root_rotation = Quat::from_array(transform.rotation);
        let initial = &creation.compounds[compound_index as usize];
        let spec = *graph.part(part).expect("compiled source remains in graph");
        let placement = BuildTransform {
            origin: initial.root_translation,
            rotation: root_rotation * initial.root_rotation.conjugate(),
            translation: root_translation,
        };
        // A part inside a region is drawn once, as its region.
        if let Some(id) = graph.region_of(part) {
            if drawn_regions.contains(&id) {
                continue;
            }
            drawn_regions.push(id);
            if let Some(region) = graph.region(id) {
                let first_vertex = positions.len();
                if graph.owner_has_shape_features(mechanic_core::SolidOwner::Region(id)) {
                    let solid = graph
                        .evaluated_solid_shared(mechanic_core::SolidOwner::Region(id))
                        .expect("compiled region feature geometry replays");
                    append_evaluated_solid(
                        &solid,
                        placement,
                        &mut positions,
                        &mut normals,
                        &mut uvs,
                        &mut tangents,
                        &mut indices,
                    );
                } else {
                    append_region(
                        region,
                        placement.with_frame(graph.region_frame(id).expect("region exists")),
                        &mut positions,
                        &mut normals,
                        &mut uvs,
                        &mut tangents,
                        &mut indices,
                    );
                }
                colors.extend(std::iter::repeat_n(
                    chroma::encode_appearance(region.appearance()),
                    positions.len() - first_vertex,
                ));
            }
            continue;
        }
        let frame = graph.part_frame(part).expect("part exists");
        let texture_offset = pipe_texture_offsets.get(&part).copied().unwrap_or_default();
        if spec.is_layered() {
            append_layered_part(
                graph,
                placement,
                part,
                spec,
                material,
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut tangents,
                &mut colors,
                &mut indices,
            );
            continue;
        }
        let first_vertex = positions.len();
        if graph.owner_has_shape_features(mechanic_core::SolidOwner::Part(part)) {
            let solid = graph
                .evaluated_solid_shared(mechanic_core::SolidOwner::Part(part))
                .expect("compiled feature geometry replays");
            append_evaluated_solid(
                &solid,
                placement,
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut tangents,
                &mut indices,
            );
        } else {
            append_textured_part(
                spec,
                placement.point(graph.part_position(part).expect("part exists")),
                placement.rotation
                    * graph.part_rotation(part).expect("part exists")
                    * super::gear::tooth_phase(&tooth_phases, part),
                placement.with_frame(frame),
                texture_offset,
                pipe_end_faces(part, &welded_pipe_ends),
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut tangents,
                &mut indices,
            );
        }
        colors.extend(std::iter::repeat_n(
            chroma::encode_appearance(spec.appearance().expect("ordinary parts have appearances")),
            positions.len() - first_vertex,
        ));
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, tangents)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_indices(Indices::U32(indices))
}

pub(crate) fn local_simulation_authored_mesh(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    local_transforms: &[GpuTransform],
    compound: u32,
    appearance: AuthoredPart,
    active_dimension_link: Option<mechanic_core::DimensionLinkId>,
) -> Mesh {
    combined_simulation_authored_mesh_filtered(
        graph,
        creation,
        local_transforms,
        Some(compound),
        appearance,
        active_dimension_link,
    )
}

pub(crate) fn combined_simulation_authored_mesh_filtered(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    compound_filter: Option<u32>,
    appearance: AuthoredPart,
    active_dimension_link: Option<mechanic_core::DimensionLinkId>,
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    let mut indices = Vec::new();
    for &(part, compound_index) in creation.part_to_compound.iter().filter(|(part, compound)| {
        compound_filter.is_none_or(|wanted| wanted == *compound)
            && graph
                .part(*part)
                .is_some_and(|spec| appearance.matches(graph, *part, *spec, active_dimension_link))
    }) {
        let transform = transforms[compound_index as usize];
        let root_translation = Vec3::from_array(transform.position[..3].try_into().unwrap());
        let root_rotation = Quat::from_array(transform.rotation)
            * creation.compounds[compound_index as usize]
                .root_rotation
                .conjugate();
        let initial = &creation.compounds[compound_index as usize];
        let spec = *graph.part(part).expect("compiled source remains in graph");
        let local_center =
            graph.part_position(part).expect("part exists") - initial.root_translation;
        let cuboid = spec
            .as_cuboid()
            .expect("authored machine appearances have cuboid envelopes");
        append_authored_cuboid(
            root_translation + root_rotation * local_center,
            root_rotation * graph.part_rotation(part).expect("part exists"),
            cuboid.size_meters(),
            appearance,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut tangents,
            &mut indices,
        );
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, tangents)
    .with_inserted_indices(Indices::U32(indices))
}

pub(crate) fn combined_simulation_bearing_mesh(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    placed_bearings: &[PlacedBearing],
) -> Mesh {
    combined_simulation_bearing_mesh_filtered(graph, creation, transforms, None, placed_bearings)
}

pub(crate) fn local_simulation_bearing_mesh(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    local_transforms: &[GpuTransform],
    compound: u32,
    placed_bearings: &[PlacedBearing],
) -> Mesh {
    combined_simulation_bearing_mesh_filtered(
        graph,
        creation,
        local_transforms,
        Some(compound),
        placed_bearings,
    )
}

pub(crate) fn combined_simulation_bearing_mesh_filtered(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    compound_filter: Option<u32>,
    placed_bearings: &[PlacedBearing],
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    let mut indices = Vec::new();

    for compiled in &creation.bearings {
        if compound_filter.is_some_and(|wanted| wanted != compiled.compound_a) {
            continue;
        }
        let bearing = graph
            .bearing(compiled.source_bearing)
            .expect("compiled bearing source remains in graph");
        if bearing.kind.is_translational() {
            continue;
        }
        if placed_bearings
            .iter()
            .any(|&socket| bearing_uses_socket(bearing, socket))
        {
            continue;
        }
        let (anchor, axis) = transform_bearing_pose(
            transforms[compiled.compound_a as usize],
            compiled.local_anchor_a,
            compiled.local_axis_a,
        );
        append_bearing_cylinder(
            anchor,
            axis,
            bearing.dimensions,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut tangents,
            &mut indices,
        );
    }

    for bearing in placed_bearings {
        if bearing.kind.is_translational() {
            continue;
        }
        let FaceOwner::Part(source_part) = bearing.source.owner else {
            continue;
        };
        let Some(compound_index) = creation
            .part_to_compound
            .iter()
            .find_map(|&(part, index)| (part == source_part).then_some(index))
        else {
            continue;
        };
        if compound_filter.is_some_and(|wanted| wanted != compound_index) {
            continue;
        }
        let initial = &creation.compounds[compound_index as usize];
        let inverse_initial_rotation = initial.root_rotation.inverse();
        let local_anchor = inverse_initial_rotation * (bearing.anchor - initial.root_translation);
        let source_axis = face_geometry_from_ref(bearing.source, Some(graph)).normal;
        let local_axis = inverse_initial_rotation * source_axis;
        let (anchor, axis) = transform_bearing_pose(
            transforms[compound_index as usize],
            local_anchor,
            local_axis,
        );
        append_bearing_cylinder(
            anchor,
            axis,
            bearing.dimensions,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut tangents,
            &mut indices,
        );
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, tangents)
    .with_inserted_indices(Indices::U32(indices))
}

pub(crate) fn simulation_body_has_bearing(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    compound: u32,
    placed_bearings: &[PlacedBearing],
) -> bool {
    creation.bearings.iter().any(|bearing| {
        matches!(bearing.kind, mechanic_core::JointKind::Rotational)
            && bearing.compound_a == compound
            && graph.bearing(bearing.source_bearing).is_some_and(|source| {
                !placed_bearings
                    .iter()
                    .any(|&socket| bearing_uses_socket(source, socket))
            })
    }) || placed_bearings.iter().any(|bearing| {
        if bearing.kind.is_translational() {
            return false;
        }
        let FaceOwner::Part(source) = bearing.source.owner else {
            return false;
        };
        creation
            .part_to_compound
            .iter()
            .any(|&(part, body)| part == source && body == compound)
    })
}
