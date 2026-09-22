//! Editor construction meshes: blocks, layered parts, regions, and authored machine parts.

use crate::builder::BLOCK_SIZE_METERS;
use crate::editor::state::EditorState;
use crate::render::authored::{
    AUTHORED_CUBE_INDICES, AUTHORED_CUBE_NORMALS, AUTHORED_CUBE_POSITIONS, AUTHORED_CUBE_TANGENTS,
    CUBE_INDICES, CUBE_NORMALS, authored_uvs,
};
use crate::render::mesh::pipe::{
    PipeEndFaces, PipeTextureOffset, append_cylinder_shape, append_cylinder_shape_with_end_faces,
    append_pipe_bend_shape, append_pipe_bend_shape_with_end_faces, append_pipe_junction_shape,
    pipe_end_faces, pipe_texture_offsets, welded_pipe_ends,
};
use crate::{AuthoredPart, builder, chroma, shape_tool};
use bevy::asset::RenderAssetUsages;
use bevy::mesh::Indices;
use bevy::prelude::{IVec3, Mesh, Quat, Vec2, Vec3, vec};
use bevy::render::render_resource::PrimitiveTopology;
use mechanic_core::{
    CellGrid, ConstructionGraph, ConstructionMaterial, CuboidSpec, CylinderDimensions, FaceOwner,
    GRID_UNIT_METERS, GridRotation, MaterialAppearance, POSITION_TICKS_PER_GRID_UNIT, PartId,
    PartPiece, PartSpec, PipeBendDimensions, RegionId, ShapeRegion, face_neighbour_offset,
};
use std::collections::{HashMap, HashSet};

pub(crate) const CUBE_POSITIONS: [[f32; 3]; 24] = [
    [-0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5],
    [0.5, -0.5, -0.5],
];

pub(crate) fn single_authored_part_mesh(appearance: AuthoredPart) -> Mesh {
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, AUTHORED_CUBE_POSITIONS.to_vec())
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, AUTHORED_CUBE_NORMALS.to_vec())
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, authored_uvs(appearance).to_vec())
    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, AUTHORED_CUBE_TANGENTS.to_vec())
    .with_inserted_indices(Indices::U32(AUTHORED_CUBE_INDICES.to_vec()))
}

/// The region as it would look with the current cage drag applied.
///
/// A drag is not in the graph until it is released, so previewing it means
/// folding the proposed offsets — mirrors included — into a copy.
pub(crate) fn preview_region(
    graph: &ConstructionGraph,
    state: &EditorState,
    mirror: shape_tool::ShapeMirror,
) -> Option<(RegionId, ShapeRegion)> {
    let id = state.active_region?;
    let mut region = graph.region(id)?.clone();
    if let Some(drag) = state.vertex_drag.as_ref() {
        for (index, offset) in shape_tool::drag_edits(&region, drag, mirror) {
            // A drag that would leave the box simply does not preview; the
            // command would reject it anyway.
            let _ = region.set_offset(index, offset);
        }
    }
    Some((id, region))
}

#[cfg(test)]
pub(crate) fn combined_construction_mesh(graph: &ConstructionGraph) -> Mesh {
    combined_construction_mesh_filtered(graph, None, None)
}

pub(crate) fn combined_material_construction_mesh(
    graph: &ConstructionGraph,
    preview: Option<&(RegionId, ShapeRegion)>,
    material: ConstructionMaterial,
) -> Mesh {
    combined_construction_mesh_filtered(graph, preview, Some(material))
}

/// Builds the construction mesh, substituting `preview` for the region it names
/// so a cage drag can be seen before it is committed.
#[expect(clippy::too_many_lines)]
pub(crate) fn combined_construction_mesh_filtered(
    graph: &ConstructionGraph,
    preview: Option<&(RegionId, ShapeRegion)>,
    material: Option<ConstructionMaterial>,
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();
    let pipe_texture_offsets = pipe_texture_offsets(graph);
    let welded_pipe_ends = welded_pipe_ends(graph);
    let rigid_groups = rigid_render_groups(graph);
    let tooth_phases = mechanic_core::gear_phases(graph);
    let mut mergeable_blocks = Vec::new();
    // A part inside a region hands its surface to that region, so drawing both
    // would render the same material twice.
    for (part, spec) in graph.parts().filter(|(_, spec)| {
        ordinary_materials(**spec)
            .any(|part_material| material.is_none_or(|wanted| wanted == part_material))
    }) {
        if graph.region_of(part).is_some() {
            continue;
        }
        if spec.is_layered() {
            append_layered_part(
                graph,
                BuildTransform::IDENTITY,
                part,
                *spec,
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
        if let PartSpec::Cuboid(cuboid) = *spec
            && cuboid
                .dimensions
                .iter()
                .all(|dimension| dimension.units() == 1)
            && cuboid.pose.rotation == GridRotation::default()
            && cuboid.rack().is_none()
            && graph.part_frame(part) == Some(mechanic_core::ConstructionFrame::IDENTITY)
            && !graph.owner_has_shape_features(mechanic_core::SolidOwner::Part(part))
        {
            mergeable_blocks.push((rigid_groups[part.index() as usize], cuboid));
            continue;
        }
        let texture_offset = pipe_texture_offsets.get(&part).copied().unwrap_or_default();
        let first_vertex = positions.len();
        if graph.owner_has_shape_features(mechanic_core::SolidOwner::Part(part)) {
            let solid = graph
                .evaluated_solid_shared(mechanic_core::SolidOwner::Part(part))
                .expect("committed feature geometry replays");
            append_evaluated_solid(
                &solid,
                BuildTransform::IDENTITY,
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut tangents,
                &mut indices,
            );
        } else {
            append_textured_part(
                *spec,
                graph.part_position(part).expect("part exists"),
                graph.part_rotation(part).expect("part exists")
                    * super::gear::tooth_phase(&tooth_phases, part),
                BuildTransform::IDENTITY.with_frame(graph.part_frame(part).expect("part exists")),
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
    append_merged_block_cuboids(
        &mergeable_blocks,
        &mut positions,
        &mut normals,
        &mut uvs,
        &mut tangents,
        &mut colors,
        &mut indices,
    );
    for (id, region) in graph.regions() {
        if material.is_some_and(|wanted| wanted != region.material()) {
            continue;
        }
        let shown = match preview {
            Some((preview_id, previewed)) if *preview_id == id => previewed,
            _ => region,
        };
        let first_vertex = positions.len();
        if preview.is_none()
            && graph.owner_has_shape_features(mechanic_core::SolidOwner::Region(id))
        {
            let solid = graph
                .evaluated_solid_shared(mechanic_core::SolidOwner::Region(id))
                .expect("committed region feature geometry replays");
            append_evaluated_solid(
                &solid,
                BuildTransform::IDENTITY,
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut tangents,
                &mut indices,
            );
        } else {
            append_region(
                shown,
                BuildTransform::IDENTITY.with_frame(graph.region_frame(id).expect("region exists")),
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut tangents,
                &mut indices,
            );
        }
        colors.extend(std::iter::repeat_n(
            chroma::encode_appearance(shown.appearance()),
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

pub(crate) fn rigid_render_groups(graph: &ConstructionGraph) -> Vec<usize> {
    let parts = graph.parts().map(|(part, _)| part).collect::<Vec<_>>();
    let mut dense =
        vec![usize::MAX; parts.iter().map(|part| part.index()).max().unwrap_or(0) as usize + 1];
    for (index, part) in parts.iter().enumerate() {
        dense[part.index() as usize] = index;
    }
    let mut parents = (0..parts.len()).collect::<Vec<_>>();
    for (_, weld) in graph.welds() {
        if let (FaceOwner::Part(first), FaceOwner::Part(second)) =
            (weld.first.owner, weld.second.owner)
        {
            union_render_groups(
                &mut parents,
                dense[first.index() as usize],
                dense[second.index() as usize],
            );
        }
    }
    for (_, link) in graph.rigid_links() {
        union_render_groups(
            &mut parents,
            dense[link.first.index() as usize],
            dense[link.second.index() as usize],
        );
    }
    let mut groups = vec![usize::MAX; dense.len()];
    for (index, part) in parts.into_iter().enumerate() {
        groups[part.index() as usize] = find_render_group(&mut parents, index);
    }
    groups
}

pub(crate) fn find_render_group(parents: &mut [usize], mut index: usize) -> usize {
    while parents[index] != index {
        parents[index] = parents[parents[index]];
        index = parents[index];
    }
    index
}

pub(crate) fn union_render_groups(parents: &mut [usize], first: usize, second: usize) {
    let first = find_render_group(parents, first);
    let second = find_render_group(parents, second);
    if first != second {
        parents[second] = first;
    }
}

#[expect(clippy::cast_precision_loss)]
pub(crate) fn append_merged_block_cuboids(
    blocks: &[(usize, CuboidSpec)],
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let mut groups: Vec<(usize, MaterialAppearance, HashMap<[i32; 3], CuboidSpec>)> = Vec::new();
    for &(rigid_group, spec) in blocks {
        let group = groups.iter_mut().find(|(other_group, appearance, _)| {
            *other_group == rigid_group && *appearance == spec.appearance
        });
        let cells = if let Some((_, _, cells)) = group {
            cells
        } else {
            groups.push((rigid_group, spec.appearance, HashMap::new()));
            &mut groups.last_mut().expect("block group was appended").2
        };
        cells.insert(spec.pose.translation_position_ticks().to_array(), spec);
    }

    for (_, appearance, mut cells) in groups {
        let mut starts = cells.keys().copied().collect::<Vec<_>>();
        starts.sort_unstable();
        for start in starts {
            if !cells.contains_key(&start) {
                continue;
            }
            let mut counts = [1_i32; 3];
            while cells.contains_key(&[
                start[0] + counts[0] * POSITION_TICKS_PER_GRID_UNIT,
                start[1],
                start[2],
            ]) {
                counts[0] += 1;
            }
            'grow_y: loop {
                for x in 0..counts[0] {
                    if !cells.contains_key(&[
                        start[0] + x * POSITION_TICKS_PER_GRID_UNIT,
                        start[1] + counts[1] * POSITION_TICKS_PER_GRID_UNIT,
                        start[2],
                    ]) {
                        break 'grow_y;
                    }
                }
                counts[1] += 1;
            }
            'grow_z: loop {
                for x in 0..counts[0] {
                    for y in 0..counts[1] {
                        if !cells.contains_key(&[
                            start[0] + x * POSITION_TICKS_PER_GRID_UNIT,
                            start[1] + y * POSITION_TICKS_PER_GRID_UNIT,
                            start[2] + counts[2] * POSITION_TICKS_PER_GRID_UNIT,
                        ]) {
                            break 'grow_z;
                        }
                    }
                }
                counts[2] += 1;
            }
            for x in 0..counts[0] {
                for y in 0..counts[1] {
                    for z in 0..counts[2] {
                        cells.remove(&[
                            start[0] + x * POSITION_TICKS_PER_GRID_UNIT,
                            start[1] + y * POSITION_TICKS_PER_GRID_UNIT,
                            start[2] + z * POSITION_TICKS_PER_GRID_UNIT,
                        ]);
                    }
                }
            }

            let first_vertex = positions.len();
            let block_counts = Vec3::new(counts[0] as f32, counts[1] as f32, counts[2] as f32);
            let center = IVec3::from_array(start).as_vec3() * mechanic_core::POSITION_TICK_METERS
                + (block_counts - Vec3::ONE) * (BLOCK_SIZE_METERS * 0.5);
            append_transformed_cuboid(
                center,
                Quat::IDENTITY,
                block_counts * BLOCK_SIZE_METERS,
                positions,
                normals,
                indices,
            );
            append_cuboid_texture_coordinates(first_vertex, positions, normals, uvs, tangents);
            colors.extend(std::iter::repeat_n(
                chroma::encode_appearance(appearance),
                positions.len() - first_vertex,
            ));
        }
    }
}

pub(crate) fn append_cuboid_texture_coordinates(
    first: usize,
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
) {
    for (&position, &normal) in positions[first..].iter().zip(&normals[first..]) {
        let position = Vec3::from_array(position);
        let normal = Vec3::from_array(normal);
        let absolute = normal.abs();
        let (uv, tangent) = if absolute.y >= absolute.x && absolute.y >= absolute.z {
            ([position.x, position.z], Vec3::X)
        } else if absolute.x >= absolute.z {
            ([position.z, position.y], Vec3::Z)
        } else {
            ([position.x, position.y], Vec3::X)
        };
        uvs.push(uv.map(|value| value / MATERIAL_TEXTURE_METERS_PER_REPEAT));
        tangents.push([tangent.x, tangent.y, tangent.z, 1.0]);
    }
}

/// Every material an ordinary part draws with: its core and each layer.
pub(crate) fn ordinary_materials(spec: PartSpec) -> impl Iterator<Item = ConstructionMaterial> {
    ordinary_material(spec)
        .into_iter()
        .chain(spec.material_layers().iter().map(|layer| layer.material))
}

/// Draws a layered part band by band into `material`'s mesh from its
/// evaluated solid, colouring each band with its own appearance.
#[expect(clippy::too_many_arguments)]
pub(crate) fn append_layered_part(
    graph: &ConstructionGraph,
    placement: BuildTransform,
    part: PartId,
    spec: PartSpec,
    material: Option<ConstructionMaterial>,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let solid = graph
        .evaluated_solid_shared(mechanic_core::SolidOwner::Part(part))
        .expect("committed layer geometry evaluates");
    for band in (0..=spec.material_layers().len()).filter_map(|band| u8::try_from(band).ok()) {
        let Some((band_material, appearance)) = spec.band(band) else {
            continue;
        };
        if material.is_some_and(|wanted| wanted != band_material) {
            continue;
        }
        let first_vertex = positions.len();
        append_evaluated_band(
            &solid,
            placement,
            Some(band),
            positions,
            normals,
            uvs,
            tangents,
            indices,
        );
        colors.extend(std::iter::repeat_n(
            chroma::encode_appearance(appearance),
            positions.len() - first_vertex,
        ));
    }
}

pub(crate) const fn ordinary_material(spec: PartSpec) -> Option<ConstructionMaterial> {
    match spec {
        PartSpec::Cuboid(cuboid) => Some(cuboid.material),
        PartSpec::Cylinder(cylinder) => Some(cylinder.material),
        PartSpec::PipeBend(bend) => Some(bend.material),
        PartSpec::PipeJunction(junction) => Some(junction.material),
        PartSpec::Controller(_)
        | PartSpec::Engine(_)
        | PartSpec::Transmission(_)
        | PartSpec::Servo(_)
        | PartSpec::Seat(_)
        | PartSpec::Dial(_)
        | PartSpec::Button(_)
        | PartSpec::Input(_)
        | PartSpec::DimensionLink(_) => None,
    }
}

/// Control blocks render as their own teal mesh so they stand out from the
/// construction they steer.
#[cfg(test)]
pub(crate) fn combined_controller_mesh(graph: &ConstructionGraph) -> Mesh {
    combined_authored_construction_mesh(graph, AuthoredPart::Controller, None)
}

pub(crate) fn combined_authored_construction_mesh(
    graph: &ConstructionGraph,
    appearance: AuthoredPart,
    active_dimension_link: Option<mechanic_core::DimensionLinkId>,
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    let mut indices = Vec::new();
    for (part, spec) in graph
        .parts()
        .filter(|(part, spec)| appearance.matches(graph, *part, **spec, active_dimension_link))
    {
        let cuboid = spec
            .as_cuboid()
            .expect("authored machine appearances have cuboid envelopes");
        append_authored_cuboid(
            graph.part_position(part).expect("part exists"),
            graph.part_rotation(part).expect("part exists"),
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

pub(crate) fn combined_parts_mesh_scaled(specs: &[PartSpec], scale_factor: f32) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    for &spec in specs {
        append_part(
            spec,
            scale_factor,
            &mut positions,
            &mut normals,
            &mut indices,
        );
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

#[expect(clippy::too_many_lines, reason = "one arm per part kind")]
pub(crate) fn append_part(
    spec: PartSpec,
    scale_factor: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    match spec {
        PartSpec::Cuboid(spec) if spec.rack().is_some() => super::gear::append_rack_cuboid(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            spec,
            scale_factor,
            positions,
            normals,
            indices,
        ),
        PartSpec::Cuboid(spec) => append_transformed_cuboid(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            spec.size_meters() * scale_factor,
            positions,
            normals,
            indices,
        ),
        PartSpec::Controller(spec) => append_transformed_cuboid(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            spec.cuboid().size_meters() * scale_factor,
            positions,
            normals,
            indices,
        ),
        PartSpec::Engine(spec) => append_transformed_cuboid(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            spec.cuboid().size_meters() * scale_factor,
            positions,
            normals,
            indices,
        ),
        PartSpec::Transmission(spec) => append_transformed_cuboid(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            spec.cuboid().size_meters() * scale_factor,
            positions,
            normals,
            indices,
        ),
        PartSpec::Servo(spec) => append_transformed_cuboid(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            spec.cuboid().size_meters() * scale_factor,
            positions,
            normals,
            indices,
        ),
        PartSpec::Seat(spec) => append_transformed_cuboid(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            spec.cuboid().size_meters() * scale_factor,
            positions,
            normals,
            indices,
        ),
        spec @ (PartSpec::Dial(_) | PartSpec::Button(_)) => append_transformed_cuboid(
            spec.pose().translation(),
            spec.pose().rotation.quaternion(),
            spec.size_meters() * scale_factor,
            positions,
            normals,
            indices,
        ),
        PartSpec::Input(spec) => append_transformed_cuboid(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            spec.cuboid().size_meters() * scale_factor,
            positions,
            normals,
            indices,
        ),
        PartSpec::DimensionLink(spec) => append_transformed_cuboid(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            spec.cuboid().size_meters() * scale_factor,
            positions,
            normals,
            indices,
        ),
        PartSpec::Cylinder(spec) => {
            append_cylinder_part(spec, scale_factor, positions, normals, indices);
        }
        PartSpec::PipeJunction(junction) => append_pipe_junction_shape(
            junction.pose.translation(),
            junction.pose.rotation.quaternion(),
            junction,
            scale_factor,
            positions,
            normals,
            indices,
        ),
        PartSpec::PipeBend(spec) => append_pipe_bend_shape(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            spec.dimensions,
            scale_factor,
            positions,
            normals,
            indices,
        ),
    }
}

// A cylinder as drawn: plain, or with the spiral or teeth cut into its walls.
fn append_cylinder_part(
    spec: mechanic_core::CylinderSpec,
    scale_factor: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let (translation, rotation) = (spec.pose.translation(), spec.pose.rotation.quaternion());
    if spec.gear().is_some() {
        super::gear::append_gear_cylinder(
            translation,
            rotation,
            spec,
            scale_factor,
            positions,
            normals,
            indices,
        );
    } else if spec.spiral().is_some() {
        super::spiral::append_spiral_cylinder(
            translation,
            rotation,
            spec,
            scale_factor,
            positions,
            normals,
            indices,
        );
    } else {
        append_cylinder_shape(
            translation,
            rotation,
            spec.dimensions,
            scale_factor,
            positions,
            normals,
            indices,
        );
    }
}

pub(crate) fn single_cylinder_mesh(dimensions: CylinderDimensions) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    append_cylinder_shape(
        Vec3::ZERO,
        Quat::IDENTITY,
        dimensions,
        1.0,
        &mut positions,
        &mut normals,
        &mut indices,
    );
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

pub(crate) fn append_transformed_cuboid(
    translation: Vec3,
    rotation: Quat,
    size: Vec3,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let base_index = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");
    positions.extend(
        CUBE_POSITIONS.map(|position| {
            (translation + rotation * (Vec3::from_array(position) * size)).to_array()
        }),
    );
    normals.extend(CUBE_NORMALS.map(|normal| (rotation * Vec3::from_array(normal)).to_array()));
    indices.extend(CUBE_INDICES.map(|index| base_index + index));
}

/// Maps build-space geometry into the space a mesh is being drawn in.
///
/// The construction view draws parts where they were authored; the simulation
/// view draws them where their compound has moved to. Shaped geometry is
/// generated in build space either way, so it needs this to follow a body.
#[derive(Clone, Copy)]
pub(crate) struct BuildTransform {
    /// Build-space point that maps onto `translation`.
    pub(crate) origin: Vec3,
    /// Rotation applied about `origin`.
    pub(crate) rotation: Quat,
    /// Where `origin` ends up.
    pub(crate) translation: Vec3,
}

impl BuildTransform {
    /// Draws build-space geometry exactly where it was authored.
    pub(crate) const IDENTITY: Self = Self {
        origin: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        translation: Vec3::ZERO,
    };

    pub(crate) fn with_frame(self, frame: mechanic_core::ConstructionFrame) -> Self {
        Self {
            origin: Vec3::ZERO,
            rotation: self.rotation * frame.rotation(),
            translation: self.point(frame.translation()),
        }
    }

    pub(crate) fn point(self, point: Vec3) -> Vec3 {
        self.translation + self.rotation * (point - self.origin)
    }

    pub(crate) fn direction(self, direction: Vec3) -> Vec3 {
        self.rotation * direction
    }
}

/// Emits the exterior surface of a shaped region, with UVs and tangents.
///
/// The geometry comes from the same decomposition the colliders come from, so
/// what is drawn and what is collided against cannot drift apart. Faces
/// interior to the region are dropped: a cell face whose neighbour is also part
/// of the region is inside the solid, and a piece face with no grid provenance
/// is interior to its own cell.
pub(crate) fn append_region(
    region: &ShapeRegion,
    placement: BuildTransform,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let first = positions.len();
    append_region_surface(region, placement, positions, normals, indices);
    // The same triplanar projection ordinary blocks use, so a shaped face keeps
    // the material's scale.
    for (&position, &normal) in positions[first..].iter().zip(&normals[first..]) {
        let position = Vec3::from_array(position);
        let normal = Vec3::from_array(normal);
        let absolute = normal.abs();
        let (uv, tangent) = if absolute.y >= absolute.x && absolute.y >= absolute.z {
            ([position.x, position.z], Vec3::X)
        } else if absolute.x >= absolute.z {
            ([position.z, position.y], Vec3::Z)
        } else {
            ([position.x, position.y], Vec3::X)
        };
        uvs.push(uv.map(|value| value / MATERIAL_TEXTURE_METERS_PER_REPEAT));
        tangents.push([tangent.x, tangent.y, tangent.z, 1.0]);
    }
}

/// Emits an evaluated feature boundary. Tessellation seams are retained only
/// as triangle edges; surface provenance controls hard versus smooth normals,
/// while the ordinary triplanar projection preserves material scale.
pub(crate) fn append_evaluated_solid(
    solid: &mechanic_core::EvaluatedSolid,
    placement: BuildTransform,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    append_evaluated_band(
        solid, placement, None, positions, normals, uvs, tangents, indices,
    );
}

/// Averages incident face normals once per vertex and smoothing group.
/// Hard faces and coincident vertices in other groups never contribute.
pub(crate) fn evaluated_smooth_normals(
    solid: &mechanic_core::EvaluatedSolid,
) -> HashMap<(u32, u32), Vec3> {
    let mut normals = HashMap::new();
    let mut face_vertices = HashSet::new();
    for surface in &solid.surfaces {
        if surface.smoothing_group == 0 {
            continue;
        }
        face_vertices.clear();
        let mut edge = surface.half_edge;
        loop {
            let half_edge = solid.half_edges[edge as usize];
            // A face contributes once at each vertex, even if its loop visits
            // that vertex twice. Keep surface order for stable floating-point sums.
            if face_vertices.insert(half_edge.origin) {
                *normals
                    .entry((surface.smoothing_group, half_edge.origin))
                    .or_insert(Vec3::ZERO) += surface.normal;
            }
            edge = half_edge.next;
            if edge == surface.half_edge {
                break;
            }
        }
    }
    for normal in normals.values_mut() {
        *normal = normal.normalize_or_zero();
    }
    normals
}

/// Emits the evaluated boundary of one material band, or every band.
#[expect(clippy::too_many_arguments)]
pub(crate) fn append_evaluated_band(
    solid: &mechanic_core::EvaluatedSolid,
    placement: BuildTransform,
    band: Option<u8>,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let smooth_normals = evaluated_smooth_normals(solid);
    let projection_normals = solid
        .surfaces
        .iter()
        .fold(HashMap::new(), |mut normals, surface| {
            normals.entry(surface.key).or_insert(surface.normal);
            normals
        });
    for surface in &solid.surfaces {
        let mut loop_edges = Vec::new();
        let mut edge = surface.half_edge;
        loop {
            loop_edges.push(edge);
            edge = solid.half_edges[edge as usize].next;
            if edge == surface.half_edge {
                break;
            }
        }
        if loop_edges.len() < 3 || band.is_some_and(|band| band != surface.band) {
            continue;
        }
        let base = u32::try_from(positions.len()).expect("construction mesh fits 32-bit indices");
        for &half_edge in &loop_edges {
            let vertex_index = solid.half_edges[half_edge as usize].origin;
            let position = solid.vertices[vertex_index as usize].position;
            let normal = if surface.smoothing_group == 0 {
                surface.normal
            } else {
                smooth_normals[&(surface.smoothing_group, vertex_index)]
            };
            let world_position = placement.point(position);
            let world_normal = placement.direction(normal).normalize_or_zero();
            // A rounded patch keeps the projection of its originating face.
            // Choosing from the smoothed vertex normal would change dominant
            // axes halfway through a 90-degree fillet and rotate the material.
            let projection_normal = projection_normals
                .get(&surface.uv_provenance)
                .copied()
                .unwrap_or(surface.normal);
            let absolute = projection_normal.abs();
            let (uv, tangent) = if absolute.y >= absolute.x && absolute.y >= absolute.z {
                ([position.x, position.z], Vec3::X)
            } else if absolute.x >= absolute.z {
                ([position.z, position.y], Vec3::Z)
            } else {
                ([position.x, position.y], Vec3::X)
            };
            positions.push(world_position.to_array());
            normals.push(world_normal.to_array());
            uvs.push(uv.map(|value| value / MATERIAL_TEXTURE_METERS_PER_REPEAT));
            let tangent = placement.direction(tangent);
            tangents.push([tangent.x, tangent.y, tangent.z, 1.0]);
        }
        for step in 1..loop_edges.len() - 1 {
            let step = u32::try_from(step).expect("surface polygons fit u32");
            indices.extend([base, base + step, base + step + 1]);
        }
    }
}

/// Emits just the positions, normals, and indices of a region's surface.
pub(crate) fn append_region_surface(
    region: &ShapeRegion,
    placement: BuildTransform,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let grid = region.grid();
    for piece in builder::region_pieces(region) {
        match piece {
            PartPiece::Cuboid {
                center,
                half_extents,
                cell_min,
                cell_span,
                ..
            } => {
                for (axis, sign) in (0..3).flat_map(|axis| [(axis, 1_i32), (axis, -1_i32)]) {
                    if box_face_is_interior(&grid, cell_min, cell_span, axis, sign) {
                        continue;
                    }
                    append_axis_quad(
                        center,
                        half_extents,
                        axis,
                        sign,
                        placement,
                        positions,
                        normals,
                        indices,
                    );
                }
            }
            PartPiece::Convex(convex) => {
                for face in &convex.faces {
                    let Some(gridface) = face.grid_face else {
                        continue;
                    };
                    if grid.contains(gridface.cell + face_neighbour_offset(gridface.face)) {
                        continue;
                    }
                    let base = u32::try_from(positions.len())
                        .expect("construction mesh fits 32-bit indices");
                    for &index in &face.indices {
                        positions.push(placement.point(convex.vertices[index as usize]).to_array());
                        normals.push(placement.direction(face.normal).to_array());
                    }
                    for step in 1..face.indices.len() - 1 {
                        let step = u32::try_from(step).expect("a piece face has few vertices");
                        indices.extend([base, base + step, base + step + 1]);
                    }
                }
            }
        }
    }
}

/// Whether every cell across one side of a box-cover box is still inside the
/// part, which makes that whole side interior geometry.
pub(crate) fn box_face_is_interior(
    grid: &CellGrid,
    cell_min: IVec3,
    cell_span: IVec3,
    axis: usize,
    sign: i32,
) -> bool {
    let mut neighbour = cell_min;
    neighbour[axis] += if sign > 0 { cell_span[axis] } else { -1 };
    let tangents = [(axis + 1) % 3, (axis + 2) % 3];
    (0..cell_span[tangents[0]]).all(|first| {
        (0..cell_span[tangents[1]]).all(|second| {
            let mut cell = neighbour;
            cell[tangents[0]] += first;
            cell[tangents[1]] += second;
            grid.contains(cell)
        })
    })
}

/// Emits one axis-aligned face of a box, wound outward.
#[expect(clippy::too_many_arguments)]
pub(crate) fn append_axis_quad(
    center: Vec3,
    half_extents: Vec3,
    axis: usize,
    sign: i32,
    placement: BuildTransform,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let mut normal = Vec3::ZERO;
    normal[axis] = if sign > 0 { 1.0 } else { -1.0 };
    let tangents = [(axis + 1) % 3, (axis + 2) % 3];
    let mut first = Vec3::ZERO;
    first[tangents[0]] = half_extents[tangents[0]];
    let mut second = Vec3::ZERO;
    second[tangents[1]] = half_extents[tangents[1]];
    // Flip the winding on negative faces so every quad faces outward.
    if sign < 0 {
        core::mem::swap(&mut first, &mut second);
    }
    let anchor = center + normal * half_extents[axis];
    let corners = [
        anchor - first - second,
        anchor + first - second,
        anchor + first + second,
        anchor - first + second,
    ];
    let base = u32::try_from(positions.len()).expect("construction mesh fits 32-bit indices");
    let world_normal = placement.direction(normal).to_array();
    for corner in corners {
        positions.push(placement.point(corner).to_array());
        normals.push(world_normal);
    }
    indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
}

pub(crate) const MATERIAL_TEXTURE_PIXELS_PER_SIDE: f32 = 3_072.0;

pub(crate) const MATERIAL_TEXTURE_PIXELS_PER_BLOCK: f32 = 512.0;

pub(crate) const MATERIAL_TEXTURE_METERS_PER_REPEAT: f32 =
    GRID_UNIT_METERS * MATERIAL_TEXTURE_PIXELS_PER_SIDE / MATERIAL_TEXTURE_PIXELS_PER_BLOCK;

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "one dispatch per ordinary part kind, geometry then texture"
)]
pub(crate) fn append_textured_part(
    spec: PartSpec,
    translation: Vec3,
    rotation: Quat,
    placement: BuildTransform,
    texture_offset: PipeTextureOffset,
    end_faces: PipeEndFaces,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let _ = placement;
    let first = positions.len();
    match spec {
        PartSpec::Cuboid(cuboid) if cuboid.rack().is_some() => super::gear::append_rack_cuboid(
            translation,
            rotation,
            cuboid,
            1.0,
            positions,
            normals,
            indices,
        ),
        PartSpec::Cuboid(cuboid) => append_transformed_cuboid(
            translation,
            rotation,
            cuboid.size_meters(),
            positions,
            normals,
            indices,
        ),
        PartSpec::Cylinder(cylinder) if cylinder.gear().is_some() => {
            super::gear::append_gear_cylinder(
                translation,
                rotation,
                cylinder,
                1.0,
                positions,
                normals,
                indices,
            );
        }
        PartSpec::Cylinder(cylinder) if cylinder.spiral().is_some() => {
            super::spiral::append_spiral_cylinder(
                translation,
                rotation,
                cylinder,
                1.0,
                positions,
                normals,
                indices,
            );
        }
        PartSpec::Cylinder(cylinder) => append_cylinder_shape_with_end_faces(
            translation,
            rotation,
            cylinder.dimensions,
            1.0,
            end_faces,
            texture_offset.v_angle,
            positions,
            normals,
            indices,
        ),
        PartSpec::PipeBend(bend) => append_pipe_bend_shape_with_end_faces(
            translation,
            rotation,
            bend.dimensions,
            1.0,
            end_faces,
            texture_offset.v_angle,
            positions,
            normals,
            indices,
        ),
        PartSpec::PipeJunction(junction) => append_pipe_junction_shape(
            translation,
            rotation,
            junction,
            1.0,
            positions,
            normals,
            indices,
        ),
        PartSpec::Controller(_)
        | PartSpec::Engine(_)
        | PartSpec::Transmission(_)
        | PartSpec::Servo(_)
        | PartSpec::Seat(_)
        | PartSpec::Dial(_)
        | PartSpec::Button(_)
        | PartSpec::Input(_)
        | PartSpec::DimensionLink(_) => {
            unreachable!("authored parts render in their own texture batches")
        }
    }

    match spec {
        // Junctions take block projection: their crossing arms have no single
        // pipe direction to wrap a texture around.
        PartSpec::Cuboid(_) | PartSpec::PipeJunction(_) => {
            let frame_rotation = rotation * spec.pose().rotation.quaternion().conjugate();
            let frame_translation = translation - frame_rotation * spec.pose().translation();
            for (&position, &normal) in positions[first..].iter().zip(&normals[first..]) {
                let local_position =
                    frame_rotation.conjugate() * (Vec3::from_array(position) - frame_translation);
                let local_normal = frame_rotation.conjugate() * Vec3::from_array(normal);
                let absolute = local_normal.abs();
                let (uv, tangent) = if absolute.y >= absolute.x && absolute.y >= absolute.z {
                    ([local_position.x, local_position.z], Vec3::X)
                } else if absolute.x >= absolute.z {
                    ([local_position.z, local_position.y], Vec3::Z)
                } else {
                    ([local_position.x, local_position.y], Vec3::X)
                };
                uvs.push(uv.map(|value| value / MATERIAL_TEXTURE_METERS_PER_REPEAT));
                let tangent = frame_rotation * tangent;
                tangents.push([tangent.x, tangent.y, tangent.z, 1.0]);
            }
        }
        PartSpec::Cylinder(_) => {
            append_cylinder_texture_coordinates(
                translation,
                rotation,
                first,
                positions,
                normals,
                texture_offset.v_angle,
                uvs,
                tangents,
            );
        }
        PartSpec::PipeBend(bend) => {
            append_pipe_bend_texture_coordinates(
                bend.dimensions,
                translation,
                rotation,
                first,
                positions,
                normals,
                texture_offset.v_angle,
                uvs,
                tangents,
            );
        }
        PartSpec::Controller(_)
        | PartSpec::Engine(_)
        | PartSpec::Transmission(_)
        | PartSpec::Servo(_)
        | PartSpec::Seat(_)
        | PartSpec::Dial(_)
        | PartSpec::Button(_)
        | PartSpec::Input(_)
        | PartSpec::DimensionLink(_) => unreachable!(),
    }
    for uv in &mut uvs[first..] {
        uv[0] += texture_offset.u / MATERIAL_TEXTURE_METERS_PER_REPEAT;
    }
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn append_cylinder_texture_coordinates(
    translation: Vec3,
    rotation: Quat,
    first: usize,
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    v_angle_offset: f32,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
) {
    let inverse = rotation.inverse();
    // Full pipe rings start at the transported UV cut. Seeding the unwrap at
    // that exact angle disambiguates atan2's equivalent -PI/+PI result.
    let mut outer_angle = Some(-v_angle_offset);
    let mut inner_angle = Some(-v_angle_offset);
    for (&position, &normal) in positions[first..].iter().zip(&normals[first..]) {
        let local = inverse * (Vec3::from_array(position) - translation);
        let local_normal = inverse * Vec3::from_array(normal);
        let radial = Vec3::new(local.x, 0.0, local.z);
        let radial_direction = radial.normalize_or_zero();
        let (uv, local_tangent, local_bitangent) = if local_normal.y.abs() > 0.9 {
            ([local.x, local.z], Vec3::X, Vec3::Z)
        } else if radial_direction != Vec3::ZERO && local_normal.dot(radial_direction).abs() > 0.5 {
            let mut angle = local.z.atan2(local.x);
            let previous = if local_normal.dot(radial_direction) > 0.0 {
                &mut outer_angle
            } else {
                &mut inner_angle
            };
            if let Some(previous) = *previous {
                while angle - previous > std::f32::consts::PI {
                    angle -= std::f32::consts::TAU;
                }
                while angle - previous < -std::f32::consts::PI {
                    angle += std::f32::consts::TAU;
                }
            }
            *previous = Some(angle);
            (
                [local.y, (angle + v_angle_offset) * radial.length()],
                Vec3::Y,
                Vec3::new(-angle.sin(), 0.0, angle.cos()),
            )
        } else {
            ([local.y, radial.length()], Vec3::Y, radial_direction)
        };
        uvs.push(uv.map(|value| value / MATERIAL_TEXTURE_METERS_PER_REPEAT));
        let tangent = rotation * local_tangent;
        let bitangent = rotation * local_bitangent;
        let normal = Vec3::from_array(normal);
        let handedness = if tangent.cross(normal).dot(bitangent) < 0.0 {
            -1.0
        } else {
            1.0
        };
        tangents.push([tangent.x, tangent.y, tangent.z, handedness]);
    }
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn append_pipe_bend_texture_coordinates(
    dimensions: PipeBendDimensions,
    translation: Vec3,
    rotation: Quat,
    first: usize,
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    v_angle_offset: f32,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
) {
    const ARC_SLICES: usize = 12;
    const RADIAL_SIDES: usize = 24;
    const CURVED_VERTICES_PER_SECTOR: usize = 8;
    let inverse = rotation.inverse();
    let radius = dimensions.radius();
    for (vertex, (&position, &normal)) in
        positions[first..].iter().zip(&normals[first..]).enumerate()
    {
        let local = inverse * (Vec3::from_array(position) - translation);
        let from_curve_center = Vec2::new(local.x + radius, local.y - radius);
        let theta = from_curve_center.y.atan2(from_curve_center.x);
        let cross_x = from_curve_center.length() - radius;
        let phi = if vertex < ARC_SLICES * RADIAL_SIDES * CURVED_VERTICES_PER_SECTOR {
            let side = (vertex / CURVED_VERTICES_PER_SECTOR) % RADIAL_SIDES;
            let within_sector = vertex % CURVED_VERTICES_PER_SECTOR;
            let boundary = if matches!(within_sector, 0 | 1 | 6 | 7) {
                side
            } else {
                side + 1
            };
            std::f32::consts::TAU * f32::from(u16::try_from(boundary).unwrap())
                / f32::from(u16::try_from(RADIAL_SIDES).unwrap())
                - v_angle_offset
        } else {
            local.z.atan2(cross_x)
        };
        let surface_radius = cross_x.hypot(local.z);
        uvs.push([
            theta * radius / MATERIAL_TEXTURE_METERS_PER_REPEAT,
            (phi + v_angle_offset) * surface_radius / MATERIAL_TEXTURE_METERS_PER_REPEAT,
        ]);
        let tangent = rotation * Vec3::new(-theta.sin(), theta.cos(), 0.0);
        let radial = Vec3::new(theta.cos(), theta.sin(), 0.0);
        let bitangent = rotation * (-radial * phi.sin() + Vec3::Z * phi.cos());
        let normal = Vec3::from_array(normal);
        let handedness = if tangent.cross(normal).dot(bitangent) < 0.0 {
            -1.0
        } else {
            1.0
        };
        tangents.push([tangent.x, tangent.y, tangent.z, handedness]);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "appends every vertex stream of one authored cuboid"
)]
pub(crate) fn append_authored_cuboid(
    translation: Vec3,
    rotation: Quat,
    size: Vec3,
    appearance: AuthoredPart,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let base_index = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");
    positions.extend(
        AUTHORED_CUBE_POSITIONS.map(|position| {
            (translation + rotation * (Vec3::from_array(position) * size)).to_array()
        }),
    );
    normals.extend(
        AUTHORED_CUBE_NORMALS.map(|normal| (rotation * Vec3::from_array(normal)).to_array()),
    );
    uvs.extend(authored_uvs(appearance));
    tangents.extend(AUTHORED_CUBE_TANGENTS.map(|tangent| {
        let tangent_xyz = rotation * Vec3::from_array(tangent[..3].try_into().unwrap());
        [tangent_xyz.x, tangent_xyz.y, tangent_xyz.z, tangent[3]]
    }));
    indices.extend(AUTHORED_CUBE_INDICES.map(|index| base_index + index));
}
