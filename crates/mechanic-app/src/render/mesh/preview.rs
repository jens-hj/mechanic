//! Placement, deletion, and layer preview meshes.

#[cfg(test)]
use crate::DELETE_PREVIEW_SCALE;
use crate::builder::{BlockVolume, part_world_bounds};
use crate::render::mesh::construction::{
    CUBE_POSITIONS, append_transformed_cuboid, combined_parts_mesh_scaled,
};
use crate::{BLOCK_SHEET_PREVIEW_INSET_METERS, CUBE_INDICES, CUBE_NORMALS};
use bevy::asset::RenderAssetUsages;
use bevy::mesh::Indices;
use bevy::prelude::{Mesh, Quat, Vec3, vec};
use bevy::render::render_resource::PrimitiveTopology;
#[cfg(test)]
use mechanic_core::CuboidSpec;
use mechanic_core::PartSpec;

#[cfg(test)]
pub(crate) fn delete_preview_mesh(specs: &[PartSpec]) -> Mesh {
    combined_parts_mesh_scaled(specs, DELETE_PREVIEW_SCALE)
}

/// Exact world bounds of a block sheet, including the outer half-block skin.
#[cfg(test)]
pub(crate) fn block_sheet_bounds(specs: &[CuboidSpec]) -> Option<(Vec3, Vec3)> {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for &spec in specs {
        let (part_minimum, part_maximum) = part_world_bounds(PartSpec::Cuboid(spec));
        minimum = minimum.min(part_minimum);
        maximum = maximum.max(part_maximum);
    }
    minimum.is_finite().then_some((minimum, maximum))
}

/// One exterior cuboid spanning the live sheet preview, inset just enough to
/// keep its contact faces from being coplanar with opaque construction.
///
/// Built directly from bounds because a valid 4,096-block sheet can be wider
/// than the construction API's per-cuboid dimension limit.
#[cfg(test)]
pub(crate) fn block_sheet_preview_mesh(specs: &[CuboidSpec]) -> Mesh {
    let (minimum, maximum) = block_sheet_bounds(specs).expect("a block drag contains a block");
    block_bounds_preview_mesh(minimum, maximum)
}

pub(crate) fn block_volume_preview_mesh(volume: BlockVolume) -> Mesh {
    let (minimum, maximum) = volume.bounds();
    block_bounds_preview_mesh(minimum, maximum)
}

pub(crate) fn block_bounds_preview_mesh(minimum: Vec3, maximum: Vec3) -> Mesh {
    let visual_minimum = minimum + Vec3::splat(BLOCK_SHEET_PREVIEW_INSET_METERS);
    let visual_maximum = maximum - Vec3::splat(BLOCK_SHEET_PREVIEW_INSET_METERS);
    let mut positions = Vec::with_capacity(CUBE_POSITIONS.len());
    let mut normals = Vec::with_capacity(CUBE_NORMALS.len());
    let mut indices = Vec::with_capacity(CUBE_INDICES.len());
    append_transformed_cuboid(
        (visual_minimum + visual_maximum) * 0.5,
        Quat::IDENTITY,
        visual_maximum - visual_minimum,
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

/// Ghost of a layer edit: the exterior skin of the layered parts as one shape.
///
/// A flat surface spreads the layer over many blocks, and drawing each block
/// would show every wall between them through the translucent ghost. Faces
/// that touch another member are dropped, so it reads like a block-sheet drag.
/// Parts that are not axis-aligned boxes keep their own full shells.
pub(crate) fn layer_preview_mesh(specs: &[PartSpec]) -> Mesh {
    let boxes = specs
        .iter()
        .map(|&spec| {
            let cuboid = spec.as_cuboid()?;
            let rotation = cuboid.pose.rotation.quaternion();
            [Vec3::X, Vec3::Y, Vec3::Z]
                .into_iter()
                .all(|axis| (rotation * axis).abs().max_element() > 1.0 - 1e-4)
                .then(|| part_world_bounds(spec))
        })
        .collect::<Option<Vec<_>>>();
    let Some(boxes) = boxes else {
        return combined_parts_mesh_scaled(specs, 1.004);
    };
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    for (index, &(minimum, maximum)) in boxes.iter().enumerate() {
        for axis in 0..3 {
            for positive in [false, true] {
                let plane = if positive {
                    maximum[axis]
                } else {
                    minimum[axis]
                };
                let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                let mut pieces = vec![([minimum[u], minimum[v]], [maximum[u], maximum[v]])];
                for (other, &(other_minimum, other_maximum)) in boxes.iter().enumerate() {
                    // A neighbour hides this face where it fills the space just
                    // beyond it.
                    let beyond = if positive {
                        plane + LAYER_PREVIEW_CONTACT_METERS
                    } else {
                        plane - LAYER_PREVIEW_CONTACT_METERS
                    };
                    if other == index
                        || beyond <= other_minimum[axis]
                        || beyond >= other_maximum[axis]
                    {
                        continue;
                    }
                    let cutter = (
                        [other_minimum[u], other_minimum[v]],
                        [other_maximum[u], other_maximum[v]],
                    );
                    pieces = pieces
                        .into_iter()
                        .flat_map(|piece| subtract_rectangle(piece, cutter))
                        .collect();
                }
                let normal = if positive { 1.0 } else { -1.0 };
                for (low, high) in pieces {
                    let corner = |a: f32, b: f32| {
                        let mut point = Vec3::ZERO;
                        point[axis] = plane - normal * BLOCK_SHEET_PREVIEW_INSET_METERS;
                        point[u] = a;
                        point[v] = b;
                        point.to_array()
                    };
                    let base =
                        u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");
                    positions.extend([
                        corner(low[0], low[1]),
                        corner(high[0], low[1]),
                        corner(high[0], high[1]),
                        corner(low[0], high[1]),
                    ]);
                    let mut face_normal = [0.0; 3];
                    face_normal[axis] = normal;
                    normals.extend([face_normal; 4]);
                    // (u, v, axis) is right-handed, so counter-clockwise in
                    // (u, v) faces +axis.
                    if positive {
                        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
                    } else {
                        indices.extend([base, base + 2, base + 1, base, base + 3, base + 2]);
                    }
                }
            }
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

/// How far past a face a neighbour must reach to count as touching it.
pub(crate) const LAYER_PREVIEW_CONTACT_METERS: f32 = 1e-4;

/// The parts of `piece` outside `cutter`, as at most four rectangles.
pub(crate) fn subtract_rectangle(
    piece: ([f32; 2], [f32; 2]),
    cutter: ([f32; 2], [f32; 2]),
) -> Vec<([f32; 2], [f32; 2])> {
    let (low, high) = piece;
    let cut_low = [low[0].max(cutter.0[0]), low[1].max(cutter.0[1])];
    let cut_high = [high[0].min(cutter.1[0]), high[1].min(cutter.1[1])];
    let epsilon = LAYER_PREVIEW_CONTACT_METERS;
    if cut_high[0] - cut_low[0] <= epsilon || cut_high[1] - cut_low[1] <= epsilon {
        return vec![piece];
    }
    let mut rest = Vec::new();
    if cut_low[0] - low[0] > epsilon {
        rest.push((low, [cut_low[0], high[1]]));
    }
    if high[0] - cut_high[0] > epsilon {
        rest.push(([cut_high[0], low[1]], high));
    }
    if cut_low[1] - low[1] > epsilon {
        rest.push(([cut_low[0], low[1]], [cut_high[0], cut_low[1]]));
    }
    if high[1] - cut_high[1] > epsilon {
        rest.push(([cut_low[0], cut_high[1]], [cut_high[0], high[1]]));
    }
    rest
}
