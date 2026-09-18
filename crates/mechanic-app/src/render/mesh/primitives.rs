//! Triangle and quad assembly shared by every mesh builder.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::Indices;
use bevy::prelude::{Mesh, Vec3, vec};
use bevy::render::render_resource::PrimitiveTopology;

/// A single zero-area triangle: nothing to see, but still vertex data.
///
/// Overlays that come and go stay visible and swap to this instead of hiding,
/// because a hidden mesh has no slab allocation and writing to one makes the
/// renderer log a use-after-free.
pub(crate) fn degenerate_overlay_mesh() -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    append_mesh_triangle(
        [Vec3::ZERO; 3],
        Vec3::Y,
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

/// Keeps an empty logical batch allocated in Bevy's GPU mesh slabs.
///
/// Bevy 0.19 frees a modified mesh's previous allocation before discovering
/// that a zero-vertex replacement needs no new allocation, then still attempts
/// to upload its vertex and index data. A zero-area triangle is invisible but
/// preserves both allocations and avoids that renderer use-after-free path.
pub(crate) fn renderable_mesh(mesh: Mesh) -> Mesh {
    if mesh.count_vertices() == 0 {
        if mesh.attribute(Mesh::ATTRIBUTE_UV_0).is_some() {
            degenerate_textured_mesh()
        } else {
            degenerate_overlay_mesh()
        }
    } else {
        mesh
    }
}

pub(crate) fn degenerate_textured_mesh() -> Mesh {
    degenerate_overlay_mesh()
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; 3])
        .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, vec![[1.0, 0.0, 0.0, 1.0]; 3])
}

pub(crate) fn append_mesh_triangle(
    vertices: [Vec3; 3],
    normal: Vec3,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let base = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");
    positions.extend(vertices.map(|vertex| vertex.to_array()));
    normals.extend([normal.to_array(); 3]);
    indices.extend([base, base + 1, base + 2]);
}

pub(crate) fn append_mesh_quad(
    vertices: [Vec3; 4],
    normal: Vec3,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let base = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");
    positions.extend(vertices.map(|vertex| vertex.to_array()));
    normals.extend([normal.to_array(); 4]);
    indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
}

pub(crate) fn append_mesh_quad_with_normals(
    vertices: [Vec3; 4],
    vertex_normals: [Vec3; 4],
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let base = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");
    positions.extend(vertices.map(|vertex| vertex.to_array()));
    normals.extend(vertex_normals.map(|normal| normal.to_array()));
    indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
}
