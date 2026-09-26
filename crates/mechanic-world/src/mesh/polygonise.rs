//! Marching one lattice cube into oriented, material-weighted triangles.

use super::TerrainMeshChunk;
use super::groups::IndexGroup;
use super::lattice::LatticePoint;
use crate::transvoxel::tables::{REGULAR_CELL_CLASS, REGULAR_CELL_DATA, REGULAR_VERTEX_DATA};
use crate::{
    BRICK_EDGE_CELLS, SurfaceId, TERRAIN_CELL_METERS, TerrainFace, TerrainMaterial,
    TerrainTransitionMask, WorldCell,
};
use bevy_math::{DVec3, Vec3};

pub(super) const CUBE_CORNERS: [[i32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

#[derive(Clone, Copy)]
pub(super) struct MeshVertex {
    pub(super) position: DVec3,
    pub(super) normal: Vec3,
    pub(super) material: (TerrainMaterial, SurfaceId),
    pub(super) compaction: u8,
}

pub(super) fn polygonise_cube(
    minimum: WorldCell,
    stride: i32,
    samples: [LatticePoint; 8],
    chunk: &mut TerrainMeshChunk,
) {
    let mut positions = [DVec3::ZERO; 8];
    for (index, offset) in CUBE_CORNERS.into_iter().enumerate() {
        let cell = WorldCell::new(
            minimum.x + offset[0] * stride,
            minimum.y + offset[1] * stride,
            minimum.z + offset[2] * stride,
        );
        positions[index] = cell.centre().0 - DVec3::splat(TERRAIN_CELL_METERS * 0.5);
    }
    let case = samples
        .iter()
        .enumerate()
        .fold(0_u8, |case, (index, sample)| {
            case | if sample.sample.is_solid() {
                1 << index
            } else {
                0
            }
        });
    if case == 0 || case == u8::MAX {
        return;
    }
    let cell = REGULAR_CELL_DATA[usize::from(REGULAR_CELL_CLASS[usize::from(case)])];
    let vertex_count = usize::from(cell.geometry_counts >> 4);
    let triangle_count = usize::from(cell.geometry_counts & 0x0f);
    let mut vertices = Vec::with_capacity(vertex_count);
    for &data in &REGULAR_VERTEX_DATA[usize::from(case)][..vertex_count] {
        let edge = data & 0xff;
        let first = usize::from((edge >> 4) as u8);
        let second = usize::from((edge & 0x0f) as u8);
        let (solid, empty) = if samples[first].sample.is_solid() {
            (first, second)
        } else {
            (second, first)
        };
        let mut vertex = crossing(solid, empty, positions, samples);
        apply_transition_inset(&mut vertex, chunk);
        vertices.push(vertex);
    }
    for triangle in cell.vertex_index[..triangle_count * 3].chunks_exact(3) {
        emit_oriented_triangle(
            chunk,
            [
                vertices[usize::from(triangle[0])],
                vertices[usize::from(triangle[1])],
                vertices[usize::from(triangle[2])],
            ],
            Vec3::ZERO,
            IndexGroup::Regular,
        );
    }
}

pub(super) fn crossing(
    solid: usize,
    empty: usize,
    positions: [DVec3; 8],
    samples: [LatticePoint; 8],
) -> MeshVertex {
    let solid_density = f64::from(samples[solid].sample.density);
    let empty_density = f64::from(samples[empty].sample.density);
    let along = solid_density / (solid_density - empty_density);
    let along = along.clamp(0.0, 1.0);
    MeshVertex {
        position: positions[solid].lerp(positions[empty], along),
        normal: samples[solid]
            .normal
            .lerp(samples[empty].normal, along as f32)
            .normalize_or(Vec3::Y),
        material: crossing_material(samples[solid], samples[empty]),
        compaction: samples[solid].sample.compaction,
    }
}

pub(super) fn crossing_material(
    first: LatticePoint,
    second: LatticePoint,
) -> (TerrainMaterial, SurfaceId) {
    if let Some(authored) = first.authored.or(second.authored) {
        authored
    } else if first.sample.is_solid() {
        (second.sample.material, second.sample.surface)
    } else {
        (first.sample.material, first.sample.surface)
    }
}

pub(super) fn apply_transition_inset(vertex: &mut MeshVertex, chunk: &TerrainMeshChunk) {
    if chunk.transition_mask == TerrainTransitionMask::NONE {
        return;
    }

    let spacing = chunk.sample_spacing_metres;
    let minimum = chunk.origin.0;
    let maximum = minimum + DVec3::splat(f64::from(BRICK_EDGE_CELLS) * spacing);
    let epsilon = spacing * 1.0e-6;

    // A vertex shared with a non-transition face must keep its primary
    // position so the equal-LOD neighbor remains byte-identical.
    for face in TerrainFace::ALL {
        if !chunk.transition_mask.contains(face)
            && vertex_on_face(vertex.position, minimum, maximum, face, epsilon)
        {
            return;
        }
    }

    let mut delta = DVec3::ZERO;
    for face in TerrainFace::ALL {
        if !chunk.transition_mask.contains(face) {
            continue;
        }
        let (distance, direction) = match face {
            TerrainFace::NegativeX => (vertex.position.x - minimum.x, DVec3::X),
            TerrainFace::PositiveX => (maximum.x - vertex.position.x, DVec3::NEG_X),
            TerrainFace::NegativeY => (vertex.position.y - minimum.y, DVec3::Y),
            TerrainFace::PositiveY => (maximum.y - vertex.position.y, DVec3::NEG_Y),
            TerrainFace::NegativeZ => (vertex.position.z - minimum.z, DVec3::Z),
            TerrainFace::PositiveZ => (maximum.z - vertex.position.z, DVec3::NEG_Z),
        };
        if distance <= spacing {
            let weight = (1.0 - distance / spacing).clamp(0.0, 1.0);
            delta += direction * (weight * spacing * 0.25);
        }
    }

    let normal = vertex.normal.as_dvec3();
    delta -= normal * delta.dot(normal);
    vertex.position += delta.clamp(DVec3::splat(-spacing), DVec3::splat(spacing));
}

pub(super) fn vertex_on_face(
    position: DVec3,
    minimum: DVec3,
    maximum: DVec3,
    face: TerrainFace,
    epsilon: f64,
) -> bool {
    let distance = match face {
        TerrainFace::NegativeX => position.x - minimum.x,
        TerrainFace::PositiveX => maximum.x - position.x,
        TerrainFace::NegativeY => position.y - minimum.y,
        TerrainFace::PositiveY => maximum.y - position.y,
        TerrainFace::NegativeZ => position.z - minimum.z,
        TerrainFace::PositiveZ => maximum.z - position.z,
    };
    distance.abs() <= epsilon
}

pub(super) fn emit_oriented_triangle(
    chunk: &mut TerrainMeshChunk,
    mut triangle: [MeshVertex; 3],
    fallback_outward: Vec3,
    group: IndexGroup,
) {
    let first = triangle[0].position.as_vec3();
    let geometric =
        (triangle[1].position.as_vec3() - first).cross(triangle[2].position.as_vec3() - first);
    let smooth_outward = triangle.iter().map(|vertex| vertex.normal).sum::<Vec3>();
    let expected_outward = if smooth_outward.length_squared() > 1.0e-12 {
        smooth_outward
    } else {
        fallback_outward
    };
    if geometric.dot(expected_outward) < 0.0 {
        triangle.swap(1, 2);
    }
    if let Some(indices) = append_triangle(chunk, triangle) {
        match group {
            IndexGroup::Regular => chunk.index_groups.regular.extend_from_slice(&indices),
            IndexGroup::Transition(face) => {
                chunk.index_groups.transitions[face.index()].extend_from_slice(&indices);
            }
            IndexGroup::Cap(face) => {
                chunk.index_groups.caps[face.index()].extend_from_slice(&indices);
            }
        }
    }
}

pub(super) fn append_triangle(
    chunk: &mut TerrainMeshChunk,
    triangle: [MeshVertex; 3],
) -> Option<[u32; 3]> {
    if (triangle[1].position - triangle[0].position)
        .cross(triangle[2].position - triangle[0].position)
        .length_squared()
        <= 1.0e-20
    {
        return None;
    }
    let mut indices = [0_u32; 3];
    for (target, vertex) in indices.iter_mut().zip(triangle) {
        let position = vertex.position.as_vec3().to_array();
        let normal = vertex.normal.to_array();
        let (material, surface) = vertex.material;
        let mut weights = [0.0; TerrainMaterial::COUNT];
        weights[material.code() as usize] = 1.0;
        let key = (
            position.map(f32::to_bits),
            normal.map(f32::to_bits),
            surface.0,
        );
        *target = if let Some(&known) = chunk.vertex_cache.vertices.get(&key) {
            let shared = &mut chunk.compaction[known as usize];
            *shared = (*shared).max(vertex.compaction);
            known
        } else {
            let index =
                u32::try_from(chunk.vertices.len()).expect("one chunk has fewer than u32 vertices");
            chunk.vertices.push(position);
            chunk.normals.push(normal);
            chunk.material_weights.push(weights);
            chunk.surfaces.push(surface);
            chunk.compaction.push(vertex.compaction);
            chunk.vertex_cache.vertices.insert(key, index);
            index
        };
    }
    Some(indices)
}

pub(super) fn weighted_materials(
    chunk: &TerrainMeshChunk,
    indices: &[u32],
    barycentric: Vec3,
) -> [f32; TerrainMaterial::COUNT] {
    let mut result = [0.0; TerrainMaterial::COUNT];
    for (corner, weight) in barycentric.to_array().into_iter().enumerate() {
        let source =
            chunk.material_weights[usize::try_from(indices[corner]).expect("index fits usize")];
        for material in 0..TerrainMaterial::COUNT {
            result[material] += source[material] * weight;
        }
    }
    result
}
