//! Face caps and Transvoxel transition cells between levels of detail.

use super::TerrainMeshChunk;
use super::groups::IndexGroup;
use super::lattice::{
    LatticePoint, LatticeSample, PreparedTerrainRegion, coarse_sample_in_columns,
};
use super::polygonise::{
    MeshVertex, apply_transition_inset, crossing_material, emit_oriented_triangle,
};
use crate::transvoxel::tables::{
    TRANSITION_CELL_CLASS, TRANSITION_CELL_DATA, TRANSITION_CORNER_DATA, TRANSITION_VERTEX_DATA,
};
use crate::{TerrainFace, TerrainField, WorldCell};
use bevy_math::{DVec3, Vec3};
use std::array;
use std::collections::HashMap;

pub(super) fn generate_face_cap(
    lattice: &[LatticePoint],
    lattice_edge: usize,
    face: TerrainFace,
    chunk: &mut TerrainMeshChunk,
) {
    let cubes = lattice_edge - 1;
    let outward = face_normal(face);
    for v in 0..cubes {
        for u in 0..cubes {
            let coordinates = [(u, v), (u + 1, v), (u + 1, v + 1), (u, v + 1)];
            let points = coordinates.map(|(u, v)| {
                let (x, y, z) = face_coordinate(face, u, v, cubes);
                let lattice_index = x + y * lattice_edge + z * lattice_edge.pow(2);
                let position = chunk.origin.0
                    + DVec3::new(
                        f64::from(u32::try_from(x).expect("cap coordinate fits u32")),
                        f64::from(u32::try_from(y).expect("cap coordinate fits u32")),
                        f64::from(u32::try_from(z).expect("cap coordinate fits u32")),
                    ) * chunk.sample_spacing_metres;
                (lattice[lattice_index], position)
            });
            let case = points
                .iter()
                .enumerate()
                .fold(0_u8, |case, (index, point)| {
                    case | if point.0.sample.is_solid() {
                        1 << index
                    } else {
                        0
                    }
                });
            if case == 0 {
                continue;
            }
            if case == 0b0101 || case == 0b1010 {
                for corner in (0..4).filter(|&corner| points[corner].0.sample.is_solid()) {
                    let previous = (corner + 3) % 4;
                    let next = (corner + 1) % 4;
                    emit_oriented_triangle(
                        chunk,
                        [
                            cap_crossing(points[corner], points[previous], outward),
                            cap_vertex(points[corner], outward),
                            cap_crossing(points[corner], points[next], outward),
                        ],
                        outward,
                        IndexGroup::Cap(face),
                    );
                }
            } else {
                let mut polygon = [cap_vertex(points[0], outward); 6];
                let mut polygon_length = 0;
                for current in 0..4 {
                    let next = (current + 1) % 4;
                    if points[current].0.sample.is_solid() {
                        polygon[polygon_length] = cap_vertex(points[current], outward);
                        polygon_length += 1;
                    }
                    if points[current].0.sample.is_solid() != points[next].0.sample.is_solid() {
                        polygon[polygon_length] =
                            cap_crossing(points[current], points[next], outward);
                        polygon_length += 1;
                    }
                }
                for index in 1..polygon_length.saturating_sub(1) {
                    emit_oriented_triangle(
                        chunk,
                        [polygon[0], polygon[index], polygon[index + 1]],
                        outward,
                        IndexGroup::Cap(face),
                    );
                }
            }
        }
    }
}

pub(super) fn cap_vertex(point: (LatticePoint, DVec3), outward: Vec3) -> MeshVertex {
    MeshVertex {
        position: point.1,
        normal: outward,
        material: point.0.sample.material,
    }
}

pub(super) fn cap_crossing(
    first: (LatticePoint, DVec3),
    second: (LatticePoint, DVec3),
    outward: Vec3,
) -> MeshVertex {
    let first_density = f64::from(first.0.sample.density);
    let second_density = f64::from(second.0.sample.density);
    let along = (first_density / (first_density - second_density)).clamp(0.0, 1.0);
    let material = crossing_material(first.0, second.0);
    MeshVertex {
        position: first.1.lerp(second.1, along),
        normal: outward,
        material,
    }
}

pub(super) fn generate_transition_face(
    lattice: &[LatticePoint],
    lattice_edge: usize,
    face: TerrainFace,
    chunk: &mut TerrainMeshChunk,
) {
    let cubes = lattice_edge - 1;
    for v in (0..cubes).step_by(2) {
        for u in (0..cubes).step_by(2) {
            let fine = array::from_fn::<_, 9, _>(|index| {
                let du = index % 3;
                let dv = index / 3;
                let (x, y, z) = face_coordinate(face, u + du, v + dv, cubes);
                let point = lattice[x + y * lattice_edge + z * lattice_edge.pow(2)];
                let position = chunk.origin.0
                    + DVec3::new(
                        f64::from(u32::try_from(x).expect("transition coordinate fits u32")),
                        f64::from(u32::try_from(y).expect("transition coordinate fits u32")),
                        f64::from(u32::try_from(z).expect("transition coordinate fits u32")),
                    ) * chunk.sample_spacing_metres;
                (point, position)
            });
            // The official tables encode the eight perimeter samples clockwise,
            // followed by the centre, rather than the row-major order above.
            let case = [0, 1, 2, 5, 8, 7, 6, 3, 4].into_iter().enumerate().fold(
                0_u16,
                |case, (bit, point)| {
                    case | if fine[point].0.sample.is_solid() {
                        1 << bit
                    } else {
                        0
                    }
                },
            );
            if case == 0 || case == 0x1ff {
                continue;
            }
            let class = TRANSITION_CELL_CLASS[usize::from(case)];
            let reverse = class & 0x80 != 0;
            let cell = TRANSITION_CELL_DATA[usize::from(class & 0x7f)];
            let vertex_count = usize::from(cell.geometry_counts >> 4);
            let triangle_count = usize::from(cell.geometry_counts & 0x0f);
            let mut vertices = Vec::with_capacity(vertex_count);
            for &data in &TRANSITION_VERTEX_DATA[usize::from(case)][..vertex_count] {
                let edge = data & 0xff;
                let first_index = usize::from((edge >> 4) as u8);
                let second_index = usize::from((edge & 0x0f) as u8);
                let first = transition_point(first_index, &fine);
                let second = transition_point(second_index, &fine);
                let first_density = f64::from(first.0.sample.density);
                let second_density = f64::from(second.0.sample.density);
                let along = (first_density / (first_density - second_density)).clamp(0.0, 1.0);
                let mut vertex = MeshVertex {
                    position: first.1.lerp(second.1, along),
                    normal: first
                        .0
                        .normal
                        .lerp(second.0.normal, along as f32)
                        .normalize_or(face_normal(face)),
                    material: crossing_material(first.0, second.0),
                };
                if first_index < 9 || second_index < 9 {
                    apply_transition_inset(&mut vertex, chunk);
                }
                vertices.push(vertex);
            }
            for triangle in cell.vertex_index[..triangle_count * 3].chunks_exact(3) {
                let mut triangle = [
                    vertices[usize::from(triangle[0])],
                    vertices[usize::from(triangle[1])],
                    vertices[usize::from(triangle[2])],
                ];
                if reverse {
                    triangle.swap(1, 2);
                }
                emit_oriented_triangle(chunk, triangle, Vec3::ZERO, IndexGroup::Transition(face));
            }
        }
    }
}

pub(super) fn transition_coarse_lattice_point(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    cell: WorldCell,
    stride: i32,
    samples: &mut HashMap<WorldCell, LatticeSample>,
    columns: &mut HashMap<(i32, i32), crate::generation::TerrainColumnSample>,
) -> LatticePoint {
    let sample = transition_coarse_sample(field, edits, cell, stride, samples, columns);
    let density =
        |offset: [i32; 3],
         samples: &mut HashMap<WorldCell, LatticeSample>,
         columns: &mut HashMap<(i32, i32), crate::generation::TerrainColumnSample>| {
            transition_coarse_sample(
                field,
                edits,
                WorldCell::new(
                    cell.x + offset[0] * stride,
                    cell.y + offset[1] * stride,
                    cell.z + offset[2] * stride,
                ),
                stride,
                samples,
                columns,
            )
            .sample
            .density
        };
    let gradient = Vec3::new(
        density([1, 0, 0], samples, columns) - density([-1, 0, 0], samples, columns),
        density([0, 1, 0], samples, columns) - density([0, -1, 0], samples, columns),
        density([0, 0, 1], samples, columns) - density([0, 0, -1], samples, columns),
    );
    LatticePoint {
        sample: sample.sample,
        normal: (-gradient).normalize_or(Vec3::Y),
        authored_material: sample.authored_material,
    }
}

pub(super) fn transition_coarse_sample(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    cell: WorldCell,
    stride: i32,
    samples: &mut HashMap<WorldCell, LatticeSample>,
    columns: &mut HashMap<(i32, i32), crate::generation::TerrainColumnSample>,
) -> LatticeSample {
    if let Some(&sample) = samples.get(&cell) {
        return sample;
    }
    let prepared = [(-1, -1), (0, -1), (-1, 0), (0, 0)].map(|(x, z)| {
        let column_cell = WorldCell::new(cell.x + x, cell.y, cell.z + z);
        *columns
            .entry((column_cell.x, column_cell.z))
            .or_insert_with(|| {
                let position = column_cell.centre();
                field.sample_column(position.0.x, position.0.z)
            })
    });
    let sample = coarse_sample_in_columns(field, edits, cell, stride, prepared);
    samples.insert(cell, sample);
    sample
}

pub(super) fn transition_point(
    index: usize,
    fine: &[(LatticePoint, DVec3); 9],
) -> (LatticePoint, DVec3) {
    let reuse_data = TRANSITION_CORNER_DATA[index];
    debug_assert!(reuse_data <= 0x87);
    match index {
        0..=8 => fine[index],
        9 => fine[0],
        10 => fine[2],
        11 => fine[6],
        12 => fine[8],
        _ => unreachable!("official transition endpoint is 0 through C"),
    }
}

pub(super) fn face_coordinate(
    face: TerrainFace,
    u: usize,
    v: usize,
    cubes: usize,
) -> (usize, usize, usize) {
    match face {
        TerrainFace::NegativeX => (0, u, v),
        TerrainFace::PositiveX => (cubes, u, v),
        TerrainFace::NegativeY => (u, 0, v),
        TerrainFace::PositiveY => (u, cubes, v),
        TerrainFace::NegativeZ => (u, v, 0),
        TerrainFace::PositiveZ => (u, v, cubes),
    }
}

pub(super) fn face_normal(face: TerrainFace) -> Vec3 {
    match face {
        TerrainFace::NegativeX => Vec3::NEG_X,
        TerrainFace::PositiveX => Vec3::X,
        TerrainFace::NegativeY => Vec3::NEG_Y,
        TerrainFace::PositiveY => Vec3::Y,
        TerrainFace::NegativeZ => Vec3::NEG_Z,
        TerrainFace::PositiveZ => Vec3::Z,
    }
}
