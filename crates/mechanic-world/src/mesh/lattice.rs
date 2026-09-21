//! Density lattices sampled with a one-cell halo, including edited and coarse neighbours.

use super::TerrainMeshRequest;
use super::transition::{face_coordinate, transition_coarse_lattice_point};
use crate::{
    BRICK_EDGE_CELLS, TERRAIN_CELL_METERS, TerrainBrick, TerrainFace, TerrainField,
    TerrainMaterial, TerrainNodeId, TerrainOctreeSnapshot, TerrainSample, TerrainTransitionMask,
    WorldCell,
};
use bevy_math::{IVec3, Vec3};
use std::collections::HashMap;

/// Node-local view of promoted terrain used throughout one mesh job.
///
/// Preparing the view traverses the sparse octree once. Subsequent point and
/// range queries touch only the few promoted bricks overlapping the mesh halo,
/// avoiding a depth-27 root walk for every coarse lattice sample.
#[derive(Clone, Debug, Default)]
pub struct PreparedTerrainRegion<'a> {
    pub(super) bricks: HashMap<crate::BrickCoord, &'a crate::TerrainBrick>,
}

impl<'a> PreparedTerrainRegion<'a> {
    /// Prepares the full sampling and transition halo for one mesh request.
    ///
    /// # Panics
    ///
    /// Panics if the requested streamed node lies outside `i32` cell space.
    pub fn for_mesh_request(
        terrain: &'a TerrainOctreeSnapshot,
        request: TerrainMeshRequest,
    ) -> Self {
        let stride = 1_i32 << request.node.level;
        let minimum = request
            .node
            .minimum_cell_i64()
            .map(|cell| i32::try_from(cell).expect("streamed node lies in i32 cell space"));
        let edge = BRICK_EDGE_CELLS * stride;
        // Coarse transition gradients reach four fine strides outside a node.
        // The extra cell covers the lower interpolation neighborhood.
        let padding = 4 * stride + 1;
        Self::between(
            terrain,
            WorldCell::new(
                minimum[0] - padding,
                minimum[1] - padding,
                minimum[2] - padding,
            ),
            WorldCell::new(
                minimum[0] + edge + padding,
                minimum[1] + edge + padding,
                minimum[2] + edge + padding,
            ),
        )
    }

    /// Prepares promoted bricks intersecting an inclusive cell range.
    pub fn between(
        terrain: &'a TerrainOctreeSnapshot,
        minimum: WorldCell,
        maximum: WorldCell,
    ) -> Self {
        debug_assert!(minimum.x <= maximum.x);
        debug_assert!(minimum.y <= maximum.y);
        debug_assert!(minimum.z <= maximum.z);
        let bricks = terrain
            .bricks_between(minimum.brick(), maximum.brick())
            .map(|brick| (brick.coordinate(), brick))
            .collect();
        Self { bricks }
    }

    /// Number of promoted bricks retained by this job-local view.
    pub fn promoted_brick_count(&self) -> usize {
        self.bricks.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.bricks.is_empty()
    }

    pub(super) fn brick(&self, coordinate: crate::BrickCoord) -> Option<&TerrainBrick> {
        self.bricks.get(&coordinate).copied()
    }

    pub(super) fn minimum_promoted_density_between(
        &self,
        minimum: WorldCell,
        maximum: WorldCell,
    ) -> Option<f32> {
        let minimum_brick = minimum.brick();
        let maximum_brick = maximum.brick();
        let mut result = f32::INFINITY;
        for z in minimum_brick.z..=maximum_brick.z {
            for y in minimum_brick.y..=maximum_brick.y {
                for x in minimum_brick.x..=maximum_brick.x {
                    let Some(brick) = self.brick(crate::BrickCoord::new(x, y, z)) else {
                        continue;
                    };
                    let brick_minimum = brick.coordinate().minimum_cell();
                    let first = IVec3::new(
                        minimum.x.max(brick_minimum.x) - brick_minimum.x,
                        minimum.y.max(brick_minimum.y) - brick_minimum.y,
                        minimum.z.max(brick_minimum.z) - brick_minimum.z,
                    );
                    let last = IVec3::new(
                        maximum.x.min(brick_minimum.x + BRICK_EDGE_CELLS - 1) - brick_minimum.x,
                        maximum.y.min(brick_minimum.y + BRICK_EDGE_CELLS - 1) - brick_minimum.y,
                        maximum.z.min(brick_minimum.z + BRICK_EDGE_CELLS - 1) - brick_minimum.z,
                    );
                    if first == IVec3::ZERO && last == IVec3::splat(BRICK_EDGE_CELLS - 1) {
                        result = result.min(brick.minimum_density());
                        continue;
                    }
                    for local_z in first.z..=last.z {
                        for local_y in first.y..=last.y {
                            for local_x in first.x..=last.x {
                                let density = brick
                                    .sample(IVec3::new(local_x, local_y, local_z))
                                    .expect("clamped coordinate is inside prepared brick")
                                    .density;
                                result = result.min(density);
                            }
                        }
                    }
                }
            }
        }
        result.is_finite().then_some(result)
    }
}

#[derive(Clone, Copy)]
pub(super) struct LatticePoint {
    pub(super) sample: TerrainSample,
    pub(super) normal: Vec3,
    pub(super) authored_material: Option<TerrainMaterial>,
}

#[derive(Clone, Copy)]
pub(super) struct LatticeSample {
    pub(super) sample: TerrainSample,
    pub(super) authored_material: Option<TerrainMaterial>,
}

#[expect(
    clippy::too_many_lines,
    reason = "fine and coarse halo preparation share indexing contracts"
)]
pub(super) fn sample_halo(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    node: TerrainNodeId,
    minimum: WorldCell,
    cubes: i32,
    stride: i32,
) -> Vec<LatticeSample> {
    let lattice_edge = usize::try_from(cubes + 1).expect("chunk edge is positive");
    let halo_edge = lattice_edge + 2;
    let mut halo = Vec::with_capacity(halo_edge.pow(3));
    if stride == 1 {
        let cell_edge = halo_edge + 1;
        let columns = field.cached_mesh_columns(node, || {
            let mut columns = Vec::with_capacity(cell_edge.pow(2));
            for z in -2..=cubes + 1 {
                for x in -2..=cubes + 1 {
                    let position = WorldCell::new(minimum.x + x, minimum.y, minimum.z + z).centre();
                    columns.push(field.sample_column(position.0.x, position.0.z));
                }
            }
            columns
        });
        let column_index = |x: i32, z: i32| {
            usize::try_from(x + 2).expect("halo column is positive")
                + usize::try_from(z + 2).expect("halo column is positive") * cell_edge
        };
        let mut cells = Vec::with_capacity(cell_edge.pow(3));
        let mut edited_cells = Vec::with_capacity(cell_edge.pow(3));
        for z in -2..=cubes + 1 {
            for y in -2..=cubes + 1 {
                for x in -2..=cubes + 1 {
                    let cell = WorldCell::new(minimum.x + x, minimum.y + y, minimum.z + z);
                    let coordinate = cell.brick();
                    let generated = field.sample_cell_in_column(cell, columns[column_index(x, z)]);
                    let sample = if edits.is_empty() {
                        generated
                    } else {
                        edits
                            .brick(coordinate)
                            .and_then(|brick| brick.sample(cell.local_in_brick()))
                            .unwrap_or(generated)
                    };
                    cells.push(sample);
                    edited_cells.push(sample != generated);
                }
            }
        }
        let cell_index = |x: i32, y: i32, z: i32| {
            usize::try_from(x + 2).expect("halo cell is positive")
                + usize::try_from(y + 2).expect("halo cell is positive") * cell_edge
                + usize::try_from(z + 2).expect("halo cell is positive") * cell_edge.pow(2)
        };
        for z in -1..=cubes + 1 {
            for y in -1..=cubes + 1 {
                for x in -1..=cubes + 1 {
                    halo.push(lattice_sample_from_cells(
                        &cells,
                        &edited_cells,
                        cell_index,
                        x,
                        y,
                        z,
                    ));
                }
            }
        }
    } else {
        let prepared_edge = halo_edge * 2;
        let columns = field.cached_mesh_columns(node, || {
            let mut columns = Vec::with_capacity(prepared_edge.pow(2));
            for z in -1..=cubes + 1 {
                for z_offset in [-1, 0] {
                    for x in -1..=cubes + 1 {
                        for x_offset in [-1, 0] {
                            let cell = WorldCell::new(
                                minimum.x + x * stride + x_offset,
                                minimum.y,
                                minimum.z + z * stride + z_offset,
                            );
                            let position = cell.centre();
                            columns.push(field.sample_column(position.0.x, position.0.z));
                        }
                    }
                }
            }
            columns
        });
        let column_index = |x: i32, z: i32, x_offset: usize, z_offset: usize| {
            let x = usize::try_from(x + 1).expect("coarse halo column is positive") * 2 + x_offset;
            let z = usize::try_from(z + 1).expect("coarse halo column is positive") * 2 + z_offset;
            x + z * prepared_edge
        };
        for z in -1..=cubes + 1 {
            for y in -1..=cubes + 1 {
                for x in -1..=cubes + 1 {
                    let prepared = [
                        columns[column_index(x, z, 0, 0)],
                        columns[column_index(x, z, 1, 0)],
                        columns[column_index(x, z, 0, 1)],
                        columns[column_index(x, z, 1, 1)],
                    ];
                    halo.push(coarse_sample_in_columns(
                        field,
                        edits,
                        WorldCell::new(
                            minimum.x + x * stride,
                            minimum.y + y * stride,
                            minimum.z + z * stride,
                        ),
                        stride,
                        prepared,
                    ));
                }
            }
        }
    }
    halo
}

pub(super) fn lattice_from_halo(halo: &[LatticeSample], cubes: i32) -> Vec<LatticePoint> {
    let lattice_edge = usize::try_from(cubes + 1).expect("chunk edge is positive");
    let halo_edge = lattice_edge + 2;
    let halo_index = |x: usize, y: usize, z: usize| x + y * halo_edge + z * halo_edge.pow(2);
    let mut lattice = Vec::with_capacity(lattice_edge.pow(3));
    for z in 0..=cubes {
        for y in 0..=cubes {
            for x in 0..=cubes {
                let x = usize::try_from(x + 1).expect("halo coordinate is positive");
                let y = usize::try_from(y + 1).expect("halo coordinate is positive");
                let z = usize::try_from(z + 1).expect("halo coordinate is positive");
                let sampled = halo[halo_index(x, y, z)];
                let gradient = Vec3::new(
                    halo[halo_index(x + 1, y, z)].sample.density
                        - halo[halo_index(x - 1, y, z)].sample.density,
                    halo[halo_index(x, y + 1, z)].sample.density
                        - halo[halo_index(x, y - 1, z)].sample.density,
                    halo[halo_index(x, y, z + 1)].sample.density
                        - halo[halo_index(x, y, z - 1)].sample.density,
                );
                lattice.push(LatticePoint {
                    sample: sampled.sample,
                    normal: (-gradient).normalize_or(Vec3::Y),
                    authored_material: sampled.authored_material,
                });
            }
        }
    }
    lattice
}

pub(super) fn synchronize_edited_boundary_lattice(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    minimum: WorldCell,
    fine_stride: i32,
    lattice_edge: usize,
    transition_mask: TerrainTransitionMask,
    lattice: &mut [LatticePoint],
) {
    if edits.is_empty() {
        return;
    }
    let cubes = lattice_edge - 1;
    let coarse_stride = fine_stride * 2;
    let maximum = WorldCell::new(
        minimum.x
            + i32::try_from(cubes).expect("lattice edge fits i32") * fine_stride
            + coarse_stride
            - 1,
        minimum.y
            + i32::try_from(cubes).expect("lattice edge fits i32") * fine_stride
            + coarse_stride
            - 1,
        minimum.z
            + i32::try_from(cubes).expect("lattice edge fits i32") * fine_stride
            + coarse_stride
            - 1,
    );
    if edits
        .minimum_promoted_density_between(minimum, maximum)
        .is_none()
    {
        return;
    }
    let mut coarse_samples = HashMap::<WorldCell, LatticeSample>::new();
    let mut coarse_columns = HashMap::<(i32, i32), crate::generation::TerrainColumnSample>::new();
    for face in TerrainFace::ALL {
        for v in (0..=cubes).step_by(2) {
            for u in (0..=cubes).step_by(2) {
                let (x, y, z) = face_coordinate(face, u, v, cubes);
                let mut boundary_faces = 0_u8;
                for (coordinate, negative, positive) in [
                    (x, TerrainFace::NegativeX, TerrainFace::PositiveX),
                    (y, TerrainFace::NegativeY, TerrainFace::PositiveY),
                    (z, TerrainFace::NegativeZ, TerrainFace::PositiveZ),
                ] {
                    if coordinate == 0 {
                        boundary_faces |= 1 << negative as u8;
                    } else if coordinate == cubes {
                        boundary_faces |= 1 << positive as u8;
                    }
                }
                if !transition_mask.synchronizes_boundary_feature(boundary_faces) {
                    continue;
                }
                let cell = WorldCell::new(
                    minimum.x
                        + i32::try_from(x).expect("transition coordinate fits i32") * fine_stride,
                    minimum.y
                        + i32::try_from(y).expect("transition coordinate fits i32") * fine_stride,
                    minimum.z
                        + i32::try_from(z).expect("transition coordinate fits i32") * fine_stride,
                );
                let maximum = WorldCell::new(
                    cell.x + coarse_stride - 1,
                    cell.y + coarse_stride - 1,
                    cell.z + coarse_stride - 1,
                );
                if edits
                    .minimum_promoted_density_between(cell, maximum)
                    .is_none()
                {
                    continue;
                }
                lattice[x + y * lattice_edge + z * lattice_edge.pow(2)] =
                    transition_coarse_lattice_point(
                        field,
                        edits,
                        cell,
                        coarse_stride,
                        &mut coarse_samples,
                        &mut coarse_columns,
                    );
            }
        }
    }
}

pub(super) fn coarse_sample_in_columns(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    cell: WorldCell,
    stride: i32,
    columns: [crate::generation::TerrainColumnSample; 4],
) -> LatticeSample {
    let direct = lattice_sample_in_columns(field, edits, cell, columns);
    if edits.is_empty() {
        return direct;
    }
    let maximum = WorldCell::new(
        cell.x + stride - 1,
        cell.y + stride - 1,
        cell.z + stride - 1,
    );
    let Some(minimum_promoted) = edits.minimum_promoted_density_between(cell, maximum) else {
        return direct;
    };
    // A promoted negative sample must remain visible at coarser LODs even when
    // it lies between coarse lattice points.
    if minimum_promoted < direct.sample.density {
        LatticeSample {
            sample: TerrainSample {
                compaction: 0,
                looseness: 0,
                density: minimum_promoted,
                material: direct.sample.material,
            },
            authored_material: direct.authored_material,
        }
    } else {
        direct
    }
}

pub(super) fn lattice_sample_in_columns(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    upper_cell: WorldCell,
    columns: [crate::generation::TerrainColumnSample; 4],
) -> LatticeSample {
    // `WorldCell` values are cell-centred for exact removal accounting, while
    // the meshing lattice lies on their corners. Full signed distances retain
    // smooth procedural interpolation; only neighborhoods containing a real
    // edit are reconstructed as bounded cell occupancies.
    let mut samples = [TerrainSample {
        compaction: 0,
        looseness: 0,
        density: 0.0,
        material: TerrainMaterial::Rock,
    }; 8];
    let mut edited = [false; 8];
    let mut sample_index = 0;
    for (column_index, (x, z)) in [(-1, -1), (0, -1), (-1, 0), (0, 0)].into_iter().enumerate() {
        for y in -1..=0 {
            let cell = WorldCell::new(upper_cell.x + x, upper_cell.y + y, upper_cell.z + z);
            let generated = field.sample_cell_in_column(cell, columns[column_index]);
            let sample = if edits.is_empty() {
                generated
            } else {
                edits
                    .brick(cell.brick())
                    .and_then(|brick| brick.sample(cell.local_in_brick()))
                    .unwrap_or(generated)
            };
            samples[sample_index] = sample;
            edited[sample_index] = sample != generated;
            sample_index += 1;
        }
    }
    blend_lattice_samples(samples, edited)
}

pub(super) fn lattice_sample_from_cells(
    cells: &[TerrainSample],
    edited_cells: &[bool],
    cell_index: impl Fn(i32, i32, i32) -> usize,
    upper_x: i32,
    upper_y: i32,
    upper_z: i32,
) -> LatticeSample {
    let mut samples = [TerrainSample {
        compaction: 0,
        looseness: 0,
        density: 0.0,
        material: TerrainMaterial::Rock,
    }; 8];
    let mut edited = [false; 8];
    let mut sample_index = 0;
    for z in -1..=0 {
        for y in -1..=0 {
            for x in -1..=0 {
                let index = cell_index(upper_x + x, upper_y + y, upper_z + z);
                samples[sample_index] = cells[index];
                edited[sample_index] = edited_cells[index];
                sample_index += 1;
            }
        }
    }
    blend_lattice_samples(samples, edited)
}

pub(super) fn blend_lattice_samples(
    samples: [TerrainSample; 8],
    edited: [bool; 8],
) -> LatticeSample {
    // Plastic compaction retains continuous signed distances. Binary brushes
    // need bounded occupancy reconstruction; applying it to a tiny soil load
    // would snap the previously procedural surface on its very first commit.
    let reconstructing_edit = samples
        .iter()
        .zip(edited)
        .any(|(sample, edited)| edited && sample.compaction == 0);
    let half_cell = TERRAIN_CELL_METERS as f32 * 0.5;
    let mut density = 0.0;
    let mut material = TerrainMaterial::Rock;
    let mut nearest_surface = f32::INFINITY;
    let mut authored_material = None;
    let mut nearest_authored_solid = f32::INFINITY;
    for (sample, edited) in samples.into_iter().zip(edited) {
        density += if reconstructing_edit {
            sample.density.clamp(-half_cell, half_cell)
        } else {
            sample.density
        };
        if sample.density.abs() < nearest_surface {
            nearest_surface = sample.density.abs();
            material = sample.material;
        }
        if edited && sample.is_solid() && sample.density.abs() < nearest_authored_solid {
            nearest_authored_solid = sample.density.abs();
            authored_material = Some(sample.material);
        }
    }
    LatticeSample {
        sample: TerrainSample {
            compaction: 0,
            looseness: 0,
            density: density / 8.0,
            material,
        },
        authored_material,
    }
}
