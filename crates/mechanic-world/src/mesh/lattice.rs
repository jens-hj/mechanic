//! Density lattices sampled with a one-cell halo, including edited and coarse neighbours.

use super::TerrainMeshRequest;
use super::transition::{face_coordinate, transition_coarse_lattice_point};
use crate::generation::{Lattice, LatticeColumns, corner_position};
use crate::{
    BRICK_EDGE_CELLS, SurfaceId, TERRAIN_CELL_METERS, TerrainBrick, TerrainFace, TerrainField,
    TerrainMaterial, TerrainOctreeSnapshot, TerrainSample, TerrainTransitionMask, WorldCell,
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

/// Material and surface an edit supplied, which outranks procedural painting.
pub(super) type Authored = Option<(TerrainMaterial, SurfaceId)>;

#[derive(Clone, Copy)]
pub(super) struct LatticePoint {
    pub(super) sample: TerrainSample,
    pub(super) normal: Vec3,
    pub(super) authored: Authored,
}

#[derive(Clone, Copy)]
pub(super) struct LatticeSample {
    pub(super) sample: TerrainSample,
    pub(super) authored: Authored,
    /// Untouched ground whose material has not been painted yet.
    pub(super) procedural: bool,
}

impl LatticeSample {
    pub(super) fn procedural(density: f64) -> Self {
        #[expect(clippy::cast_possible_truncation, reason = "sample densities are f32")]
        let density = density as f32;
        Self {
            sample: TerrainSample::plain(density, TerrainMaterial::Rock),
            authored: None,
            procedural: true,
        }
    }
}

/// Whether an unedited chunk evidently holds no surface, so meshing it would
/// sample its whole lattice for nothing.
pub(super) fn chunk_is_clear(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    minimum: WorldCell,
    cubes: i32,
    stride: i32,
) -> bool {
    if !edits.is_empty() {
        return false;
    }
    let halo_edge = usize::try_from(cubes + 3).expect("chunk edge is positive");
    let lattice = Lattice {
        origin: IVec3::new(minimum.x - stride, minimum.y - stride, minimum.z - stride),
        stride,
        dims: [halo_edge; 3],
        centred: false,
    };
    field.lattice_is_clear(&lattice, stride <= 1 << crate::CAVE_STREAMED_LEVEL)
}

/// Lattice samples with a one-sample halo, plus what painting them needs.
pub(super) struct Halo {
    pub(super) samples: Vec<LatticeSample>,
    columns: LatticeColumns,
}

pub(super) fn sample_halo(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    minimum: WorldCell,
    cubes: i32,
    stride: i32,
) -> Halo {
    let halo_edge = usize::try_from(cubes + 3).expect("chunk edge is positive");
    let lattice = Lattice {
        origin: IVec3::new(minimum.x - stride, minimum.y - stride, minimum.z - stride),
        stride,
        dims: [halo_edge; 3],
        centred: false,
    };
    // Culled blocks hold bounds, not densities, which an edit's crossings
    // would interpolate against.
    let enclosed = stride <= 1 << crate::CAVE_STREAMED_LEVEL;
    let (corners, columns) = field.mesh_lattice(&lattice, edits.is_empty(), enclosed);
    let mut samples: Vec<LatticeSample> = corners
        .iter()
        .map(|density| LatticeSample::procedural(*density))
        .collect();
    if edits.is_empty() {
        return Halo { samples, columns };
    }
    // Cell-centre densities of untouched ground around every lattice point,
    // used to tell edited cells from baked procedural ones.
    let generated_cells = (stride == 1).then(|| {
        let cells = Lattice {
            origin: IVec3::new(minimum.x - 2, minimum.y - 2, minimum.z - 2),
            stride: 1,
            dims: [halo_edge + 1; 3],
            centred: true,
        };
        (cells, field.density_lattice(&cells))
    });
    for z in 0..halo_edge {
        for y in 0..halo_edge {
            for x in 0..halo_edge {
                let offset = |index: usize| i32::try_from(index).expect("halo fits i32") - 1;
                let corner = WorldCell::new(
                    minimum.x + offset(x) * stride,
                    minimum.y + offset(y) * stride,
                    minimum.z + offset(z) * stride,
                );
                let index = x + halo_edge * (y + halo_edge * z);
                let procedural = samples[index];
                samples[index] = if let Some((cells, densities)) = &generated_cells {
                    edited_lattice_sample(field, edits, corner, procedural, |cell| {
                        let local = |axis: usize, value: i32| {
                            usize::try_from(value - cells.origin[axis]).expect("cell lies in halo")
                        };
                        let index = local(0, cell.x)
                            + cells.dims[0] * (local(1, cell.y) + cells.dims[1] * local(2, cell.z));
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "sample densities are f32"
                        )]
                        let density = densities[index] as f32;
                        density
                    })
                } else {
                    coarse_lattice_sample(field, edits, corner, stride, procedural)
                };
            }
        }
    }
    Halo { samples, columns }
}

/// A lattice point whose eight surrounding cells may hold edits: blends the
/// cells when any differs from untouched ground, else keeps `procedural`.
fn edited_lattice_sample(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    corner: WorldCell,
    procedural: LatticeSample,
    generated_density: impl Fn(WorldCell) -> f32,
) -> LatticeSample {
    let mut cells = [corner; 8];
    let mut samples = [procedural.sample; 8];
    let mut edited = [false; 8];
    let mut promoted = false;
    let mut index = 0;
    for z in -1..=0 {
        for y in -1..=0 {
            for x in -1..=0 {
                let cell = WorldCell::new(corner.x + x, corner.y + y, corner.z + z);
                cells[index] = cell;
                if let Some(sample) = edits
                    .brick(cell.brick())
                    .and_then(|brick| brick.sample(cell.local_in_brick()))
                {
                    promoted = true;
                    samples[index] = sample;
                    edited[index] = sample.density.to_bits() != generated_density(cell).to_bits();
                } else {
                    samples[index] =
                        TerrainSample::plain(generated_density(cell), TerrainMaterial::Rock);
                }
                index += 1;
            }
        }
    }
    if !promoted || !edited.contains(&true) {
        return procedural;
    }
    let blended = blend_lattice_samples(samples, edited);
    let (material, surface) = if edited[blended.nearest] {
        let sample = samples[blended.nearest];
        (sample.material, sample.surface)
    } else {
        let sample = field.sample_cell(cells[blended.nearest]);
        (sample.material, sample.surface)
    };
    LatticeSample {
        sample: TerrainSample {
            material,
            surface,
            ..blended.sample
        },
        authored: blended.authored,
        procedural: false,
    }
}

/// A coarse lattice point: its eight surrounding cells as at full detail, but
/// never hiding a promoted empty cell anywhere inside its stride.
pub(super) fn coarse_lattice_sample(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    corner: WorldCell,
    stride: i32,
    procedural: LatticeSample,
) -> LatticeSample {
    let maximum = WorldCell::new(
        corner.x + stride - 1,
        corner.y + stride - 1,
        corner.z + stride - 1,
    );
    let lower = WorldCell::new(corner.x - 1, corner.y - 1, corner.z - 1);
    if edits
        .minimum_promoted_density_between(lower, maximum)
        .is_none()
    {
        return procedural;
    }
    let direct = edited_lattice_sample(field, edits, corner, procedural, |cell| {
        field.cell_density(cell)
    });
    let Some(minimum_promoted) = edits.minimum_promoted_density_between(corner, maximum) else {
        return direct;
    };
    // A promoted negative sample must remain visible at coarser LODs even when
    // it lies between coarse lattice points.
    if minimum_promoted < direct.sample.density {
        LatticeSample {
            sample: TerrainSample {
                density: minimum_promoted,
                compaction: 0,
                looseness: 0,
                ..direct.sample
            },
            ..direct
        }
    } else {
        direct
    }
}

pub(super) fn lattice_from_halo(
    field: &TerrainField,
    halo: &Halo,
    minimum: WorldCell,
    cubes: i32,
    stride: i32,
) -> Vec<LatticePoint> {
    let lattice_edge = usize::try_from(cubes + 1).expect("chunk edge is positive");
    let halo_edge = lattice_edge + 2;
    let samples = &halo.samples;
    let halo_index = |x: usize, y: usize, z: usize| x + y * halo_edge + z * halo_edge.pow(2);
    let spacing = f64::from(stride) * TERRAIN_CELL_METERS;
    let mut lattice = Vec::with_capacity(lattice_edge.pow(3));
    for z in 1..=lattice_edge {
        for y in 1..=lattice_edge {
            for x in 1..=lattice_edge {
                let mut point = samples[halo_index(x, y, z)];
                let neighbours = [
                    samples[halo_index(x + 1, y, z)].sample.density,
                    samples[halo_index(x - 1, y, z)].sample.density,
                    samples[halo_index(x, y + 1, z)].sample.density,
                    samples[halo_index(x, y - 1, z)].sample.density,
                    samples[halo_index(x, y, z + 1)].sample.density,
                    samples[halo_index(x, y, z - 1)].sample.density,
                ];
                let gradient = Vec3::new(
                    neighbours[0] - neighbours[1],
                    neighbours[2] - neighbours[3],
                    neighbours[4] - neighbours[5],
                );
                if point.procedural {
                    let density = point.sample.density;
                    let solid = density > 0.0;
                    let on_boundary = [x, y, z]
                        .iter()
                        .any(|&coordinate| coordinate == 1 || coordinate == lattice_edge);
                    // Only crossing endpoints and cap corners are ever shown.
                    let shown = neighbours.iter().any(|&other| (other > 0.0) != solid)
                        || (on_boundary && solid && f64::from(density) < 3.0 * spacing);
                    if shown {
                        let position = corner_position(WorldCell::new(
                            minimum.x + (i32::try_from(x).expect("halo fits i32") - 1) * stride,
                            minimum.y + (i32::try_from(y).expect("halo fits i32") - 1) * stride,
                            minimum.z + (i32::try_from(z).expect("halo fits i32") - 1) * stride,
                        ));
                        let world_gradient = gradient.as_dvec3() / (2.0 * spacing);
                        let (material, surface) = field.paint_lattice(
                            &halo.columns,
                            [x, y, z],
                            position,
                            f64::from(density),
                            world_gradient.to_array(),
                        );
                        point.sample.material = material;
                        point.sample.surface = surface;
                    }
                }
                lattice.push(LatticePoint {
                    sample: point.sample,
                    normal: (-gradient).normalize_or(Vec3::Y),
                    authored: point.authored,
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
                    );
            }
        }
    }
}

pub(super) struct BlendedSample {
    pub(super) sample: TerrainSample,
    pub(super) authored: Authored,
    /// Which of the eight samples lies nearest the surface.
    pub(super) nearest: usize,
}

pub(super) fn blend_lattice_samples(
    samples: [TerrainSample; 8],
    edited: [bool; 8],
) -> BlendedSample {
    // Plastic compaction retains continuous signed distances. Binary brushes
    // need bounded occupancy reconstruction; applying it to a tiny soil load
    // would snap the previously procedural surface on its very first commit.
    let reconstructing_edit = samples
        .iter()
        .zip(edited)
        .any(|(sample, edited)| edited && sample.compaction == 0);
    let half_cell = TERRAIN_CELL_METERS as f32 * 0.5;
    let mut density = 0.0;
    let mut nearest = 0;
    let mut nearest_surface = f32::INFINITY;
    let mut authored = None;
    let mut nearest_authored_solid = f32::INFINITY;
    for (index, (sample, edited)) in samples.into_iter().zip(edited).enumerate() {
        density += if reconstructing_edit {
            sample.density.clamp(-half_cell, half_cell)
        } else {
            sample.density
        };
        if sample.density.abs() < nearest_surface {
            nearest_surface = sample.density.abs();
            nearest = index;
        }
        if edited && sample.is_solid() && sample.density.abs() < nearest_authored_solid {
            nearest_authored_solid = sample.density.abs();
            authored = Some((sample.material, sample.surface));
        }
    }
    BlendedSample {
        sample: TerrainSample {
            density: density / 8.0,
            compaction: 0,
            looseness: 0,
            ..samples[nearest]
        },
        authored,
        nearest,
    }
}
