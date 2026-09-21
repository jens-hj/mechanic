//! The voxel field as loose material meets it: a smooth density with a slope.

use std::collections::HashMap;

use bevy_math::{DVec3, IVec3};
use mechanic_world::{
    BRICK_EDGE_CELLS, BrickCoord, TERRAIN_CELL_METERS, TerrainField, TerrainOctree, WorldCell,
};

/// Bricks remembered before the memory starts over.
const REMEMBERED_BRICKS: usize = 96;
// Corners of a lattice cube, x fastest.
const CUBE: [[i32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];
const BRICK_CELLS: usize = (BRICK_EDGE_CELLS * BRICK_EDGE_CELLS * BRICK_EDGE_CELLS) as usize;

/// Cell densities remembered between ticks. Loose material gathers where
/// ground was just cut, and asks about the same few bricks again and again.
#[derive(Default)]
pub(super) struct Ground {
    slots: Vec<(BrickCoord, Vec<f32>)>,
    index: HashMap<BrickCoord, usize>,
    last: usize,
}

impl Ground {
    /// Forgets bricks whose ground changed.
    pub(super) fn forget(&mut self, bricks: impl IntoIterator<Item = BrickCoord>) {
        for brick in bricks {
            // A lattice sample on a brick face is read by the neighbour's cubes too,
            // but it is stored only here.
            if let Some(&slot) = self.index.get(&brick) {
                self.slots[slot].1.fill(f32::NAN);
            }
        }
    }

    pub(super) fn forget_all(&mut self) {
        self.slots.clear();
        self.index.clear();
        self.last = 0;
    }

    fn density(&mut self, terrain: &TerrainOctree, field: &TerrainField, cell: WorldCell) -> f32 {
        let brick = cell.brick();
        let slot = if self
            .slots
            .get(self.last)
            .is_some_and(|slot| slot.0 == brick)
        {
            self.last
        } else if let Some(&slot) = self.index.get(&brick) {
            slot
        } else {
            if self.slots.len() >= REMEMBERED_BRICKS {
                self.forget_all();
            }
            self.slots.push((brick, vec![f32::NAN; BRICK_CELLS]));
            self.index.insert(brick, self.slots.len() - 1);
            self.slots.len() - 1
        };
        self.last = slot;
        let local: IVec3 = cell.local_in_brick();
        let offset =
            usize::try_from((local.z * BRICK_EDGE_CELLS + local.y) * BRICK_EDGE_CELLS + local.x)
                .unwrap_or_default();
        let known = self.slots[slot].1[offset];
        if !known.is_nan() {
            return known;
        }
        let density = terrain.sample_cell(field, cell).density;
        self.slots[slot].1[offset] = density;
        density
    }

    // The eight samples of a cube inside one brick, when all are remembered.
    fn known_cube(&mut self, brick: BrickCoord, local: IVec3) -> Option<[f64; 8]> {
        let slot = if self
            .slots
            .get(self.last)
            .is_some_and(|slot| slot.0 == brick)
        {
            self.last
        } else {
            *self.index.get(&brick)?
        };
        self.last = slot;
        let samples = &self.slots[slot].1;
        let edge = usize::try_from(BRICK_EDGE_CELLS).unwrap_or_default();
        let base =
            usize::try_from((local.z * BRICK_EDGE_CELLS + local.y) * BRICK_EDGE_CELLS + local.x)
                .ok()?;
        let mut cube = [0.0; 8];
        for (index, value) in cube.iter_mut().enumerate() {
            let sample = samples
                [base + (index & 1) + ((index >> 1) & 1) * edge + ((index >> 2) & 1) * edge * edge];
            if sample.is_nan() {
                return None;
            }
            *value = f64::from(sample);
        }
        Some(cube)
    }

    /// Density, positive in the ground, and its slope per metre at a global
    /// position, read from the lattice the terrain mesh is cut from.
    pub(super) fn sample(
        &mut self,
        terrain: &TerrainOctree,
        field: &TerrainField,
        position: DVec3,
    ) -> (f64, DVec3) {
        let lattice = position / TERRAIN_CELL_METERS;
        let base = lattice.floor();
        let f = lattice - base;
        let cell = base.as_ivec3();
        let first = WorldCell::new(cell.x, cell.y, cell.z);
        let local = first.local_in_brick();
        let known = (local.max_element() < BRICK_EDGE_CELLS - 1)
            .then(|| self.known_cube(first.brick(), local))
            .flatten();
        let corner = known.unwrap_or_else(|| {
            CUBE.map(|[x, y, z]| {
                f64::from(self.density(
                    terrain,
                    field,
                    WorldCell::new(cell.x + x, cell.y + y, cell.z + z),
                ))
            })
        });
        let mix = |a: f64, b: f64, t: f64| a + (b - a) * t;
        let x00 = mix(corner[0], corner[1], f.x);
        let x10 = mix(corner[2], corner[3], f.x);
        let x01 = mix(corner[4], corner[5], f.x);
        let x11 = mix(corner[6], corner[7], f.x);
        let y0 = mix(x00, x10, f.y);
        let y1 = mix(x01, x11, f.y);
        let slope_x = mix(
            mix(corner[1] - corner[0], corner[3] - corner[2], f.y),
            mix(corner[5] - corner[4], corner[7] - corner[6], f.y),
            f.z,
        );
        let slope_y = mix(x10 - x00, x11 - x01, f.z);
        let slope_z = y1 - y0;
        (
            mix(y0, y1, f.z),
            DVec3::new(slope_x, slope_y, slope_z) / TERRAIN_CELL_METERS,
        )
    }
}
