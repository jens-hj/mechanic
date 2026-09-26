//! What water can occupy: the open terrain cells in each water cell, and the
//! seed-derived water around it.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "cell indices within one 32-cell brick"
)]

use std::collections::HashMap;

use bevy_math::{DVec3, IVec3};

use super::WaterCell;
use crate::generation::Lattice;
use crate::{BRICK_EDGE_CELLS, BrickCoord, TerrainField, TerrainSource, WaterBody, WaterSurface};

/// Terrain cells along one edge of a water cell.
pub const WATER_CELL_EDGE_CELLS: i32 = 4;

/// Water cells along one edge of a brick.
pub(super) const BRICK_EDGE_WATER_CELLS: i32 = BRICK_EDGE_CELLS / WATER_CELL_EDGE_CELLS;

/// Water cells in one brick.
const BRICK_WATER_CELLS: usize = 512;

/// Open terrain cells in each of a water cell's four layers, bottom first.
pub(super) type Openings = [u8; 4];

/// The ground water sits in.
pub trait WaterGround {
    /// Whether each terrain cell of a brick is open, x fastest, then y, then
    /// z, over the brick's 32³ cells.
    fn open_cells(&self, brick: BrickCoord) -> Vec<bool>;

    /// Seed-derived water at a point, if the point holds any.
    fn implicit(&self, point: DVec3) -> Option<WaterSurface>;

    /// Surface area of a lake, in square metres.
    fn lake_area(&self, lake: u32) -> f64;

    /// Seed-derived water over a column, whatever the ground below.
    fn surface(&self, x: f64, z: f64) -> Option<WaterSurface>;

    /// Whether seed-derived water may reach into a brick at all.
    fn may_hold_water(&self, brick: BrickCoord) -> bool;
}

/// Terrain, untouched and edited, as the ground water sits in.
#[derive(Clone, Copy)]
pub struct TerrainWater<'a, S> {
    /// Untouched terrain and its implicit water.
    pub field: &'a TerrainField,
    /// Edited bricks.
    pub edits: &'a S,
}

impl<S: TerrainSource> WaterGround for TerrainWater<'_, S> {
    fn open_cells(&self, brick: BrickCoord) -> Vec<bool> {
        let edge = BRICK_EDGE_CELLS;
        if let Some(edited) = self.edits.brick(brick) {
            let mut open = Vec::with_capacity((edge * edge * edge) as usize);
            for z in 0..edge {
                for y in 0..edge {
                    for x in 0..edge {
                        let sample = edited.sample(IVec3::new(x, y, z));
                        open.push(sample.is_none_or(|sample| !sample.is_solid()));
                    }
                }
            }
            return open;
        }
        let minimum = brick.minimum_cell();
        let lattice = Lattice {
            origin: IVec3::new(minimum.x, minimum.y, minimum.z),
            stride: 1,
            dims: [edge as usize; 3],
            centred: true,
        };
        self.field
            .density_lattice(&lattice)
            .into_iter()
            .map(|density| density <= 0.0)
            .collect()
    }

    fn implicit(&self, point: DVec3) -> Option<WaterSurface> {
        let surface = self.field.water_surface(point.x, point.z)?;
        (point.y < surface.level && self.field.is_water(point)).then_some(surface)
    }

    fn lake_area(&self, lake: u32) -> f64 {
        self.field.lake_area(lake).unwrap_or(f64::INFINITY)
    }

    fn surface(&self, x: f64, z: f64) -> Option<WaterSurface> {
        self.field.water_surface(x, z)
    }

    fn may_hold_water(&self, brick: BrickCoord) -> bool {
        let minimum =
            brick.minimum_cell().centre().0 - DVec3::splat(0.5 * crate::TERRAIN_CELL_METERS);
        let maximum = minimum + DVec3::splat(crate::BRICK_EDGE_METERS);
        self.field
            .water_level_range(minimum, maximum)
            .is_some_and(|(_, highest)| minimum.y < highest)
    }
}

/// Openings of every water cell in the bricks water has looked at.
#[derive(Clone, Debug, Default)]
pub(super) struct OpeningsCache {
    bricks: HashMap<BrickCoord, Box<[Openings; BRICK_WATER_CELLS]>>,
    /// Seed-derived water found at each cell's lowest opening.
    implicit: HashMap<WaterCell, Option<WaterSurface>>,
}

impl OpeningsCache {
    /// Openings of one water cell.
    pub(super) fn openings(&mut self, ground: &impl WaterGround, cell: WaterCell) -> Openings {
        let brick = cell.brick();
        let openings = self.bricks.entry(brick).or_insert_with(|| {
            let open = ground.open_cells(brick);
            let edge = BRICK_EDGE_CELLS as usize;
            let mut openings = Box::new([[0_u8; 4]; BRICK_WATER_CELLS]);
            for (index, &open) in open.iter().enumerate() {
                if !open {
                    continue;
                }
                let (x, y, z) = (index % edge, index / edge % edge, index / edge / edge);
                let water = WaterCell::local_index(IVec3::new(
                    (x / 4) as i32,
                    (y / 4) as i32,
                    (z / 4) as i32,
                ));
                openings[water][y % 4] += 1;
            }
            openings
        });
        openings[cell.local_index_in_brick()]
    }

    /// Seed-derived water in a cell, looked for at its lowest opening.
    pub(super) fn implicit(
        &mut self,
        ground: &impl WaterGround,
        cell: WaterCell,
    ) -> Option<WaterSurface> {
        if let Some(&found) = self.implicit.get(&cell) {
            return found;
        }
        if !ground.may_hold_water(cell.brick()) {
            self.implicit.insert(cell, None);
            return None;
        }
        let openings = self.openings(ground, cell);
        let found = openings
            .iter()
            .position(|&open| open > 0)
            .and_then(|layer| ground.implicit(cell.layer_centre(layer)));
        self.implicit.insert(cell, found);
        found
    }

    /// Forgets what the ground was in a brick.
    pub(super) fn forget(&mut self, brick: BrickCoord) {
        self.bricks.remove(&brick);
        self.implicit.retain(|cell, _| cell.brick() != brick);
    }
}

/// Whether a body of seed-derived water is a lake, and which.
pub(super) const fn lake_of(body: WaterBody) -> Option<u32> {
    match body {
        WaterBody::Lake(lake) => Some(lake),
        WaterBody::Sea | WaterBody::River | WaterBody::Pool(_) => None,
    }
}
