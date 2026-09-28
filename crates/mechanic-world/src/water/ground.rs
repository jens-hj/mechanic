//! What water can occupy: the open terrain cells in each water cell, and the
//! seed-derived water around it.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "cell indices within one 32-cell brick"
)]

use bevy_math::{DVec3, IVec3};

use super::cells::CellMap;
use super::cycle::WaterNetwork;
use super::{WATER_CELL_METRES, WaterCell};
use crate::generation::Lattice;
use crate::{
    BRICK_EDGE_CELLS, BrickCoord, LakeBasin, RiverReach, TerrainDensityClass, TerrainField,
    TerrainMaterial, TerrainSource, WaterSurface,
};

/// Terrain cells along one edge of a water cell.
pub const WATER_CELL_EDGE_CELLS: i32 = 4;

/// Water cells along one edge of a brick.
pub(super) const BRICK_EDGE_WATER_CELLS: i32 = BRICK_EDGE_CELLS / WATER_CELL_EDGE_CELLS;

/// Water cells in one brick.
const BRICK_WATER_CELLS: usize = 512;

/// Seed-derived water at the waterline of untouched ground, too shallow to
/// reach any layer's centre: a cell whose floor lies within a cell below
/// the surface over its column. Deeper untouched cells under the surface
/// are sealed voids, dry as the seed made them.
fn shore(ground: &impl WaterGround, cell: WaterCell, openings: Openings) -> Option<WaterSurface> {
    if ground.edited(cell.brick()) {
        return None;
    }
    let floor = super::floor_of(cell, openings)?;
    let centre = cell.centre();
    ground
        .surface(centre.x, centre.z)
        .filter(|surface| floor < surface.level && floor > surface.level - WATER_CELL_METRES)
}

/// Terrain cells in one water cell.
pub(super) const CELL_TERRAIN_CELLS: usize = 64;

/// Openings not yet sampled: no layer holds more than 16 open cells.
const UNKNOWN: Openings = [u8::MAX; 4];

/// Position of one of a water cell's terrain cells, x fastest, then y, then
/// z.
fn local_offset(index: usize) -> IVec3 {
    let index = index as i32;
    let edge = WATER_CELL_EDGE_CELLS;
    IVec3::new(index % edge, index / edge % edge, index / (edge * edge))
}

/// Open terrain cells in each of a water cell's four layers, bottom first.
pub(super) type Openings = [u8; 4];

/// The ground water sits in.
pub trait WaterGround: WaterNetwork {
    /// Whether each terrain cell of one water cell is open, x fastest, then
    /// y, then z, over its 4³ cells.
    fn open_cells(&self, cell: WaterCell) -> [bool; CELL_TERRAIN_CELLS];

    /// Seed-derived water at a point, if the point holds any.
    fn implicit(&self, point: DVec3) -> Option<WaterSurface>;

    /// Seed-derived water over a column, whatever the ground below.
    fn surface(&self, x: f64, z: f64) -> Option<WaterSurface>;

    /// Whether seed-derived water may reach into a brick at all.
    fn may_hold_water(&self, brick: BrickCoord) -> bool;

    /// Material of the ground at a point, where it is ground.
    fn material(&self, point: DVec3) -> Option<TerrainMaterial>;

    /// Whether a brick's ground was edited.
    fn edited(&self, brick: BrickCoord) -> bool;

    /// Whether edits opened ground in a cell that the seed left solid.
    /// Seed-derived water pours only into dug ground: untouched ground, even
    /// in an edited brick, is wet or dry as the seed made it.
    fn dug(&self, cell: WaterCell) -> bool;
}

/// Terrain, untouched and edited, as the ground water sits in.
#[derive(Clone, Copy)]
pub struct TerrainWater<'a, S> {
    /// Untouched terrain and its implicit water.
    pub field: &'a TerrainField,
    /// Edited bricks.
    pub edits: &'a S,
}

impl<S> WaterNetwork for TerrainWater<'_, S> {
    fn lake(&self, lake: u32) -> Option<LakeBasin> {
        self.field.lake_basin(lake)
    }

    fn reach(&self, reach: u32) -> Option<RiverReach> {
        self.field.river_reach(reach)
    }
}

impl<S: TerrainSource> TerrainWater<'_, S> {
    /// Open terrain cells of one water cell in its edited brick, if the brick
    /// was edited.
    fn edited_open(&self, cell: WaterCell) -> Option<[bool; CELL_TERRAIN_CELLS]> {
        let edited = self.edits.brick(cell.brick())?;
        let edge = WATER_CELL_EDGE_CELLS;
        let origin = IVec3::new(cell.x * edge, cell.y * edge, cell.z * edge);
        let local = origin - {
            let minimum = cell.brick().minimum_cell();
            IVec3::new(minimum.x, minimum.y, minimum.z)
        };
        let mut open = [false; CELL_TERRAIN_CELLS];
        for (index, open) in open.iter_mut().enumerate() {
            let sample = edited.sample(local + local_offset(index));
            *open = sample.is_none_or(|sample| !sample.is_solid());
        }
        Some(open)
    }

    /// Open terrain cells of one water cell as the seed made them.
    fn seeded_open(&self, cell: WaterCell) -> [bool; CELL_TERRAIN_CELLS] {
        let edge = WATER_CELL_EDGE_CELLS;
        let origin = IVec3::new(cell.x * edge, cell.y * edge, cell.z * edge);
        let mut open = [false; CELL_TERRAIN_CELLS];
        // Most cells water looks at are wholly air or wholly ground, which
        // the field bounds far faster than it samples.
        let centre = |cell: IVec3| crate::WorldCell::new(cell.x, cell.y, cell.z).centre().0;
        match self
            .field
            .classify(centre(origin), centre(origin + IVec3::splat(edge - 1)))
        {
            TerrainDensityClass::Empty => return [true; CELL_TERRAIN_CELLS],
            TerrainDensityClass::Solid => return open,
            TerrainDensityClass::Mixed => {}
        }
        let lattice = Lattice {
            origin,
            stride: 1,
            dims: [edge as usize; 3],
            centred: true,
        };
        for (open, density) in open.iter_mut().zip(self.field.density_lattice(&lattice)) {
            *open = density <= 0.0;
        }
        open
    }
}

impl<S: TerrainSource> WaterGround for TerrainWater<'_, S> {
    fn open_cells(&self, cell: WaterCell) -> [bool; CELL_TERRAIN_CELLS] {
        self.edited_open(cell)
            .unwrap_or_else(|| self.seeded_open(cell))
    }

    fn dug(&self, cell: WaterCell) -> bool {
        self.edited_open(cell).is_some_and(|open| {
            open.iter()
                .zip(self.seeded_open(cell))
                .any(|(&now, seeded)| now && !seeded)
        })
    }

    fn implicit(&self, point: DVec3) -> Option<WaterSurface> {
        let surface = self.field.water_surface(point.x, point.z)?;
        (point.y < surface.level && self.field.is_water(point)).then_some(surface)
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

    fn edited(&self, brick: BrickCoord) -> bool {
        self.edits.brick(brick).is_some()
    }

    fn material(&self, point: DVec3) -> Option<TerrainMaterial> {
        let sample = self
            .edits
            .sample_position(self.field, crate::WorldPosition(point));
        sample.is_solid().then_some(sample.material)
    }
}

/// Openings of every water cell in the bricks water has looked at.
#[derive(Clone, Debug, Default)]
pub(super) struct OpeningsCache {
    bricks: CellMap<BrickCoord, Box<[Openings; BRICK_WATER_CELLS]>>,
    /// Seed-derived water found in each cell.
    implicit: CellMap<WaterCell, Option<WaterSurface>>,
    /// Whether seed-derived water may reach into each brick.
    may_hold: CellMap<BrickCoord, bool>,
}

impl OpeningsCache {
    /// Openings of one water cell, sampled the first time it is asked for.
    pub(super) fn openings(&mut self, ground: &impl WaterGround, cell: WaterCell) -> Openings {
        let known = self
            .bricks
            .entry(cell.brick())
            .or_insert_with(|| Box::new([UNKNOWN; BRICK_WATER_CELLS]));
        let slot = &mut known[cell.local_index_in_brick()];
        if *slot == UNKNOWN {
            let mut openings = [0_u8; 4];
            for (index, open) in ground.open_cells(cell).into_iter().enumerate() {
                if open {
                    openings[local_offset(index).y as usize] += 1;
                }
            }
            *slot = openings;
        }
        *slot
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
        let may_hold = *self
            .may_hold
            .entry(cell.brick())
            .or_insert_with(|| ground.may_hold_water(cell.brick()));
        if !may_hold {
            self.implicit.insert(cell, None);
            return None;
        }
        // Any open layer will do, highest first: on a sloping bed the lowest
        // layer's opening may lie off the column's centre, in the ground.
        let openings = self.openings(ground, cell);
        let found = (0..openings.len())
            .rev()
            .filter(|&layer| openings[layer] > 0)
            .find_map(|layer| ground.implicit(cell.layer_centre(layer)))
            .or_else(|| shore(ground, cell, openings));
        self.implicit.insert(cell, found);
        found
    }

    /// Forgets what the ground was in a brick.
    pub(super) fn forget(&mut self, brick: BrickCoord) {
        self.bricks.remove(&brick);
        self.implicit.retain(|cell, _| cell.brick() != brick);
    }
}
