//! Stored water: pools that fill, spill, drain and merge as the ground
//! changes. See `docs/water.md`.
//!
//! Water is held in 20 cm water cells, four terrain cells to an edge. All
//! stored water lies in pools. A pool is one level and one volume over its
//! member cells, and grows by priority flood: the lowest neighbour whose floor
//! the water covers is taken in next. A neighbour it cannot take in is a
//! contact: another pool's cell, seed-derived water, or a drop the water
//! falls over. Across a contact the higher water runs to the lower at a weir
//! rate, and pools whose levels meet merge. Water that falls lands in the pool
//! below it or starts one. So a U-tube is one pool and settles level, a pit
//! fills and then spills at its lowest rim, and a trench dug from a lake fills
//! from it. Water taken from a lake lowers the whole lake, and every cubic
//! metre is booked: see [`cycle`] for where it comes from and goes.

mod cycle;
mod ground;
mod pool;
mod sheet;

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use bevy_math::{DVec2, DVec3, IVec3};
use serde::{Deserialize, Serialize};

use cycle::Cycle;
pub use cycle::{SurplusDoc, WaterLedger, WaterNetwork, WaterShift};
use ground::{BRICK_EDGE_WATER_CELLS, Openings, OpeningsCache};
pub use ground::{TerrainWater, WATER_CELL_EDGE_CELLS, WaterGround};
use pool::Pool;
use sheet::Sheet;
pub use sheet::SheetDoc;

use crate::{BrickCoord, TERRAIN_CELL_METERS, TerrainField, WaterBody, WaterSurface};

/// Edge of one water cell, in metres.
pub const WATER_CELL_METRES: f64 = TERRAIN_CELL_METERS * WATER_CELL_EDGE_CELLS as f64;

/// Height of one terrain-cell layer, in metres.
const FINE_LAYER_METRES: f64 = TERRAIN_CELL_METERS;

/// Volume of one terrain cell, in cubic metres.
const FINE_VOLUME_M3: f64 = TERRAIN_CELL_METERS * TERRAIN_CELL_METERS * TERRAIN_CELL_METERS;

/// Water spreads onto a cell only once it stands this deep over its floor,
/// in metres: a film thinner than this stays where it is.
const FILM_METRES: f64 = 0.01;

/// Pools whose levels lie this close, in metres, merge where they touch.
const MERGE_METRES: f64 = 0.005;

/// Weir coefficient for water running over a cell's edge, in m^½/s: a
/// contact passes `WEIR × width × head^1.5` cubic metres a second.
const WEIR_COEFFICIENT: f64 = 1.7;

/// Cells one pool takes in per step: how fast a front of water advances.
const SPREAD_CELLS_PER_STEP: usize = 256;

/// Deepest a falling stream is followed, in water cells.
const FALL_CELLS: i32 = 2_000;

/// Least water a pool keeps before it dries up and is forgotten, in m³.
const DRY_M3: f64 = 1.0e-6;

/// Water evaporating from a stored surface, in metres a second: 5 mm an
/// hour, so a forgotten puddle dries in a day.
const EVAPORATION_M_S: f64 = 0.005 / 3_600.0;

/// One 20 cm cube of water storage, four terrain cells to an edge.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct WaterCell {
    /// East/west water-cell coordinate.
    pub x: i32,
    /// Vertical water-cell coordinate.
    pub y: i32,
    /// North/south water-cell coordinate.
    pub z: i32,
}

impl WaterCell {
    /// Creates a water-cell coordinate.
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// The water cell containing a global position.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "world positions lie well inside the i32 cell range"
    )]
    pub fn containing(position: DVec3) -> Self {
        let cell = (position / WATER_CELL_METRES).floor();
        Self::new(cell.x as i32, cell.y as i32, cell.z as i32)
    }

    /// Height of the cell's bottom face, in metres.
    pub fn bottom(self) -> f64 {
        f64::from(self.y) * WATER_CELL_METRES
    }

    /// Global position of the cell's centre.
    pub fn centre(self) -> DVec3 {
        (DVec3::new(f64::from(self.x), f64::from(self.y), f64::from(self.z)) + 0.5)
            * WATER_CELL_METRES
    }

    fn brick(self) -> BrickCoord {
        let edge = BRICK_EDGE_WATER_CELLS;
        BrickCoord::new(
            self.x.div_euclid(edge),
            self.y.div_euclid(edge),
            self.z.div_euclid(edge),
        )
    }

    #[expect(clippy::cast_sign_loss, reason = "local coordinates are non-negative")]
    fn local_index(local: IVec3) -> usize {
        let edge = BRICK_EDGE_WATER_CELLS;
        (local.x + edge * (local.y + edge * local.z)) as usize
    }

    fn local_index_in_brick(self) -> usize {
        let edge = BRICK_EDGE_WATER_CELLS;
        Self::local_index(IVec3::new(
            self.x.rem_euclid(edge),
            self.y.rem_euclid(edge),
            self.z.rem_euclid(edge),
        ))
    }

    /// Global index of one of its terrain-cell layers.
    fn fine_layer(self, layer: usize) -> i32 {
        self.y * WATER_CELL_EDGE_CELLS + i32::try_from(layer).expect("four layers")
    }

    /// Centre of one of its layers, in the middle of the column.
    fn layer_centre(self, layer: usize) -> DVec3 {
        let centre = self.centre();
        #[expect(clippy::cast_precision_loss, reason = "four layers")]
        let y = (layer as f64 + 0.5).mul_add(FINE_LAYER_METRES, self.bottom());
        DVec3::new(centre.x, y, centre.z)
    }

    const fn below(self) -> Self {
        Self::new(self.x, self.y - 1, self.z)
    }

    const fn up(self) -> Self {
        Self::new(self.x, self.y + 1, self.z)
    }

    const fn neighbours(self) -> [Self; 6] {
        [
            Self::new(self.x - 1, self.y, self.z),
            Self::new(self.x + 1, self.y, self.z),
            Self::new(self.x, self.y, self.z - 1),
            Self::new(self.x, self.y, self.z + 1),
            Self::new(self.x, self.y - 1, self.z),
            Self::new(self.x, self.y + 1, self.z),
        ]
    }

    /// Every water cell of a brick.
    fn in_brick(brick: BrickCoord) -> impl Iterator<Item = Self> {
        let edge = BRICK_EDGE_WATER_CELLS;
        (0..edge * edge * edge).map(move |index| {
            Self::new(
                brick.x * edge + index % edge,
                brick.y * edge + index / edge % edge,
                brick.z * edge + index / edge / edge,
            )
        })
    }
}

/// Water a cell's open layers hold below a height, in m³.
fn held_in(cell: WaterCell, openings: Openings, level: f64) -> f64 {
    openings
        .iter()
        .enumerate()
        .map(|(layer, &open)| {
            #[expect(clippy::cast_precision_loss, reason = "four layers")]
            let bottom = (layer as f64).mul_add(FINE_LAYER_METRES, cell.bottom());
            let fill = ((level - bottom) / FINE_LAYER_METRES).clamp(0.0, 1.0);
            f64::from(open) * FINE_VOLUME_M3 * fill
        })
        .sum()
}

/// Height of a cell's lowest open layer, where it has one.
fn floor_of(cell: WaterCell, openings: Openings) -> Option<f64> {
    openings.iter().position(|&open| open > 0).map(|layer| {
        #[expect(clippy::cast_precision_loss, reason = "four layers")]
        let layer = layer as f64;
        layer.mul_add(FINE_LAYER_METRES, cell.bottom())
    })
}

/// Water falling from a pool over a drop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterFall {
    /// Where it leaves the pool.
    pub from: DVec3,
    /// Where it lands.
    pub to: DVec3,
    /// How much falls, in m³/s.
    pub rate_m3_s: f64,
}

/// What one water step did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WaterStep {
    /// Water moved between pools, seed-derived water and falls, in m³.
    pub moved_m3: f64,
    /// Pools after the step.
    pub pools: usize,
    /// Cells held by pools after the step.
    pub cells: usize,
    /// Cells of running water after the step.
    pub sheet_cells: usize,
    /// Streams falling this step.
    pub falls: Vec<WaterFall>,
}

/// One pool as it is drawn.
#[derive(Clone, Debug, PartialEq)]
pub struct PoolView {
    /// Pool number, as [`WaterBody::Pool`] names it.
    pub id: u32,
    /// Height of its surface, in metres.
    pub level: f64,
    /// Water held, in m³.
    pub volume_m3: f64,
    /// Member cells the surface crosses: each column's top wet cell.
    pub surface_cells: Vec<WaterCell>,
    /// Depth of water in each of those columns, in metres.
    pub depths: Vec<f64>,
}

/// One stored pool in a saved world.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoolDoc {
    /// The cell it floods out from again when the world loads.
    pub seed: WaterCell,
    /// Water held, in m³.
    pub volume_m3: f64,
}

/// Stored water of a saved world.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StoredWaterDoc {
    /// Every pool.
    pub pools: Vec<PoolDoc>,
    /// Every lake and river reach water has been drawn from or added to.
    pub bodies: Vec<SurplusDoc>,
    /// Water the sea holds beyond its seed level, in m³.
    pub sea_m3: f64,
    /// Water evaporated from stored water, not yet fallen back, in m³.
    pub air_m3: f64,
    /// Cells that filled from seed-derived water and joined it.
    pub joined: Vec<JoinedCellDoc>,
    /// Running water.
    pub sheets: Vec<SheetDoc>,
}

/// A cell that filled from seed-derived water and joined it, in a saved
/// world.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct JoinedCellDoc {
    /// The cell.
    pub cell: WaterCell,
    /// The water it joined.
    pub body: WaterBody,
    /// That water's level as the seed made it, in metres.
    pub level: f64,
    /// Water the cell holds, in m³.
    #[serde(default)]
    pub held_m3: f64,
}

/// A cell that joined seed-derived water: that water's seed surface, and
/// the water the cell holds, booked apart from the body's surplus.
#[derive(Clone, Copy, Debug)]
struct Joined {
    surface: WaterSurface,
    held: f64,
}

/// An immutable view of the water for queries off the water's own thread:
/// every wet stored cell with its pool's level, and how far each moved lake
/// and river stands from its seed surface.
#[derive(Clone, Debug)]
pub struct WaterSurfaces {
    field: Arc<TerrainField>,
    wet: HashMap<WaterCell, (u32, f64)>,
    running: HashMap<WaterCell, WaterSurface>,
    shifts: BTreeMap<WaterBody, WaterShift>,
    /// Cells that joined seed-derived water, with its undrawn surface.
    joined: HashMap<WaterCell, WaterSurface>,
}

impl WaterSurfaces {
    /// Seed-derived water only, with nothing stored and no lake drawn.
    pub fn untouched(field: Arc<TerrainField>) -> Self {
        Self {
            field,
            wet: HashMap::new(),
            running: HashMap::new(),
            shifts: BTreeMap::new(),
            joined: HashMap::new(),
        }
    }

    /// How far each moved lake and river stands from its seed surface.
    pub const fn shifts(&self) -> &BTreeMap<WaterBody, WaterShift> {
        &self.shifts
    }

    /// The water at a point: the stored pool whose water reaches it, or the
    /// seed-derived water over its column at its drawn level.
    pub fn surface(&self, point: DVec3) -> Option<WaterSurface> {
        let cell = WaterCell::containing(point);
        for dy in (-8..=2).rev() {
            let near = WaterCell::new(cell.x, cell.y + dy, cell.z);
            if let Some(&(id, level)) = self.wet.get(&near) {
                return Some(WaterSurface {
                    level,
                    body: WaterBody::Pool(id),
                    flow: DVec2::ZERO,
                });
            }
            if let Some(&running) = self.running.get(&near) {
                return Some(running);
            }
            if let Some(&joined) = self.joined.get(&near) {
                return Some(cycle::shifted(&self.shifts, joined));
            }
        }
        self.field
            .water_surface(point.x, point.z)
            .map(|surface| cycle::shifted(&self.shifts, surface))
    }
}

/// Where water moving in a step goes to or comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum End {
    Pool(u32),
    /// Seed-derived water, which books what it gains or loses.
    Body(WaterBody),
    /// Running water on a cell's floor.
    Sheet(WaterCell),
    /// A cell with no pool yet, which one starts in.
    Seed(WaterCell),
}

/// A stable key for one end of a transfer, for grouping.
const fn end_key(end: End) -> (u8, u32) {
    match end {
        End::Pool(id) | End::Body(WaterBody::Pool(id)) => (0, id),
        End::Seed(_) => (1, 0),
        End::Body(WaterBody::Sea) => (2, 0),
        End::Body(WaterBody::Lake(lake)) => (3, lake),
        End::Body(WaterBody::River(reach)) => (4, reach),
        End::Sheet(_) | End::Body(WaterBody::Running) => (5, 0),
    }
}

/// What one step's contacts do.
#[derive(Debug, Default)]
struct Exchanges {
    transfers: Vec<Transfer>,
    falls: Vec<WaterFall>,
    /// Pools whose levels met where they touch.
    merges: Vec<(u32, u32)>,
    /// Pools that reached the seed-derived water they touch, at a contact.
    joins: Vec<(u32, WaterCell)>,
}

#[derive(Clone, Copy, Debug)]
struct Transfer {
    from: End,
    to: End,
    volume: f64,
}

/// All stored water in a world.
#[derive(Clone, Debug, Default)]
pub struct WaterWorld {
    pools: BTreeMap<u32, Pool>,
    next_pool: u32,
    owner: HashMap<WaterCell, u32>,
    ground: OpeningsCache,
    /// The rivers, lakes, sea and air around stored water.
    cycle: Cycle,
    /// Free cells seed-derived water pours into.
    inlets: std::collections::BTreeSet<WaterCell>,
    /// Cells a pool filled up to the seed-derived water beside it, which then
    /// became part of that water.
    joined: HashMap<WaterCell, Joined>,
    /// Running water, by cell.
    sheets: BTreeMap<WaterCell, Sheet>,
}

impl WaterWorld {
    /// A world without stored water.
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuilds a saved world's water, flooding each pool out from its seed.
    pub fn from_doc(ground: &impl WaterGround, doc: &StoredWaterDoc) -> Self {
        let mut water = Self::new();
        water.cycle = Cycle::from_doc(&doc.bodies, doc.sea_m3, doc.air_m3);
        for joined in &doc.joined {
            water.joined.insert(
                joined.cell,
                Joined {
                    surface: WaterSurface {
                        level: joined.level,
                        body: joined.body,
                        flow: DVec2::ZERO,
                    },
                    held: joined.held_m3,
                },
            );
        }
        for pool in &doc.pools {
            water.deposit_at(ground, pool.seed, pool.volume_m3);
        }
        for sheet in &doc.sheets {
            water.add_sheet(ground, sheet.cell, sheet.volume_m3);
        }
        let ids = water.pools.keys().copied().collect::<Vec<_>>();
        for id in ids {
            water.flood(ground, id, usize::MAX);
        }
        water
    }

    /// The world's stored water, to save.
    pub fn to_doc(&self) -> StoredWaterDoc {
        StoredWaterDoc {
            pools: self
                .pools
                .values()
                .map(|pool| PoolDoc {
                    seed: pool.seed,
                    volume_m3: pool.volume,
                })
                .collect(),
            bodies: self.cycle.to_doc(),
            sea_m3: self.cycle.sea(),
            air_m3: self.cycle.air(),
            joined: {
                let mut joined = self
                    .joined
                    .iter()
                    .map(|(&cell, joined)| JoinedCellDoc {
                        cell,
                        body: joined.surface.body,
                        level: joined.surface.level,
                        held_m3: joined.held,
                    })
                    .collect::<Vec<_>>();
                joined.sort_by_key(|joined| joined.cell);
                joined
            },
            sheets: self.sheet_docs(),
        }
    }

    /// Water held in pools, in m³.
    pub fn stored_m3(&self) -> f64 {
        self.pools.values().map(|pool| pool.volume).sum()
    }

    /// Cells that filled from seed-derived water and joined it, with the
    /// water's surface at its current level.
    pub fn joined_cells(&self, ground: &impl WaterGround) -> Vec<(WaterCell, WaterSurface)> {
        let mut joined = self
            .joined
            .iter()
            .map(|(&cell, joined)| (cell, self.drawn(ground, joined.surface)))
            .collect::<Vec<_>>();
        joined.sort_by_key(|(cell, _)| *cell);
        joined
    }

    /// Water held in cells that joined seed-derived water, in m³.
    pub fn joined_m3(&self) -> f64 {
        self.joined.values().map(|joined| joined.held).sum()
    }

    /// Water a lake or river reach holds beyond its seed share, in m³:
    /// negative where water was drawn from it.
    pub fn surplus_m3(&self, body: WaterBody) -> f64 {
        self.cycle.surplus(body)
    }

    /// Every cubic metre of water in the world's books.
    pub fn ledger(&self) -> WaterLedger {
        let (lakes_m3, rivers_m3) = self.cycle.totals();
        WaterLedger {
            pools_m3: self.stored_m3(),
            running_m3: self.running_m3(),
            joined_m3: self.joined_m3(),
            lakes_m3,
            rivers_m3,
            sea_m3: self.cycle.sea(),
            air_m3: self.cycle.air(),
        }
    }

    /// A view of the water to query elsewhere.
    pub fn surfaces(&self, field: Arc<TerrainField>) -> WaterSurfaces {
        let wet = self
            .owner
            .iter()
            .filter_map(|(&cell, &id)| {
                let level = self.pools.get(&id)?.level;
                (level > cell.bottom()).then_some((cell, (id, level)))
            })
            .collect();
        let shifts = self.cycle.shifts(field.as_ref());
        let running = self
            .sheets
            .iter()
            .filter_map(|(&cell, sheet)| Some((cell, sheet.surface()?)))
            .collect();
        let joined = self
            .joined
            .iter()
            .map(|(&cell, joined)| (cell, joined.surface))
            .collect();
        WaterSurfaces {
            field,
            wet,
            running,
            shifts,
            joined,
        }
    }

    /// Every pool, to draw.
    pub fn pools(&self) -> impl Iterator<Item = PoolView> + '_ {
        self.pools.iter().map(|(&id, pool)| {
            let mut columns = BTreeMap::<(i32, i32), (WaterCell, f64)>::new();
            for (&cell, &openings) in &pool.members {
                if cell.bottom() < pool.level {
                    let floor = floor_of(cell, openings).unwrap_or_else(|| cell.bottom());
                    let column = columns.entry((cell.x, cell.z)).or_insert((cell, floor));
                    if cell.y > column.0.y {
                        column.0 = cell;
                    }
                    column.1 = column.1.min(floor);
                }
            }
            let (surface_cells, depths) = columns
                .into_values()
                .map(|(cell, floor)| (cell, (pool.level - floor).max(0.0)))
                .unzip();
            PoolView {
                id,
                level: pool.level,
                volume_m3: pool.volume,
                surface_cells,
                depths,
            }
        })
    }

    /// The water at a point: a pool whose water reaches it, running water,
    /// or the seed-derived water over its column where it stands now.
    pub fn surface(&self, ground: &impl WaterGround, point: DVec3) -> Option<WaterSurface> {
        let cell = WaterCell::containing(point);
        for dy in (-8..=2).rev() {
            let near = WaterCell::new(cell.x, cell.y + dy, cell.z);
            if let Some(&id) = self.owner.get(&near) {
                let pool = &self.pools[&id];
                if pool.level > near.bottom() {
                    return Some(WaterSurface {
                        level: pool.level,
                        body: WaterBody::Pool(id),
                        flow: DVec2::ZERO,
                    });
                }
            }
            if let Some(running) = self.running(near) {
                return Some(running);
            }
            if let Some(joined) = self.joined.get(&near) {
                return Some(self.drawn(ground, joined.surface));
            }
        }
        ground
            .surface(point.x, point.z)
            .map(|surface| self.drawn(ground, surface))
    }

    /// Seed-derived water at its current level and current.
    fn drawn(&self, ground: &impl WaterGround, surface: WaterSurface) -> WaterSurface {
        self.cycle.shift(ground, surface.body).apply(surface)
    }

    fn openings(&mut self, ground: &impl WaterGround, cell: WaterCell) -> Openings {
        self.ground.openings(ground, cell)
    }

    fn implicit(&mut self, ground: &impl WaterGround, cell: WaterCell) -> Option<WaterSurface> {
        let surface = match self.joined.get(&cell) {
            Some(joined) => joined.surface,
            None => self.ground.implicit(ground, cell)?,
        };
        let surface = self.drawn(ground, surface);
        (floor_of(cell, self.openings(ground, cell))? < surface.level).then_some(surface)
    }

    /// Whether water in a cell falls out of its bottom: into open, free
    /// space, or onto water lower than the cell, other than `pool`'s own.
    fn falls(&mut self, ground: &impl WaterGround, cell: WaterCell, pool: Option<u32>) -> bool {
        let below = cell.below();
        if self.openings(ground, cell)[0] == 0 || self.openings(ground, below)[3] == 0 {
            return false;
        }
        if let Some(&other) = self.owner.get(&below) {
            return Some(other) != pool
                && self
                    .pools
                    .get(&other)
                    .is_some_and(|other| other.level < cell.bottom());
        }
        self.implicit(ground, below)
            .is_none_or(|surface| surface.level < cell.bottom())
    }

    /// Where water entering a cell ends up, following it down any drop.
    fn landing(&mut self, ground: &impl WaterGround, mut cell: WaterCell) -> End {
        for _ in 0..FALL_CELLS {
            if let Some(&id) = self.owner.get(&cell) {
                return End::Pool(id);
            }
            if let Some(surface) = self.implicit(ground, cell) {
                return End::Body(surface.body);
            }
            let below = cell.below();
            if !self.falls(ground, cell, None) {
                // Water resting on water joins it.
                if let Some(&id) = self.owner.get(&below) {
                    return End::Pool(id);
                }
                return End::Seed(cell);
            }
            cell = below;
        }
        End::Seed(cell)
    }

    fn start_pool(&mut self, ground: &impl WaterGround, cell: WaterCell, volume: f64) -> u32 {
        let id = self.next_pool;
        self.next_pool += 1;
        let mut pool = Pool::new(cell, volume);
        let openings = self.openings(ground, cell);
        pool.rim = floor_of(cell, openings).unwrap_or_else(|| cell.bottom());
        pool.add_member(cell, openings);
        self.owner.insert(cell, id);
        self.pools.insert(id, pool);
        self.absorb_sheet(id, cell);
        self.border(ground, id, cell);
        if let Some(pool) = self.pools.get_mut(&id) {
            pool.settle();
        }
        id
    }

    /// Queues a member's neighbours, or records them as contacts.
    fn border(&mut self, ground: &impl WaterGround, id: u32, cell: WaterCell) {
        for neighbour in cell.neighbours() {
            let owner = self.owner.get(&neighbour).copied();
            if owner == Some(id) {
                continue;
            }
            let openings = self.openings(ground, neighbour);
            let Some(floor) = floor_of(neighbour, openings) else {
                continue;
            };
            let contact = owner.is_some() || self.implicit(ground, neighbour).is_some();
            let Some(pool) = self.pools.get_mut(&id) else {
                return;
            };
            if contact {
                pool.contacts.insert(neighbour);
            } else {
                pool.queue(neighbour, floor);
            }
        }
    }

    /// Takes in the neighbours a pool's water covers, lowest first, up to
    /// `budget` cells.
    fn flood(&mut self, ground: &impl WaterGround, id: u32, mut budget: usize) {
        loop {
            let Some(pool) = self.pools.get_mut(&id) else {
                return;
            };
            pool.settle();
            if budget == 0 {
                return;
            }
            let Some(cell) = pool.next_below(pool.level - FILM_METRES) else {
                return;
            };
            let openings = self.openings(ground, cell);
            let Some(floor) = floor_of(cell, openings) else {
                continue;
            };
            if self.owner.get(&cell) == Some(&id) {
                continue;
            }
            // Ground falling away below the highest floor the water crossed
            // lies beyond the pool's rim: it spills there.
            let beyond_rim = self.pools.get(&id).is_some_and(|pool| floor < pool.rim);
            if beyond_rim
                || self.owner.contains_key(&cell)
                || self.implicit(ground, cell).is_some()
                || self.falls(ground, cell, Some(id))
            {
                if let Some(pool) = self.pools.get_mut(&id) {
                    pool.contacts.insert(cell);
                }
                continue;
            }
            if let Some(pool) = self.pools.get_mut(&id) {
                pool.add_member(cell, openings);
                pool.rim = pool.rim.max(floor);
            }
            self.owner.insert(cell, id);
            self.absorb_sheet(id, cell);
            self.border(ground, id, cell);
            budget -= 1;
        }
    }

    /// Runs the water for `dt` seconds.
    pub fn step(&mut self, ground: &impl WaterGround, dt: f64) -> WaterStep {
        let ids = self.pools.keys().copied().collect::<Vec<_>>();
        for &id in &ids {
            self.flood(ground, id, SPREAD_CELLS_PER_STEP);
        }
        let mut out = Exchanges::default();
        for &id in &ids {
            self.exchange(ground, id, dt, &mut out);
        }
        self.pour_inlets(ground, dt, &mut out.transfers);
        let (mut moved_m3, mut fed) = self.apply(ground, &out.transfers);
        let (running, sheet_fed, lips) = self.step_sheets(ground, dt);
        moved_m3 += running;
        fed.extend(sheet_fed);
        out.falls.extend(lips);
        // Water that arrived floods at once, so no pool stands higher than
        // its water can reach.
        for &id in &fed {
            self.flood(ground, id, SPREAD_CELLS_PER_STEP);
        }
        // Pools join seed-derived water before they merge with each other,
        // so two pools under a lake join it rather than pool what the lake
        // presses up into them.
        for (id, contact) in out.joins {
            self.join(ground, id, contact);
        }
        for (keep, other) in out.merges {
            self.merge(keep, other);
        }
        self.settle_sheets(ground);
        self.evaporate(dt);
        self.evaporate_sheets(EVAPORATION_M_S, dt);
        self.cycle.step(ground, dt);
        self.dry_up(ground, &fed);
        for pool in self.pools.values_mut() {
            pool.settle();
        }
        WaterStep {
            moved_m3,
            pools: self.pools.len(),
            cells: self.owner.len(),
            sheet_cells: self.sheets.len(),
            falls: out.falls,
        }
    }

    /// Works out what runs across one pool's contacts over `dt`.
    fn exchange(&mut self, ground: &impl WaterGround, id: u32, dt: f64, out: &mut Exchanges) {
        let Some(pool) = self.pools.get(&id) else {
            return;
        };
        let (level, area) = (
            pool.level,
            pool.surface_area().max(WATER_CELL_METRES.powi(2)),
        );
        // A pool that fills its cells under seed-derived water standing over
        // them is that water already.
        let (top, full, rim) = (pool.top(), pool.full(), pool.rim);
        let contacts = pool.contacts.iter().copied().collect::<Vec<_>>();
        let weir = |head: f64| WEIR_COEFFICIENT * WATER_CELL_METRES * head.max(0.0).powf(1.5) * dt;
        // Water between two bodies over all the cells they touch at: summed
        // over the cells, but never more than brings their levels halfway
        // together.
        let mut levelling = BTreeMap::<((u8, u32), (u8, u32)), (Transfer, f64)>::new();
        let mut level_between = |from: End, to: End, volume: f64, even: f64| {
            let key = (end_key(from), end_key(to));
            let entry = levelling.entry(key).or_insert((
                Transfer {
                    from,
                    to,
                    volume: 0.0,
                },
                even,
            ));
            entry.0.volume += volume;
            entry.1 = entry.1.min(even);
        };
        for cell in contacts {
            let openings = self.openings(ground, cell);
            let Some(floor) = floor_of(cell, openings) else {
                self.drop_contact(id, cell);
                continue;
            };
            let own_floor = floor;
            // Water crosses into the contact only over the higher of its
            // floor and the floor of the member it leaves.
            let floor = self.lip(id, cell).map_or(floor, |lip| lip.max(floor));
            if let Some(&other) = self.owner.get(&cell) {
                if other == id {
                    self.drop_contact(id, cell);
                    continue;
                }
                let other_level = self.pools[&other].level;
                let other_area = self.pools[&other]
                    .surface_area()
                    .max(WATER_CELL_METRES.powi(2));
                if (level - other_level).abs() < MERGE_METRES && level > floor {
                    out.merges.push((id.min(other), id.max(other)));
                } else if level > other_level {
                    let head = level - other_level.max(floor);
                    let even = 0.5 * (level - other_level) * area.min(other_area);
                    level_between(End::Pool(id), End::Pool(other), weir(head), even);
                }
            } else if let Some(surface) = self.implicit(ground, cell)
                && level > floor
                && (level > surface.level - MERGE_METRES || (full && surface.level > top))
            {
                out.joins.push((id, cell));
            } else if let Some(surface) = self.implicit(ground, cell) {
                let even = 0.5 * (surface.level - level).abs() * area;
                let (from, to, head) = if surface.level > level {
                    (
                        End::Body(surface.body),
                        End::Pool(id),
                        surface.level - level.max(floor),
                    )
                } else {
                    (
                        End::Pool(id),
                        End::Body(surface.body),
                        level - surface.level.max(floor),
                    )
                };
                level_between(from, to, weir(head), even);
            } else if self.falls(ground, cell, Some(id)) {
                self.fall(ground, id, cell, [level, floor, area], dt, out);
            } else if own_floor < rim {
                // Beyond the rim the water runs off as a sheet.
                let head = level - floor;
                if head > 0.0 {
                    out.transfers.push(Transfer {
                        from: End::Pool(id),
                        to: End::Sheet(cell),
                        volume: weir(head).min(head * area),
                    });
                }
            } else {
                self.drop_contact(id, cell);
                if let Some(pool) = self.pools.get_mut(&id) {
                    pool.queue(cell, floor);
                }
            }
        }
        for (mut transfer, even) in levelling.into_values() {
            transfer.volume = transfer.volume.min(even);
            out.transfers.push(transfer);
        }
    }

    /// Water spilling from a pool over a drop at `cell`, with the pool's
    /// level, the lip's floor and the pool's surface area.
    fn fall(
        &mut self,
        ground: &impl WaterGround,
        id: u32,
        cell: WaterCell,
        [level, floor, area]: [f64; 3],
        dt: f64,
        out: &mut Exchanges,
    ) {
        let weir = |head: f64| WEIR_COEFFICIENT * WATER_CELL_METRES * head.max(0.0).powf(1.5) * dt;
        let head = level - floor;
        if head > 0.0 {
            let to = self.landing(ground, cell.below());
            let volume = weir(head).min(head * area);
            let landing = match to {
                End::Pool(other) => self.pools[&other].level.min(cell.bottom()),
                End::Seed(seed) | End::Sheet(seed) => seed.bottom(),
                End::Body(_) => cell.bottom() - WATER_CELL_METRES,
            };
            let from = cell.centre();
            // A stream leaves over the lip, never from higher than
            // the cell above it.
            let top = level.min(cell.bottom() + 2.0 * WATER_CELL_METRES);
            out.falls.push(WaterFall {
                from: DVec3::new(from.x, top, from.z),
                to: DVec3::new(from.x, landing, from.z),
                rate_m3_s: volume / dt,
            });
            out.transfers.push(Transfer {
                from: End::Pool(id),
                to,
                volume,
            });
        }
    }

    /// The lowest floor among a pool's members beside a cell.
    fn lip(&self, id: u32, cell: WaterCell) -> Option<f64> {
        let pool = self.pools.get(&id)?;
        cell.neighbours()
            .into_iter()
            .filter_map(|neighbour| {
                let openings = pool.members.get(&neighbour)?;
                // Water leaves a member upward only through its top.
                if neighbour.y < cell.y {
                    return Some(neighbour.bottom() + WATER_CELL_METRES);
                }
                floor_of(neighbour, *openings)
            })
            .reduce(f64::min)
    }

    /// Water evaporating from every pool's surface into the air.
    fn evaporate(&mut self, dt: f64) {
        let mut risen = 0.0;
        for pool in self.pools.values_mut() {
            let lost = (pool.surface_area() * EVAPORATION_M_S * dt).min(pool.volume);
            pool.volume -= lost;
            risen += lost;
        }
        self.cycle.evaporate(risen);
    }

    /// Hands the last trace of water in dried-up pools to a neighbour, a pool
    /// or seed-derived water, or else to the air, and forgets the pool.
    fn dry_up(&mut self, ground: &impl WaterGround, fed: &std::collections::BTreeSet<u32>) {
        let dry = self
            .pools
            .iter()
            .filter(|(id, pool)| pool.volume <= DRY_M3 && !fed.contains(id))
            .map(|(&id, _)| id)
            .collect::<Vec<_>>();
        for id in dry {
            let contacts = self.pools[&id].contacts.iter().copied().collect::<Vec<_>>();
            let heir = contacts
                .into_iter()
                .find_map(|cell| match self.owner.get(&cell) {
                    Some(&other) if other != id && self.pools.contains_key(&other) => {
                        Some(End::Pool(other))
                    }
                    Some(_) => None,
                    None => self
                        .implicit(ground, cell)
                        .map(|surface| End::Body(surface.body)),
                });
            let Some(pool) = self.pools.remove(&id) else {
                continue;
            };
            for cell in pool.members.keys() {
                self.owner.remove(cell);
            }
            match heir {
                Some(heir) => self.deposit_end(ground, heir, pool.volume),
                None => self.cycle.evaporate(pool.volume),
            }
        }
    }

    /// A pool that filled up to the seed-derived water it touches becomes
    /// part of it: its cells join that water and keep what they hold below
    /// its level. The rest of the pool's water, pressed up to that level,
    /// was really that water's and goes back to it.
    fn join(&mut self, ground: &impl WaterGround, id: u32, contact: WaterCell) {
        let surface = match self.joined.get(&contact) {
            Some(joined) => joined.surface,
            None => match self.ground.implicit(ground, contact) {
                Some(surface) => surface,
                None => return,
            },
        };
        let Some(pool) = self.pools.remove(&id) else {
            return;
        };
        let level = self.drawn(ground, surface).level;
        let mut spare = pool.volume;
        for (&cell, &openings) in &pool.members {
            let held = held_in(cell, openings, level);
            spare -= held;
            self.owner.remove(&cell);
            self.inlets.remove(&cell);
            self.joined.insert(cell, Joined { surface, held });
        }
        self.cycle.add(surface.body, spare);
        // Free cells around it now border the seed-derived water.
        for cell in pool.members.keys().copied().collect::<Vec<_>>() {
            for neighbour in cell.neighbours() {
                if !self.owner.contains_key(&neighbour) && !self.joined.contains_key(&neighbour) {
                    self.remeasure(ground, neighbour);
                }
            }
        }
    }

    fn drop_contact(&mut self, id: u32, cell: WaterCell) {
        if let Some(pool) = self.pools.get_mut(&id) {
            pool.contacts.remove(&cell);
        }
    }

    /// Moves the water, never taking more than a pool holds. Returns the
    /// volume moved and the pools that received water.
    fn apply(
        &mut self,
        ground: &impl WaterGround,
        transfers: &[Transfer],
    ) -> (f64, std::collections::BTreeSet<u32>) {
        let mut fed = std::collections::BTreeSet::new();
        let mut outgoing = BTreeMap::<(u8, u32), (End, f64)>::new();
        for transfer in transfers {
            outgoing
                .entry(end_key(transfer.from))
                .or_insert((transfer.from, 0.0))
                .1 += transfer.volume;
        }
        // No source gives more than it holds: a pool its volume, a lake or
        // river what it has to give.
        let scale = outgoing
            .into_iter()
            .map(|(key, (from, out))| {
                let held = match from {
                    End::Pool(id) => self.pools.get(&id).map_or(0.0, |pool| pool.volume),
                    End::Body(body) => self.cycle.available(ground, body),
                    End::Seed(_) | End::Sheet(_) => 0.0,
                };
                (key, if out > held { held / out } else { 1.0 })
            })
            .collect::<BTreeMap<_, _>>();
        let mut moved = 0.0;
        for transfer in transfers {
            let volume =
                transfer.volume * scale.get(&end_key(transfer.from)).copied().unwrap_or(1.0);
            if volume <= 0.0 {
                continue;
            }
            match transfer.from {
                End::Pool(id) => {
                    if let Some(pool) = self.pools.get_mut(&id) {
                        pool.volume -= volume;
                    }
                }
                End::Body(body) => self.cycle.add(body, -volume),
                End::Seed(_) | End::Sheet(_) => {}
            }
            if let End::Pool(id) = transfer.to {
                fed.insert(id);
            }
            self.deposit_end(ground, transfer.to, volume);
            moved += volume;
        }
        (moved, fed)
    }

    fn deposit_end(&mut self, ground: &impl WaterGround, to: End, volume: f64) {
        match to {
            End::Pool(id) => {
                if let Some(pool) = self.pools.get_mut(&id) {
                    pool.volume += volume;
                }
            }
            End::Body(body) => self.cycle.add(body, volume),
            End::Seed(cell) => {
                if let Some(&id) = self.owner.get(&cell) {
                    self.deposit_end(ground, End::Pool(id), volume);
                } else if self.stands(ground, cell, volume) {
                    self.start_pool(ground, cell, volume);
                } else {
                    self.add_sheet(ground, cell, volume);
                }
            }
            End::Sheet(cell) => self.add_sheet(ground, cell, volume),
        }
    }

    fn merge(&mut self, keep: u32, other: u32) {
        if keep == other || !self.pools.contains_key(&keep) {
            return;
        }
        let Some(other_pool) = self.pools.remove(&other) else {
            return;
        };
        for cell in other_pool.members.keys() {
            self.owner.insert(*cell, keep);
        }
        if let Some(pool) = self.pools.get_mut(&keep) {
            pool.absorb_pool(other_pool);
            pool.settle();
        }
    }

    fn deposit_at(&mut self, ground: &impl WaterGround, cell: WaterCell, volume: f64) {
        let to = self.landing(ground, cell);
        self.deposit_end(ground, to, volume);
    }

    /// Pours water in at a point: it lands in whatever lies below. Returns
    /// the volume accepted, which is all of it unless the point is inside
    /// solid ground.
    pub fn deposit(&mut self, ground: &impl WaterGround, point: DVec3, volume_m3: f64) -> f64 {
        let cell = WaterCell::containing(point);
        if volume_m3 <= 0.0 || floor_of(cell, self.openings(ground, cell)).is_none() {
            return 0.0;
        }
        self.deposit_at(ground, cell, volume_m3);
        volume_m3
    }

    /// Draws water out at a point: from the pool there, or from seed-derived
    /// water, never more than it has. Returns the volume taken.
    pub fn withdraw(&mut self, ground: &impl WaterGround, point: DVec3, volume_m3: f64) -> f64 {
        let cell = WaterCell::containing(point);
        if volume_m3 <= 0.0 {
            return 0.0;
        }
        if let Some(pool) = self.owner.get(&cell).and_then(|id| self.pools.get_mut(id)) {
            if pool.level <= cell.bottom() {
                return 0.0;
            }
            let taken = volume_m3.min(pool.volume);
            pool.volume -= taken;
            pool.settle();
            return taken;
        }
        if let Some(sheet) = self.sheets.get_mut(&cell) {
            let taken = volume_m3.min(sheet.volume);
            sheet.volume -= taken;
            return taken;
        }
        if let Some(surface) = self.implicit(ground, cell) {
            let taken = volume_m3.min(self.cycle.available(ground, surface.body));
            self.cycle.add(surface.body, -taken);
            return taken;
        }
        0.0
    }

    /// The ground changed in these bricks: pools there are re-measured, and
    /// open cells next to water start filling from it.
    pub fn terrain_changed(
        &mut self,
        ground: &impl WaterGround,
        bricks: impl IntoIterator<Item = BrickCoord>,
    ) {
        let bricks = bricks.into_iter().collect::<Vec<_>>();
        for &brick in &bricks {
            self.ground.forget(brick);
        }
        for &brick in &bricks {
            for cell in WaterCell::in_brick(brick) {
                self.remeasure(ground, cell);
            }
        }
    }

    /// Re-reads one cell's ground and lets the water around it respond.
    fn remeasure(&mut self, ground: &impl WaterGround, cell: WaterCell) {
        self.remeasure_sheet(ground, cell);
        let openings = self.openings(ground, cell);
        let floor = floor_of(cell, openings);
        if let Some(joined) = self.joined.get_mut(&cell) {
            // A joined cell holds what its ground now leaves open below its
            // water's level; the difference comes from or goes to that water.
            let surface = joined.surface;
            let level = self.cycle.shift(ground, surface.body).apply(surface).level;
            let held = held_in(cell, openings, level);
            let change = held - joined.held;
            joined.held = held;
            if floor.is_none() {
                self.joined.remove(&cell);
            }
            self.cycle.add(surface.body, -change);
            return;
        }
        if let Some(&id) = self.owner.get(&cell) {
            if floor.is_none() {
                self.owner.remove(&cell);
                if let Some(pool) = self.pools.get_mut(&id) {
                    pool.remove_member(cell);
                    pool.settle();
                }
            } else if let Some(pool) = self.pools.get_mut(&id) {
                pool.add_member(cell, openings);
                pool.settle();
            }
            self.border(ground, id, cell);
            return;
        }
        let Some(floor) = floor else {
            return;
        };
        if self.implicit(ground, cell).is_some() {
            return;
        }
        for neighbour in cell.neighbours() {
            if let Some(&id) = self.owner.get(&neighbour) {
                if let Some(pool) = self.pools.get_mut(&id) {
                    pool.queue(cell, floor);
                }
            } else if self
                .implicit(ground, neighbour)
                .is_some_and(|surface| surface.level > floor + FILM_METRES)
            {
                self.inlets.insert(cell);
            }
        }
    }

    /// Seed-derived water pouring into the free cells beside it, which lands
    /// wherever it falls.
    fn pour_inlets(&mut self, ground: &impl WaterGround, dt: f64, transfers: &mut Vec<Transfer>) {
        let inlets = self.inlets.iter().copied().collect::<Vec<_>>();
        // Room left this step in each pool and each cell a pool starts in:
        // inlets fill a pool at most halfway to their water's level, the
        // other half being its own contacts' share, and a new pool no more
        // than its first cell holds.
        let mut room = HashMap::<End, f64>::new();
        for cell in inlets {
            if self.owner.contains_key(&cell) {
                // Its pool exchanges with the water beside it directly.
                continue;
            }
            let Some(floor) = floor_of(cell, self.openings(ground, cell)) else {
                self.inlets.remove(&cell);
                continue;
            };
            let mut source = None;
            for neighbour in cell.neighbours() {
                if let Some(surface) = self.implicit(ground, neighbour) {
                    let lip = floor_of(neighbour, self.openings(ground, neighbour))
                        .map_or(floor, |other| other.max(floor));
                    if source
                        .is_none_or(|(_, best): (WaterSurface, f64)| surface.level - lip > best)
                    {
                        source = Some((surface, surface.level - lip));
                    }
                }
            }
            let Some((surface, head)) = source else {
                self.inlets.remove(&cell);
                continue;
            };
            if head <= FILM_METRES {
                continue;
            }
            let to = self.landing(ground, cell);
            let left = match to {
                End::Body(_) => continue,
                End::Pool(id) => room.entry(to).or_insert_with(|| {
                    self.pools.get(&id).map_or(0.0, |pool| {
                        let area = pool.surface_area().max(WATER_CELL_METRES.powi(2));
                        0.5 * (surface.level - pool.level).max(0.0) * area
                    })
                }),
                End::Seed(seed) | End::Sheet(seed) => {
                    let openings = self.openings(ground, seed);
                    room.entry(to).or_insert_with(|| {
                        f64::from(openings.iter().map(|&open| u32::from(open)).sum::<u32>())
                            * FINE_VOLUME_M3
                    })
                }
            };
            let volume = (WEIR_COEFFICIENT * WATER_CELL_METRES * head.powf(1.5) * dt).min(*left);
            if volume <= 0.0 {
                continue;
            }
            *left -= volume;
            transfers.push(Transfer {
                from: End::Body(surface.body),
                to,
                volume,
            });
        }
    }
}

#[cfg(test)]
mod tests;
