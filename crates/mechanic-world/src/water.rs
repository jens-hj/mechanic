//! Stored water: pools that fill, spill, drain and merge as the ground
//! changes. See `docs/water.md`.
//!
//! Water is held in 20 cm water cells, four terrain cells to an edge. All
//! stored water lies in pools. A pool is one level and one volume over its
//! member cells, and grows by priority flood: the lowest neighbour whose floor
//! the water covers is taken in next. A neighbour it cannot take in is a
//! contact: another pool's cell, seed-derived water, or a drop the water
//! falls over. Across a contact the higher water runs to the lower at a weir
//! rate, and pools whose levels meet merge. Water that falls lands at once in
//! whatever lies below it, or starts a pool there. So a U-tube is one pool and settles level, a pit
//! fills and then spills at its lowest rim, and a trench dug from a lake fills
//! from it. Water taken from a lake lowers the whole lake, and every cubic
//! metre is booked: see [`cycle`] for where it comes from and goes.

mod cells;
mod cycle;
mod grid;
mod ground;
mod pool;
mod sediment;
mod sheet;
mod soil;
mod splash;
mod surface;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use bevy_math::{DVec2, DVec3, IVec3};
use serde::{Deserialize, Serialize};

use cells::CellMap;
use cycle::Cycle;
pub use cycle::{SurplusDoc, WaterLedger, WaterNetwork, WaterShift};
use grid::SheetGrid;
use ground::{BRICK_EDGE_WATER_CELLS, Openings, OpeningsCache};
pub use ground::{TerrainWater, WATER_CELL_EDGE_CELLS, WaterGround};
use pool::Pool;
pub use sediment::{BedDoc, ErosionConfig, SedimentDoc, SedimentLedger, SedimentLoad};
use sediment::{Placed, Sediment};
pub use sheet::{RunningView, SheetDoc};
pub use soil::{SoilDoc, WetGround};
pub use surface::{SURFACE_TILE_COLUMNS, StoredSurface, SurfaceTile};

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

/// A pool takes in dry ground at once only where its water would stand at
/// least this deep over it, in metres: deep water levels out at once, while
/// over shallower ground friction holds its front back.
const DEEP_METRES: f64 = 0.3;

/// How fast a pool's edge spreads over dry ground it stands shallow on, in
/// metres a second: the pace of a flood a few centimetres deep over grass,
/// rather than a whole flat filling at the pool's level in one step.
const FRONT_M_S: f64 = 0.25;

/// How far over a cell, in water cells, ground closing over it counts as its
/// roof: 8 m, over the tallest cave a pool fills.
const ROOF_CELLS: i32 = 40;

/// Water shallower than this clings to the ground and does not run, and
/// water less than this over a lip does not pour over it, in metres.
const CLING_METRES: f64 = 0.002;

/// Pools whose levels lie this close, in metres, merge where they touch.
const MERGE_METRES: f64 = 0.005;

/// Weir coefficient for water running over a cell's edge, in m^½/s: a
/// contact passes `WEIR × width × head^1.5` cubic metres a second.
const WEIR_COEFFICIENT: f64 = 1.7;

/// Cells one pool takes in per step: how fast a front of water advances.
const SPREAD_CELLS_PER_STEP: usize = 256;

/// Deepest a falling stream is followed, in water cells.
const FALL_CELLS: i32 = 2_000;

/// Deepest ground water pressed out of filled ground rises through to open
/// space, in water cells; beyond, it seeps away into the air.
const ROOFED_CELLS: i32 = 64;

/// Gravity, in m/s².
const GRAVITY: f64 = 9.81;

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

/// Height of the ground in a cell counted from its terrain cells, where the
/// cell has any opening: the mean top of its ground over the column, up to
/// its lowest wholly open layer. Unlike [`floor_of`] it rises as ground fills
/// a layer, but by whole terrain cells; running water rests on the drawn
/// ground near it instead.
fn ground_height(cell: WaterCell, openings: Openings) -> Option<f64> {
    floor_of(cell, openings)?;
    let full = u8::try_from(WATER_CELL_EDGE_CELLS * WATER_CELL_EDGE_CELLS).expect("16 cells");
    let mut height = cell.bottom();
    for open in openings {
        if open == full {
            break;
        }
        height += FINE_LAYER_METRES * f64::from(full - open) / f64::from(full);
    }
    Some(height)
}

/// What one water step did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WaterStep {
    /// Water moved between pools, seed-derived water and running water, in
    /// m³.
    pub moved_m3: f64,
    /// Pools after the step.
    pub pools: usize,
    /// Cells held by pools after the step.
    pub cells: usize,
    /// Cells of running water after the step.
    pub sheet_cells: usize,
    /// Where the step's time went.
    pub phases: WaterPhases,
}

/// Time one water step spent in each of its phases, in milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WaterPhases {
    /// Pools taking in the ground their water covers.
    pub flood_ms: f64,
    /// Water crossing between pools, seed-derived water and inlets.
    pub exchange_ms: f64,
    /// Running water.
    pub sheets_ms: f64,
    /// Pools joining seed-derived water and merging.
    pub joins_ms: f64,
    /// Settling, evaporation, the cycle and drying up.
    pub settle_ms: f64,
}

/// Milliseconds since the last lap.
struct PhaseClock(std::time::Instant);

impl PhaseClock {
    fn new() -> Self {
        Self(std::time::Instant::now())
    }

    fn lap(&mut self) -> f64 {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.0).as_secs_f64() * 1000.0;
        self.0 = now;
        elapsed
    }
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
    /// How cloudy with sediment it is, from 0 to 1.
    pub murk: f64,
}

/// One stored pool in a saved world.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoolDoc {
    /// The cell it floods out from again when the world loads.
    pub seed: WaterCell,
    /// Water held, in m³.
    pub volume_m3: f64,
    /// Sediment it carries.
    #[serde(default)]
    pub load: SedimentLoad,
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
    /// Water held in the ground.
    pub soil: Vec<SoilDoc>,
    /// Sediment waiting to be laid, and what erosion has moved.
    pub sediment: SedimentDoc,
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

/// Keeps one pour of the transfers from `first` on into each column of
/// running water, the largest: a pool touching one column at several cells
/// up its height pours into it once, since the column is one water, and a
/// pour per cell would stack it far over the pool.
fn pour_once_per_column(transfers: &mut Vec<Transfer>, first: usize) {
    let mut poured = CellMap::<(i32, i32), usize>::default();
    for transfer in transfers.split_off(first) {
        if let End::Sheet(cell) = transfer.to {
            if let Some(&kept) = poured.get(&(cell.x, cell.z)) {
                let kept: &mut Transfer = &mut transfers[kept];
                kept.volume = kept.volume.max(transfer.volume);
                continue;
            }
            poured.insert((cell.x, cell.z), transfers.len());
        }
        transfers.push(transfer);
    }
}

/// What one step's contacts do.
#[derive(Debug, Default)]
struct Exchanges {
    transfers: Vec<Transfer>,
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
    owner: CellMap<WaterCell, u32>,
    ground: OpeningsCache,
    /// The rivers, lakes, sea and air around stored water.
    cycle: Cycle,
    /// Free cells seed-derived water pours into.
    inlets: std::collections::BTreeSet<WaterCell>,
    /// Cells a pool filled up to the seed-derived water beside it, which then
    /// became part of that water.
    joined: CellMap<WaterCell, Joined>,
    /// Running water, by cell.
    sheets: SheetGrid,
    /// Water held in the ground, by column.
    soil: CellMap<(i32, i32), soil::Soil>,
    /// The set of ground columns that wicks next.
    wick_set: u32,
    /// Height of the drawn ground at the centre of each column water wicked
    /// towards, by column and the water cell height it was sought from.
    wick_tops: CellMap<(i32, i32, i32), Option<f64>>,
    /// Height of the drawn ground at each surface corner met, by corner and
    /// the water cell height it was sought from.
    tops: CellMap<(i32, i32, i32), Option<f64>>,
    /// Floor under seed-derived water beside stored water, by column and the
    /// water cell height its surface stands at.
    floors: CellMap<(i32, i32, i32), Option<f64>>,
    /// Whether any of each water-cell column met lies in a lake's or river's
    /// reach, which the seed alone decides.
    reaches: CellMap<(i32, i32), bool>,
    /// Where each moved lake and river is drawn: where it stands, as of the
    /// last time it moved further than [`SHOWN_DROP_METRES`] from here. The
    /// lake's own sheet and the stored water meeting it are drawn at this
    /// one level, so neither shows a step against the other.
    shown: BTreeMap<WaterBody, WaterShift>,
    /// Sediment beside the water.
    sediment: Sediment,
    /// Power of the falls landing on each column, in watts, to draw.
    splashes: CellMap<(i32, i32), f64>,
    /// Energy landed on each column so far this step, in joules.
    landed: CellMap<(i32, i32), f64>,
}

/// How far a lake or river moves from where it is drawn before it is drawn
/// again where it stands, in metres: its sheet is meshed again each time.
const SHOWN_DROP_METRES: f64 = 0.02;

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
            // Seed-derived water that covers the cell anyway holds its water.
            if water.ground.implicit(ground, joined.cell).is_some() {
                water.cycle.add(joined.body, joined.held_m3);
                continue;
            }
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
            let placed = water.deposit_at(ground, pool.seed, pool.volume_m3);
            let near = (pool.seed.x, pool.seed.z, pool.seed.bottom());
            water.give_load(placed, pool.load, near);
        }
        for sheet in &doc.sheets {
            water.add_sheet(ground, sheet.cell, sheet.volume_m3);
            let (cell, load) = (sheet.cell, sheet.load);
            water.give_load(
                Placed::Sheet(cell.x, cell.z),
                load,
                (cell.x, cell.z, cell.bottom()),
            );
        }
        water.load_soil(&doc.soil);
        water.load_sediment(&doc.sediment);
        let ids = water.pools.keys().copied().collect::<Vec<_>>();
        for id in ids {
            // A saved pool fills out to its level again at once.
            if let Some(pool) = water.pools.get_mut(&id) {
                pool.front = f64::INFINITY;
            }
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
                    load: pool.load,
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
            soil: self.soil_docs(),
            sediment: self.sediment_doc(),
        }
    }

    /// Water held in pools, in m³.
    pub fn stored_m3(&self) -> f64 {
        self.pools.values().map(|pool| pool.volume).sum()
    }

    /// Cells that filled from seed-derived water and joined it, with the
    /// water's surface where it is drawn.
    pub fn joined_cells(&self) -> Vec<(WaterCell, WaterSurface)> {
        let mut joined = self
            .joined
            .iter()
            .map(|(&cell, joined)| (cell, cycle::shifted(&self.shown, joined.surface)))
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
            soil_m3: self.soil_m3(),
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
            .filter_map(|(cell, sheet)| Some((cell, sheet.surface()?)))
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
                murk: sediment::murk(pool.load, pool.volume),
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

    /// Where each moved lake and river is drawn.
    pub const fn shown_shifts(&self) -> &BTreeMap<WaterBody, WaterShift> {
        &self.shown
    }

    /// Draws each lake and river that moved further than
    /// [`SHOWN_DROP_METRES`] from where it is drawn where it now stands.
    fn show_shifts(&mut self, ground: &impl WaterNetwork) {
        let live = self.cycle.shifts(ground);
        let drop = |shifts: &BTreeMap<WaterBody, WaterShift>, body| {
            shifts
                .get(body)
                .map_or(0.0, |shift: &WaterShift| shift.drop)
        };
        let moved = live
            .keys()
            .chain(self.shown.keys())
            .any(|body| (drop(&live, body) - drop(&self.shown, body)).abs() > SHOWN_DROP_METRES);
        if moved {
            self.shown = live;
        }
    }

    fn openings(&mut self, ground: &impl WaterGround, cell: WaterCell) -> Openings {
        self.ground.openings(ground, cell)
    }

    /// Whether ground closes over a cell within [`ROOF_CELLS`]: the roof of
    /// a cave, a tunnel or an overhang, however high its water has risen.
    fn roofed(&mut self, ground: &impl WaterGround, cell: WaterCell) -> bool {
        let whole = u32::try_from(WATER_CELL_EDGE_CELLS.pow(3)).expect("64 cells");
        (1..=ROOF_CELLS).any(|up| {
            let above = WaterCell::new(cell.x, cell.y + up, cell.z);
            let open = self
                .openings(ground, above)
                .iter()
                .map(|&open| u32::from(open))
                .sum::<u32>();
            open * 2 < whole
        })
    }

    /// Height of the ground running water rests on in a cell, where the cell
    /// has any opening.
    fn sheet_floor(&mut self, ground: &impl WaterGround, cell: WaterCell) -> Option<f64> {
        self.ground.floor(ground, cell)
    }

    fn implicit(&mut self, ground: &impl WaterGround, cell: WaterCell) -> Option<WaterSurface> {
        let surface = match self.joined.get(&cell) {
            Some(joined) => joined.surface,
            None => self.ground.implicit(ground, cell)?,
        };
        let surface = self.drawn(ground, surface);
        (floor_of(cell, self.openings(ground, cell))? < surface.level).then_some(surface)
    }

    /// Whether a pool at `level` reaches a cell at once: running water is
    /// there already, the pool would stand deep over the ground under it, or
    /// the cell was dug. Trenches, pits and tunnels the player digs fill at
    /// the level of the water let into them; the natural flats around them
    /// flood only as fast as a front crosses them.
    fn reaches(&mut self, ground: &impl WaterGround, cell: WaterCell, level: f64) -> bool {
        ground.dug(cell)
            || self
                .sheet_surface(ground, cell)
                .is_some_and(|(_, depth)| depth > FILM_METRES)
            || self.deep(ground, cell, level)
    }

    /// Whether water at `level` would stand deep over the ground under a
    /// cell: at least `DEEP_METRES` over the floor of the lowest cell open to
    /// it straight below, whatever water already lies there.
    fn deep(&mut self, ground: &impl WaterGround, mut cell: WaterCell, level: f64) -> bool {
        let Some(mut floor) = floor_of(cell, self.openings(ground, cell)) else {
            return false;
        };
        while floor > level - DEEP_METRES {
            let below = cell.below();
            let lower = floor_of(below, self.openings(ground, below));
            if self.openings(ground, cell)[0] == 0
                || self.openings(ground, below)[3] == 0
                || lower.is_none()
            {
                return false;
            }
            (cell, floor) = (below, lower.unwrap_or(floor));
        }
        true
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
        // Running water standing up to the cell is no drop either.
        if self
            .sheets
            .get(below)
            .is_some_and(|sheet| sheet.surface_height() >= cell.bottom())
        {
            return false;
        }
        self.implicit(ground, below)
            .is_none_or(|surface| surface.level < cell.bottom())
    }

    /// The surface of running water in a cell's column whose floor lies
    /// below the cell, if any.
    fn running_under(&self, cell: WaterCell) -> Option<f64> {
        let sheet = self.sheets.at(self.sheets.slot(cell.x, cell.z)?);
        (sheet.present && sheet.y < cell.y).then(|| sheet.surface_height())
    }

    /// Where water entering a cell ends up, following it down any drop.
    fn landing(&mut self, ground: &impl WaterGround, cell: WaterCell) -> End {
        self.landing_cell(ground, cell).0
    }

    /// Where water entering a cell ends up, and the cell it lands in.
    fn landing_cell(&mut self, ground: &impl WaterGround, mut cell: WaterCell) -> (End, WaterCell) {
        for _ in 0..FALL_CELLS {
            if let Some(&id) = self.owner.get(&cell) {
                return (End::Pool(id), cell);
            }
            if let Some(surface) = self.implicit(ground, cell) {
                return (End::Body(surface.body), cell);
            }
            let below = cell.below();
            if !self.falls(ground, cell, None) {
                // Water resting on water joins it.
                if let Some(&id) = self.owner.get(&below) {
                    return (End::Pool(id), below);
                }
                return (End::Seed(cell), cell);
            }
            cell = below;
        }
        (End::Seed(cell), cell)
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

    /// A pool begun from running water stands no higher than that water
    /// did: running water holds its volume as if its cell were open right
    /// across, a pool only in what the ground leaves open, so a sheet in a
    /// sliver of a cell turns into a pool standing far over it. What the
    /// pool holds over `height` runs on into the lowest running water beside
    /// it, where it ran all along.
    fn shed_over(&mut self, ground: &impl WaterGround, id: u32, height: f64) {
        let Some(pool) = self.pools.get(&id) else {
            return;
        };
        let over = pool.volume - pool.held_below(height);
        if over <= 0.0 {
            return;
        }
        let contacts = pool.contacts.iter().copied().collect::<Vec<_>>();
        let lowest = contacts
            .into_iter()
            .filter_map(|cell| {
                let slot = self.sheets.slot(cell.x, cell.z)?;
                let sheet = self.sheets.at(slot);
                (sheet.present && sheet.y <= cell.y && sheet.surface_height() < height)
                    .then(|| (cell, sheet.surface_height()))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((cell, _)) = lowest {
            let mut load = SedimentLoad::default();
            if let Some(pool) = self.pools.get_mut(&id) {
                load = pool.load.part(over, pool.volume);
                pool.volume -= over;
                pool.settle();
            }
            self.add_sheet(ground, cell, over);
            self.give_load(
                Placed::Sheet(cell.x, cell.z),
                load,
                (cell.x, cell.z, cell.bottom()),
            );
        }
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
    /// `budget` cells. Dry ground the water would stand shallow on it takes
    /// in a ring at a time, as fast as its front may spread.
    fn flood(&mut self, ground: &impl WaterGround, id: u32, mut budget: usize) {
        let mut fresh = HashSet::new();
        let mut waiting = Vec::new();
        let mut spread = false;
        let finished = loop {
            let Some(pool) = self.pools.get_mut(&id) else {
                return;
            };
            pool.settle();
            if budget == 0 {
                break false;
            }
            let Some(cell) = pool.next_below(pool.level - FILM_METRES) else {
                break true;
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
            let Some((rim, level)) = self.pools.get(&id).map(|pool| (pool.rim, pool.level)) else {
                return;
            };
            // Under open sky water runs: a pool meets the cells beside it
            // there at its edge, through the running water's pipes, and
            // rises only straight up over its own water.
            let open_beside = !self.pools.get(&id).is_some_and(|pool| pool.over_own(cell))
                && !self.roofed(ground, cell);
            // Running water filling the cell up from a cell below is taken in
            // under a roof with every cell it fills, where the pool may take
            // them all; elsewhere the pool meets it at its edge, through its
            // pipes. Either way the pool never rests on it, both holding the
            // water between them.
            let (under, spill) = self
                .sheets
                .covering(cell)
                .filter(|sheet| sheet.y < cell.y)
                .map_or((None, false), |sheet| (Some(sheet.y), sheet.floor() < rim));
            let over_running = under.is_some_and(|under| {
                // Running water from below the rim is the pool's spill.
                open_beside
                    || spill
                    || (under..cell.y).any(|y| {
                        let filled = WaterCell::new(cell.x, y, cell.z);
                        self.owner.contains_key(&filled)
                            || floor_of(filled, self.openings(ground, filled)).is_none()
                    })
            });
            if floor < rim
                || self.owner.contains_key(&cell)
                || self.implicit(ground, cell).is_some()
                || self.falls(ground, cell, Some(id))
                || open_beside
                || over_running
            {
                if let Some(pool) = self.pools.get_mut(&id) {
                    pool.contacts.insert(cell);
                }
                continue;
            }
            // Over dry ground it would stand shallow on, the water spreads
            // as a front: a ring of cells beside those it held before, once
            // the front has had time to cross a cell.
            if !self.reaches(ground, cell, level) {
                let Some(pool) = self.pools.get(&id) else {
                    return;
                };
                let ready = pool.front.is_infinite()
                    || (pool.front >= WATER_CELL_METRES
                        && cell.neighbours().into_iter().any(|neighbour| {
                            pool.members.contains_key(&neighbour) && !fresh.contains(&neighbour)
                        }));
                if !ready {
                    waiting.push((cell, floor));
                    continue;
                }
                spread = true;
            }
            for y in under.unwrap_or(cell.y)..=cell.y {
                let filled = WaterCell::new(cell.x, y, cell.z);
                let openings = self.openings(ground, filled);
                let Some(floor) = floor_of(filled, openings) else {
                    continue;
                };
                if let Some(pool) = self.pools.get_mut(&id) {
                    // Only ground the water crossed makes a rim, not the bottom
                    // of a cell over its own water.
                    if !pool.over_own(filled) {
                        pool.rim = pool.rim.max(floor);
                    }
                    pool.add_member(filled, openings);
                }
                fresh.insert(filled);
                self.owner.insert(filled, id);
                self.absorb_sheet(id, filled);
                self.border(ground, id, filled);
            }
            budget -= 1;
        };
        let Some(pool) = self.pools.get_mut(&id) else {
            return;
        };
        for (cell, floor) in waiting {
            pool.queue(cell, floor);
        }
        if spread && pool.front.is_finite() {
            pool.front -= WATER_CELL_METRES;
        }
        // A reloaded pool fills out to its level at once; from then on it
        // spreads over dry flats at the pace of a front.
        if finished && pool.front.is_infinite() {
            pool.front = 0.0;
        }
    }

    /// Runs the water for `dt` seconds.
    pub fn step(&mut self, ground: &impl WaterGround, dt: f64) -> WaterStep {
        let mut clock = PhaseClock::new();
        let mut phases = WaterPhases::default();
        let ids = self.pools.keys().copied().collect::<Vec<_>>();
        for pool in self.pools.values_mut() {
            if pool.front.is_finite() {
                pool.front = (pool.front + FRONT_M_S * dt).min(WATER_CELL_METRES);
            }
        }
        for &id in &ids {
            self.flood(ground, id, SPREAD_CELLS_PER_STEP);
        }
        phases.flood_ms = clock.lap();
        let mut out = Exchanges::default();
        for &id in &ids {
            self.exchange(ground, id, dt, &mut out);
        }
        self.pour_inlets(ground, dt, &mut out.transfers);
        let (mut moved_m3, mut fed) = self.apply(ground, &out.transfers);
        phases.exchange_ms = clock.lap();
        let (running, sheet_fed) = self.step_sheets(ground, dt);
        moved_m3 += running;
        fed.extend(sheet_fed);
        self.wear_and_settle(ground, dt);
        self.settle_splashes(dt);
        phases.sheets_ms = clock.lap();
        // Water that arrived floods at once, so no pool stands higher than
        // its water can reach.
        for &id in &fed {
            self.flood(ground, id, SPREAD_CELLS_PER_STEP);
        }
        phases.flood_ms += clock.lap();
        // Pools join seed-derived water before they merge with each other,
        // so two pools under a lake join it rather than pool what the lake
        // presses up into them.
        for (id, contact) in out.joins {
            self.join(ground, id, contact);
        }
        for (keep, other) in out.merges {
            self.merge(keep, other);
        }
        phases.joins_ms = clock.lap();
        self.settle_sheets(ground);
        self.evaporate(dt);
        self.evaporate_sheets(EVAPORATION_M_S, dt);
        self.soak(ground, dt);
        self.cycle.step(ground, dt);
        self.dry_up(ground, &fed);
        for pool in self.pools.values_mut() {
            pool.settle();
        }
        phases.settle_ms = clock.lap();
        WaterStep {
            moved_m3,
            pools: self.pools.len(),
            cells: self.owner.len(),
            sheet_cells: self.sheets.len(),
            phases,
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
        let first = out.transfers.len();
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
                && (level > surface.level - MERGE_METRES
                    || (full && surface.level > top && cell.bottom() >= top - FILM_METRES))
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
                self.fall(ground, id, cell, [level, floor, area, top], dt, out);
            } else if own_floor >= rim
                && !self.pools[&id].over_own(cell)
                && !self.roofed(ground, cell)
            {
                out.transfers
                    .extend(self.open_beside(id, cell, [level.min(top), floor], dt));
            } else if own_floor < rim {
                let spill = self.spill(ground, id, cell, [level.min(top), floor, area], dt);
                out.transfers.extend(spill);
            } else {
                self.drop_contact(id, cell);
                if let Some(pool) = self.pools.get_mut(&id) {
                    pool.queue(cell, floor);
                }
            }
        }
        pour_once_per_column(&mut out.transfers, first);
        for (mut transfer, even) in levelling.into_values() {
            transfer.volume = transfer.volume.min(even);
            out.transfers.push(transfer);
        }
    }

    /// What a pool standing at `level`, no higher than its cells' top, spills
    /// past its rim onto a cell over the lip `floor`, with its surface `area`:
    /// beyond the rim the water runs off as running water, over any already
    /// running there. A pool fuller than its cells pours by no more head than
    /// its cells hold, as over a drop.
    fn spill(
        &mut self,
        ground: &impl WaterGround,
        id: u32,
        cell: WaterCell,
        [level, floor, area]: [f64; 3],
        dt: f64,
    ) -> Option<Transfer> {
        let below = self
            .sheet_surface(ground, cell)
            .map_or(floor, |(height, _)| height.max(floor));
        let head = level - below;
        (head > 0.0).then(|| Transfer {
            from: End::Pool(id),
            to: End::Sheet(cell),
            volume: (WEIR_COEFFICIENT * WATER_CELL_METRES * head.powf(1.5) * dt).min(head * area),
        })
    }

    /// What a pool pours onto a cell beside it under open sky, standing at
    /// `level` (no higher than its cells' top) over the cell's `floor`.
    /// Running water there trades with the pool through its pipes; dry
    /// ground gets running water to start with, at most what brings the two
    /// halfway level over its one column.
    fn open_beside(
        &self,
        id: u32,
        cell: WaterCell,
        [level, floor]: [f64; 2],
        dt: f64,
    ) -> Option<Transfer> {
        let running = self.sheets.slot(cell.x, cell.z).is_some_and(|slot| {
            let sheet = self.sheets.at(slot);
            sheet.present && sheet.y <= cell.y
        });
        let head = level - floor;
        if running || head <= 0.0 {
            return None;
        }
        let pool = self.pools.get(&id)?;
        let volume = (WEIR_COEFFICIENT * WATER_CELL_METRES * head.powf(1.5) * dt)
            .min(0.5 * head * WATER_CELL_METRES.powi(2))
            .min(pool.volume - pool.held_below(floor));
        (volume > 0.0).then_some(Transfer {
            from: End::Pool(id),
            to: End::Sheet(cell),
            volume,
        })
    }

    /// Water spilling from a pool over a drop at `cell`, with the pool's
    /// level, the lip's floor, the pool's surface area and the top of its
    /// cells.
    fn fall(
        &mut self,
        ground: &impl WaterGround,
        id: u32,
        cell: WaterCell,
        [level, floor, area, top]: [f64; 4],
        dt: f64,
        out: &mut Exchanges,
    ) {
        let weir = |head: f64| WEIR_COEFFICIENT * WATER_CELL_METRES * head.max(0.0).powf(1.5) * dt;
        if let Some(below) = self.running_under(cell) {
            // Running water already down the drop, however deep its floor,
            // is the same water: the pool pours into it over its weir, by its
            // head over that water, and stops at its own level: it does not
            // pour into whichever column of a channel stands low and rock it
            // in a standing wave metres high. A pool fuller than its cells
            // presses nothing up.
            let head = level.min(top) - below.max(floor);
            if head > 0.0 {
                let volume = weir(head).min(head * area);
                self.splash((cell.x, cell.z), volume, floor - below);
                out.transfers.push(Transfer {
                    from: End::Pool(id),
                    to: End::Sheet(cell),
                    volume,
                });
            }
            return;
        }
        // A pool fuller than its cells pours by no more head than its cells
        // hold: its surplus presses water up out of the ground, not over a
        // lip beside it metres at a time.
        let head = level.min(top) - floor;
        // Water a millimetre over a lip clings to it. The rest lands at once
        // wherever the drop leads.
        if head > CLING_METRES {
            let (to, landed) = self.landing_cell(ground, cell);
            let volume = weir(head).min(head * area);
            self.splash((cell.x, cell.z), volume, cell.bottom() - landed.bottom());
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
            let placed = if let Some(heir) = heir {
                self.deposit_end(ground, heir, pool.volume)
            } else {
                self.cycle.evaporate(pool.volume);
                Placed::Gone
            };
            // What it carried goes on into a pool that takes its water, and
            // otherwise settles on its bed.
            if let Placed::Pool(heir) = placed
                && let Some(heir) = self.pools.get_mut(&heir)
            {
                heir.load.add(pool.load);
            } else {
                self.settle_pool_load(&pool, pool.load);
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
        self.settle_pool_load(&pool, pool.load);
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
            // Water a pool sends on to stored water carries its share of
            // what the pool carries; into seed-derived water it leaves it.
            let mut load = SedimentLoad::default();
            let mut near = (0, 0, 0.0);
            match transfer.from {
                End::Pool(id) => {
                    if let Some(pool) = self.pools.get_mut(&id) {
                        if !matches!(transfer.to, End::Body(_)) {
                            load = pool.load.part(volume, pool.volume);
                        }
                        pool.volume -= volume;
                        near = (pool.seed.x, pool.seed.z, pool.seed.bottom());
                    }
                }
                End::Body(body) => self.cycle.add(body, -volume),
                End::Seed(_) | End::Sheet(_) => {}
            }
            if let End::Pool(id) = transfer.to {
                fed.insert(id);
            }
            if let End::Seed(cell) | End::Sheet(cell) = transfer.to {
                near = (cell.x, cell.z, cell.bottom());
            }
            let placed = self.deposit_end(ground, transfer.to, volume);
            self.give_load(placed, load, near);
            moved += volume;
        }
        (moved, fed)
    }

    /// Puts water down at one end of a transfer. Returns where it went.
    fn deposit_end(&mut self, ground: &impl WaterGround, to: End, volume: f64) -> Placed {
        match to {
            End::Pool(id) => {
                if let Some(pool) = self.pools.get_mut(&id) {
                    pool.volume += volume;
                    return Placed::Pool(id);
                }
                Placed::Gone
            }
            End::Body(body) => {
                self.cycle.add(body, volume);
                Placed::Gone
            }
            End::Seed(cell) => {
                if let Some(&id) = self.owner.get(&cell) {
                    self.deposit_end(ground, End::Pool(id), volume)
                } else if floor_of(cell, self.openings(ground, cell)).is_none() {
                    // Water set down in solid ground rises out of it.
                    self.deposit_at(ground, cell, volume)
                } else if self.sheets.covering(cell).is_some() {
                    // Water landing in running water is that water's: a pool
                    // begun there would rest on it, both holding the water
                    // between them.
                    self.add_sheet(ground, cell, volume);
                    Placed::Sheet(cell.x, cell.z)
                } else if self.stands(ground, cell, volume) {
                    Placed::Pool(self.start_pool(ground, cell, volume))
                } else {
                    self.add_sheet(ground, cell, volume);
                    Placed::Sheet(cell.x, cell.z)
                }
            }
            End::Sheet(cell) => {
                self.add_sheet(ground, cell, volume);
                Placed::Sheet(cell.x, cell.z)
            }
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

    fn deposit_at(
        &mut self,
        ground: &impl WaterGround,
        mut cell: WaterCell,
        volume: f64,
    ) -> Placed {
        // Water pressed out of ground that filled in rises to the first open
        // space over it: it never stands inside solid ground, where a pool
        // with no room would draw from the water it touches for ever.
        let mut climbed = 0;
        while floor_of(cell, self.openings(ground, cell)).is_none() {
            if climbed == ROOFED_CELLS {
                self.cycle.evaporate(volume);
                return Placed::Gone;
            }
            cell = cell.up();
            climbed += 1;
        }
        let to = self.landing(ground, cell);
        self.deposit_end(ground, to, volume)
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
        if let Some(sheet) = self.sheets.covering_mut(cell) {
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
        // The drawn ground moved in these bricks' columns, and beside them,
        // where the mesh's last cells read their samples.
        let changed = bricks
            .iter()
            .map(|brick| (brick.x, brick.z))
            .collect::<std::collections::HashSet<_>>();
        if !changed.is_empty() {
            let kept = |&(x, _, z): &(i32, i32, i32), _: &mut Option<f64>| {
                let (bx, bz) = (
                    x.div_euclid(BRICK_EDGE_WATER_CELLS),
                    z.div_euclid(BRICK_EDGE_WATER_CELLS),
                );
                !(-1..=1).any(|dx| (-1..=1).any(|dz| changed.contains(&(bx + dx, bz + dz))))
            };
            self.tops.retain(kept);
            self.floors.retain(kept);
            self.wick_tops.retain(kept);
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
                    // Filled in whole: its water rises out of the ground.
                    if pool.members.is_empty() {
                        let (volume, load) = (pool.volume, pool.load);
                        self.pools.remove(&id);
                        let placed = self.deposit_at(ground, cell.up(), volume);
                        self.give_load(placed, load, (cell.x, cell.z, cell.bottom()));
                        return;
                    }
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
            } else if ground.dug(cell)
                && self
                    .implicit(ground, neighbour)
                    .is_some_and(|surface| surface.level > floor + FILM_METRES)
            {
                // Seed-derived water pours in, and so do the cells joined to
                // it, which only ever lie under it: a pit dug under a lake's
                // edge is lake, and pours into the rest of the pit. It pours
                // only into dug ground: a natural shore beside it, even in
                // an edited brick, is as the seed left it, and pouring there
                // would feed land all along the shoreline.
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
            if floor_of(cell, self.openings(ground, cell)).is_none() {
                self.inlets.remove(&cell);
                continue;
            }
            // Water already running in the cell raises the lip the inlet
            // pours over: it never fills the cell above its source.
            let floor = self
                .sheet_surface(ground, cell)
                .map_or(f64::INFINITY, |(height, _)| height);
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
            if head <= CLING_METRES {
                continue;
            }
            let mut to = self.landing(ground, cell);
            // Inlets into one column of running water share its room.
            if let End::Seed(seed) | End::Sheet(seed) = to
                && let Some(slot) = self.sheets.slot(seed.x, seed.z)
                && self.sheets.at(slot).present
            {
                to = End::Sheet(WaterCell::new(seed.x, self.sheets.at(slot).y, seed.z));
            }
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
                    // Running water in the column fills towards the source's
                    // level however deep it grows, half the way per step, up
                    // to where seed-derived water over it begins.
                    let running = self
                        .sheets
                        .slot(seed.x, seed.z)
                        .map(|slot| *self.sheets.at(slot))
                        .filter(|sheet| sheet.present);
                    let ceiling = running.map(|sheet| {
                        let mut above = WaterCell::new(seed.x, sheet.top() + 1, seed.z);
                        for _ in 0..16 {
                            if self.implicit(ground, above).is_some() {
                                break;
                            }
                            above = above.up();
                        }
                        above.bottom()
                    });
                    room.entry(to).or_insert_with(|| match (running, ceiling) {
                        (Some(sheet), Some(ceiling)) => {
                            0.5 * (surface.level.min(ceiling) - sheet.surface_height()).max(0.0)
                                * WATER_CELL_METRES.powi(2)
                        }
                        _ => held_in(seed, openings, surface.level),
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
