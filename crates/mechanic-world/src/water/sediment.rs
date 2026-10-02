//! Sediment: running water wears soft ground away and carries it, and lays
//! it down again where the water slows.
//!
//! Water dragging over its bed with a shear stress beyond what the ground
//! holds against takes it up at a rate in proportion to the excess
//! (Partheniades): sand gives at the least drag, soil at more, turf only to
//! a strong flood, and rock never. Water wears the less the more it already
//! carries, and none once it carries all it can: a stream loaded in a steep
//! cut runs on over the ground below without cutting it, and clean water
//! spilling over a dam bites hardest. What water carries is sand, which settles within a metre
//! or two where the flow slackens and builds fans, and fines from soil and
//! turf, which cloud the water, travel on and settle in still water over
//! minutes. Both settle at their own speed in proportion to how much the
//! water carries, so a slowing stream drops sand first.
//!
//! Sediment moves with water from running water to running water and to and
//! from pools. Water leaving stored water any other way, into a lake, a
//! river or the sea, up into the air or down into the ground, leaves what
//! it carries behind, and stored water that dries up or joins a lake lays it
//! on its bed: a stream into a lake builds a delta at its mouth.
//!
//! The ground changes only on the main thread. Each column of ground keeps
//! a bed account: sediment settled but not yet laid, and erosion asked for
//! but not yet taken. Now and then the worker asks for what is worth a
//! visible change, and the ground's answer, what it actually gave up and
//! took back, reaches the water the next time it runs. Sediment enters the
//! water only once the ground has given it up, and settled sediment the
//! ground has no room for waits in its bed, so no material is made or lost:
//! what the ground lost, less what it got back, is what the water carries
//! and what waits to be laid.

use bevy_math::DVec3;
use mechanic_core::WATER_DENSITY_KG_M3;
use serde::{Deserialize, Serialize};

use super::cells::CellMap;
use super::{
    CLING_METRES, GRAVITY, WATER_CELL_EDGE_CELLS, WATER_CELL_METRES, WaterCell, WaterGround,
    WaterWorld,
};
use crate::{
    BreakageResponse, CELL_QUANTA, MATERIAL_QUANTUM_M3, SedimentApplied, SedimentChange,
    TerrainMaterial, WorldCell,
};

/// Horizontal area of one water cell, in square metres.
const CELL_AREA_M2: f64 = WATER_CELL_METRES * WATER_CELL_METRES;

/// How fast sand settles through water, in metres per second.
const SAND_SETTLING_M_S: f64 = 0.02;

/// How fast fines settle through water, in metres per second: still water a
/// metre deep clears over half an hour.
const FINES_SETTLING_M_S: f64 = 5.0e-4;

/// Fastest running water counts as running, against its own wave speed
/// (a Froude number): a film's current, from what crosses its faces, can run
/// far faster than water a few millimetres deep ever flows over the ground.
const FASTEST_FROUDE: f64 = 1.5;

/// Drag above which running water keeps sand stirred up, in Pa: sand drops
/// only where the flow slackens.
const SAND_STAYS_PA: f64 = 1.0;

/// Drag above which running water keeps fines stirred up, in Pa.
const FINES_STAY_PA: f64 = 0.2;

/// Most sediment running water carries, as a share of its volume: water
/// wears its bed the less the more it already carries, and none at all as
/// thick as this, a flood's muddy water.
const CAPACITY: f64 = 0.01;

/// The least drag any ground holds against, in Pa: sand turned to mud.
const LEAST_HOLD: f64 = 0.15;

/// Ground more than this full of water is mud, which gives at half the drag.
const MUD_FILL: f64 = 0.9;

/// Least erosion, in quanta, worth a change to the ground: half a centimetre
/// over a column.
const WORTH_QUANTA: f64 = 64.0;

/// Least sediment, in quanta, worth laying: as little as begins a new cell
/// of ground, which settled sediment mostly needs.
const LAY_QUANTA: f64 = (CELL_QUANTA - u8::MAX as u32) as f64;

/// Seconds between the worker's asks of the ground: every change remeshes
/// the ground there and its water finds its way again.
const ASK_SECONDS: f64 = 5.0;

/// Columns wear and settle in turn, one of this many sets each step.
const WEAR_SETS: u32 = 4;

/// Most columns' changes asked for at once.
const MOST_CHANGES: usize = 512;

/// How a material gives way to running water: the bed shear stress it holds
/// against, in Pa, and how fast it wears beyond that, in metres of bed per
/// second for each Pa. Soil under a stream 5 cm deep running at 0.5 m/s
/// (about 6 Pa) wears 20 cm an hour.
const fn wears(material: TerrainMaterial) -> Option<(f64, f64)> {
    match material {
        TerrainMaterial::Soil => Some((1.0, 1.1e-5)),
        TerrainMaterial::Sand => Some((0.3, 3.3e-5)),
        // Roots hold turf against all but a violent flood: grass-lined
        // channels stand some 80 Pa.
        TerrainMaterial::SurfaceCover => Some((80.0, 1.1e-5)),
        TerrainMaterial::Rock | TerrainMaterial::Iron | TerrainMaterial::Graphite => None,
    }
}

/// How fast erosion runs, for tests and benchmarks that need an hour of it
/// in a minute.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErosionConfig {
    /// Erosion's rate over the game's own, 1 in play.
    pub speed: f64,
}

impl Default for ErosionConfig {
    fn default() -> Self {
        Self { speed: 1.0 }
    }
}

/// Sediment water carries, in quanta of material.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SedimentLoad {
    /// Sand, which settles fast.
    pub sand: f64,
    /// Fines from soil and turf, which settle slowly and cloud the water.
    pub fines: f64,
}

impl SedimentLoad {
    /// Sediment of both kinds.
    pub fn total(self) -> f64 {
        self.sand + self.fines
    }

    pub(super) fn scaled(self, by: f64) -> Self {
        Self {
            sand: self.sand * by,
            fines: self.fines * by,
        }
    }

    pub(super) fn add(&mut self, other: Self) {
        self.sand += other.sand;
        self.fines += other.fines;
    }

    pub(super) fn sub(&mut self, other: Self) {
        self.sand -= other.sand;
        self.fines -= other.fines;
    }

    /// The share of this load that leaves with `volume` of `held` m³ of
    /// water, taken out of it.
    pub(super) fn part(&mut self, volume: f64, held: f64) -> Self {
        if held <= 0.0 || volume <= 0.0 {
            return Self::default();
        }
        let part = self.scaled((volume / held).min(1.0));
        self.sub(part);
        part
    }
}

/// Sediment water holding `volume` m³ carries, in kg per m³ of water:
/// fines weigh as the soil they came from.
pub(super) fn silt(load: SedimentLoad, volume: f64) -> f64 {
    if volume <= 0.0 {
        return 0.0;
    }
    let weight = |material| BreakageResponse::for_material(material).density_kg_m3;
    MATERIAL_QUANTUM_M3
        * (load.sand * weight(TerrainMaterial::Sand) + load.fines * weight(TerrainMaterial::Soil))
        / volume
}

/// One column's sediment waiting on the ground.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Bed {
    /// Height of the ground's top it was last added at, in metres.
    height: f64,
    /// Sediment settled and not yet laid.
    settled: SedimentLoad,
    /// Settled sediment asked of the ground and not yet answered.
    laying: SedimentLoad,
    /// Erosion owed, in quanta: wear not yet asked of the ground. It is no
    /// material, only a debt.
    owed: f64,
}

/// One column's sediment waiting on the ground, in a saved world.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BedDoc {
    /// The column, in water cells along x and z.
    pub column: (i32, i32),
    /// Height of the ground's top, in metres.
    pub height: f64,
    /// Sediment settled and not yet laid.
    pub settled: SedimentLoad,
}

/// What became of one column's ask of the ground.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ask {
    Take,
    Lay { sand: bool },
}

/// Where water put somewhere ended up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Placed {
    Pool(u32),
    /// Running water in this column.
    Sheet(i32, i32),
    /// Seed-derived water, or the air.
    Gone,
}

/// Sediment in the world's books, in quanta.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SedimentLedger {
    /// Carried in running water and pools.
    pub suspended: f64,
    /// Settled and not yet laid on the ground.
    pub settled: f64,
    /// Taken from the ground, all told.
    pub eroded: f64,
    /// Laid back on the ground, all told.
    pub laid: f64,
}

impl SedimentLedger {
    /// What the books lack, in quanta: what the ground lost less what it got
    /// back, less what the water carries and what waits to be laid. Zero
    /// but for rounding.
    pub fn unaccounted(&self) -> f64 {
        self.eroded - self.laid - self.suspended - self.settled
    }
}

/// Sediment state of a world beyond what its water carries, to save.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SedimentDoc {
    /// Columns with sediment waiting to be laid.
    pub beds: Vec<BedDoc>,
    /// Quanta taken from the ground, all told.
    pub eroded: f64,
    /// Quanta laid back on the ground, all told.
    pub laid: f64,
}

/// Sediment kept beside the water.
#[derive(Clone, Debug, Default)]
pub(super) struct Sediment {
    beds: CellMap<(i32, i32), Bed>,
    config: ErosionConfig,
    /// Seconds since the ground was last asked.
    since: f64,
    /// The set of columns that wears next.
    turn: u32,
    /// The columns of the changes last asked, in order, while unanswered.
    asked: Vec<((i32, i32), Ask)>,
    eroded: f64,
    laid: f64,
}

/// How fast running water `depth` metres deep running at `speed` m/s wears
/// ground of a material away, in metres of bed per second; mud gives at
/// half the drag, but soaked turf is held by its roots all the same.
fn wear_rate(material: TerrainMaterial, depth: f64, speed: f64, mud: bool) -> f64 {
    let Some((holds, rate)) = wears(material) else {
        return 0.0;
    };
    let holds = if mud && material != TerrainMaterial::SurfaceCover {
        0.5 * holds
    } else {
        holds
    };
    rate * (shear(depth, speed) - holds).max(0.0)
}

/// Bed shear stress under running water `depth` metres deep running at
/// `speed` m/s, by Manning's law, in Pa.
fn shear(depth: f64, speed: f64) -> f64 {
    let roughness = super::sheet::ROUGHNESS;
    WATER_DENSITY_KG_M3 * GRAVITY * roughness * roughness * speed * speed / depth.cbrt()
}

impl WaterWorld {
    /// Sets how fast erosion runs.
    pub fn set_erosion(&mut self, config: ErosionConfig) {
        self.sediment.config = config;
    }

    /// Sediment in the books.
    pub fn sediment_ledger(&self) -> SedimentLedger {
        let suspended = self
            .sheets
            .iter()
            .map(|(_, sheet)| sheet.load.total())
            .sum::<f64>()
            + self
                .pools
                .values()
                .map(|pool| pool.load.total())
                .sum::<f64>();
        let settled = self
            .sediment
            .beds
            .values()
            .map(|bed| bed.settled.total() + bed.laying.total())
            .sum();
        SedimentLedger {
            suspended,
            settled,
            eroded: self.sediment.eroded,
            laid: self.sediment.laid,
        }
    }

    /// Lays sediment on the bed of a column whose ground's top is near
    /// `height`.
    pub(super) fn settle_load(&mut self, column: (i32, i32), height: f64, load: SedimentLoad) {
        if load.total() <= 0.0 {
            return;
        }
        let bed = self.sediment.beds.entry(column).or_default();
        bed.height = height;
        bed.settled.add(load);
    }

    /// Sediment in water that went somewhere: into a pool or running water
    /// it is carried on, anywhere else it settles where it entered, on the
    /// ground near `near`.
    pub(super) fn give_load(&mut self, placed: Placed, load: SedimentLoad, near: (i32, i32, f64)) {
        if load.total() <= 0.0 {
            return;
        }
        match placed {
            Placed::Pool(id) => {
                if let Some(pool) = self.pools.get_mut(&id) {
                    pool.load.add(load);
                    return;
                }
            }
            Placed::Sheet(x, z) => {
                if let Some(slot) = self.sheets.slot(x, z)
                    && self.sheets.at(slot).present
                {
                    self.sheets.at_mut(slot).load.add(load);
                    return;
                }
            }
            Placed::Gone => {}
        }
        self.settle_load((near.0, near.1), near.2, load);
    }

    /// A pool's load laid evenly over its bed.
    pub(super) fn settle_pool_load(&mut self, pool: &super::Pool, load: SedimentLoad) {
        let beds = pool.beds().collect::<Vec<_>>();
        self.settle_over(&beds, pool.seed, load);
    }

    /// Sediment laid evenly over the bed columns of still water begun at
    /// `seed`.
    fn settle_over(&mut self, beds: &[(WaterCell, f64)], seed: WaterCell, load: SedimentLoad) {
        if beds.is_empty() {
            self.settle_load((seed.x, seed.z), seed.bottom(), load);
            return;
        }
        #[expect(clippy::cast_precision_loss, reason = "a count of columns")]
        let each = load.scaled(1.0 / beds.len() as f64);
        for &(cell, floor) in beds {
            self.settle_load((cell.x, cell.z), floor, each);
        }
    }

    /// Running water wearing its bed and every water dropping what it
    /// carries, over `dt`.
    pub(super) fn wear_and_settle(&mut self, ground: &impl WaterGround, dt: f64) {
        self.sediment.since += dt;
        let set = self.sediment.turn;
        self.sediment.turn = (set + 1) % WEAR_SETS;
        let speed_up = self.sediment.config.speed;
        // Each column's turn covers the steps since its last.
        let turn = dt * f64::from(WEAR_SETS);
        let ours = |x: i32, z: i32| {
            (i64::from(x) + 2 * i64::from(z)).rem_euclid(i64::from(WEAR_SETS)) == i64::from(set)
        };
        for (slot, cell) in self.sheets.wet_where(ours) {
            let dt = turn;
            let sheet = *self.sheets.at(slot);
            let depth = sheet.volume.max(0.0) / CELL_AREA_M2;
            let speed = if depth < CLING_METRES {
                0.0
            } else {
                super::sheet::current(&sheet, depth)
                    .length()
                    .min(FASTEST_FROUDE * (GRAVITY * depth).sqrt())
            };
            // Clean water dragging less than the softest mud holds against
            // neither wears nor drops anything.
            if sheet.load.total() <= 0.0 && shear(depth.max(CLING_METRES), speed) <= LEAST_HOLD {
                continue;
            }
            let column = (cell.x, cell.z);
            // What settles: each kind in proportion to how much the water
            // carries and how fast it falls through it, and only where the
            // flow is too slack to keep it stirred up (Krone).
            let mut load = sheet.load;
            let drag = shear(depth.max(CLING_METRES), speed);
            let settled = if depth < CLING_METRES {
                load
            } else {
                let falls = |settling: f64, stays: f64| {
                    (settling * dt / depth).min(1.0) * (1.0 - drag / stays).max(0.0)
                };
                SedimentLoad {
                    sand: load.sand * falls(SAND_SETTLING_M_S, SAND_STAYS_PA),
                    fines: load.fines * falls(FINES_SETTLING_M_S, FINES_STAY_PA),
                }
            };
            load.sub(settled);
            if depth >= CLING_METRES {
                let soil = self.soil_at(ground, cell, sheet.floor());
                let material = *soil.material.get_or_insert_with(|| {
                    let centre = cell.centre();
                    ground
                        .material(DVec3::new(centre.x, sheet.floor() - 0.02, centre.z))
                        .unwrap_or(TerrainMaterial::Rock)
                });
                let mud = soil.fill() > MUD_FILL;
                let carried = load.total() * MATERIAL_QUANTUM_M3 / sheet.volume;
                let rate = wear_rate(material, depth, speed, mud) * (1.0 - carried / CAPACITY);
                if rate > 0.0 {
                    let worn = speed_up * rate * CELL_AREA_M2 * dt / MATERIAL_QUANTUM_M3;
                    let bed = self.sediment.beds.entry(column).or_default();
                    bed.height = sheet.floor();
                    bed.owed += worn;
                    // Sediment waiting on this bed is taken up first, before
                    // the ground itself.
                    let ready = bed.settled.total();
                    if ready > 0.0 && bed.owed > 0.0 {
                        let up = bed.settled.part(bed.owed.min(ready), ready);
                        bed.owed -= up.total();
                        load.add(up);
                    }
                }
            }
            self.sheets.at_mut(slot).load = load;
            self.settle_load(column, sheet.floor(), settled);
        }
        // Still water drops what it carries onto its bed.
        let ids = self.pools.keys().copied().collect::<Vec<_>>();
        for id in ids {
            let Some(pool) = self.pools.get(&id) else {
                continue;
            };
            if pool.load.total() <= 0.0 {
                continue;
            }
            let depth = (pool.volume / pool.surface_area().max(CELL_AREA_M2)).max(CLING_METRES);
            let settled = SedimentLoad {
                sand: pool.load.sand * (SAND_SETTLING_M_S * dt / depth).min(1.0),
                fines: pool.load.fines * (FINES_SETTLING_M_S * dt / depth).min(1.0),
            };
            let pool = self.pools.get_mut(&id).expect("looked up above");
            pool.load.sub(settled);
            let (beds, seed) = (pool.beds().collect::<Vec<_>>(), pool.seed);
            self.settle_over(&beds, seed, settled);
        }
    }

    /// The changes to the ground the water asks for now: every column's
    /// sediment worth laying and erosion worth taking, at most every five
    /// seconds and only once the last ask was answered. Hand the ground's
    /// answer back to [`Self::sediment_applied`].
    pub fn sediment_requests(&mut self) -> Vec<SedimentChange> {
        if !self.sediment.asked.is_empty() || self.sediment.since < ASK_SECONDS {
            return Vec::new();
        }
        self.sediment.since = 0.0;
        let mut changes = Vec::new();
        let mut columns = self.sediment.beds.keys().copied().collect::<Vec<_>>();
        columns.sort_unstable();
        let edge = WATER_CELL_EDGE_CELLS;
        for column in columns {
            if changes.len() >= MOST_CHANGES {
                break;
            }
            let Some(bed) = self.sediment.beds.get_mut(&column) else {
                continue;
            };
            let change = |quanta: f64, material| SedimentChange {
                x: column.0 * edge,
                z: column.1 * edge,
                edge,
                height: bed.height,
                #[expect(clippy::cast_possible_truncation, reason = "a few thousand quanta")]
                quanta: quanta.trunc() as i64,
                material,
            };
            if bed.owed >= WORTH_QUANTA {
                changes.push(change(-bed.owed, TerrainMaterial::Soil));
                self.sediment.asked.push((column, Ask::Take));
                bed.owed = 0.0;
            }
            for sand in [true, false] {
                let ready = if sand {
                    bed.settled.sand
                } else {
                    bed.settled.fines
                };
                if ready < LAY_QUANTA {
                    continue;
                }
                let laying = ready.trunc();
                let material = if sand {
                    TerrainMaterial::Sand
                } else {
                    TerrainMaterial::Soil
                };
                changes.push(change(laying, material));
                self.sediment.asked.push((column, Ask::Lay { sand }));
                let part = if sand {
                    SedimentLoad {
                        sand: laying,
                        fines: 0.0,
                    }
                } else {
                    SedimentLoad {
                        sand: 0.0,
                        fines: laying,
                    }
                };
                bed.settled.sub(part);
                bed.laying.add(part);
            }
        }
        changes
    }

    /// What the ground did with the changes [`Self::sediment_requests`] last
    /// asked for, in the same order: material it gave up enters the water
    /// over its column, or waits on its bed where no water runs; sediment it
    /// had no room for waits to be laid again.
    pub fn sediment_applied(&mut self, applied: &[SedimentApplied]) {
        let asked = std::mem::take(&mut self.sediment.asked);
        for (index, (column, ask)) in asked.into_iter().enumerate() {
            let done = applied.get(index).copied().unwrap_or_default();
            match ask {
                Ask::Take => {
                    let sand = done.taken[TerrainMaterial::Sand.code() as usize];
                    #[expect(clippy::cast_precision_loss, reason = "a few thousand quanta")]
                    let taken = SedimentLoad {
                        sand: sand as f64,
                        fines: (done.total_taken() - sand) as f64,
                    };
                    #[expect(clippy::cast_precision_loss, reason = "a few thousand quanta")]
                    {
                        self.sediment.eroded += done.total_taken() as f64;
                    }
                    if done.total_taken() == 0 {
                        // Nothing soft left: the ground is measured again.
                        if let Some(soil) = self.soil.get_mut(&column) {
                            soil.material = None;
                        }
                    }
                    let height = self
                        .sediment
                        .beds
                        .get(&column)
                        .map_or(0.0, |bed| bed.height);
                    self.give_load(
                        Placed::Sheet(column.0, column.1),
                        taken,
                        (column.0, column.1, height),
                    );
                }
                Ask::Lay { sand } => {
                    #[expect(clippy::cast_precision_loss, reason = "a few thousand quanta")]
                    let laid = done.laid as f64;
                    self.sediment.laid += laid;
                    if let Some(bed) = self.sediment.beds.get_mut(&column) {
                        let (laying, settled) = if sand {
                            (&mut bed.laying.sand, &mut bed.settled.sand)
                        } else {
                            (&mut bed.laying.fines, &mut bed.settled.fines)
                        };
                        // What found no room waits to be laid again.
                        *settled += (*laying - laid).max(0.0);
                        *laying = 0.0;
                    }
                }
            }
        }
        self.sediment.beds.retain(|_, bed| {
            bed.settled.total() > 0.0 || bed.laying.total() > 0.0 || bed.owed > 0.0
        });
    }

    /// The ground could not be changed: what was asked waits to be asked
    /// again.
    pub fn sediment_refused(&mut self) {
        for (column, _) in std::mem::take(&mut self.sediment.asked) {
            if let Some(bed) = self.sediment.beds.get_mut(&column) {
                let laying = std::mem::take(&mut bed.laying);
                bed.settled.add(laying);
            }
        }
    }

    /// Sediment waiting beside the water, to save. Sediment being laid when
    /// the world is saved is saved as settled.
    pub(super) fn sediment_doc(&self) -> SedimentDoc {
        let mut beds = self
            .sediment
            .beds
            .iter()
            .filter(|(_, bed)| bed.settled.total() + bed.laying.total() > 0.0)
            .map(|(&column, bed)| {
                let mut settled = bed.settled;
                settled.add(bed.laying);
                BedDoc {
                    column,
                    height: bed.height,
                    settled,
                }
            })
            .collect::<Vec<_>>();
        beds.sort_by_key(|bed| bed.column);
        SedimentDoc {
            beds,
            eroded: self.sediment.eroded,
            laid: self.sediment.laid,
        }
    }

    pub(super) fn load_sediment(&mut self, doc: &SedimentDoc) {
        for bed in &doc.beds {
            self.settle_load(bed.column, bed.height, bed.settled);
        }
        self.sediment.eroded = doc.eroded;
        self.sediment.laid = doc.laid;
    }

    /// The ground changed in these terrain cells, as sediment came and went:
    /// the water cells holding them and beside them are measured again, and
    /// the water over them finds its routes and floors again. Cheaper than
    /// [`Self::terrain_changed`], which measures whole bricks again.
    pub fn ground_cells_changed(&mut self, ground: &impl WaterGround, cells: &[WorldCell]) {
        let touched = cells
            .iter()
            .map(|cell| WaterCell::containing(cell.centre().0))
            .collect::<std::collections::BTreeSet<_>>();
        for &cell in &touched {
            self.ground.forget_cell(cell);
            // The drawn ground at the column's corners and centre, as sought
            // from the heights around it.
            for y in cell.y - 2..=cell.y + 2 {
                for (dx, dz) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    self.tops.remove(&(cell.x + dx, y, cell.z + dz));
                }
                self.floors.remove(&(cell.x, y, cell.z));
                self.wick_tops.remove(&(cell.x, y, cell.z));
            }
        }
        // Running water over the change rests on the ground's new top and
        // finds its routes again; still water there is measured again whole.
        // Dry ground is measured afresh when water reaches it.
        let mut measured = std::collections::BTreeSet::new();
        for &cell in &touched {
            if let Some(slot) = self.sheets.slot(cell.x, cell.z)
                && self.sheets.at(slot).present
            {
                let at = WaterCell::new(cell.x, self.sheets.at(slot).y, cell.z);
                if measured.insert(at) {
                    self.remeasure_sheet(ground, at);
                }
                if let Some(slot) = self.sheets.slot(cell.x, cell.z)
                    && self.sheets.at(slot).present
                {
                    let floor = self.sheets.at(slot).floor();
                    if let Some(soil) = self.soil.get_mut(&(cell.x, cell.z)) {
                        soil.top = floor;
                        soil.material = None;
                    }
                }
            }
            for cell in [cell, cell.up()] {
                if (self.owner.contains_key(&cell) || self.joined.contains_key(&cell))
                    && measured.insert(cell)
                {
                    self.remeasure(ground, cell);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SedimentLoad, shear, wear_rate};
    use crate::TerrainMaterial;

    #[test]
    fn a_stream_drags_on_its_bed_harder_the_faster_it_runs() {
        // 5 cm deep at half a metre a second: about 6 Pa.
        let stream = shear(0.05, 0.5);
        assert!((5.5..6.5).contains(&stream), "{stream:.2} Pa");
        assert!(shear(0.05, 1.0) > 3.9 * stream);
        // Deeper water at the same speed drags a little less.
        assert!(shear(0.5, 0.5) < stream);
    }

    #[test]
    fn a_stream_cuts_soil_a_fifth_of_a_metre_an_hour_and_sand_faster() {
        let hour = |material, depth, speed, mud| 3_600.0 * wear_rate(material, depth, speed, mud);
        let soil = hour(TerrainMaterial::Soil, 0.05, 0.5, false);
        assert!(
            (0.15..0.25).contains(&soil),
            "soil wears {soil:.3} m an hour"
        );
        assert!(hour(TerrainMaterial::Sand, 0.05, 0.5, false) > 3.0 * soil);
        assert!(hour(TerrainMaterial::Soil, 0.05, 0.5, true) > soil);
        // A trickle wears nothing.
        assert!(hour(TerrainMaterial::Soil, 0.01, 0.05, false) <= 0.0);
    }

    #[test]
    fn no_ground_holds_against_less_drag_than_the_least() {
        for material in [
            TerrainMaterial::Soil,
            TerrainMaterial::Sand,
            TerrainMaterial::SurfaceCover,
        ] {
            let holds = super::wears(material).expect("soft ground").0;
            assert!(0.5 * holds >= super::LEAST_HOLD);
        }
    }

    #[test]
    fn turf_holds_against_a_stream_until_a_flood_strips_it_and_rock_never_wears() {
        let rate = |material, depth, speed| wear_rate(material, depth, speed, false);
        assert!(rate(TerrainMaterial::SurfaceCover, 0.05, 0.5) <= 0.0);
        // A breach's flood, half a metre deep at 3 m/s, strips it.
        assert!(rate(TerrainMaterial::SurfaceCover, 0.5, 3.0) > 0.0);
        assert!(rate(TerrainMaterial::SurfaceCover, 0.1, 1.5) <= 0.0);
        for material in [
            TerrainMaterial::Rock,
            TerrainMaterial::Iron,
            TerrainMaterial::Graphite,
        ] {
            assert!(rate(material, 0.5, 5.0) <= 0.0);
        }
    }

    #[test]
    fn water_leaving_takes_its_share_of_what_it_carries() {
        let mut load = SedimentLoad {
            sand: 40.0,
            fines: 60.0,
        };
        let part = load.part(0.25, 1.0);
        assert!((part.total() - 25.0).abs() < 1e-12);
        assert!((load.total() - 75.0).abs() < 1e-12);
        assert!((load.part(2.0, 1.0).total() - 75.0).abs() < 1e-12);
        assert!(load.total().abs() < 1e-12);
    }
}
