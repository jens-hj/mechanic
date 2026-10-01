//! Water in the ground: running water and pools soak into soil, sand and
//! ground cover, which hold it in their pores and give it up slowly, deep
//! down to the sea and up into the air. Rock takes none.
//!
//! Each column of ground water has reached keeps how much it holds, how
//! much it can hold and how fast it takes water in. Infiltration slows as
//! the ground fills (Green and Ampt, simplified): dry ground drinks a film
//! quickly, saturated ground takes nothing, and water then runs on or
//! stands on it as mud.
//!
//! Wet ground wicks water sideways into drier ground beside it, as
//! capillarity does, but only while it is wetter by more than a margin: the
//! ground beside running water and pools darkens into a damp fringe a few
//! columns wide that fades out into dry ground.

use serde::{Deserialize, Serialize};

use super::{WATER_CELL_METRES, WaterCell, WaterGround, WaterWorld};
use crate::TerrainMaterial;

/// Horizontal area of one water cell, in square metres.
const CELL_AREA_M2: f64 = WATER_CELL_METRES * WATER_CELL_METRES;

/// Depth of ground that holds water, in metres.
const SOIL_METRES: f64 = 0.5;

/// Fraction of soil water that drains deep towards the sea each second:
/// ground water moves over hours.
const DRAIN_PER_SECOND: f64 = 1.0 / 21_600.0;

/// Water soil gives up to the air, in metres per second: a fifth of what
/// open water does.
const SOIL_EVAPORATION_M_S: f64 = 0.001 / 3_600.0;

/// Least water a column keeps; less is forgotten into the air, in m³.
const DRY_SOIL_M3: f64 = 1.0e-7;

/// Fill beyond which ground is mud.
const MUD_FILL: f64 = 0.9;

/// How fast ground wicks water sideways into drier ground beside it, in
/// metres per second for each unit of fill it is wetter by beyond
/// [`WICK_GRADIENT`]. Sped up for play as infiltration is: ground beside
/// running water darkens within a minute and the fringe widens over minutes.
const WICK_M_S: f64 = 1.0e-3;

/// How much fuller ground must be than the ground beside it to wick water
/// into it: wet ground pulls water only so far, so the damp fringe beside
/// water fades out over a few columns rather than spreading on.
const WICK_GRADIENT: f64 = 0.25;

/// Most the ground's top may rise or fall from one column to the next for
/// water to wick across, in metres: it climbs no bank and falls into no cave.
const WICK_STEP_METRES: f64 = 0.25;

/// Columns wick in turn, one of this many sets each step.
const WICK_SETS: u32 = 64;

/// How a material holds water: its pores, as a fraction of its volume, and
/// how fast dry ground takes water in, in metres per second. Rates are sped
/// up for play: a film soaks into dry soil in seconds.
const fn holds(material: TerrainMaterial) -> (f64, f64) {
    match material {
        TerrainMaterial::SurfaceCover => (0.4, 6.0e-5),
        TerrainMaterial::Soil => (0.35, 5.0e-5),
        TerrainMaterial::Sand => (0.4, 2.0e-4),
        TerrainMaterial::Rock | TerrainMaterial::Iron | TerrainMaterial::Graphite => (0.0, 0.0),
    }
}

/// Water held in the ground under one column.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Soil {
    /// Water held, in m³.
    moisture: f64,
    /// Most it can hold, in m³.
    capacity: f64,
    /// How fast it takes water in when dry, in metres per second.
    rate: f64,
    /// Height of the ground's top, in metres.
    top: f64,
    /// The water that has stood on it since it last wicked, if any: its top
    /// is then wet through, whatever it holds below.
    covered: Option<Cover>,
}

/// Water standing on a column of ground, which feeds what it wicks away.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cover {
    /// Running water in the column.
    Sheet,
    /// A pool, by its id.
    Pool(u32),
}

/// Where water wicking out of a column comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    /// Water standing on it.
    Cover(Cover),
    /// Its own ground.
    Ground,
}

/// Water in the ground under one column, in a saved world.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoilDoc {
    /// The column, in water cells along x and z.
    pub column: (i32, i32),
    /// Height of the ground's top, in metres.
    pub top: f64,
    /// Water held, in m³.
    pub moisture_m3: f64,
    /// Most it can hold, in m³.
    pub capacity_m3: f64,
    /// How fast it takes water in when dry, in metres per second.
    pub rate_m_s: f64,
}

/// How wet the ground is under one column, to draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WetGround {
    /// The column, in water cells along x and z.
    pub column: (i32, i32),
    /// Height of the ground's top, in metres.
    pub top: f64,
    /// How full its pores are, from 0 to 1.
    pub fill: f64,
    /// Water it holds, and running water standing on it, as a depth over
    /// the column, in metres: a film too thin to see adds next to nothing.
    pub soaked: f64,
}

impl Soil {
    /// How full its pores are: ground without pores never fills.
    fn fill(&self) -> f64 {
        if self.capacity > 0.0 {
            self.moisture / self.capacity
        } else {
            0.0
        }
    }

    /// How wet it is for wicking: ground under water is wet through.
    fn wetness(&self) -> f64 {
        if self.covered.is_some() {
            1.0
        } else {
            self.fill()
        }
    }

    /// Water it takes in over `dt` seconds from water standing on it, at
    /// most `offered`.
    fn take(&mut self, offered: f64, dt: f64) -> f64 {
        let room = (self.capacity - self.moisture).max(0.0);
        if room <= 0.0 {
            return 0.0;
        }
        // Dry ground drinks at three times its rate, wet ground at its
        // rate until it is full.
        let rate = self.rate * 2.0f64.mul_add((1.0 - self.fill()).max(0.0), 1.0);
        let taken = offered.min(rate * CELL_AREA_M2 * dt).min(room).max(0.0);
        self.moisture += taken;
        taken
    }
}

impl WaterWorld {
    /// Water held in the ground, in m³.
    pub fn soil_m3(&self) -> f64 {
        self.soil.values().map(|soil| soil.moisture).sum()
    }

    /// How full the ground's pores are under a column, from 0 to 1: 0
    /// where no water has reached, above 0.9 mud.
    pub fn soil_fill(&self, x: i32, z: i32) -> f64 {
        self.soil.get(&(x, z)).map_or(0.0, Soil::fill)
    }

    /// Whether the ground under a column is mud.
    pub fn is_mud(&self, x: i32, z: i32) -> bool {
        self.soil_fill(x, z) > MUD_FILL
    }

    /// Every column of wet ground, to draw: ground water has soaked into,
    /// and ground running water stands on.
    ///
    /// Running water counts by its depth, and its column keeps the height
    /// its ground was measured at: the film at the front of running water
    /// comes and goes from step to step, and would otherwise flicker.
    pub fn wet_ground(&self) -> Vec<WetGround> {
        let mut wet = self
            .soil
            .iter()
            .filter(|(_, soil)| soil.moisture > 0.0)
            .map(|(&column, soil)| {
                (
                    column,
                    WetGround {
                        column,
                        top: soil.top,
                        fill: soil.fill(),
                        soaked: soil.moisture / CELL_AREA_M2,
                    },
                )
            })
            .collect::<super::cells::CellMap<_, _>>();
        for (cell, sheet) in self.sheets.iter() {
            let depth = sheet.volume.max(0.0) / CELL_AREA_M2;
            wet.entry((cell.x, cell.z))
                .and_modify(|wet| wet.soaked += depth)
                .or_insert(WetGround {
                    column: (cell.x, cell.z),
                    top: self
                        .soil
                        .get(&(cell.x, cell.z))
                        .map_or(sheet.floor(), |soil| soil.top),
                    fill: self.soil_fill(cell.x, cell.z),
                    soaked: depth,
                });
        }
        wet.into_values().collect()
    }

    pub(super) fn soil_docs(&self) -> Vec<SoilDoc> {
        self.soil
            .iter()
            .map(|(&column, soil)| SoilDoc {
                column,
                top: soil.top,
                moisture_m3: soil.moisture,
                capacity_m3: soil.capacity,
                rate_m_s: soil.rate,
            })
            .collect()
    }

    pub(super) fn load_soil(&mut self, docs: &[SoilDoc]) {
        for doc in docs {
            self.soil.insert(
                doc.column,
                Soil {
                    moisture: doc.moisture_m3,
                    capacity: doc.capacity_m3,
                    rate: doc.rate_m_s,
                    top: doc.top,
                    covered: None,
                },
            );
        }
    }

    /// The ground under a cell whose water rests on `floor`, measured the
    /// first time water reaches it.
    fn soil_at(&mut self, ground: &impl WaterGround, cell: WaterCell, floor: f64) -> &mut Soil {
        self.soil.entry((cell.x, cell.z)).or_insert_with(|| {
            let centre = cell.centre();
            let below = bevy_math::DVec3::new(centre.x, floor - 0.02, centre.z);
            let (pores, rate) = ground.material(below).map_or((0.0, 0.0), holds);
            Soil {
                moisture: 0.0,
                capacity: pores * SOIL_METRES * CELL_AREA_M2,
                rate,
                top: floor,
                covered: None,
            }
        })
    }

    /// How much the ground under a column water wicks into from ground
    /// whose top is at `top` can hold, and how wet it is, measured the first
    /// time: none where the drawn ground lies further than
    /// [`WICK_STEP_METRES`] above or below.
    fn soil_beside(
        &mut self,
        ground: &impl WaterGround,
        column: (i32, i32),
        top: f64,
    ) -> Option<(f64, f64)> {
        if let Some(soil) = self.soil.get(&column) {
            return ((soil.top - top).abs() <= WICK_STEP_METRES)
                .then(|| (soil.capacity, soil.wetness()));
        }
        let centre = WaterCell::new(column.0, 0, column.1).centre();
        // Ground a column cannot wick into is met again every turn it
        // wicks, so the drawn ground's height is remembered.
        #[expect(clippy::cast_possible_truncation, reason = "a cell height")]
        let from = (column.0, (top / WATER_CELL_METRES).floor() as i32, column.1);
        let beside = (*self.wick_tops.entry(from).or_insert_with(|| {
            ground.ground_top(
                centre.x,
                centre.z,
                top + WICK_STEP_METRES,
                2.0 * WICK_STEP_METRES,
            )
        }))?;
        let below = bevy_math::DVec3::new(centre.x, beside - 0.02, centre.z);
        let (pores, rate) = ground.material(below).map_or((0.0, 0.0), holds);
        let capacity = pores * SOIL_METRES * CELL_AREA_M2;
        self.soil.insert(
            column,
            Soil {
                moisture: 0.0,
                capacity,
                rate,
                top: beside,
                covered: None,
            },
        );
        Some((capacity, 0.0))
    }

    /// Water wicking sideways from wet ground into drier ground beside it,
    /// over `dt`. One set of columns wicks each step, over that set's turn.
    ///
    /// Ground under water is wet through, and the water on it feeds what it
    /// wicks away: drawn from its own pores instead, a column a film once
    /// crossed would empty into the dry ground beside it in one turn and
    /// take the water back the next.
    fn wick(&mut self, ground: &impl WaterGround, dt: f64) {
        let set = self.wick_set;
        self.wick_set = (set + 1) % WICK_SETS;
        let dt = dt * f64::from(WICK_SETS);
        let turn = |(x, z): (i32, i32)| {
            (i64::from(x) + 3 * i64::from(z)).rem_euclid(i64::from(WICK_SETS)) == i64::from(set)
        };
        let mut turns = Vec::new();
        for (&column, soil) in &mut self.soil {
            if turn(column) {
                turns.push((
                    column,
                    soil.covered.take(),
                    soil.fill(),
                    soil.top,
                    soil.moisture,
                ));
            }
        }
        let mut senders = Vec::new();
        for (column, cover, fill, top, moisture) in turns {
            let standing = cover.map_or(0.0, |cover| self.standing(column, cover));
            if let Some(cover) = cover.filter(|_| standing > 0.0) {
                senders.push((column, Source::Cover(cover), 1.0, top, standing));
            } else if fill > WICK_GRADIENT && moisture > 0.0 {
                senders.push((column, Source::Ground, fill, top, moisture));
            }
        }
        let mut flows = Vec::new();
        for (column, source, wetness, top, budget) in senders {
            let first = flows.len();
            let mut given = 0.0;
            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let beside = (column.0 + dx, column.1 + dz);
                let Some((capacity, beside_wetness)) = self.soil_beside(ground, beside, top) else {
                    continue;
                };
                if capacity <= 0.0 {
                    continue;
                }
                let pull = wetness - beside_wetness - WICK_GRADIENT;
                if pull > 0.0 {
                    let flow = WICK_M_S * CELL_AREA_M2 * dt * pull;
                    flows.push((column, source, beside, flow));
                    given += flow;
                }
            }
            // A column gives no more than it has.
            if given > budget {
                for flow in &mut flows[first..] {
                    flow.3 *= budget / given;
                }
            }
        }
        for (from, source, to, flow) in flows {
            let room = self
                .soil
                .get(&to)
                .map_or(0.0, |soil| (soil.capacity - soil.moisture).max(0.0));
            let held = match source {
                Source::Cover(cover) => self.standing(from, cover),
                Source::Ground => self.soil.get(&from).map_or(0.0, |soil| soil.moisture),
            };
            let moved = flow.min(room).min(held);
            if moved <= 0.0 {
                continue;
            }
            match source {
                Source::Cover(Cover::Sheet) => {
                    if let Some(slot) = self.sheets.slot(from.0, from.1) {
                        self.sheets.at_mut(slot).volume -= moved;
                    }
                }
                Source::Cover(Cover::Pool(id)) => {
                    if let Some(pool) = self.pools.get_mut(&id) {
                        pool.volume -= moved;
                    }
                }
                Source::Ground => {
                    if let Some(soil) = self.soil.get_mut(&from) {
                        soil.moisture -= moved;
                    }
                }
            }
            if let Some(soil) = self.soil.get_mut(&to) {
                soil.moisture += moved;
            }
        }
    }

    /// Water standing on a column that may wick into the ground beside it,
    /// in m³.
    fn standing(&self, column: (i32, i32), cover: Cover) -> f64 {
        match cover {
            Cover::Sheet => self
                .sheets
                .slot(column.0, column.1)
                .map(|slot| self.sheets.at(slot))
                .filter(|sheet| sheet.present)
                .map_or(0.0, |sheet| sheet.volume.max(0.0)),
            Cover::Pool(id) => self.pools.get(&id).map_or(0.0, |pool| pool.volume.max(0.0)),
        }
    }

    /// Water soaking from running water and pools into the ground under
    /// them, and out of the ground deep down and into the air, over `dt`.
    pub(super) fn soak(&mut self, ground: &impl WaterGround, dt: f64) {
        for (slot, cell) in self.sheets.wet() {
            let sheet = *self.sheets.at(slot);
            let soil = self.soil_at(ground, cell, sheet.floor());
            soil.covered = Some(Cover::Sheet);
            let taken = soil.take(sheet.volume, dt);
            self.sheets.at_mut(slot).volume -= taken;
        }
        let beds = self
            .pools
            .iter()
            .flat_map(|(&id, pool)| pool.beds().map(move |(cell, floor)| (id, cell, floor)))
            .collect::<Vec<_>>();
        for (id, cell, floor) in beds {
            let offered = self.pools.get(&id).map_or(0.0, |pool| pool.volume);
            let soil = self.soil_at(ground, cell, floor);
            soil.covered = Some(Cover::Pool(id));
            let taken = soil.take(offered, dt);
            if let Some(pool) = self.pools.get_mut(&id) {
                pool.volume -= taken;
            }
        }
        self.wick(ground, dt);
        let (mut drained, mut risen) = (0.0, 0.0);
        self.soil.retain(|_, soil| {
            let down = soil.moisture * DRAIN_PER_SECOND * dt;
            let up = (SOIL_EVAPORATION_M_S * CELL_AREA_M2 * dt).min(soil.moisture - down);
            soil.moisture -= down + up;
            drained += down;
            risen += up;
            // Ground without pores is remembered, so it is not measured
            // again every step water runs over it.
            if soil.capacity > 0.0 && soil.moisture < DRY_SOIL_M3 {
                risen += soil.moisture;
                return false;
            }
            true
        });
        self.cycle.add(crate::WaterBody::Sea, drained);
        self.cycle.evaporate(risen);
    }
}
