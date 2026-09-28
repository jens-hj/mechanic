//! Water in the ground: running water and pools soak into soil, sand and
//! ground cover, which hold it in their pores and give it up slowly, deep
//! down to the sea and up into the air. Rock takes none.
//!
//! Each column of ground water has reached keeps how much it holds, how
//! much it can hold and how fast it takes water in. Infiltration slows as
//! the ground fills (Green and Ampt, simplified): dry ground drinks a film
//! quickly, saturated ground takes nothing, and water then runs on or
//! stands on it as mud.

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
    /// Water it holds as a depth over the column, in metres; running water
    /// on it counts in full.
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
    /// and ground running water covers, which is wet through.
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
            wet.insert(
                (cell.x, cell.z),
                WetGround {
                    column: (cell.x, cell.z),
                    top: sheet.floor(),
                    fill: 1.0,
                    soaked: f64::INFINITY,
                },
            );
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
            }
        })
    }

    /// Water soaking from running water and pools into the ground under
    /// them, and out of the ground deep down and into the air, over `dt`.
    pub(super) fn soak(&mut self, ground: &impl WaterGround, dt: f64) {
        for (slot, cell) in self.sheets.wet() {
            let sheet = *self.sheets.at(slot);
            let taken = self
                .soil_at(ground, cell, sheet.floor())
                .take(sheet.volume, dt);
            self.sheets.at_mut(slot).volume -= taken;
        }
        let beds = self
            .pools
            .iter()
            .flat_map(|(&id, pool)| pool.beds().map(move |(cell, floor)| (id, cell, floor)))
            .collect::<Vec<_>>();
        for (id, cell, floor) in beds {
            let offered = self.pools.get(&id).map_or(0.0, |pool| pool.volume);
            let taken = self.soil_at(ground, cell, floor).take(offered, dt);
            if let Some(pool) = self.pools.get_mut(&id) {
                pool.volume -= taken;
            }
        }
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
