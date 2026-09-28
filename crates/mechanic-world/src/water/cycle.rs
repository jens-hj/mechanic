//! Where water comes from and where it goes: the rivers, lakes, sea and air
//! around stored water.
//!
//! The seed world is a steady state. Rain on each catchment runs down its
//! rivers, through its lakes and into the sea, and the sea gives the same
//! back to the air. Nothing about that steady state is stored or stepped.
//! What is stored is only how far each body has been moved from it: the
//! water a river reach, a lake, the sea or the air holds beyond its untouched
//! share, negative where water was drawn from it. That surplus runs on down
//! the network like the water itself, so a trench dug from a river draws no
//! more than the river brings, a river below a dam runs low until the dam
//! spills, and a drained lake stops spilling and fills again from its
//! inflow. Every cubic metre stays counted.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::WEIR_COEFFICIENT;
use crate::{LakeBasin, Outflow, RiverReach, TerrainField, WaterBody, WaterSurface};

/// Highest a body's surface rises over its seed level, in metres: the rest
/// of a surplus stands unseen. Under the shore margin, so no water stands
/// against air.
const RISE_METRES: f64 = 0.15;

/// How long water evaporated from stored water stays in the air before it
/// falls back, in seconds. It rains out over the land and runs to the sea;
/// the ledger books it to the sea directly.
const AIR_SECONDS: f64 = 3_600.0;

/// Surplus below which a body hands the rest on at once, in m³.
const SETTLED_M3: f64 = 1.0e-9;

/// The rivers and lakes a world's water drains through.
pub trait WaterNetwork {
    /// A lake as the water cycle sees it.
    fn lake(&self, lake: u32) -> Option<LakeBasin>;

    /// A river reach as the water cycle sees it.
    fn reach(&self, reach: u32) -> Option<RiverReach>;
}

impl WaterNetwork for TerrainField {
    fn lake(&self, lake: u32) -> Option<LakeBasin> {
        self.lake_basin(lake)
    }

    fn reach(&self, reach: u32) -> Option<RiverReach> {
        self.river_reach(reach)
    }
}

/// How far a body's surface stands from its seed level and how fast it runs
/// against its seed current.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterShift {
    /// How far its surface lies below its seed level, in metres; negative
    /// where it stands higher.
    pub drop: f64,
    /// Its current against its seed current.
    pub flow_scale: f64,
}

impl WaterShift {
    /// No change from the seed.
    pub const NONE: Self = Self {
        drop: 0.0,
        flow_scale: 1.0,
    };

    /// A seed surface moved by this shift.
    pub fn apply(self, mut surface: WaterSurface) -> WaterSurface {
        surface.level -= self.drop;
        surface.flow *= self.flow_scale;
        surface
    }
}

/// Every cubic metre of water in a world's books, beyond the seed world's
/// steady state. Only water poured in or drawn out from outside the world
/// changes the total.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WaterLedger {
    /// Water in stored pools, in m³.
    pub pools_m3: f64,
    /// Water running over the ground, in m³.
    pub running_m3: f64,
    /// Water in flight over drops, in m³.
    pub falling_m3: f64,
    /// Water in cells dug beside or under seed-derived water that joined
    /// it, in m³.
    pub joined_m3: f64,
    /// Water lakes hold beyond their seed levels, in m³.
    pub lakes_m3: f64,
    /// Water rivers carry beyond their seed flow, in m³.
    pub rivers_m3: f64,
    /// Water held in the ground, in m³.
    pub soil_m3: f64,
    /// Water the sea holds beyond its seed level, in m³.
    pub sea_m3: f64,
    /// Water evaporated from stored water, not yet fallen back, in m³.
    pub air_m3: f64,
}

impl WaterLedger {
    /// All of it, in m³.
    pub fn total(&self) -> f64 {
        self.pools_m3
            + self.running_m3
            + self.falling_m3
            + self.joined_m3
            + self.lakes_m3
            + self.rivers_m3
            + self.soil_m3
            + self.sea_m3
            + self.air_m3
    }
}

/// Water one body of seed-derived water holds beyond its untouched share,
/// in a saved world.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurplusDoc {
    /// The lake or river reach.
    pub body: WaterBody,
    /// Water it holds beyond its share, in m³; negative where water was
    /// drawn from it.
    pub volume_m3: f64,
}

/// The water around stored water: each body's surplus over the seed world.
#[derive(Clone, Debug, Default)]
pub(super) struct Cycle {
    /// Surplus of each lake and river reach, in m³.
    surplus: BTreeMap<WaterBody, f64>,
    /// Surplus of the sea, in m³.
    sea: f64,
    /// Water in the air, in m³.
    air: f64,
}

impl Cycle {
    pub(super) fn from_doc(bodies: &[SurplusDoc], sea: f64, air: f64) -> Self {
        Self {
            surplus: bodies.iter().map(|doc| (doc.body, doc.volume_m3)).collect(),
            sea,
            air,
        }
    }

    pub(super) fn to_doc(&self) -> Vec<SurplusDoc> {
        self.surplus
            .iter()
            .map(|(&body, &volume_m3)| SurplusDoc { body, volume_m3 })
            .collect()
    }

    pub(super) const fn sea(&self) -> f64 {
        self.sea
    }

    pub(super) const fn air(&self) -> f64 {
        self.air
    }

    /// Water a body holds beyond its seed share, in m³.
    pub(super) fn surplus(&self, body: WaterBody) -> f64 {
        match body {
            WaterBody::Sea => self.sea,
            _ => self.surplus.get(&body).copied().unwrap_or(0.0),
        }
    }

    /// Surplus of every lake and of every river reach, in m³.
    pub(super) fn totals(&self) -> (f64, f64) {
        self.surplus
            .iter()
            .fold((0.0, 0.0), |(lakes, rivers), (body, &volume)| match body {
                WaterBody::Lake(_) => (lakes + volume, rivers),
                _ => (lakes, rivers + volume),
            })
    }

    /// Most water that can be drawn from a body now, in m³: what a lake
    /// holds, what a river reach holds in its channel. The sea is endless
    /// to anything a world can hold.
    pub(super) fn available(&self, ground: &impl WaterNetwork, body: WaterBody) -> f64 {
        let held = match body {
            WaterBody::Sea => return f64::INFINITY,
            WaterBody::Pool(_) | WaterBody::Running => return 0.0,
            WaterBody::Lake(lake) => ground.lake(lake).map_or(0.0, |lake| lake.volume_m3),
            WaterBody::River(reach) => ground
                .reach(reach)
                .map_or(0.0, |reach| reach.discharge_m3_s * reach.travel_seconds),
        };
        (held + self.surplus(body)).max(0.0)
    }

    /// Adds water to a body; a negative volume draws it.
    pub(super) fn add(&mut self, body: WaterBody, volume: f64) {
        match body {
            WaterBody::Sea => self.sea += volume,
            WaterBody::Pool(_) | WaterBody::Running => {}
            _ => *self.surplus.entry(body).or_default() += volume,
        }
    }

    /// Water evaporated from stored water.
    pub(super) fn evaporate(&mut self, volume: f64) {
        self.air += volume;
    }

    /// How far a body stands from its seed surface.
    pub(super) fn shift(&self, ground: &impl WaterNetwork, body: WaterBody) -> WaterShift {
        let surplus = self.surplus(body);
        if surplus == 0.0 {
            return WaterShift::NONE;
        }
        match body {
            WaterBody::Lake(lake) => ground
                .lake(lake)
                .map_or(WaterShift::NONE, |lake| lake_shift(lake, surplus)),
            WaterBody::River(reach) => ground
                .reach(reach)
                .map_or(WaterShift::NONE, |reach| river_shift(reach, surplus)),
            WaterBody::Sea | WaterBody::Pool(_) | WaterBody::Running => WaterShift::NONE,
        }
    }

    /// How far every moved body stands from its seed surface.
    pub(super) fn shifts(&self, ground: &impl WaterNetwork) -> BTreeMap<WaterBody, WaterShift> {
        self.surplus
            .keys()
            .map(|&body| (body, self.shift(ground, body)))
            .filter(|(_, shift)| *shift != WaterShift::NONE)
            .collect()
    }

    /// Runs every surplus on down the network for `dt` seconds: each river
    /// reach passes on its surplus over its travel time, each lake spills
    /// its surplus over its rim, and the air rains back out.
    pub(super) fn step(&mut self, ground: &impl WaterNetwork, dt: f64) {
        let bodies = self.surplus.keys().copied().collect::<Vec<_>>();
        for body in bodies {
            let surplus = self.surplus(body);
            let (passed, outflow) = match body {
                WaterBody::River(reach) => {
                    let Some(reach) = ground.reach(reach) else {
                        continue;
                    };
                    (
                        surplus * (dt / reach.travel_seconds.max(dt)).min(1.0),
                        reach.outflow,
                    )
                }
                WaterBody::Lake(lake) => {
                    let Some(lake) = ground.lake(lake) else {
                        continue;
                    };
                    (lake_spill(lake, surplus, dt), lake.outflow)
                }
                WaterBody::Sea | WaterBody::Pool(_) | WaterBody::Running => continue,
            };
            let passed = if surplus.abs() < SETTLED_M3 {
                surplus
            } else {
                passed
            };
            self.pass(body, passed, outflow);
        }
        let fallen = if self.air.abs() < SETTLED_M3 {
            self.air
        } else {
            self.air * (dt / AIR_SECONDS).min(1.0)
        };
        self.air -= fallen;
        self.sea += fallen;
    }

    /// Moves surplus from a body to where it drains.
    fn pass(&mut self, body: WaterBody, volume: f64, outflow: Outflow) {
        if volume == 0.0 {
            return;
        }
        if let Some(surplus) = self.surplus.get_mut(&body) {
            *surplus -= volume;
            if *surplus == 0.0 {
                self.surplus.remove(&body);
            }
        }
        let to = match outflow {
            Outflow::River(reach) => WaterBody::River(reach),
            Outflow::Lake(lake) => WaterBody::Lake(lake),
            Outflow::Sea => WaterBody::Sea,
        };
        self.add(to, volume);
    }
}

/// A lake's surface after its surplus spreads over it.
fn lake_shift(lake: LakeBasin, surplus: f64) -> WaterShift {
    WaterShift {
        drop: (-surplus / lake.area_m2).max(-RISE_METRES),
        flow_scale: 1.0,
    }
}

/// A river reach's surface with its surplus flowing through it: depth
/// follows discharge to the power 0.6 and speed to the power 0.4, as in a
/// wide channel. A reach that carries nothing lies dry at its bed.
fn river_shift(reach: RiverReach, surplus: f64) -> WaterShift {
    let seed = reach.discharge_m3_s.max(f64::MIN_POSITIVE);
    let ratio = ((seed + surplus / reach.travel_seconds.max(1.0)) / seed).max(0.0);
    WaterShift {
        drop: (reach.depth * (1.0 - ratio.powf(0.6))).max(-RISE_METRES),
        flow_scale: ratio.powf(0.4),
    }
}

/// Surplus a lake spills over its rim in `dt`: the weir flow at its level
/// against the weir flow at its seed level, never more than it holds beyond
/// its seed share, nor more than it lacks.
fn lake_spill(lake: LakeBasin, surplus: f64, dt: f64) -> f64 {
    let rate = |head: f64| WEIR_COEFFICIENT * lake.spill_width * head.max(0.0).powf(1.5);
    let seed_head = (lake.discharge_m3_s / (WEIR_COEFFICIENT * lake.spill_width)).powf(2.0 / 3.0);
    let head = seed_head + surplus / lake.area_m2;
    let passed = (rate(head) - lake.discharge_m3_s) * dt;
    if surplus > 0.0 {
        passed.clamp(0.0, surplus)
    } else {
        passed.clamp(surplus, 0.0)
    }
}

/// The seed water in a column at its current level: `surface` moved by its
/// body's shift.
pub(super) fn shifted(
    shifts: &BTreeMap<WaterBody, WaterShift>,
    surface: WaterSurface,
) -> WaterSurface {
    shifts
        .get(&surface.body)
        .map_or(surface, |shift| shift.apply(surface))
}

#[cfg(test)]
mod tests {
    use super::{Cycle, WaterNetwork, lake_spill, river_shift};
    use crate::{LakeBasin, Outflow, RiverReach, WaterBody};

    /// A stream running into a lake, which spills into a river to the sea.
    struct Valley;

    impl WaterNetwork for Valley {
        fn lake(&self, lake: u32) -> Option<LakeBasin> {
            (lake == 0).then_some(LakeBasin {
                outflow: Outflow::River(1),
                ..LAKE
            })
        }

        fn reach(&self, reach: u32) -> Option<RiverReach> {
            let stream = RiverReach {
                discharge_m3_s: 1.0,
                travel_seconds: 60.0,
                depth: 1.0,
                outflow: Outflow::Lake(0),
            };
            match reach {
                0 => Some(stream),
                1 => Some(RiverReach {
                    outflow: Outflow::Sea,
                    ..stream
                }),
                _ => None,
            }
        }
    }

    fn run(cycle: &mut Cycle, seconds: u32) {
        for _ in 0..seconds * 20 {
            cycle.step(&Valley, 0.05);
        }
    }

    fn total(cycle: &Cycle) -> f64 {
        let (lakes, rivers) = cycle.totals();
        lakes + rivers + cycle.sea() + cycle.air()
    }

    const LAKE: LakeBasin = LakeBasin {
        area_m2: 10_000.0,
        volume_m3: 20_000.0,
        discharge_m3_s: 1.0,
        spill_width: 4.0,
        outflow: Outflow::Sea,
    };

    #[test]
    fn a_lower_lake_spills_less_and_a_higher_one_more() {
        let low = lake_spill(LAKE, -100.0, 1.0);
        let high = lake_spill(LAKE, 100.0, 1.0);
        assert!((-1.0..0.0).contains(&low), "low lake passed {low}");
        assert!(high > 0.0 && high <= 100.0, "high lake passed {high}");
        assert!(lake_spill(LAKE, -1.0e6, 1.0) >= -1.0, "spilt below nothing");
    }

    #[test]
    fn a_river_drawn_dry_falls_to_its_bed() {
        let reach = RiverReach {
            discharge_m3_s: 2.0,
            travel_seconds: 50.0,
            depth: 1.5,
            outflow: Outflow::Sea,
        };
        let dry = river_shift(reach, -100.0);
        assert!((dry.drop - 1.5).abs() < 1.0e-9 && dry.flow_scale.abs() < 1.0e-9);
        let half = river_shift(reach, -50.0);
        assert!(half.drop > 0.0 && half.drop < 1.5);
        assert!(river_shift(reach, 1.0e6).drop >= -0.15);
    }

    #[test]
    fn water_drawn_from_a_stream_shows_downstream_after_its_travel_time() {
        let mut cycle = Cycle::default();
        cycle.add(WaterBody::River(0), -30.0);
        run(&mut cycle, 1);
        assert!(
            cycle.surplus(WaterBody::Lake(0)) > -1.0,
            "the lake felt it at once"
        );
        run(&mut cycle, 120);
        assert!(
            cycle.surplus(WaterBody::Lake(0)) < -10.0,
            "the lake never felt it"
        );
        assert!(
            (total(&cycle) - -30.0).abs() < 1.0e-9,
            "water was made or lost"
        );
    }

    #[test]
    fn a_drained_lake_stops_spilling_and_refills_from_its_inflow() {
        let mut cycle = Cycle::default();
        cycle.add(WaterBody::Lake(0), -5_000.0);
        run(&mut cycle, 600);
        // It spills nothing while its inflow of 1 m³/s refills it, so the
        // river below it and then the sea run short by what it keeps.
        let lake = cycle.surplus(WaterBody::Lake(0));
        assert!(
            (lake - -4_400.0).abs() < 1.0,
            "the lake refilled to {lake:.1} m³"
        );
        let below = cycle.surplus(WaterBody::River(1)) + cycle.sea();
        assert!(
            (below - -600.0).abs() < 1.0,
            "downstream ran short by {below:.1} m³"
        );
        assert!(
            (total(&cycle) - -5_000.0).abs() < 1.0e-6,
            "water was made or lost"
        );
    }

    #[test]
    fn a_lake_raised_over_its_rim_spills_the_surplus_on() {
        let mut cycle = Cycle::default();
        cycle.add(WaterBody::Lake(0), 100.0);
        run(&mut cycle, 7_200);
        assert!(cycle.surplus(WaterBody::Lake(0)) < 5.0);
        assert!(cycle.sea() > 90.0, "the sea got {:.1} m³", cycle.sea());
        assert!((total(&cycle) - 100.0).abs() < 1.0e-6);
    }

    #[test]
    fn a_river_holds_only_what_its_channel_carries() {
        let cycle = Cycle::default();
        assert!((cycle.available(&Valley, WaterBody::River(0)) - 60.0).abs() < 1.0e-9);
        assert!((cycle.available(&Valley, WaterBody::Lake(0)) - 20_000.0).abs() < 1.0e-9);
    }
}
