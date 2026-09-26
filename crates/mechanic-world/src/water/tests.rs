//! Stored water: pools settle level, fill and spill, fill from lakes, follow
//! the ground as it changes, and survive a save.

use bevy_math::{DVec2, DVec3};

use super::{WaterGround, WaterNetwork, WaterWorld};
use crate::{
    BRICK_EDGE_CELLS, BrickCoord, LakeBasin, Outflow, RiverReach, TERRAIN_CELL_METERS, WaterBody,
    WaterSurface,
};

/// An axis-aligned box of open space.
#[derive(Clone, Copy)]
struct Room {
    minimum: DVec3,
    maximum: DVec3,
}

const fn room(minimum: [f64; 3], maximum: [f64; 3]) -> Room {
    Room {
        minimum: DVec3::from_array(minimum),
        maximum: DVec3::from_array(maximum),
    }
}

/// Ground solid everywhere but in its rooms, with an optional lake: open
/// ground where `lake_room` holds it, below `lake_level`.
struct Ground {
    rooms: Vec<Room>,
    lake: Option<(Room, f64)>,
    /// Makes the lake a river reach carrying this.
    river: Option<RiverReach>,
}

impl Ground {
    fn open(&self, point: DVec3) -> bool {
        self.rooms
            .iter()
            .chain(self.lake.iter().map(|(room, _)| room))
            .any(|room| point.cmpge(room.minimum).all() && point.cmplt(room.maximum).all())
    }

    fn lake_at(&self, x: f64, z: f64) -> Option<WaterSurface> {
        let (room, level) = self.lake?;
        let inside =
            x >= room.minimum.x && x < room.maximum.x && z >= room.minimum.z && z < room.maximum.z;
        inside.then_some(WaterSurface {
            level,
            body: if self.river.is_some() {
                WaterBody::River(0)
            } else {
                WaterBody::Lake(0)
            },
            flow: DVec2::ZERO,
        })
    }

    /// Every brick a box of rooms touches.
    fn bricks(&self) -> Vec<BrickCoord> {
        let edge = f64::from(BRICK_EDGE_CELLS) * TERRAIN_CELL_METERS;
        let mut bricks = Vec::new();
        for room in &self.rooms {
            #[expect(clippy::cast_possible_truncation, reason = "a few bricks")]
            let index = |value: f64| (value / edge).floor() as i32;
            for z in index(room.minimum.z)..=index(room.maximum.z) {
                for y in index(room.minimum.y)..=index(room.maximum.y) {
                    for x in index(room.minimum.x)..=index(room.maximum.x) {
                        bricks.push(BrickCoord::new(x, y, z));
                    }
                }
            }
        }
        bricks.sort_by_key(|brick| (brick.x, brick.y, brick.z));
        bricks.dedup();
        bricks
    }
}

/// The test lake: 100 m square, 2 m deep on average, spilling to the sea
/// only what stands over its rim.
const LAKE: LakeBasin = LakeBasin {
    area_m2: 10_000.0,
    volume_m3: 20_000.0,
    discharge_m3_s: 0.0,
    spill_width: 4.0,
    outflow: Outflow::Sea,
};

impl WaterNetwork for Ground {
    fn lake(&self, _lake: u32) -> Option<LakeBasin> {
        Some(LAKE)
    }

    fn reach(&self, _reach: u32) -> Option<RiverReach> {
        self.river
    }
}

impl WaterGround for Ground {
    fn open_cells(&self, brick: BrickCoord) -> Vec<bool> {
        let minimum = brick.minimum_cell();
        let edge = BRICK_EDGE_CELLS;
        let mut open = Vec::new();
        for z in 0..edge {
            for y in 0..edge {
                for x in 0..edge {
                    let centre = crate::WorldCell::new(minimum.x + x, minimum.y + y, minimum.z + z)
                        .centre()
                        .0;
                    open.push(self.open(centre));
                }
            }
        }
        open
    }

    fn implicit(&self, point: DVec3) -> Option<WaterSurface> {
        // Seed-derived water knows only the untouched lake, never a room dug
        // beside or under it.
        let (room, _) = self.lake?;
        let untouched = point.cmpge(room.minimum).all() && point.cmplt(room.maximum).all();
        self.lake_at(point.x, point.z)
            .filter(|surface| point.y < surface.level && untouched)
    }

    fn surface(&self, x: f64, z: f64) -> Option<WaterSurface> {
        self.lake_at(x, z)
    }

    fn may_hold_water(&self, _brick: BrickCoord) -> bool {
        self.lake.is_some()
    }
}

/// Runs the water for `seconds` in steps of a twentieth of a second.
fn run(water: &mut WaterWorld, ground: &Ground, seconds: u32) {
    for _ in 0..seconds * 20 {
        water.step(ground, 0.05);
    }
}

/// Water drawn from the test lake, in m³.
fn drawn(water: &WaterWorld) -> f64 {
    -water.surplus_m3(WaterBody::Lake(0))
}

fn level_at(water: &WaterWorld, ground: &Ground, point: DVec3) -> f64 {
    water
        .surface(ground, point)
        .map_or(f64::NEG_INFINITY, |surface| surface.level)
}

/// Two 40 cm shafts 3 m tall, joined by a tunnel along their feet.
fn u_tube() -> Ground {
    Ground {
        rooms: vec![
            room([0.0, 0.0, 0.0], [0.4, 3.0, 0.4]),
            room([2.0, 0.0, 0.0], [2.4, 3.0, 0.4]),
            room([0.0, 0.0, 0.0], [2.4, 0.4, 0.4]),
        ],
        lake: None,
        river: None,
    }
}

#[test]
fn water_poured_into_a_u_tube_settles_level_in_both_arms() {
    let ground = u_tube();
    let mut water = WaterWorld::new();
    assert!((water.deposit(&ground, DVec3::new(0.2, 2.9, 0.2), 0.5) - 0.5).abs() < f64::EPSILON);
    run(&mut water, &ground, 10);
    // The tunnel holds 0.384 m³; the rest stands 0.3625 m up both 0.16 m² arms.
    let left = level_at(&water, &ground, DVec3::new(0.2, 0.6, 0.2));
    let right = level_at(&water, &ground, DVec3::new(2.2, 0.6, 0.2));
    assert!((left - 0.7625).abs() < 0.02, "left arm at {left:.3} m");
    assert!(
        (left - right).abs() < 1.0e-9,
        "arms at {left:.3} and {right:.3} m"
    );
    assert!((water.ledger().total() - 0.5).abs() < 1.0e-9);
}

/// A 1 m pit in a floor with walls round it, and a lower basin to the east
/// beyond a 20 cm rim.
fn pit_and_basin() -> Ground {
    Ground {
        rooms: vec![
            room([0.0, -1.0, 0.0], [1.0, 2.0, 1.0]),
            room([-0.4, 0.0, -0.4], [1.0, 2.0, 1.4]),
            room([1.2, -0.6, -0.4], [4.0, 2.0, 1.4]),
            room([1.0, 0.0, -0.4], [1.2, 2.0, 1.4]),
        ],
        lake: None,
        river: None,
    }
}

#[test]
fn a_pit_fills_then_spills_over_its_rim_into_lower_ground() {
    let ground = pit_and_basin();
    let mut water = WaterWorld::new();
    water.deposit(&ground, DVec3::new(0.5, 1.5, 0.5), 1.5);
    run(&mut water, &ground, 60);
    let pit = level_at(&water, &ground, DVec3::new(0.5, -0.5, 0.5));
    let basin = level_at(&water, &ground, DVec3::new(3.0, -0.5, 0.5));
    assert!(pit > -0.02, "the pit stands at {pit:.3} m");
    assert!(basin > -0.6, "nothing spilled into the basin");
    assert!(
        basin < 0.0,
        "the basin filled to {basin:.3} m, over the rim"
    );
    assert!(
        (water.ledger().total() - 1.5).abs() < 1.0e-9,
        "water was made or lost"
    );
}

/// A lake to the west, and a trench dug east from it into a bank, with a
/// cave under the bank's far end.
fn lake_and_trench(cave: bool) -> Ground {
    let mut rooms = vec![room([0.0, -0.5, 0.0], [4.0, 3.0, 0.6])];
    if cave {
        rooms.push(room([4.0, -3.0, -1.0], [8.0, 0.0, 2.0]));
    }
    rooms.push(room([-20.0, 1.0, -20.0], [20.0, 3.0, 20.0]));
    Ground {
        rooms,
        lake: Some((room([-20.0, -2.0, -20.0], [0.0, 3.0, 20.0]), 0.8)),
        river: None,
    }
}

#[test]
fn a_trench_dug_from_a_lake_fills_to_the_lake_level_and_joins_it() {
    let ground = lake_and_trench(false);
    let mut water = WaterWorld::new();
    water.terrain_changed(&ground, ground.bricks());
    run(&mut water, &ground, 30);
    let trench = water
        .surface(&ground, DVec3::new(3.0, 0.0, 0.3))
        .expect("the trench holds water");
    assert_eq!(
        trench.body,
        WaterBody::Lake(0),
        "the trench did not join the lake"
    );
    assert!(
        (trench.level - 0.8).abs() < 0.01,
        "the trench stands at {:.3} m",
        trench.level
    );
    let held = water.stored_m3() + water.joined_m3();
    assert!(held > 2.5, "the trench holds only {held:.3} m³");
    // Joined cells follow the lake as it drops, which its hollow's area does
    // not count: the books close to the drop over their area.
    assert!(
        (drawn(&water) - held).abs() < 0.01,
        "the trench holds {held:.3} m³ but the lake gave {:.3} m³",
        drawn(&water)
    );
}

#[test]
fn a_breached_lake_drains_into_a_cave_until_the_levels_meet() {
    let ground = lake_and_trench(true);
    let mut water = WaterWorld::new();
    water.terrain_changed(&ground, ground.bricks());
    run(&mut water, &ground, 120);
    let held = water.stored_m3() + water.joined_m3();
    assert!(held > 36.0, "the cave holds {held:.1} m³");
    let cave = level_at(&water, &ground, DVec3::new(6.0, -2.0, 0.5));
    let lake = level_at(&water, &ground, DVec3::new(-5.0, 0.0, 0.5));
    assert!(
        (cave - lake).abs() < 0.01,
        "cave at {cave:.3} m, lake at {lake:.3} m"
    );
    assert!(lake < 0.8, "the lake did not go down");
    assert!((drawn(&water) - held).abs() < 0.05);
}

/// A lake over a bed 2 m down, with a 1 m hole dug into the bed.
fn lake_with_hole() -> Ground {
    Ground {
        rooms: vec![room([0.0, -3.0, 0.0], [1.0, -2.0, 1.0])],
        lake: Some((room([-10.0, -2.0, -10.0], [10.0, 3.0, 10.0]), 0.8)),
        river: None,
    }
}

#[test]
fn a_hole_dug_under_a_lake_fills_and_becomes_lake() {
    let ground = lake_with_hole();
    let mut water = WaterWorld::new();
    water.terrain_changed(&ground, ground.bricks());
    run(&mut water, &ground, 30);
    assert_eq!(
        water.pools().count(),
        0,
        "the hole is still a pool of its own"
    );
    let hole = water
        .surface(&ground, DVec3::new(0.5, -2.5, 0.5))
        .expect("the hole holds water");
    assert_eq!(hole.body, WaterBody::Lake(0));
    // The lake gave the hole its 1 m³ and dropped by that over its area.
    assert!(
        (drawn(&water) - 1.0).abs() < 0.02,
        "drawn {:.3} m³",
        drawn(&water)
    );
    assert!((hole.level - (0.8 - drawn(&water) / 10_000.0)).abs() < 1.0e-9);
}

#[test]
fn water_poured_in_from_a_lake_never_stands_above_the_lake() {
    let ground = lake_with_hole();
    let mut water = WaterWorld::new();
    water.terrain_changed(&ground, ground.bricks());
    for step in 0..600 {
        water.step(&ground, 0.05);
        for pool in water.pools() {
            assert!(
                pool.level < 0.8 + 0.01,
                "step {step}: pool {} stands at {:.3} m over a lake at 0.8 m",
                pool.id,
                pool.level
            );
        }
    }
}

/// A closed 1 m box open at the top, and the same with a block in one half
/// of its floor.
fn pit(filled: bool) -> Ground {
    let mut rooms = vec![room([0.0, -1.0, 0.0], [1.0, 1.0, 1.0])];
    if filled {
        rooms = vec![
            room([0.5, -1.0, 0.0], [1.0, 1.0, 1.0]),
            room([0.0, -0.6, 0.0], [0.5, 1.0, 1.0]),
        ];
    }
    Ground {
        rooms,
        lake: None,
        river: None,
    }
}

#[test]
fn ground_filled_into_a_pool_raises_its_level() {
    let mut water = WaterWorld::new();
    water.deposit(&pit(false), DVec3::new(0.5, 0.5, 0.5), 0.5);
    run(&mut water, &pit(false), 5);
    let before = level_at(&water, &pit(false), DVec3::new(0.7, -0.8, 0.5));
    assert!(
        (before + 0.5).abs() < 0.01,
        "the pit stands at {before:.3} m"
    );
    let filled = pit(true);
    water.terrain_changed(&filled, filled.bricks());
    run(&mut water, &filled, 5);
    // 0.2 m³ fills the lower 0.4 m of the open half; 0.3 m³ stands on the full floor.
    let after = level_at(&water, &filled, DVec3::new(0.7, -0.8, 0.5));
    assert!((after + 0.3).abs() < 0.01, "the pit stands at {after:.3} m");
    assert!((water.ledger().total() - 0.5).abs() < 1.0e-9);
}

#[test]
fn withdrawing_from_a_pool_lowers_its_level() {
    let ground = pit(false);
    let mut water = WaterWorld::new();
    water.deposit(&ground, DVec3::new(0.5, 0.5, 0.5), 0.5);
    run(&mut water, &ground, 5);
    let taken = water.withdraw(&ground, DVec3::new(0.5, -0.9, 0.5), 0.2);
    assert!((taken - 0.2).abs() < 1.0e-12);
    let level = level_at(&water, &ground, DVec3::new(0.5, -0.9, 0.5));
    assert!((level + 0.7).abs() < 0.01, "the pit stands at {level:.3} m");
}

#[test]
fn stored_water_survives_a_save() {
    let ground = u_tube();
    let mut water = WaterWorld::new();
    water.deposit(&ground, DVec3::new(0.2, 2.9, 0.2), 0.5);
    run(&mut water, &ground, 10);
    let loaded = WaterWorld::from_doc(&ground, &water.to_doc());
    let point = DVec3::new(2.2, 0.6, 0.2);
    assert!((level_at(&loaded, &ground, point) - level_at(&water, &ground, point)).abs() < 1.0e-9);
    assert!((loaded.ledger().total() - water.ledger().total()).abs() < 1.0e-9);
    assert!((loaded.ledger().total() - 0.5).abs() < 1.0e-9);
}

#[test]
fn a_trench_dug_from_a_river_draws_no_more_than_the_river_carries() {
    // A stream carrying 10 litres a second, a minute's run long: 0.6 m³ in
    // its channel, and 0.6 m³ more over the minute.
    let mut ground = lake_and_trench(true);
    ground.river = Some(RiverReach {
        discharge_m3_s: 0.01,
        travel_seconds: 60.0,
        depth: 1.5,
        outflow: Outflow::Sea,
    });
    let mut water = WaterWorld::new();
    water.terrain_changed(&ground, ground.bricks());
    run(&mut water, &ground, 60);
    let held = water.stored_m3() + water.joined_m3();
    assert!(held > 0.5, "only {held:.2} m³ ran in");
    assert!(held < 1.2 + 1.0e-6, "the stream gave {held:.2} m³");
    let ledger = water.ledger();
    assert!(ledger.total().abs() < 1.0e-9, "water was made or lost");
    let stream = level_at(&water, &ground, DVec3::new(-5.0, 0.0, 0.5));
    assert!(stream < 0.8 - 0.2, "the stream stands at {stream:.2} m");
}

#[test]
fn a_lake_gives_no_more_than_it_holds() {
    let ground = lake_with_hole();
    let mut water = WaterWorld::new();
    let taken = water.withdraw(&ground, DVec3::new(-5.0, 0.0, -5.0), 50_000.0);
    assert!(
        (taken - LAKE.volume_m3).abs() < 1.0e-6,
        "took {taken:.1} m³"
    );
    assert!(water.withdraw(&ground, DVec3::new(-5.0, 0.0, -5.0), 1.0) < 1.0e-9);
}

#[test]
fn a_puddle_evaporates_into_the_air() {
    let ground = pit(false);
    let mut water = WaterWorld::new();
    // A 1 cm puddle on a 1 m² floor dries in two hours.
    water.deposit(&ground, DVec3::new(0.5, 0.5, 0.5), 0.01);
    for _ in 0..3 * 60 {
        water.step(&ground, 60.0);
    }
    assert_eq!(water.pools().count(), 0, "the puddle is still there");
    let ledger = water.ledger();
    assert!(
        (ledger.total() - 0.01).abs() < 1.0e-9,
        "water was made or lost"
    );
    assert!(ledger.air_m3 + ledger.sea_m3 > 0.01 - 1.0e-9);
}

mod terrain;
