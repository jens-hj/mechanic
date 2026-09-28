//! Stored water: pools settle level, fill and spill, fill from lakes, follow
//! the ground as it changes, and survive a save.

use bevy_math::{DVec2, DVec3};

use super::ground::CELL_TERRAIN_CELLS;
use super::{WATER_CELL_EDGE_CELLS, WaterCell, WaterGround, WaterNetwork, WaterWorld};
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
    /// Ground of bare rock, which takes in no water, rather than soil: most
    /// tests keep their water out of the ground.
    rock: bool,
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
    fn open_cells(&self, cell: WaterCell) -> [bool; CELL_TERRAIN_CELLS] {
        let edge = WATER_CELL_EDGE_CELLS;
        let mut open = [false; CELL_TERRAIN_CELLS];
        for (index, open) in open.iter_mut().enumerate() {
            let index = i32::try_from(index).expect("64 cells");
            let centre = crate::WorldCell::new(
                cell.x * edge + index % 4,
                cell.y * edge + index / 4 % 4,
                cell.z * edge + index / 16,
            )
            .centre()
            .0;
            *open = self.open(centre);
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

    fn edited(&self, _brick: BrickCoord) -> bool {
        true
    }

    fn dug(&self, _cell: WaterCell) -> bool {
        // Every room is dug out of solid ground.
        true
    }

    fn material(&self, point: DVec3) -> Option<crate::TerrainMaterial> {
        (!self.open(point)).then_some(if self.rock {
            crate::TerrainMaterial::Rock
        } else {
            crate::TerrainMaterial::Soil
        })
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

/// Water held in pools, joined cells, running water and falls, in m³.
fn held(water: &WaterWorld) -> f64 {
    water.stored_m3() + water.joined_m3() + water.running_m3() + water.ledger().falling_m3
}

/// Water standing or running with its surface below `height`, in m³.
fn held_below(water: &WaterWorld, height: f64) -> f64 {
    let pools = water
        .pools()
        .filter(|pool| pool.level < height)
        .map(|pool| pool.volume_m3)
        .sum::<f64>();
    let area = super::WATER_CELL_METRES * super::WATER_CELL_METRES;
    let running = water
        .running_cells()
        .into_iter()
        .filter(|running| running.level < height)
        .map(|running| running.depth * area)
        .sum::<f64>();
    pools + running
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
        rock: true,
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
        rock: true,
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
        rock: true,
    }
}

#[test]
fn a_trench_dug_from_a_lake_fills_to_the_lake_level() {
    let ground = lake_and_trench(false);
    let mut water = WaterWorld::new();
    water.terrain_changed(&ground, ground.bricks());
    run(&mut water, &ground, 30);
    let trench = water
        .surface(&ground, DVec3::new(3.0, 0.0, 0.3))
        .expect("the trench holds water");
    // Beside the lake, not under it: running water at the lake's level.
    assert_eq!(trench.body, WaterBody::Running);
    assert!(
        (trench.level - 0.8).abs() < 0.01,
        "the trench stands at {:.3} m",
        trench.level
    );
    let held = held(&water);
    assert!(held > 2.5, "the trench holds only {held:.3} m³");
    // What the lake gave is in the trench, or risen off it into the air and
    // come down on the sea.
    let ledger = water.ledger();
    assert!(
        (drawn(&water) - held - ledger.air_m3 - ledger.sea_m3).abs() < 1.0e-9,
        "the trench holds {held:.3} m³ but the lake gave {:.3} m³",
        drawn(&water)
    );
}

#[test]
fn a_breached_lake_drains_into_a_cave_until_the_levels_meet() {
    let ground = lake_and_trench(true);
    let mut water = WaterWorld::new();
    water.terrain_changed(&ground, ground.bricks());
    // The last few centimetres run through the trench at a few litres a
    // second.
    run(&mut water, &ground, 240);
    let held = held(&water);
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

/// A shallow channel dug from a lake, its floor 10 cm under the lake,
/// opening into a pit a metre deeper.
fn lake_channel_and_pit() -> Ground {
    Ground {
        rooms: vec![
            room([0.0, 0.7, 0.0], [3.0, 3.0, 0.6]),
            room([3.0, -0.5, -0.4], [4.6, 3.0, 1.0]),
            room([-20.0, 1.0, -20.0], [20.0, 3.0, 20.0]),
        ],
        lake: Some((room([-20.0, -2.0, -20.0], [0.0, 3.0, 20.0]), 0.8)),
        river: None,
        rock: true,
    }
}

#[test]
fn a_channel_from_a_lake_into_a_pit_fills_and_comes_to_rest() {
    let ground = lake_channel_and_pit();
    let mut water = WaterWorld::new();
    water.terrain_changed(&ground, ground.bricks());
    // The channel runs into the pit at about 7 L/s, water 5 cm deep at a
    // quarter of a metre a second: it takes some seven minutes to fill.
    run(&mut water, &ground, 600);
    let step = water.step(&ground, 0.05);
    assert!(step.falls.is_empty(), "water still falls into the pit");
    // Only a trickle from the lake, making up what rises into the air.
    assert!(
        step.moved_m3 < 5.0e-5,
        "{:.2e} m³ still moves each step",
        step.moved_m3
    );
    // Running water at rest at the lake's level, fed through the channel.
    for point in [DVec3::new(1.5, 0.75, 0.3), DVec3::new(3.8, 0.0, 0.3)] {
        let level = level_at(&water, &ground, point);
        assert!(
            (level - 0.8).abs() < 0.01,
            "at {point} the dig stands at {level:.3} m"
        );
    }
    // The pit's 3.1 m³ below the channel came from the lake.
    let held = held(&water);
    assert!(held > 3.0, "the dig holds only {held:.3} m³");
    let ledger = water.ledger();
    assert!((drawn(&water) - held - ledger.air_m3 - ledger.sea_m3).abs() < 1.0e-9);
}

/// A lake over a bed 2 m down, with a 1 m hole dug into the bed.
fn lake_with_hole() -> Ground {
    Ground {
        rooms: vec![room([0.0, -3.0, 0.0], [1.0, -2.0, 1.0])],
        lake: Some((room([-10.0, -2.0, -10.0], [10.0, 3.0, 10.0]), 0.8)),
        river: None,
        rock: true,
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
        rock: true,
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
    let held =
        water.stored_m3() + water.joined_m3() + water.running_m3() + water.ledger().falling_m3;
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
    assert!(
        water.pools().count() == 0 && water.running_m3() < 1.0e-9,
        "the puddle is still there"
    );
    let ledger = water.ledger();
    assert!(
        (ledger.total() - 0.01).abs() < 1.0e-9,
        "water was made or lost"
    );
    assert!(ledger.air_m3 + ledger.sea_m3 > 0.01 - 1.0e-9);
}

/// A stair of eight 20 cm steps, 40 cm deep, down to a pit 1.2 m below the
/// last step: a slope a sheet runs down.
fn stair() -> Ground {
    let mut rooms = (0..8)
        .map(|step| {
            let x = f64::from(step) * 0.4;
            room([x, -0.2 * f64::from(step), 0.0], [x + 0.4, 3.0, 0.4])
        })
        .collect::<Vec<_>>();
    rooms.push(room([3.2, -3.0, 0.0], [4.0, 3.0, 0.4]));
    Ground {
        rooms,
        lake: None,
        river: None,
        rock: true,
    }
}

#[test]
fn water_spilled_on_a_slope_runs_down_it_before_it_pools() {
    let ground = stair();
    let mut water = WaterWorld::new();
    // A bucketful, 12 L: a film a few centimetres deep, and what clings to
    // the steps stays behind.
    let spilled = 0.012;
    water.deposit(&ground, DVec3::new(0.2, 1.0, 0.2), spilled);
    run_steps(&mut water, &ground, 10);
    assert_eq!(water.pools().count(), 0, "the water pooled at once");
    let fastest = water
        .running_cells()
        .into_iter()
        .map(|running| running.flow.x)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        fastest > 0.05,
        "the water runs at most {fastest:.3} m/s downhill"
    );
    // A save in mid-run keeps every drop.
    let saved = WaterWorld::from_doc(&ground, &water.to_doc());
    assert!((saved.ledger().total() - spilled).abs() < 1.0e-9);
    run(&mut water, &ground, 30);
    // A film up to 2 mm deep clings to each of the eight steps, some 2.6 L
    // in all; the rest runs down into the pit.
    let pit = held_below(&water, -1.5);
    assert!(pit > 0.7 * spilled, "the pit holds {pit:.3} m³");
    assert!(
        (water.ledger().total() - spilled).abs() < 1.0e-9,
        "water was made or lost"
    );
}

/// A slope falling one in four over 4 m, 2.4 m wide, its ground in 20 cm
/// columns rough by up to 4 cm.
fn rough_slope() -> Ground {
    let mut rooms = Vec::new();
    for i in 0..20 {
        for k in 0..12 {
            let (x, z) = (f64::from(i) * 0.2, f64::from(k) * 0.2);
            // A fixed scatter standing in for the lumps of real ground.
            let lump = f64::from((i * 7 + k * 13 + i * k * 5) % 9) / 8.0 * 0.04;
            let height = -x / 4.0 + lump;
            rooms.push(room([x, height, z], [x + 0.2, 2.0, z + 0.2]));
        }
    }
    rooms.push(room([4.0, -3.0, 0.0], [5.0, 2.0, 2.4]));
    Ground {
        rooms,
        lake: None,
        river: None,
        rock: true,
    }
}

#[test]
fn a_trickle_down_rough_ground_gathers_into_rills() {
    let ground = rough_slope();
    let mut water = WaterWorld::new();
    let mut discharge = std::collections::HashMap::<(i32, i32), f64>::new();
    for step in 0..400 {
        water.deposit(&ground, DVec3::new(0.3, 1.0, 1.2), 0.00005);
        water.step(&ground, 0.05);
        if step >= 200 {
            for running in water.running_cells() {
                *discharge
                    .entry((running.cell.x, running.cell.z))
                    .or_default() += running.depth * running.flow.length();
            }
        }
    }
    let mut shares = discharge.values().copied().collect::<Vec<_>>();
    shares.sort_by(|a, b| b.total_cmp(a));
    let total = shares.iter().sum::<f64>();
    let mut carried = 0.0;
    let carrying = shares
        .iter()
        .take_while(|&&share| {
            carried += share;
            carried - share < 0.8 * total
        })
        .count();
    // A litre a second gathers into a few paths down the lumps: most of the
    // ground it wets carries little of it.
    assert!(
        carrying * 10 < shares.len() * 3,
        "{carrying} of {} wetted columns carry 80% of the flow",
        shares.len()
    );
}

#[test]
fn a_flood_down_rough_ground_runs_as_one_sheet_without_pools() {
    let ground = rough_slope();
    let mut water = WaterWorld::new();
    for _ in 0..600 {
        water.deposit(&ground, DVec3::new(0.3, 1.0, 1.2), 0.0005);
        water.step(&ground, 0.05);
        // Lumps on open ground are no pits: water fills and runs over them
        // as running water, never as flat pools stepping down the slope.
        assert_eq!(water.pools().count(), 0, "the flood broke into pools");
    }
    let wet = water.running_cells();
    assert!(wet.len() > 100, "only {} columns are wet", wet.len());
}

#[test]
fn a_pond_in_a_hollow_settles_flat_as_still_running_water() {
    let ground = pit(false);
    let mut water = WaterWorld::new();
    // Poured in over three seconds, as a hose would.
    for _ in 0..60 {
        water.deposit(&ground, DVec3::new(0.5, 0.5, 0.5), 0.005);
        water.step(&ground, 0.05);
    }
    run(&mut water, &ground, 20);
    assert_eq!(water.pools().count(), 0, "the pond became a pool");
    let levels = water
        .running_cells()
        .into_iter()
        .map(|running| running.level)
        .collect::<Vec<_>>();
    let (low, high) = levels
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), &level| {
            (low.min(level), high.max(level))
        });
    assert_eq!(levels.len(), 25, "the pond covers {} columns", levels.len());
    assert!(
        high - low < 0.001,
        "the pond lies from {low:.4} to {high:.4} m"
    );
    // 0.3 m³ over the square metre floor at -1 m.
    assert!((low + 0.7).abs() < 0.005, "the pond stands at {low:.3} m");
    // Still water costs nothing to run.
    let slot = water.sheets.slot(2, 2).expect("the pond's tile");
    assert!(water.sheets.asleep(slot.tile), "the still pond is awake");
}

/// A lake to the west, a bank a metre high, and beyond it a flat field
/// 10 cm under the lake, with a 40 cm breach cut through the bank.
fn lake_bank_and_field() -> Ground {
    Ground {
        rooms: vec![
            room([-20.0, 1.0, -20.0], [20.0, 3.0, 20.0]),
            room([0.4, 0.7, -6.0], [12.0, 3.0, 6.0]),
            room([0.0, 0.7, -0.2], [0.4, 3.0, 0.2]),
        ],
        lake: Some((room([-20.0, -2.0, -20.0], [0.0, 3.0, 20.0]), 0.8)),
        river: None,
        rock: true,
    }
}

/// How far the farthest stored water lies from a point, in metres.
fn front(water: &WaterWorld, ground: &Ground, from: DVec2) -> f64 {
    let distance = |cell: WaterCell| {
        let centre = cell.centre();
        (DVec2::new(centre.x, centre.z) - from).length()
    };
    let running = water
        .running_cells()
        .into_iter()
        .map(|running| running.cell);
    let pools = water.pools().flat_map(|pool| pool.surface_cells);
    let joined = water.joined_cells(ground).into_iter().map(|(cell, _)| cell);
    running
        .chain(pools)
        .chain(joined)
        .map(distance)
        .fold(0.0, f64::max)
}

#[test]
fn a_breach_floods_a_field_as_a_front_moving_at_shallow_water_speed() {
    let ground = lake_bank_and_field();
    let mut water = WaterWorld::new();
    water.terrain_changed(&ground, ground.bricks());
    let breach = DVec2::new(0.4, 0.0);
    let mut fronts = Vec::new();
    for _ in 0..30 {
        run(&mut water, &ground, 1);
        fronts.push(front(&water, &ground, breach));
    }
    // Water 10 cm deep over grass spreads at a few tens of centimetres a
    // second, slowing as it thins: it never races along the ground at the
    // lake's level.
    assert!(fronts[0] < 1.0, "a metre out after a second: {fronts:.2?}");
    assert!(
        fronts[9] < 4.0,
        "four metres out after ten seconds: {fronts:.2?}"
    );
    assert!(
        fronts[29] > fronts[9] + 0.5,
        "the flood stopped: {fronts:.2?}"
    );
    assert_eq!(
        water.joined_cells(&ground).len(),
        0,
        "the field joined the lake"
    );
    let ledger = water.ledger();
    assert!(ledger.total().abs() < 1.0e-9, "water was made or lost");
}

/// A flat floor 2 m square, of soil or of rock.
fn floor(rock: bool) -> Ground {
    Ground {
        rooms: vec![room([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])],
        lake: None,
        river: None,
        rock,
    }
}

#[test]
fn a_film_soaks_into_soil_but_stays_on_rock() {
    for rock in [false, true] {
        let ground = floor(rock);
        let mut water = WaterWorld::new();
        let spilled = 0.002;
        water.deposit(&ground, DVec3::new(1.0, 0.5, 1.0), spilled);
        for step in 0..2_400 {
            water.step(&ground, 0.05);
            assert!(
                (water.ledger().total() - spilled).abs() < 1.0e-12,
                "step {step} made or lost water"
            );
        }
        let above = water.running_m3() + water.stored_m3();
        if rock {
            // Only the air takes some of the film spread over the floor.
            assert!(
                above > 0.8 * spilled,
                "rock drank {:.2e} m³",
                spilled - above
            );
            assert!(water.soil_m3() < 1.0e-12);
        } else {
            assert!(
                above < 0.01 * spilled,
                "{above:.2e} m³ still stands on soil"
            );
            assert!(water.soil_m3() > 0.9 * spilled);
            assert!(!water.is_mud(5, 5), "two litres made mud");
        }
    }
}

#[test]
fn a_flood_on_soil_turns_it_to_mud_and_the_rest_stands() {
    let ground = pit(false);
    let ground = Ground {
        rock: false,
        ..ground
    };
    let mut water = WaterWorld::new();
    water.deposit(&ground, DVec3::new(0.5, -0.5, 0.5), 0.4);
    // Soil under standing water fills at a few centimetres an hour.
    run(&mut water, &ground, 3_600);
    // A metre square of soil half a metre deep holds some 175 litres.
    let soil = water.soil_m3();
    assert!((0.14..0.18).contains(&soil), "the soil holds {soil:.3} m³");
    assert!(water.is_mud(2, 2), "saturated soil is not mud");
    // Saturated soil keeps draining deep and drinking a little more.
    let standing = water.stored_m3();
    assert!(standing > 0.15, "only {standing:.3} m³ stands on the mud");
}

/// Runs the water for `steps` twentieths of a second.
fn run_steps(water: &mut WaterWorld, ground: &Ground, steps: u32) {
    for _ in 0..steps {
        water.step(ground, 0.05);
    }
}

#[test]
fn a_pool_spills_over_its_rim_and_runs_down_the_slope_beyond() {
    // A basin 40 cm deep behind the stair's top step, which is its rim.
    let mut ground = stair();
    ground.rooms.push(room([-0.8, -0.4, 0.0], [0.0, 3.0, 0.4]));
    let mut water = WaterWorld::new();
    water.deposit(&ground, DVec3::new(-0.4, 1.0, 0.2), 0.2);
    run(&mut water, &ground, 60);
    let basin = level_at(&water, &ground, DVec3::new(-0.4, -0.2, 0.2));
    assert!(
        (-0.01..0.05).contains(&basin),
        "the basin stands at {basin:.3} m"
    );
    let pit = held_below(&water, -1.5);
    // The basin holds 0.128 m³ below its rim; the rest runs on, less the
    // film left standing over the rim.
    assert!((0.05..0.072).contains(&pit), "the pit got {pit:.3} m³");
    assert!((water.ledger().total() - 0.2).abs() < 1.0e-9);
}

/// A basin 20 cm deep behind a rim on a cliff top, over a pit 5 m below.
fn cliff() -> Ground {
    Ground {
        rooms: vec![
            room([0.0, -0.2, 0.0], [0.8, 3.0, 0.4]),
            room([0.8, 0.0, 0.0], [1.0, 3.0, 0.4]),
            room([1.0, -5.0, 0.0], [3.0, 3.0, 0.4]),
        ],
        lake: None,
        river: None,
        rock: true,
    }
}

#[test]
fn a_pool_spilling_over_a_cliff_holds_water_in_the_air_on_the_way_down() {
    let ground = cliff();
    let mut water = WaterWorld::new();
    water.deposit(&ground, DVec3::new(0.4, 1.0, 0.2), 0.2);
    let mut most_in_air = 0.0_f64;
    let mut arc = None;
    for _ in 0..20 {
        let step = water.step(&ground, 0.05);
        most_in_air = most_in_air.max(water.ledger().falling_m3);
        if let Some(fall) = step.falls.into_iter().find(|fall| fall.points.len() > 4) {
            arc = Some(fall);
        }
    }
    assert!(
        most_in_air > 0.005,
        "only {most_in_air:.4} m³ was ever in the air"
    );
    let arc = arc.expect("no stream poured over the cliff");
    let (top, bottom) = (arc.points[0], arc.points[arc.points.len() - 1]);
    assert!(
        bottom.y < top.y - 0.5,
        "the stream runs from {top} to {bottom}"
    );
    assert!(
        bottom.x > top.x,
        "the stream falls straight down, from {top} to {bottom}"
    );
    run(&mut water, &ground, 60);
    // The basin still trickles over its rim, but the stream is spent.
    assert!(water.ledger().falling_m3 < 1.0e-4, "the stream still runs");
    let pit = held_below(&water, -4.0);
    assert!(pit > 0.1, "the pit got {pit:.3} m³");
    assert!(
        (water.ledger().total() - 0.2).abs() < 1.0e-9,
        "water was made or lost"
    );
}

mod terrain;

#[test]
fn a_pit_dug_under_a_lakes_edge_fills_the_rest_of_the_pit() {
    // A shallow lake whose bank is dug into a pit deeper than its bed:
    // the part of the pit under the lake is lake, and pours into the rest.
    let ground = Ground {
        rooms: vec![
            room([-20.0, 1.0, -20.0], [20.0, 3.0, 20.0]),
            room([-1.0, -1.0, -2.0], [4.0, 3.0, 2.0]),
        ],
        lake: Some((room([-20.0, 0.4, -20.0], [0.0, 3.0, 20.0]), 0.8)),
        river: None,
        rock: true,
    };
    let mut water = WaterWorld::new();
    water.terrain_changed(&ground, ground.bricks());
    run(&mut water, &ground, 30);
    // Only cells under the lake join it.
    assert!(
        water
            .joined_cells(&ground)
            .iter()
            .all(|(cell, _)| cell.centre().x < 0.0),
        "the lake spread beyond what lies under it"
    );
    let level = level_at(&water, &ground, DVec3::new(3.0, 0.0, 0.0));
    assert!(
        (level - 0.8).abs() < 0.02,
        "the far end of the pit stands at {level:.3}"
    );
    let ledger = water.ledger();
    assert!(
        (drawn(&water) - held(&water) - ledger.air_m3 - ledger.sea_m3).abs() < 1.0e-6,
        "water was made or lost"
    );
}
