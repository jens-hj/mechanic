//! Erosion: running water wears soft ground, carries it and lays it down,
//! and no material is made or lost on the way.

use bevy_math::DVec3;

use super::{Ground, lake_bank_and_field, natural, pit, room};
use crate::water::{ErosionConfig, SedimentLoad, WaterCell, WaterWorld};
use crate::{SedimentApplied, TerrainMaterial};

/// Answers the water's asks as soft ground would: it gives all it is asked
/// for, as soil, and takes all it is given. Returns how many changes it was
/// asked for.
fn answer(water: &mut WaterWorld) -> usize {
    let asks = water.sediment_requests();
    let applied = asks
        .iter()
        .map(|change| {
            let mut done = SedimentApplied::default();
            if change.quanta < 0 {
                done.taken[TerrainMaterial::Soil.code() as usize] = change.quanta.unsigned_abs();
            } else {
                done.laid = change.quanta.unsigned_abs();
            }
            done
        })
        .collect::<Vec<_>>();
    water.sediment_applied(&applied);
    asks.len()
}

/// Stirs sediment into the running water in a cell's column.
fn muddy(water: &mut WaterWorld, cell: WaterCell, load: SedimentLoad) {
    let slot = water
        .sheets
        .slot(cell.x, cell.z)
        .expect("running water in the cell");
    water.sheets.at_mut(slot).load.add(load);
}

#[test]
fn matter_is_conserved_while_water_erodes_and_lays_sediment() {
    let ground = Ground {
        rock: false,
        ..lake_bank_and_field()
    };
    let mut water = WaterWorld::new();
    // An hour's wear in six seconds.
    water.set_erosion(ErosionConfig { speed: 600.0 });
    water.terrain_changed(&ground, ground.bricks());
    let mut asked = 0;
    for step in 0..1_200 {
        water.step(&ground, 0.05);
        asked += answer(&mut water);
        let books = water.sediment_ledger();
        assert!(
            books.unaccounted().abs() < 1.0e-6 * books.eroded.max(1.0),
            "step {step}: sediment made or lost: {books:?}"
        );
        assert!(water.ledger().total().abs() < 1.0e-9, "water made or lost");
    }
    let books = water.sediment_ledger();
    assert!(asked > 0, "the ground was never asked");
    assert!(books.eroded > 1_000.0, "the breach wore nothing: {books:?}");
    assert!(books.laid > 0.0, "nothing was laid down: {books:?}");
}

#[test]
fn a_still_pool_clears_as_its_sediment_settles_onto_its_bed() {
    let ground = pit(false);
    let mut water = WaterWorld::new();
    water.deposit(&ground, DVec3::new(0.5, 0.5, 0.5), 0.5);
    super::run(&mut water, &ground, 5);
    let stirred = SedimentLoad {
        sand: 2_000.0,
        fines: 2_000.0,
    };
    let id = *water.pools.keys().next().expect("the pit holds a pool");
    water
        .pools
        .get_mut(&id)
        .expect("the pool")
        .load
        .add(stirred);
    // Two minutes in water half a metre deep: sand falls through it in
    // half a minute, fines over a quarter of an hour.
    for _ in 0..2_400 {
        water.step(&ground, 0.05);
        answer(&mut water);
    }
    let left = water.pools[&id].load;
    assert!(
        left.sand < 0.02 * stirred.sand,
        "{:.1} quanta of sand still float",
        left.sand
    );
    assert!(
        (0.3 * stirred.fines..0.98 * stirred.fines).contains(&left.fines),
        "{:.1} of {:.1} quanta of fines still float",
        left.fines,
        stirred.fines
    );
    let books = water.sediment_ledger();
    let held = books.suspended + books.settled + books.laid;
    assert!(
        (held - stirred.total()).abs() < 1.0e-6,
        "sediment made or lost: {books:?}"
    );
    assert!(books.eroded <= 0.0, "a rock pit wore away");
}

/// A natural lake under open sky, its level at -0.5 m, and a flat channel
/// 20 cm wide running 6 m east from its shore with its floor at -0.3 m.
fn stream_into_a_lake() -> Ground {
    Ground {
        rooms: vec![
            natural([-10.0, 0.0, -10.0], [10.0, 20.0, 10.0]),
            room([0.0, -0.3, 0.0], [6.0, 20.0, 0.2]),
        ],
        lake: Some((natural([-10.0, -2.0, -10.0], [0.0, 20.0, 10.0]), -0.5)),
        river: None,
        rock: true,
    }
}

#[test]
fn sediment_running_into_a_lake_settles_at_its_mouth() {
    let ground = stream_into_a_lake();
    let mut water = WaterWorld::new();
    water.terrain_changed(&ground, ground.bricks());
    let source = WaterCell::containing(DVec3::new(5.9, -0.2, 0.1));
    let mut stirred = 0.0;
    for step in 0..600 {
        water.deposit(&ground, DVec3::new(5.9, 0.0, 0.1), 0.002);
        if step < 300 && water.sheets.slot(source.x, source.z).is_some() {
            let load = SedimentLoad {
                sand: 5.0,
                fines: 5.0,
            };
            muddy(&mut water, source, load);
            stirred += load.total();
        }
        water.step(&ground, 0.05);
        answer(&mut water);
    }
    let books = water.sediment_ledger();
    let held = books.suspended + books.settled + books.laid;
    assert!(
        (held - stirred).abs() < 1.0e-6 * stirred,
        "sediment made or lost: {books:?} of {stirred}"
    );
    // What reached the lake's edge stayed there: nothing settles out in
    // the lake, and most of it lies within a metre of the mouth.
    let near_mouth = water
        .sediment_doc()
        .beds
        .iter()
        .filter(|bed| (-2..5).contains(&bed.column.0))
        .map(|bed| bed.settled.total())
        .sum::<f64>();
    assert!(
        water
            .sediment_doc()
            .beds
            .iter()
            .all(|bed| bed.column.0 >= -2),
        "sediment settled out in the lake"
    );
    assert!(
        near_mouth + books.laid > 0.3 * stirred,
        "only {near_mouth:.0} and {:.0} laid of {stirred:.0} by the mouth",
        books.laid
    );
}

#[test]
fn sediment_the_ground_had_no_room_for_waits_to_be_laid_again() {
    let ground = pit(false);
    let mut water = WaterWorld::new();
    water.deposit(&ground, DVec3::new(0.5, 0.5, 0.5), 0.5);
    super::run(&mut water, &ground, 5);
    let id = *water.pools.keys().next().expect("the pit holds a pool");
    water
        .pools
        .get_mut(&id)
        .expect("the pool")
        .load
        .add(SedimentLoad {
            sand: 4_000.0,
            fines: 0.0,
        });
    super::run(&mut water, &ground, 20);
    let asks = water.sediment_requests();
    assert!(!asks.is_empty(), "settled sand asked for no change");
    let settled = water.sediment_ledger().settled;
    // The ground refuses outright, then lays nothing.
    water.sediment_refused();
    assert!((water.sediment_ledger().settled - settled).abs() < 1e-9);
    super::run(&mut water, &ground, 3);
    let asks = water.sediment_requests();
    assert!(!asks.is_empty());
    water.sediment_applied(&vec![SedimentApplied::default(); asks.len()]);
    let books = water.sediment_ledger();
    assert!((books.suspended + books.settled - 4_000.0).abs() < 1.0e-6);
    assert!(books.laid <= 0.0);
}

#[test]
fn carried_and_settled_sediment_survive_a_save() {
    let ground = pit(false);
    let mut water = WaterWorld::new();
    water.deposit(&ground, DVec3::new(0.5, 0.5, 0.5), 0.5);
    super::run(&mut water, &ground, 5);
    let id = *water.pools.keys().next().expect("the pit holds a pool");
    water
        .pools
        .get_mut(&id)
        .expect("the pool")
        .load
        .add(SedimentLoad {
            sand: 300.0,
            fines: 700.0,
        });
    super::run(&mut water, &ground, 10);
    let before = water.sediment_ledger();
    assert!(before.suspended > 0.0 && before.settled > 0.0);
    let loaded = WaterWorld::from_doc(&ground, &water.to_doc());
    let after = loaded.sediment_ledger();
    assert!((after.suspended - before.suspended).abs() < 1.0e-9);
    assert!((after.settled - before.settled).abs() < 1.0e-9);
}

/// A soil slope near spawn, dry and with no water near: the high end of a
/// fall line 6 m long dropping 0.6 to 1.5 m, and its low end.
fn soil_slope(field: &crate::TerrainField) -> (DVec3, DVec3) {
    let spawn = field.safe_spawn().0;
    let ground = |x: f64, z: f64| field.topmost_surface(x, z);
    let soil = |point: DVec3| {
        let terrain = crate::TerrainOctree::default();
        let ground = crate::TerrainWater {
            field,
            edits: &terrain,
        };
        crate::water::WaterGround::material(&ground, point) == Some(TerrainMaterial::Soil)
    };
    for ring in 1..60 {
        let radius = f64::from(ring) * 3.0;
        for step in 0..ring * 8 {
            let angle = f64::from(step) / f64::from(ring * 8) * std::f64::consts::TAU;
            let (x, z) = (
                spawn.x + radius * angle.cos(),
                spawn.z + radius * angle.sin(),
            );
            let Some(high) = ground(x, z) else {
                continue;
            };
            for direction in 0..16 {
                let angle = f64::from(direction) * std::f64::consts::TAU / 16.0;
                let (lx, lz) = (x + 6.0 * angle.cos(), z + 6.0 * angle.sin());
                let Some(low) = ground(lx, lz) else {
                    continue;
                };
                let fall = high - low;
                // An even fall: every metre lower than the last.
                let even = (1..6).all(|metre| {
                    let t = f64::from(metre) / 6.0;
                    ground(x + t * (lx - x), z + t * (lz - z))
                        .is_some_and(|y| y < high - 0.05 * f64::from(metre) && y > low)
                });
                let dry = (0..=6).all(|metre| {
                    let t = f64::from(metre) / 6.0;
                    field
                        .water_surface(x + t * (lx - x), z + t * (lz - z))
                        .is_none()
                });
                if (0.6..1.5).contains(&fall)
                    && even
                    && dry
                    && soil(DVec3::new(x, high - 0.35, z))
                    && soil(DVec3::new(lx, low - 0.35, lz))
                {
                    return (DVec3::new(x, high, z), DVec3::new(lx, low, lz));
                }
            }
        }
    }
    panic!("no soil slope near spawn");
}

/// Digs a trench 40 cm deep down a fall line into a pit at its foot, and
/// runs a stream into its head for `seconds` with erosion `speed` times its
/// rate, the ground changing as the water asks. Returns the terrain, the
/// water, and the trench's ends.
fn run_a_stream(
    field: &crate::TerrainField,
    speed: f64,
    seconds: u32,
    dam: bool,
) -> (crate::TerrainOctree, WaterWorld, DVec3, DVec3) {
    use crate::{TerrainOctree, TerrainWater, WorldPosition};
    let (high, low) = soil_slope(field);
    let mut terrain = TerrainOctree::default();
    let mut bricks = Vec::new();
    let steps = 30;
    for step in 0..=steps {
        let point = high.lerp(low, f64::from(step) / f64::from(steps));
        let ground = field.topmost_surface(point.x, point.z).unwrap_or(point.y);
        let centre = WorldPosition(DVec3::new(point.x, ground - 0.1, point.z));
        let outcome = terrain.excavate_sphere(field, centre, 0.3).unwrap();
        bricks.extend_from_slice(outcome.changed_brick_coordinates());
    }
    let pit = WorldPosition(low - DVec3::Y * 0.5);
    let outcome = terrain.excavate_sphere(field, pit, 0.8).unwrap();
    bricks.extend_from_slice(outcome.changed_brick_coordinates());
    if dam {
        // A dirt dam filling the trench's middle to 15 cm under its banks,
        // so water spills over the dam and not round it.
        let middle = high.lerp(low, 0.5);
        let ground = field
            .topmost_surface(middle.x, middle.z)
            .unwrap_or(middle.y);
        let across = (low - high).normalize().cross(DVec3::Y);
        for step in -3..=3 {
            let point = middle + across * (0.1 * f64::from(step));
            let centre = WorldPosition(DVec3::new(point.x, ground - 0.35, point.z));
            let outcome = terrain
                .add_sphere(field, centre, 0.2, TerrainMaterial::Soil)
                .unwrap();
            bricks.extend_from_slice(outcome.changed_brick_coordinates());
        }
    }
    let mut water = WaterWorld::new();
    water.set_erosion(ErosionConfig { speed });
    water.terrain_changed(
        &TerrainWater {
            field,
            edits: &terrain,
        },
        bricks,
    );
    // Clean water fills the pond behind a dam; without one, the stream
    // runs from the trench's head.
    let head = if dam {
        high.lerp(low, 0.5) - (low - high).normalize() * 0.6
    } else {
        high + (low - high).normalize() * 0.3
    };
    for _ in 0..seconds * 20 {
        let ground = TerrainWater {
            field,
            edits: &terrain,
        };
        water.deposit(&ground, DVec3::new(head.x, high.y + 0.2, head.z), 0.001);
        water.step(&ground, 0.05);
        let asks = water.sediment_requests();
        if !asks.is_empty() {
            let (outcome, applied) = terrain.exchange_sediment(field, &asks);
            water.sediment_applied(&applied);
            let ground = TerrainWater {
                field,
                edits: &terrain,
            };
            water.ground_cells_changed(&ground, &outcome.sediment_cells);
        }
    }
    (terrain, water, high, low)
}

/// Height of the ground's drawn top over a point, searching down from
/// `from`: its top cell's middle plus how far its density reaches.
fn bed_at(
    terrain: &crate::TerrainOctree,
    field: &crate::TerrainField,
    point: DVec3,
    from: f64,
) -> f64 {
    #[expect(clippy::cast_possible_truncation, reason = "heights in the world")]
    let cell = |value: f64| (value / crate::TERRAIN_CELL_METERS).floor() as i32;
    let (x, z) = (cell(point.x), cell(point.z));
    for y in (cell(from - 2.0)..=cell(from)).rev() {
        let sample = terrain.sample_cell(field, crate::WorldCell::new(x, y, z));
        if sample.is_solid() {
            return (f64::from(y) + 0.5) * crate::TERRAIN_CELL_METERS + f64::from(sample.density);
        }
    }
    f64::NEG_INFINITY
}

/// The mean height of a trench's bed along its upper half.
fn upper_bed(
    terrain: &crate::TerrainOctree,
    field: &crate::TerrainField,
    high: DVec3,
    low: DVec3,
) -> f64 {
    let points = (2..=10)
        .map(|step| high.lerp(low, f64::from(step) / 20.0))
        .collect::<Vec<_>>();
    #[expect(clippy::cast_precision_loss, reason = "a few points")]
    let count = points.len() as f64;
    points
        .into_iter()
        .map(|point| bed_at(terrain, field, point, point.y + 0.5))
        .sum::<f64>()
        / count
}

#[test]
fn a_stream_down_a_dug_trench_cuts_its_bed() {
    let field = crate::TerrainField::new(crate::WorldSeed(42));
    let (dug, _, high, low) = run_a_stream(&field, 1.0, 0, false);
    let before = upper_bed(&dug, &field, high, low);
    // Two hours of wear in two minutes.
    let (terrain, water, _, _) = run_a_stream(&field, 60.0, 120, false);
    let after = upper_bed(&terrain, &field, high, low);
    let books = water.sediment_ledger();
    eprintln!("{books:?} bed {before:.3} -> {after:.3}");
    assert!(
        books.unaccounted().abs() < 1.0e-6 * books.eroded.max(1.0),
        "sediment made or lost: {books:?}"
    );
    assert!(
        after < before - 0.02,
        "the trench's bed went from {before:.3} m to {after:.3} m: {books:?}"
    );
}

#[test]
fn water_over_a_dirt_dam_cuts_a_notch_through_it() {
    let field = crate::TerrainField::new(crate::WorldSeed(42));
    let (built, _, high, low) = run_a_stream(&field, 1.0, 0, true);
    let middle = high.lerp(low, 0.5);
    // The crest: the lowest point across the dam, where the water spills.
    let crest = |terrain: &crate::TerrainOctree| {
        let across = (low - high).normalize().cross(DVec3::Y);
        (-4..=4)
            .map(|step| middle + across * (0.05 * f64::from(step)))
            .map(|point| bed_at(terrain, &field, point, point.y + 1.0))
            .fold(f64::INFINITY, f64::min)
    };
    let before = crest(&built);
    // Two hours of wear in two minutes.
    let (terrain, water, _, _) = run_a_stream(&field, 60.0, 120, true);
    let after = crest(&terrain);
    let books = water.sediment_ledger();
    eprintln!("{books:?} crest {before:.3} -> {after:.3}");
    assert!(books.unaccounted().abs() < 1.0e-6 * books.eroded.max(1.0));
    assert!(
        after < before - 0.1,
        "the dam's crest went from {before:.3} m to {after:.3} m"
    );
}
