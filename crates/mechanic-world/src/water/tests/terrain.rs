//! Stored water on generated terrain: a trench dug from a real lake fills.

use bevy_math::{DVec2, DVec3};

use crate::water::{TerrainWater, WaterWorld};
use crate::{
    TerrainField, TerrainOctree, WaterBody, WaterTile, WorldPosition, WorldSeed, joined_water_sheet,
};

/// A lake column near spawn and a dry bank beside it: the lake point, the
/// bank point, and the lake's level.
fn lake_shore(field: &TerrainField) -> (DVec3, DVec3, f64) {
    let spawn = field.safe_spawn().0;
    for ring in 1..80 {
        let radius = f64::from(ring) * 4.0;
        for step in 0..ring * 8 {
            let angle = f64::from(step) / f64::from(ring * 8) * std::f64::consts::TAU;
            let (x, z) = (
                spawn.x + radius * angle.cos(),
                spawn.z + radius * angle.sin(),
            );
            let Some(surface) = field.water_surface(x, z) else {
                continue;
            };
            let deep = field
                .topmost_surface(x, z)
                .is_some_and(|ground| ground < surface.level - 0.6);
            if !matches!(surface.body, WaterBody::Lake(_)) || !deep {
                continue;
            }
            for direction in 0..8 {
                let angle = f64::from(direction) * std::f64::consts::FRAC_PI_4;
                for reach in 2..12 {
                    let (bx, bz) = (
                        x + f64::from(reach) * angle.cos(),
                        z + f64::from(reach) * angle.sin(),
                    );
                    if field
                        .topmost_surface(bx, bz)
                        .is_some_and(|ground| ground > surface.level + 0.3)
                    {
                        let level = surface.level;
                        return (DVec3::new(x, level, z), DVec3::new(bx, level, bz), level);
                    }
                }
            }
        }
    }
    panic!("no lake shore near spawn");
}

#[test]
fn a_trench_dug_from_a_generated_lake_fills_to_its_level() {
    let field = TerrainField::new(WorldSeed(42));
    let (lake, bank, level) = lake_shore(&field);
    let mut terrain = TerrainOctree::default();
    let mut bricks = Vec::new();
    // A trench from well inside the bank out into the lake, its floor half
    // a metre under the water.
    let beyond = bank + (bank - lake).normalize() * 2.0;
    let steps = 40;
    for step in 0..=steps {
        let point = beyond.lerp(lake, f64::from(step) / f64::from(steps));
        let centre = WorldPosition(DVec3::new(point.x, level - 0.2, point.z));
        let outcome = terrain.excavate_sphere(&field, centre, 0.5).unwrap();
        bricks.extend_from_slice(outcome.changed_brick_coordinates());
    }
    let ground = TerrainWater {
        field: &field,
        edits: &terrain,
    };
    let mut water = WaterWorld::new();
    water.terrain_changed(&ground, bricks);
    for step in 0..600 {
        let before = water.ledger();
        water.step(&ground, 0.05);
        let after = water.ledger();
        assert!(
            (after.total() - before.total()).abs() < 1.0e-9,
            "step {step} made or lost water: {before:?} -> {after:?}"
        );
    }
    let dug = DVec3::new(beyond.x, level - 0.5, beyond.z);
    let surface = water.surface(&ground, dug).expect("the trench holds water");
    assert!(
        matches!(surface.body, WaterBody::Lake(_)),
        "the dug trench did not join the lake"
    );
    assert!(
        (surface.level - level).abs() < 0.05,
        "the trench stands at {:.3} m under a lake at {level:.3} m",
        surface.level
    );
    let held = water.stored_m3() + water.joined_m3();
    assert!(held > 0.5, "only {held:.2} m³ ran in");
    // The lake's own sheet runs on over the trench, as deep as it was dug.
    let joined = water
        .joined_cells(&ground)
        .into_iter()
        .map(|(cell, _)| cell)
        .collect::<std::collections::HashSet<_>>();
    let tile = WaterTile {
        minimum: [beyond.x - 4.0, beyond.z - 4.0],
        edge: 8.0,
        cells: 32,
    };
    let shifts = water.cycle.shifts(&ground);
    let sheet = joined_water_sheet(
        &field,
        &terrain,
        tile,
        &shifts,
        &joined,
        &std::collections::HashSet::new(),
    )
    .expect("the trench shows water");
    let deepest = sheet
        .vertices
        .iter()
        .zip(&sheet.depths)
        .filter(|(vertex, _)| {
            let point = DVec2::new(
                f64::from(vertex[0]) + tile.minimum[0],
                f64::from(vertex[2]) + tile.minimum[1],
            );
            point.distance(DVec2::new(beyond.x, beyond.z)) < 1.0
        })
        .map(|(_, &depth)| depth)
        .fold(0.0_f32, f32::max);
    assert!(deepest > 0.4, "the trench shows {deepest:.2} m of water");
    // Water fills the dug trench and does not creep off along the shore.
    let flat = |point: DVec3| DVec2::new(point.x, point.z);
    let (from, to) = (flat(beyond), flat(lake));
    for (cell, _) in water.joined_cells(&ground) {
        let point = flat(cell.centre());
        let along = ((point - from).dot(to - from) / from.distance_squared(to)).clamp(0.0, 1.0);
        let off = point.distance(from.lerp(to, along));
        assert!(
            off < 2.5,
            "{cell:?} joined the lake {off:.1} m off the trench"
        );
    }
}

/// Whether a point lies under any triangle of a sheet, seen from above.
fn covers(sheet: &crate::WaterSheet, point: DVec2) -> bool {
    let at = |index: u32| {
        let vertex = sheet.vertices[index as usize];
        DVec2::new(
            f64::from(vertex[0]) + sheet.origin.0.x,
            f64::from(vertex[2]) + sheet.origin.0.z,
        )
    };
    sheet.indices.chunks(3).any(|triangle| {
        let [a, b, c] = [at(triangle[0]), at(triangle[1]), at(triangle[2])];
        let side = |p: DVec2, q: DVec2| (q - p).perp_dot(point - p);
        let (first, second, third) = (side(a, b), side(b, c), side(c, a));
        (first >= 0.0 && second >= 0.0 && third >= 0.0)
            || (first <= 0.0 && second <= 0.0 && third <= 0.0)
    })
}

#[test]
fn a_lakes_sheet_meets_the_running_water_in_a_channel_dug_from_it() {
    let field = TerrainField::new(WorldSeed(42));
    let (lake, bank, level) = lake_shore(&field);
    let mut terrain = TerrainOctree::default();
    let mut bricks = Vec::new();
    // A wide channel from the bank out into the lake, deeper than its edge.
    let beyond = bank + (bank - lake).normalize() * 3.0;
    let steps = 60;
    for step in 0..=steps {
        let point = beyond.lerp(lake, f64::from(step) / f64::from(steps));
        for lift in [0.2, 1.0] {
            let centre = WorldPosition(DVec3::new(point.x, level - lift, point.z));
            let outcome = terrain.excavate_sphere(&field, centre, 1.2).unwrap();
            bricks.extend_from_slice(outcome.changed_brick_coordinates());
        }
    }
    let ground = TerrainWater {
        field: &field,
        edits: &terrain,
    };
    let mut water = WaterWorld::new();
    water.terrain_changed(&ground, bricks);
    for _ in 0..1_200 {
        water.step(&ground, 0.05);
    }
    let joined = water
        .joined_cells(&ground)
        .into_iter()
        .map(|(cell, _)| cell)
        .collect::<std::collections::HashSet<_>>();
    let meeting = water
        .meeting_columns(&ground)
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
    assert!(!meeting.is_empty(), "no running water meets the lake");
    // The lake's finest tiles: 64 m with a vertex every metre.
    let middle = (beyond + lake) * 0.5;
    let tile = WaterTile {
        minimum: [middle.x - 32.0, middle.z - 32.0],
        edge: 64.0,
        cells: 64,
    };
    let shifts = water.cycle.shifts(&ground);
    let sheet = joined_water_sheet(&field, &terrain, tile, &shifts, &joined, &meeting)
        .expect("the lake shows");
    // Every lake column beside the running water lies under the lake's
    // sheet: none is left bare between the two surfaces.
    let centre = |x: i32, z: i32| {
        let centre = crate::water::WaterCell::new(x, 0, z).centre();
        DVec2::new(centre.x, centre.z)
    };
    let mut beside = 0;
    for &(x, z) in &meeting {
        for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, nz) = (x + dx, z + dz);
            let point = centre(nx, nz);
            let probe = DVec3::new(point.x, level - 0.02, point.y);
            let lake_water = field.is_water(probe)
                || joined.contains(&crate::water::WaterCell::containing(probe));
            if meeting.contains(&(nx, nz)) || !lake_water {
                continue;
            }
            beside += 1;
            assert!(covers(&sheet, point), "the lake leaves ({nx}, {nz}) bare");
        }
    }
    assert!(beside > 0, "the running water touches no lake water");
}
