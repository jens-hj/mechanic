//! Stored water on generated terrain: a trench dug from a real lake fills.

use bevy_math::{DVec2, DVec3};

use crate::water::{TerrainWater, WaterWorld};
use crate::{TerrainField, TerrainOctree, WaterBody, WorldPosition, WorldSeed};

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
