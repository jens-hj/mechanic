//! Water: bodies float at their density, sink slower than they fall, and ride
//! the current.

use bevy_math::{DVec2, DVec3};
use mechanic_core::{
    BuildCommand, BuildPose, ConstructionGraph, ConstructionMaterial, CuboidSpec, GRAVITY,
};
use mechanic_world::{WaterBody, WaterSurface};

use super::{World, pose};
use crate::{MachineState, WaterSource};

/// Still or flowing water at one level everywhere.
struct Pond {
    level: f64,
    flow: DVec2,
}

impl WaterSource for Pond {
    fn surface(&self, _point: DVec3) -> Option<WaterSurface> {
        Some(WaterSurface {
            level: self.level,
            body: WaterBody::Lake(0),
            flow: self.flow,
        })
    }
}

/// A 1 m cube of `material` with its centre at `centre`, over the rock floor.
fn cube(material: ConstructionMaterial, centre: DVec3, water: Option<Pond>) -> World {
    let mut graph = ConstructionGraph::new();
    let mut spec = CuboidSpec::new([4, 4, 4], BuildPose::default()).unwrap();
    spec.material = material;
    graph.apply(BuildCommand::Spawn(spec)).unwrap();
    let creation = graph.compile().unwrap();
    let rows = creation.dynamics.elimination_parent.len();
    let state = MachineState {
        poses: vec![pose(centre)],
        coordinates: Vec::new(),
        velocities: vec![0.0; rows],
    };
    let mut world = World::new(creation, state);
    world.water = water.map(|water| Box::new(water) as Box<dyn WaterSource>);
    world
}

fn centre(world: &World) -> DVec3 {
    world.machine.snapshot().state.poses[0].position
}

#[test]
fn a_wooden_block_floats_at_its_density_ratio() {
    let water = Pond {
        level: 3.0,
        flow: DVec2::ZERO,
    };
    let mut world = cube(
        ConstructionMaterial::Wood,
        DVec3::new(0.0, 3.4, 0.0),
        Some(water),
    );
    for _ in 0..600 {
        world.tick(GRAVITY);
    }
    // Wood is 0.7 as dense as water, so 0.7 m of the 1 m cube lies under.
    let height = centre(&world).y;
    assert!(
        (height - 2.8).abs() < 0.05,
        "the block settled with its centre at {height:.3} m"
    );
    let before = centre(&world).y;
    world.tick(GRAVITY);
    assert!((centre(&world).y - before).abs() < 0.002, "still bobbing");
}

#[test]
fn a_steel_block_sinks_slower_than_it_falls_and_levels_off() {
    let start = DVec3::new(0.0, 100.0, 0.0);
    let mut air = cube(ConstructionMaterial::Steel, start, None);
    let mut water = cube(
        ConstructionMaterial::Steel,
        start,
        Some(Pond {
            level: 200.0,
            flow: DVec2::ZERO,
        }),
    );
    for _ in 0..60 {
        air.tick(GRAVITY);
        water.tick(GRAVITY);
    }
    let fell = start.y - centre(&air).y;
    let sank = start.y - centre(&water).y;
    assert!(
        sank < 0.8 * fell,
        "steel sank {sank:.2} m in water and fell {fell:.2} m in air"
    );
    // Over the next seconds drag balances its weight in water: a 1 m steel
    // cube sinks at about 11 m/s, where in air it would pass 40 m/s.
    let speed = |world: &mut World| {
        let before = centre(world).y;
        world.tick(GRAVITY);
        (before - centre(world).y) * 60.0
    };
    for _ in 0..180 {
        water.tick(GRAVITY);
    }
    let settled = speed(&mut water);
    assert!(
        (8.0..14.0).contains(&settled),
        "steel sinks at {settled:.1} m/s after four seconds"
    );
}

#[test]
fn a_current_carries_a_floating_block_downstream() {
    let mut world = cube(
        ConstructionMaterial::Wood,
        DVec3::new(0.0, 2.8, 0.0),
        Some(Pond {
            level: 3.0,
            flow: DVec2::new(1.0, 0.0),
        }),
    );
    for _ in 0..300 {
        world.tick(GRAVITY);
    }
    let drift = centre(&world);
    assert!(drift.x > 2.5, "the block drifted only {:.2} m", drift.x);
    assert!(drift.z.abs() < 0.05, "the block drifted sideways");
}
