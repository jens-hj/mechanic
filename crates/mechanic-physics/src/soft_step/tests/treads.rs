//! Treads: lugs hold a body on soil that a smooth face slides over, and press
//! the ground harder under the same load.

use bevy_math::DVec3;
use mechanic_core::{
    BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, FaceKind, GRAVITY,
    LayerFace, STANDARD_GRAVITY_M_S2, TreadPattern, TreadSpec,
};
use mechanic_world::TerrainMaterial;

use super::{World, floor};
use crate::MachineState;

/// A 1 m steel cube resting on a floor of `material`, its bottom cut with
/// `tread`.
fn cube_on(material: TerrainMaterial, tread: Option<TreadSpec>) -> World {
    let mut graph = ConstructionGraph::new();
    let spec = CuboidSpec::new([4, 4, 4], BuildPose::default()).unwrap();
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
        unreachable!("spawning reports the part")
    };
    graph
        .apply(BuildCommand::SetTread {
            part,
            surface: LayerFace::Face(FaceKind::NegativeY),
            tread,
        })
        .unwrap();
    let creation = graph.compile().unwrap();
    let mut state = MachineState::at_rest(&creation);
    state.poses[0].position.y = 0.501;
    World::over(creation, state, floor(material))
}

fn deep(pattern: TreadPattern) -> TreadSpec {
    TreadSpec::new(pattern, mechanic_core::MAX_TREAD_DEPTH_MM).unwrap()
}

#[test]
fn lugs_hold_on_overloaded_soil_that_a_smooth_face_slides_over() {
    // A steel cube's own weight is more than loose soil carries, and the
    // failed soil under its whole face holds a smooth face back with about
    // half of it.
    let pull = DVec3::new(0.6, -1.0, 0.0) * STANDARD_GRAVITY_M_S2;
    let travelled = |tread| {
        let mut world = cube_on(TerrainMaterial::Soil, tread);
        let mut last = world.tick(pull);
        for _ in 0..60 {
            last = world.tick(pull);
        }
        last.poses[0].position.x
    };
    let smooth = travelled(None);
    let lugged = travelled(Some(deep(TreadPattern::Block)));
    assert!(smooth > 0.5, "the smooth cube slid {smooth:.3} m");
    assert!(
        lugged.abs() < 0.05,
        "lugs slid {lugged:.3} m where smooth slid {smooth:.3} m"
    );
}

#[test]
fn lugs_press_the_ground_harder_under_the_same_load() {
    let pressure = |tread| {
        let mut world = cube_on(TerrainMaterial::Soil, tread);
        for _ in 0..30 {
            world.tick(GRAVITY);
        }
        world
            .machine
            .terrain_loads()
            .iter()
            .map(|load| f64::from(load.soil_patch(DVec3::ZERO).pressure_pa))
            .sum::<f64>()
    };
    let smooth = pressure(None);
    let studs = deep(TreadPattern::Studded);
    let studded = pressure(Some(studs));
    let expected = f64::from(studs.response().pressure_factor());
    assert!(smooth > 0.0);
    assert!(
        (studded / smooth / expected - 1.0).abs() < 0.1,
        "studs pressed {studded:.0} Pa where smooth pressed {smooth:.0} Pa"
    );
}
