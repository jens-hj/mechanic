mod bounds;

use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
};

use bevy_math::{DVec2, DVec3, Vec3};
use mechanic_core::{
    BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, GridRotation,
};

use super::{
    ActiveTerrainScene, FoundationSample, FoundationSpatialIndex, FoundationSupport,
    KinematicCapsule, KinematicInput, TerrainDensity, TerrainSpatialIndex, raycast_density,
};
use crate::{
    BrickCoord, KinematicCollisionScene, TerrainMaterial, TerrainMeshRequest, TerrainNodeId,
    TerrainOctree, TerrainRayHit, TerrainTransitionMask, WorldPosition, WorldSeed, mesh_chunk,
};

struct Plane;

impl TerrainDensity for Plane {
    fn density(&self, position: WorldPosition) -> f32 {
        (-position.0.y) as f32
    }

    fn material(&self, _position: WorldPosition) -> TerrainMaterial {
        TerrainMaterial::Soil
    }
}

struct Slope;

impl TerrainDensity for Slope {
    fn density(&self, position: WorldPosition) -> f32 {
        (position.0.x * 0.5 - position.0.y) as f32
    }

    fn material(&self, _position: WorldPosition) -> TerrainMaterial {
        TerrainMaterial::Soil
    }
}

fn tick_terrain(
    capsule: &mut KinematicCapsule,
    terrain: &impl TerrainDensity,
    input: KinematicInput,
) {
    let mut scene = KinematicCollisionScene {
        terrain,
        construction: None,
        floating_origin: DVec3::ZERO,
    };
    capsule.tick(&mut scene, input, 1.0 / 60.0);
}

struct CountingPlane(Cell<usize>);

struct Empty;

impl TerrainDensity for Empty {
    fn density(&self, _position: WorldPosition) -> f32 {
        -1.0
    }

    fn material(&self, _position: WorldPosition) -> TerrainMaterial {
        TerrainMaterial::Rock
    }
}

impl TerrainDensity for CountingPlane {
    fn density(&self, position: WorldPosition) -> f32 {
        self.0.set(self.0.get() + 1);
        (-position.0.y) as f32
    }

    fn material(&self, _position: WorldPosition) -> TerrainMaterial {
        TerrainMaterial::Soil
    }
}

#[test]
fn changed_bricks_refresh_only_intersecting_foundation_samples() {
    let near = WorldPosition(DVec3::new(0.1, 0.0, 0.1));
    let far = WorldPosition(DVec3::new(4.0, 0.0, 0.1));
    let mut support = FoundationSupport {
        samples: vec![
            FoundationSample {
                position: near,
                valid: true,
            },
            FoundationSample {
                position: far,
                valid: true,
            },
        ],
    };
    let changed = BTreeSet::from([near.cell().unwrap().brick()]);
    let terrain = CountingPlane(Cell::new(0));

    let refresh = support.refresh_changed(&terrain, &changed);

    assert_eq!(refresh.sampled, 1);
    assert!(terrain.0.get() > 0);
    assert!(support.samples[1].valid);
}

#[test]
fn brick_index_returns_only_overlapping_foundations() {
    let mut graph = ConstructionGraph::default();
    let BuildOutcome::Spawned(near_part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1, 1, 1], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        panic!("spawn reports a part");
    };
    let BuildOutcome::Spawned(far_part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_position_ticks(
                    bevy_math::IVec3::new(1600, 0, 0),
                    GridRotation::default(),
                ),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        panic!("spawn reports a part");
    };
    let near_position = WorldPosition(DVec3::new(0.1, 0.0, 0.1));
    let far_position = WorldPosition(DVec3::new(4.0, 0.0, 0.1));
    let near_support = FoundationSupport {
        samples: vec![FoundationSample {
            position: near_position,
            valid: true,
        }],
    };
    let far_support = FoundationSupport {
        samples: vec![FoundationSample {
            position: far_position,
            valid: true,
        }],
    };
    let mut index = FoundationSpatialIndex::default();
    index.insert(near_part, &near_support);
    let mut staged = FoundationSpatialIndex::default();
    staged.insert(far_part, &far_support);
    index.append(staged);

    assert_eq!(
        index.candidates(&BTreeSet::from([near_position.cell().unwrap().brick()])),
        BTreeSet::from([near_part])
    );
    index.remove(near_part);
    assert!(
        index
            .candidates(&BTreeSet::from([near_position.cell().unwrap().brick()]))
            .is_empty()
    );
}

#[test]
fn capsule_lands_and_jumps_to_requested_height() {
    let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::new(0.0, 1.0, 0.0)));
    for _ in 0..120 {
        tick_terrain(&mut capsule, &Plane, KinematicInput::default());
    }
    assert!(capsule.grounded);
    assert!(capsule.position.0.y.abs() < 0.02);
    tick_terrain(
        &mut capsule,
        &Plane,
        KinematicInput {
            movement: DVec2::ZERO,
            sprint: false,
            jump: true,
            jump_held: false,
        },
    );
    let mut peak = capsule.position.0.y;
    let mut airborne_ticks = 1;
    for _ in 0..120 {
        tick_terrain(&mut capsule, &Plane, KinematicInput::default());
        peak = peak.max(capsule.position.0.y);
        airborne_ticks += 1;
        if capsule.grounded {
            break;
        }
    }
    assert!(
        (peak - capsule.config.jump_height).abs() < 0.08,
        "peak was {peak}"
    );
    assert!(
        (28..=32).contains(&airborne_ticks),
        "jump lasted {airborne_ticks} ticks"
    );
}

fn jump_peak(held: impl Fn(usize) -> bool) -> f64 {
    let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::ZERO));
    capsule.grounded = true;
    let mut peak: f64 = 0.0;
    for tick in 0..120 {
        tick_terrain(
            &mut capsule,
            &Plane,
            KinematicInput {
                jump: tick == 0,
                jump_held: held(tick),
                ..KinematicInput::default()
            },
        );
        peak = peak.max(capsule.position.0.y);
        if tick > 60 {
            assert!(capsule.grounded, "holding jump must not repeat on landing");
        }
    }
    peak
}

#[test]
fn holding_jump_progressively_increases_height_by_up_to_fifty_percent() {
    let tap = jump_peak(|_| false);
    let short_hold = jump_peak(|tick| tick < 6);
    let long_hold = jump_peak(|tick| tick < 12);
    let full_hold = jump_peak(|_| true);
    assert!(tap < short_hold && short_hold < long_hold && long_hold < full_hold);
    assert!(
        (full_hold / tap - 1.5).abs() < 0.04,
        "tap {tap}, held {full_hold}"
    );
}

#[test]
fn pressing_jump_again_in_midair_does_not_restore_extra_lift() {
    let released = jump_peak(|tick| tick < 6);
    let pressed_again = jump_peak(|tick| tick != 6);
    assert!((released - pressed_again).abs() < 1.0e-10);
}

#[test]
fn holding_jump_during_a_fall_does_not_slow_gravity() {
    let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::new(0.0, 10.0, 0.0)));
    tick_terrain(
        &mut capsule,
        &Plane,
        KinematicInput {
            jump: true,
            jump_held: true,
            ..KinematicInput::default()
        },
    );
    assert!((capsule.velocity.y + capsule.config.airborne_gravity / 60.0).abs() < 1.0e-10);
}

#[test]
fn grounded_capsule_stays_still_on_a_walkable_slope() {
    let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::new(0.0, 0.1, 0.0)));
    for _ in 0..60 {
        tick_terrain(&mut capsule, &Slope, KinematicInput::default());
    }
    assert!(capsule.grounded);
    let settled = capsule.position;

    for _ in 0..120 {
        tick_terrain(&mut capsule, &Slope, KinematicInput::default());
    }

    assert!(
        capsule.position.0.abs_diff_eq(settled.0, 1.0e-6),
        "stationary capsule drifted from {settled:?} to {:?}",
        capsule.position
    );
}

#[test]
fn grounded_capsule_stays_still_on_a_walkable_construction_slope() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([16, 1, 16], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        panic!("cuboid spawn returns its part");
    };
    let mut creation = graph.compile_with_static_parts([part]).unwrap();
    let mechanic_core::ColliderShape::Cuboid { local_rotation, .. } =
        &mut creation.colliders[0].shape
    else {
        panic!("unshaped cuboid keeps its fast-path collider");
    };
    *local_rotation = bevy_math::Quat::from_rotation_z(30.0_f32.to_radians());
    let mut index = crate::ConstructionCollisionIndex::new(&creation);
    let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::Y));
    for _ in 0..120 {
        let mut scene = KinematicCollisionScene {
            terrain: &Empty,
            construction: Some(&mut index),
            floating_origin: DVec3::ZERO,
        };
        capsule.tick(&mut scene, KinematicInput::default(), 1.0 / 60.0);
    }
    assert!(capsule.grounded);
    let settled = capsule.position;

    for _ in 0..120 {
        let mut scene = KinematicCollisionScene {
            terrain: &Empty,
            construction: Some(&mut index),
            floating_origin: DVec3::ZERO,
        };
        capsule.tick(&mut scene, KinematicInput::default(), 1.0 / 60.0);
    }

    assert!(
        capsule.position.0.abs_diff_eq(settled.0, 1.0e-5),
        "stationary capsule drifted from {settled:?} to {:?}",
        capsule.position
    );
}

#[test]
fn grounded_capsule_does_not_slide_from_a_flat_construction_edge() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([8, 1, 8], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        panic!("cuboid spawn returns its part");
    };
    let creation = graph.compile_with_static_parts([part]).unwrap();
    let mut index = crate::ConstructionCollisionIndex::new(&creation);
    let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::Y));
    for _ in 0..120 {
        let mut scene = KinematicCollisionScene {
            terrain: &Empty,
            construction: Some(&mut index),
            floating_origin: DVec3::ZERO,
        };
        capsule.tick(&mut scene, KinematicInput::default(), 1.0 / 60.0);
    }
    assert!(capsule.grounded);
    capsule.position.0.x = 1.05;
    let edge = capsule.position;

    for _ in 0..120 {
        let mut scene = KinematicCollisionScene {
            terrain: &Empty,
            construction: Some(&mut index),
            floating_origin: DVec3::ZERO,
        };
        capsule.tick(&mut scene, KinematicInput::default(), 1.0 / 60.0);
    }

    assert!(capsule.grounded);
    assert!(
        (capsule.position.0.x - edge.0.x).abs() < 1.0e-3
            && (capsule.position.0.z - edge.0.z).abs() < 1.0e-3
            && (capsule.position.0.y - edge.0.y).abs() < 0.01,
        "stationary capsule drifted from {edge:?} to {:?}",
        capsule.position
    );
}

#[test]
fn vertical_construction_contact_does_not_slow_a_fall() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [1, 8, 8],
                BuildPose::new(bevy_math::IVec3::new(4, 4, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        panic!("cuboid spawn returns its part");
    };
    let creation = graph.compile_with_static_parts([part]).unwrap();
    let mut index = crate::ConstructionCollisionIndex::new(&creation);
    let mut against_wall = KinematicCapsule::new(WorldPosition(DVec3::new(0.57, 0.1, 0.0)));
    against_wall.velocity = DVec3::new(1.0, -2.0, 0.0);
    let mut free_fall = against_wall;
    let input = KinematicInput {
        movement: DVec2::X,
        sprint: false,
        jump: false,
        jump_held: false,
    };
    let mut wall_scene = KinematicCollisionScene {
        terrain: &Empty,
        construction: Some(&mut index),
        floating_origin: DVec3::ZERO,
    };
    let result = against_wall.tick(&mut wall_scene, input, 1.0 / 60.0);
    let mut empty_scene = KinematicCollisionScene {
        terrain: &Empty,
        construction: None,
        floating_origin: DVec3::ZERO,
    };
    free_fall.tick(&mut empty_scene, input, 1.0 / 60.0);

    assert!(
        result.resolved_contacts > 0,
        "the capsule should reach the wall"
    );
    assert!(
        (against_wall.velocity.y - free_fall.velocity.y).abs() < 1.0e-6,
        "wall contact slowed the fall: {} instead of {}",
        against_wall.velocity.y,
        free_fall.velocity.y,
    );
}

#[test]
fn sprint_uses_configured_faster_speed() {
    let mut walking = KinematicCapsule::new(WorldPosition(DVec3::ZERO));
    walking.grounded = true;
    let mut sprinting = walking;

    for _ in 0..20 {
        tick_terrain(
            &mut walking,
            &Plane,
            KinematicInput {
                movement: DVec2::X,
                sprint: false,
                jump: false,
                jump_held: false,
            },
        );
        tick_terrain(
            &mut sprinting,
            &Plane,
            KinematicInput {
                movement: DVec2::X,
                sprint: true,
                jump: false,
                jump_held: false,
            },
        );
    }

    assert!((walking.velocity.x - walking.config.walk_speed).abs() < f64::EPSILON);
    assert!((sprinting.velocity.x - sprinting.config.sprint_speed).abs() < f64::EPSILON);
    assert!(sprinting.position.0.x > walking.position.0.x);
}

#[test]
fn dynamic_construction_supports_and_receives_opposite_player_impulses() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([8, 1, 8], BuildPose::default()).unwrap(),
        ))
        .unwrap();
    let creation = graph.compile().unwrap();
    let mut index = crate::ConstructionCollisionIndex::new(&creation);
    let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::Y));
    let mut landing_reaction = None;
    for _ in 0..120 {
        let mut scene = KinematicCollisionScene {
            terrain: &Empty,
            construction: Some(&mut index),
            floating_origin: DVec3::ZERO,
        };
        let result = capsule.tick(&mut scene, KinematicInput::default(), 1.0 / 60.0);
        landing_reaction = result
            .reaction_impulses()
            .first()
            .copied()
            .or(landing_reaction);
    }
    assert!(capsule.grounded);
    assert_eq!(
        capsule.support.map(|support| support.compound_index),
        Some(0)
    );
    let reaction = landing_reaction.expect("dynamic floor receives player reaction");
    assert!(reaction.impulse.y < 0.0, "{reaction:?}");

    let mut scene = KinematicCollisionScene {
        terrain: &Empty,
        construction: Some(&mut index),
        floating_origin: DVec3::ZERO,
    };
    let load = capsule.tick(&mut scene, KinematicInput::default(), 1.0 / 60.0);
    assert!(
        load.reaction_impulses()
            .iter()
            .any(|reaction| reaction.impulse.y < 0.0),
        "a stationary player must load a dynamic platform"
    );
}

#[test]
fn moving_support_carries_position_yaw_and_jump_velocity() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([16, 1, 16], BuildPose::default()).unwrap(),
        ))
        .unwrap();
    let creation = graph.compile().unwrap();
    let mut index = crate::ConstructionCollisionIndex::new(&creation);
    let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::new(0.5, 1.0, 0.0)));
    for _ in 0..120 {
        let mut scene = KinematicCollisionScene {
            terrain: &Empty,
            construction: Some(&mut index),
            floating_origin: DVec3::ZERO,
        };
        capsule.tick(&mut scene, KinematicInput::default(), 1.0 / 60.0);
    }
    assert!(capsule.grounded);
    let before = capsule.position;
    let pose = crate::ConstructionBodyState {
        translation: Vec3::X,
        rotation: bevy_math::Quat::from_rotation_y(core::f32::consts::FRAC_PI_2),
        linear_velocity: Vec3::new(2.0, 0.0, 0.0),
        angular_velocity: Vec3::Y,
    };
    assert!(index.refit_dynamic(&[pose]));
    let mut scene = KinematicCollisionScene {
        terrain: &Empty,
        construction: Some(&mut index),
        floating_origin: DVec3::ZERO,
    };
    let carried = capsule.tick(&mut scene, KinematicInput::default(), 1.0 / 60.0);
    assert_ne!(capsule.position, before);
    assert!((carried.support_yaw_delta - core::f32::consts::FRAC_PI_2).abs() < 1.0e-4);
    let inherited_x = capsule
        .support
        .map(|support| {
            pose.velocity_at(pose.transform_point(support.local_anchor))
                .x
        })
        .unwrap();

    let mut scene = KinematicCollisionScene {
        terrain: &Empty,
        construction: Some(&mut index),
        floating_origin: DVec3::ZERO,
    };
    capsule.tick(
        &mut scene,
        KinematicInput {
            movement: DVec2::ZERO,
            sprint: false,
            jump: true,
            jump_held: false,
        },
        1.0 / 60.0,
    );
    assert!(!capsule.grounded);
    assert!(
        (capsule.velocity.x - f64::from(inherited_x)).abs() < 0.05,
        "{:?}",
        capsule.velocity
    );
}

#[test]
fn construction_step_climbs_only_to_the_compiled_top_face() {
    let mut graph = ConstructionGraph::new();
    let parts = graph
        .apply_batch([
            BuildCommand::Spawn(CuboidSpec::new([16, 1, 16], BuildPose::default()).unwrap()),
            BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 1, 8],
                    BuildPose::new(bevy_math::IVec3::new(4, 1, 0), GridRotation::default()),
                )
                .unwrap(),
            ),
        ])
        .unwrap()
        .into_iter()
        .filter_map(|outcome| match outcome {
            BuildOutcome::Spawned(part) => Some(part),
            _ => None,
        })
        .collect::<Vec<_>>();
    let creation = graph.compile_with_static_parts(parts).unwrap();
    let step_top = creation
        .colliders
        .iter()
        .filter_map(|collider| match collider.shape {
            mechanic_core::ColliderShape::Cuboid { half_extents, .. } => Some(
                creation.compounds[collider.compound_index as usize]
                    .root_translation
                    .y
                    + collider.local_center.y
                    + half_extents.y,
            ),
            mechanic_core::ColliderShape::Convex(_) => None,
        })
        .fold(f32::NEG_INFINITY, f32::max);
    let mut index = crate::ConstructionCollisionIndex::new(&creation);
    let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::new(-0.5, 0.125, 0.0)));
    capsule.grounded = true;
    let mut reported_step = 0.0_f32;
    let mut maximum_height = capsule.position.0.y;
    for _ in 0..25 {
        let before = capsule.position;
        let mut scene = KinematicCollisionScene {
            terrain: &Empty,
            construction: Some(&mut index),
            floating_origin: DVec3::ZERO,
        };
        let result = capsule.tick(
            &mut scene,
            KinematicInput {
                movement: DVec2::X,
                sprint: false,
                jump: false,
                jump_held: false,
            },
            1.0 / 60.0,
        );
        reported_step = reported_step.max(result.stepped_height);
        maximum_height = maximum_height.max(capsule.position.0.y);
        if result.stepped_height > 0.0 {
            assert!(
                (capsule.position.0.y - before.0.y - f64::from(result.stepped_height)).abs()
                    < 1.0e-5
            );
        }
    }
    for _ in 0..30 {
        let mut scene = KinematicCollisionScene {
            terrain: &Empty,
            construction: Some(&mut index),
            floating_origin: DVec3::ZERO,
        };
        capsule.tick(&mut scene, KinematicInput::default(), 1.0 / 60.0);
    }
    assert!(capsule.position.0.x > 0.5, "{:?}", capsule.position);
    assert!(reported_step > 0.0, "the ledge should use the step path");
    assert!(
        maximum_height <= f64::from(step_top) + 1.0e-3,
        "step overshot its top: {maximum_height} > {step_top}",
    );
    assert!(
        (capsule.position.0.y - f64::from(step_top)).abs() < 0.02,
        "{:?}, step top {step_top}",
        capsule.position,
    );
    assert!(capsule.grounded);
}

#[test]
fn construction_step_can_be_climbed_at_an_angle() {
    let mut graph = ConstructionGraph::new();
    let parts = graph
        .apply_batch([
            BuildCommand::Spawn(CuboidSpec::new([16, 1, 16], BuildPose::default()).unwrap()),
            BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 1, 8],
                    BuildPose::new(bevy_math::IVec3::new(4, 1, 0), GridRotation::default()),
                )
                .unwrap(),
            ),
        ])
        .unwrap()
        .into_iter()
        .filter_map(|outcome| match outcome {
            BuildOutcome::Spawned(part) => Some(part),
            _ => None,
        })
        .collect::<Vec<_>>();
    let creation = graph.compile_with_static_parts(parts).unwrap();
    let mut index = crate::ConstructionCollisionIndex::new(&creation);
    let direction = DVec2::new(1.0, 1.0).normalize();
    let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::new(0.19, 0.125, -0.5)));
    capsule.grounded = true;
    capsule.velocity = DVec3::new(direction.x, 0.0, direction.y) * capsule.config.walk_speed;
    let mut scene = KinematicCollisionScene {
        terrain: &Empty,
        construction: Some(&mut index),
        floating_origin: DVec3::ZERO,
    };
    let result = capsule.tick(
        &mut scene,
        KinematicInput {
            movement: direction,
            sprint: false,
            jump: false,
            jump_held: false,
        },
        1.0 / 60.0,
    );

    assert!(
        result.stepped_height > 0.0,
        "the angled approach should use the step path: {:?}",
        capsule.position
    );
    assert!(capsule.position.0.y > 0.3, "{:?}", capsule.position);
    assert!(capsule.grounded);
}

#[test]
fn grazing_a_step_edge_does_not_repeat_step_and_fall() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [4, 1, 8],
                BuildPose::new(bevy_math::IVec3::new(4, 0, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        panic!("cuboid spawn returns its part");
    };
    let creation = graph.compile_with_static_parts([part]).unwrap();
    let mut index = crate::ConstructionCollisionIndex::new(&creation);
    let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::new(-0.5, 0.0, 1.01)));
    capsule.grounded = true;
    let mut step_count = 0;
    let mut largest_rise = 0.0_f64;
    for _ in 0..120 {
        let previous_height = capsule.position.0.y;
        let mut scene = KinematicCollisionScene {
            terrain: &Plane,
            construction: Some(&mut index),
            floating_origin: DVec3::ZERO,
        };
        let result = capsule.tick(
            &mut scene,
            KinematicInput {
                movement: DVec2::X,
                sprint: false,
                jump: false,
                jump_held: false,
            },
            1.0 / 60.0,
        );
        step_count += usize::from(result.stepped_height > 0.0);
        largest_rise = largest_rise.max(capsule.position.0.y - previous_height);
    }

    assert_eq!(step_count, 0, "a grazing path must stay beside the step");
    assert!(largest_rise < 0.04, "grazing path jumped by {largest_rise}");
}

#[test]
fn foundation_detaches_only_after_final_anchor_is_lost() {
    let hit = raycast_density(
        &Plane,
        WorldPosition(DVec3::new(0.0, 1.0, 0.0)),
        DVec3::NEG_Y,
        2.0,
    )
    .unwrap();
    let mut support = FoundationSupport::rectangular(&Plane, hit, 0.25, 0.25);
    assert_eq!(support.valid_count(), 25);
    assert!(!support.refresh(&Plane));
    assert!(support.has_valid_anchor());
}

#[test]
fn one_block_foundation_samples_twenty_five_terrain_points() {
    let terrain = CountingPlane(Cell::new(0));
    let support = FoundationSupport::rectangular(
        &terrain,
        TerrainRayHit {
            position: WorldPosition(DVec3::ZERO),
            normal: Vec3::Y,
            distance: 0.0,
            material_weights: [0.0; TerrainMaterial::COUNT],
            chunk_generation: 0,
            triangle: 0,
        },
        0.25,
        0.25,
    );

    assert_eq!(support.sample_count(), 25);
    assert_eq!(terrain.0.get(), 500);
}

#[test]
fn active_octree_and_chunk_bvhs_match_direct_chunk_raycast() {
    let field = crate::TerrainField::new(WorldSeed(9));
    let terrain = TerrainOctree::default().snapshot();
    let nodes = [BrickCoord::new(-1, 2, -1), BrickCoord::new(0, 2, -1)];
    let mut chunks = BTreeMap::new();
    let mut ready = BTreeMap::new();
    let mut index = TerrainSpatialIndex::default();
    for coordinate in nodes {
        let id = TerrainNodeId::leaf(coordinate);
        let chunk = mesh_chunk(
            &field,
            &terrain,
            TerrainMeshRequest {
                node: id,
                generation: 3,
                transition_mask: TerrainTransitionMask::NONE,
            },
        );
        chunks.insert(id, chunk);
        ready.insert(id, TerrainTransitionMask::NONE);
        index.insert(id);
    }
    let origin = WorldPosition(DVec3::new(-0.8, 10.0, -0.8));
    let direct = chunks
        .values()
        .filter_map(|chunk| {
            chunk.raycast_sealed(TerrainTransitionMask::NONE, origin, DVec3::NEG_Y, 20.0)
        })
        .min_by(|first, second| first.distance.total_cmp(&second.distance))
        .unwrap();
    let accelerated = ActiveTerrainScene {
        chunks: &chunks,
        ready_faces: &ready,
        spatial_index: &index,
    }
    .raycast(origin, DVec3::NEG_Y, 20.0)
    .unwrap();
    assert!((accelerated.distance - direct.distance).abs() < 1.0e-9);

    index.remove(TerrainNodeId::leaf(nodes[0]));
    assert!(!index.contains(TerrainNodeId::leaf(nodes[0])));
}
