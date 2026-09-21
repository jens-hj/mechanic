mod jumping;
mod terrain_mesh;

use super::*;

struct Incline(f64);

impl TerrainDensity for Incline {
    fn density(&self, position: WorldPosition) -> f32 {
        (position.0.x * self.0.sin() - position.0.y * self.0.cos()) as f32
    }

    fn material(&self, _: WorldPosition) -> TerrainMaterial {
        TerrainMaterial::Soil
    }
}

fn incline(degrees: f64) -> Incline {
    Incline(degrees.to_radians())
}

fn landed(terrain: &impl TerrainDensity) -> KinematicCapsule {
    let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::Y));
    for _ in 0..120 {
        tick_terrain(&mut capsule, terrain, KinematicInput::default());
    }
    assert!(capsule.grounded, "failed to land: {capsule:?}");
    capsule
}

#[test]
fn walkable_terrain_slopes_support_walking_stopping_and_jumping() {
    for angle in [20.0, 35.0, 45.0, 60.0] {
        let terrain = incline(angle);
        for direction in [DVec2::X, DVec2::NEG_X, DVec2::Y] {
            let mut capsule = landed(&terrain);
            let start = capsule.position.0;
            for _ in 0..60 {
                tick_terrain(
                    &mut capsule,
                    &terrain,
                    KinematicInput {
                        movement: direction,
                        ..KinematicInput::default()
                    },
                );
                assert!(
                    capsule.grounded,
                    "lost support on {angle} degrees: {capsule:?}"
                );
            }
            let travelled = capsule.position.0 - start;
            assert!(DVec2::new(travelled.x, travelled.z).dot(direction) > 2.0);
            let mut jumping = capsule;
            for tick in 0..3 {
                tick_terrain(
                    &mut jumping,
                    &terrain,
                    KinematicInput {
                        movement: direction,
                        jump: tick == 0,
                        ..KinematicInput::default()
                    },
                );
                assert!(
                    !jumping.grounded && jumping.velocity.y > 0.0,
                    "running jump at {angle}: {jumping:?}"
                );
            }
            tick_terrain(&mut capsule, &terrain, KinematicInput::default());
            let stopped = capsule.position.0;
            for _ in 0..120 {
                tick_terrain(&mut capsule, &terrain, KinematicInput::default());
            }
            assert!(
                capsule.position.0.abs_diff_eq(stopped, 0.002),
                "drift at {angle}: {stopped:?} -> {:?}",
                capsule.position
            );
            tick_terrain(
                &mut capsule,
                &terrain,
                KinematicInput {
                    jump: true,
                    ..KinematicInput::default()
                },
            );
            assert!(
                !capsule.grounded && capsule.velocity.y > 4.0,
                "jump at {angle}: {capsule:?}"
            );
        }
    }
}

fn slope_construction(angle: f64) -> crate::ConstructionCollisionIndex {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([32, 1, 32], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        panic!("spawn");
    };
    let mut creation = graph.compile_with_static_parts([part]).unwrap();
    let mechanic_core::ColliderShape::Cuboid {
        local_rotation,
        half_extents,
    } = &mut creation.colliders[0].shape
    else {
        panic!("cuboid");
    };
    *local_rotation = bevy_math::Quat::from_rotation_z(angle.to_radians() as f32);
    // Keep the test route away from finite ramp edges at every slope.
    half_extents.x = 16.0;
    half_extents.z = 16.0;
    crate::ConstructionCollisionIndex::new(&creation)
}

#[test]
fn walkable_construction_slopes_support_walking_stopping_and_jumping() {
    for angle in [20.0, 35.0, 45.0, 60.0] {
        for direction in [DVec2::X, DVec2::NEG_X, DVec2::Y] {
            let mut index = slope_construction(angle);
            let mut scene = KinematicCollisionScene {
                terrain: &Empty,
                construction: Some(&mut index),
                floating_origin: DVec3::ZERO,
            };
            let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::Y * 2.0));
            for _ in 0..120 {
                capsule.tick(
                    &mut scene,
                    KinematicInput::default(),
                    mechanic_core::TICK_SECONDS,
                );
            }
            assert!(capsule.grounded, "landing at {angle}: {capsule:?}");
            let start = capsule.position.0;
            for _ in 0..60 {
                capsule.tick(
                    &mut scene,
                    KinematicInput {
                        movement: direction,
                        ..KinematicInput::default()
                    },
                    mechanic_core::TICK_SECONDS,
                );
                assert!(capsule.grounded, "walking at {angle}: {capsule:?}");
            }
            let travelled = capsule.position.0 - start;
            assert!(
                DVec2::new(travelled.x, travelled.z).dot(direction) > 1.0,
                "stuck at {angle}: {travelled:?}"
            );
            let mut jumping = capsule;
            for tick in 0..3 {
                jumping.tick(
                    &mut scene,
                    KinematicInput {
                        movement: direction,
                        jump: tick == 0,
                        ..KinematicInput::default()
                    },
                    mechanic_core::TICK_SECONDS,
                );
                assert!(
                    !jumping.grounded && jumping.velocity.y > 0.0,
                    "running jump at {angle}: {jumping:?}"
                );
            }
            capsule.tick(
                &mut scene,
                KinematicInput::default(),
                mechanic_core::TICK_SECONDS,
            );
            let stopped = capsule.position.0;
            for _ in 0..120 {
                capsule.tick(
                    &mut scene,
                    KinematicInput::default(),
                    mechanic_core::TICK_SECONDS,
                );
            }
            assert!(
                capsule.position.0.abs_diff_eq(stopped, 0.002),
                "drift at {angle}"
            );
            capsule.tick(
                &mut scene,
                KinematicInput {
                    jump: true,
                    ..KinematicInput::default()
                },
                mechanic_core::TICK_SECONDS,
            );
            assert!(
                !capsule.grounded && capsule.velocity.y > 4.0,
                "jump at {angle}: {capsule:?}"
            );
        }
    }
}

#[test]
fn steep_terrain_slides_allow_recovery_jumps() {
    for angle in [65.0, 75.0] {
        let terrain = incline(angle);
        let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::Y));
        for _ in 0..60 {
            tick_terrain(&mut capsule, &terrain, KinematicInput::default());
        }
        assert!(!capsule.grounded);
        let before = capsule.position.0;
        for _ in 0..15 {
            tick_terrain(&mut capsule, &terrain, KinematicInput::default());
        }
        assert!(
            capsule.position.0.y < before.y - 0.05,
            "must slide at {angle}"
        );
        tick_terrain(
            &mut capsule,
            &terrain,
            KinematicInput {
                jump: true,
                ..KinematicInput::default()
            },
        );
        assert!(
            capsule.velocity.y > 4.0 && !capsule.grounded,
            "recovery at {angle}: {capsule:?}"
        );
    }
}

#[test]
fn steep_construction_slides_allow_recovery_jumps() {
    for angle in [65.0, 75.0] {
        let mut index = slope_construction(angle);
        let mut scene = KinematicCollisionScene {
            terrain: &Empty,
            construction: Some(&mut index),
            floating_origin: DVec3::ZERO,
        };
        let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::Y * 2.0));
        for _ in 0..30 {
            capsule.tick(
                &mut scene,
                KinematicInput::default(),
                mechanic_core::TICK_SECONDS,
            );
        }
        assert!(!capsule.grounded);
        capsule.tick(
            &mut scene,
            KinematicInput {
                jump: true,
                ..KinematicInput::default()
            },
            mechanic_core::TICK_SECONDS,
        );
        assert!(capsule.velocity.y > 4.0, "recovery at {angle}: {capsule:?}");
    }
}
