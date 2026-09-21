use super::*;

#[test]
fn jump_grace_survives_a_short_support_gap_but_expires() {
    for (ticks, should_jump) in [(3, true), (12, false)] {
        let mut capsule = landed(&Plane);
        for _ in 0..ticks {
            tick_terrain(&mut capsule, &Empty, KinematicInput::default());
        }
        tick_terrain(
            &mut capsule,
            &Empty,
            KinematicInput {
                jump: true,
                ..KinematicInput::default()
            },
        );
        assert_eq!(capsule.velocity.y > 0.0, should_jump);
    }
}

#[test]
fn buffered_jump_fires_on_landing_but_old_presses_expire() {
    for (height, should_jump) in [(0.1, true), (2.0, false)] {
        let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::Y * height));
        let mut jumped = false;
        for tick in 0..120 {
            tick_terrain(
                &mut capsule,
                &Plane,
                KinematicInput {
                    jump: tick == 0,
                    ..KinematicInput::default()
                },
            );
            jumped |= capsule.velocity.y > 0.0;
        }
        assert_eq!(jumped, should_jump);
    }
}

#[test]
fn jump_consumes_grace_and_reset_clears_pending_presses() {
    let mut capsule = landed(&Plane);
    tick_terrain(
        &mut capsule,
        &Plane,
        KinematicInput {
            jump: true,
            ..KinematicInput::default()
        },
    );
    let launched = capsule.velocity.y;
    tick_terrain(
        &mut capsule,
        &Plane,
        KinematicInput {
            jump: true,
            ..KinematicInput::default()
        },
    );
    assert!(capsule.velocity.y < launched, "no second jump during grace");
    capsule.reset_motion();
    capsule.position = WorldPosition(DVec3::ZERO);
    for _ in 0..30 {
        tick_terrain(&mut capsule, &Plane, KinematicInput::default());
        assert!(capsule.velocity.y <= 0.0);
    }
}

#[test]
fn near_vertical_terrain_does_not_grant_jumps() {
    for angle in [85.0, 90.0, 180.0] {
        let terrain = incline(angle);
        let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::new(-0.2, 0.0, 0.0)));
        for tick in 0..20 {
            tick_terrain(
                &mut capsule,
                &terrain,
                KinematicInput {
                    jump: tick == 10,
                    ..KinematicInput::default()
                },
            );
        }
        assert!(
            capsule.velocity.y <= 0.0,
            "wall/ceiling granted jump at {angle}"
        );
    }
}
