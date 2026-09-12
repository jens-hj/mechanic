#![allow(clippy::float_cmp)] // Recovery must consume exactly zero physical time.
use super::*;

#[test]
fn split_recovery_changes_positions_without_time_velocity_or_physical_impulses() {
    let (creation, geometry, mut poses) = cube();
    poses[0].position.y = 0.499;
    let initial = MachineState {
        poses,
        velocities: vec![0.3, -0.2, 0.1, 0.4, -0.5, 0.6],
        ..MachineState::at_rest(&creation)
    };
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    let run = || {
        let mut state = initial.clone();
        let mut diagnostics = JointTickDiagnostics::default();
        let result = super::super::super::recovery::correct(
            &creation,
            &[],
            &mut state,
            fixed(1),
            Some(&terrain),
            &mut diagnostics,
        );
        assert!(result.is_ok(), "{result:?} {diagnostics:?}");
        assert_eq!(state.velocities, initial.velocities);
        assert_eq!(state.coordinates, initial.coordinates);
        assert_eq!(diagnostics.accepted_seconds, 0.0);
        assert!(diagnostics.drive_impulses.is_empty());
        assert_eq!(diagnostics.impact_attempts, 0);
        assert!(diagnostics.terrain_recovery_passes > 0);
        let query = scene
            .recovery_contacts(&geometry, &state.poses, DVec3::ZERO)
            .unwrap();
        assert!(query.contacts.iter().all(|point| point.depth <= 1e-12));
        let deepest = mechanic_core::ContactPolytope::from_collider(&creation.colliders[0])
            .unwrap()
            .transformed(state.poses[0].position, state.poses[0].rotation)
            .unwrap()
            .bounds()[0]
            .y;
        assert!(deepest >= -1e-12);
        state
    };
    assert_eq!(run(), run());
}

#[test]
fn split_recovery_rejects_excessive_penetration_before_mutating_state() {
    let (creation, geometry, mut poses) = cube();
    poses[0].position.y -= 0.003;
    let initial = MachineState {
        poses,
        ..MachineState::at_rest(&creation)
    };
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    let mut state = initial.clone();
    let mut diagnostics = JointTickDiagnostics::default();
    assert_eq!(
        super::super::super::recovery::correct(
            &creation,
            &[],
            &mut state,
            fixed(1),
            Some(&terrain),
            &mut diagnostics,
        ),
        Err(PhysicsError::NotConverged)
    );
    assert_eq!(state, initial);
    assert_eq!(diagnostics.factorizations, 0);
    assert_eq!(diagnostics.accepted_seconds, 0.0);
}

#[test]
fn split_recovery_does_not_extend_finite_terrain_into_empty_space() {
    let (creation, geometry, mut poses) = cube();
    poses[0].position.x = 3.0;
    let initial = MachineState {
        poses,
        ..MachineState::at_rest(&creation)
    };
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    let mut state = initial.clone();
    let mut diagnostics = JointTickDiagnostics::default();
    super::super::super::recovery::correct(
        &creation,
        &[],
        &mut state,
        fixed(1),
        Some(&terrain),
        &mut diagnostics,
    )
    .unwrap();
    assert_eq!(state, initial);
    assert_eq!(diagnostics.terrain_recovery_passes, 0);
    assert_eq!(diagnostics.position_factor_solves, 0);
}
