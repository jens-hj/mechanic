use super::*;
use bevy_math::{DMat3, IVec3, Vec3};
use mechanic_core::{
    BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, FaceKind,
    FaceRef, GridRotation, PartId,
};

fn spawn(graph: &mut ConstructionGraph, position: IVec3, size: [u8; 3]) -> PartId {
    let BuildOutcome::Spawned(id) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(size, BuildPose::new(position, GridRotation::default())).unwrap(),
        ))
        .unwrap()
    else {
        panic!("expected spawned body");
    };
    id
}

fn free_body() -> CompiledCreation {
    let mut graph = ConstructionGraph::new();
    spawn(&mut graph, IVec3::ZERO, [4, 2, 1]);
    graph.compile().unwrap()
}

fn tree(anchored: bool, reverse: bool) -> CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let root = spawn(&mut graph, IVec3::ZERO, [4, 4, 4]);
    let middle = spawn(&mut graph, IVec3::new(4, 0, 0), [4, 4, 4]);
    let tip = spawn(&mut graph, IVec3::new(4, 4, 0), [4, 4, 4]);
    for (a, face_a, b, face_b, anchor, axis) in [
        (
            root,
            FaceKind::PositiveX,
            middle,
            FaceKind::NegativeX,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::X,
        ),
        (
            middle,
            FaceKind::PositiveY,
            tip,
            FaceKind::NegativeY,
            Vec3::new(1.0, 0.5, 0.0),
            Vec3::Y,
        ),
    ] {
        let (a, face_a, b, face_b, axis) = if reverse {
            (b, face_b, a, face_a, -axis)
        } else {
            (a, face_a, b, face_b, axis)
        };
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(a, face_a),
                FaceRef::part(b, face_b),
                anchor,
                axis,
            )))
            .unwrap();
    }
    graph
        .compile_with_static_parts(if anchored { vec![root] } else { vec![] })
        .unwrap()
}

// Independent physical invariants reconstructed from per-body COM motion.
fn invariants(creation: &CompiledCreation, state: &MachineState) -> (DVec3, DVec3, f64) {
    let model = MachineDynamics::assemble(creation, &state.poses, &state.coordinates).unwrap();
    let motion = model.body_motions(&state.velocities).unwrap();
    let mut linear = DVec3::ZERO;
    let mut angular = DVec3::ZERO;
    let mut energy = 0.0;
    for (body, inertia) in creation.dynamics.inertias.iter().enumerate() {
        if creation.compounds[body].is_static {
            continue;
        }
        let pose = model.poses[body];
        let rotation = DMat3::from_quat(pose.rotation);
        let center = pose.position + pose.rotation * inertia.center.as_dvec3();
        let momentum = motion[body].linear * f64::from(inertia.mass);
        let spin =
            rotation * inertia.rotational.as_dmat3() * rotation.transpose() * motion[body].angular;
        linear += momentum;
        angular += center.cross(momentum) + spin;
        energy += 0.5 * (motion[body].linear.dot(momentum) + motion[body].angular.dot(spin));
    }
    (linear, angular, energy)
}

#[test]
fn free_fall_matches_ballistic_motion_at_the_external_clock() {
    let creation = free_body();
    let mut state = MachineState::at_rest(&creation);
    let start = state.poses[0].position;
    state.velocities[..3].copy_from_slice(&[2.0, 3.0, -1.0]);
    let mut world = CpuFreeMotion::new(creation, 7, state).unwrap();
    let gravity = mechanic_core::GRAVITY;
    for _ in 0..60 {
        world.step(gravity, 1, &[]).unwrap();
    }
    let snapshot = world.snapshot();
    assert_eq!(snapshot.tick, 60);
    let position_error =
        (snapshot.state.poses[0].position - start - DVec3::new(2.0, 3.0, -1.0) - gravity * 0.5)
            .length();
    println!("ballistic_position_error_m={position_error:e}");
    assert!(
        (snapshot.state.poses[0].position - start - DVec3::new(2.0, 3.0, -1.0) - gravity * 0.5)
            .length()
            < 1e-11
    );
    assert!(
        (snapshot.state.velocities[1] - (3.0 - mechanic_core::STANDARD_GRAVITY_M_S2)).abs() < 1e-11
    );
}

#[test]
fn off_centre_impulse_changes_momentum_once_independent_of_substeps() {
    let creation = free_body();
    let state = MachineState::at_rest(&creation);
    let point = state.poses[0].position + DVec3::X;
    let impulse = DVec3::Y * 10.0;
    for substeps in [1, 2, 4, 8] {
        let mut world = CpuFreeMotion::new(creation.clone(), 2, state.clone()).unwrap();
        world
            .step(
                DVec3::ZERO,
                substeps,
                &[ExternalImpulse {
                    tick: 1,
                    topology_generation: 2,
                    body: 0,
                    point,
                    impulse,
                }],
            )
            .unwrap();
        let (linear, angular, _) = invariants(&creation, &world.snapshot().state);
        assert!((linear - impulse).length() < 1e-10);
        assert!((angular - point.cross(impulse)).length() < 1e-9);
        world.step(DVec3::ZERO, substeps, &[]).unwrap();
        let (after, _, _) = invariants(&creation, &world.snapshot().state);
        assert!((after - linear).length() < 1e-10);
    }
}

#[test]
fn asymmetric_free_rotation_converges_to_conserved_angular_momentum_and_energy() {
    let creation = free_body();
    let mut state = MachineState::at_rest(&creation);
    state.velocities[3..6].copy_from_slice(&[3.0, -2.0, 4.0]);
    let (_, initial_angular, initial_energy) = invariants(&creation, &state);
    let mut errors = Vec::new();
    for substeps in [1, 2, 4, 8] {
        let mut world = CpuFreeMotion::new(creation.clone(), 1, state.clone()).unwrap();
        for _ in 0..120 {
            world.step(DVec3::ZERO, substeps, &[]).unwrap();
        }
        let (linear, angular, energy) = invariants(&creation, &world.snapshot().state);
        assert!(linear.length() < 1e-10);
        errors.push(
            ((angular - initial_angular).length() / initial_angular.length())
                .max((energy / initial_energy - 1.0).abs()),
        );
        assert!((world.snapshot().state.poses[0].rotation.length_squared() - 1.0).abs() < 1e-14);
    }
    assert!(errors[3] < 1e-8, "{errors:?}");
    println!("free_rotation_relative_conservation_errors_1_2_4_8={errors:?}");
    for pair in errors.windows(2) {
        assert!(pair[1] < pair[0] / 10.0, "{errors:?}");
    }
}

#[test]
fn floating_tree_conserves_momentum_and_keeps_exact_joint_anchors_and_axes() {
    for reverse in [false, true] {
        let creation = tree(false, reverse);
        assert_eq!(
            creation
                .loop_topology
                .body_parents
                .iter()
                .any(|p| p.bearing_direction == 1),
            reverse
        );
        let mut state = MachineState::at_rest(&creation);
        state.coordinates.copy_from_slice(&[0.3, -0.4]);
        state
            .velocities
            .copy_from_slice(&[0.3, -0.1, 0.2, 0.7, 0.4, -0.5, 1.3, -1.7]);
        let (initial_linear, initial_angular, initial_energy) = invariants(&creation, &state);
        let mut world = CpuFreeMotion::new(creation.clone(), 1, state).unwrap();
        for _ in 0..120 {
            let snapshot = world.step(DVec3::ZERO, 4, &[]).unwrap();
            for bearing in &creation.bearings {
                let a = snapshot.state.poses[bearing.compound_a as usize];
                let b = snapshot.state.poses[bearing.compound_b as usize];
                let anchor = a.position + a.rotation * bearing.local_anchor_a.as_dvec3()
                    - b.position
                    - b.rotation * bearing.local_anchor_b.as_dvec3();
                let axis_a = a.rotation * bearing.local_axis_a.as_dvec3();
                let axis_b = b.rotation * bearing.local_axis_b.as_dvec3();
                assert!(anchor.length() < 1e-12);
                assert!(axis_a.cross(axis_b).length() < 1e-12);
            }
        }
        let (linear, angular, energy) = invariants(&creation, &world.snapshot().state);
        println!(
            "floating_tree reverse={reverse} relative_linear_error={:e} relative_angular_error={:e} relative_energy_error={:e}",
            (linear - initial_linear).length() / initial_linear.length(),
            (angular - initial_angular).length() / initial_angular.length(),
            (energy / initial_energy - 1.0).abs()
        );
        assert!((linear - initial_linear).length() / initial_linear.length() < 1e-8);
        assert!((angular - initial_angular).length() / initial_angular.length() < 1e-8);
        assert!((energy / initial_energy - 1.0).abs() < 1e-8);
    }
}

#[test]
fn anchored_tree_keeps_its_root_fixed_while_joint_motion_evolves() {
    let creation = tree(true, false);
    let mut state = MachineState::at_rest(&creation);
    state.velocities.copy_from_slice(&[1.2, -0.8]);
    let root = state.poses[0];
    let mut world = CpuFreeMotion::new(creation, 1, state).unwrap();
    for _ in 0..60 {
        world.step(mechanic_core::GRAVITY, 4, &[]).unwrap();
    }
    assert_eq!(world.snapshot().state.poses[0], root);
    assert!(world.snapshot().state.coordinates[0].abs() > 0.1);
}

#[test]
fn rejected_commands_and_failed_numerics_preserve_the_last_snapshot() {
    let creation = free_body();
    let state = MachineState::at_rest(&creation);
    let mut world = CpuFreeMotion::new(creation, 9, state).unwrap();
    let before = world.snapshot().clone();
    let command = ExternalImpulse {
        tick: 1,
        topology_generation: 9,
        body: 0,
        point: DVec3::X,
        impulse: DVec3::Y,
    };
    for invalid in [
        ExternalImpulse { tick: 2, ..command },
        ExternalImpulse {
            topology_generation: 8,
            ..command
        },
        ExternalImpulse { body: 1, ..command },
        ExternalImpulse {
            impulse: DVec3::NAN,
            ..command
        },
    ] {
        assert_eq!(
            world.step(DVec3::ZERO, 1, &[command, invalid]),
            Err(PhysicsError::InvalidCommand)
        );
        assert_eq!(world.snapshot(), &before);
    }
    // Finite input overflows during dynamics evaluation, after impulse processing.
    assert!(world.step(DVec3::splat(f64::MAX), 1, &[command]).is_err());
    assert_eq!(world.snapshot(), &before);
    assert!(world.step(DVec3::ZERO, 3, &[]).is_err());
    assert_eq!(world.snapshot(), &before);
    world.step(DVec3::ZERO, 1, &[command]).unwrap();
    assert_eq!(world.snapshot().tick, 1);
}

#[test]
fn repeated_fixed_input_ticks_have_identical_completed_hashes() {
    let creation = tree(false, false);
    let state = MachineState::at_rest(&creation);
    let mut a = CpuFreeMotion::new(creation.clone(), 42, state.clone()).unwrap();
    let mut b = CpuFreeMotion::new(creation, 42, state).unwrap();
    for tick in 1..=60 {
        let command = ExternalImpulse {
            tick,
            topology_generation: 42,
            body: 2,
            point: DVec3::new(0.1, 0.2, 0.3),
            impulse: DVec3::new(0.7, -0.4, 0.5),
        };
        let a = a.step(mechanic_core::GRAVITY, 2, &[command]).unwrap();
        let b = b.step(mechanic_core::GRAVITY, 2, &[command]).unwrap();
        assert_eq!(a.state_hash(), b.state_hash(), "tick {tick}");
        assert_eq!(a, b);
    }
    println!(
        "repeated_tree_ticks=60 final_state_hash={:016x}",
        a.snapshot().state_hash()
    );
}

#[test]
fn impulse_at_a_child_accelerates_the_whole_machine_with_correct_total_momentum() {
    let creation = tree(false, false);
    let state = MachineState::at_rest(&creation);
    let point = state.poses[2].position + DVec3::Z * 0.2;
    let impulse = DVec3::new(20.0, 40.0, -30.0);
    let mut world = CpuFreeMotion::new(creation.clone(), 1, state).unwrap();
    world
        .step(
            DVec3::ZERO,
            4,
            &[ExternalImpulse {
                tick: 1,
                topology_generation: 1,
                body: 2,
                point,
                impulse,
            }],
        )
        .unwrap();
    let (linear, angular, _) = invariants(&creation, &world.snapshot().state);
    assert!((linear - impulse).length() < 1e-8);
    assert!((angular - point.cross(impulse)).length() < 1e-8);
    assert!(
        world.snapshot().state.velocities[..6]
            .iter()
            .any(|v| v.abs() > 1e-5)
    );
}

#[test]
fn principal_axis_spin_matches_the_analytic_orientation() {
    let creation = free_body();
    let mut state = MachineState::at_rest(&creation);
    state.velocities[5] = 3.0;
    let initial = state.poses[0].rotation;
    let mut world = CpuFreeMotion::new(creation, 1, state).unwrap();
    for _ in 0..60 {
        world.step(DVec3::ZERO, 4, &[]).unwrap();
    }
    let expected = DQuat::from_rotation_z(3.0) * initial;
    let difference = world.snapshot().state.poses[0].rotation * expected.inverse();
    assert!(DVec3::new(difference.x, difference.y, difference.z).length() < 1e-10);
}

#[test]
fn authored_drive_and_loop_constraints_cannot_be_silently_ignored() {
    let mut creation = tree(false, false);
    creation.coordinate_drives[0].max_angle = 0.5;
    let state = MachineState::at_rest(&creation);
    assert!(matches!(
        CpuFreeMotion::new(creation.clone(), 1, state),
        Err(PhysicsError::UnsupportedFreeMotion)
    ));
    creation.coordinate_drives[0] = CoordinateDrive::PASSIVE;
    creation
        .dynamics
        .loops
        .push(mechanic_core::LoopConstraintPattern {
            bearing: 0,
            branch_heads: [None, None],
        });
    let state = MachineState::at_rest(&creation);
    assert!(matches!(
        CpuFreeMotion::new(creation, 1, state),
        Err(PhysicsError::UnsupportedFreeMotion)
    ));
    let mut creation = tree(false, false);
    creation.bearings[0].kind = JointKind::Suspension(
        mechanic_core::SuspensionSpec::new(Some(mechanic_core::SpringSpec::default()), None, None)
            .unwrap(),
    );
    let state = MachineState::at_rest(&creation);
    assert!(matches!(
        CpuFreeMotion::new(creation, 1, state),
        Err(PhysicsError::UnsupportedFreeMotion)
    ));
}
