use super::*;
use bevy_math::IVec3;
use mechanic_core::{BuildCommand, BuildPose, ConstructionGraph, CuboidSpec, GridRotation};

fn bodies(count: i32) -> CompiledCreation {
    let mut graph = ConstructionGraph::new();
    for x in 0..count {
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::new(IVec3::new(x * 2, 0, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
    }
    graph.compile().unwrap()
}

#[test]
fn full_root_rotation_is_preserved_between_matching_endpoint_orientations() {
    let creation = bodies(1);
    let initial = MachineState::at_rest(&creation);
    let mut displacement = vec![0.0; 6];
    displacement[5] = std::f64::consts::TAU;
    let motion = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
    let start = motion.poses_at(0.0).unwrap()[0];
    let middle = motion.poses_at(0.5).unwrap()[0];
    let end = motion.poses_at(1.0).unwrap()[0];
    assert!((end.rotation.dot(start.rotation).abs() - 1.0).abs() < 1e-12);
    assert!((middle.rotation * DVec3::X).distance(start.rotation * -DVec3::X) < 1e-12);
    assert!((motion.bounds()[0].angular_speed - std::f64::consts::TAU).abs() < 1e-12);
    assert_eq!(motion.generation(), 7);
}

#[test]
fn pose_only_queries_are_not_limited_by_dense_inertia_capacity() {
    let creation = bodies(100);
    let initial = MachineState::at_rest(&creation);
    let displacement = vec![0.01; 600];
    let motion = MachineMotion::new(&creation, 1, &initial, &displacement).unwrap();
    assert_eq!(motion.poses_at(0.25).unwrap().len(), 100);
    assert!(matches!(
        MachineDynamics::assemble(&creation, &initial.poses, &[]),
        Err(PhysicsError::ReferenceCapacity)
    ));
}

#[test]
fn invalid_displacements_and_fractions_fail_without_changing_the_initial_state() {
    let creation = bodies(1);
    let initial = MachineState::at_rest(&creation);
    let expected = initial.clone();
    assert!(MachineMotion::new(&creation, 1, &initial, &[0.0; 5]).is_err());
    assert!(MachineMotion::new(&creation, 1, &initial, &[f64::NAN; 6]).is_err());
    let displacement = [0.0; 6];
    let motion = MachineMotion::new(&creation, 1, &initial, &displacement).unwrap();
    assert!(motion.poses_at(1.1).is_err());
    assert!(motion.poses_at(f64::NAN).is_err());
    assert_eq!(initial, expected);
}

#[test]
fn a_near_unit_root_has_the_same_normalized_pose_at_cached_and_sampled_start() {
    let creation = bodies(1);
    let mut initial = MachineState::at_rest(&creation);
    initial.poses[0].rotation = bevy_math::DQuat::from_rotation_z(0.3) * (1.0 + 1e-12);
    let saved = initial.clone();
    let displacement = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1];
    let motion = MachineMotion::new(&creation, 1, &initial, &displacement).unwrap();
    assert_eq!(motion.initial_poses(), motion.poses_at(0.0).unwrap());
    assert_eq!(initial, saved);
    let mut invalid = initial.clone();
    invalid.poses.clear();
    assert!(MachineMotion::new(&creation, 1, &invalid, &displacement).is_err());
    invalid = initial;
    invalid.poses[0].rotation *= 2.0;
    assert!(MachineMotion::new(&creation, 1, &invalid, &displacement).is_err());
}

#[test]
fn motion_bounds_round_outward_and_reject_invalid_radius_inputs() {
    assert!(upper_length(DVec3::new(3.0, 4.0, 0.0)) >= 5.0);
    assert!(upper_length(DVec3::splat(1e-200)) >= 3.0_f64.sqrt() * 1e-200);
    let bound = MotionBound {
        origin_speed: 0.1,
        angular_speed: 0.2,
        ..MotionBound::default()
    };
    assert!(bound.point_speed(0.3) >= 0.16);
    assert_eq!(
        MotionBound::default().point_speed(1.0).to_bits(),
        0.0_f64.to_bits()
    );
    for radius in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(bound.point_speed(radius).is_infinite());
        assert!(bound.point_acceleration(radius).is_infinite());
    }
    assert!(bound.point_acceleration(0.3) >= 0.012);
}

#[test]
fn authored_car_spatial_derivatives_and_acceleration_bounds_cover_reconstructed_motion() {
    let instance: mechanic_world::WorldCreationInstanceDoc = ron::from_str(include_str!(
        "../../../mechanic-bench/tests/fixtures/driven_car_instance.ron"
    ))
    .unwrap();
    let loaded = instance.creation.into_graph().unwrap();
    let creation = loaded
        .graph
        .compile_with_suspension_sockets([], &loaded.sockets)
        .unwrap();
    let mut initial = MachineState::at_rest(&creation);
    for (i, q) in initial.coordinates.iter_mut().enumerate() {
        *q = if i % 2 == 0 { 0.07 } else { -0.11 };
    }
    let displacement = (0..initial.velocities.len())
        .map(|i| if i % 2 == 0 { 0.6 } else { -0.4 })
        .collect::<Vec<_>>();
    let motion = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
    // Derivative checks use a forward finite difference and its rigorous
    // acceleration-based truncation bound. Additional 1e-8 covers cancellation
    // in reconstructed floating-point positions divided by h; it is test-only.
    let h = 1e-4;
    for fraction in [0.0, 0.17, 0.63, 0.999] {
        let poses = motion.poses_at(fraction).unwrap();
        let next = motion.poses_at(fraction + h).unwrap();
        let velocities = motion.velocities_at_poses(&poses);
        for (body, pose) in poses.iter().enumerate() {
            for local in [DVec3::ZERO, DVec3::new(0.3, -0.4, 0.2)] {
                let point = pose.position + pose.rotation * local;
                let endpoint = next[body].position + next[body].rotation * local;
                let difference = (endpoint - point) / h;
                let error =
                    0.5 * motion.bounds()[body].point_acceleration(local.length().next_up()) * h
                        + 1e-8;
                for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
                    let [lower, upper] = velocities[body]
                        .point_projection(pose.position, point, axis)
                        .unwrap();
                    let observed = difference.dot(axis);
                    assert!(
                        observed >= lower - error && observed <= upper + error,
                        "body={body} fraction={fraction} local={local} observed={observed} interval=[{lower}, {upper}] error={error}"
                    );
                }
            }
        }
    }
}
