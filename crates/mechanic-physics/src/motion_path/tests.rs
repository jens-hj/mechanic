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

fn authored_car() -> CompiledCreation {
    let instance: mechanic_world::WorldCreationInstanceDoc = ron::from_str(include_str!(
        "../../../mechanic-bench/tests/fixtures/driven_car_instance.ron"
    ))
    .unwrap();
    let loaded = instance.creation.into_graph().unwrap();
    loaded
        .graph
        .compile_with_sockets([], &loaded.sockets)
        .unwrap()
}

// The authored car with its joints displaced, and a path moving every row.
fn tumbling_car(creation: &CompiledCreation) -> (MachineState, Vec<f64>) {
    let mut initial = MachineState::at_rest(creation);
    for (i, q) in initial.coordinates.iter_mut().enumerate() {
        *q = if i % 2 == 0 { 0.07 } else { -0.11 };
    }
    let displacement = (0..initial.velocities.len())
        .map(|i| if i % 2 == 0 { 0.6 } else { -0.4 })
        .collect::<Vec<_>>();
    (initial, displacement)
}

#[test]
fn authored_car_spatial_derivatives_and_acceleration_bounds_cover_reconstructed_motion() {
    let creation = authored_car();
    let (initial, displacement) = tumbling_car(&creation);
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

#[test]
fn tree_bounds_cover_motion_relative_to_the_tree_root() {
    let creation = authored_car();
    let (initial, displacement) = tumbling_car(&creation);
    let motion = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
    let root_of = |mut body: usize| {
        while !creation.loop_topology.body_parents[body].is_root {
            body = creation.loop_topology.body_parents[body].parent_body as usize;
        }
        body
    };
    let rooted = |poses: &[BodyPose], body: usize, local: DVec3| {
        let root = poses[root_of(body)];
        let pose = poses[body];
        root.rotation.inverse() * (pose.position + pose.rotation * local - root.position)
    };
    let start = motion.initial_poses();
    let mut moved_roots = 0;
    for body in 0..start.len() {
        let bound = motion.tree_bounds()[body];
        if root_of(body) == body {
            assert!(bound.point_speed(1.0) <= 0.0, "root {body}: {bound:?}");
            moved_roots += usize::from(motion.bounds()[body].point_speed(1.0) > 0.0);
        }
    }
    assert!(moved_roots > 0, "the path moves no root");
    for step in 1..=64 {
        let fraction = f64::from(step) / 64.0;
        let poses = motion.poses_at(fraction).unwrap();
        for body in 0..poses.len() {
            for local in [DVec3::ZERO, DVec3::new(0.3, -0.4, 0.2)] {
                let moved = rooted(&poses, body, local).distance(rooted(start, body, local));
                let bound = motion.tree_bounds()[body].point_speed(local.length().next_up());
                assert!(
                    moved <= bound * fraction + 1e-9,
                    "body {body} at {fraction}: moved {moved} in the root frame, bound {bound}"
                );
            }
        }
    }
}

#[test]
fn symmetric_shape_bounds_cover_their_sampled_centres_and_axes() {
    let creation = authored_car();
    let (initial, displacement) = tumbling_car(&creation);
    let motion = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
    let start = motion.initial_poses();
    let samples = (1..=64)
        .map(|step| {
            let fraction = f64::from(step) / 64.0;
            (fraction, motion.poses_at(fraction).unwrap())
        })
        .collect::<Vec<_>>();
    for (body, spin) in motion.spins().iter().enumerate() {
        let axis = if spin.axis == DVec3::ZERO {
            DVec3::X
        } else {
            spin.axis
        };
        let across = axis.any_orthonormal_vector();
        for (centre, symmetry) in [
            (spin.pivot + 0.2 * axis, axis),
            (spin.pivot + 0.3 * across - 0.1 * axis, axis),
            (spin.pivot + 0.2 * axis, (axis + 0.5 * across).normalize()),
        ] {
            let [speed, turn] = spin.symmetric(centre, symmetry, 0.0);
            let placed = |pose: BodyPose| {
                (
                    pose.position + pose.rotation * centre,
                    pose.rotation * symmetry,
                )
            };
            let (first_centre, first_axis) = placed(start[body]);
            for (fraction, poses) in &samples {
                let (centre, axis) = placed(poses[body]);
                let moved = centre.distance(first_centre);
                assert!(
                    moved <= speed * fraction + 1e-9,
                    "body {body} at {fraction}: centre moved {moved}, bound {speed}"
                );
                // The sine of the angle turned never exceeds the angle.
                let turned = axis.cross(first_axis).length();
                assert!(
                    turned <= turn * fraction + 1e-9,
                    "body {body} at {fraction}: axis turned {turned}, bound {turn}"
                );
            }
        }
    }
}

#[test]
fn a_wheel_spinning_on_a_still_car_moves_no_shape_about_its_axle() {
    let creation = authored_car();
    let initial = MachineState::at_rest(&creation);
    let mut spinning = 0;
    for (body, row) in creation.dynamics.body_bearings.iter().enumerate() {
        let Some(bearing) = row.map(|row| creation.bearings[row]) else {
            continue;
        };
        let Some(coordinate) = bearing.coordinate_index else {
            continue;
        };
        if bearing.kind.is_translational() {
            continue;
        }
        // 400 rad/s over one 1/240 s substep, with the rest of the car still.
        let mut displacement = vec![0.0; initial.velocities.len()];
        displacement[creation.dynamics.coordinate_velocities[coordinate as usize]] = 400.0 / 240.0;
        let motion = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
        let spin = motion.spins()[body];
        let [speed, turn] = spin.symmetric(spin.pivot + 0.1 * spin.axis, spin.axis, 0.5);
        assert!(motion.bounds()[body].point_speed(0.5) > 0.8);
        assert!(speed < 1e-9 && turn < 1e-12, "body {body}: {speed} {turn}");
        spinning += 1;
    }
    assert!(spinning > 0);
}
