//! Held bodies and their release.

use super::*;

#[test]
#[expect(clippy::float_cmp, reason = "held velocities must remain exactly zero")]
pub(super) fn held_rotational_and_linear_components_ignore_drives_impulses_and_reconstruction() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    for creation in [
        pendulum_creation(false),
        linear_test_creation(Vec3::Z, false),
        mixed_linear_creation(1, true),
    ] {
        let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
        gpu.enable_async_readback();
        let mut poses = creation
            .compounds
            .iter()
            .enumerate()
            .map(|(body, compound)| crate::GpuTransform {
                position: (compound.root_translation + Vec3::Y * 5.0)
                    .extend(0.0)
                    .to_array(),
                // Deliberately violate joint alignment during a prescribed animation.
                rotation: bevy_math::Quat::from_rotation_y(if body == 0 { 0.7 } else { -1.0 })
                    .to_array(),
            })
            .collect::<Vec<_>>();
        let holds = vec![true; poses.len()];
        gpu.set_body_holds(&queue, &holds).unwrap();
        gpu.prescribe_held_poses(&queue, &poses).unwrap();
        let mut drives = creation
            .coordinate_drives
            .iter()
            .copied()
            .map(crate::GpuMechanismDrive::from)
            .collect::<Vec<_>>();
        for drive in &mut drives {
            drive.mode = crate::DRIVE_MODE_SPEED;
            drive.target_speed = 10.0;
            drive.max_speed = 10.0;
            drive.max_acceleration = 1000.0;
        }
        gpu.write_mechanism_drives(&queue, &drives).unwrap();
        for tick in 1..=8 {
            for (body, pose) in poses.iter().enumerate() {
                gpu.apply_impulse(
                    &device,
                    &queue,
                    u32::try_from(body).unwrap(),
                    transform_position(*pose) + Vec3::X,
                    Vec3::new(1.0e5, -1.0e5, 1.0e5),
                )
                .unwrap();
            }
            gpu.dispatch_tick(&device, &queue, tick);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let state = gpu.poll_tick_readback(&device).unwrap().unwrap();
            assert_eq!(state.diagnostics.error_flags, 0);
            assert_eq!(state.transforms, poses);
            assert!(
                state
                    .velocities
                    .iter()
                    .all(|v| v.linear == [0.0; 4] && v.angular == [0.0; 4])
            );
            assert!(
                state
                    .coordinates
                    .iter()
                    .all(|q| q.position == 0.0 && q.velocity == 0.0)
            );
        }
        // Free-fall release assertions below apply to the ungrounded fixtures.
        if creation.compounds.iter().any(|body| body.is_static) {
            continue;
        }
        for (pose, compound) in poses.iter_mut().zip(&creation.compounds) {
            pose.rotation = compound.root_rotation.to_array();
        }
        gpu.prescribe_held_poses(&queue, &poses).unwrap();
        gpu.write_mechanism_drives(
            &queue,
            &vec![crate::GpuMechanismDrive::PASSIVE; drives.len()],
        )
        .unwrap();
        gpu.set_body_holds(&queue, &vec![false; poses.len()])
            .unwrap();
        gpu.dispatch_tick(&device, &queue, 9);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let released = gpu.poll_tick_readback(&device).unwrap().unwrap();
        assert_eq!(released.diagnostics.error_flags, 0);
        assert!(
            released
                .velocities
                .iter()
                .all(|v| v.linear[1] < 0.0 && v.linear[1] > -0.2)
        );
    }
}

#[test]
pub(super) fn held_body_supports_a_falling_neighbor_without_moving() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let mut graph = ConstructionGraph::new();
    for y in [8, 16] {
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4; 3],
                    BuildPose::new(IVec3::new(0, y, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
    }
    let creation = graph.compile().unwrap();
    let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
    gpu.enable_async_readback();
    gpu.set_body_holds(&queue, &[true, false]).unwrap();
    let mut final_state = None;
    for tick in 1..=120 {
        gpu.dispatch_tick(&device, &queue, tick);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let state = gpu.poll_tick_readback(&device).unwrap().unwrap();
        assert_eq!(state.diagnostics.error_flags, 0);
        assert!((state.transforms[0].position[1] - 2.0).abs() < 1.0e-6);
        final_state = Some(state);
    }
    let y = final_state.unwrap().transforms[1].position[1];
    assert!(
        (y - 3.0).abs() < 0.03,
        "neighbor must land on held body: {y}"
    );
}

#[test]
pub(super) fn releasing_a_loaded_hold_discards_only_changed_contact_warmstarts() {
    let (device, queue) = test_device().expect("hold release regression requires an adapter");
    let mut graph = ConstructionGraph::new();
    for x in [0, 16] {
        for y in [8, 12] {
            graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4; 3],
                        BuildPose::new(IVec3::new(x, y, 0), GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap();
        }
    }
    let creation = graph.compile().unwrap();
    let config = GpuPhysicsConfig {
        solver_iterations: 1,
        ground_plane_enabled: false,
        ..Default::default()
    };
    let released = GpuPhysics::new_with_config(&device, &queue, &creation, config).unwrap();
    let unchanged = GpuPhysics::new_with_config(&device, &queue, &creation, config).unwrap();
    for gpu in [&released, &unchanged] {
        gpu.enable_async_readback();
        gpu.set_body_holds(&queue, &[true, false, true, false])
            .unwrap();
    }
    let mut last_state = None;
    for tick in 1..=60 {
        for gpu in [&released, &unchanged] {
            gpu.dispatch_tick(&device, &queue, tick);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let state = gpu.poll_tick_readback(&device).unwrap().unwrap();
            assert_eq!(state.diagnostics.error_flags, 0);
            last_state = Some(state);
        }
    }
    let state = last_state.unwrap();
    let cold = GpuPhysics::new_with_config(&device, &queue, &creation, config).unwrap();
    cold.enable_async_readback();
    cold.set_body_holds(&queue, &[false, false, true, false])
        .unwrap();
    cold.write_body_states(&queue, &state.transforms, &state.velocities)
        .unwrap();
    released
        .set_body_holds(&queue, &[false, false, true, false])
        .unwrap();
    let mut results = Vec::new();
    for gpu in [&released, &unchanged, &cold] {
        gpu.dispatch_tick(&device, &queue, 61);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let state = gpu.poll_tick_readback(&device).unwrap().unwrap();
        assert_eq!(state.diagnostics.error_flags, 0);
        results.push(state);
    }
    // The released pair must behave like the same state with an empty cache.
    // The unrelated pair must retain its prior cache and match the control.
    for (bodies, reference) in [(0..2, 2), (2..4, 1)] {
        for body in bodies {
            let actual = results[0].velocities[body];
            let expected = results[reference].velocities[body];
            for (actual, expected) in actual
                .linear
                .into_iter()
                .chain(actual.angular)
                .zip(expected.linear.into_iter().chain(expected.angular))
            {
                assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
            }
        }
    }
}
