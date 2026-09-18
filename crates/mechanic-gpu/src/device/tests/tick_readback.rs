//! Asynchronous and authoritative tick readback.

use super::*;

#[test]
pub(super) fn late_mapping_callbacks_cannot_reorder_publication() {
    assert_eq!(
        super::super::readback::oldest_completed_readback([(2, 7, 1), (0, 8, 0)].into_iter()),
        None
    );
    assert_eq!(
        super::super::readback::oldest_completed_readback([(2, 7, 0), (0, 8, 0)].into_iter()),
        Some(2)
    );
}

#[test]
pub(super) fn submission_sequence_and_execution_evidence_survive_scheduler_gaps() {
    let (device, queue) = test_device().expect("execution evidence requires a real GPU adapter");
    let creation = pendulum_creation(false);
    let gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        &creation,
        GpuPhysicsConfig {
            collisions_enabled: false,
            ..Default::default()
        },
    )
    .unwrap();
    gpu.enable_async_readback();
    for (sequence, tick) in [(1, 4), (2, 22)] {
        let submission = gpu.dispatch_tick(&device, &queue, tick);
        assert_eq!(submission.submission_sequence, sequence);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let completed = gpu.poll_tick_readback(&device).unwrap().unwrap();
        assert_eq!(completed.submission_sequence, sequence);
        assert_eq!(completed.tick_index, tick);
        assert_eq!(completed.diagnostics.error_flags, 0);
        let evidence = completed.diagnostics.execution;
        assert_eq!(
            evidence.integrated_bodies as usize,
            creation.compounds.len()
        );
        assert_eq!(evidence.published_bodies as usize, creation.compounds.len());
        assert_eq!(
            evidence.validated_bearings as usize,
            creation.bearings.len()
        );
        assert_eq!(evidence.stage_mask, 1 | 2 | 32 | 64);
    }
    queue.write_buffer(
        &gpu.diagnostics,
        0,
        bytemuck::bytes_of(&crate::INVALID_NUMERIC_FLAG),
    );
    gpu.dispatch_tick(&device, &queue, 23);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let completed = gpu.poll_tick_readback(&device).unwrap().unwrap();
    assert_ne!(completed.diagnostics.error_flags, 0);
    assert_eq!(completed.diagnostics.execution.integrated_bodies, 0);
    assert_eq!(completed.diagnostics.execution.published_bodies, 0);
}

#[test]
pub(super) fn callback_timing_distinguishes_servicing_from_later_consumption() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let creation = pendulum_creation(false);
    let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
    gpu.enable_async_readback();
    gpu.enable_readback_timing();
    gpu.dispatch_tick(&device, &queue, 1);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let serviced = std::time::Instant::now();
    std::thread::sleep(std::time::Duration::from_millis(5));
    let readback = gpu.poll_tick_readback(&device).unwrap().unwrap();
    let callback = readback.callbacks_completed_at.unwrap();
    assert!(callback <= serviced);
    assert_eq!(readback.callbacks_during_poll, Some(false));
    assert!(callback.elapsed() >= std::time::Duration::from_millis(5));
    assert!(readback.submission_to_callbacks_ms.unwrap() <= readback.submission_to_readback_ms);
    assert_eq!(readback.diagnostics.error_flags, 0);
    assert_eq!(
        gpu.async_readback_slots_available(),
        ASYNC_READBACK_RING_SIZE
    );
}

#[test]
pub(super) fn asynchronous_tick_readback_is_monotonic_and_tick_matched() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let creation = pendulum_creation(false);
    let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
    gpu.enable_async_readback();
    let ring = u64::try_from(ASYNC_READBACK_RING_SIZE).unwrap();
    for tick in 1..=ring {
        let started = std::time::Instant::now();
        let submission = gpu.dispatch_tick(&device, &queue, tick);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let timings = submission.cpu_timings;
        let stages = [
            timings.encoding_ms,
            timings.finalization_ms,
            timings.submission_ms,
            timings.readback_setup_ms,
        ];
        assert!(
            stages
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
        assert!(stages.iter().sum::<f64>() <= elapsed_ms);
        assert_eq!(submission.tick_index, tick);
    }
    // A full ring is the app's submission budget: the backlog waits on the
    // CPU rather than growing an unbounded GPU queue.
    assert_eq!(gpu.async_readback_slots_available(), 0);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let mut completed = Vec::new();
    while let Some(readback) = gpu.poll_tick_readback(&device).unwrap() {
        assert_eq!(readback.callbacks_completed_at, None);
        assert_eq!(readback.submission_to_callbacks_ms, None);
        assert_eq!(readback.callbacks_during_poll, None);
        assert_eq!(readback.snapshot_slot, (readback.tick_index % 3) as u8);
        assert_eq!(readback.transforms.len(), creation.compounds.len());
        completed.push(readback.tick_index);
    }
    assert_eq!(
        gpu.async_readback_slots_available(),
        ASYNC_READBACK_RING_SIZE
    );
    gpu.dispatch_tick(&device, &queue, ring + 1);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let readback = gpu.poll_tick_readback(&device).unwrap().unwrap();
    completed.push(readback.tick_index);
    assert_eq!(
        completed,
        (1..=ring + 1).collect::<Vec<_>>(),
        "every submitted tick is read back once, in order"
    );
}

pub(super) fn copy_state_rows<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    count: u32,
) -> Vec<T> {
    let size = u64::from(count) * size_of::<T>() as u64;
    let destination = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("independent state verification"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(source, 0, &destination, 0, size);
    queue.submit([encoder.finish()]);
    super::super::readback::map_for_read(device, &destination).unwrap();
    let rows = super::super::readback::mapped_rows(&destination, count);
    destination.unmap();
    rows
}

#[test]
pub(super) fn authoritative_readback_keeps_velocities_and_coordinates_with_their_tick() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    for creation in [
        pendulum_creation(false),
        linear_test_creation(Vec3::Z, true),
    ] {
        let gpu = GpuPhysics::new_with_config(
            &device,
            &queue,
            &creation,
            GpuPhysicsConfig {
                collisions_enabled: false,
                ..Default::default()
            },
        )
        .unwrap();
        gpu.enable_async_readback();
        gpu.initialize_mechanism_coordinates(
            &queue,
            &[crate::GpuMechanismCoordinate {
                position: 0.1,
                velocity: 0.3,
            }],
        )
        .unwrap();
        let mut expected = Vec::new();
        for tick in 1..=3 {
            gpu.dispatch_tick(&device, &queue, tick);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            // Independently copy authoritative buffers now; delayed async
            // consumption must still return these values after later ticks.
            expected.push((
                copy_state_rows::<[f32; 4]>(
                    &device,
                    &queue,
                    &gpu.linear_velocities,
                    gpu.body_count,
                ),
                copy_state_rows::<[f32; 4]>(
                    &device,
                    &queue,
                    &gpu.angular_velocities,
                    gpu.body_count,
                ),
                copy_state_rows::<crate::GpuMechanismCoordinate>(
                    &device,
                    &queue,
                    &gpu.mechanism.coordinates,
                    gpu.mechanism.coordinate_count,
                ),
            ));
        }
        for (index, (linear, angular, coordinates)) in expected.into_iter().enumerate() {
            let state = gpu.poll_tick_readback(&device).unwrap().unwrap();
            assert_eq!(state.tick_index, index as u64 + 1);
            assert_eq!(state.diagnostics.error_flags, 0);
            assert_eq!(state.coordinates, coordinates);
            assert_eq!(
                state
                    .velocities
                    .iter()
                    .map(|v| v.linear)
                    .collect::<Vec<_>>(),
                linear
            );
            assert_eq!(
                state
                    .velocities
                    .iter()
                    .map(|v| v.angular)
                    .collect::<Vec<_>>(),
                angular
            );
            assert!(
                state
                    .coordinates
                    .iter()
                    .all(|q| q.position.is_finite() && q.velocity.is_finite())
            );
        }
    }
}

#[test]
pub(super) fn restored_authoritative_state_continues_rotational_and_linear_motion() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    for creation in [
        pendulum_creation(false),
        linear_test_creation(Vec3::Z, false),
    ] {
        let config = GpuPhysicsConfig {
            collisions_enabled: false,
            ..Default::default()
        };
        let original = GpuPhysics::new_with_config(&device, &queue, &creation, config).unwrap();
        original.enable_async_readback();
        original
            .initialize_mechanism_coordinates(
                &queue,
                &[crate::GpuMechanismCoordinate {
                    position: 0.1,
                    velocity: 0.3,
                }],
            )
            .unwrap();
        original.dispatch_tick(&device, &queue, 1);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let saved = original.poll_tick_readback(&device).unwrap().unwrap();
        assert_eq!(saved.diagnostics.error_flags, 0);
        let replacement = GpuPhysics::new_with_config(&device, &queue, &creation, config).unwrap();
        replacement.enable_async_readback();
        replacement
            .write_body_states(&queue, &saved.transforms, &saved.velocities)
            .unwrap();
        replacement
            .initialize_mechanism_coordinates(&queue, &saved.coordinates)
            .unwrap();
        original.dispatch_tick(&device, &queue, 2);
        replacement.dispatch_tick(&device, &queue, 2);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let expected = original.poll_tick_readback(&device).unwrap().unwrap();
        let actual = replacement.poll_tick_readback(&device).unwrap().unwrap();
        assert_eq!(actual.diagnostics.error_flags, 0);
        for (actual, expected) in actual.transforms.iter().zip(&expected.transforms) {
            for (a, b) in actual
                .position
                .iter()
                .chain(&actual.rotation)
                .zip(expected.position.iter().chain(&expected.rotation))
            {
                assert!((a - b).abs() < 1.0e-5, "restored pose diverged: {a} != {b}");
            }
        }
        for (actual, expected) in actual.velocities.iter().zip(&expected.velocities) {
            for (a, b) in actual
                .linear
                .iter()
                .chain(&actual.angular)
                .zip(expected.linear.iter().chain(&expected.angular))
            {
                assert!(
                    (a - b).abs() < 1.0e-4,
                    "restored velocity diverged: {a} != {b}"
                );
            }
        }
        for (actual, expected) in actual.coordinates.iter().zip(&expected.coordinates) {
            assert!((actual.position - expected.position).abs() < 1.0e-5);
            assert!((actual.velocity - expected.velocity).abs() < 1.0e-4);
        }
    }
}

#[test]
pub(super) fn authoritative_readback_supports_scenes_without_joints() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [1; 3],
                BuildPose::new(IVec3::new(0, 80, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap();
    let creation = graph.compile().unwrap();
    let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
    gpu.enable_async_readback();
    for tick in 1..=3 {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mut previous_velocity = 0.0;
    for tick in 1..=3 {
        let state = gpu.poll_tick_readback(&device).unwrap().unwrap();
        assert_eq!(state.tick_index, tick);
        assert_eq!(state.diagnostics.error_flags, 0);
        assert!(state.coordinates.is_empty());
        assert_eq!(state.velocities.len(), 1);
        let velocity = state.velocities[0].linear[1];
        assert!(
            velocity < previous_velocity,
            "gravity must accelerate on each captured tick"
        );
        previous_velocity = velocity;
    }
}
