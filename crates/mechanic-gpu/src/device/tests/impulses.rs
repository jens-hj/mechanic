//! External impulses and the angular response to off-centre contact.

use super::*;

#[test]
pub(super) fn off_centre_external_impulse_changes_linear_and_angular_motion() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let mut graph = ConstructionGraph::new();
    let spec = CuboidSpec::new(
        [4, 4, 4],
        BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
        unreachable!()
    };
    let creation = graph.compile().unwrap();
    let body = creation.part_to_compound[0].1;
    let initial = creation.compounds[body as usize].root_translation;
    let properties = creation.compounds[body as usize].mass_properties;
    let arm = Vec3::Y * 0.5;
    let impulse = Vec3::X * properties.mass;
    let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
    gpu.apply_impulse(&device, &queue, body, initial + arm, impulse)
        .unwrap();
    gpu.dispatch_tick(&device, &queue, 1);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let snapshot = gpu.read_snapshot_transforms(&device, &queue, 1).unwrap();
    let expected_displacement =
        impulse.x / properties.mass * 0.999 * mechanic_core::TICK_SECONDS_F32;
    assert!(
        (snapshot[body as usize].position[0] - initial.x - expected_displacement).abs() < 1.0e-5
    );
    let angular_velocity = properties.inverse_inertia * arm.cross(impulse) * 0.98;
    let half_spin = angular_velocity * (0.5 * mechanic_core::TICK_SECONDS_F32);
    let expected_rotation =
        bevy_math::Quat::from_xyzw(half_spin.x, half_spin.y, half_spin.z, 1.0).normalize();
    assert!(
        bevy_math::Quat::from_array(snapshot[body as usize].rotation)
            .abs_diff_eq(expected_rotation, 1.0e-5)
    );
    assert_eq!(creation.part_to_compound[0].0, part);
}

#[test]
pub(super) fn external_impulse_batches_chunk_repeated_rows_and_validate_atomically() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [4, 4, 4],
                BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap();
    let creation = graph.compile().unwrap();
    let initial = creation.compounds[0].root_translation;
    let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();

    let mut invalid =
        vec![GpuExternalImpulse::new(0, initial, Vec3::X); EXTERNAL_IMPULSE_BATCH_CAPACITY + 1];
    invalid[EXTERNAL_IMPULSE_BATCH_CAPACITY].metadata[0] = 1;
    assert_eq!(
        gpu.dispatch_tick_with_impulses(&device, &queue, 1, &invalid)
            .unwrap_err(),
        GpuImpulseError::BodyIndexOutOfRange {
            body_index: 1,
            body_count: 1,
        }
    );
    gpu.dispatch_tick(&device, &queue, 1);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let unchanged = gpu.read_snapshot_transforms(&device, &queue, 1).unwrap();
    assert!((unchanged[0].position[0] - initial.x).abs() < 1.0e-6);

    let rows = vec![
        GpuExternalImpulse::new(0, initial, Vec3::X * 10.0);
        EXTERNAL_IMPULSE_BATCH_CAPACITY + 1
    ];
    gpu.dispatch_tick_with_impulses(&device, &queue, 2, &rows)
        .unwrap();
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let moved = gpu.read_snapshot_transforms(&device, &queue, 2).unwrap();
    assert!(moved[0].position[0] > initial.x);

    let non_finite = [GpuExternalImpulse::new(0, initial, Vec3::splat(f32::NAN))];
    assert_eq!(
        gpu.dispatch_tick_with_impulses(&device, &queue, 3, &non_finite)
            .unwrap_err(),
        GpuImpulseError::NonFinite
    );
    assert_eq!(
        gpu.apply_impulses(&device, &queue, &rows[..=EXTERNAL_IMPULSE_BATCH_CAPACITY],)
            .unwrap_err(),
        GpuImpulseError::BatchCapacity {
            provided: EXTERNAL_IMPULSE_BATCH_CAPACITY + 1,
            capacity: EXTERNAL_IMPULSE_BATCH_CAPACITY,
        }
    );
}

#[test]
pub(super) fn offset_ground_contact_applies_angular_impulse_about_compound_centre() {
    let mut graph = ConstructionGraph::new();
    let lower_spec = CuboidSpec::new(
        [2, 2, 2],
        BuildPose::new(IVec3::new(4, 1, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(lower) = graph.apply(BuildCommand::Spawn(lower_spec)).unwrap() else {
        unreachable!()
    };
    let upper_spec = CuboidSpec::new(
        [2, 2, 2],
        BuildPose::new(IVec3::new(-4, 9, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(upper) = graph.apply(BuildCommand::Spawn(upper_spec)).unwrap() else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec {
            first: lower,
            second: upper,
        }))
        .unwrap();
    let creation = graph.compile().unwrap();
    assert!(creation.colliders[0].local_center.x.abs() > 0.5);
    let Some((device, queue)) = test_device() else {
        return;
    };
    let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
    for tick in 1..=4 {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let snapshot = gpu.read_snapshot_transforms(&device, &queue, 1).unwrap();
    let rotation = bevy_math::Quat::from_array(snapshot[0].rotation);
    assert!(
        rotation.z > 1.0e-4,
        "offset support acted through the compound centre: {rotation:?}"
    );
}

#[test]
pub(super) fn external_impulse_drives_a_bearing_coordinate() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let creation = pendulum_creation(true);
    let child = creation.bearings[0].compound_b;
    let run = |impulse: Option<Vec3>| {
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
        if let Some(impulse) = impulse {
            gpu.apply_impulse(
                &device,
                &queue,
                child,
                creation.compounds[child as usize].root_translation,
                impulse,
            )
            .unwrap();
        }
        for tick in 1..=10 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        gpu.read_snapshot_transforms(&device, &queue, 1).unwrap()[child as usize]
    };
    let baseline = run(None);
    let struck = run(Some(Vec3::NEG_Y * 2_000.0));
    let baseline_rotation = bevy_math::Quat::from_array(baseline.rotation);
    let struck_rotation = bevy_math::Quat::from_array(struck.rotation);
    let difference = baseline_rotation.angle_between(struck_rotation);
    assert!(
        difference > 0.01,
        "external impulse changed rotation by only {difference} rad: baseline={baseline_rotation:?}, struck={struck_rotation:?}"
    );
}
