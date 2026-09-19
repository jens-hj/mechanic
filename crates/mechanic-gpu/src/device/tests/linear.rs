//! Linear bearings: stops, locked axes, drives, and mixed loops.

use super::*;

pub(super) fn mixed_linear_creation(
    copies: i32,
    close_loop: bool,
) -> mechanic_core::CompiledCreation {
    let mut graph = ConstructionGraph::new();
    for copy in 0..copies {
        let z_ticks = copy * 1600;
        let mut spawn = |dimensions, x, y| {
            spawned_part(
                graph
                    .apply(BuildCommand::Spawn(
                        CuboidSpec::new(
                            dimensions,
                            BuildPose::from_position_ticks(
                                IVec3::new(x, y, z_ticks),
                                GridRotation::default(),
                            ),
                        )
                        .unwrap(),
                    ))
                    .unwrap(),
            )
        };
        let base = spawn([4, 4, 4], 0, 200);
        let carriage = spawn([1, 1, 1], 290, 200);
        let arm = spawn([1, 1, 1], 390, 200);
        let support = close_loop.then(|| spawn([1, 1, 1], 530, 200));
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(base, FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();
        let z = f32::from(i16::try_from(copy).unwrap()) * 4.0;
        let linear_kind = |normal, length| {
            mechanic_core::BearingKind::Linear(mechanic_core::LinearBearing {
                dimensions: mechanic_core::LinearBearingDimensions::new(length, 0.1).unwrap(),
                mount_normal: normal,
                face: mechanic_core::CarriageFace::Top,
            })
        };
        graph
            .apply(BuildCommand::AddBearing(
                BearingSpec::new(
                    FaceRef::part(base, FaceKind::PositiveX),
                    FaceRef::part(carriage, FaceKind::NegativeX),
                    Vec3::new(0.5, 0.5, z),
                    Vec3::Z,
                )
                .with_kind(linear_kind(Vec3::X, 1.0)),
            ))
            .unwrap();
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(carriage, FaceKind::PositiveX),
                FaceRef::part(arm, FaceKind::NegativeX),
                Vec3::new(0.85, 0.5, z),
                Vec3::X,
            )))
            .unwrap();
        if let Some(support) = support {
            graph
                .apply(BuildCommand::RigidLink(RigidLinkSpec {
                    first: base,
                    second: support,
                }))
                .unwrap();
            graph
                .apply(BuildCommand::AddBearing(
                    BearingSpec::new(
                        FaceRef::part(support, FaceKind::NegativeX),
                        FaceRef::part(arm, FaceKind::PositiveX),
                        Vec3::new(1.2, 0.5, z),
                        Vec3::Z,
                    )
                    .with_kind(linear_kind(Vec3::NEG_X, 0.5)),
                ))
                .unwrap();
        }
    }
    graph.compile().unwrap()
}

pub(super) fn assert_mixed_linear_constraints(
    gpu: &GpuPhysics,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    creation: &mechanic_core::CompiledCreation,
    tick: u64,
) -> Vec<crate::GpuTransform> {
    let snapshot = gpu
        .read_snapshot_transforms(device, queue, u8::try_from(tick % 3).unwrap())
        .unwrap();
    for (index, bearing) in creation.bearings.iter().enumerate() {
        let a = snapshot[bearing.compound_a as usize];
        let b = snapshot[bearing.compound_b as usize];
        let qa = bevy_math::Quat::from_array(a.rotation);
        let qb = bevy_math::Quat::from_array(b.rotation);
        let delta = transform_position(b) + qb * bearing.local_anchor_b
            - transform_position(a)
            - qa * bearing.local_anchor_a;
        let axis = qa * bearing.local_axis_a;
        let (position_error, rotation_error) = match bearing.kind {
            mechanic_core::BearingKind::Linear(_)
            | mechanic_core::BearingKind::Suspension(_)
            | mechanic_core::BearingKind::Piston(_) => {
                let displacement = delta.dot(axis);
                let [lower, upper] = bearing.kind.bounds();
                let residual = delta - axis * displacement.clamp(lower, upper);
                let rotation = qa.conjugate() * qb;
                (
                    residual.length(),
                    2.0 * rotation.xyz().length().atan2(rotation.w.abs()),
                )
            }
            mechanic_core::BearingKind::Rotational => {
                let other_axis = qb * bearing.local_axis_b;
                (
                    delta.length(),
                    axis.cross(other_axis).length().atan2(axis.dot(other_axis)),
                )
            }
        };
        assert!(
            position_error <= mechanic_core::ANCHOR_TOLERANCE_METERS,
            "tick {tick}, bearing {index} {:?}: position error {position_error}",
            bearing.kind
        );
        assert!(
            rotation_error.to_degrees() <= mechanic_core::AXIS_TOLERANCE_DEGREES,
            "tick {tick}, bearing {index} {:?}: rotation error {} degrees",
            bearing.kind,
            rotation_error.to_degrees()
        );
    }
    let diagnostics = gpu.read_last_tick(device).unwrap();
    assert_eq!(diagnostics.error_flags, 0, "tick {tick}: {diagnostics:?}");
    assert!(
        diagnostics.anchor_residual_meters <= mechanic_core::ANCHOR_TOLERANCE_METERS,
        "{diagnostics:?}"
    );
    assert!(
        diagnostics.axis_residual_degrees <= mechanic_core::AXIS_TOLERANCE_DEGREES,
        "{diagnostics:?}"
    );
    snapshot
}

#[test]
pub(super) fn linear_mixed_chains_preserve_constraints_in_small_and_parallel_routes() {
    let (device, queue) = test_device().expect("linear GPU regression requires an adapter");
    for copies in [1, 33] {
        let mut creation = mixed_linear_creation(copies, false);
        assert_eq!(creation.loop_topology.closure_bearings.len(), 0);
        assert_eq!(
            uses_fused_velocity_schedule(
                u32::try_from(creation.bearings.len()).unwrap(),
                u32::try_from(creation.compounds.len()).unwrap()
            ),
            copies == 1
        );
        for bearing in &creation.bearings {
            let coordinate = bearing.coordinate_index.unwrap() as usize;
            creation.coordinate_drives[coordinate] = linear_speed_drive(0.5);
        }
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
        for tick in 1..=90 {
            gpu.dispatch_tick(&device, &queue, tick);
            if tick == 1 || tick % 10 == 0 {
                assert_mixed_linear_constraints(&gpu, &device, &queue, &creation, tick);
            }
        }
        let (position, snapshot) = linear_test_snapshot(&gpu, &device, &queue, &creation, 90);
        assert!(
            (position - 0.425).abs() <= mechanic_core::ANCHOR_TOLERANCE_METERS,
            "{position}"
        );
        let relative = relative_bearing_rotation(&snapshot, &creation.bearings[1]);
        assert!(
            relative.xyz().length() > 0.1,
            "rotational child failed to move: {relative:?}"
        );
    }
}

#[test]
pub(super) fn linear_mixed_loops_enforce_narrower_closure_stops_in_all_routes() {
    let (device, queue) = test_device().expect("linear GPU regression requires an adapter");
    for copies in [1, 22] {
        let mut creation = mixed_linear_creation(copies, true);
        assert_eq!(
            creation.loop_topology.closure_bearings.len(),
            usize::try_from(copies).unwrap()
        );
        assert!(
            creation
                .bearings
                .iter()
                .filter(|b| b.coordinate_index.is_none())
                .all(|b| matches!(b.kind, mechanic_core::BearingKind::Linear(_)))
        );
        assert_eq!(
            uses_fused_velocity_schedule(
                u32::try_from(creation.bearings.len()).unwrap(),
                u32::try_from(creation.compounds.len()).unwrap()
            ),
            copies == 1
        );
        for bearing in &creation.bearings {
            if bearing.kind.is_translational()
                && let Some(coordinate) = bearing.coordinate_index
            {
                creation.coordinate_drives[coordinate as usize] = linear_speed_drive(0.5);
            }
        }
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
        for tick in 1..=90 {
            gpu.dispatch_tick(&device, &queue, tick);
            if tick == 1 || tick % 5 == 0 {
                assert_mixed_linear_constraints(&gpu, &device, &queue, &creation, tick);
            }
        }
        let (position, _) = linear_test_snapshot(&gpu, &device, &queue, &creation, 90);
        assert!(
            (position - 0.175).abs() <= mechanic_core::ANCHOR_TOLERANCE_METERS,
            "linear closure's narrower end stop was not enforced: {position}"
        );
    }
}

#[test]
pub(super) fn linear_mixed_loops_lock_orientation_after_off_centre_impacts() {
    let (device, queue) = test_device().expect("linear GPU regression requires an adapter");
    for copies in [1, 22] {
        let creation = mixed_linear_creation(copies, true);
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
        for bearing in creation
            .bearings
            .iter()
            .filter(|bearing| matches!(bearing.kind, mechanic_core::BearingKind::Rotational))
        {
            let child = &creation.compounds[bearing.compound_b as usize];
            gpu.apply_impulse(
                &device,
                &queue,
                bearing.compound_b,
                child.root_translation + Vec3::Y * 0.1,
                Vec3::Z * child.mass_properties.mass * 0.1,
            )
            .unwrap();
        }
        for tick in 1..=30 {
            gpu.dispatch_tick(&device, &queue, tick);
            assert_mixed_linear_constraints(&gpu, &device, &queue, &creation, tick);
        }
    }
}

pub(super) fn linear_test_creation(axis: Vec3, grounded: bool) -> mechanic_core::CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let base = spawned_part(
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 4, 4],
                    BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap(),
    );
    let carriage = spawned_part(
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_position_ticks(
                        IVec3::new(290, 200, 0),
                        GridRotation::default(),
                    ),
                )
                .unwrap(),
            ))
            .unwrap(),
    );
    if grounded {
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(base, FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();
    }
    graph
        .apply(BuildCommand::AddBearing(
            BearingSpec::new(
                FaceRef::part(base, FaceKind::PositiveX),
                FaceRef::part(carriage, FaceKind::NegativeX),
                Vec3::new(0.5, 0.5, 0.0),
                axis,
            )
            .with_kind(mechanic_core::BearingKind::Linear(
                mechanic_core::LinearBearing {
                    dimensions: mechanic_core::LinearBearingDimensions::default(),
                    mount_normal: Vec3::X,
                    face: mechanic_core::CarriageFace::Top,
                },
            )),
        ))
        .unwrap();
    graph.compile().unwrap()
}

pub(super) fn linear_test_snapshot(
    gpu: &GpuPhysics,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    creation: &mechanic_core::CompiledCreation,
    tick: u64,
) -> (f32, Vec<crate::GpuTransform>) {
    let snapshot = gpu
        .read_snapshot_transforms(device, queue, u8::try_from(tick % 3).unwrap())
        .unwrap();
    let bearing = creation.bearings[0];
    let a = snapshot[bearing.compound_a as usize];
    let b = snapshot[bearing.compound_b as usize];
    let rotation_a = bevy_math::Quat::from_array(a.rotation);
    let rotation_b = bevy_math::Quat::from_array(b.rotation);
    let separation = transform_position(b) + rotation_b * bearing.local_anchor_b
        - transform_position(a)
        - rotation_a * bearing.local_anchor_a;
    let axis = rotation_a * bearing.local_axis_a;
    let displacement = separation.dot(axis);
    let sideways = (separation - axis * displacement).length();
    assert!(
        sideways <= mechanic_core::ANCHOR_TOLERANCE_METERS,
        "tick {tick}: transverse displacement {sideways} m"
    );
    let relative_rotation = rotation_a.conjugate() * rotation_b;
    let orientation_error = (2.0
        * relative_rotation
            .xyz()
            .length()
            .atan2(relative_rotation.w.abs()))
    .to_degrees();
    assert!(
        orientation_error <= mechanic_core::AXIS_TOLERANCE_DEGREES,
        "tick {tick}: relative rotation {orientation_error} degrees"
    );
    let [minimum, maximum] = bearing.kind.bounds();
    assert!(
        displacement >= minimum - mechanic_core::ANCHOR_TOLERANCE_METERS
            && displacement <= maximum + mechanic_core::ANCHOR_TOLERANCE_METERS,
        "tick {tick}: displacement {displacement} outside [{minimum}, {maximum}]"
    );
    let diagnostics = gpu.read_last_tick(device).unwrap();
    assert_eq!(diagnostics.error_flags, 0, "tick {tick}: {diagnostics:?}");
    assert!(
        diagnostics.anchor_residual_meters <= mechanic_core::ANCHOR_TOLERANCE_METERS,
        "{diagnostics:?}"
    );
    assert!(
        diagnostics.axis_residual_degrees <= mechanic_core::AXIS_TOLERANCE_DEGREES,
        "{diagnostics:?}"
    );
    (displacement, snapshot)
}

#[test]
pub(super) fn linear_passive_vertical_carriage_hits_both_physical_stops_without_power() {
    let (device, queue) = test_device().expect("linear GPU regression requires an adapter");
    for axis in [Vec3::Y, Vec3::NEG_Y] {
        let creation = linear_test_creation(axis, true);
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
        let mut last = 0.0;
        for tick in 1..=120 {
            gpu.dispatch_tick(&device, &queue, tick);
            if tick % 10 == 0 {
                let (position, _) = linear_test_snapshot(&gpu, &device, &queue, &creation, tick);
                assert!(
                    (position - last) * axis.y <= mechanic_core::ANCHOR_TOLERANCE_METERS,
                    "passive stop rebounded from {last} to {position}"
                );
                last = position;
            }
        }
        assert!(
            (last + axis.y * 0.425).abs() <= mechanic_core::ANCHOR_TOLERANCE_METERS,
            "gravity did not reach the end stop: {last}"
        );
    }
}

#[test]
pub(super) fn linear_off_centre_impact_locks_sideways_motion_and_all_rotation() {
    let (device, queue) = test_device().expect("linear GPU regression requires an adapter");
    let creation = linear_test_creation(Vec3::Z, true);
    let bearing = creation.bearings[0];
    let body = &creation.compounds[bearing.compound_b as usize];
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
    gpu.apply_impulse(
        &device,
        &queue,
        bearing.compound_b,
        body.root_translation + Vec3::Y * 0.1,
        Vec3::new(3.0, 2.0, 8.0) * body.mass_properties.mass,
    )
    .unwrap();
    for tick in 1..=90 {
        gpu.dispatch_tick(&device, &queue, tick);
        linear_test_snapshot(&gpu, &device, &queue, &creation, tick);
    }
    let (position, _) = linear_test_snapshot(&gpu, &device, &queue, &creation, 90);
    assert!(
        (position - 0.425).abs() <= mechanic_core::ANCHOR_TOLERANCE_METERS,
        "impact failed to reach the stop: {position}"
    );
}

#[test]
pub(super) fn linear_floating_mount_receives_end_stop_reaction_after_impact() {
    let (device, queue) = test_device().expect("linear GPU regression requires an adapter");
    for sign in [-1.0, 1.0] {
        let creation = linear_test_creation(Vec3::Z, false);
        let bearing = creation.bearings[0];
        let child = &creation.compounds[bearing.compound_b as usize];
        let base = &creation.compounds[bearing.compound_a as usize];
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
        gpu.apply_impulse(
            &device,
            &queue,
            bearing.compound_b,
            child.root_translation,
            Vec3::Z * sign * 20.0 * child.mass_properties.mass,
        )
        .unwrap();
        for tick in 1..=60 {
            gpu.dispatch_tick(&device, &queue, tick);
            linear_test_snapshot(&gpu, &device, &queue, &creation, tick);
        }
        let (position, snapshot) = linear_test_snapshot(&gpu, &device, &queue, &creation, 60);
        assert!(
            (position - sign * 0.425).abs() <= mechanic_core::ANCHOR_TOLERANCE_METERS,
            "floating impact did not remain at stop: {position}"
        );
        let movement =
            transform_position(snapshot[bearing.compound_a as usize]) - base.root_translation;
        assert!(
            movement.z * sign > 0.01,
            "mount received no stop reaction: {movement:?}"
        );
    }
}

pub(super) fn linear_speed_drive(speed: f32) -> CoordinateDrive {
    CoordinateDrive {
        mode: DriveMode::Speed,
        target_speed: speed,
        target_angle: 0.0,
        max_speed: 2.0,
        max_acceleration: 100.0,
        source_a_max_acceleration: 100.0,
        source_a_no_load_speed: 2.0,
        source_b_max_acceleration: 0.0,
        source_b_no_load_speed: 0.0,
        min_angle: f32::NEG_INFINITY,
        max_angle: f32::INFINITY,
    }
}

#[test]
pub(super) fn linear_position_drive_seeks_and_holds_against_gravity_after_reprogramming() {
    let (device, queue) = test_device().expect("linear GPU regression requires an adapter");
    let mut creation = linear_test_creation(Vec3::Y, true);
    let position_drive = |target| CoordinateDrive {
        mode: DriveMode::Angle,
        target_angle: target,
        min_angle: -0.25,
        max_angle: 0.25,
        ..linear_speed_drive(0.0)
    };
    creation.coordinate_drives[0] = position_drive(0.15);
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
    for (phase, target) in [(0, 0.15), (1, -0.20)] {
        gpu.write_mechanism_drives(
            &queue,
            &[crate::GpuMechanismDrive::from(position_drive(target))],
        )
        .unwrap();
        for offset in 1..=180 {
            let tick = phase * 180 + offset;
            gpu.dispatch_tick(&device, &queue, tick);
            let (position, _) = linear_test_snapshot(&gpu, &device, &queue, &creation, tick);
            if offset > 120 {
                assert!(
                    (position - target).abs() < 0.001,
                    "position {position}, target {target}"
                );
            }
        }
    }
}

#[test]
pub(super) fn linear_sustained_power_respects_stops_and_reverses_immediately() {
    let (device, queue) = test_device().expect("linear GPU regression requires an adapter");
    let mut creation = linear_test_creation(Vec3::Z, true);
    creation.coordinate_drives[0] = linear_speed_drive(1.0);
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
    for tick in 1..=120 {
        gpu.dispatch_tick(&device, &queue, tick);
        linear_test_snapshot(&gpu, &device, &queue, &creation, tick);
    }
    let (upper, _) = linear_test_snapshot(&gpu, &device, &queue, &creation, 120);
    assert!(
        (upper - 0.425).abs() <= mechanic_core::ANCHOR_TOLERANCE_METERS,
        "{upper}"
    );
    gpu.write_mechanism_drives(
        &queue,
        &[crate::GpuMechanismDrive::from(linear_speed_drive(-1.0))],
    )
    .unwrap();
    gpu.dispatch_tick(&device, &queue, 121);
    let (reversed, _) = linear_test_snapshot(&gpu, &device, &queue, &creation, 121);
    assert!(
        reversed < upper - 1.0e-4,
        "reverse remained stuck: {upper} -> {reversed}"
    );
    for tick in 122..=240 {
        gpu.dispatch_tick(&device, &queue, tick);
        linear_test_snapshot(&gpu, &device, &queue, &creation, tick);
    }
    let (lower, _) = linear_test_snapshot(&gpu, &device, &queue, &creation, 240);
    assert!(
        (lower + 0.425).abs() <= mechanic_core::ANCHOR_TOLERANCE_METERS,
        "{lower}"
    );
    gpu.write_mechanism_drives(
        &queue,
        &[crate::GpuMechanismDrive::from(CoordinateDrive::default())],
    )
    .unwrap();
    for tick in 241..=270 {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    let (unpowered, _) = linear_test_snapshot(&gpu, &device, &queue, &creation, 270);
    assert!(
        (unpowered - lower).abs() <= mechanic_core::ANCHOR_TOLERANCE_METERS,
        "removing power moved stationary horizontal carriage: {lower} -> {unpowered}"
    );
}
