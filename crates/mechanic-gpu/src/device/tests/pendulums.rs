//! Passive bearings: pendulums, hinges, and motion carried through joints.

use super::*;

pub(super) fn pendulum_creation(grounded: bool) -> mechanic_core::CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let mut spawn = |units| {
        let spec =
            CuboidSpec::new([4, 4, 4], BuildPose::new(units, GridRotation::default())).unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        part
    };
    let root = spawn(IVec3::new(0, 2, 0));
    let arm_a = spawn(IVec3::new(4, 2, 0));
    let arm_b = spawn(IVec3::new(4, 2, 4));
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(arm_a, FaceKind::PositiveZ),
            second: FaceRef::part(arm_b, FaceKind::NegativeZ),
        }))
        .unwrap();
    if grounded {
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(root, FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();
    }
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(root, FaceKind::PositiveX),
            FaceRef::part(arm_a, FaceKind::NegativeX),
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::X,
        )))
        .unwrap();
    graph.compile().unwrap()
}

pub(super) fn tall_pendulum_creation(second_link: bool) -> mechanic_core::CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let (support, arm, child) = {
        let mut spawn = |units| {
            let spec =
                CuboidSpec::new([2, 2, 2], BuildPose::new(units, GridRotation::default())).unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        };
        let support = (0..7)
            .map(|row| spawn(IVec3::new(0, row * 2 + 1, 0)))
            .collect::<Vec<_>>();
        let arm = (0..4)
            .map(|column| spawn(IVec3::new(2, 13, column * 2)))
            .collect::<Vec<_>>();
        let child = second_link.then(|| {
            (0..3)
                .map(|row| spawn(IVec3::new(2, row * 2 + 13, 8)))
                .collect::<Vec<_>>()
        });
        (support, arm, child)
    };
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(support[0], FaceKind::NegativeY),
            second: FaceRef::ground(),
        }))
        .unwrap();
    for pair in support.windows(2) {
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(pair[0], FaceKind::PositiveY),
                second: FaceRef::part(pair[1], FaceKind::NegativeY),
            }))
            .unwrap();
    }
    for pair in arm.windows(2) {
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(pair[0], FaceKind::PositiveZ),
                second: FaceRef::part(pair[1], FaceKind::NegativeZ),
            }))
            .unwrap();
    }
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(support[6], FaceKind::PositiveX),
            FaceRef::part(arm[0], FaceKind::NegativeX),
            Vec3::new(0.25, 3.25, 0.0),
            Vec3::X,
        )))
        .unwrap();
    if let Some(child) = child {
        for pair in child.windows(2) {
            graph
                .apply(BuildCommand::Weld(WeldSpec {
                    first: FaceRef::part(pair[0], FaceKind::PositiveY),
                    second: FaceRef::part(pair[1], FaceKind::NegativeY),
                }))
                .unwrap();
        }
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(arm[3], FaceKind::PositiveZ),
                FaceRef::part(child[0], FaceKind::NegativeZ),
                Vec3::new(0.5, 3.25, 1.75),
                Vec3::Z,
            )))
            .unwrap();
    }
    graph.compile().unwrap()
}

pub(super) fn branching_pendulum_creation(
    arm_count: usize,
    hanging: bool,
) -> mechanic_core::CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let root = CuboidSpec::new(
        [8, 4, 8],
        BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(root) = graph.apply(BuildCommand::Spawn(root)).unwrap() else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(root, FaceKind::NegativeY),
            second: FaceRef::ground(),
        }))
        .unwrap();
    let bar = CuboidSpec::new(
        [24, 2, 2],
        BuildPose::from_half_grid(IVec3::new(0, 10, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(bar) = graph.apply(BuildCommand::Spawn(bar)).unwrap() else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(root, FaceKind::PositiveY),
            FaceRef::part(bar, FaceKind::NegativeY),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::Y,
        )))
        .unwrap();

    for (center, bar_face, arm_face, anchor, axis) in [
        (
            IVec3::new(26, if hanging { 7 } else { 16 }, 1),
            FaceKind::PositiveX,
            FaceKind::NegativeX,
            Vec3::new(3.0, 1.25, 0.0),
            Vec3::X,
        ),
        (
            IVec3::new(-26, if hanging { 7 } else { 16 }, -1),
            FaceKind::NegativeX,
            FaceKind::PositiveX,
            Vec3::new(-3.0, 1.25, 0.0),
            Vec3::NEG_X,
        ),
    ]
    .into_iter()
    .take(arm_count)
    {
        let arm = CuboidSpec::new(
            [2, 8, 2],
            BuildPose::from_half_grid(center, GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(arm) = graph.apply(BuildCommand::Spawn(arm)).unwrap() else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(bar, bar_face),
                FaceRef::part(arm, arm_face),
                anchor,
                axis,
            )))
            .unwrap();
    }
    graph.compile().unwrap()
}

pub(super) fn swinging_arm_with_coaxial_rotor_creation() -> mechanic_core::CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let root = CuboidSpec::new(
        [8, 8, 8],
        BuildPose::new(IVec3::new(0, 4, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(root) = graph.apply(BuildCommand::Spawn(root)).unwrap() else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(root, FaceKind::NegativeY),
            second: FaceRef::ground(),
        }))
        .unwrap();

    let arm = CuboidSpec::new(
        [2, 2, 24],
        BuildPose::from_half_grid(IVec3::new(10, 10, 24), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(arm) = graph.apply(BuildCommand::Spawn(arm)).unwrap() else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(root, FaceKind::PositiveX),
            FaceRef::part(arm, FaceKind::NegativeX),
            Vec3::new(1.0, 1.25, 0.25),
            Vec3::X,
        )))
        .unwrap();

    let rotor = CylinderSpec::new(
        CylinderDimensions::new(0.25, 0.20, 0.75).unwrap(),
        BuildPose::from_half_grid(IVec3::new(10, 5, 46), GridRotation::default()),
    );
    let BuildOutcome::Spawned(rotor) = graph.apply(BuildCommand::SpawnCylinder(rotor)).unwrap()
    else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(arm, FaceKind::NegativeY),
            FaceRef::part(rotor, FaceKind::PositiveY),
            Vec3::new(1.25, 1.0, 5.75),
            Vec3::NEG_Y,
        )))
        .unwrap();
    graph.compile().unwrap()
}

pub(super) fn relative_bearing_rotation(
    snapshot: &[crate::GpuTransform],
    bearing: &mechanic_core::CompiledBearing,
) -> bevy_math::Quat {
    let a = bevy_math::Quat::from_array(snapshot[bearing.compound_a as usize].rotation);
    let b = bevy_math::Quat::from_array(snapshot[bearing.compound_b as usize].rotation);
    a.conjugate() * b
}

pub(super) fn run_long_pendulum(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    creation: &mechanic_core::CompiledCreation,
    collisions_enabled: bool,
) -> (Vec<f32>, Vec<f32>, super::super::GpuTickReadback) {
    let gpu = GpuPhysics::new_with_config(
        device,
        queue,
        creation,
        GpuPhysicsConfig {
            collisions_enabled,
            ..Default::default()
        },
    )
    .unwrap();
    for tick in 1..=1_200 {
        gpu.dispatch_tick(device, queue, tick);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let current = gpu.read_snapshot_transforms(device, queue, 0).unwrap();
    let previous = gpu.read_snapshot_transforms(device, queue, 2).unwrap();
    let diagnostics = gpu.read_last_tick(device).unwrap();
    let mut angles = Vec::with_capacity(creation.bearings.len());
    let mut angular_speeds = Vec::with_capacity(creation.bearings.len());
    for bearing in &creation.bearings {
        let current_relative = relative_bearing_rotation(&current, bearing);
        let previous_relative = relative_bearing_rotation(&previous, bearing);
        angles.push(
            2.0 * current_relative
                .xyz()
                .dot(bearing.local_axis_a)
                .atan2(current_relative.w),
        );
        let delta = previous_relative.conjugate() * current_relative;
        angular_speeds.push(2.0 * delta.xyz().length().atan2(delta.w.abs()) * 60.0);
    }
    (angles, angular_speeds, diagnostics)
}

#[test]
pub(super) fn tall_single_and_double_pendulums_dissipate_energy() {
    let Some((device, queue)) = test_device() else {
        return;
    };

    let single = tall_pendulum_creation(false);
    let assert_compacted_topology =
        |creation: &mechanic_core::CompiledCreation, part_count, body_count| {
            assert_eq!(creation.part_to_compound.len(), part_count);
            assert_eq!(creation.compounds.len(), body_count);
            assert_eq!(creation.bearings.len(), body_count - 1);
            assert_eq!(creation.colliders.len(), body_count);
            for compound in &creation.compounds {
                assert_eq!(compound.collider_range.len(), 1);
                let collider = &creation.colliders[compound.collider_range.start as usize];
                let mechanic_core::ColliderShape::Cuboid {
                    local_rotation,
                    half_extents,
                } = collider.shape
                else {
                    panic!("compacted pendulum link must remain a cuboid");
                };
                assert_eq!(local_rotation, bevy_math::Quat::IDENTITY);
                let (minimum, maximum) = match compound.source_parts.len() {
                    7 => (Vec3::new(-0.25, 0.0, -0.25), Vec3::new(0.25, 3.5, 0.25)),
                    4 => (Vec3::new(0.25, 3.0, -0.25), Vec3::new(0.75, 3.5, 1.75)),
                    3 => (Vec3::new(0.25, 3.0, 1.75), Vec3::new(0.75, 4.5, 2.25)),
                    count => panic!("unexpected pendulum link with {count} parts"),
                };
                let center = compound.root_translation + collider.local_center;
                assert!((center - half_extents).abs_diff_eq(minimum, 1.0e-6));
                assert!((center + half_extents).abs_diff_eq(maximum, 1.0e-6));
                assert_eq!(compound.is_static, compound.source_parts.len() == 7);
            }
        };
    assert_compacted_topology(&single, 11, 2);
    let (single_angles, single_speeds, single_diagnostics) =
        run_long_pendulum(&device, &queue, &single, true);
    assert_eq!(single_diagnostics.error_flags, 0);
    assert_eq!(single_diagnostics.active_contact_count, 0);
    assert!(
        (single_angles[0] - std::f32::consts::FRAC_PI_2).abs() < 0.02,
        "single pendulum angle was {}",
        single_angles[0]
    );
    assert!(single_speeds[0] < 0.2);

    let double = tall_pendulum_creation(true);
    assert_compacted_topology(&double, 14, 3);
    let (free_angles, free_speeds, free_diagnostics) =
        run_long_pendulum(&device, &queue, &double, false);
    assert_eq!(free_diagnostics.error_flags, 0);
    assert!(
        (free_angles[0] - std::f32::consts::FRAC_PI_2).abs() < 0.02,
        "double pendulum retained free angles {free_angles:?} and speeds {free_speeds:?}"
    );
    assert!(
        (free_angles[1] + std::f32::consts::FRAC_PI_2).abs() < 0.02,
        "double pendulum retained free angles {free_angles:?} and speeds {free_speeds:?}"
    );
    assert!(
        free_speeds.iter().all(|speed| *speed < 1.0e-4),
        "double pendulum retained free speeds {free_speeds:?}"
    );

    let (_, contact_speeds, contact_diagnostics) =
        run_long_pendulum(&device, &queue, &double, true);
    assert_eq!(contact_diagnostics.error_flags, 0);
    assert!(
        contact_speeds.iter().all(|speed| *speed < 1.0e-4),
        "double pendulum retained contact speeds {contact_speeds:?}"
    );
}

#[test]
pub(super) fn branching_pendulum_reaches_rest_instead_of_retaining_spin() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    for (arm_count, hanging) in [(1, false), (2, false), (1, true), (2, true)] {
        let creation = branching_pendulum_creation(arm_count, hanging);
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
        if arm_count == 1 && hanging {
            gpu.initialize_mechanism_coordinates(
                &queue,
                &[
                    GpuMechanismCoordinate {
                        position: 0.0,
                        velocity: 0.0,
                    },
                    GpuMechanismCoordinate {
                        position: 0.35,
                        velocity: 0.0,
                    },
                ],
            )
            .unwrap();
        }
        let sample = |tick: u64| {
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let current = gpu
                .read_snapshot_transforms(&device, &queue, u8::try_from(tick % 3).unwrap())
                .unwrap();
            let previous = gpu
                .read_snapshot_transforms(&device, &queue, u8::try_from((tick - 1) % 3).unwrap())
                .unwrap();
            creation
                .bearings
                .iter()
                .map(|bearing| {
                    let current_relative = relative_bearing_rotation(&current, bearing);
                    let previous_relative = relative_bearing_rotation(&previous, bearing);
                    let delta = previous_relative.conjugate() * current_relative;
                    2.0 * delta.xyz().length().atan2(delta.w.abs()) * 60.0
                })
                .collect::<Vec<_>>()
        };
        for tick in 1..=1_200 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        let early = sample(1_200);
        for tick in 1_201..=2_400 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        let late = sample(2_400);
        assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
        assert!(
            late.iter().all(|speed| *speed < 1.0e-4)
                && late
                    .iter()
                    .zip(&early)
                    .all(|(late, early)| *late <= *early + 1.0e-5),
            "{arm_count}-arm hanging={hanging} bearing speeds grew from {early:?} to {late:?}"
        );
    }
}

#[test]
pub(super) fn moving_coaxial_rotor_does_not_gain_perpetual_spin() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let creation = swinging_arm_with_coaxial_rotor_creation();
    let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
    gpu.initialize_mechanism_coordinates(
        &queue,
        &[
            GpuMechanismCoordinate {
                position: 0.35,
                velocity: 0.0,
            },
            GpuMechanismCoordinate {
                position: 0.0,
                velocity: 0.0,
            },
        ],
    )
    .unwrap();
    let sample = |tick: u64| {
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let current = gpu
            .read_snapshot_transforms(&device, &queue, u8::try_from(tick % 3).unwrap())
            .unwrap();
        let previous = gpu
            .read_snapshot_transforms(&device, &queue, u8::try_from((tick - 1) % 3).unwrap())
            .unwrap();
        creation
            .bearings
            .iter()
            .map(|bearing| {
                let current_relative = relative_bearing_rotation(&current, bearing);
                let previous_relative = relative_bearing_rotation(&previous, bearing);
                let delta = previous_relative.conjugate() * current_relative;
                2.0 * delta.xyz().length().atan2(delta.w.abs()) * 60.0
            })
            .collect::<Vec<_>>()
    };
    for tick in 1..=1_200 {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    let early_speeds = sample(1_200);
    for tick in 1_201..=2_400 {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    let late_speeds = sample(2_400);
    let diagnostics = gpu.read_last_tick(&device).unwrap();
    assert_eq!(diagnostics.error_flags, 0, "{diagnostics:?}");
    assert!(
        late_speeds.iter().all(|speed| *speed < 0.02)
            && late_speeds
                .iter()
                .zip(&early_speeds)
                .all(|(late, early)| *late <= *early + 1.0e-5),
        "bearing state gained speed from {early_speeds:?} to {late_speeds:?}"
    );
}

#[test]
pub(super) fn grounded_offset_pendulum_swings_without_detaching() {
    let creation = pendulum_creation(true);
    let Some((snapshot, readback)) = run_ticks(&creation, 30, false) else {
        return;
    };
    assert_eq!(readback.error_flags, 0);
    assert!(snapshot[1].rotation[0].abs() > 1.0e-3);
    let root = Vec3::from_array(snapshot[0].position[..3].try_into().unwrap());
    assert!(root.abs_diff_eq(creation.compounds[0].root_translation, 1.0e-5));
}

#[test]
pub(super) fn freely_falling_hinge_has_no_gravity_induced_relative_rotation() {
    let creation = pendulum_creation(false);
    let Some((snapshot, readback)) = run_ticks(&creation, 30, false) else {
        return;
    };
    assert_eq!(readback.error_flags, 0);
    assert!(snapshot[0].position[1] < creation.compounds[0].root_translation.y - 0.5);
    assert!(snapshot[1].rotation[0].abs() < 1.0e-4);
    assert!(snapshot[1].rotation[1].abs() < 1.0e-4);
    assert!(snapshot[1].rotation[2].abs() < 1.0e-4);
}

#[test]
pub(super) fn contact_supported_unwelded_tower_drives_welded_arm() {
    let mut graph = ConstructionGraph::new();
    let mut parts = Vec::new();
    for y in [2, 6, 10, 14] {
        let spec = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, y, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        parts.push(part);
    }
    for z in [0, 4, 8] {
        let spec = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(4, 14, z), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        parts.push(part);
    }
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(parts[4], FaceKind::PositiveZ),
            second: FaceRef::part(parts[5], FaceKind::NegativeZ),
        }))
        .unwrap();
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(parts[5], FaceKind::PositiveZ),
            second: FaceRef::part(parts[6], FaceKind::NegativeZ),
        }))
        .unwrap();
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(parts[3], FaceKind::PositiveX),
            FaceRef::part(parts[4], FaceKind::NegativeX),
            Vec3::new(0.5, 3.5, 0.0),
            Vec3::X,
        )))
        .unwrap();
    let creation = graph.compile().unwrap();
    let Some((snapshot, readback)) = run_ticks(&creation, 90, true) else {
        return;
    };
    assert_eq!(readback.error_flags, 0);
    let bearing = creation.bearings[0];
    let body_a = bearing.compound_a as usize;
    let body_b = bearing.compound_b as usize;
    let rotation_a = bevy_math::Quat::from_array(snapshot[body_a].rotation);
    let rotation_b = bevy_math::Quat::from_array(snapshot[body_b].rotation);
    let anchor_a = Vec3::from_array(snapshot[body_a].position[..3].try_into().unwrap())
        + rotation_a * bearing.local_anchor_a;
    let anchor_b = Vec3::from_array(snapshot[body_b].position[..3].try_into().unwrap())
        + rotation_b * bearing.local_anchor_b;
    assert!(anchor_a.abs_diff_eq(anchor_b, 1.0e-5));
    assert!(snapshot[body_b].rotation[0].abs() > 1.0e-3);
    assert!(
        snapshot[body_a].position[1] > 0.5,
        "contact-supported body fell to {} m",
        snapshot[body_a].position[1]
    );
}

#[test]
pub(super) fn double_pendulum_transfers_motion_through_both_bearings() {
    let mut graph = ConstructionGraph::new();
    let mut spawned = Vec::new();
    for units in [
        IVec3::new(0, 2, 0),
        IVec3::new(4, 2, 0),
        IVec3::new(4, 2, 4),
        IVec3::new(8, 2, 0),
        IVec3::new(8, 2, 4),
    ] {
        let spec =
            CuboidSpec::new([4, 4, 4], BuildPose::new(units, GridRotation::default())).unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        spawned.push(part);
    }
    for (a, b) in [(1, 2), (3, 4)] {
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(spawned[a], FaceKind::PositiveZ),
                second: FaceRef::part(spawned[b], FaceKind::NegativeZ),
            }))
            .unwrap();
    }
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(spawned[0], FaceKind::NegativeY),
            second: FaceRef::ground(),
        }))
        .unwrap();
    for (a, b, x) in [(0, 1, 0.5), (1, 3, 1.5)] {
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(spawned[a], FaceKind::PositiveX),
                FaceRef::part(spawned[b], FaceKind::NegativeX),
                Vec3::new(x, 0.5, 0.0),
                Vec3::X,
            )))
            .unwrap();
    }
    let creation = graph.compile().unwrap();
    let Some((snapshot, readback)) = run_ticks(&creation, 30, false) else {
        return;
    };
    assert_eq!(readback.error_flags, 0);
    let first = bevy_math::Quat::from_array(snapshot[1].rotation);
    let second = bevy_math::Quat::from_array(snapshot[2].rotation);
    assert!(first.x.abs() > 1.0e-3);
    assert!((first.conjugate() * second).x.abs() > 1.0e-4);
}

#[test]
pub(super) fn balanced_child_contacts_move_root_without_spurious_joint_motion() {
    let mut graph = ConstructionGraph::new();
    let mut spawned = Vec::new();
    for units in [
        IVec3::new(0, 6, 0),
        IVec3::new(4, 6, 0),
        IVec3::new(4, 2, 0),
        IVec3::new(4, 2, 4),
    ] {
        let spec =
            CuboidSpec::new([4, 4, 4], BuildPose::new(units, GridRotation::default())).unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        spawned.push(part);
    }
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(spawned[1], FaceKind::NegativeY),
            second: FaceRef::part(spawned[2], FaceKind::PositiveY),
        }))
        .unwrap();
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(spawned[2], FaceKind::PositiveZ),
            second: FaceRef::part(spawned[3], FaceKind::NegativeZ),
        }))
        .unwrap();
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(spawned[0], FaceKind::PositiveX),
            FaceRef::part(spawned[1], FaceKind::NegativeX),
            Vec3::new(0.5, 1.5, 0.0),
            Vec3::X,
        )))
        .unwrap();
    let creation = graph.compile().unwrap();
    let Some((free, _)) = run_ticks(&creation, 8, false) else {
        return;
    };
    let Some((contact, readback)) = run_ticks(&creation, 8, true) else {
        return;
    };
    assert_eq!(readback.error_flags, 0);
    let root = creation.bearings[0].compound_a as usize;
    let child = creation.bearings[0].compound_b as usize;
    assert!(contact[root].position[1] > free[root].position[1] + 1.0e-4);
    let root_rotation = bevy_math::Quat::from_array(contact[root].rotation);
    let child_rotation = bevy_math::Quat::from_array(contact[child].rotation);
    assert!((root_rotation.conjugate() * child_rotation).x.abs() < 1.0e-4);
}
