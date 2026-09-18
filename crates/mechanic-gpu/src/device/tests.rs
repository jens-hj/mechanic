use super::ASYNC_READBACK_RING_SIZE;
use bevy_math::{IVec3, Vec3};
use mechanic_core::{
    BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, ConstructionMaterial,
    CoordinateDrive, CuboidSpec, CylinderDimensions, CylinderSpec, DriveMode, EngineKind, FaceKind,
    FaceRef, GearboxConfig, GridRotation, PartId, PipeBendDimensions, PipeBendSpec, RigidLinkSpec,
    ServoSpec, WeldSpec,
};

use crate::GpuMechanismCoordinate;

use super::{
    EXTERNAL_IMPULSE_BATCH_CAPACITY, FULL_CYLINDER_GROUND_FIRST, GpuExternalImpulse,
    GpuGroundPlane, GpuGroundPlaneError, GpuImpulseError, GpuPhysics, GpuPhysicsConfig,
    GpuPhysicsPipelines, contact_pair_capacity, full_cylinder_ground_data,
    uses_fused_contact_schedule, uses_fused_velocity_schedule,
};

#[test]
fn fused_small_mechanism_schedule_has_explicit_size_and_scene_boundaries() {
    assert!(uses_fused_velocity_schedule(64, 256));
    assert!(!uses_fused_velocity_schedule(65, 256));
    assert!(!uses_fused_velocity_schedule(64, 257));

    assert!(uses_fused_contact_schedule(4, true));
    assert!(uses_fused_contact_schedule(64, true));
    assert!(!uses_fused_contact_schedule(65, true));
    assert!(uses_fused_contact_schedule(64, false));
    assert!(!uses_fused_contact_schedule(65, false));
}

#[test]
fn collision_buffers_scale_with_the_scene_up_to_the_hard_limit() {
    assert_eq!(contact_pair_capacity(0), 256);
    assert_eq!(contact_pair_capacity(1), 256);
    assert_eq!(contact_pair_capacity(16), 256);
    assert_eq!(contact_pair_capacity(1_024), 1_048_576);
    assert_eq!(
        contact_pair_capacity(crate::MAX_COLLIDERS),
        u32::try_from(crate::MAX_CONTACT_PAIRS).unwrap()
    );
}

#[test]
fn replacement_scene_reuses_every_compiled_shader_and_pipeline() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap();
    let creation = graph.compile().unwrap();
    let pipelines = GpuPhysicsPipelines::new();
    let first = GpuPhysics::new_with_pipelines(
        &device,
        &queue,
        &creation,
        GpuPhysicsConfig::default(),
        &pipelines,
    )
    .unwrap();
    let shader_count = pipelines.shaders.lock().unwrap().len();
    let pipeline_count = pipelines.pipelines.lock().unwrap().len();
    drop(first);

    let _replacement = GpuPhysics::new_with_pipelines(
        &device,
        &queue,
        &creation,
        GpuPhysicsConfig::default(),
        &pipelines,
    )
    .unwrap();

    assert!(shader_count > 0);
    assert!(pipeline_count > shader_count);
    assert_eq!(pipelines.shaders.lock().unwrap().len(), shader_count);
    assert_eq!(pipelines.pipelines.lock().unwrap().len(), pipeline_count);
}

#[test]
fn physics_wgsl_parses_and_validates_without_a_gpu() {
    for (name, source) in [
        ("physics", include_str!("../kernels/physics.wgsl")),
        ("collision", include_str!("../kernels/collision.wgsl")),
        ("lbvh", include_str!("../kernels/lbvh.wgsl")),
        ("bearings", include_str!("../kernels/bearings.wgsl")),
        ("mechanism", include_str!("../kernels/mechanism.wgsl")),
        ("articulated", include_str!("../kernels/articulated.wgsl")),
        ("closure", include_str!("../kernels/closure.wgsl")),
        ("snapshot", include_str!("../kernels/snapshot.wgsl")),
    ] {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{name} WGSL parses: {error}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{name} WGSL validates: {error:#?}"));
    }
}

#[test]
fn gated_recovery_timestamps_remain_ordered() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .unwrap();
    if !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        return;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::TIMESTAMP_QUERY,
        ..Default::default()
    }))
    .unwrap();
    let mut gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        &pendulum_creation(false),
        GpuPhysicsConfig {
            ground_plane_enabled: false,
            ..Default::default()
        },
    )
    .unwrap();
    let mut chunk = super::terrain::tests::rigid_chunk();
    chunk.origin.0.y -= 100.0;
    gpu.write_terrain_chunks(&device, &queue, [&chunk], bevy_math::DVec3::ZERO)
        .unwrap();
    for tick in 1..=30 {
        gpu.dispatch_tick(&device, &queue, tick);
        let result = gpu.read_last_tick(&device).unwrap();
        assert_eq!(result.error_flags, 0);
        assert_eq!(result.contact_count, 0);
        let stages = result.kernel_timings.unwrap();
        assert!(stages.terrain_recovery_ms > 0.0);
        assert!(
            stages.recovery_projection_ms <= stages.terrain_recovery_ms,
            "invalid nested span: {stages:?}"
        );
        assert!(stages.terrain_recovery_ms <= result.gpu_tick_ms.unwrap());
    }
}

fn pendulum_creation(grounded: bool) -> mechanic_core::CompiledCreation {
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

fn tall_pendulum_creation(second_link: bool) -> mechanic_core::CompiledCreation {
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

fn branching_pendulum_creation(arm_count: usize, hanging: bool) -> mechanic_core::CompiledCreation {
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

fn swinging_arm_with_coaxial_rotor_creation() -> mechanic_core::CompiledCreation {
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

fn relative_bearing_rotation(
    snapshot: &[crate::GpuTransform],
    bearing: &mechanic_core::CompiledBearing,
) -> bevy_math::Quat {
    let a = bevy_math::Quat::from_array(snapshot[bearing.compound_a as usize].rotation);
    let b = bevy_math::Quat::from_array(snapshot[bearing.compound_b as usize].rotation);
    a.conjugate() * b
}

fn run_long_pendulum(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    creation: &mechanic_core::CompiledCreation,
    collisions_enabled: bool,
) -> (Vec<f32>, Vec<f32>, super::GpuTickReadback) {
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
fn tall_single_and_double_pendulums_dissipate_energy() {
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
fn branching_pendulum_reaches_rest_instead_of_retaining_spin() {
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
fn moving_coaxial_rotor_does_not_gain_perpetual_spin() {
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

/// Grounded base plus a hinged arm wired to one control block.
///
/// `loaded` extends the arm sideways off the hinge axis so gravity applies a
/// real torque; otherwise the arm's centre of mass sits on the axis.
fn driven_arm(
    axis: Vec3,
    limits: mechanic_core::DriveLimits,
    program: mechanic_core::DriveProgram,
    loaded: bool,
) -> (ConstructionGraph, mechanic_core::CompiledCreation) {
    let mut graph = ConstructionGraph::new();
    let mut spawn = |units| {
        let spec =
            CuboidSpec::new([4, 4, 4], BuildPose::new(units, GridRotation::default())).unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        part
    };
    let (base_units, arm_units, source_face, target_face, anchor) = if axis == Vec3::X {
        (
            IVec3::new(0, 2, 0),
            IVec3::new(4, 2, 0),
            FaceKind::PositiveX,
            FaceKind::NegativeX,
            Vec3::new(0.5, 0.5, 0.0),
        )
    } else {
        (
            IVec3::new(0, 2, 0),
            IVec3::new(0, 6, 0),
            FaceKind::PositiveY,
            FaceKind::NegativeY,
            Vec3::new(0.0, 1.0, 0.0),
        )
    };
    let base = spawn(base_units);
    let arm = spawn(arm_units);
    let outrigger = loaded.then(|| spawn(arm_units + IVec3::new(0, 0, 4)));
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(base, FaceKind::NegativeY),
            second: FaceRef::ground(),
        }))
        .unwrap();
    if let Some(outrigger) = outrigger {
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(arm, FaceKind::PositiveZ),
                second: FaceRef::part(outrigger, FaceKind::NegativeZ),
            }))
            .unwrap();
    }
    let BuildOutcome::BearingAdded(bearing) = graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(base, source_face),
            FaceRef::part(arm, target_face),
            anchor,
            axis,
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(
            mechanic_core::ControllerSpec::new(BuildPose::new(
                IVec3::new(0, 40, 0),
                GridRotation::default(),
            )),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let mut link = mechanic_core::DriveLinkSpec::new(controller, bearing);
    link.limits = limits;
    link.program = program;
    graph.apply(BuildCommand::AddDriveLink(link)).unwrap();
    let mut creation = graph.compile().unwrap();
    creation.coordinate_drives[0] = test_coordinate_drive(
        &creation,
        limits,
        program
            .state(0)
            .expect("test program has one state")
            .target(),
    );
    (graph, creation)
}

fn test_coordinate_drive(
    creation: &mechanic_core::CompiledCreation,
    limits: mechanic_core::DriveLimits,
    target: mechanic_core::DriveTarget,
) -> CoordinateDrive {
    let axis_inertia = creation.loop_topology.coordinate_axis_inertia[0];
    let torque = limits.max_torque_newton_meters();
    let acceleration = if torque.is_infinite() {
        100.0
    } else {
        torque / axis_inertia
    };
    CoordinateDrive {
        mode: if target.angle().is_some() {
            DriveMode::Angle
        } else {
            DriveMode::Speed
        },
        target_speed: target.speed().unwrap_or(0.0),
        target_angle: target.angle().unwrap_or(0.0),
        max_speed: limits.max_speed_rad_s(),
        max_acceleration: acceleration,
        source_a_max_acceleration: acceleration,
        source_a_no_load_speed: limits.max_speed_rad_s(),
        source_b_max_acceleration: 0.0,
        source_b_no_load_speed: 0.0,
        min_angle: limits.min_angle(),
        max_angle: limits.max_angle(),
    }
}

/// Signed joint angle of the first bearing, read back from a snapshot.
fn joint_angle(
    snapshot: &[crate::GpuTransform],
    creation: &mechanic_core::CompiledCreation,
) -> f32 {
    let bearing = &creation.bearings[0];
    let delta = relative_bearing_rotation(snapshot, bearing);
    let axis = bearing.local_axis_a.normalize();
    2.0 * delta.xyz().dot(axis).atan2(delta.w)
}

fn run_driven_arm(
    creation: &mechanic_core::CompiledCreation,
    ticks: u64,
) -> Option<(f32, super::GpuTickReadback)> {
    let (device, queue) = test_device()?;
    let gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        creation,
        GpuPhysicsConfig {
            collisions_enabled: false,
            ..Default::default()
        },
    )
    .unwrap();
    for tick in 1..=ticks {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let snapshot = gpu
        .read_snapshot_transforms(&device, &queue, u8::try_from(ticks % 3).unwrap())
        .unwrap();
    let diagnostics = gpu.read_last_tick(&device).unwrap();
    Some((joint_angle(&snapshot, creation), diagnostics))
}

/// Limits with the given maximum torque and no travel stops.
fn limits(max_speed: f32, torque: f32) -> mechanic_core::DriveLimits {
    mechanic_core::DriveLimits::new(max_speed, torque, None).expect("test limits are in range")
}

/// Single-state program holding one target forever.
fn holding(target: mechanic_core::DriveTarget) -> mechanic_core::DriveProgram {
    mechanic_core::DriveProgram::new(
        &[mechanic_core::DriveState::new(target).expect("test target is in range")],
        false,
    )
    .expect("a one-state program is valid")
}

#[test]
fn speed_state_advances_a_bearing_coordinate_at_its_target_speed() {
    let (_, creation) = driven_arm(
        Vec3::X,
        limits(3.0, f32::INFINITY),
        holding(mechanic_core::DriveTarget::Speed(1.0)),
        false,
    );
    let Some((angle, diagnostics)) = run_driven_arm(&creation, 60) else {
        return;
    };

    assert_eq!(diagnostics.error_flags, 0);
    assert!(
        (angle - 1.0).abs() < 0.05,
        "one second at 1 rad/s should reach about 1 rad, got {angle}"
    );
}

#[test]
fn negative_target_speed_drives_the_joint_the_other_way() {
    let (_, creation) = driven_arm(
        Vec3::X,
        limits(3.0, f32::INFINITY),
        holding(mechanic_core::DriveTarget::Speed(-1.0)),
        false,
    );
    let Some((angle, diagnostics)) = run_driven_arm(&creation, 60) else {
        return;
    };

    assert_eq!(diagnostics.error_flags, 0);
    assert!(
        (angle + 1.0).abs() < 0.05,
        "a negative target speed should reach about -1 rad, got {angle}"
    );
}

#[test]
fn max_speed_caps_a_faster_state_target() {
    let (_, creation) = driven_arm(
        Vec3::X,
        limits(0.5, f32::INFINITY),
        holding(mechanic_core::DriveTarget::Speed(3.0)),
        false,
    );
    let Some((angle, diagnostics)) = run_driven_arm(&creation, 60) else {
        return;
    };

    assert_eq!(diagnostics.error_flags, 0);
    assert!(
        (angle - 0.5).abs() < 0.05,
        "the row's 0.5 rad/s ceiling should hold the joint to 0.5 rad, got {angle}"
    );
}

#[test]
fn angle_state_reaches_its_target_and_holds_without_overshooting() {
    let (_, creation) = driven_arm(
        Vec3::X,
        limits(3.0, 400.0),
        holding(mechanic_core::DriveTarget::Angle(0.8)),
        false,
    );
    // Long enough that a servo which overshoots would be caught swinging
    // back through the target rather than sitting on it.
    let Some((angle, diagnostics)) = run_driven_arm(&creation, 240) else {
        return;
    };

    assert_eq!(diagnostics.error_flags, 0);
    assert!(
        (angle - 0.8).abs() < 0.02,
        "the joint should settle on its 0.8 rad target, got {angle}"
    );
}

#[test]
fn weak_drive_stalls_lifting_a_gravity_loaded_arm() {
    // The outrigger hangs off the hinge axis, so gravity applies a positive
    // torque about it. Driving negative means the motor must lift that load,
    // which is the only direction in which stalling is observable.
    let program = holding(mechanic_core::DriveTarget::Speed(-1.0));
    let (_, strong_creation) = driven_arm(Vec3::X, limits(3.0, f32::INFINITY), program, true);
    let (_, weak_creation) = driven_arm(Vec3::X, limits(3.0, 0.5), program, true);

    let Some((strong_angle, strong_diagnostics)) = run_driven_arm(&strong_creation, 60) else {
        return;
    };
    let (weak_angle, weak_diagnostics) = run_driven_arm(&weak_creation, 60).unwrap();

    assert_eq!(strong_diagnostics.error_flags, 0);
    assert_eq!(weak_diagnostics.error_flags, 0);
    assert!(
        (strong_angle + 1.0).abs() < 0.05,
        "an unlimited motor holds its target under load, got {strong_angle}"
    );
    assert!(
        weak_angle > strong_angle + 0.5,
        "a 0.5 N·m motor should stall well short of {strong_angle}, got {weak_angle}"
    );
}

#[test]
fn driven_coordinate_stops_and_holds_at_its_travel_limit() {
    let stopped = mechanic_core::DriveLimits::new(3.0, f32::INFINITY, Some((-0.2, 0.2)))
        .expect("test limits are in range");
    let (_, creation) = driven_arm(
        Vec3::X,
        stopped,
        holding(mechanic_core::DriveTarget::Speed(1.0)),
        false,
    );
    let Some((angle, diagnostics)) = run_driven_arm(&creation, 120) else {
        return;
    };

    assert_eq!(diagnostics.error_flags, 0);
    assert!(
        (angle - 0.2).abs() < 0.02,
        "the joint should hold at its 0.2 rad limit, got {angle}"
    );
}

#[test]
fn vertical_axis_drive_is_not_zeroed_by_the_sleep_clamp() {
    // Below GRAVITY_ALIGNED_BEARING_SLEEP_SPEED, which stops passive joints.
    let (_, creation) = driven_arm(
        Vec3::Y,
        limits(3.0, f32::INFINITY),
        holding(mechanic_core::DriveTarget::Speed(0.002)),
        false,
    );
    let Some((angle, diagnostics)) = run_driven_arm(&creation, 600) else {
        return;
    };

    assert_eq!(diagnostics.error_flags, 0);
    assert!(
        angle > 0.015,
        "ten seconds at 0.002 rad/s should still turn the joint, got {angle}"
    );
}

#[test]
fn reprogramming_a_wire_changes_the_drive_without_reloading_the_scene() {
    let (mut graph, creation) = driven_arm(
        Vec3::X,
        limits(3.0, f32::INFINITY),
        holding(mechanic_core::DriveTarget::Speed(1.0)),
        false,
    );
    let Some((device, queue)) = test_device() else {
        return;
    };
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

    assert_eq!(
        gpu.write_mechanism_drives(&queue, &[]),
        Err(super::GpuPhysicsError::DriveStateCount {
            provided: 0,
            required: 1,
        })
    );

    let (link, drive_spec) = graph
        .drive_links()
        .map(|(id, spec)| (id, *spec))
        .next()
        .unwrap();
    graph
        .apply(BuildCommand::SetDriveLink {
            link,
            limits: limits(3.0, f32::INFINITY),
            program: holding(mechanic_core::DriveTarget::Speed(-2.0)),
            name: mechanic_core::DriveName::EMPTY,
            actuator: drive_spec.actuator,
        })
        .unwrap();
    let target = graph
        .drive_links()
        .next()
        .and_then(|(_, spec)| spec.resolved_target(0))
        .unwrap();
    let rows = [test_coordinate_drive(
        &creation,
        limits(3.0, f32::INFINITY),
        target,
    )]
    .into_iter()
    .map(crate::GpuMechanismDrive::from)
    .collect::<Vec<_>>();
    gpu.write_mechanism_drives(&queue, &rows).unwrap();

    for tick in 1..=60 {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let snapshot = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
    assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
    let angle = joint_angle(&snapshot, &creation);
    assert!(
        (angle + 2.0).abs() < 0.1,
        "the live reprogram should drive -2 rad/s, got {angle}"
    );
}

fn mixed_linear_creation(copies: i32, close_loop: bool) -> mechanic_core::CompiledCreation {
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

fn assert_mixed_linear_constraints(
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
            mechanic_core::BearingKind::Linear(_) | mechanic_core::BearingKind::Suspension(_) => {
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
fn linear_mixed_chains_preserve_constraints_in_small_and_parallel_routes() {
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
fn linear_mixed_loops_enforce_narrower_closure_stops_in_all_routes() {
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
fn linear_mixed_loops_lock_orientation_after_off_centre_impacts() {
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

fn linear_test_creation(axis: Vec3, grounded: bool) -> mechanic_core::CompiledCreation {
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

fn linear_test_snapshot(
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
fn linear_passive_vertical_carriage_hits_both_physical_stops_without_power() {
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
fn linear_off_centre_impact_locks_sideways_motion_and_all_rotation() {
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
fn linear_floating_mount_receives_end_stop_reaction_after_impact() {
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

fn linear_speed_drive(speed: f32) -> CoordinateDrive {
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
fn linear_position_drive_seeks_and_holds_against_gravity_after_reprogramming() {
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
fn linear_sustained_power_respects_stops_and_reverses_immediately() {
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

fn suspension_test_creation(
    spec: mechanic_core::SuspensionSpec,
    vertical: bool,
    copies: i32,
) -> mechanic_core::CompiledCreation {
    suspension_test_creation_with_anchor(spec, vertical, copies, true)
}

fn suspension_test_creation_with_anchor(
    spec: mechanic_core::SuspensionSpec,
    vertical: bool,
    copies: i32,
    grounded: bool,
) -> mechanic_core::CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let axis = if vertical { Vec3::Y } else { Vec3::X };
    let source_face = if vertical {
        FaceKind::PositiveY
    } else {
        FaceKind::PositiveX
    };
    let target_face = if vertical {
        FaceKind::NegativeY
    } else {
        FaceKind::NegativeX
    };
    for copy in 0..copies {
        let base_ticks = IVec3::new(0, 200, copy * 1600);
        #[allow(clippy::cast_possible_truncation)]
        let spacing_ticks = ((spec.initial_length() + 0.625) / 0.0025).round() as i32;
        let offset = if vertical { IVec3::Y } else { IVec3::X } * spacing_ticks;
        let mut spawn = |ticks, dimensions| {
            spawned_part(
                graph
                    .apply(BuildCommand::Spawn(
                        CuboidSpec::new(
                            dimensions,
                            BuildPose::from_position_ticks(ticks, GridRotation::default()),
                        )
                        .unwrap(),
                    ))
                    .unwrap(),
            )
        };
        let base = spawn(base_ticks, [4, 4, 4]);
        let tip = spawn(base_ticks + offset, [1, 1, 1]);
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
                    FaceRef::part(base, source_face),
                    FaceRef::part(tip, target_face),
                    base_ticks.as_vec3() * 0.0025 + axis * 0.5,
                    axis,
                )
                .with_kind(mechanic_core::BearingKind::Suspension(spec)),
            ))
            .unwrap();
    }
    graph.compile().unwrap()
}

fn suspension_test_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    creation: &mechanic_core::CompiledCreation,
) -> GpuPhysics {
    GpuPhysics::new_with_config(
        device,
        queue,
        creation,
        GpuPhysicsConfig {
            collisions_enabled: false,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn suspension_loaded_spring_equilibrium_in_small_and_parallel_routes() {
    use mechanic_core::{ShockSpec, SpringSpec, SuspensionSpec};
    let (device, queue) = test_device().expect("suspension regression requires a GPU");
    let spring = SpringSpec::default();
    let spec = SuspensionSpec::new(Some(spring), Some(ShockSpec::default()), None).unwrap();
    for copies in [1, 65] {
        let creation = suspension_test_creation(spec, true, copies);
        assert_eq!(
            uses_fused_velocity_schedule(
                u32::try_from(creation.bearings.len()).unwrap(),
                u32::try_from(creation.compounds.len()).unwrap()
            ),
            copies == 1
        );
        let bearing = creation.bearings[0];
        let expected = creation.compounds[bearing.compound_b as usize]
            .mass_properties
            .mass
            * mechanic_core::STANDARD_GRAVITY_M_S2_F32
            / spring.rate();
        let gpu = suspension_test_gpu(&device, &queue, &creation);
        for tick in 1..=360 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        let (displacement, _) = linear_test_snapshot(&gpu, &device, &queue, &creation, 360);
        assert!(
            (-displacement - expected).abs() < expected * 0.1 + 0.0001,
            "copies {copies}: compression {}, equilibrium {expected}",
            -displacement
        );
        assert_mixed_linear_constraints(&gpu, &device, &queue, &creation, 360);
        eprintln!(
            "suspension copies {copies}: {:?}",
            gpu.read_last_tick(&device).unwrap()
        );
    }
}

#[test]
fn suspension_damper_starting_stroke_is_force_free_and_rebound_is_stronger() {
    use mechanic_core::{ShockBodyEnd, ShockSpec, SuspensionSpec};
    let (device, queue) = test_device().expect("suspension regression requires a GPU");
    let spec = SuspensionSpec::new(
        None,
        Some(ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.075, 1.0, 1.6).unwrap()),
        None,
    )
    .unwrap();
    let creation = suspension_test_creation(spec, false, 1);
    let mut distances = Vec::new();
    for speed in [0.0_f32, -0.1, 0.1] {
        let gpu = suspension_test_gpu(&device, &queue, &creation);
        let bearing = creation.bearings[0];
        let body = &creation.compounds[bearing.compound_b as usize];
        if speed != 0.0 {
            gpu.apply_impulse(
                &device,
                &queue,
                bearing.compound_b,
                body.root_translation,
                Vec3::X * speed * body.mass_properties.mass,
            )
            .unwrap();
        }
        for tick in 1..=60 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        let (position, _) = linear_test_snapshot(&gpu, &device, &queue, &creation, 60);
        distances.push(position.abs());
        if speed == 0.0 {
            assert!(
                position.abs() < 0.00001,
                "stationary damper moved {position}"
            );
        } else {
            assert!(
                position * speed > 0.0
                    && position.abs() < speed.abs() * 60.0 * mechanic_core::TICK_SECONDS_F32 * 0.9,
                "damper failed to decay: speed {speed}, travel {position}"
            );
        }
    }
    assert!(
        distances[2] < distances[1] * 0.9,
        "compression/rebound travel {distances:?}"
    );
}

#[test]
fn suspension_spring_oscillates_and_preload_reduces_loaded_compression() {
    use mechanic_core::{SpringSpec, SuspensionSpec};
    let (device, queue) = test_device().expect("suspension regression requires a GPU");
    let mut compressions = Vec::new();
    for preload in [0.0, 0.025] {
        let spring = SpringSpec::new(0.5, 0.16, 0.12, 6, preload).unwrap();
        let spec = SuspensionSpec::new(Some(spring), None, None).unwrap();
        let creation = suspension_test_creation(spec, true, 1);
        let gpu = suspension_test_gpu(&device, &queue, &creation);
        let mut samples = Vec::new();
        for tick in 1..=60 {
            gpu.dispatch_tick(&device, &queue, tick);
            samples.push(-linear_test_snapshot(&gpu, &device, &queue, &creation, tick).0);
        }
        if preload == 0.0 {
            assert!(samples.windows(2).any(|s| s[1] > s[0] + 0.00001));
            assert!(
                samples.windows(2).any(|s| s[1] < s[0] - 0.00001),
                "no spring rebound: {samples:?}"
            );
        }
        compressions.push(samples.iter().copied().fold(0.0_f32, f32::max));
    }
    assert!(
        compressions[1] < compressions[0] * 0.1,
        "preload had no effect: {compressions:?}"
    );
}

#[test]
fn suspension_shock_bottoms_and_rubber_supports_load_in_both_body_orientations() {
    use mechanic_core::{BumpStopSpec, ShockBodyEnd, ShockSpec, SuspensionSpec};
    let (device, queue) = test_device().expect("suspension regression requires a GPU");
    let mut masses = Vec::new();
    for body_end in [ShockBodyEnd::Source, ShockBodyEnd::Opposite] {
        for with_stop in [false, true] {
            let shock = ShockSpec::new(0.5, 0.1, body_end, 0.0, 1.0, 1.6).unwrap();
            let stop = with_stop.then(|| BumpStopSpec::new(0.05, 0.06).unwrap());
            let spec = SuspensionSpec::new(None, Some(shock), stop).unwrap();
            let creation = suspension_test_creation(spec, true, 1);
            let bearing = creation.bearings[0];
            let mass = creation.compounds[bearing.compound_b as usize]
                .mass_properties
                .mass;
            if !with_stop {
                masses.push(mass);
            }
            let gpu = suspension_test_gpu(&device, &queue, &creation);
            for tick in 1..=360 {
                gpu.dispatch_tick(&device, &queue, tick);
            }
            let (position, _) = linear_test_snapshot(&gpu, &device, &queue, &creation, 360);
            let compression = -position;
            if with_stop {
                assert!(
                    compression > spec.bump_contact().unwrap(),
                    "rubber never contacted: {compression}"
                );
                assert!(
                    compression < spec.compression_limit().0,
                    "rubber reached crush limit under small load"
                );
                let force = spec.elastic_force(compression);
                assert!(
                    (force - mass * mechanic_core::STANDARD_GRAVITY_M_S2_F32).abs()
                        < mass * mechanic_core::STANDARD_GRAVITY_M_S2_F32 * 0.15,
                    "rubber load mismatch: force {force}, load {}",
                    mass * mechanic_core::STANDARD_GRAVITY_M_S2_F32
                );
            } else {
                assert!(
                    (compression - spec.compression_limit().0).abs() < 0.0001,
                    "shock failed to bottom: {compression}, limit {}",
                    spec.compression_limit().0
                );
            }
        }
    }
    assert!(
        masses[1] > masses[0],
        "body-end reversal did not move body mass: {masses:?}"
    );
}

#[test]
fn suspension_passive_forces_preserve_mixed_loop_closures_in_both_routes() {
    use mechanic_core::{BearingKind, ShockSpec, SpringSpec, SuspensionSpec};
    let (device, queue) = test_device().expect("suspension regression requires a GPU");
    let spec = SuspensionSpec::new(
        Some(SpringSpec::default()),
        Some(ShockSpec::default()),
        None,
    )
    .unwrap();
    for copies in [1, 22] {
        // Reuse compiled mixed-loop topology to isolate the passive solver.
        // Graph placement and suspension mass ownership have separate fixtures.
        let mut creation = mixed_linear_creation(copies, true);
        for bearing in &mut creation.bearings {
            if bearing.kind.is_translational() {
                bearing.kind = BearingKind::Suspension(spec);
            }
        }
        assert_eq!(
            creation.loop_topology.closure_bearings.len(),
            usize::try_from(copies).unwrap()
        );
        let gpu = suspension_test_gpu(&device, &queue, &creation);
        let bearing = creation.bearings[0];
        let body = &creation.compounds[bearing.compound_b as usize];
        gpu.apply_impulse(
            &device,
            &queue,
            bearing.compound_b,
            body.root_translation,
            Vec3::NEG_Z * body.mass_properties.mass,
        )
        .unwrap();
        let mut minimum = 0.0_f32;
        for tick in 1..=90 {
            gpu.dispatch_tick(&device, &queue, tick);
            if tick % 5 == 0 {
                assert_mixed_linear_constraints(&gpu, &device, &queue, &creation, tick);
                minimum =
                    minimum.min(linear_test_snapshot(&gpu, &device, &queue, &creation, tick).0);
            }
        }
        assert!(
            minimum < -0.001,
            "loop suspension failed to compress: {minimum}"
        );
    }
}

#[test]
fn suspension_floating_mount_receives_bottom_out_reaction() {
    use mechanic_core::{ShockBodyEnd, ShockSpec, SuspensionSpec};
    let (device, queue) = test_device().expect("suspension regression requires a GPU");
    let spec = SuspensionSpec::new(
        None,
        Some(ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.075, 0.0, 0.0).unwrap()),
        None,
    )
    .unwrap();
    for copies in [1, 65] {
        let creation = suspension_test_creation_with_anchor(spec, false, copies, false);
        let gpu = suspension_test_gpu(&device, &queue, &creation);
        let bearing = creation.bearings[0];
        let tip = &creation.compounds[bearing.compound_b as usize];
        let base = &creation.compounds[bearing.compound_a as usize];
        gpu.apply_impulse(
            &device,
            &queue,
            bearing.compound_b,
            tip.root_translation,
            Vec3::NEG_X * 20.0 * tip.mass_properties.mass,
        )
        .unwrap();
        for tick in 1..=15 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        let (position, snapshot) = linear_test_snapshot(&gpu, &device, &queue, &creation, 15);
        assert!(
            (position - spec.bounds()[0]).abs() < 0.0001,
            "floating shock did not bottom: {position}"
        );
        let movement =
            transform_position(snapshot[bearing.compound_a as usize]) - base.root_translation;
        assert!(
            movement.x < -0.0001,
            "floating mount received no reaction: {movement:?}"
        );
    }
}

#[test]
#[allow(clippy::float_cmp)] // Held velocities must be exactly zero.
fn suspension_holds_suppress_passive_forces_and_release_restores_spring_motion() {
    use mechanic_core::{ShockBodyEnd, ShockSpec, SpringSpec, SuspensionSpec};
    let (device, queue) = test_device().expect("suspension regression requires a GPU");
    let spec = SuspensionSpec::new(
        Some(SpringSpec::default()),
        Some(ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.05, 1.0, 1.6).unwrap()),
        None,
    )
    .unwrap();
    for copies in [1, 65] {
        let creation = suspension_test_creation_with_anchor(spec, false, copies, false);
        let gpu = suspension_test_gpu(&device, &queue, &creation);
        gpu.enable_async_readback();
        let poses = creation
            .compounds
            .iter()
            .map(|body| crate::GpuTransform {
                position: body.root_translation.extend(0.0).to_array(),
                rotation: body.root_rotation.to_array(),
            })
            .collect::<Vec<_>>();
        let mut holds = vec![true; poses.len()];
        gpu.set_body_holds(&queue, &holds).unwrap();
        gpu.prescribe_held_poses(&queue, &poses).unwrap();
        for tick in 1..=3 {
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
        }
        let bearing = creation.bearings[0];
        holds[bearing.compound_a as usize] = false;
        holds[bearing.compound_b as usize] = false;
        gpu.set_body_holds(&queue, &holds).unwrap();
        let mut last = None;
        for tick in 4..=33 {
            gpu.dispatch_tick(&device, &queue, tick);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let state = gpu.poll_tick_readback(&device).unwrap().unwrap();
            assert_eq!(state.diagnostics.error_flags, 0);
            for (index, held) in holds.iter().enumerate() {
                if *held {
                    assert_eq!(state.transforms[index], poses[index]);
                }
            }
            last = Some(state);
        }
        let state = last.unwrap();
        let base = bearing.compound_a as usize;
        assert!(
            state.transforms[base].position[0] < poses[base].position[0] - 0.00001,
            "released mount received no spring reaction"
        );
        let tip = bearing.compound_b as usize;
        assert!(
            state.transforms[tip].position[0] > poses[tip].position[0] + 0.02,
            "released spring did not extend from initial compression"
        );
    }
}

#[test]
fn suspension_rubber_crush_limit_resists_sustained_overload_in_both_routes() {
    use mechanic_core::{BumpStopSpec, ShockSpec, SuspensionSpec};
    let (device, queue) = test_device().expect("suspension regression requires a GPU");
    let spec = SuspensionSpec::new(
        None,
        Some(ShockSpec::default()),
        Some(BumpStopSpec::new(0.05, 0.06).unwrap()),
    )
    .unwrap();
    let force = spec.elastic_force(spec.compression_limit().0) * 4.0;
    for copies in [1, 65] {
        let creation = suspension_test_creation(spec, false, copies);
        let gpu = suspension_test_gpu(&device, &queue, &creation);
        let bearing = creation.bearings[0];
        let body = &creation.compounds[bearing.compound_b as usize];
        let mut contact_point = body.root_translation;
        let mut position = 0.0;
        for tick in 1..=60 {
            gpu.apply_impulse(
                &device,
                &queue,
                bearing.compound_b,
                contact_point,
                Vec3::NEG_X * force * mechanic_core::TICK_SECONDS_F32,
            )
            .unwrap();
            gpu.dispatch_tick(&device, &queue, tick);
            let (displacement, snapshot) =
                linear_test_snapshot(&gpu, &device, &queue, &creation, tick);
            position = displacement;
            contact_point = transform_position(snapshot[bearing.compound_b as usize]);
        }
        assert!(
            (-position - spec.compression_limit().0).abs() < 0.0001,
            "rubber failed to stop at 55% crush: {} vs {}",
            -position,
            spec.compression_limit().0
        );
    }
}

fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    eprintln!("GPU test adapter: {:?}", adapter.get_info());
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("mechanic articulated test device"),
        ..Default::default()
    }))
    .ok()
}

#[allow(clippy::too_many_lines)]
fn colliding_pipe_mechanism(grounded: bool) -> mechanic_core::CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let pose = |ticks, rotation| BuildPose::from_position_ticks(IVec3::from_array(ticks), rotation);
    let base = spawned_part(
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([16, 1, 16], pose([0, -50, 0], GridRotation::default())).unwrap(),
            ))
            .unwrap(),
    );
    let carriage = spawned_part(
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1, 1, 1], pose([0, 90, 0], GridRotation::default()))
                    .unwrap()
                    .with_material(ConstructionMaterial::Aluminium),
            ))
            .unwrap(),
    );
    let mut stems = Vec::new();
    for (points, bend_rotation, tip_rotation, length, material) in [
        (
            [[0, 190, 0], [0, 340, 0], [-250, 340, 0]],
            GridRotation::new(0, 0, 1),
            GridRotation::new(0, 0, 1),
            0.25,
            ConstructionMaterial::Aluminium,
        ),
        (
            [[500, 100, 450], [500, 300, 450], [750, 300, 450]],
            GridRotation::new(0, 2, 1),
            GridRotation::new(0, 0, 3),
            0.5,
            ConstructionMaterial::Steel,
        ),
    ] {
        let stem = spawned_part(
            graph
                .apply(BuildCommand::SpawnCylinder(
                    CylinderSpec::new(
                        CylinderDimensions::new(0.25, 0.0, length).unwrap(),
                        pose(points[0], GridRotation::default()),
                    )
                    .with_material(material),
                ))
                .unwrap(),
        );
        let bend = spawned_part(
            graph
                .apply(BuildCommand::SpawnPipeBend(
                    PipeBendSpec::new(
                        PipeBendDimensions::new(0.25, 0.0, 1).unwrap(),
                        pose(points[1], bend_rotation),
                    )
                    .with_material(material),
                ))
                .unwrap(),
        );
        let tip = spawned_part(
            graph
                .apply(BuildCommand::SpawnCylinder(
                    CylinderSpec::new(
                        CylinderDimensions::new(0.25, 0.0, 0.75).unwrap(),
                        pose(points[2], tip_rotation),
                    )
                    .with_material(material),
                ))
                .unwrap(),
        );
        for second in [bend, tip] {
            graph
                .apply(BuildCommand::RigidLink(RigidLinkSpec {
                    first: stem,
                    second,
                }))
                .unwrap();
        }
        stems.push(stem);
    }
    graph
        .apply(BuildCommand::AddBearing(
            BearingSpec::new(
                FaceRef::part(base, FaceKind::PositiveY),
                FaceRef::part(carriage, FaceKind::NegativeY),
                Vec3::ZERO,
                Vec3::X,
            )
            .with_kind(mechanic_core::BearingKind::Linear(
                mechanic_core::LinearBearing {
                    dimensions: mechanic_core::LinearBearingDimensions::new(1.75, 0.4).unwrap(),
                    mount_normal: Vec3::Y,
                    face: mechanic_core::CarriageFace::Top,
                },
            )),
        ))
        .unwrap();
    for (parent, child, anchor) in [
        (carriage, stems[0], Vec3::new(0.0, 0.35, 0.0)),
        (base, stems[1], Vec3::new(1.25, 0.0, 1.125)),
    ] {
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(parent, FaceKind::PositiveY),
                FaceRef::part(child, FaceKind::NegativeY),
                anchor,
                Vec3::Y,
            )))
            .unwrap();
    }
    let creation = graph
        .compile_with_static_parts(grounded.then_some(base))
        .unwrap();
    let pipe_bodies = stems
        .iter()
        .map(|part| {
            creation
                .part_to_compound
                .iter()
                .find(|(p, _)| p == part)
                .unwrap()
                .1
        })
        .collect::<Vec<_>>();
    assert_eq!(creation.loop_topology.mechanism_components.len(), 1);
    assert!(
        !creation
            .collision_suppression
            .contains(&[pipe_bodies[0], pipe_bodies[1]])
    );
    creation
}

#[test]
fn dense_pipe_contacts_on_bearings_and_a_rail_remain_bounded() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    for grounded in [true, false] {
        for rail_position in [0.4, 0.5] {
            let creation = colliding_pipe_mechanism(grounded);
            let gpu = GpuPhysics::new_with_config(
                &device,
                &queue,
                &creation,
                GpuPhysicsConfig {
                    ground_plane_enabled: false,
                    ..Default::default()
                },
            )
            .unwrap();
            gpu.dispatch_tick(&device, &queue, 1);
            let coordinates = [
                GpuMechanismCoordinate {
                    position: rail_position,
                    velocity: 0.3,
                },
                GpuMechanismCoordinate {
                    position: 3.0 * std::f32::consts::FRAC_PI_4,
                    velocity: 0.0,
                },
                GpuMechanismCoordinate {
                    position: std::f32::consts::FRAC_PI_2,
                    velocity: 0.0,
                },
            ];
            gpu.initialize_mechanism_coordinates(&queue, &coordinates)
                .unwrap();
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            gpu.encode_mechanism_forward_kinematics(&mut encoder, None, 0);
            super::direct_compute_pass(
                &mut encoder,
                "initialize pipe motion",
                &gpu.mechanism.reconstruct_velocities_pipeline,
                &gpu.mechanism.reconstruct_velocities_bind_group,
                1,
                None,
            );
            queue.submit([encoder.finish()]);
            let mut maximum_contacts = 0;
            let mut maximum_speed = 0.0_f32;
            for tick in 2..=600 {
                gpu.dispatch_tick(&device, &queue, tick);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let state = copy_state_rows::<GpuMechanismCoordinate>(
                    &device,
                    &queue,
                    &gpu.mechanism.coordinates,
                    gpu.mechanism.coordinate_count,
                );
                let diagnostics = gpu.read_last_tick(&device).unwrap();
                assert_eq!(diagnostics.error_flags, 0);
                maximum_contacts = maximum_contacts.max(diagnostics.active_contact_count);
                for coordinate in state {
                    maximum_speed = maximum_speed.max(coordinate.velocity.abs());
                    assert!(
                        coordinate.position.is_finite()
                            && coordinate.velocity.is_finite()
                            && coordinate.velocity.abs() < 3.0,
                        "grounded={grounded} rail={rail_position} tick={tick}: {coordinate:?}"
                    );
                }
            }
            eprintln!(
                "pipe contacts: grounded={grounded} rail={rail_position} max_contacts={maximum_contacts} max_joint_speed={maximum_speed}"
            );
            assert!(
                maximum_contacts > 8,
                "fixture must exercise dense internal contacts"
            );
        }
    }
}

struct ArticulatedCarFixture {
    creation: mechanic_core::CompiledCreation,
    chassis: u32,
    dynamic_bodies: Vec<u32>,
    wheel_bodies: Vec<u32>,
}

fn spawned_part(outcome: BuildOutcome) -> PartId {
    let BuildOutcome::Spawned(part) = outcome else {
        unreachable!()
    };
    part
}

fn articulated_car_fixture() -> ArticulatedCarFixture {
    let mut graph = ConstructionGraph::new();
    let chassis_spec = CuboidSpec::new(
        [8, 2, 12],
        BuildPose::new(IVec3::new(0, 5, 0), GridRotation::default()),
    )
    .unwrap();
    let chassis_part = spawned_part(graph.apply(BuildCommand::Spawn(chassis_spec)).unwrap());

    let mut knuckles = Vec::new();
    let mut wheels = Vec::new();
    for z_units in [-4, 4] {
        let anchor_z = if z_units < 0 { -1.0 } else { 1.0 };
        for x_units in [-3, 3] {
            let steering_anchor_x = if x_units < 0 { -0.75 } else { 0.75 };
            let knuckle_spec = CuboidSpec::new(
                [2, 2, 2],
                BuildPose::new(IVec3::new(x_units, 3, z_units), GridRotation::default()),
            )
            .unwrap();
            let knuckle = spawned_part(graph.apply(BuildCommand::Spawn(knuckle_spec)).unwrap());
            knuckles.push(knuckle);

            let wheel_spec = CylinderSpec::new(
                CylinderDimensions::new(1.0, 0.0, 0.5).unwrap(),
                BuildPose::new(
                    IVec3::new(if x_units < 0 { -5 } else { 5 }, 3, z_units),
                    GridRotation::new(0, 0, 1),
                ),
            );
            let wheel = spawned_part(
                graph
                    .apply(BuildCommand::SpawnCylinder(wheel_spec))
                    .unwrap(),
            );
            wheels.push(wheel);

            graph
                .apply(BuildCommand::AddBearing(BearingSpec::new(
                    FaceRef::part(chassis_part, FaceKind::NegativeY),
                    FaceRef::part(knuckle, FaceKind::PositiveY),
                    Vec3::new(steering_anchor_x, 1.0, anchor_z),
                    Vec3::NEG_Y,
                )))
                .unwrap();
            let (source_face, target_face, axis, anchor_x) = if x_units < 0 {
                (FaceKind::NegativeX, FaceKind::NegativeY, Vec3::NEG_X, -1.0)
            } else {
                (FaceKind::PositiveX, FaceKind::PositiveY, Vec3::X, 1.0)
            };
            graph
                .apply(BuildCommand::AddBearing(BearingSpec::new(
                    FaceRef::part(knuckle, source_face),
                    FaceRef::part(wheel, target_face),
                    Vec3::new(anchor_x, 0.75, anchor_z),
                    axis,
                )))
                .unwrap();
        }
    }

    let wall_spec = CuboidSpec::new(
        [16, 8, 2],
        BuildPose::new(IVec3::new(0, 4, -10), GridRotation::default()),
    )
    .unwrap();
    let wall = spawned_part(graph.apply(BuildCommand::Spawn(wall_spec)).unwrap());
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(wall, FaceKind::NegativeY),
            second: FaceRef::ground(),
        }))
        .unwrap();

    let creation = graph.compile().unwrap();
    let body_for = |part| {
        creation
            .part_to_compound
            .iter()
            .find_map(|(candidate, body)| (*candidate == part).then_some(*body))
            .unwrap()
    };
    let chassis = body_for(chassis_part);
    let mut dynamic_bodies = vec![chassis];
    dynamic_bodies.extend(knuckles.iter().copied().map(body_for));
    let wheel_bodies = wheels.iter().copied().map(body_for).collect::<Vec<_>>();
    dynamic_bodies.extend(wheel_bodies.iter().copied());
    dynamic_bodies.sort_unstable();
    dynamic_bodies.dedup();
    ArticulatedCarFixture {
        creation,
        chassis,
        dynamic_bodies,
        wheel_bodies,
    }
}

fn pipe_bend_suspension_car_fixture() -> ArticulatedCarFixture {
    // The overshoot-only fixture models a heavy hydraulic steering rack.
    // Powered steering coverage below uses the production Servo torque.
    pipe_bend_suspension_car_fixture_with_steering_torque(64_000.0)
}

#[allow(clippy::too_many_lines)]
fn pipe_bend_suspension_car_fixture_with_steering_torque(
    steering_torque: f32,
) -> ArticulatedCarFixture {
    pipe_bend_car_fixture(steering_torque, false)
}

#[allow(clippy::too_many_lines)]
fn pipe_bend_car_fixture(steering_torque: f32, front_steering_only: bool) -> ArticulatedCarFixture {
    let mut graph = ConstructionGraph::new();
    let static_marker = spawned_part(
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [2, 2, 2],
                    BuildPose::new(IVec3::new(0, 1, -400), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap(),
    );
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(static_marker, FaceKind::NegativeY),
            second: FaceRef::ground(),
        }))
        .unwrap();
    let chassis_spec = CuboidSpec::new(
        [12, 1, 8],
        BuildPose::from_position_ticks(IVec3::new(0, 500, 0), GridRotation::default()),
    )
    .unwrap()
    .with_material(ConstructionMaterial::Stone);
    let chassis_part = spawned_part(graph.apply(BuildCommand::Spawn(chassis_spec)).unwrap());
    let bend_dimensions = PipeBendDimensions::new(0.2, 0.0, 2).unwrap();

    // Two-block bends need the chassis an eighth of a metre higher than the
    // captured one-block-radius bends did to keep the wheel axles in place.
    let corners = [
        (600, 1.5, 1, 0.75, 1.125),
        (-600, -1.5, 1, 0.75, 1.125),
        (-600, -1.5, -1, -0.75, -1.125),
        (600, 1.5, -1, -0.75, -1.125),
    ];
    let mut bends = [None; 4];
    let mut wheels = [None; 4];
    // Match the unfavorable child-before-parent ordering of a garage-built
    // car. Correct contact support must not depend on creation order.
    for (wheel, corner) in [
        (true, 0),
        (false, 1),
        (true, 1),
        (false, 2),
        (true, 2),
        (false, 3),
        (false, 0),
        (true, 3),
    ] {
        let (x_ticks, _, z_sign, _, _) = corners[corner];
        let bend_rotation = if z_sign > 0 {
            GridRotation::new(0, 3, 3)
        } else {
            GridRotation::new(0, 1, 3)
        };
        let wheel_rotation = if z_sign > 0 {
            GridRotation::new(1, 0, 0)
        } else {
            GridRotation::new(1, 2, 2)
        };
        if wheel {
            wheels[corner] = Some(spawned_part(
                graph
                    .apply(BuildCommand::SpawnCylinder(
                        CylinderSpec::new(
                            CylinderDimensions::new(0.95, 0.0, 0.25).unwrap(),
                            BuildPose::from_position_ticks(
                                IVec3::new(x_ticks, 290, z_sign * 500),
                                wheel_rotation,
                            ),
                        )
                        .with_material(ConstructionMaterial::Rubber),
                    ))
                    .unwrap(),
            ));
        } else {
            bends[corner] = Some(spawned_part(
                graph
                    .apply(BuildCommand::SpawnPipeBend(PipeBendSpec::new(
                        bend_dimensions,
                        BuildPose::from_position_ticks(
                            IVec3::new(x_ticks, 300, z_sign * 300),
                            bend_rotation,
                        ),
                    )))
                    .unwrap(),
            ));
        }
    }
    let ballast = spawned_part(
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 3, 4],
                    BuildPose::new(IVec3::new(-1, 5, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap(),
    );
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec {
            first: chassis_part,
            second: ballast,
        }))
        .unwrap();
    let bends = bends.map(Option::unwrap);
    let wheels = wheels.map(Option::unwrap);
    for corner in [0, 3, 2, 1] {
        let (_, x, _, bend_z, _) = corners[corner];
        if front_steering_only && x < 0.0 {
            graph
                .apply(BuildCommand::RigidLink(RigidLinkSpec {
                    first: chassis_part,
                    second: bends[corner],
                }))
                .unwrap();
            continue;
        }
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(chassis_part, FaceKind::NegativeY),
                FaceRef::part(bends[corner], FaceKind::NegativeX),
                Vec3::new(x, 1.125, bend_z),
                Vec3::NEG_Y,
            )))
            .unwrap();
    }
    for corner in [1, 0, 3, 2] {
        let (_, x, z_sign, _, wheel_z) = corners[corner];
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(bends[corner], FaceKind::PositiveY),
                FaceRef::part(wheels[corner], FaceKind::NegativeY),
                Vec3::new(x, 0.75, wheel_z),
                if z_sign > 0 { Vec3::Z } else { Vec3::NEG_Z },
            )))
            .unwrap();
    }

    let mut creation = graph.compile().unwrap();
    for coordinate in 0..if front_steering_only { 2 } else { 4 } {
        let acceleration =
            steering_torque / creation.loop_topology.coordinate_axis_inertia[coordinate];
        let drive = &mut creation.coordinate_drives[coordinate];
        *drive = CoordinateDrive {
            mode: DriveMode::Angle,
            target_speed: 0.0,
            target_angle: 0.0,
            max_speed: std::f32::consts::PI,
            max_acceleration: acceleration,
            source_a_max_acceleration: acceleration,
            source_a_no_load_speed: std::f32::consts::PI,
            source_b_max_acceleration: 0.0,
            source_b_no_load_speed: 0.0,
            min_angle: -std::f32::consts::FRAC_PI_4,
            max_angle: std::f32::consts::FRAC_PI_4,
        };
    }
    let body_for = |part| {
        creation
            .part_to_compound
            .iter()
            .find_map(|(candidate, body)| (*candidate == part).then_some(*body))
            .unwrap()
    };
    let chassis = body_for(chassis_part);
    let mut dynamic_bodies = vec![chassis];
    dynamic_bodies.extend(bends.into_iter().map(body_for));
    let wheel_bodies = wheels.iter().copied().map(body_for).collect::<Vec<_>>();
    dynamic_bodies.extend(wheel_bodies.iter().copied());
    dynamic_bodies.sort_unstable();
    dynamic_bodies.dedup();
    ArticulatedCarFixture {
        creation,
        chassis,
        dynamic_bodies,
        wheel_bodies,
    }
}

fn rigid_axle_car_fixture() -> ArticulatedCarFixture {
    let mut graph = ConstructionGraph::new();
    let chassis_part = spawned_part(
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [12, 2, 11],
                    BuildPose::from_position_ticks(IVec3::new(0, 300, 0), GridRotation::default()),
                )
                .unwrap()
                .with_material(ConstructionMaterial::Stone),
            ))
            .unwrap(),
    );
    let mut wheels = Vec::new();
    for x_ticks in [-400_i32, 400] {
        for z_ticks in [-600_i32, 600] {
            let z_sign = z_ticks.signum();
            let anchor_x = if x_ticks < 0 { -1.0 } else { 1.0 };
            let anchor_z = if z_ticks < 0 { -1.375 } else { 1.375 };
            let wheel = spawned_part(
                graph
                    .apply(BuildCommand::SpawnCylinder(
                        CylinderSpec::new(
                            CylinderDimensions::new(0.95, 0.0, 0.25).unwrap(),
                            BuildPose::from_position_ticks(
                                IVec3::new(x_ticks, 200, z_ticks),
                                if z_sign > 0 {
                                    GridRotation::new(1, 0, 0)
                                } else {
                                    GridRotation::new(1, 2, 2)
                                },
                            ),
                        )
                        .with_material(ConstructionMaterial::Rubber),
                    ))
                    .unwrap(),
            );
            wheels.push(wheel);
            graph
                .apply(BuildCommand::AddBearing(BearingSpec::new(
                    FaceRef::part(
                        chassis_part,
                        if z_sign > 0 {
                            FaceKind::PositiveZ
                        } else {
                            FaceKind::NegativeZ
                        },
                    ),
                    FaceRef::part(wheel, FaceKind::NegativeY),
                    Vec3::new(anchor_x, 0.5, anchor_z),
                    if z_sign > 0 { Vec3::Z } else { Vec3::NEG_Z },
                )))
                .unwrap();
        }
    }

    let mut creation = graph.compile().unwrap();
    for drive in &mut creation.coordinate_drives {
        drive.mode = DriveMode::Speed;
        drive.target_speed = 0.0;
        drive.max_speed = std::f32::consts::TAU * 6.0;
    }
    let body_for = |part| {
        creation
            .part_to_compound
            .iter()
            .find_map(|(candidate, body)| (*candidate == part).then_some(*body))
            .unwrap()
    };
    let chassis = body_for(chassis_part);
    let wheel_bodies = wheels.into_iter().map(body_for).collect::<Vec<_>>();
    let mut dynamic_bodies = vec![chassis];
    dynamic_bodies.extend(wheel_bodies.iter().copied());
    ArticulatedCarFixture {
        creation,
        chassis,
        dynamic_bodies,
        wheel_bodies,
    }
}

#[test]
fn full_cylinder_ground_contacts_match_visual_wheel_radius() {
    let fixture = articulated_car_fixture();
    let ground_data = full_cylinder_ground_data(&fixture.creation.colliders);
    let analytic = ground_data
        .iter()
        .filter(|ground| ground.role != 0)
        .collect::<Vec<_>>();
    let primary = ground_data
        .iter()
        .filter(|ground| ground.role == FULL_CYLINDER_GROUND_FIRST)
        .collect::<Vec<_>>();
    let secondary_count = ground_data
        .iter()
        .filter(|ground| ground.role > FULL_CYLINDER_GROUND_FIRST)
        .count();
    assert_eq!(primary.len(), 4);
    assert_eq!(analytic.len(), 64);
    assert_eq!(secondary_count, 60);
    assert!(analytic.iter().all(|ground| {
        (ground.center_radius - 0.25).abs() < 1.0e-6 && (ground.outer_radius - 0.5).abs() < 1.0e-6
    }));
    for role in 1..=16 {
        assert_eq!(
            analytic.iter().filter(|ground| ground.role == role).count(),
            4
        );
    }

    let mut sector_graph = ConstructionGraph::new();
    let sector_dimensions = CylinderDimensions::new(1.0, 0.0, 0.5)
        .unwrap()
        .with_sweep_angle_degrees(180)
        .unwrap();
    sector_graph
        .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
            sector_dimensions,
            BuildPose::default(),
        )))
        .unwrap();
    let sector = sector_graph.compile().unwrap();
    assert!(
        full_cylinder_ground_data(&sector.colliders)
            .iter()
            .all(|ground| ground.role == 0)
    );
}

fn transform_position(transform: crate::GpuTransform) -> Vec3 {
    Vec3::new(
        transform.position[0],
        transform.position[1],
        transform.position[2],
    )
}

fn snapshot_speed(current: crate::GpuTransform, previous: crate::GpuTransform) -> f32 {
    (transform_position(current) - transform_position(previous)).length() * 60.0
}

fn bearing_speed(
    current: &[crate::GpuTransform],
    previous: &[crate::GpuTransform],
    bearing: &mechanic_core::CompiledBearing,
) -> f32 {
    let current_relative = relative_bearing_rotation(current, bearing);
    let previous_relative = relative_bearing_rotation(previous, bearing);
    let delta = previous_relative.conjugate() * current_relative;
    2.0 * delta.xyz().length().atan2(delta.w.abs()) * 60.0
}

#[test]
fn gpu_pipelines_construct_on_noop_backend() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
        label: Some("mechanic pipeline validation device"),
        ..Default::default()
    });
    let creation = pendulum_creation(true);
    for mechanism_self_collisions in [true, false] {
        GpuPhysics::new_with_config(
            &device,
            &queue,
            &creation,
            GpuPhysicsConfig {
                collisions_enabled: true,
                ground_plane_enabled: true,
                mechanism_self_collisions,
                solver_iterations: 8,
            },
        )
        .unwrap();
    }
}

#[test]
fn collider_local_ground_planes_support_bodies_at_different_heights() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let mut graph = ConstructionGraph::new();
    let low = match graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [4; 3],
                BuildPose::from_half_grid(IVec3::new(-16, 16, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    {
        BuildOutcome::Spawned(part) => part,
        other => panic!("unexpected build outcome {other:?}"),
    };
    let high = match graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [4; 3],
                BuildPose::from_half_grid(IVec3::new(16, 16, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    {
        BuildOutcome::Spawned(part) => part,
        other => panic!("unexpected build outcome {other:?}"),
    };
    let creation = graph.compile().unwrap();
    let body_for = |part| {
        creation
            .part_to_compound
            .iter()
            .find_map(|(candidate, body)| (*candidate == part).then_some(*body as usize))
            .unwrap()
    };
    let planes = creation
        .colliders
        .iter()
        .map(|collider| GpuGroundPlane {
            normal: Vec3::Y,
            offset: if collider.source_part == low {
                0.0
            } else {
                0.75
            },
        })
        .collect::<Vec<_>>();
    let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
    assert_eq!(
        gpu.write_ground_planes(&queue, &planes[..1]),
        Err(GpuGroundPlaneError::PlaneCount {
            provided: 1,
            expected: 2,
        })
    );
    gpu.write_ground_planes(&queue, &planes).unwrap();
    for tick in 1..=180 {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let snapshot = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
    let low_y = snapshot[body_for(low)].position[1];
    let high_y = snapshot[body_for(high)].position[1];
    assert!((low_y - 0.5).abs() < 0.02, "low body settled at {low_y}");
    assert!(
        (high_y - 1.25).abs() < 0.02,
        "high body settled at {high_y}"
    );
    assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
}

#[test]
fn late_mapping_callbacks_cannot_reorder_publication() {
    assert_eq!(
        super::oldest_completed_readback([(2, 7, 1), (0, 8, 0)].into_iter()),
        None
    );
    assert_eq!(
        super::oldest_completed_readback([(2, 7, 0), (0, 8, 0)].into_iter()),
        Some(2)
    );
}

#[test]
fn submission_sequence_and_execution_evidence_survive_scheduler_gaps() {
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
fn callback_timing_distinguishes_servicing_from_later_consumption() {
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
fn asynchronous_tick_readback_is_monotonic_and_tick_matched() {
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

fn copy_state_rows<T: bytemuck::Pod>(
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
    super::map_for_read(device, &destination).unwrap();
    let rows = super::mapped_rows(&destination, count);
    destination.unmap();
    rows
}

#[test]
fn authoritative_readback_keeps_velocities_and_coordinates_with_their_tick() {
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
#[allow(clippy::float_cmp)] // Held velocities must remain exactly zero.
fn held_rotational_and_linear_components_ignore_drives_impulses_and_reconstruction() {
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
fn held_body_supports_a_falling_neighbor_without_moving() {
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
fn releasing_a_loaded_hold_discards_only_changed_contact_warmstarts() {
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

#[test]
fn restored_authoritative_state_continues_rotational_and_linear_motion() {
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
fn authoritative_readback_supports_scenes_without_joints() {
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

#[test]
fn off_centre_external_impulse_changes_linear_and_angular_motion() {
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
fn external_impulse_batches_chunk_repeated_rows_and_validate_atomically() {
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
fn offset_ground_contact_applies_angular_impulse_about_compound_centre() {
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
fn articulated_car_drop_settles_without_drift_or_ground_penetration() {
    let fixture = articulated_car_fixture();
    let Some((device, queue)) = test_device() else {
        return;
    };
    let gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        &fixture.creation,
        GpuPhysicsConfig {
            mechanism_self_collisions: false,
            solver_iterations: 8,
            ..Default::default()
        },
    )
    .unwrap();
    let initial_chassis = fixture.creation.compounds[fixture.chassis as usize].root_translation;
    let mut previous_sample_tick = 0;
    let sample_ticks = (10..=300).step_by(10).chain((330..=1_200).step_by(30));
    for sample_tick in sample_ticks {
        for tick in previous_sample_tick + 1..=sample_tick {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        previous_sample_tick = sample_tick;
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let snapshot = gpu
            .read_snapshot_transforms(&device, &queue, (sample_tick % 3) as u8)
            .unwrap();
        let chassis_rotation =
            bevy_math::Quat::from_array(snapshot[fixture.chassis as usize].rotation);
        assert!(
            transform_position(snapshot[fixture.chassis as usize]).y >= 0.74,
            "chassis entered the ground at tick {sample_tick}"
        );
        assert!(
            (chassis_rotation * Vec3::Y).dot(Vec3::Y) > 0.9,
            "unpowered chassis tipped at tick {sample_tick}"
        );
        for &wheel in &fixture.wheel_bodies {
            let wheel_height = transform_position(snapshot[wheel as usize]).y;
            assert!(
                wheel_height >= 0.495,
                "wheel {wheel} entered the ground at tick {sample_tick}: y={wheel_height}"
            );
        }
    }

    let current = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
    let previous = gpu.read_snapshot_transforms(&device, &queue, 2).unwrap();
    let diagnostics = gpu.read_last_tick(&device).unwrap();
    assert_eq!(diagnostics.error_flags, 0);
    assert!(diagnostics.anchor_residual_meters <= mechanic_core::ANCHOR_TOLERANCE_METERS);
    assert!(diagnostics.axis_residual_degrees <= mechanic_core::AXIS_TOLERANCE_DEGREES);
    let chassis_position = transform_position(current[fixture.chassis as usize]);
    let horizontal_drift = Vec3::new(
        chassis_position.x - initial_chassis.x,
        0.0,
        chassis_position.z - initial_chassis.z,
    )
    .length();
    assert!(
        horizontal_drift < 0.05,
        "unpowered chassis drifted {horizontal_drift} m"
    );
    let max_linear_speed = fixture
        .dynamic_bodies
        .iter()
        .map(|&body| snapshot_speed(current[body as usize], previous[body as usize]))
        .fold(0.0_f32, f32::max);
    let max_bearing_speed = fixture
        .creation
        .bearings
        .iter()
        .map(|bearing| bearing_speed(&current, &previous, bearing))
        .fold(0.0_f32, f32::max);
    assert!(
        max_linear_speed < 0.02,
        "linear speed was {max_linear_speed}"
    );
    assert!(
        max_bearing_speed < 0.02,
        "bearing angular speed was {max_bearing_speed}"
    );
}

#[test]
fn curved_suspension_car_lands_without_contact_correction_launch() {
    let fixture = pipe_bend_suspension_car_fixture();
    let ground_data = full_cylinder_ground_data(&fixture.creation.colliders);
    assert_eq!(
        ground_data
            .iter()
            .filter(|data| data.role == FULL_CYLINDER_GROUND_FIRST)
            .count(),
        4
    );
    assert_eq!(fixture.creation.bearings.len(), 8);
    assert_eq!(fixture.dynamic_bodies.len(), 9);
    assert!(
        fixture
            .creation
            .compounds
            .iter()
            .any(|compound| compound.collider_range.len() >= 192)
    );
    let Some((device, queue)) = test_device() else {
        return;
    };
    let gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        &fixture.creation,
        GpuPhysicsConfig {
            mechanism_self_collisions: false,
            ..Default::default()
        },
    )
    .unwrap();
    let initial_chassis = fixture.creation.compounds[fixture.chassis as usize].root_translation;
    let mut maximum_chassis_height = initial_chassis.y;
    let mut minimum_wheel_height = f32::INFINITY;
    for tick in 1..=1_200 {
        gpu.dispatch_tick(&device, &queue, tick);
        if tick.is_multiple_of(30) {
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let snapshot = gpu
                .read_snapshot_transforms(&device, &queue, (tick % 3) as u8)
                .unwrap();
            maximum_chassis_height = maximum_chassis_height
                .max(transform_position(snapshot[fixture.chassis as usize]).y);
            minimum_wheel_height = fixture
                .wheel_bodies
                .iter()
                .map(|&wheel| transform_position(snapshot[wheel as usize]).y)
                .fold(minimum_wheel_height, f32::min);
        }
    }

    let diagnostics = gpu.read_last_tick(&device).unwrap();
    let final_snapshot = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
    let final_wheel_height = fixture
        .wheel_bodies
        .iter()
        .map(|&wheel| transform_position(final_snapshot[wheel as usize]).y)
        .fold(f32::INFINITY, f32::min);
    assert_eq!(diagnostics.error_flags, 0);
    assert_eq!(diagnostics.planned_solver_sweeps, 8);
    assert_eq!(diagnostics.executed_solver_sweeps, 8);
    assert!(
        maximum_chassis_height < initial_chassis.y + 0.5,
        "ground correction launched the chassis from {} m to {maximum_chassis_height} m",
        initial_chassis.y,
    );
    assert!(
        minimum_wheel_height >= 0.39,
        "a wheel sank through the ground to {minimum_wheel_height} m"
    );
    assert!(
        final_wheel_height >= 0.44,
        "a wheel remained below the ground at {final_wheel_height} m"
    );
}

#[test]
fn steering_servos_reach_angle_without_overshooting() {
    let fixture = pipe_bend_suspension_car_fixture();
    let Some((device, queue)) = test_device() else {
        return;
    };
    let gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        &fixture.creation,
        GpuPhysicsConfig {
            mechanism_self_collisions: false,
            ..Default::default()
        },
    )
    .unwrap();
    for tick in 1..=240 {
        gpu.dispatch_tick(&device, &queue, tick);
    }

    let target = std::f32::consts::FRAC_PI_6;
    let mut drives = fixture
        .creation
        .coordinate_drives
        .iter()
        .copied()
        .map(crate::GpuMechanismDrive::from)
        .collect::<Vec<_>>();
    for drive in &mut drives[..4] {
        drive.target_angle = target;
    }
    gpu.write_mechanism_drives(&queue, &drives).unwrap();

    let bearing_angle = |snapshot: &[crate::GpuTransform], coordinate: usize| {
        let source_bearing = fixture.creation.loop_topology.tree_bearings[coordinate];
        let bearing = fixture
            .creation
            .bearings
            .iter()
            .find(|bearing| bearing.source_bearing == source_bearing)
            .unwrap();
        let relative = relative_bearing_rotation(snapshot, bearing);
        2.0 * relative
            .xyz()
            .dot(bearing.local_axis_a.normalize())
            .atan2(relative.w)
    };
    let mut maximum_angles = [f32::NEG_INFINITY; 4];
    let mut final_angles = [0.0; 4];
    let mut previous_tick = 240;
    for sample_tick in 241..=480 {
        for tick in previous_tick + 1..=sample_tick {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        previous_tick = sample_tick;
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let snapshot = gpu
            .read_snapshot_transforms(&device, &queue, (sample_tick % 3) as u8)
            .unwrap();
        for coordinate in 0..4 {
            let angle = bearing_angle(&snapshot, coordinate);
            maximum_angles[coordinate] = maximum_angles[coordinate].max(angle);
            final_angles[coordinate] = angle;
        }
    }

    let tolerance = 0.15_f32.to_radians();
    for coordinate in 0..4 {
        assert!(
            maximum_angles[coordinate] <= target + tolerance,
            "steering coordinate {coordinate} overshot 30 degrees to {} degrees",
            maximum_angles[coordinate].to_degrees(),
        );
        assert!(
            (final_angles[coordinate] - target).abs() <= tolerance,
            "steering coordinate {coordinate} settled at {} degrees",
            final_angles[coordinate].to_degrees(),
        );
    }
    assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
}

#[test]
#[allow(clippy::too_many_lines)]
fn production_servos_hold_steering_under_first_gear_gas_drive() {
    const FIRST_GEAR_RATIO: f32 = 3.0;
    const COMMAND_SPEED: f32 = std::f32::consts::TAU * 6.0;
    const STEERING_COORDINATES: usize = 4;
    const HOLD_TOLERANCE: f32 = std::f32::consts::PI / 180.0;

    let mut fixture = pipe_bend_suspension_car_fixture_with_steering_torque(
        ServoSpec::STALL_TORQUE_NEWTON_METERS,
    );
    assert_eq!(fixture.creation.coordinate_drives.len(), 8);
    for coordinate in 0..STEERING_COORDINATES {
        let inertia = fixture.creation.loop_topology.coordinate_axis_inertia[coordinate];
        let compiled_torque =
            fixture.creation.coordinate_drives[coordinate].source_a_max_acceleration * inertia;
        assert!((compiled_torque - ServoSpec::STALL_TORQUE_NEWTON_METERS).abs() < 0.01);
    }
    for coordinate in STEERING_COORDINATES..fixture.creation.coordinate_drives.len() {
        let source_bearing = fixture.creation.loop_topology.tree_bearings[coordinate];
        let bearing = fixture
            .creation
            .bearings
            .iter()
            .find(|bearing| bearing.source_bearing == source_bearing)
            .unwrap();
        let world_axis = fixture.creation.compounds[bearing.compound_a as usize].root_rotation
            * bearing.local_axis_a;
        let inertia = fixture.creation.loop_topology.coordinate_axis_inertia[coordinate];
        let acceleration =
            EngineKind::Gas.stall_torque_newton_meters() * FIRST_GEAR_RATIO / 4.0 / inertia;
        fixture.creation.coordinate_drives[coordinate] = CoordinateDrive {
            mode: DriveMode::Speed,
            target_speed: 0.0,
            target_angle: 0.0,
            max_speed: mechanic_core::rpm_to_rad_s(EngineKind::Gas.no_load_rpm())
                / FIRST_GEAR_RATIO,
            max_acceleration: acceleration,
            source_a_max_acceleration: 0.0,
            source_a_no_load_speed: 0.0,
            source_b_max_acceleration: acceleration,
            source_b_no_load_speed: mechanic_core::rpm_to_rad_s(EngineKind::Gas.no_load_rpm())
                / FIRST_GEAR_RATIO,
            min_angle: f32::NEG_INFINITY,
            max_angle: f32::INFINITY,
        };
        assert!(world_axis.cross(Vec3::Y).dot(Vec3::X).abs() > 0.99);
    }

    let Some((device, queue)) = test_device() else {
        return;
    };
    let gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        &fixture.creation,
        GpuPhysicsConfig {
            mechanism_self_collisions: false,
            ..Default::default()
        },
    )
    .unwrap();
    for tick in 1..=240 {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    let settled = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
    let starting_position = transform_position(settled[fixture.chassis as usize]);
    let mut drives = fixture
        .creation
        .coordinate_drives
        .iter()
        .copied()
        .map(crate::GpuMechanismDrive::from)
        .collect::<Vec<_>>();
    for (coordinate, drive) in drives.iter_mut().enumerate().skip(STEERING_COORDINATES) {
        let source_bearing = fixture.creation.loop_topology.tree_bearings[coordinate];
        let bearing = fixture
            .creation
            .bearings
            .iter()
            .find(|bearing| bearing.source_bearing == source_bearing)
            .unwrap();
        let world_axis = fixture.creation.compounds[bearing.compound_a as usize].root_rotation
            * bearing.local_axis_a;
        drive.target_speed = world_axis.cross(Vec3::Y).dot(Vec3::X).signum() * COMMAND_SPEED;
    }
    gpu.write_mechanism_drives(&queue, &drives).unwrap();
    let bearing_angle = |snapshot: &[crate::GpuTransform], coordinate: usize| {
        let source_bearing = fixture.creation.loop_topology.tree_bearings[coordinate];
        let bearing = fixture
            .creation
            .bearings
            .iter()
            .find(|bearing| bearing.source_bearing == source_bearing)
            .unwrap();
        let relative = relative_bearing_rotation(snapshot, bearing);
        2.0 * relative
            .xyz()
            .dot(bearing.local_axis_a.normalize())
            .atan2(relative.w)
    };

    let mut maximum_center_error = 0.0_f32;
    let mut maximum_horizontal_travel = 0.0_f32;
    for sample_tick in (243..=600).step_by(3) {
        for tick in sample_tick - 2..=sample_tick {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        let snapshot = gpu
            .read_snapshot_transforms(&device, &queue, (sample_tick % 3) as u8)
            .unwrap();
        let displacement =
            transform_position(snapshot[fixture.chassis as usize]) - starting_position;
        maximum_horizontal_travel =
            maximum_horizontal_travel.max(Vec3::new(displacement.x, 0.0, displacement.z).length());
        for coordinate in 0..STEERING_COORDINATES {
            maximum_center_error =
                maximum_center_error.max(bearing_angle(&snapshot, coordinate).abs());
        }
    }
    assert!(
        maximum_horizontal_travel > 1.0,
        "first-gear drive moved the chassis only {maximum_horizontal_travel} m",
    );
    assert!(
        maximum_center_error <= HOLD_TOLERANCE,
        "full first-gear drive deflected centred steering by {} degrees",
        maximum_center_error.to_degrees(),
    );

    let target = std::f32::consts::FRAC_PI_6;
    for drive in &mut drives[..STEERING_COORDINATES] {
        drive.target_angle = target;
    }
    gpu.write_mechanism_drives(&queue, &drives).unwrap();
    let mut maximum_settled_error = 0.0_f32;
    let mut final_angles = [0.0; STEERING_COORDINATES];
    for sample_tick in (603..=1_080).step_by(3) {
        for tick in sample_tick - 2..=sample_tick {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        let snapshot = gpu
            .read_snapshot_transforms(&device, &queue, (sample_tick % 3) as u8)
            .unwrap();
        for (coordinate, final_angle) in final_angles.iter_mut().enumerate() {
            let angle = bearing_angle(&snapshot, coordinate);
            *final_angle = angle;
            if sample_tick >= 960 {
                maximum_settled_error = maximum_settled_error.max((angle - target).abs());
            }
        }
    }
    for (coordinate, angle) in final_angles.into_iter().enumerate() {
        assert!(
            (angle - target).abs() <= HOLD_TOLERANCE,
            "steering coordinate {coordinate} reached {} instead of 30 degrees under drive",
            angle.to_degrees(),
        );
    }
    assert!(
        maximum_settled_error <= HOLD_TOLERANCE,
        "powered steering wandered by {} degrees after settling",
        maximum_settled_error.to_degrees(),
    );
    assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
}

#[test]
#[allow(clippy::too_many_lines)]
fn front_steered_car_turns_through_ground_friction() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let fixture = pipe_bend_car_fixture(ServoSpec::STALL_TORQUE_NEWTON_METERS, true);
    let pipelines = super::GpuPhysicsPipelines::new();
    let mut turns = Vec::new();
    for target in [-30.0_f32, 0.0, 30.0] {
        let gpu = GpuPhysics::new_with_pipelines(
            &device,
            &queue,
            &fixture.creation,
            GpuPhysicsConfig {
                mechanism_self_collisions: false,
                ..Default::default()
            },
            &pipelines,
        )
        .unwrap();
        for tick in 1..=240 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        let start = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
        let chassis = fixture.chassis as usize;
        let initial = transform_position(start[chassis]);
        let initial_rotation = bevy_math::Quat::from_array(start[chassis].rotation);
        let mut drives = fixture
            .creation
            .coordinate_drives
            .iter()
            .copied()
            .map(crate::GpuMechanismDrive::from)
            .collect::<Vec<_>>();
        for drive in &mut drives[..2] {
            drive.target_angle = target.to_radians();
        }
        for bearing in &fixture.creation.bearings {
            let coordinate = bearing.coordinate_index.unwrap() as usize;
            if coordinate < 2 {
                continue;
            }
            let acceleration =
                1_000.0 / fixture.creation.loop_topology.coordinate_axis_inertia[coordinate];
            drives[coordinate] = crate::GpuMechanismDrive::from(CoordinateDrive {
                mode: DriveMode::Speed,
                target_speed: bearing.local_axis_a.cross(Vec3::Y).dot(Vec3::X).signum() * 3.0,
                max_speed: 3.0,
                max_acceleration: acceleration,
                source_b_max_acceleration: acceleration,
                source_b_no_load_speed: 30.0,
                ..CoordinateDrive::default()
            });
        }
        gpu.write_mechanism_drives(&queue, &drives).unwrap();
        for tick in 241..=600 {
            gpu.dispatch_tick(&device, &queue, tick);
            if tick % 30 == 0 {
                let sample = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
                let up = bevy_math::Quat::from_array(sample[chassis].rotation) * Vec3::Y;
                assert!(up.y > 0.98, "car tipped at tick {tick}: {up:?}");
                let height_change = transform_position(sample[chassis]).y - initial.y;
                assert!(
                    height_change.abs() < 0.15,
                    "car bounced at tick {tick}: {height_change}"
                );
                for &wheel in &fixture.wheel_bodies {
                    assert!(
                        sample[wheel as usize].position[1] > 0.44,
                        "wheel sank at tick {tick}"
                    );
                }
            }
        }
        let end = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
        let forward = initial_rotation.conjugate()
            * bevy_math::Quat::from_array(end[chassis].rotation)
            * Vec3::X;
        let yaw = (-forward.z).atan2(forward.x).to_degrees();
        let displacement =
            initial_rotation.conjugate() * (transform_position(end[chassis]) - initial);
        eprintln!("steer={target} yaw={yaw} displacement={displacement:?}");
        for bearing in &fixture.creation.bearings[..2] {
            let relative = relative_bearing_rotation(&end, bearing);
            let angle =
                (2.0 * relative.xyz().dot(bearing.local_axis_a).atan2(relative.w)).to_degrees();
            eprintln!("steering angle={angle}");
            assert!(
                (angle - target).abs() < 3.0,
                "steering deflected: target {target}, actual {angle}"
            );
        }
        assert!(
            displacement.x > 3.0,
            "car did not drive forward: {displacement:?}"
        );
        assert!(
            forward.y.abs() < 0.1,
            "car tipped while steering: {forward:?}"
        );
        assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
        turns.push((yaw, displacement.z));
    }
    assert!(
        turns[0].0 > 20.0 && turns[0].1 < -1.0,
        "left command did not turn: {turns:?}"
    );
    assert!(
        turns[2].0 < -20.0 && turns[2].1 > 1.0,
        "right command did not turn: {turns:?}"
    );
    assert!(
        turns[1].0.abs() < 2.0 && turns[1].1.abs() < 0.2,
        "straight command drifted: {turns:?}"
    );
    assert!(
        (turns[0].0 + turns[2].0).abs() < 5.0,
        "turns were asymmetric: {turns:?}"
    );
}

mod steering;

#[test]
#[allow(clippy::too_many_lines)]
fn sustained_gas_drive_stays_forward_with_bounded_longitudinal_slip() {
    const COMMAND_SPEED: f32 = std::f32::consts::TAU * 6.0;
    const WHEEL_RADIUS: f32 = 0.475;
    const SAMPLE_TICKS: u64 = 3;
    const SAMPLE_SECONDS: f32 = 0.05;
    const DRIVE_END_TICK: u64 = 3_600;
    const REVERSE_TICKS: u64 = 720;
    const REVERSE_SECONDS: f32 = 12.0;

    let fixture = rigid_axle_car_fixture();
    let mass = fixture
        .dynamic_bodies
        .iter()
        .map(|&body| {
            fixture.creation.compounds[body as usize]
                .mass_properties
                .mass
        })
        .sum::<f32>();
    assert!(
        (11_000.0..=12_500.0).contains(&mass),
        "fixture mass was {mass} kg"
    );
    let Some((device, queue)) = test_device() else {
        return;
    };
    let gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        &fixture.creation,
        GpuPhysicsConfig {
            mechanism_self_collisions: false,
            ..Default::default()
        },
    )
    .unwrap();
    for tick in 1..=240 {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    let gearbox = GearboxConfig::for_depth(2, true);
    assert_eq!(gearbox.ratios(), [4.0, 3.0, 1.0]);
    let rolling_sign = |bearing: &mechanic_core::CompiledBearing| {
        let world_axis = fixture.creation.compounds[bearing.compound_a as usize].root_rotation
            * bearing.local_axis_a;
        world_axis.cross(Vec3::Y).dot(Vec3::X).signum()
    };
    let make_drives = |ratio: f32, command_speed: f32| {
        let output_speed = mechanic_core::rpm_to_rad_s(EngineKind::Gas.no_load_rpm()) / ratio;
        let mut drives = fixture
            .creation
            .coordinate_drives
            .iter()
            .copied()
            .map(crate::GpuMechanismDrive::from)
            .collect::<Vec<_>>();
        for (coordinate, drive) in drives.iter_mut().enumerate() {
            let source_bearing = fixture.creation.loop_topology.tree_bearings[coordinate];
            let bearing = fixture
                .creation
                .bearings
                .iter()
                .find(|bearing| bearing.source_bearing == source_bearing)
                .unwrap();
            let acceleration = EngineKind::Gas.stall_torque_newton_meters() * ratio
                / 4.0
                / fixture.creation.loop_topology.coordinate_axis_inertia[coordinate];
            *drive = crate::GpuMechanismDrive::from(CoordinateDrive {
                mode: DriveMode::Speed,
                target_speed: rolling_sign(bearing) * command_speed,
                target_angle: 0.0,
                max_speed: output_speed,
                max_acceleration: acceleration,
                source_a_max_acceleration: 0.0,
                source_a_no_load_speed: 0.0,
                source_b_max_acceleration: acceleration,
                source_b_no_load_speed: output_speed,
                min_angle: f32::NEG_INFINITY,
                max_angle: f32::INFINITY,
            });
        }
        drives
    };

    let mut ratio = gearbox.ratios()[usize::from(gearbox.reverse_gears())];
    let mut drives = make_drives(ratio, COMMAND_SPEED);
    gpu.write_mechanism_drives(&queue, &drives).unwrap();

    let mut previous = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
    let initial_x = transform_position(previous[fixture.chassis as usize]).x;
    let mut furthest_x = initial_x;
    let mut maximum_speed = 0.0_f32;
    let mut maximum_speed_tick = 0_u64;
    let mut speed_at_twelve_seconds = 0.0_f32;
    let mut shifted = false;
    let mut maximum_engine_rpm = 0.0_f32;
    let mut slips = Vec::new();
    for sample_tick in (240 + SAMPLE_TICKS..=DRIVE_END_TICK).step_by(3) {
        for tick in sample_tick - SAMPLE_TICKS + 1..=sample_tick {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        let current = gpu
            .read_snapshot_transforms(&device, &queue, (sample_tick % 3) as u8)
            .unwrap();
        let current_x = transform_position(current[fixture.chassis as usize]).x;
        let previous_x = transform_position(previous[fixture.chassis as usize]).x;
        let chassis_speed = (current_x - previous_x) / SAMPLE_SECONDS;
        let wheel_surface_speed = fixture
            .creation
            .bearings
            .iter()
            .map(|bearing| {
                let angular_velocity = |body: u32| {
                    let current_rotation =
                        bevy_math::Quat::from_array(current[body as usize].rotation);
                    let previous_rotation =
                        bevy_math::Quat::from_array(previous[body as usize].rotation);
                    let mut delta = current_rotation * previous_rotation.conjugate();
                    if delta.w < 0.0 {
                        delta = -delta;
                    }
                    let vector = delta.xyz();
                    if vector.length_squared() <= 1.0e-12 {
                        Vec3::ZERO
                    } else {
                        vector.normalize() * (2.0 * vector.length().atan2(delta.w) / SAMPLE_SECONDS)
                    }
                };
                let parent_rotation =
                    bevy_math::Quat::from_array(current[bearing.compound_a as usize].rotation);
                let axis = parent_rotation * bearing.local_axis_a.normalize();
                let relative_angular =
                    angular_velocity(bearing.compound_b) - angular_velocity(bearing.compound_a);
                relative_angular.dot(axis) * axis.cross(Vec3::Y).dot(Vec3::X) * WHEEL_RADIUS
            })
            .sum::<f32>()
            / 4.0;

        furthest_x = furthest_x.max(current_x);
        assert!(
            current_x >= furthest_x - 0.05,
            "forward displacement reversed by {} m at tick {sample_tick}",
            furthest_x - current_x,
        );
        assert!(
            chassis_speed >= -0.05,
            "chassis reversed at {chassis_speed} m/s on tick {sample_tick}",
        );
        if chassis_speed > maximum_speed {
            maximum_speed = chassis_speed;
            maximum_speed_tick = sample_tick;
        }
        if sample_tick == 960 {
            speed_at_twelve_seconds = chassis_speed;
        }
        if chassis_speed > 0.5 {
            let slip = (wheel_surface_speed - chassis_speed).abs()
                / wheel_surface_speed.abs().max(chassis_speed.abs()).max(0.5);
            slips.push(slip);
        }
        if !shifted {
            let engine_rpm =
                mechanic_core::rad_s_to_rpm(wheel_surface_speed.abs() / WHEEL_RADIUS * ratio);
            maximum_engine_rpm = maximum_engine_rpm.max(engine_rpm);
            if engine_rpm >= 0.75 * EngineKind::Gas.no_load_rpm() {
                ratio = 1.0;
                drives = make_drives(ratio, COMMAND_SPEED);
                gpu.write_mechanism_drives(&queue, &drives).unwrap();
                shifted = true;
            }
        }
        previous = current;
    }

    slips.sort_by(f32::total_cmp);
    let p95_slip = slips[(slips.len() * 95 / 100).min(slips.len() - 1)];
    let geared_ceiling = COMMAND_SPEED * WHEEL_RADIUS;
    assert!(
        shifted,
        "automatic first-to-second shift was never reached; peak engine speed {maximum_engine_rpm} RPM, chassis {maximum_speed} m/s"
    );
    assert!(
        speed_at_twelve_seconds >= 8.33,
        "vehicle reached only {speed_at_twelve_seconds} m/s after 12 seconds; peak {maximum_speed}, p95 slip {p95_slip}",
    );
    assert!(
        maximum_speed >= 13.9,
        "vehicle peaked at only {maximum_speed} m/s during sustained drive",
    );
    assert!(
        maximum_speed <= geared_ceiling + 0.25,
        "vehicle exceeded the geared no-load ceiling at tick {maximum_speed_tick}: {maximum_speed} > {geared_ceiling}; 12-second speed {speed_at_twelve_seconds}, p95 slip {p95_slip}",
    );
    assert!(
        p95_slip < 0.10,
        "p95 longitudinal slip was {:.1}%",
        p95_slip * 100.0
    );

    let coast_start = previous;
    for drive in &mut drives {
        drive.target_speed = 0.0;
        drive.max_acceleration = 0.0;
        drive.max_speed = 0.0;
        drive.source_b_max_acceleration = 0.0;
        drive.source_b_no_load_speed = 0.0;
    }
    gpu.write_mechanism_drives(&queue, &drives).unwrap();
    for tick in DRIVE_END_TICK + 1..=DRIVE_END_TICK + 120 {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    let coast_end = gpu
        .read_snapshot_transforms(&device, &queue, ((DRIVE_END_TICK + 120) % 3) as u8)
        .unwrap();
    let coast_speed = (transform_position(coast_end[fixture.chassis as usize]).x
        - transform_position(coast_start[fixture.chassis as usize]).x)
        / 2.0;
    assert!(
        coast_speed > 1.0,
        "neutral actively stopped the vehicle: {coast_speed} m/s"
    );

    let reverse = make_drives(gearbox.ratios()[0], -COMMAND_SPEED);
    gpu.write_mechanism_drives(&queue, &reverse).unwrap();
    for tick in DRIVE_END_TICK + 121..=DRIVE_END_TICK + 120 + REVERSE_TICKS {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    let reverse_end = gpu
        .read_snapshot_transforms(
            &device,
            &queue,
            ((DRIVE_END_TICK + 120 + REVERSE_TICKS) % 3) as u8,
        )
        .unwrap();
    let reverse_speed = (transform_position(reverse_end[fixture.chassis as usize]).x
        - transform_position(coast_end[fixture.chassis as usize]).x)
        / REVERSE_SECONDS;
    assert!(
        reverse_speed < -0.05,
        "explicit reverse did not reverse: {reverse_speed} m/s"
    );
    assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
}

#[test]
fn struck_articulated_wheel_recovers_without_crossing_ground() {
    let fixture = articulated_car_fixture();
    let Some((device, queue)) = test_device() else {
        return;
    };
    let gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        &fixture.creation,
        GpuPhysicsConfig {
            mechanism_self_collisions: false,
            ..Default::default()
        },
    )
    .unwrap();
    for tick in 1..=300 {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let settled = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
    let wheel = fixture.wheel_bodies[0];
    let wheel_position = transform_position(settled[wheel as usize]);
    let wheel_mass = fixture.creation.compounds[wheel as usize]
        .mass_properties
        .mass;
    gpu.apply_impulse(
        &device,
        &queue,
        wheel,
        wheel_position,
        Vec3::NEG_Y * wheel_mass * 3.0,
    )
    .unwrap();

    let mut minimum_height = f32::INFINITY;
    let mut previous_sample_tick = 300;
    let sample_ticks = (301..=360).chain((370..=900).step_by(10));
    for sample_tick in sample_ticks {
        for tick in previous_sample_tick + 1..=sample_tick {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        previous_sample_tick = sample_tick;
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let snapshot = gpu
            .read_snapshot_transforms(&device, &queue, (sample_tick % 3) as u8)
            .unwrap();
        minimum_height = minimum_height.min(transform_position(snapshot[wheel as usize]).y);
    }
    let final_snapshot = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
    let final_height = transform_position(final_snapshot[wheel as usize]).y;
    let chassis_rotation =
        bevy_math::Quat::from_array(final_snapshot[fixture.chassis as usize].rotation);
    assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
    assert!(
        minimum_height >= 0.44,
        "struck wheel crossed too far into the ground: {minimum_height} m"
    );
    assert!(
        final_height >= 0.495,
        "struck wheel remained in the ground at {final_height} m"
    );
    assert!((chassis_rotation * Vec3::Y).dot(Vec3::Y) > 0.9);
}

#[test]
fn articulated_car_wall_impact_remains_bounded_and_decays() {
    let fixture = articulated_car_fixture();
    let Some((device, queue)) = test_device() else {
        return;
    };
    let gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        &fixture.creation,
        GpuPhysicsConfig {
            mechanism_self_collisions: false,
            ..Default::default()
        },
    )
    .unwrap();
    let chassis_position = fixture.creation.compounds[fixture.chassis as usize].root_translation;
    let chassis_mass = fixture.creation.compounds[fixture.chassis as usize]
        .mass_properties
        .mass;
    gpu.apply_impulse(
        &device,
        &queue,
        fixture.chassis,
        chassis_position,
        Vec3::NEG_Z * chassis_mass * 0.35,
    )
    .unwrap();

    let mut maximum_sampled_speed = 0.0_f32;
    let mut maximum_sampled_bearing_speed = 0.0_f32;
    for sample_tick in (30..=1_200).step_by(30) {
        for tick in sample_tick - 29..=sample_tick {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let current = gpu
            .read_snapshot_transforms(&device, &queue, (sample_tick % 3) as u8)
            .unwrap();
        let previous = gpu
            .read_snapshot_transforms(&device, &queue, ((sample_tick - 1) % 3) as u8)
            .unwrap();
        for &body in &fixture.dynamic_bodies {
            let position = transform_position(current[body as usize]);
            assert!(position.is_finite(), "body {body} became non-finite");
            maximum_sampled_speed = maximum_sampled_speed.max(snapshot_speed(
                current[body as usize],
                previous[body as usize],
            ));
        }
        for bearing in &fixture.creation.bearings {
            maximum_sampled_bearing_speed =
                maximum_sampled_bearing_speed.max(bearing_speed(&current, &previous, bearing));
        }
        assert!(
            transform_position(current[fixture.chassis as usize]).y >= 0.74,
            "chassis entered the ground at tick {sample_tick}"
        );
        for &wheel in &fixture.wheel_bodies {
            assert!(
                transform_position(current[wheel as usize]).y >= 0.49,
                "wheel {wheel} entered the ground at tick {sample_tick}"
            );
        }
    }

    let current = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
    let previous = gpu.read_snapshot_transforms(&device, &queue, 2).unwrap();
    let diagnostics = gpu.read_last_tick(&device).unwrap();
    assert_eq!(diagnostics.error_flags, 0);
    assert!(diagnostics.anchor_residual_meters <= mechanic_core::ANCHOR_TOLERANCE_METERS);
    assert!(diagnostics.axis_residual_degrees <= mechanic_core::AXIS_TOLERANCE_DEGREES);
    assert!(
        maximum_sampled_speed < 2.0,
        "wall impact accelerated the car to {maximum_sampled_speed} m/s"
    );
    assert!(
        maximum_sampled_bearing_speed < 10.0,
        "wall impact accelerated a bearing to {maximum_sampled_bearing_speed} rad/s"
    );
    let final_speed = fixture
        .dynamic_bodies
        .iter()
        .map(|&body| snapshot_speed(current[body as usize], previous[body as usize]))
        .fold(0.0_f32, f32::max);
    let final_bearing_speed = fixture
        .creation
        .bearings
        .iter()
        .map(|bearing| bearing_speed(&current, &previous, bearing))
        .fold(0.0_f32, f32::max);
    assert!(
        final_speed < 0.02,
        "wall-impact motion did not decay: final speed {final_speed} m/s"
    );
    assert!(
        final_bearing_speed < 0.02,
        "wall-impact bearing motion did not decay: final speed {final_bearing_speed} rad/s"
    );
}

#[test]
fn external_impulse_drives_a_bearing_coordinate() {
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

fn run_ticks(
    creation: &mechanic_core::CompiledCreation,
    ticks: u64,
    collisions_enabled: bool,
) -> Option<(Vec<crate::GpuTransform>, crate::GpuTickReadback)> {
    let (device, queue) = test_device()?;
    let gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        creation,
        GpuPhysicsConfig {
            collisions_enabled,
            ground_plane_enabled: true,
            mechanism_self_collisions: true,
            solver_iterations: 16,
        },
    )
    .ok()?;
    for tick in 1..=ticks {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    let readback = gpu.read_last_tick(&device).ok()?;
    let snapshot = gpu
        .read_snapshot_transforms(&device, &queue, (ticks % 3) as u8)
        .ok()?;
    Some((snapshot, readback))
}

#[test]
fn terrain_triangles_contact_cuboids_convex_parts_and_cylinders_on_slopes_and_walls() {
    let (device, queue) = test_device().expect("terrain collider regression requires an adapter");
    let mut cylinder = ConstructionGraph::new();
    cylinder
        .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
            CylinderDimensions::new(1.0, 0.0, 1.0).unwrap(),
            BuildPose::from_half_grid(IVec3::new(0, 3, 0), GridRotation::default()),
        )))
        .unwrap();
    let shapes = [
        material_cube(ConstructionMaterial::Steel, 3),
        shaped_wedge(3),
        cylinder.compile().unwrap(),
    ];
    for (shape, creation) in shapes.iter().enumerate() {
        for angle in [
            0.0,
            std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_2,
        ] {
            let mut terrain = super::terrain::tests::chunk();
            let rotation = bevy_math::Quat::from_rotation_z(angle);
            for vertex in &mut terrain.vertices {
                *vertex = (rotation * Vec3::from_array(*vertex)).to_array();
            }
            let mut gpu = GpuPhysics::new_with_config(
                &device,
                &queue,
                creation,
                GpuPhysicsConfig {
                    ground_plane_enabled: false,
                    ..Default::default()
                },
            )
            .unwrap();
            gpu.write_terrain_chunks(&device, &queue, [&terrain], bevy_math::DVec3::ZERO)
                .unwrap();
            gpu.dispatch_tick(&device, &queue, 1);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let readback = gpu.read_last_tick(&device).unwrap();
            assert_eq!(readback.error_flags, 0, "shape {shape}, angle {angle}");
            assert!(
                readback.active_contact_count > 0,
                "shape {shape}, angle {angle}: {readback:?}"
            );
        }
    }
}

#[test]
fn terrain_contacts_enter_the_fused_articulated_solver() {
    let (device, queue) =
        test_device().expect("terrain articulated regression requires an adapter");
    let fixture = articulated_car_fixture();
    let mut gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        &fixture.creation,
        GpuPhysicsConfig {
            ground_plane_enabled: false,
            mechanism_self_collisions: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        gpu.solver_route(),
        super::GpuSolverRoute::FusedSmallMechanism
    );
    let mut terrain = super::terrain::tests::chunk();
    terrain.origin = mechanic_world::WorldPosition(bevy_math::DVec3::Y * 0.3);
    gpu.write_terrain_chunks(&device, &queue, [&terrain], bevy_math::DVec3::ZERO)
        .unwrap();
    gpu.dispatch_tick(&device, &queue, 1);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let readback = gpu.read_last_tick(&device).unwrap();
    assert_eq!(readback.error_flags, 0, "{readback:?}");
    assert!(readback.active_contact_count > 0, "{readback:?}");
    assert!(readback.executed_solver_sweeps > 0, "{readback:?}");
}

#[test]
fn rigid_terrain_impact_bounds_hold_for_convex_parts_and_cylinders() {
    let (device, queue) = test_device().expect("terrain impact regression requires an adapter");
    let mut cylinder = ConstructionGraph::new();
    cylinder
        .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
            CylinderDimensions::new(1.0, 0.0, 1.0).unwrap(),
            BuildPose::from_half_grid(IVec3::new(0, 5, 0), GridRotation::default()),
        )))
        .unwrap();
    for (shape, mut creation) in [shaped_wedge(5), cylinder.compile().unwrap()]
        .into_iter()
        .enumerate()
    {
        for collider in &mut creation.colliders {
            collider.material_properties.restitution = 0.0;
            collider.material_properties.youngs_modulus_pa = 200.0e9;
        }
        for speed in [1.0, 5.0, 20.0] {
            let mut gpu = GpuPhysics::new_with_config(
                &device,
                &queue,
                &creation,
                GpuPhysicsConfig {
                    ground_plane_enabled: false,
                    ..Default::default()
                },
            )
            .unwrap();
            gpu.write_terrain_chunks(
                &device,
                &queue,
                [&super::terrain::tests::rigid_chunk()],
                bevy_math::DVec3::ZERO,
            )
            .unwrap();
            gpu.enable_async_readback();
            gpu.apply_impulse(
                &device,
                &queue,
                0,
                creation.compounds[0].root_translation,
                Vec3::NEG_Y * speed * creation.compounds[0].mass_properties.mass,
            )
            .unwrap();
            for tick in 1..=120 {
                gpu.dispatch_tick(&device, &queue, tick);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let sample = gpu.poll_tick_readback(&device).unwrap().unwrap();
                assert_eq!(
                    sample.diagnostics.error_flags, 0,
                    "shape {shape}, speed {speed}, tick {tick}"
                );
                for collider in &creation.colliders {
                    let pose = sample.transforms[collider.compound_index as usize];
                    let rotation = bevy_math::Quat::from_array(pose.rotation);
                    let minimum = match &collider.shape {
                        mechanic_core::ColliderShape::Cuboid {
                            local_rotation,
                            half_extents,
                        } => {
                            let center = Vec3::from_slice(&pose.position[..3])
                                + rotation * collider.local_center;
                            center.y
                                - ((rotation * *local_rotation).inverse() * Vec3::Y)
                                    .abs()
                                    .dot(*half_extents)
                        }
                        mechanic_core::ColliderShape::Convex(convex) => convex
                            .vertices
                            .iter()
                            .map(|vertex| pose.position[1] + (rotation * *vertex).y)
                            .fold(f32::INFINITY, f32::min),
                    };
                    let limit = if tick > 100 { 0.002 } else { 0.005 };
                    assert!(
                        minimum >= -limit,
                        "shape {shape}, speed {speed}, tick {tick}: bottom {minimum}"
                    );
                }
            }
        }
    }
}

#[test]
fn terrain_recovery_preserves_articulated_and_suspension_constraints() {
    use mechanic_core::{ShockSpec, SpringSpec, SuspensionSpec};
    let (device, queue) =
        test_device().expect("articulated terrain regression requires an adapter");
    let suspension = SuspensionSpec::new(
        Some(SpringSpec::default()),
        Some(ShockSpec::default()),
        None,
    )
    .unwrap();
    let scenes = [
        articulated_car_fixture().creation,
        suspension_test_creation_with_anchor(suspension, true, 1, false),
        suspension_test_creation_with_anchor(suspension, true, 65, false),
    ];
    for (scene, mut creation) in scenes.into_iter().enumerate() {
        for collider in &mut creation.colliders {
            collider.material_properties.restitution = 0.0;
            collider.material_properties.youngs_modulus_pa = 200.0e9;
        }
        let mut gpu = GpuPhysics::new_with_config(
            &device,
            &queue,
            &creation,
            GpuPhysicsConfig {
                ground_plane_enabled: false,
                mechanism_self_collisions: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            gpu.solver_route(),
            if scene == 2 {
                super::GpuSolverRoute::General
            } else {
                super::GpuSolverRoute::FusedSmallMechanism
            }
        );
        let mut terrain = super::terrain::tests::rigid_chunk();
        for vertex in &mut terrain.vertices {
            vertex[0] *= 100.0;
            vertex[2] *= 100.0;
        }
        gpu.write_terrain_chunks(&device, &queue, [&terrain], bevy_math::DVec3::ZERO)
            .unwrap();
        gpu.enable_async_readback();
        for (body, compound) in creation.compounds.iter().enumerate() {
            if compound.is_static {
                continue;
            }
            gpu.apply_impulse(
                &device,
                &queue,
                u32::try_from(body).unwrap(),
                compound.root_translation,
                Vec3::NEG_Y * 20.0 * compound.mass_properties.mass,
            )
            .unwrap();
        }
        for tick in 1..=120 {
            gpu.dispatch_tick(&device, &queue, tick);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let sample = gpu.poll_tick_readback(&device).unwrap().unwrap();
            assert_eq!(
                sample.diagnostics.error_flags, 0,
                "scene {scene}, tick {tick}: {:?}",
                sample.diagnostics
            );
            let minimum = minimum_dynamic_collider_height(&creation, &sample.transforms);
            assert!(
                minimum >= -if tick > 100 { 0.002 } else { 0.005 },
                "scene {scene}, tick {tick}: bottom {minimum}"
            );
        }
    }
}

#[test]
fn terrain_position_recovery_preserves_intentional_restitution() {
    let (device, queue) =
        test_device().expect("terrain restitution regression requires an adapter");
    let mut peaks = Vec::new();
    for restitution in [0.0, 0.8] {
        let mut creation = material_cube(ConstructionMaterial::Steel, 5);
        for collider in &mut creation.colliders {
            collider.material_properties.restitution = restitution;
        }
        let mut gpu = GpuPhysics::new_with_config(
            &device,
            &queue,
            &creation,
            GpuPhysicsConfig {
                ground_plane_enabled: false,
                ..Default::default()
            },
        )
        .unwrap();
        gpu.write_terrain_chunks(
            &device,
            &queue,
            [&super::terrain::tests::rigid_chunk()],
            bevy_math::DVec3::ZERO,
        )
        .unwrap();
        gpu.enable_async_readback();
        gpu.apply_impulse(
            &device,
            &queue,
            0,
            creation.compounds[0].root_translation,
            Vec3::NEG_Y * 5.0 * creation.compounds[0].mass_properties.mass,
        )
        .unwrap();
        let mut peak = f32::NEG_INFINITY;
        for tick in 1..=30 {
            gpu.dispatch_tick(&device, &queue, tick);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let sample = gpu.poll_tick_readback(&device).unwrap().unwrap();
            assert_eq!(sample.diagnostics.error_flags, 0);
            peak = peak.max(sample.velocities[0].linear[1]);
        }
        peaks.push(peak);
    }
    assert!(
        peaks[0] < 0.001,
        "inelastic terrain contact rebounded: {peaks:?}"
    );
    assert!(
        peaks[1] > 1.0,
        "intentional restitution disappeared: {peaks:?}"
    );
}

fn minimum_dynamic_collider_height(
    creation: &mechanic_core::CompiledCreation,
    transforms: &[crate::GpuTransform],
) -> f32 {
    creation
        .colliders
        .iter()
        .filter(|collider| !creation.compounds[collider.compound_index as usize].is_static)
        .map(|collider| {
            let pose = transforms[collider.compound_index as usize];
            let rotation = bevy_math::Quat::from_array(pose.rotation);
            match &collider.shape {
                mechanic_core::ColliderShape::Cuboid {
                    local_rotation,
                    half_extents,
                } => {
                    let center =
                        Vec3::from_slice(&pose.position[..3]) + rotation * collider.local_center;
                    center.y
                        - ((rotation * *local_rotation).inverse() * Vec3::Y)
                            .abs()
                            .dot(*half_extents)
                }
                mechanic_core::ColliderShape::Convex(convex) => convex
                    .vertices
                    .iter()
                    .map(|vertex| pose.position[1] + (rotation * *vertex).y)
                    .fold(f32::INFINITY, f32::min),
            }
        })
        .fold(f32::INFINITY, f32::min)
}

fn material_cube(
    material: ConstructionMaterial,
    center_half_units_y: i32,
) -> mechanic_core::CompiledCreation {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [4; 3],
                BuildPose::from_half_grid(
                    IVec3::new(0, center_half_units_y, 0),
                    GridRotation::default(),
                ),
            )
            .unwrap()
            .with_material(material),
        ))
        .unwrap();
    graph.compile().unwrap()
}

/// A block whose top +z edge is collapsed onto the bottom, dropped from
/// `center_half_units_y`.
fn shaped_wedge(center_half_units_y: i32) -> mechanic_core::CompiledCreation {
    let spec = CuboidSpec::new(
        [4; 3],
        BuildPose::from_half_grid(
            IVec3::new(0, center_half_units_y, 0),
            GridRotation::default(),
        ),
    )
    .unwrap();
    let mut graph = ConstructionGraph::new();
    graph.apply(BuildCommand::Spawn(spec)).unwrap();
    let cells = mechanic_core::part_cells(spec);
    let region = mechanic_core::ShapeRegion::new(
        cells.corner_half_units(IVec3::ZERO, 0),
        cells.counts(),
        ConstructionMaterial::Steel,
    )
    .expect("the block spans at least one cell");
    let mechanic_core::BuildOutcome::RegionAdded(id) =
        graph.apply(BuildCommand::AddRegion(region)).unwrap()
    else {
        panic!("adding a region reports the region it added")
    };
    // Drop the top pair of cage corners on +z a whole cell onto the corners
    // below them, which slopes the whole top face.
    let cell =
        i16::try_from(mechanic_core::POSITION_TICKS_PER_GRID_UNIT).expect("a cell is twenty steps");
    graph
        .apply(BuildCommand::SetRegionVertices {
            region: id,
            vertices: vec![([0, 1, 1], [0, -cell, 0]), ([1, 1, 1], [0, -cell, 0])],
        })
        .expect("collapsing an edge is a legal shape");
    graph.compile().unwrap()
}

#[test]
fn a_shaped_part_compiles_to_convex_collider_rows() {
    let creation = shaped_wedge(8);
    assert!(
        creation
            .colliders
            .iter()
            .any(|collider| !collider.shape.is_cuboid()),
        "shaping must produce convex collider rows, not boxes"
    );
    assert!(
        creation.colliders.len() < 64,
        "fusing should keep the row count small; got {}",
        creation.colliders.len()
    );
}

#[test]
fn a_shaped_wedge_settles_on_the_ground_without_failing() {
    // The whole point of shaping being truthful: the solver has to accept
    // convex rows and bring them to rest like any other body.
    let Some((device, queue)) = test_device() else {
        return;
    };
    let creation = shaped_wedge(8);
    let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
    for tick in 1..=180 {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    assert_eq!(
        gpu.read_last_tick(&device).unwrap().error_flags,
        0,
        "convex colliders must not raise a failure flag"
    );
    let y = gpu
        .read_snapshot_transforms(&device, &queue, (180 % 3) as u8)
        .unwrap()[0]
        .position[1];
    assert!(
        y.is_finite() && y > -0.1,
        "the wedge should rest on the ground rather than sink; y={y}"
    );
    assert!(
        y < 1.0,
        "the wedge should have fallen from its 1 m drop; y={y}"
    );
}

#[test]
fn higher_friction_material_loses_more_sliding_speed() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let slide = |material| {
        let creation = material_cube(material, 4);
        let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
        let mass = creation.compounds[0].mass_properties.mass;
        gpu.apply_impulse(&device, &queue, 0, Vec3::new(0.0, 0.5, 0.0), Vec3::X * mass)
            .unwrap();
        for tick in 1..=90 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        gpu.read_snapshot_transforms(&device, &queue, 0).unwrap()[0].position[0]
    };
    let plastic_distance = slide(ConstructionMaterial::Plastic);
    let concrete_distance = slide(ConstructionMaterial::Concrete);
    assert!(
        concrete_distance < plastic_distance - 0.02,
        "concrete slid {concrete_distance} m while plastic slid {plastic_distance} m",
    );
}

#[test]
fn static_friction_holds_a_sub_threshold_load_while_kinetic_friction_slows_sliding() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let creation = material_cube(ConstructionMaterial::Aluminium, 4);
    let mass = creation.compounds[0].mass_properties.mass;
    let contact_center = Vec3::new(0.0, 0.5, 0.0);
    let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
    for tick in 1..=60 {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    let static_load_impulse =
        mass * mechanic_core::STANDARD_GRAVITY_M_S2_F32 * mechanic_core::TICK_SECONDS_F32 * 0.62;
    for tick in 61..=120 {
        gpu.apply_impulse(
            &device,
            &queue,
            0,
            contact_center,
            Vec3::X * static_load_impulse,
        )
        .unwrap();
        gpu.dispatch_tick(&device, &queue, tick);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let stuck = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap()[0];
    assert!(
        stuck.position[0].abs() < 0.01,
        "sub-threshold static load moved the cube {} m",
        stuck.position[0],
    );

    let sliding = GpuPhysics::new(&device, &queue, &creation).unwrap();
    sliding
        .apply_impulse(&device, &queue, 0, contact_center, Vec3::X * mass * 3.0)
        .unwrap();
    for tick in 1..=2 {
        sliding.dispatch_tick(&device, &queue, tick);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let early = sliding
        .read_snapshot_transforms(&device, &queue, 2)
        .unwrap()[0];
    let initial = sliding
        .read_snapshot_transforms(&device, &queue, 1)
        .unwrap()[0];
    let early_speed = snapshot_speed(early, initial);
    for tick in 3..=20 {
        sliding.dispatch_tick(&device, &queue, tick);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let late = sliding
        .read_snapshot_transforms(&device, &queue, 2)
        .unwrap()[0];
    let previous = sliding
        .read_snapshot_transforms(&device, &queue, 1)
        .unwrap()[0];
    let late_speed = snapshot_speed(late, previous);
    assert!(
        late_speed < early_speed - 0.5,
        "sliding speed did not decay: {early_speed} to {late_speed}"
    );
    assert!(
        late_speed > 0.5,
        "kinetic friction used the static limit: final speed {late_speed}"
    );
}

#[test]
fn higher_rolling_resistance_stops_an_otherwise_identical_cylinder_sooner() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::SpawnCylinder(
            CylinderSpec::new(
                CylinderDimensions::new(1.0, 0.0, 1.0).unwrap(),
                BuildPose::new(IVec3::new(0, 2, 0), GridRotation::new(0, 0, 1)),
            )
            .with_material(ConstructionMaterial::Steel),
        ))
        .unwrap();
    let base = graph.compile().unwrap();
    let roll = |rolling_resistance: f32| {
        let mut creation = base.clone();
        for collider in &mut creation.colliders {
            collider.material_properties.rolling_resistance = rolling_resistance;
        }
        let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
        let mass = creation.compounds[0].mass_properties.mass;
        gpu.apply_impulse(
            &device,
            &queue,
            0,
            Vec3::new(0.0, 0.75, 0.0),
            Vec3::Z * mass * 2.0,
        )
        .unwrap();
        for tick in 1..=120 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
        gpu.read_snapshot_transforms(&device, &queue, 0).unwrap()[0].position[2]
    };
    let low_distance = roll(0.002);
    let high_distance = roll(0.040);
    assert!(
        high_distance < low_distance - 0.05,
        "high rolling resistance travelled {high_distance} m; low travelled {low_distance} m",
    );
}

#[test]
fn flat_disc_stops_sliding_without_long_term_contact_drift() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
            CylinderDimensions::new(1.0, 0.0, 0.5).unwrap(),
            BuildPose::new(IVec3::new(0, 4, 0), GridRotation::default()),
        )))
        .unwrap();
    let creation = graph.compile().unwrap();
    let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
    let mass = creation.compounds[0].mass_properties.mass;
    gpu.apply_impulse(
        &device,
        &queue,
        0,
        Vec3::new(0.35, 0.65, 0.0),
        Vec3::new(1.0, -0.35, 0.2) * mass,
    )
    .unwrap();

    for tick in 1..=600 {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let settled = transform_position(gpu.read_snapshot_transforms(&device, &queue, 0).unwrap()[0]);

    for tick in 601..=1_200 {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let later = transform_position(gpu.read_snapshot_transforms(&device, &queue, 0).unwrap()[0]);
    let tail_drift = Vec3::new(later.x - settled.x, 0.0, later.z - settled.z).length();
    assert!(
        tail_drift < 0.005,
        "flat disc drifted {tail_drift} m after it should have stopped"
    );
    assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
}

#[test]
fn flat_disc_landing_on_another_disc_does_not_gain_energy() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let mut graph = ConstructionGraph::new();
    let dimensions = CylinderDimensions::new(1.0, 0.0, 0.25).unwrap();
    let BuildOutcome::Spawned(lower) = graph
        .apply(BuildCommand::SpawnCylinder(
            CylinderSpec::new(
                dimensions,
                BuildPose::from_half_grid(IVec3::new(0, 1, 0), GridRotation::default()),
            )
            .with_material(ConstructionMaterial::Concrete),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(lower, FaceKind::NegativeY),
            second: FaceRef::ground(),
        }))
        .unwrap();
    graph
        .apply(BuildCommand::SpawnCylinder(
            CylinderSpec::new(
                dimensions,
                BuildPose::new(IVec3::new(0, 4, 0), GridRotation::default()),
            )
            .with_material(ConstructionMaterial::Concrete),
        ))
        .unwrap();
    let creation = graph.compile().unwrap();
    let dynamic_body = creation
        .compounds
        .iter()
        .position(|compound| !compound.is_static)
        .unwrap();
    let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
    let mut maximum_height_after_impact = 0.0_f32;
    for tick in 1..=120 {
        gpu.dispatch_tick(&device, &queue, tick);
        if tick >= 25 {
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let position = transform_position(
                gpu.read_snapshot_transforms(&device, &queue, (tick % 3) as u8)
                    .unwrap()[dynamic_body],
            );
            maximum_height_after_impact = maximum_height_after_impact.max(position.y);
            assert!(position.is_finite(), "stacked disc became non-finite");
            assert!(
                position.x.abs() < 0.25 && position.z.abs() < 0.25,
                "stacked disc was launched sideways to {position:?}"
            );
        }
    }
    assert!(
        maximum_height_after_impact < 0.9,
        "stacked disc rebounded above {maximum_height_after_impact} m"
    );
    assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
}

#[test]
fn lower_modulus_allows_more_transient_penetration_without_failure() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let base = material_cube(ConstructionMaterial::Steel, 16);
    let minimum_height = |youngs_modulus_pa: f32| {
        let mut creation = base.clone();
        for collider in &mut creation.colliders {
            collider.material_properties.youngs_modulus_pa = youngs_modulus_pa;
        }
        let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
        let mut minimum = f32::INFINITY;
        for tick in 1..=60 {
            gpu.dispatch_tick(&device, &queue, tick);
            if tick >= 25 {
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                minimum = minimum.min(
                    gpu.read_snapshot_transforms(&device, &queue, (tick % 3) as u8)
                        .unwrap()[0]
                        .position[1],
                );
            }
        }
        assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
        minimum
    };
    let stiff_height = minimum_height(200.0e9);
    let soft_height = minimum_height(0.01e9);
    assert!(
        soft_height < stiff_height - 1.0e-4,
        "soft contact reached {soft_height} m; stiff contact reached {stiff_height} m",
    );
    assert!(
        soft_height > 0.4,
        "soft contact became unstable at {soft_height} m"
    );
}

#[test]
fn plastic_rebounds_more_than_concrete() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let rebound_height = |material| {
        let creation = material_cube(material, 16);
        let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
        let mut maximum_after_impact = 0.5_f32;
        for tick in 1..=100 {
            gpu.dispatch_tick(&device, &queue, tick);
            if tick >= 40 {
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let y = gpu
                    .read_snapshot_transforms(&device, &queue, (tick % 3) as u8)
                    .unwrap()[0]
                    .position[1];
                maximum_after_impact = maximum_after_impact.max(y);
            }
        }
        maximum_after_impact
    };
    let plastic_height = rebound_height(ConstructionMaterial::Plastic);
    let concrete_height = rebound_height(ConstructionMaterial::Concrete);
    assert!(
        plastic_height > concrete_height + 0.05,
        "plastic rebounded to {plastic_height} m while concrete reached {concrete_height} m",
    );
}

#[test]
fn sub_threshold_ground_contact_settles_without_repeated_bounce() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let creation = material_cube(ConstructionMaterial::Plastic, 4);
    let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
    let mass = creation.compounds[0].mass_properties.mass;
    gpu.apply_impulse(
        &device,
        &queue,
        0,
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::NEG_Y * mass * 0.5,
    )
    .unwrap();
    let mut tail = Vec::new();
    for tick in 1..=120 {
        gpu.dispatch_tick(&device, &queue, tick);
        if tick >= 100 {
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            tail.push(
                gpu.read_snapshot_transforms(&device, &queue, (tick % 3) as u8)
                    .unwrap()[0]
                    .position[1],
            );
        }
    }
    let movement = tail
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        movement < 1.0e-3,
        "settled contact moved {movement} m per tick"
    );
}

#[test]
fn flat_cylinder_face_stays_supported_above_ground() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
            CylinderDimensions::new(1.0, 0.0, 0.5).unwrap(),
            BuildPose::new(IVec3::new(0, 4, 0), GridRotation::default()),
        )))
        .unwrap();
    let creation = graph.compile().unwrap();
    let Some((snapshot, readback)) = run_ticks(&creation, 120, true) else {
        return;
    };
    assert_eq!(readback.error_flags, 0);
    let position = transform_position(snapshot[0]);
    let rotation = bevy_math::Quat::from_array(snapshot[0].rotation);
    assert!(
        position.y >= 0.245,
        "flat cylinder sank through its 0.25 m half-length to {} m",
        position.y
    );
    assert!((rotation * Vec3::Y).dot(Vec3::Y) > 0.98);
}

#[test]
fn grounded_offset_pendulum_swings_without_detaching() {
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
fn freely_falling_hinge_has_no_gravity_induced_relative_rotation() {
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

fn cylinder_drop_creation(
    inner_diameter: f32,
    drop_x_half_units: i32,
) -> mechanic_core::CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let support_spec = CuboidSpec::new(
        [4, 4, 4],
        BuildPose::new(IVec3::new(8, 2, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(support) = graph.apply(BuildCommand::Spawn(support_spec)).unwrap()
    else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(support, FaceKind::NegativeY),
            second: FaceRef::ground(),
        }))
        .unwrap();
    let cylinder_spec = CylinderSpec::new(
        CylinderDimensions::new(1.0, inner_diameter, 1.0).unwrap(),
        BuildPose::new(IVec3::new(0, 12, 0), GridRotation::default()),
    );
    let BuildOutcome::Spawned(cylinder) = graph
        .apply(BuildCommand::SpawnCylinder(cylinder_spec))
        .unwrap()
    else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec {
            first: support,
            second: cylinder,
        }))
        .unwrap();
    let drop_spec = CuboidSpec::new(
        [1, 1, 1],
        BuildPose::from_half_grid(
            IVec3::new(drop_x_half_units, 40, 0),
            GridRotation::default(),
        ),
    )
    .unwrap();
    graph.apply(BuildCommand::Spawn(drop_spec)).unwrap();
    graph.compile().unwrap()
}

#[test]
fn gpu_cylinder_bore_is_passable_and_annular_material_blocks_motion() {
    let hollow = cylinder_drop_creation(0.60, 0);
    let solid = cylinder_drop_creation(0.0, 0);
    let annular = cylinder_drop_creation(0.60, 3);
    let Some((hollow_snapshot, hollow_readback)) = run_ticks(&hollow, 60, true) else {
        return;
    };
    let Some((solid_snapshot, solid_readback)) = run_ticks(&solid, 60, true) else {
        return;
    };
    let Some((annular_snapshot, annular_readback)) = run_ticks(&annular, 60, true) else {
        return;
    };
    assert_eq!(hollow_readback.error_flags, 0);
    assert_eq!(solid_readback.error_flags, 0);
    assert_eq!(annular_readback.error_flags, 0);
    let falling_body = 1;
    assert!(hollow_snapshot[falling_body].position[1] < 2.5);
    assert!(solid_snapshot[falling_body].position[1] > 3.4);
    assert!(
        annular_snapshot[falling_body].position[1] > 3.35,
        "annular drop reached y={} instead of resting on the ring",
        annular_snapshot[falling_body].position[1]
    );
}

fn pipe_bend_drop_creation(
    inner_diameter: f32,
    drop_x_half_units: i32,
) -> mechanic_core::CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let support_spec = CuboidSpec::new(
        [4, 4, 4],
        BuildPose::new(IVec3::new(8, 2, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(support) = graph.apply(BuildCommand::Spawn(support_spec)).unwrap()
    else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(support, FaceKind::NegativeY),
            second: FaceRef::ground(),
        }))
        .unwrap();
    let bend_spec = PipeBendSpec::new(
        PipeBendDimensions::new(1.0, inner_diameter, 6).unwrap(),
        BuildPose::from_half_grid(IVec3::new(0, 16, 0), GridRotation::default()),
    );
    let BuildOutcome::Spawned(bend) = graph.apply(BuildCommand::SpawnPipeBend(bend_spec)).unwrap()
    else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec {
            first: support,
            second: bend,
        }))
        .unwrap();
    let drop_spec = CuboidSpec::new(
        [1, 1, 1],
        BuildPose::from_half_grid(
            IVec3::new(drop_x_half_units, 32, 0),
            GridRotation::default(),
        ),
    )
    .unwrap();
    graph.apply(BuildCommand::Spawn(drop_spec)).unwrap();
    graph.compile().unwrap()
}

#[test]
fn gpu_pipe_bend_bore_is_passable_and_annular_material_blocks_motion() {
    let hollow = pipe_bend_drop_creation(0.60, 0);
    let solid = pipe_bend_drop_creation(0.0, 0);
    let annular = pipe_bend_drop_creation(0.60, 3);
    let Some((hollow_snapshot, hollow_readback)) = run_ticks(&hollow, 60, true) else {
        return;
    };
    let Some((solid_snapshot, solid_readback)) = run_ticks(&solid, 60, true) else {
        return;
    };
    let Some((annular_snapshot, annular_readback)) = run_ticks(&annular, 60, true) else {
        return;
    };
    assert_eq!(hollow_readback.error_flags, 0);
    assert_eq!(solid_readback.error_flags, 0);
    assert_eq!(annular_readback.error_flags, 0);
    let falling_body = 1;
    assert!(
        hollow_snapshot[falling_body].position[1] < 2.95,
        "centered drop did not enter the bend bore: y={}",
        hollow_snapshot[falling_body].position[1]
    );
    assert!(solid_snapshot[falling_body].position[1] > 3.0);
    assert!(annular_snapshot[falling_body].position[1] > 3.0);
}

#[test]
fn contact_supported_unwelded_tower_drives_welded_arm() {
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
fn double_pendulum_transfers_motion_through_both_bearings() {
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
fn balanced_child_contacts_move_root_without_spurious_joint_motion() {
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
