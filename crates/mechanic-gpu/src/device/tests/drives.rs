//! Driven rotary coordinates: speed, angle, limits, stalls, and reprogramming.

use super::*;

/// Grounded base plus a hinged arm wired to one control block.
///
/// `loaded` extends the arm sideways off the hinge axis so gravity applies a
/// real torque; otherwise the arm's centre of mass sits on the axis.
pub(super) fn driven_arm(
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

pub(super) fn test_coordinate_drive(
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
pub(super) fn joint_angle(
    snapshot: &[crate::GpuTransform],
    creation: &mechanic_core::CompiledCreation,
) -> f32 {
    let bearing = &creation.bearings[0];
    let delta = relative_bearing_rotation(snapshot, bearing);
    let axis = bearing.local_axis_a.normalize();
    2.0 * delta.xyz().dot(axis).atan2(delta.w)
}

pub(super) fn run_driven_arm(
    creation: &mechanic_core::CompiledCreation,
    ticks: u64,
) -> Option<(f32, super::super::GpuTickReadback)> {
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
pub(super) fn limits(max_speed: f32, torque: f32) -> mechanic_core::DriveLimits {
    mechanic_core::DriveLimits::new(max_speed, torque, None).expect("test limits are in range")
}

/// Single-state program holding one target forever.
pub(super) fn holding(target: mechanic_core::DriveTarget) -> mechanic_core::DriveProgram {
    mechanic_core::DriveProgram::new(
        &[mechanic_core::DriveState::new(target).expect("test target is in range")],
        false,
    )
    .expect("a one-state program is valid")
}

#[test]
pub(super) fn speed_state_advances_a_bearing_coordinate_at_its_target_speed() {
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
pub(super) fn negative_target_speed_drives_the_joint_the_other_way() {
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
pub(super) fn max_speed_caps_a_faster_state_target() {
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
pub(super) fn angle_state_reaches_its_target_and_holds_without_overshooting() {
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
pub(super) fn weak_drive_stalls_lifting_a_gravity_loaded_arm() {
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
pub(super) fn driven_coordinate_stops_and_holds_at_its_travel_limit() {
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
pub(super) fn vertical_axis_drive_is_not_zeroed_by_the_sleep_clamp() {
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
pub(super) fn reprogramming_a_wire_changes_the_drive_without_reloading_the_scene() {
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
        Err(super::super::GpuPhysicsError::DriveStateCount {
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
