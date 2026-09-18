//! Articulated cars: landing, steering, drive, and impacts.

use super::*;

pub(super) struct ArticulatedCarFixture {
    pub(super) creation: mechanic_core::CompiledCreation,
    pub(super) chassis: u32,
    pub(super) dynamic_bodies: Vec<u32>,
    pub(super) wheel_bodies: Vec<u32>,
}

pub(super) fn articulated_car_fixture() -> ArticulatedCarFixture {
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

pub(super) fn pipe_bend_suspension_car_fixture() -> ArticulatedCarFixture {
    // The overshoot-only fixture models a heavy hydraulic steering rack.
    // Powered steering coverage below uses the production Servo torque.
    pipe_bend_suspension_car_fixture_with_steering_torque(64_000.0)
}

pub(super) fn pipe_bend_suspension_car_fixture_with_steering_torque(
    steering_torque: f32,
) -> ArticulatedCarFixture {
    pipe_bend_car_fixture(steering_torque, false)
}

#[expect(clippy::too_many_lines)]
pub(super) fn pipe_bend_car_fixture(
    steering_torque: f32,
    front_steering_only: bool,
) -> ArticulatedCarFixture {
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

pub(super) fn rigid_axle_car_fixture() -> ArticulatedCarFixture {
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
pub(super) fn full_cylinder_ground_contacts_match_visual_wheel_radius() {
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

pub(super) fn transform_position(transform: crate::GpuTransform) -> Vec3 {
    Vec3::new(
        transform.position[0],
        transform.position[1],
        transform.position[2],
    )
}

pub(super) fn snapshot_speed(current: crate::GpuTransform, previous: crate::GpuTransform) -> f32 {
    (transform_position(current) - transform_position(previous)).length() * 60.0
}

pub(super) fn bearing_speed(
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
pub(super) fn articulated_car_drop_settles_without_drift_or_ground_penetration() {
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
pub(super) fn curved_suspension_car_lands_without_contact_correction_launch() {
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
pub(super) fn steering_servos_reach_angle_without_overshooting() {
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
#[expect(clippy::too_many_lines)]
pub(super) fn production_servos_hold_steering_under_first_gear_gas_drive() {
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
#[expect(clippy::too_many_lines)]
pub(super) fn front_steered_car_turns_through_ground_friction() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let fixture = pipe_bend_car_fixture(ServoSpec::STALL_TORQUE_NEWTON_METERS, true);
    let pipelines = super::super::GpuPhysicsPipelines::new();
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

#[test]
#[expect(clippy::too_many_lines)]
pub(super) fn sustained_gas_drive_stays_forward_with_bounded_longitudinal_slip() {
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
pub(super) fn struck_articulated_wheel_recovers_without_crossing_ground() {
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
pub(super) fn articulated_car_wall_impact_remains_bounded_and_decays() {
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
