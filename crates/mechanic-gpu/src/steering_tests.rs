// High-speed return-to-center regression from a garage-built vehicle.

#[test]
#[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
fn high_speed_steering_returns_to_center_after_release() {
    let (device, queue) = test_device().expect("this regression requires a GPU adapter");
    let document: mechanic_core::CreationDocument =
        ron::from_str(include_str!("../tests/fixtures/front_steered_car.mech")).unwrap();
    let creation = document.into_graph().unwrap().graph.compile().unwrap();
    assert_eq!(creation.coordinate_drives.len(), 6);
    for (coordinate, drive) in creation.coordinate_drives[..2].iter().enumerate() {
        assert_eq!(drive.mode, DriveMode::Angle);
        let torque = drive.source_a_max_acceleration
            * creation.loop_topology.coordinate_axis_inertia[coordinate];
        assert!((torque - ServoSpec::STALL_TORQUE_NEWTON_METERS).abs() < 0.01);
    }
    let chassis = creation
        .compounds
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.mass_properties.mass.total_cmp(&b.mass_properties.mass))
        .unwrap()
        .0;
    let wheel_bodies = creation
        .bearings
        .iter()
        .filter(|bearing| bearing.coordinate_index.unwrap() >= 2)
        .map(|bearing| bearing.compound_b)
        .collect::<Vec<_>>();
    let pipelines = super::GpuPhysicsPipelines::new();
    for turn in [-30.0_f32, 30.0] {
        let gpu = GpuPhysics::new_with_pipelines(
            &device,
            &queue,
            &creation,
            GpuPhysicsConfig {
                mechanism_self_collisions: false,
                ..Default::default()
            },
            &pipelines,
        )
        .unwrap();
        for tick in 1..=180 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        let start = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
        let speed = 50.0 / 3.6;
        let wheel_speed = speed / 0.45;
        let velocities = creation
            .compounds
            .iter()
            .enumerate()
            .map(|(body, compound)| crate::GpuVelocity {
                linear: if compound.is_static {
                    [0.0; 4]
                } else {
                    [-speed, 0.0, 0.0, 0.0]
                },
                angular: if wheel_bodies.contains(&(body as u32)) {
                    [0.0, 0.0, wheel_speed, 0.0]
                } else {
                    [0.0; 4]
                },
            })
            .collect::<Vec<_>>();
        gpu.write_body_states(&queue, &start, &velocities).unwrap();
        let mut drives = creation
            .coordinate_drives
            .iter()
            .copied()
            .map(crate::GpuMechanismDrive::from)
            .collect::<Vec<_>>();
        for bearing in &creation.bearings {
            let coordinate = bearing.coordinate_index.unwrap() as usize;
            if coordinate < 2 {
                continue;
            }
            // Same two gas engines, shared over four driven wheel coordinates.
            let acceleration = 3000.0 / creation.loop_topology.coordinate_axis_inertia[coordinate];
            drives[coordinate] = crate::GpuMechanismDrive::from(CoordinateDrive {
                mode: DriveMode::Speed,
                target_speed: bearing
                    .local_axis_a
                    .cross(Vec3::Y)
                    .dot(Vec3::NEG_X)
                    .signum()
                    * wheel_speed,
                max_speed: mechanic_core::rpm_to_rad_s(EngineKind::Gas.no_load_rpm()),
                max_acceleration: acceleration,
                source_b_max_acceleration: acceleration,
                source_b_no_load_speed: mechanic_core::rpm_to_rad_s(EngineKind::Gas.no_load_rpm()),
                ..CoordinateDrive::default()
            });
        }
        gpu.write_mechanism_drives(&queue, &drives).unwrap();
        let mut previous = transform_position(start[chassis]);
        let mut max_return_error = 0.0_f32;
        for tick in 181..=720 {
            if tick == 241 || tick == 421 {
                for drive in &mut drives[..2] {
                    drive.target_angle = if tick == 241 { turn.to_radians() } else { 0.0 };
                }
                gpu.write_mechanism_drives(&queue, &drives).unwrap();
            }
            gpu.dispatch_tick(&device, &queue, tick);
            if tick % 30 != 0 {
                continue;
            }
            let snapshot = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
            let position = transform_position(snapshot[chassis]);
            let actual_kmh = (position - previous).length() * 2.0 * 3.6;
            previous = position;
            let up = bevy_math::Quat::from_array(snapshot[chassis].rotation) * Vec3::Y;
            assert!(up.y > 0.98, "vehicle tipped at tick {tick}");
            assert!(
                (position.y - start[chassis].position[1]).abs() < 0.15,
                "vehicle bounced at tick {tick}"
            );
            assert!(
                actual_kmh > 30.0,
                "regression must stay at driving speed: {actual_kmh}"
            );
            for &wheel in &wheel_bodies {
                assert!(snapshot[wheel as usize].position[1] > 0.42);
            }
            let angles = creation.bearings[..2]
                .iter()
                .map(|bearing| {
                    let relative = relative_bearing_rotation(&snapshot, bearing);
                    (2.0 * relative.xyz().dot(bearing.local_axis_a).atan2(relative.w)).to_degrees()
                })
                .collect::<Vec<_>>();
            if (330..=420).contains(&tick) {
                for angle in &angles {
                    assert!(
                        (angle - turn).abs() < 5.0,
                        "turn {turn}, tick {tick}, angle {angle}"
                    );
                }
            }
            if tick >= 540 {
                for angle in &angles {
                    max_return_error = max_return_error.max(angle.abs());
                }
            }
            let diagnostics = gpu.read_last_tick(&device).unwrap();
            assert_eq!(diagnostics.error_flags, 0);
            assert_eq!(diagnostics.planned_solver_sweeps, 8);
            assert_eq!(diagnostics.executed_solver_sweeps, 8);
            eprintln!(
                "turn={turn} tick={tick} speed={actual_kmh:.1} angles={angles:?} sweeps={}",
                diagnostics.executed_solver_sweeps
            );
        }
        assert!(
            max_return_error < 2.0,
            "return error {max_return_error} degrees"
        );
    }
}
