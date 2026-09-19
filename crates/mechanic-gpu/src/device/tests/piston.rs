//! Pistons: programmed extension, hold under load, and full-stroke stops.

use super::*;

/// A one-metre base carrying a collapsed 2 x 4 piston, bare or with a block on its head.
fn piston_test_creation(side: bool, grounded: bool, bare: bool) -> mechanic_core::CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let mut spawn = |dimensions, ticks| {
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
    let support = spawn([4, 4, 4], IVec3::new(0, 200, 0));
    let (mount, axis, head, face) = if side {
        (
            mechanic_core::PistonMount::Side {
                mount_normal: Vec3::Y,
            },
            Vec3::X,
            IVec3::new(150, 450, 0),
            FaceKind::NegativeX,
        )
    } else {
        (
            mechanic_core::PistonMount::End,
            Vec3::Y,
            IVec3::new(0, 650, 0),
            FaceKind::NegativeY,
        )
    };
    let load = (!bare).then(|| spawn([1, 1, 1], head));
    if grounded {
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(support, FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();
    }
    let source = FaceRef::part(support, FaceKind::PositiveY);
    let anchor = Vec3::new(0.0, 1.0, 0.0);
    let kind = mechanic_core::JointKind::Piston(mechanic_core::Piston {
        dimensions: mechanic_core::PistonDimensions::new(2, 4).unwrap(),
        mount,
    });
    graph
        .apply(BuildCommand::AddBearing(load.map_or_else(
            || BearingSpec::bare(source, anchor, axis, kind),
            |load| {
                BearingSpec::new(source, FaceRef::part(load, face), anchor, axis).with_kind(kind)
            },
        )))
        .unwrap();
    graph.compile().unwrap()
}

fn piston_drive(mode: DriveMode, target: f32) -> CoordinateDrive {
    CoordinateDrive {
        mode,
        target_speed: if mode == DriveMode::Speed {
            target
        } else {
            0.0
        },
        target_angle: if mode == DriveMode::Angle {
            target
        } else {
            0.0
        },
        max_speed: 2.0,
        max_acceleration: 100.0,
        source_a_max_acceleration: 100.0,
        source_a_no_load_speed: 2.0,
        source_b_max_acceleration: 0.0,
        source_b_no_load_speed: 0.0,
        min_angle: 0.0,
        max_angle: 2.0,
    }
}

#[test]
fn piston_extends_to_programmed_lengths_and_holds_its_load_against_gravity() {
    let (device, queue) = test_device().expect("piston GPU regression requires an adapter");
    let mut creation = piston_test_creation(false, true, false);
    creation.coordinate_drives[0] = piston_drive(DriveMode::Angle, 1.0);
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
    for (phase, target) in [(0, 1.0), (1, 0.25), (2, 0.0)] {
        gpu.write_mechanism_drives(
            &queue,
            &[crate::GpuMechanismDrive::from(piston_drive(
                DriveMode::Angle,
                target,
            ))],
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
fn unpowered_piston_stays_collapsed_under_its_load() {
    let (device, queue) = test_device().expect("piston GPU regression requires an adapter");
    let creation = piston_test_creation(false, true, false);
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
    }
    let (position, _) = linear_test_snapshot(&gpu, &device, &queue, &creation, 90);
    assert!(
        position.abs() <= mechanic_core::ANCHOR_TOLERANCE_METERS,
        "the collapsed stop let the head sink to {position}"
    );
}

#[test]
fn side_mounted_piston_under_sustained_power_stops_at_full_stroke() {
    let (device, queue) = test_device().expect("piston GPU regression requires an adapter");
    let mut creation = piston_test_creation(true, true, false);
    creation.coordinate_drives[0] = piston_drive(DriveMode::Speed, 1.5);
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
    for tick in 1..=180 {
        gpu.dispatch_tick(&device, &queue, tick);
        if tick > 120 {
            let (position, _) = linear_test_snapshot(&gpu, &device, &queue, &creation, tick);
            assert!(
                (position - 2.0).abs() <= mechanic_core::ANCHOR_TOLERANCE_METERS,
                "tick {tick}: sustained power did not hold full stroke: {position}"
            );
        }
    }
}

#[test]
fn extending_piston_pushes_a_floating_base_back() {
    let (device, queue) = test_device().expect("piston GPU regression requires an adapter");
    let mut creation = piston_test_creation(true, false, false);
    creation.coordinate_drives[0] = piston_drive(DriveMode::Speed, 1.5);
    let bearing = creation.bearings[0];
    let base = creation.compounds[bearing.compound_a as usize].root_translation;
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
    for tick in 1..=60 {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    let (position, snapshot) = linear_test_snapshot(&gpu, &device, &queue, &creation, 60);
    assert!(position > 1.0, "the piston barely extended: {position}");
    let movement = transform_position(snapshot[bearing.compound_a as usize]) - base;
    assert!(
        movement.x < -0.01,
        "the base received no reaction: {movement:?}"
    );
}

#[test]
fn bare_piston_extends_its_own_head_with_collisions_running() {
    let (device, queue) = test_device().expect("piston GPU regression requires an adapter");
    for side in [false, true] {
        let mut creation = piston_test_creation(side, true, true);
        creation.coordinate_drives[0] = piston_drive(DriveMode::Angle, 1.0);
        let gpu =
            GpuPhysics::new_with_config(&device, &queue, &creation, GpuPhysicsConfig::default())
                .unwrap();
        for tick in 1..=180 {
            gpu.dispatch_tick(&device, &queue, tick);
            let (position, _) = linear_test_snapshot(&gpu, &device, &queue, &creation, tick);
            if tick > 120 {
                assert!((position - 1.0).abs() < 0.001, "side {side}: {position}");
            }
        }
    }
}
