//! Materials, friction, restitution, shaped parts, cylinders, and pipes in contact.

use super::*;

#[expect(clippy::too_many_lines)]
pub(super) fn colliding_pipe_mechanism(grounded: bool) -> mechanic_core::CompiledCreation {
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
            .with_kind(mechanic_core::JointKind::Linear(
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
pub(super) fn dense_pipe_contacts_on_bearings_and_a_rail_remain_bounded() {
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
            super::super::wgpu_util::direct_compute_pass(
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

#[test]
pub(super) fn collider_local_ground_planes_support_bodies_at_different_heights() {
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

pub(super) fn material_cube(
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
pub(super) fn shaped_wedge(center_half_units_y: i32) -> mechanic_core::CompiledCreation {
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
pub(super) fn a_shaped_part_compiles_to_convex_collider_rows() {
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
pub(super) fn a_shaped_wedge_settles_on_the_ground_without_failing() {
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
pub(super) fn higher_friction_material_loses_more_sliding_speed() {
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
pub(super) fn static_friction_holds_a_sub_threshold_load_while_kinetic_friction_slows_sliding() {
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
pub(super) fn higher_rolling_resistance_stops_an_otherwise_identical_cylinder_sooner() {
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
pub(super) fn flat_disc_stops_sliding_without_long_term_contact_drift() {
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
pub(super) fn flat_disc_landing_on_another_disc_does_not_gain_energy() {
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
pub(super) fn lower_modulus_allows_more_transient_penetration_without_failure() {
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
pub(super) fn plastic_rebounds_more_than_concrete() {
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
pub(super) fn sub_threshold_ground_contact_settles_without_repeated_bounce() {
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
pub(super) fn flat_cylinder_face_stays_supported_above_ground() {
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

pub(super) fn cylinder_drop_creation(
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
pub(super) fn gpu_cylinder_bore_is_passable_and_annular_material_blocks_motion() {
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

pub(super) fn pipe_bend_drop_creation(
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
pub(super) fn gpu_pipe_bend_bore_is_passable_and_annular_material_blocks_motion() {
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
