use super::*;
use mechanic_core::{BuildCommand, BuildPose, ConstructionGraph, CuboidSpec};

fn cube() -> CompiledCreation {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([4, 4, 4], BuildPose::default()).unwrap(),
        ))
        .unwrap();
    graph.compile().unwrap()
}

#[test]
fn only_an_explicit_gpu_setting_leaves_the_cpu_route() {
    assert_eq!(route_from(None), Route::Cpu);
    assert_eq!(route_from(Some("")), Route::Cpu);
    assert_eq!(route_from(Some("cpu")), Route::Cpu);
    assert_eq!(route_from(Some("nonsense")), Route::Cpu);
    assert_eq!(route_from(Some("gpu")), Route::Gpu);
    assert_eq!(route_from(Some(" GPU ")), Route::Gpu);
}

#[test]
fn a_drive_row_survives_the_round_trip_through_the_uploaded_form() {
    let creation = cube();
    for drive in creation.coordinate_drives.iter().copied().chain([
        CoordinateDrive {
            mode: mechanic_core::DriveMode::Speed,
            target_speed: 2.5,
            target_angle: 0.0,
            max_speed: 7.0,
            max_acceleration: 12.0,
            source_a_max_acceleration: 12.0,
            source_a_no_load_speed: 7.0,
            source_b_max_acceleration: 3.0,
            source_b_no_load_speed: 1.5,
            min_angle: f32::NEG_INFINITY,
            max_angle: f32::INFINITY,
        },
        CoordinateDrive {
            mode: mechanic_core::DriveMode::Angle,
            target_speed: 0.0,
            target_angle: -0.75,
            max_speed: 3.0,
            max_acceleration: f32::INFINITY,
            source_a_max_acceleration: f32::INFINITY,
            source_a_no_load_speed: 3.0,
            source_b_max_acceleration: 0.0,
            source_b_no_load_speed: 0.0,
            min_angle: -1.5,
            max_angle: 1.5,
        },
    ]) {
        let row = GpuMechanismDrive::from(drive);
        assert_eq!(CoordinateDrive::from(row), drive);
    }
}

// A hinged pair: one free root body plus one joint coordinate.
fn hinge() -> CompiledCreation {
    use mechanic_core::{BearingSpec, BuildOutcome, FaceKind, FaceRef, GridRotation, PartId};
    let mut graph = ConstructionGraph::new();
    let mut spawn = |ticks: bevy::math::IVec3| {
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 4, 4],
                    BuildPose::from_position_ticks(ticks, GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            panic!("spawn expected");
        };
        part as PartId
    };
    let root = spawn(bevy::math::IVec3::ZERO);
    let tip = spawn(bevy::math::IVec3::X * 400);
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(root, FaceKind::PositiveX),
            FaceRef::part(tip, FaceKind::NegativeX),
            bevy::math::Vec3::X * 0.5,
            bevy::math::Vec3::X,
        )))
        .unwrap();
    graph.compile().unwrap()
}

#[test]
fn seeding_covers_every_generalized_row_whatever_order_bodies_are_compiled_in() {
    // Velocity rows are assigned in preorder, so the last body's range is not
    // the row count: a root compiled after its child sizes it six rows short.
    let mut creation = hinge();
    creation.dynamics.body_velocities.reverse();
    let transforms = vec![
        GpuTransform {
            position: [0.0, 1.0, 0.0, 0.0],
            rotation: bevy::math::Quat::IDENTITY.to_array(),
        };
        creation.compounds.len()
    ];
    let velocities = vec![
        GpuVelocity {
            linear: [0.5, -1.5, 0.25, 0.0],
            angular: [0.1, 0.2, 0.3, 0.0],
        };
        creation.compounds.len()
    ];
    let coordinates = vec![
        GpuMechanismCoordinate {
            position: 0.25,
            velocity: -0.75,
        };
        creation.dynamics.coordinate_velocities.len()
    ];
    assert_eq!(coordinates.len(), 1);

    let state = machine_state(&creation, &transforms, &velocities, &coordinates).unwrap();

    assert_eq!(
        state.velocities.len(),
        creation.dynamics.elimination_parent.len()
    );
    let row = creation.dynamics.coordinate_velocities[0];
    assert!((state.velocities[row] + 0.75).abs() < 1e-6);
    let root = creation
        .dynamics
        .body_velocities
        .iter()
        .find(|rows| rows.len() == 6)
        .unwrap()
        .clone();
    assert!((state.velocities[root.start + 1] + 1.5).abs() < 1e-6);
    assert!((state.velocities[root.start + 5] - 0.3).abs() < 1e-6);
}

#[test]
fn seeding_reads_root_velocity_rows_and_joint_rates() {
    let creation = cube();
    let transforms = vec![GpuTransform {
        position: [1.0, 2.0, 3.0, 0.0],
        rotation: bevy::math::Quat::IDENTITY.to_array(),
    }];
    let velocities = vec![GpuVelocity {
        linear: [0.5, -1.5, 0.25, 0.0],
        angular: [0.1, 0.2, 0.3, 0.0],
    }];
    let state = machine_state(&creation, &transforms, &velocities, &[]).unwrap();

    assert_eq!(state.poses.len(), 1);
    assert!((state.poses[0].position.y - 2.0).abs() < 1e-6);
    assert_eq!(state.velocities.len(), 6);
    assert!((state.velocities[1] + 1.5).abs() < 1e-6);
    assert!((state.velocities[5] - 0.3).abs() < 1e-6);

    // A state from another creation is refused, never silently padded.
    assert!(machine_state(&creation, &[], &velocities, &[]).is_err());
}

// A flat two-triangle floor, in the mesh form the world streams.
fn floor() -> mechanic_world::TerrainMeshChunk {
    use mechanic_world::{
        TerrainIndexGroups, TerrainMaterial, TerrainMeshChunk, TerrainNodeId,
        TerrainTriangleGroupMask, TriangleBvh, TriangleBvhNode, TriangleBvhTriangle, WorldBounds,
        WorldPosition,
    };
    let reach = 8.0_f32;
    let bounds = WorldBounds {
        minimum: WorldPosition(DVec3::new(-8.0, 0.0, -8.0)),
        maximum: WorldPosition(DVec3::new(8.0, 0.0, 8.0)),
    };
    let mut rock = [0.0; TerrainMaterial::COUNT];
    rock[usize::from(TerrainMaterial::Rock.code())] = 1.0;
    TerrainMeshChunk {
        node: TerrainNodeId::ROOT,
        origin: WorldPosition(DVec3::ZERO),
        vertices: vec![
            [-reach, 0.0, -reach],
            [-reach, 0.0, reach],
            [reach, 0.0, reach],
            [reach, 0.0, -reach],
        ],
        normals: vec![[0.0, 1.0, 0.0]; 4],
        index_groups: TerrainIndexGroups {
            regular: vec![0, 1, 2, 0, 2, 3],
            ..Default::default()
        },
        material_weights: vec![rock; 4],
        bounds,
        triangle_bvh: TriangleBvh {
            bounds,
            triangles: vec![
                TriangleBvhTriangle {
                    indices: [0, 1, 2],
                    group_mask: TerrainTriangleGroupMask::REGULAR,
                },
                TriangleBvhTriangle {
                    indices: [0, 2, 3],
                    group_mask: TerrainTriangleGroupMask::REGULAR,
                },
            ],
            nodes: vec![TriangleBvhNode {
                bounds,
                first_triangle: 0,
                triangle_count: 2,
                group_mask: TerrainTriangleGroupMask::REGULAR,
                ..Default::default()
            }],
        },
        generation: 1,
        ..Default::default()
    }
}

fn dropped_cube(height: f32) -> (CompiledCreation, Vec<GpuTransform>, Vec<GpuVelocity>) {
    let creation = cube();
    let transforms = vec![GpuTransform {
        position: [0.0, height, 0.0, 0.0],
        rotation: bevy::math::Quat::IDENTITY.to_array(),
    }];
    let velocities = vec![GpuVelocity {
        linear: [0.0; 4],
        angular: [0.0; 4],
    }];
    (creation, transforms, velocities)
}

#[test]
fn prepared_collision_installs_latest_body_state_and_rejects_another_revision() {
    let (creation, mut transforms, mut velocities) = dropped_cube(0.5);
    let prepared = PreparedRoute::new(&creation, 7).unwrap();
    // The old simulation continues moving while geometry is prepared.
    transforms[0].position = [3.0, 2.0, -1.0, 0.0];
    velocities[0].linear = [0.6, 0.0, 0.0, 0.0];
    let mut route = prepared
        .install(7, 40, &transforms, &velocities, &[])
        .unwrap();
    let published = route.published_state().unwrap();
    assert!(
        bevy::math::Vec4::from_array(published.transforms[0].position)
            .abs_diff_eq(bevy::math::Vec4::from_array(transforms[0].position), 1.0e-6)
    );
    assert!(
        bevy::math::Vec4::from_array(published.velocities[0].linear)
            .abs_diff_eq(bevy::math::Vec4::from_array(velocities[0].linear), 1.0e-6)
    );
    route.publish_terrain([], DVec3::ZERO).unwrap();
    let next = route.step(41, DVec3::ZERO, &[], &[]).unwrap();
    assert!((next.transforms[0].position[0] - 3.01).abs() < 1.0e-5);
    assert!(
        PreparedRoute::new(&creation, 7)
            .unwrap()
            .install(8, 40, &transforms, &velocities, &[])
            .is_err()
    );
    assert!(
        PreparedRoute::new(&creation, 7)
            .unwrap()
            .install(7, 40, &[], &velocities, &[])
            .is_err()
    );
}

#[test]
fn a_cube_published_on_the_floor_settles_and_keeps_publishing_ticks() {
    let (creation, transforms, velocities) = dropped_cube(0.502);
    let mut route = CpuRoute::new(&creation, 7, 0, &transforms, &velocities, &[]).unwrap();

    // Ticking before a cut would drop the cube through the world.
    assert!(!route.is_ready());
    route.publish_terrain([&floor()], DVec3::ZERO).unwrap();
    assert!(route.is_ready());

    let mut published = 0;
    for tick in 1..=30 {
        let completed = route
            .step(tick, gravity(), &[], &[])
            .unwrap_or_else(|message| panic!("tick {tick}: {message}"));
        assert_eq!(completed.transforms.len(), 1);
        assert_eq!(completed.velocities.len(), 1);
        let height = completed.transforms[0].position[1];
        assert!(
            (0.495..=0.503).contains(&height),
            "tick {tick} settled at {height}"
        );
        published += 1;
    }
    assert_eq!(published, 30);
    let resting = route.step(31, gravity(), &[], &[]).unwrap();
    // Settled in velocity too, not merely held in place.
    assert!(
        resting.velocities[0].linear[1].abs() < 1e-2,
        "resting vertical velocity {}",
        resting.velocities[0].linear[1]
    );
    assert!((resting.transforms[0].position[1] - 0.5).abs() < 0.005);
    assert_eq!(route.degraded_ticks(), 0);
}

#[test]
fn construction_publication_keeps_terrain_and_still_applies_remeshes_and_removals() {
    let (creation, transforms, velocities) = dropped_cube(0.502);
    let mut previous = CpuRoute::new(&creation, 7, 0, &transforms, &velocities, &[]).unwrap();
    previous.publish_terrain([&floor()], DVec3::ZERO).unwrap();
    let publication = previous.publication;
    let mut replacement = CpuRoute::new(&creation, 8, 10, &transforms, &velocities, &[]).unwrap();
    replacement.inherit_terrain(&mut previous);
    assert!(replacement.is_ready());
    assert!(!previous.is_ready());
    replacement
        .publish_terrain([&floor()], DVec3::ZERO)
        .unwrap();
    assert_eq!(
        replacement.publication, publication,
        "unchanged terrain must not be rebuilt after an edit"
    );
    let resting = replacement.step(11, gravity(), &[], &[]).unwrap();
    assert!(resting.transforms[0].position[1] > 0.49);

    let mut remeshed = floor();
    remeshed.generation += 1;
    replacement
        .publish_terrain([&remeshed], DVec3::ZERO)
        .unwrap();
    assert_eq!(replacement.publication, publication + 1);
    replacement.publish_terrain([], DVec3::ZERO).unwrap();
    for tick in 12..=31 {
        let state = replacement.step(tick, gravity(), &[], &[]).unwrap();
        if tick == 31 {
            assert!(state.transforms[0].position[1] < 0.4);
        }
    }
}

#[test]
fn terrain_updates_follow_the_chunks_around_the_bodies_without_republishing_unchanged_ones() {
    let (creation, transforms, velocities) = dropped_cube(0.502);
    let mut route = CpuRoute::new(&creation, 7, 0, &transforms, &velocities, &[]).unwrap();
    route.publish_terrain([&floor()], DVec3::ZERO).unwrap();
    let publication = route.publication;

    // The same chunks every frame cost nothing.
    route.publish_terrain([&floor()], DVec3::ZERO).unwrap();
    assert_eq!(route.publication, publication);
    let resting = route.step(1, gravity(), &[], &[]).unwrap();
    assert!(resting.transforms[0].position[1] > 0.49);

    // A chunk that left the cut stops supporting the cube.
    route.publish_terrain([], DVec3::ZERO).unwrap();
    let mut fallen = resting;
    for tick in 2..=20 {
        fallen = route.step(tick, gravity(), &[], &[]).unwrap();
    }
    assert!(fallen.transforms[0].position[1] < 0.4);

    // A remeshed chunk comes back as a new generation.
    let mut remeshed = floor();
    remeshed.generation = 2;
    route.publish_terrain([&remeshed], DVec3::ZERO).unwrap();
    assert!(route.publication > publication + 1);
    assert_eq!(route.chunks.get(&remeshed.node), Some(&2));
}

#[test]
fn a_cube_dropped_on_another_cube_rests_on_it_instead_of_falling_through() {
    use mechanic_core::GridRotation;
    let mut graph = ConstructionGraph::new();
    for ticks in [bevy::math::IVec3::ZERO, bevy::math::IVec3::X * 800] {
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 4, 4],
                    BuildPose::from_position_ticks(ticks, GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
    }
    let creation = graph.compile().unwrap();
    let transforms = [0.5, 1.52]
        .map(|height| GpuTransform {
            position: [0.0, height, 0.0, 0.0],
            rotation: bevy::math::Quat::IDENTITY.to_array(),
        })
        .to_vec();
    let velocities = vec![
        GpuVelocity {
            linear: [0.0; 4],
            angular: [0.0; 4],
        };
        2
    ];
    let mut route = CpuRoute::new(&creation, 7, 0, &transforms, &velocities, &[]).unwrap();
    route.publish_terrain([&floor()], DVec3::ZERO).unwrap();
    let mut last = None;
    for tick in 1..=60 {
        let completed = route
            .step(tick, gravity(), &[], &[])
            .unwrap_or_else(|message| panic!("tick {tick}: {message}"));
        let [lower, upper] = [0, 1].map(|body| completed.transforms[body].position[1]);
        assert!(
            upper - lower >= 0.99,
            "tick {tick}: upper cube sank to {upper} over {lower}"
        );
        last = Some(upper);
    }
    assert!((last.unwrap() - 1.5).abs() < 0.01);
}

#[test]
fn dropped_app_ticks_keep_joint_commands_and_impulses_on_the_next_cpu_step() {
    let creation = hinge();
    let initial = MachineState::at_rest(&creation);
    let transforms = initial
        .poses
        .iter()
        .map(|pose| GpuTransform {
            position: [
                narrow(pose.position.x),
                narrow(pose.position.y) + 5.0,
                narrow(pose.position.z),
                0.0,
            ],
            rotation: pose.rotation.to_array().map(narrow),
        })
        .collect::<Vec<_>>();
    let velocities = vec![
        GpuVelocity {
            linear: [0.0; 4],
            angular: [0.0; 4]
        };
        transforms.len()
    ];
    let coordinates = vec![
        GpuMechanismCoordinate {
            position: 0.0,
            velocity: 0.0
        };
        creation.dynamics.coordinate_velocities.len()
    ];
    let drives = creation
        .coordinate_drives
        .iter()
        .copied()
        .map(GpuMechanismDrive::from)
        .collect::<Vec<_>>();
    assert_eq!(drives.len(), 1);
    let run = |ticks: [u64; 3]| {
        let mut route =
            CpuRoute::new(&creation, 7, 10, &transforms, &velocities, &coordinates).unwrap();
        route.publish_terrain([&floor()], DVec3::ZERO).unwrap();
        for tick in ticks {
            route
                .step(
                    tick,
                    gravity(),
                    &drives,
                    &[GpuExternalImpulse::new(
                        0,
                        bevy::math::Vec3::new(0.0, 5.0, 0.0),
                        bevy::math::Vec3::X * 0.01,
                    )],
                )
                .unwrap();
        }
        assert_eq!(route.machine.snapshot().tick, 3);
        let completed = route.published_state().unwrap();
        assert!(completed.transforms[0].position[1] < transforms[0].position[1]);
        route.machine.snapshot().state_hash()
    };
    assert_eq!(run([11, 12, 13]), run([11, 23, 41]));
}

#[test]
fn a_tick_before_the_publication_is_refused_rather_than_renumbered() {
    let (creation, transforms, velocities) = dropped_cube(0.502);
    let mut route = CpuRoute::new(&creation, 7, 10, &transforms, &velocities, &[]).unwrap();
    route.publish_terrain([&floor()], DVec3::ZERO).unwrap();

    let Err(message) = route.step(9, gravity(), &[], &[]) else {
        panic!("a tick before the publication must be refused");
    };
    assert!(
        message.contains("precedes this CPU publication"),
        "{message}"
    );
    route.step(11, gravity(), &[], &[]).unwrap();
}

#[test]
fn an_unsupported_creation_is_refused_with_a_message_naming_the_route() {
    let message = unsupported(&PhysicsError::InvalidCollision);
    assert!(
        message.contains("the CPU solver cannot run this creation"),
        "{message}"
    );
    assert!(message.contains("MECHANIC_PHYSICS=gpu"), "{message}");
}

#[test]
fn a_closed_loop_creation_publishes_on_the_cpu_route() {
    use bevy::math::{IVec3, Vec3};
    use mechanic_core::{BearingSpec, BuildOutcome, FaceKind, FaceRef, GridRotation};

    // A parallelogram whose coupler's second bearing closes a loop.
    let mut graph = ConstructionGraph::new();
    let mut spawn = |ticks: IVec3, dimensions: [u8; 3]| {
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    dimensions,
                    BuildPose::from_position_ticks(ticks, GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            panic!("spawn expected");
        };
        part
    };
    let height = IVec3::Y * 800;
    let ground = spawn(height, [8, 1, 1]);
    let left = spawn(height + IVec3::new(-350, 250, 100), [1, 6, 1]);
    let right = spawn(height + IVec3::new(350, 250, 100), [1, 6, 1]);
    let coupler = spawn(height + IVec3::new(0, 500, 200), [8, 1, 1]);
    for (source, target, anchor) in [
        (ground, left, Vec3::new(-0.875, 2.0, 0.125)),
        (ground, right, Vec3::new(0.875, 2.0, 0.125)),
        (left, coupler, Vec3::new(-0.875, 3.25, 0.375)),
        (right, coupler, Vec3::new(0.875, 3.25, 0.375)),
    ] {
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(source, FaceKind::PositiveZ),
                FaceRef::part(target, FaceKind::NegativeZ),
                anchor,
                Vec3::Z,
            )))
            .unwrap();
    }
    let creation = graph.compile().unwrap();
    assert_eq!(creation.dynamics.loops.len(), 1);
    let transforms = creation
        .compounds
        .iter()
        .map(|body| GpuTransform {
            position: body.root_translation.extend(0.0).to_array(),
            rotation: body.root_rotation.to_array(),
        })
        .collect::<Vec<_>>();
    let velocities = vec![
        GpuVelocity {
            linear: [0.0; 4],
            angular: [0.0; 4],
        };
        transforms.len()
    ];
    let coordinates = vec![
        GpuMechanismCoordinate {
            position: 0.0,
            velocity: 0.0,
        };
        creation.dynamics.coordinate_velocities.len()
    ];

    let mut route = CpuRoute::new(&creation, 7, 0, &transforms, &velocities, &coordinates).unwrap();
    route.publish_terrain([&floor()], DVec3::ZERO).unwrap();
    let mut completed = None;
    for tick in 1..=60 {
        completed = Some(route.step(tick, gravity(), &[], &[]).unwrap());
    }
    let completed = completed.unwrap();
    assert!(
        completed.transforms[0].position[1] < transforms[0].position[1],
        "the linkage should fall"
    );
}

#[test]
fn a_refused_tick_names_the_tick_and_the_way_back_to_the_gpu() {
    let (creation, transforms, velocities) = dropped_cube(0.502);
    let route = CpuRoute::new(&creation, 7, 0, &transforms, &velocities, &[]).unwrap();
    let message = route.failure(12, &PhysicsError::InvalidCommand);

    assert!(message.contains("refused tick 12"), "{message}");
    assert!(message.contains("MECHANIC_PHYSICS=gpu"), "{message}");
    assert_eq!(route.degraded_ticks(), 0);
}
