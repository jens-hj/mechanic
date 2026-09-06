use super::*;
use mechanic_core::{
    BearingSpec, BuildOutcome, BuildPose, CuboidSpec, FaceKind, FaceRef, GridRotation, WeldFeature,
    WeldSelection,
};

fn spawn(graph: &mut ConstructionGraph, position: IVec3) -> PartId {
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([2; 3], BuildPose::new(position, GridRotation::default())).unwrap(),
        ))
        .unwrap()
    else {
        panic!("spawn expected")
    };
    part
}
fn face(graph: &ConstructionGraph, part: PartId, kind: FaceKind) -> Pick {
    let face = FaceRef::part(part, kind);
    let geometry = crate::builder::try_face_geometry_from_ref(face, Some(graph)).unwrap();
    Pick {
        part,
        face,
        socket: None,
        selection: WeldSelection {
            feature: WeldFeature::Face,
            point: geometry.center,
            normal: geometry.normal,
            tangent: geometry.tangent_u,
        },
    }
}

#[test]
fn socket_picking_and_publication_follow_the_live_destination_frame() {
    for kind in [
        mechanic_core::BearingKind::Rotational,
        mechanic_core::BearingKind::Linear(mechanic_core::LinearBearing {
            dimensions: mechanic_core::LinearBearingDimensions::default(),
            mount_normal: Vec3::Y,
            face: mechanic_core::CarriageFace::Top,
        }),
    ] {
        let mut graph = ConstructionGraph::new();
        let source = spawn(&mut graph, IVec3::new(0, 28, 0));
        let support = spawn(&mut graph, IVec3::new(8, 28, 0));
        let mount = face(&graph, support, FaceKind::PositiveY);
        let socket = crate::PlacedBearing {
            source: mount.face,
            anchor: mount.selection.point,
            axis: if matches!(kind, mechanic_core::BearingKind::Rotational) {
                Vec3::Y
            } else {
                Vec3::X
            },
            dimensions: mechanic_core::BearingDimensions::default(),
            kind,
        };
        let destination_motion =
            ConstructionFrame::new(Vec3::new(0.3, 0.0, 0.2), Quat::from_rotation_y(0.7)).unwrap();
        let creation = graph.compile().unwrap();
        let transforms: Vec<_> = creation
            .compounds
            .iter()
            .map(|body| {
                let authored =
                    ConstructionFrame::new(body.root_translation, body.root_rotation).unwrap();
                pose(if body.source_parts.contains(&support) {
                    destination_motion.compose(authored)
                } else {
                    authored
                })
            })
            .collect();
        let simulation = AppSimulation {
            published_graph: graph.clone(),
            creation: Some(creation),
            transforms: transforms.clone(),
            live_state: Some(crate::LivePhysicsState {
                tick: 0,
                transforms,
                velocities: vec![
                    GpuVelocity {
                        linear: [0.0; 4],
                        angular: [0.0; 4]
                    };
                    2
                ],
                coordinates: vec![],
            }),
            world_revision: Some((0, 0)),
            ..default()
        };
        let ray_point = socket.anchor
            + Vec3::Y
            + if matches!(kind, mechanic_core::BearingKind::Rotational) {
                Vec3::X * 0.08
            } else {
                Vec3::ZERO
            };
        let target = crate::weld_tool::socket::pick(
            &graph,
            &simulation,
            &[socket],
            Ray3d::new(
                destination_motion.point(ray_point),
                Dir3::new(destination_motion.vector(Vec3::NEG_Y)).unwrap(),
            ),
        )
        .unwrap();
        assert_eq!(target.part, support);
        assert!(target.socket.is_some());
        let source_pick = face(&graph, source, FaceKind::PositiveY);
        let alignment =
            mechanic_core::WeldAlignment::new(source_pick.selection, target.selection).unwrap();
        let intent = Intent::relocation(&graph, &source_pick, &target, alignment.initial());
        let world = WorldRuntime::from_world(&mut World::new());
        intent
            .validate(&graph, &simulation, &world, PlacementBounds::GarageBuild)
            .unwrap();
        let staged = intent.stage(&graph, []).unwrap();
        let compiled = staged.compile().unwrap();
        let (poses, _, coordinates) = intent.states(&compiled, &staged, &simulation).unwrap();
        assert_eq!(compiled.compounds.len(), 2);
        assert_eq!(coordinates.len(), 1);
        assert!(coordinates[0].position.abs() < 1.0e-5);
        for (index, body) in compiled.compounds.iter().enumerate() {
            assert!(
                Vec3::from_slice(&poses[index].position[..3])
                    .abs_diff_eq(destination_motion.point(body.root_translation), 1.0e-5,)
            );
        }
    }
}

fn fixture() -> (Intent, AppSimulation, PartId) {
    let mut graph = ConstructionGraph::new();
    let base = spawn(&mut graph, IVec3::ZERO);
    let arm = spawn(&mut graph, IVec3::new(0, 2, 0));
    let destination = spawn(&mut graph, IVec3::new(16, 0, 0));
    let child = spawn(&mut graph, IVec3::new(16, 2, 0));
    let unrelated = spawn(&mut graph, IVec3::new(32, 0, 0));
    for (a, b, x) in [(base, arm, 0.0), (destination, child, 4.0)] {
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(a, FaceKind::PositiveY),
                FaceRef::part(b, FaceKind::NegativeY),
                Vec3::new(x, 0.25, 0.0),
                Vec3::Y,
            )))
            .unwrap();
    }
    let creation = graph.compile().unwrap();
    let destination_motion =
        ConstructionFrame::new(Vec3::new(1.0, 7.0, 1.0), Quat::from_rotation_y(0.4)).unwrap();
    let mut transforms = Vec::new();
    let mut velocities = Vec::new();
    for body in &creation.compounds {
        let (frame, angle) = if body.source_parts.contains(&base) {
            (
                ConstructionFrame::new(Vec3::X * 20.0, Quat::IDENTITY).unwrap(),
                0.0,
            )
        } else if body.source_parts.contains(&arm) {
            (
                ConstructionFrame::new(Vec3::X * 20.0, Quat::IDENTITY).unwrap(),
                1.2,
            )
        } else if body.source_parts.contains(&unrelated) {
            (
                ConstructionFrame::new(Vec3::Y * 20.0, Quat::IDENTITY).unwrap(),
                0.0,
            )
        } else {
            (
                destination_motion,
                if body.source_parts.contains(&child) {
                    0.6
                } else {
                    0.0
                },
            )
        };
        transforms.push(pose(
            ConstructionFrame::new(
                frame.point(body.root_translation),
                frame.rotation() * Quat::from_rotation_y(angle),
            )
            .unwrap(),
        ));
        velocities.push(GpuVelocity {
            linear: [2.0, 0.0, 1.0, 0.0],
            angular: [0.0, 0.3, 0.0, 0.0],
        });
    }
    let coordinates = creation
        .loop_topology
        .tree_bearings
        .iter()
        .map(|id| {
            let source = graph.bearing(*id).unwrap().source.owner;
            GpuMechanismCoordinate {
                position: if source == FaceOwner::Part(base) {
                    1.2
                } else {
                    0.6
                },
                velocity: 0.4,
            }
        })
        .collect();
    let source = face(&graph, arm, FaceKind::PositiveX);
    let target = face(&graph, destination, FaceKind::NegativeX);
    let alignment = mechanic_core::WeldAlignment::new(source.selection, target.selection).unwrap();
    let intent = Intent::relocation(&graph, &source, &target, alignment.initial());
    let simulation = AppSimulation {
        published_graph: graph,
        creation: Some(creation),
        transforms: transforms.clone(),
        live_state: Some(crate::LivePhysicsState {
            tick: 12,
            transforms,
            velocities,
            coordinates,
        }),
        next_tick: 13,
        completed_tick: 12,
        world_revision: Some((0, 0)),
        ..default()
    };
    (intent, simulation, unrelated)
}

#[test]
fn welding_via_an_articulated_arm_uses_defaults_and_destination_motion() {
    let (intent, simulation, unrelated) = fixture();
    let graph = intent.stage(&intent.baseline, []).unwrap();
    let creation = graph.compile().unwrap();
    let (transforms, velocities, coordinates) =
        intent.states(&creation, &graph, &simulation).unwrap();
    let (_, parts, ghost) = intent.preview(&simulation).unwrap();
    let destination_frame = motion(&simulation, intent.destination.part, true).unwrap();
    assert_eq!(parts.len(), 2);
    assert!(ghost.point(intent.source.selection.point).abs_diff_eq(
        destination_frame.point(intent.destination.selection.point),
        1.0e-5
    ));
    let dest_body = simulation
        .creation
        .as_ref()
        .unwrap()
        .part_to_compound
        .iter()
        .find(|(part, _)| *part == intent.destination.part)
        .unwrap()
        .1 as usize;
    let live = simulation.live_state.as_ref().unwrap();
    let center = Vec3::from_slice(&live.transforms[dest_body].position[..3]);
    for (index, body) in creation.compounds.iter().enumerate() {
        if body
            .source_parts
            .iter()
            .any(|part| intent.parts.contains(part))
        {
            let expected = destination_frame.point(body.root_translation);
            assert!(
                Vec3::from_slice(&transforms[index].position[..3]).abs_diff_eq(expected, 1.0e-5)
            );
            assert!(
                Quat::from_array(transforms[index].rotation)
                    .abs_diff_eq(destination_frame.rotation() * body.root_rotation, 1.0e-5)
            );
            let expected_velocity =
                Vec3::new(2.0, 0.0, 1.0) + (Vec3::Y * 0.3).cross(expected - center);
            assert!(
                Vec3::from_slice(&velocities[index].linear[..3])
                    .abs_diff_eq(expected_velocity, 1.0e-5)
            );
        } else if body.source_parts.contains(&unrelated) {
            let old_index = simulation
                .creation
                .as_ref()
                .unwrap()
                .part_to_compound
                .iter()
                .find(|(part, _)| *part == unrelated)
                .unwrap()
                .1 as usize;
            assert_eq!(transforms[index], live.transforms[old_index]);
            assert_eq!(velocities[index], live.velocities[old_index]);
        }
    }
    for (index, bearing) in creation.loop_topology.tree_bearings.iter().enumerate() {
        let source = graph.bearing(*bearing).unwrap().source.owner;
        let source_joint = matches!(source, FaceOwner::Part(part) if intent.parts.contains(&part));
        assert!(
            (coordinates[index].position - if source_joint { 0.0 } else { 0.6 }).abs() < 1.0e-6
        );
        assert!(
            (coordinates[index].velocity - if source_joint { 0.0 } else { 0.4 }).abs() < 1.0e-6
        );
    }
}

#[test]
fn snapshot_obstruction_uses_the_default_source_not_its_live_arm() {
    let (intent, mut simulation, unrelated) = fixture();
    let world = WorldRuntime::from_world(&mut World::new());
    intent
        .validate(
            &intent.baseline,
            &simulation,
            &world,
            PlacementBounds::GarageBuild,
        )
        .unwrap();
    let compiled = simulation.creation.as_ref().unwrap();
    let source_body = compiled
        .part_to_compound
        .iter()
        .find(|(part, _)| *part == intent.source.part)
        .unwrap()
        .1 as usize;
    let blocker = compiled
        .part_to_compound
        .iter()
        .find(|(part, _)| *part == unrelated)
        .unwrap()
        .1 as usize;
    // Obstruct the original articulated source: its default candidate remains clear.
    let source_pose = simulation.live_state.as_ref().unwrap().transforms[source_body];
    simulation.live_state.as_mut().unwrap().transforms[blocker] = source_pose;
    intent
        .validate(
            &intent.baseline,
            &simulation,
            &world,
            PlacementBounds::GarageBuild,
        )
        .unwrap();
    // Now obstruct the ghost while the original source is still far away.
    let frame = motion(&simulation, intent.destination.part, true)
        .unwrap()
        .compose(intent.transform);
    let placed = frame.point(compiled.compounds[source_body].root_translation);
    simulation.live_state.as_mut().unwrap().transforms[blocker] =
        pose(ConstructionFrame::new(placed, frame.rotation()).unwrap());
    assert!(
        intent
            .validate(
                &intent.baseline,
                &simulation,
                &world,
                PlacementBounds::GarageBuild
            )
            .unwrap_err()
            .contains("intersects another body")
    );
}

#[test]
fn history_restores_affected_assemblies_without_rewinding_an_unrelated_body() {
    let (intent, simulation, unrelated) = fixture();
    let mut affected = intent.parts.clone();
    affected.extend(
        intent
            .baseline
            .structural_component(intent.destination.part, [])
            .unwrap()
            .parts(),
    );
    let restore = Restore::capture(
        affected,
        &intent.baseline,
        &simulation,
        &DimensionFreeze::default(),
    )
    .unwrap();
    let graph = intent.stage(&intent.baseline, []).unwrap();
    let creation = graph.compile().unwrap();
    let (mut transforms, velocities, coordinates) =
        intent.states(&creation, &graph, &simulation).unwrap();
    let index = creation
        .part_to_compound
        .iter()
        .find(|(part, _)| *part == unrelated)
        .unwrap()
        .1 as usize;
    transforms[index].position[0] += 5.0;
    let expected_unrelated = transforms[index];
    let after = AppSimulation {
        creation: Some(creation),
        published_graph: graph,
        transforms: transforms.clone(),
        live_state: Some(crate::LivePhysicsState {
            tick: 20,
            transforms,
            velocities,
            coordinates,
        }),
        world_revision: Some((1, 0)),
        ..default()
    };
    let original = intent.baseline.compile().unwrap();
    let (poses, _, coordinates) = restore.states(&original, &intent.baseline, &after).unwrap();
    for (index, body) in original.compounds.iter().enumerate() {
        if body.source_parts.contains(&unrelated) {
            assert_eq!(poses[index], expected_unrelated);
        } else {
            assert!(Vec3::from_slice(&poses[index].position[..3]).abs_diff_eq(
                Vec3::from_slice(
                    &simulation.live_state.as_ref().unwrap().transforms[index].position[..3]
                ),
                1.0e-5
            ));
        }
    }
    assert_eq!(
        coordinates,
        simulation.live_state.as_ref().unwrap().coordinates
    );
}

fn render_handles(device: wgpu::Device, queue: wgpu::Queue) -> (RenderDevice, RenderQueue) {
    (
        RenderDevice::from(device),
        RenderQueue(Arc::new(bevy::render::renderer::WgpuWrapper::new(queue))),
    )
}

#[test]
fn prepared_placement_publishes_once_and_rejections_leave_scene_intact() {
    for failure in ["", "snapshot", "revision", "cancel"] {
        let (intent, mut simulation, _) = fixture();
        let staged = intent.stage(&intent.baseline, []).unwrap();
        let mut graph = EditorGraph(intent.baseline.clone());
        let mut world = WorldRuntime::from_world(&mut World::new());
        let foundation = world.foundation_revision();
        simulation.world_revision = Some((0, foundation));
        let mut state = EditorState {
            placement_bounds: PlacementBounds::GarageBuild,
            ..default()
        };
        state.weld.request = Some(intent.clone());
        let mut publication = WorldPhysicsPublication {
            placement: Some(Publication {
                intent,
                task: None,
                ready: Some(PreparedWorldPhysics {
                    creation: staged.compile().unwrap(),
                    graph: staged,
                    gpu: None,
                }),
                foundation,
            }),
            ..default()
        };
        let mut history = EditorHistory::default();
        let mut frozen = DimensionFreeze::default();
        match failure {
            "snapshot" => simulation.live_state = None,
            "revision" => {
                spawn(&mut graph.0, IVec3::new(80, 0, 0));
            }
            "cancel" => state.weld.cancel(),
            _ => {}
        }
        let before = graph.0.clone();
        let before_poses = simulation.transforms.clone();
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let (device, queue) = render_handles(device, queue);
        assert!(maintain(
            &mut graph,
            &mut state,
            &mut history,
            &mut simulation,
            &mut frozen,
            &mut world,
            &mut publication,
            &device,
            &queue
        ));
        if failure.is_empty() {
            assert_eq!(history.undo.len(), 1);
            assert_eq!(graph.0.weld_count(), 1);
            assert!(!state.weld.busy());
            assert_eq!(simulation.world_revision, Some((1, foundation)));
        } else {
            assert!(graph.0.shares_revision(&before));
            assert!(history.undo.is_empty());
            assert_eq!(simulation.transforms, before_poses);
        }
    }
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn real_gpu_weld_publication_starts_from_default_source_coordinates() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("real GPU adapter required");
    eprintln!("Weld publication adapter: {:?}", adapter.get_info());
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let (intent, simulation, _) = fixture();
    let graph = intent.stage(&intent.baseline, []).unwrap();
    let creation = graph.compile().unwrap();
    let gpu = mechanic_gpu::GpuPhysics::new_with_config(
        &device,
        &queue,
        &creation,
        crate::GpuPhysicsConfig {
            collisions_enabled: false,
            ground_plane_enabled: false,
            ..default()
        },
    )
    .unwrap();
    let (_, render_queue) = render_handles(device.clone(), queue.clone());
    let replacement = crate::replacement_simulation_for_weld(
        PreparedWorldPhysics {
            graph,
            creation,
            gpu: Some(gpu),
        },
        &simulation,
        (1, 0),
        &render_queue,
        Some(&intent),
    )
    .unwrap();
    let gpu = replacement.gpu.as_ref().unwrap();
    gpu.dispatch_tick(&device, &queue, 13);
    let diagnostics = gpu.read_last_tick(&device).unwrap();
    assert_eq!(diagnostics.error_flags, 0);
    let poses = gpu.read_snapshot_transforms(&device, &queue, 1).unwrap();
    for (actual, expected) in poses.iter().zip(&replacement.transforms) {
        assert!(
            Vec3::from_slice(&actual.position[..3])
                .distance(Vec3::from_slice(&expected.position[..3]))
                < 0.05
        );
    }
}

#[test]
fn default_ghost_includes_joints_and_relocation_updates_saved_sockets() {
    let (intent, _, _) = fixture();
    let graph = &intent.baseline;
    let joint = graph.bearings().find(|(_, joint)| matches!(joint.source.owner, FaceOwner::Part(part) if intent.parts.contains(&part))).unwrap().1;
    let socket = crate::PlacedBearing {
        kind: joint.kind,
        axis: joint.axis,
        source: joint.source,
        anchor: joint.shared_anchor,
        dimensions: joint.dimensions,
    };
    let parts = crate::frame_visuals::parts_preview_mesh(
        graph,
        &AppSimulation::default(),
        &intent.parts,
        None,
        1.0,
    );
    let ghost = crate::weld_tool::preview_mesh(graph, &intent.parts, &[socket]);
    assert!(ghost.count_vertices() > parts.count_vertices());
    let mut state = EditorState {
        placed_bearings: vec![socket],
        ..default()
    };
    intent.place_sockets(&mut state);
    assert!(
        state.placed_bearings[0]
            .anchor
            .abs_diff_eq(intent.transform.point(socket.anchor), 1.0e-6)
    );
    assert!(
        state.placed_bearings[0]
            .axis
            .abs_diff_eq(intent.transform.vector(socket.axis), 1.0e-6)
    );
}

#[test]
fn unrelated_authored_obstruction_does_not_reject_a_clear_live_endpoint() {
    let (mut intent, mut simulation, unrelated) = fixture();
    let source_position = intent
        .transform
        .point(intent.baseline.part_position(intent.source.part).unwrap());
    let old_position = intent.baseline.part_position(unrelated).unwrap();
    intent
        .baseline
        .reframe_parts(
            [unrelated],
            ConstructionFrame::new(source_position - old_position, Quat::IDENTITY).unwrap(),
        )
        .unwrap();
    simulation.published_graph = intent.baseline.clone();
    simulation.creation = Some(intent.baseline.compile().unwrap());
    let world = WorldRuntime::from_world(&mut World::new());
    intent
        .validate(
            &intent.baseline,
            &simulation,
            &world,
            PlacementBounds::GarageBuild,
        )
        .unwrap();
}

#[test]
fn bearing_connected_bodies_weld_in_place_and_same_body_is_rejected() {
    let mut graph = ConstructionGraph::new();
    let base = spawn(&mut graph, IVec3::new(0, 28, 0));
    let arm = spawn(&mut graph, IVec3::new(0, 30, 0));
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(base, FaceKind::PositiveY),
            FaceRef::part(arm, FaceKind::NegativeY),
            Vec3::Y * 7.25,
            Vec3::Y,
        )))
        .unwrap();
    let simulation = AppSimulation::default();
    let source = face(&graph, base, FaceKind::PositiveY);
    let destination = face(&graph, arm, FaceKind::NegativeY);
    assert!(Intent::in_place(&graph, &simulation, &source, &source).is_err());
    let intent = Intent::in_place(&graph, &simulation, &source, &destination).unwrap();
    let world = WorldRuntime::from_world(&mut World::new());
    intent
        .validate(&graph, &simulation, &world, PlacementBounds::GarageBuild)
        .unwrap();
    let staged = intent.stage(&graph, []).unwrap();
    assert_eq!(staged.part_frame(base), graph.part_frame(base));
    assert_eq!(staged.part_frame(arm), graph.part_frame(arm));
    assert_eq!(staged.weld_count(), 1);
    assert!(crate::weld_lockup_warning(&graph, &staged).is_some());
}
