use bevy_math::{IVec3, Quat, Vec3};

use super::{
    BearingSocket, CREATION_FORMAT_VERSION, CreationDocument, CreationError, FaceOwnerDoc, PartDoc,
    TopologySourceDoc,
};
use crate::{
    ActuatorAssignment, BearingDimensions, BearingSpec, BuildCommand, BuildOutcome, BuildPose,
    ConstructionGraph, ConstructionMaterial, ControllerSpec, CuboidSpec, CylinderDimensions,
    CylinderSpec, DimensionLinkId, DimensionLinkSpec, DriveDwell, DriveKey, DriveLimits,
    DriveLinkSpec, DriveName, DriveProgram, DriveRelease, DriveState, DriveTarget, DriveTrigger,
    EdgeChainRef, EdgeTreatment, EngineKind, EngineSpec, FaceKind, FaceRef, GearKey, GearKeyChord,
    GridRotation, InputSeatLinkSpec, InputSpec, MaterialAppearance, MaterialColor, MaterialDye,
    MaterialFinish, PartSpec, PipeBendDimensions, PipeBendSpec, RigidLinkSpec,
    SeatControllerLinkSpec, SeatSpec, ServoSpec, ShapeFeature, ShapeRegion, ShiftMode, SolidOwner,
    TopologySource, WeldSpec,
};

fn suspension_document() -> CreationDocument {
    let mut graph = ConstructionGraph::new();
    let source = spawned(
        graph
            .apply(BuildCommand::Spawn(cuboid([4; 3], IVec3::new(0, 2, 0))))
            .unwrap(),
    );
    let target = spawned(
        graph
            .apply(BuildCommand::Spawn(cuboid([4; 3], IVec3::new(6, 2, 0))))
            .unwrap(),
    );
    let spec = crate::SuspensionSpec::new(
        Some(crate::SpringSpec::default()),
        Some(crate::ShockSpec::default()),
        Some(crate::BumpStopSpec::new(0.05, 0.06).unwrap()),
    )
    .unwrap();
    graph
        .apply(BuildCommand::AddBearing(
            BearingSpec::new(
                FaceRef::part(source, FaceKind::PositiveX),
                FaceRef::part(target, FaceKind::NegativeX),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::X,
            )
            .with_kind(crate::JointKind::Suspension(spec)),
        ))
        .unwrap();
    let socket = BearingSocket {
        kind: crate::JointKind::Suspension(spec),
        axis: Vec3::Y,
        source: FaceRef::part(source, FaceKind::PositiveY),
        anchor: Vec3::Y,
        dimensions: BearingDimensions::default(),
    };
    CreationDocument::from_graph(&graph, "Suspension", &[socket])
}

#[test]
fn suspension_round_trip_and_cardinal_placement_preserve_attached_and_socket_mass() {
    let document = suspension_document();
    let original = round_trip(&document).into_graph().unwrap();
    let original_compiled = original
        .graph
        .compile_with_sockets([], &original.sockets)
        .unwrap();
    let mut transformed = round_trip(&document);
    transformed.transform_cardinal(1, IVec3::new(8, 0, 4));
    let loaded = round_trip(&transformed).into_graph().unwrap();
    assert_eq!(loaded.graph.bearing_count(), 1);
    assert_eq!(loaded.sockets.len(), 1);
    assert_eq!(loaded.sockets[0].kind, original.sockets[0].kind);
    assert!(
        loaded.sockets[0]
            .anchor
            .abs_diff_eq(Vec3::new(1.0, 1.0, 0.5), 1.0e-6)
    );
    assert!(loaded.sockets[0].axis.abs_diff_eq(Vec3::Y, 1.0e-6));
    let bearing = loaded.graph.bearings().next().unwrap().1;
    assert!(bearing.axis.abs_diff_eq(Vec3::NEG_Z, 1.0e-6));
    let compiled = loaded
        .graph
        .compile_with_sockets([], &loaded.sockets)
        .unwrap();
    for (before, after) in original_compiled.compounds.iter().zip(&compiled.compounds) {
        assert!((before.mass_properties.mass - after.mass_properties.mass).abs() < 0.001);
        let center = before.mass_properties.center_of_mass;
        let expected = Vec3::new(center.z + 1.0, center.y, -center.x + 0.5);
        assert!(
            after
                .mass_properties
                .center_of_mass
                .abs_diff_eq(expected, 1.0e-5)
        );
    }
}

#[test]
fn suspension_socket_loading_rejects_invalid_axis_anchor_and_support() {
    let document = suspension_document();
    for axis in [
        [0.0; 3],
        [0.0, 2.0, 0.0],
        [1.0, 0.0, 0.0],
        [f32::NAN; 3],
        [f32::INFINITY; 3],
    ] {
        let mut invalid = document.clone();
        invalid.sockets[0].axis = axis;
        assert!(matches!(
            invalid.into_graph(),
            Err(CreationError::Graph(crate::GraphError::InvalidBearingAxis))
        ));
    }
    for anchor in [[0.0, 1.1, 0.0], [9.0, 1.0, 0.0], [f32::NAN; 3]] {
        let mut invalid = document.clone();
        invalid.sockets[0].anchor = anchor;
        assert!(matches!(
            invalid.into_graph(),
            Err(CreationError::Graph(
                crate::GraphError::BearingAnchorOutsideFaces
            ))
        ));
    }
    let mut ground = document.clone();
    ground.sockets[0].source.owner = FaceOwnerDoc::Ground;
    assert!(matches!(
        ground.into_graph(),
        Err(CreationError::Graph(crate::GraphError::BearingOnGround))
    ));
}

fn cuboid(dimensions: [u8; 3], units: IVec3) -> CuboidSpec {
    CuboidSpec::new(dimensions, BuildPose::new(units, GridRotation::default()))
        .expect("test dimensions are in range")
}

#[test]
fn tool_view_snapshot_restores_canonical_parts_and_socket_frames() {
    let mut graph = ConstructionGraph::new();
    let part = spawned(
        graph
            .apply(BuildCommand::Spawn(cuboid([4; 3], IVec3::ZERO)))
            .unwrap(),
    );
    graph
        .apply(BuildCommand::Spawn(cuboid([4; 3], IVec3::new(16, 0, 0))))
        .unwrap();
    let frame = crate::ConstructionFrame::new(
        Vec3::new(2.0, 3.0, -1.0),
        Quat::from_rotation_z(0.4) * Quat::from_rotation_y(-0.7),
    )
    .unwrap();
    graph.reframe_parts([part], frame).unwrap();
    let view = graph
        .in_edit_frame(graph.part_frame_id(part).unwrap())
        .unwrap();
    let rail = crate::LinearBearing {
        dimensions: crate::LinearBearingDimensions::default(),
        mount_normal: Vec3::Y,
        face: crate::CarriageFace::Top,
    };
    let sockets =
        [crate::JointKind::Rotational, crate::JointKind::Linear(rail)].map(|kind| BearingSocket {
            kind,
            axis: Vec3::X,
            source: FaceRef::part(part, FaceKind::PositiveY),
            anchor: Vec3::Y * 0.5,
            dimensions: BearingDimensions::default(),
        });
    let document = CreationDocument::from_graph(&view, "View snapshot", &sockets);
    let restored = round_trip(&document).into_graph().unwrap();
    for ((original, _), (loaded, _)) in graph.parts().zip(restored.graph.parts()) {
        assert!(
            graph
                .part_position(original)
                .unwrap()
                .abs_diff_eq(restored.graph.part_position(loaded).unwrap(), 1.0e-5)
        );
        assert!(
            graph
                .part_rotation(original)
                .unwrap()
                .abs_diff_eq(restored.graph.part_rotation(loaded).unwrap(), 1.0e-5)
        );
    }
    assert_eq!(
        restored.graph.view_to_build(),
        crate::ConstructionFrame::IDENTITY
    );
    for socket in restored.sockets {
        assert!(
            socket
                .anchor
                .abs_diff_eq(frame.point(Vec3::Y * 0.5), 1.0e-5)
        );
        assert!(socket.axis.abs_diff_eq(frame.vector(Vec3::X), 1.0e-5));
        if let crate::JointKind::Linear(rail) = socket.kind {
            assert!(rail.mount_normal.abs_diff_eq(frame.vector(Vec3::Y), 1.0e-5));
        }
    }
}

#[test]
fn canonical_socket_snapshot_preserves_exact_coordinate_bits() {
    let mut graph = ConstructionGraph::new();
    let part = spawned(
        graph
            .apply(BuildCommand::Spawn(cuboid([4; 3], IVec3::ZERO)))
            .unwrap(),
    );
    let normal = Vec3::new(-0.0, 1.0, -0.0);
    let socket = BearingSocket {
        kind: crate::JointKind::Linear(crate::LinearBearing {
            dimensions: crate::LinearBearingDimensions::default(),
            mount_normal: normal,
            face: crate::CarriageFace::Top,
        }),
        axis: Vec3::new(1.0, -0.0, -0.0),
        source: FaceRef::part(part, FaceKind::PositiveY),
        anchor: Vec3::new(-0.0, 0.1, 0.125),
        dimensions: BearingDimensions::default(),
    };
    let document = CreationDocument::from_graph(&graph, "Canonical", &[socket]);
    let saved = document.sockets[0];
    assert_eq!(
        saved.anchor.map(f32::to_bits),
        socket.anchor.to_array().map(f32::to_bits)
    );
    assert_eq!(
        saved.axis.map(f32::to_bits),
        socket.axis.to_array().map(f32::to_bits)
    );
    let crate::JointKind::Linear(rail) = saved.kind else {
        unreachable!()
    };
    assert_eq!(
        rail.mount_normal.to_array().map(f32::to_bits),
        normal.to_array().map(f32::to_bits)
    );
}

fn spawned(outcome: BuildOutcome) -> crate::PartId {
    match outcome {
        BuildOutcome::Spawned(part) => part,
        other => panic!("expected a spawn outcome, got {other:?}"),
    }
}

fn round_trip(document: &CreationDocument) -> CreationDocument {
    let text = ron::ser::to_string_pretty(document, ron::ser::PrettyConfig::default())
        .expect("a creation document serializes");
    ron::from_str(&text).expect("a serialized creation document parses")
}

#[test]
fn arbitrary_construction_frame_round_trip_preserves_local_grid_and_world_pose() {
    let mut graph = ConstructionGraph::new();
    let part = spawned(
        graph
            .apply(BuildCommand::Spawn(cuboid([2, 2, 2], IVec3::new(4, 8, -2))))
            .unwrap(),
    );
    let local = graph.part(part).unwrap().pose();
    let frame =
        crate::ConstructionFrame::new(Vec3::new(3.0, 5.0, -1.0), Quat::from_rotation_y(0.73))
            .unwrap();
    graph.reframe_parts([part], frame).unwrap();
    let original = CreationDocument::from_graph(&graph, "Reframed", &[]);
    let restored = round_trip(&original).into_graph().unwrap().graph;
    let part = restored.parts().next().unwrap().0;
    assert_eq!(restored.part(part).unwrap().pose(), local);
    assert!(
        restored
            .part_position(part)
            .unwrap()
            .abs_diff_eq(frame.point(local.translation()), 1.0e-5)
    );
    assert_eq!(
        CreationDocument::from_graph(&restored, "Reframed", &[]),
        original
    );
}

#[test]
fn cardinal_transfer_applies_once_to_framed_world_geometry() {
    let mut graph = ConstructionGraph::new();
    let part = spawned(
        graph
            .apply(BuildCommand::Spawn(cuboid([2, 2, 2], IVec3::new(4, 8, -2))))
            .unwrap(),
    );
    graph
        .reframe_parts(
            [part],
            crate::ConstructionFrame::new(Vec3::new(3.0, 5.0, -1.0), Quat::from_rotation_x(0.43))
                .unwrap(),
        )
        .unwrap();
    let position = graph.part_position(part).unwrap();
    let rotation = graph.part_rotation(part).unwrap();
    let cardinal = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    let mut document = CreationDocument::from_graph(&graph, "Transfer", &[]);
    document.transform_cardinal(1, IVec3::new(8, 16, -24));
    let restored = document.into_graph().unwrap().graph;
    let part = restored.parts().next().unwrap().0;
    assert!(
        restored
            .part_position(part)
            .unwrap()
            .abs_diff_eq(cardinal * position + Vec3::new(1.0, 2.0, -3.0), 1.0e-5)
    );
    assert!(
        restored
            .part_rotation(part)
            .unwrap()
            .abs_diff_eq(cardinal * rotation, 1.0e-5)
    );
}

#[test]
fn append_remaps_frame_membership_without_moving_either_creation() {
    let mut graph = ConstructionGraph::new();
    let part = spawned(
        graph
            .apply(BuildCommand::Spawn(cuboid([1, 1, 1], IVec3::ZERO)))
            .unwrap(),
    );
    let mut first = CreationDocument::from_graph(&graph, "Combined", &[]);
    graph
        .reframe_parts(
            [part],
            crate::ConstructionFrame::new(Vec3::X * 5.0, Quat::from_rotation_y(0.61)).unwrap(),
        )
        .unwrap();
    first
        .append(CreationDocument::from_graph(&graph, "Second", &[]))
        .unwrap();
    let restored = first.into_graph().unwrap().graph;
    let positions = restored
        .parts()
        .map(|(id, _)| restored.part_position(id).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(positions, vec![Vec3::ZERO, Vec3::X * 5.0]);
}

#[test]
fn appended_overlapping_local_regions_round_trip_in_separate_frames() {
    let mut graph = ConstructionGraph::new();
    let spec = CuboidSpec::new(
        [1, 1, 1],
        BuildPose::from_half_grid(IVec3::ONE, GridRotation::default()),
    )
    .unwrap();
    let part = spawned(graph.apply(BuildCommand::Spawn(spec)).unwrap());
    let region = ShapeRegion::from_origin_steps(IVec3::ZERO, IVec3::ONE, spec.material).unwrap();
    graph.apply(BuildCommand::AddRegion(region)).unwrap();
    let mut document = CreationDocument::from_graph(&graph, "Two grids", &[]);
    graph
        .reframe_parts(
            [part],
            crate::ConstructionFrame::new(Vec3::X * 5.0, Quat::IDENTITY).unwrap(),
        )
        .unwrap();
    document
        .append(CreationDocument::from_graph(&graph, "Translated grid", &[]))
        .unwrap();
    let restored = round_trip(&document).into_graph().unwrap().graph;
    let parts = restored.parts().map(|(id, _)| id).collect::<Vec<_>>();
    assert_eq!(restored.regions().count(), 2);
    assert_ne!(restored.region_of(parts[0]), restored.region_of(parts[1]));
    for &part in &parts {
        assert_eq!(
            restored.region_frame_id(restored.region_of(part).unwrap()),
            restored.part_frame_id(part)
        );
    }
    assert!(
        restored
            .part_position(parts[0])
            .unwrap()
            .abs_diff_eq(Vec3::splat(0.125), 1.0e-5)
    );
    assert!(
        restored
            .part_position(parts[1])
            .unwrap()
            .abs_diff_eq(Vec3::new(5.125, 0.125, 0.125), 1.0e-5)
    );
    assert_eq!(
        restored.edit_frame_id(),
        crate::ConstructionFrameId::default()
    );
    assert_eq!(restored.compile().unwrap().compounds.len(), 2);

    let mut missing = document.clone();
    missing.region_frames.pop();
    assert!(matches!(
        missing.into_graph(),
        Err(CreationError::RegionFrameMembershipCount)
    ));
    document.region_frames[1] = 99;
    assert!(matches!(
        document.into_graph(),
        Err(CreationError::MissingFrame(99))
    ));
}

#[test]
fn framed_transmissions_replay_locally_and_reject_split_membership() {
    let mut graph = ConstructionGraph::new();
    let engine = spawned(
        graph
            .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                EngineKind::Gas,
                BuildPose::default(),
            )))
            .unwrap(),
    );
    let spec = graph.next_transmission_spec(engine).unwrap();
    let child = spawned(
        graph
            .apply(BuildCommand::AttachTransmission {
                parent: engine,
                spec,
            })
            .unwrap(),
    );
    graph
        .reframe_parts(
            [engine, child],
            crate::ConstructionFrame::new(Vec3::new(3.0, 4.0, 5.0), Quat::from_rotation_z(0.37))
                .unwrap(),
        )
        .unwrap();
    let document = CreationDocument::from_graph(&graph, "Transmission", &[]);
    let restored = round_trip(&document).into_graph().unwrap().graph;
    let restored_child = restored
        .parts()
        .find_map(|(id, _)| restored.transmission_parent(id).map(|_| id))
        .unwrap();
    assert!(
        restored
            .part_position(restored_child)
            .unwrap()
            .abs_diff_eq(graph.part_position(child).unwrap(), 1.0e-5)
    );
    let mut invalid = document;
    invalid.part_frames[1] = 0;
    assert!(matches!(
        invalid.into_graph(),
        Err(CreationError::TransmissionFrame(1))
    ));
}

#[test]
fn invalid_frames_and_dense_membership_are_rejected() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(cuboid([1, 1, 1], IVec3::ZERO)))
        .unwrap();
    let original = CreationDocument::from_graph(&graph, "Invalid", &[]);
    let mut invalid = original.clone();
    invalid.part_frames.clear();
    assert!(matches!(
        invalid.into_graph(),
        Err(CreationError::FrameMembershipCount)
    ));
    let mut invalid = original.clone();
    invalid.part_frames[0] = 9;
    assert!(matches!(
        invalid.into_graph(),
        Err(CreationError::MissingFrame(9))
    ));
    for frame in [
        super::ConstructionFrameDoc {
            translation: [f32::NAN, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        },
        super::ConstructionFrameDoc {
            translation: [0.0; 3],
            rotation: [0.0; 4],
        },
    ] {
        let mut invalid = original.clone();
        invalid.frames[0] = frame;
        assert!(matches!(invalid.into_graph(), Err(CreationError::Frame(_))));
    }
    assert!(
        ron::from_str::<CreationDocument>("(version:16,name:\"Missing frames\",parts:[])").is_err()
    );
}

/// A short tower welded to the ground, carrying a driven bearing, a
/// control block, a hollow sliced cylinder, and one loose ring.
#[expect(
    clippy::too_many_lines,
    reason = "shared persistence fixture includes all connection records"
)]
fn sample() -> (ConstructionGraph, Vec<BearingSocket>) {
    let mut graph = ConstructionGraph::new();
    // A 1.0 x 0.5 x 1.0 m slab resting on the ground plane.
    let base = spawned(
        graph
            .apply(BuildCommand::Spawn(cuboid([4, 2, 4], IVec3::new(0, 1, 0))))
            .expect("the base spawns"),
    );
    // A smaller block sitting on the slab, held only by the bearing.
    let rotor = spawned(
        graph
            .apply(BuildCommand::Spawn(cuboid([2, 2, 2], IVec3::new(0, 3, 0))))
            .expect("the rotor spawns"),
    );
    // A quarter-turned control block flush on the slab's top face.
    let controller = spawned(
        graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::from_half_grid(IVec3::new(2, 6, 0), GridRotation::new(0, 1, 0)),
            )))
            .expect("the control block spawns"),
    );
    // A detached hollow, sliced cylinder, joined to the base without contact.
    let column = spawned(
        graph
            .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
                CylinderDimensions::new(1.0, 0.5, 0.75)
                    .expect("the cylinder dimensions are in range")
                    .with_sweep_angle_degrees(255)
                    .expect("255 degrees is a supported sweep"),
                BuildPose::new(IVec3::new(-8, 4, 0), GridRotation::default()),
            )))
            .expect("the cylinder spawns"),
    );
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(base, FaceKind::NegativeY),
            second: FaceRef::ground(),
        }))
        .expect("the base welds to the ground");
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec {
            first: base,
            second: column,
        }))
        .expect("the column joins the base rigidly");
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(base, FaceKind::PositiveY),
            second: FaceRef::part(controller, FaceKind::NegativeY),
        }))
        .expect("the control block welds to the base");
    let bearing = match graph
        .apply(BuildCommand::AddBearing(
            BearingSpec::new(
                FaceRef::part(base, FaceKind::PositiveY),
                FaceRef::part(rotor, FaceKind::NegativeY),
                Vec3::new(0.0, 0.5, 0.0),
                Vec3::Y,
            )
            .with_dimensions(
                BearingDimensions::new(0.5, 0.2).expect("the ring dimensions are in range"),
            ),
        ))
        .expect("the bearing is added")
    {
        BuildOutcome::BearingAdded(bearing) => bearing,
        other => panic!("expected a bearing outcome, got {other:?}"),
    };

    let program = DriveProgram::new(
        &[
            DriveState::new(DriveTarget::Angle(0.0)).expect("zero degrees is in range"),
            DriveState::new(DriveTarget::Speed(2.5))
                .expect("2.5 rad/s is in range")
                .with_dwell(Some(
                    DriveDwell::new(1.5, Some(0)).expect("1.5 s is in range"),
                ))
                .with_trigger(Some(DriveTrigger::new(
                    DriveKey::new('w').expect("W is bindable"),
                    DriveRelease::RevertTo(0),
                ))),
        ],
        true,
    )
    .expect("the program is valid");
    graph
        .apply(BuildCommand::AddDriveLink(DriveLinkSpec {
            linear_limits: None,
            controller,
            bearing,
            reversed: true,
            actuator: ActuatorAssignment::Unpowered,
            limits: DriveLimits::new(4.0, f32::INFINITY, Some((-1.0, 1.0)))
                .expect("the limits are in range"),
            program,
            name: DriveName::new("Tipper arm"),
        }))
        .expect("the wire is added");

    // A ring placed on the rotor's top face with nothing attached through it.
    let sockets = vec![BearingSocket {
        kind: crate::JointKind::Rotational,
        axis: Vec3::ZERO,
        source: FaceRef::part(rotor, FaceKind::PositiveY),
        anchor: Vec3::new(0.0, 1.0, 0.0),
        dimensions: BearingDimensions::new(0.3, 0.05).expect("the ring dimensions are in range"),
    }];
    (graph, sockets)
}

#[test]
fn sample_creation_survives_a_serialized_round_trip() {
    let (graph, sockets) = sample();
    let document = CreationDocument::from_graph(&graph, "Test Rig", &sockets);
    let restored = round_trip(&document)
        .into_graph()
        .expect("the document rebuilds");

    assert_eq!(restored.name, "Test Rig");
    assert_eq!(restored.sockets.len(), 1);
    assert_eq!(restored.graph.part_count(), graph.part_count());
    assert_eq!(restored.graph.weld_count(), graph.weld_count());
    assert_eq!(restored.graph.rigid_link_count(), graph.rigid_link_count());
    assert_eq!(restored.graph.bearing_count(), graph.bearing_count());
    assert_eq!(restored.graph.drive_link_count(), graph.drive_link_count());
    assert_eq!(
        CreationDocument::from_graph(&restored.graph, "Test Rig", &restored.sockets),
        document,
        "a second capture of the rebuilt graph must be byte-identical"
    );
}

#[test]
fn reusable_creation_remaps_every_dimension_link_id() {
    let mut graph = ConstructionGraph::new();
    for id in [2, 9] {
        graph
            .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
                DimensionLinkId(id),
                BuildPose::default(),
            )))
            .unwrap();
    }
    let mut document = CreationDocument::from_graph(&graph, "Links", &[]);
    let mut next_id = 40;
    document.remap_dimension_links(&mut next_id);
    let restored = document.into_graph().unwrap().graph;
    assert!(restored.dimension_link(DimensionLinkId(40)).is_some());
    assert!(restored.dimension_link(DimensionLinkId(41)).is_some());
    assert_eq!(next_id, 42);
}

#[test]
fn engine_kinds_survive_a_serialized_round_trip() {
    let mut graph = ConstructionGraph::new();
    for (kind, x) in [(EngineKind::Gas, -2), (EngineKind::Electric, 2)] {
        graph
            .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                kind,
                BuildPose::new(IVec3::new(x, 2, 0), GridRotation::default()),
            )))
            .expect("the engine spawns");
    }

    let document = CreationDocument::from_graph(&graph, "Engines", &[]);
    assert_eq!(document.version, CREATION_FORMAT_VERSION);
    let restored = round_trip(&document)
        .into_graph()
        .expect("the engine document rebuilds");
    let kinds = restored
        .graph
        .parts()
        .map(|(_, spec)| match spec {
            PartSpec::Engine(engine) => engine.kind,
            _ => panic!("engine document rebuilt a different part kind"),
        })
        .collect::<Vec<_>>();
    assert_eq!(kinds, [EngineKind::Gas, EngineKind::Electric]);
}

#[test]
fn servo_seat_input_and_their_routes_survive_a_round_trip() {
    let mut graph = ConstructionGraph::new();
    let servo = spawned(
        graph
            .apply(BuildCommand::SpawnServo(ServoSpec::new(
                BuildPose::default(),
            )))
            .unwrap(),
    );
    let input = spawned(
        graph
            .apply(BuildCommand::SpawnInput(InputSpec::new(
                BuildPose::default(),
            )))
            .unwrap(),
    );
    let seat = spawned(
        graph
            .apply(BuildCommand::SpawnSeat(SeatSpec::new(BuildPose::default())))
            .unwrap(),
    );
    let controller = spawned(
        graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::default(),
            )))
            .unwrap(),
    );
    graph
        .apply(BuildCommand::AddInputSeatLink(InputSeatLinkSpec {
            input,
            seat,
        }))
        .unwrap();
    graph
        .apply(BuildCommand::AddSeatControllerLink(
            SeatControllerLinkSpec { seat, controller },
        ))
        .unwrap();

    let restored = round_trip(&CreationDocument::from_graph(&graph, "Controls", &[]))
        .into_graph()
        .unwrap()
        .graph;
    assert_eq!(
        restored
            .parts()
            .filter(|(_, part)| matches!(part, PartSpec::Servo(_)))
            .count(),
        1
    );
    assert_eq!(restored.input_seat_links().count(), 1);
    assert_eq!(restored.seat_controller_links().count(), 1);
    assert!(
        restored.part(servo).is_some(),
        "canonical replay keeps part ids"
    );
}

#[test]
fn obsolete_creation_versions_are_rejected() {
    for version in 1..CREATION_FORMAT_VERSION {
        let document = CreationDocument {
            version,
            name: format!("Obsolete {version}"),
            ..CreationDocument::from_graph(&ConstructionGraph::new(), "obsolete", &[])
        };
        assert!(matches!(
            document.into_graph(),
            Err(CreationError::UnsupportedVersion(found)) if found == version
        ));
    }
}

#[test]
fn current_version_round_trips_all_construction_materials() {
    let mut graph = ConstructionGraph::new();
    for (index, material) in ConstructionMaterial::ALL.into_iter().enumerate() {
        let position = IVec3::new(i32::try_from(index).unwrap() * 8, 0, 0);
        graph
            .apply(BuildCommand::Spawn(
                cuboid([1, 1, 1], position).with_material(material),
            ))
            .unwrap();
    }
    let document = CreationDocument::from_graph(&graph, "Materials", &[]);
    assert_eq!(document.version, CREATION_FORMAT_VERSION);
    let restored = round_trip(&document).into_graph().unwrap();
    let materials = restored
        .graph
        .parts()
        .filter_map(|(_, spec)| spec.as_cuboid().map(|cuboid| cuboid.material))
        .collect::<Vec<_>>();
    assert_eq!(materials, ConstructionMaterial::ALL);
}

#[test]
fn current_version_round_trips_construction_appearances() {
    let appearance = MaterialAppearance::new(
        MaterialColor::Dye(MaterialDye::new([224, 86, 31], 2.0).unwrap()),
        MaterialFinish::Anodised,
    );
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            cuboid([1, 1, 1], IVec3::ZERO).with_appearance(appearance),
        ))
        .unwrap();
    let document = CreationDocument::from_graph(&graph, "Chroma", &[]);
    let encoded = ron::ser::to_string(&document).unwrap();
    assert!(encoded.contains("appearance"));
    let restored = ron::from_str::<CreationDocument>(&encoded)
        .unwrap()
        .into_graph()
        .unwrap();
    assert_eq!(
        restored.graph.parts().next().unwrap().1.appearance(),
        Some(appearance)
    );
}

#[test]
fn construction_documents_without_appearance_are_rejected() {
    let missing = r"Cuboid(
            dimensions: [1, 1, 1],
            pose: (translation_ticks: [0, 0, 0], rotation: [0, 0, 0]),
            material: Steel,
        )";
    assert!(ron::from_str::<PartDoc>(missing).is_err());
}

#[test]
fn rebuilt_creation_compiles_to_the_same_bodies() {
    let (graph, sockets) = sample();
    let expected = graph.compile().expect("the sample compiles");
    let restored = CreationDocument::from_graph(&graph, "Test Rig", &sockets)
        .into_graph()
        .expect("the document rebuilds");
    let actual = restored.graph.compile().expect("the rebuild compiles");

    assert_eq!(actual.compounds.len(), expected.compounds.len());
    assert_eq!(actual.bearings.len(), expected.bearings.len());
    assert_eq!(
        actual.coordinate_drives.len(),
        expected.coordinate_drives.len()
    );
}

#[test]
fn unlimited_torque_round_trips_without_encoding_an_infinity() {
    let (graph, _) = sample();
    let document = CreationDocument::from_graph(&graph, "Test Rig", &[]);
    let wire = &document.drive_links[0];
    assert_eq!(wire.limits.max_torque_newton_meters, None);

    let text = ron::ser::to_string(&document).expect("the document serializes");
    assert!(
        !text.contains("inf"),
        "an unlimited torque must not encode as a float infinity: {text}"
    );

    let restored = round_trip(&document)
        .into_graph()
        .expect("the document rebuilds");
    let (_, link) = restored
        .graph
        .drive_links()
        .next()
        .expect("the rebuilt graph keeps its wire");
    assert!(link.limits.max_torque_newton_meters().is_infinite());
    assert_eq!(link.limits.angle_limits(), Some((-1.0, 1.0)));
    assert_eq!(link.name.as_str(), "Tipper arm");
    assert!(link.reversed);
    assert!(link.program.loops());
    assert_eq!(link.program.len(), 2);
    let triggered = link.program.state(1).expect("the second state exists");
    assert_eq!(
        triggered.trigger().map(|trigger| trigger.key().symbol()),
        Some('W')
    );
    assert_eq!(
        triggered.trigger().map(DriveTrigger::release),
        Some(DriveRelease::RevertTo(0))
    );
    assert_eq!(
        triggered
            .dwell()
            .map(|dwell| (dwell.seconds(), dwell.next())),
        Some((1.5, Some(0)))
    );
}

#[test]
fn hollow_sliced_cylinder_keeps_its_bore_and_sweep() {
    let (graph, _) = sample();
    let restored = CreationDocument::from_graph(&graph, "Test Rig", &[])
        .into_graph()
        .expect("the document rebuilds");
    let dimensions = restored
        .graph
        .parts()
        .find_map(|(_, spec)| spec.as_cylinder())
        .expect("the rebuilt graph keeps its cylinder")
        .dimensions;

    assert!((dimensions.outer_diameter() - 1.0).abs() < 1.0e-6);
    assert!((dimensions.inner_diameter() - 0.5).abs() < 1.0e-6);
    assert_eq!(dimensions.axial_length_units(), 3);
    assert_eq!(dimensions.sweep_angle_degrees(), 255);
}

#[test]
fn current_version_round_trips_pipe_bends() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::SpawnPipeBend(
            PipeBendSpec::new(
                PipeBendDimensions::new(0.75, 0.25, 4).unwrap(),
                BuildPose::new(IVec3::new(4, 8, 12), GridRotation::new(1, 2, 3)),
            )
            .with_material(ConstructionMaterial::Aluminium),
        ))
        .unwrap();
    let document = CreationDocument::from_graph(&graph, "Bent Pipe", &[]);
    assert_eq!(document.version, CREATION_FORMAT_VERSION);
    let loaded = round_trip(&document).into_graph().unwrap();
    let bend = loaded
        .graph
        .parts()
        .find_map(|(_, part)| part.as_pipe_bend())
        .unwrap();
    assert_eq!(bend.material, ConstructionMaterial::Aluminium);
    assert_eq!(bend.dimensions.span_blocks(), 4);
    assert!((bend.dimensions.radius() - 0.625).abs() < f32::EPSILON);
    assert!((bend.dimensions.inner_diameter() - 0.25).abs() < f32::EPSILON);
}

#[test]
fn current_version_round_trips_pipe_junctions() {
    use crate::{FaceKind, PipeArms, PipeJunctionDimensions, PipeJunctionSpec};
    let arms = PipeArms::single(FaceKind::NegativeY)
        .with(FaceKind::PositiveX)
        .with(FaceKind::PositiveZ);
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::SpawnPipeJunction(
            PipeJunctionSpec::new(
                PipeJunctionDimensions::new(0.20, 0.10).unwrap(),
                arms,
                BuildPose::new(IVec3::new(4, 8, 12), GridRotation::new(1, 2, 3)),
            )
            .with_material(ConstructionMaterial::Aluminium),
        ))
        .unwrap();
    let document = CreationDocument::from_graph(&graph, "Tee", &[]);
    let loaded = round_trip(&document).into_graph().unwrap();
    let junction = loaded
        .graph
        .parts()
        .find_map(|(_, part)| part.as_pipe_junction())
        .unwrap();
    assert_eq!(junction.arms, arms);
    assert_eq!(junction.material, ConstructionMaterial::Aluminium);
    assert!((junction.dimensions.inner_diameter() - 0.10).abs() < f32::EPSILON);
}

#[test]
fn odd_sized_cuboid_keeps_its_half_grid_offset() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [3, 1, 3],
                BuildPose::from_half_grid(IVec3::new(1, 3, 1), GridRotation::new(1, 2, 3)),
            )
            .expect("the dimensions are in range"),
        ))
        .expect("the cuboid spawns");
    let restored = CreationDocument::from_graph(&graph, "Offset", &[])
        .into_graph()
        .expect("the document rebuilds");
    let (_, original) = graph.parts().next().expect("the graph holds its cuboid");
    let (_, rebuilt) = restored
        .graph
        .parts()
        .next()
        .expect("the rebuild holds its cuboid");

    assert_eq!(rebuilt.pose(), original.pose());
}

#[test]
fn current_document_round_trips_five_centimetre_fine_placement() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_position_ticks(IVec3::new(20, 50, -20), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap();
    let document = round_trip(&CreationDocument::from_graph(&graph, "Fine", &[]));
    let PartDoc::Cuboid { pose, .. } = document.parts[0] else {
        unreachable!()
    };
    assert_eq!(pose.translation_ticks, [20, 50, -20]);
    let rebuilt = document.into_graph().unwrap().graph;
    let (_, part) = rebuilt.parts().next().unwrap();
    assert_eq!(
        part.pose().translation_position_ticks(),
        IVec3::new(20, 50, -20)
    );
}

#[test]
fn unsupported_version_is_refused() {
    let (graph, _) = sample();
    let mut document = CreationDocument::from_graph(&graph, "Test Rig", &[]);
    document.version = CREATION_FORMAT_VERSION + 1;

    assert_eq!(
        document.into_graph().err(),
        Some(CreationError::UnsupportedVersion(
            CREATION_FORMAT_VERSION + 1
        ))
    );
}

#[test]
fn weld_to_a_missing_part_is_refused() {
    let (graph, _) = sample();
    let mut document = CreationDocument::from_graph(&graph, "Test Rig", &[]);
    document.welds[0].first.owner = FaceOwnerDoc::Part(99);

    assert_eq!(
        document.into_graph().err(),
        Some(CreationError::MissingPart(99))
    );
}

#[test]
fn wire_to_a_missing_bearing_is_refused() {
    let (graph, _) = sample();
    let mut document = CreationDocument::from_graph(&graph, "Test Rig", &[]);
    document.drive_links[0].bearing = 7;

    assert_eq!(
        document.into_graph().err(),
        Some(CreationError::MissingBearing(7))
    );
}

#[test]
fn out_of_range_cuboid_dimension_is_refused() {
    let (graph, _) = sample();
    let mut document = CreationDocument::from_graph(&graph, "Test Rig", &[]);
    document.parts[0] = super::PartDoc::Cuboid {
        dimensions: [0, 1, 1],
        pose: super::PoseDoc {
            translation_ticks: [0, 0, 0],
            rotation: [0, 0, 0],
        },
        material: crate::ConstructionMaterial::Steel,
        appearance: crate::MaterialAppearance::BAKED,
        layers: Vec::new(),
    };

    assert!(matches!(
        document.into_graph(),
        Err(CreationError::Dimension(_))
    ));
}

#[test]
fn a_shaped_creation_round_trips_its_regions() {
    let spec = CuboidSpec::new(
        [1, 1, 1],
        BuildPose::from_half_grid(IVec3::ONE, GridRotation::default()),
    )
    .unwrap();
    let mut graph = ConstructionGraph::new();
    graph.apply(BuildCommand::Spawn(spec)).unwrap();
    let region = ShapeRegion::new(IVec3::ZERO, IVec3::ONE, ConstructionMaterial::Steel).unwrap();
    let BuildOutcome::RegionAdded(id) = graph.apply(BuildCommand::AddRegion(region)).unwrap()
    else {
        panic!("wrong outcome")
    };
    graph
        .apply(BuildCommand::SetRegionVertices {
            region: id,
            vertices: vec![([1, 1, 1], [-3, -4, -5])],
        })
        .unwrap();

    let document = CreationDocument::from_graph(&graph, "shaped", &[]);
    assert_eq!(document.regions.len(), 1);
    let restored = document.into_graph().unwrap().graph;
    let (_, original) = graph.regions().next().unwrap();
    let (_, replayed) = restored.regions().next().unwrap();
    assert_eq!(replayed, original);
}

#[test]
fn a_fine_placed_shape_region_round_trips_its_exact_origin() {
    let spec = CuboidSpec::new(
        [1, 1, 1],
        BuildPose::from_position_ticks(IVec3::new(60, 50, 50), GridRotation::default()),
    )
    .unwrap();
    let mut graph = ConstructionGraph::new();
    graph.apply(BuildCommand::Spawn(spec)).unwrap();
    let region = ShapeRegion::from_origin_steps(
        IVec3::new(10, 0, 0),
        IVec3::ONE,
        ConstructionMaterial::Steel,
    )
    .unwrap();
    graph.apply(BuildCommand::AddRegion(region)).unwrap();

    let document = CreationDocument::from_graph(&graph, "fine shaped", &[]);
    assert_eq!(document.regions[0].origin_steps, [10, 0, 0]);
    let restored = document.into_graph().unwrap().graph;
    let (_, replayed) = restored.regions().next().unwrap();
    assert_eq!(replayed.origin_steps(), IVec3::new(10, 0, 0));
}

#[test]
fn a_subdivided_region_round_trips_its_cage_planes() {
    let mut graph = ConstructionGraph::new();
    for x in 0..2 {
        let spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(IVec3::new(1 + x * 2, 1, 1), GridRotation::default()),
        )
        .unwrap();
        graph.apply(BuildCommand::Spawn(spec)).unwrap();
    }
    let parts = graph.parts().map(|(id, _)| id).collect::<Vec<_>>();
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec {
            first: parts[0],
            second: parts[1],
        }))
        .unwrap();
    let region = ShapeRegion::new(
        IVec3::ZERO,
        IVec3::new(2, 1, 1),
        ConstructionMaterial::Steel,
    )
    .unwrap();
    let BuildOutcome::RegionAdded(id) = graph.apply(BuildCommand::AddRegion(region)).unwrap()
    else {
        panic!("wrong outcome")
    };
    graph
        .apply(BuildCommand::SubdivideRegion {
            region: id,
            axis: 0,
            position: 1,
        })
        .unwrap();

    let document = CreationDocument::from_graph(&graph, "subdivided", &[]);
    let restored = document.into_graph().unwrap().graph;
    let (_, replayed) = restored.regions().next().unwrap();
    assert_eq!(replayed.plane_counts(), [3, 2, 2]);
}

#[test]
fn transmissions_round_trip_when_arena_order_precedes_their_parents() {
    let mut graph = ConstructionGraph::new();
    let first_disposable = spawned(
        graph
            .apply(BuildCommand::Spawn(cuboid([1, 1, 1], IVec3::new(20, 1, 0))))
            .unwrap(),
    );
    let second_disposable = spawned(
        graph
            .apply(BuildCommand::Spawn(cuboid([1, 1, 1], IVec3::new(24, 1, 0))))
            .unwrap(),
    );
    let engine = spawned(
        graph
            .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                EngineKind::Gas,
                BuildPose::new(IVec3::ZERO, GridRotation::default()),
            )))
            .unwrap(),
    );
    graph.apply(BuildCommand::Remove(first_disposable)).unwrap();
    graph
        .apply(BuildCommand::Remove(second_disposable))
        .unwrap();
    let first = spawned(
        graph
            .apply(BuildCommand::AttachTransmission {
                parent: engine,
                spec: graph.next_transmission_spec(engine).unwrap(),
            })
            .unwrap(),
    );
    graph
        .apply(BuildCommand::AttachTransmission {
            parent: first,
            spec: graph.next_transmission_spec(first).unwrap(),
        })
        .unwrap();

    let document = CreationDocument::from_graph(&graph, "Forward parents", &[]);
    assert!(matches!(
        document.parts.as_slice(),
        [
            PartDoc::Transmission { parent: 1, .. },
            PartDoc::Transmission { parent: 2, .. },
            PartDoc::Engine { .. }
        ]
    ));

    let restored = round_trip(&document).into_graph().unwrap().graph;
    assert_eq!(restored.part_count(), 3);
    let restored_engine = restored
        .parts()
        .find_map(|(part, spec)| matches!(spec, PartSpec::Engine(_)).then_some(part))
        .unwrap();
    let restored_first = restored
        .parts()
        .find_map(|(part, spec)| {
            (matches!(spec, PartSpec::Transmission(_))
                && restored.transmission_parent(part) == Some(restored_engine))
            .then_some(part)
        })
        .unwrap();
    assert!(restored.parts().any(|(part, spec)| {
        matches!(spec, PartSpec::Transmission(_))
            && restored.transmission_parent(part) == Some(restored_first)
    }));
}

#[test]
fn transmission_parents_and_gearbox_settings_round_trip() {
    let mut graph = ConstructionGraph::new();
    let controller = spawned(
        graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(IVec3::ZERO, GridRotation::default()),
            )))
            .unwrap(),
    );
    let engine = spawned(
        graph
            .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                EngineKind::Gas,
                BuildPose::new(IVec3::new(2, 0, 0), GridRotation::default()),
            )))
            .unwrap(),
    );
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(controller, FaceKind::PositiveX),
            second: FaceRef::part(engine, FaceKind::NegativeX),
        }))
        .unwrap();
    let spec = graph.next_transmission_spec(engine).unwrap();
    let transmission = spawned(
        graph
            .apply(BuildCommand::AttachTransmission {
                parent: engine,
                spec,
            })
            .unwrap(),
    );
    graph
        .apply(BuildCommand::SetGearboxMode {
            controller,
            kind: EngineKind::Gas,
            mode: ShiftMode::Manual,
        })
        .unwrap();
    graph
        .apply(BuildCommand::SetGearboxRatios {
            controller,
            kind: EngineKind::Gas,
            ratios: vec![4.0, 0.8],
        })
        .unwrap();
    graph
        .apply(BuildCommand::SetGearboxBindings {
            controller,
            kind: EngineKind::Gas,
            up: GearKeyChord::new(GearKey::PageUp),
            down: GearKeyChord {
                shift: true,
                ..GearKeyChord::new(GearKey::PageDown)
            },
        })
        .unwrap();
    graph
        .apply(BuildCommand::SetGasDivider {
            controller,
            reverse_gears: 2,
        })
        .unwrap();

    let document = CreationDocument::from_graph(&graph, "Geared", &[]);
    assert_eq!(document.gearbox_configs.len(), 1);
    assert_eq!(
        document.welds.len(),
        1,
        "the required weld is derived from its parent"
    );
    let restored = round_trip(&document).into_graph().unwrap().graph;
    let restored_transmission = restored
        .parts()
        .find_map(|(id, spec)| matches!(spec, PartSpec::Transmission(_)).then_some(id))
        .unwrap();
    let restored_engine = restored.transmission_parent(restored_transmission).unwrap();
    assert!(matches!(
        restored.part(restored_engine),
        Some(PartSpec::Engine(_))
    ));
    let restored_controller = restored
        .parts()
        .find_map(|(id, spec)| matches!(spec, PartSpec::Controller(_)).then_some(id))
        .unwrap();
    let config = restored
        .gearbox_config(restored_controller, EngineKind::Gas)
        .unwrap();
    assert_eq!(config.mode(), ShiftMode::Manual);
    assert_eq!(config.ratios(), &[4.0, 0.8]);
    assert_eq!(config.reverse_gears(), 2);
    assert_eq!(config.gear_up().key, GearKey::PageUp);
    assert!(config.gear_down().shift);
    assert!(graph.part(transmission).is_some());
}

#[test]
fn incomplete_matching_stacks_remain_saveable_and_loadable() {
    let mut graph = ConstructionGraph::new();
    let controller = spawned(
        graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(IVec3::ZERO, GridRotation::default()),
            )))
            .unwrap(),
    );
    let first = spawned(
        graph
            .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                EngineKind::Electric,
                BuildPose::new(IVec3::new(2, 0, 0), GridRotation::default()),
            )))
            .unwrap(),
    );
    let second = spawned(
        graph
            .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                EngineKind::Electric,
                BuildPose::new(IVec3::new(4, 0, 0), GridRotation::default()),
            )))
            .unwrap(),
    );
    for (left, right) in [(controller, first), (first, second)] {
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(left, FaceKind::PositiveX),
                second: FaceRef::part(right, FaceKind::NegativeX),
            }))
            .unwrap();
    }
    let spec = graph.next_transmission_spec(first).unwrap();
    graph
        .apply(BuildCommand::AttachTransmission {
            parent: first,
            spec,
        })
        .unwrap();

    let document = CreationDocument::from_graph(&graph, "Incomplete", &[]);
    let restored = round_trip(&document).into_graph().unwrap().graph;
    assert!(matches!(
        restored.compile(),
        Err(crate::TopologyError::TransmissionDepthMismatch {
            kind: EngineKind::Electric,
            ..
        })
    ));
}

#[test]
fn shape_feature_order_and_topology_provenance_round_trip() {
    let mut graph = ConstructionGraph::new();
    let part = spawned(
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [2, 2, 2],
                    BuildPose::new(IVec3::ZERO, GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap(),
    );
    let owner = SolidOwner::Part(part);
    let first_edge = graph.evaluated_solid(owner).unwrap().logical_edges[0].key;
    let BuildOutcome::ShapeFeatureAdded(first) = graph
        .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
            [EdgeChainRef {
                owner,
                edge: first_edge,
            }],
            EdgeTreatment::Chamfer,
            10,
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let generated = graph
        .evaluated_solid(owner)
        .unwrap()
        .logical_edges
        .iter()
        .find(|edge| edge.key.source == TopologySource::Feature(first))
        .expect("the chamfer introduces selectable edges")
        .key;
    let BuildOutcome::ShapeFeatureAdded(second) = graph
        .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
            [EdgeChainRef {
                owner,
                edge: generated,
            }],
            EdgeTreatment::Fillet,
            5,
        )))
        .unwrap()
    else {
        unreachable!()
    };

    let patch = graph
        .evaluated_solid(owner)
        .unwrap()
        .surfaces
        .iter()
        .find(|surface| surface.key.source == TopologySource::Feature(second))
        .expect("the fillet creates generated surface patches")
        .key;
    let patch_face = FaceRef::patch(part, FaceKind::PositiveX, patch);
    let patch_geometry = graph.face_geometry(patch_face).unwrap();
    let socket = BearingSocket {
        kind: crate::JointKind::Rotational,
        axis: Vec3::ZERO,
        source: patch_face,
        anchor: patch_geometry.center,
        dimensions: BearingDimensions::default(),
    };

    let document = CreationDocument::from_graph(&graph, "Features", &[socket]);
    assert_eq!(document.version, CREATION_FORMAT_VERSION);
    assert_eq!(document.shape_features.len(), 2);
    assert!(matches!(
        document.shape_features[1].targets[0].edge.source,
        TopologySourceDoc::Feature(0)
    ));
    let loaded = round_trip(&document).into_graph().unwrap();
    assert_eq!(
        loaded.sockets[0].source.patch.map(|key| key.local),
        Some(patch.local)
    );
    let restored = loaded.graph;
    assert_eq!(restored.shape_features().count(), 2);
    let restored_owner = SolidOwner::Part(restored.parts().next().unwrap().0);
    assert!(restored.evaluated_solid(restored_owner).is_ok());
}

#[test]
fn nested_cylinder_features_round_trip_and_compile_under_current_format() {
    let mut graph = ConstructionGraph::new();
    let dimensions = CylinderDimensions::new(0.5, 0.0, 0.5).unwrap();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
            dimensions,
            BuildPose::default(),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let owner = SolidOwner::Part(part);
    let rim = graph
        .evaluated_solid(owner)
        .unwrap()
        .logical_edges
        .iter()
        .find(|edge| edge.closed && edge.convex)
        .unwrap()
        .key;
    let BuildOutcome::ShapeFeatureAdded(chamfer) = graph
        .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
            [EdgeChainRef { owner, edge: rim }],
            EdgeTreatment::Chamfer,
            20,
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let follow_up = graph
        .evaluated_solid(owner)
        .unwrap()
        .logical_edges
        .iter()
        .find(|edge| {
            edge.key.source == TopologySource::Feature(chamfer) && edge.closed && edge.convex
        })
        .unwrap()
        .key;
    graph
        .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
            [EdgeChainRef {
                owner,
                edge: follow_up,
            }],
            EdgeTreatment::Fillet,
            20,
        )))
        .unwrap();

    let document = CreationDocument::from_graph(&graph, "Nested cylinder", &[]);
    assert_eq!(document.version, CREATION_FORMAT_VERSION);
    let restored = round_trip(&document).into_graph().unwrap().graph;
    let restored_owner = SolidOwner::Part(restored.parts().next().unwrap().0);
    let solid = restored.evaluated_solid(restored_owner).unwrap();
    assert!(solid.volume() > 0.0);
    assert!(!solid.cells.is_empty());
    let compiled = restored.compile().unwrap();
    assert!(compiled.compounds[0].mass_properties.mass > 0.0);
    assert!(!compiled.colliders.is_empty());
}

#[test]
fn material_layers_survive_a_serialized_round_trip() {
    let rubber = |spec: PartSpec, face, thickness| {
        spec.with_layer(
            face,
            thickness,
            ConstructionMaterial::Rubber,
            crate::MaterialAppearance::BAKED,
        )
        .unwrap()
    };
    let pipe = PartSpec::Cylinder(CylinderSpec::new(
        CylinderDimensions::new(1.0, 0.5, 0.5).unwrap(),
        crate::BuildPose::from_position_ticks(
            IVec3::new(0, 400, 0),
            crate::GridRotation::default(),
        ),
    ));
    let pipe = rubber(
        rubber(
            rubber(pipe, crate::LayerFace::OuterWall, 0.25),
            crate::LayerFace::Bore,
            0.1,
        ),
        crate::LayerFace::Face(crate::FaceKind::PositiveY),
        0.01,
    );
    let block = rubber(
        PartSpec::Cuboid(cuboid([2, 2, 2], IVec3::new(8, 1, 0))),
        crate::LayerFace::Face(crate::FaceKind::NegativeX),
        0.05,
    );
    let mut graph = ConstructionGraph::new();
    graph
        .apply(crate::BuildCommand::SpawnCylinder(
            pipe.as_cylinder().unwrap(),
        ))
        .unwrap();
    graph
        .apply(crate::BuildCommand::Spawn(block.as_cuboid().unwrap()))
        .unwrap();
    let document = CreationDocument::from_graph(&graph, "Layers", &Vec::new());
    let restored = round_trip(&document).into_graph().unwrap().graph;
    let mut parts = restored.parts().map(|(_, spec)| *spec);
    let restored_pipe = parts.find(|spec| spec.as_cylinder().is_some()).unwrap();
    assert!(restored_pipe.shares_core_with(pipe));
    assert_eq!(restored_pipe.material_layers().len(), 3);
    assert_eq!(restored_pipe.pose(), pipe.pose());
    assert_eq!(
        restored
            .parts()
            .map(|(_, spec)| *spec)
            .find(|spec| spec.as_cylinder().is_none()),
        Some(block)
    );
}
