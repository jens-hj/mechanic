use super::{Piston, PistonDimensions, PistonError, PistonMount};
use crate::{
    BearingSocket, BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
    CuboidSpec, DriveLinkSpec, DriveProgram, DriveState, DriveTarget, FaceKind, FaceRef,
    GraphError, GridRotation, JointKind, PartId,
};
use bevy_math::{IVec3, Vec3};

fn dimensions(blocks: u8, stages: u8) -> PistonDimensions {
    PistonDimensions::new(blocks, stages).expect("test dimensions are legal")
}

#[test]
fn every_configuration_extends_to_a_whole_number_of_blocks() {
    assert_eq!(dimensions(2, 4).extended_blocks(), 10);
    assert_eq!(dimensions(8, 6).extended_blocks(), 56);
    assert!((dimensions(2, 4).extended() - 2.5).abs() < 1.0e-6);
    assert!((dimensions(8, 6).stroke() - 12.0).abs() < 1.0e-6);
    assert!((dimensions(2, 6).head_radius() * 2.0 - 0.136).abs() < 1.0e-6);
}

#[test]
fn counts_outside_the_pack_ranges_are_rejected() {
    assert_eq!(PistonDimensions::new(1, 1), Err(PistonError::Blocks));
    assert_eq!(PistonDimensions::new(9, 1), Err(PistonError::Blocks));
    assert_eq!(PistonDimensions::new(2, 0), Err(PistonError::Stages));
    assert_eq!(PistonDimensions::new(2, 7), Err(PistonError::Stages));
}

#[test]
fn stages_draw_largest_first() {
    let offsets = dimensions(2, 3).stage_offsets(0.75);
    assert!((offsets[0] - 0.5).abs() < 1.0e-6);
    assert!((offsets[1] - 0.75).abs() < 1.0e-6);
    assert!((offsets[2] - 0.75).abs() < 1.0e-6);
    assert!((offsets[5] - 0.75).abs() < 1.0e-6);
}

#[test]
fn a_single_stage_two_block_piston_weighs_what_the_guide_measured() {
    let mass = dimensions(2, 1)
        .mass_elements()
        .iter()
        .map(|element| element.mass)
        .sum::<f32>();
    assert!((mass - 190.0).abs() < 1.0, "{mass}");
}

#[test]
fn a_side_mount_lies_on_its_supporting_face() {
    let piston = Piston {
        dimensions: dimensions(2, 4),
        mount: PistonMount::Side {
            mount_normal: Vec3::Y,
        },
    };
    let base = piston.base_center(Vec3::ZERO, Vec3::X);
    assert!(base.distance(Vec3::new(-0.25, 0.125, 0.0)) < 1.0e-6);
    let head = piston.head_center(Vec3::ZERO, Vec3::X, 1.0);
    assert!(head.distance(Vec3::new(1.25, 0.125, 0.0)) < 1.0e-6);
    let rotation = piston.rotation(Vec3::X).expect("orthonormal frame");
    assert!((rotation * Vec3::Y).distance(Vec3::X) < 1.0e-6);
    assert!((rotation * Vec3::Z).distance(Vec3::Y) < 1.0e-6);
    assert_eq!(piston.rotation(Vec3::Y), Err(PistonError::Frame));
}

fn spawn_block(graph: &mut ConstructionGraph, ticks: IVec3) -> PartId {
    let pose = BuildPose::from_position_ticks(ticks, GridRotation::default());
    let spec = CuboidSpec::new([1, 1, 1], pose).unwrap();
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
        panic!("expected a block");
    };
    part
}

/// A support block with a collapsed 2 x 4 piston on its top face and a block on the head.
fn piston_graph(side: bool) -> (ConstructionGraph, BearingSpec) {
    let mut graph = ConstructionGraph::new();
    let base = spawn_block(&mut graph, IVec3::new(0, 850, 0));
    let anchor = Vec3::new(0.0, 2.25, 0.0);
    let (mount, axis, head, face) = if side {
        (
            PistonMount::Side {
                mount_normal: Vec3::Y,
            },
            Vec3::X,
            IVec3::new(150, 950, 0),
            FaceKind::NegativeX,
        )
    } else {
        (
            PistonMount::End,
            Vec3::Y,
            IVec3::new(0, 1150, 0),
            FaceKind::NegativeY,
        )
    };
    let target = spawn_block(&mut graph, head);
    let piston = Piston {
        dimensions: dimensions(2, 4),
        mount,
    };
    let bearing = BearingSpec::new(
        FaceRef::part(base, FaceKind::PositiveY),
        FaceRef::part(target, face),
        anchor,
        axis,
    )
    .with_kind(JointKind::Piston(piston));
    (graph, bearing)
}

#[test]
fn end_and_side_mounted_pistons_compile_to_one_bounded_translation() {
    for side in [false, true] {
        let (mut graph, bearing) = piston_graph(side);
        graph.apply(BuildCommand::AddBearing(bearing)).unwrap();
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.compounds.len(), 2);
        assert_eq!(compiled.bearings.len(), 1);
        assert!(compiled.bearings[0].kind.is_translational());
        assert_eq!(
            compiled.bearings[0].kind.bounds().map(f32::to_bits),
            [0.0_f32, 2.0].map(f32::to_bits)
        );
    }
}

#[test]
fn a_bare_piston_is_a_joint_whose_head_body_later_attachments_join() {
    for side in [false, true] {
        let (mut graph, attached) = piston_graph(side);
        let bare = BearingSpec::bare(
            attached.source,
            attached.shared_anchor,
            attached.axis,
            attached.kind,
        );
        graph.apply(BuildCommand::AddBearing(bare)).unwrap();
        assert_eq!(
            graph.apply(BuildCommand::AddBearing(bare)),
            Err(GraphError::PistonHeadOccupied)
        );

        // The loose head block is a third body until it is attached.
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.compounds.len(), 3);
        assert_eq!(compiled.bearings.len(), 1);
        let head = &compiled.compounds[compiled.bearings[0].compound_b as usize];
        assert!(head.source_parts.is_empty());
        assert!(!head.is_static);
        let JointKind::Piston(piston) = attached.kind else {
            unreachable!()
        };
        let head_mass = piston
            .dimensions
            .mass_elements()
            .iter()
            .filter(|element| element.opposite)
            .map(|element| element.mass)
            .sum::<f32>();
        assert!((head.mass_properties.mass - head_mass).abs() < 1.0e-3);

        graph.apply(BuildCommand::AddBearing(attached)).unwrap();
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.compounds.len(), 2);
        assert_eq!(compiled.bearings.len(), 1);
        assert_eq!(
            compiled.loop_topology.bearing_coordinates.len(),
            2,
            "both rows address the one joint"
        );
    }
}

#[test]
fn only_hardware_with_its_own_head_has_a_joint_before_anything_is_attached() {
    let (mut graph, attached) = piston_graph(false);
    let rotary = BearingSpec::bare(
        attached.source,
        attached.shared_anchor,
        attached.axis,
        JointKind::Rotational,
    );
    assert_eq!(
        graph.apply(BuildCommand::AddBearing(rotary)),
        Err(GraphError::BearingWithoutTarget)
    );
}

#[test]
fn a_head_attachment_off_the_crown_plane_is_rejected() {
    let (mut graph, mut bearing) = piston_graph(false);
    let JointKind::Piston(ref mut piston) = bearing.kind else {
        unreachable!()
    };
    piston.dimensions = dimensions(3, 4);
    assert_eq!(
        graph.apply(BuildCommand::AddBearing(bearing)),
        Err(GraphError::BearingAnchorOutsideFaces)
    );
}

#[test]
fn piston_hardware_weighs_on_both_mounts_and_on_an_unattached_support() {
    let (mut graph, bearing) = piston_graph(false);
    let bare = graph.compile().unwrap();
    let block = bare.compounds[0].mass_properties.mass;
    let socket = BearingSocket {
        kind: bearing.kind,
        axis: bearing.axis,
        source: bearing.source,
        anchor: bearing.shared_anchor,
        dimensions: bearing.dimensions,
    };
    let hardware = dimensions(2, 4)
        .mass_elements()
        .iter()
        .map(|element| element.mass)
        .sum::<f32>();
    let carried = graph.compile_with_sockets([], &[socket]).unwrap();
    let total = |creation: &crate::CompiledCreation| {
        creation
            .compounds
            .iter()
            .map(|compound| compound.mass_properties.mass)
            .sum::<f32>()
    };
    assert!((total(&carried) - 2.0 * block - hardware).abs() < 1.0e-2);

    graph.apply(BuildCommand::AddBearing(bearing)).unwrap();
    let attached = graph.compile_with_sockets([], &[socket]).unwrap();
    assert!((total(&attached) - 2.0 * block - hardware).abs() < 1.0e-2);
    assert!(
        attached
            .compounds
            .iter()
            .all(|compound| compound.mass_properties.mass > block + 1.0)
    );
}

#[test]
fn a_piston_wire_programs_extension_from_collapsed_and_cannot_be_reversed() {
    let (mut graph, bearing) = piston_graph(true);
    let BuildOutcome::BearingAdded(id) = graph.apply(BuildCommand::AddBearing(bearing)).unwrap()
    else {
        panic!("expected bearing")
    };
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(crate::ControllerSpec::new(
            BuildPose::new(IVec3::new(0, 40, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        panic!("expected controller")
    };
    let mut link = DriveLinkSpec::new_linear(controller, id, bearing.kind.bounds());
    let limits = link.linear_limits.unwrap();
    assert_eq!(
        [limits.minimum(), limits.maximum()].map(f32::to_bits),
        [0.0_f32, 2.0].map(f32::to_bits)
    );
    link.program = DriveProgram::new(
        &[DriveState::new(DriveTarget::LinearPosition(1.5)).unwrap()],
        false,
    )
    .unwrap();
    assert_eq!(
        graph.apply(BuildCommand::AddDriveLink(DriveLinkSpec {
            reversed: true,
            ..link
        })),
        Err(GraphError::IncompatibleDrive)
    );
    graph.apply(BuildCommand::AddDriveLink(link)).unwrap();
    assert!(DriveState::new(DriveTarget::LinearPosition(12.0)).is_ok());
}

#[test]
fn pistons_survive_serialization_and_creation_transforms() {
    let (mut graph, bearing) = piston_graph(true);
    graph.apply(BuildCommand::AddBearing(bearing)).unwrap();
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::bare(
            bearing.source,
            bearing.shared_anchor,
            bearing.axis,
            bearing.kind,
        )))
        .unwrap();
    let document = crate::CreationDocument::from_graph(&graph, "piston", &[]);
    let encoded = ron::to_string(&document).unwrap();
    let mut document: crate::CreationDocument = ron::from_str(&encoded).unwrap();
    assert_eq!(document.bearings[0].kind, bearing.kind);
    assert!(document.bearings[0].target.is_some());
    assert!(document.bearings[1].target.is_none());
    document.transform_cardinal(1, IVec3::ZERO);
    let loaded = document.into_graph().unwrap();
    assert_eq!(loaded.graph.bearings().count(), 2);
    let (_, turned) = loaded.graph.bearings().next().unwrap();
    assert!(
        turned.axis.abs_diff_eq(Vec3::NEG_Z, 1.0e-5),
        "{}",
        turned.axis
    );
    assert_eq!(loaded.graph.compile().unwrap().bearings.len(), 1);
}
