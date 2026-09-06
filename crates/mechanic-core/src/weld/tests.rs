use super::*;

fn selection(feature: WeldFeature, normal: Vec3) -> WeldSelection {
    WeldSelection {
        feature,
        point: Vec3::ZERO,
        normal,
        tangent: Vec3::X,
    }
}

#[test]
fn all_feature_pairs_oppose_normals_and_keep_the_seed() {
    let features = [
        WeldFeature::Face,
        WeldFeature::Edge([-Vec3::X, Vec3::X]),
        WeldFeature::Vertex(Vec3::ZERO),
    ];
    for source in features {
        for destination in features {
            let alignment =
                WeldAlignment::new(selection(source, Vec3::Y), selection(destination, -Vec3::Y))
                    .unwrap();
            let pose = alignment.place(Vec3::ZERO, 0).unwrap();
            assert!(pose.point(Vec3::ZERO).length() < EPSILON);
            assert!(pose.vector(Vec3::Y).abs_diff_eq(Vec3::Y, EPSILON));
        }
    }
}

#[test]
fn finite_edges_and_vertex_incidence_are_preserved_in_both_orders() {
    let vertex = selection(WeldFeature::Vertex(Vec3::ZERO), Vec3::Y);
    let edge = selection(WeldFeature::Edge([-Vec3::X, Vec3::X]), -Vec3::Y);
    for (a, b) in [(vertex, edge), (edge, vertex)] {
        let alignment = WeldAlignment::new(a, b).unwrap();
        assert!(alignment.place(Vec3::X, 0).is_ok());
        assert!(alignment.place(Vec3::X * 1.01, 0).is_err());
        assert!(alignment.place(Vec3::Z * 0.05, 0).is_err());
    }
    let fixed = WeldAlignment::new(vertex, vertex).unwrap();
    assert!(fixed.place(Vec3::X * 0.05, 0).is_err());
}

#[test]
fn edge_alignment_skips_rotations_that_break_collinearity() {
    let edge = WeldFeature::Edge([-Vec3::X, Vec3::X]);
    let alignment =
        WeldAlignment::new(selection(edge, Vec3::Y), selection(edge, -Vec3::Y)).unwrap();
    assert_eq!(alignment.next_rotation(Vec3::ZERO, 0), Some(12));
    assert_eq!(alignment.next_rotation(Vec3::ZERO, 12), Some(0));
    assert!(alignment.place(Vec3::X * 2.01, 0).is_err());
}

#[test]
fn sloped_mating_faces_use_one_rigid_transform() {
    let source = selection(WeldFeature::Face, Vec3::Y);
    let mut destination = selection(WeldFeature::Face, Vec3::new(0.0, 1.0, 1.0).normalize());
    destination.point = Vec3::new(2.0, 3.0, 4.0);
    let alignment = WeldAlignment::new(source, destination).unwrap();
    let frame = alignment.place(Vec3::X * 0.25, 1).unwrap();
    assert!(
        frame
            .point(source.point)
            .abs_diff_eq(destination.point + Vec3::X * 0.25, EPSILON)
    );
    assert!(
        frame
            .vector(source.normal)
            .abs_diff_eq(-destination.normal, EPSILON)
    );
}

#[test]
fn switching_snap_increment_does_not_move_a_stationary_pointer() {
    let mut snap = WeldSnap::default();
    assert_eq!(
        snap.update(Vec2::new(0.16, 0.0), false),
        Vec2::new(0.25, 0.0)
    );
    assert_eq!(
        snap.update(Vec2::new(0.16, 0.0), true),
        Vec2::new(0.25, 0.0)
    );
    assert!(
        snap.update(Vec2::new(0.21, 0.0), true)
            .abs_diff_eq(Vec2::new(0.30, 0.0), EPSILON)
    );
    assert!(
        snap.update(Vec2::new(0.21, 0.0), false)
            .abs_diff_eq(Vec2::new(0.30, 0.0), EPSILON)
    );
}

fn rect(x: f32, y: f32, width: f32, height: f32) -> Vec<Vec2> {
    vec![
        Vec2::new(x, y),
        Vec2::new(x + width, y),
        Vec2::new(x + width, y + height),
        Vec2::new(x, y + height),
    ]
}

#[test]
fn exact_contact_and_adjacent_material_contain_a_square() {
    let destination = vec![rect(0.0, 0.0, 0.05, 0.05)];
    assert!(weld_contact_square(&destination, &destination).is_ok());
    let source = vec![rect(0.0, 0.0, 0.02, 0.05), rect(0.02, 0.0, 0.03, 0.05)];
    assert!(weld_contact_square(&source, &destination).is_ok());
}

#[test]
fn narrow_disconnected_and_holed_contacts_are_rejected() {
    let destination = vec![rect(-1.0, -1.0, 2.0, 2.0)];
    for source in [
        vec![rect(0.0, 0.0, 0.049, 0.10)],
        vec![rect(0.0, 0.0, 0.025, 0.05), rect(0.026, 0.0, 0.025, 0.05)],
        vec![
            rect(0.0, 0.0, 0.05, 0.02),
            rect(0.0, 0.03, 0.05, 0.02),
            rect(0.0, 0.02, 0.02, 0.01),
            rect(0.03, 0.02, 0.02, 0.01),
        ],
        vec![rect(0.0, 0.0, 0.0, 0.05)],
    ] {
        assert_eq!(
            weld_contact_square(&source, &destination),
            Err(WeldRejection::InsufficientContact)
        );
        assert_eq!(
            weld_contact_square(&destination, &source),
            Err(WeldRejection::InsufficientContact)
        );
    }
}

fn spawn(graph: &mut crate::ConstructionGraph, units: bevy_math::IVec3) -> crate::PartId {
    let spec = crate::CuboidSpec::new(
        [2, 2, 2],
        crate::BuildPose::new(units, crate::GridRotation::default()),
    )
    .unwrap();
    let crate::BuildOutcome::Spawned(part) = graph.apply(crate::BuildCommand::Spawn(spec)).unwrap()
    else {
        panic!("spawn expected")
    };
    part
}

#[test]
fn topology_picks_reject_stale_revisions_and_nonincident_faces() {
    let mut graph = crate::ConstructionGraph::new();
    let part = spawn(&mut graph, bevy_math::IVec3::ZERO);
    let owner = crate::SolidOwner::Part(part);
    let solid = graph.evaluated_solid(owner).unwrap();
    let surface = &solid.surfaces[0];
    let normal = surface.normal;
    let patch = surface.key;
    let pick = WeldPick::new(
        &graph,
        owner,
        WeldFeatureRef::Face(patch),
        patch,
        normal * 0.25,
    )
    .unwrap();
    assert!(pick.resolve(&graph).is_ok());
    assert!(
        WeldPick::new(
            &graph,
            owner,
            WeldFeatureRef::Face(patch),
            patch,
            normal * 0.25 + normal.any_orthonormal_vector()
        )
        .is_err()
    );
    spawn(&mut graph, bevy_math::IVec3::new(8, 0, 0));
    assert!(pick.resolve(&graph).is_err());
}

#[test]
fn cube_edges_and_real_corners_resolve_against_adjacent_surfaces() {
    let mut graph = crate::ConstructionGraph::new();
    let part = spawn(&mut graph, bevy_math::IVec3::ZERO);
    let owner = crate::SolidOwner::Part(part);
    let solid = graph.evaluated_solid(owner).unwrap();
    for edge in &solid.logical_edges {
        let ends = straight_edge(&solid, edge).unwrap();
        let half = solid.half_edges[edge.half_edges[0] as usize];
        let mating = solid.surfaces[half.face as usize].key;
        let pick = WeldPick::new(
            &graph,
            owner,
            WeldFeatureRef::Edge(edge.key),
            mating,
            (ends[0] + ends[1]) * 0.5,
        )
        .unwrap();
        assert!(matches!(
            pick.resolve(&graph).unwrap().feature,
            WeldFeature::Edge(_)
        ));
    }
    for half in &solid.half_edges {
        let mating = solid.surfaces[half.face as usize].key;
        WeldPick::new(
            &graph,
            owner,
            WeldFeatureRef::Vertex(half.origin),
            mating,
            solid.vertices[half.origin as usize].position,
        )
        .unwrap();
    }
}

#[test]
fn placement_moves_first_assembly_and_preserves_joint_defaults_and_destination() {
    use crate::{BearingSpec, BuildCommand, ConstructionFrame, FaceKind, FaceRef};
    let mut graph = crate::ConstructionGraph::new();
    let base = spawn(&mut graph, bevy_math::IVec3::ZERO);
    let arm = spawn(&mut graph, bevy_math::IVec3::new(0, 2, 0));
    let destination = spawn(&mut graph, bevy_math::IVec3::new(16, 0, 0));
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(base, FaceKind::PositiveY),
            FaceRef::part(arm, FaceKind::NegativeY),
            Vec3::Y * 0.25,
            Vec3::Y,
        )))
        .unwrap();
    let original = graph.clone();
    let transform = ConstructionFrame::new(Vec3::X * 3.5, Quat::IDENTITY).unwrap();
    let placement = WeldPlacement::stage(
        &graph,
        FaceRef::part(base, FaceKind::PositiveX),
        FaceRef::part(destination, FaceKind::NegativeX),
        transform,
        [],
    )
    .unwrap();
    assert!(graph.shares_revision(&original));
    assert_eq!(
        placement.graph.part_position(destination),
        graph.part_position(destination)
    );
    assert!(
        placement
            .graph
            .part_position(arm)
            .unwrap()
            .abs_diff_eq(Vec3::new(3.5, 0.5, 0.0), EPSILON)
    );
    assert_eq!(placement.creation.bearings.len(), 1);
    let world =
        ConstructionFrame::new(Vec3::new(5.0, 3.0, -1.0), Quat::from_rotation_z(0.3)).unwrap();
    for (index, frame) in placement.default_source_frames(world) {
        let body = &placement.creation.compounds[index];
        assert!(
            frame
                .translation()
                .abs_diff_eq(world.point(body.root_translation), EPSILON)
        );
        assert!(
            frame
                .rotation()
                .abs_diff_eq(world.rotation() * body.root_rotation, EPSILON)
        );
    }
    let document = crate::CreationDocument::from_graph(&placement.graph, "Placed", &[]);
    let restored = document.into_graph().unwrap().graph;
    assert!(
        restored
            .part_position(arm)
            .unwrap()
            .abs_diff_eq(placement.graph.part_position(arm).unwrap(), EPSILON)
    );
    assert_eq!(restored.compile().unwrap().bearings.len(), 1);
}

#[test]
fn default_assembly_obstructions_and_terrain_anchors_reject_transactionally() {
    use crate::{BuildCommand, ConstructionFrame, FaceKind, FaceRef, RigidLinkSpec};
    let mut graph = crate::ConstructionGraph::new();
    let source = spawn(&mut graph, bevy_math::IVec3::ZERO);
    let limb = spawn(&mut graph, bevy_math::IVec3::new(0, 2, 0));
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec {
            first: source,
            second: limb,
        }))
        .unwrap();
    let destination = spawn(&mut graph, bevy_math::IVec3::new(16, 0, 0));
    let transform = ConstructionFrame::new(Vec3::X * 3.5, Quat::IDENTITY).unwrap();
    let source_face = FaceRef::part(source, FaceKind::PositiveX);
    let destination_face = FaceRef::part(destination, FaceKind::NegativeX);
    assert!(
        WeldPlacement::stage(&graph, source_face, destination_face, transform, [limb])
            .unwrap_err()
            .contains("anchored")
    );
    let obstruction = spawn(&mut graph, bevy_math::IVec3::new(14, 2, 0));
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec {
            first: destination,
            second: obstruction,
        }))
        .unwrap();
    let original = graph.clone();
    assert!(
        WeldPlacement::stage(&graph, source_face, destination_face, transform, [])
            .unwrap_err()
            .contains("default pose")
    );
    assert!(graph.shares_revision(&original));
    assert_eq!(graph.weld_count(), 0);
}

#[test]
fn cylinder_ends_are_selectable_but_curved_rims_and_tessellation_vertices_are_not() {
    let mut graph = crate::ConstructionGraph::new();
    let dimensions = crate::CylinderDimensions::new(0.5, 0.0, 0.5).unwrap();
    let spec = crate::CylinderSpec::new(dimensions, crate::BuildPose::default());
    let crate::BuildOutcome::Spawned(part) = graph
        .apply(crate::BuildCommand::SpawnCylinder(spec))
        .unwrap()
    else {
        panic!("spawn expected")
    };
    let owner = crate::SolidOwner::Part(part);
    let solid = graph.evaluated_solid(owner).unwrap();
    let end = solid
        .surfaces
        .iter()
        .find(|surface| surface.normal.abs_diff_eq(Vec3::Y, EPSILON))
        .unwrap();
    WeldPick::new(
        &graph,
        owner,
        WeldFeatureRef::Face(end.key),
        end.key,
        Vec3::Y * 0.25,
    )
    .unwrap();
    for edge in &solid.logical_edges {
        assert!(straight_edge(&solid, edge).is_none());
    }
    let half = solid.half_edges[end.half_edge as usize];
    assert!(
        WeldPick::new(
            &graph,
            owner,
            WeldFeatureRef::Vertex(half.origin),
            end.key,
            solid.vertices[half.origin as usize].position
        )
        .is_err()
    );
}

#[test]
fn invalid_material_cannot_supply_missing_contact() {
    let invalid = vec![vec![Vec2::splat(f32::NAN); 3]];
    let valid = vec![rect(0.0, 0.0, 0.1, 0.1)];
    assert_eq!(
        weld_contact_square(&invalid, &valid),
        Err(WeldRejection::InsufficientContact)
    );
    assert_eq!(
        weld_contact_square(&valid, &invalid),
        Err(WeldRejection::InsufficientContact)
    );
}

#[test]
fn a_cylinder_end_weld_uses_the_entire_annulus_and_keeps_the_hole() {
    use crate::{
        BuildCommand, BuildOutcome, BuildPose, CylinderDimensions, CylinderSpec, FaceKind, FaceRef,
        GridRotation, WeldSpec,
    };
    let mut graph = crate::ConstructionGraph::new();
    let BuildOutcome::Spawned(ring) = graph
        .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
            CylinderDimensions::new(1.0, 0.5, 0.25).unwrap(),
            BuildPose::default(),
        )))
        .unwrap()
    else {
        panic!("spawn expected")
    };
    let BuildOutcome::Spawned(block) = graph
        .apply(BuildCommand::Spawn(
            crate::CuboidSpec::new(
                [1; 3],
                BuildPose::new(bevy_math::IVec3::new(2, 1, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        panic!("spawn expected")
    };
    let solid = graph
        .evaluated_solid(crate::SolidOwner::Part(ring))
        .unwrap();
    let patch = solid
        .surfaces
        .iter()
        .find(|surface| surface.normal.abs_diff_eq(Vec3::Y, EPSILON))
        .unwrap()
        .key;
    let end = FaceRef::patch(ring, FaceKind::PositiveY, patch);
    let bottom = FaceRef::part(block, FaceKind::NegativeY);
    assert!(graph.weld_feature_on_faces(
        &[end],
        WeldFeature::Vertex(Vec3::new(0.4, 0.125, 0.0)),
        ConstructionFrame::IDENTITY
    ));
    assert!(!graph.weld_feature_on_faces(
        &[end],
        WeldFeature::Vertex(Vec3::new(0.0, 0.125, 0.0)),
        ConstructionFrame::IDENTITY
    ));
    graph.weld_contact_square(&[bottom], &[end]).unwrap();
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: bottom,
            second: end,
        }))
        .unwrap();
    assert_eq!(graph.compile().unwrap().compounds.len(), 1);
}
