use super::*;
use bevy::mesh::VertexAttributeValues;
use mechanic_core::{
    BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionFrame, CuboidSpec,
    EdgeChainRef, EdgeTreatment, FaceKind, FaceRef, GridRotation, ShapeFeature, ShapeRegion,
    WeldSpec,
};
use mechanic_gpu::GpuTransform;

fn spawn(graph: &mut ConstructionGraph, center: IVec3) -> PartId {
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [1; 3],
                BuildPose::from_half_grid(center, GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    part
}

fn positions(mesh: &Mesh) -> &[[f32; 3]] {
    let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        panic!("highlight positions")
    };
    positions
}

fn normals(mesh: &Mesh) -> &[[f32; 3]] {
    let Some(VertexAttributeValues::Float32x3(normals)) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
    else {
        panic!("highlight normals")
    };
    normals
}

#[test]
fn articulated_second_body_preview_follows_its_pose_in_the_first_bodys_gesture_frame() {
    let mut graph = ConstructionGraph::new();
    let first = spawn(&mut graph, IVec3::new(1, 1, 1));
    let second = spawn(&mut graph, IVec3::new(3, 1, 1));
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(first, FaceKind::PositiveX),
            FaceRef::part(second, FaceKind::NegativeX),
            Vec3::new(0.25, 0.125, 0.125),
            Vec3::X,
        )))
        .unwrap();
    let frame =
        ConstructionFrame::new(Vec3::new(2.0, 3.0, -1.0), Quat::from_rotation_z(0.6)).unwrap();
    graph.reframe_parts([first, second], frame).unwrap();
    let creation = graph.compile().unwrap();
    let second_body = creation
        .part_to_compound
        .iter()
        .find(|(part, _)| *part == second)
        .unwrap()
        .1 as usize;
    let mut transforms = creation
        .compounds
        .iter()
        .map(|body| GpuTransform {
            position: body.root_translation.extend(0.0).to_array(),
            rotation: body.root_rotation.to_array(),
        })
        .collect::<Vec<_>>();
    let second_center = Vec3::new(8.0, 5.0, 4.0);
    transforms[second_body] = GpuTransform {
        position: second_center.extend(0.0).to_array(),
        rotation: (Quat::from_rotation_y(1.1) * creation.compounds[second_body].root_rotation)
            .to_array(),
    };
    let simulation = AppSimulation {
        creation: Some(creation),
        published_graph: graph.clone(),
        transforms,
        ..Default::default()
    };
    let context = EditContext::resolve(&graph, &simulation, first).unwrap();
    let local = graph.in_edit_frame(context.frame).unwrap();
    let mesh = weld_preview_mesh(&local, &simulation, second, Some(context));
    let points = positions(&mesh);
    assert_eq!(points.len(), 24, "the other articulated body is excluded");
    let center = points
        .iter()
        .map(|point| Vec3::from_array(*point))
        .sum::<Vec3>()
        / 24.0;
    assert!(context.frame_to_world.point(center).distance(second_center) < 1.0e-5);
    let changed = AppSimulation {
        transforms: simulation
            .transforms
            .iter()
            .enumerate()
            .map(|(body, pose)| {
                let mut pose = *pose;
                if body == second_body {
                    pose.position[0] += 2.0;
                }
                pose
            })
            .collect(),
        ..simulation
    };
    let mesh = weld_preview_mesh(&local, &changed, second, Some(context));
    let center = positions(&mesh)
        .iter()
        .map(|point| Vec3::from_array(*point))
        .sum::<Vec3>()
        / 24.0;
    assert!(
        context
            .frame_to_world
            .point(center)
            .distance(second_center + Vec3::X * 2.0)
            < 1.0e-5
    );
}

#[test]
fn shaped_region_is_highlighted_once_without_applying_its_frame_twice() {
    let mut graph = ConstructionGraph::new();
    let first = spawn(&mut graph, IVec3::new(1, 1, 1));
    let second = spawn(&mut graph, IVec3::new(3, 1, 1));
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(first, FaceKind::PositiveX),
            second: FaceRef::part(second, FaceKind::NegativeX),
        }))
        .unwrap();
    let region = ShapeRegion::new(
        IVec3::ZERO,
        IVec3::new(2, 1, 1),
        mechanic_core::ConstructionMaterial::Steel,
    )
    .unwrap();
    let BuildOutcome::RegionAdded(region) = graph.apply(BuildCommand::AddRegion(region)).unwrap()
    else {
        unreachable!()
    };
    let owner = SolidOwner::Region(region);
    let edge = graph.evaluated_solid(owner).unwrap().logical_edges[0].key;
    graph
        .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
            [EdgeChainRef { owner, edge }],
            EdgeTreatment::Fillet,
            20,
        )))
        .unwrap();
    let original = weld_preview_mesh(&graph, &AppSimulation::default(), first, None);
    let frame =
        ConstructionFrame::new(Vec3::new(4.0, 3.0, 2.0), Quat::from_rotation_y(0.8)).unwrap();
    graph.reframe_parts([first, second], frame).unwrap();
    let context = EditContext {
        anchor: first,
        frame: graph.part_frame_id(first).unwrap(),
        frame_to_world: frame,
    };
    let local = graph.in_edit_frame(context.frame).unwrap();
    let framed = weld_preview_mesh(&local, &AppSimulation::default(), first, Some(context));
    assert_eq!(positions(&original).len(), positions(&framed).len());
    for (original, framed) in positions(&original).iter().zip(positions(&framed)) {
        assert!(Vec3::from_array(*original).distance(Vec3::from_array(*framed)) < 1.0e-5);
    }
    for (original, framed) in normals(&original).iter().zip(normals(&framed)) {
        assert!(Vec3::from_array(*original).distance(Vec3::from_array(*framed)) < 1.0e-5);
    }
    let solid = graph.evaluated_solid(owner).unwrap();
    let mut expected_positions = Vec::new();
    crate::render::mesh::construction::append_evaluated_solid(
        &solid,
        BuildTransform::IDENTITY,
        &mut expected_positions,
        &mut Vec::new(),
        &mut Vec::new(),
        &mut Vec::new(),
        &mut Vec::new(),
    );
    assert_eq!(
        positions(&framed).len(),
        expected_positions.len(),
        "one evaluated region rather than one copy per member"
    );
}

#[test]
fn delete_preview_highlights_only_selected_framed_parts_of_a_welded_body() {
    let mut graph = ConstructionGraph::new();
    let first = spawn(&mut graph, IVec3::ONE);
    let second = spawn(&mut graph, IVec3::ONE);
    graph
        .reframe_parts(
            [second],
            ConstructionFrame::new(Vec3::X * 10.0, Quat::IDENTITY).unwrap(),
        )
        .unwrap();
    graph
        .apply(BuildCommand::RigidLink(mechanic_core::RigidLinkSpec {
            first,
            second,
        }))
        .unwrap();
    let mesh = parts_preview_mesh(&graph, &AppSimulation::default(), &[second], None, 1.018);
    assert_eq!(positions(&mesh).len(), 24);
    assert!(
        positions(&mesh)
            .iter()
            .all(|point| point[0] > 9.9 && point[0] < 10.4)
    );
}
