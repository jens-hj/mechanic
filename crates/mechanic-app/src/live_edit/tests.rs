use super::*;
use mechanic_core::{
    BearingDimensions, BuildCommand, BuildOutcome, BuildPose, CuboidSpec, FaceKind, FaceRef,
    GridRotation, JointKind,
};
use mechanic_gpu::GpuTransform;

fn framed_graph() -> (ConstructionGraph, PartId, ConstructionFrame) {
    let mut graph = ConstructionGraph::new();
    let spec = CuboidSpec::new(
        [2, 2, 2],
        BuildPose::new(IVec3::new(4, 8, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
        panic!("part expected")
    };
    let frame =
        ConstructionFrame::new(Vec3::new(4.0, 2.0, -1.0), Quat::from_rotation_y(0.63)).unwrap();
    graph.reframe_parts([part], frame).unwrap();
    (graph, part, frame)
}

fn socket(part: PartId, frame: ConstructionFrame) -> PlacedBearing {
    transform_bearing(
        PlacedBearing {
            kind: JointKind::Rotational,
            axis: Vec3::X,
            source: FaceRef::part(part, FaceKind::PositiveX),
            anchor: Vec3::new(1.25, 2.0, 0.0),
            dimensions: BearingDimensions::default(),
        },
        frame,
    )
}

#[test]
fn rays_follow_translating_and_rotating_body_without_changing_local_target() {
    let (graph, part, authored) = framed_graph();
    let creation = graph.compile().unwrap();
    let initial = &creation.compounds[0];
    let local_origin = Vec3::new(1.0, 2.0, 4.0);
    for (translation, rotation) in [
        (Vec3::new(12.0, 4.0, -7.0), Quat::IDENTITY),
        (Vec3::new(-3.0, 8.0, 5.0), Quat::from_rotation_z(0.8)),
    ] {
        let motion = ConstructionFrame::new(translation, rotation).unwrap();
        let live_frame = motion.compose(authored);
        let simulation = AppSimulation {
            creation: Some(creation.clone()),
            transforms: vec![GpuTransform {
                position: motion
                    .point(initial.root_translation)
                    .extend(0.0)
                    .to_array(),
                rotation: (rotation * initial.root_rotation).to_array(),
            }],
            ..Default::default()
        };
        let context = EditContext::resolve(&graph, &simulation, part).unwrap();
        let ray = context.ray(Ray3d::new(
            live_frame.point(local_origin),
            Dir3::new(live_frame.vector(Vec3::NEG_Z)).unwrap(),
        ));
        assert_eq!(context.anchor, part);
        assert_eq!(context.frame, graph.part_frame_id(part).unwrap());
        assert!(ray.origin.abs_diff_eq(local_origin, 1.0e-5));
        assert!(ray.direction.as_vec3().abs_diff_eq(Vec3::NEG_Z, 1.0e-5));
    }
}

#[test]
fn no_op_editor_view_preserves_exact_graph_revision_and_socket_coordinates() {
    let (graph, part, frame) = framed_graph();
    let original = graph.clone();
    let original_socket = socket(part, frame);
    let mut target = EditorGraph(graph);
    let mut state = EditorState {
        edit_context: EditContext::resolve(&target.0, &AppSimulation::default(), part),
        placed_bearings: vec![original_socket],
        ..Default::default()
    };
    {
        let mut view = EditorView::new(&mut target, &mut state);
        let (local, state) = view.parts();
        assert!(
            local
                .0
                .part_position(part)
                .unwrap()
                .abs_diff_eq(Vec3::new(1.0, 2.0, 0.0), 1.0e-5)
        );
        assert!(
            state.placed_bearings[0]
                .anchor
                .abs_diff_eq(Vec3::new(1.25, 2.0, 0.0), 1.0e-5)
        );
    }
    assert!(target.0.shares_revision(&original));
    assert_eq!(state.placed_bearings, vec![original_socket]);
    assert_eq!(target.0.part_frame(part), Some(frame));
}

#[test]
fn local_edits_publish_canonical_positions_and_inherit_frame_and_source_identity() {
    let (graph, anchor, frame) = framed_graph();
    let frame_id = graph.part_frame_id(anchor).unwrap();
    let mut target = EditorGraph(graph);
    let mut state = EditorState {
        edit_context: EditContext::resolve(&target.0, &AppSimulation::default(), anchor),
        placed_bearings: vec![socket(anchor, frame)],
        ..Default::default()
    };
    let added;
    {
        let mut view = EditorView::new(&mut target, &mut state);
        let (local, state) = view.parts();
        let spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::new(IVec3::new(8, 8, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = local.0.apply(BuildCommand::Spawn(spec)).unwrap() else {
            panic!("part expected")
        };
        added = part;
        state.placed_bearings[0].anchor += Vec3::Y * 0.25;
    }
    assert_eq!(target.0.part_frame_id(added), Some(frame_id));
    assert_eq!(target.0.edit_source(added), Some(anchor));
    assert_eq!(target.0.view_to_build(), ConstructionFrame::IDENTITY);
    assert!(
        target
            .0
            .part_position(added)
            .unwrap()
            .abs_diff_eq(frame.point(Vec3::new(2.0, 2.0, 0.0)), 1.0e-5)
    );
    assert!(
        state.placed_bearings[0]
            .anchor
            .abs_diff_eq(frame.point(Vec3::new(1.25, 2.25, 0.0)), 1.0e-5)
    );
}

#[test]
fn removed_or_unpublished_anchor_cannot_resolve_against_live_physics() {
    let (mut graph, part, _) = framed_graph();
    let simulation = AppSimulation {
        creation: Some(graph.compile().unwrap()),
        ..Default::default()
    };
    graph.apply(BuildCommand::Remove(part)).unwrap();
    assert!(EditContext::resolve(&graph, &AppSimulation::default(), part).is_none());
    let spec = CuboidSpec::new([1, 1, 1], BuildPose::default()).unwrap();
    let BuildOutcome::Spawned(unpublished) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
        panic!("part expected")
    };
    assert!(EditContext::resolve(&graph, &simulation, unpublished).is_none());
}

#[test]
fn gesture_accepts_same_live_body_and_pending_parts_but_not_other_bodies() {
    let (mut graph, anchor, _) = framed_graph();
    let frame = graph.part_frame_id(anchor).unwrap();
    graph.set_edit_frame(frame).unwrap();
    let spec = CuboidSpec::new(
        [1, 1, 1],
        BuildPose::new(IVec3::new(12, 8, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(other_body) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
        panic!("part expected")
    };
    let creation = graph.compile().unwrap();
    let transforms = creation
        .compounds
        .iter()
        .map(|body| GpuTransform {
            position: body.root_translation.extend(0.0).to_array(),
            rotation: body.root_rotation.to_array(),
        })
        .collect();
    let simulation = AppSimulation {
        creation: Some(creation),
        transforms,
        ..Default::default()
    };
    let context = EditContext::resolve(&graph, &simulation, anchor).unwrap();
    assert!(context.accepts_part(&graph, &simulation, anchor));
    assert!(!context.accepts_part(&graph, &simulation, other_body));
    assert!(context.accepts_part(&graph, &AppSimulation::default(), other_body));
    let BuildOutcome::Spawned(pending) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
        panic!("part expected")
    };
    assert!(context.accepts_part(&graph, &simulation, pending));
    graph.apply(BuildCommand::Remove(pending)).unwrap();
    assert!(!context.accepts_part(&graph, &simulation, pending));
}

#[test]
fn no_op_resource_view_does_not_report_a_graph_change_but_edits_do() {
    #[derive(Resource, Default)]
    struct ObservedChanges(Vec<bool>);
    #[derive(Resource, Default)]
    struct ApplyEdit(bool);

    fn edit(mut graph: ResMut<EditorGraph>, mut state: ResMut<EditorState>, apply: Res<ApplyEdit>) {
        let mut view = EditorView::new(&mut graph, &mut state);
        if apply.0 {
            let (local, _) = view.parts();
            local
                .0
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new([1, 1, 1], BuildPose::default()).unwrap(),
                ))
                .unwrap();
        }
    }

    fn observe(graph: Res<EditorGraph>, mut changes: ResMut<ObservedChanges>) {
        changes.0.push(graph.is_changed());
    }

    let (graph, anchor, _) = framed_graph();
    let state = EditorState {
        edit_context: EditContext::resolve(&graph, &AppSimulation::default(), anchor),
        ..Default::default()
    };
    let mut app = App::new();
    app.insert_resource(EditorGraph(graph))
        .insert_resource(state)
        .init_resource::<ObservedChanges>()
        .init_resource::<ApplyEdit>()
        .add_systems(Update, (edit, observe).chain());
    app.update();
    app.update();
    app.world_mut().resource_mut::<ApplyEdit>().0 = true;
    app.update();
    assert_eq!(
        app.world().resource::<ObservedChanges>().0,
        vec![true, false, true]
    );
}

#[test]
fn refresh_preserves_removed_anchor_for_gesture_cancellation() {
    let (mut graph, anchor, _) = framed_graph();
    let context = EditContext::resolve(&graph, &AppSimulation::default(), anchor).unwrap();
    graph.apply(BuildCommand::Remove(anchor)).unwrap();
    let mut app = App::new();
    app.insert_resource(EditorGraph(graph))
        .insert_resource(AppSimulation::default())
        .insert_resource(EditorState {
            edit_context: Some(context),
            ..Default::default()
        })
        .add_systems(Update, refresh_context);
    app.update();
    let retained = app.world().resource::<EditorState>().edit_context.unwrap();
    assert_eq!(retained.anchor, anchor);
    assert_eq!(retained.frame, context.frame);
    assert_eq!(retained.frame_to_world, context.frame_to_world);
}
