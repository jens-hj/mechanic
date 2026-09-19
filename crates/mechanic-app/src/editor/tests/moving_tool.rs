//! Tool transactions on authored grids carried by moving physical bodies.

use crate::editor::build_actions::handle_chroma_actions;
use crate::editor::build_actions::stage_part_deletion_preserving_bearings;
use crate::*;
use mechanic_core::{
    BuildPose, CarriageFace, ConstructionFrame, FaceKind, LinearBearing, LinearBearingDimensions,
};
use mechanic_gpu::GpuTransform;

fn spawn(graph: &mut ConstructionGraph, x: i32) -> PartId {
    let spec = CuboidSpec::new(
        [2; 3],
        BuildPose::new(IVec3::new(x, 8, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
        panic!("part expected")
    };
    part
}

fn moving_fixture(
    mut graph: ConstructionGraph,
    anchor: PartId,
) -> (EditorGraph, EditorState, AppSimulation, ConstructionFrame) {
    let frame =
        ConstructionFrame::new(Vec3::new(4.0, 1.0, -3.0), Quat::from_rotation_y(0.63)).unwrap();
    let parts = graph.parts().map(|(id, _)| id).collect::<Vec<_>>();
    graph.reframe_parts(parts, frame).unwrap();
    let creation = graph.compile().unwrap();
    let motion =
        ConstructionFrame::new(Vec3::new(-7.0, 8.0, 5.0), Quat::from_rotation_z(0.47)).unwrap();
    let transforms = creation
        .compounds
        .iter()
        .map(|body| GpuTransform {
            position: motion.point(body.root_translation).extend(0.0).to_array(),
            rotation: (motion.rotation() * body.root_rotation).to_array(),
        })
        .collect();
    let simulation = AppSimulation {
        creation: Some(creation),
        transforms,
        published_graph: graph.clone(),
        ..Default::default()
    };
    let state = EditorState {
        edit_context: live_edit::EditContext::resolve(&graph, &simulation, anchor),
        ..Default::default()
    };
    assert!(
        !state
            .edit_context
            .unwrap()
            .frame_to_world
            .translation()
            .abs_diff_eq(frame.translation(), 0.1)
    );
    (EditorGraph(graph), state, simulation, frame)
}

fn single_fixture() -> (
    EditorGraph,
    EditorState,
    AppSimulation,
    ConstructionFrame,
    PartId,
) {
    let mut graph = ConstructionGraph::new();
    let anchor = spawn(&mut graph, 0);
    let (graph, state, simulation, frame) = moving_fixture(graph, anchor);
    (graph, state, simulation, frame, anchor)
}

fn top_hit(anchor: PartId) -> SurfaceHit {
    SurfaceHit {
        distance: 1.0,
        point: Vec3::new(0.0, 2.25, 0.0),
        face: FaceRef::part(anchor, FaceKind::PositiveY),
    }
}

#[test]
fn moving_surface_placement_uses_real_builder_and_inherits_pose_source() {
    let (mut graph, mut state, _, frame, anchor) = single_fixture();
    let local_center;
    let added;
    {
        let mut view = live_edit::EditorView::new(&mut graph, &mut state);
        let (graph, _) = view.parts();
        let candidate = builder::candidates::candidate_from_hit(&graph.0, top_hit(anchor));
        local_center = candidate.spec.pose.translation();
        graph.0 = builder::stage_cuboid(&graph.0, candidate).unwrap();
        added = graph
            .0
            .parts()
            .find_map(|(part, _)| (part != anchor).then_some(part))
            .unwrap();
    }
    assert_eq!(graph.0.part_frame_id(added), graph.0.part_frame_id(anchor));
    assert_eq!(graph.0.edit_source(added), Some(anchor));
    assert!(
        graph
            .0
            .part_position(added)
            .unwrap()
            .abs_diff_eq(frame.point(local_center), 1.0e-5)
    );
    assert_eq!(graph.0.compile().unwrap().compounds.len(), 1);
}

#[test]
fn removing_middle_block_splits_moving_creation_and_history_restores_canonical_graph() {
    let mut construction = ConstructionGraph::new();
    let first = spawn(&mut construction, 0);
    let middle = spawn(&mut construction, 2);
    let last = spawn(&mut construction, 4);
    for (a, b) in [(first, middle), (middle, last)] {
        construction
            .apply(BuildCommand::Weld(mechanic_core::WeldSpec {
                first: FaceRef::part(a, FaceKind::PositiveX),
                second: FaceRef::part(b, FaceKind::NegativeX),
            }))
            .unwrap();
    }
    let (mut graph, mut state, _, _) = moving_fixture(construction, first);
    let before = graph.0.clone();
    let mut history = EditorHistory::default();
    {
        let mut view = live_edit::EditorView::new(&mut graph, &mut state);
        let (graph, state) = view.parts();
        let previous = EditorSnapshot::capture(&graph.0, state);
        let (next, bearings, _) =
            stage_part_deletion_preserving_bearings(&graph.0, &state.placed_bearings, &[middle])
                .unwrap();
        graph.0 = next;
        state.placed_bearings = bearings;
        history.commit(previous);
    }
    assert_eq!(graph.0.compile().unwrap().compounds.len(), 2);
    let deleted = EditorSnapshot::capture(&graph.0, &state);
    let restored = history.undo(deleted).unwrap();
    assert_eq!(restored.graph.view_to_build(), ConstructionFrame::IDENTITY);
    assert_eq!(restored.graph.compile().unwrap().compounds.len(), 1);
    for part in [first, middle, last] {
        assert!(
            restored
                .graph
                .part_position(part)
                .unwrap()
                .abs_diff_eq(before.part_position(part).unwrap(), 1.0e-5)
        );
    }
    let redone = history.redo(restored).unwrap();
    assert!(redone.graph.part(middle).is_none());
    assert_eq!(redone.graph.view_to_build(), ConstructionFrame::IDENTITY);
    assert_eq!(redone.graph.compile().unwrap().compounds.len(), 2);
}

#[test]
fn chroma_stroke_on_moving_body_changes_only_appearance_and_records_one_history_entry() {
    let (mut graph, mut state, _, _, anchor) = single_fixture();
    let before_position = graph.0.part_position(anchor).unwrap();
    let brush = MaterialAppearance {
        color: mechanic_core::MaterialColor::Baked,
        finish: mechanic_core::MaterialFinish::Painted,
    };
    let mut actions = ButtonInput::<GameAction>::default();
    actions.press(GameAction::Primary);
    let mut history = EditorHistory::default();
    {
        let mut view = live_edit::EditorView::new(&mut graph, &mut state);
        let (graph, state) = view.parts();
        state.hovered = Some(top_hit(anchor));
        handle_chroma_actions(&actions, &mut graph.0, state, &mut history, brush);
        actions.clear();
        actions.release(GameAction::Primary);
        handle_chroma_actions(&actions, &mut graph.0, state, &mut history, brush);
    }
    assert_eq!(graph.0.part(anchor).unwrap().appearance(), Some(brush));
    assert!(
        graph
            .0
            .part_position(anchor)
            .unwrap()
            .abs_diff_eq(before_position, 1.0e-5)
    );
    assert_eq!(history.undo.len(), 1);
    let prior = history
        .undo(EditorSnapshot::capture(&graph.0, &state))
        .unwrap();
    assert_eq!(
        prior.graph.part(anchor).unwrap().appearance(),
        Some(MaterialAppearance::BAKED)
    );
}

#[test]
fn rotational_and_linear_attachments_and_controller_wiring_return_to_build_space() {
    for linear in [false, true] {
        let (mut graph, mut state, _, frame, anchor_part) = single_fixture();
        let source = top_hit(anchor_part).face;
        let anchor = top_hit(anchor_part).point;
        let bearing;
        {
            let mut view = live_edit::EditorView::new(&mut graph, &mut state);
            let (graph, _) = view.parts();
            if linear {
                let rail = LinearBearing {
                    dimensions: LinearBearingDimensions::default(),
                    mount_normal: Vec3::Y,
                    face: CarriageFace::Top,
                };
                let surface =
                    builder::bearings::linear_carriage_face(anchor, rail, Vec3::X).unwrap();
                let candidate = builder::linear_block_candidate(
                    anchor,
                    rail,
                    Vec3::X,
                    surface.center,
                    [1; 3],
                    GridRotation::default(),
                )
                .unwrap();
                graph.0 = builder::stage_linear_block_batch_in_bounds(
                    &graph.0,
                    candidate,
                    &[candidate.spec],
                    builder::LinearAttachment {
                        source,
                        anchor,
                        rail,
                        axis: Vec3::X,
                        rigid_targets: &[],
                    },
                    builder::PlacementBounds::Garage,
                )
                .unwrap();
            } else {
                let candidate = builder::bearing_attachment_candidate(&graph.0, source, anchor);
                graph.0 = builder::stage_bearing_attachment(
                    &graph.0,
                    candidate,
                    source,
                    anchor,
                    BearingDimensions::default(),
                )
                .unwrap();
            }
            bearing = graph.0.bearings().next().unwrap().0;
            let BuildOutcome::Spawned(controller) = graph
                .0
                .apply(BuildCommand::SpawnController(ControllerSpec::new(
                    BuildPose::new(IVec3::new(-8, 8, 0), GridRotation::default()),
                )))
                .unwrap()
            else {
                panic!("controller expected")
            };
            let link = if linear {
                mechanic_core::DriveLinkSpec::new_linear(
                    controller,
                    bearing,
                    LinearBearingDimensions::default().bounds(),
                )
            } else {
                mechanic_core::DriveLinkSpec::new(controller, bearing)
            };
            graph.0.apply(BuildCommand::AddDriveLink(link)).unwrap();
        }
        let joint = graph.0.bearing(bearing).unwrap();
        assert!(joint.shared_anchor.abs_diff_eq(frame.point(anchor), 1.0e-5));
        assert!(
            joint
                .axis
                .abs_diff_eq(frame.vector(if linear { Vec3::X } else { Vec3::Y }), 1.0e-5)
        );
        assert_eq!(graph.0.drive_links().count(), 1);
        let compiled = graph.0.compile().unwrap();
        assert_eq!(compiled.bearings.len(), 1);
        assert_eq!(
            matches!(
                compiled.bearings[0].kind,
                mechanic_core::JointKind::Linear(_)
            ),
            linear
        );
    }
}

#[test]
fn drag_plane_refresh_keeps_start_grid_as_body_moves_and_turns() {
    let (graph, state, mut simulation, _, anchor) = single_fixture();
    let start_context = state.edit_context.unwrap();
    let start = CuboidSpec::new(
        [1; 3],
        BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
    )
    .unwrap();
    let expected = Vec3::new(0.5, 2.0, 0.75);
    for turn in [0.0, 0.8] {
        let body = &simulation.creation.as_ref().unwrap().compounds[0];
        let movement = ConstructionFrame::new(
            Vec3::new(10.0, 6.0, turn * 3.0),
            Quat::from_rotation_x(turn),
        )
        .unwrap();
        simulation.transforms[0] = GpuTransform {
            position: movement.point(body.root_translation).extend(0.0).to_array(),
            rotation: (movement.rotation() * body.root_rotation).to_array(),
        };
        let refreshed =
            live_edit::EditContext::resolve(&graph.0, &simulation, start_context.anchor).unwrap();
        assert_eq!(refreshed.frame, start_context.frame);
        assert_eq!(refreshed.anchor, anchor);
        let ray = refreshed.ray(Ray3d::new(
            refreshed.frame_to_world.point(expected + Vec3::Y * 3.0),
            Dir3::new(refreshed.frame_to_world.vector(Vec3::NEG_Y)).unwrap(),
        ));
        let end = builder::raycast_placement_plane_point(
            ray.origin,
            ray.direction.as_vec3(),
            start,
            builder::PlacementPlane::Xz,
        )
        .unwrap();
        assert!(end.abs_diff_eq(expected, 1.0e-5));
    }
}

#[test]
fn moving_region_vertex_nudge_keeps_region_membership_and_local_cage_edits() {
    let (mut graph, mut state, _, frame, anchor) = single_fixture();
    let region;
    {
        let mut view = live_edit::EditorView::new(&mut graph, &mut state);
        let (graph, _) = view.parts();
        let cuboid = graph.0.part(anchor).unwrap().as_cuboid().unwrap();
        let cells = mechanic_core::part_cells(cuboid);
        let shape = mechanic_core::ShapeRegion::from_origin_steps(
            cells.corner_steps(IVec3::ZERO, 0),
            cells.counts(),
            cuboid.material,
        )
        .unwrap();
        let BuildOutcome::RegionAdded(id) = graph.0.apply(BuildCommand::AddRegion(shape)).unwrap()
        else {
            panic!("region expected")
        };
        region = id;
        let edits = shape_tool::nudge_edits(
            graph.0.region(region).unwrap(),
            &[[0, 0, 0]],
            0,
            1,
            shape_tool::ShapeSnap { steps: 5 },
            shape_tool::ShapeMirror::default(),
        );
        assert!(!edits.is_empty());
        graph
            .0
            .apply(BuildCommand::SetRegionVertices {
                region,
                vertices: edits,
            })
            .unwrap();
    }
    assert_eq!(graph.0.region_of(anchor), Some(region));
    assert_eq!(
        graph.0.region_frame_id(region),
        graph.0.part_frame_id(anchor)
    );
    assert_eq!(graph.0.region(region).unwrap().offset([0, 0, 0]), [5, 0, 0]);
    let actual = graph.0.region_frame(region).unwrap();
    assert!(
        actual
            .translation()
            .abs_diff_eq(frame.translation(), 1.0e-5)
    );
    assert!(actual.rotation().abs_diff_eq(frame.rotation(), 1.0e-5));
    assert!(!graph.0.compile().unwrap().colliders.is_empty());
}

#[test]
fn moving_part_feature_keeps_topology_and_reduces_volume_after_canonical_publication() {
    let (mut graph, mut state, _, _, part) = single_fixture();
    let owner = mechanic_core::SolidOwner::Part(part);
    let before = graph.0.evaluated_solid(owner).unwrap().volume();
    let feature;
    {
        let mut view = live_edit::EditorView::new(&mut graph, &mut state);
        let (graph, _) = view.parts();
        let edge = graph.0.evaluated_solid(owner).unwrap().logical_edges[0].key;
        let command = mechanic_core::ShapeFeature::new(
            [mechanic_core::EdgeChainRef { owner, edge }],
            mechanic_core::EdgeTreatment::Chamfer,
            10,
        );
        let BuildOutcome::ShapeFeatureAdded(id) = graph
            .0
            .apply(BuildCommand::AddShapeFeature(command))
            .unwrap()
        else {
            panic!("feature expected")
        };
        feature = id;
    }
    assert!(graph.0.evaluated_solid(owner).unwrap().volume() < before);
    assert!(graph.0.shape_features().any(|(id, _)| id == feature));
    assert_eq!(graph.0.view_to_build(), ConstructionFrame::IDENTITY);
    assert!(
        graph
            .0
            .compile()
            .unwrap()
            .colliders
            .iter()
            .any(|c| matches!(c.shape, mechanic_core::ColliderShape::Convex(_)))
    );
}
