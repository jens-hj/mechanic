mod moving_tool;

use bevy::prelude::{App, ButtonInput, IVec3, KeyCode, Quat, State, Update, Vec2, Vec3};
use mechanic_core::{
    BearingDimensions, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
    ConstructionMaterial, ControllerSpec, CuboidSpec, CylinderDimensions, CylinderSpec,
    DimensionLinkId, DimensionLinkSpec, DriveLinkSpec, EdgeChainRef, EdgeTreatment, FaceKind,
    FaceOwner, FaceRef, GridRotation, MaterialAppearance, MaterialColor, MaterialDye,
    MaterialFinish, POSITION_TICK_METERS, PartId, PartSpec, PendingOperation, RigidLinkSpec,
    ShapeFeature, ShapeRegion, SolidOwner, WeldSpec,
};
use mechanic_gpu::GpuTransform;

use crate::builder::candidates::candidate_from_hit;
use crate::builder::{
    BlockVolume, PlacementGrid, PlacementPlane, SurfaceHit, bearing_attachment_candidate,
    raycast_construction,
};
use crate::builder::{SmartGuide, block_sheet_specs};
use crate::camera::{MaterialWheelState, PlayerState};
use crate::controls::GameAction;
use crate::editor::build_actions::{
    PlacedBearing, handle_block_actions, handle_build_actions, handle_chroma_actions,
    stage_part_deletion_preserving_bearings,
};
use crate::editor::dimensions::{
    BearingDimensionTarget, BearingToolSettings, CylinderDimensionTarget, CylinderToolSettings,
    adjusted_bearing_dimensions, adjusted_cylinder_dimensions,
    requested_bearing_dimension_adjustment, requested_cylinder_dimension_adjustment,
};
use crate::editor::hammer::{
    HAMMER_CHARGE_SECONDS, HAMMER_MAX_IMPULSE, HAMMER_MIN_IMPULSE, hammer_delivery,
    hammer_impulse_magnitude, hammer_point_travel,
};
use crate::editor::history::{EditorHistory, HistoryAction, apply_history_action};
use crate::editor::hover::{
    BearingOffsetDrag, BlockAttachment, BlockDrag, PointerSample, bearing_offset_from_rays,
    clear_editor_hover, delete_box_parts, handle_tool_change, refresh_bearing_offset_drag,
    refresh_block_drag, refresh_tool_preview,
};
use crate::editor::pipe::{
    PipeDrag, PipeEditMode, closest_axis_parameter, constrained_pipe_bend_span, pipe_corner_inset,
    pipe_pointer_delta, pipe_turn_direction, rebase_pipe_path,
};
use crate::editor::preview::{bearing_attachment_is_highlighted, tool_status_line};
use crate::editor::raycast::{
    SimulationHit, raycast_placed_bearings, raycast_rotational_bearings, raycast_simulation,
};
use crate::editor::shape_actions::{RegionDrag, commit_region_drag, region_area};
use crate::editor::shape_actions::{
    active_drag_plane, choose_region, closer_feature_hit, handle_feature_shape_actions,
    refresh_region_drag, tangent_feature_chain, weld_connected_shape_owners,
};
use crate::editor::shortcuts::cycle_orientation;
use crate::editor::state::{EditorGraph, EditorState};
use crate::editor::wiring::{WireConnection, WireDrag, WireDragStep, WireEnd};
use crate::editor::wiring::{
    connect_control_link, connect_drive_wire, wire_drag_step, wire_end_under_cursor,
};
use crate::hotbar::{SelectedTool, Tool};
use crate::pose::simulation_placed_bearing_pose;
use crate::render::authored::{AUTHORED_ORIENTATION_COUNT, AUTHORED_ORIENTATIONS};
use crate::render::mesh::preview::block_sheet_bounds;
use crate::simulation::state::AppSimulation;

fn pointer_sample(cursor: Vec2, ray_origin: Vec3, ray_direction: Vec3) -> PointerSample {
    PointerSample {
        cursor,
        ray_origin,
        ray_direction,
    }
}

#[test]
fn blocked_delete_release_clears_the_target_that_would_block_tab() {
    let mut state = EditorState {
        delete_target: Some(crate::editor::hover::DeleteTarget::PlacedBearing(0)),
        ..Default::default()
    };

    assert!(state.world_drag_active());
    assert!(state.cancel_delete_gesture());
    assert!(!state.world_drag_active());
}

#[test]
fn right_click_deletes_a_dimension_link_from_a_moving_construction() {
    assert_right_click_deletes_dimension_link(SelectedTool::from_editor_tool(Tool::DimensionLink));
}

#[test]
fn empty_hands_can_delete_a_dimension_link_from_a_moving_construction() {
    assert_right_click_deletes_dimension_link(SelectedTool {
        tool: None,
        ..Default::default()
    });
}

fn assert_right_click_deletes_dimension_link(selection: SelectedTool) {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(link) = graph
        .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
            DimensionLinkId(7),
            BuildPose::default(),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let state = EditorState {
        hovered_simulation: Some(SimulationHit {
            part: link,
            body_index: 0,
            distance: 1.0,
            point: Vec3::ZERO,
            normal: Vec3::Y,
        }),
        ..Default::default()
    };
    let mut actions = ButtonInput::default();
    actions.press(GameAction::Secondary);
    let mut app = App::new();
    app.insert_resource(actions)
        .insert_resource(ButtonInput::<KeyCode>::default())
        .insert_resource(EditorGraph(graph))
        .insert_resource(state)
        .insert_resource(EditorHistory::default())
        .insert_resource(crate::chroma::ChromaBrush::default())
        .insert_resource(AppSimulation::default())
        .insert_resource(selection)
        .insert_resource(BearingToolSettings::default())
        .insert_resource(CylinderToolSettings::default())
        .insert_resource(crate::ui::UiInput::default())
        .insert_resource(MaterialWheelState::default())
        .insert_resource(PlayerState {
            input_captured: true,
            ..Default::default()
        })
        .init_resource::<bevy::input::mouse::AccumulatedMouseMotion>()
        .add_systems(Update, handle_build_actions);

    app.update();
    assert!(
        app.world()
            .resource::<EditorState>()
            .delete_target
            .is_some()
    );
    {
        let mut actions = app.world_mut().resource_mut::<ButtonInput<GameAction>>();
        actions.clear();
        actions.release(GameAction::Secondary);
    }
    app.update();

    assert!(app.world().resource::<EditorGraph>().0.part(link).is_none());
    assert_eq!(
        app.world().resource::<EditorState>().feedback.as_deref(),
        Some("Deleted Dimension Link and incident connections")
    );
}

#[test]
fn connector_can_begin_wiring_a_moving_bearing() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [4, 4, 4],
                BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let bearing = PlacedBearing {
        kind: mechanic_core::JointKind::Rotational,
        axis: Vec3::ZERO,
        source: FaceRef::part(part, FaceKind::PositiveY),
        anchor: Vec3::Y,
        dimensions: BearingDimensions::default(),
    };
    let state = EditorState {
        hovered_bearing: Some(0),
        placed_bearings: vec![bearing],
        ..Default::default()
    };
    let mut actions = ButtonInput::default();
    actions.press(GameAction::Primary);
    let mut app = App::new();
    app.insert_resource(actions)
        .insert_resource(EditorGraph(graph))
        .insert_resource(state)
        .insert_resource(EditorHistory::default())
        .insert_resource(crate::chroma::ChromaBrush::default())
        .insert_resource(AppSimulation::default())
        .insert_resource(SelectedTool::from_editor_tool(Tool::Connector))
        .insert_resource(BearingToolSettings::default())
        .insert_resource(CylinderToolSettings::default())
        .insert_resource(crate::ui::UiInput::default())
        .insert_resource(MaterialWheelState::default())
        .insert_resource(PlayerState {
            input_captured: true,
            ..Default::default()
        })
        .init_resource::<bevy::input::mouse::AccumulatedMouseMotion>()
        .add_systems(Update, handle_build_actions);

    app.update();

    assert_eq!(
        app.world().resource::<EditorState>().wire_drag,
        Some(WireDrag {
            from: WireEnd::Bearing(0),
            armed: false,
        })
    );
}

#[test]
fn connector_recognizes_a_moving_control_block() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::default(),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let state = EditorState {
        hovered_simulation: Some(SimulationHit {
            part: controller,
            body_index: 0,
            distance: 1.0,
            point: Vec3::ZERO,
            normal: Vec3::Y,
        }),
        ..Default::default()
    };

    assert_eq!(
        wire_end_under_cursor(&graph, &state),
        Some(WireEnd::Controller(controller))
    );
}

#[test]
fn feature_drag_reuses_validated_geometry_only_for_matching_inputs() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([2; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let owner = SolidOwner::Part(part);
    let target = EdgeChainRef {
        owner,
        edge: graph.evaluated_solid(owner).unwrap().logical_edges[0].key,
    };
    let mut drag = crate::shape_tool::FeatureDrag::begin(
        crate::shape_tool::FeatureEdgeHit {
            target,
            point: Vec3::ZERO,
            tangent: Vec3::Z,
            bisector: Vec3::X,
            distance: 0.0,
        },
        vec![target],
        EdgeTreatment::Fillet,
        None,
        0,
        Vec3::Y,
        Vec3::NEG_Y,
    );
    drag.amount_ticks =
        crate::editor::shape_actions::clamp_feature_amount(&graph, &mut drag, 10, 1);
    assert_eq!(drag.amount_ticks, 10);
    let preview = crate::editor::preview::feature_drag_preview_graph(&graph, &drag).unwrap();
    assert!(preview.shares_revision(&drag.validated_preview.as_ref().unwrap().graph));
    assert_eq!(
        graph.shape_features().count(),
        0,
        "a preview never commits the feature"
    );
    drag.amount_ticks = 20;
    let adjusted = crate::editor::preview::feature_drag_preview_graph(&graph, &drag).unwrap();
    assert!(
        adjusted.evaluated_solid(owner).unwrap().volume()
            < preview.evaluated_solid(owner).unwrap().volume()
    );
    drag.amount_ticks = 10;
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap();
    let revised = crate::editor::preview::feature_drag_preview_graph(&graph, &drag).unwrap();
    assert_eq!(
        revised.part_count(),
        2,
        "a cached preview cannot hide a later placement"
    );
    assert_eq!(
        revised.evaluated_solid(owner).unwrap(),
        preview.evaluated_solid(owner).unwrap()
    );
}

#[test]
fn committing_a_feature_clears_its_edge_selection() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let owner = SolidOwner::Part(part);
    let target = EdgeChainRef {
        owner,
        edge: graph.evaluated_solid(owner).unwrap().logical_edges[0].key,
    };
    let hit = crate::shape_tool::FeatureEdgeHit {
        target,
        point: Vec3::ZERO,
        tangent: Vec3::Z,
        bisector: Vec3::X,
        distance: 0.0,
    };
    let ray_origin = Vec3::Y;
    let ray_direction = Vec3::NEG_Y;
    let mut state = EditorState {
        pointer_ray: Some((ray_origin, ray_direction)),
        feature_focus: Some(owner),
        selected_feature_edges: vec![target],
        feature_drag: Some(crate::shape_tool::FeatureDrag::begin(
            hit,
            vec![target],
            EdgeTreatment::Fillet,
            None,
            20,
            ray_origin,
            ray_direction,
        )),
        ..Default::default()
    };
    let mut actions = ButtonInput::default();
    actions.press(GameAction::Primary);
    actions.clear();
    actions.release(GameAction::Primary);
    let keys = ButtonInput::<KeyCode>::default();
    let mut history = EditorHistory::default();
    let player = PlayerState {
        input_captured: true,
        ..Default::default()
    };

    handle_feature_shape_actions(
        &actions,
        &keys,
        &mut graph,
        &mut state,
        &mut history,
        crate::shape_tool::ShapeSnap::feature_default(),
        crate::shape_tool::ShapeEditMode::Fillet,
        crate::ui::UiInput::default(),
        &player,
        &MaterialWheelState::default(),
    );

    assert_eq!(graph.shape_features().count(), 1);
    assert!(state.selected_feature_edges.is_empty());
    assert_eq!(state.selected_shape_feature, None);
    assert_eq!(history.undo.len(), 1);
}

#[test]
fn every_pipe_run_cylinder_rim_accepts_a_fillet() {
    let pieces = crate::builder::pipe_run_pieces(
        &[
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.875, 1.0, 0.0),
            Vec3::new(0.875, 1.0, 0.875),
        ],
        &[crate::builder::PipeNode::Bend { span: 1 }],
        CylinderDimensions::new(0.25, 0.0, 1.0).unwrap(),
        ConstructionMaterial::Wood,
    )
    .unwrap();
    let graph = crate::builder::stage_pipe_run(
        &ConstructionGraph::new(),
        &pieces,
        crate::builder::PipeRunAttachment::AutoWeld {
            source: FaceOwner::Ground,
        },
    )
    .unwrap();

    let mut rims = 0;
    for (part, spec) in graph.parts() {
        let PartSpec::Cylinder(_) = spec else {
            continue;
        };
        let owner = SolidOwner::Part(part);
        let solid = graph.evaluated_solid(owner).unwrap();
        for logical in solid
            .logical_edges
            .iter()
            .filter(|edge| edge.closed && edge.convex)
        {
            rims += 1;
            graph
                .clone()
                .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
                    [EdgeChainRef {
                        owner,
                        edge: logical.key,
                    }],
                    EdgeTreatment::Fillet,
                    20,
                )))
                .unwrap_or_else(|error| panic!("fillet rejected on {part:?}: {error}"));
        }
    }
    assert_eq!(rims, 4, "both pipe cylinders offer two rims");
}

#[test]
fn closer_generated_edge_wins_over_a_dashed_feature_source() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([2; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let owner = SolidOwner::Part(part);
    let target = EdgeChainRef {
        owner,
        edge: graph.evaluated_solid(owner).unwrap().logical_edges[0].key,
    };
    let BuildOutcome::ShapeFeatureAdded(feature) = graph
        .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
            [target],
            EdgeTreatment::Chamfer,
            10,
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let hit = |distance| crate::shape_tool::FeatureEdgeHit {
        target,
        point: Vec3::ZERO,
        tangent: Vec3::X,
        bisector: Vec3::Y,
        distance,
    };

    assert_eq!(
        closer_feature_hit(Some(hit(0.01)), Some((feature, hit(0.03)))),
        (None, Some(hit(0.01)))
    );
    assert_eq!(
        closer_feature_hit(Some(hit(0.04)), Some((feature, hit(0.02)))),
        (Some(feature), Some(hit(0.02)))
    );
    assert_eq!(
        closer_feature_hit(Some(hit(0.02)), Some((feature, hit(0.02)))),
        (None, Some(hit(0.02))),
        "an equal-distance real edge must not enter source adjustment mode"
    );
}

#[test]
fn tangent_edge_selection_traverses_one_weld_component() {
    let mut graph = ConstructionGraph::new();
    let mut parts = Vec::new();
    for x in 0..4 {
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1; 3],
                    BuildPose::new(IVec3::new(x, 0, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        parts.push(part);
    }
    for pair in parts[..3].windows(2) {
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(pair[0], FaceKind::PositiveX),
                second: FaceRef::part(pair[1], FaceKind::NegativeX),
            }))
            .unwrap();
    }

    let target_for = |part| {
        let owner = SolidOwner::Part(part);
        let solid = graph.evaluated_solid(owner).unwrap();
        let edge = solid
            .logical_edges
            .iter()
            .find(|logical| {
                let half_edge = solid.half_edges[logical.half_edges[0] as usize];
                let twin = solid.half_edges[half_edge.twin as usize];
                let patches = [
                    solid.surfaces[half_edge.face as usize].key.local,
                    solid.surfaces[twin.face as usize].key.local,
                ];
                patches.contains(&3) && patches.contains(&4)
            })
            .expect("the positive-Y/negative-Z edge exists")
            .key;
        EdgeChainRef { owner, edge }
    };
    let initial = target_for(parts[0]);

    let connected = weld_connected_shape_owners(&graph, initial.owner);
    assert_eq!(connected.len(), 3);
    assert!(!connected.contains(&SolidOwner::Part(parts[3])));
    assert_eq!(
        tangent_feature_chain(&graph, initial),
        parts[..3]
            .iter()
            .copied()
            .map(target_for)
            .collect::<Vec<_>>()
    );
}

#[test]
fn chroma_drag_paints_each_crossed_part_as_one_undoable_stroke() {
    let mut graph = ConstructionGraph::new();
    let mut ids = Vec::new();
    for x in [0, 4] {
        let BuildOutcome::Spawned(id) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::new(IVec3::new(x, 0, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        ids.push(id);
    }
    let paint = MaterialAppearance::new(
        MaterialColor::Dye(MaterialDye::new([42, 76, 199], 1.0).unwrap()),
        MaterialFinish::Painted,
    );
    let hit = |part| SurfaceHit {
        distance: 1.0,
        point: Vec3::ZERO,
        face: FaceRef::part(part, FaceKind::PositiveY),
    };
    let mut state = EditorState {
        hovered: Some(hit(ids[0])),
        ..Default::default()
    };
    let mut history = EditorHistory::default();
    let mut mouse = ButtonInput::default();

    mouse.press(GameAction::Primary);
    handle_chroma_actions(&mouse, &mut graph, &mut state, &mut history, paint);
    mouse.clear();
    state.hovered = Some(hit(ids[1]));
    handle_chroma_actions(&mouse, &mut graph, &mut state, &mut history, paint);
    mouse.clear();
    mouse.release(GameAction::Primary);
    handle_chroma_actions(&mouse, &mut graph, &mut state, &mut history, paint);

    assert_eq!(history.undo.len(), 1);
    assert!(
        ids.into_iter()
            .all(|id| graph.part(id).unwrap().appearance() == Some(paint))
    );
    assert!(apply_history_action(
        HistoryAction::Undo,
        &mut graph,
        &mut state,
        &mut history,
    ));
    assert!(
        graph
            .parts()
            .all(|(_, part)| part.appearance() == Some(MaterialAppearance::BAKED))
    );
}

#[test]
fn chroma_remove_restores_baked_once_and_a_noop_stroke_adds_no_history() {
    let paint = MaterialAppearance::new(
        MaterialColor::Dye(MaterialDye::new([224, 86, 31], 1.0).unwrap()),
        MaterialFinish::Anodised,
    );
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::default())
                .unwrap()
                .with_appearance(paint),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let mut state = EditorState {
        hovered: Some(SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::part(part, FaceKind::PositiveY),
        }),
        ..Default::default()
    };
    let mut history = EditorHistory::default();
    let mut mouse = ButtonInput::default();

    for expected_history in [1, 1] {
        mouse.press(GameAction::Secondary);
        handle_chroma_actions(&mouse, &mut graph, &mut state, &mut history, paint);
        mouse.clear();
        mouse.release(GameAction::Secondary);
        handle_chroma_actions(&mouse, &mut graph, &mut state, &mut history, paint);
        mouse.clear();
        assert_eq!(history.undo.len(), expected_history);
        assert_eq!(
            graph.part(part).unwrap().appearance(),
            Some(MaterialAppearance::BAKED)
        );
    }
}

#[test]
fn rotate_cycles_every_authored_tool_through_all_grid_orientations() {
    for tool in [Tool::Controller, Tool::GasEngine, Tool::ElectricEngine] {
        let mut state = EditorState::default();
        for expected in (1..AUTHORED_ORIENTATION_COUNT).chain(std::iter::once(0)) {
            cycle_orientation(&mut state, tool);
            assert_eq!(
                state.authored_orientation,
                expected,
                "{} should rotate",
                tool.label()
            );
        }
    }
}

#[test]
fn rotate_cycles_the_axis_of_an_active_vertex_drag() {
    let region = ShapeRegion::new(IVec3::ZERO, IVec3::ONE, ConstructionMaterial::Steel).unwrap();
    let start = crate::shape_tool::vertex_position(&region, [0, 0, 0]).unwrap();
    let ray_origin = start + Vec3::Z * 2.0;
    let ray_direction = Vec3::NEG_Z;
    let drag =
        crate::shape_tool::begin_group_drag(&region, [0, 0, 0], &[], ray_origin, ray_direction);
    let mut state = EditorState {
        pointer_position: Some(Vec2::ZERO),
        pointer_ray: Some((ray_origin, ray_direction)),
        vertex_drag: Some(drag),
        ..Default::default()
    };

    assert_eq!(cycle_orientation(&mut state, Tool::Shape), "Shape axis: Y");
    assert_eq!(state.vertex_drag.as_ref().unwrap().axis, 1);
}

#[test]
fn authored_orientation_cycle_contains_all_24_cube_orientations_once() {
    let mut signatures = Vec::new();
    for rotation in AUTHORED_ORIENTATIONS {
        let quaternion = rotation.quaternion();
        let signature = [Vec3::X, Vec3::Y, Vec3::Z]
            .map(|axis| (quaternion * axis).round().as_ivec3().to_array());
        assert!(!signatures.contains(&signature));
        signatures.push(signature);
    }
    assert_eq!(signatures.len(), 24);
}

#[test]
fn authored_preview_uses_a_tipped_rotation_selected_with_q() {
    let graph = ConstructionGraph::new();
    let mut state = EditorState {
        hovered: Some(SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        }),
        authored_orientation: 16,
        ..Default::default()
    };

    refresh_tool_preview(&graph, &mut state, Tool::GasEngine);

    let preview = state.preview.expect("gas engine has a ground preview");
    assert_eq!(preview.spec.pose.rotation.quarter_turns_xyz(), [1, 0, 0]);
    let (minimum, maximum) = crate::builder::part_world_bounds(preview.spec.into());
    assert!((maximum.y - minimum.y - 0.75).abs() < 1.0e-6);
    assert!((maximum.z - minimum.z - 0.50).abs() < 1.0e-6);
}

#[test]
fn bearing_shortcuts_are_gated_and_adjust_the_requested_diameter() {
    let mut keyboard = ButtonInput::default();
    keyboard.press(GameAction::BearingOuterIncrease);
    assert_eq!(
        requested_bearing_dimension_adjustment(&keyboard, Some(Tool::Bearing), false),
        Some((BearingDimensionTarget::Outer, 1))
    );
    assert_eq!(
        requested_bearing_dimension_adjustment(&keyboard, Some(Tool::Block), false),
        None
    );
    assert_eq!(
        requested_bearing_dimension_adjustment(&keyboard, Some(Tool::Bearing), true),
        None
    );

    keyboard.reset_all();
    keyboard.press(GameAction::NudgeUp);
    assert_eq!(
        requested_bearing_dimension_adjustment(&keyboard, Some(Tool::Bearing), false),
        None
    );

    let increased = adjusted_bearing_dimensions(
        BearingDimensions::default(),
        BearingDimensionTarget::Outer,
        1,
    );
    assert!((increased.outer_diameter() - 0.30).abs() < 1.0e-6);
    assert!((increased.inner_diameter() - 0.10).abs() < f32::EPSILON);

    keyboard.reset_all();
    keyboard.press(GameAction::BearingInnerIncrease);
    assert_eq!(
        requested_bearing_dimension_adjustment(&keyboard, Some(Tool::Bearing), false),
        Some((BearingDimensionTarget::Inner, 1))
    );
}

#[test]
fn cylinder_shortcuts_adjust_and_clamp_without_graph_history() {
    let mut keyboard = ButtonInput::default();
    keyboard.press(GameAction::CylinderOuterIncrease);
    assert_eq!(
        requested_cylinder_dimension_adjustment(&keyboard, Some(Tool::Cylinder), false),
        Some((CylinderDimensionTarget::Outer, 1))
    );
    keyboard.reset_all();
    keyboard.press(GameAction::CylinderInnerIncrease);
    assert_eq!(
        requested_cylinder_dimension_adjustment(&keyboard, Some(Tool::Cylinder), false),
        Some((CylinderDimensionTarget::Inner, 1))
    );
    keyboard.reset_all();
    keyboard.press(GameAction::CylinderSweepDecrease);
    assert_eq!(
        requested_cylinder_dimension_adjustment(&keyboard, Some(Tool::Cylinder), false),
        Some((CylinderDimensionTarget::Sweep, -1))
    );
    assert!(requested_cylinder_dimension_adjustment(&keyboard, Some(Tool::Block), false).is_none());

    let dimensions = CylinderDimensions::new(0.25, 0.20, 0.25).unwrap();
    let reduced = adjusted_cylinder_dimensions(dimensions, CylinderDimensionTarget::Outer, -1);
    assert!((reduced.outer_diameter() - 0.20).abs() < 1.0e-6);
    assert!((reduced.inner_diameter() - 0.15).abs() < 1.0e-6);
    let minimum = adjusted_cylinder_dimensions(reduced, CylinderDimensionTarget::Length, -1);
    assert!((minimum.axial_length() - 0.25).abs() < f32::EPSILON);
    let slice = adjusted_cylinder_dimensions(minimum, CylinderDimensionTarget::Sweep, -1);
    assert_eq!(slice.sweep_angle_degrees(), 345);
    let minimum_sweep = (0..30).fold(slice, |dimensions, _| {
        adjusted_cylinder_dimensions(dimensions, CylinderDimensionTarget::Sweep, -1)
    });
    assert_eq!(minimum_sweep.sweep_angle_degrees(), 15);

    let graph = ConstructionGraph::new();
    let history = EditorHistory::default();
    let settings = CylinderToolSettings {
        dimensions: minimum,
        ..Default::default()
    };
    assert_eq!(graph.part_count(), 0);
    assert!(history.undo.is_empty() && history.redo.is_empty());
    assert_eq!(settings.dimensions, minimum);
}

#[test]
fn bearing_adjustments_clamp_and_remain_outside_history() {
    let mut settings = BearingToolSettings {
        dimensions: BearingDimensions::new(0.20, 0.15).unwrap(),
    };
    let history = EditorHistory::default();
    settings.dimensions =
        adjusted_bearing_dimensions(settings.dimensions, BearingDimensionTarget::Outer, -1);
    assert!((settings.dimensions.outer_diameter() - 0.15).abs() < 1.0e-6);
    assert!((settings.dimensions.inner_diameter() - 0.10).abs() < 1.0e-6);
    assert!(history.undo.is_empty());
    assert!(history.redo.is_empty());

    let minimum = adjusted_bearing_dimensions(
        BearingDimensions::new(0.05, 0.0).unwrap(),
        BearingDimensionTarget::Outer,
        -1,
    );
    assert_eq!(minimum, BearingDimensions::new(0.05, 0.0).unwrap());
    let maximum_inner = adjusted_bearing_dimensions(
        BearingDimensions::default(),
        BearingDimensionTarget::Inner,
        1,
    );
    assert!((maximum_inner.inner_diameter() - 0.15).abs() < 1.0e-6);

    let hud = tool_status_line(
        Tool::Bearing,
        settings.dimensions,
        CylinderDimensions::default(),
        None,
        mechanic_core::ConstructionMaterial::Steel,
    );
    assert!(hud.contains("Outer: 0.15 m"));
    assert!(hud.contains("Inner: 0.10 m"));
    assert!(hud.contains("Shift+←/→"));
}

#[test]
fn hammer_charge_is_monotonic_and_clamped() {
    let tap = hammer_impulse_magnitude(0.0);
    let half = hammer_impulse_magnitude(HAMMER_CHARGE_SECONDS * 0.5);
    let full = hammer_impulse_magnitude(HAMMER_CHARGE_SECONDS);
    assert!((tap - HAMMER_MIN_IMPULSE).abs() < f32::EPSILON);
    assert!(tap < half && half < full);
    assert!((full - HAMMER_MAX_IMPULSE).abs() < f32::EPSILON);
    assert!((full - 4_000.0).abs() < f32::EPSILON);
    assert!((hammer_impulse_magnitude(100.0) - HAMMER_MAX_IMPULSE).abs() < f32::EPSILON);
}

#[test]
fn hard_hammer_impulses_are_delivered_in_collision_safe_steps() {
    let mut graph = ConstructionGraph::new();
    let spec = CuboidSpec::new(
        [1, 1, 1],
        BuildPose::from_half_grid(IVec3::new(0, 1, 0), GridRotation::default()),
    )
    .unwrap();
    graph.apply(BuildCommand::Spawn(spec)).unwrap();
    let creation = graph.compile().unwrap();
    let root = creation.compounds[0].root_translation;
    let transform = GpuTransform {
        position: [root.x, root.y, root.z, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
    };
    let impulse = Vec3::X * HAMMER_MAX_IMPULSE;
    let local_point = Vec3::new(0.0, 0.125, 0.0);

    let (ticks, impulse_per_tick) = hammer_delivery(&creation, transform, 0, local_point, impulse);

    assert!(ticks > 1);
    assert!(ticks <= crate::editor::hammer::HAMMER_MAX_DELIVERY_TICKS);
    assert!(impulse_per_tick.length() * f32::from(ticks) < impulse.length());
    assert!(
        hammer_point_travel(&creation, transform, 0, local_point, impulse_per_tick)
            <= crate::editor::hammer::HAMMER_MAX_POINT_TRAVEL_PER_TICK + f32::EPSILON
    );

    let mut heavy_graph = ConstructionGraph::new();
    let heavy = CuboidSpec::new(
        [4, 4, 4],
        BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
    )
    .unwrap();
    heavy_graph.apply(BuildCommand::Spawn(heavy)).unwrap();
    let heavy_creation = heavy_graph.compile().unwrap();
    let heavy_root = heavy_creation.compounds[0].root_translation;
    let heavy_transform = GpuTransform {
        position: [heavy_root.x, heavy_root.y, heavy_root.z, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
    };
    let (heavy_ticks, heavy_impulse_per_tick) =
        hammer_delivery(&heavy_creation, heavy_transform, 0, Vec3::ZERO, impulse);
    assert!(
        (heavy_impulse_per_tick.length() * f32::from(heavy_ticks) - impulse.length()).abs()
            < 1.0e-3
    );
}

#[test]
fn moving_frame_raycast_matches_exact_authored_feature_geometry() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([4; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let owner = mechanic_core::SolidOwner::Part(part);
    let edge = graph.evaluated_solid(owner).unwrap().logical_edges[0].key;
    graph
        .apply(BuildCommand::AddShapeFeature(
            mechanic_core::ShapeFeature::new(
                [mechanic_core::EdgeChainRef { owner, edge }],
                mechanic_core::EdgeTreatment::Chamfer,
                20,
            ),
        ))
        .unwrap();
    let frame = mechanic_core::ConstructionFrame::new(
        Vec3::new(3.0, 1.0, -2.0),
        Quat::from_rotation_z(0.61) * Quat::from_rotation_x(-0.4),
    )
    .unwrap();
    graph.reframe_parts([part], frame).unwrap();
    let creation = graph.compile().unwrap();
    let position = Vec3::new(-4.0, 5.0, 8.0);
    let rotation = Quat::from_rotation_y(0.7) * Quat::from_rotation_x(0.3);
    let transforms = [GpuTransform {
        position: position.extend(0.0).to_array(),
        rotation: rotation.to_array(),
    }];
    for x in [-0.49, 0.0, 0.49] {
        for z in [-0.49, 0.0, 0.49] {
            let origin = frame.point(Vec3::new(x, 2.0, z));
            let direction = frame.vector(Vec3::NEG_Y);
            let authored = crate::builder::raycast::raycast_construction_with_ground(
                &graph, origin, direction, None,
            );
            let world_origin =
                position + rotation * (origin - creation.compounds[0].root_translation);
            let world_direction = rotation * direction;
            let live = raycast_simulation(
                &graph,
                &creation,
                &transforms,
                world_origin,
                world_direction,
            );
            assert_eq!(authored.is_some(), live.is_some());
            if let (Some(authored), Some(live)) = (authored, live) {
                assert_eq!(live.part, part);
                assert!((live.distance - authored.distance).abs() < 1.0e-4);
                let expected =
                    position + rotation * (authored.point - creation.compounds[0].root_translation);
                assert!(live.point.abs_diff_eq(expected, 1.0e-4));
            }
        }
    }
}

#[test]
fn moving_frame_bearing_raycast_uses_composed_anchor_and_axis() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([4; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let frame = mechanic_core::ConstructionFrame::new(
        Vec3::new(2.0, 3.0, 4.0),
        Quat::from_rotation_z(0.65),
    )
    .unwrap();
    graph.reframe_parts([part], frame).unwrap();
    let bearing = PlacedBearing {
        kind: mechanic_core::JointKind::Rotational,
        axis: Vec3::ZERO,
        source: FaceRef::part(part, FaceKind::PositiveY),
        anchor: frame.point(Vec3::Y * 0.5),
        dimensions: BearingDimensions::new(0.5, 0.1).unwrap(),
    };
    let creation = graph.compile().unwrap();
    let position = Vec3::new(-2.0, 4.0, 1.0);
    let rotation = Quat::from_rotation_x(-0.4);
    let transforms = [GpuTransform {
        position: position.extend(0.0).to_array(),
        rotation: rotation.to_array(),
    }];
    let anchor = position + rotation * (bearing.anchor - creation.compounds[0].root_translation);
    let axis = rotation * frame.vector(Vec3::Y);
    let origin = anchor + axis * 2.0 + rotation * frame.vector(Vec3::X * 0.15);
    let expected = crate::editor::raycast::raycast_bearing_annulus(
        origin,
        -axis,
        anchor,
        axis,
        bearing.dimensions,
    )
    .unwrap();
    let actual = crate::editor::raycast::raycast_simulation_bearings(
        &graph,
        &creation,
        &transforms,
        &[bearing],
        origin,
        -axis,
    )
    .unwrap();
    assert!((actual.1 - expected).abs() < 1.0e-5);
}

#[test]
fn hammer_raycast_uses_the_current_simulated_pose() {
    let mut graph = ConstructionGraph::new();
    let spec = CuboidSpec::new(
        [4, 4, 4],
        BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(_) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
        unreachable!()
    };
    let creation = graph.compile().unwrap();
    let transforms = [GpuTransform {
        position: [5.0, 1.0, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
    }];

    let hit = raycast_simulation(
        &graph,
        &creation,
        &transforms,
        Vec3::new(5.0, 1.0, 5.0),
        Vec3::NEG_Z,
    )
    .unwrap();
    assert_eq!(hit.body_index, 0);
    assert!(hit.point.abs_diff_eq(Vec3::new(5.0, 1.0, 0.5), 1.0e-5));
    assert!(
        raycast_simulation(
            &graph,
            &creation,
            &transforms,
            Vec3::new(0.0, 1.0, 5.0),
            Vec3::NEG_Z,
        )
        .is_none()
    );
}

#[test]
fn hammer_hits_pipe_walls_before_through_and_after_a_bend() {
    let pieces = crate::builder::pipe_run_pieces(
        &[
            Vec3::Y,
            Vec3::new(0.875, 1.0, 0.0),
            Vec3::new(0.875, 1.875, 0.0),
        ],
        &[crate::builder::PipeNode::Bend { span: 1 }],
        CylinderDimensions::new(0.25, 0.0, 1.0).unwrap(),
        ConstructionMaterial::Steel,
    )
    .unwrap();
    let graph = crate::builder::stage_pipe_run(
        &ConstructionGraph::new(),
        &pieces,
        crate::builder::PipeRunAttachment::Free,
    )
    .unwrap();
    let creation = graph.compile().unwrap();
    assert_eq!(creation.compounds.len(), 1);
    let root = &creation.compounds[0];
    for (position, rotation) in [
        (root.root_translation, root.root_rotation),
        (Vec3::new(3.0, 4.0, -2.0), Quat::from_rotation_y(0.7)),
    ] {
        let transforms = [GpuTransform {
            position: position.extend(0.0).to_array(),
            rotation: rotation.to_array(),
        }];
        let world_from_build = rotation * root.root_rotation.inverse();
        for point in [
            Vec3::new(0.375, 0.0, 0.0),
            Vec3::new(
                0.75 + 0.25 / 2.0_f32.sqrt(),
                0.25 - 0.25 / 2.0_f32.sqrt(),
                0.0,
            ),
            Vec3::new(1.0, 0.625, 0.0),
        ] {
            let build_origin = point + Vec3::Y + Vec3::Z;
            let authored = crate::builder::raycast::raycast_construction_with_ground(
                &graph,
                build_origin,
                Vec3::NEG_Z,
                None,
            )
            .expect("pipe wall must be visible");
            let direction = world_from_build * Vec3::NEG_Z;
            let hit = raycast_simulation(
                &graph,
                &creation,
                &transforms,
                position + world_from_build * (build_origin - root.root_translation),
                direction,
            )
            .expect("hammer must accept curved pipe walls");
            assert_eq!(hit.body_index, 0);
            assert!((hit.distance - authored.distance).abs() < 1.0e-5);
            assert!(hit.normal.abs_diff_eq(-direction, 1.0e-5));
        }
    }
}

#[test]
fn hammer_raycast_respects_a_cylinder_slice() {
    let mut graph = ConstructionGraph::new();
    let dimensions = CylinderDimensions::new(1.0, 0.0, 1.0)
        .unwrap()
        .with_sweep_angle_degrees(90)
        .unwrap();
    let spec = CylinderSpec::new(
        dimensions,
        BuildPose::new(IVec3::ZERO, GridRotation::default()),
    );
    let BuildOutcome::Spawned(_) = graph.apply(BuildCommand::SpawnCylinder(spec)).unwrap() else {
        unreachable!()
    };
    let creation = graph.compile().unwrap();
    let root = creation.compounds[0].root_translation;
    let transforms = [GpuTransform {
        position: [root.x, root.y, root.z, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
    }];

    assert!(
        raycast_simulation(
            &graph,
            &creation,
            &transforms,
            Vec3::new(0.3, 2.0, 0.0),
            Vec3::NEG_Y,
        )
        .is_some()
    );
    assert!(
        raycast_simulation(
            &graph,
            &creation,
            &transforms,
            Vec3::new(-0.3, 2.0, 0.0),
            Vec3::NEG_Y,
        )
        .is_none()
    );
}

/// A solid, welded slab of one-cell blocks with its minimum corner at the
/// origin, which is what a region drag needs underneath it.
fn welded_slab(size: IVec3) -> ConstructionGraph {
    let mut graph = ConstructionGraph::new();
    let mut previous: Option<PartId> = None;
    for z in 0..size.z {
        for y in 0..size.y {
            for x in 0..size.x {
                let spec = CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_half_grid(
                        IVec3::ONE + IVec3::new(x, y, z) * 2,
                        GridRotation::default(),
                    ),
                )
                .unwrap();
                let BuildOutcome::Spawned(id) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
                else {
                    unreachable!()
                };
                if let Some(first) = previous {
                    graph
                        .apply(BuildCommand::RigidLink(RigidLinkSpec { first, second: id }))
                        .unwrap();
                }
                previous = Some(id);
            }
        }
    }
    graph
}

fn region_drag_on(
    graph: &ConstructionGraph,
    plane: PlacementPlane,
    press: PointerSample,
) -> RegionDrag {
    let (_, spec) = graph.parts().next().expect("the slab has blocks");
    let start = spec.as_cuboid().expect("the slab is made of blocks");
    RegionDrag {
        start,
        press,
        plane,
        anchor_span: IVec3::ZERO,
        span: IVec3::ZERO,
        last_span: None,
        region: region_area(start, IVec3::ZERO),
        error: None,
    }
}

#[test]
fn shape_selection_uses_the_world_resolved_hover_hit() {
    let mut graph = welded_slab(IVec3::ONE);
    let ray_origin = Vec3::new(0.125, 2.0, 0.125);
    let ray_direction = Vec3::NEG_Y;
    let hit = raycast_construction(&graph, ray_origin, ray_direction)
        .expect("the world-resolved ray hits the block");
    let mut state = EditorState {
        hovered: Some(hit),
        pointer_position: Some(Vec2::ZERO),
        pointer_ray: Some((ray_origin, ray_direction)),
        ..EditorState::default()
    };
    let mut actions = ButtonInput::default();
    actions.press(GameAction::Primary);

    choose_region(
        &actions,
        &mut graph,
        &mut state,
        &mut EditorHistory::default(),
        Some(Vec2::ZERO),
        ray_origin,
        ray_direction,
    );

    assert!(state.region_drag.is_some());
}

#[test]
fn shape_selection_preserves_a_fine_placed_blocks_origin() {
    let mut graph = ConstructionGraph::new();
    let spec = CuboidSpec::new(
        [1, 1, 1],
        BuildPose::from_position_ticks(IVec3::new(60, 50, 50), GridRotation::default()),
    )
    .unwrap();
    graph.apply(BuildCommand::Spawn(spec)).unwrap();
    let ray_origin = Vec3::new(0.15, 2.0, 0.125);
    let ray_direction = Vec3::NEG_Y;
    let hit = raycast_construction(&graph, ray_origin, ray_direction)
        .expect("the ray hits the fine-placed block");
    let FaceOwner::Part(part) = hit.face.owner else {
        unreachable!("the ray hit a block")
    };
    let mut state = EditorState {
        hovered: Some(hit),
        pointer_position: Some(Vec2::ZERO),
        pointer_ray: Some((ray_origin, ray_direction)),
        ..EditorState::default()
    };
    let mut actions = ButtonInput::default();
    actions.press(GameAction::Primary);

    choose_region(
        &actions,
        &mut graph,
        &mut state,
        &mut EditorHistory::default(),
        Some(Vec2::ZERO),
        ray_origin,
        ray_direction,
    );
    commit_region_drag(&mut graph, &mut state, &mut EditorHistory::default());

    let region = state.active_region.and_then(|id| graph.region(id)).unwrap();
    assert_eq!(region.origin_steps(), IVec3::new(10, 0, 0));
    assert!(
        (region.bounds_steps().0.as_vec3() * POSITION_TICK_METERS)
            .abs_diff_eq(Vec3::new(0.025, 0.0, 0.0), 1.0e-7)
    );
    assert_eq!(graph.region_of(part), state.active_region);
}

#[test]
fn dragging_across_blocks_claims_all_of_them_as_one_region() {
    let graph = welded_slab(IVec3::new(3, 2, 1));
    // Straight down onto the top of the first block, which is the XZ plane
    // the pointer then slides along.
    let press = pointer_sample(Vec2::ZERO, Vec3::new(0.125, 2.0, 0.125), Vec3::NEG_Y);
    let mut state = EditorState {
        region_drag: Some(region_drag_on(&graph, PlacementPlane::Xz, press)),
        ..Default::default()
    };

    refresh_region_drag(
        &graph,
        &mut state,
        Vec2::new(100.0, 0.0),
        press.ray_origin,
        (Vec3::new(0.625, 0.0, 0.125) - press.ray_origin).normalize(),
    );

    let drag = state.region_drag.as_ref().unwrap();
    assert_eq!(drag.span, IVec3::new(2, 0, 0));
    assert_eq!(drag.region.size_cells(), IVec3::new(3, 1, 1));
    assert_eq!(drag.error, None, "three welded blocks are a valid area");
}

#[test]
fn rotate_mid_area_drag_extrudes_the_selection_into_a_box() {
    let graph = welded_slab(IVec3::new(3, 2, 1));
    let press = pointer_sample(Vec2::ZERO, Vec3::new(0.125, 2.0, 0.125), Vec3::NEG_Y);
    let mut state = EditorState {
        region_drag: Some(region_drag_on(&graph, PlacementPlane::Xz, press)),
        ..Default::default()
    };
    refresh_region_drag(
        &graph,
        &mut state,
        Vec2::new(100.0, 0.0),
        press.ray_origin,
        (Vec3::new(0.625, 0.0, 0.125) - press.ray_origin).normalize(),
    );

    // What Rotate does: keep the rectangle already dragged and re-anchor here.
    let rotated = pointer_sample(Vec2::ZERO, Vec3::new(0.125, 0.125, 2.0), Vec3::NEG_Z);
    {
        let drag = state.region_drag.as_mut().unwrap();
        drag.plane = drag.plane.cycle();
        assert_eq!(drag.plane, PlacementPlane::Xy);
        drag.anchor_span = drag.span;
        drag.press = rotated;
        drag.last_span = None;
    }

    refresh_region_drag(
        &graph,
        &mut state,
        Vec2::new(0.0, 100.0),
        rotated.ray_origin,
        (Vec3::new(0.125, 0.375, 0.125) - rotated.ray_origin).normalize(),
    );

    let drag = state.region_drag.as_ref().unwrap();
    assert_eq!(
        drag.span,
        IVec3::new(2, 1, 0),
        "the rotation keeps the extent and grows the third axis"
    );
    assert_eq!(drag.region.size_cells(), IVec3::new(3, 2, 1));
    assert_eq!(drag.error, None);
}

#[test]
fn releasing_a_valid_area_opens_it_for_editing() {
    let mut graph = welded_slab(IVec3::new(2, 1, 1));
    let press = pointer_sample(Vec2::ZERO, Vec3::new(0.125, 2.0, 0.125), Vec3::NEG_Y);
    let mut state = EditorState {
        region_drag: Some(region_drag_on(&graph, PlacementPlane::Xz, press)),
        ..Default::default()
    };
    refresh_region_drag(
        &graph,
        &mut state,
        Vec2::new(100.0, 0.0),
        press.ray_origin,
        (Vec3::new(0.375, 0.0, 0.125) - press.ray_origin).normalize(),
    );
    let mut history = EditorHistory::default();

    commit_region_drag(&mut graph, &mut state, &mut history);

    assert!(state.region_drag.is_none());
    let region = state.active_region.and_then(|id| graph.region(id)).unwrap();
    assert_eq!(region.size_cells(), IVec3::new(2, 1, 1));
    assert_eq!(history.undo.len(), 1);
}

#[test]
fn an_area_reaching_past_the_blocks_is_refused_rather_than_claimed() {
    let mut graph = welded_slab(IVec3::new(2, 1, 1));
    let press = pointer_sample(Vec2::ZERO, Vec3::new(0.125, 2.0, 0.125), Vec3::NEG_Y);
    let mut state = EditorState {
        region_drag: Some(region_drag_on(&graph, PlacementPlane::Xz, press)),
        ..Default::default()
    };
    // Three cells wide over a two-block slab: the far cell is empty.
    refresh_region_drag(
        &graph,
        &mut state,
        Vec2::new(100.0, 0.0),
        press.ray_origin,
        (Vec3::new(0.625, 0.0, 0.125) - press.ray_origin).normalize(),
    );
    assert!(state.region_drag.as_ref().unwrap().error.is_some());

    let mut history = EditorHistory::default();
    commit_region_drag(&mut graph, &mut state, &mut history);

    assert_eq!(graph.regions().count(), 0);
    assert!(state.active_region.is_none());
    assert!(history.undo.is_empty());
}

#[test]
fn placing_blocks_shows_the_same_plane_as_choosing_an_area() {
    let graph = ConstructionGraph::new();
    let hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::ZERO,
        face: FaceRef::ground(),
    };
    let candidate = candidate_from_hit(&graph, hit);
    let specs = block_sheet_specs(candidate.spec, IVec3::new(4, 1, 2), PlacementPlane::Xz).unwrap();
    let state = EditorState {
        block_drag: Some(BlockDrag {
            start: candidate,
            attachment: BlockAttachment::AutoWeld {
                source: FaceOwner::Ground,
            },
            start_guides: Vec::new(),
            press: pointer_sample(Vec2::ZERO, Vec3::Y, Vec3::NEG_Y),
            plane: PlacementPlane::Xz,
            anchor_span: IVec3::ZERO,
            span: IVec3::new(2, 0, 1),
            last_span: Some(IVec3::new(2, 0, 1)),
            volume: BlockVolume::new(candidate.spec, IVec3::new(2, 0, 1)).unwrap(),
            error: None,
        }),
        ..Default::default()
    };

    let (low, high, plane) =
        active_drag_plane(&state, &AppSimulation::default()).expect("a block drag has a plane");
    assert_eq!(plane, PlacementPlane::Xz);
    // Centred on the blocks about to be placed, exactly as an area is.
    let (sheet_low, sheet_high) = block_sheet_bounds(&specs).unwrap();
    assert!(low.abs_diff_eq(sheet_low, 1.0e-6));
    assert!(high.abs_diff_eq(sheet_high, 1.0e-6));

    assert!(
        active_drag_plane(&EditorState::default(), &AppSimulation::default()).is_none(),
        "no drag, no plane"
    );
}

#[test]
fn selecting_a_tool_cancels_pending_editor_state() {
    let mut graph = ConstructionGraph::new();
    let spec = CuboidSpec::new(
        [4, 4, 4],
        BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::BeginPending(PendingOperation::Weld(
            FaceRef::part(part, FaceKind::PositiveY),
        )))
        .unwrap();

    let mut app = App::new();
    app.insert_resource(EditorGraph(graph))
        .insert_resource(EditorState::default())
        .insert_resource(SelectedTool::from_editor_tool(Tool::Bearing))
        .add_systems(Update, handle_tool_change);

    app.update();

    assert!(app.world().resource::<EditorGraph>().0.pending().is_none());
}

#[test]
fn weld_highlight_contains_the_entire_rigid_body_only() {
    let mut graph = ConstructionGraph::new();
    let specs = [
        CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap(),
        CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 6, 0), GridRotation::default()),
        )
        .unwrap(),
        CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 10, 0), GridRotation::default()),
        )
        .unwrap(),
    ];
    let parts = specs.map(|spec| {
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        part
    });
    graph
        .apply(BuildCommand::Weld(mechanic_core::WeldSpec {
            first: FaceRef::part(parts[0], FaceKind::PositiveY),
            second: FaceRef::part(parts[1], FaceKind::NegativeY),
        }))
        .unwrap();
    graph
        .apply(BuildCommand::AddBearing(mechanic_core::BearingSpec::new(
            FaceRef::part(parts[1], FaceKind::PositiveY),
            FaceRef::part(parts[2], FaceKind::NegativeY),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::Y,
        )))
        .unwrap();

    assert_eq!(
        crate::builder::rigid_body_parts(&graph, parts[0]),
        parts[..2]
    );
    assert_eq!(
        crate::builder::rigid_body_parts(&graph, parts[2]),
        vec![parts[2]]
    );
}

#[test]
fn block_click_places_on_release_through_drag_path() {
    let mut graph = ConstructionGraph::new();
    let hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::ZERO,
        face: mechanic_core::FaceRef::ground(),
    };
    let candidate = candidate_from_hit(&graph, hit);
    let press = pointer_sample(
        Vec2::new(320.0, 240.0),
        Vec3::new(-0.3, 2.0, -0.2),
        Vec3::new(0.2, -1.0, 0.3),
    );
    let mut state = EditorState {
        hovered: Some(hit),
        preview: Some(candidate),
        pointer_position: Some(press.cursor),
        pointer_ray: Some((press.ray_origin, press.ray_direction)),
        ..Default::default()
    };
    let mut mouse = ButtonInput::default();
    let mut history = EditorHistory::default();

    mouse.press(GameAction::Primary);
    handle_block_actions(&mouse, &mut graph, &mut state, &mut history);
    assert_eq!(graph.part_count(), 0);
    assert!(state.block_drag.is_some());
    refresh_block_drag(
        &graph,
        &mut state,
        press.cursor,
        press.ray_origin,
        press.ray_direction,
    );
    assert_eq!(state.block_drag.as_ref().unwrap().volume.count(), 1);

    mouse.clear();
    mouse.release(GameAction::Primary);
    handle_block_actions(&mouse, &mut graph, &mut state, &mut history);
    assert_eq!(graph.part_count(), 1);
    assert!(state.block_drag.is_none());
    assert_eq!(history.undo.len(), 1);
}

#[test]
fn block_drag_dead_zone_and_motion_are_relative_to_the_press() {
    let graph = ConstructionGraph::new();
    let hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::ZERO,
        face: FaceRef::ground(),
    };
    let candidate = candidate_from_hit(&graph, hit);
    let press = pointer_sample(Vec2::ZERO, Vec3::new(0.0, 2.0, 0.0), Vec3::NEG_Y);
    let start_guide = SmartGuide {
        axis: 0,
        coordinate: 0.0,
        from: Vec3::ZERO,
        to: Vec3::Z,
    };
    let mut state = EditorState {
        block_drag: Some(BlockDrag {
            start: candidate,
            attachment: BlockAttachment::AutoWeld {
                source: FaceOwner::Ground,
            },
            start_guides: vec![start_guide],
            press,
            plane: PlacementPlane::Xz,
            anchor_span: IVec3::ZERO,
            span: IVec3::ZERO,
            last_span: None,
            volume: BlockVolume::new(candidate.spec, IVec3::ZERO).unwrap(),
            error: None,
        }),
        smart_guides: vec![start_guide],
        ..Default::default()
    };

    refresh_block_drag(
        &graph,
        &mut state,
        Vec2::new(4.99, 0.0),
        press.ray_origin,
        Quat::from_rotation_z(0.003) * Vec3::NEG_Y,
    );
    assert_eq!(state.block_drag.as_ref().unwrap().volume.count(), 1);

    for (target, expected) in [
        (Vec3::new(0.50, 2.0, 0.25), 6),
        (Vec3::new(-0.50, 2.0, -0.25), 6),
        (Vec3::new(0.50, 2.0, 0.0), 3),
        (Vec3::new(0.0, 2.0, -0.50), 3),
    ] {
        refresh_block_drag(
            &graph,
            &mut state,
            Vec2::new(10.0, 0.0),
            press.ray_origin,
            (Vec3::new(target.x, 0.0, target.z) - press.ray_origin).normalize(),
        );
        assert_eq!(state.block_drag.as_ref().unwrap().volume.count(), expected);
        assert!(state.smart_guides.contains(&start_guide));
    }
}

#[test]
fn cycling_the_plane_without_pointer_motion_stays_one_by_one() {
    let graph = ConstructionGraph::new();
    let hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::ZERO,
        face: FaceRef::ground(),
    };
    let candidate = candidate_from_hit(&graph, hit);
    let press = pointer_sample(
        Vec2::new(100.0, 100.0),
        Vec3::new(0.0, 2.0, 2.0),
        Vec3::new(0.0, -1.0, -1.0),
    );
    let mut state = EditorState {
        block_drag: Some(BlockDrag {
            start: candidate,
            attachment: BlockAttachment::AutoWeld {
                source: FaceOwner::Ground,
            },
            start_guides: Vec::new(),
            press,
            plane: PlacementPlane::Xz,
            anchor_span: IVec3::ZERO,
            span: IVec3::ZERO,
            last_span: None,
            volume: BlockVolume::new(candidate.spec, IVec3::ZERO).unwrap(),
            error: None,
        }),
        ..Default::default()
    };

    let drag = state.block_drag.as_mut().unwrap();
    drag.plane = drag.plane.cycle();
    assert_eq!(drag.plane, PlacementPlane::Xy);
    refresh_block_drag(
        &graph,
        &mut state,
        press.cursor,
        press.ray_origin,
        press.ray_direction,
    );
    assert_eq!(state.block_drag.as_ref().unwrap().volume.count(), 1);

    refresh_block_drag(
        &graph,
        &mut state,
        Vec2::new(110.0, 100.0),
        press.ray_origin,
        (Vec3::new(0.5, 0.0, 0.0) - press.ray_origin).normalize(),
    );
    assert_eq!(state.block_drag.as_ref().unwrap().volume.count(), 3);
}

#[test]
fn dragged_placement_is_one_atomic_history_step() {
    let mut graph = ConstructionGraph::new();
    let hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::ZERO,
        face: FaceRef::ground(),
    };
    let candidate = candidate_from_hit(&graph, hit);
    let mut state = EditorState {
        block_drag: Some(BlockDrag {
            start: candidate,
            attachment: BlockAttachment::AutoWeld {
                source: FaceOwner::Ground,
            },
            start_guides: Vec::new(),
            press: pointer_sample(Vec2::ZERO, Vec3::Y, Vec3::NEG_Y),
            plane: PlacementPlane::Xz,
            anchor_span: IVec3::ZERO,
            span: IVec3::new(2, 0, 1),
            last_span: Some(IVec3::new(2, 0, 1)),
            volume: BlockVolume::new(candidate.spec, IVec3::new(2, 0, 1)).unwrap(),
            error: None,
        }),
        ..Default::default()
    };
    let mut history = EditorHistory::default();
    let mut mouse = ButtonInput::default();
    mouse.press(GameAction::Primary);
    mouse.clear();
    mouse.release(GameAction::Primary);

    handle_block_actions(&mouse, &mut graph, &mut state, &mut history);

    assert_eq!(graph.part_count(), 6);
    assert_eq!(graph.weld_count(), 13);
    assert_eq!(history.undo.len(), 1);
    apply_history_action(HistoryAction::Undo, &mut graph, &mut state, &mut history);
    assert_eq!(graph.part_count(), 0);
    assert_eq!(graph.weld_count(), 0);
}

#[test]
fn wiring_picks_a_bearing_through_the_hole_the_ring_pick_misses() {
    let mut graph = ConstructionGraph::new();
    let support = CuboidSpec::new(
        [4, 4, 4],
        BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(support)).unwrap() else {
        unreachable!()
    };
    let bearing = PlacedBearing {
        kind: mechanic_core::JointKind::Rotational,
        axis: Vec3::ZERO,
        source: FaceRef::part(part, FaceKind::PositiveY),
        anchor: Vec3::Y,
        dimensions: BearingDimensions::default(),
    };

    // Straight down the axis passes through the hole, and whatever is
    // threaded through it, so the ring pick finds nothing there.
    let axis = Vec3::new(0.0, 3.0, 0.0);
    assert!(
        raycast_placed_bearings(
            &graph,
            None,
            &[bearing],
            axis,
            Vec3::NEG_Y,
            crate::editor::raycast::BearingPick::Ring
        )
        .is_none()
    );
    assert_eq!(
        raycast_placed_bearings(
            &graph,
            None,
            &[bearing],
            axis,
            Vec3::NEG_Y,
            crate::editor::raycast::BearingPick::Disc
        )
        .map(|hit| hit.0),
        Some(0)
    );

    // Past the rim it still misses, so the disc does not swallow the block.
    let outside = Vec3::new(bearing.dimensions.outer_diameter(), 3.0, 0.0);
    assert!(
        raycast_placed_bearings(
            &graph,
            None,
            &[bearing],
            outside,
            Vec3::NEG_Y,
            crate::editor::raycast::BearingPick::Disc
        )
        .is_none()
    );
}

#[test]
fn connector_pick_follows_a_simulated_bearing() {
    let mut graph = ConstructionGraph::new();
    let support = CuboidSpec::new(
        [4, 4, 4],
        BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(support)).unwrap() else {
        unreachable!()
    };
    let bearing = PlacedBearing {
        kind: mechanic_core::JointKind::Rotational,
        axis: Vec3::ZERO,
        source: FaceRef::part(part, FaceKind::PositiveY),
        anchor: Vec3::Y,
        dimensions: BearingDimensions::default(),
    };
    let creation = graph.compile().unwrap();
    let compound = creation
        .part_to_compound
        .iter()
        .find_map(|&(candidate, compound)| (candidate == part).then_some(compound))
        .unwrap();
    let translation = Vec3::new(4.0, 0.0, 0.0);
    let mut transforms = creation
        .compounds
        .iter()
        .map(|compound| GpuTransform {
            position: compound.root_translation.extend(0.0).to_array(),
            rotation: compound.root_rotation.to_array(),
        })
        .collect::<Vec<_>>();
    transforms[compound as usize].position =
        (creation.compounds[compound as usize].root_translation + translation)
            .extend(0.0)
            .to_array();
    let ray_origin = bearing.anchor + translation + Vec3::Y * 3.0;

    assert!(
        raycast_placed_bearings(
            &graph,
            None,
            &[bearing],
            ray_origin,
            Vec3::NEG_Y,
            crate::editor::raycast::BearingPick::Disc
        )
        .is_none(),
        "the authored bearing no longer sits under the pointer"
    );
    assert_eq!(
        raycast_rotational_bearings(
            &[bearing],
            ray_origin,
            Vec3::NEG_Y,
            crate::editor::raycast::BearingPick::Disc,
            |bearing| { simulation_placed_bearing_pose(&graph, &creation, &transforms, bearing) }
        )
        .map(|hit| hit.0),
        Some(0),
        "the connector should pick the bearing at its simulated pose"
    );
}

#[test]
fn placed_bearing_is_picked_before_support_and_attaches_on_release() {
    let mut graph = ConstructionGraph::new();
    let support = CuboidSpec::new(
        [4, 4, 4],
        BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(support)).unwrap() else {
        unreachable!()
    };
    let bearing = PlacedBearing {
        kind: mechanic_core::JointKind::Rotational,
        axis: Vec3::ZERO,
        source: FaceRef::part(part, FaceKind::PositiveY),
        anchor: Vec3::Y,
        dimensions: BearingDimensions::default(),
    };
    let mut state = EditorState {
        placed_bearings: vec![bearing],
        hovered_bearing: Some(0),
        attachment_bearing: Some(0),
        pointer_position: Some(Vec2::ZERO),
        pointer_ray: Some((Vec3::new(0.1, 3.0, 0.0), Vec3::NEG_Y)),
        preview: Some(bearing_attachment_candidate(
            &graph,
            bearing.source,
            bearing.anchor,
        )),
        ..Default::default()
    };

    let origin = Vec3::new(0.1, 3.0, 0.0);
    let (_, bearing_distance) = raycast_placed_bearings(
        &graph,
        None,
        &state.placed_bearings,
        origin,
        Vec3::NEG_Y,
        crate::editor::raycast::BearingPick::Ring,
    )
    .unwrap();
    let support_distance = raycast_construction(&graph, origin, Vec3::NEG_Y)
        .unwrap()
        .distance;
    assert!(bearing_distance < support_distance);
    assert!(
        raycast_placed_bearings(
            &graph,
            None,
            &state.placed_bearings,
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::NEG_Y,
            crate::editor::raycast::BearingPick::Ring
        )
        .is_none()
    );
    let tiny_hole = PlacedBearing {
        dimensions: BearingDimensions::new(0.25, 0.001).unwrap(),
        ..bearing
    };
    assert!(
        raycast_placed_bearings(
            &graph,
            None,
            &[tiny_hole],
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::NEG_Y,
            crate::editor::raycast::BearingPick::Ring
        )
        .is_none()
    );
    assert_eq!(graph.part_count(), 1);
    assert_eq!(graph.bearing_count(), 0);

    let mut mouse = ButtonInput::default();
    let mut history = EditorHistory::default();
    mouse.press(GameAction::Primary);
    handle_block_actions(&mouse, &mut graph, &mut state, &mut history);
    assert_eq!(graph.part_count(), 1);
    assert_eq!(graph.bearing_count(), 0);
    assert_eq!(state.placed_bearings.len(), 1);
    assert!(state.block_drag.is_some());

    mouse.clear();
    mouse.release(GameAction::Primary);
    handle_block_actions(&mouse, &mut graph, &mut state, &mut history);

    assert_eq!(state.placed_bearings, vec![bearing]);
    assert_eq!(graph.part_count(), 2);
    assert_eq!(graph.bearing_count(), 1);
    assert_eq!(graph.weld_count(), 0);
}

#[test]
fn oversized_bearing_claims_offset_block_preview_and_highlights_attachment() {
    let mut graph = ConstructionGraph::new();
    let support = CuboidSpec::new(
        [4, 4, 4],
        BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(support)).unwrap() else {
        unreachable!()
    };
    let bearing = PlacedBearing {
        kind: mechanic_core::JointKind::Rotational,
        axis: Vec3::ZERO,
        source: FaceRef::part(part, FaceKind::PositiveY),
        anchor: Vec3::Y,
        dimensions: BearingDimensions::new(0.80, 0.10).unwrap(),
    };
    let mut state = EditorState {
        hovered: Some(SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.36, 1.0, 0.0),
            face: bearing.source,
        }),
        placed_bearings: vec![bearing],
        pointer_position: Some(Vec2::ZERO),
        pointer_ray: Some((Vec3::new(0.36, 3.0, 0.0), Vec3::NEG_Y)),
        ..Default::default()
    };

    refresh_tool_preview(&graph, &mut state, Tool::Block);

    assert_eq!(state.hovered_bearing, None);
    assert_eq!(state.attachment_bearing, Some(0));
    assert!(state.preview_error.is_none());
    assert!(bearing_attachment_is_highlighted(
        Tool::Block,
        state.attachment_bearing,
        state.preview_error.as_ref(),
    ));
    let preview = state.preview.unwrap();
    assert!((preview.spec.pose.translation().x - 0.375).abs() < 1.0e-6);

    let mut mouse = ButtonInput::default();
    let mut history = EditorHistory::default();
    mouse.press(GameAction::Primary);
    handle_block_actions(&mouse, &mut graph, &mut state, &mut history);
    mouse.clear();
    mouse.release(GameAction::Primary);
    handle_block_actions(&mouse, &mut graph, &mut state, &mut history);

    assert_eq!(state.placed_bearings, vec![bearing]);
    assert_eq!(graph.bearing_count(), 1);
    assert_eq!(graph.weld_count(), 0);
    assert_eq!(
        graph.bearings().next().unwrap().1.dimensions,
        bearing.dimensions
    );
}

#[test]
fn bearing_claims_an_offset_pipe_preview_but_centres_it_by_default() {
    let mut graph = ConstructionGraph::new();
    let support = CuboidSpec::new(
        [4, 4, 4],
        BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(support)).unwrap() else {
        unreachable!()
    };
    let bearing = PlacedBearing {
        kind: mechanic_core::JointKind::Rotational,
        axis: Vec3::ZERO,
        source: FaceRef::part(part, FaceKind::PositiveY),
        anchor: Vec3::Y,
        dimensions: BearingDimensions::new(0.80, 0.10).unwrap(),
    };
    let mut state = EditorState {
        hovered: Some(SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.36, 1.0, 0.0),
            face: bearing.source,
        }),
        placed_bearings: vec![bearing],
        pointer_position: Some(Vec2::ZERO),
        pointer_ray: Some((Vec3::new(0.36, 3.0, 0.0), Vec3::NEG_Y)),
        ..Default::default()
    };

    refresh_tool_preview(&graph, &mut state, Tool::Cylinder);

    assert_eq!(state.attachment_bearing, Some(0));
    let preview = state.cylinder_preview.unwrap();
    let direction = preview.spec.pose.rotation.quaternion() * Vec3::Y;
    let inlet_center =
        preview.spec.pose.translation() - direction * preview.spec.dimensions.axial_length() * 0.5;
    assert!(inlet_center.abs_diff_eq(bearing.anchor, 1.0e-5));
}

#[test]
fn right_click_through_bearing_hole_deletes_block_but_keeps_bearing() {
    let mut graph = ConstructionGraph::new();
    let parts = [IVec3::new(0, 1, 0), IVec3::new(2, 1, 0)].map(|center| {
        let spec = CuboidSpec::new(
            [1; 3],
            BuildPose::from_half_grid(center, GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        part
    });
    let center_face = FaceRef::part(parts[0], FaceKind::PositiveY);
    let bearing = PlacedBearing {
        kind: mechanic_core::JointKind::Rotational,
        axis: Vec3::ZERO,
        source: FaceRef::part(parts[1], FaceKind::PositiveY),
        anchor: Vec3::new(0.0, 0.25, 0.0),
        dimensions: BearingDimensions::new(0.75, 0.40).unwrap(),
    };
    let state = EditorState {
        hovered: Some(SurfaceHit {
            distance: 1.0,
            point: bearing.anchor,
            face: center_face,
        }),
        hovered_bearing: None,
        attachment_bearing: Some(0),
        placed_bearings: vec![bearing],
        // A delete drag anchors on the press, so it needs the pointer.
        pointer_position: Some(Vec2::ZERO),
        pointer_ray: Some((Vec3::Y, Vec3::NEG_Y)),
        ..Default::default()
    };
    let mut mouse = ButtonInput::default();
    mouse.press(GameAction::Secondary);
    let mut app = App::new();
    app.insert_resource(mouse)
        .insert_resource(ButtonInput::<KeyCode>::default())
        .insert_resource(EditorGraph(graph))
        .insert_resource(state)
        .insert_resource(EditorHistory::default())
        .insert_resource(crate::chroma::ChromaBrush::default())
        .insert_resource(AppSimulation::default())
        .insert_resource(SelectedTool::from_editor_tool(Tool::Block))
        .insert_resource(BearingToolSettings::default())
        .insert_resource(CylinderToolSettings::default())
        .insert_resource(crate::ui::UiInput::default())
        .insert_resource(MaterialWheelState::default())
        .insert_resource(PlayerState {
            input_captured: true,
            ..Default::default()
        })
        .init_resource::<bevy::input::mouse::AccumulatedMouseMotion>()
        .add_systems(Update, handle_build_actions);

    app.update();
    {
        let state = app.world().resource::<EditorState>();
        assert!(state.delete_target.is_none());
        assert!(state.delete_drag.is_some());
    }
    {
        let mut mouse = app.world_mut().resource_mut::<ButtonInput<GameAction>>();
        mouse.clear();
        mouse.release(GameAction::Secondary);
    }
    app.update();

    let graph = app.world().resource::<EditorGraph>();
    let state = app.world().resource::<EditorState>();
    assert!(graph.0.part(parts[0]).is_none());
    assert!(graph.0.part(parts[1]).is_some());
    assert_eq!(state.placed_bearings, vec![bearing]);
}

#[test]
fn deleting_current_support_rehomes_bearing_to_remaining_ring_support() {
    let mut graph = ConstructionGraph::new();
    let supports = [IVec3::new(-1, 1, 0), IVec3::new(1, 1, 0)].map(|center| {
        let spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(center, GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        part
    });
    let target_spec = CuboidSpec::new(
        [1, 1, 1],
        BuildPose::from_half_grid(IVec3::new(0, 3, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(target) = graph.apply(BuildCommand::Spawn(target_spec)).unwrap()
    else {
        unreachable!()
    };
    let socket = PlacedBearing {
        kind: mechanic_core::JointKind::Rotational,
        axis: Vec3::ZERO,
        source: FaceRef::part(supports[0], FaceKind::PositiveY),
        anchor: Vec3::new(0.0, 0.25, 0.0),
        dimensions: BearingDimensions::new(0.50, 0.10).unwrap(),
    };
    graph
        .apply(BuildCommand::AddBearing(
            mechanic_core::BearingSpec::new(
                socket.source,
                FaceRef::part(target, FaceKind::NegativeY),
                socket.anchor,
                Vec3::Y,
            )
            .with_dimensions(socket.dimensions),
        ))
        .unwrap();

    let (graph, sockets, migrated) =
        stage_part_deletion_preserving_bearings(&graph, &[socket], &[supports[0]]).unwrap();

    assert_eq!(migrated, 1);
    assert!(graph.part(supports[0]).is_none());
    assert!(graph.part(supports[1]).is_some());
    assert_eq!(sockets.len(), 1);
    assert_eq!(
        sockets[0].source,
        FaceRef::part(supports[1], FaceKind::PositiveY)
    );
    let bearing = graph.bearings().next().unwrap().1;
    assert_eq!(bearing.source, sockets[0].source);
    assert_eq!(
        bearing.target,
        Some(FaceRef::part(target, FaceKind::NegativeY))
    );
    assert_eq!(graph.compile().unwrap().bearings.len(), 1);

    let (graph, sockets, migrated) =
        stage_part_deletion_preserving_bearings(&graph, &sockets, &[supports[1]]).unwrap();
    assert_eq!(migrated, 0);
    assert!(sockets.is_empty());
    assert_eq!(graph.bearing_count(), 0);
    assert!(graph.part(target).is_some());
}

#[test]
fn deleting_linear_support_preserves_occupied_side_and_travel_axis() {
    use mechanic_core::{CarriageFace, JointKind, LinearBearing, LinearBearingDimensions};
    let mut graph = ConstructionGraph::new();
    let supports = [IVec3::new(0, 1, 0), IVec3::new(1, 1, 0)].map(|center| {
        let spec =
            CuboidSpec::new([1; 3], BuildPose::new(center, GridRotation::default())).unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        part
    });
    let rail = LinearBearing {
        dimensions: LinearBearingDimensions::new(1.0, 0.4).unwrap(),
        mount_normal: Vec3::Y,
        face: CarriageFace::PositiveSide,
    };
    let socket = PlacedBearing {
        source: FaceRef::part(supports[0], FaceKind::PositiveY),
        anchor: Vec3::new(0.0, 0.375, 0.0),
        axis: Vec3::X,
        dimensions: BearingDimensions::default(),
        kind: JointKind::Linear(LinearBearing {
            face: CarriageFace::Top,
            ..rail
        }),
    };
    let surface =
        crate::builder::bearings::linear_carriage_face(socket.anchor, rail, socket.axis).unwrap();
    let candidate = crate::builder::linear_block_candidate(
        socket.anchor,
        rail,
        socket.axis,
        surface.center,
        [1; 3],
        GridRotation::default(),
    )
    .unwrap();
    let graph = crate::builder::stage_linear_block_batch_in_bounds(
        &graph,
        candidate,
        &[candidate.spec],
        crate::builder::LinearAttachment {
            source: socket.source,
            anchor: socket.anchor,
            rail,
            axis: socket.axis,
            rigid_targets: &[],
        },
        crate::builder::PlacementBounds::Garage,
    )
    .unwrap();
    let (graph, sockets, migrated) =
        stage_part_deletion_preserving_bearings(&graph, &[socket], &[supports[0]]).unwrap();
    assert_eq!(migrated, 1);
    assert_eq!(
        sockets[0].source.owner,
        mechanic_core::FaceOwner::Part(supports[1])
    );
    let bearing = graph.bearings().next().unwrap().1;
    assert_eq!(bearing.kind, JointKind::Linear(rail));
    assert_eq!(bearing.axis, Vec3::X);
    assert_eq!(graph.compile().unwrap().bearings.len(), 1);
}

#[test]
fn deleting_reusable_socket_removes_all_of_its_joint_attachments() {
    let mut graph = ConstructionGraph::new();
    let support_spec = CuboidSpec::new(
        [4, 4, 4],
        BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(support) = graph.apply(BuildCommand::Spawn(support_spec)).unwrap()
    else {
        unreachable!()
    };
    let targets = [IVec3::new(0, 9, 0), IVec3::new(2, 9, 0)].map(|center| {
        let target_spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(center, GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(target) = graph.apply(BuildCommand::Spawn(target_spec)).unwrap()
        else {
            unreachable!()
        };
        target
    });
    let socket = PlacedBearing {
        kind: mechanic_core::JointKind::Rotational,
        axis: Vec3::ZERO,
        source: FaceRef::part(support, FaceKind::PositiveY),
        anchor: Vec3::Y,
        dimensions: BearingDimensions::new(0.80, 0.10).unwrap(),
    };
    for target in targets {
        graph
            .apply(BuildCommand::AddBearing(
                mechanic_core::BearingSpec::new(
                    socket.source,
                    FaceRef::part(target, FaceKind::NegativeY),
                    socket.anchor,
                    Vec3::Y,
                )
                .with_dimensions(socket.dimensions),
            ))
            .unwrap();
    }
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec {
            first: targets[0],
            second: targets[1],
        }))
        .unwrap();
    let state = EditorState {
        hovered_bearing: Some(0),
        placed_bearings: vec![socket],
        ..Default::default()
    };
    let mut mouse = ButtonInput::default();
    mouse.press(GameAction::Secondary);
    let mut app = App::new();
    app.insert_resource(mouse)
        .insert_resource(ButtonInput::<KeyCode>::default())
        .insert_resource(EditorGraph(graph))
        .insert_resource(state)
        .insert_resource(EditorHistory::default())
        .insert_resource(crate::chroma::ChromaBrush::default())
        .insert_resource(AppSimulation::default())
        .insert_resource(SelectedTool::from_editor_tool(Tool::Block))
        .insert_resource(BearingToolSettings::default())
        .insert_resource(CylinderToolSettings::default())
        .insert_resource(crate::ui::UiInput::default())
        .insert_resource(MaterialWheelState::default())
        .insert_resource(PlayerState {
            input_captured: true,
            ..Default::default()
        })
        .init_resource::<bevy::input::mouse::AccumulatedMouseMotion>()
        .add_systems(Update, handle_build_actions);

    app.update();
    {
        let mut mouse = app.world_mut().resource_mut::<ButtonInput<GameAction>>();
        mouse.clear();
        mouse.release(GameAction::Secondary);
    }
    app.update();

    let graph = app.world().resource::<EditorGraph>();
    let state = app.world().resource::<EditorState>();
    assert_eq!(graph.0.part_count(), 3);
    assert_eq!(graph.0.bearing_count(), 0);
    assert_eq!(graph.0.rigid_link_count(), 0);
    assert!(state.placed_bearings.is_empty());
}

#[test]
fn delete_drag_uses_composed_centers_instead_of_other_frames_local_grid() {
    let mut graph = ConstructionGraph::new();
    let start = CuboidSpec::new([1; 3], BuildPose::default()).unwrap();
    let BuildOutcome::Spawned(local) = graph.apply(BuildCommand::Spawn(start)).unwrap() else {
        unreachable!()
    };
    let BuildOutcome::Spawned(remote) = graph.apply(BuildCommand::Spawn(start)).unwrap() else {
        unreachable!()
    };
    graph
        .reframe_parts(
            [remote],
            mechanic_core::ConstructionFrame::new(Vec3::X * 10.0, Quat::IDENTITY).unwrap(),
        )
        .unwrap();
    assert_eq!(
        delete_box_parts(&graph, start, IVec3::ZERO).unwrap(),
        vec![local]
    );
}

#[test]
fn delete_drag_selects_only_the_box_it_spans() {
    let mut graph = ConstructionGraph::new();
    let centers = [
        IVec3::new(1, 1, 1),
        IVec3::new(3, 1, 1),
        IVec3::new(1, 1, 3),
        IVec3::new(3, 1, 3),
        IVec3::new(1, 3, 1),
    ];
    let mut parts = Vec::new();
    for center in centers {
        let spec = CuboidSpec::new(
            [1; 3],
            BuildPose::from_half_grid(center, GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        parts.push(part);
    }
    let start = graph.part(parts[0]).copied().unwrap().as_cuboid().unwrap();

    // A flat span is the four in that plane; the block above is untouched.
    let flat = delete_box_parts(&graph, start, IVec3::new(1, 0, 1)).unwrap();
    assert_eq!(flat.len(), 4);
    assert!(!flat.contains(&parts[4]));

    // Rotating into the third axis reaches the one above too.
    let boxed = delete_box_parts(&graph, start, IVec3::new(1, 1, 1)).unwrap();
    assert_eq!(boxed.len(), 5);
    assert!(boxed.contains(&parts[4]));
}

fn wired_socket_graph() -> (ConstructionGraph, PlacedBearing, PartId) {
    let mut graph = ConstructionGraph::new();
    let spawn = |graph: &mut ConstructionGraph, x: i32| {
        let BuildOutcome::Spawned(id) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 4, 4],
                    BuildPose::new(IVec3::new(x, 2, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        id
    };
    let left = spawn(&mut graph, 0);
    let right = spawn(&mut graph, 4);
    let source = FaceRef::part(left, FaceKind::PositiveX);
    let anchor = Vec3::new(0.5, 0.5, 0.0);
    graph
        .apply(BuildCommand::AddBearing(mechanic_core::BearingSpec::new(
            source,
            FaceRef::part(right, FaceKind::NegativeX),
            anchor,
            Vec3::X,
        )))
        .unwrap();
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::new(IVec3::new(0, 12, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let socket = PlacedBearing {
        kind: mechanic_core::JointKind::Rotational,
        axis: Vec3::ZERO,
        source,
        anchor,
        dimensions: BearingDimensions::default(),
    };
    (graph, socket, controller)
}

#[test]
fn dragging_a_control_block_onto_a_bearing_wires_every_row_of_that_socket() {
    let (mut graph, socket, controller) = wired_socket_graph();
    let mut state = EditorState {
        hovered_bearing: Some(0),
        placed_bearings: vec![socket],
        ..Default::default()
    };
    let mut history = EditorHistory::default();
    let block = WireEnd::Controller(controller);

    assert_eq!(
        wire_drag_step(None, Some(block), true),
        WireDragStep::Begin(block)
    );
    assert_eq!(
        wire_drag_step(
            Some(WireDrag {
                from: block,
                armed: false
            }),
            Some(WireEnd::Bearing(0)),
            false
        ),
        WireDragStep::Connect(WireConnection::Drive {
            controller,
            bearing: 0
        })
    );

    let message = connect_drive_wire(&mut graph, &mut state, &mut history, controller, 0);
    assert!(message.contains("Wired"), "{message}");
    assert_eq!(graph.drive_link_count(), 1);
    assert!(state.construction_mesh_dirty);
}

#[test]
fn making_the_same_control_connection_twice_removes_it() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(input) = graph
        .apply(BuildCommand::SpawnInput(mechanic_core::InputSpec::new(
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::Spawned(seat) = graph
        .apply(BuildCommand::SpawnSeat(mechanic_core::SeatSpec::new(
            BuildPose::new(IVec3::new(8, 2, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::new(IVec3::new(16, 2, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let mut state = EditorState::default();
    let mut history = EditorHistory::default();

    let message = connect_control_link(
        &mut graph,
        &mut state,
        &mut history,
        BuildCommand::AddInputSeatLink(mechanic_core::InputSeatLinkSpec { input, seat }),
        "Linked Input to Seat",
    );

    assert_eq!(message, "Linked Input to Seat");
    assert_eq!(graph.input_seat_links().count(), 1);
    // The wire is a line in the drive overlay, and that overlay is only
    // rebuilt on request: without the flag it stays invisible until some
    // unrelated edit happens to dirty the mesh.
    assert!(state.construction_mesh_dirty);

    state.construction_mesh_dirty = false;
    let message = connect_control_link(
        &mut graph,
        &mut state,
        &mut history,
        BuildCommand::AddInputSeatLink(mechanic_core::InputSeatLinkSpec { input, seat }),
        "Linked Input to Seat",
    );

    assert_eq!(message, "Removed Input-to-Seat link");
    assert_eq!(graph.input_seat_links().count(), 0);
    assert!(state.construction_mesh_dirty);

    for expected in [
        "Linked Seat to Controller",
        "Removed Seat-to-Controller link",
    ] {
        state.construction_mesh_dirty = false;
        let message = connect_control_link(
            &mut graph,
            &mut state,
            &mut history,
            BuildCommand::AddSeatControllerLink(mechanic_core::SeatControllerLinkSpec {
                seat,
                controller,
            }),
            "Linked Seat to Controller",
        );

        assert_eq!(message, expected);
        assert!(state.construction_mesh_dirty);
    }
    assert_eq!(graph.seat_controller_links().count(), 0);
}

#[test]
fn a_wire_can_be_dragged_from_the_bearing_end_as_well() {
    let (_, _, controller) = wired_socket_graph();
    let drag = Some(WireDrag {
        from: WireEnd::Bearing(0),
        armed: false,
    });
    assert_eq!(
        wire_drag_step(drag, Some(WireEnd::Controller(controller)), false),
        WireDragStep::Connect(WireConnection::Drive {
            controller,
            bearing: 0
        })
    );
    // Two ends of the same kind never pair up.
    assert_eq!(
        wire_drag_step(drag, Some(WireEnd::Bearing(1)), false),
        WireDragStep::Cancel
    );
}

#[test]
fn releasing_where_the_wire_started_leaves_it_armed_for_a_second_click() {
    let (_, _, controller) = wired_socket_graph();
    let block = WireEnd::Controller(controller);
    let drag = WireDrag {
        from: block,
        armed: false,
    };

    assert_eq!(
        wire_drag_step(Some(drag), Some(block), false),
        WireDragStep::Arm
    );
    assert_eq!(
        wire_drag_step(
            Some(WireDrag {
                armed: true,
                ..drag
            }),
            Some(WireEnd::Bearing(0)),
            true
        ),
        WireDragStep::Connect(WireConnection::Drive {
            controller,
            bearing: 0
        })
    );
    // Letting go over empty space drops the wire instead.
    assert_eq!(
        wire_drag_step(Some(drag), None, false),
        WireDragStep::Cancel
    );
}

#[test]
fn making_the_same_drive_connection_twice_removes_it() {
    let (mut graph, socket, controller) = wired_socket_graph();
    let bearing = graph.bearings().next().unwrap().0;
    graph
        .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
            controller, bearing,
        )))
        .unwrap();
    let mut state = EditorState {
        hovered_bearing: Some(0),
        placed_bearings: vec![socket],
        ..Default::default()
    };
    let mut history = EditorHistory::default();

    let message = connect_drive_wire(&mut graph, &mut state, &mut history, controller, 0);
    assert!(message.contains("Removed"), "{message}");
    assert_eq!(graph.drive_link_count(), 0);
    assert!(graph.bearing_drive_link(bearing).is_none());
    assert!(state.construction_mesh_dirty);
}

#[test]
fn right_clicking_a_wired_bearing_changes_its_default_direction() {
    let (mut graph, socket, controller) = wired_socket_graph();
    let bearing = graph.bearings().next().unwrap().0;
    graph
        .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
            controller, bearing,
        )))
        .unwrap();
    let state = EditorState {
        hovered_bearing: Some(0),
        placed_bearings: vec![socket],
        ..Default::default()
    };
    let mut mouse = ButtonInput::default();
    mouse.press(GameAction::Secondary);
    let mut app = App::new();
    app.insert_resource(mouse)
        .insert_resource(ButtonInput::<KeyCode>::default())
        .insert_resource(EditorGraph(graph))
        .insert_resource(state)
        .insert_resource(EditorHistory::default())
        .insert_resource(crate::chroma::ChromaBrush::default())
        .insert_resource(AppSimulation::default())
        .insert_resource(SelectedTool::from_editor_tool(Tool::Block))
        .insert_resource(BearingToolSettings::default())
        .insert_resource(CylinderToolSettings::default())
        .insert_resource(crate::ui::UiInput::default())
        .insert_resource(MaterialWheelState::default())
        .insert_resource(PlayerState {
            input_captured: true,
            ..Default::default()
        })
        .init_resource::<bevy::input::mouse::AccumulatedMouseMotion>()
        .add_systems(Update, handle_build_actions);

    app.update();
    {
        let mut mouse = app.world_mut().resource_mut::<ButtonInput<GameAction>>();
        mouse.clear();
        mouse.release(GameAction::Secondary);
    }
    app.update();

    let graph = app.world().resource::<EditorGraph>();
    let state = app.world().resource::<EditorState>();
    assert_eq!(graph.0.drive_link_count(), 1);
    assert!(graph.0.drive_links().next().unwrap().1.reversed);
    assert_eq!(state.placed_bearings, vec![socket]);
    assert!(
        state
            .feedback
            .as_deref()
            .is_some_and(|message| message.contains("default direction"))
    );
}

#[test]
fn wiring_an_unattached_socket_reports_that_it_has_no_joint_yet() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(block) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [4, 4, 4],
                BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::new(IVec3::new(0, 12, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let mut state = EditorState {
        hovered_bearing: Some(0),
        placed_bearings: vec![PlacedBearing {
            kind: mechanic_core::JointKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(block, FaceKind::PositiveX),
            anchor: Vec3::new(0.5, 0.5, 0.0),
            dimensions: BearingDimensions::default(),
        }],
        ..Default::default()
    };
    let mut history = EditorHistory::default();

    let message = connect_drive_wire(&mut graph, &mut state, &mut history, controller, 0);
    assert!(message.starts_with("Cannot wire"), "{message}");
    assert_eq!(graph.drive_link_count(), 0);
}

#[test]
fn control_block_status_line_reports_how_many_bearings_are_wired() {
    let selected = tool_status_line(
        Tool::Controller,
        BearingDimensions::default(),
        CylinderDimensions::default(),
        Some(2),
        mechanic_core::ConstructionMaterial::Steel,
    );
    assert!(selected.contains("2 bearings wired"), "{selected}");
    assert!(
        selected.contains("Interact opens its program"),
        "{selected}"
    );

    let single = tool_status_line(
        Tool::Connector,
        BearingDimensions::default(),
        CylinderDimensions::default(),
        Some(1),
        mechanic_core::ConstructionMaterial::Steel,
    );
    assert!(single.contains("1 bearing wired"), "{single}");

    let none = tool_status_line(
        Tool::Controller,
        BearingDimensions::default(),
        CylinderDimensions::default(),
        None,
        mechanic_core::ConstructionMaterial::Steel,
    );
    assert!(none.contains("No block selected"), "{none}");
}

#[test]
fn pipette_copies_material_dimensions_authored_orientation_and_bearing_setup() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(cuboid) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [1; 3],
                BuildPose::new(IVec3::new(0, 1, 0), GridRotation::default()),
            )
            .unwrap()
            .with_material(ConstructionMaterial::Wood),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let cylinder_dimensions = CylinderDimensions::new(0.75, 0.25, 1.5).unwrap();
    let BuildOutcome::Spawned(cylinder) = graph
        .apply(BuildCommand::SpawnCylinder(
            CylinderSpec::new(
                cylinder_dimensions,
                BuildPose::new(IVec3::new(8, 2, 0), GridRotation::default()),
            )
            .with_material(ConstructionMaterial::Concrete),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let orientation = AUTHORED_ORIENTATIONS[17];
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::new(IVec3::new(16, 2, 0), orientation),
        )))
        .unwrap()
    else {
        unreachable!()
    };

    let shaped_spec = CuboidSpec::new(
        [1; 3],
        BuildPose::new(IVec3::new(24, 1, 0), GridRotation::default()),
    )
    .unwrap()
    .with_material(ConstructionMaterial::Concrete);
    let BuildOutcome::Spawned(shaped) = graph.apply(BuildCommand::Spawn(shaped_spec)).unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::RegionAdded(region) = graph
        .apply(BuildCommand::AddRegion(
            crate::editor::shape_actions::region_area(shaped_spec, IVec3::ZERO),
        ))
        .unwrap()
    else {
        unreachable!()
    };

    let mut state = EditorState::default();
    let mut selection = SelectedTool::default();
    selection.clear();
    let mut material = crate::hotbar::SelectedMaterial(ConstructionMaterial::Steel);
    let mut bearing = BearingToolSettings::default();
    let mut cylinder_settings = CylinderToolSettings::default();
    macro_rules! apply {
        ($setup:expr) => {
            crate::editor::shortcuts::apply_pipette_setup(
                $setup,
                &graph,
                &mut state,
                &mut selection,
                &mut material,
                &mut bearing,
                &mut cylinder_settings,
            )
        };
    }

    apply!(crate::editor::shortcuts::PipetteSetup::Part(cuboid));
    assert_eq!(selection.active_editor_tool(), Some(Tool::Block));
    assert_eq!(material.0, ConstructionMaterial::Wood);

    apply!(crate::editor::shortcuts::PipetteSetup::Part(cylinder));
    assert_eq!(selection.active_editor_tool(), Some(Tool::Cylinder));
    assert_eq!(material.0, ConstructionMaterial::Concrete);
    assert_eq!(cylinder_settings.dimensions, cylinder_dimensions);

    apply!(crate::editor::shortcuts::PipetteSetup::Part(controller));
    assert_eq!(selection.active_editor_tool(), Some(Tool::Controller));
    assert_eq!(state.authored_orientation, 17);

    let dimensions = BearingDimensions::new(0.9, 0.4).unwrap();
    apply!(crate::editor::shortcuts::PipetteSetup::Bearing(dimensions));
    assert_eq!(selection.active_editor_tool(), Some(Tool::Bearing));
    assert_eq!(bearing.dimensions, dimensions);

    material.0 = ConstructionMaterial::Wood;
    apply!(crate::editor::shortcuts::PipetteSetup::Ground);
    assert_eq!(selection.active_editor_tool(), Some(Tool::Block));
    assert_eq!(material.0, ConstructionMaterial::Wood);

    apply!(crate::editor::shortcuts::PipetteSetup::Part(shaped));
    assert_eq!(selection.active_editor_tool(), Some(Tool::Shape));
    assert_eq!(state.active_region, Some(region));
    assert_eq!(material.0, ConstructionMaterial::Concrete);
}

#[test]
fn pipette_uses_simulation_space_and_reports_no_target() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let creation = graph.compile().unwrap();
    let simulation = AppSimulation {
        transforms: vec![GpuTransform {
            position: [10.0, 0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }],
        creation: Some(creation),
        published_graph: graph.clone(),
        ..Default::default()
    };
    assert_eq!(
        crate::editor::shortcuts::pipette_at_ray(
            &graph,
            &EditorState::default(),
            &simulation,
            Vec3::new(10.0, 0.0, 5.0),
            Vec3::NEG_Z,
        ),
        Some(crate::editor::shortcuts::PipetteSetup::Part(part)),
    );
    assert_eq!(
        crate::editor::shortcuts::pipette_at_ray(
            &ConstructionGraph::new(),
            &EditorState::default(),
            &AppSimulation::default(),
            Vec3::Y,
            Vec3::Y,
        ),
        None,
    );
}

#[test]
fn clearing_the_hand_cancels_pending_edits_and_hammer_charge() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    crate::builder::begin_weld(&mut graph, FaceRef::part(part, FaceKind::PositiveY)).unwrap();
    let mut state = EditorState::default();
    let mut selection = SelectedTool::from_editor_tool(Tool::Weld);
    let mut hammer = crate::editor::hammer::HammerInteraction {
        charging: Some(crate::editor::hammer::HammerCharge {
            body_index: 0,
            local_point: Vec3::ZERO,
            direction: Vec3::Y,
            elapsed_seconds: 1.0,
            local_normal: Vec3::Y,
        }),
        pending: None,
    };
    crate::editor::shortcuts::clear_held_tool(&mut graph, &mut state, &mut selection, &mut hammer);
    assert_eq!(selection.active_editor_tool(), None);
    assert!(graph.pending().is_none());
    assert!(hammer.charging.is_none() && hammer.pending.is_none());
}

#[test]
fn the_panel_opens_on_a_hovered_control_block_and_blocks_the_keyboard() {
    let (graph, _, controller) = wired_socket_graph();
    let mut panel = crate::control_panel::ControlPanelState::default();
    assert!(!panel.is_open());
    assert!(!panel.blocks_keyboard());

    panel.open(controller);
    assert_eq!(panel.controller(), Some(controller));
    assert!(panel.blocks_keyboard(), "typing must not fire shortcuts");

    // One row per wired bearing, and none until the block is wired.
    assert!(crate::control_panel::panel_rows(&graph, controller).is_empty());

    panel.close();
    assert!(!panel.is_open());
}

#[test]
fn the_panel_opens_on_a_moving_simulation_control_block() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::default(),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let creation = graph.compile().unwrap();
    let mut actions = ButtonInput::default();
    actions.press(GameAction::Interact);
    let mut state = crate::editor::state::EditorState {
        hovered_simulation: Some(SimulationHit {
            part: controller,
            body_index: 0,
            distance: 1.0,
            point: Vec3::ZERO,
            normal: Vec3::Y,
        }),
        ..Default::default()
    };
    // `update_hover` clears ordinary editor targeting when the live hit is
    // nearer than the authored surface. The live target must survive.
    clear_editor_hover(&mut state);
    let mut app = App::new();
    app.insert_resource(actions)
        .insert_resource(crate::creation_menu::CreationMenuState::default())
        .insert_resource(crate::editor::state::EditorGraph(graph.clone()))
        .insert_resource(state)
        .insert_resource(crate::control_panel::ControlPanelState::default())
        .insert_resource(State::new(crate::world::AppSpace::World))
        .insert_resource(crate::simulation::state::AppSimulation {
            creation: Some(creation),
            published_graph: graph,
            ..Default::default()
        })
        .insert_resource(PlayerState {
            input_captured: true,
            ..Default::default()
        })
        .insert_resource(MaterialWheelState::default())
        .insert_resource(crate::pause_menu::PauseMenuState::default())
        .init_resource::<SelectedTool>()
        .add_systems(
            Update,
            crate::editor::shortcuts::handle_control_panel_shortcut,
        );

    app.update();

    assert_eq!(
        app.world()
            .resource::<crate::control_panel::ControlPanelState>()
            .controller(),
        Some(controller)
    );
}

#[test]
fn remembered_controller_does_not_capture_seat_or_empty_space_interactions() {
    let (mut graph, _, controller) = wired_socket_graph();
    let BuildOutcome::Spawned(seat) = graph
        .apply(BuildCommand::SpawnSeat(mechanic_core::SeatSpec::new(
            BuildPose::new(IVec3::new(20, 0, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        panic!("seat");
    };
    let mut actions = ButtonInput::default();
    actions.press(GameAction::Interact);
    let mut panel = crate::control_panel::ControlPanelState::default();
    panel.open(controller);
    panel.close();
    let mut app = App::new();
    app.insert_resource(actions)
        .insert_resource(crate::creation_menu::CreationMenuState::default())
        .insert_resource(crate::editor::state::EditorGraph(graph))
        .insert_resource(EditorState {
            selected_controller: Some(controller),
            ..Default::default()
        })
        .insert_resource(panel)
        .insert_resource(PlayerState {
            input_captured: true,
            ..Default::default()
        })
        .insert_resource(MaterialWheelState::default())
        .insert_resource(crate::pause_menu::PauseMenuState::default())
        .init_resource::<SelectedTool>()
        .add_systems(
            Update,
            crate::editor::shortcuts::handle_control_panel_shortcut,
        );

    for (aimed, seated) in [(None, false), (Some(seat), false), (Some(controller), true)] {
        app.world_mut()
            .resource_mut::<EditorState>()
            .hovered_simulation = aimed.map(|part| SimulationHit {
            part,
            body_index: 0,
            distance: 1.0,
            point: Vec3::ZERO,
            normal: Vec3::Y,
        });
        app.world_mut().resource_mut::<PlayerState>().seat = seated.then_some(seat);
        app.update();
        assert!(
            !app.world()
                .resource::<crate::control_panel::ControlPanelState>()
                .is_open(),
            "aim {aimed:?}, seated {seated}: remembered controller must not capture E",
        );
    }
}

#[test]
fn every_wire_of_one_socket_is_written_by_a_single_row_edit() {
    let (mut graph, socket, controller) = wired_socket_graph();
    let mut state = EditorState {
        hovered_bearing: Some(0),
        placed_bearings: vec![socket],
        ..Default::default()
    };
    let mut history = EditorHistory::default();
    connect_drive_wire(&mut graph, &mut state, &mut history, controller, 0);

    let rows = crate::control_panel::panel_rows(&graph, controller);
    assert_eq!(rows.len(), 1, "one socket is one joint row");
    let commands = crate::control_panel::set_row_commands(
        &rows[0],
        mechanic_core::DriveLimits::new(2.0, 30.0, None).unwrap(),
        mechanic_core::DriveProgram::default(),
        mechanic_core::DriveName::new("Tipper arm"),
        mechanic_core::ActuatorAssignment::Unpowered,
    );
    assert_eq!(commands.len(), rows[0].links.len());
    graph.apply_batch(commands).unwrap();
    for (_, link) in graph.controller_links(controller) {
        assert!((link.limits.max_torque_newton_meters() - 30.0).abs() < f32::EPSILON);
    }
}

#[test]
fn pipe_validation_rechecks_changes_to_world_geometry_and_placement_bounds() {
    use crate::builder::PlacementBounds;
    let mut graph = ConstructionGraph::new();
    let mut cached = None;
    let pipe = CylinderSpec::new(
        CylinderDimensions::default(),
        BuildPose::new(IVec3::Y * 8, GridRotation::default()),
    );
    let pieces = [crate::builder::PipeRunPiece {
        spec: PartSpec::Cylinder(pipe),
        inlet: FaceKind::NegativeY,
        outlet: FaceKind::PositiveY,
    }];
    for _ in 0..2 {
        assert!(
            crate::editor::pipe::PipeValidation::validate(
                &mut cached,
                &graph,
                &pieces,
                PlacementBounds::Garage
            )
            .is_ok()
        );
    }
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::SpawnCylinder(pipe)).unwrap()
    else {
        unreachable!()
    };
    assert!(
        crate::editor::pipe::PipeValidation::validate(
            &mut cached,
            &graph,
            &pieces,
            PlacementBounds::Garage
        )
        .is_err()
    );
    graph.apply(BuildCommand::Remove(part)).unwrap();
    assert!(
        crate::editor::pipe::PipeValidation::validate(
            &mut cached,
            &graph,
            &pieces,
            PlacementBounds::Garage
        )
        .is_ok()
    );
    let distant = [crate::builder::PipeRunPiece {
        spec: pieces[0].spec.with_pose(BuildPose::new(
            IVec3::new(1000, 8, 0),
            GridRotation::default(),
        )),
        ..pieces[0]
    }];
    assert!(
        crate::editor::pipe::PipeValidation::validate(
            &mut cached,
            &graph,
            &distant,
            PlacementBounds::Garage
        )
        .is_err()
    );
    assert!(
        crate::editor::pipe::PipeValidation::validate(
            &mut cached,
            &graph,
            &distant,
            PlacementBounds::World {
                origin: bevy::math::DVec2::ZERO
            }
        )
        .is_ok()
    );
}

#[test]
fn pipe_dimension_modes_cycle_without_mutating_the_current_value() {
    let endpoint = Vec3::new(0.0, 1.0, 0.0);
    let dimensions = CylinderDimensions::new(0.50, 0.25, 1.0).unwrap();
    let mut mode = PipeEditMode::Length;
    for expected in [
        PipeEditMode::OuterDiameter,
        PipeEditMode::InnerDiameter,
        PipeEditMode::Length,
    ] {
        mode = mode.next();
        assert_eq!(mode, expected);
        assert_eq!(endpoint, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(
            dimensions,
            CylinderDimensions::new(0.50, 0.25, 1.0).unwrap()
        );
    }
}

#[test]
fn pipe_bend_span_steps_freely_above_the_channel_width() {
    for outer_diameter in [0.05, 0.20, 0.25] {
        assert_eq!(constrained_pipe_bend_span(outer_diameter, 0), 1);
        assert_eq!(constrained_pipe_bend_span(outer_diameter, 3), 3);
    }
    assert_eq!(constrained_pipe_bend_span(0.30, 1), 2);
    assert_eq!(constrained_pipe_bend_span(0.30, 4), 4);
    assert_eq!(constrained_pipe_bend_span(0.20, 40), 32);
}

#[test]
fn bend_corner_sits_at_the_middle_of_the_last_channel_cell() {
    assert!((pipe_corner_inset(0.25) - 0.125).abs() < f32::EPSILON);
    assert!((pipe_corner_inset(0.30) - 0.25).abs() < f32::EPSILON);
    assert!((pipe_corner_inset(0.60) - 0.375).abs() < f32::EPSILON);
}

#[test]
fn widening_a_bent_pipe_rebases_corners_to_the_new_channel() {
    let mut corners = [Vec3::X * 0.875];
    let mut endpoint = Vec3::new(0.875, 0.875, 0.0);
    rebase_pipe_path(
        Vec3::ZERO,
        &mut corners,
        &mut endpoint,
        &[Vec3::X, Vec3::Y],
        &[2],
        pipe_corner_inset(0.25),
        pipe_corner_inset(0.50),
    );
    assert!(corners[0].abs_diff_eq(Vec3::X * 0.75, 1.0e-5));
    assert!(endpoint.abs_diff_eq(Vec3::new(0.75, 0.75, 0.0), 1.0e-5));
    let pieces = crate::builder::pipe_run_pieces(
        &[Vec3::ZERO, corners[0], endpoint],
        &[crate::builder::PipeNode::Bend { span: 2 }],
        CylinderDimensions::new(0.50, 0.0, 0.25).unwrap(),
        ConstructionMaterial::Steel,
    )
    .expect("the rebased run keeps whole-block straights");
    assert_eq!(pieces.len(), 3);
}

#[test]
fn bearing_pipe_drag_moves_only_across_the_bearing_plane() {
    let camera = Vec3::new(0.0, 2.0, -4.0);
    let press = (Vec3::ZERO - camera).normalize();
    let current = (Vec3::new(0.31, 0.0, 0.12) - camera).normalize();

    let offset = bearing_offset_from_rays(
        Vec3::ZERO,
        Vec3::Y,
        PlacementGrid::Centimetres25,
        camera,
        press,
        camera,
        current,
    )
    .unwrap();

    assert!(offset.abs_diff_eq(Vec3::X * 0.25, 1.0e-5));
}

#[test]
fn bearing_offset_drag_translates_the_whole_pipe_before_release() {
    let graph = ConstructionGraph::new();
    let dimensions = CylinderDimensions::default();
    let start = Vec3::ZERO;
    let endpoint = Vec3::Y * dimensions.axial_length();
    let camera = Vec3::new(0.0, 2.0, -4.0);
    let press_direction = (start - camera).normalize();
    let pieces = crate::builder::pipe_run_pieces(
        &[start, endpoint],
        &[],
        dimensions,
        ConstructionMaterial::Steel,
    )
    .unwrap();
    let mut state = EditorState {
        pipe_drag: Some(PipeDrag {
            attachment: BlockAttachment::Bearing {
                source: FaceRef::ground(),
                anchor: start,
                dimensions: BearingDimensions::default(),
            },
            start,
            corners: Vec::new(),
            endpoint,
            directions: vec![Vec3::Y],
            nodes: Vec::new(),
            pending_span: 1,
            branch: None,
            dimensions,
            material: ConstructionMaterial::Steel,
            appearance: MaterialAppearance::BAKED,
            mode: PipeEditMode::Length,
            bearing_offset: Some(BearingOffsetDrag {
                start,
                endpoint,
                normal: Vec3::Y,
            }),
            choosing_direction: false,
            press: pointer_sample(Vec2::ZERO, camera, press_direction),
            anchor_endpoint: endpoint,
            anchor_dimensions: dimensions,
            pieces,
            error: None,
        }),
        ..Default::default()
    };
    let current_direction = (Vec3::new(0.31, 0.0, 0.12) - camera).normalize();

    assert!(refresh_bearing_offset_drag(
        &graph,
        &mut state,
        camera,
        current_direction,
    ));

    let drag = state.pipe_drag.unwrap();
    assert!(drag.start.abs_diff_eq(Vec3::X * 0.25, 1.0e-5));
    assert!(
        drag.endpoint
            .abs_diff_eq(Vec3::X * 0.25 + Vec3::Y * 0.25, 1.0e-5)
    );
    let PartSpec::Cylinder(pipe) = drag.pieces[0].spec else {
        panic!("a straight run remains a cylinder")
    };
    assert!(
        pipe.pose
            .translation()
            .abs_diff_eq(Vec3::new(0.25, 0.125, 0.0), 1.0e-5)
    );
}

#[test]
fn pipe_turn_chooser_locks_only_perpendicular_aim_beyond_the_dead_zone() {
    let anchor = Vec3::Z;
    assert_eq!(
        pipe_turn_direction(Vec3::X, anchor, (Vec3::Z + Vec3::Y * 0.1).normalize()),
        Some(Vec3::Y)
    );
    assert!(
        pipe_turn_direction(Vec3::X, anchor, (Vec3::Z + Vec3::X * 0.1).normalize()).is_none(),
        "aim along the incoming axis cannot select a perpendicular direction"
    );
    assert!(pipe_turn_direction(Vec3::X, anchor, anchor).is_none());
}

#[test]
fn pipe_drag_reanchors_length_and_diameter_measurements() {
    let axis_origin = Vec3::ZERO;
    let direction = Vec3::Y;
    let camera = Vec3::new(0.0, 0.0, -4.0);
    let first_ray = (Vec3::new(0.0, 1.0, 0.0) - camera).normalize();
    let second_ray = (Vec3::new(0.0, 1.5, 0.0) - camera).normalize();
    let first = closest_axis_parameter(axis_origin, direction, camera, first_ray).unwrap();
    let second = closest_axis_parameter(axis_origin, direction, camera, second_ray).unwrap();
    assert!((first - 1.0).abs() < 1.0e-5);
    assert!((second - 1.5).abs() < 1.0e-5);
    assert!(pipe_pointer_delta(Vec3::Z, (Vec3::Z + Vec3::Y * 0.05).normalize()).abs() > 0.0);
}

#[test]
fn bend_activity_owns_wheel_only_after_turning_starts() {
    let dimensions = CylinderDimensions::default();
    let cylinder = CylinderSpec::new(dimensions, BuildPose::default());
    let make_drag = || PipeDrag {
        attachment: BlockAttachment::AutoWeld {
            source: FaceOwner::Ground,
        },
        start: Vec3::ZERO,
        corners: Vec::new(),
        endpoint: Vec3::Y * 0.25,
        directions: vec![Vec3::Y],
        nodes: Vec::new(),
        pending_span: 1,
        branch: None,
        dimensions,
        material: ConstructionMaterial::Steel,
        appearance: MaterialAppearance::BAKED,
        mode: PipeEditMode::Length,
        bearing_offset: None,
        choosing_direction: false,
        press: PointerSample {
            cursor: Vec2::ZERO,
            ray_origin: Vec3::ZERO,
            ray_direction: Vec3::Z,
        },
        anchor_endpoint: Vec3::Y * 0.25,
        anchor_dimensions: dimensions,
        pieces: vec![crate::builder::PipeRunPiece {
            spec: PartSpec::Cylinder(cylinder),
            inlet: FaceKind::NegativeY,
            outlet: FaceKind::PositiveY,
        }],
        error: None,
    };
    let mut state = EditorState {
        pipe_drag: Some(make_drag()),
        ..Default::default()
    };
    assert!(!state.pipe_bend_active());
    state.pipe_drag.as_mut().unwrap().choosing_direction = true;
    assert!(state.pipe_bend_active());
    state.pipe_drag.as_mut().unwrap().choosing_direction = false;
    state
        .pipe_drag
        .as_mut()
        .unwrap()
        .nodes
        .push(crate::builder::PipeNode::Bend { span: 1 });
    assert!(state.pipe_bend_active());
}

#[test]
fn first_leg_grows_to_fit_a_bend_pressed_straight_after_clicking() {
    let dimensions = CylinderDimensions::default();
    let press = PointerSample {
        cursor: Vec2::ZERO,
        ray_origin: Vec3::ZERO,
        ray_direction: Vec3::Z,
    };
    let graph = ConstructionGraph::new();
    let mut state = EditorState {
        pipe_drag: Some(PipeDrag {
            attachment: BlockAttachment::Free,
            start: Vec3::Y,
            corners: Vec::new(),
            endpoint: Vec3::Y * 1.25,
            directions: vec![Vec3::Y],
            nodes: Vec::new(),
            pending_span: 2,
            branch: None,
            dimensions,
            material: ConstructionMaterial::Steel,
            appearance: MaterialAppearance::BAKED,
            mode: PipeEditMode::Length,
            bearing_offset: None,
            choosing_direction: false,
            press,
            anchor_endpoint: Vec3::Y * 1.25,
            anchor_dimensions: dimensions,
            pieces: Vec::new(),
            error: None,
        }),
        ..Default::default()
    };

    crate::editor::pipe::begin_pipe_node(&graph, &mut state);
    let drag = state.pipe_drag.as_ref().unwrap();
    assert!(drag.choosing_direction, "the bend starts on the first leg");
    assert!(drag.endpoint.abs_diff_eq(Vec3::Y * 1.5, 1.0e-5));

    crate::editor::pipe::adjust_pipe_bend_span(&graph, &mut state, 1);
    let drag = state.pipe_drag.as_ref().unwrap();
    assert!(drag.endpoint.abs_diff_eq(Vec3::Y * 1.75, 1.0e-5));

    crate::editor::pipe::lock_pipe_node(&graph, &mut state, Vec3::Y, Vec3::X, press);
    let drag = state.pipe_drag.as_ref().unwrap();
    assert_eq!(drag.error, None);
    assert_eq!(drag.pieces.len(), 1, "the whole run is one 3 × 3 bend");
}
