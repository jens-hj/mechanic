use super::*;
use crate::editor::history::EditorHistory;
use crate::editor::state::EditorGraph;
use crate::{GameAction, Tool};
pub(crate) fn fixture() -> (EditorGraph, EditorState) {
    let mut graph = ConstructionGraph::new();
    let mechanic_core::BuildOutcome::Spawned(part) = graph
        .apply(mechanic_core::BuildCommand::Spawn(
            mechanic_core::CuboidSpec::new(
                [4, 1, 4],
                mechanic_core::BuildPose::from_position_ticks(
                    IVec3::new(0, 50, 0),
                    mechanic_core::GridRotation::default(),
                ),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        panic!("part");
    };
    let spec = SuspensionSpec::new(
        Some(SpringSpec::default()),
        Some(ShockSpec::default()),
        Some(BumpStopSpec::default()),
    )
    .unwrap();
    let socket = PlacedBearing {
        source: mechanic_core::FaceRef::part(part, mechanic_core::FaceKind::PositiveY),
        anchor: Vec3::Y * 0.25,
        axis: Vec3::Y,
        dimensions: mechanic_core::BearingDimensions::new(spec.plates().diameter, 0.0).unwrap(),
        kind: JointKind::Suspension(spec),
    };
    let mut state = EditorState {
        placed_bearings: vec![socket],
        ..Default::default()
    };
    state.suspension.controls.select(socket, 0);
    (EditorGraph(graph), state)
}
fn begin(
    graph: &mut EditorGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    parameter: Parameter,
) -> ButtonInput<GameAction> {
    state.suspension.controls.aim = Some(Aim {
        control: Control::Parameter(parameter),
        direction: Vec2::X,
        pixels_per_step: 6.0,
    });
    let mut input = ButtonInput::default();
    input.press(GameAction::Primary);
    assert!(actions(
        graph,
        state,
        history,
        &input,
        Vec2::ZERO,
        Some(Tool::Connector),
        false
    ));
    input.clear();
    input
}
#[test]
fn every_parameter_steps_and_commits_once_on_release() {
    for parameter in Parameter::ALL {
        let (mut graph, mut state) = fixture();
        let mut history = EditorHistory::default();
        let original = state.placed_bearings[0].kind;
        let mut input = begin(&mut graph, &mut state, &mut history, parameter);
        assert!(actions(
            &mut graph,
            &mut state,
            &mut history,
            &input,
            Vec2::X * 6.0,
            Some(Tool::Connector),
            false
        ));
        assert_eq!(
            state.placed_bearings[0].kind, original,
            "preview stays local"
        );
        assert_eq!(history.undo.len(), 0);
        let gesture = state.suspension.controls.gesture.as_ref().unwrap();
        assert!(
            gesture.error.is_none(),
            "{parameter:?}: {:?}",
            gesture.error
        );
        assert!(
            (parameter.value(gesture.draft)
                - (parameter.value(gesture.original) + parameter.step()))
            .abs()
                < 1e-5,
            "{parameter:?}"
        );
        input.release(GameAction::Primary);
        actions(
            &mut graph,
            &mut state,
            &mut history,
            &input,
            Vec2::ZERO,
            Some(Tool::Connector),
            false,
        );
        assert_eq!(history.undo.len(), 1, "{parameter:?}");
        assert_ne!(state.placed_bearings[0].kind, original);
        assert!(state.suspension.controls.gesture.is_none());
        crate::editor::history::apply_history_action(
            crate::editor::history::HistoryAction::Undo,
            &mut graph.0,
            &mut state,
            &mut history,
        );
        assert_eq!(state.placed_bearings[0].kind, original);
        crate::editor::history::apply_history_action(
            crate::editor::history::HistoryAction::Redo,
            &mut graph.0,
            &mut state,
            &mut history,
        );
        assert_ne!(state.placed_bearings[0].kind, original);
    }
}
#[test]
fn invalid_release_and_secondary_cancel_do_not_edit_or_disconnect() {
    for cancel in [false, true] {
        let (mut graph, mut state) = fixture();
        let mut history = EditorHistory::default();
        let original = state.placed_bearings[0];
        let mut input = begin(&mut graph, &mut state, &mut history, Parameter::SpringId);
        actions(
            &mut graph,
            &mut state,
            &mut history,
            &input,
            Vec2::X * 6000.0,
            Some(Tool::Connector),
            false,
        );
        assert!(
            state
                .suspension
                .controls
                .gesture
                .as_ref()
                .unwrap()
                .error
                .is_some()
        );
        if cancel {
            input.press(GameAction::Secondary);
        } else {
            input.release(GameAction::Primary);
        }
        assert!(actions(
            &mut graph,
            &mut state,
            &mut history,
            &input,
            Vec2::ZERO,
            Some(Tool::Connector),
            false
        ));
        assert!(state.suspension.controls.gesture.is_none());
        assert_eq!(state.placed_bearings, vec![original]);
        assert!(history.undo.is_empty());
    }
}
#[test]
fn targets_survive_reordering_and_cancel_when_replaced() {
    let (mut graph, mut state) = fixture();
    let mut history = EditorHistory::default();
    let target = state.suspension.controls.selected.unwrap();
    let mut input = begin(&mut graph, &mut state, &mut history, Parameter::Compression);
    let mut other = state.placed_bearings[0];
    other.anchor += Vec3::X;
    state.placed_bearings.insert(0, other);
    assert_eq!(target.resolve(&state), Some(1));
    state.placed_bearings.remove(1);
    input.release(GameAction::Primary);
    assert!(actions(
        &mut graph,
        &mut state,
        &mut history,
        &input,
        Vec2::X * 12.0,
        Some(Tool::Connector),
        false
    ));
    assert!(state.suspension.controls.gesture.is_none());
    assert!(history.undo.is_empty());
    assert_eq!(state.placed_bearings, vec![other]);
}
#[test]
fn independent_inputs_and_attached_spacing_are_preserved() {
    let (_, state) = fixture();
    let JointKind::Suspension(spec) = state.placed_bearings[0].kind else {
        panic!("suspension")
    };
    let changed = Parameter::SpringLength.edit(spec, 0.55, true).unwrap();
    assert_eq!(changed.shock(), spec.shock());
    assert!((changed.initial_length() - spec.initial_length()).abs() < 1e-6);
    let changed = Parameter::Compression.edit(spec, 2.0, true).unwrap();
    assert_eq!(changed.spring(), spec.spring());
    assert_eq!(changed.bump_stop(), spec.bump_stop());
    assert_eq!(
        Parameter::StartingCompression
            .edit(spec, 0.0025, true)
            .unwrap_err(),
        "Release the opposite attachment to change mount spacing"
    );
    assert!(Parameter::Coils.edit(spec, 6.5, false).is_err());
}
#[test]
fn selectors_consume_the_whole_click_and_noop_drags_do_not_commit() {
    let (mut graph, mut state) = fixture();
    let mut history = EditorHistory::default();
    state.suspension.controls.aim = Some(Aim {
        control: Control::Component(2),
        direction: Vec2::X,
        pixels_per_step: 6.0,
    });
    let mut input = ButtonInput::default();
    input.press(GameAction::Primary);
    assert!(actions(
        &mut graph,
        &mut state,
        &mut history,
        &input,
        Vec2::ZERO,
        Some(Tool::Connector),
        false
    ));
    assert_eq!(state.suspension.controls.component, 2);
    input.clear();
    input.release(GameAction::Primary);
    assert!(actions(
        &mut graph,
        &mut state,
        &mut history,
        &input,
        Vec2::ZERO,
        Some(Tool::Connector),
        false
    ));
    let mut input = begin(&mut graph, &mut state, &mut history, Parameter::Compression);
    input.release(GameAction::Primary);
    actions(
        &mut graph,
        &mut state,
        &mut history,
        &input,
        Vec2::ZERO,
        Some(Tool::Connector),
        false,
    );
    assert!(history.undo.is_empty());
}
#[test]
fn graph_only_showcase_controls_survive_save_load_and_cardinal_transforms() {
    let mut document = crate::creation_store::read_document(std::path::Path::new(
        "../../creations/suspension-playground.mech",
    ))
    .unwrap();
    document.transform_cardinal(1, IVec3::new(8, 0, 8));
    let loaded = document.into_graph().unwrap();
    let mut graph = EditorGraph(loaded.graph);
    let mut state = EditorState::default();
    crate::suspension_editor::sync_sockets(&graph.0, &mut state);
    assert_eq!(state.placed_bearings.len(), 6);
    let socket = *state
        .placed_bearings
        .iter()
        .find(|s| matches!(s.kind, JointKind::Suspension(s) if s.shock().is_some()))
        .unwrap();
    state.suspension.controls.select(socket, 1);
    let mut history = EditorHistory::default();
    let mut input = begin(&mut graph, &mut state, &mut history, Parameter::Rebound);
    input.release(GameAction::Primary);
    actions(
        &mut graph,
        &mut state,
        &mut history,
        &input,
        Vec2::X * 12.0,
        Some(Tool::Connector),
        false,
    );
    assert_eq!(history.undo.len(), 1);
    let sockets = crate::suspension_editor::sockets(&state.placed_bearings);
    let document =
        mechanic_core::CreationDocument::from_graph(&graph.0, "world controls", &sockets);
    let serialized = ron::to_string(&document).unwrap();
    let loaded = ron::from_str::<mechanic_core::CreationDocument>(&serialized)
        .unwrap()
        .into_graph()
        .unwrap();
    assert_eq!(loaded.sockets, sockets);
    assert_eq!(
        loaded
            .graph
            .bearings()
            .map(|(_, b)| b.kind)
            .collect::<Vec<_>>(),
        graph.0.bearings().map(|(_, b)| b.kind).collect::<Vec<_>>()
    );
    loaded.graph.compile().unwrap();
}
#[test]
fn captured_drag_keeps_cursor_locked_and_restores_camera_look() {
    use bevy::{
        input::mouse::AccumulatedMouseMotion,
        window::{CursorGrabMode, CursorOptions, PrimaryWindow},
    };
    let (mut graph, mut state) = fixture();
    let mut history = EditorHistory::default();
    let input = begin(&mut graph, &mut state, &mut history, Parameter::Compression);
    let worlds = crate::world::WorldListState::empty_capture_garage();
    let mut app = App::new();
    app.init_resource::<crate::physical_controls::PhysicalControls>()
        .insert_resource(Time::<()>::default())
        .insert_resource(input)
        .insert_resource(AccumulatedMouseMotion {
            delta: Vec2::X * 10.0,
        })
        .init_resource::<crate::CreationMenuState>()
        .init_resource::<crate::ControlPanelState>()
        .init_resource::<crate::PauseMenuState>()
        .insert_resource(worlds)
        .init_resource::<crate::MaterialWheelState>()
        .insert_resource(state)
        .insert_resource(crate::SelectedTool::from_editor_tool(Tool::Connector))
        .insert_resource(State::new(crate::world::AppSpace::Garage))
        .init_resource::<crate::PlayerState>()
        .add_systems(Update, crate::camera::update_player_camera);
    let window = app
        .world_mut()
        .spawn((CursorOptions::default(), PrimaryWindow))
        .id();
    let camera = app
        .world_mut()
        .spawn((
            crate::camera::PlayerCamera::default(),
            Transform::default(),
            GlobalTransform::default(),
            crate::MainCamera,
        ))
        .id();
    let yaw = app
        .world()
        .get::<crate::camera::PlayerCamera>(camera)
        .unwrap()
        .yaw;
    app.update();
    assert!(
        (app.world()
            .get::<crate::camera::PlayerCamera>(camera)
            .unwrap()
            .yaw
            - yaw)
            .abs()
            < 1e-6
    );
    let cursor = app.world().get::<CursorOptions>(window).unwrap();
    assert!(!cursor.visible);
    assert_eq!(cursor.grab_mode, CursorGrabMode::Locked);
    app.world_mut()
        .resource_mut::<EditorState>()
        .suspension
        .controls
        .gesture = None;
    app.world_mut()
        .resource_mut::<ButtonInput<GameAction>>()
        .release(GameAction::Primary);
    app.update();
    assert!(
        (app.world()
            .get::<crate::camera::PlayerCamera>(camera)
            .unwrap()
            .yaw
            - yaw)
            .abs()
            > 0.001
    );
    assert!(!app.world().get::<CursorOptions>(window).unwrap().visible);
}
