//! Connector-owned button configuration and exclusive keyboard capture.

use crate::editor::{
    history::{EditorHistory, EditorSnapshot},
    state::{EditorGraph, EditorState},
};
use bevy::prelude::*;
use mechanic_core::{BuildCommand, ButtonMode, PartId, PartSpec};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Control {
    Key,
    Clear,
    Mode,
}

#[derive(Resource, Default)]
pub(crate) struct ButtonConfiguration {
    pub selected: Option<PartId>,
    pub capturing: bool,
    pub aim: Option<Control>,
}

pub(crate) fn capture(
    mut config: ResMut<ButtonConfiguration>,
    mut simulation: ResMut<crate::simulation::state::AppSimulation>,
    mut keyboard: ResMut<ButtonInput<KeyCode>>,
    mut graph: ResMut<EditorGraph>,
    mut editor: ResMut<EditorState>,
    mut history: ResMut<EditorHistory>,
) {
    if !config.capturing {
        return;
    }
    let key = keyboard
        .get_just_pressed()
        .find_map(|key| crate::sequencer::drive_key(*key));
    let cancel = keyboard.just_pressed(KeyCode::Escape);
    keyboard.reset_all();
    if cancel || key.is_some() {
        config.capturing = false;
        if !cancel
            && let Some(input) = config.selected
            && let Some(mut next) = graph.0.input_configuration(input).cloned()
        {
            let before = graph.0.clone();
            let revision = history.current_revision;
            next.key = key;
            commit(&mut graph, &mut editor, &mut history, input, next);
            if simulation.is_running() {
                simulation.reconcile_controller_values(&before, &graph.0);
                simulation.accept_controller_edit(revision, history.current_revision, &graph.0);
            }
        }
    }
}

pub(crate) fn commit(
    graph: &mut EditorGraph,
    editor: &mut EditorState,
    history: &mut EditorHistory,
    input: PartId,
    configuration: mechanic_core::InputConfiguration,
) {
    let previous = EditorSnapshot::capture(&graph.0, editor);
    match graph.0.apply(BuildCommand::SetInputConfiguration {
        input,
        configuration,
    }) {
        Ok(_) => {
            history.commit(previous);
        }
        Err(error) => editor.feedback = Some(error.to_string()),
    }
}

pub(crate) fn act(
    config: &mut ButtonConfiguration,
    graph: &mut EditorGraph,
    editor: &mut EditorState,
    history: &mut EditorHistory,
    actions: &mut ButtonInput<crate::controls::GameAction>,
) {
    if !actions.just_pressed(crate::controls::GameAction::Primary) {
        return;
    }
    let Some(control) = config.aim else { return };
    let Some(input) = config
        .selected
        .filter(|input| matches!(graph.0.part(*input), Some(PartSpec::Button(_))))
    else {
        return;
    };
    actions.clear_just_pressed(crate::controls::GameAction::Primary);
    let Some(mut next) = graph.0.input_configuration(input).cloned() else {
        return;
    };
    match control {
        Control::Key => {
            config.capturing = true;
            return;
        }
        Control::Clear => next.key = None,
        Control::Mode => {
            next.button_mode = if next.button_mode == ButtonMode::Momentary {
                ButtonMode::Toggle
            } else {
                ButtonMode::Momentary
            }
        }
    }
    commit(graph, editor, history, input, next);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::RunSystemOnce;
    use mechanic_core::{BuildOutcome, BuildPose, ButtonSpec, InputSize};

    fn world() -> (World, PartId) {
        let mut graph = EditorGraph::default();
        let BuildOutcome::Spawned(button) = graph
            .0
            .apply(BuildCommand::SpawnButton(ButtonSpec::new(
                InputSize::Panel,
                BuildPose::default(),
            )))
            .unwrap()
        else {
            panic!("button")
        };
        let mut world = World::new();
        world.insert_resource(graph);
        world.init_resource::<crate::simulation::state::AppSimulation>();
        world.init_resource::<EditorState>();
        world.init_resource::<EditorHistory>();
        world.init_resource::<ButtonInput<KeyCode>>();
        world.insert_resource(ButtonConfiguration {
            selected: Some(button),
            capturing: true,
            aim: None,
        });
        (world, button)
    }

    #[test]
    fn capture_consumes_key_before_gameplay_and_is_undoable() {
        let (mut world, button) = world();
        world
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyW);
        world.run_system_once(capture).unwrap();
        assert_eq!(
            world
                .resource::<EditorGraph>()
                .0
                .input_configuration(button)
                .unwrap()
                .key,
            mechanic_core::DriveKey::new('W')
        );
        assert!(
            !world
                .resource::<ButtonInput<KeyCode>>()
                .pressed(KeyCode::KeyW)
        );
        assert!(
            !world
                .resource::<ButtonInput<KeyCode>>()
                .just_pressed(KeyCode::KeyW)
        );
        assert!(!world.resource::<ButtonConfiguration>().capturing);
        assert_eq!(world.resource::<EditorHistory>().undo.len(), 1);
    }

    #[test]
    fn escape_cancels_capture_without_editing_or_propagating() {
        let (mut world, button) = world();
        world
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::Escape);
        world.run_system_once(capture).unwrap();
        assert_eq!(
            world
                .resource::<EditorGraph>()
                .0
                .input_configuration(button)
                .unwrap()
                .key,
            None
        );
        assert!(
            !world
                .resource::<ButtonInput<KeyCode>>()
                .just_pressed(KeyCode::Escape)
        );
        assert!(world.resource::<EditorHistory>().undo.is_empty());
    }

    #[test]
    fn overlay_click_consumes_wiring_press_but_part_click_does_not() {
        let (mut world, button) = world();
        let mut config = ButtonConfiguration {
            selected: Some(button),
            capturing: false,
            aim: Some(Control::Mode),
        };
        let mut graph = world.remove_resource::<EditorGraph>().unwrap();
        let mut editor = EditorState::default();
        let mut history = EditorHistory::default();
        let mut actions = ButtonInput::default();
        actions.press(crate::controls::GameAction::Primary);
        act(
            &mut config,
            &mut graph,
            &mut editor,
            &mut history,
            &mut actions,
        );
        assert!(!actions.just_pressed(crate::controls::GameAction::Primary));
        assert_eq!(
            graph.0.input_configuration(button).unwrap().button_mode,
            ButtonMode::Toggle
        );
        config.aim = None;
        actions.reset_all();
        actions.press(crate::controls::GameAction::Primary);
        act(
            &mut config,
            &mut graph,
            &mut editor,
            &mut history,
            &mut actions,
        );
        assert!(actions.just_pressed(crate::controls::GameAction::Primary));
    }
}
