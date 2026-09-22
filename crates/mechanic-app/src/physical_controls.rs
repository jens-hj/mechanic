//! Physical controller interaction and transient key ownership.

use crate::camera::{MainCamera, PlayerState};
use crate::controls::GameAction;
use crate::simulation::state::AppSimulation;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use mechanic_core::{ControllerKeys, DialFeedback, NumericParameter, PartId, PartSpec};

#[derive(Resource, Default)]
pub(crate) struct PhysicalControls {
    pub keys: ControllerKeys,
    momentary: Option<PartId>,
    dial: Option<DialGesture>,
    pub(crate) feedback: Vec<String>,
}

/// A native value offered as the starting point for a mixed dial.
#[derive(Clone, Copy)]
pub(crate) struct DialChoice {
    pub(crate) target: NumericParameter,
    pub(crate) value: f32,
    pub(crate) linear: bool,
    position: f32,
}

struct DialGesture {
    part: PartId,
    position: Option<f32>,
    choices: Vec<DialChoice>,
    choice: usize,
    selected: bool,
    operation_feedback: Option<String>,
}
impl PhysicalControls {
    /// Mixed targets and the highlighted row; selection leaves the chooser.
    pub(crate) fn choices(&self) -> Option<(&[DialChoice], usize)> {
        self.dial
            .as_ref()
            .filter(|gesture| gesture.position.is_none())
            .map(|gesture| (gesture.choices.as_slice(), gesture.choice))
    }

    pub(crate) fn captures_pointer(&self) -> bool {
        self.dial.is_some()
    }
}

/// Takes chooser keys before player actions and shortcuts inspect them.
pub(crate) fn capture(
    mut controls: ResMut<PhysicalControls>,
    mut keyboard: ResMut<ButtonInput<KeyCode>>,
) {
    let Some(gesture) = controls.dial.as_mut() else {
        return;
    };
    if keyboard.just_pressed(KeyCode::Escape) {
        keyboard.reset(KeyCode::Escape);
        controls.dial = None;
        controls.feedback.clear();
        return;
    }
    if gesture.position.is_some() {
        return;
    }
    if gesture.choices.is_empty() {
        controls.dial = None;
        controls.feedback.clear();
        return;
    }
    let mut keys = keyboard.get_just_pressed().copied().collect::<Vec<_>>();
    keys.sort_by_key(|key| format!("{key:?}"));
    for key in keys {
        if gesture.selected {
            keyboard.reset(key);
            continue;
        }
        let selected = match key {
            KeyCode::ArrowDown => {
                gesture.choice = (gesture.choice + 1) % gesture.choices.len();
                false
            }
            KeyCode::ArrowUp => {
                gesture.choice =
                    (gesture.choice + gesture.choices.len() - 1) % gesture.choices.len();
                false
            }
            KeyCode::Enter => true,
            _ => {
                if let Some(index) = crate::sequencer::drive_key(key)
                    .and_then(|key| key.to_string().parse::<usize>().ok())
                    .and_then(|digit| digit.checked_sub(1))
                    .filter(|index| *index < gesture.choices.len())
                {
                    gesture.choice = index;
                    true
                } else {
                    continue;
                }
            }
        };
        keyboard.reset(key);
        if selected {
            gesture.position = Some(gesture.choices[gesture.choice].position);
            gesture.selected = true;
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "physical interaction combines existing input and world resources"
)]
pub(crate) fn interact(
    mut controls: ResMut<PhysicalControls>,
    mut actions: ResMut<ButtonInput<GameAction>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    authored: Res<crate::editor::state::EditorGraph>,
    world: Res<crate::world::WorldRuntime>,
    pause: Res<crate::pause_menu::PauseMenuState>,
    mut editor: ResMut<crate::editor::state::EditorState>,
    mut simulation: ResMut<AppSimulation>,
    overlay: Res<crate::ui::UiInput>,
    config: Res<crate::button_config::ButtonConfiguration>,
    player: Res<PlayerState>,
    camera: Single<&GlobalTransform, With<MainCamera>>,
    windows: Query<&Window>,
    frozen: Res<crate::freeze::DimensionFreeze>,
) {
    if !simulation.is_running() {
        controls.keys.reset();
        controls.momentary = None;
        controls.dial = None;
        controls.feedback.clear();
        return;
    }
    let active = player.world_input_active()
        && windows.iter().any(|window| window.focused)
        && !overlay.blocks_keyboard()
        && !config.capturing
        && !pause.blocks_world_input();
    let graph = &authored.0;
    let origin = player
        .seat
        .and_then(|seat| crate::seat::seat_world_pose(graph, &simulation, seat))
        .map_or(
            player.position + Vec3::Y * crate::camera::eye_height(player.crouch),
            |(position, rotation)| position + rotation * Vec3::Y * crate::camera::SEATED_EYE_HEIGHT,
        );
    let hit = active
        .then(|| {
            crate::seat::raycast_seat_interaction(
                graph,
                &simulation,
                origin,
                camera.forward().as_vec3(),
            )
        })
        .flatten()
        .filter(|(_, distance)| {
            *distance <= 3.0
                && world
                    .raycast_terrain(origin, camera.forward().as_vec3(), 3.0)
                    .is_none_or(|(_, terrain)| terrain > *distance)
        })
        .map(|(part, _)| part);
    let suspended = frozen.suspended_controllers(&simulation);
    if let Some(button) = controls.momentary
        && (hit != Some(button) || !actions.pressed(GameAction::Interact))
    {
        controls.keys.release_button(button);
        controls.momentary = None;
    }
    if actions.just_pressed(GameAction::Interact)
        && let Some(button) = hit
        && matches!(graph.part(button), Some(PartSpec::Button(_)))
    {
        if !graph
            .input_configuration(button)
            .and_then(|config| config.controller)
            .is_some_and(|controller| suspended.contains(&controller))
        {
            controls.keys.press_button(graph, button);
        }
        controls.momentary = Some(button);
        actions.clear_just_pressed(GameAction::Interact);
    }
    let blocked_dial = controls.dial.as_ref().is_some_and(|gesture| {
        hit != Some(gesture.part)
            || !actions.pressed(GameAction::Interact)
            || graph
                .input_configuration(gesture.part)
                .and_then(|config| config.controller)
                .is_none_or(|controller| suspended.contains(&controller))
    });
    if blocked_dial || keyboard.just_pressed(KeyCode::Escape) {
        controls.dial = None;
        controls.feedback.clear();
    }
    if actions.just_pressed(GameAction::Interact)
        && let Some(part) = hit
        && matches!(graph.part(part), Some(PartSpec::Dial(_)))
    {
        actions.clear_just_pressed(GameAction::Interact);
        if let Some(config) = graph.input_configuration(part)
            && config
                .controller
                .is_some_and(|controller| !suspended.contains(&controller))
            && !config.analog.is_empty()
        {
            let effective = simulation.effective_graph();
            let values = config
                .analog
                .iter()
                .filter_map(|mapping| {
                    effective
                        .numeric_value(mapping.target)
                        .map(|value| (mapping, value))
                })
                .collect::<Vec<_>>();
            let position = match DialFeedback::from_values(
                values
                    .iter()
                    .map(|(mapping, value)| (mapping.range, *value)),
            ) {
                DialFeedback::Uniform(value) => Some(value),
                DialFeedback::Mixed => None,
            };
            let choices = values
                .iter()
                .filter_map(|(mapping, value)| {
                    mapping.range.position(*value).map(|position| DialChoice {
                        target: mapping.target,
                        value: *value,
                        linear: match mapping.target {
                            NumericParameter::Drive { link, .. } => effective
                                .drive_link(link)
                                .is_some_and(|spec| spec.linear_limits.is_some()),
                            NumericParameter::Gear { .. } => false,
                        },
                        position,
                    })
                })
                .collect();
            controls.dial = Some(DialGesture {
                part,
                position,
                choices,
                choice: 0,
                selected: false,
                operation_feedback: None,
            });
        }
    }
    if let Some(mut gesture) = controls.dial.take() {
        let selected = std::mem::take(&mut gesture.selected);
        controls.feedback.clear();
        if gesture.position.is_none() {
            controls
                .feedback
                .push("Mixed · ↑/↓ then Enter, or 1–9. Hold Interact.".into());
        }
        if let Some(position) = &mut gesture.position {
            controls
                .feedback
                .push("Hold Interact and drag horizontally · Shift: fine · Esc: finish".into());
            let fine =
                keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight);
            if !selected && motion.delta.x != 0.0 {
                let next = (*position + motion.delta.x * if fine { 0.0002 } else { 0.002 })
                    .clamp(0.0, 1.0);
                match simulation.operate_dial(gesture.part, next) {
                    Ok(constrained) => {
                        *position = next;
                        editor.drive_rows_dirty = true;
                        gesture.operation_feedback =
                            constrained.then(|| "Value constrained by controller limits".into());
                    }
                    Err(error) => gesture.operation_feedback = Some(error.to_string()),
                }
            }
        }
        if let Some(message) = &gesture.operation_feedback {
            controls.feedback.push(message.clone());
        }
        controls.dial = Some(gesture);
    }
    let controller = player
        .seat
        .filter(|seat| graph.seat_input(*seat).is_some())
        .and_then(|seat| graph.seat_controller(seat));
    let seated = controller
        .filter(|_| active)
        .into_iter()
        .flat_map(|controller| {
            keyboard.get_pressed().filter_map(move |key| {
                crate::sequencer::drive_key(*key).map(|key| (controller, key))
            })
        });
    controls.keys.update(graph, seated);
}

#[cfg(test)]
mod tests {
    mod interaction;
    use super::*;
    use crate::editor::state::EditorGraph;
    use mechanic_core::{
        BuildCommand, BuildOutcome, BuildPose, CreationDocument, DialSpec, InputSize,
    };

    #[derive(Resource, Default)]
    struct ObservedKeys(Vec<KeyCode>);

    fn chooser() -> App {
        let mut authored = EditorGraph::default();
        let BuildOutcome::Spawned(part) = authored
            .0
            .apply(BuildCommand::SpawnDial(DialSpec::new(
                InputSize::Panel,
                BuildPose::default(),
            )))
            .unwrap()
        else {
            panic!("expected dial")
        };
        let mut app = App::new();
        app.insert_resource(authored)
            .insert_resource(PhysicalControls {
                dial: Some(DialGesture {
                    part,
                    position: None,
                    choices: [0.2, 0.8]
                        .into_iter()
                        .map(|position| DialChoice {
                            target: NumericParameter::Gear {
                                controller: part,
                                kind: mechanic_core::EngineKind::Gas,
                                parameter: mechanic_core::GearParameter::ReverseCount,
                            },
                            value: position,
                            linear: false,
                            position,
                        })
                        .collect(),
                    choice: 0,
                    selected: false,
                    operation_feedback: None,
                }),
                feedback: vec!["Mixed".into()],
                ..Default::default()
            })
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<ObservedKeys>()
            .add_systems(Update, (capture, observe).chain());
        app
    }

    fn observe(keys: Res<ButtonInput<KeyCode>>, mut observed: ResMut<ObservedKeys>) {
        observed.0 = keys.get_pressed().copied().collect();
    }

    fn press(app: &mut App, key: KeyCode) {
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(key);
        app.update();
        let keys = app.world().resource::<ButtonInput<KeyCode>>();
        assert!(!keys.pressed(key));
        assert!(!keys.just_pressed(key));
        assert!(!keys.just_released(key));
        assert!(!app.world().resource::<ObservedKeys>().0.contains(&key));
    }

    #[test]
    fn mixed_chooser_cycles_before_downstream_actions_receive_keys() {
        let mut app = chooser();
        for (key, expected) in [
            (KeyCode::ArrowUp, 1),
            (KeyCode::ArrowDown, 0),
            (KeyCode::ArrowDown, 1),
        ] {
            press(&mut app, key);
            let controls = app.world().resource::<PhysicalControls>();
            let gesture = controls.dial.as_ref().unwrap();
            assert_eq!(gesture.choice, expected);
            assert_eq!(gesture.position, None);
            assert!(!gesture.selected);
            assert!(controls.captures_pointer());
        }
    }

    #[test]
    fn mixed_selection_sets_pointer_without_writing_authored_values() {
        for selection in [KeyCode::Enter, KeyCode::Digit2] {
            let mut app = chooser();
            let before = CreationDocument::from_graph(
                &app.world().resource::<EditorGraph>().0,
                "Capture",
                &[],
            );
            if selection == KeyCode::Enter {
                press(&mut app, KeyCode::ArrowDown);
            }
            press(&mut app, selection);
            let controls = app.world().resource::<PhysicalControls>();
            let gesture = controls.dial.as_ref().unwrap();
            assert_eq!(gesture.position, Some(0.8));
            assert!(
                gesture.selected,
                "selection suppresses movement on this frame"
            );
            assert!(controls.captures_pointer());
            assert_eq!(
                before,
                CreationDocument::from_graph(
                    &app.world().resource::<EditorGraph>().0,
                    "Capture",
                    &[],
                )
            );
        }
    }

    #[test]
    fn escape_cancels_mixed_and_selected_capture_before_shortcuts() {
        for selected in [false, true] {
            let mut app = chooser();
            if selected {
                press(&mut app, KeyCode::Enter);
            }
            press(&mut app, KeyCode::Escape);
            let controls = app.world().resource::<PhysicalControls>();
            assert!(!controls.captures_pointer());
            assert!(controls.feedback.is_empty());
        }
    }
}
