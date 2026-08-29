//! Full-window pause modal and its inline exit confirmation.

#![allow(clippy::wildcard_imports)] // Mosaic's authoring vocabulary is meant to be globbed.

use bevy_mosaic::ui::*;
use mosaic_core::Effect;
use mosaic_macros::{component, view};

use super::components::{Action, ActionProps};
#[allow(unused_imports)]
use super::styles::*;
#[allow(unused_imports, clippy::wildcard_imports)]
use super::theme::*;
use super::{Handles, PauseAction, UiIntent};
use crate::pause_menu::PausePage;
use crate::{
    controls::{Controls, GameAction, InputChord},
    pause_menu::BindingCapture,
};

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Model {
    pub(crate) open: bool,
    pub(crate) page: PausePage,
    pub(crate) camera_fov_degrees: f32,
    pub(crate) controls: Controls,
    pub(crate) capture: Option<BindingCapture>,
    pub(crate) vehicle_conflicts: Vec<GameAction>,
}

#[component]
pub(crate) fn PauseMenu(handles: Handles) -> Element {
    let model = handles.pause;
    let fov = handles.pause_fov;
    let changed = handles.clone();
    Effect::new(move || {
        let value = fov.get();
        let applied = model.with(|found| found.camera_fov_degrees);
        if (value - applied).abs() > f32::EPSILON {
            changed.ask(UiIntent::Pause(PauseAction::SetCameraFov(value)));
        }
    });

    let options = handles.clone();
    let controls = handles.clone();
    let main = handles.clone();
    view! {
        col #mechanic.pause-veil width:fill height:fill align:center justify:center {
            col #mechanic.pause-sheet width:700px height:min-content gap:16px pad:24px {
                if model.with(|found| found.page == PausePage::Options) {
                    PauseOptions handles:(options.clone()) fov:(fov)
                } else if model.with(|found| found.page == PausePage::Controls) {
                    PauseControls handles:(controls.clone()) model:(model)
                } else {
                    PauseMain handles:(main.clone()) model:(model)
                }
            }
        }
    }
}

#[component]
fn PauseControls(handles: Handles, model: State<Model>) -> Element {
    let reset = handles.clone();
    let back = handles.clone();
    view! {
        col width:fill height:min-content gap:12px {
            text #mechanic.title "CONTROLS"
            text #mechanic.caption "Duplicates are allowed; conflicting actions fire together when their contexts overlap."
            // A scroll fills its parent; the wrapper gives that viewport a real
            // height while the scroll's own styles size its content.
            col width:fill height:520px {
                scroll {
                    col width:fill height:min-content gap:6px pad:(right:12px bottom:8px) {
                        for (action, ()) in { GameAction::ALL.map(|action| (action, ())) } {
                            (binding_entry(&handles, model, *action))
                        }
                    }
                }
            }
            row width:fill height:min-content gap:10px {
                Action label:"Reset All"
                    on-click:(move || reset.ask(UiIntent::Pause(PauseAction::ResetControls)))
                    width:1fr height:42px {
                    text #mechanic.value "Reset All"
                }
                Action label:"Back"
                    on-click:(move || back.ask(UiIntent::Pause(PauseAction::Back)))
                    width:1fr height:42px {
                    text #mechanic.value "Back"
                }
            }
        }
    }
}

fn binding_entry(handles: &Handles, model: State<Model>, action: GameAction) -> Element {
    view! {
        col width:fill height:min-content gap:4px {
            if is_group_start(action) {
                text #mechanic.section pad:(top:10px bottom:2px) (action.group())
            }
            (binding_row(handles, model, action))
        }
    }
}

fn is_group_start(action: GameAction) -> bool {
    GameAction::ALL
        .iter()
        .position(|candidate| candidate.group() == action.group())
        .is_some_and(|index| GameAction::ALL[index] == action)
}

fn binding_row(handles: &Handles, model: State<Model>, action: GameAction) -> Element {
    let conflict = move || {
        model.with(|found| {
            found.controls.conflicts(action) || found.vehicle_conflicts.contains(&action)
        })
    };
    view! {
        row #mechanic.list-row width:fill height:40px align:center gap:8px pad:(left:10px right:8px top:4px bottom:4px) {
            text #mechanic.label width:190px (action.label())
            (binding_chip(handles, model, action, 0))
            (binding_chip(handles, model, action, 1))
            text #mechanic.caption width:72px font-color:accent.danger {
                if conflict() { "CONFLICT" } else { "" }
            }
        }
    }
}

fn binding_chip(
    handles: &Handles,
    model: State<Model>,
    action: GameAction,
    slot: usize,
) -> Element {
    let bind = handles.clone();
    let clear = handles.clone();
    let binding_semantics = format!("{} binding {}", action.label(), slot + 1);
    let clear_semantics = format!("Clear {} binding {}", action.label(), slot + 1);
    let label = move || {
        model.with(|found| {
            if found.capture == Some(BindingCapture { action, slot }) {
                "Press input…".to_owned()
            } else {
                found.controls.binding(action).0[slot]
                    .map_or_else(|| "Unbound".to_owned(), InputChord::label)
            }
        })
    };
    view! {
        row width:1fr height:30px gap:4px {
            Action label:(binding_semantics)
                on-click:(move || bind.ask(UiIntent::Pause(PauseAction::BeginBindingCapture(action, slot))))
                width:1fr height:30px {
                text #mechanic.caption text-wrap:none { label() }
            }
            Action label:(clear_semantics)
                on-click:(move || clear.ask(UiIntent::Pause(PauseAction::ClearBinding(action, slot))))
                width:28px height:30px {
                text #mechanic.caption "×"
            }
        }
    }
}

#[component]
fn PauseOptions(handles: Handles, fov: State<f32>) -> Element {
    let back = handles.clone();
    view! {
        col width:fill height:min-content gap:16px {
            text #mechanic.title "OPTIONS"
            text #mechanic.caption "Camera"
            row width:fill height:min-content align:center gap:14px {
                text #mechanic.label width:110px "FIELD OF VIEW"
                slider #mechanic.pause-slider width:1fr min:45 max:100 step:5 fov
                text #mechanic.value width:52px align:end {
                    format!("{:.0}°", $fov)
                }
            }
            Action label:"Back"
                on-click:(move || back.ask(UiIntent::Pause(PauseAction::Back)))
                width:fill height:42px {
                text #mechanic.value "Back"
            }
        }
    }
}

#[component]
fn PauseMain(handles: Handles, model: State<Model>) -> Element {
    let continue_action = handles.clone();
    let options_action = handles.clone();
    let controls_action = handles.clone();
    let exit_action = handles.clone();
    let cancel_exit = handles.clone();
    let confirm_exit = handles.clone();
    view! {
        col width:fill height:min-content gap:16px {
            text #mechanic.title "MENU"
            Action label:"Continue"
                on-click:(move || continue_action.ask(UiIntent::Pause(PauseAction::Continue)))
                width:fill height:42px {
                text #mechanic.value "Continue"
            }
            Action label:"Options"
                on-click:(move || options_action.ask(UiIntent::Pause(PauseAction::OpenOptions)))
                width:fill height:42px {
                text #mechanic.value "Options"
            }
            Action label:"Controls"
                on-click:(move || controls_action.ask(UiIntent::Pause(PauseAction::OpenControls)))
                width:fill height:42px {
                text #mechanic.value "Controls"
            }
            if model.with(|found| found.page == PausePage::ExitConfirmation) {
                col #mechanic.pause-confirm width:fill height:min-content gap:10px pad:14px {
                    text #mechanic.value "Unsaved construction changes"
                    text #mechanic.caption
                        "Exit without saving? This construction cannot be recovered."
                    row width:fill height:min-content gap:10px {
                        Action label:"Cancel"
                            on-click:({
                                let action = cancel_exit.clone();
                                move || action.ask(UiIntent::Pause(PauseAction::CancelExit))
                            })
                            width:1fr height:42px {
                            text "Cancel"
                        }
                        Action #mechanic.action-danger label:"Exit Without Saving"
                            on-click:({
                                let action = confirm_exit.clone();
                                move || action.ask(UiIntent::Pause(PauseAction::ExitWithoutSaving))
                            })
                            width:1fr height:42px {
                            text "Exit Without Saving"
                        }
                    }
                }
            } else {
                Action #mechanic.action-danger label:"Exit"
                    on-click:({
                        let action = exit_action.clone();
                        move || action.ask(UiIntent::Pause(PauseAction::Exit))
                    })
                    width:fill height:42px {
                    text #mechanic.value "Exit"
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Model;
    use crate::controls::Controls;
    use crate::pause_menu::PausePage;
    use crate::ui::{PauseAction, UiIntent, testing::Overlay};

    fn showing(page: PausePage) -> Overlay {
        let overlay = Overlay::mount();
        overlay.handles.pause.set(Model {
            open: true,
            page,
            camera_fov_degrees: 65.0,
            controls: Controls::default(),
            capture: None,
            vehicle_conflicts: Vec::new(),
        });
        overlay.handles.pause_fov.set(65.0);
        overlay.settle();
        overlay
    }

    #[test]
    fn the_modal_owns_every_pointer_position() {
        let overlay = showing(PausePage::Main);
        for point in [
            mosaic_core::Vector2::new(0.0, 0.0),
            mosaic_core::Vector2::new(800.0, 450.0),
            mosaic_core::Vector2::new(1599.0, 899.0),
        ] {
            assert!(overlay.wants_pointer_at(point));
        }
    }

    #[test]
    fn continue_and_exit_emit_their_intents() {
        let overlay = showing(PausePage::Main);
        let mut actions: Vec<_> = overlay
            .reachable_boxes()
            .into_iter()
            .filter(|rect| (rect.size.height - 42.0).abs() < 0.5 && rect.size.width > 450.0)
            .collect();
        actions.sort_by(|left, right| left.origin.y.total_cmp(&right.origin.y));
        overlay.click(actions[0].center());
        overlay.click(actions.last().expect("exit action").center());
        assert_eq!(
            overlay.intents(),
            vec![
                UiIntent::Pause(PauseAction::Continue),
                UiIntent::Pause(PauseAction::Exit),
            ]
        );
    }

    #[test]
    fn options_is_a_main_menu_action() {
        let overlay = showing(PausePage::Main);
        let mut actions: Vec<_> = overlay
            .reachable_boxes()
            .into_iter()
            .filter(|rect| (rect.size.height - 42.0).abs() < 0.5 && rect.size.width > 450.0)
            .collect();
        actions.sort_by(|left, right| left.origin.y.total_cmp(&right.origin.y));
        overlay.click(actions[1].center());
        assert_eq!(
            overlay.intents(),
            vec![UiIntent::Pause(PauseAction::OpenOptions)]
        );
    }

    #[test]
    fn fov_uses_native_slider_semantics_and_themed_parts() {
        let overlay = showing(PausePage::Options);
        assert_eq!(overlay.numeric_semantics(), Some((65.0, 45.0, 100.0, 5.0)));
        assert!(
            overlay.labels().iter().any(|label| label.contains("65°")),
            "the current degree value is exposed with the control"
        );
        let shapes = overlay.shapes();
        assert!(shapes.iter().any(|shape| {
            (shape.rect.size.height - 6.0).abs() < 0.5 && shape.rect.size.width > 100.0
        }));
        assert!(shapes.iter().any(|shape| {
            (shape.rect.size.width - 18.0).abs() < 0.5
                && (shape.rect.size.height - 18.0).abs() < 0.5
        }));
    }

    #[test]
    fn controls_page_exposes_two_slots_capture_and_conflicts() {
        let overlay = showing(PausePage::Controls);
        let mut model = overlay.handles.pause.get_untracked();
        model.capture = Some(crate::pause_menu::BindingCapture {
            action: crate::controls::GameAction::MoveForward,
            slot: 0,
        });
        model.vehicle_conflicts = vec![crate::controls::GameAction::MoveForward];
        overlay.handles.pause.set(model);
        overlay.settle();
        let labels = overlay.labels();
        assert!(labels.iter().any(|label| label == "Move Forward binding 1"));
        assert!(
            labels
                .iter()
                .any(|label| label == "Clear Move Forward binding 1")
        );
        assert!(labels.iter().any(|label| label.contains("Press input")));
        assert!(labels.iter().any(|label| label.contains("CONFLICT")));
        assert!(
            overlay
                .reachable_boxes()
                .iter()
                .any(|rect| (rect.size.height - 30.0).abs() < 0.5),
            "at least one binding chip is visible and reachable in the scroll viewport"
        );
    }
}
