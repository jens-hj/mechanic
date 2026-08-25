//! The creation picker: save what is built, open what was saved.
//!
//! A modal, and the one panel that is meant to cover the window — while it is
//! up the world behind it is not clickable, which is what a modal is. Every
//! decision it makes (does this name need confirming, has this delete been
//! asked for twice) stays in [`CreationMenuState`]; this only shows what that
//! state says and reports what was clicked.

#![allow(clippy::wildcard_imports)] // Mosaic's authoring vocabulary is meant to be globbed.

use std::path::PathBuf;

use bevy_mosaic::ui::*;
use mosaic_core::Effect;
use mosaic_core::theme::color;
use mosaic_macros::view;
use mosaic_widgets::input::{EventCtx, Key, KeyEvent, KeyEventKind};

#[allow(clippy::wildcard_imports)] // The design tokens are read as bare names.
use super::theme::*;
use super::{CreationsAction, Handles, UiIntent, display_font};
use crate::creation_menu::CreationMenuState;
use crate::showcase::CreationPreset;

/// How wide the sheet is.
pub(crate) const SHEET: f32 = 640.0;

/// One saved creation, as the list shows it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Row {
    /// What it is called.
    name: String,
    /// Where it lives.
    path: PathBuf,
    /// How much is in it.
    summary: String,
    /// Whether deleting it has been asked for once already.
    confirming: bool,
}

/// Everything the picker shows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Model {
    /// Whether it is up at all.
    pub(crate) open: bool,
    /// The name typed so far.
    name: String,
    /// Where the files are, shown so the standard place is never a mystery.
    directory: String,
    /// A one-line message, when there is something to say.
    notice: String,
    /// Whether saving would write over an existing creation.
    replacing: bool,
    /// What is on disk.
    rows: Vec<Row>,
}

/// Reads the picker's state into what it shows.
pub(crate) fn capture(menu: &CreationMenuState) -> Model {
    if !menu.is_open() {
        return Model::default();
    }
    Model {
        open: true,
        name: menu.name().to_owned(),
        directory: format!("Stored in {}", menu.directory().display()),
        notice: menu.notice().unwrap_or_default().to_owned(),
        replacing: menu.is_replacing(),
        rows: menu
            .entries()
            .iter()
            .map(|entry| Row {
                name: entry.name.clone(),
                path: entry.path.clone(),
                summary: format!(
                    "{} part{}, {} joint{}",
                    entry.part_count,
                    if entry.part_count == 1 { "" } else { "s" },
                    entry.joint_count,
                    if entry.joint_count == 1 { "" } else { "s" },
                ),
                confirming: menu.is_confirming_delete(&entry.path),
            })
            .collect(),
    }
}

/// The picker, over a veil that dims the world behind it.
///
/// The veil covers the window on purpose: while the picker is up, a click that
/// lands anywhere else is a click on the picker, not on the machine behind it.
pub(crate) fn view(handles: &Handles) -> Element {
    let model = handles.creations;
    let sheet = handles.clone();
    view! {
        col width:fill height:fill fill:picker.veil font-color:ink.fg {
            // A `scroll`'s attributes style its *content*, not its box — the
            // widget always fills its parent — so the sheet cannot be the
            // scroll and be centred too. The scroll is the area instead: its
            // content spans the veil, stands at least a windowful tall, and
            // centres the sheet inside that. A sheet taller than the window
            // makes the content taller than the viewport, which is what there
            // is to scroll.
            scroll width:fill min-height:100% align:center justify:center {
                (dialog(&sheet, model))
            }
        }
    }
}

/// The sheet itself.
fn dialog(handles: &Handles, model: State<Model>) -> Element {
    let rows = handles.clone();
    let presets = handles.clone();
    let cancel = handles.clone();
    let field = name_field(handles);
    let count = move || model.with(|found| found.rows.len());
    let empty = move || count() == 0;
    let notice = move || model.with(|found| found.notice.clone());
    let directory = move || model.with(|found| found.directory.clone());
    // The sheet hugs its contents rather than taking a share of anything: a
    // list given a share of a column that hugs has no share to take, and
    // collapses to nothing.
    view! {
        col width:{ Length::px(SHEET) } height:min-content
            pad:24px radius:14px fill:picker.sheet
            stroke:(width:1px color:picker.edge)
            shadow:(offset:(x:0px y:30px) blur:90px color:#00000099) {
            col width:fill height:min-content gap:10px {
                text font-family:{ display_font() } font-size:19px font-weight:700
                    letter-spacing:2.7px "CREATIONS"
                (field)
                if !notice().is_empty() {
                    text font-size:13px font-color:picker.notice { notice() }
                }
                (heading("YOUR CREATIONS"))
                text font-size:12px font-color:ink.dim { directory() }
                if empty() {
                    text font-size:14px font-color:ink.muted
                        "Nothing saved yet. Type a name above and press Enter."
                }
                for (index, ()) in { (0..count()).map(|index| (index, ())) } {
                    (saved_row(&rows, model, *index))
                }
                (heading("PRESETS"))
                for (scene, ()) in { CreationPreset::ALL.map(|scene| (scene, ())) } {
                    (preset_button(&presets, *scene))
                }
                col width:fill height:42px align:center justify:center
                    margin:(top:10px) radius:7px fill:picker.row
                    hover { fill:picker.row-over }
                    @click:{ cancel.ask(UiIntent::Creations(CreationsAction::Cancel)) } {
                    text font-size:15px "Cancel"
                }
            }
        }
    }
}

/// One of the two section headings.
fn heading(text: &'static str) -> Element {
    view! {
        text width:fill font-family:{ display_font() } font-size:12px font-weight:700
            letter-spacing:1.7px font-color:ink.dim margin:(top:6px) (text)
    }
}

/// The name field, and the button that saves under it.
fn name_field(handles: &Handles) -> Element {
    let handles = handles.clone();
    let model = handles.creations;
    let buffer: State<String> = State::new(String::new());
    let save = handles.clone();
    let typed = handles.clone();
    // The field is the one place the picker holds text of its own, so it is
    // seeded from the model whenever the model's name changes underneath it —
    // a cancel clears the name, and the field has to follow. An effect rather
    // than a binding: this writes state instead of describing an element.
    Effect::new(move || {
        let wanted = model.with(|found| found.name.clone());
        if buffer.get_untracked() != wanted {
            buffer.set(wanted);
        }
    });
    let label = move || {
        if model.with(|found| found.replacing) {
            "Replace"
        } else {
            "Save"
        }
    };
    view! {
        row width:fill height:min-content align:center gap:10px {
            text font-size:14px font-color:ink.muted "Save current as"
            row width:1fr height:34px align:center
                pad:(horizontal:10px vertical:0px) radius:6px fill:chip.fill
                stroke:(width:1px color:chip.edge)
                @key:{ move |event: &KeyEvent, ctx: &mut EventCtx| {
                    if !matches!(event.kind, KeyEventKind::Down { .. }) { return; }
                    match event.key {
                        Key::Enter => {
                            typed.ask(UiIntent::Creations(
                                CreationsAction::Name(buffer.get_untracked()),
                            ));
                            typed.ask(UiIntent::Creations(CreationsAction::Save));
                            ctx.stop_propagation();
                        }
                        Key::Escape => {
                            typed.ask(UiIntent::Creations(CreationsAction::Cancel));
                            ctx.stop_propagation();
                        }
                        _ => {}
                    }
                } } {
                input width:1fr font-size:15px fill:#00000000
                    pad:(horizontal:0px vertical:0px)
                    stroke:(width:0px color:#00000000) buffer
            }
            col width:96px height:34px align:center justify:center radius:6px
                fill:picker.row hover { fill:picker.row-over }
                @click:{
                    save.ask(UiIntent::Creations(CreationsAction::Name(buffer.get_untracked())));
                    save.ask(UiIntent::Creations(CreationsAction::Save));
                } {
                text font-size:14px { label().to_owned() }
            }
        }
    }
}

/// One saved creation: open it, or ask twice and delete it.
fn saved_row(handles: &Handles, model: State<Model>, index: usize) -> Element {
    let load = handles.clone();
    let remove = handles.clone();
    let row = move || model.with(|found| found.rows.get(index).cloned().unwrap_or_default());
    let confirming = move || row().confirming;
    view! {
        row width:fill height:52px align:center gap:8px {
            col width:1fr height:fill justify:center gap:3px
                pad:(horizontal:16px vertical:9px) radius:7px fill:picker.row
                stroke:(width:1px color:picker.row-edge)
                hover { fill:picker.row-over }
                @click:{ load.ask(UiIntent::Creations(CreationsAction::Load(row().path))) } {
                text font-family:{ display_font() } font-size:17px { row().name }
                text font-size:12px font-color:ink.muted { row().summary }
            }
            col width:{ if confirming() { Dimension::Px(96.0) } else { Dimension::Px(44.0) } }
                height:fill align:center justify:center radius:7px
                fill:{ if confirming() { color(picker.danger) } else { color(picker.row) } }
                font-color:{ if confirming() { color(accent.danger) } else { color(ink.muted) } }
                hover { fill:picker.danger-over }
                @click:{ remove.ask(UiIntent::Creations(CreationsAction::Delete(row().path))) } {
                text font-size:{ if confirming() { Length::px(13.0) } else { Length::px(18.0) } } {
                    if confirming() { "Delete?" } else { "×" }
                }
            }
        }
    }
}

/// One built-in scene.
fn preset_button(handles: &Handles, scene: CreationPreset) -> Element {
    let handles = handles.clone();
    view! {
        col width:fill height:min-content justify:center gap:3px
            pad:(horizontal:16px vertical:9px) radius:7px fill:picker.row
            stroke:(width:1px color:picker.row-edge)
            hover { fill:picker.row-over }
            @click:{ handles.ask(UiIntent::Creations(CreationsAction::Preset(scene))) } {
            text font-family:{ display_font() } font-size:16px (scene.label())
            text font-size:12px font-color:ink.muted (scene.description())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{Model, capture};
    use crate::creation_menu::CreationMenuState;
    use crate::creation_store::SavedCreation;
    use crate::showcase::CreationPreset;
    use crate::ui::testing::Overlay;
    use crate::ui::{CreationsAction, UiIntent};

    /// A picker showing two saved creations.
    fn stocked() -> Model {
        let mut state = CreationMenuState::default();
        state.open(
            vec![
                SavedCreation {
                    name: "Walker v3".to_owned(),
                    path: PathBuf::from("/creations/walker-v3.mech"),
                    part_count: 12,
                    joint_count: 4,
                },
                SavedCreation {
                    name: "Gearbox".to_owned(),
                    path: PathBuf::from("/creations/gearbox.mech"),
                    part_count: 1,
                    joint_count: 1,
                },
            ],
            String::new(),
            PathBuf::from("/creations"),
        );
        capture(&state)
    }

    /// The picker, up and laid out.
    fn showing() -> Overlay {
        let overlay = Overlay::mount();
        overlay.handles.creations.set(stocked());
        overlay.settle();
        overlay
    }

    #[test]
    fn a_row_summarises_what_is_in_the_creation() {
        let model = stocked();
        assert_eq!(model.rows[0].summary, "12 parts, 4 joints");
        assert_eq!(
            model.rows[1].summary, "1 part, 1 joint",
            "one of a thing is not plural",
        );
    }

    #[test]
    fn a_closed_picker_draws_nothing() {
        let overlay = Overlay::mount();
        let resting = overlay.element_count();
        overlay.handles.creations.set(stocked());
        overlay.settle();
        assert!(overlay.element_count() > resting, "the picker is up");

        overlay.handles.creations.set(Model::default());
        overlay.settle();
        assert_eq!(
            overlay.element_count(),
            resting,
            "and is put away entirely when it closes",
        );
    }

    #[test]
    fn clicking_a_row_asks_for_that_creation() {
        let overlay = showing();
        let row = overlay
            .reachable_boxes()
            .into_iter()
            .find(|rect| (rect.size.height - 52.0).abs() < 0.5 && rect.size.width > 400.0)
            .expect("a saved row is reachable");
        overlay.click(row.center());
        assert_eq!(
            overlay.intents(),
            vec![UiIntent::Creations(CreationsAction::Load(PathBuf::from(
                "/creations/walker-v3.mech"
            )))],
        );
    }

    #[test]
    fn clicking_a_preset_asks_for_that_scene() {
        let overlay = showing();
        let tree = overlay.rects();
        // The presets are the full-width buttons below the list, in the order
        // the scenes are declared.
        let sheet = tree
            .iter()
            .find(|(_, rect)| (rect.size.width - 640.0).abs() < 0.5)
            .expect("the sheet is laid out")
            .1;
        let preset = overlay
            .reachable_boxes()
            .into_iter()
            .filter(|rect| {
                (rect.size.width - (sheet.size.width - 48.0)).abs() < 0.5 && rect.size.height > 42.5
            })
            .min_by(|left, right| left.origin.y.total_cmp(&right.origin.y))
            .expect("a preset is reachable");
        overlay.click(preset.center());
        assert_eq!(
            overlay.intents(),
            vec![UiIntent::Creations(CreationsAction::Preset(
                CreationPreset::ALL[0]
            ))],
        );
    }

    #[test]
    fn the_delete_button_widens_once_it_has_been_asked() {
        let overlay = showing();
        let narrow = overlay
            .reachable_boxes()
            .into_iter()
            .filter(|rect| {
                (rect.size.width - 44.0).abs() < 0.5 && (rect.size.height - 52.0).abs() < 0.5
            })
            .count();
        assert_eq!(narrow, 2, "one delete button per row");

        overlay.handles.creations.update(|model| {
            model.rows[0].confirming = true;
        });
        overlay.settle();
        let asked = overlay
            .reachable_boxes()
            .into_iter()
            .filter(|rect| {
                (rect.size.width - 96.0).abs() < 0.5 && (rect.size.height - 52.0).abs() < 0.5
            })
            .count();
        assert_eq!(asked, 1, "only the row that was asked about widens");
    }

    #[test]
    fn clicking_delete_asks_for_that_file() {
        let overlay = showing();
        let delete = overlay
            .reachable_boxes()
            .into_iter()
            .filter(|rect| {
                (rect.size.width - 44.0).abs() < 0.5 && (rect.size.height - 52.0).abs() < 0.5
            })
            .min_by(|left, right| left.origin.y.total_cmp(&right.origin.y))
            .expect("the delete button is reachable");
        overlay.click(delete.center());
        assert_eq!(
            overlay.intents(),
            vec![UiIntent::Creations(CreationsAction::Delete(PathBuf::from(
                "/creations/walker-v3.mech"
            )))],
        );
    }

    #[test]
    fn cancelling_backs_out() {
        let overlay = showing();
        let cancel = overlay
            .reachable_boxes()
            .into_iter()
            .find(|rect| (rect.size.height - 42.0).abs() < 0.5)
            .expect("the cancel button is reachable");
        overlay.click(cancel.center());
        assert_eq!(
            overlay.intents(),
            vec![UiIntent::Creations(CreationsAction::Cancel)],
        );
    }
}
