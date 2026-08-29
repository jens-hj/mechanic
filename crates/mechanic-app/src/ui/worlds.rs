//! Full-window Mosaic world list and creation form.

#![allow(clippy::wildcard_imports)]

use std::path::PathBuf;

use bevy_mosaic::ui::*;
use mosaic_macros::{component, view};
use mosaic_widgets::input::{EventCtx, Key, KeyEvent, KeyEventKind};

use super::components::{Action, ActionProps, PanelSurface, PanelSurfaceProps};
#[allow(unused_imports)]
use super::styles::*;
#[allow(unused_imports, clippy::wildcard_imports)]
use super::theme::*;
use super::{Handles, UiIntent, WorldAction};
use crate::world::{WorldListPhase, WorldListState};
use mechanic_world::SavedWorldStatus;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Row {
    name: String,
    seed: String,
    last_played: String,
    status: String,
    path: PathBuf,
    confirming_delete: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Model {
    pub(crate) open: bool,
    loading: bool,
    progress: String,
    notice: String,
    rows: Vec<Row>,
}

pub(crate) fn capture(state: &WorldListState) -> Model {
    Model {
        open: state.is_open(),
        loading: state.phase() == WorldListPhase::Loading,
        progress: {
            let progress = state.loading_progress();
            format!("Terrain nodes {} / {}", progress.resolved, progress.total)
        },
        notice: state.notice().unwrap_or_default().to_owned(),
        rows: state
            .entries()
            .iter()
            .map(|entry| Row {
                name: entry
                    .name
                    .clone()
                    .unwrap_or_else(|| "Unreadable world".to_owned()),
                seed: entry
                    .seed
                    .map_or_else(|| "—".to_owned(), |seed| seed.0.to_string()),
                last_played: entry.last_played_unix_seconds.map_or_else(
                    || "Unknown".to_owned(),
                    |value| format!("Last played {value}"),
                ),
                status: match &entry.status {
                    SavedWorldStatus::Current => "CURRENT".to_owned(),
                    SavedWorldStatus::Outdated => "OUTDATED — OPENING REMOVES IT".to_owned(),
                    SavedWorldStatus::Corrupt { file, .. } => {
                        format!("CORRUPT — PRESERVED: {}", file.display())
                    }
                },
                path: entry.path.clone(),
                confirming_delete: state.is_confirming_delete(&entry.path),
            })
            .collect(),
    }
}

#[component]
pub(crate) fn WorldList(handles: Handles) -> Element {
    let model = handles.worlds;
    let name: State<String> = State::new(String::new());
    let seed: State<String> = State::new(String::new());
    let create_key = handles.clone();
    let create_button = handles.clone();
    let rows = handles.clone();
    let notice = move || model.with(|found| found.notice.clone());
    let title = move || {
        model.with(|found| {
            if found.loading {
                "LOADING WORLD".to_owned()
            } else {
                "WORLDS".to_owned()
            }
        })
    };
    let progress = move || model.with(|found| found.progress.clone());
    let count = move || model.with(|found| found.rows.len());
    view! {
        col width:fill height:fill fill:picker.screen align:center justify:center font-color:ink.fg {
            PanelSurface elevated:true width:720px height:min-content pad:24px fill:picker.sheet {
                col width:fill height:min-content gap:12px {
                    text #mechanic.title { title() }
                    if model.with(|found| found.loading) {
                        text #mechanic.value { progress() }
                        text #mechanic.caption "Preparing collision and visible terrain within 16 m…"
                    }
                    row width:fill height:min-content gap:8px align:center {
                        input #mechanic.field width:1fr height:36px name
                        input #mechanic.field width:180px height:36px
                            @key:{ move |event: &KeyEvent, ctx: &mut EventCtx| {
                                if matches!(event.kind, KeyEventKind::Down { .. }) && event.key == Key::Enter {
                                    create_key.ask(UiIntent::Worlds(WorldAction::Create {
                                        name: name.get_untracked(),
                                        seed: seed.get_untracked(),
                                    }));
                                    ctx.stop_propagation();
                                }
                            } }
                            seed
                        Action label:"Create world"
                            on-click:(move || create_button.ask(UiIntent::Worlds(WorldAction::Create {
                                name: name.get_untracked(),
                                seed: seed.get_untracked(),
                            })))
                            width:110px height:36px {
                            text #mechanic.value "Create"
                        }
                    }
                    text #mechanic.caption "Name · optional numeric seed (blank uses OS randomness)"
                    if !notice().is_empty() {
                        text font-color:picker.notice { notice() }
                    }
                    if count() == 0 {
                        text #mechanic.caption "No worlds yet."
                    }
                    for (index, ()) in { (0..count()).map(|index| (index, ())) } {
                        (world_row(&rows, model, *index))
                    }
                }
            }
        }
    }
}

fn world_row(handles: &Handles, model: State<Model>, index: usize) -> Element {
    let open = handles.clone();
    let remove = handles.clone();
    let row = move || model.with(|found| found.rows.get(index).cloned().unwrap_or_default());
    view! {
        row #mechanic.list-row width:fill height:58px align:center gap:8px pad:8px {
            Action label:"Open world"
                on-click:(move || open.ask(UiIntent::Worlds(WorldAction::Open(row().path))))
                width:1fr height:fill align:start justify:center {
                text #mechanic.value { row().name }
                text #mechanic.caption { format!("Seed {} · {} · {}", row().seed, row().last_played, row().status) }
            }
            Action label:"Delete world"
                on-click:(move || remove.ask(UiIntent::Worlds(WorldAction::Delete(row().path))))
                width:92px height:fill {
                text font-color:accent.danger {
                    if row().confirming_delete { "Delete?" } else { "Delete" }
                }
            }
        }
    }
}
