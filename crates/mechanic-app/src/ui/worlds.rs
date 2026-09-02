//! Full-window Mosaic world list and creation form.

#![allow(clippy::wildcard_imports)]

use std::path::PathBuf;

use bevy_mosaic::ui::*;
use mosaic_core::theme::color;
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
    resolved: usize,
    total: usize,
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
        resolved: state.loading_progress().resolved,
        total: state.loading_progress().total,
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
    let console = handles.clone();
    let progress = move || model.with(|found| found.progress.clone());
    let progress_width = move || model.with(|found| loading_bar_width(found.resolved, found.total));
    let count = move || model.with(|found| found.rows.len());
    view! {
        col width:fill height:fill fill:picker.screen align:center font-color:ink.fg
            pad:(horizontal:32px vertical:24px) {
            col width:fill max-width:1320px height:fill gap:18px {
                row width:fill height:72px shrink:0 align:center {
                    col width:1fr height:min-content gap:3px {
                        text #mechanic.section font-color:accent.key "MECHANIC // WORLD GATE"
                        text #mechanic.title font-size:text-size.hero "DIMENSION ROUTING CONSOLE"
                    }
                    col width:min-content height:min-content align:end gap:4px {
                        row width:min-content height:min-content gap:7px align:center {
                            el width:8px height:8px radius:2px exponent:1 fill:status-color.good {}
                            text #mechanic.label font-color:status-color.good "SYSTEM READY"
                        }
                        text #mechanic.caption { format!("{:02} LOCAL WORLDS", count()) }
                    }
                }

                el width:fill height:1px shrink:0 fill:shell-rule {}

                if model.with(|found| found.loading) {
                    PanelSurface elevated:true width:fill height:1fr pad:32px fill:picker.sheet {
                        col width:fill height:fill align:center justify:center gap:18px {
                            row width:min-content height:min-content align:center gap:10px {
                                el width:12px height:12px radius:3px exponent:1 fill:accent.speed {}
                                text #mechanic.section font-color:accent.speed
                                    "DIMENSION LINK // SYNCHRONIZING"
                            }
                            text #mechanic.title font-size:text-size.hero "ROUTING WORLD"
                            text #mechanic.value { progress() }
                            el width:320px height:4px radius:2px exponent:1 fill:dial.track {
                                el width:{ progress_width() } height:fill radius:2px exponent:1
                                    fill:accent.speed {}
                            }
                            text #mechanic.caption
                                "Preparing collision, visible terrain, and foundation index within 16 m…"
                        }
                    }
                } else {
                    WorldConsole handles:(console.clone()) model:(model) name:(name) seed:(seed)
                }

                row width:fill height:24px shrink:0 align:center {
                    text #mechanic.caption width:1fr "STORAGE // LOCAL"
                    text #mechanic.caption "TERRAIN STREAM // OCTREE   ·   GARAGE LINK // STANDBY"
                }
            }
        }
    }
}

#[component]
fn WorldConsole(
    handles: Handles,
    model: State<Model>,
    name: State<String>,
    seed: State<String>,
) -> Element {
    let create_key = handles.clone();
    let create_button = handles.clone();
    let rows = handles.clone();
    let notice = move || model.with(|found| found.notice.clone());
    let count = move || model.with(|found| found.rows.len());
    view! {
        row width:fill height:1fr gap:18px {
            PanelSurface elevated:true width:1fr height:fill pad:20px fill:picker.sheet {
                col width:fill height:fill gap:14px {
                    row width:fill height:min-content align:center {
                        col width:1fr height:min-content gap:3px {
                            text #mechanic.section "LOCAL WORLD ARCHIVE"
                            text #mechanic.caption
                                "Select a terrain instance and establish a Garage link."
                        }
                        text #mechanic.label font-color:accent.speed {
                            format!("INDEX // {:02}", count())
                        }
                    }
                    el width:fill height:1px shrink:0 fill:shell-rule {}
                    scroll width:fill height:1fr gap:8px pad:(right:8px) {
                        if count() == 0 {
                            col width:fill min-height:420px align:center justify:center gap:8px {
                                text #mechanic.section font-color:ink.muted "ARCHIVE EMPTY"
                                text #mechanic.caption "Provision the first world from the console."
                            }
                        }
                        for (index, ()) in { (0..count()).map(|index| (index, ())) } {
                            (world_row(&rows, model, *index))
                        }
                    }
                }
            }

            PanelSurface elevated:true width:390px height:fill pad:22px fill:picker.sheet {
                col width:fill height:fill gap:14px {
                    text #mechanic.section font-color:accent.key "PROVISION NEW WORLD"
                    text #mechanic.caption
                        "Initialize a procedural terrain volume and bind it to the local archive."
                    el width:fill height:1px shrink:0 fill:shell-rule {}

                    col width:fill height:min-content gap:6px {
                        text #mechanic.label "WORLD DESIGNATION"
                        input #mechanic.field width:fill height:40px name
                        text #mechanic.caption "Required // stored as the archive identity"
                    }

                    col width:fill height:min-content gap:6px {
                        text #mechanic.label "GENERATION SEED"
                        input #mechanic.field width:fill height:40px
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
                        text #mechanic.caption "Optional U64 // blank requests OS entropy"
                    }

                    if !notice().is_empty() {
                        col width:fill height:min-content pad:10px radius:6px exponent:1
                            fill:wash.angle stroke:(width:1px color:picker.notice) {
                            text #mechanic.label font-color:picker.notice "SYSTEM NOTICE"
                            text #mechanic.caption font-color:picker.notice { notice() }
                        }
                    }

                    el width:fill height:1fr {}
                    col width:fill height:min-content gap:7px pad:12px radius:8px exponent:1
                        fill:control.rest stroke:(width:1px color:chip.edge) {
                        text #mechanic.label font-color:ink.dim "LINK PROTOCOL"
                        text #mechanic.caption "01 // Generate deterministic terrain"
                        text #mechanic.caption "02 // Build local collision field"
                        text #mechanic.caption "03 // Transfer from Garage"
                    }
                    Action label:"Generate world"
                        on-click:(move || create_button.ask(UiIntent::Worlds(WorldAction::Create {
                            name: name.get_untracked(),
                            seed: seed.get_untracked(),
                        })))
                        width:fill height:44px fill:wash.key
                        stroke:(width:1px color:accent.key) {
                        text #mechanic.value font-color:accent.key "GENERATE INSTANCE"
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
        row width:fill height:72px shrink:0 align:center gap:8px {
            Action label:"Open world"
                on-click:(move || open.ask(UiIntent::Worlds(WorldAction::Open(row().path))))
                width:1fr height:fill pad:(horizontal:12px vertical:8px) align:start justify:center {
                row width:fill height:fill align:center gap:12px {
                    col width:54px height:fill align:center justify:center radius:7px exponent:1
                        fill:wash.key stroke:(width:1px color:chip.edge) {
                        text #mechanic.label font-color:accent.key { format!("W-{:02}", index + 1) }
                    }
                    col width:1fr height:min-content gap:4px {
                        text #mechanic.value { row().name }
                        text #mechanic.caption { format!("SEED // {}   ·   {}", row().seed, row().last_played) }
                    }
                    text font-size:text-size.tiny letter-spacing:text-tracking.label
                        font-color:{ row_status_color(&row().status) } { row().status }
                }
            }
            Action label:"Delete world"
                on-click:(move || remove.ask(UiIntent::Worlds(WorldAction::Delete(row().path))))
                width:104px height:fill {
                col width:fill height:min-content align:center gap:3px {
                    text #mechanic.label font-color:accent.danger {
                        if row().confirming_delete { "CONFIRM?" } else { "DELETE" }
                    }
                    text #mechanic.caption "LOCAL DATA"
                }
            }
        }
    }
}

fn row_status_color(status: &str) -> Color {
    if status.starts_with("CORRUPT") {
        color(accent.danger)
    } else if status.starts_with("OUTDATED") {
        color(status_color.warn)
    } else {
        color(status_color.good)
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "terrain startup counts are tiny compared with f32's exact integer range"
)]
fn loading_bar_width(resolved: usize, total: usize) -> Length {
    let fraction = if total == 0 {
        0.0
    } else {
        resolved as f32 / total as f32
    };
    Length::px(320.0 * fraction.clamp(0.0, 1.0))
}
