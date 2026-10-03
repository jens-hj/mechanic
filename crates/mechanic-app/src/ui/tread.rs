//! The Matter Manipulator's tread brush: a status card while cutting, and a
//! workbench for choosing a pattern, drawing a custom tile, and setting depth.

#![allow(
    clippy::wildcard_imports,
    reason = "Mosaic's authoring vocabulary is meant to be globbed"
)]

use bevy_mosaic::ui::*;
use mechanic_core::{MAX_TREAD_DEPTH_MM, MIN_TREAD_DEPTH_MM, TREAD_TILE_CELLS, TreadPattern};
use mosaic_core::{Effect, theme::color};
use mosaic_macros::{component, view};

use super::Handles;
use super::components::{PanelSurface, PanelSurfaceProps};
use super::styles::*;
use super::theme::*;
use crate::controls::GameAction;
use crate::tread::{TreadBrush, response_summary, tread_label};

const STATUS_WIDTH: f32 = 260.0;
const RIGHT: f32 = 18.0;
const HOTBAR_CLEARANCE: f32 = 178.0;

/// Every cell of a tile, row by row, for laying the tile out as a grid.
#[expect(clippy::cast_possible_truncation, reason = "64 cells")]
const CELLS: [(u32, ()); 64] = {
    let mut cells = [(0, ()); 64];
    let mut index = 0;
    while index < 64 {
        cells[index].0 = index as u32;
        index += 1;
    }
    cells
};

/// Every pattern a button picks, the built-ins then the hand-drawn tile.
const PATTERNS: [(usize, ()); 6] = [(0, ()), (1, ()), (2, ()), (3, ()), (4, ()), (5, ())];

fn pattern_at(index: usize) -> TreadPattern {
    TreadPattern::BUILT_IN
        .get(index)
        .copied()
        .unwrap_or_else(|| TreadPattern::Custom(TreadBrush::default().custom))
}

/// Whether a cell of the brush's tile is a raised lug.
fn raised(brush: State<TreadBrush>, cell: u32) -> bool {
    brush
        .get()
        .tread
        .pattern()
        .mask()
        .raised(cell % TREAD_TILE_CELLS, cell / TREAD_TILE_CELLS)
}

/// Read-only brush state that remains visible while gameplay owns the mouse.
#[component]
pub(crate) fn TreadStatus(handles: Handles) -> Element {
    let brush = handles.tread;
    let controls = handles.controls;
    let viewport = handles.viewport;
    let size = State::new(Size::ZERO);
    let at = move || {
        let window = viewport.get();
        let own = size.get();
        (
            Length::px((window.width - own.width - RIGHT).max(0.0)),
            Length::px((window.height - own.height - HOTBAR_CLEARANCE).max(0.0)),
        )
    };

    view! {
        col #mechanic.panel #mechanic.elevated nohit width:{ Length::px(STATUS_WIDTH) }
            height:min-content gap:8px pad:12px translate:(x:{ at().0 } y:{ at().1 })
            @layout:{ move |bounds: Rect| {
                if size.get_untracked() != bounds.size {
                    size.set(bounds.size);
                }
            } } {
            row width:fill height:min-content align:center gap:10px nohit {
                grid width:44px height:44px shrink:0 nohit
                    cols:{ GridTracks::repeat(8, [GridTrack::fr(1.0)]) }
                    rows:{ GridTracks::repeat(8, [GridTrack::fr(1.0)]) }
                    gap:1px {
                    for (cell, ()) in { CELLS } {
                        (status_cell(brush, *cell))
                    }
                }
                col width:1fr height:min-content gap:3px nohit {
                    text #mechanic.section "TREAD"
                    text #mechanic.caption { tread_label(brush.get().tread) }
                }
            }
            text #mechanic.caption { response_summary(brush.get().tread) }
            text #mechanic.caption font-color:accent.key {
                format!(
                    "Press {} to configure",
                    controls.with(|bindings| bindings.label(GameAction::MaterialWheel)),
                )
            }
            text #mechanic.caption "L-drag Cut  ·  Q Sample  ·  Right-drag Smooth"
        }
    }
}

fn status_cell(brush: State<TreadBrush>, cell: u32) -> Element {
    view! {
        el width:fill height:fill nohit
            fill:{ if raised(brush, cell) { color(accent.speed) } else { color(lane.fill) } } {}
    }
}

#[component]
pub(crate) fn TreadPanel(handles: Handles) -> Element {
    let brush = handles.tread;
    let depth = State::new(f32::from(brush.get_untracked().tread.depth_mm()));
    // A sampled tread moves the slider; the slider moves the brush.
    Effect::new(move || {
        let wanted = f32::from(brush.get().tread.depth_mm());
        if (depth.get_untracked() - wanted).abs() > f32::EPSILON {
            depth.set(wanted);
        }
    });
    Effect::new(move || {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the slider spans a few dozen millimetres"
        )]
        let millimetres = depth.get().round().max(0.0) as u8;
        if brush.get_untracked().tread.depth_mm() != millimetres {
            brush.update(|current| *current = current.with_depth(millimetres));
        }
    });
    let controls = handles.controls;

    view! {
        col #mechanic.pause-veil width:fill height:fill align:center justify:center {
            PanelSurface elevated:true width:fill height:fill margin:36px
                shadow:(mosaic_core::theme::typed(
                    modal_shadow,
                    || ShadowSpec::new(Vector2::ZERO, 0.0, 0.0, Color::TRANSPARENT),
                )) {
                row width:fill height:86px shrink:0 align:center justify:between
                    pad:(horizontal:22px vertical:14px)
                    stroke:(width:1px color:shell-rule edges:bottom) {
                    col width:1fr height:min-content gap:2px {
                        text #mechanic.title text-wrap:none "TREAD WORKBENCH"
                        text #mechanic.caption text-wrap:none
                            "Choose the pattern and depth the Matter Manipulator cuts into surfaces"
                    }
                }

                row width:fill height:1fr gap:16px pad:20px {
                    col width:240px height:fill shrink:0 gap:10px pad:18px radius:12px exponent:1
                        fill:lane.fill stroke:(width:1px color:lane.edge) {
                        text #mechanic.section "PATTERN"
                        for (index, ()) in { PATTERNS } {
                            (pattern_button(brush, pattern_at(*index)))
                        }
                    }

                    col width:1fr height:fill gap:14px pad:18px radius:12px exponent:1
                        fill:lane.fill stroke:(width:1px color:lane.edge) {
                        col width:fill height:min-content gap:3px {
                            text #mechanic.section "TILE"
                            text #mechanic.caption
                                "One block square, rolling left to right. Click a cell to raise or cut it; drawing on a built-in pattern starts a custom copy."
                        }
                        row width:fill height:1fr align:center justify:center {
                            grid width:368px height:368px shrink:0
                                cols:{ GridTracks::repeat(8, [GridTrack::fr(1.0)]) }
                                rows:{ GridTracks::repeat(8, [GridTrack::fr(1.0)]) }
                                gap:6px {
                                for (cell, ()) in { CELLS } {
                                    (tile_cell(brush, *cell))
                                }
                            }
                        }
                        (depth_slider(depth))
                    }

                    col width:320px height:fill shrink:0 gap:14px {
                        col width:fill height:1fr align:center justify:center gap:10px
                            pad:16px radius:12px exponent:1 fill:card.fill
                            stroke:(width:1px color:card.edge-on) {
                            text #mechanic.section "CURRENT BRUSH"
                            text #mechanic.value text-wrap:none { tread_label(brush.get().tread) }
                            (effect_line("Contact", move || {
                                format!("{:.0}%", brush.get().tread.response().contact_ratio * 100.0)
                            }))
                            (effect_line("Grip on soil and sand", move || {
                                format!("×{:.2}", brush.get().tread.response().yielding_grip())
                            }))
                            (effect_line("Grip on rock", move || {
                                format!("×{:.2}", brush.get().tread.response().firm_grip())
                            }))
                            (effect_line("Ground pressure", move || {
                                format!("×{:.1}", brush.get().tread.response().pressure_factor())
                            }))
                            text #mechanic.caption
                                "Deeper lugs bite harder into ground that gives; fewer lugs press harder and sink further."
                        }
                    }
                }

                row width:fill height:46px shrink:0 align:center justify:between
                    pad:(horizontal:22px vertical:0px)
                    stroke:(width:1px color:shell-rule edges:top) {
                    text #mechanic.caption "L-drag  Cut   ·   Q  Sample   ·   Right-drag  Smooth"
                    text #mechanic.caption text-wrap:none font-color:accent.key {
                        format!(
                            "{}  Close workbench",
                            controls.with(|bindings| bindings.label(GameAction::MaterialWheel)),
                        )
                    }
                }
            }
        }
    }
}

fn pattern_button(brush: State<TreadBrush>, pattern: TreadPattern) -> Element {
    let active = move || match (brush.get().tread.pattern(), pattern) {
        (TreadPattern::Custom(_), TreadPattern::Custom(_)) => true,
        (current, wanted) => current == wanted,
    };
    view! {
        button #mechanic.action width:fill height:42px shrink:0
            fill:{ if active() { color(control.pressed) } else { color(control.rest) } }
            stroke:(width:1px color:{ if active() { color(accent.key) } else { color(chip.edge) } })
            @click:{ brush.update(|current| *current = current.with_pattern(pattern)); }
            { pattern.label() }
    }
}

fn tile_cell(brush: State<TreadBrush>, cell: u32) -> Element {
    view! {
        el width:fill height:fill radius:4px exponent:1
            fill:{ if raised(brush, cell) { color(accent.speed) } else { color(card.fill) } }
            stroke:(width:1px color:chip.edge)
            @click:{
                brush.update(|current| {
                    *current = current
                        .with_cell_toggled(cell % TREAD_TILE_CELLS, cell / TREAD_TILE_CELLS);
                });
            } {}
    }
}

fn depth_slider(depth: State<f32>) -> Element {
    view! {
        row width:fill height:44px shrink:0 align:center gap:10px {
            text #mechanic.label width:106px shrink:0 "DEPTH"
            {
                let slider = slider_styled(
                    parent,
                    depth,
                    f32::from(MIN_TREAD_DEPTH_MM)..=f32::from(MAX_TREAD_DEPTH_MM),
                    Some(1.0),
                    SliderStyle {
                        track: color(dial.track),
                        fill: color(accent.key),
                        thumb: color(ink.fg),
                        thumb_hover: color(accent.key),
                        focus: color(control.focus),
                        track_height: 5.0,
                        thumb_size: 14.0,
                    },
                );
                slider.root().restyle(|style| style.grow(1.0).basis(0.0));
            }
            text #mechanic.value width:62px shrink:0 align:end {
                format!("{:.0} mm", depth.get())
            }
        }
    }
}

fn effect_line(label: &'static str, value: impl Fn() -> String + 'static) -> Element {
    view! {
        row width:fill height:min-content align:center justify:between {
            text #mechanic.label text-wrap:none { label }
            text #mechanic.value text-wrap:none { value() }
        }
    }
}
