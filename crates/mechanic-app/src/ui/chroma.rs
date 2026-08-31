//! Compact controls for the Matter Manipulator's session-wide appearance brush.

#![allow(clippy::wildcard_imports)] // Mosaic's authoring vocabulary is meant to be globbed.

use std::{cell::Cell, rc::Rc};

use bevy_mosaic::ui::*;
use mechanic_core::{
    MaterialAppearance, MaterialColor, MaterialDye, MaterialFinish, MaterialShift,
};
use mosaic_core::{Effect, theme::color};
use mosaic_macros::{component, view};

use super::Handles;
use super::components::{PanelSurface, PanelSurfaceProps};
#[allow(unused_imports)] // Style constants are consumed by `view!` expansion.
use super::styles::*;
#[allow(unused_imports, clippy::wildcard_imports)]
use super::theme::*;
use crate::chroma::representative_srgb;
use crate::controls::GameAction;

const STATUS_WIDTH: f32 = 260.0;
const RIGHT: f32 = 18.0;
const HOTBAR_CLEARANCE: f32 = 178.0;
const PRESETS: [[u8; 3]; 12] = [
    [0xF4, 0xF4, 0xF2],
    [0xC8, 0xCC, 0xD0],
    [0x14, 0x16, 0x1A],
    [0xE0, 0x56, 0x1F],
    [0xE8, 0xB3, 0x2A],
    [0x8F, 0xC9, 0x3A],
    [0x2E, 0x7D, 0x46],
    [0x22, 0xA6, 0xB3],
    [0x2A, 0x4C, 0xC7],
    [0x7A, 0x3F, 0xBF],
    [0xC0, 0x38, 0x7E],
    [0xB4, 0x23, 0x2A],
];

#[derive(Clone, Copy)]
enum ColorMode {
    Baked,
    Shift,
    Dye,
}

#[derive(Clone, Copy)]
struct ColorStates {
    hue: State<f32>,
    chroma: State<f32>,
    lightness: State<f32>,
    dye: State<Color>,
    structure: State<f32>,
}

/// Read-only brush state that remains visible while gameplay owns the mouse.
#[component]
pub(crate) fn ChromaStatus(handles: Handles) -> Element {
    let appearance = handles.chroma;
    let material = handles.material;
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
                el width:44px height:44px radius:7px
                    fill:{ rgb_color(representative_srgb(material.get(), appearance.get())) }
                    stroke:(width:1px color:chip.edge) nohit {}
                col width:1fr height:min-content gap:3px nohit {
                    text #mechanic.section "CHROMA"
                    text #mechanic.caption {
                        format!("{}  ·  {}", color_summary(appearance.get()), finish_label(appearance.get().finish))
                    }
                }
            }
            text #mechanic.caption font-color:accent.key {
                format!(
                    "Press {} to configure",
                    controls.with(|bindings| bindings.label(GameAction::MaterialWheel)),
                )
            }
            text #mechanic.caption "L-drag Paint  ·  Q Sample  ·  Right-drag Remove"
        }
    }
}

#[component]
pub(crate) fn ChromaPanel(handles: Handles) -> Element {
    let appearance = handles.chroma;
    let initial = appearance.get_untracked();
    let initial_shift = match initial.color {
        MaterialColor::Shift(shift) => shift,
        MaterialColor::Baked | MaterialColor::Dye(_) => default_shift(),
    };
    let initial_dye = match initial.color {
        MaterialColor::Dye(dye) => dye,
        MaterialColor::Baked | MaterialColor::Shift(_) => default_dye(),
    };
    let hue = State::new(initial_shift.hue_degrees());
    let chroma = State::new(initial_shift.chroma());
    let lightness = State::new(initial_shift.lightness());
    let structure = State::new(initial_dye.structure());
    let dye_color = State::new(rgb_color(initial_dye.target_rgb()));
    let hex = State::new(format_hex(initial_dye.target_rgb()));
    let last_hex_target = Rc::new(Cell::new(initial_dye.target_rgb()));
    let color_states = ColorStates {
        hue,
        chroma,
        lightness,
        dye: dye_color,
        structure,
    };

    // Sampling and tool-state changes flow back into the native controls.
    let synced_hex_target = Rc::clone(&last_hex_target);
    Effect::new(move || match appearance.get() {
        MaterialAppearance {
            color: MaterialColor::Shift(shift),
            ..
        } => {
            set_f32_if_changed(hue, shift.hue_degrees());
            set_f32_if_changed(chroma, shift.chroma());
            set_f32_if_changed(lightness, shift.lightness());
        }
        MaterialAppearance {
            color: MaterialColor::Dye(dye),
            ..
        } => {
            set_f32_if_changed(structure, dye.structure());
            let synced_color = rgb_color(dye.target_rgb());
            if dye_color.get_untracked() != synced_color {
                dye_color.set(synced_color);
            }
            if synced_hex_target.get() != dye.target_rgb() {
                hex.set(format_hex(dye.target_rgb()));
                synced_hex_target.set(dye.target_rgb());
            }
        }
        MaterialAppearance {
            color: MaterialColor::Baked,
            ..
        } => {}
    });

    Effect::new(move || {
        if matches!(appearance.get().color, MaterialColor::Shift(_)) {
            set_shift(appearance, hue.get(), chroma.get(), lightness.get());
        }
    });
    Effect::new(move || {
        if matches!(appearance.get().color, MaterialColor::Dye(_)) {
            let [r, g, b, _] = dye_color.get().to_srgb8();
            set_dye(appearance, [r, g, b], structure.get());
        }
    });
    {
        let last_hex_target = Rc::clone(&last_hex_target);
        Effect::new(move || {
            if let Some(target) = parse_hex(&hex.get()) {
                last_hex_target.set(target);
                let parsed = rgb_color(target);
                if dye_color.get_untracked() != parsed {
                    dye_color.set(parsed);
                }
            }
        });
    }

    let baked = handles.clone();
    let shifted = handles.clone();
    let dyed = handles.clone();
    let baked_finish = handles.clone();
    let anodised_finish = handles.clone();
    let painted_finish = handles.clone();
    let restore = handles.clone();
    let swatch_material = handles.material;
    let controls = handles.controls;
    let dye_field_height = (handles.viewport.get_untracked().height - 440.0).clamp(240.0, 760.0);

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
                    row width:1fr height:min-content align:center gap:14px {
                        el width:50px height:50px shrink:0 radius:10px
                            fill:{ rgb_color(representative_srgb(swatch_material.get(), appearance.get())) }
                            stroke:(width:1px color:accent.key) {}
                        col width:1fr height:min-content gap:2px {
                            text #mechanic.title text-wrap:none "CHROMA WORKBENCH"
                            text #mechanic.caption text-wrap:none
                                "Configure the global appearance brush · live preview on the right"
                        }
                    }
                    row width:min-content height:min-content shrink:0 align:center {
                        button #mechanic.action width:132px height:38px shrink:0 text-wrap:none
                            @click:{ restore.chroma.set(MaterialAppearance::BAKED); }
                            "Restore all"
                    }
                }

                row width:fill height:1fr gap:16px pad:20px {
                    col width:1fr height:fill gap:14px pad:18px radius:12px
                        fill:lane.fill stroke:(width:1px color:lane.edge) {
                        col width:fill height:min-content gap:3px {
                            text #mechanic.section "COLOR TREATMENT"
                            text #mechanic.caption
                                "Keep the source, shift its character, or dye toward an exact color."
                        }
                        row width:fill height:42px gap:8px shrink:0 {
                            (mode_button(&baked, ColorMode::Baked, "Original", color_states))
                            (mode_button(&shifted, ColorMode::Shift, "Shift", color_states))
                            (mode_button(&dyed, ColorMode::Dye, "Dye", color_states))
                        }

                        if matches!(appearance.get().color, MaterialColor::Baked) {
                            col width:fill height:1fr align:center justify:center gap:8px
                                radius:10px fill:card.fill stroke:(width:1px color:card.edge) {
                                text #mechanic.value "SOURCE APPEARANCE"
                                text #mechanic.caption
                                    "Uses the material's authored color and texture without recoloring."
                            }
                        }

                        if matches!(appearance.get().color, MaterialColor::Shift(_)) {
                            col width:fill height:1fr gap:12px pad:(top:8px) {
                                (value_slider("HUE ROTATION", hue, -180.0, 180.0, 1.0, "°"))
                                (value_slider("CHROMA", chroma, 0.0, 1.8, 0.01, "×"))
                                (value_slider("LIGHTNESS", lightness, 0.0, 2.0, 0.01, "×"))
                                text #mechanic.caption pad:(top:6px)
                                    "Shift preserves the material's baked variation and protected detail."
                            }
                        }

                        if matches!(appearance.get().color, MaterialColor::Dye(_)) {
                            row width:fill height:1fr gap:18px pad:(top:4px) {
                                col width:1fr height:fill gap:8px {
                                    text #mechanic.label "SATURATION / VALUE"
                                    {
                                        let color_control = color_picker_styled(
                                            parent,
                                            dye_color,
                                            ColorPickerStyle {
                                                field_height: dye_field_height,
                                                hue_height: 20.0,
                                                gap: 9.0,
                                                radius: 8.0,
                                                handle_size: 18.0,
                                                handle: color(ink.fg),
                                                handle_edge: color(shell),
                                                focus: color(control.focus),
                                            },
                                        );
                                        color_control.root().label("Dye color");
                                    }
                                }
                                col width:260px height:min-content shrink:0 gap:10px {
                                    text #mechanic.label "HEX COLOR"
                                    input #mechanic.field width:fill height:38px hex
                                    text #mechanic.label pad:(top:4px) "PRESETS"
                                    grid width:fill cols:{ GridTracks::repeat(4, [GridTrack::fr(1.0)]) }
                                        rows:{ GridTracks::repeat(3, [GridTrack::Fixed(Length::px(38.0))]) }
                                        gap:7px {
                                        for (rgb, ()) in { PRESETS.map(|rgb| (rgb, ())) } {
                                            (preset(dye_color, *rgb))
                                        }
                                    }
                                    (stacked_value_slider("STRUCTURE", structure, 0.0, 3.0, 0.01, "×"))
                                    text #mechanic.caption
                                        "Structure controls how much baked lightness variation remains."
                                }
                            }
                        }
                    }

                    col width:320px height:fill shrink:0 gap:14px {
                        col width:fill height:1fr align:center justify:center gap:10px
                            pad:16px radius:12px fill:card.fill
                            stroke:(width:1px color:card.edge-on) {
                            text #mechanic.section "CURRENT BRUSH"
                            el width:210px height:164px radius:12px
                                fill:{ rgb_color(representative_srgb(swatch_material.get(), appearance.get())) }
                                stroke:(width:2px color:chip.edge-over) {}
                            text #mechanic.value text-wrap:none { color_summary(appearance.get()) }
                            text #mechanic.caption text-wrap:none { finish_label(appearance.get().finish) }
                        }
                        col width:fill height:min-content gap:10px pad:16px radius:12px
                            fill:lane.fill stroke:(width:1px color:lane.edge) {
                            text #mechanic.section "FINISH"
                            text #mechanic.caption text-wrap:none
                                "Controls reflectivity and roughness."
                            (finish_button(&baked_finish, MaterialFinish::Baked, "As baked"))
                            (finish_button(&anodised_finish, MaterialFinish::Anodised, "Anodised"))
                            (finish_button(&painted_finish, MaterialFinish::Painted, "Painted"))
                        }
                    }
                }

                row width:fill height:46px shrink:0 align:center justify:between
                    pad:(horizontal:22px vertical:0px)
                    stroke:(width:1px color:shell-rule edges:top) {
                    text #mechanic.caption "L-drag  Paint   ·   Q  Sample   ·   Right-drag  Remove Paint"
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

fn mode_button(
    handles: &Handles,
    kind: ColorMode,
    label: &'static str,
    states: ColorStates,
) -> Element {
    let handles = handles.clone();
    let active = move || {
        matches!(
            (kind, handles.chroma.get().color),
            (ColorMode::Baked, MaterialColor::Baked)
                | (ColorMode::Shift, MaterialColor::Shift(_))
                | (ColorMode::Dye, MaterialColor::Dye(_))
        )
    };
    view! {
        button #mechanic.action width:1fr height:42px
            fill:{ if active() { color(control.pressed) } else { color(control.rest) } }
            stroke:(width:1px color:{ if active() { color(accent.key) } else { color(chip.edge) } })
            @click:{ set_mode(handles.chroma, kind, states); }
            { label }
    }
}

fn finish_button(handles: &Handles, finish: MaterialFinish, label: &'static str) -> Element {
    let handles = handles.clone();
    let active = move || handles.chroma.get().finish == finish;
    view! {
        button #mechanic.action width:fill height:42px shrink:0
            fill:{ if active() { color(control.pressed) } else { color(control.rest) } }
            stroke:(width:1px color:{ if active() { color(accent.key) } else { color(chip.edge) } })
            @click:{ handles.chroma.update(|appearance| appearance.finish = finish); }
            { label }
    }
}

fn value_slider(
    label: &'static str,
    value: State<f32>,
    min: f32,
    max: f32,
    step: f32,
    suffix: &'static str,
) -> Element {
    view! {
        row width:fill height:44px shrink:0 align:center gap:10px {
            text #mechanic.label width:106px shrink:0 { label }
            {
                let slider = slider_styled(
                    parent,
                    value,
                    min..=max,
                    Some(step),
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
                if suffix == "°" {
                    format!("{:.0}{suffix}", value.get())
                } else {
                    format!("{:.2}{suffix}", value.get())
                }
            }
        }
    }
}

fn stacked_value_slider(
    label: &'static str,
    value: State<f32>,
    min: f32,
    max: f32,
    step: f32,
    suffix: &'static str,
) -> Element {
    view! {
        col width:fill height:min-content gap:8px pad:(top:6px) {
            row width:fill height:min-content align:center justify:between {
                text #mechanic.label text-wrap:none { label }
                text #mechanic.value text-wrap:none {
                    format!("{:.2}{suffix}", value.get())
                }
            }
            {
                let slider = slider_styled(
                    parent,
                    value,
                    min..=max,
                    Some(step),
                    SliderStyle {
                        track: color(dial.track),
                        fill: color(accent.key),
                        thumb: color(ink.fg),
                        thumb_hover: color(accent.key),
                        focus: color(control.focus),
                        track_height: 7.0,
                        thumb_size: 18.0,
                    },
                );
                slider.root().restyle(|style| {
                    style.width(Dimension::Fill).shrink(0.0)
                });
            }
        }
    }
}

fn preset(dye_color: State<Color>, rgb: [u8; 3]) -> Element {
    view! {
        el width:fill height:fill radius:6px fill:{ rgb_color(rgb) }
            stroke:(width:1px color:chip.edge)
            @click:{ dye_color.set(rgb_color(rgb)); } {}
    }
}

fn set_mode(appearance: State<MaterialAppearance>, kind: ColorMode, states: ColorStates) {
    let current = appearance.get_untracked();
    let color = match kind {
        ColorMode::Baked => MaterialColor::Baked,
        ColorMode::Shift => MaterialColor::Shift(
            MaterialShift::new(
                states.hue.get_untracked(),
                states.chroma.get_untracked(),
                states.lightness.get_untracked(),
            )
            .expect("slider ranges always make a valid Shift"),
        ),
        ColorMode::Dye => {
            let [r, g, b, _] = states.dye.get_untracked().to_srgb8();
            MaterialColor::Dye(
                MaterialDye::new([r, g, b], states.structure.get_untracked())
                    .expect("picker controls always make a valid Dye"),
            )
        }
    };
    appearance.set(MaterialAppearance {
        color,
        finish: current.finish,
    });
}

fn set_shift(appearance: State<MaterialAppearance>, hue: f32, chroma: f32, lightness: f32) {
    let current = appearance.get_untracked();
    let shift = MaterialShift::new(hue, chroma, lightness)
        .expect("slider ranges always make a valid Shift");
    let wanted = MaterialAppearance {
        color: MaterialColor::Shift(shift),
        finish: current.finish,
    };
    if current != wanted {
        appearance.set(wanted);
    }
}

fn set_dye(appearance: State<MaterialAppearance>, target: [u8; 3], structure: f32) {
    let current = appearance.get_untracked();
    let dye = MaterialDye::new(target, structure).expect("slider ranges always make a valid Dye");
    let wanted = MaterialAppearance {
        color: MaterialColor::Dye(dye),
        finish: current.finish,
    };
    if current != wanted {
        appearance.set(wanted);
    }
}

fn set_f32_if_changed(state: State<f32>, value: f32) {
    if (state.get_untracked() - value).abs() > f32::EPSILON {
        state.set(value);
    }
}

fn default_shift() -> MaterialShift {
    MaterialShift::new(0.0, 1.0, 1.0).expect("default Shift is valid")
}

fn default_dye() -> MaterialDye {
    MaterialDye::new([0xE0, 0x56, 0x1F], 1.0).expect("default Dye is valid")
}

fn color_summary(appearance: MaterialAppearance) -> String {
    match appearance.color {
        MaterialColor::Baked => "Original".to_owned(),
        MaterialColor::Shift(shift) => format!(
            "Shift {:.0}° / {:.2}× / {:.2}×",
            shift.hue_degrees(),
            shift.chroma(),
            shift.lightness(),
        ),
        MaterialColor::Dye(dye) => format!(
            "Dye {} / {:.2}×",
            format_hex(dye.target_rgb()),
            dye.structure(),
        ),
    }
}

const fn finish_label(finish: MaterialFinish) -> &'static str {
    match finish {
        MaterialFinish::Baked => "As baked",
        MaterialFinish::Anodised => "Anodised",
        MaterialFinish::Painted => "Painted",
    }
}

fn rgb_color([r, g, b]: [u8; 3]) -> Color {
    Color::from_srgb8(r, g, b, 255)
}

fn parse_hex(value: &str) -> Option<[u8; 3]> {
    let value = value.trim().strip_prefix('#').unwrap_or(value.trim());
    if value.len() != 6 {
        return None;
    }
    Some([
        u8::from_str_radix(&value[0..2], 16).ok()?,
        u8::from_str_radix(&value[2..4], 16).ok()?,
        u8::from_str_radix(&value[4..6], 16).ok()?,
    ])
}

fn format_hex([r, g, b]: [u8; 3]) -> String {
    format!("#{r:02X}{g:02X}{b:02X}")
}

#[cfg(test)]
mod tests {
    use mechanic_core::{MaterialAppearance, MaterialColor, MaterialDye, MaterialFinish};

    use super::{color_summary, finish_label, parse_hex};

    #[test]
    fn hex_accepts_rgb8_and_leaves_invalid_text_uncommitted() {
        assert_eq!(parse_hex("#E0561F"), Some([0xE0, 0x56, 0x1F]));
        assert_eq!(parse_hex("2a4cc7"), Some([0x2A, 0x4C, 0xC7]));
        assert_eq!(parse_hex("#12345"), None);
        assert_eq!(parse_hex("#GG0000"), None);
    }

    #[test]
    fn status_describes_the_active_brush_without_controls() {
        let appearance = MaterialAppearance::new(
            MaterialColor::Dye(MaterialDye::new([42, 76, 199], 1.25).unwrap()),
            MaterialFinish::Painted,
        );
        assert_eq!(color_summary(appearance), "Dye #2A4CC7 / 1.25×");
        assert_eq!(finish_label(appearance.finish), "Painted");
    }
}
