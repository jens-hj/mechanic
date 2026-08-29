//! The overlay's independently installable design-token schemes.
//!
//! Colour, geometry, typography, effects, and icons are separate schemes so a
//! future palette or density switch only invalidates the values it owns. The
//! default themes below preserve the measurements and colours the overlay was
//! authored with.

#[allow(clippy::wildcard_imports)] // Mosaic's authoring vocabulary is meant to be globbed.
use bevy_mosaic::ui::*;

/// Where the design's two typefaces are looked for.
pub(crate) const FONT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/fonts");

/// The typeface the design uses for titles and section headings.
pub(crate) const DISPLAY_FAMILY: &str = "Chakra Petch";

/// The typeface the design uses for body copy, controls, labels, and numbers.
pub(crate) const BODY_FAMILY: &str = "JetBrains Mono";

mosaic_macros::scheme! {
    /// Surfaces, ink, accents, interaction states, and specialised block colours.
    pub(crate) MechanicPalette {
        shell:Color, shell-edge:Color, shell-rule:Color,
        ink { fg:Color, muted:Color, legend:Color, dim:Color, faint:Color },
        accent { angle:Color, speed:Color, key:Color, time:Color, danger:Color },
        control { rest:Color, hover:Color, pressed:Color, focus:Color },
        status-color { good:Color, warn:Color, bad:Color },
        wash { angle:Color, speed:Color, key:Color, time:Color, mode-angle:Color,
               mode-speed:Color, chip-travel:Color, chip-loop:Color, badge:Color,
               pill-angle:Color, pill-speed:Color, travel-arc:Color, capturing:Color,
               delete:Color },
        lane { edge:Color, edge-on:Color, fill:Color, fill-on:Color },
        badge { fill:Color, edge:Color, ink:Color },
        reticle { edge:Color, fill-over:Color },
        chip { fill:Color, edge:Color, edge-over:Color, speed:Color, torque:Color,
               travel-edge:Color, loop-edge:Color },
        preset { edge:Color, fill-over:Color },
        card { fill:Color, edge:Color, edge-on:Color },
        dial { track:Color, tick:Color, limit:Color, grip:Color, knob:Color },
        key { edge-off:Color, ink-off:Color, ink-on:Color },
        mode { off:Color },
        port { fill:Color, fill-over:Color, idle:Color, off:Color },
        add { edge:Color, ring:Color, fill-over:Color },
        bar { fill:Color, slot:Color, slot-over:Color, slot-on:Color,
              edge:Color, edge-over:Color, edge-on:Color, shortcut:Color },
        help { title:Color, body:Color, muted:Color, good:Color, warn:Color, bad:Color },
        picker { screen:Color, veil:Color, sheet:Color, edge:Color, row:Color, row-over:Color,
                 row-edge:Color, danger:Color, danger-over:Color, notice:Color },
    }
}

mosaic_macros::scheme! {
    /// Repeated overlay measurements. Procedural drawing geometry stays in Rust.
    pub(crate) MechanicMetrics {
        space { xxs:Length, xs:Length, sm:Length, md:Length, lg:Length, xl:Length },
        pad { panel:Length, sheet:Length, action-x:Length, action-y:Length },
        radius { field:Length, action:Length, chip:Length, panel:Length,
                 elevated:Length, badge:Length },
        control-size { compact-height:Length, height:Length, row-height:Length },
        border { hairline:Scalar, strong:Scalar },
        panel-size { help-width:Length, inset:Length, modal-width:Length,
                     summary-width:Length, summary-height:Length },
    }
}

mosaic_macros::scheme! {
    /// The semantic type scale used by every overlay surface.
    pub(crate) MechanicType {
        typeface { body:FontFamily, display:FontFamily },
        text-size { micro:Length, tiny:Length, caption:Length, label:Length, body:Length,
                    value:Length, section:Length, heading:Length, title:Length, hero:Length },
        text-weight { medium:Scalar, bold:Scalar },
        text-tracking { tight:Length, label:Length, help-title:Length,
                        section:Length, title:Length },
    }
}

mosaic_macros::scheme! {
    /// Elevation and the shared interaction/theme transition.
    pub(crate) MechanicEffects {
        panel-shadow:Shadow,
        modal-shadow:Shadow,
        motion:Transition,
    }
}

mosaic_macros::scheme! {
    /// Every embedded mark the control block draws.
    pub(crate) MechanicIcons {
        mark:Svg,
        legend-angle:Svg, legend-spin:Svg, legend-key:Svg, legend-time:Svg,
        locate:Svg,
        chip-speed:Svg, chip-torque:Svg, chip-travel-limited:Svg, chip-travel-free:Svg,
        chip-loop:Svg, chip-once:Svg,
        preset-steer:Svg, preset-drive:Svg, preset-spin:Svg,
        home:Svg, mode-angle:Svg, mode-speed:Svg, delete:Svg, keyboard:Svg,
        port-latch:Svg, port-linked:Svg, port-dwell:Svg,
        label-release:Svg, label-dwell:Svg,
    }
}

/// The default palette, straight off the design.
pub(crate) fn palette() -> MechanicPalette {
    mosaic_macros::theme! { MechanicPalette {
        shell:#070C11F7, shell-edge:#24444C, shell-rule:#142430,
        ink { fg:#DCE9F2, muted:#8FA6B6, legend:#90A6B6, dim:#7E95A6, faint:#6E869A },
        accent { angle:#F2A33C, speed:#3FCBE0, key:#2FD8B4, time:#9C8BF0, danger:#E2565A },
        control { rest:#0B141C, hover:#122029, pressed:#0C2425, focus:#2FD8B4 },
        status-color { good:#2FD8B4, warn:#F2A33C, bad:#E2565A },
        wash { angle:#231E16, speed:#0E232A, key:#0C2425, time:#191B2C, mode-angle:#322E25,
               mode-speed:#163540, chip-travel:#1A1814, chip-loop:#131623, badge:#0D2B29,
               pill-angle:#2E2B25, pill-speed:#15313C, travel-arc:#322E25, capturing:#122F33,
               delete:#2A1416 },
        lane { edge:#132330, edge-on:#25454F, fill:#090F15, fill-on:#0C151C },
        badge { fill:#0D1A22, edge:#254050, ink:#8FA6B6 },
        reticle { edge:#1E3546, fill-over:#0E2226 },
        chip { fill:#0B141C, edge:#1D3644, edge-over:#3A5F72, speed:#9FD6E4, torque:#E0BE86,
               travel-edge:#4A3A20, loop-edge:#3A3363 },
        preset { edge:#23394A, fill-over:#0D1D22 },
        card { fill:#0E1821, edge:#1B2C39, edge-on:#28465A },
        dial { track:#16232F, tick:#6E869A, limit:#8A6A38, grip:#C98A34, knob:#0B131B },
        key { edge-off:#2A3E4E, ink-off:#54697A, ink-on:#0A1A16 },
        mode { off:#4A6070 },
        port { fill:#0A121A, fill-over:#12202B, idle:#54697A, off:#2A3E4E },
        add { edge:#22394A, ring:#1A2A38, fill-over:#0A141A },
        bar { fill:#070C11F2, slot:#0B141C, slot-over:#122029, slot-on:#0C2425,
              edge:#1B2C39, edge-over:#3A5F72, edge-on:#2FD8B4, shortcut:#54697A },
        help { title:#2FD8B4, body:#DCE9F2, muted:#7E95A6, good:#2FD8B4,
               warn:#F2A33C, bad:#E2565A },
        picker { screen:#03060A, veil:#03060AC2, sheet:#070C11FA, edge:#24444C, row:#0B141C,
                 row-over:#12202B, row-edge:#1D3644, danger:#2A1416,
                 danger-over:#3A1A1D, notice:#F2A33C },
    } }
}

/// The default spacing and density.
pub(crate) fn metrics() -> MechanicMetrics {
    mosaic_macros::theme! { MechanicMetrics {
        space { xxs:2px, xs:3px, sm:6px, md:8px, lg:10px, xl:16px },
        pad { panel:16px, sheet:24px, action-x:16px, action-y:9px },
        radius { field:6px, action:7px, chip:8px, panel:10px, elevated:14px, badge:13px },
        control-size { compact-height:34px, height:42px, row-height:52px },
        border { hairline:1, strong:2 },
        panel-size { help-width:720px, inset:16px, modal-width:640px,
                     summary-width:300px, summary-height:102px },
    } }
}

/// The default typography.
pub(crate) fn typography() -> MechanicType {
    mosaic_macros::theme! { MechanicType {
        typeface {
            body:(FontFamily::Named(BODY_FAMILY.into())),
            display:(FontFamily::Named(DISPLAY_FAMILY.into())),
        },
        text-size { micro:8px, tiny:9px, caption:12px, label:13px, body:14px,
                    value:15px, section:16px, heading:17px, title:19px, hero:26px },
        text-weight { medium:600, bold:700 },
        text-tracking { tight:0px, label:0.6px, help-title:1.4px,
                        section:1.7px, title:2.7px },
    } }
}

/// Resolves an authored font weight into Mosaic's integer text weight.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "theme weights are clamped and rounded before conversion"
)]
pub(crate) fn resolved_weight(token: mosaic_core::theme::ScalarToken) -> u16 {
    mosaic_core::theme::scalar(token)
        .round()
        .clamp(1.0, 1_000.0) as u16
}

/// The default elevation and 140 ms motion.
pub(crate) fn effects() -> MechanicEffects {
    mosaic_macros::theme! { MechanicEffects {
        panel-shadow:(offset:(x:0px y:12px) blur:26px color:#00000073),
        modal-shadow:(offset:(x:0px y:30px) blur:90px color:#00000099),
        motion:(all:ease(out 140ms)),
    } }
}

/// The icon set, embedded at compile time.
pub(crate) fn icons() -> MechanicIcons {
    mosaic_macros::theme! { MechanicIcons {
        mark:"assets/control-block/mark.svg",
        legend-angle:"assets/control-block/legend-angle.svg",
        legend-spin:"assets/control-block/legend-spin.svg",
        legend-key:"assets/control-block/legend-key.svg",
        legend-time:"assets/control-block/legend-time.svg",
        locate:"assets/control-block/locate.svg",
        chip-speed:"assets/control-block/chip-speed.svg",
        chip-torque:"assets/control-block/chip-torque.svg",
        chip-travel-limited:"assets/control-block/chip-travel-limited.svg",
        chip-travel-free:"assets/control-block/chip-travel-free.svg",
        chip-loop:"assets/control-block/chip-loop.svg",
        chip-once:"assets/control-block/chip-once.svg",
        preset-steer:"assets/control-block/preset-steer.svg",
        preset-drive:"assets/control-block/preset-drive.svg",
        preset-spin:"assets/control-block/preset-spin.svg",
        home:"assets/control-block/home.svg",
        mode-angle:"assets/control-block/mode-angle.svg",
        mode-speed:"assets/control-block/mode-speed.svg",
        delete:"assets/control-block/delete.svg",
        keyboard:"assets/control-block/keyboard.svg",
        port-latch:"assets/control-block/port-latch.svg",
        port-linked:"assets/control-block/port-linked.svg",
        port-dwell:"assets/control-block/port-dwell.svg",
        label-release:"assets/control-block/label-release.svg",
        label-dwell:"assets/control-block/label-dwell.svg",
    } }
}

/// Installs one default theme for every scheme the overlay reads.
pub(crate) fn install() {
    install_theme(&palette());
    install_theme(&metrics());
    install_theme(&typography());
    install_theme(&effects());
    install_theme(&icons());
}
