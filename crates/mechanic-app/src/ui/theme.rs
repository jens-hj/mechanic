//! The whole overlay's palette and icon set.
//!
//! The colours carry meaning rather than decoration, and the views read better
//! when they say so: amber is positional (an angle, a travel limit), cyan is
//! continuous (a speed), teal is bound or focused, and lilac is temporal (a
//! dwell, a loop). Dashed edges mean "not configured yet" everywhere they
//! appear.
//!
//! One palette for every panel. The control block was drawn from these tokens
//! first and the rest of the UI was restyled onto them, which is what stops the
//! overlay reading as two applications sharing a window.

#[allow(clippy::wildcard_imports)] // Mosaic's authoring vocabulary is meant to be globbed.
use bevy_mosaic::ui::*;

mosaic_macros::scheme! {
    /// Every colour the control block paints with.
    pub Block {
        // The panel itself.
        shell:Color, shell-edge:Color, shell-rule:Color,
        // Text, from loudest to quietest.
        ink { fg:Color, muted:Color, legend:Color, dim:Color, faint:Color },
        // What the accents mean.
        accent { angle:Color, speed:Color, key:Color, time:Color, danger:Color },
        // The design's accent washes, pre-composed against the shell: it
        // authored them as CSS `rgba()`, which blends in sRGB, while the
        // renderer composites in linear space and would read them far
        // brighter.
        wash { angle:Color, speed:Color, key:Color, time:Color, mode-angle:Color,
               mode-speed:Color, chip-travel:Color, chip-loop:Color, badge:Color,
               pill-angle:Color, pill-speed:Color, travel-arc:Color, capturing:Color,
               delete:Color },
        // One joint's row.
        lane { edge:Color, edge-on:Color, fill:Color, fill-on:Color },
        badge { fill:Color, edge:Color, ink:Color },
        reticle { edge:Color, fill-over:Color },
        // The four property chips.
        chip { fill:Color, edge:Color, edge-over:Color, speed:Color, torque:Color,
               travel-edge:Color, loop-edge:Color },
        preset { edge:Color, fill-over:Color },
        // One state.
        card { fill:Color, edge:Color, edge-on:Color },
        dial { track:Color, tick:Color, limit:Color, grip:Color, knob:Color },
        key { edge-off:Color, ink-off:Color, ink-on:Color },
        mode { off:Color },
        port { fill:Color, fill-over:Color, idle:Color, off:Color },
        add { edge:Color, ring:Color, fill-over:Color },
        // One hotbar slot, and the tooltip above the bar.
        bar { fill:Color, slot:Color, slot-over:Color, slot-on:Color,
              edge:Color, edge-over:Color, edge-on:Color, shortcut:Color },
        // The help panel's lines, which are graded by how loudly they speak.
        help { title:Color, body:Color, muted:Color, good:Color, warn:Color, bad:Color },
        // The creation picker, and the sheet it darkens the world with.
        picker { veil:Color, sheet:Color, edge:Color, row:Color, row-over:Color,
                 row-edge:Color, danger:Color, danger-over:Color, notice:Color },
    }
}

mosaic_macros::scheme! {
    /// Every mark the control block draws.
    pub BlockIcons {
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

/// The palette, straight off the design.
pub(crate) fn palette() -> Block {
    mosaic_macros::theme! { Block {
        shell:#070C11F7, shell-edge:#24444C, shell-rule:#142430,
        ink { fg:#DCE9F2, muted:#8FA6B6, legend:#90A6B6, dim:#7E95A6, faint:#6E869A },
        accent { angle:#F2A33C, speed:#3FCBE0, key:#2FD8B4, time:#9C8BF0, danger:#E2565A },
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
        picker { veil:#03060AC2, sheet:#070C11FA, edge:#24444C, row:#0B141C,
                 row-over:#12202B, row-edge:#1D3644, danger:#2A1416,
                 danger-over:#3A1A1D, notice:#F2A33C },
    } }
}

/// The icon set, embedded at compile time.
pub(crate) fn icons() -> BlockIcons {
    mosaic_macros::theme! { BlockIcons {
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

/// Installs the palette and icons, which every `view!` below reads by token.
pub(crate) fn install() {
    install_theme(&palette());
    install_theme(&icons());
}
