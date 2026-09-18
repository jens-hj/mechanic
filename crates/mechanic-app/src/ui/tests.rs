use bevy_mosaic::ui::{Color, FontFamily, Length};
use mechanic_core::{
    ConstructionMaterial, MaterialAppearance, MaterialColor, MaterialDye, MaterialFinish,
};
use mosaic_core::Vector2;
use mosaic_core::theme::{color, install as install_theme, typed};
use mosaic_widgets::Ui;

use super::material_wheel;
use super::testing::{Overlay, VIEWPORT, away};
use super::theme::{BODY_FAMILY, DISPLAY_FAMILY, bar as bar_tokens, metrics, palette, typeface};
use super::{creations, escape_is_consumed, load_fonts, theme, worlds};
use crate::hotbar::{SelectedTool, Tool};

#[test]
fn bundled_fonts_cover_the_authored_type_roles() {
    let ui = Ui::new();
    load_fonts(&ui);
    theme::install();
    let fonts = ui.fonts();
    let fonts = fonts.borrow();

    assert!(fonts.has_family(DISPLAY_FAMILY));
    assert!(fonts.has_family(BODY_FAMILY));
    assert_eq!(
        typed(typeface.display, FontFamily::default),
        FontFamily::Named(DISPLAY_FAMILY.into()),
    );
    assert_eq!(
        typed(typeface.body, FontFamily::default),
        FontFamily::Named(BODY_FAMILY.into()),
    );
}

#[test]
fn handles_mount_with_the_world_gate_already_visible() {
    let mut initial_worlds = worlds::Model::default();
    initial_worlds.open = true;

    let handles = super::Handles::new(initial_worlds);

    assert!(handles.worlds.get_untracked().open);
}

/// The bug this guards against: a `stack` child fills its parent unless it
/// is told not to, and one full-bleed panel makes every point in the window
/// read as "over the overlay" — the machine behind it stops taking the
/// pointer at all, everywhere, with nothing on screen to explain why.
#[test]
fn the_world_shows_through_the_gaps_between_panels() {
    let overlay = Overlay::mount();
    // The middle of the window, where nothing is drawn: the help panel is
    // in the corner and hidden, and the hotbar is along the bottom.
    for at in [
        Vector2::new(VIEWPORT.width / 2.0, VIEWPORT.height / 2.0),
        Vector2::new(VIEWPORT.width - 8.0, 8.0),
        Vector2::new(8.0, VIEWPORT.height / 2.0),
    ] {
        assert!(
            !overlay.wants_pointer_at(at),
            "the world must keep the pointer at {at:?}",
        );
    }
}

/// And the panels themselves do take it, or a click would fall through the
/// thing it landed on.
#[test]
fn a_panel_takes_the_pointer_over_itself() {
    let overlay = Overlay::mount();
    let bar = overlay
        .reachable_boxes()
        .into_iter()
        .find(|rect| (rect.size.width - 64.0).abs() < 0.5)
        .expect("the hotbar is on screen");
    assert!(overlay.wants_pointer_at(bar.center()));
}

#[test]
fn chroma_status_is_read_only_and_tab_configuration_expands_for_dye() {
    let overlay = Overlay::mount();
    let ordinary_count = overlay.element_count();

    overlay
        .handles
        .hotbar
        .set(SelectedTool::from_editor_tool(Tool::Chroma));
    overlay.settle();
    let status_count = overlay.element_count();
    assert!(status_count > ordinary_count, "the Chroma status mounted");
    let status = overlay
        .shapes()
        .into_iter()
        .find(|shape| (shape.rect.size.width - 260.0).abs() < 0.5)
        .expect("read-only Chroma status panel");
    assert!(
        !overlay.wants_pointer_at(status.rect.center()),
        "the gameplay status must not claim the camera-owned pointer",
    );

    overlay.handles.material_wheel.set(material_wheel::Model {
        open: true,
        chroma_config: true,
        highlighted: None,
    });
    overlay.settle();
    let baked_count = overlay.element_count();
    assert!(
        baked_count > status_count,
        "holding the selector mounts the interactive Chroma editor"
    );

    overlay.handles.chroma.set(MaterialAppearance::new(
        MaterialColor::Dye(MaterialDye::new([224, 86, 31], 1.0).unwrap()),
        MaterialFinish::Baked,
    ));
    overlay.settle();
    assert!(
        overlay.element_count() > baked_count,
        "Dye mounts the picker, presets, hex field, and structure control",
    );
    let workbench = overlay
        .shapes()
        .into_iter()
        .find(|shape| {
            shape.rect.size.width > VIEWPORT.width * 0.85
                && shape.rect.size.width < VIEWPORT.width
                && shape.rect.size.height > VIEWPORT.height * 0.85
                && shape.rect.size.height < VIEWPORT.height
        })
        .expect("large inset Chroma workbench");
    assert!(away(workbench.rect.center(), Vector2::new(800.0, 450.0)) < 1.0);
    let expected_field_height = (VIEWPORT.height - 440.0).clamp(240.0, 760.0);
    assert!(
        overlay.shapes().into_iter().any(|shape| {
            shape.rect.size.width > 300.0
                && (shape.rect.size.height - expected_field_height).abs() < 0.5
        }),
        "the Dye saturation/value field scales to its available area"
    );
    assert!(
        overlay
            .labels()
            .iter()
            .any(|label| label == "Saturation and value"),
        "the large picker remains accessible",
    );

    overlay.handles.chroma.set(MaterialAppearance::BAKED);
    overlay
        .handles
        .material_wheel
        .set(material_wheel::Model::default());
    overlay.handles.hotbar.set(SelectedTool::default());
    overlay.settle();
    assert_eq!(overlay.element_count(), ordinary_count);
}

#[test]
fn typed_boundaries_preserve_panel_and_world_pointer_ownership() {
    let overlay = Overlay::mount();
    overlay.handles.help_open.set(true);
    overlay.settle();

    assert!(overlay.wants_pointer_at(Vector2::new(32.0, 32.0)));
    assert!(!overlay.wants_pointer_at(Vector2::new(900.0, 300.0)));
}

#[test]
fn reticle_is_centred_and_never_takes_world_input() {
    let overlay = Overlay::mount();
    let centre = Vector2::new(VIEWPORT.width / 2.0, VIEWPORT.height / 2.0);
    assert!(
        overlay
            .ink()
            .into_iter()
            .any(|rect| away(rect.center(), centre) < 1.0 && rect.size.width < 40.0),
        "the compact reticle is painted at viewport centre",
    );
    assert!(!overlay.wants_pointer_at(centre));
}

#[test]
fn material_wheel_is_large_textured_and_paints_the_highlight_last() {
    let overlay = Overlay::mount();
    overlay.handles.material_wheel.set(material_wheel::Model {
        open: true,
        chroma_config: false,
        highlighted: Some(crate::hotbar::WheelChoice::ConstructionMaterial(
            ConstructionMaterial::Concrete,
        )),
    });
    overlay.settle();
    let centre = Vector2::new(VIEWPORT.width / 2.0, VIEWPORT.height / 2.0);
    let sectors = overlay
        .shapes()
        .into_iter()
        .filter(|shape| {
            away(shape.rect.center(), centre) < 1.0
                && shape
                    .strokes
                    .first()
                    .is_some_and(|stroke| (stroke.width - 72.0).abs() < f32::EPSILON)
        })
        .collect::<Vec<_>>();
    assert_eq!(sectors.len(), ConstructionMaterial::ALL.len());
    assert_eq!(
        sectors
            .last()
            .and_then(|shape| shape.strokes.first())
            .and_then(|stroke| stroke.color.as_solid()),
        Some(color(bar_tokens.slot_over)),
    );
    assert!(sectors.iter().all(|shape| shape.rect.size.width > 250.0));

    let last_sector_paint = overlay
        .indexed_shapes()
        .into_iter()
        .filter(|(_, shape)| {
            shape
                .strokes
                .first()
                .is_some_and(|stroke| (stroke.width - 72.0).abs() < f32::EPSILON)
        })
        .map(|(index, _)| index)
        .max()
        .expect("material sector paint");
    let first_block_paint = overlay
        .indexed_images()
        .into_iter()
        .filter(|(_, intrinsic, _)| *intrinsic == (96, 106))
        .map(|(index, _, _)| index)
        .min()
        .expect("material block paint");
    assert!(
        last_sector_paint < first_block_paint,
        "all sector backgrounds paint beneath every material block"
    );

    let block_previews = overlay
        .images()
        .into_iter()
        .filter(|(intrinsic, destination)| {
            *intrinsic == (96, 106)
                && (destination.size.width - 54.0).abs() < f32::EPSILON
                && destination.size.height > 59.0
                && destination.size.height <= 60.0
        })
        .count();
    assert_eq!(block_previews, ConstructionMaterial::ALL.len());

    let label_texture = overlay
        .images()
        .into_iter()
        .find(|(intrinsic, destination)| {
            *intrinsic == (3072, 3072)
                && (destination.size.width - 236.0).abs() < f32::EPSILON
                && (destination.size.height - 54.0).abs() < f32::EPSILON
        })
        .expect("the selected material textures the high-resolution swatch");
    assert!((label_texture.1.center().x - centre.x).abs() < f32::EPSILON);
    assert!(
        label_texture.1.origin.y > centre.y + 190.0,
        "the material swatch is spaced below the wheel"
    );

    assert!(overlay.shapes().into_iter().any(|shape| {
        (shape.rect.size.width - 236.0).abs() < f32::EPSILON
            && (shape.rect.size.height - 54.0).abs() < f32::EPSILON
            && (shape.radii.tl - 5.0).abs() < f32::EPSILON
    }));

    assert!(!overlay.wants_pointer_at(centre));
}

#[test]
fn alternate_palette_and_metric_themes_update_the_mounted_tree() {
    let overlay = Overlay::mount();
    overlay.handles.help_open.set(true);
    overlay.settle();
    let elements = overlay.element_count();

    let switched_shell = Color::from_rgb_hex(0x0017_304A);
    let mut switched_palette = palette();
    switched_palette.shell = switched_shell;
    let mut switched_metrics = metrics();
    switched_metrics.radius.panel = Length::px(23.0);
    install_theme(&switched_palette);
    install_theme(&switched_metrics);
    overlay.settle();

    let panel = overlay
        .shapes()
        .into_iter()
        .find(|shape| (shape.rect.size.width - 720.0).abs() < 0.5)
        .expect("the already-mounted help panel repaints");
    assert_eq!(panel.fill.as_solid(), Some(switched_shell));
    assert!((panel.radii.tl - 23.0).abs() < f32::EPSILON);
    assert_eq!(
        overlay.element_count(),
        elements,
        "theme changes do not rebuild"
    );

    theme::install();
}

#[test]
fn escape_is_reserved_only_for_active_panel_editing() {
    assert!(!escape_is_consumed(false, false, false));
    assert!(escape_is_consumed(true, false, false));
    assert!(escape_is_consumed(false, true, false));
    assert!(escape_is_consumed(false, false, true));
}

/// The picker is the one panel that is meant to cover the window: while it
/// is up, a click anywhere is a click on it.
#[test]
fn the_open_picker_covers_the_window() {
    let overlay = Overlay::mount();
    overlay.handles.creations.update(|model| model.open = true);
    overlay.settle();
    for at in [
        Vector2::new(VIEWPORT.width / 2.0, VIEWPORT.height / 2.0),
        Vector2::new(4.0, 4.0),
        Vector2::new(VIEWPORT.width - 4.0, VIEWPORT.height - 4.0),
    ] {
        assert!(
            overlay.wants_pointer_at(at),
            "a modal that lets clicks through at {at:?} is not a modal",
        );
    }
}

#[test]
fn world_picker_is_an_opaque_standalone_launch_screen() {
    let overlay = Overlay::mount();
    overlay.handles.worlds.update(|model| model.open = true);
    overlay.settle();

    let screen = overlay
        .shapes()
        .into_iter()
        .find(|shape| {
            (shape.rect.size.width - VIEWPORT.width).abs() < 0.5
                && (shape.rect.size.height - VIEWPORT.height).abs() < 0.5
                && shape.fill.as_solid() == Some(color(theme::picker.screen))
        })
        .expect("the world picker paints the full window");
    assert_eq!(screen.fill.as_solid(), Some(color(theme::picker.screen)));

    let centre = Vector2::new(VIEWPORT.width / 2.0, VIEWPORT.height / 2.0);
    assert!(overlay.wants_pointer_at(centre));
    let labels = overlay.labels();
    assert!(labels.iter().any(|label| label == "MECHANIC // WORLD GATE"));
    assert!(labels.iter().any(|label| label == "Generate world"));
    assert!(
        !overlay
            .ink()
            .into_iter()
            .any(|rect| away(rect.center(), centre) < 1.0 && rect.size.width < 40.0),
        "the in-game reticle is not mounted behind the world picker",
    );
}

#[test]
fn overlay_uses_diagonal_ui_corners_and_a_round_pipe_profile() {
    let overlay = Overlay::mount();
    overlay.handles.help_open.set(true);
    overlay.handles.worlds.set(worlds::Model::default());
    overlay.settle();

    let rounded = overlay
        .shapes()
        .into_iter()
        .filter(|shape| {
            shape.radii.tl > 0.0
                || shape.radii.tr > 0.0
                || shape.radii.br > 0.0
                || shape.radii.bl > 0.0
        })
        .collect::<Vec<_>>();
    assert!(
        !rounded.is_empty(),
        "the overlay has rounded surfaces to inspect"
    );
    let pipe_profiles = rounded
        .iter()
        .filter(|shape| {
            let size = shape.rect.size;
            ((size.width - 26.0).abs() < f32::EPSILON
                && (size.height - 30.0).abs() < f32::EPSILON
                && (shape.radii.tl - 13.0).abs() < f32::EPSILON)
                || ((size.width - 10.0).abs() < f32::EPSILON
                    && (size.height - 16.0).abs() < f32::EPSILON
                    && (shape.radii.tl - 5.0).abs() < f32::EPSILON)
        })
        .collect::<Vec<_>>();
    assert_eq!(pipe_profiles.len(), 2, "the pipe icon has two profiles");
    assert!(pipe_profiles.iter().all(|shape| {
        (shape.exponents.tl - 2.0).abs() < f32::EPSILON
            && (shape.exponents.tr - 2.0).abs() < f32::EPSILON
            && (shape.exponents.br - 2.0).abs() < f32::EPSILON
            && (shape.exponents.bl - 2.0).abs() < f32::EPSILON
    }));

    let non_diagonal_ui = rounded
        .iter()
        .filter(|shape| !pipe_profiles.contains(shape))
        .filter(|shape| {
            (shape.exponents.tl - 1.0).abs() > f32::EPSILON
                || (shape.exponents.tr - 1.0).abs() > f32::EPSILON
                || (shape.exponents.br - 1.0).abs() > f32::EPSILON
                || (shape.exponents.bl - 1.0).abs() > f32::EPSILON
        })
        .map(|shape| (shape.rect, shape.radii, shape.exponents))
        .collect::<Vec<_>>();
    assert!(
        non_diagonal_ui.is_empty(),
        "UI shapes without diagonal corners: {non_diagonal_ui:?}"
    );
}

/// A `scroll` fills its parent whatever size is written on it — the size
/// styles the content it holds — so a sheet that *is* the scroll cannot be
/// centred by its veil, and lands in the corner instead.
#[test]
fn the_picker_sits_in_the_middle_of_the_window() {
    let overlay = Overlay::mount();
    overlay.handles.creations.update(|model| model.open = true);
    overlay.settle();

    let sheet = overlay
        .rects()
        .into_iter()
        .map(|(_, rect)| rect)
        .find(|rect| (rect.size.width - creations::SHEET).abs() < 0.5)
        .expect("the picker is on screen");
    let (sits, middle) = (
        sheet.center(),
        Vector2::new(VIEWPORT.width / 2.0, VIEWPORT.height / 2.0),
    );
    assert!(
        (sits.x - middle.x).abs() < 0.5 && (sits.y - middle.y).abs() < 0.5,
        "the picker sits at {sits:?} in a window centred on {middle:?}",
    );
}
