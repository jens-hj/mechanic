//! The tool hotbar along the bottom of the window.
//!
//! The icons are drawn rather than drawn *from* anything — each is a handful of
//! rectangles, rings and bars in a 40×40 box, carrying across the numbers the
//! panel has always used. Two conversions to watch, both of which draw something
//! plausible when they are wrong: a placed shape sits at its *centre*, not at
//! its near corner, and a round cap reaches half a stroke past its endpoint, so
//! a bar of a given length is a line that much shorter.

#![allow(clippy::wildcard_imports)] // Mosaic's authoring vocabulary is meant to be globbed.

use bevy_mosaic::ui::*;
use mechanic_core::ConstructionMaterial;
use mosaic_core::{Rect, Vector2, theme::color};
use mosaic_macros::{component, view};
use mosaic_widgets::input::{EventCtx, PointerButton};

use super::components::{OverlayBadge, OverlayBadgeProps};
#[allow(unused_imports)] // Style constants are consumed by `view!` expansion.
use super::styles::*;
#[allow(clippy::wildcard_imports)] // The design tokens are read as bare names.
use super::theme::*;
use super::{Handles, UiIntent};
use crate::controls::GameAction;
use crate::hotbar::{MainTool, MatterMode, Tool};

/// Side of one slot.
const SLOT: f32 = 64.0;

/// Side of the icon box inside a slot.
const ICON: f32 = 40.0;

/// How far the bar floats above the bottom of the window.
const FLOAT: f32 = 18.0;

/// Corner radius of one slot.
const SLOT_RADIUS: f32 = 7.0;

/// The gap between the bar's edge and the slots inside it.
const BAR_PAD: f32 = 8.0;

/// The bar's own corner radius.
///
/// Derived rather than chosen: two corners are concentric when the outer
/// radius is the inner one plus the distance between them, and only then do
/// the two curves stay parallel instead of pinching at the diagonal.
const BAR_RADIUS: f32 = SLOT_RADIUS + BAR_PAD;

const MATERIAL_MENU_WIDTH: f32 = 156.0;
const MATERIAL_ROW_HEIGHT: f32 = 34.0;
const MATERIAL_MENU_GAP: f32 = 8.0;
const MATERIAL_MENU_HEIGHT: f32 = {
    let mut height = 0.0;
    let mut row = 0;
    while row < ConstructionMaterial::ALL.len() {
        height += MATERIAL_ROW_HEIGHT;
        row += 1;
    }
    height
};
const MATERIAL_MENU_OFFSET_Y: f32 = -((MATERIAL_MENU_HEIGHT + SLOT) * 0.5 + MATERIAL_MENU_GAP);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HoverTarget {
    Tool(MainTool),
    Mode(MatterMode),
}

/// The bar, and the tooltip that rides above it.
///
/// Placed against the bottom edge by measuring itself: the window's size is
/// pushed in, and a column that hugs its slots is the only one that knows how
/// wide it ended up.
#[component]
pub(crate) fn Hotbar(handles: Handles) -> Element {
    let viewport = handles.viewport;
    let hovered = handles.hovered;
    let selected_material = handles.material;
    let selected_terrain_material = handles.terrain_material;
    let selection = handles.hotbar;
    let material_menu = handles.material_menu;
    let mode_slots = handles.clone();
    let tool_slots = handles.clone();
    let size: State<Size> = State::new(Size::ZERO);
    let named = move || {
        hovered.get().map_or_else(String::new, |target| {
            let selected = selection.get();
            match target {
                HoverTarget::Tool(MainTool::MatterManipulator) => format!(
                    "Matter Manipulator · {} · {}",
                    selected.matter_mode.label(),
                    contextual_choice(
                        selected.matter_mode,
                        selected.item.label(),
                        selected_material.get().label(),
                        terrain_material_label(selected_terrain_material.get()),
                    )
                ),
                HoverTarget::Tool(tool) => tool.label().to_owned(),
                HoverTarget::Mode(matter_mode) => format!(
                    "{} · {}",
                    matter_mode.label(),
                    contextual_choice(
                        matter_mode,
                        selected.item.label(),
                        selected_material.get().label(),
                        terrain_material_label(selected_terrain_material.get()),
                    )
                ),
            }
        })
    };
    let at = move || {
        let window = viewport.get();
        let own = size.get();
        (
            Length::px((window.width - own.width) / 2.0),
            Length::px(window.height - FLOAT - own.height),
        )
    };
    view! {
        col width:min-content height:min-content align:center gap:6px
            translate:(x:{ at().0 } y:{ at().1 })
            @layout:{ move |bounds: Rect| {
                if size.get_untracked() != bounds.size {
                    size.set(bounds.size);
                }
            } } {
            if hovered.get().is_some() && material_menu.get().is_none() {
                OverlayBadge width:max-content height:24px
                    pad:(left:10px right:12px top:0px bottom:0px) radius:12px {
                    text #mechanic.caption text-wrap:none font-color:accent.key { named() }
                }
            }
            if handles.hotbar.with(|selected| {
                selected.tool == Some(MainTool::MatterManipulator)
            }) {
                MatterModes handles:(mode_slots.clone())
            }
            row width:min-content height:min-content align:center gap:8px
                pad:{ Edges::all(Length::px(BAR_PAD)) } radius:{ Length::px(BAR_RADIUS) }
                fill:bar.fill stroke:(width:1px color:shell-edge) {
                for (tool, ()) in { MainTool::ALL.map(|tool| (tool, ())) } {
                    (tool_slot(&tool_slots, *tool))
                }
            }
        }
    }
}

#[component]
fn MatterModes(handles: Handles) -> Element {
    view! {
        row width:min-content height:min-content align:center gap:8px
            pad:{ Edges::all(Length::px(BAR_PAD)) }
            radius:{ Length::px(BAR_RADIUS) }
            fill:bar.fill stroke:(width:1px color:shell-edge) {
            for (matter_mode, ()) in {
                MatterMode::ALL.map(|matter_mode| (matter_mode, ()))
            } {
                (mode_slot(&handles, *matter_mode))
            }
        }
    }
}

/// One slot: an icon, the key that picks it, and whether it is in hand.
///
/// A stack rather than a column: the shortcut is placed in the corner by
/// `translate:`, but a translate moves what is drawn without giving back the
/// room it was laid out in — so in a column the icon and the label centre as a
/// pair and the icon rides half a line of text high in its slot. Stacked, both
/// centre on the slot and only the label moves.
fn tool_slot(handles: &Handles, tool: MainTool) -> Element {
    let handles = handles.clone();
    let selection = handles.hotbar;
    let hovered = handles.hovered;
    let controls = handles.controls;
    let shortcut = move || controls.with(|bindings| bindings.label(GameAction::for_tool(tool)));
    let held = move || selection.get().tool == Some(tool);
    let icon = icon(match tool {
        MainTool::MatterManipulator => Tool::Block,
        MainTool::Welder => Tool::Weld,
        MainTool::Connector => Tool::Connector,
        MainTool::Hammer => Tool::Hammer,
    });
    view! {
        stack width:{ Length::px(SLOT) } height:{ Length::px(SLOT) } align:center justify:center
            radius:{ Length::px(SLOT_RADIUS) }
            fill:{ if held() { color(bar.slot_on) } else { color(bar.slot) } }
            stroke:(width:1px color:{
                if held() { color(bar.edge_on) } else { color(bar.edge) }
            })
            @pointer:{ move |event: &PointerEvent, _: &mut EventCtx| match event.kind {
                PointerEventKind::Enter => hovered.set(Some(HoverTarget::Tool(tool))),
                // Only when this slot is still the one being named: the pointer
                // enters the next slot before it leaves this one.
                PointerEventKind::Leave
                    if hovered.get_untracked() == Some(HoverTarget::Tool(tool)) => {
                    hovered.set(None);
                }
                _ => {}
            } }
            @click:{ handles.ask(UiIntent::Tool(tool)); }
            hover { fill:bar.slot-over stroke:(width:1px color:bar.edge-over) }
            pressed { fill:control.pressed stroke:(width:1px color:bar.edge-on) } {
            (icon)
            text #mechanic.caption font-color:bar.shortcut translate:(x:-22px y:-20px)
                { shortcut() }
        }
    }
}

fn mode_slot(handles: &Handles, matter_mode: MatterMode) -> Element {
    let handles = handles.clone();
    let selection = handles.hotbar;
    let hovered = handles.hovered;
    let menu = handles.material_menu;
    let material_hover = handles.material_hover;
    let material_mode = matches!(matter_mode, MatterMode::Block | MatterMode::Cylinder);
    let controls = handles.controls;
    let shortcut = move || {
        controls.with(|bindings| {
            hotbar_shortcut_label(bindings.label(GameAction::for_mode(matter_mode)))
        })
    };
    let gesture = handles.clone();
    let held = move || {
        let selected = selection.get();
        selected.tool == Some(MainTool::MatterManipulator) && selected.matter_mode == matter_mode
    };
    let icon = match matter_mode {
        MatterMode::Block => icon(Tool::Block),
        MatterMode::Cylinder => icon(Tool::Cylinder),
        MatterMode::Item => icon(selection.get().item.editor_tool()),
        MatterMode::Terrain => terrain_icon(),
        MatterMode::Manipulate => icon(Tool::Shape),
    };
    view! {
        stack width:{ Length::px(SLOT) } height:{ Length::px(SLOT) } align:center justify:center
            radius:{ Length::px(SLOT_RADIUS) }
            fill:{ if held() { color(bar.slot_on) } else { color(bar.slot) } }
            stroke:(width:1px color:{
                if held() { color(bar.edge_on) } else { color(bar.edge) }
            })
            @pointer:{ move |event: &PointerEvent, _: &mut EventCtx| match event.kind {
                PointerEventKind::Enter => hovered.set(Some(HoverTarget::Mode(matter_mode))),
                PointerEventKind::Leave
                    if hovered.get_untracked() == Some(HoverTarget::Mode(matter_mode)) => {
                    hovered.set(None);
                }
                _ => {}
            } }
            @pointer:{ move |event: &PointerEvent, ctx: &mut EventCtx| {
                if !material_mode {
                    return;
                }
                match event.kind {
                    PointerEventKind::Down(PointerButton::Primary) => {
                        ctx.claim_pointer();
                        ctx.suppress_click();
                        menu.set(Some(matter_mode));
                        material_hover.set(material_at(event.position, ctx.target_rect()));
                    }
                    PointerEventKind::Move if menu.get_untracked() == Some(matter_mode) => {
                        material_hover.set(material_at(event.position, ctx.target_rect()));
                    }
                    PointerEventKind::Up(PointerButton::Primary)
                        if menu.get_untracked() == Some(matter_mode) => {
                        if let Some(material) = material_at(event.position, ctx.target_rect()) {
                            gesture.ask(UiIntent::MaterialMode(material, matter_mode));
                        } else if ctx.target_rect().contains(event.position) {
                            gesture.ask(UiIntent::MatterMode(matter_mode));
                        }
                        menu.set(None);
                        material_hover.set(None);
                    }
                    PointerEventKind::Cancel if menu.get_untracked() == Some(matter_mode) => {
                        menu.set(None);
                        material_hover.set(None);
                    }
                    _ => {}
                }
            } }
            @click:{ if !material_mode { handles.ask(UiIntent::MatterMode(matter_mode)); } }
            hover { fill:bar.slot-over stroke:(width:1px color:bar.edge-over) }
            pressed { fill:control.pressed stroke:(width:1px color:bar.edge-on) } {
            (icon)
            if menu.get() == Some(matter_mode) {
                col width:{ Length::px(MATERIAL_MENU_WIDTH) } height:min-content
                    translate:(x:0px y:{ Length::px(MATERIAL_MENU_OFFSET_Y) }) radius:8px clip
                    fill:bar.fill stroke:(width:1px color:shell-edge) {
                    for (material, ()) in { ConstructionMaterial::ALL.map(|material| (material, ())) } {
                        (material_row(*material, material_hover))
                    }
                }
            }
            text #mechanic.caption font-color:bar.shortcut translate:(x:-22px y:-20px)
                { shortcut() }
        }
    }
}

const fn terrain_material_label(material: mechanic_world::TerrainMaterial) -> &'static str {
    match material {
        mechanic_world::TerrainMaterial::SurfaceCover => "Grass",
        mechanic_world::TerrainMaterial::Soil => "Dirt",
        mechanic_world::TerrainMaterial::Rock => "Stone",
        mechanic_world::TerrainMaterial::Sand => "Sand",
        mechanic_world::TerrainMaterial::Iron => "Iron",
        mechanic_world::TerrainMaterial::Graphite => "Graphite",
    }
}

fn hotbar_shortcut_label(label: String) -> String {
    label.strip_prefix("Shift+").unwrap_or(&label).to_owned()
}

const fn contextual_choice(
    matter_mode: MatterMode,
    item: &'static str,
    construction_material: &'static str,
    terrain_material: &'static str,
) -> &'static str {
    match matter_mode {
        MatterMode::Block | MatterMode::Cylinder => construction_material,
        MatterMode::Item => item,
        MatterMode::Terrain => terrain_material,
        MatterMode::Manipulate => "Region editing",
    }
}

fn material_row(
    material: ConstructionMaterial,
    highlighted: State<Option<ConstructionMaterial>>,
) -> Element {
    view! {
        row #mechanic.list-row width:fill height:{ Length::px(MATERIAL_ROW_HEIGHT) } align:center
            gap:8px pad:(left:5px right:10px top:4px bottom:4px) radius:0px
            stroke:(width:0px color:bar.edge)
            fill:{ if highlighted.get() == Some(material) {
                color(bar.slot_over)
            } else {
                color(bar.slot)
            } } {
            (material_thumbnail(material))
            text #mechanic.caption font-color:ink.fg text-wrap:none { material.label() }
        }
    }
}

fn material_at(position: Vector2, slot: Rect) -> Option<ConstructionMaterial> {
    let left = slot.origin.x + (slot.size.width - MATERIAL_MENU_WIDTH) * 0.5;
    let top = slot.origin.y - MATERIAL_MENU_GAP - MATERIAL_MENU_HEIGHT;
    if position.x < left
        || position.x > left + MATERIAL_MENU_WIDTH
        || position.y < top
        || position.y >= top + MATERIAL_MENU_HEIGHT
    {
        return None;
    }
    let mut row_top = top;
    for material in ConstructionMaterial::ALL {
        let row_bottom = row_top + MATERIAL_ROW_HEIGHT;
        if position.y < row_bottom {
            return Some(material);
        }
        row_top = row_bottom;
    }
    None
}

pub(crate) fn material_thumbnail(material: ConstructionMaterial) -> Element {
    match material {
        ConstructionMaterial::Aluminium => view! {
            el width:26px height:26px radius:4px clip {
                img fit:cover width:fill height:fill
                    "assets/materials/aluminium/aluminium_thumbnail.png"
            }
        },
        ConstructionMaterial::Graphite => view! {
            el width:26px height:26px radius:4px clip {
                img fit:cover width:fill height:fill
                    "assets/materials/graphite/graphite_thumbnail.png"
            }
        },
        ConstructionMaterial::CarbonFiber => view! {
            el width:26px height:26px radius:4px clip {
                img fit:cover width:fill height:fill
                    "assets/materials/carbon_fiber/carbon_fiber_thumbnail.png"
            }
        },
        ConstructionMaterial::Concrete => view! {
            el width:26px height:26px radius:4px clip {
                img fit:cover width:fill height:fill
                    "assets/materials/concrete/concrete_thumbnail.png"
            }
        },
        ConstructionMaterial::Dirt => view! {
            el width:26px height:26px radius:4px clip {
                img fit:cover width:fill height:fill "assets/materials/dirt/dirt_thumbnail.png"
            }
        },
        ConstructionMaterial::Iron => view! {
            el width:26px height:26px radius:4px clip {
                img fit:cover width:fill height:fill "assets/materials/iron/iron_thumbnail.png"
            }
        },
        ConstructionMaterial::Plastic => view! {
            el width:26px height:26px radius:4px clip {
                img fit:cover width:fill height:fill
                    "assets/materials/plastic/plastic_thumbnail.png"
            }
        },
        ConstructionMaterial::Rubber => view! {
            el width:26px height:26px radius:4px clip {
                img fit:cover width:fill height:fill "assets/materials/rubber/rubber_thumbnail.png"
            }
        },
        ConstructionMaterial::Sand => view! {
            el width:26px height:26px radius:4px clip {
                img fit:cover width:fill height:fill "assets/materials/sand/sand_thumbnail.png"
            }
        },
        ConstructionMaterial::Steel => view! {
            el width:26px height:26px radius:4px clip {
                img fit:cover width:fill height:fill "assets/materials/steel/steel_thumbnail.png"
            }
        },
        ConstructionMaterial::Stone => view! {
            el width:26px height:26px radius:4px clip {
                img fit:cover width:fill height:fill "assets/materials/stone/stone_thumbnail.png"
            }
        },
        ConstructionMaterial::Wood => view! {
            el width:26px height:26px radius:4px clip {
                img fit:cover width:fill height:fill "assets/materials/wood/wood_thumbnail.png"
            }
        },
    }
}

/// What one tool looks like.
///
/// A fixed frame, because everything inside it is placed by coordinate: an
/// unsized canvas shrinks onto its own drawing and slides it into the corner.
#[allow(clippy::too_many_lines)] // Nine drawings, each a short list of coordinates.
pub(super) fn icon(tool: Tool) -> Element {
    match tool {
        Tool::Block => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                rect at:(x:23px y:16px) size:(w:22px h:22px) stroke:(width:2px color:ink.muted)
                rect at:(x:18.5px y:24px) size:(w:25px h:24px) fill:accent.speed
                    stroke:(width:2px color:ink.fg)
            }
        },
        Tool::Cylinder => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                rect at:(x:20px y:20px) size:(w:26px h:30px) radius:13px fill:accent.speed
                    stroke:(width:2px color:ink.fg)
                rect at:(x:20px y:20px) size:(w:10px h:16px) radius:5px fill:bar.slot
            }
        },
        Tool::Bearing => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                circle at:(x:20px y:20px) radius:12.5px stroke:(width:7px color:accent.angle)
            }
        },
        Tool::Weld => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                line from:(x:10.58px y:26.1px) to:(x:29.42px y:12.9px)
                    stroke:(width:7px cap:round color:ink.fg)
                line from:(x:10.58px y:12.9px) to:(x:29.42px y:26.1px)
                    stroke:(width:7px cap:round color:ink.fg)
                circle at:(x:20px y:20px) radius:4px fill:accent.angle
            }
        },
        // The handle is square to the head, which is what reads as a hammer:
        // both bars turn by the same angle, but one starts across and the other
        // down, so the turn leaves them perpendicular.
        Tool::Hammer => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                line from:(x:15.81px y:17.67px) to:(x:25.19px y:35.33px)
                    stroke:(width:7px cap:round color:dial.grip)
                line from:(x:11.61px y:14.96px) to:(x:28.39px y:6.04px)
                    stroke:(width:11px cap:round color:ink.fg)
            }
        },
        Tool::Controller => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                rect at:(x:20px y:20px) size:(w:26px h:26px) radius:4px fill:accent.key
                    stroke:(width:2px color:ink.fg)
                circle at:(x:20px y:20px) radius:5px fill:bar.slot
            }
        },
        Tool::Connector => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                line from:(x:7.02px y:25.24px) to:(x:32.98px y:14.76px)
                    stroke:(width:4px cap:round color:accent.key)
                rect at:(x:8.5px y:28.5px) size:(w:11px h:11px) radius:3px fill:accent.key
                circle at:(x:31px y:11px) radius:4.5px stroke:(width:3px color:accent.angle)
            }
        },
        Tool::GasEngine => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                rect at:(x:20px y:20px) size:(w:30px h:24px) radius:3px fill:bar.slot
                    stroke:(width:2px color:accent.angle)
                circle at:(x:12px y:20px) radius:5px stroke:(width:2px color:accent.angle)
                line from:(x:20px y:13px) to:(x:31px y:13px)
                    stroke:(width:3px cap:round color:ink.fg)
                line from:(x:20px y:20px) to:(x:31px y:20px)
                    stroke:(width:3px cap:round color:ink.fg)
                line from:(x:20px y:27px) to:(x:31px y:27px)
                    stroke:(width:3px cap:round color:ink.fg)
            }
        },
        Tool::ElectricEngine => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                rect at:(x:20px y:20px) size:(w:28px h:28px) radius:4px fill:bar.slot
                    stroke:(width:2px color:accent.speed)
                circle at:(x:20px y:20px) radius:9px stroke:(width:3px color:accent.speed)
                line from:(x:20px y:11px) to:(x:20px y:29px)
                    stroke:(width:3px cap:round color:ink.fg)
            }
        },
        Tool::Transmission => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                rect at:(x:20px y:20px) size:(w:30px h:22px) radius:3px fill:bar.slot
                    stroke:(width:2px color:accent.speed)
                circle at:(x:15px y:20px) radius:6px stroke:(width:3px color:accent.angle)
                circle at:(x:26px y:20px) radius:6px stroke:(width:3px color:accent.speed)
            }
        },
        Tool::Servo => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                rect at:(x:20px y:20px) size:(w:27px h:27px) radius:4px fill:bar.slot
                    stroke:(width:2px color:accent.key)
                circle at:(x:20px y:20px) radius:8px stroke:(width:3px color:accent.angle)
                circle at:(x:20px y:20px) radius:2px fill:ink.fg
            }
        },
        Tool::Seat => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                rect at:(x:20px y:17px) size:(w:29px h:12px) radius:5px fill:accent.speed
                    stroke:(width:2px color:ink.fg)
                line from:(x:10px y:23px) to:(x:10px y:28px)
                    stroke:(width:3px cap:round color:ink.fg)
                line from:(x:30px y:23px) to:(x:30px y:28px)
                    stroke:(width:3px cap:round color:ink.fg)
            }
        },
        Tool::Input => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                rect at:(x:20px y:20px) size:(w:31px h:21px) radius:4px fill:bar.slot
                    stroke:(width:2px color:accent.key)
                circle at:(x:12px y:20px) radius:2.5px fill:accent.key
                circle at:(x:20px y:20px) radius:2.5px fill:accent.key
                circle at:(x:28px y:20px) radius:2.5px fill:accent.key
            }
        },
        // A block with one corner pulled away, and handles on the corners that
        // move: the wedge is what shaping is for.
        Tool::Shape => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                line from:(x:10px y:30px) to:(x:30px y:30px) stroke:(width:2px color:ink.fg)
                line from:(x:10px y:30px) to:(x:10px y:14px) stroke:(width:2px color:ink.fg)
                line from:(x:30px y:30px) to:(x:10px y:14px) stroke:(width:2px color:accent.key)
                circle at:(x:10px y:14px) radius:3.5px fill:accent.key
                circle at:(x:30px y:30px) radius:3px fill:ink.fg
                circle at:(x:10px y:30px) radius:3px fill:ink.fg
            }
        },
    }
}

/// A mound under a hovering brush, unique to terrain sculpting.
fn terrain_icon() -> Element {
    view! {
        canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
            line from:(x:7px y:30px) to:(x:14px y:23px)
                stroke:(width:4px cap:round color:accent.angle)
            line from:(x:14px y:23px) to:(x:22px y:27px)
                stroke:(width:4px cap:round color:accent.angle)
            line from:(x:22px y:27px) to:(x:33px y:18px)
                stroke:(width:4px cap:round color:accent.angle)
            line from:(x:10px y:33px) to:(x:32px y:33px)
                stroke:(width:4px cap:round color:ink.fg)
            circle at:(x:25px y:10px) radius:5px fill:accent.key
            line from:(x:25px y:15px) to:(x:25px y:20px)
                stroke:(width:2px cap:round color:accent.key)
        }
    }
}

#[cfg(test)]
mod consolidated_tests {
    use mechanic_core::ConstructionMaterial;
    use mosaic_core::{Rect, Vector2};
    use mosaic_widgets::input::{PointerButton, PointerEventKind};

    use super::{
        MATERIAL_MENU_GAP, MATERIAL_MENU_HEIGHT, MATERIAL_ROW_HEIGHT, SLOT, hotbar_shortcut_label,
    };
    use crate::{
        hotbar::{MainTool, MatterMode, SelectedTool},
        ui::{UiIntent, testing::Overlay},
    };

    fn slot_rows(overlay: &Overlay) -> (Vec<Rect>, Vec<Rect>) {
        let mut slots = overlay
            .reachable_boxes()
            .into_iter()
            .filter(|rect| {
                (rect.size.width - SLOT).abs() < 0.5 && (rect.size.height - SLOT).abs() < 0.5
            })
            .collect::<Vec<_>>();
        slots.sort_by(|left, right| {
            left.origin
                .y
                .total_cmp(&right.origin.y)
                .then(left.origin.x.total_cmp(&right.origin.x))
        });
        let main_y = slots
            .iter()
            .map(|slot| slot.origin.y)
            .max_by(f32::total_cmp)
            .expect("hotbar slots exist");
        let main = slots
            .iter()
            .copied()
            .filter(|slot| (slot.origin.y - main_y).abs() < 0.5)
            .collect();
        let modes = slots
            .into_iter()
            .filter(|slot| (slot.origin.y - main_y).abs() >= 0.5)
            .collect();
        (main, modes)
    }

    #[test]
    fn matter_manipulator_shows_five_modes_above_four_main_tools() {
        let overlay = Overlay::mount();
        let (main, modes) = slot_rows(&overlay);
        assert_eq!(main.len(), MainTool::ALL.len());
        assert_eq!(modes.len(), MatterMode::ALL.len());
        assert!(modes.iter().all(|mode| mode.max_y() < main[0].origin.y));
    }

    #[test]
    fn matter_mode_shortcuts_omit_the_shift_prefix() {
        assert_eq!(hotbar_shortcut_label("Shift+4".to_owned()), "4");
        assert_eq!(hotbar_shortcut_label("Unbound".to_owned()), "Unbound");
    }

    #[test]
    fn main_tools_and_matter_modes_are_clickable_in_order() {
        let overlay = Overlay::mount();
        let (main, modes) = slot_rows(&overlay);
        for (slot, tool) in main.into_iter().zip(MainTool::ALL) {
            overlay.click(slot.center());
            assert_eq!(overlay.intents(), vec![UiIntent::Tool(tool)]);
        }
        for (slot, matter_mode) in modes.into_iter().zip(MatterMode::ALL) {
            overlay.click(slot.center());
            assert_eq!(overlay.intents(), vec![UiIntent::MatterMode(matter_mode)]);
        }
    }

    #[test]
    fn secondary_row_is_hidden_for_other_main_tools() {
        let overlay = Overlay::mount();
        let mut selected = SelectedTool::default();
        selected.select_tool(MainTool::Welder);
        overlay.handles.hotbar.set(selected);
        overlay.settle();
        let (main, modes) = slot_rows(&overlay);
        assert_eq!(main.len(), MainTool::ALL.len());
        assert!(modes.is_empty());
    }

    #[test]
    fn block_and_cylinder_modes_keep_the_drag_material_picker() {
        for (index, matter_mode) in [MatterMode::Block, MatterMode::Cylinder]
            .into_iter()
            .enumerate()
        {
            let overlay = Overlay::mount();
            let slot = slot_rows(&overlay).1[index];
            overlay.dispatch(
                PointerEventKind::Down(PointerButton::Primary),
                slot.center(),
            );
            assert_eq!(
                overlay.handles.material_menu.get_untracked(),
                Some(matter_mode)
            );

            let row = ConstructionMaterial::ALL
                .iter()
                .position(|material| *material == ConstructionMaterial::Concrete)
                .expect("Concrete is selectable");
            let position = Vector2::new(
                slot.center().x,
                slot.origin.y - MATERIAL_MENU_GAP - MATERIAL_MENU_HEIGHT
                    + MATERIAL_ROW_HEIGHT
                        * (f32::from(u16::try_from(row).expect("material row fits u16")) + 0.5),
            );
            overlay.dispatch(PointerEventKind::Move, position);
            overlay.dispatch(PointerEventKind::Up(PointerButton::Primary), position);
            assert_eq!(
                overlay.intents(),
                vec![UiIntent::MaterialMode(
                    ConstructionMaterial::Concrete,
                    matter_mode
                )]
            );
        }
    }
}
