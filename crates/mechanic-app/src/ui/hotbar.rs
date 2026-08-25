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
use mosaic_macros::view;
use mosaic_widgets::input::{EventCtx, PointerButton};

#[allow(clippy::wildcard_imports)] // The design tokens are read as bare names.
use super::theme::*;
use super::{Handles, UiIntent};
use crate::hotbar::Tool;

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

/// The tools, in the order their number keys run.
const TOOLS: [Tool; 14] = [
    Tool::Block,
    Tool::Cylinder,
    Tool::Bearing,
    Tool::Weld,
    Tool::Hammer,
    Tool::Controller,
    Tool::Connector,
    Tool::GasEngine,
    Tool::ElectricEngine,
    Tool::Transmission,
    Tool::Servo,
    Tool::Seat,
    Tool::Input,
    Tool::Shape,
];

/// The bar, and the tooltip that rides above it.
///
/// Placed against the bottom edge by measuring itself: the window's size is
/// pushed in, and a column that hugs its slots is the only one that knows how
/// wide it ended up.
pub(crate) fn view(handles: &Handles) -> Element {
    let viewport = handles.viewport;
    let hovered = handles.hovered;
    let selected_material = handles.material;
    let material_menu = handles.material_menu;
    let slots = handles.clone();
    let size: State<Size> = State::new(Size::ZERO);
    let named = move || {
        hovered.get().map_or_else(String::new, |tool| {
            if matches!(tool, Tool::Block | Tool::Cylinder) {
                format!("{} · {}", tool.label(), selected_material.get().label())
            } else {
                tool.label().to_owned()
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
                row width:max-content height:24px align:center justify:center
                    pad:(left:10px right:12px top:0px bottom:0px) radius:12px
                    fill:port.fill stroke:(width:1px color:accent.key) {
                    text font-size:12px text-wrap:none font-color:accent.key { named() }
                }
            }
            row width:min-content height:min-content align:center gap:8px
                pad:{ Edges::all(Length::px(BAR_PAD)) } radius:{ Length::px(BAR_RADIUS) }
                fill:bar.fill stroke:(width:1px color:shell-edge) {
                for (tool, ()) in { TOOLS.map(|tool| (tool, ())) } {
                    (slot(&slots, *tool))
                }
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
fn slot(handles: &Handles, tool: Tool) -> Element {
    let handles = handles.clone();
    let selection = handles.hotbar;
    let hovered = handles.hovered;
    let menu = handles.material_menu;
    let material_hover = handles.material_hover;
    let material_tool = matches!(tool, Tool::Block | Tool::Cylinder);
    let gesture = handles.clone();
    let held = move || selection.get() == tool;
    let icon = icon(tool);
    view! {
        stack width:{ Length::px(SLOT) } height:{ Length::px(SLOT) }
            align:center justify:center radius:{ Length::px(SLOT_RADIUS) }
            fill:{ if held() { color(bar.slot_on) } else { color(bar.slot) } }
            stroke:(width:1px color:{
                if held() { color(bar.edge_on) } else { color(bar.edge) }
            })
            hover { fill:bar.slot-over stroke:(width:1px color:bar.edge-over) }
            @pointer:{ move |event: &PointerEvent, _: &mut EventCtx| match event.kind {
                PointerEventKind::Enter => hovered.set(Some(tool)),
                // Only when this slot is still the one being named: the pointer
                // enters the next slot before it leaves this one.
                PointerEventKind::Leave if hovered.get_untracked() == Some(tool) => {
                    hovered.set(None);
                }
                _ => {}
            } }
            @pointer:{ move |event: &PointerEvent, ctx: &mut EventCtx| {
                if !material_tool {
                    return;
                }
                match event.kind {
                    PointerEventKind::Down(PointerButton::Primary) => {
                        ctx.claim_pointer();
                        ctx.suppress_click();
                        menu.set(Some(tool));
                        material_hover.set(material_at(event.position, ctx.target_rect()));
                    }
                    PointerEventKind::Move if menu.get_untracked() == Some(tool) => {
                        material_hover.set(material_at(event.position, ctx.target_rect()));
                    }
                    PointerEventKind::Up(PointerButton::Primary)
                        if menu.get_untracked() == Some(tool) => {
                        if let Some(material) = material_at(event.position, ctx.target_rect()) {
                            gesture.ask(UiIntent::MaterialTool(material, tool));
                        } else if ctx.target_rect().contains(event.position) {
                            gesture.ask(UiIntent::Tool(tool));
                        }
                        menu.set(None);
                        material_hover.set(None);
                    }
                    PointerEventKind::Cancel if menu.get_untracked() == Some(tool) => {
                        menu.set(None);
                        material_hover.set(None);
                    }
                    _ => {}
                }
            } }
            @click:{ if !material_tool { handles.ask(UiIntent::Tool(tool)); } } {
            (icon)
            if menu.get() == Some(tool) {
                col width:{ Length::px(MATERIAL_MENU_WIDTH) } height:min-content
                    translate:(x:0px y:{ Length::px(MATERIAL_MENU_OFFSET_Y) })
                    radius:8px clip fill:bar.fill stroke:(width:1px color:shell-edge) {
                    for (material, ()) in { ConstructionMaterial::ALL.map(|material| (material, ())) } {
                        (material_row(*material, material_hover))
                    }
                }
            }
            text font-size:12px font-color:bar.shortcut
                translate:(x:-22px y:-20px) (tool.shortcut())
        }
    }
}

fn material_row(
    material: ConstructionMaterial,
    highlighted: State<Option<ConstructionMaterial>>,
) -> Element {
    view! {
        row width:fill height:{ Length::px(MATERIAL_ROW_HEIGHT) }
            align:center gap:8px pad:(left:5px right:10px top:4px bottom:4px)
            fill:{ if highlighted.get() == Some(material) {
                color(bar.slot_over)
            } else {
                color(bar.slot)
            } } {
            (material_thumbnail(material))
            text font-size:12px font-color:ink.fg text-wrap:none { material.label() }
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

fn material_thumbnail(material: ConstructionMaterial) -> Element {
    match material {
        ConstructionMaterial::Aluminium => view! {
            el width:26px height:26px radius:4px clip {
                img fit:cover width:fill height:fill
                    "assets/materials/aluminium/aluminium_thumbnail.png"
            }
        },
        ConstructionMaterial::Concrete => view! {
            el width:26px height:26px radius:4px clip {
                img fit:cover width:fill height:fill
                    "assets/materials/concrete/concrete_thumbnail.png"
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
                img fit:cover width:fill height:fill
                    "assets/materials/rubber/rubber_thumbnail.png"
            }
        },
        ConstructionMaterial::Steel => view! {
            el width:26px height:26px radius:4px clip {
                img fit:cover width:fill height:fill
                    "assets/materials/steel/steel_thumbnail.png"
            }
        },
        ConstructionMaterial::Wood => view! {
            el width:26px height:26px radius:4px clip {
                img fit:cover width:fill height:fill
                    "assets/materials/wood/wood_thumbnail.png"
            }
        },
    }
}

/// What one tool looks like.
///
/// A fixed frame, because everything inside it is placed by coordinate: an
/// unsized canvas shrinks onto its own drawing and slides it into the corner.
#[allow(clippy::too_many_lines)] // Nine drawings, each a short list of coordinates.
fn icon(tool: Tool) -> Element {
    match tool {
        Tool::Block => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                rect at:(x:23px y:16px) size:(w:22px h:22px)
                    stroke:(width:2px color:ink.muted)
                rect at:(x:18.5px y:24px) size:(w:25px h:24px)
                    fill:accent.speed stroke:(width:2px color:ink.fg)
            }
        },
        Tool::Cylinder => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                rect at:(x:20px y:20px) size:(w:26px h:30px) radius:13px
                    fill:accent.speed stroke:(width:2px color:ink.fg)
                rect at:(x:20px y:20px) size:(w:10px h:16px) radius:5px
                    fill:bar.slot
            }
        },
        Tool::Bearing => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                circle at:(x:20px y:20px) radius:12.5px
                    stroke:(width:7px color:accent.angle)
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
                rect at:(x:20px y:20px) size:(w:26px h:26px) radius:4px
                    fill:accent.key stroke:(width:2px color:ink.fg)
                circle at:(x:20px y:20px) radius:5px fill:bar.slot
            }
        },
        Tool::Connector => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                line from:(x:7.02px y:25.24px) to:(x:32.98px y:14.76px)
                    stroke:(width:4px cap:round color:accent.key)
                rect at:(x:8.5px y:28.5px) size:(w:11px h:11px) radius:3px
                    fill:accent.key
                circle at:(x:31px y:11px) radius:4.5px
                    stroke:(width:3px color:accent.angle)
            }
        },
        Tool::GasEngine => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                rect at:(x:20px y:20px) size:(w:30px h:24px) radius:3px
                    fill:bar.slot stroke:(width:2px color:accent.angle)
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
                rect at:(x:20px y:20px) size:(w:28px h:28px) radius:4px
                    fill:bar.slot stroke:(width:2px color:accent.speed)
                circle at:(x:20px y:20px) radius:9px stroke:(width:3px color:accent.speed)
                line from:(x:20px y:11px) to:(x:20px y:29px)
                    stroke:(width:3px cap:round color:ink.fg)
            }
        },
        Tool::Transmission => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                rect at:(x:20px y:20px) size:(w:30px h:22px) radius:3px
                    fill:bar.slot stroke:(width:2px color:accent.speed)
                circle at:(x:15px y:20px) radius:6px stroke:(width:3px color:accent.angle)
                circle at:(x:26px y:20px) radius:6px stroke:(width:3px color:accent.speed)
            }
        },
        Tool::Servo => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                rect at:(x:20px y:20px) size:(w:27px h:27px) radius:4px
                    fill:bar.slot stroke:(width:2px color:accent.key)
                circle at:(x:20px y:20px) radius:8px stroke:(width:3px color:accent.angle)
                circle at:(x:20px y:20px) radius:2px fill:ink.fg
            }
        },
        Tool::Seat => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                rect at:(x:20px y:17px) size:(w:29px h:12px) radius:5px
                    fill:accent.speed stroke:(width:2px color:ink.fg)
                line from:(x:10px y:23px) to:(x:10px y:28px)
                    stroke:(width:3px cap:round color:ink.fg)
                line from:(x:30px y:23px) to:(x:30px y:28px)
                    stroke:(width:3px cap:round color:ink.fg)
            }
        },
        Tool::Input => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                rect at:(x:20px y:20px) size:(w:31px h:21px) radius:4px
                    fill:bar.slot stroke:(width:2px color:accent.key)
                circle at:(x:12px y:20px) radius:2.5px fill:accent.key
                circle at:(x:20px y:20px) radius:2.5px fill:accent.key
                circle at:(x:28px y:20px) radius:2.5px fill:accent.key
            }
        },
        // A block with one corner pulled away, and handles on the corners that
        // move: the wedge is what shaping is for.
        Tool::Shape => view! {
            canvas width:{ Length::px(ICON) } height:{ Length::px(ICON) } {
                line from:(x:10px y:30px) to:(x:30px y:30px)
                    stroke:(width:2px color:ink.fg)
                line from:(x:10px y:30px) to:(x:10px y:14px)
                    stroke:(width:2px color:ink.fg)
                line from:(x:30px y:30px) to:(x:10px y:14px)
                    stroke:(width:2px color:accent.key)
                circle at:(x:10px y:14px) radius:3.5px fill:accent.key
                circle at:(x:30px y:30px) radius:3px fill:ink.fg
                circle at:(x:10px y:30px) radius:3px fill:ink.fg
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use mechanic_core::ConstructionMaterial;
    use mosaic_core::{Rect, Vector2};
    use mosaic_widgets::input::{PointerButton, PointerEventKind};

    use super::{ICON, MATERIAL_MENU_GAP, MATERIAL_MENU_HEIGHT, MATERIAL_ROW_HEIGHT, SLOT, TOOLS};
    use crate::hotbar::Tool;
    use crate::ui::testing::{Overlay, VIEWPORT};
    use crate::ui::{UiIntent, testing};

    /// Every slot on the bar, left to right.
    fn slots(overlay: &Overlay) -> Vec<Rect> {
        let mut found: Vec<Rect> = overlay
            .reachable_boxes()
            .into_iter()
            .filter(|rect| {
                (rect.size.width - SLOT).abs() < 0.5 && (rect.size.height - SLOT).abs() < 0.5
            })
            .collect();
        found.sort_by(|left, right| left.origin.x.total_cmp(&right.origin.x));
        found
    }

    #[test]
    fn every_tool_has_a_slot_the_pointer_can_reach() {
        let overlay = Overlay::mount();
        assert_eq!(slots(&overlay).len(), TOOLS.len());
    }

    /// The bar places itself against an edge it cannot see, so it has to be
    /// told how big the window is — and measure how big it is itself.
    #[test]
    fn the_bar_sits_along_the_bottom_of_the_window() {
        let overlay = Overlay::mount();
        let slots = slots(&overlay);
        let first = slots.first().expect("the bar is on screen");
        let last = slots.last().expect("the bar is on screen");

        let bottom = last.origin.y + last.size.height;
        assert!(
            (VIEWPORT.height - bottom - super::FLOAT - 8.0).abs() < 1.0,
            "the bar floats above the bottom edge; its slots end at {bottom}",
        );
        let centre = f32::midpoint(first.origin.x, last.origin.x + last.size.width);
        assert!(
            (centre - VIEWPORT.width / 2.0).abs() < 1.0,
            "the bar is centred; it runs to {centre}",
        );
    }

    #[test]
    fn clicking_a_slot_asks_for_that_tool() {
        let overlay = Overlay::mount();
        let slots = slots(&overlay);
        for (index, tool) in TOOLS.into_iter().enumerate() {
            overlay.click(slots[index].center());
            assert_eq!(
                overlay.intents(),
                vec![UiIntent::Tool(tool)],
                "the slot at {index} is {tool:?}",
            );
        }
    }

    #[test]
    fn material_tools_open_on_press_and_select_a_dragged_row() {
        for (slot_index, tool) in [(0, Tool::Block), (1, Tool::Cylinder)] {
            let overlay = Overlay::mount();
            let slot = slots(&overlay)[slot_index];
            overlay.dispatch(
                PointerEventKind::Down(PointerButton::Primary),
                slot.center(),
            );
            assert_eq!(overlay.handles.material_menu.get_untracked(), Some(tool));
            assert_eq!(
                overlay.handles.material.get_untracked(),
                ConstructionMaterial::Steel
            );

            let concrete = Vector2::new(
                slot.center().x,
                slot.origin.y - MATERIAL_MENU_GAP - MATERIAL_MENU_HEIGHT
                    + MATERIAL_ROW_HEIGHT * 1.5,
            );
            overlay.dispatch(PointerEventKind::Move, concrete);
            assert_eq!(
                overlay.handles.material_hover.get_untracked(),
                Some(ConstructionMaterial::Concrete),
            );
            overlay.dispatch(PointerEventKind::Up(PointerButton::Primary), concrete);
            assert_eq!(
                overlay.intents(),
                vec![UiIntent::MaterialTool(ConstructionMaterial::Concrete, tool)],
            );
            assert_eq!(overlay.handles.material_menu.get_untracked(), None);
        }
    }

    #[test]
    fn every_material_has_a_selectable_picker_row() {
        let overlay = Overlay::mount();
        let slot = slots(&overlay)[0];
        let top = slot.origin.y - MATERIAL_MENU_GAP - MATERIAL_MENU_HEIGHT;

        let mut row_center_y = top + MATERIAL_ROW_HEIGHT * 0.5;
        for material in ConstructionMaterial::ALL {
            let row_center = Vector2::new(slot.center().x, row_center_y);
            assert_eq!(super::material_at(row_center, slot), Some(material));
            row_center_y += MATERIAL_ROW_HEIGHT;
        }
    }

    #[test]
    fn material_menu_is_drawn_directly_above_its_slot() {
        let overlay = Overlay::mount();
        let slot = slots(&overlay)[0];
        overlay.dispatch(
            PointerEventKind::Down(PointerButton::Primary),
            slot.center(),
        );
        overlay.settle();

        let menu = overlay
            .rects()
            .into_iter()
            .find(|(_, rect)| {
                (rect.size.width - super::MATERIAL_MENU_WIDTH).abs() < 0.5
                    && (rect.size.height - super::MATERIAL_MENU_HEIGHT).abs() < 0.5
            })
            .expect("the material menu is laid out")
            .1;

        assert!(
            (slot.origin.y - menu.max_y() - MATERIAL_MENU_GAP).abs() < 0.5,
            "menu {menu:?} should sit {MATERIAL_MENU_GAP}px above slot {slot:?}",
        );
    }

    #[test]
    fn releasing_a_material_gesture_on_its_slot_selects_only_the_tool() {
        let overlay = Overlay::mount();
        let slot = slots(&overlay)[0];
        overlay.dispatch(
            PointerEventKind::Down(PointerButton::Primary),
            slot.center(),
        );
        overlay.dispatch(PointerEventKind::Up(PointerButton::Primary), slot.center());
        assert_eq!(overlay.intents(), vec![UiIntent::Tool(Tool::Block)]);
        assert_eq!(
            overlay.handles.material.get_untracked(),
            ConstructionMaterial::Steel
        );
    }

    #[test]
    fn releasing_a_material_gesture_elsewhere_cancels_it() {
        let overlay = Overlay::mount();
        let slot = slots(&overlay)[1];
        overlay.dispatch(
            PointerEventKind::Down(PointerButton::Primary),
            slot.center(),
        );
        overlay.dispatch(
            PointerEventKind::Up(PointerButton::Primary),
            Vector2::new(4.0, 4.0),
        );
        assert!(overlay.intents().is_empty());
        assert_eq!(overlay.handles.material_menu.get_untracked(), None);
    }

    #[test]
    fn block_tool_is_visibly_named_blocker_placer() {
        assert_eq!(Tool::Block.label(), "Blocker Placer");
    }

    /// The tooltip is a sibling of the slots rather than a child, so hover
    /// cannot style it from inside — a slot has to say what it is.
    #[test]
    fn hovering_a_slot_names_the_tool_above_the_bar() {
        let overlay = Overlay::mount();
        let slots = slots(&overlay);
        assert_eq!(overlay.handles.hovered.get_untracked(), None);

        let resting = overlay.element_count();
        // A move rather than an Enter: hovering is something dispatch works out
        // from where the pointer is, not something a caller announces.
        overlay.dispatch(PointerEventKind::Move, slots[2].center());
        assert_eq!(
            overlay.handles.hovered.get_untracked(),
            Some(Tool::Bearing),
            "the third slot is the bearing tool",
        );
        assert!(
            overlay.element_count() > resting,
            "the tooltip appears alongside the bar",
        );

        overlay.dispatch(PointerEventKind::Move, Vector2::new(4.0, 4.0));
        assert_eq!(overlay.handles.hovered.get_untracked(), None);
        assert_eq!(
            overlay.element_count(),
            resting,
            "and goes away with the pointer",
        );
    }

    #[test]
    fn multi_word_tooltips_enclose_their_label() {
        let overlay = Overlay::mount();
        let slots = slots(&overlay);
        let control_block = TOOLS
            .iter()
            .position(|tool| *tool == Tool::Controller)
            .expect("control block is in the hotbar");
        overlay.dispatch(PointerEventKind::Move, slots[control_block].center());

        let tree = overlay.rects();
        let (tooltip_index, (tooltip_depth, tooltip)) = tree
            .iter()
            .enumerate()
            .find(|(_, (_, rect))| {
                (rect.size.height - 24.0).abs() < 0.5 && rect.max_y() < slots[0].origin.y
            })
            .expect("the tooltip is above the hotbar");
        let labels = tree[tooltip_index + 1..]
            .iter()
            .take_while(|(depth, _)| depth > tooltip_depth)
            .map(|(_, rect)| *rect);
        for label in labels {
            assert!(
                label.origin.x >= tooltip.origin.x - 0.5 && label.max_x() <= tooltip.max_x() + 0.5,
                "tooltip label {label:?} overflows {tooltip:?}",
            );
        }
    }

    /// The bug this guards against: an unsized canvas shrinks onto its own
    /// drawing and pulls it flush with the corner, so every icon slides up and
    /// left out of its slot by however much clearance its shapes happened to
    /// leave.
    #[test]
    fn each_icon_is_drawn_from_the_corner_of_its_own_box() {
        let overlay = Overlay::mount();
        let tree = overlay.rects();
        let mut canvases = 0;
        for (index, (slot_depth, slot)) in tree.iter().enumerate() {
            if (slot.size.width - SLOT).abs() > 0.5 {
                continue;
            }
            // The icon's canvas is the first box of its size inside the slot;
            // everything under it is a mark placed from the canvas's corner, so
            // they all share its origin.
            let inside = tree[index + 1..]
                .iter()
                .take_while(|(depth, _)| depth > slot_depth);
            let mut marks = inside.skip_while(|(_, rect)| (rect.size.width - ICON).abs() > 0.5);
            let Some((canvas_depth, canvas)) = marks.next() else {
                continue;
            };
            canvases += 1;
            for (mark_depth, mark) in marks {
                if mark_depth <= canvas_depth {
                    break;
                }
                assert_eq!(
                    mark.origin, canvas.origin,
                    "every mark of an icon is placed from the icon's own corner",
                );
            }
        }
        assert_eq!(canvases, TOOLS.len(), "one icon box per tool");
    }

    /// Each slot paired with the box its icon is drawn inside.
    fn icon_boxes(overlay: &Overlay) -> Vec<(Rect, Rect)> {
        let mut found: Vec<(Rect, Rect)> = Vec::new();
        for slot in slots(overlay) {
            let inside = overlay.rects().into_iter().find(|(_, rect)| {
                (rect.size.width - ICON).abs() < 0.5
                    && (rect.size.height - ICON).abs() < 0.5
                    && slot.intersects(rect)
            });
            found.push((slot, inside.expect("the slot holds an icon").1));
        }
        found
    }

    /// The bug this guards against: both conversions the drawing vocabulary
    /// needs draw something plausible when they are wrong. A placed shape sits
    /// at its centre rather than its near corner, so an icon written from the
    /// corners the old panel used lands half its own size up and to the left;
    /// and a round cap reaches half a stroke past the endpoint, so a bar
    /// written at its full length overhangs by a stroke width. Both leave marks
    /// on screen, so the only thing that catches them is what the ink covers.
    #[test]
    fn every_icon_draws_inside_its_own_box() {
        let overlay = Overlay::mount();
        let ink = overlay.ink();
        let boxes = icon_boxes(&overlay);
        assert_eq!(boxes.len(), TOOLS.len(), "one icon box per tool");

        for (tool, (slot, icon)) in TOOLS.into_iter().zip(boxes) {
            // The slot's own centre, not just the icon's: the shortcut label
            // shares the slot, and a label that takes up room in the flow
            // centres *with* the icon rather than over it.
            let (sits, middle) = (icon.center(), slot.center());
            assert!(
                (sits.x - middle.x).abs() < 0.5 && (sits.y - middle.y).abs() < 0.5,
                "{tool:?} sits at {sits:?} in a slot centred on {middle:?}",
            );

            // A mark belongs to this icon if it is centred in it and is not the
            // chrome around it: the slot and the bar are painted marks too, and
            // both cover the icon they sit behind.
            let mut marks = ink.iter().copied().filter(|mark| {
                icon.contains(mark.center()) && mark.size.width < SLOT && mark.size.height < SLOT
            });
            let first = marks
                .next()
                .unwrap_or_else(|| panic!("{tool:?} draws nothing"));
            let drawn = marks.fold(first, |union, mark| union.union(mark));

            assert!(
                drawn.origin.x >= icon.origin.x - 0.5
                    && drawn.origin.y >= icon.origin.y - 0.5
                    && drawn.max_x() <= icon.max_x() + 0.5
                    && drawn.max_y() <= icon.max_y() + 0.5,
                "{tool:?} draws {drawn:?}, outside its {icon:?}",
            );
            let (centre, wanted) = (drawn.center(), icon.center());
            assert!(
                (centre.x - wanted.x).abs() < 2.0 && (centre.y - wanted.y).abs() < 2.0,
                "{tool:?} draws off-centre, at {centre:?} rather than {wanted:?}",
            );
        }
    }

    #[test]
    fn the_slot_in_hand_reads_differently_from_the_rest() {
        let overlay = Overlay::mount();
        overlay.handles.hotbar.set(Tool::Weld);
        overlay.settle();
        // Nothing to assert about colour without painting a frame; what matters
        // is that the binding runs at all rather than panicking on a freed
        // element, which is what a rebuilt slot would do.
        assert_eq!(slots(&overlay).len(), TOOLS.len());
        let _ = testing::away(slots(&overlay)[0].center(), slots(&overlay)[1].center());
    }
}
