//! Small composition primitives shared by the overlay's panels.

#![allow(clippy::wildcard_imports)] // Mosaic's authoring vocabulary is meant to be globbed.

use bevy_mosaic::ui::*;
use mosaic_macros::{component, view};

#[allow(unused_imports)] // Style constants are consumed by `view!` expansion.
use super::styles::*;
#[allow(clippy::wildcard_imports)] // Design tokens are read as bare names.
use super::theme::*;

/// A standard overlay surface. Callers own its size and placement.
#[component]
pub(crate) fn PanelSurface(elevated: bool, children: Children) -> Element {
    if elevated {
        view! {
            col #mechanic.panel #mechanic.elevated font-color:ink.fg {
                children
            }
        }
    } else {
        view! {
            col #mechanic.panel font-color:ink.fg {
                children
            }
        }
    }
}

/// A simple click-only action with button semantics but no keyboard focus.
#[component]
pub(crate) fn Action(label: String, on_click: impl Fn() + 'static, children: Children) -> Element {
    view! {
        col #mechanic.action role:button label:(label) @click:{ on_click() } {
            children
        }
    }
}

/// A non-interactive badge whose caller retains size and placement control.
#[component]
pub(crate) fn OverlayBadge(children: Children) -> Element {
    view! {
        col #mechanic.badge nohit {
            children
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy_mosaic::ui::*;
    use mosaic_macros::view;

    use super::{
        Action, ActionProps, OverlayBadge, OverlayBadgeProps, PanelSurface, PanelSurfaceProps,
    };
    use crate::ui::{styles::*, theme};

    #[test]
    fn shared_components_compose_caller_styles() {
        mosaic_core::builtins::install();
        theme::install();
        let ui = Ui::new();
        let _ambient = ui.enter();
        ui.mount(&view! {
            col width:min-content height:min-content {
                PanelSurface elevated:true width:123px height:40px radius:5px exponent:1 {
                    text "panel"
                }
                Action #mechanic.action-danger label:"Remove" on-click:(|| {}) width:91px
                    height:31px {
                    text "action"
                }
                OverlayBadge width:37px height:19px radius:4px exponent:1 {
                    text "7"
                }
            }
        });
        ui.frame(Size::new(320.0, 200.0), 1.0).expect("frame");

        let scene = ui.scene();
        let shapes: Vec<_> = scene
            .cmds
            .iter()
            .filter_map(|cmd| match cmd {
                PaintCmd::Shape(shape) => Some(shape),
                _ => None,
            })
            .collect();
        let panel = shapes
            .iter()
            .find(|shape| (shape.rect.size.width - 123.0).abs() < 0.5)
            .expect("panel surface");
        assert!(
            (panel.radii.tl - 5.0).abs() < f32::EPSILON,
            "caller radius wins",
        );
        assert_eq!(panel.shadows.len(), 1, "elevation still composes");
        assert!(shapes.iter().any(|shape| {
            (shape.rect.size.width - 91.0).abs() < 0.5
                && (shape.rect.size.height - 31.0).abs() < 0.5
        }));
        assert!(shapes.iter().any(|shape| {
            (shape.rect.size.width - 37.0).abs() < 0.5
                && (shape.rect.size.height - 19.0).abs() < 0.5
                && (shape.radii.tl - 4.0).abs() < 0.5
        }));
        let action = ui
            .inspection_snapshot()
            .nodes
            .into_iter()
            .find(|node| node.label.as_deref() == Some("Remove"))
            .expect("semantic action");
        assert_eq!(action.role, Role::Button);
    }
}
