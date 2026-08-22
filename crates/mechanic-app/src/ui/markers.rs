//! The floating number over each driven joint.
//!
//! Screen-space chips rather than meshes, so they stay upright and legible at
//! any camera angle. The projection stays in the ECS — it needs the camera —
//! and only the result, a point and a number, reaches the tree.

#![allow(clippy::wildcard_imports)] // Mosaic's authoring vocabulary is meant to be globbed.

use bevy::math::{Vec2, Vec3};
use bevy::prelude::{Camera, GlobalTransform};
use bevy_mosaic::ui::*;
use mosaic_macros::view;

use super::Handles;
#[allow(clippy::wildcard_imports)] // The design tokens are read as bare names.
use super::theme::*;
use crate::hotbar::Tool;
use crate::{AppSimulation, EditorGraph};

/// Width of one chip. Fixed rather than hugging its digits: a lane holds at
/// most eight states, so two digits is the widest it goes, and a constant size
/// is what lets the chip centre itself on its joint with no measuring.
const CHIP_W: f32 = 30.0;

/// Height of one chip.
const CHIP_H: f32 = 26.0;

/// One joint's number, where it lands on screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Marker {
    /// Which joint of its control block this is.
    number: usize,
    /// Where it sits, in window coordinates.
    at: Vec2,
}

/// Which joints want a number, and where they are in the world.
///
/// Kept apart from the projection so what is numbered can be checked without a
/// camera: whether a joint is labelled at all is a question about the machine,
/// not about where it happens to be on screen.
pub(crate) fn wanted(
    graph: &EditorGraph,
    simulation: &AppSimulation,
    tool: Tool,
) -> Vec<(usize, Vec3)> {
    if !crate::drive_xray_is_visible(tool, crate::driven_bearing_count(&graph.0)) {
        return Vec::new();
    }
    match (simulation.creation.as_ref(), simulation.is_running()) {
        (Some(creation), true) => crate::joint_number_labels(&graph.0, |bearing| {
            crate::simulation_bearing_pose(&graph.0, creation, &simulation.transforms, bearing)
                .map(|(anchor, _)| anchor)
        }),
        _ => crate::joint_number_labels(&graph.0, |bearing| Some(bearing.shared_anchor)),
    }
}

/// Projects every wanted number onto the screen.
///
/// A joint behind the camera is dropped rather than hidden: the list is what is
/// on screen, so there is nothing to pool and nothing stale to pin to an edge.
pub(crate) fn capture(
    graph: &EditorGraph,
    simulation: &AppSimulation,
    tool: Tool,
    camera: &(&Camera, &GlobalTransform),
) -> Vec<Marker> {
    let (camera, transform) = *camera;
    wanted(graph, simulation, tool)
        .into_iter()
        .filter_map(|(number, anchor)| {
            let at = camera.world_to_viewport(transform, anchor).ok()?;
            Some(Marker { number, at })
        })
        .collect()
}

/// Every chip currently on screen.
pub(crate) fn view(handles: &Handles) -> Element {
    let markers = handles.markers;
    let count = move || markers.with(Vec::len);
    view! {
        stack width:fill height:fill align:start justify:start nohit {
            for (index, ()) in { (0..count()).map(|index| (index, ())) } {
                (chip(markers, *index))
            }
        }
    }
}

/// One chip, read out of the list rather than handed its contents, so a camera
/// move re-evaluates a binding instead of rebuilding every chip.
fn chip(markers: State<Vec<Marker>>, index: usize) -> Element {
    let found = move || markers.with(|list| list.get(index).copied());
    let at = move || {
        let marker = found().unwrap_or(Marker {
            number: 0,
            at: Vec2::ZERO,
        });
        (
            Length::px(marker.at.x - CHIP_W / 2.0),
            Length::px(marker.at.y - CHIP_H / 2.0),
        )
    };
    let numeral = move || found().map_or_else(String::new, |marker| marker.number.to_string());
    view! {
        col width:{ Length::px(CHIP_W) } height:{ Length::px(CHIP_H) }
            align:center justify:center radius:13px
            translate:(x:{ at().0 } y:{ at().1 })
            fill:port.fill stroke:(width:2px color:accent.key) {
            text font-size:15px font-weight:700 font-color:ink.fg { numeral() }
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy::math::Vec2;

    use super::{CHIP_H, CHIP_W, Marker};
    use crate::ui::testing::Overlay;

    /// Two joints, numbered, somewhere on screen.
    fn two() -> Vec<Marker> {
        vec![
            Marker {
                number: 1,
                at: Vec2::new(400.0, 300.0),
            },
            Marker {
                number: 2,
                at: Vec2::new(900.0, 500.0),
            },
        ]
    }

    #[test]
    fn one_chip_lands_centred_on_each_joint() {
        let overlay = Overlay::mount();
        overlay.handles.markers.set(two());
        overlay.settle();

        let mut chips: Vec<_> = overlay
            .rects()
            .into_iter()
            .filter(|(_, rect)| {
                (rect.size.width - CHIP_W).abs() < 0.5 && (rect.size.height - CHIP_H).abs() < 0.5
            })
            .map(|(_, rect)| rect)
            .collect();
        chips.sort_by(|left, right| left.origin.x.total_cmp(&right.origin.x));
        assert_eq!(chips.len(), 2, "one chip per joint");
        for (chip, marker) in chips.iter().zip(two()) {
            let centre = chip.center();
            assert!(
                (centre.x - marker.at.x).abs() < 0.5 && (centre.y - marker.at.y).abs() < 0.5,
                "the chip for joint {} sits at {centre:?}, not on {:?}",
                marker.number,
                marker.at,
            );
        }
    }

    /// The chips are a drawing over the world, not something to click: a
    /// pointer landing on one belongs to the machine underneath.
    #[test]
    fn a_chip_does_not_take_the_pointer_from_the_world() {
        let overlay = Overlay::mount();
        overlay.handles.markers.set(two());
        overlay.settle();
        assert!(!overlay.wants_pointer_at(mosaic_core::Vector2::new(400.0, 300.0)));
    }

    #[test]
    fn joints_leaving_the_screen_take_their_chips_with_them() {
        let overlay = Overlay::mount();
        overlay.handles.markers.set(two());
        overlay.settle();
        let showing = overlay.element_count();

        overlay.handles.markers.set(Vec::new());
        overlay.settle();
        assert!(overlay.element_count() < showing);
    }
}
