//! Screen-upright dimensions for the live block-sheet preview.

#![allow(clippy::wildcard_imports)] // Mosaic's authoring vocabulary is meant to be globbed.

use bevy::math::{Vec2, Vec3};
use bevy::prelude::{Camera, GlobalTransform};
use bevy_mosaic::ui::*;
use mosaic_macros::view;

use super::Handles;
#[allow(clippy::wildcard_imports)] // The design tokens are read as bare names.
use super::theme::*;
use crate::{AppSimulation, EditorState, block_sheet_bounds};

const LABEL_W: f32 = 132.0;
const LABEL_H: f32 = 28.0;
const LABEL_OUTSIDE_GAP: f32 = 10.0;
const SUMMARY_W: f32 = 300.0;
const SUMMARY_H: f32 = 102.0;
const SUMMARY_MARGIN: f32 = 18.0;

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Model {
    edges: Vec<EdgeLabel>,
    summary: Option<Summary>,
}

impl Model {
    fn is_visible(&self) -> bool {
        self.summary.is_some()
    }
}

#[derive(Clone, Debug, PartialEq)]
struct EdgeLabel {
    at: Vec2,
    text: String,
}

#[derive(Clone, Debug, PartialEq)]
struct Summary {
    counts: String,
    metres: String,
    /// Which plane the pointer is driving, so `Q` reports what it changed to.
    plane: String,
}

/// Captures a complete overlay only when every sheet edge can be projected.
pub(crate) fn capture(
    state: &EditorState,
    simulation: &AppSimulation,
    camera: &(&Camera, &GlobalTransform),
) -> Model {
    if simulation.is_running() {
        return Model::default();
    }
    let (camera, transform) = *camera;
    // An area drag measures a volume; a block drag measures the sheet it is
    // filling. Only one of the two can be open at a time.
    if let Some(drag) = state.region_drag.as_ref() {
        let (low, high) = drag.region.bounds_steps();
        return capture_box(
            low.as_vec3() * mechanic_core::STEP_METERS,
            high.as_vec3() * mechanic_core::STEP_METERS,
            drag.region.size_cells(),
            drag.plane,
            |point| camera.world_to_viewport(transform, point).ok(),
        );
    }
    let Some(drag) = state.block_drag.as_ref() else {
        return Model::default();
    };
    if drag.specs.len() <= 1
        || matches!(
            drag.error,
            Some(crate::PlacementError::DragPlaneUnavailable)
        )
    {
        return Model::default();
    }
    capture_sheet(&drag.specs, drag.plane, |point| {
        camera.world_to_viewport(transform, point).ok()
    })
}

/// Labels all three edges of a box and summarises it, so a dragged area reports
/// how big it is without the user counting blocks.
fn capture_box(
    minimum: Vec3,
    maximum: Vec3,
    counts: bevy::math::IVec3,
    plane: crate::PlacementPlane,
    mut project: impl FnMut(Vec3) -> Option<Vec2>,
) -> Model {
    let Some(centre) = project((minimum + maximum) * 0.5) else {
        return Model::default();
    };
    let lengths = maximum - minimum;
    let mut edges = Vec::with_capacity(3);
    for axis in 0..3 {
        let (first, second) = ((axis + 1) % 3, (axis + 2) % 3);
        // Of the four parallel edges, the one furthest out on screen is the one
        // a label can sit beside without landing on top of the box.
        let mut furthest: Option<(f32, Vec2, Vec2, Vec2)> = None;
        for (high_first, high_second) in
            [(false, false), (true, false), (true, true), (false, true)]
        {
            let mut from = Vec3::ZERO;
            from[first] = if high_first {
                maximum[first]
            } else {
                minimum[first]
            };
            from[second] = if high_second {
                maximum[second]
            } else {
                minimum[second]
            };
            let mut to = from;
            from[axis] = minimum[axis];
            to[axis] = maximum[axis];
            let (Some(from), Some(to)) = (project(from), project(to)) else {
                continue;
            };
            let middle = (from + to) * 0.5;
            let distance = middle.distance(centre);
            if furthest.is_none_or(|(best, ..)| distance > best) {
                furthest = Some((distance, middle, from, to));
            }
        }
        let Some((_, middle, from, to)) = furthest else {
            return Model::default();
        };
        let Some(at) = outside_edge_label(middle, from, to, centre) else {
            return Model::default();
        };
        edges.push(EdgeLabel {
            at,
            text: format!("{} · {:.2} m", counts[axis], lengths[axis]),
        });
    }
    Model {
        edges,
        summary: Some(Summary {
            counts: format!(
                "{} × {} × {} = {}",
                counts.x,
                counts.y,
                counts.z,
                counts.element_product()
            ),
            metres: format!(
                "{:.2} × {:.2} × {:.2} m = {:.3} m³",
                lengths.x,
                lengths.y,
                lengths.z,
                lengths.x * lengths.y * lengths.z
            ),
            plane: format!("{} plane · Q rotates", plane.label()),
        }),
    }
}

fn capture_sheet(
    specs: &[mechanic_core::CuboidSpec],
    plane: crate::PlacementPlane,
    mut project: impl FnMut(Vec3) -> Option<Vec2>,
) -> Model {
    let Some((minimum, maximum)) = block_sheet_bounds(specs) else {
        return Model::default();
    };
    let axes = plane.tangent_axes();
    let lengths = axes.map(|axis| maximum[axis] - minimum[axis]);
    let block_half_units = i32::from(specs[0].dimensions[0].units()) * 2;
    let counts = axes.map(|axis| {
        let (minimum, maximum) = specs.iter().fold((i32::MAX, i32::MIN), |bounds, spec| {
            let coordinate = spec.pose.translation_half_units()[axis];
            (bounds.0.min(coordinate), bounds.1.max(coordinate))
        });
        usize::try_from(maximum.saturating_sub(minimum) / block_half_units + 1)
            .expect("block-sheet count fits usize")
    });
    if counts == [1, 1] {
        return Model::default();
    }

    let normal_axis = plane.normal_axis();
    let normal = f32::midpoint(minimum[normal_axis], maximum[normal_axis]);
    let middle = (minimum + maximum) * 0.5;
    let mut points = [Vec3::ZERO; 4];
    points[0][axes[0]] = middle[axes[0]];
    points[0][axes[1]] = minimum[axes[1]];
    points[1][axes[0]] = middle[axes[0]];
    points[1][axes[1]] = maximum[axes[1]];
    points[2][axes[0]] = minimum[axes[0]];
    points[2][axes[1]] = middle[axes[1]];
    points[3][axes[0]] = maximum[axes[0]];
    points[3][axes[1]] = middle[axes[1]];
    for point in &mut points {
        point[normal_axis] = normal;
    }

    let mut corners = [Vec3::ZERO; 4];
    for (corner, [first, second]) in corners.iter_mut().zip([
        [minimum[axes[0]], minimum[axes[1]]],
        [maximum[axes[0]], minimum[axes[1]]],
        [maximum[axes[0]], maximum[axes[1]]],
        [minimum[axes[0]], maximum[axes[1]]],
    ]) {
        corner[axes[0]] = first;
        corner[axes[1]] = second;
        corner[normal_axis] = normal;
    }

    let Some([first, second, third, fourth]) = projected_edge_labels(points, corners, &mut project)
    else {
        return Model::default();
    };
    let text = [
        format!("{} · {:.2} m", counts[0], lengths[0]),
        format!("{} · {:.2} m", counts[1], lengths[1]),
    ];
    Model {
        edges: vec![
            EdgeLabel {
                at: first,
                text: text[0].clone(),
            },
            EdgeLabel {
                at: second,
                text: text[0].clone(),
            },
            EdgeLabel {
                at: third,
                text: text[1].clone(),
            },
            EdgeLabel {
                at: fourth,
                text: text[1].clone(),
            },
        ],
        summary: Some(Summary {
            counts: format!(
                "{} × {} = {}",
                counts[0],
                counts[1],
                counts[0].saturating_mul(counts[1])
            ),
            metres: format!(
                "{:.2} m × {:.2} m = {:.2} m²",
                lengths[0],
                lengths[1],
                lengths[0] * lengths[1]
            ),
            plane: format!("{} plane · Q rotates", plane.label()),
        }),
    }
}

fn projected_edge_labels(
    midpoints: [Vec3; 4],
    corners: [Vec3; 4],
    mut project: impl FnMut(Vec3) -> Option<Vec2>,
) -> Option<[Vec2; 4]> {
    let [Some(first), Some(second), Some(third), Some(fourth)] = midpoints.map(&mut project) else {
        return None;
    };
    let [
        Some(lower_left),
        Some(lower_right),
        Some(upper_right),
        Some(upper_left),
    ] = corners.map(&mut project)
    else {
        return None;
    };
    let centre = (lower_left + lower_right + upper_right + upper_left) * 0.25;
    Some([
        outside_edge_label(first, lower_left, lower_right, centre)?,
        outside_edge_label(second, upper_left, upper_right, centre)?,
        outside_edge_label(third, lower_left, upper_left, centre)?,
        outside_edge_label(fourth, lower_right, upper_right, centre)?,
    ])
}

/// Moves a screen-upright label wholly beyond its projected edge.
fn outside_edge_label(at: Vec2, from: Vec2, to: Vec2, centre: Vec2) -> Option<Vec2> {
    let edge = to - from;
    let length = edge.length();
    if !length.is_finite() || length <= f32::EPSILON {
        return None;
    }
    let tangent = edge / length;
    let mut outward = Vec2::new(-tangent.y, tangent.x);
    if outward.dot(at - centre) < 0.0 {
        outward = -outward;
    }
    let label_radius = outward.x.abs() * LABEL_W * 0.5 + outward.y.abs() * LABEL_H * 0.5;
    Some(at + outward * (label_radius + LABEL_OUTSIDE_GAP))
}

pub(crate) fn view(handles: &Handles) -> Element {
    let dimensions = handles.dimensions;
    let viewport = handles.viewport;
    let count = move || dimensions.with(|model| model.edges.len());
    view! {
        stack width:fill height:fill align:start justify:start nohit {
            for (index, ()) in { (0..count()).map(|index| (index, ())) } {
                (edge(dimensions, *index))
            }
            if dimensions.with(Model::is_visible) {
                (summary(dimensions, viewport))
            }
        }
    }
}

fn edge(dimensions: State<Model>, index: usize) -> Element {
    let found = move || dimensions.with(|model| model.edges.get(index).cloned());
    let at = move || {
        found().map_or((Length::px(0.0), Length::px(0.0)), |label| {
            (
                Length::px(label.at.x - LABEL_W / 2.0),
                Length::px(label.at.y - LABEL_H / 2.0),
            )
        })
    };
    let text = move || found().map_or_else(String::new, |label| label.text);
    view! {
        row width:{ Length::px(LABEL_W) } height:{ Length::px(LABEL_H) }
            align:center justify:center radius:14px
            translate:(x:{ at().0 } y:{ at().1 })
            fill:port.fill stroke:(width:1px color:accent.key) {
            text font-size:14px font-weight:700 text-wrap:none font-color:ink.fg { text() }
        }
    }
}

fn summary(dimensions: State<Model>, viewport: State<Size>) -> Element {
    let counts = move || {
        dimensions.with(|model| {
            model
                .summary
                .as_ref()
                .map_or_else(String::new, |summary| summary.counts.clone())
        })
    };
    let metres = move || {
        dimensions.with(|model| {
            model
                .summary
                .as_ref()
                .map_or_else(String::new, |summary| summary.metres.clone())
        })
    };
    let plane = move || {
        dimensions.with(|model| {
            model
                .summary
                .as_ref()
                .map_or_else(String::new, |summary| summary.plane.clone())
        })
    };
    let at =
        move || Length::px((viewport.get().width - SUMMARY_W - SUMMARY_MARGIN).max(SUMMARY_MARGIN));
    view! {
        col width:{ Length::px(SUMMARY_W) } height:{ Length::px(SUMMARY_H) }
            align:end justify:center gap:2px
            pad:(left:14px right:14px top:8px bottom:8px) radius:8px
            translate:(x:{ at() } y:{ Length::px(SUMMARY_MARGIN) })
            fill:shell stroke:(width:1px color:shell-edge) {
            text font-size:26px font-weight:700 text-wrap:none font-color:accent.key { counts() }
            text font-size:14px font-weight:700 text-wrap:none font-color:ink.fg { metres() }
            text font-size:13px font-weight:700 text-wrap:none font-color:accent.speed { plane() }
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy::math::{IVec3, Vec2, Vec3};
    use mechanic_core::{BuildPose, CuboidSpec, GridRotation};

    use super::{
        EdgeLabel, LABEL_H, LABEL_OUTSIDE_GAP, LABEL_W, Model, SUMMARY_H, SUMMARY_MARGIN,
        SUMMARY_W, Summary, capture_box, capture_sheet, outside_edge_label,
    };
    use crate::PlacementPlane;
    use crate::builder::block_sheet_specs;
    use crate::ui::testing::{Overlay, VIEWPORT};

    fn model() -> Model {
        Model {
            edges: [
                Vec2::new(300.0, 300.0),
                Vec2::new(700.0, 300.0),
                Vec2::new(300.0, 600.0),
                Vec2::new(700.0, 600.0),
            ]
            .into_iter()
            .map(|at| EdgeLabel {
                at,
                text: "6 · 1.50 m".to_owned(),
            })
            .collect(),
            summary: Some(Summary {
                counts: "6 × 4 = 24".to_owned(),
                metres: "1.50 m × 1.00 m = 1.50 m²".to_owned(),
                plane: "XZ plane · Q rotates".to_owned(),
            }),
        }
    }

    fn sheet(plane: PlacementPlane, steps: [i32; 2]) -> Vec<CuboidSpec> {
        let start = CuboidSpec::new(
            [1; 3],
            BuildPose::from_half_grid(IVec3::ONE, GridRotation::default()),
        )
        .unwrap();
        let mut endpoint = start.pose.translation_half_units();
        for (axis, step) in plane.tangent_axes().into_iter().zip(steps) {
            endpoint[axis] += step * 2;
        }
        block_sheet_specs(start, endpoint, plane).unwrap()
    }

    #[test]
    fn capture_formats_four_edges_and_plane_ordered_summary() {
        for plane in [PlacementPlane::Xy, PlacementPlane::Xz, PlacementPlane::Yz] {
            let captured = capture_sheet(&sheet(plane, [5, 3]), plane, |point| {
                Some(Vec2::new(
                    point[plane.tangent_axes()[0]] * 100.0 + 500.0,
                    point[plane.tangent_axes()[1]] * 100.0 + 400.0,
                ))
            });
            assert_eq!(captured.edges.len(), 4);
            assert_eq!(captured.edges[0].text, "6 · 1.50 m");
            assert_eq!(captured.edges[1].text, "6 · 1.50 m");
            assert_eq!(captured.edges[2].text, "4 · 1.00 m");
            assert_eq!(captured.edges[3].text, "4 · 1.00 m");
            let summary = captured.summary.unwrap();
            assert_eq!(summary.counts, "6 × 4 = 24");
            assert_eq!(summary.metres, "1.50 m × 1.00 m = 1.50 m²");
            assert_eq!(
                summary.plane,
                format!("{} plane · Q rotates", plane.label())
            );

            let [lower, upper, left, right] = captured.edges.as_slice() else {
                unreachable!()
            };
            assert!(lower.at.y + LABEL_H * 0.5 <= 400.0 - LABEL_OUTSIDE_GAP);
            assert!(upper.at.y - LABEL_H * 0.5 >= 500.0 + LABEL_OUTSIDE_GAP);
            assert!(left.at.x + LABEL_W * 0.5 <= 500.0 - LABEL_OUTSIDE_GAP);
            assert!(right.at.x - LABEL_W * 0.5 >= 650.0 + LABEL_OUTSIDE_GAP);
        }
    }

    #[test]
    fn diagonal_edge_offset_keeps_the_whole_label_outside() {
        let edge_from = Vec2::ZERO;
        let edge_to = Vec2::splat(100.0);
        let midpoint = Vec2::splat(50.0);
        let shape_centre = Vec2::new(50.0, 100.0);
        let label = outside_edge_label(midpoint, edge_from, edge_to, shape_centre).unwrap();
        let outward = (label - midpoint).normalize();
        let radius = outward.x.abs() * LABEL_W * 0.5 + outward.y.abs() * LABEL_H * 0.5;

        assert!(outward.dot(midpoint - shape_centre) > 0.0);
        assert!(((label - midpoint).length() - radius - LABEL_OUTSIDE_GAP).abs() < 1.0e-5);
    }

    #[test]
    fn one_by_one_or_an_unprojectable_edge_has_no_dimensions() {
        assert_eq!(
            capture_sheet(
                &sheet(PlacementPlane::Xz, [0, 0]),
                PlacementPlane::Xz,
                |_| { Some(Vec2::ZERO) }
            ),
            Model::default()
        );
        assert_eq!(
            capture_sheet(
                &sheet(PlacementPlane::Xz, [2, 1]),
                PlacementPlane::Xz,
                |_| None
            ),
            Model::default()
        );
    }

    /// A skewed projection, so the three axes land in three different screen
    /// directions the way a perspective camera makes them.
    fn skewed(point: Vec3) -> Vec2 {
        Vec2::new(
            point.x.mul_add(180.0, point.z * 70.0) + 500.0,
            point.y.mul_add(-180.0, point.z * 40.0) + 400.0,
        )
    }

    #[test]
    fn a_dragged_area_is_labelled_on_all_three_axes() {
        let captured = capture_box(
            Vec3::ZERO,
            Vec3::new(0.75, 0.5, 0.25),
            IVec3::new(3, 2, 1),
            PlacementPlane::Xz,
            |point| Some(skewed(point)),
        );

        let texts: Vec<_> = captured
            .edges
            .iter()
            .map(|edge| edge.text.clone())
            .collect();
        assert_eq!(texts, ["3 · 0.75 m", "2 · 0.50 m", "1 · 0.25 m"]);
        let summary = captured.summary.expect("an area reports its size");
        assert_eq!(summary.counts, "3 × 2 × 1 = 6");
        assert_eq!(summary.metres, "0.75 × 0.50 × 0.25 m = 0.094 m³");
        assert_eq!(summary.plane, "XZ plane · Q rotates");

        // Every label is pushed away from the box rather than sitting on it.
        let centre = skewed(Vec3::new(0.375, 0.25, 0.125));
        for edge in &captured.edges {
            assert!(edge.at.distance(centre) > LABEL_OUTSIDE_GAP);
        }
    }

    #[test]
    fn an_area_the_camera_cannot_project_has_no_dimensions() {
        assert_eq!(
            capture_box(
                Vec3::ZERO,
                Vec3::splat(0.25),
                IVec3::ONE,
                PlacementPlane::Xz,
                |_| None,
            ),
            Model::default()
        );
    }

    #[test]
    fn dimension_labels_do_not_take_world_input() {
        let overlay = Overlay::mount();
        overlay.handles.dimensions.set(model());
        overlay.settle();
        for at in [
            mosaic_core::Vector2::new(300.0, 300.0),
            mosaic_core::Vector2::new(VIEWPORT.width - 30.0, SUMMARY_MARGIN + 20.0),
        ] {
            assert!(!overlay.wants_pointer_at(at));
        }
    }

    #[test]
    fn summary_stays_anchored_to_the_top_right() {
        let overlay = Overlay::mount();
        overlay.handles.dimensions.set(model());
        overlay.settle();
        let summary = overlay
            .rects()
            .into_iter()
            .map(|(_, rect)| rect)
            .find(|rect| {
                (rect.size.width - SUMMARY_W).abs() < 0.5
                    && (rect.size.height - SUMMARY_H).abs() < 0.5
            })
            .expect("dimension summary is visible");
        assert!((summary.origin.x - (VIEWPORT.width - SUMMARY_W - SUMMARY_MARGIN)).abs() < 0.5);
        assert!((summary.origin.y - SUMMARY_MARGIN).abs() < 0.5);
    }
}
