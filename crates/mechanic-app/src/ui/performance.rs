//! Compact, pointer-transparent performance diagnostics.

#![allow(clippy::wildcard_imports)] // Mosaic's authoring vocabulary is meant to be globbed.

use bevy_mosaic::ui::*;
use mechanic_gpu::MAX_CONTACT_PAIRS;
use mosaic_core::theme::color;
use mosaic_macros::{component, view};

use super::components::{PanelSurface, PanelSurfaceProps};
#[allow(unused_imports)] // Style constants are consumed by `view!` expansion.
use super::styles::*;
#[allow(clippy::wildcard_imports)] // Design tokens are read as bare names.
use super::theme::*;
use crate::performance::PerformanceSnapshot;

const PANEL_WIDTH: f32 = 292.0;
const PANEL_INSET: f32 = 16.0;

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Model {
    open: bool,
    frame_rows: Vec<Row>,
    physics_rows: Vec<Row>,
}

impl Model {
    pub(crate) const fn is_open(&self) -> bool {
        self.open
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Row {
    label: &'static str,
    value: String,
    tone: Tone,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
enum Tone {
    #[default]
    Neutral,
    Good,
    Warn,
    Bad,
}

impl Tone {
    fn paint(self) -> Color {
        match self {
            Self::Neutral => color(ink.fg),
            Self::Good => color(status_color.good),
            Self::Warn => color(status_color.warn),
            Self::Bad => color(status_color.bad),
        }
    }
}

pub(crate) fn capture(snapshot: &PerformanceSnapshot) -> Model {
    let timings = snapshot.kernel_timings.unwrap_or_default();
    let has_kernel_timings = snapshot.kernel_timings.is_some();
    Model {
        open: snapshot.open,
        frame_rows: vec![
            rate_row("FPS", snapshot.fps, 55.0, 30.0),
            timing_row("Frame average", snapshot.frame_ms, 16.7, 33.3),
            timing_row("Frame p95", snapshot.frame_p95_ms, 16.7, 33.3),
            timing_row("Render CPU", snapshot.render_cpu_ms, 8.0, 16.7),
            timing_row("Render GPU", snapshot.render_gpu_ms, 8.0, 16.7),
        ],
        physics_rows: vec![
            rate_row(
                "Simulation rate",
                snapshot.simulation_ticks_per_second,
                55.0,
                30.0,
            ),
            timing_row("Physics CPU", snapshot.physics_cpu_ms, 8.0, 16.7),
            timing_row("Physics GPU", snapshot.physics_gpu_ms, 8.0, 16.7),
            timing_row(
                "Contact solver",
                has_kernel_timings.then_some(timings.contact_solver_ms),
                5.0,
                10.0,
            ),
            timing_row(
                "Broadphase",
                has_kernel_timings.then_some(timings.broadphase_ms),
                2.0,
                5.0,
            ),
            timing_row(
                "Narrowphase",
                has_kernel_timings.then_some(timings.narrowphase_ms),
                2.0,
                5.0,
            ),
            timing_row(
                "Integration",
                has_kernel_timings.then_some(timings.integration_ms),
                1.0,
                3.0,
            ),
            count_pair_row(
                "Bodies / colliders",
                snapshot.body_count,
                snapshot.collider_count,
            ),
            capacity_row("Broadphase pairs", snapshot.pair_count),
            contact_row(snapshot.active_contact_count, snapshot.contact_count),
            flags_row(snapshot.error_flags),
            timing_row("Terrain stage", snapshot.terrain_stage_ms, 2.0, 8.0),
            timing_row(
                "Terrain selection",
                snapshot.terrain_selection_ms,
                8.0,
                100.0,
            ),
            timing_row("Terrain sampling", snapshot.terrain_sampling_ms, 2.0, 4.0),
            timing_row(
                "Terrain polygonize",
                snapshot.terrain_polygonization_ms,
                2.0,
                4.0,
            ),
            timing_row("Terrain seams/caps", snapshot.terrain_seams_ms, 1.0, 4.0),
            timing_row("Terrain BVH", snapshot.terrain_bvh_ms, 1.0, 4.0),
            timing_row("Terrain publish", snapshot.terrain_publication_ms, 1.0, 2.0),
            timing_row(
                "Oldest terrain job",
                snapshot.terrain_queue_age_ms,
                16.0,
                100.0,
            ),
            count_pair_row(
                "Local terrain ready",
                snapshot.terrain_local_resolved,
                snapshot.terrain_local_total,
            ),
            count_u64_row("Bounds cache bytes", snapshot.terrain_bounds_cache_bytes),
            count_u64_row("Terrain triangles", snapshot.terrain_triangle_count),
            count_row("Streaming backlog", snapshot.terrain_streaming_backlog),
            count_u64_row("Terrain remeshes", snapshot.terrain_remesh_count),
            flags_row_named("Terrain overflow", snapshot.terrain_overflow_flags),
            timing_row(
                "Foundation refresh",
                snapshot.foundation_refresh_ms,
                1.0,
                2.0,
            ),
            count_u64_row("Foundation candidates", snapshot.foundation_candidate_count),
            count_u64_row("Foundation samples", snapshot.foundation_sample_count),
        ],
    }
}

#[component]
pub(crate) fn PerformanceOverlay(model: State<Model>, viewport: State<Size>) -> Element {
    let at =
        move || Length::px((viewport.get().width - PANEL_WIDTH - PANEL_INSET).max(PANEL_INSET));
    view! {
        stack width:0px height:0px nohit
            translate:(x:{ at() } y:{ Length::px(PANEL_INSET) }) {
            PanelSurface elevated:false width:(Length::px(PANEL_WIDTH)) height:min-content
                gap:5px pad:(left:12px right:12px top:10px bottom:10px) {
                row width:fill height:22px align:center justify:between {
                    text #mechanic.section font-color:accent.speed "PERFORMANCE"
                    text #mechanic.caption "F3"
                }
                (section("FRAME / RENDER"))
                for (row, ()) in {
                    model.get().frame_rows.into_iter().map(|row| (row, ()))
                } {
                    (metric_row(row.clone()))
                }
                (section("PHYSICS"))
                for (row, ()) in {
                    model.get().physics_rows.into_iter().map(|row| (row, ()))
                } {
                    (metric_row(row.clone()))
                }
            }
        }
    }
}

fn section(label: &'static str) -> Element {
    view! {
        text #mechanic.label width:fill pad:(top:5px bottom:1px) (label)
    }
}

fn metric_row(row: Row) -> Element {
    let tone = row.tone.paint();
    view! {
        row width:fill height:17px align:center justify:between {
            text #mechanic.caption font-color:ink.muted (row.label)
            text #mechanic.caption font-color:{ tone } text-wrap:none (row.value)
        }
    }
}

fn rate_row(label: &'static str, value: Option<f64>, good: f64, warn: f64) -> Row {
    Row {
        label,
        value: value.map_or_else(not_available, |value| format!("{value:.1}")),
        tone: value.map_or(Tone::Neutral, |value| {
            if value >= good {
                Tone::Good
            } else if value >= warn {
                Tone::Warn
            } else {
                Tone::Bad
            }
        }),
    }
}

fn timing_row(label: &'static str, value: Option<f64>, good: f64, warn: f64) -> Row {
    Row {
        label,
        value: value.map_or_else(not_available, |value| format!("{value:.2} ms")),
        tone: value.map_or(Tone::Neutral, |value| {
            if value <= good {
                Tone::Good
            } else if value <= warn {
                Tone::Warn
            } else {
                Tone::Bad
            }
        }),
    }
}

fn count_pair_row(label: &'static str, first: Option<u32>, second: Option<u32>) -> Row {
    Row {
        label,
        value: first
            .zip(second)
            .map_or_else(not_available, |(first, second)| {
                format!("{first} / {second}")
            }),
        tone: Tone::Neutral,
    }
}

fn capacity_row(label: &'static str, value: Option<u32>) -> Row {
    let capacity = u32::try_from(MAX_CONTACT_PAIRS).unwrap_or(u32::MAX);
    Row {
        label,
        value: value.map_or_else(not_available, |value| value.to_string()),
        tone: value.map_or(Tone::Neutral, |value| {
            if value >= capacity {
                Tone::Bad
            } else if u64::from(value) * 4 >= u64::from(capacity) * 3 {
                Tone::Warn
            } else {
                Tone::Neutral
            }
        }),
    }
}

fn count_row(label: &'static str, value: Option<u32>) -> Row {
    Row {
        label,
        value: value.map_or_else(not_available, |value| value.to_string()),
        tone: Tone::Neutral,
    }
}

fn count_u64_row(label: &'static str, value: Option<u64>) -> Row {
    Row {
        label,
        value: value.map_or_else(not_available, |value| value.to_string()),
        tone: Tone::Neutral,
    }
}

fn contact_row(active: Option<u32>, generated: Option<u32>) -> Row {
    Row {
        label: "Contacts active / made",
        value: active
            .zip(generated)
            .map_or_else(not_available, |(active, generated)| {
                format!("{active} / {generated}")
            }),
        tone: Tone::Neutral,
    }
}

fn flags_row(flags: Option<u32>) -> Row {
    flags_row_named("Failure flags", flags)
}

fn flags_row_named(label: &'static str, flags: Option<u32>) -> Row {
    Row {
        label,
        value: flags.map_or_else(not_available, |flags| format!("0x{flags:08X}")),
        tone: flags.map_or(Tone::Neutral, |flags| {
            if flags == 0 { Tone::Good } else { Tone::Bad }
        }),
    }
}

fn not_available() -> String {
    "N/A".to_owned()
}

#[cfg(test)]
mod tests {
    use mechanic_gpu::{GpuKernelTimings, MAX_CONTACT_PAIRS};

    use super::{Tone, capture};
    use crate::performance::PerformanceSnapshot;
    use crate::ui::testing::{Overlay, VIEWPORT};

    #[test]
    fn capture_keeps_the_expensive_contact_counters_visible() {
        let model = capture(&PerformanceSnapshot {
            open: true,
            fps: Some(42.0),
            kernel_timings: Some(GpuKernelTimings {
                contact_solver_ms: 12.0,
                ..GpuKernelTimings::default()
            }),
            pair_count: Some(u32::try_from(MAX_CONTACT_PAIRS).unwrap()),
            contact_count: Some(900),
            active_contact_count: Some(700),
            error_flags: Some(0),
            ..PerformanceSnapshot::default()
        });

        assert!(model.open);
        assert_eq!(model.frame_rows[0].value, "42.0");
        assert_eq!(model.physics_rows[3].value, "12.00 ms");
        assert_eq!(model.physics_rows[3].tone, Tone::Bad);
        assert_eq!(model.physics_rows[8].tone, Tone::Bad);
        assert_eq!(model.physics_rows[9].value, "700 / 900");
        assert_eq!(model.physics_rows[10].tone, Tone::Good);
    }

    #[test]
    fn performance_panel_stays_in_the_corner_without_taking_the_pointer() {
        let overlay = Overlay::mount();
        overlay
            .handles
            .performance
            .set(capture(&PerformanceSnapshot {
                open: true,
                fps: Some(60.0),
                ..PerformanceSnapshot::default()
            }));
        overlay.settle();

        assert!(!overlay.wants_pointer_at(mosaic_core::Vector2::new(VIEWPORT.width - 30.0, 30.0,)));
        assert!(overlay.shapes().iter().any(|shape| {
            (shape.rect.size.width - super::PANEL_WIDTH).abs() < 0.5
                && shape.rect.origin.x > VIEWPORT.width / 2.0
        }));
    }
}
