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

const COLUMN_WIDTH: f32 = 268.0;
const PANEL_WIDTH: f32 = COLUMN_WIDTH * 3.0 + 48.0;
const PANEL_INSET: f32 = 16.0;

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Model {
    open: bool,
    frame_rows: Vec<Row>,
    physics_rows: Vec<Row>,
    terrain_rows: Vec<Row>,
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

#[allow(clippy::too_many_lines)]
pub(crate) fn capture(snapshot: &PerformanceSnapshot) -> Model {
    let timings = snapshot.kernel_timings.unwrap_or_default();
    let has_kernel_timings = snapshot.kernel_timings.is_some();
    Model {
        open: snapshot.open,
        frame_rows: vec![
            rate_row("FPS", snapshot.fps, 55.0, 30.0),
            Row {
                label: "Render experiment",
                value: snapshot.render_experiment.label().to_owned(),
                tone: if snapshot.render_experiment
                    == crate::render_experiments::RenderExperiment::Baseline
                {
                    Tone::Neutral
                } else {
                    Tone::Warn
                },
            },
            timing_row("Frame average", snapshot.frame_ms, 16.7, 33.3),
            timing_row("Frame p95", snapshot.frame_p95_ms, 16.7, 33.3),
            timing_row("Window acquire", snapshot.window_acquire_cpu_ms, 8.0, 16.7),
            timing_row("Render/present CPU", snapshot.render_cpu_ms, 8.0, 16.7),
            timing_row("Tracked GPU span", snapshot.render_gpu_ms, 8.0, 16.7),
            Row {
                label: "GPU scope",
                value: "Tracked only".to_owned(),
                tone: Tone::Neutral,
            },
            Row {
                label: "GPU sample",
                value: snapshot.render_health.status.label().to_owned(),
                tone: if snapshot.render_health.status
                    == crate::render_diagnostics::GpuSampleStatus::Complete
                {
                    Tone::Good
                } else {
                    Tone::Warn
                },
            },
            count_pair_row(
                "GPU pairs valid / total",
                Some(snapshot.render_health.valid_pairs),
                Some(snapshot.render_health.total_pairs),
            ),
            extent_row(
                "World target",
                snapshot
                    .render_extent
                    .map(|extent| extent.target.to_array()),
            ),
            extent_row(
                "World viewport",
                snapshot
                    .render_extent
                    .map(|extent| extent.viewport.to_array()),
            ),
            count_row(
                "World MSAA samples",
                snapshot.render_extent.map(|extent| extent.samples),
            ),
            timing_row(
                "Prepass + shadows",
                snapshot.render_breakdown.prepass_ms,
                4.0,
                8.0,
            ),
            timing_row(
                "World opaque",
                snapshot.render_breakdown.opaque_ms,
                4.0,
                8.0,
            ),
            timing_row("World post FX", snapshot.render_breakdown.post_ms, 4.0, 8.0),
            timing_row("Mosaic UI", snapshot.render_breakdown.ui_ms, 2.0, 4.0),
            timing_row(
                "World transparency",
                snapshot.render_breakdown.transparent_ms,
                4.0,
                8.0,
            ),
            timing_row(
                "World output",
                snapshot.render_breakdown.world_output_ms,
                4.0,
                8.0,
            ),
            timing_row("X-ray tracked", snapshot.render_breakdown.xray_ms, 4.0, 8.0),
            timing_row(
                "X-ray output",
                snapshot.render_breakdown.xray_output_ms,
                4.0,
                8.0,
            ),
            timing_row(
                "Other tracked",
                snapshot.render_breakdown.other_ms,
                4.0,
                8.0,
            ),
        ],
        physics_rows: vec![
            rate_row(
                "Simulation rate",
                snapshot.simulation_ticks_per_second,
                55.0,
                30.0,
            ),
            count_u64_row("Physics backlog", snapshot.tick_backlog),
            count_u64_row("Physics ticks dropped", snapshot.dropped_ticks),
            timing_row("Physics CPU", snapshot.physics_cpu_ms, 8.0, 16.7),
            timing_row(
                "CPU encoding",
                snapshot.physics_submission_timings.map(|t| t.encoding_ms),
                2.0,
                8.0,
            ),
            timing_row(
                "CPU finalization",
                snapshot
                    .physics_submission_timings
                    .map(|t| t.finalization_ms),
                2.0,
                8.0,
            ),
            timing_row(
                "CPU queue submit",
                snapshot.physics_submission_timings.map(|t| t.submission_ms),
                2.0,
                8.0,
            ),
            timing_row(
                "CPU readback setup",
                snapshot
                    .physics_submission_timings
                    .map(|t| t.readback_setup_ms),
                2.0,
                8.0,
            ),
            count_row(
                "Ticks submitted / frame",
                snapshot.ticks_submitted_per_frame,
            ),
            count_row("In-flight tick slots", snapshot.in_flight_tick_count),
            timing_row(
                "Submission to readback",
                snapshot.submission_to_readback_ms,
                16.7,
                50.0,
            ),
            timing_row("Visual update", snapshot.visual_update_ms, 2.0, 8.0),
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
            count_pair_row(
                "Solver sweeps",
                snapshot.executed_solver_sweeps,
                snapshot.planned_solver_sweeps,
            ),
            flags_row(snapshot.error_flags),
        ],
        terrain_rows: vec![
            timing_row("Terrain stage", snapshot.terrain_stage_ms, 2.0, 8.0),
            timing_row(
                "Terrain selection worker",
                snapshot.terrain_selection_ms,
                8.0,
                100.0,
            ),
            count_u64_row("Terrain reselections", snapshot.terrain_selection_count),
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
            timing_row(
                "Player collision",
                snapshot.player_collision_query_ms,
                0.25,
                0.5,
            ),
            timing_row(
                "Dynamic index refit",
                snapshot.dynamic_collision_refit_ms,
                2.0,
                4.0,
            ),
            count_row("Player candidates", snapshot.player_collision_candidates),
            count_row("Player contacts", snapshot.player_collision_contacts),
            count_row("Queued reactions", snapshot.player_reaction_impulses),
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
                row gap:12px align:start {
                    col width:(Length::px(COLUMN_WIDTH)) gap:5px {
                        (section("FRAME / RENDER"))
                        for (row, ()) in {
                            model.get().frame_rows.into_iter().map(|row| (row, ()))
                        } {
                            (metric_row(row.clone()))
                        }
                    }
                    col width:(Length::px(COLUMN_WIDTH)) gap:5px {
                        (section("PHYSICS"))
                        for (row, ()) in {
                            model.get().physics_rows.into_iter().map(|row| (row, ()))
                        } {
                            (metric_row(row.clone()))
                        }
                    }
                    col width:(Length::px(COLUMN_WIDTH)) gap:5px {
                        (section("TERRAIN / COLLISION"))
                        for (row, ()) in {
                            model.get().terrain_rows.into_iter().map(|row| (row, ()))
                        } {
                            (metric_row(row.clone()))
                        }
                    }
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

fn extent_row(label: &'static str, value: Option<[u32; 2]>) -> Row {
    Row {
        label,
        value: value.map_or_else(not_available, |[width, height]| {
            format!("{width} × {height}")
        }),
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
    fn unavailable_render_measurements_are_not_reported_as_zero() {
        let model = capture(&PerformanceSnapshot::default());
        for label in [
            "Window acquire",
            "Render/present CPU",
            "Tracked GPU span",
            "World target",
            "World viewport",
            "World MSAA samples",
            "Prepass + shadows",
            "World opaque",
            "World post FX",
            "Mosaic UI",
            "Other tracked",
            "World transparency",
            "World output",
            "X-ray tracked",
            "X-ray output",
        ] {
            let row = model
                .frame_rows
                .iter()
                .find(|row| row.label == label)
                .unwrap();
            assert_eq!(row.value, "N/A");
            assert_eq!(row.tone, Tone::Neutral);
        }
    }

    #[test]
    fn partial_gpu_sample_explains_missing_timestamps_without_hiding_valid_groups() {
        use crate::render_diagnostics::{GpuSampleHealth, GpuSampleStatus, TimestampError};
        let model = capture(&PerformanceSnapshot {
            render_health: GpuSampleHealth {
                status: GpuSampleStatus::Partial(TimestampError::MissingEnd),
                valid_pairs: 4,
                total_pairs: 5,
            },
            render_breakdown: crate::render_diagnostics::GpuBreakdown {
                opaque_ms: Some(12.0),
                ..Default::default()
            },
            ..Default::default()
        });
        for (label, expected) in [
            ("GPU sample", "Partial: no end"),
            ("GPU pairs valid / total", "4 / 5"),
            ("Tracked GPU span", "N/A"),
            ("Prepass + shadows", "N/A"),
            ("World opaque", "12.00 ms"),
        ] {
            assert_eq!(
                model
                    .frame_rows
                    .iter()
                    .find(|row| row.label == label)
                    .unwrap()
                    .value,
                expected
            );
        }
    }

    #[test]
    fn render_breakdown_displays_physical_extent_and_separate_passes() {
        let model = capture(&PerformanceSnapshot {
            render_extent: Some(crate::render_diagnostics::RenderExtent {
                target: bevy::prelude::UVec2::new(3840, 2160),
                viewport: bevy::prelude::UVec2::new(1920, 1080),
                samples: 4,
            }),
            render_breakdown: crate::render_diagnostics::GpuBreakdown {
                prepass_ms: Some(1.0),
                terrain_ms: None,
                opaque_other_ms: None,
                opaque_ms: Some(30.0),
                post_ms: Some(2.0),
                ui_ms: Some(3.0),
                transparent_ms: Some(5.0),
                world_output_ms: Some(6.0),
                xray_ms: Some(7.0),
                xray_output_ms: Some(8.0),
                other_ms: Some(4.0),
            },
            ..Default::default()
        });
        for (label, expected) in [
            ("World target", "3840 × 2160"),
            ("World viewport", "1920 × 1080"),
            ("World MSAA samples", "4"),
            ("Prepass + shadows", "1.00 ms"),
            ("World opaque", "30.00 ms"),
            ("World post FX", "2.00 ms"),
            ("Mosaic UI", "3.00 ms"),
            ("World transparency", "5.00 ms"),
            ("World output", "6.00 ms"),
            ("X-ray tracked", "7.00 ms"),
            ("X-ray output", "8.00 ms"),
            ("Other tracked", "4.00 ms"),
        ] {
            assert_eq!(
                model
                    .frame_rows
                    .iter()
                    .find(|row| row.label == label)
                    .unwrap()
                    .value,
                expected
            );
        }
    }

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
        let row = |label| {
            model
                .physics_rows
                .iter()
                .find(|row| row.label == label)
                .expect("diagnostic row is present")
        };
        assert_eq!(row("Contact solver").value, "12.00 ms");
        assert_eq!(row("Contact solver").tone, Tone::Bad);
        assert_eq!(row("Broadphase pairs").tone, Tone::Bad);
        assert_eq!(row("Contacts active / made").value, "700 / 900");
        assert_eq!(row("Failure flags").tone, Tone::Good);
    }

    #[test]
    fn diagnostic_render_modes_are_named_and_warned_in_the_overlay() {
        use crate::render_experiments::RenderExperiment;
        for (mode, label, tone) in [
            (RenderExperiment::Baseline, "Baseline", Tone::Neutral),
            (RenderExperiment::NoMsaa, "No MSAA", Tone::Warn),
            (
                RenderExperiment::SimpleTerrain,
                "Simple terrain",
                Tone::Warn,
            ),
        ] {
            let model = capture(&PerformanceSnapshot {
                open: true,
                render_experiment: mode,
                ..PerformanceSnapshot::default()
            });
            let row = model
                .frame_rows
                .iter()
                .find(|row| row.label == "Render experiment")
                .unwrap();
            assert_eq!(row.value, label);
            assert_eq!(row.tone, tone);
        }
    }

    #[test]
    fn performance_panel_stays_in_the_corner_without_taking_the_pointer() {
        for viewport in [VIEWPORT, mosaic_core::Size::new(1280.0, 720.0)] {
            let overlay = Overlay::mount();
            overlay.handles.viewport.set(viewport);
            overlay
                .handles
                .performance
                .set(capture(&PerformanceSnapshot {
                    open: true,
                    fps: Some(60.0),
                    ..PerformanceSnapshot::default()
                }));
            overlay.settle();

            for label in [
                "CPU encoding",
                "CPU finalization",
                "CPU queue submit",
                "CPU readback setup",
                "World target",
                "World viewport",
                "World opaque",
                "Prepass + shadows",
                "Mosaic UI",
                "TERRAIN / COLLISION",
                "Queued reactions",
            ] {
                assert!(overlay.labels().contains(&label.to_owned()));
            }
            assert!(
                !overlay.wants_pointer_at(mosaic_core::Vector2::new(viewport.width - 30.0, 30.0,))
            );
            assert!(overlay.shapes().iter().any(|shape| {
                (shape.rect.size.width - super::PANEL_WIDTH).abs() < 0.5
                    && (shape.rect.origin.x + shape.rect.size.width
                        - (viewport.width - super::PANEL_INSET))
                        .abs()
                        < 0.5
                    && shape.rect.origin.y + shape.rect.size.height <= viewport.height
            }));
        }
    }

    #[test]
    fn submission_stages_display_separately_and_missing_timings_stay_unknown() {
        let known = capture(&PerformanceSnapshot {
            physics_submission_timings: Some(mechanic_gpu::GpuSubmissionTimings {
                encoding_ms: 1.25,
                finalization_ms: 2.5,
                submission_ms: 25.0,
                readback_setup_ms: 0.125,
            }),
            ..PerformanceSnapshot::default()
        });
        let unknown = capture(&PerformanceSnapshot::default());
        for (label, expected) in [
            ("CPU encoding", "1.25 ms"),
            ("CPU finalization", "2.50 ms"),
            ("CPU queue submit", "25.00 ms"),
            ("CPU readback setup", "0.12 ms"),
        ] {
            let row = known
                .physics_rows
                .iter()
                .find(|row| row.label == label)
                .unwrap();
            assert_eq!(row.value, expected);
            assert_eq!(
                unknown
                    .physics_rows
                    .iter()
                    .find(|row| row.label == label)
                    .unwrap()
                    .value,
                "N/A"
            );
        }
    }
}
