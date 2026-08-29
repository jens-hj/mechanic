//! Lightweight sampling for the opt-in performance overlay.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::{
    prelude::{ButtonInput, KeyCode, Real, Res, ResMut, Resource, Single, State, Time, With},
    window::{PresentMode, PrimaryWindow, Window},
};
use mechanic_gpu::GpuKernelTimings;

use crate::AppSimulation;

const FRAME_HISTORY_LENGTH: usize = 120;
const DISPLAY_INTERVAL: Duration = Duration::from_millis(250);

/// Values shown by the performance overlay, refreshed at a human-readable rate.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PerformanceSnapshot {
    pub(crate) open: bool,
    pub(crate) fps: Option<f64>,
    pub(crate) frame_ms: Option<f64>,
    pub(crate) frame_p95_ms: Option<f64>,
    pub(crate) render_cpu_ms: Option<f64>,
    pub(crate) render_gpu_ms: Option<f64>,
    pub(crate) simulation_ticks_per_second: Option<f64>,
    pub(crate) physics_cpu_ms: Option<f64>,
    pub(crate) physics_gpu_ms: Option<f64>,
    pub(crate) kernel_timings: Option<GpuKernelTimings>,
    pub(crate) body_count: Option<u32>,
    pub(crate) collider_count: Option<u32>,
    pub(crate) pair_count: Option<u32>,
    pub(crate) contact_count: Option<u32>,
    pub(crate) active_contact_count: Option<u32>,
    pub(crate) error_flags: Option<u32>,
    pub(crate) terrain_stage_ms: Option<f64>,
    pub(crate) terrain_selection_ms: Option<f64>,
    pub(crate) terrain_sampling_ms: Option<f64>,
    pub(crate) terrain_polygonization_ms: Option<f64>,
    pub(crate) terrain_seams_ms: Option<f64>,
    pub(crate) terrain_bvh_ms: Option<f64>,
    pub(crate) terrain_publication_ms: Option<f64>,
    pub(crate) terrain_queue_age_ms: Option<f64>,
    pub(crate) terrain_bounds_cache_bytes: Option<u64>,
    pub(crate) terrain_local_resolved: Option<u32>,
    pub(crate) terrain_local_total: Option<u32>,
    pub(crate) terrain_triangle_count: Option<u64>,
    pub(crate) terrain_streaming_backlog: Option<u32>,
    pub(crate) terrain_remesh_count: Option<u64>,
    pub(crate) terrain_overflow_flags: Option<u32>,
    pub(crate) foundation_refresh_ms: Option<f64>,
    pub(crate) foundation_candidate_count: Option<u64>,
    pub(crate) foundation_sample_count: Option<u64>,
}

/// Rolling measurements behind the opt-in performance overlay.
#[derive(Resource, Debug)]
pub(crate) struct PerformanceMetrics {
    open: bool,
    snapshot: PerformanceSnapshot,
    frame_samples_ms: VecDeque<f64>,
    last_frame_measurement: Option<Instant>,
    last_tick_index: u64,
    refresh_elapsed: Duration,
    force_refresh: bool,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            open: false,
            snapshot: PerformanceSnapshot::default(),
            frame_samples_ms: VecDeque::with_capacity(FRAME_HISTORY_LENGTH),
            last_frame_measurement: None,
            last_tick_index: 0,
            refresh_elapsed: Duration::ZERO,
            force_refresh: false,
        }
    }
}

impl PerformanceMetrics {
    pub(crate) const fn snapshot(&self) -> PerformanceSnapshot {
        self.snapshot
    }

    fn toggle(&mut self) -> bool {
        self.open = !self.open;
        self.force_refresh = true;
        self.refresh_elapsed = Duration::ZERO;
        if !self.open {
            self.snapshot.open = false;
        }
        self.open
    }

    fn note_frame(&mut self, diagnostics: &DiagnosticsStore) {
        let Some(measurement) = diagnostics
            .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
            .and_then(|diagnostic| diagnostic.measurement())
        else {
            return;
        };
        if self.last_frame_measurement == Some(measurement.time) {
            return;
        }
        self.last_frame_measurement = Some(measurement.time);
        if self.frame_samples_ms.len() == FRAME_HISTORY_LENGTH {
            self.frame_samples_ms.pop_front();
        }
        self.frame_samples_ms.push_back(measurement.value);
    }
}

/// F3 is deliberately a debug-only shortcut, like the existing F8 frame freeze.
pub(crate) fn toggle(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut metrics: ResMut<PerformanceMetrics>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
) {
    if keyboard.just_pressed(KeyCode::F3) {
        window.present_mode = performance_present_mode(metrics.toggle());
    }
}

const fn performance_present_mode(open: bool) -> PresentMode {
    if open {
        PresentMode::AutoNoVsync
    } else {
        PresentMode::Fifo
    }
}

/// Samples existing Bevy and physics diagnostics; it never waits for the GPU.
#[allow(clippy::similar_names)] // CPU/GPU pairs deliberately share metric names.
pub(crate) fn sample(
    time: Res<Time<Real>>,
    diagnostics: Res<DiagnosticsStore>,
    simulation: Res<AppSimulation>,
    space: Res<State<crate::world::AppSpace>>,
    world_diagnostics: Res<crate::world::WorldDiagnostics>,
    mut metrics: ResMut<PerformanceMetrics>,
) {
    metrics.note_frame(&diagnostics);
    if !metrics.open {
        metrics.refresh_elapsed = Duration::ZERO;
        metrics.last_tick_index = simulation.next_tick;
        return;
    }

    metrics.refresh_elapsed = metrics.refresh_elapsed.saturating_add(time.delta());
    if !metrics.force_refresh && metrics.refresh_elapsed < DISPLAY_INTERVAL {
        return;
    }

    let elapsed_seconds = metrics.refresh_elapsed.as_secs_f64();
    let completed_ticks = simulation.next_tick.saturating_sub(metrics.last_tick_index);
    let completed_ticks = u32::try_from(completed_ticks).unwrap_or(u32::MAX);
    let ticks_per_second =
        (elapsed_seconds > 0.0).then(|| f64::from(completed_ticks) / elapsed_seconds);
    metrics.refresh_elapsed = Duration::ZERO;
    metrics.last_tick_index = simulation.next_tick;
    metrics.force_refresh = false;

    let running = simulation.is_running();
    let creation = simulation.creation.as_ref().filter(|_| running);
    let readback = simulation.last_tick_readback.filter(|_| running);
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(bevy::diagnostic::Diagnostic::smoothed);
    let frame_ms = mean(&metrics.frame_samples_ms);
    let frame_p95_ms = percentile_95(&metrics.frame_samples_ms);
    let render_cpu_ms = render_time(&diagnostics, "elapsed_cpu");
    let render_gpu_ms = render_time(&diagnostics, "elapsed_gpu");
    let in_world = *space.get() == crate::world::AppSpace::World;

    metrics.snapshot = PerformanceSnapshot {
        open: true,
        fps,
        frame_ms,
        frame_p95_ms,
        render_cpu_ms,
        render_gpu_ms,
        simulation_ticks_per_second: running.then_some(ticks_per_second).flatten(),
        physics_cpu_ms: running.then_some(simulation.physics_cpu_ms).flatten(),
        physics_gpu_ms: readback.and_then(|value| value.gpu_tick_ms),
        kernel_timings: readback.and_then(|value| value.kernel_timings),
        body_count: creation.map(|value| capped_u32(value.compounds.len())),
        collider_count: creation.map(|value| capped_u32(value.colliders.len())),
        pair_count: readback.map(|value| value.pair_count),
        contact_count: readback.map(|value| value.contact_count),
        active_contact_count: readback.map(|value| value.active_contact_count),
        error_flags: readback.map(|value| value.error_flags),
        terrain_stage_ms: in_world.then_some(world_diagnostics.terrain_stage_ms),
        terrain_selection_ms: in_world.then_some(world_diagnostics.selection_ms),
        terrain_sampling_ms: in_world.then_some(world_diagnostics.column_sampling_ms),
        terrain_polygonization_ms: in_world.then_some(world_diagnostics.polygonization_ms),
        terrain_seams_ms: in_world.then_some(world_diagnostics.transitions_caps_ms),
        terrain_bvh_ms: in_world.then_some(world_diagnostics.bvh_construction_ms),
        terrain_publication_ms: in_world.then_some(world_diagnostics.publication_ms),
        terrain_queue_age_ms: in_world.then_some(world_diagnostics.oldest_queue_age_ms),
        terrain_bounds_cache_bytes: in_world.then_some(world_diagnostics.bounds_cache_bytes),
        terrain_local_resolved: in_world.then_some(world_diagnostics.local_resolved_nodes),
        terrain_local_total: in_world.then_some(world_diagnostics.local_total_nodes),
        terrain_triangle_count: in_world.then_some(world_diagnostics.triangle_count),
        terrain_streaming_backlog: in_world.then_some(world_diagnostics.streaming_backlog),
        terrain_remesh_count: in_world.then_some(world_diagnostics.remesh_count),
        terrain_overflow_flags: in_world.then_some(world_diagnostics.overflow_flags),
        foundation_refresh_ms: in_world.then_some(world_diagnostics.foundation_refresh_ms),
        foundation_candidate_count: in_world
            .then_some(world_diagnostics.foundation_candidate_count),
        foundation_sample_count: in_world.then_some(world_diagnostics.foundation_sample_count),
    };
}

fn mean(values: &VecDeque<f64>) -> Option<f64> {
    let count = u32::try_from(values.len()).unwrap_or(u32::MAX);
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / f64::from(count))
}

fn percentile_95(values: &VecDeque<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted: Vec<_> = values.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    let rank = sorted.len().saturating_mul(95).div_ceil(100);
    sorted.get(rank.saturating_sub(1)).copied()
}

fn render_time(diagnostics: &DiagnosticsStore, suffix: &str) -> Option<f64> {
    let values: Vec<_> = diagnostics
        .iter()
        .filter(|diagnostic| is_render_timing(diagnostic.path().as_str(), suffix))
        .filter_map(bevy::diagnostic::Diagnostic::smoothed)
        .collect();
    (!values.is_empty()).then(|| values.iter().sum())
}

fn is_render_timing(path: &str, suffix: &str) -> bool {
    path.starts_with("render/") && path.ends_with(&format!("/{suffix}"))
}

fn capped_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use bevy::window::PresentMode;

    use super::{is_render_timing, mean, percentile_95, performance_present_mode};

    #[test]
    fn performance_overlay_measures_uncapped_presentation() {
        assert_eq!(performance_present_mode(true), PresentMode::AutoNoVsync);
        assert_eq!(performance_present_mode(false), PresentMode::Fifo);
    }

    #[test]
    fn frame_summary_reports_mean_and_tail_latency() {
        let values = VecDeque::from([10.0, 11.0, 12.0, 13.0, 50.0]);
        assert_eq!(mean(&values), Some(19.2));
        assert_eq!(percentile_95(&values), Some(50.0));
    }

    #[test]
    fn only_render_elapsed_diagnostics_are_aggregated() {
        assert!(is_render_timing(
            "render/main_opaque_pass_3d/elapsed_cpu",
            "elapsed_cpu"
        ));
        assert!(is_render_timing(
            "render/postprocessing/bloom/elapsed_gpu",
            "elapsed_gpu"
        ));
        assert!(!is_render_timing("frame_time", "elapsed_cpu"));
        assert!(!is_render_timing(
            "render/main_opaque_pass_3d/fragment_shader_invocations",
            "elapsed_gpu"
        ));
    }
}
