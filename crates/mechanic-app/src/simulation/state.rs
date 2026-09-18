//! The running simulation: its physics backends, tick budget, and failure handling.

use crate::editor::preview::FeaturePreviewKey;
use crate::editor::state::EditorState;
use crate::pose::simulation_part_pose;
use crate::scheduler::FixedStepScheduler;
use crate::simulation::visuals::visual_snapshot_is_due;
use crate::{cpu_physics, performance_capture, terrain_publication};
use bevy::prelude::{Quat, Resource, Vec3, error, format};
use mechanic_core::{CompiledCreation, ConstructionGraph, PartId};
use mechanic_gpu::{GpuPhysics, GpuTickReadback, GpuTransform, GpuVelocity};

/// The compiled scene and whichever solver ticks it.
///
/// The two backends are not alternatives held in one slot. `gpu` is the resident
/// scene: it exists whenever a creation is simulated and always owns terrain
/// preparation, drive resolution, and the buffers the renderer reads. `cpu` is
/// the tick route layered over it, and can be dropped mid-run: a tick the CPU
/// solver cannot complete hands the last published state to the GPU runtime.
/// Ask [`AppSimulation::tick_route`] which solver ticks rather than testing the
/// fields.
#[derive(Resource, Default)]
pub(crate) struct AppSimulation {
    /// The resident GPU scene.
    pub(crate) gpu: Option<GpuPhysics>,
    /// CPU solver, absent under `MECHANIC_PHYSICS=gpu` or after it handed a
    /// tick it could not complete to the GPU runtime.
    pub(crate) cpu: Option<Box<cpu_physics::CpuRoute>>,
    pub(crate) creation: Option<CompiledCreation>,
    /// Exact graph snapshot represented by `creation` and the live GPU scene.
    pub(crate) published_graph: ConstructionGraph,
    pub(crate) scheduler: FixedStepScheduler,
    pub(crate) next_tick: u64,
    pub(crate) tick_backlog: u64,
    /// Ticks skipped because simulated time fell too far behind wall time.
    pub(crate) dropped_ticks: u64,
    pub(crate) completed_tick: u64,
    pub(crate) previous_transforms: Vec<GpuTransform>,
    pub(crate) transforms: Vec<GpuTransform>,
    /// Latest authoritative state, independent of render snapshot throttling.
    pub(crate) live_state: Option<LivePhysicsState>,
    pub(crate) previous_snapshot_tick: u64,
    pub(crate) snapshot_tick: u64,
    pub(crate) pose_revision: u64,
    pub(crate) static_mesh_dirty: bool,
    /// Feature drag drawn into the static published meshes, if any.
    pub(crate) rendered_feature_preview: Option<FeaturePreviewKey>,
    pub(crate) render_dirty: bool,
    pub(crate) physics_cpu_ms: Option<f64>,
    pub(crate) physics_submission_timings: Option<mechanic_gpu::GpuSubmissionTimings>,
    pub(crate) ticks_submitted_per_frame: u32,
    pub(crate) in_flight_tick_count: u32,
    pub(crate) submission_to_readback_ms: Option<f64>,
    pub(crate) visual_update_ms: Option<f64>,
    pub(crate) last_tick_readback: Option<GpuTickReadback>,
    pub(crate) failure: Option<String>,
    pub(crate) world_revision: Option<(u64, u64)>,
    pub(crate) terrain_publication: terrain_publication::TerrainPublication,
}

#[derive(Clone, Debug)]
pub(crate) struct LivePhysicsState {
    pub(crate) tick: u64,
    pub(crate) transforms: Vec<GpuTransform>,
    pub(crate) velocities: Vec<GpuVelocity>,
    pub(crate) coordinates: Vec<mechanic_gpu::GpuMechanismCoordinate>,
}

/// Simulated time a scene may fall behind wall time before ticks are dropped.
///
/// An uncapped backlog never recovers: a scene that once fell behind keeps a
/// full batch due on every later frame, so the simulation stays in slow motion
/// permanently instead of catching up. Dropping the excess only changes how far
/// simulated time lags; every tick that does run is unchanged.
pub(crate) const MAXIMUM_TICK_BACKLOG: u64 = 30;

pub(crate) const MAXIMUM_TICKS_PER_FRAME: usize = 3;

pub(crate) fn next_simulation_ticks(
    scheduler: &mut FixedStepScheduler,
    next_tick: &mut u64,
    tick_backlog: &mut u64,
    dropped_ticks: &mut u64,
    elapsed: std::time::Duration,
    paused: bool,
    maximum_batch: u64,
) -> std::ops::Range<u64> {
    if paused {
        return *next_tick..*next_tick;
    }
    *tick_backlog = tick_backlog.saturating_add(scheduler.advance(elapsed).count());
    let dropped = tick_backlog.saturating_sub(MAXIMUM_TICK_BACKLOG);
    *tick_backlog -= dropped;
    if dropped != 0 {
        performance_capture::record(
            "physics_drop",
            || serde_json::json!({"first_tick": *next_tick, "count": dropped}),
        );
    }
    *next_tick = next_tick.saturating_add(dropped);
    *dropped_ticks = dropped_ticks.saturating_add(dropped);
    let first = *next_tick;
    let batch = (*tick_backlog).min(maximum_batch);
    *tick_backlog -= batch;
    *next_tick = next_tick.saturating_add(batch);
    first..*next_tick
}

pub(crate) fn stop_failed_simulation(
    simulation: &mut AppSimulation,
    state: &mut EditorState,
    error: String,
) {
    // A failed tick pauses at the last completed state, including a CPU tick
    // that has not reached the throttled visual snapshot yet.
    if let Some(live) = &simulation.live_state {
        simulation
            .previous_transforms
            .clone_from(&simulation.transforms);
        simulation.transforms.clone_from(&live.transforms);
        simulation.previous_snapshot_tick = simulation.snapshot_tick;
        simulation.snapshot_tick = live.tick;
        simulation.pose_revision = simulation.pose_revision.wrapping_add(1);
        simulation.render_dirty = true;
    }
    // The status line is easy to miss while the world simply looks frozen.
    error!("Simulation stopped: {error}");
    simulation.failure = Some(error.clone());
    state.feedback = Some(format!("Simulation stopped: {error}"));
}

impl AppSimulation {
    /// Which solver advances published ticks right now.
    pub(crate) fn tick_route(&self) -> cpu_physics::Route {
        if self.cpu.is_some() {
            cpu_physics::Route::Cpu
        } else {
            cpu_physics::Route::Gpu
        }
    }

    pub(crate) fn is_running(&self) -> bool {
        (self.gpu.is_some() || self.cpu.as_ref().is_some_and(|cpu| cpu.has_clumps()))
            && self.failure.is_none()
    }

    pub(crate) fn live_part_pose(
        &self,
        graph: &ConstructionGraph,
        part: PartId,
    ) -> Option<(Vec3, Quat)> {
        let Some(creation) = self.creation.as_ref() else {
            return Some((graph.part_position(part)?, graph.part_rotation(part)?));
        };
        simulation_part_pose(&self.published_graph, creation, &self.transforms, part)
    }

    /// Publishes one completed CPU tick exactly as a GPU readback would, so the
    /// renderer, world walking and the editor read state from one place.
    pub(crate) fn publish_cpu_tick(&mut self, tick: u64, completed: cpu_physics::Completed) {
        self.completed_tick = tick;
        performance_capture::record("physics_publication", || {
            serde_json::json!({
                "tick": tick,
                "state_hash": performance_capture::state_hash(
                    &completed.transforms,
                    &completed.velocities,
                    &completed.coordinates,
                ),
                "route": "cpu",
            })
        });
        self.live_state = Some(LivePhysicsState {
            tick,
            transforms: completed.transforms.clone(),
            velocities: completed.velocities,
            coordinates: completed.coordinates,
        });
        if visual_snapshot_is_due(self.snapshot_tick, tick) {
            self.previous_transforms =
                core::mem::replace(&mut self.transforms, completed.transforms);
            self.previous_snapshot_tick = self.snapshot_tick;
            self.snapshot_tick = tick;
            self.pose_revision = self.pose_revision.wrapping_add(1);
            self.render_dirty = true;
        }
    }

    pub(crate) fn record_performance(
        &mut self,
        cpu_elapsed: std::time::Duration,
        tick_count: usize,
        cpu_timings: mechanic_gpu::GpuSubmissionTimings,
        readback: Option<GpuTickReadback>,
    ) {
        const SMOOTHING: f64 = 0.2;
        let count = u32::try_from(tick_count).unwrap_or(u32::MAX).max(1);
        let cpu_ms = cpu_elapsed.as_secs_f64() * 1_000.0 / f64::from(count);
        self.ticks_submitted_per_frame = count;
        self.physics_cpu_ms = Some(self.physics_cpu_ms.map_or(cpu_ms, |previous| {
            previous + (cpu_ms - previous) * SMOOTHING
        }));
        let previous = self.physics_submission_timings;
        let smooth = |total: f64, previous: Option<f64>| {
            let per_tick = total / f64::from(count);
            previous.map_or(per_tick, |value| value + (per_tick - value) * SMOOTHING)
        };
        self.physics_submission_timings = Some(mechanic_gpu::GpuSubmissionTimings {
            encoding_ms: smooth(cpu_timings.encoding_ms, previous.map(|v| v.encoding_ms)),
            finalization_ms: smooth(
                cpu_timings.finalization_ms,
                previous.map(|v| v.finalization_ms),
            ),
            submission_ms: smooth(cpu_timings.submission_ms, previous.map(|v| v.submission_ms)),
            readback_setup_ms: smooth(
                cpu_timings.readback_setup_ms,
                previous.map(|v| v.readback_setup_ms),
            ),
        });
        if let Some(readback) = readback {
            self.last_tick_readback = Some(readback);
        }
    }

    pub(crate) fn record_visual_update(&mut self, elapsed: std::time::Duration) {
        const SMOOTHING: f64 = 0.2;
        let milliseconds = elapsed.as_secs_f64() * 1_000.0;
        self.visual_update_ms = Some(self.visual_update_ms.map_or(milliseconds, |previous| {
            previous + (milliseconds - previous) * SMOOTHING
        }));
    }
}
