//! Bounded event capture shared with the render thread. No GPU waits or file I/O while sampling.
use std::{
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use bevy::{prelude::*, render::renderer::RenderAdapterInfo, window::PrimaryWindow};
use serde_json::{Value, json};

const LENGTH: Duration = Duration::from_mins(1);
const WARMUP: Duration = Duration::from_secs(15);
const CAPACITY: usize = 100_000;
static ACTIVE: AtomicBool = AtomicBool::new(false);
static MEASURING: AtomicBool = AtomicBool::new(false);
static REPLAY_END: AtomicU64 = AtomicU64::new(0);
const DRAIN_TIMEOUT: Duration = Duration::from_secs(10);
static RECORDS: OnceLock<Mutex<Option<Capture>>> = OnceLock::new();

struct Capture {
    started: Instant,
    records: Vec<Value>,
    overflow: bool,
    closed: Option<(Duration, u64)>,
    last_submitted_tick: Option<u64>,
}

impl Capture {
    fn push(&mut self, kind: &str, data: impl FnOnce() -> Value) {
        let elapsed = self.started.elapsed();
        if self.records.len() >= CAPACITY {
            self.overflow = true;
            return;
        }
        let data = data();
        if kind == "physics_submit" {
            self.last_submitted_tick = data.get("tick").and_then(Value::as_u64);
        }
        self.records.push(
            json!({"kind": kind, "phase": if self.closed.is_some() { "drain" } else { "measurement" }, "elapsed_ms": elapsed.as_secs_f64()*1000.0, "data": data}),
        );
    }

    /// Returns whether a finished capture is interrupted; None keeps polling.
    fn progress(
        &mut self,
        next_tick: u64,
        published_tick: u64,
        in_flight: usize,
        failed: bool,
        replay_end: Option<u64>,
    ) -> Option<bool> {
        let elapsed = self.started.elapsed();
        let expired = replay_end.map_or(elapsed >= LENGTH, |end| next_tick >= end);
        if self.closed.is_none() && expired {
            self.closed = Some((elapsed, self.last_submitted_tick.unwrap_or(0)));
        }
        let drained = self
            .closed
            .is_some_and(|(_, tick)| published_tick >= tick && in_flight == 0);
        let timed_out = self
            .closed
            .is_some_and(|(at, _)| elapsed.saturating_sub(at) >= DRAIN_TIMEOUT);
        if self.overflow || failed || timed_out {
            Some(true)
        } else if drained {
            Some(false)
        } else {
            None
        }
    }
}

pub(crate) fn is_active() -> bool {
    MEASURING.load(Ordering::Relaxed)
}

pub(crate) fn is_draining() -> bool {
    ACTIVE.load(Ordering::Relaxed) && !is_active()
}

pub(crate) fn is_recording() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

pub(crate) fn replay_end_tick() -> Option<u64> {
    let end = REPLAY_END.load(Ordering::Relaxed);
    (end != 0).then_some(end)
}

/// Reserves only due, unsubmitted replay ticks; never discards backlog.
pub(crate) fn replay_batch(
    next: &mut u64,
    backlog: &mut u64,
    available: u64,
    end: u64,
) -> std::ops::Range<u64> {
    let first = *next;
    let count = (*backlog).min(available).min(end.saturating_sub(first));
    *next += count;
    *backlog -= count;
    first..*next
}

/// Exact same-build/backend state identity, including velocities and joint state.
pub(crate) fn state_hash(
    transforms: &[mechanic_gpu::GpuTransform],
    velocities: &[mechanic_gpu::GpuVelocity],
    coordinates: &[mechanic_gpu::GpuMechanismCoordinate],
) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for bytes in [
        bytemuck::cast_slice::<_, u8>(transforms),
        bytemuck::cast_slice::<_, u8>(velocities),
        bytemuck::cast_slice::<_, u8>(coordinates),
    ] {
        for byte in bytes.len().to_le_bytes().iter().chain(bytes) {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

/// Evaluates the payload only during an active capture; records each event once at its source.
pub(crate) fn record(kind: &str, data: impl FnOnce() -> Value) {
    if ACTIVE.load(Ordering::Relaxed)
        && let Some(capture) = RECORDS
            .get()
            .and_then(|records| records.lock().ok())
            .as_deref_mut()
            .and_then(Option::as_mut)
    {
        capture.push(kind, data);
    }
}

#[derive(Resource, Default)]
pub(crate) struct Recorder {
    directory: Option<PathBuf>,
    initialized: bool,
    armed: bool,
    idle_since: Option<Instant>,
    metadata: Value,
    pub(crate) completed: Option<Result<PathBuf, String>>,
}

impl Recorder {
    pub(crate) fn arm(&mut self) {
        self.armed = true;
        self.idle_since = None;
    }

    fn finish(&mut self, interrupted: bool) {
        MEASURING.store(false, Ordering::Relaxed);
        ACTIVE.store(false, Ordering::Relaxed);
        let Some(capture) = RECORDS
            .get()
            .and_then(|records| records.lock().ok())
            .and_then(|mut records| records.take())
        else {
            return;
        };
        let valid = !interrupted && !capture.overflow;
        let result = self.write_capture(capture, interrupted);
        self.completed = Some(match &result {
            Ok(path) if valid => Ok(path.clone()),
            Ok(_) => Err("interrupted or capacity-exceeded capture".to_owned()),
            Err(error) => Err(error.to_string()),
        });
        match result {
            Ok(path) => info!("Performance capture: {}", path.display()),
            Err(error) => error!("Could not write performance capture: {error}"),
        }
    }

    fn write_capture(&self, capture: Capture, interrupted: bool) -> std::io::Result<PathBuf> {
        let duration_seconds = capture.started.elapsed().as_secs_f64();
        let directory = self.directory.as_ref().expect("enabled capture directory");
        fs::create_dir_all(directory)?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = directory.join(format!("capture-{nonce}-{}.jsonl", std::process::id()));
        let mut file = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?,
        );
        writeln!(
            file,
            "{}",
            json!({"kind":"metadata", "schema":3, "duration_seconds":duration_seconds, "submission_duration_seconds":capture.closed.map(|(elapsed, _)| elapsed.as_secs_f64()), "metadata":self.metadata})
        )?;
        for record in &capture.records {
            writeln!(file, "{record}")?;
        }
        writeln!(
            file,
            "{}",
            json!({"kind":"result", "valid":!interrupted && !capture.overflow, "interrupted":interrupted, "capacity_exceeded":capture.overflow, "records":capture.records.len()})
        )?;
        file.flush()?;
        Ok(path)
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        self.finish(true);
    }
}

/// F9 arms a capture only when enabled. Streaming must stay idle for fifteen seconds.
#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn sample(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time<Real>>,
    simulation: Res<crate::simulation::state::AppSimulation>,
    world: Res<crate::world::WorldDiagnostics>,
    space: Res<State<crate::world::AppSpace>>,
    window: Single<&Window, With<PrimaryWindow>>,
    adapter: Res<RenderAdapterInfo>,
    metrics: Res<crate::performance::PerformanceMetrics>,
    render: Res<crate::render_diagnostics::RenderTimings>,
    camera: Single<&GlobalTransform, With<crate::camera::MainCamera>>,
    mut recorder: ResMut<Recorder>,
) {
    if !recorder.initialized {
        recorder.directory = crate::env::raw(crate::env::PERF_CAPTURE_DIR)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        recorder.initialized = true;
    }
    if recorder.directory.is_none() {
        return;
    }
    if keyboard.just_pressed(KeyCode::F9) && !ACTIVE.load(Ordering::Relaxed) {
        recorder.armed = true;
        recorder.idle_since = None;
        info!("Performance capture armed: waiting for idle streaming and 15 seconds of warm-up");
    }
    if recorder.armed {
        // Diagnosing a startup stall needs the frames the settled-streaming gate
        // is designed to exclude, so this opt-in records from world entry
        // instead. Such a capture contains warm-up and is not comparable with a
        // settled one.
        let from_start = crate::env::raw(crate::env::PERF_CAPTURE_FROM_START)
            .is_some_and(|value| !value.is_empty());
        let ready = *space.get() == crate::world::AppSpace::World
            && (from_start
                || (simulation.is_running()
                    && world.streaming_backlog == 0
                    && world.local_total_nodes > 0
                    && world.local_resolved_nodes == world.local_total_nodes));
        if !ready {
            recorder.idle_since = None;
        } else if from_start
            || recorder
                .idle_since
                .get_or_insert_with(Instant::now)
                .elapsed()
                >= WARMUP
        {
            recorder.metadata = json!({"requested_physics_route":format!("{:?}", crate::cpu_physics::route()), "capture_from_start":from_start, "terrain_pass_partition":crate::render_diagnostics::terrain_passes_enabled(), "automated_background":crate::automation::background(), "foreground_requested":crate::automation::foreground(), "adapter":format!("{:?}", adapter.0), "label":crate::env::text(crate::env::PERF_LABEL), "executable":std::env::current_exe().ok(), "experiment":format!("{:?}", crate::render_experiments::current()), "f3":metrics.snapshot().open, "present_mode":format!("{:?}",window.present_mode), "start_submitted_tick":simulation.next_tick.saturating_sub(1), "start_completed_tick":simulation.completed_tick, "bodies":simulation.creation.as_ref().map(|c| c.compounds.len()), "dynamic_bodies":simulation.creation.as_ref().map(|c| c.compounds.iter().filter(|body| !body.is_static).count()), "generalized_velocities":simulation.creation.as_ref().map(|c| c.dynamics.elimination_parent.len())});
            recorder.metadata["initial_state_hash"] = simulation
                .live_state
                .as_ref()
                .map(|state| state_hash(&state.transforms, &state.velocities, &state.coordinates))
                .into();
            recorder.metadata["initial_terrain_fingerprint"] = json!(
                simulation
                    .terrain_publication
                    .geometry_fingerprint()
                    .map(|hash| format!("{hash:016x}"))
            );
            recorder.metadata["initial_terrain_layout_fingerprint"] = json!(
                simulation
                    .terrain_publication
                    .layout_fingerprint()
                    .map(|hash| format!("{hash:016x}"))
            );
            recorder.metadata["initial_camera_matrix"] = json!(camera.to_matrix().to_cols_array());
            let replay_ticks = crate::automation::replay_ticks();
            recorder.metadata["replay_ticks"] = json!(replay_ticks);
            REPLAY_END.store(
                replay_ticks.map_or(0, |count| {
                    simulation
                        .next_tick
                        .checked_add(count)
                        .expect("replay tick overflow")
                }),
                Ordering::Relaxed,
            );
            *RECORDS
                .get_or_init(|| Mutex::new(None))
                .lock()
                .expect("capture mutex") = Some(Capture {
                started: Instant::now(),
                records: Vec::with_capacity(CAPACITY),
                overflow: false,
                closed: None,
                last_submitted_tick: None,
            });
            ACTIVE.store(true, Ordering::Relaxed);
            MEASURING.store(true, Ordering::Relaxed);
            recorder.armed = false;
            info!(
                "Performance capture started (replay ticks: {replay_ticks:?}; wall-time capture otherwise)"
            );
            return; // This frame began before the capture.
        }
    }
    record("frame", || {
        let extent = render.snapshot().extent;
        let [deformations, tangents, materials, picked] =
            crate::suspension_render::take_visual_work();
        json!({"suspension_deformations":deformations, "suspension_tangents":tangents, "suspension_material_writes":materials, "suspension_picked_triangles":picked, "focused":window.focused, "automated_background":crate::automation::background(), "foreground_requested":crate::automation::foreground(), "frame_ms":time.delta_secs_f64()*1000.0, "submitted_tick":simulation.next_tick.saturating_sub(1), "completed_tick":simulation.completed_tick, "backlog":simulation.tick_backlog, "dropped":simulation.dropped_ticks, "in_flight":simulation.in_flight_tick_count, "running":simulation.is_running(), "f3":metrics.snapshot().open, "present_mode":format!("{:?}",window.present_mode), "window_pixels":[window.physical_width(),window.physical_height()], "target_pixels":extent.map(|e|e.target.to_array()), "viewport_pixels":extent.map(|e|e.viewport.to_array()), "msaa":extent.map(|e|e.samples), "terrain_backlog":world.streaming_backlog, "terrain_resolved":world.local_resolved_nodes, "terrain_total":world.local_total_nodes, "terrain_stage_ms":world.terrain_stage_ms, "terrain_overflow_flags":world.overflow_flags})
    });
    if ACTIVE.load(Ordering::Relaxed) {
        let outcome = {
            let mut records = RECORDS
                .get()
                .expect("active capture")
                .lock()
                .expect("capture mutex");
            let capture = records.as_mut().expect("active capture");
            let outcome = capture.progress(
                simulation.next_tick,
                simulation.completed_tick,
                simulation
                    .gpu
                    .as_ref()
                    .map_or(0, mechanic_gpu::GpuPhysics::in_flight_tick_count),
                simulation.failure.is_some(),
                replay_end_tick(),
            );
            if capture.closed.is_some() {
                MEASURING.store(false, Ordering::Relaxed);
            }
            outcome
        };
        if let Some(interrupted) = outcome {
            recorder.finish(interrupted);
        }
    }
}

/// Opt-in scope timing, including early returns.
pub(crate) struct FreezeStage(&'static str, Option<Instant>);
impl FreezeStage {
    pub(crate) fn new(stage: &'static str) -> Self {
        Self(stage, is_recording().then(Instant::now))
    }
}
impl Drop for FreezeStage {
    fn drop(&mut self) {
        if let Some(start) = self.1 {
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            record("freeze_stage", || json!({"stage": self.0, "ms": ms}));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn state_identity_includes_velocity_and_joint_state() {
        use bytemuck::Zeroable;
        let poses = [mechanic_gpu::GpuTransform::zeroed()];
        let mut velocities = [mechanic_gpu::GpuVelocity::zeroed()];
        let mut coordinates = [mechanic_gpu::GpuMechanismCoordinate::zeroed()];
        let initial = state_hash(&poses, &velocities, &coordinates);
        assert_eq!(initial, state_hash(&poses, &velocities, &coordinates));
        velocities[0].linear[0] = 1.0;
        assert_ne!(initial, state_hash(&poses, &velocities, &coordinates));
        velocities[0] = mechanic_gpu::GpuVelocity::zeroed();
        coordinates[0].position = 1.0;
        assert_ne!(initial, state_hash(&poses, &velocities, &coordinates));
        assert_ne!(state_hash(&[], &[], &[]), state_hash(&poses, &[], &[]));
    }
    #[test]
    fn capacity_exhaustion_invalidates_without_growing_buffer() {
        let mut capture = Capture {
            started: Instant::now(),
            records: vec![Value::Null; CAPACITY],
            overflow: false,
            closed: None,
            last_submitted_tick: None,
        };
        capture.push("test", || panic!("must not construct a discarded payload"));
        assert!(capture.overflow);
        assert_eq!(capture.records.len(), CAPACITY);
    }
    #[test]
    fn capture_keeps_the_last_batch_and_tags_late_publication() {
        let mut capture = Capture {
            started: Instant::now().checked_sub(LENGTH).unwrap(),
            records: Vec::new(),
            overflow: false,
            closed: None,
            last_submitted_tick: None,
        };
        capture.push("physics_submit", || json!({"tick": 1}));
        capture.closed = Some((capture.started.elapsed(), 1));
        capture.push("physics_publication", || json!({"tick": 1}));
        assert_eq!(capture.records[0]["phase"], "measurement");
        assert_eq!(capture.records[1]["phase"], "drain");
    }
    #[test]
    fn replay_preserves_backlog_and_clamps_the_last_batch() {
        let mut next = 1;
        let mut backlog = 100;
        assert_eq!(replay_batch(&mut next, &mut backlog, 0, 6), 1..1);
        assert_eq!(backlog, 100);
        assert_eq!(replay_batch(&mut next, &mut backlog, 3, 6), 1..4);
        assert_eq!(replay_batch(&mut next, &mut backlog, 3, 6), 4..6);
        assert_eq!(replay_batch(&mut next, &mut backlog, 3, 6), 6..6);
        assert_eq!(backlog, 95);
    }
    #[test]
    fn drain_requires_publication_and_empty_ring_and_has_a_deadline() {
        let mut capture = Capture {
            started: Instant::now(),
            records: Vec::new(),
            overflow: false,
            closed: None,
            last_submitted_tick: None,
        };
        capture.push("physics_submit", || json!({"tick":3}));
        assert_eq!(capture.progress(4, 2, 0, false, Some(4)), None);
        assert!(capture.closed.is_some());
        assert_eq!(capture.progress(4, 3, 1, false, Some(4)), None);
        assert_eq!(capture.progress(4, 3, 0, false, Some(4)), Some(false));
        assert_eq!(capture.progress(4, 3, 0, true, Some(4)), Some(true));
        capture.started = Instant::now()
            .checked_sub(DRAIN_TIMEOUT + Duration::from_secs(1))
            .unwrap();
        capture.closed = Some((Duration::ZERO, 3));
        assert_eq!(capture.progress(4, 2, 1, false, Some(4)), Some(true));
    }
    #[test]
    fn wall_capture_drain_does_not_wait_for_scheduler_drops() {
        let mut capture = Capture {
            started: Instant::now().checked_sub(LENGTH).unwrap(),
            records: Vec::new(),
            overflow: false,
            closed: None,
            last_submitted_tick: None,
        };
        capture.push("physics_submit", || json!({"tick":3}));
        capture.push("physics_drop", || json!({"first_tick":4,"count":100}));
        assert_eq!(capture.progress(104, 3, 0, false, None), Some(false));
        assert_eq!(capture.closed.unwrap().1, 3);
    }
    #[test]
    fn replay_counts_wall_time_during_terrain_holds() {
        let mut scheduler = crate::scheduler::FixedStepScheduler::new();
        let mut next = 1;
        let mut backlog = scheduler.advance(Duration::from_secs(2)).count();
        assert_eq!(replay_batch(&mut next, &mut backlog, 0, 181), 1..1);
        backlog += scheduler.advance(Duration::from_secs(1)).count();
        let mut submitted = Vec::new();
        while next < 181 {
            submitted.extend(replay_batch(&mut next, &mut backlog, 3, 181));
        }
        assert_eq!(submitted, (1..181).collect::<Vec<_>>());
        assert_eq!(backlog, 0);
    }
    #[test]
    fn interrupted_capture_has_invalid_footer_and_unique_files() {
        let directory =
            std::env::temp_dir().join(format!("mechanic-capture-test-{}", std::process::id()));
        let recorder = Recorder {
            directory: Some(directory.clone()),
            initialized: false,
            armed: false,
            idle_since: None,
            metadata: Value::Null,
            completed: None,
        };
        let make = || Capture {
            started: Instant::now(),
            records: Vec::new(),
            overflow: false,
            closed: None,
            last_submitted_tick: None,
        };
        let a = recorder.write_capture(make(), true).unwrap();
        let b = recorder.write_capture(make(), false).unwrap();
        assert_ne!(a, b);
        let contents = fs::read_to_string(a).unwrap();
        let footer: Value = serde_json::from_str(contents.lines().last().unwrap()).unwrap();
        assert_eq!(footer["valid"], false);
        fs::remove_dir_all(directory).unwrap();
    }
}
