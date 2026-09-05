//! Bounded event capture shared with the render thread. No GPU waits or file I/O while sampling.
use std::{
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use bevy::{prelude::*, render::renderer::RenderAdapterInfo, window::PrimaryWindow};
use serde_json::{Value, json};

const LENGTH: Duration = Duration::from_mins(1);
const WARMUP: Duration = Duration::from_secs(15);
const CAPACITY: usize = 100_000;
static ACTIVE: AtomicBool = AtomicBool::new(false);
static RECORDS: OnceLock<Mutex<Option<Capture>>> = OnceLock::new();

struct Capture {
    started: Instant,
    records: Vec<Value>,
    overflow: bool,
}

impl Capture {
    fn push(&mut self, kind: &str, data: impl FnOnce() -> Value) {
        let elapsed = self.started.elapsed();
        if elapsed >= LENGTH {
            return;
        }
        if self.records.len() >= CAPACITY {
            self.overflow = true;
            return;
        }
        self.records.push(
            json!({"kind": kind, "elapsed_ms": elapsed.as_secs_f64()*1000.0, "data": data()}),
        );
    }
}

pub(crate) fn is_active() -> bool {
    ACTIVE.load(Ordering::Relaxed)
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
            json!({"kind":"metadata", "schema":1, "duration_seconds":60, "metadata":self.metadata})
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time<Real>>,
    simulation: Res<crate::AppSimulation>,
    world: Res<crate::world::WorldDiagnostics>,
    space: Res<State<crate::world::AppSpace>>,
    window: Single<&Window, With<PrimaryWindow>>,
    adapter: Res<RenderAdapterInfo>,
    metrics: Res<crate::performance::PerformanceMetrics>,
    render: Res<crate::render_diagnostics::RenderTimings>,
    mut recorder: ResMut<Recorder>,
) {
    if !recorder.initialized {
        recorder.directory = std::env::var_os("MECHANIC_PERF_CAPTURE_DIR")
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        recorder.initialized = true;
    }
    if recorder.directory.is_none() {
        return;
    }
    let expired = RECORDS
        .get()
        .and_then(|records| records.lock().ok())
        .is_some_and(|records| {
            records
                .as_ref()
                .is_some_and(|c| c.started.elapsed() >= LENGTH)
        });
    if expired {
        recorder.finish(false);
    }
    if keyboard.just_pressed(KeyCode::F9) && !ACTIVE.load(Ordering::Relaxed) {
        recorder.armed = true;
        recorder.idle_since = None;
        info!("Performance capture armed: waiting for idle streaming and 15 seconds of warm-up");
    }
    if recorder.armed {
        let ready = *space.get() == crate::world::AppSpace::World
            && simulation.is_running()
            && world.streaming_backlog == 0
            && world.local_total_nodes > 0
            && world.local_resolved_nodes == world.local_total_nodes;
        if !ready {
            recorder.idle_since = None;
        } else if recorder
            .idle_since
            .get_or_insert_with(Instant::now)
            .elapsed()
            >= WARMUP
        {
            recorder.metadata = json!({"terrain_pass_partition":crate::render_diagnostics::terrain_passes_enabled(), "automated_background":crate::automation::enabled(), "adapter":format!("{:?}", adapter.0), "label":std::env::var("MECHANIC_PERF_LABEL").ok(), "executable":std::env::current_exe().ok(), "experiment":format!("{:?}", crate::render_experiments::current()), "f3":metrics.snapshot().open, "present_mode":format!("{:?}",window.present_mode), "start_submitted_tick":simulation.next_tick.saturating_sub(1), "start_completed_tick":simulation.completed_tick});
            *RECORDS
                .get_or_init(|| Mutex::new(None))
                .lock()
                .expect("capture mutex") = Some(Capture {
                started: Instant::now(),
                records: Vec::with_capacity(CAPACITY),
                overflow: false,
            });
            ACTIVE.store(true, Ordering::Relaxed);
            recorder.armed = false;
            info!("Performance capture started (60 seconds)");
            return; // This frame began before the capture.
        }
    }
    record("frame", || {
        let extent = render.snapshot().extent;
        json!({"focused":window.focused, "automated_background":crate::automation::enabled(), "frame_ms":time.delta_secs_f64()*1000.0, "submitted_tick":simulation.next_tick.saturating_sub(1), "completed_tick":simulation.completed_tick, "backlog":simulation.tick_backlog, "in_flight":simulation.in_flight_tick_count, "running":simulation.is_running(), "f3":metrics.snapshot().open, "present_mode":format!("{:?}",window.present_mode), "window_pixels":[window.physical_width(),window.physical_height()], "target_pixels":extent.map(|e|e.target.to_array()), "viewport_pixels":extent.map(|e|e.viewport.to_array()), "msaa":extent.map(|e|e.samples), "terrain_backlog":world.streaming_backlog, "terrain_resolved":world.local_resolved_nodes, "terrain_total":world.local_total_nodes, "terrain_stage_ms":world.terrain_stage_ms, "terrain_overflow_flags":world.overflow_flags})
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capacity_exhaustion_invalidates_without_growing_buffer() {
        let mut capture = Capture {
            started: Instant::now(),
            records: vec![Value::Null; CAPACITY],
            overflow: false,
        };
        capture.push("test", || panic!("must not construct a discarded payload"));
        assert!(capture.overflow);
        assert_eq!(capture.records.len(), CAPACITY);
    }
    #[test]
    fn expired_capture_does_not_accept_late_events() {
        let mut capture = Capture {
            started: Instant::now().checked_sub(LENGTH).unwrap(),
            records: Vec::new(),
            overflow: false,
        };
        capture.push("test", || panic!("late event"));
        assert!(capture.records.is_empty());
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
