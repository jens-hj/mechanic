//! Every environment variable the app reads, named once.
//!
//! The app takes no command-line flags: captures, automation, and diagnostics
//! are all switched on through the environment. Modules read a variable through
//! the accessors here with one of these names, so the full configuration surface
//! is this file, and `docs/environment.md` documents exactly this list.

use std::{ffi::OsString, path::PathBuf};

/// `gpu` runs published ticks on the GPU runtime; anything else runs the CPU solver.
pub(crate) const PHYSICS: &str = "MECHANIC_PHYSICS";
/// `off` disables soil accumulation on the CPU route.
pub(crate) const SOIL: &str = "MECHANIC_SOIL";
/// Directory holding saved creations instead of the platform data directory.
pub(crate) const CREATIONS_DIR: &str = "MECHANIC_CREATIONS_DIR";
/// Launch-only rendering diagnostic.
pub(crate) const RENDER_EXPERIMENT: &str = "MECHANIC_RENDER_EXPERIMENT";

/// Directory receiving per-frame performance JSONL.
pub(crate) const PERF_CAPTURE_DIR: &str = "MECHANIC_PERF_CAPTURE_DIR";
/// Starts the performance capture at launch.
pub(crate) const PERF_CAPTURE_FROM_START: &str = "MECHANIC_PERF_CAPTURE_FROM_START";
/// Label recorded in a capture's identity.
pub(crate) const PERF_LABEL: &str = "MECHANIC_PERF_LABEL";
/// Adds per-pass terrain GPU timings to a capture.
pub(crate) const PERF_TERRAIN_PASSES: &str = "MECHANIC_PERF_TERRAIN_PASSES";
/// Directory receiving the tool-effect capture sequence.
pub(crate) const FX_CAPTURE_DIR: &str = "MECHANIC_FX_CAPTURE_DIR";
/// Directory receiving the suspension capture sequence.
pub(crate) const SUSPENSION_CAPTURE_DIR: &str = "MECHANIC_SUSPENSION_CAPTURE_DIR";

/// Test-world copy to enter without input; turns automation on.
pub(crate) const AUTO_WORLD: &str = "MECHANIC_AUTO_WORLD";
/// World store directory for an automated run.
pub(crate) const AUTO_WORLD_STORE: &str = "MECHANIC_AUTO_WORLD_STORE";
/// Foreground comparison run with a focused window.
pub(crate) const AUTO_FOREGROUND: &str = "MECHANIC_AUTO_FOREGROUND";
/// Seconds between scripted placements.
pub(crate) const AUTO_PLACE: &str = "MECHANIC_AUTO_PLACE";
/// Edge length, in blocks, of each scripted placement.
pub(crate) const AUTO_PLACE_VOLUME: &str = "MECHANIC_AUTO_PLACE_VOLUME";
/// Drives the scripted route.
pub(crate) const AUTO_DRIVE: &str = "MECHANIC_AUTO_DRIVE";
/// Drives straight instead of the steering route.
pub(crate) const AUTO_DRIVE_STRAIGHT: &str = "MECHANIC_AUTO_DRIVE_STRAIGHT";
/// Saves demonstration screenshots while driving.
pub(crate) const AUTO_DRIVING_FRAMES: &str = "MECHANIC_AUTO_DRIVING_FRAMES";
/// Number of recorded ticks to replay.
pub(crate) const AUTO_REPLAY_TICKS: &str = "MECHANIC_AUTO_REPLAY_TICKS";
/// Exercises the dimension-link freeze during an automated run.
pub(crate) const AUTO_FREEZE: &str = "MECHANIC_AUTO_FREEZE";
/// JSON hammer strike delivered during an automated run.
pub(crate) const AUTO_HAMMER: &str = "MECHANIC_AUTO_HAMMER";

/// `1` makes tests enforce their wall-clock budgets; see `cargo xtask budgets`.
#[cfg(test)]
pub(crate) const TIMING_TESTS: &str = "MECHANIC_TIMING_TESTS";
/// Creation file used by the edit-latency rendering test.
#[cfg(test)]
pub(crate) const EDIT_FIXTURE: &str = "MECHANIC_EDIT_FIXTURE";
/// Reference shader path for the paired terrain-material comparison.
#[cfg(test)]
pub(crate) const TERRAIN_REFERENCE_SHADER: &str = "MECHANIC_TERRAIN_REFERENCE_SHADER";
/// Candidate shader path for the paired terrain-material comparison.
#[cfg(test)]
pub(crate) const TERRAIN_CANDIDATE_SHADER: &str = "MECHANIC_TERRAIN_CANDIDATE_SHADER";

/// The variable's value, when it is set to valid Unicode.
pub(crate) fn text(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// The variable's raw value, when it is set at all.
pub(crate) fn raw(name: &str) -> Option<OsString> {
    std::env::var_os(name)
}

/// The variable as a path, when it is set at all.
pub(crate) fn path(name: &str) -> Option<PathBuf> {
    raw(name).map(PathBuf::from)
}

/// Whether the variable is exactly `1`, the spelling every on/off switch uses.
pub(crate) fn flag(name: &str) -> bool {
    text(name).as_deref() == Some("1")
}

/// Whether the variable is set to anything but the empty string.
pub(crate) fn is_set(name: &str) -> bool {
    raw(name).is_some_and(|value| !value.is_empty())
}
