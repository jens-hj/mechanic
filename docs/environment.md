# Environment variables

Mechanic has no command-line flags of its own outside the benchmark binaries.
Everything below is an environment variable read at startup. Unset means off.

## `mechanic-app`

| Variable | Effect |
|---|---|
| `MECHANIC_PHYSICS` | `gpu` runs published ticks on the GPU runtime. Anything else, or unset, runs the CPU solver. See [CPU physics](physics-cpu.md). |
| `MECHANIC_SOIL` | `off` disables soil accumulation on the CPU route; it is on otherwise. See [terrain response status](terrain-response-status.md). |
| `MECHANIC_CREATIONS_DIR` | Directory holding saved creations instead of the platform data directory. |
| `MECHANIC_RENDER_EXPERIMENT` | Selects a launch-only rendering diagnostic; never persisted. See [performance profiling](performance-profiling.md). |
| `MECHANIC_WORLDGEN_DIR` | Debug builds only. Generates worlds from this directory's `world.ron`, `library.ron`, and `biomes/*.ron` instead of the embedded definition, and regenerates the terrain whenever a file changes. Saves are untouched. See [world generation](world-generation.md). |
| `MECHANIC_TERRAIN_REFERENCE_SHADER`, `MECHANIC_TERRAIN_CANDIDATE_SHADER` | Shader paths for the paired terrain-material comparison. |

### Captures

| Variable | Effect |
|---|---|
| `MECHANIC_PERF_CAPTURE_DIR` | Writes per-frame performance JSONL into this directory. |
| `MECHANIC_PERF_CAPTURE_FROM_START` | Starts the performance capture at launch instead of on the capture key. |
| `MECHANIC_PERF_LABEL` | Label recorded in the capture's identity. |
| `MECHANIC_PERF_TERRAIN_PASSES` | Adds per-pass terrain GPU timings to the capture. |
| `MECHANIC_FX_CAPTURE_DIR` | Renders the tool-effect capture sequence into this directory and keeps the window unfocused. |
| `MECHANIC_INPUT_CAPTURE_DIR` | Captures native physical-input meshes and extruded keys at all three sizes, then exits. |
| `MECHANIC_SUSPENSION_CAPTURE_DIR` | Renders the suspension capture sequence into this directory. See [suspension](suspension.md). |

### Automation

Scripted runs used by `scripts/run-background-capture.py`. `MECHANIC_AUTO_WORLD`
turns automation on and requires `MECHANIC_PERF_CAPTURE_DIR`; the rest refine it.
Flags are on when set to `1`.

| Variable | Effect |
|---|---|
| `MECHANIC_AUTO_WORLD` | Enters the named test-world copy without input. |
| `MECHANIC_AUTO_WORLD_STORE` | World store directory for the run, isolating it from the player's saves. |
| `MECHANIC_AUTO_FOREGROUND` | Foreground comparison run with a focused window; the default is an unfocused background window. Forbids placement, capture-from-start, and demonstration frames. |
| `MECHANIC_AUTO_PLACE` | Places a scripted part every this many seconds. |
| `MECHANIC_AUTO_PLACE_VOLUME` | Edge length, in blocks, of each scripted placement; 1 by default. |
| `MECHANIC_AUTO_DRIVE` | Drives the scripted steering route. |
| `MECHANIC_AUTO_DRIVE_STRAIGHT` | Drives straight instead of the steering route. |
| `MECHANIC_AUTO_DRIVING_FRAMES` | Saves demonstration screenshots while driving. |
| `MECHANIC_AUTO_REPLAY_TICKS` | Replays 1 to 3,600 recorded ticks; foreground runs only. |
| `MECHANIC_AUTO_FREEZE` | Exercises the dimension-link freeze during the run. |
| `MECHANIC_AUTO_HAMMER` | JSON `{body_index, local_point, impulse}`: delivers that hammer strike during the run. |

## `mechanic-physics` diagnostics

| Variable | Effect |
|---|---|
| `MECHANIC_TRACE_EVENTS` | Prints the reference solver's impact events as they are solved. |
| `MECHANIC_TRACE_CONTINUATION` | In tests, prints each smoothing stage of the Newton continuation solve. |
| `MECHANIC_IMPACT_CAPTURE`, `MECHANIC_FORCE_CAPTURE` | Path to write a failed impact or force solve to, for offline inspection. See [CPU physics](physics-cpu.md). |

## `mechanic-bench`

| Variable | Effect |
|---|---|
| `MECHANIC_PIPE_SPEEDS` | Comma-separated speeds for the `cpu-physics` pipe-motion case; `0,1,10,40` by default. |

## Tests

| Variable | Effect |
|---|---|
| `MECHANIC_TIMING_TESTS` | `1` makes the app's tests enforce their wall-clock budgets. `cargo xtask budgets` sets it and runs them alone. |
| `MECHANIC_EDIT_FIXTURE` | Creation file used by the app's edit-latency rendering test. |

## Tooling

| Variable | Effect |
|---|---|
| `CARGO_TARGET_DIR` | `nix run .` defaults it to `$XDG_CACHE_HOME/mechanic/target`. |
| `VK_DRIVER_FILES` | Point at the lavapipe ICD to run GPU tests on the CPU, off the display GPU. |
