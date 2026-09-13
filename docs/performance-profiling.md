# Performance profiling

## Callback and terrain attribution diagnostics

With `MECHANIC_PERF_CAPTURE_DIR` enabled, physics readbacks now retain the time
of the last mapping callback. JSONL records queue-return-to-callback latency,
whether that callback ran during the consuming poll call, and callback-to-tick
publication latency. Successful and empty poll calls are recorded separately.
No additional polling, waiting, slots, dispatches or changes to the fixed timestep
are introduced. A callback may run on another thread during a poll; the boolean
describes timing, not which call caused it to run. Mapping callbacks require
device servicing, so pre-callback latency is not pure GPU time. The latency origin
is now captured immediately after queue submission returns, before map setup.

For terrain attribution on the M1 Pro, which supports stage-boundary but not
draw-boundary timestamps, use a separate opt-in diagnostic run:

```sh
MECHANIC_PERF_TERRAIN_PASSES=1 python3 scripts/run-background-capture.py \
  --world "$HOME/Library/Application Support/Mechanic/worlds/test4" \
  --output /tmp/test4-terrain-attribution-01
```

This requires the current release build (`cargo build --release -p mechanic-app`).
Capture metadata marks `terrain_pass_partition`. A local pinned
[`bevy_core_pipeline` patch](../vendor/bevy_core_pipeline/MECHANIC-PATCH.md)
partitions contiguous opaque draws without reordering them. Terrain and remaining
opaque runs retain the same shaders, geometry, MSAA, viewport, depth and color
contents. Each run receives actual pass-boundary timestamps. Additional attachment
store/load and resolve work makes this an attribution experiment, not a faster
rendering mode or an exact terrain cost in the original unsplit pass.

The summary reports `terrain_ms` and `opaque_other_ms` separately; `opaque_ms`
is unavailable for partitioned world passes. Missing/invalid samples remain
unavailable. Do not add overlapping pass intervals. A normal run with
`MECHANIC_PERF_TERRAIN_PASSES=0` provides completion-timing evidence without this
render-pass perturbation. Compare frozen copies of the same world, and keep
foreground acceptance benchmarks separate from background diagnostics.

Focused verification:

```sh
cargo test -p mechanic-gpu callback_timing_distinguishes_servicing_from_later_consumption -- --nocapture
cargo test -p mechanic-gpu asynchronous_tick_readback_is_monotonic_and_tick_matched -- --nocapture
MECHANIC_PERF_CAPTURE_DIR=/tmp/mechanic-pixel-diagnostic MECHANIC_PERF_TERRAIN_PASSES=1 \
  cargo test -p mechanic-app terrain_pass_partition_preserves_pixels_and_reports_separate_timing -- --ignored --nocapture
python3 scripts/test-perf-capture.py
```

The ignored pixel test needs a real GPU and uses the same terrain shader plus
another opaque material at 4096×2524, 4× MSAA. It compares partitioned and normal
output byte-for-byte and requires valid positive timings for both groups.

Latest remaining-wait investigation: [background CPU/GPU evidence](performance-results/README.md#2026-09-05-background-bottleneck).
The raw summarizer now includes separate render GPU group distributions, physics
CPU stages, sample observation age, in-flight occupancy and focused-frame counts.
Use `python3 scripts/summarize-perf-capture.py PATH_TO_CAPTURE.jsonl` after a run;
these intervals are not additive. Native CPU profiles collected during a capture
make that run diagnostic, not a clean acceptance benchmark.

## Controlled rendering comparisons

These launch-only modes isolate one rendering variable at a time. They do not
change saved settings, terrain geometry/LOD, physics, or vehicle data. F3 shows
`Render experiment`; non-baseline modes are amber. Invalid or combined values
are rejected at startup. Run one app instance at a time:

```sh
MECHANIC_RENDER_EXPERIMENT=baseline cargo run --release -p mechanic-app
MECHANIC_RENDER_EXPERIMENT=no-msaa cargo run --release -p mechanic-app
MECHANIC_RENDER_EXPERIMENT=simple-terrain cargo run --release -p mechanic-app
```

- `baseline`: normal materials and 4× MSAA; also the default when unset.
- `no-msaa`: normal materials, MSAA off on both the main and x-ray cameras
  sharing the world target. F3 should show one world sample.
- `simple-terrain`: original 4× MSAA, but terrain uses a constant-color/roughness
  material with geometric normals instead of texture sampling/normal mapping.
  Forward PBR lighting, fog, meshes, visibility, texture loading and streaming
  remain unchanged. Vehicles and other materials are untouched. The ground
  intentionally looks plain; this is not a proposed permanent quality setting.

Use full-size windows with matching F3 target dimensions, the same position,
camera framing and F3 visibility. Once streaming backlog and terrain work are
zero, wait at least 15 seconds for compilation/rolling samples to settle. Capture
FPS, TPS, opaque/other tracked GPU timings, world samples, window acquire and
CPU queue submit. Repeat baseline afterward to check drift. Do not sum
overlapping GPU intervals or claim a gain from a single asynchronous sample.

The real-material offscreen pixel regression can be run separately for each mode:

```sh
MECHANIC_RENDER_EXPERIMENT=simple-terrain cargo test -p mechanic-app terrain_experiment_renders_pixels_with_the_real_material --offline -- --ignored --nocapture
```

Substitute `baseline` and `no-msaa` for the other configurations. This deliberately
ignored test requires a real GPU and verifies drawing, not integrated FPS/TPS.

### Paired terrain-shader benchmark

For a new experiment, set `MECHANIC_TERRAIN_REFERENCE_SHADER` and
`MECHANIC_TERRAIN_CANDIDATE_SHADER` to explicit WGSL files to compare against the
actual starting shader. Without these variables, the existing frozen reference
and compiled-in runtime shader remain the defaults. Keep
`MECHANIC_PERF_TERRAIN_PASSES=0`. Require the pixel gate before an in-game capture.
The [single-projection experiment](performance-results/README.md#2026-09-05-terrain-single-projection)
was rejected at this gate and its runtime change was removed.

```sh
MECHANIC_RENDER_EXPERIMENT=baseline cargo test --release -p mechanic-app terrain_shader_preserves_pixels_and_measures_gpu_cost --offline -- --ignored --nocapture
```

Run without another Mechanic instance. This real-Bevy, offscreen test compares
the production shader against a frozen pre-optimization reference using authored
terrain maps and runtime mip generation. The target is 4096×2524 with 4× MSAA.
Cases cover flat grass, a 524,288-triangle grass patch, and sloped blends of all
six materials. Each case uses A/B/B/A order, 30 warm-up frames after shader
replacement, and 80 opaque-pass GPU samples per measurement. Pixel readback is
disabled during timing; timing comes from actual render-pass timestamps, not CPU
frame duration. Test-only blocking polls drain the GPU between frames.

The test prints paired medians and writes reference/candidate PNGs to a unique
`mechanic-terrain-shader-<pid>` directory under the system temporary directory.
It rejects pixel differences exceeding two 8-bit channel values. Performance is
reported rather than asserted because GPU clocks and other applications vary.
These are isolated shader measurements, not world FPS/TPS or scale-gate proof.

## Profiling captures

Reference captures use an M1 Pro at 1920×1080, an uncapped presentation mode,
and a release-equivalent binary with symbols:

```sh
cargo build --profile profiling -p mechanic-app
cargo build --profile profiling -p mechanic-app --features profiling-tracy
```

For CPU evidence, open Xcode Instruments, choose **Time Profiler**, and attach
to `target/profiling/mechanic-app`. Enter the world, allow the loading screen to
finish, then capture idle, fixed-path travel, and one continuous terrain-brush
stroke separately. Keep the camera path and world seed fixed and record the
capture duration with the benchmark JSONL.

For GPU evidence, launch the same binary from Xcode's Metal frame capture or
attach with **Metal System Trace**. Capture a fully streamed idle frame before
capturing travel or digging so shader and upload costs are not confused with
cold asset loading. Retain the display resolution, LOD distances, terrain
materials, and 1 km horizon used by the shipped configuration.

The headless deterministic terrain scenarios emit machine-readable JSONL:

```sh
cargo run --profile profiling -p mechanic-bench -- --scenario terrain_stream
cargo run --profile profiling -p mechanic-bench -- --scenario terrain_dig
```

The deterministic player/construction collision gate is CPU-only. The TEST2
vehicle gate is a separate GPU run with the captured 94-part, 9-body,
842-collider, 8-bearing topology:

```sh
cargo run --release -p mechanic-bench -- --scenario player_collision --seconds 30 --warmup 5
cargo run --release -p mechanic-bench -- --scenario test2_car --seconds 30 --warmup 5
```

The vehicle JSONL names all six phases and reports its immutable solver route,
configured iterations, planned/executed fused sweeps, per-tick CPU encoding,
submission/readback latency, in-flight slots, and visual-update cost. Its gate
also requires four active ground contacts, eight fused sweeps, residuals within
the existing physics tolerances, 60 TPS, and GPU physics p95 at or below 8.3 ms.
Each fused contact projection propagates through the contacted body's tree path
to the root and back (at most 64 bearings). Bearings solve their five constraints
as a block, and motors use the resulting constrained effective inertia. The
reported sweep count counts contact sweeps, including this path work; it is not
a count of individual bearing projections. The GPU test
`front_steered_car_turns_through_ground_friction` separately checks left/right
chassis yaw, straight-line drift, steering angle, and support while driving.
Small mechanisms with angle drives additionally use at least 32 velocity
projection iterations **before** advancing their joint angles. This resolves
wheel-drive/steering coupling without increasing the eight contact sweeps or
the accumulated per-tick motor torque budget. Mechanisms without angle drives
retain the configured velocity iteration count. The separate
`high_speed_steering_returns_to_center_after_release` GPU regression uses the
garage-built `front_steered_car.mech` fixture, holds both turn directions at
50 km/h, then checks return to center, ground support, and failure flags.
For the integrated 1920×1080 capture, wait for local terrain to read 36/36 and
streaming backlog to reach zero, then record 30 seconds. The F3 “Terrain selection worker”
row is asynchronous worker duration; “Terrain reselections” is its completion
frequency counter, not main-thread frame time.

It uses exactly 131,072 indexed static colliders (32 local candidates per
query) and 20,000 moving one-collider bodies. On the M1 Pro, warmed complete
query p95 must remain at or below 0.25 ms and dynamic top-level refit p95 at or
below 2.0 ms, with stable traversal/candidate capacities after warm-up. A
retained 1,800-sample release check after 300 warm-up samples on 2026-09-02
reported 0.009 ms query p95, 1.089 ms refit p95, 32 candidates, 1,800 contacts,
and passed both gates.

The articulated size sweep isolates the load-time serial/general route boundary:

```sh
for scenario in open_bearing four_bearing_contact bearings_16 bearings_64 bearings_65 bearings_256; do
  cargo run --profile profiling -p mechanic-bench -- --scenario "$scenario"
done
```

F3 splits **Physics CPU** into four non-overlapping
wall-clock stages: **CPU encoding** (config upload, commands, staging copies),
**CPU finalization** (`encoder.finish`), **CPU queue submit** (shared queue
submission, including pending uploads and resource maintenance), and **CPU
readback setup** (mapping callback registration, not completion). Each stage
is averaged per submitted tick before applying the same smoothing as the total.
The total also includes external-impulse dispatches and application bookkeeping;
later readback polling and terrain sampling are outside this timer. These are
wall-clock costs, so driver blocking and thread contention can contribute.
Missing stage measurements show as unavailable, not zero. No GPU waits are added.

F3 also separates **Window acquire** (the CPU schedule span around Bevy's
`prepare_windows`) from **Render/present CPU** (the span around `render_system`).
These include scheduling delays between the boundary systems, not just time
inside the named function. They are latest-frame samples, not the smoothed
physics CPU measurements. Window acquisition includes surface configuration
when needed and can block in Metal `nextDrawable`.

**Tracked GPU span** is the earliest start to latest end of the actual tracked
render passes in one completed sample. It is **partial coverage**, not a whole
render-graph or whole-frame GPU total; gaps and interleaved work can contribute.
Do not add it to physics GPU time. The previous independent compute markers,
and an experimental one-pixel graphics marker, both undercounted real raster
passes on Metal. The old 50–58 ms “Other GPU span” is not trustworthy attribution.

A narrow local patch to the pinned Bevy renderer attaches query pairs directly
to `RenderContext::begin_tracked_render_pass` descriptors. See
[`vendor/bevy_render/MECHANIC-PATCH.md`](../vendor/bevy_render/MECHANIC-PATCH.md)
for provenance and patch scope. Existing descriptor timestamps take precedence.
No extra draws, queue submissions or GPU waits are inserted.

Three asynchronous slots hold at most 128 pass pairs per sampled frame. Busy
rings skip sampling. GPU queries are allocated only while F3 is open; pending
readbacks can still drain after closing it. **GPU sample** reports waiting,
unsupported, readback failure, capacity overflow, or the first timestamp rejection
reason; **GPU pairs valid / total** reports coverage for that completed sample.
An invalid pair makes its entire pass group and the overall tracked span **N/A**,
but does not hide other fully valid groups from the same frame. Metal can return
a zero end timestamp for an empty shadow cascade; this is not zero-cost work and
must not be included as such. Equal nonzero start/end timestamps are valid
zero-tick intervals. Unsupported or overflowed samples remain wholly unavailable.
Every frame records its own pass metadata;
old query contents cannot masquerade as a missing camera's new measurements.
All GPU rows come from one completed slot. Group values union overlapping
intervals instead of double-counting them. Initial activation allocates query
and readback buffers; let it warm up before capturing steady-state performance.

F3 groups frame/render, physics, and terrain/collision into three columns and
labels GPU coverage **Tracked only**:

| Row | Measured scope |
| --- | --- |
| World target / viewport | Actual color texture and physical viewport dimensions; not logical window size. MSAA samples are shown separately. |
| Prepass + shadows | Tracked world prepasses and tracked shadow passes across light views. Does not include GPU preprocessing compute. |
| World opaque | Actual world opaque/alpha-masked pass, including any skybox. Terrain and vehicle drawing are not yet separated. |
| World transparency | Actual world transparent pass, when present. |
| X-ray tracked | Tracked graphics passes for the separate Mosaic/x-ray camera, excluding shadows. Not its raw post-processing, output, or UI passes. |
| Other tracked | Other intercepted render passes only. Not a subtraction from total time or an estimate of unmeasured work. |
| World post FX / Mosaic UI / World output / X-ray output | N/A: raw encoder passes and externally encoded Mosaic buffers bypass the hook. These passes are not exonerated by the tracked breakdown. |

Dimensions reflect the current prepared view and can briefly differ from an
older GPU sample while resizing. GPU intervals are not utilization counters.
See [the active checklist](performance-todo.md) for validation evidence,
integrated captures and remaining lock-contention work.

For the next comparison, keep resolution and camera framing unchanged, wait for
streaming backlog to settle (36/36 local readiness alone is insufficient), and
capture stationary and driving F3 measurements. Compare the pass breakdown
before choosing an optimization; do not infer the culprit from resident terrain
triangle count alone.

The application keeps at most three physics ticks in flight. Additional due
ticks remain in a monotonic CPU backlog, visible in the F3 overlay, until a
staging slot completes; a temporary slow frame therefore cannot amplify into
an unbounded Metal queue. On the M1 Pro, the dependency-safe single-dispatch
four-bearing contact route reduced the retained 10-second sample from 18.683 ms
GPU p95 and 44.69 TPS to 4.963 ms and 172.49 TPS with zero flags and residuals.
The load-time crossover is scene-specific: flat-ground scenes keep the fused
serial contact route only through four bearings because they can produce one
persistent contact per collider, while streamed-world scenes use it through 64
bearings because their contact set is normally sparse. Reduced-coordinate
velocity projection is fused through 64 bearings in both cases. A retained
M1 Pro no-ground sweep measured 16 and 64 bearings at 3.515 ms and 3.947 ms GPU
p95 with zero contacts, flags, or residuals. A longer flat-ground sweep measured
64 and 65 bearings at 5.452 ms and 6.143 ms GPU p95, so the dense-contact 4 ms
gate and 64/65 timing continuity remain open.

Do not infer the final frame, render-CPU, or render-GPU gates from the headless
terrain-stage numbers. Those gates require the integrated app capture on the
reference hardware.

## Matched fence-backport captures

The removable 29.0.4 backport is documented in
[`vendor/WGPU-FENCE-BACKPORT.md`](../vendor/WGPU-FENCE-BACKPORT.md).
It changes dependency-internal synchronization, not the physics schedule or
rendering quality. Do not call it an FPS/TPS improvement until the following
matched application comparisons pass.

Freeze the **dirty worktree**, including the current terrain shader and recorder,
into a new directory outside the repository, and build both release binaries:

```sh
python3 scripts/build-fence-comparison.py /tmp/mechanic-fence-pair --build
```

The baseline uses pristine cached crates.io wgpu-core/hal 29.0.4 sources. Both
snapshots use the same path patches and public dependency lockfile. The script
records every source SHA-256 and each completed binary SHA-256. It does not
reset, stash, commit, or change the original worktree. Keep the manifests and
build logs with captures; do not edit either snapshot afterward. Launch with the frozen asset root explicitly set (Bevy otherwise resolves assets
next to the copied executable), and use one instance:

```sh
cd /tmp/mechanic-fence-pair/baseline
BEVY_ASSET_ROOT="$PWD/crates/mechanic-app" \
MECHANIC_RENDER_EXPERIMENT=baseline \
MECHANIC_PERF_CAPTURE_DIR=/tmp/mechanic-fence-captures \
MECHANIC_PERF_LABEL=stationary-A1 \
../mechanic-app-baseline
```

Repeat from `candidate` with `../mechanic-app-candidate`. Copy a test world and
`car2.mech` before launching. Never use measurements from different saved world
states. Record world/vehicle hashes, starting position, camera, held tool,
preview visibility, and the exact driving route alongside the captures.

With `MECHANIC_PERF_CAPTURE_DIR` set, **F9** arms a capture. It waits until world
streaming backlog is zero and local readiness is complete, then warms up for
15 continuous seconds and records for 60 seconds. A renewed streaming backlog
resets warm-up. The log reports arming, start, and the output filename. Without
the variable, F9 has no new behavior. `MECHANIC_PERF_LABEL` is optional descriptive
metadata, not a rendering or simulation setting. Repeated F9 during capture is
ignored. Captures use at most 100,000 fixed-shape event records and write a unique
JSONL file only after completion. Orderly shutdown writes an invalid interrupted
capture; a hard kill can leave no file. Capacity exhaustion invalidates the file.
A missing/invalid completion footer must never be accepted as a measurement.

CPU frames, each physics submit/readback, render acquisition/render CPU events,
and completed asynchronous GPU samples are separate event types, stamped with
capture-relative observation time. GPU sample IDs and sample age identify delayed
results, including those originating before capture. Each completed slot is
recorded once, even if the overlay chooses a newer slot. No GPU waits are added.
F3 remains the switch for GPU queries and presentation mode: F3 closed captures
have CPU/physics events but no newly requested render GPU samples. Preserve F3
visibility throughout each run; do not interpret an absent GPU sample as zero
GPU time. Tracked GPU spans are partial coverage and must not be summed with
physics or overlapping pass groups.

```sh
python3 scripts/summarize-perf-capture.py /tmp/mechanic-fence-captures/*.jsonl
python3 scripts/test-perf-capture.py
```

The summary uses raw nearest-rank percentiles. TPS counts actual completion
and submission events over the 60-second interval; FPS uses raw CPU frame
intervals. It reports readback latency, starting/ending backlog, settings drift,
terrain work and GPU sample health separately. Review raw events for tick resets,
errors or interrupted simulation; a valid recorder footer certifies recording
completion, not a controlled experimental scene. Inspect `settings_seen` and
reject unmatched quality/presentation settings.

Run A/B/B/A for (1) the agreed stationary scene, (2) the same driving route through streamed
terrain, and (3) normal play with F3 closed. Primary conditions: **4112×2524,
4× MSAA, baseline materials**, unchanged terrain/solver settings. Keep F3 open
for primary runs (`AutoNoVsync`); closed runs retain `Fifo`. Repeat drifting
captures. Pair A1/B1 and B2/A2: require queue submission p95 at least 50% lower,
completed stationary TPS at least 10% higher in both pairs, and no more than 5%
FPS/frame-p95 regression. Driving must show no repeatable performance/control
regression. Record a fresh native CPU profile proving the acquisition-held
fence wait is gone. A smaller submission timer alone does not meet acceptance.

Window stress must additionally cover resize, minimize/restore, occlusion,
focus changes, and shutdown with outstanding work. Check seat entry/exit,
controller closing and cursor recapture. Require no hang, validation error or
device loss. Native Windows/Linux and real Metal window coverage cannot be
inferred from the NOOP dependency regression or offscreen GPU tests.

### TEST4 automated stationary setup (2026-09-05)

The user specified TEST4 with the existing `car2.mech` behind the player, out of
view but simulating, plus the existing placed blocks. Use the saved player pose;
do not enter or reposition the vehicle. The primary comparison therefore uses
this standing/offscreen scene in place of the original seated proposal.

For this run, `test4` was copied to `test4-perf` under the normal macOS world
store and only its manifest name changed to `TEST4 PERF`. The copy was reset
from the original between launches. Original scene hashes, frozen source hashes,
binary hashes, raw compressed JSONL, CPU profiles and results are retained in
[`performance-results/2026-09-05-test4-fence`](performance-results/README.md#2026-09-05-test4-fence).

Native macOS input selected the copy, set the outer window to 2056×1290 points
at (0,39), then pressed F3 and F9 after world loading. Captures verify that this
produced exactly 4112×2524 window/target/viewport pixels and 4× MSAA. The held
tool was Matter Manipulator / Block / Steel, camera unchanged, no placement
preview visible. CPU profiles used `sample PID 5 1 -file PATH` **after** recording;
they do not overlap the raw capture interval. The archived automation script
records this specific session, including its already-running first baseline;
it is not a general-purpose launcher.

To summarize archived captures, decompress them into a temporary directory and
run the existing raw-sample summarizer:

```sh
mkdir -p /tmp/mechanic-test4-results
python3 - <<'PYCODE'
import gzip
from pathlib import Path
for source in Path("docs/performance-results/2026-09-05-test4-fence").glob("capture-*.jsonl.gz"):
    (Path("/tmp/mechanic-test4-results") / source.stem).write_bytes(gzip.decompress(source.read_bytes()))
PYCODE
python3 scripts/summarize-perf-capture.py /tmp/mechanic-test4-results/*.jsonl
```

## Unattended background capture

For the compiled-dynamics redesign, source-matched foreground captures use the
same runner with explicit foreground selection:

```sh
python3 scripts/build-performance-reference.py --output .physics-reference/foreground-reference
python3 scripts/run-background-capture.py --foreground --drive \
  --binary .physics-reference/foreground-reference/mechanic-app \
  --identity .physics-reference/foreground-reference/identity.json \
  --assets .physics-reference/foreground-reference/source/crates/mechanic-app \
  --world .physics-reference/2026-09-09/car \
  --output .physics-reference/foreground-run-01
```

The build uses a copied source tree and a fresh isolated target directory.
Shared Cargo artifacts cannot establish provenance across dirty worktrees with
preserved file timestamps. The launcher checks binary and asset hashes, records
all world-file hashes, and opens a disposable store under the results directory.
It strips inherited Mechanic diagnostic switches. Foreground mode requests
focus and rejects a capture if any measured frame loses focus, changes render
settings, or fails to render at 4112x2524 with baseline 4x MSAA/AutoNoVsync/F3.
Demonstration screenshots and scripted placement are prohibited in this mode.

An earlier reference build reused the application's release target and left a
stale `mechanic-core` artifact. This can report a missing `CompiledCreation::dynamics`
field even though it exists in the current source. The existing cache was repaired
with `cargo clean -p mechanic-core --release`, followed by
`cargo build --release -p mechanic-app`. New reference builds use isolated target
directories; do not share the application's target with archived source trees.

Foreground graphics warm-up holds the loaded physical state; driving advances
once per dispatched physics tick and starts with 180 settling ticks. This
protocol differs from historical background captures and must match on both
sides of a comparison. Initial camera/state identities and completed-state
hashes accompany the trace. Terrain-readiness holds are recorded explicitly.
The external clock remains 60 Hz. Without `--replay-ticks`, the nominal capture
interval is 60 wall-clock seconds; dropped ticks still invalidate workload
matching and performance acceptance.

Use `--replay-ticks 1800` with foreground driving for a fixed 30-simulated-second
workload (or 3600 for 60 simulated seconds). The unchanged 60 Hz clock retains
every overdue tick, including time spent waiting for terrain. The run submits
exactly that count, then drains readbacks without GPU waits. This is a replay
protocol, not a change to normal application scheduling. A growing backlog or
low wall-clock TPS still fails performance acceptance even with zero dropped ticks.

Capture schema 3 records `measurement` and `drain` phases. Wall-time captures
close at the end of the frame crossing 60 seconds, preserving the complete last
batch. Both protocols wait for the last actually submitted tick to be published
and the readback ring to empty. Failure or a ten-second drain timeout invalidates
the result. Total replay TPS includes drain latency; measurement frame/render
distributions exclude drain samples. Use archived tools for historical captures;
the new summarizer deliberately rejects older schemas.

Initial terrain fingerprints cover ordered packed geometry/materials/generations
and reachable GPU allocation layout. Effective drive rows are hashed for every
scripted tick. These identities supplement the initial state/camera and source
world hashes when diagnosing repeatability.

`scripts/compare-perf-captures.py --repeat A.jsonl B.jsonl` checks identical-build
repeatability against complete submitted workloads. A shorter shared prefix or
an unpublished boundary tail does not pass. Initial terrain geometry/layout and
effective drive rows are also compared. The first state and drive divergences
are reported separately.
Neither passing the foreground protocol nor matching hashes establishes full
kernel coverage or physical-bound correctness.

The [fixed-duration replay evidence](performance-results/README.md#2026-09-10-fixed-replay)
verifies complete submission/readback/publication for two 1,800-tick Metal runs.
Both fail throughput and completed-state repeatability; retain those failures
when comparing the replacement solver.

Use this for the standing TEST4 scene while continuing to use the computer.
It drives world loading and recording inside Bevy, saves a Bevy-rendered PNG,
and exits without OS mouse clicks, keypresses or screenshot commands:

```sh
cargo build --release -p mechanic-app
python3 scripts/run-background-capture.py \
  --world "$HOME/Library/Application Support/Mechanic/worlds/test4" \
  --output /tmp/mechanic-test4-background-01
```

The output directory must be new. The launcher copies the world to a uniquely
named directory in an isolated disposable store and changes its manifest identity.
Autosaves affect only the copy, which is removed after exit or timeout. Results
remain in the output directory: `run.json`, `app.log`, raw capture JSONL,
`summary.json`, and a PNG with the same basename as the capture. `--binary` and
`--assets` can select another build; the asset root must contain `assets/`.

The opt-in `MECHANIC_AUTO_WORLD` interface is intended for this launcher. It
names an existing **test copy**, requires `MECHANIC_PERF_CAPTURE_DIR`, opens only
a current-format world, enables F3/AutoNoVsync, and arms the existing 15-second
continuous-ready warm-up plus 60-second capture. The screenshot is requested
only after recording finishes, and the app exits only after the image is saved.
Missing/outdated worlds, invalid captures and screenshot failures exit nonzero.
The app has a five-minute deadline; the launcher kills/reaps an unresponsive
process after six minutes before removing its copy. Logs survive failures.

Automation creates an unfocused 4112×2524 window, keeps Bevy updating while
unfocused, suppresses live keyboard/mouse input, and releases cursor capture.
The small vendored `bevy_winit` change exposes winit's existing macOS launch
activation option; normal launches retain their previous behavior. No native
App Nap override is installed. A completely occluded/minimized window can still
behave differently or stall on macOS; deadline failures are reported, not
silently converted into successful measurements.

These are **background diagnostics**, not controlled foreground performance
benchmarks. Metadata and every frame carry `automated_background: true`; frames
also record `focused`. Other applications compete for GPU/CPU resources and
macOS controls drawable availability. Do not mix these timings with the earlier
foreground A/B/B/A acceptance results. The screenshot uses the window's render
target through Bevy; this is not a headless/offscreen rendering mode and does
not bypass Metal drawable acquisition. This runner automates the stationary
scene only; it does not claim driving or seat/controller interaction coverage.

Verification commands:

```sh
python3 scripts/test-background-capture.py
python3 scripts/test-perf-capture.py
cargo test -p mechanic-app automation_discards
cargo test -p mechanic-app performance_capture::tests
cargo clippy -p mechanic-app --all-targets -- -D warnings
cargo fmt --all -- --check
```

## Editing stalls: scripted placement captures

Placing a block republishes the whole world physics scene. Use scripted
placements to capture that path without OS input:

```sh
cargo build --release -p mechanic-app
python3 scripts/run-background-capture.py --place 5 \
  --world "$HOME/Library/Application Support/Mechanic/worlds/<world>" \
  --output /tmp/mechanic-placement-01
```

`--place SECONDS` (`MECHANIC_AUTO_PLACE`) commits one disposable cuboid above
the construction on that cadence, from the moment the world starts playing, so
a world whose construction is entirely static still reaches a running
simulation. Placements republish the scene, so the tick sequence is not
continuous and `summarize-perf-capture.py` is skipped for these runs; read the
raw JSONL instead. A capture that never starts logs its phase, world notice and
terrain readiness every two seconds — an outdated saved world reports the
unsupported creation version there rather than timing out silently.

Records attributing the edit path: `scripted_placement`, `world_physics_request`,
`world_physics_prepare` (worker `compile_ms`/`scene_ms`), `world_physics_install`
(main-thread `install_ms`), `visual_mesh_sync`, and `terrain_publication`
(`snapshot_ms` and `publication_ms` are main-thread; `reused_chunks` versus
`uploaded_chunks` shows whether resident terrain was kept).

A replacement scene inherits the retired scene's terrain: the packed chunk cache
and the uploaded device buffer move across, and the buffer is rebound
immediately, so an edit publishes no terrain at all. Before that, every
placement in the suspension world repacked and re-uploaded the whole cut —
3140 chunks, 1.07 GB, roughly two seconds of main-thread snapshotting plus a
second of upload — and frame p99 was 4.7 s. After it, placements do no terrain
work, frame p50 fell from 95 ms to 38 ms, and worst-case frames fell from 7.3 s
to 0.36 s. The remaining steady-state cost of that world (6.5 M resident terrain
collision triangles, ~55 ms physics GPU per frame) is a separate problem.

Focused verification:

```sh
cargo test -p mechanic-gpu adopted_terrain_residency_collides_without_uploading_chunks_again
cargo test -p mechanic-app inheriting_keeps_the_accepted_cut
```

## Suspension car physics and World driving

`cargo run --release -p mechanic-bench --bin suspension-world -- --seconds 60`
loads the installed suspension car against production-generated terrain; `--plane`
selects the control. These are 60 simulated seconds with serialized per-tick
readbacks, not a World frame/backlog gate. Use
`scripts/summarize-suspension-benchmark.py` for completed JSONL traces.

The background World runner accepts `--drive` to exercise the sole input-linked
seat through application-level W/A/D states. `--demonstration` additionally saves
frames every five seconds and perturbs frame timing. Physics readbacks include
terrain contact generation, rotational sweep, positional recovery, and nested
recovery projection timings. Terrain publication records worker preparation,
publication latency, upload/copy bytes, and reused chunk counts. The [performance
report](performance-results/README.md#2026-09-07-suspension-world) records the unmet
acceptance gate and the older baseline's GPU timestamp limitations.
