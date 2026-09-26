# Vehicle performance checklist

The active architectural redesign follows
[Compiled machine dynamics](compiled-machine-dynamics.md), including its current
checkpoint and exact acceptance gates. The dated profiling pauses below are
historical; they do not pause the authorized CPU/GPU and production-rendering work.

## Declarative worldgen streaming (2026-09-25)

After the switch to declarative 3D world generation, walking through a world
fell to 5.8 FPS:

| Measure | Value |
|---|---|
| Resident terrain triangles | 38.6M |
| Terrain publication (main thread) | 44 ms per frame |
| Streaming backlog | 21.7k |

Nearby chunks were missing, their neighbours' caps showing as pits.

- [x] Tighten region bounds so selection stops meshing empty nodes: blend
  bounds from biome weight ranges, Taylor noise bounds, and distance-aware
  river bounds. Nodes per cut fell from 13–29k to 9–18k.
- [x] Add LOD level 6 (3.2 m samples) from 640 m to the 1 km horizon. Tie
  caves to level ≤ 2 in both selection and meshing.
- [x] Order streaming by distance ring and view before seams, and cancel
  stale jobs. `walking_never_uncovers_nearby_ground` covers holes.
- [x] Make publication incremental: replacement groups, a running triangle
  total, no main-world copy of terrain meshes.
- [x] Budget resident terrain triangles (about 6M) with a stepped detail
  scale.
- [x] Stop breakage from re-sampling untouched ground every tick. That was
  the largest main-loop cost once the new generator made point sampling
  expensive.
- [x] Give async compute half the cores (5 on the M1 Pro), not Bevy's
  quarter, and keep 6 jobs queued per worker.
- Background capture of the reporter's world on Apple M1 Pro / Metal, taken
  as a diagnostic rather than an acceptance run:
  - The median frame fell from 152 to 63–80 ms, and GPU opaque from 37.7 to
    20–25 ms.
  - Sampling shows every thread mostly idle. The remaining frame time is the
    background window's presentation pacing, so the next step is a
    controlled foreground capture.
- [ ] Controlled foreground capture while walking across biomes.
- [ ] Reduce per-chunk sampling for scattered 3D-warped shapes. The
  heaviest biomes still take 20–35 ms p95 per chunk.
- [x] Carve layers (tunnels, entrances, shafts, ravines, trenches) keep
  their cost bounded: `Fissure` bounds from the centre's value, gradient and
  curvature; per-block skipping of voids more than 16 m away; planar values
  shared by blocks stacked over the same columns; stack evaluation of small
  scatter shapes. A full cut still costs 1.5–1.9× the carve-free world,
  about half of it real tunnel and ravine geometry near the player.

## BLOB physics optimization (2026-09-08)

- [x] Limit World physics terrain publication to snapped, generously margined
  interest regions around compiled body positions. Keep ticking on an accepted
  cut; block before first publication, across rebases, and when travel exhausts
  the 48 m safety allowance.
- [x] Skip terrain BVH traversal for immovable colliders. Their terrain contacts
  were already discarded by contact preparation because inverse mass is zero.
- [x] Expand the asynchronous readback ring from three to 12 slots and cap wall
  time catch-up at 30 ticks, reporting dropped ticks in F3 and captures.
- [x] Keep a dynamic vehicle backlog from starving render submission: at most
  three physics ticks are submitted per rendered frame. In the driven `car`
  save this raised background FPS 8.59 → 9.95 and cut frame p95 241.69 →
  132.89 ms with all eight sweeps and zero failure flags.
- [x] Use the exact closed-form spring/damper solution during ordinary travel;
  retain all eight Newton iterations whenever the rubber bump stop engages.
  The ten-test Metal suspension suite passes.
- [x] Restore dynamic-root mechanisms to the established fully separated
  terrain-recovery route after a driven car fell through terrain. A deterministic
  60-second straight full-throttle real-save run stays grounded; the exact
  intermittent failure did not reproduce in the A/B automation runs.
- [x] Move terrain streaming with an occupied seat's simulated pose. Previously
  it stayed at the seat-entry point, so vehicles eventually outran the resident
  collision meshes. Prioritize that same local region and hold physics whenever
  its current terrain is unresolved. The `car` save stayed grounded through a
  60-second Metal capture spanning 18 reselections and 37 collision publications.
- [x] Measure the real BLOB world before and after on Apple M1 Pro / Metal.
  [Evidence](performance-results/README.md#2026-09-08-blob-physics): first terrain
  publication 41.81 → 11.31 s, terrain-contact GPU p50 537.21 → 0.0557 ms,
  whole-physics GPU p50 545.17 → 6.73 ms, completed TPS 0.47 → 48.60, final
  backlog 422 → 0, zero dropped ticks and zero failure flags after the change.
  Frame rate remains render-bound near 19 FPS; render quality was not changed.
- [x] Run the prescribed headless comparisons and archive failures without
  weakening gates. `test2_car` and the suspension plane control pass. The longer
  `four_bar` run sets the closure flag, `bearings_4` misses its 4 ms p95 budget,
  and `dense_100k` regresses with far more active contacts. Those fixtures do not
  execute the optimized terrain path and remain broader-worktree follow-up work.
- [ ] Diagnose the headless `four_bar`, `bearings_4`, and `dense_100k` regressions
  as a separate physics pass. Do not weaken tolerances, iteration counts, or
  coverage gates.

## Paused after profiling

The fence backport and opt-in diagnostics are retained. No new terrain shader
candidate passed the quality/performance gates; the single-projection candidate
was removed. Material normal maps must remain independent; sharing normal-map
samples across materials is not an accepted optimization direction.

Further optimization is paused pending stronger GPU attribution within terrain
(texture bandwidth, fragment arithmetic, or geometry), ideally from a working
Metal capture. Existing correctness failures and performance gates below remain
open. Background diagnostics must not be presented as controlled acceptance runs.

## Latest remaining-wait investigation

- [x] Collect fresh background TEST4 raw timings and two native CPU stack profiles.
  [Evidence and limitations](performance-results/README.md#2026-09-05-background-bottleneck):
  opaque p50 20.7 ms, acquisition 26.4 ms, physics readback 96.9 ms, all three slots
  occupied. Main-thread render-world handoff and Metal drawable waits remain.
  Current scene has 20 bodies; do not compare throughput with the earlier 17-body scene.
- [x] Separate mapping-callback observation from app publication with opt-in
  timestamps and empty/successful poll records, preserving polling frequency.
- [x] Attribute opaque cost to terrain versus other draws before choosing the
  next quality-preserving render optimization. Native xctrace still exits 137;
  exact GPU queue/execution/presentation timeline remains unavailable.
  [Callback and terrain evidence](performance-results/README.md#2026-09-05-callback-terrain-attribution):
  opt-in actual-pass partition measures terrain p50 21.27 ms versus other opaque
  0.89 ms; native pixel comparison is byte-identical. Normal-path callback-to-
  publication p50/p95 1.13/33.29 ms, with 26.2% above one physics timestep.
  First run reported focus, second stayed unfocused: no paired performance claim.
- [ ] Choose and measure one quality-preserving terrain-rendering candidate,
  then compare integrated results with terrain partitioning disabled and matched
  scene/focus/presentation conditions. Continue reporting publication-delay tails.
  - [x] Test a single-projection UV/gradient sampling fast path.
    [Rejected](performance-results/README.md#2026-09-05-terrain-single-projection):
    blended-terrain maximum channel delta 24 (gate 2), about 6% slower in that
    fixture, and no repeatable flat-terrain gain. A/A control is byte-identical.
    Runtime shader restored exactly; no integrated run or gain claimed.

## Evidence

- Release app, Apple M1 Pro / Metal: seated driving around 20 FPS and 24–27 TPS.
- F3: about 27 ms in queue submission despite roughly 6 ms GPU physics.
- Seated CPU capture: 420/450 physics-submission samples waiting on a wgpu
  read/write lock. Concurrent drawable acquisition holds the device fence read
  lock while waiting in Metal `nextDrawable`; submission needs its write lock.
- Separate captures: repeated cursor-grab calls occupied 13–14% of main-thread
  samples. Camera code writes unchanged cursor options every frame.
- Post-diagnostic driving screenshot (2026-09-05): 19.4 FPS, 55.84 ms frame
  average / 67.16 ms p95, 26.5 TPS, physics backlog 1,191. Window acquisition
  29.19 ms, render/present CPU 3.27 ms, GPU render span 42.34 ms. Physics CPU
  30.26 ms, including 28.91 ms queue submission; physics GPU 6.42 ms. Eight
  sweeps and zero failure flags. These overlapping/differently sampled spans
  must not be added together.
- This screenshot supports the previously sampled drawable/fence contention,
  but does not isolate the expensive GPU render pass or prove a cursor gain.
  Terrain is locally ready (36/36), but streaming backlog remains 1,123;
  4,425,705 terrain triangles are resident, not a measured visible draw count.
- First pass-breakdown capture: 4112×2524 physical target/viewport, 4× MSAA;
  13.8 FPS, 21.0 TPS. GPU render span 52.37 ms: pre/shadows 1.17 ms, opaque
  0.17 ms, post FX 0.09 ms, UI 0.08 ms, other 50.85 ms (about 97%). Window
  acquisition 52.65 ms, physics submission 44.80 ms, physics GPU 4.97 ms.
  Streaming backlog 2,276 and zero physics failure flags. This near-idle
  capture shows 3.0 km/h; an initially duplicated driving attachment was later
  corrected.
- Corrected driving capture: 45.7 km/h, 11.5 FPS, 17.8 TPS, frame average
  75.36 ms / p95 95.64 ms. Same 4112×2524 target and 4× MSAA. GPU render span
  59.63 ms: pre/shadows 1.13 ms, opaque 0.30 ms, post FX 0.11 ms, UI 0.24 ms,
  other 57.85 ms (about 97%). Window acquisition 41.63 ms, physics submission
  28.56 ms, physics GPU 6.19 ms, submission-to-readback 140.58 ms, physics
  backlog 1,389. Terrain backlog 1,120, resident triangles 4,431,959, local
  readiness 36/36, eight sweeps and zero failure flags. Both speed samples show
  the same attribution gap. Neither is fully streamed, and framing/geometry
  differ, so the change cannot be attributed to driving speed alone. Validate
  raster timing and classify the residual before requesting further captures.
- Real 1024×1024, 4× MSAA raster regression invalidated the external markers:
  compute markers measured 1.339 ms around a 3.518 ms actual pass. An experimental
  one-pixel graphics marker also undercounted under concurrent GPU tests:
  3.368 ms around a 7.567 ms pass. Previous GPU span/breakdown screenshots are
  not reliable attribution; their CPU counters remain separate evidence.
- Native profiling fallback: `xcrun xctrace version` and `list templates` both
  exited 137 with no output. User approved a pinned local `bevy_render` patch
  instead, attaching timestamp pairs to actual tracked render-pass descriptors.
- Fully streamed comparison: both 4112×2524, 4× MSAA, 6,235,373 resident terrain
  triangles, backlog 0, local readiness 36/36, terrain stage 0 ms. Near-idle
  (0.2 km/h): 19.7 FPS, 29.1 TPS, acquire 38.36 ms, submit 22.40 ms, GPU physics
  5.92 ms, physics backlog 8,787. Driving (57.4 km/h): 22.0 FPS, 27.4 TPS,
  acquire 26.86 ms, submit 22.29 ms, GPU physics 5.53 ms, backlog 10,946.
  The bottleneck persists without streaming; elapsed time between captures is
  unknown, so these do not quantify physics backlog growth per second.
- All tracked GPU rows were N/A in both captures. Reproduced with Bevy's real
  offscreen PBR/extraction/diagnostics pipeline on Metal: four shadow cascades
  and one opaque pass were recorded; the fourth cascade repeatedly had a
  nonzero start and zero end, while opaque had a valid pair. The all-or-nothing
  parser discarded the entire sample. Preserve complete groups, invalidate
  affected groups and total span, and display sample status and valid-pair counts.
- Corrected partial-sample capture: 17.6 FPS, 29.1 TPS, 4112×2524, 4× MSAA;
  streaming backlog 0 and terrain stage 0 ms. Valid timestamp pairs 7/8:
  prepass/shadows 0.64 ms, opaque 31.45 ms, transparency 30.43 ms, x-ray
  0.80 ms. These pass intervals may overlap and must not be summed. Physics
  GPU 6.11 ms versus CPU queue submission 23.84 ms; window acquire 49.91 ms.
  Avatar materials remain in AlphaMode::Blend even at full opacity and are
  rewritten every frame. Test removing this unnecessary transparent work;
  the capture alone does not establish its incremental GPU cost.

- Resolution comparison after the avatar fix: full 4112×2524 versus 2058×1236,
  both 4× MSAA, 6,253,341 resident terrain triangles, streaming backlog 0 and
  complete tracked samples (7/7). FPS 18.5 → 29.8, TPS 25.1 → 42.8,
  opaque 35.11 → 13.69 ms, tracked GPU span 46.07 → 18.77 ms,
  window acquire 58.08 → 28.09 ms, GPU physics 6.45 → 5.62 ms.
  Strong resolution-dependent cost, but not proof of texture sampling versus
  MSAA bandwidth; scene framing/aspect and asynchronous samples are not exact.
  Transparency was absent after the avatar change, without a demonstrated gain.

- Full-size render experiments: baseline / no-MSAA / simple-terrain at
  4112×2524, complete tracked samples (7/7), streaming backlog 0, terrain
  stage 0 ms, local readiness 36/36 and 6,235,373 resident terrain triangles.
  FPS: 20.0 / 30.0 / 30.9; TPS: 30.2 / 46.1 / 47.2;
  frame average: 41.30 / 30.35 / 29.47 ms;
  opaque: 24.18 / 12.60 / 15.09 ms;
  tracked GPU span: 30.72 / 19.11 / 19.23 ms;
  CPU queue submission: 18.83 / 12.15 / 13.82 ms.
  Simple terrain retains 4× MSAA and the visible held tool, supporting
  substantial terrain-material cost. No-MSAA reports one sample but its held
  tool is absent, so its improvement cannot be attributed entirely to MSAA.
  These are individual captures, not repeated benchmark averages; savings
  cannot be added or extrapolated to a combined mode. Physics throughput
  improves without solver changes, but remains below 60 TPS.

- Post-consolidation world capture: baseline, 4112×2524, 4× MSAA, complete
  9/9 pairs, streaming backlog 0, terrain stage 0 ms, readiness 36/36 and
  6,235,373 resident triangles. FPS 20.4, TPS 23.2, frame average 43.11 ms,
  p95 57.08 ms, opaque 23.83 ms, tracked span 32.70 ms, queue submission
  18.09 ms, window acquisition 31.22 ms and GPU physics 5.57 ms.
  Against the earlier baseline (20.0 FPS, 30.2 TPS, opaque 24.18 ms), this
  does not demonstrate a meaningful integrated improvement. Framing differs
  and a translucent placement preview is visible: transparency now reports
  25.13 ms, whereas the earlier baseline had no transparent pass. Pass spans
  can overlap; this is not evidence of an additional 25 ms preview cost.
  Prior sampled drawable/fence contention remains the next scheduling target;
  isolated shader savings must not be presented as demonstrated world gains.

## Work items

- [x] Stop redundant cursor-option updates. Regression reproduced before the
  fix; 24 camera tests pass, including unchanged frames, panel release/recapture,
  and externally changed cursor options.
- [ ] Manually verify cursor capture across seat entry/exit, controller closing,
  and window focus changes in the release app.
- [x] Separate Window acquire and Render/present CPU spans. Keep GPU collection
  bounded to three asynchronous slots, with no queries when F3 is closed.
- [x] Validate replacement actual-pass GPU timing hook on Metal. Label coverage
  as tracked passes only; raw post-processing, output and Mosaic UI remain N/A.
  Do not retain the old external-marker measurements as verification evidence.
- [ ] Capture seated driving after those changes, including fully streamed idle
  and driving. Record FPS, TPS, CPU queue submission, GPU render time, and
  physics backlog growth. Do not substitute headless TPS for app FPS.
  - Fully streamed idle/driving comparison received and recorded above; corrected
    GPU attribution and physics backlog growth per second remain outstanding.
- [ ] Identify and reduce the measured rendering/presentation bottleneck.
  - [x] Add physical world target/viewport resolution and MSAA. Three-column F3
    layout keeps the breakdown and existing counters visible.
  - [x] Reproduce invalid external marker attribution around real raster passes.
    Replace markers with a narrow actual-pass hook in pinned local Bevy source.
    Add world transparency, x-ray tracked work and other tracked pass groups.
  - [x] Reproduce all-N/A output in the real PBR pipeline. Isolate incomplete
    timestamp groups and expose rejection status plus valid/total pair counts.
    Missing shadow timestamps must not hide valid opaque measurements.
  - [x] Use opaque avatar materials at full visibility, preserve the camera
    fade, and skip unchanged material writes. Both regressions failed before
    the fix; opacity transitions and actual asset modification events now pass.
  - [ ] Compare the avatar change at fixed framing/resolution after restart;
    disappearance of a pass alone is not proof of an FPS gain.
  - [x] Add mutually exclusive, launch-only `no-msaa` and `simple-terrain`
    experiments with an F3 mode label. Default rendering remains unchanged;
    neither experiment is saved. See `performance-profiling.md` for commands.
  - [x] Measure both experiments at full size against an explicit baseline.
    Simple terrain bypasses material texture sampling and normal mapping while
    retaining geometry, forward lighting, fog, texture loading and 4× MSAA.
    No-MSAA changes both cameras sharing the target, leaving materials unchanged.
  - [x] Consolidate projection branches across the terrain color, ORM and
    normal maps, preserving samples, normal blending, mip selection and lighting.
    Paired real-material release test at 4096×2524, 4× MSAA on Apple M1 Pro / Metal:
    reference/candidate opaque medians (two A/B/B/A measurements per case):
    flat grass 3.274/3.134 vs 2.818/3.048 ms; dense 524,288-triangle grass
    5.787/5.775 vs 5.127/5.236 ms; blended slopes 5.104/5.010 vs 4.278/4.291 ms.
    Pixel max channel deltas were 0 / 0 / 1 respectively. Shared explicit
    gradients were also tested but discarded: no consistent measured gain.
    Final repeat with PNG exports and a strict maximum channel delta of two:
    flat 3.393/3.425 vs 2.734/3.070 ms; dense 9.676/9.625 vs 9.396/9.465 ms;
    blended 9.408/9.316 vs 7.149/7.765 ms. Pixel deltas remained 0 / 0 / 1.
    Timings varied between runs; the dense-case saving was only about 2% in
    the repeat. Exported reference/candidate blends were visually inspected.
    Mip generation, zero-weight material skipping and distant projection
    reduction already exist; do not propose them as new fixes.
  - [ ] Verify the consolidated terrain shader in the normal release world,
    at matched framing/resolution with streaming settled. The isolated GPU
    result is not yet evidence of integrated FPS/TPS improvement.
    First post-change capture received above; no meaningful gain demonstrated,
    and preview/framing differences prevent a controlled before/after verdict.
  - [ ] Repeat the MSAA comparison with matching held-tool visibility before
    attributing the full measured gain to MSAA or changing rendering defaults.
  - [ ] Extend upstream hooks or obtain a native capture if expensive raw
    post-processing, final output/compositing or Mosaic UI remain unmeasured.
    Do not attribute the former 50–58 ms residual to any subsystem without proof.
  - [ ] Capture the new breakdown at fixed resolution/framing, stationary and
    driving, after streaming settles. Then use controlled comparisons to
    distinguish pixel shading from geometry cost.
    Terrain already has `NotShadowCaster`; do not assume all resident terrain
    triangles are also being rendered into shadow maps.
- [ ] Evaluate scheduling or an upstream wgpu lock-scope change so drawable
  acquisition cannot stall physics submission. Verify a real FPS/TPS gain, not
  just movement of the wait into another timer.
- [ ] Repeat vehicle correctness and performance checks after any scheduling fix.

## Constraints

Keep steering torque, 32 pre-integration angle iterations, and eight contact
sweeps unchanged. Preserve collision behavior and visual quality. No changes to
saved vehicles. Cursor and diagnostic work does not by itself resolve the lock.

## Verification

- Consolidated shader: `MECHANIC_RENDER_EXPERIMENT=baseline cargo test --release
  -p mechanic-app terrain_shader_preserves_pixels_and_measures_gpu_cost --offline
  -- --ignored --nocapture` passed twice on Apple M1 Pro / Metal. Final run
  exported comparison PNGs under system temp `mechanic-terrain-shader-32419`.
  No-MSAA real-material pixel regression also passed on the same adapter.
  Focused CPU gates: 55 rendering tests, 3 terrain tests, 3 experiment tests;
  app all-target Clippy with warnings denied, formatting and whitespace passed.
- Rendering experiments: 3 configuration tests, 10 performance/UI tests,
  55 rendering tests and 24 camera tests passed. Clippy (`mechanic-app`, all
  targets, warnings denied), formatting and diff whitespace checks passed.
- Apple M1 Pro / Metal: `MECHANIC_RENDER_EXPERIMENT=<mode> cargo test -p
  mechanic-app terrain_experiment_renders_pixels_with_the_real_material --offline
  -- --ignored --nocapture` passed separately for `baseline`, `no-msaa`, and
  `simple-terrain`. The fixture uses the app's tonemapper, compares against an
  empty target and rejects error/loading output. Settled RGBA center pixels:
  baseline/no-MSAA [252,251,252,255], simple terrain [204,227,167,255], empty
  [0,0,0,255]. This proves actual material drawing, not app FPS/TPS gains.
- `cargo test -p mechanic-app rendering_tests:: --offline -- --nocapture`:
  55 passed, including opaque/fading/hidden avatar material transitions and
  zero material modification events on repeated unchanged updates. These are
  CPU material/event tests, not GPU performance or visual validation.
- Avatar follow-up: 24 camera tests, `cargo clippy -p mechanic-app --all-targets
  --offline -- -D warnings`, formatting and diff whitespace checks passed.
- `cargo test -p mechanic-app camera::tests -- --nocapture`: 24 passed; the new
  unchanged-cursor test failed before the conditional-write fix.
- `cargo test -p mechanic-app performance::tests --offline -- --nocapture`: 9 passed,
  including physical pixel dimensions, separate pass values, unavailable values,
  pointer-transparent overlay layout at 1280×720 and 1600×900, and per-tick CPU
  stage averages, and partial-sample status/valid-pair presentation.
- `cargo test -p mechanic-app -p bevy_mosaic --lib -- --nocapture`: all 18 Mosaic
  bridge library tests passed with the application's unified dependency features.
- `cargo test -p mechanic-app render_diagnostics::tests --offline -- --nocapture`:
  7 passed on Apple M1 Pro / Metal, including actual 1024×1024 4× MSAA raster
  work, world/x-ray attribution, three-slot busy-ring skipping, disabled queries,
  missing-view reuse, preserved descriptor timestamps, allocator overflow, and
  overlapping interval accounting. The real offscreen Bevy PBR regression
  failed before the partial-group fix and now publishes valid opaque timing
  with status Partial(MissingEnd), 4/5 valid pairs. Parser tests cover missing,
  reversed/stale, truncated and equal nonzero timestamps without discarding
  unrelated valid groups. Supersedes the old external-marker suite.
- Integrated FPS/TPS gains and manual cursor/focus behavior are not yet verified.
- `cargo clippy -p mechanic-app -p mechanic-gpu -p bevy_mosaic --all-targets -- -D warnings`,
  `cargo fmt --all -- --check`, and `git diff --check`: passed.

## Fence backport phase (2026-09-05)

- [x] Preserve the supplied dirty source with SHA-256 fingerprints before edits.
  Local archive: `/tmp/mechanic-fence-comparison/original`; fingerprint manifest
  `original-fingerprints.json`. Existing terrain, rendering, and physics changes
  remain intact.
- [x] Vendor and patch both wgpu-core/hal 29.0.4. Public wgpu/Naga versions and
  Bevy/Mosaic revisions are unchanged. `cargo tree --offline -i wgpu-core` and
  `-i wgpu-hal` confirm one stack shared by all consumers; raw tree is
  `/tmp/mechanic-fence-stack.txt`.
- [x] Add bounded opt-in F9 raw capture and raw percentile summary. Three recorder
  tests and four Python summary tests cover capacity, expiry, interruption,
  filenames, percentile ranking, invalid files, nonconsecutive ticks and duplicate GPU samples.
- [x] Reproduce baseline acquisition/submission coupling at the HAL boundary.
  Pristine 29.0.4 fails `submission stalled behind HAL acquisition` after one
  second, cleans up and exits. Raw log:
  `/tmp/mechanic-fence-regression-baseline-final.log`.
- [x] Candidate dependency regression passes with concurrent submit/poll/present
  and 100 reentrant completion callbacks, normally and with `wgpu_validate_locks`.
  Logs: `/tmp/mechanic-fence-regression-final.log`,
  `/tmp/mechanic-fence-ranked-final.log`. This is NOOP/core synchronization proof,
  not a real drawable/window stress result.
- [x] Workspace Clippy with warnings denied and formatting pass. Clippy log:
  `/tmp/mechanic-fence-clippy-final.log`.
- [ ] Clear real Metal workspace correctness gates. Host: Apple M1 Pro, 16 GPU
  cores, vendor 0x106b, Metal supported (`/tmp/mechanic-fence-adapter.txt`).
  The sandbox had no GPU; its hardware skips are not correctness evidence.
  Host `cargo test --workspace --offline -- --test-threads=1` stopped after an
  app CPU timing assertion (11.357 ms mesh publication during concurrent release
  compilation): 551 passed, 1 failed, 2 ignored. Log:
  `/tmp/mechanic-fence-tests-metal.log`. The unchanged mesh-publication test
  subsequently passed in isolation (`/tmp/mechanic-fence-mesh-retry.log`).
  Separate real Metal `cargo test -p mechanic-gpu --offline -- --test-threads=1
  --nocapture`: **54 passed, 13 failed**, identically on candidate and pristine
  frozen baseline. Logs `/tmp/mechanic-fence-gpu-metal.log` and
  `/tmp/mechanic-fence-gpu-baseline.log`. Every failing diagnostic, including
  printed numerical values, matches between builds. Async readback publication,
  ring backpressure, impulse batches, and friction-driven turning pass.
  Required steering-under-gas, off-centre impulse and collision gates remain
  blocked by these pre-existing failures; no solver/test thresholds were edited.
- [x] Metal production dependency compile plus Vulkan/GLES HAL compile on macOS:
  `cargo check --manifest-path vendor/wgpu-hal/Cargo.toml --features vulkan,gles`.
  Log `/tmp/mechanic-fence-hal-check.log`. This is compile coverage only for
  Vulkan/GLES, and provides no DX12 or non-macOS runtime coverage.
- [ ] Run existing macOS/Linux/Windows matrix remotely. Both dependency regression
  modes are now included in CI. Only the aarch64-apple-darwin target is installed
  here; no remote CI, commits, or publication were requested or performed.
- [x] Complete paired release binaries and manifests in `/tmp/mechanic-fence-release`.
  Both application/asset trees, Cargo.toml and Cargo.lock have identical hashes;
  current runtime sources still match the candidate snapshot.
  Baseline binary SHA-256:
  `5be7bbb4fdff45bf62237f4f17e060a8118a29ad44f69eea642aa8a091d486f3`.
  Candidate binary SHA-256:
  `29fdcea53d387f729dea8d578219b58252892e7061d67294770be6bc35367b24`.
  Shared terrain shader SHA-256:
  `423ac980f25c94d70be0bb318bda5ab1b6f76773c92b5c565136439547334447`.
- [ ] Matched stationary/driving/F3-closed A/B/B/A captures and fresh CPU profile.
  No application-level gain or acceptance threshold is claimed yet.
- [ ] Real Metal window stress and manual seat/controller/cursor checks.

Acceptance remains **open** until the matched application and window gates pass.
Keep 4112×2524, 4× MSAA, baseline materials, steering torque, 32 angle iterations,
eight contact sweeps and three in-flight physics slots unchanged. If the original
lock wait disappears without throughput gain, identify the next limiting wait
before proposing additional scheduling or quality changes.

Additional final verification:

- `cargo check --workspace --all-targets --offline`: passed on macOS.
- `cargo test -p mechanic-world --offline`: 82 passed.
- Real Metal selector-window smoke: three cycles of 1000×700 / 1400×900
  resizing, minimize/restore, hide/unhide, Finder focus and recapture, then
  restoring the original 1280×748 outer window size. No validation errors,
  device loss or hangs; closing the window exited with status 0. Log
  `/tmp/mechanic-fence-window-final.log`; actions in
  `/tmp/mechanic-fence-window-actions.txt`. This is **selector-only** coverage,
  not loaded-world physics, seat/controller behavior or sustained occlusion.
  An initial launch lacked assets because the copied executable used its own
  asset root; that run is excluded. The guide now sets `BEVY_ASSET_ROOT` to the
  frozen snapshot explicitly, and the corrected launch had no asset errors.

The identical baseline/candidate Metal failures are:

```text
articulated_car_drop_settles_without_drift_or_ground_penetration
articulated_car_wall_impact_remains_bounded_and_decays
flat_disc_landing_on_another_disc_does_not_gain_energy
gpu_cylinder_bore_is_passable_and_annular_material_blocks_motion
gpu_pipe_bend_bore_is_passable_and_annular_material_blocks_motion
lower_modulus_allows_more_transient_penetration_without_failure
off_centre_external_impulse_changes_linear_and_angular_motion
plastic_rebounds_more_than_concrete
production_servos_hold_steering_under_first_gear_gas_drive
static_friction_holds_a_sub_threshold_load_while_kinetic_friction_slows_sliding
steering_servos_reach_angle_without_overshooting
struck_articulated_wheel_recovers_without_crossing_ground
tall_single_and_double_pendulums_dissipate_energy
```

For example, both builds report 6.062436° steering deflection under first-gear
load, 30.812975° steering overshoot, 0.39855438 m static-friction drift, and the
same `FacesDoNotTouch` fixture error. These failures prevent marking the original
all-correctness-gates requirement complete. They were not fixed or weakened as
part of this dependency synchronization change. The subsequent TEST4 results
below supersede the earlier absence of application captures; acceptance remains open.


## TEST4 automatic comparison — 2026-09-05

- [x] Run standing/offscreen-car stationary A/B/B/A, per the user's scene correction.
  Full results, raw JSONL, hashes and native profiles:
  [TEST4 report](performance-results/README.md#2026-09-05-test4-fence).
- [x] Confirm matched 4112×2524, 4× MSAA, baseline materials, F3/AutoNoVsync;
  idle streaming, consecutive ticks, zero flags, complete unique GPU samples.
  Original TEST4 and car2 hashes unchanged after running a resettable copy.
- [x] Submission p95 improvement: 25.48/25.85 ms baseline vs 0.353/0.344 ms
  candidate, approximately 98.6% lower in both pairs.
- [ ] Required completed-TPS improvement: **failed**, +0.56% and +1.49%,
  against a 10% threshold. FPS stayed within 0.56%; frame p95 within 2.59%.
  Readback p95 increased from 80–81 ms to 97 ms; backlog grows in every run.
- [x] Identify continuing render wait: candidate native profile includes Metal
  `nextDrawable` semaphore wait; acquisition remains expensive. Profiles are
  not a clean isolated before/after fence-stack proof. Do not infer a new fix
  from this alone.
- [ ] Driving, F3-closed normal play and full loaded-world stress remain unrun.
  Stop at the failed stationary throughput gate; no scheduling or quality changes.

The dependency stall regression is fixed, but this phase has **not** delivered
its required application-level throughput gain. Four automatic runs and normal
window shutdown passed; the overall correctness and performance acceptance
requirements remain unmet.


## Unattended background runner — 2026-09-05

- [x] In-app loading, warm-up/capture, Bevy screenshot and automatic exit.
- [x] Disposable uniquely named world copy, cleanup on success/failure/timeout.
- [x] No live input, cursor capture or macOS launch-focus stealing in auto mode.
- [x] Explicit background metadata; raw per-frame focus state.
- [x] Real TEST4 Metal run: all 1,733 CPU frames unfocused, zero physics flags,
  valid screenshot/capture, exit zero, original manifest unchanged, copy removed.
  [Verification and raw capture](performance-results/README.md#2026-09-05-background-runner).

This is stationary background diagnostic automation. Controlled foreground
benchmarks, scripted driving, headless rendering and full window-stress coverage
remain separate work; the runner does not change their acceptance status.
