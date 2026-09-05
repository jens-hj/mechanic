# TEST4 fence-backport comparison — 2026-09-05

**Stationary acceptance failed:** the backport reduced submission p95 by 98.6%,
but completed physics TPS improved only 0.6% and 1.5%, below the required 10%.
FPS and frame p95 stayed within the 5% regression limit. No general application
performance improvement or sustained 60 FPS/TPS is claimed.

## Scene and hardware

Apple M1 Pro, Metal, 16 GPU cores; macOS 27.0 (26A5421a). Exact wgpu adapter
metadata is in `summary.json`. Frozen release executables and source manifests
are identified by the retained SHA-256 files. Baseline uses pristine wgpu-core
and wgpu-hal 29.0.4; candidate uses the backport. App sources and assets match.

User-requested scene: TEST4, existing car2 behind the player and out of view,
with existing placed blocks and physics running. No seat entry or vehicle
repositioning. Saved player translation: (175.32777879163731,
1.566084736691276, -788.3843398614591); stored rotation (0,0,0,1).
Matter Manipulator / Block / Steel held, no placement preview visible.
The B1 post-capture screenshot visually matched the baseline scene.
The overlay showed 17 bodies / 2069 colliders and three in-flight slots.

The original world was copied to `TEST4 PERF`, with only its manifest name
changed. That disposable copy was reset from the original before each launch.
All 179 original world files and `car2.mech` were fingerprinted and verified
unchanged after the sequence. See `scene-fingerprints.json`.

Every captured frame used 4112×2524 window/target/viewport pixels, 4× MSAA,
baseline materials, F3 open, and AutoNoVsync. Terrain backlog stayed zero,
physics flags stayed zero, submitted/readback ticks were consecutive, and all
render GPU samples reported Complete. No duplicate asynchronous GPU samples.
Each capture had the recorder's continuous 15-second ready warm-up and
60-second measurement. One app instance ran at a time. Screenshots and five-second
native CPU samples were collected outside the measurement intervals.

## Raw-sample results

| Run | FPS | Completed TPS | Submit p95 ms | Frame p95 ms | Readback p95 ms | Backlog start → end |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A1 | 30.54 | 32.58 | 25.478 | 43.029 | 80.008 | 1012 → 2656 |
| B1 | 30.37 | 32.77 | 0.352 | 41.914 | 96.606 | 1036 → 2669 |
| B2 | 30.13 | 32.93 | 0.344 | 43.520 | 97.371 | 1053 → 2675 |
| A2 | 30.21 | 32.45 | 25.845 | 42.942 | 80.986 | 1073 → 2724 |

A1/B1: submission p95 −98.62%, completed TPS +0.56%, FPS −0.56%, frame p95 −2.59%.
A2/B2: submission p95 −98.67%, completed TPS +1.49%, FPS −0.25%, frame p95 +1.35%.
Baseline and candidate repeats are consistent enough that the missing 10%
throughput gain is not explained by run-to-run drift. Backlog grows substantially
in every run; this scene remains below the fixed 60 TPS target.

## Remaining wait and decision

Candidate B1's native profile contains the rendering path
`prepare_windows → get_current_texture → Surface::acquire_texture →
CAMetalLayer nextDrawable → semaphore_timedwait_trap`. The raw acquisition timer
also remains large. Physics submission is now short, but completion/readback
p95 grows to roughly 97 ms from 80–81 ms. This is consistent with continued
render/drawable pacing and GPU completion pressure; it does not establish a
single exclusive throughput bottleneck or justify a scheduling change.

The short profiles vary in which waits they sample. They do **not** by themselves
provide a clean baseline-fence versus candidate-fence stack comparison. The
specific coupling is covered by the dependency regression that fails on pristine
29.0.4 and passes on the backport; the raw submission measurements agree with
that fix. Do not overstate the native profile evidence.

Stop at the failed stationary throughput gate. Driving, F3-closed normal-play,
and loaded-world window/seat/controller stress were not executed in this batch.
All four windows closed with exit status zero after profiling. This proves
ordinary loaded-world shutdown only, not the full stress matrix. The existing
13 identical baseline/candidate Metal test failures and unavailable remote
platform coverage remain open. No solver, timestep, queue depth, steering,
rendering-quality, or scheduling changes were made during measurement.

## Artifacts and reproduction

`summary.json` contains the full raw percentile summaries, acquisition/render
CPU timings, GPU sample health, backlog and exact adapter/settings. The
`capture-*.jsonl.gz` files retain all events; decompress and use
`python3 scripts/summarize-perf-capture.py FILES` from the repository root.
`mechanic-fence-*-profile.txt.gz` are native `sample PID 5 1 -file PATH` outputs.
`A1.log` through `A2.log` and `automation.log` record lifecycle and capture paths.

`automation.py` archives the exact session orchestration, including an already
running first baseline. It is not a standalone general launcher: it expects the
frozen binaries, the compiled native click helper (`native-click.swift`) and this session's disposable
`test4-perf` directory. The full launch/capture procedure and decompression
command are in [the profiling guide](../../performance-profiling.md#test4-automated-stationary-setup-2026-09-05).
