# Callback and terrain attribution

Terrain is the dominant measured opaque workload in this stationary TEST4 scene.
Completion publication also has a frame-sized tail, but its median is small.
These are diagnostic findings, not an accepted FPS/TPS improvement.

## Collection and classification

Apple M1 Pro / Metal; release build; 4112×2524 target and viewport; 4× MSAA;
baseline materials; F3 open; AutoNoVsync; 20 bodies / 2,083 colliders. Saved
stationary player view, car behind player, held matter manipulator visible.
Streaming backlog stayed zero. Both 60-second captures followed idle streaming
and 15 seconds of warm-up. No compiler or CPU profiler ran during recording.
No keyboard/mouse events or OS focus-switch commands were sent.

The normal run (18:05:24–18:06:24 UTC, 2026-09-05) reported **all 1,838 frames
focused**. The collection wrapper stopped on that unexpected classification and
removed its source copy. The second run proceeded only after checking every file
of a fresh frozen copy against the first copy's SHA-256 manifest. The partitioned
run (18:08:33–18:09:33 UTC) reported **all 1,663 frames unfocused**; a read-only
OS check during that recording also confirmed another application was frontmost.
The first run's focus cause was not established. **Do not treat this pair as a
matched background or controlled foreground performance comparison.**

Both apps saved in-app PNGs and exited. All disposable copies were removed;
the original world's complete file/hash map still matched after both runs.
Raw JSONL is retained compressed under `normal/` and `partitioned/`, with logs
and summaries. Compressed source/binary and world fingerprints are retained here.
The two one-shot collection scripts preserve the stopped wrapper and verified
continuation. Screenshots remain under `/tmp/mechanic-attribution-pair-01/` in
their corresponding run directories. Visual inspection confirmed the same scene,
body count, held tool, and settings.

## Completion observations

| Measurement | Normal p50 / p95 ms | Partitioned p50 / p95 ms |
| --- | ---: | ---: |
| Queue submission | 0.163 / 0.306 | 0.167 / 0.314 |
| Measured physics GPU stages | 5.530 / 6.446 | 5.148 / 8.060 |
| Queue return to final mapping callback | 65.945 / 94.508 | 66.895 / 101.525 |
| Final callback to tick publication | 1.134 / 33.287 | 1.231 / 38.184 |
| Queue return to readback consumption | 89.474 / 96.980 | 72.115 / 108.019 |
| Successful and empty poll duration | 0.0023 / 0.0120 | 0.0025 / 0.0128 |

Normal: 2,055 consecutive completed ticks, zero failure flags. The final callback
ran before the consuming poll call for 2,014 ticks, and during it for 41 ticks.
Callback-to-publication mean was 8.810 ms; 538/2,055 (26.2%) exceeded 16.67 ms,
and 102 exceeded 33.33 ms. There were 2,055 successful polls, 1,838 empty polls,
and no polling errors. This demonstrates a meaningful delayed-publication tail,
not expensive execution inside the poll function itself.

Partitioned: 2,148 consecutive completions, zero failure flags; 2,100 final
callbacks preceded the consuming poll and 48 ran during it. Callback servicing
depends on wgpu maintenance and other queue activity. These timestamps **cannot
separate GPU queue residence from time before callback servicing**; a callback
running during a poll may also have been invoked by another thread. Independent
medians must not be subtracted or added to construct a nonexistent timeline.

For context only: normal 30.64 FPS / 34.25 completed TPS, backlog 1,406→2,949;
partitioned 27.73 FPS / 35.80 TPS, backlog 1,410→2,860. Different focus state and
extra render-pass boundaries preclude an optimization claim from these numbers.

## Terrain observations

Normal opaque p50/p95 was 19.413/21.436 ms, from 1,838 Complete samples.
The partitioned run produced 1,664 Complete samples, each with one terrain run:

| Actual diagnostic render group | p50 ms | p95 ms |
| --- | ---: | ---: |
| Terrain opaque | 21.271 | 22.038 |
| Remaining opaque | 0.894 | 1.743 |
| Prepass/shadows | 0.864 | 1.009 |
| X-ray | 0.804 | 0.901 |
| Other tracked passes | 1.430 | 2.214 |
| Tracked span (partial coverage) | 26.475 | 28.249 |

These are actual pass-boundary timestamps. The M1 Pro reports support for
stage-boundary sampling but not draw-boundary or dispatch-boundary sampling;
therefore inserting timestamps between terrain and other draws inside the
original pass would not work. The opt-in partition keeps prepared draw order,
batching, shaders, geometry, viewport, depth/color contents and MSAA. Extra
attachment store/load and resolve work changes the workload, so terrain's
partition time is not its exact incremental cost in the original pass. Intervals
may overlap; do not sum groups or turn their medians into utilization percentages.

The large terrain/remaining-opaque difference supports prioritizing a
quality-preserving terrain rendering optimization. It does not yet isolate
texture sampling, fragment arithmetic, bandwidth, or geometry within terrain.
Use the existing real-material pixel benchmark for a specific candidate, then
verify integrated gains with **partitioning disabled**, a frozen scene and
matched focus/presentation conditions. Keep publication-delay tails visible in
that comparison; do not expand the queue or change scheduling based on this
diagnostic alone.

## Implementation and checks

- Physics callback timestamps are opt-in through the capture directory. No extra
  polling, waits, slots or dispatches; no GPU ABI or saved-data changes. The
  latency origin is immediately after queue submission returns, before map setup.
- Terrain partitioning additionally requires `MECHANIC_PERF_TERRAIN_PASSES=1`.
  Pinned Bevy core-pipeline provenance and the diagnostic overhead are documented
  in `vendor/bevy_core_pipeline/MECHANIC-PATCH.md`. Normal rendering stays unsplit.
- Native callback servicing/delayed-consumption regression: passed.
- Native consecutive, tick-matched, three-slot ring regression: passed, including
  absent callback timing fields with instrumentation disabled.
- Native 4096×2524, 4× MSAA pixel regression, terrain plus another opaque material:
  **0 differing bytes out of 41,353,216**, valid positive timing for each group.
- Seven render timestamp regressions passed with native Metal access, including
  actual raster timing, bounded queries and incomplete shadow handling.
- Three recorder regressions, six Python summary regressions, the disposable-world
  cleanup regression, and the dependency-local contiguous-order regression passed.
- Release build; app/GPU Clippy all targets with warnings denied; workspace and
  modified-vendor formatting; `git diff --check`: passed.

Native command logs and dependency-resolution output are retained here. Linux/
Windows hardware and the full workspace GPU numerical suite were not rerun;
earlier correctness failures and original foreground acceptance gates remain open.
