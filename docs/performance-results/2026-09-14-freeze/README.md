# Hammer freeze and height movement

Measured with the release profile on Apple M1 Pro / Metal. The saved `builder` world has 14 published bodies (13 dynamic) and 18 generalized velocities. The launcher copies the save into its own disposable store, hashes the source and assets before and after, and removes only that copy. CPU captures use `MECHANIC_PHYSICS=cpu`; Metal still renders the application. These are background diagnostic captures, not controlled foreground rendering benchmarks.

The baseline adds timers and scripted input to the pre-existing uncommitted work. It confirms that exhaustive held-pair checks dominate the freeze hitch: internal endpoint maximum 782.58 ms, internal sweep maximum 1764.15 ms, planning maximum 3120.95 ms, and freeze update maximum 4906.87 ms. Hold application maximum is 1.37 ms. Terrain sweeps also reach 233.20 ms. Rendering is measured separately: tracked GPU span p95 37.95 ms and render CPU p95 5.00 ms.

The implementation caches held collider geometry for each published hold, clears pose samples between queries, and uses swept AABB sweep-and-prune before the existing SAT checks. Bounds include angular travel about the body origin, including offset geometry. Midpoint samples and scratch allocations are reused. A separating SAT axis ends a successful interval test immediately. A complete-interval proof also avoids the initial-overlap query; uncertain sweeps retain the original tangled-part rule. Collision suppression, same-body exclusions, initially tangled escape, and exact endpoint rejection remain in place.

Settled height moves are explicit common translations, with one interpolation step shared by all held bodies. These reuse the accepted internal arrangement; initial alignment, arrows during alignment, and publication changes still require internal validation. Local terrain bounds form a hierarchy: a clear enclosing sphere can discard a whole group, while uncertain leaves retain the existing terrain triangle BVH and precise per-collider checks. Traversal stops at the first obstruction. A downward search that cannot make progress beyond the existing rejection tolerance returns immediately instead of repeating all fourteen probes. Terrain is queried afresh, so no terrain-generation or origin-dependent world-space certificate survives a query.

The sequence freezes after two measured seconds, requests an early raise at 2.1–2.3 seconds, holds raise at 10–14 seconds, holds lower at 20–26 seconds, and releases at 40 seconds. Real elapsed time schedules input; the normal repeat handler, planner, animation, persistence and CPU/GPU hold functions execute it. Extremely slow baseline frames can miss short input windows. The current launcher additionally verifies accepted alignment, increasing and decreasing target heights, and final release from captured states.

`freeze_stage` timings include early returns and are opt-in. `active_total` excludes idle updates, and input processing has its own p95 and maximum. `active_movement_total` excludes the scripted release frame; entry, height input and release also have separate summaries. `freeze_input_update_interval_ms` measures the cadence at the freeze-update entry point: Bevy's render pipeline delays its published frame delta, so directly attributing the next reported frame delta to an input is incorrect. Overall frame delta and render CPU/GPU metrics remain separate in the summary. Stage timings nest; do not add planning to its constituent clearance stages.

## Final CPU result

| Metric | Baseline | Optimized |
| --- | ---: | ---: |
| Active freeze/height processing p95 | 4906.87 ms | 3.61 ms |
| Freeze entry maximum | 4906.87 ms | 29.46 ms |
| Height-input processing p95 | 4405.96 ms | 3.61 ms |
| Height-input processing maximum | 4405.96 ms | 15.53 ms |
| Input-update interval p95 | 4923.39 ms | 44.92 ms |
| Overall frame delta p95 | 47.37 ms | 45.35 ms |
| Overall frame delta maximum | 5100.51 ms | 67.00 ms |
| Render CPU p95 | 5.00 ms | 5.07 ms |
| Tracked GPU render span p95 | 37.95 ms | 38.09 ms |

**The 5 ms p95 processing target passes; the 16.7 ms maximum target does not.** Entry still takes 29.46 ms. Rendering also exceeds a 16.7 ms frame budget independently of freeze processing. These numbers are not a claim of 60 fps or a completed worst-case latency gate.

The final CPU capture completed the alignment, early arrow input, sustained raise, final terrain-limited lowering, and release checks. It had zero degraded CPU ticks, zero physics failure flags, and complete publication drainage. The initial-overlap fast path reduced internal sweep maximum from 41.00 ms in the preceding candidate to 13.35 ms. The downward-repeat shortcut reduced planning p95 from 15.30 ms to 3.07 ms.

## GPU limitation

Both the original instrumented baseline and the optimized executable stop the saved `builder` world at GPU physics tick 13 with failure flags `4` (`CONSTRAINT_NON_CONVERGENCE_FLAG`), before warm-up or scripted input. Logs and launcher records are retained as `baseline-gpu-*` and `saved-gpu-*`. This is a pre-existing saved-world GPU failure, not a passed GPU smoke test of the car.

A minimal persisted fixture is provided in `fixtures/gpu-smoke`: one ungrounded Dimension Link above the same procedural terrain seed, with the normal paired empty Garage document. It exercises the exact same executable, terrain checks, repeat handler, prescribed holds, height animation, target persistence, and release protocol without the failing articulated car solver. This fixture passed the complete scripted protocol with zero physics failure flags and complete publication drainage. Its active-movement processing was 1.58 ms p95 / 2.38 ms maximum. Both final CPU and GPU fixture runs use the same binary SHA-256. This is a behavior smoke check, not evidence that the saved car works on GPU.

## Reproduction

```sh
cargo build --release -p mechanic-app
python3 scripts/run-background-capture.py --binary target/release/mechanic-app \
  --world "$HOME/Library/Application Support/mechanic/worlds/builder" \
  --output /tmp/mechanic-freeze-cpu --physics cpu --freeze
# GPU behavior smoke fixture (the saved car fails before input on GPU):
python3 scripts/run-background-capture.py --binary target/release/mechanic-app \
  --world docs/performance-results/2026-09-14-freeze/fixtures/gpu-smoke \
  --output /tmp/mechanic-freeze-gpu-smoke --physics gpu --freeze
```

The final build uses `CARGO_TARGET_DIR=/tmp/mechanic-freeze-target` because an external cleanup removed this repository's `target` directory while the release compiler, doctests and Clippy were running. The executables were preserved separately under `/tmp` while measuring. All task-owned temporary build trees, executables and full capture directories were removed afterward to reclaim disk space; compressed JSONL, source/binary hashes, logs, summaries and the smoke fixture remain here.

## Verification notes

The focused freeze run passed 37 tests, including deterministic comparisons against exhaustive held-pair enumeration, crossing and tangled parts, rotational offsets, collision suppression, common translation, arrows during alignment, cache replacement after publication and weld restore, origin shifts, terrain walls and ceilings, and the final downward clearance step. The CPU hold test filter passed seven tests.

The broad app run passed 773 tests, ignored six, and failed the previously recorded `ui::suspension::tests::leader_geometry_follows_projected_positions_without_remounting`. The remaining workspace unit tests passed core (281), CPU physics (190), and world (91). GPU passed 110, ignored one, and failed the same 11 tests listed in `../2026-09-14-cpu-scale/verification/workspace-metal.log`; no newly failing GPU test name was observed. The initial doctest pass was interrupted by deleted build dependencies; the isolated release-profile retry completed successfully. Final workspace Clippy and formatting checks pass. Verification logs are retained in `verification/`.

The first candidate capture is retained as `interrupted.*` solely as diagnostic evidence. It overlapped the GPU suite and lost its Metal device (`Cannot allocate sample buffer`); it is invalid and excluded from acceptance comparisons. Its partial record showed internal endpoint time down to 2.5 ms but expensive terrain sweeps, prompting the hierarchical terrain bounds.

The valid intermediate terrain-bound CPU run (`terrain-cpu.*`) reduced maximum freeze processing to 66.62 ms but still measured 15.14 ms active-movement p95. Its trace identified initial-overlap queries during alignment and repeated final-floor bisection as remaining costs; those motivated the final interval-proof and downward-search fast paths.

The launcher regression suite passed four tests, and the capture-summary suite passed seventeen. Source save and asset hashes are identical between the baseline and final CPU runs.
