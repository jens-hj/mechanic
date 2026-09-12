# Fixed-duration foreground car replay

Capture closing and publication draining are verified on Metal. Two identical-build
runs each submitted, read back, and published all 1,800 ticks, with no dropped ticks
or missing final states. Performance and repeatability still fail. These are
reference repetitions, not an optimization result or an A/B solver comparison.

## Implementation and identity

Schema 3 closes submissions at a frame boundary and records a separate drain
phase. Completion requires publication of the last actually submitted tick and
an empty readback ring. A ten-second drain deadline invalidates incomplete work.
The last submitted tick is tracked explicitly: a scheduler drop cannot become a
publication target. Frame/render distributions exclude drain samples; completed
physics throughput includes drain time.

Foreground `--replay-ticks N` retains overdue work at the existing 60 Hz clock,
including wall time spent waiting for terrain. It stops after exactly N submitted
ticks. Normal application scheduling is unchanged. Initial packed-terrain and
reachable GPU layout fingerprints, plus effective drive-row hashes per tick,
provide additional repeatability evidence. GPU solver and shader behavior were
not changed in this continuation.

The source-matched release was built in a fresh isolated target directory:

```sh
python3 scripts/build-performance-reference.py \
  --output .physics-reference/replay-final-20260910
```

`build-identity.json` records the source hashes, command, toolchain, and executable
SHA-256 `fe1a5a6f5d99d048792809c3afd2bff276d4d604fbb9f8b5f66a86a3ef4b0775`.
The build took 7 minutes 44 seconds. Complete source, assets, and executable are
retained outside Cargo's cleanable directory at that ignored reference path.
`reference-sources.tar.gz` retains the source/text subset here; media remain
identified by hash. `capture-tools.tar.gz` contains the exact schema-3 tools.
Historical schema-2 captures retain their own archived readers.

## Protocol

Run serially with distinct output paths ending in `01` and `02`:

```sh
python3 scripts/run-background-capture.py --foreground --drive --replay-ticks 1800 \
  --binary .physics-reference/replay-final-20260910/mechanic-app \
  --identity .physics-reference/replay-final-20260910/identity.json \
  --assets .physics-reference/replay-final-20260910/source/crates/mechanic-app \
  --world .physics-reference/2026-09-09/car \
  --output .physics-reference/replay-car-run-01
```

Both runs used Apple M1 Pro / Metal, after builds and hardware tests finished.
Every measured frame was focused at native 4112×2524, 4× MSAA, AutoNoVsync, F3.
The launcher used a disposable world in an isolated store and verified source
world and asset hashes before and after execution. Graphics warmed for 15
continuously ready seconds with physics held at the loaded state. The identical
W/A/D script then ran for 30 simulated seconds, including 180 settling ticks
before throttle. Other operating-system work was not disabled. Two repetitions
do not establish a confidence interval or the later A/B/B/A acceptance protocol.

## Results

| Metric | Run 01 | Run 02 |
| --- | ---: | ---: |
| Submitted / read back / published ticks | 1,800 / 1,800 / 1,800 | 1,800 / 1,800 / 1,800 |
| Total measured wall time, including drain | 77.730 s | 77.589 s |
| Drain time / drained readbacks | 0.315 s / 9 | 0.348 s / 9 |
| Completed physics TPS | 23.16 | 23.20 |
| GPU physics p95 | 45.515 ms | 44.915 ms |
| FPS | 13.70 | 13.47 |
| Frame p95 | 171.227 ms | 170.056 ms |
| Dropped ticks / pending publications | 0 / 0 | 0 / 0 |
| Recorded backlog, first → last measurement frame | 0 → 2,843 | 0 → 2,833 |
| Terrain-readiness hold events | 459 | 439 |
| Physics error flags observed | 0 | 0 |

Zero drops now preserve the requested workload; the growing backlog and low TPS
still fail acceptance. These timings cannot establish a speedup against the
earlier 60-wall-second captures, which executed different simulated durations.

Run 01 CPU encoding/finalization/submission/readback-setup p95 values are
0.104/2.376/0.811/0.007 ms. Submission-to-readback p95 is 507.986 ms;
callback-to-publication p95 is 86.143 ms. Terrain uploads total 194,004,608 bytes.
The summaries retain separate CPU, GPU, transfer-byte, and publication measures.
Their distributions overlap or describe different populations and must not be
added into a synthetic tick latency. Transfer execution time is not independently
isolated by these counters.

All combined render GPU samples remain `Partial: reversed`. Affected spans are
unavailable, not zero; valid opaque GPU p95 is 29.557 ms in run 01 and does not
isolate terrain rendering. The first run's screenshot was inspected and shows
the authored car on terrain; it cannot establish millimetre penetration bounds.
Zero error flags and partial execution counters do not prove physical acceptance.

## Repeatability failure

The initial state (`ad26c619b20ad19f`), camera, ordered packed terrain
(`e313908181fd1030`), and reachable GPU layout (`1464615737a723a5`) match.
All 1,800 scripted inputs and effective drive-row hashes match. The first 13
completed state hashes match; actual tick **14** (script tick **13**) diverges.
Both runs have 58 active contacts at that tick. This is during settling, before
throttle, terrain holds, or the first captured terrain replacement (about 18 s).
The earlier wall-time pair first diverged at tick 12; the first divergent tick
is itself not stable across pairs.

`repeat-comparison.json` rejects the runs solely for differing completed-state
hashes. `first-divergence.json` preserves the first 20 ticks and the first terrain
replacement/hold events. Matching initial terrain and drive identities narrows
the investigation but does not isolate an exact GPU operation. Later terrain
replacement streams differ after the trajectories have already diverged.

To reproduce the comparison, decompress each raw capture alongside its own
`run.json`, then run:

```sh
python3 scripts/compare-perf-captures.py --repeat \
  run-01/capture.jsonl run-02/capture.jsonl
```

Exit status 1 is the retained repeatability failure, not a missing capture.

## Verification and remaining work

Workspace Clippy, formatting, whitespace checks, eight capture tests, one terrain
identity test, and 21 Python tests pass. The focused hardware residency test passes:

```sh
cargo test -p mechanic-gpu --offline \
  adopted_terrain_residency_collides_without_uploading_chunks_again \
  -- --test-threads=1 --nocapture
```

Its log identifies Apple M1 Pro / Metal. Hardware work and foreground runs were
serialized. The workspace test command with one test thread reaches the app:
739 passed, one failed, six ignored. The failure is the previously observed
`ui::suspension::tests::leader_geometry_follows_projected_positions_without_remounting`.
The earlier intermittent Closure Lab failure did not recur in this run; it
remains recorded. Cargo stops at the app failure, so this is not a passing full
workspace or full GPU suite. The original GPU failures remain open.

Stage 1 still needs complete executed-kernel coverage and localization of the
old backend's repeatability failure. Preserve these failures as replacement
regressions. The next major implementation remains the complete CPU tick and
actual-car coupled contact experiment, including physical bounds. The algebra
microbenchmark is not that experiment. All car speedup, native rendering,
100,000-body, and physical acceptance gates remain open.
