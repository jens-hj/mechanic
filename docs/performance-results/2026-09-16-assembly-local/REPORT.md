# Assembly-local collision queries

## Implementation

Contact validity is owned by compiled mechanical components, including loop
closures and suspension connections; unconnected bodies get their own component.
Terrain and internal contacts have separate budgets. Cross-component ownership
uses conservative participant budgets and combined relative travel, with linear
storage instead of a quadratic table. A refresh retains untouched contact objects
and impulses and orders the merged contacts deterministically, keeping recovery
rows after ordinary contacts.

Every tick still begins with a full query. Retained allocations contain no valid
proof at the next tick boundary. Assembly empty-space certificates bind geometry
identity, topology, terrain publication, and floating origin. Reconstructed
conservative body bounds must remain contained, and the shared broad phase checks
other assemblies again. Impacts discard the certificates. Failed validation falls
back to querying.

Compact moving-collider metadata avoids rescanning fixed geometry. Motion travel
is measured once per moving collider and aggregated per assembly for both
contact invalidation and sweep activation. Body-local aggregate bounds and radii
are built with the collision geometry.
Actual generalized substep motion, including complete rotations, moving ancestors,
and suspension extension, supplies conservative body swept envelopes. Terrain and
body rejection happen before detailed collider paths and trees. Surviving terrain
candidates are shared across a body's colliders. The original continuous narrow
phase and impact response remain in use. Both fast activation and insufficient
contact coverage trigger sweeps; low speed alone cannot skip them.

The existing numerical contact-query fallback now tries half the speculative
search reach before zero reach. Nearly parallel car faces otherwise commonly
discarded all positive query coverage, forcing redundant continuous work. This
does not change the physical contact or impact tolerances. Initial finite support
features can be reused only at exactly their queried body poses.

New capture counters report refreshed/reused contact groups, detailed sweep
preparations, failed clearance validations, and exact-pose support reuse. Solver,
query, and continuous-query timings remain separate diagnostics. The global
solver, timestep, collision geometry, world formats, GPU physics, and frame
scheduling are unchanged.

## Reproduction and baseline

The baseline is the incoming working tree, including the previous agent's
uncommitted motion work. Its release CPU executable, source archive, and original
diff are preserved under `.physics-reference/assembly-local-baseline/`.
The CPU executable SHA-256 is
`e6546e003b15aa9c9db545c4735d09f8ad8e284955c14ef6bc0d7d9217d487ce`.

`run.py` runs three alternating release pairs, 120 warmup and 600 measured ticks
per speed, with the same saved snapshots and a pipe placed 20 m from the background.
`manifest.json` records executable and input hashes. Speeds are 0, 1, 10, and
40 rad/s; gravity remains enabled, so a nominal zero-speed pipe can subsequently
move. `summary.json` contains paired median and p95 results, degradation counts,
pipe-speed differences, and state hashes. Generated threshold cases use
`scripts/run-cpu-motion-benchmark.py`. Profiling, builds, tests, and app captures
are kept outside accepted timing runs.

Native captures use `run_native.py`, the existing foreground/replay capture
protocol, and disposable copies of the saved worlds restored to the supplied
scene snapshots. The camera is fixed. The same requested hammer impulse is
delivered through the app's normal stability-limited hammer calculation; each
capture logs the actual per-tick impulse. Raw captures remain under
`.physics-reference/assembly-local-native-*`; compact summaries and screenshots
are kept here. The live world store is never opened by these runs.

## Verification

Focused tests compare filtered and exhaustive contact/sweep outcomes and exercise
full-turn impacts, off-centre rotation, moving ancestors, suspension travel,
approaching and simultaneous assemblies, terrain/topology/origin invalidation,
and exact-pose support reuse. The saved wishbone regression checks retained
support, free pipe rotation, and toleranced car trajectories. Operation counts
are reported in benchmarks rather than asserted in tests.



## Release timing results

Median of three paired p50 CPU tick times, in milliseconds:

| Scene | Speed (rad/s) | Baseline | Candidate | Change |
|---|---:|---:|---:|---:|
| alone | 0 | 0.0386 | 0.0328 | -15.2% |
| alone | 1 | 0.0385 | 0.0326 | -15.4% |
| alone | 10 | 0.0385 | 0.0327 | -15.2% |
| alone | 40 | 0.2698 | 0.0425 | -84.2% |
| wishbone | 0 | 1.6508 | 1.5144 | -8.3% |
| wishbone | 1 | 1.6485 | 1.5105 | -8.4% |
| wishbone | 10 | 1.6490 | 1.5213 | -7.7% |
| wishbone | 40 | 2.5608 | 1.5164 | -40.8% |
| builder-with | 0 | 1.9105 | 1.7061 | -10.7% |
| builder-with | 1 | 1.9098 | 1.7058 | -10.7% |
| builder-with | 10 | 1.9164 | 1.7132 | -10.6% |
| builder-with | 40 | 3.0804 | 1.7193 | -44.2% |
| builder-without | 0 | 0.1333 | 0.0841 | -36.9% |
| builder-without | 1 | 0.1333 | 0.0825 | -38.1% |
| builder-without | 10 | 0.1335 | 0.0872 | -34.7% |
| builder-without | 40 | 0.8685 | 0.1071 | -87.7% |

The additional fast-motion cost of the wishbone is
`(with-car fast − with-car slow) − (without-car fast − without-car slow)`.
Slow and fast are the 0 and 40 rad/s input cases.

- wishbone: 0.6788 ms before, -0.0077 ms after (101.1% raw reduction; values beyond 100% mean no measurable remaining extra cost).
- builder-with: 0.4348 ms before, -0.0098 ms after (102.3% raw reduction; values beyond 100% mean no measurable remaining extra cost).

All three paired ratios are retained in `summary.json`; no samples are removed.
Final pipe-speed differences are 0.0 rad/s. Candidate degraded ticks: 0.
Whole-scene hashes can differ because refreshed contacts are reused differently;
observable support and toleranced trajectories are checked by the regressions.

Slow saved-scene cases with a repeatable (>5% in at least two pairs) regression: 0.

## Saved-scene sweep threshold

The same four snapshots were also run at 0.99 and 1.01 times the pipe's actual
activation speed. Baseline physics is unchanged; `cpu-physics-speeds` adds only
the same benchmark environment-variable speed selection used by the candidate.
The original baseline executable remains preserved. The manifest records both
executables and exact requested speeds. Gravity remains enabled.

| Scene | Input speed | Baseline p50 | Candidate p50 | Paired ratios |
|---|---:|---:|---:|---|
| alone | 27.4569 | 0.0586 ms | 0.0370 ms | 0.635, 0.627, 0.635 |
| alone | 28.0116 | 0.0584 ms | 0.0365 ms | 0.626, 0.652, 0.617 |
| wishbone | 27.4569 | 1.6490 ms | 1.5210 ms | 0.914, 0.922, 0.924 |
| wishbone | 28.0116 | 1.6926 ms | 1.5206 ms | 0.887, 0.900, 0.901 |
| builder-with | 27.4569 | 1.9113 ms | 1.7167 ms | 0.897, 0.900, 0.897 |
| builder-with | 28.0116 | 1.9652 ms | 1.7173 ms | 0.871, 0.879, 0.874 |
| builder-without | 27.4569 | 0.1931 ms | 0.1004 ms | 0.520, 0.521, 0.519 |
| builder-without | 28.0116 | 0.1964 ms | 0.1000 ms | 0.509, 0.512, 0.506 |

## Additional synthetic microbenchmarks

The additional 23 generated cases are in `threshold-final/`. 13 show
repeatable p50 regressions above 5%; these are a remaining limitation, separate
from the four saved-scene comparisons. Near activation, the baseline skips some
continuous work on speed alone. The candidate performs the missing conservative
coverage checks; tiny scenes cannot amortize their bookkeeping and coarse-bound
cost over large collider sets.

| Fixture | Fixed unrelated bodies | Speed | Baseline p50 | Candidate p50 |
|---|---:|---:|---:|---:|
| rotor | 0 | 8.697 | 7.04 µs | 8.42 µs |
| rotor | 0 | 8.872 | 8.38 µs | 10.71 µs |
| rotor | 0 | 500.000 | 8.75 µs | 9.83 µs |
| rotor | 32 | 8.697 | 24.50 µs | 34.79 µs |
| rotor | 32 | 8.872 | 39.75 µs | 52.12 µs |
| rotor | 32 | 500.000 | 40.38 µs | 67.00 µs |
| translation | 0 | 11.880 | 5.71 µs | 6.67 µs |
| translation | 0 | 12.120 | 6.58 µs | 8.21 µs |
| translation | 0 | 500.000 | 6.50 µs | 9.58 µs |
| translation | 32 | 11.880 | 23.96 µs | 34.96 µs |
| translation | 32 | 12.120 | 39.29 µs | 52.29 µs |
| translation | 32 | 500.000 | 39.42 µs | 68.62 µs |
| vehicle | 0 | 1.000 | 258.92 µs | 274.54 µs |

Generated candidate degraded ticks: 0. Exact final state hashes match in 21/23 cases.

## Remaining CPU cost

Average diagnostic time per candidate tick at 40 rad/s (median across three
runs). These means have separate scopes and do not sum to wall-clock p50.

| Scene | Contact queries | Continuous motion/query | Dynamics | Constraints | Detailed paths/tick |
|---|---:|---:|---:|---:|---:|
| alone | 0.0459 ms | 0.0106 ms | 0.0034 ms | 0.0002 ms | 0.0 |
| wishbone | 1.0846 ms | 0.0696 ms | 0.0223 ms | 0.0410 ms | 0.0 |
| builder-with | 1.2455 ms | 0.0814 ms | 0.0252 ms | 0.0414 ms | 0.0 |
| builder-without | 0.1317 ms | 0.0216 ms | 0.0058 ms | 0.0002 ms | 0.0 |

The car still requires full initial contact queries and global solver work each
tick. Unrelated repeated preparation is reduced; those underlying costs remain.

## Verification commands

- `cargo test --release -p mechanic-physics --lib`: 202 passed.
- `cargo clippy --workspace --all-targets -- -D warnings`: passed.
- `cargo fmt --all -- --check` and `git diff --check`: passed.
- `python3 scripts/test-background-capture.py`: 4 passed.
- Release `cpu-physics --scenario fast-impacts`, `car-drive`, and `car-drop`: see JSONL results; fast-impact cases have no tunnelling.

## Native fixed-camera verification

Final binaries were captured on Apple M1 Pro / Metal, at 4112 × 2524 pixels,
AutoNoVsync, MSAA 4, and F3 enabled. Every measured frame was focused; all
foreground protocols passed. Each capture replays and publishes 600 CPU ticks.
Initial camera matrices, state hashes, and terrain geometry fingerprints
match between each baseline/candidate pair. No capture had degraded
ticks, dropped ticks, terrain-busy frames, or an undrained publication backlog.

GPU terrain-buffer layout fingerprints differ in the final fast-builder and
slow-pipe pairs, despite identical terrain geometry. These fingerprints include
allocation offsets and packed BVH layout, so final native timing is not a strict
matched-layout comparison. The earlier fast-builder pair had matching layouts
and independently measured 9.08 → 27.42 FPS and 26.71 → 59.78 TPS. The final
candidate preserves every published state from that earlier candidate capture.
The three paired headless saved-scene runs provide the controlled CPU comparison.

The fast recipe delivers 128 body-local `[0, 3, 0]` hammer-equivalent impulses at
local point `[-0.25, 0, -0.375]`, followed by coasting to tick 600. Every pulse
passes through the existing hammer delivery calculation. The actual impulses
match exactly between binaries and arrive on ticks 1 through 128, one tick per
pulse. Both binaries activate continuous sweeps. A single `[0, 200, 0]` strike
uses the normal 12-tick delivery and supplies the slow control; it does not
activate the pipe sweep workload. The app capture hook changes no normal input
or scheduling behavior.

| Scene / input | Baseline FPS → candidate | Baseline TPS → candidate | CPU p50 ms | CPU p95 ms |
|---|---:|---:|---:|---:|
| Pipe / fast | 34.67 → 35.00 | 59.89 → 59.81 | 3.318 → 0.227 | 3.500 → 0.257 |
| Builder / fast | 9.03 → 27.49 | 26.59 → 59.93 | 33.934 → 3.797 | 35.008 → 5.739 |
| Pipe / slow | 34.96 → 34.91 | 59.96 → 59.82 | 0.085 → 0.084 | 0.099 → 0.102 |

Three fresh slow-builder pairs test the 5% CPU regression limit:

| Pair | Baseline CPU p50 | Candidate CPU p50 | Change | Baseline FPS → candidate | Baseline TPS → candidate |
|---|---:|---:|---:|---:|---:|
| 1 | 3.394 ms | 3.539 ms | +4.26% | 27.25 → 27.46 | 59.85 → 59.85 |
| 2 | 3.412 ms | 3.560 ms | +4.33% | 27.45 → 27.43 | 59.82 → 59.82 |
| 3 | 3.527 ms | 3.585 ms | +1.64% | 27.57 → 27.09 | 59.86 → 59.91 |

All three final pairs remain below 5%; median paired change is +4.26%.
An earlier implementation exceeded the limit in two of three native controls.
The final change stores compact moving-collider metadata and avoids repeating
fixed-collider motion arithmetic. All 16 saved-scene final states and all 600
slow-builder publications remain identical to that earlier implementation.
Earlier captures and binaries are retained in `.physics-reference/assembly-local-before-thin/`
and the original slow-control directories; the tables above use the final build.

In the final fast builder capture, continuous-query time averages 29.552 → 0.332 ms/tick.
Contact queries still average 3.192 ms, dynamics 0.032 ms, and constraints 0.055 ms.
The candidate's median tracked GPU span is 34.94 ms/frame.
The measured fast-motion frame/tick collapse is removed in this fixed view.
Rendering still limits the builder to roughly 27 FPS at this native resolution;
the result does not establish that every visible lag source is fixed. The broad
synthetic performance regressions above also remain unresolved.

The copied builder world uses the garage matching its restored instance,
preventing duplicate dimension-link IDs. This preparation affects only the
disposable copy. All 454 files in the two live world stores remain unchanged
(see `live-world-check.json`). Full captures and fixed-camera images are retained
locally; the two fast-builder screenshots accompany this report. Profiling is
in `profile-final.txt` from the preceding implementation, before the compact
motion-data optimization; it is excluded from timing comparisons.
