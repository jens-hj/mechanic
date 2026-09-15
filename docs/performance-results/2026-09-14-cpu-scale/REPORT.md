# CPU scaling evidence — Apple M1 Pro

The 2 ms complete-tick target at 10× separated and the separate 120 FPS target are **not met**. Full connected-workload acceptance is not established. This report distinguishes completed measurements from incomplete diagnostics. Runtime remains single-threaded, with unchanged solver settings, contact tolerances, rendering settings, and authored collider decomposition. The latest follow-up reduces complete-tick p95 from roughly 44.5 ms to 30.1 ms in repeated 10× separated replays, with all paired state hashes identical.

## Reproduction and identities

The immutable builder fixture is in `crates/mechanic-bench/tests/fixtures/builder-world`. Each copy has 13 bodies, 18 generalized velocities and 1,744 colliders. Replicas are spaced eight metres apart over the same saved procedural terrain seed. Connected workloads rigidly link chassis, retaining suspension coordinates.

`baseline-source.json` describes the reconstructed baseline: original dense CPU runtime plus the pre-existing uncommitted freeze/hold work. The baseline executable was preserved before subsequent changes. `local-rows-identity.json` and `local-rows-app-identity.json` record source and executable SHA-256 hashes for the earlier component-local release builds. `matched-final/manifest.json` records exact commands, binary hashes and run durations. Earlier `matched/`, `matched-pre-local-rows/`, `candidate-source.json`, `final-candidate-identity.json`, `foreground/` and `foreground-final/` are intermediate evidence, not final acceptance runs.

## Foreground app — current 1× capture

The rebuilt native app completed 600 CPU ticks with no dropped ticks, skipped completion sequences, degraded ticks or GPU fallback. Measured completion rate was 59.77 TPS over the 10.038-second capture window. This is a short 1× app capture, not a sustained 10× scheduler gate.

| Metric | p95 |
|---|---:|
| Complete CPU tick including conversion/publication | 1.858 ms |
| Contact queries | 1.202 ms |
| Dynamics | 0.037 ms |
| Row preparation | 0.464 ms |
| Constraint solving | 0.074 ms |
| Continuous collision | 0.079 ms |
| Terrain update per frame | 0.284 ms |
| Foreground frame | 41.540 ms |
| Tracked rendering GPU span | 35.448 ms |

Rendering achieved 28.70 FPS on Apple M1 Pro / Metal at 4112×2524, MSAA 4, AutoNoVsync, with F3 displayed and focus verified on all 288 measured frames. The opaque GPU pass alone had 27.683 ms p95. Rendering independently misses the 8.33 ms frame budget; no renderer redesign or quality reduction was made. Raw capture, inspected screenshot, launch identity and protocol checks are in `clipping-followup/foreground/`, with completed build identity in `clipping-followup/app-identity.json`.

The earlier component-local 1× app capture in `foreground-local-rows/` was 1.880 ms CPU tick p95 and 41.384 ms frame p95. The current short 1× result is similar; the 32% gain below belongs to the sustained 10× separated headless workload.

## Connected assembly correctness

The original 10× connected runs in `matched-final/` degraded on every measured tick. Their apparent low timings are invalid performance results. The root spatial inertia factor rejected roundoff asymmetry slightly above its generic dense-matrix symmetry threshold. Its internal fixed-size Cholesky now uses the mathematically symmetric lower triangle and still rejects non-finite values and non-positive pivots, without a pivot shift. The public dense reference validation is unchanged.

A long rigid chassis with all suspensions active now passes the free-fall regression. The 30-tick connected cold diagnostic in `root-factor-diagnostic/` has zero degraded ticks. Full connected timing acceptance remains open.

## Additional clipping optimization

A native sampling profile identified repeated proximity-extrusion construction as the largest remaining steady query cost. The follow-up retains the last extrusion only when all source vertices, planes, edges, the triangle normal and margin match. Each finite triangle is still independently clipped. Clipping also reuses vertex-to-plane distances and skips copies for wholly interior polygons.

Two repeated 10× separated comparisons, each with 600 warm-up and 3,600 measured ticks, confirm **32.2–32.6% lower complete-tick p95**:

| Metric | Before this follow-up | After |
|---|---:|---:|
| Complete CPU tick p95 | 44.425–44.606 ms | 30.078–30.131 ms |
| Collision query p95 | 35.158–35.294 ms | 20.592–20.721 ms |
| Row preparation p95 | 7.718–7.720 ms | 7.728–7.762 ms |
| Compute throughput (not app scheduler TPS) | 28.42–28.51 ticks/s | 41.25–41.28 ticks/s |

All 8,400 paired state hashes match exactly; repeated runs also match within each executable. All four runs have zero degraded ticks and no terrain-region escape. Maximum penetration (1.884 mm), closure gap (18.527 mm) and closure angle (2.107e-8 rad) are identical. This workload still misses both the 2 ms tick budget and 60 ticks/s.

Both executables include the root-factor correction. Raw full-length JSONL and command/binary manifests are in `clipping-followup/matched/`; `clipping-followup/comparison.json` contains stage p95s and state comparisons. Source/binary identity, the isolated code patch, short diagnostics and separate sampling profiles are in `clipping-followup/`. Profiled runs are excluded from timing comparisons.

Separate cold diagnostics also match every state: 120 ticks at 1× (10.224 → 8.486 ms p95), and 30 ticks at connected 10× (476.311 → 467.879 ms p95). Neither degrades. These short impact samples are behavior checks, not sustained connected acceptance. Continuous collision dominates the connected cold case; the 32% improvement must not be generalized to that workload.

## Cold starts and holds

The 120-tick saved-terrain cold replay improved from 33.547 ms baseline p95 to 10.263 ms candidate p95. All 120 state hashes were identical. Both reported zero degraded ticks, maximum penetration 9.882 mm and maximum closure gap 40.737 mm.

The separate hold/release replay is **incomplete in both recorded binaries** (baseline and the candidate before the last component-local row change). Both completed tick 480 with identical state hashes for every tick. The second release at tick 481 produced no completion for over four minutes in baseline and over two minutes in candidate; both were terminated. Exact observation durations and partial raw JSONL are retained in `quality-final/*hold-status.json` and `*hold.jsonl`. This exposes a pre-existing release/recovery performance limit; it is not a passing hold gate.

## Verification

* Clipping follow-up: all 279 core tests (including 34 contact-geometry tests) and all 184 CPU physics tests passed. Logs are in `verification/clipping-followup-tests.log`.
* Workspace Clippy with `--all-targets -- -D warnings`: passed. Formatting and diff whitespace checks passed.
* Python capture/summary/comparison and benchmark evidence tests: 26 passed.
* Hardware workspace run used Apple M1 Pro, Metal. App: 765 passed, six ignored, one unchanged UI test failed (`leader_geometry_follows_projected_positions_without_remounting`). The subsequent workspace run skipped that test and reached GPU: 110 passed, 11 failed, one ignored. Core: 278 passed; world: 91 passed. Full logs are in `verification/`.
* GPU failures include car impacts, bores, material/friction response, steering/servo loads and pendulum dissipation. GPU solver/kernel code was not changed, but these failures were not separately proven on the reconstructed baseline; the workspace cannot be called green.
* Existing final CPU quality scenarios (`car-drop`, `car-drive`, `block-pile`, `fast-impacts`, `four-bar`) reported no degraded ticks. Their final raw reports and `reference-fixtures` are in `quality-local-rows/`. The block-pile diagnostic is 2.557 ms p95 versus the original recorded 14.983 ms.

## Remaining limits

Sparse runtime kinematics retains the reference arithmetic order and stores ancestor paths; it is not linear storage for arbitrary deep trees. Contact rows, articulated factor arenas, collision traversal, transformed shapes and terrain clipping buffers are reused. Closure/drive rows, returned contacts and some kinematics temporaries still allocate. `solver_scratch_bytes` and `solver_scratch_growth_bytes` report retained solver arenas only, not total allocator activity.

The finite headless terrain region is meshed from the saved seed at leaf resolution. Replays report whether body origins leave that region; they do not reproduce live app terrain residency. Neither headless throughput nor the short 1× capture proves 60 TPS or rendering acceptance for the 10× app workloads.
