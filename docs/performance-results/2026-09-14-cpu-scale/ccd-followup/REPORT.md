# Continuous collision reuse and paired hierarchy traversal

The 10× connected cold CCD spike falls from approximately **394 ms to 90 ms**. Complete cold-tick p99 and worst-tick cost roughly halve. Full-length 10× connected p95 improves **1.1–3.7%** across three comparisons, with no separated-workload regression in the three measured pairs. **Cold p95 regresses**, as detailed below. The original **2 ms p95 and 60 TPS targets remain unmet at 10×**.

All **64,440 paired ticks** across **27 comparisons** have identical state hashes, contact/candidate counts, SAT evaluation counts and physical diagnostics. Repeated runs also match exactly. No accepted run stalled, degraded, or escaped the meshed terrain region. `evidence-audit.json` records the checks; `verify-evidence.py` reruns them.

## Implementation

Continuous sweeps share the compiled geometry's exact-pose cache with contact queries. Fraction-zero terrain and collider checks use `MachineMotion::initial_poses()`. Initial spatial velocities are computed lazily once per sweep, independently of the persistent pose cache. Nonzero fractions retain pose reconstruction and advancement arithmetic. Two geometry buffers retain allocations; translation-only terrain impacts preserve the original two-stage transformation. Initial terrain support clipping uses retained scratch. Locks are acquired in pose-cache, sweep-scratch, pair-scratch order.

Collider discovery traverses both existing trees, rejects non-overlapping node pairs, and splits the non-leaf node with more descendant leaves; ties split the first tree. Body suppression and static/static rejection precede traversal. Canonical collider pairs keep their original sorted order. A private scoped guard exposes the retained candidate slice. Compact nodes store child indices plus one value: a row at a leaf or a descendant count at an internal node.

`TerrainSweepQuery` adds full-shape transformation, shape-cache-hit and hierarchy node-pair counters. `SoftStepDiagnostics`, benchmark tick JSONL and app CPU capture records expose these and pose, velocity, SAT and candidate counters under `continuous_*` names. Proximity/recovery counts remain separate. Counters measure sweep-query work; path preparation occurs before the query. No terrain contact results or approximate pose fractions are cached.

Solver settings, tolerances, earliest-hit ordering, inconclusive-query handling and single-threaded physics execution are unchanged. Freeze/hold and the root-factor correction are included in the baseline and preserved. Motion-bound redesign, constraint-response caching, sleeping, multithreading and freeze/release stall repair remain outside this change.

## Full-length release comparisons

Every pair uses **600 warm-up ticks followed by 3,600 measured ticks**. Values below are complete-tick p95 ranges across three comparisons, in milliseconds.

| Workload | Baseline p95 | Candidate p95 |
|---|---:|---:|
| 1× connected | 2.442–2.446 | 2.416–2.423 |
| 2× connected | 5.035–5.055 | 4.996–5.013 |
| 5× connected | 19.173–19.257 | 19.143–19.330 |
| 10× connected | 54.982–55.605 | 53.556–54.656 |
| 10× separated | 29.973–31.918 | 29.835–31.130 |

The 1× and 2× gains are small; 5× is effectively flat. At 10× connected, mean tick cost changes from 38.821–39.010 ms to 37.987–38.669 ms, yielding **25.86–26.32 compute ticks/s** for the candidate. This is headless compute throughput, not app scheduler TPS, and remains below 60. The 10× separated candidate provides 40.23–41.66 compute ticks/s and also misses both targets.

Raw full connected comparisons are in `connected/`. The three separated comparisons comprise the initial `packed-separated-check/` pair and the two pairs in `separated/`; all use the same archived binaries and tick protocol. Candidate separated p95 changes are −0.46%, −2.50% and −0.26%, respectively. Baseline/candidate execution order is alternated by the runner. No builds, tests, profiling or app capture overlap these accepted timing runs.

The connected fixture has 13, 25, 61 and 121 bodies at 1×, 2×, 5× and 10×; rigidly joining chassis preserves suspension coordinates. The 10× connected case has 126 generalized velocities and 17,440 colliders. The 10× separated case has 130 bodies and the same collider count. Terrain comes from the saved generation-20 fixture and its procedural seed, not a diagnostic floor.

## Separate cold-impact comparisons

`cold/` contains three fresh 120-tick comparisons at every scale, with no warm-up. Cold complete-tick p95 is **higher at every scale**:

| Workload | Baseline cold p95 | Candidate cold p95 |
|---|---:|---:|
| 1× connected | 8.500–8.537 | 9.036–9.375 |
| 2× connected | 17.219–17.453 | 17.996–18.787 |
| 5× connected | 54.218–54.395 | 55.760–56.776 |
| 10× connected | 133.458–136.786 | 137.552–138.931 |

For 10× connected, the more extreme tail and average improve repeatably:

| Complete cold-tick metric | Baseline (ms) | Candidate (ms) |
|---|---:|---:|
| Mean | 77.479–77.867 | 72.747–72.875 |
| p99 | 467.720–470.937 | 242.012–245.586 |
| Maximum | 606.986–609.131 | 309.320–310.122 |

The 10× mean reduction is approximately 6%; cold p95 increases 0.6–4.1%. This tradeoff is retained because the targeted CCD tail and sustained large-connected workload improve repeatably without a repeatable separated regression. It does not meet the original latency or throughput gates.

At cold tick 8, the final candidate reduces continuous-query time from 393.649–394.697 ms to 90.115–90.403 ms. Its actual query work is:

| Counter | Instrumented baseline | Candidate |
|---|---:|---:|
| Full shape transformations | 25,121 | 9 |
| Starting-shape cache hits | 0 | 25,112 |
| Pose reconstructions | 21,441 | 9 |
| Velocity traversals | 5,270 | 3 |
| Collider hierarchy node-pair tests | 39,295 | 30,774 |
| Collider candidates | 1,840 | 1,840 |
| Triangle candidates | 21,432 | 21,432 |
| SAT evaluations | 23,281 | 23,281 |

## Foreground app and rendering

The final rebuilt app completed the disposable 1× fixture capture: **600 CPU ticks**, zero dropped or degraded ticks, every state published, and focus verified on every measured frame. Complete CPU tick p95 is **1.763 ms**; measured completion rate is **59.93 TPS** over the short capture. This does not establish a sustained 10× scheduler gate.

Rendering independently records **28.98 FPS** and **38.898 ms frame p95** on Apple M1 Pro / Metal, at 4112×2524, MSAA 4 and AutoNoVsync. Its 120 FPS target is unmet. The screenshot was inspected; F3 shows the CPU route, 13 bodies and 1,744 colliders. Raw capture, screenshot, launch identity and passing focus/dimension protocol checks are in `foreground/`. All 600 CPU records expose the new CCD counters. The source fixture remains untouched.

## Verification and existing workspace failures

- Final core/CPU/app tests after node compaction: **279 core and 189 CPU tests pass**; app **765 pass, one existing leader-marker UI failure, six ignored**. Command: `cargo test -p mechanic-core -p mechanic-physics -p mechanic-app --no-fail-fast`, with Metal access for the app's terrain-publication test.
- Thirty collision tests cover exhaustive paired-tree enumeration, large uneven trees with swept bounds, rotated builder pipes, suppressed pairs, finite terrain edges, full rotations, fast impacts, simultaneous arrivals, pose changes, terrain republication and origin rebasing. Public and new-contact sweeps compare exact hit identities/fractions against test-only uncached execution. Different motion paths at identical poses and cold versus warm compiled geometry match.
- `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all -- --check`, and `git diff --check` pass. **27 capture/benchmark-tool tests pass**, including rejecting different replay hashes despite matching physical quality.
- Final CPU quality scenarios (`reference-fixtures`, `car-drop`, `car-drive`, `block-pile`, `fast-impacts`, `four-bar`) complete. Every simulation case reports zero degraded ticks; reference fixtures converge. `quality/` retains results and binary hashes. These runs overlap build/verification work and are behavioral evidence, not matched timing evidence.
- The full `cargo test --workspace --no-fail-fast` hardware run before node compaction uses **Apple M1 Pro / Metal**. GPU: **110 pass, 11 fail, one ignored**; the 11 failure names exactly match the prior `verification/workspace-metal.log`. App has the same leader-marker failure. `workspace-failure-comparison.json` records the comparison. The initial sandbox run lacked a Metal adapter; hardware access resolved those environment failures. The workspace is not green. GPU/kernel code is unchanged; final affected core/CPU/app checks are reported above.

Full logs are retained under `verification/`.

## Identities, rejected work and reproduction

`baseline-identity.json`, `candidate-identity.json` and `app-identity.json` record executable and source-file SHA-256 hashes. The instrumented baseline was built and archived before optimization. Preserved executables live under `.physics-reference/ccd-followup/`; source archives include the pre-existing uncommitted work. Archives omit unchanged app assets, whose hashes remain in the identities. No archived source was rebuilt in the active Cargo target directory.

`baseline-cold-10x.jsonl` reproduces the reported 395 ms CCD tick before optimization. `optimization.patch` isolates the optimization and evidence-runner change relative to the instrumented baseline. `analysis.json` contains per-run stage timings, means, p99 and state comparisons; `analyze.py` regenerates it. The benchmark runner now requires exact replay hashes for its acceptance gate, in addition to existing quality checks. The 300-second no-progress watchdog is retained. No final run timed out or was incomplete.

The initial expanded-node implementation repeatedly worsened cold p95 by 2–9% and was rejected. Its executables, source identities, cold comparisons, quality runs and app capture remain under `expanded-node-*`; the wrapper was stopped after its cold group completed. Compact nodes remove the extra field and avoid unnecessary bounds reads during internal refits. `cold-diagnostic/` overlaps compilation, and the first 1× repeat in `packed-check/` overlaps Clippy; neither is used for timing acceptance.

`measure.sh` records the final cold, connected and additional separated commands; the separately recorded `packed-separated-check/manifest.json` supplies the first full separated pair. Reruns need fresh output directories. `verify-evidence.py` checks all 54 final replay files, executable hashes, protocols, complete tick sequences, all observed physics/candidate values, zero degradation/escape and repeat determinism.
