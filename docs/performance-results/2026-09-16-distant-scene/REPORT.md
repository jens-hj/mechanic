# Distant-scene amplification of the saved pipe

The user confirmed that the bearing pipe swings smoothly in world `new`, but causes lag in `builder` despite having nothing nearby to hit. The comparison below reproduces the amplification without placing obstacles near the pipe.

## Controlled comparison

The read-only snapshots are `new-instance-8.ron` and `builder-instance-141.ron`. The live saves were not changed. `pipe-scene` places the entire small `new` construction 20 metres along X, either by itself or with the unrelated builder construction retained around its original location. The isolated scene has 437 colliders and 6 bodies; the combined scene has 7,140 colliders and 39 bodies. The small scene includes its scattered blocks.

A flat diagnostic floor at authored y=4 is used. Parts whose compiled bounds reach this plane are anchored. This approximates the app's foundations but does not reconstruct its terrain-density anchoring or streamed terrain. Both variants use the identical floor, pipe geometry, gravity, solver settings, and initial pipe motion. The construction settles for 120 ticks; each case resets to that state, sets the pipe bearing's initial speed to 0, 1, 10, or 40 rad/s, and measures 600 ticks. Gravity remains active, so the lower-speed pipe can oscillate; the table labels initial speed rather than claiming a stationary orientation throughout.

Command:

```sh
target/release/cpu-physics --scenario pipe-scene   --instance docs/performance-results/2026-09-16-distant-scene/new-instance-8.ron   --background docs/performance-results/2026-09-16-distant-scene/builder-instance-141.ron   --warmup 120 --ticks 600
```

Omit `--background` for the isolated scene. Three paired runs alternated executable order. Compilation, profiling, and application captures were separate from the accepted timings. `manifest.json` identifies both binaries; JSONL preserves raw summaries and state hashes.

## Diagnosis and implementation

One fast collider activates scene-wide sweeps. Those sweeps prepare bounds for every collider, including distant stationary or slowly settling bodies. Fast rotation can also repeat the scene-wide proximity and recovery queries. The prior tightening of swept bounds did not eliminate this preparation cost.

A native three-second `sample` profile of the fast combined scene, after the initial start-bound cache change, attributed 535 of 911 sampled continuous-query stacks to the bounds-preparation collection. This includes bounds arithmetic, conservative acceleration evaluation, and endpoint transforms. `profile-after-start-cache.txt` preserves the profile. It is attribution evidence, not a timing comparison.

Changes:

- Reuse the cached initial collider bound only when its owning body's pose matches exactly. Cache ownership fixes topology. Changed poses still recompute bounds; terrain changes remain handled by the existing publication path.
- Compute each initial bound once within path preparation.
- When total point travel is below the query padding, retain the already-conservative speed envelope. Avoid more expensive endpoint/curvature calculations that add little precision in this case. This retains full-path coverage and does not skip collision evaluation.
- Reject body pairs before refitting their detailed collider trees. A body tree is refitted lazily on its first viable pair in that query, never reused stale across queries.

No geometry, timestep, speed limit, collision tolerance, solver iterations, or impact handling changed. All sweep candidates remain subject to the existing collision checks.

## Results

Median of the three per-run measurements, milliseconds per complete CPU tick:

| Scene | Initial speed (rad/s) | Before p50 | After p50 | Before p95 | After p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| alone | 0 | 0.042 | 0.039 | 0.044 | 0.041 |
| alone | 1 | 0.041 | 0.039 | 0.044 | 0.041 |
| alone | 10 | 0.041 | 0.039 | 0.065 | 0.058 |
| alone | 40 | 0.340 | 0.277 | 0.372 | 0.303 |
| builder | 0 | 2.550 | 2.407 | 2.754 | 2.607 |
| builder | 1 | 2.553 | 2.406 | 2.747 | 2.605 |
| builder | 10 | 2.567 | 2.405 | 2.787 | 2.611 |
| builder | 40 | 5.836 | 4.058 | 6.152 | 4.362 |

For the combined scene, the fast case drops about 30% in total tick cost. The added cost over the low-speed case falls from approximately 3.3 ms to 1.7 ms. All before/after final state hashes match exactly in every paired case. Both binaries report zero degraded ticks. The fast isolated and combined cases have matching final pipe speeds, and their body-pair sweep candidate counts are zero. The combined case still evaluates unrelated bodies against the floor, explaining why spatial separation alone did not remove the extra work.

## Remaining costs and limits

Fast motion still triggers scene-wide traversal and more contact queries. The combined fast case remains slower than both the combined low-speed case and the isolated fast pipe. This change reduces the reproduced amplification; it does not remove all scene-size dependence. Contact queries and remaining continuous-query preparation are further costs.

This is a controlled full-tick CPU reproduction using the saved pipe, not a replay of the user's hammer timing, walking collision, rendered frames, or streamed builder terrain. It establishes no app FPS improvement. The release app is rebuilt for retrying the original interaction.

## Verification

- Workspace Clippy with warnings denied: passed.
- Formatting and diff whitespace checks: passed.
- `cargo test --release -p mechanic-physics --lib`: all 196 tests passed, including cached/uncached sweep agreement, exhaustive collider-pair comparisons, fast impacts, full-turn rotation, moving ancestors, and suspension.
- Release fast-impact, car-drive, and car-drop checks: zero degraded ticks; all nine fast-impact cases report no tunnelling. Reports are retained separately; their timings are diagnostic only because compilation overlapped.
- `cargo build --release -p mechanic-app`: passed. Updated executable: `target/release/mechanic-app`; native frame-rate verification remains outstanding.
