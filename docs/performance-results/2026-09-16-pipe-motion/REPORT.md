# Saved builder pipe-motion follow-up

The earlier single-cuboid rotor benchmark did not represent the many-piece pipe geometry in the user's builder world. A pipe bend compiles to 192 colliders; the preserved generation 138 contains 6,703 colliders, 33 bodies, and 15 independent joint coordinates.

## Reproduction and scope

`builder-instance-138.ron` is a read-only snapshot of the saved construction. The live world was not changed. `cpu-physics --scenario pipe-motion --instance PATH --warmup 3 --ticks 12` repeats each single-joint substep path from the authored poses at speeds 0, 1, 10, and 40 units/second (rad/s for rotational joints, m/s for suspension). Path duration is 1/240 second. Other generalized displacements are zero. The benchmark calls the production continuous collision query directly with 1 mm tolerance and 64 evaluations per candidate. It excludes terrain, integration, contact forces, the hammer input, rendering, and camera motion.

This isolates a reproduced source of collision cost. It is not a full replay of the screenshot's hammer interaction, and the pictured bearing assembly has not been positively identified in generation 138. No app FPS gain or resolution of that exact interaction is claimed.

The baseline includes the previous fast-motion changes and the identical benchmark harness. Three alternating before/after pairs ran without a compiler or profiler running. `manifest.json` records executable and snapshot SHA-256 hashes. Raw JSONL and `summary.json` preserve all 60 cases in each run. Initial results before eliminating redundant full-circle work are retained under `initial-chord/`.

## Cause and change

The previous bounds expanded each collider by its total point travel, intersected with a full rotation circle when available. Short arcs could still cover large unreachable regions. One clear path at 40 rad/s produced 31,524 candidate pairs and roughly 318 ms of collision work for just one substep.

The new bound uses each material point's endpoint chord plus the existing conservative acceleration bound: over normalized time [0,1], deviation from the chord is at most A/8. Taking endpoint shape bounds and expanding by that amount encloses all intermediate positions. The acceleration bound includes moving ancestors, off-centre bearings, angular acceleration, and suspension travel. Full turns cannot disappear just because endpoint orientations match. The original conservative envelope remains an additional bound, and long fixed-axis paths also retain the full-circle envelope. Short paths avoid computing the extra circle.

No collision geometry, tolerances, time steps, speed limits, impact handling, or scheduler behavior changed.

## Results

Median of three per-run p50 sweep times, in milliseconds:

| Coordinate | Speed (rad/s) | Before | After | Candidate pairs |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 0.498 | 0.503 | 0 → 0 |
| 0 | 1 | 0.537 | 0.553 | 0 → 0 |
| 0 | 10 | 1.919 | 0.555 | 166 → 0 |
| 0 | 40 | 317.520 | 0.562 | 31524 → 0 |
| 1 | 40 | 17.146 | 16.765 | 21948 → 21581 |
| 2 | 40 | 131.339 | 16.581 | 31519 → 21540 |
| 3 | 40 | 57.574 | 0.644 | 21946 → 1 |

Every before/after query outcome matched exactly, including impact target, fraction, and separation for paths that actually hit geometry. The most expensive clear sweep fell by about 99.8%. This is a sweep-time result, not a whole-tick or FPS measurement.

Remaining costs: collision-producing paths still take roughly 16–17 ms per sweep and retain over 21,000 pairs. Traversing the full scene costs about 0.5 ms even with no candidates. Slow-case differences are small in absolute terms but are not proven zero: coordinate 0 at 1 rad/s rose from 0.537 to 0.553 ms in this short batch. These measurements do not establish the original end-to-end acceptance gate.

## Verification

- `cargo test --release -p mechanic-physics --lib`: 196 passed. Extended sampled-path checks cover small positive/negative rotations and moving ancestors, alongside existing full-turn, suspension, and collision regressions.
- `cargo clippy --workspace --all-targets -- -D warnings`: passed.
- `cargo fmt --all -- --check` and `git diff --check`: passed.
- Release `fast-impacts`: all nine cases reported no tunnelling and no degraded ticks.
- Release `car-drive` and `car-drop`: zero degraded ticks.
- Quality-report timings are not comparison evidence: these ran alongside app compilation.
- `cargo build --release -p mechanic-app`: passed; updated executable is `target/release/mechanic-app`.

## Whole-tick follow-up

`tick-regressions/` records three paired runs of the existing 23 generated rotor, translation, and driving cases (120 warmup and 600 measured ticks per case) after compilation finished. All final state hashes match and both binaries report zero degraded ticks. See its summary for per-case ratios; these generated scenes do not replace the missing exact hammer replay.

Slow and driven whole-tick cases stayed near baseline in this batch. The isolated 500 rad/s cuboid rotor has a repeatable approximately 6% p50 increase (0.008750 to 0.009291 ms); preparing the additional chord envelope costs time when there is no collision work to reject. This remaining overhead is retained explicitly rather than claiming improvement in every fast case.
