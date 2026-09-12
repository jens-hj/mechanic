# BLOB physics optimization

This pass fixes the delayed and extremely slow suspension physics in the real
`blob` world without changing the 60 Hz timestep, eight contact sweeps, mechanism
iteration counts, contact approximation, or render quality.

The scene compiles 3,262 parts and 6,215 welds into 21 bodies and 29 colliders.
Fifteen bodies and 23 colliders are static. Before this pass, every resident
terrain chunk was published to physics and every static collider traversed that
terrain, even though `prepare_contacts` later discarded all resulting contacts.
The app now publishes a snapped, generously margined terrain cut around the 21
body positions and keeps ticking on an accepted cut while a wider cut prepares.
Ticks still block before the first cut, across a floating-origin rebase, and when
a body crosses the 48 m safety allowance. The terrain kernel rejects immovable
colliders before BVH traversal. The async readback ring is 12 slots, and elapsed
time catch-up is capped at 30 ticks with dropped time reported explicitly.

## Real World result

Both 60-second background captures used the same world manifest, release binary,
Apple M1 Pro / Metal, 4112 × 2524 target, 4× MSAA, no scripted input, and capture
from process start. They are unattended diagnostics rather than controlled
foreground frame benchmarks.

| Measurement | Before | After |
| --- | ---: | ---: |
| First terrain publication | 41.81 s | 11.31 s |
| First cut | 3,053 chunks | 603 chunks |
| First upload | 1,042.2 MB | 206.0 MB |
| First publication CPU | 2,026.7 ms | 87.8 ms |
| Terrain-contact GPU p50 | 537.21 ms | 0.0557 ms |
| Whole physics GPU p50 | 545.17 ms | 6.73 ms |
| Completed physics TPS | 0.47 | 48.60 |
| Backlog start → end | 0 → 422 | 0 → 0 |
| Dropped ticks | unavailable | 0 |
| Physics failure flags | 0 | 0 |
| FPS | 19.26 | 19.25 |

The post-change average includes the initial terrain gate. Once the first cut
publishes, physics runs near its 60 TPS target. FPS stays near 19 because the
remaining frame limit is rendering/presentation: window acquisition is 43.33 ms
p50 and opaque GPU rendering is 20.62 ms p50. Render quality was outside scope.

The static-collider skip is behavior preserving: terrain contacts have no second
body, and the preparation kernel discards them whenever the owning body's inverse
mass is zero. The added GPU regression observes four contacts for the movable
fixture and zero for the otherwise identical immovable fixture.

## Headless checks

All runs used a five-second warm-up and 30-second measurement except the two
60-second suspension-world traces.

| Scenario | TPS | GPU p95 | Flags | Result |
| --- | ---: | ---: | ---: | --- |
| `four_bar` | 145.40 | 6.174 ms | 4 | correctness failed after the longer run |
| `bearings_4` | 174.93 | 4.893 ms | 0 | correct; missed 4 ms budget |
| `test2_car` | 136.30 | 6.699 ms | 0 | passed |
| `dense_100k` | 6.67 | 148.454 ms | 9 | failed; incomplete coverage remains |
| suspension terrain | 13.11 | 165.505 ms | 0 | finite, but known overturn remains |
| suspension plane | 62.40 | 14.888 ms | 0 | stable and passed 60 TPS |

`four_bar`, `bearings_4`, and `dense_100k` do not exercise the optimized terrain
publication or terrain-contact path. `GpuCollider.shape.w` is read only by
`generate_terrain_contacts`; the app cut, readback ring, and backlog cap are also
absent from this serialized harness. Their failures therefore describe the
broader current worktree and are retained rather than attributed to this pass.
The August `dense_100k` result was 68.32 TPS and 14.013 ms GPU p95 with 221,824
active contacts; the current fixture reports 1,374,718 active contacts.

The suspension terrain result sits within the previous candidate range of
11.17–13.32 TPS. That harness always publishes a dense 3×3×3 neighborhood and has
no static construction, so it does not model this pass's World cut reduction.

## Verification

- `cargo fmt --all -- --check`: passed.
- `cargo clippy --workspace --all-targets --offline -- -D warnings`: passed.
- Capture summarizer regressions: 8 passed.
- Focused terrain publication, scheduler, readback-ring, and GPU terrain tests:
  passed, including 27 serial GPU terrain tests on Apple M1 Pro / Metal.
- `cargo test --workspace --offline`: 733 app tests passed, then the known
  `leader_geometry_follows_projected_positions_without_remounting` failure stopped
  the run. Repeating with only that test skipped passed all 733 remaining app
  tests, including `dynamic_presets_do_not_gain_unbounded_spin`. The GPU crate then
  reported 34 adapter `NotFound` failures when its real-GPU tests requested Metal
  adapters concurrently; 84 GPU tests passed. Serial focused GPU tests pass.

Raw evidence is in `before-world.jsonl`, `after-world.jsonl`, `headless.jsonl`,
`suspension-terrain.jsonl`, and `suspension-plane.jsonl`. The corresponding World
summaries and post-change screenshot are archived beside them.

## Grounded suspension follow-up

A small grounded mechanism previously encoded each contact-solver and terrain
recovery dispatch as a separate Metal compute pass. In the settled BLOB scene
those dispatches had zero work, but their pass boundaries still occupied the GPU.
The small grounded route now records the same kernels in the same order in fewer
passes, including all eight configured contact sweeps and all three terrain
correction rounds. Large or looped mechanisms retain the general schedule.

Matched 60-second debug captures on Apple M1 Pro, after terrain publication and
with zero active contacts, measured:

| Measurement | Before | Merged pass |
| --- | ---: | ---: |
| FPS | 16.24 | 24.50 |
| Frame p50 | 61.05 ms | 41.12 ms |
| Physics GPU p50 | 6.91 ms | 2.88 ms |
| Contact solver p50 | 2.62 ms | 1.07 ms |
| Terrain recovery p50 | 2.51 ms | 0.40 ms |
| Completed physics TPS | 60 | 60 |
| Dropped ticks / failure flags | 0 / 0 | 0 / 0 |

The serial Metal terrain suite passed 28 tests (one diagnostic ignored), including
fast impacts and articulated suspension recovery. Raw follow-up evidence is in
`spring-pass-merge-debug.jsonl`.

## Dynamic suspension follow-up

The `car` world exposes a different limit: its movable root forces contact
impulses through the suspension tree after every contact point. The solver keeps
that ordered path because batching a wheel's manifold points let the existing
car-drop regression penetrate the ground. The rejected batching code is not in
the worktree.

The accepted path solves ordinary spring/damper travel analytically. Rubber
bump-stop engagement retains the existing eight Newton iterations. App catch-up
is limited to three tick submissions per rendered frame so a persistent backlog
cannot consume every available readback slot before Bevy submits the next frame.
Every submitted tick still uses the fixed timestep and all eight contact sweeps.

The before trace recorded 53.1 seconds before the automation deadline; the after
trace completed a valid 60-second capture and screenshot. Both used a disposable
copy of the same `car` save, scripted driving, release app, Apple M1 Pro / Metal,
4112 × 2524, and 4× MSAA.

| Measurement | Before | After |
| --- | ---: | ---: |
| FPS | 8.59 | 9.95 |
| Frame p50 | 104.06 ms | 99.78 ms |
| Frame p95 | 241.69 ms | 132.89 ms |
| Tick submissions / frame | 4.02 | 3.00 |
| Contact solver p50 | 10.45 ms | 10.87 ms |
| Recovery projection p50 | 1.04 ms | 0.12 ms |
| Failure flags | 0 | 0 |

The final screenshot shows the car intact at 28.1 km/h with eight of eight
solver sweeps and zero failure flags. Raw evidence is in
`car-dynamic-before.jsonl`, `car-dynamic-after.jsonl`, and
`car-dynamic-after.png`. The summarizer currently rejects these traces because
readback-lag at the capture boundary leaves 10–17 skipped tick indices outside
the sampled dropped-tick counter range; the raw result record is valid and tick
indices remain strictly increasing.

### Dynamic terrain-recovery correction (2026-09-09)

A manual full-speed run fell through terrain after enabling the pass-merged
small-tree recovery route for movable roots. The deterministic automation did
not reproduce the intermittent failure: both separated and merged routes stayed
grounded for 60 seconds of straight full throttle. Because the merged route had
no dynamic-root terrain regression, dynamic-root mechanisms conservatively
return to the established fully separated recovery passes. Anchored small trees
retain the optimization.

The final separated-route capture saved twelve five-second driving frames,
completed 1,821 physics readbacks with zero failure flags, and kept terrain
contacts through the final tick. The strict ground-plane car-drop test still has
its existing 5.4 mm wheel-penetration failure at tick 30 in both routes; it does
not exercise this terrain path and is not evidence for this correction. Raw
evidence is in `car-ground-fix.jsonl` and `car-ground-fix.png`. Its valid result
record completed, while the summarizer again rejected only the known
capture-boundary dropped-tick accounting mismatch.

### Moving terrain-streaming focus correction (2026-09-09)

The remaining fall-through had a separate cause. While seated, the camera and
seat followed the simulated vehicle but `PlayerState::position` stayed at the
seat-entry point. Terrain selection therefore remained centred on that stale
position. Physics could republish only the terrain already resident there, so a
vehicle eventually drove beyond the available collision meshes even though its
96 m physics interest box was moving correctly.

Terrain selection now resolves the occupied seat's live simulated pose and
continues using player position while walking. Its critical and pinned nodes use
the same focus. Physics waits when any current local critical node is unresolved
instead of advancing over a known streaming hole.

A real Metal run of the `car` save remained grounded through a 60-second capture
from world entry, ending at 24.2 km/h after 18 terrain reselections and 37
physics-terrain publications. It completed 1,371 readbacks with zero error flags
and still had 22 terrain contacts on the final completed tick. The readiness gate
blocked 337 sampled frames while terrain lagged; the final frame safely held
physics with 48 of 50 local nodes ready. The run includes five-second screenshots
and is diagnostic rather than a performance comparison. Raw evidence is in
`car-streaming-focus.jsonl` and `car-streaming-focus.png`.
