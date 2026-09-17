# CPU soil response — 17–18 September 2026

A generated suspension car drove on its saved seed-91 terrain, with five seconds
of rest followed by ten seconds of W input. Rigid and soil runs used the same
release executable, ran sequentially, and had no concurrent compilation. The
host CPU was an Intel Core i5-12600K. Exact commands and input/binary hashes are
in [manifest.json](manifest.json).

| Measured quantity | Rigid | Soil |
| --- | ---: | ---: |
| Maximum measured mesh rut depth | 0 mm | 7.408 mm |
| Degraded ticks, including warm-up | 0 / 900 | 0 / 900 |
| Solver tick p95, 600 driving ticks | 3.631 ms | 3.733 ms |
| Total tick p95, including synchronous meshing | 133.543 ms | 143.926 ms |
| Remeshed chunks during driving | 0 | 1,089 |

The deepest rut reaches 6.226 mm after one second, 7.218 mm after four seconds,
and 7.408 mm after driving starts, then remains at that maximum through the end
of the run. Maximum rut depth is measured by raycasting regenerated meshes at
loaded cell centres against their pre-compression heights. Summed density
displacement is reported separately and is not a rut-depth measurement.

Solver p95 increased 2.8% and total tick p95 increased 7.8% in this single matched
pair. **The no-regression acceptance check is not met.** These results do not
establish a stable statistical performance difference. The benchmark rebuilds
meshes synchronously and includes halo neighbors; the app uses its existing
asynchronous edit/remesh pipeline. Both runs also mesh new terrain while driving,
which accounts for substantial total tick cost even without soil.

The fixed-load world tests establish hardening and reduced sinking on a second
pass. A maximum-depth plateau in a moving-car replay alone does not establish
that every loaded cell has stopped changing.

Raw results: [rigid.jsonl](rigid.jsonl), [soil.jsonl](soil.jsonl),
[summary.json](summary.json). Every record has `kernel_coverage_complete: false`.
No GPU or clump scale gate is claimed.

## Implementation corrections to the original plan

- Contact impulses are per substep. The CPU collector sums accepted substeps and
  includes final restitution, rather than reporting only the last substep.
- Manifold grouping includes collider identity because manifold numbers are
  local to a collider. Reused vectors avoid per-tick collector allocations after
  capacity stabilizes.
- Soil needs continuous mesher reconstruction. Applying the existing binary
  brush clamping to its first tiny load can snap procedural surfaces. Compacted
  samples retain continuous densities; excavation resets compaction and retains
  the existing bounded reconstruction. A regression covers several sub-cell
  surface positions.
- The edit worker receives accumulated per-cell displacement tied to the source
  sample, rather than recomputing pressure when a queued patch finally executes.
  Stale loads are dropped. Ready cells are grouped by brick within a worker batch.
- The replay now respects the saved instance translation. Rotated instances or
  stored joint coordinates are rejected explicitly.

## Verification

- `cargo test -p mechanic-world --offline`: 101 tests passed, including continuous
  first-load geometry, hardening, collapse, rigid materials, exact persistence,
  and meshed height after reload.
- `cargo test -p mechanic-physics --offline`: one existing failure,
  `captured_blocks_landing_on_each_other_come_to_rest`. It was reproduced with
  the unmodified physics sources and identical final state; see
  [baseline failure](baseline-physics-failure.txt).
- The resting-weight load test passes with two and eight substeps. The separate
  impact-load regression passes and accounts for restitution.
- `cargo test -p mechanic-app --offline soil_commits_on_sixth_tick_and_survives_world_reload`:
  passed. This exercises cadence, queue saturation, foundation invalidation and
  actual WorldStore persistence.
- `cargo clippy --workspace --all-targets --offline -- -D warnings`: passed.
- `cargo fmt --all -- --check`: passed.
- The app built with debug symbols and incremental compilation disabled to fit
  disk space. Those environment settings do not alter the release benchmark.

## App restart and persistence

The CPU app ran the saved suspension car for one minute, exited, and reopened
that same isolated world. The first direct executable launch lacked an asset
root, so its visual capture was discarded; physics and autosave still ran. The
reload supplied `BEVY_ASSET_ROOT=/home/j/repos/mechanic/crates/mechanic-app` and
loaded textures and shaders successfully.

All **3,477 compacted cells in 135 v3 bricks** retained or increased their
compaction after restart; none lost compaction. The reloaded app exited
successfully with 142/142 local terrain nodes ready and zero streaming backlog.
The renderer used an AMD Radeon RX 9070 XT, RADV/Mesa 26.2.2, Vulkan.

[App reload audit](app-reload.json), [reload log](app-reload.log), and
[reload screenshot](app-reload.png) retain the evidence. The screenshot shows
zero CPU degraded ticks and 164 dropped startup ticks. This was a capture from
startup, not a settled app performance comparison. Its diagnostics overlay
obscures the car, so a close-up visual rut demonstration remains outstanding;
mesh-height tests and the headless replay establish the geometric deformation.

The reload command was:

```sh
BEVY_ASSET_ROOT=/home/j/repos/mechanic/crates/mechanic-app \
MECHANIC_AUTO_WORLD='Suspension performance' \
MECHANIC_AUTO_WORLD_STORE=/tmp/mechanic-soil-app-tzyd9fls/worlds \
MECHANIC_PERF_CAPTURE_DIR=/tmp/mechanic-soil-app-tzyd9fls/reload-capture \
MECHANIC_PERF_CAPTURE_FROM_START=1 \
target/debug/mechanic-app
```

The first run additionally set `MECHANIC_AUTO_DRIVE=1` and
`MECHANIC_AUTO_DRIVE_STRAIGHT=1`. Every app run used the isolated world copy.

## Scope and remaining work

Compaction removes volume. It does not conserve material, create berms or
clumps, or atomically publish bodies together with terrain. Only upward-facing
ground compacts. GPU load readback and remesh performance remain follow-up work.
The rigid-ground scripted-driving issues remain open.

Brick v3 directly replaces v2. The preserved builder-world fixture contains no
terrain bricks. Verification used fresh worlds under `/tmp`; existing personal
worlds were not deleted or migrated.
