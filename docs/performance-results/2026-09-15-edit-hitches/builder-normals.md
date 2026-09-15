# Remaining edit hitches in the current builder save

The user reproduced pauses after releasing block/pipe placement and at every
chamfer/fillet drag step, using CPU physics. The earlier generation-20 fixture
did not reproduce the expensive geometry now present in the save. A read-only
snapshot of generation 119 does: rebuilding ordinary body meshes takes about
1.54 seconds, even with evaluated solids already cached.

## Cause and fix

`append_evaluated_band` calculated each smooth vertex normal by scanning every
surface in the solid, then scanning all half-edges for each candidate surface.
This work repeated for every emitted corner. Two rounded rubber cylinders alone
took about 640 ms each. Rebuilding all body meshes after a construction
publication, or changing a feature-preview amount, ran those same searches on
unrelated existing rounded parts.

`evaluated_smooth_normals` now traverses face loops once and accumulates normals
by `(smoothing group, vertex)`. A face contributes once per vertex. Face order,
hard seams, smoothing groups, and material-band behavior are preserved. The
mesh emitter looks up these accumulated normals instead of searching the solid.
No physics behavior or geometry resolution changed.

## CPU placement publication

The first native comparison exposed another main-thread cost: installing CPU
physics took 98–103 ms per placement, even after fixing normals. A separate
probe measured `MachineCollisionGeometry::new` at 145.8 ms in the test profile;
initializing the machine's numerical state took 0.24 ms.

`cpu_physics::PreparedRoute` now builds immutable collision geometry in the
existing world-physics preparation worker. It owns the matching compiled
creation and revision. Installation checks the revision and supplies the latest
authoritative body and joint state, so bodies continue moving during preparation
without jumping back to the worker's starting poses. Both ordinary placement
and explicit weld publication use this path. Terrain residency transfer remains
at the publication boundary.

## Saved-world CPU measurement

Same generation-119 graph, 30 compiled bodies, 69,183 emitted vertices. These
measurements use the optimized test profile; they measure ordinary body mesh
construction, excluding ECS entity changes and GPU upload. The diagnostic also
checks one edge on each of 121 cuboids with each treatment; these attempts took
less than 1 ms each and include rejected edits, not a simulated pointer gesture.

| Measurement | Before | After |
| --- | ---: | ---: |
| Warm complete body mesh rebuild | 1,563–1,572 ms | 8.6–11.3 ms |

The initial, less instrumented baseline was 1,543–1,546 ms. Per-body timing
and texture/end-cap overhead are retained in `builder-normals-before.txt` and
`builder-normals-after.txt`. The before run also shows that ordinary pipe texture
bookkeeping is small compared with smooth-normal generation.

The three compressed RON files named `builder-generation-119-*` preserve only
the selected generation and world manifest; original file hashes are in
`builder-generation-119-sha256.json`. The user's world was not opened for writing.

Reproduce the mesh measurement from the repository root:

```sh
gzip -dc docs/performance-results/2026-09-15-edit-hitches/builder-generation-119-world.ron.gz > /tmp/builder-edit-world.ron
MECHANIC_EDIT_FIXTURE=/tmp/builder-edit-world.ron \
  cargo test -p mechanic-app --bin mechanic-app measure_builder_body_mesh_rebuild -- --ignored --nocapture
```

## Verification

The new regression compares every emitted normal for a filleted cuboid and
rounded cylinder against an independent exhaustive incident-face reference,
including both smooth surfaces and hard seams. The 64 rendering tests pass;
two hardware/diagnostic tests are ignored. Clippy for all app targets passes
with warnings denied.

The final broad app run passed 795 tests, with eight ignored and two filtered out:
the Metal-dependent inherited-cut test and the already documented suspension
UI marker test. Formatting and whitespace checks passed.

```sh
cargo test -p mechanic-app --bin mechanic-app -- \
  --skip terrain_publication::tests::an_inherited_cut_initializes_cpu_terrain_after_a_body_split \
  --skip ui::suspension::tests::leader_geometry_follows_projected_positions_without_remounting
cargo clippy -p mechanic-app --all-targets -- -D warnings
cargo fmt --all -- --check
```

A CPU-route regression prepares geometry, changes the supplied body position
and velocity, then verifies installation and the next tick use that newer state.
It also rejects a mismatched revision and incomplete state arrays. Existing
terrain-transfer, remeshing, removal, collision, and joint tests still pass.

## Native diagnosis

Apple M1 Pro, Metal renderer, CPU physics, the same generation-119 snapshot,
scripted placements every five seconds, and disposable world stores. These
captures set the existing `MECHANIC_PERF_CAPTURE_FROM_START=1` switch through
small executable launchers, so they include loading and terrain streaming.
The default all-terrain warm-up timed out before capture; that attempt is not
used as a measurement. An initial GPU-route attempt hit the already documented
tick-2 flags-4 failure; the user subsequently confirmed CPU mode.

`builder-native-before.jsonl.gz` preserves the old executable's trace.
`builder-native-normals-only.jsonl.gz` preserves the intermediate build before
moving CPU geometry preparation. Both completed without overflow or interruption.
The actual binary hashes and launcher contents are recorded separately in
`builder-native-executables.json`, since the capture launcher's own run record
hashes the shim. These runs are diagnostics with streaming, not controlled
steady-state FPS comparisons.

In the old trace, intervals between consecutive frame records that contain a
placement installation take 2.09–2.23 seconds. With the normals-only fix, body
mesh rebuild events take 3.1–3.6 ms; the 100 ms CPU installation cost remains
visible separately. Frame records use the app's time resource, so wall-clock
intervals between records are used to identify long pauses rather than relying
on its clamped `frame_ms` field.

The final release capture (`builder-native-after.jsonl.gz`) completed twelve
placements without interruption or overflow. CPU geometry preparation took
96.7–101.4 ms on the worker. Main-thread physics installation took 0.39–9.11 ms,
and body mesh rebuilding took 2.97–3.56 ms. The installation frame-record gaps
were 42–202 ms, versus 2.09–2.23 seconds before both fixes. Ongoing terrain
streaming and simulation still affect frame time, so this is evidence that the
edit stalls were removed, not a claim of steady-state 60 fps. Per-run stage
counts and ranges are retained in `builder-native-summary.json`.

The final release executable was rebuilt successfully after both fixes. Final
Clippy, formatting, whitespace, and the updated prepared-state regression pass.

This fixes the measured shared mesh-generation stall. It does not establish a
whole-app 60-fps gate or prove all first-time feature evaluations fit a frame.
