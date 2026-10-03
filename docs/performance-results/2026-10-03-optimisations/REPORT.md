# Optimisation pass (issue #63)

A broad audit of CPU physics, world generation and streaming, water, GPU
physics, construction edits and the app, followed by changes that leave every
system's output unchanged or, where the arithmetic order moved, statistically
the same. Each change was measured against the commit before the pass,
`a2cd0d0`.

## Conditions

- Apple M1 Pro, 10 cores, 16 GB, macOS, Metal.
- `profiling` profile (release with debug info).
- The machine was shared with other build and test sessions throughout; load
  average ranged from 5 to over 200. Wall-clock timings are therefore paired
  A/B/A/B runs and reported as ranges. Deterministic workloads are compared by
  the CPU cycles and instructions `/usr/bin/time -l` reports, which hold up
  far better under contention.
- Changes meant to be bit-identical were checked through the CPU solver's
  per-tick state hash, or a digest of every meshed chunk's vertices, normals,
  materials, surfaces and indices (`terrain-cut` now prints it as
  `mesh_digest`).
- CPU profiles: macOS `sample` at 1 ms.

## Baselines and results

| Workload | Measure | Before | After |
|---|---|---|---|
| `cpu-physics --scenario builder-scale --copies 1`, 600 ticks (13 bodies, 1,744 colliders, ~2,650 contacts) | CPU cycles | 64.0 G | 45.4–46.1 G (−28 %) |
| same | tick median / p95 | 34.5 / 49.3 ms | 23.0 / 34.5 ms |
| same | contact rows / query, median | 16.3 / 17.6 ms | 7.5 / 14.9 ms |
| same | contacts, degraded ticks | 2,652, 0 | 2,652, 0 |
| `terrain-cut --seed 42`, all nine biomes (108,408 nodes) | CPU cycles | 4,334 G | 3,503 G (−19 %) |
| same | instructions | 17,994 G | 14,354 G (−20 %) |
| same | sampling / extraction CPU | 5,773 / 1,066 s | 4,062 / 537 s |
| same | cold selection, summed over biomes | 14.3 s | 7.2 s |
| `terrain-cut --biome titan_crags` | CPU cycles | 603 G | 487–495 G (−18 %) |
| `water-breach` (120 s of flooding) | CPU cycles | 449 G | 333 G (−26 %) |
| same | remeasure / sediment p95 | 19.4 / 12.9 ms | 15.5 / 10.7 ms |
| `mechanic-bench --scenario terrain_dig` | cached selection p95 | 27–33 ms | 17–18 ms |
| same | terrain stage p50 | 29–32 ms | 19 ms |

Unchanged by design, and measured only as baselines:

| Workload | Measure | Value |
|---|---|---|
| `cpu-physics` small scenes (car-drive, car-drop, four-bar, gear-train, wheel-roll) | tick p95 | < 0.2 ms |
| `mechanic-bench --scenario smoke` (GPU, 1,024 bodies) | engine tick p50 / p95 | 5.3 / 7.0 ms |
| `bearings_4` | engine tick p95 | 10.7 ms |
| `four_bar` | engine tick p95; flags | 8.6 ms; `error_flags` 4 (known) |
| `test2_car` | engine / GPU tick p95 | 8.5 / 5.9 ms |
| `dense_100k` | engine tick p50; TPS | 164 ms; 6.1 (`error_flags` 9, known) |
| `player_collision` | query / refit p95 | 0.008 / 1.21 ms (gate passes) |
| `material-clumps` | tick p95 | 2.0 ms |
| `edit-latency` | pipe-junction boundary query | 0.34 ms |

Peak memory showed no consistent change. The builder replay peaks at
123–128 MB either way. `terrain-cut` peaks anywhere from 75 to 119 MB in
repeated runs of the same binary, depending on how its ten workers' jobs
overlap; four paired single-biome runs split two each way.

## Changes

### CPU physics

1. **Contact rows share each body's responses.** Every contact row is a
   combination of six spatial unit impulses at its body's centre of mass, so a
   substep now solves those six per touched body
   (`MachineKinematics::body_responses`) and combines them, instead of running
   the articulated factor once per row: about 32,000 solves per builder tick
   become about 300. The combined responses match direct solves to 1e-9
   relative on every row checked, and usually bit for bit: the builder replay
   keeps the baseline's state hash for its first 1,749 ticks, and eight
   nanometre-jittered 4,200-tick replays matched the baseline throughout.
   Contact rows fell 16.3 → 7.5 ms.
2. **Separation cache hashed with Fx, candidate pairs sorted as one key.**
   SipHash was 8 % of the tick; the order of the sorted pairs is unchanged.
   Bit-identical.
3. **Fits looked up by hash; each collider keeps its own shape buffer.** The
   joined-pair filter did a binary search per candidate pair, and a
   re-transformed shape took any spare buffer and regrew it. Bit-identical,
   about 4–5 % fewer cycles.

### World generation, streaming and water

4. **Tape argument registers gathered once.** Point, interval and grid
   evaluation re-derived each op's inputs from an iterator at every point;
   `Op::inputs` alone was 7 % of meshing. Interval evaluation also keeps
   small tapes on the stack. Bit-identical.
5. **Grid rows evaluated as lanes.** Arithmetic ops read a contiguous row or a
   broadcast value instead of a strided, bounds-checked index per element.
   Bit-identical.
6. **Fx hashing for the mesher's vertex and sample maps.** Extraction CPU fell
   38 %. Bit-identical.
7. **Lattice columns through the thread's column cache.** Stacked water cells
   and edits over the same columns stop re-evaluating them. Bit-identical;
   water −15.6 % cycles.
8. **Terrain cut balanced and stitched with hashed membership.** Balancing
   checked every selected node's six faces against a `BTreeSet` on every
   pass. It now uses a hash set, and since the cut never overlaps it only
   looks up the levels that can matter: two or more above a node when
   balancing, exactly one above when stitching. Identical cut and meshes.

### Fixed along the way

9. **Level-6 owners were invisible to balancing and stitching.** Level 6 was
   added for the 640 m–1 km band, but the owner lookup still stopped at level
   5, so level-5 nodes facing level-6 ones got no transition faces and the
   2:1 balance was not enforced against level 6. Fixed; in the verdant_hills
   cut this adds 9,294 transition triangles and no nodes.

## Rejected after measurement

- Remembering the face that last separated a collider pair and testing it
  first: no measurable change.
- Reusing projections for exactly opposite faces in the separating-axis test:
  1 % slower (the per-call table cost more than it saved).
- Four independent accumulators in the vertex projection: 17 % slower.
- Batching each scatter instance's shape across a grid row: at most 2.5 %,
  within noise on two of three biomes, not worth the code.

## Correctness checks

`cargo xtask test` on the branch, Apple M1 Pro / Metal:

- `mechanic-core` 400 pass; `mechanic-world` 272 pass; `mechanic-app` 947
  pass, 16 ignored; `mechanic-bench`, `xtask` and the WGSL checks pass.
- `mechanic-gpu`: 101 pass, 11 fail, all failing identically at `a2cd0d0`:
  five contact tests, four vehicle tests, the pendulum test, and
  `generated_caves_stop_wall_and_ceiling_impacts` (0.005108 m penetration on
  tick 1 in both).
- `mechanic-physics`: 249 pass, 2 fail. `a_box_dropped_on_a_resting_box…`
  fails at `a2cd0d0` too. `captured_blocks_dropped_across_a_ledge_come_to_rest`
  is new; see below.
- The `bevy_mosaic` doctest failed once with E0460 while other sessions
  rebuilt `bevy` in the shared target directory, and passes on its own.
- `cargo xtask fmt`, `consistency` and `lint` pass.

### The ledge test

`captured_blocks_dropped_across_a_ledge_come_to_rest` drops three captured
blocks across a ledge and samples the pile after 600 ticks. The landing is
chaotic, and the test's own comment says so. At `a2cd0d0`, raising the
starting poses by 1–15 nm buries a block beyond the 5 mm limit in 8 of 16
runs, two of them by about 3.8 cm. With change 1 the same sweep gives 6 of
16, one at 3.9 cm. Change 1 moves rounding, and the unperturbed start now
lands in a failing pose.

The failures are a real solver defect, not a buried block at rest. The top
block topples off an edge (correctly), then never settles: its contact with
the block below alternates between a ten-point face manifold and a single edge
point, the overlap jumps to about 3.7 cm every six or seven ticks, and the
push-out bounces it back. Fixing that limit cycle is physics work outside
this pass and is left as a follow-up; the test was not relaxed.

### Long builder runs

A full 4,200-tick builder replay degrades from tick 3,000 in both builds
("speed limit": a body passes the 500 rad/s cap), and eventually the machine
comes apart: continuous broadphase then finds millions of candidate pairs and
ticks take seconds. In eight jittered replays both builds degraded on the
same tick and came apart on the same tick (or not at all) every time. This
matches the known spin-gain flakiness, predates the pass, and is a physics
fix rather than a performance one.

## Not measured

- **In-app frame time.** Two background captures failed: the builder fixture
  never leaves loading (its saved player pose finds no terrain collision with
  the current generator), and a `water11` capture hung at start-up while
  another session held the GPU for app tests. App-side effects here are
  inferred from the headless workloads the app runs.
- **GPU frame cost** of the 5↔6 transition faces added by fix 9.

## Remaining opportunities

- The CPU collision query is still about 15 ms on the builder: convex
  separation (21 %), manifold clipping (14 %), pair traversal (13 %). The
  buried-vertex query re-traverses candidate pairs the proximity query just
  found; filtering that list instead would save the second traversal.
- Water's ground lookup still evaluates density cell by cell down a column
  and misses the 1,024-slot column cache across a ~20,000-cell sheet. A larger
  or water-owned cache would cut it, at several MB per thread.
- Per-column point evaluation in `CompiledWorld::column` could use grid
  evaluation across a lattice's columns.
- The ledge limit cycle and the builder's spin gain (above).
