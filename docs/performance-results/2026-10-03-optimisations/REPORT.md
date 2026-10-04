# Optimisation pass (issue #63)

A broad audit of CPU physics, world generation and streaming, water, GPU
physics, construction edits and the app, followed by changes that leave every
system's output unchanged. Each change was measured against the commit before
the pass, `a2cd0d0`.

PR #67 first shipped a contact-row change that reordered arithmetic and
altered the builder's long-run behaviour. The follow-up below replaces it
with exact row solves; builder figures marked "PR #67" are that superseded
state, kept as history.

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

## Follow-up: exact contact rows (after PR #67)

PR #67 shipped change 1 (shared per-body responses). Its replay of the
builder kept the baseline's state for 1,749 ticks, parted at tick 1,750 and
came apart at tick 2,624 instead of the baseline's 3,699. The baseline spread
argued below does not establish that as equivalent, so the follow-up replaces
change 1 with a contact-row path that is exact.

**Isolation.** `main` after PR #67 with only the shared-basis path switched
off matches `a2cd0d0`'s builder state hash on all 3,698 ticks before the
baseline comes apart. Every other change in the pass is therefore
bit-identical over the full run, and change 1 alone caused the divergence.

**Replacement.** Contact rows go back to one direct articulated solve per
row, as at `a2cd0d0`. Two changes recover speed without touching the
arithmetic:

- The backward sweep skips a body whose own entry and accumulated load are
  both exactly zero. Such a body only ever passed zeros up.
- A contact's three or five rows are solved together (`solve_ranges_many`):
  one walk over the bodies, each row seeing exactly the operations its own
  solve performs.

Builder replay, interleaved, load 2.6–9:

| | Baseline `a2cd0d0` | Shared basis (PR #67) | Exact rows |
|---|---|---|---|
| tick median | 33.0–33.4 ms | 21.3–21.4 ms | 25.9–26.1 ms (−22 %) |
| tick p95 | 46.7–46.9 ms | 31.0–31.1 ms | 36.4–36.6 ms (−22 %) |
| contact rows median | 16.1 ms | 7.1 ms | 11.4–11.7 ms |
| CPU cycles | 62.5–62.9 G | 43.8–43.9 G | 49.7 G (−21 %) |
| state hash vs baseline | — | parts at tick 1,750 | equal on all 3,698 ticks |
| comes apart at | 3,699 | 2,624 | 3,699 (same state) |
| maximum resident / scratch | 130 MB / 8.457 MB | — | 123 MB / 8.460 MB |

The physics suite matches the baseline (250 pass; only
`a_box_dropped_on_a_resting_box…` fails, as at `a2cd0d0`), and the ledge
test's outcome is the baseline's by construction.

**After the treads merge (`ff3faf3`).** Remeasured on the merged head against
the same `a2cd0d0` binary, three interleaved runs each, load 2.5–9. The state
hash still equals the baseline's on every tick, including 22 ticks past the
breakup.

| Builder replay | Baseline `a2cd0d0` | Exact rows on `ff3faf3` |
|---|---|---|
| 600 ticks: tick median | 32.9–33.3 ms | 25.9–26.3 ms (−21 %) |
| 600 ticks: tick p95 | 46.8–47.2 ms | 36.2–36.7 ms (−22 %) |
| 600 ticks: CPU cycles | 62.9–63.2 G | 49.8–50.1 G (−21 %) |
| 3,698 ticks: tick median | 32.4–32.6 ms | 26.2–26.6 ms (−19 %) |
| 3,698 ticks: tick p95 | 46.1–46.5 ms | 36.9–37.2 ms (−20 %) |
| 3,698 ticks: CPU cycles | 381–384 G | 306–308 G (−20 %) |

The longer run includes the late ticks, where the build is coming apart and
contact rows are a smaller share of each tick.

**Unknown save fields.** `stored_water_with_an_unknown_field_keeps_what_this_build_knows`
loads a water save with an unrecognised 50,000-entry list and checks the
known fields survive. It takes 0.12 s with the vendored ron and 72 s against
unpatched ron 0.12.2.

## Final serial comparison

Run one at a time, baseline and current interleaved, with the load average
logged before every run. Cycle counts measure work done; elapsed time
measures what a player waits for, and only the builder runs had quiet enough
conditions to compare it.

| Workload | Runs | Load before runs | CPU cycles | Elapsed |
|---|---|---|---|---|
| Builder replay, 600 ticks, exact rows (current) | 2 + 2 | 2.6–9 | 62.5–62.9 G → 49.7 G (−21 %) | tick median 33.0–33.4 → 25.9–26.1 ms (−22 %); p95 46.7–46.9 → 36.4–36.6 ms (−22 %); mean 31.5 → 24.8 ms (−21 %) |
| Builder replay, PR #67 shared basis (superseded) | 3 + 3 | 6.7–7.9 | 63.4–64.5 G → 44.4–45.8 G (−29.5 %) | tick median 33.4–34.6 → 21.6–22.6 ms (−35 %); p95 48.4–49.0 → 32.0–33.2 ms (−33 %) |
| `terrain-cut`, titan_crags | 3 + 3 | 6.7–108 | 593–598 G → 479–483 G (−19.4 %) | not comparable: 25–41 s either way under shifting load |
| `water-breach` | 2 + 2 | 32–120 | 455–461 G → 338–344 G (−25.6 %) | step p50 7.2–7.3 → 5.8–6.1 ms (−18 %); p95s load-dominated |
| `terrain-cut`, nine biomes | 1 + 1 | 16 / 177 | 4,222 G → 3,439 G (−18.5 %) | not comparable |

History, PR #67: after the lone-body change, two further interleaved builder runs per build
(load 7.5–18): baseline 63.7–63.9 G cycles, tick median 33.6–33.9 ms; before
the change 44.9–45.1 G, 21.95–22.03 ms; after it 44.87–44.91 G, 21.82–21.90 ms.
All six runs ended on the same state hash.

Peak memory in the same runs (maximum resident / peak footprint):

| Workload | Baseline | Current |
|---|---|---|
| Builder replay, exact rows | 130 / 126 MB (one run) | 123 / 120 MB (one run) |
| Builder replay, PR #67 shared basis | 114–125 / 110–122 MB | 123–124 / 119–120 MB |
| titan_crags | 117–129 / 127–128 MB | 106–133 / 122–135 MB |
| `water-breach` | 138–141 / 149–155 MB | 140–143 / 155 MB |
| nine biomes | 129 / 128 MB | 137 / 140 MB |

Builder and water memory are the same within run-to-run spread. The terrain
cut's peak depends on how its ten workers' jobs overlap; with the current
build finishing jobs faster, more can be in flight at once, and the single
nine-biome pair differs by 8–12 MB. That is one pair under very different
load and is not taken as a trend.

Quality in the same runs: the builder's final state hash after 600 ticks is
identical in all six runs (`9819739108337549482`), with 2,652 median contacts
and no degraded ticks; water's ledger error (2.6e-12 m³) and eroded quanta
(419,190) are identical; the terrain cuts have identical node counts, and
their meshes differ only by fix 9's transition faces (titan_crags digest
`c1d491861a604221` → `c1a23420bcb0358c`, the same as the selection-only
variant with the fix, and unchanged by every other change).

## Baselines and results

| Workload | Measure | Before | After |
|---|---|---|---|
| `cpu-physics --scenario builder-scale --copies 1`, 600 ticks (13 bodies, 1,744 colliders, ~2,650 contacts) | CPU cycles | 62.5–62.9 G | 49.7 G (−21 %) |
| same | tick median / p95 | 33.0–33.4 / 46.7–46.9 ms | 25.9–26.1 / 36.4–36.6 ms |
| same | contact rows / query, median | 16.1 / 16.9 ms | 11.4 / 13.4 ms |
| same | contacts, degraded ticks, state hash | 2,652, 0 | 2,652, 0, identical |
| `terrain-cut --seed 42`, all nine biomes (108,408 nodes) | CPU cycles | 4,334 G | 3,503 G (−19 %) |
| same | instructions | 17,994 G | 14,354 G (−20 %) |
| same | sampling / extraction CPU | 5,773 / 1,066 s | 4,062 / 537 s |
| same | cold selection, summed over biomes | 14.3 s | 7.2 s |
| `terrain-cut --biome titan_crags` | CPU cycles | 603 G | 487–495 G (−18 %) |
| `water-breach` (120 s of flooding) | CPU cycles | 449 G | 333 G (−26 %) |
| same | remeasure / sediment p95 | 19.4 / 12.9 ms | 15.5 / 10.7 ms |
| `mechanic-bench --scenario terrain_dig` | cached selection p95 | 27–33 ms | 17–18 ms |
| same | terrain stage p50 | 29–32 ms | 19 ms |

### Builder replay by phase (PR #67 shared basis, superseded)

Full replays of the same fixture, same flags, same hardware, two interleaved
repetitions per build, stopping when any contact overlaps by more than 10 cm
(the machine coming apart). Medians and p95 in ms, ranges over the two
repetitions:

| Ticks | Baseline median / p95 | Current median / p95 | State |
|---|---|---|---|
| 1–600 | 34.6–36.2 / 49.5–110.0 | 22.7–23.5 / 33.9–34.7 | bit-identical |
| 601–1,749 | 29.5–33.6 / 47.1–50.9 | 23.3–23.6 / 32.4–33.2 | bit-identical |
| 1,750–2,999 | 36.6 / 49.6–49.9 | 25.3–25.4 / 33.7–33.8 | trajectories differ |
| 3,000–end | 32.5 / 47.1–47.8 (degraded) | — | — |
| comes apart at | tick 3,699 | tick 2,624 | |

The two builds keep the same state hash until tick 1,749. At 1,750 a contact
row rounds differently and the replays part; from then on they are different
simulations of a chaotic machine. The current build's replay comes apart at
2,624 (its last tick took 2.4 s), the baseline's at 3,699 after 699 ticks
degraded by the 500 rad/s speed limit. Across eight further starts raised by
1–8 nm, both builds stayed bit-identical with each other and came apart on
the same tick or not at all: 3,388, never, 2,649, 3,720, never, never, never,
3,401. The current build's 2,624 sits beside the baseline's own 2,649, so
this is read as the existing instability rather than a regression; it is not
proof of equal long-run behaviour.

Per-tick components over ticks 1–600, as means (medians do not add up,
because ticks alternate between about 650 and 2,700 contacts):

| Mean ms | Baseline | Current |
|---|---|---|
| tick | 33.1 | 23.6 |
| contact rows | 11.5 (35 %) | 5.2 (22 %) |
| collision query | 18.3 (55 %) | 15.0 (64 %) |
| constraints, continuous | 2.1, 0.8 | 2.1, 0.8 |

The 30 % share the first profile gave contact rows is its self time on one
inlined line. The `rows_ms` timer also covers closure, mesh and joint rows,
and its median (16.1 ms) sits on the high-contact ticks; its mean is 11.5 ms,
35 % of the mean tick. Rows saved 6.3 ms and the query 3.3 ms of the 9.5 ms
mean saving.

Memory over the same 2,600 ticks, before either replay comes apart: maximum
resident 107–108 MB baseline, 102–106 MB current; peak footprint 126–144 MB
against 143–144 MB, overlapping between repetitions of one build; solver
scratch 8.528 MB against 8.545 MB, the 16.6 KB being the per-body response
buffers. CPU cycles 279–285 G against 198–199 G (−29 %).

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

1. **Contact rows share each body's responses** (superseded by the exact
   rows above; kept here as the record of what PR #67 measured). Every contact row is a
   combination of six spatial unit impulses at its body's centre of mass, so a
   substep now solves those six per touched body
   (`MachineKinematics::body_responses`) and combines them, instead of running
   the articulated factor once per row: about 32,000 solves per builder tick
   become about 300. The combined responses match direct solves to 1e-9
   relative on every row checked, and usually bit for bit: the builder replay
   keeps the baseline's state hash for its first 1,749 ticks, and eight
   nanometre-jittered 4,200-tick replays matched the baseline throughout.
   Contact rows fell 16.3 → 7.5 ms. A contact whose bodies are all lone
   free bodies (a component of six velocities) keeps the direct solve: its
   6×6 block costs about as much as combining six bases, and it keeps such
   scenes bit-identical to the baseline. The builder is one articulated
   component and is unaffected; its state hash is unchanged.
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
   cut this adds 9,294 transition triangles and no nodes, and 72,317 over the
   nine biomes (+0.08 %). Three tests pin it, and all three fail with the old
   range: a level-5 node beside a level-6 node stitches to it; balancing
   splits a level-6 node two levels coarser than its neighbour; and every
   face of a real cut toward a one-level-coarser neighbour, level 6 included,
   carries a transition.

### Ron: opening saves with fields this build does not know

10. **Vendored ron 0.12.2 with two unreleased upstream fixes** (#608, #610).
    Serde skips a field it does not recognise through `deserialize_any`, and
    ron's untyped number parser searched the whole rest of the document for
    `..` on every number. `water11`, saved by a build that also stores grass,
    carries a 7.3 MB `grass` list this branch does not declare. Neither the
    baseline nor the current app had opened it after six minutes; the main
    thread was inside that search. Parsing the same 36.4 MB `water.ron` now
    takes 0.52 s. Copies of it with the unknown list cut to 0.25, 0.5 and
    1 MB took 1.1, 3.2 and 11.5 s before (quadratic) and 0.39–0.40 s after.
    Typed fields never took this path, so documents without unknown fields
    parse as before.

## Evidence for each change

| Change | Evidence | Kind |
|---|---|---|
| 1 contact rows | combined vs direct response ≤ 1e-9 relative on every row; builder state hash equal to tick 1,749; 8 jittered 4,200-tick replays equal throughout; physics suite | tolerance, then statistical |
| 2 Fx separation cache, one-key sort | builder state hash equal over 300 ticks; the existing candidate-pair test compares the list with a brute-force lexicographic list, so order and the absence of duplicates are pinned | bit-identical |
| 3 fits by hash, shape buffers | builder state hash equal over 300 ticks | bit-identical |
| 4–6 tape arguments, lanes, mesher maps | `mesh_digest` of 108,408 nodes over nine biomes; edited-ground digest below | bit-identical |
| 7 lattice columns cached | `mesh_digest`; edited-ground digest; `water-breach` ledger error and eroded quanta identical | bit-identical |
| 8 hashed cut membership | node lists and `mesh_digest` identical with and without it | bit-identical |
| 9 level-6 owners | node counts unchanged, digest differs by the new transition faces | intended change |
| 10 ron | parsed documents compared by sheet count; upstream's own regression tests | upstream fix |

**Edited ground, boundaries and materials.** A temporary tool dug two
spheres and added a sand and a rock sphere at four places in each of two
seeds, then meshed every node of levels 0–3 in a 3×3×3 block around each
edit with four transition masks: none, all six faces, −X only, +Z only. That
is 432 chunks per place, 3,456 in all, 4.77 million triangles. The digest
covers positions, normals, material weights, surfaces, compaction and final
indices, and matched the baseline at all eight places. The current build took
23 % fewer cycles.

**Empty nodes.** Plan item 4, culling provably empty nodes before sampling,
was not implemented: the existing coarse `lattice_is_clear` check already
skips the clearest cases, and the remaining empty jobs are ones bounds cannot
prove empty. The evaluator changes cover empty and non-empty jobs alike; the
nine-biome digest includes the 50–75 % of jobs that end empty, and the
non-empty share is where most of the saving lands (sampling 5,773 → 4,062
CPU-s).

## Audited with no change made

- **GPU physics.** On `test2_car` the bench's main thread spends 82 % of its
  time waiting for the GPU and 12 % encoding (about 0.9 ms a tick, as the
  bench's own encoding p95 says). The LBVH sort's 256 floor is a single local
  workgroup sort with no global passes, so lowering it saves nothing. Kernel
  time sits in the contact solver; changing it means changing the solver.
- **Construction edits.** `edit-latency`: a block's boundary query costs
  2.5 µs, a pipe junction's 0.34 ms, cached queries 0.03–0.05 µs.
- **UI.** Every panel compares its snapshot with the last one pushed and
  skips re-rendering when nothing changed; no per-frame rebuild was found.
- **Loose material.** `material-clumps` p95 2.0 ms for 256 awake bodies.
- **Mesh BVH.** Already built with `select_nth_unstable`.
- **Water stepping** runs on a worker, never on the frame.

## Rejected after measurement

- Remembering the face that last separated a collider pair and testing it
  first: no measurable change.
- Reusing projections for exactly opposite faces in the separating-axis test:
  1 % slower (the per-call table cost more than it saved).
- Four independent accumulators in the vertex projection: 17 % slower.
- Batching each scatter instance's shape across a grid row: at most 2.5 %,
  within noise on two of three biomes, not worth the code.

## Correctness checks

The branch is **not** all green, and neither is `main`. Locally,
`mechanic-physics` fails one test and `mechanic-gpu` eleven, all identically
at `a2cd0d0`. In CI, where `main` at `0c056ec` also fails, the extra failures
are environmental: the Linux runner has no GPU adapter (40 GPU tests and one
app test), and Windows' DX12 shader compiler rejects a GPU kernel (91 GPU
tests). The figures below are from before merging `main` and before the
lone-body change, which fixed the ledge test.

**CI on the branch after merging `main` (`31d92e7`) fails exactly the tests
`main` fails at `0c056ec`:**

| Runner | App | GPU | Physics |
|---|---|---|---|
| ubuntu-latest | 1: `an_inherited_cut…` (no GPU adapter) | 40 (no GPU adapter) | pass |
| macos-latest | pass | 11 | 1: `a_box_dropped_on_a_resting_box…` |
| windows-latest | 4: DX12 shader compiler rejects a GPU kernel | 91: same | 2: box, and the ledge test |

The ledge test fails on Windows in `main`'s CI too, and at `a2cd0d0` on a
throwaway branch run (`ao/mechanic-3/ci-baseline`). On loose blocks the
branch is now bit-identical to `main`, so it fails there the same way.

`cargo xtask test` on the branch, Apple M1 Pro / Metal:

- `mechanic-core` 400 pass; `mechanic-world` 275 pass (272, plus the three
  level-6 tests added after the full run); `mechanic-app` 947
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

`captured_blocks_dropped_across_a_ledge_come_to_rest` failed on the branch
until loose bodies went back to the direct solve (change 1, last paragraph).
It now passes, and the drop's state hash matches the baseline on every one
of its 600 ticks.

Before that, the history was: both builds reached near rest by tick 100; the
baseline stayed there (tick 600: 21 contacts, 1.60 mm, fastest speed 0.0013),
while with shared bases the top block tipped off the edge at tick 106 and
fell into a contact limit cycle (tick 600: 20 contacts, 38.9 mm, 0.37). The
only numerical difference was summation order: 26–28 % of response entries
differed by at most 6.5e-16 relative (about three units in the last place)
from tick 1. Over 64 nudged starts the failure rates were the same within
noise:

| Fixture | Build | Overlap ≥ 5 mm at tick 600 | Speed ≥ 0.2 | Median / p90 / max overlap |
|---|---|---|---|---|
| ledge | baseline | 30 / 64 | 21 / 64 | 4.9 / 37.5 / 38.6 mm |
| ledge | shared bases | 31 / 64 | 22 / 64 | 4.8 / 6.7 / 38.4 mm |
| leaning | baseline | 3 / 64 | 3 / 64 | 2.9 / 3.6 / 36.7 mm |
| leaning | shared bases | 3 / 64 | 5 / 64 | 2.9 / 3.7 / 36.7 mm |

The test passes only because its fixed start lands on a calm outcome. Half
of nearby starts fail in every build, and `main`'s own CI fails it on
Windows. The underlying defect remains: a block toppled off an edge never
settles, its contact with the block below alternating between a ten-point
face manifold and a single edge point with about 3.7 cm overlap. That is the
"box hanging over a ledge" case `EDGE_FACE_ALIGNMENT` addresses, failing once
the tilt passes its 2.6° window, and a follow-up for the contact model.

### Long builder runs

See "Builder replay by phase". Both builds' replays eventually come apart:
continuous broadphase then finds millions of candidate pairs and ticks take
seconds. Before that the baseline degrades ("speed limit", a body past
500 rad/s) from tick 3,000 in most starts. This matches the known spin-gain
flakiness and predates the pass; it is a physics fix rather than a
performance one.

## In the app

Background captures (`scripts/run-background-capture.py --from-start`) of a
copy of the player's `water11` world with its unknown `grass` list removed,
so that the baseline can open it too: 4112×2524, 4× MSAA, unfocused window,
60 s from world entry including streaming. Two runs per build, interleaved.
These are diagnostics, not acceptance runs: a world with no creation never
starts the simulation, so the settled-capture gate cannot apply, and the
unfocused window's presentation pacing sets the frame rate.

| | Baseline | Current |
|---|---|---|
| frame p50 / p95 | 50.4–51.3 / 65.8–67.2 ms | 52.6–58.2 / 64.6–78.3 ms |
| FPS | 12.9–13.0 | 12.5–13.7 |
| frames with terrain streaming busy | 808–811 | 705–722 |
| opaque GPU p50 | 30.6–31.1 ms | 31.5–31.8 ms |
| transparent GPU p50 | 39.9–40.7 ms | 41.8–43.2 ms |

Streaming settles sooner, as the headless meshing results predict. Frame and
GPU times overlap between builds except for one current run that other load
visibly disturbed (966 frames instead of about 1,050, render CPU p95 doubled).
Nothing on the render side changed apart from fix 9's seam triangles, so no
frame-time gain or loss is claimed.

The transparent pass is the water surface: with `MECHANIC_WATER=off` it
disappears, yet background FPS only rises from 13.7 to 14.7, so in these
captures the GPU passes overlap with presentation waits (acquire p50 about
38 ms) rather than adding up. Attributing a foreground frame needs a focused
capture, which would take over the user's screen, so none was run.

The original `water11`, with its `grass` list, loads in the current app; the
window appears 17 s after launch and the world reaches play with its 87 local
terrain nodes resolved. Neither app had opened it after six minutes before
change 10.

The builder fixture world cannot be captured: its saved player pose finds no
terrain collision under the current generator, so loading never ends, in
either build.

## Remaining opportunities

- The CPU collision query is still about 15 ms on the builder: convex
  separation (21 %), manifold clipping (14 %), pair traversal (13 %). The
  proximity query's pair traversal is about 4.4 ms a tick over ~60,000
  candidate pairs, nearly all of them neighbouring blocks of adjacent bodies
  within the speculative margin. (Reusing it for the buried-vertex query was
  measured and rejected: that traversal costs 0.34 ms.)
- Saved floats are written as full decimal expansions (`0.000…0128…` for
  1.3e-30), which is much of why `water.ron` is 36 MB.
- Water's ground lookup still evaluates density cell by cell down a column
  and misses the 1,024-slot column cache across a ~20,000-cell sheet. A larger
  or water-owned cache would cut it, at several MB per thread.
- Per-column point evaluation in `CompiledWorld::column` could use grid
  evaluation across a lattice's columns.
- The water surface's transparent pass is about 40 ms of GPU time at
  4112×2524 with 4× MSAA in background captures; a focused capture would
  show how much of it reaches frame time.
- The ledge limit cycle and the builder's spin gain (above).
