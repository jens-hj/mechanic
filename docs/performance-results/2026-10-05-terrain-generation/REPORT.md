# Optimisation pass: terrain generation

Issue [#82](https://github.com/jens-hj/mechanic/issues/82). Terrain generation
is the heaviest CPU workload in the repository. Its two biggest offenders are
now about a third cheaper, and every output is bit-identical to the baseline.

## Conditions

- Apple M1 Pro, 10 cores, 16 GB, macOS. `profiling` profile (release with
  debug info).
- Baseline is `1f120a3`, v0.6.6. The candidate is `be79cbb`, v0.6.7. Both were
  built through the managed Cargo storage launcher.
- Each workload was run as a base/candidate pair, alternating and one run at
  a time, with nothing else heavy on the machine. Two full rounds were run.
  The figures below are from the second round, on the committed code; the
  first round agreed within a few percent.
- Cycles and instructions come from `/usr/bin/time -l`. They are the primary
  measure because they hold up under load far better than wall time.
- Profiles were taken with macOS `sample`, at 1 ms per sample for one biome
  and 5 ms for all nine.

## Finding the offenders

The heaviest reproducible workload is `terrain-cut --seed 42`: the full
streaming cut around all nine biome hearts. It costs about 4.3 T cycles. For
comparison, the 600-tick CPU physics builder replay costs about 36 G cycles,
and its remaining cost is spread across many functions after the 2026-10-04
pass.

Shares of terrain-cut worker samples, all nine biomes:

| Share | Offender |
|---|---|
| ~32 % | **Point-at-a-time tape interpretation** (`Tape::eval`). The pieces: scattered shapes evaluated once per point and per instance inside grid blocks (12 %), lattice columns (11 %), and per-vertex surface rules and density probes (8 %). Grid blocks also ran every op as eight-point rows, so dispatch was amortised over just 8 values. |
| ~26 % | **Noise lookups** (`FastNoiseLite::gen_noise_single_2d/3d` and its wrappers) |
| ~10 % | The interval cull (`ground_interval`), itself mostly noise intervals |

An exact memo of noise samples was considered and rejected. Repeated
coordinates account for only 8 % of 2D lookups and 0.2 % of 3D lookups
(measured with a probe build). Noise had to become cheaper per lookup.

## 1. Tape interpretation: one lane engine for every batch

The expression tapes are now evaluated op by op over many points at once,
wherever many points are known together:

- **Grid blocks.** Each op runs once over its whole extent, with inputs over
  fewer axes spread to match, instead of one 8-point loop per grid row.
  Common ops are tight loops, including the arithmetic, clamp, smoothstep,
  pow, terrace, smooth unions and primitives.
- **Scattered shapes.** Inside a block, each nearby instance runs its shape
  tape over every point within its reach (`Tape::eval_many`). Points are
  processed in 128-point chunks that stay in L1. Sub-expressions that depend
  only on the instance's vars are evaluated once per instance. A
  squared-distance prefilter, with a 1e-9 relative margin, skips the exact
  `hypot` test only where that test must fail.
- **Lattice columns.** Climate, biome heights and carve roofs, floors and
  tops run together for every column a lattice misses in the column cache.
  The column is then assembled by the same code as `column()`, which is now
  split into a shared assembly step so the two paths cannot drift.
- **The cull.** Blocks are visited with y innermost, so stacked blocks share
  their x/z extent. Every x/z-only bound is computed once per extent: climate
  intervals, blend weights, river and shore bounds, carve roofs, and the
  planar ops inside each biome's density tape. One-shot callers such as
  streaming selection keep the original allocation-free path.
- **Surface rules.** Rules that read the same `Field` expression compile it
  once and evaluate it once per painted vertex. This matters in titan_crags
  and verdant_hills, where two rules read the same scatter-backed field.

Per point and per instance the order is unchanged, and every element goes
through the same scalar arithmetic, often the very same functions. The
results are therefore bit for bit those of point evaluation.

## 2. Noise: exact lane kernels

FastNoiseLite's three hot generators are reimplemented in
`generation/noise/kernels.rs`: 3D Perlin with `ImproveXZPlanes` (every smooth
3D kind), 2D `OpenSimplex2`, and 2D `OpenSimplex2S`.

- **Exactness.** Each kernel repeats the library's code statement for
  statement, with the same `f32`/`f64` types, literal constants and summation
  order. Only the branches become per-lane selects: a conditional add is
  `if cond { sum + term } else { sum }`, never `sum + 0.0`, which would flip
  a negative zero.
- **Layout.** Each kernel works on eight points at a time in
  structure-of-arrays form, so LLVM keeps the hash and interpolation
  arithmetic in NEON registers. The per-call layers (fractal and type
  dispatch, coordinate transform) disappear.
- **Wiring.** `NoiseGen::sample_many` feeds the kernels one octave at a time
  across all points, and each point still sums its octaves in order. The
  lane engine calls it, so grid blocks, scattered shapes and columns all
  benefit. Other noise kinds fall back to the library, point by point.

In isolation, 3D Perlin lookups dropped from 15.8 to 9.4 ns per point, with
every one of 1 M results bit-identical. Gather-free gradients, computed from
the hash bits instead of the table, were also tried and rejected as slower.

## Results

Second round, on the committed code. Ranges span the runs.

### Fixed-work workloads

| Workload | Measure | Baseline | Optimised | Change |
|---|---|---|---|---|
| `terrain-cut --seed 42`, all 9 biomes (2 + 2) | CPU cycles | 4,386–4,395 G | 3,059–3,093 G | **−30 %** |
| | instructions | 17,763–17,766 G | 12,124–12,130 G | −32 % |
| | wall time | 294 s | 216 s | −27 % |
| | worker sampling CPU | 2,264 s | 1,590 s | −30 % |
| `terrain-cut --biome titan_crags` (3 + 3) | CPU cycles | 743–752 G | 470–480 G | **−37 %** |
| | instructions | 3,098–3,099 G | 1,870–1,871 G | −40 % |
| | wall time | 47.7–53.2 s | 31.0–36.3 s | −33 % |
| `terrain-cut --biome verdant_hills` (3 + 3) | CPU cycles | 525–529 G | 385–387 G | **−27 %** |
| | instructions | 2,089–2,090 G | 1,489–1,490 G | −29 % |
| | wall time | 34.8–37.6 s | 27.0–28.4 s | −24 % |
| `water-breach`, 120 s of flooding (2 + 2) | CPU cycles | 371.4–371.6 G | 344.7–345.9 G | −7 % |
| | instructions | 1,590 G | 1,466 G | −8 % |
| | wall time | 115.3–115.5 s | 106.7–108.4 s | −7 % |
| `material-clumps` (1 + 1) | CPU cycles | 4.6 G | 4.6 G | unchanged: generation is negligible here |
| `worldgen-preview --views --carves` (3 + 3 across both rounds) | instructions | 21,794–21,797 G | 21,787–21,789 G | unchanged |
| | CPU cycles | 5,421–5,521 G | 5,440–5,517 G | unchanged within noise |

The preview ray-marches single points, which this pass does not batch. Its
wall time swung by ±15 % on both binaries with run order. One candidate run
stalled at 662 s of wall time for 2,464 s of user time, against 2,560 s for
its baseline pair. This is consistent with machine contention rather than
work, since instructions and cycles were flat.

### Per biome, all-biome cut (2 + 2)

| Biome | Worker sampling CPU | Worst per-level sampling p95 |
|---|---|---|
| verdant_hills | 301–305 → 204–205 s (−32 %) | 97–99 → 69–70 ms |
| dune_sea | 196–197 → 131–132 s (−33 %) | 91–93 → 68–69 ms |
| arch_steppe | 260–261 → 180–183 s (−30 %) | 93–97 → 68–70 ms |
| titan_crags | 423–426 → 267–272 s (−36 %) | 129–130 → 61 ms |
| karst_needles | 426–436 → 297–305 s (−30 %) | 86–89 → 59–60 ms |
| gyroid_reef | 240–244 → 174–178 s (−27 %) | 81–84 → 59–60 ms |
| drift_isles | 249–251 → 178 s (−29 %) | 84–87 → 64 ms |
| shelf_mire | 213–216 → 135–141 s (−36 %) | 85–90 → 65–73 ms |
| sunken_coast | 184–185 → 130–136 s (−28 %) | 80–85 → 58–64 ms |

Streaming selection is unchanged: −2.9 % in total, and within run-to-run
noise for every biome.

### Time-boxed streaming scenarios (`mechanic-bench`, 30 s + 5 s warm-up, 2 + 2)

These run for a fixed time, so the optimised build simply does more work.
Throughput and latencies are the measures to compare.

| Scenario | Measure | Baseline | Optimised | Change |
|---|---|---|---|---|
| `terrain_dig` | meshing jobs per second | 62.0–63.4 | 90.0–92.8 | **+45 %** |
| | column sampling p95 | 74.2–75.8 ms | 29.6–30.4 ms | −60 % |
| | extraction p95 | 77.2–78.4 ms | 32.4–33.3 ms | −58 % |
| | terrain stage p95 | 147–150 ms | 100–102 ms | −32 % |
| | local area resolved | 209–218 of 295 nodes; never ready | 295 of 295; ready at 1.49–1.51 s | now completes |
| | uncapped FPS | 29.9–30.7 | 35.5–36.7 | +19 % |
| `terrain_stream` | meshing jobs per second | 194–207 | 252–254 | +25 % |
| | column sampling p95 | 33.3–36.2 ms | 23.2–23.8 ms | −32 % |
| | extraction p95 | 39.0–41.4 ms | 28.1–28.8 ms | −29 % |
| | local area ready | 383–398 ms | 262–270 ms | −32 % |
| | uncapped FPS | 519–566 | 591–611 | +10 % |

### Memory

- **Fixed-work peak resident memory** grew a little: terrain-cut 144–193 →
  151–202 MB, and water-breach 149–157 → 172–183 MB. The cause is per-op
  whole-block buffers and the reusable lane scratch.
- **Time-boxed scenarios** grew more (dig 570–574 → 645–675 MB, stream
  755–769 → 825–833 MB). They hold more finished chunks at the end of the
  window, so this is not a like-for-like comparison.

## What changed shares

After the pass, all nine biomes (profiled one candidate before the final
selection fix, which does not touch these paths):

- **Point interpretation** (`Tape::eval`) fell from about 32 % to about 11 %
  of samples. What is left is per-vertex surface painting, tree seating,
  which is a serial bisection, and nested scatter lookups.
- **Noise** stays the largest single cost. Most of it is now the kernels'
  own arithmetic; the remaining library calls come from point paths and from
  noise intervals in the cull.

## Correctness

Nothing was tuned or relaxed.

- **Every output is identical.** The checks below cover both rounds and
  every run:
  - `terrain-cut` `mesh_digest`, triangles, nodes and empty jobs for all
    nine biomes, plus titan_crags and verdant_hills on their own;
  - all 74 `worldgen-preview` files (maps, relief, sections, perspective and
    carve views), compared by SHA-256;
  - every non-timing field of `water-breach` (121 lines, including the
    matter ledger) and `material-clumps` (361 lines).
- **The time-boxed scenarios differ only in progress.** Fields such as
  backlog and the largest chunk's vertex count also differ between two
  baseline runs.
- **One GPU test touches generated terrain:**
  `generated_caves_stop_wall_and_ceiling_impacts`. It fails at clean `HEAD`
  and on this branch with the same penetration value, 0.005107999.
- **New regression tests** compare each batched path bit for bit with the
  path it replaces:
  - `many_points_evaluate_exactly_as_one_at_a_time` covers `eval_many`
    against `eval`. It spans every op family, shared and per-point lanes, and
    multi-chunk counts.
  - `many_points_sample_exactly_as_one_at_a_time` in `scatter` covers
    batched scatter against `sample_among` and `sample`, with rock depth
    both per point and shared.
  - `many_points_sample_exactly_as_one_at_a_time` in `noise` covers
    `sample_many` against `sample` for every noise kind, dimension and
    fractal.
  - `kernels_match_fastnoise_lite_bit_for_bit` checks the kernels against
    the library itself: about 16 k probes per seed, including lattice lines,
    nudged integers and far coordinates, across four seeds.
  - `many_columns_match_one_at_a_time_bit_for_bit` covers `columns_many` and
    `lattice_columns` against `column`. It includes biome borders, rivers,
    carve layers and the world's edge.
  - `stacked_boxes_bound_the_ground_as_fresh_ones_do` covers the cached cull
    against the one-shot cull, including switches between extents.
  - `rules_reading_one_field_share_it_and_paint_as_if_each_read_it` covers
    the shared surface fields.
- **Suites:**
  - `mechanic-world`: 345 pass.
  - `mechanic-app`: 987 pass, 23 ignored.
  - `mechanic-core` and `mechanic-bench`: pass.
  - `mechanic-physics`: 258 pass. `a_box_dropped_on_a_resting_box…` fails,
    as it does on clean `main`.
  - `mechanic-gpu`: 101 pass and 11 fail. These are the known Metal
    failures at clean `HEAD` on this machine.
  - `xtask` lint (warnings denied), fmt, consistency and doc are clean.

## What remains

- Point paths: per-vertex surface painting and density probes in the mesher,
  and tree seating. These would need the mesher to batch its vertices.
- Noise intervals in the cull, at 5–7 % of samples. A kernel call could
  answer an octave's centre and Taylor probes together, but the expected
  gain is about 2 %.
- Noise kinds without a kernel (`Value`, `Cells`, `CellEdges`, 2D `Perlin`)
  still go through the library one point at a time.
