# Optimisation pass: construction pairs and contact rows

The builder replay was profiled at `31b134b`. Two offenders held about two
thirds of every CPU physics tick. Both are now faster, and every workload's
output is bit-identical to the baseline.

## Conditions

- Apple M1 Pro, 10 cores, macOS. `profiling` profile (release with debug info).
- Baseline: `31b134b` (`main`). Candidate: this branch, version 0.5.2.
- Runs were interleaved one at a time, baseline first. Load average before
  each run was 2.9–3.9. Cycles and instructions come from `/usr/bin/time -l`.
- Profiles: macOS `sample` at 1 ms.

## The two offenders

Builder replay, 600 ticks, baseline profile:

| Share of samples | Where |
|---|---|
| ~35 % | Construction pairs: dual-tree candidates, a comparison sort, then a separating-axis test (SAT) per pair (`query_groups`) |
| ~30 % | Contact rows: one articulated solve per row through all 13 bodies (`substep`, inlined `Contact::rows`) |
| ~35 % | Everything else: constraint passes, manifold clipping, continuous sweep |

Counting pairs explained the first offender. Over 600 ticks, 33.3 M pair
separations were computed and 98 % returned "separated beyond reach". The
separation cache was cleared on every pose change, so each query proved the
same separations again from scratch.

The second offender is latency-bound. A row's forward sweep is a serial chain
per body (cross product → dot → divide → axpy), so a lone solve mostly waits
on its previous operation.

### 1. Construction pairs: witness faces with drift bounds

Each collider pair now remembers the face that last proved it separated, the
gap that face measured, and how far the two bodies had moved relative to each
other by then. Moving bodies can shrink the gap by at most their relative
travel plus a generous rounding margin, so a pair still inside that margin
skips all work. Otherwise its remembered face is tested first, and only then
does the full SAT run. The records sit in per-collider lists that queries
walk in pair order instead of hashing. Candidate pairs are sorted by radix
rather than comparison, and manifold clipping reuses the query's scratch.

Of 36 M pair visits, 31.7 M are now settled by the drift bound alone, 0.7 M
by one face test, and 0.9 M need the full test.

### 2. Contact rows: one interleaved solve per chunk of contacts

The articulated solve for one row goes through the same sequence of
floating-point operations whether it runs alone or beside other rows. Rows
from 64 contacts at a time are now solved together, laid out with one value
per row in each slot. The body-by-body sweep runs every row's identical
scalar arithmetic side by side, so the CPU overlaps independent rows instead
of stalling on each row's chain. The backward sweep's "nothing to pass up"
skip became a per-row select, which keeps the loop branch-free with the same
result. Row storage was rewritten as plain loops in the same summation order.

## Results

| Workload | Measure | Baseline | Optimised | Change |
|---|---|---|---|---|
| Builder replay, 1 copy, 600 ticks (3 + 3 runs) | CPU cycles | 50.07–50.10 G | 35.66–35.73 G | −28.8 % |
| | instructions | 205.5–205.7 G | 150.6–150.7 G | −26.7 % |
| | wall time | 15.27–15.29 s | 10.76–10.78 s | −29.5 % |
| | tick mean / median | 25.0 / 25.9–26.2 ms | 17.5 / 18.8–19.2 ms | −30 % / −27 % |
| | tick p95 | 36.4–37.0 ms | 23.8–24.1 ms | −35 % |
| | collision query, mean | 13.9 ms | 8.2 ms | −41 % |
| | contact rows, mean | 8.1 ms | 6.3 ms | −22 % |
| | peak resident memory | 123–131 MB | 112–113 MB | −10 % |
| Builder replay, full 3,698 ticks | CPU cycles | 306.8 G | 206.0 G | −32.8 % |
| | tick mean / p95 | 25.7 / 37.2 ms | 17.2 / 23.9 ms | −33 % / −36 % |
| Builder, 10 copies, 60 + 240 ticks | CPU cycles | 172.2 G | 135.2 G | −21.5 % |
| | tick mean / p95 | 175.2 / 373.3 ms | 135.8 / 269.7 ms | −22 % / −28 % |
| Builder, 10 copies connected, 60 + 240 ticks | CPU cycles | 345.0 G | 265.4 G | −23.1 % |
| | tick mean / p95 | 354.2 / 394.9 ms | 271.4 / 295.5 ms | −23 % / −25 % |
| | contact rows, mean | 204.5 ms | 147.1 ms | −28 % |
| `block-pile` | CPU cycles | 3.03 G | 2.79 G | −8.0 % |
| `fast-motion` | CPU cycles | 16.35 G | 15.33 G | −6.3 % |
| `four-bar` | CPU cycles | 227 M | 207 M | −8.7 % |
| `car-drive`, `gear-train`, `wheel-roll`, `car-drop`, `fast-impacts` | CPU cycles | 112–316 M | 110–308 M | −2 % to −4.5 % |
| `reference-fixtures` (exact reference solver) | CPU cycles | 24.47 G | 24.50 G | unchanged |

Constraint passes and the continuous sweep were not touched and measure the
same, as expected. GPU physics, world generation and the app's rendering
paths share no changed code.

The small scenes are single runs that last well under a second, so their
few percent are close to noise.

## Correctness

Each change preserves the arithmetic exactly; nothing was tuned or relaxed.

- **Every reported field is identical.** All non-timing JSONL fields match
  the baseline on every tick: the full 3,698-tick builder replay, 1 copy,
  10 copies, 10 connected copies, and all ten other `cpu-physics` scenes.
  The fields include the state hash, contact counts, penetration, degraded
  ticks, pair candidates and hierarchy node tests.
- **The state hash alone was not enough.** In the first 600 builder ticks no
  contact row carries an impulse. A deliberately mismatched build still
  matched the hash over that window. Two temporary in-process checks were
  therefore run over the full 3,698-tick replay:
  - each of 216 M drift-bound skips was re-checked with the full SAT, which
    agreed every time;
  - each of 109 M contact rows was compared bit for bit (mass, Jacobian,
    response, separation) against the original per-contact solve and the
    original iterator code.
- **New regression tests:**
  - `lanes_solve_each_side_bit_for_bit_as_a_lone_solve` covers floating,
    anchored, reversed and branched mechanisms, including zero, sparse and
    non-finite sides;
  - `remembered_separations_find_every_contact_a_fresh_query_finds` checks
    that a cache reused across moving poses finds exactly a fresh cache's
    contacts;
  - `a_named_separating_face_answers_exactly_as_the_full_test` covers the
    witnessed SAT;
  - `radix_sorted_pairs_match_a_comparison_sort` covers the pair sort;
  - the dual-tree test now also pins the traversal's test count and output
    order.
- **Suites:**
  - `mechanic-core`: 420 pass.
  - `mechanic-physics`: 258 pass. One fails, `a_box_dropped_on_a_resting_box…`,
    which also fails on clean `main`.
  - `mechanic-app`: 981 pass, 22 ignored.
  - `mechanic-bench`: passes.
  - Clippy with warnings denied, rustfmt, `xtask consistency` and rustdoc
    are clean.

## Diagnostics that changed

`solver_scratch_bytes` now includes the retained chunk buffers, a few hundred
kilobytes for the builder.

## What remains

Shares of the optimised builder tick's samples:

| Share | Where |
|---|---|
| 15 % | The per-pair cache walk |
| 15 % | Manifold clipping for touching pairs |
| 13 % | Candidate generation |
| 13 % | Gauss–Seidel passes |
| 6 % | Contact Jacobians |
| 5 % | Row storage |

The passes still read rows through per-row heap vectors, and a contiguous
row arena is the obvious next layout change.
