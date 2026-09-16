# CPU fast-motion collision work

## Change

CPU sweeps now intersect the general articulated envelope with directional
translation bounds or a full fixed-axis rotation envelope. The rotation envelope
uses the actual pivot and encloses every turn, including off-centre bearings and
a linearly translating parent. Moving ancestor axes retain the general bound.
The sweep computes collider bounds once, rejects individual triangles within BVH
leaves, and avoids updating transformed-shape caches when no terrain or body pair
is reachable.

An empty contact set can be reused within a tick that has activated continuous
sweeps, and only after a broadphase proof clears an expanded region for **all**
colliders. Each reuse reconstructs current
child poses and checks containment, contact margins, origin, topology generation,
and terrain publication generation. Escaping the region restores ordinary contact
queries. The proof is discarded after the tick. Continuous collision checks still
run at the original threshold. There are no timestep, impact-policy, tolerance,
geometry, speed-limit, scheduler, save-format, or GPU changes.

App captures now include `requeries`, `empty_contact_reuses`, `continuous_sweeps`,
and `continuous_hits`. Clearance-proof work is included in `query_ms`.

## Reproduction protocol

```sh
cargo build --release -p mechanic-bench --bin cpu-physics
target/release/cpu-physics --scenario fast-motion --warmup 120 --ticks 600
python3 scripts/run-cpu-motion-benchmark.py --baseline /path/to/baseline \
  --candidate /path/to/candidate --output /path/to/new-results
```

`manifest.json` identifies baseline source, executable hashes and measurement
protocol. The baseline uses unmodified HEAD physics plus the benchmark harness
and the same diagnostic struct layout (its reuse counter remains zero). It is
built from an isolated source copy. Three pairs alternate execution order; no
profiler or builds run during timing. All timings are wall-clock tick duration.
Query and continuous times in the JSONL are totals over measured ticks. The
candidate query total also includes clearance proofs and requery pose reconstruction.
`matched-before-query-gate/` preserves the first comparison: a repeatable 3–6%
regression below the sweep threshold led to restricting clearance-proof setup to
ticks with active continuous sweeps. The final short comparison is in `matched/`; the longer follow-up is in
`matched-long/`.

The generated fixtures are a free bearing rotor on a welded static base, a freely
translating cube, and a four-wheel driven vehicle. Rotor and translation cases
run without gravity at identical geometry, with 0 or 32 unrelated static bodies.
Their speed matrix contains rest, slow motion, 99% and 101% of the solver's actual
5 cm point-travel threshold, and fast motion. Threshold calculation includes the
bearing-to-body-origin lever arm. A large finite floor keeps translations over
the same terrain throughout the run. The driven vehicle uses gravity and three
motor targets; its cuboid wheels deliberately retain ordinary finite contact
geometry.

## Headless results

Median of three runs, 120 warmup plus 3,000 measured ticks per case. Times in ms.
Rotor speed is rad/s, translation speed is m/s, and vehicle speed is the motor
target in rad/s; actual vehicle motion depends on contact and drive effort.

| Case | Speed | Baseline p50 | Candidate p50 | Baseline p95 | Candidate p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| rotor + 32 static bodies | 0.000 | 0.019292 | 0.019417 | 0.020167 | 0.020583 |
| rotor + 32 static bodies | 1.000 | 0.019500 | 0.019708 | 0.020750 | 0.021000 |
| rotor + 32 static bodies | 8.697 | 0.024750 | 0.025000 | 0.026334 | 0.026583 |
| rotor + 32 static bodies | 8.872 | 0.048958 | 0.044541 | 0.052792 | 0.048625 |
| rotor + 32 static bodies | 500.000 | 0.138292 | 0.044500 | 0.149167 | 0.048125 |
| translation + 0 static bodies | 500.000 | 0.010292 | 0.006542 | 0.010875 | 0.006916 |
| translation + 32 static bodies | 500.000 | 0.052458 | 0.043916 | 0.056458 | 0.047041 |
| vehicle + 0 static bodies | 0.000 | 0.199542 | 0.199625 | 0.220792 | 0.221458 |
| vehicle + 0 static bodies | 1.000 | 0.259916 | 0.260041 | 0.317875 | 0.315958 |
| vehicle + 0 static bodies | 100.000 | 0.293875 | 0.293958 | 0.350375 | 0.350125 |

The fast rotor with 32 unrelated bodies improves p50 by 67.8% and p95 by
67.7%. In the first long pair, continuous candidates fall from 24,000 to zero,
while all 12,000 sweeps still execute. The 9,000 repeated contact queries become
9,000 validated empty-region reuses. Continuous time falls from 327.9 ms to
61.3 ms total; query/proof time falls from 58.4 ms to 43.3 ms total.

The fast translating cube improves p50 by 36.4% alone and 16.3% with unrelated
bodies. Every final short and long pair has identical final state hashes and
zero degraded ticks across all 23 cases. Nine fast-impact quality cases report
no tunnelling. Driving, drop, pile, and four-bar quality cases have no degraded ticks.

The initial repeatable 3–6% below-threshold regression is removed. Final vehicle
medians differ by less than 0.1% in the long batch. Slow rotor/translation medians
remain near parity (roughly 0–2% variation); small differences change sign between
the short and long batches. These runs establish the large fast-motion reduction,
but do not establish exact equality of slow-case cost. Quality-file timings are
diagnostic only: compilation overlapped those physical checks.

## Verification

The CPU regression suite includes full-turn impacts between matching endpoint
orientations, fast impacts, saved-car driving, suspension, and body contacts.
Added coverage samples the complete conservative envelope for directional motion,
multiple root turns, off-centre bearings, moving parents, and suspension; checks
clearance invalidation after terrain/topology changes and approaching bodies;
and verifies a fast rotor retains speed while its static base remains fixed.
Two existing clearance tests now assert physical clearance without requiring
narrowphase evaluations that broadphase rejection can eliminate.

## App verification

Final release build on Apple M1 Pro / Metal, 4112 × 2524, MSAA 4, AutoNoVsync,
with F3 visible. Source hashes were checked before and after the build.

The fixed-camera idle builder capture measured **27.5 FPS / 60.0 completed TPS**,
CPU tick p50/p95 **0.844 / 1.028 ms**, and zero degraded ticks. That fixture has no
linked input seat, so its requested driving check correctly failed; it is retained
only as an idle measurement in `foreground-native/`.

For driving, a disposable copy of the same terrain used the existing linked
`driven_car_instance.ron` fixture (`driving-fixture.json` records the exact recipe).
The capture passed all foreground/route/driving/drain checks: **31.7 FPS / 60.0 TPS**,
3,599 scripted inputs for 3,599 submitted ticks, 1,903 focused frames, and **zero
degraded ticks**. CPU tick p50/p95 was **1.579 / 6.475 ms**. Continuous collision
work remained the largest CPU stage: p50/p95 **0.930 / 5.196 ms**, with 13,646
sweeps, 11 hits, 10,241 re-queries, and no empty-region reuses in this contact-heavy
run. These are remaining costs, not evidence of a free-motion cache failure.

Automation suppresses manual camera input, but driving seats the player and the
view follows the vehicle. Thus only the idle capture is fixed in world space.
These app captures verify route, throughput, and diagnostics; **no before/after
FPS gain or fixed-world-camera driving comparison is claimed**. The final driving
screenshot also contains ongoing terrain streaming and an incomplete GPU timing
sample, so it is not a GPU scale-gate result.

The sandbox launch could not access a GPU; native launches succeeded. Captures
used disposable world copies and did not edit the source fixtures. Raw JSONL is
archived losslessly as `.jsonl.gz`; summaries and full-resolution screenshots
remain alongside it.

## Remaining costs and limits

Bounds construction, articulation reconstruction, and broadphase traversal still
scale with scene size. General articulated paths still use the original looser
bounds when their axes move. The empty-space certificate is conservative: terrain
chunk bounds or other bodies can prevent reuse even when a narrowphase query
would find no contacts. Contact-heavy driving still pays dynamics, contact-row,
and constraint-solver costs. Headless tick timings alone do not establish an FPS
gain or any README scale gate.
