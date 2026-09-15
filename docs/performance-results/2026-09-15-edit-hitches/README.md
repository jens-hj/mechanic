# Construction edit hitches

## Current builder save: remaining placement and feature-drag hitches

The generation-119 save revealed a further smooth-normal generation stall and
synchronous CPU collision preparation. See [the current builder investigation](builder-normals.md)
for the fixes, preserved fixture, measurements, and verification.

## Follow-up: pipes in small worlds

The reported distinction—blocks are fast in a new world but slow in `builder`,
while pipes are always laggy—led to additional pipe-specific fixes:

- The identity-frame overlap path regenerated the target's collision boxes for
  every candidate wall box. Both decompositions now happen once per pair. A
  conservative bound on the actual collision boxes rejects distant pairs before
  the unchanged separating-axis tests. This includes the wall-box overhang past
  a pipe's ideal circular radius and works in rotated construction frames.
- Idle cylinder and branch previews rewrote mesh assets every frame. Pipe drags
  also rebuilt their mesh after unchanged pointer samples. Preview meshes now
  update only when their geometry changes. The shared block/pipe/branch/layer
  handle tracks which geometry it currently contains, including differing scales.
  A straight preview uses its candidate's actual dimensions.
- Pipe drag validation reuses its result only while the graph revision, pieces,
  and placement bounds match. Moving or editing world geometry invalidates it.

The CPU-only overlap probe runs 100 distant pair queries per shape in the
development profile. It measures this placement subroutine, not complete frame
time or GPU upload latency:

| Pair | Before | After |
| --- | ---: | ---: |
| Straight pipes | 1.715 ms | 0.312 ms |
| Pipe bends | 256.763 ms | 3.033 ms |
| Junctions | 24.048 ms | 0.789 ms |

```sh
cargo test -p mechanic-app --bin mechanic-app measure_pipe_overlap_latency \
  -- --ignored --nocapture
```

The probe has no timing assertion. All 36 pipe-focused tests pass, including
rotated-frame comparisons with exhaustive collision checks, mesh reuse across
shared-handle transitions, and validation after geometry/bounds changes. App
Clippy passes with all targets and warnings denied. Original follow-up logs are
retained in `pipe-overlap-before.txt` and `pipe-overlap-after.txt`.

The follow-up broad app run passed 793 tests, with seven ignored and two explicit
exclusions: the previously documented suspension UI marker failure and the
Metal-dependent terrain publication test already passed with device access in
the first run. Formatting and whitespace checks also pass.
The follow-up release executable was rebuilt successfully after these checks.
The whole-app captures below precede this additional pipe work; the follow-up
pipe measurements are the CPU overlap probe and the preview/validation regressions.

## Causes and changes

- A construction publication created an empty CPU terrain scene. Its next update
  cloned and validated all 752 unchanged collision chunks on the main thread.
  The saved-world diagnostic captured twelve such rebuilds, taking 373–700 ms
  each. CPU terrain residency now transfers between construction generations,
  along with its publication counter, chunk generations, and origin. Construction
  colliders and contact state remain new. Subsequent terrain updates still apply
  remeshes, removals, and floating-origin changes.
- Picking, feature validation, compilation, and rendering repeatedly evaluated
  the same solid. Graph revisions now share a memo keyed by the exact base,
  ordered feature records, and construction frame. Storage retains one entry per
  owner arena slot; edits and undo cannot return a boundary for different inputs.
  Evaluation runs outside the cache lock. Read-only consumers share the result
  instead of cloning all boundary and collision arrays.
- A fillet/chamfer drag discarded its successfully validated preview and applied
  the same edit again for rendering. It now retains that preview and reuses it
  when both the source revision and feature parameters match.

## Geometry measurement

Command: `cargo run -p mechanic-bench --bin edit-latency` (development profile).
These are CPU microbenchmarks, not frame timings. The before measurements were
taken in this session before adding the cache. The benchmark also includes pipe
and chamfer measurements in its final version.

| Workload | Before | After |
| --- | ---: | ---: |
| 100 block boundary queries, returning owned results | 1.679 ms | 0.333 ms |
| 100 junction boundary queries, returning owned results | 635.268 ms | 37.315 ms |
| 20 fillet preview steps, ten owned queries per step | 26.817 ms | 2.972 ms |

The junction result includes one cold evaluation. After priming, 100 shared
read-only queries take 0.004 ms. This latter measurement excludes the first
evaluation and does not represent end-to-end junction placement latency.

## Whole-app placement diagnosis

The launcher uses a disposable copy of
`crates/mechanic-bench/tests/fixtures/builder-world`. The recorded pre-change
diagnostic uses the existing release executable, whose hash is retained in
`placement-before-run.json`; its exact source snapshot was not reconstructed.
It uses CPU physics and Apple M1 Pro / Metal rendering. Background captures and
different streaming warm-up durations are unsuitable for controlled overall
frame-rate comparisons. Individual terrain rebuild events establish the stall.

The updated release capture completed twelve measured placements. None rebuilt
the entire cut. Two real cut changes updated 94 and 98 chunks, taking 63.66 and
56.38 ms respectively. Other updates reused the retained terrain. Across all
424 terrain-update samples, p95 was 0.377 ms. Maximum terrain-update time fell
from 700.07 to 63.66 ms; this is a reduction, not elimination of every frame
stall. Construction publication maximum was 16.25 ms in the updated run, versus
31.45 ms in the diagnosis. Starting body counts were 32 and 35 respectively,
because scripted placement also runs during streaming warm-up.

Compressed traces, executable/fixture hashes, and stage summaries are retained
as `placement-before*` and `placement-after*`. Both captures completed without
interruption or record overflow. The updated release build completed successfully.

```sh
cargo build --release -p mechanic-app
python3 scripts/run-background-capture.py \
  --binary target/release/mechanic-app \
  --world crates/mechanic-bench/tests/fixtures/builder-world \
  --output /tmp/mechanic-placement-capture --physics cpu --place 5
```

The existing GPU executable stopped this fixture at tick 2 with failure flags 4
before capture. The saved-world GPU failure is also documented in the earlier
freeze report. This task does not claim to fix that solver failure.

## Verification

- Core: 301 tests passed, including cache reuse across placement, changed bases,
  reused arena slots, region cages, frames, feature amounts, replay prefixes,
  rejected edits, and concurrent immutable revisions.
- App broad run: 789 passed, six ignored. One Metal-dependent test passed when
  rerun with device access. The remaining suspension UI marker test also fails
  alone and is already recorded in the September 14 freeze report.
- CPU route after terrain transfer: all 13 tests passed, including retained
  terrain support followed by remeshing and removal.
- App regression verifies preview reuse and rejection of cached previews after
  changing the amount or placing another part.
- Clippy for core, app, and bench with all targets and warnings denied passed.
- Formatting and whitespace checks passed.

Whole-app interactive fillet/chamfer frame latency has not been measured. Large
first-time solid evaluations, combined mesh rebuilds, and genuine CPU terrain
cut updates remain synchronous. No 60-fps or worst-case frame-budget gate is
claimed.
