# Source-matched foreground car baseline

Stage 1 remains open. Two native foreground captures now have verified source,
binary, asset, saved-world, initial-state, and initial-camera identities. Both
fail performance and repeatability acceptance. These are repeated reference
runs, not an A/B solver comparison or an optimization result.

## Build and preservation

The original complete reference was copied and hash-verified outside `target`
at `.physics-reference/2026-09-09/`; `preservation.json` there identifies all 40
preserved files. Its application was rebuilt from the original archive, with
the command, binary identity, and log retained here. That application retains
the historical capture protocol and is not used for these foreground results.

A subsequent shared-target build of the repaired capture harness reused a stale
`mechanic-core` artifact and failed because `CompiledCreation::dynamics` was
absent from that artifact. The failure is retained. Reference builds now use
fresh isolated target directories, excluding shared Cargo artifacts entirely.

The measured executable was built with:

```sh
python3 scripts/build-performance-reference.py \
  --output .physics-reference/foreground-baseline-isolated-20260910
```

`build-identity.json` identifies every copied source file, the toolchain, build
command, and executable SHA-256. `isolated-build.log` records the successful
7-minute-9-second release build. Complete sources, assets, and executable remain
under that ignored reference directory. `reference-sources.tar.gz` preserves the
source/text subset here; runtime media are identified by hash. The exact capture
and analysis scripts used for the runs are in `capture-tools.tar.gz` because
launcher/analysis checks continued to improve after freezing the executable.

## Protocol

Run this command with distinct `--output` paths ending in `01` and `02`:

```sh
python3 scripts/run-background-capture.py --foreground --drive \
  --binary .physics-reference/foreground-baseline-isolated-20260910/mechanic-app \
  --identity .physics-reference/foreground-baseline-isolated-20260910/identity.json \
  --assets .physics-reference/foreground-baseline-isolated-20260910/source/crates/mechanic-app \
  --world .physics-reference/2026-09-09/car \
  --output .physics-reference/foreground-car-run-01
```

Both ran serially on Apple M1 Pro / Metal, after builds/tests finished. No other
Mechanic instance was running. Every measured frame was focused and rendered at
4112x2524, with 4x MSAA, AutoNoVsync, and F3 enabled. Other operating-system work
was not disabled; these two repetitions do not establish a confidence interval.

The application loaded a disposable copy in an isolated store. Source worlds
and assets were hash-checked before/after execution. Graphics warmed for 15
continuously ready seconds without advancing physics. The 60-wall-second
capture then applied the existing W/A/D script at each dispatched tick, with
180 settling ticks before throttle. This differs from historical wall-time
physics warm-up and must be matched in future comparisons. Scheduler drops are
still counted; no sleeping, lower resolution, altered material, or solver
tolerance was introduced to improve the numbers.

## Measurements

| Metric | Run 01 | Run 02 |
| --- | ---: | ---: |
| FPS | 14.19 | 14.15 |
| Completed physics TPS | 24.35 | 24.00 |
| GPU physics p95 | 35.701 ms | 36.146 ms |
| Dropped ticks during capture | 1,334 | 1,328 |
| Terrain-readiness hold events | 360 | 364 |
| Submitted / completed ticks | 1,470 / 1,461 | 1,449 / 1,440 |
| Submissions pending at capture end | 9 | 9 |
| Physics failure flags observed | 0 | 0 |

Run 01 frame p95 was 116.214 ms. CPU encoding/finalization/submission/readback
setup p95 values were 0.105/2.396/0.820/0.007 ms. Submission-to-readback p95 was
342.929 ms; callback-to-publication p95 was 84.412 ms. These distributions
overlap or describe different samples and must not be added. Full distributions
and per-stage timings are in each run's summary and compressed raw capture.

All render GPU samples reported `Partial: reversed`. The combined render span
and affected passes are unavailable, not zero. Valid individual opaque spans
in run 01 had p95 29.961 ms; these do not isolate terrain from construction.
Raw frame timing remains available. The image was inspected and shows the
authored suspension car on terrain; a screenshot cannot establish millimetre
penetration bounds. Zero physics flags and partial execution counters likewise
do not establish full execution coverage or physical correctness.

## Repeatability failure

Both captures start with state hash `ad26c619b20ad19f` and identical camera
matrices. Completed-state hashes first differ at actual tick **12** (script tick
11), while both scripts hold no keys. The first eleven states match. Active
contacts grow from 10 at tick 11 to 58 at tick 12 in both runs. The first dropped
tick is 25 in both runs, so dropped input does not explain the first divergence.
The first captured terrain replacement is around 12 seconds, also later.

The initial resident terrain geometry/order was not hashed, so the cause is
not yet isolated to solver ordering versus collision/terrain preparation.
`first-divergence.json` retains the state-hash/contact prefix and subsequent
first drop/hold/terrain-publication events. The comparison correctly rejects
these runs: states diverge, later simulated durations/input sequences differ
because of drops, and final submitted states are not completely captured.
A matching shared prefix cannot prove whole-run repeatability or a speedup.

## Verification and next work

Workspace Clippy, formatting, and whitespace checks pass. Focused checks pass:
2 automation, 4 capture, 25 sequencer, and 19 Python regression tests. The full
workspace command reaches the app and reports 734 passed, 2 failed, 6 ignored:
the existing suspension UI failure and a Closure Lab flags=4 failure in
`dynamic_presets_do_not_gain_unbounded_spin`. The latter passes a focused rerun;
it remains an observed intermittent failure. The workspace is not green.
Hardware checks used Metal and were serialized; no GPU solver/shader code was
changed in this continuation. The previously recorded full GPU suite failures
remain open.

Before a matched solver comparison, finish the capture boundary/drain and
fixed-simulated-duration replay protocol, identify the initial terrain and
effective drive rows, and isolate the tick-12 divergence. Finish actual kernel
coverage instrumentation without relying on scenario names or zero flags.
Preserve these failing runs as regressions while completing the CPU tick and
coupled-contact experiment. All vehicle, rendering, and scale gates remain open.
