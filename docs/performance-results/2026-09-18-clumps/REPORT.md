# CPU material breakage and clumps — 18 September 2026

Two headless release fixtures ran sequentially on an Intel Core i5-12600K with
no concurrent compilation. `material-clumps` drops 256 always-active 10 cm
clumps, cycling through every terrain material, onto a flat triangle floor.
`material-mining` presses a powered rotary cutter into production-meshed
editable terrain and lets the breakage system extract clumps from measured
contact stress and work. Exact commands, binary hashes and the source commit are
in [manifest.json](manifest.json).

## 256 active clumps

| Measured quantity | Value |
| --- | ---: |
| Bodies, all active for every tick | 256 |
| Degraded ticks | 0 / 360 |
| Physics tick median | 77.3 ms |
| Physics tick p95, traced run | 103.9 ms |
| Physics tick p95, two untraced repeats | 113.6 ms, 112.6 ms |
| Physics tick maximum | 141.6 ms |
| Contacts per tick | 1,014 – 5,807 |
| Material quanta at start and end | 1,044,480 |

**The 256-clump replay does not fit a 60 Hz tick.** Its p95 is six to seven
times the 16.7 ms budget, so no clump scale gate is claimed. Cost rises as the
pile settles and contact count grows: the first ten ticks average 36 ms and the
last sixty average 89 ms. The fixture is a deliberate worst case. Every body
stays awake in one dense stack, whereas the app lets settled mineral fragments
sleep and turns settled sand, soil and cover back into terrain. Those paths are
not measured here. The spread between traced and untraced runs is run-to-run
variation on a desktop session, not a tracing cost; one-minute load average was
about 5 when the runs started.

## Powered extraction

Every run is 180 ticks with zero degraded ticks. One terrain cell holds 510
quanta.

| Material | Tool mass | First extraction | Cells removed | Clumps | Cells redeposited | Physics p95 | Total tick p95 | Remesh | Publication |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sand | 80 kg | tick 12 | 10 | 4 | 2 | 1.20 ms | 1.25 ms | 13.5 ms | 1.7 ms |
| Cover | 80 kg | tick 24 | 10 | 5 | 0 | 1.23 ms | 1.25 ms | 13.4 ms | 1.9 ms |
| Soil | 80 kg | tick 42 | 11 | 4 | 2 | 1.07 ms | 1.17 ms | 13.4 ms | 1.7 ms |
| Graphite | 1,600 kg | tick 108 | 5 | 5 | 0 | 0.73 ms | 0.85 ms | 13.6 ms | 1.9 ms |
| Rock | 8,000 kg | tick 48 | 4 | 4 | 0 | 0.82 ms | 0.84 ms | 10.2 ms | 1.2 ms |
| Iron | 20,000 kg | tick 60 | 4 | 4 | 0 | 0.71 ms | 0.72 ms | 10.2 ms | 1.2 ms |

Material is conserved in every run: removed cells × 510 equals the quanta held
by clumps plus the quanta in redeposited cells. The 360-tick `--settle` sand run
removes 10 cells, redeposits all 10 and ends with no clumps and no clump quanta.

Remesh and publication are summed over the run and measured synchronously in
the benchmark. A run remeshes on only three to eight ticks, so the tick p95 does
not include that work; the worst single tick, remesh included, is 4.2 – 4.8 ms.
The app uses its asynchronous edit and remesh pipeline, so these figures are
not app frame costs.

### What the extraction table does not show

The plan asked for the resistance ordering to be demonstrated with identical
tools. **It is not.** The fixture gives each material class its own tool mass,
and the mass also sets the feed load and motor torque limit, so the table shows
that each material *can* be broken by some fixture rather than ranking them
under one. The mineral masses are also feed-load proxies, not plausible bodies:
a 5 × 10 × 20 cm cutter weighing 20 t has no physical counterpart. An
identical-tool comparison and a fixture with a realistic mass and an external
feed force remain outstanding.

An earlier revision drove sand with the rock fixture. That run hit the solver
speed limit on 79 of 180 ticks, from tick 100 onward, after removing 151 – 163
cells into fast fragments. The committed fixture avoids the case by giving soft
ground a light cutter; it is not fixed in the solver, and its capture was not
retained. An overpowered tool on soft ground should be expected to degrade.

## Replay viewer

`python3 build-replay.py` embeds the captures into `replay.html`, a standalone
viewer of the measured poses and terrain meshes. The output is derived and
ignored by Git.

Raw results: [clumps](clumps.jsonl.gz), [sand](sand.jsonl.gz),
[soil](soil.jsonl.gz), [cover](cover.jsonl.gz), [graphite](graphite.jsonl.gz),
[rock](rock.jsonl.gz), [iron](iron.jsonl.gz), [deposition](deposition.jsonl.gz),
[summary.json](summary.json). Every record has
`kernel_coverage_complete: false`. No GPU or clump scale gate is claimed.

## Verification

- `cargo fmt --all -- --check`: passed.
- `cargo clippy --workspace --all-targets -- -D warnings`: passed.
- `cargo check --workspace --all-targets` passes at every commit of the series.
- `cargo test --workspace --no-fail-fast`: mechanic-core (323), mechanic-world
  (105), mechanic-physics (218), mechanic-bench and bevy_mosaic pass;
  mechanic-app passes 802 of 807. The GPU and suspension failures below
  reproduce on the parent commit `1d95095` with unmodified sources; the timing
  test is in a file this series does not touch.
  - Four GPU tests lose the device after about 15 s with `Parent device is
    lost` on an AMD Radeon RX 9070 XT, RADV/Mesa 26.2.2, Vulkan:
    mechanic-gpu `articulated_car_drop_settles_without_drift_or_ground_penetration`
    and mechanic-app `dynamic_presets_do_not_gain_unbounded_spin`,
    `showcase_runs_1_200_gpu_ticks_without_failure_or_blowup` and
    `smaller_creation_presets_run_120_gpu_ticks_without_failure`. In a parallel
    run the mechanic-gpu test binary then exits with SIGSEGV, so the rest of
    that crate's tests do not report.
  - mechanic-app `leader_geometry_follows_projected_positions_without_remounting`
    fails with `leader marker missing`.
  - mechanic-app `fast_volume_path_keeps_4096_blocks_individual_with_exact_welds`
    asserts a 16.7 ms wall-clock bound in a debug build and fails intermittently
    under load; it failed one of three isolated runs at 16.9 ms.
- mechanic-app `an_inherited_cut_initializes_cpu_terrain_after_a_body_split`
  failed only on machines with a saved world containing clumps, because app
  test fixtures loaded the player's real world store. Fixtures now start from an
  empty store, and the test passes.

New regression tests cover stress-and-work gating at the exposed contact, work
tracking motion rather than stationary contact correction, atomic extraction
that preserves compressed quantity, soft deposition against persistent rock,
clump collisions with terrain and each other, adding and removing clumps without
disturbing authored state, clumps simulating without authored bodies across a
global rebase, one snapshot holding terrain and clumps together, and saving
during an unpublished transfer.

## Scope and remaining work

Simulation is CPU only. Publishing a world that holds clumps on the GPU route
returns an explicit error instead of dropping the bodies; that rejection has no
test yet.

Not yet covered by tests or captures: comparable breakage across substep counts
and contact tessellations, capacity saturation and sleeping-fragment wake
reservation, bucket transport of soil, hard-fragment sleep and wake, partial
deposition, concurrent terrain edits and failed publication, floating-origin
shifts with live clumps, and construction rebuilds. Large boulders, collapse of
unsupported ground and secondary fragmentation are deferred by design.

World format 6 directly replaces format 5, so format 5 worlds now report as
unsupported. Personal worlds were neither migrated nor deleted.
