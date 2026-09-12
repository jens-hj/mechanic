# Frozen baseline and compiled response experiment

This record starts the compiled-machine redesign. It does not close the driven
car, native rendering, or either 100,000-body gate. Dates span the local midnight
between September 9 and 10.

## Reference identity

`reference-manifest.json` identifies the starting HEAD, every tracked/untracked
non-ignored file, existing release binaries, rebuilt benchmark binary, source
archive, and every file in the copied `car` save. `status.txt` records the starting
dirty worktree. The authored input and driving script sources are in
`reference-sources.tar.gz`; binary/media assets are identified by hash rather
than duplicated in this smaller archive. The original complete 214,854,455-byte
worktree archive, binaries, test executable, world copy, and binary patch remain
in `target/physics-reference/2026-09-09/` locally.

The full worktree archive SHA-256 is
`1a92ab3de91f690a33df2079201bc897d49220a49d8e773ddf11f682957d45d9`.
The archived benchmark was rebuilt with
`cargo build -p mechanic-bench --release --offline` before edits. The frozen GPU
test executable was compiled from the same sources. Existing release app binary
source correspondence is unverified; no new controlled foreground app baseline
is claimed here.

## Fresh reproduced failures

Hardware: **Apple M1 Pro / Metal**. Hardware runs were serialized. Both benchmark
runs used `--warmup 5 --seconds 30` (300 warm-up ticks and 1,800 measured ticks at
the unchanged 60 Hz external clock). These are serialized headless throughput
runs, not native-resolution integrated acceptance. The first sandboxed adapter
request failed; the recorded runs used actual Metal access.

| Frozen workload | TPS | GPU p95 | Flags | Observation |
| --- | ---: | ---: | ---: | --- |
| `four_bar` | 160.16 | 4.804 ms | 4 | Axis error 0.001187°, above 0.001° |
| `dense_100k` | 6.37 | 155.594 ms | 9 | 1,408,911 active contacts; 4,524,480 requested pairs |

Raw benchmark output is retained in `frozen-benchmarks.json`, including its old
coverage labels and old timestamp-byte undercount. These labels are historical
output, not new assertions of coverage. Candidate reporting removes the allowlist
and counts all 28 timestamp words.

The frozen command

```sh
target/physics-reference/2026-09-09/mechanic-gpu-tests \
  device::tests::articulated_car_drop_settles_without_drift_or_ground_penetration \
  --exact --test-threads=1
```

ran one test on Apple M1 Pro / Metal and failed at tick 30: wheel 2 centre
`y=0.48957208`, approximately **10.43 mm** penetration for the 0.5 m radius.
`car-drop.log` retains the failure. This programmatically constructed GPU drop
fixture is distinct from the saved driven-car fixture below. Neither fixture's
tolerances were relaxed.

## CPU algebra experiment

`crates/mechanic-bench/tests/fixtures/driven_car_instance.ron` is byte-identical
to `car/generations/11/world.ron` in the frozen save. The experiment uses its
authored bind pose: 11 bodies, 16 generalized velocities, and 11 synthetic
horizontal support rows. It deliberately does not replay saved motion, root
placement, input scripts, terrain, real contact manifolds, or drive programs.

`compiled-response-initial.json` retains the first implementation: 64 sweeps,
residual approximately `6.742e-6`, and failure to reach the requested `1e-8`.
That implementation still solved inverse dynamics within every contact sweep.
Its intermediate numerical source was not separately archived, so this record
is diagnostic rather than a reproducible A/B baseline.

The revised implementation forms the small W once, iterates entirely in
constraint space, then applies the generalized response once. It uses 12
inverse-dynamics applications for the 11 rows, independent of sweep count.
The recorded run in `compiled-response.json` reaches approximately `9.220e-9`
in 113 sweeps, with identical response hashes for 1,000 repetitions after 100
warm-up samples. The initial revised run measured about 0.19 ms p95 for assembly,
factorization, gravity projection, support-probe construction, and solving.
The JSON contains the latest measured value. The old/new iteration budgets
differ, and these are neither foreground A/B/B/A runs nor a physics speedup claim.

The saved-car tests compare every joint's point Jacobian with central differences
of reconstructed body motion and verify uniform free fall without joint
stretching. Analytic tests cover coupled chassis/wheel impulse response, duplicate
and inconsistent bilateral rows, unilateral separation, a circular friction bound,
off-diagonal coupling with a clamped contact, and repeatability across the 128/129
row storage boundary. The standalone numerical solver remains an experiment.

## Verification

The workspace test command is `cargo test --workspace --offline -- --test-threads=1`.
The application passes 734 tests, skips six explicitly ignored tests, and fails
the existing `ui::suspension::tests::leader_geometry_follows_projected_positions_without_remounting`
assertion (`leader marker missing at Vec2(850.0, 450.0)`). It is not a green
workspace. Remaining crates and the GPU suite are checked separately; detailed
logs and the final counts are recorded in `verification.json`. The GPU suite
passes 109 tests and fails 11 (one ignored); all 11 failures also fail in the
frozen executable. The new focused evidence test passes with an explicitly
required Apple M1 Pro / Metal adapter. CPU checks pass 243 core, 88 world, eight
physics, and seven benchmark tests; all 11 capture-summary regressions pass.
Workspace Clippy, formatting, and diff-whitespace checks pass.

The initial instrumentation exposed a new ninth-storage-binding failure in the
fused contact shader. That marker was removed, preserving the portable eight
binding limit. The rerun passes the three affected application GPU showcase
tests. Full kernel coverage remains explicitly unproven; no unavailable or
zero-test GPU run is counted as a pass.

See [the delivery status](../../compiled-machine-dynamics.md) for remaining work
and unchanged acceptance requirements. The first major driven-car decision is
still open. The CPU experiment is not connected to application tick publication,
and the existing GPU runtime and rendering path remain active.
