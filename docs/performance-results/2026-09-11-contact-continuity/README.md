# Contact convergence and release localization, 2026-09-11

The captured 26-row cold-car system now converges within its original 256-sweep
budget. The required 120-tick cold-drop regression still fails at tick 6 with the
production 128-event-trial policy; its previous completed snapshot is retained.
No complete-tick performance measurement or acceptance gate is claimed.

## Diagnosis and retained changes

The preceding checkpoint's independent PGS probe did not establish static
breakaway. A distinct NumPy least-squares Newton reference now reaches the recorded
static-mode equations at a projected velocity residual of 9.95e-11. Its first
contact has approximately 7.91004 Ns normal impulse, tangent impulse at the
0.894427 static coefficient limit, and 5.69e-6 m/s remaining tangent velocity.
All original normal, drive, tangent and rolling laws are checked independently in
the retained Rust regression. This supports the existing transition to kinetic
friction; neither the coefficients nor the promotion criterion were changed.
The reference is an algebra solution, not proof of uniqueness or of a continuous
force trajectory.

Instrumenting the production control then showed the actual budget problem:
with the captured warm impulses it proves breakaway only at sweep 241. It runs
out at sweep 256 during the kinetic phase (residual 3.80e-5). Starting at zero
proves breakaway at sweep 61 and finishes in 106 sweeps (residual 5.32e-10).
The reference and production control reach the same static branch within their
respective residual bounds. Contact/drive rows include a weak independent
sticking direction; solver failure was not evidence of inconsistent constraints.

The retained small-system solver discards a supplied warm impulse guess at most
once after a stalled 16-residual window. It keeps W/H, already established
friction modes, all spent work, the original iteration budget and the original
final residual check. It resets only the impulse iterate and Newton search
history. The captured solve now takes 154 total sweeps, has one restart and one
proved friction transition, and produces exactly the same final impulses and
velocities as the 106-sweep cold solve. A 64-sweep control still reports failure.
The restart allocates no new response matrix. Diagnostics and benchmark JSON
report `warm_start_restarts`; no change was made to the >128-row route.

A separate event trace shows repeated accepted nanosecond prefixes approaching a
released point's velocity reversal. The retained locator keeps the initial
material-point/joint velocity row and a bounded time bracket while reintegrating
from the unchanged start. Clear trials extend toward the reversal; the locator
commits after the row reaches the existing velocity tolerance or an intervening
collision is resolved. Every accepted path still passes finite-geometry CCD and
its independent depth certificate. The production event budget remains 128.
This locates the existing frozen-row reversal candidate; it does **not** certify
general nonlinear or rotating continuous-force trajectories.

A captured 29.019 microsecond authored-car interval requires 90 event trials and
33 accepted prefixes under the previous policy, versus 43 trials and 8 prefixes
with localization. The new regression repeats exactly, finishes within 64 trials,
and independently checks every collider against the finite floor and 5 mm bound.
Against a 128-interval reference, maximum generalized velocity, coordinate and
body-position differences are 6.01e-6, 1.94e-8 and 3.62e-9 respectively (mixed
linear/angular SI units for generalized values). The 32- and 64-interval velocity
errors reduce to 2.37e-7 and 7.86e-8. Fine integration has different impact counts;
this local agreement does not establish global contact continuity or settling.

## Reproduction and controls

Run from the repository root:

```sh
cargo test -p mechanic-physics --offline stalled_warm_contact_restarts -- --nocapture
cargo test -p mechanic-physics --offline car_contact_reversal_uses_bounded_trials -- --nocapture
cargo test -p mechanic-physics --offline saved_car_cold_drop_remains_bounded_through_settling -- --nocapture
```

The last command is a required **failing** regression. The exact car construction
remains `crates/mechanic-bench/tests/fixtures/driven_car_instance.ron`.
`cold_settling.ron` in the physics response test fixtures stores the Cholesky lower
factor, complete blocks and warm impulses; it is byte-identical to the preceding
checkpoint's captured system. `cold_settling_static.ron` retains the independent
static solution. `car_reversal.ron` stores body positions/quaternions, generalized
coordinates/velocities and the captured time proposal (the test requests 1000
proposals' duration).

The independent reference requires the already installed Python 3.13 and NumPy
2.3.0; no production dependency was added. These two archived scripts run 500
iterations then continue the exact saved iterate for up to 5000, using
`/private/tmp/mechanic-contact-reference.npz` for their intermediate state:

```sh
OPENBLAS_NUM_THREADS=1 python3.13 docs/performance-results/2026-09-11-contact-continuity/reference-initial.py
OPENBLAS_NUM_THREADS=1 python3.13 docs/performance-results/2026-09-11-contact-continuity/reference-continued.py
```

They converge after 500 + 2050 updates in this recorded run. Reference scripts
retain their diagnostic prints and original paths. `rejected-solver-experiments.tar.gz`
retains unsuccessful bounded active-polynomial and Anderson proposals; neither is
in runtime code. Anderson's fixed-point formulation was checked against
[Walker and Ni](https://users.wpi.edu/~walker/Papers/Walker-Ni%2CSINUM%2CV49%2C1715-1735.pdf);
no convergence theorem for these non-associated friction equations is claimed.
The extra 1024-event control fixture is diagnostic only and is kept outside
runtime/test source. It advances through tick 49, then fails at tick 50 (20.17 s
correctness-test wall time, not a benchmark). Prior retries include finite-constraint
failures, but the final failure phase still needs an explicit audit; the aggregate
residual alone does not identify it. Full verification and source identities are
recorded below and in `identity.json`.

## Next work

Resolve the remaining tick-6 contact-event workload without larger production
budgets, relaxed tolerances or missing physical time. Preserve this failing case,
then require the full 120-tick drop at default and fixed 1/2/4/8 policies with
repeated hashes and physical bounds. Continue with valid persistent contacts,
continuous-force/rotation trajectory validation, authored driving and the shared
world interface in `docs/contact-impact-next.md`. Application, rendering, GPU and
scale stages remain open.

## Final verification

`cargo test --workspace --no-fail-fast --offline -- --test-threads=1` used the real
Metal adapter with hardware tests serialized. It is **not green**: the same 13
failure names as the preceding checkpoint remain (11 GPU, one UI, and the required
CPU cold-settling case). The two added physics regressions pass.

| Target | Passed | Failed | Ignored |
| --- | ---: | ---: | ---: |
| Mosaic | 18 | 0 | 0 |
| App | 739 | 1 | 6 |
| Bench binaries/fixtures | 12 | 0 | 0 |
| Core | 268 | 0 | 0 |
| GPU | 110 | 11 | 1 |
| Physics | 104 | 1 | 0 |
| World | 91 | 0 | 0 |

The separately run terrain-residency test passes and identifies **Apple M1 Pro /
Metal** in `adapter.log`. `physics_wgsl_parses_and_validates_without_a_gpu` passes.
These are correctness runs; no matched foreground performance capture was made.

The release app build, workspace/all-targets Clippy with warnings denied,
formatting and diff checks pass. The first cold-impact test repeats the unchanged
`47ca32e667997189` hash at one nominal substep. The app continues to use the existing
GPU runtime; these CPU changes introduce no GPU ABI or shader changes.

`implementation.tar.gz` contains 56 changed/new source files over the same frozen
`.physics-reference/replay-final-20260910/source` base as the preceding checkpoint.
The manifest retains 517 source-file hashes, test/binary identities and exact
commands. Source hashes remained unchanged through verification. A later focused
Cargo invocation rebuilt the physics test executable; its before/after hashes
are recorded separately, and the final executable was also run directly to bind
its hash to the complete physics result. Completed first-impact and focused replay states repeat exactly.
