# Coupled contact search — active continuation

This is an unfinished solver experiment, not a physics or frame acceptance result.
The original event-resolved cold-drop regression remains active. The separate
endpoint-contact policy also remains experimental and has not passed cold settling.

## Retained runtime work

- A 240-row impact (16 generalized velocities, 48 contacts with tangential and
  rolling friction) stalled above 6e-6 at the original 256-iteration budget.
- Smoothing continuation now proposes contact impulses and checks the original,
  unmodified nonsmooth equations before accepting them. Only contact-only systems
  above 128 rows and at most 64 generalized coordinates use this search. It runs
  once after 16 ordinary iterations and spends at most 128 remaining iterations;
  those iterations count against the original total budget. It has at most 32
  smoothing stages, 32 Newton steps per stage, and 32 backtracks per step.
- Well-conditioned local Newton rows are eliminated. Weak rows are retained in
  a mixed system capped at 128 total generalized/contact rows. Column-pivoted
  Householder QR and a minimum-norm least-squares direction avoid unnecessarily
  large changes between almost equivalent contact impulse distributions.
  Every original contact row remains in final validation. No dense global W is
  allocated above 128 scalar contact rows. A search matrix is not a physics
  compliance term and does not modify the required residual or material law.
- Two captured endpoint impacts pass the original normal/friction/rolling checks
  and repeat exactly. Their generalized velocity changes also agree with a
  separately implemented full NumPy SVD reference. See the results below.
- Separate finite-terrain position recovery uses actual penetrating vertices
  and an active normal basis. Its final multipliers are reapplied through H to
  check every original normal inequality. Physical velocities and elapsed time
  remain unchanged. Focused recovery and dependent/inconsistent normal tests pass.
- The captured non-nested event-prefix replay now asserts reference-refinement
  errors, elapsed-time accounting, actuator budgets, finite-floor bounds and
  repeatability; it previously only printed the refinement errors.

## Rejected controls

Joint normal updates, a normal-only seed followed by the original friction solve,
a reduced contact subset, a larger dense-response threshold, and a Krylov retry
failed to resolve the original 240-row problem. Logs are retained; none establishes
physical acceptance. The dense threshold remains 128.

A shifted reduced smoothing solve passed the first capture but failed a later
240-row impact. Smaller shifts and Krylov corrections also failed. Independent
NumPy dense least-squares controls resolved both. At a captured Newton step, the
mixed basic solution had norm 347 versus 31 for the reference despite similar
machine motion. This motivated the bounded minimum-norm QR search.

## Independent reference

`reference.py` uses NumPy only and accepts a RON fixture with mass/Jacobian/target
and friction data. Example (from repository root):

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3.13 docs/performance-results/2026-09-11-coupled-search/reference.py --fixture crates/mechanic-physics/src/response/tests/fixtures/late_endpoint_impact.ron --output /private/tmp/late-reference.ron
```

Smoothing continuation for complementarity equations is established numerical
work; see [Chen–Mangasarian continuation methods](https://epubs.siam.org/doi/10.1137/S1052623497316191)
and [smoothing Newton methods](https://www.polyu.edu.hk/ama/profile/dfsun/files/Smoothing2002.pdf).
These references motivate the search approach; they do not prove convergence of
this non-associated Coulomb implementation. Its physical validators remain decisive.

## Verified impact results

| Capture | Total directions/sweeps | Continuation directions | Original residual | Independent velocity error |
|---|---:|---:|---:|---:|
| First endpoint | 70 | 54 | 3.47e-10 | 1.74e-10 |
| Later endpoint | 78 | 62 | 8.16e-11 | 2.76e-11 |

These are algebra checks with zero simulated duration. The first solve applies H
3,597 times; the later solve applies it 4,020 times. They establish neither the
complete physics latency target nor a speedup. Every original row is validated.

The third 240-row capture still fails the runtime's original 256-iteration limit.
A separate central-path reference now **does** produce a physically valid solution.
The Rust validator accepts that supplied solution in one sweep, with residual
2.22e-11, and independently checks normal complementarity, tangent and rolling
bounds/opposition, and reconstruction through H. Thus nonexistence of a physical
solution is not the explanation for this failure. The runtime still cannot find it.

The reference traces a path whose smoothing parameter reverses direction twice
near 1.4e-6–3.4e-6. A monotone parameter schedule cannot follow that same segment.
The trace alone remains unconverged; a stable analytic elimination of its disk
barrier completes the search. This is an expensive **offline existence reference**,
not a retained runtime algorithm or an excuse to increase its iteration budget.
No reference impulses are supplied by the application or the cold-start regression.

Reproduce the exact two-stage control from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3.13 docs/performance-results/2026-09-11-coupled-search/third-reference/trace.py --fixture crates/mechanic-physics/src/response/tests/fixtures/third_endpoint_impact.ron --output /private/tmp/third-trace.ron
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3.13 docs/performance-results/2026-09-11-coupled-search/third-reference/finish.py --fixture crates/mechanic-physics/src/response/tests/fixtures/third_endpoint_impact.ron --initial /private/tmp/third-trace.ron --output /private/tmp/third-solution.ron
```

The finishing script and ordinary `reference.py` emit explicit status/identity
JSON and return a nonzero exit code when original laws fail. Failed candidates
are retained as evidence; their existence on disk never establishes convergence.
The stable ordinary reference passes the first two fixtures and rejects the third.

## Verification and unresolved physical gates

- `cargo build --release -p mechanic-app --offline`: pass, [log](release-app.log).
- `cargo test --workspace --no-fail-fast --offline -- --test-threads=1`: completed
  serially on **Apple M1 Pro / Metal**, [log](workspace-tests.log). App: 739 passed,
  1 failed, 6 ignored. GPU: 110 passed, 11 failed, 1 ignored. The same twelve app/GPU
  failures were already present. Core 270, world 91, Mosaic 18, and benchmark 12
  tests pass. GPU WGSL parsing/validation passes. The workspace is not green.
- Subsequent complete physics run: **133 passed, 4 failed**, [log](physics.log).
  The independent third-impact proof adds a passing regression; the original
  from-scratch impact regression remains failing.
- Workspace Clippy with all targets and `-D warnings`, and formatting: pass.
- Required event-resolved cold drop: fails at external tick **17**. Experimental
  endpoint policy with adaptive retries: fails at tick **20**. Neither publishes
  its failed candidate; both retain the previous complete snapshot.
- Fixed endpoint policies at 60 Hz all fail: one subdivision at tick **8**, two at
  **4**, four at **2**, eight at **11**. [Matrix log](fixed-policy-matrix.log) and
  [exact control source](fixed-policy-matrix-control.rs). They use the same fixture,
  tolerance, depth bounds and 120-tick objective. No policy is selected as passing.

Additional rejected third-impact controls include normal subsets, normal hulls,
rolling redistribution and zero-rolling diagnostics, several friction/smoothing
homotopies, natural-map and Fischer–Burmeister variants, approximately two million
point PGS sweeps, and null-space release proposals. None passes the unchanged
physical gate. Changing the rolling coefficient or dropping original rows is not
retained. The successful central-path reference is distinguished from these failures.

## Immediate continuation

Implement a bounded search that reaches the independently verified third solution
from actual runtime inputs. Check derivative and elimination algebra independently,
retain the 128-row response storage limit and account for every direction/factor
application. Reduced velocity/normal central-path controls, including friction elimination
and recentering, also fail. Exact rolling-row aggregation and fixed rolling-stick
hypotheses have not produced a valid bounded candidate. None is retained in Rust. Then repeat sustained cold settling, the
1/2/4/8 physical comparisons, authored driving, and full-tick foreground A/B/B/A.

Split-recovery and endpoint-policy physical/reference coverage, common world
publication, loops, body/body collision, streaming/sleeping, application rendering,
portable GPU formulation and all integrated performance milestones remain open.

## Source and evidence identity

[identity.json](identity.json) identifies 539 source files, the app/test binaries,
the adapter, and the retained evidence. [implementation.tar.gz](implementation.tar.gz)
contains 79 changed source files to overlay on the preserved full source reference
at `.physics-reference/replay-final-20260910/source`; use a separate Cargo target
directory when rebuilding it. No reference runtime has been removed.

[third-controls.tar.gz](third-controls.tar.gz) retains the additional offline
control programs, captured inputs, candidates and logs. Temporary path names in
those rejected controls refer to files in that archive; extract them under
`/private/tmp` to replay those exact commands. Use the standalone two-stage
reference above for the verified third solution. A reduced/hypothetical problem
passing its own residual is insufficient: its reconstructed torque and all original
constraints must also pass. In the rolling-stick controls, even the two converged
subproblems require torque far beyond the original rolling capacity.
