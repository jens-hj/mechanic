# Articulated inertia and analytic contact-block factors

These are retained CPU component improvements. **No complete-world, driven-car,
256-joint physics, rendering, or 100,000-body gate has passed.** The GPU application
remains authoritative and the CPU experiment defaults to the dense inertia
reference. Required cold settling still fails at tick 6 with its 128-trial policy;
the last valid snapshot is retained. The preceding
[continuity checkpoint](../2026-09-11-contact-continuity/README.md) remains the
physical baseline. [Rejected event controls](../2026-09-11-event-search-controls/README.md)
are preserved separately.

## Retained implementation

`DynamicsFactor::articulated` reconstructs tree poses, rotates physical spatial
inertias to world axes, and eliminates scalar joints in compiled postorder. For
joint motion S, U = I S and d = Sᵀ U + implicit diagonal. The parent receives the
transformed I - U Uᵀ/d. A floating root has a six-dimensional Cholesky factor;
anchored roots have no generalized motion. Reverse/forward impulse passes reuse
U and d. No generalized mass matrix or inverse is built on this route. Preparation,
factor storage and each solve are linear in body count; each solve currently
allocates a six-scalar-per-body scratch arena, included in measurements.

This implements established articulated elimination, using Mechanic's compiled
joint directions, schedules, COM offsets and implicit spring/damper diagonal.
[Featherstone's dynamics documentation](https://royfeatherstone.org/spatial/v2/)
describes articulated algorithms for fixed and floating trees. No external source
implementation was copied. Closure equations still need explicit constraint rows;
the joint runtime continues to reject authored loops.

`JointTickSettings.factorization` explicitly selects `DenseReference` or
`Articulated` for finite force solves, instantaneous impacts, split corrections
and external impulses. Diagnostics label that choice. Pose/Jacobian/midpoint
force assembly still uses the dense reference, with its 512-coordinate cap; this
change does not make the complete runtime linear or support a 600-joint tick.
Initialization's inertia validation remains dense. The new standalone factor is
not subject to that dense capacity cap.

The reduced Newton route (>128 contact rows, at most 64 generalized coordinates)
now factors `(1+s)I-P` analytically per contact point. The normal is scalar;
tangent and rolling blocks have radial/orthogonal components coupled to that
normal. The calculation uses the rounded direction's actual squared norm and
avoids a cancellation-prone inverse subtraction. It replaces local LU construction
and repeated triangular solves. Generalized LU, the original projected residual,
friction transitions, shifts, line search and iteration budgets remain unchanged.
The local-factor counter includes these analytic preparations. The dense ≤128-row
and large-machine Krylov routes remain unchanged.

Finite-path validation also caches exactly one whole-tree pose sample at t=0.5
inside each immutable query. Every triangle still receives its envelope test;
other subdivision times reconstruct independently. Nothing persists across query,
motion, origin, topology or terrain changes. The failing dense drop keeps exactly
650 event trials, 734 factorizations and its tick-6 failure, while path/query pose
reconstructions fall from 120,131 to 81,251: 38,880 exact cache hits.

## Matched component measurements

All measurements below ran serially on this Apple M1 Pro host, CPU f64, using the
same executable per factor comparison and fixed fixtures. Each process has 100
warm-up samples and 1,000 measured samples. Ordering is A/B/B/A. This is a sample
protocol for algebra experiments, **not** the unchanged 5 s / 30 s scale-gate
protocol. All simulated durations are zero; collision, publication and GPU work
are explicitly included or excluded in the JSON records.

| Experiment | A total p95 ms (two runs) | B total p95 ms (two runs) | Interpretation |
| --- | --- | --- | --- |
| Authored car: dense → tree inertia, factor plus 32 impulses | 0.036875 / 0.018667 | 0.013667 / 0.012333 | About 1.4–3×; short samples show substantial A-run variation |
| Shared 256-joint chain: dense → tree inertia, same work | 25.435208 / 25.426792 | 0.436583 / 0.438834 | About 58× for this component |
| Car, 240 finite-floor contact rows: dense → tree factor, original local LU | 9.986792 / 9.983834 | 9.958625 / 9.706375 | Inertia alone provides little total benefit here |
| Same finite-contact workload, tree factor: local LU → analytic point factors | 9.744750 / 9.721459 | 5.235250 / 5.233458 | About 1.86× for the complete fixed-pose experiment |

The 256-joint tree factor's preparation is approximately 0.076 ms versus 23.74 ms
for dense pose/Jacobian/mass assembly and factorization. Its 32 impulse solves,
including RHS copies and scratch allocation, take approximately 0.36 ms versus
1.73 ms. This fixture is the same shared chain as `bearings_256`; it has no loops
or contacts in this measurement and does not meet the 256-joint physics gate.

Every inertia sample checks all 32 responses against a precomputed dense
reference. Maximum relative differences are 1.95e-17 (car) and 9.88e-16 (chain).
Hashes repeat exactly within each factor method and differ across methods.
The analytic contact experiment retains 73 iterations and residual
6.4015995743283676e-9. Its repeated hash is `14d05f3e76e0c08c`; original local LU
is `745617f99b86b009`. Different arithmetic is not claimed bit-identical across
implementations. The independent point-matrix and physical regression checks are
required in addition to residuals and hashes.

The first finite-contact series is retained under `overlapped-response-control/`
and excluded in full: the fourth run overlapped Clippy. The entire A/B/B/A series
was then repeated without concurrent agent builds or tests. `commands.json` marks
both series; no overlapped result appears in the table above.

## Verification and reproducibility

Focused physics after the retained changes: **112 passed, 1 failed**, no ignored
physics tests. The failure is the existing required 120-tick cold-car settling
test, still at tick 6. A diagnostic source variant selecting the articulated
factor by default, before the analytic point change, had **110 passed, 1 failed**
(the same tick-6 case). Repeating that control after the analytic point change
has **112 passed, 1 failed**, again tick 6. Both variants are archived; neither is
the runtime default. The final first-impact hash remains `47ca32e667997189`, repeated
exactly twice after restoring the retained source. The permanent first-supported-car test exercises both factor selections at
1/2/4/8 substeps with exact repeatability and independent finite-floor bounds.

Additional checks cover every independent inverse column, rotated and translated
roots, anchored/reversed joints, COM offsets, 1000:1 inertia ratios, independent
components, authored car suspension, 3- and 129-row contact response, friction
bounds, invalid inputs, and independent torque balance for a 600-joint coaxial
chain. Analytic point tests compare every original matrix row and pivoted LU
through clamp/apex/interior/exterior/static/kinetic/rolling cases and shifts
1e-2 to 1e-8; the full reduced Newton row test also passes.

The full workspace run on Apple M1 Pro / Metal completed with the **same 13
failure names** as the preceding checkpoint: 11 GPU physics failures, the UI
suspension leader failure, and required CPU cold settling. Totals are Mosaic
18 passes; app 739 passes / 1 failure / 6 ignored; benchmark 12 passes; core
268 passes; GPU 110 passes / 11 failures / 1 ignored; physics 112 passes /
1 failure; world 91 passes. WGSL validation passes. Hardware tests ran serially;
the separate terrain-residency check prints the actual adapter and passes.
`cargo build --release -p mechanic-app --offline`, workspace Clippy with warnings
denied, formatting and diff checks pass. The workspace is **not green**.
Exact commands, source/binary identities and result hashes are in `identity.json`.

`inertia-implementation.tar.gz` retains the physics and benchmark source for the
original inertia measurements; `inertia-source-hashes.json` and
`benchmark-binaries.json` identify it. The analytic-point subdirectory retains
its four-file overlay, hashes, exact commands and binary identities. The final
implementation archive/manifest use the same full frozen reference base as the
continuity checkpoint. Rebuild references in isolated Cargo target directories.

## Remaining work

The cold-car event workload remains a correctness blocker, independent of these
speedups. [Positive-gap controls](positive-gap-control/README.md) now test the
mismatch between a zero-gap root target and the existing positive activation
window / strict post-impact travel bound. They fail at ticks 7 and 6 and are
rejected; no search change is retained. These results do not justify further
arbitrary interpolation constants. Investigate sustained contact/force trajectory
consistency and event work using the retained failures. Do not enlarge the window
or budgets, suppress valid features, or accept a partial tick.

Dense midpoint force/Jacobian assembly and per-solve scratch allocations remain
optimization work. A full tree operator must match mass application, bias,
gravity and point Jacobians against the reference before replacing it. Actual
idle-to-driving command resolution, streamed-terrain transactions, body/body
contacts, loops, sleeping, common world publication, rendering and portable GPU
work remain in the [master plan](../../compiled-machine-dynamics.md).
