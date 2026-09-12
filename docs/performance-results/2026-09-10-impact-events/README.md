# Bounded first-impact events

The private CPU experiment can now resolve a newly encountered finite-terrain
impact and integrate the remaining time. The public runtime remains joint-only;
the app still uses its original GPU runtime. No complete-world/performance gate passes.

## Physical interval and publication

CCD returns an explicit first-impact candidate. The trial is discarded and a shorter
interval is reintegrated from the unchanged starting state, including forces, drives
and stops. Every accepted prefix passes the full finite penetration certificate.
Only the complete 60 Hz tick publishes commands and state; exhausted event work
retries the entire unpublished tick, then holds the previous snapshot.

Activation requires the actual collider/chunk/triangle feature, a fixed 1e-12 m
normal-gap bound, and an outward bound on every collider's remaining point motion.
The separated-pair search uses half the gap bound; rotational CCD approaches within
one quarter. Existing intersections keep their manifolds. Split position paths
remain separate from physical velocity and cannot advance time to resolve an event.

A captured wheel case showed clipping roundoff amplified by a nearly parallel face.
The shared geometry query now searches inside the finite clipped polygon for a
feasible opposing point with bounded bisection. It never clamps the gap or enlarges
the activation bound. Failed numerical activation queries request a retry, not clear
space. The captured geometry and earlier failures remain regression evidence.

## Solver experiment and controls

Contact Newton systems through 128 scalar rows use bounded dense matrices. A
regularized proposal runs first; when it cannot make progress, complete pivoting
constructs an unshifted rank-revealing proposal. Up to 32 backtracking samples and a
16-residual nonmonotone window permit friction transitions. The history resets after
a static/kinetic mode change. Final convergence still checks the original unshifted
physical residual. Above 128 rows the prior reduced/Krylov routes remain unchanged.
Dense storage counts the original matrix plus its mutable elimination copy when
checking a rank-revealing proposal. All proposals/factors/backtracking work are counted.

Nonmonotone line search is an established idea; see the
[original paper](https://epubs.siam.org/doi/10.1137/0723046) and the
[description of a recent-maximum criterion](https://epubs.siam.org/doi/pdf/10.1137/S1052623403428208).
This application to nonsmooth projected friction is an experiment, not a transferred
convergence theorem. Final physical checks and bounded failure remain necessary.

Retained controls explain why simpler changes were not accepted:

- The prior solver failed the cold impact after every 1/2/4/8 retry. Raising its
  dense variant to 8,192 iterations still failed; the last residual stayed near
  1.2264e-6. More iterations did not solve the problem.
- Direct regularized contact Newton reduced inverse-mass applications but reached
  the same failure. No timing improvement is claimed from that operation count.
- Unshifted rank handling with eight backtracking samples failed. With 32 samples
  it passed the cold car but regressed eight existing test assertions, including
  physical support/bounce cases. That candidate was rejected.
- Trying a rank proposal only after regularized stagnation preserved physical
  regressions but still failed the cold car. Bounded nonmonotone acceptance passed
  both. Raw control logs and pre-experiment source archives are retained; intermediate
  proposal variations are described here, not claimed as separately frozen builds.

## Verified scope

The tests cover fast cold translation, gravity reintegration at impact, a rotating
cube's first finite edge contact, finite holes, simultaneous old floor/new wall,
initial contact activation, event-budget exhaustion, command rollback and repeatability.
Existing analytic friction, rolling, momentum, drive, suspension and stop tests pass.

The exact authored car starts 5 cm above a finite Rock floor with -4 m/s root
velocity, gravity and its authored idle drives. It completes its first external tick
after retries to four subdivisions. Repeated snapshots match exactly:
`273462a3b30cd1ca`. The accepted intervals total 1/60 s. Whole accepted paths meet the
configured 2 mm penetration certificate; independent final collider vertices meet
the 5 mm impact bound. This is one cold-impact tick, not a sustained drop/settling,
driving, streaming or performance measurement. The existing 120-tick supported car
regressions also pass at every fixed 1/2/4/8 policy.

Current CPU packages pass: core 262, physics 93, world 91, saved-car bench 7 and
fixture 5. Workspace Clippy, formatting and whitespace checks pass. Fresh serial
workspace validation on Apple M1 Pro / Metal reproduces exactly the same 11 GPU
failures and one UI failure: no new or removed failure names. GPU tests report
110 passed/11 failed/1 ignored; app tests 739 passed/1 failed/6 ignored. Shader
validation and the separately adapter-labelled terrain-residency test pass. The
workspace is not green. The release app build also passes; its binary hash and
raw build log are recorded.

Continue with [release/re-impact and sustained trajectories](../../contact-impact-next.md).
All [master-plan performance and physical acceptance gates](../../compiled-machine-dynamics.md)
remain unchanged and open.
