# Finish sustained cold contact dynamics

Durable continuation for the private CPU experiment. First-impact integration is
implemented; sustained cold-car settling and the complete-world interface remain
open. The user requests continued implementation without long chat summaries.

## Implemented boundary

The [event checkpoint](performance-results/2026-09-10-impact-events/README.md)
extends the earlier instantaneous activation and supported-tick experiments:

- Exact linear CCD uses cached endpoints only when accumulated ancestor angular
  motion is zero; rotational CCD retains bounded conservative advancement.
- Private trials return `Complete` or an explicit first-impact candidate. Shorter
  intervals are reintegrated from the unchanged start. Accepted prefixes subtract
  from the remaining time; 128 trials per nominal substep bound event work.
- Excluding initial supports only gathers new-impact candidates. Every accepted
  physical or correction path still needs the full finite penetration certificate.
- Numerical-zero activation is fixed at 1e-12 m, with a half-bound finite proximity
  search and quarter-bound conservative-advancement approach. Endpoint activation
  requires the actual collider/chunk/triangle feature and an outward bound on all
  remaining collider-point motion. Invalid/inconclusive queries fail; they are never
  interpreted as empty terrain. Split corrections cannot advance physical time.
- Existing intersections retain their manifolds. Nearly parallel faces can amplify
  clipping roundoff into a false normal gap. A captured wheel fixture now verifies
  bounded movement within the clipped polygon to find a true finite opposing point;
  the gap is neither clamped nor granted a larger activation tolerance.
- Small contact Newton systems use bounded dense matrices through 128 scalar rows.
  Regularized proposals precede rank-revealing proposals when needed, with up to
  32 backtracking samples and a 16-residual nonmonotone window. Friction-mode changes
  reset the window. The original final physical residual remains the convergence
  condition. Larger contact systems retain the previous reduced/Krylov routes.
- Analytic translation, gravity reintegration, rotation, finite holes, floor plus
  new wall, event exhaustion, command rollback and repeatability tests pass.
- The authored car's first cold impact (5 cm initial clearance, -4 m/s root velocity,
  gravity and authored idle drives) completes the full 60 Hz tick after retries to
  four subdivisions. Repeated completed snapshots match exactly. Independent final
  vertex penetration and whole accepted-path bounds pass. This is one impact tick,
  not a sustained drop, driving, streaming or a performance acceptance run.

The [contact-release checkpoint](performance-results/2026-09-10-contact-release/README.md)
then removes finite support forces from initially separating points and reintegrates
candidate normal-velocity reversals. Analytic vertical launch/return tests pass at
all four substep settings, including return within one tick. All 95 physics tests
pass; the first cold-car hash remains unchanged. This does not certify general
nonlinear/rotational dense trajectories.

## Active continuation: [contact kinematics](performance-results/2026-09-11-contact-kinematics/README.md)
and [coupled contact search](performance-results/2026-09-11-coupled-search/README.md).
Latest retained work: [retry diagnostics, interval reuse and small-impact
continuation](performance-results/2026-09-11-retry-reuse/README.md). The application
still uses the original GPU solver; the CPU experiment is not a selectable complete
backend. Unchanged-state reuse removes repeated geometry/model preparation without
changing the validated replay state. A captured 50-row impact now passes its
original laws and independent reference. The required event-resolved drop still
fails at tick 17, with EventSearch now rejecting all four attempts. The endpoint
policy still fails at tick 20 (tick 11 with eight fixed subdivisions), and the
third 240-row runtime impact remains failing. Native serial tests on Apple M1 Pro /
Metal: physics 135 passed / 4 failed; app 739 passed / 1 failed / 6 ignored; GPU
110 passed / 11 failed / 1 ignored. The twelve app/GPU failures were already present.
Clippy and formatting pass. All complete physical/performance milestones remain open.

The subsequent central-path Rust prototype was rejected and removed: the later
240-row fixture still fails, and the third fixture misses the independent velocity
comparison despite meeting the contact residual. Its archived source and independent
215-direction NumPy calculation are recorded in the retry-reuse checkpoint. Do not
treat the independent calculation as a completed runtime solver.

## Immediate work

1. Resolve the exact `third_endpoint_impact.ron` contact problem without changing
   the normal/friction/rolling laws, 1e-9 residual, 256-iteration budget, or 128-row
   dense-response boundary. Retain independently failed candidate searches. Two
   earlier captures pass, and the later one's reconstructed velocity agrees with
   an independent full NumPy SVD reference within 2.76e-11. The third fixture
   now has an independently traced, fully validated solution; finding it within
   the runtime budget remains open. That is algebra proof
   for one impact, not sustained motion or performance acceptance.
2. Resolve the original event policy's now separately diagnosed EventSearch
   failure in all four retries at tick 17; it does not use the experimental slow endpoint/recovery policy. Keep both
   policies' failures visible. Increasing subdivision count alone fails: fixed
   eight subdivisions encounter a different unresolved impact at tick 11.
3. Complete 120 external ticks and fixed 1/2/4/8 comparisons, including independent
   5 mm maximum / 2 mm settling penetration, joint errors, per-tick repeated hashes,
   actuator/time accounting and unchanged publication after failed attempts.
   Then execute authored driving and the common complete-world interface.

Earlier investigation checkpoints remain linked in the master plan. Their tick-6,
50, and 1024-trial diagnostic controls describe superseded experiments, not the
current runtime or acceptance policy. All integrated milestones remain open.

## Next bounded steps

1. **General contact release and re-impact:** the vertical analytic cases now pass.
   Prove mixed-support release, rotating-point re-entry and force/rotation reversals
   cannot hide an excursion. The current candidate uses an initial material-point
   row and linear endpoint velocity estimate, followed by reintegration. A midpoint
   generalized drift is not a complete dense trajectory of the continuous force
   equation. Add a converged rotating-edge reference before broadening the API.
2. Extend the cold car through sustained drop/settling trajectories, independently
   checking 5 mm maximum and 2 mm settling penetration and exact reconstructed joint
   limits. Compare fixed 1/2/4/8 policies and repeated per-tick state hashes. Cover
   wheel/chassis mass ratios, active stops, seams, thin finite edges, grazing,
   simultaneous/repeated impacts, rebases and terrain edits. Keep failed fixtures.
3. Audit impact energy, static/kinetic transitions and rolling work against analytic
   or converged references. A converged algebra residual is not sufficient physical
   proof, especially for mixed separating/closing multi-contact impacts.
4. Add separately counted geometry-repair work and preserve partial query diagnostics
   on numerical errors. Counters currently retain query attempts and successfully
   returned work; a query that returns an error can lose its internal partial work.
   Keep CPU/GPU/transfer/publication latency separate in eventual complete-tick runs.
5. Finish verification of experimental split terrain recovery, then persistent contact validity/transactions and
   collision residency readiness. Then shared authored command resolution, actual
   driving and complete scene publication; body/body contacts and loops remain open.
6. Freeze matched foreground driven-car performance only after physical coverage
   passes. No first-impact test or cold fixed-pose result satisfies the ≤1 ms/10×
   driven-car gate. Continue all [master-plan stages](compiled-machine-dynamics.md).

[Catto's continuous collision presentation](https://box2d.org/files/ErinCatto_ContinuousCollision_GDC2013.pdf)
discusses substepping remaining time and restitution/ghost-contact pitfalls; it does
not provide a complete algorithm for this runtime. Nonmonotone residual history is
an experimental specialization of an established search idea, not a convergence
proof for these nonsmooth friction equations. Independent physical validation and
bounded unpublished failure remain required.

## Contact-model comparison boundary

A solver failure is not proof that the physical equations have no solution.
[MuJoCo's computation reference](https://mujoco.readthedocs.io/en/stable/computation/index.html#soft-contact-model)
describes a soft convex contact model with different physical equations from hard
Coulomb complementarity. Its convergence properties cannot simply be assigned to
this experiment's normal-clamp/disk equations. Any compliance experiment must state
the model change, connect it to material behavior, and independently pass impact,
friction and support bounds. A numerical shift must not become unreported physical
softness. No contact-model replacement has been accepted in this checkpoint.

## Actual driving inputs must preserve the authored program

Do not call the airborne test's uniform `target_speed = 8` overwrite an actual W
input. The app's `DriveSequencer` resolves authored states and per-wire reversal;
`GearboxRuntime` selects gas/electric gearing and disengages incompatible gas
contributions. They currently live in `mechanic-app/src/sequencer.rs` and output
GPU rows. A shared core command resolver or a frozen exact resolved-row replay is
needed for matched CPU/GPU driving. Current app captures retain row hashes, not
necessarily every row value; inspect capture payloads before relying on replay.
Avoid duplicating a partial controller implementation inside physics tests.

## Measured solver work still matters

A first supported car tick at one subdivision used 21 nonlinear force updates,
120 constraint sweeps/Newton proposals, 5,819 constraint H applications, 44 model
assemblies, 258 scalar rows, 160 interval envelopes and 296 path/query poses in the
initial diagnostic. No performance gate is implied. Analytic radial/orthogonal projection-block factors are now implemented and
verified in the articulated-factor checkpoint above, replacing local LU on the
bounded generalized Newton route. The full fixed-pose response falls to roughly
5.23 ms, still above the complete-tick target even before time integration and
publication. Dense midpoint assembly and the cold-contact event workload remain;
do not let component timing replace complete physical acceptance.
