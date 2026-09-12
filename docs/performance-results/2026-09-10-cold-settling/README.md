# Cold drop, timed stops and contact continuity

**Open milestone.** The required 120-tick cold-car drop/settling test remains active
and failing. No complete-world, driven-car, or performance gate passes. The app
still uses the existing GPU runtime. These are private CPU experiments.

## Retained changes

- Joint stops activate on arrival. Crossing/reversal times propose shorter force
  integrations from the unchanged state. An analytic floating suspension verifies
  arrival time, momentum exchange and remaining motion without position correction.
  This removes the captured tick-3 correction path that crossed the floor.
- Separating-point reversal proposals must at least halve a rejected interval.
  The previous secant ratios approached one and exhausted hundreds of trials.
- Rotational CCD supplements total-speed advancement with per-vertex fixed-axis
  quadratic certificates. Tree traversal computes signed spatial derivatives and
  conservative acceleration bounds, including ancestor rotation and translational
  Coriolis terms. Directed arithmetic evaluates g + u*t - A*t²/2; its concavity and
  positive endpoints prove separation along the defined generalized drift. Bounded
  bisection proposes a certified prefix. Whole-path penetration validation remains
  independent. This is not a dense trajectory certificate for the force ODE.
- Quaternion derivative arithmetic follows the pinned glam vector action. Cached
  start poses use the same root normalization as sampled poses; configuration
  validation still precedes indexed reconstruction. The original state is unchanged.
- Impact-time localization retains clear/crossing trial times while reintegrating
  from one start. A clear shortened trial may finish before the long trial's impact;
  blindly committing such prefixes exhausted 128 trials for an analytic body only
  10 nm above the floor. A fresh finite endpoint query can establish numerical-zero
  arrival on an independently validated clear path. The activation bound stays
  1e-12 m. Trial count stays 128 per nominal substep in the retained policy.
- Drive impulse diagnostics commit only with accepted intervals. Rejected trials,
  including endpoint-query/impact failures, cannot add drive impulse accounting.
  Query error counts and force/finite-constraint/impact residuals are separate.
  Internal partial traversal work on error is still not fully returned.
- A captured 15-row impact stalled near 1.73e-9 even after 8,192 iterations. One
  bounded inactive-contact hypothesis now solves it in 25 total iterations at
  about 2.01e-11. The hypothesis borrows H and copies the bounded response/layout;
  extra matrix storage and work are reported. It gets at most 32 remaining sweeps
  within the original 256-sweep budget, and cannot recurse. Acceptance checks every
  ORIGINAL contact row, cone and final factor-based velocity response. Rejected
  hypotheses retain the original iterate and modes. A captured rejection regression
  reproduces the original final impulses/velocities exactly (29 versus 20 sweeps).
- A nearly parallel wheel face amplified transform/clipping roundoff although an
  actual hull corner lay only 3.25e-14 m above the finite triangle. If face activation
  cannot supply bounded points, unchanged hull vertices can establish contact only
  when their measured gap meets the same bound and outward interval edge predicates
  certify their projection inside the finite triangle. No supporting plane is
  extended over holes; no gap is clamped. The four-corner reduction is retained.

## Controls and remaining failures

The archives and logs retain successive source states; names identify the stage,
not a claim that every intermediate candidate passed all tests.

| Control | Observed result |
| --- | --- |
| Original sustained cold drop | Tick 3: release refinements exhausted |
| Guaranteed release progress | Tick 3: joint correction crossed terrain |
| Timed joint stops | Tick 4: total-speed CCD exhausted near a parallel gap |
| Quadratic CCD | Tick 5: repeated tiny reintegrated clear prefixes |
| Bracketed localization plus fresh endpoints | Tick 6: production event budget exhausted |
| Endpoint-gap secant candidate | Tick 5; rejected and archived |
| 1,024-event diagnostic before contact trial | Tick 11: 15-row impact residual stalled |
| Unconditional inactive hypothesis | Tick 9: wrong hypothesis failed; rejected |
| Bounded hypothesis, discard on failed original validation | Tick 12: finite contact query failed |
| Actual-vertex activation, 1,024-event diagnostic | Tick 38: another coupled constraint solve fails |

The 1,024-event setting exists only in archived diagnostic fixtures. It is NOT the
retained production policy and does not satisfy an acceptance gate. More iterations
alone did not solve the captured impact. Rank-first Newton proposals also failed:
its current derivative had rank 12 with an incompatible residual. Dropping a trial
impulse guess alone could re-enter the same loaded-contact state; a bounded active
hypothesis with full original validation was required.

The required fixture is the authored car, idle authored drives, a finite 128 m
square Rock floor, 5 cm clearance, -4 m/s initial root vertical speed and gravity.
It requests 120 external 60 Hz ticks, independently checks every final collider
vertex against 5 mm maximum penetration, and checks the last 30 ticks against
2 mm settling penetration. Failure verifies retention of the prior completed
snapshot. It remains a required failure, not an ignored test or passing hold.

The 26-row remaining system also has an independent Python PGS probe. With the
recorded friction modes its residual is 3.40e-6 after 8,192 sweeps. Forcing the one
initially static point to kinetic friction reaches 1.20e-10, but that is a changed
mode assumption, NOT a valid breakaway proof. The static point was inside its
static cone in the unfinished original iterate. No forced mode change was retained.
The probe is reproducible with `python3 docs/performance-results/2026-09-10-cold-settling/contact-mode-probe.py`.

Final workspace verification: core 268 passed, physics 102 passed plus the active
cold-settling failure, world 91 passed, benchmark car/fixture tests 7/5 passed.
Metal GPU tests: 110 passed, the same 11 failed, one ignored. App tests: 739 passed,
the same UI failure, six ignored. WGSL validation passed. The actual adapter is
Apple M1 Pro / Metal. This is a failing workspace; the additional failure is the
new required cold-settling test, not a newly broken previously passing regression.

The release application builds with the current sources; final workspace Clippy,
formatting and diff whitespace checks pass. The first cold-impact tick completes
at one nominal substep on the first attempt, with two identical snapshot hashes
`47ca32e667997189`. This one-tick result is not sustained-drop or performance proof.

The final verification commands, source/binary identities and actual totals are in
`identity.json` and the final logs. Prior hardware failures are reported separately.
Continue through [the execution plan](../../contact-impact-next.md); all broader
[redesign gates](../../compiled-machine-dynamics.md) remain open.
