# Contact kinematics and activation — work in progress

This continues the articulated-factor checkpoint. No physics, app, or scale
acceptance gate has passed. The required 120-tick cold drop is still failing.
This record is being updated during implementation; it is not a final source or
benchmark identity. No performance A/B run has been made for these changes.

## Physical defects reproduced independently

- A frictionless rotating cube edge needs `J a + Jdot v = 0`. The old frozen-row
  force calculation supplied 40,195 N against the analytic 30,535.616 N, 31.63%
  too much. Initial point-acceleration bias fixes its small-step force limit.
- Maintained material points now use the reconstructed endpoint normal velocity
  during nonlinear force iteration. The initial analytic bias seeds the iteration;
  exact endpoint targets are refreshed and independently checked before acceptance.
  This is a discrete endpoint condition, not a proof of a continuous rotating
  force trajectory or of contact-position consistency. Friction and rolling
  rows still use the interval's initial Jacobian and original laws.
- Release detection and its bracket follow the same local material point through
  rotation. An independently sampled free rotating edge rises throughout the
  interval, while its old frozen row falsely reports a velocity reversal.
- Endpoint checks use a linear compiled tree velocity traversal, verified against
  every dense-Jacobian column in rotated/reversed/anchored mechanisms, nonzero COM
  offsets and the authored car; a 600-joint coaxial case also passes independently.
  Full midpoint force/inertia assembly remains dense.
- Core activation returned an empty half-window search before checking actual
  vertices inside the full existing 1e-12 m bound. A cube at 6e-13 through 9e-13 m
  reproduces this. The existing certified finite-vertex fallback now also runs on
  empty valid searches. No point is moved, no tolerance increases, and existing
  intersection manifolds remain unchanged. Outside-window geometry stays empty.

## Event and constraint work

A completely integrated and geometry-validated prefix can be saved while
localizing a longer trial's impact. If that longer reintegration predicts an
interior grazing event earlier than this known-clear prefix ends, its paths are
not nested and cannot establish an endpoint bracket. The saved prefix can then
commit with its original elapsed time and actuator impulses. Only one prefix is
stored, and topology/terrain/start state stay unchanged. The 128-trial bound and
unpublished external-tick rollback remain in force.

A captured 9.801 ms car interval completes in 22 trials, preserves full elapsed
time and torque budgets, repeats exactly and has independently checked finite
floor bounds. Finer 32/64/128 integrations show decreasing state differences;
this is a local replay, not sustained driving or a general ODE proof. Final
regression assertions and reference verification are still being completed.

The activation fix moves the default cold-drop failure from tick 12 to tick 17.
Four failed impact systems are retained. The final 55-row/16-coordinate system
stalls at approximately 1.53e-9 residual even at 4,096 sweeps, versus the unchanged
1e-9 requirement. Removing projection from Newton trial directions did not fix
it and was restored out of production source.

A loaded point had positive normal slack below 1e-9 while another coupled normal
violated 1e-9. The inactive-contact hypothesis refused to consider that point.
Allowing any strictly positive slack to propose the single bounded hypothesis
solves this captured system in 24 sweeps with residual 9.688234487667136e-11.
Every original normal/friction/rolling row is independently checked, and repeated
impulses and velocities match. Final acceptance tolerances, equations, iteration
budgets, and rejection of an invalid hypothesis are unchanged.

## Current verification and next work

`cargo test -p mechanic-core -p mechanic-physics --offline -- --test-threads=1`
passed 269 core tests. After the final hypothesis change, the physics-only suite
has 119 passes and the existing cold-settling failure. Core/physics all-target
Clippy passes with warnings denied. Full workspace/GPU/build verification for this
new work is pending; the preceding checkpoint's results are not new passes.

The latest required cold-drop run still fails at tick 17: 410 event trials across
retries, three event-work holds, 20 completed instantaneous solves of 21 attempts.
The captured final impact solve above is fixed, but this does not fix the whole
tick. Aggregated residual maxima include discarded retries and do not identify
the final cause. Trace the remaining final event/release sequence before further
numerical changes. Keep the full test active and preserve the last valid snapshot.
Continue the remaining complete-world, actual-driving, collision/residency,
rendering, GPU, and acceptance stages in the master plan.
