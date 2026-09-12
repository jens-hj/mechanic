# Complete the moving contact tick

Historical execution notes after the CPU surface-substep experiment. First-impact
event handling has since been implemented; use [the current continuation plan](contact-impact-next.md)
for the remaining work. The public CPU runtime remains joint-only. Keep all
acceptance gates in [the master plan](compiled-machine-dynamics.md) unchanged.

## Boundary at the supported-tick checkpoint

- Exact generalized trajectories and bounded finite-triangle CCD exist.
- `TerrainContactScene::validate_penetration` now traverses swept candidates and
  certifies supported intervals with finite-prism envelopes. It returns explicit
  excess/unconverged outcomes, tagged generations, and bounded work counters.
- The private supported-tick experiment now refreshes manifolds each substep and
  validates actual physical displacements before accepting them. Split joint
  corrections are checked separately, with physical velocities restored even
  after failure. Its only commit point follows the entire external tick.
- New-contact sweeps run after supported-path validation, skip only actual
  initially touching pairs, and explicitly hold newly encountered impacts.
  This remains an incomplete private experiment; the public runtime stays joint-only.
- Real surface manifolds share the implicit substep factor with drives/stops and
  suspension, but only internal numerical tests supply those manifolds.
- Explicit `PreparedConstraints::solve_from` warm starts are revalidated. Joint
  nonlinear updates reuse them within the same immutable prepared response.
- The real 240-row cold car response passes 1e-8, including independent physical
  checks, but both tested Newton routes fail the stricter 1e-9 cold response.
  Preserve this case and check bounded substep retries; do not publish its failure.
- No complete terrain/body collision tick, persistent feature cache, split terrain
  recovery, loop runtime, or `PhysicsWorld` backend interface exists yet.

## Next implementation sequence

1. **Implemented for the private support experiment:** send the actual midpoint
   generalized rates × interval into `MachineMotion`, preserving unwrapped
   rotations. Check split joint-correction displacement separately. No endpoint
   quaternion interpolation or independent body paths are substituted.
2. Connect first-impact CCD to unpublished integration. Reintegrate a shortened
   interval from its unchanged start when an event changes velocity; bound events
   and count every rejected trial, factor, collision query, and retry. Whole-tick
   impulses and drive commands must not be consumed twice.
3. **Implemented as a validation boundary:** `sweep` retains its public zero-time
   touch behavior. The private new-contact sweep excludes an initially touching
   pair only after the same whole trajectory passed the finite depth bound.
   First impacts currently hold the tick, even below the allowed penetration
   depth; a floor support cannot hide a newly encountered wall. Actual impact
   activation is still required.
4. Refresh geometry during generalized split position correction, preserving
   physical velocity. Certify the correction path too: pushing out of a floor can
   otherwise cross a wall. Reconstruct and validate all final anchors, axes,
   finite contact depths, velocities, and generations before publication.
5. Add generation/frame-tagged contact impulse persistence. Prepare candidate
   caches separately and commit them only with a completed snapshot. Refresh
   normals and separations; invalidate changed geometry/materials/topology,
   incompatible frames, and separated features. A corner ordinal alone is not
   proof that the same geometric anchor survived.
6. Add compound/local collision hierarchy and body/body contacts, including the
   compiled collision-suppression rules. The app backend must not silently omit
   self-collision or inter-machine impacts.
7. Add loop closure rows, rank handling, and the common whole-scene CPU/GPU
   interface. Keep fixed 60 Hz commands and last-valid publication semantics.
8. Run the actual saved car through drop/driving/braking/steering/suspension and
   impacts at 1/2/4/8 substeps, with exact repeated hashes. Only then perform the
   complete collision/solve/publication timing decision against the frozen app.

## Supported-trajectory proof requires care

A negative SAT gap is **not** a bound on normal penetration at a finite triangle
edge. Whole-pair overlap can persist while a new part of the collider crosses an
edge. Sampling start/mid/end depths is also not a continuous certificate.

The implemented reference reconstructs interval midpoints and expands convex
half-spaces by the maximum point displacement. The envelope also encloses the
represented vertices, accounting for f32 face/vertex disagreement. Triangle-prism
side planes retain finite edges, with outward padding for represented-normal
nonorthogonality and subtraction error. These are validation envelopes only.

The bound uses nonnegative dual certificates rather than treating an incomplete
set of numerically feasible primal vertices as an optimum. For `Ax ≤ b`, any
nonnegative `λ` gives `c·x ≤ λ·b + |c − Aᵀλ|₁ |x|∞`. Directed f64 intervals enclose
the offset and residual. Six axis certificates first establish boundedness via
`|x|∞ ≤ B + ε |x|∞`, with `ε < 1`; subsequent coordinate bounds tighten the
radius. Singular/inaccurate three-normal solves only lose useful certificates.
No inverse or successful rank guess is treated as proof. The underlying weak
inequality is standard [LP duality, Boyd and Vandenberghe §5.2](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf);
the residual/radius certificate and finite-envelope specialization are this
implementation's construction.

The query subdivides deterministically and reports exhaustion when its work or
floating-point progress bound expires. The per-triangle limit also applies after
an accepted interval, when siblings remain pending. Unevaluated interval diagnostics
use `None`, not invented zero depths. Bounds do not generate supporting impulses.

Regressions cover analytic and oblique depths, finite edges/holes, near-singular
normals, unbounded plane data despite finite vertices, face/vertex disagreement,
rebases, stationary support, sliding subdivision, and full rotations hidden at
start/mid/end. A full-turn cube is refused even when its retained manifold corners
look shallow. Retained polygon corners do not necessarily contain the deepest
projected convex feature; their maximum depth is only an observed witness, never
a certificate. The interval query may therefore return `Unconverged` near the
threshold rather than localize a measured violation. Preserve this conservative
failure until a deeper-feature query improves it.

The scalar envelope certificate does not establish a complete application path:
force integration must provide its actual generalized displacement, correction
paths require separate checks, terrain residency must be complete, and independent
body/body constraints and loop handling remain absent. Measure interval cost on
the actual car before optimizing the reference enumeration or using it at scale.

## Required regressions

- Cold wheel drop and high chassis/wheel mass ratio, plus support with steering,
  drives, limits, asymmetric damping, restitution, and rolling resistance.
- Grazing/rotational impacts, full turns, finite edges/holes/seams, rebases, edits,
  and terrain-generation changes during preparation.
- Static/kinetic transition and positive-friction-work checks, independently of
  the projected residual. The 1e-8 projected-Krylov control failed such a check;
  the tighter 1e-9 cold solve did not converge within 256 sweeps.
- Exhausted collision/force/contact/correction bounds retain the prior completed
  state, commands, and cache. Retry matches a direct run at the accepted quality
  policy, with every impulse consumed once and no invalid intermediate publication.
- Terrain readiness holds count as missed performance acceptance. Sleeping must
  not remove active work from either 100,000-body gate.

## Supported-tick checkpoint

The private experiment publishes stationary supported ticks at 1/2/4/8 substeps
with exact repeated hashes. A deliberately limited interval budget forces three
rejected attempts before an eight-substep success; the result matches a direct
eight-substep run bit for bit and consumes its impulse once. Exhausted retries,
failed drive changes, wrong topology, and edits lifting supporting terrain retain
the prior completed state. Joint correction rejects an unsafe terrain path without
leaking scratch rates into physical velocity. Diagnostics count rejected paths,
initial-contact tests, SAT/envelope work, and impact holds. Core 260, physics 75,
world 91, saved-car 7, and fixture 5 pass. See the
[supported-tick evidence](performance-results/2026-09-10-supported-ticks/README.md).

These checks do not complete impact handling: a new shallow contact whose depth
bound passes must still hold, because its collision impulse has not been applied.
Cold drops, restitution timing, new contact anchors, generalized terrain recovery,
body/body collision, loops, residency readiness and common terrain-tagged snapshot
publication remain open. Keep the private terrain route out of the app until
those semantics are implemented and physically verified.

## Initial activation and sustained actual-car support

Initial closing contacts now use a coupled instantaneous mass solve with active
hard stops, followed by sustaining rows rebuilt from outgoing velocity. No position
or penetration bias enters that impulse. Restitution applies once; tests include
failed impulse rollback and stop reaction transmitted to the floating root.

The actual authored car now completes two simulated seconds at all fixed 1/2/4/8
policies with per-tick repeatable hashes and about 1 mm maximum flat-floor vertex
penetration. Authored idle drives remain active; this is not yet a driving test.
Core 260, physics 81, world 91 and saved-car bench 7 pass. The public backend is
still joint-only and new in-interval impacts still hold. See the
[activation evidence](performance-results/2026-09-10-impact-activation/README.md)
and the concrete [event-integration continuation](contact-impact-next.md).
