# Finite supported-path validation

Verified geometry checkpoint; complete terrain ticks and every physics/application
performance acceptance gate remain open. Sources are frozen before subsequent
substep/publication integration work.

`ContactPolytope::triangle_penetration_bound` encloses a convex displacement
interval and intersects its bounds with a finite triangle's normal prism. Weak
duality certificates bound depth without assuming that a numerically incomplete
set of primal vertices gives an optimum. Directed scalar intervals enclose the
dual offset/residual and geometric arithmetic. Six axis certificates establish
boundedness first. Singular proposals can only weaken the bound. Unbounded face
data, overflow, or uncertifiable inputs fail explicitly.

The envelope contains both the represented faces and vertices, including their
small f32 discrepancies. It never changes physical contact geometry or generates
support impulses. Finite prism planes include roundoff/nonorthogonality padding.
Compiled collider radii now enclose both representations; articulated motion-bound
arithmetic rounds outward and preserves exact zero motion.

`TerrainContactScene::validate_penetration` visits swept hierarchy candidates,
reconstructs midpoint poses, and subdivides inconclusive intervals deterministically.
It checks existing supports instead of skipping zero-gap triangle pairs. Work and
progress bounds return explicit failure; accepted left intervals do not let their
pending siblings exceed the budget. Queries retain topology/terrain generations
and actual candidate, pose, envelope, and certified-interval counts. Unevaluated
intervals have absent depth diagnostics, not invented zero values.

Tests cover analytic box depths/reach, oblique geometry, finite edges/holes,
near-singular planes, unbounded faces despite finite vertices, representation
disagreement, overflow, rebasing, stationary support, sliding subdivision, and a
full rotation whose start/mid/end look safe. Retained manifold corners can miss
the deepest feature; their depth is an observed witness only. The full-turn path
is refused conservatively, sometimes as unresolved work near the depth threshold.
No sampled depth is used to certify an unsampled interval.

The actual saved car's 0.2 mm root translation over its finite-floor support is
bounded at a 2 mm limit: 136 candidate triangles, 136 envelopes, 136 certified
intervals, identical repeated outcomes/work. This query has no force integration,
simulated-duration throughput measurement, or publication; it is not a car tick gate.
The next step is passing actual physical/correction displacements from unpublished
substeps into this query, then first-impact activation and split terrain recovery.

Validation:

- Current CPU packages: core 260, physics 67, world 91, saved-car 7 and fixture 5 pass.
- Workspace Clippy, formatting, whitespace checks, and release app build pass.
- Serial full workspace tests on Apple M1 Pro / Metal retain exactly the same
  11 GPU failures and one UI failure. GPU has 110 passes; app has 739 passes.
  The full run preceded the last CPU-only overflow/outward-motion hardening and
  saved-car query test; CPU packages were rerun afterwards.
- Adapter-labelled adopted-terrain residency and WGSL validation pass. No green
  workspace or unavailable-adapter pass is claimed.

`identity.json` records source, fixture, executable, and log hashes and commands.
`implementation-sources.tar.gz` overlays the frozen replay reference. Mathematical
notes and remaining work are in [the moving-contact plan](../../compiled-contact-tick-next.md).
