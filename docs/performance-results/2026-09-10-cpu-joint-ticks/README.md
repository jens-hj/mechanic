# CPU joint ticks and finite terrain collision foundation

This checkpoint implements coupled joint ticks and real finite collision queries.
It does **not** integrate terrain into a moving CPU tick or pass any car, frame,
scale, or speedup gate. The app still uses the previous GPU runtime.

## Joint dynamics

`CpuJointMachine` advances at 60 Hz with authored suspension, bounded drives,
torque-speed envelopes, braking/back-driving, and predictive inelastic stops.
`CompiledDynamics.coordinate_velocities` removes per-tick coordinate searches.
`PreparedConstraints` binds the response to one immutable numerical factor and
reuses it when nonlinear iterations change only RHS targets. Dense response is
still limited to 128 scalar rows; larger systems use implicit factor applications.

The effective inertia is factored once per substep. Nonlinear implicit midpoint
updates include pose-dependent mass, gravity, inertial bias, spring force,
asymmetric damping, and progressive rubber bump force. The final velocity
residual evaluates the actual midpoint equation. Diagnostics count numerical
assemblies separately from factorization and factor applications; this reference
still performs repeated assemblies and allocations and has no performance claim.

Stop position correction is separate from physical velocity. Commands carry tick
and topology generation. Drive changes and candidate state publish together only
after validation. Failed numerical attempts restart from the same post-command
state at bounded 1/2/4/8 subdivisions; impulses are not duplicated on retry.
Loops are rejected explicitly by the joint-only API.

## Rejected integration candidate and physical checks

Backward Euler failed the one-second undamped spring comparison. The retained
`rejected-backward-euler-oscillation.log` and source archive record phase-space
errors 1.000/0.957/0.788/0.540 at 1/2/4/8 subdivisions. Eight subdivisions still
lost unacceptable oscillation amplitude, so that integration was replaced.

| Check | Current result |
| --- | --- |
| Midpoint undamped oscillator phase-space error, 1/2/4/8 subdivisions | 0.453291 / 0.116874 / 0.029404 / 0.007362 |
| Same oscillator relative phase-space amplitude drift | <1e-8 at every subdivision policy |
| Asymmetric free rotation versus RK4 reference, 1/2/4/8 subdivisions | 0.002590 / 0.000647 / 0.000162 / 0.0000403 |
| Loaded spring equilibrium position error, zero and 5 mm preload | 1.735e-18 m / 0 m |
| Progressive bump midpoint momentum residual | 1.157e-6 N·s |
| Bounded retry fixture | one rejected attempt, then two substeps; matches direct two-substep state exactly |
| Saved car, 120 airborne ticks with forward/reverse resolved commands | every repeated hash identical; final `54bf728a1a59d30e` |
| Saved car finite-floor normal response | 48 contacts; residual 8.585e-10; repeated impulses and external momentum balance pass |

The oscillator error compares analytic position/velocity phase space, normalized
by initial amplitude and amplitude × frequency. The rotation error is the larger
of orientation angle error and normalized velocity error. These tests establish
convergence; they do not select a production quality policy from frame time.

The saved car keeps its original 11 bodies, 16 generalized velocities, 78
colliders, suspension, and authored drive envelopes. Its airborne test replaces
resolved speed targets with +8 rad/s at tick 1 and −4 rad/s at tick 61; both runs
consume the same commands. Nonzero accumulated motor impulse is asserted.
This script is an integration regression, not the frozen application's driving
input or its terrain workload.

## Finite geometry and world publication

Core collision geometry retains the compiled box/convex decomposition. Triangle
clipping preserves finite edges and holes. Opposing contact points intersect all
convex faces, avoiding support points outside the actual collider on oblique
terrain. Continuous translational SAT handles clear endpoints across a fast
crossing. Conservative rigid sweeps bound translational and angular reach and
preserve unwrapped rotation: a full-turn bar/floor impact is detected even though
its endpoint poses match. Work exhaustion is an explicit unconverged result.
Articulated trajectory CCD remains open and must not use independent body sweeps.

The world spatial index now accepts actual mesh bounds and refits changed ancestor
paths. Ray, nearest, and bounds queries use those aggregates. A generated terrain
fixture has a vertex 2.384e-8 m outside its nominal owning box; the actual-bound
hierarchy preserves that finite mesh instead of clipping it or missing it.

`TerrainContactScene` shares `Arc<TerrainCollisionChunk>` data. Publication checks
BVH reachability, child bounds/masks, triangle coverage, vertices, active indices,
material weights, duplicates, and generations before changing the visible scene.
Unchanged chunks retain allocation and feature publication identities. The CPU
and GPU share normalized triangle material blending in `mechanic-world`; GPU ABI
and shaders are unchanged by this checkpoint.

Queries keep small finite triangles collidable without an area cutoff, reuse
chunk BVHs, and retain the existing four corner/curved crown
reduction, exact mixed-material separation, and unreduced overflow beyond 16
surface groups. Tests cover seams, material boundaries, generated queries versus
exhaustive clipping, rebases, changed active groups, malformed-publication
rollback, and stable contact identities. This is not residency readiness,
persistent impulse caching, compound collision broadphase, or sleeping.

## Verification

Exact commands, source overlay hashes, fixture hashes, and executable hashes are
recorded in `identity.json`. Apply `implementation-sources.tar.gz` to the retained
base `.physics-reference/replay-final-20260910/source`. Build any reference in an
isolated target directory, never the live workspace's target directory.

- 251 core tests, 39 CPU physics tests, and 91 world tests pass.
- Five saved-car algebra/integration tests pass, including the final command script.
- Workspace Clippy, formatting, and whitespace checks pass.
- Release `mechanic-app` builds; no app runtime/backend switch was introduced.
- The complete serial workspace run reports 739 app passes, the existing
  `ui::suspension::tests::leader_geometry_follows_projected_positions_without_remounting`
  failure, and six ignored app tests.
- The same run executes the complete GPU target: 110 pass, 11 fail, one ignored.
  Its eleven failing names exactly match the retained earlier GPU run in
  `2026-09-10-compiled-dynamics/gpu-tests.log`. WGSL validation passes.
- A focused serial terrain-residency hardware test passes and explicitly records
  **Apple M1 Pro / Metal**. No unavailable adapter was counted as a passing test.

The full workspace command used `--no-fail-fast -- --test-threads=1`, so the UI
failure did not prevent GPU/CPU/world coverage. Final focused checks cover the
subsequent bounds-query adjustment, small-triangle regression, and saved-car
command test. The workspace
remains failing; ignored tests are not included as passing evidence.

Next work is the collision-aware constrained tick: exact articulated swept paths,
refreshed contact blocks, static/kinetic friction, restitution, rolling resistance,
split recovery, and loops. Then run the actual car drop/drive and matched complete
physics measurements. The full remaining plan is in `docs/compiled-machine-dynamics.md`.
