# Terrain response implementation status

The firm-ground, permanent deformation, and physical clump feature is incomplete.
None of its four milestone acceptance gates has passed.

## Contact investigation

`mechanic-gpu/src/terrain.rs` owns streamed collision chunks and a CPU triangle
contact helper. Repository call-site inspection found that `terrain_contacts`
is only called by its unit tests. App world simulations now upload published
terrain meshes directly to the GPU contact path. The previous per-part support
planes have been removed from world simulation; explicit benchmark and garage
plane APIs remain available.

The helper previously measured convex support with the absolute projection of
each vertex. For an asymmetric hull extending 0.25 m below its origin and 1 m
above it, this incorrectly reported ground contact with the origin 0.4 m above
the ground. Support now uses signed projection toward the terrain. Regression
coverage checks separation, overlap depth, and the rotated hull.

Chunk replacement now rejects generations older than or equal to either the
active or pending generation. Unloading invalidates the removed generation's
contacts and cancels its queued replacement. Focused tests cover these lifecycle
cases and retain the existing capacity-overflow and pinned-chunk checks.

The baseline solver added penetration recovery to restitution speed. This
produced upward velocity even with restitution explicitly zero. Terrain contacts
now retain their geometric depth separately and omit recovery from the physical
velocity target. Position-only scratch velocities resolve penetration, project
joint constraints, update root/joint poses, and reuse forward kinematics and
loop-closure projection. Physical body and generalized velocities are not
replaced by these correction velocities.

## GPU terrain contact path

`GpuPhysics::write_terrain_chunks` now uploads a complete collision scene into
an independent stackless BVH buffer. It rebases global coordinates in double
precision, includes only active triangle groups, checks geometry and device
capacity before replacing the scene, and clears cached contact generations on
replacement. A failed upload leaves the existing scene intact.

The GPU kernel reads current body poses, clips finite triangles against cuboid
and convex collider faces (including compiled cylinder pieces), reduces nearby triangles with identical contact responses to four outer
support points plus a crown point for curved patches, mixes the
triangle's terrain material response with construction material response, and
feeds the existing general and fused contact solvers. `mechanic-world` owns the
initial terrain friction/restitution tuning. No construction collider ABI change
is required. World simulation now uses this path before submitting ticks.

Metal regressions pass for boxes, convex parts, and cylinders on flat ground,
slopes, and vertical walls; finite edges with overlapping AABBs; failed upload
rollback; and contact execution in the fused articulated car route. These prove
contact generation and routing, not impact/settling acceptance.

Continuous SAT now detects translational sweeps using poses captured on the
GPU before each tick's integration. An initial rotational sweep now covers
unjointed bodies crossing between clear endpoint poses; see the scope below.
Cached terrain normal/friction impulses are capped by the current support
budget so a cached impact cannot launch a resting body on the following tick.

The repeated zero-restitution impact trace now records 1.042 mm maximum
penetration at 20 m/s, approximately 1 mm settling penetration, and no positive
outgoing velocity. All 180 samples have zero failure flags. See the
[recovery measurements and JSONL](performance-results/2026-09-07-terrain-contact-recovery/README.md)
and the retained
[baseline](performance-results/2026-09-07-terrain-contact-baseline/README.md).

Rigid-rock impact regressions cover small boxes that cross the surface entirely
in a tick, boxes, asymmetric convex parts, cylinders, the articulated car, and
free suspension assemblies with 1 and 65 joints. Both general and fused routes
are exercised. These fixtures pass 5 mm maximum and 2 mm settling penetration;
intentional restitution has a separate passing test. This does not prove the
remaining streaming/terrain acceptance cases.

## Required next work

Complete milestone 1 before introducing soil deformation: full rotational sweep coverage,
expanded articulated impacts on generated terrain, and additional
edge/streaming checks.
Verify the impact bounds on representative published terrain. The new bounded
serial position solve also needs profiling before asserting the performance gate.

Persistent soil state, pressure response, material transfers, runtime clumps,
atomic terrain/body publication, persistence format updates, the 256-clump
benchmark, and a visual demonstration have not been implemented.

## Verification on 2026-09-07

- `cargo test -p mechanic-gpu terrain::tests --lib`: six passed.
- `cargo fmt --all -- --check`: passed.
- `cargo clippy --workspace --all-targets -- -D warnings`: passed.
- `cargo test --workspace`: stopped in the app suite with 725 passed, one
  failed, and five ignored. The failure was
  `ui::suspension::tests::leader_geometry_follows_projected_positions_without_remounting`
  (missing leader marker at `(850, 450)`) in the pre-existing untracked suspension
  UI file. The terrain changes do not call that code.
- The sandboxed GPU suite could not acquire an adapter. Its apparent passes
  include tests that return early without hardware; they are not GPU proof.
- A retry of `cargo test -p mechanic-gpu --lib -- --nocapture` outside the
  sandbox acquired **Apple M1 Pro / Metal**. Failures observed include articulated
  car ground penetration, cylinder and pipe-bend annular collisions, and soft
  contact instability. These production solver tests do not use the modified
  terrain helper. That run completed with 85 passed and 10 failed. No solver
  acceptance or zero-failure-flags claim is made.
- After adding the GPU terrain path, `cargo test -p mechanic-gpu terrain_ --lib
  -- --nocapture` on **Apple M1 Pro / Metal** passed six tests; the diagnostic
  impact trace was ignored by that run and passed separately with `--ignored`.
- Workspace Clippy and formatting passed after the GPU terrain additions. The
  workspace test rerun reproduced the same single suspension UI failure.

Run logs are in `/tmp/mechanic-terrain-workspace-tests.log`,
`/tmp/mechanic-terrain-clippy.log`, and
`/tmp/mechanic-terrain-gpu-tests-unsandboxed.log`. No impact benchmark JSONL,
publication-latency measurement, or visual capture has been produced. The new
impact correctness JSONL above is separate from the requested performance gate.

After split recovery, the focused Metal terrain suite passed 16 tests (one
explicit diagnostic trace ignored by the normal run). The extended 65-joint
suspension test passed separately after that suite, and the impact trace passed
with all 180 samples recorded. See the recovery report for command details.

Workspace formatting and Clippy pass. The workspace test run still stops at
the same suspension UI failure. The existing plane-based
`plastic_rebounds_more_than_concrete` test was also rerun: it still fails with
exactly the baseline values (plastic 0.5122585 m, concrete 0.5 m). It already
failed in the original 85-pass/10-failure Metal run; the new terrain restitution
test passes. The full milestone gate therefore remains incomplete.

Coplanar support reduction now spans chunk seams and retains each selected
point's source triangle/corner for cache identity and geometry refresh. Distinct
contact responses remain separate. The local table holds 16 planes; additional
planes use the original contact emission path and retain the existing explicit
capacity failure flag. This bounds planar mesh contacts, not arbitrary curved
or highly fragmented terrain contacts. Geometry refresh marks vanished finite
triangle points inactive instead of reusing them as infinite-plane constraints.

The focused Metal terrain suite after reduction passed 17 tests with one ignored
trace (`/tmp/mechanic-terrain-reduction-tests.log`). Its new 20 m/s dense fixture
covers 1,152 triangles on a 5 cm grid split across two chunks, emits at most four
contacts, meets the 5 mm / 2 mm penetration limits for 120 ticks, and records no
positive rebound or failure flags. Separate follow-up assertions verify distinct
material responses (eight contacts) and 17 distinct planes exceeding the local
table (68 contacts), with zero failure flags
(`/tmp/mechanic-terrain-reduction-boundaries.log`). Adapter: Apple M1 Pro / Metal.
Workspace Clippy passed again (`/tmp/mechanic-terrain-reduction-clippy.log`).
These are correctness checks; release tick/publication latency remains unmeasured.

The post-reduction workspace rerun again stopped at the same suspension UI
leader-marker failure (`/tmp/mechanic-terrain-reduction-workspace.log`); it does
not establish a passing workspace gate. Formatting and diff whitespace checks
also passed.

## Dense curved terrain follow-up

A 5 cm curved mesh reproduced contact-capacity failure with the coplanar-only
reducer. Nearby normals now share a support group when their dot product exceeds
0.995 and their plane separation at the collider centre is under 2.5 cm.
Parallel distinct planes and different material responses remain separate. Each
retained point still uses its own source triangle normal, identity, and refreshed
geometry. Curved groups retain the highest point relative to the opposing
collider face in addition to the four outer points. The 16-group limit still
falls back to unreduced emission with explicit contact-capacity failure flags;
arbitrarily fragmented terrain has not been proved capacity-safe.

The curved fixture also exposed overcorrection from measuring depth against the
infinite plane through a finite triangle. Depth now uses the retained point and
the opposing collider face. The position-only correction still leaves physical
velocities untouched. Fixture penetration is independently measured by clipping
terrain triangles against the box footprint in box space, including polygon
edge intersections. It does not substitute a horizontal plane for curved soil.

The dense fixture now covers crowns and bowls, both horizontal and tilted by
20 degrees, at 20 m/s for 120 ticks. Its rebound assertion measures upward world
velocity under gravity; downhill motion can have a small outward component
relative to the initial slope normal without being upward rebound.

Inspection confirmed that `active_terrain` can contain replacement meshes before
their publication is complete. The app integration below follows
`active_terrain_index`, as `ActiveTerrainScene` does, to avoid simultaneously
uploading obsolete and hidden replacement chunks.

Dense fixture results and JSONL are saved in
[the dense-terrain report](performance-results/2026-09-07-dense-terrain-contacts/README.md).
Across flat, crown, bowl, and tilted fixtures, maximum penetration was 1.092 mm,
settling penetration at most 1.050 mm, and maximum positive world-Y velocity zero.
Each fixture maintained support for all 120 ticks with at most five contacts and
zero failure flags. Workspace Clippy, formatting, and whitespace checks passed.

Final focused command `cargo test -p mechanic-gpu terrain --lib -- --nocapture
--test-threads=1`: 18 passed, one diagnostic trace ignored, on Apple M1 Pro / Metal
(`/tmp/mechanic-curved-terrain-final-measurements.log`). The workspace rerun again
stopped at the pre-existing suspension UI leader-marker failure: 725 passed,
one failed, five ignored in the app suite
(`/tmp/mechanic-curved-terrain-workspace.log`). None of the four full milestone
gates is complete.

## App terrain publication

World simulation now uploads the published terrain collision cut before tick
submission. `WorldRuntime::physics_terrain` selects only spatial-index owners;
hidden replacements are excluded. `terrain_publication` caches the exact node
and generation list plus the floating origin, avoiding repeated uploads/cache
invalidation on unchanged frames. A replacement physics scene starts with no
publication key, so rebuilds and weld publications receive the current terrain
before their first tick. The world-only assembly and per-part ground-plane paths
have been removed. Benchmark and garage plane APIs remain unchanged.

Recovery pipelines and scratch buffers are prepared in the existing physics
worker. Geometry packing and upload currently run synchronously when the
published cut changes. Queue ordering puts their contact-cache invalidation
after old submitted ticks and before ticks using the new bindings. Upload errors
preserve the previous GPU scene/key and stop app tick submission. This does not
implement milestone 4's asynchronous terrain/body deformation transaction or
prove responsive rendering during large mesh uploads.

`cargo test -p mechanic-app real_gpu_terrain_publication -- --ignored --nocapture`
passed on **Apple M1 Pro / Metal** with zero failure flags. The regression submits
a 20 m/s impact, replaces terrain while that tick is still in flight, and verifies
the old snapshot, retirement without body reset, failed-upload rollback, and
origin rebasing. The test is explicitly ignored by ordinary app tests because it
requires a real adapter. Read-only tests verify hidden replacement filtering and
publication-key invalidation for remeshing, retirement, and origin changes.
Log: `/tmp/mechanic-app-terrain-gpu.log`.

The workspace rerun completed the app suite with 726 passed, one failed, and six
ignored; the failure remains the existing suspension UI leader-marker test
(`/tmp/mechanic-app-terrain-workspace.log`). Workspace Clippy, formatting, and
whitespace checks pass. Rotational CCD and the remaining milestone gates are
still incomplete; generated-world coverage is extended below.

## Production-mesher impact coverage

New regression fixtures use seed 91 and the production volumetric mesher, its
material weights, regular collision groups, and BVHs. Three surface fixtures
cover the spawn, transition region, and terrain 300 m from spawn; each uses
27 bricks and runs 120 ticks after a 20 m/s box impact. Generated cave wall and
ceiling fixtures run 10 ticks after a 20 m/s impact normal to their boundaries.
The penetration checker now rebases each chunk's global origin and excludes
inactive geometry when clipping the real triangles to the box footprint.

Surface maximum penetration is at most 1.725 mm; all three meet the 2 mm settling
bound and 1 mm/s upward-velocity limit. Cave maximum penetration is below 0.953 mm,
and the first contact tick leaves less than 5% of the initial kinetic energy,
including rotation. Cave settling is not asserted because gravity can pull the
body away from the boundary. All samples have zero failure flags on Apple M1 Pro
/ Metal. See the [report and JSONL](performance-results/2026-09-07-generated-terrain-contacts/README.md).

The initial cave-wall test incorrectly required nearly zero centre-of-mass
velocity along the wall normal. An off-centre contact produces supported pivoting;
that assertion was replaced by the penetration/contact and kinetic-energy checks.
No production solver changes were needed for these generated fixtures.

Workspace Clippy, formatting, and whitespace checks pass. The workspace rerun
still stops at the same suspension UI test: 726 passed, one failed, six ignored
in the app suite (`/tmp/mechanic-generated-terrain-workspace.log`). This adds
bounded generated-geometry coverage; it does not complete rotational CCD or the
remaining milestones.

Final focused command `cargo test -p mechanic-gpu terrain --lib -- --nocapture
--test-threads=1`: 20 passed, one diagnostic trace ignored, on Apple M1 Pro / Metal
(`/tmp/mechanic-generated-terrain-suite.log`).

## Initial rotational CCD

A new spinning-box regression reproduced a missed crossing: a 0.5 × 0.25 × 0.25 m
body rotating at 50 rad/s crosses the ground between two clear endpoint poses,
with vertex speeds below 16 m/s. The previous translational sweep emitted no
contacts. The new pass captures per-body sweep fractions, traverses the terrain
BVH, and advances conservatively using finite-triangle SAT separation and bounds
on translation and angular motion. Fractions are reduced atomically across a
body's colliders, then applied to poses before normal contact generation.
Physical velocities remain inputs to the existing impulse solver.

Fractional rotation reconstructs the normalized Euler root integration path;
normalizing a linear mix of endpoint quaternions would change its timing relative
to translation. A control-derived regression verifies consistent fractional time.
The BVH query intersects a swept sphere with an angularly expanded translational
AABB. Endpoint contacts use the existing discrete contact/recovery path. Zero-area
triangles are ignored consistently with ordinary terrain contacts.

The compile-time body mask excludes static and bearing-connected bodies from
independent pose clamping. Initial overlaps also retain discrete recovery. Thus
this does not yet cover articulated rotational CCD or a fast crossing that starts
in overlap. The bounded 256-iteration search can conservatively shorten unresolved
near-grazing motion. A clamped tick's remaining fraction is not re-integrated after
contact impulses. Additional collider-family rotational fixtures and release
performance measurement are still required.

The formerly missed crossing now produces two contacts, approximately 0.0023 mm
penetration, and 28.809 rad/s outgoing angular Z velocity. Kinetic energy, including
translation, is 42.682% of its initial value. Controls retain full rotation beside
a finite triangle with overlapping AABB, parallel to the terrain, and above a
zero-area triangle. See the [rotational report and JSONL](performance-results/2026-09-07-rotational-terrain-contact/README.md).

On Apple M1 Pro / Metal, the focused terrain suite passed 21 tests with one ignored
trace (`/tmp/mechanic-rotational-euler-suite.log`). The final timing/energy/control
regression passed separately (`/tmp/mechanic-rotational-controls.log`). App terrain
publication and in-flight replacement passed with the new sweep resources
(`/tmp/mechanic-rotational-app-publication.log`). Workspace Clippy, formatting,
and whitespace checks pass. The workspace test run still stops at the same
suspension UI failure: 726 passed, one failed, six ignored in the app suite
(`/tmp/mechanic-rotational-workspace.log`). Full milestone acceptance remains open.

## Suspension World performance follow-up

The subsequent suspension-car optimization adds a top-level chunk BVH,
worker-prepared terrain updates, generation/seam-aware packed geometry reuse,
incremental GPU allocations, and placement-only floating-origin updates.
App ticks wait for the current terrain cut; stale preparations and failed
validation cannot replace the accepted scene. Contact-cache invalidation remains
conservative for the complete cache. Recovery now compacts connected affected
contacts/joints and gates unnecessary work without lowering its iteration budgets.

The complete car/terrain driving gate remains unmet. The installed car overturns
in the scripted baseline and candidate sequences; zero kernel failure flags alone
are insufficient to establish stable driving. See the [suspension World performance
report](performance-results/2026-09-07-suspension-world/README.md) for release
measurements, timestamp limitations, publication regressions, and World capture
instructions. This work does not complete the firm-ground/deformation/clump or
larger scale milestones.
