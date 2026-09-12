# Compiled machine dynamics redesign

Status: baseline repair, compiled CPU free/joint dynamics, finite collision and
articulated sweeps, and coupled surface-response/substep experiments are implemented.
The redesign is **not complete**, and no application, vehicle, or scale acceptance
gate has passed. The existing GPU runtime remains authoritative. No superseded
runtime has been removed.

Current work: [CPU solver repair](cpu-solver-repair.md), explicitly requested so the
new solver can be used. Contact search, finite-bound arithmetic and position
recovery fixes are retained in the [CPU repair checkpoint](performance-results/2026-09-11-cpu-solver-repair/README.md).
Complete cold settling and app integration remain open. The latest tested
trajectory rejects the default drop at tick 17, the endpoint policy at tick 30,
and fixed eight substeps at tick 1. The support-transition algebra regression
still fails although its independent reference passes the original laws. This
candidate must not be promoted to the app. Earlier frozen results below retain
their original source identities.

Active continuation: [contact kinematics](performance-results/2026-09-11-contact-kinematics/README.md)
and [coupled contact search](performance-results/2026-09-11-coupled-search/README.md).
Earlier frozen work: [retry diagnostics, interval reuse and small-impact
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

Independent app rendering candidate: [immutable body mesh frustum culling](performance-results/2026-09-11-body-frustum/README.md).
The actual Bevy visibility regression and 60 rendering tests pass. Native visual
comparisons and matched performance measurements remain pending; this does not
enable the new physics backend or satisfy a rendering acceptance gate.

Earlier component checkpoint: [articulated inertia and analytic contact-block factors](performance-results/2026-09-11-articulated-factor/README.md).
A linear articulated factor is selectable in the CPU experiment. Matched component
runs reduce a 256-joint chain's factor-plus-32-impulse p95 from about 25.43 ms to
0.44 ms. Analytic local projection factors reduce the car's 240-row fixed-pose
contact experiment from about 9.73 ms to 5.23 ms. Both have **zero simulated time**
and satisfy no physics-tick gate. Dense force/Jacobian assembly remains.
Exact per-query midpoint reuse removes 38,880 duplicate pose reconstructions in
the failing drop without changing its failure or solver work. Focused physics has
112 passes and the still-required tick-6 settling failure. The checkpoint records
full verification, retained source, matched measurements and their limits.
The [continuation](contact-impact-next.md) tracks the cold-contact blocker and
remaining world architecture; all integrated performance gates remain open.

## Agent execution checkpoint

This document is the durable execution plan. Keep user-facing updates brief;
record implementation details, commands, failures, and acceptance evidence here
or in the linked experiment records. The user reaffirmed the full redesign on
2026-09-10; completing the algebra experiment does not complete that request.

All stages below remain open. Execute them in order, allowing prerequisite
interface work early. Do not replace a physical or integrated gate with a
microbenchmark, relax tolerances, or claim an unexecuted hardware test passed.

1. **Trustworthy application baseline.** Preserve the complete reference outside
   `target` before cleaning. Build the reference app from frozen sources and
   freeze the saved world, starting state, tick-indexed input, terrain, camera,
   resolution, presentation settings, adapter, and binary identity. Complete
   execution accounting within portable binding limits. Record per-stage CPU/GPU
   execution, transfers, publication and presentation latency, critical-path
   latency, allocations, scratch memory, contacts, iterations, and dispatches.
   Exit with reproducible foreground captures and automated detection of drops,
   backlog, incomplete execution, and invalid publications. Existing failing
   trajectories are regression inputs, not physical reference solutions.
2. **Complete CPU tick.** Introduce the minimum shared runtime interface now so
   benchmark and app can exercise the same implementation: state, tick-indexed
   commands, topology/terrain generations, completed snapshots, and diagnostics.
   Implement and verify free motion/orientation and velocity-dependent forces;
   impulses, bounded drives/back-driving, suspension and stops; loop constraints
   and rank deficiency; real manifolds, restitution, static/kinetic friction and
   rolling resistance; split position correction; then validation and bounded
   retries from unchanged pre-tick state. Exit only with analytic/converged
   physical regressions passing and repeatable completed-state hashes.
3. **Actual driven-car decision.** Use the frozen car, real finite terrain,
   materials, and driving script. Cover drops, mass ratios, steering under load,
   suspension limits, braking, and impacts. Compare 1/2/4/8 substeps at 60 Hz;
   choose by physical acceptance, not frame time. Verify small-W iterations do
   no repeated machine factor solves. Exit with foreground A/B/B/A evidence for
   both >=10x lower physics p95 and <=1 ms complete physics p95, including
   collision and validated publication. If it fails, retain evidence, profile
   the bottleneck, and keep this milestone open.
4. **Scaling, collision, residency, and sleeping.** Replace dense generalized
   factors with articulated/sparse operations, checked against the dense
   reference. Reuse symbolic topology and scratch, refresh numerical factors,
   retain the 128-row contact-response storage boundary, and test singular loops
   and changing contacts. Implement the collision/terrain/sleeping work below.
   Exit when 32 vehicles, a 256-joint machine with loops/contacts, and 10,000
   resting bodies through disturbance/waking each pass the <=4 ms physics p95
   and physical gates. Readiness holds count against acceptance.
5. **Application and persistent rendering.** Integrate completed snapshots and
   interpolation, persistent construction transforms, terrain buffers/indirect
   rendering, then frustum and conservative occlusion culling. Evaluate TAA and
   material caching separately against fixed visual comparisons. Bound queued
   rendering and separate presentation waits from publication. Exit first at
   native 60 FPS with equivalent perceived quality; pursue 120 FPS/8.33 ms p95
   as the stretch gate. Reject visually failing candidates.
6. **Portable GPU formulation.** Port the proven dynamics with independent small
   islands, staged large trees, deterministic ordering, and explicit global
   dispatch dependencies. Keep whole-scene backend selection at load time.
   Exit with same-backend repeatability, cross-backend physical-bound agreement,
   complete execution coverage, and serialized recorded hardware results.
7. **Final acceptance and retirement.** Repeat matched integrated runs and the
   unchanged dense_100k, loops_100k, and separate 1080p gates below. Run workspace
   tests, shader validation, Clippy, and formatting; report existing failures
   explicitly. Remove superseded paths only after replacements pass.

Each stage must retain reviewable source, exact reproducible commands, raw
measurements, physical errors, and an explicit pass/fail decision. Update this
checkpoint as work advances. The immediate next work is stage 1, then the
complete CPU tick and actual-car proof; no application speedup is established.

Stage 1 continuation (2026-09-10): the complete original reference is now copied
and hash-verified at `.physics-reference/2026-09-09/`, outside Cargo's cleanable
directory. The original application has also been rebuilt from that archive;
`app-rebuilt-identity.json` identifies its executable. It retains the historical
background capture protocol, so it is not the repaired foreground baseline.

The repaired baseline harness adds explicit foreground capture, isolated world
stores, source/binary/asset/world identities, per-dispatched-tick driving input,
initial camera/state identity, completed-state hashes, and terrain-readiness hold
events. Foreground warm-up renders the loaded state without advancing physics;
the unchanged script then includes 180 settling ticks before throttle. This is
an explicit workload change from historical wall-time warm-up and must be used
on both sides of future comparisons. The sequencer's automatic gearbox still
uses available published speeds; completed-state repeatability must be measured,
not inferred from deterministic input scheduling.

Use `scripts/build-performance-reference.py --output .physics-reference/NAME`
to copy sources and build that copy with locked offline dependencies. Use the
resulting executable, `identity.json`, and `source/crates/mechanic-app` asset root
with `scripts/run-background-capture.py --foreground --drive --identity ...`.
The launcher rejects lost focus, dimension/presentation changes, mismatched
assets or binaries, and missing per-tick scripted inputs. Passing this capture
protocol does not establish full kernel coverage or physical acceptance.

Two source-matched foreground car runs are now retained in the
[foreground baseline record](performance-results/2026-09-10-foreground-baseline/README.md).
They meet the focus/resolution capture protocol but run at approximately 14.2
FPS and 24 physics TPS, with 35.7-36.1 ms GPU physics p95, over 1,300 dropped
ticks, and over 350 terrain-readiness hold events each. The same initial state
and camera produce different completed-state hashes at tick 12, before throttle
or the first dropped tick. Initial resident terrain identity is not yet captured,
so the source of that divergence remains open. Every combined render GPU sample
is partial/reversed; affected spans are unavailable. No acceptance gate passed.

Immediate stage-1 follow-up: localize the repeatability divergence and complete
kernel coverage evidence. Capture closing/draining, fixed-simulated-duration
replay, and terrain/drive identities are now verified below. Preserve the old
solver's failures as replacement regressions; do not require it to become a
physically correct reference trajectory. Source-matched builds must use fresh
isolated target directories after the observed shared-cache stale-artifact failure.

Fixed-duration replay continuation: capture schema 3 closes submissions at a frame
boundary and records subsequent readbacks/publications in a drain phase. It
finishes only after the target tick is published and the readback ring is empty;
a ten-second drain deadline invalidates an incomplete capture. Renderer samples
from drain are separated from measurement, while replay throughput includes the
drain's wall time. Historical schema-2 evidence retains its archived tools.
`--foreground --drive --replay-ticks N` requests 1..3600 contiguous ticks at the
same 60 Hz external clock. Replay retains overdue ticks and counts real elapsed
time during terrain holds; slow runs therefore expose growing backlog instead
of silently changing the physical workload. Normal application scheduling is
unchanged. Initial packed-terrain and device-layout fingerprints, plus effective
drive-row hashes for each tick, help localize the earlier divergence.

The [fixed-replay record](performance-results/2026-09-10-fixed-replay/README.md)
retains two serial native foreground runs on Apple M1 Pro / Metal. Each submitted,
read back, and published exactly 1,800 ticks; the drain captured the final nine
readbacks with zero missing states or dropped ticks. Both still run at about
23.2 TPS, with 44.9–45.5 ms GPU physics p95 and backlog growing above 2,800 ticks.
Initial state, camera, packed terrain, GPU terrain layout, and all effective
drive-row hashes match. Completed states first diverge at actual tick 14 in
this pair, before throttle or terrain replacement. This narrows the investigation
without isolating an exact GPU operation. Full execution coverage remains open.
The changed simulated duration prevents using these numbers as a speedup against
the earlier wall-time pair. No physical or performance gate passed.

Verification passes Clippy, formatting, eight capture regressions, one terrain
identity test, one Metal residency test, and 21 Python tests. The workspace run
still fails the existing suspension UI test (739 app tests passed, one failed,
six ignored); the earlier intermittent Closure Lab failure remains recorded.
Continue toward the actual CPU tick/contact experiment; do not equate a complete
capture or the algebra microbenchmark with a completed physics milestone.

## Preserved reference

The dirty worktree was frozen before edits. The source archive, per-file hashes,
world hashes, commands, and reproduced failures are in
[the experiment record](performance-results/2026-09-10-compiled-dynamics/README.md).
The complete worktree and binaries are also retained locally under
`target/physics-reference/2026-09-09/`. That directory must be preserved before
running `cargo clean`; the smaller source archive in the experiment record
survives a clean. Existing app binaries are identified but their correspondence
to the frozen sources is unverified; the benchmark binary was rebuilt from the
frozen worktree before any source edits.

The new capture schema records a contiguous submission sequence separately from
the scheduler's possibly discontinuous tick number. Completions carry the
sequence from submission, including submissions before capture. Scheduler drops
are recorded as exact tick ranges. The summarizer validates each stream and
overlapping submission/completion identities; submitted gaps must match their
drop ranges. Boundary readbacks, pending submissions, completed-interval skipped
ticks, and drops during capture are separate quantities. Old captures cannot
prove this accounting and are explicitly rejected; no compatibility reader was
added. Asynchronous publication now waits for the oldest pending mapping rather
than allowing a later callback to overtake it.

GPU diagnostics grow from 32 to 48 bytes, with the per-body contact-count tail
moving from word 8 to word 12. Rust and WGSL change together. Device-written
stage markers and integration, bearing-validation, and published-body counters
replace inference from scenario names. Stage entry is only partial evidence:
the fused contact kernel already consumes the portable eight-storage-buffer
limit and has no diagnostic binding. Internal tree/closure kernels are also not
individually instrumented. `kernel_coverage_complete` therefore remains false
for all scenarios, including smoke; the benchmark reports measurements but
returns a failed gate until complete execution coverage can be proven. This
deliberately removes the old small-scenario allowlist. Instrumentation costs are
included in candidate timings; old and instrumented timings are not a solver A/B.
Timestamp readback accounting now includes all 28 returned timestamp words.

## CPU experiment

`CompiledCreation::dynamics` contains direct body/joint lookups, component ranges,
forward and reverse traversals, local spatial inertia, generalized velocity
ranges, and a compressed elimination tree. Loop sparsity is represented by two
ancestor-chain heads per closure. Symbolic storage is linear; it does not expand
long paths or allocate a dense matrix for 100,000-body constructions.

`mechanic-physics` reconstructs tree poses and point Jacobians, assembles coupled
generalized inertia in double precision, projects uniform gravity, and factors
positive-definite effective dynamics. A caller can include an implicit diagonal
`dt*c + dt²*k`. The original fixed-pose benchmark still does not integrate ticks.
Velocity-dependent forces and free-motion reference ticks are now implemented
below; drive budgets, springs, and complete constrained ticks remain open.
Its dense generalized matrix is
explicitly limited to 512 velocities; this is a reference implementation, not the
eventual articulated factorization for large machines.

The constraint experiment constructs `W = J H⁻¹ Jᵀ` using factor solves. Through
128 scalar rows it retains W and changes only constraint-space values during
the repeated solve, reconstructing generalized motion once afterward. Above
128 rows it applies response through factor solves without allocating W.
Individual block diagonals are retained (each block is bounded to 128 rows), so
total block scratch is linear in the number of rows, with that fixed bound.
Bounded blocks use a deterministic projected gradient with a spectral-radius
bound; bilateral blocks use rank-revealing elimination and reject inconsistent
dependent equations. Friction supports a circular Coulomb disk with one supplied
coefficient; static/kinetic transitions and rolling resistance remain open.
Residuals are projected constraint residuals, not just absence of numeric flags.

The saved car fixture comes from the frozen world's generation 11. The
`compiled-response` benchmark uses its authored bind pose with synthetic
horizontal support probes. It times assembly, factorization, row construction,
and impulse solving, and checks repeated response hashes. It **does not** measure
collision detection, terrain, actual support manifolds, motion, commands, or
publication. Its timings cannot establish the 1 ms physics-tick gate or a car
speedup. The first variant and its non-converged result are retained in the
experiment record.

Run the algebra experiment with:

```sh
cargo run -p mechanic-bench --bin compiled-response --release --offline -- \
  crates/mechanic-bench/tests/fixtures/driven_car_instance.ron
```

An optional second argument bounds block sweeps (default 256). A failure emits
the measured record and exits unsuccessfully. The physical gate stays false
even when the algebra converges.

The formulation follows the established generalized-inertia and constraint
elimination descriptions in [Featherstone's spatial dynamics overview](https://royfeatherstone.org/spatial/v2/)
and [MuJoCo's computation documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html).
Substeps with persistent anchors remain an experiment motivated by
[Box2D's solver comparisons](https://box2d.org/posts/2024/02/solver2d/).
No external solver implementation was copied.

## CPU free-motion reference

The [free-motion evidence](performance-results/2026-09-10-cpu-free-motion/README.md)
records the next prerequisite for the constrained CPU runtime. `MachineDynamics`
now reconstructs COM velocities and computes inertial bias using a root-before-
child acceleration pass, including centripetal, Coriolis, and gyroscopic terms.
The saved car's rotational/suspension topology agrees with independently
differentiated body motion to a maximum scaled error of 5.44e-10. Runtime kernels
perform no finite differencing or topology searches.

`CpuFreeMotion` advances passive unbounded revolute trees at the external 60 Hz
clock. It accepts tick-indexed, topology-generation-tagged external impulses;
keeps candidates private; and publishes reconstructed snapshots only after
numerical validation. A rejected command or failed numerical step preserves the
last snapshot. It explicitly rejects authored drives, stops, suspension/linear
joints, and loops; it deliberately has no collision evaluation or terrain input.
It is not the common `PhysicsWorld` interface or the application's CPU backend.

This dense RK4 path is a convergence reference with fixed 1/2/4/8 subdivisions.
It performs four numerical evaluations/factorizations per substep, plus impulse
and final validation work. It does not implement the proposed once-per-substep
effective contact dynamics, bounded retries, contact residuals, or optimized
scratch reuse. Do not benchmark it as the actual-car tick candidate or silently
strip the car's unsupported authored features to make it load.

Analytic/conservation checks cover ballistic motion, principal-axis orientation,
off-centre impulse momentum, impulse transfer through the whole tree, asymmetric
torque-free rotation, fixed roots, exact tree anchors/axes in both traversal
directions, failed-publication rollback, and repeated completed-state hashes.
The 1/2/4/8-substep rotation error decreases by approximately 16× per halving.
The two-second floating-tree test retains anchor errors below 1e-12 m and axis
cross-product errors below 1e-12; these free-motion tests do not prove terrain
impact, loop, or driven-car bounds.

The joint tick implementation below now uses these primitives. Keep the RK4
reference for convergence checks. Stage 1's complete coverage and old GPU
repeatability localization remain open; no vehicle, rendering, scale, or speedup
gate has passed.

## CPU joint ticks and next collision proof

`CpuJointMachine` now consumes the authored tree's drives, suspension, and travel
limits at 60 Hz. Tick/generation-tagged drive changes and external impulses commit
only with the snapshot. Numerical failure retries the same post-command state at
bounded 1/2/4/8 subdivisions; exhausted attempts preserve both state and drives.
This remains a **joint-only** runtime: no terrain/body contacts or closed loops
are omitted behind an application backend option.

One effective inertia factor per substep prepares reusable constraint response.
The nonlinear implicit midpoint iteration refreshes pose-dependent inertia,
gravity, gyroscopic/Coriolis forces, and asymmetric suspension damping. Its
residual checks the full midpoint equation, not a lagged-force approximation.
Numerical assemblies and factor applications are counted separately; repeated
assemblies and allocations remain optimization work. Drives enforce physical
effort and torque-speed limits, including back-driving/stalling. Stop recovery
uses a separate position correction without adding physical velocity.

The first backward-Euler candidate failed the undamped oscillator comparison:
at eight subdivisions its one-second phase-space error was 0.540. Its failing
log and source are retained in
`performance-results/2026-09-10-cpu-joint-ticks/`. Implicit midpoint reduces that
error to 0.00736 and preserves oscillator amplitude to the test's 1e-8 bound.
Free asymmetric rotation agrees with the RK4 reference with errors
0.00259/0.000647/0.000162/0.0000403 at 1/2/4/8 subdivisions. These comparisons
establish convergence, **not a production substep choice**.

The saved car loads with its authored suspension and drive envelopes unchanged
and repeats 120 airborne joint ticks exactly, including deterministic forward
and reverse resolved speed commands. This is a physical integration check, not the
driven-terrain experiment or a performance gate. Next: finite triangle manifolds
through existing terrain BVHs, continuous collision handling, material friction
and rolling response, loop constraints, then the actual car drop/drive tests.

Finite collision primitives are now implemented and verified separately. Shared
`mechanic-core` geometry clips finite triangles against the compiled box/convex
decomposition, measures opposing points on the actual surfaces, and provides
continuous translational SAT and conservative rigid rotational sweeps. Unwrapped
rotation catches a full-turn impact with identical endpoint poses. Exhausted
sweep work reports non-convergence; it never reports separation by default.
These rigid sweeps **do not yet cover reconstructed articulated trajectories**.

`TerrainContactScene` consumes shared immutable world chunks, validates complete
BVHs/materials/active triangle coverage before atomic generation publication,
and reuses the existing world hierarchy with incremental actual-bound refits.
The generated-mesh test found a vertex 2.384e-8 m outside nominal node bounds;
using actual mesh bounds prevents a seam miss without modifying terrain geometry.
CPU and GPU now call the same normalized triangle material blending function.
Manifold reduction retains the existing four corners, curved crown, material
boundaries, and unreduced overflow beyond 16 surface groups. Contact identities
include topology, chunk geometry, and per-chunk publication generations.

Tests cover finite edges/holes, fast translation, full-turn rigid impacts,
generated BVH versus exhaustive queries, rebases, active-group changes, failed
publication rollback, and actual finite supports entering the coupled response
solve. The saved 11-body/78-collider car produces 48 retained contacts on a finite
two-triangle floor; its normal response repeats and balances external momentum
with residual 8.585e-10. This is a fixed-pose normal response check, **not a car
drop, full friction solve, collision-aware tick, or performance measurement**.

Current [joint/collision evidence](performance-results/2026-09-10-cpu-joint-ticks/README.md)
retains sources, commands, rejected integration evidence, and verification results.
The release app builds. Workspace tests still fail the existing suspension UI
test and the same 11 old GPU regressions; no failing gate has been marked green.

### Next implementation checkpoint

Ongoing after the archived foundation: pose-only reconstruction, exact
`MachineMotion` trajectories, and articulated terrain sweeps are implemented.
A full-turn hinged bar hits at fraction 0.0732201842 despite clear endpoint poses;
the saved car's sampled point speeds stay within the derived ancestor/suspension
bounds (largest ratio 0.617). No inertia is assembled during these queries.
Static/kinetic friction and rolling rows now form coupled manifold blocks, with
static breakaway determined only after the static solve converges.

The **240-row / 12-manifold** fixed-pose car surface response now passes the
unchanged 256-sweep / 1e-8 bound: residual 6.237585133123822e-9 in 172 sweeps.
Safeguarded, adaptively shifted matrix-free Newton proposals resolve nearly
coincident wheel support rows while acceptance checks the unshifted physical
contact equations. Work includes 4,706 dynamics-factor applications, 1,529 Krylov
applications, 8,256 small preconditioner factorizations, and 872 line-search
residual evaluations; this is not a performance gate. Repeated full impulses and
motion match exactly, and independent momentum, support, friction, rolling, and
restitution checks pass. Core 251, world 91, physics 54, saved-car 6 tests and
Clippy/formatting pass. The earlier full workspace's 11 GPU / 1 UI failures remain.

Sources, exact identities, passing tests, and rejected candidates are archived in
[the articulated contact checkpoint](performance-results/2026-09-10-articulated-contact-steps/README.md).
The complete moving-tick milestone remains open. Next: finite near-contact
queries for continuous impact handling, then refreshed contacts in the coupled
implicit tick, certified trajectories, split recovery, and publication tests.

1. Extract pose reconstruction from mass assembly so continuous queries can
   follow exact generalized-coordinate trajectories without assembling inertia.
   Bound swept reach through every ancestor, including suspension extension and
   unwrapped root/joint rotation; do not sweep independent endpoint body poses.
2. Integrate refreshed finite manifolds into the same once-per-substep effective
   response as drives/stops. Flatten all block rows explicitly, preserving the
   128-row dense limit and implicit response above it. Include friction state,
   restitution, rolling response, and split position correction.
3. Exercise wheel drop, chassis/wheel mass ratios, seams, rotational impacts,
   retries, and exact repeated tick hashes. Retain last valid state on exhausted
   CCD/contact/force convergence bounds. Readiness holds remain explicit failures
   of performance acceptance, not a way to skip the physical workload.
4. Add closed-loop equations/rank handling and the common scene interface; only
   then benchmark collision + solving + validated publication on the real driven
   car against the frozen reference. Broadphase compounds, persistent contacts,
   scratch reuse, sleeping, rendering, GPU specialization, and integrated gates
   remain in the delivery list below.

### Surface substeps and measured solver cost (continuation)

Finite near-contact queries now preserve convex silhouette planes, holes, true
surface points, and signed separation. Internal implicit substeps accept real
manifold rows in the same factor response as suspension/drives/stops. Sustained
support, supported spring equilibrium, and static/kinetic friction at 1/2/4/8
subdivisions pass. The public CPU API remains joint-only pending continuous
activation, persistent-contact path validation, split terrain recovery, and
validated scene publication; body/body collision and loops are also still open.

A real finite-floor fixed-pose cost mode now retains raw samples and per-stage
CPU timings. Its first measured p95 was 31.591 ms. Reusing pose-local block scales
and Newton-local derivatives reduced initial repeated p95 to 24.559/24.687 ms
while retaining identical result bits. This remains much too slow and is not an
application comparison. The bounded generalized Newton reduction subsequently measured about 10.015 ms
p95, retaining all 240 rows; final paired B p95 was 10.071/10.025 ms against
A 31.481/31.727 ms. It still misses the complete-tick budget. A stricter
1e-9 cold-contact solve remains unconverged and must not publish. Explicit warm
starts now reuse impulses only within the same prepared response during nonlinear
force updates. Next is continuous contact activation, split terrain recovery,
persistent-contact validity, and complete tick publication with bounded retries. See the [surface-substep record](performance-results/2026-09-10-cpu-surface-substeps/README.md)
for sources, commands, rejected/retained candidates, and the next measurement.

### Continuous supported-path validation (continuation)

A finite-prism interval query now checks already touching geometry throughout an
articulated path. Directed interval dual certificates bound penetration and reject
unbounded face geometry. Deterministic subdivision reports excess depth or
unconverged work explicitly. Stationary/sliding support, finite edges, rebases,
and full-turn paths hidden at sampled poses are regression cases. The public runtime remains joint-only. A private supported-tick experiment now
checks physical and split joint-correction paths, refreshes contact manifolds each
substep, and retries from the unchanged post-command state. Its supported ticks
repeat at 1/2/4/8 subdivisions; terrain failure does not consume impulses/drive
changes or overwrite the completed snapshot. New impacts are explicitly held
until activation is implemented. Full collision ticks and all performance gates
remain open. Implementation details and remaining integration
steps are in [the moving-contact plan](compiled-contact-tick-next.md), with sources
and verification in [the supported-tick record](performance-results/2026-09-10-supported-ticks/README.md).

### Initial impacts and sustained supported car (continuation)

Initial closing contacts now receive an instantaneous coupled mass response with
active hard stops. The finite-time substep rebuilds its contact targets from
outgoing motion, applying restitution once and adding no penetration velocity bias.
The authored car completes 120 supported ticks at every fixed 1/2/4/8 policy with
exact per-tick repeatability and about 1 mm maximum flat-floor penetration. These
runs retain authored idle drives; they are not driving, cold drops, streamed terrain
or performance acceptance. New impacts during an interval still hold, and the
public runtime remains joint-only. See [the activation record](performance-results/2026-09-10-impact-activation/README.md)
and [the next event-integration steps](contact-impact-next.md).

The next [CCD checkpoint](performance-results/2026-09-10-linear-ccd/README.md)
adds exact linear intervals, cached endpoint poses and certified below-surface
separation. The [event checkpoint](performance-results/2026-09-10-impact-events/README.md)
then resolves new impacts through reintegrated private trials and completes their
remaining physical time. Analytic cube tests and the authored car's first cold
impact pass, with repeated car snapshot hash `273462a3b30cd1ca`. Bounded dense
Newton proposals, rank handling and nonmonotone line search are now used through
128 contact rows; larger contact routes remain unchanged. The car needs retries
to four substeps for this tick. This is physical first-impact evidence, not sustained
drop/driving or performance acceptance. Contact release/re-impact, force reversals,
complete diagnostics on query errors, recovery, loops, body/body collision, residency
and application integration remain open in the [continuation plan](contact-impact-next.md).

The [contact-release checkpoint](performance-results/2026-09-10-contact-release/README.md)
removes support forces from separating points and reintegrates candidate velocity
reversals. Analytic vertical release/return passes at 1/2/4/8 subdivisions, including
same-triangle re-impact within one tick; all 95 physics tests pass. General rotating
release and dense force trajectories remain open. The release app builds, and the
fresh serial Metal workspace run still has exactly 11 GPU failures and one UI failure.

## Delivery and acceptance still required

1. Finish baseline coverage instrumentation; freeze a source-matched driven-car
   foreground baseline with terrain, input, camera, native resolution, and
   presentation settings. Preserve car-drop, four-bar, and dense growth fixtures.
2. Complete the CPU tick experiment: coupled velocity-dependent forces, bounded
   drives and stops, implicit spring/damper forces, loop constraints, friction
   transitions, rolling resistance, real manifolds, split position correction,
   persistent contacts, and one/two/four/eight substep comparisons. Retry failed
   unpublished ticks at bounded finer substeps, retaining the last valid snapshot
   after final failure. Then prove the actual driven-car improvement.
3. Add compound broadphase/local BVHs, static/sleeping/moving partitions, moving
   refits, deterministic GPU radix rebuilds with scene-relative quantization,
   incremental terrain-chunk hierarchy/generations, swept residency including
   angular reach/preparation latency, readiness holds, and whole-island sleeping
   with deterministic waking. Preserve finite terrain, holes, seams, and materials.
4. Add `PhysicsWorld`, explicit whole-scene backend selection, tick-indexed drives
   and impulses, topology/terrain generations, completed snapshots, diagnostics,
   and application integration. Implement persistent construction/terrain rendering
   and interpolation. Test frustum then conservative occlusion culling, native TAA
   with motion/history invalidation and unfiltered UI, and bounded chunk-local
   unlit material caching. Keep normal contributions and verify seams/mips.
   Separate publication latency from presentation waits and bound queued work.
5. Implement the same formulation on portable GPU: independent small islands in
   workgroups, staged large trees, deterministic batches/reductions, no device-wide
   dependencies hidden behind workgroup barriers, and no per-tick backend migration.
6. Run matched foreground A/B/B/A acceptance and visual comparisons. Remove old
   paths only when replacements pass; retain failures with milestones open.

The external physics clock stays at 60 Hz. All physics targets include collision,
solving, and validated publication, with CPU/GPU work, transfers, and publication
latency reported separately:

| Workload | Required target |
| --- | --- |
| Existing driven car on streamed terrain | ≥10× improvement and ≤1 ms physics p95 |
| 32 interacting vehicles | ≤4 ms physics p95 |
| One 256-joint machine with contacts and representative loops | ≤4 ms physics p95 |
| 10,000 mostly resting bodies with disturbance/waking | ≤4 ms physics p95 |
| Native 4112×2524 application, equivalent perceived quality | 60 FPS intermediate; 120 FPS and ≤8.33 ms frame p95 stretch |

Physical proof must cover free motion, momentum, torque/back-driving, spring
equilibrium/damping/stops, static/kinetic friction, restitution and rolling
resistance against analytic or converged references. Preserve 0.01 mm anchor and
0.001° axis limits and applicable rigid-terrain impact limits of 5 mm maximum and
2 mm settling penetration. Include fast/rotational impacts, wheel/chassis mass
ratios, singular loops, seams, delays, origin rebases, edits, and waking. Repeated
fixed-input state hashes must match within each backend, with physical-bound
agreement across CPU and GPU. Algebra response hashes alone do not establish it.

Integrated acceptance requires sustained 60 TPS, zero dropped ticks, no growing
backlog, and zero invalid publications. Terrain readiness holds count as missed
performance acceptance. Native visual checks include driving, disocclusion, thin
geometry, transitions, close terrain, and editing for blur/ghosting/shimmer/popping.
`dense_100k` and `loops_100k` retain their exact active counts, sleeping restrictions,
complete execution coverage, 5-second warm-up/30-second measurement, 60 TPS and
≤16.67 ms GPU p95. The separate 1080p ≤5 ms integrated frame gate remains intact.
