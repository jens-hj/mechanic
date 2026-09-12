# CPU free-motion foundation

The CPU experiment now advances free bodies and passive revolute trees through
completed 60 Hz ticks. Analytic, conservation, rollback, and repeatability checks
pass. This is a reference foundation, not the actual-car contact solver or an
authoritative scene backend. No optimization or integrated acceptance gate passed.

## Implementation

`MachineDynamics::body_motions` reconstructs COM velocities from the compiled
generalized rows. `inertial_bias` computes the velocity-dependent term in
`H v_dot = force - C` with an ordered tree acceleration pass. It includes root
centripetal acceleration, transported parent acceleration, relative-joint
Coriolis/centripetal acceleration, and world-space gyroscopic torque. Generalized
force projection reuses the pose Jacobians. It does not finite-difference poses
or search for topology during evaluation.

The inertia/bias decomposition and use of body motion to compute energy/momentum
follow established rigid-body dynamics described in
[Featherstone's dynamics documentation](https://royfeatherstone.org/spatial/v2/).
The implementation was derived for Mechanic's world-space velocity convention;
no external solver source was copied.

`MachineState`, `ExternalImpulse`, and `CpuSnapshot` provide explicit coordinate,
command, and completed-state data. `CpuFreeMotion` owns immutable compiled
topology and the last completed snapshot. Commands identify the next tick and
topology generation, apply once before internal integration, and reject stale
identities or invalid body/impulse values. Candidate state remains private until
final pose reconstruction, finite-motion checks, and inertia factorization pass.
Any failure leaves the prior tick and state untouched. Snapshot hashing uses a
fixed byte/component order and includes tick, generation, poses, coordinates,
and generalized velocities.

The integrator uses dense RK4 with a fixed choice of 1, 2, 4, or 8 subdivisions
of the unchanged 1/60-second tick. It performs four derivative/factor evaluations
per substep, plus impulse preparation and final validation. It is intentionally
a convergence reference for future constrained integration. It does not satisfy
the planned once-per-substep effective contact factorization or scratch-reuse
goals. No performance measurements are claimed for it.

Only free bodies and passive unbounded revolute trees are supported by this tick
API. It rejects authored drives/stops, linear/suspension joints, and loops. It
has no terrain input or collision detection; the free-motion name and contract
make this explicit. The saved car cannot be loaded into this tick path by
discarding its authored physical features. The broader algebra methods still
support mixed joint topology for the force/Jacobian checks below.

## Physical evidence

| Check | Recorded result |
| --- | --- |
| One-second ballistic position error | 6.024e-15 m |
| Two-second asymmetric free rotation, relative conservation error at 1/2/4/8 substeps | 9.644e-7 / 6.103e-8 / 3.854e-9 / 2.424e-10 |
| Two-second floating tree, relative linear momentum error | 8.296e-12 |
| Same tree, relative angular momentum error | 2.499e-11 |
| Same tree, relative kinetic energy error | 2.512e-12 |
| Tree anchor error and axis cross-product magnitude at every tick | <1e-12 m / <1e-12 |
| Saved car bias versus independently differentiated COM/angular motion | maximum scaled error 5.438e-10 |
| Repeated 60-tick fixed-input tree run | every hash/state identical; final hash `05ebcb7b614803fb` |

The rotation error is the larger of relative angular-momentum and energy errors.
It decreases approximately 16× per timestep halving. Tree conservation uses four
substeps and exercises both directions of the compiled bearing traversal; the
test asserts that the reversed fixture actually uses reverse traversal. Both
cases produce the stated errors. Anchor and axis checks exceed the required
joint precision in these free-motion fixtures, but establish no loop, contact,
or driven-machine bound.

Other checks verify principal-axis orientation against an analytic rotation,
off-centre impulse linear/angular momentum for all four subdivision policies,
impulse transfer from a child through the whole machine, a stationary anchored
root with evolving joints, and transactional rollback after both bad commands
and finite input that overflows during dynamics evaluation. A valid command can
then retry the same external tick without a duplicated impulse. This is caller-
initiated retry proof, not the proposed automatic finer-substep failure policy.

The saved-car test uses the original authored fixture, 11 bodies and 16 generalized
velocities, mixed rotational/suspension coordinates, nonzero pose and velocity,
and rotated roots. Central differences of reconstructed body velocities provide
independent acceleration estimates; Newton-Euler forces are projected back into
every generalized column and compared to analytic bias. Error is scaled by
`max(1, abs(expected))`. This tests inertial terms, not suspension spring forces,
support, terrain collision, integration of that car, or penetration.

## Reproduction and verification

`identity.json` records source hashes, test executable hashes, the pinned toolchain,
and exact commands. `implementation-sources.tar.gz` retains the CPU crate and
modified car algebra fixture tests. Apply these sources to the complete retained
base `.physics-reference/replay-final-20260910/source`; its source tree hash is
recorded in the identity. No dependencies, GPU ABI, or shaders changed here.

```sh
cargo test -p mechanic-physics --offline -- --nocapture
cargo test -p mechanic-bench --bin compiled-response --offline -- --nocapture
cargo test -p mechanic-gpu --offline \
  physics_wgsl_parses_and_validates_without_a_gpu -- --test-threads=1
cargo clippy --workspace --all-targets --offline -- -D warnings
cargo fmt --all -- --check
git diff --check
```

All pass: 18 CPU tests (eight existing, ten new), three saved-car algebra tests
(one new), and WGSL validation. The pre-change eight-test log is retained.

`cargo test --workspace --offline -- --test-threads=1` was run with access to the
real Metal adapter and serialized hardware tests. Mosaic reports 18 passes; the
app reports 739 passes, one failure, and six ignored. The failure remains
`ui::suspension::tests::leader_geometry_follows_projected_positions_without_remounting`.
The previously intermittent Closure Lab failure did not recur, but its earlier
evidence remains. Cargo stops at the app, so this command does not establish a
passing complete workspace/GPU suite. The focused CPU, car algebra, and shader
checks above cover the changed code independently. No hardware performance or
GPU solver improvement is claimed.

The short serial
`adopted_terrain_residency_collides_without_uploading_chunks_again` GPU check
also passes. Its `adapter-check.log` explicitly records Apple M1 Pro / Metal;
an unavailable adapter was not counted as a pass.

## Next work

Reuse the verified motion/bias primitives in a constrained tick with once-per-
substep effective factors. Add bounded drives/back-driving, implicit suspension,
hard stops, real manifolds, loops/rank handling, restitution/friction/rolling
resistance, split position correction, and bounded unpublished retries. Keep
this RK4 path for convergence comparisons. Then connect the same complete
runtime to benchmarks and application through the common `PhysicsWorld` interface.

The saved-car physical and speedup decision remains open. Baseline execution
coverage and old GPU repeatability localization also remain open. Preserve the
earlier failed captures and exact 100,000-body gates; do not substitute the
free-motion or fixed-pose algebra tests for their requirements.
