# Compiled machine dynamics

Design of the reduced-coordinate machine model shared by the CPU solvers. How
the two CPU solvers use it, and how to select them, is in
[CPU physics](physics-cpu.md). The soft-step CPU solver is the default
application solver.

## Compiled schedule

`CompiledCreation::dynamics` holds direct body/joint lookups, component ranges,
forward and reverse traversals, local spatial inertia, generalized velocity
ranges, and a compressed elimination tree. Loop sparsity is two ancestor-chain
heads per closure. Symbolic storage is linear; it does not expand long paths or
allocate a dense matrix for 100,000-body constructions.

Floating roots own six velocity rows; every other row is a joint rate. Rows are
assigned in preorder, not body order.

Meshes between toothed parts are not joints: `CompiledCreation::gear_links`
carries each as two compound-local sides, and a solver adds one no-slip row per
mesh beside the closures ([gears](gears.md)). The two compounds of a mesh never
collide.

## Machine model (`mechanic-physics`)

`MachineDynamics` reconstructs tree poses and point Jacobians, assembles coupled
generalized inertia in double precision, projects uniform gravity, and computes
inertial bias (centripetal, Coriolis, gyroscopic) with a root-before-child
acceleration pass. Runtime kernels do no finite differencing or topology search.

Two factorizations solve the same effective dynamics `H = M + diag(d)`, where a
caller may add an implicit diagonal `dt·c + dt²·k` for springs and dampers:

- `DenseReference`: dense Cholesky, limited to 512 generalized velocities.
- `Articulated`: linear-work tree factor. On the 256-joint chain it reduced
  factor plus 32 impulse solves from about 25 ms to 0.44 ms p95. Jacobian and
  force assembly are still dense.

Constraint response is `W = J H⁻¹ Jᵀ`, built from factor solves. Up to 128
scalar rows the solver keeps W explicitly; above that, it applies response
through factor solves without allocating W.

## Joint forces

Joint laws use the same compiled parameters as the GPU backend:

- Suspension: spring with preload, asymmetric damping, progressive rubber bump
  stop (`PassiveForce`).
- Drives: speed and angle targets, with effort budgets recovered in SI from the
  compile-time normalization. The budget includes torque–speed fade and
  back-driving (`drive_target`, `drive_budget`).
- Travel limits: from the bearing kind intersected with drive angle limits.

## Collision geometry

`MachineCollisionGeometry` keeps each collider's exact box/convex decomposition
and material. A solid full cylinder uses its analytic prism instead of its
sixteen tangent boxes, which would duplicate every shared corner.

`TerrainContactScene` consumes the world's immutable terrain chunks and BVHs. It
validates complete BVHs, materials and active triangle coverage before atomic
generation publication, and refits with actual mesh bounds so seams are not
missed. Queries return `TerrainContact` points with:

- body and opposing body (terrain or another collider)
- world points, normal, depth and signed separation
- mixed material response `[μ static, μ kinetic, restitution, rolling]`
- a stable feature identity (topology, collider, triangle or collider, corner)

Manifold reduction keeps four outer corners plus a curved crown, keeps material
boundaries separate, and falls back to unreduced points beyond 16 surface groups.
Within one mechanism, colliders meet only where they were built apart
(`MachineCollisionGeometry::built_fits`): two colliders built touching, all of
their two bodies resting on the face they share, the two parts a bearing joins
and the meshing parts across a mesh are exempt, and a body pair exempt
throughout is dropped before its collider trees. A block a jointed body was
built clear of stops it. Separate mechanisms always collide. The GPU runtime
still exempts joined bodies whole (`collision_suppression`).

`TerrainContact::{point_row, angular_row}` turn a contact into generalized rows.
Both CPU solvers build their contact, friction and rolling rows from these.

Continuous primitives exist for the exact reference: translational SAT sweeps,
conservative rotational and articulated sweeps (`MachineMotion`), and interval
penetration certificates. Exhausted work reports non-convergence, never
separation.

## Reference integrators

- `CpuFreeMotion`: RK4 for passive unbounded revolute trees, with fixed 1/2/4/8
  subdivisions. It is the convergence reference for free rotation. The error
  shrinks about 16× per halving, and floating-tree anchors hold below 1e-12 m.
- `CpuJointMachine`: implicit midpoint with drives, suspension, stops and
  contacts, plus event search. It is described in [CPU physics](physics-cpu.md)
  as the exact reference solver.

## Algebra benchmark

The saved car fixture comes from the frozen world's generation 11.
`compiled-response` times assembly, factorization, row construction and impulse
solving on its bind pose:

```sh
cargo run -p mechanic-bench --bin compiled-response --release -- \
  crates/mechanic-bench/tests/fixtures/driven_car_instance.ron
```

It does not measure collision, motion, commands or publication.

## Long-term performance targets

All physics targets include collision, solving and publication at the 60 Hz
external clock:

| Workload | Target |
| --- | --- |
| Existing driven car on streamed terrain | ≤1 ms physics p95 |
| 32 interacting vehicles | ≤4 ms physics p95 |
| One 256-joint machine with contacts and loops | ≤4 ms physics p95 |
| 10,000 mostly resting bodies with disturbance/waking | ≤4 ms physics p95 |

Open architecture work towards those targets:

- exact loop constraints on the CPU (the soft-step solver closes loops with soft
  rows; see [CPU physics](physics-cpu.md#closure-rows))
- compound broadphase and sleeping islands
- persistent construction and terrain rendering with interpolation
- a portable GPU port of whichever CPU formulation proves out

## References

The formulation follows the generalized-inertia and constraint elimination
descriptions in [Featherstone's spatial dynamics overview](https://royfeatherstone.org/spatial/v2/)
and [MuJoCo's computation documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html).
The soft-step solver follows [Box2D's solver comparisons](https://box2d.org/posts/2024/02/solver2d/)
and Box2D v3's soft step. No external solver implementation was copied.
