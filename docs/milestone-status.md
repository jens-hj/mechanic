# Milestone status

The benchmark-first stop gate is active. A deliberately limited builder and
simulation prototype now exists as a pre-gate input, construction-graph, and
physics exercise; the production renderer has not been started.

## Implemented and exercised

- Rust 1.97.1 workspace with Bevy 0.19.0 and the wgpu version used by Bevy.
- Transactional `ConstructionGraph` and stable generational IDs.
- Quarter-metre dimensions, grid poses, 90-degree rotations, face geometry,
  touching-face weld validation, bearing anchor/axis validation, deletion, and
  pending-operation cancellation.
- Union-find weld compilation, static ground propagation, cuboid compound mass,
  centre of mass, full rotated inertia, and the parallel-axis theorem.
- Canonical bearing spanning forest, hard closure-edge list, collision
  suppression pairs, fixed/floating component roots, stable parent/traversal
  metadata, multi-ground-anchor closure selection, and invalid post-weld
  self-bearing rejection.
- Fixed-capacity GPU ABI, shared-device compute upload, struct-of-arrays body
  state, gravity/integration/damping, invalid-number flags, three GPU snapshot
  slots, timestamp queries, and fixed-size diagnostics readback.
- GPU Morton-code generation, bitonic sorting, Karras LBVH topology, bottom-up
  bounds, fixed-capacity pair traversal, OBB SAT, analytic ground contacts, and
  direct-bearing collision suppression.
- Persistent manifold caching with stale-row reclamation, active-contact
  compaction, warm starting, friction, fixed projected iterations, angular
  impulses, and world inverse-inertia precomputation.
- Root-plus-bearing-coordinate mechanism state, uploaded direct spatial inertia,
  articulated/bias/generalized-force scratch, mass-weighted bearing velocity
  projection, and parallel pointer-jump pose reconstruction. Gravity and contact
  changes are projected back into floating-root and joint velocities rather than
  being discarded when child poses are rebuilt.
- Twelve fixed loop-correction steps using matrix-free Jacobian products and an
  eight-round diagonally preconditioned CG solve per Newton step. Correction
  work is skipped through GPU indirect dispatch once the strict tolerance is met.
- Headless four-bar correctness cases exercised on Metal: a 0.001-radian
  coordinate perturbation converges to a 9-micrometre anchor residual with no
  failure flag, while a deliberately inconsistent closure reports explicit
  constraint non-convergence.
- Adapter-backed correctness cases for a grounded offset pendulum, a freely
  falling hinge, a double pendulum, child-body contact feedback, and the reported
  contact-supported unwelded four-block tower with a three-block welded arm. The
  tests also retain CPU-only WGSL and no-op pipeline-layout validation on hosts
  without a compute adapter.
- Per-stage GPU timestamp reporting for integration, mechanism FK, LBVH,
  narrowphase, contact solve, bearing validation, and snapshot publication.
- CPU OBB SAT/manifold reference and rational 60 Hz scheduler.
- Exact `dense_100k` and `loops_100k` generators and JSONL gate output.
- CPU-side WGSL validation plus macOS, Linux, and Windows compile/test CI.
- Pre-gate builder prototype with a fixed orbital camera target, CPU
  face picking, 0.5/1/2 m cube placement, two-object weld selection, two-step
  zero-volume bearing attachment, right-click deletion, validation feedback,
  combined committed part/bearing meshes, and geometry-shaped white/red ghosts.
- Space-controlled GPU physics preview using the shared Bevy device, fixed 60 Hz
  scheduling, failure-checked snapshots, and synchronous CPU mesh readback.
- Control blocks, a two-click connector tool, per-wire direction reversal, and
  per-bearing state programs edited in a panel opened with `E`. A state holds a
  target angle or speed and is entered by key or left after a dwell, so one
  block can steer, drive, pose, and run timed procedures at once. Angle targets
  ramp and settle within a torque budget; travel limits stop and hold. Programs
  are re-derived from the graph and written to the GPU live, so a running
  mechanism can be reprogrammed without recompiling. The joint x-ray gained
  spin arcs, limit ticks, and controller wires.

## Gate-blocking work

The scale gate remains blocked by profiling/optimization of the general
articulated path and by the integrated GPU-culling/indirect render proof with
constraint-preserving interpolation. Correctness work does not by itself prove
the 100,000-body performance target.

The current angular-contact build clears a 5-second warm-up plus 10-second dense
sample (`p95 GPU 14.013 ms`, `68.32 TPS`, zero flags), but the mandated 30-second
dense measurement has not completed successfully on this build. A one-second
100,000-body loop smoke with the new closure pass clears the raw budget
(`p95 GPU 9.098 ms`, `74.52 TPS`, zero residual and flags). The earlier complete
30-second loop sample (`p95 GPU 3.147 ms`) predates closure correction and is
retained only as historical evidence. None of these results opens the gate
because coverage is incomplete.

Scale scenarios therefore always report `kernel_coverage_complete: false`, even
when the existing integration pass is under budget. This is intentional: a
partial kernel cannot unlock editor work.

The next implementation slice is contraction-schedule optimization and a fresh
Metal profiling pass for both full physics windows, including nested-loop and
larger-perturbation cases, before starting the integrated render proof.
The prototype is not a gate result: its GPU snapshots are synchronously read
back to a CPU-built mesh rather than consumed by indirect draws, production
culling, interpolation, and a render benchmark.
Bearings still attach only a new cuboid and cannot connect two existing
compounds, so the UI cannot create closed bearing loops yet. Production renderer
and production-scale editor work remain stopped.
