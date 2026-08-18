# Mechanic

Mechanic is a benchmark-first construction sandbox foundation. Bevy 0.19 owns
the application and render integration. Rigid-body simulation is being built as
custom GPU compute; no third-party physics engine is used.

The repository currently implements the milestone-1 foundation:

- a transactional CPU `ConstructionGraph` with stable generational handles;
- exact weld-group compilation into compound bodies;
- cuboid mass, center-of-mass, and inertia calculation;
- canonical bearing spanning forests and explicit loop-closure edges;
- fixed-capacity GPU ABI, failure states, snapshot handles, GPU LBVH/SAT/contact
  kernels, and coordinate-space bearing FK;
- headless correctness and scale-gate benchmark entry points.

A construction and simulation prototype is available as an explicit pre-gate
exception. It exercises input, `ConstructionGraph` editing, and the existing GPU
physics pipeline. The production GPU-driven render path remains gated until both
100,000-body scenarios pass the physics and integrated-render budgets on the
reference M1 Pro.

## Builder and simulation prototype

```sh
cargo run -p mechanic-app
```

- Option/Alt + left-drag to orbit (middle-drag also works) and use the mouse
  wheel or trackpad scroll to zoom. Right-click removes the hovered cuboid after
  showing a red delete ghost.
- Press `1` for cuboids, then left-click the white cuboid ghost to place it.
- Press `2` for welds, then left-click two touching existing objects. The weld
  compiles both objects into one rigid compound without spawning geometry.
- Press `3` for bearings. The first left-click places a 0.25 m diameter visual
  cylinder 5 cm into and 5 cm out of a cuboid face; the second attaches the
  selected cuboid through that zero-volume physics joint.
- Press `Q`/`E` to select a 0.5 m, 1 m, or 2 m cube.
- Press `Escape` to cancel a pending weld or bearing selection.
- Press `Space` to compile the current construction and start simulation. Press
  `Space` again to stop and return to the editable construction.

Bearing placement requires a cuboid support; the ground can support standalone
cubes or be selected as one side of a weld. Valid placement ghosts are
transparent white. Invalid placement and deletion ghosts are transparent red
and match the geometry affected by the action. Simulation uses synchronous CPU
snapshot readback to update the prototype mesh, so it is not evidence for the
integrated-render gate.

## Commands

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p mechanic-app
cargo run -p mechanic-bench -- --scenario smoke
cargo run -p mechanic-bench --release -- --scenario four_bar
cargo run -p mechanic-bench --release -- --scenario invalid_loop
cargo run -p mechanic-bench --release -- --scenario dense_100k --seconds 30 --warmup 5
cargo run -p mechanic-bench --release -- --scenario loops_100k --seconds 30 --warmup 5
```

Benchmark output is machine-readable JSONL. The four-bar cases prove correction
and explicit rejection but do not unlock editor work. A scale gate only passes
when the requested scene has its exact required body count, complete production
kernel coverage, no dropped contacts or runtime failure, 60 TPS, and p95 tick
cost at most 16.67 ms.

## Capacity policy

Capacities are fixed at 131,072 bodies and 262,144 bearings. Capacity overflow,
invalid numeric state, device loss, and constraint non-convergence are terminal
for the current simulation load. The runtime never adapts solver quality or
publishes a failed tick.
