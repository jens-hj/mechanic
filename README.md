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

- Option/Alt + left-drag to orbit (middle-drag also works), Shift + left-drag
  to move the orbital centre across the ground plane, and use the mouse wheel
  or trackpad scroll to zoom. Right-click removes one hovered cuboid; hold and
  drag to preview a flat rectangular deletion plane. `Q` cycles its plane just
  like block placement, and releasing removes the selected cuboids atomically.
- Use the clickable hotbar at the bottom of the window or press `1` for Block,
  `2` for Bearing, `3` for Weld, and `4` for Hammer. Hover an icon to see its
  tool name. Tool selection persists when the simulation mode changes.
- With Block selected, click and release the white ghost to place one block, or
  hold and drag to preview a flat rectangular sheet of blocks. While dragging,
  press `Q` to cycle the `XZ`, `XY`, and `YZ` planes; release to place the whole
  sheet, or press `Escape`/right-click to cancel. A drag is limited to 4,096
  blocks and commits atomically. Blocks have a fixed 0.25 m cube size.
- With Weld selected, left-click two touching existing objects. The weld
  compiles both objects into one rigid compound without spawning geometry.
- With Bearing selected, the first left-click places a 0.25 m diameter visual
  cylinder 5 cm into and 5 cm out of a cuboid face without creating a block.
  Switch to Block and hover the bearing; it highlights when targeted, and a
  click attaches a new block through that zero-volume physics joint instead of
  welding it to the support. Holding and dragging from the highlighted bearing
  attaches an internally welded sheet through that one bearing. Right-click
  removes an unattached bearing.
- Press `P` to load the deterministic 20,000-part kinetic showcase. Loading is
  immediate from an empty editor; replacing an existing construction requires
  two consecutive presses. The showcase remains fully editable.
- Press `Escape` to cancel a pending weld or bearing selection.
- Press `Space` to compile the current construction and start simulation. Press
  `Space` again to stop and return to the editable construction.
- While simulating with Hammer selected, press and hold the left mouse button
  on a moving cuboid to charge a strike, then release to apply an impulse at
  that exact point along the camera ray. A quick click gives a light tap;
  charging for 1.5 seconds reaches maximum strength. Tools remain selectable in
  either mode, but build tools act only while building and Hammer acts only
  while simulating; `Space` is the only mode switch.

New blocks automatically weld to every face-touching block. This includes
blocks placed beside or on top of existing blocks and all neighbors inside a
dragged sheet. Blocks are not automatically welded to the ground; use the Weld
tool when a fixed ground connection is intended.

Bearing placement requires a cuboid support; the ground can support standalone
cubes or be selected as one side of a weld. Valid placement ghosts are
transparent white. Invalid placement and deletion ghosts are transparent red
and match the geometry affected by the action. To keep heavy scenes responsive,
simulation runs at most one fixed tick per rendered frame, updates only the
dynamic mesh at a throttled cadence, and hides bearing cylinders until build
mode resumes. It still uses synchronous CPU snapshot readback, so it is not
evidence for the integrated-render gate.

The showcase disables contact between bodies in its single articulated
mechanism so densely packed rope branches cannot feed collision energy into one
another. Ground contact and collisions with the separate loose target blocks
remain enabled. Other editor constructions keep mechanism self-collision on.

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
