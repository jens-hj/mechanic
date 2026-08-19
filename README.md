# Mechanic

Mechanic is a benchmark-first construction sandbox foundation. Bevy 0.19 owns
the application and render integration. Rigid-body simulation is being built as
custom GPU compute; no third-party physics engine is used.

The repository currently implements the milestone-1 foundation:

- a transactional CPU `ConstructionGraph` with stable generational handles;
- exact weld-group compilation into compound bodies;
- exact cuboid and hollow-cylinder mass, center-of-mass, and inertia calculation;
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

- Press `?` to show or hide the controls and status overlay. It starts hidden
  so the construction view stays unobstructed.
- Option/Alt + left-drag to orbit (middle-drag also works), Shift + left-drag
  to move the orbital centre across the ground plane, and use the mouse wheel
  or trackpad scroll to zoom. Right-click removes one hovered cylinder. On a
  cuboid, hold and drag to preview a flat rectangular deletion plane. `Q`
  cycles its plane like block placement, and releasing removes the selected
  cuboids atomically.
- Use the clickable hotbar at the bottom of the window or press `1` for Block,
  `2` for Cylinder, `3` for Bearing, `4` for Weld, `5` for Hammer, and `6` for
  Joint X-ray.
  Hover an icon to see its tool name. Tool selection persists when the
  simulation mode changes.
- With Block selected, click and release the white ghost to place one block, or
  hold and drag to preview a flat rectangular sheet of blocks. While dragging,
  press `Q` to cycle the `XZ`, `XY`, and `YZ` planes; release to place the whole
  sheet, or press `Escape`/right-click to cancel. A drag is limited to 4,096
  blocks and commits atomically. Blocks have a fixed 0.25 m cube size.
- With Cylinder selected, click a flat ground, cuboid, cylinder-end, or bearing
  socket face to place one load-bearing cylinder with its local Y axis along
  the face normal. Left/Right adjusts outer diameter by 0.05 m,
  Shift+Left/Right adjusts inner diameter by 0.05 m, and Down/Up adjusts axial
  length by 0.25 m. Shift+Down/Up adjusts the retained slice angle by 15° from
  15° through a full 360° cylinder; the slice is centred on the cylinder's
  local +X direction. The supported ranges are 0.05–8.00 m outer diameter,
  0.00 m through outer-minus-0.05 m inner diameter, and 0.25–8.00 m length.
  Defaults are 0.25 m outer diameter, solid (0.00 m inner), 0.25 m length, and
  a full 360° sweep. Reducing the outer diameter clamps the bore to preserve
  the 0.05 m wall. These transient values survive tool and simulation changes
  but do not add undo-history entries.
- With Weld selected, left-click two touching existing objects. The weld
  compiles both objects into one rigid compound without spawning geometry.
- With Bearing selected, Left/Right adjusts the outer diameter by 0.05 m,
  while Shift+Left/Right adjusts the inner diameter by 0.05 m. Up/Down have no
  bearing action because bearings have no adjustable axial length.
  The HUD and placement ghost show the current values. The first left-click
  places that orange ring 5 cm into and 5 cm out of a cuboid face or cylinder
  end without
  creating a block. The default is 0.25 m outer and 0.10 m inner diameter.
  Switch to Block and move a block ghost onto any face area covered by the
  bearing ring. The bearing and block ghost turn green when the attachment is
  active; clicking attaches the new block through that zero-volume physics
  joint instead of welding it to the support. Large bearings can claim offset
  block faces under their overhang, while faces entirely inside the hole remain
  ordinary block placements. Holding and dragging from the green attachment
  preview attaches an internally welded sheet through that one bearing. The
  socket remains available afterward, so another green placement can attach a
  block group to another covered part of the same ring. Every group attached
  directly to one socket becomes part of the same rigid rotor, even across a
  gap, and they all share that bearing's single rotational motion.
  Right-click on the orange ring removes the bearing and all of its joint
  attachments without deleting their blocks; right-clicking through its hole
  reaches and removes the block behind it. The bearing remains supported by
  blocks under its ring surface, so a block in the hole can be removed
  independently. Removing its current support block automatically moves the
  bearing's ownership to another block under the ring; the bearing disappears
  only when no covered support remains.
- With Joint X-ray selected in build mode, every bearing socket is drawn
  through the construction for inspection.
- Press `P` in build mode to open the creation picker. It offers deterministic
  kinetic scenes with 256, 1,024, 4,096, or 20,000 parts. Pendulum Garden tests
  branched joints and counterweights, Mobile Workshop mixes welded compounds
  with branching payloads, and Closure Lab combines hard loop closures with
  falling contact stacks. Choosing one replaces the current construction as a
  single undoable edit; every scene remains fully editable.
- In build mode, press `Ctrl+Z` or `Cmd+Z` to undo and add `Shift` to redo.
  History retains the latest 64 committed construction edits for the current
  launch. Starting or leaving simulation does not clear it.
- Press `Escape` to cancel a pending weld or bearing selection.
- Press `Space` to compile the current construction and start simulation. While
  it is running, `Space` pauses or resumes at the exact current pose and
  `Shift+Space` restarts from the original construction. Press `Escape` to
  leave simulation and return to build mode.
- While simulating with Hammer selected, press and hold the left mouse button
  on a moving cuboid to charge a strike, then release to apply an impulse at
  that exact point along the camera ray. A quick click gives a light tap;
  charging for 1.5 seconds reaches maximum strength. Strong impacts are
  delivered over at most 12 physics ticks, with excess force on very light
  bodies limited so they cannot cross thin collision geometry between ticks.
  Tools remain selectable in either mode, but build tools act only while
  building and Hammer acts only while actively simulating.

New blocks and cylinders automatically weld through positive-area material
overlap on touching flat faces, including only the material retained by a
cylinder slice. This includes
blocks placed beside or on top of existing blocks and all neighbors inside a
dragged sheet. Blocks touching the ground are also welded to it automatically,
so construction placed on the platform is fixed by default. When placement
starts on a bearing-connected rigid group, its new blocks weld only to that
clicked rotor and to each other; touching blocks outside that rotor remain
physically separate.

Bearings require a cuboid face or flat annular or sector-shaped cylinder end;
curved cylinder walls and radial slice walls never provide placement, weld, or
bearing geometry. The ground can support blocks or cylinders and can be
selected as one side of a weld. A face entirely inside a cylinder bore or
outside its retained slice stays disconnected. Curved and radial slice walls
remain selectable for deletion, Hammer strikes, and object-level selection,
and rays through a bore or omitted sector reach objects behind it. Valid placement ghosts are
transparent white. Invalid placement and deletion ghosts are transparent red
and match the geometry affected by the action. Bearing rings may visually
overhang their supporting faces. Their holes are visual only: they do not cut,
bore, change the mass of, or alter collisions for connected blocks. Bearings
remain visible and follow their supporting bodies during simulation, but still
have no mass or collision geometry. To keep heavy scenes responsive, simulation
runs at most one fixed tick per rendered frame and updates moving geometry at a
throttled cadence. It still uses synchronous CPU snapshot readback, so it is not
evidence for the integrated-render gate.

Each full cylinder is rendered and picked as one smooth 24-segment annular mesh;
slices retain the same 15° visual resolution and add their exact radial cut
walls. Physics compiles every cylinder or slice to 16 radial cuboid colliders,
leaving the bore and omitted sector physically passable without changing the
GPU ABI or collision kernels. Cuboids still compile to one collider each; under the
131,072-collider limit an all-cylinder creation therefore supports about 8,192
parts.

The three smaller creation-picker scenes keep contact within articulated
mechanisms enabled. The full 20,000-part stress scene disables that contact to
keep its dense layout responsive; ground contact and collisions with separate
loose target blocks remain enabled.

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
