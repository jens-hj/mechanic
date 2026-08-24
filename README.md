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
  `2` for Cylinder, `3` for Bearing, `4` for Weld, `5` for Hammer, `6` for
  Control Block, and `7` for Connector.
  Hover an icon to see its tool name. Tool selection persists when the
  simulation mode changes.
- With Block selected, click and release the white ghost to place one block, or
  hold and drag from the press position to preview a flat rectangular sheet of
  blocks. The preview shows the sheet as one cuboid with block counts and metre
  dimensions. While dragging, press `Q` to cycle the `XZ`, `XY`, and `YZ`
  planes; release to place the whole sheet, or press `Escape`/right-click to
  cancel. A drag is limited to 4,096 blocks and commits atomically. Blocks have
  a fixed 0.25 m cube size.
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
- With Control Block or Connector selected, every bearing socket is drawn
  through the construction for inspection. Driven bearings additionally show a
  teal spin arc pointing the way their active state turns them, a straight wire
  back to the control block steering them, two radial ticks at their travel
  limits, and a floating number naming the joint. That number is the row number
  its control block's panel gives it, so `Joint 3` in the table is the joint
  wearing a `3` in the world. Two wires reaching one physical joint share a
  panel row, and so share one number. The drive overlay stays visible while the
  simulation runs — arcs and wires follow the moving bodies and the arc flips
  as a joint changes state, so you can see which joint a key drives.
- With Control Block selected, click the platform or a face to place a fixed
  0.25 m teal control block. It welds, collides, and carries mass like an
  ordinary block. A control block holds no settings of its own: what it does
  lives on the wires running from it, one program per bearing. Clicking an
  existing block selects it; press `E` to open its panel.
- With Connector selected, press the left mouse button on a control block and
  drag to a bearing, or start on the bearing and drag to the block — wiring runs
  in either direction. Whatever the pointer is over that a wire can land on — a
  joint or a control block — is drawn slightly oversized through the
  construction, so the target is visible before the button goes down. The wire
  follows the pointer and snaps to whichever end would complete it. Pressing and releasing without moving leaves the wire
  armed, so click-then-click works too. Wiring aims at the whole joint, hole and
  pin included, rather than at the thin ring. Dragging an already-wired pair
  again reverses its direction, and right-clicking a wired bearing removes the
  wire. A bearing with no part attached through it yet cannot be wired — attach
  one first. Each
  bearing obeys at most one control block, while one control block can drive
  any number of bearings, each with its own program.
- Press `E` over a control block, or with one selected, to open its panel. It
  lists one row per wired joint. Each row carries its own maximum speed and
  torque, optional travel limits, and a loop toggle, followed by its ordered
  states. Click a cell to change it, or click a speed, torque, value, or dwell
  cell and type a number — `Enter` commits, `Escape` cancels, `Backspace`
  deletes. Angles are entered and shown in degrees. Torque and dwell also accept
  `none` or `inf`, which is what an empty cell commits and what an empty cell
  displays while you type: unlimited torque, and a state that never advances on
  its own. An infinite dwell and no dwell are the same setting. A state with no
  dwell is left when its key goes up instead, so its `then` cell names where it
  hands off — `stay` keeps it latched, and `→S1` sends it back. That is the same
  setting the `on release` cell shows, editable from either column. Escape closes the panel once nothing
  is being edited. The panel works in both build and simulation mode, so a
  machine can be reprogrammed while it runs. It is a fixed panel inset from the
  window edges, so it neither resizes as values change nor runs off the bottom
  of the screen: a block with more joints than fit scrolls with the mouse wheel,
  and the title and hint line stay put above the table while it does.

  Worked examples, one row each:

  | Goal | Program |
  |---|---|
  | Steering | `S1 0°` · `S2 30°` key `A` ⇥S1 · `S3 -30°` key `D` ⇥S1 |
  | Driving | `S1 0/s` · `S2 3/s` key `W` ⇥S1 · `S3 -3/s` key `S` ⇥S1 |
  | Arm poses | `S1 30°` key `Q` · `S2 40°` key `W` · `S3 80°` key `R`, all holding |
  | Procedure | `S1 0°` key `R` · `S2 90°` key `S`, 2 s →S3 · `S3 -90°`, 4 s →S2 |
- A state sets the joint's target: an **angle** it seeks and holds, or a
  **speed** it turns at. A state is entered by pressing its bound key, and left
  either when a bound key is released or when its dwell time elapses. A key
  cell arms capture — the next key you press is bound; clicking a bound key
  clears it. Letters and digits bind, except `E`, which opens the panel. A
  released key either holds the state or reverts to a named one, which is what
  makes hold-to-steer work. A dwell hands off to the following
  state by default, or to any state you pick, so a two-state cycle runs forever
  while a reset key still interrupts it. The same key may drive several joints
  at once, but not two states of one joint.
- Motion is never instant. An angle state ramps toward its target within the
  row's torque budget and brakes so it settles without overshooting; a speed
  state accelerates into its target. Gravity and contacts can slow, stall, or
  back-drive either, and a weak torque genuinely fails to lift a load. With
  travel limits on, the joint stops and holds at the limit.
- Drive programs are the only values editable while the simulation runs. They
  change no topology, mass, or buffer size, so new targets are written straight
  to the GPU without recompiling or restarting.
- Press `P` in build mode to open the creations screen, or `Ctrl+S`/`Cmd+S` to
  open it ready to save. It both saves the current creation and opens a saved
  or preset one. While it is open it owns the keyboard, so letters and digits
  type into its name field rather than firing shortcuts.
  - Type a name and press `Enter` (or click **Save**) to write the current
    construction to disk. Saving over an existing name asks once before it
    replaces it. Saving changes no construction, so it adds no undo entry.
  - Click a saved creation to open it. Its `×` asks once, then deletes the file
    for good.
  - Below the divider are the deterministic kinetic presets with 256, 1,024,
    4,096, or 20,000 parts. Pendulum Garden tests branched joints and
    counterweights, Mobile Workshop mixes welded compounds with branching
    payloads, and Closure Lab combines hard loop closures with falling contact
    stacks.
  - Opening anything replaces the current construction as a single undoable
    edit; every scene remains fully editable.
  - `Escape` clears a half-typed name, then closes the screen.
- Creations are stored where the app finds them by itself, with no path to
  configure: `~/Library/Application Support/mechanic/creations` on macOS,
  `$XDG_DATA_HOME/mechanic/creations` (or `~/.local/share/mechanic/creations`)
  on Linux, and `%APPDATA%\mechanic\creations` on Windows. Set
  `MECHANIC_CREATIONS_DIR` to put them somewhere else. The screen prints the
  directory it is reading. Files are RON with a `.mech` extension, named after
  the creation's slug; the display name lives inside the file, so renaming
  either one is lossless. A save writes to a temporary file and renames it into
  place, so an interrupted write cannot destroy an existing good save.
- A saved creation holds exactly what the editor authors: parts, welds, rigid
  links, bearings, drive wires with their limits and programs, and bearing
  rings that nothing hangs from yet. Everything else — compiled bodies, mass
  and inertia, loop topology, GPU buffers — is recomputed on load. The file
  numbers its rows by position rather than by handle, and loading replays them
  through the same validating constructors the editor uses, so a hand-edited
  file cannot install an invalid construction.
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

A driven bearing needs its own mechanism coordinate, so compilation prefers
driven bearings as spanning-tree edges over passive ones. A driven bearing that
can only be a loop-closure edge is rejected with an explicit error rather than
silently losing its drive.

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
