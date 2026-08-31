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

## Platform setup

The repository pins Rust in `rust-toolchain.toml`, Cargo dependencies in
`Cargo.lock`, and the macOS/Linux environment in `flake.lock`. Mosaic is a
private first-party dependency, so every machine must have an SSH key with
access to `gitlab.com/unincorporated/mosaic/mosaic.git`.

### macOS and Linux with Nix

Install Nix with flakes enabled, then run the app directly:

```sh
nix run .
```

For a development shell containing the pinned Rust toolchain and, on Linux,
the Vulkan, Wayland, and X11 runtime libraries:

```sh
nix develop
cargo build --workspace
cargo run -p mechanic-app
```

The flake supports Intel and Apple Silicon macOS plus x86-64 and AArch64 Linux.
A working Metal or Vulkan driver is still required to run the GPU application.

### Native Windows

Install Git, [rustup](https://rustup.rs/), and Visual Studio 2022 Build Tools
with the **Desktop development with C++** workload. In PowerShell, the helper
uses the repository-pinned toolchain and propagates command failures:

```powershell
.\scripts\dev.ps1 build
.\scripts\dev.ps1 run
.\scripts\dev.ps1 test
.\scripts\dev.ps1 check
```

Nix does not provide the native Windows environment; use WSL only when a Linux
build is acceptable. Running the graphical app natively uses DirectX 12 and
requires a current GPU driver.

### Developing Mechanic and Mosaic together

Normal builds use the exact upstream Mosaic revision in `Cargo.toml`. To test
uncommitted Mosaic work, copy `.cargo/local-mosaic.toml.example` to
`.cargo/local-mosaic.toml`, replace the placeholder paths, and opt into it:

```sh
cargo --config .cargo/local-mosaic.toml run -p mechanic-app
```

Removing that ignored local file returns immediately to the portable pinned
dependency; no manifest edit is needed.

## Builder and simulation prototype

```sh
cargo run -p mechanic-app
```

- Press `?` to show or hide the controls and status overlay. It starts hidden
  so the construction view stays unobstructed.
- Press `F3` to toggle the pointer-transparent performance overlay. It reports
  FPS, average and p95 frame time, render CPU/GPU time when the adapter exposes
  it, actual simulation tick rate, physics CPU/GPU time, individual collision
  stages, scene/contact counts, and solver failure flags.
- Option/Alt + left-drag to orbit (middle-drag also works), Shift + left-drag
  to move the orbital centre across the ground plane, and use the mouse wheel
  or trackpad scroll to zoom. Right-click removes one hovered cylinder. On a
  cuboid, hold and drag to preview a flat rectangular deletion plane. `Q`
  cycles its plane like block placement, and releasing removes the selected
  cuboids atomically.
- Use the clickable hotbar at the bottom of the window or press `1` for Block,
  `2` for Cylinder, `3` for Bearing, `4` for Weld, `5` for Hammer, `6` for
  Control Block, `7` for Connector, `8` for Gas Engine, and `9` for Electric
  Engine. `[` selects Shape.
  Hover an icon to see its tool name. Tool selection persists when the
  simulation mode changes.
- Hold `Tab` to open the ten-material wheel. The highlighted material shows
  five 1–5 ratings beneath its name: Weight, Grip, Bounce, Roll Resistance,
  and Softness. Weight, Grip, Bounce, and Roll Resistance are calibrated
  against the engine's displayed caps; Softness uses a logarithmic scale from
  rubber-like 0.01 GPa to steel-like 200 GPa.
- With Block selected, click and release the white ghost to place one block, or
  hold and drag from the press position to preview a rectangle of blocks. The
  preview shows it as one cuboid with block counts and metre dimensions. While
  dragging, press `Q` to rotate the drag into another plane **keeping the extent
  already dragged**, so a rectangle plus one `Q` plus more motion is a solid
  cuboid of blocks. Right-drag deletes the same way, `Q` and all. The plane the
  pointer is sliding along is drawn as a translucent sheet through the blocks,
  with arrows naming its two axes, so `Q` visibly rotates something. Release to
  place the whole box, or press `Escape`/right-click to cancel. A drag is limited
  to 4,096 blocks and commits atomically. Blocks have a fixed 0.25 m cube size.
- Placement always uses the space's fixed global lattice: 25 cm normally,
  5 cm while holding Shift, and 1 cm while holding Shift+Ctrl. Ctrl alone keeps
  25 cm. Nearby-object snapping starts enabled; tap Alt/Option to toggle it.
  While holding Alt/Option, scroll to change its search range in 25 cm steps
  from 25 cm to 5 m; that scroll does not zoom the camera. The HUD reports the
  active grid, object-snap state, and range. Block and pipe drags lock the grid
  selected at their first point, then continue in 25 cm construction steps.
- With Shape selected, first choose an **editable area**: drag across blocks the
  same way the Block tool places them — `Q` mid-drag rotates the plane and keeps
  the extent already dragged, so one gesture claims a whole cuboid. The outline
  shows what is being claimed, cyan while it is claimable, with the same plane
  sheet and axis arrows the Block tool draws and the area's size labelled on all
  three axes. A region must be
  filled on every cell, be one material throughout, contain each of its blocks
  whole, and lie within one rigid body; anything else is refused with the reason
  in the HUD, and clicking a block that is already in a region reopens it. The region merges into a single shape
  with eight corners to edit, and the rest of the build fades back so the area
  under the cursor is the only thing reading as solid. `Escape` leaves it.
  Drag a corner to shape it. Movement follows one world axis at a time, shown by
  a pair of arrows through the corner; press `Q` during the drag to cycle to the
  next axis while keeping the movement already made. Movement is also constrained
  to a fraction of a block rather than running free, so two corners line up
  because they landed on the same sub-grid rather than because they were matched
  by eye. `G` cycles the step
  through one block, a half, a quarter (the default, 62.5 mm), and fine 12.5 mm
  detail. **No corner may leave the region's original bounding box**, so a corner
  can only ever be drawn inward and one region can never grow into its
  neighbours. Driving a corner the whole way onto its neighbour collapses that
  edge — two collapsed corners turn a box into a wedge, four into a pyramid —
  while driving two corners through each other is refused.
  Click a corner to select it, or shift-click to build a set. A selected corner
  turns the overlay's cyan and grows, so a selection reads at a glance. Drag from
  empty space to sweep out a selection rectangle, drawn in the same cyan. The
  arrow keys and `WASD` then nudge the whole selection one step at a time: the
  keys read as screen directions and resolve to whichever grid axis lies nearest,
  so a nudge goes where it looks like it should while still landing on the grid.
  Orbiting the camera is how the third axis is reached.
  Bring the pointer near a region **edge** and it offers a new corner at the
  nearest grid position along it; taking it inserts a whole cage plane with
  every new corner interpolated from its neighbours, so the surface does not
  move — the cage simply gains a row of handles. `X` and `Z` mirror across the region's own
  centre planes, independently, and a corner on an active centre plane keeps its
  offset along that normal at zero. A whole mirrored edit, group included, is one
  undo entry.
  Shaping is exact rather than cosmetic: a region's mass, centre of mass, and full
  inertia tensor are integrated from the same geometry that is drawn, and its
  collision uses convex polytope rows built from that same decomposition, so the
  hitbox always matches what is on screen. The blocks a region covers stop
  emitting geometry of their own, and deleting one of them deletes the region.
  Only ordinary blocks can be claimed; cylinders, engines, servos, seats, control
  blocks, and Input blocks are machined components and keep their shape.
- Parts can only be placed on **flat** surfaces. A face that has been shaped is
  no longer an axis-aligned rectangle, so nothing can sit flush on it until its
  corners are brought back onto the grid — which is also how a mounting surface
  is made somewhere there was not one before.
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
  An object here is a whole rigid body, which is what the highlight shows:
  contact anywhere between the two bodies is enough, and two parts of one body
  cannot be welded to each other. A weld may close a loop through a bearing.
  When that leaves a bearing with both sides in one body the ghost turns amber
  and the HUD says how many bearings the weld locks; the weld still goes
  through, and a locked bearing compiles to no joint at all rather than
  blocking the build. A bearing driven by a control block is the exception,
  because dropping its joint would silently kill the drive.
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
  textured 2x2x1 control block (0.50 x 0.50 x 0.25 m). It welds, collides, and
  carries mass like an ordinary block. A control block holds no settings of its
  own: what it does lives on the wires running from it, one program per bearing.
  Press `Q` before placement to rotate it by 90 degrees. Clicking an existing
  block selects it; press `E` to open its panel.
- With Gas Engine or Electric Engine selected, click the platform or a face to
  place its textured 2x2x3 or 2x2x2 body. Engines currently weld, collide, and
  carry mass, but do not drive the mechanism yet. Press `Q` to rotate the
  placement preview by 90 degrees.
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
have no mass or collision geometry. The independent 60 Hz scheduler retains
every due tick without dropping catch-up work and submits only when one of three
fixed staging slots is available. Diagnostics and prototype-render transforms
arrive through tick-sequenced asynchronous staging, so an app frame never
blocks on GPU mapping, an overloaded GPU cannot create an unbounded submission
queue, and a failed tick never overwrites a snapshot. The
prototype still maps bulk transforms and rebuilds CPU meshes after completion,
so it is not evidence for the integrated GPU-render gate.

Each full cylinder is rendered and picked as one smooth 24-segment annular mesh;
slices retain the same 15° visual resolution and add their exact radial cut
walls. Physics compiles every cylinder or slice to 16 radial cuboid colliders,
leaving the bore and omitted sector physically passable without changing the
GPU ABI or collision kernels. Adjacent grid-aligned cuboids in the same rigid
body compact greedily when their contact material matches. Authored mass and
inertia, material boundaries, separate rigid bodies, cylinders, shaped regions,
and non-grid geometry remain unchanged. Under the 131,072-collider limit an
all-cylinder creation therefore supports about 8,192 parts.

Construction materials carry density, separate static and kinetic friction,
restitution, rolling resistance, and Young's modulus. Contact pairs mix both
friction coefficients and rolling resistance by geometric mean, take the
higher restitution, and add their nominal block compliances. The solver retains
two-axis static friction until the static limit is exceeded, then clamps to the
lower kinetic limit; it also resists rolling about axes tangent to the contact
normal without adding torsional spin friction. The ground uses Concrete's
surface preset. Machine parts keep their fixed low-friction, non-bouncy contact
response, while bearings remain massless and collisionless.

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
cargo run -p mechanic-bench --release -- --scenario open_bearing
cargo run -p mechanic-bench --release -- --scenario four_bearing_contact
cargo run -p mechanic-bench --release -- --scenario bearings_16
cargo run -p mechanic-bench --release -- --scenario bearings_64
cargo run -p mechanic-bench --release -- --scenario bearings_65
cargo run -p mechanic-bench --release -- --scenario bearings_256
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
