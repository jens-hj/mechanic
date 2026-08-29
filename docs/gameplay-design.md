# Gameplay design

This document records intended player-facing rules and experiences. It describes
design direction rather than currently implemented behavior or technical
architecture.

## Garage and Dimension Links

### Core concept

The Garage exists in a pocket dimension. It is a safe, physics-free construction
space where the player can work on one creation at a time, separate from the
simulated world.

The player begins with a small Garage and can expand its dimensions through
gameplay.

### Linking and travel

- A **Dimension Link** is a physical module built into a creation. It must be
  structurally attached to the creation through welds or mechanical joints. It
  must also be connected to the creation's controller, either through a
  physically welded chain or with the Connector tool.
- Multiple creations may each have a Dimension Link, but only one creation can
  be actively linked to the Garage at a time.
- Activating one Dimension Link deactivates every other Dimension Link. Later
  progression may let the player keep multiple Dimension Links active at once.
  This replacement happens only after a successful transfer; a failed
  activation leaves the previous Dimension Link active.
- The controller determines the condition that enables the Dimension Link. For
  example, it could activate when the player enters a seat or presses a chosen
  key.
- When the player is near the actively linked creation and its enable condition
  is met, the player and creation move into the Garage.
- If a different creation is already in the Garage, it must be returned to the
  world before another creation can be linked and brought in, unless its
  Dimension Link is removed and it is deliberately left behind as Garage
  clutter.
- The player can always enter and leave the Garage without taking a creation.
  This allows them to retrieve the creation currently inside or return it to the
  world to make room for another.
- The player and each creation exist in only one place at a time; entering the
  Garage removes them from the world rather than creating copies.
- Only the player entering with the creation transfers. Other players and
  passengers are left behind in the world.
- The transferred creation includes everything connected to its Dimension Link
  through welds or mechanical joints. Loose physics bodies, including loose
  cargo, remain in the world.
- A creation welded to the ground cannot enter the Garage, because its connected
  assembly effectively includes the world. Activation is refused.
- When the player leaves the Garage with a creation, they both appear where the
  player entered the Garage. The creation is in its authored static pose and has
  no linear or angular motion.
- The return position is always the player's entry point for that Garage visit,
  not the creation's earlier world position. A player can therefore store a
  creation, enter the Garage later from somewhere else, and deliberately
  retrieve the creation at the new location.
- A creation may remain stored in the Garage indefinitely while the player is
  elsewhere.
- The player may remove a creation's Dimension Link while working in the
  Garage, but the creation cannot leave until a Dimension Link is structurally
  attached and connected to its controller again.
- If the player moves the Dimension Link onto a disconnected assembly, that
  assembly becomes the creation eligible to leave. The previously linked
  assembly remains behind as clutter.
- Any stored assembly can later become a creation again by attaching a valid
  Dimension Link, connecting it to a Controller, and taking it out of the
  Garage.

Having a Dimension Link and being actively linked are distinct states: the
former makes a creation eligible, while the latter selects the one creation
that can currently use the Garage.

### Garage behavior

- The creation floats near the center of the Garage. Existing clutter may move
  its arrival point: the game finds a free volume large enough in all three
  dimensions and places the creation at the center of that available space.
- On entry, the creation immediately resets from its current simulated pose to
  its authored static pose.
- The Garage is strictly a physics-free building environment; creations cannot
  be test-run there.
- There is no gravity in the Garage.
- The player can float freely around the creation in all three dimensions.
- Outside the Garage, in the world, the creation uses the normal physics
  simulation.
- While the player is in the Garage, the outside world is either paused or runs
  extremely slowly. The exact behavior remains to be decided.

### Garage capacity

A creation can enter only if it fits within the Garage's current build space. If
it is too large, activation is refused. The player is told which dimensions are
out of bounds and by how much, so they can make the creation smaller or expand
the Garage.

The fit check uses the creation's authored static pose: the position of all its
parts as built and as they reset when physics is not applied. Its current
simulated pose does not affect whether it fits.

The player can define each controlled joint's authored resting pose through its
Controller block. A separate initial-pose column at the start of the joint's
program swimlane records the resting, starting position used by Garage reset and
fit calculations. Changing this value in the Garage updates the creation's pose
visibly and immediately.

An uncontrolled joint uses the angle at which it was originally built as its
resting pose. Connecting it to a Controller unlocks the ability to change that
pose.

The fit check tries all four cardinal rotations around the up axis. It may not
tilt the creation around either horizontal axis. When multiple rotations fit,
the game chooses the most natural alignment, placing the creation's long axis
along the Garage's long axis.

The build space is the volume in which creation geometry may exist. The Garage
walls sit outside it, leaving a navigable margin in which the player can move
but cannot build. The current direction is to make this margin one Garage cell
deep on every side; the exact depth remains to be decided. Building is strictly
blocked at the build-space boundary.

Editing may split the creation into disconnected pieces. Only the assembly
structurally connected to the Dimension Link can leave with it. Every other
piece remains in the Garage as persistent clutter, occupies some of the finite
build space, and can make later work harder. The game does not clean this up
automatically; keeping or removing the clutter is the player's choice. Clutter
is saved permanently and remains across loading and restarting the game.

Linkless assemblies may remain in the Garage while another linked creation is
brought in. Existing clutter participates in the admission check: entry is
refused if the incoming creation would overlap it, even when both otherwise fit
inside the Garage build space.

### Building in the world

The Garage is not required for construction. The player can build or modify a
creation directly in the world while physics remains enabled and continues to
affect it during placement and deletion. This includes live changes to vehicles
and static constructions, such as a house welded to the ground.

A world edit may make a creation too large for the Garage. The edit is still
allowed, but the player is notified and cannot bring that creation into the
Garage until it fits again or the Garage is expanded.

### Open design questions

- Is the outside world fully paused while the player is in the Garage, or does
  it continue at a very low time scale?
- How does the player enter or leave the Garage without a creation?
- What happens if the player's world return location becomes obstructed or
  unsafe before they leave the Garage?
- How large is one Garage cell, and should the navigable margin use exactly one
  cell on every side?
- How should Garage orientation be chosen when a creation or Garage has no
  unique long axis?
- If later progression allows multiple active Dimension Links, how many
  creations may occupy the Garage at once?
- What resources or progression unlock Garage expansions, and are its three
  dimensions upgraded independently?
