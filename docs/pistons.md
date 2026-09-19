# Pistons

Piston is in the placeable picker beside Bearing and Linear Bearing. It is a
telescopic ram: one block square, a whole number of blocks long closed, and it
pushes and pulls whatever is attached to its head.

Tap Rotate to choose an end mount or one of four side mounts. Hold left-click on
a flat construction face and move the mouse to set the closed length; press
Rotate while holding to switch to the stage count; release to place. Right-click
cancels. Then attach blocks or cylinders to the head crown with the block or
cylinder tool, or weld an existing assembly onto it. Everything rigidly attached
to the head moves as one body.

## The rule

A stage draws its own whole length, so every configuration extends onto a block
line:

| quantity | value |
| --- | --- |
| closed | `blocks × 0.25 m`, 2–8 blocks |
| stages | 1–6 |
| stroke | `stages × closed`, 0.5–12 m |
| extended | `closed × (stages + 1)` |
| section of stage *k* | `0.25 − 2k × 0.0095 m`; stage 0 is the body |

Stages draw largest first. The collapsed envelope is exactly `0.25 × 0.25 ×
closed`; a piston is always built collapsed and its coordinate is the head's
extension in metres, from 0 to the stroke.

## Mounting

- **End mount.** The base rings sit on the face and the piston extends along its
  normal. The anchor is the base centre.
- **Side mount.** Two saddle brackets hold the body flat on the face and it
  extends parallel to it, in any of four directions. The brackets live in the
  corners the round section wastes, so the assembly is still one block square.
  The anchor is the centre of the collapsed body's footprint on the face; the
  body may overhang its support.

Both mounts are rigid. Articulation comes from bearings placed outside the
piston. Only the head crown takes attachments, and one head carries one piston's
attachments. Deleting the support deletes the piston with it.

## Programming

A piston is driven through a Controller wire exactly as a linear bearing is.
Program states are extensions in metres measured from collapsed, or speeds in
m/s; the wire's travel limits narrow the physical `0…stroke` range. Servos seek
and hold an extension and Engines drive a speed, through the existing power
allocation and the 0.25 m per output revolution gearing. A piston extends one
way, so its wire cannot be reversed. Unwired, the head rests on its collapsed
stop and can be pulled out to full stroke by a load.

`MAX_LINEAR_TRAVEL_METERS` is 12 m, the stroke of the longest piston.

## Implemented

- `PistonDimensions`, `PistonMount` and `Piston` in `mechanic-core` validate the
  counts and the mounting frame. `BearingKind::Piston` is translational with
  bounds `[0, stroke]`. The solvers, GPU upload, and drive compilation are
  generic over `is_translational()` and `bounds()`; there is no solver, ABI, or
  WGSL change, and the creation format is unchanged at 17.
- Hardware mass follows the guide: the body and intermediate stage tubes weigh on
  the supporting body, the solid last stage and its aluminium crown on the head
  body. A 2-block single-stage piston is 190 kg. An unattached piston weighs
  wholly on its support.
- `piston_meshes` ports the asset pack's `game` detail level: banded barrel, skin
  pads, ports, glands, keyed stages, tapped head crown, and saddle brackets. The
  mesh is built once per configuration; extension only translates each stage.
  Seventeen finishes reuse the steel, aluminium and rubber textures. Stage skins
  walk from ground steel to chrome by stage index.
- `hardware_mesh` holds the lathe, ring and ear-clipping helpers the suspension,
  rail and piston meshes share, and `HardwareFinish`.

## Not ported from the asset pack

- The 16 MPa thrust and constant-flow speed laws. Force and speed come from the
  wired actuators.
- `high`, `far` and `silhouette` detail levels, and the profile atlas.
- Skin pads as mount faces; the saddle bracket is the side mount.
- Side-load derating, buckling limits, and smallest-first retraction order.
- The pack's port discs stand 0.5 mm proud of the 250 mm section. Here they sit
  0.2 mm inside it so a side-mounted piston stays within its block.

Pistons, like rails and suspension, have no colliders of their own.

## Verification

```sh
cargo xtask ci
cargo test -p mechanic-core piston
cargo test -p mechanic-physics a_piston_lifts
cargo test -p mechanic-gpu device::tests::piston -- --test-threads=1
cargo test -p mechanic-app piston
```

The GPU tests need a real adapter; they were run on Apple M1 Pro, Metal. They
cover seeking and holding three programmed extensions under load, resting on the
collapsed stop unpowered, holding full stroke under sustained power from a side
mount, and the reaction on a floating base. The CPU soft-step test covers the
same lift on the solver the app runs by default.

A free-falling floating base under sustained power drifts a few centimetres off
the full-stroke stop over several seconds on the GPU route; the grounded case
holds it exactly. This is not specific to pistons and is not asserted.
