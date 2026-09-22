# Gears and meshes

Teeth go on cylinders, racks along block faces, and any two toothed parts whose
pitch surfaces touch mesh: spur trains, compound reductions, ring gears and
planetary sets, rack-and-pinion, bevel pairs and differentials, worms on spiral
cylinders, and a nut on a thread. Meshing parts never touch. They are magnetic
gears: once in mesh they follow each other without slip, and the solver holds
that as one row per mesh instead of tooth contact.

## The model

Teeth are a feature the envelope hides, as a spiral is. A cylinder carries a
`GearSpec`; a cuboid carries a `RackSpec` on one face. For placement, welds,
bearings, picking, mass and collision a toothed part is its plain cylinder or
block; only rendering and the mesh see the teeth.

| Setting | Meaning |
| --- | --- |
| Module | tooth size, 5 mm to 5 cm in 2.5 mm steps; the pitch diameter is module × teeth |
| Teeth | 6 to 800; two meshing parts must share a module |
| Kind | spur (teeth on the wall), ring (teeth on the bore, pointing in), bevel (teeth on a cone with a whole-degree cone angle and a large end) |
| Rack | the face the teeth are cut into and the local axis they run along |

The tooth count decides the cylinder: external teeth reach the outer diameter,
`module × (teeth + 2)`, and a ring's teeth its bore, `module × (teeth − 2)`. The
rack's teeth are cut one addendum into its face, with the pitch plane a module
below the surface. A toothed part takes no material layers, spiral or Shape
features, and the other way round.

A rack longer than one block is a rack on each block. Rack teeth are spaced
from the construction frame's origin rather than from the block's centre
(`RackSpec::tooth_line`), so racks cut into neighbouring blocks of one body
continue each other tooth for tooth, and the teeth at a block's edge are cut
where the edge is. The pinion side of a rack mesh follows the rack's pitch
plane, not the block, so a pinion meshed with one block rolls the length of
the run; a pinion standing over the joint between two blocks meshes the rack
once.

Meshing teeth interleave, so the envelopes of two meshing gears overlap by two
modules. Placement allows that: against a part it could mesh with (teeth, a
rack or a thread) a toothed cylinder claims only the space inside its root
circle, and against anything else its tooth tips. In the simulation the
toothed, racked and threaded parts of two meshed bodies never touch, while the
rest of those bodies still collide, so a block built beside a rack's travel
stops it; break the mesh and the overlapping envelopes push the parts apart,
as loose gears would.

A mesh is a `GearLinkSpec`: two parts and nothing else. Everything about it
comes from the parts' teeth and rest poses, resolved by `mechanic_core::mesh`
into a `GearMesh`: two sides, each a point on its axis in the pitch plane, that
axis and a pitch radius, plus a thread advance for worms and screws. The kinds
are spur or ring pairs and bevel pairs (`Gears`), a spur gear on a rack
(`Rack`), a spur gear across a threaded cylinder (`Worm`), and any part riding a
thread (`Screw`). The ratio is tooth counts, or lead over pitch circumference
for a worm; the built centre distance does not enter it. That is why a mesh only
has to be roughly tangent: pitch surfaces up to one module apart or half a
module into each other mesh, axes within about a degree of parallel,
intersecting or perpendicular as the kind needs, and teeth do not have to be
phased. Parts welded into one body cannot mesh, and taking the teeth, rack or
spiral off a part drops its meshes.

## What sees the teeth

- **Render** (`mechanic-app/src/render/mesh/gear.rs`): trapezoid teeth, a
  quarter pitch of tip and root with sloped flanks, extruded along the axis or
  along a bevel cone, with the plain bore or rim and end caps; a rack as its
  block sunk by the root depth with one prism per tooth on its line, the end
  ones cut at the block's edge. Each gear is
  drawn turned by its phase from `mechanic_core::gear_phases`, so at the rest
  pose its teeth sit in its partners' gaps: racks keep their teeth where they
  are cut, the lowest gear of a train keeps its own, and the rest follow mesh
  by mesh. That is rendering only; the solver's slip bias keeps the parts
  where they started, so the teeth stay interleaved while the machine runs. A
  worm's wheel is not phased to the thread, and a closed train such as a
  planetary set is phased along one path through it.
- **Compile** (`mechanic-core/src/compile.rs`): each mesh becomes a
  `CompiledGearLink` whose sides are in compound-local coordinates, and the two
  compounds are joined: the CPU solver exempts their meshing parts from contact
  (`meshing_parts`), the GPU runtime the two bodies whole. A mesh whose parts ended up in one rigid
  body, or that no longer meets, compiles to nothing rather than refusing.
- **Solver** (`mechanic-physics/src/gear_mesh.rs`): one soft bilateral row
  holding `S(a) = S(b)`, where a side's surface speed is the velocity of its
  pitch point along the common tangent, plus advance × spin about the thread
  axis for a worm or screw. Written at pitch points rather than against a
  housing, the same row is exact for a planetary set's turning carrier and for
  a differential's spider gears. The row is rebuilt at every substep's pose,
  warm-started across ticks like a closure, and biased by the slip it has
  accumulated so teeth stay phased; `SoftStepDiagnostics::mesh_slip` reports
  the worst. The exact reference solver adds the same Jacobian as an unbounded
  block. The GPU runtime uploads a creation with meshes, because the world
  keeps a GPU scene resident for every creation as the CPU route's fallback,
  but refuses to tick it (`GpuDispatchError::UnsupportedGearLinks`) rather
  than run without them; a CPU failure on a geared creation therefore stops
  the simulation instead of handing it to the GPU.

## The tool

Matter Manipulator → Gear (Shift+9). Click a plain cylinder to cut teeth into it,
a block face to cut a rack along it, or a bare bearing to place a one-block gear
centred on it. For a longer rack, press on one block's face and release over
the last: every block welded to the first whose face lies in the same plane
and under the rectangle dragged out is racked in one step, along the run's
longer side (Rotate takes the shorter), and the covered blocks are tinted
until the release. Right-click cancels the drag. Until the tooth count is set by hand the teeth reach: a new gear
takes the module of the nearest toothed, racked or threaded part its axis
suits and as many teeth as put their pitch surfaces tangent, so a bearing
placed roughly beside a gear gets a gear that meshes with it, and the status
line says so with the ratio ("Click: 38-tooth spur gear … ; meshes 24-tooth
spur (1:1.58)"). Rounding to whole teeth leaves at most half a module, inside
the mesh tolerance. With nothing in reach a plain cylinder sizes the count to
its own diameter. Once the count is set by hand the teeth size the cylinder
instead, and the line says what they miss ("pitch circle 40 mm short of
24-tooth spur", or the module the partner needs), so the bearing can be moved.
With several parts in reach the gear fits the one the smallest gear reaches
and skips any whose fitted gear would run into another part; the line counts
the rest ("1 of 2; Rotate for the next") and Rotate steps through them.
Cylinder Length keys step the tooth count (six at a time, one with Fine held),
Cylinder Outer keys the module, Pipe Turn cycles spur, ring and bevel, and
Rotate turns a rack across its face's shorter side. Changing a setting while
pointing at a toothed part reshapes it at once, so the part is its own preview;
Interact picks a part's settings up, Shape Snap resets them. A ring on a solid
cylinder falls back to external teeth rather than being refused.

Every cut or placed gear meshes by itself with each toothed, racked or threaded
part its pitch surface reaches; the status line names them with the ratio from
this part's side. To mesh parts that are already toothed, press one and release
on the other, or click one and then the other; a spiral cylinder dragged onto a
gear meshes as a worm, onto anything else it becomes that part's lead screw.
Right-click takes teeth or a rack off, with their meshes, or breaks every mesh
of a toothed part that is pointed at.

## Behaviour

`cargo test -p mechanic-physics soft_step::tests::gears` holds, without gravity
and with the tolerances of a game:

- a 24-tooth pinion driving a 36-tooth wheel turns it at −2/3 of its speed and
  angle within 1 %, with under 2 mm of tooth slip after two seconds
- a 24:36 then 12:36 compound train multiplies to 2/9
- a sun on its carrier inside a fixed 72-tooth ring turns the carrier at a
  quarter of the sun's speed
- a rack moves at its pinion's pitch speed
- a single-start worm turns a 24-tooth wheel one lead per turn
- a differential of four mitre gears averages its outputs unloaded and doubles
  the free side when the other is held
- a geared machine repeats exactly from identical inputs

`cargo run -p mechanic-bench --release --bin cpu-physics -- --scenario gear-train`
runs a two-stage reduction and a three-planet set, eight meshes, at 0.08 ms per
tick at p50 and 0.10 ms at p95 on an M1 Pro, with at most 5 mm of slip over ten
seconds.

Known limits: drive budgets use the tree's axis inertia, so a motor on a gear
feels only that gear and not the train behind it; a lead screw's nut needs its
own guide against turning, such as a linear bearing; bevel pairs are not
fitted, their count is set by hand; the bearing tool does not yet snap to the
pitch distance of hand-set teeth; the GPU runtime has no mesh kernel yet.
