# Spirals on cylinders

A full cylinder can carry a spiral on its outer wall, its bore, or both: auger
flights, screw conveyors, threads, twist-drill flutes. It stays one part with the
same welds and bearings; only its walls change.

## The model

A spiral is a wall profile swept along a helix. The profile is **depth below the
wall's surface against position along the axis, over one pitch**: a periodic
polyline of at most eight points on a 1.25 cm grid
(`mechanic_core::SpiralProfile`). Two points at one position make a square flank;
sloped segments give V, buttress and wood-auger flights. Undercuts cannot be
drawn. The rest is four numbers (`SpiralSpec`):

| Setting | Meaning |
| --- | --- |
| Pitch | distance between neighbouring ridges, 5 cm to 8 m |
| Starts | 1 to 8 identical ridges; the lead is pitch × starts |
| Hand | right or left |
| Taper | one end narrowing linearly to a tip diameter over a length |

The cylinder's dimensions are the envelope. Cutting a groove into a thick
cylinder and adding a ridge onto a thin one are stored the same way: the tool's
**Add** mode widens the envelope by the ridge height first, and is refused where
the wider part would not fit, as a wall layer is. At least 2.5 cm of wall always
remains. A spiral cylinder takes no material layers and no chamfers or fillets,
and the other way round.

`SpiralSpec` is a reference to an interned record. Parts are copied by value all
over the builder, and a drawn profile is far larger than everything else a part
carries.

## What sees the spiral

For placement, faces, welds, bearings, picking and overlap, a spiral cylinder is
its envelope cylinder. Three consumers see the spiral itself:

- **Colliders** (`compile/colliders.rs`): the straight core as sixteen boxes like
  any cylinder, plus one convex hull per ridge segment per angular step
  (`solid/spiral.rs`, twelve steps a turn, coarsened to six to stay under 512
  ridge colliders; a finer pitch than that is refused). A tapered core is convex
  wedges. There is no analytic rolling cylinder for a spiral part.
- **Mass** (`compile/mass.rs`): the core as an exact annulus, the rest from the
  same hulls at 24 steps a turn. Within 2 % of the swept volume.
- **Render** (`mechanic-app/src/render/mesh/spiral.rs`): helix-aligned wall
  quads at 48 columns a turn, clipped at the end planes and where a taper
  begins, with analytic normals and ring caps.

## The tool

Matter Manipulator → Spiral (Shift+8). Click a cylinder to put the spiral on it;
right-click takes it off (filling the groove in Cut mode, stripping the ridge in
Add mode). Changing a setting while pointing at a spiral cylinder reshapes it at
once, so the part itself is the preview; Interact picks a spiral's settings up.
The first cylinder the tool meets sizes pitch and depth to itself. The settings
reuse the cylinder tool's dimension keys: length → pitch, inner → ridge width,
outer → depth, sweep → starts; the wheel sets the taper length at the end being
pointed at, up/down the tip diameter, Pipe Turn the ridge shape, Rotate the hand,
Mirror X cut or add, Mirror Z outer wall or bore.

A drawn profile editor is planned; the tool currently offers square, V and
buttress ridges.

## Moving spoil

Spoil meets a spiral as it meets any machine part. In a 60 cm bore, a 55 cm
right-hand auger with a 30 cm pitch turning clockwise seen from above lifts all
of 40 clods out in five seconds at 12 rad/s (115 rpm), most of them in fifteen at
6 rad/s, and none at 3 rad/s: as in reality it is the bore wall's friction on
spoil flung against it that makes spoil climb, so a servo's 30 rpm only stirs.
A stopped auger holds its spoil on the flights. Turned the wrong way it presses
spoil down; left running, spoil crushed against the floor squirts up past the
flights in the end. Clods gather to 19 cm across, so channels narrower than about
20 cm jam.
