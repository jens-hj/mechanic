# Treads

A tread is a pattern of raised lugs and recessed grooves cut into one surface
of a part: a tyre's outer wall, a track pad's underside, a ski's sole. Any face
of an ordinary block takes one, as does a full cylinder's outer wall, bore, or
either end cap. A tread grips harder on ground that gives, a little less on
rock, and presses soft ground harder so the surface sinks in further.

## The model

A tread is a `TreadSpec`: a `TreadPattern` and a depth from 2 to 30 mm. The
pattern is one tile of eight by eight cells, each a raised lug or a groove,
repeated over the surface; a tile is one block (25 cm) square. Columns run
along the surface's first direction, which on a cylinder wall is the way it
rolls, and rows across it.

| Pattern | Layout | Contact |
| --- | --- | --- |
| Ribbed | ribs running the way the surface rolls | 75% |
| Lugged | bars straight across | 50% |
| Chevron | bars in a V, the tractor tyre's directional lugs | 50% |
| Block | staggered square blocks, the all-terrain knobbly | 56% |
| Studded | small separate studs on a wide floor | 25% |
| Custom | a tile drawn by hand; it keeps at least one lug and one groove | any |

Treads are surface detail, not geometry. The envelope stays where it was (the
lug tops are the surface), so placement, welds, bearings, picking, mass and
colliders are unchanged. The graph keeps them beside the parts, one
`SurfaceTreads` per treaded part, set with `BuildCommand::SetTread`. A part
spec stays small that way, which matters because every part is passed around
by value. The graph drops a tread when its surface can no longer carry one:
the part is removed, teeth or a spiral recut it, a bore layer fills its bore,
or a Shape region takes the part's surfaces over. Pipes, sectors, authored
parts and region blocks take no tread.

Creation files keep each part's treads in a `treads` list on its cuboid or
cylinder row: the surface, the pattern (a custom tile as its 64 cell bits) and
the depth. The field is omitted when a part has none, so the format version
did not change.

## Contact

`TreadSpec::response` turns a tread into two numbers, game tuning rather than
measured soil mechanics:

- **Contact ratio**: the share of the tile that is raised, at least a quarter.
  Finer lugs than that press no harder; the ground between them carries the
  rest.
- **Bite**: twice the metres of lug wall in each square metre of tread times
  the depth, at most 1. Deeper grooves and more edges bite harder.

Compiling a creation gives every collider row of a treaded part the index of
its part's `CompiledTreads`: the part's frame in its compound, its radii if it
is a cylinder, and its treads. The CPU solver uses them for terrain contacts.
For each contact it takes the touching surface from the contact normal in the
part's frame (and, on a cylinder, the radius, to tell the outer wall from the
bore), then:

- **Ground that yields** (surface cover, sand, soil): static and kinetic
  friction, and the hold failed ground still gives, rise by `1 + bite`. A
  steel cube heavy enough to break loose soil slides under a sideways pull a
  smooth face cannot hold and a deep block tread can.
- **Rigid ground** (rock, ore): friction falls by up to a fifth, in proportion
  to the grooved share of the surface, because the grooves take area out of
  contact.
- **Sinking**: the load the contact reports to the world presses
  `1 / contact ratio` harder, so the soil under the lugs compacts and gives
  way sooner. Studs sink furthest.

Contacts between bodies of a creation, the GPU route, and the spoil solver
ignore treads.

## Drawing

The part is drawn as usual, then each treaded surface's triangles are replaced
by its relief: lug tops on the surface, groove floors the depth below it, and
walls between them. A groove that reaches the surface's edge ends against the
neighbouring face, which is drawn whole. Around a cylinder the tile repeats a
whole number of times, at least four, so the pattern closes on itself. A part
with Shape features keeps its plain surfaces: its trimmed faces are no longer
the rectangles and rings a relief is laid out on.

## The tool

Matter Manipulator → Tread (Shift+0). Left-drag cuts the brush's tread into
every surface passed over, right-drag smooths them, and one drag is one undo
step. `Q` samples the tread under the cursor into the brush. The status line
names the surface under the cursor and what a click would make of it, and the
card above the hotbar shows the brush's tile, depth, and how it changes grip
and pressure.

Press the selector key (Tab) for the workbench: choose a built-in pattern or
Custom, set the depth, and click cells of the tile to raise or cut them.
Drawing on a built-in pattern starts a custom copy of it, and the brush keeps
the drawn tile while another pattern is chosen.
