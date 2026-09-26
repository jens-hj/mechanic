# Water

Water follows the terrain's rules. Untouched water is a pure function of the
seed and the world definition: a level per column and a rule, with nothing
stored. Stored water records only where water differs from that, the way
edited bricks override untouched ground. Every cubic metre is conserved:
water comes from somewhere and goes somewhere (see [The cycle](#the-cycle)).
Only `WaterWorld::deposit` and `withdraw`, the brush and pumps, add or remove
it.

## The model

- **Seed-derived water is implicit.** The sea, lakes and rivers below.
- **Stored water lies in pools.** Water is held in 20 cm water cells, four
  terrain cells to an edge. A pool is one level and one volume over its member
  cells; how full each cell is follows from the level and the cell's open
  terrain cells. A pool grows by priority flood, the algorithm the drainage
  already uses: the lowest neighbour whose floor the water covers by a
  centimetre is taken in next. A U-tube is one pool, so it settles level in
  both arms without pressure being simulated.
- **Water moves across contacts.** A neighbour a pool cannot take in is a
  contact: another pool's cell, seed-derived water, or a drop, where the cell's
  bottom opens onto free space or lower water. Across a contact the higher
  water runs to the lower at a weir rate, `1.7 × 0.2 m × head^1.5` a second,
  where the head is measured over the lip the water must cross. Pools whose
  levels come within 5 mm merge. Water pouring over a drop is launched into
  the air (see [Waterfalls](#waterfalls)).
- **Pools spill over their rims.** A pool records the highest floor its
  water crossed from its seed. Ground beyond it that lies lower is past the
  rim: the pool does not take it in but spills onto it as running water.
- **Running water is a sheet.** See [Running water](#running-water).
- **Seed-derived water pours in.** A free cell beside the sea, a lake or a
  river is an inlet: the water pours in and lands wherever it falls. A pool
  touching seed-derived water trades with it both ways, never past the point
  where their levels meet, and once it stands at that water's level, or fills
  its own cells under it, it joins it: its cells become part of the lake, sea
  or river, and whatever it counted above its own cells returns to it. A hole
  dug under a lake or a trench cut from it therefore ends up lake. The water
  a joined cell holds is booked to the cell; filling the cell with ground
  hands it back. Water taken from a lake lowers the whole lake by the volume
  over its area; water returned to it raises it again. A lake covers at
  least a 32 m drainage cell, so a hole of a few cubic metres lowers it by
  millimetres.
- **Only what changed is kept.** A save holds each pool's seed cell and
  volume, each sheet cell's volume, the parcels in flight, each lake's and river reach's surplus, the sea's and the air's,
  and the cells that joined seed-derived water with what they hold. Loading
  floods every pool out from its seed again.

## Running water

Water that lands where it cannot stand runs (`water/sheet.rs`). A sheet cell
holds a volume on its floor, and pipes to its four horizontal neighbours
carry water between them: the virtual-pipe model of shallow water. Each
pipe's flow gathers speed with the difference in surface height across it
and loses it to friction at 2 per second, and no cell sends more than it
holds, so volume is exact. The flows give each sheet cell a current, which
buoyancy and drag see. Pipes step four times per water step.

- **Slopes and lips.** A sheet climbs a step of one water cell and runs
  down one. A bigger drop is a lip: the water pours over it at the speed it
  ran at.
- **Where it ends.** Running water reaching a pool or seed-derived water
  joins it. A cell whose water is at least a centimetre deep, has no lower
  neighbour and has barely drained for ten steps becomes a pool, which then
  floods its hollow and takes in the sheets there. Water landing in a hollow
  starts a pool at once.
- **Films.** Water shallower than 2 mm clings to the ground and does not
  run; it evaporates.

## Waterfalls

Water pouring over a lip, from a pool or a sheet, is launched in one parcel
a step with its speed over the lip: the weir speed `√(g × head)` from a
pool, the sheet's current from a sheet (`water/jet.rs`). Each parcel falls
along its arc, swept through the water cells 10 cm at a time, until it meets
a pool, seed-derived water, running water or the ground, where it lands. The
water in flight is in the ledger, so a tall fall holds water and a stream
keeps falling for its flight time after its source stops. `WaterStep::falls`
gives each stream's launch point and parcels, which the app draws as a
ribbon whose width follows the rate. Water less than 2 mm over a lip clings
to it.

## The cycle

The seed world is a steady state of a water cycle. Rain falls on land, runs
down the rivers, through the lakes and into the sea, and the sea gives the
same back to the air. None of that steady state is stored or stepped. What
is stored is how far each body has been moved from it: the water a river
reach, a lake, the sea or the air holds beyond its untouched share, negative
where water was drawn out (`water/cycle.rs`). `WaterWorld::ledger` sums every
entry, and its total changes only through `deposit` and `withdraw`.

- **Rain.** Each drainage cell gets 4,000 mm a year at neutral humidity,
  scaled by `1 + humidity` from the climate. That is game-wet: a stream at
  its source carries about 0.15 m³/s. A reach's discharge is the rain on its
  whole catchment. It is a constant in code, `RAIN_MM_PER_YEAR`, for now,
  since changing `world.ron` outdates every saved world.
- **Rivers carry a budget.** A reach holds its discharge times its travel
  time in its channel. That, and then its discharge as it refills, is all a
  trench, inlet or pump can draw from it. A reach passes its surplus
  downstream over its travel time, so water drawn at one point reaches the
  river below after the time the water takes to get there. The reach's
  surface drops with its flow, depth following discharge to the 0.6 and its
  current to the 0.4, down to its bed when it runs dry.
- **Lakes balance inflow against spill.** A lake spills over its rim as a
  weir, and in the seed world the spill equals its inflow. A lake drawn
  below its seed level spills less and fills again from its inflow, while
  the river below it runs short. A lake raised above it spills the surplus
  on. Nothing can draw a lake below its bed: it gives at most the water its
  hollow holds.
- **The sea receives.** Rivers deliver to it, and the sea gives endlessly
  to anything a world can hold, booking what it gives.
- **Evaporation.** Pools and running water lose 5 mm an hour from their
  surface to the air, so a forgotten puddle dries in a day. The air rains it back out within
  an hour; the ledger books it to the sea, where it would end up.

Reaches that run through a lake are that lake's water. A lake with no river
leaving it spills straight to the sea.

Status: implicit water, its preview and rendering, buoyancy and drag on the
CPU route, and stored water in the app are done. Pumps have their API
(`WaterWorld::deposit` and `withdraw`) but no part uses it yet. Pore water and
mud are open.

## Implicit water

Traced when a world compiles, on the 32 m drainage grid
(`mechanic-world/src/generation/water.rs`):

- **Sea.** Connected ground below `sea_level` covering at least `sea_area`
  km². Rivers drain to it and to the world's edge.
- **Lakes.** Every hollow the priority-flood fill raises, at least
  `lake_depth` deep, filled to where it spills. A basin below sea level that
  is cut off from the sea is a lake at its rim, not sea.
- **Rivers.** Each segment's level, interpolated along it, with a surface
  current along the segment that grows with its catchment.

A point is water when the ground there is open, it lies below its column's
level, and the open space belongs to the surface. Space a carve layer opened
counts only where the ground would be open without it, or where the carve
breaks the sea floor, so ocean trenches flood and buried tunnels stay dry.

No water may stand against air. Three rules keep it held:

- **Reach.** A sea or lake reaches two drainage cells (64 m) past its own
  points, over ground at or above its level. A river's water reaches 8 m past
  its channel. Within that reach the terrain draws the shore wherever it
  crosses the water, so most shores are untouched ground. Water never reaches
  over lower ground, so an outlet is not flooded.
- **Shores.** Only beyond the reach, where the ground still lies below the
  water, is it raised to `shore_margin` above the water, falling away at
  `shore_slope`. A shore never applies where another body is wet, so a river
  leaving a lake is not dammed.
- **Sealed roofs.** Below the water line of lakes, rivers and their shores,
  every carve layer keeps 4 m of rock over its voids. Above it, tunnels and
  ravines open as authored. Under the sea they are not sealed.

`TerrainField::water_surface`, `is_water` and `water_level_range` answer the
queries. Tests sample a few hundred thousand water points around lakes,
rivers and the sea and require fewer than one in a thousand to touch dry open
ground. Where channels run close, the river surface blends their levels so it
has no step, and a river crossing the reach of a lake far below it keeps its
banks.

## Rendering

`mechanic_world::water_sheet` meshes one square tile as a grid at each
column's level. The sheet is not clipped to the shore: it runs on under
ground above the water, where the terrain hides it, so a shoreline is wherever
the terrain crosses the surface at any level of detail. It is cut only where
open ground at the surface is not water, such as a dry void under a lake.
A dry vertex beside the water carries the surface on under a bank that stands
above it, so a sheet ends inside the ground instead of at its grid. Each
vertex carries the depth of water under it (to 16 m) and its current.

The app (`world/water_render.rs`) keeps a quadtree of tiles around the camera:
2,048 m tiles split down to 64 m ones, with 64 cells each, out to 4 km. Tiles
mesh on worker threads, and a tile the camera leaves stays until everything
that replaces it is ready. `water_material.wgsl` colours by depth, fades to
clear at the shore, ripples along the current, and lights through Bevy's
PBR path, so the sky reflects in it. Water neither casts nor receives shadows.

Stored water draws after each water step. A pool is a flat quad over each
open column. Running water is a quad per cell whose corners stand at the mean
surface of the running cells around them, so a sheet down a slope is one
surface, and it carries each cell's current, so its ripples run with it. A
stream in flight is two crossed ribbons along its arc, wider as more water
pours. Lakes and rivers moved from their seed level mesh again once they
move 2 cm.

`MECHANIC_WATER=off` hides it. Looking over a lake at 4112 × 2524 on an M1 Pro,
the frame rose from 41 ms without water to 45–47 ms with it.

## In the app

`world/water.rs` steps the world's `WaterWorld` at 20 Hz, at most three
steps a frame, and publishes a `WaterSurfaces` view after each step. Every
committed terrain edit (brush, spoil and soil alike) goes through
`commit_terrain_edit_result`, which tells the water which bricks changed:
pools there are measured again, and free cells beside seed-derived water
become inlets. Loading a world floods its pools out again and finds the
inlets in every edited brick.

Stored water is saved in `water.ron` beside `material.bin`. A world without
one has no stored water.

`world/water_render.rs` draws each stored pool as a flat quad per surface
column, drawn again when its level moves 5 mm or its surface changes shape,
and falling water as crossed ribbons. Water under seed-derived water has no
surface of its own: a pool filling under a lake and a stream falling under it
are not drawn. Cells that joined a lake are drawn only where the lake's own
sheet does not reach, such as a trench cut into its bank. A lake drawn down by more than 2 cm
since its tiles were meshed has them meshed again at its new level.

## Floating

The CPU route floats bodies (`mechanic-physics/src/buoyancy.rs`). Each
dynamic collider is cut once into probes of at most 12.5 cm, each a volume at
a point in its body; a cylinder is probed from its exact hull, not its sixteen
overlapping tangent boxes. Each tick looks the water up once per collider,
through `WaterSource`: the app answers it with the world's `WaterSurfaces`, so
bodies float in stored pools and drawn-down lakes alike. Every substep then adds,
beside gravity:

- **Buoyancy:** the weight of the water a probe displaces, applied at the
  probe. A probe goes under gradually over its own height, so a body crossing
  the surface feels a smooth force.
- **Drag:** pressure drag on the collider's frontal area (the face of a cube
  of its volume, shared among its probes) plus light viscous damping, against
  the body's motion relative to the current. It can slow a probe to the
  current within a substep but never reverse it.

A 1 m wooden cube settles 0.7 m deep, a 1 m steel cube sinks at about
11 m/s, and a floating block rides a 1 m/s current. Bodies do not displace
the water: a lake's level ignores what floats in it. A hull counts only the
solid material it is built from, so a closed steel boat sinks. The GPU
route has no buoyancy.

Spawns keep a metre above any water within ten metres.

## Known limits

- A lake's level falls by the water drawn over its seed area. As a real lake
  shrinks, its shores would move in; here its surface just drops.
- A cell dug into a river keeps its water when the reach runs dry.
- Water poured into a lake shows only up to 15 cm above its seed level; the
  rest spills on within the hour.

- A pool cut in two by new ground stays one pool, at one level.
- Water crosses between any two water cells that both have open terrain
  cells, so a wall thinner than 20 cm leaks.
- Where a surface-breaking carve meets a buried part of the same layer below
  a water level, the two can meet at a wall of water. Stored water will
  resolve this by flowing when the region wakes.
- Rivers are still traced before carves, so a ravine does not redirect them.

## Authoring

`world.ron`'s `water` section holds `sea_area`, `lake_depth`, `shore_margin`
and `shore_slope`. Fewer, deeper lakes come from a larger `lake_depth`.
`worldgen-preview` paints water by depth in `relief.png`, `biomes.png`, the
sections and the views, and reports the lake count.
