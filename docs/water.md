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
  both arms without pressure being simulated. Natural, dry ground the pool
  would stand less than 30 cm deep on it takes in a ring of cells at a
  time, at 0.25 m/s, the pace of a flood a few centimetres deep over grass:
  otherwise a pool a few centimetres over a flat floods the whole flat at
  its level in one step, a sheet racing round the contour of every hill
  beside it. Deep ground, ground running water already reached, and ground
  the player dug it takes in at once, so a dug trench or pit fills at the
  level of the water let into it. A saved pool fills out to its level at
  once when the world loads.
- **Water moves across contacts.** A neighbour a pool cannot take in is a
  contact: another pool's cell, seed-derived water, or a drop, where the cell's
  bottom opens onto free space or lower water. Across a contact the higher
  water runs to the lower at a weir rate, `1.7 × 0.2 m × head^1.5` a second,
  where the head is measured over the lip the water must cross. Pools whose
  levels come within 5 mm merge. Water pouring over a drop lands at once
  (see [Drops](#drops)).
- **Pools spill over their rims.** A pool records the highest ground its
  water crossed from its seed; the bottom of a cell over its own water is
  no rim. Ground beyond it that lies lower is past the
  rim: the pool does not take it in but spills onto it as running water.
- **Running water is a sheet.** See [Running water](#running-water).
- **Seed-derived water pours in.** A free cell beside the sea, a lake or a
  river is an inlet, if the player dug it: the water pours in and lands
  wherever it falls. Natural shore stays as the seed left it, even in an
  edited brick; pouring there would feed every patch of land along the
  shoreline at once, a flood racing along the contour at the lake's level
  and never stopping when the dig is filled in. Into
  running water it fills the column, however deep, half the way to the
  source's level each step, up to where seed-derived water over it begins,
  while the head over the lip is at least 2 mm. Cells joined to it pour in
  too: they only ever lie under it, so a pit dug under a lake's edge is lake
  and fills the rest of the pit. A pool touching seed-derived water trades
  with it both ways, never past the point where their levels meet, and once
  it stands at that water's level, or fills its own cells under it, it joins
  it: its cells become part of the lake, sea or river, and whatever it
  counted above its own cells returns to it. A hole dug under a lake
  therefore ends up lake; a trench cut from it fills with running water at
  the lake's level. The water a joined cell holds is booked to the cell;
  filling the cell with ground hands it back. Running water or a pool buried
  by fill rises to the first open space above; it never stands inside the
  ground, where a pool with no room would draw on the lake beside it for
  ever. Water taken from a lake lowers the whole lake by the volume
  over its area; water returned to it raises it again. A lake covers at
  least a 32 m drainage cell, so a hole of a few cubic metres lowers it by
  millimetres.
- **Only what changed is kept.** A save holds each pool's seed cell and
  volume, each sheet cell's volume, each lake's and river reach's surplus, the sea's and the air's,
  and the cells that joined seed-derived water with what they hold. Loading
  floods every pool out from its seed again.

## Running water

Water that lands where it cannot stand runs (`water/sheet.rs`). A sheet cell
holds a volume on its floor, and pipes to its four horizontal neighbours
carry water between them: the virtual-pipe model of shallow water. Each
pipe's flow gathers speed with the difference in surface height across it
and loses it to friction, and no cell sends more than it holds, so volume is
exact. Friction is 0.5 per second plus bed friction by Manning's law,
`g n² |v| / h^(4/3)` with `n` = 0.03, which grows as water thins: a film
barely creeps while a stream runs at a few metres a second. A pipe is as deep
as the water over its face, up to a metre, so a film is pushed as a film:
together with Manning's friction this gives Manning's law, and a flood 10 cm
deep spreads over flat grass at a few tens of centimetres a second. Water
reaches ground only by flowing there. A sheet rests on the drawn ground at
its column's centre, read from the terrain's density as the mesh draws it,
within half a terrain cell of the ground its open 5 cm cells make. Counted
from those cells alone, as the mean top of the ground over the column, a
slope gentler than one terrain cell a water cell is a stair of flat terraces
5 cm high: water spread across each terrace, along the ground's contour and
round a hill, as readily as down it, and its depth rose and fell in stripes
down every slope, which showed as bare teeth in thin water. The height is
kept per cell until the ground there is edited. The
flows give each sheet cell a current, which
buoyancy and drag see. Pipes step in substeps of at most 12.5 ms, four per
water step.

- **Slopes and lips.** A sheet climbs a step of one water cell and runs
  down a drop of up to five cells, a metre, as a steep chute. A taller drop
  is a lip: the water pours over it and lands at once (see [Drops](#drops)).
- **Storage.** Sheets live in dense tiles of 32 × 32 columns
  (`water/grid.rs`), one sheet per column. Where each face of a sheet leads
  is worked out once from the ground and kept until the ground under it
  changes; standing water in the way is looked up once per step, and not at
  all where the neighbour already runs. The pipes themselves are plain
  arithmetic over the tiles.
- **Ponds are running water.** Under open sky a sheet may be as deep as it
  likes; its faces are worked out from the cell its surface is in, so a pond
  looks over its banks from its top. A hollow fills and spills on as running
  water, flat when it comes to rest. Earlier, water resting in any hollow
  became a flat pool over the 20 cm cells, and a flood down lumpy ground broke
  into pools stepping down the slope like contour lines.
- **Sleep.** A tile whose water neither changes nor moves by more than
  0.02 mm a step for 40 steps sleeps: its pipes do not run. Water arriving,
  higher water beside it, a deposit or a change to its ground wakes it, and
  it wakes for one step in every 200 to follow slow changes.
- **Where it ends.** Running water pours into a pool or seed-derived water
  through its faces. A pool beside it or over it takes it in, and
  seed-derived water only over it, as in a hole dug under a lake; every
  cell of a deep sheet joins, and nothing beside it: a pool there would
  flood the whole pit to the lake's level in one step and join it. Running water beside a lake at the lake's level stays running
  water: letting the lake take in whatever touched it spread the lake along
  its contour a cell a step, faster than water flows. A lip whose drop is
  filled with running water up to it leads onto that water.
  Water rising against a roof, the cell over its surface mostly shut, becomes
  a pool: caves, tunnels and U-tubes are pools. These checks run when a
  sheet's surface enters a new cell, and every 16 steps otherwise.
- **Films.** Water shallower than 2 mm clings to the ground and does not
  run; it evaporates.

## Water in the ground

Running water and pools soak into the ground under them
(`water/soil.rs`). Each column of ground water reaches keeps how much it
holds, how much it can hold and how fast it takes water in, from the
material under it: ground cover and soil hold pores of 0.4 and 0.35 of half
a metre of ground (7–8 L a column), sand 0.4, and rock and ores none. Dry
ground drinks at three times its rate and wet ground at its rate until it is
full (Green and Ampt, simplified); the rates, 0.05 mm/s for soil and
0.2 mm/s for sand, are sped up for play so a film soaks in within seconds
while a pond lasts for hours. Ground more than nine-tenths full is mud
(`WaterWorld::is_mud`). Ground water drains deep towards the sea over some
six hours and gives a millimetre an hour to the air, so it too stays in the
cycle; the ledger counts it as `soil_m3`, and saves keep it.

The terrain shader darkens and glosses wet ground: the worker publishes each
wet column's ground height and the water soaked into it, and the app writes
them into a 256 × 256 map around the camera, 20 cm a texel, that the
terrain material samples. A few millimetres soaked in darken the ground by
two thirds of its full wetness; a column only counts as wet within 40 cm of
the height it was measured at, so a cave under a wet field stays dry.

## Drops

Water pouring over a lip, from a pool at its weir rate or from a sheet at
the rate its pipe carries, lands in the same step wherever the drop leads:
followed straight down from the lip's cell to the first pool, seed-derived
water, running water or floor it meets. Nothing is in flight and nothing is
drawn between the lip and where the water lands. Streams in flight, drawn as
crossed ribbons, were removed: they landed a step late and in parcels, rocked
the water they fell into, and showed as shards over every small drop. A pool
over a drop with running water already in it, however deep that water's
floor, pours into it over its weir instead, by its head over that water, and
stops at its own level. Water set down inside solid ground rises to the
first open space over it. Water less than 2 mm over a lip clings to it.

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

Stored water is one surface (`water/surface.rs`): running sheets, the open
tops of pools, and cells joined to a lake beyond the lake's own sheet, meshed
together in tiles of 32 × 32 columns. Every column of visible water is a
quad, and each corner stands at the depth-weighted mean level of the columns
around it that lie within a metre of each other, so neighbouring columns
share corners: water down a slope is one smooth ramp, a pool meets the stream
feeding it on one edge, and no gaps open between columns at different
heights. Water more than a metre apart is separate, with a fall between.
Shallow water lies over the drawn ground instead: below 2 cm deep, each
column counts at a corner as the ground there plus its own depth, and from
10 cm it counts at its own level, blending between, and the surface's
normal follows the ground as far as the water lies on it. A mean level over
a film and the deeper rills beside it sank the film's corners into the
ground, and the ground showed through in sharp teeth down the slope. Draping
only lifts a corner, and the normal turns only where it lifted: the drawn
ground and a column's counted floor differ by millimetres to centimetres, so
draping a lake's shallow edge would move it off the lake's level in a band
along the shore's contour, lit as a slope. The ground's height is read
exactly where the terrain mesh draws it: from the density at each lattice
corner, as the mesher samples it, or the blend of the eight cells around a
corner where they were edited. Coarser terrain meshes share the finest
mesh's samples at their own corners, which fall on water corners, so they
agree there too. The height is kept per corner until the ground there is
edited. Corners beside dry ground take no depth, so water
fades out at its edges: the shader fades it from 3 mm to 12 mm deep, so
water a centimetre or two over a lump still shows its surface. A lake, sea or river beside stored water
weighs in on the corners they share as metre-deep water at its current level,
so a flood or a dug channel meets it at the lake's level and colour, with no
step between the two surfaces. It does so only where that water shows, not
where its sheet runs on under the bank, or the edge would sink into the
ground in teeth. The lake's own sheet, a metre between vertices, runs on over
running water standing within 10 cm of its level and fades out there, as it
does under a bank, rather than stopping a square short and leaving the dug
ground between the two surfaces bare. That sheet cannot follow a seam between
the two a column wide, so lake columns beside running water at the lake's
level are drawn with the running water, as deep water at the lake's level.
Normals follow the surface, and each corner carries the current, so ripples
run with it. Lakes and rivers moved from their seed level mesh again
once they move 2 cm.

`MECHANIC_WATER=off` hides it. Looking over a lake at 4112 × 2524 on an M1 Pro,
the frame rose from 41 ms without water to 45–47 ms with it.

## In the app

`world/water.rs` steps the world's `WaterWorld` at 20 Hz on a worker
(`WaterRunner`). Each frame hands the worker the water time owed, at most
three steps, whenever its last batch is done, and publishes what that batch
left: a `WaterSurfaces` view for buoyancy, and the pools, joined cells,
running water to draw. A worker that falls behind slows the water
down; the frame never waits for it, except to save. Every committed terrain
edit (brush, spoil and soil alike) goes through `commit_terrain_edit_result`,
which queues the bricks that changed for the worker's next batch: pools
there are measured again, and free cells beside seed-derived water become
inlets. Loading a world floods its pools out again and finds the
inlets in every edited brick.

Stored water is saved in `water.ron` beside `material.bin`. A world without
one has no stored water.

The worker meshes the surface tiles whose water changed, each tile
fingerprinted to the 2 mm, and `world/water_render.rs` swaps just those
entities in; unchanged tiles stay, and all of them follow the floating
origin. Water under seed-derived water has no surface of its own: a pool
filling under a lake is not drawn. A lake drawn down by more than
2 cm since its tiles were meshed has them meshed again at its new level, and
lake tiles over cells that joined or left the lake mesh again.

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

## Performance

`cargo run -p mechanic-bench --release --bin water-breach [seed]` cuts a
trench through a lake's bank above open land (seed 1 by default: a lake over
a 17 m hillside) and steps the water for 120 s (`BREACH_SECONDS`), printing
JSONL per simulated second and a summary with each phase's p95
(`WaterPhases`). On an M1 Pro the flood peaks at about 55,000 running cells,
with a step p95 of 25 ms. Most cells then are films a few millimetres deep
spread over the slope.

The ground is read per water cell and cached: a cell the field bounds as
wholly air or wholly ground costs one interval test, and only cells the
surface crosses sample the field's density.

## Known limits

- A lake's level falls by the water drawn over its seed area. As a real lake
  shrinks, its shores would move in; here its surface just drops.
- A cell dug into a river keeps its water when the reach runs dry.
- Water poured into a lake shows only up to 15 cm above its seed level; the
  rest spills on within the hour.

- A pool cut in two by new ground stays one pool, at one level.
- A pool still floods a flat the player dug at its level at once: only
  natural ground makes it spread as a front. Pacing dug trenches too left
  pools there overfull, pressing water far above its source. An overfull
  pool pours over a lip by no more head than its cells hold.
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
