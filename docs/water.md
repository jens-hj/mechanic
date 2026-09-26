# Water

Water follows the terrain's rules. Untouched water is a pure function of the
seed and the world definition: a level per column and a rule, with nothing
stored. Later phases store only where water differs from that, the way edited
bricks override untouched ground. Water shares the terrain's quantum
(`MATERIAL_QUANTUM_M3`) and is conserved, except at declared sources and
sinks: the sea, river inflow and the brush.

## The model

One value, moisture, describes every 5 cm cell. In an empty cell it is liquid
water; in a solid cell it is pore water, which later makes mud. It is stored
per 20 cm water cell (4³ terrain cells, 8³ per brick) as `u16` quanta, and
the 5 cm view is derived by filling a water cell's fine cells from the bottom.

- **Still water is a pool.** A connected body of full water cells is one
  level and one volume. It grows by incremental priority-flood, the algorithm
  the drainage already uses. A U-tube is a single pool, so pressure comes out
  right without being simulated. A spill point is the lowest border cell with
  open space below it that is not part of the pool.
- **Only moving water is simulated.** Flowing cells form a sparse active set
  that sleeps, like spoil clumps.
- **Seed-derived water is implicit.** The sea, lakes and rivers below.

Status: the implicit water, its preview and its rendering are done. Buoyancy,
stored water, pools, flow, pumps and pore water are open.

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
ground; seed 42 measures 0 for lakes and the sea and 23 in 100,000 for
rivers.

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

`MECHANIC_WATER=off` hides it. Looking over a lake at 4112 × 2524 on an M1 Pro,
the frame rose from 41 ms without water to 45–47 ms with it.

## Known limits

- Dug ground does not fill yet. A hole dug inside a lake's footprint shows no
  water until stored water arrives, and edits never move a shore.
- Where a surface-breaking carve meets a buried part of the same layer below
  a water level, the two can meet at a wall of water. Stored water will
  resolve this by flowing when the region wakes.
- Rivers are still traced before carves, so a ravine does not redirect them.

## Authoring

`world.ron`'s `water` section holds `sea_area`, `lake_depth`, `shore_margin`
and `shore_slope`. Fewer, deeper lakes come from a larger `lake_depth`.
`worldgen-preview` paints water by depth in `relief.png`, `biomes.png`, the
sections and the views, and reports the lake count.
