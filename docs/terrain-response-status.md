# Terrain response status

Firm ground, permanent deformation and physical clumps make up one feature. It
is incomplete: rigid ground contact is partly done; persistent soil compaction
and pressure response now run on the CPU route. Material transfer and clumps
have not started.

## GPU terrain contact path

`GpuPhysics::write_terrain_chunks` uploads a complete collision scene into an
independent stackless BVH buffer:

- Global coordinates are rebased in double precision.
- Only active triangle groups are included.
- Geometry and device capacity are checked before the scene is replaced.
- A failed upload leaves the existing scene intact.

The collision kernel clips finite triangles against cuboid, convex and compiled
cylinder colliders. Nearby triangles with identical responses reduce to four
outer support points plus a crown point for curved patches. Normals group when
their dot product exceeds 0.995 and plane separation is under 2.5 cm. The local
table holds 16 groups; beyond that, contacts are emitted unreduced with a
capacity failure flag.

Materials mix terrain and construction responses. `mechanic-world` owns the
terrain friction and restitution tuning.

Penetration depth is kept separate from the physical velocity target, so a
zero-restitution impact never rebounds. Position-only correction resolves
penetration without touching physical velocities. Cached impulses are capped by
the current support budget, so a cached impact cannot launch a resting body on
the next tick.

## Continuous collision

- Translational SAT sweeps use poses captured before each tick's integration.
- Rotational CCD covers unjointed bodies crossing between clear endpoint poses.
  It uses conservative advancement against finite-triangle SAT bounds and
  reconstructs fractional rotation along the Euler root path.
- Not covered: articulated rotational CCD, fast crossings that start in
  overlap, and re-integrating a clamped tick's remaining fraction.

## Measured behaviour (Apple M1 Pro / Metal)

| Fixture | Result |
| --- | --- |
| Flat, crown, bowl and tilted 5 cm meshes, 20 m/s impact | ≤1.09 mm max, ≤1.05 mm settling penetration, no rebound |
| Production-mesher surface at spawn, transition and 300 m | ≤1.73 mm max, <2 mm settling, <1 mm/s upward |
| Generated cave wall and ceiling | <0.95 mm max, <5% kinetic energy after first contact |
| Spinning box, 50 rad/s crossing between clear poses | Caught; 0.0023 mm penetration |
| Boxes, convex parts, cylinders, articulated car, 1- and 65-joint suspension | 5 mm max and 2 mm settling limits met |

Tests: `cargo test -p mechanic-gpu terrain --lib -- --test-threads=1` needs a
real adapter.

## App terrain publication

World simulation uploads the published terrain cut before tick submission:

- `WorldRuntime::physics_terrain` selects only spatial-index owners; hidden
  replacements are excluded.
- `terrain_publication` caches the node and generation list plus the floating
  origin, so unchanged frames don't re-upload.
- Terrain preparation runs on the physics worker. A top-level chunk BVH,
  seam-aware packed geometry reuse and incremental GPU allocations bound upload
  cost.
- Upload errors keep the previous scene.

Both the GPU and CPU routes receive the same accepted cut.

## Persistent soil response (CPU)

CPU contacts report normal impulse integrated over accepted substeps. The world
converts pressure into quantised plastic displacement and hardening for surface
cover, sand and soil. Rock, iron and graphite remain rigid. Each column yields
only its exposed cell; a collapsed cell lets the next cell receive pressure.
Compacted samples retain continuous signed distances during meshing; binary
excavation and fill retain bounded occupancy reconstruction. This prevents a
sub-millimetre first load from snapping a procedural surface to the cell grid.
The response is game tuning, not measured soil data. Only upward-facing ground
(normal Y at least 0.25) compacts; walls and ceilings remain rigid. Each terrain
manifold reports one load over an oriented footprint: as long as its farthest
points, as wide as they stray from that line, never narrower than half a cell.

The app accumulates displacement without promoting terrain. Cells cross a 2 mm
commit threshold and enter the existing asynchronous edit worker at most every
six CPU ticks. One commit delivers at most 12.5 mm per cell. The accumulator is
bounded to 16,384 cells; saturation or a full edit queue drops pending pressure.
Accumulated loads carry the original sample so a later excavation, fill or
compression causes stale loads to be discarded. Normal edit publication,
foundation refresh and autosave also apply to soil.

`MECHANIC_SOIL=off` disables app deformation. The GPU route does not produce
soil loads. Both routes read brick format **v4**: a compaction byte and a
looseness byte per RLE record, with an eight-byte in-memory sample. Earlier brick
versions are rejected; delete `material.bin` in a pre-production world to start
it again on undisturbed ground (the creations in it are kept). The
builder-world fixture has no terrain bricks and needs no format replacement.

Material is never made or destroyed. Every solid cell holds 510 quanta however
packed it is: pressing packs a cell, and a cell pressed flat is squeezed out as
510 quanta of spoil that settle beside whatever pressed it, which is the berm
along a rut. Cells break out whole and settle whole, so the ground's solid cells
and the clumps' quanta always add up. The player's terrain brush is outside
this: it is a creative tool that adds and removes ground freely.
The outstanding driving and collision work below remains open; this change does
not claim those gates or a scale gate.

The headless replay uses a fresh suspension-car world:

```sh
cargo run -p mechanic-bench --release --bin suspension-world -- \
  --write-world /tmp/mechanic-soil-worlds
cargo run -p mechanic-bench --release --bin cpu-physics -- \
  --scenario world-drive --soil \
  --instance /tmp/mechanic-soil-worlds/suspension-performance \
  --warmup 300 --ticks 600
```

Omit `--soil` for a rigid comparison. Replay edits remain in memory. JSONL
reports measured mesh rut depth, summed cell displacement, remesh count/time,
solver p95, and total tick p95 including synchronous benchmark meshing. Every
record keeps `kernel_coverage_complete: false`. The replay respects the saved
instance translation; rotated instances and saved joint coordinates are
explicitly unsupported.

World tests cover light loads, hardening, repeated passes, rigid minerals,
collapse, stale accumulated loads, malformed patches, exact RLE persistence,
and meshed surface height after reload. The CPU load test checks resting weight
with two different substep counts. The app integration test checks cadence,
queue saturation, foundation invalidation and persisted world reload.

See [the CPU soil replay report](performance-results/2026-09-17-soil/REPORT.md)
for measured results and remaining validation limits.

The [large steel cube replay](performance-results/2026-09-18-large-surface/REPORT.md)
reproduces a separate CPU contact explosion. Buried-vertex augmentation now
keeps actual convex vertices instead of reintroducing every clipped triangle
intersection. A 2 m rigid-terrain cube improves from 36.8 to 9.6 ms physics p95
on an i5-12600K; soil remeshing and varied-surface contact counts remain costly.

Contact stress and delivered work now break terrain into world-owned clumps on
the CPU route. Sand, soil, cover and rock settle back into loose terrain, rock
as rubble; iron and graphite stay in pieces and sleep. Terrain and clumps save in one
snapshot. See
[the clump report](performance-results/2026-09-18-clumps/REPORT.md). Extraction
conserves material in every measured run. That report's 256-clump cost of
104 – 114 ms, and its atomic terrain-and-body publication, describe clumps as
solver bodies, which the spoil solver below replaced.

Every cell knows how loose it is. A solid cell holds 510 quanta less its
looseness; undisturbed and brushed ground is 0, and spoil is laid at about 102,
so it takes a quarter to a third more room than the hole it left. A clump is
laid completely, shared evenly over its cells, so no crumbs arise, and no cell is
laid fuller than 459 quanta, so laid ground is always told from undisturbed
ground: a lone cell lies as two. Broken rock is laid the same way and is then
rubble, which breaks out at 60 kPa rather than bedrock's 2 MPa; rocks never
gather into bigger rocks. Loose ground
carries and resists about `(quanta / 510)²` of what undisturbed ground does.
Spoil lies no steeper than its repose: about 34° for soil and cover, 27° for
sand (`mechanic-world/src/edits/repose.rs`). Each laid cell drops to the floor of
its column and runs downhill while the ground beside it is lower than the
repose allows; a steel block holds it up as a bank does, and nothing is laid
under a machine part, close beneath one, or beside a moving one. Loose ground
that is later undercut slides: columns beside every change are looked over, and
a loose top cell that no longer stands is broken out and laid again further
down. Spoil that would run on into a hole a machine works in is not laid at
all, and loose ground left like that waits and is looked at again until the
machine has gone. Undisturbed ground stands at any angle. One
`ClumpCollection::transfer` does laying, sliding and breaking out for the app
and both replays.

Soft ground pressed past its hardened bearing capacity is failing: it carries
the body straight up and holds it sideways with no more than its strength, so a
pressed tool keeps turning and its slip digs. Soft ground driven sideways beyond
four times its breakage stress is crushed without slip, and broken material
leaves along the tool's motion. Rolling cylinders are exempt from the strength
limit. A saved face drill that stalled on contact now bores 84 cm in 52 s; see
[the ploughing report](performance-results/2026-09-20-ploughing/REPORT.md).

Terrain collides from outside only, so a clump that gets under the surface used
to fall without end; clamped at the solver's speed limit it degraded every tick,
and a degraded tick discarded every body's terrain loads, silencing compaction,
breakage and settling for the whole world. Loads are now discarded only by a
tick that rolled back, and a clump buried in solid ground is absorbed into it.

Loose material is no longer part of the machine solve. Each clump moves as a
sphere of its volume against the voxel field, with its own small solver
(`mechanic-physics/src/spoil.rs`): it agrees with an edit the moment the edit
commits and is pushed out of solid ground rather than lost under a one-sided
mesh. Machine colliders push and carry spoil and feel it one tick later as an
impulse per body. Breaking ground out and laying spoil down are ordinary
in-place terrain edits at up to 10 Hz; nothing is prepared, cut over or waited
for, so physics never pauses. Soft spoil becomes ground after 0.25 s at rest in
the lowest free cell within two cells, crumbs of less than a cell merge until
they fill one, and only whole loose cells are laid down. The budget is 4,096
awake clumps; past it, soft ground is laid straight back down. Scattered broken
cells still gather into clods, soft clods that lie touching gather into clods of
up to 27 cells, and anything lying still for a second sleeps, on the ground or
on a deck. Friction holds a heap of clods together as it holds one to the
ground, and a clod held up by others gathers no falling speed. A deck or bucket
holds spoil; only the ground takes it back. On the saved face drill the CPU tick fell from a
44 ms median to 0.22 ms and digging runs continuously; see
[the spoil report](performance-results/2026-09-21-spoil/REPORT.md).

## Open work

1. Articulated rotational CCD and crossings that start in overlap.
2. Profile the serial position solve on representative published terrain.
3. Stable scripted driving of the suspension car on generated terrain. The
   installed car still overturns in the scripted sequence.
4. Soil follow-up: remesh performance, a close-up visual rut demonstration, and
   a compact GPU load readback.
5. Clump follow-up: spoil that stacks and rolls by its shape, looseness in the
   grip yield of collision meshes, a tint for loose ground, an identical-tool
   resistance
   comparison using a fixture with a realistic mass and an external feed force,
   the validation gaps listed in the clump report, and an in-app visual
   demonstration.
6. Ploughing follow-up: play the face drill and a spinning wheel on sand in the
   app, feed-limited spin, and steep faces that stay rigid until crushed.
