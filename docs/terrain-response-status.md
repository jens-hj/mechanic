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
soil loads. Both routes read brick format **v3**, which adds one compaction byte
per RLE record while retaining an eight-byte in-memory sample. Earlier brick
versions are rejected; regenerate terrain-bearing pre-production worlds. The
builder-world fixture has no terrain bricks and needs no format replacement.

Compression removes volume without conserving material or producing berms.
There are no clumps, material transfers or atomic terrain/body publication.
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
the CPU route. Sand, soil and cover settle back into low-compaction terrain;
rock, iron and graphite stay physical and sleep. Terrain edits and clump changes
publish as one generation and save in one snapshot, and the GPU route rejects a
world that holds clumps. See
[the clump report](performance-results/2026-09-18-clumps/REPORT.md). Extraction
conserves material in every measured run, but the 256-clump replay runs at a
104 – 114 ms physics p95 on an i5-12600K, so no clump scale gate is claimed.

Soft ground pressed past its hardened bearing capacity is failing: it carries
the body straight up and holds it sideways with no more than its strength, so a
pressed tool keeps turning and its slip digs. Soft ground driven sideways beyond
four times its breakage stress is crushed without slip, and broken material
leaves along the tool's motion. Rolling cylinders are exempt from the strength
limit. A saved face drill that stalled on contact now bores 84 cm in 52 s; see
[the ploughing report](performance-results/2026-09-20-ploughing/REPORT.md).

## Open work

1. Articulated rotational CCD and crossings that start in overlap.
2. Profile the serial position solve on representative published terrain.
3. Stable scripted driving of the suspension car on generated terrain. The
   installed car still overturns in the scripted sequence.
4. Soil follow-up: remesh performance, a close-up visual rut demonstration, and
   a compact GPU load readback.
5. Clump follow-up: the 256-clump tick cost, an identical-tool resistance
   comparison using a fixture with a realistic mass and an external feed force,
   the validation gaps listed in the clump report, and an in-app visual
   demonstration.
6. Ploughing follow-up: play the face drill and a spinning wheel on sand in the
   app, feed-limited spin, and steep faces that stay rigid until crushed.
