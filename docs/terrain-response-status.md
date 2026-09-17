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
(normal Y at least 0.25) compacts; walls and ceilings remain rigid.

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

## Open work

1. Articulated rotational CCD and crossings that start in overlap.
2. Profile the serial position solve on representative published terrain.
3. Stable scripted driving of the suspension car on generated terrain. The
   installed car still overturns in the scripted sequence.
4. Soil follow-up: remesh performance, a close-up visual rut demonstration, and
   a compact GPU load readback.
5. Then, in order:
   - material transfers and runtime clumps
   - atomic terrain/body publication
   - the 256-clump benchmark and a visual demonstration
