# Terrain response status

Firm ground, permanent deformation and physical clumps make up one feature. It
is incomplete: milestone 1 (rigid ground contact) is partly done, and soil,
deformation and clumps have not started.

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

## Open work

1. Articulated rotational CCD and crossings that start in overlap.
2. Profile the serial position solve on representative published terrain.
3. Stable scripted driving of the suspension car on generated terrain. The
   installed car still overturns in the scripted sequence.
4. Then, in order:
   - persistent soil state and pressure response
   - material transfers and runtime clumps
   - atomic terrain/body publication
   - the 256-clump benchmark and a visual demonstration
