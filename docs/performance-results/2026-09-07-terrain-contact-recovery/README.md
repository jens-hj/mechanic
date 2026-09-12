# Terrain impact recovery

Measured on **Apple M1 Pro / Metal**, using the same isolated 1 m cube and rigid
soil surface as the [original baseline](../2026-09-07-terrain-contact-baseline/README.md).
Restitution is zero, construction Young's modulus is 200 GPa, and the explicit
ground plane is disabled. One centre-of-mass impulse starts each 60-tick run.

| Initial downward speed | Maximum published penetration | Penetration at tick 60 | Maximum outgoing upward speed |
| --- | --- | --- | --- |
| 1 m/s | 1.004 mm | 1.000 mm | None; velocity remains nonpositive |
| 5 m/s | 1.011 mm | 1.000 mm | None; velocity remains nonpositive |
| 20 m/s | 1.042 mm | 1.000 mm | None; velocity remains nonpositive |

All 180 samples have zero failure flags. The previous 20 m/s run reached
348.14 mm penetration and 1.144 m/s upward velocity with zero restitution.

The changes separate positional correction from physical velocity, retain
construction compliance, provide up to four support points per triangle, and
limit cached terrain impulses to the current closing-speed budget. Position
corrections use separate linear/angular buffers, joint constraint projection,
and three bounded geometry updates. Gravity and external impulses are applied
once per tick. The physical solver still has finite iteration error: the
settled box's normal velocity is approximately -0.0182 m/s while its published
pose remains supported.

```sh
cargo test -p mechanic-gpu terrain_downward_impact_trace --lib -- --ignored --nocapture
cargo test -p mechanic-gpu terrain --lib -- --nocapture --test-threads=1
```

`impacts.jsonl` contains same-tick authoritative pose and velocity readbacks.
`incoming_y_mps` includes the fixture's gravity/damping step.
`position_correction_y_m` compares the published height with the unconstrained
height predicted from the preceding pose and incoming velocity. The physical
penetration-recovery velocity target is zero; intentional restitution remains
a separate physical response.

Additional rigid-rock regressions cover 25 cm boxes crossing the entire surface
in one tick, 1 m boxes, asymmetric convex parts, cylinders, an articulated car,
and free suspension assemblies with 1 and 65 joints. They enforce 5 mm maximum
and 2 mm settling penetration. The suspension cases exercise both solver routes.
Intentional terrain restitution also has a focused regression.

This is not completion of milestone 1 or the 256-clump performance gate. App
publication, rotational CCD, reduction across densely tessellated triangles,
and broader edge/cave/streaming coverage remain. Recovery currently uses serial
contact/joint sweeps; dense-terrain latency and 60 TPS performance are unmeasured.
No clumps, terrain deformation, or visual demonstration are included.
