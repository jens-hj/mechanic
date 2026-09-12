# Production-mesher contact correctness

Adapter: **Apple M1 Pro / Metal**. Seed: **91**. These are test-profile
correctness fixtures, not release throughput or publication-latency measurements.

Reproduce both generated-surface and generated-cave tests:

```sh
cargo test -p mechanic-gpu terrain::generated --lib -- --nocapture --test-threads=1
```

Each fixture uses 27 production-mesher bricks at 5 cm resolution, including the
real BVHs and material weights. A uniform 1 m box strikes at 20 m/s with zero
construction restitution and 200 GPa modulus. No ground planes are enabled.
Surface fixtures use `(x, z) = (0, 0), (180, 0), (300, 0)` and run 120 ticks.
Cave fixtures raycast from the generated chamber toward +Y and +X, orient the
box to the boundary normal, and run 10 ticks. Gravity remains world -Y.

Penetration uses the real terrain triangles clipped against the box footprint
in box space, with global chunk origins correctly rebased and inactive seam
closures excluded. Surface fixtures assert <=5 mm maximum penetration, <=2 mm
in ticks 101–120, and no upward velocity above 1 mm/s. Cave fixtures assert
<=5 mm penetration and first-tick kinetic energy below 5% of impact energy,
including angular motion. They do not assert cave settling or COM velocity zero:
an off-centre wall contact can cause supported pivoting, and gravity can pull a
body away from a ceiling.

| Fixture | Triangles | Maximum contacts | Maximum penetration (mm) | Ticks |
| --- | ---: | ---: | ---: | ---: |
| cave_ceiling | 26,244 | 38 | 0.534 | 10 |
| cave_wall | 40,340 | 19 | 0.953 | 10 |
| surface_0 | 18,432 | 4 | 1.042 | 120 |
| surface_180 | 21,844 | 5 | 1.136 | 120 |
| surface_300 | 20,287 | 5 | 1.725 | 120 |

Every recorded tick had zero failure flags. Raw summaries: [impacts.jsonl](impacts.jsonl).
This does not establish rotational CCD, all generated terrain configurations,
articulated impacts on generated terrain, the clump budget, or the 60 TPS gate.
