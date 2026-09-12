# Dense terrain contact correctness

Adapter: **Apple M1 Pro / Metal**. These are test-profile correctness fixtures,
not release performance measurements. No publication latency or 60 TPS claim.

Reproduce:

```sh
cargo test -p mechanic-gpu terrain --lib -- --nocapture --test-threads=1
```

Each fixture drops a rigid 1 m box at 20 m/s onto 1,152 triangles on a 5 cm grid,
split across two chunks. The curved height field is `y = -curvature * (x² + z²)`,
then rotated around Z together with the initial box pose and impact direction.
Gravity stays in world -Y. Rock and construction restitution are zero.

Penetration is the maximum terrain height above the box bottom after clipping
triangles against its footprint in box space. Settling covers ticks 101–120.
Upward velocity is world Y, clamped to zero for the recorded positive maximum.
Every fixture had terrain contacts in all 120 ticks and zero failure flags.

| Surface | Tilt | Maximum penetration (mm) | Settling (mm) | Maximum contacts |
| --- | ---: | ---: | ---: | ---: |
| Crown | 0° | 1.092 | 1.001 | 5 |
| Crown | 20° | 1.092 | 1.049 | 5 |
| Bowl | 0° | 0.998 | 0.998 | 4 |
| Bowl | 20° | 1.008 | 0.998 | 5 |
| Flat | 0° | 1.042 | 1.000 | 4 |

No fixture recorded positive upward velocity. Raw summaries are in
[impacts.jsonl](impacts.jsonl). These results cover the stated gentle curvature;
they do not establish rotational CCD, arbitrary fragmented-mesh capacity, app
publication, soil deformation, clump behavior, or the release benchmark gate.
