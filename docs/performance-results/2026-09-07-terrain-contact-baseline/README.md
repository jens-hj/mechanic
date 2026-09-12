# Terrain impact baseline

Recorded on Apple M1 Pro / Metal with the new finite-triangle GPU contact path.
This is a correctness diagnostic, not the requested 256-clump performance gate.

The fixture is an isolated 1 m cube starting with its centre 0.625 m above a
soil triangle. Construction restitution is explicitly zero, construction Young's
modulus is 200 GPa, terrain restitution is zero, and the explicit ground plane
is disabled. One downward impulse is applied at the centre of mass. The solver
uses its general route with eight sweeps and 60 Hz ticks.

| Initial downward speed | Maximum penetration | Penetration at tick 60 | Maximum outgoing upward speed |
| --- | --- | --- | --- |
| 1 m/s | 41.71 mm | 17.69 mm | 0.573 m/s |
| 5 m/s | 78.44 mm | 17.69 mm | 0.963 m/s |
| 20 m/s | 348.14 mm | 17.70 mm | 1.144 m/s |

All 180 samples have zero GPU failure flags. These results **fail** the required
5 mm maximum and 2 mm settling limits and show upward velocity with zero
restitution. They must not be treated as firm-ground acceptance.

Run on a machine with a GPU adapter:

```sh
cargo test -p mechanic-gpu terrain_downward_impact_trace --lib -- --ignored --nocapture
```

`impacts.jsonl` contains only the JSON records from that command. Outgoing
velocity and pose are authoritative readbacks from the same completed tick.
Incoming velocity is calculated from the preceding tick's velocity and the
fixture's gravity/damping step. `recovery_target_mps` is the requested speed
calculated from the penetration-bias formula; it is not a separately measured
impulse contribution. Restitution contributes zero in this fixture.

This baseline establishes the failure before split positional recovery and
swept collision detection are implemented. It contains no publication latency,
throughput, suspension, or clump-budget measurement.
