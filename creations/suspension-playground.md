# Suspension — Motion Playground

Load **Suspension - Motion Playground** from saved creations and start simulation.
The 6 × 5 m platform has six independent, color-coded stations with identical
plastic load blocks. Equip Connector and aim at a component to adjust its world controls.
Its base starts at the Garage's 5 m build height, matching the linear-bearing
playground, and has no floor weld. Saved-creation loading also centers incoming
creations in the editable Garage volume.

| Color | Station | What to watch |
| --- | --- | --- |
| Gold | Spring only | Undamped bounce under the load |
| Orange | Preloaded spring | The same spring with 75 mm preload holds the load higher |
| Teal | Light damping | A spring and upright shock with 1× compression / 1.6× rebound |
| Blue | Heavy damping | The same assembly with 30× / 48× damping settles slowly |
| Purple | Shock and bump stop | No spring: the load sinks until the rubber supports it |
| Pink | Inverted assembly | Spring, reversed shock body and rubber stop, with 2× / 3.2× damping |

Gold/orange/teal occupy the negative-Z row; blue/purple/pink the positive-Z row.
All springs use 750 mm length, 160/137.5 mm OD/ID and 16 active coils, yielding
about 3 N/mm. Shocks are 750 mm extended with 100 mm bodies. Stops are 100 mm
long and 65 mm wide. Mounts remain rigid, allowing axial motion only.

Stop and restart simulation to compare the initial drop again. Remove a coil
or shock independently, recolor components, or adjust preload and damping.
The shock-only station also makes rubber contact easy to inspect.

Regenerate and verify with:

```sh
cargo run -p mechanic-gpu --example suspension_playground
```

The generator validates graph construction, RON round-trip and compilation,
then runs the platform with a grounded test base for 600 GPU ticks. It removes
the test ground weld, raises the saved assembly into the Garage, and validates
the resulting file before writing it.
Verified on Apple M1 Pro / Metal with zero failure flags and finite transforms.
