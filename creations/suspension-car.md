# Suspension — Sprint AWD

A blue, lightweight timber-frame test car with a 2.5 m wheelbase, 3 m wheel track,
four rubber wheels, independent spring/shock suspension, front steering, and two
gas engines driving all four wheels. The seat faces forward (negative X).

Open saved creations with **P** and select **Suspension - Sprint AWD**. In the
World, enter the seat using Interact (default **E**), then hold **W** to accelerate,
**S** for reverse, and **A/D** to steer. Releasing a key returns its command to zero.
Steering is limited to approximately 11.5 degrees for speed stability; allow a
wide turning circle. Reverse is limited to 12 rad/s at the wheels.

The four suspensions have 250 mm extended length, 57.5 mm available compression,
54.9 N/mm springs, 5 mm preload, 8× compression damping, 12× rebound damping,
and rubber bump stops. Equip Connector and aim at a suspension to adjust it.
The saved car has no ground weld and starts inside the Garage's editable band.

If using the repository copy directly, launch from the repository root:

```sh
MECHANIC_CREATIONS_DIR="$PWD/creations" cargo run -p mechanic-app
```

Regenerate and verify on a machine with a GPU:

```sh
cargo run -p mechanic-gpu --example suspension_car
```

The generator checks RON round-trip and graph compilation, settles the car for
three seconds, accelerates, steers both ways, and releases steering. On Apple M1
Pro / Metal, 1,800 ticks reached **61.0 km/h** with **zero failure flags** and the
chassis up-vector's vertical component above 0.95 throughout sampled checks.
This is a flat-plane functional check, not a terrain or frame-rate benchmark.
The generator reuses the existing front-steered-car fixture's control wiring.

Use the World to try slopes and impacts against the current streamed terrain.
Persistent soil dents and loose dirt clumps are not implemented yet.
