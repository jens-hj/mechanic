# CPU physics

`mechanic-physics` has two CPU solvers over the same
[compiled machine model](compiled-machine-dynamics.md):

| Solver | Role | Guarantees |
| --- | --- | --- |
| Soft step (`CpuMachine`) | Runs the app's CPU route | Publishes every tick; quality is measured, not gated |
| Exact reference (`reference::CpuJointMachine`) | Offline validation | Publishes only when the original contact laws hold; otherwise returns an error |

Mechanic is an experimental game. Physics must stay live and plausible, so the
app never waits on a solver that can refuse a tick. The exact solver stays
because it has found real geometry bugs: faceted wheels, duplicated cylinder
corners, and vertex activation. It works as a checker, not a runtime.

## Selecting the route

`MECHANIC_PHYSICS=cpu cargo run -p mechanic-app` steps ticks on the CPU. The GPU
scene stays resident and keeps owning terrain preparation, drive resolution and
every buffer the renderer reads; only the tick changes. Anything other than
`cpu` selects the GPU runtime.

A creation the CPU cannot run (closed mechanism loops, for now) is refused with
a message. An invalid tick input hands the last published state to the GPU
runtime, which keeps simulating until the next construction publication.

## Soft-step solver

`CpuMachine` (`crates/mechanic-physics/src/soft_step/`) applies Box2D v3's soft
step to the reduced-coordinate machine. Tree joints are exact by
reconstruction; contacts, drives and joint limits are rows in generalized
coordinates. Each row stores `H⁻¹Jᵀ` once per substep, so an impulse updates
every coupled body at once.

Each 60 Hz tick:

1. Validates commands. Invalid input is the only error.
2. Applies external impulses.
3. Queries contacts once. The margin is the 2 cm speculative gap plus the
   fastest body's travel in one tick. Submerged collider vertices from the
   recovery query are added, since a clipped manifold misses a tilted body's
   buried corner.
4. Runs 4 substeps. Each one:
   - factors `M + implicit suspension slope`
   - integrates gravity, gyroscopic bias and suspension force
   - warm starts
   - runs one biased Gauss–Seidel pass over drive, limit and contact rows
   - advances positions
   - runs one unbiased relaxing pass
5. Applies restitution to contacts that arrived faster than 1 m/s.
6. Publishes. Non-finite results restore the substep start with zero velocity,
   and speeds are clamped to 500. Both mark the tick `degraded`.

Row laws:

- **Normal rows** are speculative while separated (`bias = gap / dt`). While
  overlapping they are soft: 30 Hz, damping ratio 10, 1 mm slop, pushed apart at
  no more than 3 m/s. The relaxing pass removes that bias so pushes don't add
  energy.
- **Friction and rolling** are disks limited by the normal impulse. Static
  friction applies below 0.05 m/s of slip at the start of the tick.
- **Drives** are clamped to `drive_budget` per substep. **Joint limits** are
  normal rows on the coordinate.
- Warm-start impulses carry across ticks, keyed by contact feature.

Measured on the saved car (release build, Apple M1 Pro), 600 ticks:

- about 0.43 ms per tick
- 1.6 cm deepest penetration on a 4 m/s cold drop
- 1.3 mm at rest

More iterations didn't change the car. 8 substeps made the landing deeper. The
captured ledge blocks rock slightly (≤0.2 rad/s) whatever the settings, so the
defaults stay at 4 substeps × 1 iteration.

Known limits:

- no closed loops
- one contact query per tick, so a very fast rotating body can still tunnel
- contact normals stay fixed within a tick
- **Cost scales badly with loose bodies.** Machine assembly and the dense
  `MachineDynamics` matrix cover every body together, so 27 loose blocks
  (`block-pile`) cost about 13 ms per tick. Independent bodies should become
  separate islands.
- **Traction under drive is poor.** In `car-drive` the wheels reach 8 rad/s but
  the car covers only 1.3 m in 9 s, and sits about 2 cm deep while driving.
  Wheel friction and load under drive torque need investigating next.

## Exact reference solver

`CpuJointMachine` integrates with implicit midpoint. It uses:

- nonlinear force iteration with drives, suspension and stops
- event search that splits substeps at each contact arrival or release
- continuous path certificates and split position recovery
- a smoothed-continuation Newton contact solve checked against the original
  unsmoothed contact laws

A tick that cannot meet those laws is retried at 2, 4 and 8 substeps, then
rejected without changing state. The saved car still meets unsolved states
within the first seconds of a cold drop. That is expected for this solver and
is not a bug to chase.

`MECHANIC_IMPACT_CAPTURE=<path>` and `MECHANIC_FORCE_CAPTURE=<path>` write the
exact algebra of a failing solve from test builds. Captured solves live in
`crates/mechanic-bench/fixtures/cpu-reference/`.

## Quality bench

```sh
cargo run -p mechanic-bench --release --bin cpu-physics -- --scenario car-drop
cargo run -p mechanic-bench --release --bin cpu-physics -- --scenario car-drive
cargo run -p mechanic-bench --release --bin cpu-physics -- --scenario block-pile
cargo run -p mechanic-bench --release --bin cpu-physics -- --scenario reference-fixtures
```

The soft-step scenarios print one JSONL record:

- p50/p95 tick time
- deepest and settled penetration
- degraded ticks
- horizontal travel and fastest final speed

`reference-fixtures` prints one record per captured exact-solver solve: rows,
convergence, residual, iterations, time and worst contact-law violation.

Every scenario exits successfully. Use them to see whether a change helps or
hurts.

## Testing approach

- `cargo test` asserts behaviour with game tolerances: bodies settle,
  restitution bounces, drives move the car, and identical inputs repeat exactly.
- Solver internals (iteration counts, storage bounds, captured tick numbers)
  belong in the bench, not in tests.
