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
scene stays resident and keeps owning drive resolution and every buffer the
renderer reads; only the tick changes. Anything other than `cpu` selects the GPU
runtime.

The CPU contact scene updates every frame from the terrain chunks around the
bodies, publishing only chunks that appeared, remeshed or left. It never waits
for the GPU terrain preparation. Ticks wait for local terrain streaming once per
floating origin (world entry); after that, terrain still streaming ahead of a
moving vehicle never holds physics.

Closed mechanism loops run on the CPU route, such as a double wishbone closed
with a Join weld. A Dimension Link freeze holds its creation on both routes: held
bodies get a huge generalized inertia, so everything else meets them as immovable,
and their pose, joints and rest are restored after every substep. An invalid tick input hands the last published state to the GPU
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
3. Queries contacts. Each collider's margin is the 2 cm speculative gap, the
   fall under gravity, and how far its body can carry it in a tick (ancestor
   rotation and suspension travel included), capped at 5 cm. Submerged collider
   vertices from the recovery query are added, since a clipped manifold misses
   a tilted body's buried corner; one beside a clipped point replaces it.
4. Runs 4 substeps. Each one:
   - factors `M + implicit suspension slope`
   - integrates gravity, gyroscopic bias and suspension force
   - warm starts
   - runs one biased Gauss–Seidel pass over drive, limit and contact rows
     (8 when a contact arrived faster than 5 cm per substep)
   - sweeps colliders travelling more than 5 cm in the substep, and stops short
     of an arrival that would end more than 1 cm deep
   - advances positions
   - runs one unbiased relaxing pass (8 after a fast arrival)

   Contacts are queried again before the next substep when a collider has
   travelled past its margin, a body has turned more than 0.25 rad, or the sweep
   cut the substep short. Impulses carry over by contact feature.
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
- **Loop closures** are solved after limits and before contacts (below).
- Warm-start impulses carry across ticks, keyed by contact feature.

### Closure rows

The compiler turns a loop into a spanning tree plus closure bearings, which have
no coordinate. Each closure is a set of soft rows between its two bodies, built
from the same generalized point and angular Jacobians as contacts:

| Closure | Held | Free |
| --- | --- | --- |
| Rotational | anchor (3 rows), axis direction (2) | turning about the axis |
| Linear | rail line (2), orientation (3) | travel, with stop rows at its ends |
| Suspension | as linear | travel, pushed by the spring and damper |

- **Block solve.** The anchor rows and the orientation rows are each solved
  as one small block (`J·H⁻¹·Jᵀ`, up to 3 × 3). Separate Gauss–Seidel rows on
  one anchor converge too slowly at a single pass.
- **Redundant directions are dropped.** A direction the tree already holds,
  such as a planar linkage's out-of-plane motion, cancels to rounding noise.
  Solving it flung a dropped four-bar to the speed limit on its first contact.
- **Anchor rows act at the anchors' midpoint.** Taken at each body's own
  anchor, a loaded loop's soft gap gave rigid motion of the whole loop a
  gap-long lever, so the redundant row survived. The builder cart's strut
  closures then held its tail up and slowly lifted it about the wheels.
- **Softness.** 60 Hz with damping ratio 2, capped at a quarter of the
  substep rate. The relaxing pass is unbiased. A loop seeded open closes at no
  more than 3 m/s instead of snapping shut.
- **Warm starts.** Impulses carry across ticks, and reset after a degraded
  tick.
- **Suspension.** A suspension closure's spring and damper are applied
  explicitly along its rail.
- **Diagnostics.** `closure_position_error` and `closure_angle_error` report
  the widest gap after each tick.

Measured on the saved car (release build, Apple M1 Pro), 600 ticks:

- 0.49 ms per tick at p95
- 1.3 mm deepest penetration on a 4 m/s cold drop
- 1.3 mm at rest

More iterations didn't change the car. 8 substeps made the landing deeper. The
captured ledge blocks rock slightly (≤0.2 rad/s) whatever the settings, so the
defaults stay at 4 substeps × 1 iteration.

### Fast collisions

Speed is handled in three layers, as in Box2D v3:

1. **Speculative contacts** cover the next 5 cm of each collider's travel.
2. **Re-queries** keep that window ahead of a collider moving faster.
3. **A continuous sweep** checks each fast substep along its exact
   reduced-coordinate path (`sweep_new_contacts`), ignoring pairs already
   touching. If the collider would end more than 1 cm into the target, positions
   advance only to 1 mm short of the arrival, velocities are kept, and the
   re-queried rows take the approach out. The whole machine stops there for the
   rest of that substep.

Choices that measurements forced:

- **Margins stay small.** Within a wide margin, a tilted collider's gap is
  measured up its side faces, so most points of a far manifold sit at the
  margin instead of on the corners that arrive. An 80 m/s block landed on one
  row and tipped over.
- **A collider already at the gap is never held.** Holding it until its rows
  caught up stalled stacks and articulated crashes for whole ticks.
- **Extra passes only for arrivals above 5 cm per substep.** A single pass loads
  a fast manifold unevenly and friction spins the body (a 120 m/s cube dropped
  flat left spinning at 104 rad/s). Applying the extra passes from 1 m/s slowed
  driving by 40% and kept landed blocks rocking.
- **Restitution runs once.** Repeating it over coupled contacts added energy.

`fast-impacts`, deepest penetration (release build, Apple M1 Pro, 90 ticks):

| Case | Before | After |
| --- | --- | --- |
| 25 cm cube dropped at 30 / 60 / 120 m/s | 3.1 cm / 0 / 3.0 cm | 3.0 / 1.9 / 2.8 cm |
| 25 cm cube dropped at 250 m/s | 384 m, through the floor | 0.1 mm |
| Cube grazing the floor at 120 m/s | 0 | 1.3 cm |
| Cube into a wall at 100 m/s | 4.0 cm | 2.2 cm |
| 2 m bar spinning at 60 rad/s | 0 | 3.6 mm |
| 1 m block onto a resting block at 80 m/s | 3.8 cm | 0.8 cm |
| Saved car into a wall at 40 m/s | 20 cm | 3.0 cm |

No case tunnels or degrades. Single bodies cost at most 0.1 ms per tick at p95;
the car crash reaches 2.1 ms while it hits. Slow scenes changed by at most 7%.

Known limits:

- closures are soft, so a loaded loop stretches by a few millimetres; a dropped
  four-bar opened up to about 5 mm on landing
- drive budgets use the tree's axis inertia, so a driven bearing inside a loop
  may be stronger or weaker than on the GPU
- contact normals stay fixed between queries
- two colliders that pass completely through each other within one substep are
  not caught; terrain is, because a collider below a surface stays buried in it
- a cut substep loses the rest of that substep's motion for the whole machine
- bodies spawned inside geometry are not rescued
- **Cost scales badly with loose bodies.** Machine assembly and the dense
  `MachineDynamics` matrix cover every body together, so 27 loose blocks
  (`block-pile`) cost about 15.5 ms per tick at p95. Independent bodies should
  become separate islands.
- **Traction under drive is poor.** In `car-drive` the wheels reach 8 rad/s but
  the car covers only 1.4 m in 9 s, and sits about 1 cm deep while driving.
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
cargo run -p mechanic-bench --release --bin cpu-physics -- --scenario fast-impacts
cargo run -p mechanic-bench --release --bin cpu-physics -- --scenario four-bar
cargo run -p mechanic-bench --release --bin cpu-physics -- --scenario reference-fixtures
```

The soft-step scenarios print one JSONL record:

- p50/p95 tick time
- deepest and settled penetration
- degraded ticks
- horizontal travel and fastest final speed

`fast-impacts` prints one record per case with p50/p95 tick time, deepest
penetration, `tunnelled` (a collider ended more than one block past a surface),
degraded ticks, re-queries and continuous hits.

`reference-fixtures` prints one record per captured exact-solver solve: rows,
convergence, residual, iterations, time and worst contact-law violation.

Every scenario exits successfully. Use them to see whether a change helps or
hurts.

## Testing approach

- `cargo test` asserts behaviour with game tolerances: bodies settle,
  restitution bounces, drives move the car, and identical inputs repeat exactly.
- Solver internals (iteration counts, storage bounds, captured tick numbers)
  belong in the bench, not in tests.
