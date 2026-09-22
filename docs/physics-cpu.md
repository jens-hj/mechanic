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

`cargo run -p mechanic-app` steps ticks on the CPU by default. The GPU scene
stays resident and keeps owning drive resolution and every buffer the renderer
reads; only the tick changes. `MECHANIC_PHYSICS=gpu` selects the GPU runtime;
any other value keeps the CPU route.

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
   a tilted body's buried corner; one beside a clipped point replaces it. A
   solid cylinder meets terrain as its exact circle and needs no buried
   vertices (see [Rolling cylinders](#rolling-cylinders)).
4. Runs 4 substeps. Each one:
   - factors `M + implicit suspension slope`
   - integrates gravity, gyroscopic bias and suspension force
   - warm starts
   - runs one biased Gauss–Seidel pass over drive, limit and contact rows
     (8 when a contact arrived faster than 5 cm per substep)
   - sweeps colliders travelling more than 5 cm in the substep, and stops short
     of an arrival that would end more than 1 cm deep. A recovery query at the
     end of the substep runs first: when no contact there is that deep, no
     arrival could cut the substep and the sweep is skipped
   - advances positions
   - runs one unbiased relaxing pass (8 after a fast arrival)

   Contacts are queried again before the next substep when a collider has
   travelled past its margin, a body has turned more than 0.25 rad, or the sweep
   cut the substep short. Impulses carry over by contact feature. Against
   terrain, a cylinder's turn about its own axis counts as neither travel nor
   turn (see [Rolling cylinders](#rolling-cylinders)). Between bodies of one
   construction, travel is measured in its root body's frame, since carrying
   the whole construction changes no distance inside it, and each body keeps
   its own budget, so a spinning wheel refreshes only its own pairs. Rotation
   still counts in the world, because contact normals stay fixed there.
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
- **Failing ground.** A terrain contact knows the hardened bearing capacity under
  it and its manifold's footprint. Once its load passes what that ground carries,
  its rows act straight up and horizontally instead of along the surface, and
  the horizontal disk is limited by the ground's strength rather than the load.
  It releases at half the limit, and a contact re-queried on a collider already
  breaking the ground starts out failing. Rock, ore, body pairs and rolling
  cylinders never fail.
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

### Mesh rows

A mesh between two toothed parts ([gears](gears.md)) is one soft bilateral row,
`S(a) − S(b) = 0`, where a side's surface speed is the velocity of its pitch
point along the common tangent plus, for a worm or screw, the thread advance
times its spin about the thread axis (`gear_mesh.rs`). It is built from the same
point and angular Jacobians as a closure, at every substep's pose, solved after
the closures with the joint softness, warm-started across ticks, and dropped
when it cancels to noise inside one body. Each mesh integrates the slip its row
leaves and feeds it back as the row's position error, so teeth stay phased;
`mesh_slip` reports the worst. The `gear-train` bench holds eight meshes at
0.08 ms per tick with at most 5 mm of slip over ten seconds.

### Rolling cylinders

A solid full cylinder compiles to sixteen tangent boxes. Collided as that
prism, a 1 m wheel's axle rose and fell 1 cm every 20 cm and every facet landing
took speed, so a free wheel stopped or toppled within seconds. The soft step
meets terrain with the exact cylinder instead (`ContactCylinder`):

- **Contacts.** Against each triangle the side supports along its lowest line,
  clipped to the triangle; a cylinder on an end supplies rim points. Where the
  lowest points lie outside the triangle, the lowest point over its edges and
  corners is found exactly, so a kerb edge holds the wheel. Every point carries
  its true depth.
- **Manifold.** A triangle beside the lowest line reports its own nearest point
  higher up the flank. As a group corner it held the wheel ahead of or behind
  its axle and braked it, so a flank point is kept only where no lowest-line
  point of its group lies at or below it: at a kerb or in a crease. A group
  also keeps its deepest point when every corner is shallower. A cap tipping
  flat has its lowest rim point between two corner directions; without it the
  disc sank 5.8 mm.
- **Anchors.** Every terrain contact keeps its place relative to the contact
  normal and the axis rather than the material. A side contact stays under the
  axle as the wheel turns, and a cap's rim points stay where they touch as it
  spins. A material anchor lifts off the circle by `r(1 − cos θ)`.
- **Spin.** Turning about its own axis moves none of a cylinder, so terrain
  contact budgets and sweeps bound it by its centre's speed and its axis's turn
  rate. The spin is a root body's angular displacement or a revolute bearing's
  turn; the rest of the body's rotation still counts.
- **Envelope.** The prism still bounds the cylinder for broadphase, sweeps,
  clearance certificates and contact with other bodies. A sweep hit on the
  prism cuts a substep only if the exact cylinder would end buried, and a
  sweep advances at the cylinder's speed, since the prism's gap never exceeds
  the cylinder's. Budgets against other bodies still count spin, because the
  prism's corners turn.
- **Culling.** A triangle is rejected before any exact contact or sweep step
  by a lower bound on its distance: the cylinder lies within its radius of its
  axis segment, between its cap planes, and within its reach of the triangle's
  plane. A sphere around the triangle tries the axis bound first. Contacts
  count a cylinder under a triangle as buried in it, so beneath the plane only
  the axis's sideways distance counts there.
- **Flanks.** Where the lowest line passes more than 10 µm aside of a triangle,
  its points would all be flank points. Such a triangle is resolved only if no
  point on the lowest line of its support group lies at or below the bound on
  its separation; otherwise the manifold would discard every point it found.
- The exact reference solver keeps the prism: its event search and sweep
  certificates are built on that polytope.

`wheel-roll` rolls a free 1 m wheel (aluminium core, rubber tyre) across 0.5 m
terrain triangles for 600 ticks (release build, Intel i5-12600K):

| Case | Axle height range | Speed kept | Travelled | Lateral drift | Mean query |
| --- | --- | --- | --- | --- | --- |
| Flat, 1 m/s, prism | 10.9 mm | 0% | 2.2 m | 1.7 mm | 0.037 ms |
| Flat, 1 m/s, circle | 0.4 mm | 28% | 6.4 m | 0.1 mm | 0.010 ms |
| Flat, 5 m/s, prism | 143 mm | 4% | 17.6 m | 4.1 m | 0.039 ms |
| Flat, 5 m/s, circle | 0.5 mm | 86% | 46.4 m | 7 mm | 0.018 ms |
| 5 cm swells, 5 m/s, prism | toppled | 0% | 13.5 m | 0.35 m | 0.065 ms |
| 5 cm swells, 5 m/s, circle | 2.2 mm | 86% | 46.2 m | 9 mm | 0.019 ms |
| Flat, 15 m/s, circle | 0.5 mm | 95% | 146.4 m | 0.55 m | 0.034 ms |

The saved car, before and after:

| | Prism | Circle |
| --- | --- | --- |
| `car-drop` p50 / p95 tick | 0.31 / 0.33 ms | 0.09 / 0.10 ms |
| `car-drop` deepest penetration | 1.3 mm | 1.4 mm |
| `car-drive` p50 / p95 tick | 0.32 / 0.41 ms | 0.17 / 0.19 ms |
| `car-drive` travel in 9 s | 0.9 m | 2.4 m |

Rubber on rock rolls with a 5 mm resistance length, which alone leaves about
35% of 1 m/s and 87% of 5 m/s after ten seconds.

Counting spin as travel swept a 5 m/s wheel about twice a tick. Ignoring it,
with the same ride and speed kept:

| Case | Sweeps in 600 ticks | Mean sweep time | p50 tick |
| --- | --- | --- | --- |
| Flat, 5 m/s | 1323 → 0 | 0.14 → 0.001 ms | 0.16 → 0.03 ms |
| 5 cm swells, 5 m/s | 1311 → 0 | 0.13 → 0.001 ms | 0.15 → 0.03 ms |
| Flat, 15 m/s | 2400 → 2400 | 0.38 → 0.32 ms | 0.41 → 0.35 ms |
| `car-drive` | 1327 → 1327 | 0.041 → 0.017 ms | 0.17 → 0.12 ms |

### Driving on fine terrain

`world-drive` replays a saved world: its construction with terrain foundations
anchored, meshed 5 cm terrain around every moving body, and the throttle held.
The saved "double wishbone 2" car has 21 bodies, 3,848 colliders and 1.5–1.6 m
wheels. Every change above kept its speed and contacts identical; release build,
Intel i5-12600K, one-second windows:

| Speed | p50 tick | Mean query | Mean continuous | Sweeps per tick |
| --- | --- | --- | --- | --- |
| 5.4 m/s | 9.5 → 7.0 ms | 8.0 → 5.5 ms | 1.1 → 1.3 ms | 4 → 0 |
| 10.9 m/s | 23.8 → 8.3 ms | 8.7 → 5.4 ms | 15.4 → 2.5 ms | 4 → 0 |
| 14.9 m/s | 58.5 → 12.8 ms | 13.5 → 8.4 ms | 44.7 → 3.9 ms | 4 → 0 |
| 18.0 m/s | 79.0 → 15.3 ms | 16.7 → 10.2 ms | 59.9 → 4.5 ms | 4 → 0 |

Each sweep had checked every wheel against the few hundred triangles under it,
though a rolling wheel never arrives buried. The internal body pairs, mostly
wishbone pipes against each other, were re-queried every substep because the
whole car moved and its wheels spun.

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
  may be stronger or weaker than on the GPU, and a motor on a gear feels only
  that gear rather than the train it drives
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
  the car covers only 2.4 m in 9 s, though it sits about 1 mm deep. Rolling on
  circles helped (0.9 m on the prism); wheel friction and load under drive
  torque need investigating next.
- **Fast travel still re-queries terrain every substep.** From about 5 m/s the
  saved wishbone car queries its contacts every substep, and above 12 m/s,
  5 cm per substep, the end-of-substep recovery check runs every substep too.
  At 18 m/s it still takes about 15 ms per tick, close to the 16.7 ms a 60 Hz
  tick allows.
- **Fast spin in flight grows.** A 0.95 m steel wheel spinning at 150 rad/s
  that leaves the floor with a few rad/s of wobble gains wobble every tick and
  reaches the speed clamp within about 15 ticks. The gyroscopic bias is applied
  once per substep, which at that speed is 0.6 rad of turn.

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
cargo run -p mechanic-bench --release --bin cpu-physics -- --scenario wheel-roll
cargo run -p mechanic-bench --release --bin cpu-physics -- --scenario world-drive --instance <world directory>
cargo run -p mechanic-bench --release --bin cpu-physics -- --scenario reference-fixtures
```

The soft-step scenarios print one JSONL record:

- p50/p95 tick time
- mean sweep time and sweep count
- deepest and settled penetration
- degraded ticks
- horizontal travel and fastest final speed

`fast-impacts` prints one record per case with p50/p95 tick time, deepest
penetration, `tunnelled` (a collider ended more than one block past a surface),
degraded ticks, re-queries and continuous hits.

`wheel-roll` prints one record per floor and speed with the axle height range,
speed kept, lateral drift, axle tilt, mean query, sweep and solve time, and
sweep and re-query counts.

`world-drive` takes a saved world's directory, such as
`~/.local/share/Mechanic/worlds/test`. It prints the construction's size, then
one record per simulated second with the fastest body's speed, p50 and maximum
tick time, and mean query, continuous and solve time, sweeps, re-queries,
candidate triangles and pairs, and contacts. `--soil` compacts the ground
under load. `--tool` also breaks it into spoil, steps the world's clumps with
the spoil solver as the app does, and adds a `tool` object per second: each
joint's position, speed and mean drive effort, ground load, slip work, footprint
sizes, cells broken out and laid down, live clumps, spoil time per tick and the
materials under load. `car-drive` and `car-drop` take
`--ground rock|soil|sand|cover`.

`reference-fixtures` prints one record per captured exact-solver solve: rows,
convergence, residual, iterations, time and worst contact-law violation.

Every scenario exits successfully. Use them to see whether a change helps or
hurts.

## Testing approach

- `cargo test` asserts behaviour with game tolerances: bodies settle,
  restitution bounces, drives move the car, and identical inputs repeat exactly.
- Solver internals (iteration counts, storage bounds, captured tick numbers)
  belong in the bench, not in tests.
