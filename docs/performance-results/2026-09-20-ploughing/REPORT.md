# Tools that cut soft ground — 20 September 2026

A saved world holds a four-wedge face drill: a terrain-anchored frame, a servo
piston pressing a disc with four wedge teeth straight down, and a bearing
turning the disc on a gas and an electric engine together (6.5 kN·m, 360 rpm
target). In the app the disc stopped dead as its teeth touched grass and never
dug. `world-drive --soil --tool` replays that save headlessly on an Apple M1
Pro, release build. `--tool` adds breakage with counted spoil, and reports joint
motion, ground load and drive effort each second.

| Measured over 52 s after touchdown | Before | After |
| --- | ---: | ---: |
| Spin one second after touchdown | 0 rad/s | 2 – 6 rad/s |
| Turned in total | 0.2 rad | 214 rad |
| Piston advance into the ground | 7 cm in 18 s, then stops | 84 cm |
| Cells broken out per second | not replayed | 75 – 145 |
| Slip work per second | 0 – 300 J | 9 – 14 kJ |
| Motor effort | 6.5 kN·m, stalled | 5.6 – 5.9 kN·m, turning |
| Servo feed | 180 kN | 105 – 165 kN |
| Degraded ticks | 0 | 0 |

Raw records: [before](drill-before.jsonl.gz), [after](drill-after.jsonl.gz).
"Before" is the parent commit's solver with only the reporting added.

## What was wrong

1. **The servo is a 30-tonne press and the ground was rigid until edited.**
   The teeth met 25 – 30 MPa against a 15 – 40 kPa bearing capacity. Coulomb
   friction under 180 kN is about 43 kN·m at the tooth radius, against 6.5 kN·m.
2. **A stalled tool does no work**, and breakage needs work, so nothing dug.
3. **Compaction followed contact points, not contact patches.** A 25 cm edge
   was a 12.5 cm-radius disc around each end, its pressure the per-point load
   over the whole manifold's area.
4. **Pits hold tools.** Once cells left, teeth sat on the sloped sides of their
   own dimples. Under 200 kN a 5° slope is 17 kN sideways with no friction at all.

## What changed

- A terrain manifold loads an **oriented rectangle** (`LoadFootprint`), never
  narrower than half a cell and up to 2 m across. Soil and breakage take one
  patch per manifold with the manifold's whole load.
- Collision chunks carry per-vertex compaction, so a contact knows the
  **hardened bearing capacity** of the ground it touches.
- **Ground pressed past that strength is failing.** It still carries the body,
  straight up, but holds it sideways with no more than its strength over the
  footprint. Rock, ore, other bodies and any contact within the ground's
  strength are untouched. Rolling cylinders are exempt: against a rigid mesh a
  wheel is a line, but on soft ground it sinks until its patch carries it.
- **Soft ground driven sideways hard enough is crushed** without slip: the
  horizontal share of a normal load beyond four times the material's breakage
  stress earns breakage work at the soil law's rate. Weight resting on level
  ground has no horizontal share and never mines.
- Broken material **leaves along the tool's motion**: half the tool's surface
  speed plus 0.5 m/s off the surface, at most 4 m/s.
- In the app, compaction holds its 10 Hz commits while broken cells wait, since
  a material transfer needs an idle edit queue.

## Checks against regressions

`car-drive --ground soil` and `--ground sand` travel 2.390 m and 2.412 m, equal
to the parent commit on the same floors to four figures. Before the cylinder
exemption they travelled 1.63 m and 0.57 m: a wheel's line contact read as
permanently overloading fresh ground.

## Not measured, and known limits

- **The app was not run.** The replay applies edits synchronously and counts
  spoil instead of simulating clumps. In the app, extraction waits on the
  asynchronous edit queue and the 256-clump budget, so a hole is expected to
  clog with its own spoil. Thrown clumps are covered by a unit test only.
- Spin stays far below its 37.7 rad/s target while the servo feeds at full
  force: the teeth are pressed into fresh ground as fast as it clears. A slower
  feed should let the head spin up; that was not tried.
- Steep faces (normal Y below 0.25) stay rigid until crushed. The replay shows
  brief 1 – 5 MN load spikes when a tooth meets one. No tick degraded.
- A heavy body on tiny non-rolling feet slides easily over soft ground it
  overloads. That is the intended reading of "the ground is failing", but it has
  not been played with.
- Every record has `kernel_coverage_complete: false`. No scale gate is claimed.
