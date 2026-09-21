# Loose material outside the machine solver — 21 September 2026

Digging with the saved face drill (`wefwefwefwef`, Apple M1 Pro) ran at about
4 fps and came in bursts: a pile of clumps was dug free, the app nearly froze
until they were absorbed, then the next pile. This change makes spoil cheap and
removes every pause, so digging and settling run continuously.

## What was slow

Measured in the real app on a disposable copy of the save
(`scripts/run-background-capture.py --from-start`, 60 s from world entry,
unfocused window, 4112 × 2524, MSAA × 4):

| Cost | Before |
| --- | ---: |
| CPU physics tick, median (119 clumps, ~2,700 contacts) | 43.8 ms |
| Ticks run in 60 s | 768 of 3,600 |
| Rebuilding the whole collision scene and solver per material transfer | 358 ms median, 12.3 s in total |
| Material transfers in 60 s | 36, every one pausing physics |
| Cells dug in 60 s | 0: the clump budget was full of clumps waiting to settle |
| Streaming backlog while digging | stuck near 1,200 nodes |

Clumps were full bodies in the articulated soft-step solver, about 0.3 ms each.
Every transfer deep-copied every active terrain chunk into a fresh collision
scene and rebuilt the solver, and physics waited from the start of a transfer to
its cutover. Settling needed one simulated second, which at 12 ticks a second
took five real ones, so the budget of 256 clumps stayed full and digging stopped
until it drained. Each change in clump count also forced a whole-cut terrain
reselection. One guess was wrong: the search for deposit cells cost 0.6 ms.

## What changed

- **Spoil solver** (`mechanic-physics/src/spoil.rs`). Each clump moves as a
  sphere of its volume against the voxel field itself: trilinear density with its
  gradient, read through a per-brick cache that an edit invalidates. It agrees
  with an edit the moment the edit commits, is two-sided, and pushes a buried
  clump out, so nothing falls under a one-sided mesh. The machine's colliders
  push and carry spoil as moving shapes; the summed reaction returns to the
  machine as one external impulse per body on the next tick, so a loaded bucket
  weighs what it holds. Clumps separate from each other through a spatial hash.
  Speed is capped at 15 m/s.
- **Transfers are ordinary edits.** Breaking ground out and laying spoil down
  happen in place on the persistent octree, between edit batches, at most every
  100 ms. There is no prepared copy, no cutover, no scene rebuild, and physics
  never waits. The mesh follows as after any other edit.
- **Settling.** Soft spoil becomes ground after 0.25 s at rest, in the lowest
  free cell within two cells, so it fills the hollow it lies in before it heaps.
  Cells inside a machine collider are left free. Crumbs of less than a cell
  that rest in the same three-cell block merge until they fill one; only whole
  loose cells are laid down. The earlier partial-cell deposit, which marked
  loose spoil as fully hardened ground, is gone.
- **Budget.** 4,096 awake clumps instead of 256. Past it, freshly cut soft
  ground is laid straight back down instead of stopping the dig.
- **Rendering.** One shared lumpy clod mesh per material, scaled per clump, so
  clumps batch.
- Clumps no longer steer terrain streaming or the physics terrain cut, and the
  GPU route no longer refuses a world that holds them.

The save format is unchanged.

## Follow-up the same day: rest and conservation

Played in the app, clods on a steel deck kept spinning, and some on the ground
crept from facet to facet and never settled. A clump now lies on a collider as
it lies on the ground: it rolls only with how it moves over what carries it,
static friction holds it where it stopped on a slope it can hold, and anything
still for a second sleeps until the deck moves or the ground changes. A deck or
bucket holds spoil; only the ground takes it back.

Material is now conserved exactly. Pressing used to destroy it twice: a packed
cell counted as `510 − compaction` quanta, and a cell pressed flat vanished.
Every solid cell now holds 510 quanta however packed, and a cell pressed flat
leaves as 510 quanta of spoil beside whatever pressed it: the berm along a rut.
The replay checks the balance and fails if a quantum is unaccounted for; over
3,000 ticks the ground gave up 1,524 cells and the difference was 0.

In the save's deepened bore, spoil fell back under the head, could not settle
there and was stirred for ever; live clumps passed 1,600 and kept climbing. Soft
clods of one material that lie touching now gather into clods of up to 27 cells.
The same replay holds 110 – 170 clumps at 0.2 – 0.35 ms a tick. Spoil in flight
is not gathered, so what is thrown looks as before.

Since cells break out and settle whole, crumbs of less than a cell no longer
arise. The 218 such crumbs in the save, left by the earlier build, gather over a
3.2 m patch into the largest: 241 saved clumps became 46 within ten seconds.

## Follow-up: loose ground, repose and bulking

Spoil built thin vertical walls around the hole, and some clods never settled.
The search for where to lay spoil looked only 4 cells down and 2 sideways, so a
clod on a wall found only the wall top. A clod caught between a dirt bank and a
steel block touched no ground from below, so it never counted as resting, and
gravity kept adding to a speed it could not use.

Cells now carry a looseness byte (brick format v4; v3 is rejected). Spoil is
laid loose, runs downhill to its repose as it is laid, and slides when it is
later undercut; undisturbed ground still stands at any angle. `material-clumps
--pour 1000` lays 1,000 cells of soil as 1,333 with no step between neighbouring
columns higher than one cell and no quantum unaccounted for. Settling counts any
terrain contact, and a clump falls no faster than it actually fell, so a wedged
clod stops and is laid from the bottom of the gap up.

A conservation bug surfaced on the way. A cell pressed flat stays its column's
exposed cell while its density runs on down to empty, and each further press
reported it pressed out again: up to ten cells of spoil from one cell of ground.
The balance check could not see it because it summed what the edits reported.
It now counts the ground itself, brick by brick, before and after. On the drill
replay (fresh ground, 3,600 ticks) the ground lost 1,871,370 quanta, the clumps
hold 1,871,370 more, no soft clump lay unsettled for 3 s, and no tick degraded;
solver p50 3.7 ms, spoil 0.2 – 0.65 ms a tick. In the app the transfer costs
0.09 ms median, 0.93 ms p95.

Spoil pressed out under a stalled head was also being laid straight back under
it. Pressed-out spoil now has to come to rest like any clod, and
`SpoilMachine::keeps_clear` keeps ground from being laid inside a machine part,
within 20 cm beneath one, or within 15 cm of a moving one.

## Follow-up: clods hopping beside the hole

Clods at the brim of the bore hopped a few centimetres up and down and never
settled. A 150 s drill replay with a tracker found 166 such clods away from the
machine, some hopping for over a minute. Three causes:

- Spoil that wanted to run into the bore, where the working head refuses it, was
  laid where it stood instead, and built one-cell towers up the bore. A clod on
  that comb sinks between the towers, is thrown up 5 cm and falls back. Spoil
  that would run on but for a machine is now not laid at all: it stays a clod
  until it can. Loose ground already lying like that, or held up only by a
  machine, is remembered and looked at again, a few columns per transfer, so the
  brim slides into the hole once the machine is gone.
- A clod held up only by other clods kept gathering falling speed, to 1 m/s,
  until it plunged through them and was thrown back. After clods are pushed
  apart their speed along gravity is bounded by how far they really moved, and
  is never turned about.
- What the head stirred at the bottom of the bore shivered up through the whole
  pile. A clod lying on other spoil within its friction cone, that moved less
  than a centimetre, now stays where it was, as one on the ground already did.

A clod buried by spoil laid over it comes up on top when it is laid, instead of
being refused. Sliding now takes a cell only where laying it again ends lower,
so the two agree, and an empty slide no longer reports an edit every transfer.
In the same replay hopping now ends within seconds of the drill stalling; while
digging, the pile in the bore still shifts as it is fed from below. 3,600 ticks:
no quantum unaccounted for, no clump unsettled for 3 s, no degraded tick, spoil
0.65 ms a tick with 303 clods, car traction unchanged.

## After

Same capture, same save:

| Measured | Before | After |
| --- | ---: | ---: |
| CPU physics tick, median / p95 | 43.8 / 62.9 ms | 0.22 / 2.57 ms |
| Spoil step per frame, median / p95 | — | 0.12 / 0.43 ms |
| Ticks run in 60 s | 768 | 3,394 |
| Dropped or degraded ticks | 181 dropped | 0 |
| Material transfers in 60 s | 36, each a 358 ms pause | 222, 0.24 ms median, no pause |
| Cells dug / spoil clumps laid down | 0 / 53 | 970 / 731 |
| Frame p95 | 379 ms | 50.7 ms |
| Longest frame after the first 5 s | 663 ms | 167 ms |
| Streaming backlog at 60 s | ~1,200 | 37 |

The remaining frame time is not digging. `render_acquire` waits a median of
30 ms for the GPU, which spends 38 ms a frame on 7 million terrain triangles at
4112 × 2524 with MSAA × 4; the figure is the same with nothing moving. The
longest frames after world entry are all such waits in an unfocused window.
Terrain rendering cost is a separate item.

Headless replay of the same save (`cpu-physics --scenario world-drive --soil
--tool`, 2,400 ticks): solver p95 2.8 ms, spoil 0.1 ms a tick, 22 – 35 cells a
second dug with the electric-only bearing, settling keeping pace, live clumps
between 50 and 85, no degraded ticks.

`material-clumps` now pours clumps of every material onto generated ground,
all kept awake:

| Awake clumps | Tick p95 | Previously, as solver bodies |
| ---: | ---: | ---: |
| 256 | 0.31 ms | 104 – 114 ms (i5-12600K) |
| 2,048 in one dense heap | 2.9 ms | — |
| 4,096 in one dense heap | 8.2 ms | — |

## Limits

- Spoil collides as spheres. It heaps, but it does not stack like boxes, and a
  long fragment rolls like a ball.
- The machine feels spoil one tick late and as one impulse per body.
- Settled spoil spreads at most two cells, so heaps are steeper than a real
  angle of repose would leave them.
- Settled spoil is as dense as the ground it came from. Looser spoil that takes
  more room than the hole it left needs a looseness value per cell, and so a new
  terrain brick format.
- One tick in one 3,000-tick replay of the deepened bore hit the machine
  solver's speed limit; the next replay had none.
- Edit-to-mesh latency is whatever the streaming pipeline gives an edited node;
  edited nodes already go first. It was not measured separately.
- The clod look and the focused-window frame rate have not been judged by eye.

Raw results: [app before](app-before.jsonl.gz), [app after](app-after.jsonl.gz),
[drill replay](drill-replay-after.jsonl.gz),
[drill replay with the material balance](drill-replay-conserved.jsonl.gz),
[drill replay on loose ground](drill-replay-loose-ground.jsonl.gz). Every record keeps
`kernel_coverage_complete: false`. No scale gate is claimed.
