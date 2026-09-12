# Unpublished supported ticks and explicit impact holds

Verified private CPU experiment. The public CPU runtime remains joint-only and
the app remains on the GPU backend. Complete contact ticks and all physics/frame
acceptance gates remain open.

The existing whole-tick transaction now has a private terrain context. It refreshes
finite manifolds at every numerical substep, couples them with drives/stops and
passive forces, then validates the actual midpoint generalized displacement before
accepting motion. Joint position correction has a separate path check and restores
physical velocity even on failure. No independent body interpolation is introduced.

After a whole path passes its depth limit, a new-contact sweep skips only pairs
with an actual initial finite contact. Every newly approached pair still runs CCD.
New impacts currently hold the unpublished tick, including shallow impacts below
the penetration limit. This is a deliberate incomplete milestone, not a completed
impact solver or a successful throughput gate. A floor support cannot hide a new
wall impact. The public sweep retains its original zero-time touch behavior.

Geometry, path and constraint failures feed the same bounded 1/2/4/8 whole-tick
retry loop. Each attempt begins from the unchanged post-command state; completed
snapshots and drive commands commit only after the entire external tick succeeds.
Diagnostics retain contact queries, initial-contact tests, SAT/envelope work,
correction paths, rejected attempts and impact holds.

Eight regressions cover:

- 120 supported ticks, repeated exactly at each 1/2/4/8 substep policy, with no
  support drift and one effective factor per numerical substep.
- A one-envelope work budget that rejects 1/2/4 policies and accepts eight
  subdivisions. The complete snapshot matches a direct eight-substep run and
  applies the external impulse once.
- Final exhaustion preserving state and permitting the same tick-indexed command
  to be retried; failed terrain work also leaves drive changes uncommitted.
- Split joint recovery refusing an unsafe terrain path without leaking scratch
  velocity, and completing in clear geometry with physical velocity preserved.
- Wrong topology and raised supporting terrain refusing invalid publication.
- Shallow first impact and a new finite wall beside an existing floor both
  holding rather than silently passing through the support-only solver.

Current CPU packages pass: core 260, physics 75, world 91, saved-car 7, fixture 5.
Workspace Clippy, formatting and whitespace checks pass. The existing public
saved-car joint-only trajectory still hashes `fcba50d819857c8c`; its full fixed-pose
surface response remains 73 sweeps at 6.401599574328369e-9 residual. No complete
car contact trajectory or performance comparison is implied.

The preceding supported-path checkpoint's serial Apple M1 Pro / Metal workspace
run retains the same 11 GPU failures and one UI failure. Its release app build
passes. Only CPU physics and tests changed afterwards; app dependencies and GPU
runtime remain unchanged. The workspace is not green.

`identity.json` records commands, source/fixture hashes, log hashes, and the overlay
against the frozen replay reference. Next implement actual first-impact activation
and reintegrated event intervals, then generalized terrain recovery, contact cache
transactions, body/body collision, loops and common terrain-tagged publication.
The cold car's stricter 1e-9 solve remains a required failure case. Follow the
[moving-contact plan](../../compiled-contact-tick-next.md).
