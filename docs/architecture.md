# Architecture invariants

## Ownership

The editable construction graph exists only on the CPU and can mutate only in
`Building` state. Compilation is all-or-nothing. Welds are collapsed with
union-find; no weld reaches the solver as a soft constraint.

Control blocks are ordinary parts with no settings of their own. Behaviour lives
on the wire from a block to a bearing: a speed and torque envelope plus an
ordered list of states, one program per bearing. Wires are graph edges, cascaded
on deletion like welds. Drive rows are derived from the graph rather than stored
on bearings, so they can be rebuilt at any time from the same compiled topology.
Compilation also records every graph bearing's coordinate, including rows
collapsed as the same physical joint, so a wire on any of them addresses the
right row.

A sequencer in the app advances each wire's program while the simulation ticks.
Time is counted in dispatched physics ticks rather than frames, so pausing
freezes every dwell and a slow frame never skips one.

Drive programs are the single exception to build-only mutation: their resolved
rows may be written to the GPU while `Running`, because they change no topology,
no mass, and no buffer size, leaving every compiled row index valid. Everything
else still requires `Building`.

The compiled simulation is data-oriented. Parts become collider rows, compounds
become body rows, and tree bearings become one-coordinate mechanism rows. Each
bearing component has deterministic fixed or floating roots, parent/direction
rows, root-before-child and child-before-root traversals, and leaf contraction
rounds. An edge that would join two already-grounded trees is a closure rather
than silently making one grounded body dynamic. There is no entity, transform,
mesh, or material per part.

## Persistence

A saved creation is the authored graph and nothing derived from it: parts,
welds, rigid links, bearings, drive wires with their limits and programs, and
the bearing rings the editor holds that no part hangs from yet. That set is the
same one the undo history snapshots, which is the definition of "the whole
creation". Compiled bodies, mass and inertia, loop topology, GPU buffers, and
sequencer cursors are all recomputed on load.

Graph handles are generational and minted privately, so they are not stable
across a rebuild and cannot appear in a file. A creation document numbers its
rows by position instead, and loading replays them as `BuildCommand`s through
`apply_batch`, remapping dense indices onto the handles the arenas return.
Every value passes through the same validating constructors the editor uses,
and the rebuilt graph is compiled before it replaces the current one, so
neither a corrupt file nor a hand-edited one can install an invalid
construction.

`mechanic-core` owns the document and its conversions and performs no
filesystem access; the app owns where files live, reads and writes them, and
maps the editor's unattached rings to and from the document's sockets.

## Clocks and publication

Physics advances at an explicit 60 Hz fixed step. A tick is published to the
three-slot GPU snapshot ring only after all kernels finish, numeric checks pass,
the contact pair buffer has not overflowed, and joint closure is within 0.01 mm
anchor error and 0.001 degree axis error. Rendering consumes the newest complete
pair of sequence-numbered slots and never waits for bulk CPU readback.

## Kernel order

1. Apply gravity and damping to every dynamic compound; only fixed/floating
   mechanism roots advance as free body poses.
2. Project body velocity changes onto the bearing manifold, advance the
   authoritative root and joint coordinates, and rebuild body poses and twists
   by parallel pointer-jump traversal. A driven coordinate accelerates toward a
   desired speed within a per-coordinate budget derived from its wire's torque
   and the child subtree's inertia about the joint axis, then clamps to its
   travel limits. In angle mode the desired speed follows a trapezoid profile —
   never more than the remaining error can brake off — so the joint settles on
   its target instead of hunting. Because the speed it corrects from is measured
   from real body motion, a weak drive stalls or is back-driven rather than
   overriding the mechanism. Driven bearings are preferred as spanning-tree
   edges so a drive always owns an independent coordinate.
3. Correct loop positions with 12 fixed Newton steps. Each step solves its
   matrix-free normal equation with eight diagonally preconditioned CG rounds.
4. Build Morton codes and an LBVH over the current cuboid world bounds.
5. Generate OBB and ground contacts with fixed-capacity overflow accounting.
6. Warm start and solve normal/friction impulses for a fixed projected budget,
   then project their body-space changes back into root and joint velocities.
7. Validate finite state and the 0.01 mm/0.001 degree bearing tolerances.
8. Publish the next snapshot and timestamp report.

Scenes without bearings retain the direct free-body integration/contact path.
Mechanism timing covers reduced-coordinate projection and forward kinematics;
contact timing includes contact projection before articulated feedback.

The initial WGSL proof keeps these stages as separate entry points even where a
smoke implementation is deliberately minimal. This preserves profiling and
failure boundaries needed before scale-specific kernel work.

## Gate

`dense_100k` is exactly 100,000 active 1 m cuboids in a packed contact-heavy
stack with sleeping disabled. `loops_100k` is one active 100,000-body bearing
lattice with closed loops and collisions disabled. Each runs for 5 seconds of
warm-up plus 30 seconds measured on the reference machine. Both require 60 TPS
and p95 GPU tick cost no greater than 16.67 ms. Integrated rendering additionally
requires p95 engine frame cost no greater than 5 ms at 1920x1080 while physics
continues to pass.

Until those recorded gates pass, `mechanic-app` remains a prototype. It can run
the GPU physics pipeline, but synchronously reads snapshots back to CPU-rebuilt
part/bearing meshes. Its CPU picking, transient ghost meshes, editor HUD, and
snapshot readback do not exercise or stand in for the required GPU-culling and
indirect production render path.
