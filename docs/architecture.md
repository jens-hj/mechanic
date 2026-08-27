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

Welds may close a loop through a bearing. A bearing whose two sides land in one
compound is locked by construction: the weld already fixes their relative pose,
so the bearing compiles to no joint, no coordinate, and no constraint rather
than failing the build. Only a *driven* collapsed bearing is an error
(`TopologyError::SelfBearing`), because silently dropping that joint would leave
a control block wired to nothing.

A sequencer in the app advances each wire's program while the simulation ticks.
Time is counted in dispatched physics ticks rather than frames, so pausing
freezes every dwell and a slow frame never skips one.

Drive programs are the single exception to build-only mutation: their resolved
rows may be written to the GPU while `Running`, because they change no topology,
no mass, and no buffer size, leaving every compiled row index valid. Everything
else still requires `Building`.

Engines of the same kind in one Controller's directly welded machine module are
one physical engine line. Adding engines scales that line's stall torque and
bearing capacity linearly; it does not scale no-load RPM. Every engine in the
line therefore shares one gearbox and must carry the same transmission depth.
A mismatched line is a valid incomplete build which can be edited, saved, and
loaded, but compilation refuses to simulate it until the physical stacks match.

A transmission is a fixed 2×2×1 machine part with a graph-owned parent relation.
It can only continue an engine's local positive-Z output or the current chain
tail, inherits the root engine orientation and appearance, and owns a required
weld which cannot be removed separately. Removing an engine or upstream
transmission cascades through the downstream chain. Persistent gearbox settings
belong to `(Controller, EngineKind)` graph records rather than to the Controller
part. Active gears and pending direction changes are transient simulation state.
Ideal gearing is resolved when drive rows are uploaded: each engine family's
stall torque is multiplied by its active input-to-output ratio and its no-load
output speed is divided by the same ratio. Gas and electric contributions use
independent ratios, and this changes no GPU ABI or topology row.

The compiled simulation is data-oriented. Parts become collider rows, compounds
become body rows, and tree bearings become one-coordinate mechanism rows. Each
bearing component has deterministic fixed or floating roots, parent/direction
rows, root-before-child and child-before-root traversals, and leaf contraction
rounds. An edge that would join two already-grounded trees is a closure rather
than silently making one grounded body dynamic. There is no entity, transform,
mesh, or material per part.

An editable area is a *shape region*: a solid cuboid of blocks claimed for
editing. A region owns the geometry of the blocks it covers, so those blocks stop
emitting colliders, mass, and mesh of their own and the region emits one merged
shape instead. Claiming an area requires every cell filled, one material
throughout, one rigid body, and no overlap with an existing region; deleting any
of its blocks deletes the region, the way deleting a part already cascades its
welds.

A region's geometry is its *control cage*, a grid of vertices. A fresh region has
two planes per axis — eight corners, one hexahedron. Subdividing inserts a whole
plane, never a lone vertex, so the cage stays a valid grid of hexahedra and the
decomposition keeps holding. Every vertex is clamped to the region's original
bounding box, so a corner can only be drawn inward and one region can never grow
into its neighbours.

One decomposition turns a grid of cells into convex pieces, and the compiler, the
render mesh, and the editor raycast all consume it. Because they share it, the
collider, the visible surface, and the cursor cannot drift apart. Cells with no
displaced corner are covered by as few boxes as possible, so an unshaped part
still compiles to the single box it always did and the box-versus-box solver path
is untouched. Shaped cells are split by the Freudenthal scheme, whose corner
labelling makes two neighbouring cells triangulate their shared face identically,
and the resulting tetrahedra are then fused back into the largest convex pieces
that reproduce their union exactly. Fusing is refused where it would
re-triangulate a non-planar grid face, because a convex hull would pick the
opposite diagonal from the neighbouring cell and split the surface open.

Two cage vertices driven through each other would turn a cell inside out, and
`SetRegionVertices` rejects that rather than letting self-intersecting geometry
reach the solver.

Parts may only be placed on faces that are still flat: every cage vertex on that
face resting on the grid. A shaped face is no longer an axis-aligned rectangle,
so nothing could sit flush on it.

## Persistence

A saved creation is the authored graph and nothing derived from it: parts,
welds, rigid links, bearings, drive wires with their limits and programs,
transmission parent references, per-controller gearbox settings, the shape
regions with their cage planes and displaced vertices, and the bearing rings the
editor holds that no part hangs from yet. That set is the
same one the undo history snapshots, which is the definition of "the whole
creation". Compiled bodies, mass and inertia, loop topology, GPU buffers, and
sequencer cursors are all recomputed on load.

Graph handles are generational and minted privately, so they are not stable
across a rebuild and cannot appear in a file. A creation document numbers its
rows by position instead, and loading replays them as `BuildCommand`s through
`apply_batch`, remapping dense indices onto the handles the arenas return.
Every value passes through the same validating constructors the editor uses,
and the rebuilt graph is compiled before it replaces the current one. The sole
recognized exception is a same-type transmission-depth mismatch: it installs as
an explicitly incomplete build so a player can finish matching the stacks. All
other compilation errors still prevent the candidate from replacing the current
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
6. Warm start and solve compliant normal impulses, persistent two-axis
   static/kinetic friction, and tangent-axis rolling resistance for a fixed
   projected budget, then project their body-space changes back into root and
   joint velocities. Surface friction and rolling coefficients mix by geometric
   mean, restitution by maximum, and nominal block compliance additively. The
   normal constraint uses `gamma = combined_compliance / dt²` in both its
   effective-mass denominator and accumulated-impulse feedback. The infinite
   ground plane uses Concrete's surface properties.
7. Validate finite state and the 0.01 mm/0.001 degree bearing tolerances.
8. Publish the next snapshot and timestamp report.

Scenes without bearings retain the direct free-body integration/contact path.
Mechanism timing covers reduced-coordinate projection and forward kinematics;
contact timing includes contact projection before articulated feedback.

## Contact ABI

`GpuContact` remains 64 bytes and the 2,097,152-row pair/contact capacity is
unchanged. Narrowphase temporarily carries combined compliance in the future
normal-impulse lane. Preparation then stores target normal speed in the former
penetration lane and packs static friction, kinetic friction, rolling
resistance, and compliance response into the spare lane; an analytic-cylinder
bit lives in contact metadata.

`GpuCollider` grows from 96 to 112 bytes to hold explicit surface-response and
elasticity vectors, adding 2 MiB at the 131,072-collider cap. A 32-byte uniform
holds the ground surface. `GpuPersistentManifold` grows from 48 to 64 bytes so
normal, two-axis sliding, and two-axis rolling impulses survive between ticks;
that adds 32 MiB at maximum pair capacity. These are fixed allocations and do
not alter overflow behavior or adapt solver quality at runtime.

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
