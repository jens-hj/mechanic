# Live construction editing and dimension-link freezing

Construction frames, moving tool contexts, arbitrary-angle welding, and active-link
hold controls are implemented. Verification is not fully green: eleven GPU
regressions remain, and the interactive vehicle walkthrough is pending.

## Behavior

Hammer + F toggles the active dimension link without aiming. Freezing restores the
structural creation's authored joint defaults, levels it to the nearest cardinal
heading around the link pivot, and rounds height upward to the 25 cm grid. Terrain,
surrounding construction, and nonadjacent held parts are checked along the path;
a clearance lift is attempted within 20 m. Unsafe paths are refused.

Up/down select 25 cm height steps immediately, repeating after 300 ms and then
every 100 ms. Opposite keys cancel repetition. Motion uses a 100 ms exponential
position interpolation and quaternion slerp with an exact final settle. Releasing
during alignment waits for alignment to finish. Tool changes retain the hold;
changing the active link releases the previous creation. Bindings are rebindable
and obey UI, focus, and contextual shortcut routing.

Lowering clips the final step to 5 cm of terrain clearance, even between grid
heights. Terrain clearance uses collider-versus-triangle separation against the
active collision mesh, with padded bounds pruning the terrain triangle hierarchy.
It no longer performs dense face sampling and repeated nearest-terrain searches.
This permits wide, thin platforms to approach the ground and checks small terrain
features directly. Surrounding construction still blocks obstructed steps.

## Implementation

- Graph-owned translation/quaternion frames compose each part's local grid pose.
  Geometry, colliders, inertia, picking, rendering, regions, and saved creations
  use composed poses. Creation format 16 requires frame and membership arrays.
- A stable part anchors each tool gesture. Rays enter its local grid; previews
  return to its current world pose. Editor history stores canonical coordinates.
  New/replacement parts retain source identity through publication and splits.
  Deleted anchors cancel attached gestures; delete selections use composed centers
  and stay on their starting body. No-op views preserve graph change detection.
- Feature welds move the first structural creation in its authored default pose
  onto the second creation's current pose. Picking, constrained dragging, and
  ghost rendering live in `weld_tool`; core topology, alignment, contact-square,
  and default-construction validation live in `mechanic_core::weld`.
  See [feature weld placement](feature-weld-placement.md) for transaction semantics.
- GPU readbacks capture transforms, velocities, and rotational/linear coordinates
  from one completed tick. Publication drains old submissions, rejects stale
  generations, remaps stable identities, and retains surviving joint state,
  unchanged drive-program progress, and gearbox state.
- Held components have zero inverse mass/inertia and suspended drives, joints,
  reconstruction, and solver corrections. Prescribed poses remain collidable.
  Hold/release discards velocities and invalidates affected contact warm starts.
  CPU rendering, picking, and player collision use the same held poses as the GPU.
- Held bearing programs also pause when their controller is on another creation.
  Unrelated programs continue. Detached fragments resume physics.
- World format 5 stores the frozen link, global target, cardinal heading, and
  matching construction generation. Autosave pairs the target with the last
  validated graph. Rejected edits restore the accepted editor snapshot. Release,
  link deletion/change, and Garage transfer clear the saved hold. Restore installs
  the validated authored pose before simulation starts.

Formats are replaced directly without compatibility readers. GPU bearing flags
reserve bit 1 for suspension; contact-cache stamps include endpoint mass modes.
Existing uncommitted linear-bearing work is retained.

## Verification (2026-09-06)

- `cargo test --workspace --exclude mechanic-gpu`: 961 passed, three ignored.
  After the final tool-review fixes, `cargo test -p mechanic-app` passed all 650
  app tests with three ignored, including four additional regressions.
- `cargo test --workspace -- --test-threads=1` with Metal access: app/core/benchmark
  tests passed, including the GPU-backed app showcase tests. The GPU crate had
  72 passes and 11 failures listed below; Cargo stopped before later crates.
- Moving-tool regressions cover placement, split deletion with undo/redo,
  painting, rotational/linear bearings and wiring, ongoing drags, and shaping.
  Weld tests cover arbitrary-angle contact, default-pose rejection, frame/save
  round trips, stale publication rejection, and merged momentum.
- Freeze regressions cover cardinal/height math, repeat timing, exact settle,
  swept terrain/construction clearance, held self-collision and joint exclusions,
  persisted targets, publication rejection, and floating-origin conversion.
- Real GPU hold tests passed on **Apple M1 Pro / Metal**, including gravity,
  impulses, rotational/linear drives, mixed-joint closure loops, neighboring
  collision, and stable release with contact-cache invalidation.
- Workspace build, formatting, Clippy, and whitespace checks passed.
- `cargo run -p mechanic-bench -- --scenario smoke`: Apple M1 Pro / Metal,
  1,024 bodies, 60 ticks, zero error flags. This is a debug smoke run, not a
  release scale-gate measurement.
- The app opened successfully on Metal. The manual moving-vehicle edit/freeze/
  height/weld/release/reload walkthrough has not been completed.

## Remaining GPU failures

- `articulated_car_drop_settles_without_drift_or_ground_penetration`
- `flat_disc_landing_on_another_disc_does_not_gain_energy`
- `gpu_cylinder_bore_is_passable_and_annular_material_blocks_motion`
- `gpu_pipe_bend_bore_is_passable_and_annular_material_blocks_motion`
- `lower_modulus_allows_more_transient_penetration_without_failure`
- `plastic_rebounds_more_than_concrete`
- `production_servos_hold_steering_under_first_gear_gas_drive`
- `static_friction_holds_a_sub_threshold_load_while_kinetic_friction_slows_sliding`
- `steering_servos_reach_angle_without_overshooting`
- `struck_articulated_wheel_recovers_without_crossing_ground`
- `tall_single_and_double_pendulums_dissipate_energy`

Three invalid fixture assumptions were corrected without relaxing dynamics
assertions: mass-independent impulse thresholds, a grounded disc positioned above
the ground, and pendulum counts that ignored collider compaction. The corrected
impulse test passes analytically; disc and double-pendulum dynamics still fail.
No before-change GPU baseline was captured, so remaining failures are not claimed
to be pre-existing. External freeze clearance uses conservative collider bounds
and can refuse geometrically safe tight passages. General restoration of unfrozen
live physics poses remains outside this change.
