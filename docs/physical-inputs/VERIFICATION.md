# Physical input verification — 2026-09-22

Host: Apple M1 Pro, Metal. No scale or frame-budget claim is made.

## Automated checks

`CARGO_NET_OFFLINE=true cargo xtask ci` completed. Formatting, consistency,
API documentation, and Python script checks passed. Its test and lint phases
were not green. Follow-up results after the final fixes are recorded below.

The full GPU suite reported 102 passed, 10 failed, and one ignored. These ten
physics behaviour failures also appeared in the previous full run:

- `contacts::gpu_cylinder_bore_is_passable_and_annular_material_blocks_motion`
- `contacts::gpu_pipe_bend_bore_is_passable_and_annular_material_blocks_motion`
- `contacts::lower_modulus_allows_more_transient_penetration_without_failure`
- `contacts::static_friction_holds_a_sub_threshold_load_while_kinetic_friction_slows_sliding`
- `contacts::plastic_rebounds_more_than_concrete`
- `pendulums::tall_single_and_double_pendulums_dissipate_energy`
- `vehicles::articulated_car_drop_settles_without_drift_or_ground_penetration`
- `vehicles::articulated_car_wall_impact_remains_bounded_and_decays`
- `vehicles::production_servos_hold_steering_under_first_gear_gas_drive`
- `vehicles::struck_articulated_wheel_recovers_without_crossing_ground`

The CPU physics suite reported 242 passed and one failure:
`soft_step::tests::a_box_dropped_on_a_resting_box_comes_to_rest_on_top_of_it`.
These failures remain unresolved; this change does not claim full CI acceptance.

The follow-up `cargo xtask test -p mechanic-app -p mechanic-core` completed all
895 app tests successfully (eight ignored) and all 398 core tests successfully.
This includes the full-system dial interaction tests and final UI layout and
endpoint-initialization regressions. Because the task runner retained
`--workspace`, it then started an unnecessary repeat of the GPU suite; that
remaining run was explicitly stopped after the app/core suites had completed.

After the final label margins and button callout sizing changes,
`cargo xtask test --bin mechanic-app` passed all 896 app tests (eight ignored).

Focused UI verification also passed all 140 tests before the final refinements.
Final `cargo xtask lint` passed across every workspace target. The final
`cargo xtask consistency` check passed as well.

The standalone Mosaic outline tests passed all three cases:

```
CARGO_NET_OFFLINE=true CARGO_TARGET_DIR=/tmp/mechanic-mosaic-target \
  cargo test --manifest-path vendor/mosaic-text/Cargo.toml --lib outlines
```

Eight focused GPU linear-drive tests passed on Metal. Shared solver-row tests
also cover angular and linear dial updates with an unchanged compiled creation.

## Native inspection

The opt-in native capture harness renders all three input sizes with released
and held key labels, plus the controller assignment editor and the Connector's
button configuration overlay. The geometry showcases use prescribed poses and
untextured finish colors; they verify silhouettes, key geometry and illumination,
not simulated physical operation or construction texture quality.

The first inspection caught stretched dial controls; regression tests now check
compact layout and nonoverlap. A later capture, run alongside the GPU suite,
stopped with a Metal counter-sample-buffer allocation error and device loss.
The isolated retry completed all ten captures successfully on Metal. The final
assignment editor and button callouts were inspected with readable, separated
controls; the key labels were inspected at all three sizes.

- [Dial assignment editor](native/dial-assignment.png)
- [Button Connector configuration](native/button-connector.png)
- [Held key labels at all three sizes](native/button-labels.png)

Actual production interaction systems are exercised by app tests for on-foot
and seated buttons, occlusion, combined keys, and cancellation. Dial tests exercise
horizontal dragging, Shift fine adjustment, mixed selection without writes,
multiple targets, and release/reach/focus/pause cancellation. A manual play-through
on moving constructions has not been completed.
