# Linear bearings

Linear Bearing is available in the placeable picker. Place the rail underside on an overlapping flat construction face, then attach blocks or cylinders to the carriage top or either side. One carriage face can be occupied; direct attachments share one moving assembly. Rails may overhang their support. Unattached carriages remain centred.

Arrow Left/Right adjusts length by 250 mm; Shift adjusts width by 25 mm. Ctrl reduces either step to 2.5 mm. Rotate selects four travel directions. These actions are rebindable. The preview reports dimensions, usable travel and direction.

## Implemented

- `LinearBearingDimensions` validates 0.25–8 m length and 0.05–0.40 m width on the 2.5 mm lattice. Default rail dimensions are 1 m × 100 mm; snapped carriage width is 130 mm. Physical travel is length minus 150 mm.
- `BearingKind` distinguishes rotational joints from linear rail frames. Linear joints select one top/side carriage face, reject a conflicting occupied face, and compile physical bounds separately from programming.
- `linear_bearing_meshes` ports the archive's profiles, chamfers, stops, and fixings into renderer-independent Rust mesh chunks. Eight finish definitions and separate rail/carriage ownership are exposed for renderer integration.
- GPU bearing rows encode the kind in `local_axis_a.w` and physical bounds in `local_anchor_a.w`/`local_anchor_b.w`. Coordinate state uses position and velocity, with units determined by the bearing kind. Buffer sizes for bearing and coordinate rows remain unchanged. Collision world-inertia scratch rows now also carry position, retaining the eight-storage-buffer limit for contact projection.
- Linear tree kinematics, five bilateral velocity constraints, predictive unilateral stops, drive forces, floating-mount reactions, and mixed-joint closure Jacobians are implemented. Translation closure corrections use full affine steps; rotational corrections retain damping.
- Linear position/speed targets and linear drive limits use metres, m/s, and newtons. Actuator sharing and gearing precede conversion using 0.25 m per output revolution. Motor and Servo target compatibility is validated. Controller lane models and editing support SI units and straight travel indicators.
- Creation format 15 persists kinds, rail frames, sockets, and linear programs. Cardinal creation transforms preserve rail frames. The committed creation fixture is updated directly; older formats are not read.

## Editor integration

The procedural geometry uses existing aluminium and steel textures. Rails follow the mounting compound and carriages follow the attached compound. Placement ghosts, occupied-face highlighting, raycasting, pipette sampling, deletion, undo/redo and load framing include linear rails. Placement bounds contain the full rail/carriage travel envelope. Support deletion can migrate a rail to an overlapping coplanar surviving face while preserving its axis and occupied attachment face.

Blocks and pipe runs use the 25 mm attachment lattice with 2.5 mm construction poses. Pipe bend corners preserve that fine lattice. Nested rails mount through attached construction parts.

Controller lanes show signed positions relative to centre, speed in m/s, and force in newtons. Programmable travel can narrow the physical range. Engines control speed and Servos seek/hold position, with existing power allocation, gearing and state sequencing.

## Verification

Focused GPU regressions run on Apple M1 Pro, Metal. They cover passive travel to both stops, off-centre impacts, floating-base reaction, sustained drive and immediate reversal, power removal, mixed chains, and loops whose closure rail has tighter stops. Mixed fixtures exercise both small and parallel routes (66 bearings), using the existing 10 µm translation and 0.001° orientation tolerances and zero diagnostic flags. These tests do not constitute a scale-gate claim.

The full GPU suite has existing failures on this machine: a clean checkout of the starting revision reproduced all twelve failures seen with this change (plus one additional baseline failure). Full-workspace success is therefore not claimed. Focused linear diagnostics pass without failure flags. Formatting and strict workspace Clippy are also checked.

### Reproduction

```sh
cargo test --workspace -- --test-threads=1
cargo test -p mechanic-core -p mechanic-world -- --test-threads=1
cargo test -p mechanic-gpu device::tests::linear_ -- --test-threads=1
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Native screenshot verification used the real Bevy renderer and Apple M1 Pro/Metal adapter. Minimum/default/maximum geometry, carriage highlighting and Controller SI units were inspected. A temporary capture harness ran actual GPU snapshots to the negative stop, centre and positive stop: ±0.425 m at the stops, centre within 0.5 mm, with zero transverse residual, orientation residual and failure flags. Final captures emitted no renderer errors. Temporary harness hooks were removed from production code.

Local verification artifacts (screenshots, snapshot diagnostics and logs) are in `target/linear-bearing-verification/`. These generated artifacts are intentionally outside tracked source.

Final results: 574 app tests passed (3 ignored), 192 core tests passed, 82 world tests passed, 18 Mosaic integration tests passed and 4 benchmark tests passed. The final GPU suite passed 64 tests and failed 12; all twelve failures were also present at starting revision `a57006a` (that baseline failed 13). The eight focused linear GPU tests passed in both the focused and complete GPU runs. Strict workspace Clippy and formatting passed.
