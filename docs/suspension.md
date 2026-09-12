# Independent suspension

Spring and Shock are separate items in the Matter Manipulator picker. Place one
on a flat construction face and attach construction to the opposite plate.
Inserting the other component reuses the mounts and matches the extended length.
Only the inserted dimensions adapt to the host; existing inputs remain unchanged.
Hold left-click on a mounting face and move the mouse to adjust length. While
holding, press R to cycle length, OD, ID and coil count for springs, or length
and OD for shocks. Release to place; right-click cancels. Dimensions step by
2.5 mm. Insertion fixes length to the shared mounts and rejects incompatible
diameter edits. The on-screen feedback shows the active dimension and values.

Both mount orientations are rigid. External bearings supply articulation.

With Connector equipped, aim at suspension to reveal world controls. Aim at a
Spring, Shock or Bump Stop selector to reach enclosed components. The assembly
stays selected while aiming at its controls; another assembly takes selection.
Interact selects the aimed suspension with Connector, or hints to equip it.
Escape or switching tools dismisses the controls.

Hold left-click on a control and move the mouse; release commits one undoable
adjustment. The cursor stays captured and camera look pauses for the gesture.
Right-click or Escape cancels and consumes the gesture. Dimensions follow their
projected axis in 2.5 mm steps; end-on dimensions, whole coils and 0.1× damping
multipliers use horizontal dragging. Compression and rebound span 0–100×.
Spring preload is additional natural spring length, not a moving mounting plate.
Shock body reversal is a click-and-release control. All 13 component inputs
remain independent. Attached mount-spacing edits report “Release the opposite
attachment to change mount spacing”; compatible spacing-preserving edits remain
available. Invalid drafts retain the last valid geometry in red and cannot commit.

The overlay distinguishes authored extended lengths from simulated separation,
and reports spring rate, shock stroke, remaining travel and the limiting
component, with initial, current, bump-contact and compression-limit markers.
Local drafts never change the running simulation. Release uses the validated
live-edit path; bearing identity, undo/redo and serialization remain unchanged.
Camera motion updates projections and poses; damping edits reuse mesh topology.

Rubber in the pipe tool targets a shock as a bump stop. Hold left-click, move to
size, press R to cycle free length and OD, then release. Bore and orientation
follow the shaft. Initial stop dimensions are 50 × 60 mm. Ordinary pipes retain
their existing behavior outside this target. Invalid stop fits cannot fall through
to pipe placement.

Triangle picking distinguishes wire, shock hardware and rubber. Deleting wire
leaves a shock; deleting a shock removes its stop. Shared plate dimensions and
installed mount spacing persist until the last component is removed. Pipette,
Chroma, undo/redo, creation transforms and format 17 serialization preserve the
independent components. The current format replaces version 16 directly.

## Dimensions and physical laws

The supplied `suspension.zip` guide supplies the spring profiles, finishes and
reference values. Spring steel uses shear modulus 79.3 GPa. For wire diameter
`d = (OD − ID)/2`, mean coil diameter `D = (OD + ID)/2`, and `n` active turns,
stiffness is `G d⁴ / (8 D³ n)` in N/m. Two dead end turns contribute to solid
height `(n + 2)d`. The core enforces the guide's dimensional ranges, integer coil
counts, 2.5 mm input grid, 50 mm minimum travel, and 25% minimum closed/natural
length ratio. Preload adds natural length beyond maximum assembly extension.

Damping is `22000 × π/4 × (body_OD² − shaft_OD²)` times independent multipliers:
1× compression and 1.6× rebound by default, each adjustable from 0 to 100.
Starting compression sets a build pose; a stationary shock adds no force.

Rubber uses modulus 5.5 MPa and a 55% crush limit. With free length `h`, bearing
area `A`, and `k₀ = E A / h`, its tangent stiffness, force and energy are:

```
k(x) = k₀ (1 + 12.8 (x/h)²)
F(x) = k₀ (x + (12.8/3) x³/h²)
U(x) = k₀ (x²/2 + (12.8/12) x⁴/h²)
```

Contact comes from the shaft-side plate's actual distance to the gland shoulder,
independently of spring presence. The first spring-solid, shock-bottoming or
rubber-crush limit bounds compression. Mass and axisymmetric inertia use the
guide densities and fill fractions. Wire mass splits between mounts; the chosen
body end owns the shock body, and the other end owns shaft and stop. Shared
plates count once. Socket-aware compilation includes unattached hardware in the
source body's mass, centre of mass and inertia.

## Game packaging constants

These rules define the game's single-stage construction, not physical laws:

- Shaft diameter: `max(8 mm, 0.34 × body_OD)`.
- Internal hardware allowance: `max(20 mm, 0.5 × body_OD)`.
- Minimum exposed shaft: `max(10 mm, 0.6 × shaft_diameter)`.
- The guide's gland extends `0.2 × body_OD` beyond the rigid body cylinder.
- After subtracting both shared plates, gland, minimum exposure and internal
  allowance, half the remainder is the largest contained stroke. Round down to
  2.5 mm and reject less than 50 mm. The shaft retains piston engagement within
  the hardware allowance throughout this stroke.

The adjuster collar is 1.1× body OD and participates in the 2 mm radial clearance
check. Stop rib crowns and the shaft clamp also participate. Plate thickness and
lattice diameter follow the guide, sized from the installed hardware envelope.
The body never changes length during animation.

## Solver and rendering

Suspension is one passive translational bearing with additive component forces.
It cannot accept a Controller drive. GPU bearing rows grow from 80 to 112 bytes;
all five matching WGSL declarations change together. Spring/damper/rubber
parameters are separate from powered drive budgets. The reserved constraint
state lane holds the accumulated passive impulse and resets every tick.

Both solver routes use a backward-Euler compliant impulse solve, including
predicted rubber contact within the current tick. Hard stops remain unilateral.
Tree kinematics, loop closures, floating reactions and held-joint flags reuse the
existing axial constraint machinery.

Renderer-independent meshes retain topology and UVs while updating compression
vertices and normals. Materials reuse aluminium, steel, rubber and the Chroma
shader. Guide finish colors are normalized against texture color profiles rather
than multiplied into already-dark steel twice. No shock atlas is introduced.

## Verification

Focused regressions cover guide values and invalid inputs, contained shaft travel,
component insertion/removal, retained mount spacing, mass ownership, attachment
validation, stable live-edit IDs, persistence/transforms, editor targeting and
history, Mosaic controls, mesh winding, deformation and live preview frames.
Native GPU scenarios cover equilibrium, oscillation, asymmetric damping, preload,
starting compression, rubber equilibrium/crush, bottoming, mass reversal,
floating reactions, hold/release and mixed loops in both solver routes.

Reproduction commands (GPU tests, native capture and smoke require a Metal adapter):

```sh
cargo test --workspace --offline
cargo test -p mechanic-app -p mechanic-world --offline
cargo test -p mechanic-app suspension --offline
cargo clippy --workspace --all-targets --offline -- -D warnings
cargo fmt --all -- --check
cargo run -p mechanic-bench --offline -- --scenario smoke
BEVY_ASSET_ROOT="$PWD/crates/mechanic-app" MECHANIC_SUSPENSION_CAPTURE_DIR="$PWD/target/suspension-verification/world-controls" cargo run -p mechanic-app --offline
```

## World-control verification — 7 September 2026

The final app suite passed 725 tests, with five existing ignored tests. Its 27
suspension-filtered tests cover all 13 parameters, stepping, local previews,
invalid release, cancellation, single-entry undo/redo, socket reordering/removal,
independent component inputs, attached spacing, component selectors, rubber
hold/R/release, actual Mosaic hit bounds, captured cursor and restored camera
look, transformed showcase save/load, and cached material/geometry reuse.
The world suite passed 88 tests. Strict workspace Clippy, formatting and
whitespace checks passed.

A concurrent app-test/native-capture run hit the existing 4,096-block mesh
publication timing assertion at 6.88 ms. The complete app suite passed after
capture exited; the contention-run log is retained as
`app-tests-concurrent-capture.log` beside the final results.

The native workspace run on **Apple M1 Pro / Metal** passed core (242), Mosaic
integration (18), benchmark unit tests (4), and 82 GPU tests, including every
suspension GPU scenario. The workspace suite remains red on the ten pre-existing
GPU collision, steering and pendulum failures. All ten assertion messages and
numeric values match the retained [baseline comparison](../target/suspension-verification/suspension-baseline-comparison.md)
for `fa825fe6e73b3abb4b0665cb1e514cc159dae1be`. This controls change does not modify
physics, WGSL, GPU ABI or the creation format.

Native smoke passed on the same adapter: 1,024 bodies, 60 ticks, zero error flags,
complete kernel coverage, 7.139 ms p95 engine tick, 4.596 ms p95 GPU tick, and
165.58 physics ticks/s. Correctness and smoke budget checks passed. These checks
do not establish a scale gate.

The native capture harness produced 26 screenshots under
[target/suspension-verification/world-controls](../target/suspension-verification/world-controls).
The first 17 use prescribed static separations, covering standalone springs,
standalone shocks, and combined assemblies at extension, intermediate compression,
bump contact and bottom-out, in both shock orientations. These are geometry
evidence, not simulated equilibrium measurements. The remaining captures show
spring/shock/rubber selectors and handles, reversed hardware, invalid fit,
locked mount spacing, and the showcase editing scenarios.

- [Spring controls](../target/suspension-verification/world-controls/17-spring-controls.png)
- [Shock controls](../target/suspension-verification/world-controls/18-shock-controls.png)
- [Enclosed bump-stop controls](../target/suspension-verification/world-controls/19-rubber-controls.png)
- [Reversed shock controls](../target/suspension-verification/world-controls/20-reversed-controls.png)
- [Invalid fit](../target/suspension-verification/world-controls/21-invalid-fit.png)
- [Locked spacing](../target/suspension-verification/world-controls/25-locked-spacing.png)

Native frame measurements on the saved six-station showcase, after one second
of warmup, used the same capture adapter. Physics was stopped; these are editor
render/input frame timings. Ghost movement and damping use the real preview and
transaction handlers. Camera movement changes the native camera pose.

| Scenario | Frames | Mean ms | p95 ms | Mesh topology generations |
| --- | ---: | ---: | ---: | ---: |
| Camera movement | 164 | 12.25 | 17.73 | 0 |
| Ghost movement | 161 | 12.49 | 15.93 | 0 |
| Damping drag | 161 | 12.47 | 17.72 | 0 |

Both render and picking topology generations are counted. Per-frame pose,
projection and spring deformation updates do not count as topology generation;
switching preview materials preserves mesh handles. JSON measurements and all
verification logs are retained with the screenshots. The capture harness changes
only in-memory fixtures and does not modify saved worlds or creations.
