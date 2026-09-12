# Initial impact activation and sustained supported car

Verified CPU experiment. Initial actual contacts activate; new impacts encountered
during an interval still hold. The public runtime remains joint-only and the app
uses the original GPU runtime. No complete contact/driving or performance gate passes.

Closing initial contacts now receive one instantaneous coupled mass-factor solve,
including active hard stops. Finite actuator and passive forces have zero impulse
at zero elapsed time. The following finite-duration substep rebuilds sustaining
contact targets from outgoing velocity, so restitution is not applied twice.
Impacts move no positions and contain no penetration velocity bias. Failed solves
preserve the physical state. Each attempted/accepted activation, row, factor solve
and solver sweep is recorded separately from finite-time force work.

Regressions verify restitution once, unchanged positions and penetration-independent
bounce, failed-solve rollback, joint-stop reaction back-driving a floating root,
and outgoing-velocity advancement at 1/2/4/8 subdivisions. Existing support,
terrain-edit rollback and new-impact holds remain covered.

The exact authored car fixture now runs inside the private terrain-tick experiment
on a finite 128 m square Rock floor, starting with 1 mm overlap. Its authored idle
drives, suspension, gravity and contacts remain active. It completes 120 external
ticks (two simulated seconds) at each fixed policy, repeated twice:

| Substeps | Final state hash | Maximum vertex penetration (m) |
| --- | --- | ---: |
| 1 | `2e1dacca83ccbe43` | 0.0010000000143751686 |
| 2 | `3ee19385b3ca51e7` | 0.0010000000119606 |
| 4 | `871a710e76296f1c` | 0.0009999999881755706 |
| 8 | `70bc10a59fd9b87a` | 0.0009999999944617644 |

Every tick hash matches its repeat at the same policy. An independent flat-floor
vertex bound checks penetration and verifies each collider remains within the
finite floor footprint. These are support trajectories under authored idle drives,
not driving, streamed terrain, a cold drop or a performance measurement. The
stricter 1e-9 cold fixed-pose response failure remains separately retained.

Current CPU packages pass: core 260, physics 81, world 91, saved-car bench 7 and
fixture 5, with no ignored physics tests. Workspace Clippy, formatting and whitespace
checks pass. The preceding serial Apple M1 Pro / Metal workspace run still has the
same 11 GPU failures and one UI failure; its release app build passes. Subsequent
changes do not alter GPU runtime or app dependencies. The workspace is not green.

`identity.json` records commands, sources, fixture and log hashes and the overlay
against the frozen replay reference. RON is an existing workspace dependency added
only for loading the exact authored fixture in CPU tests. Continue with
[first-impact event integration](../../contact-impact-next.md), preserving all
[master-plan gates](../../compiled-machine-dynamics.md).
