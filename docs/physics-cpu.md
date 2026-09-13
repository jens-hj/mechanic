# CPU physics

`mechanic-physics` has two CPU solvers over the same
[compiled machine model](compiled-machine-dynamics.md):

| Solver | Role | Guarantees |
| --- | --- | --- |
| Soft step (`CpuMachine`) | Runs the app's CPU route | Publishes every tick; quality is measured, not gated |
| Exact reference (`reference::CpuJointMachine`) | Offline validation | Publishes only when the original contact laws hold; otherwise returns an error |

Mechanic is an experimental game. Physics must stay live and plausible, so the
app never waits on a solver that can refuse a tick. The exact solver stays
because it has found real geometry bugs: faceted wheels, duplicated cylinder
corners, and vertex activation. It works as a checker, not a runtime.

## Selecting the route

`MECHANIC_PHYSICS=cpu cargo run -p mechanic-app` steps ticks on the CPU. The GPU
scene stays resident and keeps owning terrain preparation, drive resolution and
every buffer the renderer reads; only the tick changes. Anything other than
`cpu` selects the GPU runtime.

A creation the CPU cannot run (closed mechanism loops, for now) is refused with
a message. An invalid tick input hands the last published state to the GPU
runtime, which keeps simulating until the next construction publication.

## Soft-step solver

Status: being introduced. This section is filled in with the implementation.

## Exact reference solver

`CpuJointMachine` integrates with implicit midpoint. It uses:

- nonlinear force iteration with drives, suspension and stops
- event search that splits substeps at each contact arrival or release
- continuous path certificates and split position recovery
- a smoothed-continuation Newton contact solve checked against the original
  unsmoothed contact laws

A tick that cannot meet those laws is retried at 2, 4 and 8 substeps, then
rejected without changing state. The saved car still meets unsolved states
within the first seconds of a cold drop. That is expected for this solver and
is not a bug to chase.

`MECHANIC_IMPACT_CAPTURE=<path>` and `MECHANIC_FORCE_CAPTURE=<path>` write the
exact algebra of a failing solve from test builds. Captured solves live in
`crates/mechanic-bench/fixtures/cpu-reference/`.

## Quality bench

```sh
cargo run -p mechanic-bench --release --bin cpu-physics -- --scenario reference-fixtures
```

This prints one JSONL record per captured solve: rows, convergence, residual,
iterations, time and worst contact-law violation. It always exits successfully;
use it to see whether a reference-solver change helps or hurts.

## Testing approach

- `cargo test` asserts behaviour with game tolerances: bodies settle,
  restitution bounces, drives move the car, and identical inputs repeat exactly.
- Solver internals (iteration counts, storage bounds, captured tick numbers)
  belong in the bench, not in tests.
