# Surface substeps and solver cost

Verified implementation checkpoint. Complete CPU scene ticks and every application
performance acceptance gate remain open. The app still uses the GPU runtime.

Finite proximity queries preserve silhouettes, finite edges/holes, true opposing
points, and signed separation. Instantaneous impact solving rejects separated
proximity points. Internal implicit substeps couple real terrain manifolds with
suspension, drives, and stops through the same effective dynamics factor. Tests
cover sustained support, analytic supported spring equilibrium, and static/kinetic
friction at 1/2/4/8 subdivisions. Explicit warm starts apply only within an unchanged
prepared response, project current bounds, and revalidate the physical residual.
Continuous activation, supported trajectory validation, split recovery, contact
cache transactions, body/body collision, loops, and complete publication remain open.

The saved-car finite-floor experiment retains 11 bodies, 16 generalized velocities,
78 colliders, 48 contacts, 12 manifolds, and 240 scalar rows. It warms 100 fixed-pose
samples and measures 1,000; simulated duration is zero. It measures CPU assembly,
factor/free velocity, finite collision/row construction, and preparation/solve
separately. No GPU work, transfers, or state publication occur.

| Standalone paired run | Total CPU p95 (ms) | Residual | Sweeps |
| --- | ---: | ---: | ---: |
| A1 frozen initial solver | 31.481167 | 6.237585e-9 | 172 |
| B1 reduced Newton | 10.070916 | 6.401600e-9 | 73 |
| B2 reduced Newton | 10.025250 | 6.401600e-9 | 73 |
| A2 frozen initial solver | 31.726667 | 6.237585e-9 | 172 |

This is approximately a 3.1× fixed-pose improvement, still far above the intended
1 ms complete car tick. It is not the foreground integrated A/B/B/A acceptance run.
Repeated result hashes match within each executable: A `deb1388ad1af327d`,
B `9e6ad5a0f65be446`. Cross-build hash equality is not claimed.

The first reuse pass cached block spectral bounds, projection derivatives, and
scratch storage without changing bits; its paired p95 was 24.559/24.687 ms. A
maximum-scaled norm then measured 23.064 ms. A separate sampling profile identified
Newton/Krylov work as the dominant cost. The retained candidate eliminates contact
Newton unknowns through local projection blocks and factors a bounded generalized
system for at most 64 coordinates. Larger systems retain bounded matrix-free
Krylov work. Global contact-response storage remains zero above 128 rows.

The retained car solve uses 2,648 H factor applications, 73 generalized Newton
factorizations, 3,504 local factorizations, 46 accepted proposals, and 290 line
search evaluations; peak reduced scratch is 4,096 scalars. Every proposal is
projected before evaluating the original maximum projected velocity residual.
Independent momentum, support, friction/rolling bounds, and dissipative-work checks
pass. The nonlinear substep still refreshes and validates its full force residual.

Rejected controls are retained with sources and logs. Squared-merit acceptance
failed at 7.25e-4 residual. A projected Krylov control passed the default residual
but failed the independent positive-friction-work bound. The tighter 1e-9 cold
solve remains unconverged after 256 sweeps, at approximately 1.99629e-9. The explicit
failure capture exits 1 and emits diagnostics with no publication or p95. The
joint substep uses an inner 1e-9 constraint tolerance, so a complete car contact
tick cannot be claimed from the passing 1e-8 cold experiment.

```sh
cargo build --release -p mechanic-bench --bin compiled-response --offline
# A uses .physics-reference/cpu-surface-before-reuse/compiled-response.
# B uses target/release/compiled-response. Run A/B/B/A serially, no builds/tests.
target/release/compiled-response \
  crates/mechanic-bench/tests/fixtures/driven_car_instance.ron 256 finite-terrain
# Required retained failure; exits nonzero:
target/release/compiled-response \
  crates/mechanic-bench/tests/fixtures/driven_car_instance.ron 256 finite-terrain 1e-9
```

Workspace tests were run serially on Apple M1 Pro / Metal: core 253, physics 62,
world 91, saved-car 6 pass. App has 739 passes and the same one UI failure; GPU has
110 passes and the same 11 physical failures as the prior checkpoint. Full failure
names/output are in `workspace-tests.log`; unavailable hardware is not counted as
passing. The adapter-labelled adopted-terrain test and WGSL validation pass.
Workspace Clippy, formatting, whitespace checks, and the release app build pass.
The workspace is not green.

`identity.json` records commands, source/fixture/executable hashes, results, and the
source overlay archive against `.physics-reference/replay-final-20260910/source`.
Intermediate candidate sources, rejected controls, raw measurements, and separate
profiling evidence are retained here. Continue with the concrete sequence in
[the moving-contact plan](../../compiled-contact-tick-next.md).
