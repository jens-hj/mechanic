# Retry diagnostics and valid reuse

The application still uses the original GPU solver. These changes affect the
private CPU experiment and its tests; they deliver no claimed application speedup.
No complete physics, rendering, or scale acceptance gate passes.

## Retained changes

- Failed ticks record each subdivision attempt's stage, error and validated but
  unpublished elapsed time. Successful retries retain that history and clear the
  final failure stage. Validation, impulses, finite integration, terrain queries,
  path certificates, recovery, impact and event search are distinguished.
- Event reintegration reuses the initial pose dynamics and finite terrain contacts
  while the initial state stays exactly unchanged. An accepted prefix invalidates
  both. Topology, terrain, materials and origin remain immutably borrowed for that
  interval. Numerical factors, contact velocities and force solves still refresh.
  This is not a cross-tick contact/impulse cache or complete world publication.
- Stalled small contact-only systems may use the existing bounded smoothing
  continuation after the ordinary dense search and a full stalled residual window.
  It spends at most 128 remaining directions inside the original 256 total; every
  original nonsmoothed law still decides convergence. The dense W limit is 128 rows.

The unchanged-state replay reduces contact queries **29 → 10** and dynamics
assemblies **379 → 360**. State, actuator impulses, elapsed time, event trials,
factorizations and constraint iterations match the uncached control exactly.
The test also requires invalidation at each accepted prefix. These are work counts,
not a timing or complete-physics acceptance result.

## Physical evidence

Before continuation, the default drop's tick-17 retries failed in this order:
1 substep: EventSearch; 2: EventSearch; 4: Impact; 8: EventSearch.
A cumulative maximum impact residual had obscured the final rejection phase.

The captured four-substep impact has 50 rows and 16 generalized coordinates.
Its original 256-iteration solve fails at residual 2.1252906647024895e-5.
The retained continuation solves it in **87 total directions, 38 continuation**,
with original residual **1.0627281081627305e-11**. An independent NumPy SVD solution
passes all original laws; Rust's generalized velocity agrees within
**2.2245748266484094e-14**. Repeated Rust results match exactly. A 64-direction
control remains unconverged and consumes exactly its budget. No reference impulses
are provided to the runtime or cold-start regression.

Reproduce the independent algebra reference:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3.13 docs/performance-results/2026-09-11-coupled-search/reference.py --fixture crates/mechanic-physics/src/response/tests/fixtures/event_retry_impact.ron --output /private/tmp/event-retry-reference.ron
```

The full default drop still fails at tick 17, now with **EventSearch in all four
attempts** and all 22 attempted impacts converged. The endpoint policy still fails
at tick 20, or tick 11 with fixed eight subdivisions. The third 240-row runtime
regression remains failing. Fixing one algebra problem does not complete settling.

## Rejected event controls

An accelerated quadratic generalized path preserves the integrated endpoint and
adds the initial derivative. Translation/turning/impact and articulated finite difference derivative checks pass. However, the default drop fails at tick 11
from path-query exhaustion; using certified sweep prefixes only as reintegration
proposals moves failure to tick 12, still earlier than the retained implementation.
The archived candidate also needs interval enclosure of the root exponential-map
velocity calculation before it could be a rigorous motion certificate. Neither
candidate is retained in runtime. The analytic gravity-impact test reaches its
physical bounds but its old expectation of more than one refinement fails.

Doubling from each accepted non-nested prefix instead of restarting with the
whole remaining interval fails at tick 5. It is also removed. No event budget,
contact window, physical tolerance or publication rule was relaxed.

## Rejected central-path runtime prototype

An independent NumPy central-path calculation reaches the third 240-row fixture's
original laws in 215 directions, with residual 2.5046641312601725e-11. Its mixed
direction solve retains at most 71 rows. This calculation still stores the full
response matrix and is algebra evidence, not a portable runtime implementation.

The test-only Rust port does not meet the retained comparisons: it fails the later
240-row fixture, and its third-fixture velocity differs from the independent
reference by approximately 2e-7, above the 1e-8 bound. Polishing within the original
256-direction budget does not resolve that discrepancy. Passing a contact residual
alone does not establish reference agreement. The cause remains unresolved.

The entire Rust candidate was removed and the previously verified physics source
restored; all 541 frozen source hashes were checked before the subsequent app
visibility change. `central-prototype.tar.gz` retains the independent calculations,
candidate sources, logs and failed comparisons. No central-path solver has been
enabled in the application or retained CPU runtime.

## Verification

- `cargo test --workspace --no-fail-fast --offline -- --test-threads=1`, native
  serialized **Apple M1 Pro / Metal**: completed, **not green**. Physics 135 passed /
  4 failed; app 739 passed / 1 failed / 6 ignored; GPU 110 passed / 11 failed /
  1 ignored. Core 270 and world 91 pass. The same twelve app/GPU failures are
  recorded in the preceding checkpoint. WGSL validation passes.
- The earlier sandbox run could not expose the adapter; it is separately retained
  and does not count as a hardware pass.
- `cargo clippy --workspace --all-targets --offline -- -D warnings`: pass.
- `cargo fmt --all -- --check`: pass.

`identity.json` freezes 541 build-source hashes and a 13-file overlay on the
preceding coupled-search checkpoint. Apply its overlay after that checkpoint's
base and overlay; use a fresh Cargo target directory when rebuilding references.
Raw logs and rejected control sources are preserved beside this document.
