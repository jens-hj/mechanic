# CPU contact solver repair — incomplete

The application still uses `GpuPhysics`. This checkpoint does **not** make the
CPU backend usable for sustained driving. It retains several independently tested
repairs and the still-failing required regressions. No physical, performance,
vehicle, or integrated acceptance gate has passed.

## Retained changes

- Bounded central-path proposals supplement the contact smoothing search. They
  include scalar motor/stop/bilateral bounds. Failed proposals can improve the
  unpublished starting guess, but only the original unsmoothed contact laws and
  physical tolerance authorize convergence. Every search consumes the original
  256-direction budget; no larger per-tick budget was introduced.
- Finite-interval drive equations use a stable slack-space form near saturation.
  This avoids subtracting a barrier gap smaller than one impulse ULP. The actual
  drive bounds, static/kinetic mode proof, and final residual remain unchanged.
- Above 128 mixed equations, streamed rank-revealing QR stores at most 128 basis
  directions and applies the response implicitly. It does not allocate the full
  larger response or mixed matrix. This is a correctness experiment with high
  operator counts, not a performance improvement. Matrix-storage diagnostics
  include original and eliminated local matrices, inverse-dynamics columns and
  all simultaneously live rectangular QR factors. Candidate validation response
  applications are counted even when the candidate fails.
- Split position recovery can stop at a certified correction prefix when it
  reaches new finite geometry, refresh its constraints, and continue. Recovery
  does not change physical velocity or elapsed time. Its regression independently
  checks every collider against the floor, coordinate bounds, and repeatability.
- Numerical-zero activation preserves an actual slightly penetrating hull vertex
  when rounded face clipping returns empty. Signed gaps remain within the same
  fixed 1e-12 m window. No finite triangle is enlarged or replaced by a plane.

The core geometry regression, recovery regression, local derivative checks,
implicit operator/transpose tests, redundant minimum-norm solve, and bounded-rank
storage test pass. Several previously failing captured impacts now pass their
original laws. A passing instantaneous algebra problem is not a completed tick.

## Required failures

The final complete CPU run has **148 passes / 4 failures**:

| Required case | Failure |
| --- | --- |
| Default event-resolved 120-tick cold drop | Tick 17, event search exhausted |
| Endpoint-policy cold drop with bounded retries | Tick 30, impact solve rejected |
| Endpoint-policy cold drop with fixed eight substeps | Tick 1, impact solve rejected |
| Captured support-transition impact | Original residual about 1.58e-6; 256-direction cap exhausted |

These are observed results from this source trajectory. Earlier exploration
reached ticks 24, 30, 36, 57, and 61 on different builds. They do not form one
accepted trajectory and do not establish monotonic physical progress. In
particular, the fixed-eight result is earlier than the prior checkpoint's tick-11
rejection. Do not promote the search candidate to the application.

The final captured-case suite passes 14 / 15 tests. The final complete response
suite passes 56 / 57; the support-transition runtime solve remains failing. The
independent dense NumPy reference solves that same fixture with original residual
3.81e-12, and its impulses independently pass the Rust contact laws and dynamics
check. It supplies evidence that the algebra is solvable, never a runtime initial
guess. Its script and input hashes are retained in the reference status JSON.

The older third-impact reference and independently traced solutions differ by
approximately 1.7e-7 m/s. This still does not prove the earlier 1e-8 reference
comparison. Do not replace that comparison with a residual-only claim or silently
change its reference.

## Rejected controls

The raw archive preserves full diagnostics, captured inputs, and candidate source
files, including these controls:

- Dense 144-equation and rank-truncated dense controls solve a settling fixture;
  they were removed because they exceed the runtime matrix-storage boundary.
- Restarted GMRES and independent LSQR controls do not solve the retained cases.
- Smaller local pivots do not resolve the settling failure.
- Unsmoothed Newton polishing does not solve the support-transition failure and
  was removed. Reserving more final iterations alone also fails.
- Continuing failed monotone stages, restarting smoothing from zero, and adding
  linear iterative refinement all fail the support-transition case. These
  controls were removed; the bounded branch-following candidate remains.

No activation window, joint bound, penetration gate, contact residual tolerance,
or publication rule was relaxed. Failed ticks remain unpublished.

## Verification and source identity

`identity.json` records 553 source/fixture hashes and a standalone source archive.
Rebuild frozen references with a separate Cargo target directory. Do not share
cached binaries between a frozen source tree and the dirty development tree.

Final commands and full outputs are retained under `verification/`:

- `cargo test -p mechanic-physics --offline -- --test-threads=1`: 148 pass /
  4 fail. All four required failures remain active tests.
- `cargo test --workspace --exclude mechanic-physics --no-fail-fast --offline -- --test-threads=1`,
  native and serialized on **Apple M1 Pro / Metal**: core 271 pass; world 91 pass;
  app 740 pass / 1 fail / 6 ignored; GPU 110 pass / 11 fail / 1 ignored. WGSL
  validation passes. All twelve app/GPU failure names match the earlier frozen
  checkpoint; no unavailable adapter is counted as a pass. Workspace is not green.
- `cargo clippy --workspace --all-targets --offline -- -D warnings`: pass.
- `cargo fmt --all -- --check`: pass.
- `cargo build --release -p mechanic-app --offline`: pass. This builds the existing
  GPU application, not a selectable CPU backend.

`investigation.tar.gz` preserves the raw exploratory files and rejected candidate
sources. `identity.json` also records the final test/app binary and archive hashes.
Test durations under concurrent compilation are not performance measurements.

Remaining work: resolve the captured contact search and sustained event-search
failures, repeat all 120-tick policies and hashes, then implement and validate the
shared world interface and actual authored driving. Body/body collision, loops,
streaming readiness, sleeping, CPU application selection, and all speed gates
remain open in the master plan.
