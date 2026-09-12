# Suspension car in generated terrain

The 60 TPS / 16.67 ms p95 acceptance gate has **not passed**. The generated-terrain
baseline also overturns during steering, despite finite transforms and zero
kernel failure flags. Do not interpret successful process exits as stable-car
or performance acceptance.

## Reproduction

```sh
cargo run --release -p mechanic-bench --bin suspension-world -- --seconds 60
cargo run --release -p mechanic-bench --bin suspension-world -- --seconds 60 --plane
```

The fixture loads `creations/suspension-car.mech` without changing its springs,
damping, restitution, drive limits, materials, or geometry. Creation SHA-256:
`5522ce6bb8538ee4c9bfc6fca9ba0698a3da5d2ef11351676e7038833589c56e`.
The adapter is Apple M1 Pro / Metal. Both versions use release optimization,
seed 91, three simulated seconds of settling, full-speed acceleration, alternating
three-second straight/left/straight/right phases, and production leaf meshes.
A 3×3×3 brick neighborhood is replaced at brick crossings and every 600 ticks.

Each run measures 3,600 completed ticks (60 **simulated** seconds) after warm-up;
slow runs take more than 60 wall-clock seconds. Readback is deliberately serialized
to associate each preparation/publication cost and transform with its tick.
Consequently its zero submitted/completed queue backlog is imposed by the harness,
not proof of the World scheduler's backlog gate. These are physics stress traces,
not World frame timings. Meshing the replacement neighborhood is included in
`preparation_ms`; this harness does not model the World's asynchronous meshing
scheduler. Do not attribute all of that CPU time to GPU geometry packing.

The preserved baseline executable was built before the traversal/publication and
recovery optimizations, with the added GPU stage timers. `binaries.json` identifies
the executables used for repeated measurements. The original baseline trace does
not contain uploaded-byte counters; do not infer exact bytes from triangle count.
Later repeated baseline runs exposed unordered multi-pass markers: some GPU totals
exceed wall tick latency and a nested projection span exceeds its parent. All
baseline GPU timing distributions are therefore omitted from the final comparison.
Their wall timing, contacts, poses, and failure flags remain available. Candidate
traces use the corrected ordered markers and are validated before summarization.
The candidate records actual CPU-to-GPU upload bytes and chunk reuse.

GPU stage durations come from timestamp queries. `terrain_ms` measures the whole
terrain BVH/contact-generation kernel; it does not distinguish traversal inside
that kernel from finite-triangle SAT, clipping, and support reduction. Terrain traversal is a subset
of narrowphase, and recovery projection is a subset of recovery: do not add those
subsets to their parents. Metal drops timestamp writes on empty compute passes,
so diagnostic stage boundaries use a one-workgroup marker dispatch. Candidate
markers use a bit-preserving atomic access to the pose buffer to retain ordering
when intervening indirect work is zero. An initial candidate trace with unordered
markers was rejected and excluded. These markers add diagnostic overhead. Wall tick latency includes preparation, publication,
submission, GPU execution, and synchronous readback servicing.

## Measured results

Nearest-rank percentiles recomputed from raw ticks; all latencies below are milliseconds.

| Run | Completed TPS | Tick p50 | Tick p95 | Tick p99 |
| --- | ---: | ---: | ---: | ---: |
| baseline-01 | 12.52 | 69.23 | 169.38 | 189.89 |
| baseline-02 | 11.73 | 71.38 | 176.41 | 196.32 |
| baseline-03 | 12.36 | 69.33 | 171.87 | 196.94 |
| candidate-01 | 13.32 | 66.91 | 164.99 | 188.12 |
| candidate-02 | 11.72 | 70.90 | 174.23 | 195.40 |
| candidate-03 | 11.17 | 73.50 | 175.27 | 200.52 |
| plane-01 | 63.28 | 15.70 | 16.98 | 17.08 |

There is no repeatable overall improvement in this stress fixture. Candidate TPS
ranges overlap the baseline and its median is lower. All six terrain runs
overturn; all completed runs retain zero kernel failure flags and finite transforms.

The corrected candidate terrain contact kernel has p95 **136–147 ms**, versus
**5.5–6.2 ms** recovery and **11.3–11.4 ms** contact-solver p95. Terrain contact
generation remains the measured primary GPU bottleneck; top-level pruning alone
does not remove its expensive work on intersecting chunks. Recovery projection
is nested inside recovery and must not be added again.

Candidate CPU publication p95 per update is **0.77–0.83 ms**. Rebuilding and packing
the entire requested neighborhood takes **114–115 ms p95** per update in this
synchronous stress harness. This is not the World worker-publication latency.
The forced replacement workload uploads **325–404 MB** per run; generations are
deliberately changed on replacement, so this workload does not demonstrate
unchanged-generation upload savings. Those are verified by the incremental
publication regression; the headless traces do not establish World savings.

See [summary.json](summary.json) and the individual JSONL traces for distributions,
upload totals, publication counts, and explicit timestamp-validity counts.
Baseline GPU distributions are intentionally absent, not interpreted as zero.

## Implementation

- A balanced stackless top-level BVH selects chunks, retaining their triangle
  BVHs and exact finite-triangle narrow phase.
- Worker preparation caches chunk-local packed rows by node, generation, global
  chunk origin, and active collision groups. A floating-origin shift changes
  placement rows, without repacking triangles. Global origin subtraction remains
  double precision. The main-thread handoff reuses cached packed chunks and copies
  only changed mesh geometry before worker preparation.
- Geometry allocations survive publication. Changed chunks reuse retired ranges;
  buffer growth copies retained geometry on the GPU. Only changed geometry,
  placement rows, and the small top-level hierarchy are uploaded from the CPU.
- Prepared updates carry source revision and physics origin. The app rejects stale
  cuts before any publication and waits before submitting dependent ticks.
  Validation failures preserve the previous accepted scene. Publication clears
  contact warm starts; invalidation is conservative for the complete contact cache.
- Unjointed-body eligibility suppresses rotational sweep/clamp dispatches for
  fully articulated scenes. Previous poses are still captured for translational CCD.
- Recovery builds a compact contact/joint list from penetrating components,
  including components connected through contacts. Indirect dispatch skips
  unnecessary correction and pose projection. Inner sweeps terminate early only
  at an exact zero-impulse fixed point, stricter than the existing tolerance.
  Maximum budgets remain three correction rounds and 32 sweeps.

## World capture

Create a disposable fixture in a scratch world store:

```sh
cargo run --release -p mechanic-bench --bin suspension-world -- \
  --write-world /tmp/mechanic-suspension-worlds
```

The background runner expects its source in the application's world store and
makes another disposable copy before launching. It never sends OS input:

```sh
python3 scripts/run-background-capture.py --world /path/to/test-world \
  --output /tmp/suspension-world-capture --drive
```

`--drive` sends application-level W/A/D state to the sole input-linked seat, using
submitted physics ticks as its clock. The normal recorder keeps frame timings,
physics stage timings, publication latency/bytes, and backlog separate. Add
`--demonstration` for a screenshot every five seconds; that run is a visual
demonstration and its frame timings include screenshot overhead. Background
captures are diagnostics, not a controlled foreground acceptance benchmark.

The final 60-second background demonstration completed after the changed-only
worker handoff fix. See [driving recording](world-demo/driving-5x.mp4),
[final screenshot](world-demo/capture-1788805580964420000-60260.png), and
[World summary](world-demo/summary.json). The recording samples one frame every
five wall-clock seconds, played at one frame per second (5× time-lapse).
The disposable source and run copies were removed afterward.

| World measurement | p50 ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: |
| Frame | 45.72 | 130.70 | 138.10 |
| Physics GPU | 18.00 | 50.29 | 52.08 |
| Submission to readback | 149.43 | 247.21 | 254.18 |
| Contact solver | 11.60 | 34.55 | 35.98 |
| Terrain contact kernel | 0.68 | 5.44 | 6.34 |
| Terrain recovery | 1.61 | 11.81 | 13.19 |

World completed **16.85 TPS**; backlog grew **709 → 3,288 ticks**. Failure flags
remained zero. The World contact solver is the largest measured physics stage,
whereas the deliberately dense headless terrain fixture is dominated by terrain
contact generation. Submission-to-readback includes queueing and is not isolated
tick execution. This capture contains no terrain publications after warm-up, so
it cannot establish streamed-publication latency or upload savings. Window and
render target were 4112 × 2524 with 4× MSAA. Most render timestamp samples report
`Partial: reversed`; do not use the sparse complete render spans as a reliable
render GPU distribution. Frame wall timings and physics timings are recorded
separately. No controlled foreground or three-run World acceptance is claimed.

## Verification

- `cargo test -p mechanic-gpu terrain --lib -- --nocapture --test-threads=1`:
  26 passed, one diagnostic trace ignored on M1 Pro / Metal, after the final
  recovery allocation and worker-handoff changes.
  Covers generated impacts/caves, 20 m/s penetration, dense curved terrain, slopes,
  rotational CCD, restitution, and articulated/suspension recovery through the
  existing General and Fused fixtures.
- The extended incremental-publication regression passed separately after the
  buffer-growth change. It checks unchanged reuse, partial replacement, stale
  revisions/origins, failed publication retaining support, origin-shift reuse,
  and retirement. CPU tests cover seam selection and failed-preparation cache reuse.
- `cargo test -p mechanic-core -p mechanic-world -p mechanic-bench`: 242 core,
  88 world, and four benchmark tests passed.
- `cargo test --workspace -- --test-threads=1`: app suite stopped at the existing
  `ui::suspension::tests::leader_geometry_follows_projected_positions_without_remounting`
  failure (`leader marker missing at Vec2(850.0, 450.0)`); 727 passed and six ignored.
  This same failure is documented in the preceding terrain work. The command did
  not establish a passing workspace gate.
- The Metal zero-work timestamp regression passed: all nested projection spans
  remained bounded by the enclosing recovery span. A 120-sample car timing check
  also contained zero invalid nested spans.
- Workspace Clippy, formatting, and diff whitespace checks pass. Capture summary
  regression tests pass (seven tests, including rejection of invalid nested GPU spans).

This benchmark is separate from the repository's larger scale gates. Soil
permanent deformation and clumps are outside this change.
