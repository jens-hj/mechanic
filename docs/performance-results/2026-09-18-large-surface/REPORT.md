# Large steel cube CPU contact cost

## Reproduction and cause

The reported cube had already been removed; its estimated edge length was 1–2 m. These are isolated reproductions, not measurements of that original scene. A single steel cube is dropped 20 cm onto generated terrain (seed 91), using the existing world-drive replay. Widths 4, 8 and 16 construction blocks mean 1, 2 and 4 m edges. The cube has one collider and no drives.

The CPU physical contact query first reduces terrain triangles to support manifolds, then augments them with submerged convex vertices. That second query incorrectly used full overlap recovery, which also returns every clipped triangle intersection. Small positive depths admitted those intersections into a linear-search merge, producing quadratic work and thousands of physical constraints beneath a broad face.

The fix gives buried vertices a separate query. It retains actual penetrating hull vertices and does not use the activation fallback. Full penetration recovery keeps its existing intersection geometry. The existing surface reduction rules and GPU path are unchanged.

## Method

- Intel Core i5-12600K, Linux x86_64; rustc 1.97.1.
- Release profile, thin LTO, one codegen unit; CPU solver only.
- Baseline production geometry/physics code: `c82718427f8a35204f2dea9d8962f577b568215b`.
- Both binaries include the same new benchmark scenario and timing fields.
- One sequential run per variant, with no concurrent compilation; 60 warmup ticks and 120 measured ticks.
- Every run uses a fresh temporary saved world. Edits stay in memory.
- Raw before/after JSONL and extracted summaries accompany this report.

```sh
CARGO_INCREMENTAL=0 cargo build -p mechanic-bench --release --offline --bin cpu-physics
target/release/cpu-physics --scenario large-surface --block-width 8 --warmup 60 --ticks 120
target/release/cpu-physics --scenario large-surface --block-width 8 --warmup 60 --ticks 120 --soil
```

Repeat with `--block-width 4` and `16` for the other sizes.

## Results

Times are milliseconds. Contacts are mean per-tick diagnostic counts.

| Edge | Soil | Physics p95 before → after | Total tick p95 before → after | Contacts before → after |
|---|---|---:|---:|---:|
| 1 m | Off | 5.66 → 2.63 | 5.68 → 2.64 | 405 → 7 |
| 1 m | On | 1.04 → 1.98 | 194.75 → 116.06 | 28 → 23 |
| 2 m | Off | 36.76 → 9.64 | 36.79 → 9.66 | 1672 → 7 |
| 2 m | On | 2.72 → 4.60 | 202.77 → 200.45 | 57 → 168 |
| 4 m | Off | 353.01 → 38.07 | 353.06 → 38.09 | 4128 → 7 |
| 4 m | On | 19.02 → 22.19 | 304.06 → 280.36 | 1400 → 2009 |

All runs report zero degraded ticks and `kernel_coverage_complete: false`.

Rigid terrain isolates the contact bug: the 2 m cube improves 3.8× and the 4 m stress case 9.3×. Soil changes contact distribution and mesh geometry, so those runs do not show the same solver improvement: physics p95 increases in all three soil variants while total tick p95 decreases. Mean soil accumulation/commit cost drops from 4.48 to 0.91 ms for the 2 m cube.

The replay meshes synchronously; its total tick p95 includes full brick remeshing. The app meshes asynchronously, so these totals are not app frame timings. Soil remeshing remains expensive, and sufficiently varied surface normals can exceed the existing 16 support groups and emit unreduced contacts. This change does not resolve that separate limit. The original multi-body scene was unavailable, so its frame rate and exact reported 200–300 ms spike have not been remeasured. No scale gate or universal frame-time target is claimed.

## Verification

- Core regression checks that a triangle wholly inside the bottom face produces no buried vertices, while genuine penetrating corners remain available.
- Physics regression drops a 2 m cube onto finely meshed generated terrain and checks that it settles with less than 5 mm penetration and speed below 0.05 m/s.
- `cargo test -p mechanic-core -p mechanic-physics --offline`: 323 core and 215 physics tests pass, including the previously failing captured block-stacking case.
- `cargo clippy --workspace --all-targets --offline -- -D warnings`: passes.
- `cargo fmt --all -- --check`: passes.
- `cargo build -p mechanic-app --offline`: passes; the debug app binary includes the fix.

Build/test commands used `CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0` to limit artifact size. The user must restart an already running app to use the rebuilt binary.
