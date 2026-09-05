# Single-projection terrain shader experiment — rejected

The candidate failed the existing image gate and showed no repeatable GPU gain
in the flat fixtures. The runtime shader was restored byte-for-byte to the
starting shader. No TEST4 comparison was run: a candidate that fails appearance
validation does not advance to application performance measurement.

## Candidate and scope

The starting shader has separate sampling code for X, Y and Z projections.
The candidate added a fast path when only one projection exceeds the existing
0.001 sampling threshold: select its UV, weight and normal mapping, then issue
one shared set of color, ORM and normal texture instructions. Other projections
retain the starting path. Matching gradients were selected from derivatives of
the original terrain coordinates and passed to `textureSampleGrad`, to avoid
derivatives of UVs selected across different axes. Normalization, material
weights, textures, projection thresholds and footprint fade were retained.

The exact rejected source is `candidate.wgsl`; the starting source is
`baseline.wgsl`. Neither is a runtime mode. An initial compile error used the
reserved WGSL name `active`; it was corrected before the measured comparison.
Both logs are retained. The precise cause of the image differences was not
isolated; the change in sampling/gradient evaluation is a plausible mechanism,
not a proven attribution. The quality failure is sufficient to reject this
candidate without weakening the gate or trying another sampling variation.

## Native Metal results

Apple M1 Pro, 4096×2524, 4× MSAA, real terrain maps and runtime mips, baseline
materials, no diagnostic pass partitioning. The existing offscreen test collected
80 actual opaque-pass GPU timestamps after shader-change warm-up per measurement,
in A/B/B/A order. These are medians in milliseconds, not smoothed F3 values.

| Fixture | A1 / A2 | B1 / B2 | Maximum channel delta | Channels over 2 |
| --- | ---: | ---: | ---: | ---: |
| Flat, 33×33 grid | 3.144 / 2.718 | 2.731 / 2.731 | 1 | 0 |
| Flat, 513×513 grid | 5.260 / 5.263 | 5.280 / 5.309 | 1 | 0 |
| Blended hills, 33×33 grid | 4.412 / 4.423 | 4.704 / 4.690 | **24** | **3,476** |

The flat coarse A1/A2 drift rules out crediting its first apparent reduction as
a repeatable gain. The dense fixture did not improve; the blended fixture was
about 6% slower in both pairings and exceeded the fixed maximum channel-delta
gate of 2. Images contain 41,353,216 channel bytes each.

A subsequent A/A control returned **zero differing bytes in every fixture**.
Clippy ran concurrently with that control, so its GPU times are not performance
evidence; it was used solely to check deterministic image output. The measured
candidate screening used the optimized development test executable, not a release
acceptance benchmark. No release performance or integrated gain is claimed.

## Reproduce and inspect

The test now accepts optional reference/candidate WGSL paths. This preserves the
older frozen reference used by previous work while allowing each new experiment
to compare against its actual starting point. Without either variable the
existing reference and compiled-in candidate remain the defaults.

```sh
MECHANIC_RENDER_EXPERIMENT=baseline MECHANIC_PERF_TERRAIN_PASSES=0 \
MECHANIC_TERRAIN_REFERENCE_SHADER="$PWD/docs/performance-results/2026-09-05-terrain-single-projection/baseline.wgsl" \
MECHANIC_TERRAIN_CANDIDATE_SHADER="$PWD/docs/performance-results/2026-09-05-terrain-single-projection/candidate.wgsl" \
cargo test -p mechanic-app terrain_shader_preserves_pixels_and_measures_gpu_cost -- --ignored --nocapture
```

Expected: the blended fixture rejects this candidate. For the passing A/A control,
point both variables to `baseline.wgsl`. Use `--release` for future performance
acceptance runs, and require the image gate before an integrated comparison.

Native logs include exact image paths under `mechanic-terrain-shader-52942`
(candidate) and `mechanic-terrain-shader-52965` (control) in the system temporary
directory. Source/test-binary fingerprints are retained in `fingerprints.json`.

Verification: A/A GPU/pixel control passed; app all-target Clippy with warnings
denied passed; formatting and `git diff --check` passed; restored runtime shader
matched `baseline.wgsl` exactly. The only retained code change is optional source
paths in the ignored benchmark. No solver, visual-quality setting, queue,
scheduling, world or vehicle changes were made. No full workspace/hardware matrix
rerun was warranted for that test-only change.
