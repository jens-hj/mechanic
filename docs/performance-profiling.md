# Performance profiling

Reference captures use an M1 Pro at 1920×1080, an uncapped presentation mode,
and a release-equivalent binary with symbols:

```sh
cargo build --profile profiling -p mechanic-app
cargo build --profile profiling -p mechanic-app --features profiling-tracy
```

For CPU evidence, open Xcode Instruments, choose **Time Profiler**, and attach
to `target/profiling/mechanic-app`. Enter the world, allow the loading screen to
finish, then capture idle, fixed-path travel, and one continuous terrain-brush
stroke separately. Keep the camera path and world seed fixed and record the
capture duration with the benchmark JSONL.

For GPU evidence, launch the same binary from Xcode's Metal frame capture or
attach with **Metal System Trace**. Capture a fully streamed idle frame before
capturing travel or digging so shader and upload costs are not confused with
cold asset loading. Retain the display resolution, LOD distances, terrain
materials, and 1 km horizon used by the shipped configuration.

The headless deterministic terrain scenarios emit machine-readable JSONL:

```sh
cargo run --profile profiling -p mechanic-bench -- --scenario terrain_stream
cargo run --profile profiling -p mechanic-bench -- --scenario terrain_dig
```

The articulated size sweep isolates the load-time serial/general route boundary:

```sh
for scenario in open_bearing four_bearing_contact bearings_16 bearings_64 bearings_65 bearings_256; do
  cargo run --profile profiling -p mechanic-bench -- --scenario "$scenario"
done
```

The application keeps at most three physics ticks in flight. Additional due
ticks remain in a monotonic CPU backlog, visible in the F3 overlay, until a
staging slot completes; a temporary slow frame therefore cannot amplify into
an unbounded Metal queue. On the M1 Pro, the dependency-safe single-dispatch
four-bearing contact route reduced the retained 10-second sample from 18.683 ms
GPU p95 and 44.69 TPS to 4.963 ms and 172.49 TPS with zero flags and residuals.
The load-time crossover is scene-specific: flat-ground scenes keep the fused
serial contact route only through four bearings because they can produce one
persistent contact per collider, while streamed-world scenes use it through 64
bearings because their contact set is normally sparse. Reduced-coordinate
velocity projection is fused through 64 bearings in both cases. A retained
M1 Pro no-ground sweep measured 16 and 64 bearings at 3.515 ms and 3.947 ms GPU
p95 with zero contacts, flags, or residuals. A longer flat-ground sweep measured
64 and 65 bearings at 5.452 ms and 6.143 ms GPU p95, so the dense-contact 4 ms
gate and 64/65 timing continuity remain open.

Do not infer the final frame, render-CPU, or render-GPU gates from the headless
terrain-stage numbers. Those gates require the integrated app capture on the
reference hardware.
