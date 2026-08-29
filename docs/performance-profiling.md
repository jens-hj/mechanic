# Performance profiling

Reference captures use an M1 Pro at 1920×1080, an uncapped presentation mode,
and a release-equivalent binary with symbols:

```sh
cargo build --profile profiling -p mechanic-app
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

Do not infer the final frame, render-CPU, or render-GPU gates from the headless
terrain-stage numbers. Those gates require the integrated app capture on the
reference hardware.
