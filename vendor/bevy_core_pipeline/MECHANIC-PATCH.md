# Opt-in opaque pass partition

Source: Bevy v0.19.0, commit c6f634ca9f406d68ba5109d921247b654cb42c10,
`crates/bevy_core_pipeline`. Upstream MIT/Apache-2.0 licenses and formatting
configuration retained. Sibling dependencies reference the same pinned git tag;
workspace lint inheritance removed, matching the other vendored Bevy crates.

`core_3d/main_opaque_pass_3d_node.rs` adds the optional `OpaquePassPartition`
resource. Without it, opaque rendering uses the existing pass. With it, a view's
opaque draws can be partitioned into contiguous labeled runs, each in an actual
render pass. Draw order, batching, shaders, viewport, depth and color attachment
contents are preserved. Alpha-masked drawing and skybox remain afterward. An empty
trailing pass is avoided because Metal can omit its end timestamp.

This is diagnostic instrumentation, not an optimization: extra attachment
store/load and MSAA resolve operations change the workload. Per-partition times
are not the exact incremental cost of those draws in the unsplit pass. The app
enables this only with MECHANIC_PERF_TERRAIN_PASSES=1 and a capture directory.

Depends on the local bevy_render `draw_batch_keys` / `render_filtered` hooks.
Their traversal order must match, including all batching modes and unbatchable
and non-mesh items. A debug assertion checks the visited count per partition;
a regression checks contiguous run order, and the app has a real GPU pixel and
timestamp check with both terrain and another opaque material.
