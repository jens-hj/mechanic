# Local render-pass timestamp hook

Source: https://github.com/bevyengine/bevy, tag `v0.19.0`, commit
`c6f634ca9f406d68ba5109d921247b654cb42c10` (`crates/bevy_render`).
Upstream MIT/Apache-2.0 licenses are included.

Local changes: sibling dependencies reference the same pinned Bevy tag; workspace
lint inheritance is removed. `renderer/render_context.rs` exposes an optional
timestamp allocator for tracked render passes, re-exported by `renderer/mod.rs`.
Existing pass timestamp writes take precedence. No additional GPU work is inserted.
Raw encoder passes and externally encoded command buffers are not intercepted.

Keep this patch minimal and remove it once an equivalent upstream hook is available.

Additional diagnostic hooks in `render_phase/mod.rs`: `BinnedRenderPhase`
exposes prepared draw batch keys in render order, and `render_filtered` accepts
a predicate over those same prepared items. Normal `render` selects every item.
Filtering occurs before draw-function availability checks, preserving traversal
indices even for unavailable functions. Batching, indirect ranges and draw order
are unchanged. The optional bevy_core_pipeline opaque partition uses these hooks
to put timestamp boundaries around contiguous material runs on adapters that
cannot timestamp inside render passes. Keep the key traversal synchronized with
all five paths in the draw traversal.
