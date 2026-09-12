# Immutable simulation mesh visibility

The application already renders immutable body-local meshes beneath moving compound
transforms. This candidate removes `NoFrustumCulling` from the ordinary, authored
and bearing body mesh bundles, allowing Bevy to calculate local bounds and use the
propagated transforms for visibility. Other editor and overlay bundles are unchanged.
The physics backend remains the existing GPU runtime.

The regression uses the same mesh bundle as production and runs Bevy's actual
transform propagation, bounds calculation and visibility systems. It checks bodies
in view, outside the frustum, behind the camera, crossing its edge and returning
into view, both before and after a large origin rebase. Rotation remains enabled
and the mesh vertices must stay unchanged throughout the sequence.

- `cargo test -p mechanic-app --offline rendering_tests:: -- --test-threads=1`:
  60 passed, 0 failed, 1 hardware test ignored.
- `cargo clippy --workspace --all-targets --offline -- -D warnings`: passed.
- `cargo fmt --all -- --check`: passed.
- `cargo build --release -p mechanic-app --offline`: passed.
- `cargo test -p mechanic-app --offline grounded_functional_blocks_keep_visuals_during_live_publication -- --ignored --exact rendering_tests::grounded_functional_blocks_keep_visuals_during_live_publication --nocapture --test-threads=1`:
  passed separately on the native **Apple M1 Pro / Metal** adapter. This exercises
  live physics publication and mesh ownership; it is not an image comparison.

`source.patch` applies after the retry-reuse checkpoint; `identity.json` records
the before/after source hashes. These focused checks establish ECS visibility
behavior, not native rendered image quality or a frame-time improvement. Matched
foreground A/B/B/A performance and visual comparisons remain required before this
candidate satisfies the rendering retention gate. No integrated gate has passed.
