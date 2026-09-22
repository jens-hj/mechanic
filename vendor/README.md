# Vendored crates

Each directory is an upstream crate at a pinned release with one Mechanic
change. The workspace `[patch]` tables in the root `Cargo.toml` substitute them;
they are excluded from the workspace, so `cargo xtask ci` does not lint or test
them. CI runs the wgpu fence regression test separately.

| Crate | Note | Change |
|---|---|---|
| `bevy_core_pipeline` | [`MECHANIC-PATCH.md`](bevy_core_pipeline/MECHANIC-PATCH.md) | `OpaquePassPartition` |
| `bevy_render` | [`MECHANIC-PATCH.md`](bevy_render/MECHANIC-PATCH.md) | Render-pass timestamp hook |
| `mosaic-macros` | [`MECHANIC-PATCH.md`](mosaic-macros/MECHANIC-PATCH.md) | Reactive shape geometry with static appearance |
| `mosaic-text` | [`MECHANIC-PATCH.md`](mosaic-text/MECHANIC-PATCH.md) | Positioned scalable glyph outlines from existing shaping |
| `bevy_winit` | [`MECHANIC-PATCH.md`](bevy_winit/MECHANIC-PATCH.md) | `WinitPlugin::prevent_activation` |
| `wgpu-core`, `wgpu-hal` | [`WGPU-FENCE-BACKPORT.md`](WGPU-FENCE-BACKPORT.md) | Fence-acquisition backport; the literal patch is [`wgpu-fence-upstream.patch`](wgpu-fence-upstream.patch) |

Every note names the upstream revision the copy was taken from.

## Updating a vendored crate

1. Read the crate's note for its upstream source and the intent of the change.
2. Replace the directory with the new upstream release of the same crate.
3. Re-apply the change. For wgpu, apply `wgpu-fence-upstream.patch`, or drop the
   vendored copy entirely once the release contains the upstream fix.
4. Update the note's source revision, and the version in the root `Cargo.toml`
   if the release changed.
5. Run `cargo xtask ci` and the two fence tests from `.github/workflows/ci.yml`.

A change that upstream has released is no longer vendored: delete the directory,
its `[patch]` entry, and its `exclude` entry together.
