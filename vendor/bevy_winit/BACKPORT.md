# Optional macOS activation control

Source: Bevy v0.19.0, commit c6f634ca (the workspace-pinned git checkout).
Original licenses retained. Cargo sibling paths resolve to that same git tag;
workspace lint inheritance removed for the standalone vendored package.

The only runtime change exposes `WinitPlugin::prevent_activation` (default false).
On macOS it passes the inverse to winit's existing
`EventLoopBuilderExtMacOS::with_activate_ignoring_other_apps`. Other platforms
ignore it. The default preserves existing behavior. Mechanic enables it only
for opt-in automatic background captures and also creates the window unfocused.
No event-loop, rendering, presentation or physics scheduling changes are made here.
