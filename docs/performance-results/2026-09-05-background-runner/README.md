# Background runner verification — 2026-09-05

Real Apple M1 Pro / Metal run passed using `scripts/run-background-capture.py`
with TEST4 and output `/tmp/mechanic-background-verify-01`. Release build passed.
Ghostty was foreground before and after launch. All 1,733 captured CPU frames
reported `focused: false` and `automated_background: true`. No OS input or
screenshot automation was used. The 60-second capture completed, Bevy saved
its PNG, the process exited with status zero and the launcher removed its
unique temporary world. The original world manifest hash remained unchanged.

Raw capture and summary are retained here. Every frame reported 4112×2524,
4× MSAA and AutoNoVsync. Terrain backlog stayed zero; physics failure flags
were zero and all 1,734 asynchronous render GPU samples were Complete.
Submitted/readback ticks passed the existing consecutive-tick validation.
These timings are background diagnostics, not a new foreground A/B comparison.

The Bevy screenshot was visually inspected: saved stationary scene, matter
manipulator, car offscreen, F3 visible. Original full-resolution PNG:
`/tmp/mechanic-background-verify-01/capture-1788627038767868000-48963.png`.
The test proves unfocused operation on this Mac; it does not certify every
minimized/occluded/App Nap state or other operating systems.

Validation passed:
- Input suppression regression (keyboard, mouse, camera motion and scroll).
- Existing normal cursor-capture regression.
- All three recorder regressions.
- Launcher timeout/world-isolation regression and four raw-summary tests.
- `cargo clippy -p mechanic-app --all-targets --offline -- -D warnings`.
- Workspace formatting, vendored bevy_winit formatting and `git diff --check`.

No workspace-wide physics acceptance claims are added by this check.
