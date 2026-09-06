# Native tool effects

`ToolFxPlugin` ports the supplied `effects.zip` reference (`fx.js` alongside this
file) into the app. The existing weld publication and dimension-hold physics are
unchanged. Effects are transient and do not add persistence fields.

The app owns 1,400 shard slots, 900 trace slots, 220 halo dash slots, and 12 solid
outline segments. There are three persistent GPU instance buffers and at most
three particle draws in the world camera: scalene triangles, traces, and combined
halo/outline lines. Saturated pools and the bounded one-shot queue drop excess
requests. No particle entities, materials, shadow passes, or collision queries
are created. Shards write depth; additive lines only test depth. Custom batches
opt out of automatic mesh batching and are extracted only for the world camera.

Accepted hammer releases emit once at the current body-relative impact point and
surface normal, independent of the number of impulse delivery ticks. Matter
captures the target before a release mutates editor hover/context and emits once
only if that gesture commits a history revision. Weld emission requires a valid
active destination gesture; connector emission resolves the current wire endpoint.
The animated rig exposes its actual active-end transform, rotating Matter ring,
and connector phase. Connector traces also stream from all six animated plate-tip
lamps, using their propagated world transforms. The seven streams converge at
the live wire endpoint and rebuild each frame (81 segments, one trace draw). The ordinary focus, pointer blocking, selected-tool, and
viewmodel-visibility rules gate tool emission. Holding a creation does not disable
editing feedback.

Freeze casts are requested only after `begin` succeeds. Loading a saved hold does
not call that path. A read-only hold snapshot supplies the link identity and exact
world bounds of held colliders. The FX layer caches bounds by construction/pose
revision and link identity. Wakes use actual prescribed vertical displacement,
interpolating emission along the traversed interval. Hold release clears violet
particles and lines and resets the aperture multiplier without clearing editing
feedback. World transitions and floating-origin shifts clear transient storage.

## Deliberate adaptations

- Numerical counts, lifetimes, ramps, scalene vertices, pull terms, gravity, and
  tone/size steps follow `fx.js`, including its 0.24 s sledge trace lifetime.
- Sustained emission uses fractional accumulators at the reference's expected
  60 Hz density: Matter 120 shards/s; welder 300 arcs, 180 shards, and 30 wisps/s;
  lift wakes 120 traces/s. Simulation delta is capped at 50 ms.
- Connector segments are replaced every frame, including the four spool segments;
  their phase follows the rig instead of accumulating overlapping old beams.
- Welder spatter uses the live destination surface basis rather than a fixed
  upward axis. Gravity remains world-down.
- The halo uses a live world-space AABB, padded 75 mm. Large bounds scale the
  55:125 dash/period ratio proportionally. Short edges that would fall entirely
  within a scaled gap retain a moving marker; every edge remains represented.
- Dash travel (0.16 m/s) and the 0.55 Hz triangular aperture pulse use the same
  clamped clock. Only the six central atlas aperture regions multiply emission
  by 0.75–2.30. The fracture field keeps its original emission.
- The reference bypasses tonemapping. Native FX participate in the existing world
  tonemapper, so HDR bloom and opaque scene occlusion remain coherent.
- Bloom uses additive composition, intensity 0.035, threshold 0.65, softness 0.2,
  and no low-frequency boost. Bevy's default bloom pyramid contributes 16 passes,
  counted separately from the three particle batches.
- World HDR rendering, bloom, and the existing world tonemapper finish first.
  The viewmodel clears a transparent LDR intermediate and skips its output copy;
  the x-ray/Mosaic camera loads that intermediate and composites it once over the
  world. Both overlay cameras use the same MSAA. UI and viewmodel receive no bloom.

## Reproduction

The debug-only capture fixture renders every effect, a moving hold with editing,
and a no-effects/no-bloom baseline against dark and bright backdrops using the
app's real shaders, viewmodel, x-ray and Mosaic passes. It sets a 75° FOV so the
large rig is visible, requests no window activation, disables vsync for timing,
and writes screenshots plus frame-time JSON. The fixture uses synthetic emitter
and hold inputs; gameplay acceptance and cancellation are covered by tests.

```sh
cargo build -p mechanic-app
BEVY_ASSET_ROOT="$PWD/crates/mechanic-app" \
MECHANIC_FX_CAPTURE_DIR=/tmp/mechanic-fx-capture \
MECHANIC_CREATIONS_DIR=/tmp/mechanic-fx-capture/creations \
target/debug/mechanic-app
```

Use both directory variables to isolate settings and test saves. The fixture
redirects its world store beneath the capture directory and exits on completion.
It is absent from release builds. Frame times are a small debug-render fixture,
not a controlled performance benchmark or a scale-gate result.

## Verification

- Focused effects tests cover accepted/rejected placement, once-per-gesture
  consumption, saturation, expiry, stepped tones, emission rates, hitches,
  connector retarget/cancel, all-edge halo coverage, shared pulse range, atlas
  masking, stationary/moving wakes, and selective hold/world cleanup.
- Existing weld tests also assert emission starts only at a valid active gesture
  and stops on invalidation/publication. Freeze tests verify held-only bounds and
  pose tracking. Existing physics/alignment tests remain unchanged.
- `cargo test --workspace --offline` ran both sandboxed and with Metal access.
  The Metal GPU suite reports 72 passed and 11 failed. The 11 failing test names
  exactly match `/tmp/mechanic-weld-workspace-final.log`, the existing baseline
  report; no core or GPU code changed. These are the inherited car/contact,
  cylinder/pipe-bore, material-friction/restitution, servo, and pendulum failures.
- `cargo test --workspace --exclude mechanic-gpu --offline` passes (including
  683 app tests, four ignored hardware tests, 221 core and 88 world tests).
- `cargo clippy --workspace --all-targets --offline -- -D warnings` and
  `cargo fmt --all -- --check` are required completion checks.

Capture measurements and visual inspection are recorded in `verification.md`.
