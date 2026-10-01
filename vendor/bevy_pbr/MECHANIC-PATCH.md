# Environment updates and co-located camera shadows

Source: Bevy v0.19.0, commit c6f634ca9f406d68ba5109d921247b654cb42c10,
`crates/bevy_pbr`. Upstream licenses and formatting retained. Sibling dependencies
use the pinned git tag; workspace lint inheritance removed.

- `generate::EnvironmentMapGeneration` optionally gates source cubemap generation,
  downsampling, and filtering without replacing textures. Absent means the
  upstream every-frame behavior. Atmosphere generation happens after filtering;
  clients must allow consecutive frames to publish source then filtered output.
- `AtmosphereProbePipeline` is re-exported so clients can await pipeline readiness.
- `DirectionalShadowSource` is a render-world camera component. It reuses the
  immediately preceding camera's directional shadow uniforms and texture contents
  only when its source entity, world transform, and ordered directional-light list
  match. Otherwise it falls back to independent shadows. Cameras must run in that
  order and source cascade coverage must contain the consuming geometry. Local
  light and ambient/cluster data remain per-camera. Stale overlay shadow views are
  removed. CPU cascade calculation remains upstream behavior.
- Atmosphere LUT, environment, and sky passes record optional diagnostic spans.

App regression fixtures exercise cadence, warm-up, transitions, pixel equality,
shadow reuse/fallback, and paired GPU timings. No shader ABI changes.
