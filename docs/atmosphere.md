# Atmosphere and solar time

The main outdoor camera uses Bevy 0.19's Earth scattering medium and default
lookup-table quality, aerial perspective, and a 128-pixel environment cubemap.
Reflections refresh every 250 ms in two consecutive frames (source generation,
then filtering); the visible sky and direct lighting still update every frame.
Shader warm-up, time jumps, space transitions, and altitude changes force updates.
The garage keeps its authored exposure, fog, lighting, and
filtered environment. X-ray, viewmodel, and UI cameras do not get atmosphere
settings. The viewmodel shares the main camera’s exposure, ambient light, filtered
environment maps, and atmospheric transmittance texture. Celestial lights also
illuminate its render layer; the fixed garage key light is disabled outdoors.
The tool also reuses the world camera’s directional shadow maps when the views
share a transform and directional lights, preserving occlusion without another
set of shadow draws. Displaced or incompatible views fall back to their own maps.
This preserves the transparent tool overlay without a second sky or LUT pass.
Only explicitly marked static environments permanently retire their generator.

World format 8 stores the elapsed `day` and `time_of_day_seconds`, finite seconds
in `[0, 86400)`. New worlds start at 09:00 on day zero. One hour of active outdoor
play advances one solar day, independent of simulation speed. Pause menus,
selection/loading, garage visits, and debug frame freeze stop advancement.
Autosave and exit saves include time-only changes. There is no offline
advancement or old-format migration.

The sun, moons, and stars follow the world's [star system](celestial.md). Each
star is a directional light with a sun disk sized to its apparent diameter,
coloured by its temperature, and attenuated by the atmosphere. The brightest
light above the horizon, star or moonlight, is the only one that casts shadows.
The moons are drawn as lit spheres; one moonlight carries the light of every
risen moon. The generated cubemap supplies the background stars and the galaxy,
fixed in the system's ecliptic frame and turned by the skybox. Night exposure
and fill are intentionally brighter than a photometrically realistic night, and
adapt upwards for bright moons and distant companion stars.

The atmosphere's local tangent plane follows the camera horizontally and uses
the floating origin's vertical offset. Rebasing therefore preserves the camera's
height above the atmosphere's surface without bending the finite world around a
planet. This is for ground-level travel, not orbital views.

With developer tools enabled, `[` / `]` adjust time by one hour, `Shift+[` /
`Shift+]` by one day, and `F8` toggles cycle advancement. These actions use the existing focus and modal-input gates.
The overlay shows the day, time, running/paused/fixture status, the star system, and each moon's phase. The cycle pause is
session-only; time adjustments persist. `Shift+F8` freezes the entire debug frame.

`MECHANIC_SKY_TIME=6`, `12`, `18`, or `0` (hours since day zero) fixes the rendered time without changing
the saved clock. Fixed-time launches default to a temporary world store, so player
saves are isolated. An automated capture can explicitly supply a disposable store
with `MECHANIC_AUTO_WORLD_STORE`.

## Verification

Focused tests cover the hour-long cycle, frame-rate independence, midnight wrap,
invalid persisted times, per-world round trips, time-only saves, pause/input gates,
map restoration, continued outdoor generation, celestial alignment, and rebasing.

The opt-in real-GPU fixture renders the production terrain and water shaders,
matte and metallic objects, atmosphere, and bloom at four times with a fixed
camera. Additional disk views point along the same sun/moon direction used by
lighting. It writes PNGs and `timings.json` to the printed temporary directory:

```sh
cargo xtask test --exclude mechanic-bench atmosphere_captures_four_times_and_reports_cost -- --ignored --nocapture
```

It compares the previous fixed-light/fog/environment configuration with the new
sky at 768 × 512, with 4× MSAA and the existing tonemapper. Measurements exclude
pixel readback and warm-up, but explicitly wait for GPU completion each frame.
They are synchronized offscreen CPU+GPU latency, not interactive throughput or
isolated atmosphere/environment-pass GPU costs. A separate native night capture checks the multitool and UI composition during
streaming; it is not a settled-world performance measurement. Active construction
tool effects and fully settled distant terrain still need an interactive review.


## Captured evidence, 2026-09-29

Apple M1 Pro, Metal, development profile with optimized dependencies. The final
offscreen capture ran after the CI workload finished. Each timing row contains
60 frames, after warm-up, with readback disabled while measuring.

| Scene | Mean frame ms | p95 frame ms |
| --- | ---: | ---: |
| Previous fixed sky configuration | 16.85 | 22.40 |
| Sunrise | 23.19 | 26.87 |
| Noon | 26.46 | 32.07 |
| Sunset | 24.89 | 28.30 |
| Midnight | 25.24 | 28.32 |

The complete sky/lighting change adds approximately 6.34–9.61 ms to synchronized
frame latency in this fixture. The atmosphere and environment generator use
passes that the current app instrumentation does not isolate, so these numbers
must not be presented as separate GPU pass costs or expected player frame times.

- Fixed view: [sunrise](atmosphere/2026-09-29/sunrise.png), [noon](atmosphere/2026-09-29/noon.png), [sunset](atmosphere/2026-09-29/sunset.png), [midnight](atmosphere/2026-09-29/midnight.png).
- Alignment: [sunrise disk](atmosphere/2026-09-29/sunrise-disk.png), [noon disk](atmosphere/2026-09-29/noon-disk.png), [full moon](atmosphere/2026-09-29/midnight-disk.png).
- [Baseline](atmosphere/2026-09-29/baseline.png) and [raw timings](atmosphere/2026-09-29/timings.json).
- [Native midnight UI check](atmosphere/2026-09-29/native-midnight.png): 4112 × 2524, disposable copy of a saved world, still streaming distant terrain. This preceded the final star/moon brightness adjustment. The source player manifest's hash remained unchanged; the copy retained 09:00 in its save while rendering midnight.

The captures show blue daytime skies, warm twilight, readable night surfaces,
changing water/metal reflections, and sun/moon disk alignment. Garage environment
restoration and overlay-camera exclusion are additionally covered by ECS tests.

Verification completed:

- Eight focused sky tests passed, including the queued static-generator race.
- Solar persistence, invalid-time, time-only save, and developer-gating tests passed.
- The full app suite in CI passed 938 tests (10 opt-in tests ignored).
- The opt-in production-material GPU capture passed on Metal.
- Consistency checks passed. The final formatting check reported only concurrently edited water tests.

`cargo xtask ci` completed but is **not green** in this shared working tree:
11 GPU physics tests, one CPU stacking test, and three water tests failed. Its
lint errors were in water/world-generation tests, and documentation failed on the
water `step` method and an unresolved `Self::weights` link. Formatting also caught
concurrent edits in a water test.
No water or physics behavior was changed as part of the sky implementation.

### First-person lighting regression

The tool camera shares the main view's filtered environment, exposure, ambient
light, and atmospheric transmittance LUT. Its garage key light is off outdoors.
On Apple M1 Pro / Metal, the two-camera regression produced identical surface
RGB values in the world and tool views: high sun `(164,154,138)`, low sun
`(241,117,16)`, four-lux illumination `(67,62,55)`, and a below-horizon sun
`(0,0,0)`. The tool target's background remained transparent. The test uses the
same tonemapper as the app and allows three channel levels of aerial-perspective
difference. Unit tests cover repeated garage visits and changing exposure/maps.

The regression is `tool_and_world_receive_matching_atmospheric_light` (ignored
by default; run with `--ignored --nocapture`). Nineteen focused app tests passed;
the GPU comparison passed in an isolated harness importing the production module.
The native updated app captured [the moonlit hand tool](atmosphere/2026-09-29/tool-midnight.png) with the
HUD intact. [GPU test output](atmosphere/2026-09-29/tool-lighting.txt) records the
adapter and pixel comparisons. This capture includes terrain streaming and
concurrent compilation; its frame timings are not performance measurements.


### Performance follow-up

The renderer has two small opt-in controls in
[`bevy_pbr`](../vendor/bevy_pbr/MECHANIC-PATCH.md): retain environment textures
between generation/filtering updates, and reuse directional shadows for a
co-located overlay camera. The default atmosphere quality, cubemap resolution,
MSAA, and main-camera shadows are unchanged. Timing spans now label atmosphere
LUT, sky, and environment passes for subsequent profiling.

A paired Metal benchmark on Apple M1 Pro rendered the same settled scene at
768 × 512 with a metallic foreground object on the tool layer. Each variant ran
three times, reversing order in the middle round, with 60 measured frames per
sample and no pixel readback while timing. Values below average the three means
and the three per-run p95 values (the latter is not a pooled p95).

| Configuration | Mean ms | Average per-run p95 ms | Tool shadow views |
| --- | ---: | ---: | ---: |
| Every-frame reflections, independent tool shadows | 44.92 | 49.16 | 4 |
| Reflection cadence only | 38.69 | 46.70 | 4 |
| Shared tool shadows only | 41.40 | 46.94 | 0 |
| Both optimizations | 35.91 | 44.04 | 0 |

The combined change reduces synchronized frame latency by **20.1%** and the
average per-run p95 by **10.4%**. These are CPU+GPU offscreen latencies, not
interactive FPS or isolated pass costs. This fixture includes an extra camera
and foreground object, so its absolute times cannot be compared directly to the
earlier single-camera fixture. The benchmark ran separately from this task’s
builds.

[Raw samples](atmosphere/2026-09-29/optimization/optimization.json) and
[benchmark output](atmosphere/2026-09-29/optimization/benchmark.txt) retain the
configuration, adapter, and individual runs. Compare the
[before](atmosphere/2026-09-29/optimization/optimization-before.png) and
[combined](atmosphere/2026-09-29/optimization/optimization-combined.png) captures.
The sky, matte sphere, metal sphere, and foreground-object sample regions are
pixel-identical across variants; changed pixels lie entirely within the animated
water band. [Pixel comparisons](atmosphere/2026-09-29/optimization/pixel-regions.json)
record the regions and differences. Sunrise, noon, sunset, and midnight captures
are saved beside the benchmark.

Regression coverage includes cadence, shader warm-up, time jumps, altitude,
reentry, matching atmospheric lighting, world-object shadow occlusion, and
independent-shadow fallback when an overlay camera moves. The
[GPU roof test](atmosphere/2026-09-29/optimization/shadow-regression.txt)
measured foreground red-channel values of 0 under an offscreen world-layer roof
and 164 after removing it, matching the world camera in both cases. The sky continues drawing every frame; only its
slowly changing reflection environment is updated less frequently.


Follow-up verification: the app build passed, all **944 app tests** and **238
world tests** passed, and the Metal lighting/roof/fallback regression and paired
benchmark passed. `cargo xtask ci` completed: consistency and Python checks passed,
but CI remains red on the unrelated water-material test lint
(`generation/tests/water_materials.rs:239`), 11 GPU physics tests, one CPU physics
test, and the broken `Self::weights` documentation link in world generation.
These failures also existed before the rendering optimization. Formatting passed
at CI start; a subsequent check encountered newly edited, unrelated water
benchmark files. Those files were left untouched.
