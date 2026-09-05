# Remaining waits after the fence backport

The fresh background diagnostic identifies a remaining render-world handoff
wait, with substantial opaque GPU cost and a saturated physics readback ring.
It does not establish an exclusive GPU bottleneck or demonstrate an optimization.

## Conditions and provenance

Apple M1 Pro / Metal, current release binary, disposable copy of current TEST4,
saved stationary player pose, car behind the player, held tool visible. All
1,676 CPU frames were unfocused, 4112×2524, 4× MSAA, baseline materials, F3 open,
AutoNoVsync, zero streaming backlog. All 1,677 render GPU samples were Complete;
three originated before the recording interval. Physics ticks were consecutive,
failure flags zero. Capture ran 17:38:56–17:39:56 UTC on 2026-09-05 after idle
streaming and 15 seconds of warm-up. App saved its screenshot and exited normally.
The temporary world was removed and the source manifest hash remained unchanged.

The inspected screenshot shows **20 bodies / 2,083 colliders**, versus 17 / 2,069
in the earlier background verification. The saved scene has changed: this is
not a matched comparison with that run or the foreground A/B/B/A experiment.
No app, shader, quality, solver or scheduling changes were made in this investigation.

`run.json` records the world manifest fingerprint; `fingerprints.json.gz` records
the binary and current source/assets/dependencies after the run. The executable
was the existing release build, not rebuilt for this investigation. The raw
capture and two native CPU profiles are gzip-compressed here. The in-app PNG is
at `/tmp/mechanic-background-bottleneck-01/capture-1788629996034469000-50241.png`.
`capture-command.py` preserves the one-shot collection command, including native
`sample PID 5 1 -file PATH` invocations during the recording interval. Profiler
overhead and other desktop activity are uncontrolled; numbers are diagnostic.

## Raw distributions

| Measurement | p50 ms | p95 ms |
| --- | ---: | ---: |
| CPU frame | 35.189 | 46.111 |
| Physics queue submission | 0.160 | 0.696 |
| Physics command finalization | 1.159 | 3.407 |
| Physics measured GPU stages | 7.209 | 9.373 |
| Submission to observed readback | 96.867 | 104.261 |
| Window acquisition schedule span | 26.396 | 39.003 |
| Render/present CPU schedule span | 4.333 | 13.727 |
| Opaque GPU pass | 20.715 | 22.421 |
| Tracked GPU span (partial coverage) | 25.479 | 28.687 |
| Render GPU sample observation age | 104.916 | 118.987 |

Throughput: 27.95 FPS / 31.43 completed TPS; backlog 1,623→3,327. All frame
observations had three physics ticks in flight. These are separate distributions:
GPU spans may overlap and must not be summed. Sample age is not GPU execution time.

## Mechanism and limits

Both native CPU samples contain
`prepare_windows → get_current_texture → acquire_texture → CAMetalLayer nextDrawable
→ dispatch semaphore wait`. The main thread also parks inside
`renderer_extract → RenderAppChannels` while waiting for the render world:
552/1,557 main-thread samples in the first profile and 644/1,685 in the second
are in the condition-variable wait at that stack. These are sampled stack counts,
not exact durations or CPU utilization.

Source confirms the dependency: `vendor/bevy_render/src/pipelined_rendering.rs`
`renderer_extract` awaits the render-world channel before extraction. App physics
readbacks are polled during Update in `poll_simulation_readbacks`, and submission
in `advance_simulation` is limited by available slots. The fence fix permits
concurrent submission during drawable acquisition, but does not make the next
main Update independent of rendering. Thus a slow render/presentation cycle can
delay completion observation and slot reuse even without submission lock stalls.

Ranked interpretation:

1. **Established:** expensive opaque pass, drawable wait, render-world handoff
   wait, and full physics ring coexist. The remaining measured wait is not
   explained by a large physics queue-submission timer.
2. **Supported hypothesis:** shared GPU/render-presentation pressure plus
   frame-paced completion observation limits physics throughput. Opaque rendering
   is the largest measured render group and the next performance target.
3. **Unresolved attribution:** the 97 ms readback latency includes queue residence,
   GPU work, callback servicing and time until app polling. Current observations
   cannot split these. Nor do opaque timestamps isolate terrain from other opaque
   draws or distinguish fragment work from bandwidth. Do not subtract independent
   medians to invent a queue wait, or divide ring size by median latency as a proof.

`xcrun xctrace list templates` exited 137 with no output both in the sandbox and
with approved native access, repeating the earlier tooling limitation. No Metal
System Trace was obtained. A synchronized GPU timeline remains unavailable.

## Next bounded experiment

Before selecting a runtime change, add opt-in timing of physics mapping callbacks
and the following app observation, plus successful/empty poll events. This can
separate callback-to-publication delay from pre-callback latency without extra
polling, waits, slots or scheduling changes. Callbacks themselves require device
servicing, so this still cannot label pre-callback latency pure GPU time. Pair
that with terrain-specific opaque-pass attribution or a working Metal timeline
before choosing a quality-preserving render optimization. Keep the current scene
frozen for the next matched candidate comparison.

The summary tool now exposes individual GPU groups, CPU submission stages,
render sample age, slot occupancy percentiles and focused-frame counts. Five
summary regressions and the disposable-world cleanup regression pass;
`git diff --check` passes. No runtime code changed, so workspace/GPU suites were
not rerun. Earlier correctness failures and foreground acceptance gates remain open.
