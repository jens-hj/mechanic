# Tool effects verification — 2026-09-06

Apple M1 Pro, Metal, 2560 × 1440 physical pixels, 75° FOV, MSAA 4×.
The debug fixture ran with vsync disabled after compilation and the physics test
suite completed. No shader or pipeline validation errors were logged.

All 14 screenshots (six effects plus baseline, dark and bright) are archived at
[`/tmp/mechanic-fx-captures.zip`](/tmp/mechanic-fx-captures.zip). Individual PNGs are
under `/tmp/mechanic-fx-final/`. The [fixture instructions](README.md#reproduction)
reproduce the captures; [raw measurements](measurements.json) retain sample counts
and p95 values.

| Fixture | Dark mean / p95 (ms) | Bright mean / p95 (ms) | Peak particle draws | Bloom passes |
|---|---:|---:|---:|---:|
| baseline | 9.41 / 17.13 | 9.65 / 16.40 | 0 | 0 |
| sledge | 9.61 / 13.04 | 9.50 / 12.82 | 2 | 16 |
| matter | 11.47 / 16.40 | 10.01 / 11.59 | 2 | 16 |
| welder | 9.51 / 12.05 | 9.75 / 12.91 | 2 | 16 |
| connector | 9.36 / 12.48 | 9.73 / 12.01 | 1 | 16 |
| freeze | 9.67 / 11.55 | 9.81 / 13.50 | 3 | 16 |
| lift_edit | 9.85 / 11.64 | 10.46 / 12.80 | 3 | 16 |

These are application frame intervals, not GPU-only durations. The baseline
disables effects and bloom while retaining HDR composition. Different tools also
change viewmodel geometry; small timing differences cannot isolate FX cost.
This is visual-fixture evidence, not a performance or scale-gate claim.

Visual inspection confirmed distinct amber, mint, red/warm, cyan, and violet
effects; hard shard edges; target occlusion; visible outer dashes and faint inner
bounds; readable aperture plates; and sharp, unbloomed UI/viewmodel edges. The
bright-background traces are deliberately faint. The lift/edit fixture shows
violet wakes alongside Matter feedback while its target moves. It supplies
synthetic hold and gesture frames rather than exercising a saved-world input
sequence; state acceptance, cancellation, restoration, and motion behavior are
covered by the existing and new tests.

Final checks: 683 app tests passed, four hardware tests ignored; 13 focused FX
tests passed; the non-GPU workspace suite passed; Clippy with warnings denied and
formatting passed. Metal GPU tests: 72 passed, 11 inherited failures. The exact
failure set matches the pre-existing weld-baseline report, recorded in
[gpu-test-failures.json](gpu-test-failures.json). No core/GPU source changed.

## Rotating connector plate streams

The connector now retains its central stream and adds one from each of the six
animated tip lamps. Origins use propagated world transforms from the active tool
end. All seven streams converge at the current endpoint, rebuilding 81 segments
within the existing single trace batch.

Verification: 685 app tests passed (four hardware tests ignored), including 15
focused FX tests. New regressions cover all six moving origins, endpoint tracking,
no beam accumulation, cancellation, active-end switching, and removed emitters.
Workspace Clippy with warnings denied and formatting checks passed.

Updated visual capture was attempted twice on Apple M1 Pro / Metal, but both runs
stopped before screenshots: Metal could not allocate a counter sample buffer and
Bevy reported device loss. Logs: `/tmp/mechanic-fx-plates.log` and
`/tmp/mechanic-fx-plates-retry.log`. The screenshots and measurements above predate
this plate-stream addition; visual verification of the addition remains pending.
