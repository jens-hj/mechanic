# Performance and investigation results

The dated checkpoints that used to live here (about 150 MB of captures, logs,
source snapshots and reference scripts) were removed from the working tree.
They remain in git history. To restore one:

```sh
dir=2026-09-10-fixed-replay
commit=$(git log -1 --format=%H --diff-filter=D -- "docs/performance-results/$dir")
git checkout "$commit^" -- "docs/performance-results/$dir"
```

| Checkpoint | Topic |
| --- | --- |
| <a id="2026-09-05-background-bottleneck"></a>2026-09-05-background-bottleneck | Remaining waits after the fence backport |
| <a id="2026-09-05-background-runner"></a>2026-09-05-background-runner | Background runner verification |
| <a id="2026-09-05-callback-terrain-attribution"></a>2026-09-05-callback-terrain-attribution | Callback and terrain attribution |
| <a id="2026-09-05-terrain-single-projection"></a>2026-09-05-terrain-single-projection | Single-projection terrain shader experiment (rejected) |
| <a id="2026-09-05-test4-fence"></a>2026-09-05-test4-fence | TEST4 fence-backport comparison |
| <a id="2026-09-07-dense-terrain-contacts"></a>2026-09-07-dense-terrain-contacts | Dense terrain contact correctness |
| <a id="2026-09-07-generated-terrain-contacts"></a>2026-09-07-generated-terrain-contacts | Production-mesher contact correctness |
| <a id="2026-09-07-rotational-terrain-contact"></a>2026-09-07-rotational-terrain-contact | Rotational terrain contact correctness |
| <a id="2026-09-07-suspension-world"></a>2026-09-07-suspension-world | Suspension car in generated terrain |
| <a id="2026-09-07-terrain-contact-baseline"></a>2026-09-07-terrain-contact-baseline | Terrain impact baseline |
| <a id="2026-09-07-terrain-contact-recovery"></a>2026-09-07-terrain-contact-recovery | Terrain impact recovery |
| <a id="2026-09-08-blob-physics"></a>2026-09-08-blob-physics | BLOB physics optimization |
| <a id="2026-09-10-articulated-contact-steps"></a>2026-09-10-articulated-contact-steps | Articulated CCD and coupled surface response |
| <a id="2026-09-10-cold-settling"></a>2026-09-10-cold-settling | Cold drop, timed stops and contact continuity |
| <a id="2026-09-10-compiled-dynamics"></a>2026-09-10-compiled-dynamics | Frozen baseline and compiled response experiment |
| <a id="2026-09-10-contact-release"></a>2026-09-10-contact-release | Contact release and return |
| <a id="2026-09-10-cpu-free-motion"></a>2026-09-10-cpu-free-motion | CPU free-motion foundation |
| <a id="2026-09-10-cpu-joint-ticks"></a>2026-09-10-cpu-joint-ticks | CPU joint ticks and finite terrain collision |
| <a id="2026-09-10-cpu-surface-substeps"></a>2026-09-10-cpu-surface-substeps | Surface substeps and solver cost |
| <a id="2026-09-10-fixed-replay"></a>2026-09-10-fixed-replay | Fixed-duration foreground car replay |
| <a id="2026-09-10-foreground-baseline"></a>2026-09-10-foreground-baseline | Source-matched foreground car baseline |
| <a id="2026-09-10-impact-activation"></a>2026-09-10-impact-activation | Initial impact activation and sustained supported car |
| <a id="2026-09-10-impact-events"></a>2026-09-10-impact-events | Bounded first-impact events |
| <a id="2026-09-10-linear-ccd"></a>2026-09-10-linear-ccd | Exact linear CCD and below-surface separation |
| <a id="2026-09-10-release-cache-repair"></a>2026-09-10-release-cache-repair | Release cache repair |
| <a id="2026-09-10-supported-paths"></a>2026-09-10-supported-paths | Finite supported-path validation |
| <a id="2026-09-10-supported-ticks"></a>2026-09-10-supported-ticks | Unpublished supported ticks and impact holds |
| <a id="2026-09-11-articulated-factor"></a>2026-09-11-articulated-factor | Articulated inertia and analytic contact-block factors |
| <a id="2026-09-11-body-frustum"></a>2026-09-11-body-frustum | Immutable simulation mesh visibility |
| <a id="2026-09-11-contact-continuity"></a>2026-09-11-contact-continuity | Contact convergence and release localization |
| <a id="2026-09-11-contact-kinematics"></a>2026-09-11-contact-kinematics | Contact kinematics and activation |
| <a id="2026-09-11-coupled-search"></a>2026-09-11-coupled-search | Coupled contact search |
| <a id="2026-09-11-cpu-solver-repair"></a>2026-09-11-cpu-solver-repair | CPU contact solver repair |
| <a id="2026-09-11-event-search-controls"></a>2026-09-11-event-search-controls | Rejected cold-contact event controls |
| <a id="2026-09-11-retry-reuse"></a>2026-09-11-retry-reuse | Retry diagnostics and valid reuse |
