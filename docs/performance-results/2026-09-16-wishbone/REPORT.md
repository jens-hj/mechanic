# Wishbone assembly restoration and distant-pipe investigation

## Restoration

The creation menu silently omitted `double-wishbone-2.mech` because deserialization failed at line 861: `PipeBend` was missing `span_blocks`. Its older `radius_units` field described a different bend geometry. Simply renaming the field produced invalid welds (`FacesDoNotTouch`); replacing bend welds with rigid memberships still left invalid bearing anchors. Neither attempted conversion was installed.

Recovered the intact structural assembly from **builder generation 171**, immediately before the assembly disappears in generation 172. Used `ConstructionGraph::structural_component` and `partition`, retained its unattached sockets, and serialized with `CreationDocument::from_graph`. The recovered document successfully deserializes, rebuilds its graph, and compiles: **105 parts, 16 bearings, 13 bodies, 1,763 colliders**. Its 142 welds remain welds; no recovery rigid links were introduced.

Installed the recovered document at:

`/Users/jens/Library/Application Support/Mechanic/creations/double-wishbone-2.mech`

Display name: **double wishbone 2**. Closing and reopening P refreshes the menu. The recovered file is the working in-world version, not a geometrically identical conversion of the obsolete library file: its bends use current geometry and four straight pipe lengths differ from that older file. Those differences already existed in builder. No live world was changed.

Installed SHA256: `1721f10e2d1805e5fd9053e0473a9b8bae1f5b8e7189f21fb18e53b708f5dbe2`.

The original unreadable file and validated recovered file are retained alongside this report. `builder-instance-171.ron` is the recovery source; generation 172 is the after-removal comparison.

## Controlled measurements

Release `cpu-physics --scenario pipe-scene`, same pipe from new generation 8 positioned 20 m away on X, 120 settling ticks, 600 measured ticks per initial speed, three runs with reversed scene order on alternate runs. `run.py`, `manifest.json`, raw JSONL and `summary.json` provide reproduction and results. No compilation or profiler ran during these comparisons.

Median of three tick p50s; milliseconds:

| Scene | Initial pipe speed 0 rad/s | Initial pipe speed 40 rad/s | Fast p95 |
|---|---:|---:|---:|
| Pipe and scattered blocks from new | 0.039 | 0.270 | 0.285 |
| Same + recovered wishbone assembly | 1.638 | 2.554 | 2.832 |
| Same + builder generation 171, containing wishbone | 1.902 | 3.065 | 3.368 |
| Same + builder generation 172, after wishbone removal | 0.133 | 0.881 | 0.955 |

The isolated wishbone addition is a controlled construction comparison. The builder comparison uses historical whole-world snapshots. Initial speeds 1 and 10 rad/s are also recorded. Gravity remains active; the initial zero-speed pipe later oscillates, so the column is not a permanently stationary scene.

The wishbone is fully dynamic: the pipe-only and pipe-plus-wishbone scenarios both have exactly five static bodies / 229 static colliders. No wishbone body was anchored by the benchmark's foundation rule. All four fast cases end with identical pipe velocity, 38.2332845312282 rad/s. Every measured case has zero degraded ticks; repeat runs have identical state hashes within each scenario and speed.

## Attribution

For the pipe plus wishbone at initial 40 rad/s, average measured work per tick:

- Contact queries: **1.277 ms**.
- Continuous collision: **0.970 ms**.
- Dynamics: **0.022 ms**.
- Constraint solve: **0.039 ms**.

These counters identify collision processing as the dominant cost, not the double-wishbone constraint solve. They are instrumentation totals averaged across ticks, whereas the table reports timing percentiles.

The fast pipe causes 2,400 sweeps and 1,800 contact re-queries over 600 ticks. Sweeps have **zero body-pair candidates**; the remaining sweep candidates are terrain. The pipe cannot collide with the distant car in these runs. Nonetheless the car adds substantial collision work:

1. `soft_step/solve.rs::continuous_fraction` starts `sweep_new_contacts` across the scene when **any** collider exceeds the 5 cm substep threshold. Candidate pruning occurs after motion and bounds preparation.
2. `soft_step/mod.rs` accumulates travel and rotation globally. A fast pipe invalidates the contact query for the whole machine, including distant car contacts.
3. Empty-contact reuse requires the **whole contact list** to be empty. The car's ground/internal contacts prevent this fast path even when the pipe's own surrounding space is clear. Pipe-only fast runs perform 438 re-queries; adding the car raises this to 1,800.
4. The wishbone's modest authored part count expands into 1,763 collision shapes, including eight pipe bends. Repeated scene traversal is consequently significant.

This supports the user's observation: distance prevents actual pipe/car collisions but does not currently isolate their collision-query costs. A next implementation target is conservative per-assembly query reuse / sweep preparation, including contacts and approaching pairs across assemblies. Simply ignoring slow bodies would not establish collision safety.

## Scope and verification

This follow-up restores the library file and extends benchmark output with static-body counts and proximity-query work counters. It makes no further physics or app behavior changes.

- Recovered creation: deserialization, graph replay and suspension-aware compilation passed; installed bytes match validated artifact.
- `cargo build --release -p mechanic-bench --bin cpu-physics`: passed.
- `cargo fmt --all -- --check`: passed.
- `cargo clippy -p mechanic-bench --bin cpu-physics -- -D warnings`: passed.
- All 48 measured scenario/speed/run combinations: zero degraded ticks.

The harness uses authored construction poses and a diagnostic plane at y=4, not builder's actual streamed terrain or saved dynamic state. Initial bearing velocity substitutes for the hammer impulse. No native app hammer replay or FPS comparison was performed. These results establish reproducible CPU amplification and its dominant phase; they do not establish the exact magnitude of the user's on-screen lag or fix the remaining amplification.
