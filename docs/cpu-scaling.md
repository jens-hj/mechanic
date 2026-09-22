# CPU construction scaling

The target remains **2 ms p95 for a complete CPU tick at both 10× builder workloads**, 60 TPS without dropped/degraded ticks or GPU fallback, and a separate 8.33 ms foreground frame p95. The optimizations here do not by themselves establish those gates.

## Reproduction

`crates/mechanic-bench/tests/fixtures/builder-world` preserves generation 20 of the builder world: its procedural terrain seed and generator version, construction, root pose, Dimension Link identity, and Garage document. `fixture.json` contains source hashes. The creation has 13 bodies, 18 generalized velocities and 1,744 authored colliders. No player save is opened for writing.

```
cargo run --release -p mechanic-bench --bin cpu-physics -- \
  --scenario builder-scale --copies 10 --warmup 600 --ticks 3600
```

Add `--connected` to rigidly link the first chassis cuboid of every copy. Copies are eight metres apart; links add no collision geometry, and suspension coordinates remain active. A downward off-centre impulse is applied every 120 ticks. `--hold` exercises whole-assembly hold/release separately. `--warmup 0 --ticks 120` records cold impacts. `--floor` is a diagnostic finite floor, **not the saved-terrain workload**.

The headless replay meshes the saved seed at leaf resolution around the starting assembly with a four-metre guard region. It reports escape from that region. It does not reproduce the app's changing terrain residency or renderer. Its publication measurement converts body poses and COM velocities to renderer precision; app publication and terrain-update times have separate capture events.

For repeated matched binaries:

```
python3 scripts/run-cpu-scale-benchmark.py \
  --baseline /path/to/baseline --candidate /path/to/candidate \
  --output /path/to/new-results
```

The default is three repetitions at 1×, 2×, 5× and 10×, each separated and connected, with 600 warm-up and 3,600 measured ticks. The runner alternates binary order, records executable hashes and commands, and retains raw JSONL even on failure. Headless throughput is not proof of the app scheduler maintaining 60 TPS.

`run-background-capture.py --physics cpu` explicitly selects CPU physics in a disposable copy of the fixture. Capture events identify actual CPU completions; a mixed GPU/CPU run is rejected by the launcher. Foreground runs retain the launcher's existing source/build identity and unchanged-render-settings checks.

## Runtime changes

* Body bounds are traversed before immutable collider trees. Same-body, static/static and wholly exempt pairs never descend into collider pairs; jointed bodies descend and drop the collider pairs they were built touching. Exact transformed vertex bounds precede convex transformation. Terrain candidates descend through body/collider bounds, and individual triangle bounds reject false positives from batched BVH leaves before clipping.
* A topology-owned cache keys construction geometry by exact body poses. Pose changes invalidate affected shapes and separation results. Geometry is in simulation coordinates; terrain publication and floating-origin offsets are evaluated afresh, so terrain-dependent results are not cached. Proximity/recovery share shapes and margin-aware pair separation results.
* Continuous broadphase intersects the original origin/radius bound with the initial convex bounds expanded by the integrated point-speed bound. Both enclose the entire motion, including full turns. Narrowphase, activation distances, contact ordering and authored pipe openings remain unchanged.
* Face separation can reject a pair before edge-axis work when it proves separation beyond the query margin. Retained results use the reference feature-selection algorithm.
* Finite-triangle clipping shares each vertex's plane distance across its two edges and leaves wholly interior polygons in place. Proximity queries retain the last extrusion when source geometry, normal and margin match; finite triangles are still clipped independently, preserving holes and edge contacts.
* `MachineKinematics` reconstructs poses and ancestor-path Jacobians without constructing a generalized mass matrix or imposing the dense reference's 512-velocity cap. The dense implementation remains available for comparisons. The articulated factor accepts reconstructed poses directly.
* Contact Jacobian clearing, response preparation, scratch clearing and sparse row storage are restricted to the affected compiled component ranges. Factor traversals skip unrelated components. Solver rows store only nonzero Jacobian/response entries. Cross-component contacts retain both responses. Factor RHS scratch is reused across rows; collision traversal, transformed shapes, and tick candidate/rollback buffers are reused.

The runtime deliberately retains reference arithmetic order in sparse Jacobian projections: changing to a differently ordered force projection made the existing car-wall regression exceed its unchanged 5 cm tolerance by 12 micrometres. This is **not yet a linear-storage kinematics implementation for arbitrarily deep trees**. Contact-row and articulated-factor arenas now retain capacity across substeps/ticks. `solver_scratch_bytes` and `solver_scratch_growth_bytes` measure those arenas only; they are not total allocator statistics. Finite-triangle clipping polygons and extrusion geometry also retain capacity across terrain queries. Closure/drive rows, kinematics and returned contact vectors still have temporary allocations.

## Evidence and limits

Raw measurements and build/source identities are under `performance-results/2026-09-14-cpu-scale`. The initial 27-block baseline was 14.98 ms p95. Intermediate matrix-free/component-local runs reduced this to 5.98 ms with identical reported penetration and motion.

In an exploratory 60-tick saved-terrain builder replay, tighter continuous bounds reduced the worst tick from 2,472 ms to 33.5 ms and p95 from 37.8 ms to 15.8 ms. State hashes were identical between those two runs. These are cold diagnostic samples, not the prescribed warmed 10× acceptance runs. The 2 ms goal is not established by them.

Full commands, raw measurements, incomplete diagnostics and the current verification status are documented in [the evidence report](performance-results/2026-09-14-cpu-scale/REPORT.md).

The hardware-enabled app suite passed 765 tests with six ignored and one failure in the unchanged `ui::suspension::tests::leader_geometry_follows_projected_positions_without_remounting` test. The initial sandboxed run also lacked access to a Metal adapter. Physics-specific dense comparison, large runtime, hold/release and existing quality tests passed before the final broadphase verification pass. See the recorded verification logs for the final status.
