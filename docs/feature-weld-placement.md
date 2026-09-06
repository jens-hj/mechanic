# Feature weld placement

The first selected creation moves in its authored internal arrangement onto the
second creation's current pose. Original source physics continues during preview.
The endpoint is validated; travel is not animated.

## Interaction and geometry

Select a flat face, straight logical edge, or real corner, then press a destination
feature, hold to drag, and release to commit. The approached adjacent face fixes
the mating plane. Escape, secondary action, and changing tools cancel. Invalid
release keeps the source selection. Green/red ghosts indicate endpoint validity;
feature outlines identify the selected surfaces. Ghosts include authored joint
visuals and never use live source articulation.

Vertex pairs cannot slide. Vertex–edge pairs slide on the finite edge; edge pairs
stay collinear. Pairs involving faces slide in the mating plane, preserving feature
incidence in actual material. Initial hover placement aligns the two face-local grid directions and snaps their
relative lattice offset at 25 cm, or 5 cm with Shift. Weld selection uses the shape
tool’s solid edge bars and corner markers in the welder’s red accent color. Drag displacement
snaps relative to that starting alignment; changing increments during dragging
preserves stationary position.
R advances through 15-degree orientations, skipping incompatible incidence (edge
pairs generally permit half-turns). Obstructed compatible orientations remain invalid.

`WeldPick` resolves evaluated topology only at its captured graph revision. Curved
chains and tessellation seams are excluded; flat cylinder ends remain selectable.
`WeldAlignment` owns the feature constraints, and `WeldSnap` owns snap phase.
`WeldRejection` documents geometry rejection reasons. App feedback also reports
stale publication, terrain, bounds, and obstruction failures.

The selected rigid bodies' coplanar material must contain a continuous 5 × 5 cm
square in destination tangent axes. Convex material cells are unioned by coverage;
holes, gaps, narrow strips, line contact, and disconnected area totals cannot pass.
`WeldPlacement` reframes the source's entire structural component, validates the
combined authored source/destination arrangement, and creates a face weld. Runtime
validation separately checks all source default bodies against current destination
and unrelated bodies, terrain, and Garage bounds, allowing 1 mm penetration tolerance.
No edge-contact rigid-link fallback is used.

## Publication and history

`weld_publication::Intent` retains the graph revision, feature picks, and one
source-to-destination authored transform across asynchronous preparation. The graph
and history remain unchanged until a private replacement passes final authoritative
snapshot validation. Scene/foundation changes or cancellation discard preparation.

The explicit placement state-transfer path initializes every source body from its
placed compiled default. Source joint coordinates and relative velocities reset to
zero (the compiler's authored reference). All source bodies inherit the destination
body's rigid velocity field. Destination articulation and unrelated state use the
ordinary pose-preserving transfer path. Normal controls resume after installation.

Destination hold state controls the resulting assembly. A live destination releases
a held source, while a held destination holds the combined creation. Dimension-link
parts remain. Graph, GPU state, holds, socket positions, and one history entry publish
together. History captures only affected assemblies, restores their placement and
hold state, and preserves unrelated bodies' current motion. Saved construction uses
the existing frame and joint formats without migration.

Bodies connected through joints use the in-place path, which validates both authored
and current contact and rejects a merge that would move existing body poses. Same-body
welds are rejected. Terrain-anchored sources cannot relocate. Joint-lock warnings are
retained for in-place loops.

## Verification

`cargo test --workspace --exclude mechanic-gpu` passed 999 tests (four ignored).
`cargo fmt --all -- --check`, Clippy with warnings denied, and `git diff --check`
passed.

Colocated core and app tests cover feature-pair constraints, snap phase, contact
coverage, evaluated cylinder ends, authored articulation, live obstruction, moving
destination dragging, cancellation, stale publication, history, and hold transitions.
The ignored `real_gpu_weld_publication_starts_from_default_source_coordinates` test
runs the actual replacement and a GPU tick; it passed on Apple M1 Pro / Metal using:

```sh
cargo test -p mechanic-app real_gpu_weld_publication_starts_from_default_source_coordinates -- --ignored --nocapture
```

The final `cargo test --workspace -- --test-threads=1` run passed the app and core
suites, then reported 72 passes and eleven failures in `mechanic-gpu`. The failing
dynamics tests are listed in
[live construction editing](live-construction-editing.md#remaining-gpu-failures).
Native ghost tracking and mouse drag/release have automated handler coverage but
still require a manual visual walkthrough.
