# Physical controller inputs

Buttons store a typed optional key and Momentary/Toggle mode. Connector callouts
capture A–Z/0–9 before player input and shortcuts. The controller-local key merger
combines seated keyboard and physical sources, including unmodified gearbox
bindings. Cap depression, rim emission and extruded key emission use that merged
held state. Transient latches are never serialized.

Dials bind beside numeric controller fields, including grouped drive rows, enabled
travel/dwell, contributions and editable gearbox settings. The assignment editor
uses displayed units, native saved ranges, separate reversal, shared legal bounds
and atomic capacity validation. Selecting/applying a mapping does not operate it.
The Dials overview supports names, assignment lists, live position/Mixed and Locate.

Reachable eye-based Interact handles buttons and horizontal dial dragging before
seat exit. Construction and terrain occlusion, focus, reach, pause and controller
freeze gate operation. Mixed dials offer a keyboard chooser; selecting a target
sets only the starting pointer. Shift provides fine adjustment. Invalid batches
leave every value unchanged and keep the error visible.

`AppSimulation::effective_graph` supplies transient controller numbers to the
sequencer, dwell handling, controller snapshots, input feedback and shared CPU/GPU
drive rows. Authored values remain separate. Manual edits reconcile their own
parameters while preserving other runtime targets. Numeric controller commits can
advance an otherwise-current published revision without recompiling topology.
Reset drops transient values and held sources; mappings remain in creation files.

The saved format directly replaces button state lists/decorative labels with keys;
there is no compatibility reader. Duplication, reference cleanup and history use
the same graph configuration. Stable new dial names start at `Dial 1`.

## Rendering and Mosaic

Six independently authored baked assets retain core-owned material legends and
mesh ownership. The app caches part meshes and extruded key meshes. Mosaic's
`ShapedText::outlines` exposes its existing shaped glyphs as scalable geometry;
the app tessellates/extrudes that output, including glyph holes. The source patch
and the reactive shape-geometry fix are recorded in `vendor/mosaic-text` and
`vendor/mosaic-macros`, each with its upstream revision and licenses.

`MECHANIC_INPUT_CAPTURE_DIR=/tmp/mechanic-inputs cargo run -p mechanic-app`
produces native released and illuminated/depressed showcases at all three sizes.
These prescribed visual poses verify rendering; they do not simulate controller
interaction.

## Verification

Focused coverage includes controller-local key aggregation, toggle cleanup,
button configuration/capture/wiring priority, unmodified gearbox equivalence,
assignment replacement/unlink, unit conversion, mixed feedback, chooser capture,
transient/authored separation, atomic rejection, and unchanged-creation drive-row
updates. Label geometry tests cover all 36 keys at all three sizes and the counter
in O. Persistence and six-input placement/rotation/compile tests cover the saved
contract.

Native screenshots were inspected on Metal, and final app/core, lint, and
consistency results are recorded in [Verification](VERIFICATION.md). Full CI
remains red on ten GPU physics behaviour failures and one CPU stacking failure;
those failures are unresolved. The earlier suspension overlay reactivity failure
was addressed by the Mosaic patch.
