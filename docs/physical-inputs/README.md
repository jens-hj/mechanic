# Rotary Dial and Pushbutton

See [implementation status](IMPLEMENTATION.md) for completed foundations and remaining work.

The archived [browser concept](concept/index.html) defines the six approved models.
Run `python3 -m http.server 8765 --directory docs/physical-inputs/concept` and
open `http://127.0.0.1:8765`. Source and inspection images are preserved unchanged.
The concept is a visual reference, not a completed game implementation. The mesh
exporter corrects the 10 cm and 25 cm spring/guide-rod tops to terminate below
actual cap undersides; the archived source remains unchanged.

## Approved behavior (supersedes concept notes)

Sizes are 5, 10, and 25 cm mounting footprints, with independently authored heights
and mechanics. Default size is 5 cm; default button mode is momentary. Existing
seat-routed Input is named **Keyboard Input**.

Each physical input connects to one controller independently of welds and seats.
A dial may bind multiple targets in that controller. Reconnecting clears dial mappings;
a button retains its configured key and loses its transient held source.
Each numeric parameter accepts one dial. Mappings use native-unit endpoints and
inversion. All editable numeric controller settings are eligible, including
integer settings. Controller edits and dial batches share validation; coupled
invalid batches fail atomically. Reserve actuator capacity from possible mapped
contributions. Manual edits remain available.

The assignment editor labels its chooser **Dial to assign** and highlights the
selected dial. A sole connected dial is preselected for new assignments. With
several dials, choose one explicitly; Apply stays dimmed and inactive until a
connected dial is selected. Connecting or naming a dial alone does not save a
parameter assignment.

Authored values live in the graph. Simulation uses separate effective values,
shared by snapshots, sequencing, drives, and feedback. Reset restores authored
values. No dial position or button-on flag is saved. Unlink retains the current
value in its authored/runtime layer.

Dial feedback inverse-maps requested controller values. Agreement gives Uniform;
disagreement gives Mixed. Mixed retains the last meaningful pointer (initially
centered), shows an interrupted amber scale, and opens a target chooser before
operation. Choosing a starting target changes no values. Subsequent dragging
writes every mapping atomically. Dwell uses effective duration against elapsed
time, allowing a changed duration to transition on the next tick.

Buttons are controller-local keys, configured with the Connector overlay even
before connection. Key capture accepts A–Z and 0–9, normalizes uppercase, consumes
keyboard input, and supports Clear and Escape cancellation. The configured key is
the cap label; new buttons are blank. No decorative label or direct state binding
is saved. Overlay clicks take precedence over wiring; part clicks keep wiring.

Momentary buttons hold their key during Interact; toggle buttons latch their own
source on each press. Seated Keyboard Input and button sources combine per
controller/key, with logical edges only when that combined held state changes.
State triggers and matching unmodified gearbox bindings receive those edges;
virtual keys never enter global gameplay input. Releasing one source cannot
release another. All buttons sharing a controller/key have the same cap and light
feedback. State transitions never create key input or light feedback. Disconnect,
delete, key changes, and reset clear a button's source; toggle latches are transient.

On foot or seated, aim within the existing 3 m eye-based reach and use rebindable
Interact. Targeted controls consume Interact before seat exit. Picking respects
occlusion and modest minimum targets. Hold and drag horizontally for dials; Shift
is fine adjustment. Restore mouse-look and release momentary sources on end,
cancel, focus loss, or reach loss. Pause/freeze prevents commands and advancement.
Unconnected controls say “Not connected” and cannot control joints.

Configuration, links, and bindings participate in documents, duplication,
undo/redo, deletion, and reference remapping. Deleted/incompatible references are
removed; surviving state and gear references are remapped. Replace document format
directly without compatibility readers.

Use core-owned geometry/material legends and true sub-block envelopes throughout
placement, snapping, welds, bounds, picking, mass, and collision. Cache geometry
by kind/size and springs in three poses. Animate transforms, visibility, and
emission without rebuilding geometry. Labels show only the optional configured
ASCII key, extruded and cached by key/font/size, and emissive while the key is held.
Mosaic must expose reusable positioned glyph outlines upstream, with bitmap
silhouettes and unavailable-glyph reporting; do not duplicate text shaping here.

## Delivery gates

1. Graph, six authored models, actual envelopes, persistence and history.
2. Controller mappings, runtime values, feedback, capacity, and source edges.
3. Connector, picker, bindings editor, on-foot/seated interaction.
4. Upstream text outlines, extruded labels, native visual inspection and captures.

Focused regression coverage must exercise geometry clearances, inverse mappings,
integer/coupled validation, reset/capacity, held-source combinations, mixed
dial selection, dwell feedback, interaction cancellation, remapping, and transient
value exclusion. Exercise angular and linear CPU/GPU drives without topology
rebuilds. Run `cargo xtask ci`; report hardware-dependent checks not performed.
Audio and multiplayer arbitration are outside scope.
