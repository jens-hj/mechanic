# Physical inputs — concept 02

Archived standalone Three.js geometry concept. Its interactive demo retains the
original state-binding and Unicode-label experiments. The native game's current
configuration and interaction contract is described in [Physical controller
inputs](../README.md); the button descriptions below reflect that replacement.

Open http://127.0.0.1:8765. Restart with `python3 -m http.server 8765 --bind 127.0.0.1 --directory /tmp/mechanic-input-concept`.

## Three authored sizes

Sizes describe mounting footprints, not mandatory cubic heights. One scene unit represents 25 cm. Every root transform stays at scale (1, 1, 1); dimensions, mounting method, detail count, travel and protective hardware differ by size.

| Footprint | Dial | Button |
|---|---|---|
| 5 × 5 cm / Panel | Collar mount, 12 shallow flutes, 7 scale marks, fingertip grip | Round cap, recessed bezel, enclosed single plunger, 3 mm travel |
| 10 × 10 cm / Utility | Two mounting ears, 14 deep flutes, 13 marks, open protective bezel | Wide chamfered cap, diagonal side guards, two guided springs, 7.5 mm travel |
| 25 × 25 cm / Industrial | Four feet, 20 scallops, 19 marks, raised spindle, bolted scale guard | Broad cap, continuous mitred tubular guards, two visible springs, 21.25 mm travel |

The Size family view shows all six parts at their actual relative size. The floor grid is 25 cm. Size selectors open individually inspectable, interactive models. Close-up cameras frame the selected size; they do not resize the model.

Sub-block attachment/snap rules remain a design decision for the eventual game integration: the 5 and 10 cm parts need surface placement within the construction grid. This concept defines physical footprints, not new Rust placement behaviour.

## Revised geometry and materials

- Dial ticks point radially inward at every angle, and sit on scale plates. Illumination advances with the dial from empty to full across its 270° sweep.
- The grip is a single closed fluted mesh, with bevelled ends. There are no intersecting tooth boxes or coincident drum surfaces.
- Guard tubes use shared mitre cross sections and diagonal shoulder segments, eliminating disconnected cylinder ends. Dial guards are continuous machined arcs with bolted supports.
- Guide rods terminate below the button's lowest position. Springs compress with the cap; their wire diameter remains fixed. Clearance was checked in both rest and pressed poses for every size.
- Cap depression follows the connected controller's combined held key. Momentary interaction contributes a source until release; Toggle latches its own source. Seated keyboard input and other buttons holding the same key keep every matching cap depressed until the last source releases.
- Input parts use the game's steel, aluminium, rubber and plastic base-colour, normal and roughness/metalness maps. Coloured plastic trims retain the game's normal/ORM finish with authored amber/mint pigment. Texture projection preserves the game map scale of 1.5 m per repeat. These web lights are for design inspection, not a Bevy lighting match.

## Button labels

Native buttons start blank. Configure a supported A–Z or 0–9 key with the Connector;
that key becomes the raised cap label. Clear removes both the key assignment and
label. There is no independently editable decorative label.

The native renderer caches extruded meshes from Mosaic's reusable shaped glyph
outlines. The browser's raster-contour Unicode experiment is historical.

## Controller behaviour

Connect inputs to the controller owning the affected joints. A dial supplies a continuous 0–100% value, linear across 270° with hard stops, retaining its value on release.

Any adjustable numeric controller slider can expose a binding action. Pick the connected dial, set endpoint values in the parameter's units, or invert the direction. One dial may feed many parameters across joints and states; each parameter has one source and its own range. Values remain within valid physical bounds. State-local parameters apply only while their state is active. Inactive states may preview their targets.

Buttons contribute their configured key only to the connected controller. Existing
joint-state keys and matching unmodified gearbox bindings receive logical edges
from the combined seated keyboard and button sources. Releasing one source cannot
release another. Controller state transitions never manufacture key input.

Unlinking a dial mapping preserves the current controller value. Dial operation
changes transient effective values; reset restores authored values. Disconnecting,
deleting, changing a button key, or resetting clears that button's held source.
Each input connects to one controller.

## Game interaction and cost

Aim at a dial and hold Interact, then drag horizontally with fine adjustment available. Show the value near the part during interaction. Release to restore camera control. Aim and hold to press a momentary button; use one press to toggle. Build mode reuses controller connections and compact binding chips.

Shared materials and static housings can batch; animate only dial and cap transforms. Production springs should use a cheap bounded deformation or preset poses, with simpler distant LODs. Small controls need a generous interaction target independent of their visible mesh. Inputs emit values/events rather than adding physics joints. Audio, collisions, persistence, range checks, full controller editing, placement rules and game integration remain outside this concept.

## State readability revision

Released buttons have a dark, non-emissive perimeter and side seam. Held buttons
light both the top-facing perimeter and side seam. The 5 cm bezel aligns with the
depressed cap; the 10 and 25 cm guards have fixed witness edges at that height.
Cap, rim, and label all reflect the controller's combined held key, so buttons
sharing that controller/key agree, including when a seated keyboard holds it.

The extruded key label emits mint light while the combined key is held. Unrelated
metal hardware does not glow.
