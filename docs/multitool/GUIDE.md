# Multitool — asset pack and implementation guide

The mechanic's one tool. A 1.80 m staff with a hex prism at each end; the prism the
mechanic is holding forward is the live tool, the far one is a blank. Flip the staff
180° and the roles swap.

```
multitool_pack/
  GUIDE.md                  this file
  multitool-model.js        the rig: geometry, folds, animation states
  Multitool 3D.dc.html      the viewer that mounts the rig (three-d-stage + panel)
  three-d-stage.js          viewer shell (renderer, lighting, OBJ/GLB export)
  Multitool Concepts.dc.html concept board — pass 01 silhouettes, pass 02 folds
```

| Tool | Accent | Fold | Deployed size (X × Y × Z, m) |
|---|---|---|---|
| Sledge | amber `#C98A34` | all six panels laminate, three per side | 0.21 × 0.38 |
| Matter manipulator | mint `#2FD8B4` | three alternating panels become the ring fingers | 0.31 × 0.31 |
| Welder | red `#E2565A` | 2 tines up, 2 blast shields low, 2 heatsink fins flat | 0.72 × 0.30 |
| Connector | cyan `#2FA8D8` | all six open as a stepped antenna cage | 0.46 × 0.46 |

Staff is 1.80 m end to end, 100 mm across the flats, grip 360 mm centred on the
middle. Accents come from the block family (see the root `README.md`); the welder is
the one tool that leaves the family and takes alert red, because it is the only tool
that changes a build irreversibly.

## The rule the whole design rests on

**Nothing is ever spawned, drawn, or holstered.** Every part of every tool is one of
the six shell panels or lives inside the prism. When a fold finishes, the panels are
load-bearing: no strike face means no hammer, no fingers means no ring, no shields
means the welder throws sparks in the player's face. If you add a fifth tool, it must
obey this — find a job for all six panels or explain which three are idle.

## Geometry

Authored in metres, +Y up, staff centred on the origin, front of a deployed end is
+Z. Names are stable and exported verbatim; the exporter emits one material per
`PALETTE` key.

Key constants at the top of `multitool-model.js`:

| Const | Value | Meaning |
|---|---|---|
| `AF` | 0.050 | hex apothem — 100 mm across flats |
| `PL` | 0.280 | panel length |
| `PT` | 0.011 | panel thickness |
| `PW` | 0.052 | panel width |
| `PRISM_Y0` | 0.600 | prism base height on the staff |
| `HALF_LEN` | 0.900 | staff half length |

### Panel rig — five nested transforms per panel

Every panel is the same six-deep chain. Nothing else moves.

```
panel_carrier_k   yaw around the staff       (index panels to new quadrants)
  radial          z offset from the axis     (breathe out / clamp in)
    slide         y travel along the shaft   (run down)
      hinge       x rotation                 (fold)
        spinner   y rotation + z scale       (turn inside out / thicken)
          plate + role attachments
```

Role attachments (`tip`, `wing`, `fins`, `strike`, `clamp`, `innerFace`) are groups
shown or hidden by role, so one panel mesh serves all four tools.

## The three beats

Deployment is one 0–1 progress value per end, `state.p[key]`, split into three
overlapping windows. Beats 1 and 2 are identical for every tool; **only beat 3
differs.** That is what makes the flip feel like one mechanism rather than four.

| Beat | Window on `p` | What moves | Per-tool? |
|---|---|---|---|
| 1 unlatch | 0.00 – 0.18 | `radial` +8 mm, all six at once, seam light on | no |
| 2 run down | 0.18 – 0.50 | `slide` −100 mm, core revealed | no |
| 3 fold | 0.55 – 1.00 | `hinge`, `spinner`, `carrier` yaw to per-tool targets | **yes** |

Total 0.75 s at the default rate. Stowing is the same value driven back to 0, so the
animation is symmetric and needs no separate clips.

### Adding a tool

Add an entry to `FOLDS`, keyed by tool id, returning per panel index `k`:

```js
{ r, s, f, y, th, spin, role }
//  r    radial z at full deploy (m)
//  s    slide y at full deploy (m)
//  f    hinge x at full deploy (rad)
//  y    extra carrier yaw at full deploy (rad) — used to index panels around
//  th   spinner z scale at full deploy (panels forging thicker; sledge uses 2.0)
//  spin spinner y rotation (Math.PI turns a panel inside out)
//  role which attachment group is visible
```

Then add a core group under `makeEnd` (`cores.<id>`) and an accent key to `ACCENT`.

## Accent and emissive rules

- Accent bands are **gated on beat 1** (`lit = e1 > 0.04`). A fully stowed prism is
  bare metal, so a closed end never tells the player which tool is inside — that is
  the whole reason the staff reads as an innate staff when walked with.
- Emissive is deliberately thin: a tipped edge, one band, one lens per tool. The tool
  sits bottom-right on screen for the entire game; anything brighter blooms over the
  thing being worked on.
- Accent micro-bands and the machined `innerFace` plate are hidden on the sledge's
  laminated panels — the strike group carries its own bands there.

## Idle and active states

Set by `api.setUse(bool)` — wire this to the fire button, not to selection.

| Tool | Idle | Active |
|---|---|---|
| Sledge | none, deliberately | none — the swing is animation on the character, not the tool |
| Matter manipulator | payload cube turns slowly in the ring | shot every 1.15 s: live trio splays out fast, eases back, then hands off to the other three panels |
| Welder | arc sphere breathes | arc swells; all six panels jitter on two out-of-phase frequencies (hinge flutter + radial rattle) |
| Connector | ribs wave ±3°, each on its own phase | panel ring winds around the mast at 3.8 rad/s and the ribs draw 42% closer, as if spooling wire |

The sledge having no active state is intentional: it is the only tool whose feedback
should come entirely from impact.

## API

`mount(stage)` returns:

| Call | Effect |
|---|---|
| `setTool(id)` | replaces the near end's tool and re-deploys from 0 |
| `setOtherTool(id)` | sets the far end's tool without animating |
| `setDeployed(bool)` | deploy or stow the near end |
| `setUse(bool)` | active state on/off (see table above) |
| `flip()` | stows the near end, deploys the far one, spins the staff 180°; returns the newly live tool id |

In game, `flip()` is the only tool-change verb the player has. Selection should
choose what the *far* end becomes (`setOtherTool`) while it is still closed, so the
change is invisible until the flip reveals it.

## Implementation notes for the engine side

- **One skeleton, four clips.** Beats 1–2 are shared, so authoring this as one
  additive rig with a per-tool beat-3 clip is cheaper than four separate animations
  and keeps the mechanism reading as one object.
- **Flip is the state change.** Run the far end's beats 1–2 in mirror during the same
  turn so the mechanic's hands never stop moving; do not gate input on the animation
  finishing — let beat 3 finish after control returns.
- **Collision.** Use the stowed prism capsule for everything except the sledge; give
  the sledge the head box only during the strike window. The connector cage should
  never have collision — the shape reads fragile on purpose.
- **Never scale the staff.** The 1.80 m length is what lets it double as a walking
  staff and what makes the 0.72 m welder shields read as wide.
- **Z-fighting.** The rig is authored with every mating face deliberately offset (see
  the clamp/lip/lamina spacings). If you retopologise, keep the rule: no two visible
  faces coplanar, and no core part wider than the panel stack it sits inside.

## Exporting meshes

Open `Multitool 3D.dc.html`, pick a tool, wait for the fold to finish, then use the
toolbar top-right: **OBJ+MTL** or **GLB**. The export captures the current pose, so
export once per tool (and once stowed) if you want the four folds as separate files.
Material and mesh names survive the export.

## Open items

- No textures yet. Everything is flat PBR material; the next pass would take the
  chosen folds through the same atlas / normal / ORM / emissive pipeline as the
  blocks.
- No first-person or third-person mount points authored — say where the hands go and
  they can be added as named empties.
