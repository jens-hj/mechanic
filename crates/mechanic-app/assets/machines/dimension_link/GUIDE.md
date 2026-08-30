# Dimension Link — texture pack

**This is one block in two states.** Not two blocks. Same mesh, same UV layout, same
atlas packing, same 2×1×1 collision box. `enabled/` and `disabled/` differ only in what
is painted into the four maps, so you can swap the material's textures at runtime and
nothing about the geometry, the UVs or the placement changes.

```
dimension_link_pack/
├── enabled/     ← the live state: the rift is open and containment is losing
├── disabled/    ← the dead state: capped, spliced, welded, no light
└── GUIDE.md
```

Each state folder holds the same six files:

| File | What it is |
| --- | --- |
| `dimension_link_base_color.png` | 2048², sRGB, no baked lighting, opaque alpha |
| `dimension_link_normal.png` | 2048², linear, OpenGL convention (+Y up) |
| `dimension_link_orm.png` | 2048², linear. R = ambient occlusion, G = roughness, B = metallic |
| `dimension_link_emissive.png` | 2048², sRGB. Black where nothing emits |
| `dimension_link.glb` | The block, textures embedded, ready to drop in |
| `reference_sheet.png` | Axonometric hero + orthographic views, for eyeballing the intent |

## Mesh and placement

- 2×1×1 block units at 0.25 m per unit → **0.50 × 0.25 × 0.25 m**.
- Y up, front faces +Z, origin centred in the collision box.
- One mesh, one material, one UV set (UV0). Six quads, 24 verts, explicit tangents.
- Atlas pack is a 4×4 cell grid: `pz [0,0,2,1] · nz [2,0,2,1] · py [0,1,2,1] ·
  ny [2,1,2,1] · px [0,2,1,1] · nx [1,2,1,1]`. Both states use it identically.

## Swapping states

Keep one material and reassign the four maps, or keep two materials and switch the
index. Either way:

- **Do not** swap the mesh, the UVs or the tangents — they are byte-identical.
- Emissive intensity should stay on the same value for both states. The disabled
  emissive map is verified fully black (max channel 0), so the glow goes away on its own;
  if you instead zero the intensity, the enabled state will need it restored exactly.
- Both states are built for a bloom pass on the emissive buffer only. The enabled map
  lights the two innermost aperture plates plus hairlines in the inner third of each
  crack — nothing else. If you widen what glows, bloom eats the plate edges and the
  aperture flattens into a blob.

## What the two states are saying

Every face is mirror-symmetric about both axes with the aperture dead centre, on both
states — the thing inside does not care which side is the front, so all six faces are
the front.

**Enabled.** Read from the centre out, each band owns one idea: the open plate stack
driving straight at the viewer; a clamp of four arc segments with gaps on the axes (a
closed ring reads as a finished part, a segmented one reads as an assembly under load);
wedges prying out through those four gaps; a thin strap bolted straight over the
aperture and an outboard pair clear of the clamp; corner turnbuckles on top, because
external hardware sits over the straps; and a fracture field walking out to the plate
edges, authored in one quadrant and mirrored into the other three so the spalling is
symmetric while still reading as damage. Rift violet is `#BC4FE8` — one step past the
servo family colour, because this is the only block in the set whose contents are not
machinery.

**Disabled.** Nothing was deleted to turn it off, because none of the damage is
undoable. The aperture is capped by a blank cross-seamed lid on a shadow gap, dogged
into the clamp, with a socket boss at centre — no plate stack, no depth, nothing to look
into. The four clamp gaps are spliced with bolted plates, so the clamp is a closed ring
again. Every crack is welded: a muted bead runs its full length, repaired rather than
absent, and it is the only tell that this block was ever open. The accent drops out of
every band — hazard rows go bare grey, strap crowns and turnbuckle rods lose their
strained highlight — and nothing emits.

## Sampling notes

- Clamp both U and V. There is no padding between atlas cells beyond the packer's own
  cell edges, so wrapping bleeds a neighbouring face into the seam.
- Linear filtering with mipmaps. The art is flat vector-tech: point filtering makes the
  hairlines crawl, and aggressive anisotropy is not needed at 2048² for a 25 cm box.
- `orm` is a linear/no-colour-space texture. If your engine wants separate AO, roughness
  and metallic textures, split the channels rather than re-encoding to sRGB.
