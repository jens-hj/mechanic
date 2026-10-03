# Terrain materials

Grass, dirt and stone are drawn procedurally by default, so no stretch of
ground repeats. Their textures tiled every 1.5 m; the procedural look keeps
their style: wind-combed grass with light flecks, banded soil with outlined
pebbles, and cracked slabs under clusters of pale chips. Sand, iron,
graphite, copper and wood still come from their textures.

The **Options → Graphics → Ground** setting switches between *Procedural* and
*Textured*. It is stored in `settings.ron` as `procedural_ground`. Switched
off, the terrain shader is built without the procedural path and costs
exactly what it did before; the field cache sits idle.

## How it works

```
camera ──► TerrainCacheFocus ──► render world: plan_regions
                                     │  exposed strips, ≤ 1 M texels/frame
                                     ▼
                     terrain_field_cache.wgsl (compute)
                                     │  writes fields
                                     ▼
        stone / soil / grass field arrays (3 planes × 10 levels × 512²)
                                     │  + window table
                                     ▼
 terrain_material.wgsl ── per projection: pick level ─► sample fields ─► compose look
                         + grain tile (fine flecks, grit, speckle)
```

**Fields, not colours.** The cache holds what the looks are composed from:
the distance to each slab's border, each chip's and pebble's signed distance
to its outline, cell ids, and noise for strokes, bands and broad tone. The
fragment shader turns those into colour, occlusion, roughness, tint mask and
facet slopes. Outlines come from the distance fields, so they stay crisp
however far a cache texel is magnified.

**Clipmaps.** Each projection plane (x: yz, y: xz, z: xy) has ten levels of
512² texels, from 3 mm texels covering 1.5 m to 1.5 m texels covering 768 m,
all centred on the camera. Levels are stored toroidally: a window moves in
32-texel snaps and only the strips it exposes are generated. Coarse levels
are always generated first, a level moves whole or not at all, and an
unfilled level is marked so the shader skips it. Three `Rgba8Unorm` arrays
take about 90 MB.

| Array | R | G | B | A |
|---|---|---|---|---|
| stone | slab border distance | chip signed distance | slab id | chip id |
| soil | band noise | pebble signed distance | pebble id | — |
| grass | stroke noise | tone | — | — |

**Level choice.** Per projection the shader takes the finest level whose
texels are no finer than the pixel's geometric-mean width (at least a quarter
of its long side), and coarser if the window cannot reach the point. It
blends into the next level over the last 30% of an octave and the last 15% of
a window's reach. Features smaller than about two texels or two pixels fade
to their mean.

**Grain.** Detail under a centimetre (grass flecks, dirt grit, stone
speckle) would need cache texels finer than the windows allow away from the
camera. It comes from a 512² tile generated at startup (`grain_image`) and
sampled with its mips, shifted by one of four offsets in regions drawn by the
unique stroke noise so no stretch of grain repeats.

**Surfaces.** Every palette look on the grass, dirt or stone texture set uses
the recipe, tinted and recoloured as before; a recolour keeps the recipe's
base colour as its mean. Wilt, wetness, boundary crumbling and the erosion
overlay are unchanged. A look's `scale` does not apply to procedural layers,
because the cache holds one field space.

## Tuning

Shapes are constants in `assets/shaders/terrain_field_cache.wgsl`
(generator); palettes and how fields become colour are in the procedural
section of `assets/shaders/terrain_material.wgsl`. Sizes shared by both are
marked as mirrors. The gallery renders every look close up, across a field,
on a cliff, and along grass/dirt/stone borders, reading the shader from disk
so shader edits need no rebuild:

```sh
MECHANIC_TERRAIN_REFERENCE_SHADER=/path/to/old/terrain_material.wgsl \
  cargo test -p mechanic-app --release terrain_material_gallery -- --ignored --nocapture
```

Images land in `$TMPDIR/mechanic-terrain-gallery`, `-new` beside `-old` when
a reference shader is given.

To add a recipe: give its `TextureSet` a number in `procedural_recipe`
(`world/terrain_render.rs`) and the matching `RECIPE_*` constant, generate
any fields it needs in the cache shader (a free channel, or a new array), and
add a `*_look` composing them.

## Tests and measurements

- `cargo test -p mechanic-app terrain_cache`: window planning, strips, budget,
  grain wrapping.
- `procedural_ground_does_not_repeat` (ignored, real GPU): the strongest
  self-similarity of a top-down 12 m patch for shifts of 1–6 m. The textures
  scored 0.92 at 3 m; procedural grass, dirt and stone score 0.20, 0.09 and 0.03.
- `terrain_shader_preserves_pixels_and_measures_gpu_cost` (ignored): runs
  with procedural ground off, so it still guards the textured path.
- `procedural_terrain_gpu_cost` and `procedural_cache_update_cost` (ignored):
  paired opaque timings per material, and frame cost while the cache moves.

On the M1 Pro, one procedural material filling a 4096×2524 MSAA4 view
measured opaque medians of about 12–14 ms against 7.5 ms textured, which is
the worst case. Generating fields is cheap: refilling 1 M texels a frame did
not change frame time measurably. The in-world terrain pass has not been
measured with procedural ground yet.

## Limits

- Grass strokes (8 cm wide) fade beyond roughly 15–25 m from the camera,
  where the reaching level's texels are too coarse to hold them.
- The cache follows one camera, the `MainCamera`; another view sees fields
  centred elsewhere and falls back to coarser levels.
- After a teleport, fine levels refill over a few frames; coarse ones first.
