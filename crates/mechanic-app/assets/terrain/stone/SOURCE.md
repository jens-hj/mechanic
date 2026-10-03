# Terrain stone maps

Rock 01 by Rob Tuytel, Poly Haven: https://polyhaven.com/a/rock_01
CC0: https://polyhaven.com/license

2K PNG maps downloaded from the Poly Haven files API on 2026-10-03:
- `rock_01_diff_2k.png` → `stone_base_color.png` (sRGB diffuse)
- `rock_01_nor_gl_2k.png` → `stone_normal.png` (linear OpenGL tangent normal)
- `rock_01_arm_2k.png` → `stone_orm.png` (linear AO, roughness, metalness)

The maps use the terrain's existing 1.5-metre base repeat. They
were resized to 1536 × 1536 with `sips -z 1536 1536` to match the terrain
array contract; the existing terrain loader builds the mip chain.
Construction-material stone maps are separate and unchanged.
