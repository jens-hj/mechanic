---
name: mechanic-material-import
description: Import updated Mechanic construction-material PNG archives, reconcile archive materials with the supported set, normalize runtime maps, regenerate picker thumbnails, and verify the asset contract. Use for construction-material PBR texture sets; do not use for authored machine textures.
---

# Mechanic Material Import

Use the bundled importer instead of copying texture files individually:

```bash
.agents/skills/mechanic-material-import/scripts/import-materials.sh /path/to/materials.zip
```

Before importing, run `--check`. If the archive adds materials or omits supported materials, stop and ask the user whether each difference should be added, removed, or ignored. List both the added and missing names. Do not infer that an archive difference changes the supported material set. After the user decides, update the construction-material enum, runtime loading, UI, compatibility tests, and the importer's `materials` list as required before retrying.

The archive must contain canonical `base_color`, `normal`, and `orm` PNG maps for every supported construction material beneath one `materials_styled/` directory. Underscore-prefixed source alternates are ignored. Source maps may be 1024, 2048, or 3072 pixels square, RGBA, and 8-bit. The importer validates every canonical map before touching tracked assets, normalizes runtime maps to 3072×3072, and regenerates both 48×48 flat picker thumbnails and 96×106 isometric block thumbnails from the normalized base colors. This preserves the repository's 512 texture pixels per construction block and keeps both material selectors in sync.

Use `--check` to exercise validation and conversion without installing files:

```bash
.agents/skills/mechanic-material-import/scripts/import-materials.sh --check /path/to/materials.zip
```

After import, inspect `git status --short` and the material portion of `git diff --stat`. Preserve unrelated worktree changes. Run `cargo test -p mechanic-app rendering_tests::material_maps_use_repeat_sampling_and_explicit_color_spaces` when Rust code or material-loading behavior also changed; the script handles the structural checks for a texture-only import, but appearance still needs a visual sample.
