# Embedded fonts

## DejaVuSans.ttf

DejaVu Sans, the embedded fallback font (`EMBEDDED_FONT` in `src/lib.rs`). It is
shaped when no more specific family resolves, and is what makes shaping/metrics
tests deterministic without system fonts.

- Upstream: <https://dejavu-fonts.github.io/>
- License: the DejaVu Fonts License (a permissive, redistributable free-software
  license derived from the Bitstream Vera Fonts License). Full text:
  <https://dejavu-fonts.github.io/License.html>

The font is redistributed unmodified under that license, which permits bundling
and redistribution. Retrieve the canonical `LICENSE` from the upstream release
if a copy of the full text must ship alongside the binary.
