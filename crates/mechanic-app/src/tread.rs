//! The Tread tool's brush: the pattern and depth the Matter Manipulator cuts
//! into surfaces.

use bevy::prelude::Resource;
use mechanic_core::{DEFAULT_TREAD_DEPTH_MM, TreadMask, TreadPattern, TreadSpec};

/// One session-wide tread brush.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TreadBrush {
    /// What a cut makes of a surface.
    pub(crate) tread: TreadSpec,
    /// The hand-drawn tile, kept while a built-in pattern is chosen so going
    /// back to Custom finds it again.
    pub(crate) custom: TreadMask,
}

impl Default for TreadBrush {
    fn default() -> Self {
        Self {
            tread: TreadSpec::new(TreadPattern::Block, DEFAULT_TREAD_DEPTH_MM)
                .expect("the default depth is in range"),
            custom: TreadPattern::Chevron.mask(),
        }
    }
}

impl TreadBrush {
    /// The brush cutting `pattern` at its current depth. Choosing Custom picks
    /// the kept hand-drawn tile back up.
    #[must_use]
    pub(crate) fn with_pattern(self, pattern: TreadPattern) -> Self {
        let pattern = match pattern {
            TreadPattern::Custom(_) => TreadPattern::Custom(self.custom),
            built_in => built_in,
        };
        self.with_tread(pattern, self.tread.depth_mm())
    }

    /// The brush cutting to `depth_mm`, clamped to the range a tread allows.
    #[must_use]
    pub(crate) fn with_depth(self, depth_mm: u8) -> Self {
        let depth = depth_mm.clamp(
            mechanic_core::MIN_TREAD_DEPTH_MM,
            mechanic_core::MAX_TREAD_DEPTH_MM,
        );
        self.with_tread(self.tread.pattern(), depth)
    }

    /// The brush with one cell of its tile flipped. A built-in pattern becomes
    /// a hand-drawn copy of itself first, so drawing starts from what is shown.
    /// A flip that would leave no lug or no groove is refused.
    #[must_use]
    pub(crate) fn with_cell_toggled(self, column: u32, row: u32) -> Self {
        let Ok(custom) = self.tread.pattern().mask().toggled(column, row) else {
            return self;
        };
        Self { custom, ..self }.with_tread(TreadPattern::Custom(custom), self.tread.depth_mm())
    }

    /// The brush matching a sampled tread, keeping a sampled tile to draw on.
    #[must_use]
    pub(crate) fn sampled(self, tread: TreadSpec) -> Self {
        let custom = match tread.pattern() {
            TreadPattern::Custom(mask) => mask,
            _ => self.custom,
        };
        Self { tread, custom }
    }

    fn with_tread(self, pattern: TreadPattern, depth_mm: u8) -> Self {
        Self {
            tread: TreadSpec::new(pattern, depth_mm).unwrap_or(self.tread),
            ..self
        }
    }
}

/// How a tread changes grip and sinkage, for the status line and the panel.
pub(crate) fn response_summary(tread: TreadSpec) -> String {
    let response = tread.response();
    format!(
        "{:.0}% contact · soft ground grip ×{:.2} · rock grip ×{:.2} · presses ×{:.1}",
        response.contact_ratio * 100.0,
        response.yielding_grip(),
        response.firm_grip(),
        response.pressure_factor(),
    )
}

/// The brush's pattern and depth, e.g. `Chevron · 12 mm`.
pub(crate) fn tread_label(tread: TreadSpec) -> String {
    format!("{} · {} mm", tread.pattern().label(), tread.depth_mm())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drawing_on_a_built_in_pattern_starts_a_custom_copy_of_it() {
        let brush = TreadBrush::default().with_pattern(TreadPattern::Ribbed);
        let drawn = brush.with_cell_toggled(0, 3);
        let TreadPattern::Custom(mask) = drawn.tread.pattern() else {
            panic!("drawing makes a custom pattern");
        };
        assert_eq!(mask, drawn.custom);
        assert_eq!(
            mask.bits() ^ TreadPattern::Ribbed.mask().bits(),
            1 << (3 * 8)
        );
        assert_eq!(drawn.tread.depth_mm(), brush.tread.depth_mm());
        // Back on a built-in pattern, Custom finds the drawing again.
        let again = drawn
            .with_pattern(TreadPattern::Studded)
            .with_pattern(TreadPattern::Custom(TreadPattern::Block.mask()));
        assert_eq!(again.tread.pattern(), TreadPattern::Custom(mask));
    }

    #[test]
    fn depth_stays_in_range_and_a_tile_keeps_a_lug() {
        let brush = TreadBrush::default();
        assert_eq!(
            brush.with_depth(0).tread.depth_mm(),
            mechanic_core::MIN_TREAD_DEPTH_MM
        );
        assert_eq!(
            brush.with_depth(200).tread.depth_mm(),
            mechanic_core::MAX_TREAD_DEPTH_MM
        );
        let one = TreadMask::new(1).unwrap();
        let single = brush.sampled(TreadSpec::new(TreadPattern::Custom(one), 9).unwrap());
        assert_eq!(single.with_cell_toggled(0, 0), single);
    }
}
