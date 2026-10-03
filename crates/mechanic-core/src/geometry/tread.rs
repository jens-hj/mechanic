//! Tread patterns cut into a part's surfaces.
//!
//! A tread is a tile of raised lugs and recessed grooves repeated over one
//! surface, cut to a depth below it. The envelope does not change: the lug tops
//! are the surface. What a tread changes is how the surface meets the ground.
//! Lugs bite into ground that yields, so soil and sand grip harder, and they
//! carry the load on less area, so the ground under them sinks further. On
//! rock, and against other bodies, a tread gives up a little grip with the area
//! its grooves take out of contact.

use super::face::FaceKind;
use super::grid::GRID_UNIT_METERS;
use super::layers::LayerFace;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Cells along each side of one tread tile.
pub const TREAD_TILE_CELLS: u32 = 8;

/// Side of one tread tile, in metres: one construction block.
pub const TREAD_TILE_METERS: f32 = GRID_UNIT_METERS;

/// Side of one tread cell, in metres.
#[expect(clippy::cast_precision_loss, reason = "eight cells")]
pub const TREAD_CELL_METERS: f32 = TREAD_TILE_METERS / TREAD_TILE_CELLS as f32;

/// Shallowest tread, in millimetres.
pub const MIN_TREAD_DEPTH_MM: u8 = 2;

/// Deepest tread, in millimetres.
pub const MAX_TREAD_DEPTH_MM: u8 = 30;

/// Depth a new tread is cut to, in millimetres.
pub const DEFAULT_TREAD_DEPTH_MM: u8 = 10;

/// Least share of a surface counted as carrying load. Finer lugs than this
/// press no harder: the ground between them carries the rest.
const MIN_CONTACT_RATIO: f32 = 0.25;

/// Grip a tread adds on yielding ground per metre of lug wall in each square
/// metre of surface, per metre of depth.
const BITE_PER_WALL_DEPTH: f32 = 2.0;

/// Most grip a tread adds on yielding ground, as a share of the smooth grip.
const MAX_BITE: f32 = 1.0;

/// Grip a tread with no lugs left would give up on rigid ground, as a share.
const FIRM_GRIP_LOSS: f32 = 0.2;

/// Invalid tread, or a surface that cannot take one.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum TreadError {
    /// No cell of the tile is raised, so nothing would touch the ground.
    #[error("a tread pattern needs at least one raised cell")]
    NoLugs,
    /// Every cell of the tile is raised, so there is no groove to cut.
    #[error("a tread pattern needs at least one groove")]
    NoGrooves,
    /// The depth is outside the supported range.
    #[error("a tread must be between {MIN_TREAD_DEPTH_MM} and {MAX_TREAD_DEPTH_MM} mm deep")]
    DepthOutOfRange,
    /// The part has no such surface, or the surface cannot take a tread.
    #[error("this surface cannot take a tread")]
    UnsupportedSurface,
    /// A bore tread needs a bore.
    #[error("only a hollow cylinder has a bore to cut a tread into")]
    BoreRequired,
    /// A cylinder sector has open sides that a tread cannot wrap around.
    #[error("only a full cylinder takes a tread")]
    PartialCylinder,
    /// The part carries gear teeth or a rack.
    #[error("a toothed part does not take a tread")]
    ToothedPart,
    /// The cylinder carries a spiral.
    #[error("a spiral cylinder does not take a tread")]
    SpiralPart,
}

/// One tread tile: eight by eight cells, each raised or recessed.
///
/// Bit `row * 8 + column` is set where the cell is a raised lug. Columns run
/// along the surface's first direction, which on a cylinder wall is the way it
/// rolls; rows run across it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct TreadMask(u64);

impl TreadMask {
    /// Creates a tile from its cell bits.
    ///
    /// # Errors
    ///
    /// Returns [`TreadError::NoLugs`] for an empty tile and
    /// [`TreadError::NoGrooves`] for a full one.
    pub const fn new(bits: u64) -> Result<Self, TreadError> {
        match bits {
            0 => Err(TreadError::NoLugs),
            u64::MAX => Err(TreadError::NoGrooves),
            bits => Ok(Self(bits)),
        }
    }

    /// Builds a tile from a rule naming each raised cell. The rule must leave
    /// at least one cell raised and one recessed.
    fn from_rule(raised: impl Fn(u32, u32) -> bool) -> Self {
        let mut bits = 0;
        let mut row = 0;
        while row < TREAD_TILE_CELLS {
            let mut column = 0;
            while column < TREAD_TILE_CELLS {
                if raised(column, row) {
                    bits |= 1 << (row * TREAD_TILE_CELLS + column);
                }
                column += 1;
            }
            row += 1;
        }
        Self(bits)
    }

    /// The cell bits.
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Whether a cell is a raised lug. Coordinates wrap, so any cell of a
    /// surface tiled with this pattern can be asked directly.
    pub const fn raised(self, column: u32, row: u32) -> bool {
        let column = column % TREAD_TILE_CELLS;
        let row = row % TREAD_TILE_CELLS;
        self.0 & (1 << (row * TREAD_TILE_CELLS + column)) != 0
    }

    /// The tile with one cell flipped.
    ///
    /// # Errors
    ///
    /// Returns the [`TreadMask::new`] error when the flip would leave the tile
    /// all lugs or all grooves.
    pub const fn toggled(self, column: u32, row: u32) -> Result<Self, TreadError> {
        let column = column % TREAD_TILE_CELLS;
        let row = row % TREAD_TILE_CELLS;
        Self::new(self.0 ^ (1 << (row * TREAD_TILE_CELLS + column)))
    }

    /// Share of the tile that is raised and meets the ground.
    #[expect(clippy::cast_precision_loss, reason = "at most 64 cells")]
    pub const fn contact_ratio(self) -> f32 {
        self.0.count_ones() as f32 / (TREAD_TILE_CELLS * TREAD_TILE_CELLS) as f32
    }

    /// Cell sides where a lug meets a groove, across the tile and its wrap to
    /// the next one: each is one cell-length of lug wall.
    pub const fn wall_cells(self) -> u32 {
        let mut walls = 0;
        let mut row = 0;
        while row < TREAD_TILE_CELLS {
            let mut column = 0;
            while column < TREAD_TILE_CELLS {
                let raised = self.raised(column, row);
                walls += (raised != self.raised(column + 1, row)) as u32;
                walls += (raised != self.raised(column, row + 1)) as u32;
                column += 1;
            }
            row += 1;
        }
        walls
    }
}

impl<'de> Deserialize<'de> for TreadMask {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Self::new(u64::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// The lug layout of a tread.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TreadPattern {
    /// Ribs running the way the surface rolls: steady sideways, quiet ahead.
    Ribbed,
    /// Bars straight across: paddles that dig in ahead.
    Lugged,
    /// Bars in a V, the tractor tyre's directional lugs.
    Chevron,
    /// Staggered square blocks, the all-terrain knobbly.
    Block,
    /// Small separate studs on a wide floor: the least area, the most bite.
    Studded,
    /// A tile drawn by hand.
    Custom(TreadMask),
}

impl TreadPattern {
    /// Every built-in pattern in display order.
    pub const BUILT_IN: [Self; 5] = [
        Self::Ribbed,
        Self::Lugged,
        Self::Chevron,
        Self::Block,
        Self::Studded,
    ];

    /// Human-readable pattern name.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Ribbed => "Ribbed",
            Self::Lugged => "Lugged",
            Self::Chevron => "Chevron",
            Self::Block => "Block",
            Self::Studded => "Studded",
            Self::Custom(_) => "Custom",
        }
    }

    /// The tile this pattern repeats.
    pub fn mask(self) -> TreadMask {
        const CHEVRON_OFFSET: [u32; 8] = [3, 2, 1, 0, 0, 1, 2, 3];
        const fn stagger(row: u32) -> u32 {
            if row < 4 { 0 } else { 2 }
        }
        match self {
            Self::Ribbed => TreadMask::from_rule(|_, row| row % 4 != 3),
            Self::Lugged => TreadMask::from_rule(|column, _| column % 4 < 2),
            Self::Chevron => {
                TreadMask::from_rule(|column, row| (column + CHEVRON_OFFSET[row as usize]) % 4 < 2)
            }
            Self::Block => {
                TreadMask::from_rule(|column, row| row % 4 != 3 && (column + stagger(row)) % 4 != 3)
            }
            Self::Studded => {
                TreadMask::from_rule(|column, row| row % 4 < 2 && (column + stagger(row)) % 4 < 2)
            }
            Self::Custom(mask) => mask,
        }
    }
}

/// A validated tread: a pattern cut to a depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct TreadSpec {
    pattern: TreadPattern,
    depth_mm: u8,
}

impl TreadSpec {
    /// Creates a tread.
    ///
    /// # Errors
    ///
    /// Returns [`TreadError::DepthOutOfRange`] outside
    /// [`MIN_TREAD_DEPTH_MM`]..=[`MAX_TREAD_DEPTH_MM`].
    pub const fn new(pattern: TreadPattern, depth_mm: u8) -> Result<Self, TreadError> {
        if depth_mm < MIN_TREAD_DEPTH_MM || depth_mm > MAX_TREAD_DEPTH_MM {
            return Err(TreadError::DepthOutOfRange);
        }
        Ok(Self { pattern, depth_mm })
    }

    /// The lug layout.
    pub const fn pattern(self) -> TreadPattern {
        self.pattern
    }

    /// Groove depth in millimetres.
    pub const fn depth_mm(self) -> u8 {
        self.depth_mm
    }

    /// Groove depth in metres.
    pub fn depth_meters(self) -> f32 {
        f32::from(self.depth_mm) * 1.0e-3
    }

    /// How this tread changes the way its surface meets the ground.
    #[expect(clippy::cast_precision_loss, reason = "at most 128 walls")]
    pub fn response(self) -> TreadResponse {
        let mask = self.pattern.mask();
        let tile_cells = (TREAD_TILE_CELLS * TREAD_TILE_CELLS) as f32;
        // Metres of lug wall in each square metre of tread.
        let walls_per_meter = mask.wall_cells() as f32 / (tile_cells * TREAD_CELL_METERS);
        TreadResponse {
            contact_ratio: mask.contact_ratio().max(MIN_CONTACT_RATIO),
            bite: (BITE_PER_WALL_DEPTH * walls_per_meter * self.depth_meters()).min(MAX_BITE),
        }
    }
}

impl<'de> Deserialize<'de> for TreadSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            pattern: TreadPattern,
            depth_mm: u8,
        }
        let raw = Raw::deserialize(deserializer)?;
        Self::new(raw.pattern, raw.depth_mm).map_err(serde::de::Error::custom)
    }
}

/// How a tread changes its surface's contact. Game tuning, not measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TreadResponse {
    /// Share of the surface the lugs carry the load on, at least a quarter.
    pub contact_ratio: f32,
    /// Grip the lugs add on yielding ground, as a share of the smooth grip.
    pub bite: f32,
}

impl TreadResponse {
    /// A smooth surface.
    pub const SMOOTH: Self = Self {
        contact_ratio: 1.0,
        bite: 0.0,
    };

    /// Friction multiplier against ground that yields: the lugs dig in and
    /// shear it rather than sliding over it.
    pub fn yielding_grip(self) -> f32 {
        1.0 + self.bite
    }

    /// Friction multiplier against rigid ground and other bodies: the grooves
    /// take some of the surface out of contact.
    pub fn firm_grip(self) -> f32 {
        1.0 - FIRM_GRIP_LOSS * (1.0 - self.contact_ratio)
    }

    /// How much harder the lugs press than a smooth surface under the same
    /// load, so how much further they sink into soft ground.
    pub fn pressure_factor(self) -> f32 {
        self.contact_ratio.recip()
    }
}

/// Every surface a part's tread can be cut into, in slot order.
const TREAD_SURFACES: [LayerFace; 8] = [
    LayerFace::Face(FaceKind::PositiveX),
    LayerFace::Face(FaceKind::NegativeX),
    LayerFace::Face(FaceKind::PositiveY),
    LayerFace::Face(FaceKind::NegativeY),
    LayerFace::Face(FaceKind::PositiveZ),
    LayerFace::Face(FaceKind::NegativeZ),
    LayerFace::OuterWall,
    LayerFace::Bore,
];

const fn tread_slot(surface: LayerFace) -> usize {
    match surface {
        LayerFace::Face(FaceKind::PositiveX) => 0,
        LayerFace::Face(FaceKind::NegativeX) => 1,
        LayerFace::Face(FaceKind::PositiveY) => 2,
        LayerFace::Face(FaceKind::NegativeY) => 3,
        LayerFace::Face(FaceKind::PositiveZ) => 4,
        LayerFace::Face(FaceKind::NegativeZ) => 5,
        LayerFace::OuterWall => 6,
        LayerFace::Bore => 7,
    }
}

/// The treads cut into one part, at most one per surface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SurfaceTreads {
    slots: [Option<TreadSpec>; 8],
}

impl SurfaceTreads {
    /// No treads: every surface smooth.
    pub const NONE: Self = Self { slots: [None; 8] };

    /// The tread on one surface.
    pub const fn get(self, surface: LayerFace) -> Option<TreadSpec> {
        self.slots[tread_slot(surface)]
    }

    /// Whether every surface is smooth.
    pub fn is_empty(self) -> bool {
        self.slots.iter().all(Option::is_none)
    }

    /// Every tread with its surface, in a fixed surface order.
    pub fn iter(self) -> impl Iterator<Item = (LayerFace, TreadSpec)> {
        TREAD_SURFACES
            .into_iter()
            .zip(self.slots)
            .filter_map(|(surface, tread)| tread.map(|tread| (surface, tread)))
    }

    /// These treads with one surface's replaced; `None` makes it smooth.
    #[must_use]
    pub const fn with(mut self, surface: LayerFace, tread: Option<TreadSpec>) -> Self {
        self.slots[tread_slot(surface)] = tread;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tread(pattern: TreadPattern, depth_mm: u8) -> TreadSpec {
        TreadSpec::new(pattern, depth_mm).expect("valid tread")
    }

    #[test]
    fn every_built_in_pattern_has_lugs_and_grooves() {
        for pattern in TreadPattern::BUILT_IN {
            let mask = pattern.mask();
            assert_eq!(TreadMask::new(mask.bits()), Ok(mask), "{}", pattern.label());
            assert!(mask.wall_cells() > 0, "{}", pattern.label());
        }
    }

    #[test]
    fn a_tile_must_keep_a_lug_and_a_groove() {
        assert_eq!(TreadMask::new(0), Err(TreadError::NoLugs));
        assert_eq!(TreadMask::new(u64::MAX), Err(TreadError::NoGrooves));
        let one = TreadMask::new(1).expect("one lug");
        assert_eq!(one.toggled(0, 0), Err(TreadError::NoLugs));
        let toggled = one.toggled(9, 1).expect("cells wrap");
        assert!(toggled.raised(1, 1));
        assert!(toggled.raised(1, 9));
    }

    #[test]
    fn depth_is_bounded() {
        assert_eq!(
            TreadSpec::new(TreadPattern::Block, MIN_TREAD_DEPTH_MM - 1),
            Err(TreadError::DepthOutOfRange)
        );
        assert_eq!(
            TreadSpec::new(TreadPattern::Block, MAX_TREAD_DEPTH_MM + 1),
            Err(TreadError::DepthOutOfRange)
        );
        assert!(TreadSpec::new(TreadPattern::Block, MAX_TREAD_DEPTH_MM).is_ok());
    }

    #[test]
    fn deeper_treads_bite_harder_up_to_a_limit() {
        for pattern in TreadPattern::BUILT_IN {
            let shallow = tread(pattern, MIN_TREAD_DEPTH_MM).response();
            let deep = tread(pattern, MAX_TREAD_DEPTH_MM).response();
            assert!(
                deep.yielding_grip() > shallow.yielding_grip(),
                "{pattern:?}"
            );
            assert!(shallow.yielding_grip() > 1.0);
            assert!(deep.yielding_grip() <= 1.0 + MAX_BITE);
        }
    }

    #[test]
    fn treads_trade_firm_grip_for_soft_ground_grip_and_sinkage() {
        for pattern in TreadPattern::BUILT_IN {
            let response = tread(pattern, DEFAULT_TREAD_DEPTH_MM).response();
            assert!(response.firm_grip() < 1.0, "{pattern:?}");
            assert!(response.firm_grip() >= 1.0 - FIRM_GRIP_LOSS);
            assert!(response.pressure_factor() > 1.0, "{pattern:?}");
        }
        // Studs leave the least area: they sink furthest and grip rock least.
        let studs = tread(TreadPattern::Studded, 10).response();
        let ribs = tread(TreadPattern::Ribbed, 10).response();
        assert!(studs.pressure_factor() > ribs.pressure_factor());
        assert!(studs.firm_grip() < ribs.firm_grip());
    }

    #[test]
    fn a_sparse_custom_tile_presses_no_harder_than_the_floor_allows() {
        let pin = tread(
            TreadPattern::Custom(TreadMask::new(1).expect("one lug")),
            MAX_TREAD_DEPTH_MM,
        );
        let response = pin.response();
        assert!((response.pressure_factor() - MIN_CONTACT_RATIO.recip()).abs() < 1.0e-6);
    }

    #[test]
    fn surface_treads_keep_one_tread_per_surface() {
        let block = tread(TreadPattern::Block, 12);
        let ribs = tread(TreadPattern::Ribbed, 4);
        let treads = SurfaceTreads::NONE
            .with(LayerFace::OuterWall, Some(block))
            .with(LayerFace::Face(FaceKind::NegativeY), Some(ribs))
            .with(LayerFace::OuterWall, Some(ribs));
        assert_eq!(treads.get(LayerFace::OuterWall), Some(ribs));
        assert_eq!(treads.iter().count(), 2);
        let smooth = treads
            .with(LayerFace::OuterWall, None)
            .with(LayerFace::Face(FaceKind::NegativeY), None);
        assert!(smooth.is_empty());
        assert_eq!(smooth, SurfaceTreads::NONE);
    }

    #[test]
    fn a_tread_reads_back_from_its_serialized_form() {
        let custom = tread(
            TreadPattern::Custom(TreadMask::new(0x00ff_00ff_00ff_00ff).expect("stripes")),
            7,
        );
        let text = ron::to_string(&custom).expect("serializes");
        assert_eq!(ron::from_str::<TreadSpec>(&text).ok(), Some(custom));
        assert!(ron::from_str::<TreadSpec>("(pattern:Block,depth_mm:99)").is_err());
        assert!(ron::from_str::<TreadSpec>("(pattern:Custom(0),depth_mm:9)").is_err());
    }
}
