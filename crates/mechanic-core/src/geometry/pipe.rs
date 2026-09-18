//! Pipe bends and pipe junctions.

use super::cylinder::{
    CylinderDimensions, MAX_CYLINDER_OUTER_DIAMETER, MIN_CYLINDER_DIAMETER_GAP,
    MIN_CYLINDER_OUTER_DIAMETER,
};
use super::face::FaceKind;
use super::grid::{BuildPose, GRID_UNIT_METERS, GridDimension, MAX_GRID_UNITS};
use super::material::ConstructionMaterial;
use crate::MaterialAppearance;
use thiserror::Error;

/// Radial sides used by the authored pipe-bend render and picking surface.
pub const PIPE_BEND_RADIAL_SIDES: u16 = 24;

/// Centreline slices used by the authored quarter-torus surface.
pub const PIPE_BEND_ARC_SLICES: u16 = 12;

/// Invalid dimensions for a cardinal 90-degree pipe bend.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum PipeBendDimensionError {
    /// The outer diameter was not finite.
    #[error("pipe bend outer diameter must be finite")]
    NonFiniteOuterDiameter,
    /// The outer diameter was outside the cylinder range.
    #[error("pipe bend outer diameter must be between 0.05 m and 8.00 m")]
    OuterDiameterOutOfRange,
    /// The inner diameter was not finite.
    #[error("pipe bend inner diameter must be finite")]
    NonFiniteInnerDiameter,
    /// The inner diameter was negative or left too little wall material.
    #[error(
        "pipe bend inner diameter must be non-negative and at least 0.05 m smaller than the outer diameter"
    )]
    InnerDiameterOutOfRange,
    /// The square span was zero or larger than the grid limit.
    #[error("pipe bend span must be between 1 and {MAX_GRID_UNITS} blocks")]
    SpanOutOfRange,
    /// The square span is narrower than the pipe's block channel.
    #[error("pipe bend span must be at least the outer diameter rounded up to a block")]
    SpanTooSmallForDiameter,
}

/// Validated cross-section and block span for a cardinal quarter-torus.
///
/// A bend fills an `N × N` block square, where `N` is its span. The pipe runs
/// in a channel `W` blocks wide (its outer diameter rounded up to a block), so
/// the centreline sits `W / 2` blocks inside the square's edges and the
/// centreline radius is `(N − W / 2)` blocks. Straight legs entering and
/// leaving the bend therefore stay on the same block alignment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PipeBendDimensions {
    pub(super) outer_diameter: f32,
    pub(super) inner_diameter: f32,
    pub(super) span: GridDimension,
}

impl PipeBendDimensions {
    /// Default span: a bend inside one construction block.
    pub const DEFAULT_SPAN: u8 = 1;

    /// Creates validated pipe-bend dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`PipeBendDimensionError`] when the annular cross-section is
    /// invalid, the span is outside the grid range, or the span is narrower
    /// than the pipe's block channel.
    pub fn new(
        outer_diameter: f32,
        inner_diameter: f32,
        span_blocks: u8,
    ) -> Result<Self, PipeBendDimensionError> {
        if !outer_diameter.is_finite() {
            return Err(PipeBendDimensionError::NonFiniteOuterDiameter);
        }
        if !(MIN_CYLINDER_OUTER_DIAMETER..=MAX_CYLINDER_OUTER_DIAMETER).contains(&outer_diameter) {
            return Err(PipeBendDimensionError::OuterDiameterOutOfRange);
        }
        if !inner_diameter.is_finite() {
            return Err(PipeBendDimensionError::NonFiniteInnerDiameter);
        }
        if inner_diameter < 0.0 || inner_diameter > outer_diameter - MIN_CYLINDER_DIAMETER_GAP {
            return Err(PipeBendDimensionError::InnerDiameterOutOfRange);
        }
        let span =
            GridDimension::new(span_blocks).map_err(|_| PipeBendDimensionError::SpanOutOfRange)?;
        if span_blocks < Self::minimum_span(outer_diameter) {
            return Err(PipeBendDimensionError::SpanTooSmallForDiameter);
        }
        Ok(Self {
            outer_diameter,
            inner_diameter,
            span,
        })
    }

    /// Outer diameter in metres.
    pub const fn outer_diameter(self) -> f32 {
        self.outer_diameter
    }

    /// Inner diameter in metres. Zero represents a solid bend.
    pub const fn inner_diameter(self) -> f32 {
        self.inner_diameter
    }

    /// Side of the bend's square footprint, in blocks.
    pub const fn span_blocks(self) -> u8 {
        self.span.units()
    }

    /// Centreline radius in metres: `(span − channel / 2)` blocks.
    pub fn radius(self) -> f32 {
        (f32::from(self.span.units()) - f32::from(Self::channel_blocks(self.outer_diameter)) * 0.5)
            * GRID_UNIT_METERS
    }

    /// Width in blocks of the channel a pipe of `outer_diameter` runs in.
    pub fn channel_blocks(outer_diameter: f32) -> u8 {
        let blocks = (outer_diameter / GRID_UNIT_METERS - 1.0e-4).ceil();
        (1..=MAX_GRID_UNITS)
            .find(|&units| f32::from(units) >= blocks)
            .unwrap_or(MAX_GRID_UNITS)
    }

    /// Smallest span that holds a pipe of `outer_diameter`.
    pub fn minimum_span(outer_diameter: f32) -> u8 {
        Self::channel_blocks(outer_diameter)
    }
}

impl Default for PipeBendDimensions {
    fn default() -> Self {
        Self {
            outer_diameter: CylinderDimensions::DEFAULT_OUTER_DIAMETER,
            inner_diameter: CylinderDimensions::DEFAULT_INNER_DIAMETER,
            span: GridDimension(Self::DEFAULT_SPAN),
        }
    }
}

/// Cardinal 90-degree pipe bend with local negative-X inlet and positive-Y outlet.
///
/// The pose translation is the theoretical sharp corner. The centreline is
/// tangent at `(-radius, 0, 0)` and `(0, radius, 0)` in local space. When the
/// pipe fills its channel the inner wall pinches to a crease at
/// `(-radius, radius, 0)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PipeBendSpec {
    /// Validated annular dimensions and centreline radius.
    pub dimensions: PipeBendDimensions,
    /// Sharp-corner position and cardinal orientation.
    pub pose: BuildPose,
    /// Material used for appearance, mass, and contact response.
    pub material: ConstructionMaterial,
    /// Independent color and finish treatment.
    pub appearance: MaterialAppearance,
}

impl PipeBendSpec {
    /// Creates a pipe bend from validated dimensions and a build pose.
    pub const fn new(dimensions: PipeBendDimensions, pose: BuildPose) -> Self {
        Self {
            dimensions,
            pose,
            material: ConstructionMaterial::Steel,
            appearance: MaterialAppearance::BAKED,
        }
    }

    /// Uses an explicit construction material.
    #[must_use]
    pub const fn with_material(mut self, material: ConstructionMaterial) -> Self {
        self.material = material;
        self
    }

    /// Uses an explicit construction appearance.
    #[must_use]
    pub const fn with_appearance(mut self, appearance: MaterialAppearance) -> Self {
        self.appearance = appearance;
        self
    }
}

/// Faces a pipe junction can open, in arm-bit order.
pub(super) const PIPE_ARM_FACES: [FaceKind; 6] = [
    FaceKind::PositiveX,
    FaceKind::NegativeX,
    FaceKind::PositiveY,
    FaceKind::NegativeY,
    FaceKind::PositiveZ,
    FaceKind::NegativeZ,
];

/// Invalid pipe-junction cross-section or opening set.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum PipeJunctionError {
    /// The outer diameter was not finite or outside the cylinder range.
    #[error("pipe junction outer diameter must be between 0.05 m and 8.00 m")]
    OuterDiameterOutOfRange,
    /// The inner diameter was not finite, negative, or left too little wall.
    #[error(
        "pipe junction inner diameter must be non-negative and at least 0.05 m smaller than the outer diameter"
    )]
    InnerDiameterOutOfRange,
    /// No face was open.
    #[error("pipe junction needs at least one open face")]
    NoArms,
}

/// Non-empty set of a pipe junction's open faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PipeArms(pub(super) u8);

impl PipeArms {
    /// Every face open.
    pub const ALL: Self = Self(0b11_1111);

    /// Creates an arm set from face bits ordered +X, −X, +Y, −Y, +Z, −Z.
    ///
    /// # Errors
    ///
    /// Returns [`PipeJunctionError::NoArms`] when no face bit is set.
    pub const fn from_bits(bits: u8) -> Result<Self, PipeJunctionError> {
        let bits = bits & 0b11_1111;
        if bits == 0 {
            Err(PipeJunctionError::NoArms)
        } else {
            Ok(Self(bits))
        }
    }

    /// An arm set with exactly one open face.
    pub const fn single(face: FaceKind) -> Self {
        Self(pipe_arm_bit(face))
    }

    /// Face bits ordered +X, −X, +Y, −Y, +Z, −Z.
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether `face` is open.
    pub const fn contains(self, face: FaceKind) -> bool {
        self.0 & pipe_arm_bit(face) != 0
    }

    /// This set with `face` opened too.
    #[must_use]
    pub const fn with(self, face: FaceKind) -> Self {
        Self(self.0 | pipe_arm_bit(face))
    }

    /// Open faces in arm-bit order.
    pub fn faces(self) -> impl Iterator<Item = FaceKind> {
        PIPE_ARM_FACES
            .into_iter()
            .filter(move |&face| self.contains(face))
    }

    /// Number of open faces.
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }
}

pub(super) const fn pipe_arm_bit(face: FaceKind) -> u8 {
    match face {
        FaceKind::PositiveX => 1,
        FaceKind::NegativeX => 1 << 1,
        FaceKind::PositiveY => 1 << 2,
        FaceKind::NegativeY => 1 << 3,
        FaceKind::PositiveZ => 1 << 4,
        FaceKind::NegativeZ => 1 << 5,
    }
}

/// Validated annular cross-section of a pipe junction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PipeJunctionDimensions {
    pub(super) outer_diameter: f32,
    pub(super) inner_diameter: f32,
}

impl PipeJunctionDimensions {
    /// Creates a validated junction cross-section.
    ///
    /// # Errors
    ///
    /// Returns [`PipeJunctionError`] when either diameter is out of range.
    pub fn new(outer_diameter: f32, inner_diameter: f32) -> Result<Self, PipeJunctionError> {
        if !outer_diameter.is_finite()
            || !(MIN_CYLINDER_OUTER_DIAMETER..=MAX_CYLINDER_OUTER_DIAMETER)
                .contains(&outer_diameter)
        {
            return Err(PipeJunctionError::OuterDiameterOutOfRange);
        }
        if !inner_diameter.is_finite()
            || inner_diameter < 0.0
            || inner_diameter > outer_diameter - MIN_CYLINDER_DIAMETER_GAP
        {
            return Err(PipeJunctionError::InnerDiameterOutOfRange);
        }
        Ok(Self {
            outer_diameter,
            inner_diameter,
        })
    }

    /// Outer diameter of each pipe end, in metres.
    pub const fn outer_diameter(self) -> f32 {
        self.outer_diameter
    }

    /// Bore diameter in metres. Zero represents a solid junction.
    pub const fn inner_diameter(self) -> f32 {
        self.inner_diameter
    }

    /// Side of the junction's channel cell in blocks: the pipe's channel width.
    pub fn cell_blocks(self) -> u8 {
        PipeBendDimensions::channel_blocks(self.outer_diameter)
    }

    /// Distance from the junction centre to each arm end, in metres.
    pub fn half_side(self) -> f32 {
        f32::from(self.cell_blocks()) * GRID_UNIT_METERS * 0.5
    }
}

/// Fitting that joins pipe arms on any of the six faces of a channel cell.
///
/// The pose translation is the centre of a cell one pipe channel wide, so a
/// junction replaces exactly one channel cell of a straight run. Each open face
/// carries an arm of pipe from the centre to an annular end on the cell face;
/// the fitting has no other connection faces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PipeJunctionSpec {
    /// Validated annular cross-section shared by every open face.
    pub dimensions: PipeJunctionDimensions,
    /// Open faces, in local space.
    pub arms: PipeArms,
    /// Channel cell centre and cardinal orientation.
    pub pose: BuildPose,
    /// Material used for appearance, mass, and contact response.
    pub material: ConstructionMaterial,
    /// Independent color and finish treatment.
    pub appearance: MaterialAppearance,
}

impl PipeJunctionSpec {
    /// Creates a junction from validated dimensions, open faces, and a pose.
    pub const fn new(dimensions: PipeJunctionDimensions, arms: PipeArms, pose: BuildPose) -> Self {
        Self {
            dimensions,
            arms,
            pose,
            material: ConstructionMaterial::Steel,
            appearance: MaterialAppearance::BAKED,
        }
    }

    /// Uses an explicit construction material.
    #[must_use]
    pub const fn with_material(mut self, material: ConstructionMaterial) -> Self {
        self.material = material;
        self
    }

    /// Uses an explicit construction appearance.
    #[must_use]
    pub const fn with_appearance(mut self, appearance: MaterialAppearance) -> Self {
        self.appearance = appearance;
        self
    }

    /// This junction with one more open face.
    #[must_use]
    pub const fn with_arm(mut self, face: FaceKind) -> Self {
        self.arms = self.arms.with(face);
        self
    }

    /// Number of box colliders [`crate::pipe_junction_wall_boxes`] produces.
    pub fn collider_count(self) -> usize {
        crate::pipe_junction::pipe_junction_box_count(self)
    }
}
