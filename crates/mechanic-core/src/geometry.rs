use bevy_math::{EulerRot, IVec3, Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{MaterialAppearance, PartId};

/// Length of one exact authored position tick: 2.5 mm.
pub const POSITION_TICK_METERS: f32 = 0.0025;
/// Exact position ticks spanning one 25 cm construction cell.
pub const POSITION_TICKS_PER_GRID_UNIT: i32 = 100;
/// Exact position ticks spanning half a construction cell.
pub const POSITION_TICKS_PER_HALF_GRID_UNIT: i32 = POSITION_TICKS_PER_GRID_UNIT / 2;

/// Stable identity of a Dimension Link within one saved world and its Garage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DimensionLinkId(pub u64);

/// A selectable material for ordinary construction parts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstructionMaterial {
    /// Lightweight aluminium alloy.
    Aluminium,
    /// Graphite construction material.
    #[serde(alias = "Carbon")]
    Graphite,
    /// Carbon-fibre composite construction material.
    CarbonFiber,
    /// Dense, high-friction concrete.
    Concrete,
    /// Dense conductive copper.
    Copper,
    /// Compactable earth fill.
    Dirt,
    /// General-purpose structural iron.
    Iron,
    /// Lightweight resilient plastic.
    Plastic,
    /// Compliant high-grip rubber.
    Rubber,
    /// Granular mineral fill.
    Sand,
    /// General-purpose structural steel.
    #[default]
    Steel,
    /// Dense natural stone.
    Stone,
    /// Lightweight timber.
    Wood,
}

impl ConstructionMaterial {
    /// Every selectable material in alphabetical display order.
    pub const ALL: [Self; 13] = [
        Self::Aluminium,
        Self::CarbonFiber,
        Self::Concrete,
        Self::Copper,
        Self::Dirt,
        Self::Graphite,
        Self::Iron,
        Self::Plastic,
        Self::Rubber,
        Self::Sand,
        Self::Steel,
        Self::Stone,
        Self::Wood,
    ];

    /// Human-readable material name.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Aluminium => "Aluminium",
            Self::CarbonFiber => "Carbon Fiber",
            Self::Concrete => "Concrete",
            Self::Copper => "Copper",
            Self::Dirt => "Dirt",
            Self::Graphite => "Graphite",
            Self::Iron => "Iron",
            Self::Plastic => "Plastic",
            Self::Rubber => "Rubber",
            Self::Sand => "Sand",
            Self::Steel => "Steel",
            Self::Stone => "Stone",
            Self::Wood => "Wood",
        }
    }

    /// Density and contact response used by construction physics.
    pub const fn properties(self) -> MaterialProperties {
        match self {
            Self::Aluminium => MaterialProperties::new(2_700.0, 0.61, 0.47, 0.25, 0.004, 69.0e9),
            Self::CarbonFiber => MaterialProperties::new(1_600.0, 0.40, 0.30, 0.20, 0.008, 70.0e9),
            Self::Concrete => MaterialProperties::new(2_400.0, 0.80, 0.65, 0.05, 0.020, 30.0e9),
            Self::Copper => MaterialProperties::new(8_960.0, 0.53, 0.36, 0.20, 0.003, 117.0e9),
            Self::Dirt => MaterialProperties::new(1_600.0, 0.72, 0.55, 0.05, 0.030, 0.05e9),
            Self::Graphite => MaterialProperties::new(1_900.0, 0.25, 0.15, 0.10, 0.010, 12.0e9),
            Self::Iron => MaterialProperties::new(7_870.0, 0.70, 0.55, 0.15, 0.003, 170.0e9),
            Self::Plastic => MaterialProperties::new(950.0, 0.40, 0.30, 0.40, 0.020, 1.0e9),
            Self::Rubber => MaterialProperties::new(1_100.0, 1.00, 0.80, 0.70, 0.040, 0.01e9),
            Self::Sand => MaterialProperties::new(1_700.0, 0.65, 0.50, 0.05, 0.035, 0.03e9),
            Self::Steel => MaterialProperties::new(7_850.0, 0.74, 0.57, 0.20, 0.002, 200.0e9),
            Self::Stone => MaterialProperties::new(2_700.0, 0.60, 0.48, 0.05, 0.015, 50.0e9),
            Self::Wood => MaterialProperties::new(700.0, 0.48, 0.30, 0.15, 0.025, 10.0e9),
        }
    }
}

/// Physical properties belonging to one construction material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialProperties {
    /// Density in kilograms per cubic metre.
    pub density_kg_m3: f32,
    /// Static Coulomb friction coefficient.
    pub static_friction: f32,
    /// Kinetic Coulomb friction coefficient.
    pub dynamic_friction: f32,
    /// Coefficient of restitution.
    pub restitution: f32,
    /// Dimensionless rolling-resistance coefficient.
    pub rolling_resistance: f32,
    /// Young's modulus in pascals.
    pub youngs_modulus_pa: f32,
}

impl MaterialProperties {
    const fn new(
        density_kg_m3: f32,
        static_friction: f32,
        dynamic_friction: f32,
        restitution: f32,
        rolling_resistance: f32,
        youngs_modulus_pa: f32,
    ) -> Self {
        Self {
            density_kg_m3,
            static_friction,
            dynamic_friction,
            restitution,
            rolling_resistance,
            youngs_modulus_pa,
        }
    }

    /// Nominal normal compliance of one 25 cm construction block, in metres
    /// per newton.
    ///
    /// A one-dimensional block column has stiffness `E A / L`. At the engine's
    /// nominal one-block contact area (`A = L²`) this reduces to
    /// `C = 1 / (E × L)`, with `L = GRID_UNIT_METERS`.
    pub const fn nominal_block_compliance(self) -> f32 {
        1.0 / (self.youngs_modulus_pa * GRID_UNIT_METERS)
    }
}

/// Construction-grid spacing in metres.
pub const GRID_UNIT_METERS: f32 = 0.25;

/// Largest cuboid dimension in grid units (8 m).
pub const MAX_GRID_UNITS: u8 = 32;

/// Smallest supported cylinder outer diameter, in metres.
pub const MIN_CYLINDER_OUTER_DIAMETER: f32 = 0.05;

/// Largest supported cylinder outer diameter, in metres.
pub const MAX_CYLINDER_OUTER_DIAMETER: f32 = 8.0;

/// Minimum difference between a cylinder's outer and inner diameters, in metres.
pub const MIN_CYLINDER_DIAMETER_GAP: f32 = 0.05;

/// Smallest supported retained cylinder sector, in degrees.
pub const MIN_CYLINDER_SWEEP_DEGREES: u16 = 15;

/// Full-cylinder sweep angle, in degrees.
pub const MAX_CYLINDER_SWEEP_DEGREES: u16 = 360;

/// Adjustment increment for retained cylinder sectors, in degrees.
pub const CYLINDER_SWEEP_STEP_DEGREES: u16 = 15;

/// Smallest supported pipe-bend centreline radius, in metres.
pub const MIN_PIPE_BEND_RADIUS: f32 = GRID_UNIT_METERS;

/// Largest supported pipe-bend centreline radius, in metres.
pub const MAX_PIPE_BEND_RADIUS: f32 = 8.0;

/// Radial sides used by the authored pipe-bend render and picking surface.
pub const PIPE_BEND_RADIAL_SIDES: u16 = 24;

/// Centreline slices used by the authored quarter-torus surface.
pub const PIPE_BEND_ARC_SLICES: u16 = 12;

/// Converts a world position into its nearest quarter-metre grid coordinate.
pub fn snap_world_to_grid(position: Vec3) -> IVec3 {
    (position / GRID_UNIT_METERS).round().as_ivec3()
}

/// Invalid cuboid grid dimension.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("grid dimensions must be between 1 and {MAX_GRID_UNITS} quarter-metre units; got {0}")]
pub struct DimensionError(pub u8);

/// Cuboid dimension stored as a validated integer count of grid units.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GridDimension(u8);

impl GridDimension {
    /// Creates a dimension in inclusive range 1..=32.
    ///
    /// # Errors
    ///
    /// Returns [`DimensionError`] for zero or more than 32 grid units.
    pub const fn new(units: u8) -> Result<Self, DimensionError> {
        if units == 0 || units > MAX_GRID_UNITS {
            Err(DimensionError(units))
        } else {
            Ok(Self(units))
        }
    }

    /// Integer count of quarter-metre units.
    pub const fn units(self) -> u8 {
        self.0
    }

    /// Dimension in metres.
    pub fn meters(self) -> f32 {
        f32::from(self.0) * GRID_UNIT_METERS
    }
}

impl TryFrom<u8> for GridDimension {
    type Error = DimensionError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

/// Axis used by grid-aligned faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    /// Local x axis.
    X,
    /// Local y axis.
    Y,
    /// Local z axis.
    Z,
}

impl Axis {
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }

    pub(crate) const fn unit(self) -> Vec3 {
        match self {
            Self::X => Vec3::X,
            Self::Y => Vec3::Y,
            Self::Z => Vec3::Z,
        }
    }
}

/// A rotation composed of 90-degree turns around local x, y, and z.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GridRotation {
    quarter_turns_xyz: [u8; 3],
}

impl GridRotation {
    /// Creates a rotation, normalizing each count modulo four.
    pub const fn new(x: u8, y: u8, z: u8) -> Self {
        Self {
            quarter_turns_xyz: [x % 4, y % 4, z % 4],
        }
    }

    /// Normalized quarter turns in x/y/z Euler order.
    pub const fn quarter_turns_xyz(self) -> [u8; 3] {
        self.quarter_turns_xyz
    }

    /// Runtime quaternion corresponding to the discrete rotation.
    pub fn quaternion(self) -> Quat {
        let radians = self
            .quarter_turns_xyz
            .map(|turns| f32::from(turns) * core::f32::consts::FRAC_PI_2);
        Quat::from_euler(EulerRot::XYZ, radians[0], radians[1], radians[2])
    }

    /// Applies a world-space positive-Y cardinal rotation before this rotation.
    ///
    /// # Panics
    ///
    /// Panics only if the finite cardinal-rotation set is not closed under composition.
    #[must_use]
    pub fn rotated_y(self, quarter_turns: u8) -> Self {
        let target = GridRotation::new(0, quarter_turns, 0).quaternion() * self.quaternion();
        (0_u8..4)
            .flat_map(|x| (0_u8..4).flat_map(move |y| (0_u8..4).map(move |z| Self::new(x, y, z))))
            .find(|candidate| candidate.quaternion().abs_diff_eq(target, 1.0e-5))
            .expect("cardinal rotations are closed under composition")
    }
}

/// Grid-aligned build pose.
///
/// The primary translation remains in quarter-metre units. An internal exact
/// 2.5 mm tick offset allows precision placement while preserving the positions
/// produced by [`BuildPose::new`] and legacy half-grid construction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BuildPose {
    /// Translation in integer construction-grid units.
    pub translation_units: IVec3,
    /// Discrete 90-degree orientation.
    pub rotation: GridRotation,
    position_tick_offset: [u8; 3],
}

impl BuildPose {
    /// Creates a build pose.
    pub const fn new(translation_units: IVec3, rotation: GridRotation) -> Self {
        Self {
            translation_units,
            rotation,
            position_tick_offset: [0; 3],
        }
    }

    /// Creates a build pose from exact integer 2.5 mm centre coordinates.
    pub fn from_position_ticks(translation_ticks: IVec3, rotation: GridRotation) -> Self {
        let mut translation_units = IVec3::ZERO;
        let mut position_tick_offset = [0; 3];
        for axis in 0..3 {
            let remainder = translation_ticks[axis].rem_euclid(POSITION_TICKS_PER_GRID_UNIT);
            translation_units[axis] =
                translation_ticks[axis].div_euclid(POSITION_TICKS_PER_GRID_UNIT);
            position_tick_offset[axis] = u8::try_from(remainder).unwrap_or_default();
        }
        Self {
            translation_units,
            rotation,
            position_tick_offset,
        }
    }

    /// Creates a build pose from integer eighth-metre centre coordinates.
    ///
    /// Half-grid coordinates are useful for odd-sized cuboids, whose centres
    /// lie halfway between construction-grid lines when resting on a face.
    pub fn from_half_grid(translation_half_units: IVec3, rotation: GridRotation) -> Self {
        Self::from_position_ticks(
            translation_half_units * POSITION_TICKS_PER_HALF_GRID_UNIT,
            rotation,
        )
    }

    /// Translation in exact integer 2.5 mm position ticks.
    pub fn translation_position_ticks(self) -> IVec3 {
        self.translation_units * POSITION_TICKS_PER_GRID_UNIT
            + IVec3::new(
                i32::from(self.position_tick_offset[0]),
                i32::from(self.position_tick_offset[1]),
                i32::from(self.position_tick_offset[2]),
            )
    }

    /// Translation in legacy integer eighth-metre half-grid units.
    ///
    /// # Panics
    ///
    /// Panics when this pose was authored on the finer v8 grid and therefore
    /// cannot be represented by the legacy format without loss.
    pub fn translation_half_units(self) -> IVec3 {
        let ticks = self.translation_position_ticks();
        assert!(
            ticks
                .to_array()
                .into_iter()
                .all(|tick| tick.rem_euclid(POSITION_TICKS_PER_HALF_GRID_UNIT) == 0),
            "a precision-grid pose has no exact half-grid representation"
        );
        ticks / POSITION_TICKS_PER_HALF_GRID_UNIT
    }

    /// Translation in metres.
    pub fn translation(self) -> Vec3 {
        self.translation_position_ticks().as_vec3() * POSITION_TICK_METERS
    }
}

/// Editable cuboid dimensions and build pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CuboidSpec {
    /// Validated x/y/z dimensions.
    pub dimensions: [GridDimension; 3],
    /// Cuboid centre and orientation.
    pub pose: BuildPose,
    /// Material used for appearance, mass, and contact response.
    pub material: ConstructionMaterial,
    /// Independent color and finish treatment.
    pub appearance: MaterialAppearance,
}

/// Invalid load-bearing cylinder dimensions.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum CylinderDimensionError {
    /// The outer diameter was not finite.
    #[error("cylinder outer diameter must be finite")]
    NonFiniteOuterDiameter,
    /// The outer diameter was outside the supported range.
    #[error("cylinder outer diameter must be between 0.05 m and 8.00 m")]
    OuterDiameterOutOfRange,
    /// The inner diameter was not finite.
    #[error("cylinder inner diameter must be finite")]
    NonFiniteInnerDiameter,
    /// The inner diameter was negative or left less than the minimum wall thickness.
    #[error(
        "cylinder inner diameter must be non-negative and at least 0.05 m smaller than the outer diameter"
    )]
    InnerDiameterOutOfRange,
    /// The axial length was not finite.
    #[error("cylinder axial length must be finite")]
    NonFiniteAxialLength,
    /// The axial length was outside the supported range or not a quarter-metre increment.
    #[error("cylinder axial length must be between 0.25 m and 8.00 m in 0.25 m increments")]
    AxialLengthOutOfRange,
    /// The retained angular sector was outside the supported stepped range.
    #[error("cylinder sweep angle must be between 15 and 360 degrees in 15-degree increments")]
    SweepAngleOutOfRange,
}

/// Validated dimensions for a solid or hollow cylinder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CylinderDimensions {
    outer_diameter: f32,
    inner_diameter: f32,
    axial_length: GridDimension,
    sweep_angle_degrees: u16,
}

impl CylinderDimensions {
    /// Default cylinder outer diameter, in metres.
    pub const DEFAULT_OUTER_DIAMETER: f32 = 0.25;
    /// Default cylinder inner diameter, in metres.
    pub const DEFAULT_INNER_DIAMETER: f32 = 0.0;
    /// Default cylinder axial length, in metres.
    pub const DEFAULT_AXIAL_LENGTH: f32 = 0.25;

    /// Creates validated cylinder dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`CylinderDimensionError`] when a diameter is non-finite or out
    /// of range, or when the length is not a supported quarter-metre increment.
    pub fn new(
        outer_diameter: f32,
        inner_diameter: f32,
        axial_length: f32,
    ) -> Result<Self, CylinderDimensionError> {
        if !outer_diameter.is_finite() {
            return Err(CylinderDimensionError::NonFiniteOuterDiameter);
        }
        if !(MIN_CYLINDER_OUTER_DIAMETER..=MAX_CYLINDER_OUTER_DIAMETER).contains(&outer_diameter) {
            return Err(CylinderDimensionError::OuterDiameterOutOfRange);
        }
        if !inner_diameter.is_finite() {
            return Err(CylinderDimensionError::NonFiniteInnerDiameter);
        }
        if inner_diameter < 0.0 || inner_diameter > outer_diameter - MIN_CYLINDER_DIAMETER_GAP {
            return Err(CylinderDimensionError::InnerDiameterOutOfRange);
        }
        if !axial_length.is_finite() {
            return Err(CylinderDimensionError::NonFiniteAxialLength);
        }
        let length_units = axial_length / GRID_UNIT_METERS;
        let rounded_units = length_units.round();
        if (length_units - rounded_units).abs() > 1.0e-5
            || !(1.0..=f32::from(MAX_GRID_UNITS)).contains(&rounded_units)
        {
            return Err(CylinderDimensionError::AxialLengthOutOfRange);
        }
        let units = (1..=MAX_GRID_UNITS)
            .find(|&units| (f32::from(units) - rounded_units).abs() < 1.0e-5)
            .ok_or(CylinderDimensionError::AxialLengthOutOfRange)?;
        let axial_length =
            GridDimension::new(units).map_err(|_| CylinderDimensionError::AxialLengthOutOfRange)?;
        Ok(Self {
            outer_diameter,
            inner_diameter,
            axial_length,
            sweep_angle_degrees: MAX_CYLINDER_SWEEP_DEGREES,
        })
    }

    /// Sets the retained angular sector centred on local positive X.
    ///
    /// # Errors
    ///
    /// Returns [`CylinderDimensionError::SweepAngleOutOfRange`] unless the
    /// angle is between 15 and 360 degrees in 15-degree increments.
    pub const fn with_sweep_angle_degrees(
        mut self,
        sweep_angle_degrees: u16,
    ) -> Result<Self, CylinderDimensionError> {
        if sweep_angle_degrees < MIN_CYLINDER_SWEEP_DEGREES
            || sweep_angle_degrees > MAX_CYLINDER_SWEEP_DEGREES
            || !sweep_angle_degrees.is_multiple_of(CYLINDER_SWEEP_STEP_DEGREES)
        {
            return Err(CylinderDimensionError::SweepAngleOutOfRange);
        }
        self.sweep_angle_degrees = sweep_angle_degrees;
        Ok(self)
    }

    /// Outer diameter in metres.
    pub const fn outer_diameter(self) -> f32 {
        self.outer_diameter
    }

    /// Inner diameter in metres. Zero represents a solid cylinder.
    pub const fn inner_diameter(self) -> f32 {
        self.inner_diameter
    }

    /// Axial length in metres.
    pub fn axial_length(self) -> f32 {
        self.axial_length.meters()
    }

    /// Axial length in quarter-metre grid units.
    pub const fn axial_length_units(self) -> u8 {
        self.axial_length.units()
    }

    /// Retained angular sector in degrees, centred on local positive X.
    pub const fn sweep_angle_degrees(self) -> u16 {
        self.sweep_angle_degrees
    }

    /// Retained angular sector in radians.
    pub fn sweep_angle_radians(self) -> f32 {
        f32::from(self.sweep_angle_degrees).to_radians()
    }
}

impl Default for CylinderDimensions {
    fn default() -> Self {
        Self {
            outer_diameter: Self::DEFAULT_OUTER_DIAMETER,
            inner_diameter: Self::DEFAULT_INNER_DIAMETER,
            axial_length: GridDimension(1),
            sweep_angle_degrees: MAX_CYLINDER_SWEEP_DEGREES,
        }
    }
}

/// Editable cylinder dimensions and build pose. Its axis is local Y.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CylinderSpec {
    /// Validated solid or hollow dimensions.
    pub dimensions: CylinderDimensions,
    /// Cylinder centre and cardinal orientation.
    pub pose: BuildPose,
    /// Material used for appearance, mass, and contact response.
    pub material: ConstructionMaterial,
    /// Independent color and finish treatment.
    pub appearance: MaterialAppearance,
}

impl CylinderSpec {
    /// Creates a cylinder from validated dimensions and a build pose.
    pub const fn new(dimensions: CylinderDimensions, pose: BuildPose) -> Self {
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
    /// The centreline radius was not finite.
    #[error("pipe bend centreline radius must be finite")]
    NonFiniteRadius,
    /// The radius was outside the grid range or not a quarter-metre increment.
    #[error("pipe bend centreline radius must be between 0.25 m and 8.00 m in 0.25 m increments")]
    RadiusOutOfRange,
    /// The centreline radius would fold the outer wall through the bend.
    #[error(
        "pipe bend radius must be at least one block and the outer diameter rounded up to a block"
    )]
    RadiusTooSmallForDiameter,
}

/// Validated cross-section and centreline radius for a cardinal quarter-torus.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PipeBendDimensions {
    outer_diameter: f32,
    inner_diameter: f32,
    radius: GridDimension,
}

impl PipeBendDimensions {
    /// Default centreline radius: one construction block.
    pub const DEFAULT_RADIUS: f32 = GRID_UNIT_METERS;

    /// Creates validated pipe-bend dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`PipeBendDimensionError`] when the annular cross-section is
    /// invalid, the radius is not on the construction grid, or the radius is
    /// shorter than the outer diameter rounded up to one block.
    pub fn new(
        outer_diameter: f32,
        inner_diameter: f32,
        radius: f32,
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
        if !radius.is_finite() {
            return Err(PipeBendDimensionError::NonFiniteRadius);
        }
        let radius_units = radius / GRID_UNIT_METERS;
        let rounded_units = radius_units.round();
        if (radius_units - rounded_units).abs() > 1.0e-5
            || !(1.0..=f32::from(MAX_GRID_UNITS)).contains(&rounded_units)
        {
            return Err(PipeBendDimensionError::RadiusOutOfRange);
        }
        let units = (1..=MAX_GRID_UNITS)
            .find(|&units| (f32::from(units) - rounded_units).abs() < 1.0e-5)
            .ok_or(PipeBendDimensionError::RadiusOutOfRange)?;
        let minimum_units = (outer_diameter / GRID_UNIT_METERS).ceil().max(1.0);
        if f32::from(units) < minimum_units {
            return Err(PipeBendDimensionError::RadiusTooSmallForDiameter);
        }
        Ok(Self {
            outer_diameter,
            inner_diameter,
            radius: GridDimension::new(units)
                .map_err(|_| PipeBendDimensionError::RadiusOutOfRange)?,
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

    /// Centreline radius in metres.
    pub fn radius(self) -> f32 {
        self.radius.meters()
    }

    /// Centreline radius in quarter-metre grid units.
    pub const fn radius_units(self) -> u8 {
        self.radius.units()
    }

    /// Minimum valid radius for `outer_diameter`, rounded up to a block.
    pub fn minimum_radius(outer_diameter: f32) -> f32 {
        (outer_diameter / GRID_UNIT_METERS).ceil().max(1.0) * GRID_UNIT_METERS
    }
}

impl Default for PipeBendDimensions {
    fn default() -> Self {
        Self {
            outer_diameter: CylinderDimensions::DEFAULT_OUTER_DIAMETER,
            inner_diameter: CylinderDimensions::DEFAULT_INNER_DIAMETER,
            radius: GridDimension(1),
        }
    }
}

/// Cardinal 90-degree pipe bend with local negative-X inlet and positive-Y outlet.
///
/// The pose translation is the theoretical sharp corner. The centreline is
/// tangent at `(-radius, 0, 0)` and `(0, radius, 0)` in local space.
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

/// Editable control block. Its shape is a fixed 2×2×1-grid-unit cuboid; what it
/// does lives on the drive links wired from it, one program per bearing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControllerSpec {
    /// Control-block centre and cardinal orientation.
    pub pose: BuildPose,
}

impl ControllerSpec {
    /// Fixed local x/y/z side lengths in grid units.
    pub const GRID_UNITS: [u8; 3] = [2, 2, 1];

    /// Creates a control block with the given pose.
    pub const fn new(pose: BuildPose) -> Self {
        Self { pose }
    }

    /// Fixed cuboid shape backing every control block.
    ///
    /// # Panics
    ///
    /// Never in practice: the fixed side length is a valid grid dimension.
    pub fn cuboid(self) -> CuboidSpec {
        CuboidSpec::new(Self::GRID_UNITS, self.pose)
            .expect("the fixed control-block dimensions are valid")
    }
}

/// Authored engine appearance and future behaviour family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EngineKind {
    /// Combustion engine with a fixed 2×2×3-grid-unit envelope.
    Gas,
    /// Electric engine with a fixed 2×2×2-grid-unit envelope.
    Electric,
}

impl EngineKind {
    /// Fixed local x/y/z side lengths in grid units.
    pub const fn grid_units(self) -> [u8; 3] {
        match self {
            Self::Gas => [2, 2, 3],
            Self::Electric => [2, 2, 2],
        }
    }

    /// Stall torque supplied by one engine, in newton metres.
    pub const fn stall_torque_newton_meters(self) -> f32 {
        match self {
            Self::Gas => 200.0,
            Self::Electric => 500.0,
        }
    }

    /// No-load shaft speed supplied by one engine, in revolutions per minute.
    pub const fn no_load_rpm(self) -> f32 {
        match self {
            Self::Gas => 220.0,
            Self::Electric => 120.0,
        }
    }

    /// Number of physical bearing coordinates one engine can feed.
    pub const fn bearing_capacity(self) -> u32 {
        4
    }
}

/// Fixed-size engine part.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineSpec {
    /// Which authored engine this part represents.
    pub kind: EngineKind,
    /// Engine centre and cardinal orientation.
    pub pose: BuildPose,
}

/// Fixed-size servo angle actuator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ServoSpec {
    /// Servo centre and cardinal orientation.
    pub pose: BuildPose,
}

impl ServoSpec {
    /// Fixed local x/y/z side lengths in grid units.
    pub const GRID_UNITS: [u8; 3] = [1, 1, 1];
    /// Stall torque supplied by one servo, in newton metres.
    pub const STALL_TORQUE_NEWTON_METERS: f32 = 150.0;
    /// Maximum servo motion in revolutions per minute.
    pub const NO_LOAD_RPM: f32 = 30.0;

    /// Creates a servo with the given pose.
    pub const fn new(pose: BuildPose) -> Self {
        Self { pose }
    }

    /// Fixed cuboid envelope backing the servo.
    ///
    /// # Panics
    ///
    /// Never: the fixed dimensions are valid grid dimensions.
    pub fn cuboid(self) -> CuboidSpec {
        CuboidSpec::new(Self::GRID_UNITS, self.pose).expect("servo dimensions are valid")
    }
}

/// Fixed-size seat cushion. Local positive Y is up and positive Z is forward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeatSpec {
    /// Seat centre and cardinal orientation.
    pub pose: BuildPose,
}

impl SeatSpec {
    /// Two-by-two footprint and one-grid-unit cushion height.
    pub const GRID_UNITS: [u8; 3] = [2, 1, 2];

    /// Creates a seat cushion with the given pose.
    pub const fn new(pose: BuildPose) -> Self {
        Self { pose }
    }

    /// Fixed cuboid envelope backing the seat.
    ///
    /// # Panics
    ///
    /// Never: the fixed dimensions are valid grid dimensions.
    pub fn cuboid(self) -> CuboidSpec {
        CuboidSpec::new(Self::GRID_UNITS, self.pose).expect("seat dimensions are valid")
    }
}

/// Fixed-size keyboard input router.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InputSpec {
    /// Input centre and cardinal orientation.
    pub pose: BuildPose,
}

/// Fixed-size portal anchor used to move one structural assembly to and from a Garage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DimensionLinkSpec {
    /// Stable per-world identity retained when the assembly changes spaces.
    pub id: DimensionLinkId,
    /// Link centre and cardinal orientation.
    pub pose: BuildPose,
}

impl DimensionLinkSpec {
    /// Fixed local x/y/z side lengths in grid units (50 × 25 × 25 cm).
    pub const GRID_UNITS: [u8; 3] = [2, 1, 1];

    /// Creates a Dimension Link with a stable per-world identity.
    pub const fn new(id: DimensionLinkId, pose: BuildPose) -> Self {
        Self { id, pose }
    }

    /// Fixed collision and placement envelope backing every Dimension Link.
    ///
    /// # Panics
    ///
    /// Never panics because the fixed dimensions are valid grid dimensions.
    pub fn cuboid(self) -> CuboidSpec {
        CuboidSpec::new(Self::GRID_UNITS, self.pose).expect("Dimension Link dimensions are valid")
    }
}

impl InputSpec {
    /// Fixed local x/y/z side lengths in grid units.
    pub const GRID_UNITS: [u8; 3] = [2, 1, 1];

    /// Creates an input with the given pose.
    pub const fn new(pose: BuildPose) -> Self {
        Self { pose }
    }

    /// Fixed cuboid envelope backing the input.
    ///
    /// # Panics
    ///
    /// Never: the fixed dimensions are valid grid dimensions.
    pub fn cuboid(self) -> CuboidSpec {
        CuboidSpec::new(Self::GRID_UNITS, self.pose).expect("input dimensions are valid")
    }
}

impl EngineSpec {
    /// Creates an engine of `kind` with the given pose.
    pub const fn new(kind: EngineKind, pose: BuildPose) -> Self {
        Self { kind, pose }
    }

    /// Fixed cuboid shape backing this engine kind.
    ///
    /// # Panics
    ///
    /// Never in practice: both fixed engine envelopes use valid dimensions.
    pub fn cuboid(self) -> CuboidSpec {
        CuboidSpec::new(self.kind.grid_units(), self.pose)
            .expect("the fixed engine dimensions are valid")
    }
}

/// Fixed-size transmission block. Its appearance is derived from its root engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransmissionSpec {
    /// Transmission centre and inherited engine orientation.
    pub pose: BuildPose,
}

impl TransmissionSpec {
    /// Fixed local x/y/z side lengths in grid units.
    pub const GRID_UNITS: [u8; 3] = [2, 2, 1];

    /// Creates a transmission at the supplied candidate pose.
    pub const fn new(pose: BuildPose) -> Self {
        Self { pose }
    }

    /// Fixed cuboid envelope backing every transmission.
    ///
    /// # Panics
    ///
    /// Never in practice: the fixed transmission envelope uses valid dimensions.
    pub fn cuboid(self) -> CuboidSpec {
        CuboidSpec::new(Self::GRID_UNITS, self.pose).expect("transmission dimensions are valid")
    }
}

/// A construction part with shape-specific dimensions and a shared build pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PartSpec {
    /// Rectangular cuboid.
    Cuboid(CuboidSpec),
    /// Solid or hollow cylinder whose axis is local Y.
    Cylinder(CylinderSpec),
    /// Cardinal 90-degree quarter-torus pipe bend.
    PipeBend(PipeBendSpec),
    /// Fixed-size control block driving the bearings wired to it.
    Controller(ControllerSpec),
    /// Fixed-size inert engine with an authored appearance.
    Engine(EngineSpec),
    /// Fixed-size transmission whose appearance comes from its root engine.
    Transmission(TransmissionSpec),
    /// Fixed-size servo angle actuator.
    Servo(ServoSpec),
    /// Fixed-size seat cushion.
    Seat(SeatSpec),
    /// Fixed-size keyboard input router.
    Input(InputSpec),
    /// Fixed-size Dimension Link portal anchor.
    DimensionLink(DimensionLinkSpec),
}

impl PartSpec {
    /// Part build pose.
    pub const fn pose(self) -> BuildPose {
        match self {
            Self::Cuboid(spec) => spec.pose,
            Self::Cylinder(spec) => spec.pose,
            Self::PipeBend(spec) => spec.pose,
            Self::Controller(spec) => spec.pose,
            Self::Engine(spec) => spec.pose,
            Self::Transmission(spec) => spec.pose,
            Self::Servo(spec) => spec.pose,
            Self::Seat(spec) => spec.pose,
            Self::Input(spec) => spec.pose,
            Self::DimensionLink(spec) => spec.pose,
        }
    }

    /// Construction appearance, or `None` for authored machine parts.
    pub const fn appearance(self) -> Option<MaterialAppearance> {
        match self {
            Self::Cuboid(spec) => Some(spec.appearance),
            Self::Cylinder(spec) => Some(spec.appearance),
            Self::PipeBend(spec) => Some(spec.appearance),
            Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => None,
        }
    }

    /// Returns an ordinary construction part with a replacement appearance.
    pub(crate) const fn with_appearance(self, appearance: MaterialAppearance) -> Option<Self> {
        match self {
            Self::Cuboid(spec) => Some(Self::Cuboid(spec.with_appearance(appearance))),
            Self::Cylinder(spec) => Some(Self::Cylinder(spec.with_appearance(appearance))),
            Self::PipeBend(spec) => Some(Self::PipeBend(spec.with_appearance(appearance))),
            Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => None,
        }
    }

    /// Returns this part with a replacement authored pose.
    #[must_use]
    pub const fn with_pose(self, pose: BuildPose) -> Self {
        match self {
            Self::Cuboid(mut spec) => {
                spec.pose = pose;
                Self::Cuboid(spec)
            }
            Self::Cylinder(mut spec) => {
                spec.pose = pose;
                Self::Cylinder(spec)
            }
            Self::PipeBend(mut spec) => {
                spec.pose = pose;
                Self::PipeBend(spec)
            }
            Self::Controller(mut spec) => {
                spec.pose = pose;
                Self::Controller(spec)
            }
            Self::Engine(mut spec) => {
                spec.pose = pose;
                Self::Engine(spec)
            }
            Self::Transmission(mut spec) => {
                spec.pose = pose;
                Self::Transmission(spec)
            }
            Self::Servo(mut spec) => {
                spec.pose = pose;
                Self::Servo(spec)
            }
            Self::Seat(mut spec) => {
                spec.pose = pose;
                Self::Seat(spec)
            }
            Self::Input(mut spec) => {
                spec.pose = pose;
                Self::Input(spec)
            }
            Self::DimensionLink(mut spec) => {
                spec.pose = pose;
                Self::DimensionLink(spec)
            }
        }
    }

    /// Returns the cuboid shape backing this part, when it has one. Control
    /// blocks report their fixed cube.
    pub fn as_cuboid(self) -> Option<CuboidSpec> {
        match self {
            Self::Cuboid(spec) => Some(spec),
            Self::Controller(spec) => Some(spec.cuboid()),
            Self::Engine(spec) => Some(spec.cuboid()),
            Self::Transmission(spec) => Some(spec.cuboid()),
            Self::Servo(spec) => Some(spec.cuboid()),
            Self::Seat(spec) => Some(spec.cuboid()),
            Self::Input(spec) => Some(spec.cuboid()),
            Self::DimensionLink(spec) => Some(spec.cuboid()),
            Self::Cylinder(_) | Self::PipeBend(_) => None,
        }
    }

    /// Returns the control-block shape, when this part is a control block.
    pub const fn as_controller(self) -> Option<ControllerSpec> {
        match self {
            Self::Controller(spec) => Some(spec),
            Self::Cuboid(_)
            | Self::Cylinder(_)
            | Self::PipeBend(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => None,
        }
    }

    /// Returns the cylinder shape, when this part is a cylinder.
    pub const fn as_cylinder(self) -> Option<CylinderSpec> {
        match self {
            Self::Cylinder(spec) => Some(spec),
            Self::Cuboid(_)
            | Self::PipeBend(_)
            | Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => None,
        }
    }

    /// Returns the pipe-bend shape, when this part is a bend.
    pub const fn as_pipe_bend(self) -> Option<PipeBendSpec> {
        match self {
            Self::PipeBend(spec) => Some(spec),
            Self::Cuboid(_)
            | Self::Cylinder(_)
            | Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => None,
        }
    }

    /// Axis-aligned local dimensions. Cylinders return diameter/length/diameter.
    pub fn size_meters(self) -> Vec3 {
        match self {
            Self::Cuboid(spec) => spec.size_meters(),
            Self::Controller(spec) => spec.cuboid().size_meters(),
            Self::Engine(spec) => spec.cuboid().size_meters(),
            Self::Transmission(spec) => spec.cuboid().size_meters(),
            Self::Servo(spec) => spec.cuboid().size_meters(),
            Self::Seat(spec) => spec.cuboid().size_meters(),
            Self::Input(spec) => spec.cuboid().size_meters(),
            Self::DimensionLink(spec) => spec.cuboid().size_meters(),
            Self::Cylinder(spec) => Vec3::new(
                spec.dimensions.outer_diameter(),
                spec.dimensions.axial_length(),
                spec.dimensions.outer_diameter(),
            ),
            Self::PipeBend(spec) => {
                let outer = spec.dimensions.outer_diameter();
                let radius = spec.dimensions.radius();
                Vec3::new(radius + outer * 0.5, radius + outer * 0.5, outer)
            }
        }
    }
}

impl PartialEq<CuboidSpec> for PartSpec {
    fn eq(&self, other: &CuboidSpec) -> bool {
        matches!(self, Self::Cuboid(spec) if spec == other)
    }
}

impl From<CuboidSpec> for PartSpec {
    fn from(value: CuboidSpec) -> Self {
        Self::Cuboid(value)
    }
}

impl From<CylinderSpec> for PartSpec {
    fn from(value: CylinderSpec) -> Self {
        Self::Cylinder(value)
    }
}

impl From<PipeBendSpec> for PartSpec {
    fn from(value: PipeBendSpec) -> Self {
        Self::PipeBend(value)
    }
}

impl From<ControllerSpec> for PartSpec {
    fn from(value: ControllerSpec) -> Self {
        Self::Controller(value)
    }
}

impl From<EngineSpec> for PartSpec {
    fn from(value: EngineSpec) -> Self {
        Self::Engine(value)
    }
}

impl From<TransmissionSpec> for PartSpec {
    fn from(value: TransmissionSpec) -> Self {
        Self::Transmission(value)
    }
}

impl From<ServoSpec> for PartSpec {
    fn from(value: ServoSpec) -> Self {
        Self::Servo(value)
    }
}

impl From<SeatSpec> for PartSpec {
    fn from(value: SeatSpec) -> Self {
        Self::Seat(value)
    }
}

impl From<InputSpec> for PartSpec {
    fn from(value: InputSpec) -> Self {
        Self::Input(value)
    }
}

impl From<DimensionLinkSpec> for PartSpec {
    fn from(value: DimensionLinkSpec) -> Self {
        Self::DimensionLink(value)
    }
}

impl CuboidSpec {
    /// Creates a cuboid from integer quarter-metre dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`DimensionError`] when any dimension is outside 1..=32.
    pub fn new(dimensions: [u8; 3], pose: BuildPose) -> Result<Self, DimensionError> {
        let [x, y, z] = dimensions;
        Ok(Self {
            dimensions: [
                GridDimension::new(x)?,
                GridDimension::new(y)?,
                GridDimension::new(z)?,
            ],
            pose,
            material: ConstructionMaterial::Steel,
            appearance: MaterialAppearance::BAKED,
        })
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

    /// Cuboid side lengths in metres.
    pub fn size_meters(self) -> Vec3 {
        Vec3::new(
            self.dimensions[0].meters(),
            self.dimensions[1].meters(),
            self.dimensions[2].meters(),
        )
    }
}

/// One of the six oriented local faces. Cylinders expose only their Y ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FaceKind {
    /// Positive local x face.
    PositiveX,
    /// Negative local x face.
    NegativeX,
    /// Positive local y face.
    PositiveY,
    /// Negative local y face.
    NegativeY,
    /// Positive local z face.
    PositiveZ,
    /// Negative local z face.
    NegativeZ,
}

impl FaceKind {
    /// Face with the opposite local normal.
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Self::PositiveX => Self::NegativeX,
            Self::NegativeX => Self::PositiveX,
            Self::PositiveY => Self::NegativeY,
            Self::NegativeY => Self::PositiveY,
            Self::PositiveZ => Self::NegativeZ,
            Self::NegativeZ => Self::PositiveZ,
        }
    }

    pub(crate) const fn axis(self) -> Axis {
        match self {
            Self::PositiveX | Self::NegativeX => Axis::X,
            Self::PositiveY | Self::NegativeY => Axis::Y,
            Self::PositiveZ | Self::NegativeZ => Axis::Z,
        }
    }

    pub(crate) const fn sign(self) -> f32 {
        match self {
            Self::PositiveX | Self::PositiveY | Self::PositiveZ => 1.0,
            Self::NegativeX | Self::NegativeY | Self::NegativeZ => -1.0,
        }
    }

    pub(crate) const fn tangent_axes(self) -> (Axis, Axis) {
        match self.axis() {
            Axis::X => (Axis::Y, Axis::Z),
            Axis::Y => (Axis::X, Axis::Z),
            Axis::Z => (Axis::X, Axis::Y),
        }
    }
}

/// Object owning a selectable face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FaceOwner {
    /// A user-created construction part.
    Part(PartId),
    /// The central static ground plane. Only its positive-y face is valid.
    Ground,
}

/// Stable reference to a part or ground face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaceRef {
    /// Face owner.
    pub owner: FaceOwner,
    /// Oriented face on that owner.
    pub face: FaceKind,
    /// Stable evaluated planar patch. `None` names the primitive face and lets
    /// feature evaluation resolve its trimmed remainder.
    pub patch: Option<crate::SurfacePatchKey>,
}

impl FaceRef {
    /// Creates a part-face reference.
    pub const fn part(part: PartId, face: FaceKind) -> Self {
        Self {
            owner: FaceOwner::Part(part),
            face,
            patch: None,
        }
    }

    /// Creates a reference to one evaluated planar surface patch.
    pub const fn patch(part: PartId, face: FaceKind, patch: crate::SurfacePatchKey) -> Self {
        Self {
            owner: FaceOwner::Part(part),
            face,
            patch: Some(patch),
        }
    }

    /// The ground's upward-facing plane.
    pub const fn ground() -> Self {
        Self {
            owner: FaceOwner::Ground,
            face: FaceKind::PositiveY,
            patch: None,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FaceGeometry {
    pub(crate) center: Vec3,
    pub(crate) normal: Vec3,
    pub(crate) tangent_u: Vec3,
    pub(crate) tangent_v: Vec3,
    pub(crate) profile: FaceProfile,
}

#[derive(Clone, Debug)]
pub(crate) enum FaceProfile {
    Rectangle {
        half_u: f32,
        half_v: f32,
    },
    Annulus {
        inner_radius: f32,
        outer_radius: f32,
    },
    AnnularSector {
        inner_radius: f32,
        outer_radius: f32,
        half_angle: f32,
    },
    Polygon {
        vertices: Vec<Vec2>,
    },
    Ground,
}

pub(crate) fn cuboid_face(spec: CuboidSpec, face: FaceKind) -> FaceGeometry {
    let rotation = spec.pose.rotation.quaternion();
    let size = spec.size_meters();
    let axis = face.axis();
    let (u_axis, v_axis) = face.tangent_axes();
    let normal = snap_cardinal(rotation * axis.unit()) * face.sign();
    let tangent_u = snap_cardinal(rotation * u_axis.unit());
    let tangent_v = snap_cardinal(rotation * v_axis.unit());
    FaceGeometry {
        center: spec.pose.translation() + normal * size[axis.index()] * 0.5,
        normal,
        tangent_u,
        tangent_v,
        profile: FaceProfile::Rectangle {
            half_u: size[u_axis.index()] * 0.5,
            half_v: size[v_axis.index()] * 0.5,
        },
    }
}

pub(crate) fn cylinder_face(spec: CylinderSpec, face: FaceKind) -> Option<FaceGeometry> {
    if !matches!(face, FaceKind::PositiveY | FaceKind::NegativeY) {
        return None;
    }
    let rotation = spec.pose.rotation.quaternion();
    let normal = snap_cardinal(rotation * Vec3::Y) * face.sign();
    let profile = if spec.dimensions.sweep_angle_degrees() == MAX_CYLINDER_SWEEP_DEGREES {
        FaceProfile::Annulus {
            inner_radius: spec.dimensions.inner_diameter() * 0.5,
            outer_radius: spec.dimensions.outer_diameter() * 0.5,
        }
    } else {
        FaceProfile::AnnularSector {
            inner_radius: spec.dimensions.inner_diameter() * 0.5,
            outer_radius: spec.dimensions.outer_diameter() * 0.5,
            half_angle: spec.dimensions.sweep_angle_radians() * 0.5,
        }
    };
    Some(FaceGeometry {
        center: spec.pose.translation() + normal * spec.dimensions.axial_length() * 0.5,
        normal,
        tangent_u: snap_cardinal(rotation * Vec3::X),
        tangent_v: snap_cardinal(rotation * Vec3::Z),
        profile,
    })
}

pub(crate) fn pipe_bend_face(spec: PipeBendSpec, face: FaceKind) -> Option<FaceGeometry> {
    let (local_center, local_normal, local_u, local_v) = match face {
        FaceKind::NegativeX => (
            Vec3::new(-spec.dimensions.radius(), 0.0, 0.0),
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::Z,
        ),
        FaceKind::PositiveY => (
            Vec3::new(0.0, spec.dimensions.radius(), 0.0),
            Vec3::Y,
            Vec3::X,
            Vec3::Z,
        ),
        _ => return None,
    };
    let rotation = spec.pose.rotation.quaternion();
    Some(FaceGeometry {
        center: spec.pose.translation() + rotation * local_center,
        normal: snap_cardinal(rotation * local_normal),
        tangent_u: snap_cardinal(rotation * local_u),
        tangent_v: snap_cardinal(rotation * local_v),
        profile: FaceProfile::Annulus {
            inner_radius: spec.dimensions.inner_diameter() * 0.5,
            outer_radius: spec.dimensions.outer_diameter() * 0.5,
        },
    })
}

pub(crate) const fn ground_face() -> FaceGeometry {
    FaceGeometry {
        center: Vec3::ZERO,
        normal: Vec3::Y,
        tangent_u: Vec3::X,
        tangent_v: Vec3::Z,
        profile: FaceProfile::Ground,
    }
}

fn snap_cardinal(vector: Vec3) -> Vec3 {
    Vec3::new(vector.x.round(), vector.y.round(), vector.z.round())
}

#[cfg(test)]
mod tests {
    use bevy_math::{IVec3, Vec3};

    use super::{
        BuildPose, ConstructionMaterial, ControllerSpec, CuboidSpec, CylinderDimensionError,
        CylinderDimensions, CylinderSpec, DimensionLinkId, DimensionLinkSpec, EngineKind,
        EngineSpec, FaceKind, GridRotation, InputSpec, PipeBendDimensionError, PipeBendDimensions,
        PipeBendSpec, SeatSpec, ServoSpec, cuboid_face, pipe_bend_face, snap_world_to_grid,
    };

    #[test]
    fn snaps_world_coordinates_to_quarter_metre_grid() {
        assert_eq!(
            snap_world_to_grid(Vec3::new(0.37, -0.13, 1.99)),
            IVec3::new(1, -1, 8)
        );
    }

    #[test]
    fn dimension_link_has_a_fixed_two_by_one_by_one_block_envelope() {
        let pose =
            BuildPose::from_position_ticks(IVec3::new(7, 200, -3), GridRotation::new(0, 1, 0));
        let link = DimensionLinkSpec::new(DimensionLinkId(42), pose);
        assert_eq!(link.id, DimensionLinkId(42));
        assert_eq!(
            link.cuboid().dimensions.map(super::GridDimension::units),
            [2, 1, 1]
        );
        assert_eq!(link.cuboid().pose, pose);
        assert!(
            link.cuboid()
                .size_meters()
                .abs_diff_eq(Vec3::new(0.5, 0.25, 0.25), 1.0e-6)
        );
    }

    #[test]
    fn half_grid_pose_preserves_eighth_metre_centres_in_both_directions() {
        let pose = BuildPose::from_half_grid(IVec3::new(-3, 1, 4), GridRotation::default());

        assert_eq!(pose.translation_half_units(), IVec3::new(-3, 1, 4));
        assert!(
            pose.translation()
                .abs_diff_eq(Vec3::new(-0.375, 0.125, 0.5), 1.0e-6)
        );
        assert_eq!(
            BuildPose::new(IVec3::new(-3, 1, 4), GridRotation::default()).translation_half_units(),
            IVec3::new(-6, 2, 8)
        );
    }

    #[test]
    fn pose_preserves_fine_position_ticks_in_both_directions() {
        let pose =
            BuildPose::from_position_ticks(IVec3::new(-130, 20, 190), GridRotation::new(1, 2, 3));
        assert_eq!(pose.translation_position_ticks(), IVec3::new(-130, 20, 190));
        assert!(
            pose.translation()
                .abs_diff_eq(Vec3::new(-0.325, 0.05, 0.475), 1.0e-6)
        );
    }

    #[test]
    fn one_centimetre_positions_round_trip_exactly_across_zero() {
        for ticks in [IVec3::new(4, -4, 8), IVec3::new(-4, 4, -8)] {
            let pose = BuildPose::from_position_ticks(ticks, GridRotation::default());
            assert_eq!(pose.translation_position_ticks(), ticks);
            assert!(
                pose.translation()
                    .abs_diff_eq(ticks.as_vec3() * 0.0025, 1.0e-7)
            );
        }
    }

    #[test]
    fn rejects_dimensions_outside_builder_range() {
        assert!(CuboidSpec::new([0, 4, 4], BuildPose::default()).is_err());
        assert!(CuboidSpec::new([33, 4, 4], BuildPose::default()).is_err());
        assert!(CuboidSpec::new([1, 32, 1], BuildPose::default()).is_ok());
    }

    #[test]
    fn construction_material_property_rows_are_exact() {
        let expected: [(ConstructionMaterial, [f32; 6]); 12] = [
            (
                ConstructionMaterial::Aluminium,
                [2_700.0, 0.61, 0.47, 0.25, 0.004, 69.0e9],
            ),
            (
                ConstructionMaterial::CarbonFiber,
                [1_600.0, 0.40, 0.30, 0.20, 0.008, 70.0e9],
            ),
            (
                ConstructionMaterial::Concrete,
                [2_400.0, 0.80, 0.65, 0.05, 0.020, 30.0e9],
            ),
            (
                ConstructionMaterial::Dirt,
                [1_600.0, 0.72, 0.55, 0.05, 0.030, 0.05e9],
            ),
            (
                ConstructionMaterial::Graphite,
                [1_900.0, 0.25, 0.15, 0.10, 0.010, 12.0e9],
            ),
            (
                ConstructionMaterial::Iron,
                [7_870.0, 0.70, 0.55, 0.15, 0.003, 170.0e9],
            ),
            (
                ConstructionMaterial::Plastic,
                [950.0, 0.40, 0.30, 0.40, 0.020, 1.0e9],
            ),
            (
                ConstructionMaterial::Rubber,
                [1_100.0, 1.00, 0.80, 0.70, 0.040, 0.01e9],
            ),
            (
                ConstructionMaterial::Sand,
                [1_700.0, 0.65, 0.50, 0.05, 0.035, 0.03e9],
            ),
            (
                ConstructionMaterial::Steel,
                [7_850.0, 0.74, 0.57, 0.20, 0.002, 200.0e9],
            ),
            (
                ConstructionMaterial::Stone,
                [2_700.0, 0.60, 0.48, 0.05, 0.015, 50.0e9],
            ),
            (
                ConstructionMaterial::Wood,
                [700.0, 0.48, 0.30, 0.15, 0.025, 10.0e9],
            ),
        ];
        for (material, expected) in expected {
            let properties = material.properties();
            let actual = [
                properties.density_kg_m3,
                properties.static_friction,
                properties.dynamic_friction,
                properties.restitution,
                properties.rolling_resistance,
                properties.youngs_modulus_pa,
            ];
            assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
            assert!(properties.density_kg_m3 > 0.0);
            assert!(properties.static_friction >= properties.dynamic_friction);
            assert!((0.0..=1.0).contains(&properties.dynamic_friction));
            assert!((0.0..=1.0).contains(&properties.restitution));
            assert!((0.0..=1.0).contains(&properties.rolling_resistance));
            assert!(properties.youngs_modulus_pa > 0.0);
        }
    }

    #[test]
    fn nominal_block_compliance_uses_the_construction_scale() {
        let steel = ConstructionMaterial::Steel.properties();
        assert_eq!(
            steel.nominal_block_compliance().to_bits(),
            (1.0 / (200.0e9 * super::GRID_UNIT_METERS)).to_bits(),
        );
        assert!(
            ConstructionMaterial::Rubber
                .properties()
                .nominal_block_compliance()
                > steel.nominal_block_compliance()
        );
    }

    #[test]
    fn ordinary_part_specs_default_to_steel_and_accept_an_explicit_material() {
        let cuboid = CuboidSpec::new([1; 3], BuildPose::default()).unwrap();
        let cylinder = CylinderSpec::new(CylinderDimensions::default(), BuildPose::default());
        assert_eq!(cuboid.material, ConstructionMaterial::Steel);
        assert_eq!(cylinder.material, ConstructionMaterial::Steel);
        assert_eq!(
            cuboid.with_material(ConstructionMaterial::Wood).material,
            ConstructionMaterial::Wood,
        );
        assert_eq!(
            cylinder
                .with_material(ConstructionMaterial::Plastic)
                .material,
            ConstructionMaterial::Plastic,
        );
    }

    #[test]
    fn authored_parts_keep_their_fixed_grid_envelopes() {
        assert_eq!(
            ControllerSpec::new(BuildPose::default())
                .cuboid()
                .dimensions
                .map(super::GridDimension::units),
            [2, 2, 1]
        );
        assert_eq!(
            EngineSpec::new(EngineKind::Gas, BuildPose::default())
                .cuboid()
                .dimensions
                .map(super::GridDimension::units),
            [2, 2, 3]
        );
        assert_eq!(
            EngineSpec::new(EngineKind::Electric, BuildPose::default())
                .cuboid()
                .dimensions
                .map(super::GridDimension::units),
            [2, 2, 2]
        );
        assert_eq!(
            ServoSpec::new(BuildPose::default())
                .cuboid()
                .dimensions
                .map(super::GridDimension::units),
            [1, 1, 1]
        );
        assert_eq!(
            SeatSpec::new(BuildPose::default())
                .cuboid()
                .dimensions
                .map(super::GridDimension::units),
            [2, 1, 2]
        );
        assert_eq!(
            InputSpec::new(BuildPose::default())
                .cuboid()
                .dimensions
                .map(super::GridDimension::units),
            [2, 1, 1]
        );
        for (actual, expected) in [
            (EngineKind::Electric.stall_torque_newton_meters(), 500.0),
            (EngineKind::Electric.no_load_rpm(), 120.0),
            (EngineKind::Gas.stall_torque_newton_meters(), 200.0),
            (EngineKind::Gas.no_load_rpm(), 220.0),
            (ServoSpec::STALL_TORQUE_NEWTON_METERS, 150.0),
            (ServoSpec::NO_LOAD_RPM, 30.0),
        ] {
            assert!((actual - expected).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn cylinder_dimensions_validate_defaults_bounds_wall_and_length_grid() {
        let dimensions = CylinderDimensions::default();
        assert!((dimensions.outer_diameter() - 0.25).abs() < f32::EPSILON);
        assert!(dimensions.inner_diameter().abs() < f32::EPSILON);
        assert!((dimensions.axial_length() - 0.25).abs() < f32::EPSILON);
        assert_eq!(dimensions.sweep_angle_degrees(), 360);
        assert!(CylinderDimensions::new(0.05, 0.0, 0.25).is_ok());
        assert!(CylinderDimensions::new(8.0, 7.95, 8.0).is_ok());
        assert_eq!(
            CylinderDimensions::new(0.049, 0.0, 0.25),
            Err(CylinderDimensionError::OuterDiameterOutOfRange)
        );
        assert_eq!(
            CylinderDimensions::new(0.25, 0.201, 0.25),
            Err(CylinderDimensionError::InnerDiameterOutOfRange)
        );
        assert_eq!(
            CylinderDimensions::new(0.25, 0.0, 0.30),
            Err(CylinderDimensionError::AxialLengthOutOfRange)
        );
        assert_eq!(
            CylinderDimensions::new(0.25, 0.0, 8.25),
            Err(CylinderDimensionError::AxialLengthOutOfRange)
        );
        assert!(
            dimensions
                .with_sweep_angle_degrees(15)
                .is_ok_and(|dimensions| dimensions.sweep_angle_degrees() == 15)
        );
        assert_eq!(
            dimensions.with_sweep_angle_degrees(14),
            Err(CylinderDimensionError::SweepAngleOutOfRange)
        );
        assert_eq!(
            dimensions.with_sweep_angle_degrees(361),
            Err(CylinderDimensionError::SweepAngleOutOfRange)
        );
    }

    #[test]
    fn rotates_face_orientation_in_quarter_turns() {
        let spec = CuboidSpec::new(
            [4, 8, 4],
            BuildPose::new(IVec3::ZERO, GridRotation::new(0, 0, 1)),
        )
        .unwrap();
        let face = cuboid_face(spec, FaceKind::PositiveX);

        assert!(face.normal.abs_diff_eq(Vec3::Y, 1.0e-6));
        assert!(face.center.abs_diff_eq(Vec3::new(0.0, 0.5, 0.0), 1.0e-6));
    }

    #[test]
    fn pipe_bend_dimensions_enforce_grid_radius_and_outer_diameter_clearance() {
        assert!(PipeBendDimensions::new(0.25, 0.10, 0.25).is_ok());
        assert_eq!(
            PipeBendDimensions::new(0.30, 0.10, 0.25),
            Err(PipeBendDimensionError::RadiusTooSmallForDiameter)
        );
        assert!(PipeBendDimensions::new(0.30, 0.10, 0.50).is_ok());
        assert_eq!(
            PipeBendDimensions::new(0.25, 0.21, 0.25),
            Err(PipeBendDimensionError::InnerDiameterOutOfRange)
        );
        assert_eq!(
            PipeBendDimensions::new(0.25, 0.10, 0.30),
            Err(PipeBendDimensionError::RadiusOutOfRange)
        );
    }

    #[test]
    fn pipe_bend_exposes_only_its_two_tangent_annular_ends() {
        let spec = PipeBendSpec::new(
            PipeBendDimensions::new(0.25, 0.10, 0.50).unwrap(),
            BuildPose::new(IVec3::new(4, 8, 0), GridRotation::new(0, 0, 1)),
        );
        let inlet = pipe_bend_face(spec, FaceKind::NegativeX).unwrap();
        let outlet = pipe_bend_face(spec, FaceKind::PositiveY).unwrap();
        assert!(inlet.normal.abs_diff_eq(Vec3::NEG_Y, 1.0e-6));
        assert!(outlet.normal.abs_diff_eq(Vec3::NEG_X, 1.0e-6));
        assert!(pipe_bend_face(spec, FaceKind::PositiveZ).is_none());
        assert!(((inlet.center - spec.pose.translation()).length() - 0.5).abs() < 1.0e-5);
        assert!(((outlet.center - spec.pose.translation()).length() - 0.5).abs() < 1.0e-5);
    }
}
