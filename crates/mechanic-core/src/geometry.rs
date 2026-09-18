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
            Self::Rubber => MaterialProperties::new(1_100.0, 1.00, 0.80, 0.70, 0.010, 0.01e9),
            Self::Sand => MaterialProperties::new(1_700.0, 0.65, 0.50, 0.05, 0.035, 0.03e9),
            Self::Steel => MaterialProperties::new(7_850.0, 0.74, 0.57, 0.20, 0.002, 200.0e9),
            Self::Stone => MaterialProperties::new(2_700.0, 0.60, 0.48, 0.05, 0.015, 50.0e9),
            Self::Wood => MaterialProperties::new(700.0, 0.48, 0.30, 0.15, 0.025, 10.0e9),
        }
    }
}

/// How a surface answers contact: the four coefficients every solver mixes
/// pairwise. Construction materials and terrain materials both resolve to this.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceResponse {
    /// Static Coulomb friction coefficient.
    pub static_friction: f32,
    /// Kinetic Coulomb friction coefficient.
    pub dynamic_friction: f32,
    /// Coefficient of restitution.
    pub restitution: f32,
    /// Dimensionless rolling-resistance coefficient.
    pub rolling_resistance: f32,
}

impl SurfaceResponse {
    /// Static friction, dynamic friction, restitution, and rolling resistance.
    pub const fn to_array(self) -> [f32; 4] {
        [
            self.static_friction,
            self.dynamic_friction,
            self.restitution,
            self.rolling_resistance,
        ]
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

    /// The contact coefficients, without density or stiffness.
    pub const fn surface_response(self) -> SurfaceResponse {
        SurfaceResponse {
            static_friction: self.static_friction,
            dynamic_friction: self.dynamic_friction,
            restitution: self.restitution,
            rolling_resistance: self.rolling_resistance,
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
            .find(|candidate| {
                let rotation = candidate.quaternion();
                rotation.abs_diff_eq(target, 1.0e-5) || rotation.abs_diff_eq(-target, 1.0e-5)
            })
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
///
/// A layered cuboid is one solid. `dimensions` are its grid-aligned core, while
/// `pose` and [`CuboidSpec::size_meters`] describe the whole envelope including
/// every [`MaterialLayer`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CuboidSpec {
    /// Validated x/y/z core dimensions.
    pub dimensions: [GridDimension; 3],
    /// Envelope centre and orientation.
    pub pose: BuildPose,
    /// Core material used for appearance, mass, and contact response.
    pub material: ConstructionMaterial,
    /// Core color and finish treatment.
    pub appearance: MaterialAppearance,
    layers: MaterialLayers,
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
///
/// Authored cylinders have quarter-metre lengths; only end-cap material layers
/// lengthen an envelope off that grid, in whole position ticks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CylinderDimensions {
    outer_diameter: f32,
    inner_diameter: f32,
    axial_length_ticks: u16,
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
        Self::validated(
            outer_diameter,
            inner_diameter,
            u16::from(units) * GRID_UNIT_TICKS,
            MAX_CYLINDER_SWEEP_DEGREES,
        )
    }

    /// Validates a whole envelope, including an off-grid tick length.
    fn validated(
        outer_diameter: f32,
        inner_diameter: f32,
        axial_length_ticks: u16,
        sweep_angle_degrees: u16,
    ) -> Result<Self, CylinderDimensionError> {
        if !outer_diameter.is_finite() {
            return Err(CylinderDimensionError::NonFiniteOuterDiameter);
        }
        if !(MIN_CYLINDER_OUTER_DIAMETER..=MAX_CYLINDER_OUTER_DIAMETER + 1.0e-4)
            .contains(&outer_diameter)
        {
            return Err(CylinderDimensionError::OuterDiameterOutOfRange);
        }
        if !inner_diameter.is_finite() {
            return Err(CylinderDimensionError::NonFiniteInnerDiameter);
        }
        if inner_diameter < 0.0
            || inner_diameter > outer_diameter - MIN_CYLINDER_DIAMETER_GAP + 1.0e-4
        {
            return Err(CylinderDimensionError::InnerDiameterOutOfRange);
        }
        if !(GRID_UNIT_TICKS..=u16::from(MAX_GRID_UNITS) * GRID_UNIT_TICKS)
            .contains(&axial_length_ticks)
        {
            return Err(CylinderDimensionError::AxialLengthOutOfRange);
        }
        Self {
            outer_diameter,
            inner_diameter,
            axial_length_ticks,
            sweep_angle_degrees: MAX_CYLINDER_SWEEP_DEGREES,
        }
        .with_sweep_angle_degrees(sweep_angle_degrees)
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
        f32::from(self.axial_length_ticks) * POSITION_TICK_METERS
    }

    /// Axial length in whole quarter-metre grid units, rounded down.
    #[expect(clippy::cast_possible_truncation, reason = "at most 32 units")]
    pub const fn axial_length_units(self) -> u8 {
        (self.axial_length_ticks / GRID_UNIT_TICKS) as u8
    }

    /// Axial length in position ticks.
    pub const fn axial_length_ticks(self) -> u16 {
        self.axial_length_ticks
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
            axial_length_ticks: GRID_UNIT_TICKS,
            sweep_angle_degrees: MAX_CYLINDER_SWEEP_DEGREES,
        }
    }
}

/// Position ticks in one quarter-metre grid unit.
#[expect(clippy::cast_possible_truncation, reason = "100 ticks")]
const GRID_UNIT_TICKS: u16 = POSITION_TICKS_PER_GRID_UNIT as u16;

/// Most material layers one part carries over its core.
pub const MAX_PART_LAYERS: usize = 4;

/// Thinnest material layer, in metres.
pub const MIN_LAYER_THICKNESS_METERS: f32 = 0.01;

/// Surface a material layer grows from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerFace {
    /// A flat cuboid face or cylinder end cap, growing outward along its normal.
    Face(FaceKind),
    /// A cylinder's curved outer wall, growing its outer diameter.
    OuterWall,
    /// A hollow cylinder's bore, shrinking its inner diameter.
    Bore,
}

/// One material layer laid over a part's envelope at the time it was added.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialLayer {
    /// Surface the layer was laid on.
    pub face: LayerFace,
    /// Thickness in metres, along the face normal or the radius.
    pub thickness: f32,
    /// Material used for appearance, mass, and contact response.
    pub material: ConstructionMaterial,
    /// Independent color and finish treatment.
    pub appearance: MaterialAppearance,
}

impl MaterialLayer {
    const UNUSED: Self = Self {
        face: LayerFace::OuterWall,
        thickness: 0.0,
        material: ConstructionMaterial::Steel,
        appearance: MaterialAppearance::BAKED,
    };
}

/// A part's material layers, oldest first.
///
/// A layered part is still one solid. Its envelope already includes every
/// layer; the layers only divide that envelope's material. The part's own
/// material is band zero, and layer `i` is band `i + 1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialLayers {
    layers: [MaterialLayer; MAX_PART_LAYERS],
    count: u8,
}

impl MaterialLayers {
    /// No layers: the whole part is its own material.
    pub const NONE: Self = Self {
        layers: [MaterialLayer::UNUSED; MAX_PART_LAYERS],
        count: 0,
    };

    /// Number of layers.
    pub const fn len(self) -> usize {
        self.count as usize
    }

    /// Whether there are no layers.
    pub const fn is_empty(self) -> bool {
        self.count == 0
    }

    /// Layer `index`, oldest first.
    pub const fn get(self, index: usize) -> Option<MaterialLayer> {
        if index < self.len() {
            Some(self.layers[index])
        } else {
            None
        }
    }

    /// Every layer, oldest first.
    pub fn iter(self) -> impl DoubleEndedIterator<Item = MaterialLayer> {
        self.layers.into_iter().take(self.len())
    }

    fn pushed(mut self, layer: MaterialLayer) -> Result<Self, LayerError> {
        let len = self.len();
        let slot = self.layers.get_mut(len).ok_or(LayerError::TooManyLayers)?;
        *slot = layer;
        self.count += 1;
        Ok(self)
    }

    fn with_appearance(mut self, index: usize, appearance: MaterialAppearance) -> Option<Self> {
        let len = self.len();
        self.layers[..len].get_mut(index)?.appearance = appearance;
        Some(self)
    }
}

/// Invalid material layer.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum LayerError {
    /// The part already carries the maximum number of layers.
    #[error("a part holds at most {MAX_PART_LAYERS} material layers")]
    TooManyLayers,
    /// The layer is thinner than the minimum or off the 5 mm step of flat layers.
    #[error("a material layer must be at least 1 cm thick, and a flat layer a whole 5 mm step")]
    ThicknessOutOfRange,
    /// A bore layer needs a bore to line.
    #[error("only a hollow cylinder can take a layer inside its bore")]
    BoreRequired,
    /// The part has no such layerable surface.
    #[error("this surface cannot take a material layer")]
    UnsupportedFace,
    /// The layered envelope would outgrow the largest part.
    #[error("a layered part must stay within 8 m")]
    TooLarge,
    /// The layered cylinder envelope is invalid.
    #[error(transparent)]
    Cylinder(#[from] CylinderDimensionError),
}

/// Part-local space owned by one layer: beyond the envelope it was laid on.
///
/// Later layers win where regions overlap, which is exactly the corner a later
/// layer's full face covers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LayerRegion {
    /// Points with `sign * local[axis] > distance`.
    Beyond {
        /// Local axis index.
        axis: usize,
        /// Outward direction along that axis, `1.0` or `-1.0`.
        sign: f32,
        /// Distance of the earlier face from the part centre.
        distance: f32,
    },
    /// Points farther than this radius from the local Y axis.
    OutsideRadius(f32),
    /// Points nearer than this radius to the local Y axis.
    InsideRadius(f32),
}

impl LayerRegion {
    /// Whether a part-local point lies in this layer's region.
    pub fn contains(self, local: Vec3) -> bool {
        match self {
            Self::Beyond {
                axis,
                sign,
                distance,
            } => sign * local[axis] > distance,
            Self::OutsideRadius(radius) => Vec2::new(local.x, local.z).length() > radius,
            Self::InsideRadius(radius) => Vec2::new(local.x, local.z).length() < radius,
        }
    }
}

/// Part-local envelope unwound one layer at a time.
#[derive(Clone, Copy)]
struct LayerEnvelope {
    minimum: Vec3,
    maximum: Vec3,
    outer_radius: f32,
    inner_radius: f32,
}

fn unwind_layer_regions(mut envelope: LayerEnvelope, layers: MaterialLayers) -> Vec<LayerRegion> {
    let mut regions = layers
        .iter()
        .rev()
        .map(|layer| match layer.face {
            LayerFace::Face(face) => {
                let axis = face.axis().index();
                if face.sign() > 0.0 {
                    envelope.maximum[axis] -= layer.thickness;
                    LayerRegion::Beyond {
                        axis,
                        sign: 1.0,
                        distance: envelope.maximum[axis],
                    }
                } else {
                    envelope.minimum[axis] += layer.thickness;
                    LayerRegion::Beyond {
                        axis,
                        sign: -1.0,
                        distance: -envelope.minimum[axis],
                    }
                }
            }
            LayerFace::OuterWall => {
                envelope.outer_radius -= layer.thickness;
                LayerRegion::OutsideRadius(envelope.outer_radius)
            }
            LayerFace::Bore => {
                envelope.inner_radius += layer.thickness;
                LayerRegion::InsideRadius(envelope.inner_radius)
            }
        })
        .collect::<Vec<_>>();
    regions.reverse();
    regions
}

/// Validates a layer thickness; flat layers also return their even tick count,
/// so the half-thickness centre shift stays on the position grid.
#[expect(
    clippy::cast_possible_truncation,
    reason = "checked against the 8 m envelope first"
)]
fn layer_thickness_ticks(thickness: f32, flat: bool) -> Result<i32, LayerError> {
    if !thickness.is_finite() || thickness < MIN_LAYER_THICKNESS_METERS - 1.0e-5 {
        return Err(LayerError::ThicknessOutOfRange);
    }
    if thickness > MAX_CYLINDER_OUTER_DIAMETER {
        return Err(LayerError::TooLarge);
    }
    let ticks = thickness / POSITION_TICK_METERS;
    let rounded = ticks.round();
    if flat && ((ticks - rounded).abs() > 1.0e-3 || rounded as i32 % 2 != 0) {
        return Err(LayerError::ThicknessOutOfRange);
    }
    Ok(rounded as i32)
}

/// Moves a pose along one of its rotated local axes by whole ticks.
#[expect(
    clippy::cast_possible_truncation,
    reason = "cardinal rotations keep whole ticks"
)]
fn shifted_pose(pose: BuildPose, local_ticks: Vec3) -> BuildPose {
    let shift = (pose.rotation.quaternion() * local_ticks).round();
    BuildPose::from_position_ticks(
        pose.translation_position_ticks()
            + IVec3::new(shift.x as i32, shift.y as i32, shift.z as i32),
        pose.rotation,
    )
}

/// Editable cylinder dimensions and build pose. Its axis is local Y.
///
/// A layered cylinder is still one solid: `dimensions` and `pose` describe its
/// whole envelope, and [`MaterialLayers`] only divide that envelope's material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CylinderSpec {
    /// Validated solid or hollow envelope dimensions.
    pub dimensions: CylinderDimensions,
    /// Envelope centre and cardinal orientation.
    pub pose: BuildPose,
    /// Core material used for appearance, mass, and contact response.
    pub material: ConstructionMaterial,
    /// Core color and finish treatment.
    pub appearance: MaterialAppearance,
    layers: MaterialLayers,
}

impl CylinderSpec {
    /// Creates a cylinder from validated dimensions and a build pose.
    pub const fn new(dimensions: CylinderDimensions, pose: BuildPose) -> Self {
        Self {
            dimensions,
            pose,
            material: ConstructionMaterial::Steel,
            appearance: MaterialAppearance::BAKED,
            layers: MaterialLayers::NONE,
        }
    }

    /// Material layers over the core, oldest first.
    pub const fn layers(self) -> MaterialLayers {
        self.layers
    }

    /// Material of the outermost curved wall, which meets the world.
    pub fn outer_contact_material(self) -> ConstructionMaterial {
        self.layers
            .iter()
            .rev()
            .find(|layer| layer.face == LayerFace::OuterWall)
            .map_or(self.material, |layer| layer.material)
    }

    /// Adds a material layer to the outer wall, bore, or one end cap.
    ///
    /// A bore layer thicker than the bore's radius fills it to a solid core. An
    /// end-cap layer lengthens the cylinder and moves its centre outward by half
    /// the thickness, so the opposite cap stays put.
    ///
    /// # Errors
    ///
    /// Returns [`LayerError`] when the thickness, face, layer count, or grown
    /// envelope is invalid.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        reason = "validated tick lengths within 8 m"
    )]
    pub fn with_layer(
        self,
        face: LayerFace,
        thickness: f32,
        material: ConstructionMaterial,
        appearance: MaterialAppearance,
    ) -> Result<Self, LayerError> {
        let dimensions = self.dimensions;
        let layer = |thickness| MaterialLayer {
            face,
            thickness,
            material,
            appearance,
        };
        let validated = |outer, inner, ticks| {
            CylinderDimensions::validated(outer, inner, ticks, dimensions.sweep_angle_degrees)
        };
        let (dimensions, pose, layers) = match face {
            LayerFace::OuterWall => {
                layer_thickness_ticks(thickness, false)?;
                let layers = self.layers.pushed(layer(thickness))?;
                let outer = dimensions.outer_diameter + 2.0 * thickness;
                if outer > MAX_CYLINDER_OUTER_DIAMETER + 1.0e-4 {
                    return Err(LayerError::TooLarge);
                }
                (
                    validated(
                        outer,
                        dimensions.inner_diameter,
                        dimensions.axial_length_ticks,
                    )?,
                    self.pose,
                    layers,
                )
            }
            LayerFace::Bore => {
                layer_thickness_ticks(thickness, false)?;
                if dimensions.inner_diameter <= 0.0 {
                    return Err(LayerError::BoreRequired);
                }
                let inner = (dimensions.inner_diameter - 2.0 * thickness).max(0.0);
                let layers = self
                    .layers
                    .pushed(layer((dimensions.inner_diameter - inner) * 0.5))?;
                (
                    validated(
                        dimensions.outer_diameter,
                        inner,
                        dimensions.axial_length_ticks,
                    )?,
                    self.pose,
                    layers,
                )
            }
            LayerFace::Face(cap @ (FaceKind::PositiveY | FaceKind::NegativeY)) => {
                let ticks = layer_thickness_ticks(thickness, true)?;
                let layers = self
                    .layers
                    .pushed(layer(ticks as f32 * POSITION_TICK_METERS))?;
                let length = i32::from(dimensions.axial_length_ticks) + ticks;
                if length > i32::from(MAX_GRID_UNITS) * POSITION_TICKS_PER_GRID_UNIT {
                    return Err(LayerError::TooLarge);
                }
                (
                    validated(
                        dimensions.outer_diameter,
                        dimensions.inner_diameter,
                        length as u16,
                    )?,
                    shifted_pose(self.pose, Vec3::Y * cap.sign() * (ticks / 2) as f32),
                    layers,
                )
            }
            LayerFace::Face(_) => return Err(LayerError::UnsupportedFace),
        };
        Ok(Self {
            dimensions,
            pose,
            layers,
            ..self
        })
    }

    /// The core this cylinder's layers were laid on.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        reason = "unwinds validated ticks within 8 m"
    )]
    pub fn without_layers(self) -> Self {
        let mut outer = self.dimensions.outer_diameter;
        let mut inner = self.dimensions.inner_diameter;
        let mut length = i32::from(self.dimensions.axial_length_ticks);
        let mut shift = 0;
        for layer in self.layers.iter() {
            match layer.face {
                LayerFace::OuterWall => outer -= 2.0 * layer.thickness,
                LayerFace::Bore => inner += 2.0 * layer.thickness,
                LayerFace::Face(face) => {
                    let ticks = (layer.thickness / POSITION_TICK_METERS).round() as i32;
                    length -= ticks;
                    shift += ticks / 2 * face.sign() as i32;
                }
            }
        }
        Self {
            dimensions: CylinderDimensions {
                outer_diameter: outer,
                inner_diameter: if inner < 1.0e-5 { 0.0 } else { inner },
                axial_length_ticks: length.max(0) as u16,
                ..self.dimensions
            },
            pose: shifted_pose(self.pose, Vec3::Y * -(shift as f32)),
            layers: MaterialLayers::NONE,
            ..self
        }
    }

    /// Each layer's part-local region, oldest first.
    pub fn layer_regions(self) -> Vec<LayerRegion> {
        let half_length = self.dimensions.axial_length() * 0.5;
        let outer = self.dimensions.outer_diameter * 0.5;
        unwind_layer_regions(
            LayerEnvelope {
                minimum: Vec3::new(-outer, -half_length, -outer),
                maximum: Vec3::new(outer, half_length, outer),
                outer_radius: outer,
                inner_radius: self.dimensions.inner_diameter * 0.5,
            },
            self.layers,
        )
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
    outer_diameter: f32,
    inner_diameter: f32,
    span: GridDimension,
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
            Self::Gas => 6_000.0,
            Self::Electric => 500.0,
        }
    }

    /// No-load shaft speed supplied by one engine, in revolutions per minute.
    pub const fn no_load_rpm(self) -> f32 {
        match self {
            Self::Gas => 360.0,
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
    pub const STALL_TORQUE_NEWTON_METERS: f32 = 12_000.0;
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

/// Faces a pipe junction can open, in arm-bit order.
const PIPE_ARM_FACES: [FaceKind; 6] = [
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
pub struct PipeArms(u8);

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

const fn pipe_arm_bit(face: FaceKind) -> u8 {
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
    outer_diameter: f32,
    inner_diameter: f32,
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

/// A construction part with shape-specific dimensions and a shared build pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PartSpec {
    /// Rectangular cuboid.
    Cuboid(CuboidSpec),
    /// Solid or hollow cylinder whose axis is local Y.
    Cylinder(CylinderSpec),
    /// Cardinal 90-degree quarter-torus pipe bend.
    PipeBend(PipeBendSpec),
    /// Fitting joining pipe arms on any of its six faces.
    PipeJunction(PipeJunctionSpec),
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
            Self::PipeJunction(spec) => spec.pose,
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
            Self::PipeJunction(spec) => Some(spec.appearance),
            Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => None,
        }
    }

    /// Material layers over an ordinary part's core; none for other parts.
    pub const fn material_layers(self) -> MaterialLayers {
        match self {
            Self::Cuboid(spec) => spec.layers,
            Self::Cylinder(spec) => spec.layers,
            Self::PipeBend(_)
            | Self::PipeJunction(_)
            | Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => MaterialLayers::NONE,
        }
    }

    /// Whether the part carries any material layer.
    pub const fn is_layered(self) -> bool {
        !self.material_layers().is_empty()
    }

    /// Material and appearance of one band: zero is the core, `i + 1` layer `i`.
    pub fn band(self, band: u8) -> Option<(ConstructionMaterial, MaterialAppearance)> {
        let core = match self {
            Self::Cuboid(spec) => (spec.material, spec.appearance),
            Self::Cylinder(spec) => (spec.material, spec.appearance),
            Self::PipeBend(spec) => (spec.material, spec.appearance),
            Self::PipeJunction(spec) => (spec.material, spec.appearance),
            Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => return None,
        };
        match band.checked_sub(1) {
            None => Some(core),
            Some(index) => self
                .material_layers()
                .get(usize::from(index))
                .map(|layer| (layer.material, layer.appearance)),
        }
    }

    /// Returns the part with one band's appearance replaced, or `None` when the
    /// band does not exist.
    pub fn with_band_appearance(self, band: u8, appearance: MaterialAppearance) -> Option<Self> {
        let Some(index) = band.checked_sub(1) else {
            return self.with_appearance(appearance);
        };
        let index = usize::from(index);
        match self {
            Self::Cuboid(mut spec) => {
                spec.layers = spec.layers.with_appearance(index, appearance)?;
                Some(Self::Cuboid(spec))
            }
            Self::Cylinder(mut spec) => {
                spec.layers = spec.layers.with_appearance(index, appearance)?;
                Some(Self::Cylinder(spec))
            }
            Self::PipeBend(_)
            | Self::PipeJunction(_)
            | Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => None,
        }
    }

    /// Adds a material layer to a cuboid face or a cylinder wall, bore, or cap.
    ///
    /// # Errors
    ///
    /// Returns [`LayerError::UnsupportedFace`] for parts that take no layers, or
    /// the cuboid or cylinder layer error.
    pub fn with_layer(
        self,
        face: LayerFace,
        thickness: f32,
        material: ConstructionMaterial,
        appearance: MaterialAppearance,
    ) -> Result<Self, LayerError> {
        match self {
            Self::Cuboid(spec) => spec
                .with_layer(face, thickness, material, appearance)
                .map(Self::Cuboid),
            Self::Cylinder(spec) => spec
                .with_layer(face, thickness, material, appearance)
                .map(Self::Cylinder),
            Self::PipeBend(_)
            | Self::PipeJunction(_)
            | Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => Err(LayerError::UnsupportedFace),
        }
    }

    /// The core an ordinary part's layers were laid on; other parts unchanged.
    #[must_use]
    pub fn without_layers(self) -> Self {
        match self {
            Self::Cuboid(spec) => Self::Cuboid(spec.without_layers()),
            Self::Cylinder(spec) => Self::Cylinder(spec.without_layers()),
            other => other,
        }
    }

    /// Whether two parts are the same core, differing at most in their layers.
    pub fn shares_core_with(self, other: Self) -> bool {
        match (self.without_layers(), other.without_layers()) {
            (Self::Cylinder(first), Self::Cylinder(second)) => {
                let (a, b) = (first.dimensions, second.dimensions);
                first.pose == second.pose
                    && first.material == second.material
                    && a.axial_length_ticks == b.axial_length_ticks
                    && a.sweep_angle_degrees == b.sweep_angle_degrees
                    && (a.outer_diameter - b.outer_diameter).abs() < 1.0e-4
                    && (a.inner_diameter - b.inner_diameter).abs() < 1.0e-4
            }
            (Self::Cuboid(first), Self::Cuboid(second)) => {
                Self::Cuboid(first.with_appearance(second.appearance)) == Self::Cuboid(second)
            }
            _ => false,
        }
    }

    /// Each layer's part-local region, oldest first.
    pub fn layer_regions(self) -> Vec<LayerRegion> {
        match self {
            Self::Cuboid(spec) => spec.layer_regions(),
            Self::Cylinder(spec) => spec.layer_regions(),
            _ => Vec::new(),
        }
    }

    /// Band owning a part-local point: the newest layer whose region contains
    /// it, or the core.
    #[expect(clippy::cast_possible_truncation, reason = "at most MAX_PART_LAYERS")]
    pub fn band_at_local_point(self, local: Vec3) -> u8 {
        self.layer_regions()
            .iter()
            .rposition(|region| region.contains(local))
            .map_or(0, |index| index as u8 + 1)
    }

    /// Returns an ordinary construction part with a replacement appearance.
    pub(crate) const fn with_appearance(self, appearance: MaterialAppearance) -> Option<Self> {
        match self {
            Self::Cuboid(spec) => Some(Self::Cuboid(spec.with_appearance(appearance))),
            Self::Cylinder(spec) => Some(Self::Cylinder(spec.with_appearance(appearance))),
            Self::PipeBend(spec) => Some(Self::PipeBend(spec.with_appearance(appearance))),
            Self::PipeJunction(spec) => Some(Self::PipeJunction(spec.with_appearance(appearance))),
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
            Self::PipeJunction(mut spec) => {
                spec.pose = pose;
                Self::PipeJunction(spec)
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
            Self::Cylinder(_) | Self::PipeBend(_) | Self::PipeJunction(_) => None,
        }
    }

    /// Returns the control-block shape, when this part is a control block.
    pub const fn as_controller(self) -> Option<ControllerSpec> {
        match self {
            Self::Controller(spec) => Some(spec),
            Self::Cuboid(_)
            | Self::Cylinder(_)
            | Self::PipeBend(_)
            | Self::PipeJunction(_)
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
            | Self::PipeJunction(_)
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
            | Self::PipeJunction(_)
            | Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => None,
        }
    }

    /// Returns the pipe-junction shape, when this part is a junction.
    pub const fn as_pipe_junction(self) -> Option<PipeJunctionSpec> {
        match self {
            Self::PipeJunction(spec) => Some(spec),
            Self::Cuboid(_)
            | Self::Cylinder(_)
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
            Self::PipeJunction(spec) => Vec3::splat(spec.dimensions.half_side() * 2.0),
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

impl From<PipeJunctionSpec> for PartSpec {
    fn from(value: PipeJunctionSpec) -> Self {
        Self::PipeJunction(value)
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
            layers: MaterialLayers::NONE,
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

    /// Envelope side lengths in metres, including every layer.
    pub fn size_meters(self) -> Vec3 {
        let mut size = Vec3::new(
            self.dimensions[0].meters(),
            self.dimensions[1].meters(),
            self.dimensions[2].meters(),
        );
        for layer in self.layers.iter() {
            if let LayerFace::Face(face) = layer.face {
                size[face.axis().index()] += layer.thickness;
            }
        }
        size
    }

    /// Material layers over the core, oldest first.
    pub const fn layers(self) -> MaterialLayers {
        self.layers
    }

    /// Adds a flat material layer to one face, moving the centre outward by
    /// half the thickness so the opposite face stays put.
    ///
    /// # Errors
    ///
    /// Returns [`LayerError`] when the face is not flat, the thickness is not a
    /// whole 5 mm step of at least 1 cm, the part has the most layers, or the
    /// envelope would exceed 8 m.
    #[expect(clippy::cast_precision_loss, reason = "tick counts within 8 m")]
    pub fn with_layer(
        self,
        face: LayerFace,
        thickness: f32,
        material: ConstructionMaterial,
        appearance: MaterialAppearance,
    ) -> Result<Self, LayerError> {
        let LayerFace::Face(kind) = face else {
            return Err(LayerError::UnsupportedFace);
        };
        let ticks = layer_thickness_ticks(thickness, true)?;
        let thickness = ticks as f32 * POSITION_TICK_METERS;
        let layers = self.layers.pushed(MaterialLayer {
            face,
            thickness,
            material,
            appearance,
        })?;
        if self.size_meters()[kind.axis().index()] + thickness
            > f32::from(MAX_GRID_UNITS) * GRID_UNIT_METERS + 1.0e-4
        {
            return Err(LayerError::TooLarge);
        }
        Ok(Self {
            pose: shifted_pose(
                self.pose,
                kind.axis().unit() * kind.sign() * (ticks / 2) as f32,
            ),
            layers,
            ..self
        })
    }

    /// The grid-aligned core this cuboid's layers were laid on.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "whole ticks within 8 m"
    )]
    pub fn without_layers(self) -> Self {
        let shift = self
            .layers
            .iter()
            .filter_map(|layer| match layer.face {
                LayerFace::Face(face) => Some(
                    face.axis().unit()
                        * face.sign()
                        * ((layer.thickness / POSITION_TICK_METERS).round() as i32 / 2) as f32,
                ),
                LayerFace::OuterWall | LayerFace::Bore => None,
            })
            .sum::<Vec3>();
        Self {
            pose: shifted_pose(self.pose, -shift),
            layers: MaterialLayers::NONE,
            ..self
        }
    }

    /// Each layer's part-local region, oldest first.
    pub fn layer_regions(self) -> Vec<LayerRegion> {
        let half = self.size_meters() * 0.5;
        unwind_layer_regions(
            LayerEnvelope {
                minimum: -half,
                maximum: half,
                outer_radius: 0.0,
                inner_radius: 0.0,
            },
            self.layers,
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

/// Open arm end of a junction; closed faces are not connection faces.
pub(crate) fn pipe_junction_face(spec: PipeJunctionSpec, face: FaceKind) -> Option<FaceGeometry> {
    if !spec.arms.contains(face) {
        return None;
    }
    let rotation = spec.pose.rotation.quaternion();
    let (u_axis, v_axis) = face.tangent_axes();
    let normal = snap_cardinal(rotation * face.axis().unit()) * face.sign();
    Some(FaceGeometry {
        center: spec.pose.translation() + normal * spec.dimensions.half_side(),
        normal,
        tangent_u: snap_cardinal(rotation * u_axis.unit()),
        tangent_v: snap_cardinal(rotation * v_axis.unit()),
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
mod tests;
