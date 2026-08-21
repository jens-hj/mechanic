use bevy_math::{EulerRot, IVec3, Quat, Vec3};
use thiserror::Error;

use crate::PartId;

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
}

/// Grid-aligned build pose.
///
/// The primary translation remains in quarter-metre units. An internal
/// half-grid offset allows odd-sized cuboids to sit flush against grid-aligned
/// faces without changing the positions produced by [`BuildPose::new`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BuildPose {
    /// Translation in integer construction-grid units.
    pub translation_units: IVec3,
    /// Discrete 90-degree orientation.
    pub rotation: GridRotation,
    half_grid_offset: [u8; 3],
}

impl BuildPose {
    /// Creates a build pose.
    pub const fn new(translation_units: IVec3, rotation: GridRotation) -> Self {
        Self {
            translation_units,
            rotation,
            half_grid_offset: [0; 3],
        }
    }

    /// Creates a build pose from integer eighth-metre centre coordinates.
    ///
    /// Half-grid coordinates are useful for odd-sized cuboids, whose centres
    /// lie halfway between construction-grid lines when resting on a face.
    pub fn from_half_grid(translation_half_units: IVec3, rotation: GridRotation) -> Self {
        let mut translation_units = IVec3::ZERO;
        let mut half_grid_offset = [0; 3];
        for axis in 0..3 {
            let remainder = translation_half_units[axis].rem_euclid(2);
            translation_units[axis] = (translation_half_units[axis] - remainder) / 2;
            half_grid_offset[axis] = u8::from(remainder != 0);
        }
        Self {
            translation_units,
            rotation,
            half_grid_offset,
        }
    }

    /// Translation in integer eighth-metre half-grid units.
    pub fn translation_half_units(self) -> IVec3 {
        self.translation_units * 2
            + IVec3::new(
                i32::from(self.half_grid_offset[0]),
                i32::from(self.half_grid_offset[1]),
                i32::from(self.half_grid_offset[2]),
            )
    }

    /// Translation in metres.
    pub fn translation(self) -> Vec3 {
        self.translation_half_units().as_vec3() * (GRID_UNIT_METERS * 0.5)
    }
}

/// Editable cuboid dimensions and build pose.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CuboidSpec {
    /// Validated x/y/z dimensions.
    pub dimensions: [GridDimension; 3],
    /// Cuboid centre and orientation.
    pub pose: BuildPose,
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
}

impl CylinderSpec {
    /// Creates a cylinder from validated dimensions and a build pose.
    pub const fn new(dimensions: CylinderDimensions, pose: BuildPose) -> Self {
        Self { dimensions, pose }
    }
}

/// Editable control block. Its shape is a fixed one-grid-unit cube; what it
/// does lives on the drive links wired from it, one program per bearing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControllerSpec {
    /// Control-block centre and cardinal orientation.
    pub pose: BuildPose,
}

impl ControllerSpec {
    /// Fixed control-block side length in grid units.
    pub const GRID_UNITS: u8 = 1;

    /// Creates a control block with the given pose.
    pub const fn new(pose: BuildPose) -> Self {
        Self { pose }
    }

    /// Fixed cube shape backing every control block.
    ///
    /// # Panics
    ///
    /// Never in practice: the fixed side length is a valid grid dimension.
    pub fn cuboid(self) -> CuboidSpec {
        CuboidSpec::new([Self::GRID_UNITS; 3], self.pose)
            .expect("the fixed control-block size is a valid grid dimension")
    }
}

/// A construction part with shape-specific dimensions and a shared build pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PartSpec {
    /// Rectangular cuboid.
    Cuboid(CuboidSpec),
    /// Solid or hollow cylinder whose axis is local Y.
    Cylinder(CylinderSpec),
    /// Fixed-size control block driving the bearings wired to it.
    Controller(ControllerSpec),
}

impl PartSpec {
    /// Part build pose.
    pub const fn pose(self) -> BuildPose {
        match self {
            Self::Cuboid(spec) => spec.pose,
            Self::Cylinder(spec) => spec.pose,
            Self::Controller(spec) => spec.pose,
        }
    }

    /// Returns the cuboid shape backing this part, when it has one. Control
    /// blocks report their fixed cube.
    pub fn as_cuboid(self) -> Option<CuboidSpec> {
        match self {
            Self::Cuboid(spec) => Some(spec),
            Self::Controller(spec) => Some(spec.cuboid()),
            Self::Cylinder(_) => None,
        }
    }

    /// Returns the control-block shape, when this part is a control block.
    pub const fn as_controller(self) -> Option<ControllerSpec> {
        match self {
            Self::Controller(spec) => Some(spec),
            Self::Cuboid(_) | Self::Cylinder(_) => None,
        }
    }

    /// Returns the cylinder shape, when this part is a cylinder.
    pub const fn as_cylinder(self) -> Option<CylinderSpec> {
        match self {
            Self::Cylinder(spec) => Some(spec),
            Self::Cuboid(_) | Self::Controller(_) => None,
        }
    }

    /// Axis-aligned local dimensions. Cylinders return diameter/length/diameter.
    pub fn size_meters(self) -> Vec3 {
        match self {
            Self::Cuboid(spec) => spec.size_meters(),
            Self::Controller(spec) => spec.cuboid().size_meters(),
            Self::Cylinder(spec) => Vec3::new(
                spec.dimensions.outer_diameter(),
                spec.dimensions.axial_length(),
                spec.dimensions.outer_diameter(),
            ),
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

impl From<ControllerSpec> for PartSpec {
    fn from(value: ControllerSpec) -> Self {
        Self::Controller(value)
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
        })
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
}

impl FaceRef {
    /// Creates a part-face reference.
    pub const fn part(part: PartId, face: FaceKind) -> Self {
        Self {
            owner: FaceOwner::Part(part),
            face,
        }
    }

    /// The ground's upward-facing plane.
    pub const fn ground() -> Self {
        Self {
            owner: FaceOwner::Ground,
            face: FaceKind::PositiveY,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FaceGeometry {
    pub(crate) center: Vec3,
    pub(crate) normal: Vec3,
    pub(crate) tangent_u: Vec3,
    pub(crate) tangent_v: Vec3,
    pub(crate) profile: FaceProfile,
}

#[derive(Clone, Copy, Debug)]
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
        BuildPose, CuboidSpec, CylinderDimensionError, CylinderDimensions, FaceKind, GridRotation,
        cuboid_face, snap_world_to_grid,
    };

    #[test]
    fn snaps_world_coordinates_to_quarter_metre_grid() {
        assert_eq!(
            snap_world_to_grid(Vec3::new(0.37, -0.13, 1.99)),
            IVec3::new(1, -1, 8)
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
    fn rejects_dimensions_outside_builder_range() {
        assert!(CuboidSpec::new([0, 4, 4], BuildPose::default()).is_err());
        assert!(CuboidSpec::new([33, 4, 4], BuildPose::default()).is_err());
        assert!(CuboidSpec::new([1, 32, 1], BuildPose::default()).is_ok());
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
}
