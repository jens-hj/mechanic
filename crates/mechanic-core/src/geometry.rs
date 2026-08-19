use bevy_math::{EulerRot, IVec3, Quat, Vec3};
use thiserror::Error;

use crate::PartId;

/// Construction-grid spacing in metres.
pub const GRID_UNIT_METERS: f32 = 0.25;

/// Largest cuboid dimension in grid units (8 m).
pub const MAX_GRID_UNITS: u8 = 32;

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

/// One of the six oriented cuboid faces.
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
    /// A user-created cuboid.
    Part(PartId),
    /// The central static ground plane. Only its positive-y face is valid.
    Ground,
}

/// Stable reference to a cuboid or ground face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaceRef {
    /// Face owner.
    pub owner: FaceOwner,
    /// Oriented face on that owner.
    pub face: FaceKind,
}

impl FaceRef {
    /// Creates a cuboid-face reference.
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
    pub(crate) half_u: f32,
    pub(crate) half_v: f32,
    pub(crate) infinite: bool,
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
        half_u: size[u_axis.index()] * 0.5,
        half_v: size[v_axis.index()] * 0.5,
        infinite: false,
    }
}

pub(crate) const fn ground_face() -> FaceGeometry {
    FaceGeometry {
        center: Vec3::ZERO,
        normal: Vec3::Y,
        tangent_u: Vec3::X,
        tangent_v: Vec3::Z,
        half_u: f32::INFINITY,
        half_v: f32::INFINITY,
        infinite: true,
    }
}

fn snap_cardinal(vector: Vec3) -> Vec3 {
    Vec3::new(vector.x.round(), vector.y.round(), vector.z.round())
}

#[cfg(test)]
mod tests {
    use bevy_math::{IVec3, Vec3};

    use super::{BuildPose, CuboidSpec, FaceKind, GridRotation, cuboid_face, snap_world_to_grid};

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
