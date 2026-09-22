//! The construction grid: units, position ticks, dimensions, rotations, and poses.

use bevy_math::{EulerRot, IVec3, Quat, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Length of one exact authored position tick: 2.5 mm.
pub const POSITION_TICK_METERS: f32 = 0.0025;

/// Exact position ticks spanning one 25 cm construction cell.
pub const POSITION_TICKS_PER_GRID_UNIT: i32 = 100;

/// Exact position ticks spanning half a construction cell.
pub const POSITION_TICKS_PER_HALF_GRID_UNIT: i32 = POSITION_TICKS_PER_GRID_UNIT / 2;

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
pub struct GridDimension(pub(super) u8);

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Axis {
    /// Local x axis.
    X,
    /// Local y axis.
    Y,
    /// Local z axis.
    Z,
}

impl Axis {
    /// The axis's component index: 0, 1 or 2.
    pub const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }

    /// The axis's unit vector.
    pub const fn unit(self) -> Vec3 {
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
    pub(super) quarter_turns_xyz: [u8; 3],
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
    pub(super) position_tick_offset: [u8; 3],
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

/// Position ticks in one quarter-metre grid unit.
#[expect(clippy::cast_possible_truncation, reason = "100 ticks")]
pub(super) const GRID_UNIT_TICKS: u16 = POSITION_TICKS_PER_GRID_UNIT as u16;

pub(super) fn snap_cardinal(vector: Vec3) -> Vec3 {
    Vec3::new(vector.x.round(), vector.y.round(), vector.z.round())
}
