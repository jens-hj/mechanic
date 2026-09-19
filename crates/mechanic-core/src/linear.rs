//! Geometry and physical dimensions shared by linear-bearing construction and simulation.

use bevy_math::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Distance travelled per output revolution, in metres.
pub const LINEAR_METERS_PER_REVOLUTION: f32 = 0.25;
/// Distance travelled per output radian, in metres.
pub const LINEAR_METERS_PER_RADIAN: f32 = LINEAR_METERS_PER_REVOLUTION / core::f32::consts::TAU;

/// Invalid rail dimensions or attachment frame.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum LinearBearingError {
    /// Length must be finite and within the supported interval.
    #[error("linear bearing length must be between 0.25 and 8 m")]
    Length,
    /// Width must be finite and within the supported interval.
    #[error("linear bearing width must be between 0.05 and 0.40 m")]
    Width,
    /// Dimensions must lie on the construction position grid.
    #[error("linear bearing dimensions must be multiples of 2.5 mm")]
    Quantization,
    /// The mounting frame must be orthonormal.
    #[error("linear bearing travel and mounting axes must be perpendicular unit vectors")]
    Frame,
}

/// Validated, exactly quantized linear-bearing dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "[f32; 2]", into = "[f32; 2]")]
pub struct LinearBearingDimensions {
    length_ticks: u16,
    width_ticks: u16,
}

impl LinearBearingDimensions {
    /// Carriage top above the mounting plane, in metres.
    pub const HEIGHT: f32 = 0.100;
    /// Length of the moving carriage, in metres.
    pub const CARRIAGE_LENGTH: f32 = 0.120;
    /// Thickness of each physical end stop, in metres.
    pub const END_STOP: f32 = 0.015;
    /// Attachment lattice pitch, in metres.
    pub const ATTACHMENT_PITCH: f32 = 0.025;

    /// Creates dimensions on the 2.5 mm grid.
    ///
    /// # Errors
    /// Returns an error for non-finite, out-of-range, or off-grid dimensions.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "validated positive, integral ticks fit u16"
    )]
    pub fn new(length: f32, width: f32) -> Result<Self, LinearBearingError> {
        if !(0.25..=8.0).contains(&length) {
            return Err(LinearBearingError::Length);
        }
        if !(0.05..=0.40).contains(&width) {
            return Err(LinearBearingError::Width);
        }
        let ticks = [length, width].map(|value| value * 400.0);
        if ticks
            .iter()
            .any(|value| (value - value.round()).abs() > 0.00025)
        {
            return Err(LinearBearingError::Quantization);
        }
        Ok(Self {
            length_ticks: ticks[0].round() as u16,
            width_ticks: ticks[1].round() as u16,
        })
    }

    /// Rail length, including both stops, in metres.
    pub fn length(self) -> f32 {
        f32::from(self.length_ticks) / 400.0
    }

    /// Rail width in metres.
    pub fn width(self) -> f32 {
        f32::from(self.width_ticks) / 400.0
    }

    /// Quantized outer carriage half-width; the inner clearance stays unchanged.
    pub fn carriage_half_width(self) -> f32 {
        ((self.width() / 2.0 + 0.016) / crate::POSITION_TICK_METERS).round()
            * crate::POSITION_TICK_METERS
    }

    /// Total physical travel, in metres.
    pub fn travel(self) -> f32 {
        self.length() - 0.150
    }

    /// Physical displacement limits relative to the centred build pose.
    pub fn bounds(self) -> [f32; 2] {
        [-self.travel() / 2.0, self.travel() / 2.0]
    }
}

impl Default for LinearBearingDimensions {
    fn default() -> Self {
        Self {
            length_ticks: 400,
            width_ticks: 40,
        }
    }
}

impl TryFrom<[f32; 2]> for LinearBearingDimensions {
    type Error = LinearBearingError;
    fn try_from(value: [f32; 2]) -> Result<Self, Self::Error> {
        Self::new(value[0], value[1])
    }
}

impl From<LinearBearingDimensions> for [f32; 2] {
    fn from(value: LinearBearingDimensions) -> Self {
        [value.length(), value.width()]
    }
}

/// The single carriage face occupied by direct attachments.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CarriageFace {
    /// Broad upper face.
    #[default]
    Top,
    /// Side in the positive rail-local Z direction.
    PositiveSide,
    /// Side in the negative rail-local Z direction.
    NegativeSide,
}

impl CarriageFace {
    /// Outward local normal; rail travel is local X and mounting normal is Y.
    pub const fn normal(self) -> Vec3 {
        match self {
            Self::Top => Vec3::Y,
            Self::PositiveSide => Vec3::Z,
            Self::NegativeSide => Vec3::NEG_Z,
        }
    }

    /// Centred local attachment-plane origin, in metres.
    pub fn origin(self, dimensions: LinearBearingDimensions) -> Vec3 {
        match self {
            Self::Top => Vec3::new(0.0, LinearBearingDimensions::HEIGHT, 0.0),
            Self::PositiveSide => Vec3::new(0.0, 0.055, dimensions.carriage_half_width()),
            Self::NegativeSide => Vec3::new(0.0, 0.055, -dimensions.carriage_half_width()),
        }
    }

    /// Rectangular attachment extent along travel and across the selected face.
    pub fn size(self, dimensions: LinearBearingDimensions) -> Vec2 {
        Vec2::new(
            LinearBearingDimensions::CARRIAGE_LENGTH,
            match self {
                Self::Top => dimensions.carriage_half_width() * 2.0,
                Self::PositiveSide | Self::NegativeSide => 0.09,
            },
        )
    }
}

/// A rail frame and the occupied carriage surface. The anchor is the rail's underside centre.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearBearing {
    /// Rail dimensions.
    pub dimensions: LinearBearingDimensions,
    /// World-space mounting normal, perpendicular to the bearing travel axis.
    pub mount_normal: Vec3,
    /// Selected carriage attachment surface.
    pub face: CarriageFace,
}

impl LinearBearing {
    /// Validates the rail frame and returns its local-to-world rotation.
    ///
    /// # Errors
    /// Returns an error unless both axes form an orthonormal frame.
    pub fn rotation(self, travel_axis: Vec3) -> Result<Quat, LinearBearingError> {
        if !travel_axis.is_finite()
            || !self.mount_normal.is_finite()
            || (travel_axis.length_squared() - 1.0).abs() > 1.0e-5
            || (self.mount_normal.length_squared() - 1.0).abs() > 1.0e-5
            || travel_axis.dot(self.mount_normal).abs() > 1.0e-5
        {
            return Err(LinearBearingError::Frame);
        }
        Ok(Quat::from_mat3(&bevy_math::Mat3::from_cols(
            travel_axis,
            self.mount_normal,
            travel_axis.cross(self.mount_normal),
        )))
    }
}

#[cfg(test)]
mod tests;
