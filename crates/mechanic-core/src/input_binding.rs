//! Controller references and value mappings for physical inputs.

use crate::{ButtonMode, DriveKey, DriveLinkId, EngineKind, PartId};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Numeric controller field in its native units.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DriveParameter {
    /// Angular state position in radians.
    AngularPosition(u8),
    /// Angular state speed in radians per second.
    AngularSpeed(u8),
    /// Linear state position in metres.
    LinearPosition(u8),
    /// Linear state speed in metres per second.
    LinearSpeed(u8),
    /// Minimum programmed travel in the joint's native units.
    TravelMinimum,
    /// Maximum programmed travel in the joint's native units.
    TravelMaximum,
    /// State duration in seconds.
    Dwell(u8),
    /// Electric motor contribution, in integer percent.
    ElectricContribution,
    /// Gas motor contribution, in integer percent.
    GasContribution,
}

/// Editable gearbox number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum GearParameter {
    /// Input-to-output ratio, indexed in gear order.
    Ratio(u8),
    /// Integer number of reverse gears.
    ReverseCount,
}

/// Typed reference to one editable numeric controller setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NumericParameter {
    /// A driven joint's setting.
    Drive {
        /// Stable drive link.
        link: DriveLinkId,
        /// Numeric field.
        parameter: DriveParameter,
    },
    /// A controller's engine-family gearbox setting.
    Gear {
        /// Controller owning the gearbox.
        controller: PartId,
        /// Engine family.
        kind: EngineKind,
        /// Numeric field.
        parameter: GearParameter,
    },
}

/// Shared native-unit bounds for one enabled numeric editor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NumericMetadata {
    /// Smallest currently legal endpoint.
    pub minimum: f32,
    /// Largest currently legal endpoint.
    pub maximum: f32,
    /// Whole-number editing step; absent for continuous parameters.
    pub integer_step: Option<f32>,
}

/// A finite, nonconstant linear mapping stored in the target's native units.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "AnalogRangeDoc")]
pub struct AnalogRange {
    endpoints: [f32; 2],
    inverted: bool,
}

#[derive(Deserialize)]
struct AnalogRangeDoc {
    endpoints: [f32; 2],
    inverted: bool,
}

impl TryFrom<AnalogRangeDoc> for AnalogRange {
    type Error = InputBindingError;

    fn try_from(doc: AnalogRangeDoc) -> Result<Self, Self::Error> {
        Self::new(doc.endpoints, doc.inverted)
    }
}

/// Why a physical input configuration or mapping was rejected.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum InputBindingError {
    /// Equal or nonfinite endpoints cannot establish a dial position.
    #[error("dial endpoints must be finite and minimum must be below maximum")]
    InvalidRange,
    /// A referenced input, controller, joint, or state is missing or incompatible.
    #[error("input binding target is missing or incompatible")]
    InvalidTarget,
    /// The input is not connected to the target's controller.
    #[error("input must be connected to the target controller")]
    WrongController,
    /// One numeric setting has at most one dial source.
    #[error("parameter already has a dial binding")]
    DuplicateParameter,
    /// Possible contribution requires hardware that is not installed.
    #[error("dial range exceeds available actuator capacity")]
    InsufficientCapacity,
}

impl AnalogRange {
    /// Creates a mapping, with increasing endpoints and a separate direction.
    ///
    /// # Errors
    /// Returns [`InputBindingError::InvalidRange`] for nonfinite or unordered endpoints.
    pub fn new(endpoints: [f32; 2], inverted: bool) -> Result<Self, InputBindingError> {
        if endpoints.iter().any(|value| !value.is_finite()) || endpoints[0] >= endpoints[1] {
            return Err(InputBindingError::InvalidRange);
        }
        Ok(Self {
            endpoints,
            inverted,
        })
    }

    /// Native values at the authored endpoints, before inversion.
    pub const fn endpoints(self) -> [f32; 2] {
        self.endpoints
    }

    /// Whether direction is inverted.
    pub const fn inverted(self) -> bool {
        self.inverted
    }

    /// Maps a finite normalized position to the target's native units.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "convex combination stays within finite f32 endpoints"
    )]
    pub fn value(self, position: f32) -> Option<f32> {
        if !position.is_finite() {
            return None;
        }
        let position = position.clamp(0.0, 1.0);
        let position = if self.inverted {
            1.0 - position
        } else {
            position
        };
        // f64 prevents finite f32 endpoints overflowing during subtraction.
        Some(
            (f64::from(self.endpoints[0]) * f64::from(1.0 - position)
                + f64::from(self.endpoints[1]) * f64::from(position)) as f32,
        )
    }

    /// Inverse mapping, preserving out-of-range values so they produce Mixed.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "inverse result uses the public f32 controller precision"
    )]
    pub fn position(self, value: f32) -> Option<f32> {
        if !value.is_finite() {
            return None;
        }
        let position = ((f64::from(value) - f64::from(self.endpoints[0]))
            / (f64::from(self.endpoints[1]) - f64::from(self.endpoints[0])))
            as f32;
        Some(if self.inverted {
            1.0 - position
        } else {
            position
        })
    }
}

/// One dial-to-parameter mapping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalogMapping {
    /// Parameter controlled by this mapping.
    pub target: NumericParameter,
    /// Native-unit range and direction.
    pub range: AnalogRange,
}

/// Saved physical input configuration. Controller values supply all feedback.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InputConfiguration {
    /// Player-facing name.
    pub name: String,
    /// Optional controller link, independent of weld and seat connectivity.
    pub controller: Option<PartId>,
    /// Momentary by default; used only by buttons.
    pub button_mode: ButtonMode,
    /// Controller key, blank by default; used only by buttons.
    pub key: Option<DriveKey>,
    /// Mappings from this dial to numeric controller settings.
    pub analog: Vec<AnalogMapping>,
}

/// Requested controller values reflected by a rotary dial.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DialFeedback {
    /// Every inverse mapping agrees on this normalized position.
    Uniform(f32),
    /// Targets disagree or fall outside their configured ranges.
    Mixed,
}

impl DialFeedback {
    /// Infers feedback without mutating controller values or producing commands.
    pub fn from_values(values: impl IntoIterator<Item = (AnalogRange, f32)>) -> Self {
        let mut previous: Option<f32> = None;
        for (range, value) in values {
            let Some(position) = range.position(value) else {
                return Self::Mixed;
            };
            if !(0.0..=1.0).contains(&position)
                || previous.is_some_and(|old| (old - position).abs() > 1e-4)
            {
                return Self::Mixed;
            }
            previous = Some(position);
        }
        Self::Uniform(previous.unwrap_or(0.5))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_feedback_compares_positions_across_units_and_inversion() {
        let speed = AnalogRange::new([-10.0, 20.0], false).unwrap();
        let travel = AnalogRange::new([0.0, 2.0], true).unwrap();
        assert_eq!(
            DialFeedback::from_values([(speed, -2.5), (travel, 1.5)]),
            DialFeedback::Uniform(0.25)
        );
        assert_eq!(
            DialFeedback::from_values([(speed, -2.5), (travel, 1.0)]),
            DialFeedback::Mixed
        );
        assert_eq!(travel.value(0.25), Some(1.5));
    }

    #[test]
    fn out_of_range_controller_edits_are_mixed_instead_of_clipped_feedback() {
        let range = AnalogRange::new([0.0, 1.0], false).unwrap();
        assert_eq!(
            DialFeedback::from_values([(range, 2.0)]),
            DialFeedback::Mixed
        );
        assert_eq!(range.value(-1.0), Some(0.0));
        assert_eq!(range.value(f32::NAN), None);
    }

    #[test]
    fn finite_extreme_ranges_do_not_overflow() {
        assert!(ron::from_str::<AnalogRange>("(endpoints:(1.0,1.0),inverted:false)").is_err());
        let range = AnalogRange::new([-f32::MAX, f32::MAX], false).unwrap();
        assert_eq!(range.value(0.5), Some(0.0));
        assert_eq!(range.position(0.0), Some(0.5));
        assert!(AnalogRange::new([f32::INFINITY, 1.0], false).is_err());
        assert!(AnalogRange::new([1.0, 1.0], false).is_err());
    }
}
