//! Authored physical input dimensions. Local −Y is the mounting face.

use bevy_math::Vec3;
use serde::{Deserialize, Serialize};

use crate::BuildPose;

/// Mounting footprint of an independently authored input model.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum InputSize {
    /// Five centimetre panel control.
    #[default]
    Panel,
    /// Ten centimetre utility control.
    Utility,
    /// Twenty-five centimetre industrial control.
    Industrial,
}

impl InputSize {
    /// Picker order, smallest first.
    pub const ALL: [Self; 3] = [Self::Panel, Self::Utility, Self::Industrial];

    /// Square mounting footprint in metres.
    pub const fn meters(self) -> f32 {
        match self {
            Self::Panel => 0.05,
            Self::Utility => 0.10,
            Self::Industrial => 0.25,
        }
    }

    /// Visible size label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Panel => "5 cm",
            Self::Utility => "10 cm",
            Self::Industrial => "25 cm",
        }
    }

    /// Button cap travel in metres.
    pub const fn button_travel(self) -> f32 {
        match self {
            Self::Panel => 0.003,
            Self::Utility => 0.0075,
            Self::Industrial => 0.02125,
        }
    }
}

/// Rotary dial geometry and placement. Values belong to its controller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DialSpec {
    /// Independently authored model size.
    pub size: InputSize,
    /// Envelope centre and cardinal orientation.
    pub pose: BuildPose,
}

impl DialSpec {
    /// Creates an input without controller values or transient pointer state.
    pub const fn new(size: InputSize, pose: BuildPose) -> Self {
        Self { size, pose }
    }

    /// Actual placement and collision envelope in local metres.
    pub const fn size_meters(self) -> Vec3 {
        let height = match self.size {
            InputSize::Panel => 0.035_217,
            InputSize::Utility => 0.050_629,
            InputSize::Industrial => 0.149_293,
        };
        Vec3::new(self.size.meters(), height, self.size.meters())
    }
}

/// Pushbutton geometry and placement. Configuration belongs to the graph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ButtonSpec {
    /// Independently authored model size.
    pub size: InputSize,
    /// Envelope centre and cardinal orientation.
    pub pose: BuildPose,
}

impl ButtonSpec {
    /// Creates a button without saved activation state.
    pub const fn new(size: InputSize, pose: BuildPose) -> Self {
        Self { size, pose }
    }

    /// Actual released placement and collision envelope in local metres.
    pub const fn size_meters(self) -> Vec3 {
        let height = match self.size {
            InputSize::Panel => 0.027_150,
            InputSize::Utility => 0.041_500,
            InputSize::Industrial => 0.133_750,
        };
        Vec3::new(self.size.meters(), height, self.size.meters())
    }
}

/// How an operated button produces command edges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ButtonMode {
    /// Activate while held and request release when interaction ends.
    #[default]
    Momentary,
    /// Each press toggles this button's own held-key source.
    Toggle,
}

/// Button geometry pose; runtime appearance follows the combined controller key.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ButtonFeedback {
    /// The configured key is released or unassigned.
    #[default]
    Off,
    /// Intermediate baked spring pose, reserved for geometry previews.
    Mixed,
    /// The configured key is held by at least one controller input source.
    On,
}

impl ButtonFeedback {
    /// Cap displacement as a fraction of full travel.
    pub const fn depression(self) -> f32 {
        match self {
            Self::Off => 0.0,
            Self::Mixed => 0.5,
            Self::On => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelopes_keep_actual_footprints_and_independent_heights() {
        for size in InputSize::ALL {
            let dial = DialSpec::new(size, BuildPose::default()).size_meters();
            let button = ButtonSpec::new(size, BuildPose::default()).size_meters();
            assert!((dial.x - size.meters()).abs() < f32::EPSILON);
            assert!((button.z - size.meters()).abs() < f32::EPSILON);
            assert!(dial.y < size.meters());
            assert!(button.y < size.meters());
            assert!(button.y > size.button_travel());
        }
    }
}
