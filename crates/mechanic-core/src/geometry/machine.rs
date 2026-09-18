//! Authored machine parts: controllers, engines, transmissions, servos, seats, inputs, and dimension links.

use super::CuboidSpec;
use super::grid::BuildPose;
use serde::{Deserialize, Serialize};

/// Stable identity of a Dimension Link within one saved world and its Garage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DimensionLinkId(pub u64);

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
