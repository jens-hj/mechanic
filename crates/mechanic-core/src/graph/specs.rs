//! Authoring inputs for bearings, welds, rigid links, drive wires, and control links.

use crate::{
    ActuatorAssignment, BearingId, DriveLimits, DriveName, DriveProgram, DriveTarget, EngineKind,
    FaceRef, PartId,
};
use bevy_math::Vec3;
use thiserror::Error;

/// Smallest supported bearing outer diameter, in metres.
pub const MIN_BEARING_OUTER_DIAMETER: f32 = 0.05;

/// Largest supported bearing outer diameter, in metres.
pub const MAX_BEARING_OUTER_DIAMETER: f32 = 8.0;

/// Minimum difference between a bearing's outer and inner diameters, in metres.
pub const MIN_BEARING_DIAMETER_GAP: f32 = 0.05;

/// Invalid visual bearing dimensions.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum BearingDimensionError {
    /// The outer diameter was not finite.
    #[error("bearing outer diameter must be finite")]
    NonFiniteOuterDiameter,
    /// The outer diameter was outside the supported range.
    #[error("bearing outer diameter must be between 0.05 m and 8.00 m")]
    OuterDiameterOutOfRange,
    /// The inner diameter was not finite.
    #[error("bearing inner diameter must be finite")]
    NonFiniteInnerDiameter,
    /// The inner diameter was negative or left less than the minimum ring thickness.
    #[error(
        "bearing inner diameter must be non-negative and at least 0.05 m smaller than the outer diameter"
    )]
    InnerDiameterOutOfRange,
}

/// Validated visual dimensions for a passive bearing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BearingDimensions {
    pub(super) outer_diameter: f32,
    pub(super) inner_diameter: f32,
}

impl BearingDimensions {
    /// Default bearing outer diameter, in metres.
    pub const DEFAULT_OUTER_DIAMETER: f32 = 0.25;

    /// Default bearing inner diameter, in metres.
    pub const DEFAULT_INNER_DIAMETER: f32 = 0.10;

    /// Creates validated visual bearing dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`BearingDimensionError`] when either diameter is non-finite,
    /// the outer diameter is outside `0.05..=8.00` metres, or the inner
    /// diameter is outside `0.00..=outer - 0.05` metres.
    pub fn new(outer_diameter: f32, inner_diameter: f32) -> Result<Self, BearingDimensionError> {
        if !outer_diameter.is_finite() {
            return Err(BearingDimensionError::NonFiniteOuterDiameter);
        }
        if !(MIN_BEARING_OUTER_DIAMETER..=MAX_BEARING_OUTER_DIAMETER).contains(&outer_diameter) {
            return Err(BearingDimensionError::OuterDiameterOutOfRange);
        }
        if !inner_diameter.is_finite() {
            return Err(BearingDimensionError::NonFiniteInnerDiameter);
        }
        if inner_diameter < 0.0 || inner_diameter > outer_diameter - MIN_BEARING_DIAMETER_GAP {
            return Err(BearingDimensionError::InnerDiameterOutOfRange);
        }
        Ok(Self {
            outer_diameter,
            inner_diameter,
        })
    }

    /// Outer diameter in metres.
    pub const fn outer_diameter(self) -> f32 {
        self.outer_diameter
    }

    /// Inner diameter in metres. Zero represents a solid disc.
    pub const fn inner_diameter(self) -> f32 {
        self.inner_diameter
    }
}

impl Default for BearingDimensions {
    fn default() -> Self {
        Self {
            outer_diameter: Self::DEFAULT_OUTER_DIAMETER,
            inner_diameter: Self::DEFAULT_INNER_DIAMETER,
        }
    }
}

/// Explicit weld between two touching faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeldSpec {
    /// First selected face.
    pub first: FaceRef,
    /// Second selected face.
    pub second: FaceRef,
}

/// Non-geometric rigid membership between two parts.
///
/// Unlike a weld, a rigid link does not require touching faces and creates no
/// visible geometry. It lets one connector own separated parts as one body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RigidLinkSpec {
    /// First part in the shared rigid body.
    pub first: PartId,
    /// Second part in the shared rigid body.
    pub second: PartId,
}

/// A mesh between two toothed parts, or a nut on a thread.
///
/// The parts never touch: the solver couples them magnetically at their pitch
/// point, so the link carries no geometry of its own. Everything about the
/// mesh comes from the two parts' teeth and rest poses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GearLinkSpec {
    /// First meshing part.
    pub first: PartId,
    /// Second meshing part.
    pub second: PartId,
}

impl GearLinkSpec {
    /// Whether this link joins `part` to anything.
    pub fn references(self, part: PartId) -> bool {
        self.first == part || self.second == part
    }

    /// The part on the other side of `part`, if `part` is one of the two.
    pub fn other(self, part: PartId) -> Option<PartId> {
        if self.first == part {
            Some(self.second)
        } else if self.second == part {
            Some(self.first)
        } else {
            None
        }
    }
}

/// Wire from a control block to one bearing it drives.
///
/// The wire carries everything about how that one bearing behaves: its speed
/// and torque envelope and the ordered states it moves through. A control block
/// is only the identity that owns a set of wires, so two bearings on the same
/// block can run entirely different programs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriveLinkSpec {
    /// Typed linear envelope; required for a linear joint.
    pub linear_limits: Option<crate::LinearDriveLimits>,
    /// Control-block part this wire belongs to.
    pub controller: PartId,
    /// Bearing driven through this wire.
    pub bearing: BearingId,
    /// Whether this bearing runs opposite the programmed direction.
    pub reversed: bool,
    /// Physical actuator family assigned to this joint.
    pub actuator: ActuatorAssignment,
    /// Speed, torque, and travel envelope of this bearing.
    pub limits: DriveLimits,
    /// Ordered states this bearing moves through.
    pub program: DriveProgram,
    /// What the panel calls the joint this wire drives. Empty means the panel
    /// falls back to the joint's number.
    pub name: DriveName,
}

impl DriveLinkSpec {
    /// Wires a bearing to a control block with default limits and a single
    /// state that holds the bearing still.
    pub fn new(controller: PartId, bearing: BearingId) -> Self {
        Self {
            controller,
            bearing,
            linear_limits: None,
            reversed: false,
            actuator: ActuatorAssignment::Unpowered,
            limits: DriveLimits::default(),
            program: DriveProgram::default(),
            name: DriveName::EMPTY,
        }
    }

    /// Creates a stationary linear program spanning a translational joint's
    /// full physical travel, as [`crate::JointKind::bounds`] reports it.
    ///
    /// # Panics
    /// Panics if the bounds violate the drive envelope invariants, which
    /// validated rail and piston dimensions never do.
    pub fn new_linear(controller: PartId, bearing: BearingId, bounds: [f32; 2]) -> Self {
        let [minimum, maximum] = bounds;
        let mut link = Self::new(controller, bearing);
        link.linear_limits = Some(
            crate::LinearDriveLimits::new(1.0, f32::MAX, minimum, maximum)
                .expect("validated linear travel"),
        );
        link.program = DriveProgram::new(
            &[crate::DriveState::new(DriveTarget::LinearSpeed(0.0)).expect("zero speed")],
            false,
        )
        .expect("one state");
        link
    }

    /// What this wire asks of its bearing in the given state, with reversal
    /// applied.
    pub fn resolved_target(&self, state: u8) -> Option<DriveTarget> {
        let target = self.program.state(state)?.target();
        Some(if self.reversed {
            target.reversed()
        } else {
            target
        })
    }
}

/// Logical keyboard route from one Input block to one Seat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputSeatLinkSpec {
    /// Input block producing keyboard events.
    pub input: PartId,
    /// Seat whose occupant owns those events.
    pub seat: PartId,
}

/// Logical keyboard route from one Seat to one Controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeatControllerLinkSpec {
    /// Seat whose occupant supplies keyboard events.
    pub seat: PartId,
    /// Controller receiving those events.
    pub controller: PartId,
}

/// Actuator hardware and current assignment demand in one Controller module.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ActuatorInventory {
    /// Electric engines in the module.
    pub electric_engines: u32,
    /// Gas engines in the module.
    pub gas_engines: u32,
    /// Servos in the module.
    pub servos: u32,
    /// Physical joints using electric power.
    pub electric_joints: u32,
    /// Physical joints using gas power.
    pub gas_joints: u32,
    /// Physical joints using servo power.
    pub servo_joints: u32,
    /// Common electric transmission depth, or `None` when absent or mismatched.
    pub electric_transmission_depth: Option<u8>,
    /// Common gas transmission depth, or `None` when absent or mismatched.
    pub gas_transmission_depth: Option<u8>,
    /// Whether electric engines in this module have different chain depths.
    pub electric_transmission_mismatch: bool,
    /// Whether gas engines in this module have different chain depths.
    pub gas_transmission_mismatch: bool,
}

impl ActuatorInventory {
    /// Available electric bearing ports.
    pub const fn electric_capacity(self) -> u32 {
        self.electric_engines * EngineKind::Electric.bearing_capacity()
    }

    /// Available gas bearing ports.
    pub const fn gas_capacity(self) -> u32 {
        self.gas_engines * EngineKind::Gas.bearing_capacity()
    }

    /// Available dedicated servo ports.
    pub const fn servo_capacity(self) -> u32 {
        self.servos
    }

    /// Whether an editable electric gearbox is present and unambiguous.
    pub const fn electric_gearbox_available(self) -> bool {
        matches!(self.electric_transmission_depth, Some(1..))
            && !self.electric_transmission_mismatch
    }

    /// Whether an editable gas gearbox is present and unambiguous.
    pub const fn gas_gearbox_available(self) -> bool {
        matches!(self.gas_transmission_depth, Some(1..)) && !self.gas_transmission_mismatch
    }
}

/// One-degree-of-freedom bearing between two faces. It is passive unless a
/// control block is wired to it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BearingSpec {
    /// Permitted motion and physical rail geometry.
    pub kind: crate::JointKind,
    /// Face whose outward normal establishes the bearing axis.
    pub source: FaceRef,
    /// Compatible face on the attached side. `None` is a joint whose moving
    /// side is the hardware's own head with nothing built on it yet.
    pub target: Option<FaceRef>,
    /// Shared world-space anchor selected on both faces.
    pub shared_anchor: Vec3,
    /// Unit world-space axis, equal to the source-face normal.
    pub axis: Vec3,
    /// Visual-only outer and inner diameters.
    pub dimensions: BearingDimensions,
}

impl BearingSpec {
    /// Creates a bearing specification. Geometry is validated on insertion.
    pub const fn new(source: FaceRef, target: FaceRef, shared_anchor: Vec3, axis: Vec3) -> Self {
        Self {
            kind: crate::JointKind::Rotational,
            source,
            target: Some(target),
            shared_anchor,
            axis,
            dimensions: BearingDimensions {
                outer_diameter: BearingDimensions::DEFAULT_OUTER_DIAMETER,
                inner_diameter: BearingDimensions::DEFAULT_INNER_DIAMETER,
            },
        }
    }

    /// Creates the joint of hardware that carries its own moving head, before
    /// anything is attached to that head.
    #[must_use]
    pub const fn bare(
        source: FaceRef,
        shared_anchor: Vec3,
        axis: Vec3,
        kind: crate::JointKind,
    ) -> Self {
        let mut spec = Self::new(source, source, shared_anchor, axis);
        spec.target = None;
        spec.kind = kind;
        spec
    }

    /// Selects the physical bearing variant. Geometry is validated on insertion.
    #[must_use]
    pub const fn with_kind(mut self, kind: crate::JointKind) -> Self {
        self.kind = kind;
        self
    }

    /// Applies custom validated visual dimensions.
    #[must_use]
    pub const fn with_dimensions(mut self, dimensions: BearingDimensions) -> Self {
        self.dimensions = dimensions;
        self
    }
}
