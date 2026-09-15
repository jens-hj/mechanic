use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use bevy_math::{IVec3, Vec2, Vec3};
use thiserror::Error;

mod solid_cache;

use crate::{
    ANCHOR_TOLERANCE_METERS, AXIS_TOLERANCE_DEGREES, ActuatorAssignment, BearingId, BuildPose,
    CageIndex, ConstructionMaterial, ControllerSpec, CuboidSpec, CylinderSpec, DimensionLinkId,
    DimensionLinkSpec, DriveLimits, DriveLinkId, DriveName, DriveProgram, DriveTarget, EngineKind,
    EngineSpec, FaceKind, FaceOwner, FaceRef, GearKeyChord, GearboxConfig, GearboxError,
    InputSeatLinkId, InputSpec, MaterialAppearance, PartId, PartSpec, PipeBendSpec,
    PipeJunctionSpec, RegionError, RegionId, RigidLinkId, SeatControllerLinkId, SeatSpec,
    ServoSpec, ShapeFeature, ShapeFeatureId, ShapeRegion, ShiftMode, SolidError, SolidOwner,
    TransmissionSpec, WeldId,
    geometry::{
        FaceGeometry, FaceProfile, cuboid_face, cylinder_face, ground_face, pipe_bend_face,
        pipe_junction_face,
    },
    id::Arena,
};

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
    outer_diameter: f32,
    inner_diameter: f32,
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

    /// Creates a centred linear program with the rail's full physical travel.
    ///
    /// # Panics
    /// Panics if validated rail dimensions violate the drive envelope invariants.
    pub fn new_linear(
        controller: PartId,
        bearing: BearingId,
        dimensions: crate::LinearBearingDimensions,
    ) -> Self {
        let [minimum, maximum] = dimensions.bounds();
        let mut link = Self::new(controller, bearing);
        link.linear_limits = Some(
            crate::LinearDriveLimits::new(1.0, f32::MAX, minimum, maximum)
                .expect("validated rail travel"),
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
    pub kind: crate::BearingKind,
    /// Face whose outward normal establishes the bearing axis.
    pub source: FaceRef,
    /// Compatible face on the attached side.
    pub target: FaceRef,
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
            kind: crate::BearingKind::Rotational,
            source,
            target,
            shared_anchor,
            axis,
            dimensions: BearingDimensions {
                outer_diameter: BearingDimensions::DEFAULT_OUTER_DIAMETER,
                inner_diameter: BearingDimensions::DEFAULT_INNER_DIAMETER,
            },
        }
    }

    /// Selects the physical bearing variant. Geometry is validated on insertion.
    #[must_use]
    pub const fn with_kind(mut self, kind: crate::BearingKind) -> Self {
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

/// UI operation that has a selected first endpoint but has not mutated topology.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PendingOperation {
    /// First face of a weld.
    Weld(FaceRef),
    /// Source face and anchor of a bearing.
    Bearing {
        /// Selected source face.
        source: FaceRef,
        /// Selected point on that face.
        anchor: Vec3,
    },
    /// Control block selected as the first endpoint of a drive wire.
    DriveLink(PartId),
}

/// Scope of one construction appearance edit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AppearanceTarget {
    /// One ordinary construction part.
    Part(PartId),
    /// One radial material band of a layered cylinder, counted from the innermost.
    CylinderBand {
        /// Layered cylinder.
        part: PartId,
        /// Band index.
        band: u8,
    },
    /// A whole shaped region and each of its member parts.
    Region(RegionId),
}

/// Atomic edit request for a construction graph.
#[derive(Clone, Debug, PartialEq)]
pub enum BuildCommand {
    /// Spawn a standalone cuboid.
    Spawn(CuboidSpec),
    /// Spawn a standalone solid or hollow cylinder.
    SpawnCylinder(CylinderSpec),
    /// Spawn a standalone cardinal 90-degree pipe bend.
    SpawnPipeBend(PipeBendSpec),
    /// Spawns a cube pipe junction.
    SpawnPipeJunction(PipeJunctionSpec),
    /// Remove a part and every connection referencing it.
    Remove(PartId),
    /// Remove one weld while leaving its endpoint parts intact.
    RemoveWeld(WeldId),
    /// Remove one non-geometric rigid link.
    RemoveRigidLink(RigidLinkId),
    /// Replace only a construction target's color and finish treatment.
    SetAppearance {
        /// Part or shaped region to update.
        target: AppearanceTarget,
        /// Replacement appearance.
        appearance: MaterialAppearance,
    },
    /// Replace a cylinder's envelope and material bands in place, keeping its
    /// part identity, pose, features, and connections.
    SetCylinder {
        /// Cylinder being reshaped.
        part: PartId,
        /// Replacement with the same pose.
        spec: CylinderSpec,
    },
    /// Remove one bearing while leaving its endpoint parts intact.
    RemoveBearing(BearingId),
    /// Merge the groups containing two touching faces.
    Weld(WeldSpec),
    /// Merge two parts rigidly without requiring face contact.
    RigidLink(RigidLinkSpec),
    /// Spawn a control block.
    SpawnController(ControllerSpec),
    /// Spawn an inert engine.
    SpawnEngine(EngineSpec),
    /// Attach a transmission to an engine or the current tail of its output chain.
    AttachTransmission {
        /// Engine or transmission whose local positive-Z face receives the block.
        parent: PartId,
        /// Candidate block. Its pose must exactly continue the root engine orientation.
        spec: TransmissionSpec,
    },
    /// Spawn a servo.
    SpawnServo(ServoSpec),
    /// Spawn a seat cushion.
    SpawnSeat(SeatSpec),
    /// Spawn an Input block.
    SpawnInput(InputSpec),
    /// Spawn a Dimension Link with a stable per-world identity.
    SpawnDimensionLink(DimensionLinkSpec),
    /// Add a passive bearing.
    AddBearing(BearingSpec),
    /// Updates all attachments sharing one suspension without changing bearing IDs.
    SetSuspension {
        /// Representative bearing in the mounting assembly.
        bearing: BearingId,
        /// Validated replacement, preserving attached mount spacing.
        spec: crate::SuspensionSpec,
    },
    /// Wire a control block to one bearing.
    AddDriveLink(DriveLinkSpec),
    /// Changes the typed SI-unit envelope of a linear bearing.
    SetLinearDriveLimits {
        /// Wire to update.
        link: DriveLinkId,
        /// New limits, contained by physical travel.
        limits: crate::LinearDriveLimits,
    },
    /// Remove one control-block wire, leaving its endpoints intact.
    RemoveDriveLink(DriveLinkId),
    /// Link one Input block to one Seat.
    AddInputSeatLink(InputSeatLinkSpec),
    /// Remove an Input-to-Seat link.
    RemoveInputSeatLink(InputSeatLinkId),
    /// Link one Seat to one Controller.
    AddSeatControllerLink(SeatControllerLinkSpec),
    /// Remove a Seat-to-Controller link.
    RemoveSeatControllerLink(SeatControllerLinkId),
    /// Replace one drive wire's limits and program.
    SetDriveLink {
        /// Wire being reprogrammed.
        link: DriveLinkId,
        /// Replacement speed, torque, and travel envelope.
        limits: DriveLimits,
        /// Replacement state program.
        program: DriveProgram,
        /// Replacement joint name.
        name: DriveName,
        /// Replacement actuator assignment.
        actuator: ActuatorAssignment,
    },
    /// Change automatic/manual shifting for one controller engine lane.
    SetGearboxMode {
        /// Controller owning the lane.
        controller: PartId,
        /// Engine family being edited.
        kind: EngineKind,
        /// Replacement mode.
        mode: ShiftMode,
    },
    /// Replace every ratio in one controller engine lane.
    SetGearboxRatios {
        /// Controller owning the lane.
        controller: PartId,
        /// Engine family being edited.
        kind: EngineKind,
        /// Strictly descending input-to-output ratios.
        ratios: Vec<f32>,
    },
    /// Replace the manual shift bindings for one controller engine lane.
    SetGearboxBindings {
        /// Controller owning the lane.
        controller: PartId,
        /// Engine family being edited.
        kind: EngineKind,
        /// Upshift chord.
        up: GearKeyChord,
        /// Downshift chord.
        down: GearKeyChord,
    },
    /// Move the divider between reverse and forward gas gears.
    SetGasDivider {
        /// Controller owning the gas lane.
        controller: PartId,
        /// Number of ratios on the reverse side.
        reverse_gears: u8,
    },
    /// Claim a solid cuboid of blocks as an editable shape region.
    AddRegion(ShapeRegion),
    /// Release a region, returning its blocks to their own box geometry.
    RemoveRegion(RegionId),
    /// Move cage vertices. Applied as one batch so a group drag, or an edit
    /// expanded across the mirror planes, stays a single undo entry.
    SetRegionVertices {
        /// Region being shaped.
        region: RegionId,
        /// Vertices and their new displacements.
        vertices: Vec<(CageIndex, [i16; 3])>,
    },
    /// Insert a cage plane, giving the region a new row of handles.
    SubdivideRegion {
        /// Region being subdivided.
        region: RegionId,
        /// Axis to split.
        axis: usize,
        /// Position along that axis, in cells from the region origin.
        position: i32,
    },
    /// Append one parametric chamfer or fillet in global replay order.
    AddShapeFeature(ShapeFeature),
    /// Replace one feature's positive equal setback or radius.
    SetShapeFeatureAmount {
        /// Feature being adjusted.
        feature: ShapeFeatureId,
        /// Replacement amount in exact position ticks.
        amount_ticks: u32,
    },
    /// Remove one feature when all downstream features still replay.
    RemoveShapeFeature(ShapeFeatureId),
    /// Record a non-mutating first endpoint for a two-step tool.
    BeginPending(PendingOperation),
    /// Cancel the current incomplete tool operation.
    CancelPending,
}

/// Value returned by a successful build command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildOutcome {
    /// A part was created.
    Spawned(PartId),
    /// A part and its incident connections were removed.
    Removed,
    /// A weld was created.
    Welded(WeldId),
    /// A non-geometric rigid link was created.
    RigidLinked(RigidLinkId),
    /// A bearing was created.
    BearingAdded(BearingId),
    /// A control-block wire was created.
    DriveLinked(DriveLinkId),
    /// An Input-to-Seat link was created.
    InputSeatLinked(InputSeatLinkId),
    /// A Seat-to-Controller link was created.
    SeatControllerLinked(SeatControllerLinkId),
    /// A drive wire's limits or program were replaced.
    DriveUpdated,
    /// A persistent gearbox setting was replaced.
    GearboxUpdated,
    /// A region was claimed.
    RegionAdded(RegionId),
    /// A region's cage changed.
    RegionUpdated,
    /// A parametric shape feature was appended.
    ShapeFeatureAdded(ShapeFeatureId),
    /// A parametric shape feature changed or was removed.
    ShapeFeatureUpdated,
    /// A part or region appearance changed.
    AppearanceUpdated,
    /// A cylinder's envelope or material bands changed.
    CylinderUpdated,
    /// A pending operation was recorded.
    Pending,
    /// A pending operation was cancelled, or there was nothing to cancel.
    Cancelled,
}

/// Validation failure. Failed commands leave the graph byte-for-byte equivalent.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum GraphError {
    /// Cylinder material bands do not fit the cylinder envelope.
    #[error(transparent)]
    CylinderBands(crate::CylinderDimensionError),
    /// A cylinder-only edit named another kind of part.
    #[error("part {0:?} is not a cylinder")]
    NotCylinder(PartId),
    /// A cylinder reshape tried to move the cylinder.
    #[error("reshaping cylinder {0:?} cannot change its pose")]
    CylinderMoved(PartId),
    /// A rail already has an attachment on a different carriage face.
    #[error("a linear carriage can only have one occupied attachment face")]
    LinearCarriageOccupied,
    /// Program units or programmable travel do not match the physical joint.
    #[error(
        "drive targets and limits must match the bearing kind and remain inside physical travel"
    )]
    IncompatibleDrive,
    /// Invalid linear-bearing frame.
    #[error(transparent)]
    LinearBearing(#[from] crate::LinearBearingError),
    /// Invalid suspension geometry or attempted powered suspension.
    #[error(transparent)]
    Suspension(#[from] crate::SuspensionError),
    /// A part handle is stale or unknown.
    #[error("unknown or stale part handle {0:?}")]
    MissingPart(PartId),
    /// A weld handle is stale or unknown.
    #[error("unknown or stale weld handle {0:?}")]
    MissingWeld(WeldId),
    /// A rigid-link handle is stale or unknown.
    #[error("unknown or stale rigid-link handle {0:?}")]
    MissingRigidLink(RigidLinkId),
    /// A bearing handle is stale or unknown.
    #[error("unknown or stale bearing handle {0:?}")]
    MissingBearing(BearingId),
    /// A drive-link handle is stale or unknown.
    #[error("unknown or stale drive-link handle {0:?}")]
    MissingDriveLink(DriveLinkId),
    /// An Input-to-Seat link handle is stale or unknown.
    #[error("unknown or stale Input-to-Seat link handle {0:?}")]
    MissingInputSeatLink(InputSeatLinkId),
    /// A Seat-to-Controller link handle is stale or unknown.
    #[error("unknown or stale Seat-to-Controller link handle {0:?}")]
    MissingSeatControllerLink(SeatControllerLinkId),
    /// A part referenced as a control block is a different kind of part.
    #[error("part {0:?} is not a control block")]
    NotAController(PartId),
    /// A transmission parent is neither an engine nor a transmission.
    #[error("part {0:?} cannot carry a transmission")]
    InvalidTransmissionParent(PartId),
    /// A transmission can only extend the current chain tail.
    #[error("part {0:?} already has a transmission on its positive-Z output")]
    TransmissionOutputOccupied(PartId),
    /// Transmission candidate pose did not exactly continue the engine output axis.
    #[error(
        "a transmission must inherit its root engine orientation and attach -Z to the parent +Z face"
    )]
    InvalidTransmissionPose,
    /// A transmission chain reached its supported block limit.
    #[error("an engine transmission chain supports at most 17 blocks")]
    TransmissionLimitReached,
    /// A required transmission weld cannot be removed independently.
    #[error("weld {0:?} is required by a transmission; remove the transmission instead")]
    RequiredTransmissionWeld(WeldId),
    /// The requested controller/type has no editable, unambiguous gearbox.
    #[error("controller {controller:?} has no editable {kind:?} gearbox")]
    GearboxUnavailable {
        /// Controller being edited.
        controller: PartId,
        /// Engine family being edited.
        kind: EngineKind,
    },
    /// Same-type engines cannot share gearing until every chain has equal depth.
    #[error("controller {controller:?} has mismatched {kind:?} transmission depths {depths:?}")]
    TransmissionDepthMismatch {
        /// Controller identifying the machine module.
        controller: PartId,
        /// Engine family with inconsistent stacks.
        kind: EngineKind,
        /// Sorted physical depths found in the module.
        depths: Vec<u8>,
    },
    /// A gearbox edit did not satisfy ratio or divider invariants.
    #[error(transparent)]
    InvalidGearbox(#[from] GearboxError),
    /// A ratio edit had a different count than the physical stack provides.
    #[error("gearbox needs {expected} ratios for its transmission depth, but got {actual}")]
    GearCountMismatch {
        /// Required physical gear count.
        expected: usize,
        /// Supplied ratio count.
        actual: usize,
    },
    /// A part referenced as an Input block is a different kind of part.
    #[error("part {0:?} is not an Input block")]
    NotAnInput(PartId),
    /// Another Dimension Link in this graph already carries the same world identity.
    #[error("Dimension Link ID {0:?} already exists in this construction")]
    DuplicateDimensionLink(DimensionLinkId),
    /// A part referenced as a Seat is a different kind of part.
    #[error("part {0:?} is not a Seat")]
    NotASeat(PartId),
    /// An Input block already serves another Seat.
    #[error("Input block {0:?} is already linked to a Seat")]
    InputAlreadyLinked(PartId),
    /// A Seat already has an Input block.
    #[error("Seat {0:?} already has an Input link")]
    SeatAlreadyHasInput(PartId),
    /// A Seat already has a Controller.
    #[error("Seat {0:?} already has a Controller link")]
    SeatAlreadyHasController(PartId),
    /// A bearing already obeys another control block.
    #[error("bearing {0:?} is already driven by a control block")]
    BearingAlreadyDriven(BearingId),
    /// Only the positive-y ground face exists.
    #[error("the ground only exposes its positive-y face")]
    InvalidGroundFace,
    /// Cylinders expose only their two flat local-Y ends as connection faces.
    #[error("cylinders expose only their positive-y and negative-y flat ends")]
    InvalidCylinderFace,
    /// Pipe bends expose only their local negative-X inlet and positive-Y outlet.
    #[error("pipe bends expose only their negative-x inlet and positive-y outlet")]
    InvalidPipeBendFace,
    /// Pipe junctions expose only the ends of their open arms.
    #[error("pipe junctions expose only the ends of their open arms")]
    InvalidPipeJunctionFace,
    /// A connection selected the same endpoint twice.
    #[error("a connection requires two distinct faces")]
    SameFace,
    /// A rigid link selected the same part twice.
    #[error("a rigid link requires two distinct parts")]
    SameRigidLinkPart,
    /// The selected weld faces are not coplanar, opposed, and overlapping.
    #[error("weld faces do not touch over a positive area")]
    FacesDoNotTouch,
    /// Bearing faces do not have opposite normals.
    #[error("bearing endpoint faces are not opposed")]
    BearingFacesNotOpposed,
    /// Bearing anchor misses its source face or its ring misses the target face.
    #[error("bearing anchor or ring does not overlap the selected endpoint faces")]
    BearingAnchorOutsideFaces,
    /// Stored bearing axis is not a finite unit source-face normal.
    #[error("bearing axis must be finite, unit length, and equal the source-face normal")]
    InvalidBearingAxis,
    /// A bearing cannot connect a face to the ground in this milestone.
    #[error("bearings require two part endpoints")]
    BearingOnGround,
    /// A region handle is stale or unknown.
    #[error("region {0:?} is not live")]
    MissingRegion(RegionId),
    /// A shape-feature handle is stale or unknown.
    #[error("shape feature {0:?} is not live")]
    MissingShapeFeature(ShapeFeatureId),
    /// A connection references a planar patch no longer present after replay.
    #[error("surface patch {patch:?} is not present on {owner:?}")]
    MissingSurfacePatch {
        /// Solid expected to carry the patch.
        owner: SolidOwner,
        /// Stable missing patch.
        patch: crate::SurfacePatchKey,
    },
    /// A feature target owner is absent or is authored machine geometry.
    #[error("shape feature target {0:?} is not an editable construction solid")]
    InvalidShapeFeatureOwner(SolidOwner),
    /// Parametric solid evaluation rejected a feature or downstream replay.
    #[error(transparent)]
    InvalidSolid(#[from] SolidError),
    /// Appearance editing was requested for an authored machine part.
    #[error("part {0:?} has an authored appearance")]
    AuthoredAppearance(PartId),
    /// A region rejected the change.
    #[error(transparent)]
    InvalidRegion(#[from] RegionError),
    /// The chosen area is not a solid cuboid of blocks.
    #[error("a region needs every cell filled by a block; {0} are empty")]
    RegionNotSolid(usize),
    /// The chosen area mixes materials.
    #[error("a region must be one material throughout")]
    RegionMixedMaterials,
    /// The chosen area mixes color or finish treatments.
    #[error("a region must have one appearance throughout")]
    RegionMixedAppearances,
    /// The chosen area spans more than one rigid body.
    #[error("a region must lie within one rigid body")]
    RegionSpansBodies,
    /// The chosen area overlaps a region that already exists.
    #[error("that area overlaps region {0:?}")]
    RegionOverlaps(RegionId),
    /// The chosen area holds part of a block but not all of it.
    #[error("a region must contain each of its blocks whole")]
    RegionSplitsPart,
    /// A standalone featured part must have its features removed before a
    /// Shape region can take ownership of it.
    #[error("part {0:?} has Shape features; remove them before claiming a region")]
    FeaturedPartInRegion(PartId),
    /// A cage move would turn one of the region's cells inside out.
    #[error("region {0:?} would have a cell turned inside out")]
    InvertedCell(RegionId),
}

/// Editable, CPU-owned construction topology.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct ConstructionGraphData {
    solid_cache: solid_cache::SolidCache,
    pub(crate) construction_frames: crate::frame::ConstructionFrames,
    pub(crate) parts: Arena<PartSpec, PartId>,
    pub(crate) welds: Arena<WeldSpec, WeldId>,
    pub(crate) rigid_links: Arena<RigidLinkSpec, RigidLinkId>,
    pub(crate) bearings: Arena<BearingSpec, BearingId>,
    pub(crate) drive_links: Arena<DriveLinkSpec, DriveLinkId>,
    pub(crate) input_seat_links: Arena<InputSeatLinkSpec, InputSeatLinkId>,
    pub(crate) seat_controller_links: Arena<SeatControllerLinkSpec, SeatControllerLinkId>,
    /// Transmission part to its engine-or-transmission parent.
    pub(crate) transmission_parents: BTreeMap<PartId, PartId>,
    /// Transmission part to the weld created atomically with it.
    pub(crate) transmission_welds: BTreeMap<PartId, WeldId>,
    /// Persistent per-controller, per-engine-family gearbox overrides.
    pub(crate) gearbox_configs: BTreeMap<(PartId, EngineKind), GearboxConfig>,
    /// Editable shape regions. A region owns the geometry of the blocks it
    /// covers, so those blocks stop emitting boxes of their own.
    pub(crate) regions: Arena<ShapeRegion, RegionId>,
    /// Shape features in stable-handle storage.
    pub(crate) shape_features: Arena<ShapeFeature, ShapeFeatureId>,
    /// Explicit global replay order, independent of arena slot reuse.
    pub(crate) shape_feature_order: Vec<ShapeFeatureId>,
    pending: Option<PendingOperation>,
}

/// Copy-on-write editable construction graph.
///
/// Clones are immutable shared revisions. The first mutation of a staged edit
/// copies graph storage once, which keeps history capture constant-time.
#[derive(Clone, Debug, Default)]
pub struct ConstructionGraph {
    data: Arc<ConstructionGraphData>,
}

impl core::ops::Deref for ConstructionGraph {
    type Target = ConstructionGraphData;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl core::ops::DerefMut for ConstructionGraph {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.data)
    }
}

/// One privately owned graph revision being staged with a single deep clone.
///
/// Dropping an edit discards every mutation. Call [`Self::finish`] only after
/// every command succeeds.
#[derive(Clone, Debug)]
pub struct ConstructionGraphEdit {
    graph: ConstructionGraph,
}

/// Complete structural assembly reached from one seed part.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralComponent {
    parts: BTreeSet<PartId>,
    touches_authored_ground: bool,
}

impl StructuralComponent {
    /// Whether this assembly contains a part.
    pub fn contains(&self, part: PartId) -> bool {
        self.parts.contains(&part)
    }

    /// Parts in stable graph order.
    pub fn parts(&self) -> impl Iterator<Item = PartId> + '_ {
        self.parts.iter().copied()
    }

    /// Whether a graph weld or caller-provided terrain anchor grounds the assembly.
    pub const fn touches_authored_ground(&self) -> bool {
        self.touches_authored_ground
    }
}

/// A construction separated into the selected assembly and everything left behind.
#[derive(Clone, Debug)]
pub struct GraphPartition {
    /// Graph with the selected structural assembly removed.
    pub remainder: ConstructionGraph,
    /// Graph containing only the selected structural assembly.
    pub component: ConstructionGraph,
}

impl ConstructionGraphEdit {
    /// Reserves storage for a known bulk edit without changing graph contents.
    pub fn reserve_parts_and_welds(&mut self, parts: usize, welds: usize) {
        self.graph.parts.reserve(parts);
        self.graph.welds.reserve(welds);
    }

    /// Inserts an already placement-validated cuboid batch and returns stable IDs.
    pub fn spawn_cuboids(&mut self, specs: impl IntoIterator<Item = CuboidSpec>) -> Vec<PartId> {
        specs
            .into_iter()
            .map(|spec| self.graph.insert_edit_part(spec.into()))
            .collect()
    }

    /// Applies one validated command without cloning the staged graph again.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError`] when the command violates graph or geometry invariants.
    pub fn apply(&mut self, command: BuildCommand) -> Result<BuildOutcome, GraphError> {
        self.graph.apply_validated(command)
    }

    /// Applies ordered validated commands without cloning the staged graph again.
    ///
    /// # Errors
    ///
    /// Returns the first [`GraphError`]. The caller must discard this edit on failure.
    pub fn apply_batch(
        &mut self,
        commands: impl IntoIterator<Item = BuildCommand>,
    ) -> Result<Vec<BuildOutcome>, GraphError> {
        commands
            .into_iter()
            .map(|command| self.graph.apply_validated(command))
            .collect()
    }

    /// Applies validated commands while discarding outcomes the caller does not need.
    ///
    /// # Errors
    ///
    /// Returns the first [`GraphError`]. The caller must discard this edit on failure.
    pub fn apply_batch_discarding_outcomes(
        &mut self,
        commands: impl IntoIterator<Item = BuildCommand>,
    ) -> Result<(), GraphError> {
        for command in commands {
            self.graph.apply_validated(command)?;
        }
        Ok(())
    }

    /// Commits the staged revision to its caller.
    pub fn finish(self) -> ConstructionGraph {
        self.graph
    }
}

impl core::ops::Deref for ConstructionGraphEdit {
    type Target = ConstructionGraph;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl ConstructionGraph {
    /// Whether two handles still refer to the same immutable graph revision.
    pub fn shares_revision(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }

    pub(crate) fn transform_bearing_space(&mut self, transform: crate::ConstructionFrame) {
        let bearings = self
            .bearings
            .iter()
            .map(|(id, bearing)| (id, *bearing))
            .collect::<Vec<_>>();
        for (id, mut bearing) in bearings {
            bearing.shared_anchor = transform.point(bearing.shared_anchor);
            bearing.axis = transform.vector(bearing.axis);
            if let crate::BearingKind::Linear(rail) = &mut bearing.kind {
                rail.mount_normal = transform.vector(rail.mount_normal);
            }
            if let Some(destination) = self.bearings.get_mut(id) {
                *destination = bearing;
            }
        }
        if let Some(PendingOperation::Bearing { anchor, .. }) = &mut self.pending {
            *anchor = transform.point(*anchor);
        }
    }

    /// Creates an empty graph in paused build mode.
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts a transaction by deep-cloning this revision exactly once.
    pub fn begin_edit(&self) -> ConstructionGraphEdit {
        ConstructionGraphEdit {
            graph: self.clone(),
        }
    }

    /// Traverses every structural connection from `seed`.
    ///
    /// Welds, rigid links, bearings, and shared shape regions all join the
    /// component. `authored_ground_parts` supplies terrain-foundation anchors
    /// owned by the application in addition to explicit graph ground welds.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::MissingPart`] when `seed` is not live in this graph.
    pub fn structural_component(
        &self,
        seed: PartId,
        authored_ground_parts: impl IntoIterator<Item = PartId>,
    ) -> Result<StructuralComponent, GraphError> {
        self.parts.get(seed).ok_or(GraphError::MissingPart(seed))?;
        let authored_ground_parts = authored_ground_parts.into_iter().collect::<BTreeSet<_>>();
        let mut parts = BTreeSet::from([seed]);
        let mut frontier = vec![seed];
        let mut touches_authored_ground = false;
        while let Some(part) = frontier.pop() {
            touches_authored_ground |= authored_ground_parts.contains(&part);
            let mut visit = |other: PartId| {
                if parts.insert(other) {
                    frontier.push(other);
                }
            };
            for (_, weld) in self.welds.iter() {
                match (weld.first.owner, weld.second.owner) {
                    (FaceOwner::Part(first), FaceOwner::Part(second)) if first == part => {
                        visit(second);
                    }
                    (FaceOwner::Part(first), FaceOwner::Part(second)) if second == part => {
                        visit(first);
                    }
                    (FaceOwner::Part(candidate), FaceOwner::Ground)
                    | (FaceOwner::Ground, FaceOwner::Part(candidate))
                        if candidate == part =>
                    {
                        touches_authored_ground = true;
                    }
                    _ => {}
                }
            }
            for (_, link) in self.rigid_links.iter() {
                if link.first == part {
                    visit(link.second);
                } else if link.second == part {
                    visit(link.first);
                }
            }
            for (_, bearing) in self.bearings.iter() {
                let endpoints = match (bearing.source.owner, bearing.target.owner) {
                    (FaceOwner::Part(first), FaceOwner::Part(second)) => Some((first, second)),
                    _ => None,
                };
                if let Some((first, second)) = endpoints {
                    if first == part {
                        visit(second);
                    } else if second == part {
                        visit(first);
                    }
                }
            }
            if let Some(region) = self.region_of(part) {
                let members = self
                    .parts()
                    .filter_map(|(candidate, _)| {
                        (self.region_of(candidate) == Some(region)).then_some(candidate)
                    })
                    .collect::<Vec<_>>();
                for member in members {
                    visit(member);
                }
            }
        }
        Ok(StructuralComponent {
            parts,
            touches_authored_ground,
        })
    }

    /// Splits one complete structural assembly from this graph while retaining
    /// every connection, program, region, and graph-owned relation on its side.
    ///
    /// # Panics
    ///
    /// Panics only if removing a live part from an internally valid cloned graph fails.
    pub fn partition(&self, component: &StructuralComponent) -> GraphPartition {
        let all_parts = self.parts().map(|(part, _)| part).collect::<Vec<_>>();
        let mut remainder = self.clone();
        let mut extracted = self.clone();
        for part in &all_parts {
            if component.contains(*part) {
                if remainder.part(*part).is_some() {
                    remainder
                        .apply(BuildCommand::Remove(*part))
                        .expect("partition removes live parts");
                }
            } else if extracted.part(*part).is_some() {
                extracted
                    .apply(BuildCommand::Remove(*part))
                    .expect("partition removes live parts");
            }
        }
        GraphPartition {
            remainder,
            component: extracted,
        }
    }

    /// Applies an edit transactionally. On failure, no mutation is retained.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError`] when a handle is stale or connection geometry is invalid.
    pub fn apply(&mut self, command: BuildCommand) -> Result<BuildOutcome, GraphError> {
        let mut staged = self.clone();
        let outcome = staged.apply_validated(command)?;
        *self = staged;
        Ok(outcome)
    }

    /// Applies a batch atomically while cloning the graph only once. This is
    /// intended for benchmark scene generation and bulk UI paste operations.
    ///
    /// # Errors
    ///
    /// Returns the first [`GraphError`] and retains none of the batch mutations.
    pub fn apply_batch(
        &mut self,
        commands: impl IntoIterator<Item = BuildCommand>,
    ) -> Result<Vec<BuildOutcome>, GraphError> {
        let mut staged = self.clone();
        let outcomes = commands
            .into_iter()
            .map(|command| staged.apply_validated(command))
            .collect::<Result<Vec<_>, _>>()?;
        *self = staged;
        Ok(outcomes)
    }

    /// Every live shape region.
    pub fn regions(&self) -> impl Iterator<Item = (RegionId, &ShapeRegion)> {
        self.regions.iter()
    }

    /// Parametric edge features in explicit replay order.
    pub fn shape_features(&self) -> impl Iterator<Item = (ShapeFeatureId, &ShapeFeature)> {
        self.shape_feature_order
            .iter()
            .filter_map(|&id| self.shape_features.get(id).map(|feature| (id, feature)))
    }

    /// Retrieves one live parametric edge feature.
    pub fn shape_feature(&self, id: ShapeFeatureId) -> Option<&ShapeFeature> {
        self.shape_features.get(id)
    }

    /// Evaluates one construction owner from its base plus ordered features.
    ///
    /// # Errors
    ///
    /// Returns the first feature replay or base-generation failure.
    pub fn evaluated_solid(&self, owner: SolidOwner) -> Result<crate::EvaluatedSolid, GraphError> {
        self.evaluated_solid_shared(owner)
            .map(|solid| (*solid).clone())
    }

    /// Shares the cached boundary for read-only picking, validation and rendering.
    /// The cache compares the base geometry, ordered features and construction
    /// frame, so immutable revisions can safely reuse unchanged owners.
    ///
    /// # Errors
    /// Returns the first feature replay or base-generation failure.
    pub fn evaluated_solid_shared(
        &self,
        owner: SolidOwner,
    ) -> Result<Arc<crate::EvaluatedSolid>, GraphError> {
        self.evaluated_solid_until(owner, None)
    }

    /// Evaluates the source boundary seen by an existing feature, excluding
    /// that feature and every downstream record.
    ///
    /// # Errors
    ///
    /// Returns an error when the feature is missing or the owner's base and
    /// preceding features cannot be evaluated.
    pub fn evaluated_solid_before(
        &self,
        owner: SolidOwner,
        feature: ShapeFeatureId,
    ) -> Result<crate::EvaluatedSolid, GraphError> {
        if self.shape_features.get(feature).is_none() {
            return Err(GraphError::MissingShapeFeature(feature));
        }
        self.evaluated_solid_until(owner, Some(feature))
            .map(|solid| (*solid).clone())
    }

    fn evaluated_solid_until(
        &self,
        owner: SolidOwner,
        stop_before: Option<ShapeFeatureId>,
    ) -> Result<Arc<crate::EvaluatedSolid>, GraphError> {
        let features = self
            .shape_features()
            .take_while(|(id, _)| Some(*id) != stop_before)
            .filter_map(|(id, feature)| {
                let targets = feature
                    .targets
                    .iter()
                    .copied()
                    .filter(|target| target.owner == owner)
                    .collect::<Vec<_>>();
                (!targets.is_empty()).then_some({
                    (
                        id,
                        ShapeFeature {
                            targets,
                            treatment: feature.treatment,
                            amount_ticks: feature.amount_ticks,
                        },
                    )
                })
            })
            .collect::<Vec<_>>();
        let base = match owner {
            SolidOwner::Part(part) => {
                let spec = self.parts.get(part).ok_or(GraphError::MissingPart(part))?;
                solid_cache::SolidBase::Part(spec)
            }
            SolidOwner::Region(region) => {
                let shape = self
                    .regions
                    .get(region)
                    .ok_or(GraphError::MissingRegion(region))?;
                solid_cache::SolidBase::Region(shape)
            }
        };
        self.solid_cache
            .evaluate(owner, base, &features, self.owner_frame(owner))
    }

    /// Whether an owner has any committed parametric feature.
    pub fn owner_has_shape_features(&self, owner: SolidOwner) -> bool {
        self.shape_features()
            .any(|(_, feature)| feature.targets.iter().any(|target| target.owner == owner))
    }

    /// One live shape region.
    pub fn region(&self, id: RegionId) -> Option<&ShapeRegion> {
        self.regions.get(id)
    }

    /// The region covering this part, when one does.
    pub fn region_of(&self, part: PartId) -> Option<RegionId> {
        let spec = self.parts.get(part)?;
        let frame = self.part_frame_id(part)?;
        let cuboid = spec.as_cuboid()?;
        let cells = crate::part_cells(cuboid);
        let origin = cells.corner_steps(IVec3::ZERO, 0);
        self.regions.iter().find_map(|(id, region)| {
            (self.region_frame_id(id) == Some(frame) && region.covers_cell(origin)).then_some(id)
        })
    }

    /// Retrieves a live construction part.
    pub fn part(&self, id: PartId) -> Option<&PartSpec> {
        self.parts.get(id)
    }

    /// Retrieves a live weld.
    pub fn weld(&self, id: WeldId) -> Option<&WeldSpec> {
        self.welds.get(id)
    }

    /// Retrieves a live non-geometric rigid link.
    pub fn rigid_link(&self, id: RigidLinkId) -> Option<&RigidLinkSpec> {
        self.rigid_links.get(id)
    }

    /// Retrieves a live bearing.
    pub fn bearing(&self, id: BearingId) -> Option<&BearingSpec> {
        self.bearings.get(id)
    }

    /// Retrieves a live control-block wire.
    pub fn drive_link(&self, id: DriveLinkId) -> Option<&DriveLinkSpec> {
        self.drive_links.get(id)
    }

    /// Retrieves a live Input-to-Seat link.
    pub fn input_seat_link(&self, id: InputSeatLinkId) -> Option<&InputSeatLinkSpec> {
        self.input_seat_links.get(id)
    }

    /// Retrieves a live Seat-to-Controller link.
    pub fn seat_controller_link(
        &self,
        id: SeatControllerLinkId,
    ) -> Option<&SeatControllerLinkSpec> {
        self.seat_controller_links.get(id)
    }

    /// Whether a live part is a control block.
    pub fn is_controller(&self, part: PartId) -> bool {
        self.parts
            .get(part)
            .is_some_and(|spec| spec.as_controller().is_some())
    }

    /// Whether a live part is a Seat.
    pub fn is_seat(&self, part: PartId) -> bool {
        matches!(self.parts.get(part), Some(PartSpec::Seat(_)))
    }

    /// Whether a live part is an Input block.
    pub fn is_input(&self, part: PartId) -> bool {
        matches!(self.parts.get(part), Some(PartSpec::Input(_)))
    }

    /// Retrieves the Dimension Link identity carried by a part, when present.
    pub fn dimension_link_id(&self, part: PartId) -> Option<DimensionLinkId> {
        match self.parts.get(part)? {
            PartSpec::DimensionLink(link) => Some(link.id),
            _ => None,
        }
    }

    /// Finds the unique part carrying a Dimension Link identity.
    pub fn dimension_link(&self, id: DimensionLinkId) -> Option<PartId> {
        self.parts.iter().find_map(|(part, spec)| {
            matches!(spec, PartSpec::DimensionLink(link) if link.id == id).then_some(part)
        })
    }

    /// Input block linked to a Seat, when present.
    pub fn seat_input(&self, seat: PartId) -> Option<PartId> {
        self.input_seat_links
            .iter()
            .find_map(|(_, link)| (link.seat == seat).then_some(link.input))
    }

    /// Controller linked to a Seat, when present.
    pub fn seat_controller(&self, seat: PartId) -> Option<PartId> {
        self.seat_controller_links
            .iter()
            .find_map(|(_, link)| (link.seat == seat).then_some(link.controller))
    }

    /// The wire driving one bearing, when a control block owns it.
    pub fn bearing_drive_link(&self, bearing: BearingId) -> Option<(DriveLinkId, &DriveLinkSpec)> {
        self.drive_links
            .iter()
            .find(|(_, link)| link.bearing == bearing)
    }

    /// Every wire owned by one control block, in canonical slot order.
    pub fn controller_links(
        &self,
        controller: PartId,
    ) -> impl Iterator<Item = (DriveLinkId, &DriveLinkSpec)> {
        self.drive_links
            .iter()
            .filter(move |(_, link)| link.controller == controller)
    }

    /// Iterates live parts in canonical slot order.
    pub fn parts(&self) -> impl Iterator<Item = (PartId, &PartSpec)> {
        self.parts.iter()
    }

    /// Iterates live welds in canonical slot order.
    pub fn welds(&self) -> impl Iterator<Item = (WeldId, &WeldSpec)> {
        self.welds.iter()
    }

    /// Iterates live non-geometric rigid links in canonical slot order.
    pub fn rigid_links(&self) -> impl Iterator<Item = (RigidLinkId, &RigidLinkSpec)> {
        self.rigid_links.iter()
    }

    /// Iterates live bearings in canonical slot order.
    pub fn bearings(&self) -> impl Iterator<Item = (BearingId, &BearingSpec)> {
        self.bearings.iter()
    }

    /// Iterates live control-block wires in canonical slot order.
    pub fn drive_links(&self) -> impl Iterator<Item = (DriveLinkId, &DriveLinkSpec)> {
        self.drive_links.iter()
    }

    /// Iterates live Input-to-Seat links in canonical slot order.
    pub fn input_seat_links(&self) -> impl Iterator<Item = (InputSeatLinkId, &InputSeatLinkSpec)> {
        self.input_seat_links.iter()
    }

    /// Iterates live Seat-to-Controller links in canonical slot order.
    pub fn seat_controller_links(
        &self,
    ) -> impl Iterator<Item = (SeatControllerLinkId, &SeatControllerLinkSpec)> {
        self.seat_controller_links.iter()
    }

    /// Current incomplete two-step operation, if any.
    pub fn pending(&self) -> Option<PendingOperation> {
        self.pending
    }

    /// Number of live parts.
    pub fn part_count(&self) -> usize {
        self.parts.len()
    }

    /// Number of live welds.
    pub fn weld_count(&self) -> usize {
        self.welds.len()
    }

    /// Number of live non-geometric rigid links.
    pub fn rigid_link_count(&self) -> usize {
        self.rigid_links.len()
    }

    /// Number of live bearings.
    pub fn bearing_count(&self) -> usize {
        self.bearings.len()
    }

    /// Number of live control-block wires.
    pub fn drive_link_count(&self) -> usize {
        self.drive_links.len()
    }

    /// Parent of a transmission, when `part` is one in a live chain.
    pub fn transmission_parent(&self, part: PartId) -> Option<PartId> {
        self.transmission_parents.get(&part).copied()
    }

    /// Required weld belonging to a transmission.
    pub fn transmission_weld(&self, part: PartId) -> Option<WeldId> {
        self.transmission_welds.get(&part).copied()
    }

    /// Root engine and physical depth of a transmission chain member.
    pub fn transmission_root(&self, part: PartId) -> Option<(PartId, EngineKind, u8)> {
        let mut current = part;
        let mut depth = 0_u8;
        loop {
            match self.parts.get(current)? {
                PartSpec::Engine(engine) => return Some((current, engine.kind, depth)),
                PartSpec::Transmission(_) => {
                    current = *self.transmission_parents.get(&current)?;
                    depth = depth.checked_add(1)?;
                }
                _ => return None,
            }
        }
    }

    /// Number of transmission blocks downstream of one engine.
    pub fn engine_transmission_depth(&self, engine: PartId) -> Option<u8> {
        matches!(self.parts.get(engine), Some(PartSpec::Engine(_))).then_some(())?;
        let mut current = engine;
        let mut depth = 0_u8;
        while let Some((&child, _)) = self
            .transmission_parents
            .iter()
            .find(|(_, parent)| **parent == current)
        {
            current = child;
            depth = depth.saturating_add(1);
        }
        Some(depth)
    }

    /// Exact candidate pose which extends `parent` along the root engine's local +Z axis.
    ///
    /// # Errors
    ///
    /// Returns an error if `parent` is not an engine-line tail, its output is occupied,
    /// or the chain has reached the seventeen-block limit.
    pub fn next_transmission_spec(&self, parent: PartId) -> Result<TransmissionSpec, GraphError> {
        let parent_spec = self
            .parts
            .get(parent)
            .copied()
            .ok_or(GraphError::MissingPart(parent))?;
        let (root, _, depth) = match parent_spec {
            PartSpec::Engine(engine) => (parent, engine.kind, 0),
            PartSpec::Transmission(_) => self
                .transmission_root(parent)
                .ok_or(GraphError::InvalidTransmissionParent(parent))?,
            _ => return Err(GraphError::InvalidTransmissionParent(parent)),
        };
        if self
            .transmission_parents
            .values()
            .any(|candidate| *candidate == parent)
        {
            return Err(GraphError::TransmissionOutputOccupied(parent));
        }
        let output = FaceRef::part(parent, FaceKind::PositiveZ);
        if self
            .welds
            .iter()
            .any(|(_, weld)| weld.first == output || weld.second == output)
        {
            return Err(GraphError::TransmissionOutputOccupied(parent));
        }
        if depth >= 17 {
            return Err(GraphError::TransmissionLimitReached);
        }
        let Some(root_pose) = self.parts.get(root).copied().and_then(|spec| match spec {
            PartSpec::Engine(engine) => Some(engine.pose),
            _ => None,
        }) else {
            return Err(GraphError::InvalidTransmissionParent(parent));
        };
        let parent_z_units = match parent_spec {
            PartSpec::Engine(engine) => engine.kind.grid_units()[2],
            PartSpec::Transmission(_) => TransmissionSpec::GRID_UNITS[2],
            _ => unreachable!(),
        };
        let local_z = root_pose.rotation.quaternion() * Vec3::Z;
        let direction = local_z.round().as_ivec3();
        let centre = parent_spec.pose().translation_position_ticks()
            + direction
                * i32::from(parent_z_units + TransmissionSpec::GRID_UNITS[2])
                * crate::POSITION_TICKS_PER_HALF_GRID_UNIT;
        Ok(TransmissionSpec::new(BuildPose::from_position_ticks(
            centre,
            root_pose.rotation,
        )))
    }

    /// Derived authored appearance for one transmission.
    pub fn transmission_kind(&self, transmission: PartId) -> Option<EngineKind> {
        self.transmission_root(transmission)
            .map(|(_, kind, _)| kind)
    }

    /// Physical same-type engine depths in a Controller's direct machine module.
    pub fn transmission_depths(&self, controller: PartId, kind: EngineKind) -> Option<Vec<u8>> {
        self.is_controller(controller).then(|| {
            let mut depths = self
                .machine_module(controller)
                .into_iter()
                .filter_map(|part| match self.parts.get(part) {
                    Some(PartSpec::Engine(engine)) if engine.kind == kind => {
                        self.engine_transmission_depth(part)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            depths.sort_unstable();
            depths
        })
    }

    /// Effective gearbox settings, including defaults when no override was authored.
    ///
    /// # Errors
    ///
    /// Returns an error if the controller has no engines of `kind` or their physical
    /// transmission depths do not match.
    pub fn gearbox_config(
        &self,
        controller: PartId,
        kind: EngineKind,
    ) -> Result<GearboxConfig, GraphError> {
        let depth = self.common_transmission_depth(controller, kind)?;
        let gear_count = usize::from(depth) + 1;
        Ok(self
            .gearbox_configs
            .get(&(controller, kind))
            .cloned()
            .unwrap_or_else(|| GearboxConfig::for_depth(depth, kind == EngineKind::Gas))
            .resized(gear_count))
    }

    /// Explicit gearbox records in stable controller/type order.
    pub fn gearbox_configs(&self) -> impl Iterator<Item = ((PartId, EngineKind), &GearboxConfig)> {
        self.gearbox_configs
            .iter()
            .map(|(&(controller, kind), config)| ((controller, kind), config))
    }

    fn common_transmission_depth(
        &self,
        controller: PartId,
        kind: EngineKind,
    ) -> Result<u8, GraphError> {
        if !self.is_controller(controller) {
            return Err(if self.parts.get(controller).is_some() {
                GraphError::NotAController(controller)
            } else {
                GraphError::MissingPart(controller)
            });
        }
        let depths = self
            .transmission_depths(controller, kind)
            .expect("the controller was checked above");
        let Some(&depth) = depths.first() else {
            return Err(GraphError::GearboxUnavailable { controller, kind });
        };
        if depths.iter().any(|candidate| *candidate != depth) {
            return Err(GraphError::TransmissionDepthMismatch {
                controller,
                kind,
                depths,
            });
        }
        Ok(depth)
    }

    fn editable_gearbox(
        &self,
        controller: PartId,
        kind: EngineKind,
    ) -> Result<GearboxConfig, GraphError> {
        let depth = self.common_transmission_depth(controller, kind)?;
        if depth == 0 {
            return Err(GraphError::GearboxUnavailable { controller, kind });
        }
        self.gearbox_config(controller, kind)
    }

    /// Actuator hardware and graph-level assignment demand in a Controller's
    /// direct machine-only weld module.
    pub fn actuator_inventory(&self, controller: PartId) -> Option<ActuatorInventory> {
        self.is_controller(controller).then(|| {
            let members = self.machine_module(controller);
            let mut inventory = ActuatorInventory::default();
            for part in &members {
                match self.parts.get(*part) {
                    Some(PartSpec::Engine(engine)) => match engine.kind {
                        EngineKind::Electric => inventory.electric_engines += 1,
                        EngineKind::Gas => inventory.gas_engines += 1,
                    },
                    Some(PartSpec::Servo(_)) => inventory.servos += 1,
                    _ => {}
                }
            }
            for kind in [EngineKind::Electric, EngineKind::Gas] {
                let mut depths = members
                    .iter()
                    .filter_map(|part| match self.parts.get(*part) {
                        Some(PartSpec::Engine(engine)) if engine.kind == kind => {
                            self.engine_transmission_depth(*part)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                depths.sort_unstable();
                let mismatch = depths
                    .first()
                    .is_some_and(|first| depths.iter().any(|depth| depth != first));
                let common = (!mismatch).then(|| depths.first().copied()).flatten();
                match kind {
                    EngineKind::Electric => {
                        inventory.electric_transmission_depth = common;
                        inventory.electric_transmission_mismatch = mismatch;
                    }
                    EngineKind::Gas => {
                        inventory.gas_transmission_depth = common;
                        inventory.gas_transmission_mismatch = mismatch;
                    }
                }
            }
            let mut electric_coordinates = Vec::<(Vec3, Vec3)>::new();
            let mut gas_coordinates = Vec::<(Vec3, Vec3)>::new();
            let mut servo_coordinates = Vec::<(Vec3, Vec3)>::new();
            for (_, link) in self
                .drive_links
                .iter()
                .filter(|(_, link)| members.contains(&link.controller))
            {
                let Some(bearing) = self.bearings.get(link.bearing) else {
                    continue;
                };
                let coordinate = (bearing.shared_anchor, bearing.axis);
                let contains = |coordinates: &[(Vec3, Vec3)]| {
                    coordinates.iter().any(|(anchor, axis)| {
                        anchor.abs_diff_eq(coordinate.0, 1.0e-5)
                            && axis.abs_diff_eq(coordinate.1, 1.0e-5)
                    })
                };
                if link.actuator.uses_electric() && !contains(&electric_coordinates) {
                    electric_coordinates.push(coordinate);
                }
                if link.actuator.uses_gas() && !contains(&gas_coordinates) {
                    gas_coordinates.push(coordinate);
                }
                if link.actuator.uses_servo() && !contains(&servo_coordinates) {
                    servo_coordinates.push(coordinate);
                }
            }
            inventory.electric_joints =
                u32::try_from(electric_coordinates.len()).unwrap_or(u32::MAX);
            inventory.gas_joints = u32::try_from(gas_coordinates.len()).unwrap_or(u32::MAX);
            inventory.servo_joints = u32::try_from(servo_coordinates.len()).unwrap_or(u32::MAX);
            inventory
        })
    }

    /// Machine parts directly connected through machine-to-machine welds.
    pub(crate) fn machine_module(&self, start: PartId) -> BTreeSet<PartId> {
        let mut found = BTreeSet::from([start]);
        let mut pending = vec![start];
        while let Some(part) = pending.pop() {
            for (_, weld) in self.welds.iter() {
                let (FaceOwner::Part(first), FaceOwner::Part(second)) =
                    (weld.first.owner, weld.second.owner)
                else {
                    continue;
                };
                let next = if first == part {
                    second
                } else if second == part {
                    first
                } else {
                    continue;
                };
                if self.is_machine_part(part) && self.is_machine_part(next) && found.insert(next) {
                    pending.push(next);
                }
            }
        }
        found
    }

    fn is_machine_part(&self, part: PartId) -> bool {
        matches!(
            self.parts.get(part),
            Some(
                PartSpec::Controller(_)
                    | PartSpec::Engine(_)
                    | PartSpec::Transmission(_)
                    | PartSpec::Servo(_)
                    | PartSpec::DimensionLink(_)
            )
        )
    }

    /// Collects adjoining material on the selected rigid body's mating plane.
    ///
    /// # Errors
    /// Rejects missing faces or a construction that cannot compile.
    pub fn weld_mating_faces(
        &self,
        selected: FaceRef,
    ) -> Result<Vec<FaceRef>, crate::WeldRejection> {
        let reject = crate::WeldRejection::InvalidFeature;
        let target = self.face_geometry(selected).map_err(|_| reject)?;
        let FaceOwner::Part(part) = selected.owner else {
            return Ok(vec![selected]);
        };
        let creation = self.compile().map_err(|_| reject)?;
        let body = creation
            .part_to_compound
            .iter()
            .find_map(|&(id, body)| (id == part).then_some(body))
            .ok_or(reject)?;
        let mut faces = vec![selected];
        let mut seen = BTreeSet::new();
        for &part in &creation.compounds[body as usize].source_parts {
            let owner = self
                .region_of(part)
                .map_or(SolidOwner::Part(part), SolidOwner::Region);
            if !seen.insert(owner) {
                continue;
            }
            let candidates = if let Ok(solid) = self.evaluated_solid_shared(owner) {
                solid
                    .surfaces
                    .iter()
                    .filter(|surface| surface.smoothing_group == 0)
                    .map(|surface| FaceRef::patch(part, FaceKind::PositiveY, surface.key))
                    .collect::<Vec<_>>()
            } else {
                [
                    FaceKind::NegativeX,
                    FaceKind::PositiveX,
                    FaceKind::NegativeY,
                    FaceKind::PositiveY,
                    FaceKind::NegativeZ,
                    FaceKind::PositiveZ,
                ]
                .into_iter()
                .map(|kind| FaceRef::part(part, kind))
                .collect()
            };
            for face in candidates {
                if faces.contains(&face) {
                    continue;
                }
                if let Ok(geometry) = self.face_geometry(face)
                    && geometry.normal.abs_diff_eq(target.normal, 1.0e-5)
                    && (geometry.center - target.center).dot(target.normal).abs()
                        <= ANCHOR_TOLERANCE_METERS
                {
                    faces.push(face);
                }
            }
        }
        Ok(faces)
    }

    /// Checks that a transformed corner or finite edge remains incident to mating material.
    pub fn weld_feature_on_faces(
        &self,
        faces: &[FaceRef],
        feature: crate::WeldFeature,
        transform: crate::ConstructionFrame,
    ) -> bool {
        let ends = match feature {
            crate::WeldFeature::Face => return true,
            crate::WeldFeature::Vertex(point) => [transform.point(point); 2],
            crate::WeldFeature::Edge(ends) => ends.map(|point| transform.point(point)),
        };
        faces.iter().any(|&face| {
            let Ok(geometry) = self.face_geometry(face) else {
                return false;
            };
            if ends.iter().any(|point| {
                (*point - geometry.center).dot(geometry.normal).abs() > ANCHOR_TOLERANCE_METERS
            }) {
                return false;
            }
            if matches!(geometry.profile, FaceProfile::Ground) {
                return true;
            }
            let points = ends.map(|point| {
                let offset = point - geometry.center;
                Vec2::new(
                    offset.dot(geometry.tangent_u),
                    offset.dot(geometry.tangent_v),
                )
            });
            self.weld_material_cells(
                face,
                &geometry,
                crate::ConstructionFrame::IDENTITY,
                geometry.normal,
            )
            .iter()
            .any(|polygon| {
                let winding = polygon
                    .iter()
                    .zip(polygon.iter().cycle().skip(1))
                    .map(|(a, b)| a.perp_dot(*b))
                    .sum::<f32>()
                    .signum();
                let mut low = 0.0_f32;
                let mut high = 1.0_f32;
                for (a, b) in polygon.iter().zip(polygon.iter().cycle().skip(1)) {
                    let normal = (*b - *a).perp() * winding;
                    let start = (points[0] - *a).dot(normal);
                    let delta = (points[1] - points[0]).dot(normal);
                    if delta.abs() < 1.0e-10 {
                        if start < -1.0e-7 {
                            return false;
                        }
                    } else {
                        let boundary = -start / delta;
                        if delta > 0.0 {
                            low = low.max(boundary);
                        } else {
                            high = high.min(boundary);
                        }
                    }
                }
                low <= high + 1.0e-5
            })
        })
    }

    /// Validates continuous mating material containing a 5 × 5 cm square.
    /// Lists may include adjoining coplanar faces on the same respective rigid body.
    /// The first destination face defines the square's tangent axes and plane.
    ///
    /// # Errors
    /// Rejects missing faces, noncoplanar mating planes, and insufficient continuous material.
    pub fn weld_contact_square(
        &self,
        source: &[FaceRef],
        destination: &[FaceRef],
    ) -> Result<Vec3, crate::WeldRejection> {
        self.weld_contact_square_transformed(
            source,
            destination,
            crate::ConstructionFrame::IDENTITY,
            crate::ConstructionFrame::IDENTITY,
        )
    }

    /// Validates mating material under two independent body motions.
    ///
    /// # Errors
    /// Rejects missing faces, noncoplanar surfaces, or insufficient continuous contact.
    pub fn weld_contact_square_transformed(
        &self,
        source: &[FaceRef],
        destination: &[FaceRef],
        source_motion: crate::ConstructionFrame,
        destination_motion: crate::ConstructionFrame,
    ) -> Result<Vec3, crate::WeldRejection> {
        let rejection = crate::WeldRejection::InsufficientContact;
        let mut target = self
            .face_geometry(*destination.first().ok_or(rejection)?)
            .map_err(|_| crate::WeldRejection::InvalidFeature)?;
        target.center = destination_motion.point(target.center);
        target.normal = destination_motion.vector(target.normal);
        target.tangent_u = destination_motion.vector(target.tangent_u);
        target.tangent_v = destination_motion.vector(target.tangent_v);
        let collect = |faces: &[FaceRef], opposed: bool| {
            let motion = if opposed {
                source_motion
            } else {
                destination_motion
            };
            let expected = if opposed {
                -target.normal
            } else {
                target.normal
            };
            faces
                .iter()
                .flat_map(|&face| self.weld_material_cells(face, &target, motion, expected))
                .collect::<Vec<_>>()
        };
        let mut source_cells = collect(source, true);
        let mut destination_cells = collect(destination, false);
        // An infinite ground plane contributes no finite material boundary.
        if source.iter().any(|face| face.owner == FaceOwner::Ground)
            && target
                .normal
                .abs_diff_eq(-source_motion.vector(Vec3::Y), 1.0e-5)
        {
            source_cells.clone_from(&destination_cells);
        }
        if destination
            .iter()
            .any(|face| face.owner == FaceOwner::Ground)
        {
            destination_cells.clone_from(&source_cells);
        }
        let center = crate::weld_contact_square(&source_cells, &destination_cells)?;
        Ok(target.center + target.tangent_u * center.x + target.tangent_v * center.y)
    }

    fn weld_material_cells(
        &self,
        face: FaceRef,
        target: &FaceGeometry,
        motion: crate::ConstructionFrame,
        expected: Vec3,
    ) -> Vec<Vec<Vec2>> {
        let Ok(mut geometry) = self.face_geometry(face) else {
            return Vec::new();
        };
        geometry.center = motion.point(geometry.center);
        geometry.normal = motion.vector(geometry.normal);
        geometry.tangent_u = motion.vector(geometry.tangent_u);
        geometry.tangent_v = motion.vector(geometry.tangent_v);
        if !geometry.normal.abs_diff_eq(expected, 1.0e-5)
            || (geometry.center - target.center).dot(target.normal).abs() > ANCHOR_TOLERANCE_METERS
        {
            return Vec::new();
        }
        if let FaceOwner::Part(part) = face.owner {
            let owner = self
                .region_of(part)
                .map_or(SolidOwner::Part(part), SolidOwner::Region);
            if let Ok(solid) = self.evaluated_solid_shared(owner) {
                return solid
                    .surfaces
                    .iter()
                    .filter_map(|surface| {
                        if !motion.vector(surface.normal).abs_diff_eq(expected, 1.0e-5) {
                            return None;
                        }
                        let mut points = Vec::new();
                        let mut edge = surface.half_edge;
                        loop {
                            let half = solid.half_edges[edge as usize];
                            let offset = motion
                                .point(solid.vertices[half.origin as usize].position)
                                - target.center;
                            if offset.dot(target.normal).abs() > ANCHOR_TOLERANCE_METERS {
                                return None;
                            }
                            points.push(Vec2::new(
                                offset.dot(target.tangent_u),
                                offset.dot(target.tangent_v),
                            ));
                            edge = half.next;
                            if edge == surface.half_edge {
                                break;
                            }
                        }
                        Some(points)
                    })
                    .collect();
            }
        }
        profile_cells(&geometry, target.center, target.tangent_u, target.tangent_v)
    }

    pub(crate) fn face_geometry(&self, face: FaceRef) -> Result<FaceGeometry, GraphError> {
        match face.owner {
            FaceOwner::Part(part) => {
                let spec = self
                    .parts
                    .get(part)
                    .copied()
                    .ok_or(GraphError::MissingPart(part))?;
                let owner = self
                    .region_of(part)
                    .map_or(SolidOwner::Part(part), SolidOwner::Region);
                if face.patch.is_some() || self.owner_has_shape_features(owner) {
                    let patch = face
                        .patch
                        .unwrap_or_else(|| primitive_surface_patch(spec, face.face));
                    return self.evaluated_patch_geometry(owner, patch);
                }
                match spec {
                    PartSpec::Cuboid(spec) => Ok(cuboid_face(spec, face.face)),
                    PartSpec::Controller(spec) => Ok(cuboid_face(spec.cuboid(), face.face)),
                    PartSpec::Engine(spec) => Ok(cuboid_face(spec.cuboid(), face.face)),
                    PartSpec::Transmission(spec) => Ok(cuboid_face(spec.cuboid(), face.face)),
                    PartSpec::Servo(spec) => Ok(cuboid_face(spec.cuboid(), face.face)),
                    PartSpec::Seat(spec) => Ok(cuboid_face(spec.cuboid(), face.face)),
                    PartSpec::Input(spec) => Ok(cuboid_face(spec.cuboid(), face.face)),
                    PartSpec::DimensionLink(spec) => Ok(cuboid_face(spec.cuboid(), face.face)),
                    PartSpec::Cylinder(spec) => {
                        cylinder_face(spec, face.face).ok_or(GraphError::InvalidCylinderFace)
                    }
                    PartSpec::PipeBend(spec) => {
                        pipe_bend_face(spec, face.face).ok_or(GraphError::InvalidPipeBendFace)
                    }
                    PartSpec::PipeJunction(spec) => pipe_junction_face(spec, face.face)
                        .ok_or(GraphError::InvalidPipeJunctionFace),
                }
                .map(|mut geometry| {
                    let frame = self.part_frame(part).expect("face part exists");
                    geometry.center = frame.point(geometry.center);
                    geometry.normal = frame.vector(geometry.normal);
                    geometry.tangent_u = frame.vector(geometry.tangent_u);
                    geometry.tangent_v = frame.vector(geometry.tangent_v);
                    geometry
                })
            }
            FaceOwner::Ground if face.face == FaceKind::PositiveY => Ok(ground_face()),
            FaceOwner::Ground => Err(GraphError::InvalidGroundFace),
        }
    }

    fn evaluated_patch_geometry(
        &self,
        owner: SolidOwner,
        patch: crate::SurfacePatchKey,
    ) -> Result<FaceGeometry, GraphError> {
        let solid = self.evaluated_solid_shared(owner)?;
        let surface = solid
            .surfaces
            .iter()
            .find(|surface| surface.key == patch)
            .ok_or(GraphError::MissingSurfacePatch { owner, patch })?;
        let mut points = Vec::<Vec3>::new();
        let mut edge = surface.half_edge;
        loop {
            let half_edge = solid.half_edges[edge as usize];
            points.push(solid.vertices[half_edge.origin as usize].position);
            edge = half_edge.next;
            if edge == surface.half_edge {
                break;
            }
        }
        if points.len() < 3 {
            return Err(GraphError::MissingSurfacePatch { owner, patch });
        }
        let point_count = f32::from(
            u16::try_from(points.len())
                .map_err(|_| GraphError::MissingSurfacePatch { owner, patch })?,
        );
        let center = points.iter().copied().sum::<Vec3>() / point_count;
        let normal = surface.normal.normalize_or_zero();
        let tangent_u = normal.any_orthonormal_vector();
        let tangent_v = normal.cross(tangent_u);
        let vertices = points
            .iter()
            .map(|point| {
                let offset = *point - center;
                Vec2::new(offset.dot(tangent_u), offset.dot(tangent_v))
            })
            .collect();
        Ok(FaceGeometry {
            center,
            normal,
            tangent_u,
            tangent_v,
            profile: FaceProfile::Polygon { vertices },
        })
    }

    /// Whether a region's cage has turned any of its cells inside out.
    fn reject_inverted_region(&self, region: RegionId) -> Result<(), GraphError> {
        let shape = self
            .regions
            .get(region)
            .ok_or(GraphError::MissingRegion(region))?;
        let grid = shape.grid();
        if crate::shape::has_inverted_cell(&grid, &|cell, corner| shape.corner_steps(cell, corner))
        {
            return Err(GraphError::InvertedCell(region));
        }
        Ok(())
    }

    /// Checks the rules an area must satisfy before it can become a region,
    /// without adding one, so a drag can preview whether it would be accepted.
    ///
    /// # Errors
    ///
    /// Reports the first rule the area breaks.
    pub fn check_region_area(&self, region: &ShapeRegion) -> Result<(), GraphError> {
        self.validate_region_area(region, self.edit_frame_id())
    }

    /// Checks the rules an area must satisfy before it can become a region:
    /// every cell filled, one material, one rigid body, whole blocks only, and
    /// nothing already claiming the space.
    fn validate_region_area(
        &self,
        region: &ShapeRegion,
        frame: crate::ConstructionFrameId,
    ) -> Result<(), GraphError> {
        if let Some((id, _)) = self.regions.iter().find(|(id, existing)| {
            self.region_frame_id(*id) == Some(frame) && existing.overlaps(region)
        }) {
            return Err(GraphError::RegionOverlaps(id));
        }

        let mut occupants: BTreeMap<[i32; 3], (PartId, ConstructionMaterial, MaterialAppearance)> =
            BTreeMap::new();
        for (id, spec) in self.parts.iter() {
            if self.part_frame_id(id) != Some(frame) {
                continue;
            }
            let Some(cuboid) = spec.as_cuboid() else {
                continue;
            };
            // Only ordinary blocks make up a region; the fixed authored machine
            // parts are components, not material.
            if !matches!(spec, PartSpec::Cuboid(_)) {
                continue;
            }
            let cells = crate::part_cells(cuboid);
            let counts = cells.counts();
            for z in 0..counts.z {
                for y in 0..counts.y {
                    for x in 0..counts.x {
                        let corner = cells.corner_steps(IVec3::new(x, y, z), 0);
                        occupants
                            .insert(corner.to_array(), (id, cuboid.material, cuboid.appearance));
                    }
                }
            }
        }

        let size = region.size_cells();
        let origin = region.origin_steps();
        let mut material: Option<ConstructionMaterial> = None;
        let mut appearance: Option<MaterialAppearance> = None;
        let mut members: Vec<PartId> = Vec::new();
        let mut claimed: BTreeMap<PartId, i32> = BTreeMap::new();
        let mut empty = 0_usize;
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let corner = origin + IVec3::new(x, y, z) * crate::shape::STEPS_PER_CELL;
                    let Some(&(part, cell_material, cell_appearance)) =
                        occupants.get(&corner.to_array())
                    else {
                        empty += 1;
                        continue;
                    };
                    if *material.get_or_insert(cell_material) != cell_material {
                        return Err(GraphError::RegionMixedMaterials);
                    }
                    if *appearance.get_or_insert(cell_appearance) != cell_appearance {
                        return Err(GraphError::RegionMixedAppearances);
                    }
                    if !members.contains(&part) {
                        members.push(part);
                    }
                    *claimed.entry(part).or_default() += 1;
                }
            }
        }
        if empty > 0 {
            return Err(GraphError::RegionNotSolid(empty));
        }
        if material != Some(region.material()) {
            return Err(GraphError::RegionMixedMaterials);
        }
        if appearance != Some(region.appearance()) {
            return Err(GraphError::RegionMixedAppearances);
        }

        // A member hands its whole surface and mass to the region, so an area
        // that holds only some of a block's cells would lose the rest.
        for (&part, &cells) in &claimed {
            if self.owner_has_shape_features(SolidOwner::Part(part)) {
                return Err(GraphError::FeaturedPartInRegion(part));
            }
            let whole = self
                .parts
                .get(part)
                .and_then(|spec| spec.as_cuboid())
                .map_or(0, |cuboid| {
                    crate::part_cells(cuboid).counts().element_product()
                });
            if cells != whole {
                return Err(GraphError::RegionSplitsPart);
            }
        }

        // One rigid body: walking the welds from any member must reach them all.
        if let Some(&seed) = members.first() {
            let body = self.rigid_group(seed);
            if !members.iter().all(|part| body.contains(part)) {
                return Err(GraphError::RegionSpansBodies);
            }
        }
        Ok(())
    }

    /// Every part welded, directly or transitively, to `seed`.
    fn rigid_group(&self, seed: PartId) -> BTreeSet<PartId> {
        let mut reached = BTreeSet::from([seed]);
        let mut frontier = vec![seed];
        while let Some(part) = frontier.pop() {
            let mut visit = |other: PartId| {
                if reached.insert(other) {
                    frontier.push(other);
                }
            };
            for (_, weld) in self.welds.iter() {
                if let (FaceOwner::Part(first), FaceOwner::Part(second)) =
                    (weld.first.owner, weld.second.owner)
                {
                    if first == part {
                        visit(second);
                    } else if second == part {
                        visit(first);
                    }
                }
            }
            for (_, link) in self.rigid_links.iter() {
                if link.first == part {
                    visit(link.second);
                } else if link.second == part {
                    visit(link.first);
                }
            }
        }
        reached
    }

    /// Every part joined to `seed` through actual face welds. Shape features
    /// must not propagate through non-geometric rigid links.
    fn weld_group(&self, seed: PartId) -> BTreeSet<PartId> {
        let mut reached = BTreeSet::from([seed]);
        let mut frontier = vec![seed];
        while let Some(part) = frontier.pop() {
            for (_, weld) in self.welds.iter() {
                let (FaceOwner::Part(first), FaceOwner::Part(second)) =
                    (weld.first.owner, weld.second.owner)
                else {
                    continue;
                };
                let other = if first == part {
                    Some(second)
                } else if second == part {
                    Some(first)
                } else {
                    None
                };
                if let Some(other) = other
                    && reached.insert(other)
                {
                    frontier.push(other);
                }
            }
        }
        reached
    }

    fn insert_edit_part(&mut self, spec: PartSpec) -> PartId {
        let id = self.parts.insert(spec);
        let frame = self.edit_frame_id();
        self.construction_frames.members.insert(id, frame);
        id
    }

    #[allow(clippy::too_many_lines)] // One exhaustive command dispatch reads better whole.
    fn apply_validated(&mut self, command: BuildCommand) -> Result<BuildOutcome, GraphError> {
        match command {
            BuildCommand::Spawn(spec) => {
                let id = self.insert_edit_part(spec.into());
                Ok(BuildOutcome::Spawned(id))
            }
            BuildCommand::SpawnCylinder(spec) => {
                spec.validate_bands().map_err(GraphError::CylinderBands)?;
                let id = self.insert_edit_part(spec.into());
                Ok(BuildOutcome::Spawned(id))
            }
            BuildCommand::SpawnPipeBend(spec) => {
                let id = self.insert_edit_part(spec.into());
                Ok(BuildOutcome::Spawned(id))
            }
            BuildCommand::SpawnPipeJunction(spec) => {
                let id = self.insert_edit_part(spec.into());
                Ok(BuildOutcome::Spawned(id))
            }
            BuildCommand::SpawnController(spec) => {
                let id = self.insert_edit_part(spec.into());
                Ok(BuildOutcome::Spawned(id))
            }
            BuildCommand::SpawnEngine(spec) => {
                let id = self.insert_edit_part(spec.into());
                Ok(BuildOutcome::Spawned(id))
            }
            BuildCommand::AttachTransmission { parent, spec } => {
                let expected = self.next_transmission_spec(parent)?;
                if spec != expected {
                    return Err(GraphError::InvalidTransmissionPose);
                }
                let id = self.insert_edit_part(spec.into());
                let weld = WeldSpec {
                    first: FaceRef::part(parent, FaceKind::PositiveZ),
                    second: FaceRef::part(id, FaceKind::NegativeZ),
                };
                let frame = self
                    .part_frame_id(parent)
                    .expect("transmission parent exists");
                self.construction_frames.members.insert(id, frame);
                self.set_edit_source(id, parent)
                    .expect("a new transmission cannot be its own ancestor");
                self.validate_weld(weld)?;
                let weld_id = self.welds.insert(weld);
                self.transmission_parents.insert(id, parent);
                self.transmission_welds.insert(id, weld_id);
                self.pending = None;
                Ok(BuildOutcome::Spawned(id))
            }
            BuildCommand::SpawnServo(spec) => {
                let id = self.insert_edit_part(spec.into());
                Ok(BuildOutcome::Spawned(id))
            }
            BuildCommand::SpawnSeat(spec) => {
                let id = self.insert_edit_part(spec.into());
                Ok(BuildOutcome::Spawned(id))
            }
            BuildCommand::SpawnInput(spec) => {
                let id = self.insert_edit_part(spec.into());
                Ok(BuildOutcome::Spawned(id))
            }
            BuildCommand::SpawnDimensionLink(spec) => {
                if self.dimension_link(spec.id).is_some() {
                    return Err(GraphError::DuplicateDimensionLink(spec.id));
                }
                let id = self.insert_edit_part(spec.into());
                Ok(BuildOutcome::Spawned(id))
            }
            BuildCommand::SetAppearance { target, appearance } => {
                let target = match target {
                    AppearanceTarget::Part(part) => self
                        .region_of(part)
                        .map_or(AppearanceTarget::Part(part), AppearanceTarget::Region),
                    AppearanceTarget::Region(region) => AppearanceTarget::Region(region),
                    band @ AppearanceTarget::CylinderBand { .. } => band,
                };
                match target {
                    AppearanceTarget::CylinderBand { part, band } => {
                        let cylinder = self.cylinder(part)?;
                        if usize::from(band) >= cylinder.band_count() {
                            return Err(GraphError::CylinderBands(
                                crate::CylinderDimensionError::BandOutOfRange,
                            ));
                        }
                        *self
                            .parts
                            .get_mut(part)
                            .expect("the validated part remains live") = PartSpec::Cylinder(
                            cylinder.with_band_appearance(usize::from(band), appearance),
                        );
                    }
                    AppearanceTarget::Part(id) => {
                        let spec = self
                            .parts
                            .get(id)
                            .copied()
                            .ok_or(GraphError::MissingPart(id))?;
                        let replacement = spec
                            .with_appearance(appearance)
                            .ok_or(GraphError::AuthoredAppearance(id))?;
                        *self
                            .parts
                            .get_mut(id)
                            .expect("the validated part remains live") = replacement;
                    }
                    AppearanceTarget::Region(id) => {
                        self.regions.get(id).ok_or(GraphError::MissingRegion(id))?;
                        let members = self
                            .parts()
                            .filter_map(|(part, _)| {
                                (self.region_of(part) == Some(id)).then_some(part)
                            })
                            .collect::<Vec<_>>();
                        for part in members {
                            let spec = self
                                .parts
                                .get(part)
                                .copied()
                                .ok_or(GraphError::MissingPart(part))?;
                            let replacement = spec
                                .with_appearance(appearance)
                                .ok_or(GraphError::AuthoredAppearance(part))?;
                            *self
                                .parts
                                .get_mut(part)
                                .expect("a region member remains live") = replacement;
                        }
                        self.regions
                            .get_mut(id)
                            .expect("the validated region remains live")
                            .set_appearance(appearance);
                    }
                }
                Ok(BuildOutcome::AppearanceUpdated)
            }
            BuildCommand::Remove(id) => {
                self.parts.get(id).ok_or(GraphError::MissingPart(id))?;
                let mut removed_parts = BTreeSet::from([id]);
                let mut frontier = vec![id];
                while let Some(parent) = frontier.pop() {
                    for (&child, _) in self
                        .transmission_parents
                        .iter()
                        .filter(|(_, candidate)| **candidate == parent)
                    {
                        if removed_parts.insert(child) {
                            frontier.push(child);
                        }
                    }
                }
                let welds = self
                    .welds
                    .iter()
                    .filter_map(|(weld_id, weld)| {
                        removed_parts
                            .iter()
                            .any(|part| weld_references(*weld, *part))
                            .then_some(weld_id)
                    })
                    .collect::<Vec<_>>();
                let bearings = self
                    .bearings
                    .iter()
                    .filter_map(|(bearing_id, bearing)| {
                        removed_parts
                            .iter()
                            .any(|part| bearing_references(*bearing, *part))
                            .then_some(bearing_id)
                    })
                    .collect::<Vec<_>>();
                let rigid_links = self
                    .rigid_links
                    .iter()
                    .filter_map(|(link_id, link)| {
                        removed_parts
                            .iter()
                            .any(|part| rigid_link_references(*link, *part))
                            .then_some(link_id)
                    })
                    .collect::<Vec<_>>();
                let removed_bearings = bearings.iter().copied().collect::<BTreeSet<_>>();
                let drive_links = self
                    .drive_links
                    .iter()
                    .filter_map(|(link_id, link)| {
                        (removed_parts.contains(&link.controller)
                            || removed_bearings.contains(&link.bearing))
                        .then_some(link_id)
                    })
                    .collect::<Vec<_>>();
                let input_seat_links = self
                    .input_seat_links
                    .iter()
                    .filter_map(|(link_id, link)| {
                        (removed_parts.contains(&link.input) || removed_parts.contains(&link.seat))
                            .then_some(link_id)
                    })
                    .collect::<Vec<_>>();
                let seat_controller_links = self
                    .seat_controller_links
                    .iter()
                    .filter_map(|(link_id, link)| {
                        (removed_parts.contains(&link.seat)
                            || removed_parts.contains(&link.controller))
                        .then_some(link_id)
                    })
                    .collect::<Vec<_>>();
                for weld in welds {
                    self.welds.remove(weld);
                }
                for link in rigid_links {
                    self.rigid_links.remove(link);
                }
                for link in drive_links {
                    self.drive_links.remove(link);
                }
                for link in input_seat_links {
                    self.input_seat_links.remove(link);
                }
                for link in seat_controller_links {
                    self.seat_controller_links.remove(link);
                }
                for bearing in bearings {
                    self.bearings.remove(bearing);
                }
                // A region needs every cell filled, so losing one of its blocks
                // ends it — the same cascade welds already get.
                let regions = removed_parts
                    .iter()
                    .filter_map(|part| self.region_of(*part))
                    .collect::<BTreeSet<_>>();
                for region in regions {
                    self.regions.remove(region);
                    self.construction_frames.region_frames.remove(&region);
                    self.remove_shape_owner(SolidOwner::Region(region));
                }
                self.gearbox_configs
                    .retain(|(controller, _), _| !removed_parts.contains(controller));
                self.transmission_parents.retain(|child, parent| {
                    !removed_parts.contains(child) && !removed_parts.contains(parent)
                });
                self.transmission_welds
                    .retain(|part, _| !removed_parts.contains(part));
                for part in removed_parts {
                    self.remove_shape_owner(SolidOwner::Part(part));
                    self.parts.remove(part);
                    self.construction_frames.members.remove(&part);
                }
                self.pending = None;
                Ok(BuildOutcome::Removed)
            }
            BuildCommand::RemoveWeld(id) => {
                if self.transmission_welds.values().any(|weld| *weld == id) {
                    return Err(GraphError::RequiredTransmissionWeld(id));
                }
                self.welds.remove(id).ok_or(GraphError::MissingWeld(id))?;
                self.pending = None;
                Ok(BuildOutcome::Removed)
            }
            BuildCommand::RemoveRigidLink(id) => {
                self.rigid_links
                    .remove(id)
                    .ok_or(GraphError::MissingRigidLink(id))?;
                self.pending = None;
                Ok(BuildOutcome::Removed)
            }
            BuildCommand::RemoveBearing(id) => {
                self.bearings
                    .remove(id)
                    .ok_or(GraphError::MissingBearing(id))?;
                let drive_links = self
                    .drive_links
                    .iter()
                    .filter_map(|(link_id, link)| (link.bearing == id).then_some(link_id))
                    .collect::<Vec<_>>();
                for link in drive_links {
                    self.drive_links.remove(link);
                }
                self.pending = None;
                Ok(BuildOutcome::Removed)
            }
            BuildCommand::Weld(spec) => {
                self.validate_weld(spec)?;
                let id = self.welds.insert(spec);
                self.pending = None;
                Ok(BuildOutcome::Welded(id))
            }
            BuildCommand::RigidLink(spec) => {
                self.validate_rigid_link(spec)?;
                let id = self.rigid_links.insert(spec);
                self.pending = None;
                Ok(BuildOutcome::RigidLinked(id))
            }
            BuildCommand::AddBearing(spec) => {
                self.validate_bearing(spec)?;
                let id = self.bearings.insert(spec);
                self.pending = None;
                Ok(BuildOutcome::BearingAdded(id))
            }
            BuildCommand::SetSuspension { bearing, spec } => {
                let current = *self
                    .bearing(bearing)
                    .ok_or(GraphError::MissingBearing(bearing))?;
                let crate::BearingKind::Suspension(old) = current.kind else {
                    return Err(GraphError::IncompatibleDrive);
                };
                old.validate_edit(spec, true)?;
                let ids = self
                    .bearings()
                    .filter(|(_, b)| {
                        matches!(b.kind, crate::BearingKind::Suspension(_))
                            && b.source == current.source
                            && b.shared_anchor == current.shared_anchor
                            && b.axis == current.axis
                    })
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>();
                let mut candidate = self.clone();
                for &id in &ids {
                    candidate
                        .bearings
                        .get_mut(id)
                        .expect("existing attachment")
                        .kind = crate::BearingKind::Suspension(spec);
                }
                for id in ids {
                    candidate
                        .validate_bearing(*candidate.bearing(id).expect("existing attachment"))?;
                }
                self.bearings = candidate.bearings.clone();
                self.pending = None;
                Ok(BuildOutcome::BearingAdded(bearing))
            }
            BuildCommand::SetLinearDriveLimits { link, limits } => {
                let current = self
                    .drive_link(link)
                    .ok_or(GraphError::MissingDriveLink(link))?;
                self.validate_drive_units(current.bearing, current.program, Some(limits))?;
                self.drive_links
                    .get_mut(link)
                    .expect("validated link")
                    .linear_limits = Some(limits);
                Ok(BuildOutcome::DriveUpdated)
            }
            BuildCommand::AddDriveLink(spec) => {
                self.validate_drive_link(&spec)?;
                let id = self.drive_links.insert(spec);
                self.pending = None;
                Ok(BuildOutcome::DriveLinked(id))
            }
            BuildCommand::RemoveDriveLink(id) => {
                self.drive_links
                    .remove(id)
                    .ok_or(GraphError::MissingDriveLink(id))?;
                self.pending = None;
                Ok(BuildOutcome::Removed)
            }
            BuildCommand::AddInputSeatLink(spec) => {
                self.validate_input_seat_link(spec)?;
                let id = self.input_seat_links.insert(spec);
                self.pending = None;
                Ok(BuildOutcome::InputSeatLinked(id))
            }
            BuildCommand::RemoveInputSeatLink(id) => {
                self.input_seat_links
                    .remove(id)
                    .ok_or(GraphError::MissingInputSeatLink(id))?;
                self.pending = None;
                Ok(BuildOutcome::Removed)
            }
            BuildCommand::AddSeatControllerLink(spec) => {
                self.validate_seat_controller_link(spec)?;
                let id = self.seat_controller_links.insert(spec);
                self.pending = None;
                Ok(BuildOutcome::SeatControllerLinked(id))
            }
            BuildCommand::RemoveSeatControllerLink(id) => {
                self.seat_controller_links
                    .remove(id)
                    .ok_or(GraphError::MissingSeatControllerLink(id))?;
                self.pending = None;
                Ok(BuildOutcome::Removed)
            }
            BuildCommand::SetDriveLink {
                link,
                limits,
                program,
                name,
                actuator,
            } => {
                let current = self
                    .drive_link(link)
                    .ok_or(GraphError::MissingDriveLink(link))?;
                self.validate_drive_units(current.bearing, program, current.linear_limits)?;
                let spec = self
                    .drive_links
                    .get_mut(link)
                    .ok_or(GraphError::MissingDriveLink(link))?;
                spec.limits = limits;
                spec.program = program;
                spec.name = name;
                spec.actuator = actuator;
                Ok(BuildOutcome::DriveUpdated)
            }
            BuildCommand::SetGearboxMode {
                controller,
                kind,
                mode,
            } => {
                let mut config = self.editable_gearbox(controller, kind)?;
                config.set_mode(mode);
                self.gearbox_configs.insert((controller, kind), config);
                Ok(BuildOutcome::GearboxUpdated)
            }
            BuildCommand::SetGearboxRatios {
                controller,
                kind,
                ratios,
            } => {
                let mut config = self.editable_gearbox(controller, kind)?;
                if ratios.len() != config.ratios().len() {
                    return Err(GraphError::GearCountMismatch {
                        expected: config.ratios().len(),
                        actual: ratios.len(),
                    });
                }
                config.set_ratios(ratios)?;
                self.gearbox_configs.insert((controller, kind), config);
                Ok(BuildOutcome::GearboxUpdated)
            }
            BuildCommand::SetGearboxBindings {
                controller,
                kind,
                up,
                down,
            } => {
                let mut config = self.editable_gearbox(controller, kind)?;
                config.set_bindings(up, down);
                self.gearbox_configs.insert((controller, kind), config);
                Ok(BuildOutcome::GearboxUpdated)
            }
            BuildCommand::SetGasDivider {
                controller,
                reverse_gears,
            } => {
                let kind = EngineKind::Gas;
                let mut config = self.editable_gearbox(controller, kind)?;
                config.set_reverse_gears(reverse_gears)?;
                self.gearbox_configs.insert((controller, kind), config);
                Ok(BuildOutcome::GearboxUpdated)
            }
            BuildCommand::AddRegion(region) => {
                let frame = self.edit_frame_id();
                self.validate_region_area(&region, frame)?;
                let id = self.regions.insert(region);
                self.construction_frames.region_frames.insert(id, frame);
                Ok(BuildOutcome::RegionAdded(id))
            }
            BuildCommand::RemoveRegion(id) => {
                self.regions
                    .remove(id)
                    .ok_or(GraphError::MissingRegion(id))?;
                self.construction_frames.region_frames.remove(&id);
                self.remove_shape_owner(SolidOwner::Region(id));
                Ok(BuildOutcome::Removed)
            }
            BuildCommand::SetRegionVertices { region, vertices } => {
                let shape = self
                    .regions
                    .get_mut(region)
                    .ok_or(GraphError::MissingRegion(region))?;
                for (index, offset) in vertices {
                    shape.set_offset(index, offset)?;
                }
                // Two vertices driven through each other would turn a cell
                // inside out. The caller applied this to a staged clone, so
                // rejecting here leaves the live graph untouched.
                self.reject_inverted_region(region)?;
                self.validate_shape_owner_replay(SolidOwner::Region(region))?;
                self.validate_shape_owner_connections(SolidOwner::Region(region))?;
                Ok(BuildOutcome::RegionUpdated)
            }
            BuildCommand::SubdivideRegion {
                region,
                axis,
                position,
            } => {
                let shape = self
                    .regions
                    .get_mut(region)
                    .ok_or(GraphError::MissingRegion(region))?;
                shape.subdivide(axis, position)?;
                self.reject_inverted_region(region)?;
                self.validate_shape_owner_replay(SolidOwner::Region(region))?;
                self.validate_shape_owner_connections(SolidOwner::Region(region))?;
                Ok(BuildOutcome::RegionUpdated)
            }
            BuildCommand::AddShapeFeature(mut feature) => {
                if let Some((region, edges, frame)) = self.promoted_region_for_feature(&feature) {
                    let region = self.regions.insert(region);
                    self.construction_frames.region_frames.insert(region, frame);
                    feature.targets = edges
                        .into_iter()
                        .map(|edge| crate::EdgeChainRef {
                            owner: SolidOwner::Region(region),
                            edge,
                        })
                        .collect();
                }
                self.validate_shape_feature_targets(&feature)?;
                let owners = feature
                    .targets
                    .iter()
                    .map(|target| target.owner)
                    .collect::<BTreeSet<_>>();
                let id = self.shape_features.insert(feature);
                self.shape_feature_order.push(id);
                for owner in owners {
                    self.validate_shape_owner_replay(owner)?;
                    self.validate_shape_owner_connections(owner)?;
                }
                self.pending = None;
                Ok(BuildOutcome::ShapeFeatureAdded(id))
            }
            BuildCommand::SetCylinder { part, spec } => {
                if self.cylinder(part)?.pose != spec.pose {
                    return Err(GraphError::CylinderMoved(part));
                }
                spec.validate_bands().map_err(GraphError::CylinderBands)?;
                *self
                    .parts
                    .get_mut(part)
                    .expect("the validated part remains live") = PartSpec::Cylinder(spec);
                let owner = SolidOwner::Part(part);
                self.validate_shape_owner_replay(owner)?;
                self.validate_shape_owner_connections(owner)?;
                Ok(BuildOutcome::CylinderUpdated)
            }
            BuildCommand::SetShapeFeatureAmount {
                feature,
                amount_ticks,
            } => {
                if amount_ticks == 0 {
                    return Err(SolidError::ZeroAmount.into());
                }
                let owners = self
                    .shape_features
                    .get(feature)
                    .ok_or(GraphError::MissingShapeFeature(feature))?
                    .targets
                    .iter()
                    .map(|target| target.owner)
                    .collect::<BTreeSet<_>>();
                self.shape_features
                    .get_mut(feature)
                    .expect("the validated feature remains live")
                    .amount_ticks = amount_ticks;
                for owner in owners {
                    self.validate_shape_owner_replay(owner)?;
                    self.validate_shape_owner_connections(owner)?;
                }
                Ok(BuildOutcome::ShapeFeatureUpdated)
            }
            BuildCommand::RemoveShapeFeature(feature) => {
                let owners = self
                    .shape_features
                    .get(feature)
                    .ok_or(GraphError::MissingShapeFeature(feature))?
                    .targets
                    .iter()
                    .map(|target| target.owner)
                    .collect::<BTreeSet<_>>();
                self.shape_features
                    .remove(feature)
                    .expect("the validated feature remains live");
                self.shape_feature_order
                    .retain(|candidate| *candidate != feature);
                for owner in owners {
                    self.validate_shape_owner_replay(owner)?;
                    self.validate_shape_owner_connections(owner)?;
                }
                Ok(BuildOutcome::ShapeFeatureUpdated)
            }
            BuildCommand::BeginPending(pending) => {
                match pending {
                    PendingOperation::Weld(face) => {
                        self.face_geometry(face)?;
                    }
                    PendingOperation::Bearing { source, anchor } => {
                        let geometry = self.face_geometry(source)?;
                        if !anchor.is_finite() || !point_on_face(anchor, &geometry) {
                            return Err(GraphError::BearingAnchorOutsideFaces);
                        }
                    }
                    PendingOperation::DriveLink(controller) => {
                        self.parts
                            .get(controller)
                            .copied()
                            .ok_or(GraphError::MissingPart(controller))?
                            .as_controller()
                            .ok_or(GraphError::NotAController(controller))?;
                    }
                }
                self.pending = Some(pending);
                Ok(BuildOutcome::Pending)
            }
            BuildCommand::CancelPending => {
                self.pending = None;
                Ok(BuildOutcome::Cancelled)
            }
        }
    }

    fn validate_weld(&self, spec: WeldSpec) -> Result<(), GraphError> {
        if spec.first == spec.second {
            return Err(GraphError::SameFace);
        }
        if self.regions.is_empty()
            && self.shape_features.is_empty()
            && self.face_frame(spec.first) == self.face_frame(spec.second)
            && let Some(result) = self.validate_simple_grid_weld(spec)
        {
            return result;
        }
        let first = self.face_geometry(spec.first)?;
        let second = self.face_geometry(spec.second)?;
        if spec.first.patch.is_some() || spec.second.patch.is_some() {
            if first.normal.dot(second.normal) > -1.0 + axis_cosine_tolerance()
                || (first.center - second.center).dot(second.normal).abs() > ANCHOR_TOLERANCE_METERS
            {
                return Err(GraphError::FacesDoNotTouch);
            }
            if matches!(first.profile, FaceProfile::Ground)
                || matches!(second.profile, FaceProfile::Ground)
            {
                return Ok(());
            }
            let a = self.weld_material_cells(
                spec.first,
                &second,
                crate::ConstructionFrame::IDENTITY,
                -second.normal,
            );
            let b = self.weld_material_cells(
                spec.second,
                &second,
                crate::ConstructionFrame::IDENTITY,
                second.normal,
            );
            return if a
                .iter()
                .any(|a| b.iter().any(|b| convex_polygons_overlap(a, b)))
            {
                Ok(())
            } else {
                Err(GraphError::FacesDoNotTouch)
            };
        }
        if faces_touch(&first, &second) {
            Ok(())
        } else {
            Err(GraphError::FacesDoNotTouch)
        }
    }

    fn validate_simple_grid_weld(&self, spec: WeldSpec) -> Option<Result<(), GraphError>> {
        let simple_cuboid = |face: FaceRef| -> Result<Option<CuboidSpec>, GraphError> {
            let FaceOwner::Part(part) = face.owner else {
                return Ok(None);
            };
            let part = self
                .parts
                .get(part)
                .copied()
                .ok_or(GraphError::MissingPart(part))?;
            let PartSpec::Cuboid(cuboid) = part else {
                return Ok(None);
            };
            Ok(
                (face.patch.is_none() && cuboid.pose.rotation == crate::GridRotation::default())
                    .then_some(cuboid),
            )
        };

        match (spec.first.owner, spec.second.owner) {
            (FaceOwner::Part(_), FaceOwner::Part(_)) => {
                let first = match simple_cuboid(spec.first) {
                    Ok(Some(cuboid)) => cuboid,
                    Ok(None) => return None,
                    Err(error) => return Some(Err(error)),
                };
                let second = match simple_cuboid(spec.second) {
                    Ok(Some(cuboid)) => cuboid,
                    Ok(None) => return None,
                    Err(error) => return Some(Err(error)),
                };
                Some(
                    if simple_grid_faces_touch(first, spec.first.face, second, spec.second.face) {
                        Ok(())
                    } else {
                        Err(GraphError::FacesDoNotTouch)
                    },
                )
            }
            (FaceOwner::Part(_), FaceOwner::Ground) => {
                if spec.second != FaceRef::ground() {
                    return None;
                }
                let cuboid = match simple_cuboid(spec.first) {
                    Ok(Some(cuboid)) => cuboid,
                    Ok(None) => return None,
                    Err(error) => return Some(Err(error)),
                };
                Some(if simple_grid_face_on_ground(cuboid, spec.first.face) {
                    Ok(())
                } else {
                    Err(GraphError::FacesDoNotTouch)
                })
            }
            (FaceOwner::Ground, FaceOwner::Part(_)) => {
                if spec.first != FaceRef::ground() {
                    return None;
                }
                let cuboid = match simple_cuboid(spec.second) {
                    Ok(Some(cuboid)) => cuboid,
                    Ok(None) => return None,
                    Err(error) => return Some(Err(error)),
                };
                Some(if simple_grid_face_on_ground(cuboid, spec.second.face) {
                    Ok(())
                } else {
                    Err(GraphError::FacesDoNotTouch)
                })
            }
            (FaceOwner::Ground, FaceOwner::Ground) => None,
        }
    }

    fn validate_rigid_link(&self, spec: RigidLinkSpec) -> Result<(), GraphError> {
        self.parts
            .get(spec.first)
            .ok_or(GraphError::MissingPart(spec.first))?;
        self.parts
            .get(spec.second)
            .ok_or(GraphError::MissingPart(spec.second))?;
        if spec.first == spec.second {
            return Err(GraphError::SameRigidLinkPart);
        }
        Ok(())
    }

    fn validate_drive_units(
        &self,
        bearing: BearingId,
        program: DriveProgram,
        limits: Option<crate::LinearDriveLimits>,
    ) -> Result<(), GraphError> {
        let bearing = self
            .bearing(bearing)
            .ok_or(GraphError::MissingBearing(bearing))?;
        if matches!(bearing.kind, crate::BearingKind::Suspension(_)) {
            return Err(GraphError::IncompatibleDrive);
        }
        let linear = matches!(bearing.kind, crate::BearingKind::Linear(_));
        if program
            .states()
            .iter()
            .any(|state| state.target().is_linear() != linear)
            || limits.is_some() != linear
        {
            return Err(GraphError::IncompatibleDrive);
        }
        if let Some(limits) = limits {
            let [minimum, maximum] = bearing.kind.bounds();
            if limits.minimum() < minimum || limits.maximum() > maximum {
                return Err(GraphError::IncompatibleDrive);
            }
        }
        Ok(())
    }

    fn validate_drive_link(&self, spec: &DriveLinkSpec) -> Result<(), GraphError> {
        self.parts
            .get(spec.controller)
            .copied()
            .ok_or(GraphError::MissingPart(spec.controller))?
            .as_controller()
            .ok_or(GraphError::NotAController(spec.controller))?;
        self.bearings
            .get(spec.bearing)
            .ok_or(GraphError::MissingBearing(spec.bearing))?;
        self.validate_drive_units(spec.bearing, spec.program, spec.linear_limits)?;
        if self
            .drive_links
            .iter()
            .any(|(_, link)| link.bearing == spec.bearing)
        {
            return Err(GraphError::BearingAlreadyDriven(spec.bearing));
        }
        Ok(())
    }

    fn validate_input_seat_link(&self, spec: InputSeatLinkSpec) -> Result<(), GraphError> {
        match self.parts.get(spec.input) {
            Some(PartSpec::Input(_)) => {}
            Some(_) => return Err(GraphError::NotAnInput(spec.input)),
            None => return Err(GraphError::MissingPart(spec.input)),
        }
        match self.parts.get(spec.seat) {
            Some(PartSpec::Seat(_)) => {}
            Some(_) => return Err(GraphError::NotASeat(spec.seat)),
            None => return Err(GraphError::MissingPart(spec.seat)),
        }
        if self
            .input_seat_links
            .iter()
            .any(|(_, link)| link.input == spec.input)
        {
            return Err(GraphError::InputAlreadyLinked(spec.input));
        }
        if self
            .input_seat_links
            .iter()
            .any(|(_, link)| link.seat == spec.seat)
        {
            return Err(GraphError::SeatAlreadyHasInput(spec.seat));
        }
        Ok(())
    }

    fn validate_seat_controller_link(
        &self,
        spec: SeatControllerLinkSpec,
    ) -> Result<(), GraphError> {
        match self.parts.get(spec.seat) {
            Some(PartSpec::Seat(_)) => {}
            Some(_) => return Err(GraphError::NotASeat(spec.seat)),
            None => return Err(GraphError::MissingPart(spec.seat)),
        }
        self.parts
            .get(spec.controller)
            .copied()
            .ok_or(GraphError::MissingPart(spec.controller))?
            .as_controller()
            .ok_or(GraphError::NotAController(spec.controller))?;
        if self
            .seat_controller_links
            .iter()
            .any(|(_, link)| link.seat == spec.seat)
        {
            return Err(GraphError::SeatAlreadyHasController(spec.seat));
        }
        Ok(())
    }

    fn validate_shape_feature_targets(&self, feature: &ShapeFeature) -> Result<(), GraphError> {
        if feature.amount_ticks == 0 {
            return Err(SolidError::ZeroAmount.into());
        }
        if feature.targets.is_empty() {
            return Err(SolidError::ZeroVolume.into());
        }
        for target in &feature.targets {
            match target.owner {
                SolidOwner::Part(part) => match self.parts.get(part) {
                    Some(
                        PartSpec::Cuboid(_)
                        | PartSpec::Cylinder(_)
                        | PartSpec::PipeBend(_)
                        | PartSpec::PipeJunction(_),
                    ) => {}
                    Some(_) => return Err(GraphError::InvalidShapeFeatureOwner(target.owner)),
                    None => return Err(GraphError::MissingPart(part)),
                },
                SolidOwner::Region(region) => {
                    if self.regions.get(region).is_none() {
                        return Err(GraphError::MissingRegion(region));
                    }
                }
            }
        }
        Ok(())
    }

    /// Fuses a rectangular welded block assembly before its first cross-part
    /// edge feature. The region then owns one continuous volume, so a fillet or
    /// chamfer can pass an internal 25 cm block boundary and trim the blocks
    /// behind the originally selected edge.
    fn promoted_region_for_feature(
        &self,
        feature: &ShapeFeature,
    ) -> Option<(
        ShapeRegion,
        Vec<crate::TopologyKey>,
        crate::ConstructionFrameId,
    )> {
        let target_parts = feature
            .targets
            .iter()
            .map(|target| match target.owner {
                SolidOwner::Part(part) => Some(part),
                SolidOwner::Region(_) => None,
            })
            .collect::<Option<BTreeSet<_>>>()?;
        let seed = *target_parts.first()?;
        let frame = self.part_frame_id(seed)?;
        if target_parts
            .iter()
            .any(|&part| self.part_frame_id(part) != Some(frame))
        {
            return None;
        }
        let seed_spec = self.parts.get(seed)?.as_cuboid()?;
        let material = seed_spec.material;
        let appearance = seed_spec.appearance;
        let welded = self.weld_group(seed);
        if !target_parts.iter().all(|part| welded.contains(part)) {
            return None;
        }

        let mut minimum = IVec3::splat(i32::MAX);
        let mut maximum = IVec3::splat(i32::MIN);
        let mut members = 0_usize;
        for part in welded {
            if self.part_frame_id(part) != Some(frame) || self.region_of(part).is_some() {
                continue;
            }
            let Some(PartSpec::Cuboid(cuboid)) = self.parts.get(part).copied() else {
                continue;
            };
            if cuboid.material != material || cuboid.appearance != appearance {
                continue;
            }
            let cells = crate::part_cells(cuboid);
            let origin = cells.corner_steps(IVec3::ZERO, 0);
            minimum = minimum.min(origin);
            maximum = maximum.max(origin + cells.counts() * crate::shape::STEPS_PER_CELL);
            members += 1;
        }
        if members < 2
            || members < target_parts.len()
            || minimum.cmpeq(IVec3::splat(i32::MAX)).all()
        {
            return None;
        }
        let extent = maximum - minimum;
        if extent.rem_euclid(IVec3::splat(crate::shape::STEPS_PER_CELL)) != IVec3::ZERO {
            return None;
        }
        let region = ShapeRegion::from_origin_steps(
            minimum,
            extent / crate::shape::STEPS_PER_CELL,
            material,
        )
        .ok()?
        .with_appearance(appearance);
        self.validate_region_area(&region, frame).ok()?;

        let solid = crate::evaluate_region_solid(&region, []).ok()?;
        let edges = feature
            .targets
            .iter()
            .map(|target| target.edge)
            .collect::<BTreeSet<_>>();
        if !edges.iter().all(|edge| {
            solid
                .logical_edge(*edge)
                .is_some_and(|logical| logical.convex)
        }) {
            return None;
        }
        Some((region, edges.into_iter().collect(), frame))
    }

    fn validate_shape_owner_replay(&self, owner: SolidOwner) -> Result<(), GraphError> {
        self.evaluated_solid_shared(owner).map(|_| ())
    }

    fn validate_shape_owner_connections(&self, owner: SolidOwner) -> Result<(), GraphError> {
        for (_, weld) in self.welds.iter() {
            if let SolidOwner::Region(region) = owner
                && let (FaceOwner::Part(first), FaceOwner::Part(second)) =
                    (weld.first.owner, weld.second.owner)
                && self.region_of(first) == Some(region)
                && self.region_of(second) == Some(region)
            {
                // These welds establish the rigid membership from which the
                // region was claimed. Their faces are internal to the fused
                // solid and are not placement patches that a feature must
                // preserve.
                continue;
            }
            if self.face_is_on_owner(weld.first, owner) || self.face_is_on_owner(weld.second, owner)
            {
                self.validate_weld(*weld)?;
            }
        }
        for (_, bearing) in self.bearings.iter() {
            if self.face_is_on_owner(bearing.source, owner)
                || self.face_is_on_owner(bearing.target, owner)
            {
                self.validate_bearing(*bearing)?;
            }
        }
        Ok(())
    }

    fn cylinder(&self, part: PartId) -> Result<CylinderSpec, GraphError> {
        match self.parts.get(part).copied() {
            Some(PartSpec::Cylinder(cylinder)) => Ok(cylinder),
            Some(_) => Err(GraphError::NotCylinder(part)),
            None => Err(GraphError::MissingPart(part)),
        }
    }

    fn face_is_on_owner(&self, face: FaceRef, owner: SolidOwner) -> bool {
        let FaceOwner::Part(part) = face.owner else {
            return false;
        };
        match owner {
            SolidOwner::Part(candidate) => part == candidate,
            SolidOwner::Region(region) => self.region_of(part) == Some(region),
        }
    }

    fn remove_shape_owner(&mut self, owner: SolidOwner) {
        let affected = self
            .shape_feature_order
            .iter()
            .copied()
            .filter(|id| {
                self.shape_features.get(*id).is_some_and(|feature| {
                    feature.targets.iter().any(|target| target.owner == owner)
                })
            })
            .collect::<Vec<_>>();
        for id in affected {
            let remove = if let Some(feature) = self.shape_features.get_mut(id) {
                feature.targets.retain(|target| target.owner != owner);
                feature.targets.is_empty()
            } else {
                false
            };
            if remove {
                self.shape_features.remove(id);
                self.shape_feature_order
                    .retain(|candidate| *candidate != id);
            }
        }
    }

    pub(crate) fn validate_suspension_socket(
        &self,
        socket: crate::BearingSocket,
    ) -> Result<(), GraphError> {
        let crate::BearingKind::Suspension(spec) = socket.kind else {
            return Ok(());
        };
        if matches!(socket.source.owner, FaceOwner::Ground) {
            return Err(GraphError::BearingOnGround);
        }
        let source = self.face_geometry(socket.source)?;
        if !socket.axis.is_finite()
            || (socket.axis.length_squared() - 1.0).abs() > 1.0e-5
            || socket.axis.dot(source.normal) < 1.0 - axis_cosine_tolerance()
        {
            return Err(GraphError::InvalidBearingAxis);
        }
        let mount = FaceGeometry {
            center: socket.anchor,
            normal: source.normal,
            tangent_u: source.tangent_u,
            tangent_v: source.tangent_v,
            profile: FaceProfile::Annulus {
                inner_radius: 0.0,
                outer_radius: spec.plates().diameter / 2.0,
            },
        };
        if !socket.anchor.is_finite()
            || (source.center - socket.anchor).dot(socket.axis).abs() > ANCHOR_TOLERANCE_METERS
            || !profiles_overlap(&mount, &source)
        {
            return Err(GraphError::BearingAnchorOutsideFaces);
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn validate_bearing(&self, spec: BearingSpec) -> Result<(), GraphError> {
        if spec.source == spec.target {
            return Err(GraphError::SameFace);
        }
        if matches!(spec.source.owner, FaceOwner::Ground)
            || matches!(spec.target.owner, FaceOwner::Ground)
        {
            return Err(GraphError::BearingOnGround);
        }
        let source = self.face_geometry(spec.source)?;
        let target = self.face_geometry(spec.target)?;
        if let crate::BearingKind::Suspension(suspension) = spec.kind {
            if self.bearings().any(|(_, existing)| {
                existing.source == spec.source
                    && existing.shared_anchor.distance(spec.shared_anchor) < ANCHOR_TOLERANCE_METERS
                    && existing.kind != spec.kind
            }) {
                return Err(GraphError::Suspension(crate::SuspensionError::SharedMounts));
            }

            if !spec.axis.is_finite()
                || (spec.axis.length_squared() - 1.0).abs() > 1.0e-5
                || spec.axis.dot(source.normal) < 1.0 - axis_cosine_tolerance()
            {
                return Err(GraphError::InvalidBearingAxis);
            }
            if source.normal.dot(target.normal) > -1.0 + axis_cosine_tolerance() {
                return Err(GraphError::BearingFacesNotOpposed);
            }
            let opposite = spec.shared_anchor + spec.axis * suspension.initial_length();
            let radius = suspension.plates().diameter / 2.0;
            let mount = FaceGeometry {
                center: spec.shared_anchor,
                normal: source.normal,
                tangent_u: source.tangent_u,
                tangent_v: source.tangent_v,
                profile: FaceProfile::Annulus {
                    inner_radius: 0.0,
                    outer_radius: radius,
                },
            };
            let other = FaceGeometry {
                center: opposite,
                ..mount.clone()
            };
            if !spec.shared_anchor.is_finite()
                || (source.center - spec.shared_anchor).dot(spec.axis).abs()
                    > ANCHOR_TOLERANCE_METERS
                || (target.center - opposite).dot(spec.axis).abs() > ANCHOR_TOLERANCE_METERS
                || !profiles_overlap(&mount, &source)
                || !profiles_overlap(&other, &target)
            {
                return Err(GraphError::BearingAnchorOutsideFaces);
            }
            return Ok(());
        }
        if let crate::BearingKind::Linear(rail) = spec.kind {
            if self.bearings().any(|(_, existing)| {
                existing.source == spec.source
                    && existing.shared_anchor.distance(spec.shared_anchor) < ANCHOR_TOLERANCE_METERS
                    && matches!(existing.kind, crate::BearingKind::Linear(other) if other != rail)
            }) {
                return Err(GraphError::LinearCarriageOccupied);
            }
            let rotation = rail.rotation(spec.axis)?;
            if source.normal.dot(rail.mount_normal) < 1.0 - 1.0e-5
                || target.normal.dot(rotation * rail.face.normal()) > -1.0 + 1.0e-5
            {
                return Err(GraphError::BearingFacesNotOpposed);
            }
            let mount = FaceGeometry {
                center: spec.shared_anchor,
                normal: rail.mount_normal,
                tangent_u: spec.axis,
                tangent_v: spec.axis.cross(rail.mount_normal),
                profile: FaceProfile::Rectangle {
                    half_u: rail.dimensions.length() / 2.0,
                    half_v: rail.dimensions.width() / 2.0,
                },
            };
            let size = rail.face.size(rail.dimensions);
            let surface = FaceGeometry {
                center: spec.shared_anchor + rotation * rail.face.origin(rail.dimensions),
                normal: rotation * rail.face.normal(),
                tangent_u: spec.axis,
                tangent_v: spec.axis.cross(rotation * rail.face.normal()),
                profile: FaceProfile::Rectangle {
                    half_u: size.x / 2.0,
                    half_v: size.y / 2.0,
                },
            };
            if !spec.shared_anchor.is_finite()
                || (source.center - mount.center).dot(source.normal).abs() > ANCHOR_TOLERANCE_METERS
                || (target.center - surface.center).dot(target.normal).abs()
                    > ANCHOR_TOLERANCE_METERS
                || !profiles_overlap(&mount, &source)
                || !profiles_overlap(&surface, &target)
            {
                return Err(GraphError::BearingAnchorOutsideFaces);
            }
            return Ok(());
        }
        if source.normal.dot(target.normal) > -1.0 + axis_cosine_tolerance() {
            return Err(GraphError::BearingFacesNotOpposed);
        }
        if !spec.shared_anchor.is_finite()
            || !bearing_ring_overlaps_face(spec.shared_anchor, spec.dimensions, &source)
            || !bearing_ring_overlaps_face(spec.shared_anchor, spec.dimensions, &target)
        {
            return Err(GraphError::BearingAnchorOutsideFaces);
        }
        let length = spec.axis.length();
        if !spec.axis.is_finite()
            || (length - 1.0).abs() > 1.0e-5
            || spec.axis.dot(source.normal) < 1.0 - axis_cosine_tolerance()
        {
            return Err(GraphError::InvalidBearingAxis);
        }
        Ok(())
    }
}

fn face_references(face: FaceRef, part: PartId) -> bool {
    face.owner == FaceOwner::Part(part)
}

const fn primitive_surface_patch(spec: PartSpec, face: FaceKind) -> crate::SurfacePatchKey {
    let local = match spec {
        PartSpec::Cylinder(_) => match face {
            FaceKind::NegativeY => 0,
            FaceKind::PositiveY => 1,
            _ => u32::MAX,
        },
        PartSpec::PipeBend(_) => match face {
            FaceKind::NegativeX => 0,
            FaceKind::PositiveY => 1,
            _ => u32::MAX,
        },
        _ => match face {
            FaceKind::NegativeX => 0,
            FaceKind::PositiveX => 1,
            FaceKind::NegativeY => 2,
            FaceKind::PositiveY => 3,
            FaceKind::NegativeZ => 4,
            FaceKind::PositiveZ => 5,
        },
    };
    crate::SurfacePatchKey {
        source: crate::TopologySource::Base,
        local,
    }
}

fn weld_references(weld: WeldSpec, part: PartId) -> bool {
    face_references(weld.first, part) || face_references(weld.second, part)
}

fn simple_grid_faces_touch(
    first: CuboidSpec,
    first_face: FaceKind,
    second: CuboidSpec,
    second_face: FaceKind,
) -> bool {
    let (axis, first_sign) = simple_face_axis(first_face);
    let (second_axis, second_sign) = simple_face_axis(second_face);
    if axis != second_axis || first_sign == second_sign {
        return false;
    }
    let first_center = first.pose.translation_position_ticks();
    let second_center = second.pose.translation_position_ticks();
    let first_half =
        i32::from(first.dimensions[axis].units()) * crate::POSITION_TICKS_PER_HALF_GRID_UNIT;
    let second_half =
        i32::from(second.dimensions[axis].units()) * crate::POSITION_TICKS_PER_HALF_GRID_UNIT;
    if first_center[axis] + first_sign * first_half
        != second_center[axis] + second_sign * second_half
    {
        return false;
    }
    (0..3).filter(|&tangent| tangent != axis).all(|tangent| {
        let distance = (first_center[tangent] - second_center[tangent]).abs();
        let combined_half = (i32::from(first.dimensions[tangent].units())
            + i32::from(second.dimensions[tangent].units()))
            * crate::POSITION_TICKS_PER_HALF_GRID_UNIT;
        distance < combined_half
    })
}

fn simple_grid_face_on_ground(cuboid: CuboidSpec, face: FaceKind) -> bool {
    face == FaceKind::NegativeY
        && cuboid.pose.translation_position_ticks().y
            == i32::from(cuboid.dimensions[1].units()) * crate::POSITION_TICKS_PER_HALF_GRID_UNIT
}

const fn simple_face_axis(face: FaceKind) -> (usize, i32) {
    match face {
        FaceKind::PositiveX => (0, 1),
        FaceKind::NegativeX => (0, -1),
        FaceKind::PositiveY => (1, 1),
        FaceKind::NegativeY => (1, -1),
        FaceKind::PositiveZ => (2, 1),
        FaceKind::NegativeZ => (2, -1),
    }
}

fn rigid_link_references(link: RigidLinkSpec, part: PartId) -> bool {
    link.first == part || link.second == part
}

fn bearing_references(bearing: BearingSpec, part: PartId) -> bool {
    face_references(bearing.source, part) || face_references(bearing.target, part)
}

fn point_on_face(point: Vec3, face: &FaceGeometry) -> bool {
    let offset = point - face.center;
    if offset.dot(face.normal).abs() > ANCHOR_TOLERANCE_METERS {
        return false;
    }
    point_in_profile(
        offset.dot(face.tangent_u),
        offset.dot(face.tangent_v),
        &face.profile,
    )
}

fn bearing_ring_overlaps_face(
    anchor: Vec3,
    dimensions: BearingDimensions,
    face: &FaceGeometry,
) -> bool {
    let offset = anchor - face.center;
    if offset.dot(face.normal).abs() > ANCHOR_TOLERANCE_METERS
        || matches!(face.profile, FaceProfile::Ground)
    {
        return false;
    }
    let ring = FaceGeometry {
        center: anchor,
        normal: face.normal,
        tangent_u: face.tangent_u,
        tangent_v: face.tangent_v,
        profile: FaceProfile::Annulus {
            inner_radius: dimensions.inner_diameter() * 0.5,
            outer_radius: dimensions.outer_diameter() * 0.5,
        },
    };
    profiles_overlap(&ring, face)
}

fn faces_touch(first: &FaceGeometry, second: &FaceGeometry) -> bool {
    if first.normal.dot(second.normal) > -1.0 + axis_cosine_tolerance() {
        return false;
    }
    let separation = (second.center - first.center).dot(first.normal).abs();
    if separation > ANCHOR_TOLERANCE_METERS {
        return false;
    }
    if matches!(&first.profile, FaceProfile::Ground)
        || matches!(&second.profile, FaceProfile::Ground)
    {
        return true;
    }
    profiles_overlap(first, second)
}

fn point_in_profile(u: f32, v: f32, profile: &FaceProfile) -> bool {
    match profile {
        FaceProfile::Rectangle { half_u, half_v } => {
            u.abs() <= half_u + ANCHOR_TOLERANCE_METERS
                && v.abs() <= half_v + ANCHOR_TOLERANCE_METERS
        }
        FaceProfile::Annulus {
            inner_radius,
            outer_radius,
        } => {
            let radius_squared = u.mul_add(u, v * v);
            radius_squared >= (inner_radius - ANCHOR_TOLERANCE_METERS).max(0.0).powi(2)
                && radius_squared <= (outer_radius + ANCHOR_TOLERANCE_METERS).powi(2)
        }
        FaceProfile::AnnularSector {
            inner_radius,
            outer_radius,
            half_angle,
        } => {
            let radius_squared = u.mul_add(u, v * v);
            radius_squared >= (inner_radius - ANCHOR_TOLERANCE_METERS).max(0.0).powi(2)
                && radius_squared <= (outer_radius + ANCHOR_TOLERANCE_METERS).powi(2)
                && v.atan2(u).abs() <= half_angle + ANCHOR_TOLERANCE_METERS
        }
        FaceProfile::Polygon { vertices } => point_in_convex_polygon(Vec2::new(u, v), vertices),
        FaceProfile::Ground => true,
    }
}

fn profiles_overlap(first: &FaceGeometry, second: &FaceGeometry) -> bool {
    match (&first.profile, &second.profile) {
        (FaceProfile::Rectangle { half_u, half_v }, FaceProfile::Rectangle { .. }) => {
            positive_rect_overlap(first, second, first.tangent_u, *half_u)
                && positive_rect_overlap(first, second, first.tangent_v, *half_v)
        }
        (FaceProfile::Annulus { .. }, FaceProfile::Rectangle { .. }) => {
            annulus_rectangle_overlap(first, second)
        }
        (FaceProfile::Rectangle { .. }, FaceProfile::Annulus { .. }) => {
            annulus_rectangle_overlap(second, first)
        }
        (FaceProfile::Annulus { .. }, FaceProfile::Annulus { .. }) => annuli_overlap(first, second),
        (FaceProfile::Ground, _) | (_, FaceProfile::Ground) => true,
        (FaceProfile::AnnularSector { .. } | FaceProfile::Polygon { .. }, _)
        | (_, FaceProfile::AnnularSector { .. } | FaceProfile::Polygon { .. }) => {
            sector_profiles_overlap(first, second)
        }
    }
}

fn sector_profiles_overlap(first: &FaceGeometry, second: &FaceGeometry) -> bool {
    let first_cells = profile_cells(first, first.center, first.tangent_u, first.tangent_v);
    let second_cells = profile_cells(second, first.center, first.tangent_u, first.tangent_v);
    first_cells.iter().any(|first| {
        second_cells
            .iter()
            .any(|second| convex_polygons_overlap(first, second))
    })
}

fn profile_cells(
    face: &FaceGeometry,
    origin: Vec3,
    plane_u: Vec3,
    plane_v: Vec3,
) -> Vec<Vec<Vec2>> {
    let project = |point: Vec3| {
        let offset = point - origin;
        Vec2::new(offset.dot(plane_u), offset.dot(plane_v))
    };
    match &face.profile {
        FaceProfile::Rectangle { half_u, half_v } => vec![vec![
            project(face.center - face.tangent_u * *half_u - face.tangent_v * *half_v),
            project(face.center + face.tangent_u * *half_u - face.tangent_v * *half_v),
            project(face.center + face.tangent_u * *half_u + face.tangent_v * *half_v),
            project(face.center - face.tangent_u * *half_u + face.tangent_v * *half_v),
        ]],
        FaceProfile::Annulus {
            inner_radius,
            outer_radius,
        } => annular_profile_cells(
            face,
            origin,
            plane_u,
            plane_v,
            *inner_radius,
            *outer_radius,
            core::f32::consts::PI,
        ),
        FaceProfile::AnnularSector {
            inner_radius,
            outer_radius,
            half_angle,
        } => annular_profile_cells(
            face,
            origin,
            plane_u,
            plane_v,
            *inner_radius,
            *outer_radius,
            *half_angle,
        ),
        FaceProfile::Polygon { vertices } => vec![
            vertices
                .iter()
                .map(|vertex| {
                    project(face.center + face.tangent_u * vertex.x + face.tangent_v * vertex.y)
                })
                .collect(),
        ],
        FaceProfile::Ground => Vec::new(),
    }
}

fn point_in_convex_polygon(point: Vec2, vertices: &[Vec2]) -> bool {
    if vertices.len() < 3 {
        return false;
    }
    let mut sign = 0.0_f32;
    for index in 0..vertices.len() {
        let edge = vertices[(index + 1) % vertices.len()] - vertices[index];
        let offset = point - vertices[index];
        let cross = edge.perp_dot(offset);
        if cross.abs() <= ANCHOR_TOLERANCE_METERS {
            continue;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if sign * cross < 0.0 {
            return false;
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn annular_profile_cells(
    face: &FaceGeometry,
    origin: Vec3,
    plane_u: Vec3,
    plane_v: Vec3,
    inner_radius: f32,
    outer_radius: f32,
    half_angle: f32,
) -> Vec<Vec<Vec2>> {
    let sweep = half_angle * 2.0;
    let segment_count = (1_u16..=24)
        .find(|&count| (f32::from(count) * (core::f32::consts::PI / 12.0) - sweep).abs() < 1.0e-4)
        .expect("annular profiles use 15-degree increments");
    let project = |point: Vec3| {
        let offset = point - origin;
        Vec2::new(offset.dot(plane_u), offset.dot(plane_v))
    };
    (0..segment_count)
        .map(|segment| {
            let first_angle = -half_angle + sweep * f32::from(segment) / f32::from(segment_count);
            let second_angle =
                -half_angle + sweep * f32::from(segment + 1) / f32::from(segment_count);
            let radial = |angle: f32| face.tangent_u * angle.cos() + face.tangent_v * angle.sin();
            let outer_first = project(face.center + radial(first_angle) * outer_radius);
            let outer_second = project(face.center + radial(second_angle) * outer_radius);
            if inner_radius == 0.0 {
                vec![project(face.center), outer_first, outer_second]
            } else {
                vec![
                    project(face.center + radial(first_angle) * inner_radius),
                    outer_first,
                    outer_second,
                    project(face.center + radial(second_angle) * inner_radius),
                ]
            }
        })
        .collect()
}

fn convex_polygons_overlap(first: &[Vec2], second: &[Vec2]) -> bool {
    first
        .iter()
        .zip(first.iter().cycle().skip(1))
        .chain(second.iter().zip(second.iter().cycle().skip(1)))
        .all(|(start, end)| {
            let edge = *end - *start;
            let axis = Vec2::new(-edge.y, edge.x).normalize();
            let project = |polygon: &[Vec2]| {
                polygon.iter().fold(
                    (f32::INFINITY, f32::NEG_INFINITY),
                    |(minimum, maximum), point| {
                        let value = point.dot(axis);
                        (minimum.min(value), maximum.max(value))
                    },
                )
            };
            let (first_minimum, first_maximum) = project(first);
            let (second_minimum, second_maximum) = project(second);
            first_maximum.min(second_maximum) - first_minimum.max(second_minimum)
                > ANCHOR_TOLERANCE_METERS
        })
}

fn positive_rect_overlap(
    first: &FaceGeometry,
    second: &FaceGeometry,
    axis: Vec3,
    first_half: f32,
) -> bool {
    let FaceProfile::Rectangle { half_u, half_v } = &second.profile else {
        unreachable!()
    };
    let second_half =
        second.tangent_u.dot(axis).abs() * *half_u + second.tangent_v.dot(axis).abs() * *half_v;
    let centre_distance = (second.center - first.center).dot(axis).abs();
    first_half + second_half - centre_distance > ANCHOR_TOLERANCE_METERS
}

fn annulus_rectangle_overlap(annulus: &FaceGeometry, rectangle: &FaceGeometry) -> bool {
    let FaceProfile::Annulus {
        inner_radius,
        outer_radius,
    } = &annulus.profile
    else {
        unreachable!()
    };
    let FaceProfile::Rectangle { half_u, half_v } = &rectangle.profile else {
        unreachable!()
    };
    let offset = annulus.center - rectangle.center;
    let center_u = offset.dot(rectangle.tangent_u).abs();
    let center_v = offset.dot(rectangle.tangent_v).abs();
    let nearest_u = (center_u - *half_u).max(0.0);
    let nearest_v = (center_v - *half_v).max(0.0);
    let nearest_squared = nearest_u.mul_add(nearest_u, nearest_v * nearest_v);
    let farthest_u = center_u + *half_u;
    let farthest_v = center_v + *half_v;
    let farthest_squared = farthest_u.mul_add(farthest_u, farthest_v * farthest_v);
    nearest_squared < (*outer_radius - ANCHOR_TOLERANCE_METERS).max(0.0).powi(2)
        && farthest_squared > (*inner_radius + ANCHOR_TOLERANCE_METERS).powi(2)
}

fn annuli_overlap(first: &FaceGeometry, second: &FaceGeometry) -> bool {
    let FaceProfile::Annulus {
        inner_radius: inner_a,
        outer_radius: outer_a,
    } = &first.profile
    else {
        unreachable!()
    };
    let FaceProfile::Annulus {
        inner_radius: inner_b,
        outer_radius: outer_b,
    } = &second.profile
    else {
        unreachable!()
    };
    let offset = second.center - first.center;
    let distance = Vec3::new(
        offset.dot(first.tangent_u),
        offset.dot(first.tangent_v),
        0.0,
    )
    .length();
    distance < *outer_a + *outer_b - ANCHOR_TOLERANCE_METERS
        && distance + *outer_a > *inner_b + ANCHOR_TOLERANCE_METERS
        && distance + *outer_b > *inner_a + ANCHOR_TOLERANCE_METERS
}

fn axis_cosine_tolerance() -> f32 {
    1.0 - AXIS_TOLERANCE_DEGREES.to_radians().cos()
}

#[cfg(test)]
mod tests {
    use bevy_math::{IVec3, Vec3};

    use super::{
        AppearanceTarget, BearingDimensionError, BearingDimensions, BearingSpec, BuildCommand,
        BuildOutcome, ConstructionGraph, DriveLinkSpec, GraphError, PendingOperation,
        RigidLinkSpec, WeldSpec,
    };
    use crate::{
        ActuatorAssignment, BearingId, BuildPose, ConstructionMaterial, ControllerSpec, CuboidSpec,
        CylinderDimensions, CylinderSpec, DimensionLinkId, DimensionLinkSpec, DriveLimits,
        DriveName, DriveProgram, DriveState, DriveTarget, EngineKind, EngineSpec, FaceKind,
        FaceRef, GridRotation, InputSeatLinkSpec, InputSpec, MaterialAppearance, MaterialColor,
        MaterialDye, MaterialFinish, PartId, PartSpec, RegionError, RegionId,
        SeatControllerLinkSpec, SeatSpec, ShapeRegion, ShiftMode, TransmissionSpec,
    };

    fn suspension_fixture() -> (ConstructionGraph, [BearingSpec; 2]) {
        let mut graph = ConstructionGraph::new();
        let source = spawn(&mut graph, cube_at(0));
        let targets = [150, 250].map(|y| {
            spawn(
                &mut graph,
                CuboidSpec::new(
                    [1; 3],
                    BuildPose::from_position_ticks(IVec3::new(450, y, 0), GridRotation::default()),
                )
                .unwrap(),
            )
        });
        // The bearing attachment workflow groups construction on the opposite plate.
        graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: targets[0],
                second: targets[1],
            }))
            .unwrap();
        let spec = crate::SuspensionSpec::new(
            Some(crate::SpringSpec::default()),
            Some(crate::ShockSpec::default()),
            None,
        )
        .unwrap();
        let bearings = targets.map(|target| {
            BearingSpec::new(
                FaceRef::part(source, FaceKind::PositiveX),
                FaceRef::part(target, FaceKind::NegativeX),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::X,
            )
            .with_kind(crate::BearingKind::Suspension(spec))
        });
        (graph, bearings)
    }

    #[test]
    fn suspension_shared_mounts_compile_once_and_live_edits_preserve_attachment_ids() {
        let (mut graph, bearings) = suspension_fixture();
        let ids = bearings.map(|spec| {
            let BuildOutcome::BearingAdded(id) =
                graph.apply(BuildCommand::AddBearing(spec)).unwrap()
            else {
                unreachable!()
            };
            id
        });
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.bearings.len(), 1);
        assert_eq!(compiled.compounds.len(), 2);
        let changed = crate::SuspensionSpec::new(
            Some(crate::SpringSpec::new(0.5, 0.16, 0.12, 6, 0.025).unwrap()),
            Some(
                crate::ShockSpec::new(0.5, 0.1, crate::ShockBodyEnd::Opposite, 0.0, 2.0, 3.0)
                    .unwrap(),
            ),
            None,
        )
        .unwrap();
        graph
            .apply(BuildCommand::SetSuspension {
                bearing: ids[0],
                spec: changed,
            })
            .unwrap();
        assert_eq!(graph.bearing_count(), 2);
        for id in ids {
            assert_eq!(
                graph.bearing(id).unwrap().kind,
                crate::BearingKind::Suspension(changed)
            );
        }
        assert_eq!(graph.compile().unwrap().bearings.len(), 1);
        let moved = crate::SuspensionSpec::new(
            changed.spring(),
            Some(
                crate::ShockSpec::new(0.5, 0.1, crate::ShockBodyEnd::Opposite, 0.025, 2.0, 3.0)
                    .unwrap(),
            ),
            None,
        )
        .unwrap();
        assert_eq!(
            graph.apply(BuildCommand::SetSuspension {
                bearing: ids[0],
                spec: moved
            }),
            Err(GraphError::Suspension(
                crate::SuspensionError::AttachedSpacing
            ))
        );
        for id in ids {
            assert_eq!(
                graph.bearing(id).unwrap().kind,
                crate::BearingKind::Suspension(changed)
            );
        }
    }

    #[test]
    fn suspension_attachment_rejects_bad_axes_support_spacing_and_conflicting_specs() {
        let (mut graph, bearings) = suspension_fixture();
        for axis in [Vec3::ZERO, Vec3::X * 2.0, Vec3::Y, Vec3::splat(f32::NAN)] {
            let mut bad = bearings[0];
            bad.axis = axis;
            assert_eq!(
                graph.apply(BuildCommand::AddBearing(bad)),
                Err(GraphError::InvalidBearingAxis)
            );
        }
        for anchor in [
            Vec3::new(0.6, 0.5, 0.0),
            Vec3::new(0.5, 9.0, 0.0),
            Vec3::splat(f32::NAN),
        ] {
            let mut bad = bearings[0];
            bad.shared_anchor = anchor;
            assert_eq!(
                graph.apply(BuildCommand::AddBearing(bad)),
                Err(GraphError::BearingAnchorOutsideFaces)
            );
        }
        let mut wrong_spacing = bearings[0];
        wrong_spacing.kind = crate::BearingKind::Suspension(
            crate::SuspensionSpec::new(
                Some(crate::SpringSpec::default()),
                Some(
                    crate::ShockSpec::new(0.5, 0.1, crate::ShockBodyEnd::Source, 0.025, 1.0, 1.6)
                        .unwrap(),
                ),
                None,
            )
            .unwrap(),
        );
        assert_eq!(
            graph.apply(BuildCommand::AddBearing(wrong_spacing)),
            Err(GraphError::BearingAnchorOutsideFaces)
        );
        graph.apply(BuildCommand::AddBearing(bearings[0])).unwrap();
        let mut bad = bearings[1];
        bad.kind = crate::BearingKind::Suspension(
            crate::SuspensionSpec::new(Some(crate::SpringSpec::default()), None, None).unwrap(),
        );
        assert_eq!(
            graph.apply(BuildCommand::AddBearing(bad)),
            Err(GraphError::Suspension(crate::SuspensionError::SharedMounts))
        );
        assert_eq!(graph.bearing_count(), 1);
    }

    #[test]
    fn suspension_cannot_receive_a_controller_motor_assignment() {
        let (mut graph, bearings) = suspension_fixture();
        let BuildOutcome::BearingAdded(bearing) =
            graph.apply(BuildCommand::AddBearing(bearings[0])).unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(IVec3::new(20, 2, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        assert_eq!(
            graph.apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
                controller, bearing
            ))),
            Err(GraphError::IncompatibleDrive)
        );
    }

    fn cube_at(x: i32) -> CuboidSpec {
        CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(x, 2, 0), GridRotation::default()),
        )
        .unwrap()
    }

    fn spawn(graph: &mut ConstructionGraph, spec: CuboidSpec) -> crate::PartId {
        let BuildOutcome::Spawned(id) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            panic!("spawn returned wrong outcome")
        };
        id
    }

    fn spawn_cylinder(
        graph: &mut ConstructionGraph,
        dimensions: CylinderDimensions,
        pose: BuildPose,
    ) -> crate::PartId {
        let BuildOutcome::Spawned(id) = graph
            .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
                dimensions, pose,
            )))
            .unwrap()
        else {
            unreachable!()
        };
        id
    }

    #[test]
    fn dimension_link_ids_are_unique_and_queryable() {
        let mut graph = ConstructionGraph::new();
        let spec = DimensionLinkSpec::new(DimensionLinkId(9), BuildPose::default());
        let BuildOutcome::Spawned(link) =
            graph.apply(BuildCommand::SpawnDimensionLink(spec)).unwrap()
        else {
            unreachable!()
        };
        assert_eq!(graph.dimension_link(DimensionLinkId(9)), Some(link));
        assert_eq!(graph.dimension_link_id(link), Some(DimensionLinkId(9)));
        assert_eq!(
            graph.apply(BuildCommand::SpawnDimensionLink(spec)),
            Err(GraphError::DuplicateDimensionLink(DimensionLinkId(9)))
        );
    }

    #[test]
    fn structural_component_crosses_every_joint_and_partitions_cleanly() {
        let mut graph = ConstructionGraph::new();
        let welded = spawn(&mut graph, cube_at(0));
        let hinged = spawn(&mut graph, cube_at(4));
        let linked = spawn(&mut graph, cube_at(8));
        let cargo = spawn(&mut graph, cube_at(20));
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(welded, FaceKind::PositiveX),
                second: FaceRef::part(hinged, FaceKind::NegativeX),
            }))
            .unwrap();
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(hinged, FaceKind::PositiveX),
                FaceRef::part(linked, FaceKind::NegativeX),
                Vec3::new(1.5, 0.5, 0.0),
                Vec3::X,
            )))
            .unwrap();
        graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: linked,
                second: cargo,
            }))
            .unwrap();

        let component = graph.structural_component(welded, [cargo]).unwrap();
        assert_eq!(component.parts().count(), 4);
        assert!(component.touches_authored_ground());
        let partition = graph.partition(&component);
        assert_eq!(partition.component.part_count(), 4);
        assert_eq!(partition.component.weld_count(), 1);
        assert_eq!(partition.component.bearing_count(), 1);
        assert_eq!(partition.component.rigid_link_count(), 1);
        assert_eq!(partition.remainder.part_count(), 0);
    }

    #[test]
    fn input_routes_are_typed_cardinal_and_cascade_with_the_seat() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(input) = graph
            .apply(BuildCommand::SpawnInput(InputSpec::new(
                BuildPose::default(),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(seat) = graph
            .apply(BuildCommand::SpawnSeat(SeatSpec::new(BuildPose::default())))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::default(),
            )))
            .unwrap()
        else {
            unreachable!()
        };

        graph
            .apply(BuildCommand::AddInputSeatLink(InputSeatLinkSpec {
                input,
                seat,
            }))
            .unwrap();
        graph
            .apply(BuildCommand::AddSeatControllerLink(
                SeatControllerLinkSpec { seat, controller },
            ))
            .unwrap();
        assert_eq!(graph.seat_input(seat), Some(input));
        assert_eq!(graph.seat_controller(seat), Some(controller));
        assert_eq!(
            graph.apply(BuildCommand::AddInputSeatLink(InputSeatLinkSpec {
                input,
                seat,
            })),
            Err(GraphError::InputAlreadyLinked(input))
        );

        graph.apply(BuildCommand::Remove(seat)).unwrap();
        assert_eq!(graph.input_seat_links().count(), 0);
        assert_eq!(graph.seat_controller_links().count(), 0);
        assert!(graph.part(input).is_some());
        assert!(graph.part(controller).is_some());
    }

    #[test]
    fn structural_blocks_do_not_bridge_controller_actuator_modules() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::from_half_grid(IVec3::ZERO, GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let bridge = spawn(
            &mut graph,
            CuboidSpec::new(
                [1, 2, 1],
                BuildPose::from_half_grid(IVec3::new(3, 0, 0), GridRotation::default()),
            )
            .unwrap(),
        );
        let BuildOutcome::Spawned(engine) = graph
            .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                EngineKind::Electric,
                BuildPose::from_half_grid(IVec3::new(6, 0, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        for (first, second) in [(controller, bridge), (bridge, engine)] {
            graph
                .apply(BuildCommand::Weld(WeldSpec {
                    first: FaceRef::part(first, FaceKind::PositiveX),
                    second: FaceRef::part(second, FaceKind::NegativeX),
                }))
                .unwrap();
        }

        let inventory = graph.actuator_inventory(controller).unwrap();
        assert_eq!(inventory.electric_engines, 0);
    }

    #[test]
    fn graph_stores_generalized_parts_and_rejects_cylinder_walls_as_faces() {
        let mut graph = ConstructionGraph::new();
        let cylinder = spawn_cylinder(
            &mut graph,
            CylinderDimensions::default(),
            BuildPose::default(),
        );
        assert!(matches!(graph.part(cylinder), Some(PartSpec::Cylinder(_))));
        assert_eq!(
            graph.apply(BuildCommand::BeginPending(PendingOperation::Weld(
                FaceRef::part(cylinder, FaceKind::PositiveX)
            ))),
            Err(GraphError::InvalidCylinderFace)
        );
        assert!(graph.pending().is_none());
    }

    #[test]
    fn mixed_welds_require_positive_annular_material_overlap() {
        let mut graph = ConstructionGraph::new();
        let cylinder = spawn_cylinder(
            &mut graph,
            CylinderDimensions::new(1.0, 0.5, 0.25).unwrap(),
            BuildPose::default(),
        );
        let centered = spawn(
            &mut graph,
            CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_half_grid(bevy_math::IVec3::new(0, 2, 0), GridRotation::default()),
            )
            .unwrap(),
        );
        let weld = |part| {
            BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(cylinder, FaceKind::PositiveY),
                second: FaceRef::part(part, FaceKind::NegativeY),
            })
        };
        assert_eq!(
            graph.apply(weld(centered)),
            Err(GraphError::FacesDoNotTouch)
        );

        let ring = spawn(
            &mut graph,
            CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_half_grid(bevy_math::IVec3::new(3, 2, 0), GridRotation::default()),
            )
            .unwrap(),
        );
        assert!(matches!(
            graph.apply(weld(ring)),
            Ok(BuildOutcome::Welded(_))
        ));
    }

    #[test]
    fn cylinder_sector_end_connects_only_through_retained_material() {
        let dimensions = CylinderDimensions::new(1.0, 0.0, 0.25)
            .unwrap()
            .with_sweep_angle_degrees(90)
            .unwrap();
        let attempt = |x_half_units| {
            let mut graph = ConstructionGraph::new();
            let cylinder = spawn_cylinder(&mut graph, dimensions, BuildPose::default());
            let block = spawn(
                &mut graph,
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_half_grid(
                        IVec3::new(x_half_units, 2, 0),
                        GridRotation::default(),
                    ),
                )
                .unwrap(),
            );
            graph.apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(cylinder, FaceKind::PositiveY),
                second: FaceRef::part(block, FaceKind::NegativeY),
            }))
        };

        assert!(matches!(attempt(3), Ok(BuildOutcome::Welded(_))));
        assert_eq!(attempt(-3), Err(GraphError::FacesDoNotTouch));
    }

    #[test]
    fn bearing_dimensions_validate_defaults_bounds_and_ring_gap() {
        let default = BearingDimensions::default();
        assert!((default.outer_diameter() - 0.25).abs() < f32::EPSILON);
        assert!((default.inner_diameter() - 0.10).abs() < f32::EPSILON);

        let solid_minimum = BearingDimensions::new(0.05, 0.0).unwrap();
        assert!((solid_minimum.outer_diameter() - 0.05).abs() < f32::EPSILON);
        assert!(solid_minimum.inner_diameter().abs() < f32::EPSILON);
        assert!(BearingDimensions::new(8.0, 7.95).is_ok());
        assert!(BearingDimensions::new(1.234, 0.678).is_ok());

        assert_eq!(
            BearingDimensions::new(f32::NAN, 0.0),
            Err(BearingDimensionError::NonFiniteOuterDiameter)
        );
        assert_eq!(
            BearingDimensions::new(f32::INFINITY, 0.0),
            Err(BearingDimensionError::NonFiniteOuterDiameter)
        );
        assert_eq!(
            BearingDimensions::new(0.049, 0.0),
            Err(BearingDimensionError::OuterDiameterOutOfRange)
        );
        assert_eq!(
            BearingDimensions::new(8.001, 0.0),
            Err(BearingDimensionError::OuterDiameterOutOfRange)
        );
        assert_eq!(
            BearingDimensions::new(0.25, f32::NAN),
            Err(BearingDimensionError::NonFiniteInnerDiameter)
        );
        assert_eq!(
            BearingDimensions::new(0.25, -0.001),
            Err(BearingDimensionError::InnerDiameterOutOfRange)
        );
        assert_eq!(
            BearingDimensions::new(0.25, 0.201),
            Err(BearingDimensionError::InnerDiameterOutOfRange)
        );
    }

    #[test]
    fn bearing_spec_defaults_and_accepts_custom_dimensions() {
        let source = FaceRef::ground();
        let target = FaceRef::ground();
        let default = BearingSpec::new(source, target, Vec3::ZERO, Vec3::Y);
        assert_eq!(default.dimensions, BearingDimensions::default());

        let custom = BearingDimensions::new(2.5, 1.25).unwrap();
        assert_eq!(default.with_dimensions(custom).dimensions, custom);
    }

    #[test]
    fn weld_requires_touching_opposed_faces() {
        let mut graph = ConstructionGraph::new();
        let left = spawn(&mut graph, cube_at(0));
        let right = spawn(&mut graph, cube_at(4));
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(left, FaceKind::PositiveX),
                second: FaceRef::part(right, FaceKind::NegativeX),
            }))
            .unwrap();
        assert_eq!(graph.weld_count(), 1);
    }

    #[test]
    fn failed_command_is_transactional() {
        let mut graph = ConstructionGraph::new();
        let left = spawn(&mut graph, cube_at(0));
        let right = spawn(&mut graph, cube_at(8));
        let result = graph.apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(left, FaceKind::PositiveX),
            second: FaceRef::part(right, FaceKind::NegativeX),
        }));

        assert_eq!(result, Err(GraphError::FacesDoNotTouch));
        assert_eq!(graph.part_count(), 2);
        assert_eq!(graph.weld_count(), 0);
    }

    #[test]
    fn deletion_cascades_connections_and_invalidates_old_handle() {
        let mut graph = ConstructionGraph::new();
        let left = spawn(&mut graph, cube_at(0));
        let right = spawn(&mut graph, cube_at(4));
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(left, FaceKind::PositiveX),
                second: FaceRef::part(right, FaceKind::NegativeX),
            }))
            .unwrap();

        graph.apply(BuildCommand::Remove(left)).unwrap();

        assert!(graph.part(left).is_none());
        assert_eq!(graph.weld_count(), 0);
    }

    #[test]
    fn rigid_links_validate_distinct_parts_and_cascade_on_deletion() {
        let mut graph = ConstructionGraph::new();
        let first = spawn(&mut graph, cube_at(0));
        let second = spawn(&mut graph, cube_at(8));
        assert_eq!(
            graph.apply(BuildCommand::RigidLink(RigidLinkSpec {
                first,
                second: first,
            })),
            Err(GraphError::SameRigidLinkPart)
        );

        let BuildOutcome::RigidLinked(link) = graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec { first, second }))
            .unwrap()
        else {
            unreachable!()
        };
        assert_eq!(graph.rigid_link_count(), 1);
        assert_eq!(graph.rigid_link(link).unwrap().second, second);

        graph.apply(BuildCommand::Remove(first)).unwrap();
        assert_eq!(graph.rigid_link_count(), 0);
        assert_eq!(
            graph.apply(BuildCommand::RemoveRigidLink(link)),
            Err(GraphError::MissingRigidLink(link))
        );
    }

    #[test]
    fn bearing_axis_and_anchor_are_derived_geometry() {
        let mut graph = ConstructionGraph::new();
        let left = spawn(&mut graph, cube_at(0));
        let right = spawn(&mut graph, cube_at(4));
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(left, FaceKind::PositiveX),
                FaceRef::part(right, FaceKind::NegativeX),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::X,
            )))
            .unwrap();

        assert_eq!(graph.bearing_count(), 1);
    }

    #[test]
    fn oversized_bearing_can_attach_to_an_offset_face_covered_by_its_ring() {
        let mut graph = ConstructionGraph::new();
        let source = spawn(&mut graph, cube_at(0));
        let target = spawn(
            &mut graph,
            CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_half_grid(IVec3::new(5, 9, 1), GridRotation::default()),
            )
            .unwrap(),
        );
        let dimensions = BearingDimensions::new(2.0, 0.10).unwrap();

        graph
            .apply(BuildCommand::AddBearing(
                BearingSpec::new(
                    FaceRef::part(source, FaceKind::PositiveX),
                    FaceRef::part(target, FaceKind::NegativeX),
                    Vec3::new(0.5, 0.5, 0.0),
                    Vec3::X,
                )
                .with_dimensions(dimensions),
            ))
            .unwrap();

        assert_eq!(graph.bearing_count(), 1);
    }

    #[test]
    fn bearing_does_not_attach_to_a_face_entirely_inside_its_hole() {
        let mut graph = ConstructionGraph::new();
        let source = spawn(&mut graph, cube_at(0));
        let target = spawn(
            &mut graph,
            CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_half_grid(IVec3::new(5, 4, 0), GridRotation::default()),
            )
            .unwrap(),
        );
        let dimensions = BearingDimensions::new(2.0, 1.0).unwrap();

        assert_eq!(
            graph.apply(BuildCommand::AddBearing(
                BearingSpec::new(
                    FaceRef::part(source, FaceKind::PositiveX),
                    FaceRef::part(target, FaceKind::NegativeX),
                    Vec3::new(0.5, 0.5, 0.0),
                    Vec3::X,
                )
                .with_dimensions(dimensions),
            )),
            Err(GraphError::BearingAnchorOutsideFaces)
        );
    }

    #[test]
    fn connections_can_be_deleted_independently() {
        let mut graph = ConstructionGraph::new();
        let left = spawn(&mut graph, cube_at(0));
        let right = spawn(&mut graph, cube_at(4));
        let BuildOutcome::BearingAdded(bearing) = graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(left, FaceKind::PositiveX),
                FaceRef::part(right, FaceKind::NegativeX),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::X,
            )))
            .unwrap()
        else {
            panic!("wrong bearing outcome")
        };

        graph.apply(BuildCommand::RemoveBearing(bearing)).unwrap();

        assert_eq!(graph.part_count(), 2);
        assert_eq!(graph.bearing_count(), 0);
        assert_eq!(
            graph.apply(BuildCommand::RemoveBearing(bearing)),
            Err(GraphError::MissingBearing(bearing))
        );
    }

    fn controller_at(x: i32) -> ControllerSpec {
        ControllerSpec::new(BuildPose::from_half_grid(
            IVec3::new(x, 2, 0),
            GridRotation::default(),
        ))
    }

    fn hinged_pair(graph: &mut ConstructionGraph) -> BearingId {
        let left = spawn(graph, cube_at(0));
        let right = spawn(graph, cube_at(4));
        let BuildOutcome::BearingAdded(bearing) = graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(left, FaceKind::PositiveX),
                FaceRef::part(right, FaceKind::NegativeX),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::X,
            )))
            .unwrap()
        else {
            unreachable!()
        };
        bearing
    }

    fn spawn_controller(graph: &mut ConstructionGraph, spec: ControllerSpec) -> PartId {
        let BuildOutcome::Spawned(id) = graph.apply(BuildCommand::SpawnController(spec)).unwrap()
        else {
            unreachable!()
        };
        id
    }

    #[test]
    fn control_block_exposes_cuboid_faces_and_reprogrammable_wires() {
        let mut graph = ConstructionGraph::new();
        let bearing = hinged_pair(&mut graph);
        let controller = spawn_controller(&mut graph, controller_at(20));
        assert!(matches!(
            graph.part(controller),
            Some(PartSpec::Controller(_))
        ));
        assert!(
            graph
                .face_geometry(FaceRef::part(controller, FaceKind::PositiveX))
                .is_ok()
        );
        assert!(graph.is_controller(controller));

        let BuildOutcome::DriveLinked(link) = graph
            .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
                controller, bearing,
            )))
            .unwrap()
        else {
            unreachable!()
        };
        assert_eq!(
            graph.drive_link(link).map(|spec| spec.limits),
            Some(DriveLimits::default())
        );

        let limits = DriveLimits::new(2.0, 12.0, Some((-0.5, 0.5))).unwrap();
        let program =
            DriveProgram::new(&[DriveState::new(DriveTarget::Angle(0.25)).unwrap()], false)
                .unwrap();
        assert_eq!(
            graph.apply(BuildCommand::SetDriveLink {
                link,
                limits,
                program,
                name: DriveName::new("Steer · front left"),
                actuator: ActuatorAssignment::Unpowered,
            }),
            Ok(BuildOutcome::DriveUpdated)
        );
        let stored = graph.drive_link(link).copied().unwrap();
        assert_eq!(stored.limits, limits);
        assert_eq!(stored.program, program);
        assert_eq!(stored.name.as_str(), "Steer · front left");

        // A control block owns its wires, and reprogramming one leaves the
        // block itself untouched.
        assert_eq!(graph.controller_links(controller).count(), 1);
    }

    #[test]
    fn drive_link_requires_a_controller_part_and_an_undriven_bearing() {
        let mut graph = ConstructionGraph::new();
        let bearing = hinged_pair(&mut graph);
        let block = spawn(&mut graph, cube_at(12));
        let controller = spawn_controller(&mut graph, controller_at(20));
        let other = spawn_controller(&mut graph, controller_at(28));

        assert_eq!(
            graph.apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
                block, bearing
            ))),
            Err(GraphError::NotAController(block))
        );
        assert_eq!(
            graph.apply(BuildCommand::BeginPending(PendingOperation::DriveLink(
                block
            ))),
            Err(GraphError::NotAController(block))
        );

        assert!(matches!(
            graph.apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
                controller, bearing
            ))),
            Ok(BuildOutcome::DriveLinked(_))
        ));
        assert_eq!(graph.drive_link_count(), 1);
        assert_eq!(
            graph.apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
                other, bearing
            ))),
            Err(GraphError::BearingAlreadyDriven(bearing))
        );
    }

    #[test]
    fn reversed_wire_flips_the_programmed_direction() {
        let mut graph = ConstructionGraph::new();
        let bearing = hinged_pair(&mut graph);
        let controller = spawn_controller(&mut graph, controller_at(20));
        let program =
            DriveProgram::new(&[DriveState::new(DriveTarget::Speed(2.0)).unwrap()], false).unwrap();

        let mut spec = DriveLinkSpec::new(controller, bearing);
        spec.program = program;
        let BuildOutcome::DriveLinked(link) =
            graph.apply(BuildCommand::AddDriveLink(spec)).unwrap()
        else {
            unreachable!()
        };
        assert_eq!(
            graph
                .drive_link(link)
                .and_then(|spec| spec.resolved_target(0)),
            Some(DriveTarget::Speed(2.0))
        );

        graph.apply(BuildCommand::RemoveDriveLink(link)).unwrap();
        assert!(graph.bearing_drive_link(bearing).is_none());

        spec.reversed = true;
        graph.apply(BuildCommand::AddDriveLink(spec)).unwrap();
        assert_eq!(
            graph
                .bearing_drive_link(bearing)
                .and_then(|(_, link)| link.resolved_target(0)),
            Some(DriveTarget::Speed(-2.0))
        );
    }

    #[test]
    fn deleting_a_controller_or_bearing_cascades_its_drive_links() {
        let mut graph = ConstructionGraph::new();
        let bearing = hinged_pair(&mut graph);
        let controller = spawn_controller(&mut graph, controller_at(20));
        let BuildOutcome::DriveLinked(link) = graph
            .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
                controller, bearing,
            )))
            .unwrap()
        else {
            unreachable!()
        };

        graph.apply(BuildCommand::RemoveBearing(bearing)).unwrap();
        assert_eq!(graph.drive_link_count(), 0);
        assert_eq!(
            graph.apply(BuildCommand::RemoveDriveLink(link)),
            Err(GraphError::MissingDriveLink(link))
        );

        let mut graph = ConstructionGraph::new();
        let bearing = hinged_pair(&mut graph);
        let controller = spawn_controller(&mut graph, controller_at(20));
        graph
            .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
                controller, bearing,
            )))
            .unwrap();
        graph.apply(BuildCommand::Remove(controller)).unwrap();
        assert_eq!(graph.drive_link_count(), 0);
        assert_eq!(graph.bearing_count(), 1);
    }

    #[test]
    fn deleting_a_bearings_support_part_also_drops_its_drive_wire() {
        let mut graph = ConstructionGraph::new();
        let left = spawn(&mut graph, cube_at(0));
        let right = spawn(&mut graph, cube_at(4));
        let BuildOutcome::BearingAdded(bearing) = graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(left, FaceKind::PositiveX),
                FaceRef::part(right, FaceKind::NegativeX),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::X,
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let controller = spawn_controller(&mut graph, controller_at(20));
        graph
            .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
                controller, bearing,
            )))
            .unwrap();

        graph.apply(BuildCommand::Remove(left)).unwrap();

        assert_eq!(graph.bearing_count(), 0);
        assert_eq!(graph.drive_link_count(), 0);
    }

    /// One construction cell, in the steps a cage vertex moves in.
    fn cell_steps() -> i16 {
        i16::try_from(crate::STEPS_PER_CELL).expect("a cell is twenty steps")
    }

    /// A solid run of `size` one-cell blocks welded together from the origin.
    fn welded_blocks(size: IVec3, material: ConstructionMaterial) -> ConstructionGraph {
        let mut graph = ConstructionGraph::new();
        let mut previous: Option<(PartId, IVec3)> = None;
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let at = IVec3::new(x, y, z);
                    let spec = CuboidSpec::new(
                        [1, 1, 1],
                        BuildPose::from_half_grid(IVec3::ONE + at * 2, GridRotation::default()),
                    )
                    .unwrap()
                    .with_material(material);
                    let BuildOutcome::Spawned(id) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
                    else {
                        panic!("wrong spawn outcome")
                    };
                    // Weld everything into one body so a region may claim it.
                    if let Some((earlier, _)) = previous {
                        graph
                            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                                first: earlier,
                                second: id,
                            }))
                            .unwrap();
                    }
                    previous = Some((id, at));
                }
            }
        }
        graph
    }

    fn add_region(
        graph: &mut ConstructionGraph,
        size: IVec3,
        material: ConstructionMaterial,
    ) -> Result<RegionId, GraphError> {
        let region = ShapeRegion::new(IVec3::ZERO, size, material).unwrap();
        match graph.apply(BuildCommand::AddRegion(region))? {
            BuildOutcome::RegionAdded(id) => Ok(id),
            other => panic!("wrong outcome {other:?}"),
        }
    }

    #[test]
    fn a_fresh_region_has_eight_cage_vertices() {
        let mut graph = welded_blocks(IVec3::new(2, 2, 2), ConstructionMaterial::Steel);
        let id = add_region(&mut graph, IVec3::new(2, 2, 2), ConstructionMaterial::Steel).unwrap();
        let region = graph.region(id).unwrap();
        assert_eq!(region.plane_counts(), [2, 2, 2]);
        assert_eq!(region.vertices().count(), 8, "a fresh cage is a box");
        assert!(region.is_unshaped());
    }

    #[test]
    fn a_region_refuses_an_area_with_a_hole() {
        // Two of the four cells filled: the area is not solid.
        let mut graph = welded_blocks(IVec3::new(2, 1, 1), ConstructionMaterial::Steel);
        assert!(matches!(
            add_region(&mut graph, IVec3::new(2, 2, 1), ConstructionMaterial::Steel),
            Err(GraphError::RegionNotSolid(2))
        ));
        assert_eq!(graph.regions().count(), 0);
    }

    #[test]
    fn a_region_can_span_a_whole_run_of_welded_blocks() {
        let mut graph = welded_blocks(IVec3::new(3, 2, 2), ConstructionMaterial::Steel);
        let id = add_region(&mut graph, IVec3::new(3, 2, 2), ConstructionMaterial::Steel).unwrap();
        let region = graph.region(id).unwrap();
        assert_eq!(region.size_cells(), IVec3::new(3, 2, 2));
        // Twelve blocks, one shape: the cage still has only its eight corners.
        assert_eq!(region.vertices().count(), 8);
    }

    #[test]
    fn a_region_refuses_to_hold_only_part_of_a_block() {
        // One two-cell beam, and an area covering just one of its cells.
        let mut graph = ConstructionGraph::new();
        let spec = CuboidSpec::new(
            [2, 1, 1],
            BuildPose::from_half_grid(IVec3::new(2, 1, 1), GridRotation::default()),
        )
        .unwrap()
        .with_material(ConstructionMaterial::Steel);
        graph.apply(BuildCommand::Spawn(spec)).unwrap();
        assert!(matches!(
            add_region(&mut graph, IVec3::ONE, ConstructionMaterial::Steel),
            Err(GraphError::RegionSplitsPart)
        ));
        // The whole beam is fine.
        add_region(&mut graph, IVec3::new(2, 1, 1), ConstructionMaterial::Steel).unwrap();
    }

    #[test]
    fn a_region_refuses_mixed_materials() {
        let mut graph = welded_blocks(IVec3::new(1, 1, 1), ConstructionMaterial::Steel);
        let spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(IVec3::new(3, 1, 1), GridRotation::default()),
        )
        .unwrap()
        .with_material(ConstructionMaterial::Wood);
        graph.apply(BuildCommand::Spawn(spec)).unwrap();
        assert!(matches!(
            add_region(&mut graph, IVec3::new(2, 1, 1), ConstructionMaterial::Steel),
            Err(GraphError::RegionMixedMaterials)
        ));
    }

    fn blue_paint() -> MaterialAppearance {
        MaterialAppearance::new(
            MaterialColor::Dye(MaterialDye::new([42, 76, 199], 1.25).unwrap()),
            MaterialFinish::Painted,
        )
    }

    #[test]
    fn a_region_requires_one_appearance_and_painting_it_updates_every_member() {
        let size = IVec3::new(2, 1, 1);
        let mut graph = welded_blocks(size, ConstructionMaterial::Steel);
        let first = graph.parts().next().unwrap().0;
        graph
            .apply(BuildCommand::SetAppearance {
                target: AppearanceTarget::Part(first),
                appearance: blue_paint(),
            })
            .unwrap();
        assert_eq!(
            add_region(&mut graph, size, ConstructionMaterial::Steel),
            Err(GraphError::RegionMixedAppearances)
        );

        graph
            .apply(BuildCommand::SetAppearance {
                target: AppearanceTarget::Part(first),
                appearance: MaterialAppearance::BAKED,
            })
            .unwrap();
        let region = add_region(&mut graph, size, ConstructionMaterial::Steel).unwrap();
        graph
            .apply(BuildCommand::SetAppearance {
                target: AppearanceTarget::Part(first),
                appearance: blue_paint(),
            })
            .unwrap();
        assert_eq!(graph.region(region).unwrap().appearance(), blue_paint());
        assert!(
            graph
                .parts()
                .filter(|(part, _)| graph.region_of(*part) == Some(region))
                .all(|(_, part)| part.appearance() == Some(blue_paint()))
        );
        graph
            .apply(BuildCommand::SetAppearance {
                target: AppearanceTarget::Region(region),
                appearance: MaterialAppearance::BAKED,
            })
            .unwrap();
        assert_eq!(
            graph
                .apply(BuildCommand::SetAppearance {
                    target: AppearanceTarget::Region(region),
                    appearance: blue_paint(),
                })
                .unwrap(),
            BuildOutcome::AppearanceUpdated
        );
        assert_eq!(graph.region(region).unwrap().appearance(), blue_paint());
        assert!(
            graph
                .parts()
                .filter(|(part, _)| graph.region_of(*part) == Some(region))
                .all(|(_, part)| part.appearance() == Some(blue_paint()))
        );
    }

    #[test]
    fn authored_machine_parts_reject_appearance_edits() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::default(),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        assert_eq!(
            graph.apply(BuildCommand::SetAppearance {
                target: AppearanceTarget::Part(controller),
                appearance: blue_paint(),
            }),
            Err(GraphError::AuthoredAppearance(controller))
        );
    }

    #[test]
    fn a_region_refuses_overlapping_another() {
        let mut graph = welded_blocks(IVec3::new(3, 1, 1), ConstructionMaterial::Steel);
        add_region(&mut graph, IVec3::new(2, 1, 1), ConstructionMaterial::Steel).unwrap();
        let overlapping = ShapeRegion::new(
            IVec3::new(2, 0, 0),
            IVec3::new(2, 1, 1),
            ConstructionMaterial::Steel,
        )
        .unwrap();
        assert!(matches!(
            graph.apply(BuildCommand::AddRegion(overlapping)),
            Err(GraphError::RegionOverlaps(_))
        ));
        assert_eq!(graph.regions().count(), 1);
    }

    #[test]
    fn deleting_a_block_deletes_the_region_it_belonged_to() {
        // A region needs every cell filled, so it cannot outlive its blocks.
        let mut graph = welded_blocks(IVec3::new(2, 1, 1), ConstructionMaterial::Steel);
        add_region(&mut graph, IVec3::new(2, 1, 1), ConstructionMaterial::Steel).unwrap();
        let victim = graph.parts().next().expect("the graph has blocks").0;
        graph.apply(BuildCommand::Remove(victim)).unwrap();
        assert_eq!(graph.regions().count(), 0);
    }

    #[test]
    fn a_cage_vertex_cannot_leave_the_regions_bounding_box() {
        let mut graph = welded_blocks(IVec3::new(1, 1, 1), ConstructionMaterial::Steel);
        let id = add_region(&mut graph, IVec3::ONE, ConstructionMaterial::Steel).unwrap();
        // Corner [0,0,0] sits at the minimum, so it can only move inward.
        assert!(matches!(
            graph.apply(BuildCommand::SetRegionVertices {
                region: id,
                vertices: vec![([0, 0, 0], [-1, 0, 0])],
            }),
            Err(GraphError::InvalidRegion(RegionError::OutsideBounds))
        ));
        assert!(
            graph.region(id).unwrap().is_unshaped(),
            "a rejected move must leave the graph byte-for-byte equivalent"
        );
        // Inward as far as the far face is fine.
        graph
            .apply(BuildCommand::SetRegionVertices {
                region: id,
                vertices: vec![([0, 0, 0], [cell_steps(), 0, 0])],
            })
            .unwrap();
    }

    #[test]
    fn a_cage_move_that_would_invert_a_cell_is_rejected_and_changes_nothing() {
        let mut graph = welded_blocks(IVec3::new(1, 1, 1), ConstructionMaterial::Steel);
        let id = add_region(&mut graph, IVec3::ONE, ConstructionMaterial::Steel).unwrap();
        let cell = cell_steps();
        assert!(matches!(
            graph.apply(BuildCommand::SetRegionVertices {
                region: id,
                vertices: vec![([0, 0, 0], [cell, 0, 0]), ([1, 0, 0], [-cell, 0, 0])],
            }),
            Err(GraphError::InvertedCell(_))
        ));
        assert!(graph.region(id).unwrap().is_unshaped());
    }

    #[test]
    fn collapsing_an_edge_into_a_wedge_is_accepted() {
        let mut graph = welded_blocks(IVec3::new(1, 1, 1), ConstructionMaterial::Steel);
        let id = add_region(&mut graph, IVec3::ONE, ConstructionMaterial::Steel).unwrap();
        let cell = cell_steps();
        graph
            .apply(BuildCommand::SetRegionVertices {
                region: id,
                vertices: vec![([0, 1, 1], [0, -cell, 0]), ([1, 1, 1], [0, -cell, 0])],
            })
            .expect("collapsing an edge makes a wedge");
        assert_eq!(graph.region(id).unwrap().offsets().count(), 2);
    }

    #[test]
    fn subdividing_inserts_a_whole_plane_without_moving_the_surface() {
        let mut graph = welded_blocks(IVec3::new(2, 1, 1), ConstructionMaterial::Steel);
        let id = add_region(&mut graph, IVec3::new(2, 1, 1), ConstructionMaterial::Steel).unwrap();
        let cell = cell_steps();
        graph
            .apply(BuildCommand::SetRegionVertices {
                region: id,
                vertices: vec![([1, 1, 0], [0, -cell, 0]), ([1, 1, 1], [0, -cell, 0])],
            })
            .unwrap();
        let before = surface_corners(graph.region(id).unwrap());

        graph
            .apply(BuildCommand::SubdivideRegion {
                region: id,
                axis: 0,
                position: 1,
            })
            .unwrap();
        let region = graph.region(id).unwrap();
        assert_eq!(
            region.plane_counts(),
            [3, 2, 2],
            "a whole plane is inserted"
        );
        assert_eq!(
            surface_corners(region),
            before,
            "the cage gains handles; the surface must not move"
        );
    }

    /// The eight outer cage corners, for comparing a surface before and after.
    fn surface_corners(region: &ShapeRegion) -> Vec<[i32; 3]> {
        let [x, y, z] = region.plane_counts();
        let last = [x - 1, y - 1, z - 1];
        let mut corners = Vec::new();
        for corner in 0..8_usize {
            let index = [
                u16::try_from(if corner & 1 == 0 { 0 } else { last[0] }).unwrap(),
                u16::try_from(if corner & 2 == 0 { 0 } else { last[1] }).unwrap(),
                u16::try_from(if corner & 4 == 0 { 0 } else { last[2] }).unwrap(),
            ];
            corners.push(region.vertex_steps(index).unwrap().to_array());
        }
        corners
    }

    fn engine(graph: &mut ConstructionGraph, kind: EngineKind, units: IVec3) -> PartId {
        let BuildOutcome::Spawned(engine) = graph
            .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                kind,
                BuildPose::new(units, GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        engine
    }

    fn attach(graph: &mut ConstructionGraph, parent: PartId) -> PartId {
        let spec = graph.next_transmission_spec(parent).unwrap();
        let BuildOutcome::Spawned(transmission) = graph
            .apply(BuildCommand::AttachTransmission { parent, spec })
            .unwrap()
        else {
            unreachable!()
        };
        transmission
    }

    #[test]
    fn transmissions_form_one_inherited_seventeen_block_chain() {
        let mut graph = ConstructionGraph::new();
        let engine = engine(&mut graph, EngineKind::Gas, IVec3::ZERO);
        let mut tail = engine;
        for depth in 1..=17 {
            tail = attach(&mut graph, tail);
            assert_eq!(
                graph.transmission_root(tail),
                Some((engine, EngineKind::Gas, depth))
            );
            assert_eq!(graph.transmission_kind(tail), Some(EngineKind::Gas));
            assert_eq!(
                graph.part(tail).unwrap().pose().rotation,
                GridRotation::default()
            );
        }
        assert_eq!(
            graph.next_transmission_spec(tail),
            Err(GraphError::TransmissionLimitReached)
        );
        assert_eq!(graph.engine_transmission_depth(engine), Some(17));
    }

    #[test]
    fn transmission_attachment_is_atomic_and_protects_its_weld() {
        let mut graph = ConstructionGraph::new();
        let engine = engine(&mut graph, EngineKind::Electric, IVec3::ZERO);
        let expected = graph.next_transmission_spec(engine).unwrap();
        let wrong = TransmissionSpec::new(BuildPose::new(
            IVec3::new(0, 0, 99),
            GridRotation::new(0, 1, 0),
        ));
        assert_eq!(
            graph.apply(BuildCommand::AttachTransmission {
                parent: engine,
                spec: wrong,
            }),
            Err(GraphError::InvalidTransmissionPose)
        );
        assert_eq!(graph.part_count(), 1, "failed attachment changes nothing");

        let BuildOutcome::Spawned(first) = graph
            .apply(BuildCommand::AttachTransmission {
                parent: engine,
                spec: expected,
            })
            .unwrap()
        else {
            unreachable!()
        };
        assert_eq!(
            graph.next_transmission_spec(engine),
            Err(GraphError::TransmissionOutputOccupied(engine))
        );
        let weld = graph.transmission_weld(first).unwrap();
        assert_eq!(
            graph.apply(BuildCommand::RemoveWeld(weld)),
            Err(GraphError::RequiredTransmissionWeld(weld))
        );
    }

    #[test]
    fn deleting_upstream_transmission_parts_cascades_downstream() {
        let mut graph = ConstructionGraph::new();
        let engine = engine(&mut graph, EngineKind::Gas, IVec3::ZERO);
        let first = attach(&mut graph, engine);
        let second = attach(&mut graph, first);
        let third = attach(&mut graph, second);

        graph.apply(BuildCommand::Remove(second)).unwrap();
        assert!(graph.part(engine).is_some());
        assert!(graph.part(first).is_some());
        assert!(graph.part(second).is_none());
        assert!(graph.part(third).is_none());
        assert_eq!(graph.engine_transmission_depth(engine), Some(1));

        graph.apply(BuildCommand::Remove(engine)).unwrap();
        assert_eq!(graph.part_count(), 0);
        assert_eq!(graph.weld_count(), 0);
    }

    #[test]
    fn same_type_stack_mismatch_is_visible_and_blocks_compile_only() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(IVec3::ZERO, GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let first = engine(&mut graph, EngineKind::Electric, IVec3::new(2, 0, 0));
        let second = engine(&mut graph, EngineKind::Electric, IVec3::new(4, 0, 0));
        for (left, right) in [(controller, first), (first, second)] {
            graph
                .apply(BuildCommand::Weld(WeldSpec {
                    first: FaceRef::part(left, FaceKind::PositiveX),
                    second: FaceRef::part(right, FaceKind::NegativeX),
                }))
                .unwrap();
        }
        attach(&mut graph, first);

        let inventory = graph.actuator_inventory(controller).unwrap();
        assert_eq!(inventory.electric_engines, 2);
        assert!(inventory.electric_transmission_mismatch);
        assert_eq!(inventory.electric_transmission_depth, None);
        assert!(matches!(
            graph.compile(),
            Err(crate::TopologyError::TransmissionDepthMismatch {
                kind: EngineKind::Electric,
                ..
            })
        ));
    }

    #[test]
    fn gearbox_edits_validate_physical_gear_count_and_are_transactional() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(IVec3::ZERO, GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let engine = engine(&mut graph, EngineKind::Gas, IVec3::new(2, 0, 0));
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(controller, FaceKind::PositiveX),
                second: FaceRef::part(engine, FaceKind::NegativeX),
            }))
            .unwrap();
        attach(&mut graph, engine);

        graph
            .apply(BuildCommand::SetGearboxMode {
                controller,
                kind: EngineKind::Gas,
                mode: ShiftMode::Manual,
            })
            .unwrap();
        assert_eq!(
            graph
                .gearbox_config(controller, EngineKind::Gas)
                .unwrap()
                .mode(),
            ShiftMode::Manual
        );
        assert!(matches!(
            graph.apply(BuildCommand::SetGearboxRatios {
                controller,
                kind: EngineKind::Gas,
                ratios: vec![1.0],
            }),
            Err(GraphError::GearCountMismatch { .. })
        ));
        assert_eq!(
            graph
                .gearbox_config(controller, EngineKind::Gas)
                .unwrap()
                .ratios(),
            &[3.0, 1.0]
        );
    }

    #[test]
    fn shape_features_add_adjust_remove_and_reject_invalid_amounts_atomically() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [2, 2, 2],
                    BuildPose::new(IVec3::ZERO, GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let owner = crate::SolidOwner::Part(part);
        let edge = graph.evaluated_solid(owner).unwrap().logical_edges[0].key;
        let feature = crate::ShapeFeature::new(
            [crate::EdgeChainRef { owner, edge }],
            crate::EdgeTreatment::Chamfer,
            10,
        );
        let BuildOutcome::ShapeFeatureAdded(id) =
            graph.apply(BuildCommand::AddShapeFeature(feature)).unwrap()
        else {
            unreachable!()
        };
        let chamfered_volume = graph.evaluated_solid(owner).unwrap().volume();
        assert!(chamfered_volume < 0.5_f64.powi(3));

        graph
            .apply(BuildCommand::SetShapeFeatureAmount {
                feature: id,
                amount_ticks: 20,
            })
            .unwrap();
        assert!(graph.evaluated_solid(owner).unwrap().volume() < chamfered_volume);

        let snapshot = graph.evaluated_solid(owner).unwrap();
        assert!(matches!(
            graph.apply(BuildCommand::SetShapeFeatureAmount {
                feature: id,
                amount_ticks: 0,
            }),
            Err(GraphError::InvalidSolid(crate::SolidError::ZeroAmount))
        ));
        assert_eq!(graph.evaluated_solid(owner).unwrap(), snapshot);

        graph.apply(BuildCommand::RemoveShapeFeature(id)).unwrap();
        assert!(!graph.owner_has_shape_features(owner));
        assert!((graph.evaluated_solid(owner).unwrap().volume() - 0.5_f64.powi(3)).abs() < 1.0e-8);
    }

    #[test]
    fn five_centimetre_fillet_revalidates_a_welded_tangent_chain() {
        let mut graph = ConstructionGraph::new();
        let first = spawn(
            &mut graph,
            CuboidSpec::new(
                [1, 1, 1],
                BuildPose::new(IVec3::ZERO, GridRotation::default()),
            )
            .unwrap(),
        );
        let second = spawn(
            &mut graph,
            CuboidSpec::new([1, 1, 1], BuildPose::new(IVec3::X, GridRotation::default())).unwrap(),
        );
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(first, FaceKind::PositiveX),
                second: FaceRef::part(second, FaceKind::NegativeX),
            }))
            .unwrap();

        let target_for = |part| {
            let owner = crate::SolidOwner::Part(part);
            let solid = graph.evaluated_solid(owner).unwrap();
            let edge = solid
                .logical_edges
                .iter()
                .find(|logical| {
                    let half_edge = solid.half_edges[logical.half_edges[0] as usize];
                    let twin = solid.half_edges[half_edge.twin as usize];
                    let patches = [
                        solid.surfaces[half_edge.face as usize].key.local,
                        solid.surfaces[twin.face as usize].key.local,
                    ];
                    patches.contains(&3) && patches.contains(&4)
                })
                .expect("the positive-Y/negative-Z edge exists")
                .key;
            crate::EdgeChainRef { owner, edge }
        };
        let targets = [target_for(first), target_for(second)];
        let BuildOutcome::ShapeFeatureAdded(feature) = graph
            .apply(BuildCommand::AddShapeFeature(crate::ShapeFeature::new(
                targets,
                crate::EdgeTreatment::Fillet,
                20,
            )))
            .unwrap()
        else {
            unreachable!()
        };

        assert_eq!(graph.shape_feature(feature).unwrap().amount_ticks, 20);
        graph
            .apply(BuildCommand::SetShapeFeatureAmount {
                feature,
                amount_ticks: 100,
            })
            .unwrap();
        assert!(matches!(
            graph.apply(BuildCommand::SetShapeFeatureAmount {
                feature,
                amount_ticks: 120,
            }),
            Err(GraphError::InvalidSolid(crate::SolidError::AmountTooLarge(id))) if id == feature
        ));
        assert_eq!(graph.shape_feature(feature).unwrap().amount_ticks, 100);
    }

    #[test]
    fn welded_block_solid_accepts_treatments_past_one_block() {
        let mut graph = ConstructionGraph::new();
        let mut parts = std::collections::BTreeMap::new();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let part = spawn(
                        &mut graph,
                        CuboidSpec::new(
                            [1, 1, 1],
                            BuildPose::new(IVec3::new(x, y, z), GridRotation::default()),
                        )
                        .unwrap(),
                    );
                    parts.insert([x, y, z], part);
                }
            }
        }
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let cell = IVec3::new(x, y, z);
                    for (axis, positive, negative) in [
                        (0, FaceKind::PositiveX, FaceKind::NegativeX),
                        (1, FaceKind::PositiveY, FaceKind::NegativeY),
                        (2, FaceKind::PositiveZ, FaceKind::NegativeZ),
                    ] {
                        let mut neighbour = cell;
                        neighbour[axis] += 1;
                        if let Some(&next) = parts.get(&neighbour.to_array()) {
                            graph
                                .apply(BuildCommand::Weld(WeldSpec {
                                    first: FaceRef::part(parts[&cell.to_array()], positive),
                                    second: FaceRef::part(next, negative),
                                }))
                                .unwrap();
                        }
                    }
                }
            }
        }

        let targets = (0..2)
            .map(|z| {
                let part = parts[&[1, 1, z]];
                let owner = crate::SolidOwner::Part(part);
                let solid = graph.evaluated_solid(owner).unwrap();
                let edge = solid
                    .logical_edges
                    .iter()
                    .find(|logical| {
                        let half_edge = solid.half_edges[logical.half_edges[0] as usize];
                        let twin = solid.half_edges[half_edge.twin as usize];
                        let patches = [
                            solid.surfaces[half_edge.face as usize].key.local,
                            solid.surfaces[twin.face as usize].key.local,
                        ];
                        patches.contains(&1) && patches.contains(&3)
                    })
                    .expect("the positive-X/positive-Y edge exists")
                    .key;
                crate::EdgeChainRef { owner, edge }
            })
            .collect::<Vec<_>>();

        for treatment in [crate::EdgeTreatment::Fillet, crate::EdgeTreatment::Chamfer] {
            let mut candidate = graph.clone();
            let BuildOutcome::ShapeFeatureAdded(feature) = candidate
                .apply(BuildCommand::AddShapeFeature(crate::ShapeFeature::new(
                    targets.clone(),
                    treatment,
                    180,
                )))
                .unwrap_or_else(|error| panic!("45 cm {treatment:?} failed: {error}"))
            else {
                unreachable!()
            };
            let owner = candidate.shape_feature(feature).unwrap().targets[0].owner;
            assert!(matches!(owner, crate::SolidOwner::Region(_)));
            assert!(candidate.evaluated_solid(owner).unwrap().volume() < 0.125);
            candidate.compile().unwrap();
        }
    }

    fn grounded_filleted_cylinder() -> (ConstructionGraph, PartId, CylinderSpec) {
        let mut graph = ConstructionGraph::new();
        let spec = CylinderSpec::new(
            CylinderDimensions::new(1.0, 0.0, 0.5).unwrap(),
            BuildPose::from_position_ticks(IVec3::new(0, 100, 0), GridRotation::default()),
        );
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::SpawnCylinder(spec)).unwrap()
        else {
            panic!("spawning a cylinder reports its part");
        };
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(part, FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();
        let owner = crate::SolidOwner::Part(part);
        let rim = graph
            .evaluated_solid(owner)
            .unwrap()
            .logical_edges
            .iter()
            .find(|edge| edge.closed && edge.convex)
            .unwrap()
            .key;
        graph
            .apply(BuildCommand::AddShapeFeature(crate::ShapeFeature::new(
                [crate::EdgeChainRef { owner, edge: rim }],
                crate::EdgeTreatment::Fillet,
                120,
            )))
            .unwrap();
        (graph, part, spec)
    }

    #[test]
    fn layering_a_cylinder_keeps_its_welds_and_features() {
        let (mut graph, part, spec) = grounded_filleted_cylinder();
        let layered = spec
            .with_layer(
                crate::LayerSide::Outer,
                0.25,
                ConstructionMaterial::Rubber,
                MaterialAppearance::BAKED,
            )
            .unwrap();
        assert_eq!(
            graph.apply(BuildCommand::SetCylinder {
                part,
                spec: layered
            }),
            Ok(BuildOutcome::CylinderUpdated)
        );
        assert_eq!(graph.part(part), Some(&PartSpec::Cylinder(layered)));
        assert_eq!(graph.weld_count(), 1);
        let solid = graph
            .evaluated_solid(crate::SolidOwner::Part(part))
            .unwrap();
        assert!(solid.surfaces.iter().any(|surface| surface.band == 0));
        assert!(solid.surfaces.iter().any(|surface| surface.band == 1));
    }

    #[test]
    fn a_cylinder_reshape_that_breaks_its_features_is_rejected() {
        let (mut graph, part, spec) = grounded_filleted_cylinder();
        let narrow = CylinderSpec::new(CylinderDimensions::new(0.5, 0.0, 0.5).unwrap(), spec.pose);
        assert!(
            graph
                .apply(BuildCommand::SetCylinder { part, spec: narrow })
                .is_err()
        );
        assert_eq!(graph.part(part), Some(&PartSpec::Cylinder(spec)));
        let moved = CylinderSpec::new(spec.dimensions, BuildPose::default());
        assert_eq!(
            graph.apply(BuildCommand::SetCylinder { part, spec: moved }),
            Err(GraphError::CylinderMoved(part))
        );
    }

    #[test]
    fn painting_one_cylinder_band_leaves_the_others() {
        let (mut graph, part, spec) = grounded_filleted_cylinder();
        let layered = spec
            .with_layer(
                crate::LayerSide::Outer,
                0.25,
                ConstructionMaterial::Rubber,
                MaterialAppearance::BAKED,
            )
            .unwrap();
        graph
            .apply(BuildCommand::SetCylinder {
                part,
                spec: layered,
            })
            .unwrap();
        let paint = MaterialAppearance {
            finish: MaterialFinish::Painted,
            ..MaterialAppearance::BAKED
        };
        graph
            .apply(BuildCommand::SetAppearance {
                target: AppearanceTarget::CylinderBand { part, band: 0 },
                appearance: paint,
            })
            .unwrap();
        let Some(PartSpec::Cylinder(painted)) = graph.part(part).copied() else {
            panic!("the part is still a cylinder");
        };
        assert_eq!(painted.band(0).unwrap().appearance, paint);
        assert_eq!(painted.appearance, MaterialAppearance::BAKED);
        assert_eq!(
            graph.apply(BuildCommand::SetAppearance {
                target: AppearanceTarget::CylinderBand { part, band: 2 },
                appearance: paint,
            }),
            Err(GraphError::CylinderBands(
                crate::CylinderDimensionError::BandOutOfRange
            ))
        );
    }
}
