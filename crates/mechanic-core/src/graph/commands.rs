//! The build commands a graph accepts, their outcomes, and what is pending.

use super::specs::{
    BearingSpec, DriveLinkSpec, InputSeatLinkSpec, RigidLinkSpec, SeatControllerLinkSpec, WeldSpec,
};
use crate::{
    ActuatorAssignment, BearingId, CageIndex, ControllerSpec, CuboidSpec, CylinderSpec,
    DimensionLinkSpec, DriveLimits, DriveLinkId, DriveName, DriveProgram, EngineKind, EngineSpec,
    FaceRef, GearKeyChord, InputSeatLinkId, InputSpec, MaterialAppearance, PartId, PartSpec,
    PipeBendSpec, PipeJunctionSpec, RegionId, RigidLinkId, SeatControllerLinkId, SeatSpec,
    ServoSpec, ShapeFeature, ShapeFeatureId, ShapeRegion, ShiftMode, TransmissionSpec, WeldId,
};
use bevy_math::Vec3;

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
    /// One material band of a layered part: zero is the core, `i + 1` layer `i`.
    PartBand {
        /// Layered part.
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
    /// Replace a cuboid's or cylinder's material layers in place, keeping its
    /// part identity, core, features, and connections.
    SetLayers {
        /// Part being layered.
        part: PartId,
        /// Replacement sharing the part's core.
        spec: PartSpec,
    },
    /// Replace a cylinder's spiral in place, keeping its part identity and
    /// connections. The diameters may change with it: a ridge added onto a
    /// cylinder widens the envelope the spiral is drawn into.
    SetSpiral {
        /// Cylinder being cut.
        part: PartId,
        /// Replacement with the same pose, length, material, and appearance.
        spec: crate::CylinderSpec,
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
    /// A part's material layers changed.
    LayersUpdated,
    /// A cylinder's spiral changed.
    SpiralUpdated,
    /// A pending operation was recorded.
    Pending,
    /// A pending operation was cancelled, or there was nothing to cancel.
    Cancelled,
}
