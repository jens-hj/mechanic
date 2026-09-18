//! Why the graph refused a command.

use crate::{
    BearingId, DimensionLinkId, DriveLinkId, EngineKind, GearboxError, InputSeatLinkId, PartId,
    RegionError, RegionId, RigidLinkId, SeatControllerLinkId, ShapeFeatureId, SolidError,
    SolidOwner, WeldId,
};
use thiserror::Error;

/// Validation failure. Failed commands leave the graph byte-for-byte equivalent.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum GraphError {
    /// A material layer does not fit its part.
    #[error(transparent)]
    Layer(#[from] crate::LayerError),
    /// A layer edit changed more of a part than its material layers.
    #[error("part {0:?} can only change its material layers")]
    LayerCoreChanged(PartId),
    /// Material layers and shaped regions cannot be combined.
    #[error("part {0:?} cannot combine material layers with a shaped region")]
    LayeredPartInRegion(PartId),
    /// An appearance edit named a material band the part does not have.
    #[error("part {0:?} has no material band {1}")]
    MissingBand(PartId, u8),
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
