use std::collections::BTreeSet;

use bevy_math::{Vec2, Vec3};
use thiserror::Error;

use crate::{
    ANCHOR_TOLERANCE_METERS, AXIS_TOLERANCE_DEGREES, BearingId, ControllerSpec, CuboidSpec,
    CylinderSpec, DriveLimits, DriveLinkId, DriveName, DriveProgram, DriveTarget, FaceKind,
    FaceOwner, FaceRef, PartId, PartSpec, RigidLinkId, WeldId,
    geometry::{FaceGeometry, FaceProfile, cuboid_face, cylinder_face, ground_face},
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
    /// Control-block part this wire belongs to.
    pub controller: PartId,
    /// Bearing driven through this wire.
    pub bearing: BearingId,
    /// Whether this bearing runs opposite the programmed direction.
    pub reversed: bool,
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
            reversed: false,
            limits: DriveLimits::default(),
            program: DriveProgram::default(),
            name: DriveName::EMPTY,
        }
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

/// One-degree-of-freedom bearing between two faces. It is passive unless a
/// control block is wired to it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BearingSpec {
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

/// Atomic edit request for a construction graph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BuildCommand {
    /// Spawn a standalone cuboid.
    Spawn(CuboidSpec),
    /// Spawn a standalone solid or hollow cylinder.
    SpawnCylinder(CylinderSpec),
    /// Remove a part and every connection referencing it.
    Remove(PartId),
    /// Remove one weld while leaving its endpoint parts intact.
    RemoveWeld(WeldId),
    /// Remove one non-geometric rigid link.
    RemoveRigidLink(RigidLinkId),
    /// Remove one bearing while leaving its endpoint parts intact.
    RemoveBearing(BearingId),
    /// Merge the groups containing two touching faces.
    Weld(WeldSpec),
    /// Merge two parts rigidly without requiring face contact.
    RigidLink(RigidLinkSpec),
    /// Spawn a control block.
    SpawnController(ControllerSpec),
    /// Add a passive bearing.
    AddBearing(BearingSpec),
    /// Wire a control block to one bearing.
    AddDriveLink(DriveLinkSpec),
    /// Remove one control-block wire, leaving its endpoints intact.
    RemoveDriveLink(DriveLinkId),
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
    },
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
    /// A drive wire's limits or program were replaced.
    DriveUpdated,
    /// A pending operation was recorded.
    Pending,
    /// A pending operation was cancelled, or there was nothing to cancel.
    Cancelled,
}

/// Validation failure. Failed commands leave the graph byte-for-byte equivalent.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum GraphError {
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
    /// A part referenced as a control block is a different kind of part.
    #[error("part {0:?} is not a control block")]
    NotAController(PartId),
    /// A bearing already obeys another control block.
    #[error("bearing {0:?} is already driven by a control block")]
    BearingAlreadyDriven(BearingId),
    /// Only the positive-y ground face exists.
    #[error("the ground only exposes its positive-y face")]
    InvalidGroundFace,
    /// Cylinders expose only their two flat local-Y ends as connection faces.
    #[error("cylinders expose only their positive-y and negative-y flat ends")]
    InvalidCylinderFace,
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
}

/// Editable, CPU-owned construction topology.
#[derive(Clone, Debug, Default)]
pub struct ConstructionGraph {
    pub(crate) parts: Arena<PartSpec, PartId>,
    pub(crate) welds: Arena<WeldSpec, WeldId>,
    pub(crate) rigid_links: Arena<RigidLinkSpec, RigidLinkId>,
    pub(crate) bearings: Arena<BearingSpec, BearingId>,
    pub(crate) drive_links: Arena<DriveLinkSpec, DriveLinkId>,
    pending: Option<PendingOperation>,
}

impl ConstructionGraph {
    /// Creates an empty graph in paused build mode.
    pub fn new() -> Self {
        Self::default()
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

    /// Whether a live part is a control block.
    pub fn is_controller(&self, part: PartId) -> bool {
        self.parts
            .get(part)
            .is_some_and(|spec| spec.as_controller().is_some())
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

    /// Current incomplete two-step operation, if any.
    pub const fn pending(&self) -> Option<PendingOperation> {
        self.pending
    }

    /// Number of live parts.
    pub const fn part_count(&self) -> usize {
        self.parts.len()
    }

    /// Number of live welds.
    pub const fn weld_count(&self) -> usize {
        self.welds.len()
    }

    /// Number of live non-geometric rigid links.
    pub const fn rigid_link_count(&self) -> usize {
        self.rigid_links.len()
    }

    /// Number of live bearings.
    pub const fn bearing_count(&self) -> usize {
        self.bearings.len()
    }

    /// Number of live control-block wires.
    pub const fn drive_link_count(&self) -> usize {
        self.drive_links.len()
    }

    pub(crate) fn face_geometry(&self, face: FaceRef) -> Result<FaceGeometry, GraphError> {
        match face.owner {
            FaceOwner::Part(part) => match self.parts.get(part).copied() {
                Some(PartSpec::Cuboid(spec)) => Ok(cuboid_face(spec, face.face)),
                Some(PartSpec::Controller(spec)) => Ok(cuboid_face(spec.cuboid(), face.face)),
                Some(PartSpec::Cylinder(spec)) => {
                    cylinder_face(spec, face.face).ok_or(GraphError::InvalidCylinderFace)
                }
                None => Err(GraphError::MissingPart(part)),
            },
            FaceOwner::Ground if face.face == FaceKind::PositiveY => Ok(ground_face()),
            FaceOwner::Ground => Err(GraphError::InvalidGroundFace),
        }
    }

    #[allow(clippy::too_many_lines)] // One exhaustive command dispatch reads better whole.
    fn apply_validated(&mut self, command: BuildCommand) -> Result<BuildOutcome, GraphError> {
        match command {
            BuildCommand::Spawn(spec) => {
                let id = self.parts.insert(spec.into());
                Ok(BuildOutcome::Spawned(id))
            }
            BuildCommand::SpawnCylinder(spec) => {
                let id = self.parts.insert(spec.into());
                Ok(BuildOutcome::Spawned(id))
            }
            BuildCommand::SpawnController(spec) => {
                let id = self.parts.insert(spec.into());
                Ok(BuildOutcome::Spawned(id))
            }
            BuildCommand::Remove(id) => {
                self.parts.get(id).ok_or(GraphError::MissingPart(id))?;
                let welds = self
                    .welds
                    .iter()
                    .filter_map(|(weld_id, weld)| weld_references(*weld, id).then_some(weld_id))
                    .collect::<Vec<_>>();
                let bearings = self
                    .bearings
                    .iter()
                    .filter_map(|(bearing_id, bearing)| {
                        bearing_references(*bearing, id).then_some(bearing_id)
                    })
                    .collect::<Vec<_>>();
                let rigid_links = self
                    .rigid_links
                    .iter()
                    .filter_map(|(link_id, link)| {
                        rigid_link_references(*link, id).then_some(link_id)
                    })
                    .collect::<Vec<_>>();
                let removed_bearings = bearings.iter().copied().collect::<BTreeSet<_>>();
                let drive_links = self
                    .drive_links
                    .iter()
                    .filter_map(|(link_id, link)| {
                        (link.controller == id || removed_bearings.contains(&link.bearing))
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
                for bearing in bearings {
                    self.bearings.remove(bearing);
                }
                self.parts.remove(id);
                self.pending = None;
                Ok(BuildOutcome::Removed)
            }
            BuildCommand::RemoveWeld(id) => {
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
            BuildCommand::AddDriveLink(spec) => {
                self.validate_drive_link(spec)?;
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
            BuildCommand::SetDriveLink {
                link,
                limits,
                program,
                name,
            } => {
                let spec = self
                    .drive_links
                    .get_mut(link)
                    .ok_or(GraphError::MissingDriveLink(link))?;
                spec.limits = limits;
                spec.program = program;
                spec.name = name;
                Ok(BuildOutcome::DriveUpdated)
            }
            BuildCommand::BeginPending(pending) => {
                match pending {
                    PendingOperation::Weld(face) => {
                        self.face_geometry(face)?;
                    }
                    PendingOperation::Bearing { source, anchor } => {
                        let geometry = self.face_geometry(source)?;
                        if !anchor.is_finite() || !point_on_face(anchor, geometry) {
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
        let first = self.face_geometry(spec.first)?;
        let second = self.face_geometry(spec.second)?;
        if faces_touch(first, second) {
            Ok(())
        } else {
            Err(GraphError::FacesDoNotTouch)
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

    fn validate_drive_link(&self, spec: DriveLinkSpec) -> Result<(), GraphError> {
        self.parts
            .get(spec.controller)
            .copied()
            .ok_or(GraphError::MissingPart(spec.controller))?
            .as_controller()
            .ok_or(GraphError::NotAController(spec.controller))?;
        self.bearings
            .get(spec.bearing)
            .ok_or(GraphError::MissingBearing(spec.bearing))?;
        if self
            .drive_links
            .iter()
            .any(|(_, link)| link.bearing == spec.bearing)
        {
            return Err(GraphError::BearingAlreadyDriven(spec.bearing));
        }
        Ok(())
    }

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
        if source.normal.dot(target.normal) > -1.0 + axis_cosine_tolerance() {
            return Err(GraphError::BearingFacesNotOpposed);
        }
        if !spec.shared_anchor.is_finite()
            || !bearing_ring_overlaps_face(spec.shared_anchor, spec.dimensions, source)
            || !bearing_ring_overlaps_face(spec.shared_anchor, spec.dimensions, target)
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

fn weld_references(weld: WeldSpec, part: PartId) -> bool {
    face_references(weld.first, part) || face_references(weld.second, part)
}

fn rigid_link_references(link: RigidLinkSpec, part: PartId) -> bool {
    link.first == part || link.second == part
}

fn bearing_references(bearing: BearingSpec, part: PartId) -> bool {
    face_references(bearing.source, part) || face_references(bearing.target, part)
}

fn point_on_face(point: Vec3, face: FaceGeometry) -> bool {
    let offset = point - face.center;
    if offset.dot(face.normal).abs() > ANCHOR_TOLERANCE_METERS {
        return false;
    }
    point_in_profile(
        offset.dot(face.tangent_u),
        offset.dot(face.tangent_v),
        face.profile,
    )
}

fn bearing_ring_overlaps_face(
    anchor: Vec3,
    dimensions: BearingDimensions,
    face: FaceGeometry,
) -> bool {
    let offset = anchor - face.center;
    if offset.dot(face.normal).abs() > ANCHOR_TOLERANCE_METERS
        || matches!(face.profile, FaceProfile::Ground)
    {
        return false;
    }
    profiles_overlap(
        FaceGeometry {
            center: anchor,
            normal: face.normal,
            tangent_u: face.tangent_u,
            tangent_v: face.tangent_v,
            profile: FaceProfile::Annulus {
                inner_radius: dimensions.inner_diameter() * 0.5,
                outer_radius: dimensions.outer_diameter() * 0.5,
            },
        },
        face,
    )
}

fn faces_touch(first: FaceGeometry, second: FaceGeometry) -> bool {
    if first.normal.dot(second.normal) > -1.0 + axis_cosine_tolerance() {
        return false;
    }
    let separation = (second.center - first.center).dot(first.normal).abs();
    if separation > ANCHOR_TOLERANCE_METERS {
        return false;
    }
    if matches!(first.profile, FaceProfile::Ground) || matches!(second.profile, FaceProfile::Ground)
    {
        return true;
    }
    profiles_overlap(first, second)
}

fn point_in_profile(u: f32, v: f32, profile: FaceProfile) -> bool {
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
        FaceProfile::Ground => true,
    }
}

fn profiles_overlap(first: FaceGeometry, second: FaceGeometry) -> bool {
    match (first.profile, second.profile) {
        (FaceProfile::Rectangle { half_u, half_v }, FaceProfile::Rectangle { .. }) => {
            positive_rect_overlap(first, second, first.tangent_u, half_u)
                && positive_rect_overlap(first, second, first.tangent_v, half_v)
        }
        (FaceProfile::Annulus { .. }, FaceProfile::Rectangle { .. }) => {
            annulus_rectangle_overlap(first, second)
        }
        (FaceProfile::Rectangle { .. }, FaceProfile::Annulus { .. }) => {
            annulus_rectangle_overlap(second, first)
        }
        (FaceProfile::Annulus { .. }, FaceProfile::Annulus { .. }) => annuli_overlap(first, second),
        (FaceProfile::Ground, _) | (_, FaceProfile::Ground) => true,
        (FaceProfile::AnnularSector { .. }, _) | (_, FaceProfile::AnnularSector { .. }) => {
            sector_profiles_overlap(first, second)
        }
    }
}

fn sector_profiles_overlap(first: FaceGeometry, second: FaceGeometry) -> bool {
    let first_cells = profile_cells(first, first.center, first.tangent_u, first.tangent_v);
    let second_cells = profile_cells(second, first.center, first.tangent_u, first.tangent_v);
    first_cells.iter().any(|first| {
        second_cells
            .iter()
            .any(|second| convex_polygons_overlap(first, second))
    })
}

fn profile_cells(face: FaceGeometry, origin: Vec3, plane_u: Vec3, plane_v: Vec3) -> Vec<Vec<Vec2>> {
    let project = |point: Vec3| {
        let offset = point - origin;
        Vec2::new(offset.dot(plane_u), offset.dot(plane_v))
    };
    match face.profile {
        FaceProfile::Rectangle { half_u, half_v } => vec![vec![
            project(face.center - face.tangent_u * half_u - face.tangent_v * half_v),
            project(face.center + face.tangent_u * half_u - face.tangent_v * half_v),
            project(face.center + face.tangent_u * half_u + face.tangent_v * half_v),
            project(face.center - face.tangent_u * half_u + face.tangent_v * half_v),
        ]],
        FaceProfile::Annulus {
            inner_radius,
            outer_radius,
        } => annular_profile_cells(
            face,
            origin,
            plane_u,
            plane_v,
            inner_radius,
            outer_radius,
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
            inner_radius,
            outer_radius,
            half_angle,
        ),
        FaceProfile::Ground => Vec::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn annular_profile_cells(
    face: FaceGeometry,
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
    first: FaceGeometry,
    second: FaceGeometry,
    axis: Vec3,
    first_half: f32,
) -> bool {
    let FaceProfile::Rectangle { half_u, half_v } = second.profile else {
        unreachable!()
    };
    let second_half =
        second.tangent_u.dot(axis).abs() * half_u + second.tangent_v.dot(axis).abs() * half_v;
    let centre_distance = (second.center - first.center).dot(axis).abs();
    first_half + second_half - centre_distance > ANCHOR_TOLERANCE_METERS
}

fn annulus_rectangle_overlap(annulus: FaceGeometry, rectangle: FaceGeometry) -> bool {
    let FaceProfile::Annulus {
        inner_radius,
        outer_radius,
    } = annulus.profile
    else {
        unreachable!()
    };
    let FaceProfile::Rectangle { half_u, half_v } = rectangle.profile else {
        unreachable!()
    };
    let offset = annulus.center - rectangle.center;
    let center_u = offset.dot(rectangle.tangent_u).abs();
    let center_v = offset.dot(rectangle.tangent_v).abs();
    let nearest_u = (center_u - half_u).max(0.0);
    let nearest_v = (center_v - half_v).max(0.0);
    let nearest_squared = nearest_u.mul_add(nearest_u, nearest_v * nearest_v);
    let farthest_u = center_u + half_u;
    let farthest_v = center_v + half_v;
    let farthest_squared = farthest_u.mul_add(farthest_u, farthest_v * farthest_v);
    nearest_squared < (outer_radius - ANCHOR_TOLERANCE_METERS).max(0.0).powi(2)
        && farthest_squared > (inner_radius + ANCHOR_TOLERANCE_METERS).powi(2)
}

fn annuli_overlap(first: FaceGeometry, second: FaceGeometry) -> bool {
    let FaceProfile::Annulus {
        inner_radius: inner_a,
        outer_radius: outer_a,
    } = first.profile
    else {
        unreachable!()
    };
    let FaceProfile::Annulus {
        inner_radius: inner_b,
        outer_radius: outer_b,
    } = second.profile
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
    distance < outer_a + outer_b - ANCHOR_TOLERANCE_METERS
        && distance + outer_a > inner_b + ANCHOR_TOLERANCE_METERS
        && distance + outer_b > inner_a + ANCHOR_TOLERANCE_METERS
}

fn axis_cosine_tolerance() -> f32 {
    1.0 - AXIS_TOLERANCE_DEGREES.to_radians().cos()
}

#[cfg(test)]
mod tests {
    use bevy_math::{IVec3, Vec3};

    use super::{
        BearingDimensionError, BearingDimensions, BearingSpec, BuildCommand, BuildOutcome,
        ConstructionGraph, DriveLinkSpec, GraphError, PendingOperation, RigidLinkSpec, WeldSpec,
    };
    use crate::{
        BearingId, BuildPose, ControllerSpec, CuboidSpec, CylinderDimensions, CylinderSpec,
        DriveLimits, DriveName, DriveProgram, DriveState, DriveTarget, FaceKind, FaceRef,
        GridRotation, PartId, PartSpec,
    };

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
}
