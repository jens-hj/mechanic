use bevy_math::Vec3;
use thiserror::Error;

use crate::{
    ANCHOR_TOLERANCE_METERS, AXIS_TOLERANCE_DEGREES, BearingId, CuboidSpec, FaceKind, FaceOwner,
    FaceRef, PartId, WeldId,
    geometry::{FaceGeometry, cuboid_face, ground_face},
    id::Arena,
};

/// Explicit weld between two touching faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeldSpec {
    /// First selected face.
    pub first: FaceRef,
    /// Second selected face.
    pub second: FaceRef,
}

/// Passive one-degree-of-freedom bearing between two faces.
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
}

impl BearingSpec {
    /// Creates a bearing specification. Geometry is validated on insertion.
    pub const fn new(source: FaceRef, target: FaceRef, shared_anchor: Vec3, axis: Vec3) -> Self {
        Self {
            source,
            target,
            shared_anchor,
            axis,
        }
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
}

/// Atomic edit request for a construction graph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BuildCommand {
    /// Spawn a standalone cuboid.
    Spawn(CuboidSpec),
    /// Remove a cuboid and every connection referencing it.
    Remove(PartId),
    /// Remove one weld while leaving its endpoint parts intact.
    RemoveWeld(WeldId),
    /// Remove one bearing while leaving its endpoint parts intact.
    RemoveBearing(BearingId),
    /// Merge the groups containing two touching faces.
    Weld(WeldSpec),
    /// Add a passive bearing.
    AddBearing(BearingSpec),
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
    /// A bearing was created.
    BearingAdded(BearingId),
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
    /// A bearing handle is stale or unknown.
    #[error("unknown or stale bearing handle {0:?}")]
    MissingBearing(BearingId),
    /// Only the positive-y ground face exists.
    #[error("the ground only exposes its positive-y face")]
    InvalidGroundFace,
    /// A connection selected the same endpoint twice.
    #[error("a connection requires two distinct faces")]
    SameFace,
    /// The selected weld faces are not coplanar, opposed, and overlapping.
    #[error("weld faces do not touch over a positive area")]
    FacesDoNotTouch,
    /// Bearing faces do not have opposite normals.
    #[error("bearing endpoint faces are not opposed")]
    BearingFacesNotOpposed,
    /// Bearing anchor does not lie on both selected faces.
    #[error("bearing anchor does not lie on both endpoint faces")]
    BearingAnchorOutsideFaces,
    /// Stored bearing axis is not a finite unit source-face normal.
    #[error("bearing axis must be finite, unit length, and equal the source-face normal")]
    InvalidBearingAxis,
    /// A bearing cannot connect a face to the ground in this milestone.
    #[error("bearings require two cuboid endpoints")]
    BearingOnGround,
}

/// Editable, CPU-owned construction topology.
#[derive(Clone, Debug, Default)]
pub struct ConstructionGraph {
    pub(crate) parts: Arena<CuboidSpec, PartId>,
    pub(crate) welds: Arena<WeldSpec, WeldId>,
    pub(crate) bearings: Arena<BearingSpec, BearingId>,
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

    /// Retrieves a live cuboid.
    pub fn part(&self, id: PartId) -> Option<&CuboidSpec> {
        self.parts.get(id)
    }

    /// Retrieves a live weld.
    pub fn weld(&self, id: WeldId) -> Option<&WeldSpec> {
        self.welds.get(id)
    }

    /// Retrieves a live bearing.
    pub fn bearing(&self, id: BearingId) -> Option<&BearingSpec> {
        self.bearings.get(id)
    }

    /// Iterates live cuboids in canonical slot order.
    pub fn parts(&self) -> impl Iterator<Item = (PartId, &CuboidSpec)> {
        self.parts.iter()
    }

    /// Iterates live welds in canonical slot order.
    pub fn welds(&self) -> impl Iterator<Item = (WeldId, &WeldSpec)> {
        self.welds.iter()
    }

    /// Iterates live bearings in canonical slot order.
    pub fn bearings(&self) -> impl Iterator<Item = (BearingId, &BearingSpec)> {
        self.bearings.iter()
    }

    /// Current incomplete two-step operation, if any.
    pub const fn pending(&self) -> Option<PendingOperation> {
        self.pending
    }

    /// Number of live cuboids.
    pub const fn part_count(&self) -> usize {
        self.parts.len()
    }

    /// Number of live welds.
    pub const fn weld_count(&self) -> usize {
        self.welds.len()
    }

    /// Number of live bearings.
    pub const fn bearing_count(&self) -> usize {
        self.bearings.len()
    }

    pub(crate) fn face_geometry(&self, face: FaceRef) -> Result<FaceGeometry, GraphError> {
        match face.owner {
            FaceOwner::Part(part) => self
                .parts
                .get(part)
                .copied()
                .map(|spec| cuboid_face(spec, face.face))
                .ok_or(GraphError::MissingPart(part)),
            FaceOwner::Ground if face.face == FaceKind::PositiveY => Ok(ground_face()),
            FaceOwner::Ground => Err(GraphError::InvalidGroundFace),
        }
    }

    fn apply_validated(&mut self, command: BuildCommand) -> Result<BuildOutcome, GraphError> {
        match command {
            BuildCommand::Spawn(spec) => {
                let id = self.parts.insert(spec);
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
                for weld in welds {
                    self.welds.remove(weld);
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
            BuildCommand::RemoveBearing(id) => {
                self.bearings
                    .remove(id)
                    .ok_or(GraphError::MissingBearing(id))?;
                self.pending = None;
                Ok(BuildOutcome::Removed)
            }
            BuildCommand::Weld(spec) => {
                self.validate_weld(spec)?;
                let id = self.welds.insert(spec);
                self.pending = None;
                Ok(BuildOutcome::Welded(id))
            }
            BuildCommand::AddBearing(spec) => {
                self.validate_bearing(spec)?;
                let id = self.bearings.insert(spec);
                self.pending = None;
                Ok(BuildOutcome::BearingAdded(id))
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
            || !point_on_face(spec.shared_anchor, source)
            || !point_on_face(spec.shared_anchor, target)
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

fn bearing_references(bearing: BearingSpec, part: PartId) -> bool {
    face_references(bearing.source, part) || face_references(bearing.target, part)
}

fn point_on_face(point: Vec3, face: FaceGeometry) -> bool {
    let offset = point - face.center;
    if offset.dot(face.normal).abs() > ANCHOR_TOLERANCE_METERS {
        return false;
    }
    face.infinite
        || (offset.dot(face.tangent_u).abs() <= face.half_u + ANCHOR_TOLERANCE_METERS
            && offset.dot(face.tangent_v).abs() <= face.half_v + ANCHOR_TOLERANCE_METERS)
}

fn faces_touch(first: FaceGeometry, second: FaceGeometry) -> bool {
    if first.normal.dot(second.normal) > -1.0 + axis_cosine_tolerance() {
        return false;
    }
    let separation = (second.center - first.center).dot(first.normal).abs();
    if separation > ANCHOR_TOLERANCE_METERS {
        return false;
    }
    if first.infinite || second.infinite {
        return true;
    }

    positive_overlap(first, second, first.tangent_u, first.half_u)
        && positive_overlap(first, second, first.tangent_v, first.half_v)
}

fn positive_overlap(
    first: FaceGeometry,
    second: FaceGeometry,
    axis: Vec3,
    first_half: f32,
) -> bool {
    let second_half = second.tangent_u.dot(axis).abs() * second.half_u
        + second.tangent_v.dot(axis).abs() * second.half_v;
    let centre_distance = (second.center - first.center).dot(axis).abs();
    first_half + second_half - centre_distance > ANCHOR_TOLERANCE_METERS
}

fn axis_cosine_tolerance() -> f32 {
    1.0 - AXIS_TOLERANCE_DEGREES.to_radians().cos()
}

#[cfg(test)]
mod tests {
    use bevy_math::{IVec3, Vec3};

    use super::{BearingSpec, BuildCommand, BuildOutcome, ConstructionGraph, GraphError, WeldSpec};
    use crate::{BuildPose, CuboidSpec, FaceKind, FaceRef, GridRotation};

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
}
