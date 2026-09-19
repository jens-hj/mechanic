//! Rigid authored frames keep exact local grids independent of assembly orientation.

use std::collections::{BTreeMap, BTreeSet};

use bevy_math::{Quat, Vec3};
use thiserror::Error;

use crate::{BearingId, ConstructionGraph, FaceOwner, PartId, RegionId, SolidOwner};

/// Stable graph-owned identity of an authored construction frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstructionFrameId(pub u64);

/// A validated rigid transform from a local construction grid into build space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstructionFrame {
    translation: Vec3,
    rotation: Quat,
}

impl ConstructionFrame {
    /// The default build grid.
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    };

    /// Creates a finite rigid frame with a unit quaternion.
    ///
    /// # Errors
    /// Rejects nonfinite coordinates and nonunit rotations.
    pub fn new(translation: Vec3, rotation: Quat) -> Result<Self, FrameError> {
        if !translation.is_finite()
            || !rotation.is_finite()
            || (rotation.length_squared() - 1.0).abs() > 1.0e-4
        {
            return Err(FrameError::InvalidTransform);
        }
        Ok(Self {
            translation,
            rotation: rotation.normalize(),
        })
    }

    /// Translation in build-space metres.
    pub const fn translation(self) -> Vec3 {
        self.translation
    }
    /// Orientation of the local grid in build space.
    pub const fn rotation(self) -> Quat {
        self.rotation
    }
    /// Transforms a local point into build space.
    pub fn point(self, point: Vec3) -> Vec3 {
        self.translation + self.rotation * point
    }
    /// Transforms a direction without translation.
    pub fn vector(self, vector: Vec3) -> Vec3 {
        self.rotation * vector
    }
    /// Inverse mapping from build space to this grid.
    #[must_use]
    pub fn inverse(self) -> Self {
        let rotation = self.rotation.conjugate();
        Self {
            translation: rotation * -self.translation,
            rotation,
        }
    }
    /// Applies `local` followed by this transform.
    #[must_use]
    pub fn compose(self, local: Self) -> Self {
        Self {
            translation: self.point(local.translation),
            rotation: (self.rotation * local.rotation).normalize(),
        }
    }

    pub(crate) fn transform_solid(self, solid: &mut crate::EvaluatedSolid) {
        if self == Self::IDENTITY {
            return;
        }
        for vertex in &mut solid.vertices {
            vertex.position = self.point(vertex.position);
        }
        for surface in &mut solid.surfaces {
            surface.normal = self.vector(surface.normal);
        }
        for cell in &mut solid.cells {
            let piece = &mut cell.piece;
            for vertex in &mut piece.vertices {
                *vertex = self.point(*vertex);
            }
            for face in &mut piece.faces {
                face.normal = self.vector(face.normal);
                face.offset += face.normal.dot(self.translation);
            }
            for direction in &mut piece.edge_directions {
                *direction = self.vector(*direction);
            }
            piece.centroid = self.point(piece.centroid);
        }
    }
}

/// Invalid construction-frame mutation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum FrameError {
    /// Transform data must describe a finite rigid transform.
    #[error("construction frame requires finite translation and a unit quaternion")]
    InvalidTransform,
    /// The supplied part no longer exists.
    #[error("missing construction part {0:?}")]
    MissingPart(PartId),
    /// The supplied frame no longer exists.
    #[error("missing construction frame {0:?}")]
    MissingFrame(ConstructionFrameId),
    /// A shape region cannot straddle independently moving authored grids.
    #[error("reframing must include all parts of shape region {0:?}")]
    PartialRegion(RegionId),
    /// Connected bearings must be reframed as a complete mechanism.
    #[error("reframing must include both sides of bearing {0:?}")]
    PartialBearing(BearingId),
    /// A replacement cannot be its own ancestor.
    #[error("construction edit ancestry cannot contain a cycle")]
    CyclicAncestry,
    /// Frame identities must not wrap or be reused.
    #[error("construction frame identities exhausted")]
    Exhausted,
}

#[derive(Clone, Debug)]
pub(crate) struct ConstructionFrames {
    pub(crate) frames: BTreeMap<ConstructionFrameId, ConstructionFrame>,
    pub(crate) members: BTreeMap<PartId, ConstructionFrameId>,
    pub(crate) region_frames: BTreeMap<RegionId, ConstructionFrameId>,
    pub(crate) editing_frame: ConstructionFrameId,
    view_to_build: ConstructionFrame,
    origins: BTreeMap<PartId, PartId>,
    next_id: u64,
}

impl Default for ConstructionFrames {
    fn default() -> Self {
        Self {
            frames: BTreeMap::from([(ConstructionFrameId(0), ConstructionFrame::IDENTITY)]),
            members: BTreeMap::new(),
            region_frames: BTreeMap::new(),
            editing_frame: ConstructionFrameId::default(),
            view_to_build: ConstructionFrame::IDENTITY,
            origins: BTreeMap::new(),
            next_id: 1,
        }
    }
}

impl ConstructionGraph {
    pub(crate) fn face_frame(&self, face: crate::FaceRef) -> Option<ConstructionFrame> {
        match face.owner {
            FaceOwner::Ground => Some(ConstructionFrame::IDENTITY),
            FaceOwner::Part(part) => self.part_frame(part),
        }
    }
    /// Maps coordinates of this temporary tool view back into canonical build space.
    pub fn view_to_build(&self) -> ConstructionFrame {
        self.construction_frames.view_to_build
    }

    /// Returns an independent tool view whose coordinates use the selected local grid.
    /// Part poses and frame identities remain unchanged.
    ///
    /// # Errors
    /// Rejects a frame that does not belong to this graph.
    pub fn in_edit_frame(&self, frame: ConstructionFrameId) -> Result<Self, FrameError> {
        let selected = self
            .construction_frames
            .frames
            .get(&frame)
            .copied()
            .ok_or(FrameError::MissingFrame(frame))?;
        if selected == ConstructionFrame::IDENTITY && self.edit_frame_id() == frame {
            return Ok(self.clone());
        }
        let mut view = self.clone();
        let inverse = selected.inverse();
        if selected != ConstructionFrame::IDENTITY {
            for transform in view.construction_frames.frames.values_mut() {
                *transform = inverse.compose(*transform);
            }
            // The selected grid is exactly the view basis, without roundoff drift.
            view.construction_frames
                .frames
                .insert(frame, ConstructionFrame::IDENTITY);
            view.transform_bearing_space(inverse);
            view.construction_frames.view_to_build = if frame == ConstructionFrameId::default() {
                ConstructionFrame::IDENTITY
            } else {
                self.view_to_build().compose(selected)
            };
        }
        view.construction_frames.editing_frame = frame;
        Ok(view)
    }

    /// Returns canonical build-space geometry for persistence, history, and publication.
    /// An already canonical default-grid graph keeps its shared revision unchanged.
    #[must_use]
    pub fn canonicalized(&self) -> Self {
        let transform = self.view_to_build();
        if transform == ConstructionFrame::IDENTITY
            && self.edit_frame_id() == ConstructionFrameId::default()
        {
            return self.clone();
        }
        let mut canonical = self.clone();
        if transform != ConstructionFrame::IDENTITY {
            for frame in canonical.construction_frames.frames.values_mut() {
                *frame = transform.compose(*frame);
            }
            canonical.transform_bearing_space(transform);
            canonical
                .construction_frames
                .frames
                .insert(ConstructionFrameId::default(), ConstructionFrame::IDENTITY);
        }
        canonical.construction_frames.view_to_build = ConstructionFrame::IDENTITY;
        canonical.construction_frames.editing_frame = ConstructionFrameId::default();
        canonical
    }

    /// Grid inherited by newly spawned parts and newly claimed shape regions.
    pub fn edit_frame_id(&self) -> ConstructionFrameId {
        self.construction_frames.editing_frame
    }

    /// Rigid transform of the active authoring grid.
    pub fn edit_frame(&self) -> ConstructionFrame {
        self.construction_frames.frames[&self.edit_frame_id()]
    }

    /// Selects the grid used by subsequent local construction commands.
    ///
    /// # Errors
    /// Rejects a frame that does not belong to this graph.
    pub fn set_edit_frame(&mut self, frame: ConstructionFrameId) -> Result<(), FrameError> {
        if !self.construction_frames.frames.contains_key(&frame) {
            return Err(FrameError::MissingFrame(frame));
        }
        self.construction_frames.editing_frame = frame;
        Ok(())
    }

    /// Explicit authored-grid membership of a live shape region.
    pub fn region_frame_id(&self, region: RegionId) -> Option<ConstructionFrameId> {
        self.region(region)
            .and_then(|_| self.construction_frames.region_frames.get(&region).copied())
    }

    /// Rigid transform of a live shape region's local grid.
    pub fn region_frame(&self, region: RegionId) -> Option<ConstructionFrame> {
        self.region_frame_id(region)
            .and_then(|id| self.construction_frames.frames.get(&id).copied())
    }

    /// Source identity retained for live publication of replacement geometry.
    /// Removed intermediate identities remain queryable through the ancestry chain.
    pub fn edit_source(&self, part: PartId) -> Option<PartId> {
        self.construction_frames.origins.get(&part).copied()
    }

    /// Records the prior live part whose pose and velocity field a new part inherits.
    ///
    /// # Errors
    /// Rejects a missing destination or a cyclic ancestry chain.
    pub fn set_edit_source(&mut self, part: PartId, source: PartId) -> Result<(), FrameError> {
        if self.part(part).is_none() {
            return Err(FrameError::MissingPart(part));
        }
        let mut cursor = Some(source);
        while let Some(ancestor) = cursor {
            if ancestor == part {
                return Err(FrameError::CyclicAncestry);
            }
            cursor = self.edit_source(ancestor);
        }
        self.construction_frames.origins.insert(part, source);
        Ok(())
    }

    /// Every graph-owned rigid frame, including the default grid.
    pub fn construction_frames(
        &self,
    ) -> impl Iterator<Item = (ConstructionFrameId, ConstructionFrame)> + '_ {
        self.construction_frames
            .frames
            .iter()
            .map(|(&id, &frame)| (id, frame))
    }

    /// Stable frame membership; newly authored parts use the default grid.
    pub fn part_frame_id(&self, part: PartId) -> Option<ConstructionFrameId> {
        self.part(part).map(|_| {
            self.construction_frames
                .members
                .get(&part)
                .copied()
                .unwrap_or_default()
        })
    }

    /// Resolves the authored grid containing a part.
    pub fn part_frame(&self, part: PartId) -> Option<ConstructionFrame> {
        self.part_frame_id(part)
            .and_then(|id| self.construction_frames.frames.get(&id).copied())
    }

    /// Composed part centre in build space, leaving its exact local `BuildPose` intact.
    pub fn part_position(&self, part: PartId) -> Option<Vec3> {
        Some(
            self.part_frame(part)?
                .point(self.part(part)?.pose().translation()),
        )
    }

    /// Composed part orientation in build space.
    pub fn part_rotation(&self, part: PartId) -> Option<Quat> {
        Some(self.part_frame(part)?.rotation() * self.part(part)?.pose().rotation.quaternion())
    }

    /// Adds a graph-owned frame for later part assignment.
    ///
    /// # Errors
    /// Returns an error if frame identities are exhausted.
    pub fn add_construction_frame(
        &mut self,
        frame: ConstructionFrame,
    ) -> Result<ConstructionFrameId, FrameError> {
        let id = ConstructionFrameId(self.construction_frames.next_id);
        let next = id.0.checked_add(1).ok_or(FrameError::Exhausted)?;
        self.construction_frames.next_id = next;
        self.construction_frames.frames.insert(id, frame);
        Ok(id)
    }

    /// Assigns an unconnected/new part to an existing authored frame.
    /// Connection geometry must be validated when publishing the edited graph.
    ///
    /// # Errors
    /// Rejects missing handles and parts already owned by a shape region.
    pub fn assign_part_frame(
        &mut self,
        part: PartId,
        frame: ConstructionFrameId,
    ) -> Result<(), FrameError> {
        if self.part(part).is_none() {
            return Err(FrameError::MissingPart(part));
        }
        if !self.construction_frames.frames.contains_key(&frame) {
            return Err(FrameError::MissingFrame(frame));
        }
        if let Some(region) = self.region_of(part) {
            return Err(FrameError::PartialRegion(region));
        }
        self.construction_frames.members.insert(part, frame);
        Ok(())
    }

    /// Rigidly reframes selected parts without quantizing their local grid poses.
    /// Each original frame retains its internal relative arrangement.
    ///
    /// # Errors
    /// Rejects missing parts, partial shape regions, and exhausted frame identities.
    ///
    /// # Panics
    /// Panics only if a bearing disappears inside this synchronous transaction.
    pub fn reframe_parts(
        &mut self,
        parts: impl IntoIterator<Item = PartId>,
        transform: ConstructionFrame,
    ) -> Result<(), FrameError> {
        let parts = parts.into_iter().collect::<BTreeSet<_>>();
        let mut groups = BTreeMap::<ConstructionFrameId, Vec<PartId>>::new();
        for &part in &parts {
            let frame = self
                .part_frame_id(part)
                .ok_or(FrameError::MissingPart(part))?;
            if let Some(region) = self.region_of(part)
                && self.parts().any(|(other, _)| {
                    self.region_of(other) == Some(region) && !parts.contains(&other)
                })
            {
                return Err(FrameError::PartialRegion(region));
            }
            groups.entry(frame).or_default().push(part);
        }
        let mut staged = self.clone();
        let bearings = self
            .bearings()
            .map(|(id, bearing)| (id, *bearing))
            .collect::<Vec<_>>();
        for (id, mut bearing) in bearings {
            let included = |owner| matches!(owner, FaceOwner::Part(part) if parts.contains(&part));
            let first = included(bearing.source.owner);
            let second = included(bearing.target.owner);
            if first != second {
                return Err(FrameError::PartialBearing(id));
            }
            if first {
                bearing.shared_anchor = transform.point(bearing.shared_anchor);
                bearing.axis = transform.vector(bearing.axis);
                bearing.kind.rotate(|vector| transform.vector(vector));
                *staged.bearings.get_mut(id).expect("bearing is live") = bearing;
            }
        }
        for (frame, members) in groups {
            let composed = transform.compose(staged.construction_frames.frames[&frame]);
            let id = staged.add_construction_frame(composed)?;
            for part in members {
                if let Some(region) = self.region_of(part) {
                    staged.construction_frames.region_frames.insert(region, id);
                }
                staged.construction_frames.members.insert(part, id);
            }
        }
        *self = staged;
        Ok(())
    }

    pub(crate) fn owner_frame(&self, owner: SolidOwner) -> ConstructionFrame {
        let frame = match owner {
            SolidOwner::Part(part) => self.part_frame_id(part),
            SolidOwner::Region(region) => self.region_frame_id(region),
        };
        frame
            .and_then(|id| self.construction_frames.frames.get(&id).copied())
            .unwrap_or(ConstructionFrame::IDENTITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BuildCommand, BuildOutcome, BuildPose, ConstructionMaterial, CuboidSpec, GridRotation,
        ShapeRegion,
    };
    use bevy_math::IVec3;

    fn spawn_and_claim(graph: &mut ConstructionGraph) -> (PartId, RegionId) {
        let spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(IVec3::ONE, GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            panic!("spawned part expected")
        };
        let region = ShapeRegion::from_origin_steps(
            IVec3::ZERO,
            IVec3::ONE,
            ConstructionMaterial::default(),
        )
        .unwrap();
        let BuildOutcome::RegionAdded(region) =
            graph.apply(BuildCommand::AddRegion(region)).unwrap()
        else {
            panic!("added region expected")
        };
        (part, region)
    }

    #[test]
    fn overlapping_local_regions_keep_independent_frame_membership() {
        let mut graph = ConstructionGraph::new();
        let (first_part, first_region) = spawn_and_claim(&mut graph);
        let offset = ConstructionFrame::new(Vec3::X * 5.0, Quat::from_rotation_y(0.4)).unwrap();
        let second_frame = graph.add_construction_frame(offset).unwrap();
        graph.set_edit_frame(second_frame).unwrap();
        let (second_part, second_region) = spawn_and_claim(&mut graph);
        assert_eq!(graph.part_frame_id(second_part), Some(second_frame));
        assert_eq!(graph.region_frame_id(second_region), Some(second_frame));
        assert_eq!(graph.region_of(first_part), Some(first_region));
        assert_eq!(graph.region_of(second_part), Some(second_region));
        assert_eq!(
            graph.region_frame(first_region),
            Some(ConstructionFrame::IDENTITY)
        );
        assert_eq!(graph.region_frame(second_region), Some(offset));

        let shift = ConstructionFrame::new(Vec3::Y * 3.0, Quat::IDENTITY).unwrap();
        graph.reframe_parts([second_part], shift).unwrap();
        assert_eq!(graph.region_of(first_part), Some(first_region));
        assert_eq!(graph.region_of(second_part), Some(second_region));
        assert_eq!(
            graph.region_frame_id(second_region),
            graph.part_frame_id(second_part)
        );
        assert_eq!(
            graph.region_frame(second_region),
            Some(shift.compose(offset))
        );
        graph.compile().unwrap();
    }

    #[test]
    fn tool_view_round_trip_preserves_local_poses_and_bearing_geometry() {
        let mut graph = ConstructionGraph::new();
        let mut parts = Vec::new();
        for x in [0, 2] {
            let spec = CuboidSpec::new(
                [2, 2, 2],
                BuildPose::new(IVec3::new(x, 0, 0), GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                panic!("part expected")
            };
            parts.push(part);
        }
        let source = crate::FaceRef::part(parts[0], crate::FaceKind::PositiveX);
        graph
            .apply(BuildCommand::AddBearing(crate::BearingSpec::new(
                source,
                crate::FaceRef::part(parts[1], crate::FaceKind::NegativeX),
                Vec3::X * 0.25,
                Vec3::X,
            )))
            .unwrap();
        let transform =
            ConstructionFrame::new(Vec3::new(5.0, 3.0, -2.0), Quat::from_rotation_y(0.71)).unwrap();
        graph
            .reframe_parts(parts.iter().copied(), transform)
            .unwrap();
        let anchor = graph.bearings().next().unwrap().1.shared_anchor;
        graph
            .apply(BuildCommand::BeginPending(
                crate::PendingOperation::Bearing { source, anchor },
            ))
            .unwrap();
        let original = graph.clone();
        let frame = graph.part_frame_id(parts[0]).unwrap();
        let view = graph.in_edit_frame(frame).unwrap();
        assert!(!view.shares_revision(&graph));
        assert!(original.shares_revision(&graph));
        assert_eq!(view.part(parts[0]), graph.part(parts[0]));
        assert_eq!(view.edit_frame_id(), frame);
        assert_eq!(view.edit_frame(), ConstructionFrame::IDENTITY);
        assert!(view.in_edit_frame(frame).unwrap().shares_revision(&view));
        assert!(
            view.part_position(parts[1])
                .unwrap()
                .abs_diff_eq(Vec3::X * 0.5, 1.0e-5)
        );
        let bearing = *view.bearings().next().unwrap().1;
        assert!(bearing.shared_anchor.abs_diff_eq(Vec3::X * 0.25, 1.0e-5));
        assert!(bearing.axis.abs_diff_eq(Vec3::X, 1.0e-5));
        assert!(
            view.view_to_build()
                .point(bearing.shared_anchor)
                .abs_diff_eq(anchor, 1.0e-5)
        );
        let restored = view.canonicalized();
        assert_eq!(restored.edit_frame_id(), ConstructionFrameId::default());
        assert_eq!(restored.view_to_build(), ConstructionFrame::IDENTITY);
        assert!(restored.canonicalized().shares_revision(&restored));
        for part in parts {
            assert_eq!(restored.part(part), graph.part(part));
            assert!(
                restored
                    .part_position(part)
                    .unwrap()
                    .abs_diff_eq(graph.part_position(part).unwrap(), 1.0e-5)
            );
        }
        let restored_bearing = restored.bearings().next().unwrap().1;
        let original_bearing = graph.bearings().next().unwrap().1;
        assert!(
            restored_bearing
                .shared_anchor
                .abs_diff_eq(original_bearing.shared_anchor, 1.0e-5)
        );
        assert!(
            restored_bearing
                .axis
                .abs_diff_eq(original_bearing.axis, 1.0e-5)
        );
        let Some(crate::PendingOperation::Bearing {
            anchor: restored_anchor,
            ..
        }) = restored.pending()
        else {
            panic!("pending bearing expected")
        };
        assert!(restored_anchor.abs_diff_eq(anchor, 1.0e-5));
        assert!(
            graph
                .in_edit_frame(ConstructionFrameId::default())
                .unwrap()
                .shares_revision(&graph)
        );
        assert!(graph.canonicalized().shares_revision(&graph));
    }

    #[test]
    fn missing_edit_frame_does_not_change_active_grid() {
        let mut graph = ConstructionGraph::new();
        assert!(matches!(
            graph.set_edit_frame(ConstructionFrameId(99)),
            Err(FrameError::MissingFrame(_))
        ));
        assert_eq!(graph.edit_frame_id(), ConstructionFrameId::default());
        assert_eq!(graph.edit_frame(), ConstructionFrame::IDENTITY);
    }
}
