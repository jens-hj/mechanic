//! Read access to parts, welds, links, bearings, regions, and their evaluated solids.

use super::commands::PendingOperation;
use super::error::GraphError;
use super::specs::{
    BearingSpec, DriveLinkSpec, GearLinkSpec, InputSeatLinkSpec, RigidLinkSpec,
    SeatControllerLinkSpec, WeldSpec,
};
use super::{ConstructionGraph, solid_cache};
use crate::{
    BearingId, DimensionLinkId, DriveLinkId, GearLinkId, InputSeatLinkId, PartId, PartSpec,
    RegionId, RigidLinkId, SeatControllerLinkId, ShapeFeature, ShapeFeatureId, ShapeRegion,
    SolidOwner, WeldId,
};
use bevy_math::IVec3;
use std::sync::Arc;

impl ConstructionGraph {
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

    pub(super) fn evaluated_solid_until(
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

    /// Retrieves a live mesh.
    pub fn gear_link(&self, id: GearLinkId) -> Option<&GearLinkSpec> {
        self.gear_links.get(id)
    }

    /// Every mesh one part takes part in, in canonical slot order.
    pub fn part_gear_links(
        &self,
        part: PartId,
    ) -> impl Iterator<Item = (GearLinkId, &GearLinkSpec)> {
        self.gear_links
            .iter()
            .filter(move |(_, link)| link.references(part))
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

    /// Iterates live meshes in canonical slot order.
    pub fn gear_links(&self) -> impl Iterator<Item = (GearLinkId, &GearLinkSpec)> {
        self.gear_links.iter()
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
}
