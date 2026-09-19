//! Applying one validated build command to the graph.

use super::ConstructionGraph;
use super::commands::{AppearanceTarget, BuildCommand, BuildOutcome, PendingOperation};
use super::error::GraphError;
use super::predicates::{
    bearing_references, point_on_face, rigid_link_references, weld_references,
};
use super::specs::WeldSpec;
use crate::{EngineKind, FaceKind, FaceOwner, FaceRef, PartId, PartSpec, SolidError, SolidOwner};
use std::collections::BTreeSet;

impl ConstructionGraph {
    /// Every part welded, directly or transitively, to `seed`.
    pub(super) fn rigid_group(&self, seed: PartId) -> BTreeSet<PartId> {
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
    pub(super) fn weld_group(&self, seed: PartId) -> BTreeSet<PartId> {
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

    pub(super) fn insert_edit_part(&mut self, spec: PartSpec) -> PartId {
        let id = self.parts.insert(spec);
        let frame = self.edit_frame_id();
        self.construction_frames.members.insert(id, frame);
        id
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one exhaustive command dispatch reads better whole"
    )]
    pub(super) fn apply_validated(
        &mut self,
        command: BuildCommand,
    ) -> Result<BuildOutcome, GraphError> {
        match command {
            BuildCommand::Spawn(spec) => {
                let id = self.insert_edit_part(spec.into());
                Ok(BuildOutcome::Spawned(id))
            }
            BuildCommand::SpawnCylinder(spec) => {
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
                    band @ AppearanceTarget::PartBand { .. } => band,
                };
                match target {
                    AppearanceTarget::PartBand { part, band } => {
                        let replacement = self
                            .parts
                            .get(part)
                            .copied()
                            .ok_or(GraphError::MissingPart(part))?
                            .with_band_appearance(band, appearance)
                            .ok_or(GraphError::MissingBand(part, band))?;
                        *self
                            .parts
                            .get_mut(part)
                            .expect("the validated part remains live") = replacement;
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
                if spec.target.is_none()
                    && self.bearings().any(|(_, existing)| {
                        existing.target.is_none()
                            && existing.source == spec.source
                            && existing.shared_anchor.distance(spec.shared_anchor)
                                < crate::ANCHOR_TOLERANCE_METERS
                    })
                {
                    return Err(GraphError::PistonHeadOccupied);
                }
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
            BuildCommand::SetLayers { part, spec } => {
                let current = self
                    .parts
                    .get(part)
                    .copied()
                    .ok_or(GraphError::MissingPart(part))?;
                if !current.shares_core_with(spec) {
                    return Err(GraphError::LayerCoreChanged(part));
                }
                if self.region_of(part).is_some() {
                    return Err(GraphError::LayeredPartInRegion(part));
                }
                *self
                    .parts
                    .get_mut(part)
                    .expect("the validated part remains live") = spec;
                let owner = SolidOwner::Part(part);
                self.validate_shape_owner_replay(owner)?;
                self.validate_shape_owner_connections(owner)?;
                Ok(BuildOutcome::LayersUpdated)
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
}
