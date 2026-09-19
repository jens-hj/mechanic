//! Validation of welds, links, drives, shape features, and bearings before they are applied.

use super::ConstructionGraph;
use super::error::GraphError;
use super::predicates::{
    axis_cosine_tolerance, bearing_ring_overlaps_face, convex_polygons_overlap, faces_touch,
    profiles_overlap, simple_grid_face_on_ground, simple_grid_faces_touch,
};
use super::specs::{
    BearingSpec, DriveLinkSpec, InputSeatLinkSpec, RigidLinkSpec, SeatControllerLinkSpec, WeldSpec,
};
use crate::geometry::{FaceGeometry, FaceProfile};
use crate::{
    ANCHOR_TOLERANCE_METERS, BearingId, CuboidSpec, DriveProgram, FaceOwner, FaceRef, PartSpec,
    ShapeFeature, ShapeRegion, SolidError, SolidOwner,
};
use bevy_math::{IVec3, Quat, Vec3};
use std::collections::BTreeSet;

/// Checks a piston's frame and that its mount lies on the supporting face,
/// returning the piston's local-to-world rotation.
fn piston_mount(
    piston: crate::Piston,
    anchor: Vec3,
    axis: Vec3,
    source: &FaceGeometry,
) -> Result<Quat, GraphError> {
    let rotation = piston.rotation(axis)?;
    let half_section = crate::PistonDimensions::SECTION / 2.0;
    let mount = match piston.mount {
        crate::PistonMount::End => FaceGeometry {
            center: anchor,
            normal: axis,
            tangent_u: source.tangent_u,
            tangent_v: source.tangent_v,
            profile: FaceProfile::Annulus {
                inner_radius: 0.0,
                outer_radius: half_section,
            },
        },
        crate::PistonMount::Side { mount_normal } => FaceGeometry {
            center: anchor,
            normal: mount_normal,
            tangent_u: axis,
            tangent_v: axis.cross(mount_normal),
            profile: FaceProfile::Rectangle {
                half_u: piston.dimensions.closed() / 2.0,
                half_v: half_section,
            },
        },
    };
    if source.normal.dot(mount.normal) < 1.0 - axis_cosine_tolerance() {
        return Err(match piston.mount {
            crate::PistonMount::End => GraphError::InvalidBearingAxis,
            crate::PistonMount::Side { .. } => GraphError::BearingFacesNotOpposed,
        });
    }
    if !anchor.is_finite()
        || (source.center - anchor).dot(source.normal).abs() > ANCHOR_TOLERANCE_METERS
        || !profiles_overlap(&mount, source)
    {
        return Err(GraphError::BearingAnchorOutsideFaces);
    }
    Ok(rotation)
}

impl ConstructionGraph {
    pub(super) fn validate_weld(&self, spec: WeldSpec) -> Result<(), GraphError> {
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

    pub(super) fn validate_simple_grid_weld(
        &self,
        spec: WeldSpec,
    ) -> Option<Result<(), GraphError>> {
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
            // Layered cuboids leave the grid, so they take the general check.
            Ok((face.patch.is_none()
                && cuboid.pose.rotation == crate::GridRotation::default()
                && cuboid.layers().is_empty())
            .then_some(cuboid))
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

    pub(super) fn validate_rigid_link(&self, spec: RigidLinkSpec) -> Result<(), GraphError> {
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

    pub(super) fn validate_drive_units(
        &self,
        bearing: BearingId,
        program: DriveProgram,
        limits: Option<crate::LinearDriveLimits>,
    ) -> Result<(), GraphError> {
        let bearing = self
            .bearing(bearing)
            .ok_or(GraphError::MissingBearing(bearing))?;
        if !bearing.kind.accepts_drive() {
            return Err(GraphError::IncompatibleDrive);
        }
        let linear = bearing.kind.is_translational();
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

    pub(super) fn validate_drive_link(&self, spec: &DriveLinkSpec) -> Result<(), GraphError> {
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
        if spec.reversed
            && self
                .bearing(spec.bearing)
                .is_some_and(|bearing| bearing.kind.is_one_sided())
        {
            return Err(GraphError::IncompatibleDrive);
        }
        if self
            .drive_links
            .iter()
            .any(|(_, link)| link.bearing == spec.bearing)
        {
            return Err(GraphError::BearingAlreadyDriven(spec.bearing));
        }
        Ok(())
    }

    pub(super) fn validate_input_seat_link(
        &self,
        spec: InputSeatLinkSpec,
    ) -> Result<(), GraphError> {
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

    pub(super) fn validate_seat_controller_link(
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

    pub(super) fn validate_shape_feature_targets(
        &self,
        feature: &ShapeFeature,
    ) -> Result<(), GraphError> {
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
    pub(super) fn promoted_region_for_feature(
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
            maximum = maximum.max(origin + cells.counts() * crate::POSITION_TICKS_PER_GRID_UNIT);
            members += 1;
        }
        if members < 2
            || members < target_parts.len()
            || minimum.cmpeq(IVec3::splat(i32::MAX)).all()
        {
            return None;
        }
        let extent = maximum - minimum;
        if extent.rem_euclid(IVec3::splat(crate::POSITION_TICKS_PER_GRID_UNIT)) != IVec3::ZERO {
            return None;
        }
        let region = ShapeRegion::from_origin_steps(
            minimum,
            extent / crate::POSITION_TICKS_PER_GRID_UNIT,
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

    pub(super) fn validate_shape_owner_replay(&self, owner: SolidOwner) -> Result<(), GraphError> {
        self.evaluated_solid_shared(owner).map(|_| ())
    }

    pub(super) fn validate_shape_owner_connections(
        &self,
        owner: SolidOwner,
    ) -> Result<(), GraphError> {
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
                || bearing
                    .target
                    .is_some_and(|target| self.face_is_on_owner(target, owner))
            {
                self.validate_bearing(*bearing)?;
            }
        }
        Ok(())
    }

    pub(super) fn face_is_on_owner(&self, face: FaceRef, owner: SolidOwner) -> bool {
        let FaceOwner::Part(part) = face.owner else {
            return false;
        };
        match owner {
            SolidOwner::Part(candidate) => part == candidate,
            SolidOwner::Region(region) => self.region_of(part) == Some(region),
        }
    }

    pub(super) fn remove_shape_owner(&mut self, owner: SolidOwner) {
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

    /// Checks an unattached suspension or piston socket, whose hardware the
    /// compiler weighs, against its supporting face. Other kinds always pass.
    ///
    /// # Errors
    /// Returns the same frame, axis, and anchor errors an attached bearing would.
    pub fn validate_socket(&self, socket: crate::BearingSocket) -> Result<(), GraphError> {
        let spec = match socket.kind {
            crate::BearingKind::Suspension(spec) => spec,
            crate::BearingKind::Piston(piston) => {
                if matches!(socket.source.owner, FaceOwner::Ground) {
                    return Err(GraphError::BearingOnGround);
                }
                let source = self.face_geometry(socket.source)?;
                return piston_mount(piston, socket.anchor, socket.axis, &source).map(|_| ());
            }
            crate::BearingKind::Rotational | crate::BearingKind::Linear(_) => return Ok(()),
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

    /// A joint with nothing on its moving side: only hardware that carries its
    /// own head has one.
    fn validate_bare_bearing(&self, spec: BearingSpec) -> Result<(), GraphError> {
        if !spec.kind.owns_head() {
            return Err(GraphError::BearingWithoutTarget);
        }
        if self.bearings().any(|(_, existing)| {
            existing.source == spec.source
                && existing.shared_anchor.distance(spec.shared_anchor) < ANCHOR_TOLERANCE_METERS
                && existing.kind != spec.kind
        }) {
            return Err(GraphError::PistonHeadOccupied);
        }
        self.validate_socket(crate::BearingSocket {
            kind: spec.kind,
            axis: spec.axis,
            source: spec.source,
            anchor: spec.shared_anchor,
            dimensions: spec.dimensions,
        })
    }

    #[expect(clippy::too_many_lines)]
    pub(super) fn validate_bearing(&self, spec: BearingSpec) -> Result<(), GraphError> {
        let Some(target) = spec.target else {
            return self.validate_bare_bearing(spec);
        };
        if spec.source == target {
            return Err(GraphError::SameFace);
        }
        if matches!(spec.source.owner, FaceOwner::Ground)
            || matches!(target.owner, FaceOwner::Ground)
        {
            return Err(GraphError::BearingOnGround);
        }
        let source = self.face_geometry(spec.source)?;
        let target = self.face_geometry(target)?;
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
        if let crate::BearingKind::Piston(piston) = spec.kind {
            if self.bearings().any(|(_, existing)| {
                existing.source == spec.source
                    && existing.shared_anchor.distance(spec.shared_anchor) < ANCHOR_TOLERANCE_METERS
                    && existing.kind != spec.kind
            }) {
                return Err(GraphError::PistonHeadOccupied);
            }
            let rotation = piston_mount(piston, spec.shared_anchor, spec.axis, &source)?;
            if target.normal.dot(spec.axis) > -1.0 + axis_cosine_tolerance() {
                return Err(GraphError::BearingFacesNotOpposed);
            }
            let head = FaceGeometry {
                center: piston.head_center(spec.shared_anchor, spec.axis, 0.0),
                normal: spec.axis,
                tangent_u: rotation * Vec3::X,
                tangent_v: rotation * Vec3::Z,
                profile: FaceProfile::Annulus {
                    inner_radius: 0.0,
                    outer_radius: piston.dimensions.head_radius(),
                },
            };
            if (target.center - head.center).dot(spec.axis).abs() > ANCHOR_TOLERANCE_METERS
                || !profiles_overlap(&head, &target)
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
