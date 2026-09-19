//! Rotary bearings, linear bearings, and suspension: anchors, attachments, and staging.

use super::candidates::{candidate_from_hit, oriented_grid_dimensions};
use super::faces::{
    FaceGeometry, FaceProfile, cylinder_face_geometry, face_geometry, face_geometry_from_ref,
    face_is_flat, faces_share_plane_and_normal, overlap_center, part_face_geometry,
    point_in_profile, profiles_overlap, try_face_geometry_from_ref,
};
use super::grid::{
    cardinal_axis, face_for_normal, rotation_y_to_normal, snap_cardinal, snap_global_center_ticks,
    snap_world_to_position_ticks,
};
use super::pipes::rotation_xy_to_directions;
use super::snap::PlacementSnapIndex;
use super::staging::{
    stage_bearing_block_batch_in_bounds, stage_connected_block_batch,
    stage_connected_block_volume_in_bounds, stage_connected_cylinder,
};
use super::welds::rigid_body_parts;
use super::{
    ALL_FACES, BLOCK_SIZE_UNITS, BlockVolume, BlockVolumePlacement, CONTACT_EPSILON,
    CylinderPlacementCandidate, PlacementBounds, PlacementCandidate, PlacementError, PlacementGrid,
    PlacementSupport, Result, SurfaceHit, ToString, Vec, Vec3,
};
use mechanic_core::{
    BearingDimensions, BearingId, BearingKind, BearingSpec, BuildPose, ConstructionGraph,
    CuboidSpec, CylinderDimensions, CylinderSpec, FaceKind, FaceOwner, FaceRef, GridRotation,
    LinearBearing, LinearBearingDimensions, POSITION_TICK_METERS, PartId,
};
use std::collections::HashSet;

#[derive(Clone, Copy)]
pub(super) struct BearingAttachment<'a> {
    pub(super) source: FaceRef,
    pub(super) anchor: Vec3,
    pub(super) dimensions: BearingDimensions,
    pub(super) kind: BearingKind,
    pub(super) axis: Vec3,
    pub(super) rigid_targets: &'a [PartId],
}

impl<'a> BearingAttachment<'a> {
    pub(super) fn rotational(
        graph: &ConstructionGraph,
        source: FaceRef,
        anchor: Vec3,
        dimensions: BearingDimensions,
        rigid_targets: &'a [PartId],
    ) -> Self {
        Self {
            source,
            anchor,
            dimensions,
            kind: BearingKind::Rotational,
            axis: face_geometry_from_ref(source, Some(graph)).normal,
            rigid_targets,
        }
    }
}

/// A rail socket and any existing direct attachments on its occupied face.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LinearAttachment<'a> {
    pub(crate) source: FaceRef,
    pub(crate) anchor: Vec3,
    pub(crate) rail: LinearBearing,
    pub(crate) axis: Vec3,
    pub(crate) rigid_targets: &'a [PartId],
}

impl<'a> From<LinearAttachment<'a>> for BearingAttachment<'a> {
    fn from(attachment: LinearAttachment<'a>) -> Self {
        Self {
            source: attachment.source,
            anchor: attachment.anchor,
            dimensions: BearingDimensions::default(),
            kind: BearingKind::Linear(attachment.rail),
            axis: attachment.axis,
            rigid_targets: attachment.rigid_targets,
        }
    }
}

pub(super) fn validate_linear_attachment(
    graph: &ConstructionGraph,
    attachment: LinearAttachment<'_>,
) -> Result<(), PlacementError> {
    if !linear_mount_overlaps_face(
        graph,
        attachment.source,
        attachment.anchor,
        attachment.rail,
        attachment.axis,
    ) {
        return Err(PlacementError::BearingOutsideFace);
    }
    if graph.bearings().any(|(_, bearing)| {
        bearing.source == attachment.source
            && bearing.shared_anchor.abs_diff_eq(attachment.anchor, CONTACT_EPSILON)
            && bearing.axis.abs_diff_eq(attachment.axis, CONTACT_EPSILON)
            && matches!(bearing.kind, BearingKind::Linear(existing) if existing.face != attachment.rail.face)
    }) {
        return Err(PlacementError::Graph("the carriage already has attachments on another face".into()));
    }
    Ok(())
}

pub(crate) fn stage_linear_block_batch_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    attachment: LinearAttachment<'_>,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    validate_linear_attachment(graph, attachment)?;
    stage_connected_block_batch(graph, start, specs, Some(attachment.into()), None, bounds)
}

pub(crate) fn stage_linear_block_volume_in_bounds(
    graph: &ConstructionGraph,
    index: &PlacementSnapIndex,
    start: PlacementCandidate,
    volume: BlockVolume,
    attachment: LinearAttachment<'_>,
    bounds: PlacementBounds,
    publication_generation: u64,
) -> Result<BlockVolumePlacement, PlacementError> {
    validate_linear_attachment(graph, attachment)?;
    stage_connected_block_volume_in_bounds(
        graph,
        index,
        start,
        volume,
        Some(attachment.into()),
        None,
        bounds,
        publication_generation,
    )
}

pub(crate) fn stage_linear_cylinder_in_bounds(
    graph: &ConstructionGraph,
    candidate: CylinderPlacementCandidate,
    attachment: LinearAttachment<'_>,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    validate_linear_attachment(graph, attachment)?;
    stage_connected_cylinder(graph, candidate, Some(attachment.into()), None, bounds)
}

/// Bearings whose two sides have ended up in one rigid body, so the joint can
/// no longer turn. Welding into a loop is allowed; this reports the cost.
pub(crate) fn locked_bearings(graph: &ConstructionGraph) -> Vec<BearingId> {
    graph
        .bearings()
        .filter_map(|(id, spec)| bearing_is_locked(graph, spec).then_some(id))
        .collect()
}

pub(super) fn bearing_is_locked(graph: &ConstructionGraph, spec: &BearingSpec) -> bool {
    let (FaceOwner::Part(source), FaceOwner::Part(target)) = (spec.source.owner, spec.target.owner)
    else {
        return false;
    };
    source == target || rigid_body_parts(graph, source).contains(&target)
}

/// How many bearings `after` locks that `before` left free.
pub(crate) fn newly_locked_bearings(
    before: &ConstructionGraph,
    after: &ConstructionGraph,
) -> usize {
    let was_locked = locked_bearings(before);
    locked_bearings(after)
        .into_iter()
        .filter(|id| !was_locked.contains(id))
        .count()
}

#[cfg(test)]
pub(crate) fn bearing_anchor_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
) -> Result<Vec3, PlacementError> {
    bearing_anchor_from_hit_with_grid(
        graph,
        hit,
        PlacementGrid::Centimetres25,
        PlacementBounds::Garage,
    )
}

pub(crate) fn bearing_anchor_from_hit_with_grid(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    grid: PlacementGrid,
    bounds: PlacementBounds,
) -> Result<Vec3, PlacementError> {
    if matches!(hit.face.owner, FaceOwner::Ground) {
        return Err(PlacementError::BearingOnGround);
    }
    let face =
        try_face_geometry_from_ref(hit.face, Some(graph)).ok_or(PlacementError::CurvedSurface)?;
    let (normal_axis, _) = cardinal_axis(face.normal);
    let anchor_ticks = snap_global_center_ticks(
        snap_world_to_position_ticks(hit.point),
        [BLOCK_SIZE_UNITS; 3],
        grid,
        bounds,
    );
    let mut anchor = anchor_ticks.as_vec3() * POSITION_TICK_METERS;
    anchor[normal_axis] = face.center[normal_axis];
    let offset = anchor - face.center;
    let u = offset.dot(face.tangent_u);
    let v = offset.dot(face.tangent_v);
    let inside_face_extent = match face.profile {
        FaceProfile::Annulus { outer_radius, .. }
        | FaceProfile::AnnularSector { outer_radius, .. } => {
            u.mul_add(u, v * v) <= (outer_radius + CONTACT_EPSILON).powi(2)
        }
        _ => point_in_profile(u, v, &face.profile),
    };
    if !inside_face_extent {
        return Err(PlacementError::BearingOutsideFace);
    }
    Ok(anchor)
}

/// Rail undersides need coplanar material overlap, not full support containment.
pub(crate) fn linear_mount_overlaps_face(
    graph: &ConstructionGraph,
    source: FaceRef,
    anchor: Vec3,
    rail: LinearBearing,
    axis: Vec3,
) -> bool {
    if matches!(source.owner, FaceOwner::Ground)
        || !face_is_flat(graph, source)
        || !anchor.is_finite()
        || rail.rotation(axis).is_err()
    {
        return false;
    }
    let Some(face) = try_face_geometry_from_ref(source, Some(graph)) else {
        return false;
    };
    let mount = FaceGeometry {
        center: anchor,
        normal: rail.mount_normal,
        tangent_u: axis,
        tangent_v: axis.cross(rail.mount_normal),
        profile: FaceProfile::Rectangle {
            half_u: rail.dimensions.length() * 0.5,
            half_v: rail.dimensions.width() * 0.5,
        },
    };
    faces_share_plane_and_normal(&mount, &face) && profiles_overlap(&mount, &face)
}

/// Finds a surviving coplanar support when the original mounting part is deleted.
/// The rail may overhang the replacement; only its underside must overlap.
pub(crate) fn linear_support_face_excluding(
    graph: &ConstructionGraph,
    selected_face: FaceRef,
    anchor: Vec3,
    rail: LinearBearing,
    axis: Vec3,
    excluded_parts: &HashSet<PartId>,
) -> Option<FaceRef> {
    let selected = try_face_geometry_from_ref(selected_face, Some(graph))?;
    graph
        .parts()
        .filter(|(part, _)| !excluded_parts.contains(part))
        .find_map(|(part, _)| {
            ALL_FACES.into_iter().find_map(|face| {
                let candidate = FaceRef::part(part, face);
                let geometry = try_face_geometry_from_ref(candidate, Some(graph))?;
                (faces_share_plane_and_normal(&selected, &geometry)
                    && linear_mount_overlaps_face(graph, candidate, anchor, rail, axis))
                .then_some(candidate)
            })
        })
}

pub(crate) fn linear_carriage_face(
    anchor: Vec3,
    rail: LinearBearing,
    axis: Vec3,
) -> Result<FaceGeometry, PlacementError> {
    let rotation = rail
        .rotation(axis)
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    let normal = snap_cardinal(rotation * rail.face.normal());
    let size = rail.face.size(rail.dimensions);
    Ok(FaceGeometry {
        center: anchor + rotation * rail.face.origin(rail.dimensions),
        normal,
        tangent_u: axis,
        tangent_v: axis.cross(normal),
        profile: FaceProfile::Rectangle {
            half_u: size.x * 0.5,
            half_v: size.y * 0.5,
        },
    })
}

pub(super) fn linear_lattice_point(face: &FaceGeometry, point: Vec3) -> Vec3 {
    let delta = point - face.center;
    let pitch = LinearBearingDimensions::ATTACHMENT_PITCH;
    face.center
        + face.tangent_u * ((delta.dot(face.tangent_u) / pitch).round() * pitch)
        + face.tangent_v * ((delta.dot(face.tangent_v) / pitch).round() * pitch)
}

pub(crate) fn linear_block_candidate(
    anchor: Vec3,
    rail: LinearBearing,
    axis: Vec3,
    hit_point: Vec3,
    dimensions: [u8; 3],
    rotation: GridRotation,
) -> Result<PlacementCandidate, PlacementError> {
    let surface = linear_carriage_face(anchor, rail, axis)?;
    let point = linear_lattice_point(&surface, hit_point);
    let world_dimensions = oriented_grid_dimensions(dimensions, rotation);
    let normal_axis = cardinal_axis(surface.normal).0;
    let center = point + surface.normal * (f32::from(world_dimensions[normal_axis]) * 0.125);
    let spec = CuboidSpec::new(
        dimensions,
        BuildPose::from_position_ticks(snap_world_to_position_ticks(center), rotation),
    )
    .map_err(|error| PlacementError::Graph(error.to_string()))?;
    let attached_face = face_for_normal(rotation.quaternion().inverse() * -surface.normal);
    let candidate_face = face_geometry(spec, attached_face);
    Ok(PlacementCandidate {
        spec,
        attached_face,
        anchor: overlap_center(&surface, &candidate_face),
        support: PlacementSupport::Bearing,
    })
}

pub(crate) fn linear_cylinder_candidate(
    anchor: Vec3,
    rail: LinearBearing,
    axis: Vec3,
    hit_point: Vec3,
    dimensions: CylinderDimensions,
    quarter_turns: u8,
) -> Result<CylinderPlacementCandidate, PlacementError> {
    let surface = linear_carriage_face(anchor, rail, axis)?;
    let point = linear_lattice_point(&surface, hit_point);
    let frame = rotation_y_to_normal(surface.normal).quaternion()
        * GridRotation::new(0, quarter_turns % 4, 0).quaternion();
    let rotation = rotation_xy_to_directions(frame * Vec3::X, surface.normal)
        .expect("cardinal carriage faces have a cardinal cylinder frame");
    let center = point + surface.normal * (dimensions.axial_length() * 0.5);
    let spec = CylinderSpec::new(
        dimensions,
        BuildPose::from_position_ticks(snap_world_to_position_ticks(center), rotation),
    );
    let attached_face = FaceKind::NegativeY;
    let candidate_face = cylinder_face_geometry(spec, attached_face).expect("cylinder end is flat");
    Ok(CylinderPlacementCandidate {
        spec,
        attached_face,
        anchor: overlap_center(&surface, &candidate_face),
        support: PlacementSupport::Bearing,
    })
}

pub(crate) fn bearing_attachment_candidate(
    graph: &ConstructionGraph,
    source: FaceRef,
    anchor: Vec3,
) -> PlacementCandidate {
    candidate_from_hit(
        graph,
        SurfaceHit {
            distance: 0.0,
            point: anchor,
            face: source,
        },
    )
}

pub(crate) fn bearing_support_face(
    graph: &ConstructionGraph,
    selected_face: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
) -> Option<FaceRef> {
    bearing_support_face_excluding(graph, selected_face, anchor, dimensions, &HashSet::new())
}

pub(crate) fn bearing_support_face_excluding(
    graph: &ConstructionGraph,
    selected_face: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    excluded_parts: &HashSet<PartId>,
) -> Option<FaceRef> {
    let selected = face_geometry_from_ref(selected_face, Some(graph));
    let mut fallback = None;
    for (part, spec) in graph.parts() {
        if excluded_parts.contains(&part) {
            continue;
        }
        for face_kind in ALL_FACES {
            let face_ref = FaceRef::part(part, face_kind);
            let Some(face) = part_face_geometry(*spec, face_kind) else {
                continue;
            };
            if !faces_share_plane_and_normal(&selected, &face)
                || !bearing_ring_overlaps_face(anchor, dimensions, &face)
            {
                continue;
            }
            if bearing_ring_contains_face_center(anchor, dimensions, &face) {
                return Some(face_ref);
            }
            fallback.get_or_insert(face_ref);
        }
    }
    fallback
}

pub(crate) fn bearing_overlaps_candidate(
    graph: &ConstructionGraph,
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    candidate: PlacementCandidate,
) -> bool {
    let source_face = face_geometry_from_ref(source, Some(graph));
    let target_face = face_geometry(candidate.spec, candidate.attached_face);
    if source_face.normal.dot(target_face.normal) > -1.0 + CONTACT_EPSILON
        || (source_face.center - target_face.center)
            .dot(source_face.normal)
            .abs()
            > CONTACT_EPSILON
    {
        return false;
    }
    bearing_ring_overlaps_face(anchor, dimensions, &target_face)
}

pub(crate) fn bearing_overlaps_cylinder_candidate(
    graph: &ConstructionGraph,
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    candidate: CylinderPlacementCandidate,
) -> bool {
    let source_face = face_geometry_from_ref(source, Some(graph));
    let target_face = cylinder_face_geometry(candidate.spec, candidate.attached_face)
        .expect("cylinder attachment face is flat");
    source_face.normal.dot(target_face.normal) <= -1.0 + CONTACT_EPSILON
        && (source_face.center - target_face.center)
            .dot(source_face.normal)
            .abs()
            <= CONTACT_EPSILON
        && bearing_ring_overlaps_face(anchor, dimensions, &target_face)
}

/// Moves a cylinder laterally so its attachment-face centre starts on the
/// bearing axis. The axial position and orientation already come from the
/// supporting face and remain unchanged.
pub(crate) fn center_cylinder_candidate_on_bearing(
    mut candidate: CylinderPlacementCandidate,
    anchor: Vec3,
) -> CylinderPlacementCandidate {
    let attachment = cylinder_face_geometry(candidate.spec, candidate.attached_face)
        .expect("cylinder attachment face is flat");
    let translation_ticks = candidate.spec.pose.translation_position_ticks()
        + snap_world_to_position_ticks(anchor - attachment.center);
    candidate.spec.pose =
        BuildPose::from_position_ticks(translation_ticks, candidate.spec.pose.rotation);
    candidate.anchor = Some(anchor);
    candidate.support = PlacementSupport::Bearing;
    candidate
}

#[cfg(test)]
pub(crate) fn stage_bearing_attachment(
    graph: &ConstructionGraph,
    candidate: PlacementCandidate,
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
) -> Result<ConstructionGraph, PlacementError> {
    stage_bearing_attachment_in_bounds(
        graph,
        candidate,
        source,
        anchor,
        dimensions,
        PlacementBounds::Garage,
    )
}

pub(crate) fn stage_bearing_attachment_in_bounds(
    graph: &ConstructionGraph,
    candidate: PlacementCandidate,
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_bearing_block_batch_in_bounds(
        graph,
        candidate,
        &[candidate.spec],
        source,
        anchor,
        dimensions,
        &[],
        bounds,
    )
}

pub(super) fn bearing_ring_overlaps_face(
    anchor: Vec3,
    dimensions: BearingDimensions,
    face: &FaceGeometry,
) -> bool {
    profiles_overlap(
        &FaceGeometry {
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

pub(super) fn bearing_ring_contains_face_center(
    anchor: Vec3,
    dimensions: BearingDimensions,
    face: &FaceGeometry,
) -> bool {
    let offset = anchor - face.center;
    if offset.dot(face.normal).abs() > CONTACT_EPSILON {
        return false;
    }
    let radial = offset - face.normal * offset.dot(face.normal);
    point_in_profile(
        radial.dot(face.tangent_u),
        radial.dot(face.tangent_v),
        &FaceProfile::Annulus {
            inner_radius: dimensions.inner_diameter() * 0.5,
            outer_radius: dimensions.outer_diameter() * 0.5,
        },
    )
}

pub(crate) fn plate_block_candidate(
    socket: crate::editor::build_actions::PlacedBearing,
) -> Result<PlacementCandidate, PlacementError> {
    let Some((plate, _)) = socket.moving_plate() else {
        return Err(PlacementError::BearingOutsideFace);
    };
    let center = plate + socket.axis * 0.125;
    Ok(PlacementCandidate {
        spec: CuboidSpec::new(
            [1; 3],
            BuildPose::from_position_ticks(
                snap_world_to_position_ticks(center),
                GridRotation::default(),
            ),
        )
        .map_err(|e| PlacementError::Graph(e.to_string()))?,
        attached_face: face_for_normal(-socket.axis),
        anchor: Some(plate),
        support: PlacementSupport::Bearing,
    })
}

pub(crate) fn plate_cylinder_candidate(
    socket: crate::editor::build_actions::PlacedBearing,
    dimensions: CylinderDimensions,
) -> Result<CylinderPlacementCandidate, PlacementError> {
    let Some((plate, _)) = socket.moving_plate() else {
        return Err(PlacementError::BearingOutsideFace);
    };
    let center = plate + socket.axis * (dimensions.axial_length() / 2.0);
    Ok(CylinderPlacementCandidate {
        spec: CylinderSpec::new(
            dimensions,
            BuildPose::from_position_ticks(
                snap_world_to_position_ticks(center),
                rotation_y_to_normal(socket.axis),
            ),
        ),
        attached_face: FaceKind::NegativeY,
        anchor: Some(plate),
        support: PlacementSupport::Bearing,
    })
}

pub(super) fn plate_attachment(
    socket: crate::editor::build_actions::PlacedBearing,
    targets: &[PartId],
) -> BearingAttachment<'_> {
    BearingAttachment {
        source: socket.source,
        anchor: socket.anchor,
        dimensions: socket.dimensions,
        kind: socket.kind,
        axis: socket.axis,
        rigid_targets: targets,
    }
}

pub(crate) fn stage_plate_block(
    graph: &ConstructionGraph,
    socket: crate::editor::build_actions::PlacedBearing,
    candidate: PlacementCandidate,
    targets: &[PartId],
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_block_batch(
        graph,
        candidate,
        &[candidate.spec],
        Some(plate_attachment(socket, targets)),
        None,
        bounds,
    )
}

pub(crate) fn stage_plate_cylinder(
    graph: &ConstructionGraph,
    socket: crate::editor::build_actions::PlacedBearing,
    candidate: CylinderPlacementCandidate,
    targets: &[PartId],
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_cylinder(
        graph,
        candidate,
        Some(plate_attachment(socket, targets)),
        None,
        bounds,
    )
}
