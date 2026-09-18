//! Cursor rays against the construction: parts, regions, evaluated solids, and the ground.

use super::bounds::pipe_bend_collision_boxes;
use super::faces::{
    FaceGeometry, FaceProfile, cylinder_face_geometry, face_normal, point_in_profile,
    profiles_overlap,
};
use super::grid::face_for_normal;
use super::pipes::face_toward;
use super::{
    CONTACT_EPSILON, GROUND_HALF_SIZE, OrientedCuboidHit, PlacementPlane, Quat, SurfaceHit, Vec,
    Vec3,
};
use mechanic_core::{
    ConstructionGraph, ConvexPiece, CuboidSpec, CylinderDimensions, CylinderSpec, FaceKind,
    FaceRef, PartId, PartPiece, PartSpec, PipeBendSpec, PipeJunctionSpec, ShapeRegion,
};
use std::cmp::Ordering;
use std::collections::HashSet;

pub(crate) fn raycast_construction(
    graph: &ConstructionGraph,
    origin: Vec3,
    direction: Vec3,
) -> Option<SurfaceHit> {
    raycast_construction_with_ground(graph, origin, direction, raycast_ground(origin, direction))
}

pub(crate) fn raycast_construction_with_ground(
    graph: &ConstructionGraph,
    origin: Vec3,
    direction: Vec3,
    ground: Option<SurfaceHit>,
) -> Option<SurfaceHit> {
    raycast_construction_filtered_with_ground(graph, origin, direction, ground, |_| true)
}

/// Restricts eligible parts before choosing a representative region or nearest hit.
pub(crate) fn raycast_construction_filtered_with_ground(
    graph: &ConstructionGraph,
    origin: Vec3,
    direction: Vec3,
    ground: Option<SurfaceHit>,
    accepts_part: impl Fn(PartId) -> bool,
) -> Option<SurfaceHit> {
    if !origin.is_finite() || !direction.is_finite() || direction.length_squared() < f32::EPSILON {
        return None;
    }
    let direction = direction.normalize();
    raycast_sources(graph, accepts_part)
        .filter_map(|(part, _, _)| raycast_part_in_construction(graph, part, origin, direction))
        .chain(ground)
        .filter(|hit| hit.distance >= 0.0 && hit.distance.is_finite())
        .min_by(|left, right| left.distance.total_cmp(&right.distance))
}

/// Exact authored surface for one part, including its shared region and frame.
pub(crate) fn raycast_part_in_construction(
    graph: &ConstructionGraph,
    part: PartId,
    origin: Vec3,
    direction: Vec3,
) -> Option<SurfaceHit> {
    if !origin.is_finite() || !direction.is_finite() || direction.length_squared() < f32::EPSILON {
        return None;
    }
    let direction = direction.normalize();
    let spec = *graph.part(part)?;
    let region = graph.region_of(part);
    if let Some(id) = region {
        let region = graph.region(id)?;
        if graph.owner_has_shape_features(mechanic_core::SolidOwner::Region(id)) {
            let solid = graph
                .evaluated_solid_shared(mechanic_core::SolidOwner::Region(id))
                .ok()?;
            return raycast_evaluated_solid(
                origin,
                direction,
                part,
                &solid,
                graph.part_frame(part)?,
            );
        }
        let frame = graph.part_frame(part)?;
        let inverse = frame.inverse();
        return raycast_region(
            inverse.point(origin),
            inverse.vector(direction),
            part,
            region,
        )
        .map(|hit| composed_surface_hit(hit, frame));
    }
    if graph.owner_has_shape_features(mechanic_core::SolidOwner::Part(part)) {
        let solid = graph
            .evaluated_solid_shared(mechanic_core::SolidOwner::Part(part))
            .ok()?;
        return raycast_evaluated_solid(origin, direction, part, &solid, graph.part_frame(part)?);
    }
    let frame = graph.part_frame(part)?;
    let inverse = frame.inverse();
    raycast_part(inverse.point(origin), inverse.vector(direction), part, spec)
        .map(|hit| composed_surface_hit(hit, frame))
}

pub(super) fn composed_surface_hit(
    mut hit: SurfaceHit,
    frame: mechanic_core::ConstructionFrame,
) -> SurfaceHit {
    hit.point = frame.point(hit.point);
    // A rigid frame preserves ray distance and the identity of the local face.
    hit
}

/// One representative part for each region, plus every standalone part.
///
/// A region owns one shared surface even when hundreds of blocks fill it. The
/// representative part only supplies the legacy `FaceOwner::Part` returned
/// by picking; the region geometry itself must be tested exactly once.
pub(super) fn raycast_sources<'a>(
    graph: &'a ConstructionGraph,
    accepts_part: impl Fn(PartId) -> bool + 'a,
) -> impl Iterator<Item = (PartId, PartSpec, Option<mechanic_core::RegionId>)> + 'a {
    let mut seen_regions = HashSet::new();
    graph.parts().filter_map(move |(part, spec)| {
        if !accepts_part(part) {
            return None;
        }
        let region = graph.region_of(part);
        if region.is_some_and(|region| !seen_regions.insert(region)) {
            return None;
        }
        Some((part, *spec, region))
    })
}

pub(crate) fn raycast_construction_for_annulus(
    graph: &ConstructionGraph,
    origin: Vec3,
    direction: Vec3,
    inner_diameter: f32,
    outer_diameter: f32,
) -> Option<SurfaceHit> {
    let ground = raycast_ground(origin, direction);
    raycast_construction_for_annulus_with_ground(
        graph,
        origin,
        direction,
        inner_diameter,
        outer_diameter,
        ground,
    )
}

pub(crate) fn raycast_construction_for_annulus_with_ground(
    graph: &ConstructionGraph,
    origin: Vec3,
    direction: Vec3,
    inner_diameter: f32,
    outer_diameter: f32,
    ground: Option<SurfaceHit>,
) -> Option<SurfaceHit> {
    raycast_construction_for_annulus_filtered_with_ground(
        graph,
        origin,
        direction,
        inner_diameter,
        outer_diameter,
        ground,
        |_| true,
    )
}

pub(crate) fn raycast_construction_for_annulus_filtered_with_ground(
    graph: &ConstructionGraph,
    origin: Vec3,
    direction: Vec3,
    inner_diameter: f32,
    outer_diameter: f32,
    ground: Option<SurfaceHit>,
    accepts_part: impl Fn(PartId) -> bool,
) -> Option<SurfaceHit> {
    if !origin.is_finite()
        || !direction.is_finite()
        || direction.length_squared() < f32::EPSILON
        || !inner_diameter.is_finite()
        || !outer_diameter.is_finite()
        || inner_diameter < 0.0
        || outer_diameter <= inner_diameter
    {
        return None;
    }
    let direction = direction.normalize();
    let placement_profile = FaceProfile::Annulus {
        inner_radius: inner_diameter * 0.5,
        outer_radius: outer_diameter * 0.5,
    };
    raycast_construction_filtered_with_ground(graph, origin, direction, ground, &accepts_part)
        .into_iter()
        .chain(
            graph
                .parts()
                .filter(|(part, _)| accepts_part(*part))
                .filter_map(|(part, spec)| match spec {
                    PartSpec::Cylinder(spec) => {
                        let frame = graph.part_frame(part)?;
                        let inverse = frame.inverse();
                        raycast_cylinder_bore_obstruction(
                            inverse.point(origin),
                            inverse.vector(direction),
                            part,
                            *spec,
                            &placement_profile,
                        )
                        .map(|hit| composed_surface_hit(hit, frame))
                    }
                    PartSpec::PipeBend(_)
                    | PartSpec::PipeJunction(_)
                    | PartSpec::Cuboid(_)
                    | PartSpec::Controller(_)
                    | PartSpec::Engine(_)
                    | PartSpec::Transmission(_)
                    | PartSpec::Servo(_)
                    | PartSpec::Seat(_)
                    | PartSpec::Input(_)
                    | PartSpec::DimensionLink(_) => None,
                }),
        )
        .min_by(|left, right| left.distance.total_cmp(&right.distance))
}

pub(super) fn raycast_cylinder_bore_obstruction(
    origin: Vec3,
    direction: Vec3,
    part: PartId,
    spec: CylinderSpec,
    placement_profile: &FaceProfile,
) -> Option<SurfaceHit> {
    let rotation = spec.pose.rotation.quaternion();
    let inverse = rotation.inverse();
    let local_origin = inverse * (origin - spec.pose.translation());
    let local_direction = inverse * direction;
    if local_direction.y.abs() <= f32::EPSILON {
        return None;
    }
    let outer_radius = spec.dimensions.outer_diameter() * 0.5;
    let half_length = spec.dimensions.axial_length() * 0.5;
    [
        (half_length, FaceKind::PositiveY),
        (-half_length, FaceKind::NegativeY),
    ]
    .into_iter()
    .filter_map(|(y, face_kind)| {
        let distance = (y - local_origin.y) / local_direction.y;
        if distance < 0.0 {
            return None;
        }
        let local_point = local_origin + local_direction * distance;
        let radial_squared = local_point
            .x
            .mul_add(local_point.x, local_point.z * local_point.z);
        if radial_squared > (outer_radius + CONTACT_EPSILON).powi(2) {
            return None;
        }
        let support = cylinder_face_geometry(spec, face_kind)
            .expect("cylinder end faces always expose flat geometry");
        if point_in_profile(local_point.x, local_point.z, &support.profile) {
            return None;
        }
        let placement = FaceGeometry {
            center: origin + direction * distance,
            normal: -support.normal,
            tangent_u: support.tangent_u,
            tangent_v: support.tangent_v,
            profile: placement_profile.clone(),
        };
        profiles_overlap(&support, &placement).then_some(SurfaceHit {
            distance,
            point: placement.center,
            face: FaceRef::part(part, face_kind),
        })
    })
    .min_by(|left, right| left.distance.total_cmp(&right.distance))
}

/// Intersects a pointer ray with the plane through the dragged block's centre.
///
/// This deliberately leaves the point unsnapped. Block dragging subtracts two
/// such points before quantizing, so the press position rather than the snapped
/// block centre is the gesture's origin.
pub(crate) fn raycast_placement_plane_point(
    origin: Vec3,
    direction: Vec3,
    start: CuboidSpec,
    plane: PlacementPlane,
) -> Option<Vec3> {
    let axis = plane.normal_axis();
    let denominator = direction[axis];
    if !origin.is_finite() || !direction.is_finite() || denominator.abs() <= f32::EPSILON {
        return None;
    }
    let coordinate = start.pose.translation()[axis];
    let distance = (coordinate - origin[axis]) / denominator;
    if distance < 0.0 || !distance.is_finite() {
        return None;
    }
    Some(origin + direction * distance)
}

pub(super) fn raycast_ground(origin: Vec3, direction: Vec3) -> Option<SurfaceHit> {
    raycast_horizontal_surface(origin, direction, 0.0)
}

pub(super) fn raycast_horizontal_surface(
    origin: Vec3,
    direction: Vec3,
    height: f32,
) -> Option<SurfaceHit> {
    // The platform is a build surface from above, not a wall that hides the
    // construction when the camera is underneath it.
    if direction.y >= -f32::EPSILON {
        return None;
    }
    let distance = (height - origin.y) / direction.y;
    let point = origin + direction * distance;
    (distance >= 0.0 && point.x.abs() <= GROUND_HALF_SIZE && point.z.abs() <= GROUND_HALF_SIZE)
        .then_some(SurfaceHit {
            distance,
            point,
            face: FaceRef::ground(),
        })
}

pub(super) fn raycast_cuboid(
    origin: Vec3,
    direction: Vec3,
    part: PartId,
    spec: CuboidSpec,
) -> Option<SurfaceHit> {
    let hit = raycast_oriented_cuboid(
        origin,
        direction,
        spec.pose.translation(),
        spec.pose.rotation.quaternion(),
        spec.size_meters() * 0.5,
    )?;
    Some(SurfaceHit {
        distance: hit.distance,
        point: hit.point,
        face: FaceRef::part(part, face_for_normal(hit.local_normal)),
    })
}

/// Raycasts one shaped region against the same pieces its colliders and its
/// mesh come from, so the cursor lands where the surface actually is.
///
/// The reported face is still the grid face the surface came from, not the
/// tilted plane the ray met. Placement, welding, and face snapping therefore go
/// on working in grid coordinates: the grid stays the grid, and only the hit
/// test gets truthful.
pub(crate) fn raycast_region(
    origin: Vec3,
    direction: Vec3,
    part: PartId,
    region: &ShapeRegion,
) -> Option<SurfaceHit> {
    let direction = direction.normalize();
    let inverse_rotation = Quat::IDENTITY;
    let mut best: Option<SurfaceHit> = None;
    for piece in region_pieces(region) {
        let hit =
            match piece {
                PartPiece::Cuboid {
                    center,
                    half_extents,
                    rotation,
                    ..
                } => raycast_oriented_cuboid(origin, direction, center, rotation, half_extents)
                    .map(|hit| SurfaceHit {
                        distance: hit.distance,
                        point: hit.point,
                        face: FaceRef::part(
                            part,
                            face_for_normal(inverse_rotation * (rotation * hit.local_normal)),
                        ),
                    }),
                PartPiece::Convex(convex) => raycast_convex_piece(origin, direction, &convex).map(
                    |(distance, point, normal)| SurfaceHit {
                        distance,
                        point,
                        face: FaceRef::part(part, face_for_normal(inverse_rotation * normal)),
                    },
                ),
            };
        let Some(hit) = hit else {
            continue;
        };
        if best.is_none_or(|best| hit.distance < best.distance) {
            best = Some(hit);
        }
    }
    best
}

pub(super) fn raycast_evaluated_solid(
    origin: Vec3,
    direction: Vec3,
    part: PartId,
    solid: &mechanic_core::EvaluatedSolid,
    frame: mechanic_core::ConstructionFrame,
) -> Option<SurfaceHit> {
    solid
        .surfaces
        .iter()
        .filter_map(|surface| {
            let (distance, point) = raycast_evaluated_surface(origin, direction, solid, surface)?;
            let placement_surface = if surface.smoothing_group == 0 {
                surface
            } else {
                // A rounded facet is not itself a mounting plane. Route the hit
                // to the nearest retained planar base patch so a block can still
                // bridge the recess wherever its face has real flat contact.
                solid
                    .surfaces
                    .iter()
                    .filter(|candidate| candidate.smoothing_group == 0)
                    .filter(|candidate| {
                        matches!(candidate.key.source, mechanic_core::TopologySource::Base)
                    })
                    .max_by(|left, right| {
                        left.normal
                            .dot(surface.normal)
                            .total_cmp(&right.normal.dot(surface.normal))
                    })?
            };
            Some(SurfaceHit {
                distance,
                point,
                face: FaceRef::patch(
                    part,
                    face_for_normal(frame.inverse().vector(placement_surface.normal)),
                    placement_surface.key,
                ),
            })
        })
        .min_by(|left, right| {
            left.distance
                .partial_cmp(&right.distance)
                .unwrap_or(Ordering::Equal)
        })
}

pub(crate) fn raycast_evaluated_surface(
    origin: Vec3,
    direction: Vec3,
    solid: &mechanic_core::EvaluatedSolid,
    surface: &mechanic_core::SurfacePatch,
) -> Option<(f32, Vec3)> {
    let first_edge = solid.half_edges.get(surface.half_edge as usize)?;
    let plane_point = solid.vertices.get(first_edge.origin as usize)?.position;
    let denominator = direction.dot(surface.normal);
    if denominator.abs() <= f32::EPSILON {
        return None;
    }
    let distance = (plane_point - origin).dot(surface.normal) / denominator;
    if distance < 0.0 || !distance.is_finite() {
        return None;
    }
    let point = origin + direction * distance;
    let mut edge = surface.half_edge;
    loop {
        let half_edge = solid.half_edges.get(edge as usize)?;
        let next = solid.half_edges.get(half_edge.next as usize)?;
        let start = solid.vertices.get(half_edge.origin as usize)?.position;
        let end = solid.vertices.get(next.origin as usize)?.position;
        if surface.normal.dot((end - start).cross(point - start)) < -CONTACT_EPSILON {
            return None;
        }
        edge = half_edge.next;
        if edge == surface.half_edge {
            break;
        }
    }
    Some((distance, point))
}

/// Slab-clips a ray against a convex piece, returning where it enters.
pub(super) fn raycast_convex_piece(
    origin: Vec3,
    direction: Vec3,
    piece: &ConvexPiece,
) -> Option<(f32, Vec3, Vec3)> {
    let mut near = f32::NEG_INFINITY;
    let mut far = f32::INFINITY;
    let mut entry_normal = Vec3::Y;
    for face in &piece.faces {
        let denominator = direction.dot(face.normal);
        let distance = face.offset - origin.dot(face.normal);
        if denominator.abs() <= f32::EPSILON {
            // Parallel to this plane: outside it means the ray misses entirely.
            if distance < 0.0 {
                return None;
            }
            continue;
        }
        let crossing = distance / denominator;
        if denominator < 0.0 {
            if crossing > near {
                near = crossing;
                entry_normal = face.normal;
            }
        } else {
            far = far.min(crossing);
        }
        if near > far {
            return None;
        }
    }
    if !near.is_finite() || near < 0.0 {
        return None;
    }
    Some((near, origin + direction * near, entry_normal))
}

/// The convex pieces one region's cage describes.
pub(crate) fn region_pieces(region: &ShapeRegion) -> Vec<PartPiece> {
    let grid = region.grid();
    mechanic_core::decompose(&grid, &|cell, corner| region.corner_steps(cell, corner))
}

pub(super) fn raycast_part(
    origin: Vec3,
    direction: Vec3,
    part: PartId,
    spec: PartSpec,
) -> Option<SurfaceHit> {
    match spec {
        PartSpec::Cuboid(spec) => raycast_cuboid(origin, direction, part, spec),
        PartSpec::Controller(spec) => raycast_cuboid(origin, direction, part, spec.cuboid()),
        PartSpec::Engine(spec) => raycast_cuboid(origin, direction, part, spec.cuboid()),
        PartSpec::Transmission(spec) => raycast_cuboid(origin, direction, part, spec.cuboid()),
        PartSpec::Servo(spec) => raycast_cuboid(origin, direction, part, spec.cuboid()),
        PartSpec::Seat(spec) => raycast_cuboid(origin, direction, part, spec.cuboid()),
        PartSpec::Input(spec) => raycast_cuboid(origin, direction, part, spec.cuboid()),
        PartSpec::DimensionLink(spec) => raycast_cuboid(origin, direction, part, spec.cuboid()),
        PartSpec::Cylinder(spec) => raycast_cylinder(origin, direction, part, spec),
        PartSpec::PipeBend(spec) => raycast_pipe_bend(origin, direction, part, spec),
        PartSpec::PipeJunction(spec) => raycast_pipe_junction(origin, direction, part, spec),
    }
}

/// Hits a junction's arms as capped pipes. A hit on an open arm's flat end
/// reports that arm; a hit on a wall reports the cube face nearest the hit,
/// which only branching uses.
pub(super) fn raycast_pipe_junction(
    origin: Vec3,
    direction: Vec3,
    part: PartId,
    spec: PipeJunctionSpec,
) -> Option<SurfaceHit> {
    let direction = direction.normalize();
    let inverse = spec.pose.rotation.quaternion().inverse();
    let local_origin = inverse * (origin - spec.pose.translation());
    let local_direction = inverse * direction;
    let radius = spec.dimensions.outer_diameter() * 0.5;
    let reach = spec.dimensions.half_side();
    let mut nearest: Option<(f32, Option<FaceKind>)> = None;
    let mut offer = |distance: f32, end: Option<FaceKind>| {
        if distance >= 0.0 && nearest.is_none_or(|(best, _)| distance < best) {
            nearest = Some((distance, end));
        }
    };
    for arm in spec.arms.faces() {
        let axis = face_normal(arm);
        let along_origin = local_origin.dot(axis);
        let along_direction = local_direction.dot(axis);
        let lateral_origin = local_origin - axis * along_origin;
        let lateral_direction = local_direction - axis * along_direction;
        let a = lateral_direction.length_squared();
        let b = 2.0 * lateral_origin.dot(lateral_direction);
        let c = lateral_origin.length_squared() - radius * radius;
        let discriminant = b * b - 4.0 * a * c;
        if a > 1.0e-12 && discriminant >= 0.0 {
            for distance in [
                (-b - discriminant.sqrt()) / (2.0 * a),
                (-b + discriminant.sqrt()) / (2.0 * a),
            ] {
                if (0.0..=reach).contains(&(along_origin + along_direction * distance)) {
                    offer(distance, None);
                }
            }
        }
        if along_direction.abs() > 1.0e-12 {
            let distance = (reach - along_origin) / along_direction;
            if (lateral_origin + lateral_direction * distance).length_squared() <= radius * radius {
                offer(distance, Some(arm));
            }
        }
    }
    // The ball at the centre.
    let b = local_origin.dot(local_direction);
    let discriminant = b * b - (local_origin.length_squared() - radius * radius);
    if discriminant >= 0.0 {
        offer(-b - discriminant.sqrt(), None);
    }
    let (distance, end) = nearest?;
    let point = origin + direction * distance;
    let face = end.unwrap_or_else(|| face_toward(inverse * (point - spec.pose.translation())));
    Some(SurfaceHit {
        distance,
        point,
        face: FaceRef::part(part, face),
    })
}

pub(super) fn raycast_pipe_bend(
    origin: Vec3,
    direction: Vec3,
    part: PartId,
    spec: PipeBendSpec,
) -> Option<SurfaceHit> {
    let direction = direction.normalize();
    let rotation = spec.pose.rotation.quaternion();
    let inverse = rotation.inverse();
    let local_origin = inverse * (origin - spec.pose.translation());
    let local_direction = inverse * direction;
    let outer = spec.dimensions.outer_diameter() * 0.5;
    let inner = spec.dimensions.inner_diameter() * 0.5;
    let radius = spec.dimensions.radius();
    let mut candidates = Vec::new();
    for (center, normal, face) in [
        (
            Vec3::new(-radius, 0.0, 0.0),
            Vec3::NEG_X,
            FaceKind::NegativeX,
        ),
        (Vec3::new(0.0, radius, 0.0), Vec3::Y, FaceKind::PositiveY),
    ] {
        let denominator = local_direction.dot(normal);
        if denominator.abs() <= f32::EPSILON {
            continue;
        }
        let distance = (center - local_origin).dot(normal) / denominator;
        if distance < 0.0 {
            continue;
        }
        let offset = local_origin + local_direction * distance - center;
        let radial_squared = offset.length_squared();
        if radial_squared >= inner * inner - CONTACT_EPSILON
            && radial_squared <= outer * outer + CONTACT_EPSILON
        {
            candidates.push((distance, face));
        }
    }
    for collider in pipe_bend_collision_boxes(spec) {
        if let Some(hit) = raycast_oriented_cuboid(
            origin,
            direction,
            collider.center,
            collider.rotation,
            collider.half,
        ) {
            candidates.push((hit.distance, FaceKind::PositiveZ));
        }
    }
    let (distance, face) = candidates
        .into_iter()
        .min_by(|left, right| left.0.total_cmp(&right.0))?;
    Some(SurfaceHit {
        distance,
        point: origin + direction * distance,
        face: FaceRef::part(part, face),
    })
}

pub(super) fn raycast_cylinder(
    origin: Vec3,
    direction: Vec3,
    part: PartId,
    spec: CylinderSpec,
) -> Option<SurfaceHit> {
    let direction = direction.normalize();
    let rotation = spec.pose.rotation.quaternion();
    let inverse = rotation.inverse();
    let local_origin = inverse * (origin - spec.pose.translation());
    let local_direction = inverse * direction;
    let outer = spec.dimensions.outer_diameter() * 0.5;
    let inner = spec.dimensions.inner_diameter() * 0.5;
    let half_length = spec.dimensions.axial_length() * 0.5;
    let mut candidates = Vec::with_capacity(6);

    if local_direction.y.abs() > f32::EPSILON {
        for (y, face) in [
            (half_length, FaceKind::PositiveY),
            (-half_length, FaceKind::NegativeY),
        ] {
            let distance = (y - local_origin.y) / local_direction.y;
            if distance >= 0.0 {
                let point = local_origin + local_direction * distance;
                let radial_squared = point.x.mul_add(point.x, point.z * point.z);
                if radial_squared <= outer * outer + CONTACT_EPSILON
                    && radial_squared >= inner * inner - CONTACT_EPSILON
                    && point_in_cylinder_sweep(point.x, point.z, spec.dimensions)
                {
                    candidates.push((distance, face));
                }
            }
        }
    }
    for radius in [outer, inner] {
        if radius <= 0.0 {
            continue;
        }
        let a = local_direction
            .x
            .mul_add(local_direction.x, local_direction.z * local_direction.z);
        if a <= f32::EPSILON {
            continue;
        }
        let b = 2.0
            * local_origin
                .x
                .mul_add(local_direction.x, local_origin.z * local_direction.z);
        let c = local_origin
            .x
            .mul_add(local_origin.x, local_origin.z * local_origin.z)
            - radius * radius;
        let discriminant = b.mul_add(b, -4.0 * a * c);
        if discriminant < 0.0 {
            continue;
        }
        let root = discriminant.sqrt();
        for distance in [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)] {
            if distance >= 0.0 {
                let y = local_origin.y + local_direction.y * distance;
                let point = local_origin + local_direction * distance;
                if y.abs() <= half_length + CONTACT_EPSILON
                    && point_in_cylinder_sweep(point.x, point.z, spec.dimensions)
                {
                    candidates.push((distance, FaceKind::PositiveX));
                }
            }
        }
    }
    if spec.dimensions.sweep_angle_degrees() < 360 {
        let half_sweep = spec.dimensions.sweep_angle_radians() * 0.5;
        for (angle, outward) in [(-half_sweep, -1.0_f32), (half_sweep, 1.0_f32)] {
            let radial = Vec3::new(angle.cos(), 0.0, angle.sin());
            let angular = Vec3::new(-angle.sin(), 0.0, angle.cos()) * outward;
            let denominator = local_direction.dot(angular);
            if denominator.abs() <= f32::EPSILON {
                continue;
            }
            let distance = -local_origin.dot(angular) / denominator;
            if distance < 0.0 {
                continue;
            }
            let point = local_origin + local_direction * distance;
            let radius = point.dot(radial);
            if point.y.abs() <= half_length + CONTACT_EPSILON
                && radius >= inner - CONTACT_EPSILON
                && radius <= outer + CONTACT_EPSILON
            {
                candidates.push((distance, FaceKind::PositiveX));
            }
        }
    }
    let (distance, face) = candidates
        .into_iter()
        .min_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal))?;
    Some(SurfaceHit {
        distance,
        point: origin + direction * distance,
        face: FaceRef::part(part, face),
    })
}

pub(super) fn point_in_cylinder_sweep(x: f32, z: f32, dimensions: CylinderDimensions) -> bool {
    dimensions.sweep_angle_degrees() == 360
        || z.atan2(x).abs() <= dimensions.sweep_angle_radians() * 0.5 + CONTACT_EPSILON
}

pub(crate) fn raycast_oriented_cuboid(
    origin: Vec3,
    direction: Vec3,
    center: Vec3,
    rotation: Quat,
    half_extents: Vec3,
) -> Option<OrientedCuboidHit> {
    if !origin.is_finite()
        || !direction.is_finite()
        || direction.length_squared() < f32::EPSILON
        || !center.is_finite()
        || !rotation.is_finite()
        || !half_extents.is_finite()
        || half_extents.cmple(Vec3::ZERO).any()
    {
        return None;
    }
    let direction = direction.normalize();
    let inverse_rotation = rotation.inverse();
    let local_origin = inverse_rotation * (origin - center);
    let local_direction = inverse_rotation * direction;
    let mut near = f32::NEG_INFINITY;
    let mut far = f32::INFINITY;
    let mut hit_axis = 0;
    let mut hit_sign = -1.0;

    for axis in 0..3 {
        if local_direction[axis].abs() <= f32::EPSILON {
            if local_origin[axis] < -half_extents[axis] || local_origin[axis] > half_extents[axis] {
                return None;
            }
            continue;
        }
        let inverse = local_direction[axis].recip();
        let first = (-half_extents[axis] - local_origin[axis]) * inverse;
        let second = (half_extents[axis] - local_origin[axis]) * inverse;
        let axis_near = first.min(second);
        let axis_far = first.max(second);
        if axis_near > near {
            near = axis_near;
            hit_axis = axis;
            hit_sign = if first < second { -1.0 } else { 1.0 };
        }
        far = far.min(axis_far);
        if near > far {
            return None;
        }
    }
    if far < 0.0 {
        return None;
    }
    let distance = near.max(0.0);
    let local_normal = Vec3::from_array(match hit_axis {
        0 => [hit_sign, 0.0, 0.0],
        1 => [0.0, hit_sign, 0.0],
        _ => [0.0, 0.0, hit_sign],
    });
    Some(OrientedCuboidHit {
        distance,
        point: origin + direction * distance,
        local_normal,
    })
}
