//! Replaying chamfers and fillets: edge profiles, junction planes, clipping, and clearance.

use super::model::{
    EdgeTreatment, ShapeFeature, SolidError, SurfacePatchKey, TopologyKey, TopologySource,
};
use super::polygon::{
    ClipPlane, EPSILON, EdgeSegment, PointKey, PolyCell, cell_volume, clip_cell, point_key,
    polygon_normal,
};
use super::stitch::{stitch, topology_word};
use crate::ShapeFeatureId;
use bevy_math::DVec3;
use std::collections::{BTreeMap, BTreeSet};

pub(super) const FILLET_MAX_FACET_DEGREES: f64 = 7.5;

pub(super) fn replay_features(
    mut cells: Vec<PolyCell>,
    features: impl IntoIterator<Item = (ShapeFeatureId, ShapeFeature)>,
) -> Result<Vec<PolyCell>, SolidError> {
    for (feature_id, feature) in features {
        if feature.amount_ticks == 0 {
            return Err(SolidError::ZeroAmount);
        }
        let (_, segments) = stitch(&cells)?;
        let selected = feature
            .targets
            .iter()
            .map(|target| target.edge)
            .collect::<BTreeSet<_>>();
        for key in &selected {
            if !segments.iter().any(|segment| segment.key == *key) {
                return Err(SolidError::MissingEdge {
                    feature: feature_id,
                    edge: *key,
                });
            }
            if segments
                .iter()
                .any(|segment| segment.key == *key && !segment.convex)
            {
                return Err(SolidError::NonConvexEdge(*key));
            }
        }
        let amount = f64::from(feature.amount_ticks) * f64::from(crate::POSITION_TICK_METERS);
        validate_feature_clearance(
            &cells,
            &segments,
            &selected,
            feature.treatment,
            amount,
            feature_id,
        )?;
        let mut planes_by_cell = BTreeMap::<usize, Vec<ClipPlane>>::new();
        let profiles = vertex_profiles(feature.treatment, amount, &segments, &selected);
        for segment in segments
            .iter()
            .filter(|segment| selected.contains(&segment.key))
        {
            append_edge_profile_planes(
                feature_id,
                feature.treatment,
                &profiles,
                segment,
                &mut planes_by_cell,
            );
        }
        if feature.treatment == EdgeTreatment::Fillet {
            append_fillet_junction_planes(
                feature_id,
                amount,
                &segments,
                &selected,
                &mut planes_by_cell,
            );
        }
        clip_feature_cells(&mut cells, planes_by_cell, feature_id)?;
        snap_to_profile_points(&mut cells, &profiles);
        cells.retain(|cell| cell_volume(cell) > EPSILON);
        if cells.is_empty() {
            return Err(SolidError::AmountTooLarge(feature_id));
        }
    }
    Ok(cells)
}

/// Treatment cross-sections shared by every segment meeting at a chain vertex,
/// keyed by segment half-edge and vertex.
pub(super) type VertexProfiles = BTreeMap<(u32, PointKey), Vec<DVec3>>;

/// Computes each selected segment's cross-section at both of its endpoints.
///
/// As in Blender's bevel, boundary points are placed once per vertex and both
/// strips meeting there reuse them. A chain crossing convex-cell seams, such as
/// a cylinder rim with one wedge cell per segment, is then cut on either side
/// of each seam along the same polyline, so the interior seam faces still
/// cancel when the boundary is stitched. Sampling the profile per segment
/// instead left each side with its own arc wherever exact symmetry was lost.
pub(super) fn vertex_profiles(
    treatment: EdgeTreatment,
    amount: f64,
    segments: &[EdgeSegment],
    selected: &BTreeSet<TopologyKey>,
) -> VertexProfiles {
    let mut incident = BTreeMap::<(TopologyKey, PointKey), Vec<(&EdgeSegment, DVec3)>>::new();
    for segment in segments
        .iter()
        .filter(|segment| selected.contains(&segment.key))
    {
        for point in [segment.a, segment.b] {
            incident
                .entry((segment.key, point_key(point)))
                .or_default()
                .push((segment, point));
        }
    }
    // One facet count per chain. Posed parts measure a right-angle rim a hair
    // either side of 90°, and vertices that disagreed on the count could not
    // be joined by strip facets.
    let mut chain_facets = BTreeMap::<TopologyKey, usize>::new();
    for segment in segments
        .iter()
        .filter(|segment| selected.contains(&segment.key))
    {
        let angle = segment
            .first_normal
            .dot(segment.second_normal)
            .clamp(-1.0, 1.0)
            .acos();
        let count = chain_facets.entry(segment.key).or_insert(1);
        *count = (*count).max(fillet_facets(angle));
    }
    // Inside a chain a vertex takes the mean of both segments' face normals,
    // which on a faceted rim is the true surface normal there.
    let mut chain_normals = BTreeMap::<(TopologyKey, PointKey), (DVec3, DVec3)>::new();
    for (&(key, vertex_key), uses) in &incident {
        if let [(first, _), (second, _)] = uses.as_slice() {
            let first_normal = (first.first_normal + second.first_normal).normalize_or_zero();
            let second_normal = (first.second_normal + second.second_normal).normalize_or_zero();
            if first_normal != DVec3::ZERO && second_normal != DVec3::ZERO {
                chain_normals.insert((key, vertex_key), (first_normal, second_normal));
            }
        }
    }
    let mut profiles = VertexProfiles::new();
    for ((key, vertex_key), uses) in incident {
        let facets = chain_facets.get(&key).copied().unwrap_or(1);
        for (segment, vertex) in uses {
            let (first_normal, second_normal) = chain_normals
                .get(&(key, vertex_key))
                .copied()
                .or_else(|| {
                    // An open chain ends at a vertex only one segment reaches.
                    // Continuing the normals' turn past that segment ends a
                    // curved rim on the surface normal at its last vertex, as
                    // a whole rim would, instead of on its last facet's normal.
                    let other = if point_key(segment.a) == vertex_key {
                        segment.b
                    } else {
                        segment.a
                    };
                    chain_normals
                        .get(&(key, point_key(other)))
                        .map(|&(first, second)| {
                            (
                                slerp_unit(first, segment.first_normal, 2.0),
                                slerp_unit(second, segment.second_normal, 2.0),
                            )
                        })
                })
                .unwrap_or((segment.first_normal, segment.second_normal));
            if let Some(profile) = vertex_profile(
                treatment,
                amount,
                facets,
                vertex,
                first_normal,
                second_normal,
            ) {
                profiles.insert((segment.half_edge, vertex_key), profile);
            }
        }
    }
    profiles
}

/// Cross-section of a treatment at `vertex`, from the first face to the second.
///
/// A fillet samples its circular arc from one face tangency to the other; a
/// chamfer is the straight cut between the two setback points.
/// `facets` is shared by the whole chain so neighbouring profiles pair up
/// step for step; chamfers ignore it.
pub(super) fn vertex_profile(
    treatment: EdgeTreatment,
    amount: f64,
    facets: usize,
    vertex: DVec3,
    first_normal: DVec3,
    second_normal: DVec3,
) -> Option<Vec<DVec3>> {
    let dot = first_normal.dot(second_normal).clamp(-1.0, 1.0);
    let angle = dot.acos();
    if !(1.0e-6..=core::f64::consts::PI - 1.0e-6).contains(&angle) {
        return None;
    }
    Some(match treatment {
        EdgeTreatment::Chamfer => {
            let first_inward = -(second_normal - first_normal * dot).normalize();
            let second_inward = -(first_normal - second_normal * dot).normalize();
            vec![
                vertex + first_inward * amount,
                vertex + second_inward * amount,
            ]
        }
        EdgeTreatment::Fillet => {
            let facets = facets.max(1);
            let centre = vertex - (first_normal + second_normal) * (amount / (1.0 + dot));
            (0..=facets)
                .map(|step| {
                    centre
                        + slerp_unit(first_normal, second_normal, step as f64 / facets as f64)
                            * amount
                })
                .collect()
        }
    })
}

/// Facets a fillet needs across a dihedral `angle`, tolerating float noise
/// so a right-angle edge always gets the same count.
pub(super) fn fillet_facets(angle: f64) -> usize {
    ((angle.to_degrees() / FILLET_MAX_FACET_DEGREES - 1.0e-6).ceil() as usize).max(1)
}

pub(super) fn append_edge_profile_planes(
    feature: ShapeFeatureId,
    treatment: EdgeTreatment,
    profiles: &VertexProfiles,
    segment: &EdgeSegment,
    planes_by_cell: &mut BTreeMap<usize, Vec<ClipPlane>>,
) {
    let (Some(start), Some(end)) = (
        profiles.get(&(segment.half_edge, point_key(segment.a))),
        profiles.get(&(segment.half_edge, point_key(segment.b))),
    ) else {
        return;
    };
    if start.len() != end.len() {
        return;
    }
    let steps = start.len() - 1;
    let outward = segment.first_normal + segment.second_normal;
    let planes = planes_by_cell.entry(segment.cell).or_default();
    for step_index in 0..steps {
        let profile_step = step_index + 1;
        let smooth_with: Vec<_> = if treatment == EdgeTreatment::Fillet {
            [
                (step_index == 0).then_some(segment.first_family),
                (step_index + 1 == steps).then_some(segment.second_family),
            ]
            .into_iter()
            .flatten()
            .collect()
        } else {
            Vec::new()
        };
        let facet = [
            start[step_index],
            start[step_index + 1],
            end[step_index + 1],
            end[step_index],
        ];
        for (normal, offset) in strip_facet_planes(facet, outward) {
            planes.push(ClipPlane {
                normal,
                offset,
                patch: generated_patch_key(feature, segment, profile_step),
                family: generated_patch_family_key(feature, segment.key, profile_step),
                smoothing_group: u32::from(treatment == EdgeTreatment::Fillet)
                    * feature.index().saturating_add(1),
                smooth_with: smooth_with.clone(),
                uv_provenance: segment.uv_provenance,
            });
        }
    }
}

/// Outward clip planes through one strip quad between two vertex profiles.
///
/// Rotational and straight chains give planar quads. Otherwise the quad is
/// split along the diagonal that keeps the pair convex seen from outside, so
/// the solid remains the intersection of the half-spaces.
pub(super) fn strip_facet_planes(facet: [DVec3; 4], outward: DVec3) -> Vec<(DVec3, f64)> {
    let plane = |points: [DVec3; 3]| {
        let normal = (points[1] - points[0])
            .cross(points[2] - points[0])
            .normalize_or_zero();
        let normal = if normal.dot(outward) < 0.0 {
            -normal
        } else {
            normal
        };
        (normal, normal.dot(points[0]))
    };
    let newell = (0..4).fold(DVec3::ZERO, |sum, index| {
        let current = facet[index];
        let next = facet[(index + 1) % 4];
        sum + DVec3::new(
            (current.y - next.y) * (current.z + next.z),
            (current.z - next.z) * (current.x + next.x),
            (current.x - next.x) * (current.y + next.y),
        )
    });
    if let Some(normal) = newell.try_normalize() {
        let normal = if normal.dot(outward) < 0.0 {
            -normal
        } else {
            normal
        };
        let offset = facet.iter().map(|point| normal.dot(*point)).sum::<f64>() * 0.25;
        // Well under one stitch key, so a plane this far from a corner still
        // meets the neighbouring cell's cut at the same key.
        if facet
            .iter()
            .all(|point| (normal.dot(*point) - offset).abs() <= 1.0e-6)
        {
            return vec![(normal, offset)];
        }
    }
    for [first, second] in [[[0, 1, 2], [0, 2, 3]], [[0, 1, 3], [1, 2, 3]]] {
        let triangles = [first, second].map(|indices| plane(indices.map(|index| facet[index])));
        let opposite = [
            facet[(0..4).find(|index| !first.contains(index)).unwrap_or(0)],
            facet[(0..4).find(|index| !second.contains(index)).unwrap_or(0)],
        ];
        if triangles
            .iter()
            .zip(opposite)
            .all(|((normal, offset), point)| normal.dot(point) - offset <= 1.0e-9)
        {
            return triangles
                .into_iter()
                .filter(|(normal, _)| *normal != DVec3::ZERO)
                .collect();
        }
    }
    Vec::new()
}

pub(super) fn append_fillet_junction_planes(
    feature: ShapeFeatureId,
    radius: f64,
    segments: &[EdgeSegment],
    selected: &BTreeSet<TopologyKey>,
    planes_by_cell: &mut BTreeMap<usize, Vec<ClipPlane>>,
) {
    let mut incident = BTreeMap::<(usize, PointKey), Vec<&EdgeSegment>>::new();
    for segment in segments
        .iter()
        .filter(|segment| selected.contains(&segment.key))
    {
        for point in [segment.a, segment.b] {
            incident
                .entry((segment.cell, point_key(point)))
                .or_default()
                .push(segment);
        }
    }
    for ((cell, vertex_key), incident_edges) in incident {
        if distinct_edge_count(&incident_edges) < 3 {
            continue;
        }
        let mut normals = BTreeMap::<PointKey, DVec3>::new();
        for edge in &incident_edges {
            for normal in [edge.first_normal, edge.second_normal] {
                normals.entry(point_key(normal)).or_insert(normal);
            }
        }
        let normals = normals.into_values().collect::<Vec<_>>();
        if normals.len() != 3 {
            continue;
        }
        let vertex = incident_edges[0].a;
        let vertex = if point_key(vertex) == vertex_key {
            vertex
        } else {
            incident_edges[0].b
        };
        let [first, second, third] = [normals[0], normals[1], normals[2]];
        let smooth_with = incident_surface_families(&incident_edges);
        let determinant = first.dot(second.cross(third));
        if determinant.abs() <= EPSILON {
            continue;
        }
        let distances = [
            first.dot(vertex) - radius,
            second.dot(vertex) - radius,
            third.dot(vertex) - radius,
        ];
        let centre = (distances[0] * second.cross(third)
            + distances[1] * third.cross(first)
            + distances[2] * first.cross(second))
            / determinant;
        let maximum_angle = first
            .dot(second)
            .min(second.dot(third))
            .min(third.dot(first))
            .clamp(-1.0, 1.0)
            .acos();
        let steps =
            ((maximum_angle.to_degrees() / FILLET_MAX_FACET_DEGREES).ceil() as usize).max(1);
        let direction = |first_weight: usize, second_weight: usize| {
            let third_weight = steps - first_weight - second_weight;
            (first * first_weight as f64
                + second * second_weight as f64
                + third * third_weight as f64)
                .normalize()
        };
        let mut ordinal = 0_usize;
        for first_weight in 0..steps {
            for second_weight in 0..steps - first_weight {
                let a = direction(first_weight, second_weight);
                let b = direction(first_weight + 1, second_weight);
                let c = direction(first_weight, second_weight + 1);
                push_fillet_junction_plane(
                    feature,
                    vertex,
                    centre,
                    radius,
                    [a, b, c],
                    ordinal,
                    incident_edges[0].uv_provenance,
                    &smooth_with,
                    planes_by_cell.entry(cell).or_default(),
                );
                ordinal += 1;
                if first_weight + second_weight + 1 < steps {
                    let d = direction(first_weight + 1, second_weight + 1);
                    push_fillet_junction_plane(
                        feature,
                        vertex,
                        centre,
                        radius,
                        [b, d, c],
                        ordinal,
                        incident_edges[0].uv_provenance,
                        &smooth_with,
                        planes_by_cell.entry(cell).or_default(),
                    );
                    ordinal += 1;
                }
            }
        }
    }
}

pub(super) fn distinct_edge_count(segments: &[&EdgeSegment]) -> usize {
    segments
        .iter()
        .map(|edge| edge.key)
        .collect::<BTreeSet<_>>()
        .len()
}

pub(super) fn incident_surface_families(segments: &[&EdgeSegment]) -> Vec<SurfacePatchKey> {
    segments
        .iter()
        .flat_map(|edge| [edge.first_family, edge.second_family])
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[expect(clippy::too_many_arguments)]
pub(super) fn push_fillet_junction_plane(
    feature: ShapeFeatureId,
    vertex: DVec3,
    centre: DVec3,
    radius: f64,
    directions: [DVec3; 3],
    ordinal: usize,
    uv_provenance: SurfacePatchKey,
    smooth_with: &[SurfacePatchKey],
    planes: &mut Vec<ClipPlane>,
) {
    let points = directions.map(|direction| centre + direction * radius);
    let mut normal = (points[1] - points[0])
        .cross(points[2] - points[0])
        .normalize();
    if normal.dot(directions.into_iter().sum()) < 0.0 {
        normal = -normal;
    }
    planes.push(ClipPlane {
        normal,
        offset: normal.dot(points[0]),
        patch: generated_junction_patch_key(feature, vertex, ordinal),
        family: generated_junction_family_key(feature, vertex),
        smoothing_group: feature.index().saturating_add(1),
        smooth_with: smooth_with.to_vec(),
        uv_provenance,
    });
}

pub(super) fn generated_junction_family_key(
    feature: ShapeFeatureId,
    vertex: DVec3,
) -> SurfacePatchKey {
    let mut hash = 2_166_136_261_u32;
    for value in [0x4a46_414d, point_word(point_key(vertex))] {
        hash ^= value;
        hash = hash.wrapping_mul(16_777_619);
    }
    SurfacePatchKey {
        source: TopologySource::Feature(feature),
        local: hash,
    }
}

pub(super) fn generated_junction_patch_key(
    feature: ShapeFeatureId,
    vertex: DVec3,
    ordinal: usize,
) -> SurfacePatchKey {
    let mut hash = 2_166_136_261_u32;
    for value in [
        0x4a55_4e43,
        point_word(point_key(vertex)),
        u32::try_from(ordinal).unwrap_or(u32::MAX),
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(16_777_619);
    }
    SurfacePatchKey {
        source: TopologySource::Feature(feature),
        local: hash,
    }
}

/// Welds clipped vertices onto the vertex-profile points they were cut through.
///
/// Neighbouring cells reach the same profile point through different plane
/// intersections, which can land either side of a stitch-key boundary. Blender
/// avoids this by reusing one vertex; snapping both copies onto the shared
/// point does the same here.
pub(super) fn snap_to_profile_points(cells: &mut [PolyCell], profiles: &VertexProfiles) {
    const SNAP_DISTANCE: f64 = 1.0e-6;
    let mut points = BTreeMap::<PointKey, Vec<DVec3>>::new();
    for &point in profiles.values().flatten() {
        let bucket = points.entry(point_key(point)).or_default();
        if !bucket
            .iter()
            .any(|existing| existing.distance_squared(point) <= EPSILON * EPSILON)
        {
            bucket.push(point);
        }
    }
    for vertex in cells
        .iter_mut()
        .flat_map(|cell| cell.faces.iter_mut())
        .flat_map(|face| face.vertices.iter_mut())
    {
        let [x, y, z] = point_key(*vertex).0;
        let nearest = (-1..=1)
            .flat_map(|dx| (-1..=1).flat_map(move |dy| (-1..=1).map(move |dz| [dx, dy, dz])))
            .filter_map(|[dx, dy, dz]| points.get(&PointKey([x + dx, y + dy, z + dz])))
            .flatten()
            .copied()
            .filter(|point| point.distance_squared(*vertex) <= SNAP_DISTANCE * SNAP_DISTANCE)
            .min_by(|left, right| {
                left.distance_squared(*vertex)
                    .total_cmp(&right.distance_squared(*vertex))
            });
        if let Some(point) = nearest {
            *vertex = point;
        }
    }
}

pub(super) fn clip_feature_cells(
    cells: &mut [PolyCell],
    planes_by_cell: BTreeMap<usize, Vec<ClipPlane>>,
    feature_id: ShapeFeatureId,
) -> Result<(), SolidError> {
    for (cell_index, planes) in planes_by_cell {
        let Some(cell) = cells.get_mut(cell_index) else {
            continue;
        };
        for plane in planes {
            *cell = clip_cell(cell, plane).ok_or(SolidError::AmountTooLarge(feature_id))?;
        }
    }
    Ok(())
}

pub(super) fn validate_feature_clearance(
    cells: &[PolyCell],
    segments: &[EdgeSegment],
    selected: &BTreeSet<TopologyKey>,
    treatment: EdgeTreatment,
    amount: f64,
    feature: ShapeFeatureId,
) -> Result<(), SolidError> {
    for segment in segments
        .iter()
        .filter(|segment| selected.contains(&segment.key))
    {
        let dot = segment
            .first_normal
            .dot(segment.second_normal)
            .clamp(-1.0, 1.0);
        let angle = dot.acos();
        if !(1.0e-6..=core::f64::consts::PI - 1.0e-6).contains(&angle) {
            continue;
        }
        let setback = match treatment {
            EdgeTreatment::Chamfer => amount,
            EdgeTreatment::Fillet => amount * (angle * 0.5).tan(),
        };
        let first_inward = -(segment.second_normal - segment.first_normal * dot).normalize();
        let second_inward = -(segment.first_normal - segment.second_normal * dot).normalize();
        let Some(cell) = cells.get(segment.cell) else {
            return Err(SolidError::AmountTooLarge(feature));
        };
        let first_clearance = face_clearance(cell, segment, segment.first_normal, first_inward);
        let second_clearance = face_clearance(cell, segment, segment.second_normal, second_inward);
        if setback > first_clearance + EPSILON || setback > second_clearance + EPSILON {
            return Err(SolidError::AmountTooLarge(feature));
        }
    }
    Ok(())
}

pub(super) fn face_clearance(
    cell: &PolyCell,
    segment: &EdgeSegment,
    normal: DVec3,
    inward: DVec3,
) -> f64 {
    cell.faces
        .iter()
        .filter(|face| polygon_normal(&face.vertices).dot(normal) >= 1.0 - 1.0e-6)
        .filter(|face| {
            face.vertices
                .iter()
                .all(|vertex| normal.dot(*vertex - segment.a).abs() <= 1.0e-6)
        })
        .flat_map(|face| face.vertices.iter())
        .map(|vertex| inward.dot(*vertex - segment.a))
        .fold(0.0, f64::max)
}

pub(super) fn generated_patch_key(
    feature: ShapeFeatureId,
    segment: &EdgeSegment,
    step: usize,
) -> SurfacePatchKey {
    let mut hash = 2_166_136_261_u32;
    for value in [
        segment.key.local,
        u32::try_from(step).unwrap_or(u32::MAX),
        point_word(point_key(segment.a)),
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(16_777_619);
    }
    SurfacePatchKey {
        source: TopologySource::Feature(feature),
        local: hash,
    }
}

pub(super) fn generated_patch_family_key(
    feature: ShapeFeatureId,
    target: TopologyKey,
    step: usize,
) -> SurfacePatchKey {
    let mut hash = 2_166_136_261_u32;
    for value in [
        0x4641_4d49,
        topology_word(target),
        target.local,
        u32::try_from(step).unwrap_or(u32::MAX),
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(16_777_619);
    }
    SurfacePatchKey {
        source: TopologySource::Feature(feature),
        local: hash,
    }
}

pub(super) fn point_word(point: PointKey) -> u32 {
    point.0.into_iter().fold(0_u32, |hash, value| {
        let bytes = value.cast_unsigned().to_le_bytes();
        let low = u32::from_le_bytes(bytes[..4].try_into().expect("four low bytes"));
        let high = u32::from_le_bytes(bytes[4..].try_into().expect("four high bytes"));
        hash.rotate_left(5) ^ low ^ high
    })
}

pub(super) fn slerp_unit(a: DVec3, b: DVec3, fraction: f64) -> DVec3 {
    let angle = a.dot(b).clamp(-1.0, 1.0).acos();
    if angle < EPSILON {
        return a;
    }
    ((a * ((1.0 - fraction) * angle).sin() + b * (fraction * angle).sin()) / angle.sin())
        .normalize()
}
