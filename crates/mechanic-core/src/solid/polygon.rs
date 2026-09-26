//! Polyhedral cells and the clipping, cleaning, and measuring primitives over them.

use super::model::{SurfacePatchKey, TopologyKey};
use crate::{ConvexFace, ConvexPiece};
use bevy_math::{DVec3, Vec3};
use std::collections::BTreeMap;

pub(super) const EPSILON: f64 = 1.0e-8;

// Independently clipped neighboring cells can differ below float render
// precision. Ten-micrometre keys stitch those seams without approaching the
// 2.5 mm authored position grid.
pub(super) const KEY_SCALE: f64 = 100_000.0;

#[derive(Clone, Debug)]
pub(super) struct PolyFace {
    pub(super) vertices: Vec<DVec3>,
    // Facet identity remains distinct for picking and UV provenance.
    pub(super) patch: SurfacePatchKey,
    // Family identity joins tessellated facets into one logical boundary.
    pub(super) family: SurfacePatchKey,
    pub(super) smoothing_group: u32,
    // Families joined tangentially to this facet are not sharp edges.
    pub(super) smooth_with: Vec<SurfacePatchKey>,
    pub(super) uv_provenance: SurfacePatchKey,
    // Lies inside a completely filled cell grid, either on a plane between two
    // cells or between two pieces of one cell. The other side covers the same
    // area however it happened to be split, so the face is interior without
    // having to match a twin polygon.
    pub(super) interior: bool,
}

#[derive(Clone, Debug)]
pub(super) struct PolyCell {
    pub(super) faces: Vec<PolyFace>,
    // Radial material band of a layered cylinder; zero everywhere else.
    pub(super) band: u8,
}

#[derive(Clone)]
pub(super) struct ClipPlane {
    pub(super) normal: DVec3,
    pub(super) offset: f64,
    pub(super) patch: SurfacePatchKey,
    pub(super) family: SurfacePatchKey,
    pub(super) smoothing_group: u32,
    pub(super) smooth_with: Vec<SurfacePatchKey>,
    pub(super) uv_provenance: SurfacePatchKey,
}

#[derive(Clone)]
pub(super) struct EdgeSegment {
    pub(super) key: TopologyKey,
    pub(super) half_edge: u32,
    pub(super) a: DVec3,
    pub(super) b: DVec3,
    pub(super) first_normal: DVec3,
    pub(super) second_normal: DVec3,
    pub(super) first_family: SurfacePatchKey,
    pub(super) second_family: SurfacePatchKey,
    pub(super) uv_provenance: SurfacePatchKey,
    pub(super) cell: usize,
    pub(super) convex: bool,
}

pub(super) fn clip_cell(cell: &PolyCell, plane: ClipPlane) -> Option<PolyCell> {
    let mut faces = Vec::new();
    let mut cap = Vec::<DVec3>::new();
    for face in &cell.faces {
        let mut polygon = Vec::new();
        for index in 0..face.vertices.len() {
            let current = face.vertices[index];
            let next = face.vertices[(index + 1) % face.vertices.len()];
            let current_distance = plane.normal.dot(current) - plane.offset;
            let next_distance = plane.normal.dot(next) - plane.offset;
            let current_inside = current_distance <= EPSILON;
            let next_inside = next_distance <= EPSILON;
            if current_inside {
                push_unique(&mut polygon, current);
            }
            if current_inside != next_inside {
                let fraction = current_distance / (current_distance - next_distance);
                let intersection = current.lerp(next, fraction);
                push_unique(&mut polygon, intersection);
                push_unique_global(&mut cap, intersection);
            }
        }
        clean_polygon(&mut polygon);
        if polygon.len() >= 3 {
            faces.push(PolyFace {
                vertices: polygon,
                patch: face.patch,
                family: face.family,
                smoothing_group: face.smoothing_group,
                smooth_with: face.smooth_with.clone(),
                uv_provenance: face.uv_provenance,
                interior: face.interior,
            });
        }
    }
    if cap.len() >= 3 {
        let center = cap.iter().copied().sum::<DVec3>() / cap.len() as f64;
        let tangent = plane.normal.any_orthonormal_vector();
        let bitangent = plane.normal.cross(tangent);
        cap.sort_by(|left, right| {
            let l = *left - center;
            let r = *right - center;
            l.dot(bitangent)
                .atan2(l.dot(tangent))
                .total_cmp(&r.dot(bitangent).atan2(r.dot(tangent)))
        });
        if polygon_normal(&cap).dot(plane.normal) < 0.0 {
            cap.reverse();
        }
        faces.push(PolyFace {
            vertices: cap,
            patch: plane.patch,
            family: plane.family,
            smoothing_group: plane.smoothing_group,
            smooth_with: plane.smooth_with,
            uv_provenance: plane.uv_provenance,
            interior: false,
        });
    }
    let result = PolyCell {
        faces,
        band: cell.band,
    };
    (result.faces.len() >= 4 && cell_volume(&result) > EPSILON).then_some(result)
}

/// Collapses vertices shared by a pinched profile, such as the crease of a
/// pipe bend whose inner wall meets at the centre of curvature.
pub(super) fn without_repeated_vertices(mut face: Vec<DVec3>) -> Vec<DVec3> {
    face.dedup_by(|next, previous| next.distance_squared(*previous) <= EPSILON * EPSILON);
    if face.len() > 1 && face[0].distance_squared(face[face.len() - 1]) <= EPSILON * EPSILON {
        face.pop();
    }
    face
}

pub(super) fn poly_cell_to_convex(cell: &PolyCell) -> Option<ConvexPiece> {
    let mut vertices = Vec::<Vec3>::new();
    let mut index_by_key = BTreeMap::<PointKey, u32>::new();
    let mut faces = Vec::new();
    for face in &cell.faces {
        let indices = face
            .vertices
            .iter()
            .map(|&point| {
                *index_by_key.entry(point_key(point)).or_insert_with(|| {
                    let index = u32::try_from(vertices.len()).unwrap_or(u32::MAX);
                    vertices.push(point.as_vec3());
                    index
                })
            })
            .collect::<Vec<_>>();
        let normal = polygon_normal(&face.vertices).as_vec3().normalize_or_zero();
        faces.push(ConvexFace {
            normal,
            offset: normal.dot(vertices[indices[0] as usize]),
            indices,
            grid_face: None,
        });
    }
    let mut edges = Vec::<Vec3>::new();
    for face in &faces {
        for index in 0..face.indices.len() {
            let a = vertices[face.indices[index] as usize];
            let b = vertices[face.indices[(index + 1) % face.indices.len()] as usize];
            let mut direction = (b - a).normalize_or_zero();
            if direction.x < -1.0e-6
                || (direction.x.abs() <= 1.0e-6
                    && (direction.y < -1.0e-6
                        || (direction.y.abs() <= 1.0e-6 && direction.z < 0.0)))
            {
                direction = -direction;
            }
            if direction != Vec3::ZERO
                && !edges
                    .iter()
                    .any(|other| other.abs_diff_eq(direction, 1.0e-5))
            {
                edges.push(direction);
            }
        }
    }
    let volume = cell_volume(cell);
    let centroid = cell_centroid(cell);
    (volume > EPSILON).then_some(ConvexPiece {
        vertices,
        faces,
        edge_directions: edges,
        centroid: centroid.as_vec3(),
        volume: volume as f32,
    })
}

pub(super) fn cell_volume(cell: &PolyCell) -> f64 {
    cell.faces
        .iter()
        .map(|face| {
            let anchor = face.vertices[0];
            (1..face.vertices.len() - 1)
                .map(|index| anchor.dot(face.vertices[index].cross(face.vertices[index + 1])) / 6.0)
                .sum::<f64>()
        })
        .sum::<f64>()
        .abs()
}

pub(super) fn cell_centroid(cell: &PolyCell) -> DVec3 {
    let reference = cell
        .faces
        .first()
        .and_then(|face| face.vertices.first())
        .copied()
        .unwrap_or(DVec3::ZERO);
    let mut volume = 0.0;
    let mut moment = DVec3::ZERO;
    for face in &cell.faces {
        for index in 1..face.vertices.len() - 1 {
            let a = face.vertices[0];
            let b = face.vertices[index];
            let c = face.vertices[index + 1];
            let signed = (a - reference).dot((b - reference).cross(c - reference)) / 6.0;
            volume += signed;
            moment += (reference + a + b + c) * (signed / 4.0);
        }
    }
    if volume.abs() > EPSILON {
        moment / volume
    } else {
        reference
    }
}

pub(super) fn polygon_normal(vertices: &[DVec3]) -> DVec3 {
    let mut normal = DVec3::ZERO;
    for index in 0..vertices.len() {
        let current = vertices[index];
        let next = vertices[(index + 1) % vertices.len()];
        normal += current.cross(next);
    }
    normal.normalize_or_zero()
}

pub(super) fn clean_polygon(polygon: &mut Vec<DVec3>) {
    if polygon.len() > 1
        && polygon[0].distance_squared(*polygon.last().unwrap()) <= EPSILON * EPSILON
    {
        polygon.pop();
    }
    let mut index = 0;
    while polygon.len() >= 3 && index < polygon.len() {
        let previous = polygon[(index + polygon.len() - 1) % polygon.len()];
        let current = polygon[index];
        let next = polygon[(index + 1) % polygon.len()];
        if (current - previous).cross(next - current).length_squared() <= EPSILON * EPSILON {
            polygon.remove(index);
        } else {
            index += 1;
        }
    }
}

pub(super) fn push_unique(points: &mut Vec<DVec3>, point: DVec3) {
    if points
        .last()
        .is_none_or(|last| last.distance_squared(point) > EPSILON * EPSILON)
    {
        points.push(point);
    }
}

pub(super) fn push_unique_global(points: &mut Vec<DVec3>, point: DVec3) {
    if !points
        .iter()
        .any(|other| other.distance_squared(point) <= EPSILON * EPSILON)
    {
        points.push(point);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct PointKey(pub(super) [i64; 3]);

pub(super) fn point_key(point: DVec3) -> PointKey {
    PointKey([
        (point.x * KEY_SCALE).round() as i64,
        (point.y * KEY_SCALE).round() as i64,
        (point.z * KEY_SCALE).round() as i64,
    ])
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct FaceSignature(pub(super) Vec<PointKey>);

pub(super) fn face_signature(vertices: &[DVec3]) -> FaceSignature {
    let mut keys = vertices.iter().copied().map(point_key).collect::<Vec<_>>();
    keys.sort_unstable();
    FaceSignature(keys)
}
