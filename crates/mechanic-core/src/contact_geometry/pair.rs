//! Convex/convex separation and reference faces for contacts between two bodies.
//!
//! Terrain supplies finite triangles; another collider supplies a closed convex
//! solid. One separating axis per pair selects a single contact normal, so a box
//! resting on another box never also reports its side faces as horizontal
//! supports, which clipping against every face triangle would.

use super::{ContactGeometryError, ContactPolytope, contact_tangents, project};
use bevy_math::DVec3;

// An edge axis replaces the best face axis only when it separates clearly
// further. Nearly parallel resting faces then keep one stable face normal.
const EDGE_AXIS_BIAS: f64 = 1e-6;

// A winning edge axis within about 2.6 degrees of a face normal is a face lying
// across the other solid's edge, tilted by drift: a box hanging over a ledge.
// Its support is the whole clipped face, not one point that flips between the
// face's two crossing edges every tick and lets the box rock into the ledge.
const EDGE_FACE_ALIGNMENT: f64 = 0.999;

/// Feature that realizes the largest separating-axis gap between two solids.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConvexFeature {
    /// Face plane of the receiving polytope, by plane row.
    OwnFace(usize),
    /// Face plane of the other polytope, by plane row.
    OtherFace(usize),
    /// Closest points on one edge of each polytope, receiving polytope first.
    Edges([DVec3; 2]),
}

/// Largest separating-axis gap between two convex solids in one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConvexSeparation {
    /// Unit axis pointing from the other polytope toward the receiving one.
    pub axis: DVec3,
    /// Positive values bound the distance from below; nonpositive means overlap,
    /// and its magnitude is the overlap along `axis`.
    pub separation: f64,
    /// Feature realizing the gap.
    pub feature: ConvexFeature,
}

impl ContactPolytope {
    /// Separating-axis test against another convex solid over both face sets
    /// and every edge-pair cross product. Face axes are preferred unless an edge
    /// axis separates clearly further, so resting faces keep a stable normal.
    ///
    /// # Errors
    /// Rejects non-finite geometry, or an edge axis without a representable edge.
    pub fn convex_separation(
        &self,
        other: &Self,
    ) -> Result<ConvexSeparation, ContactGeometryError> {
        let mut face: Option<ConvexSeparation> = None;
        let mut consider_face = |axis: DVec3, separation: f64, feature| {
            if face.is_none_or(|best| separation > best.separation) {
                face = Some(ConvexSeparation {
                    axis,
                    separation,
                    feature,
                });
            }
        };
        for (row, plane) in self.planes.iter().enumerate() {
            let Some(normal) = plane.truncate().try_normalize() else {
                return Err(ContactGeometryError);
            };
            let gap = project(&other.vertices, normal)[0] - project(&self.vertices, normal)[1];
            consider_face(-normal, gap, ConvexFeature::OwnFace(row));
        }
        for (row, plane) in other.planes.iter().enumerate() {
            let Some(normal) = plane.truncate().try_normalize() else {
                return Err(ContactGeometryError);
            };
            let gap = project(&self.vertices, normal)[0] - project(&other.vertices, normal)[1];
            consider_face(normal, gap, ConvexFeature::OtherFace(row));
        }
        let face = face.ok_or(ContactGeometryError)?;
        let mut edge: Option<(DVec3, f64, usize, usize)> = None;
        for (own_row, own) in self.edges.iter().enumerate() {
            for (other_row, theirs) in other.edges.iter().enumerate() {
                let cross = own.normalize().cross(theirs.normalize());
                // Near-parallel edges have no well-defined cross axis; their
                // separating directions are already among the face normals.
                if cross.length_squared() < 1e-12 {
                    continue;
                }
                let axis = cross.normalize();
                let own_range = project(&self.vertices, axis);
                let other_range = project(&other.vertices, axis);
                for (axis, gap) in [
                    (axis, own_range[0] - other_range[1]),
                    (-axis, other_range[0] - own_range[1]),
                ] {
                    if edge.is_none_or(|best| gap > best.1) {
                        edge = Some((axis, gap, own_row, other_row));
                    }
                }
            }
        }
        if !face.separation.is_finite() || !face.axis.is_finite() {
            return Err(ContactGeometryError);
        }
        let Some((axis, separation, own_row, other_row)) = edge else {
            return Ok(face);
        };
        if separation <= face.separation + EDGE_AXIS_BIAS {
            return Ok(face);
        }
        // Keep the edge axis and its tighter gap for distance bounds; only the
        // manifold comes from the aligned face.
        let aligned =
            self.planes
                .iter()
                .enumerate()
                .map(|(row, plane)| (-plane.truncate().normalize(), ConvexFeature::OwnFace(row)))
                .chain(other.planes.iter().enumerate().map(|(row, plane)| {
                    (plane.truncate().normalize(), ConvexFeature::OtherFace(row))
                }))
                .map(|(normal, feature)| (normal.dot(axis), feature))
                .filter(|(alignment, _)| *alignment >= EDGE_FACE_ALIGNMENT)
                .max_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((_, feature)) = aligned {
            return Ok(ConvexSeparation {
                axis,
                separation,
                feature,
            });
        }
        let own = support_edge(self, -axis, self.edges[own_row]);
        let theirs = support_edge(other, axis, other.edges[other_row]);
        let (Some(own), Some(theirs)) = (own, theirs) else {
            // A vertex, not an edge, realizes this axis on one side: the face
            // feature still describes that contact without inventing an edge.
            return Ok(face);
        };
        Ok(ConvexSeparation {
            axis,
            separation,
            feature: ConvexFeature::Edges(closest_segment_points(own, theirs)),
        })
    }

    /// Fan triangles of one face polygon, wound so each triangle's normal is the
    /// face's outward normal. The polygon is the vertex set on that face plane.
    ///
    /// # Errors
    /// Rejects an unknown plane, or a face with fewer than three distinct vertices.
    pub fn face_triangles(&self, plane: usize) -> Result<Vec<[DVec3; 3]>, ContactGeometryError> {
        let normal = self
            .planes
            .get(plane)
            .and_then(|plane| plane.truncate().try_normalize())
            .ok_or(ContactGeometryError)?;
        let support = project(&self.vertices, normal)[1];
        let scale = self
            .vertices
            .iter()
            .map(|vertex| vertex.abs().max_element())
            .fold(1.0, f64::max);
        // Compiled faces and vertices are rounded independently from f32 data.
        let tolerance = 1e-6 * scale;
        let mut polygon = Vec::<DVec3>::new();
        for &vertex in &self.vertices {
            if vertex.dot(normal) >= support - tolerance
                && polygon.iter().all(|kept| kept.distance(vertex) > tolerance)
            {
                polygon.push(vertex);
            }
        }
        if polygon.len() < 3 {
            return Err(ContactGeometryError);
        }
        let count = u32::try_from(polygon.len()).map_err(|_| ContactGeometryError)?;
        let center = polygon.iter().copied().sum::<DVec3>() / f64::from(count);
        let [u, v] = contact_tangents(normal);
        polygon.sort_by(|a, b| {
            let angle = |point: &DVec3| (*point - center).dot(v).atan2((*point - center).dot(u));
            angle(a).total_cmp(&angle(b))
        });
        // `u × v` is the normal, so increasing angle winds counter-clockwise
        // about it and every fan triangle faces outward.
        Ok((1..polygon.len() - 1)
            .map(|index| [polygon[0], polygon[index], polygon[index + 1]])
            .filter(|triangle| {
                (triangle[1] - triangle[0])
                    .cross(triangle[2] - triangle[0])
                    .dot(normal)
                    > 0.0
            })
            .collect())
    }

    /// Upper bound on overlap between two convex solids anywhere along a path on
    /// which their relative point displacement stays within `displacement`.
    /// Their separating-axis gap moves by at most that much along a fixed axis,
    /// so None proves they cannot intersect on the path.
    /// This validation bound must never generate contact impulses.
    ///
    /// # Errors
    /// Rejects a negative or non-finite displacement, or invalid geometry.
    pub fn convex_penetration_bound(
        &self,
        other: &Self,
        displacement: f64,
    ) -> Result<Option<f64>, ContactGeometryError> {
        if !displacement.is_finite() || displacement < 0.0 {
            return Err(ContactGeometryError);
        }
        let separation = self.convex_separation(other)?.separation;
        if separation > displacement.next_up() {
            return Ok(None);
        }
        Ok(Some((displacement - separation).next_up().max(0.0)))
    }
}

// The two vertices of `polytope` farthest along `direction` that span an edge
// parallel to `edge`. Several supporting vertices mean a face, not an edge.
fn support_edge(polytope: &ContactPolytope, direction: DVec3, edge: DVec3) -> Option<[DVec3; 2]> {
    let support = project(&polytope.vertices, direction)[1];
    let scale = polytope
        .vertices
        .iter()
        .map(|vertex| vertex.abs().max_element())
        .fold(1.0, f64::max);
    let tolerance = 1e-6 * scale;
    let supporting = polytope
        .vertices
        .iter()
        .copied()
        .filter(|vertex| vertex.dot(direction) >= support - tolerance)
        .collect::<Vec<_>>();
    let edge = edge.try_normalize()?;
    let mut best: Option<[DVec3; 2]> = None;
    for (index, &a) in supporting.iter().enumerate() {
        for &b in &supporting[index + 1..] {
            let span = b - a;
            if span.length() > tolerance
                && span.normalize().cross(edge).length() < 1e-6
                && best.is_none_or(|[c, d]| span.length() > c.distance(d))
            {
                best = Some([a, b]);
            }
        }
    }
    best
}

// Closest points between two segments, first segment's point first.
fn closest_segment_points(first: [DVec3; 2], second: [DVec3; 2]) -> [DVec3; 2] {
    let along_first = first[1] - first[0];
    let along_second = second[1] - second[0];
    let offset = first[0] - second[0];
    let first_length = along_first.length_squared();
    let second_length = along_second.length_squared();
    let second_offset = along_second.dot(offset);
    let first_offset = along_first.dot(offset);
    let alignment = along_first.dot(along_second);
    let denominator = first_length * second_length - alignment * alignment;
    let mut on_first = if denominator > 0.0 {
        ((alignment * second_offset - first_offset * second_length) / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut on_second = (alignment * on_first + second_offset) / second_length;
    if on_second < 0.0 {
        on_second = 0.0;
        on_first = (-first_offset / first_length).clamp(0.0, 1.0);
    } else if on_second > 1.0 {
        on_second = 1.0;
        on_first = ((alignment - first_offset) / first_length).clamp(0.0, 1.0);
    }
    [
        first[0] + along_first * on_first,
        second[0] + along_second * on_second,
    ]
}

#[cfg(test)]
mod tests {
    use super::super::tests::cube;
    use super::*;
    use bevy_math::DQuat;

    #[test]
    fn resting_boxes_are_separated_by_one_of_their_touching_faces() {
        let lower = cube();
        let upper = cube()
            .transformed(DVec3::Y * 0.999, DQuat::IDENTITY)
            .unwrap();
        let separation = upper.convex_separation(&lower).unwrap();
        assert!((separation.separation + 0.001).abs() < 1e-12);
        assert!(separation.axis.abs_diff_eq(DVec3::Y, 1e-12));
        // Both touching faces realize the same gap; either is a valid reference,
        // and its outward normal must point at the other box.
        let (triangles, outward, level) = match separation.feature {
            ConvexFeature::OtherFace(plane) => (lower.face_triangles(plane), DVec3::Y, 0.5),
            ConvexFeature::OwnFace(plane) => (upper.face_triangles(plane), -DVec3::Y, 0.499),
            ConvexFeature::Edges(_) => panic!("parallel faces cannot select an edge axis"),
        };
        let triangles = triangles.unwrap();
        assert_eq!(triangles.len(), 2);
        for triangle in triangles {
            let normal = (triangle[1] - triangle[0]).cross(triangle[2] - triangle[0]);
            assert!(normal.normalize().abs_diff_eq(outward, 1e-12));
            assert!(triangle.iter().all(|point| (point.y - level).abs() < 1e-12));
        }
        let reversed = lower.convex_separation(&upper).unwrap();
        assert!((reversed.separation + 0.001).abs() < 1e-12);
        assert!(reversed.axis.abs_diff_eq(-DVec3::Y, 1e-12));
        // Six faces give six triangulated quads, each facing outward.
        for plane in 0..6 {
            let normal = lower.planes[plane].truncate();
            for triangle in lower.face_triangles(plane).unwrap() {
                let winding = (triangle[1] - triangle[0]).cross(triangle[2] - triangle[0]);
                assert!(winding.normalize().abs_diff_eq(normal, 1e-12));
            }
        }
        assert!(lower.face_triangles(6).is_err());
    }

    #[test]
    fn crossed_edges_meet_at_one_point_on_their_cross_axis() {
        let lower = cube()
            .transformed(
                DVec3::ZERO,
                DQuat::from_rotation_x(std::f64::consts::FRAC_PI_4),
            )
            .unwrap();
        let half_diagonal = std::f64::consts::FRAC_1_SQRT_2;
        let upper = cube()
            .transformed(
                DVec3::Y * (2.0 * half_diagonal + 0.01),
                DQuat::from_rotation_z(std::f64::consts::FRAC_PI_4),
            )
            .unwrap();
        let separation = upper.convex_separation(&lower).unwrap();
        assert!((separation.separation - 0.01).abs() < 1e-9);
        assert!(separation.axis.abs_diff_eq(DVec3::Y, 1e-9));
        let ConvexFeature::Edges([own, theirs]) = separation.feature else {
            panic!("expected crossed edges, got {:?}", separation.feature);
        };
        assert!(own.abs_diff_eq(DVec3::Y * (half_diagonal + 0.01), 1e-9));
        assert!(theirs.abs_diff_eq(DVec3::Y * half_diagonal, 1e-9));
    }

    #[test]
    fn penetration_bound_certifies_clear_paths_and_encloses_overlap() {
        let lower = cube();
        let apart = cube().transformed(DVec3::X * 1.1, DQuat::IDENTITY).unwrap();
        assert_eq!(apart.convex_penetration_bound(&lower, 0.05).unwrap(), None);
        let bound = apart
            .convex_penetration_bound(&lower, 0.15)
            .unwrap()
            .unwrap();
        assert!((bound - 0.05).abs() < 1e-12);
        let overlapping = cube()
            .transformed(DVec3::X * 0.99, DQuat::IDENTITY)
            .unwrap();
        let bound = overlapping
            .convex_penetration_bound(&lower, 0.0)
            .unwrap()
            .unwrap();
        assert!((bound - 0.01).abs() < 1e-12);
        assert!(apart.convex_penetration_bound(&lower, -1.0).is_err());
    }

    #[test]
    fn a_slightly_tilted_box_over_a_ledge_rests_on_a_face_not_a_crossed_edge() {
        let lower = cube();
        // Hangs a third off the lower box's +x edge, sunk 1 mm at that edge and
        // rolled 3 mrad so its overhanging side dips below the lower top face.
        let tilt = DQuat::from_rotation_z(-0.003);
        let upper = cube()
            .transformed(DVec3::new(0.35, 0.999 + 0.15 * 0.003, 0.0), tilt)
            .unwrap();
        let separation = upper.convex_separation(&lower).unwrap();
        let (ConvexFeature::OwnFace(plane) | ConvexFeature::OtherFace(plane)) = separation.feature
        else {
            panic!("{separation:?}");
        };
        assert!(separation.axis.y > 0.999, "{separation:?}");
        assert!(separation.separation < 0.0 && separation.separation > -0.002);
        let (face, _) = match separation.feature {
            ConvexFeature::OwnFace(_) => (&upper, &lower),
            _ => (&lower, &upper),
        };
        assert_eq!(face.face_triangles(plane).unwrap().len(), 2);
    }

    #[test]
    fn solids_moving_together_certify_the_whole_interval_across_a_tiny_gap() {
        use crate::ContactVelocity;
        let lower = cube();
        let resting = cube()
            .transformed(DVec3::Y * (1.0 + 2e-12), DQuat::IDENTITY)
            .unwrap();
        let sliding = ContactVelocity::new(DVec3::new(3.0, -0.5, 0.0), DVec3::ZERO);
        let prefix = resting
            .convex_motion_prefix(DVec3::Y, sliding, &lower, DVec3::ZERO, sliding, 0.0, 1.0)
            .unwrap();
        assert_eq!(prefix.to_bits(), 1.0_f64.to_bits(), "{prefix}");
        let falling = cube().transformed(DVec3::Y * 1.1, DQuat::IDENTITY).unwrap();
        let prefix = falling
            .convex_motion_prefix(
                DVec3::Y * 1.1,
                ContactVelocity::new(-DVec3::Y, DVec3::ZERO),
                &lower,
                DVec3::ZERO,
                ContactVelocity::new(DVec3::ZERO, DVec3::ZERO),
                0.0,
                1.0,
            )
            .unwrap();
        assert!(prefix < 0.1 && prefix > 0.1 - 1e-9, "{prefix}");
        let touching = cube().transformed(DVec3::Y, DQuat::IDENTITY).unwrap();
        let still = ContactVelocity::new(DVec3::ZERO, DVec3::ZERO);
        assert_eq!(
            touching
                .convex_motion_prefix(DVec3::Y, still, &lower, DVec3::ZERO, still, 0.0, 1.0)
                .unwrap()
                .to_bits(),
            0.0_f64.to_bits()
        );
    }
}
