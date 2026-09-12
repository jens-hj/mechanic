//! Numerical-zero activation from actual hull vertices when face-ray clipping is
//! ill conditioned. A vertex is never moved toward a supporting infinite plane.

use super::{
    ContactGeometryError, ContactPolytope, TriangleContactPoint, contact_tangents,
    envelope::interval::Interval, triangle_normal,
};
use bevy_math::DVec3;

impl ContactPolytope {
    pub(super) fn vertex_activation(
        &self,
        triangle: [DVec3; 3],
        maximum_gap: f64,
    ) -> Result<Vec<TriangleContactPoint>, ContactGeometryError> {
        let normal = triangle_normal(triangle)?;
        let mut candidates = Vec::new();
        for &body_point in &self.vertices {
            let distance = (body_point - triangle[0]).dot(normal);
            if !(-maximum_gap..=maximum_gap).contains(&distance) {
                continue;
            }
            let triangle_point = body_point - distance * normal;
            let measured_gap = (body_point - triangle_point).dot(normal);
            if (-maximum_gap..=maximum_gap).contains(&measured_gap)
                && inside_triangle(triangle, triangle_point, normal)
            {
                candidates.push(TriangleContactPoint {
                    triangle_point,
                    body_point,
                    normal,
                    depth: (-measured_gap).max(0.0),
                });
            }
        }
        if candidates.is_empty() {
            // No certified vertex is not proof that the face query was empty.
            return Err(ContactGeometryError);
        }
        let [u, v] = contact_tangents(normal);
        let mut contacts: Vec<TriangleContactPoint> = Vec::with_capacity(4);
        for direction in [u + v, u - v, -u - v, -u + v] {
            let mut selected = candidates[0];
            for &candidate in &candidates[1..] {
                if candidate.triangle_point.dot(direction) > selected.triangle_point.dot(direction)
                {
                    selected = candidate;
                }
            }
            if !contacts
                .iter()
                .any(|p| p.triangle_point.distance(selected.triangle_point) < 1e-5)
            {
                contacts.push(selected);
            }
        }
        Ok(contacts)
    }
}

// All three oriented edge predicates must have nonnegative lower bounds.
// An uncertain edge is rejected, preserving finite holes and triangle boundaries.
pub(super) fn inside_triangle(triangle: [DVec3; 3], point: DVec3, normal: DVec3) -> bool {
    (0..3).all(|edge| {
        let a = triangle[edge];
        let b = triangle[(edge + 1) % 3];
        let direction: [Interval; 3] =
            std::array::from_fn(|i| Interval::point(b[i]).sub(Interval::point(a[i])));
        let offset: [Interval; 3] =
            std::array::from_fn(|i| Interval::point(point[i]).sub(Interval::point(a[i])));
        let side = (0..3).fold(Interval::point(0.0), |sum, i| {
            let j = (i + 1) % 3;
            let k = (i + 2) % 3;
            sum.add(
                direction[j]
                    .mul(offset[k])
                    .sub(direction[k].mul(offset[j]))
                    .scale(normal[i]),
            )
        });
        side.lo.is_finite() && side.lo >= 0.0
    })
}
