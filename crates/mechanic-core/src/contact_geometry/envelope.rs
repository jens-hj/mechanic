//! Finite support envelopes using weak-duality certificates, never guessed LP vertices.
//!
//! For nonnegative λ and Ax ≤ b, c·x ≤ λ·b + |c − Aᵀλ|₁ |x|∞.
//! Directed scalar intervals enclose both terms. Inaccurate/singular candidate
//! solves can only loosen a certificate, not manufacture a smaller bound.

use super::{ContactGeometryError, ContactPolytope, triangle_normal};
use bevy_math::{DVec3, DVec4};

pub(super) mod interval;
use interval::{Interval, dot};

#[derive(Clone, Copy)]
struct Certificate {
    offset: f64,
    residual: f64,
}

impl ContactPolytope {
    /// Enclosing radius about the frame origin for both the face half-spaces and
    /// vertices. Unlike a vertex radius alone, this detects unbounded face data.
    ///
    /// # Errors
    /// Rejects geometry whose finite radius cannot be certified.
    pub fn conservative_radius(&self) -> Result<f64, ContactGeometryError> {
        let radius = radius_bound(&self.planes)?;
        let vertex_radius = self.vertices.iter().try_fold(0.0_f64, |radius, &vertex| {
            let candidate = dot(vertex, vertex).sqrt()?.hi;
            Ok::<_, ContactGeometryError>(radius.max(candidate))
        })?;
        Ok(radius.max(vertex_radius))
    }

    /// Upper bound on normal penetration anywhere inside the finite triangle's
    /// normal prism, after every convex point can move by `displacement` metres.
    /// A nonpositive bound proves no penetration. A positive bound is not a hit:
    /// the envelope can overestimate overlap, especially in an empty prism.
    /// None proves the entire envelope lies below the finite triangle's normal
    /// projection range, so there is no surface intersection. A numerical error
    /// remains an error, never a separation result.
    /// This validation envelope must never generate contact impulses.
    ///
    /// # Errors
    /// Rejects invalid inputs or geometry/numerics without a finite certificate.
    pub fn triangle_penetration_bound(
        &self,
        triangle: [DVec3; 3],
        displacement: f64,
    ) -> Result<Option<f64>, ContactGeometryError> {
        let normal = triangle_normal(triangle)?;
        if !displacement.is_finite() || displacement < 0.0 {
            return Err(ContactGeometryError);
        }
        // Recenter arithmetic; face offsets are rounded outward, never inward.
        let center = self.vertices[0];
        let mut planes = self
            .planes
            .iter()
            .map(|plane| {
                let axis = plane.truncate();
                // Authored f32 face/vertex representations can differ slightly.
                // Enclose both, without modifying the physical narrowphase.
                let support = self.vertices.iter().try_fold(plane.w, |offset, &point| {
                    Ok::<_, ContactGeometryError>(offset.max(dot(axis, point).finite()?.hi))
                })?;
                let offset = Interval::point(support)
                    .sub(dot(axis, center))
                    .add(dot(axis, axis).sqrt()?.scale(displacement));
                Ok(axis.extend(offset.finite()?.hi))
            })
            .collect::<Result<Vec<_>, ContactGeometryError>>()?;
        let radius = radius_bound(&planes)?;
        // Enclose every vertex projection: the represented normal need not be
        // exactly perpendicular to the represented triangle edges.
        let mut plane_level = dot_difference(normal, triangle[0], center).finite()?;
        for &point in &triangle[1..] {
            let projection = dot_difference(normal, point, center).finite()?;
            plane_level.lo = plane_level.lo.min(projection.lo);
            plane_level.hi = plane_level.hi.max(projection.hi);
        }
        if support_bound(&planes, normal, radius)? < plane_level.lo {
            return Ok(None);
        }
        let vertices = triangle.map(|point| point - center);
        // The prism sides use represented normals. Pad their offsets for normal
        // nonorthogonality and subtraction roundoff, so they enclose the actual
        // finite triangle and every relevant point along its normal column.
        let normal_length = dot(normal, normal).sqrt()?;
        let mut triangle_radius = 0.0_f64;
        for point in triangle {
            let squared = (0..3).fold(Interval::point(0.0), |sum, axis| {
                let delta = Interval::point(point[axis]).sub(Interval::point(center[axis]));
                sum.add(delta.square())
            });
            triangle_radius = triangle_radius.max(squared.sqrt()?.hi);
        }
        let column_reach = Interval::point(radius)
            .add(Interval::point(triangle_radius))
            .div_positive(normal_length)?;
        for edge in 0..3 {
            let axis = (vertices[(edge + 1) % 3] - vertices[edge]).cross(normal);
            if !axis.is_finite() || axis.length_squared() == 0.0 {
                return Err(ContactGeometryError);
            }
            let mut offset = f64::NEG_INFINITY;
            for point in triangle {
                offset = offset.max(dot_difference(axis, point, center).hi);
            }
            let padding = dot(axis, normal).abs().mul(column_reach);
            planes.push(axis.extend(Interval::point(offset).add(padding).finite()?.hi));
        }
        let upper = support_bound(&planes, -normal, radius)?;
        Ok(Some(plane_level.add(Interval::point(upper)).finite()?.hi))
    }
}

fn dot_difference(axis: DVec3, point: DVec3, center: DVec3) -> Interval {
    (0..3).fold(Interval::point(0.0), |sum, row| {
        sum.add(
            Interval::point(point[row])
                .sub(Interval::point(center[row]))
                .scale(axis[row]),
        )
    })
}

fn radius_bound(planes: &[DVec4]) -> Result<f64, ContactGeometryError> {
    let mut offset = 0.0_f64;
    let mut residual = 0.0_f64;
    let mut coordinate_certificates = Vec::with_capacity(6);
    for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
        for sign in [-1.0, 1.0] {
            let mut best: Option<Certificate> = None;
            certificates(planes, axis * sign, |candidate| {
                if candidate.residual < 0.25
                    && best.is_none_or(|old| {
                        candidate.offset.max(0.0) / (1.0 - candidate.residual)
                            < old.offset.max(0.0) / (1.0 - old.residual)
                    })
                {
                    best = Some(candidate);
                }
            });
            let best = best.ok_or(ContactGeometryError)?;
            offset = offset.max(best.offset);
            residual = residual.max(best.residual);
            coordinate_certificates.push(best);
        }
    }
    // |x|∞ ≤ B + ε|x|∞ establishes boundedness itself; no vertex-radius assumption.
    let infinity_radius = Interval::point(offset)
        .div_positive(Interval::point(1.0).sub(Interval::point(residual)))?;
    let mut squared_radius = Interval::point(0.0);
    for pair in coordinate_certificates.chunks_exact(2) {
        let extent = pair.iter().fold(0.0_f64, |bound, certificate| {
            bound.max(
                Interval::point(certificate.offset)
                    .add(Interval::point(certificate.residual).mul(infinity_radius))
                    .hi,
            )
        });
        squared_radius = squared_radius.add(Interval::point(extent).square());
    }
    let radius = squared_radius.sqrt()?.finite()?.hi;
    Ok(radius)
}

fn support_bound(
    planes: &[DVec4],
    direction: DVec3,
    radius: f64,
) -> Result<f64, ContactGeometryError> {
    let mut best = dot(direction, direction).sqrt()?.scale(radius).hi;
    certificates(planes, direction, |candidate| {
        let upper = Interval::point(candidate.offset)
            .add(Interval::point(candidate.residual).scale(radius));
        if upper.hi.is_finite() {
            best = best.min(upper.hi);
        }
    });
    Ok(best)
}

fn certificates(planes: &[DVec4], direction: DVec3, mut visit: impl FnMut(Certificate)) {
    for first in 0..planes.len() {
        for second in first + 1..planes.len() {
            for third in second + 1..planes.len() {
                let rows = [planes[first], planes[second], planes[third]];
                let [a, b, c] = rows.map(DVec4::truncate);
                let determinant = a.dot(b.cross(c));
                if determinant == 0.0 || !determinant.is_finite() {
                    continue;
                }
                // These are proposals only. The interval residual below certifies
                // the represented nonnegative coefficients independently of solve accuracy.
                let weights = [
                    direction.dot(b.cross(c)) / determinant,
                    a.dot(direction.cross(c)) / determinant,
                    a.dot(b.cross(direction)) / determinant,
                ]
                .map(|value| value.max(0.0));
                if weights.iter().any(|value| !value.is_finite()) {
                    continue;
                }
                let mut offset = Interval::point(0.0);
                let mut residual = Interval::point(0.0);
                for (row, weight) in rows.iter().zip(weights) {
                    offset = offset.add(Interval::point(row.w).scale(weight));
                }
                for axis in 0..3 {
                    let mut error = Interval::point(direction[axis]);
                    for (row, weight) in rows.iter().zip(weights) {
                        error = error.sub(Interval::point(row[axis]).scale(weight));
                    }
                    residual = residual.add(error.abs());
                }
                if offset.hi.is_finite() && residual.hi.is_finite() {
                    visit(Certificate {
                        offset: offset.hi,
                        residual: residual.hi,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
