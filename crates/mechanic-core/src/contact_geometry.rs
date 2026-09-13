//! Double-precision finite convex/triangle queries shared by physics backends.

use bevy_math::{DQuat, DVec3, DVec4};

use crate::{ColliderShape, LocalCollider};

mod activation;
mod envelope;
mod pair;
mod quadratic;
pub use pair::{ConvexFeature, ConvexSeparation};
pub use quadratic::ContactVelocity;

/// Invalid compiled geometry or query input. A failed query is not an empty hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("non-finite, degenerate, or inconsistent contact geometry")]
pub struct ContactGeometryError;

/// Cached convex geometry in one coordinate frame. Vertices, planes, and edge
/// directions retain the compiled shape; cylinders use their authored decomposition.
#[derive(Clone, Debug)]
pub struct ContactPolytope {
    vertices: Vec<DVec3>,
    planes: Vec<DVec4>,
    edges: Vec<DVec3>,
}

/// One retained finite triangle support, with opposing points on both surfaces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleContactPoint {
    /// Point on the finite triangle, in the query frame.
    pub triangle_point: DVec3,
    /// Point on the opposing convex surface along the triangle normal.
    pub body_point: DVec3,
    /// Outward triangle normal, following its winding.
    pub normal: DVec3,
    /// Nonnegative overlap measured along the triangle normal, in metres.
    pub depth: f64,
}

/// Rigid motion over a normalized interval, with an unwrapped world rotation.
/// Supplying only endpoint quaternions would lose collisions during full turns.
#[derive(Clone, Copy, Debug)]
pub struct RigidContactSweep {
    /// Linear displacement of the body origin.
    pub translation: DVec3,
    /// World axis multiplied by the complete signed rotation angle in radians.
    pub angular_displacement: DVec3,
}

/// Bounded continuous collision result. Non-convergence never means separation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SweepOutcome {
    /// The complete rigid trajectory is separated from the finite triangle.
    Separated {
        /// Number of exact separating-axis evaluations.
        evaluations: usize,
    },
    /// Conservative first contact within the requested separation tolerance.
    Impact {
        /// Fraction of the full motion, in [0, 1].
        fraction: f64,
        /// Signed SAT separation at the returned pose, in metres.
        separation: f64,
        /// Number of exact separating-axis evaluations.
        evaluations: usize,
    },
    /// Work/progress bound exhausted. The caller must retry or retain old state.
    Unconverged {
        /// Last evaluated fraction; the remainder has not been certified clear.
        fraction: f64,
        /// Number of exact separating-axis evaluations.
        evaluations: usize,
    },
}

impl ContactPolytope {
    /// Caches a compiled convex hull in compound-local coordinates, such as the
    /// exact prism behind a compiled cylinder.
    ///
    /// # Errors
    /// Rejects invalid, empty, or degenerate compiled geometry.
    pub fn from_convex(hull: &crate::CompiledConvex) -> Result<Self, ContactGeometryError> {
        Self {
            vertices: hull.vertices.iter().map(|v| v.as_dvec3()).collect(),
            planes: hull.face_planes.iter().map(|v| v.as_dvec4()).collect(),
            edges: hull.edge_directions.iter().map(|v| v.as_dvec3()).collect(),
        }
        .validated()
    }

    /// Caches a compiled collider in compound-local coordinates.
    ///
    /// # Errors
    /// Rejects invalid, empty, or degenerate compiled geometry.
    pub fn from_collider(collider: &LocalCollider) -> Result<Self, ContactGeometryError> {
        let result = match &collider.shape {
            ColliderShape::Cuboid {
                local_rotation,
                half_extents,
            } => {
                let half = half_extents.as_dvec3();
                let rotation = local_rotation.as_dquat();
                let center = collider.local_center.as_dvec3();
                if !half.is_finite()
                    || half.min_element() <= 0.0
                    || !valid_rotation(rotation)
                    || !center.is_finite()
                {
                    return Err(ContactGeometryError);
                }
                let rotation = rotation.normalize();
                let edges = [DVec3::X, DVec3::Y, DVec3::Z].map(|v| rotation * v);
                let mut planes = Vec::with_capacity(6);
                for (axis, &extent) in edges.iter().zip(half.as_ref()) {
                    for sign in [-1.0, 1.0] {
                        let normal = *axis * sign;
                        planes.push(normal.extend(normal.dot(center) + extent));
                    }
                }
                let vertices = (0..8)
                    .map(|corner| {
                        let signs = DVec3::new(
                            if corner & 1 == 0 { -1.0 } else { 1.0 },
                            if corner & 2 == 0 { -1.0 } else { 1.0 },
                            if corner & 4 == 0 { -1.0 } else { 1.0 },
                        );
                        center + rotation * (half * signs)
                    })
                    .collect();
                Self {
                    vertices,
                    planes,
                    edges: edges.to_vec(),
                }
            }
            ColliderShape::Convex(hull) => Self {
                vertices: hull.vertices.iter().map(|v| v.as_dvec3()).collect(),
                planes: hull.face_planes.iter().map(|v| v.as_dvec4()).collect(),
                edges: hull.edge_directions.iter().map(|v| v.as_dvec3()).collect(),
            },
        };
        result.validated()
    }

    // Rejects geometry that cannot describe a closed convex solid.
    fn validated(self) -> Result<Self, ContactGeometryError> {
        if self.vertices.len() < 4
            || self.planes.len() < 4
            || self.edges.len() < 3
            || self.vertices.iter().any(|v| !v.is_finite())
            || self
                .planes
                .iter()
                .any(|v| !v.is_finite() || v.truncate().length_squared() < 1e-20)
            || self
                .edges
                .iter()
                .any(|v| !v.is_finite() || v.length_squared() < 1e-20)
        {
            return Err(ContactGeometryError);
        }
        Ok(self)
    }

    /// Transforms cached local geometry into a scene-relative query frame.
    ///
    /// # Errors
    /// Rejects a non-finite translation or non-unit rotation.
    pub fn transformed(
        &self,
        position: DVec3,
        rotation: DQuat,
    ) -> Result<Self, ContactGeometryError> {
        if !position.is_finite() || !valid_rotation(rotation) {
            return Err(ContactGeometryError);
        }
        let rotation = rotation.normalize();
        Ok(Self {
            vertices: self
                .vertices
                .iter()
                .map(|v| position + rotation * *v)
                .collect(),
            planes: self
                .planes
                .iter()
                .map(|p| {
                    let normal = rotation * p.truncate();
                    normal.extend(p.w + normal.dot(position))
                })
                .collect(),
            edges: self.edges.iter().map(|v| rotation * *v).collect(),
        })
    }

    /// Inclusive minimum/maximum of the exact transformed vertices.
    pub fn bounds(&self) -> [DVec3; 2] {
        self.vertices.iter().fold(
            [DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)],
            |[lo, hi], &v| [lo.min(v), hi.max(v)],
        )
    }

    /// Clips the finite triangle against every convex face, then keeps up to
    /// four unique diagonal-extreme supports, matching the terrain manifold rule.
    /// It never substitutes an infinite plane or fills a hole between triangles.
    ///
    /// # Errors
    /// Rejects non-finite or zero-area triangles.
    pub fn triangle_contacts(
        &self,
        triangle: [DVec3; 3],
    ) -> Result<Vec<TriangleContactPoint>, ContactGeometryError> {
        let normal = triangle_normal(triangle)?;
        let mut polygon = triangle.to_vec();
        for plane in &self.planes {
            let mut clipped = Vec::with_capacity(polygon.len() + 1);
            for index in 0..polygon.len() {
                let a = polygon[index];
                let b = polygon[(index + 1) % polygon.len()];
                let da = plane.truncate().dot(a) - plane.w;
                let db = plane.truncate().dot(b) - plane.w;
                if da <= 0.0 {
                    clipped.push(a);
                }
                if (da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0) {
                    clipped.push(a.lerp(b, da / (da - db)));
                }
            }
            polygon = clipped;
            if polygon.is_empty() {
                return Ok(Vec::new());
            }
        }
        let [u, v] = contact_tangents(normal);
        let mut contacts: Vec<TriangleContactPoint> = Vec::with_capacity(4);
        for direction in [u + v, u - v, -u - v, -u + v] {
            let mut point = polygon[0];
            for &candidate in &polygon[1..] {
                if candidate.dot(direction) > point.dot(direction) {
                    point = candidate;
                }
            }
            if contacts
                .iter()
                .any(|p| p.triangle_point.distance(point) < 1e-5)
            {
                continue;
            }
            // Intersect a line through this finite support with *all* convex
            // faces. A whole-hull support or one chosen face can invent a point
            // outside the collider on oblique terrain or at its edges.
            let lower = self
                .planes
                .iter()
                .filter_map(|plane| {
                    let slope = plane.truncate().dot(normal);
                    (slope < -1e-12).then(|| (plane.w - plane.truncate().dot(point)) / slope)
                })
                .fold(f64::NEG_INFINITY, f64::max);
            if !lower.is_finite() {
                return Err(ContactGeometryError);
            }
            let depth = (-lower).max(0.0);
            contacts.push(TriangleContactPoint {
                triangle_point: point,
                body_point: point - depth * normal,
                normal,
                depth,
            });
        }
        Ok(contacts)
    }

    /// Finite points for split penetration recovery, separate from the physical
    /// four-corner contact manifold. Retains intersection supports and adds actual
    /// penetrating vertices whose projections are certified inside the triangle.
    /// No infinite plane, expanded footprint, or physical velocity bias is used.
    ///
    /// # Errors
    /// Rejects invalid triangles or failed intersection geometry.
    pub fn triangle_recovery_contacts(
        &self,
        triangle: [DVec3; 3],
    ) -> Result<Vec<TriangleContactPoint>, ContactGeometryError> {
        let mut points = self.triangle_contacts(triangle)?;
        if points.is_empty() {
            return Ok(points);
        }
        let normal = triangle_normal(triangle)?;
        for &body_point in &self.vertices {
            let gap = (body_point - triangle[0]).dot(normal);
            if gap >= 0.0 || !gap.is_finite() {
                continue;
            }
            let triangle_point = body_point - gap * normal;
            if activation::inside_triangle(triangle, triangle_point, normal)
                && points.iter().all(|point| point.body_point != body_point)
            {
                points.push(TriangleContactPoint {
                    body_point,
                    triangle_point,
                    normal,
                    depth: -gap,
                });
            }
        }
        Ok(points)
    }

    /// Finds opposing finite surface points within a nonnegative normal gap.
    /// The collider is extruded toward the triangle by `margin`, including its
    /// silhouette planes. Tangential faces are not inflated, so a nearby hole or
    /// finite triangle edge cannot become an infinite supporting plane.
    /// Returned body points lie on the original collider; positive separation
    /// is `(body_point - triangle_point).dot(normal)` and has zero depth.
    ///
    /// # Errors
    /// Rejects invalid triangles, non-finite margins, or negative margins.
    pub fn triangle_proximity(
        &self,
        triangle: [DVec3; 3],
        margin: f64,
    ) -> Result<Vec<TriangleContactPoint>, ContactGeometryError> {
        self.proximity_with_bound(triangle, margin, margin)
    }

    /// Actual finite intersections, or opposing points within a numerical-zero
    /// activation bound. Separated pairs search half that bound, reserving room
    /// for clipping arithmetic; every reconstructed gap must meet the full bound.
    /// Existing intersection manifolds are preserved exactly.
    ///
    /// # Errors
    /// Rejects invalid geometry/bounds or points that cannot meet the gap bound.
    pub fn triangle_activation_contacts(
        &self,
        triangle: [DVec3; 3],
        maximum_gap: f64,
    ) -> Result<Vec<TriangleContactPoint>, ContactGeometryError> {
        if !maximum_gap.is_finite() || maximum_gap <= 0.0 {
            return Err(ContactGeometryError);
        }
        let points = self.triangle_contacts(triangle)?;
        if !points.is_empty() {
            return Ok(points);
        }
        match self.proximity_with_bound(triangle, maximum_gap * 0.5, maximum_gap) {
            // An empty inner search is not evidence that the outer half of the
            // activation window is empty. Certify unchanged finite hull vertices
            // there before returning an empty manifold. Failure to find a vertex
            // preserves this already-valid empty query; it does not create one.
            Ok(points) if points.is_empty() => Ok(self
                .vertex_activation(triangle, maximum_gap)
                .unwrap_or(points)),
            Ok(points)
                if points
                    .iter()
                    .all(|p| (p.body_point - p.triangle_point).dot(p.normal) <= maximum_gap) =>
            {
                Ok(points)
            }
            // Near-parallel face equations can amplify transform/clipping
            // roundoff even when an actual hull corner is within the bound.
            // Use that unchanged vertex only with a certified finite projection.
            _ => self.vertex_activation(triangle, maximum_gap),
        }
    }

    #[allow(clippy::too_many_lines)] // Bounded finite clipping and explicit gap feasibility checks.
    fn proximity_with_bound(
        &self,
        triangle: [DVec3; 3],
        margin: f64,
        maximum_gap: f64,
    ) -> Result<Vec<TriangleContactPoint>, ContactGeometryError> {
        if !margin.is_finite() || margin < 0.0 {
            return Err(ContactGeometryError);
        }
        if margin == 0.0 {
            return self.triangle_contacts(triangle);
        }
        let normal = triangle_normal(triangle)?;
        let mut extruded = self.clone();
        extruded
            .vertices
            .extend(self.vertices.iter().map(|point| *point - margin * normal));
        for plane in &mut extruded.planes {
            plane.w += margin * (-plane.truncate().dot(normal)).max(0.0);
        }
        // Minkowski sum with a segment also has silhouette faces generated by
        // each original edge crossed with the extrusion direction. Merely
        // shifting existing planes overestimates oblique convex footprints.
        for edge in &self.edges {
            if let Some(axis) = edge.cross(normal).try_normalize() {
                let [minimum, maximum] = project(&self.vertices, axis);
                extruded.planes.push(axis.extend(maximum));
                extruded.planes.push((-axis).extend(-minimum));
            }
        }
        if extruded.vertices.iter().any(|point| !point.is_finite())
            || extruded.planes.iter().any(|plane| !plane.is_finite())
        {
            return Err(ContactGeometryError);
        }
        let mut contacts = extruded.triangle_contacts(triangle)?;
        if contacts.is_empty() {
            return Ok(contacts);
        }
        let center = contacts.iter().map(|p| p.triangle_point).sum::<DVec3>()
            / f64::from(u32::try_from(contacts.len()).map_err(|_| ContactGeometryError)?);
        for point in &mut contacts {
            let mut separation = self.normal_column_lower(point.triangle_point, normal)?;
            if separation > maximum_gap {
                // Clipping roundoff at a nearly normal-parallel face can become
                // a large normal gap after division by its tiny slope. Move only
                // within the finite clipped polygon, toward its convex interior.
                // Keep the nearest representable feasible point found by bounded
                // bisection; never clamp the gap or increase the requested margin.
                let center_gap = self.normal_column_lower(center, normal)?;
                if center_gap > maximum_gap {
                    return Err(ContactGeometryError);
                }
                let original = point.triangle_point;
                let mut outside = 0.0;
                let mut inside = 1.0;
                let mut feasible = center;
                separation = center_gap;
                for _ in 0..64 {
                    let fraction = (outside + inside) * 0.5;
                    if fraction <= outside || fraction >= inside {
                        break;
                    }
                    let candidate = original.lerp(center, fraction);
                    let gap = self.normal_column_lower(candidate, normal)?;
                    if gap <= maximum_gap {
                        inside = fraction;
                        feasible = candidate;
                        separation = gap;
                    } else {
                        outside = fraction;
                    }
                }
                point.triangle_point = feasible;
            }
            point.body_point = point.triangle_point + separation * normal;
            point.depth = (-separation).max(0.0);
        }
        Ok(contacts)
    }

    fn normal_column_lower(
        &self,
        point: DVec3,
        normal: DVec3,
    ) -> Result<f64, ContactGeometryError> {
        let lower = self
            .planes
            .iter()
            .filter_map(|plane| {
                let slope = plane.truncate().dot(normal);
                (slope < -1e-12).then(|| (plane.w - plane.truncate().dot(point)) / slope)
            })
            .fold(f64::NEG_INFINITY, f64::max);
        if !lower.is_finite() {
            return Err(ContactGeometryError);
        }
        Ok(lower)
    }

    /// Exact continuous SAT interval for a fixed-orientation convex translating
    /// by `displacement` past a finite triangle. Fractions are within [0, 1].
    /// Rotation requires a separate conservative angular sweep; this method does
    /// not approximate it with endpoint translation.
    ///
    /// # Errors
    /// Rejects non-finite motion or invalid triangles.
    pub fn translation_interval(
        &self,
        triangle: [DVec3; 3],
        displacement: DVec3,
    ) -> Result<Option<[f64; 2]>, ContactGeometryError> {
        let normal = triangle_normal(triangle)?;
        if !displacement.is_finite() {
            return Err(ContactGeometryError);
        }
        let triangle_edges = [
            triangle[1] - triangle[0],
            triangle[2] - triangle[1],
            triangle[0] - triangle[2],
        ];
        let separating_directions = std::iter::once(normal)
            .chain(self.planes.iter().map(|p| p.truncate()))
            .chain(
                self.edges
                    .iter()
                    .flat_map(|edge| triangle_edges.map(|other| edge.cross(other))),
            );
        let mut interval: [f64; 2] = [0.0, 1.0];
        for axis in separating_directions {
            let Some(axis) = axis.try_normalize() else {
                continue;
            };
            let body = project(&self.vertices, axis);
            let surface = project(&triangle, axis);
            let speed = axis.dot(displacement);
            let low = surface[0] - body[1];
            let high = surface[1] - body[0];
            if speed == 0.0 {
                if low > 0.0 || high < 0.0 {
                    return Ok(None);
                }
            } else {
                let a = low / speed;
                let b = high / speed;
                interval = [interval[0].max(a.min(b)), interval[1].min(a.max(b))];
                if interval[0] > interval[1] {
                    return Ok(None);
                }
            }
        }
        Ok(Some(interval))
    }

    /// Largest separating-axis gap against the exact finite triangle. A positive
    /// value is a conservative lower bound on distance; nonpositive means overlap.
    ///
    /// # Errors
    /// Rejects a non-finite or zero-area triangle.
    pub fn triangle_separation(&self, triangle: [DVec3; 3]) -> Result<f64, ContactGeometryError> {
        Ok(self.triangle_separating_axis(triangle)?.1)
    }

    fn triangle_separating_axis(
        &self,
        triangle: [DVec3; 3],
    ) -> Result<(DVec3, f64), ContactGeometryError> {
        let normal = triangle_normal(triangle)?;
        let triangle_edges = [
            triangle[1] - triangle[0],
            triangle[2] - triangle[1],
            triangle[0] - triangle[2],
        ];
        let directions = std::iter::once(normal)
            .chain(self.planes.iter().map(|p| p.truncate()))
            .chain(
                self.edges
                    .iter()
                    .flat_map(|edge| triangle_edges.map(|other| edge.cross(other))),
            );
        let mut separation = f64::NEG_INFINITY;
        let mut selected = normal;
        for direction in directions {
            let Some(direction) = direction.try_normalize() else {
                continue;
            };
            let body = project(&self.vertices, direction);
            let surface = project(&triangle, direction);
            for (gap, axis) in [
                (surface[0] - body[1], -direction),
                (body[0] - surface[1], direction),
            ] {
                if gap > separation {
                    separation = gap;
                    selected = axis;
                }
            }
        }
        Ok((selected, separation))
    }

    /// Conservative advancement for simultaneous translation and rotation of
    /// this local convex. The speed bound includes every vertex's angular reach.
    /// Articulated trajectories need their own reconstructed-pose path and reach
    /// bound; independently sweeping their endpoint body poses is insufficient.
    ///
    /// # Errors
    /// Rejects invalid motion, pose, triangle, tolerance, or an empty work bound.
    #[allow(clippy::too_many_arguments)] // Explicit initial pose and independent numerical bounds.
    pub fn rigid_triangle_sweep(
        &self,
        triangle: [DVec3; 3],
        position: DVec3,
        rotation: DQuat,
        motion: RigidContactSweep,
        tolerance: f64,
        max_evaluations: usize,
    ) -> Result<SweepOutcome, ContactGeometryError> {
        triangle_normal(triangle)?;
        if !position.is_finite()
            || !valid_rotation(rotation)
            || !motion.translation.is_finite()
            || !motion.angular_displacement.is_finite()
            || !tolerance.is_finite()
            || tolerance <= 0.0
            || max_evaluations == 0
        {
            return Err(ContactGeometryError);
        }
        let radius = self
            .vertices
            .iter()
            .map(|point| point.length())
            .fold(0.0, f64::max);
        let speed_bound =
            motion.translation.length() + motion.angular_displacement.length() * radius;
        if !speed_bound.is_finite() {
            return Err(ContactGeometryError);
        }
        let mut fraction = 0.0;
        for evaluations in 1..=max_evaluations {
            let pose_rotation =
                DQuat::from_scaled_axis(motion.angular_displacement * fraction) * rotation;
            let body = self.transformed(position + motion.translation * fraction, pose_rotation)?;
            let separation = body.triangle_separation(triangle)?;
            if separation <= tolerance {
                return Ok(SweepOutcome::Impact {
                    fraction,
                    separation,
                    evaluations,
                });
            }
            if speed_bound == 0.0 || separation > speed_bound * (1.0 - fraction) {
                return Ok(SweepOutcome::Separated { evaluations });
            }
            let next = fraction + separation / speed_bound;
            if evaluations == max_evaluations || next <= fraction {
                return Ok(SweepOutcome::Unconverged {
                    fraction,
                    evaluations,
                });
            }
            fraction = next.min(1.0);
        }
        unreachable!("nonzero bounded loop returns its final outcome")
    }

    /// Outward convex face normal most opposed to a surface normal. Used only
    /// to retain a crown point during curved-terrain manifold reduction.
    pub fn opposing_normal(&self, surface_normal: DVec3) -> DVec3 {
        self.planes[1..]
            .iter()
            .fold(self.planes[0].truncate(), |selected, plane| {
                let candidate = plane.truncate();
                if candidate.dot(surface_normal) < selected.dot(surface_normal) {
                    candidate
                } else {
                    selected
                }
            })
    }
}

fn valid_rotation(rotation: DQuat) -> bool {
    rotation.is_finite() && (rotation.length_squared() - 1.0).abs() < 1e-6
}

fn triangle_normal(triangle: [DVec3; 3]) -> Result<DVec3, ContactGeometryError> {
    if triangle.iter().any(|v| !v.is_finite()) {
        return Err(ContactGeometryError);
    }
    (triangle[1] - triangle[0])
        .cross(triangle[2] - triangle[0])
        .try_normalize()
        .ok_or(ContactGeometryError)
}

fn project(points: &[DVec3], axis: DVec3) -> [f64; 2] {
    points
        .iter()
        .fold([f64::INFINITY, f64::NEG_INFINITY], |[lo, hi], p| {
            let distance = p.dot(axis);
            [lo.min(distance), hi.max(distance)]
        })
}

/// Stable orthonormal tangent frame using the same branch as the GPU solver.
pub(crate) fn contact_tangents(normal: DVec3) -> [DVec3; 2] {
    let reference = if normal.y.abs() > 0.9 {
        DVec3::X
    } else {
        DVec3::Y
    };
    let u = reference.cross(normal).normalize();
    [u, normal.cross(u)]
}

#[cfg(test)]
mod tests;
