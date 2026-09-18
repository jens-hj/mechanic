//! Exact contact between a solid capped cylinder and a finite triangle.
//!
//! A cylinder collides as a circle, not as the prism that bounds it, so a wheel
//! rolls without lifting over facet corners. Every reported point is where the
//! triangle normal's column through the triangle point enters the cylinder,
//! which is the same depth law [`ContactPolytope`](crate::ContactPolytope) uses for its clipped points.

use super::{ContactGeometryError, TriangleContactPoint, contact_tangents, triangle_normal};
use crate::CompiledCylinder;
use bevy_math::{DQuat, DVec3};

// Radial alignment below which the cylinder stands on an end cap, matching the
// GPU route's `CYLINDER_MANIFOLD_ALIGNMENT`. Above it the side supports.
const END_ON_ALIGNMENT: f64 = 0.05;

// How far aside of a triangle the lowest line must pass, in metres, for none of
// the triangle's points to count as on it.
const FLANK_GAP: f64 = 1e-5;

// Subtracted from a clearance bound to cover its rounding, in metres.
const CLEARANCE_ROUNDING: f64 = 1e-6;

// Squared lengths below this are treated as parallel directions.
const PARALLEL: f64 = 1e-18;

// Height differences within this are one support level, in metres.
const LEVEL: f64 = 1e-9;

/// What a triangle supports on a cylinder.
#[derive(Clone, Debug, PartialEq)]
pub enum TriangleSupport {
    /// Every contact point within the margin.
    Points(Vec<TriangleContactPoint>),
    /// The side's lowest line passes wide of the triangle: every point it
    /// supports lies more than 10 µm off that line and no nearer the surface
    /// than this separation. [`ContactCylinder::triangle_contacts`] finds them.
    Flank(f64),
}

/// A solid capped cylinder in one coordinate frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactCylinder {
    center: DVec3,
    axis: DVec3,
    radius: f64,
    half_length: f64,
}

/// Where a contact sits on a cylinder, relative to the contact normal and the
/// axis rather than the material. A cylinder fills the same space however far
/// it turns about its own axis, so a rolling wheel keeps its support under the
/// axle and a spinning cap keeps its rim points where they touch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CylinderAnchor {
    axial: f64,
    // Offset from the axis as a fraction of the radius. On the side it is in the
    // frame of the lowest generator toward the surface and the axis crossed with
    // it; on an end, in the cap's fixed rim directions.
    radial: [f64; 2],
    end_on: bool,
}

impl ContactCylinder {
    /// A cylinder about `axis` through `center`.
    ///
    /// # Errors
    /// Rejects non-finite or degenerate dimensions or axis.
    pub fn new(
        center: DVec3,
        axis: DVec3,
        radius: f64,
        half_length: f64,
    ) -> Result<Self, ContactGeometryError> {
        if !center.is_finite()
            || !radius.is_finite()
            || !half_length.is_finite()
            || radius <= 0.0
            || half_length <= 0.0
        {
            return Err(ContactGeometryError);
        }
        Ok(Self {
            center,
            axis: axis.try_normalize().ok_or(ContactGeometryError)?,
            radius,
            half_length,
        })
    }

    /// Caches a compiled solid cylinder in compound-local coordinates.
    ///
    /// # Errors
    /// Rejects non-finite or degenerate dimensions or orientation.
    pub fn from_compiled(cylinder: &CompiledCylinder) -> Result<Self, ContactGeometryError> {
        Self::new(
            cylinder.local_center.as_dvec3(),
            cylinder.local_rotation.as_dquat() * DVec3::Y,
            f64::from(cylinder.outer_radius),
            f64::from(cylinder.half_length),
        )
    }

    /// Axis midpoint.
    pub const fn center(&self) -> DVec3 {
        self.center
    }

    /// Unit axis direction.
    pub const fn axis(&self) -> DVec3 {
        self.axis
    }

    /// Outer radius in metres.
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    /// Half the axial length in metres.
    pub const fn half_length(&self) -> f64 {
        self.half_length
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
        if !position.is_finite() || !super::valid_rotation(rotation) {
            return Err(ContactGeometryError);
        }
        let rotation = rotation.normalize();
        Ok(Self {
            center: position + rotation * self.center,
            axis: rotation * self.axis,
            ..*self
        })
    }

    /// Farthest point along `direction`.
    pub fn support(&self, direction: DVec3) -> DVec3 {
        let axial = self.axis.dot(direction);
        let radial = (direction - axial * self.axis)
            .try_normalize()
            .unwrap_or(DVec3::ZERO);
        let cap = if axial == 0.0 {
            0.0
        } else {
            self.half_length.copysign(axial)
        };
        self.center + cap * self.axis + self.radius * radial
    }

    /// Outward surface normal most opposed to a surface normal: the side facing
    /// it, or the end cap when the cylinder stands on an end. Used only to keep
    /// a crown point during curved-terrain manifold reduction.
    pub fn opposing_normal(&self, surface_normal: DVec3) -> DVec3 {
        match self.side_frame(surface_normal) {
            Some([down, _]) => down,
            None => -self.axis * self.axis.dot(surface_normal).signum(),
        }
    }

    /// Finite contact points against a triangle within `margin` of separation,
    /// using the triangle's winding normal. Side contacts follow the lowest
    /// generator line; an end cap supplies rim points. Where the lowest points
    /// lie outside the triangle, the lowest point over its edges and corners is
    /// found exactly, so an edge such as a kerb is never passed through.
    ///
    /// # Errors
    /// Rejects non-finite or zero-area triangles and invalid margins.
    pub fn triangle_contacts(
        &self,
        triangle: [DVec3; 3],
        margin: f64,
    ) -> Result<Vec<TriangleContactPoint>, ContactGeometryError> {
        match self.triangle_points(triangle, margin, false)? {
            TriangleSupport::Points(points) => Ok(points),
            TriangleSupport::Flank(_) => Ok(Vec::new()),
        }
    }

    /// The contact points against a triangle, as [`Self::triangle_contacts`],
    /// unless the side's lowest line passes wide of it. Then every point the
    /// triangle supports lies up the side, off that line, and only a lower
    /// bound on their separation is returned: a caller already holding a
    /// deeper point on the line may not need them.
    ///
    /// # Errors
    /// Rejects non-finite or zero-area triangles and invalid margins.
    pub fn triangle_support(
        &self,
        triangle: [DVec3; 3],
        margin: f64,
    ) -> Result<TriangleSupport, ContactGeometryError> {
        self.triangle_points(triangle, margin, true)
    }

    fn triangle_points(
        &self,
        triangle: [DVec3; 3],
        margin: f64,
        flank: bool,
    ) -> Result<TriangleSupport, ContactGeometryError> {
        if !margin.is_finite() || margin < 0.0 {
            return Err(ContactGeometryError);
        }
        let normal = triangle_normal(triangle)?;
        let plane = normal.dot(triangle[0]);
        let alignment = self.axis.dot(normal);
        let radial = normal - alignment * self.axis;
        let spread = radial.length();
        let lowest = normal.dot(self.center)
            - self.radius * spread
            - self.half_length * alignment.abs()
            - plane;
        let none = || Ok(TriangleSupport::Points(Vec::new()));
        if lowest > margin {
            return none();
        }
        let sphere = self.sphere_clearance(triangle, normal);
        if sphere > margin {
            return none();
        }
        let column = self.column_clearance(triangle, normal, lowest);
        if column > margin {
            return none();
        }
        let mut candidates = Vec::with_capacity(8);
        if spread >= END_ON_ALIGNMENT {
            let generator = self.center - self.radius * (radial / spread);
            // A point on the lowest line lies over the line's own segment, so
            // one over a triangle this far aside is off it.
            if flank
                && self
                    .clip_generator(triangle, normal, generator, FLANK_GAP)
                    .is_none()
            {
                return Ok(TriangleSupport::Flank(lowest.max(sphere).max(column)));
            }
            let reached =
                self.clip_generator(triangle, normal, generator, 0.0)
                    .map(|[start, end]| {
                        for t in if end - start > LEVEL {
                            vec![start, end]
                        } else {
                            vec![0.5 * (start + end)]
                        } {
                            let body_point = generator + t * self.axis;
                            candidates.push(point(
                                body_point,
                                normal,
                                normal.dot(body_point) - plane,
                            ));
                        }
                        // The deepest part of the generator: all of it when level,
                        // otherwise the end nearer the surface.
                        2.0 * self.half_length * alignment.abs() <= LEVEL
                            || (alignment > 0.0 && start <= -self.half_length + LEVEL / alignment)
                            || (alignment < 0.0 && end >= self.half_length + LEVEL / alignment)
                    });
            if reached != Some(true) {
                self.lowest_over_boundary(triangle, normal, plane, &mut candidates);
            }
        } else {
            let cap = self.center - self.half_length.copysign(alignment) * self.axis;
            // Fixed rim directions keep contact identities while the cap
            // wobbles; a tilted cap also reports its lowest rim point.
            let lowest = (spread > 1e-9).then(|| -radial / spread);
            for direction in self
                .cap_frame(normal)
                .into_iter()
                .flat_map(|[first, second]| [first, second, -first, -second])
                .chain(lowest)
            {
                let rim = cap + self.radius * direction;
                let foot = rim - (normal.dot(rim) - plane) * normal;
                if inside(triangle, normal, foot)
                    && let Some(separation) = self.column_entry(foot, normal)
                {
                    candidates.push(point(foot + separation * normal, normal, separation));
                }
            }
            self.boundary_candidates(triangle, normal, plane, &mut candidates);
        }
        candidates.retain(|point| separation(point) <= margin);
        Ok(TriangleSupport::Points(reduce(candidates, normal)))
    }

    /// A lower bound on the distance to a triangle, zero where they may touch.
    ///
    /// Costs a few segment tests, so it can reject a triangle before an exact
    /// contact search or sweep. The cylinder lies within its radius of the axis
    /// segment, between its cap planes, and on one side of its lowest and
    /// highest points along the triangle normal; each gives a bound, and on a
    /// wheel above a surface the axis bound is its exact gap.
    pub fn triangle_clearance(&self, triangle: [DVec3; 3]) -> f64 {
        let Ok(normal) = triangle_normal(triangle) else {
            return 0.0;
        };
        let ends = [
            self.center - self.half_length * self.axis,
            self.center + self.half_length * self.axis,
        ];
        let axis = segment_triangle_distance(ends, triangle, normal) - self.radius;
        let [low, high] = triangle.iter().fold(
            [f64::INFINITY, f64::NEG_INFINITY],
            |[low, high], &vertex| {
                let axial = self.axis.dot(vertex - self.center);
                [low.min(axial), high.max(axial)]
            },
        );
        let caps = (low - self.half_length).max(-self.half_length - high);
        let alignment = self.axis.dot(normal);
        let reach = self.radius * (normal - alignment * self.axis).length()
            + self.half_length * alignment.abs();
        let height = normal.dot(self.center - triangle[0]);
        let plane = height.abs() - reach;
        // Rounding in the tests above stays far below a micrometre.
        (axis.max(caps).max(plane) - CLEARANCE_ROUNDING).max(0.0)
    }

    // A cheaper, looser lower bound on every contact's separation, from a sphere
    // around the triangle. It holds only while the axis is above the triangle's
    // plane, where the column under the triangle is no nearer than the triangle.
    fn sphere_clearance(&self, triangle: [DVec3; 3], normal: DVec3) -> f64 {
        let ends = [
            self.center - self.half_length * self.axis,
            self.center + self.half_length * self.axis,
        ];
        if ends.iter().any(|&end| normal.dot(end - triangle[0]) <= 0.0) {
            return 0.0;
        }
        let middle = (triangle[0] + triangle[1] + triangle[2]) / 3.0;
        let spread = triangle
            .iter()
            .map(|&vertex| vertex.distance(middle))
            .fold(0.0, f64::max);
        let offset = middle - self.center;
        let along = self
            .axis
            .dot(offset)
            .clamp(-self.half_length, self.half_length);
        let distance = (offset - along * self.axis).length();
        (distance - spread - self.radius - CLEARANCE_ROUNDING).max(0.0)
    }

    // A lower bound on every contact's separation. A contact is where a column
    // down the triangle normal enters the cylinder, so a cylinder anywhere below
    // the triangle is buried in it: beneath the plane only the axis's sideways
    // distance from the triangle counts.
    fn column_clearance(&self, triangle: [DVec3; 3], normal: DVec3, lowest: f64) -> f64 {
        if lowest >= 0.0 {
            return self.triangle_clearance(triangle);
        }
        let plane = normal.dot(triangle[0]);
        let ends = [
            self.center - self.half_length * self.axis,
            self.center + self.half_length * self.axis,
        ];
        let heights = ends.map(|end| normal.dot(end) - plane);
        let floor = |point: DVec3| point - (normal.dot(point) - plane) * normal;
        let distance = match heights {
            [first, second] if first > 0.0 && second > 0.0 => {
                segment_triangle_distance(ends, triangle, normal)
            }
            [first, second] if first <= 0.0 && second <= 0.0 => {
                segment_triangle_distance(ends.map(floor), triangle, normal)
            }
            [first, second] => {
                let crossing = ends[0] + (ends[1] - ends[0]) * (first / (first - second));
                let [above, below] = if first > 0.0 {
                    ends
                } else {
                    [ends[1], ends[0]]
                };
                segment_triangle_distance([above, crossing], triangle, normal).min(
                    segment_triangle_distance([floor(below), crossing], triangle, normal),
                )
            }
        };
        (distance - self.radius - CLEARANCE_ROUNDING).max(0.0)
    }

    /// The anchor of a contact at `body_point`, on the side or a cap, against a
    /// surface with `normal`.
    pub fn anchor(&self, normal: DVec3, body_point: DVec3) -> Option<CylinderAnchor> {
        let end_on = self.side_frame(normal).is_none();
        let [down, across] = self.anchor_frame(normal, end_on)?;
        let offset = body_point - self.center;
        let axial = self.axis.dot(offset);
        let radial = (offset - axial * self.axis) / self.radius;
        Some(CylinderAnchor {
            axial,
            radial: [radial.dot(down), radial.dot(across)],
            end_on,
        })
    }

    /// Distance of a point from the side's lowest line toward a surface with
    /// `normal`, or none when the cylinder stands on an end.
    pub fn lowest_line_distance(&self, normal: DVec3, point: DVec3) -> Option<f64> {
        let [down, _] = self.side_frame(normal)?;
        let offset = point - (self.center + self.radius * down);
        Some((offset - self.axis.dot(offset) * self.axis).length())
    }

    /// Point of an anchor for this pose, or none once the cylinder has tipped
    /// between standing on an end and lying on its side.
    pub fn anchor_point(&self, normal: DVec3, anchor: CylinderAnchor) -> Option<DVec3> {
        let [down, across] = self.anchor_frame(normal, anchor.end_on)?;
        Some(
            self.center
                + anchor.axial * self.axis
                + self.radius * (anchor.radial[0] * down + anchor.radial[1] * across),
        )
    }

    fn anchor_frame(&self, normal: DVec3, end_on: bool) -> Option<[DVec3; 2]> {
        if end_on {
            self.cap_frame(normal)
        } else {
            self.side_frame(normal)
        }
    }

    // Rim directions of a cap fixed by the surface's own tangents, so they don't
    // turn with the cylinder.
    fn cap_frame(&self, normal: DVec3) -> Option<[DVec3; 2]> {
        let [u, _] = contact_tangents(normal);
        let rim = u - self.axis.dot(u) * self.axis;
        let length = rim.length();
        (length >= END_ON_ALIGNMENT).then(|| {
            let first = rim * length.recip();
            [first, self.axis.cross(first)]
        })
    }

    fn side_frame(&self, normal: DVec3) -> Option<[DVec3; 2]> {
        let radial = normal - self.axis.dot(normal) * self.axis;
        let spread = radial.length();
        (spread >= END_ON_ALIGNMENT).then(|| {
            let down = -radial / spread;
            [down, self.axis.cross(down)]
        })
    }

    // The generator's parameter interval whose points project into the
    // triangle, widened outward by `widen` metres. Every edge plane contains the normal, so projection along it
    // leaves each edge test unchanged.
    fn clip_generator(
        &self,
        triangle: [DVec3; 3],
        normal: DVec3,
        generator: DVec3,
        widen: f64,
    ) -> Option<[f64; 2]> {
        let [mut start, mut end] = [-self.half_length, self.half_length];
        for edge in 0..3 {
            let from = triangle[edge];
            let inward = normal.cross(triangle[(edge + 1) % 3] - from);
            let offset = inward.dot(generator - from) + widen * inward.length();
            let rate = inward.dot(self.axis);
            if rate.abs() <= 1e-12 * inward.length() {
                if offset < 0.0 {
                    return None;
                }
            } else if rate > 0.0 {
                start = start.max(-offset / rate);
            } else {
                end = end.min(-offset / rate);
            }
        }
        (start <= end).then_some([start, end])
    }

    // Adds the lowest cylinder points over the triangle's edges and corners,
    // keeping both ends when a whole edge segment supports at one level.
    fn lowest_over_boundary(
        &self,
        triangle: [DVec3; 3],
        normal: DVec3,
        plane: f64,
        output: &mut Vec<TriangleContactPoint>,
    ) {
        let mut boundary = Vec::with_capacity(24);
        self.boundary_candidates(triangle, normal, plane, &mut boundary);
        let Some(lowest) = boundary.iter().map(separation).reduce(f64::min) else {
            return;
        };
        let level = boundary
            .into_iter()
            .filter(|point| separation(point) <= lowest + LEVEL)
            .collect::<Vec<_>>();
        let [u, v] = contact_tangents(normal);
        let direction = level
            .iter()
            .fold(DVec3::ZERO, |span, point| {
                let offset = point.triangle_point - level[0].triangle_point;
                if offset.length_squared() > span.length_squared() {
                    offset
                } else {
                    span
                }
            })
            .try_normalize()
            .unwrap_or(u + v);
        for sign in [1.0, -1.0] {
            let extreme = level.iter().copied().max_by(|a, b| {
                (sign * a.triangle_point.dot(direction))
                    .total_cmp(&(sign * b.triangle_point.dot(direction)))
            });
            if let Some(extreme) = extreme
                && !output
                    .iter()
                    .any(|p| p.triangle_point.distance(extreme.triangle_point) < 1e-5)
            {
                output.push(extreme);
            }
        }
    }

    // Column entries at every corner and at each edge's candidate minima. Over
    // one edge the cylinder is lowest at a corner, at the side's lowest point
    // above the edge line, or where a cap plane meets the side.
    fn boundary_candidates(
        &self,
        triangle: [DVec3; 3],
        normal: DVec3,
        plane: f64,
        output: &mut Vec<TriangleContactPoint>,
    ) {
        let mut entry = |foot: DVec3| {
            let foot = foot - (normal.dot(foot) - plane) * normal;
            if let Some(separation) = self.column_entry(foot, normal) {
                output.push(point(foot + separation * normal, normal, separation));
            }
        };
        let alignment = self.axis.dot(normal);
        let side = normal - alignment * self.axis;
        for edge in 0..3 {
            let from = triangle[edge];
            entry(from);
            let span = triangle[(edge + 1) % 3] - from;
            let length = span.length();
            let along = span / length;
            let offset = from - self.center;
            let axial = self.axis.dot(offset);
            let axial_rate = self.axis.dot(along);
            let radial = offset - axial * self.axis;
            let travel = along - axial_rate * self.axis;
            let mut at = |t: f64| {
                if t.is_finite() {
                    entry(from + t.clamp(0.0, length) * along);
                }
            };
            if travel.length_squared() > PARALLEL {
                // Remove the edge's own radial direction, leaving the column's
                // closest approach to the axis over every point of the edge line.
                let reject =
                    |vector: DVec3| vector - travel.dot(vector) / travel.length_squared() * travel;
                let (fixed, rising) = (reject(radial), reject(side));
                if let Some(height) = lower_root(rising, fixed, self.radius) {
                    at(-travel.dot(radial + height * side) / travel.length_squared());
                }
            } else if axial_rate.abs() > 0.0 {
                // An edge along the axis: every point of it within the caps
                // meets the side at one height.
                if let Some(height) = lower_root(side, radial, self.radius) {
                    for cap in [-self.half_length, self.half_length] {
                        at((cap - axial - height * alignment) / axial_rate);
                    }
                }
            }
            for cap in [-self.half_length, self.half_length] {
                if axial_rate.abs() > 1e-12 {
                    // Along this cap plane the edge parameter follows height.
                    let base = (cap - axial) / axial_rate;
                    let fixed = radial + base * travel;
                    let rising = side - alignment / axial_rate * travel;
                    if let Some(height) = lower_root(rising, fixed, self.radius) {
                        at(base - height * alignment / axial_rate);
                    }
                } else if alignment.abs() > 1e-12 && travel.length_squared() > PARALLEL {
                    let height = (cap - axial) / alignment;
                    at(-travel.dot(radial + height * side) / travel.length_squared());
                }
            }
        }
    }

    // Height along `normal` at which the column through `foot` enters the
    // cylinder, if it meets it at all. A column grazing an edge or rim within
    // `LEVEL` still meets it: the lowest point over a boundary is often exactly
    // such a single-point touch, which rounding would otherwise discard.
    fn column_entry(&self, foot: DVec3, normal: DVec3) -> Option<f64> {
        let offset = foot - self.center;
        let axial = self.axis.dot(offset);
        let alignment = self.axis.dot(normal);
        let radial = offset - axial * self.axis;
        let side = normal - alignment * self.axis;
        let [mut low, mut high] = [f64::NEG_INFINITY, f64::INFINITY];
        let quadratic = side.length_squared();
        if quadratic > PARALLEL {
            let half = radial.dot(side);
            let discriminant =
                half * half - quadratic * (radial.length_squared() - self.radius * self.radius);
            // Missing the side by a distance d leaves about −2·r·d·quadratic.
            if discriminant < -2.0 * self.radius * LEVEL * quadratic {
                return None;
            }
            let root = discriminant.max(0.0).sqrt();
            [low, high] = [(-half - root) / quadratic, (-half + root) / quadratic];
        } else if radial.length() > self.radius + LEVEL {
            return None;
        }
        if alignment.abs() > 1e-12 {
            let [a, b] = [
                (-self.half_length - axial) / alignment,
                (self.half_length - axial) / alignment,
            ];
            low = low.max(a.min(b));
            high = high.min(a.max(b));
        } else if axial.abs() > self.half_length + LEVEL {
            return None;
        }
        (low.is_finite() && low <= high + LEVEL / alignment.abs().max(1e-3)).then_some(low)
    }
}

// Smallest height `h` with |fixed + h·rising| = radius.
fn lower_root(rising: DVec3, fixed: DVec3, radius: f64) -> Option<f64> {
    let quadratic = rising.length_squared();
    if quadratic <= PARALLEL {
        return None;
    }
    let half = fixed.dot(rising);
    let discriminant = half * half - quadratic * (fixed.length_squared() - radius * radius);
    (discriminant >= 0.0).then(|| (-half - discriminant.sqrt()) / quadratic)
}

fn point(body_point: DVec3, normal: DVec3, separation: f64) -> TriangleContactPoint {
    TriangleContactPoint {
        triangle_point: body_point - separation * normal,
        body_point,
        normal,
        depth: (-separation).max(0.0),
    }
}

fn separation(point: &TriangleContactPoint) -> f64 {
    (point.body_point - point.triangle_point).dot(point.normal)
}

// Distance between a segment and a triangle. Apart from a crossing, the
// closest pair has an endpoint over the triangle or a point on one of its edges.
fn segment_triangle_distance(segment: [DVec3; 2], triangle: [DVec3; 3], normal: DVec3) -> f64 {
    let plane = normal.dot(triangle[0]);
    let heights = segment.map(|end| normal.dot(end) - plane);
    if heights[0] * heights[1] < 0.0 {
        let crossing =
            segment[0] + (segment[1] - segment[0]) * (heights[0] / (heights[0] - heights[1]));
        if inside(triangle, normal, crossing) {
            return 0.0;
        }
    }
    let over = segment
        .iter()
        .zip(heights)
        .filter(|&(&end, height)| inside(triangle, normal, end - height * normal))
        .map(|(_, height)| height.abs());
    let edges = (0..3).map(|edge| {
        let side = [triangle[edge], triangle[(edge + 1) % 3]];
        if (segment[1] - segment[0]).length_squared() <= PARALLEL {
            let along = side[1] - side[0];
            let t = (along.dot(segment[0] - side[0]) / along.length_squared()).clamp(0.0, 1.0);
            return segment[0].distance(side[0] + t * along);
        }
        let [a, b] = super::pair::closest_segment_points(segment, side);
        a.distance(b)
    });
    over.chain(edges).fold(f64::INFINITY, f64::min)
}

fn inside(triangle: [DVec3; 3], normal: DVec3, point: DVec3) -> bool {
    (0..3).all(|edge| {
        let from = triangle[edge];
        normal
            .cross(triangle[(edge + 1) % 3] - from)
            .dot(point - from)
            >= 0.0
    })
}

// Unique points in the order found, which keeps contact identities stable, at
// most five: the diagonal extremes a polytope manifold keeps, and the deepest.
fn reduce(candidates: Vec<TriangleContactPoint>, normal: DVec3) -> Vec<TriangleContactPoint> {
    let close = |a: &TriangleContactPoint, b: &TriangleContactPoint| {
        a.triangle_point.distance(b.triangle_point) < 1e-5
    };
    let mut unique: Vec<TriangleContactPoint> = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        match unique.iter_mut().find(|point| close(point, &candidate)) {
            Some(point) if separation(&candidate) < separation(point) => *point = candidate,
            Some(_) => {}
            None => unique.push(candidate),
        }
    }
    if unique.len() <= 5 {
        return unique;
    }
    let [u, v] = contact_tangents(normal);
    let mut kept: Vec<TriangleContactPoint> = Vec::with_capacity(4);
    for direction in [u + v, u - v, -u - v, -u + v] {
        let extreme = unique
            .iter()
            .copied()
            .max_by(|a, b| {
                a.triangle_point
                    .dot(direction)
                    .total_cmp(&b.triangle_point.dot(direction))
            })
            .expect("more than five unique points");
        if !kept.iter().any(|point| close(point, &extreme)) {
            kept.push(extreme);
        }
    }
    let deepest = unique
        .iter()
        .copied()
        .min_by(|a, b| separation(a).total_cmp(&separation(b)))
        .expect("more than five unique points");
    if !kept.iter().any(|point| close(point, &deepest)) {
        kept.push(deepest);
    }
    kept
}

#[cfg(test)]
mod tests;
