//! Fixed-axis Taylor certificates for a finite convex trajectory.

use super::{ContactGeometryError, ContactPolytope, envelope::interval::Interval};
use bevy_math::{DQuat, DVec3};

type Vector = [Interval; 3];

fn vector(value: DVec3) -> Vector {
    value.to_array().map(Interval::point)
}

fn difference(a: DVec3, b: DVec3) -> Vector {
    std::array::from_fn(|i| Interval::point(a[i]).sub(Interval::point(b[i])))
}

fn add(a: Vector, b: Vector) -> Vector {
    std::array::from_fn(|i| a[i].add(b[i]))
}

fn cross(a: Vector, b: Vector) -> Vector {
    std::array::from_fn(|i| {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        a[j].mul(b[k]).sub(a[k].mul(b[j]))
    })
}

fn dot(a: Vector, b: DVec3) -> Interval {
    (0..3).fold(Interval::point(0.0), |sum, i| sum.add(a[i].scale(b[i])))
}

fn rotate(rotation: DQuat, value: DVec3) -> Vector {
    // Match glam's quaternion vector action, enclosing every multiply/add.
    // Do not replace |q|² by exactly one: normalized stored quaternions can retain
    // roundoff. No previously rounded world axis or anchor is reused.
    let q = vector(rotation.xyz());
    let radial = dot(q, value).scale(2.0);
    let scale = Interval::point(rotation.w)
        .square()
        .sub(dot(q, rotation.xyz()));
    let cross_scale = Interval::point(rotation.w).scale(2.0);
    add(
        add(
            vector(value).map(|x| x.mul(scale)),
            q.map(|x| x.mul(radial)),
        ),
        cross(q, vector(value)).map(|x| x.mul(cross_scale)),
    )
}

/// World linear velocity at a body origin and angular velocity, enclosed with
/// outward arithmetic while traversing a mechanism. Rates use path-fraction units.
/// This is a geometric derivative, independent of mass, forces or contact impulses.
#[derive(Clone, Copy, Debug)]
pub struct ContactVelocity {
    linear: Vector,
    angular: Vector,
}

impl ContactVelocity {
    /// Starts with the supplied world translation and rotation rates.
    pub fn new(linear: DVec3, angular: DVec3) -> Self {
        Self {
            linear: vector(linear),
            angular: vector(angular),
        }
    }

    /// Moves the linear-velocity reference between two points on a rigid body.
    #[must_use]
    pub fn shifted(self, from: DVec3, to: DVec3) -> Self {
        Self {
            linear: add(self.linear, cross(self.angular, difference(to, from))),
            ..self
        }
    }

    /// Adds relative translation along an axis in a unit-quaternion parent frame,
    /// retaining the supplied local axis length and enclosing frame arithmetic.
    #[must_use]
    pub fn translated(self, frame: DQuat, axis: DVec3, rate: f64) -> Self {
        Self {
            linear: add(self.linear, rotate(frame, axis).map(|x| x.scale(rate))),
            ..self
        }
    }

    /// Adds rotation about a parent-local anchor to a velocity at the child origin.
    /// The local axis and parent quaternion must be normalized. Frame transforms
    /// and anchor subtraction use outward intervals, including cancellation.
    #[allow(clippy::too_many_arguments)] // Explicit parent frame, joint geometry, rate and child origin.
    #[must_use]
    pub fn rotated(
        self,
        parent: DVec3,
        frame: DQuat,
        axis: DVec3,
        anchor: DVec3,
        rate: f64,
        origin: DVec3,
    ) -> Self {
        let rotation = rotate(frame, axis).map(|x| x.scale(rate));
        let offset = add(
            difference(origin, parent),
            rotate(frame, anchor).map(|x| x.scale(-1.0)),
        );
        Self {
            linear: add(self.linear, cross(rotation, offset)),
            angular: add(self.angular, rotation),
        }
    }

    /// Encloses the signed velocity of a material point projected on a world axis.
    ///
    /// # Errors
    /// Rejects non-finite inputs or arithmetic overflow.
    pub fn point_projection(
        self,
        origin: DVec3,
        point: DVec3,
        axis: DVec3,
    ) -> Result<[f64; 2], ContactGeometryError> {
        let projected = dot(self.shifted(origin, point).linear, axis).finite()?;
        Ok([projected.lo, projected.hi])
    }
}

impl ContactPolytope {
    /// Certifies a separated prefix using each vertex's signed axis velocity and
    /// an upper bound on its acceleration magnitude over the complete interval.
    /// A fixed-axis gap satisfies g(t) >= g(0) + g'(0)t - A t²/2. The lower
    /// polynomial is concave, so positive endpoints certify every intervening time.
    /// All projections and polynomial operations round outward. Bounded bisection
    /// only proposes a shorter certificate; zero means no certified progress.
    /// The caller must supply a valid acceleration bound and the actual derivative
    /// of the same geometric trajectory. This does not certify force integration.
    ///
    /// # Errors
    /// Rejects invalid geometry, non-finite kinematics or negative time/acceleration.
    pub fn triangle_motion_prefix(
        &self,
        triangle: [DVec3; 3],
        origin: DVec3,
        velocity: ContactVelocity,
        point_acceleration: f64,
        maximum: f64,
    ) -> Result<f64, ContactGeometryError> {
        if !origin.is_finite()
            || !point_acceleration.is_finite()
            || point_acceleration < 0.0
            || !maximum.is_finite()
            || maximum < 0.0
        {
            return Err(ContactGeometryError);
        }
        for component in velocity.linear.into_iter().chain(velocity.angular) {
            component.finite()?;
        }
        let (axis, _) = self.triangle_separating_axis(triangle)?;
        // Recenter before projection, preserving small gaps far from the origin.
        let surface = triangle
            .into_iter()
            .try_fold(f64::NEG_INFINITY, |max, point| {
                Ok::<_, ContactGeometryError>(
                    max.max(dot(difference(point, triangle[0]), axis).finite()?.hi),
                )
            })?;
        let coefficients = self
            .vertices
            .iter()
            .map(|&point| {
                let gap = dot(difference(point, triangle[0]), axis)
                    .sub(Interval::point(surface))
                    .finite()?;
                let rate = dot(velocity.shifted(origin, point).linear, axis).finite()?;
                Ok::<_, ContactGeometryError>((gap, rate))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if coefficients.iter().any(|(gap, _)| gap.lo <= 0.0) {
            return Ok(0.0);
        }
        let acceleration = super::envelope::interval::dot(axis, axis)
            .sqrt()?
            .scale(point_acceleration)
            .scale(0.5)
            .finite()?;
        let certified = |time: f64| {
            let t = Interval::point(time);
            let curvature = acceleration.mul(t.square());
            coefficients
                .iter()
                .all(|&(gap, rate)| gap.add(rate.mul(t)).sub(curvature).lo > 0.0)
        };
        if certified(maximum) {
            return Ok(maximum);
        }
        Ok(largest_certified(maximum, certified))
    }

    /// Certifies a separated prefix against another moving convex solid, on the
    /// fixed axis of their current largest separating-axis gap. Each solid's
    /// projection moves with its vertices' signed axis velocities, and
    /// `point_acceleration` bounds both solids' point accelerations together.
    /// Bodies moving together therefore keep a tiny gap for the whole interval,
    /// where dividing the gap by both absolute speeds would barely advance.
    ///
    /// # Errors
    /// Rejects invalid geometry, non-finite kinematics or negative time/acceleration.
    #[allow(clippy::too_many_arguments)] // Both solids' origins and velocities.
    pub fn convex_motion_prefix(
        &self,
        origin: DVec3,
        velocity: ContactVelocity,
        other: &Self,
        other_origin: DVec3,
        other_velocity: ContactVelocity,
        point_acceleration: f64,
        maximum: f64,
    ) -> Result<f64, ContactGeometryError> {
        if !origin.is_finite()
            || !other_origin.is_finite()
            || !point_acceleration.is_finite()
            || point_acceleration < 0.0
            || !maximum.is_finite()
            || maximum < 0.0
        {
            return Err(ContactGeometryError);
        }
        for component in [velocity, other_velocity]
            .into_iter()
            .flat_map(|velocity| velocity.linear.into_iter().chain(velocity.angular))
        {
            component.finite()?;
        }
        let axis = self.convex_separation(other)?.axis;
        // Recenter on one vertex of the other solid so small gaps survive far
        // from the world origin.
        let reference = other.vertices[0];
        let projections = |polytope: &Self, origin: DVec3, velocity: ContactVelocity| {
            polytope
                .vertices
                .iter()
                .map(|&point| {
                    let position = dot(difference(point, reference), axis).finite()?;
                    let rate = dot(velocity.shifted(origin, point).linear, axis).finite()?;
                    Ok::<_, ContactGeometryError>((position, rate))
                })
                .collect::<Result<Vec<_>, _>>()
        };
        let own = projections(self, origin, velocity)?;
        let opposing = projections(other, other_origin, other_velocity)?;
        let acceleration = super::envelope::interval::dot(axis, axis)
            .sqrt()?
            .scale(point_acceleration)
            .scale(0.5)
            .finite()?;
        let certified = |time: f64| {
            let t = Interval::point(time);
            let lowest = own
                .iter()
                .map(|&(position, rate)| position.add(rate.mul(t)).lo)
                .fold(f64::INFINITY, f64::min);
            let highest = opposing
                .iter()
                .map(|&(position, rate)| position.add(rate.mul(t)).hi)
                .fold(f64::NEG_INFINITY, f64::max);
            Interval::point(lowest)
                .sub(Interval::point(highest))
                .sub(acceleration.mul(t.square()))
                .lo
                > 0.0
        };
        if !certified(0.0) {
            return Ok(0.0);
        }
        Ok(largest_certified(maximum, certified))
    }
}

// Concave certificates hold on a prefix, so bisection finds its bounded end.
fn largest_certified(maximum: f64, certified: impl Fn(f64) -> bool) -> f64 {
    if certified(maximum) {
        return maximum;
    }
    let (mut lower, mut upper) = (0.0, maximum);
    for _ in 0..48 {
        let middle = lower + (upper - lower) * 0.5;
        if certified(middle) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    lower
}

#[cfg(test)]
mod tests;
