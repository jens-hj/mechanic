//! The no-slip row of a mesh between two bodies.
//!
//! Meshing parts never touch; they are coupled magnetically at their pitch
//! point. Each mesh is one bilateral generalized row `S(a) − S(b) = 0`, where a
//! side's surface speed `S` is the velocity of its pitch point along the common
//! tangent, plus the thread advance times its spin about the thread axis for a
//! worm or screw. Because the row is written in terms of pitch points rather
//! than a housing, it holds for gears on a fixed frame, on a turning carrier in
//! a planetary set, on a rack, across a worm, and for a nut on a thread alike.

use bevy_math::DVec3;
use mechanic_core::{CompiledGearLink, GearLinkKind};

use crate::{BodyPose, MachineDynamics, MachineKinematics, PhysicsError};

/// What a mesh row needs from a machine model: where its bodies are and the
/// generalized rows of a point's velocity and a body's spin.
pub(crate) trait MeshModel {
    fn pose(&self, body: usize) -> Option<BodyPose>;
    fn point_row(
        &self,
        body: usize,
        point: DVec3,
        direction: DVec3,
    ) -> Result<Vec<f64>, PhysicsError>;
    fn angular_row(&self, body: usize, direction: DVec3) -> Result<Vec<f64>, PhysicsError>;
}

impl MeshModel for MachineKinematics<'_> {
    fn pose(&self, body: usize) -> Option<BodyPose> {
        self.poses.get(body).copied()
    }

    fn point_row(
        &self,
        body: usize,
        point: DVec3,
        direction: DVec3,
    ) -> Result<Vec<f64>, PhysicsError> {
        Self::point_row(self, body, point, direction)
    }

    fn angular_row(&self, body: usize, direction: DVec3) -> Result<Vec<f64>, PhysicsError> {
        Self::angular_row(self, body, direction)
    }
}

impl MeshModel for MachineDynamics {
    fn pose(&self, body: usize) -> Option<BodyPose> {
        self.poses.get(body).copied()
    }

    fn point_row(
        &self,
        body: usize,
        point: DVec3,
        direction: DVec3,
    ) -> Result<Vec<f64>, PhysicsError> {
        Self::point_row(self, body, point, direction)
    }

    fn angular_row(&self, body: usize, direction: DVec3) -> Result<Vec<f64>, PhysicsError> {
        Self::angular_row(self, body, direction)
    }
}

/// One side's contribution to the mesh row.
struct Side {
    body: usize,
    /// World point whose velocity along the tangent is the surface speed.
    point: DVec3,
    /// Metres of surface travel per radian of spin about the tangent, for a
    /// thread; zero for a toothed wheel or rack.
    advance: f64,
}

fn perpendicular_part(vector: DVec3, axis: DVec3) -> DVec3 {
    vector - axis * vector.dot(axis)
}

/// The generalized row holding a mesh at the model's current pose, or `None`
/// when the pose is degenerate or the tree already holds the direction.
///
/// # Errors
///
/// Returns [`PhysicsError`] when a side names a body the model lacks or the
/// pose is not finite.
#[expect(
    clippy::too_many_lines,
    reason = "one match over the four mesh kinds reads better whole"
)]
pub(crate) fn mesh_jacobian<M: MeshModel>(
    model: &M,
    link: &CompiledGearLink,
) -> Result<Option<Vec<f64>>, PhysicsError> {
    let side = |index: usize| {
        let side = link.sides[index];
        let body = side.compound as usize;
        let pose = model.pose(body).ok_or(PhysicsError::InvalidDynamics)?;
        Ok::<_, PhysicsError>((
            body,
            pose.position + pose.rotation * side.local_center.as_dvec3(),
            (pose.rotation * side.local_axis.as_dvec3()).normalize(),
            f64::from(side.pitch_radius),
        ))
    };
    let (a, center_a, axis_a, radius_a) = side(0)?;
    let (b, center_b, axis_b, radius_b) = side(1)?;
    let advance = f64::from(link.advance);
    // Direction from a wheel's axis towards the other side, in its pitch plane.
    let toward = |from: DVec3, axis: DVec3, target: DVec3| {
        let direction = perpendicular_part(target - from, axis);
        (direction.length_squared() > 1.0e-12).then(|| direction.normalize())
    };
    let (tangent, sides) = match link.kind {
        GearLinkKind::Gears => {
            let (Some(toward_b), Some(toward_a)) = (
                toward(center_a, axis_a, center_b),
                toward(center_b, axis_b, center_a),
            ) else {
                return Ok(None);
            };
            // A wheel's pitch point lies towards its partner, except that a
            // pinion inside a ring touches it on its far side. Internal teeth
            // carry a negative radius, so the partner's sign places the point.
            let point_a = center_a + toward_b * radius_a.abs() * radius_b.signum();
            let point_b = center_b + toward_a * radius_b.abs() * radius_a.signum();
            (
                axis_a.cross(toward_b).normalize(),
                [
                    Side {
                        body: a,
                        point: point_a,
                        advance: 0.0,
                    },
                    Side {
                        body: b,
                        point: point_b,
                        advance: 0.0,
                    },
                ],
            )
        }
        GearLinkKind::Rack => {
            // The rack side's axis is its pitch plane's outward normal.
            let Some(down) = toward(center_a, axis_a, center_a - axis_b) else {
                return Ok(None);
            };
            let pitch_point = center_a + down * radius_a;
            (
                axis_a.cross(down).normalize(),
                [
                    Side {
                        body: a,
                        point: pitch_point,
                        advance: 0.0,
                    },
                    Side {
                        body: b,
                        point: pitch_point - axis_b * (pitch_point - center_b).dot(axis_b),
                        advance: 0.0,
                    },
                ],
            )
        }
        GearLinkKind::Worm => {
            let Some(toward_worm) = toward(center_a, axis_a, center_b) else {
                return Ok(None);
            };
            (
                axis_b,
                [
                    Side {
                        body: a,
                        point: center_a + toward_worm * radius_a,
                        advance,
                    },
                    Side {
                        body: b,
                        point: center_b,
                        advance,
                    },
                ],
            )
        }
        GearLinkKind::Screw => (
            axis_b,
            [
                Side {
                    body: a,
                    point: center_a,
                    advance,
                },
                Side {
                    body: b,
                    point: center_b,
                    advance,
                },
            ],
        ),
    };
    if !tangent.is_finite() {
        return Ok(None);
    }
    let mut jacobian = Vec::new();
    let mut reference = 0.0;
    for (side, sign) in sides.iter().zip([1.0, -1.0]) {
        let mut row = model.point_row(side.body, side.point, tangent)?;
        if side.advance != 0.0 {
            for (value, spin) in row.iter_mut().zip(model.angular_row(side.body, tangent)?) {
                *value += side.advance * spin;
            }
        }
        reference += row.iter().map(|value| value * value).sum::<f64>();
        if jacobian.is_empty() {
            jacobian = row;
            for value in &mut jacobian {
                *value *= sign;
            }
        } else {
            for (value, other) in jacobian.iter_mut().zip(row) {
                *value += sign * other;
            }
        }
    }
    // A mesh inside one rigid body, or one the tree already holds, cancels to
    // rounding noise; solving that noise would fling the machine apart.
    let length = jacobian.iter().map(|value| value * value).sum::<f64>();
    Ok((length > 1.0e-10 * reference).then_some(jacobian))
}
