//! Exact tree trajectories and conservative motion bounds for continuous queries.

use bevy_math::DVec3;
use mechanic_core::{CompiledCreation, ContactVelocity};

use crate::{
    BodyPose, MachineDynamics, MachineState, PhysicsError, free_motion::advance_positions,
};

/// Bounds on body-origin translation and angular motion per unit path fraction.
/// They include every ancestor joint, not just endpoint displacement.
#[derive(Clone, Copy, Debug, Default)]
pub struct MotionBound {
    /// Maximum speed of the body origin over the complete normalized path.
    pub origin_speed: f64,
    /// Maximum world angular speed over that path.
    pub angular_speed: f64,
    /// Maximum acceleration of the body origin over the normalized path.
    pub origin_acceleration: f64,
    /// Maximum angular acceleration caused by moving ancestor axes.
    pub angular_acceleration: f64,
}

impl MotionBound {
    /// Conservative acceleration magnitude of every point within the radius.
    pub fn point_acceleration(self, radius: f64) -> f64 {
        if !self.point_speed(radius).is_finite()
            || !self.origin_acceleration.is_finite()
            || self.origin_acceleration < 0.0
            || !self.angular_acceleration.is_finite()
            || self.angular_acceleration < 0.0
        {
            return f64::INFINITY;
        }
        upper_add(
            self.origin_acceleration,
            upper_product(
                upper_add(
                    self.angular_acceleration,
                    upper_product(self.angular_speed, self.angular_speed),
                ),
                radius,
            ),
        )
    }

    /// Conservative speed of any point within `radius` of the body origin.
    /// Returns infinity for invalid inputs; callers must reject an infinite bound.
    pub fn point_speed(self, radius: f64) -> f64 {
        if !radius.is_finite()
            || radius < 0.0
            || !self.origin_speed.is_finite()
            || self.origin_speed < 0.0
            || !self.angular_speed.is_finite()
            || self.angular_speed < 0.0
        {
            return f64::INFINITY;
        }
        upper_add(self.origin_speed, upper_product(self.angular_speed, radius))
    }
}

/// A candidate path specified by generalized displacements, including unwrapped
/// world root rotation and joint angles. Poses are reconstructed at every query;
/// this path never interpolates disconnected body endpoint poses.
/// It describes geometry only and neither integrates forces nor publishes state.
pub struct MachineMotion<'a> {
    pub(crate) creation: &'a CompiledCreation,
    initial: &'a MachineState,
    displacement: &'a [f64],
    generation: u64,
    start: Vec<BodyPose>,
    end: Vec<BodyPose>,
    preparation_pose_evaluations: usize,
    bounds: Vec<MotionBound>,
}

impl<'a> MachineMotion<'a> {
    /// Prepares the exact tree path and one root-before-child motion-bound pass.
    /// Generalized displacement uses the same row order as state velocities.
    ///
    /// # Errors
    /// Rejects invalid state/dimensions, non-finite motion, or overflow. A closed
    /// loop follows its tree path; closure equations are not projected.
    #[allow(clippy::too_many_lines)] // Root and joint bounds follow the same ordered reconstruction schedule.
    pub fn new(
        creation: &'a CompiledCreation,
        generation: u64,
        initial: &'a MachineState,
        displacement: &'a [f64],
    ) -> Result<Self, PhysicsError> {
        if displacement.len() != creation.dynamics.elimination_parent.len()
            || displacement.iter().any(|value| !value.is_finite())
            || initial.velocities.len() != displacement.len()
        {
            return Err(PhysicsError::InvalidDynamics);
        }
        MachineDynamics::validate_configuration(creation, &initial.poses, &initial.coordinates)?;
        // Use the same normalized root path at fraction zero and later samples.
        // A merely near-unit input quaternion must not create a start discontinuity.
        let mut beginning = initial.clone();
        beginning.velocities.copy_from_slice(displacement);
        advance_positions(creation, &mut beginning, 0.0);
        let start =
            MachineDynamics::reconstruct_poses(creation, &beginning.poses, &beginning.coordinates)?;
        let mut bounds = vec![MotionBound::default(); start.len()];
        for &body in &creation.dynamics.preorder {
            let topology = creation.loop_topology.body_parents[body];
            let rows = creation.dynamics.body_velocities[body].clone();
            if topology.is_root {
                if !rows.is_empty() {
                    let motion = &displacement[rows];
                    bounds[body] = MotionBound {
                        origin_speed: upper_length(DVec3::new(motion[0], motion[1], motion[2])),
                        angular_speed: upper_length(DVec3::new(motion[3], motion[4], motion[5])),
                        ..MotionBound::default()
                    };
                }
                continue;
            }
            let parent = topology.parent_body as usize;
            let row = creation.dynamics.body_bearings[body].ok_or(PhysicsError::InvalidDynamics)?;
            let bearing = creation.bearings[row];
            let coordinate = bearing
                .coordinate_index
                .ok_or(PhysicsError::InvalidDynamics)? as usize;
            let change = displacement[rows.start].abs();
            let inherited = bounds[parent];
            let (parent_anchor, child_anchor, axis) = if topology.bearing_direction == 0 {
                (
                    bearing.local_anchor_a,
                    bearing.local_anchor_b,
                    bearing.local_axis_a,
                )
            } else {
                (
                    bearing.local_anchor_b,
                    bearing.local_anchor_a,
                    bearing.local_axis_b,
                )
            };
            if bearing.kind.is_translational() {
                let inverse = creation.compounds[parent]
                    .root_rotation
                    .as_dquat()
                    .inverse();
                let bind = inverse
                    * (creation.compounds[body].root_translation
                        - creation.compounds[parent].root_translation)
                        .as_dvec3();
                let extent = initial.coordinates[coordinate].abs().max(
                    (initial.coordinates[coordinate] + displacement[rows.start])
                        .abs()
                        .next_up(),
                );
                let axis_length = upper_length(axis.as_dvec3());
                let reach = upper_add(upper_length(bind), upper_product(extent, axis_length));
                bounds[body] = MotionBound {
                    origin_speed: upper_add(
                        upper_add(
                            inherited.origin_speed,
                            upper_product(inherited.angular_speed, reach),
                        ),
                        upper_product(change, axis_length),
                    ),
                    angular_speed: inherited.angular_speed,
                    angular_acceleration: inherited.angular_acceleration,
                    origin_acceleration: upper_add(
                        inherited.point_acceleration(reach),
                        upper_product(
                            2.0,
                            upper_product(
                                inherited.angular_speed,
                                upper_product(change, axis_length),
                            ),
                        ),
                    ),
                };
            } else {
                let child_reach = upper_length(child_anchor.as_dvec3());
                let reach = upper_add(upper_length(parent_anchor.as_dvec3()), child_reach);
                let angular_speed = upper_add(inherited.angular_speed, change);
                let angular_acceleration = upper_add(
                    inherited.angular_acceleration,
                    upper_product(inherited.angular_speed, change),
                );
                bounds[body] = MotionBound {
                    origin_speed: upper_add(
                        upper_add(
                            inherited.origin_speed,
                            upper_product(inherited.angular_speed, reach),
                        ),
                        upper_product(change, child_reach),
                    ),
                    angular_speed,
                    angular_acceleration,
                    origin_acceleration: upper_add(
                        inherited.point_acceleration(upper_length(parent_anchor.as_dvec3())),
                        upper_product(
                            upper_add(
                                angular_acceleration,
                                upper_product(angular_speed, angular_speed),
                            ),
                            child_reach,
                        ),
                    ),
                };
            }
        }
        if bounds.iter().any(|bound| {
            !bound.point_speed(0.0).is_finite() || !bound.point_acceleration(0.0).is_finite()
        }) {
            return Err(PhysicsError::InvalidDynamics);
        }
        let mut path = Self {
            creation,
            initial,
            displacement,
            generation,
            start,
            end: Vec::new(),
            preparation_pose_evaluations: 1,
            bounds,
        };
        path.end = path.poses_at(1.0)?;
        path.preparation_pose_evaluations += 1;
        Ok(path)
    }

    /// Topology generation supplied with this candidate path.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Reconstructed poses at the beginning of the candidate path.
    pub fn initial_poses(&self) -> &[BodyPose] {
        &self.start
    }

    /// Reconstructed endpoint already validated during path preparation. Equal
    /// endpoint rotations alone do not establish a translation-only trajectory.
    pub fn final_poses(&self) -> &[BodyPose] {
        &self.end
    }

    /// Whole-tree reconstructions performed while preparing this path, separate
    /// from any later CCD or envelope query work.
    pub fn preparation_pose_evaluations(&self) -> usize {
        self.preparation_pose_evaluations
    }

    /// Body-indexed conservative motion bounds over the complete path.
    pub fn bounds(&self) -> &[MotionBound] {
        &self.bounds
    }

    // Exact spatial derivative of the prescribed generalized drift, at body
    // origins (not centers of mass). Traversal does not assemble inertia/Jacobians.
    pub(crate) fn velocities_at_poses(&self, poses: &[BodyPose]) -> Vec<ContactVelocity> {
        let mut velocities = vec![ContactVelocity::new(DVec3::ZERO, DVec3::ZERO); poses.len()];
        for &body in &self.creation.dynamics.preorder {
            let topology = self.creation.loop_topology.body_parents[body];
            let rows = self.creation.dynamics.body_velocities[body].clone();
            if topology.is_root {
                if !rows.is_empty() {
                    let rates = &self.displacement[rows];
                    velocities[body] = ContactVelocity::new(
                        DVec3::new(rates[0], rates[1], rates[2]),
                        DVec3::new(rates[3], rates[4], rates[5]),
                    );
                }
                continue;
            }
            let parent = topology.parent_body as usize;
            let bearing = self.creation.bearings
                [self.creation.dynamics.body_bearings[body].expect("validated tree bearing")];
            let (anchor, axis, sign) = if topology.bearing_direction == 0 {
                (bearing.local_anchor_a, bearing.local_axis_a, 1.0)
            } else {
                (bearing.local_anchor_b, bearing.local_axis_b, -1.0)
            };
            let inherited =
                velocities[parent].shifted(poses[parent].position, poses[body].position);
            let rate = sign * self.displacement[rows.start];
            velocities[body] = if bearing.kind.is_translational() {
                inherited.translated(poses[parent].rotation, axis.as_dvec3(), rate)
            } else {
                inherited.rotated(
                    poses[parent].position,
                    poses[parent].rotation,
                    axis.as_dvec3().normalize(),
                    anchor.as_dvec3(),
                    rate,
                    poses[body].position,
                )
            };
        }
        velocities
    }

    /// Reconstructs all body poses at a normalized path fraction, without inertia.
    ///
    /// # Errors
    /// Rejects fractions outside [0, 1] and invalid reconstructed geometry.
    pub fn poses_at(&self, fraction: f64) -> Result<Vec<BodyPose>, PhysicsError> {
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return Err(PhysicsError::InvalidDynamics);
        }
        let mut state = self.initial.clone();
        state.velocities.copy_from_slice(self.displacement);
        advance_positions(self.creation, &mut state, fraction);
        MachineDynamics::reconstruct_poses(self.creation, &state.poses, &state.coordinates)
    }
}

// Positive bounds round outward at every arithmetic step; preserve exact zero
// so a stationary path remains stationary. Overflow propagates to validation.
fn upper_add(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else {
        (a + b).next_up()
    }
}

fn upper_product(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else {
        (a * b).next_up()
    }
}

fn upper_length(vector: DVec3) -> f64 {
    let squared = vector.to_array().into_iter().fold(0.0, |sum, component| {
        upper_add(sum, upper_product(component.abs(), component.abs()))
    });
    if squared == 0.0 {
        0.0
    } else {
        squared.sqrt().next_up()
    }
}

#[cfg(test)]
mod tests;
