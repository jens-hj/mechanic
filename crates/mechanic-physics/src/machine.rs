//! Pose-dependent reduced-coordinate inertia and point Jacobians.

use bevy_math::{DMat3, DQuat, DVec3};
use mechanic_core::CompiledCreation;

use crate::{DynamicsFactor, PhysicsError};

/// Double-precision body pose for the CPU reference experiment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyPose {
    /// World position of the compiled body origin.
    pub position: DVec3,
    /// Unit body-to-world rotation.
    pub rotation: DQuat,
}

/// World-space twist or acceleration at a body's centre of mass.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpatialMotion {
    /// Linear component.
    pub linear: DVec3,
    /// Angular component.
    pub angular: DVec3,
}

/// Coupled numerical dynamics in a fixed pose. Rebuild after every pose change.
/// The dense reference implementation is bounded at 512 generalized velocities;
/// compiled symbolic storage itself has no such bound.
#[derive(Clone, Debug)]
pub struct MachineDynamics {
    /// Body-indexed poses reconstructed in root-before-child order.
    pub poses: Vec<BodyPose>,
    /// Row-major coupled generalized mass matrix.
    pub mass_matrix: Vec<f64>,
    jacobians: Vec<Vec<SpatialMotion>>,
    centers: Vec<DVec3>,
    size: usize,
}

impl MachineDynamics {
    /// Reconstructs an exact tree pose and its coupled mass matrix. Closure
    /// equations remain separate constraints; this does not project closed loops.
    /// `roots` is body-indexed; only canonical roots are consumed. Joint positions
    /// use the existing compiled coordinate order and metres/radians convention.
    ///
    /// # Errors
    /// Rejects non-finite state, wrong row counts, and oversized reference scenes.
    pub fn assemble(
        creation: &CompiledCreation,
        roots: &[BodyPose],
        coordinates: &[f64],
    ) -> Result<Self, PhysicsError> {
        let dynamics = &creation.dynamics;
        let size = dynamics.elimination_parent.len();
        if size > 512 {
            return Err(PhysicsError::ReferenceCapacity);
        }
        let poses = Self::reconstruct_poses(creation, roots, coordinates)?;
        let centers = poses
            .iter()
            .zip(&dynamics.inertias)
            .map(|(pose, inertia)| pose.position + pose.rotation * inertia.center.as_dvec3())
            .collect::<Vec<_>>();
        let mut jacobians = vec![vec![SpatialMotion::default(); size]; poses.len()];
        for &body in &dynamics.preorder {
            let topology = creation.loop_topology.body_parents[body];
            let velocities = dynamics.body_velocities[body].clone();
            if topology.is_root {
                if velocities.is_empty() {
                    continue;
                }
                for (index, axis) in [DVec3::X, DVec3::Y, DVec3::Z].into_iter().enumerate() {
                    jacobians[body][velocities.start + index].linear = axis;
                    jacobians[body][velocities.start + 3 + index] = SpatialMotion {
                        linear: axis.cross(centers[body] - poses[body].position),
                        angular: axis,
                    };
                }
            } else {
                let parent = topology.parent_body as usize;
                let arm = centers[body] - centers[parent];
                #[expect(clippy::needless_range_loop)]
                // Parent and child rows share the same indexed arena.
                for column in 0..size {
                    let motion = jacobians[parent][column];
                    jacobians[body][column] = SpatialMotion {
                        linear: motion.linear + motion.angular.cross(arm),
                        angular: motion.angular,
                    };
                }
                let bearing = creation.bearings
                    [dynamics.body_bearings[body].ok_or(PhysicsError::InvalidDynamics)?];
                let (anchor, axis, sign) = if topology.bearing_direction == 0 {
                    (bearing.local_anchor_a, bearing.local_axis_a, 1.0)
                } else {
                    (bearing.local_anchor_b, bearing.local_axis_b, -1.0)
                };
                let axis = poses[parent].rotation * axis.as_dvec3().normalize() * sign;
                jacobians[body][velocities.start] = if bearing.kind.is_translational() {
                    SpatialMotion {
                        linear: axis,
                        angular: DVec3::ZERO,
                    }
                } else {
                    let anchor =
                        poses[parent].position + poses[parent].rotation * anchor.as_dvec3();
                    SpatialMotion {
                        linear: axis.cross(centers[body] - anchor),
                        angular: axis,
                    }
                };
            }
        }
        let mut mass_matrix = vec![0.0; size * size];
        for (body, inertia) in dynamics.inertias.iter().enumerate() {
            if creation.compounds[body].is_static {
                continue;
            }
            let rotation = DMat3::from_quat(poses[body].rotation);
            let world_inertia = rotation * inertia.rotational.as_dmat3() * rotation.transpose();
            for row in 0..size {
                for column in 0..=row {
                    let a = jacobians[body][row];
                    let b = jacobians[body][column];
                    mass_matrix[row * size + column] += f64::from(inertia.mass)
                        * a.linear.dot(b.linear)
                        + a.angular.dot(world_inertia * b.angular);
                }
            }
        }
        for row in 0..size {
            for column in 0..row {
                mass_matrix[column * size + row] = mass_matrix[row * size + column];
            }
        }
        if mass_matrix.iter().any(|x| !x.is_finite()) {
            return Err(PhysicsError::InvalidDynamics);
        }
        Ok(Self {
            poses,
            mass_matrix,
            jacobians,
            centers,
            size,
        })
    }

    /// Reconstructs exact tree poses without assembling Jacobians or inertia.
    /// Geometry queries may use this path beyond the dense reference's size cap.
    /// Closed-loop equations are separate and are not projected here.
    ///
    /// # Errors
    /// Rejects invalid dimensions, non-finite positions, or non-unit roots.
    pub fn reconstruct_poses(
        creation: &CompiledCreation,
        roots: &[BodyPose],
        coordinates: &[f64],
    ) -> Result<Vec<BodyPose>, PhysicsError> {
        Self::validate_configuration(creation, roots, coordinates)?;
        let dynamics = &creation.dynamics;
        let mut poses = roots.to_vec();
        for &body in &dynamics.preorder {
            let Some(row) = dynamics.body_bearings[body] else {
                continue;
            };
            let topology = creation.loop_topology.body_parents[body];
            let parent = topology.parent_body as usize;
            let bearing = creation.bearings[row];
            let coordinate = bearing
                .coordinate_index
                .ok_or(PhysicsError::InvalidDynamics)? as usize;
            let (anchor_parent, anchor_child, axis, sign) = if topology.bearing_direction == 0 {
                (
                    bearing.local_anchor_a,
                    bearing.local_anchor_b,
                    bearing.local_axis_a,
                    1.0,
                )
            } else {
                (
                    bearing.local_anchor_b,
                    bearing.local_anchor_a,
                    bearing.local_axis_b,
                    -1.0,
                )
            };
            let inverse_parent = creation.compounds[parent]
                .root_rotation
                .as_dquat()
                .inverse();
            let bind_rotation =
                (inverse_parent * creation.compounds[body].root_rotation.as_dquat()).normalize();
            let bind_position = inverse_parent
                * (creation.compounds[body].root_translation
                    - creation.compounds[parent].root_translation)
                    .as_dvec3();
            let (rotation, position) = if bearing.kind.is_translational() {
                (
                    bind_rotation,
                    bind_position + axis.as_dvec3() * (sign * coordinates[coordinate]),
                )
            } else {
                let rotation = DQuat::from_axis_angle(
                    axis.as_dvec3().normalize(),
                    sign * coordinates[coordinate],
                ) * bind_rotation;
                (
                    rotation,
                    anchor_parent.as_dvec3() - rotation * anchor_child.as_dvec3(),
                )
            };
            poses[body] = BodyPose {
                position: poses[parent].position + poses[parent].rotation * position,
                rotation: (poses[parent].rotation * rotation).normalize(),
            };
        }
        if poses
            .iter()
            .any(|pose| !pose.position.is_finite() || !pose.rotation.is_finite())
        {
            return Err(PhysicsError::InvalidDynamics);
        }
        Ok(poses)
    }

    pub(crate) fn validate_configuration(
        creation: &CompiledCreation,
        roots: &[BodyPose],
        coordinates: &[f64],
    ) -> Result<(), PhysicsError> {
        let dynamics = &creation.dynamics;
        if roots.len() != creation.compounds.len()
            || coordinates.len() != dynamics.coordinate_bearings.len()
            || coordinates.iter().any(|x| !x.is_finite())
            || roots.iter().any(|p| {
                !p.position.is_finite()
                    || !p.rotation.is_finite()
                    || (p.rotation.length_squared() - 1.0).abs() > 1e-10
            })
        {
            return Err(PhysicsError::InvalidDynamics);
        }
        Ok(())
    }

    /// Creates body-indexed initial roots for `assemble`.
    pub fn initial_roots(creation: &CompiledCreation) -> Vec<BodyPose> {
        creation
            .compounds
            .iter()
            .map(|body| BodyPose {
                position: body.root_translation.as_dvec3(),
                rotation: body.root_rotation.as_dquat().normalize(),
            })
            .collect()
    }

    /// Builds a directional point-contact Jacobian accounting for all ancestors.
    ///
    /// # Errors
    /// Rejects unknown bodies or non-finite points/directions.
    pub fn point_row(
        &self,
        body: usize,
        point: DVec3,
        direction: DVec3,
    ) -> Result<Vec<f64>, PhysicsError> {
        let Some(jacobian) = self.jacobians.get(body) else {
            return Err(PhysicsError::InvalidConstraints);
        };
        if !point.is_finite() || !direction.is_finite() {
            return Err(PhysicsError::InvalidConstraints);
        }
        let arm = point - self.centers[body];
        Ok(jacobian
            .iter()
            .map(|motion| direction.dot(motion.linear + motion.angular.cross(arm)))
            .collect())
    }

    /// Generalized row for a world-axis angular impulse, including every ancestor.
    ///
    /// # Errors
    /// Rejects an unknown body or non-finite direction.
    pub fn angular_row(&self, body: usize, direction: DVec3) -> Result<Vec<f64>, PhysicsError> {
        let row = self
            .jacobians
            .get(body)
            .ok_or(PhysicsError::InvalidConstraints)?;
        if !direction.is_finite() {
            return Err(PhysicsError::InvalidConstraints);
        }
        Ok(row
            .iter()
            .map(|motion| direction.dot(motion.angular))
            .collect())
    }

    /// Factors effective dynamics with a nonnegative diagonal implicit spring /
    /// damping contribution in generalized units (`dt*c + dt²*k`).
    ///
    /// # Errors
    /// Rejects invalid diagonal rows and non-positive effective inertia.
    pub fn factor(&self, implicit_diagonal: &[f64]) -> Result<DynamicsFactor, PhysicsError> {
        if implicit_diagonal.len() != self.size
            || implicit_diagonal.iter().any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(PhysicsError::InvalidDynamics);
        }
        let mut effective = self.mass_matrix.clone();
        for (row, &value) in implicit_diagonal.iter().enumerate() {
            effective[row * self.size + row] += value;
        }
        DynamicsFactor::new(&effective, self.size)
    }

    /// Generalized force induced by uniform gravity.
    ///
    /// # Errors
    /// Rejects invalid gravity or a creation with different body rows.
    pub fn gravity_force(
        &self,
        creation: &CompiledCreation,
        gravity: DVec3,
    ) -> Result<Vec<f64>, PhysicsError> {
        if !gravity.is_finite() || creation.compounds.len() != self.poses.len() {
            return Err(PhysicsError::InvalidDynamics);
        }
        let mut force = vec![0.0; self.size];
        for (body, compound) in creation.compounds.iter().enumerate() {
            if compound.is_static {
                continue;
            }
            for (out, motion) in force.iter_mut().zip(&self.jacobians[body]) {
                *out += f64::from(compound.mass_properties.mass) * motion.linear.dot(gravity);
            }
        }
        Ok(force)
    }

    /// Reconstructs world-space velocities at each body's centre of mass.
    /// Root velocity rows describe the body origin, followed by world angular
    /// velocity; joint rows are signed coordinate rates.
    ///
    /// # Errors
    /// Rejects incorrect row counts and non-finite velocities or results.
    pub fn body_motions(&self, velocities: &[f64]) -> Result<Vec<SpatialMotion>, PhysicsError> {
        if velocities.len() != self.size || velocities.iter().any(|v| !v.is_finite()) {
            return Err(PhysicsError::InvalidDynamics);
        }
        let motions = self
            .jacobians
            .iter()
            .map(|row| {
                row.iter()
                    .zip(velocities)
                    .fold(SpatialMotion::default(), |sum, (j, v)| SpatialMotion {
                        linear: sum.linear + j.linear * *v,
                        angular: sum.angular + j.angular * *v,
                    })
            })
            .collect::<Vec<_>>();
        if motions
            .iter()
            .any(|m| !m.linear.is_finite() || !m.angular.is_finite())
        {
            return Err(PhysicsError::InvalidDynamics);
        }
        Ok(motions)
    }

    /// Body COM velocities in a reconstructed pose, using one compiled tree pass.
    /// Callers must supply the exact tree pose for these generalized rates.
    pub(crate) fn reconstruct_motions(
        creation: &CompiledCreation,
        poses: &[BodyPose],
        rates: &[f64],
    ) -> Result<Vec<SpatialMotion>, PhysicsError> {
        let dynamics = &creation.dynamics;
        if poses.len() != creation.compounds.len()
            || rates.len() != dynamics.elimination_parent.len()
            || rates.iter().any(|v| !v.is_finite())
        {
            return Err(PhysicsError::InvalidDynamics);
        }
        let centers = poses
            .iter()
            .zip(&dynamics.inertias)
            .map(|(pose, inertia)| pose.position + pose.rotation * inertia.center.as_dvec3())
            .collect::<Vec<_>>();
        let mut motions = vec![SpatialMotion::default(); poses.len()];
        for &body in &dynamics.preorder {
            let topology = creation.loop_topology.body_parents[body];
            let rows = dynamics.body_velocities[body].clone();
            if topology.is_root {
                if !rows.is_empty() {
                    let angular = DVec3::from_slice(&rates[rows.start + 3..rows.start + 6]);
                    motions[body] = SpatialMotion {
                        linear: DVec3::from_slice(&rates[rows.start..rows.start + 3])
                            + angular.cross(centers[body] - poses[body].position),
                        angular,
                    };
                }
                continue;
            }
            let parent = topology.parent_body as usize;
            let inherited = motions[parent];
            let bearing = creation.bearings
                [dynamics.body_bearings[body].ok_or(PhysicsError::InvalidDynamics)?];
            let (anchor, axis, sign) = if topology.bearing_direction == 0 {
                (bearing.local_anchor_a, bearing.local_axis_a, 1.0)
            } else {
                (bearing.local_anchor_b, bearing.local_axis_b, -1.0)
            };
            let relative =
                poses[parent].rotation * axis.as_dvec3().normalize() * (sign * rates[rows.start]);
            let mut motion = SpatialMotion {
                linear: inherited.linear + inherited.angular.cross(centers[body] - centers[parent]),
                angular: inherited.angular,
            };
            if bearing.kind.is_translational() {
                motion.linear += relative;
            } else {
                let anchor = poses[parent].position + poses[parent].rotation * anchor.as_dvec3();
                motion.linear += relative.cross(centers[body] - anchor);
                motion.angular += relative;
            }
            motions[body] = motion;
        }
        if motions
            .iter()
            .any(|motion| !motion.linear.is_finite() || !motion.angular.is_finite())
        {
            return Err(PhysicsError::InvalidDynamics);
        }
        Ok(motions)
    }

    /// Generalized inertial bias `C(q, v)` in `H v_dot = force - C`.
    /// The creation must be the one used to assemble this model. A tree pass
    /// computes centripetal/Coriolis acceleration at fixed generalized velocity;
    /// projection includes the world-space gyroscopic torque `ω × Iω`.
    /// No pose differencing or topology search is performed.
    ///
    /// # Errors
    /// Rejects mismatched body counts and non-finite velocities or results.
    pub fn inertial_bias(
        &self,
        creation: &CompiledCreation,
        velocities: &[f64],
    ) -> Result<Vec<f64>, PhysicsError> {
        if creation.compounds.len() != self.poses.len() {
            return Err(PhysicsError::InvalidDynamics);
        }
        let motions = self.body_motions(velocities)?;
        let bias = self.bias_accelerations(creation, &motions, velocities);
        let mut result = vec![0.0; self.size];
        for (body, inertia) in creation.dynamics.inertias.iter().enumerate() {
            if creation.compounds[body].is_static {
                continue;
            }
            let rotation = DMat3::from_quat(self.poses[body].rotation);
            let world_inertia = rotation * inertia.rotational.as_dmat3() * rotation.transpose();
            let omega = motions[body].angular;
            let force = f64::from(inertia.mass) * bias[body].linear;
            let torque = world_inertia * bias[body].angular + omega.cross(world_inertia * omega);
            for (out, j) in result.iter_mut().zip(&self.jacobians[body]) {
                *out += j.linear.dot(force) + j.angular.dot(torque);
            }
        }
        if result.iter().any(|v| !v.is_finite()) {
            return Err(PhysicsError::InvalidDynamics);
        }
        Ok(result)
    }
    // Kinematic acceleration with all generalized accelerations set to zero.
    // This is shared by force bias and J-dot-v for moving contact material points.
    fn bias_accelerations(
        &self,
        creation: &CompiledCreation,
        motions: &[SpatialMotion],
        velocities: &[f64],
    ) -> Vec<SpatialMotion> {
        let mut bias = vec![SpatialMotion::default(); self.poses.len()];
        for &body in &creation.dynamics.preorder {
            let topology = creation.loop_topology.body_parents[body];
            let omega = motions[body].angular;
            if topology.is_root {
                bias[body].linear =
                    omega.cross(omega.cross(self.centers[body] - self.poses[body].position));
            } else {
                let parent = topology.parent_body as usize;
                let column = creation.dynamics.body_velocities[body].start;
                let relative = self.jacobians[body][column];
                let speed = velocities[column];
                let linear = relative.linear * speed;
                let angular = relative.angular * speed;
                let omega_parent = motions[parent].angular;
                let arm = self.centers[body] - self.centers[parent];
                bias[body] = SpatialMotion {
                    linear: bias[parent].linear
                        + bias[parent].angular.cross(arm)
                        + omega_parent.cross(omega_parent.cross(arm))
                        + 2.0 * omega_parent.cross(linear)
                        + angular.cross(linear),
                    angular: bias[parent].angular + omega_parent.cross(angular),
                };
            }
        }
        bias
    }

    /// World material-point acceleration at fixed generalized velocity (J-dot-v).
    /// One ordered tree pass serves all points; no numerical pose differences.
    pub(crate) fn point_acceleration_bias(
        &self,
        creation: &CompiledCreation,
        velocities: &[f64],
        points: &[(usize, DVec3)],
    ) -> Result<Vec<DVec3>, PhysicsError> {
        if creation.compounds.len() != self.poses.len() {
            return Err(PhysicsError::InvalidDynamics);
        }
        let motions = self.body_motions(velocities)?;
        let bias = self.bias_accelerations(creation, &motions, velocities);
        points
            .iter()
            .map(|&(body, point)| {
                let center = self
                    .centers
                    .get(body)
                    .ok_or(PhysicsError::InvalidConstraints)?;
                let arm = point - *center;
                let omega = motions[body].angular;
                let acceleration = bias[body].linear
                    + bias[body].angular.cross(arm)
                    + omega.cross(omega.cross(arm));
                if !acceleration.is_finite() {
                    return Err(PhysicsError::InvalidDynamics);
                }
                Ok(acceleration)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mechanic_core::{BuildCommand, BuildPose, ConstructionGraph, CuboidSpec, GridRotation};

    #[test]
    fn free_body_gravity_is_independent_of_mass_and_does_not_generate_torque() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 2, 1],
                    BuildPose::new(bevy_math::IVec3::ZERO, GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
        let creation = graph.compile().unwrap();
        let model =
            MachineDynamics::assemble(&creation, &MachineDynamics::initial_roots(&creation), &[])
                .unwrap();
        let mut force = model
            .gravity_force(&creation, mechanic_core::GRAVITY)
            .unwrap();
        model.factor(&[0.0; 6]).unwrap().solve(&mut force).unwrap();
        assert!((force[1] + mechanic_core::STANDARD_GRAVITY_M_S2).abs() < 1e-12);
        for index in [0, 2, 3, 4, 5] {
            assert!(force[index].abs() < 1e-12);
        }
        let row = model.point_row(0, DVec3::X, DVec3::Y).unwrap();
        assert_eq!(row, [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }
    #[test]
    fn point_acceleration_bias_matches_reconstructed_constant_rate_motion() {
        use crate::{MachineState, free_motion::advance_positions};
        let doc: mechanic_world::WorldCreationInstanceDoc = ron::from_str(include_str!(
            "../../mechanic-bench/tests/fixtures/driven_car_instance.ron"
        ))
        .unwrap();
        let loaded = doc.creation.into_graph().unwrap();
        let creation = loaded
            .graph
            .compile_with_suspension_sockets([], &loaded.sockets)
            .unwrap();
        let mut state = MachineState::at_rest(&creation);
        for (index, velocity) in state.velocities.iter_mut().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "bounded fixture dimension")]
            {
                *velocity = 0.2 * (index as f64 * 0.7).cos();
            }
        }
        let model = MachineDynamics::assemble(&creation, &state.poses, &state.coordinates).unwrap();
        let local = DVec3::new(0.31, -0.17, 0.13);
        let points: Vec<_> = model
            .poses
            .iter()
            .enumerate()
            .map(|(body, pose)| (body, pose.position + pose.rotation * local))
            .collect();
        let bias = model
            .point_acceleration_bias(&creation, &state.velocities, &points)
            .unwrap();
        let epsilon = 1e-4;
        let at = |dt| {
            let mut sampled = state.clone();
            advance_positions(&creation, &mut sampled, dt);
            MachineDynamics::reconstruct_poses(&creation, &sampled.poses, &sampled.coordinates)
                .unwrap()
        };
        let plus = at(epsilon);
        let minus = at(-epsilon);
        for (body, acceleration) in bias.iter().enumerate() {
            let p = plus[body].position + plus[body].rotation * local;
            let m = minus[body].position + minus[body].rotation * local;
            let numerical = (p - 2.0 * points[body].1 + m) / (epsilon * epsilon);
            assert!(
                acceleration.distance(numerical) < 2e-6,
                "body={body} analytic={acceleration:?} numerical={numerical:?}"
            );
        }
        assert!(
            model
                .point_acceleration_bias(
                    &creation,
                    &state.velocities,
                    &[(creation.compounds.len(), local)]
                )
                .is_err()
        );
        assert!(
            model
                .point_acceleration_bias(&creation, &state.velocities, &[(0, DVec3::NAN)])
                .is_err()
        );
    }
}
