//! Runtime tree kinematics without a generalized mass matrix.
use crate::{
    BodyPose, DynamicsFactor, DynamicsFactorization, MachineDynamics, PhysicsError, SpatialMotion,
};
use bevy_math::{DMat3, DVec3};
use mechanic_core::CompiledCreation;

/// Pose-local tree kinematics with ancestor-path Jacobians.
pub struct MachineKinematics<'a> {
    /// Exact reconstructed body poses.
    pub poses: Vec<BodyPose>,
    creation: &'a CompiledCreation,
    centers: Vec<DVec3>,
    relative: Vec<SpatialMotion>,
    jacobians: Vec<Vec<(usize, SpatialMotion)>>,
    size: usize,
    component_rows: Vec<std::ops::Range<usize>>,
}
impl<'a> MachineKinematics<'a> {
    /// Reconstructs body poses without preparing any Jacobian rows.
    ///
    /// # Errors
    /// Rejects malformed or non-finite configurations.
    pub fn reconstruct_poses(
        creation: &CompiledCreation,
        roots: &[BodyPose],
        coordinates: &[f64],
    ) -> Result<Vec<BodyPose>, PhysicsError> {
        MachineDynamics::reconstruct_poses(creation, roots, coordinates)
    }

    /// Published COM velocities from already reconstructed body poses.
    ///
    /// # Errors
    /// Rejects invalid dimensions or velocities.
    pub fn published_motions(
        creation: &CompiledCreation,
        poses: &[BodyPose],
        velocities: &[f64],
    ) -> Result<Vec<SpatialMotion>, PhysicsError> {
        MachineDynamics::reconstruct_motions(creation, poses, velocities)
    }

    /// Reconstructs a tree without the dense reference capacity limit.
    ///
    /// # Errors
    /// Rejects malformed or non-finite configurations.
    pub fn assemble(
        creation: &'a CompiledCreation,
        roots: &[BodyPose],
        coordinates: &[f64],
    ) -> Result<Self, PhysicsError> {
        let poses = MachineDynamics::reconstruct_poses(creation, roots, coordinates)?;
        let centers = poses
            .iter()
            .zip(&creation.dynamics.inertias)
            .map(|(p, i)| p.position + p.rotation * i.center.as_dvec3())
            .collect::<Vec<_>>();
        let mut relative = vec![SpatialMotion::default(); poses.len()];
        for &body in &creation.dynamics.preorder {
            let topology = creation.loop_topology.body_parents[body];
            if topology.is_root {
                continue;
            }
            let parent = topology.parent_body as usize;
            let bearing = creation.bearings
                [creation.dynamics.body_bearings[body].ok_or(PhysicsError::InvalidDynamics)?];
            let (anchor, axis, sign) = if topology.bearing_direction == 0 {
                (bearing.local_anchor_a, bearing.local_axis_a, 1.0)
            } else {
                (bearing.local_anchor_b, bearing.local_axis_b, -1.0)
            };
            let axis = poses[parent].rotation * axis.as_dvec3().normalize() * sign;
            relative[body] = if bearing.kind.is_translational() {
                SpatialMotion {
                    linear: axis,
                    angular: DVec3::ZERO,
                }
            } else {
                let anchor = poses[parent].position + poses[parent].rotation * anchor.as_dvec3();
                SpatialMotion {
                    linear: axis.cross(centers[body] - anchor),
                    angular: axis,
                }
            };
        }
        let mut jacobians = vec![Vec::<(usize, SpatialMotion)>::new(); poses.len()];
        for &body in &creation.dynamics.preorder {
            let topology = creation.loop_topology.body_parents[body];
            let rows = creation.dynamics.body_velocities[body].clone();
            if topology.is_root {
                if !rows.is_empty() {
                    for (index, axis) in [DVec3::X, DVec3::Y, DVec3::Z].into_iter().enumerate() {
                        jacobians[body].push((
                            rows.start + index,
                            SpatialMotion {
                                linear: axis,
                                angular: DVec3::ZERO,
                            },
                        ));
                    }
                    for (index, axis) in [DVec3::X, DVec3::Y, DVec3::Z].into_iter().enumerate() {
                        jacobians[body].push((
                            rows.start + 3 + index,
                            SpatialMotion {
                                linear: axis.cross(centers[body] - poses[body].position),
                                angular: axis,
                            },
                        ));
                    }
                }
            } else {
                let parent = topology.parent_body as usize;
                let arm = centers[body] - centers[parent];
                jacobians[body] = jacobians[parent]
                    .iter()
                    .map(|&(row, motion)| {
                        (
                            row,
                            SpatialMotion {
                                linear: motion.linear + motion.angular.cross(arm),
                                angular: motion.angular,
                            },
                        )
                    })
                    .collect();
                jacobians[body].push((rows.start, relative[body]));
            }
        }
        let mut component_rows = vec![0..0; poses.len()];
        for component in &creation.dynamics.components {
            for &body in &creation.dynamics.preorder[component.bodies.clone()] {
                component_rows[body] = component.velocities.clone();
            }
        }
        Ok(Self {
            component_rows,
            poses,
            creation,
            centers,
            relative,
            jacobians,
            size: creation.dynamics.elimination_parent.len(),
        })
    }

    /// World COM velocities projected from ancestor-path Jacobians.
    ///
    /// # Errors
    /// Rejects invalid generalized velocities.
    pub fn body_motions(&self, velocities: &[f64]) -> Result<Vec<SpatialMotion>, PhysicsError> {
        if velocities.len() != self.size || velocities.iter().any(|v| !v.is_finite()) {
            return Err(PhysicsError::InvalidDynamics);
        }
        let motions = self
            .jacobians
            .iter()
            .map(|row| {
                row.iter()
                    .fold(SpatialMotion::default(), |sum, &(column, j)| {
                        SpatialMotion {
                            linear: sum.linear + j.linear * velocities[column],
                            angular: sum.angular + j.angular * velocities[column],
                        }
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

    /// Directional point row using only the body's ancestor path.
    ///
    /// # Errors
    /// Rejects unknown bodies or non-finite geometry.
    pub fn point_row(
        &self,
        body: usize,
        point: DVec3,
        direction: DVec3,
    ) -> Result<Vec<f64>, PhysicsError> {
        self.row(body, point, direction, DVec3::ZERO)
    }

    /// Angular impulse row using only the body's ancestor path.
    ///
    /// # Errors
    /// Rejects unknown bodies or non-finite directions.
    pub fn angular_row(&self, body: usize, direction: DVec3) -> Result<Vec<f64>, PhysicsError> {
        self.row(body, DVec3::ZERO, DVec3::ZERO, direction)
    }

    fn row(
        &self,
        body: usize,
        point: DVec3,
        force: DVec3,
        torque: DVec3,
    ) -> Result<Vec<f64>, PhysicsError> {
        if body >= self.poses.len()
            || !point.is_finite()
            || !force.is_finite()
            || !torque.is_finite()
        {
            return Err(PhysicsError::InvalidConstraints);
        }
        let mut result = vec![0.0; self.size];
        let arm = point - self.centers[body];
        for &(row, motion) in &self.jacobians[body] {
            result[row] =
                force.dot(motion.linear + motion.angular.cross(arm)) + torque.dot(motion.angular);
        }
        Ok(result)
    }

    fn project(&self, wrenches: Vec<SpatialMotion>) -> Vec<f64> {
        let mut result = vec![0.0; self.size];
        for (body, wrench) in wrenches.into_iter().enumerate() {
            for &(row, j) in &self.jacobians[body] {
                result[row] += j.linear.dot(wrench.linear) + j.angular.dot(wrench.angular);
            }
        }
        result
    }

    pub(crate) fn contact_ranges(
        &self,
        contact: &crate::TerrainContact,
    ) -> [std::ops::Range<usize>; 2] {
        let mut ranges = [
            self.component_rows[contact.body].clone(),
            contact
                .other_body
                .map_or(0..0, |body| self.component_rows[body].clone()),
        ];
        if ranges[0] == ranges[1] {
            ranges[1] = 0..0;
        }
        ranges.sort_by_key(|range| range.start);
        ranges
    }

    pub(crate) fn contact_row(
        &self,
        contact: &crate::TerrainContact,
        direction: DVec3,
        angular: bool,
        output: &mut [f64],
    ) -> Result<(), PhysicsError> {
        if output.len() != self.size || !direction.is_finite() {
            return Err(PhysicsError::InvalidConstraints);
        }
        for range in self.contact_ranges(contact) {
            output[range].fill(0.0);
        }
        for (body, point, subtract) in std::iter::once((contact.body, contact.body_point, false))
            .chain(
                contact
                    .other_body
                    .map(|body| (body, contact.terrain_point, true)),
            )
        {
            if body >= self.poses.len() || !point.is_finite() {
                return Err(PhysicsError::InvalidConstraints);
            }
            let arm = if angular {
                -self.centers[body]
            } else {
                point - self.centers[body]
            };
            for &(row, motion) in &self.jacobians[body] {
                let value = if angular {
                    DVec3::ZERO.dot(motion.linear + motion.angular.cross(arm))
                        + direction.dot(motion.angular)
                } else {
                    direction.dot(motion.linear + motion.angular.cross(arm))
                        + DVec3::ZERO.dot(motion.angular)
                };
                if subtract {
                    output[row] -= value;
                } else {
                    output[row] = value;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn gravity_force(
        &self,
        creation: &CompiledCreation,
        gravity: DVec3,
    ) -> Result<Vec<f64>, PhysicsError> {
        if !gravity.is_finite() {
            return Err(PhysicsError::InvalidDynamics);
        }
        let mut result = vec![0.0; self.size];
        for (body, compound) in creation.compounds.iter().enumerate() {
            if compound.is_static {
                continue;
            }
            for &(row, motion) in &self.jacobians[body] {
                result[row] +=
                    f64::from(compound.mass_properties.mass) * motion.linear.dot(gravity);
            }
        }
        Ok(result)
    }

    pub(crate) fn inertial_bias(
        &self,
        creation: &CompiledCreation,
        velocities: &[f64],
    ) -> Result<Vec<f64>, PhysicsError> {
        let motions = self.body_motions(velocities)?;
        let bias = self.bias_accelerations(creation, &motions, velocities);
        let wrenches = creation
            .dynamics
            .inertias
            .iter()
            .enumerate()
            .map(|(body, inertia)| {
                if creation.compounds[body].is_static {
                    return SpatialMotion::default();
                }
                let rotation = DMat3::from_quat(self.poses[body].rotation);
                let world = rotation * inertia.rotational.as_dmat3() * rotation.transpose();
                let omega = motions[body].angular;
                SpatialMotion {
                    linear: f64::from(inertia.mass) * bias[body].linear,
                    angular: world * bias[body].angular + omega.cross(world * omega),
                }
            })
            .collect();
        let result = self.project(wrenches);
        if result.iter().any(|v| !v.is_finite()) {
            return Err(PhysicsError::InvalidDynamics);
        }
        Ok(result)
    }

    pub(crate) fn refactor(
        &self,
        selection: DynamicsFactorization,
        coordinates: &[f64],
        diagonal: &[f64],
        factor: &mut Option<DynamicsFactor>,
    ) -> Result<(), PhysicsError> {
        if selection == DynamicsFactorization::Articulated
            && let Some(factor) = factor
        {
            factor.refit_articulated(self.creation, &self.poses, diagonal)?;
        } else {
            *factor = Some(self.factor(selection, coordinates, diagonal)?);
        }
        Ok(())
    }

    pub(crate) fn factor(
        &self,
        selection: DynamicsFactorization,
        coordinates: &[f64],
        diagonal: &[f64],
    ) -> Result<DynamicsFactor, PhysicsError> {
        match selection {
            DynamicsFactorization::DenseReference => {
                MachineDynamics::assemble(self.creation, &self.poses, coordinates)?.factor(diagonal)
            }
            DynamicsFactorization::Articulated => {
                DynamicsFactor::articulated_from_poses(self.creation, &self.poses, diagonal)
            }
        }
    }
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
                let relative = self.relative[body];
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuMachine, ExternalImpulse, MachineState, SoftStepConfig};
    use mechanic_core::{BuildCommand, BuildPose, ConstructionGraph, CuboidSpec};

    #[test]
    fn runtime_rows_forces_and_impulses_match_the_dense_suspension_reference() {
        let doc: mechanic_world::WorldCreationInstanceDoc = ron::from_str(include_str!(
            "../../mechanic-bench/tests/fixtures/driven_car_instance.ron"
        ))
        .unwrap();
        let loaded = doc.creation.into_graph().unwrap();
        let creation = loaded
            .graph
            .compile_with_suspension_sockets([], &loaded.sockets)
            .unwrap();
        for phase in [0.0_f64, 0.1, -0.2] {
            let mut state = MachineState::at_rest(&creation);
            state.coordinates.fill(phase);
            for (row, velocity) in state.velocities.iter_mut().enumerate() {
                *velocity = f64::from(u32::try_from(row).unwrap()).sin();
            }
            let dense =
                MachineDynamics::assemble(&creation, &state.poses, &state.coordinates).unwrap();
            let runtime =
                MachineKinematics::assemble(&creation, &state.poses, &state.coordinates).unwrap();
            let close = |a: &[f64], b: &[f64]| {
                for (a, b) in a.iter().zip(b) {
                    assert!((a - b).abs() < 1e-9, "{a} != {b}");
                }
            };
            close(
                &dense
                    .gravity_force(&creation, mechanic_core::GRAVITY)
                    .unwrap(),
                &runtime
                    .gravity_force(&creation, mechanic_core::GRAVITY)
                    .unwrap(),
            );
            close(
                &dense.inertial_bias(&creation, &state.velocities).unwrap(),
                &runtime.inertial_bias(&creation, &state.velocities).unwrap(),
            );
            for body in 0..creation.compounds.len() {
                let point = dense.poses[body].position + DVec3::new(0.3, 0.1, -0.2);
                let mut a = dense.point_row(body, point, DVec3::Y).unwrap();
                let mut b = runtime.point_row(body, point, DVec3::Y).unwrap();
                close(&a, &b);
                close(
                    &dense.angular_row(body, DVec3::X).unwrap(),
                    &runtime.angular_row(body, DVec3::X).unwrap(),
                );
                dense
                    .factor(&vec![0.0; a.len()])
                    .unwrap()
                    .solve(&mut a)
                    .unwrap();
                runtime
                    .factor(
                        DynamicsFactorization::Articulated,
                        &state.coordinates,
                        &vec![0.0; b.len()],
                    )
                    .unwrap()
                    .solve(&mut b)
                    .unwrap();
                close(&a, &b);
            }
        }
    }

    #[test]
    fn a_long_rigid_chassis_keeps_its_suspensions_active_in_free_fall() {
        use mechanic_core::{CreationDocument, RigidLinkDoc};
        let instance: mechanic_world::WorldCreationInstanceDoc = ron::from_str(include_str!(
            "../../mechanic-bench/tests/fixtures/builder-world/generations/20/world.ron"
        ))
        .unwrap();
        let mut document =
            CreationDocument::from_graph(&ConstructionGraph::new(), "Long chassis", &[]);
        let mut next_link = 1;
        for copy in 0..10 {
            let mut parts = instance.creation.clone();
            parts.remap_dimension_links(&mut next_link);
            parts.transform_cardinal(0, bevy_math::IVec3::X * copy * 64);
            document.append(parts).unwrap();
            if copy > 0 {
                document.rigid_links.push(RigidLinkDoc {
                    first: 0,
                    second: u32::try_from(
                        usize::try_from(copy).unwrap() * instance.creation.parts.len(),
                    )
                    .unwrap(),
                });
            }
        }
        let loaded = document.into_graph().unwrap();
        let creation = loaded
            .graph
            .compile_with_suspension_sockets([], &loaded.sockets)
            .unwrap();
        let centre = |poses: &[BodyPose]| {
            poses
                .iter()
                .zip(&creation.dynamics.inertias)
                .map(|(pose, inertia)| {
                    (pose.position + pose.rotation * inertia.center.as_dvec3())
                        * f64::from(inertia.mass)
                })
                .sum::<DVec3>()
                / creation
                    .dynamics
                    .inertias
                    .iter()
                    .map(|inertia| f64::from(inertia.mass))
                    .sum::<f64>()
        };
        let state = MachineState::at_rest(&creation);
        let initial = centre(&state.poses);
        let mut machine = CpuMachine::new(creation.clone(), 1, state).unwrap();
        for _ in 0..8 {
            machine
                .step(
                    mechanic_core::GRAVITY,
                    &SoftStepConfig::default(),
                    &[],
                    &[],
                    None,
                )
                .unwrap();
            assert!(
                !machine.diagnostics().degraded,
                "{:?}",
                machine.diagnostics()
            );
        }
        assert!(centre(&machine.snapshot().state.poses).y < initial.y - 0.05);
    }

    #[test]
    fn active_runtime_above_512_velocities_applies_off_centre_impulses_and_releases_holds() {
        let mut graph = ConstructionGraph::new();
        for index in 0..96 {
            graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [1, 1, 1],
                        BuildPose::from_position_ticks(
                            bevy_math::IVec3::X * index * 200,
                            mechanic_core::GridRotation::default(),
                        ),
                    )
                    .unwrap(),
                ))
                .unwrap();
        }
        let creation = graph.compile().unwrap();
        let mut machine =
            CpuMachine::new(creation.clone(), 7, MachineState::at_rest(&creation)).unwrap();
        let body = 64;
        let initial = machine.snapshot().state.poses[body];
        let impulse = ExternalImpulse {
            tick: 1,
            topology_generation: 7,
            body,
            point: initial.position + DVec3::X * 0.1,
            impulse: DVec3::Y,
        };
        machine
            .step(
                DVec3::ZERO,
                &SoftStepConfig::default(),
                &[impulse],
                &[],
                None,
            )
            .unwrap();
        let state = &machine.snapshot().state;
        assert!(state.velocities.len() > 512);
        assert!(state.poses[body].position.y > initial.position.y);
        assert!(state.poses[body].rotation.angle_between(initial.rotation) > 0.0);
        assert_eq!(state.poses[0], MachineState::at_rest(&creation).poses[0]);
        let poses = state.poses.clone();
        machine.hold(&[true; 96], &poses).unwrap();
        machine
            .step(DVec3::NEG_Y, &SoftStepConfig::default(), &[], &[], None)
            .unwrap();
        assert_eq!(machine.snapshot().state.poses, poses);
        machine.hold(&[false; 96], &poses).unwrap();
        machine
            .step(DVec3::NEG_Y, &SoftStepConfig::default(), &[], &[], None)
            .unwrap();
        assert!(machine.snapshot().state.poses[body].position.y < poses[body].position.y);
        assert!(!machine.diagnostics().degraded);
    }
}
