//! A fixed-pose algebra experiment, explicitly not a physics-tick benchmark.

use std::error::Error;

use bevy_math::DVec3;
use mechanic_core::{ColliderShape, CompiledCreation};
#[cfg(test)]
use mechanic_physics::solve_constraints;
use mechanic_physics::{ConstraintBlock, ImpulseBounds, MachineDynamics};

use mechanic_bench::finite_support;
mod measurement;

fn main() -> Result<(), Box<dyn Error>> {
    measurement::run()
}

// Each dynamic body receives one directional support probe at its lowest local
// collider vertex. These are declared synthetic contacts: no terrain, narrowphase,
// integration, drives, or publication is timed or claimed by this experiment.
fn support_probes(
    creation: &CompiledCreation,
    model: &MachineDynamics,
    gravity: &[f64],
) -> Result<Vec<ConstraintBlock>, Box<dyn Error>> {
    let mut blocks = Vec::new();
    for (body, compound) in creation.compounds.iter().enumerate() {
        if compound.is_static {
            continue;
        }
        let pose = model.poses[body];
        let mut lowest: Option<DVec3> = None;
        for collider in &creation.colliders
            [compound.collider_range.start as usize..compound.collider_range.end as usize]
        {
            let vertices = match &collider.shape {
                ColliderShape::Cuboid {
                    local_rotation,
                    half_extents,
                } => {
                    let mut vertices = Vec::with_capacity(8);
                    for x in [-1.0, 1.0] {
                        for y in [-1.0, 1.0] {
                            for z in [-1.0, 1.0] {
                                vertices.push(
                                    collider.local_center.as_dvec3()
                                        + local_rotation.as_dquat()
                                            * (half_extents.as_dvec3() * DVec3::new(x, y, z)),
                                );
                            }
                        }
                    }
                    vertices
                }
                ColliderShape::Convex(convex) => {
                    convex.vertices.iter().map(|v| v.as_dvec3()).collect()
                }
            };
            for vertex in vertices {
                let point = pose.position + pose.rotation * vertex;
                if lowest.is_none_or(|previous| point.y < previous.y) {
                    lowest = Some(point);
                }
            }
        }
        if let Some(point) = lowest {
            let row = model.point_row(body, point, DVec3::Y)?;
            let target = -row.iter().zip(gravity).map(|(j, v)| j * v).sum::<f64>();
            blocks.push(ConstraintBlock {
                jacobian: vec![row],
                target: vec![target],
                bounds: vec![ImpulseBounds {
                    minimum: 0.0,
                    maximum: f64::INFINITY,
                }],
                contacts: Vec::new(),
            });
        }
    }
    Ok(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn car() -> CompiledCreation {
        let instance: mechanic_world::WorldCreationInstanceDoc = ron::from_str(include_str!(
            "../../../tests/fixtures/driven_car_instance.ron"
        ))
        .unwrap();
        let loaded = instance.creation.into_graph().unwrap();
        loaded
            .graph
            .compile_with_suspension_sockets([], &loaded.sockets)
            .unwrap()
    }

    #[test]
    fn saved_car_supported_trajectory_certifies_all_candidates_and_repeats() {
        use mechanic_physics::{MachineMotion, MachineState, TerrainPathOutcome};
        let creation = car();
        let mut state = MachineState::at_rest(&creation);
        let (geometry, scene) = finite_support::scene(&creation, &mut state.poses).unwrap();
        let mut displacement = vec![0.0; state.velocities.len()];
        let root = creation.dynamics.preorder[0];
        displacement[creation.dynamics.body_velocities[root].start] = 0.0002;
        let motion = MachineMotion::new(&creation, 1, &state, &displacement).unwrap();
        let query = scene
            .validate_penetration(&geometry, &motion, DVec3::ZERO, 0.002, 128)
            .unwrap();
        assert_eq!(query.outcome, TerrainPathOutcome::Bounded);
        assert!(
            !scene
                .contacts(&geometry, &state.poses, DVec3::ZERO)
                .unwrap()
                .contacts
                .is_empty()
        );
        let repeat = scene
            .validate_penetration(&geometry, &motion, DVec3::ZERO, 0.002, 128)
            .unwrap();
        assert_eq!(repeat.outcome, query.outcome);
    }

    #[test]
    fn saved_car_joint_jacobians_match_reconstructed_point_motion() {
        let creation = car();
        let roots = MachineDynamics::initial_roots(&creation);
        let positions = vec![0.0; creation.dynamics.coordinate_bearings.len()];
        let model = MachineDynamics::assemble(&creation, &roots, &positions).unwrap();
        let epsilon = 1e-5;
        for (coordinate, &bearing_row) in creation.dynamics.coordinate_bearings.iter().enumerate() {
            let bearing = creation.bearings[bearing_row];
            let body = [bearing.compound_a as usize, bearing.compound_b as usize]
                .into_iter()
                .find(|&body| creation.dynamics.body_bearings[body] == Some(bearing_row))
                .unwrap();
            let velocity = creation.dynamics.body_velocities[body].start;
            let mut plus = positions.clone();
            let mut minus = positions.clone();
            plus[coordinate] += epsilon;
            minus[coordinate] -= epsilon;
            let plus = MachineDynamics::assemble(&creation, &roots, &plus).unwrap();
            let minus = MachineDynamics::assemble(&creation, &roots, &minus).unwrap();
            for target in 0..creation.compounds.len() {
                let local_point = DVec3::new(0.1, 0.2, -0.3);
                let point =
                    model.poses[target].position + model.poses[target].rotation * local_point;
                let point_plus =
                    plus.poses[target].position + plus.poses[target].rotation * local_point;
                let point_minus =
                    minus.poses[target].position + minus.poses[target].rotation * local_point;
                let numerical = (point_plus - point_minus) / (2.0 * epsilon);
                for direction in [DVec3::X, DVec3::Y, DVec3::Z] {
                    let row = model.point_row(target, point, direction).unwrap();
                    assert!(
                        (row[velocity] - numerical.dot(direction)).abs() < 1e-8,
                        "coordinate {coordinate}, target body {target}"
                    );
                }
            }
        }
    }

    #[test]
    fn saved_car_articulated_motion_bounds_cover_joint_and_suspension_point_speed() {
        use mechanic_physics::{MachineMotion, MachineState};
        let creation = car();
        let mut initial = MachineState::at_rest(&creation);
        for (row, value) in initial.coordinates.iter_mut().enumerate() {
            *value = if row % 2 == 0 { 0.02 } else { -0.03 };
        }
        let mut displacement = vec![0.0; initial.velocities.len()];
        for (row, value) in displacement.iter_mut().enumerate() {
            *value = if row % 2 == 0 { 0.5 } else { -0.7 };
        }
        let motion = MachineMotion::new(&creation, 1, &initial, &displacement).unwrap();
        let local_point = DVec3::new(0.7, -0.2, 0.3);
        let epsilon = 1e-6;
        let mut largest_ratio = 0.0_f64;
        for sample in 1..64 {
            let fraction = f64::from(sample) / 64.0;
            let before = motion.poses_at(fraction - epsilon).unwrap();
            let after = motion.poses_at(fraction + epsilon).unwrap();
            for body in 0..creation.compounds.len() {
                let a = before[body].position + before[body].rotation * local_point;
                let b = after[body].position + after[body].rotation * local_point;
                let actual = a.distance(b) / (2.0 * epsilon);
                let bound = motion.bounds()[body].point_speed(local_point.length());
                largest_ratio = largest_ratio.max(actual / bound);
                assert!(
                    actual <= bound * (1.0 + 1e-6),
                    "body {body} fraction {fraction}: {actual} > {bound}"
                );
            }
        }
        println!("saved_car_articulated_point_speed_maximum_bound_ratio={largest_ratio:e}");
    }

    #[test]
    fn saved_car_free_fall_moves_floating_roots_without_stretching_joints() {
        let creation = car();
        let model = MachineDynamics::assemble(
            &creation,
            &MachineDynamics::initial_roots(&creation),
            &vec![0.0; creation.dynamics.coordinate_bearings.len()],
        )
        .unwrap();
        let mut acceleration = model
            .gravity_force(&creation, mechanic_core::GRAVITY)
            .unwrap();
        model
            .factor(&vec![0.0; acceleration.len()])
            .unwrap()
            .solve(&mut acceleration)
            .unwrap();
        for (body, topology) in creation.loop_topology.body_parents.iter().enumerate() {
            let rows = creation.dynamics.body_velocities[body].clone();
            for (index, row) in rows.enumerate() {
                let expected = if topology.is_root && index == 1 {
                    -mechanic_core::STANDARD_GRAVITY_M_S2
                } else {
                    0.0
                };
                assert!(
                    (acceleration[row] - expected).abs() < 1e-8,
                    "body {body}, row {row}"
                );
            }
        }
    }

    #[test]
    fn saved_car_inertial_bias_matches_differentiated_body_motion() {
        use bevy_math::{DMat3, DQuat};
        let creation = car();
        let mut roots = MachineDynamics::initial_roots(&creation);
        for root in &mut roots {
            root.rotation =
                (DQuat::from_scaled_axis(DVec3::new(0.3, -0.2, 0.1)) * root.rotation).normalize();
        }
        let coordinates = vec![0.07; creation.dynamics.coordinate_bearings.len()];
        let velocities = (0..creation.dynamics.elimination_parent.len())
            .map(|i| f64::from(u32::try_from(i % 7).unwrap()) * 0.3 - 0.8)
            .collect::<Vec<_>>();
        let model = MachineDynamics::assemble(&creation, &roots, &coordinates).unwrap();
        let motions = model.body_motions(&velocities).unwrap();
        let sample = |time: f64| {
            let mut shifted_roots = roots.clone();
            let mut shifted_coordinates = coordinates.clone();
            for &body in &creation.dynamics.preorder {
                let rows = creation.dynamics.body_velocities[body].clone();
                if rows.is_empty() {
                    continue;
                }
                if creation.loop_topology.body_parents[body].is_root {
                    let v = &velocities[rows];
                    shifted_roots[body].position += DVec3::new(v[0], v[1], v[2]) * time;
                    shifted_roots[body].rotation =
                        (DQuat::from_scaled_axis(DVec3::new(v[3], v[4], v[5]) * time)
                            * roots[body].rotation)
                            .normalize();
                } else {
                    let bearing = creation.dynamics.body_bearings[body].unwrap();
                    let coordinate = creation.bearings[bearing].coordinate_index.unwrap() as usize;
                    shifted_coordinates[coordinate] += velocities[rows.start] * time;
                }
            }
            MachineDynamics::assemble(&creation, &shifted_roots, &shifted_coordinates)
                .unwrap()
                .body_motions(&velocities)
                .unwrap()
        };
        let epsilon = 1e-5;
        let plus = sample(epsilon);
        let minus = sample(-epsilon);
        let bias = model.inertial_bias(&creation, &velocities).unwrap();
        let mut maximum_relative_error = 0.0_f64;
        for (column, actual) in bias.iter().enumerate() {
            let mut unit = vec![0.0; velocities.len()];
            unit[column] = 1.0;
            let jacobian = model.body_motions(&unit).unwrap();
            let mut expected = 0.0;
            for (body, inertia) in creation.dynamics.inertias.iter().enumerate() {
                let acceleration = (plus[body].linear - minus[body].linear) / (2.0 * epsilon);
                let alpha = (plus[body].angular - minus[body].angular) / (2.0 * epsilon);
                let rotation = DMat3::from_quat(model.poses[body].rotation);
                let world_inertia = rotation * inertia.rotational.as_dmat3() * rotation.transpose();
                let omega = motions[body].angular;
                let torque = world_inertia * alpha + omega.cross(world_inertia * omega);
                expected += jacobian[body].linear.dot(acceleration) * f64::from(inertia.mass)
                    + jacobian[body].angular.dot(torque);
            }
            let error = (actual - expected).abs() / expected.abs().max(1.0);
            maximum_relative_error = maximum_relative_error.max(error);
            assert!(
                error < 1e-7,
                "column {column}: actual {actual}, differentiated {expected}"
            );
        }
        println!("saved_car_bias_maximum_relative_error={maximum_relative_error:e}");
    }

    #[test]
    fn saved_car_joint_ticks_keep_authored_suspension_and_drives_and_repeat() {
        use mechanic_physics::{CpuJointMachine, DriveCommand, JointTickConfig, MachineState};
        let creation = car();
        let driven = creation
            .coordinate_drives
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, drive)| drive.mode == mechanic_core::DriveMode::Speed)
            .collect::<Vec<_>>();
        assert!(
            !driven.is_empty(),
            "fixture must retain its authored speed drives"
        );
        let initial = MachineState::at_rest(&creation);
        let mut first = CpuJointMachine::new(creation.clone(), 3, initial.clone()).unwrap();
        let mut second = CpuJointMachine::new(creation, 3, initial).unwrap();
        let mut effort = 0.0_f64;
        for tick in 1..=120 {
            let settings = JointTickConfig::default();
            let commands = if matches!(tick, 1 | 61) {
                driven
                    .iter()
                    .map(|&(coordinate, mut drive)| {
                        drive.target_speed = if tick == 1 { 8.0 } else { -4.0 };
                        DriveCommand {
                            tick,
                            topology_generation: 3,
                            coordinate,
                            drive,
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            let a = first
                .step(mechanic_core::GRAVITY, settings, &[], &commands)
                .unwrap()
                .state_hash();
            let b = second
                .step(mechanic_core::GRAVITY, settings, &[], &commands)
                .unwrap()
                .state_hash();
            assert_eq!(a, b, "joint-only airborne tick {tick}");
            assert!(first.diagnostics().residual <= settings.tolerance);
            effort += first
                .diagnostics()
                .drive_impulses
                .iter()
                .map(|value| value.abs())
                .sum::<f64>();
        }
        assert!(
            effort > 0.0,
            "repeatability must exercise physical motor impulses"
        );
        println!(
            "saved_car_joint_only_airborne_ticks=120 hash={:016x} absolute_drive_impulse_sum={effort:e}",
            first.snapshot().state_hash(),
        );
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "complete saved-car/finite-terrain fixture and external momentum check"
    )]
    fn saved_car_finite_terrain_response_repeats_and_balances_external_momentum() {
        let creation = car();
        let coordinates = vec![0.0; creation.dynamics.coordinate_bearings.len()];
        let mut roots = MachineDynamics::initial_roots(&creation);
        let (geometry, terrain) = finite_support::scene(&creation, &mut roots).unwrap();
        let model = MachineDynamics::assemble(&creation, &roots, &coordinates).unwrap();
        let query = terrain
            .contacts(&geometry, &model.poses, DVec3::ZERO)
            .unwrap();
        assert!(!query.contacts.is_empty());
        assert_eq!(
            query.contacts,
            terrain
                .contacts(&geometry, &model.poses, DVec3::ZERO)
                .unwrap()
                .contacts
        );
        let size = creation.dynamics.elimination_parent.len();
        let factor = model.factor(&vec![0.0; size]).unwrap();
        let mut falling = model
            .gravity_force(
                &creation,
                mechanic_core::GRAVITY / f64::from(mechanic_core::TICK_RATE_HZ),
            )
            .unwrap();
        factor.solve(&mut falling).unwrap();
        let blocks = query
            .contacts
            .iter()
            .map(|contact| {
                let row = model
                    .point_row(contact.body, contact.body_point, contact.normal)
                    .unwrap();
                let target = -row.iter().zip(&falling).map(|(j, v)| j * v).sum::<f64>();
                ConstraintBlock {
                    jacobian: vec![row],
                    target: vec![target],
                    bounds: vec![ImpulseBounds {
                        minimum: 0.0,
                        maximum: f64::INFINITY,
                    }],
                    contacts: Vec::new(),
                }
            })
            .collect::<Vec<_>>();
        let first = solve_constraints(&factor, &blocks, 1024, 1e-9).unwrap();
        let second = solve_constraints(&factor, &blocks, 1024, 1e-9).unwrap();
        assert!(
            first.converged,
            "real finite car response residual {}",
            first.residual
        );
        let mut incoming = falling.clone();
        incoming[0] = 0.3;
        let contacts = query
            .impact_constraints(&model, &incoming, 1.0, 1e-7)
            .unwrap();
        let rich = solve_constraints(&factor, &contacts.blocks, 256, 1e-8).unwrap();
        println!(
            "saved_car_full_surface_response rows={} blocks={} residual={:e} iterations={} factor_solves={} response_storage={} converged={} newton_attempts={} newton_accepts={} newton_applications={} newton_local_factorizations={} newton_line_searches={} newton_generalized_factorizations={} newton_reduced_storage={} tick_gate=false",
            rich.impulses.len(),
            contacts.blocks.len(),
            rich.residual,
            rich.iterations,
            rich.factor_solves,
            rich.response_storage,
            rich.converged,
            rich.newton_attempts,
            rich.newton_accepts,
            rich.newton_applications,
            rich.newton_local_factorizations,
            rich.newton_line_searches,
            rich.newton_generalized_factorizations,
            rich.newton_reduced_storage
        );
        assert!(
            rich.converged,
            "full friction/rolling manifold response must converge at the default iteration bound"
        );
        let repeated = solve_constraints(&factor, &contacts.blocks, 256, 1e-8).unwrap();
        assert_eq!(rich.impulses, repeated.impulses);
        assert_eq!(rich.velocity_change, repeated.velocity_change);
        assert_eq!(rich.sliding, repeated.sliding);
        validate_surface_impulses(&creation, &model, &query, &contacts, &incoming, &rich);
        // Five rows for each of the four wheels' twenty support points. A solid
        // cylinder contacts the floor as one prism, so this manifold no longer
        // carries sixteen rounded copies of every wheel's contact edge.
        assert_eq!(rich.impulses.len(), 80);
        assert_eq!(first.impulses, second.impulses);
        assert_eq!(first.velocity_change, second.velocity_change);
        for block in &blocks {
            let normal_velocity = block.jacobian[0]
                .iter()
                .zip(falling.iter().zip(&first.velocity_change))
                .map(|(j, (v, dv))| j * (v + dv))
                .sum::<f64>();
            assert!(normal_velocity >= -1e-8);
        }
        let momentum = model
            .body_motions(&first.velocity_change)
            .unwrap()
            .iter()
            .zip(&creation.compounds)
            .map(|(motion, body)| motion.linear * f64::from(body.mass_properties.mass))
            .sum::<DVec3>();
        let impulse = query
            .contacts
            .iter()
            .zip(&first.impulses)
            .map(|(contact, &magnitude)| contact.normal * magnitude)
            .sum::<DVec3>();
        assert!(momentum.distance(impulse) < 1e-8 * impulse.length().max(1.0));
        println!(
            "saved_car_finite_response bodies={} colliders={} retained_contacts={} triangle_candidates={} residual={:e} repeatable=true tick_gate=false",
            creation.compounds.len(),
            creation.colliders.len(),
            query.contacts.len(),
            query.triangle_candidates,
            first.residual
        );
    }
    fn validate_surface_impulses(
        creation: &CompiledCreation,
        model: &MachineDynamics,
        query: &mechanic_physics::TerrainContactQuery,
        constraints: &mechanic_physics::TerrainImpactConstraints,
        incoming: &[f64],
        solution: &mechanic_physics::ConstraintSolution,
    ) {
        use bevy_math::DMat3;
        let mut external_linear = DVec3::ZERO;
        let mut external_angular = DVec3::ZERO;
        let mut first = 0;
        let mut point = 0;
        for block in &constraints.blocks {
            for (local, law) in block.contacts.iter().enumerate() {
                let impulse = &solution.impulses[first + local * 5..first + local * 5 + 5];
                let contact = &query.contacts[constraints.point_indices[point]];
                let normal = contact.normal;
                let u = DVec3::X.cross(normal).normalize();
                let v = normal.cross(u);
                let linear = normal * impulse[0] + u * impulse[1] + v * impulse[2];
                let rolling = u * impulse[3] + v * impulse[4];
                external_linear += linear;
                external_angular += contact.body_point.cross(linear) + rolling;
                let speeds = block.jacobian[local * 5..local * 5 + 5]
                    .iter()
                    .map(|row| {
                        row.iter()
                            .zip(incoming.iter().zip(&solution.velocity_change))
                            .map(|(j, (old, change))| j * (old + change))
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                assert!(impulse[0] >= 0.0);
                assert!(speeds[0] >= -1e-8);
                if impulse[0] > 1e-6 {
                    assert!(speeds[0].abs() < 1e-8);
                }
                let coefficient = if solution.sliding[point] {
                    law.kinetic_coefficient
                } else {
                    law.static_coefficient
                };
                assert!(impulse[1].hypot(impulse[2]) <= coefficient * impulse[0] + 1e-12);
                assert!(
                    impulse[3].hypot(impulse[4])
                        <= law.rolling_length.unwrap() * impulse[0] + 1e-12
                );
                assert!(impulse[1] * speeds[1] + impulse[2] * speeds[2] <= 1e-7);
                assert!(impulse[3] * speeds[3] + impulse[4] * speeds[4] <= 1e-7);
                point += 1;
            }
            first += block.jacobian.len();
        }
        let motions = model.body_motions(&solution.velocity_change).unwrap();
        let mut linear = DVec3::ZERO;
        let mut angular = DVec3::ZERO;
        for (body, inertia) in creation.dynamics.inertias.iter().enumerate() {
            let pose = model.poses[body];
            let rotation = DMat3::from_quat(pose.rotation);
            let center = pose.position + pose.rotation * inertia.center.as_dvec3();
            let momentum = motions[body].linear * f64::from(inertia.mass);
            linear += momentum;
            angular += center.cross(momentum)
                + rotation
                    * inertia.rotational.as_dmat3()
                    * rotation.transpose()
                    * motions[body].angular;
        }
        assert!(linear.distance(external_linear) < 1e-8 * external_linear.length().max(1.0));
        assert!(angular.distance(external_angular) < 1e-8 * external_angular.length().max(1.0));
    }
}
