use super::*;
use crate::DynamicsFactor;
use crate::{ConstraintBlock, ContactFriction, ImpulseBounds, solve_constraints};
use bevy_math::{DQuat, IVec3, Vec3};
use mechanic_core::{
    BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, FaceKind,
    FaceRef, GridRotation,
};

fn chain(joints: i32, anchored: bool, reversed: bool) -> CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let mut parts = Vec::new();
    for index in 0..=joints {
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 2, 1],
                    BuildPose::new(IVec3::X * (index * 4), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            panic!("spawn");
        };
        parts.push(part);
    }
    for (index, pair) in parts.windows(2).enumerate() {
        let a = FaceRef::part(pair[0], FaceKind::PositiveX);
        let b = FaceRef::part(pair[1], FaceKind::NegativeX);
        let (a, b, axis) = if reversed {
            (b, a, -Vec3::X)
        } else {
            (a, b, Vec3::X)
        };
        #[expect(clippy::cast_precision_loss, reason = "bounded fixture body count")]
        let anchor = Vec3::X * (index as f32 + 0.5);
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                a, b, anchor, axis,
            )))
            .unwrap();
    }
    graph
        .compile_with_static_parts(if anchored { vec![parts[0]] } else { vec![] })
        .unwrap()
}

fn car() -> CompiledCreation {
    let doc: mechanic_world::WorldCreationInstanceDoc = ron::from_str(include_str!(
        "../../../../mechanic-bench/tests/fixtures/driven_car_instance.ron"
    ))
    .unwrap();
    let loaded = doc.creation.into_graph().unwrap();
    loaded
        .graph
        .compile_with_suspension_sockets([], &loaded.sockets)
        .unwrap()
}

fn assert_close(a: &[f64], b: &[f64], tolerance: f64) {
    let scale = b.iter().map(|v| v.abs()).fold(1.0, f64::max);
    let error = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(error < tolerance * scale, "error={error:e} scale={scale:e}");
}

fn compare(creation: &CompiledCreation, rotate: bool) {
    let mut roots = MachineDynamics::initial_roots(creation);
    if rotate {
        for root in &mut roots {
            root.position += DVec3::new(4.0, -7.0, 2.0);
            root.rotation = DQuat::from_euler(bevy_math::EulerRot::XYZ, 0.37, -0.24, 0.83);
        }
    }
    let mut coordinates = vec![0.0; creation.dynamics.coordinate_bearings.len()];
    for (q, &bearing) in coordinates
        .iter_mut()
        .zip(&creation.dynamics.coordinate_bearings)
    {
        *q = if creation.bearings[bearing].kind.is_translational() {
            0.01
        } else {
            0.29
        };
    }
    let model = MachineDynamics::assemble(creation, &roots, &coordinates).unwrap();
    let size = creation.dynamics.elimination_parent.len();
    for implicit in [0.0, 0.73, 100.0] {
        let diagonal = vec![implicit; size];
        let dense = model.factor(&diagonal).unwrap();
        let tree = DynamicsFactor::articulated(creation, &roots, &coordinates, &diagonal).unwrap();
        // Each independent impulse checks all couplings, not just a chosen motion.
        for column in 0..size {
            let mut expected = vec![0.0; size];
            expected[column] = 1.0;
            let rhs = expected.clone();
            let mut actual = expected.clone();
            dense.solve(&mut expected).unwrap();
            tree.solve(&mut actual).unwrap();
            assert_close(&actual, &expected, 2e-10);
            let applied: Vec<_> = model
                .mass_matrix
                .chunks_exact(size)
                .zip(&diagonal)
                .enumerate()
                .map(|(row, (entries, shift))| {
                    entries.iter().zip(&actual).map(|(a, b)| a * b).sum::<f64>()
                        + shift * actual[row]
                })
                .collect();
            assert_close(&applied, &rhs, 2e-9);
            let mut repeated = rhs;
            tree.solve(&mut repeated).unwrap();
            assert_eq!(actual, repeated);
        }
    }
}

#[test]
fn tree_inverse_matches_dense_for_floating_anchored_and_reversed_joints() {
    for anchored in [false, true] {
        for reversed in [false, true] {
            let mut creation = chain(5, anchored, reversed);
            // Exercise nonzero COM arms and strongly unequal physical inertias.
            for (index, inertia) in creation.dynamics.inertias.iter_mut().enumerate() {
                inertia.center += Vec3::new(0.07, -0.11, 0.04);
                let scale = if index % 2 == 0 { 0.01 } else { 10.0 };
                inertia.mass *= scale;
                inertia.rotational *= scale;
            }
            compare(&creation, false);
            compare(&creation, true);
        }
    }
}

#[test]
fn authored_car_suspension_and_branched_inertia_match_dense() {
    compare(&car(), false);
    compare(&car(), true);
}

#[test]
fn independent_floating_and_fixed_components_remain_uncoupled() {
    let mut graph = ConstructionGraph::new();
    let mut parts = Vec::new();
    for x in [0, 8, 16] {
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 2, 1],
                    BuildPose::new(IVec3::X * x, GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            panic!("spawn");
        };
        parts.push(part);
    }
    compare(&graph.compile_with_static_parts([parts[1]]).unwrap(), true);
    compare(&graph.compile_with_static_parts(parts).unwrap(), false);
}

#[test]
fn large_coaxial_chain_matches_independent_torque_balance_without_dense_storage() {
    let creation = chain(600, true, false);
    let roots = MachineDynamics::initial_roots(&creation);
    let coordinates = vec![0.0; 600];
    assert_eq!(
        MachineDynamics::assemble(&creation, &roots, &coordinates).unwrap_err(),
        PhysicsError::ReferenceCapacity
    );
    let diagonal = vec![0.23; 600];
    let factor = DynamicsFactor::articulated(&creation, &roots, &coordinates, &diagonal).unwrap();
    // For coaxial rotors, absolute acceleration is the prefix sum of relative
    // accelerations, and each actuator carries the suffix sum of inertia torque.
    let expected: Vec<_> = (0..600).map(|i| (f64::from(i) * 0.71).sin()).collect();
    let mut acceleration = 0.0;
    let mut torques = Vec::new();
    for (qdd, body) in expected.iter().zip(1..=600) {
        acceleration += qdd;
        torques
            .push(acceleration * f64::from(creation.dynamics.inertias[body].rotational.x_axis.x));
    }
    let mut total = 0.0;
    for torque in torques.iter_mut().rev() {
        total += *torque;
        *torque = total;
    }
    for (torque, qdd) in torques.iter_mut().zip(&expected) {
        *torque += 0.23 * qdd;
    }
    factor.solve(&mut torques).unwrap();
    assert_close(&torques, &expected, 1e-9);
}

#[test]
fn coupled_car_contact_response_uses_tree_inverse_with_original_friction_bounds() {
    let creation = car();
    let roots = MachineDynamics::initial_roots(&creation);
    let coordinates = vec![0.0; creation.dynamics.coordinate_bearings.len()];
    let model = MachineDynamics::assemble(&creation, &roots, &coordinates).unwrap();
    let diagonal = vec![0.01; creation.dynamics.elimination_parent.len()];
    let dense = model.factor(&diagonal).unwrap();
    let tree = DynamicsFactor::articulated(&creation, &roots, &coordinates, &diagonal).unwrap();
    let body = creation.compounds.len() - 1;
    let point = model.poses[body].position + DVec3::new(0.12, -0.17, 0.08);
    let block = ConstraintBlock {
        jacobian: [DVec3::Y, DVec3::X, DVec3::Z]
            .into_iter()
            .map(|direction| model.point_row(body, point, direction).unwrap())
            .collect(),
        target: vec![1.0, 0.01, -0.02],
        bounds: vec![
            ImpulseBounds {
                minimum: 0.0,
                maximum: f64::INFINITY,
            },
            ImpulseBounds {
                minimum: f64::NEG_INFINITY,
                maximum: f64::INFINITY,
            },
            ImpulseBounds {
                minimum: f64::NEG_INFINITY,
                maximum: f64::INFINITY,
            },
        ],
        contacts: vec![ContactFriction {
            static_coefficient: 0.8,
            kinetic_coefficient: 0.6,
            sliding: false,
            rolling_length: None,
        }],
    };
    for copies in [1, 43] {
        // Duplicate finite constraints deliberately cross the 128-row boundary;
        // a valid solution must satisfy every row on either storage route.
        let blocks = vec![block.clone(); copies];
        let a = solve_constraints(&dense, &blocks, 256, 1e-9).unwrap();
        let b = solve_constraints(&tree, &blocks, 256, 1e-9).unwrap();
        assert!(a.converged && b.converged);
        assert_close(&b.velocity_change, &a.velocity_change, 1e-9);
        assert_close(&b.impulses, &a.impulses, 1e-9);
        assert_eq!(b.response_storage, if copies == 1 { 9 } else { 0 });
        for impulse in b.impulses.chunks_exact(3) {
            assert!(impulse[0] >= 0.0);
            assert!(impulse[1].hypot(impulse[2]) <= 0.8 * impulse[0] + 1e-9);
        }
    }
}

#[test]
fn tree_factor_rejects_invalid_poses_diagonals_and_impulses() {
    let creation = chain(1, false, false);
    let mut roots = MachineDynamics::initial_roots(&creation);
    for diagonal in [
        vec![0.0; 6],
        vec![-1.0; 7],
        vec![f64::NAN; 7],
        vec![f64::INFINITY; 7],
    ] {
        assert!(DynamicsFactor::articulated(&creation, &roots, &[0.0], &diagonal).is_err());
    }
    let factor = DynamicsFactor::articulated(&creation, &roots, &[0.0], &[0.0; 7]).unwrap();
    assert!(factor.solve(&mut [0.0; 6]).is_err());
    assert!(factor.solve(&mut [f64::NAN; 7]).is_err());
    roots[0].rotation = DQuat::from_xyzw(0.0, 0.0, 0.0, 2.0);
    assert!(DynamicsFactor::articulated(&creation, &roots, &[0.0], &[0.0; 7]).is_err());
}

#[test]
fn tree_com_velocities_match_dense_jacobians_in_rotated_mixed_mechanisms() {
    for creation in [chain(5, false, false), chain(5, true, true), car()] {
        let mut creation = creation;
        for inertia in &mut creation.dynamics.inertias {
            inertia.center += Vec3::new(0.07, -0.11, 0.04);
        }
        let mut roots = MachineDynamics::initial_roots(&creation);
        for root in &mut roots {
            root.rotation = DQuat::from_euler(bevy_math::EulerRot::XYZ, 0.37, -0.24, 0.83);
        }
        let coordinates = vec![0.01; creation.dynamics.coordinate_bearings.len()];
        let model = MachineDynamics::assemble(&creation, &roots, &coordinates).unwrap();
        let size = creation.dynamics.elimination_parent.len();
        for column in 0..size {
            let mut rates = vec![0.0; size];
            rates[column] = 1.0;
            let dense = model.body_motions(&rates).unwrap();
            let tree =
                MachineDynamics::reconstruct_motions(&creation, &model.poses, &rates).unwrap();
            for (expected, actual) in dense.iter().zip(tree) {
                assert!(expected.linear.distance(actual.linear) < 1e-12);
                assert!(expected.angular.distance(actual.angular) < 1e-12);
            }
        }
    }
    let creation = chain(600, true, false);
    let poses = MachineDynamics::reconstruct_poses(
        &creation,
        &MachineDynamics::initial_roots(&creation),
        &vec![0.0; 600],
    )
    .unwrap();
    let rates = (0..600)
        .map(|i| (f64::from(i) * 0.71).sin())
        .collect::<Vec<_>>();
    let motions = MachineDynamics::reconstruct_motions(&creation, &poses, &rates).unwrap();
    let mut omega = 0.0;
    for (body, motion) in motions.iter().enumerate() {
        if body > 0 {
            omega += rates[body - 1];
        }
        assert!(motion.angular.distance(DVec3::X * omega) < 1e-12);
        assert!(motion.linear.length() < 1e-12);
    }
}
