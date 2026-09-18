use super::*;

fn bilateral(jacobian: Vec<Vec<f64>>, target: Vec<f64>) -> ConstraintBlock {
    ConstraintBlock {
        bounds: vec![
            ImpulseBounds {
                minimum: f64::NEG_INFINITY,
                maximum: f64::INFINITY
            };
            target.len()
        ],
        jacobian,
        target,
        contacts: Vec::new(),
    }
}

#[test]
fn impulse_at_light_wheel_accelerates_the_coupled_chassis() {
    // x is chassis translation; q is wheel displacement relative to chassis.
    let factor = DynamicsFactor::new(&[101.0, 1.0, 1.0, 1.0], 2).unwrap();
    let blocks = [bilateral(
        vec![vec![1.0, 1.0], vec![0.0, 1.0]],
        vec![1.0, 0.0],
    )];
    let result = solve_constraints(&factor, &blocks, 8, 1e-10).unwrap();
    assert!(result.converged);
    assert!((result.velocity_change[0] - 1.0).abs() < 1e-10);
    assert!(result.velocity_change[1].abs() < 1e-10);
    assert!((result.impulses[0] - 101.0).abs() < 1e-9);
}

#[test]
fn duplicate_loop_rows_preserve_response_and_inconsistent_rows_fail() {
    let factor = DynamicsFactor::new(&[2.0], 1).unwrap();
    let mut block = bilateral(vec![vec![1.0], vec![1.0]], vec![3.0, 3.0]);
    let result = solve_constraints(&factor, &[block.clone()], 8, 1e-10).unwrap();
    assert_eq!(result.velocity_change, [3.0]);
    block.target[1] = 4.0;
    assert_eq!(
        solve_constraints(&factor, &[block], 8, 1e-10).unwrap_err(),
        PhysicsError::InconsistentConstraints
    );
}

#[test]
fn unilateral_support_cannot_pull_a_separating_body() {
    let factor = DynamicsFactor::new(&[2.0], 1).unwrap();
    let mut block = bilateral(vec![vec![1.0]], vec![-1.0]);
    block.bounds[0].minimum = 0.0;
    let result = solve_constraints(&factor, &[block], 8, 1e-10).unwrap();
    assert!(result.converged);
    assert_eq!(result.velocity_change, [0.0]);
}

#[test]
fn nearly_coincident_supports_transfer_load_to_the_outer_contact() {
    let factor = identity_factor(2);
    // Incoming [vertical, angular] velocity is [-1, -2]. Only the outer
    // point remains in contact: impulse 1.5 leaves velocity [0.5, -0.5].
    // The other normal rows are dependent and must shed their initial load.
    for count in [3, 129] {
        let blocks = (0..count)
            .map(|index| {
                let x = if index == count - 1 { 1.0 } else { 0.999 };
                let mut block = bilateral(vec![vec![1.0, x]], vec![1.0 + 2.0 * x]);
                block.bounds[0].minimum = 0.0;
                block
            })
            .collect::<Vec<_>>();
        let result = solve_constraints(&factor, &blocks, 256, 1e-10).unwrap();
        assert!(
            result.converged,
            "rows={count} residual={}",
            result.residual
        );
        assert!((result.impulses[count - 1] - 1.5).abs() < 1e-9);
        assert!(
            result.impulses[..count - 1]
                .iter()
                .all(|value| value.abs() < 1e-9)
        );
        assert!((result.velocity_change[0] - 1.5).abs() < 1e-9);
        assert!((result.velocity_change[1] - 1.5).abs() < 1e-9);
        if count > DENSE_CONTACT_ROWS {
            assert_eq!(result.response_storage, 0);
        }
    }
}

#[test]
fn warm_impulses_reuse_an_implicit_response_and_are_revalidated_for_new_targets() {
    let factor = identity_factor(2);
    let blocks = (0..129)
        .map(|index| {
            let x = if index == 128 { 1.0 } else { 0.999 };
            let mut block = bilateral(vec![vec![1.0, x]], vec![1.0 + 2.0 * x]);
            block.bounds[0].minimum = 0.0;
            block
        })
        .collect::<Vec<_>>();
    let targets = blocks
        .iter()
        .flat_map(|block| block.target.iter().copied())
        .collect::<Vec<_>>();
    let mut prepared = PreparedConstraints::new(&factor, &blocks).unwrap();
    let cold = prepared.solve(&targets, 256, 1e-9).unwrap();
    assert!(cold.converged);
    let warm = prepared
        .solve_from(&targets, Some(&cold.impulses), 1, 1e-9)
        .unwrap();
    assert!(warm.converged);
    assert_eq!(warm.iterations, 1);
    assert!(warm.factor_solves < cold.factor_solves);
    assert_eq!(warm.response_storage, 0);
    let changed = targets
        .iter()
        .map(|target| target * 1.01)
        .collect::<Vec<_>>();
    let changed_warm = prepared
        .solve_from(&changed, Some(&warm.impulses), 256, 1e-9)
        .unwrap();
    assert!(changed_warm.converged);
    assert!((changed_warm.impulses[128] - 1.515).abs() < 1e-8);
    assert_eq!(prepared.preparation_factor_solves(), 129);
    for invalid in [vec![0.0], vec![f64::NAN; 129]] {
        assert!(matches!(
            prepared.solve_from(&targets, Some(&invalid), 256, 1e-9),
            Err(PhysicsError::InvalidConstraints)
        ));
    }
}

#[test]
fn coupling_to_a_clamped_contact_cannot_falsely_converge() {
    // W = [[2, 1], [1, 2]]. Clipping W^-1 * [-1, 1] gives [0, 1],
    // which leaves a nonzero residual on the active second contact.
    let factor = DynamicsFactor::new(&[2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0], 2).unwrap();
    let mut block = bilateral(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![-1.0, 1.0]);
    for bounds in &mut block.bounds {
        bounds.minimum = 0.0;
    }
    let result = solve_constraints(&factor, &[block], 128, 1e-10).unwrap();
    assert!(result.converged);
    assert!(result.impulses[0].abs() < 1e-10);
    assert!((result.impulses[1] - 0.5).abs() < 1e-10);
    assert!((result.velocity_change[1] - 1.0).abs() < 1e-10);
}

#[test]
fn friction_impulse_obeys_the_circular_coulomb_bound() {
    let factor = DynamicsFactor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3).unwrap();
    let mut block = bilateral(
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
        vec![2.0, 3.0, 4.0],
    );
    block.contacts = vec![ContactFriction {
        static_coefficient: 0.5,
        kinetic_coefficient: 0.5,
        sliding: false,
        rolling_length: None,
    }];
    block.bounds[0].minimum = 0.0;
    let result = solve_constraints(&factor, &[block], 8, 1e-10).unwrap();
    assert!(result.converged);
    assert!((result.impulses[1].hypot(result.impulses[2]) - 1.0).abs() < 1e-12);
}

#[test]
fn contradictory_friction_bounds_are_rejected_before_projection() {
    let factor = DynamicsFactor::new(&[1.0], 1).unwrap();
    let mut block = bilateral(vec![vec![1.0]; 3], vec![0.0; 3]);
    block.contacts = vec![ContactFriction {
        static_coefficient: 0.5,
        kinetic_coefficient: 0.5,
        sliding: false,
        rolling_length: None,
    }];
    block.bounds[0] = ImpulseBounds {
        minimum: -1.0,
        maximum: -1.0,
    };
    assert_eq!(
        solve_constraints(&factor, &[block], 8, 1e-10).unwrap_err(),
        PhysicsError::InvalidConstraints
    );
}

fn contact_block(target: Vec<f64>, sliding: bool, rolling_length: Option<f64>) -> ConstraintBlock {
    let size = target.len();
    let mut block = bilateral(
        (0..size)
            .map(|row| {
                let mut values = vec![0.0; size];
                values[row] = 1.0;
                values
            })
            .collect(),
        target,
    );
    block.bounds[0].minimum = 0.0;
    block.contacts = vec![ContactFriction {
        static_coefficient: 0.8,
        kinetic_coefficient: 0.5,
        sliding,
        rolling_length,
    }];
    block
}

fn identity_factor(size: usize) -> DynamicsFactor {
    let mut matrix = vec![0.0; size * size];
    for row in 0..size {
        matrix[row * size + row] = 1.0;
    }
    DynamicsFactor::new(&matrix, size).unwrap()
}

#[test]
fn static_support_and_initial_sliding_use_distinct_physical_friction_limits() {
    let factor = identity_factor(3);
    let held = solve_constraints(
        &factor,
        &[contact_block(vec![1.0, 0.6, 0.0], false, None)],
        16,
        1e-12,
    )
    .unwrap();
    assert!(held.converged);
    assert!((held.impulses[1] - 0.6).abs() < 1e-12);
    assert_eq!(held.sliding, [false]);
    assert_eq!(held.friction_transitions, 0);
    let moving = solve_constraints(
        &factor,
        &[contact_block(vec![1.0, 0.6, 0.0], true, None)],
        16,
        1e-12,
    )
    .unwrap();
    assert!(moving.converged);
    assert!((moving.impulses[1] - 0.5).abs() < 1e-12);
    assert_eq!(moving.sliding, [true]);
    // Free tangential speed was -0.6; kinetic friction leaves -0.1.
    assert!((moving.velocity_change[1] - 0.6 + 0.1).abs() < 1e-12);
}

#[test]
fn static_breakaway_reuses_response_and_cannot_pass_an_exhausted_solve() {
    let factor = identity_factor(3);
    let block = contact_block(vec![1.0, 1.0, 0.0], false, None);
    let mut prepared = PreparedConstraints::new(&factor, &[block]).unwrap();
    assert_eq!(prepared.preparation_factor_solves(), 3);
    let limited = prepared.solve(&[1.0, 1.0, 0.0], 1, 1e-12).unwrap();
    assert!(!limited.converged);
    assert_eq!(limited.friction_transitions, 1);
    let broken_away = prepared.solve(&[1.0, 1.0, 0.0], 16, 1e-12).unwrap();
    assert!(broken_away.converged);
    assert_eq!(broken_away.sliding, [true]);
    assert_eq!(broken_away.friction_transitions, 1);
    assert_eq!(
        broken_away.factor_solves, 1,
        "switching cones must reuse prepared W"
    );
    assert!((broken_away.impulses[1] - 0.5).abs() < 1e-12);
}

#[test]
fn coupled_normal_iterations_do_not_falsely_trigger_static_breakaway() {
    let factor = DynamicsFactor::new(&[1.0, -0.9, 0.0, -0.9, 1.0, 0.0, 0.0, 0.0, 1.0], 3).unwrap();
    // W is H^-1. The prescribed static solution is [1, 0.6, 0].
    let mut target = vec![1.0, 0.6, 0.0];
    factor.solve(&mut target).unwrap();
    let solution =
        solve_constraints(&factor, &[contact_block(target, false, None)], 1024, 1e-10).unwrap();
    assert!(solution.converged, "{}", solution.residual);
    assert_eq!(solution.friction_transitions, 0);
    assert_eq!(solution.sliding, [false]);
    assert!((solution.impulses[1] - 0.6).abs() < 1e-8);
}

#[test]
fn rolling_resistance_obeys_the_normal_load_radius_bound_and_dissipates_spin() {
    let factor = identity_factor(5);
    let solution = solve_constraints(
        &factor,
        &[contact_block(
            vec![2.0, 0.1, 0.0, 3.0, 4.0],
            false,
            Some(0.1),
        )],
        16,
        1e-12,
    )
    .unwrap();
    assert!(solution.converged);
    assert!((solution.impulses[3] - 0.12).abs() < 1e-12);
    assert!((solution.impulses[4] - 0.16).abs() < 1e-12);
    assert!((solution.impulses[3].hypot(solution.impulses[4]) - 0.2).abs() < 1e-12);
    let remaining = (-3.0 + solution.velocity_change[3]).hypot(-4.0 + solution.velocity_change[4]);
    assert!((remaining - 4.8).abs() < 1e-12);
}
