use super::*;

#[test]
fn independent_support_transition_reference_satisfies_the_runtime_contact_laws() {
    let (mass, recorded): (Vec<f64>, Vec<RecordedBlock>) =
        ron::from_str(include_str!("fixtures/support_transition_impact.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor::new(&mass, blocks[0].jacobian[0].len()).unwrap();
    let reference = include_str!("fixtures/support_transition_reference.ron");
    let (impulses, _): (Vec<f64>, Vec<f64>) = ron::from_str(reference).unwrap();
    let targets = blocks
        .iter()
        .flat_map(|block| &block.target)
        .copied()
        .collect::<Vec<_>>();
    let solution = PreparedConstraints::new(&factor, &blocks)
        .unwrap()
        .solve_from(&targets, Some(&impulses), 256, 1e-9)
        .unwrap();
    assert!(solution.converged);
    check_contact_laws(&blocks, &solution.impulses, &solution.velocity_change, true);
    check_independent_motion(&blocks, &factor, &solution, reference);
}

#[test]
fn support_transition_impact_satisfies_all_original_contact_laws() {
    let (mass, recorded): (Vec<f64>, Vec<RecordedBlock>) =
        ron::from_str(include_str!("fixtures/support_transition_impact.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor::new(&mass, blocks[0].jacobian[0].len()).unwrap();
    let solution = solve_constraints(&factor, &blocks, 256, 1e-9).unwrap();
    println!(
        "support transition residual={:e} iterations={} continuation={}",
        solution.residual, solution.iterations, solution.continuation_iterations
    );
    assert!(solution.converged);
    check_contact_laws(&blocks, &solution.impulses, &solution.velocity_change, true);
}

#[test]
fn post_drive_endpoint_impact_satisfies_all_original_contact_laws() {
    let (mass, recorded): (Vec<f64>, Vec<RecordedBlock>) =
        ron::from_str(include_str!("fixtures/post_drive_endpoint_impact.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor::new(&mass, blocks[0].jacobian[0].len()).unwrap();
    let solution = solve_constraints(&factor, &blocks, 256, 1e-9).unwrap();
    println!(
        "post-drive residual={:e} iterations={} continuation={} stages={}",
        solution.residual,
        solution.iterations,
        solution.continuation_iterations,
        solution.continuation_stages
    );
    assert!(solution.converged);
    check_contact_laws(&blocks, &solution.impulses, &solution.velocity_change, true);
}

#[test]
fn bounded_drives_and_terrain_contacts_converge_together() {
    let (mass, recorded, initial): (Vec<f64>, Vec<RecordedBlock>, Option<Vec<f64>>) =
        ron::from_str(include_str!("fixtures/coupled_drive_contact.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor::new(&mass, blocks[0].jacobian[0].len()).unwrap();
    let targets = blocks
        .iter()
        .flat_map(|block| &block.target)
        .copied()
        .collect::<Vec<_>>();
    let mut response = PreparedConstraints::new(&factor, &blocks).unwrap();
    for guess in [None, initial.as_deref()] {
        let result = response.solve_from(&targets, guess, 256, 1e-9).unwrap();
        println!(
            "coupled drives residual={:e} iterations={} continuation={} transitions={} trials={}",
            result.residual,
            result.iterations,
            result.continuation_iterations,
            result.friction_transitions,
            result.continuation_trials
        );
        assert!(result.converged);
        let mut final_blocks = blocks.clone();
        let mut modes = result.sliding.iter();
        for contact in final_blocks
            .iter_mut()
            .flat_map(|block| &mut block.contacts)
        {
            contact.sliding = *modes.next().unwrap();
        }
        assert!(modes.next().is_none());
        check_contact_laws(
            &final_blocks,
            &result.impulses,
            &result.velocity_change,
            false,
        );
        let mut first = 0;
        for block in &blocks {
            if block.contacts.is_empty() {
                for (index, bounds) in block.bounds.iter().enumerate() {
                    let impulse = result.impulses[first + index];
                    let slack =
                        dot(&block.jacobian[index], &result.velocity_change) - block.target[index];
                    assert!(impulse >= bounds.minimum && impulse <= bounds.maximum);
                    if impulse > bounds.minimum {
                        assert!(slack <= 1e-9);
                    }
                    if impulse < bounds.maximum {
                        assert!(slack >= -1e-9);
                    }
                }
            }
            first += block.target.len();
        }
    }
}

#[test]
fn later_settling_endpoint_impact_satisfies_all_original_contact_laws() {
    let (mass, recorded): (Vec<f64>, Vec<RecordedBlock>) =
        ron::from_str(include_str!("fixtures/later_settling_endpoint_impact.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor::new(&mass, blocks[0].jacobian[0].len()).unwrap();
    let solution = solve_constraints(&factor, &blocks, 256, 1e-9).unwrap();
    println!(
        "later settling residual={:e} iterations={} continuation={}",
        solution.residual, solution.iterations, solution.continuation_iterations
    );
    assert!(solution.converged);
    check_contact_laws(&blocks, &solution.impulses, &solution.velocity_change, true);
}

#[test]
fn settling_endpoint_impact_satisfies_all_original_contact_laws() {
    let (mass, recorded): (Vec<f64>, Vec<RecordedBlock>) =
        ron::from_str(include_str!("fixtures/settling_endpoint_impact.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor::new(&mass, blocks[0].jacobian[0].len()).unwrap();
    let solution = solve_constraints(&factor, &blocks, 256, 1e-9).unwrap();
    println!(
        "settling residual={:e} iterations={} continuation={}",
        solution.residual, solution.iterations, solution.continuation_iterations
    );
    assert!(solution.converged);
    check_contact_laws(&blocks, &solution.impulses, &solution.velocity_change, true);
}

// Captured at tick 1 of the fixed-eight endpoint drop: contacts plus active
// joint stops, which previously bypassed the full-space contact search.
#[test]
fn fixed_eight_stop_and_contact_impact_satisfies_all_original_laws() {
    let (mass, recorded): (Vec<f64>, Vec<RecordedBlock>) =
        ron::from_str(include_str!("fixtures/fixed_eight_stop_impact.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor::new(&mass, blocks[0].jacobian[0].len()).unwrap();
    let solution = solve_constraints(&factor, &blocks, 256, 1e-9).unwrap();
    println!(
        "stop impact residual={:e} iterations={} continuation={} full_rows={}",
        solution.residual,
        solution.iterations,
        solution.continuation_iterations,
        solution.continuation_full_rows
    );
    assert!(solution.converged);
    check_contact_laws(&blocks, &solution.impulses, &solution.velocity_change, true);
    let repeated = solve_constraints(&factor, &blocks, 256, 1e-9).unwrap();
    assert_eq!(solution.impulses, repeated.impulses);
}

// Captured at tick 11 of the fixed-eight endpoint drop once tick 1 passed.
#[test]
fn fixed_eight_tick_eleven_impact_satisfies_all_original_contact_laws() {
    let (mass, recorded): (Vec<f64>, Vec<RecordedBlock>) =
        ron::from_str(include_str!("fixtures/fixed_eight_tick11_impact.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor::new(&mass, blocks[0].jacobian[0].len()).unwrap();
    let solution = solve_constraints(&factor, &blocks, 256, 1e-9).unwrap();
    println!(
        "tick eleven residual={:e} iterations={} continuation={} full_rows={}",
        solution.residual,
        solution.iterations,
        solution.continuation_iterations,
        solution.continuation_full_rows
    );
    assert!(solution.converged);
    check_contact_laws(&blocks, &solution.impulses, &solution.velocity_change, true);
}

// Captured inside the tick-37 endpoint-policy force loop: static contacts plus
// bounded drive rows, from both a cold and the recorded warm start.
#[test]
fn endpoint_tick37_drives_and_contacts_satisfy_all_original_laws() {
    let (mass, recorded, initial): (Vec<f64>, Vec<RecordedBlock>, Option<Vec<f64>>) =
        ron::from_str(include_str!("fixtures/endpoint_tick37_force.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor::new(&mass, blocks[0].jacobian[0].len()).unwrap();
    let targets = blocks
        .iter()
        .flat_map(|block| &block.target)
        .copied()
        .collect::<Vec<_>>();
    let mut response = PreparedConstraints::new(&factor, &blocks).unwrap();
    for guess in [None, initial.as_deref()] {
        let result = response.solve_from(&targets, guess, 256, 1e-9).unwrap();
        println!(
            "tick37 warm={} residual={:e} iterations={} continuation={} full_rows={}",
            guess.is_some(),
            result.residual,
            result.iterations,
            result.continuation_iterations,
            result.continuation_full_rows
        );
        assert!(result.converged);
        let mut final_blocks = blocks.clone();
        let mut modes = result.sliding.iter();
        for contact in final_blocks
            .iter_mut()
            .flat_map(|block| &mut block.contacts)
        {
            contact.sliding = *modes.next().unwrap();
        }
        check_contact_laws(
            &final_blocks,
            &result.impulses,
            &result.velocity_change,
            false,
        );
    }
}

// Captured at tick 94 of the endpoint-policy drop once tick 37 passed.
#[test]
fn endpoint_tick94_impact_satisfies_all_original_contact_laws() {
    let (mass, recorded): (Vec<f64>, Vec<RecordedBlock>) =
        ron::from_str(include_str!("fixtures/endpoint_tick94_impact.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor::new(&mass, blocks[0].jacobian[0].len()).unwrap();
    let solution = solve_constraints(&factor, &blocks, 256, 1e-9).unwrap();
    println!(
        "tick94 residual={:e} iterations={} continuation={} full_rows={}",
        solution.residual,
        solution.iterations,
        solution.continuation_iterations,
        solution.continuation_full_rows
    );
    assert!(solution.converged);
    check_contact_laws(&blocks, &solution.impulses, &solution.velocity_change, true);
}

// Captured at tick 115 of the endpoint-policy drop once manifold rolling merged.
#[test]
fn endpoint_tick115_impact_satisfies_all_original_contact_laws() {
    let (mass, recorded): (Vec<f64>, Vec<RecordedBlock>) =
        ron::from_str(include_str!("fixtures/endpoint_tick115_impact.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor::new(&mass, blocks[0].jacobian[0].len()).unwrap();
    let solution = solve_constraints(&factor, &blocks, 256, 1e-9).unwrap();
    println!(
        "tick115 residual={:e} iterations={} continuation={} full_rows={}",
        solution.residual,
        solution.iterations,
        solution.continuation_iterations,
        solution.continuation_full_rows
    );
    assert!(solution.converged);
    check_contact_laws(&blocks, &solution.impulses, &solution.velocity_change, true);
}

type RecordedBlock = (
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<(f64, f64)>,
    Vec<(f64, f64, bool, Option<f64>)>,
);

#[test]
fn event_resolved_car_retry_impact_satisfies_all_original_contact_laws() {
    let (mass, recorded): (Vec<f64>, Vec<RecordedBlock>) =
        ron::from_str(include_str!("fixtures/event_retry_impact.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor::new(&mass, blocks[0].jacobian[0].len()).unwrap();
    let solution = solve_constraints(&factor, &blocks, 256, 1e-9).unwrap();
    assert!(solution.converged, "{solution:?}");
    assert!(solution.iterations <= 256);
    assert_eq!(solution.response_storage, 50 * 50);
    assert_eq!(solution.continuation_trials, 1);
    assert!(solution.continuation_iterations > 0 && solution.continuation_iterations <= 128);
    check_contact_laws(&blocks, &solution.impulses, &solution.velocity_change, true);
    check_independent_motion(
        &blocks,
        &factor,
        &solution,
        include_str!("fixtures/event_retry_reference.ron"),
    );
    let repeated = solve_constraints(&factor, &blocks, 256, 1e-9).unwrap();
    assert_eq!(solution.impulses, repeated.impulses);
    assert_eq!(solution.velocity_change, repeated.velocity_change);
    let exhausted = solve_constraints(&factor, &blocks, 64, 1e-9).unwrap();
    assert!(!exhausted.converged);
    assert_eq!(exhausted.iterations, 64);
    assert!(exhausted.continuation_iterations < solution.continuation_iterations);
    println!(
        "event_retry_impact residual={:e} iterations={} continuation={}",
        solution.residual, solution.iterations, solution.continuation_iterations
    );
}

#[test]
fn captured_cold_car_impact_converges_without_relaxing_its_velocity_residual() {
    let (mass, recorded): (Vec<f64>, Vec<RecordedBlock>) =
        ron::from_str(include_str!("fixtures/cold_car_impact.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor::new(&mass, blocks[0].jacobian[0].len()).unwrap();
    let solution = solve_constraints(&factor, &blocks, 256, 1e-9).unwrap();
    assert!(solution.converged, "residual={:e}", solution.residual);
    assert!(solution.residual <= 1e-9);
    assert!(solution.iterations <= 256);
    // Independently check every original contact, including the inactive
    // hypothesis. No normal row may be hidden by the trial's temporary bound.
    let mut first = 0;
    for block in &blocks {
        let mut row = 0;
        for contact in &block.contacts {
            let normal = solution.impulses[first + row];
            let slack = dot(&block.jacobian[row], &solution.velocity_change) - block.target[row];
            assert!(normal >= 0.0 && slack >= -1e-9);
            if normal > 0.0 {
                assert!(slack.abs() <= 1e-9);
            }
            let tangent =
                solution.impulses[first + row + 1].hypot(solution.impulses[first + row + 2]);
            assert!(tangent <= contact.kinetic_coefficient * normal + 1e-12);
            if let Some(length) = contact.rolling_length {
                let rolling =
                    solution.impulses[first + row + 3].hypot(solution.impulses[first + row + 4]);
                assert!(rolling <= length * normal + 1e-12);
            }
            row += contact.rows();
        }
        first += block.target.len();
    }
    let repeated = solve_constraints(&factor, &blocks, 256, 1e-9).unwrap();
    assert_eq!(solution.impulses, repeated.impulses);
    assert_eq!(solution.velocity_change, repeated.velocity_change);
    println!(
        "captured_impact residual={:e} iterations={} inactive_trials={}",
        solution.residual, solution.iterations, solution.active_set_trials
    );
}

fn blocks(recorded: Vec<RecordedBlock>) -> Vec<ConstraintBlock> {
    recorded
        .into_iter()
        .map(|(jacobian, target, bounds, contacts)| ConstraintBlock {
            jacobian,
            target,
            bounds: bounds
                .into_iter()
                .map(|(minimum, maximum)| ImpulseBounds { minimum, maximum })
                .collect(),
            contacts: contacts
                .into_iter()
                .map(
                    |(static_coefficient, kinetic_coefficient, sliding, rolling_length)| {
                        ContactFriction {
                            static_coefficient,
                            kinetic_coefficient,
                            sliding,
                            rolling_length,
                        }
                    },
                )
                .collect(),
        })
        .collect::<Vec<_>>()
}

#[test]
fn a_wrong_inactive_contact_trial_is_discarded_without_losing_the_original_solution() {
    let (lower, recorded, initial): (Vec<f64>, Vec<RecordedBlock>, Option<Vec<f64>>) =
        ron::from_str(include_str!("fixtures/rejected_contact_trial.ron")).unwrap();
    let blocks = blocks(recorded);
    let size = blocks[0].jacobian[0].len();
    assert_eq!(lower.len(), size * size);
    let factor = DynamicsFactor {
        size,
        storage: super::super::FactorStorage::Dense(lower),
    };
    let targets = blocks
        .iter()
        .flat_map(|b| b.target.clone())
        .collect::<Vec<_>>();
    let mut response = PreparedConstraints::new(&factor, &blocks).unwrap();
    let original = response
        .solve_with_contact_trial(&targets, initial.as_deref(), 256, 1e-9, false)
        .unwrap();
    let actual = response
        .solve_from(&targets, initial.as_deref(), 256, 1e-9)
        .unwrap();
    println!("rejected_trial original={original:?} actual={actual:?}");
    assert!(original.converged && actual.converged);
    assert!(actual.active_set_trial_rejections > 0);
    assert!(actual.iterations <= 256);
    for (a, b) in actual.velocity_change.iter().zip(&original.velocity_change) {
        assert!((a - b).abs() < 1e-8);
    }
}

// Captured finite-duration car solve: the warm guess reaches the static cone
// only at sweep 241, leaving too little of its 256-sweep budget after breakaway.
#[test]
fn stalled_warm_contact_restarts_within_the_original_budget_and_proves_breakaway() {
    let (lower, recorded, initial): (Vec<f64>, Vec<RecordedBlock>, Option<Vec<f64>>) =
        ron::from_str(include_str!("fixtures/cold_settling.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor {
        size: blocks[0].jacobian[0].len(),
        storage: super::super::FactorStorage::Dense(lower),
    };
    let targets = blocks
        .iter()
        .flat_map(|b| b.target.clone())
        .collect::<Vec<_>>();
    let mut response = PreparedConstraints::new(&factor, &blocks).unwrap();
    let cold = response.solve_from(&targets, None, 256, 1e-9).unwrap();
    let actual = response
        .solve_from(&targets, initial.as_deref(), 256, 1e-9)
        .unwrap();
    assert!(cold.converged && actual.converged);
    assert_eq!(cold.warm_start_restarts, 0);
    assert_eq!(actual.warm_start_restarts, 1);
    assert!(actual.iterations > cold.iterations && actual.iterations <= 256);
    assert_eq!(actual.impulses, cold.impulses);
    assert_eq!(actual.velocity_change, cold.velocity_change);
    assert_eq!(actual.sliding, [true; 4]);
    assert_eq!(actual.friction_transitions, 1);
    check_contact_laws(&blocks, &actual.impulses, &actual.velocity_change, true);
    let repeated = response
        .solve_from(&targets, initial.as_deref(), 256, 1e-9)
        .unwrap();
    assert_eq!(actual.impulses, repeated.impulses);
    assert_eq!(actual.velocity_change, repeated.velocity_change);
    let limited = response
        .solve_from(&targets, initial.as_deref(), 64, 1e-9)
        .unwrap();
    assert_eq!(limited.warm_start_restarts, 1);
    assert_eq!(limited.iterations, 64);
    assert!(!limited.converged && limited.residual > 1e-9);
    println!(
        "cold_settling warm_sweeps={} cold_sweeps={} residual={:e} restarts={}",
        actual.iterations, cold.iterations, actual.residual, actual.warm_start_restarts
    );

    // Independent NumPy reference retained in the evidence checkpoint. Before
    // promotion it satisfies every recorded static-mode equation, yet the first
    // point has nonzero slip at its static impulse limit. This establishes the
    // existing breakaway rule without inferring it from an unfinished iterate.
    let static_impulses: Vec<f64> =
        ron::from_str(include_str!("fixtures/cold_settling_static.ron")).unwrap();
    let rows = blocks.iter().flat_map(|b| &b.jacobian).collect::<Vec<_>>();
    let (velocity, _) = response_motion(&factor, &rows, &static_impulses).unwrap();
    check_contact_laws(&blocks, &static_impulses, &velocity, false);
    let slip = (dot(rows[7], &velocity) - targets[7]).hypot(dot(rows[8], &velocity) - targets[8]);
    assert!(slip > 5e-6);
    assert!(
        (static_impulses[7].hypot(static_impulses[8])
            - blocks[6].contacts[0].static_coefficient * static_impulses[6])
            .abs()
            < 1e-8
    );
}

// Checks bounds, normal complementarity, and disk direction/complementarity
// directly in physical row velocities, without using the solver's projection.
fn check_contact_laws(
    blocks: &[ConstraintBlock],
    impulses: &[f64],
    velocity: &[f64],
    kinetic: bool,
) {
    let mut first = 0;
    for block in blocks {
        let slack = block
            .jacobian
            .iter()
            .zip(&block.target)
            .map(|(row, target)| dot(row, velocity) - target)
            .collect::<Vec<_>>();
        for (local, bounds) in block.bounds.iter().enumerate() {
            let impulse = impulses[first + local];
            assert!(impulse >= bounds.minimum - 1e-8 && impulse <= bounds.maximum + 1e-8);
            if block.contacts.is_empty() {
                if impulse > bounds.minimum + 1e-8 {
                    assert!(slack[local] <= 1e-8);
                }
                if impulse < bounds.maximum - 1e-8 {
                    assert!(slack[local] >= -1e-8);
                }
            }
        }
        let mut local = 0;
        for law in &block.contacts {
            let normal = impulses[first + local];
            assert!(normal >= 0.0 && slack[local] >= -1e-9);
            if normal > 1e-8 {
                assert!(slack[local].abs() <= 1e-9);
            }
            let coefficient = if kinetic || law.sliding {
                law.kinetic_coefficient
            } else {
                law.static_coefficient
            };
            for (offset, coefficient) in std::iter::once((1, coefficient))
                .chain(law.rolling_length.map(|length| (3, length)))
            {
                let a = impulses[first + local + offset];
                let b = impulses[first + local + offset + 1];
                let u = slack[local + offset];
                let v = slack[local + offset + 1];
                let radius = coefficient * normal;
                assert!(a.hypot(b) <= radius + 1e-8);
                if u.hypot(v) > 1e-8 {
                    assert!((a.hypot(b) - radius).abs() < 1e-8);
                    // Sliding friction opposes motion, including rolling motion.
                    assert!(
                        a * u + b * v <= 0.0,
                        "row={} offset={offset} normal={normal:e} impulse=({a:e},{b:e}) slip=({u:e},{v:e})",
                        first + local,
                    );
                    assert!((a * v - b * u).abs() / radius.max(1e-12) < 1e-8);
                }
            }
            local += law.rows();
        }
        first += block.target.len();
    }
}

#[test]
fn a_loaded_near_dependent_contact_can_release_without_relaxing_any_original_row() {
    let (mass, recorded): (Vec<f64>, Vec<RecordedBlock>) =
        ron::from_str(include_str!("fixtures/late_cold_impact.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor::new(&mass, blocks[0].jacobian[0].len()).unwrap();
    let run = || {
        let solution = solve_constraints(&factor, &blocks, 256, 1e-9).unwrap();
        assert!(solution.converged, "residual={:e}", solution.residual);
        assert!(solution.iterations <= 256);
        assert_eq!(solution.active_set_trials, 1);
        check_contact_laws(&blocks, &solution.impulses, &solution.velocity_change, true);
        println!(
            "late impact residual={:e} iterations={} inactive_trials={}",
            solution.residual, solution.iterations, solution.active_set_trials
        );
        solution
    };
    let first = run();
    let repeated = run();
    assert_eq!(first.impulses, repeated.impulses);
    assert_eq!(first.velocity_change, repeated.velocity_change);
}

#[test]
fn captured_endpoint_impact_satisfies_all_original_contact_laws() {
    let (mass, recorded): (Vec<f64>, Vec<RecordedBlock>) =
        ron::from_str(include_str!("fixtures/slow_endpoint_impact.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor::new(&mass, blocks[0].jacobian[0].len()).unwrap();
    let run = || {
        let solution = solve_constraints(&factor, &blocks, 256, 1e-9).unwrap();
        println!(
            "endpoint converged={} residual={:e} iterations={} continuation={} stages={} factors={}",
            solution.converged,
            solution.residual,
            solution.iterations,
            solution.continuation_iterations,
            solution.continuation_stages,
            solution.factor_solves
        );
        assert!(solution.converged);
        assert!(solution.iterations <= 256);
        assert_eq!(solution.response_storage, 0);
        assert_eq!(solution.newton_contact_storage, 0);
        assert_eq!(solution.continuation_trials, 1);
        check_contact_laws(&blocks, &solution.impulses, &solution.velocity_change, true);
        solution
    };
    let first = run();
    let repeated = run();
    assert_eq!(first.impulses, repeated.impulses);
    assert_eq!(first.velocity_change, repeated.velocity_change);
    check_independent_motion(
        &blocks,
        &factor,
        &first,
        include_str!("fixtures/slow_endpoint_reference.ron"),
    );
}

#[test]
fn later_endpoint_impact_satisfies_all_original_contact_laws() {
    let (mass, recorded): (Vec<f64>, Vec<RecordedBlock>) =
        ron::from_str(include_str!("fixtures/late_endpoint_impact.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor::new(&mass, blocks[0].jacobian[0].len()).unwrap();
    let run = || {
        let solution = solve_constraints(&factor, &blocks, 256, 1e-9).unwrap();
        println!(
            "late_endpoint converged={} residual={:e} iterations={} continuation={} mixed_rows={} factors={}",
            solution.converged,
            solution.residual,
            solution.iterations,
            solution.continuation_iterations,
            solution.continuation_mixed_rows,
            solution.factor_solves
        );
        assert!(solution.converged);
        assert!(solution.iterations <= 256);
        assert_eq!(solution.response_storage, 0);
        assert_eq!(solution.newton_contact_storage, 0);
        assert!(solution.continuation_mixed_rows <= 128);
        check_contact_laws(&blocks, &solution.impulses, &solution.velocity_change, true);
        solution
    };
    let first = run();
    let repeated = run();
    assert_eq!(first.impulses, repeated.impulses);
    assert_eq!(first.velocity_change, repeated.velocity_change);
    check_independent_motion(
        &blocks,
        &factor,
        &first,
        include_str!("fixtures/late_endpoint_reference.ron"),
    );
    let exhausted = solve_constraints(&factor, &blocks, 32, 1e-9).unwrap();
    assert!(!exhausted.converged);
    assert!(exhausted.iterations <= 32);
}

#[test]
fn third_endpoint_impact_satisfies_all_original_contact_laws() {
    let (mass, recorded): (Vec<f64>, Vec<RecordedBlock>) =
        ron::from_str(include_str!("fixtures/third_endpoint_impact.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor::new(&mass, blocks[0].jacobian[0].len()).unwrap();
    let solution = solve_constraints(&factor, &blocks, 256, 1e-9).unwrap();
    println!(
        "third_endpoint converged={} residual={:e} iterations={} continuation={} stages={}",
        solution.converged,
        solution.residual,
        solution.iterations,
        solution.continuation_iterations,
        solution.continuation_stages
    );
    assert!(solution.converged);
    check_contact_laws(&blocks, &solution.impulses, &solution.velocity_change, true);
    assert!(solution.iterations <= 256);
    assert_eq!(solution.response_storage, 0);
    assert_eq!(solution.newton_contact_storage, 0);
    assert!(solution.continuation_mixed_rows <= 128);
    let repeated = solve_constraints(&factor, &blocks, 256, 1e-9).unwrap();
    assert_eq!(solution.impulses, repeated.impulses);
    assert_eq!(solution.velocity_change, repeated.velocity_change);
    let limited = solve_constraints(&factor, &blocks, 128, 1e-9).unwrap();
    assert!(!limited.converged);
    assert!(limited.iterations <= 128);
}

#[test]
fn independently_traced_third_impact_has_a_valid_solution() {
    let (mass, recorded): (Vec<f64>, Vec<RecordedBlock>) =
        ron::from_str(include_str!("fixtures/third_endpoint_impact.ron")).unwrap();
    let blocks = blocks(recorded);
    let factor = DynamicsFactor::new(&mass, blocks[0].jacobian[0].len()).unwrap();
    let reference = include_str!("fixtures/third_endpoint_reference.ron");
    let (impulses, _): (Vec<f64>, Vec<f64>) = ron::from_str(reference).unwrap();
    let targets = blocks
        .iter()
        .flat_map(|block| &block.target)
        .copied()
        .collect::<Vec<_>>();
    let mut response = PreparedConstraints::new(&factor, &blocks).unwrap();
    let solution = response
        .solve_from(&targets, Some(&impulses), 256, 1e-9)
        .unwrap();
    assert!(
        solution.converged,
        "reference residual={:e}",
        solution.residual
    );
    assert!(solution.residual <= 1e-9);
    check_contact_laws(&blocks, &solution.impulses, &solution.velocity_change, true);
    check_independent_motion(&blocks, &factor, &solution, reference);
    println!(
        "third_reference residual={:e} iterations={}",
        solution.residual, solution.iterations
    );
}

// The dense NumPy SVD reference has a separate implementation and search path.
// Validate its impulses through the actual dynamics before comparing motion.
fn check_independent_motion(
    blocks: &[ConstraintBlock],
    factor: &DynamicsFactor,
    solution: &ConstraintSolution,
    reference: &str,
) {
    let (impulses, velocity): (Vec<f64>, Vec<f64>) = ron::from_str(reference).unwrap();
    let rows = blocks
        .iter()
        .flat_map(|block| &block.jacobian)
        .collect::<Vec<_>>();
    let (reconstructed, _) = response_motion(factor, &rows, &impulses).unwrap();
    assert_eq!(reconstructed.len(), velocity.len());
    assert_eq!(solution.velocity_change.len(), velocity.len());
    for (actual, expected) in reconstructed.iter().zip(&velocity) {
        assert!((actual - expected).abs() < 1e-12);
    }
    check_contact_laws(blocks, &impulses, &reconstructed, true);
    let error = solution
        .velocity_change
        .iter()
        .zip(&velocity)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    println!("independent_impact_velocity_error={error:e}");
    assert!(error < 1e-8);
}
