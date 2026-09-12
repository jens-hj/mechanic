//! Analytic contact semantics for one free box on the flat floor, stated without
//! the saved car. Both the block material and Rock have zero restitution, so an
//! impact stops the box at the surface; a separating box keeps ballistic motion.
//! These bounds describe the intended physics, not a solver's counters.

use super::*;
use mechanic_core::ContactPolytope;

// Lowest collider point above the floor plane, measured independently of the
// solver's own contact queries. Negative values are penetration.
fn clearance(creation: &CompiledCreation, state: &MachineState) -> f64 {
    creation
        .colliders
        .iter()
        .map(|collider| {
            let pose = state.poses[collider.compound_index as usize];
            ContactPolytope::from_collider(collider)
                .unwrap()
                .transformed(pose.position, pose.rotation)
                .unwrap()
                .bounds()[0]
                .y
        })
        .fold(f64::INFINITY, f64::min)
}

// The cube's half extent is 0.5 m, so `height` is the initial clearance.
fn box_on_floor(
    height: f64,
    vertical: f64,
) -> (CompiledCreation, MachineCollisionGeometry, MachineState) {
    let (creation, geometry, _) = cube();
    let mut initial = MachineState::at_rest(&creation);
    initial.poses[0].position.y = 0.5 + height;
    initial.velocities[1] = vertical;
    (creation, geometry, initial)
}

#[test]
fn an_inelastic_drop_stops_at_the_surface_for_every_substep_policy() {
    for substeps in [1, 2, 4, 8] {
        let (creation, geometry, initial) = box_on_floor(0.002, -1.0);
        let scene = scene();
        let terrain = context(&scene, &geometry, 7);
        let mut machine = CpuJointMachine::new(creation.clone(), 7, initial).unwrap();
        for tick in 1..=10 {
            let result = machine
                .step_candidate(-DVec3::Y * 9.81, fixed(substeps), &[], &[], Some(&terrain))
                .map(|_| ());
            assert!(
                result.is_ok(),
                "substeps={substeps} tick={tick} result={result:?} diagnostics={:?}",
                machine.diagnostics()
            );
            let state = &machine.snapshot().state;
            let vertical = state.velocities[1];
            let clearance = clearance(&creation, state);
            println!(
                "drop substeps={substeps} tick={tick} vertical={vertical:e} clearance={clearance:e}"
            );
            // Gravity is the only force and restitution is zero, so the box can
            // never move upward, and it may not sink through the surface.
            assert!(
                vertical <= 1e-6,
                "substeps={substeps} tick={tick} rebound velocity={vertical:e}"
            );
            assert!(
                clearance >= -1e-6,
                "substeps={substeps} tick={tick} penetration clearance={clearance:e}"
            );
        }
        let state = &machine.snapshot().state;
        // Zero restitution: the box neither rebounds nor keeps sinking.
        assert!(
            state.velocities[1].abs() <= 1e-6,
            "substeps={substeps} vertical velocity={:e}",
            state.velocities[1]
        );
        let clearance = clearance(&creation, state);
        assert!(
            clearance.abs() <= 1e-6,
            "substeps={substeps} clearance={clearance:e}"
        );
    }
}

#[test]
fn a_settled_box_stays_within_the_activation_window() {
    let (creation, geometry, initial) = box_on_floor(0.0, 0.0);
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    let mut machine = CpuJointMachine::new(creation.clone(), 7, initial).unwrap();
    for tick in 1..=30 {
        let result = machine
            .step_candidate(
                -DVec3::Y * 9.81,
                JointTickSettings::default(),
                &[],
                &[],
                Some(&terrain),
            )
            .map(|_| ());
        assert!(
            result.is_ok(),
            "tick={tick} result={result:?} diagnostics={:?}",
            machine.diagnostics()
        );
        assert_eq!(
            machine.diagnostics().terrain_impact_holds,
            0,
            "tick={tick} event search exhausted its trials"
        );
        let clearance = clearance(&creation, &machine.snapshot().state);
        // A loaded support must stay at numerical zero. Drifting above the
        // activation window is what leaves the event search chasing an arrival
        // it can never localize.
        assert!(
            clearance.abs() <= 1e-9,
            "tick={tick} clearance={clearance:e}"
        );
    }
}

#[test]
fn a_separating_box_leaves_without_a_contact_impulse() {
    let (creation, geometry, initial) = box_on_floor(0.0, 0.5);
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    let mut machine = CpuJointMachine::new(creation, 7, initial).unwrap();
    let result = machine
        .step_candidate(
            -DVec3::Y * 9.81,
            JointTickSettings::default(),
            &[],
            &[],
            Some(&terrain),
        )
        .map(|_| ());
    assert!(
        result.is_ok(),
        "result={result:?} diagnostics={:?}",
        machine.diagnostics()
    );
    // Pure ballistic motion: a support cannot pull or push a departing box.
    let expected = 0.5 - 9.81 * TICK_SECONDS;
    let actual = machine.snapshot().state.velocities[1];
    assert!(
        (actual - expected).abs() <= 1e-9,
        "expected={expected:e} actual={actual:e}"
    );
}

#[test]
fn a_box_that_returns_within_one_tick_stops_on_the_surface() {
    // 0.05 m/s upward reaches its apex after 5.1 ms and returns after 10.2 ms,
    // inside one 60 Hz tick.
    let (creation, geometry, initial) = box_on_floor(0.0, 0.05);
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    let mut machine = CpuJointMachine::new(creation.clone(), 7, initial).unwrap();
    let result = machine
        .step_candidate(
            -DVec3::Y * 9.81,
            JointTickSettings::default(),
            &[],
            &[],
            Some(&terrain),
        )
        .map(|_| ());
    assert!(
        result.is_ok(),
        "result={result:?} diagnostics={:?}",
        machine.diagnostics()
    );
    let state = &machine.snapshot().state;
    assert!(
        state.velocities[1].abs() <= 1e-6,
        "vertical velocity={:e}",
        state.velocities[1]
    );
    let clearance = clearance(&creation, state);
    assert!(clearance.abs() <= 1e-6, "clearance={clearance:e}");
}
