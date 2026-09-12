//! Analytic contact semantics for one free box on the flat floor, stated without
//! the saved car. The box is the default steel block (restitution 0.2) and Rock's
//! surface restitution is zero; a contact mixes them by taking the larger value.
//! Restitution applies only above the policy's restitution threshold, so an
//! impact either rebounds at 0.2 of its incoming speed or stops dead.
//! These bounds describe the intended physics, not a solver's counters.

use super::*;
use mechanic_core::ContactPolytope;

/// Default steel block restitution, mixed against Rock's zero by the maximum.
const BLOCK_RESTITUTION: f64 = 0.2;

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

// Free fall from `speed` over `height` reaches the surface at this time.
fn arrival_time(height: f64, speed: f64) -> f64 {
    (2.0_f64 * 9.81 * height + speed * speed)
        .sqrt()
        .mul_add(1.0, -speed)
        / 9.81
}

// Splits the tick at the analytic arrival time, separating the impact itself from
// the ballistic remainder that follows a rebound.
#[test]
fn an_impact_above_the_threshold_rebounds_at_the_material_restitution() {
    let (creation, geometry, initial) = box_on_floor(0.002, -1.0);
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    let machine = CpuJointMachine::new(creation.clone(), 7, initial.clone()).unwrap();
    let arrival = arrival_time(0.002, 1.0);
    let incoming = 1.0 + 9.81 * arrival;
    assert!(
        incoming > terrain.restitution_threshold,
        "this case must exceed the restitution threshold: incoming={incoming:e}"
    );
    let mut state = initial;
    let mut diagnostics = JointTickDiagnostics::default();
    diagnostics.drive_impulses.resize(machine.drives.len(), 0.0);
    let advance = |state: &mut MachineState, duration, diagnostics: &mut JointTickDiagnostics| {
        events::advance_interval(
            &creation,
            &machine.passive,
            &machine.drives,
            state,
            -DVec3::Y * 9.81,
            duration,
            JointTickSettings::default(),
            Some(&terrain),
            diagnostics,
        )
    };
    advance(&mut state, arrival, &mut diagnostics).unwrap();
    let rebound = state.velocities[1];
    assert!(
        (rebound - BLOCK_RESTITUTION * incoming).abs() <= 1e-6,
        "rebound={rebound:e} expected={:e}",
        BLOCK_RESTITUTION * incoming
    );

    // The remainder is then pure ballistic rise: the support cannot act on a box
    // that is leaving the surface.
    let remainder = TICK_SECONDS - arrival;
    advance(&mut state, remainder, &mut diagnostics).unwrap();
    let expected = rebound - 9.81 * remainder;
    assert!(
        (state.velocities[1] - expected).abs() <= 1e-6,
        "vertical={:e} expected={expected:e}",
        state.velocities[1]
    );
    assert!(clearance(&creation, &state) >= -1e-6);
}

#[test]
fn an_impact_below_the_threshold_stops_dead_for_every_substep_policy() {
    for substeps in [1, 2, 4, 8] {
        let (creation, geometry, initial) = box_on_floor(0.002, -0.5);
        let scene = scene();
        let terrain = context(&scene, &geometry, 7);
        let incoming = 0.5 + 9.81 * arrival_time(0.002, 0.5);
        assert!(
            incoming < terrain.restitution_threshold,
            "this case must stay below the restitution threshold: incoming={incoming:e}"
        );
        let mut machine = CpuJointMachine::new(creation.clone(), 7, initial).unwrap();
        for tick in 1..=4 {
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
            // Below the threshold the impact is fully inelastic, so the box never
            // moves upward and never sinks through the surface.
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
        assert!(
            state.velocities[1].abs() <= 1e-6,
            "substeps={substeps} resting velocity={:e}",
            state.velocities[1]
        );
        let clearance = clearance(&creation, state);
        assert!(
            clearance.abs() <= 1e-6,
            "substeps={substeps} resting clearance={clearance:e}"
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
    // inside one 60 Hz tick, well below the restitution threshold.
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
