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

// One sprung wheel is the smallest fixture with the car's suspension coupling.
// Unlike the direct-substep suspension test, this drives the complete tick path,
// so the event search and support activation participate.
#[test]
fn a_sprung_wheel_settles_on_the_floor_at_its_spring_equilibrium() {
    use mechanic_core::{ShockBodyEnd, ShockSpec, SpringSpec, SuspensionSpec};
    let spring = SpringSpec::default();
    let shock = ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.0, 20.0, 20.0).unwrap();
    let spec = SuspensionSpec::new(Some(spring), Some(shock), None).unwrap();
    let creation = super::super::suspension(spec, false);
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    let mut initial = MachineState::at_rest(&creation);
    for pose in &mut initial.poses {
        // The lowest collider point starts 2 mm clear, so the wheel drops,
        // impacts and then settles on its spring.
        pose.position.y += 0.502;
    }
    initial.poses =
        MachineDynamics::reconstruct_poses(&creation, &initial.poses, &initial.coordinates)
            .unwrap();
    let mass = f64::from(creation.compounds[1].mass_properties.mass);
    let equilibrium = -mass * 9.81 / f64::from(spring.rate());
    let mut machine = CpuJointMachine::new(creation.clone(), 7, initial).unwrap();
    for tick in 1..=120 {
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
        assert!(
            clearance >= -1e-6,
            "tick={tick} penetration clearance={clearance:e}"
        );
    }
    let state = &machine.snapshot().state;
    let resting = clearance(&creation, state);
    println!(
        "sprung wheel clearance={resting:e} coordinate={:e} equilibrium={equilibrium:e}",
        state.coordinates[0]
    );
    // The sprung mass hangs at mg/k once the wheel rests on the surface.
    assert!(
        (state.coordinates[0] - equilibrium).abs() <= 1e-3,
        "coordinate={:e} equilibrium={equilibrium:e}",
        state.coordinates[0]
    );
    assert!(resting.abs() <= 1e-9, "resting clearance={resting:e}");
}

// Twelve cubes rigidly linked in a row: one body, twelve separate colliders, and
// a 12 m manifold. Rigid links share one body between separated parts, so each
// cube keeps its own collider instead of merging into a single box as welding
// would. The floor is widened to hold it.
fn long_body() -> (
    CompiledCreation,
    MachineCollisionGeometry,
    TerrainContactScene,
) {
    use crate::terrain_contacts::tests::terrain as terrain_chunk;
    use mechanic_core::{BuildCommand, ConstructionGraph, RigidLinkSpec};
    use mechanic_world::TerrainMaterial;
    use std::sync::Arc;
    let mut graph = ConstructionGraph::new();
    let parts = (0..12)
        .map(|index| super::super::spawn(&mut graph, IVec3::X * (600 * index), [4, 4, 4]))
        .collect::<Vec<_>>();
    for pair in parts.windows(2) {
        graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: pair[0],
                second: pair[1],
            }))
            .unwrap();
    }
    let creation = graph.compile().unwrap();
    assert_eq!(
        creation.colliders.len(),
        12,
        "each cube must keep its own collider"
    );
    assert_eq!(
        creation.compounds.len(),
        1,
        "the linked cubes must form one rigid body"
    );
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    let mut chunk = terrain_chunk([TerrainMaterial::Rock; 2]);
    let expanded = Arc::make_mut(&mut chunk);
    for vertex in &mut expanded.vertices {
        vertex[0] *= 64.0;
        vertex[2] *= 64.0;
    }
    expanded.bounds.minimum.0 *= 64.0;
    expanded.bounds.maximum.0 *= 64.0;
    expanded.triangle_bvh.bounds = expanded.bounds;
    expanded.triangle_bvh.nodes[0].bounds = expanded.bounds;
    let mut scene = TerrainContactScene::default();
    scene.publish(1, &[chunk], &[]).unwrap();
    (creation, geometry, scene)
}

// Twelve welded cubes resting flat give one rigid body about 240 contact rows
// over six degrees of freedom: the car's failing impacts have the same shape
// (48 points, 240 rows, 16 coordinates) without its suspension or wheels.
#[test]
fn a_long_multi_collider_body_settles_on_a_redundant_manifold() {
    let (creation, geometry, scene) = long_body();
    let terrain = context(&scene, &geometry, 7);

    let mut initial = MachineState::at_rest(&creation);
    for pose in &mut initial.poses {
        pose.position.y += 0.502;
    }
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
        assert!(
            clearance >= -1e-6,
            "tick={tick} penetration clearance={clearance:e}"
        );
    }
    let diagnostics = machine.diagnostics();
    println!(
        "redundant manifold surface_points={} impact_rows={} constraint_rows={} clearance={:e}",
        diagnostics.surface_points,
        diagnostics.impact_rows_prepared,
        diagnostics.constraint_rows_prepared,
        clearance(&creation, &machine.snapshot().state)
    );
    let state = &machine.snapshot().state;
    assert!(
        state.velocities[1].abs() <= 1e-6,
        "resting velocity={:e}",
        state.velocities[1]
    );
}

// A faceted cylinder is the only contact shape the saved car has that the boxes
// above do not, and rolling one was what exposed the event-search chase: a solid
// cylinder's sixteen tangent boxes place every shared contact edge twice, rounded
// once per box, so one copy kept arriving as a new impact while the other was
// already a loaded support. The compiled cylinder now carries an exact hull and
// this rolls for a second within the policy's own depth bound.
//
// Rolling resistance and an inelastic surface only remove mechanical energy,
// whichever facet carries the load.
#[test]
fn a_rolling_wheel_only_loses_energy_on_its_facets() {
    use mechanic_core::{BuildCommand, ConstructionGraph, CylinderDimensions, CylinderSpec};
    let radius = 0.475_f64;
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
            CylinderDimensions::new(0.95, 0.0, 0.25).expect("the wheel dimensions are in range"),
            // The authored axis is local Y; a quarter turn about X lays it along Z,
            // so the wheel rolls along X.
            BuildPose::from_position_ticks(IVec3::Y * 300, GridRotation::new(1, 0, 0)),
        )))
        .unwrap();
    let creation = graph.compile().unwrap();
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);

    // Seat the faceted hull exactly on the surface: the lowest vertex of a facet
    // is nearer the axis than the authored radius.
    let mut initial = MachineState::at_rest(&creation);
    let seated = clearance(&creation, &initial);
    initial.poses[0].position.y -= seated;
    let forward = 1.0;
    initial.velocities[0] = forward;
    // Contact-point velocity vanishes when the spin matches the forward speed.
    initial.velocities[5] = -forward / radius;

    // Rolling on facets is not steady: pivoting onto each edge trades linear for
    // angular motion and back, so neither speed alone is monotone. Mechanical
    // energy is: gravity and an inelastic, resisted contact can only remove it.
    let properties = creation.compounds[0].mass_properties;
    let mass = f64::from(properties.mass);
    let inertia = properties.inertia.as_dmat3();
    let energy = |state: &MachineState| {
        let linear = DVec3::new(
            state.velocities[0],
            state.velocities[1],
            state.velocities[2],
        );
        let angular = DVec3::new(
            state.velocities[3],
            state.velocities[4],
            state.velocities[5],
        );
        let rotation = state.poses[0].rotation;
        let world = bevy_math::DMat3::from_quat(rotation)
            * inertia
            * bevy_math::DMat3::from_quat(rotation.inverse());
        0.5 * mass * linear.length_squared()
            + 0.5 * angular.dot(world * angular)
            + mass * 9.81 * state.poses[0].position.y
    };

    let mut machine = CpuJointMachine::new(creation.clone(), 7, initial.clone()).unwrap();
    let mut previous = energy(&initial);
    for tick in 1..=60 {
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
        let state = &machine.snapshot().state;
        let clearance = clearance(&creation, state);
        // A sixteen-sided hull pivots onto each new edge, so the wheel leaves the
        // surface between facets and lands again inside a tick. A linear midpoint
        // path cannot see that landing; the policy's own depth certificate is what
        // bounds it, and the next substep's contact set carries the impulse.
        assert!(
            clearance >= -terrain.maximum_depth,
            "tick={tick} penetration clearance={clearance:e}"
        );
        let current = energy(state);
        assert!(
            current <= previous + 1e-6,
            "tick={tick} energy={current:e} previous={previous:e}"
        );
        assert!(
            state.velocities[0] >= -1e-9,
            "tick={tick} the wheel must not roll backwards: {:e}",
            state.velocities[0]
        );
        previous = current;
    }
    let state = &machine.snapshot().state;
    println!(
        "rolling wheel forward={:e} spin={:e} clearance={:e} energy={:e}",
        state.velocities[0],
        state.velocities[5],
        clearance(&creation, state),
        energy(state)
    );
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

// A support that reverses inside the interval: the long body rests flat and turns
// slowly, so its far end lifts while its near end stays loaded. The search has to
// localize that release in time, and finer subdivisions of the same interval must
// reach the same state.
#[test]
fn a_reversing_support_is_localized_in_bounded_trials_and_agrees_with_finer_steps() {
    let (creation, geometry, scene) = long_body();
    let mut terrain = context(&scene, &geometry, 7);
    terrain.maximum_depth = 0.005;
    let machine =
        CpuJointMachine::new(creation.clone(), 7, MachineState::at_rest(&creation)).unwrap();
    let mut initial = MachineState::at_rest(&creation);
    for pose in &mut initial.poses {
        pose.position.y += 0.5;
    }
    initial.velocities[0] = 0.5;
    initial.velocities[5] = 0.05;

    let mut results = Vec::new();
    for steps in [1, 1, 8, 32] {
        let mut state = initial.clone();
        let mut diagnostics = JointTickDiagnostics::default();
        diagnostics
            .drive_impulses
            .resize(creation.dynamics.coordinate_bearings.len(), 0.0);
        for _ in 0..steps {
            events::advance_interval(
                &creation,
                &machine.passive,
                &machine.drives,
                &mut state,
                -DVec3::Y * 9.81,
                TICK_SECONDS / f64::from(steps),
                fixed(1),
                Some(&terrain),
                &mut diagnostics,
            )
            .unwrap();
        }
        assert_eq!(diagnostics.terrain_impact_holds, 0);
        if steps == 1 {
            assert!(
                diagnostics.release_localizations > 0,
                "this case must localize a release: {diagnostics:?}"
            );
            assert!(
                diagnostics.event_trials <= 64,
                "trials={}",
                diagnostics.event_trials
            );
        }
        let clearance = clearance(&creation, &state);
        assert!(clearance >= -1e-6, "steps={steps} clearance={clearance:e}");
        results.push(state);
    }
    assert_eq!(
        results[0], results[1],
        "the same interval must repeat exactly"
    );
    let reference = results.last().unwrap();
    for (index, state) in results[..results.len() - 1].iter().enumerate() {
        let velocity = state
            .velocities
            .iter()
            .zip(&reference.velocities)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let position = state
            .poses
            .iter()
            .zip(&reference.poses)
            .map(|(a, b)| a.position.distance(b.position))
            .fold(0.0, f64::max);
        println!("reversal steps_index={index} velocity={velocity:e} position={position:e}");
        assert!(velocity < 1e-5, "velocity error {velocity:e}");
        assert!(position < 1e-7, "position error {position:e}");
    }
}

// The long body's settled manifold is the case that exceeds the dense response
// bound: forty-eight support points, five rows each. Its response must stay
// implicit, and solving it twice must give the same impulses.
#[test]
fn a_manifold_above_the_dense_bound_solves_implicitly() {
    let (creation, geometry, scene) = long_body();
    let terrain = context(&scene, &geometry, 7);
    let mut state = MachineState::at_rest(&creation);
    for pose in &mut state.poses {
        pose.position.y += 0.5;
    }
    let mut diagnostics = JointTickDiagnostics::default();
    let query = terrain.contacts(&state, &mut diagnostics).unwrap();
    let model = MachineDynamics::assemble(&creation, &state.poses, &state.coordinates).unwrap();
    let mut incoming = state.velocities.clone();
    incoming[1] = -0.5;
    incoming[0] = 0.3;
    let contacts = query
        .impact_constraints(&model, &incoming, 1.0, 1e-7)
        .unwrap();
    let factor = crate::DynamicsFactorization::DenseReference
        .factor(
            &creation,
            &model,
            &state.coordinates,
            &vec![0.0; incoming.len()],
        )
        .unwrap();
    let solution = crate::solve_constraints(&factor, &contacts.blocks, 256, 1e-8).unwrap();
    println!(
        "long body rows={} blocks={} storage={} residual={:e}",
        solution.impulses.len(),
        contacts.blocks.len(),
        solution.response_storage,
        solution.residual
    );
    assert!(solution.impulses.len() > crate::DENSE_CONTACT_ROWS);
    assert!(solution.converged, "residual {:e}", solution.residual);
    assert_eq!(
        solution.response_storage, 0,
        "a manifold above the dense bound must stay implicit"
    );
    let repeat = crate::solve_constraints(&factor, &contacts.blocks, 256, 1e-8).unwrap();
    assert_eq!(solution.impulses, repeat.impulses);
}
