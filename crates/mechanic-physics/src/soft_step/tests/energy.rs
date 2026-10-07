//! Frictionless joints neither gain nor lose energy over long runs.

use super::*;

fn energy(creation: &CompiledCreation, state: &MachineState, gravity: DVec3) -> f64 {
    MachineKinematics::assemble(creation, &state.poses, &state.coordinates)
        .unwrap()
        .mechanical_energy(creation, &state.velocities, gravity)
        .unwrap()
}

// A 1 m steel block 5 m up with a 0.5 × 0.5 × 1.5 m arm on a bearing along its
// +X face. The arm reaches past the face along Z, its centre of mass 0.625 m
// off the axis, so it is an unbalanced load. Returns the arm's body.
fn unbalanced_rotor(anchored: bool) -> (CompiledCreation, usize) {
    let mut graph = ConstructionGraph::new();
    let root = spawn(&mut graph, IVec3::Y * 2000, [4, 4, 4]);
    let arm = spawn(&mut graph, IVec3::new(300, 2000, 400), [2, 2, 6]);
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(root, FaceKind::PositiveX),
            FaceRef::part(arm, FaceKind::NegativeX),
            Vec3::new(0.5, 5.0, 0.375),
            Vec3::X,
        )))
        .unwrap();
    let creation = graph
        .compile_with_static_parts(if anchored { vec![root] } else { vec![] })
        .unwrap();
    let arm = creation
        .loop_topology
        .body_parents
        .iter()
        .position(|parents| !parents.is_root)
        .unwrap();
    (creation, arm)
}

fn spin_rotor(creation: &CompiledCreation, speed: f64) -> MachineState {
    let mut state = MachineState::at_rest(creation);
    state.velocities[creation.dynamics.coordinate_velocities[0]] = speed;
    state
}

// Lowest and highest energy over `ticks`, measured from the start, in joules.
fn energy_swing(world: &mut World, gravity: DVec3, ticks: usize) -> [f64; 2] {
    let initial = energy(world.creation(), &world.machine.snapshot().state, gravity);
    (0..ticks).fold([0.0, 0.0], |[lost, gained], _| {
        let state = world.tick(gravity);
        let change = energy(world.creation(), &state, gravity) - initial;
        [lost.min(change), gained.max(change)]
    })
}

#[test]
fn an_unbalanced_pendulum_on_an_anchored_bearing_swings_without_gaining_energy() {
    let (creation, arm) = unbalanced_rotor(true);
    // Released horizontal, the arm falls through m g r to the bottom of its swing.
    let swing = f64::from(creation.dynamics.inertias[arm].mass)
        * mechanic_core::STANDARD_GRAVITY_M_S2
        * 0.625;
    let state = MachineState::at_rest(&creation);
    let mut world = World::new(creation, state);
    let [lost, gained] = energy_swing(&mut world, GRAVITY, 60 * 60);
    assert!(
        gained < 0.02 * swing && lost > -0.02 * swing,
        "energy moved {lost:.1}..{gained:.1} J of a {swing:.0} J swing"
    );
}

#[test]
fn an_unbalanced_rotor_on_an_anchored_bearing_keeps_its_speed() {
    let (creation, _) = unbalanced_rotor(true);
    let state = spin_rotor(&creation, 10.0);
    let spinning = energy(&creation, &state, DVec3::ZERO);
    let mut world = World::new(creation, state);
    // Gravity speeds it up and slows it down every turn, but only the
    // semi-implicit step's O(dt) wobble may remain.
    let [lost, gained] = energy_swing(&mut world, GRAVITY, 60 * 60);
    assert!(
        gained < 0.02 * spinning && lost > -0.02 * spinning,
        "energy moved {lost:.1}..{gained:.1} J of {spinning:.0} J spinning"
    );
}

#[test]
fn an_unbalanced_rotor_on_a_free_body_keeps_its_energy() {
    let (creation, _) = unbalanced_rotor(false);
    let mut state = spin_rotor(&creation, 20.0);
    // Cancel the spinning arm's momentum, so the pair whirls in place rather
    // than drifting into the floor.
    let model = MachineKinematics::assemble(&creation, &state.poses, &state.coordinates).unwrap();
    let motions = model.body_motions(&state.velocities).unwrap();
    let (momentum, mass) = motions.iter().zip(&creation.dynamics.inertias).fold(
        (DVec3::ZERO, 0.0),
        |(momentum, mass), (motion, inertia)| {
            let body = f64::from(inertia.mass);
            (momentum + motion.linear * body, mass + body)
        },
    );
    let root = creation
        .loop_topology
        .body_parents
        .iter()
        .position(|parents| parents.is_root)
        .unwrap();
    launch(&creation, &mut state, root, -momentum / mass, DVec3::ZERO);
    let initial = energy(&creation, &state, DVec3::ZERO);
    let mut world = World::new(creation, state);
    // With its inertial forces taken at the start of each substep it reached
    // the speed limit within ten seconds. The whirl still trades a few percent
    // between substeps, bounded.
    let [lost, gained] = energy_swing(&mut world, DVec3::ZERO, 60 * 30);
    assert!(
        gained < 0.05 * initial && lost > -0.05 * initial,
        "energy moved {lost:.1}..{gained:.1} J of {initial:.0} J"
    );
}

#[test]
fn a_fast_spinning_wheel_wobbling_in_flight_does_not_gain_energy() {
    let (creation, mut state) = rolling_wheel(0.0);
    state.poses[0].position.y += 3.0;
    // 150 rad/s about its axle, with a 3 rad/s wobble across it.
    state.velocities[3..6].copy_from_slice(&[3.0, 0.0, -150.0]);
    let initial = energy(&creation, &state, DVec3::ZERO);
    let mut world = World::new(creation, state);
    // It used to reach the speed limit within about fifteen ticks.
    let [_, gained] = energy_swing(&mut world, DVec3::ZERO, 120);
    assert!(
        gained < 1e-3 * initial,
        "gained {gained:.1} J of {initial:.0} J"
    );
}
