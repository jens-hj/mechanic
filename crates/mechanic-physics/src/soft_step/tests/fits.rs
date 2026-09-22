//! What a jointed body runs into: whatever it was built apart from.

use super::{World, motor, spawn};
use bevy_math::{DVec3, IVec3, Vec3};
use mechanic_core::{
    BearingSpec, BuildCommand, CompiledCreation, ConstructionGraph, FaceKind, FaceRef,
    RigidLinkSpec,
};

use crate::MachineState;

// A 1 m arm hinged flat on a long post, driven round its Y axis, and a block
// on the post's top a quarter turn on, clear of the arm as built.
fn arm_on_a_post(stop: bool) -> CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let post = spawn(&mut graph, IVec3::new(0, 500, 0), [1, 1, 5]);
    let arm = spawn(&mut graph, IVec3::new(150, 600, 0), [4, 1, 1]);
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(post, FaceKind::PositiveY),
            FaceRef::part(arm, FaceKind::NegativeY),
            Vec3::new(0.0, 1.375, 0.0),
            Vec3::Y,
        )))
        .unwrap();
    if stop {
        let block = spawn(&mut graph, IVec3::new(0, 600, -200), [1, 1, 1]);
        graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: post,
                second: block,
            }))
            .unwrap();
    }
    graph.compile_with_static_parts(vec![post]).unwrap()
}

// Drives the arm for four seconds and returns how far it has turned, how
// fast it still turns, how many contacts hold it and how deep they are.
fn swung(stop: bool) -> (f64, f64, usize, f64) {
    let creation = arm_on_a_post(stop);
    let drive = motor(&creation, 200.0, 10.0);
    let row = creation.dynamics.coordinate_velocities[0];
    let state = MachineState::at_rest(&creation);
    let mut world = World::new(creation, state);
    let command = world.command(0, drive);
    world.tick_with(DVec3::ZERO, &[command]);
    for _ in 1..240 {
        world.tick(DVec3::ZERO);
    }
    let state = &world.machine.snapshot().state;
    let angle = state.coordinates[0].abs();
    let diagnostics = world.machine.diagnostics();
    (
        angle,
        state.velocities[row].abs(),
        diagnostics.contacts,
        diagnostics.maximum_penetration,
    )
}

#[test]
fn an_arm_on_a_hinge_stops_at_a_block_built_beside_its_mount() {
    let (angle, speed, contacts, _) = swung(false);
    assert!(
        speed > 5.0 && contacts == 0,
        "the free arm turns at {speed} rad/s ({angle} rad) on {contacts} contacts"
    );
    // The arm's leading edge meets the block's near face about 0.95 rad on.
    let (angle, speed, contacts, penetration) = swung(true);
    assert!(
        (0.8..1.3).contains(&angle) && speed < 0.5 && contacts > 0,
        "the arm turned {angle} rad and still turns at {speed} rad/s on {contacts} contacts {penetration} m deep"
    );
}
