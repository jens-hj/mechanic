use super::*;
use crate::terrain_contacts::tests::{loose_cubes, pose};

fn tick(
    world: &mut CpuJointMachine,
    gravity: DVec3,
    terrain: &TerrainSubstep<'_>,
    tick: u64,
) -> MachineState {
    let outcome = world
        .step_candidate(
            gravity,
            JointTickSettings::default(),
            &[],
            &[],
            Some(terrain),
        )
        .map(|snapshot| snapshot.state.clone());
    outcome.unwrap_or_else(|error| {
        panic!(
            "tick {tick}: {error:?} at {:?}",
            world.diagnostics().failure_stage
        )
    })
}

#[test]
fn a_box_dropped_on_a_resting_box_comes_to_rest_on_top_of_it() {
    let creation = loose_cubes();
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    let initial = MachineState {
        poses: vec![pose(DVec3::Y * 0.5), pose(DVec3::Y * 1.52)],
        ..MachineState::at_rest(&creation)
    };
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    let mut world = CpuJointMachine::new(creation, 7, initial).unwrap();
    let mut state = world.snapshot().state.clone();
    for tick_index in 1..=120 {
        state = tick(&mut world, mechanic_core::GRAVITY, &terrain, tick_index);
        let [lower, upper] = [state.poses[0].position, state.poses[1].position];
        assert!(
            upper.y - lower.y >= 1.0 - terrain.maximum_depth,
            "tick {tick_index}: the upper box sank into the lower one, {} apart",
            upper.y - lower.y
        );
        assert!(lower.y >= 0.5 - terrain.maximum_depth);
    }
    assert!((state.poses[0].position.y - 0.5).abs() < 1e-6);
    assert!((state.poses[1].position.y - 1.5).abs() < 1e-6);
    assert!(
        state
            .velocities
            .iter()
            .all(|velocity| velocity.abs() < 1e-6),
        "{:?}",
        state.velocities
    );
}

#[test]
fn a_sliding_box_pushes_a_floating_box_without_passing_through_it() {
    let creation = loose_cubes();
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    let mut initial = MachineState {
        poses: vec![
            pose(DVec3::new(0.0, 5.0, 0.0)),
            pose(DVec3::new(1.2, 5.0, 0.0)),
        ],
        ..MachineState::at_rest(&creation)
    };
    let x = |body: usize| creation.dynamics.body_velocities[body].start;
    initial.velocities[x(0)] = 1.0;
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    let mut world = CpuJointMachine::new(creation.clone(), 7, initial).unwrap();
    let mut state = world.snapshot().state.clone();
    for tick_index in 1..=60 {
        state = tick(&mut world, DVec3::ZERO, &terrain, tick_index);
        let gap = state.poses[1].position.x - state.poses[0].position.x;
        assert!(
            gap >= 1.0 - terrain.maximum_depth,
            "tick {tick_index}: boxes overlap, centres {gap} apart"
        );
        // Equal masses: the contact impulses are internal, so momentum stays put.
        let momentum = state.velocities[x(0)] + state.velocities[x(1)];
        assert!(
            (momentum - 1.0).abs() < 1e-9,
            "tick {tick_index}: {momentum}"
        );
    }
    assert!(state.velocities[x(1)] >= state.velocities[x(0)] - 1e-9);
    assert!(state.velocities[x(1)] > 0.4, "{:?}", state.velocities);
}

#[test]
fn boxes_stacked_a_hair_apart_keep_ticking_while_they_glide_together() {
    let creation = loose_cubes();
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    // Just outside the pair contact window, so the pair is swept rather than
    // excluded as an initial support, and moving fast together.
    let mut initial = MachineState {
        poses: vec![
            pose(DVec3::new(0.0, 5.0, 0.0)),
            pose(DVec3::new(0.0, 6.0 + 2e-6, 0.0)),
        ],
        ..MachineState::at_rest(&creation)
    };
    for body in 0..2 {
        let rows = creation.dynamics.body_velocities[body].start;
        initial.velocities[rows] = 2.0;
        initial.velocities[rows + 1] = -0.5;
    }
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    let mut world = CpuJointMachine::new(creation, 7, initial).unwrap();
    for tick_index in 1..=10 {
        let state = tick(&mut world, DVec3::ZERO, &terrain, tick_index);
        let gap = state.poses[1].position.y - state.poses[0].position.y;
        assert!(
            (gap - (1.0 + 2e-6)).abs() < 1e-9,
            "tick {tick_index}: gap {gap}"
        );
    }
}

/// Drops a construction captured from an app world save onto flat rock and runs
/// it under the app route's depth policy, failing on any refused tick. Returns
/// the settled state and each body's lowest collision point.
fn settle_captured(instance: &str, ticks: u64) -> (MachineState, Vec<f64>, f64) {
    let instance: mechanic_world::WorldCreationInstanceDoc = ron::from_str(instance).unwrap();
    let loaded = instance.creation.into_graph().unwrap();
    let creation = loaded
        .graph
        .compile_with_suspension_sockets([], &loaded.sockets)
        .unwrap();
    let shapes = collision_shapes(&creation);
    let lowest_points = |state: &MachineState| {
        let mut lowest = vec![f64::INFINITY; state.poses.len()];
        for (body, shape) in &shapes {
            let pose = state.poses[*body];
            let bottom = shape
                .transformed(pose.position, pose.rotation)
                .unwrap()
                .bounds()[0]
                .y;
            lowest[*body] = lowest[*body].min(bottom);
        }
        lowest
    };
    let mut initial = MachineState::at_rest(&creation);
    let lowest = lowest_points(&initial)
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    for pose in &mut initial.poses {
        pose.position.y -= lowest + 0.001;
    }
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    let mut chunk = terrain([TerrainMaterial::Rock; 2]);
    let expanded = std::sync::Arc::make_mut(&mut chunk);
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
    let terrain = TerrainSubstep {
        maximum_depth: 0.005,
        ..context(&scene, &geometry, 7)
    };
    let mut world = CpuJointMachine::new(creation, 7, initial).unwrap();
    let mut state = world.snapshot().state.clone();
    for tick_index in 1..=ticks {
        state = tick(&mut world, mechanic_core::GRAVITY, &terrain, tick_index);
    }
    let lowest = lowest_points(&state);
    (state, lowest, terrain.maximum_depth)
}

/// Captured from the app: a two-block piece dropped half over the edge of a
/// beam, and a three-block bar landing on it, froze the CPU route three ways:
/// the bar resting on the piece stalled the sweep while both still moved; the
/// tilted piece rested on one crossed-edge point that flipped every tick; and
/// the piece rocking on the edge chased micron lifts through every event trial.
#[test]
fn blocks_dropped_across_a_ledge_settle_instead_of_stalling_the_tick() {
    let (state, lowest, depth) =
        settle_captured(include_str!("fixtures/ledge_blocks_instance.ron"), 240);
    let ground = lowest[0];
    assert!(ground.abs() < depth, "{lowest:?}");
    // The bar lies on the beam's top face, a block above the ground.
    assert!((lowest[1] - (ground + 0.25)).abs() < depth, "{lowest:?}");
    assert!(
        lowest.iter().all(|&bottom| bottom > ground - depth),
        "{lowest:?}"
    );
    assert!(
        state
            .velocities
            .iter()
            .all(|velocity| velocity.abs() < 1e-6),
        "{:?}",
        state.velocities
    );
}

/// Captured from the app: the same two pieces dropped on a short block froze
/// just after landing. The bar rolled over the block's edge a fraction of a
/// micron away, far slower than any sweep bound admits, and then its contact
/// released and relanded at ever-shorter prefixes.
#[test]
fn blocks_landing_on_each_other_keep_ticking_until_they_rest() {
    let (state, lowest, depth) =
        settle_captured(include_str!("fixtures/leaning_blocks_instance.ron"), 600);
    let ground = lowest[0];
    assert!(ground.abs() < depth, "{lowest:?}");
    assert!((lowest[1] - (ground + 0.25)).abs() < depth, "{lowest:?}");
    assert!(
        lowest.iter().all(|&bottom| bottom > ground - depth),
        "{lowest:?}"
    );
    assert!(
        state
            .velocities
            .iter()
            .all(|velocity| velocity.abs() < 1e-6),
        "{:?}",
        state.velocities
    );
}
