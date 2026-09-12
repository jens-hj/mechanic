use super::*;
use crate::{
    MachineCollisionGeometry, TerrainContactScene,
    terrain_contacts::tests::{cube, terrain},
};
use mechanic_world::TerrainMaterial;

fn scene() -> TerrainContactScene {
    let mut scene = TerrainContactScene::default();
    scene
        .publish(1, &[terrain([TerrainMaterial::Rock; 2])], &[])
        .unwrap();
    scene
}

fn context<'a>(
    scene: &'a TerrainContactScene,
    geometry: &'a MachineCollisionGeometry,
    generation: u64,
) -> TerrainSubstep<'a> {
    TerrainSubstep {
        integration: TerrainIntegration::EventResolved,
        scene,
        geometry,
        topology_generation: generation,
        origin: DVec3::ZERO,
        maximum_depth: 0.002,
        maximum_evaluations: 128,
        maximum_event_trials: 128,
        restitution_threshold: 1.0,
        stiction_threshold: 1e-7,
    }
}

#[test]
fn supported_ticks_validate_motion_before_publication_and_repeat_at_each_substep_policy() {
    let (creation, geometry, poses) = cube();
    let initial = MachineState {
        poses,
        ..MachineState::at_rest(&creation)
    };
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    for subdivisions in [1, 2, 4, 8] {
        let run = || {
            let mut world = CpuJointMachine::new(creation.clone(), 7, initial.clone()).unwrap();
            for tick in 1..=120 {
                world
                    .step_candidate(
                        -DVec3::Y * 9.81,
                        fixed(subdivisions),
                        &[],
                        &[],
                        Some(&terrain),
                    )
                    .unwrap();
                assert_eq!(world.snapshot().tick, tick);
                assert!(
                    world.snapshot().state.poses[0]
                        .position
                        .distance(initial.poses[0].position)
                        < 1e-9
                );
                assert!(
                    world
                        .snapshot()
                        .state
                        .velocities
                        .iter()
                        .all(|velocity| velocity.abs() < 1e-8)
                );
                let diagnostics = world.diagnostics();
                assert!(diagnostics.published);
                assert_eq!(diagnostics.terrain_contact_queries, subdivisions as usize);
                assert_eq!(diagnostics.terrain_path_queries, subdivisions as usize);
                assert_eq!(diagnostics.terrain_path_rejections, 0);
                assert_eq!(diagnostics.factorizations, subdivisions as usize);
                assert_eq!(diagnostics.terrain_envelopes, 2 * subdivisions as usize);
            }
            world.snapshot().state_hash()
        };
        assert_eq!(run(), run());
    }
}

#[test]
fn failed_terrain_path_retries_match_direct_quality_and_consume_impulse_once() {
    let (creation, geometry, poses) = cube();
    let initial = MachineState {
        poses,
        ..MachineState::at_rest(&creation)
    };
    let scene = scene();
    let mut terrain = context(&scene, &geometry, 7);
    terrain.maximum_evaluations = 1;
    let impulse = ExternalImpulse {
        tick: 1,
        topology_generation: 7,
        body: 0,
        point: initial.poses[0].position,
        impulse: DVec3::X * (f64::from(creation.dynamics.inertias[0].mass) * 0.5),
    };
    let mut retried = CpuJointMachine::new(creation.clone(), 7, initial.clone()).unwrap();
    retried
        .step_candidate(
            DVec3::ZERO,
            JointTickSettings::default(),
            &[impulse],
            &[],
            Some(&terrain),
        )
        .unwrap();
    assert_eq!(retried.diagnostics().attempts, 4);
    assert_eq!(retried.diagnostics().substeps, 8);
    assert_eq!(retried.diagnostics().terrain_path_rejections, 3);
    assert_eq!(retried.diagnostics().terrain_path_queries, 11);
    assert!((retried.snapshot().state.velocities[0] - 0.5).abs() < 1e-12);
    assert!((retried.snapshot().state.poses[0].position.x - TICK_SECONDS * 0.5).abs() < 1e-12);
    let mut direct = CpuJointMachine::new(creation, 7, initial).unwrap();
    direct
        .step_candidate(DVec3::ZERO, fixed(8), &[impulse], &[], Some(&terrain))
        .unwrap();
    assert_eq!(
        retried.snapshot().state_hash(),
        direct.snapshot().state_hash()
    );
    assert_eq!(retried.snapshot(), direct.snapshot());
}

#[test]
fn final_terrain_failure_retains_completed_state_and_does_not_consume_a_command() {
    let (creation, geometry, poses) = cube();
    let initial = MachineState {
        poses,
        ..MachineState::at_rest(&creation)
    };
    let scene = scene();
    let mut terrain = context(&scene, &geometry, 7);
    terrain.maximum_evaluations = 1;
    let impulse = ExternalImpulse {
        tick: 1,
        topology_generation: 7,
        body: 0,
        point: initial.poses[0].position,
        impulse: DVec3::X * (f64::from(creation.dynamics.inertias[0].mass) * 0.5),
    };
    let mut world = CpuJointMachine::new(creation.clone(), 7, initial.clone()).unwrap();
    let before = world.snapshot().clone();
    let settings = JointTickSettings {
        maximum_substeps: 4,
        ..Default::default()
    };
    assert_eq!(
        world.step_candidate(DVec3::ZERO, settings, &[impulse], &[], Some(&terrain)),
        Err(PhysicsError::NotConverged)
    );
    assert_eq!(world.snapshot(), &before);
    assert!(!world.diagnostics().published);
    assert_eq!(world.diagnostics().attempts, 3);
    assert_eq!(world.diagnostics().terrain_path_rejections, 3);
    world
        .step_candidate(DVec3::ZERO, fixed(8), &[impulse], &[], Some(&terrain))
        .unwrap();
    let mut direct = CpuJointMachine::new(creation, 7, initial).unwrap();
    direct
        .step_candidate(DVec3::ZERO, fixed(8), &[impulse], &[], Some(&terrain))
        .unwrap();
    assert_eq!(world.snapshot(), direct.snapshot());
}

#[test]
fn split_joint_recovery_checks_terrain_and_restores_physical_velocity_on_failure() {
    let creation = rotor(true);
    let geometry = MachineCollisionGeometry::new(&creation, 1).unwrap();
    let scene = scene();
    let terrain = context(&scene, &geometry, 1);
    let mut state = MachineState::at_rest(&creation);
    state.coordinates[0] = 1e-3;
    state.velocities[0] = 0.1;
    let mut drives = creation.coordinate_drives.clone();
    drives[0].min_angle = 0.0;
    drives[0].max_angle = 0.0;
    let model = MachineDynamics::assemble(&creation, &state.poses, &state.coordinates).unwrap();
    let factor = model.factor(&vec![0.0; state.velocities.len()]).unwrap();
    let before = state.clone();
    let mut diagnostics = JointTickDiagnostics::default();
    assert_eq!(
        correct_joint_positions(
            &creation,
            &drives,
            &mut state,
            &factor,
            fixed(1),
            Some(&terrain),
            &mut diagnostics
        ),
        Err(PhysicsError::NotConverged)
    );
    assert_eq!(state, before);
    assert_eq!(diagnostics.terrain_correction_paths, 1);
    assert_eq!(diagnostics.terrain_path_rejections, 1);
    assert!(diagnostics.position_factor_solves > 0);
    let empty = TerrainContactScene::default();
    let clear = context(&empty, &geometry, 1);
    correct_joint_positions(
        &creation,
        &drives,
        &mut state,
        &factor,
        fixed(1),
        Some(&clear),
        &mut diagnostics,
    )
    .unwrap();
    assert!(state.coordinates[0].abs() < 1e-10);
    assert_eq!(state.velocities, before.velocities);
    assert_eq!(diagnostics.terrain_correction_paths, 2);
}

#[test]
fn terrain_failure_does_not_commit_a_drive_change() {
    let creation = rotor(true);
    let geometry = MachineCollisionGeometry::new(&creation, 1).unwrap();
    let scene = scene();
    let terrain = context(&scene, &geometry, 1);
    let state = MachineState::at_rest(&creation);
    let drive = motor(&creation, 10.0, 1.0);
    let command = DriveCommand {
        tick: 1,
        topology_generation: 1,
        coordinate: 0,
        drive,
    };
    let mut world = CpuJointMachine::new(creation, 1, state).unwrap();
    let before = world.snapshot().clone();
    assert_eq!(
        world.step_candidate(
            DVec3::ZERO,
            JointTickSettings::default(),
            &[],
            &[command],
            Some(&terrain)
        ),
        Err(PhysicsError::NotConverged)
    );
    assert_eq!(world.snapshot(), &before);
    assert!(!world.diagnostics().published);
    assert!(world.diagnostics().terrain_path_rejections > 0);
    world.step(DVec3::ZERO, fixed(1), &[], &[]).unwrap();
    assert!(world.snapshot().state.velocities[0].abs() < 1e-12);
    assert!(world.diagnostics().drive_impulses[0].abs() < 1e-12);
}

#[test]
fn changed_supporting_terrain_or_wrong_topology_cannot_publish_an_invalid_tick() {
    let (creation, geometry, poses) = cube();
    let initial = MachineState {
        poses,
        ..MachineState::at_rest(&creation)
    };
    let mut scene = scene();
    let mut world = CpuJointMachine::new(creation, 7, initial).unwrap();
    world
        .step_candidate(
            -DVec3::Y * 9.81,
            fixed(1),
            &[],
            &[],
            Some(&context(&scene, &geometry, 7)),
        )
        .unwrap();
    let before = world.snapshot().clone();
    assert_eq!(
        world.step_candidate(
            DVec3::ZERO,
            fixed(1),
            &[],
            &[],
            Some(&context(&scene, &geometry, 8))
        ),
        Err(PhysicsError::InvalidCollision)
    );
    assert_eq!(world.snapshot(), &before);
    assert_eq!(world.diagnostics().terrain_path_queries, 0);
    let mut chunk = terrain([TerrainMaterial::Rock; 2]);
    let changed = std::sync::Arc::make_mut(&mut chunk);
    changed.generation += 1;
    for vertex in &mut changed.vertices {
        vertex[1] += 0.01;
    }
    changed.bounds.minimum.0.y = f64::from(changed.vertices[0][1]);
    changed.bounds.maximum.0.y = changed.bounds.minimum.0.y;
    changed.triangle_bvh.bounds = changed.bounds;
    changed.triangle_bvh.nodes[0].bounds = changed.bounds;
    scene.publish(2, &[chunk], &[]).unwrap();
    assert_eq!(
        world.step_candidate(
            -DVec3::Y * 9.81,
            JointTickSettings::default(),
            &[],
            &[],
            Some(&context(&scene, &geometry, 7))
        ),
        Err(PhysicsError::NotConverged)
    );
    assert_eq!(world.snapshot(), &before);
    assert!(!world.diagnostics().published);
    assert_eq!(world.diagnostics().terrain_path_rejections, 4);
}

mod first_impacts;

mod saved_car;

mod rolling_edge;

mod recovery;
