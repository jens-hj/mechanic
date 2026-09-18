use super::*;
use bevy_math::{IVec3, Vec3};
use mechanic_core::{
    BearingKind, BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec,
    FaceKind, FaceRef, GridRotation, PartId, ShockBodyEnd, ShockSpec, SpringSpec, SuspensionSpec,
};

fn spawn(graph: &mut ConstructionGraph, ticks: IVec3, dimensions: [u8; 3]) -> PartId {
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                dimensions,
                BuildPose::from_position_ticks(ticks, GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        panic!("spawn expected");
    };
    part
}

fn suspension(spec: SuspensionSpec, anchored: bool) -> CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let root = spawn(&mut graph, IVec3::ZERO, [4, 4, 4]);
    #[allow(clippy::cast_possible_truncation)]
    // Fixture construction spacing is bounded and on the authored lattice.
    let spacing = ((spec.initial_length() + 0.625) / 0.0025).round() as i32;
    let tip = spawn(&mut graph, IVec3::Y * spacing, [1, 1, 1]);
    graph
        .apply(BuildCommand::AddBearing(
            BearingSpec::new(
                FaceRef::part(root, FaceKind::PositiveY),
                FaceRef::part(tip, FaceKind::NegativeY),
                Vec3::Y * 0.5,
                Vec3::Y,
            )
            .with_kind(BearingKind::Suspension(spec)),
        ))
        .unwrap();
    graph
        .compile_with_static_parts(if anchored { vec![root] } else { vec![] })
        .unwrap()
}

pub(super) fn rotor(anchored: bool) -> CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let root = spawn(&mut graph, IVec3::ZERO, [4, 4, 4]);
    let tip = spawn(&mut graph, IVec3::X * 400, [4, 4, 4]);
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(root, FaceKind::PositiveX),
            FaceRef::part(tip, FaceKind::NegativeX),
            Vec3::X * 0.5,
            Vec3::X,
        )))
        .unwrap();
    graph
        .compile_with_static_parts(if anchored { vec![root] } else { vec![] })
        .unwrap()
}

fn fixed(substeps: u32) -> JointTickConfig {
    JointTickConfig {
        substeps,
        maximum_substeps: substeps,
        ..Default::default()
    }
}

#[test]
fn refreshed_terrain_manifolds_hold_a_free_body_without_velocity_or_pose_drift() {
    use crate::{
        TerrainContactScene,
        terrain_contacts::tests::{cube, terrain},
    };
    use mechanic_world::TerrainMaterial;
    let (creation, geometry, poses) = cube();
    let mut scene = TerrainContactScene::default();
    scene
        .publish(1, &[terrain([TerrainMaterial::Rock; 2])], &[])
        .unwrap();
    let initial = MachineState {
        poses,
        ..MachineState::at_rest(&creation)
    };
    let run = || {
        let mut state = initial.clone();
        let mut diagnostics = JointTickDiagnostics::default();
        for _ in 0..120 {
            let query = scene
                .contacts(&geometry, &state.poses, DVec3::ZERO)
                .unwrap();
            assert_eq!(query.contacts.len(), 4);
            substep(
                &creation,
                &[],
                &[],
                &mut state,
                mechanic_core::GRAVITY,
                TICK_SECONDS,
                fixed(1),
                Some(SubstepContacts {
                    query: &query,
                    restitution_threshold: 1.0,
                    stiction_threshold: 1e-7,
                }),
                None,
                &mut diagnostics,
            )
            .unwrap();
            assert!(state.poses[0].position.distance(initial.poses[0].position) < 1e-9);
            assert!(state.velocities.iter().all(|value| value.abs() < 1e-8));
        }
        assert_eq!(
            diagnostics.factorizations, 120,
            "surface and force response share one factor per substep"
        );
        state
    };
    assert_eq!(run(), run());
}

#[test]
fn coupled_surface_substeps_hold_static_load_and_apply_kinetic_deceleration() {
    use crate::{
        TerrainContactScene,
        terrain_contacts::tests::{cube, terrain},
    };
    use mechanic_world::TerrainMaterial;
    let (creation, geometry, poses) = cube();
    let mut scene = TerrainContactScene::default();
    scene
        .publish(1, &[terrain([TerrainMaterial::Rock; 2])], &[])
        .unwrap();
    for subdivisions in [1, 2, 4, 8] {
        for sliding in [false, true] {
            let mut state = MachineState {
                poses: poses.clone(),
                ..MachineState::at_rest(&creation)
            };
            state.velocities[0] = if sliding { 1.0 } else { 0.0 };
            let gravity = if sliding {
                mechanic_core::GRAVITY
            } else {
                DVec3::new(0.981, -mechanic_core::STANDARD_GRAVITY_M_S2, 0.0)
            };
            let mut coefficient = 0.0;
            for _ in 0..subdivisions {
                let query = scene
                    .contacts(&geometry, &state.poses, DVec3::ZERO)
                    .unwrap();
                coefficient = query.contacts[0].response[1];
                substep(
                    &creation,
                    &[],
                    &[],
                    &mut state,
                    gravity,
                    TICK_SECONDS / f64::from(subdivisions),
                    fixed(subdivisions),
                    Some(SubstepContacts {
                        query: &query,
                        restitution_threshold: 1.0,
                        stiction_threshold: 1e-7,
                    }),
                    None,
                    &mut JointTickDiagnostics::default(),
                )
                .unwrap();
            }
            let expected = if sliding {
                1.0 - coefficient * mechanic_core::STANDARD_GRAVITY_M_S2 * TICK_SECONDS
            } else {
                0.0
            };
            assert!(
                (state.velocities[0] - expected).abs() < 1e-8,
                "substeps={subdivisions} sliding={sliding} velocity={} expected={expected}",
                state.velocities[0]
            );
            assert!(state.velocities[1..].iter().all(|value| value.abs() < 1e-8));
        }
    }
}

#[test]
fn terrain_support_and_suspension_share_the_implicit_force_response() {
    use crate::{MachineCollisionGeometry, TerrainContactScene, terrain_contacts::tests::terrain};
    use mechanic_world::TerrainMaterial;
    let spring = SpringSpec::default();
    let shock = ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.0, 20.0, 20.0).unwrap();
    let spec = SuspensionSpec::new(Some(spring), Some(shock), None).unwrap();
    let creation = suspension(spec, false);
    let geometry = MachineCollisionGeometry::new(&creation, 1).unwrap();
    let mut scene = TerrainContactScene::default();
    scene
        .publish(1, &[terrain([TerrainMaterial::Rock; 2])], &[])
        .unwrap();
    let mut state = MachineState::at_rest(&creation);
    for pose in &mut state.poses {
        pose.position.y += 0.499;
    }
    state.poses =
        MachineDynamics::reconstruct_poses(&creation, &state.poses, &state.coordinates).unwrap();
    let initial_height = state.poses[0].position.y;
    let mass = f64::from(creation.compounds[1].mass_properties.mass);
    let expected = -mass * mechanic_core::STANDARD_GRAVITY_M_S2 / f64::from(spring.rate());
    let passive = [PassiveForce::from_kind(creation.bearings[0].kind)];
    let mut diagnostics = JointTickDiagnostics {
        drive_impulses: vec![0.0],
        ..Default::default()
    };
    for _ in 0..600 {
        let query = scene
            .contacts(&geometry, &state.poses, DVec3::ZERO)
            .unwrap();
        substep(
            &creation,
            &passive,
            &creation.coordinate_drives,
            &mut state,
            mechanic_core::GRAVITY,
            TICK_SECONDS,
            fixed(1),
            Some(SubstepContacts {
                query: &query,
                restitution_threshold: 1.0,
                stiction_threshold: 1e-7,
            }),
            None,
            &mut diagnostics,
        )
        .unwrap();
        assert!((state.poses[0].position.y - initial_height).abs() < 1e-8);
    }
    assert!(
        (state.coordinates[0] - expected).abs() < 1e-6,
        "position={} expected={expected}",
        state.coordinates[0]
    );
    assert_eq!(diagnostics.factorizations, 600);
}

fn motor(creation: &CompiledCreation, effort: f32, target: f32) -> CoordinateDrive {
    let acceleration = effort / creation.loop_topology.coordinate_axis_inertia[0];
    CoordinateDrive {
        mode: DriveMode::Speed,
        target_speed: target,
        max_speed: 100.0,
        max_acceleration: acceleration,
        source_a_max_acceleration: acceleration,
        source_a_no_load_speed: 100.0,
        ..CoordinateDrive::PASSIVE
    }
}

#[test]
fn motor_budget_accelerates_rotor_and_back_drives_the_floating_mount() {
    let creation = rotor(false);
    let mut state = MachineState::at_rest(&creation);
    let row = creation.dynamics.coordinate_velocities[0];
    let drive = motor(&creation, 120.0, 10.0);
    let model = MachineDynamics::assemble(&creation, &state.poses, &state.coordinates).unwrap();
    let impulse = drive_budget(
        drive,
        f64::from(creation.loop_topology.coordinate_axis_inertia[0]),
        0.0,
        10.0,
        TICK_SECONDS,
    );
    let mut expected = vec![0.0; state.velocities.len()];
    expected[row] = impulse;
    model
        .factor(&vec![0.0; expected.len()])
        .unwrap()
        .solve(&mut expected)
        .unwrap();
    let mut world = CpuJointMachine::new(creation.clone(), 7, state.clone()).unwrap();
    let command = DriveCommand {
        tick: 1,
        topology_generation: 7,
        coordinate: 0,
        drive,
    };
    state = world
        .step(DVec3::ZERO, fixed(1), &[], &[command])
        .unwrap()
        .state
        .clone();
    for (actual, expected) in state.velocities.iter().zip(&expected) {
        assert!((actual - expected).abs() < 1e-9);
    }
    assert!(state.velocities[3] < 0.0 && state.velocities[row] > 0.0);
    assert!((world.diagnostics().drive_impulses[0] - impulse).abs() < 1e-10);
    assert_eq!(world.diagnostics().factorizations, 1);
    assert_eq!(world.diagnostics().response_preparation_solves, 1);
    assert!(world.diagnostics().residual <= 1e-8);
}

#[test]
fn speed_torque_curve_fades_acceleration_but_keeps_braking_effort() {
    let creation = rotor(true);
    let mut initial = MachineState::at_rest(&creation);
    initial.velocities[0] = 50.0;
    for (target, fraction) in [(100.0, 0.5), (0.0, -1.0)] {
        let drive = motor(&creation, 120.0, target);
        let mut world = CpuJointMachine::new(creation.clone(), 1, initial.clone()).unwrap();
        world
            .step(
                DVec3::ZERO,
                fixed(1),
                &[],
                &[DriveCommand {
                    tick: 1,
                    topology_generation: 1,
                    coordinate: 0,
                    drive,
                }],
            )
            .unwrap();
        assert!(
            (world.diagnostics().drive_impulses[0] - fraction * 120.0 * TICK_SECONDS).abs() < 1e-6
        );
    }
}

#[test]
fn hard_stop_reacts_against_a_stalled_drive_without_consuming_its_budget() {
    let creation = rotor(true);
    let mut drive = motor(&creation, 120.0, 10.0);
    drive.max_angle = 0.1;
    let mut initial = MachineState::at_rest(&creation);
    initial.coordinates[0] = f64::from(drive.max_angle);
    let mut world = CpuJointMachine::new(creation, 1, initial.clone()).unwrap();
    world
        .step(
            DVec3::ZERO,
            fixed(1),
            &[],
            &[DriveCommand {
                tick: 1,
                topology_generation: 1,
                coordinate: 0,
                drive,
            }],
        )
        .unwrap();
    assert!((world.snapshot().state.coordinates[0] - initial.coordinates[0]).abs() < 1e-10);
    assert!(world.snapshot().state.velocities[0].abs() < 1e-8);
    assert!((world.diagnostics().drive_impulses[0] - 2.0).abs() < 1e-6);
}

#[test]
fn implicit_spring_and_asymmetric_damping_match_the_scalar_midpoint_solution() {
    let spring = SpringSpec::default();
    let shock = ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.0, 4.0, 9.0).unwrap();
    let spec = SuspensionSpec::new(Some(spring), Some(shock), None).unwrap();
    let creation = suspension(spec, true);
    let mass = f64::from(creation.compounds[1].mass_properties.mass);
    for velocity in [-1.0, 1.0] {
        let mut initial = MachineState::at_rest(&creation);
        initial.coordinates[0] = -0.04;
        initial.velocities[0] = velocity;
        let k = f64::from(spring.rate());
        let c = f64::from(shock.damping(velocity < 0.0));
        let diagonal = 0.5 * TICK_SECONDS * c + 0.25 * TICK_SECONDS.powi(2) * k;
        let expected = ((mass - diagonal) * velocity + TICK_SECONDS * k * 0.04) / (mass + diagonal);
        // Force can reverse compression within a tick; this fixture remains on its selected damping branch.
        assert_eq!(expected.is_sign_negative(), velocity.is_sign_negative());
        let mut world = CpuJointMachine::new(creation.clone(), 1, initial).unwrap();
        world.step(DVec3::ZERO, fixed(1), &[], &[]).unwrap();
        assert!((world.snapshot().state.velocities[0] - expected).abs() < 1e-8);
        assert_eq!(world.diagnostics().factorizations, 1);
        assert!(world.diagnostics().response_preparation_solves <= 2);
        assert!(world.diagnostics().residual <= 1e-8);
    }
}

#[test]
fn loaded_spring_reaches_the_authored_equilibrium_with_preload() {
    for preload in [0.0, 0.005] {
        let spring = SpringSpec::new(0.5, 0.16, 0.12, 6, preload).unwrap();
        let shock = ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.0, 20.0, 20.0).unwrap();
        let spec = SuspensionSpec::new(Some(spring), Some(shock), None).unwrap();
        let creation = suspension(spec, true);
        let expected = f64::from(spec.passive_rows()[0][1])
            - f64::from(creation.compounds[1].mass_properties.mass)
                * mechanic_core::STANDARD_GRAVITY_M_S2
                / f64::from(spring.rate());
        let initial = MachineState::at_rest(&creation);
        let mut world = CpuJointMachine::new(creation, 1, initial).unwrap();
        for _ in 0..360 {
            world
                .step(mechanic_core::GRAVITY, fixed(2), &[], &[])
                .unwrap();
        }
        let error = (world.snapshot().state.coordinates[0] - expected).abs();
        println!("loaded_spring preload={preload} equilibrium_error_m={error:e}");
        assert!(error < 1e-7);
        assert!(world.snapshot().state.velocities[0].abs() < 1e-7);
    }
}

#[test]
fn progressive_rubber_and_damping_satisfy_the_actual_midpoint_force_equation() {
    let shock = ShockSpec::default();
    let stop = mechanic_core::BumpStopSpec::default();
    let spec = SuspensionSpec::new(None, Some(shock), Some(stop)).unwrap();
    let creation = suspension(spec, true);
    let mass = f64::from(creation.compounds[1].mass_properties.mass);
    let mut initial = MachineState::at_rest(&creation);
    initial.coordinates[0] = -f64::from(spec.bump_contact().unwrap()) - 0.005;
    initial.velocities[0] = -0.5;
    let mut world = CpuJointMachine::new(creation, 1, initial.clone()).unwrap();
    world.step(DVec3::ZERO, fixed(1), &[], &[]).unwrap();
    let state = &world.snapshot().state;
    let force = PassiveForce::from_kind(BearingKind::Suspension(spec)).force(
        0.5 * (initial.coordinates[0] + state.coordinates[0]),
        0.5 * (initial.velocities[0] + state.velocities[0]),
    );
    let error = (mass * (state.velocities[0] - initial.velocities[0]) - TICK_SECONDS * force).abs();
    println!(
        "progressive_bump momentum_residual_ns={error:e} iterations={}",
        world.diagnostics().force_iterations
    );
    assert!(error < mass * 1e-7);
    assert_eq!(world.diagnostics().factorizations, 1);
    assert!(world.diagnostics().residual <= 1e-8);
}

#[test]
fn failed_tick_and_drive_change_are_rolled_back_then_can_be_retried() {
    let creation = rotor(true);
    let mut world =
        CpuJointMachine::new(creation.clone(), 1, MachineState::at_rest(&creation)).unwrap();
    let before = world.snapshot().clone();
    let mut drive = motor(&creation, 120.0, 10.0);
    drive.max_angle = 0.0;
    let command = DriveCommand {
        tick: 1,
        topology_generation: 1,
        coordinate: 0,
        drive,
    };
    let settings = JointTickConfig {
        constraint_iterations: 1,
        maximum_substeps: 8,
        ..Default::default()
    };
    assert_eq!(
        world.step(DVec3::ZERO, settings, &[], &[command]),
        Err(PhysicsError::NotConverged)
    );
    assert_eq!(world.snapshot(), &before);
    assert!(!world.diagnostics().published);
    assert_eq!(world.diagnostics().attempts, 4);
    assert_eq!(
        world
            .diagnostics()
            .attempt_failures
            .iter()
            .map(|failure| failure.substeps)
            .collect::<Vec<_>>(),
        [1, 2, 4, 8],
    );
    assert!(world.diagnostics().attempt_failures.iter().all(|failure| {
        failure.stage == JointFailureStage::FiniteStep
            && failure.error == PhysicsError::NotConverged
            && failure.accepted_seconds < TICK_SECONDS
    }));
    assert_eq!(
        world.diagnostics().failure_stage,
        Some(JointFailureStage::FiniteStep)
    );
    // The failed drive command was never committed.
    world.step(DVec3::ZERO, fixed(1), &[], &[]).unwrap();
    assert!(world.diagnostics().attempt_failures.is_empty());
    assert_eq!(world.diagnostics().failure_stage, None);
    assert!(world.snapshot().state.velocities[0].abs() < 1e-12);
    assert!(world.diagnostics().drive_impulses[0].abs() < 1e-12);
}

#[test]
fn whole_tick_impulses_are_not_reapplied_at_each_substep() {
    let creation = rotor(false);
    let initial = MachineState::at_rest(&creation);
    let mass = creation
        .dynamics
        .inertias
        .iter()
        .map(|i| f64::from(i.mass))
        .sum::<f64>();
    for substeps in [1, 2, 4, 8] {
        let mut world = CpuJointMachine::new(creation.clone(), 1, initial.clone()).unwrap();
        let impulse = ExternalImpulse {
            tick: 1,
            topology_generation: 1,
            body: 1,
            point: initial.poses[1].position,
            impulse: DVec3::X * 10.0,
        };
        world
            .step(DVec3::ZERO, fixed(substeps), &[impulse], &[])
            .unwrap();
        assert!((world.snapshot().state.velocities[0] - 10.0 / mass).abs() < 1e-12);
        assert_eq!(world.diagnostics().factorizations, substeps as usize + 1);
    }
}

#[test]
fn nonlinear_retry_restarts_the_same_tick_without_duplicating_impulses() {
    let spec = SuspensionSpec::new(
        None,
        Some(ShockSpec::default()),
        Some(mechanic_core::BumpStopSpec::default()),
    )
    .unwrap();
    let creation = suspension(spec, true);
    let mut initial = MachineState::at_rest(&creation);
    initial.coordinates[0] = -f64::from(spec.bump_contact().unwrap()) - 0.005;
    initial.velocities[0] = -0.5;
    let impulse = ExternalImpulse {
        tick: 1,
        topology_generation: 1,
        body: 1,
        point: initial.poses[1].position,
        impulse: DVec3::Y * -20.0,
    };
    let settings = JointTickConfig {
        force_iterations: 8,
        ..Default::default()
    };
    let mut automatic = CpuJointMachine::new(creation.clone(), 1, initial.clone()).unwrap();
    automatic
        .step(DVec3::ZERO, settings, &[impulse], &[])
        .unwrap();
    let accepted = automatic.diagnostics().substeps;
    println!(
        "nonlinear_retry attempts={} accepted_substeps={accepted}",
        automatic.diagnostics().attempts
    );
    assert!(accepted > 1);
    assert_eq!(automatic.diagnostics().failure_stage, None);
    assert_eq!(
        automatic.diagnostics().attempt_failures.len(),
        automatic.diagnostics().attempts as usize - 1
    );
    assert!(
        automatic
            .diagnostics()
            .attempt_failures
            .iter()
            .all(|failure| {
                failure.stage == JointFailureStage::FiniteStep && failure.substeps < accepted
            })
    );
    let mut direct = CpuJointMachine::new(creation, 1, initial).unwrap();
    direct
        .step(
            DVec3::ZERO,
            JointTickConfig {
                substeps: accepted,
                maximum_substeps: accepted,
                ..settings
            },
            &[impulse],
            &[],
        )
        .unwrap();
    assert_eq!(automatic.snapshot(), direct.snapshot());
    assert!(automatic.diagnostics().factorizations > direct.diagnostics().factorizations);
}

#[test]
fn timed_travel_stop_preserves_floating_machine_momentum_without_rebound() {
    let shock = ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.0, 0.0, 0.0).unwrap();
    let spec = SuspensionSpec::new(None, Some(shock), None).unwrap();
    let creation = suspension(spec, false);
    let mut initial = MachineState::at_rest(&creation);
    initial.coordinates[0] = -0.001;
    let row = creation.dynamics.coordinate_velocities[0];
    initial.velocities[row] = 10.0;
    let momentum = |state: &MachineState| {
        let model = MachineDynamics::assemble(&creation, &state.poses, &state.coordinates).unwrap();
        model
            .body_motions(&state.velocities)
            .unwrap()
            .iter()
            .zip(&creation.dynamics.inertias)
            .map(|(motion, inertia)| motion.linear * f64::from(inertia.mass))
            .sum::<DVec3>()
    };
    let expected = momentum(&initial);
    let mut world = CpuJointMachine::new(creation.clone(), 1, initial).unwrap();
    for _ in 0..10 {
        world.step(DVec3::ZERO, fixed(1), &[], &[]).unwrap();
        assert!(world.snapshot().state.coordinates[0].abs() < 1e-10);
        assert!((momentum(&world.snapshot().state) - expected).length() < 1e-8);
    }
    assert!(world.snapshot().state.velocities[row].abs() < 1e-8);
    assert!(world.snapshot().state.velocities[1] > 0.0);
}

#[test]
fn unlimited_drive_remains_unbounded_and_can_be_disabled_at_no_load() {
    let creation = rotor(true);
    let mut drive = motor(&creation, 120.0, 1.0);
    drive.max_acceleration = f32::INFINITY;
    drive.source_a_max_acceleration = f32::INFINITY;
    let mut world =
        CpuJointMachine::new(creation.clone(), 1, MachineState::at_rest(&creation)).unwrap();
    world
        .step(
            DVec3::ZERO,
            fixed(1),
            &[],
            &[DriveCommand {
                tick: 1,
                topology_generation: 1,
                coordinate: 0,
                drive,
            }],
        )
        .unwrap();
    assert!((world.snapshot().state.velocities[0] - 1.0).abs() < 1e-10);
    drive.source_a_no_load_speed = 0.0;
    assert!(drive_budget(drive, 1.0, 0.0, 1.0, TICK_SECONDS).abs() < 1e-12);
}

#[test]
fn undamped_spring_oscillation_converges_without_artificial_decay() {
    let spring = SpringSpec::default();
    let spec = SuspensionSpec::new(Some(spring), None, None).unwrap();
    let creation = suspension(spec, true);
    let mass = f64::from(creation.compounds[1].mass_properties.mass);
    let omega = (f64::from(spring.rate()) / mass).sqrt();
    let equilibrium = -mass * mechanic_core::STANDARD_GRAVITY_M_S2 / f64::from(spring.rate());
    let amplitude = 0.004;
    let mut errors = Vec::new();
    for substeps in [1, 2, 4, 8] {
        let mut initial = MachineState::at_rest(&creation);
        initial.coordinates[0] = equilibrium + amplitude;
        let mut world = CpuJointMachine::new(creation.clone(), 1, initial).unwrap();
        let mut maximum = 0.0_f64;
        for tick in 1..=60 {
            world
                .step(mechanic_core::GRAVITY, fixed(substeps), &[], &[])
                .unwrap();
            let time = f64::from(tick) * TICK_SECONDS;
            let position = equilibrium + amplitude * (omega * time).cos();
            let speed = -amplitude * omega * (omega * time).sin();
            let error = ((world.snapshot().state.coordinates[0] - position) / amplitude)
                .hypot((world.snapshot().state.velocities[0] - speed) / (amplitude * omega));
            maximum = maximum.max(error);
        }
        errors.push(maximum);
        let energy_ratio = ((world.snapshot().state.coordinates[0] - equilibrium) / amplitude)
            .powi(2)
            + (world.snapshot().state.velocities[0] / (amplitude * omega)).powi(2);
        assert!(
            (energy_ratio - 1.0).abs() < 1e-8,
            "undamped spring lost energy at {substeps} substeps: {energy_ratio}"
        );
    }
    println!("undamped_spring_phase_space_error_1_2_4_8={errors:?}");
    assert!(
        errors[3] < 0.01,
        "eight-substep spring motion must be within 1% of the analytic phase-space amplitude"
    );
}

#[test]
fn nonlinear_midpoint_rotation_converges_to_the_rk4_free_motion_reference() {
    let mut graph = ConstructionGraph::new();
    spawn(&mut graph, IVec3::ZERO, [4, 2, 1]);
    let creation = graph.compile().unwrap();
    let mut initial = MachineState::at_rest(&creation);
    initial.velocities[3..6].copy_from_slice(&[3.0, -2.0, 4.0]);
    let mut reference = crate::CpuFreeMotion::new(creation.clone(), 1, initial.clone()).unwrap();
    for _ in 0..120 {
        reference.step(DVec3::ZERO, 8, &[]).unwrap();
    }
    let mut errors = Vec::new();
    for substeps in [1, 2, 4, 8] {
        let mut world = CpuJointMachine::new(creation.clone(), 1, initial.clone()).unwrap();
        for _ in 0..120 {
            world.step(DVec3::ZERO, fixed(substeps), &[], &[]).unwrap();
        }
        let q = world.snapshot().state.poses[0].rotation
            * reference.snapshot().state.poses[0].rotation.inverse();
        let angular_error = q.to_scaled_axis().length();
        let velocity_error = world
            .snapshot()
            .state
            .velocities
            .iter()
            .zip(&reference.snapshot().state.velocities)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
            / 29.0_f64.sqrt();
        errors.push(angular_error.max(velocity_error));
    }
    println!("midpoint_free_rotation_reference_errors_1_2_4_8={errors:?}");
    assert!(errors[3] < 0.001);
    for pair in errors.windows(2) {
        assert!(pair[1] < pair[0] / 3.5);
    }
}

mod terrain_ticks;

#[test]
fn a_joint_limit_activates_at_arrival_and_then_advances_with_coupled_velocity() {
    let shock = ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.0, 0.0, 0.0).unwrap();
    let spec = SuspensionSpec::new(None, Some(shock), None).unwrap();
    let creation = suspension(spec, false);
    let mut initial = MachineState::at_rest(&creation);
    initial.coordinates[0] = -0.001;
    let row = creation.dynamics.coordinate_velocities[0];
    initial.velocities[row] = 10.0;
    let world = CpuJointMachine::new(creation, 1, initial).unwrap();
    let initial = world.snapshot().state.clone();
    let model =
        MachineDynamics::assemble(&world.creation, &initial.poses, &initial.coordinates).unwrap();
    let momentum = model
        .body_motions(&initial.velocities)
        .unwrap()
        .iter()
        .zip(&world.creation.dynamics.inertias)
        .map(|(motion, inertia)| motion.linear * f64::from(inertia.mass))
        .sum::<DVec3>();
    let total_mass = world
        .creation
        .dynamics
        .inertias
        .iter()
        .map(|i| f64::from(i.mass))
        .sum::<f64>();
    let outgoing = momentum / total_mass;
    let impact_time = 0.001 / 10.0;
    for duration in [impact_time * 0.5, impact_time, TICK_SECONDS] {
        let mut state = initial.clone();
        let mut diagnostics = JointTickDiagnostics {
            drive_impulses: vec![0.0; world.drives.len()],
            ..Default::default()
        };
        events::advance_interval(
            &world.creation,
            &world.passive,
            &world.drives,
            &mut state,
            DVec3::ZERO,
            duration,
            fixed(1),
            None,
            &mut diagnostics,
        )
        .unwrap();
        let expected_q = -0.001 + 10.0 * duration.min(impact_time);
        assert!((state.coordinates[0] - expected_q).abs() < 1e-10);
        let expected_rate = if duration < impact_time { 10.0 } else { 0.0 };
        assert!((state.velocities[row] - expected_rate).abs() < 1e-8);
        let expected_root =
            initial.poses[0].position + outgoing * (duration - impact_time).max(0.0);
        assert!(state.poses[0].position.distance(expected_root) < 1e-10);
        assert_eq!(diagnostics.position_factor_solves, 0);
        assert_eq!(
            diagnostics.impact_events,
            usize::from(duration >= impact_time)
        );
    }
}
