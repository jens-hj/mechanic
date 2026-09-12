//! Actual authored car in the private terrain-tick experiment. Not a world gate.

use super::*;
use mechanic_core::ContactPolytope;

fn fixture() -> (
    CompiledCreation,
    MachineState,
    MachineCollisionGeometry,
    TerrainContactScene,
) {
    let instance: mechanic_world::WorldCreationInstanceDoc = ron::from_str(include_str!(
        "../../../../../mechanic-bench/tests/fixtures/driven_car_instance.ron"
    ))
    .unwrap();
    let loaded = instance.creation.into_graph().unwrap();
    let creation = loaded
        .graph
        .compile_with_suspension_sockets([], &loaded.sockets)
        .unwrap();
    let mut initial = MachineState::at_rest(&creation);
    let mut lowest = f64::INFINITY;
    for collider in &creation.colliders {
        let pose = initial.poses[collider.compound_index as usize];
        lowest = lowest.min(
            ContactPolytope::from_collider(collider)
                .unwrap()
                .transformed(pose.position, pose.rotation)
                .unwrap()
                .bounds()[0]
                .y,
        );
    }
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
    (creation, initial, geometry, scene)
}

#[test]
fn saved_car_first_supported_tick_repeats_with_authored_drives_and_suspension() {
    let (creation, initial, geometry, scene) = fixture();
    let terrain = context(&scene, &geometry, 7);
    for factorization in [
        crate::DynamicsFactorization::DenseReference,
        crate::DynamicsFactorization::Articulated,
    ] {
        for substeps in [1, 2, 4, 8] {
            let run = || {
                let mut world = CpuJointMachine::new(creation.clone(), 7, initial.clone()).unwrap();
                world
                    .step_candidate(
                        -DVec3::Y * 9.81,
                        JointTickSettings {
                            factorization,
                            ..fixed(substeps)
                        },
                        &[],
                        &[],
                        Some(&terrain),
                    )
                    .unwrap();
                assert_eq!(world.snapshot().tick, 1);
                assert!(
                    world
                        .diagnostics()
                        .drive_impulses
                        .iter()
                        .any(|value| value.abs() > 1e-3)
                );
                for collider in &creation.colliders {
                    let pose = world.snapshot().state.poses[collider.compound_index as usize];
                    let bounds = ContactPolytope::from_collider(collider)
                        .unwrap()
                        .transformed(pose.position, pose.rotation)
                        .unwrap()
                        .bounds();
                    // The entire projected collider lies within this known finite floor.
                    assert!(bounds[0].x > -64.0 && bounds[1].x < 64.0);
                    assert!(bounds[0].z > -64.0 && bounds[1].z < 64.0);
                    assert!(bounds[0].y >= -0.002);
                }
                println!(
                    "saved_car_support factor={factorization:?} substeps={substeps} hash={:016x} diagnostics={:?}",
                    world.snapshot().state_hash(),
                    world.diagnostics()
                );
                world.snapshot().clone()
            };
            assert_eq!(run(), run());
        }
    }
}

#[test]
fn saved_car_sustained_support_repeats_at_each_substep_policy() {
    let (creation, initial, geometry, scene) = fixture();
    let terrain = context(&scene, &geometry, 7);
    let shapes = creation
        .colliders
        .iter()
        .map(|collider| {
            (
                collider.compound_index as usize,
                ContactPolytope::from_collider(collider).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    for substeps in [1, 2, 4, 8] {
        let run = || {
            let mut world = CpuJointMachine::new(creation.clone(), 7, initial.clone()).unwrap();
            let mut hashes = Vec::with_capacity(120);
            let mut maximum_depth = 0.0_f64;
            for _ in 0..120 {
                world
                    .step_candidate(-DVec3::Y * 9.81, fixed(substeps), &[], &[], Some(&terrain))
                    .unwrap();
                assert!(world.diagnostics().published);
                assert_eq!(world.diagnostics().terrain_impact_holds, 0);
                for (body, shape) in &shapes {
                    let pose = world.snapshot().state.poses[*body];
                    let bounds = shape
                        .transformed(pose.position, pose.rotation)
                        .unwrap()
                        .bounds();
                    assert!(bounds[0].x > -64.0 && bounds[1].x < 64.0);
                    assert!(bounds[0].z > -64.0 && bounds[1].z < 64.0);
                    maximum_depth = maximum_depth.max(-bounds[0].y);
                }
                hashes.push(world.snapshot().state_hash());
            }
            assert!(maximum_depth <= 0.002);
            println!(
                "saved_car_sustained substeps={substeps} completed_ticks={} hash={:016x} maximum_vertex_depth_m={maximum_depth:e}",
                world.snapshot().tick,
                world.snapshot().state_hash()
            );
            hashes
        };
        assert_eq!(run(), run());
    }
}

#[test]
fn saved_car_linear_suspension_paths_are_exact_affine_translations() {
    let (creation, initial, geometry, scene) = fixture();
    let mut displacement = vec![0.0; initial.velocities.len()];
    displacement[..3].copy_from_slice(&[0.1, 0.2, 0.3]);
    let mut sliding_joints = 0;
    for (coordinate, &bearing) in creation.dynamics.coordinate_bearings.iter().enumerate() {
        let kind = creation.bearings[bearing].kind;
        if kind.is_translational() {
            let bounds = kind.bounds();
            displacement[creation.dynamics.coordinate_velocities[coordinate]] =
                0.001_f64.clamp(f64::from(bounds[0]), f64::from(bounds[1]));
            sliding_joints += 1;
        }
    }
    assert!(sliding_joints > 0);
    let motion = crate::MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
    assert!(
        motion
            .bounds()
            .iter()
            .all(|bound| bound.angular_speed == 0.0)
    );
    for fraction in [0.25, 0.5, 0.75] {
        for (body, pose) in motion.poses_at(fraction).unwrap().iter().enumerate() {
            let start = motion.initial_poses()[body];
            let end = motion.final_poses()[body];
            assert!(
                pose.position
                    .distance(start.position.lerp(end.position, fraction))
                    < 1e-12
            );
            assert!((pose.rotation.dot(start.rotation).abs() - 1.0).abs() < 1e-12);
        }
    }
    let query = scene
        .sweep(&geometry, &motion, DVec3::ZERO, 1e-12, 1)
        .unwrap();
    assert!(query.linear_interval_evaluations > 0);
    assert_eq!(query.pose_evaluations, 0);
}

#[test]
fn activation_preserves_existing_authored_car_support_manifolds() {
    let (_, initial, geometry, scene) = fixture();
    let actual = scene
        .contacts(&geometry, &initial.poses, DVec3::ZERO)
        .unwrap();
    let activation = scene
        .activation_contacts(&geometry, &initial.poses, DVec3::ZERO)
        .unwrap();
    assert_eq!(activation.contacts, actual.contacts);
    assert_eq!(activation.unreduced_points, actual.unreduced_points);
}

#[test]
fn saved_car_cold_drop_resolves_its_first_impact_before_publication() {
    let (creation, mut initial, geometry, scene) = fixture();
    for pose in &mut initial.poses {
        pose.position.y += 0.051;
    }
    initial.velocities[1] = -4.0;
    let terrain = context(&scene, &geometry, 7);
    let run = || {
        let mut world = CpuJointMachine::new(creation.clone(), 7, initial.clone()).unwrap();
        let result = world
            .step_candidate(
                -DVec3::Y * 9.81,
                JointTickSettings::default(),
                &[],
                &[],
                Some(&terrain),
            )
            .map(|_| ());
        assert!(result.is_ok(), "{result:?} {:?}", world.diagnostics());
        assert_eq!(world.snapshot().tick, 1);
        assert!(world.diagnostics().impact_events > 0);
        println!(
            "cold_car_first_impact hash={:016x} diagnostics={:?}",
            world.snapshot().state_hash(),
            world.diagnostics()
        );
        for collider in &world.creation.colliders {
            let pose = world.snapshot().state.poses[collider.compound_index as usize];
            let shape = ContactPolytope::from_collider(collider)
                .unwrap()
                .transformed(pose.position, pose.rotation)
                .unwrap();
            assert!(shape.bounds()[0].y >= -0.005);
        }
        world.snapshot().clone()
    };
    assert_eq!(run(), run());
}

#[test]
fn saved_car_cold_drop_remains_bounded_through_settling() {
    cold_drop(
        TerrainIntegration::EventResolved,
        JointTickSettings::default(),
    );
}

#[test]
fn endpoint_contact_car_cold_drop_remains_bounded_through_settling() {
    cold_drop(
        TerrainIntegration::SlowContactsAtEndpoint,
        JointTickSettings::default(),
    );
}

// Captured completed state before the event-policy cold drop's tick-17 search
// exhausted all four retries. One tick reproduces that search in isolation.
#[test]
fn event_policy_tick17_search_completes_from_captured_state() {
    type Recorded = (u64, Vec<([f64; 3], [f64; 4])>, Vec<f64>, Vec<f64>);
    let (creation, _, geometry, scene) = fixture();
    let (_, poses, coordinates, velocities): Recorded =
        ron::from_str(include_str!("car_tick17_event_state.ron")).unwrap();
    let initial = MachineState {
        poses: poses
            .into_iter()
            .map(|(p, q)| crate::BodyPose {
                position: DVec3::from_array(p),
                rotation: bevy_math::DQuat::from_array(q),
            })
            .collect(),
        coordinates,
        velocities,
    };
    let mut world = CpuJointMachine::new(creation, 7, initial).unwrap();
    let mut terrain = context(&scene, &geometry, 7);
    terrain.maximum_depth = 0.005;
    let result = world
        .step_candidate(
            -DVec3::Y * 9.81,
            JointTickSettings::default(),
            &[],
            &[],
            Some(&terrain),
        )
        .map(|_| ());
    let diagnostics = world.diagnostics();
    println!(
        "tick17 result={result:?} attempts={} trials={} prefixes={} refinements={} localizations={} accepted={:e} holds={} failures={:?}",
        diagnostics.attempts,
        diagnostics.event_trials,
        diagnostics.event_prefix_commits,
        diagnostics.event_refinements,
        diagnostics.event_localizations,
        diagnostics.accepted_seconds,
        diagnostics.terrain_impact_holds,
        diagnostics.attempt_failures
    );
    assert!(result.is_ok());
}

fn cold_drop(integration: TerrainIntegration, settings: JointTickSettings) {
    let (creation, mut initial, geometry, scene) = fixture();
    for pose in &mut initial.poses {
        pose.position.y += 0.051;
    }
    initial.velocities[1] = -4.0;
    let mut terrain = context(&scene, &geometry, 7);
    terrain.maximum_depth = 0.005;
    terrain.integration = integration;
    let shapes = creation
        .colliders
        .iter()
        .map(|collider| {
            (
                collider.compound_index as usize,
                ContactPolytope::from_collider(collider).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let mut world = CpuJointMachine::new(creation, 7, initial).unwrap();
    let mut maximum_depth = 0.0_f64;
    let mut settling_depth = 0.0_f64;
    for tick in 1..=120 {
        let previous = world.snapshot().clone();
        let result = world
            .step_candidate(-DVec3::Y * 9.81, settings, &[], &[], Some(&terrain))
            .map(|_| ());
        if result.is_err() {
            // Exact last completed state, for single-tick regressions of this failure.
            if let Some(path) = std::env::var_os("MECHANIC_STATE_CAPTURE") {
                let state = &previous.state;
                let poses = state
                    .poses
                    .iter()
                    .map(|pose| (pose.position.to_array(), pose.rotation.to_array()))
                    .collect::<Vec<_>>();
                std::fs::write(
                    path,
                    ron::to_string(&(tick, poses, &state.coordinates, &state.velocities)).unwrap(),
                )
                .unwrap();
            }
            assert_eq!(
                world.snapshot(),
                &previous,
                "failed tick changed publication"
            );
            assert!(!world.diagnostics().published);
        }
        assert!(
            result.is_ok(),
            "tick={tick} result={result:?} diagnostics={:?}",
            world.diagnostics()
        );
        for (body, shape) in &shapes {
            let pose = world.snapshot().state.poses[*body];
            let bounds = shape
                .transformed(pose.position, pose.rotation)
                .unwrap()
                .bounds();
            assert!(bounds[0].x > -64.0 && bounds[1].x < 64.0);
            assert!(bounds[0].z > -64.0 && bounds[1].z < 64.0);
            maximum_depth = maximum_depth.max(-bounds[0].y);
            if tick > 90 {
                settling_depth = settling_depth.max(-bounds[0].y);
            }
        }
        assert!(maximum_depth <= 0.005);
    }
    assert!(settling_depth <= 0.002, "settling depth {settling_depth}");
    println!(
        "cold_car_settling hash={:016x} maximum_depth_m={maximum_depth} settling_depth_m={settling_depth}",
        world.snapshot().state_hash()
    );
}

#[test]
fn car_contact_reversal_uses_bounded_trials_and_agrees_with_finer_integration() {
    type Recorded = (Vec<([f64; 3], [f64; 4])>, Vec<f64>, Vec<f64>, f64);
    let (creation, _, geometry, scene) = fixture();
    let (poses, coordinates, velocities, proposal): Recorded =
        ron::from_str(include_str!("car_reversal.ron")).unwrap();
    let initial = MachineState {
        poses: poses
            .into_iter()
            .map(|(position, rotation)| crate::BodyPose {
                position: DVec3::from_array(position),
                rotation: bevy_math::DQuat::from_array(rotation),
            })
            .collect(),
        coordinates,
        velocities,
    };
    let world = CpuJointMachine::new(creation.clone(), 7, initial.clone()).unwrap();
    let mut results = Vec::new();
    for steps in [1, 1, 32, 64, 128] {
        let mut state = initial.clone();
        let mut diagnostics = JointTickDiagnostics::default();
        diagnostics
            .drive_impulses
            .resize(creation.dynamics.coordinate_bearings.len(), 0.0);
        let terrain = context(&scene, &geometry, 7);
        for _ in 0..steps {
            events::advance_interval(
                &creation,
                &world.passive,
                &world.drives,
                &mut state,
                -DVec3::Y * 9.81,
                proposal * 1000.0 / f64::from(steps),
                fixed(1),
                Some(&terrain),
                &mut diagnostics,
            )
            .unwrap();
        }
        assert!((diagnostics.accepted_seconds - proposal * 1000.0).abs() < 1e-17);
        assert_eq!(diagnostics.terrain_impact_holds, 0);
        if steps == 1 {
            // The former commit-each-short-prefix policy spends 90 trials here.
            assert!(diagnostics.event_trials <= 64);
            assert!(diagnostics.release_localizations > 0);
        }
        for collider in &creation.colliders {
            let pose = state.poses[collider.compound_index as usize];
            let bounds = ContactPolytope::from_collider(collider)
                .unwrap()
                .transformed(pose.position, pose.rotation)
                .unwrap()
                .bounds();
            assert!(bounds[0].x > -64.0 && bounds[1].x < 64.0);
            assert!(bounds[0].z > -64.0 && bounds[1].z < 64.0);
            assert!(bounds[0].y >= -0.005);
        }
        println!(
            "car_reversal steps={steps} trials={} accepted={} impacts={}",
            diagnostics.event_trials, diagnostics.accepted_intervals, diagnostics.impact_events
        );
        results.push(state);
    }
    assert_eq!(results[0], results[1]);
    // The 32/64/128 interval reference converges in state, though contact event
    // counts differ. This is a 29 microsecond replay, not a full rotating ODE proof.
    let reference = &results[4];
    for (index, state) in results[..4].iter().enumerate() {
        let velocity_error = state
            .velocities
            .iter()
            .zip(&reference.velocities)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let coordinate_error = state
            .coordinates
            .iter()
            .zip(&reference.coordinates)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let position_error = state
            .poses
            .iter()
            .zip(&reference.poses)
            .map(|(a, b)| a.position.distance(b.position))
            .fold(0.0, f64::max);
        assert!(velocity_error < if index < 2 { 1e-5 } else { 3e-7 });
        assert!(coordinate_error < if index < 2 { 3e-8 } else { 2e-11 });
        assert!(position_error < if index < 2 { 1e-8 } else { 2e-12 });
        println!(
            "car_reversal reference_error velocity={velocity_error:e} coordinate={coordinate_error:e} position={position_error:e}"
        );
    }
}

#[test]
#[allow(clippy::too_many_lines)] // Replay, independent bounds, and progressively finer reference.
fn a_validated_car_prefix_advances_time_once_when_reintegration_is_not_nested() {
    type Recorded = (Vec<([f64; 3], [f64; 4])>, Vec<f64>, Vec<f64>, f64);
    let (creation, _, geometry, scene) = fixture();
    let (poses, coordinates, velocities, proposal): Recorded =
        ron::from_str(include_str!("car_grazing_prefix.ron")).unwrap();
    let initial = MachineState {
        poses: poses
            .into_iter()
            .map(|(p, q)| crate::BodyPose {
                position: DVec3::from_array(p),
                rotation: bevy_math::DQuat::from_array(q),
            })
            .collect(),
        coordinates,
        velocities,
    };
    let world = CpuJointMachine::new(creation.clone(), 7, initial.clone()).unwrap();
    let mut results = Vec::new();
    for steps in [1, 1, 32, 64, 128] {
        let duration = proposal / f64::from(steps);
        let mut state = initial.clone();
        let mut diagnostics = JointTickDiagnostics::default();
        diagnostics
            .drive_impulses
            .resize(creation.dynamics.coordinate_bearings.len(), 0.0);
        let mut terrain = context(&scene, &geometry, 7);
        terrain.maximum_depth = 0.005;
        for _ in 0..steps {
            let result = events::advance_interval(
                &creation,
                &world.passive,
                &world.drives,
                &mut state,
                -DVec3::Y * 9.81,
                duration,
                fixed(1),
                Some(&terrain),
                &mut diagnostics,
            );
            assert!(result.is_ok(), "{result:?} {diagnostics:?}");
        }
        assert!((diagnostics.accepted_seconds - proposal).abs() < 1e-15);
        assert_eq!(diagnostics.terrain_impact_holds, 0);
        if steps == 1 {
            assert!(diagnostics.event_prefix_commits > 0);
        }
        for (coordinate, (drive, impulse)) in world
            .drives
            .iter()
            .zip(&diagnostics.drive_impulses)
            .enumerate()
        {
            let stall = f64::from(drive.source_a_max_acceleration)
                + f64::from(drive.source_b_max_acceleration);
            let limit = proposal
                * f64::from(creation.loop_topology.coordinate_axis_inertia[coordinate])
                * stall;
            assert!(
                impulse.abs() <= limit + 1e-9,
                "drive budget exceeded: {impulse} > {limit}"
            );
        }
        for collider in &creation.colliders {
            let pose = state.poses[collider.compound_index as usize];
            let bounds = ContactPolytope::from_collider(collider)
                .unwrap()
                .transformed(pose.position, pose.rotation)
                .unwrap()
                .bounds();
            assert!(bounds[0].x > -64.0 && bounds[1].x < 64.0);
            assert!(bounds[0].z > -64.0 && bounds[1].z < 64.0);
            assert!(bounds[0].y >= -0.005);
        }
        println!(
            "grazing steps={steps} trials={} prefixes={} time={}",
            diagnostics.event_trials,
            diagnostics.event_prefix_commits,
            diagnostics.accepted_seconds
        );
        results.push(state);
    }
    assert_eq!(results[0], results[1]);
    for (index, state) in results[..4].iter().enumerate() {
        let reference = &results[4];
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
        let coordinate = state
            .coordinates
            .iter()
            .zip(&reference.coordinates)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        println!(
            "grazing reference_error velocity={velocity:e} position={position:e} coordinate={coordinate:e}"
        );
        // Compare the complete committed trajectory, including its certified
        // prefixes, with refinement of the same physical interval. These are
        // local integration checks; they do not establish the cold-drop gate.
        let bounds = match index {
            0 | 1 => [7e-3, 2e-4, 7e-4],
            2 => [2e-4, 6e-7, 5e-6],
            _ => [6e-5, 3e-7, 1.2e-6],
        };
        for (error, bound) in [velocity, position, coordinate].into_iter().zip(bounds) {
            assert!(
                error < bound,
                "refinement index={index} error={error:e} bound={bound:e}"
            );
        }
    }
}

#[test]
fn reintegration_reuses_initial_geometry_until_a_validated_prefix_changes_the_state() {
    type Recorded = (Vec<([f64; 3], [f64; 4])>, Vec<f64>, Vec<f64>, f64);
    let (creation, _, geometry, scene) = fixture();
    let (poses, coordinates, velocities, duration): Recorded =
        ron::from_str(include_str!("car_grazing_prefix.ron")).unwrap();
    let initial = MachineState {
        poses: poses
            .into_iter()
            .map(|(p, q)| crate::BodyPose {
                position: DVec3::from_array(p),
                rotation: bevy_math::DQuat::from_array(q),
            })
            .collect(),
        coordinates,
        velocities,
    };
    let world = CpuJointMachine::new(creation.clone(), 7, initial.clone()).unwrap();
    let mut terrain = context(&scene, &geometry, 7);
    terrain.maximum_depth = 0.005;
    let run = |cached| {
        let mut state = initial.clone();
        let mut diagnostics = JointTickDiagnostics::default();
        diagnostics.drive_impulses.resize(world.drives.len(), 0.0);
        let advance = if cached {
            events::advance_interval_cached::<true>
        } else {
            events::advance_interval_cached::<false>
        };
        advance(
            &creation,
            &world.passive,
            &world.drives,
            &mut state,
            -DVec3::Y * 9.81,
            duration,
            fixed(1),
            Some(&terrain),
            &mut diagnostics,
        )
        .unwrap();
        (state, diagnostics)
    };
    let (uncached, before) = run(false);
    let (cached, after) = run(true);
    assert_eq!(cached, uncached);
    assert_eq!(after.drive_impulses, before.drive_impulses);
    assert_eq!(
        after.accepted_seconds.to_bits(),
        before.accepted_seconds.to_bits()
    );
    assert_eq!(after.event_trials, before.event_trials);
    assert_eq!(after.factorizations, before.factorizations);
    assert_eq!(after.constraint_iterations, before.constraint_iterations);
    assert!(after.event_prefix_commits > 0 && after.accepted_intervals > 1);
    assert!(after.terrain_contact_cache_hits > 0);
    assert_eq!(before.terrain_contact_cache_hits, 0);
    assert_eq!(before.initial_dynamics_cache_hits, 0);
    assert_eq!(
        before.terrain_contact_queries - after.terrain_contact_queries,
        after.terrain_contact_cache_hits
    );
    assert_eq!(
        before.dynamics_assemblies - after.dynamics_assemblies,
        after.initial_dynamics_cache_hits
    );
    // Every accepted prefix starts a new cache lifetime. It cannot reuse the
    // previous support geometry, even when the same terrain generation remains.
    assert_eq!(
        after.event_trials - after.initial_dynamics_cache_hits,
        after.accepted_intervals
    );
    println!(
        "interval_cache queries={}->{} assemblies={}->{} trials={}",
        before.terrain_contact_queries,
        after.terrain_contact_queries,
        before.dynamics_assemblies,
        after.dynamics_assemblies,
        after.event_trials
    );
}

#[test]
fn split_recovery_clears_recorded_car_penetration_without_changing_generalized_velocity() {
    type Recorded = (Vec<([f64; 3], [f64; 4])>, Vec<f64>, Vec<f64>, f64);
    let (creation, _, geometry, scene) = fixture();
    let (poses, coordinates, velocities, _): Recorded =
        ron::from_str(include_str!("car_grazing_drift.ron")).unwrap();
    let initial = MachineState {
        poses: poses
            .into_iter()
            .map(|(p, q)| crate::BodyPose {
                position: DVec3::from_array(p),
                rotation: bevy_math::DQuat::from_array(q),
            })
            .collect(),
        coordinates,
        velocities,
    };
    let world = CpuJointMachine::new(creation.clone(), 7, initial.clone()).unwrap();
    let terrain = context(&scene, &geometry, 7);
    let before = scene
        .recovery_contacts(&geometry, &initial.poses, DVec3::ZERO)
        .unwrap();
    assert!(before.contacts.iter().any(|point| point.depth > 1e-8));
    let run = || {
        let mut state = initial.clone();
        let mut diagnostics = JointTickDiagnostics::default();
        super::super::super::recovery::correct(
            &creation,
            &world.drives,
            &mut state,
            fixed(1),
            Some(&terrain),
            &mut diagnostics,
        )
        .unwrap();
        assert_eq!(state.velocities, initial.velocities);
        assert!(diagnostics.accepted_seconds.abs() < f64::MIN_POSITIVE);
        assert_eq!(diagnostics.impact_events, 0);
        assert!(diagnostics.terrain_recovery_passes > 0);
        for collider in &creation.colliders {
            let pose = state.poses[collider.compound_index as usize];
            let bounds = ContactPolytope::from_collider(collider)
                .unwrap()
                .transformed(pose.position, pose.rotation)
                .unwrap()
                .bounds();
            assert!(bounds[0].y >= -1e-12);
        }
        for (index, &value) in state.coordinates.iter().enumerate() {
            let [lower, upper] = bounds(&creation, &world.drives, index);
            assert!(value >= lower && value <= upper);
        }
        state
    };
    assert_eq!(run(), run());
}

#[test]
fn endpoint_contact_cold_drop_with_eight_fixed_substeps() {
    cold_drop(TerrainIntegration::SlowContactsAtEndpoint, fixed(8));
}

#[test]
fn car_recovery_refreshes_contacts_encountered_by_the_correction() {
    type Recorded = (Vec<([f64; 3], [f64; 4])>, Vec<f64>, Vec<f64>, Vec<f64>);
    let (creation, _, geometry, scene) = fixture();
    let (poses, coordinates, velocities, displacement): Recorded =
        ron::from_str(include_str!("car_recovery_new_contact.ron")).unwrap();
    let initial = MachineState {
        poses: poses
            .into_iter()
            .map(|(position, rotation)| crate::BodyPose {
                position: DVec3::from_array(position),
                rotation: bevy_math::DQuat::from_array(rotation),
            })
            .collect(),
        coordinates,
        velocities,
    };
    let mut terrain = context(&scene, &geometry, 7);
    terrain.maximum_depth = 0.005;
    let mut path = initial.clone();
    path.velocities = displacement;
    let mut diagnostics = JointTickDiagnostics::default();
    let outcome = super::super::super::events::validate_path(
        &creation,
        &path,
        1.0,
        Some(&terrain),
        true,
        None,
        &mut diagnostics,
    );
    assert!(matches!(
        outcome,
        Ok(super::super::super::events::PathOutcome::Refine(_))
    ));
    let world = CpuJointMachine::new(creation.clone(), 7, initial.clone()).unwrap();
    let run = || {
        let mut state = initial.clone();
        let mut diagnostics = JointTickDiagnostics::default();
        let result = super::super::super::recovery::correct(
            &creation,
            &world.drives,
            &mut state,
            fixed(8),
            Some(&terrain),
            &mut diagnostics,
        );
        assert!(result.is_ok(), "{result:?} {diagnostics:?}");
        assert_eq!(state.velocities, initial.velocities);
        assert_eq!(diagnostics.accepted_seconds.to_bits(), 0.0_f64.to_bits());
        assert_eq!(diagnostics.impact_events, 0);
        assert!(diagnostics.terrain_recovery_passes > 1);
        for collider in &creation.colliders {
            let pose = state.poses[collider.compound_index as usize];
            let bounds = ContactPolytope::from_collider(collider)
                .unwrap()
                .transformed(pose.position, pose.rotation)
                .unwrap()
                .bounds();
            assert!(bounds[0].y >= -1e-12);
        }
        for (index, &value) in state.coordinates.iter().enumerate() {
            let [lower, upper] = bounds(&creation, &world.drives, index);
            assert!(value >= lower && value <= upper);
        }
        state
    };
    assert_eq!(run(), run());
}
