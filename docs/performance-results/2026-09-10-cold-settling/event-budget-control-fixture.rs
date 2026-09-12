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
    for substeps in [1, 2, 4, 8] {
        let run = || {
            let mut world = CpuJointMachine::new(creation.clone(), 7, initial.clone()).unwrap();
            world
                .step_candidate(-DVec3::Y * 9.81, fixed(substeps), &[], &[], Some(&terrain))
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
                "saved_car_support substeps={substeps} hash={:016x} diagnostics={:?}",
                world.snapshot().state_hash(),
                world.diagnostics()
            );
            world.snapshot().clone()
        };
        assert_eq!(run(), run());
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
    let (creation, mut initial, geometry, scene) = fixture();
    for pose in &mut initial.poses {
        pose.position.y += 0.051;
    }
    initial.velocities[1] = -4.0;
    let mut terrain = context(&scene, &geometry, 7);
    terrain.maximum_depth = 0.005;
    terrain.maximum_event_trials = 1024;
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
        let result = world
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
