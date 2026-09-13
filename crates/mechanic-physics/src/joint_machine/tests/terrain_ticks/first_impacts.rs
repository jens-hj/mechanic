use super::*;

#[test]
fn shallow_first_impact_activates_even_when_its_depth_envelope_would_pass() {
    let (creation, geometry, _) = cube();
    let mut initial = MachineState::at_rest(&creation);
    initial.poses[0].position.y = 0.5001;
    initial.velocities[1] = -0.012;
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    let displacement = initial
        .velocities
        .iter()
        .map(|v| v * TICK_SECONDS)
        .collect::<Vec<_>>();
    let motion = crate::MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
    assert_eq!(
        scene
            .validate_penetration(&geometry, &motion, DVec3::ZERO, 0.002, 128)
            .unwrap()
            .outcome,
        crate::TerrainPathOutcome::Bounded
    );
    let mut world = CpuJointMachine::new(creation, 7, initial).unwrap();
    world
        .step_candidate(DVec3::ZERO, fixed(1), &[], &[], Some(&terrain))
        .unwrap();
    assert!((world.snapshot().state.poses[0].position.y - 0.5).abs() < 1e-10);
    assert!(world.snapshot().state.velocities[1].abs() < 1e-8);
    assert_eq!(world.diagnostics().impact_events, 1);
    assert_eq!(world.diagnostics().event_refinements, 1);
    assert_eq!(
        world.diagnostics().accepted_seconds.to_bits(),
        TICK_SECONDS.to_bits()
    );
    assert_eq!(world.diagnostics().terrain_impact_holds, 0);
    assert_eq!(world.diagnostics().attempts, 1);
}

#[test]
fn existing_floor_support_does_not_hide_a_new_finite_wall_impact() {
    use mechanic_world::{TerrainTriangleGroupMask, TriangleBvhTriangle};
    let (creation, geometry, poses) = cube();
    let mut initial = MachineState {
        poses,
        ..MachineState::at_rest(&creation)
    };
    initial.velocities[0] = 0.012;
    let mut chunk = terrain([TerrainMaterial::Rock; 2]);
    let changed = std::sync::Arc::make_mut(&mut chunk);
    changed
        .vertices
        .extend([[0.5001, 0.0, -1.0], [0.5001, 0.0, 1.0], [0.5001, 1.0, 0.0]]);
    changed.indices.extend([6, 7, 8]);
    changed
        .material_weights
        .extend([changed.material_weights[0]; 3]);
    changed.bounds.maximum.0.y = 1.0;
    changed.triangle_bvh.bounds = changed.bounds;
    changed.triangle_bvh.nodes[0].bounds = changed.bounds;
    changed.triangle_bvh.nodes[0].triangle_count = 3;
    changed.triangle_bvh.triangles.push(TriangleBvhTriangle {
        indices: [6, 7, 8],
        group_mask: TerrainTriangleGroupMask::REGULAR,
    });
    let mut scene = TerrainContactScene::default();
    scene.publish(1, &[chunk], &[]).unwrap();
    let terrain = context(&scene, &geometry, 7);
    let contacts = scene
        .contacts(&geometry, &initial.poses, DVec3::ZERO)
        .unwrap();
    assert!(contacts.contacts.iter().all(|point| matches!(
        point.feature.obstacle,
        crate::ContactObstacle::Terrain { triangle, .. } if triangle < 2
    )));
    assert!(!contacts.contacts.is_empty());
    let displacement = initial
        .velocities
        .iter()
        .map(|v| v * TICK_SECONDS)
        .collect::<Vec<_>>();
    let motion = crate::MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
    assert_eq!(
        scene
            .validate_penetration(&geometry, &motion, DVec3::ZERO, 0.002, 128)
            .unwrap()
            .outcome,
        crate::TerrainPathOutcome::Bounded
    );
    let sweep = scene
        .sweep_new_contacts(&geometry, &motion, DVec3::ZERO, 1e-12, 128)
        .unwrap();
    let crate::TerrainSweepOutcome::Impact(hit) = sweep.outcome else {
        panic!("{:?}", sweep.outcome)
    };
    assert!(matches!(
        hit.target,
        crate::ContactTarget::Terrain { triangle: 2, .. }
    ));
    assert_eq!(sweep.supported_pairs, 2);
    let mut world = CpuJointMachine::new(creation, 7, initial).unwrap();
    world
        .step_candidate(DVec3::ZERO, fixed(1), &[], &[], Some(&terrain))
        .unwrap();
    assert!(
        (world.snapshot().state.poses[0].position.x - (f64::from(0.5001_f32) - 0.5)).abs() < 1e-9
    );
    assert!(world.snapshot().state.velocities[0].abs() < 1e-8);
    assert!(world.snapshot().state.poses[0].position.y >= 0.498_999_999);
    assert_eq!(world.diagnostics().impact_events, 1);
    assert_eq!(world.diagnostics().terrain_impact_holds, 0);
    assert_eq!(
        world.diagnostics().accepted_seconds.to_bits(),
        TICK_SECONDS.to_bits()
    );
}

#[test]
fn an_initial_touching_impact_bounces_once_then_advances_with_outgoing_velocity() {
    let (creation, geometry, poses) = cube();
    let mut initial = MachineState {
        poses,
        ..MachineState::at_rest(&creation)
    };
    initial.velocities[1] = -4.0;
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    let query = scene
        .contacts(&geometry, &initial.poses, DVec3::ZERO)
        .unwrap();
    let rebound = 4.0 * query.contacts[0].response[2];
    assert!(rebound > 0.0 && rebound <= 4.0);
    for subdivisions in [1, 2, 4, 8] {
        let mut world = CpuJointMachine::new(creation.clone(), 7, initial.clone()).unwrap();
        world
            .step_candidate(DVec3::ZERO, fixed(subdivisions), &[], &[], Some(&terrain))
            .unwrap();
        assert_eq!(world.diagnostics().impact_events, 1);
        assert_eq!(
            world.diagnostics().factorizations,
            subdivisions as usize + 1
        );
        assert!((world.snapshot().state.velocities[1] - rebound).abs() < 1e-8);
        let expected_height = initial.poses[0].position.y + rebound * TICK_SECONDS;
        assert!((world.snapshot().state.poses[0].position.y - expected_height).abs() < 1e-9);
        world
            .step_candidate(DVec3::ZERO, fixed(subdivisions), &[], &[], Some(&terrain))
            .unwrap();
        assert_eq!(world.diagnostics().impact_events, 0);
        assert!((world.snapshot().state.velocities[1] - rebound).abs() < 1e-8);
    }
}

#[test]
fn cold_fast_impact_matches_analytic_bounce_and_uses_the_remaining_tick() {
    let (creation, geometry, _) = cube();
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    let mut initial = MachineState::at_rest(&creation);
    initial.poses[0].position.y = 1.0;
    initial.velocities[1] = -100.0;
    let mut touching = initial.poses.clone();
    touching[0].position.y = 0.5;
    let restitution = scene
        .contacts(&geometry, &touching, DVec3::ZERO)
        .unwrap()
        .contacts[0]
        .response[2];
    let impact_time = 0.5 / 100.0;
    let expected_velocity = 100.0 * restitution;
    let expected_height = 0.5 + expected_velocity * (TICK_SECONDS - impact_time);
    for subdivisions in [1, 2, 4, 8] {
        let run = || {
            let mut world = CpuJointMachine::new(creation.clone(), 7, initial.clone()).unwrap();
            world
                .step_candidate(DVec3::ZERO, fixed(subdivisions), &[], &[], Some(&terrain))
                .unwrap();
            assert!((world.snapshot().state.poses[0].position.y - expected_height).abs() < 1e-8);
            assert!((world.snapshot().state.velocities[1] - expected_velocity).abs() < 1e-8);
            assert_eq!(world.diagnostics().impact_events, 1);
            assert!((world.diagnostics().accepted_seconds - TICK_SECONDS).abs() < 1e-17);
            world.snapshot().state_hash()
        };
        assert_eq!(run(), run());
    }
}

#[test]
fn gravity_is_reintegrated_at_the_impact_time_before_restitution() {
    let (creation, geometry, _) = cube();
    let scene = scene();
    let mut terrain = context(&scene, &geometry, 7);
    terrain.restitution_threshold = 0.0;
    let mut initial = MachineState::at_rest(&creation);
    let gap = 0.0001;
    initial.poses[0].position.y = 0.5 + gap;
    let mut touching = initial.poses.clone();
    touching[0].position.y = 0.5;
    let restitution = scene
        .contacts(&geometry, &touching, DVec3::ZERO)
        .unwrap()
        .contacts[0]
        .response[2];
    let impact_time = (2.0 * gap / 9.81_f64).sqrt();
    let rebound = restitution * 9.81 * impact_time;
    let remaining = TICK_SECONDS - impact_time;
    // The full tick can contain repeated bounces; isolate one analytic impact.
    assert!(remaining > 0.0);
    let world = CpuJointMachine::new(creation, 7, initial).unwrap();
    // Inspect a single physical interval ending before the second analytic hit.
    let duration = impact_time + rebound / 9.81 * 0.5;
    let mut state = world.snapshot().state.clone();
    let mut diagnostics = JointTickDiagnostics::default();
    events::advance_interval(
        &world.creation,
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
    let after = duration - impact_time;
    assert!((state.velocities[1] - (rebound - 9.81 * after)).abs() < 1e-8);
    assert!(
        (state.poses[0].position.y - (0.5 + rebound * after - 0.5 * 9.81 * after * after)).abs()
            < 1e-9
    );
    assert_eq!(diagnostics.impact_events, 1);
    assert!(diagnostics.event_refinements > 1);
    assert!((diagnostics.accepted_seconds - duration).abs() < 1e-17);
    // The scratch interval has not published or consumed an external tick.
    assert_eq!(world.snapshot().tick, 0);
}

#[test]
fn a_rotating_cube_reaches_a_real_finite_edge_before_applying_impact() {
    let (creation, geometry, _) = cube();
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    let mut initial = MachineState::at_rest(&creation);
    initial.poses[0].position.y = 0.6;
    initial.velocities[5] = 60.0;
    let impact_angle = (1.2 / 2.0_f64.sqrt()).asin() - std::f64::consts::FRAC_PI_4;
    let duration = impact_angle / 60.0 + 1e-5;
    let run = || {
        let mut state = initial.clone();
        let mut diagnostics = JointTickDiagnostics::default();
        let result = events::advance_interval(
            &creation,
            &[],
            &[],
            &mut state,
            DVec3::ZERO,
            duration,
            fixed(1),
            Some(&terrain),
            &mut diagnostics,
        );
        assert!(result.is_ok(), "{result:?} {diagnostics:?}");
        assert_eq!(diagnostics.impact_events, 1);
        assert!(diagnostics.terrain_separation_evaluations > 0);
        assert!(state.velocities[1] > 0.0);
        assert!(state.velocities[5] < 60.0);
        assert!((diagnostics.accepted_seconds - duration).abs() < 1e-17);
        let shape = mechanic_core::ContactPolytope::from_collider(&creation.colliders[0])
            .unwrap()
            .transformed(state.poses[0].position, state.poses[0].rotation)
            .unwrap();
        assert!(shape.bounds()[0].y >= -0.002);
        state
    };
    assert_eq!(run(), run());
}

#[test]
fn a_fast_drop_through_a_finite_hole_does_not_activate_a_ghost_surface() {
    let (creation, geometry, _) = cube();
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    let mut initial = MachineState::at_rest(&creation);
    initial.poses[0].position = DVec3::new(4.0, 1.0, 0.0);
    initial.velocities[1] = -100.0;
    let mut world = CpuJointMachine::new(creation, 7, initial).unwrap();
    world
        .step_candidate(DVec3::ZERO, fixed(1), &[], &[], Some(&terrain))
        .unwrap();
    assert_eq!(world.diagnostics().impact_events, 0);
    assert!(
        (world.snapshot().state.poses[0].position.y - (1.0 - 100.0 * TICK_SECONDS)).abs() < 1e-12
    );
    assert!((world.snapshot().state.velocities[1] + 100.0).abs() < 1e-12);
}

#[test]
fn separating_support_releases_before_gravity_reverses_its_velocity() {
    let (creation, geometry, _) = cube();
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    let mut initial = MachineState::at_rest(&creation);
    initial.poses[0].position.y = 0.5;
    initial.velocities[1] = 0.1;
    let expected_height = 0.5 + 0.1 * TICK_SECONDS - 0.5 * 9.81 * TICK_SECONDS * TICK_SECONDS;
    let expected_velocity = 0.1 - 9.81 * TICK_SECONDS;
    assert!(expected_height > 0.5 && expected_velocity < 0.0);
    for subdivisions in [1, 2, 4, 8] {
        let mut world = CpuJointMachine::new(creation.clone(), 7, initial.clone()).unwrap();
        world
            .step_candidate(
                -DVec3::Y * 9.81,
                fixed(subdivisions),
                &[],
                &[],
                Some(&terrain),
            )
            .unwrap();
        assert!(
            (world.snapshot().state.poses[0].position.y - expected_height).abs() < 1e-10,
            "substeps={subdivisions} state={:?}",
            world.snapshot().state
        );
        assert!((world.snapshot().state.velocities[1] - expected_velocity).abs() < 1e-9);
        assert_eq!(world.diagnostics().impact_events, 0);
        world
            .step_candidate(
                -DVec3::Y * 9.81,
                fixed(subdivisions),
                &[],
                &[],
                Some(&terrain),
            )
            .unwrap();
        assert!((world.snapshot().state.poses[0].position.y - 0.5).abs() < 1e-9);
        assert!(world.snapshot().state.velocities[1].abs() < 1e-8);
        assert_eq!(world.diagnostics().impact_events, 1);
    }
}

#[test]
fn released_support_can_return_and_impact_the_same_triangle_within_one_tick() {
    let (creation, geometry, _) = cube();
    let scene = scene();
    let terrain = context(&scene, &geometry, 7);
    let mut initial = MachineState::at_rest(&creation);
    initial.poses[0].position.y = 0.5;
    initial.velocities[1] = 0.05;
    assert!(2.0 * initial.velocities[1] / 9.81 < TICK_SECONDS);
    for subdivisions in [1, 2, 4, 8] {
        let mut world = CpuJointMachine::new(creation.clone(), 7, initial.clone()).unwrap();
        world
            .step_candidate(
                -DVec3::Y * 9.81,
                fixed(subdivisions),
                &[],
                &[],
                Some(&terrain),
            )
            .unwrap();
        assert!(
            (world.snapshot().state.poses[0].position.y - 0.5).abs() < 1e-9,
            "substeps={subdivisions} state={:?}",
            world.snapshot().state
        );
        assert!(world.snapshot().state.velocities[1].abs() < 1e-8);
        assert_eq!(world.diagnostics().impact_events, 1);
    }
}

#[test]
fn first_impact_on_an_interior_terrain_triangle_survives_manifold_reduction() {
    use mechanic_world::{TerrainTriangleGroupMask, TriangleBvhTriangle};
    let (creation, geometry, _) = cube();
    let mut initial = MachineState::at_rest(&creation);
    initial.poses[0].position.y = 0.5001;
    initial.velocities[1] = -0.012;
    let mut chunk = terrain([TerrainMaterial::Rock; 2]);
    let changed = std::sync::Arc::make_mut(&mut chunk);
    changed.vertices.clear();
    changed.indices.clear();
    changed.triangle_bvh.triangles.clear();
    // Visit the interior tile first. It witnesses arrival, but none of its
    // points are extreme corners of the final four-point floor manifold.
    let spans = [(-1.0, -0.2), (-0.2, 0.2), (0.2, 1.0)];
    let tiles = [(1, 1)].into_iter().chain(
        (0..3)
            .flat_map(|x| (0..3).map(move |z| (x, z)))
            .filter(|&tile| tile != (1, 1)),
    );
    for (x, z) in tiles {
        let (x0, x1) = spans[x];
        let (z0, z1) = spans[z];
        let base = u32::try_from(changed.vertices.len()).unwrap();
        changed
            .vertices
            .extend([[x0, 0.0, z0], [x0, 0.0, z1], [x1, 0.0, z1], [x1, 0.0, z0]]);
        for indices in [[base, base + 1, base + 2], [base, base + 2, base + 3]] {
            changed.indices.extend(indices);
            changed.triangle_bvh.triangles.push(TriangleBvhTriangle {
                indices,
                group_mask: TerrainTriangleGroupMask::REGULAR,
            });
        }
    }
    changed
        .material_weights
        .resize(changed.vertices.len(), changed.material_weights[0]);
    changed.triangle_bvh.nodes[0].triangle_count = 18;
    let mut scene = TerrainContactScene::default();
    scene.publish(1, &[chunk], &[]).unwrap();
    let mut touching = initial.clone();
    touching.poses[0].position.y = 0.5;
    let contacts = scene
        .activation_contacts(&geometry, &touching.poses, DVec3::ZERO)
        .unwrap();
    assert!(contacts.contacts.iter().all(|point| matches!(
        point.feature.obstacle,
        crate::ContactObstacle::Terrain { triangle, .. } if triangle >= 2
    )));
    let terrain = context(&scene, &geometry, 7);
    let mut world = CpuJointMachine::new(creation, 7, initial).unwrap();
    world
        .step_candidate(DVec3::ZERO, fixed(1), &[], &[], Some(&terrain))
        .unwrap();
    assert!((world.snapshot().state.poses[0].position.y - 0.5).abs() < 1e-10);
    assert!(world.snapshot().state.velocities[1].abs() < 1e-8);
    assert_eq!(world.diagnostics().terrain_impact_holds, 0);
    for _ in 0..60 {
        world
            .step_candidate(-DVec3::Y * 9.81, fixed(1), &[], &[], Some(&terrain))
            .unwrap();
        assert!((world.snapshot().state.poses[0].position.y - 0.5).abs() < 1e-9);
        assert!(
            world
                .snapshot()
                .state
                .velocities
                .iter()
                .all(|v| v.abs() < 1e-8)
        );
    }
}
