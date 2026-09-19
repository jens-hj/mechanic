use super::super::*;
use crate::{MachineMotion, MachineState};
use bevy_math::IVec3;
use mechanic_core::{
    BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, FaceKind,
    FaceRef, GridRotation,
};

fn encloses_samples(geometry: &MachineCollisionGeometry, motion: &MachineMotion<'_>) {
    let coarse = geometry.body_path_bounds(motion, 0.0).unwrap();
    for collider in &geometry.colliders {
        let envelope = path_bounds(collider, motion, 0.0).unwrap();
        for step in 0..=128 {
            let pose = motion.poses_at(f64::from(step) / 128.0).unwrap()[collider.body];
            let actual = collider
                .local
                .transformed_bounds(pose.position, pose.rotation)
                .unwrap();
            assert!(actual[0].cmpge(coarse[collider.body][0]).all());
            assert!(actual[1].cmple(coarse[collider.body][1]).all());
            assert!(
                actual[0].cmpge(envelope[0]).all(),
                "{actual:?} outside {envelope:?}"
            );
            assert!(
                actual[1].cmple(envelope[1]).all(),
                "{actual:?} outside {envelope:?}"
            );
        }
    }
}

#[test]
fn directional_translation_and_multiple_turns_enclose_the_whole_path() {
    let (creation, geometry, _) = tests::cube();
    let initial = MachineState::at_rest(&creation);
    for displacement in [
        [30.0, -2.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, -4.0, 0.0, 8.0 * std::f64::consts::TAU, 0.0],
        [3.0, 0.0, -4.0, 5.0, -20.0, 7.0],
        [0.3, -0.2, 0.1, 0.01, -0.15, 0.07],
        [-0.2, 0.1, 0.0, -0.6, 0.3, -0.2],
    ] {
        let motion = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
        encloses_samples(&geometry, &motion);
    }
}

#[test]
fn off_centre_bearings_and_moving_ancestors_keep_conservative_bounds() {
    let mut graph = ConstructionGraph::new();
    let mut add = |dimensions, ticks| {
        let BuildOutcome::Spawned(id) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    dimensions,
                    BuildPose::from_position_ticks(ticks, GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            panic!()
        };
        id
    };
    let base = add([1, 1, 1], IVec3::new(0, 400, 0));
    let bar = add([16, 1, 1], IVec3::new(700, 400, 100));
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(base, FaceKind::PositiveZ),
            FaceRef::part(bar, FaceKind::NegativeZ),
            Vec3::new(0.0, 1.0, 0.125),
            Vec3::Z,
        )))
        .unwrap();
    for fixed in [true, false] {
        let creation = graph
            .compile_with_static_parts(if fixed { vec![base] } else { vec![] })
            .unwrap();
        let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
        let initial = MachineState::at_rest(&creation);
        for (angular, change) in [
            (0.0, -4.0 * std::f64::consts::TAU),
            (2.0, -4.0 * std::f64::consts::TAU),
            (0.0, 0.16),
            (0.0, -0.16),
            (0.1, -0.16),
            (-0.1, 0.16),
        ] {
            let mut displacement = vec![0.0; initial.velocities.len()];
            if !fixed {
                displacement[..6].copy_from_slice(&[3.0, -1.0, 0.0, angular, 0.0, 0.0]);
            }
            displacement[creation.dynamics.coordinate_velocities[0]] = change;
            let motion = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
            encloses_samples(&geometry, &motion);
        }
    }
}

#[test]
fn empty_region_rejects_escaped_motion_and_changed_terrain_or_topology() {
    let (creation, geometry, _) = tests::cube();
    let mut initial = MachineState::at_rest(&creation);
    initial.poses[0].position.y = 2.0;
    let displacement = [0.0, 0.0, 0.0, 0.0, 50.0, 0.0];
    let motion = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
    let mut scene = TerrainContactScene::default();
    let region = scene
        .empty_contact_region(&geometry, &motion, DVec3::ZERO, 0.1)
        .unwrap();
    assert!(region.contains(
        &scene,
        &geometry,
        &motion.poses_at(0.4).unwrap(),
        DVec3::ZERO,
        &[0.02]
    ));
    let mut escaped = initial.poses.clone();
    escaped[0].position.y -= 2.0;
    assert!(!region.contains(&scene, &geometry, &escaped, DVec3::ZERO, &[0.02]));
    let changed = MachineCollisionGeometry::new(&creation, 8).unwrap();
    assert!(!region.contains(&scene, &changed, &initial.poses, DVec3::ZERO, &[0.02]));
    scene
        .publish(
            1,
            &[tests::terrain([mechanic_world::TerrainMaterial::Rock; 2])],
            &[],
        )
        .unwrap();
    assert!(!region.contains(&scene, &geometry, &initial.poses, DVec3::ZERO, &[0.02]));
}

#[test]
fn suspension_travel_with_rotating_ancestors_stays_inside_the_envelope() {
    let instance: mechanic_world::WorldCreationInstanceDoc = ron::from_str(include_str!(
        "../../../../mechanic-bench/tests/fixtures/driven_car_instance.ron"
    ))
    .unwrap();
    let loaded = instance.creation.into_graph().unwrap();
    let creation = loaded
        .graph
        .compile_with_sockets([], &loaded.sockets)
        .unwrap();
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    let mut initial = MachineState::at_rest(&creation);
    initial.coordinates.iter_mut().for_each(|q| *q = 0.07);
    let displacement = (0..initial.velocities.len())
        .map(|row| if row % 2 == 0 { 0.6 } else { -0.4 })
        .collect::<Vec<_>>();
    let motion = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
    encloses_samples(&geometry, &motion);
}

#[test]
fn empty_region_cannot_hide_an_approaching_body() {
    let creation = tests::loose_cubes();
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    let mut initial = MachineState::at_rest(&creation);
    initial.poses[0].position = DVec3::new(-2.0, 3.0, 0.0);
    initial.poses[1].position = DVec3::new(2.0, 3.0, 0.0);
    let mut displacement = vec![0.0; initial.velocities.len()];
    let scene = TerrainContactScene::default();
    let motion = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
    let region = scene
        .empty_contact_region(&geometry, &motion, DVec3::ZERO, 0.1)
        .unwrap();
    displacement[creation.dynamics.body_velocities[0].start] = 4.0;
    let approach = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
    assert!(
        scene
            .empty_contact_region(&geometry, &approach, DVec3::ZERO, 0.1)
            .is_none()
    );
    assert!(!region.contains(
        &scene,
        &geometry,
        &approach.poses_at(0.5).unwrap(),
        DVec3::ZERO,
        &[0.02, 0.02]
    ));
    assert!(matches!(
        scene
            .sweep(&geometry, &approach, DVec3::ZERO, 1e-9, 128)
            .unwrap()
            .outcome,
        TerrainSweepOutcome::Impact(_)
    ));
}

#[test]
fn assembly_clearance_rechecks_other_assemblies_and_publications() {
    let creation = tests::loose_cubes();
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    let mut initial = MachineState::at_rest(&creation);
    initial.poses[0].position = DVec3::new(-2.0, 3.0, 0.0);
    initial.poses[1].position = DVec3::new(2.0, 3.0, 0.0);
    let zero = vec![0.0; initial.velocities.len()];
    let motion = MachineMotion::new(&creation, 7, &initial, &zero).unwrap();
    let mut scene = TerrainContactScene::default();
    let region = scene
        .assembly_clearance(&geometry, &motion, DVec3::ZERO, 0.1, geometry.assemblies[0])
        .unwrap();
    assert!(region.contains(&scene, &geometry, &motion, DVec3::ZERO, 0.02));
    assert!(!region.contains(&scene, &geometry, &motion, DVec3::X, 0.02));
    let changed = MachineCollisionGeometry::new(&creation, 8).unwrap();
    assert!(!region.contains(&scene, &changed, &motion, DVec3::ZERO, 0.02));
    let mut approach = initial.clone();
    approach.poses[1].position.x = -1.9;
    let entering = MachineMotion::new(&creation, 7, &approach, &zero).unwrap();
    assert!(!region.contains(&scene, &geometry, &entering, DVec3::ZERO, 0.02));
    // Both participants move along paths whose endpoint gaps alone are unsafe.
    let mut relative = zero.clone();
    relative[creation.dynamics.body_velocities[0].start] = 2.0;
    relative[creation.dynamics.body_velocities[1].start] = -2.0;
    let closing = MachineMotion::new(&creation, 7, &initial, &relative).unwrap();
    assert!(!region.contains(&scene, &geometry, &closing, DVec3::ZERO, 0.02));
    assert!(matches!(
        scene
            .sweep(&geometry, &closing, DVec3::ZERO, 1e-9, 128)
            .unwrap()
            .outcome,
        TerrainSweepOutcome::Impact(_)
    ));
    scene
        .publish(
            1,
            &[tests::terrain([mechanic_world::TerrainMaterial::Rock; 2])],
            &[],
        )
        .unwrap();
    assert!(!region.contains(&scene, &geometry, &motion, DVec3::ZERO, 0.02));
}

#[test]
fn local_contact_queries_match_exhaustive_contacts_for_an_approaching_pair() {
    let creation = tests::loose_cubes();
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    let mut state = MachineState::at_rest(&creation);
    state.poses[0].position = DVec3::new(0.0, 3.0, 0.0);
    state.poses[1].position = DVec3::new(1.01, 3.0, 0.0);
    let mut groups = ContactGroups::new(geometry.assembly_count, &[0.01, 0.01], 0.0);
    groups.advance(&geometry, &[0.006, 0.006], &[0.0, 0.0], 0.2, false);
    let scene = TerrainContactScene::default();
    let full = scene
        .proximity(&geometry, &state.poses, DVec3::ZERO, 0.02)
        .unwrap();
    let local = scene
        .proximity_groups(
            &geometry,
            &state.poses,
            DVec3::ZERO,
            &[0.02, 0.02],
            Some(&groups),
        )
        .unwrap();
    assert!(!full.contacts.is_empty());
    assert_eq!(
        local.contacts.iter().map(|c| c.feature).collect::<Vec<_>>(),
        full.contacts.iter().map(|c| c.feature).collect::<Vec<_>>()
    );
    for (a, b) in local.contacts.iter().zip(&full.contacts) {
        assert_eq!(a.body_point, b.body_point);
        assert_eq!(a.normal, b.normal);
        assert!((a.separation - b.separation).abs() < 1e-12);
    }
}

#[test]
fn slow_approaching_assemblies_without_combined_coverage_still_sweep() {
    let creation = tests::loose_cubes();
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    let mut state = MachineState::at_rest(&creation);
    state.poses[0].position = DVec3::new(0.0, 3.0, 0.0);
    state.poses[1].position = DVec3::new(1.03, 3.0, 0.0);
    let mut delta = vec![0.0; state.velocities.len()];
    delta[creation.dynamics.body_velocities[0].start] = 0.019;
    delta[creation.dynamics.body_velocities[1].start] = -0.019;
    let motion = MachineMotion::new(&creation, 7, &state, &delta).unwrap();
    let travel = [0.019, 0.019];
    let mut groups = ContactGroups::new(geometry.assembly_count, &[0.02, 0.02], 0.0);
    groups.advance(&geometry, &travel, &[0.0, 0.0], 0.2, false);
    groups.require_fast(&geometry, &travel, 0.05, 0.2);
    assert!(groups.needs_sweep(&geometry));
    let scene = TerrainContactScene::default();
    let full = scene
        .sweep_new_contacts(&geometry, &motion, DVec3::ZERO, 1e-9, 128)
        .unwrap();
    for threshold in [0.05, 0.01] {
        // Both below activation and both fast must retain the relative impact.
        groups.require_fast(&geometry, &travel, threshold, 0.2);
        let local = scene
            .sweep_contact_groups(&geometry, &motion, DVec3::ZERO, 1e-9, 128, &groups, &[])
            .unwrap();
        assert!(matches!(local.outcome, TerrainSweepOutcome::Impact(_)));
        assert_eq!(local.outcome, full.outcome);
    }
}

#[test]
fn narrowed_pair_queries_keep_combined_travel_conservative() {
    let creation = tests::loose_cubes();
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    let mut state = MachineState::at_rest(&creation);
    state.poses[0].position = DVec3::new(0.0, 3.0, 0.0);
    state.poses[1].position = DVec3::new(1.001, 3.0, 0.0);
    let mut groups = ContactGroups::new(geometry.assembly_count, &[0.01, 0.01], 0.0);
    groups.advance(&geometry, &[0.006, 0.006], &[0.0, 0.0], 0.2, false);
    groups.refreshed(&geometry, &[0.01, 0.01], &[0.002, 0.002], 0.0);
    state.poses[0].position.x += 0.0015;
    state.poses[1].position.x -= 0.0015;
    groups.advance(&geometry, &[0.0015, 0.0015], &[0.0, 0.0], 0.2, false);
    let scene = TerrainContactScene::default();
    let full = scene
        .proximity(&geometry, &state.poses, DVec3::ZERO, 0.002)
        .unwrap();
    let local = scene
        .proximity_groups(
            &geometry,
            &state.poses,
            DVec3::ZERO,
            &[0.002, 0.002],
            Some(&groups),
        )
        .unwrap();
    assert!(!full.contacts.is_empty());
    assert_eq!(
        local.contacts.iter().map(|c| c.feature).collect::<Vec<_>>(),
        full.contacts.iter().map(|c| c.feature).collect::<Vec<_>>()
    );
}

#[test]
fn finite_support_reuse_matches_fresh_initial_support_queries() {
    let (creation, geometry, _) = tests::cube();
    let mut scene = TerrainContactScene::default();
    scene
        .publish(
            1,
            &[tests::terrain([mechanic_world::TerrainMaterial::Rock; 2])],
            &[],
        )
        .unwrap();
    for gap in [0.0, 1e-12, 0.019] {
        let mut state = MachineState::at_rest(&creation);
        state.poses[0].position.y = 0.5 + gap;
        let contacts = scene
            .proximity(&geometry, &state.poses, DVec3::ZERO, 0.02)
            .unwrap();
        for displacement in [
            [0.2, -0.1, 0.0, 0.0, 0.0, 0.1],
            [0.0, 0.2, 0.0, 0.0, 6.3, 0.0],
        ] {
            let motion = MachineMotion::new(&creation, 7, &state, &displacement).unwrap();
            let mut groups = ContactGroups::new(geometry.assembly_count, &[0.02], 0.0);
            groups.require_fast(&geometry, &[1.0], 0.05, 0.25);
            let full = scene
                .sweep_new_contacts(&geometry, &motion, DVec3::ZERO, 1e-9, 128)
                .unwrap();
            let cached = scene
                .sweep_contact_groups(
                    &geometry,
                    &motion,
                    DVec3::ZERO,
                    1e-9,
                    128,
                    &groups,
                    &contacts.activation_features,
                )
                .unwrap();
            assert_eq!(cached.outcome, full.outcome);
        }
    }
}

#[test]
fn a_car_moving_as_one_keeps_its_internal_contacts_but_not_its_terrain_contacts() {
    let instance: mechanic_world::WorldCreationInstanceDoc = ron::from_str(include_str!(
        "../../../../mechanic-bench/tests/fixtures/driven_car_instance.ron"
    ))
    .unwrap();
    let loaded = instance.creation.into_graph().unwrap();
    let creation = loaded
        .graph
        .compile_with_sockets([], &loaded.sockets)
        .unwrap();
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    let assembly = (0..geometry.assembly_count)
        .find(|&assembly| geometry.internal_collisions[assembly])
        .expect("the car's bodies can touch each other");
    let bodies = (0..geometry.bodies)
        .filter(|&body| {
            geometry.assemblies[body] == assembly && !geometry.body_colliders[body].is_empty()
        })
        .collect::<Vec<_>>();
    let state = MachineState::at_rest(&creation);
    let root = bodies
        .iter()
        .copied()
        .find(|&body| creation.loop_topology.body_parents[body].is_root)
        .unwrap();
    // A root carrying the car 30 cm forward and turning it a little.
    let mut displacement = vec![0.0; state.velocities.len()];
    let rows = creation.dynamics.body_velocities[root].clone();
    displacement[rows.start] = 0.3;
    displacement[rows.start + 4] = 0.05;
    let motion = MachineMotion::new(&creation, 7, &state, &displacement).unwrap();
    let margins = vec![0.07; geometry.colliders.len()];
    let mut groups = ContactGroups::new(geometry.assembly_count, &margins, 0.02);
    let measured = groups.measure(&geometry, &motion);
    groups.advance_measured(&geometry, &measured, 0.25, false);
    assert!(groups.includes(&geometry, root, None));
    for &body in &bodies {
        for &other in &bodies {
            assert!(!groups.includes(&geometry, body, Some(other)));
        }
    }
}
