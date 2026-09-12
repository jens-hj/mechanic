use super::*;
use crate::{
    MachineState,
    terrain_contacts::tests::{cube, terrain},
};
use mechanic_world::TerrainMaterial;
use std::sync::Arc;

fn scene() -> TerrainContactScene {
    let mut scene = TerrainContactScene::default();
    scene
        .publish(1, &[terrain([TerrainMaterial::Rock; 2])], &[])
        .unwrap();
    scene
}

#[test]
fn stationary_existing_support_is_bounded_without_ignoring_its_triangles() {
    let (creation, geometry, _) = cube();
    let mut state = MachineState::at_rest(&creation);
    state.poses[0].position.y = 0.499;
    let displacement = [0.0; 6];
    let motion = MachineMotion::new(&creation, 7, &state, &displacement).unwrap();
    let scene = scene();
    let query = scene
        .validate_penetration(&geometry, &motion, DVec3::ZERO, 0.002, 1)
        .unwrap();
    assert_eq!(query.outcome, TerrainPathOutcome::Bounded);
    assert_eq!(query.triangle_candidates, 2);
    assert_eq!(query.certified_intervals, 2);
    assert_eq!(query.envelope_evaluations, 2);
    assert_eq!(query.pose_evaluations, 1);
    assert_eq!(query.pose_cache_hits, 1);
    assert_eq!(query.topology_generation, 7);
    assert_eq!(query.terrain_generation, 1);
    let failed = scene
        .validate_penetration(&geometry, &motion, DVec3::ZERO, 0.0005, 1)
        .unwrap();
    assert!(matches!(
        failed.outcome,
        TerrainPathOutcome::ExcessPenetration(_)
    ));
    assert_eq!(failed.pose_evaluations, 1);
    assert_eq!(failed.pose_cache_hits, 0);
    // A new query must sample its own motion, even with identical generations.
    state.poses[0].position.y = 0.49;
    let deeper = MachineMotion::new(&creation, 7, &state, &displacement).unwrap();
    let changed = scene
        .validate_penetration(&geometry, &deeper, DVec3::ZERO, 0.002, 1)
        .unwrap();
    assert!(matches!(
        changed.outcome,
        TerrainPathOutcome::ExcessPenetration(_)
    ));
    assert_eq!(changed.pose_evaluations, 1);
    assert_eq!(changed.pose_cache_hits, 0);
}

#[test]
fn full_rotation_rejects_penetration_hidden_at_start_midpoint_and_end() {
    let (creation, geometry, _) = cube();
    let mut state = MachineState::at_rest(&creation);
    state.poses[0].position.y = 0.5;
    let saved = state.clone();
    let displacement = [0.0, 0.0, 0.0, 0.0, 0.0, std::f64::consts::TAU];
    let motion = MachineMotion::new(&creation, 7, &state, &displacement).unwrap();
    let scene = scene();
    for fraction in [0.0, 0.5, 1.0] {
        let points = scene
            .contacts(&geometry, &motion.poses_at(fraction).unwrap(), DVec3::ZERO)
            .unwrap();
        assert!(points.contacts.iter().all(|point| point.depth < 1e-12));
    }
    let query = scene
        .validate_penetration(&geometry, &motion, DVec3::ZERO, 0.005, 128)
        .unwrap();
    assert!(!matches!(query.outcome, TerrainPathOutcome::Bounded));
    assert!(query.pose_evaluations > 2);
    assert_eq!(state, saved);
    assert_eq!(
        scene
            .validate_penetration(&geometry, &motion, DVec3::ZERO, 0.005, 128)
            .unwrap()
            .outcome,
        query.outcome
    );
}

#[test]
fn sliding_support_subdivides_and_exhaustion_never_claims_clearance() {
    let (creation, geometry, _) = cube();
    let mut state = MachineState::at_rest(&creation);
    state.poses[0].position.y = 0.5;
    let displacement = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
    let motion = MachineMotion::new(&creation, 7, &state, &displacement).unwrap();
    let scene = scene();
    for limit in [1, 6, 32] {
        let failed = scene
            .validate_penetration(&geometry, &motion, DVec3::ZERO, 0.002, limit)
            .unwrap();
        assert!(matches!(failed.outcome, TerrainPathOutcome::Unconverged(_)));
        assert!(failed.envelope_evaluations <= limit);
    }
    let query = scene
        .validate_penetration(&geometry, &motion, DVec3::ZERO, 0.002, 128)
        .unwrap();
    assert_eq!(query.outcome, TerrainPathOutcome::Bounded);
    assert!(query.certified_intervals > query.triangle_candidates);
}

#[test]
fn finite_holes_and_origin_rebases_preserve_supported_path_results() {
    let (creation, geometry, _) = cube();
    let mut state = MachineState::at_rest(&creation);
    state.poses[0].position = DVec3::new(4.0, 0.5, 0.0);
    let displacement = [0.0, 0.0, 0.0, 0.0, 0.0, std::f64::consts::TAU];
    let motion = MachineMotion::new(&creation, 7, &state, &displacement).unwrap();
    let mut scene = scene();
    let miss = scene
        .validate_penetration(&geometry, &motion, DVec3::ZERO, 0.002, 128)
        .unwrap();
    assert_eq!(miss.outcome, TerrainPathOutcome::Bounded);
    assert_eq!(miss.triangle_candidates, 0);
    state.poses[0].position.x = 0.0;
    let motion = MachineMotion::new(&creation, 7, &state, &displacement).unwrap();
    let expected = scene
        .validate_penetration(&geometry, &motion, DVec3::ZERO, 0.002, 128)
        .unwrap();
    let shift = DVec3::new(1e6, 2e6, -3e6);
    let mut chunk = terrain([TerrainMaterial::Rock; 2]);
    let updated = Arc::make_mut(&mut chunk);
    updated.origin.0 += shift;
    updated.bounds.minimum.0 += shift;
    updated.bounds.maximum.0 += shift;
    updated.triangle_bvh.bounds = updated.bounds;
    updated.triangle_bvh.nodes[0].bounds = updated.bounds;
    scene.publish(2, &[chunk], &[]).unwrap();
    let actual = scene
        .validate_penetration(&geometry, &motion, shift, 0.002, 128)
        .unwrap();
    assert_eq!(actual.outcome, expected.outcome);
    assert_eq!(actual.terrain_generation, 2);
    assert_eq!(actual.envelope_evaluations, expected.envelope_evaluations);
    for bad_depth in [-1.0, f64::INFINITY, f64::NAN] {
        assert!(
            scene
                .validate_penetration(&geometry, &motion, shift, bad_depth, 128)
                .is_err()
        );
    }
    let wrong = MachineMotion::new(&creation, 8, &state, &displacement).unwrap();
    assert!(
        scene
            .validate_penetration(&geometry, &wrong, shift, 0.002, 128)
            .is_err()
    );
}

#[test]
fn a_separated_body_below_a_finite_surface_does_not_get_a_penetration_hold() {
    let (creation, geometry, _) = cube();
    let mut state = MachineState::at_rest(&creation);
    state.poses[0].position.y = -0.6;
    let scene = scene();
    assert!(
        scene
            .contacts(&geometry, &state.poses, DVec3::ZERO)
            .unwrap()
            .contacts
            .is_empty()
    );
    let displacement = [0.0; 6];
    let motion = MachineMotion::new(&creation, 7, &state, &displacement).unwrap();
    let query = scene
        .validate_penetration(&geometry, &motion, DVec3::ZERO, 0.002, 1)
        .unwrap();
    assert_eq!(query.outcome, TerrainPathOutcome::Bounded);
    assert_eq!(query.certified_intervals, 2);
}

#[test]
fn motion_below_a_finite_surface_is_certified_after_subdivision() {
    let (creation, geometry, _) = cube();
    let mut state = MachineState::at_rest(&creation);
    state.poses[0].position.y = -0.6;
    let displacement = [0.4, 0.0, 0.0, 0.0, 0.0, 0.0];
    let motion = MachineMotion::new(&creation, 7, &state, &displacement).unwrap();
    let query = scene()
        .validate_penetration(&geometry, &motion, DVec3::ZERO, 0.002, 128)
        .unwrap();
    assert_eq!(query.outcome, TerrainPathOutcome::Bounded);
    assert!(query.certified_intervals > query.triangle_candidates);
}
