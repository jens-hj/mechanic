use super::*;
use crate::{
    MachineState,
    terrain_contacts::tests::{cube, terrain},
};
use bevy_math::{IVec3, Vec3};
use mechanic_core::{
    BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, FaceKind,
    FaceRef, GridRotation,
};
use mechanic_world::TerrainMaterial;
use std::sync::Arc;

#[test]
fn articulated_full_turn_finds_a_finite_impact_between_clear_endpoint_poses() {
    let mut graph = ConstructionGraph::new();
    let mut add = |dimensions, ticks| {
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
    };
    let base = add([1, 1, 1], IVec3::new(0, 400, 0));
    let bar = add([16, 1, 1], IVec3::new(0, 400, 100));
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(base, FaceKind::PositiveZ),
            FaceRef::part(bar, FaceKind::NegativeZ),
            Vec3::new(0.0, 1.0, 0.125),
            Vec3::Z,
        )))
        .unwrap();
    let creation = graph.compile_with_static_parts([base]).unwrap();
    let geometry = MachineCollisionGeometry::new(&creation, 1).unwrap();
    let initial = MachineState::at_rest(&creation);
    assert_eq!(initial.velocities.len(), 1);
    let displacement = [std::f64::consts::TAU];
    let motion = MachineMotion::new(&creation, 1, &initial, &displacement).unwrap();
    let mut chunk = terrain([TerrainMaterial::Rock; 2]);
    let chunk_mut = Arc::make_mut(&mut chunk);
    for vertex in &mut chunk_mut.vertices {
        vertex[0] *= 8.0;
        vertex[2] *= 8.0;
    }
    chunk_mut.bounds.minimum.0 *= 8.0;
    chunk_mut.bounds.maximum.0 *= 8.0;
    chunk_mut.triangle_bvh.bounds = chunk_mut.bounds;
    chunk_mut.triangle_bvh.nodes[0].bounds = chunk_mut.bounds;
    let mut scene = TerrainContactScene::default();
    scene.publish(1, &[chunk], &[]).unwrap();
    for fraction in [0.0, 1.0] {
        assert!(
            scene
                .contacts(&geometry, &motion.poses_at(fraction).unwrap(), DVec3::ZERO)
                .unwrap()
                .contacts
                .is_empty()
        );
    }
    let query = scene
        .sweep(&geometry, &motion, DVec3::ZERO, 1e-9, 128)
        .unwrap();
    let TerrainSweepOutcome::Impact(hit) = query.outcome else {
        panic!("articulated path must hit floor: {:?}", query.outcome);
    };
    let angle = (1.0 / 2.0_f64.hypot(0.125)).asin() - (0.125_f64 / 2.0).atan();
    assert!((hit.fraction - angle / std::f64::consts::TAU).abs() < 1e-8);
    assert!(query.pose_evaluations > 2);
    assert_eq!(query.pose_evaluations, query.separation_evaluations);
    assert_eq!(
        query.linear_interval_evaluations, 0,
        "full rotations must use the rotational path"
    );
    let exhausted = scene
        .sweep(&geometry, &motion, DVec3::ZERO, 1e-9, 1)
        .unwrap();
    assert!(matches!(
        exhausted.outcome,
        TerrainSweepOutcome::Unconverged(_)
    ));
    assert_eq!(exhausted.separation_evaluations, 1);
    assert_eq!(query.topology_generation, 1);
    assert_eq!(query.terrain_generation, 1);
    println!(
        "articulated_full_turn_toi={} pose_evaluations={}",
        hit.fraction, query.pose_evaluations
    );
}

#[test]
fn exact_linear_sweep_handles_fast_translation_and_finite_holes() {
    let (creation, geometry, _) = cube();
    let mut initial = MachineState::at_rest(&creation);
    initial.poses[0].position = DVec3::Y * 10.0;
    let saved = initial.clone();
    let displacement = [0.0, -20.0, 0.0, 0.0, 0.0, 0.0];
    let motion = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
    let mut scene = TerrainContactScene::default();
    scene
        .publish(1, &[terrain([TerrainMaterial::Rock; 2])], &[])
        .unwrap();
    let query = scene
        .sweep(&geometry, &motion, DVec3::ZERO, 1e-9, 128)
        .unwrap();
    let TerrainSweepOutcome::Impact(hit) = query.outcome else {
        panic!("fast translation must hit floor");
    };
    assert!((hit.fraction - 0.475).abs() < 1e-12);
    let bounded = scene
        .sweep(&geometry, &motion, DVec3::ZERO, 1e-9, 1)
        .unwrap();
    assert_eq!(bounded.outcome, query.outcome);
    assert_eq!(bounded.pose_evaluations, 0);
    assert_eq!(bounded.linear_interval_evaluations, 2);
    assert_eq!(initial, saved);
    initial.poses[0].position.x = 4.0;
    let miss = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
    assert!(matches!(
        scene
            .sweep(&geometry, &miss, DVec3::ZERO, 1e-9, 128)
            .unwrap()
            .outcome,
        TerrainSweepOutcome::Clear
    ));
}

#[test]
fn sweep_rejects_a_different_topology_and_preserves_origin_rebased_impact_time() {
    let (creation, geometry, _) = cube();
    let mut initial = MachineState::at_rest(&creation);
    initial.poses[0].position.y = 10.0;
    let displacement = [0.0, -20.0, 0.0, 0.0, 0.0, 0.0];
    let mut scene = TerrainContactScene::default();
    let mut chunk = terrain([TerrainMaterial::Rock; 2]);
    scene.publish(1, &[Arc::clone(&chunk)], &[]).unwrap();
    let wrong = MachineMotion::new(&creation, 9, &initial, &displacement).unwrap();
    assert!(
        scene
            .sweep(&geometry, &wrong, DVec3::ZERO, 1e-9, 128)
            .is_err()
    );
    let motion = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
    let expected = scene
        .sweep(&geometry, &motion, DVec3::ZERO, 1e-9, 128)
        .unwrap();
    let shift = DVec3::new(1e6, 2e6, -3e6);
    let updated = Arc::make_mut(&mut chunk);
    updated.origin.0 += shift;
    updated.bounds.minimum.0 += shift;
    updated.bounds.maximum.0 += shift;
    updated.triangle_bvh.bounds = updated.bounds;
    updated.triangle_bvh.nodes[0].bounds = updated.bounds;
    scene.publish(2, &[chunk], &[]).unwrap();
    let rebased = scene.sweep(&geometry, &motion, shift, 1e-9, 128).unwrap();
    assert_eq!(rebased.outcome, expected.outcome);
    assert_eq!(rebased.terrain_generation, 2);
}

#[test]
fn near_separated_parallel_motion_does_not_invent_a_first_impact() {
    let (creation, geometry, _) = cube();
    let mut initial = MachineState::at_rest(&creation);
    initial.poses[0].position.y = 0.5 + 1e-13;
    let displacement = [3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let motion = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
    let mut scene = TerrainContactScene::default();
    scene
        .publish(1, &[terrain([TerrainMaterial::Rock; 2])], &[])
        .unwrap();
    assert!(
        scene
            .contacts(&geometry, &initial.poses, DVec3::ZERO)
            .unwrap()
            .contacts
            .is_empty()
    );
    let query = scene
        .sweep(&geometry, &motion, DVec3::ZERO, 1e-9, 1)
        .unwrap();
    assert_eq!(query.outcome, TerrainSweepOutcome::Clear);
    assert_eq!(query.linear_interval_evaluations, 2);
    assert_eq!(query.pose_evaluations, 0);
    assert_eq!(query.separation_evaluations, 0);
    assert_eq!(motion.preparation_pose_evaluations(), 2);
    assert_eq!(motion.final_poses(), motion.poses_at(1.0).unwrap());
}

#[test]
fn rotating_parallel_motion_certifies_clearance_without_spending_tangential_speed() {
    let (creation, geometry, _) = cube();
    let mut initial = MachineState::at_rest(&creation);
    initial.poses[0].position.y = 0.5 + 1e-8;
    // Rotation about Y preserves the true floor gap. The old total-speed bound
    // advanced by about 3e-9 of this path per SAT evaluation and exhausted 128.
    let displacement = [3.0, 0.0, 0.0, 0.0, 0.001, 0.0];
    let motion = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
    let mut scene = TerrainContactScene::default();
    scene
        .publish(1, &[terrain([TerrainMaterial::Rock; 2])], &[])
        .unwrap();
    let query = scene
        .sweep(&geometry, &motion, DVec3::ZERO, 1e-12, 128)
        .unwrap();
    assert_eq!(query.outcome, TerrainSweepOutcome::Clear);
    assert!(query.quadratic_interval_evaluations > 0);
    assert_eq!(
        query.velocity_evaluations,
        query.quadratic_interval_evaluations
    );
    assert_eq!(query.linear_interval_evaluations, 0);
    assert!(query.separation_evaluations < 32);
}
