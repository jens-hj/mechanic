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
        .checked_sweep(&geometry, &motion, DVec3::ZERO, 1e-9, 128)
        .unwrap();
    let TerrainSweepOutcome::Impact(hit) = query.outcome else {
        panic!("articulated path must hit floor: {:?}", query.outcome);
    };
    let angle = (1.0 / 2.0_f64.hypot(0.125)).asin() - (0.125_f64 / 2.0).atan();
    assert!((hit.fraction - angle / std::f64::consts::TAU).abs() < 1e-8);
    assert!(query.pose_evaluations > 2);
    assert!(query.pose_evaluations < query.separation_evaluations);
    assert_eq!(
        query.linear_interval_evaluations, 0,
        "full rotations must use the rotational path"
    );
    let exhausted = scene
        .checked_sweep(&geometry, &motion, DVec3::ZERO, 1e-9, 1)
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
        .checked_sweep(&geometry, &motion, DVec3::ZERO, 1e-9, 128)
        .unwrap();
    let TerrainSweepOutcome::Impact(hit) = query.outcome else {
        panic!("fast translation must hit floor");
    };
    assert!((hit.fraction - 0.475).abs() < 1e-12);
    let bounded = scene
        .checked_sweep(&geometry, &motion, DVec3::ZERO, 1e-9, 1)
        .unwrap();
    assert_eq!(bounded.outcome, query.outcome);
    assert_eq!(bounded.pose_evaluations, 0);
    assert_eq!(bounded.linear_interval_evaluations, 2);
    assert_eq!(initial, saved);
    initial.poses[0].position.x = 4.0;
    let miss = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
    assert!(matches!(
        scene
            .checked_sweep(&geometry, &miss, DVec3::ZERO, 1e-9, 128)
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
            .checked_sweep(&geometry, &wrong, DVec3::ZERO, 1e-9, 128)
            .is_err()
    );
    let motion = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
    let expected = scene
        .checked_sweep(&geometry, &motion, DVec3::ZERO, 1e-9, 128)
        .unwrap();
    let shift = DVec3::new(1e6, 2e6, -3e6);
    let updated = Arc::make_mut(&mut chunk);
    updated.origin.0 += shift;
    updated.bounds.minimum.0 += shift;
    updated.bounds.maximum.0 += shift;
    updated.triangle_bvh.bounds = updated.bounds;
    updated.triangle_bvh.nodes[0].bounds = updated.bounds;
    scene.publish(2, &[chunk], &[]).unwrap();
    let rebased = scene
        .checked_sweep(&geometry, &motion, shift, 1e-9, 128)
        .unwrap();
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
        .checked_sweep(&geometry, &motion, DVec3::ZERO, 1e-9, 1)
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
        .checked_sweep(&geometry, &motion, DVec3::ZERO, 1e-12, 128)
        .unwrap();
    assert_eq!(query.outcome, TerrainSweepOutcome::Clear);
    assert!(query.quadratic_interval_evaluations > 0);
    assert!(query.velocity_evaluations < query.quadratic_interval_evaluations);
    assert_eq!(query.linear_interval_evaluations, 0);
    assert!(query.separation_evaluations < 32);
}

#[test]
fn reused_starting_geometry_preserves_impacts_across_motion_paths_and_publications() {
    let (creation, geometry, _) = cube();
    let mut scene = TerrainContactScene::default();
    let mut initial = MachineState::at_rest(&creation);
    // Alternate translations and rotations at identical poses, then move the
    // body. A fresh compiled geometry is the cold-cache reference each time.
    for height in [0.51, 2.0, 0.51] {
        initial.poses[0].position.y = height;
        for displacement in [
            [3.0, 0.0, 0.0, 0.0, 0.001, 0.0],
            [0.0, -20.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, std::f64::consts::TAU],
            [0.0, 20.0, 0.0, 0.0, 0.01, 0.0],
        ] {
            let motion = MachineMotion::new(&creation, 7, &initial, &displacement).unwrap();
            for _ in 0..2 {
                let publication = scene.generation + 1;
                scene
                    .publish(publication, &[terrain([TerrainMaterial::Rock; 2])], &[])
                    .unwrap();
                let fresh = MachineCollisionGeometry::new(&creation, 7).unwrap();
                let expected = scene
                    .checked_sweep(&fresh, &motion, DVec3::ZERO, 1e-9, 128)
                    .unwrap();
                for _ in 0..2 {
                    let actual = scene
                        .checked_sweep(&geometry, &motion, DVec3::ZERO, 1e-9, 128)
                        .unwrap();
                    assert_eq!(actual.outcome, expected.outcome);
                    assert_eq!(actual.triangle_candidates, expected.triangle_candidates);
                    assert_eq!(
                        actual.collider_pair_candidates,
                        expected.collider_pair_candidates
                    );
                }
            }
        }
    }
}

impl TerrainContactScene {
    fn checked_sweep(
        &self,
        geometry: &MachineCollisionGeometry,
        motion: &MachineMotion<'_>,
        origin: DVec3,
        tolerance: f64,
        limit: usize,
    ) -> Result<TerrainSweepQuery, PhysicsError> {
        struct Restore;
        impl Drop for Restore {
            fn drop(&mut self) {
                REUSE_START.set(true);
            }
        }
        let actual = self.sweep(geometry, motion, origin, tolerance, limit);
        let new_contacts = self.sweep_new_contacts(geometry, motion, origin, tolerance, limit);
        let _restore = Restore;
        REUSE_START.set(false);
        let reference = self.sweep(geometry, motion, origin, tolerance, limit);
        let reference_new = self.sweep_new_contacts(geometry, motion, origin, tolerance, limit);
        assert_eq!(
            new_contacts.as_ref().map(|q| q.outcome),
            reference_new.as_ref().map(|q| q.outcome)
        );
        match (&actual, &reference) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a.outcome, b.outcome);
                assert_eq!(a.collider_pair_candidates, b.collider_pair_candidates);
                assert_eq!(a.triangle_candidates, b.triangle_candidates);
            }
            (Err(_), Err(_)) => {}
            _ => panic!("cached and uncached sweep validity differs"),
        }
        actual
    }
}

#[test]
fn collider_sweeps_reuse_geometry_without_reusing_another_paths_velocities() {
    let creation = crate::terrain_contacts::tests::loose_cubes();
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    let scene = TerrainContactScene::default();
    let mut state = MachineState::at_rest(&creation);
    state.poses[0].position = DVec3::ZERO;
    state.poses[1].position = DVec3::X * 1.1;
    for speed in [0.01, -3.0, 3.0, -0.01] {
        let mut displacement = vec![0.0; state.velocities.len()];
        let second = creation.dynamics.body_velocities[1].start;
        displacement[second] = speed;
        displacement[second + 5] = 0.01;
        let motion = MachineMotion::new(&creation, 7, &state, &displacement).unwrap();
        for _ in 0..2 {
            let query = scene
                .checked_sweep(&geometry, &motion, DVec3::ZERO, 1e-9, 128)
                .unwrap();
            assert_eq!(
                matches!(query.outcome, TerrainSweepOutcome::Impact(_)),
                speed < -0.1
            );
        }
    }
}

#[test]
fn rotated_pipe_sweeps_preserve_the_opening_and_annulus() {
    use mechanic_core::{PipeBendDimensions, PipeBendSpec};
    for (inner, offset, collides) in [(0.6, 0.0, false), (0.0, 0.0, true), (0.6, 0.375, true)] {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::SpawnPipeBend(PipeBendSpec::new(
                PipeBendDimensions::new(1.0, inner, 6).unwrap(),
                BuildPose::default(),
            )))
            .unwrap();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_half_grid(IVec3::splat(64), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
        let creation = graph.compile().unwrap();
        let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
        let scene = TerrainContactScene::default();
        let mut state = MachineState::at_rest(&creation);
        state.poses[1].position = DVec3::new(offset, 1.5, 0.0);
        let rotation = DQuat::from_rotation_z(0.61);
        for pose in &mut state.poses {
            pose.position = rotation * pose.position;
            pose.rotation = rotation * pose.rotation;
        }
        let mut displacement = vec![0.0; state.velocities.len()];
        let second = creation.dynamics.body_velocities[1].start;
        displacement[second..second + 3].copy_from_slice(&(rotation * -DVec3::Y * 0.5).to_array());
        let motion = MachineMotion::new(&creation, 7, &state, &displacement).unwrap();
        for _ in 0..2 {
            let query = scene
                .checked_sweep(&geometry, &motion, DVec3::ZERO, 1e-9, 128)
                .unwrap();
            assert_eq!(
                matches!(query.outcome, TerrainSweepOutcome::Impact(_)),
                collides,
                "inner={inner}, offset={offset}, outcome={:?}",
                query.outcome
            );
        }
    }
}

#[test]
fn simultaneous_terrain_arrivals_keep_the_first_collider_and_triangle() {
    let creation = crate::terrain_contacts::tests::loose_cubes();
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    let mut scene = TerrainContactScene::default();
    scene
        .publish(1, &[terrain([TerrainMaterial::Rock; 2])], &[])
        .unwrap();
    let mut state = MachineState::at_rest(&creation);
    state.poses[0].position = DVec3::new(-0.6, 2.0, 0.0);
    state.poses[1].position = DVec3::new(0.6, 2.0, 0.0);
    let mut displacement = vec![0.0; state.velocities.len()];
    for range in &creation.dynamics.body_velocities {
        displacement[range.start + 1] = -4.0;
    }
    let motion = MachineMotion::new(&creation, 7, &state, &displacement).unwrap();
    let query = scene
        .checked_sweep(&geometry, &motion, DVec3::ZERO, 1e-9, 128)
        .unwrap();
    let TerrainSweepOutcome::Impact(hit) = query.outcome else {
        panic!("simultaneous impact expected");
    };
    assert_eq!(hit.collider, 0);
    assert!((hit.fraction - 0.375).abs() < 1e-12);
}
