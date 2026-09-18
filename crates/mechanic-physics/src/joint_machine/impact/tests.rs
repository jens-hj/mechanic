use super::*;
use crate::{
    TerrainContactScene,
    terrain_contacts::tests::{cube, terrain},
};
use bevy_math::DVec3;
use mechanic_world::TerrainMaterial;

#[test]
fn impact_applies_restitution_once_without_moving_or_biasing_penetration() {
    let (creation, geometry, poses) = cube();
    let mut scene = TerrainContactScene::default();
    scene
        .publish(1, &[terrain([TerrainMaterial::Rock; 2])], &[])
        .unwrap();
    let settings = JointTickConfig::default();
    let mut previous: Option<f64> = None;
    for depth in [0.0, 0.001, 0.004] {
        let mut state = MachineState {
            poses: poses.clone(),
            ..MachineState::at_rest(&creation)
        };
        state.poses[0].position.y = 0.5 - depth;
        state.velocities[1] = -4.0;
        let positions = state.poses.clone();
        let model = MachineDynamics::assemble(&creation, &state.poses, &state.coordinates).unwrap();
        let mut query = scene
            .contacts(&geometry, &state.poses, DVec3::ZERO)
            .unwrap();
        assert!(!query.contacts.is_empty());
        for contact in &mut query.contacts {
            contact.response = [0.0, 0.0, 0.5, 0.0];
        }
        let impact = query
            .impact_constraints(&model, &state.velocities, 1.0, 1e-7)
            .unwrap();
        let mut diagnostics = JointTickDiagnostics::default();
        assert!(
            activate(
                &creation,
                &model,
                &[],
                &mut state,
                &impact,
                settings,
                &mut diagnostics
            )
            .unwrap()
        );
        assert!((state.velocities[1] - 2.0).abs() < 1e-8);
        assert!(state.velocities[3..].iter().all(|value| value.abs() < 1e-8));
        assert_eq!(state.poses, positions);
        assert_eq!(diagnostics.impact_events, 1);
        if let Some(velocity) = previous {
            assert!((state.velocities[1] - velocity).abs() < 1e-8);
        }
        previous = Some(state.velocities[1]);
        let refreshed = query
            .impact_constraints(&model, &state.velocities, 1.0, 1e-7)
            .unwrap();
        let before = state.clone();
        assert!(
            !activate(
                &creation,
                &model,
                &[],
                &mut state,
                &refreshed,
                settings,
                &mut diagnostics
            )
            .unwrap()
        );
        assert_eq!(state, before);
        assert_eq!(diagnostics.factorizations, 1);
    }
}

#[test]
fn failed_instantaneous_solve_preserves_all_physical_state() {
    let (creation, geometry, poses) = cube();
    let mut scene = TerrainContactScene::default();
    scene
        .publish(1, &[terrain([TerrainMaterial::Rock; 2])], &[])
        .unwrap();
    let mut state = MachineState {
        poses,
        ..MachineState::at_rest(&creation)
    };
    state.velocities = vec![3.0, -4.0, 1.0, 0.3, 0.2, 0.7];
    let before = state.clone();
    let model = MachineDynamics::assemble(&creation, &state.poses, &state.coordinates).unwrap();
    let query = scene
        .contacts(&geometry, &state.poses, DVec3::ZERO)
        .unwrap();
    let impact = query
        .impact_constraints(&model, &state.velocities, 1.0, 1e-7)
        .unwrap();
    let settings = JointTickConfig {
        constraint_iterations: 1,
        tolerance: 1e-30,
        ..Default::default()
    };
    let mut diagnostics = JointTickDiagnostics::default();
    assert_eq!(
        activate(
            &creation,
            &model,
            &[],
            &mut state,
            &impact,
            settings,
            &mut diagnostics
        ),
        Err(PhysicsError::NotConverged)
    );
    assert_eq!(state, before);
    assert_eq!(diagnostics.impact_attempts, 1);
    assert_eq!(diagnostics.impact_events, 0);
    assert!(diagnostics.impact_factor_solves > 0);
    assert_eq!(
        diagnostics.failure_stage,
        Some(crate::JointFailureStage::Impact)
    );
}

#[test]
fn contact_impulse_respects_an_active_joint_stop_and_back_drives_the_root() {
    let creation = crate::joint_machine::tests::rotor(false);
    let geometry = crate::MachineCollisionGeometry::new(&creation, 1).unwrap();
    let mut scene = TerrainContactScene::default();
    scene
        .publish(1, &[terrain([TerrainMaterial::Rock; 2])], &[])
        .unwrap();
    let mut initial = MachineState::at_rest(&creation);
    for pose in &mut initial.poses {
        pose.position.y += 0.499;
    }
    initial.velocities[1] = -1.0;
    let model = MachineDynamics::assemble(&creation, &initial.poses, &initial.coordinates).unwrap();
    let mut query = scene
        .contacts(&geometry, &model.poses, DVec3::ZERO)
        .unwrap();
    query
        .contacts
        .retain(|point| point.body == 1 && point.body_point.z > 0.1);
    assert!(!query.contacts.is_empty());
    for point in &mut query.contacts {
        point.response = [0.0; 4];
    }
    let impact = query
        .impact_constraints(&model, &initial.velocities, 1.0, 1e-7)
        .unwrap();
    let mut free = initial.clone();
    activate(
        &creation,
        &model,
        &creation.coordinate_drives,
        &mut free,
        &impact,
        JointTickConfig::default(),
        &mut JointTickDiagnostics::default(),
    )
    .unwrap();
    let row = creation.dynamics.coordinate_velocities[0];
    assert!(free.velocities[row] < -0.1);
    let mut drives = creation.coordinate_drives.clone();
    drives[0].min_angle = 0.0;
    let mut stopped = initial.clone();
    let mut diagnostics = JointTickDiagnostics::default();
    activate(
        &creation,
        &model,
        &drives,
        &mut stopped,
        &impact,
        JointTickConfig::default(),
        &mut diagnostics,
    )
    .unwrap();
    assert!(stopped.velocities[row] >= -1e-9);
    assert!(
        stopped.velocities[3].abs() > 1e-3,
        "stop must transmit angular reaction to root"
    );
    assert_eq!(stopped.coordinates, initial.coordinates);
    assert_eq!(stopped.poses, initial.poses);
    assert_eq!(diagnostics.impact_events, 1);
    for point in &query.contacts {
        let row = model
            .point_row(point.body, point.body_point, point.normal)
            .unwrap();
        let outgoing = row
            .iter()
            .zip(&stopped.velocities)
            .map(|(j, v)| j * v)
            .sum::<f64>();
        assert!(outgoing >= -1e-8);
    }
}
