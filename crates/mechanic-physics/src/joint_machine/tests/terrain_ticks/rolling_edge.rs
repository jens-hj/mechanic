//! Frictionless rotating-edge support with an independent constrained-body ODE.

use super::*;
use bevy_math::DQuat;

fn setup() -> (CompiledCreation, MachineState, crate::TerrainContactQuery) {
    let (creation, geometry, _) = cube();
    let mut state = MachineState::at_rest(&creation);
    let angle = 0.2;
    let omega = 2.0;
    state.poses[0].rotation = DQuat::from_rotation_z(angle);
    let arm = state.poses[0].rotation * DVec3::new(-0.5, -0.5, 0.0);
    state.poses[0].position = DVec3::Y * (-arm.y);
    state.velocities[1] = -omega * arm.x;
    state.velocities[5] = omega;
    let mut contacts = scene()
        .activation_contacts(&geometry, &state.poses, DVec3::ZERO)
        .unwrap();
    assert!(contacts.contacts.len() >= 2);
    for contact in &mut contacts.contacts {
        contact.response = [0.0; 4];
        assert!(
            (contact.body_point.x - arm.x).abs() < 1e-10,
            "point={:?} arm={arm:?} center={:?}",
            contact.body_point,
            creation.dynamics.inertias[0].center
        );
    }
    (creation, state, contacts)
}

fn normal_force(creation: &CompiledCreation, angle: f64, omega: f64) -> f64 {
    let mass = f64::from(creation.dynamics.inertias[0].mass);
    let inertia = f64::from(creation.dynamics.inertias[0].rotational.z_axis.z);
    let arm = DQuat::from_rotation_z(angle) * DVec3::new(-0.5, -0.5, 0.0);
    // a_y = N/m-g, alpha = r_x N/I and the supported material point obeys
    // 0 = a_y + alpha*r_x - omega^2*r_y. This includes changing contact J.
    (9.81 + omega * omega * arm.y) / (mass.recip() + arm.x * arm.x / inertia)
}

#[test]
fn rotating_edge_support_force_converges_to_constrained_rigid_body_equation() {
    let (creation, initial, contacts) = setup();
    let expected = normal_force(&creation, 0.2, 2.0);
    let mass = f64::from(creation.dynamics.inertias[0].mass);
    for dt in [1e-3, 1e-4, 1e-5] {
        let mut state = initial.clone();
        let mut diagnostics = JointTickDiagnostics::default();
        let outcome = substep(
            &creation,
            &[],
            &[],
            &mut state,
            -DVec3::Y * 9.81,
            dt,
            fixed(1),
            Some(SubstepContacts {
                query: &contacts,
                restitution_threshold: 1.0,
                stiction_threshold: 1e-7,
            }),
            None,
            &mut diagnostics,
        )
        .unwrap();
        assert!(matches!(outcome, events::TrialOutcome::Complete { .. }));
        let actual = mass * ((state.velocities[1] - initial.velocities[1]) / dt + 9.81);
        let relative_error = (actual - expected).abs() / expected;
        println!(
            "rolling_edge dt={dt} force={actual} reference={expected} relative_error={relative_error}"
        );
        assert!(
            relative_error < 5.0 * dt,
            "rotating support omits point acceleration: dt={dt} actual={actual} expected={expected}"
        );
    }
}

#[test]
fn rotating_support_uses_the_normal_velocity_at_its_reconstructed_end_pose() {
    let (creation, initial, contacts) = setup();
    let mut state = initial.clone();
    let mut diagnostics = JointTickDiagnostics::default();
    substep(
        &creation,
        &[],
        &[],
        &mut state,
        -DVec3::Y * 9.81,
        1e-3,
        fixed(1),
        Some(SubstepContacts {
            query: &contacts,
            restitution_threshold: 1.0,
            stiction_threshold: 1e-7,
        }),
        None,
        &mut diagnostics,
    )
    .unwrap();
    for point in &contacts.contacts {
        let initial_pose = initial.poses[point.body];
        let local = initial_pose.rotation.inverse() * (point.body_point - initial_pose.position);
        let arm = state.poses[point.body].rotation * local;
        // This cube's body origin is its COM; compute the material-point velocity
        // directly, independently of the solver's Jacobian or bias implementation.
        let linear = DVec3::from_slice(&state.velocities[..3]);
        let angular = DVec3::from_slice(&state.velocities[3..6]);
        let normal_speed = point.normal.dot(linear + angular.cross(arm));
        assert!(
            normal_speed.abs() < 1e-8,
            "end support speed={normal_speed:e}"
        );
    }
}

#[test]
fn a_rotating_released_edge_does_not_report_a_frozen_row_reversal() {
    let (creation, mut initial, contacts) = setup();
    let arm = initial.poses[0].rotation * DVec3::new(-0.5, -0.5, 0.0);
    let omega = 5.0;
    initial.velocities[1] = -omega * arm.x + 1e-3;
    initial.velocities[5] = omega;
    let model = MachineDynamics::assemble(&creation, &initial.poses, &initial.coordinates).unwrap();
    let surface =
        events::SustainingSurface::new(&contacts, &model, &initial.velocities, 1e-8).unwrap();
    assert!(surface.query.contacts.is_empty());
    let dt = 1e-3;
    let mut outgoing = initial.velocities.clone();
    outgoing[1] -= 9.81 * dt;
    let mut diagnostics = JointTickDiagnostics::default();
    assert!(
        surface
            .reversal(
                &creation,
                &initial,
                &outgoing,
                dt,
                1e-8,
                0.0,
                &mut diagnostics
            )
            .unwrap()
            .is_none()
    );
    for sample in 0..=100 {
        let time = dt * f64::from(sample) / 100.0;
        let rotated = DQuat::from_rotation_z(omega * time) * arm;
        let speed = initial.velocities[1] - 9.81 * time + omega * rotated.x;
        let height = initial.poses[0].position.y + initial.velocities[1] * time
            - 0.5 * 9.81 * time * time
            + rotated.y;
        assert!(
            speed > 0.0 && height >= -1e-15,
            "time={time:e} speed={speed:e} height={height:e}"
        );
    }
    let frozen_speed = outgoing[1] + omega * arm.x;
    assert!(
        frozen_speed < -1e-3,
        "fixture must expose the frozen-row error"
    );
}
