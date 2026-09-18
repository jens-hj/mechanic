use super::*;
use crate::contact_geometry::tests::{cube, floor};
use bevy_math::DQuat;

#[test]
#[expect(clippy::float_cmp, reason = "exact full-interval sentinel")]
fn tangential_translation_does_not_consume_normal_clearance() {
    let origin = DVec3::Y * (0.5 + 1e-8);
    let shape = cube().transformed(origin, DQuat::IDENTITY).unwrap();
    let velocity = ContactVelocity::new(DVec3::X * 100.0, DVec3::ZERO);
    assert_eq!(
        shape
            .triangle_motion_prefix(floor(), origin, velocity, 0.0, 1.0)
            .unwrap(),
        1.0
    );
}

#[test]
fn quadratic_certificate_stops_before_a_known_accelerated_impact() {
    let origin = DVec3::Y;
    let shape = cube().transformed(origin, DQuat::IDENTITY).unwrap();
    let velocity = ContactVelocity::new(DVec3::new(100.0, -1.0, 0.0), DVec3::ZERO);
    let prefix = shape
        .triangle_motion_prefix(floor(), origin, velocity, 2.0, 1.0)
        .unwrap();
    let exact = (3.0_f64.sqrt() - 1.0) * 0.5;
    assert!(prefix <= exact);
    assert!(exact - prefix < 1e-12);
}

#[test]
fn departing_endpoints_do_not_hide_an_intermediate_acceleration_excursion() {
    let origin = DVec3::Y * 0.6;
    let shape = cube().transformed(origin, DQuat::IDENTITY).unwrap();
    let velocity = ContactVelocity::new(-DVec3::Y, DVec3::ZERO);
    // y(t) = 0.6 - t + t² returns to its start but crosses the floor twice.
    // The magnitude bound must reject the complete interval despite its clear end.
    let prefix = shape
        .triangle_motion_prefix(floor(), origin, velocity, 2.0, 1.0)
        .unwrap();
    let first_impact = (1.0 - 0.6_f64.sqrt()) * 0.5;
    assert!(prefix > 0.0 && prefix < first_impact);
}

#[test]
fn signed_rigid_velocities_preserve_cancellation_at_the_support_feature() {
    let origin = DVec3::Y * (0.5 + 1e-8);
    let shape = cube().transformed(origin, DQuat::IDENTITY).unwrap();
    // At the right bottom edge, -0.5 Y translation cancels Z rotation exactly.
    // At the left edge they add: the certificate must consider every vertex.
    let velocity = ContactVelocity::new(-DVec3::Y * 0.5, DVec3::Z);
    let prefix = shape
        .triangle_motion_prefix(floor(), origin, velocity, 1.0, 1.0)
        .unwrap();
    assert!(prefix > 0.0 && prefix <= 1.000_001e-8);
    for invalid in [-1.0, f64::INFINITY, f64::NAN] {
        assert!(
            shape
                .triangle_motion_prefix(floor(), origin, velocity, invalid, 1.0)
                .is_err()
        );
    }
}
