use super::*;
use bevy_math::Quat;

fn face(normal: Vec3) -> FaceGeometry {
    FaceGeometry {
        center: Vec3::ZERO,
        normal,
        tangent_u: Vec3::X,
        tangent_v: Vec3::Z,
        profile: FaceProfile::Rectangle {
            half_u: 0.125,
            half_v: 0.125,
        },
    }
}

#[test]
fn touching_faces_accept_normal_length_roundoff() {
    let first = face(Vec3::Y * (1.0 - f32::EPSILON));
    let second = face(-Vec3::Y * (1.0 - f32::EPSILON));
    assert!(faces_touch(&first, &second));
}

#[test]
fn touching_faces_enforce_the_authored_angular_tolerance() {
    let first = face(Vec3::Y);
    for (factor, expected) in [(0.5, true), (2.0, false)] {
        let angle = AXIS_TOLERANCE_DEGREES.to_radians() * factor;
        let second = face(Quat::from_rotation_x(angle) * -Vec3::Y);
        assert_eq!(faces_touch(&first, &second), expected);
    }
    assert!(!faces_touch(&first, &first));
}
