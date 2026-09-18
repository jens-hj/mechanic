use bevy::prelude::{Quat, Vec3};

use crate::sequencer::signed_joint_speed;

#[test]
fn snapshot_pairs_measure_signed_relative_joint_speed() {
    let delta_seconds = 0.1;
    let positive = signed_joint_speed(
        Quat::IDENTITY,
        Quat::IDENTITY,
        Quat::IDENTITY,
        Quat::from_axis_angle(Vec3::Z, 0.2),
        Vec3::Z,
        delta_seconds,
    );
    let negative = signed_joint_speed(
        Quat::IDENTITY,
        Quat::IDENTITY,
        Quat::IDENTITY,
        Quat::from_axis_angle(Vec3::Z, -0.2),
        Vec3::Z,
        delta_seconds,
    );
    assert!((positive - 2.0).abs() < 1.0e-5);
    assert!((negative + 2.0).abs() < 1.0e-5);
}
