//! Pure input timing and pose interpolation for held constructions.

use bevy::prelude::{Quat, Vec3};
use mechanic_gpu::GpuTransform;

/// Repeats one-block height requests while exactly one direction remains held.
#[derive(Clone, Default)]
pub(crate) struct HeightRepeat {
    direction: i32,
    remaining: f64,
}

impl HeightRepeat {
    pub(crate) fn advance(&mut self, delta_seconds: f32, up: bool, down: bool) -> i32 {
        let direction = i32::from(up) - i32::from(down);
        if direction == 0 {
            self.reset();
            return 0;
        }
        if direction != self.direction {
            self.direction = direction;
            self.remaining = 0.3;
            return direction;
        }
        if delta_seconds.is_finite() {
            self.remaining -= f64::from(delta_seconds.max(0.0));
        }
        let mut steps = 0;
        while self.remaining <= 1.0e-8 {
            steps += direction;
            self.remaining += 0.1;
        }
        steps
    }

    pub(crate) fn reset(&mut self) {
        self.direction = 0;
        self.remaining = 0.0;
    }
}

/// Nearest yaw quarter-turn, preserving the body's projected forward direction.
/// When forward points vertically, its right vector determines the heading.
pub(crate) fn cardinal_heading(rotation: Quat) -> u8 {
    let forward = rotation * Vec3::NEG_Z;
    let yaw = if forward.x * forward.x + forward.z * forward.z > 1.0e-8 {
        (-forward.x).atan2(-forward.z)
    } else {
        let right = rotation * Vec3::X;
        (-right.z).atan2(right.x)
    };
    let sector = (yaw + std::f32::consts::FRAC_PI_4).rem_euclid(std::f32::consts::TAU);
    if sector < std::f32::consts::FRAC_PI_2 {
        0
    } else if sector < std::f32::consts::PI {
        1
    } else if sector < 3.0 * std::f32::consts::FRAC_PI_2 {
        2
    } else {
        3
    }
}

/// Rounds a global height upward onto the 25 cm construction grid.
pub(crate) fn grid_ceiling(height: f64) -> f64 {
    (height * 4.0).ceil() / 4.0
}

fn angular_error(current: Quat, target: Quat) -> f32 {
    let relative = target * current.conjugate();
    2.0 * Vec3::new(relative.x, relative.y, relative.z)
        .length()
        .atan2(relative.w.abs())
}

/// Exponential pose interpolation with a 100 ms time constant and exact settle.
pub(crate) fn smooth_pose(
    current: GpuTransform,
    target: GpuTransform,
    dt: f32,
) -> (GpuTransform, bool) {
    let position = Vec3::new(
        current.position[0],
        current.position[1],
        current.position[2],
    );
    let destination = Vec3::new(target.position[0], target.position[1], target.position[2]);
    let rotation = Quat::from_array(current.rotation);
    let orientation = Quat::from_array(target.rotation);
    let alpha = if dt.is_finite() {
        -(-dt.max(0.0) / 0.1).exp_m1()
    } else {
        0.0
    };
    let mut next_position = position.lerp(destination, alpha);
    // At world coordinates, the remaining exponential step can round to zero
    // before reaching the absolute settle tolerance. Finish those axes exactly
    // so a queued release cannot wait forever for an unrepresentable step.
    if alpha > 0.0 {
        for axis in 0..3 {
            if next_position[axis].to_bits() == position[axis].to_bits() {
                next_position[axis] = destination[axis];
            }
        }
    }
    let position = next_position;
    let rotation = rotation.slerp(orientation, alpha).normalize();
    if position.distance(destination) <= 1.0e-4 && angular_error(rotation, orientation) <= 1.0e-4 {
        return (target, true);
    }
    (
        GpuTransform {
            position: position.extend(0.0).to_array(),
            rotation: rotation.to_array(),
        },
        false,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose(position: Vec3, rotation: Quat) -> GpuTransform {
        GpuTransform {
            position: position.extend(0.0).to_array(),
            rotation: rotation.to_array(),
        }
    }

    #[test]
    fn height_steps_start_immediately_then_repeat_after_delay() {
        let mut repeat = HeightRepeat::default();
        assert_eq!(repeat.advance(0.016, true, false), 1);
        assert_eq!(repeat.advance(0.299, true, false), 0);
        assert_eq!(repeat.advance(0.001, true, false), 1);
        assert_eq!(repeat.advance(0.1, true, false), 1);
        assert_eq!(repeat.advance(0.35, true, false), 3);
        assert_eq!(repeat.advance(0.05, true, false), 1);
    }

    #[test]
    fn opposite_arrows_release_and_reset_discard_repeat_progress() {
        let mut repeat = HeightRepeat::default();
        assert_eq!(repeat.advance(0.0, true, false), 1);
        assert_eq!(repeat.advance(0.29, true, false), 0);
        assert_eq!(repeat.advance(1.0, true, true), 0);
        assert_eq!(repeat.advance(1.0, false, false), 0);
        assert_eq!(repeat.advance(0.0, true, false), 1);
        assert_eq!(repeat.advance(0.1, true, false), 0);
        assert_eq!(repeat.advance(0.0, false, true), -1);
        assert_eq!(repeat.advance(0.3, false, true), -1);
        repeat.reset();
        assert_eq!(repeat.advance(0.0, false, true), -1);
        assert_eq!(repeat.advance(0.1, false, true), 0);
    }

    #[test]
    fn grid_heights_round_up_on_both_sides_of_zero() {
        for (height, expected) in [
            (-0.26, -0.25),
            (-0.25, -0.25),
            (-0.01, 0.0),
            (0.0, 0.0),
            (0.01, 0.25),
            (0.25, 0.25),
        ] {
            assert!((grid_ceiling(height) - expected).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn heading_selects_nearest_cardinal_across_boundaries() {
        for heading in 0..4_u8 {
            let yaw = f32::from(heading) * std::f32::consts::FRAC_PI_2;
            assert_eq!(cardinal_heading(Quat::from_rotation_y(yaw)), heading);
            assert_eq!(
                cardinal_heading(Quat::from_rotation_y(
                    yaw + std::f32::consts::FRAC_PI_4 - 0.001
                )),
                heading
            );
            assert_eq!(
                cardinal_heading(Quat::from_rotation_y(
                    yaw + std::f32::consts::FRAC_PI_4 + 0.001
                )),
                (heading + 1) % 4
            );
        }
    }

    #[test]
    fn overturned_and_vertical_creations_have_deterministic_headings() {
        for heading in 0..4_u8 {
            let yaw = Quat::from_rotation_y(f32::from(heading) * std::f32::consts::FRAC_PI_2);
            assert_eq!(
                cardinal_heading(yaw * Quat::from_rotation_z(std::f32::consts::PI)),
                heading
            );
            assert_eq!(
                cardinal_heading(yaw * Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
                heading
            );
            assert_eq!(
                cardinal_heading(yaw * Quat::from_rotation_x(std::f32::consts::PI)),
                (heading + 2) % 4
            );
        }
    }

    #[test]
    fn pose_smoothing_is_invariant_to_frame_subdivision() {
        let start = pose(Vec3::ZERO, Quat::IDENTITY);
        let target = pose(Vec3::new(1.0, 3.0, -2.0), Quat::from_rotation_y(2.0));
        let (whole, settled) = smooth_pose(start, target, 0.1);
        assert!(!settled);
        let mut split = start;
        for _ in 0..10 {
            split = smooth_pose(split, target, 0.01).0;
        }
        for (a, b) in whole.position.into_iter().zip(split.position) {
            assert!((a - b).abs() < 1.0e-5);
        }
        assert!(
            angular_error(
                Quat::from_array(whole.rotation),
                Quat::from_array(split.rotation)
            ) < 1.0e-5
        );
        assert!((whole.position[0] - (1.0 - (-1.0_f32).exp())).abs() < 1.0e-6);
    }

    #[test]
    fn smoothing_finishes_alignment_at_world_coordinates_before_release() {
        for origin in [64.0, 1024.0, -1024.0] {
            let target = pose(Vec3::splat(origin) + Vec3::Y * 0.25, Quat::IDENTITY);
            let mut current = pose(Vec3::splat(origin), Quat::from_rotation_y(0.3));
            let mut settled = false;
            for _ in 0..1440 {
                (current, settled) = smooth_pose(current, target, 1.0 / 144.0);
                if settled {
                    break;
                }
            }
            assert!(settled, "alignment never finished at {origin}: {current:?}");
            assert_eq!(current, target);
        }
    }

    #[test]
    fn smoothing_settles_to_the_exact_target_and_accepts_equivalent_quaternions() {
        let target = pose(Vec3::Y, Quat::from_rotation_y(1.0));
        let mut current = pose(Vec3::ZERO, Quat::IDENTITY);
        let mut settled = false;
        for _ in 0..200 {
            (current, settled) = smooth_pose(current, target, 0.01);
            if settled {
                break;
            }
        }
        assert!(settled);
        assert_eq!(current, target);
        let opposite = pose(Vec3::Y, -Quat::from_array(target.rotation));
        assert_eq!(smooth_pose(opposite, target, 0.0), (target, true));
    }
}
