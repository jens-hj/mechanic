//! Oriented-box overlap on the CPU: a separating-axis test.

use bevy_math::{Quat, Vec3};

/// Oriented cuboid used by the CPU narrowphase reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obb {
    /// World-space centre.
    pub center: Vec3,
    /// World-space orientation.
    pub orientation: Quat,
    /// Positive local half extents.
    pub half_extents: Vec3,
}

/// Minimum-translation result from all 15 cuboid SAT axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbContact {
    /// Unit normal pointing from the first OBB toward the second.
    pub normal: Vec3,
    /// Non-negative overlap along the minimum axis.
    pub penetration: f32,
}

/// Runs OBB-vs-OBB separating-axis tests for three face axes from each box and
/// nine edge cross products.
pub fn obb_sat(a: Obb, b: Obb) -> Option<ObbContact> {
    let a_axes = axes(a.orientation);
    let b_axes = axes(b.orientation);
    let center_delta = b.center - a.center;
    let mut minimum = ObbContact {
        normal: Vec3::X,
        penetration: f32::INFINITY,
    };

    for axis in a_axes.into_iter().chain(b_axes) {
        test_axis(a, b, center_delta, axis, &mut minimum)?;
    }
    for a_axis in a_axes {
        for b_axis in b_axes {
            let cross = a_axis.cross(b_axis);
            if cross.length_squared() > 1.0e-10 {
                test_axis(a, b, center_delta, cross.normalize(), &mut minimum)?;
            }
        }
    }
    Some(minimum)
}

fn test_axis(
    a: Obb,
    b: Obb,
    center_delta: Vec3,
    axis: Vec3,
    minimum: &mut ObbContact,
) -> Option<()> {
    let radius_a = projection_radius(a, axis);
    let radius_b = projection_radius(b, axis);
    let signed_distance = center_delta.dot(axis);
    let penetration = radius_a + radius_b - signed_distance.abs();
    if penetration < -1.0e-6 {
        return None;
    }
    if penetration < minimum.penetration {
        minimum.penetration = penetration.max(0.0);
        minimum.normal = if signed_distance < 0.0 { -axis } else { axis };
    }
    Some(())
}

fn projection_radius(obb: Obb, axis: Vec3) -> f32 {
    let basis = axes(obb.orientation);
    basis[0].dot(axis).abs() * obb.half_extents.x
        + basis[1].dot(axis).abs() * obb.half_extents.y
        + basis[2].dot(axis).abs() * obb.half_extents.z
}

fn axes(orientation: Quat) -> [Vec3; 3] {
    [
        orientation * Vec3::X,
        orientation * Vec3::Y,
        orientation * Vec3::Z,
    ]
}

#[cfg(test)]
mod tests {
    use bevy_math::{Quat, Vec3};

    use super::{Obb, obb_sat};

    fn cube(center: Vec3) -> Obb {
        Obb {
            center,
            orientation: Quat::IDENTITY,
            half_extents: Vec3::splat(0.5),
        }
    }

    #[test]
    fn sat_rejects_separated_boxes() {
        assert!(obb_sat(cube(Vec3::ZERO), cube(Vec3::new(1.01, 0.0, 0.0))).is_none());
    }

    #[test]
    fn sat_finds_minimum_face_axis() {
        let contact = obb_sat(cube(Vec3::ZERO), cube(Vec3::new(0.75, 0.0, 0.0))).unwrap();
        assert!(contact.normal.abs_diff_eq(Vec3::X, 1.0e-6));
        assert!((contact.penetration - 0.25).abs() < 1.0e-6);
    }

    #[test]
    fn rotated_edge_overlap_is_detected() {
        let rotated = Obb {
            center: Vec3::new(0.8, 0.0, 0.0),
            orientation: Quat::from_rotation_y(core::f32::consts::FRAC_PI_4),
            half_extents: Vec3::splat(0.5),
        };
        assert!(obb_sat(cube(Vec3::ZERO), rotated).is_some());
    }
}
