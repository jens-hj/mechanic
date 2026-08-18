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
pub struct SatContact {
    /// Unit normal pointing from the first OBB toward the second.
    pub normal: Vec3,
    /// Non-negative overlap along the minimum axis.
    pub penetration: f32,
}

/// One persistent-manifold candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactPoint {
    /// World-space contact position.
    pub position: Vec3,
    /// Overlap associated with this point.
    pub penetration: f32,
}

/// Deterministically reduced contact manifold with at most four points.
#[derive(Clone, Debug, PartialEq)]
pub struct ContactManifold {
    /// Normal shared by every point, directed from A to B.
    pub normal: Vec3,
    /// Stable contact points. Face contacts retain up to four corners.
    pub points: Vec<ContactPoint>,
}

/// Runs OBB-vs-OBB separating-axis tests for three face axes from each box and
/// nine edge cross products.
pub fn obb_sat(a: Obb, b: Obb) -> Option<SatContact> {
    let a_axes = axes(a.orientation);
    let b_axes = axes(b.orientation);
    let center_delta = b.center - a.center;
    let mut minimum = SatContact {
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

/// Builds a small CPU reference manifold from the SAT result. Vertex-in-box
/// candidates cover face contacts; an edge-contact fallback guarantees one point.
pub fn obb_contact_manifold(a: Obb, b: Obb) -> Option<ContactManifold> {
    let sat = obb_sat(a, b)?;
    let mut candidates = Vec::with_capacity(8);
    for vertex in vertices(a) {
        if contains_point(b, vertex) {
            push_unique(&mut candidates, vertex);
        }
    }
    for vertex in vertices(b) {
        if contains_point(a, vertex) {
            push_unique(&mut candidates, vertex);
        }
    }
    if candidates.is_empty() {
        let point_a = support_point(a, sat.normal);
        let point_b = support_point(b, -sat.normal);
        candidates.push((point_a + point_b) * 0.5);
    }
    candidates.sort_by(|left, right| {
        left.x
            .total_cmp(&right.x)
            .then(left.y.total_cmp(&right.y))
            .then(left.z.total_cmp(&right.z))
    });
    candidates.truncate(4);
    Some(ContactManifold {
        normal: sat.normal,
        points: candidates
            .into_iter()
            .map(|position| ContactPoint {
                position,
                penetration: sat.penetration,
            })
            .collect(),
    })
}

fn test_axis(
    a: Obb,
    b: Obb,
    center_delta: Vec3,
    axis: Vec3,
    minimum: &mut SatContact,
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

fn vertices(obb: Obb) -> [Vec3; 8] {
    let basis = axes(obb.orientation);
    let x = basis[0] * obb.half_extents.x;
    let y = basis[1] * obb.half_extents.y;
    let z = basis[2] * obb.half_extents.z;
    [
        obb.center - x - y - z,
        obb.center - x - y + z,
        obb.center - x + y - z,
        obb.center - x + y + z,
        obb.center + x - y - z,
        obb.center + x - y + z,
        obb.center + x + y - z,
        obb.center + x + y + z,
    ]
}

fn contains_point(obb: Obb, point: Vec3) -> bool {
    let local = obb.orientation.conjugate() * (point - obb.center);
    local.x.abs() <= obb.half_extents.x + 1.0e-5
        && local.y.abs() <= obb.half_extents.y + 1.0e-5
        && local.z.abs() <= obb.half_extents.z + 1.0e-5
}

fn support_point(obb: Obb, direction: Vec3) -> Vec3 {
    let basis = axes(obb.orientation);
    let extents = [obb.half_extents.x, obb.half_extents.y, obb.half_extents.z];
    basis
        .into_iter()
        .zip(extents)
        .fold(obb.center, |point, (axis, extent)| {
            point + axis * extent * axis.dot(direction).signum()
        })
}

fn push_unique(points: &mut Vec<Vec3>, candidate: Vec3) {
    if points
        .iter()
        .all(|point| point.distance_squared(candidate) > 1.0e-10)
    {
        points.push(candidate);
    }
}

#[cfg(test)]
mod tests {
    use bevy_math::{Quat, Vec3};

    use super::{Obb, obb_contact_manifold, obb_sat};

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
    fn face_contact_retains_four_stable_points() {
        let manifold =
            obb_contact_manifold(cube(Vec3::ZERO), cube(Vec3::new(1.0, 0.0, 0.0))).unwrap();
        assert_eq!(manifold.points.len(), 4);
        assert!(
            manifold
                .points
                .iter()
                .all(|point| (point.position.x - 0.5).abs() < 1.0e-6)
        );
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
