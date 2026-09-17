use super::*;

const RADIUS: f64 = 0.5;
const HALF_LENGTH: f64 = 0.125;

// A wheel on its Z axle with its lowest line `height` above y = 0.
fn wheel(height: f64) -> ContactCylinder {
    ContactCylinder::new(DVec3::Y * (RADIUS + height), DVec3::Z, RADIUS, HALF_LENGTH).unwrap()
}

// Upward-facing rectangle from `low` to `high` corners at height y, split on its
// diagonal: the `low` corner's half first.
fn square(low: [f64; 2], high: [f64; 2], y: f64) -> [[DVec3; 3]; 2] {
    let corner = |x, z| DVec3::new(x, y, z);
    let first = corner(low[0], low[1]);
    let diagonal = corner(high[0], high[1]);
    [
        [first, corner(low[0], high[1]), diagonal],
        [first, diagonal, corner(high[0], low[1])],
    ]
}

fn separation_of(point: &TriangleContactPoint) -> f64 {
    (point.body_point - point.triangle_point).dot(point.normal)
}

fn inside_cylinder(cylinder: &ContactCylinder, point: DVec3, tolerance: f64) -> bool {
    let offset = point - cylinder.center();
    let axial = cylinder.axis().dot(offset);
    (offset - axial * cylinder.axis()).length() <= cylinder.radius() + tolerance
        && axial.abs() <= cylinder.half_length() + tolerance
}

#[test]
fn a_level_wheel_supports_along_its_lowest_line_at_its_true_radius() {
    let floor = super::super::tests::floor();
    let points = wheel(-0.001).triangle_contacts(floor, 0.0).unwrap();
    assert_eq!(points.len(), 2, "{points:?}");
    for point in &points {
        assert!((point.depth - 0.001).abs() < 1e-12, "{point:?}");
        assert!(point.normal.abs_diff_eq(DVec3::Y, 1e-12));
        assert!(point.body_point.x.abs() < 1e-12, "{point:?}");
        assert!((point.body_point.z.abs() - HALF_LENGTH).abs() < 1e-12);
        assert!(point.triangle_point.y.abs() < 1e-12);
    }
}

#[test]
fn a_wheel_over_a_triangle_seam_keeps_the_whole_line_at_one_depth() {
    let mut points = Vec::new();
    for triangle in square([-1.0, -1.0], [1.0, 1.0], 0.0) {
        points.extend(wheel(-0.002).triangle_contacts(triangle, 0.0).unwrap());
    }
    assert!(points.len() >= 3, "{points:?}");
    for point in &points {
        assert!((point.depth - 0.002).abs() < 1e-12, "{point:?}");
        assert!(point.body_point.x.abs() < 1e-12, "{point:?}");
    }
    let [low, high] = points
        .iter()
        .fold([f64::INFINITY, f64::NEG_INFINITY], |[lo, hi], p| {
            [lo.min(p.body_point.z), hi.max(p.body_point.z)]
        });
    assert!((low + HALF_LENGTH).abs() < 1e-12 && (high - HALF_LENGTH).abs() < 1e-12);
}

#[test]
fn a_kerb_edge_along_the_axle_holds_the_wheel_on_a_line_at_its_exact_depth() {
    // The kerb top starts 0.3 m ahead of the axle, 0.1 m up. There the circle
    // is 0.4 m below the axle, so an axle at 0.49 m overlaps the kerb by 1 cm.
    let wheel = ContactCylinder::new(DVec3::Y * 0.49, DVec3::Z, RADIUS, HALF_LENGTH).unwrap();
    let mut points = Vec::new();
    for triangle in square([0.3, -1.0], [2.0, 1.0], 0.1) {
        points.extend(wheel.triangle_contacts(triangle, 0.0).unwrap());
    }
    let deepest = points.iter().map(|p| p.depth).fold(0.0, f64::max);
    assert!((deepest - 0.01).abs() < 1e-9, "{points:?}");
    let at_edge = points
        .iter()
        .filter(|p| (p.depth - 0.01).abs() < 1e-9 && (p.triangle_point.x - 0.3).abs() < 1e-9)
        .map(|p| p.body_point.z)
        .collect::<Vec<_>>();
    assert!(
        at_edge.iter().any(|z| (z - HALF_LENGTH).abs() < 1e-9)
            && at_edge.iter().any(|z| (z + HALF_LENGTH).abs() < 1e-9),
        "{points:?}"
    );
}

#[test]
fn a_leaning_wheel_is_deepest_at_its_lower_rim() {
    let axis = DQuat::from_rotation_x(0.2) * DVec3::Z;
    let wheel = ContactCylinder::new(DVec3::Y * 0.5, axis, RADIUS, HALF_LENGTH).unwrap();
    let lowest = wheel.support(DVec3::NEG_Y);
    let points = wheel
        .triangle_contacts(super::super::tests::floor(), 0.0)
        .unwrap();
    let deepest = points
        .iter()
        .max_by(|a, b| a.depth.total_cmp(&b.depth))
        .unwrap();
    assert!((deepest.depth + lowest.y).abs() < 1e-12, "{points:?}");
    assert!(deepest.body_point.abs_diff_eq(lowest, 1e-12));
}

#[test]
fn a_cylinder_standing_on_its_cap_keeps_four_rim_points() {
    let drum = ContactCylinder::new(
        DVec3::Y * (HALF_LENGTH - 0.001),
        DVec3::Y,
        RADIUS,
        HALF_LENGTH,
    )
    .unwrap();
    let points = drum
        .triangle_contacts(super::super::tests::floor(), 0.0)
        .unwrap();
    assert_eq!(points.len(), 4, "{points:?}");
    for point in points {
        assert!((point.depth - 0.001).abs() < 1e-12, "{point:?}");
        assert!((point.body_point.with_y(0.0).length() - RADIUS).abs() < 1e-12);
    }
}

#[test]
fn a_separated_wheel_reports_points_only_within_the_margin() {
    let floor = super::super::tests::floor();
    let above = wheel(0.01);
    assert!(above.triangle_contacts(floor, 0.005).unwrap().is_empty());
    let points = above.triangle_contacts(floor, 0.02).unwrap();
    assert_eq!(points.len(), 2);
    for point in points {
        assert!(point.depth.abs() < f64::EPSILON);
        assert!((separation_of(&point) - 0.01).abs() < 1e-12);
    }
    for margin in [-1.0, f64::INFINITY, f64::NAN] {
        assert!(above.triangle_contacts(floor, margin).is_err());
    }
}

#[test]
fn a_side_anchor_stays_under_a_spinning_rolling_wheel() {
    let local = ContactCylinder::new(DVec3::ZERO, DVec3::Z, RADIUS, HALF_LENGTH).unwrap();
    let wheel = local
        .transformed(DVec3::Y * RADIUS, DQuat::IDENTITY)
        .unwrap();
    let bottom = DVec3::new(0.0, 0.0, 0.05);
    let anchor = wheel.anchor(DVec3::Y, bottom).unwrap();
    // Roll 0.3 rad forward: the centre moves, the support stays underneath.
    let rolled = local
        .transformed(DVec3::new(0.15, RADIUS, 0.0), DQuat::from_rotation_z(-0.3))
        .unwrap();
    let point = rolled.anchor_point(DVec3::Y, anchor).unwrap();
    assert!(
        point.abs_diff_eq(bottom + DVec3::X * 0.15, 1e-12),
        "{point}"
    );
}

#[test]
fn a_cap_contact_stays_put_as_the_cylinder_spins() {
    let local = ContactCylinder::new(DVec3::ZERO, DVec3::Y, RADIUS, HALF_LENGTH).unwrap();
    let floor = super::super::tests::floor();
    let normal = triangle_normal(floor).unwrap();
    // Standing on its lower cap, tipped just short of resting on a rim edge.
    let tilt = DQuat::from_rotation_x(0.03);
    let pose = |spin: f64| {
        local
            .transformed(
                DVec3::Y * (HALF_LENGTH - 0.001),
                tilt * DQuat::from_rotation_y(spin),
            )
            .unwrap()
    };
    let standing = pose(0.0);
    let rims = standing.triangle_contacts(floor, 0.05).unwrap();
    assert!(rims.len() >= 4, "{rims:?}");
    let anchors = rims
        .iter()
        .map(|rim| standing.anchor(normal, rim.body_point).unwrap())
        .collect::<Vec<_>>();
    let spun = pose(1.1);
    for (rim, &anchor) in rims.iter().zip(&anchors) {
        let point = spun.anchor_point(normal, anchor).unwrap();
        assert!(point.abs_diff_eq(rim.body_point, 1e-12), "{point} {rim:?}");
    }

    // A point pressed into a lying wheel's cap face, against an oblique surface.
    let wheel = ContactCylinder::new(DVec3::ZERO, DVec3::Z, RADIUS, HALF_LENGTH).unwrap();
    let oblique = DVec3::new(0.3, 1.0, 0.4).normalize();
    let face = DVec3::new(0.1, -0.2, HALF_LENGTH);
    let anchor = wheel.anchor(oblique, face).unwrap();
    let spun = wheel
        .transformed(DVec3::ZERO, DQuat::from_rotation_z(2.0))
        .unwrap();
    let point = spun.anchor_point(oblique, anchor).unwrap();
    assert!(point.abs_diff_eq(face, 1e-12), "{point}");
    // Once tipped onto its cap, a side anchor no longer places a point.
    let side = wheel.anchor(DVec3::Y, DVec3::NEG_Y * RADIUS).unwrap();
    let tipped = wheel
        .transformed(
            DVec3::ZERO,
            DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2),
        )
        .unwrap();
    assert!(tipped.anchor_point(DVec3::Y, side).is_none());
}

// Deterministic stream in [0, 1).
struct Stream(u64);

impl Stream {
    #[allow(clippy::cast_precision_loss)] // 53 random bits are exact.
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next()
    }

    fn unit(&mut self) -> DVec3 {
        loop {
            let v = DVec3::new(
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            );
            if let Some(unit) = v.try_normalize().filter(|_| v.length() <= 1.0) {
                return unit;
            }
        }
    }
}

// Lowest cylinder entry above a dense grid of triangle points, found only from
// the inside test: an independent upper bound on the exact minimum.
#[allow(clippy::cast_precision_loss)] // Small grid counts.
fn sampled_minimum(cylinder: &ContactCylinder, triangle: [DVec3; 3]) -> Option<f64> {
    const GRID: usize = 48;
    const SCAN: usize = 160;
    let normal = triangle_normal(triangle).unwrap();
    let reach = cylinder.radius() + cylinder.half_length();
    let plane = normal.dot(triangle[0]);
    let middle = normal.dot(cylinder.center()) - plane;
    let mut best: Option<f64> = None;
    for i in 0..=GRID {
        for j in 0..=GRID - i {
            let [s, t] = [i as f64 / GRID as f64, j as f64 / GRID as f64];
            let foot =
                triangle[0] + s * (triangle[1] - triangle[0]) + t * (triangle[2] - triangle[0]);
            let (low, high) = (middle - reach, middle + reach);
            let Some(hit) = (0..=SCAN)
                .map(|k| low + (high - low) * k as f64 / SCAN as f64)
                .find(|&h| inside_cylinder(cylinder, foot + h * normal, 0.0))
            else {
                continue;
            };
            let [mut outside, mut inside] = [hit - (high - low) / SCAN as f64, hit];
            for _ in 0..60 {
                let middle = 0.5 * (outside + inside);
                if inside_cylinder(cylinder, foot + middle * normal, 0.0) {
                    inside = middle;
                } else {
                    outside = middle;
                }
            }
            best = Some(best.map_or(inside, |b: f64| b.min(inside)));
        }
    }
    best
}

#[test]
fn contacts_are_real_surface_points_and_never_miss_a_lower_one() {
    let mut stream = Stream(7);
    let mut compared = 0;
    for case in 0..160 {
        let radius = stream.range(0.1, 1.0);
        let half_length = stream.range(0.05, 0.6);
        let cylinder =
            ContactCylinder::new(DVec3::ZERO, stream.unit(), radius, half_length).unwrap();
        // Triangles of varied size near the cylinder, many crossing its edges.
        let center = stream.unit() * stream.range(0.0, radius + half_length);
        let size = stream.range(0.1, 2.0);
        let triangle = [0, 1, 2].map(|_| center + stream.unit() * size);
        if triangle_normal(triangle).is_err() {
            continue;
        }
        let normal = triangle_normal(triangle).unwrap();
        let points = cylinder.triangle_contacts(triangle, 10.0).unwrap();
        for point in &points {
            let tolerance = 1e-7;
            assert!(
                inside_cylinder(&cylinder, point.body_point, tolerance),
                "case {case}: {point:?}"
            );
            assert!(
                !inside_cylinder(&cylinder, point.body_point - 1e-5 * normal, 0.0),
                "case {case}: not the column entry {point:?}"
            );
            let barycentric = inside(triangle, normal, point.triangle_point)
                || (0..3).all(|edge| {
                    let from = triangle[edge];
                    let inward = normal.cross(triangle[(edge + 1) % 3] - from);
                    inward.dot(point.triangle_point - from) >= -tolerance * inward.length()
                });
            assert!(barycentric, "case {case}: outside the triangle {point:?}");
        }
        if let Some(sampled) = sampled_minimum(&cylinder, triangle) {
            compared += 1;
            let found = points.iter().map(separation_of).reduce(f64::min);
            assert!(
                found.is_some_and(|found| found <= sampled + 1e-6),
                "case {case}: exact {found:?} above sampled {sampled}"
            );
        }
    }
    assert!(compared > 60, "only {compared} cases met the cylinder");
}

// Distance from a point to the solid cylinder.
fn point_distance(cylinder: &ContactCylinder, point: DVec3) -> f64 {
    let offset = point - cylinder.center();
    let axial = cylinder.axis().dot(offset);
    let radial = (offset - axial * cylinder.axis()).length();
    (axial.abs() - cylinder.half_length())
        .max(0.0)
        .hypot((radial - cylinder.radius()).max(0.0))
}

#[test]
fn a_wheel_over_level_ground_has_its_exact_clearance() {
    // Triangles with an edge along the ground ahead of the wheel and beside its cap.
    let ahead = [
        DVec3::new(0.3, 0.0, -1.0),
        DVec3::new(0.3, 0.0, 1.0),
        DVec3::new(0.6, 0.0, 0.0),
    ];
    let beside = [
        DVec3::new(1.0, 0.0, HALF_LENGTH + 0.1),
        DVec3::new(-1.0, 0.0, HALF_LENGTH + 0.1),
        DVec3::new(0.0, 0.0, 1.0),
    ];
    for height in [0.0, -0.001] {
        let expected = (RADIUS + height).hypot(0.3) - RADIUS;
        let clearance = wheel(height).triangle_clearance(ahead);
        assert!(
            (clearance - expected).abs() < 2e-6,
            "{clearance} ≠ {expected}"
        );
        let clearance = wheel(height).triangle_clearance(beside);
        assert!((clearance - 0.1).abs() < 2e-6, "{clearance}");
    }
    let floor = super::super::tests::floor();
    assert!(wheel(-0.001).triangle_clearance(floor) <= 0.0);
    assert!((wheel(0.2).triangle_clearance(floor) - 0.2).abs() < 2e-6);
}

#[test]
fn clearance_never_exceeds_the_distance_or_a_contact_separation() {
    const GRID: usize = 64;
    let mut stream = Stream(11);
    let mut separated = 0;
    for case in 0..200 {
        let radius = stream.range(0.1, 1.0);
        let half_length = stream.range(0.05, 0.6);
        let cylinder =
            ContactCylinder::new(DVec3::ZERO, stream.unit(), radius, half_length).unwrap();
        let center = stream.unit() * stream.range(0.0, 2.0 * (radius + half_length));
        let size = stream.range(0.1, 2.0);
        let triangle = [0, 1, 2].map(|_| center + stream.unit() * size);
        if triangle_normal(triangle).is_err() {
            continue;
        }
        let clearance = cylinder.triangle_clearance(triangle);
        separated += usize::from(clearance > 0.0);
        #[allow(clippy::cast_precision_loss)] // Small grid counts.
        let sampled = (0..=GRID)
            .flat_map(|i| (0..=GRID - i).map(move |j| [i, j]))
            .map(|[i, j]| {
                let [s, t] = [i as f64 / GRID as f64, j as f64 / GRID as f64];
                let point =
                    triangle[0] + s * (triangle[1] - triangle[0]) + t * (triangle[2] - triangle[0]);
                point_distance(&cylinder, point)
            })
            .fold(f64::INFINITY, f64::min);
        assert!(
            clearance <= sampled,
            "case {case}: clearance {clearance} above distance {sampled}"
        );
        let normal = triangle_normal(triangle).unwrap();
        let lowest = normal.dot(cylinder.center() - triangle[0])
            - radius * (normal - normal.dot(cylinder.axis()) * cylinder.axis()).length()
            - half_length * normal.dot(cylinder.axis()).abs();
        let column = cylinder.column_clearance(triangle, normal, lowest);
        let sphere = cylinder.sphere_clearance(triangle, normal);
        for point in cylinder.triangle_contacts(triangle, 10.0).unwrap() {
            let gap = separation_of(&point).max(0.0);
            assert!(
                column <= gap && sphere <= gap,
                "case {case}: clearance {column} or {sphere} above contact {point:?}"
            );
        }
    }
    assert!(separated > 60, "only {separated} separated cases");
}
