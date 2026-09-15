mod activation_vertex;
use super::*;
use crate::{BuildCommand, BuildPose, CompiledConvex, ConstructionGraph, CuboidSpec};

pub(super) fn cube() -> ContactPolytope {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([4, 4, 4], BuildPose::default()).unwrap(),
        ))
        .unwrap();
    ContactPolytope::from_collider(&graph.compile().unwrap().colliders[0]).unwrap()
}

pub(super) fn floor() -> [DVec3; 3] {
    [
        DVec3::new(-10.0, 0.0, -10.0),
        DVec3::new(0.0, 0.0, 10.0),
        DVec3::new(10.0, 0.0, -10.0),
    ]
}

#[test]
fn resting_box_retains_four_finite_supports_with_measured_depth() {
    let body = cube()
        .transformed(DVec3::Y * 0.499, DQuat::IDENTITY)
        .unwrap();
    let points = body.triangle_contacts(floor()).unwrap();
    assert_eq!(points.len(), 4);
    for point in points {
        assert!((point.depth - 0.001).abs() < 1e-12);
        assert!(point.normal.abs_diff_eq(DVec3::Y, 1e-12));
        assert!(point.body_point.y.abs() <= 0.001 + 1e-12);
        assert!(point.triangle_point.y.abs() < 1e-12);
    }
}

#[test]
fn near_contacts_keep_true_surface_points_and_do_not_bridge_finite_edges() {
    let body = cube()
        .transformed(DVec3::Y * 0.500_001, DQuat::IDENTITY)
        .unwrap();
    assert!(body.triangle_contacts(floor()).unwrap().is_empty());
    assert!(body.triangle_proximity(floor(), 0.5e-6).unwrap().is_empty());
    let contacts = body.triangle_proximity(floor(), 2e-6).unwrap();
    assert_eq!(contacts.len(), 4);
    for point in contacts {
        assert!(point.depth.abs() < 1e-15);
        assert!((point.body_point.y - 1e-6).abs() < 1e-12);
        assert!(point.triangle_point.y.abs() < 1e-12);
    }
    let outside = [
        DVec3::new(0.501, 0.0, -1.0),
        DVec3::new(0.501, 0.0, 1.0),
        DVec3::new(2.0, 0.0, 0.0),
    ];
    assert!(body.triangle_proximity(outside, 0.1).unwrap().is_empty());
    for margin in [-1.0, f64::INFINITY, f64::NAN] {
        assert!(body.triangle_proximity(floor(), margin).is_err());
    }
}

#[test]
fn reused_proximity_geometry_tracks_pose_normal_margin_and_finite_edges() {
    let shapes = [
        cube()
            .transformed(DVec3::Y * 0.501, DQuat::IDENTITY)
            .unwrap(),
        cube()
            .transformed(DVec3::Y * 0.3, DQuat::from_rotation_z(0.37))
            .unwrap(),
    ];
    let triangles = [
        floor(),
        floor().map(|point| point + DVec3::Y * 0.1),
        [DVec3::X * 2.0, DVec3::X * 3.0 + DVec3::Z, DVec3::X * 3.0],
        floor().map(|point| DQuat::from_rotation_x(0.15) * point),
    ];
    let mut scratch = TriangleClipScratch::default();
    let mut hits = 0;
    let mut misses = 0;
    for shape in shapes.iter().cycle().take(4) {
        for margin in [0.02, 0.001, 0.0, 0.02] {
            for triangle in triangles.into_iter().cycle().take(8) {
                let expected = shape.triangle_proximity(triangle, margin).unwrap();
                for _ in 0..2 {
                    let actual = shape
                        .triangle_proximity_with_scratch(triangle, margin, &mut scratch)
                        .unwrap();
                    assert_eq!(actual, expected);
                    if actual.is_empty() {
                        misses += 1;
                    } else {
                        hits += 1;
                    }
                }
            }
        }
    }
    assert!(hits > 0 && misses > 0);
}

#[test]
fn oblique_proximity_extrusion_preserves_the_actual_convex_silhouette() {
    let rotation = DQuat::from_rotation_x(0.4) * DQuat::from_rotation_z(0.7);
    let body = cube().transformed(DVec3::Y, rotation).unwrap();
    let points = body.triangle_proximity(floor(), 2.0).unwrap();
    assert!(!points.is_empty());
    for point in points {
        let mut on_surface = false;
        for plane in &body.planes {
            let error = plane.truncate().dot(point.body_point) - plane.w;
            assert!(error <= 1e-12, "extrusion invented a body point: {error}");
            on_surface |= error.abs() < 1e-12;
        }
        assert!(on_surface);
        assert!((point.body_point - point.triangle_point).dot(point.normal) <= 2.0 + 1e-12);
    }
}

#[test]
fn finite_edges_holes_and_below_surface_separation_produce_no_contact() {
    let triangle = [DVec3::ZERO, DVec3::Z, DVec3::X];
    for position in [
        DVec3::new(4.0, 0.0, 0.0),
        DVec3::new(0.1, -2.0, 0.1),
        DVec3::new(0.1, 2.0, 0.1),
    ] {
        let body = cube().transformed(position, DQuat::IDENTITY).unwrap();
        assert!(body.triangle_contacts(triangle).unwrap().is_empty());
    }
    let body = cube()
        .transformed(DVec3::new(4.0, 2.0, 0.0), DQuat::IDENTITY)
        .unwrap();
    assert!(
        body.translation_interval(triangle, -DVec3::Y * 4.0)
            .unwrap()
            .is_none()
    );
}

#[test]
fn oblique_contacts_have_points_on_both_actual_surfaces() {
    let body = cube()
        .transformed(DVec3::Y * 0.2, DQuat::from_rotation_z(0.67))
        .unwrap();
    let triangle = [
        DVec3::new(-0.3, 0.0, -1.0),
        DVec3::new(0.0, 0.0, 1.0),
        DVec3::new(0.3, 0.0, -1.0),
    ];
    let contacts = body.triangle_contacts(triangle).unwrap();
    assert!(contacts.len() >= 3);
    for point in contacts {
        let mut on_surface = false;
        for plane in &body.planes {
            let error = plane.truncate().dot(point.body_point) - plane.w;
            assert!(error <= 1e-12, "body point outside convex: {error}");
            on_surface |= error.abs() < 1e-12;
        }
        assert!(on_surface);
        for edge in 0..3 {
            assert!(
                (triangle[(edge + 1) % 3] - triangle[edge])
                    .cross(point.triangle_point - triangle[edge])
                    .dot(point.normal)
                    >= -1e-12
            );
        }
    }
}

#[test]
fn continuous_translation_finds_fast_crossing_even_when_both_endpoints_are_clear() {
    let body = cube()
        .transformed(DVec3::Y * 10.0, DQuat::IDENTITY)
        .unwrap();
    assert!(body.triangle_contacts(floor()).unwrap().is_empty());
    let [entry, exit] = body
        .translation_interval(floor(), -DVec3::Y * 20.0)
        .unwrap()
        .unwrap();
    assert!((entry - 0.475).abs() < 1e-12);
    assert!((exit - 0.525).abs() < 1e-12);
    let impact = cube()
        .transformed(DVec3::Y * (10.0 - 20.0 * entry), DQuat::IDENTITY)
        .unwrap();
    assert_eq!(impact.triangle_contacts(floor()).unwrap().len(), 4);
}

#[test]
fn convex_plane_offsets_and_local_box_rotation_give_equivalent_queries() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([4, 4, 4], BuildPose::default()).unwrap(),
        ))
        .unwrap();
    let mut collider = graph.compile().unwrap().colliders[0].clone();
    let reference = cube();
    collider.shape = ColliderShape::Convex(CompiledConvex {
        vertices: reference.vertices.iter().map(|v| v.as_vec3()).collect(),
        face_planes: reference.planes.iter().map(|v| v.as_vec4()).collect(),
        edge_directions: reference.edges.iter().map(|v| v.as_vec3()).collect(),
    });
    let convex = ContactPolytope::from_collider(&collider).unwrap();
    let position = DVec3::new(0.1, 0.49, -0.1);
    let rotation = DQuat::from_rotation_z(0.1);
    let a = reference.transformed(position, rotation).unwrap();
    let b = convex.transformed(position, rotation).unwrap();
    assert_eq!(
        a.triangle_contacts(floor()).unwrap(),
        b.triangle_contacts(floor()).unwrap()
    );
    assert_eq!(
        a.translation_interval(floor(), DVec3::Y).unwrap(),
        b.translation_interval(floor(), DVec3::Y).unwrap()
    );
}

#[test]
fn invalid_triangle_and_motion_are_errors_instead_of_missing_contacts() {
    assert_eq!(
        cube().triangle_contacts([DVec3::ZERO; 3]),
        Err(ContactGeometryError)
    );
    assert_eq!(
        cube().translation_interval(floor(), DVec3::splat(f64::NAN)),
        Err(ContactGeometryError)
    );
}

#[test]
fn rotational_sweep_finds_an_impact_hidden_by_identical_endpoint_poses() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([16, 1, 1], BuildPose::default()).unwrap(),
        ))
        .unwrap();
    let bar = ContactPolytope::from_collider(&graph.compile().unwrap().colliders[0]).unwrap();
    let position = DVec3::Y;
    assert!(
        bar.transformed(position, DQuat::IDENTITY)
            .unwrap()
            .triangle_contacts(floor())
            .unwrap()
            .is_empty()
    );
    let motion = RigidContactSweep {
        translation: DVec3::ZERO,
        angular_displacement: DVec3::Z * std::f64::consts::TAU,
    };
    let SweepOutcome::Impact {
        fraction,
        separation,
        ..
    } = bar
        .rigid_triangle_sweep(floor(), position, DQuat::IDENTITY, motion, 1e-9, 128)
        .unwrap()
    else {
        panic!("full rotation must hit the floor");
    };
    let angle = (1.0 / 2.0_f64.hypot(0.125)).asin() - (0.125_f64 / 2.0).atan();
    assert!((fraction - angle / std::f64::consts::TAU).abs() < 1e-9);
    assert!(separation <= 1e-9);
    assert!(fraction > 0.0 && fraction < 0.25);
}

#[test]
fn conservative_sweep_matches_exact_translation_and_reports_exhaustion() {
    let position = DVec3::Y * 10.0;
    let motion = RigidContactSweep {
        translation: -DVec3::Y * 20.0,
        angular_displacement: DVec3::ZERO,
    };
    let SweepOutcome::Impact { fraction, .. } = cube()
        .rigid_triangle_sweep(floor(), position, DQuat::IDENTITY, motion, 1e-10, 16)
        .unwrap()
    else {
        panic!("translation crosses floor");
    };
    assert!((fraction - 0.475).abs() < 1e-12);
    assert!(matches!(
        cube()
            .rigid_triangle_sweep(floor(), position, DQuat::IDENTITY, motion, 1e-10, 1)
            .unwrap(),
        SweepOutcome::Unconverged { .. }
    ));
    let finite = [DVec3::ZERO, DVec3::Z, DVec3::X];
    assert!(matches!(
        cube()
            .rigid_triangle_sweep(
                finite,
                position + DVec3::X * 4.0,
                DQuat::IDENTITY,
                motion,
                1e-10,
                128
            )
            .unwrap(),
        SweepOutcome::Separated { .. }
    ));
}

#[test]
#[allow(clippy::unreadable_literal)] // Exact captured wheel geometry, preserving the failing floating-point inputs.
fn near_parallel_wheel_faces_keep_activation_points_inside_the_gap_bound() {
    let body = ContactPolytope {
        vertices: vec![
            DVec3::new(1.167303073688573, 0.4842541952894062, -4.874999994767269),
            DVec3::new(0.9950955323341613, 0.06850841723287254, -4.8749999843018035),
            DVec3::new(1.167303044458893, 0.4842542011035479, -5.1249999947672675),
            DVec3::new(0.9950955031044812, 0.06850842304701427, -5.124999984301802),
            DVec3::new(1.3326970221869952, 0.4157457780567515, -4.8750000156981885),
            DVec3::new(
                1.1604894808325832,
                2.1782575743145571e-13,
                -4.875000005232723,
            ),
            DVec3::new(1.3326969929573151, 0.41574578387089317, -5.125000015698187),
            DVec3::new(1.1604894516029032, 5.81435949387199e-9, -5.125000005232721),
        ],
        planes: vec![
            DVec4::new(
                0.38268343536967525,
                0.9238795312667465,
                -2.3256591506647993e-8,
                0.8941002026905399,
            ),
            DVec4::new(
                -0.38268343536967525,
                -0.9238795312667465,
                2.3256591506647993e-8,
                -0.44410021461146887,
            ),
            DVec4::new(
                1.1691872002910034e-7,
                -2.325656667664086e-8,
                0.9999999999999929,
                -4.874999869549743,
            ),
            DVec4::new(
                -1.1691872002910034e-7,
                2.325656667664086e-8,
                -0.9999999999999929,
                5.124999869549743,
            ),
            DVec4::new(
                -0.9238795312667393,
                0.3826834353696752,
                1.1691871507906294e-7,
                -0.8931319274988309,
            ),
            DVec4::new(
                0.9238795312667393,
                -0.3826834353696752,
                -1.1691871507906294e-7,
                1.0721530475702419,
            ),
        ],
        edges: vec![
            DVec3::new(
                -0.38268343536967525,
                -0.9238795312667465,
                2.3256591506647993e-8,
            ),
            DVec3::new(
                -1.1691872002910034e-7,
                2.325656667664086e-8,
                -0.9999999999999929,
            ),
            DVec3::new(
                0.9238795312667393,
                -0.3826834353696752,
                -1.1691871507906294e-7,
            ),
        ],
    };
    let triangle = [
        DVec3::new(-64.0, 0.0, -64.0),
        DVec3::new(64.0, 0.0, 64.0),
        DVec3::new(64.0, 0.0, -64.0),
    ];
    assert!(body.triangle_contacts(triangle).unwrap().is_empty());
    let points = body.triangle_activation_contacts(triangle, 1e-12).unwrap();
    assert!(!points.is_empty());
    for point in points {
        let gap = (point.body_point - point.triangle_point).dot(point.normal);
        assert!((0.0..=1e-12).contains(&gap), "gap={gap}");
        assert!(point.triangle_point.y.abs() < 1e-15);
        assert!(point.triangle_point.x >= point.triangle_point.z);
        assert!(point.triangle_point.x <= 64.0 && point.triangle_point.z >= -64.0);
        for plane in &body.planes {
            assert!(plane.truncate().dot(point.body_point) - plane.w < 1e-12);
        }
    }
}

#[test]
fn split_recovery_retains_a_tilted_box_vertex_without_changing_its_contact_manifold() {
    let rotation = DQuat::from_rotation_z(0.2);
    let minimum = cube().transformed(DVec3::ZERO, rotation).unwrap().bounds()[0].y;
    let shape = cube()
        .transformed(DVec3::Y * (-minimum - 0.001), rotation)
        .unwrap();
    let physical = shape.triangle_contacts(floor()).unwrap();
    assert_eq!(physical.len(), 4);
    let recovery = shape.triangle_recovery_contacts(floor()).unwrap();
    let deepest = recovery.iter().map(|point| point.depth).fold(0.0, f64::max);
    assert!((deepest - 0.001).abs() < 1e-12);
    assert_eq!(&recovery[..physical.len()], physical);
    let outside = [
        DVec3::new(2.0, 0.0, -1.0),
        DVec3::new(2.0, 0.0, 1.0),
        DVec3::new(4.0, 0.0, 0.0),
    ];
    assert!(
        shape
            .triangle_recovery_contacts(outside)
            .unwrap()
            .is_empty()
    );
}
