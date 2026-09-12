use super::*;
use crate::contact_geometry::tests::{cube, floor};
use bevy_math::DQuat;

#[test]
fn envelope_bounds_analytic_box_depth_and_motion_reach_outward() {
    let body = cube()
        .transformed(DVec3::Y * 0.499, DQuat::IDENTITY)
        .unwrap();
    for displacement in [0.0, 1e-6, 0.002, 0.1] {
        let bound = body
            .triangle_penetration_bound(floor(), displacement)
            .unwrap()
            .unwrap();
        let exact = 0.5 - 0.499 + displacement;
        assert!(bound >= exact, "{bound} underestimates {exact}");
        assert!(bound - exact < 1e-12, "loose box bound: {bound}");
    }
    assert!(cube().conservative_radius().unwrap() >= 0.75_f64.sqrt());
    for displacement in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(
            body.triangle_penetration_bound(floor(), displacement)
                .is_err()
        );
    }
}

#[test]
fn finite_prism_excludes_deep_geometry_beyond_a_terrain_edge() {
    let body = cube()
        .transformed(
            DVec3::ZERO,
            DQuat::from_rotation_z(std::f64::consts::FRAC_PI_4),
        )
        .unwrap();
    let triangle = [
        DVec3::new(0.6, 0.0, -2.0),
        DVec3::new(0.6, 0.0, 2.0),
        DVec3::new(2.0, 0.0, 0.0),
    ];
    let bound = body
        .triangle_penetration_bound(triangle, 0.0)
        .unwrap()
        .unwrap();
    let exact = std::f64::consts::FRAC_1_SQRT_2 - 0.6;
    assert!(bound >= exact - 1e-15);
    assert!(bound < exact + 1e-12, "finite prism lost the edge: {bound}");
    assert!(
        body.triangle_penetration_bound(floor(), 0.0)
            .unwrap()
            .unwrap()
            > 0.7
    );
}

#[test]
fn certified_radius_detects_unbounded_planes_even_with_finite_vertices() {
    let mut body = cube();
    body.planes.retain(|plane| plane.y >= 0.0);
    assert!(body.bounds().iter().all(|point| point.is_finite()));
    assert!(body.conservative_radius().is_err());
    assert!(body.triangle_penetration_bound(floor(), 0.0).is_err());
}

#[test]
fn ill_conditioned_dual_proposals_cannot_underestimate_a_feasible_point() {
    let mut body = cube();
    for epsilon in [1e-8, 1e-12, 1e-16] {
        body.planes
            .push(DVec4::new(1.0, epsilon, 0.0, 0.5 + epsilon * 0.5));
        body.planes
            .push(DVec4::new(1.0, 0.0, epsilon, 0.5 + epsilon * 0.5));
    }
    let radius = radius_bound(&body.planes).unwrap();
    for direction in [DVec3::ONE, DVec3::new(1.0, -2.0, 3.0), DVec3::NEG_Y] {
        let bound = support_bound(&body.planes, direction, radius).unwrap();
        let exact = direction.abs().element_sum() * 0.5;
        assert!(bound >= exact, "{bound} < {exact}");
        assert!(bound - exact < 1e-12);
    }
}

#[test]
fn envelopes_enclose_all_oblique_contact_depths_and_rebased_queries() {
    for index in 0..25 {
        let rotation = DQuat::from_rotation_x(f64::from(index) * 0.17)
            * DQuat::from_rotation_z(f64::from(index) * 0.31);
        let body = cube().transformed(DVec3::Y * 0.4, rotation).unwrap();
        let bound = body
            .triangle_penetration_bound(floor(), 0.0)
            .unwrap()
            .unwrap();
        for point in body.triangle_contacts(floor()).unwrap() {
            assert!(
                point.depth <= bound,
                "contact {} exceeds {bound}",
                point.depth
            );
        }
        let shift = DVec3::new(1e6, -2e6, 3e6);
        let shifted = body.transformed(shift, DQuat::IDENTITY).unwrap();
        let rebased = shifted
            .triangle_penetration_bound(floor().map(|point| point + shift), 0.0)
            .unwrap()
            .unwrap();
        assert!((bound - rebased).abs() < 1e-7);
    }
}

#[test]
fn envelope_encloses_vertex_and_plane_representation_disagreement() {
    let mut body = cube();
    for vertex in &mut body.vertices {
        if vertex.y < 0.0 {
            vertex.y -= 1e-6;
        }
    }
    let bound = body
        .triangle_penetration_bound(floor(), 0.0)
        .unwrap()
        .unwrap();
    assert!(bound >= 0.500_001);
    assert!(bound < 0.500_001 + 1e-12);
}

#[test]
fn envelope_arithmetic_overflow_is_an_error_instead_of_a_small_bound() {
    let mut body = cube();
    body.vertices[0] = DVec3::splat(f64::MAX);
    assert!(body.triangle_penetration_bound(floor(), 0.0).is_err());
    assert!(body.conservative_radius().is_err());
}

#[test]
fn below_surface_separation_requires_the_whole_envelope_to_miss() {
    for rotation in [DQuat::IDENTITY, DQuat::from_rotation_z(0.37)] {
        let body = cube().transformed(-DVec3::Y * 0.75, rotation).unwrap();
        let triangle = floor();
        assert_eq!(
            body.triangle_penetration_bound(triangle, 0.01).unwrap(),
            None
        );
        assert!(
            body.triangle_penetration_bound(triangle, 0.8)
                .unwrap()
                .is_some()
        );
        let shift = DVec3::new(1e6, -2e6, 3e6);
        assert_eq!(
            body.transformed(shift, DQuat::IDENTITY)
                .unwrap()
                .triangle_penetration_bound(triangle.map(|p| p + shift), 0.01)
                .unwrap(),
            None
        );
    }
}
