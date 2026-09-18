use super::*;

#[test]
fn numerical_zero_vertex_overlap_survives_an_empty_rounded_face_query() {
    type Recorded = (Vec<[f64; 3]>, Vec<[f64; 4]>, Vec<[f64; 3]>, Vec<[f64; 3]>);
    let (vertices, planes, edges, triangle): Recorded =
        ron::from_str(include_str!("activation_roundoff.ron")).unwrap();
    let shape = ContactPolytope {
        vertices: vertices.into_iter().map(DVec3::from_array).collect(),
        planes: planes.into_iter().map(DVec4::from_array).collect(),
        edges: edges.into_iter().map(DVec3::from_array).collect(),
    };
    let triangle = triangle
        .into_iter()
        .map(DVec3::from_array)
        .collect::<Vec<_>>();
    let triangle: [DVec3; 3] = triangle.try_into().unwrap();
    assert!(shape.triangle_contacts(triangle).unwrap().is_empty());
    let points = shape.triangle_activation_contacts(triangle, 1e-12).unwrap();
    assert!(!points.is_empty());
    for point in points {
        assert!(shape.vertices.contains(&point.body_point));
        assert!(super::super::activation::inside_triangle(
            triangle,
            point.triangle_point,
            point.normal
        ));
        let gap = (point.body_point - point.triangle_point).dot(point.normal);
        assert!((-1e-12..0.0).contains(&gap));
        assert_eq!(point.depth.to_bits(), (-gap).to_bits());
    }
    let underneath = shape.transformed(-DVec3::Y * 1.0, DQuat::IDENTITY).unwrap();
    assert!(underneath.vertex_activation(triangle, 1e-12).is_err());
}

#[test]
fn vertex_activation_requires_an_actual_corner_inside_the_finite_triangle() {
    let shape = cube()
        .transformed(DVec3::Y * (0.5 + 1e-13), DQuat::IDENTITY)
        .unwrap();
    assert_eq!(shape.vertex_activation(floor(), 1e-12).unwrap().len(), 4);
    let outside = [
        DVec3::new(2.0, 0.0, -1.0),
        DVec3::new(2.0, 0.0, 1.0),
        DVec3::new(4.0, 0.0, 0.0),
    ];
    assert!(shape.vertex_activation(outside, 1e-12).is_err());
    let interior_patch = [
        DVec3::new(-0.1, 0.0, -0.1),
        DVec3::new(0.0, 0.0, 0.1),
        DVec3::new(0.1, 0.0, -0.1),
    ];
    // The ordinary face query may handle this patch. Vertex activation has no
    // certified vertex there and must preserve the inconclusive/error result.
    assert!(shape.vertex_activation(interior_patch, 1e-12).is_err());
    assert!(shape.vertex_activation(floor(), 1e-14).is_err());
}

#[test]
#[expect(
    clippy::unreadable_literal,
    reason = "preserve the captured round-trip decimal geometry"
)]
fn a_near_parallel_wheel_corner_keeps_its_actual_finite_vertex_contact() {
    let shape = ContactPolytope {
        vertices: vec![
            DVec3::new(1.1699691648242851, 0.48426522089167057, -4.868374194577161),
            DVec3::new(0.9984856575708123, 0.0682232146018491, -4.869937848487868),
            DVec3::new(1.172247123557441, 0.48426585759780544, -5.118363816168948),
            DVec3::new(1.0007636163039682, 0.06822385130798397, -5.119927470079654),
            DVec3::new(1.3354740908255784, 0.416042006289854, -4.86686625216329),
            DVec3::new(
                1.1639905835721054,
                3.2529534621517087e-14,
                -4.868429906073997,
            ),
            DVec3::new(1.3377520495587343, 0.41604264299598887, -5.116855873755077),
            DVec3::new(1.1662685423052614, 6.367061674561469e-7, -5.118419527665783),
        ],
        planes: vec![
            DVec4::new(
                0.3810744706582989,
                0.9245378162470469,
                0.0034747865602868877,
                0.8766503287577612,
            ),
            DVec4::new(
                -0.3810744706582989,
                -0.9245378162470469,
                -0.0034747865602868877,
                -0.4266503406786901,
            ),
            DVec4::new(
                -0.00911183493262353,
                -2.5468245394400905e-6,
                0.9999584863671465,
                -4.87883388992294,
            ),
            DVec4::new(
                0.00911183493262353,
                2.5468245394400905e-6,
                -0.9999584863671465,
                5.12883388992294,
            ),
            DVec4::new(
                -0.9244994441732554,
                0.3810903125541975,
                -0.008423265440798095,
                -0.8560794500851101,
            ),
            DVec4::new(
                0.9244994441732554,
                -0.3810903125541975,
                0.008423265440798095,
                1.0351005701565212,
            ),
        ],
        edges: vec![
            DVec3::new(
                -0.3810744706582989,
                -0.9245378162470469,
                -0.0034747865602868877,
            ),
            DVec3::new(
                0.00911183493262353,
                2.5468245394400905e-6,
                -0.9999584863671465,
            ),
            DVec3::new(
                0.9244994441732554,
                -0.3810903125541975,
                0.008423265440798095,
            ),
        ],
    };
    let triangle = [
        DVec3::new(-64.0, 0.0, -64.0),
        DVec3::new(64.0, 0.0, 64.0),
        DVec3::new(64.0, 0.0, -64.0),
    ];
    assert!(shape.triangle_contacts(triangle).unwrap().is_empty());
    let contacts = shape.triangle_activation_contacts(triangle, 1e-12).unwrap();
    assert!(!contacts.is_empty());
    for contact in contacts {
        let gap = (contact.body_point - contact.triangle_point).dot(contact.normal);
        assert!((0.0..=1e-12).contains(&gap));
        assert!(shape.vertices.contains(&contact.body_point));
        for edge in 0..3 {
            assert!(
                (triangle[(edge + 1) % 3] - triangle[edge])
                    .cross(contact.triangle_point - triangle[edge])
                    .dot(contact.normal)
                    >= 0.0
            );
        }
    }
}

#[test]
fn an_empty_inner_search_still_activates_finite_vertices_inside_the_full_bound() {
    for gap in [6e-13, 7.5e-13, 9e-13] {
        let shape = cube()
            .transformed(DVec3::Y * (0.5 + gap), DQuat::IDENTITY)
            .unwrap();
        assert!(
            shape
                .triangle_proximity(floor(), 0.5e-12)
                .unwrap()
                .is_empty()
        );
        let points = shape.triangle_activation_contacts(floor(), 1e-12).unwrap();
        assert_eq!(
            points.len(),
            4,
            "finite vertices disappeared at gap={gap:e}"
        );
        for point in points {
            assert!(shape.vertices.contains(&point.body_point));
            assert!((point.body_point - point.triangle_point).dot(point.normal) <= 1e-12);
        }
    }
    let outside = cube()
        .transformed(DVec3::Y * (0.5 + 1.1e-12), DQuat::IDENTITY)
        .unwrap();
    assert!(
        outside
            .triangle_activation_contacts(floor(), 1e-12)
            .unwrap()
            .is_empty()
    );
}
