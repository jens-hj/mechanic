use bevy_math::IVec3;

use super::*;
use crate::id::Handle;
use crate::{
    BuildPose, ConstructionMaterial, CuboidSpec, CylinderDimensions, CylinderSpec, GridRotation,
    PipeBendDimensions, PipeBendSpec,
};

fn cube() -> PartSpec {
    PartSpec::Cuboid(
        CuboidSpec::new(
            [1, 1, 1],
            BuildPose::new(IVec3::ZERO, GridRotation::default()),
        )
        .unwrap()
        .with_material(ConstructionMaterial::Steel),
    )
}

#[test]
fn cuboid_boundary_has_six_patches_and_twelve_logical_edges() {
    let solid = evaluate_part_solid(cube(), []).unwrap();
    assert_eq!(solid.surfaces.len(), 6);
    assert_eq!(solid.logical_edges.len(), 12);
    assert_eq!(solid.cells.len(), 1);
    assert!((solid.volume() - 0.25_f64.powi(3)).abs() < 1.0e-8);
}

#[test]
fn chamfer_replaces_one_edge_with_a_generated_patch() {
    let base = evaluate_part_solid(cube(), []).unwrap();
    let edge = base.logical_edges[0].key;
    let feature_id = ShapeFeatureId::from_parts(0, 0);
    let feature = ShapeFeature::new(
        [EdgeChainRef {
            owner: SolidOwner::Part(crate::PartId::from_parts(0, 0)),
            edge,
        }],
        EdgeTreatment::Chamfer,
        10,
    );
    let rounded = evaluate_part_solid(cube(), [(feature_id, feature)]).unwrap();
    assert!(
        rounded
            .surfaces
            .iter()
            .any(|surface| surface.key.source == TopologySource::Feature(feature_id))
    );
    assert!(rounded.volume() < base.volume());
}

#[test]
fn vertex_profile_endpoints_are_face_tangencies() {
    let radius = 0.05;
    let profile = vertex_profile(
        EdgeTreatment::Fillet,
        radius,
        fillet_facets(core::f64::consts::FRAC_PI_2),
        DVec3::ZERO,
        DVec3::X,
        DVec3::Y,
    )
    .unwrap();
    let centre = DVec3::new(-radius, -radius, 0.0);

    assert_eq!(profile.len(), 13, "a right-angle fillet has twelve facets");
    assert!(profile[0].abs_diff_eq(centre + DVec3::X * radius, EPSILON));
    assert!(profile[12].abs_diff_eq(centre + DVec3::Y * radius, EPSILON));
    assert!(
        profile
            .iter()
            .all(|point| (point.distance(centre) - radius).abs() < EPSILON)
    );
}

#[test]
fn cuboid_fillet_accepts_sub_block_radii_in_five_centimetre_steps() {
    let base = evaluate_part_solid(cube(), []).unwrap();
    let edge = base.logical_edges[0].key;
    let feature_id = ShapeFeatureId::from_parts(0, 0);
    for amount_ticks in [20, 40, 60, 80] {
        let feature = ShapeFeature::new(
            [EdgeChainRef {
                owner: SolidOwner::Part(crate::PartId::from_parts(0, 0)),
                edge,
            }],
            EdgeTreatment::Fillet,
            amount_ticks,
        );
        let filleted = evaluate_part_solid(cube(), [(feature_id, feature)])
            .unwrap_or_else(|error| panic!("{amount_ticks} ticks was rejected: {error}"));
        if amount_ticks == 20 {
            let radius = f64::from(amount_ticks) * f64::from(crate::POSITION_TICK_METERS);
            let expected =
                0.25_f64.powi(3) - 0.25 * radius.powi(2) * (1.0 - core::f64::consts::FRAC_PI_4);
            assert!(
                (filleted.volume() - expected).abs() < 2.0e-6,
                "five centimetres produced volume {}, expected {expected}",
                filleted.volume()
            );
        }
    }
}

#[test]
fn cuboid_treatments_accept_a_full_block_amount_but_not_more() {
    let base = evaluate_part_solid(cube(), []).unwrap();
    let edge = base.logical_edges[0].key;
    let feature_id = ShapeFeatureId::from_parts(0, 0);
    for treatment in [EdgeTreatment::Fillet, EdgeTreatment::Chamfer] {
        let target = || EdgeChainRef {
            owner: SolidOwner::Part(crate::PartId::from_parts(0, 0)),
            edge,
        };
        evaluate_part_solid(
            cube(),
            [(feature_id, ShapeFeature::new([target()], treatment, 100))],
        )
        .unwrap_or_else(|error| panic!("full-block {treatment:?} failed: {error}"));

        assert_eq!(
            evaluate_part_solid(
                cube(),
                [(feature_id, ShapeFeature::new([target()], treatment, 120),)],
            ),
            Err(SolidError::AmountTooLarge(feature_id))
        );
    }
}

#[test]
fn larger_cuboids_accept_fillets_and_chamfers_past_one_block() {
    let spec = PartSpec::Cuboid(
        CuboidSpec::new(
            [2, 2, 2],
            BuildPose::new(IVec3::ZERO, GridRotation::default()),
        )
        .unwrap()
        .with_material(ConstructionMaterial::Steel),
    );
    let base = evaluate_part_solid(spec, []).unwrap();
    let edge = base.logical_edges[0].key;
    let feature_id = ShapeFeatureId::from_parts(0, 0);
    for treatment in [EdgeTreatment::Fillet, EdgeTreatment::Chamfer] {
        let feature = ShapeFeature::new(
            [EdgeChainRef {
                owner: SolidOwner::Part(crate::PartId::from_parts(0, 0)),
                edge,
            }],
            treatment,
            120,
        );
        evaluate_part_solid(spec, [(feature_id, feature)])
            .unwrap_or_else(|error| panic!("{treatment:?} past one block failed: {error}"));
    }
}

#[test]
fn three_incident_fillet_edges_round_their_shared_corner() {
    let base = evaluate_part_solid(cube(), []).unwrap();
    let corner = Vec3::splat(0.125);
    let edges = base
        .logical_edges
        .iter()
        .filter(|logical| {
            logical.half_edges.iter().any(|&edge_index| {
                let edge = base.half_edges[edge_index as usize];
                let next = base.half_edges[edge.next as usize];
                base.vertices[edge.origin as usize]
                    .position
                    .abs_diff_eq(corner, 1.0e-6)
                    || base.vertices[next.origin as usize]
                        .position
                        .abs_diff_eq(corner, 1.0e-6)
            })
        })
        .map(|logical| EdgeChainRef {
            owner: SolidOwner::Part(crate::PartId::from_parts(0, 0)),
            edge: logical.key,
        })
        .collect::<Vec<_>>();
    assert_eq!(edges.len(), 3);
    let feature_id = ShapeFeatureId::from_parts(0, 0);
    let filleted = evaluate_part_solid(
        cube(),
        [(
            feature_id,
            ShapeFeature::new(edges, EdgeTreatment::Fillet, 20),
        )],
    )
    .unwrap();

    let junction = filleted
        .surfaces
        .iter()
        .filter(|surface| {
            surface.key.source == TopologySource::Feature(feature_id)
                && surface.normal.x > 0.1
                && surface.normal.y > 0.1
                && surface.normal.z > 0.1
        })
        .collect::<Vec<_>>();
    assert!(!junction.is_empty());
    let sphere_centre = Vec3::splat(0.075);
    for surface in junction {
        let mut edge = surface.half_edge;
        loop {
            let half_edge = filleted.half_edges[edge as usize];
            let position = filleted.vertices[half_edge.origin as usize].position;
            assert!((position.distance(sphere_centre) - 0.05).abs() < 1.0e-5);
            edge = half_edge.next;
            if edge == surface.half_edge {
                break;
            }
        }
    }
    assert!(filleted.logical_edges.iter().all(|logical| {
        logical.half_edges.iter().all(|&edge_index| {
            let edge = filleted.half_edges[edge_index as usize];
            let twin = filleted.half_edges[edge.twin as usize];
            let first = filleted.surfaces[edge.face as usize].smoothing_group;
            let second = filleted.surfaces[twin.face as usize].smoothing_group;
            first == 0 || first != second
        })
    }));
}

#[test]
fn one_logical_edge_does_not_join_matching_edges_on_separate_cells() {
    let half = Vec3::splat(0.125);
    let cells = vec![
        cuboid_cell(Vec3::ZERO, half, Quat::IDENTITY),
        cuboid_cell(Vec3::X * 0.5, half, Quat::IDENTITY),
    ];
    let base = evaluate(cells.clone(), []).unwrap();
    assert_eq!(base.logical_edges.len(), 24);

    let edge = base.logical_edges[0].key;
    let feature_id = ShapeFeatureId::from_parts(0, 0);
    let feature = ShapeFeature::new(
        [EdgeChainRef {
            owner: SolidOwner::Region(RegionId::from_parts(0, 0)),
            edge,
        }],
        EdgeTreatment::Fillet,
        20,
    );
    let filleted = evaluate(cells, [(feature_id, feature)]).unwrap();
    let radius = 0.05_f64;
    let expected =
        2.0 * 0.25_f64.powi(3) - 0.25 * radius.powi(2) * (1.0 - core::f64::consts::FRAC_PI_4);
    assert!((filleted.volume() - expected).abs() < 2.0e-6);
}

#[test]
fn concave_region_edges_are_not_offered_as_subtractive_fillet_targets() {
    let half = Vec3::splat(0.125);
    let cells = vec![
        cuboid_cell(Vec3::ZERO, half, Quat::IDENTITY),
        cuboid_cell(Vec3::X * 0.25, half, Quat::IDENTITY),
        cuboid_cell(Vec3::Y * 0.25, half, Quat::IDENTITY),
    ];

    let solid = evaluate(cells, []).unwrap();
    assert!(
        solid.logical_edges.iter().any(|edge| !edge.convex),
        "the inside corner of an L-shaped region is concave"
    );
}

#[test]
fn five_centimetre_fillet_is_available_on_a_sloped_region() {
    let mut region = ShapeRegion::new(
        IVec3::ZERO,
        IVec3::new(6, 3, 2),
        ConstructionMaterial::Steel,
    )
    .unwrap();
    region.set_offset([0, 1, 1], [0, -15, 0]).unwrap();
    region.set_offset([1, 1, 1], [0, -15, 0]).unwrap();
    let base = evaluate_region_solid(&region, []).unwrap();
    let base_volume = base.volume();
    let owner = SolidOwner::Region(RegionId::from_parts(0, 0));
    for (index, logical) in base
        .logical_edges
        .iter()
        .filter(|edge| edge.convex)
        .enumerate()
    {
        let feature_id = ShapeFeatureId::from_parts(0, 0);
        let feature = ShapeFeature::new(
            [EdgeChainRef {
                owner,
                edge: logical.key,
            }],
            EdgeTreatment::Fillet,
            20,
        );
        let filleted = evaluate_region_solid(&region, [(feature_id, feature)])
            .unwrap_or_else(|error| panic!("sloped edge {index} rejected 5 cm: {error}"));
        assert!(
            base_volume - filleted.volume() < 0.01,
            "sloped edge {index} removed {} cubic metres at 5 cm",
            base_volume - filleted.volume()
        );
    }
}

#[test]
fn region_fillets_accept_a_full_block_radius() {
    let region = ShapeRegion::new(
        IVec3::ZERO,
        IVec3::new(4, 4, 4),
        ConstructionMaterial::Steel,
    )
    .unwrap();
    let base = evaluate_region_solid(&region, []).unwrap();
    let owner = SolidOwner::Region(RegionId::from_parts(0, 0));
    for (index, logical) in base
        .logical_edges
        .iter()
        .filter(|edge| edge.convex)
        .enumerate()
    {
        let feature_id = ShapeFeatureId::from_parts(0, 0);
        let feature = ShapeFeature::new(
            [EdgeChainRef {
                owner,
                edge: logical.key,
            }],
            EdgeTreatment::Fillet,
            100,
        );
        evaluate_region_solid(&region, [(feature_id, feature)])
            .unwrap_or_else(|error| panic!("region edge {index} rejected 25 cm: {error}"));
    }
}

#[test]
fn larger_regions_accept_fillets_and_chamfers_past_one_block() {
    let region = ShapeRegion::new(
        IVec3::ZERO,
        IVec3::new(4, 4, 4),
        ConstructionMaterial::Steel,
    )
    .unwrap();
    let base = evaluate_region_solid(&region, []).unwrap();
    let owner = SolidOwner::Region(RegionId::from_parts(0, 0));
    let edge = base
        .logical_edges
        .iter()
        .find(|edge| edge.convex)
        .unwrap()
        .key;
    let feature_id = ShapeFeatureId::from_parts(0, 0);
    for treatment in [EdgeTreatment::Fillet, EdgeTreatment::Chamfer] {
        let feature = ShapeFeature::new([EdgeChainRef { owner, edge }], treatment, 120);
        evaluate_region_solid(&region, [(feature_id, feature)])
            .unwrap_or_else(|error| panic!("region {treatment:?} past one block failed: {error}"));
    }
}

#[test]
fn shaped_region_boundary_excludes_internal_convex_decomposition_faces() {
    let mut region =
        ShapeRegion::new(IVec3::ZERO, IVec3::ONE, ConstructionMaterial::Steel).unwrap();
    region.set_offset([1, 1, 1], [-20, -20, -20]).unwrap();
    let pieces = decompose(&region.grid(), &|cell, corner| {
        region.corner_steps(cell, corner)
    });
    let expected_boundary_faces = pieces
        .iter()
        .map(|piece| match piece {
            PartPiece::Cuboid { .. } => 6,
            PartPiece::Convex(piece) => piece
                .faces
                .iter()
                .filter(|face| face.grid_face.is_some())
                .count(),
        })
        .sum::<usize>();

    let solid = evaluate_region_solid(&region, []).unwrap();
    assert_eq!(
        solid.surfaces.len(),
        expected_boundary_faces,
        "the feature boundary must not expose internal convex-cell faces"
    );
}

#[test]
fn five_centimetre_fillet_is_available_on_a_generated_chamfer_edge() {
    let spec = PartSpec::Cuboid(
        CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::ZERO, GridRotation::default()),
        )
        .unwrap(),
    );
    let owner = SolidOwner::Part(crate::PartId::from_parts(0, 0));
    let base = evaluate_part_solid(spec, []).unwrap();
    let chamfer_id = ShapeFeatureId::from_parts(0, 0);
    let chamfer = ShapeFeature::new(
        [EdgeChainRef {
            owner,
            edge: base.logical_edges[0].key,
        }],
        EdgeTreatment::Chamfer,
        100,
    );
    let chamfered = evaluate_part_solid(spec, [(chamfer_id, chamfer.clone())]).unwrap();
    let generated = chamfered
        .logical_edges
        .iter()
        .find(|edge| edge.key.source == TopologySource::Feature(chamfer_id) && edge.convex)
        .unwrap();
    let fillet_id = ShapeFeatureId::from_parts(1, 0);
    let fillet = ShapeFeature::new(
        [EdgeChainRef {
            owner,
            edge: generated.key,
        }],
        EdgeTreatment::Fillet,
        20,
    );

    evaluate_part_solid(spec, [(chamfer_id, chamfer), (fillet_id, fillet)])
        .expect("a generated chamfer edge accepts a 5 cm fillet");
}

#[test]
fn cylinder_rims_are_closed_logical_chains_not_tessellation_edges() {
    let spec = PartSpec::Cylinder(CylinderSpec::new(
        CylinderDimensions::default(),
        BuildPose::default(),
    ));
    let solid = evaluate_part_solid(spec, []).unwrap();
    assert!(
        solid
            .logical_edges
            .iter()
            .filter(|edge| edge.closed)
            .count()
            >= 2
    );
    assert!(solid.logical_edges.len() < 24);
    assert!(solid.volume() > 0.0);
}

fn cylinder_spec(inner_diameter: f32, sweep_degrees: u16) -> PartSpec {
    let dimensions = CylinderDimensions::new(0.5, inner_diameter, 0.5)
        .unwrap()
        .with_sweep_angle_degrees(sweep_degrees)
        .unwrap();
    PartSpec::Cylinder(CylinderSpec::new(dimensions, BuildPose::default()))
}

fn target(owner: SolidOwner, edge: TopologyKey) -> EdgeChainRef {
    EdgeChainRef { owner, edge }
}

#[test]
fn solid_and_hollow_cylinder_rims_accept_five_centimetre_treatments() {
    let owner = SolidOwner::Part(crate::PartId::from_parts(0, 0));
    for spec in [cylinder_spec(0.0, 360), cylinder_spec(0.25, 360)] {
        let base = evaluate_part_solid(spec, []).unwrap();
        let rims = base
            .logical_edges
            .iter()
            .filter(|edge| edge.closed && edge.convex)
            .map(|edge| edge.key)
            .collect::<Vec<_>>();
        assert_eq!(
            rims.len(),
            if matches!(spec, PartSpec::Cylinder(cylinder) if cylinder.dimensions.inner_diameter() > 0.0)
            {
                4
            } else {
                2
            }
        );
        assert!(rims.iter().all(|key| {
            base.logical_edge(*key)
                .is_some_and(|edge| edge.half_edges.len() == 24)
        }));
        for (edge_index, edge) in rims.into_iter().enumerate() {
            for treatment in [EdgeTreatment::Chamfer, EdgeTreatment::Fillet] {
                let feature = ShapeFeatureId::from_parts(edge_index as u32, 0);
                evaluate_part_solid(
                    spec,
                    [(
                        feature,
                        ShapeFeature::new([target(owner, edge)], treatment, 20),
                    )],
                )
                .unwrap_or_else(|error| {
                    panic!("{treatment:?} rejected cylinder rim {edge_index}: {error}")
                });
            }
        }
    }
}

#[test]
fn translated_and_rotated_cylinder_rims_accept_treatments() {
    let owner = SolidOwner::Part(crate::PartId::from_parts(0, 0));
    let poses = [
        BuildPose::new(IVec3::new(1, 4, 0), GridRotation::default()),
        BuildPose::new(IVec3::ZERO, GridRotation::new(0, 0, 3)),
        BuildPose::new(IVec3::new(-3, 2, 7), GridRotation::new(1, 2, 0)),
    ];
    for pose in poses {
        for (inner_diameter, sweep_degrees) in [(0.0, 360), (0.25, 360), (0.25, 90)] {
            let dimensions = CylinderDimensions::new(0.5, inner_diameter, 0.5)
                .unwrap()
                .with_sweep_angle_degrees(sweep_degrees)
                .unwrap();
            let spec = PartSpec::Cylinder(CylinderSpec::new(dimensions, pose));
            let base = evaluate_part_solid(spec, []).unwrap();
            let rims = base
                .logical_edges
                .iter()
                .filter(|edge| edge.convex && edge.half_edges.len() > 1)
                .map(|edge| edge.key)
                .collect::<Vec<_>>();
            assert!(!rims.is_empty());
            for edge in rims {
                for treatment in [EdgeTreatment::Chamfer, EdgeTreatment::Fillet] {
                    let treated = evaluate_part_solid(
                        spec,
                        [(
                            ShapeFeatureId::from_parts(0, 0),
                            ShapeFeature::new([target(owner, edge)], treatment, 20),
                        )],
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "{treatment:?} rejected a rim of {dimensions:?} at {pose:?}: {error}"
                        )
                    });
                    assert!(treated.volume() > 0.0 && treated.volume() < base.volume());
                }
            }
        }
    }
}

#[test]
fn posed_cylinder_fillet_vertices_lie_on_the_rim_torus() {
    let owner = SolidOwner::Part(crate::PartId::from_parts(0, 0));
    let spec = PartSpec::Cylinder(CylinderSpec::new(
        CylinderDimensions::new(0.5, 0.0, 0.5).unwrap(),
        BuildPose::new(IVec3::new(1, 4, 0), GridRotation::default()),
    ));
    let axis = DVec3::new(0.25, 1.0, 0.0);
    let (outer_radius, half_length, fillet_radius) = (0.25, 0.25, 0.05);
    let base = evaluate_part_solid(spec, []).unwrap();
    for rim in base
        .logical_edges
        .iter()
        .filter(|edge| edge.closed && edge.convex)
    {
        let feature = ShapeFeatureId::from_parts(0, 0);
        let filleted = evaluate_part_solid(
            spec,
            [(
                feature,
                ShapeFeature::new([target(owner, rim.key)], EdgeTreatment::Fillet, 20),
            )],
        )
        .unwrap();
        let mut checked = 0;
        for surface in filleted
            .surfaces
            .iter()
            .filter(|surface| surface.key.source == TopologySource::Feature(feature))
        {
            let mut edge = surface.half_edge;
            loop {
                let half_edge = filleted.half_edges[edge as usize];
                let offset = filleted.vertices[half_edge.origin as usize]
                    .position
                    .as_dvec3()
                    - axis;
                let radial = offset.x.hypot(offset.z) - (outer_radius - fillet_radius);
                let axial = offset.y.abs() - (half_length - fillet_radius);
                assert!(
                    (radial.hypot(axial) - fillet_radius).abs() < 1.0e-5,
                    "fillet vertex {offset:?} is off the rim torus"
                );
                checked += 1;
                edge = half_edge.next;
                if edge == surface.half_edge {
                    break;
                }
            }
        }
        assert!(checked > 0);
    }
}

#[test]
fn hollow_sector_curved_rims_and_axial_cut_edges_accept_treatments() {
    let spec = cylinder_spec(0.25, 90);
    let owner = SolidOwner::Part(crate::PartId::from_parts(0, 0));
    let base = evaluate_part_solid(spec, []).unwrap();
    let curved_rims = base
        .logical_edges
        .iter()
        .filter(|edge| !edge.closed && edge.convex && edge.half_edges.len() > 1)
        .map(|edge| edge.key)
        .collect::<Vec<_>>();
    assert_eq!(curved_rims.len(), 4);
    let axial_cuts = base
        .logical_edges
        .iter()
        .filter(|edge| {
            edge.convex && edge.half_edges.len() == 1 && {
                let half_edge = base.half_edges[edge.half_edges[0] as usize];
                let next = base.half_edges[half_edge.next as usize];
                let a = base.vertices[half_edge.origin as usize].position;
                let b = base.vertices[next.origin as usize].position;
                (a.y - b.y).abs() > 0.49
            }
        })
        .map(|edge| edge.key)
        .collect::<Vec<_>>();
    assert_eq!(axial_cuts.len(), 4);

    for (index, edge) in curved_rims.iter().copied().enumerate() {
        for treatment in [EdgeTreatment::Chamfer, EdgeTreatment::Fillet] {
            let feature = ShapeFeatureId::from_parts(index as u32, 0);
            evaluate_part_solid(
                spec,
                [(
                    feature,
                    ShapeFeature::new([target(owner, edge)], treatment, 20),
                )],
            )
            .unwrap_or_else(|error| panic!("sector {treatment:?} failed: {error}"));
        }
    }
    for (index, edge) in axial_cuts.iter().copied().enumerate() {
        for treatment in [EdgeTreatment::Chamfer, EdgeTreatment::Fillet] {
            let feature = ShapeFeatureId::from_parts(index as u32 + 4, 0);
            evaluate_part_solid(
                spec,
                [(
                    feature,
                    ShapeFeature::new([target(owner, edge)], treatment, 5),
                )],
            )
            .unwrap_or_else(|error| panic!("sector axial {treatment:?} failed: {error}"));
        }
    }

    let fillet_id = ShapeFeatureId::from_parts(8, 0);
    let filleted = evaluate_part_solid(
        spec,
        [(
            fillet_id,
            ShapeFeature::new([target(owner, curved_rims[0])], EdgeTreatment::Fillet, 20),
        )],
    )
    .unwrap();
    assert!(filleted.logical_edges.iter().any(|edge| {
        edge.key.source == TopologySource::Feature(fillet_id) && edge.convex && !edge.closed
    }));
}

#[test]
fn chamfered_cylinder_rim_produces_two_treatable_closed_chains() {
    let spec = cylinder_spec(0.0, 360);
    let owner = SolidOwner::Part(crate::PartId::from_parts(0, 0));
    let base = evaluate_part_solid(spec, []).unwrap();
    let rim = base
        .logical_edges
        .iter()
        .find(|edge| edge.closed && edge.convex)
        .unwrap()
        .key;
    let chamfer_id = ShapeFeatureId::from_parts(0, 0);
    let chamfer = ShapeFeature::new([target(owner, rim)], EdgeTreatment::Chamfer, 20);
    let chamfered = evaluate_part_solid(spec, [(chamfer_id, chamfer.clone())]).unwrap();
    let generated = chamfered
        .logical_edges
        .iter()
        .filter(|edge| {
            edge.key.source == TopologySource::Feature(chamfer_id) && edge.closed && edge.convex
        })
        .map(|edge| edge.key)
        .collect::<Vec<_>>();
    assert_eq!(generated.len(), 2);
    assert!(generated.iter().all(|key| {
        chamfered
            .logical_edge(*key)
            .is_some_and(|edge| edge.half_edges.len() == 24)
    }));

    for edge in generated {
        for treatment in [EdgeTreatment::Chamfer, EdgeTreatment::Fillet] {
            let follow_up_id = ShapeFeatureId::from_parts(1, 0);
            let follow_up = ShapeFeature::new([target(owner, edge)], treatment, 20);
            evaluate_part_solid(
                spec,
                [(chamfer_id, chamfer.clone()), (follow_up_id, follow_up)],
            )
            .unwrap_or_else(|error| panic!("follow-up {treatment:?} failed: {error}"));
        }
    }
}

#[test]
fn cylinder_fillet_tangencies_are_not_selectable_edges() {
    let spec = cylinder_spec(0.0, 360);
    let owner = SolidOwner::Part(crate::PartId::from_parts(0, 0));
    let base = evaluate_part_solid(spec, []).unwrap();
    let rim = base
        .logical_edges
        .iter()
        .find(|edge| edge.closed && edge.convex)
        .unwrap()
        .key;
    let feature = ShapeFeatureId::from_parts(0, 0);
    let filleted = evaluate_part_solid(
        spec,
        [(
            feature,
            ShapeFeature::new([target(owner, rim)], EdgeTreatment::Fillet, 20),
        )],
    )
    .unwrap();

    assert!(
        filleted
            .logical_edges
            .iter()
            .all(|edge| { edge.key.source != TopologySource::Feature(feature) })
    );
    assert!(
        filleted
            .logical_edges
            .iter()
            .any(|edge| edge.closed && edge.convex)
    );
}

#[test]
fn pipe_bend_generator_is_manifold_for_solid_and_hollow_profiles() {
    for dimensions in [
        PipeBendDimensions::new(0.20, 0.0, 2).unwrap(),
        PipeBendDimensions::new(0.25, 0.0, 1).unwrap(),
        PipeBendDimensions::new(0.25, 0.10, 1).unwrap(),
        PipeBendDimensions::default(),
    ] {
        let solid = evaluate_part_solid(
            PartSpec::PipeBend(PipeBendSpec::new(dimensions, BuildPose::default())),
            [],
        )
        .unwrap();
        assert!(!solid.cells.is_empty());
        assert!(solid.logical_edges.iter().any(|edge| edge.closed));
    }
}

#[test]
fn pipe_junction_generator_evaluates_every_arm_set_solid_and_hollow() {
    use crate::{PipeArms, PipeJunctionDimensions, PipeJunctionSpec};
    let (radius, reach) = (0.10_f64, 0.125_f64);
    for inner in [0.0, 0.10] {
        let bore = f64::from(inner) * 0.5;
        for bits in [0b00_0001, 0b00_0011, 0b01_0011, 0b11_1111] {
            let arms = PipeArms::from_bits(bits).unwrap();
            let spec = PipeJunctionSpec::new(
                PipeJunctionDimensions::new(0.20, inner).unwrap(),
                arms,
                BuildPose::default(),
            );
            let solid = evaluate_part_solid(PartSpec::PipeJunction(spec), [])
                .unwrap_or_else(|error| panic!("bits {bits:06b}, inner {inner}: {error}"));
            let volume = solid
                .cells
                .iter()
                .map(|cell| f64::from(cell.piece.volume))
                .sum::<f64>();
            // A lone arm is a pipe capped by half the centre ball; opposite
            // arms make one straight pipe across the cell.
            let expected = match bits {
                0b00_0001 => {
                    core::f64::consts::PI
                        * ((radius * radius - bore * bore) * reach
                            + (radius.powi(3) - bore.powi(3)) * 2.0 / 3.0)
                }
                0b00_0011 => core::f64::consts::PI * (radius * radius - bore * bore) * 2.0 * reach,
                _ => continue,
            };
            assert!(
                (volume - expected).abs() < expected * 0.03,
                "bits {bits:06b}, inner {inner}: volume {volume} vs {expected}"
            );
        }
    }
}

fn layered_wheel() -> (PartSpec, PartSpec) {
    let core = CylinderSpec::new(
        CylinderDimensions::new(1.0, 0.0, 0.5).unwrap(),
        BuildPose::default(),
    );
    let layered = core
        .with_layer(
            crate::LayerFace::OuterWall,
            0.25,
            ConstructionMaterial::Rubber,
            crate::MaterialAppearance::BAKED,
        )
        .unwrap();
    let envelope = CylinderSpec::new(layered.dimensions, BuildPose::default());
    (PartSpec::Cylinder(layered), PartSpec::Cylinder(envelope))
}

#[test]
fn layered_cylinder_band_interfaces_stitch_away() {
    let (layered, envelope) = layered_wheel();
    let plain = evaluate_part_solid(envelope, []).unwrap();
    let banded = evaluate_part_solid(layered, []).unwrap();
    assert!((banded.volume() - plain.volume()).abs() < 1.0e-4 * plain.volume());
    assert_eq!(banded.logical_edges.len(), plain.logical_edges.len());
    assert!(banded.cells.iter().any(|cell| cell.band == 0));
    assert!(banded.cells.iter().any(|cell| cell.band == 1));
}

#[test]
fn fillet_deeper_than_the_outer_layer_cuts_through_both_bands() {
    let owner = SolidOwner::Part(crate::PartId::from_parts(0, 0));
    let (layered, envelope) = layered_wheel();
    let rim = evaluate_part_solid(envelope, [])
        .unwrap()
        .logical_edges
        .iter()
        .find(|edge| edge.closed && edge.convex)
        .unwrap()
        .key;
    let fillet = [(
        ShapeFeatureId::from_parts(0, 0),
        ShapeFeature::new([target(owner, rim)], EdgeTreatment::Fillet, 120),
    )];
    let plain = evaluate_part_solid(envelope, fillet.clone()).unwrap();
    let banded = evaluate_part_solid(layered, fillet).unwrap();
    assert!((banded.volume() - plain.volume()).abs() < 1.0e-4 * plain.volume());
    assert_eq!(banded.logical_edges.len(), plain.logical_edges.len());
    let rounded = |band| {
        banded
            .surfaces
            .iter()
            .any(|surface| surface.smoothing_group != 0 && surface.band == band)
    };
    assert!(rounded(0), "a 30 cm fillet reaches the steel core");
    assert!(rounded(1), "the fillet also rounds the rubber");
}

fn rubber(spec: PartSpec, face: crate::LayerFace, thickness: f32) -> PartSpec {
    spec.with_layer(
        face,
        thickness,
        ConstructionMaterial::Rubber,
        crate::MaterialAppearance::BAKED,
    )
    .unwrap()
}

#[test]
fn cap_and_wall_layer_interfaces_stitch_away() {
    let core = PartSpec::Cylinder(CylinderSpec::new(
        CylinderDimensions::new(1.0, 0.5, 0.5).unwrap(),
        BuildPose::default(),
    ));
    let layered = rubber(
        rubber(
            rubber(core, crate::LayerFace::Face(FaceKind::PositiveY), 0.1),
            crate::LayerFace::OuterWall,
            0.25,
        ),
        crate::LayerFace::Bore,
        0.05,
    );
    let cylinder = layered.as_cylinder().unwrap();
    let envelope = PartSpec::Cylinder(CylinderSpec::new(cylinder.dimensions, cylinder.pose));
    let plain = evaluate_part_solid(envelope, []).unwrap();
    let banded = evaluate_part_solid(layered, []).unwrap();
    assert!((banded.volume() - plain.volume()).abs() < 1.0e-4 * plain.volume());
    assert_eq!(banded.logical_edges.len(), plain.logical_edges.len());
    for band in 0..=3 {
        assert!(
            banded.cells.iter().any(|cell| cell.band == band),
            "band {band} has cells"
        );
    }
}

#[test]
fn fillet_deeper_than_a_face_layer_cuts_through_both_bands() {
    let owner = SolidOwner::Part(crate::PartId::from_parts(0, 0));
    let core = PartSpec::Cuboid(CuboidSpec::new([4, 4, 4], BuildPose::default()).unwrap());
    let layered = rubber(core, crate::LayerFace::Face(FaceKind::PositiveY), 0.25);
    let envelope = PartSpec::Cuboid(CuboidSpec::new([4, 5, 4], layered.pose()).unwrap());
    let plain_base = evaluate_part_solid(envelope, []).unwrap();
    // The top edge along x on the +z side.
    let top_edge = plain_base
        .logical_edges
        .iter()
        .find(|edge| {
            edge.half_edges.iter().all(|&half_edge| {
                let origin = plain_base.half_edges[half_edge as usize].origin;
                let position = plain_base.vertices[origin as usize].position;
                position.y > 0.6 && position.z > 0.4
            })
        })
        .unwrap()
        .key;
    let fillet = [(
        ShapeFeatureId::from_parts(0, 0),
        ShapeFeature::new([target(owner, top_edge)], EdgeTreatment::Fillet, 120),
    )];
    let plain = evaluate_part_solid(envelope, fillet.clone()).unwrap();
    let banded = evaluate_part_solid(layered, fillet).unwrap();
    assert!((banded.volume() - plain.volume()).abs() < 1.0e-4 * plain.volume());
    assert_eq!(banded.logical_edges.len(), plain.logical_edges.len());
    let rounded = |band| {
        banded
            .surfaces
            .iter()
            .any(|surface| surface.smoothing_group != 0 && surface.band == band)
    };
    assert!(
        rounded(0),
        "a 30 cm fillet reaches the core under a 25 cm layer"
    );
    assert!(rounded(1), "the fillet also rounds the layer");
}
