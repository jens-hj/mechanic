use bevy_math::{IVec3, Vec3};

use super::mass::outer_product;
use crate::PartPiece;
use crate::{
    ActuatorAssignment, BearingDimensions, BearingId, BearingSpec, BuildCommand, BuildOutcome,
    BuildPose, ColliderShape, ConstructionGraph, ConstructionMaterial, ControllerSpec,
    CoordinateDrive, CuboidSpec, CylinderDimensions, CylinderSpec, DriveLimits, DriveLinkSpec,
    DriveMode, DriveProgram, DriveState, DriveTarget, EdgeChainRef, EdgeTreatment, EngineKind,
    EngineSpec, FaceKind, FaceRef, GearSelection, GridRotation, PIPE_BEND_COLLIDER_COUNT, PartId,
    PartSpec, PipeBendDimensions, PipeBendSpec, RigidLinkSpec, ShapeFeature, SolidOwner,
    TopologyError, WeldSpec,
};
use bevy_math::{Mat3, Quat};

#[test]
fn unattached_suspension_socket_adds_carried_mass_center_and_inertia_once() {
    let mut graph = ConstructionGraph::new();
    let source = spawn(&mut graph, IVec3::new(0, 2, 0));
    let spec = crate::SuspensionSpec::new(
        Some(crate::SpringSpec::default()),
        Some(crate::ShockSpec::default()),
        None,
    )
    .unwrap();
    let socket = crate::BearingSocket {
        kind: crate::JointKind::Suspension(spec),
        axis: Vec3::X,
        source: FaceRef::part(source, FaceKind::PositiveX),
        anchor: Vec3::new(0.5, 0.5, 0.0),
        dimensions: BearingDimensions::default(),
    };
    let bare = graph.compile().unwrap().compounds[0].mass_properties;
    let compiled = graph.compile_with_sockets([], &[socket]).unwrap();
    let actual = compiled.compounds[0].mass_properties;
    let elements = spec.mass_elements();
    let added_mass: f32 = elements.iter().map(|element| element.mass).sum();
    let total_mass = bare.mass + added_mass;
    let expected_center = (bare.center_of_mass * bare.mass
        + elements
            .iter()
            .map(|element| (socket.anchor + socket.axis * element.center) * element.mass)
            .sum::<Vec3>())
        / total_mass;
    assert!((actual.mass - total_mass).abs() < 0.001);
    assert!(actual.center_of_mass.abs_diff_eq(expected_center, 1.0e-6));
    let shift = bare.center_of_mass - expected_center;
    let mut expected_inertia = bare.inertia
        + bare.mass * (Mat3::IDENTITY * shift.length_squared() - outer_product(shift, shift));
    for element in elements {
        let offset = socket.anchor + socket.axis * element.center - expected_center;
        expected_inertia += Mat3::from_diagonal(Vec3::new(
            element.axial_inertia,
            element.transverse_inertia,
            element.transverse_inertia,
        )) + element.mass
            * (Mat3::IDENTITY * offset.length_squared() - outer_product(offset, offset));
    }
    assert!(actual.inertia.abs_diff_eq(expected_inertia, 0.001));
    assert_eq!(
        compiled,
        graph.compile_with_sockets([], &[socket, socket]).unwrap()
    );
    let anchored = graph.compile_with_sockets([source], &[socket]).unwrap();
    assert!(anchored.compounds[0].is_static);
    assert!(anchored.compounds[0].mass_properties.inverse_mass.abs() < f32::EPSILON);
    assert!((anchored.compounds[0].mass_properties.mass - total_mass).abs() < 0.001);
}

#[test]
fn attached_suspension_socket_does_not_duplicate_either_endpoint_mass() {
    let mut graph = ConstructionGraph::new();
    let source = spawn(&mut graph, IVec3::new(0, 2, 0));
    let target = spawn(&mut graph, IVec3::new(6, 2, 0));
    let bare_mass: f32 = graph
        .compile()
        .unwrap()
        .compounds
        .iter()
        .map(|body| body.mass_properties.mass)
        .sum();
    let spec = crate::SuspensionSpec::new(
        Some(crate::SpringSpec::default()),
        Some(crate::ShockSpec::default()),
        None,
    )
    .unwrap();
    let socket = crate::BearingSocket {
        kind: crate::JointKind::Suspension(spec),
        axis: Vec3::X,
        source: FaceRef::part(source, FaceKind::PositiveX),
        anchor: Vec3::new(0.5, 0.5, 0.0),
        dimensions: BearingDimensions::default(),
    };
    graph
        .apply(BuildCommand::AddBearing(
            BearingSpec::new(
                socket.source,
                FaceRef::part(target, FaceKind::NegativeX),
                socket.anchor,
                socket.axis,
            )
            .with_kind(socket.kind),
        ))
        .unwrap();
    let attached = graph.compile().unwrap();
    assert_eq!(attached, graph.compile_with_sockets([], &[socket]).unwrap());
    let total_mass: f32 = attached
        .compounds
        .iter()
        .map(|body| body.mass_properties.mass)
        .sum();
    let added_mass: f32 = spec
        .mass_elements()
        .iter()
        .map(|element| element.mass)
        .sum();
    assert!((total_mass - bare_mass - added_mass).abs() < 0.001);
}

fn cube_at(units: IVec3) -> CuboidSpec {
    CuboidSpec::new([4, 4, 4], BuildPose::new(units, GridRotation::default())).unwrap()
}

fn assert_frame_compilation(mut graph: ConstructionGraph) {
    let original = graph.compile().unwrap();
    let frame = crate::ConstructionFrame::new(
        Vec3::new(2.1, -0.8, 1.3),
        Quat::from_rotation_y(0.61) * Quat::from_rotation_x(-0.37),
    )
    .unwrap();
    let parts = graph.parts().map(|(part, _)| part).collect::<Vec<_>>();
    graph.reframe_parts(parts, frame).unwrap();
    let compiled = graph.compile().unwrap();
    let basis = Mat3::from_quat(frame.rotation());
    assert_eq!(original.compounds.len(), compiled.compounds.len());
    for (old, new) in original.compounds.iter().zip(&compiled.compounds) {
        assert!(
            new.root_translation
                .abs_diff_eq(frame.point(old.root_translation), 1.0e-4)
        );
        assert!(
            (old.mass_properties.mass - new.mass_properties.mass).abs()
                < old.mass_properties.mass * 1.0e-4
        );
        let expected = basis * old.mass_properties.inertia * basis.transpose();
        let tolerance = expected
            .to_cols_array()
            .into_iter()
            .map(f32::abs)
            .fold(1.0, f32::max)
            * 1.0e-4;
        assert!(new.mass_properties.inertia.abs_diff_eq(expected, tolerance));
    }
    assert_eq!(original.colliders.len(), compiled.colliders.len());
    for old in &original.colliders {
        let center = frame.vector(old.local_center);
        let new = compiled
            .colliders
            .iter()
            .find(|new| new.local_center.abs_diff_eq(center, 1.0e-4))
            .expect("every original collider retains its transformed centroid");
        match (&old.shape, &new.shape) {
            (
                ColliderShape::Cuboid {
                    local_rotation: old_rotation,
                    half_extents: old_half,
                },
                ColliderShape::Cuboid {
                    local_rotation: new_rotation,
                    half_extents: new_half,
                },
            ) => {
                assert!(new_rotation.abs_diff_eq(frame.rotation() * *old_rotation, 1.0e-5));
                assert!(new_half.abs_diff_eq(*old_half, 1.0e-6));
            }
            (ColliderShape::Convex(old), ColliderShape::Convex(new)) => {
                for vertex in &old.vertices {
                    assert!(
                        new.vertices
                            .iter()
                            .any(|new| new.abs_diff_eq(frame.vector(*vertex), 1.0e-4))
                    );
                }
                for plane in &old.face_planes {
                    let expected = frame.vector(plane.truncate()).extend(plane.w);
                    assert!(
                        new.face_planes
                            .iter()
                            .any(|new| new.abs_diff_eq(expected, 1.0e-4))
                    );
                }
                for direction in &old.edge_directions {
                    let expected = frame.vector(*direction);
                    assert!(
                        new.edge_directions
                            .iter()
                            .any(|new| new.abs_diff_eq(expected, 1.0e-4)
                                || new.abs_diff_eq(-expected, 1.0e-4))
                    );
                }
            }
            _ => panic!("rigid framing must preserve collider shape"),
        }
    }
}

#[test]
fn arbitrary_frame_rotates_cuboid_inertia_and_collider() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [2, 4, 6],
                BuildPose::new(IVec3::new(4, 8, 12), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap();
    assert_frame_compilation(graph);
}

#[test]
fn arbitrary_frame_composes_raw_region_convex_planes_and_mass() {
    let (mut graph, region) = region_over_one_block();
    let cell = i16::try_from(crate::POSITION_TICKS_PER_GRID_UNIT).unwrap();
    graph
        .apply(BuildCommand::SetRegionVertices {
            region,
            vertices: vec![([0, 1, 1], [0, -cell, 0]), ([1, 1, 1], [0, -cell, 0])],
        })
        .unwrap();
    assert_frame_compilation(graph);
}

#[test]
fn arbitrary_frame_does_not_double_transform_evaluated_geometry() {
    let mut graph = ConstructionGraph::new();
    let part = spawn(&mut graph, IVec3::ZERO);
    let owner = SolidOwner::Part(part);
    let edge = graph.evaluated_solid(owner).unwrap().logical_edges[0].key;
    graph
        .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
            [EdgeChainRef { owner, edge }],
            EdgeTreatment::Chamfer,
            20,
        )))
        .unwrap();
    assert_frame_compilation(graph);
}

fn spawn(graph: &mut ConstructionGraph, units: IVec3) -> crate::PartId {
    let BuildOutcome::Spawned(id) = graph.apply(BuildCommand::Spawn(cube_at(units))).unwrap()
    else {
        panic!("wrong spawn outcome")
    };
    id
}

fn ground(graph: &mut ConstructionGraph, part: crate::PartId) {
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(part, FaceKind::NegativeY),
            second: FaceRef::ground(),
        }))
        .unwrap();
}

#[test]
fn a_solid_cylinder_also_compiles_an_exact_hull_sharing_every_corner() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
            CylinderDimensions::new(0.95, 0.0, 0.25).unwrap(),
            BuildPose::from_position_ticks(IVec3::Y * 300, GridRotation::new(1, 0, 0)),
        )))
        .unwrap();
    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.colliders.len(), super::CYLINDER_COLLIDER_COUNT);
    assert_eq!(compiled.cylinders.len(), 1);
    let cylinder = compiled.cylinders[0];
    assert_eq!(cylinder.first_collider, 0);
    assert!((cylinder.outer_radius - 0.475).abs() < 1.0e-6);
    assert!((cylinder.half_length - 0.125).abs() < 1.0e-6);

    // The hull is the union of the boxes, so every box corner lies on or
    // inside it and the facet midpoints sit at the authored radius.
    let hull = cylinder.hull();
    assert_eq!(hull.vertices.len(), 2 * super::CYLINDER_COLLIDER_COUNT);
    assert_eq!(hull.face_planes.len(), super::CYLINDER_COLLIDER_COUNT + 2);
    for plane in &hull.face_planes {
        for vertex in &hull.vertices {
            assert!(plane.truncate().dot(*vertex) <= plane.w + 1.0e-5);
        }
    }
    // Each corner is one vertex, so the two faces meeting there agree exactly,
    // which sixteen independently rounded boxes cannot do.
    for vertex in &hull.vertices {
        let touching = hull
            .face_planes
            .iter()
            .filter(|plane| (plane.truncate().dot(*vertex) - plane.w).abs() < 1.0e-6)
            .count();
        assert_eq!(touching, 3, "a prism corner meets two sides and one end");
    }
}

#[test]
fn a_hollow_cylinder_or_sector_has_no_analytic_description() {
    for dimensions in [
        CylinderDimensions::new(1.0, 0.5, 0.25).unwrap(),
        CylinderDimensions::new(1.0, 0.0, 0.25)
            .unwrap()
            .with_sweep_angle_degrees(255)
            .unwrap(),
    ] {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
                dimensions,
                BuildPose::default(),
            )))
            .unwrap();
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.colliders.len(), super::CYLINDER_COLLIDER_COUNT);
        assert!(compiled.cylinders.is_empty());
    }
}

#[test]
fn hollow_cylinder_compiles_exact_mass_inertia_and_sixteen_colliders() {
    let mut graph = ConstructionGraph::new();
    let dimensions = CylinderDimensions::new(1.0, 0.5, 2.0).unwrap();
    let spec = CylinderSpec::new(dimensions, BuildPose::default());
    graph.apply(BuildCommand::SpawnCylinder(spec)).unwrap();

    let compiled = graph.compile().unwrap();
    let properties = compiled.compounds[0].mass_properties;
    let outer = 0.5_f32;
    let inner = 0.25_f32;
    let expected_mass = crate::ConstructionMaterial::Steel
        .properties()
        .density_kg_m3
        * core::f32::consts::PI
        * (outer * outer - inner * inner)
        * 2.0;
    let expected_axial = expected_mass * (outer * outer + inner * inner) * 0.5;
    let expected_transverse = expected_mass * (3.0 * (outer * outer + inner * inner) + 4.0) / 12.0;
    assert_eq!(compiled.colliders.len(), super::CYLINDER_COLLIDER_COUNT);
    assert!((properties.mass - expected_mass).abs() < 1.0e-3);
    assert!(properties.center_of_mass.abs_diff_eq(Vec3::ZERO, 1.0e-6));
    assert!((properties.inertia.x_axis.x - expected_transverse).abs() < 1.0e-3);
    assert!((properties.inertia.y_axis.y - expected_axial).abs() < 1.0e-3);
    assert!((properties.inertia.z_axis.z - expected_transverse).abs() < 1.0e-3);
    assert!(compiled.colliders.iter().all(|collider| {
        let ColliderShape::Cuboid { half_extents, .. } = collider.shape else {
            return false;
        };
        (half_extents.y - 1.0).abs() < 1.0e-6 && collider.local_center.length() >= inner - 1.0e-6
    }));
}

#[test]
fn hollow_pipe_tee_compiles_pipe_mass_and_leaves_open_passages() {
    use crate::{PipeArms, PipeJunctionDimensions, PipeJunctionSpec};
    let arms = PipeArms::single(FaceKind::NegativeX)
        .with(FaceKind::PositiveX)
        .with(FaceKind::PositiveY);
    let spec = PipeJunctionSpec::new(
        PipeJunctionDimensions::new(0.20, 0.10).unwrap(),
        arms,
        BuildPose::default(),
    );
    let mut graph = ConstructionGraph::new();
    graph.apply(BuildCommand::SpawnPipeJunction(spec)).unwrap();
    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.colliders.len(), spec.collider_count());

    let (radius, bore, reach) = (0.10_f32, 0.05_f32, 0.125_f32);
    let annulus = std::f32::consts::PI * (radius * radius - bore * bore);
    let density = ConstructionMaterial::Steel.properties().density_kg_m3;
    let properties = compiled.compounds[0].mass_properties;
    // The through pipe alone, and with a whole side arm added, bracket the tee.
    let through = annulus * 2.0 * reach * density;
    assert!(properties.mass > through * 0.97, "{}", properties.mass);
    assert!(properties.mass < through * 1.5, "{}", properties.mass);
    assert!(
        properties.center_of_mass.y > 1.0e-4,
        "the side arm adds mass above"
    );
    assert!(properties.center_of_mass.x.abs() < 1.0e-4);

    // No wall box covers a point travelling along an open bore.
    let bore_points = (-10_i16..=10)
        .map(|step| Vec3::X * (f32::from(step) * reach * 0.1))
        .chain((0_i16..=10).map(|step| Vec3::Y * (f32::from(step) * reach * 0.1)))
        .collect::<Vec<_>>();
    for collider in &compiled.colliders {
        let ColliderShape::Cuboid {
            local_rotation,
            half_extents,
        } = collider.shape
        else {
            panic!("junction walls are boxes");
        };
        let centre = collider.local_center + properties.center_of_mass;
        for &point in &bore_points {
            let local = local_rotation.inverse() * (point - centre);
            assert!(
                (local.abs() - half_extents).max_element() > -1.0e-5,
                "a wall box covers the open bore at {point}"
            );
        }
    }
}

#[test]
fn hollow_pipe_bend_compiles_exact_quarter_torus_mass_and_full_inertia() {
    let dimensions = PipeBendDimensions::new(0.50, 0.25, 4).unwrap();
    let spec = PipeBendSpec::new(dimensions, BuildPose::default());
    let mut graph = ConstructionGraph::new();
    graph.apply(BuildCommand::SpawnPipeBend(spec)).unwrap();
    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.colliders.len(), PIPE_BEND_COLLIDER_COUNT);

    let outer = 0.25_f32;
    let inner = 0.125_f32;
    let radius = 0.75_f32;
    let expected_volume = std::f32::consts::FRAC_PI_2
        * std::f32::consts::PI
        * radius
        * (outer * outer - inner * inner);
    let properties = compiled.compounds[0].mass_properties;
    assert!(
        (properties.mass
            - expected_volume * ConstructionMaterial::Steel.properties().density_kg_m3)
            .abs()
            < 1.0e-3
    );
    let mean_q = radius + (outer * outer + inner * inner) / (4.0 * radius);
    let expected_center = Vec3::new(
        -radius + 2.0 * mean_q / std::f32::consts::PI,
        radius - 2.0 * mean_q / std::f32::consts::PI,
        0.0,
    );
    assert!(
        properties
            .center_of_mass
            .abs_diff_eq(expected_center, 1.0e-5)
    );
    assert!((properties.inertia.x_axis.x - properties.inertia.y_axis.y).abs() < 1.0e-4);
    assert!(properties.inertia.x_axis.y.abs() > 1.0e-3);
    assert!((properties.inertia.x_axis.y - properties.inertia.y_axis.x).abs() < 1.0e-4);
}

#[test]
fn cylinder_sector_compiles_exact_offset_mass_properties_and_sixteen_colliders() {
    let mut graph = ConstructionGraph::new();
    let dimensions = CylinderDimensions::new(1.0, 0.5, 2.0)
        .unwrap()
        .with_sweep_angle_degrees(90)
        .unwrap();
    graph
        .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
            dimensions,
            BuildPose::default(),
        )))
        .unwrap();

    let compiled = graph.compile().unwrap();
    let properties = compiled.compounds[0].mass_properties;
    let outer = 0.5_f32;
    let inner = 0.25_f32;
    let length = 2.0_f32;
    let sweep = core::f32::consts::FRAC_PI_2;
    let expected_mass = crate::ConstructionMaterial::Steel
        .properties()
        .density_kg_m3
        * sweep
        * (outer * outer - inner * inner)
        * length
        * 0.5;
    let expected_center_x = 4.0 * (sweep * 0.5).sin() * (outer.powi(3) - inner.powi(3))
        / (3.0 * sweep * (outer * outer - inner * inner));
    let radial_squared = outer * outer + inner * inner;
    let radial_parallel = radial_squared * (sweep + sweep.sin()) / (4.0 * sweep);
    let radial_perpendicular = radial_squared * (sweep - sweep.sin()) / (4.0 * sweep);

    assert_eq!(compiled.colliders.len(), super::CYLINDER_COLLIDER_COUNT);
    assert!((properties.mass - expected_mass).abs() < 1.0e-3);
    assert!(
        properties
            .center_of_mass
            .abs_diff_eq(Vec3::new(expected_center_x, 0.0, 0.0), 1.0e-6)
    );
    assert!(
        (properties.inertia.x_axis.x
            - expected_mass * (length * length / 12.0 + radial_perpendicular))
            .abs()
            < 1.0e-3
    );
    assert!(
        (properties.inertia.y_axis.y
            - expected_mass
                * (radial_parallel + radial_perpendicular - expected_center_x * expected_center_x))
            .abs()
            < 1.0e-3
    );
    assert!(
        compiled
            .colliders
            .iter()
            .all(|collider| { (collider.local_center + properties.center_of_mass).x > 0.0 })
    );
}

fn bearing(
    graph: &mut ConstructionGraph,
    a: crate::PartId,
    face_a: FaceKind,
    b: crate::PartId,
    face_b: FaceKind,
    anchor: Vec3,
    axis: Vec3,
) {
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(a, face_a),
            FaceRef::part(b, face_b),
            anchor,
            axis,
        )))
        .unwrap();
}

/// One steel block at the origin, claimed as a region.
fn region_over_one_block() -> (ConstructionGraph, crate::RegionId) {
    let spec = CuboidSpec::new(
        [1, 1, 1],
        BuildPose::from_half_grid(IVec3::ONE, GridRotation::default()),
    )
    .unwrap();
    let mut graph = ConstructionGraph::new();
    graph.apply(BuildCommand::Spawn(spec)).unwrap();
    let region =
        crate::ShapeRegion::new(IVec3::ZERO, IVec3::ONE, ConstructionMaterial::Steel).unwrap();
    let BuildOutcome::RegionAdded(id) = graph.apply(BuildCommand::AddRegion(region)).unwrap()
    else {
        panic!("wrong outcome")
    };
    (graph, id)
}

#[test]
fn a_regions_blocks_contribute_no_colliders_of_their_own() {
    // The region owns the geometry; counting the block too would render and
    // collide the same material twice.
    let (graph, _) = region_over_one_block();
    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.colliders.len(), 1);
    let expected = ConstructionMaterial::Steel.properties().density_kg_m3 * 0.25_f32.powi(3);
    assert!(
        (compiled.compounds[0].mass_properties.mass - expected).abs() < 1.0e-3,
        "the region's mass must replace its block's, not add to it"
    );
}

#[test]
fn collapsing_a_region_edge_halves_its_mass_and_moves_its_centroid() {
    // The bounding box means a corner can only move inward, so the exact
    // case to check is a collapse rather than a shear: the top +z edge
    // driven down onto the bottom leaves a wedge of half the material,
    // whose centroid is the triangle's.
    let (mut graph, id) = region_over_one_block();
    let cell = i16::try_from(crate::POSITION_TICKS_PER_GRID_UNIT).unwrap();
    graph
        .apply(BuildCommand::SetRegionVertices {
            region: id,
            vertices: vec![([0, 1, 1], [0, -cell, 0]), ([1, 1, 1], [0, -cell, 0])],
        })
        .unwrap();

    let properties = graph.compile().unwrap().compounds[0].mass_properties;
    let density = ConstructionMaterial::Steel.properties().density_kg_m3;
    let expected_mass = density * 0.25_f32.powi(3) * 0.5;
    assert!(
        (properties.mass - expected_mass).abs() < 1.0e-3,
        "a wedge holds half a cell: {} vs {expected_mass}",
        properties.mass
    );
    let third = 0.25_f32 / 3.0;
    let expected_center = Vec3::new(0.125, third, third);
    assert!(
        properties
            .center_of_mass
            .abs_diff_eq(expected_center, 1.0e-4),
        "wedge centroid should be the triangle's: {} vs {expected_center}",
        properties.center_of_mass
    );
}

#[test]
fn a_cage_vertex_cannot_be_pushed_out_of_the_region() {
    // Shearing a face outward is exactly what the bounding box forbids.
    let (mut graph, id) = region_over_one_block();
    assert!(
        graph
            .apply(BuildCommand::SetRegionVertices {
                region: id,
                vertices: vec![([1, 1, 1], [5, 0, 0])],
            })
            .is_err(),
        "a corner already at the maximum has nowhere outward to go"
    );
}

#[test]
#[expect(clippy::cast_precision_loss, clippy::similar_names)]
fn region_inertia_matches_numerical_integration_of_the_same_pieces() {
    // The analytic integration is derived, so check it against a brute-force
    // sum over a dense sample of the solid it claims to describe.
    let (mut graph, id) = region_over_one_block();
    graph
        .apply(BuildCommand::SetRegionVertices {
            region: id,
            vertices: vec![
                ([1, 1, 1], [-8, -6, -4]),
                ([1, 1, 0], [-4, 0, 2]),
                ([0, 0, 1], [2, 3, -5]),
            ],
        })
        .unwrap();
    let properties = graph.compile().unwrap().compounds[0].mass_properties;

    let region = graph.region(id).unwrap();
    let pieces = super::region_pieces(region);
    let inside = |point: Vec3| {
        pieces.iter().any(|piece| match piece {
            PartPiece::Cuboid {
                center,
                half_extents,
                rotation,
                ..
            } => {
                let local = rotation.inverse() * (point - *center);
                local.abs().cmple(*half_extents).all()
            }
            PartPiece::Convex(convex) => convex
                .faces
                .iter()
                .all(|face| face.normal.dot(point) <= face.offset + 1.0e-7),
        })
    };

    let steps = 90;
    let low = Vec3::splat(-0.05);
    let cell = 0.35_f32 / steps as f32;
    let cell_volume = cell * cell * cell;
    let density = ConstructionMaterial::Steel.properties().density_kg_m3;
    let mut mass = 0.0_f32;
    let mut moment = Vec3::ZERO;
    let mut samples = Vec::new();
    for ix in 0..steps {
        for iy in 0..steps {
            for iz in 0..steps {
                let point =
                    low + Vec3::new(ix as f32 + 0.5, iy as f32 + 0.5, iz as f32 + 0.5) * cell;
                if inside(point) {
                    mass += density * cell_volume;
                    moment += point * (density * cell_volume);
                    samples.push(point);
                }
            }
        }
    }
    assert!(!samples.is_empty(), "the shaped solid must contain samples");
    let center = moment / mass;
    let mut inertia = Mat3::ZERO;
    for point in &samples {
        let arm = *point - center;
        inertia += (Mat3::IDENTITY * arm.length_squared() - outer_product(arm, arm))
            * (density * cell_volume);
    }

    assert!(
        (properties.mass - mass).abs() < mass * 0.02,
        "mass {} vs sampled {mass}",
        properties.mass
    );
    assert!(
        properties.center_of_mass.abs_diff_eq(center, 2.0e-3),
        "centre {} vs sampled {center}",
        properties.center_of_mass
    );
    for axis in 0..3 {
        let analytic = properties.inertia.col(axis)[axis];
        let sampled = inertia.col(axis)[axis];
        assert!(
            (analytic - sampled).abs() < sampled.abs() * 0.05,
            "inertia axis {axis}: {analytic} vs sampled {sampled}"
        );
    }
}

#[test]
fn an_unshaped_creation_still_compiles_to_one_collider_per_block() {
    // The regression guard: regions must cost nothing where none exist.
    let mut graph = ConstructionGraph::new();
    for offset in 0..4 {
        graph
            .apply(BuildCommand::Spawn(cube_at(IVec3::new(offset * 4, 0, 0))))
            .unwrap();
    }
    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.colliders.len(), 4);
    assert!(
        compiled
            .colliders
            .iter()
            .all(|collider| collider.shape.is_cuboid()),
        "an unshaped creation must produce only boxes"
    );
}

#[test]
fn featured_cuboid_compiles_evaluated_mass_and_convex_collision() {
    let mut graph = ConstructionGraph::new();
    let part = spawn(&mut graph, IVec3::ZERO);
    let owner = SolidOwner::Part(part);
    let edge = graph.evaluated_solid(owner).unwrap().logical_edges[0].key;
    graph
        .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
            [EdgeChainRef { owner, edge }],
            EdgeTreatment::Chamfer,
            20,
        )))
        .unwrap();

    let compiled = graph.compile().unwrap();
    let uncut_mass = ConstructionMaterial::Steel.properties().density_kg_m3;
    assert!(compiled.compounds[0].mass_properties.mass < uncut_mass);
    assert!(
        compiled
            .colliders
            .iter()
            .all(|collider| matches!(collider.shape, ColliderShape::Convex(_)))
    );
}

#[test]
fn welded_same_material_cuboids_compact_without_changing_mass_or_inertia() {
    let mut graph = ConstructionGraph::new();
    let a = spawn(&mut graph, IVec3::ZERO);
    let b = spawn(&mut graph, IVec3::new(4, 0, 0));
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(a, FaceKind::PositiveX),
            second: FaceRef::part(b, FaceKind::NegativeX),
        }))
        .unwrap();

    let compiled = graph.compile().unwrap();
    let properties = compiled.compounds[0].mass_properties;
    assert_eq!(compiled.compounds.len(), 1);
    assert_eq!(compiled.colliders.len(), 1);
    let ColliderShape::Cuboid {
        local_rotation,
        half_extents,
    } = compiled.colliders[0].shape
    else {
        panic!("compacted collider must remain a cuboid")
    };
    assert!(local_rotation.abs_diff_eq(Quat::IDENTITY, 1.0e-6));
    assert!(half_extents.abs_diff_eq(Vec3::new(1.0, 0.5, 0.5), 1.0e-6));
    let cube_mass = crate::ConstructionMaterial::Steel
        .properties()
        .density_kg_m3;
    assert!((properties.mass - cube_mass * 2.0).abs() < 1.0e-3);
    assert!(
        properties
            .center_of_mass
            .abs_diff_eq(Vec3::new(0.5, 0.0, 0.0), 1.0e-6)
    );
    assert!((properties.inertia.x_axis.x - cube_mass / 3.0).abs() < 1.0e-3);
    assert!((properties.inertia.y_axis.y - cube_mass * 5.0 / 6.0).abs() < 1.0e-3);
    assert!((properties.inertia.z_axis.z - cube_mass * 5.0 / 6.0).abs() < 1.0e-3);
}

#[test]
fn every_material_scales_cuboid_and_cylinder_mass() {
    for material in ConstructionMaterial::ALL {
        let density = material.properties().density_kg_m3;
        let mut cuboid_graph = ConstructionGraph::new();
        cuboid_graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([4; 3], BuildPose::default())
                    .unwrap()
                    .with_material(material),
            ))
            .unwrap();
        let cuboid = cuboid_graph.compile().unwrap();
        assert!((cuboid.compounds[0].mass_properties.mass - density).abs() < 1.0e-3);
        assert!(
            (cuboid.compounds[0].mass_properties.inertia.x_axis.x - density / 6.0).abs() < 1.0e-3
        );

        let mut cylinder_graph = ConstructionGraph::new();
        cylinder_graph
            .apply(BuildCommand::SpawnCylinder(
                CylinderSpec::new(
                    CylinderDimensions::new(1.0, 0.0, 1.0).unwrap(),
                    BuildPose::default(),
                )
                .with_material(material),
            ))
            .unwrap();
        let cylinder = cylinder_graph.compile().unwrap();
        let expected = density * core::f32::consts::PI * 0.25;
        assert!((cylinder.compounds[0].mass_properties.mass - expected).abs() < 1.0e-3);
    }
}

#[test]
fn mixed_material_welds_sum_mass_and_keep_collider_contact_properties() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(aluminium) = graph
        .apply(BuildCommand::Spawn(
            cube_at(IVec3::ZERO).with_material(ConstructionMaterial::Aluminium),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::Spawned(wood) = graph
        .apply(BuildCommand::Spawn(
            cube_at(IVec3::new(4, 0, 0)).with_material(ConstructionMaterial::Wood),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(aluminium, FaceKind::PositiveX),
            second: FaceRef::part(wood, FaceKind::NegativeX),
        }))
        .unwrap();

    let compiled = graph.compile().unwrap();
    let mass = compiled.compounds[0].mass_properties;
    assert!((mass.mass - 3_400.0).abs() < 1.0e-3);
    assert!(
        mass.center_of_mass
            .abs_diff_eq(Vec3::new(700.0 / 3_400.0, 0.0, 0.0), 1.0e-6)
    );
    let contacts = compiled
        .colliders
        .iter()
        .map(|collider| collider.material_properties)
        .collect::<Vec<_>>();
    assert!(contacts.contains(&ConstructionMaterial::Aluminium.properties()));
    assert!(contacts.contains(&ConstructionMaterial::Wood.properties()));
}

#[test]
fn shared_bearing_attachments_compile_as_one_rotor_and_one_joint() {
    let mut graph = ConstructionGraph::new();
    let support = CuboidSpec::new(
        [4, 4, 4],
        BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(support) = graph.apply(BuildCommand::Spawn(support)).unwrap() else {
        unreachable!()
    };
    let targets = [IVec3::new(0, 9, 0), IVec3::new(2, 9, 0)].map(|center| {
        let spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(center, GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        part
    });
    let dimensions = BearingDimensions::new(0.80, 0.10).unwrap();
    for target in targets {
        graph
            .apply(BuildCommand::AddBearing(
                BearingSpec::new(
                    FaceRef::part(support, FaceKind::PositiveY),
                    FaceRef::part(target, FaceKind::NegativeY),
                    Vec3::Y,
                    Vec3::Y,
                )
                .with_dimensions(dimensions),
            ))
            .unwrap();
    }
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec {
            first: targets[0],
            second: targets[1],
        }))
        .unwrap();

    let compiled = graph.compile().unwrap();
    let compound_for = |part| {
        compiled
            .part_to_compound
            .iter()
            .find_map(|&(candidate, compound)| (candidate == part).then_some(compound))
            .unwrap()
    };
    assert_eq!(compiled.compounds.len(), 2);
    assert_eq!(compiled.bearings.len(), 1);
    assert_eq!(compound_for(targets[0]), compound_for(targets[1]));
    assert_ne!(compound_for(support), compound_for(targets[0]));
}

#[test]
fn welding_to_ground_makes_only_that_group_static() {
    let mut graph = ConstructionGraph::new();
    let grounded = spawn(&mut graph, IVec3::new(0, 2, 0));
    spawn(&mut graph, IVec3::new(8, 2, 0));
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(grounded, FaceKind::NegativeY),
            second: FaceRef::ground(),
        }))
        .unwrap();

    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.compounds.len(), 2);
    assert!(compiled.compounds[0].is_static);
    assert!(compiled.compounds[0].mass_properties.inverse_mass.abs() < f32::EPSILON);
    assert!(!compiled.compounds[1].is_static);
}

#[test]
fn external_world_anchor_makes_only_its_rigid_group_static() {
    let mut graph = ConstructionGraph::new();
    let anchored = spawn(&mut graph, IVec3::new(0, 20, 0));
    let floating = spawn(&mut graph, IVec3::new(8, 20, 0));

    let ordinary = graph.compile().unwrap();
    assert!(
        ordinary
            .compounds
            .iter()
            .all(|compound| !compound.is_static)
    );

    let compiled = graph.compile_with_static_parts([anchored]).unwrap();
    let compound_for = |part| {
        compiled
            .part_to_compound
            .iter()
            .find_map(|&(candidate, compound)| (candidate == part).then_some(compound))
            .unwrap()
    };
    assert!(compiled.compounds[compound_for(anchored) as usize].is_static);
    assert!(!compiled.compounds[compound_for(floating) as usize].is_static);
}

/// Two blocks on a bearing, then welded to each other as well.
fn bearing_welded_shut() -> (ConstructionGraph, BearingId) {
    let mut graph = ConstructionGraph::new();
    let a = spawn(&mut graph, IVec3::ZERO);
    let b = spawn(&mut graph, IVec3::new(4, 0, 0));
    let BuildOutcome::BearingAdded(bearing) = graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(a, FaceKind::PositiveX),
            FaceRef::part(b, FaceKind::NegativeX),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::X,
        )))
        .unwrap()
    else {
        panic!("wrong bearing outcome")
    };
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(a, FaceKind::PositiveX),
            second: FaceRef::part(b, FaceKind::NegativeX),
        }))
        .unwrap();
    (graph, bearing)
}

#[test]
fn bearing_collapsed_by_a_later_weld_stops_being_a_joint() {
    let (graph, _) = bearing_welded_shut();

    let compiled = graph.compile().unwrap();

    // The weld already holds the two sides together, so the bearing adds
    // no coordinate and no constraint — it is locked, not unbuildable.
    assert_eq!(compiled.compounds.len(), 1);
    assert!(compiled.bearings.is_empty());
    assert!(compiled.loop_topology.tree_bearings.is_empty());
    assert!(compiled.loop_topology.closure_bearings.is_empty());
}

#[test]
fn a_driven_bearing_collapsed_by_a_later_weld_is_rejected() {
    let (mut graph, bearing) = bearing_welded_shut();
    wire_with(
        &mut graph,
        bearing,
        DriveLimits::default(),
        DriveProgram::default(),
        false,
    );

    assert_eq!(
        graph.compile(),
        Err(TopologyError::SelfBearing {
            bearing,
            compound: 0
        })
    );
}

#[test]
fn bearing_dimensions_do_not_change_compiled_physics() {
    let compile_with = |dimensions: BearingDimensions| {
        let mut graph = ConstructionGraph::new();
        let a = spawn(&mut graph, IVec3::ZERO);
        let b = spawn(&mut graph, IVec3::new(4, 0, 0));
        graph
            .apply(BuildCommand::AddBearing(
                BearingSpec::new(
                    FaceRef::part(a, FaceKind::PositiveX),
                    FaceRef::part(b, FaceKind::NegativeX),
                    Vec3::new(0.5, 0.0, 0.0),
                    Vec3::X,
                )
                .with_dimensions(dimensions),
            ))
            .unwrap();
        graph.compile().unwrap()
    };

    assert_eq!(
        compile_with(BearingDimensions::default()),
        compile_with(BearingDimensions::new(0.50, 0.20).unwrap())
    );
}

#[test]
fn closed_square_has_one_hard_closure_edge() {
    let mut graph = ConstructionGraph::new();
    let a = spawn(&mut graph, IVec3::ZERO);
    let b = spawn(&mut graph, IVec3::new(4, 0, 0));
    let c = spawn(&mut graph, IVec3::new(4, 4, 0));
    let d = spawn(&mut graph, IVec3::new(0, 4, 0));
    let edges = [
        (
            a,
            FaceKind::PositiveX,
            b,
            FaceKind::NegativeX,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::X,
        ),
        (
            b,
            FaceKind::PositiveY,
            c,
            FaceKind::NegativeY,
            Vec3::new(1.0, 0.5, 0.0),
            Vec3::Y,
        ),
        (
            c,
            FaceKind::NegativeX,
            d,
            FaceKind::PositiveX,
            Vec3::new(0.5, 1.0, 0.0),
            Vec3::NEG_X,
        ),
        (
            d,
            FaceKind::NegativeY,
            a,
            FaceKind::PositiveY,
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::NEG_Y,
        ),
    ];
    for (source, source_face, target, target_face, anchor, axis) in edges {
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(source, source_face),
                FaceRef::part(target, target_face),
                anchor,
                axis,
            )))
            .unwrap();
    }

    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.loop_topology.tree_bearings.len(), 3);
    assert_eq!(compiled.loop_topology.closure_bearings.len(), 1);
    assert_eq!(
        compiled.loop_topology.mechanism_components,
        vec![vec![0, 1, 2, 3]]
    );
    assert_eq!(compiled.collision_suppression.len(), 4);
}

#[test]
fn floating_branch_has_one_canonical_root_and_stable_traversals() {
    let mut graph = ConstructionGraph::new();
    let root = spawn(&mut graph, IVec3::ZERO);
    let x_child = spawn(&mut graph, IVec3::new(4, 0, 0));
    let y_child = spawn(&mut graph, IVec3::new(0, 4, 0));
    bearing(
        &mut graph,
        root,
        FaceKind::PositiveX,
        x_child,
        FaceKind::NegativeX,
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::X,
    );
    bearing(
        &mut graph,
        root,
        FaceKind::PositiveY,
        y_child,
        FaceKind::NegativeY,
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::Y,
    );

    let topology = graph.compile().unwrap().loop_topology;
    assert_eq!(topology.component_roots, vec![vec![0]]);
    assert!(topology.body_parents[0].is_root);
    assert_eq!(topology.body_parents[1].parent_body, 0);
    assert_eq!(topology.body_parents[2].parent_body, 0);
    assert_eq!(topology.body_parents[1].preorder_index, 1);
    assert_eq!(topology.body_parents[2].preorder_index, 2);
    assert_eq!(topology.contraction_rounds, vec![vec![1, 2]]);
}

#[test]
fn mirrored_floating_branches_use_heavy_chassis_root_and_equal_inertia() {
    let mut graph = ConstructionGraph::new();
    let left = spawn(&mut graph, IVec3::new(-2, 0, 0));
    let right = spawn(&mut graph, IVec3::new(2, 0, 0));
    let BuildOutcome::Spawned(chassis) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [8, 4, 8],
                BuildPose::new(IVec3::new(0, 4, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    bearing(
        &mut graph,
        chassis,
        FaceKind::NegativeY,
        left,
        FaceKind::PositiveY,
        Vec3::new(-0.5, 0.5, 0.0),
        Vec3::NEG_Y,
    );
    bearing(
        &mut graph,
        chassis,
        FaceKind::NegativeY,
        right,
        FaceKind::PositiveY,
        Vec3::new(0.5, 0.5, 0.0),
        Vec3::NEG_Y,
    );

    let topology = graph.compile().unwrap().loop_topology;
    assert_eq!(topology.component_roots, vec![vec![2]]);
    assert!(topology.body_parents[2].is_root);
    assert_eq!(topology.body_parents[0].parent_body, 2);
    assert_eq!(topology.body_parents[1].parent_body, 2);
    assert_eq!(topology.coordinate_axis_inertia.len(), 2);
    assert!(
        (topology.coordinate_axis_inertia[0] - topology.coordinate_axis_inertia[1]).abs() < 1.0e-4,
        "mirrored branches compiled unequal inertia: {:?}",
        topology.coordinate_axis_inertia
    );
}

#[test]
fn grounded_body_is_root_even_when_not_the_first_compound() {
    let mut graph = ConstructionGraph::new();
    let child = spawn(&mut graph, IVec3::new(0, 6, 0));
    let root = spawn(&mut graph, IVec3::new(0, 2, 0));
    ground(&mut graph, root);
    bearing(
        &mut graph,
        child,
        FaceKind::NegativeY,
        root,
        FaceKind::PositiveY,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::NEG_Y,
    );

    let topology = graph.compile().unwrap().loop_topology;
    assert_eq!(topology.component_roots, vec![vec![1]]);
    assert_eq!(topology.body_parents[0].parent_body, 1);
    assert_eq!(topology.body_parents[0].bearing_direction, 1);
    assert!(topology.body_parents[1].is_root);
}

#[test]
fn multiple_ground_anchors_remain_fixed_roots_with_a_closure_edge() {
    let mut graph = ConstructionGraph::new();
    let left = spawn(&mut graph, IVec3::new(0, 2, 0));
    let middle = spawn(&mut graph, IVec3::new(4, 2, 0));
    let right = spawn(&mut graph, IVec3::new(8, 2, 0));
    ground(&mut graph, left);
    ground(&mut graph, right);
    bearing(
        &mut graph,
        left,
        FaceKind::PositiveX,
        middle,
        FaceKind::NegativeX,
        Vec3::new(0.5, 0.5, 0.0),
        Vec3::X,
    );
    bearing(
        &mut graph,
        middle,
        FaceKind::PositiveX,
        right,
        FaceKind::NegativeX,
        Vec3::new(1.5, 0.5, 0.0),
        Vec3::X,
    );

    let topology = graph.compile().unwrap().loop_topology;
    assert_eq!(topology.component_roots, vec![vec![0, 2]]);
    assert_eq!(topology.tree_bearings.len(), 1);
    assert_eq!(topology.closure_bearings.len(), 1);
    assert!(topology.body_parents[0].is_root);
    assert_eq!(topology.body_parents[1].parent_body, 0);
    assert!(topology.body_parents[2].is_root);
}

fn add_bearing(
    graph: &mut ConstructionGraph,
    source: PartId,
    source_face: FaceKind,
    target: PartId,
    target_face: FaceKind,
    anchor: Vec3,
    axis: Vec3,
) -> BearingId {
    let BuildOutcome::BearingAdded(bearing) = graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(source, source_face),
            FaceRef::part(target, target_face),
            anchor,
            axis,
        )))
        .unwrap()
    else {
        unreachable!()
    };
    bearing
}

fn wire_with(
    graph: &mut ConstructionGraph,
    bearing: BearingId,
    limits: DriveLimits,
    program: DriveProgram,
    reversed: bool,
) -> PartId {
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::new(IVec3::new(0, 40, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let mut spec = DriveLinkSpec::new(controller, bearing);
    spec.limits = limits;
    spec.program = program;
    spec.reversed = reversed;
    spec.actuator = ActuatorAssignment::motor(100, 0).unwrap();
    graph.apply(BuildCommand::AddDriveLink(spec)).unwrap();
    let BuildOutcome::Spawned(engine) = graph
        .apply(BuildCommand::SpawnEngine(EngineSpec::new(
            EngineKind::Electric,
            BuildPose::new(IVec3::new(0, 42, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(controller, FaceKind::PositiveY),
            second: FaceRef::part(engine, FaceKind::NegativeY),
        }))
        .unwrap();
    controller
}

fn wire(graph: &mut ConstructionGraph, bearing: BearingId, reversed: bool) -> PartId {
    wire_with(
        graph,
        bearing,
        DriveLimits::default(),
        DriveProgram::default(),
        reversed,
    )
}

fn square_loop(graph: &mut ConstructionGraph) -> [BearingId; 4] {
    let a = spawn(graph, IVec3::ZERO);
    let b = spawn(graph, IVec3::new(4, 0, 0));
    let c = spawn(graph, IVec3::new(4, 4, 0));
    let d = spawn(graph, IVec3::new(0, 4, 0));
    [
        add_bearing(
            graph,
            a,
            FaceKind::PositiveX,
            b,
            FaceKind::NegativeX,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::X,
        ),
        add_bearing(
            graph,
            b,
            FaceKind::PositiveY,
            c,
            FaceKind::NegativeY,
            Vec3::new(1.0, 0.5, 0.0),
            Vec3::Y,
        ),
        add_bearing(
            graph,
            c,
            FaceKind::NegativeX,
            d,
            FaceKind::PositiveX,
            Vec3::new(0.5, 1.0, 0.0),
            Vec3::NEG_X,
        ),
        add_bearing(
            graph,
            d,
            FaceKind::NegativeY,
            a,
            FaceKind::PositiveY,
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::NEG_Y,
        ),
    ]
}

#[test]
fn control_block_compiles_as_one_half_by_half_by_quarter_metre_collider() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::default(),
        )))
        .unwrap();

    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.colliders.len(), 1);
    assert_eq!(
        compiled.colliders[0].shape,
        ColliderShape::Cuboid {
            local_rotation: Quat::IDENTITY,
            half_extents: Vec3::new(0.25, 0.25, 0.125),
        }
    );
    let expected_mass = crate::MACHINE_PART_DENSITY_KG_M3 * 0.5 * 0.5 * 0.25;
    assert!((compiled.compounds[0].mass_properties.mass - expected_mass).abs() < 1.0e-4);
}

#[test]
fn driven_bearing_is_preferred_as_a_tree_edge_over_a_passive_one() {
    // The last edge of the square would normally become the closure. Driving
    // it must push the closure onto a passive edge instead.
    let mut graph = ConstructionGraph::new();
    let bearings = square_loop(&mut graph);
    let driven = bearings[3];
    wire(&mut graph, driven, false);

    let compiled = graph.compile().unwrap();
    assert!(compiled.loop_topology.tree_bearings.contains(&driven));
    assert_eq!(compiled.loop_topology.closure_bearings.len(), 1);
    assert!(!compiled.loop_topology.closure_bearings.contains(&driven));
}

#[test]
fn driven_bearing_forced_onto_a_closure_edge_is_rejected() {
    let mut graph = ConstructionGraph::new();
    let bearings = square_loop(&mut graph);
    for bearing in bearings {
        wire(&mut graph, bearing, false);
    }

    let Err(TopologyError::DrivenClosureBearing { bearing }) = graph.compile() else {
        panic!("a fully driven loop cannot give every edge a coordinate")
    };
    assert!(bearings.contains(&bearing));
}

#[test]
fn coordinate_axis_inertia_matches_a_hand_computed_arm() {
    let mut graph = ConstructionGraph::new();
    let base = spawn(&mut graph, IVec3::new(0, 2, 0));
    ground(&mut graph, base);
    let arm = spawn(&mut graph, IVec3::new(4, 2, 0));
    add_bearing(
        &mut graph,
        base,
        FaceKind::PositiveX,
        arm,
        FaceKind::NegativeX,
        Vec3::new(0.5, 0.5, 0.0),
        Vec3::X,
    );

    let compiled = graph.compile().unwrap();
    let inertia = compiled.loop_topology.coordinate_axis_inertia[0];
    // The arm is a 1 m cube centred 0.5 m along the +x hinge axis, so the
    // radial offset is zero and only its own x inertia contributes.
    let mass = crate::ConstructionMaterial::Steel
        .properties()
        .density_kg_m3;
    let expected = mass * (1.0 + 1.0) / 12.0;
    assert!(
        (inertia - expected).abs() < 1.0e-2,
        "axis inertia {inertia} should be about {expected}"
    );
}

#[test]
fn coordinate_axis_inertia_includes_the_whole_child_subtree() {
    let mut graph = ConstructionGraph::new();
    let base = spawn(&mut graph, IVec3::new(0, 2, 0));
    ground(&mut graph, base);
    let first = spawn(&mut graph, IVec3::new(0, 6, 0));
    let second = spawn(&mut graph, IVec3::new(0, 10, 0));
    add_bearing(
        &mut graph,
        base,
        FaceKind::PositiveY,
        first,
        FaceKind::NegativeY,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::Y,
    );
    add_bearing(
        &mut graph,
        first,
        FaceKind::PositiveY,
        second,
        FaceKind::NegativeY,
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::Y,
    );

    let inertia = graph
        .compile()
        .unwrap()
        .loop_topology
        .coordinate_axis_inertia;
    assert_eq!(inertia.len(), 2);
    assert!(
        inertia[0] > inertia[1],
        "the lower joint carries both links: {inertia:?}"
    );
}

#[test]
fn duplicate_bearing_rows_share_one_coordinate() {
    let mut graph = ConstructionGraph::new();
    let base = spawn(&mut graph, IVec3::new(0, 2, 0));
    ground(&mut graph, base);
    let arm = spawn(&mut graph, IVec3::new(4, 2, 0));
    let first = add_bearing(
        &mut graph,
        base,
        FaceKind::PositiveX,
        arm,
        FaceKind::NegativeX,
        Vec3::new(0.5, 0.5, 0.0),
        Vec3::X,
    );
    // A second row describing the same physical joint: same compounds, same
    // anchor, same axis. Compilation collapses it, but a drive may still be
    // wired to whichever row the app happens to hold.
    let duplicate = add_bearing(
        &mut graph,
        base,
        FaceKind::PositiveX,
        arm,
        FaceKind::NegativeX,
        Vec3::new(0.5, 0.5, 0.0),
        Vec3::X,
    );

    let compiled = graph.compile().unwrap();
    let coordinates = &compiled.loop_topology.bearing_coordinates;
    assert_eq!(compiled.loop_topology.tree_bearings.len(), 1);
    assert_eq!(coordinates.get(&first), coordinates.get(&duplicate));
    assert!(coordinates.contains_key(&duplicate));
}

#[test]
fn closure_bearings_have_no_coordinate_to_drive() {
    let mut graph = ConstructionGraph::new();
    let bearings = square_loop(&mut graph);
    let compiled = graph.compile().unwrap();

    let coordinates = &compiled.loop_topology.bearing_coordinates;
    for closure in &compiled.loop_topology.closure_bearings {
        assert!(!coordinates.contains_key(closure));
    }
    for tree in &compiled.loop_topology.tree_bearings {
        assert!(coordinates.contains_key(tree));
    }
    assert_eq!(
        coordinates.len(),
        bearings.len() - compiled.loop_topology.closure_bearings.len()
    );
}

#[test]
fn coordinate_drives_apply_per_wire_reverse_and_leave_undriven_rows_passive() {
    let mut graph = ConstructionGraph::new();
    let base = spawn(&mut graph, IVec3::new(0, 2, 0));
    ground(&mut graph, base);
    let first = spawn(&mut graph, IVec3::new(0, 6, 0));
    let second = spawn(&mut graph, IVec3::new(0, 10, 0));
    let lower = add_bearing(
        &mut graph,
        base,
        FaceKind::PositiveY,
        first,
        FaceKind::NegativeY,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::Y,
    );
    add_bearing(
        &mut graph,
        first,
        FaceKind::PositiveY,
        second,
        FaceKind::NegativeY,
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::Y,
    );
    let limits = DriveLimits::new(4.0, 20.0, Some((-1.0, 1.0))).unwrap();
    let program =
        DriveProgram::new(&[DriveState::new(DriveTarget::Speed(3.0)).unwrap()], false).unwrap();
    wire_with(&mut graph, lower, limits, program, true);

    let compiled = graph.compile().unwrap();
    let drives = compiled.resolve_coordinate_drives(&graph);
    assert_eq!(drives.len(), 2);
    let driven_index = compiled
        .loop_topology
        .tree_bearings
        .iter()
        .position(|&bearing| bearing == lower)
        .unwrap();
    let motor_row = drives[driven_index];
    assert_eq!(motor_row.mode, DriveMode::Speed);
    assert!((motor_row.target_speed + 3.0).abs() < f32::EPSILON);
    assert!((motor_row.min_angle + 1.0).abs() < f32::EPSILON);
    assert!((motor_row.max_angle - 1.0).abs() < f32::EPSILON);
    let expected = 500.0 / compiled.loop_topology.coordinate_axis_inertia[driven_index];
    assert!((motor_row.max_acceleration - expected).abs() < 1.0e-3);

    let passive_index = 1 - driven_index;
    assert_eq!(drives[passive_index], CoordinateDrive::PASSIVE);
}

#[test]
fn hardware_torque_replaces_the_legacy_arbitrary_torque_limit() {
    let mut graph = ConstructionGraph::new();
    let base = spawn(&mut graph, IVec3::new(0, 2, 0));
    ground(&mut graph, base);
    let arm = spawn(&mut graph, IVec3::new(4, 2, 0));
    let bearing = add_bearing(
        &mut graph,
        base,
        FaceKind::PositiveX,
        arm,
        FaceKind::NegativeX,
        Vec3::new(0.5, 0.5, 0.0),
        Vec3::X,
    );
    wire(&mut graph, bearing, false);

    let compiled = graph.compile().unwrap();
    let drives = compiled.resolve_coordinate_drives(&graph);
    assert!(drives[0].max_acceleration.is_finite());
    assert!(drives[0].max_acceleration > 0.0);
    assert!(drives[0].min_angle.is_infinite() && drives[0].min_angle < 0.0);
}

#[test]
fn one_engine_splits_its_torque_across_its_assigned_coordinates() {
    let mut graph = ConstructionGraph::new();
    let base = spawn(&mut graph, IVec3::new(0, 2, 0));
    ground(&mut graph, base);
    let right = spawn(&mut graph, IVec3::new(4, 2, 0));
    let left = spawn(&mut graph, IVec3::new(-4, 2, 0));
    let bearings = [
        add_bearing(
            &mut graph,
            base,
            FaceKind::PositiveX,
            right,
            FaceKind::NegativeX,
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::X,
        ),
        add_bearing(
            &mut graph,
            base,
            FaceKind::NegativeX,
            left,
            FaceKind::PositiveX,
            Vec3::new(-0.5, 0.5, 0.0),
            -Vec3::X,
        ),
    ];
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::new(IVec3::new(0, 40, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::Spawned(engine) = graph
        .apply(BuildCommand::SpawnEngine(EngineSpec::new(
            EngineKind::Electric,
            BuildPose::new(IVec3::new(0, 42, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(controller, FaceKind::PositiveY),
            second: FaceRef::part(engine, FaceKind::NegativeY),
        }))
        .unwrap();
    for bearing in bearings {
        let mut link = DriveLinkSpec::new(controller, bearing);
        link.actuator = ActuatorAssignment::motor(100, 0).unwrap();
        link.program =
            DriveProgram::new(&[DriveState::new(DriveTarget::Speed(20.0)).unwrap()], false)
                .unwrap();
        graph.apply(BuildCommand::AddDriveLink(link)).unwrap();
    }

    let compiled = graph.compile().unwrap();
    let drives = compiled.resolve_coordinate_drives(&graph);
    assert_eq!(drives.len(), 2);
    for (coordinate, drive) in drives.iter().enumerate() {
        let torque = drive.source_a_max_acceleration
            * compiled.loop_topology.coordinate_axis_inertia[coordinate];
        assert!((torque - 250.0).abs() < 1.0e-3);
        assert!((drive.max_speed - 4.0 * core::f32::consts::PI).abs() < 1.0e-5);
        assert!((drive.target_speed - 4.0 * core::f32::consts::PI).abs() < 1.0e-5);
    }
}

#[test]
fn ideal_gearing_multiplies_torque_divides_speed_and_can_disengage_one_family() {
    let mut graph = ConstructionGraph::new();
    let base = spawn(&mut graph, IVec3::new(0, 2, 0));
    ground(&mut graph, base);
    let arm = spawn(&mut graph, IVec3::new(4, 2, 0));
    let bearing = add_bearing(
        &mut graph,
        base,
        FaceKind::PositiveX,
        arm,
        FaceKind::NegativeX,
        Vec3::new(0.5, 0.5, 0.0),
        Vec3::X,
    );
    let controller = wire(&mut graph, bearing, false);
    let compiled = graph.compile().unwrap();
    let direct = compiled.resolve_coordinate_drives(&graph)[0];
    let geared = compiled.resolve_coordinate_drives_with_gears(
        &graph,
        &[GearSelection {
            controller,
            kind: EngineKind::Electric,
            ratio: Some(4.0),
        }],
    )[0];
    assert!(
        (geared.source_a_max_acceleration / direct.source_a_max_acceleration - 4.0).abs() < 1.0e-5
    );
    assert!((direct.source_a_no_load_speed / geared.source_a_no_load_speed - 4.0).abs() < 1.0e-5);

    let disengaged = compiled.resolve_coordinate_drives_with_gears(
        &graph,
        &[GearSelection {
            controller,
            kind: EngineKind::Electric,
            ratio: None,
        }],
    )[0];
    assert_eq!(disengaged, CoordinateDrive::PASSIVE);
}

#[test]
fn assigned_motor_without_a_touching_engine_is_rejected() {
    let mut graph = ConstructionGraph::new();
    let base = spawn(&mut graph, IVec3::new(0, 2, 0));
    ground(&mut graph, base);
    let arm = spawn(&mut graph, IVec3::new(4, 2, 0));
    let bearing = add_bearing(
        &mut graph,
        base,
        FaceKind::PositiveX,
        arm,
        FaceKind::NegativeX,
        Vec3::new(0.5, 0.5, 0.0),
        Vec3::X,
    );
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::new(IVec3::new(0, 40, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let mut link = DriveLinkSpec::new(controller, bearing);
    link.actuator = ActuatorAssignment::motor(100, 0).unwrap();
    link.program =
        DriveProgram::new(&[DriveState::new(DriveTarget::Speed(1.0)).unwrap()], false).unwrap();
    graph.apply(BuildCommand::AddDriveLink(link)).unwrap();

    assert_eq!(
        graph.compile(),
        Err(TopologyError::InsufficientElectricPorts {
            controller,
            required: 1,
            available: 0,
        })
    );
}

fn layered_steel_wheel() -> CylinderSpec {
    CylinderSpec::new(
        CylinderDimensions::new(1.0, 0.0, 2.0).unwrap(),
        BuildPose::default(),
    )
    .with_layer(
        crate::LayerFace::OuterWall,
        0.25,
        crate::ConstructionMaterial::Rubber,
        crate::MaterialAppearance::BAKED,
    )
    .unwrap()
}

#[test]
fn layered_cylinder_mass_sums_band_densities() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::SpawnCylinder(layered_steel_wheel()))
        .unwrap();
    let properties = graph.compile().unwrap().compounds[0].mass_properties;
    let density = |material: crate::ConstructionMaterial| material.properties().density_kg_m3;
    let pi = core::f32::consts::PI;
    let steel = density(crate::ConstructionMaterial::Steel) * pi * 0.25 * 2.0;
    let rubber = density(crate::ConstructionMaterial::Rubber) * pi * (0.5625 - 0.25) * 2.0;
    let axial = steel * 0.25 * 0.5 + rubber * (0.5625 + 0.25) * 0.5;
    // Bands are 24-sided prisms, a percent or two under the true circle.
    assert!((properties.mass - (steel + rubber)).abs() < 0.02 * properties.mass);
    assert!(properties.center_of_mass.abs_diff_eq(Vec3::ZERO, 1.0e-4));
    assert!((properties.inertia.y_axis.y - axial).abs() < 0.04 * axial);
}

#[test]
fn layered_cuboid_mass_sums_band_densities() {
    let core = CuboidSpec::new([4, 4, 4], BuildPose::default()).unwrap();
    let layered = PartSpec::Cuboid(core)
        .with_layer(
            crate::LayerFace::Face(crate::FaceKind::PositiveY),
            0.25,
            crate::ConstructionMaterial::Rubber,
            crate::MaterialAppearance::BAKED,
        )
        .unwrap()
        .as_cuboid()
        .unwrap();
    let mut graph = ConstructionGraph::new();
    graph.apply(BuildCommand::Spawn(layered)).unwrap();
    let compiled = graph.compile().unwrap();
    let properties = compiled.compounds[0].mass_properties;
    let steel = crate::ConstructionMaterial::Steel
        .properties()
        .density_kg_m3;
    let rubber = crate::ConstructionMaterial::Rubber
        .properties()
        .density_kg_m3
        * 0.25;
    assert!((properties.mass - (steel + rubber)).abs() < 1.0e-3 * properties.mass);
    // The core's centre sits at y = 0, the layer's at 0.625 m.
    let center_y = rubber * 0.625 / (steel + rubber);
    assert!((properties.center_of_mass.y - center_y).abs() < 1.0e-3);
    for material in [
        crate::ConstructionMaterial::Steel,
        crate::ConstructionMaterial::Rubber,
    ] {
        assert!(
            compiled
                .colliders
                .iter()
                .any(|collider| collider.material_properties == material.properties()),
            "{material:?} band has colliders"
        );
    }
}

#[test]
fn layered_wheel_keeps_analytic_contact_with_its_tyre_material() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::SpawnCylinder(layered_steel_wheel()))
        .unwrap();
    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.cylinders.len(), 1);
    assert!((compiled.cylinders[0].outer_radius - 0.75).abs() < 1.0e-4);
    assert!(compiled.colliders.iter().all(|collider| {
        collider.material_properties == crate::ConstructionMaterial::Rubber.properties()
    }));
}

#[test]
fn featured_layer_colliders_carry_their_band_material() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::SpawnCylinder(layered_steel_wheel()))
        .unwrap()
    else {
        panic!("spawning a cylinder reports its part");
    };
    let owner = crate::SolidOwner::Part(part);
    let rim = graph
        .evaluated_solid(owner)
        .unwrap()
        .logical_edges
        .iter()
        .find(|edge| edge.closed && edge.convex)
        .unwrap()
        .key;
    graph
        .apply(BuildCommand::AddShapeFeature(crate::ShapeFeature::new(
            [crate::EdgeChainRef { owner, edge: rim }],
            crate::EdgeTreatment::Fillet,
            120,
        )))
        .unwrap();
    let compiled = graph.compile().unwrap();
    for material in [
        crate::ConstructionMaterial::Steel,
        crate::ConstructionMaterial::Rubber,
    ] {
        assert!(
            compiled
                .colliders
                .iter()
                .any(|collider| collider.material_properties == material.properties()),
            "{material:?} band has colliders"
        );
    }
}

fn auger(hand: crate::SpiralHand, taper: Option<crate::SpiralTaper>) -> CylinderSpec {
    CylinderSpec::new(
        CylinderDimensions::new(0.5, 0.0, 2.0).unwrap(),
        BuildPose::default(),
    )
    .with_spiral(
        crate::SpiralSpec::new(
            100,
            1,
            hand,
            crate::SpiralProfile::square(10, 60).unwrap(),
            crate::SpiralProfile::PLAIN,
            taper,
        )
        .unwrap(),
    )
    .unwrap()
}

fn compiled_auger(spec: CylinderSpec) -> crate::CompiledCreation {
    let mut graph = ConstructionGraph::new();
    graph.apply(BuildCommand::SpawnCylinder(spec)).unwrap();
    graph.compile().unwrap()
}

#[test]
fn a_spiral_cylinder_weighs_its_core_and_its_ridge() {
    let compiled = compiled_auger(auger(crate::SpiralHand::Right, None));
    let properties = compiled.compounds[0].mass_properties;
    let density = crate::ConstructionMaterial::Steel
        .properties()
        .density_kg_m3;
    // Eight turns of a 2.5 cm ridge from the 10 cm core out to 25 cm.
    let core = core::f32::consts::PI * 0.1 * 0.1 * 2.0;
    let ridge = 8.0 * 0.025 * core::f32::consts::PI * (0.25 * 0.25 - 0.1 * 0.1);
    let expected = density * (core + ridge);
    assert!(
        (properties.mass - expected).abs() < expected * 0.02,
        "{} against {expected}",
        properties.mass
    );
    assert!(properties.center_of_mass.length() < 0.005);
    let plain = density * core::f32::consts::PI * 0.25 * 0.25 * 2.0;
    assert!(properties.mass < plain * 0.5);
}

#[test]
fn a_spiral_cylinder_collides_as_core_boxes_and_convex_ridge_runs() {
    let spec = auger(crate::SpiralHand::Right, None);
    let compiled = compiled_auger(spec);
    assert!(compiled.cylinders.is_empty());
    assert!(compiled.colliders.len() > super::CYLINDER_COLLIDER_COUNT + 8 * 12 - 1);
    assert!(compiled.colliders.len() <= super::cylinder_collider_count(spec));
    for (row, collider) in compiled.colliders.iter().enumerate() {
        match &collider.shape {
            ColliderShape::Cuboid { half_extents, .. } => {
                assert!(row < super::CYLINDER_COLLIDER_COUNT);
                assert!((half_extents.x - 0.05).abs() < 1.0e-6);
            }
            ColliderShape::Convex(convex) => {
                assert!(row >= super::CYLINDER_COLLIDER_COUNT);
                for vertex in &convex.vertices {
                    assert!(vertex.x.hypot(vertex.z) < 0.25 + 1.0e-4, "{vertex}");
                    assert!(vertex.y.abs() < 1.0 + 1.0e-4);
                }
                // A hull: no vertex lies outside any face.
                for plane in &convex.face_planes {
                    for vertex in &convex.vertices {
                        assert!(plane.truncate().dot(*vertex) - plane.w < 1.0e-4);
                    }
                }
            }
        }
    }
}

#[test]
fn a_left_hand_spiral_mirrors_a_right_hand_one() {
    let ridges = |hand| {
        compiled_auger(auger(hand, None))
            .colliders
            .iter()
            .filter(|collider| matches!(collider.shape, ColliderShape::Convex(_)))
            .map(|collider| collider.local_center)
            .collect::<Vec<_>>()
    };
    let right = ridges(crate::SpiralHand::Right);
    let left = ridges(crate::SpiralHand::Left);
    assert_eq!(right.len(), left.len());
    for centre in right {
        let mirrored = Vec3::new(centre.x, centre.y, -centre.z);
        assert!(
            left.iter().any(|other| other.distance(mirrored) < 1.0e-3),
            "no left-hand ridge run at {mirrored}"
        );
    }
}

#[test]
fn a_tapered_end_narrows_core_and_ridge_to_the_tip() {
    let taper = crate::SpiralTaper {
        end: crate::SpiralEnd::NegativeY,
        length_ticks: 200,
        tip_diameter_ticks: 20,
    };
    let tapered = compiled_auger(auger(crate::SpiralHand::Right, Some(taper)));
    let straight = compiled_auger(auger(crate::SpiralHand::Right, None));
    assert!(tapered.compounds[0].mass_properties.mass < straight.compounds[0].mass_properties.mass);
    let centre = tapered.compounds[0].mass_properties.center_of_mass;
    assert!(centre.y > 0.02, "{centre}");
    let mut reached_tip = false;
    for collider in &tapered.colliders {
        let ColliderShape::Convex(convex) = &collider.shape else {
            continue;
        };
        for vertex in &convex.vertices {
            let vertex = *vertex + centre;
            // Radii shrink linearly from 25 cm at half a metre above the tip
            // to 2.5 cm at it.
            let allowed = 0.025 + (0.25 - 0.025) * ((vertex.y + 1.0) / 0.5).clamp(0.0, 1.0);
            assert!(vertex.x.hypot(vertex.z) < allowed + 2.0e-3, "{vertex}");
            reached_tip |= vertex.y < -0.999;
        }
    }
    assert!(reached_tip);
}
