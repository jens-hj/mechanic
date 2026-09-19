use bevy_math::{IVec3, Vec3};

use super::{
    AppearanceTarget, BearingDimensionError, BearingDimensions, BearingSpec, BuildCommand,
    BuildOutcome, ConstructionGraph, DriveLinkSpec, GraphError, PendingOperation, RigidLinkSpec,
    WeldSpec,
};
use crate::{
    ActuatorAssignment, BearingId, BuildPose, ConstructionMaterial, ControllerSpec, CuboidSpec,
    CylinderDimensions, CylinderSpec, DimensionLinkId, DimensionLinkSpec, DriveLimits, DriveName,
    DriveProgram, DriveState, DriveTarget, EngineKind, EngineSpec, FaceKind, FaceRef, GridRotation,
    InputSeatLinkSpec, InputSpec, MaterialAppearance, MaterialColor, MaterialDye, MaterialFinish,
    PartId, PartSpec, RegionError, RegionId, SeatControllerLinkSpec, SeatSpec, ShapeRegion,
    ShiftMode, TransmissionSpec,
};

fn suspension_fixture() -> (ConstructionGraph, [BearingSpec; 2]) {
    let mut graph = ConstructionGraph::new();
    let source = spawn(&mut graph, cube_at(0));
    let targets = [150, 250].map(|y| {
        spawn(
            &mut graph,
            CuboidSpec::new(
                [1; 3],
                BuildPose::from_position_ticks(IVec3::new(450, y, 0), GridRotation::default()),
            )
            .unwrap(),
        )
    });
    // The bearing attachment workflow groups construction on the opposite plate.
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec {
            first: targets[0],
            second: targets[1],
        }))
        .unwrap();
    let spec = crate::SuspensionSpec::new(
        Some(crate::SpringSpec::default()),
        Some(crate::ShockSpec::default()),
        None,
    )
    .unwrap();
    let bearings = targets.map(|target| {
        BearingSpec::new(
            FaceRef::part(source, FaceKind::PositiveX),
            FaceRef::part(target, FaceKind::NegativeX),
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::X,
        )
        .with_kind(crate::JointKind::Suspension(spec))
    });
    (graph, bearings)
}

#[test]
fn suspension_shared_mounts_compile_once_and_live_edits_preserve_attachment_ids() {
    let (mut graph, bearings) = suspension_fixture();
    let ids = bearings.map(|spec| {
        let BuildOutcome::BearingAdded(id) = graph.apply(BuildCommand::AddBearing(spec)).unwrap()
        else {
            unreachable!()
        };
        id
    });
    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.bearings.len(), 1);
    assert_eq!(compiled.compounds.len(), 2);
    let changed = crate::SuspensionSpec::new(
        Some(crate::SpringSpec::new(0.5, 0.16, 0.12, 6, 0.025).unwrap()),
        Some(
            crate::ShockSpec::new(0.5, 0.1, crate::ShockBodyEnd::Opposite, 0.0, 2.0, 3.0).unwrap(),
        ),
        None,
    )
    .unwrap();
    graph
        .apply(BuildCommand::SetSuspension {
            bearing: ids[0],
            spec: changed,
        })
        .unwrap();
    assert_eq!(graph.bearing_count(), 2);
    for id in ids {
        assert_eq!(
            graph.bearing(id).unwrap().kind,
            crate::JointKind::Suspension(changed)
        );
    }
    assert_eq!(graph.compile().unwrap().bearings.len(), 1);
    let moved = crate::SuspensionSpec::new(
        changed.spring(),
        Some(
            crate::ShockSpec::new(0.5, 0.1, crate::ShockBodyEnd::Opposite, 0.025, 2.0, 3.0)
                .unwrap(),
        ),
        None,
    )
    .unwrap();
    assert_eq!(
        graph.apply(BuildCommand::SetSuspension {
            bearing: ids[0],
            spec: moved
        }),
        Err(GraphError::Suspension(
            crate::SuspensionError::AttachedSpacing
        ))
    );
    for id in ids {
        assert_eq!(
            graph.bearing(id).unwrap().kind,
            crate::JointKind::Suspension(changed)
        );
    }
}

#[test]
fn suspension_attachment_rejects_bad_axes_support_spacing_and_conflicting_specs() {
    let (mut graph, bearings) = suspension_fixture();
    for axis in [Vec3::ZERO, Vec3::X * 2.0, Vec3::Y, Vec3::splat(f32::NAN)] {
        let mut bad = bearings[0];
        bad.axis = axis;
        assert_eq!(
            graph.apply(BuildCommand::AddBearing(bad)),
            Err(GraphError::InvalidBearingAxis)
        );
    }
    for anchor in [
        Vec3::new(0.6, 0.5, 0.0),
        Vec3::new(0.5, 9.0, 0.0),
        Vec3::splat(f32::NAN),
    ] {
        let mut bad = bearings[0];
        bad.shared_anchor = anchor;
        assert_eq!(
            graph.apply(BuildCommand::AddBearing(bad)),
            Err(GraphError::BearingAnchorOutsideFaces)
        );
    }
    let mut wrong_spacing = bearings[0];
    wrong_spacing.kind = crate::JointKind::Suspension(
        crate::SuspensionSpec::new(
            Some(crate::SpringSpec::default()),
            Some(
                crate::ShockSpec::new(0.5, 0.1, crate::ShockBodyEnd::Source, 0.025, 1.0, 1.6)
                    .unwrap(),
            ),
            None,
        )
        .unwrap(),
    );
    assert_eq!(
        graph.apply(BuildCommand::AddBearing(wrong_spacing)),
        Err(GraphError::BearingAnchorOutsideFaces)
    );
    graph.apply(BuildCommand::AddBearing(bearings[0])).unwrap();
    let mut bad = bearings[1];
    bad.kind = crate::JointKind::Suspension(
        crate::SuspensionSpec::new(Some(crate::SpringSpec::default()), None, None).unwrap(),
    );
    assert_eq!(
        graph.apply(BuildCommand::AddBearing(bad)),
        Err(GraphError::Suspension(crate::SuspensionError::SharedMounts))
    );
    assert_eq!(graph.bearing_count(), 1);
}

#[test]
fn suspension_cannot_receive_a_controller_motor_assignment() {
    let (mut graph, bearings) = suspension_fixture();
    let BuildOutcome::BearingAdded(bearing) =
        graph.apply(BuildCommand::AddBearing(bearings[0])).unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::new(IVec3::new(20, 2, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    assert_eq!(
        graph.apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
            controller, bearing
        ))),
        Err(GraphError::IncompatibleDrive)
    );
}

fn cube_at(x: i32) -> CuboidSpec {
    CuboidSpec::new(
        [4, 4, 4],
        BuildPose::new(IVec3::new(x, 2, 0), GridRotation::default()),
    )
    .unwrap()
}

fn spawn(graph: &mut ConstructionGraph, spec: CuboidSpec) -> crate::PartId {
    let BuildOutcome::Spawned(id) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
        panic!("spawn returned wrong outcome")
    };
    id
}

fn spawn_cylinder(
    graph: &mut ConstructionGraph,
    dimensions: CylinderDimensions,
    pose: BuildPose,
) -> crate::PartId {
    let BuildOutcome::Spawned(id) = graph
        .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
            dimensions, pose,
        )))
        .unwrap()
    else {
        unreachable!()
    };
    id
}

#[test]
fn dimension_link_ids_are_unique_and_queryable() {
    let mut graph = ConstructionGraph::new();
    let spec = DimensionLinkSpec::new(DimensionLinkId(9), BuildPose::default());
    let BuildOutcome::Spawned(link) = graph.apply(BuildCommand::SpawnDimensionLink(spec)).unwrap()
    else {
        unreachable!()
    };
    assert_eq!(graph.dimension_link(DimensionLinkId(9)), Some(link));
    assert_eq!(graph.dimension_link_id(link), Some(DimensionLinkId(9)));
    assert_eq!(
        graph.apply(BuildCommand::SpawnDimensionLink(spec)),
        Err(GraphError::DuplicateDimensionLink(DimensionLinkId(9)))
    );
}

#[test]
fn structural_component_crosses_every_joint_and_partitions_cleanly() {
    let mut graph = ConstructionGraph::new();
    let welded = spawn(&mut graph, cube_at(0));
    let hinged = spawn(&mut graph, cube_at(4));
    let linked = spawn(&mut graph, cube_at(8));
    let cargo = spawn(&mut graph, cube_at(20));
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(welded, FaceKind::PositiveX),
            second: FaceRef::part(hinged, FaceKind::NegativeX),
        }))
        .unwrap();
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(hinged, FaceKind::PositiveX),
            FaceRef::part(linked, FaceKind::NegativeX),
            Vec3::new(1.5, 0.5, 0.0),
            Vec3::X,
        )))
        .unwrap();
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec {
            first: linked,
            second: cargo,
        }))
        .unwrap();

    let component = graph.structural_component(welded, [cargo]).unwrap();
    assert_eq!(component.parts().count(), 4);
    assert!(component.touches_authored_ground());
    let partition = graph.partition(&component);
    assert_eq!(partition.component.part_count(), 4);
    assert_eq!(partition.component.weld_count(), 1);
    assert_eq!(partition.component.bearing_count(), 1);
    assert_eq!(partition.component.rigid_link_count(), 1);
    assert_eq!(partition.remainder.part_count(), 0);
}

#[test]
fn input_routes_are_typed_cardinal_and_cascade_with_the_seat() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(input) = graph
        .apply(BuildCommand::SpawnInput(InputSpec::new(
            BuildPose::default(),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::Spawned(seat) = graph
        .apply(BuildCommand::SpawnSeat(SeatSpec::new(BuildPose::default())))
        .unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::default(),
        )))
        .unwrap()
    else {
        unreachable!()
    };

    graph
        .apply(BuildCommand::AddInputSeatLink(InputSeatLinkSpec {
            input,
            seat,
        }))
        .unwrap();
    graph
        .apply(BuildCommand::AddSeatControllerLink(
            SeatControllerLinkSpec { seat, controller },
        ))
        .unwrap();
    assert_eq!(graph.seat_input(seat), Some(input));
    assert_eq!(graph.seat_controller(seat), Some(controller));
    assert_eq!(
        graph.apply(BuildCommand::AddInputSeatLink(InputSeatLinkSpec {
            input,
            seat,
        })),
        Err(GraphError::InputAlreadyLinked(input))
    );

    graph.apply(BuildCommand::Remove(seat)).unwrap();
    assert_eq!(graph.input_seat_links().count(), 0);
    assert_eq!(graph.seat_controller_links().count(), 0);
    assert!(graph.part(input).is_some());
    assert!(graph.part(controller).is_some());
}

#[test]
fn structural_blocks_do_not_bridge_controller_actuator_modules() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::from_half_grid(IVec3::ZERO, GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let bridge = spawn(
        &mut graph,
        CuboidSpec::new(
            [1, 2, 1],
            BuildPose::from_half_grid(IVec3::new(3, 0, 0), GridRotation::default()),
        )
        .unwrap(),
    );
    let BuildOutcome::Spawned(engine) = graph
        .apply(BuildCommand::SpawnEngine(EngineSpec::new(
            EngineKind::Electric,
            BuildPose::from_half_grid(IVec3::new(6, 0, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    for (first, second) in [(controller, bridge), (bridge, engine)] {
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(first, FaceKind::PositiveX),
                second: FaceRef::part(second, FaceKind::NegativeX),
            }))
            .unwrap();
    }

    let inventory = graph.actuator_inventory(controller).unwrap();
    assert_eq!(inventory.electric_engines, 0);
}

#[test]
fn graph_stores_generalized_parts_and_rejects_cylinder_walls_as_faces() {
    let mut graph = ConstructionGraph::new();
    let cylinder = spawn_cylinder(
        &mut graph,
        CylinderDimensions::default(),
        BuildPose::default(),
    );
    assert!(matches!(graph.part(cylinder), Some(PartSpec::Cylinder(_))));
    assert_eq!(
        graph.apply(BuildCommand::BeginPending(PendingOperation::Weld(
            FaceRef::part(cylinder, FaceKind::PositiveX)
        ))),
        Err(GraphError::InvalidCylinderFace)
    );
    assert!(graph.pending().is_none());
}

#[test]
fn mixed_welds_require_positive_annular_material_overlap() {
    let mut graph = ConstructionGraph::new();
    let cylinder = spawn_cylinder(
        &mut graph,
        CylinderDimensions::new(1.0, 0.5, 0.25).unwrap(),
        BuildPose::default(),
    );
    let centered = spawn(
        &mut graph,
        CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(bevy_math::IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap(),
    );
    let weld = |part| {
        BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(cylinder, FaceKind::PositiveY),
            second: FaceRef::part(part, FaceKind::NegativeY),
        })
    };
    assert_eq!(
        graph.apply(weld(centered)),
        Err(GraphError::FacesDoNotTouch)
    );

    let ring = spawn(
        &mut graph,
        CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(bevy_math::IVec3::new(3, 2, 0), GridRotation::default()),
        )
        .unwrap(),
    );
    assert!(matches!(
        graph.apply(weld(ring)),
        Ok(BuildOutcome::Welded(_))
    ));
}

#[test]
fn cylinder_sector_end_connects_only_through_retained_material() {
    let dimensions = CylinderDimensions::new(1.0, 0.0, 0.25)
        .unwrap()
        .with_sweep_angle_degrees(90)
        .unwrap();
    let attempt = |x_half_units| {
        let mut graph = ConstructionGraph::new();
        let cylinder = spawn_cylinder(&mut graph, dimensions, BuildPose::default());
        let block = spawn(
            &mut graph,
            CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_half_grid(IVec3::new(x_half_units, 2, 0), GridRotation::default()),
            )
            .unwrap(),
        );
        graph.apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(cylinder, FaceKind::PositiveY),
            second: FaceRef::part(block, FaceKind::NegativeY),
        }))
    };

    assert!(matches!(attempt(3), Ok(BuildOutcome::Welded(_))));
    assert_eq!(attempt(-3), Err(GraphError::FacesDoNotTouch));
}

#[test]
fn bearing_dimensions_validate_defaults_bounds_and_ring_gap() {
    let default = BearingDimensions::default();
    assert!((default.outer_diameter() - 0.25).abs() < f32::EPSILON);
    assert!((default.inner_diameter() - 0.10).abs() < f32::EPSILON);

    let solid_minimum = BearingDimensions::new(0.05, 0.0).unwrap();
    assert!((solid_minimum.outer_diameter() - 0.05).abs() < f32::EPSILON);
    assert!(solid_minimum.inner_diameter().abs() < f32::EPSILON);
    assert!(BearingDimensions::new(8.0, 7.95).is_ok());
    assert!(BearingDimensions::new(1.234, 0.678).is_ok());

    assert_eq!(
        BearingDimensions::new(f32::NAN, 0.0),
        Err(BearingDimensionError::NonFiniteOuterDiameter)
    );
    assert_eq!(
        BearingDimensions::new(f32::INFINITY, 0.0),
        Err(BearingDimensionError::NonFiniteOuterDiameter)
    );
    assert_eq!(
        BearingDimensions::new(0.049, 0.0),
        Err(BearingDimensionError::OuterDiameterOutOfRange)
    );
    assert_eq!(
        BearingDimensions::new(8.001, 0.0),
        Err(BearingDimensionError::OuterDiameterOutOfRange)
    );
    assert_eq!(
        BearingDimensions::new(0.25, f32::NAN),
        Err(BearingDimensionError::NonFiniteInnerDiameter)
    );
    assert_eq!(
        BearingDimensions::new(0.25, -0.001),
        Err(BearingDimensionError::InnerDiameterOutOfRange)
    );
    assert_eq!(
        BearingDimensions::new(0.25, 0.201),
        Err(BearingDimensionError::InnerDiameterOutOfRange)
    );
}

#[test]
fn bearing_spec_defaults_and_accepts_custom_dimensions() {
    let source = FaceRef::ground();
    let target = FaceRef::ground();
    let default = BearingSpec::new(source, target, Vec3::ZERO, Vec3::Y);
    assert_eq!(default.dimensions, BearingDimensions::default());

    let custom = BearingDimensions::new(2.5, 1.25).unwrap();
    assert_eq!(default.with_dimensions(custom).dimensions, custom);
}

#[test]
fn weld_requires_touching_opposed_faces() {
    let mut graph = ConstructionGraph::new();
    let left = spawn(&mut graph, cube_at(0));
    let right = spawn(&mut graph, cube_at(4));
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(left, FaceKind::PositiveX),
            second: FaceRef::part(right, FaceKind::NegativeX),
        }))
        .unwrap();
    assert_eq!(graph.weld_count(), 1);
}

#[test]
fn failed_command_is_transactional() {
    let mut graph = ConstructionGraph::new();
    let left = spawn(&mut graph, cube_at(0));
    let right = spawn(&mut graph, cube_at(8));
    let result = graph.apply(BuildCommand::Weld(WeldSpec {
        first: FaceRef::part(left, FaceKind::PositiveX),
        second: FaceRef::part(right, FaceKind::NegativeX),
    }));

    assert_eq!(result, Err(GraphError::FacesDoNotTouch));
    assert_eq!(graph.part_count(), 2);
    assert_eq!(graph.weld_count(), 0);
}

#[test]
fn deletion_cascades_connections_and_invalidates_old_handle() {
    let mut graph = ConstructionGraph::new();
    let left = spawn(&mut graph, cube_at(0));
    let right = spawn(&mut graph, cube_at(4));
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(left, FaceKind::PositiveX),
            second: FaceRef::part(right, FaceKind::NegativeX),
        }))
        .unwrap();

    graph.apply(BuildCommand::Remove(left)).unwrap();

    assert!(graph.part(left).is_none());
    assert_eq!(graph.weld_count(), 0);
}

#[test]
fn rigid_links_validate_distinct_parts_and_cascade_on_deletion() {
    let mut graph = ConstructionGraph::new();
    let first = spawn(&mut graph, cube_at(0));
    let second = spawn(&mut graph, cube_at(8));
    assert_eq!(
        graph.apply(BuildCommand::RigidLink(RigidLinkSpec {
            first,
            second: first,
        })),
        Err(GraphError::SameRigidLinkPart)
    );

    let BuildOutcome::RigidLinked(link) = graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec { first, second }))
        .unwrap()
    else {
        unreachable!()
    };
    assert_eq!(graph.rigid_link_count(), 1);
    assert_eq!(graph.rigid_link(link).unwrap().second, second);

    graph.apply(BuildCommand::Remove(first)).unwrap();
    assert_eq!(graph.rigid_link_count(), 0);
    assert_eq!(
        graph.apply(BuildCommand::RemoveRigidLink(link)),
        Err(GraphError::MissingRigidLink(link))
    );
}

#[test]
fn bearing_axis_and_anchor_are_derived_geometry() {
    let mut graph = ConstructionGraph::new();
    let left = spawn(&mut graph, cube_at(0));
    let right = spawn(&mut graph, cube_at(4));
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(left, FaceKind::PositiveX),
            FaceRef::part(right, FaceKind::NegativeX),
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::X,
        )))
        .unwrap();

    assert_eq!(graph.bearing_count(), 1);
}

#[test]
fn oversized_bearing_can_attach_to_an_offset_face_covered_by_its_ring() {
    let mut graph = ConstructionGraph::new();
    let source = spawn(&mut graph, cube_at(0));
    let target = spawn(
        &mut graph,
        CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(IVec3::new(5, 9, 1), GridRotation::default()),
        )
        .unwrap(),
    );
    let dimensions = BearingDimensions::new(2.0, 0.10).unwrap();

    graph
        .apply(BuildCommand::AddBearing(
            BearingSpec::new(
                FaceRef::part(source, FaceKind::PositiveX),
                FaceRef::part(target, FaceKind::NegativeX),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::X,
            )
            .with_dimensions(dimensions),
        ))
        .unwrap();

    assert_eq!(graph.bearing_count(), 1);
}

#[test]
fn bearing_does_not_attach_to_a_face_entirely_inside_its_hole() {
    let mut graph = ConstructionGraph::new();
    let source = spawn(&mut graph, cube_at(0));
    let target = spawn(
        &mut graph,
        CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(IVec3::new(5, 4, 0), GridRotation::default()),
        )
        .unwrap(),
    );
    let dimensions = BearingDimensions::new(2.0, 1.0).unwrap();

    assert_eq!(
        graph.apply(BuildCommand::AddBearing(
            BearingSpec::new(
                FaceRef::part(source, FaceKind::PositiveX),
                FaceRef::part(target, FaceKind::NegativeX),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::X,
            )
            .with_dimensions(dimensions),
        )),
        Err(GraphError::BearingAnchorOutsideFaces)
    );
}

#[test]
fn connections_can_be_deleted_independently() {
    let mut graph = ConstructionGraph::new();
    let left = spawn(&mut graph, cube_at(0));
    let right = spawn(&mut graph, cube_at(4));
    let BuildOutcome::BearingAdded(bearing) = graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(left, FaceKind::PositiveX),
            FaceRef::part(right, FaceKind::NegativeX),
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::X,
        )))
        .unwrap()
    else {
        panic!("wrong bearing outcome")
    };

    graph.apply(BuildCommand::RemoveBearing(bearing)).unwrap();

    assert_eq!(graph.part_count(), 2);
    assert_eq!(graph.bearing_count(), 0);
    assert_eq!(
        graph.apply(BuildCommand::RemoveBearing(bearing)),
        Err(GraphError::MissingBearing(bearing))
    );
}

fn controller_at(x: i32) -> ControllerSpec {
    ControllerSpec::new(BuildPose::from_half_grid(
        IVec3::new(x, 2, 0),
        GridRotation::default(),
    ))
}

fn hinged_pair(graph: &mut ConstructionGraph) -> BearingId {
    let left = spawn(graph, cube_at(0));
    let right = spawn(graph, cube_at(4));
    let BuildOutcome::BearingAdded(bearing) = graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(left, FaceKind::PositiveX),
            FaceRef::part(right, FaceKind::NegativeX),
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::X,
        )))
        .unwrap()
    else {
        unreachable!()
    };
    bearing
}

fn spawn_controller(graph: &mut ConstructionGraph, spec: ControllerSpec) -> PartId {
    let BuildOutcome::Spawned(id) = graph.apply(BuildCommand::SpawnController(spec)).unwrap()
    else {
        unreachable!()
    };
    id
}

#[test]
fn control_block_exposes_cuboid_faces_and_reprogrammable_wires() {
    let mut graph = ConstructionGraph::new();
    let bearing = hinged_pair(&mut graph);
    let controller = spawn_controller(&mut graph, controller_at(20));
    assert!(matches!(
        graph.part(controller),
        Some(PartSpec::Controller(_))
    ));
    assert!(
        graph
            .face_geometry(FaceRef::part(controller, FaceKind::PositiveX))
            .is_ok()
    );
    assert!(graph.is_controller(controller));

    let BuildOutcome::DriveLinked(link) = graph
        .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
            controller, bearing,
        )))
        .unwrap()
    else {
        unreachable!()
    };
    assert_eq!(
        graph.drive_link(link).map(|spec| spec.limits),
        Some(DriveLimits::default())
    );

    let limits = DriveLimits::new(2.0, 12.0, Some((-0.5, 0.5))).unwrap();
    let program =
        DriveProgram::new(&[DriveState::new(DriveTarget::Angle(0.25)).unwrap()], false).unwrap();
    assert_eq!(
        graph.apply(BuildCommand::SetDriveLink {
            link,
            limits,
            program,
            name: DriveName::new("Steer · front left"),
            actuator: ActuatorAssignment::Unpowered,
        }),
        Ok(BuildOutcome::DriveUpdated)
    );
    let stored = graph.drive_link(link).copied().unwrap();
    assert_eq!(stored.limits, limits);
    assert_eq!(stored.program, program);
    assert_eq!(stored.name.as_str(), "Steer · front left");

    // A control block owns its wires, and reprogramming one leaves the
    // block itself untouched.
    assert_eq!(graph.controller_links(controller).count(), 1);
}

#[test]
fn drive_link_requires_a_controller_part_and_an_undriven_bearing() {
    let mut graph = ConstructionGraph::new();
    let bearing = hinged_pair(&mut graph);
    let block = spawn(&mut graph, cube_at(12));
    let controller = spawn_controller(&mut graph, controller_at(20));
    let other = spawn_controller(&mut graph, controller_at(28));

    assert_eq!(
        graph.apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
            block, bearing
        ))),
        Err(GraphError::NotAController(block))
    );
    assert_eq!(
        graph.apply(BuildCommand::BeginPending(PendingOperation::DriveLink(
            block
        ))),
        Err(GraphError::NotAController(block))
    );

    assert!(matches!(
        graph.apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
            controller, bearing
        ))),
        Ok(BuildOutcome::DriveLinked(_))
    ));
    assert_eq!(graph.drive_link_count(), 1);
    assert_eq!(
        graph.apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
            other, bearing
        ))),
        Err(GraphError::BearingAlreadyDriven(bearing))
    );
}

#[test]
fn reversed_wire_flips_the_programmed_direction() {
    let mut graph = ConstructionGraph::new();
    let bearing = hinged_pair(&mut graph);
    let controller = spawn_controller(&mut graph, controller_at(20));
    let program =
        DriveProgram::new(&[DriveState::new(DriveTarget::Speed(2.0)).unwrap()], false).unwrap();

    let mut spec = DriveLinkSpec::new(controller, bearing);
    spec.program = program;
    let BuildOutcome::DriveLinked(link) = graph.apply(BuildCommand::AddDriveLink(spec)).unwrap()
    else {
        unreachable!()
    };
    assert_eq!(
        graph
            .drive_link(link)
            .and_then(|spec| spec.resolved_target(0)),
        Some(DriveTarget::Speed(2.0))
    );

    graph.apply(BuildCommand::RemoveDriveLink(link)).unwrap();
    assert!(graph.bearing_drive_link(bearing).is_none());

    spec.reversed = true;
    graph.apply(BuildCommand::AddDriveLink(spec)).unwrap();
    assert_eq!(
        graph
            .bearing_drive_link(bearing)
            .and_then(|(_, link)| link.resolved_target(0)),
        Some(DriveTarget::Speed(-2.0))
    );
}

#[test]
fn deleting_a_controller_or_bearing_cascades_its_drive_links() {
    let mut graph = ConstructionGraph::new();
    let bearing = hinged_pair(&mut graph);
    let controller = spawn_controller(&mut graph, controller_at(20));
    let BuildOutcome::DriveLinked(link) = graph
        .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
            controller, bearing,
        )))
        .unwrap()
    else {
        unreachable!()
    };

    graph.apply(BuildCommand::RemoveBearing(bearing)).unwrap();
    assert_eq!(graph.drive_link_count(), 0);
    assert_eq!(
        graph.apply(BuildCommand::RemoveDriveLink(link)),
        Err(GraphError::MissingDriveLink(link))
    );

    let mut graph = ConstructionGraph::new();
    let bearing = hinged_pair(&mut graph);
    let controller = spawn_controller(&mut graph, controller_at(20));
    graph
        .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
            controller, bearing,
        )))
        .unwrap();
    graph.apply(BuildCommand::Remove(controller)).unwrap();
    assert_eq!(graph.drive_link_count(), 0);
    assert_eq!(graph.bearing_count(), 1);
}

#[test]
fn deleting_a_bearings_support_part_also_drops_its_drive_wire() {
    let mut graph = ConstructionGraph::new();
    let left = spawn(&mut graph, cube_at(0));
    let right = spawn(&mut graph, cube_at(4));
    let BuildOutcome::BearingAdded(bearing) = graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(left, FaceKind::PositiveX),
            FaceRef::part(right, FaceKind::NegativeX),
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::X,
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let controller = spawn_controller(&mut graph, controller_at(20));
    graph
        .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
            controller, bearing,
        )))
        .unwrap();

    graph.apply(BuildCommand::Remove(left)).unwrap();

    assert_eq!(graph.bearing_count(), 0);
    assert_eq!(graph.drive_link_count(), 0);
}

/// One construction cell, in the steps a cage vertex moves in.
fn cell_steps() -> i16 {
    i16::try_from(crate::POSITION_TICKS_PER_GRID_UNIT).expect("a cell is twenty steps")
}

/// A solid run of `size` one-cell blocks welded together from the origin.
fn welded_blocks(size: IVec3, material: ConstructionMaterial) -> ConstructionGraph {
    let mut graph = ConstructionGraph::new();
    let mut previous: Option<(PartId, IVec3)> = None;
    for z in 0..size.z {
        for y in 0..size.y {
            for x in 0..size.x {
                let at = IVec3::new(x, y, z);
                let spec = CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_half_grid(IVec3::ONE + at * 2, GridRotation::default()),
                )
                .unwrap()
                .with_material(material);
                let BuildOutcome::Spawned(id) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
                else {
                    panic!("wrong spawn outcome")
                };
                // Weld everything into one body so a region may claim it.
                if let Some((earlier, _)) = previous {
                    graph
                        .apply(BuildCommand::RigidLink(RigidLinkSpec {
                            first: earlier,
                            second: id,
                        }))
                        .unwrap();
                }
                previous = Some((id, at));
            }
        }
    }
    graph
}

fn add_region(
    graph: &mut ConstructionGraph,
    size: IVec3,
    material: ConstructionMaterial,
) -> Result<RegionId, GraphError> {
    let region = ShapeRegion::new(IVec3::ZERO, size, material).unwrap();
    match graph.apply(BuildCommand::AddRegion(region))? {
        BuildOutcome::RegionAdded(id) => Ok(id),
        other => panic!("wrong outcome {other:?}"),
    }
}

#[test]
fn a_fresh_region_has_eight_cage_vertices() {
    let mut graph = welded_blocks(IVec3::new(2, 2, 2), ConstructionMaterial::Steel);
    let id = add_region(&mut graph, IVec3::new(2, 2, 2), ConstructionMaterial::Steel).unwrap();
    let region = graph.region(id).unwrap();
    assert_eq!(region.plane_counts(), [2, 2, 2]);
    assert_eq!(region.vertices().count(), 8, "a fresh cage is a box");
    assert!(region.is_unshaped());
}

#[test]
fn a_region_refuses_an_area_with_a_hole() {
    // Two of the four cells filled: the area is not solid.
    let mut graph = welded_blocks(IVec3::new(2, 1, 1), ConstructionMaterial::Steel);
    assert!(matches!(
        add_region(&mut graph, IVec3::new(2, 2, 1), ConstructionMaterial::Steel),
        Err(GraphError::RegionNotSolid(2))
    ));
    assert_eq!(graph.regions().count(), 0);
}

#[test]
fn a_region_can_span_a_whole_run_of_welded_blocks() {
    let mut graph = welded_blocks(IVec3::new(3, 2, 2), ConstructionMaterial::Steel);
    let id = add_region(&mut graph, IVec3::new(3, 2, 2), ConstructionMaterial::Steel).unwrap();
    let region = graph.region(id).unwrap();
    assert_eq!(region.size_cells(), IVec3::new(3, 2, 2));
    // Twelve blocks, one shape: the cage still has only its eight corners.
    assert_eq!(region.vertices().count(), 8);
}

#[test]
fn a_region_refuses_to_hold_only_part_of_a_block() {
    // One two-cell beam, and an area covering just one of its cells.
    let mut graph = ConstructionGraph::new();
    let spec = CuboidSpec::new(
        [2, 1, 1],
        BuildPose::from_half_grid(IVec3::new(2, 1, 1), GridRotation::default()),
    )
    .unwrap()
    .with_material(ConstructionMaterial::Steel);
    graph.apply(BuildCommand::Spawn(spec)).unwrap();
    assert!(matches!(
        add_region(&mut graph, IVec3::ONE, ConstructionMaterial::Steel),
        Err(GraphError::RegionSplitsPart)
    ));
    // The whole beam is fine.
    add_region(&mut graph, IVec3::new(2, 1, 1), ConstructionMaterial::Steel).unwrap();
}

#[test]
fn a_region_refuses_mixed_materials() {
    let mut graph = welded_blocks(IVec3::new(1, 1, 1), ConstructionMaterial::Steel);
    let spec = CuboidSpec::new(
        [1, 1, 1],
        BuildPose::from_half_grid(IVec3::new(3, 1, 1), GridRotation::default()),
    )
    .unwrap()
    .with_material(ConstructionMaterial::Wood);
    graph.apply(BuildCommand::Spawn(spec)).unwrap();
    assert!(matches!(
        add_region(&mut graph, IVec3::new(2, 1, 1), ConstructionMaterial::Steel),
        Err(GraphError::RegionMixedMaterials)
    ));
}

fn blue_paint() -> MaterialAppearance {
    MaterialAppearance::new(
        MaterialColor::Dye(MaterialDye::new([42, 76, 199], 1.25).unwrap()),
        MaterialFinish::Painted,
    )
}

#[test]
fn a_region_requires_one_appearance_and_painting_it_updates_every_member() {
    let size = IVec3::new(2, 1, 1);
    let mut graph = welded_blocks(size, ConstructionMaterial::Steel);
    let first = graph.parts().next().unwrap().0;
    graph
        .apply(BuildCommand::SetAppearance {
            target: AppearanceTarget::Part(first),
            appearance: blue_paint(),
        })
        .unwrap();
    assert_eq!(
        add_region(&mut graph, size, ConstructionMaterial::Steel),
        Err(GraphError::RegionMixedAppearances)
    );

    graph
        .apply(BuildCommand::SetAppearance {
            target: AppearanceTarget::Part(first),
            appearance: MaterialAppearance::BAKED,
        })
        .unwrap();
    let region = add_region(&mut graph, size, ConstructionMaterial::Steel).unwrap();
    graph
        .apply(BuildCommand::SetAppearance {
            target: AppearanceTarget::Part(first),
            appearance: blue_paint(),
        })
        .unwrap();
    assert_eq!(graph.region(region).unwrap().appearance(), blue_paint());
    assert!(
        graph
            .parts()
            .filter(|(part, _)| graph.region_of(*part) == Some(region))
            .all(|(_, part)| part.appearance() == Some(blue_paint()))
    );
    graph
        .apply(BuildCommand::SetAppearance {
            target: AppearanceTarget::Region(region),
            appearance: MaterialAppearance::BAKED,
        })
        .unwrap();
    assert_eq!(
        graph
            .apply(BuildCommand::SetAppearance {
                target: AppearanceTarget::Region(region),
                appearance: blue_paint(),
            })
            .unwrap(),
        BuildOutcome::AppearanceUpdated
    );
    assert_eq!(graph.region(region).unwrap().appearance(), blue_paint());
    assert!(
        graph
            .parts()
            .filter(|(part, _)| graph.region_of(*part) == Some(region))
            .all(|(_, part)| part.appearance() == Some(blue_paint()))
    );
}

#[test]
fn authored_machine_parts_reject_appearance_edits() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::default(),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    assert_eq!(
        graph.apply(BuildCommand::SetAppearance {
            target: AppearanceTarget::Part(controller),
            appearance: blue_paint(),
        }),
        Err(GraphError::AuthoredAppearance(controller))
    );
}

#[test]
fn a_region_refuses_overlapping_another() {
    let mut graph = welded_blocks(IVec3::new(3, 1, 1), ConstructionMaterial::Steel);
    add_region(&mut graph, IVec3::new(2, 1, 1), ConstructionMaterial::Steel).unwrap();
    let overlapping = ShapeRegion::new(
        IVec3::new(2, 0, 0),
        IVec3::new(2, 1, 1),
        ConstructionMaterial::Steel,
    )
    .unwrap();
    assert!(matches!(
        graph.apply(BuildCommand::AddRegion(overlapping)),
        Err(GraphError::RegionOverlaps(_))
    ));
    assert_eq!(graph.regions().count(), 1);
}

#[test]
fn deleting_a_block_deletes_the_region_it_belonged_to() {
    // A region needs every cell filled, so it cannot outlive its blocks.
    let mut graph = welded_blocks(IVec3::new(2, 1, 1), ConstructionMaterial::Steel);
    add_region(&mut graph, IVec3::new(2, 1, 1), ConstructionMaterial::Steel).unwrap();
    let victim = graph.parts().next().expect("the graph has blocks").0;
    graph.apply(BuildCommand::Remove(victim)).unwrap();
    assert_eq!(graph.regions().count(), 0);
}

#[test]
fn a_cage_vertex_cannot_leave_the_regions_bounding_box() {
    let mut graph = welded_blocks(IVec3::new(1, 1, 1), ConstructionMaterial::Steel);
    let id = add_region(&mut graph, IVec3::ONE, ConstructionMaterial::Steel).unwrap();
    // Corner [0,0,0] sits at the minimum, so it can only move inward.
    assert!(matches!(
        graph.apply(BuildCommand::SetRegionVertices {
            region: id,
            vertices: vec![([0, 0, 0], [-1, 0, 0])],
        }),
        Err(GraphError::InvalidRegion(RegionError::OutsideBounds))
    ));
    assert!(
        graph.region(id).unwrap().is_unshaped(),
        "a rejected move must leave the graph byte-for-byte equivalent"
    );
    // Inward as far as the far face is fine.
    graph
        .apply(BuildCommand::SetRegionVertices {
            region: id,
            vertices: vec![([0, 0, 0], [cell_steps(), 0, 0])],
        })
        .unwrap();
}

#[test]
fn a_cage_move_that_would_invert_a_cell_is_rejected_and_changes_nothing() {
    let mut graph = welded_blocks(IVec3::new(1, 1, 1), ConstructionMaterial::Steel);
    let id = add_region(&mut graph, IVec3::ONE, ConstructionMaterial::Steel).unwrap();
    let cell = cell_steps();
    assert!(matches!(
        graph.apply(BuildCommand::SetRegionVertices {
            region: id,
            vertices: vec![([0, 0, 0], [cell, 0, 0]), ([1, 0, 0], [-cell, 0, 0])],
        }),
        Err(GraphError::InvertedCell(_))
    ));
    assert!(graph.region(id).unwrap().is_unshaped());
}

#[test]
fn collapsing_an_edge_into_a_wedge_is_accepted() {
    let mut graph = welded_blocks(IVec3::new(1, 1, 1), ConstructionMaterial::Steel);
    let id = add_region(&mut graph, IVec3::ONE, ConstructionMaterial::Steel).unwrap();
    let cell = cell_steps();
    graph
        .apply(BuildCommand::SetRegionVertices {
            region: id,
            vertices: vec![([0, 1, 1], [0, -cell, 0]), ([1, 1, 1], [0, -cell, 0])],
        })
        .expect("collapsing an edge makes a wedge");
    assert_eq!(graph.region(id).unwrap().offsets().count(), 2);
}

#[test]
fn subdividing_inserts_a_whole_plane_without_moving_the_surface() {
    let mut graph = welded_blocks(IVec3::new(2, 1, 1), ConstructionMaterial::Steel);
    let id = add_region(&mut graph, IVec3::new(2, 1, 1), ConstructionMaterial::Steel).unwrap();
    let cell = cell_steps();
    graph
        .apply(BuildCommand::SetRegionVertices {
            region: id,
            vertices: vec![([1, 1, 0], [0, -cell, 0]), ([1, 1, 1], [0, -cell, 0])],
        })
        .unwrap();
    let before = surface_corners(graph.region(id).unwrap());

    graph
        .apply(BuildCommand::SubdivideRegion {
            region: id,
            axis: 0,
            position: 1,
        })
        .unwrap();
    let region = graph.region(id).unwrap();
    assert_eq!(
        region.plane_counts(),
        [3, 2, 2],
        "a whole plane is inserted"
    );
    assert_eq!(
        surface_corners(region),
        before,
        "the cage gains handles; the surface must not move"
    );
}

/// The eight outer cage corners, for comparing a surface before and after.
fn surface_corners(region: &ShapeRegion) -> Vec<[i32; 3]> {
    let [x, y, z] = region.plane_counts();
    let last = [x - 1, y - 1, z - 1];
    let mut corners = Vec::new();
    for corner in 0..8_usize {
        let index = [
            u16::try_from(if corner & 1 == 0 { 0 } else { last[0] }).unwrap(),
            u16::try_from(if corner & 2 == 0 { 0 } else { last[1] }).unwrap(),
            u16::try_from(if corner & 4 == 0 { 0 } else { last[2] }).unwrap(),
        ];
        corners.push(region.vertex_steps(index).unwrap().to_array());
    }
    corners
}

fn engine(graph: &mut ConstructionGraph, kind: EngineKind, units: IVec3) -> PartId {
    let BuildOutcome::Spawned(engine) = graph
        .apply(BuildCommand::SpawnEngine(EngineSpec::new(
            kind,
            BuildPose::new(units, GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    engine
}

fn attach(graph: &mut ConstructionGraph, parent: PartId) -> PartId {
    let spec = graph.next_transmission_spec(parent).unwrap();
    let BuildOutcome::Spawned(transmission) = graph
        .apply(BuildCommand::AttachTransmission { parent, spec })
        .unwrap()
    else {
        unreachable!()
    };
    transmission
}

#[test]
fn transmissions_form_one_inherited_seventeen_block_chain() {
    let mut graph = ConstructionGraph::new();
    let engine = engine(&mut graph, EngineKind::Gas, IVec3::ZERO);
    let mut tail = engine;
    for depth in 1..=17 {
        tail = attach(&mut graph, tail);
        assert_eq!(
            graph.transmission_root(tail),
            Some((engine, EngineKind::Gas, depth))
        );
        assert_eq!(graph.transmission_kind(tail), Some(EngineKind::Gas));
        assert_eq!(
            graph.part(tail).unwrap().pose().rotation,
            GridRotation::default()
        );
    }
    assert_eq!(
        graph.next_transmission_spec(tail),
        Err(GraphError::TransmissionLimitReached)
    );
    assert_eq!(graph.engine_transmission_depth(engine), Some(17));
}

#[test]
fn transmission_attachment_is_atomic_and_protects_its_weld() {
    let mut graph = ConstructionGraph::new();
    let engine = engine(&mut graph, EngineKind::Electric, IVec3::ZERO);
    let expected = graph.next_transmission_spec(engine).unwrap();
    let wrong = TransmissionSpec::new(BuildPose::new(
        IVec3::new(0, 0, 99),
        GridRotation::new(0, 1, 0),
    ));
    assert_eq!(
        graph.apply(BuildCommand::AttachTransmission {
            parent: engine,
            spec: wrong,
        }),
        Err(GraphError::InvalidTransmissionPose)
    );
    assert_eq!(graph.part_count(), 1, "failed attachment changes nothing");

    let BuildOutcome::Spawned(first) = graph
        .apply(BuildCommand::AttachTransmission {
            parent: engine,
            spec: expected,
        })
        .unwrap()
    else {
        unreachable!()
    };
    assert_eq!(
        graph.next_transmission_spec(engine),
        Err(GraphError::TransmissionOutputOccupied(engine))
    );
    let weld = graph.transmission_weld(first).unwrap();
    assert_eq!(
        graph.apply(BuildCommand::RemoveWeld(weld)),
        Err(GraphError::RequiredTransmissionWeld(weld))
    );
}

#[test]
fn deleting_upstream_transmission_parts_cascades_downstream() {
    let mut graph = ConstructionGraph::new();
    let engine = engine(&mut graph, EngineKind::Gas, IVec3::ZERO);
    let first = attach(&mut graph, engine);
    let second = attach(&mut graph, first);
    let third = attach(&mut graph, second);

    graph.apply(BuildCommand::Remove(second)).unwrap();
    assert!(graph.part(engine).is_some());
    assert!(graph.part(first).is_some());
    assert!(graph.part(second).is_none());
    assert!(graph.part(third).is_none());
    assert_eq!(graph.engine_transmission_depth(engine), Some(1));

    graph.apply(BuildCommand::Remove(engine)).unwrap();
    assert_eq!(graph.part_count(), 0);
    assert_eq!(graph.weld_count(), 0);
}

#[test]
fn same_type_stack_mismatch_is_visible_and_blocks_compile_only() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::new(IVec3::ZERO, GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let first = engine(&mut graph, EngineKind::Electric, IVec3::new(2, 0, 0));
    let second = engine(&mut graph, EngineKind::Electric, IVec3::new(4, 0, 0));
    for (left, right) in [(controller, first), (first, second)] {
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(left, FaceKind::PositiveX),
                second: FaceRef::part(right, FaceKind::NegativeX),
            }))
            .unwrap();
    }
    attach(&mut graph, first);

    let inventory = graph.actuator_inventory(controller).unwrap();
    assert_eq!(inventory.electric_engines, 2);
    assert!(inventory.electric_transmission_mismatch);
    assert_eq!(inventory.electric_transmission_depth, None);
    assert!(matches!(
        graph.compile(),
        Err(crate::TopologyError::TransmissionDepthMismatch {
            kind: EngineKind::Electric,
            ..
        })
    ));
}

#[test]
fn gearbox_edits_validate_physical_gear_count_and_are_transactional() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::new(IVec3::ZERO, GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let engine = engine(&mut graph, EngineKind::Gas, IVec3::new(2, 0, 0));
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(controller, FaceKind::PositiveX),
            second: FaceRef::part(engine, FaceKind::NegativeX),
        }))
        .unwrap();
    attach(&mut graph, engine);

    graph
        .apply(BuildCommand::SetGearboxMode {
            controller,
            kind: EngineKind::Gas,
            mode: ShiftMode::Manual,
        })
        .unwrap();
    assert_eq!(
        graph
            .gearbox_config(controller, EngineKind::Gas)
            .unwrap()
            .mode(),
        ShiftMode::Manual
    );
    assert!(matches!(
        graph.apply(BuildCommand::SetGearboxRatios {
            controller,
            kind: EngineKind::Gas,
            ratios: vec![1.0],
        }),
        Err(GraphError::GearCountMismatch { .. })
    ));
    assert_eq!(
        graph
            .gearbox_config(controller, EngineKind::Gas)
            .unwrap()
            .ratios(),
        &[3.0, 1.0]
    );
}

#[test]
fn shape_features_add_adjust_remove_and_reject_invalid_amounts_atomically() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [2, 2, 2],
                BuildPose::new(IVec3::ZERO, GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let owner = crate::SolidOwner::Part(part);
    let edge = graph.evaluated_solid(owner).unwrap().logical_edges[0].key;
    let feature = crate::ShapeFeature::new(
        [crate::EdgeChainRef { owner, edge }],
        crate::EdgeTreatment::Chamfer,
        10,
    );
    let BuildOutcome::ShapeFeatureAdded(id) =
        graph.apply(BuildCommand::AddShapeFeature(feature)).unwrap()
    else {
        unreachable!()
    };
    let chamfered_volume = graph.evaluated_solid(owner).unwrap().volume();
    assert!(chamfered_volume < 0.5_f64.powi(3));

    graph
        .apply(BuildCommand::SetShapeFeatureAmount {
            feature: id,
            amount_ticks: 20,
        })
        .unwrap();
    assert!(graph.evaluated_solid(owner).unwrap().volume() < chamfered_volume);

    let snapshot = graph.evaluated_solid(owner).unwrap();
    assert!(matches!(
        graph.apply(BuildCommand::SetShapeFeatureAmount {
            feature: id,
            amount_ticks: 0,
        }),
        Err(GraphError::InvalidSolid(crate::SolidError::ZeroAmount))
    ));
    assert_eq!(graph.evaluated_solid(owner).unwrap(), snapshot);

    graph.apply(BuildCommand::RemoveShapeFeature(id)).unwrap();
    assert!(!graph.owner_has_shape_features(owner));
    assert!((graph.evaluated_solid(owner).unwrap().volume() - 0.5_f64.powi(3)).abs() < 1.0e-8);
}

#[test]
fn five_centimetre_fillet_revalidates_a_welded_tangent_chain() {
    let mut graph = ConstructionGraph::new();
    let first = spawn(
        &mut graph,
        CuboidSpec::new(
            [1, 1, 1],
            BuildPose::new(IVec3::ZERO, GridRotation::default()),
        )
        .unwrap(),
    );
    let second = spawn(
        &mut graph,
        CuboidSpec::new([1, 1, 1], BuildPose::new(IVec3::X, GridRotation::default())).unwrap(),
    );
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(first, FaceKind::PositiveX),
            second: FaceRef::part(second, FaceKind::NegativeX),
        }))
        .unwrap();

    let target_for = |part| {
        let owner = crate::SolidOwner::Part(part);
        let solid = graph.evaluated_solid(owner).unwrap();
        let edge = solid
            .logical_edges
            .iter()
            .find(|logical| {
                let half_edge = solid.half_edges[logical.half_edges[0] as usize];
                let twin = solid.half_edges[half_edge.twin as usize];
                let patches = [
                    solid.surfaces[half_edge.face as usize].key.local,
                    solid.surfaces[twin.face as usize].key.local,
                ];
                patches.contains(&3) && patches.contains(&4)
            })
            .expect("the positive-Y/negative-Z edge exists")
            .key;
        crate::EdgeChainRef { owner, edge }
    };
    let targets = [target_for(first), target_for(second)];
    let BuildOutcome::ShapeFeatureAdded(feature) = graph
        .apply(BuildCommand::AddShapeFeature(crate::ShapeFeature::new(
            targets,
            crate::EdgeTreatment::Fillet,
            20,
        )))
        .unwrap()
    else {
        unreachable!()
    };

    assert_eq!(graph.shape_feature(feature).unwrap().amount_ticks, 20);
    graph
        .apply(BuildCommand::SetShapeFeatureAmount {
            feature,
            amount_ticks: 100,
        })
        .unwrap();
    assert!(matches!(
        graph.apply(BuildCommand::SetShapeFeatureAmount {
            feature,
            amount_ticks: 120,
        }),
        Err(GraphError::InvalidSolid(crate::SolidError::AmountTooLarge(id))) if id == feature
    ));
    assert_eq!(graph.shape_feature(feature).unwrap().amount_ticks, 100);
}

#[test]
fn welded_block_solid_accepts_treatments_past_one_block() {
    let mut graph = ConstructionGraph::new();
    let mut parts = std::collections::BTreeMap::new();
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let part = spawn(
                    &mut graph,
                    CuboidSpec::new(
                        [1, 1, 1],
                        BuildPose::new(IVec3::new(x, y, z), GridRotation::default()),
                    )
                    .unwrap(),
                );
                parts.insert([x, y, z], part);
            }
        }
    }
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let cell = IVec3::new(x, y, z);
                for (axis, positive, negative) in [
                    (0, FaceKind::PositiveX, FaceKind::NegativeX),
                    (1, FaceKind::PositiveY, FaceKind::NegativeY),
                    (2, FaceKind::PositiveZ, FaceKind::NegativeZ),
                ] {
                    let mut neighbour = cell;
                    neighbour[axis] += 1;
                    if let Some(&next) = parts.get(&neighbour.to_array()) {
                        graph
                            .apply(BuildCommand::Weld(WeldSpec {
                                first: FaceRef::part(parts[&cell.to_array()], positive),
                                second: FaceRef::part(next, negative),
                            }))
                            .unwrap();
                    }
                }
            }
        }
    }

    let targets = (0..2)
        .map(|z| {
            let part = parts[&[1, 1, z]];
            let owner = crate::SolidOwner::Part(part);
            let solid = graph.evaluated_solid(owner).unwrap();
            let edge = solid
                .logical_edges
                .iter()
                .find(|logical| {
                    let half_edge = solid.half_edges[logical.half_edges[0] as usize];
                    let twin = solid.half_edges[half_edge.twin as usize];
                    let patches = [
                        solid.surfaces[half_edge.face as usize].key.local,
                        solid.surfaces[twin.face as usize].key.local,
                    ];
                    patches.contains(&1) && patches.contains(&3)
                })
                .expect("the positive-X/positive-Y edge exists")
                .key;
            crate::EdgeChainRef { owner, edge }
        })
        .collect::<Vec<_>>();

    for treatment in [crate::EdgeTreatment::Fillet, crate::EdgeTreatment::Chamfer] {
        let mut candidate = graph.clone();
        let BuildOutcome::ShapeFeatureAdded(feature) = candidate
            .apply(BuildCommand::AddShapeFeature(crate::ShapeFeature::new(
                targets.clone(),
                treatment,
                180,
            )))
            .unwrap_or_else(|error| panic!("45 cm {treatment:?} failed: {error}"))
        else {
            unreachable!()
        };
        let owner = candidate.shape_feature(feature).unwrap().targets[0].owner;
        assert!(matches!(owner, crate::SolidOwner::Region(_)));
        assert!(candidate.evaluated_solid(owner).unwrap().volume() < 0.125);
        candidate.compile().unwrap();
    }
}

fn grounded_filleted_cylinder() -> (ConstructionGraph, PartId, CylinderSpec) {
    let mut graph = ConstructionGraph::new();
    let spec = CylinderSpec::new(
        CylinderDimensions::new(1.0, 0.0, 0.5).unwrap(),
        BuildPose::from_position_ticks(IVec3::new(0, 100, 0), GridRotation::default()),
    );
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::SpawnCylinder(spec)).unwrap()
    else {
        panic!("spawning a cylinder reports its part");
    };
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(part, FaceKind::NegativeY),
            second: FaceRef::ground(),
        }))
        .unwrap();
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
    (graph, part, spec)
}

fn rubber_layer(spec: impl Into<PartSpec>, face: crate::LayerFace, thickness: f32) -> PartSpec {
    spec.into()
        .with_layer(
            face,
            thickness,
            ConstructionMaterial::Rubber,
            MaterialAppearance::BAKED,
        )
        .unwrap()
}

#[test]
fn layering_a_cylinder_keeps_its_welds_and_features() {
    let (mut graph, part, spec) = grounded_filleted_cylinder();
    let layered = rubber_layer(spec, crate::LayerFace::OuterWall, 0.25);
    assert_eq!(
        graph.apply(BuildCommand::SetLayers {
            part,
            spec: layered
        }),
        Ok(BuildOutcome::LayersUpdated)
    );
    assert_eq!(graph.part(part), Some(&layered));
    assert_eq!(graph.weld_count(), 1);
    let solid = graph
        .evaluated_solid(crate::SolidOwner::Part(part))
        .unwrap();
    assert!(solid.surfaces.iter().any(|surface| surface.band == 0));
    assert!(solid.surfaces.iter().any(|surface| surface.band == 1));
}

#[test]
fn a_layer_edit_that_reshapes_the_core_is_rejected() {
    let (mut graph, part, spec) = grounded_filleted_cylinder();
    let narrow = CylinderSpec::new(CylinderDimensions::new(0.5, 0.0, 0.5).unwrap(), spec.pose);
    assert_eq!(
        graph.apply(BuildCommand::SetLayers {
            part,
            spec: narrow.into()
        }),
        Err(GraphError::LayerCoreChanged(part))
    );
    assert_eq!(graph.part(part), Some(&PartSpec::Cylinder(spec)));
}

#[test]
fn a_layer_that_breaks_feature_replay_is_rejected() {
    let (mut graph, part, spec) = grounded_filleted_cylinder();
    // A bottom-cap layer buries the welded face under rubber.
    let capped = rubber_layer(spec, crate::LayerFace::Face(FaceKind::NegativeY), 0.25);
    assert!(
        graph
            .apply(BuildCommand::SetLayers { part, spec: capped })
            .is_err()
    );
    assert_eq!(graph.part(part), Some(&PartSpec::Cylinder(spec)));
}

#[test]
fn layering_a_cuboid_keeps_welds_on_other_faces() {
    let mut graph = ConstructionGraph::new();
    let spec = cube_at(0);
    let part = spawn(&mut graph, spec);
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(part, FaceKind::NegativeY),
            second: FaceRef::ground(),
        }))
        .unwrap();
    let layered = rubber_layer(spec, crate::LayerFace::Face(FaceKind::PositiveX), 0.05);
    assert_eq!(
        graph.apply(BuildCommand::SetLayers {
            part,
            spec: layered
        }),
        Ok(BuildOutcome::LayersUpdated)
    );
    assert_eq!(graph.weld_count(), 1);
    let solid = graph
        .evaluated_solid(crate::SolidOwner::Part(part))
        .unwrap();
    assert_eq!(
        solid.logical_edges.len(),
        12,
        "the layer seam is not a selectable edge"
    );
    assert!(solid.cells.iter().any(|cell| cell.band == 1));
    assert!(solid.surfaces.iter().any(|surface| surface.band == 1));
}

#[test]
fn layered_blocks_cannot_join_a_region() {
    let mut graph = ConstructionGraph::new();
    let spec = cube_at(0);
    let part = spawn(&mut graph, spec);
    let layered = rubber_layer(spec, crate::LayerFace::Face(FaceKind::PositiveY), 0.05);
    graph
        .apply(BuildCommand::SetLayers {
            part,
            spec: layered,
        })
        .unwrap();
    let cells = crate::part_cells(spec);
    let region = crate::ShapeRegion::new(
        cells.corner_half_units(IVec3::ZERO, 0),
        cells.counts(),
        ConstructionMaterial::Steel,
    )
    .unwrap();
    assert_eq!(
        graph.apply(BuildCommand::AddRegion(region)),
        Err(GraphError::LayeredPartInRegion(part))
    );
}

#[test]
fn painting_one_material_band_leaves_the_others() {
    let (mut graph, part, spec) = grounded_filleted_cylinder();
    let layered = rubber_layer(spec, crate::LayerFace::OuterWall, 0.25);
    graph
        .apply(BuildCommand::SetLayers {
            part,
            spec: layered,
        })
        .unwrap();
    let paint = MaterialAppearance {
        finish: MaterialFinish::Painted,
        ..MaterialAppearance::BAKED
    };
    graph
        .apply(BuildCommand::SetAppearance {
            target: AppearanceTarget::PartBand { part, band: 1 },
            appearance: paint,
        })
        .unwrap();
    let painted = graph.part(part).copied().unwrap();
    assert_eq!(painted.band(1).unwrap().1, paint);
    assert_eq!(painted.band(0).unwrap().1, MaterialAppearance::BAKED);
    assert_eq!(
        graph.apply(BuildCommand::SetAppearance {
            target: AppearanceTarget::PartBand { part, band: 2 },
            appearance: paint,
        }),
        Err(GraphError::MissingBand(part, 2))
    );
}
