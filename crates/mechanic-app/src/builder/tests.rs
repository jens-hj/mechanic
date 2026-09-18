#[test]
#[ignore = "CPU-only pipe placement benchmark; no timing assertions"]
fn measure_pipe_overlap_latency() {
    use std::{hint::black_box, time::Instant};
    let bend = PartSpec::PipeBend(PipeBendSpec::new(
        PipeBendDimensions::default(),
        BuildPose::default(),
    ));
    let pipe = PartSpec::Cylinder(CylinderSpec::new(
        CylinderDimensions::default(),
        BuildPose::default(),
    ));
    let junction = PartSpec::PipeJunction(mechanic_core::PipeJunctionSpec::new(
        mechanic_core::PipeJunctionDimensions::new(0.2, 0.1).unwrap(),
        mechanic_core::PipeArms::from_bits(0b01_0011).unwrap(),
        BuildPose::default(),
    ));
    for (name, spec) in [("pipe", pipe), ("bend", bend), ("junction", junction)] {
        let distant = spec.with_pose(BuildPose::new(
            IVec3::new(100, 0, 0),
            GridRotation::default(),
        ));
        let started = Instant::now();
        for _ in 0..100 {
            black_box(super::parts_overlap(black_box(spec), black_box(distant)));
        }
        eprintln!(
            "{name}: 100 distant overlap queries {:?}",
            started.elapsed()
        );
    }
}
use super::PipeNode;
use super::bearings::bearing_ring_overlaps_face;
use super::bearings::locked_bearings;
use super::candidates::candidate_from_hit;
use super::raycast::raycast_construction_with_ground;
use super::raycast::raycast_sources;
use super::snap::AxisGuide;
use super::snap::GuideKind;
use super::snap::render_free_smart_guides;

#[test]
fn pipe_overlap_pruning_matches_exhaustive_boxes_in_rotated_frames() {
    let kinds = [
        PartSpec::Cuboid(CuboidSpec::new([1; 3], BuildPose::default()).unwrap()),
        PartSpec::Cylinder(CylinderSpec::new(
            CylinderDimensions::new(0.2, 0.1, 0.5).unwrap(),
            BuildPose::default(),
        )),
        PartSpec::PipeBend(PipeBendSpec::new(
            PipeBendDimensions::default(),
            BuildPose::default(),
        )),
        PartSpec::PipeJunction(mechanic_core::PipeJunctionSpec::new(
            mechanic_core::PipeJunctionDimensions::new(0.2, 0.1).unwrap(),
            mechanic_core::PipeArms::from_bits(0b01_0011).unwrap(),
            BuildPose::default(),
        )),
    ];
    for first in kinds {
        for second in kinds {
            for rotation in [
                Quat::IDENTITY,
                Quat::from_rotation_y(0.47) * Quat::from_rotation_x(0.29),
            ] {
                for offset in [-2.0, -0.25, 0.0, 0.1, 0.25, 2.0] {
                    let frame = mechanic_core::ConstructionFrame::new(
                        Vec3::new(offset, offset * 0.5, offset),
                        rotation,
                    )
                    .unwrap();
                    let targets = super::bounds::part_collision_boxes(second)
                        .into_iter()
                        .map(|shape| super::bounds::CollisionBox {
                            center: frame.point(shape.center),
                            rotation: frame.rotation() * shape.rotation,
                            ..shape
                        })
                        .collect::<Vec<_>>();
                    let expected = super::bounds::part_collision_boxes(first)
                        .into_iter()
                        .any(|a| targets.iter().any(|&b| super::bounds::boxes_overlap(a, b)));
                    assert_eq!(
                        super::bounds::parts_overlap_with_frame(first, second, frame),
                        expected
                    );
                }
            }
        }
    }
}
use std::time::Instant;

use bevy::{
    math::DVec2,
    prelude::{IVec3, Quat, Vec3},
};
use mechanic_core::{
    BearingDimensions, BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
    ConstructionMaterial, CuboidSpec, CylinderDimensions, CylinderSpec, DimensionLinkId,
    EdgeChainRef, EdgeTreatment, EngineKind, EngineSpec, FaceKind, FaceOwner, FaceRef,
    GridRotation, POSITION_TICKS_PER_GRID_UNIT, PartId, PartSpec, PendingOperation,
    PipeBendDimensions, PipeBendSpec, RigidLinkSpec, ShapeFeature, SolidOwner, WeldSpec,
};

use super::{
    BLOCK_SIZE_METERS, BlockVolume, PipeRunAttachment, PipeRunPiece, PlacementBounds,
    PlacementCandidate, PlacementError, PlacementGrid, PlacementPlane, PlacementSnapIndex,
    PlacementSupport, SurfaceHit, bearing_anchor_from_hit, bearing_attachment_candidate,
    bearing_overlaps_candidate, bearing_support_face, begin_weld, block_box_bounds,
    block_box_specs, block_sheet_specs, block_span_from_rays, center_cylinder_candidate_on_bearing,
    cuboid_candidate_from_hit, cylinder_candidate_from_hit, face_geometry_from_ref, face_is_flat,
    free_cuboid_candidate, free_cylinder_candidate, newly_locked_bearings,
    oriented_cuboid_candidate_from_hit, oriented_cuboid_candidate_from_hit_with_grid,
    pipe_run_pieces, raycast_construction, raycast_construction_for_annulus,
    raycast_placement_plane_point, rigid_body_parts, smart_snap_block_span,
    smart_snap_cuboid_candidate, smart_snap_free_cuboid_candidate, stage_bearing_attachment,
    stage_bearing_block_batch, stage_block_batch, stage_block_batch_from_source,
    stage_block_batch_from_source_in_bounds, stage_block_batch_in_bounds,
    stage_block_volume_in_bounds, stage_controller_in_bounds, stage_cuboid,
    stage_cylinder_from_source, stage_dimension_link_in_bounds, stage_engine_from_source,
    stage_engine_in_bounds, stage_input_in_bounds, stage_pipe_run, stage_pipe_run_in_bounds,
    stage_seat_in_bounds, stage_servo_in_bounds, stage_transmission, stage_weld_objects,
    transmission_candidate_from_hit, validate_block_batch_in_bounds,
    validate_indexed_block_batch_in_bounds, validate_part,
};

#[test]
fn linear_candidates_are_flush_and_lattice_snapped_on_every_face_orientation() {
    use mechanic_core::{CarriageFace, LinearBearing, LinearBearingDimensions};
    for length in [0.25, 1.0, 8.0] {
        for normal in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::NEG_X,
            Vec3::NEG_Y,
            Vec3::NEG_Z,
        ] {
            for axis in [
                Vec3::X,
                Vec3::Y,
                Vec3::Z,
                Vec3::NEG_X,
                Vec3::NEG_Y,
                Vec3::NEG_Z,
            ] {
                if normal.dot(axis) != 0.0 {
                    continue;
                }
                for face in [
                    CarriageFace::Top,
                    CarriageFace::PositiveSide,
                    CarriageFace::NegativeSide,
                ] {
                    let rail = LinearBearing {
                        dimensions: LinearBearingDimensions::new(length, 0.1).unwrap(),
                        mount_normal: normal,
                        face,
                    };
                    let surface =
                        super::bearings::linear_carriage_face(Vec3::ZERO, rail, axis).unwrap();
                    let hit =
                        surface.center + surface.tangent_u * 0.029 + surface.tangent_v * 0.009;
                    let candidate = super::linear_block_candidate(
                        Vec3::ZERO,
                        rail,
                        axis,
                        hit,
                        [1; 3],
                        GridRotation::default(),
                    )
                    .unwrap();
                    let block_face =
                        super::faces::face_geometry(candidate.spec, candidate.attached_face);
                    assert!(
                        (block_face.center - surface.center)
                            .dot(surface.normal)
                            .abs()
                            < 1.0e-6
                    );
                    assert!((block_face.normal + surface.normal).length() < 1.0e-6);
                    assert!(candidate.anchor.is_some());
                    assert!(
                        ((block_face.center - surface.center).dot(axis) - 0.025).abs() < 1.0e-6
                    );
                    let cylinder = super::linear_cylinder_candidate(
                        Vec3::ZERO,
                        rail,
                        axis,
                        hit,
                        CylinderDimensions::new(0.25, 0.0, 0.25).unwrap(),
                        1,
                    )
                    .unwrap();
                    let cylinder_face =
                        super::faces::cylinder_face_geometry(cylinder.spec, cylinder.attached_face)
                            .unwrap();
                    assert!(
                        (cylinder_face.center - surface.center)
                            .dot(surface.normal)
                            .abs()
                            < 1.0e-6
                    );
                    assert!(cylinder.anchor.is_some());
                }
            }
        }
    }
}

#[test]
fn linear_deleted_support_migrates_to_overlapping_coplanar_survivor() {
    use mechanic_core::{CarriageFace, LinearBearing, LinearBearingDimensions};
    let mut graph = ConstructionGraph::new();
    let original = spawn_cube(&mut graph, IVec3::ZERO, 1);
    let survivor = spawn_cube(&mut graph, IVec3::X, 1);
    spawn_cube(&mut graph, IVec3::new(-1, 1, 0), 1);
    spawn_cube(&mut graph, IVec3::new(8, 0, 0), 1);
    let selected = FaceRef::part(original, FaceKind::PositiveY);
    let anchor = Vec3::new(0.0, 0.125, 0.0);
    let rail = LinearBearing {
        dimensions: LinearBearingDimensions::default(),
        mount_normal: Vec3::Y,
        face: CarriageFace::Top,
    };
    let mut deleted = std::collections::HashSet::from([original]);
    let replacement =
        super::linear_support_face_excluding(&graph, selected, anchor, rail, Vec3::X, &deleted);
    assert_eq!(
        replacement,
        Some(FaceRef::part(survivor, FaceKind::PositiveY))
    );
    // The anchor lies outside the survivor, but the long rail still overlaps it.
    assert!(
        anchor.x
            < super::face_geometry_from_ref(replacement.unwrap(), Some(&graph))
                .center
                .x
                - 0.125
    );
    deleted.insert(survivor);
    assert_eq!(
        super::linear_support_face_excluding(&graph, selected, anchor, rail, Vec3::X, &deleted),
        None,
        "raised or distant faces must not rescue an unsupported rail"
    );
}

#[test]
fn linear_rail_overhang_and_flush_side_attachment_use_real_support_overlap() {
    use mechanic_core::{CarriageFace, LinearBearing, LinearBearingDimensions};
    for (face, side) in [
        (CarriageFace::PositiveSide, 1.0),
        (CarriageFace::NegativeSide, -1.0),
    ] {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::ZERO, 1);
        let source = FaceRef::part(base, FaceKind::PositiveY);
        let rail = LinearBearing {
            dimensions: LinearBearingDimensions::default(),
            mount_normal: Vec3::Y,
            face,
        };
        let anchor = Vec3::new(0.0, 0.125, side * 0.1);
        assert!(super::linear_mount_overlaps_face(
            &graph,
            source,
            anchor,
            rail,
            Vec3::X
        ));
        assert!(!super::linear_mount_overlaps_face(
            &graph,
            source,
            anchor + Vec3::Z,
            rail,
            Vec3::X
        ));
        let surface = super::bearings::linear_carriage_face(anchor, rail, Vec3::X).unwrap();
        let candidate = super::linear_block_candidate(
            anchor,
            rail,
            Vec3::X,
            surface.center,
            [1; 3],
            GridRotation::default(),
        )
        .unwrap();
        let staged = super::stage_linear_block_batch_in_bounds(
            &graph,
            candidate,
            &[candidate.spec],
            super::LinearAttachment {
                source,
                anchor,
                rail,
                axis: Vec3::X,
                rigid_targets: &[],
            },
            PlacementBounds::Garage,
        )
        .unwrap();
        assert_eq!(staged.compile().unwrap().bearings.len(), 1);
    }
}

#[test]
fn linear_block_and_cylinder_direct_attachments_share_one_moving_compound() {
    use mechanic_core::{CarriageFace, LinearBearing, LinearBearingDimensions};
    let mut graph = ConstructionGraph::new();
    let base = spawn_cube(&mut graph, IVec3::ZERO, 1);
    let source = FaceRef::part(base, FaceKind::PositiveY);
    let rail = LinearBearing {
        dimensions: LinearBearingDimensions::default(),
        mount_normal: Vec3::Y,
        face: CarriageFace::Top,
    };
    let anchor = Vec3::new(0.0, 0.125, 0.0);
    let surface = super::bearings::linear_carriage_face(anchor, rail, Vec3::X).unwrap();
    let block = super::linear_block_candidate(
        anchor,
        rail,
        Vec3::X,
        surface.center - Vec3::Z * 0.125,
        [1; 3],
        GridRotation::default(),
    )
    .unwrap();
    let attachment = super::LinearAttachment {
        source,
        anchor,
        rail,
        axis: Vec3::X,
        rigid_targets: &[],
    };
    let mut index = PlacementSnapIndex::default();
    index.rebuild(&graph);
    let placed = super::stage_linear_block_volume_in_bounds(
        &graph,
        &index,
        block,
        BlockVolume::new(block.spec, IVec3::ZERO).unwrap(),
        attachment,
        PlacementBounds::Garage,
        7,
    )
    .unwrap();
    assert_eq!(placed.publication_generation, 7);
    let graph = placed.graph;
    let targets = placed.new_parts;
    let cylinder = super::linear_cylinder_candidate(
        anchor,
        rail,
        Vec3::X,
        surface.center + Vec3::Z * 0.125,
        CylinderDimensions::new(0.25, 0.0, 0.25).unwrap(),
        0,
    )
    .unwrap();
    let graph = super::stage_linear_cylinder_in_bounds(
        &graph,
        cylinder,
        super::LinearAttachment {
            rigid_targets: &targets,
            ..attachment
        },
        PlacementBounds::Garage,
    )
    .unwrap();
    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.compounds.len(), 2);
    assert_eq!(compiled.bearings.len(), 1);
    assert!(matches!(
        compiled.bearings[0].kind,
        mechanic_core::BearingKind::Linear(_)
    ));
    let side_rail = LinearBearing {
        face: CarriageFace::PositiveSide,
        ..rail
    };
    let side_surface = super::bearings::linear_carriage_face(anchor, side_rail, Vec3::X).unwrap();
    let side = super::linear_block_candidate(
        anchor,
        side_rail,
        Vec3::X,
        side_surface.center,
        [1; 3],
        GridRotation::default(),
    )
    .unwrap();
    assert!(
        super::stage_linear_block_batch_in_bounds(
            &graph,
            side,
            &[side.spec],
            super::LinearAttachment {
                rail: side_rail,
                rigid_targets: &targets,
                ..attachment
            },
            PlacementBounds::Garage
        )
        .is_err()
    );
}

fn spawn_cube(graph: &mut ConstructionGraph, units: IVec3, size: u8) -> mechanic_core::PartId {
    let spec = CuboidSpec::new([size; 3], BuildPose::new(units, GridRotation::default())).unwrap();
    let Ok(BuildOutcome::Spawned(part)) = graph.apply(BuildCommand::Spawn(spec)) else {
        panic!("cube must spawn");
    };
    part
}

fn ground_volume_candidate(start_ticks: IVec3) -> PlacementCandidate {
    PlacementCandidate {
        spec: CuboidSpec::new(
            [1; 3],
            BuildPose::from_position_ticks(start_ticks, GridRotation::default()),
        )
        .unwrap(),
        attached_face: FaceKind::NegativeY,
        anchor: Some(start_ticks.as_vec3() * mechanic_core::POSITION_TICK_METERS),
        support: PlacementSupport::Surface(FaceOwner::Ground),
    }
}

fn spawn_cylinder(
    graph: &mut ConstructionGraph,
    dimensions: CylinderDimensions,
    pose: BuildPose,
) -> mechanic_core::PartId {
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
            dimensions, pose,
        )))
        .unwrap()
    else {
        unreachable!()
    };
    part
}

#[test]
fn fixed_global_grid_quantises_all_three_modifier_modes() {
    let graph = ConstructionGraph::new();
    let candidate = |point: Vec3, grid| {
        oriented_cuboid_candidate_from_hit_with_grid(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point,
                face: FaceRef::ground(),
            },
            [1, 1, 1],
            GridRotation::default(),
            grid,
            PlacementBounds::Garage,
        )
    };

    assert_eq!(
        candidate(Vec3::new(0.11, 0.0, -0.11), PlacementGrid::Centimetres25)
            .spec
            .pose
            .translation_position_ticks(),
        IVec3::new(0, 50, 0)
    );
    assert_eq!(
        candidate(Vec3::new(0.038, 0.0, -0.038), PlacementGrid::Centimetres5)
            .spec
            .pose
            .translation_position_ticks(),
        IVec3::new(20, 50, -20)
    );
    assert_eq!(
        candidate(Vec3::new(0.006, 0.0, -0.006), PlacementGrid::Centimetres1)
            .spec
            .pose
            .translation_position_ticks(),
        IVec3::new(4, 50, -4)
    );
}

#[test]
fn global_grid_uses_absolute_world_origin_and_object_parity() {
    let graph = ConstructionGraph::new();
    let candidate = oriented_cuboid_candidate_from_hit_with_grid(
        &graph,
        SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.013, 0.0, 0.013),
            face: FaceRef::ground(),
        },
        [2, 1, 2],
        GridRotation::default(),
        PlacementGrid::Centimetres1,
        PlacementBounds::World {
            origin: DVec2::new(10.0, -20.0),
        },
    );
    let global = candidate.spec.pose.translation() + Vec3::new(10.0, 0.0, -20.0);
    assert!(((global.x - 0.005) / 0.01).fract().abs() < 1.0e-3);
    assert!(((global.z - 0.005) / 0.01).fract().abs() < 1.0e-3);
    assert!((global.y - 0.125).abs() < 1.0e-6);
}

#[test]
fn smart_snap_combines_center_and_edge_guides_and_falls_back_when_invalid() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_position_ticks(IVec3::new(6, 50, 400), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap();
    let hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::new(0.014, 0.0, 0.74),
        face: FaceRef::ground(),
    };
    let gridded = oriented_cuboid_candidate_from_hit_with_grid(
        &graph,
        hit,
        [1, 1, 1],
        GridRotation::default(),
        PlacementGrid::Centimetres25,
        PlacementBounds::Garage,
    );
    let mut index = PlacementSnapIndex::default();
    index.rebuild(&graph);
    let (snapped_candidate, active_guides) = smart_snap_cuboid_candidate(
        &graph,
        &index,
        hit,
        gridded,
        PlacementGrid::Centimetres25,
        1.0,
        |_| true,
    );
    assert_eq!(active_guides.len(), 2);
    assert_eq!(
        snapped_candidate.spec.pose.translation_position_ticks(),
        IVec3::new(6, 50, 300)
    );

    let (fallback, rejected_guides) = smart_snap_cuboid_candidate(
        &graph,
        &index,
        hit,
        gridded,
        PlacementGrid::Centimetres25,
        1.0,
        |_| false,
    );
    assert!(rejected_guides.is_empty());
    assert_eq!(fallback.spec.pose, gridded.spec.pose);
}

#[test]
fn dense_aligned_parts_do_not_multiply_smart_snap_validation() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply_batch((-3..=3).flat_map(|x| {
            (-3..=3).map(move |z| {
                BuildCommand::Spawn(
                    CuboidSpec::new(
                        [1; 3],
                        BuildPose::from_position_ticks(
                            IVec3::new(x * 100, 50, z * 100),
                            GridRotation::default(),
                        ),
                    )
                    .unwrap(),
                )
            })
        }))
        .unwrap();
    let hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::new(0.0, 0.0, 0.0),
        face: FaceRef::ground(),
    };
    let gridded = oriented_cuboid_candidate_from_hit_with_grid(
        &graph,
        hit,
        [1; 3],
        GridRotation::default(),
        PlacementGrid::Centimetres25,
        PlacementBounds::Garage,
    );
    let mut index = PlacementSnapIndex::default();
    index.rebuild(&graph);
    let mut validation_count = 0;

    let (fallback, guides) = smart_snap_cuboid_candidate(
        &graph,
        &index,
        hit,
        gridded,
        PlacementGrid::Centimetres25,
        1.0,
        |_| {
            validation_count += 1;
            false
        },
    );

    assert_eq!(fallback.spec.pose, gridded.spec.pose);
    assert!(guides.is_empty());
    assert!(
        validation_count <= 25,
        "equivalent guides caused {validation_count} duplicate validations"
    );
}

#[test]
fn framed_overlap_validation_agrees_across_snap_batch_and_volume_paths() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(target) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([8, 1, 1], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    graph
        .reframe_parts(
            [target],
            mechanic_core::ConstructionFrame::new(
                Vec3::new(2.0, 1.0, 3.0),
                Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
            )
            .unwrap(),
        )
        .unwrap();
    let mut index = PlacementSnapIndex::default();
    index.rebuild(&graph);
    let bounds = PlacementBounds::World {
        origin: DVec2::ZERO,
    };
    // Inside the rotated rod, at its obsolete raw origin, and inside its
    // AABB but outside its actual oriented volume, respectively.
    for (units, overlaps) in [
        (IVec3::new(8, 4, 12), true),
        (IVec3::ZERO, false),
        (IVec3::new(10, 4, 14), false),
    ] {
        let spec = CuboidSpec::new([1; 3], BuildPose::new(units, GridRotation::default())).unwrap();
        let start = PlacementCandidate {
            spec,
            attached_face: FaceKind::NegativeY,
            anchor: None,
            support: PlacementSupport::Free,
        };
        let volume = BlockVolume::new(spec, IVec3::ZERO).unwrap();
        if units != IVec3::ZERO {
            let (minimum, maximum) = super::part_world_bounds(PartSpec::Cuboid(spec));
            assert!(
                index
                    .nearby(minimum, maximum, 0.0)
                    .iter()
                    .any(|row| row.part == target)
            );
        }
        assert_eq!(index.overlaps(PartSpec::Cuboid(spec)), overlaps);
        let expected = if overlaps {
            Err(PlacementError::OverlapsPart(target))
        } else {
            Ok(())
        };
        assert_eq!(
            validate_indexed_block_batch_in_bounds(&index, start, &[spec], bounds),
            expected
        );
        assert_eq!(
            validate_block_batch_in_bounds(&graph, start, &[spec], bounds),
            expected
        );
        assert_eq!(
            super::validate_block_volume_in_bounds(&graph, &index, start, volume, bounds),
            expected
        );
    }
}

#[test]
fn framed_overlap_keeps_cylinder_bores_empty_and_identity_behavior_exact() {
    let mut graph = ConstructionGraph::new();
    let target = spawn_cylinder(
        &mut graph,
        CylinderDimensions::new(2.0, 1.0, 0.5).unwrap(),
        BuildPose::default(),
    );
    let frame =
        mechanic_core::ConstructionFrame::new(Vec3::new(2.0, 1.0, 3.0), Quat::from_rotation_y(0.6))
            .unwrap();
    graph.reframe_parts([target], frame).unwrap();
    let mut index = PlacementSnapIndex::default();
    index.rebuild(&graph);
    for (x, overlaps) in [(8, false), (11, true)] {
        let candidate = PartSpec::Cuboid(
            CuboidSpec::new(
                [1; 3],
                BuildPose::new(IVec3::new(x, 4, 12), GridRotation::default()),
            )
            .unwrap(),
        );
        assert_eq!(index.overlaps(candidate), overlaps);
        assert_eq!(
            super::bounds::parts_overlap_with_frame(
                candidate,
                *graph.part(target).unwrap(),
                mechanic_core::ConstructionFrame::IDENTITY
            ),
            super::parts_overlap(candidate, *graph.part(target).unwrap())
        );
    }
}

#[test]
fn indexed_preview_validation_rejects_only_nearby_overlaps() {
    let mut graph = ConstructionGraph::new();
    let existing = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap();
    let BuildOutcome::Spawned(existing) = existing else {
        panic!("spawn returned a different outcome");
    };
    let mut index = PlacementSnapIndex::default();
    index.rebuild(&graph);
    let candidate = |x| PlacementCandidate {
        spec: CuboidSpec::new(
            [1; 3],
            BuildPose::new(IVec3::new(x, 0, 0), GridRotation::default()),
        )
        .unwrap(),
        attached_face: FaceKind::NegativeY,
        anchor: None,
        support: PlacementSupport::Free,
    };

    assert!(matches!(
        validate_indexed_block_batch_in_bounds(
            &index,
            candidate(0),
            &[candidate(0).spec],
            PlacementBounds::World {
                origin: DVec2::ZERO,
            },
        ),
        Err(PlacementError::OverlapsPart(part)) if part == existing
    ));
    assert!(
        validate_indexed_block_batch_in_bounds(
            &index,
            candidate(1_000),
            &[candidate(1_000).spec],
            PlacementBounds::World {
                origin: DVec2::ZERO,
            },
        )
        .is_ok()
    );
}

#[test]
fn free_smart_snap_can_align_all_three_axes() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [1; 3],
                BuildPose::from_position_ticks(IVec3::new(6, 56, 406), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap();
    let mut index = PlacementSnapIndex::default();
    index.rebuild(&graph);
    let spec = CuboidSpec::new(
        [1; 3],
        BuildPose::from_position_ticks(IVec3::new(0, 50, 300), GridRotation::default()),
    )
    .unwrap();
    let gridded = PlacementCandidate {
        spec,
        attached_face: FaceKind::NegativeY,
        anchor: None,
        support: PlacementSupport::Free,
    };

    let (snapped, guides) = smart_snap_free_cuboid_candidate(
        &index,
        gridded,
        PlacementGrid::Centimetres25,
        1.0,
        |_| true,
    );

    assert_eq!(
        snapped.spec.pose.translation_position_ticks(),
        IVec3::new(6, 56, 306)
    );
    assert_eq!(guides.len(), 2);
    assert!(guides.iter().all(|guide| {
        let delta = (guide.to - guide.from).abs();
        [delta.x, delta.y, delta.z]
            .into_iter()
            .filter(|component| *component > f32::EPSILON)
            .count()
            == 1
    }));
}

#[test]
fn free_smart_snap_guides_never_connect_centers_diagonally() {
    let mut graph = ConstructionGraph::new();
    let part = spawn_cube(&mut graph, IVec3::ZERO, 1);
    let diagonal = AxisGuide {
        delta: 0.01,
        coordinate: 1.0,
        kind: GuideKind::Center,
        part,
        target_center: Vec3::new(1.0, 2.0, 3.0),
    };
    let choices = [vec![diagonal], Vec::new(), Vec::new()];

    assert!(
        render_free_smart_guides(
            [Some(diagonal), None, None],
            &choices,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .is_empty()
    );

    let cardinal = AxisGuide {
        target_center: Vec3::new(1.0, 0.0, 3.0),
        ..diagonal
    };
    let choices = [vec![cardinal], Vec::new(), Vec::new()];
    let guides = render_free_smart_guides(
        [Some(cardinal), None, None],
        &choices,
        Vec3::new(1.0, 0.0, 0.0),
    );

    assert_eq!(guides.len(), 1);
    assert_eq!(guides[0].from, Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(guides[0].to, Vec3::new(1.0, 0.0, 3.0));
}

#[test]
fn single_axis_smart_snap_preserves_the_other_grid_axis() {
    let mut graph = ConstructionGraph::new();
    let support = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_position_ticks(IVec3::new(40, 50, 280), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap();
    let BuildOutcome::Spawned(support) = support else {
        unreachable!();
    };
    let mut target_graph = ConstructionGraph::new();
    target_graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_position_ticks(IVec3::new(6, 50, 480), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap();
    let hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::new(0.014, 0.25, 0.63),
        face: FaceRef::part(support, FaceKind::PositiveY),
    };
    let gridded = oriented_cuboid_candidate_from_hit_with_grid(
        &graph,
        hit,
        [1, 1, 1],
        GridRotation::default(),
        PlacementGrid::Centimetres25,
        PlacementBounds::Garage,
    );
    let mut index = PlacementSnapIndex::default();
    index.rebuild(&target_graph);

    let (snapped, guides) = smart_snap_cuboid_candidate(
        &graph,
        &index,
        hit,
        gridded,
        PlacementGrid::Centimetres25,
        1.0,
        |_| true,
    );

    assert_eq!(
        snapped.spec.pose.translation_position_ticks(),
        IVec3::new(6, 150, 300)
    );
    assert_eq!(guides.len(), 1);
    assert_eq!(guides[0].axis, 0);
    assert!((guides[0].from.x - guides[0].to.x).abs() < f32::EPSILON);
    assert!((guides[0].from.y - guides[0].to.y).abs() < f32::EPSILON);
    assert!((guides[0].from.z - guides[0].to.z).abs() > f32::EPSILON);
}

#[test]
fn smart_snap_is_stable_within_a_grid_cell_and_shows_coincident_guides() {
    let mut graph = ConstructionGraph::new();
    for ticks in [
        IVec3::new(8, 50, 100),
        IVec3::new(8, 50, 500),
        IVec3::new(-8, 50, 700),
    ] {
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_position_ticks(ticks, GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
    }
    let mut index = PlacementSnapIndex::default();
    index.rebuild(&graph);
    let snap = |point| {
        let hit = SurfaceHit {
            distance: 1.0,
            point,
            face: FaceRef::ground(),
        };
        let gridded = oriented_cuboid_candidate_from_hit_with_grid(
            &graph,
            hit,
            [1, 1, 1],
            GridRotation::default(),
            PlacementGrid::Centimetres25,
            PlacementBounds::Garage,
        );
        smart_snap_cuboid_candidate(
            &graph,
            &index,
            hit,
            gridded,
            PlacementGrid::Centimetres25,
            1.0,
            |_| true,
        )
    };

    let (right_candidate, right_guides) = snap(Vec3::new(0.024, 0.0, 0.63));
    let (left_candidate, left_guides) = snap(Vec3::new(-0.024, 0.0, 0.63));

    assert_eq!(right_candidate.spec.pose, left_candidate.spec.pose);
    assert_eq!(
        right_candidate.spec.pose.translation_position_ticks(),
        IVec3::new(8, 50, 300)
    );
    assert_eq!(right_guides, left_guides);
    assert_eq!(right_guides.len(), 2);
    assert!(right_guides.iter().all(|guide| guide.axis == 0));
}

#[test]
fn block_drag_snaps_its_other_corner_on_two_axes_and_keeps_whole_blocks() {
    let mut graph = ConstructionGraph::new();
    for ticks in [
        IVec3::new(200, 0, 400),
        IVec3::new(200, 0, -100),
        IVec3::new(400, 0, 300),
    ] {
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_position_ticks(ticks, GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
    }
    let mut index = PlacementSnapIndex::default();
    index.rebuild(&graph);
    let start = CuboidSpec::new([1, 1, 1], BuildPose::default()).unwrap();
    let snap = |pointer| {
        smart_snap_block_span(
            &index,
            start,
            PlacementPlane::Xz,
            IVec3::new(1, 0, 2),
            pointer,
            1.0,
            |_| true,
        )
    };

    let (left, left_guides) = snap(Vec3::new(0.49, 0.0, 0.74));
    let (right, right_guides) = snap(Vec3::new(0.51, 0.0, 0.76));

    assert_eq!(left, IVec3::new(2, 0, 3));
    assert_eq!(right, left);
    assert_eq!(right_guides, left_guides);
    assert!(left_guides.iter().any(|guide| guide.axis == 0));
    assert!(left_guides.iter().any(|guide| guide.axis == 2));
    assert!(
        left_guides.iter().filter(|guide| guide.axis == 0).count() >= 2,
        "every committed object sharing the endpoint alignment may show a guide"
    );

    let specs = block_box_specs(start, left).unwrap();
    assert!(specs.iter().all(|spec| {
        spec.pose
            .translation_position_ticks()
            .to_array()
            .into_iter()
            .all(|ticks| ticks % POSITION_TICKS_PER_GRID_UNIT == 0)
    }));
}

#[test]
fn invalid_block_endpoint_guide_falls_back_independently_per_axis() {
    let mut graph = ConstructionGraph::new();
    for ticks in [IVec3::new(200, 0, 400), IVec3::new(400, 0, 300)] {
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_position_ticks(ticks, GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
    }
    let mut index = PlacementSnapIndex::default();
    index.rebuild(&graph);
    let start = CuboidSpec::new([1, 1, 1], BuildPose::default()).unwrap();

    let (span, guides) = smart_snap_block_span(
        &index,
        start,
        PlacementPlane::Xz,
        IVec3::new(1, 0, 2),
        Vec3::new(0.50, 0.0, 0.75),
        1.0,
        |candidate| candidate.x != 2,
    );

    assert_eq!(span, IVec3::new(1, 0, 3));
    assert!(guides.iter().any(|guide| guide.axis == 2));
}

#[test]
fn transmission_preview_accepts_only_the_current_positive_z_output() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(engine) = graph
        .apply(BuildCommand::SpawnEngine(EngineSpec::new(
            EngineKind::Gas,
            BuildPose::new(IVec3::new(0, 1, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let output = FaceRef::part(engine, FaceKind::PositiveZ);
    let output_geometry = face_geometry_from_ref(output, Some(&graph));
    let hit = SurfaceHit {
        distance: 0.0,
        point: output_geometry.center,
        face: output,
    };
    let (parent, candidate) = transmission_candidate_from_hit(&graph, hit).unwrap();
    let staged = stage_transmission(&graph, parent, candidate).unwrap();
    assert_eq!(staged.engine_transmission_depth(engine), Some(1));
    assert!(matches!(
        transmission_candidate_from_hit(
            &graph,
            SurfaceHit {
                face: FaceRef::part(engine, FaceKind::PositiveX),
                ..hit
            }
        ),
        Err(PlacementError::TransmissionOutputOnly)
    ));
    assert!(transmission_candidate_from_hit(&staged, hit).is_err());
}

#[test]
fn rays_pass_through_cylinder_bores_but_hit_annular_material() {
    let mut graph = ConstructionGraph::new();
    let cylinder = spawn_cylinder(
        &mut graph,
        CylinderDimensions::new(1.0, 0.5, 1.0).unwrap(),
        BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
    );
    let cube = spawn_cube(&mut graph, IVec3::new(0, 4, 0), 1);
    let through_bore = raycast_construction(&graph, Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y).unwrap();
    assert_eq!(through_bore.face.owner, FaceOwner::Part(cube));
    let annulus = raycast_construction(&graph, Vec3::new(0.4, 5.0, 0.0), Vec3::NEG_Y).unwrap();
    assert_eq!(annulus.face.owner, FaceOwner::Part(cylinder));
}

#[test]
fn cylinder_sector_raycast_hits_retained_caps_and_cut_walls_only() {
    let mut graph = ConstructionGraph::new();
    let dimensions = CylinderDimensions::new(1.0, 0.0, 1.0)
        .unwrap()
        .with_sweep_angle_degrees(90)
        .unwrap();
    let cylinder = spawn_cylinder(
        &mut graph,
        dimensions,
        BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
    );
    let cube = spawn_cube(&mut graph, IVec3::new(0, 4, 0), 4);

    let retained = raycast_construction(&graph, Vec3::new(0.3, 5.0, 0.0), Vec3::NEG_Y).unwrap();
    assert_eq!(retained.face.owner, FaceOwner::Part(cylinder));
    assert_eq!(retained.face.face, FaceKind::PositiveY);

    let missing = raycast_construction(&graph, Vec3::new(-0.3, 5.0, 0.0), Vec3::NEG_Y).unwrap();
    assert_eq!(missing.face.owner, FaceOwner::Part(cube));

    let cut_wall = raycast_construction(&graph, Vec3::new(0.3, 2.0, 2.0), Vec3::NEG_Z).unwrap();
    assert_eq!(cut_wall.face.owner, FaceOwner::Part(cylinder));
    assert_eq!(cut_wall.face.face, FaceKind::PositiveX);
}

#[test]
fn annular_placement_stops_at_a_bore_only_when_material_cannot_pass() {
    let mut graph = ConstructionGraph::new();
    let cylinder = spawn_cylinder(
        &mut graph,
        CylinderDimensions::new(1.0, 0.5, 1.0).unwrap(),
        BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
    );
    let cube = spawn_cube(&mut graph, IVec3::new(0, 4, 0), 1);
    let origin = Vec3::new(0.0, 5.0, 0.0);

    let fitting = raycast_construction_for_annulus(&graph, origin, Vec3::NEG_Y, 0.0, 0.4).unwrap();
    assert_eq!(fitting.face.owner, FaceOwner::Part(cube));

    let obstructed =
        raycast_construction_for_annulus(&graph, origin, Vec3::NEG_Y, 0.0, 0.6).unwrap();
    assert_eq!(obstructed.face.owner, FaceOwner::Part(cylinder));
    assert_eq!(obstructed.face.face, FaceKind::PositiveY);
    let candidate = cylinder_candidate_from_hit(
        &graph,
        obstructed,
        CylinderDimensions::new(0.6, 0.0, 0.25).unwrap(),
    )
    .unwrap();
    let staged = stage_cylinder_from_source(&graph, candidate, obstructed.face.owner).unwrap();
    assert_eq!(staged.weld_count(), 1);

    let surrounding_sleeve =
        raycast_construction_for_annulus(&graph, origin, Vec3::NEG_Y, 1.1, 1.2).unwrap();
    assert_eq!(surrounding_sleeve.face.owner, FaceOwner::Part(cube));
}

#[test]
fn bearing_can_center_over_a_bore_when_its_ring_has_support() {
    let mut graph = ConstructionGraph::new();
    let cylinder = spawn_cylinder(
        &mut graph,
        CylinderDimensions::new(1.0, 0.5, 1.0).unwrap(),
        BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
    );
    let source = FaceRef::part(cylinder, FaceKind::PositiveY);
    let hit = SurfaceHit {
        distance: 2.5,
        point: Vec3::new(0.0, 2.5, 0.0),
        face: source,
    };

    let anchor = bearing_anchor_from_hit(&graph, hit).unwrap();
    assert_eq!(anchor, hit.point);
    assert_eq!(
        bearing_support_face(
            &graph,
            source,
            anchor,
            BearingDimensions::new(0.6, 0.2).unwrap(),
        ),
        Some(source)
    );
    assert!(
        bearing_support_face(
            &graph,
            source,
            anchor,
            BearingDimensions::new(0.4, 0.2).unwrap(),
        )
        .is_none()
    );
}

#[test]
fn small_blocks_can_occupy_a_large_cylinder_bore() {
    let mut graph = ConstructionGraph::new();
    spawn_cylinder(
        &mut graph,
        CylinderDimensions::new(1.0, 0.6, 1.0).unwrap(),
        BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
    );
    let cube = CuboidSpec::new(
        [1; 3],
        BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
    )
    .unwrap();
    let candidate = PlacementCandidate {
        spec: cube,
        attached_face: FaceKind::NegativeY,
        anchor: Some(Vec3::ZERO),
        support: PlacementSupport::Surface(FaceOwner::Ground),
    };
    assert!(stage_cuboid(&graph, candidate).is_ok());
}

#[test]
fn blocks_can_occupy_the_missing_side_of_a_cylinder_sector() {
    let mut graph = ConstructionGraph::new();
    spawn_cylinder(
        &mut graph,
        CylinderDimensions::new(1.0, 0.0, 1.0)
            .unwrap()
            .with_sweep_angle_degrees(90)
            .unwrap(),
        BuildPose::new(IVec3::new(0, 4, 0), GridRotation::default()),
    );
    let cube = CuboidSpec::new(
        [1; 3],
        BuildPose::new(IVec3::new(-1, 4, 0), GridRotation::default()),
    )
    .unwrap();
    let candidate = PlacementCandidate {
        spec: cube,
        attached_face: FaceKind::NegativeY,
        anchor: Some(Vec3::ZERO),
        support: PlacementSupport::Surface(FaceOwner::Ground),
    };

    assert!(stage_cuboid(&graph, candidate).is_ok());
}

#[test]
fn cylinders_place_along_all_six_flat_face_normals() {
    let cases = [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ];
    for outward in cases {
        let mut graph = ConstructionGraph::new();
        let support = spawn_cube(&mut graph, IVec3::new(0, 16, 0), 4);
        let hit = super::raycast::raycast_cuboid(
            Vec3::new(0.0, 4.0, 0.0) + outward * 5.0,
            -outward,
            support,
            graph.part(support).copied().unwrap().as_cuboid().unwrap(),
        )
        .unwrap();
        let candidate = cylinder_candidate_from_hit(
            &graph,
            hit,
            CylinderDimensions::new(0.25, 0.0, 0.5).unwrap(),
        )
        .unwrap();
        let axis = candidate.spec.pose.rotation.quaternion() * Vec3::Y;
        assert!(axis.abs_diff_eq(outward, 1.0e-6));
        assert!(stage_cylinder_from_source(&graph, candidate, hit.face.owner).is_ok());
    }
}

#[test]
fn thin_annular_cylinder_places_on_a_coplanar_block_sheet() {
    let mut graph = ConstructionGraph::new();
    let mut center = None;
    for x in -2..=2 {
        for z in -2..=2 {
            let spec = CuboidSpec::new(
                [1; 3],
                BuildPose::from_half_grid(IVec3::new(x * 2, 1, z * 2), GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            if x == 0 && z == 0 {
                center = Some(part);
            }
        }
    }
    let center = center.unwrap();
    let hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::new(0.0, 0.25, 0.0),
        face: FaceRef::part(center, FaceKind::PositiveY),
    };
    let candidate = cylinder_candidate_from_hit(
        &graph,
        hit,
        CylinderDimensions::new(0.75, 0.70, 0.25).unwrap(),
    )
    .unwrap();

    assert!(candidate.anchor.is_some());
    assert!(stage_cylinder_from_source(&graph, candidate, hit.face.owner).is_ok());
}

#[test]
fn bearing_anchor_rejects_a_curved_cylinder_wall() {
    let mut graph = ConstructionGraph::new();
    let cylinder = spawn_cylinder(
        &mut graph,
        CylinderDimensions::new(0.5, 0.25, 0.5).unwrap(),
        BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
    );
    let curved_hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::new(0.25, 0.5, 0.0),
        face: FaceRef::part(cylinder, FaceKind::PositiveX),
    };

    assert_eq!(
        bearing_anchor_from_hit(&graph, curved_hit),
        Err(PlacementError::CurvedSurface)
    );
}

#[test]
fn filtered_raycast_reaches_an_accepted_part_behind_an_excluded_one() {
    let mut graph = ConstructionGraph::new();
    let near = spawn_cube(&mut graph, IVec3::ZERO, 4);
    let far = spawn_cube(&mut graph, IVec3::new(0, 0, -8), 4);
    let origin = Vec3::new(0.0, 0.0, 4.0);
    assert_eq!(
        raycast_construction_with_ground(&graph, origin, Vec3::NEG_Z, None)
            .unwrap()
            .face
            .owner,
        FaceOwner::Part(near)
    );
    let hit = super::raycast_construction_filtered_with_ground(
        &graph,
        origin,
        Vec3::NEG_Z,
        None,
        |part| part == far,
    )
    .unwrap();
    assert_eq!(hit.face.owner, FaceOwner::Part(far));
    assert!((hit.distance - 5.5).abs() < 1.0e-5);
}

#[test]
fn filtered_annulus_raycast_excludes_bore_obstructions_too() {
    let mut graph = ConstructionGraph::new();
    let far = spawn_cube(&mut graph, IVec3::ZERO, 4);
    let cylinder = spawn_cylinder(
        &mut graph,
        CylinderDimensions::new(2.0, 1.0, 0.5).unwrap(),
        BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
    );
    let origin = Vec3::Y * 4.0;
    let unfiltered = super::raycast_construction_for_annulus_with_ground(
        &graph,
        origin,
        Vec3::NEG_Y,
        0.5,
        1.5,
        None,
    )
    .unwrap();
    assert_eq!(unfiltered.face.owner, FaceOwner::Part(cylinder));
    let hit = super::raycast_construction_for_annulus_filtered_with_ground(
        &graph,
        origin,
        Vec3::NEG_Y,
        0.5,
        1.5,
        None,
        |part| part == far,
    )
    .unwrap();
    assert_eq!(hit.face.owner, FaceOwner::Part(far));
    assert!((hit.distance - 3.5).abs() < 1.0e-5);
}

#[test]
fn reframed_primitive_picking_faces_and_bounds_follow_the_authored_frame() {
    let mut graph = ConstructionGraph::new();
    let part = spawn_cube(&mut graph, IVec3::new(0, 4, 0), 4);
    let origin = Vec3::new(0.0, 4.0, 0.0);
    let hit = raycast_construction_with_ground(&graph, origin, Vec3::NEG_Y, None).unwrap();
    let face = face_geometry_from_ref(hit.face, Some(&graph));
    let frame = mechanic_core::ConstructionFrame::new(
        Vec3::new(4.0, 3.0, 2.0),
        Quat::from_rotation_z(0.47) * Quat::from_rotation_y(0.31),
    )
    .unwrap();
    graph.reframe_parts([part], frame).unwrap();
    let reframed = raycast_construction_with_ground(
        &graph,
        frame.point(origin),
        frame.vector(Vec3::NEG_Y),
        None,
    )
    .unwrap();
    assert_eq!(reframed.face, hit.face);
    assert!(reframed.point.distance(frame.point(hit.point)) < 1.0e-5);
    assert!((reframed.distance - hit.distance).abs() < 1.0e-5);
    let reframed_face = face_geometry_from_ref(reframed.face, Some(&graph));
    assert!(reframed_face.center.distance(frame.point(face.center)) < 1.0e-5);
    assert!(reframed_face.normal.distance(frame.vector(face.normal)) < 1.0e-5);
    assert!(
        reframed_face
            .tangent_u
            .distance(frame.vector(face.tangent_u))
            < 1.0e-5
    );
    assert!(
        reframed_face
            .tangent_v
            .distance(frame.vector(face.tangent_v))
            < 1.0e-5
    );
    let (minimum, maximum) = super::composed_part_world_bounds(&graph, part).unwrap();
    assert!(minimum.cmple(reframed.point + Vec3::splat(1.0e-5)).all());
    assert!(maximum.cmpge(reframed.point - Vec3::splat(1.0e-5)).all());
    assert!(((minimum + maximum) * 0.5).distance(frame.point(Vec3::Y)) < 1.0e-5);
}

#[test]
fn reframed_annulus_obstruction_uses_the_cylinder_frame() {
    let mut graph = ConstructionGraph::new();
    let part = spawn_cylinder(
        &mut graph,
        CylinderDimensions::new(1.0, 0.5, 1.0).unwrap(),
        BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
    );
    let origin = Vec3::new(0.0, 5.0, 0.0);
    let hit = super::raycast_construction_for_annulus_with_ground(
        &graph,
        origin,
        Vec3::NEG_Y,
        0.0,
        0.6,
        None,
    )
    .unwrap();
    let frame =
        mechanic_core::ConstructionFrame::new(Vec3::new(2.0, 3.0, 4.0), Quat::from_rotation_z(0.6))
            .unwrap();
    graph.reframe_parts([part], frame).unwrap();
    let reframed = super::raycast_construction_for_annulus_with_ground(
        &graph,
        frame.point(origin),
        frame.vector(Vec3::NEG_Y),
        0.0,
        0.6,
        None,
    )
    .unwrap();
    assert_eq!(reframed.face, hit.face);
    assert!(reframed.point.distance(frame.point(hit.point)) < 1.0e-5);
    assert!((reframed.distance - hit.distance).abs() < 1.0e-5);
}

#[test]
fn reframed_evaluated_solid_picking_and_faces_apply_the_frame_once() {
    let mut graph = ConstructionGraph::new();
    let part = spawn_cube(&mut graph, IVec3::new(0, 4, 0), 4);
    let owner = SolidOwner::Part(part);
    let edge = graph.evaluated_solid(owner).unwrap().logical_edges[0].key;
    graph
        .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
            [EdgeChainRef { owner, edge }],
            EdgeTreatment::Fillet,
            20,
        )))
        .unwrap();
    let origin = Vec3::new(0.0, 4.0, 0.0);
    let hit = raycast_construction_with_ground(&graph, origin, Vec3::NEG_Y, None).unwrap();
    let face = face_geometry_from_ref(hit.face, Some(&graph));
    let frame =
        mechanic_core::ConstructionFrame::new(Vec3::new(3.0, 2.0, 1.0), Quat::from_rotation_z(0.9))
            .unwrap();
    graph.reframe_parts([part], frame).unwrap();
    let reframed = raycast_construction_with_ground(
        &graph,
        frame.point(origin),
        frame.vector(Vec3::NEG_Y),
        None,
    )
    .unwrap();
    assert_eq!(reframed.face, hit.face);
    assert!(reframed.point.distance(frame.point(hit.point)) < 1.0e-5);
    let reframed_face = face_geometry_from_ref(reframed.face, Some(&graph));
    assert!(reframed_face.center.distance(frame.point(face.center)) < 1.0e-5);
    assert!(reframed_face.normal.distance(frame.vector(face.normal)) < 1.0e-5);
}

#[test]
fn raycast_selects_nearest_cuboid_face_before_ground() {
    let mut graph = ConstructionGraph::new();
    spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
    let hit = raycast_construction(&graph, Vec3::new(0.0, 4.0, 0.0), Vec3::NEG_Y)
        .expect("cube is under ray");
    assert_eq!(hit.face.face, FaceKind::PositiveY);
    assert!(matches!(hit.face.owner, FaceOwner::Part(_)));
    assert!((hit.point.y - 1.0).abs() < 1.0e-6);
}

#[test]
fn raycast_from_below_ignores_the_floor_and_reaches_the_underside() {
    let mut graph = ConstructionGraph::new();
    let part = spawn_cube(&mut graph, IVec3::new(0, 4, 0), 1);

    let hit = raycast_construction(&graph, Vec3::new(0.0, -1.0, 0.0), Vec3::Y)
        .expect("the ray reaches the elevated block");

    assert_eq!(hit.face.owner, FaceOwner::Part(part));
    assert_eq!(hit.face.face, FaceKind::NegativeY);
    assert!((hit.point.y - 0.875).abs() < 1.0e-6);
}

#[test]
fn fixed_quarter_metre_blocks_place_flush_on_ground_and_faces() {
    let graph = ConstructionGraph::new();
    let ground_hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::ZERO,
        face: mechanic_core::FaceRef::ground(),
    };
    let candidate = candidate_from_hit(&graph, ground_hit);
    assert_eq!(
        candidate
            .spec
            .dimensions
            .map(mechanic_core::GridDimension::units),
        [1; 3]
    );
    assert!((candidate.spec.pose.translation().y - BLOCK_SIZE_METERS * 0.5).abs() < 1.0e-6);
    let graph = stage_cuboid(&graph, candidate).unwrap();

    let top = raycast_construction(&graph, Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y)
        .expect("placed block is under ray");
    let attached = candidate_from_hit(&graph, top);
    assert!((attached.spec.pose.translation().y - 0.375).abs() < 1.0e-6);
    assert!(stage_cuboid(&graph, attached).is_ok());
}

#[test]
fn gas_engine_places_flush_with_its_authored_footprint_and_stays_semantic() {
    let graph = ConstructionGraph::new();
    let ground_hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::ZERO,
        face: FaceRef::ground(),
    };
    let candidate = cuboid_candidate_from_hit(&graph, ground_hit, EngineKind::Gas.grid_units());

    assert_eq!(
        candidate.spec.pose.translation(),
        Vec3::new(0.125, 0.25, 0.0)
    );
    assert_eq!(candidate.spec.size_meters(), Vec3::new(0.5, 0.5, 0.75));

    let graph =
        stage_engine_from_source(&graph, candidate, FaceOwner::Ground, EngineKind::Gas).unwrap();
    let (_, part) = graph.parts().next().expect("the engine was staged");
    assert!(matches!(
        part,
        PartSpec::Engine(engine) if engine.kind == EngineKind::Gas
    ));
    assert_eq!(graph.welds().count(), 1);
}

#[test]
fn electric_engine_spans_two_by_two_ground_cells_without_a_half_block_offset() {
    let graph = ConstructionGraph::new();
    let candidate = cuboid_candidate_from_hit(
        &graph,
        SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        },
        EngineKind::Electric.grid_units(),
    );

    assert_eq!(
        candidate.spec.pose.translation_half_units(),
        IVec3::new(1, 2, 1)
    );
    assert_eq!(candidate.spec.size_meters(), Vec3::splat(0.5));
    assert_eq!(
        super::bounds::cuboid_world_bounds(candidate.spec),
        (Vec3::new(-0.125, 0.0, -0.125), Vec3::new(0.375, 0.5, 0.375))
    );
}

#[test]
fn quarter_turn_rotates_an_authored_footprint_and_survives_staging() {
    let graph = ConstructionGraph::new();
    let hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::ZERO,
        face: FaceRef::ground(),
    };
    let candidate = oriented_cuboid_candidate_from_hit(
        &graph,
        hit,
        EngineKind::Gas.grid_units(),
        GridRotation::new(0, 1, 0),
    );

    assert_eq!(candidate.spec.pose.rotation.quarter_turns_xyz(), [0, 1, 0]);
    assert_eq!(
        candidate.spec.pose.translation_half_units(),
        IVec3::new(0, 2, 1)
    );
    assert_eq!(candidate.attached_face, FaceKind::NegativeY);
    let (minimum, maximum) = super::bounds::cuboid_world_bounds(candidate.spec);
    assert!((maximum.x - minimum.x - 0.75).abs() < 1.0e-6);
    assert!((maximum.z - minimum.z - 0.50).abs() < 1.0e-6);

    let staged =
        stage_engine_from_source(&graph, candidate, FaceOwner::Ground, EngineKind::Gas).unwrap();
    let (_, PartSpec::Engine(engine)) = staged.parts().next().unwrap() else {
        panic!("the staged part must remain an engine")
    };
    assert_eq!(engine.pose.rotation.quarter_turns_xyz(), [0, 1, 0]);
}

#[test]
fn every_authored_orientation_attaches_flush_from_every_world_face() {
    for outward in [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ] {
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    let mut graph = ConstructionGraph::new();
                    let support = spawn_cube(&mut graph, IVec3::new(0, 16, 0), 4);
                    let hit = super::raycast::raycast_cuboid(
                        Vec3::new(0.0, 4.0, 0.0) + outward * 5.0,
                        -outward,
                        support,
                        graph.part(support).copied().unwrap().as_cuboid().unwrap(),
                    )
                    .expect("ray reaches requested support face");
                    let candidate = oriented_cuboid_candidate_from_hit(
                        &graph,
                        hit,
                        EngineKind::Gas.grid_units(),
                        GridRotation::new(x, y, z),
                    );
                    let attached =
                        super::faces::face_geometry(candidate.spec, candidate.attached_face);

                    assert!(attached.normal.abs_diff_eq(-outward, 1.0e-6));
                    assert!(
                        stage_engine_from_source(
                            &graph,
                            candidate,
                            FaceOwner::Part(support),
                            EngineKind::Gas,
                        )
                        .is_ok()
                    );
                }
            }
        }
    }
}

#[test]
fn tipped_authored_part_uses_its_rotated_height_and_footprint() {
    let graph = ConstructionGraph::new();
    let candidate = oriented_cuboid_candidate_from_hit(
        &graph,
        SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        },
        EngineKind::Gas.grid_units(),
        GridRotation::new(1, 0, 0),
    );

    assert_eq!(candidate.spec.pose.rotation.quarter_turns_xyz(), [1, 0, 0]);
    let (minimum, maximum) = super::bounds::cuboid_world_bounds(candidate.spec);
    assert!((maximum.x - minimum.x - 0.5).abs() < 1.0e-6);
    assert!((maximum.y - minimum.y - 0.75).abs() < 1.0e-6);
    assert!((maximum.z - minimum.z - 0.5).abs() < 1.0e-6);
    assert!(minimum.y.abs() < 1.0e-6);
    assert_eq!(candidate.attached_face, FaceKind::PositiveZ);
}

#[test]
fn placement_works_from_all_six_cuboid_faces() {
    let cases = [
        (Vec3::X, FaceKind::PositiveX),
        (Vec3::NEG_X, FaceKind::NegativeX),
        (Vec3::Y, FaceKind::PositiveY),
        (Vec3::NEG_Y, FaceKind::NegativeY),
        (Vec3::Z, FaceKind::PositiveZ),
        (Vec3::NEG_Z, FaceKind::NegativeZ),
    ];
    for (outward, expected_face) in cases {
        let mut graph = ConstructionGraph::new();
        let part = spawn_cube(&mut graph, IVec3::new(0, 16, 0), 4);
        let hit = super::raycast::raycast_cuboid(
            Vec3::new(0.0, 4.0, 0.0) + outward * 5.0,
            -outward,
            part,
            graph.part(part).copied().unwrap().as_cuboid().unwrap(),
        )
        .expect("ray reaches requested face");
        assert_eq!(hit.face.face, expected_face);
        let candidate = candidate_from_hit(&graph, hit);
        assert!(stage_cuboid(&graph, candidate).is_ok());
    }
}

#[test]
fn side_placement_preserves_the_supporting_quarter_block_lattice() {
    for face in [
        FaceKind::PositiveX,
        FaceKind::NegativeX,
        FaceKind::PositiveZ,
        FaceKind::NegativeZ,
    ] {
        let graph = ConstructionGraph::new();
        let support = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::ground(),
            },
        );
        let graph = stage_cuboid(&graph, support).unwrap();
        let part = graph.parts().next().unwrap().0;
        let source = FaceRef::part(part, face);
        let source_face = super::face_geometry_from_ref(source, Some(&graph));

        let candidate = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: source_face.center,
                face: source,
            },
        );

        assert_eq!(
            candidate.spec.pose.translation_half_units().y,
            support.spec.pose.translation_half_units().y
        );
        assert!(stage_cuboid(&graph, candidate).is_ok());

        let bearing_candidate = bearing_attachment_candidate(&graph, source, source_face.center);
        assert_eq!(
            bearing_candidate.spec.pose.translation_half_units().y,
            support.spec.pose.translation_half_units().y
        );
        let attached = stage_bearing_attachment(
            &graph,
            bearing_candidate,
            source,
            source_face.center,
            BearingDimensions::default(),
        )
        .unwrap();
        assert_eq!(attached.bearing_count(), 1);
        assert_eq!(attached.weld_count(), 1);
    }
}

#[test]
fn framed_garage_bounds_check_every_transformed_corner() {
    use super::GROUND_HALF_SIZE;
    let local_min = Vec3::splat(-0.25);
    let local_max = Vec3::splat(0.25);
    let rotation = bevy::math::Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
    let middle_y = (crate::garage::BUILD_MIN_Y + crate::garage::BUILD_MAX_Y) * 0.5;
    let bounds_at = |translation| {
        PlacementBounds::GarageBuild
            .in_edit_frame(mechanic_core::ConstructionFrame::new(translation, rotation).unwrap())
    };
    assert!(
        super::validate_world_bounds(
            local_min,
            local_max,
            bounds_at(Vec3::new(0.0, middle_y, 0.0)),
        )
        .is_ok()
    );
    for center in [
        Vec3::new(-GROUND_HALF_SIZE + 0.3, middle_y, 0.0),
        Vec3::new(GROUND_HALF_SIZE - 0.3, middle_y, 0.0),
        Vec3::new(0.0, crate::garage::BUILD_MIN_Y + 0.3, 0.0),
        Vec3::new(0.0, crate::garage::BUILD_MAX_Y - 0.3, 0.0),
        Vec3::new(0.0, middle_y, -GROUND_HALF_SIZE + 0.2),
        Vec3::new(0.0, middle_y, GROUND_HALF_SIZE - 0.2),
    ] {
        assert_eq!(
            super::validate_world_bounds(local_min, local_max, bounds_at(center)),
            Err(PlacementError::OutsidePlatform),
            "center {center:?}",
        );
    }
    // Local coordinates can look valid while their world position is outside.
    let offset = mechanic_core::ConstructionFrame::new(
        Vec3::new(0.0, crate::garage::BUILD_MAX_Y, 0.0),
        bevy::math::Quat::IDENTITY,
    )
    .unwrap();
    let point = Vec3::new(0.0, middle_y, 0.0);
    assert!(super::validate_world_bounds(point, point, PlacementBounds::GarageBuild).is_ok());
    assert_eq!(
        super::validate_world_bounds(
            point,
            point,
            PlacementBounds::GarageBuild.in_edit_frame(offset)
        ),
        Err(PlacementError::OutsidePlatform),
    );
}

#[test]
fn placement_rejects_cubes_extending_beyond_platform() {
    let graph = ConstructionGraph::new();
    let hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::new(super::GROUND_HALF_SIZE, 0.0, 0.0),
        face: mechanic_core::FaceRef::ground(),
    };
    let candidate = candidate_from_hit(&graph, hit);
    assert!(matches!(
        stage_cuboid(&graph, candidate),
        Err(PlacementError::OutsidePlatform)
    ));
}

#[test]
fn world_terrain_placement_is_not_limited_to_the_garage_platform() {
    let graph = ConstructionGraph::new();
    let terrain_hit = SurfaceHit {
        distance: 4.0,
        point: Vec3::new(24.15, 13.337, -31.20),
        face: FaceRef::ground(),
    };
    let candidate = candidate_from_hit(&graph, terrain_hit);

    let bottom = candidate.spec.pose.translation().y - BLOCK_SIZE_METERS * 0.5;
    assert!(bottom <= terrain_hit.point.y);
    assert!(terrain_hit.point.y - bottom < 0.025);
    assert!(matches!(
        validate_block_batch_in_bounds(
            &graph,
            candidate,
            &[candidate.spec],
            PlacementBounds::Garage,
        ),
        Err(PlacementError::OutsidePlatform)
    ));
    assert!(
        validate_block_batch_in_bounds(
            &graph,
            candidate,
            &[candidate.spec],
            PlacementBounds::World {
                origin: DVec2::ZERO,
            },
        )
        .is_ok()
    );
}

#[test]
fn world_terrain_placement_does_not_create_a_flat_garage_ground_weld() {
    let graph = ConstructionGraph::new();
    let terrain_hit = SurfaceHit {
        distance: 4.0,
        point: Vec3::new(24.15, 0.0, -31.20),
        face: FaceRef::ground(),
    };
    let candidate = candidate_from_hit(&graph, terrain_hit);
    let staged = stage_block_batch_from_source_in_bounds(
        &graph,
        candidate,
        &[candidate.spec],
        FaceOwner::Ground,
        PlacementBounds::World {
            origin: DVec2::ZERO,
        },
    )
    .unwrap();

    assert_eq!(staged.weld_count(), 0);
    assert!(!staged.compile().unwrap().compounds[0].is_static);
}

#[test]
fn isolated_free_block_remains_unwelded() {
    let graph = ConstructionGraph::new();
    let candidate = free_cuboid_candidate(
        Vec3::new(0.0, crate::garage::BUILD_MIN_Y + 1.0, 0.0),
        Vec3::NEG_Z,
        [1; 3],
        GridRotation::default(),
        PlacementGrid::Centimetres25,
        PlacementBounds::GarageBuild,
    );
    let staged = stage_block_batch_in_bounds(
        &graph,
        candidate,
        &[candidate.spec],
        PlacementBounds::GarageBuild,
    )
    .unwrap();

    assert_eq!(staged.weld_count(), 0);
    assert!(!staged.compile().unwrap().compounds[0].is_static);
}

#[test]
fn free_candidates_snap_globally_and_face_the_view_cardinally() {
    let cuboid = free_cuboid_candidate(
        Vec3::new(0.18, 6.18, -0.18),
        Vec3::new(0.9, 0.1, 0.2),
        [1, 2, 3],
        GridRotation::new(0, 1, 0),
        PlacementGrid::Centimetres25,
        PlacementBounds::GarageBuild,
    );
    let ticks = cuboid.spec.pose.translation_position_ticks();
    let world_dimensions =
        super::candidates::oriented_grid_dimensions([1, 2, 3], cuboid.spec.pose.rotation);
    assert_eq!(
        ticks,
        super::grid::snap_global_center_ticks(
            super::grid::snap_world_to_position_ticks(Vec3::new(0.18, 6.18, -0.18)),
            world_dimensions,
            PlacementGrid::Centimetres25,
            PlacementBounds::GarageBuild,
        )
    );
    assert_eq!(cuboid.spec.pose.rotation, GridRotation::new(0, 1, 0));
    assert_eq!(cuboid.support, PlacementSupport::Free);

    let cylinder = free_cylinder_candidate(
        Vec3::new(0.18, 6.18, -0.18),
        Vec3::new(0.9, 0.1, 0.2),
        CylinderDimensions::default(),
        PlacementGrid::Centimetres25,
        PlacementBounds::GarageBuild,
    );
    let axis = cylinder.spec.pose.rotation.quaternion() * Vec3::Y;
    assert!(axis.abs_diff_eq(Vec3::NEG_X, 1.0e-5), "axis was {axis:?}");
    assert_eq!(cylinder.support, PlacementSupport::Free);
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one table exercises every standalone spawn path"
)]
fn every_standalone_tool_stages_an_isolated_free_part() {
    let point = Vec3::new(0.0, 7.5, 0.0);
    let direction = Vec3::NEG_Z;
    let bounds = PlacementBounds::GarageBuild;
    let grid = PlacementGrid::Centimetres25;
    let rotation = GridRotation::default();
    let cases = [
        (
            free_cuboid_candidate(
                point,
                direction,
                mechanic_core::ControllerSpec::GRID_UNITS,
                rotation,
                grid,
                bounds,
            ),
            0_u8,
        ),
        (
            free_cuboid_candidate(
                point,
                direction,
                EngineKind::Gas.grid_units(),
                rotation,
                grid,
                bounds,
            ),
            1,
        ),
        (
            free_cuboid_candidate(
                point,
                direction,
                EngineKind::Electric.grid_units(),
                rotation,
                grid,
                bounds,
            ),
            2,
        ),
        (
            free_cuboid_candidate(
                point,
                direction,
                mechanic_core::ServoSpec::GRID_UNITS,
                rotation,
                grid,
                bounds,
            ),
            3,
        ),
        (
            free_cuboid_candidate(
                point,
                direction,
                mechanic_core::SeatSpec::GRID_UNITS,
                rotation,
                grid,
                bounds,
            ),
            4,
        ),
        (
            free_cuboid_candidate(
                point,
                direction,
                mechanic_core::InputSpec::GRID_UNITS,
                rotation,
                grid,
                bounds,
            ),
            5,
        ),
        (
            free_cuboid_candidate(
                point,
                direction,
                mechanic_core::DimensionLinkSpec::GRID_UNITS,
                rotation,
                grid,
                bounds,
            ),
            6,
        ),
    ];
    for (candidate, kind) in cases {
        let graph = ConstructionGraph::new();
        let staged = match kind {
            0 => stage_controller_in_bounds(&graph, candidate, bounds),
            1 => stage_engine_in_bounds(&graph, candidate, EngineKind::Gas, bounds),
            2 => stage_engine_in_bounds(&graph, candidate, EngineKind::Electric, bounds),
            3 => stage_servo_in_bounds(&graph, candidate, bounds),
            4 => stage_seat_in_bounds(&graph, candidate, bounds),
            5 => stage_input_in_bounds(&graph, candidate, bounds),
            6 => stage_dimension_link_in_bounds(&graph, candidate, DimensionLinkId(1), bounds),
            _ => unreachable!(),
        }
        .unwrap();
        assert_eq!(staged.part_count(), 1);
        assert_eq!(staged.weld_count(), 0);
        assert!(!staged.compile().unwrap().compounds[0].is_static);
    }

    let graph = ConstructionGraph::new();
    let candidate = free_cuboid_candidate(point, direction, [1; 3], rotation, grid, bounds);
    let block = stage_block_batch_in_bounds(&graph, candidate, &[candidate.spec], bounds).unwrap();
    assert_eq!(block.weld_count(), 0);

    let cylinder = free_cylinder_candidate(
        point,
        direction,
        CylinderDimensions::default(),
        grid,
        bounds,
    );
    let pipe = [PipeRunPiece {
        spec: PartSpec::Cylinder(cylinder.spec),
        inlet: FaceKind::NegativeY,
        outlet: FaceKind::PositiveY,
    }];
    let staged = stage_pipe_run_in_bounds(&graph, &pipe, PipeRunAttachment::Free, bounds).unwrap();
    assert_eq!(staged.part_count(), 1);
    assert_eq!(staged.weld_count(), 0);
}

#[test]
fn multiple_dimension_links_with_distinct_ids_can_coexist() {
    let bounds = PlacementBounds::GarageBuild;
    let grid = PlacementGrid::Centimetres25;
    let rotation = GridRotation::default();
    let base = free_cuboid_candidate(
        Vec3::new(0.0, 7.5, 0.0),
        Vec3::NEG_Z,
        [2, 1, 1],
        rotation,
        grid,
        bounds,
    );
    let graph =
        stage_block_batch_in_bounds(&ConstructionGraph::new(), base, &[base.spec], bounds).unwrap();
    let first = free_cuboid_candidate(
        Vec3::new(-0.5, 7.5, 0.0),
        Vec3::NEG_Z,
        mechanic_core::DimensionLinkSpec::GRID_UNITS,
        rotation,
        grid,
        bounds,
    );
    let graph = stage_dimension_link_in_bounds(&graph, first, DimensionLinkId(11), bounds).unwrap();
    let second = free_cuboid_candidate(
        Vec3::new(0.5, 7.5, 0.0),
        Vec3::NEG_Z,
        mechanic_core::DimensionLinkSpec::GRID_UNITS,
        rotation,
        grid,
        bounds,
    );
    let graph =
        stage_dimension_link_in_bounds(&graph, second, DimensionLinkId(12), bounds).unwrap();

    let mut ids = graph
        .parts()
        .filter_map(|(part, _)| graph.dimension_link_id(part))
        .collect::<Vec<_>>();
    ids.sort_unstable_by_key(|id| id.0);
    assert_eq!(ids, vec![DimensionLinkId(11), DimensionLinkId(12)]);
    assert_eq!(graph.part_count(), 3);
    assert_eq!(graph.weld_count(), 2);
    assert_eq!(graph.compile().unwrap().compounds.len(), 1);
}

#[test]
fn free_parts_auto_weld_on_contact_and_reject_overlap_or_bounds_escape() {
    let bounds = PlacementBounds::GarageBuild;
    let first = free_cuboid_candidate(
        Vec3::new(0.0, 6.0, 0.0),
        Vec3::NEG_Z,
        [1; 3],
        GridRotation::default(),
        PlacementGrid::Centimetres25,
        bounds,
    );
    let graph =
        stage_block_batch_in_bounds(&ConstructionGraph::new(), first, &[first.spec], bounds)
            .unwrap();
    assert!(matches!(
        stage_block_batch_in_bounds(&graph, first, &[first.spec], bounds),
        Err(PlacementError::OverlapsPart(_))
    ));

    let touching = free_cuboid_candidate(
        first.spec.pose.translation() + Vec3::X * BLOCK_SIZE_METERS,
        Vec3::NEG_Z,
        [1; 3],
        GridRotation::default(),
        PlacementGrid::Centimetres25,
        bounds,
    );
    let welded = stage_block_batch_in_bounds(&graph, touching, &[touching.spec], bounds).unwrap();
    assert_eq!(welded.weld_count(), 1);

    let outside = free_cuboid_candidate(
        Vec3::new(super::GROUND_HALF_SIZE, 6.0, 0.0),
        Vec3::NEG_Z,
        [1; 3],
        GridRotation::default(),
        PlacementGrid::Centimetres25,
        bounds,
    );
    assert_eq!(
        validate_block_batch_in_bounds(&ConstructionGraph::new(), outside, &[outside.spec], bounds,),
        Err(PlacementError::OutsidePlatform)
    );
}

#[test]
fn world_raycast_uses_the_supplied_terrain_surface() {
    let graph = ConstructionGraph::new();
    let terrain_hit = SurfaceHit {
        distance: 6.65,
        point: Vec3::new(24.0, 13.35, -31.0),
        face: FaceRef::ground(),
    };

    let hit = raycast_construction_with_ground(
        &graph,
        Vec3::new(24.0, 20.0, -31.0),
        Vec3::NEG_Y,
        Some(terrain_hit),
    )
    .expect("terrain is the world build surface");

    assert!(matches!(hit.face.owner, FaceOwner::Ground));
    assert!(hit.point.abs_diff_eq(terrain_hit.point, 1.0e-6));
}

#[test]
fn cylinder_slice_platform_bounds_ignore_the_omitted_sector() {
    let graph = ConstructionGraph::new();
    let pose = BuildPose::new(IVec3::new(-40, 2, 0), GridRotation::default());
    let slice = CylinderDimensions::new(1.0, 0.0, 1.0)
        .unwrap()
        .with_sweep_angle_degrees(90)
        .unwrap();
    assert!(validate_part(&graph, PartSpec::Cylinder(CylinderSpec::new(slice, pose))).is_ok());

    let full = CylinderDimensions::new(1.0, 0.0, 1.0).unwrap();
    assert!(matches!(
        validate_part(&graph, PartSpec::Cylinder(CylinderSpec::new(full, pose))),
        Err(PlacementError::OutsidePlatform)
    ));
}

#[test]
fn single_block_automatically_welds_to_touching_block() {
    let mut graph = ConstructionGraph::new();
    spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
    let hit = raycast_construction(&graph, Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y)
        .expect("support block is under ray");
    let candidate = candidate_from_hit(&graph, hit);

    let graph = stage_cuboid(&graph, candidate).unwrap();

    assert_eq!(graph.part_count(), 2);
    assert_eq!(graph.weld_count(), 1);
    assert_eq!(graph.compile().unwrap().compounds.len(), 1);
}

#[test]
fn single_block_placed_on_ground_is_automatically_welded() {
    let graph = ConstructionGraph::new();
    let candidate = candidate_from_hit(
        &graph,
        SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        },
    );

    let graph = stage_cuboid(&graph, candidate).unwrap();

    assert_eq!(graph.part_count(), 1);
    assert_eq!(graph.weld_count(), 1);
    assert!(graph.compile().unwrap().compounds[0].is_static);
}

#[test]
fn dragged_sheet_is_face_connected_and_welded() {
    let graph = ConstructionGraph::new();
    let start = candidate_from_hit(
        &graph,
        SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        },
    );
    let endpoint = start.spec.pose.translation_half_units() + IVec3::new(4, 0, 2);
    let specs = block_sheet_specs(start.spec, endpoint, PlacementPlane::Xz).unwrap();

    let graph = stage_block_batch(&graph, start, &specs).unwrap();

    assert_eq!(graph.part_count(), 6);
    assert_eq!(graph.weld_count(), 13);
    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.compounds.len(), 1);
    assert!(compiled.compounds[0].is_static);
}

#[test]
fn fast_volume_path_keeps_4096_blocks_individual_with_exact_welds() {
    let graph = ConstructionGraph::new();
    let start = ground_volume_candidate(IVec3::new(-3_150, 50, -3_150));
    let volume = BlockVolume::new(start.spec, IVec3::new(63, 0, 63)).unwrap();
    let index = PlacementSnapIndex::default();

    let started = Instant::now();
    let placed = stage_block_volume_in_bounds(
        &graph,
        &index,
        start,
        volume,
        None,
        Some(FaceOwner::Ground),
        PlacementBounds::Garage,
        17,
    )
    .unwrap();
    let elapsed = started.elapsed();

    assert_eq!(placed.new_parts.len(), 4_096);
    assert_eq!(placed.graph.part_count(), 4_096);
    assert_eq!(placed.weld_count, 4_032 + 4_032 + 4_096);
    assert_eq!(placed.graph.weld_count(), placed.weld_count);
    assert_eq!(placed.publication_generation, 17);
    assert_eq!(
        placed.graph.part(placed.new_parts[0]),
        Some(&start.spec.into())
    );
    assert!(
        elapsed.as_secs_f64() < 1.0 / 60.0,
        "bulk staging regressed to {elapsed:?}"
    );
}

#[test]
fn fast_volume_path_places_a_16_cubed_solid() {
    let graph = ConstructionGraph::new();
    let start = ground_volume_candidate(IVec3::new(-750, 50, -750));
    let placed = stage_block_volume_in_bounds(
        &graph,
        &PlacementSnapIndex::default(),
        start,
        BlockVolume::new(start.spec, IVec3::splat(15)).unwrap(),
        None,
        Some(FaceOwner::Ground),
        PlacementBounds::Garage,
        3,
    )
    .unwrap();

    assert_eq!(placed.graph.part_count(), 4_096);
    assert_eq!(placed.weld_count, 3 * 15 * 16 * 16 + 16 * 16);
}

#[test]
fn fast_volume_path_welds_only_the_adjacent_boundary() {
    let empty = ConstructionGraph::new();
    let bottom_start = ground_volume_candidate(IVec3::new(-3_150, 50, -3_150));
    let sheet = BlockVolume::new(bottom_start.spec, IVec3::new(63, 0, 63)).unwrap();
    let placed = stage_block_volume_in_bounds(
        &empty,
        &PlacementSnapIndex::default(),
        bottom_start,
        sheet,
        None,
        Some(FaceOwner::Ground),
        PlacementBounds::Garage,
        1,
    )
    .unwrap();
    let mut index = PlacementSnapIndex::default();
    index.rebuild(&placed.graph);
    let top_start = PlacementCandidate {
        spec: CuboidSpec::new(
            [1; 3],
            BuildPose::from_position_ticks(
                IVec3::new(-3_150, 150, -3_150),
                GridRotation::default(),
            ),
        )
        .unwrap(),
        attached_face: FaceKind::NegativeY,
        anchor: None,
        support: PlacementSupport::Free,
    };

    let started = Instant::now();
    let top = stage_block_volume_in_bounds(
        &placed.graph,
        &index,
        top_start,
        BlockVolume::new(top_start.spec, IVec3::new(63, 0, 63)).unwrap(),
        None,
        None,
        PlacementBounds::Garage,
        2,
    )
    .unwrap();
    let elapsed = started.elapsed();

    assert_eq!(top.new_parts.len(), 4_096);
    assert_eq!(top.weld_count, 4_032 + 4_032 + 4_096);
    assert_eq!(
        top.graph.weld_count(),
        placed.graph.weld_count() + top.weld_count
    );
    assert!(
        elapsed.as_secs_f64() < 1.0 / 60.0,
        "adjacent bulk staging regressed to {elapsed:?}"
    );
}

#[test]
fn negative_volume_span_preserves_start_material_and_appearance() {
    let appearance = mechanic_core::MaterialAppearance::new(
        mechanic_core::MaterialColor::Dye(
            mechanic_core::MaterialDye::new([12, 34, 56], 1.0).unwrap(),
        ),
        mechanic_core::MaterialFinish::Painted,
    );
    let start = CuboidSpec::new(
        [1; 3],
        BuildPose::from_position_ticks(IVec3::new(200, 250, 300), GridRotation::default()),
    )
    .unwrap()
    .with_material(ConstructionMaterial::Copper)
    .with_appearance(appearance);
    let volume = BlockVolume::new(start, IVec3::new(-3, -3, -3)).unwrap();
    let specs = volume.specs().collect::<Vec<_>>();

    assert_eq!(specs.len(), 64);
    assert_eq!(specs[0], start);
    assert!(specs.iter().all(|spec| {
        spec.material == ConstructionMaterial::Copper && spec.appearance == appearance
    }));
    assert_eq!(volume.bounds().0, Vec3::new(-0.375, -0.25, -0.125));
}

#[test]
fn a_box_drag_places_a_solid_cuboid() {
    let start = CuboidSpec::new([1; 3], BuildPose::default()).unwrap();
    let specs = block_box_specs(start, IVec3::new(2, 1, 3)).unwrap();
    assert_eq!(
        specs.len(),
        3 * 2 * 4,
        "span counts blocks beyond the start"
    );

    // Every cell of the cuboid is filled exactly once: solid, no gaps and
    // no duplicates, which is what a region will later be able to claim.
    let mut centres = specs
        .iter()
        .map(|spec| spec.pose.translation_half_units().to_array())
        .collect::<Vec<_>>();
    centres.sort_unstable();
    let unique = {
        let mut copy = centres.clone();
        copy.dedup();
        copy
    };
    assert_eq!(centres, unique, "a box drag must not stack blocks");
}

#[test]
fn a_zero_span_box_drag_is_the_single_starting_block() {
    let start = CuboidSpec::new([1; 3], BuildPose::default()).unwrap();
    let specs = block_box_specs(start, IVec3::ZERO).unwrap();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].pose.translation_half_units(), IVec3::ZERO);
}

#[test]
fn box_drag_bounds_cover_the_full_selected_blocks() {
    let start = CuboidSpec::new([1; 3], BuildPose::default()).unwrap();

    assert_eq!(
        block_box_bounds(start, IVec3::new(2, 0, -1)),
        (
            Vec3::new(-0.125, -0.125, -0.375),
            Vec3::new(0.625, 0.125, 0.125),
        )
    );
}

#[test]
fn a_box_drag_keeps_the_selected_material_and_respects_the_cap() {
    let start = CuboidSpec::new([1; 3], BuildPose::default())
        .unwrap()
        .with_material(ConstructionMaterial::Wood);
    let specs = block_box_specs(start, IVec3::new(3, 3, 3)).unwrap();
    assert!(
        specs
            .iter()
            .all(|spec| spec.material == ConstructionMaterial::Wood)
    );
    assert!(matches!(
        block_box_specs(start, IVec3::new(31, 31, 31)),
        Err(PlacementError::TooManyBlocks { .. })
    ));
}

#[test]
fn rotating_the_plane_keeps_the_extent_and_extends_the_third_axis() {
    // This is what makes a big cuboid easy: drag a rectangle, press Rotate, and
    // carry on into the axis the first plane could not reach.
    let start = CuboidSpec::new([1; 3], BuildPose::default()).unwrap();
    let down = Vec3::NEG_Y;
    let press_origin = Vec3::new(0.0, 4.0, 0.0);

    // A rectangle in XZ.
    let flat = block_span_from_rays(
        start,
        PlacementPlane::Xz,
        IVec3::ZERO,
        press_origin,
        down,
        press_origin + Vec3::new(BLOCK_SIZE_METERS * 3.0, 0.0, BLOCK_SIZE_METERS * 2.0),
        down,
    )
    .expect("the XZ plane is reachable from above");
    assert_eq!(flat, IVec3::new(3, 0, 2));

    // Rotate moves into a plane containing Y; the frozen span carries over and
    // only the new plane's axes move.
    let horizontal = Vec3::new(1.0, 0.0, 0.0);
    let side_origin = Vec3::new(-4.0, 0.0, 0.0);
    let boxed = block_span_from_rays(
        start,
        PlacementPlane::Yz,
        flat,
        side_origin,
        horizontal,
        side_origin + Vec3::new(0.0, BLOCK_SIZE_METERS * 4.0, 0.0),
        horizontal,
    )
    .expect("the YZ plane is reachable from the side");
    assert_eq!(
        boxed.x, 3,
        "the axis the new plane does not own must keep its extent"
    );
    assert_eq!(boxed.y, 4, "the new axis grows from the rotation onward");
}

#[test]
fn dragged_sheet_keeps_one_selected_material_for_every_block() {
    let start = CuboidSpec::new([1; 3], BuildPose::default())
        .unwrap()
        .with_material(ConstructionMaterial::Wood);
    let specs = block_sheet_specs(start, IVec3::new(4, 0, 4), PlacementPlane::Xz).unwrap();
    assert!(
        specs
            .iter()
            .all(|spec| spec.material == ConstructionMaterial::Wood)
    );
}

#[test]
fn invalid_drag_batch_preserves_graph() {
    let graph = ConstructionGraph::new();
    let start = candidate_from_hit(
        &graph,
        SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        },
    );
    let endpoint = start.spec.pose.translation_half_units() + IVec3::new(96, 0, 0);
    let specs = block_sheet_specs(start.spec, endpoint, PlacementPlane::Xz).unwrap();

    assert!(matches!(
        stage_block_batch(&graph, start, &specs),
        Err(PlacementError::OutsidePlatform)
    ));
    assert_eq!(graph.part_count(), 0);
    assert_eq!(graph.weld_count(), 0);
}

#[test]
fn drag_plane_projection_and_cycle_are_deterministic() {
    let graph = ConstructionGraph::new();
    let start = candidate_from_hit(
        &graph,
        SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        },
    );
    let point = raycast_placement_plane_point(
        Vec3::new(2.0, 5.0, 3.0),
        Vec3::NEG_Y,
        start.spec,
        PlacementPlane::Xz,
    )
    .unwrap();

    // The plane runs through the dragged block's centre, and the point is
    // left unsnapped for the span arithmetic to quantize.
    assert!(point.abs_diff_eq(Vec3::new(2.0, BLOCK_SIZE_METERS * 0.5, 3.0), 1.0e-6));
    assert_eq!(PlacementPlane::Xz.cycle(), PlacementPlane::Xy);
    assert_eq!(PlacementPlane::Xy.cycle(), PlacementPlane::Yz);
    assert_eq!(PlacementPlane::Yz.cycle(), PlacementPlane::Xz);
}

#[test]
fn weld_selects_two_objects_without_spawning_a_part() {
    let mut graph = ConstructionGraph::new();
    let left = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
    let right = spawn_cube(&mut graph, IVec3::new(4, 2, 0), 4);
    begin_weld(&mut graph, FaceRef::part(left, FaceKind::PositiveY)).unwrap();
    assert!(matches!(graph.pending(), Some(PendingOperation::Weld(_))));

    let graph = stage_weld_objects(&graph, FaceOwner::Part(left), FaceOwner::Part(right)).unwrap();

    assert_eq!(graph.part_count(), 2);
    assert_eq!(graph.weld_count(), 1);
    assert!(graph.pending().is_none());
    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.compounds.len(), 1);
    assert_eq!(compiled.compounds[0].source_parts.len(), 2);
}

#[test]
fn weld_to_ground_resolves_contact_across_the_selected_rigid_body() {
    let mut graph = ConstructionGraph::new();
    let parts = [IVec3::new(0, 1, 0), IVec3::new(0, 3, 0)].map(|center| {
        let spec = CuboidSpec::new(
            [1; 3],
            BuildPose::from_half_grid(center, GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        part
    });
    let [bottom, top] = parts;
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(bottom, FaceKind::PositiveY),
            second: FaceRef::part(top, FaceKind::NegativeY),
        }))
        .unwrap();

    let grounded = stage_weld_objects(&graph, FaceOwner::Part(top), FaceOwner::Ground).unwrap();

    assert_eq!(grounded.weld_count(), 2);
    let compiled = grounded.compile().unwrap();
    assert_eq!(compiled.compounds.len(), 1);
    assert!(compiled.compounds[0].is_static);
}

#[test]
fn weld_resolves_contact_across_both_rigid_bodies() {
    let mut graph = ConstructionGraph::new();
    let left = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
    let middle = spawn_cube(&mut graph, IVec3::new(4, 2, 0), 4);
    let right = spawn_cube(&mut graph, IVec3::new(8, 2, 0), 4);
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(left, FaceKind::PositiveX),
            second: FaceRef::part(middle, FaceKind::NegativeX),
        }))
        .unwrap();

    // `left` does not touch `right`, but the body it belongs to does, and
    // the body is what the tool highlights and claims to weld.
    let staged = stage_weld_objects(&graph, FaceOwner::Part(left), FaceOwner::Part(right)).unwrap();

    assert_eq!(staged.weld_count(), 2);
    let compiled = staged.compile().unwrap();
    assert_eq!(compiled.compounds.len(), 1);
    assert_eq!(compiled.compounds[0].source_parts.len(), 3);
}

#[test]
fn weld_refuses_two_parts_of_one_rigid_body() {
    let mut graph = ConstructionGraph::new();
    let left = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
    let right = spawn_cube(&mut graph, IVec3::new(4, 2, 0), 4);
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(left, FaceKind::PositiveX),
            second: FaceRef::part(right, FaceKind::NegativeX),
        }))
        .unwrap();

    assert!(matches!(
        stage_weld_objects(&graph, FaceOwner::Part(left), FaceOwner::Part(right)),
        Err(PlacementError::SameObject)
    ));
    assert_eq!(graph.weld_count(), 1);
}

/// Base spanning two cells with two one-cell blocks bearing-mounted on top,
/// side by side and touching one another.
fn twin_bearing_rig() -> (ConstructionGraph, PartId, PartId, PartId) {
    let mut graph = ConstructionGraph::new();
    let base = spawn_cube(&mut graph, IVec3::ZERO, 2);
    let mounted = [-1, 1].map(|x| {
        let spec = CuboidSpec::new(
            [1; 3],
            BuildPose::from_half_grid(IVec3::new(x, 3, -1), GridRotation::default()),
        )
        .unwrap();
        let Ok(BuildOutcome::Spawned(part)) = graph.apply(BuildCommand::Spawn(spec)) else {
            panic!("block must spawn");
        };
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(base, FaceKind::PositiveY),
                FaceRef::part(part, FaceKind::NegativeY),
                Vec3::new(
                    f32::from(i8::try_from(x).expect("small")) * 0.125,
                    0.25,
                    -0.125,
                ),
                Vec3::Y,
            )))
            .unwrap();
        part
    });
    let [first, second] = mounted;
    (graph, base, first, second)
}

#[test]
fn welding_across_a_bearing_is_allowed_and_reports_the_lockup() {
    let (graph, base, mounted, _) = twin_bearing_rig();
    assert!(locked_bearings(&graph).is_empty());

    let staged =
        stage_weld_objects(&graph, FaceOwner::Part(base), FaceOwner::Part(mounted)).unwrap();

    assert_eq!(newly_locked_bearings(&graph, &staged), 1);
    // Allowed, so it has to survive compilation rather than fail later.
    staged.compile().unwrap();
}

#[test]
fn a_loop_that_leaves_every_bearing_free_reports_no_lockup() {
    let (graph, _, first, second) = twin_bearing_rig();

    // Both blocks turn on their own bearing; joining them to each other
    // closes a loop through the base without locking either joint.
    let staged =
        stage_weld_objects(&graph, FaceOwner::Part(first), FaceOwner::Part(second)).unwrap();

    assert_eq!(staged.weld_count(), 1);
    assert_eq!(newly_locked_bearings(&graph, &staged), 0);
    staged.compile().unwrap();
}

#[test]
fn weld_rejects_same_or_separated_objects_without_mutation() {
    let mut graph = ConstructionGraph::new();
    let left = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
    let far = spawn_cube(&mut graph, IVec3::new(8, 2, 0), 4);
    assert!(matches!(
        stage_weld_objects(&graph, FaceOwner::Part(left), FaceOwner::Part(left)),
        Err(PlacementError::SameObject)
    ));
    assert!(matches!(
        stage_weld_objects(&graph, FaceOwner::Part(left), FaceOwner::Part(far)),
        Err(PlacementError::ObjectsDoNotTouch)
    ));
    assert_eq!(graph.weld_count(), 0);
}

#[test]
fn bearing_anchor_snaps_without_mutating_the_graph() {
    let mut graph = ConstructionGraph::new();
    let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
    let hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::new(0.18, 1.0, -0.18),
        face: FaceRef::part(base, FaceKind::PositiveY),
    };
    let anchor = bearing_anchor_from_hit(&graph, hit).unwrap();
    assert_eq!(anchor, Vec3::new(0.25, 1.0, -0.25));

    assert_eq!(graph.part_count(), 1);
    assert_eq!(graph.bearing_count(), 0);
    assert!(graph.pending().is_none());
}

#[test]
fn bearing_anchor_uses_the_same_grid_phase_as_blocks_and_cylinders() {
    let graph = ConstructionGraph::new();
    let block = candidate_from_hit(
        &graph,
        SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        },
    );
    let graph = stage_cuboid(&graph, block).unwrap();
    let part = graph.parts().next().unwrap().0;
    let source = FaceRef::part(part, FaceKind::PositiveX);
    let face = face_geometry_from_ref(source, Some(&graph));
    let hit = SurfaceHit {
        distance: 1.0,
        point: face.center + Vec3::new(0.0, 0.02, 0.10),
        face: source,
    };

    for grid in [
        PlacementGrid::Centimetres25,
        PlacementGrid::Centimetres5,
        PlacementGrid::Centimetres1,
    ] {
        let anchor =
            super::bearing_anchor_from_hit_with_grid(&graph, hit, grid, PlacementBounds::Garage)
                .unwrap();
        let block = oriented_cuboid_candidate_from_hit_with_grid(
            &graph,
            hit,
            [1; 3],
            GridRotation::default(),
            grid,
            PlacementBounds::Garage,
        );
        let cylinder = super::cylinder_candidate_from_hit_with_grid(
            &graph,
            hit,
            CylinderDimensions::new(0.5, 0.0, 0.25).unwrap(),
            grid,
            PlacementBounds::Garage,
        )
        .unwrap();

        for axis in [1, 2] {
            assert!((anchor[axis] - block.spec.pose.translation()[axis]).abs() < 1.0e-6);
            assert!((anchor[axis] - cylinder.spec.pose.translation()[axis]).abs() < 1.0e-6);
        }
    }
}

#[test]
fn bearing_second_click_attaches_a_cuboid_without_collider_geometry_for_connector() {
    let mut graph = ConstructionGraph::new();
    let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
    let source = FaceRef::part(base, FaceKind::PositiveY);
    let hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::new(0.0, 1.0, 0.0),
        face: source,
    };
    let anchor = bearing_anchor_from_hit(&graph, hit).unwrap();
    let candidate = bearing_attachment_candidate(&graph, source, anchor);

    let dimensions = BearingDimensions::new(0.75, 0.25).unwrap();
    let graph = stage_bearing_attachment(&graph, candidate, source, anchor, dimensions).unwrap();

    assert_eq!(graph.part_count(), 2);
    assert_eq!(graph.bearing_count(), 1);
    assert_eq!(graph.bearings().next().unwrap().1.shared_anchor, anchor);
    assert_eq!(graph.bearings().next().unwrap().1.dimensions, dimensions);
    assert!(graph.pending().is_none());
}

#[test]
fn bearing_attachment_centres_a_cylinder_before_it_is_dragged() {
    let mut graph = ConstructionGraph::new();
    let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
    let source = FaceRef::part(base, FaceKind::PositiveY);
    let anchor = Vec3::new(0.25, 1.0, -0.25);
    let candidate = cylinder_candidate_from_hit(
        &graph,
        SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.50, 1.0, 0.25),
            face: source,
        },
        CylinderDimensions::default(),
    )
    .unwrap();

    let centered = center_cylinder_candidate_on_bearing(candidate, anchor);
    let face = super::faces::cylinder_face_geometry(centered.spec, centered.attached_face).unwrap();

    assert!(face.center.abs_diff_eq(anchor, 1.0e-5));
    assert_eq!(centered.anchor, Some(anchor));
    assert_eq!(centered.support, PlacementSupport::Bearing);
}

#[test]
fn oversized_bearing_attaches_to_any_block_face_overlapped_by_its_ring() {
    let mut graph = ConstructionGraph::new();
    let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
    let source = FaceRef::part(base, FaceKind::PositiveY);
    let anchor = Vec3::Y;
    let dimensions = BearingDimensions::new(0.80, 0.10).unwrap();
    let candidate = candidate_from_hit(
        &graph,
        SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.36, 1.0, 0.0),
            face: source,
        },
    );

    assert!(
        !super::faces::face_geometry(candidate.spec, candidate.attached_face)
            .center
            .abs_diff_eq(anchor, 1.0e-5)
    );
    assert!(bearing_overlaps_candidate(
        &graph, source, anchor, dimensions, candidate,
    ));
    let attached = stage_bearing_attachment(&graph, candidate, source, anchor, dimensions).unwrap();

    assert_eq!(attached.bearing_count(), 1);
    assert_eq!(attached.weld_count(), 0);
    assert_eq!(attached.bearings().next().unwrap().1.dimensions, dimensions);
}

#[test]
fn bearing_overhang_claims_a_block_placed_on_an_adjacent_support_face() {
    let mut graph = ConstructionGraph::new();
    let source_part = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
    let adjacent_part = spawn_cube(&mut graph, IVec3::new(4, 2, 0), 4);
    let source = FaceRef::part(source_part, FaceKind::PositiveY);
    let adjacent_face = FaceRef::part(adjacent_part, FaceKind::PositiveY);
    let anchor = Vec3::Y;
    let dimensions = BearingDimensions::new(2.40, 0.10).unwrap();
    let candidate = candidate_from_hit(
        &graph,
        SurfaceHit {
            distance: 1.0,
            point: Vec3::new(1.0, 1.0, 0.0),
            face: adjacent_face,
        },
    );

    assert!(bearing_overlaps_candidate(
        &graph, source, anchor, dimensions, candidate,
    ));
    let attached = stage_bearing_attachment(&graph, candidate, source, anchor, dimensions).unwrap();

    assert_eq!(attached.bearing_count(), 1);
    assert_eq!(attached.weld_count(), 0);
    assert_eq!(attached.compile().unwrap().compounds.len(), 3);
}

#[test]
fn block_face_entirely_inside_bearing_hole_is_not_covered() {
    let mut graph = ConstructionGraph::new();
    let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
    let source = FaceRef::part(base, FaceKind::PositiveY);
    let candidate = bearing_attachment_candidate(&graph, source, Vec3::Y);

    assert!(!bearing_overlaps_candidate(
        &graph,
        source,
        Vec3::Y,
        BearingDimensions::new(1.0, 0.50).unwrap(),
        candidate,
    ));
}

#[test]
fn large_hollow_bearing_uses_a_ring_block_instead_of_the_center_block() {
    let mut graph = ConstructionGraph::new();
    let mut center = None;
    for x in -1..=1 {
        for z in -1..=1 {
            let spec = CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_half_grid(IVec3::new(x * 2, 1, z * 2), GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            if x == 0 && z == 0 {
                center = Some(part);
            }
        }
    }
    let center = center.unwrap();
    let selected = FaceRef::part(center, FaceKind::PositiveY);
    let dimensions = BearingDimensions::new(0.75, 0.40).unwrap();

    let support =
        bearing_support_face(&graph, selected, Vec3::new(0.0, 0.25, 0.0), dimensions).unwrap();

    assert_ne!(support.owner, FaceOwner::Part(center));
    assert!(bearing_ring_overlaps_face(
        Vec3::new(0.0, 0.25, 0.0),
        dimensions,
        &face_geometry_from_ref(support, Some(&graph)),
    ));
}

#[test]
fn placement_from_bearing_body_welds_only_to_the_clicked_rigid_group() {
    let mut graph = ConstructionGraph::new();
    let base = spawn_cube(&mut graph, IVec3::ZERO, 1);
    let attached = spawn_cube(&mut graph, IVec3::new(0, 1, 0), 1);
    let sibling = spawn_cube(&mut graph, IVec3::new(-1, 1, 0), 1);
    let neighbour = spawn_cube(&mut graph, IVec3::new(1, 2, 0), 1);
    let dimensions = BearingDimensions::new(0.80, 0.10).unwrap();
    for target in [attached, sibling] {
        graph
            .apply(BuildCommand::AddBearing(
                BearingSpec::new(
                    FaceRef::part(base, FaceKind::PositiveY),
                    FaceRef::part(target, FaceKind::NegativeY),
                    Vec3::new(0.0, 0.125, 0.0),
                    Vec3::Y,
                )
                .with_dimensions(dimensions),
            ))
            .unwrap();
    }
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec {
            first: attached,
            second: sibling,
        }))
        .unwrap();
    let source = FaceRef::part(attached, FaceKind::PositiveY);
    let source_face = face_geometry_from_ref(source, Some(&graph));
    let candidate = candidate_from_hit(
        &graph,
        SurfaceHit {
            distance: 0.0,
            point: source_face.center,
            face: source,
        },
    );

    let staged =
        stage_block_batch_from_source(&graph, candidate, &[candidate.spec], source.owner).unwrap();
    let placed = staged
        .parts()
        .find_map(|(part, _)| graph.part(part).is_none().then_some(part))
        .unwrap();
    let attached_group = rigid_body_parts(&staged, attached);

    assert!(attached_group.contains(&placed));
    assert!(attached_group.contains(&sibling));
    assert!(!attached_group.contains(&neighbour));
    assert!(!attached_group.contains(&base));
    assert_eq!(staged.bearing_count(), 2);
    assert_eq!(staged.compile().unwrap().bearings.len(), 1);
}

#[test]
fn one_bearing_groups_multiple_direct_attachments_into_one_rotor() {
    let mut graph = ConstructionGraph::new();
    let support = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
    let source = FaceRef::part(support, FaceKind::PositiveY);
    let dimensions = BearingDimensions::new(0.80, 0.10).unwrap();
    let candidates = [0.0, 0.25].map(|x| {
        candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 0.0,
                point: Vec3::new(x, 1.0, 0.0),
                face: source,
            },
        )
    });

    for candidate in candidates {
        let rigid_targets = graph
            .bearings()
            .filter_map(|(_, bearing)| match bearing.target.owner {
                FaceOwner::Part(part) => Some(part),
                FaceOwner::Ground => None,
            })
            .collect::<Vec<_>>();
        graph = stage_bearing_block_batch(
            &graph,
            candidate,
            &[candidate.spec],
            source,
            Vec3::Y,
            dimensions,
            &rigid_targets,
        )
        .unwrap();
    }

    let targets = graph
        .parts()
        .filter_map(|(part, _)| (part != support).then_some(part))
        .collect::<Vec<_>>();
    assert_eq!(targets.len(), 2);
    assert_eq!(graph.bearing_count(), 2);
    assert_eq!(graph.weld_count(), 0);
    assert_eq!(graph.rigid_link_count(), 1);
    assert_eq!(rigid_body_parts(&graph, targets[0]), targets);
    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.compounds.len(), 2);
    assert_eq!(compiled.bearings.len(), 1);
}

#[test]
fn bearing_drag_attaches_one_internally_welded_sheet() {
    let mut graph = ConstructionGraph::new();
    let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
    let source = FaceRef::part(base, FaceKind::PositiveY);
    let anchor = Vec3::Y;
    let candidate = bearing_attachment_candidate(&graph, source, anchor);
    let mut endpoint = candidate.spec.pose.translation_half_units();
    endpoint.x += 2;
    let specs = block_sheet_specs(candidate.spec, endpoint, PlacementPlane::Xz).unwrap();

    let graph = stage_bearing_block_batch(
        &graph,
        candidate,
        &specs,
        source,
        anchor,
        BearingDimensions::default(),
        &[],
    )
    .unwrap();

    assert_eq!(graph.part_count(), 3);
    assert_eq!(graph.bearing_count(), 1);
    assert_eq!(graph.weld_count(), 1);
    assert_eq!(graph.compile().unwrap().compounds.len(), 2);
}

#[test]
fn bearing_centres_and_attaches_on_a_quarter_metre_block() {
    let graph = ConstructionGraph::new();
    let block = candidate_from_hit(
        &graph,
        SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        },
    );
    let graph = stage_cuboid(&graph, block).unwrap();
    let base = graph.parts().next().unwrap().0;
    let source = FaceRef::part(base, FaceKind::PositiveY);
    let hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::new(0.1, 0.25, -0.1),
        face: source,
    };

    let anchor = bearing_anchor_from_hit(&graph, hit).unwrap();
    assert!(anchor.abs_diff_eq(Vec3::new(0.0, 0.25, 0.0), 1.0e-6));
    let candidate = bearing_attachment_candidate(&graph, source, anchor);
    let graph = stage_bearing_attachment(
        &graph,
        candidate,
        source,
        anchor,
        BearingDimensions::default(),
    )
    .unwrap();

    assert_eq!(graph.part_count(), 2);
    assert_eq!(graph.bearing_count(), 1);
}

#[test]
fn bearing_rejects_ground_but_allows_visual_overhang_at_face_edges() {
    let mut graph = ConstructionGraph::new();
    let ground_hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::ZERO,
        face: FaceRef::ground(),
    };
    assert!(matches!(
        bearing_anchor_from_hit(&graph, ground_hit),
        Err(PlacementError::BearingOnGround)
    ));

    let part = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 2);
    let edge_hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::new(0.25, 0.5, 0.25),
        face: FaceRef::part(part, FaceKind::PositiveY),
    };
    let anchor = bearing_anchor_from_hit(&graph, edge_hit).unwrap();
    assert_eq!(anchor, Vec3::new(0.25, 0.75, 0.25));
    let candidate = bearing_attachment_candidate(&graph, edge_hit.face, anchor);
    let dimensions = BearingDimensions::new(8.0, 0.10).unwrap();
    let attached =
        stage_bearing_attachment(&graph, candidate, edge_hit.face, anchor, dimensions).unwrap();
    assert_eq!(attached.bearings().next().unwrap().1.dimensions, dimensions);
    assert_eq!(graph.part_count(), 1);
}

#[test]
fn rejected_overlap_does_not_mutate_source_graph() {
    let graph = ConstructionGraph::new();
    let base_hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::ZERO,
        face: FaceRef::ground(),
    };
    let candidate = candidate_from_hit(&graph, base_hit);
    let graph = stage_cuboid(&graph, candidate).unwrap();
    assert!(matches!(
        stage_cuboid(&graph, candidate),
        Err(PlacementError::OverlapsPart(_))
    ));
    assert_eq!(graph.part_count(), 1);
}

#[test]
fn remove_cascades_through_incident_bearing() {
    let mut graph = ConstructionGraph::new();
    let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
    let source = FaceRef::part(base, FaceKind::PositiveY);
    let anchor = Vec3::new(0.0, 1.0, 0.0);
    let candidate = bearing_attachment_candidate(&graph, source, anchor);
    let graph = stage_bearing_attachment(
        &graph,
        candidate,
        source,
        anchor,
        BearingDimensions::default(),
    )
    .unwrap();
    let top = raycast_construction(&graph, Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y).unwrap();
    let upper = match top.face.owner {
        FaceOwner::Part(part) => part,
        FaceOwner::Ground => panic!("top ray must hit attached part"),
    };
    let mut graph = graph;
    graph.apply(BuildCommand::Remove(upper)).unwrap();
    assert_eq!(graph.part_count(), 1);
    assert_eq!(graph.bearing_count(), 0);
}

#[test]
fn a_ray_meets_a_shaped_face_where_the_surface_actually_is() {
    // A wedge's sloped face sits well inside the block's old box, so a hit
    // that still lands on the box would put the cursor in mid-air.
    let spec = CuboidSpec::new(
        [1, 1, 1],
        BuildPose::from_half_grid(IVec3::new(1, 1, 1), GridRotation::default()),
    )
    .unwrap();
    let mut graph = ConstructionGraph::new();
    graph.apply(BuildCommand::Spawn(spec)).unwrap();
    let region = mechanic_core::ShapeRegion::new(
        IVec3::ZERO,
        IVec3::ONE,
        mechanic_core::ConstructionMaterial::Steel,
    )
    .unwrap();
    let BuildOutcome::RegionAdded(id) = graph.apply(BuildCommand::AddRegion(region)).unwrap()
    else {
        panic!("wrong outcome")
    };
    let cell = i16::try_from(mechanic_core::POSITION_TICKS_PER_GRID_UNIT).unwrap();
    graph
        .apply(BuildCommand::SetRegionVertices {
            region: id,
            vertices: vec![([0, 1, 1], [0, -cell, 0]), ([1, 1, 1], [0, -cell, 0])],
        })
        .expect("collapsing an edge makes a wedge");

    // Straight down onto the sloped half of the top face.
    let origin = Vec3::new(0.125, 2.0, 0.1875);
    let hit =
        raycast_construction(&graph, origin, Vec3::NEG_Y).expect("the wedge is under the ray");
    assert!(
        matches!(hit.face.owner, FaceOwner::Part(_)),
        "the ray should meet the block, not the ground"
    );
    assert!(
        hit.point.y < 0.25 - 1.0e-3,
        "the slope is below the old box top; hit at y={}",
        hit.point.y
    );
    assert!(
        hit.point.y > 0.0,
        "the hit should still be on the wedge, not through it"
    );
}

#[test]
fn raycast_tests_a_multi_block_region_once() {
    fn spawn_at(graph: &mut ConstructionGraph, half_grid: IVec3) -> PartId {
        let spec = CuboidSpec::new(
            [1; 3],
            BuildPose::from_half_grid(half_grid, GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        part
    }

    let mut graph = ConstructionGraph::new();
    let first = spawn_at(&mut graph, IVec3::new(1, 1, 1));
    let second = spawn_at(&mut graph, IVec3::new(3, 1, 1));
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(first, FaceKind::PositiveX),
            second: FaceRef::part(second, FaceKind::NegativeX),
        }))
        .unwrap();
    let region = mechanic_core::ShapeRegion::new(
        IVec3::ZERO,
        IVec3::new(2, 1, 1),
        ConstructionMaterial::Steel,
    )
    .unwrap();
    graph.apply(BuildCommand::AddRegion(region)).unwrap();
    spawn_at(&mut graph, IVec3::new(9, 1, 1));

    let sources = raycast_sources(&graph, |_| true).collect::<Vec<_>>();
    assert_eq!(sources.len(), 2, "one region and one standalone part");
    let accepted = super::raycast_construction_filtered_with_ground(
        &graph,
        Vec3::new(0.125, 2.0, 0.125),
        Vec3::NEG_Y,
        None,
        |part| part == second,
    )
    .unwrap();
    assert_eq!(accepted.face.owner, FaceOwner::Part(second));
    assert_eq!(
        sources
            .iter()
            .filter(|(_, _, region)| region.is_some())
            .count(),
        1,
        "all region members share one raycast source"
    );

    let origin = Vec3::new(0.375, 2.0, 0.125);
    let hit = raycast_construction_with_ground(&graph, origin, Vec3::NEG_Y, None).unwrap();
    let frame = mechanic_core::ConstructionFrame::new(
        Vec3::new(3.0, 4.0, 2.0),
        Quat::from_rotation_z(0.43),
    )
    .unwrap();
    graph.reframe_parts([first, second], frame).unwrap();
    let reframed = raycast_construction_with_ground(
        &graph,
        frame.point(origin),
        frame.vector(Vec3::NEG_Y),
        None,
    )
    .unwrap();
    assert_eq!(reframed.face, hit.face);
    assert!(reframed.point.distance(frame.point(hit.point)) < 1.0e-5);
}

#[test]
fn featured_region_flat_patch_accepts_placement_across_member_blocks() {
    let mut graph = ConstructionGraph::new();
    let spawn_at = |graph: &mut ConstructionGraph, center| {
        let spec = CuboidSpec::new(
            [1; 3],
            BuildPose::from_half_grid(center, GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        part
    };
    let first = spawn_at(&mut graph, IVec3::new(1, 1, 1));
    let second = spawn_at(&mut graph, IVec3::new(3, 1, 1));
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(first, FaceKind::PositiveX),
            second: FaceRef::part(second, FaceKind::NegativeX),
        }))
        .unwrap();
    let region = mechanic_core::ShapeRegion::new(
        IVec3::ZERO,
        IVec3::new(2, 1, 1),
        ConstructionMaterial::Steel,
    )
    .unwrap();
    let BuildOutcome::RegionAdded(region) = graph.apply(BuildCommand::AddRegion(region)).unwrap()
    else {
        unreachable!()
    };
    let owner = SolidOwner::Region(region);
    let edge = graph.evaluated_solid(owner).unwrap().logical_edges[0].key;
    graph
        .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
            [EdgeChainRef { owner, edge }],
            EdgeTreatment::Fillet,
            20,
        )))
        .unwrap();

    let hit = raycast_construction(&graph, Vec3::new(0.375, 2.0, 0.125), Vec3::NEG_Y)
        .expect("the retained top patch is under the ray");
    let candidate = candidate_from_hit(&graph, hit);

    assert!(
        hit.face.patch.is_some(),
        "evaluated hits retain patch identity"
    );
    assert!(
        candidate.anchor.is_some(),
        "the patch must not collapse to the representative member's 25 cm face"
    );
    validate_block_batch_in_bounds(
        &graph,
        candidate,
        &[candidate.spec],
        PlacementBounds::Garage,
    )
    .expect("a block may be placed on the second member's flat patch");
    let staged = stage_block_batch_in_bounds(
        &graph,
        candidate,
        &[candidate.spec],
        PlacementBounds::Garage,
    )
    .expect("placement commits on the evaluated patch");
    assert_eq!(staged.part_count(), 3);
    assert_eq!(staged.weld_count(), 2);
}

#[test]
fn fillet_hit_places_against_the_nearest_retained_flat_patch() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [4; 3],
                BuildPose::new(IVec3::new(1, 1, 1), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let owner = SolidOwner::Part(part);
    let base = graph.evaluated_solid(owner).unwrap();
    let logical = &base.logical_edges[0];
    let half_edge = base.half_edges[logical.half_edges[0] as usize];
    let twin = base.half_edges[half_edge.twin as usize];
    let start = base.vertices[half_edge.origin as usize].position;
    let end = base.vertices[base.half_edges[half_edge.next as usize].origin as usize].position;
    let outward = (base.surfaces[half_edge.face as usize].normal
        + base.surfaces[twin.face as usize].normal)
        .normalize();
    graph
        .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
            [EdgeChainRef {
                owner,
                edge: logical.key,
            }],
            EdgeTreatment::Fillet,
            20,
        )))
        .unwrap();

    let edge_midpoint = (start + end) * 0.5;
    let hit = raycast_construction(&graph, edge_midpoint + outward * 2.0, -outward)
        .expect("the ray crosses the rounded edge");
    let candidate = candidate_from_hit(&graph, hit);

    assert!(
        matches!(
            hit.face.patch.map(|patch| patch.source),
            Some(mechanic_core::TopologySource::Base)
        ),
        "a rounded facet routes placement to an adjacent base plane"
    );
    assert!(candidate.anchor.is_some());
}

#[test]
fn block_sheet_only_needs_one_block_on_a_promoted_filleted_region() {
    let mut graph = ConstructionGraph::new();
    let mut members = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            let spec = CuboidSpec::new(
                [1; 3],
                BuildPose::from_half_grid(
                    IVec3::new(1 + x * 2, 1 + y * 2, 1),
                    GridRotation::default(),
                ),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            if let Some(&previous) = members.last() {
                graph
                    .apply(BuildCommand::RigidLink(RigidLinkSpec {
                        first: previous,
                        second: part,
                    }))
                    .unwrap();
            }
            members.push(part);
        }
    }
    let region = mechanic_core::ShapeRegion::new(
        IVec3::ZERO,
        IVec3::new(4, 4, 1),
        ConstructionMaterial::Steel,
    )
    .unwrap();
    let BuildOutcome::RegionAdded(region) = graph.apply(BuildCommand::AddRegion(region)).unwrap()
    else {
        unreachable!()
    };
    let owner = SolidOwner::Region(region);
    let base = graph.evaluated_solid(owner).unwrap();
    let edge = base
        .logical_edges
        .iter()
        .find(|logical| {
            let half_edge = base.half_edges[logical.half_edges[0] as usize];
            let twin = base.half_edges[half_edge.twin as usize];
            let normals = [
                base.surfaces[half_edge.face as usize].normal,
                base.surfaces[twin.face as usize].normal,
            ];
            normals.contains(&Vec3::X) && normals.contains(&Vec3::Y)
        })
        .expect("the cuboid has a positive-x/positive-y edge")
        .key;
    graph
        .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
            [EdgeChainRef { owner, edge }],
            EdgeTreatment::Fillet,
            120,
        )))
        .unwrap();

    let hit = raycast_construction(&graph, Vec3::new(0.625, 2.0, 0.125), Vec3::NEG_Y)
        .expect("the retained top patch is under the ray");
    let start = candidate_from_hit(&graph, hit);
    let specs = block_box_specs(start.spec, IVec3::X).unwrap();

    assert!(start.anchor.is_some());
    let staged = stage_block_batch_in_bounds(&graph, start, &specs, PlacementBounds::Garage)
        .expect("one supported block keeps the connected sheet placeable");
    assert_eq!(staged.part_count(), 18);
}

#[test]
fn an_unshaped_part_still_reports_its_grid_face() {
    let spec = CuboidSpec::new(
        [1, 1, 1],
        BuildPose::from_half_grid(IVec3::new(1, 1, 1), GridRotation::default()),
    )
    .unwrap();
    let mut graph = ConstructionGraph::new();
    graph.apply(BuildCommand::Spawn(spec)).unwrap();
    let hit = raycast_construction(&graph, Vec3::new(0.125, 2.0, 0.125), Vec3::NEG_Y)
        .expect("the block is under the ray");
    assert_eq!(hit.face.face, FaceKind::PositiveY);
    assert!((hit.point.y - 0.25).abs() < 1.0e-4);
}

/// One block claimed as a region, with its top +z edge optionally collapsed.
fn block_with_region(shaped: bool) -> (ConstructionGraph, FaceRef) {
    let spec = CuboidSpec::new(
        [1, 1, 1],
        BuildPose::from_half_grid(IVec3::new(1, 1, 1), GridRotation::default()),
    )
    .unwrap();
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
        panic!("wrong spawn outcome")
    };
    let region = mechanic_core::ShapeRegion::new(
        IVec3::ZERO,
        IVec3::ONE,
        mechanic_core::ConstructionMaterial::Steel,
    )
    .unwrap();
    let BuildOutcome::RegionAdded(id) = graph.apply(BuildCommand::AddRegion(region)).unwrap()
    else {
        panic!("wrong outcome")
    };
    if shaped {
        let cell = i16::try_from(mechanic_core::POSITION_TICKS_PER_GRID_UNIT).unwrap();
        graph
            .apply(BuildCommand::SetRegionVertices {
                region: id,
                vertices: vec![([0, 1, 1], [0, -cell, 0]), ([1, 1, 1], [0, -cell, 0])],
            })
            .unwrap();
    }
    (graph, FaceRef::part(part, FaceKind::PositiveY))
}

#[test]
fn placement_is_refused_on_a_shaped_face() {
    // The top face has been sloped, so nothing can sit flush on it.
    let (graph, top) = block_with_region(true);
    assert!(!face_is_flat(&graph, top));
}

#[test]
fn flattening_a_shaped_face_makes_it_placeable_again() {
    // Bringing those corners back onto the grid is how a mounting surface
    // is made where the shaping had removed one.
    let (mut graph, top) = block_with_region(true);
    assert!(!face_is_flat(&graph, top));
    let id = graph.regions().next().unwrap().0;
    graph
        .apply(BuildCommand::SetRegionVertices {
            region: id,
            vertices: vec![([0, 1, 1], [0, 0, 0]), ([1, 1, 1], [0, 0, 0])],
        })
        .unwrap();
    assert!(face_is_flat(&graph, top));
}

#[test]
fn an_unshaped_face_is_always_placeable() {
    let (graph, top) = block_with_region(false);
    assert!(face_is_flat(&graph, top));
    assert!(
        face_is_flat(&graph, FaceRef::ground()),
        "the ground is always flat"
    );
}

#[test]
fn shaping_one_face_leaves_the_others_placeable() {
    // Only the face that moved loses its mounting surface.
    let (graph, _) = block_with_region(true);
    let part = graph.parts().next().unwrap().0;
    assert!(
        face_is_flat(&graph, FaceRef::part(part, FaceKind::NegativeY)),
        "the untouched underside must still take a block"
    );
}

#[test]
fn pipe_run_preserves_fine_grid_lateral_offsets() {
    let start = Vec3::new(0.05, 0.0, 0.10);
    let pieces = pipe_run_pieces(
        &[start, start + Vec3::Y * 0.25],
        &[],
        CylinderDimensions::default(),
        ConstructionMaterial::Steel,
    )
    .unwrap();
    let PartSpec::Cylinder(pipe) = pieces[0].spec else {
        panic!("a straight run produces a cylinder")
    };

    assert!(
        pipe.pose
            .translation()
            .abs_diff_eq(Vec3::new(0.05, 0.125, 0.10), 1.0e-5)
    );
}

#[test]
fn pipe_run_trims_straights_to_bend_tangencies() {
    let dimensions = CylinderDimensions::new(0.25, 0.10, 0.25).unwrap();
    // A four-block leg turns in its last block, at that block's centre.
    let pieces = pipe_run_pieces(
        &[
            Vec3::ZERO,
            Vec3::new(0.875, 0.0, 0.0),
            Vec3::new(0.875, 0.875, 0.0),
        ],
        &[PipeNode::Bend { span: 1 }],
        dimensions,
        ConstructionMaterial::Steel,
    )
    .unwrap();
    assert_eq!(pieces.len(), 3);
    assert!(matches!(pieces[1].spec, PartSpec::PipeBend(_)));
    for piece in [pieces[0], pieces[2]] {
        let PartSpec::Cylinder(cylinder) = piece.spec else {
            panic!("the ends remain straight")
        };
        assert!((cylinder.dimensions.axial_length() - 0.75).abs() < 1.0e-5);
    }
}

#[test]
fn branching_from_the_side_of_a_straight_pipe_splits_it_around_a_welded_tee() {
    let dimensions = CylinderDimensions::new(0.25, 0.10, 0.5).unwrap();
    let trunk = pipe_run_pieces(
        &[Vec3::new(0.125, 0.0, 0.125), Vec3::new(0.125, 1.0, 0.125)],
        &[],
        dimensions,
        ConstructionMaterial::Steel,
    )
    .unwrap();
    let graph = stage_pipe_run(
        &ConstructionGraph::new(),
        &trunk,
        PipeRunAttachment::AutoWeld {
            source: FaceOwner::Ground,
        },
    )
    .unwrap();
    let (pipe, _) = graph.parts().next().unwrap();
    let hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::new(0.25, 0.41, 0.125),
        face: FaceRef::part(pipe, FaceKind::PositiveX),
    };

    let (candidate, branch) =
        super::pipe_branch_candidate(&graph, hit, dimensions, Vec3::X, 0).unwrap();
    assert!(
        branch
            .junction
            .pose
            .translation()
            .abs_diff_eq(Vec3::new(0.125, 0.375, 0.125), 1.0e-5),
        "the tee takes the block cell nearest the hit"
    );
    assert!(
        candidate
            .spec
            .pose
            .translation()
            .abs_diff_eq(Vec3::new(0.5, 0.375, 0.125), 1.0e-5)
    );

    let (split, junction) = super::apply_pipe_branch(&graph, branch).unwrap();
    assert_eq!(split.part_count(), 3);
    assert_eq!(
        split.weld_count(),
        3,
        "ground weld moves to the lower straight"
    );
    let lengths = split
        .parts()
        .filter_map(|(_, spec)| spec.as_cylinder())
        .map(|cylinder| cylinder.dimensions.axial_length())
        .collect::<Vec<_>>();
    assert!(
        lengths.contains(&0.25) && lengths.contains(&0.5),
        "{lengths:?}"
    );
    let branch_pieces = pipe_run_pieces(
        &[Vec3::new(0.25, 0.375, 0.125), Vec3::new(0.75, 0.375, 0.125)],
        &[],
        dimensions,
        ConstructionMaterial::Steel,
    )
    .unwrap();
    let staged = stage_pipe_run(
        &split,
        &branch_pieces,
        PipeRunAttachment::AutoWeld {
            source: FaceOwner::Part(junction),
        },
    )
    .unwrap();
    assert_eq!(staged.part_count(), 4);
    assert_eq!(staged.weld_count(), 4);
    assert!(staged.compile().is_ok());
}

fn branchable_trunk() -> (ConstructionGraph, mechanic_core::PartId, CylinderDimensions) {
    let dimensions = CylinderDimensions::new(0.25, 0.10, 0.5).unwrap();
    let trunk = pipe_run_pieces(
        &[Vec3::new(0.125, 0.0, 0.125), Vec3::new(0.125, 1.0, 0.125)],
        &[],
        dimensions,
        ConstructionMaterial::Steel,
    )
    .unwrap();
    let graph = stage_pipe_run(
        &ConstructionGraph::new(),
        &trunk,
        PipeRunAttachment::AutoWeld {
            source: FaceOwner::Ground,
        },
    )
    .unwrap();
    let (pipe, _) = graph.parts().next().unwrap();
    (graph, pipe, dimensions)
}

#[test]
fn branch_arm_faces_the_player_and_rotating_steps_it_around_the_pipe() {
    let (graph, pipe, dimensions) = branchable_trunk();
    let hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::new(0.125, 0.41, 0.25),
        face: FaceRef::part(pipe, FaceKind::PositiveZ),
    };
    let toward_player = Vec3::new(0.3, 0.2, 1.0).normalize();
    let outlets = (0..5)
        .map(|turn| {
            let (candidate, branch) =
                super::pipe_branch_candidate(&graph, hit, dimensions, toward_player, turn).unwrap();
            assert_eq!(branch.junction.arms.count(), 3);
            let outward = (candidate.spec.pose.translation() - branch.junction.pose.translation())
                .normalize();
            let outlet = super::pipes::face_toward(outward);
            assert!(branch.junction.arms.contains(outlet), "turn {turn}");
            outlet
        })
        .collect::<Vec<_>>();
    assert_eq!(outlets[0], FaceKind::PositiveZ, "the arm faces the player");
    assert_eq!(outlets[2], FaceKind::NegativeZ);
    assert_eq!(outlets[4], outlets[0], "four turns come back around");
    assert!(
        (outlets[1] == FaceKind::PositiveX && outlets[3] == FaceKind::NegativeX)
            || (outlets[1] == FaceKind::NegativeX && outlets[3] == FaceKind::PositiveX),
        "{outlets:?}"
    );
}

#[test]
fn a_branch_off_a_lying_pipe_is_placeable_and_stays_inside_the_trunk_behind_it() {
    let dimensions = CylinderDimensions::new(0.20, 0.10, 0.5).unwrap();
    let trunk = pipe_run_pieces(
        &[Vec3::new(0.0, 0.125, 0.125), Vec3::new(2.0, 0.125, 0.125)],
        &[],
        dimensions,
        ConstructionMaterial::Steel,
    )
    .unwrap();
    let graph = stage_pipe_run(&ConstructionGraph::new(), &trunk, PipeRunAttachment::Free).unwrap();
    let (pipe, _) = graph.parts().next().unwrap();
    let hit = SurfaceHit {
        distance: 1.0,
        point: Vec3::new(0.9, 0.125, 0.225),
        face: FaceRef::part(pipe, FaceKind::PositiveZ),
    };
    let wider = CylinderDimensions::new(0.25, 0.0, 0.5).unwrap();
    let toward_player = Vec3::new(0.1, 0.3, 1.0).normalize();
    let (candidate, branch) =
        super::pipe_branch_candidate(&graph, hit, wider, toward_player, 0).unwrap();
    super::validate_cylinder_candidate_in_bounds(&graph, candidate, PlacementBounds::Garage)
        .expect("the branch preview is placeable");
    assert!(
        (candidate.spec.dimensions.outer_diameter() - 0.20).abs() < 1.0e-6,
        "the branch matches the trunk"
    );
    let center = branch.junction.pose.translation();
    let furthest = mechanic_core::pipe_junction_triangles(branch.junction)
        .iter()
        .flat_map(|triangle| triangle.outer)
        .filter(|point| point.z < -1.0e-4)
        .map(|point| bevy::math::Vec2::new(point.y, point.z).length())
        .fold(0.0_f32, f32::max);
    assert!(
        furthest <= 0.1 + 1.0e-4,
        "nothing pokes out behind the trunk at {center}: {furthest}"
    );
}

#[test]
fn branching_off_a_junction_side_opens_another_arm_and_keeps_its_welds() {
    let (graph, pipe, dimensions) = branchable_trunk();
    let side = SurfaceHit {
        distance: 1.0,
        point: Vec3::new(0.125, 0.41, 0.25),
        face: FaceRef::part(pipe, FaceKind::PositiveZ),
    };
    let (_, tee) = super::pipe_branch_candidate(&graph, side, dimensions, Vec3::Z, 0).unwrap();
    let (graph, junction) = super::apply_pipe_branch(&graph, tee).unwrap();
    let welds = graph.weld_count();
    let wall = SurfaceHit {
        distance: 1.0,
        point: tee.junction.pose.translation() - Vec3::X * 0.125,
        face: FaceRef::part(junction, FaceKind::NegativeX),
    };
    let (_, cross) =
        super::pipe_branch_candidate(&graph, wall, dimensions, Vec3::NEG_X, 0).unwrap();
    assert_eq!(cross.site, super::pipes::PipeBranchSite::Extend(junction));

    let (graph, opened) = super::apply_pipe_branch(&graph, cross).unwrap();
    let arms = graph
        .part(opened)
        .and_then(|spec| spec.as_pipe_junction())
        .unwrap()
        .arms;
    assert_eq!(arms.count(), 4);
    assert!(arms.contains(FaceKind::NegativeX));
    assert_eq!(
        graph.weld_count(),
        welds,
        "every weld on the tee moves over"
    );
    assert!(graph.compile().is_ok());
}

#[test]
fn bent_run_next_leg_runs_through_block_centres() {
    let dimensions = CylinderDimensions::new(0.25, 0.0, 0.25).unwrap();
    for span in 1..=3 {
        let corner = Vec3::new(0.875, 0.125, 0.125);
        let pieces = pipe_run_pieces(
            &[
                Vec3::new(0.0, 0.125, 0.125),
                corner,
                corner + Vec3::Y * (1.0 - 0.125),
            ],
            &[PipeNode::Bend { span }],
            dimensions,
            ConstructionMaterial::Steel,
        )
        .unwrap();
        for piece in &pieces {
            let PartSpec::Cylinder(cylinder) = piece.spec else {
                continue;
            };
            let centre = cylinder.pose.translation();
            let blocks = cylinder.dimensions.axial_length() / 0.25;
            assert!((blocks - blocks.round()).abs() < 1.0e-4, "span {span}");
            // Every straight axis stays on block centres across the run.
            for lateral in [centre.z, if centre.y > 0.2 { centre.x } else { centre.y }] {
                let offset = (lateral - 0.125) / 0.25;
                assert!((offset - offset.round()).abs() < 1.0e-4, "span {span}");
            }
        }
        let bend = pieces
            .iter()
            .find_map(|piece| piece.spec.as_pipe_bend())
            .unwrap();
        assert_eq!(bend.dimensions.span_blocks(), span);
        // The bend's footprint ends exactly on the block boundary.
        let reach = bend.dimensions.radius() + 0.125;
        assert!((reach - f32::from(span) * 0.25).abs() < 1.0e-5);
    }
}

#[test]
fn pipe_run_omits_zero_length_straights_and_keeps_per_corner_spans() {
    let dimensions = CylinderDimensions::new(0.25, 0.10, 0.25).unwrap();
    let one = pipe_run_pieces(
        &[Vec3::ZERO, Vec3::X * 0.125, Vec3::new(0.125, 0.125, 0.0)],
        &[PipeNode::Bend { span: 1 }],
        dimensions,
        ConstructionMaterial::Steel,
    )
    .unwrap();
    assert_eq!(one.len(), 1);
    assert!(matches!(one[0].spec, PartSpec::PipeBend(_)));

    let multiple = pipe_run_pieces(
        &[
            Vec3::ZERO,
            Vec3::X * 0.875,
            Vec3::new(0.875, 1.0, 0.0),
            Vec3::new(1.5, 1.0, 0.0),
        ],
        &[PipeNode::Bend { span: 1 }, PipeNode::Bend { span: 2 }],
        dimensions,
        ConstructionMaterial::Aluminium,
    )
    .unwrap();
    let radii = multiple
        .iter()
        .filter_map(|piece| piece.spec.as_pipe_bend())
        .map(|bend| bend.dimensions.radius())
        .collect::<Vec<_>>();
    assert_eq!(radii, vec![0.125, 0.375]);
}

#[test]
fn pipe_run_rejects_insufficient_between_bend_clearance() {
    let dimensions = CylinderDimensions::new(0.25, 0.10, 0.25).unwrap();
    let error = pipe_run_pieces(
        &[
            Vec3::ZERO,
            Vec3::X * 0.875,
            Vec3::new(0.875, 0.5, 0.0),
            Vec3::new(1.5, 0.5, 0.0),
        ],
        &[PipeNode::Bend { span: 2 }, PipeNode::Bend { span: 2 }],
        dimensions,
        ConstructionMaterial::Steel,
    )
    .unwrap_err();
    assert!(error.to_string().contains("0.75 m clearance"), "{error}");
}

#[test]
fn pipe_bend_end_raycasts_report_the_flat_caps() {
    let mut graph = ConstructionGraph::new();
    let dimensions = PipeBendDimensions::new(0.20, 0.10, 1).unwrap();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::SpawnPipeBend(PipeBendSpec::new(
            dimensions,
            BuildPose::default(),
        )))
        .unwrap()
    else {
        panic!("pipe bend must spawn")
    };

    for (origin, direction, face) in [
        (
            Vec3::new(0.075, dimensions.radius() + 1.0, 0.0),
            Vec3::NEG_Y,
            FaceKind::PositiveY,
        ),
        (
            Vec3::new(-dimensions.radius() - 1.0, 0.0, 0.075),
            Vec3::X,
            FaceKind::NegativeX,
        ),
    ] {
        let hit =
            raycast_construction(&graph, origin, direction).expect("the pipe bend cap is visible");
        assert_eq!(hit.face, FaceRef::part(part, face));
    }
}

#[test]
fn a_pipe_bends_right_off_the_block_face_it_starts_on() {
    for (outer_diameter, span) in [(0.15, 1), (0.25, 1), (0.25, 3), (0.5, 2)] {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::ZERO, 1);
        let start = Vec3::Y * 0.125;
        let inset = f32::from(PipeBendDimensions::channel_blocks(outer_diameter)) * 0.125;
        let corner = start + Vec3::Y * (f32::from(span) * 0.25 - inset);
        let pieces = pipe_run_pieces(
            &[
                start,
                corner,
                corner + Vec3::X * (f32::from(span) * 0.25 - inset),
            ],
            &[PipeNode::Bend { span }],
            CylinderDimensions::new(outer_diameter, 0.0, 0.25).unwrap(),
            ConstructionMaterial::Steel,
        )
        .unwrap();

        let staged = stage_pipe_run(
            &graph,
            &pieces,
            PipeRunAttachment::AutoWeld {
                source: FaceOwner::Part(base),
            },
        )
        .unwrap_or_else(|error| {
            panic!("{outer_diameter} m pipe with a {span}-block bend: {error}")
        });

        assert_eq!(pieces.len(), 1, "the bend is the whole run");
        assert_eq!(staged.part_count(), 2);
        assert_eq!(staged.weld_count(), 1, "the bend welds to the block");
    }
}

#[test]
fn sub_block_pipe_can_turn_immediately_after_an_existing_bend() {
    let mut graph = ConstructionGraph::new();
    let bend_dimensions = PipeBendDimensions::new(0.20, 0.10, 1).unwrap();
    let BuildOutcome::Spawned(source) = graph
        .apply(BuildCommand::SpawnPipeBend(PipeBendSpec::new(
            bend_dimensions,
            BuildPose::default(),
        )))
        .unwrap()
    else {
        panic!("source bend must spawn")
    };
    let start = Vec3::Y * bend_dimensions.radius();
    let corner = start + Vec3::Y * 0.125;
    let pieces = pipe_run_pieces(
        &[start, corner, corner + Vec3::X * 0.125],
        &[PipeNode::Bend { span: 1 }],
        CylinderDimensions::new(0.20, 0.10, 0.25).unwrap(),
        ConstructionMaterial::Steel,
    )
    .unwrap();

    let staged = stage_pipe_run(
        &graph,
        &pieces,
        PipeRunAttachment::AutoWeld {
            source: FaceOwner::Part(source),
        },
    )
    .expect("the new bend only touches the source at its inlet cap");

    assert_eq!(pieces.len(), 1);
    assert_eq!(staged.part_count(), 2);
    assert_eq!(staged.weld_count(), 1);
    assert!(staged.compile().is_ok());
}

#[test]
fn linear_bent_pipe_joins_existing_carriage_attachments_as_one_compound() {
    use mechanic_core::{CarriageFace, LinearBearing, LinearBearingDimensions};
    let mut graph = ConstructionGraph::new();
    let base = spawn_cube(&mut graph, IVec3::ZERO, 1);
    let source = FaceRef::part(base, FaceKind::PositiveY);
    let anchor = Vec3::new(0.0, 0.125, 0.0);
    let rail = LinearBearing {
        dimensions: LinearBearingDimensions::default(),
        mount_normal: Vec3::Y,
        face: CarriageFace::Top,
    };
    let surface = super::bearings::linear_carriage_face(anchor, rail, Vec3::X).unwrap();
    let attachment = super::LinearAttachment {
        source,
        anchor,
        rail,
        axis: Vec3::X,
        rigid_targets: &[],
    };
    let block = super::linear_block_candidate(
        anchor,
        rail,
        Vec3::X,
        surface.center - Vec3::Z * 0.125,
        [1; 3],
        GridRotation::default(),
    )
    .unwrap();
    let graph = super::stage_linear_block_batch_in_bounds(
        &graph,
        block,
        &[block.spec],
        attachment,
        PlacementBounds::Garage,
    )
    .unwrap();
    let targets = graph
        .parts()
        .filter_map(|(part, _)| (part != base).then_some(part))
        .collect::<Vec<_>>();
    let start = surface.center + Vec3::Z * 0.125;
    let corner = start + Vec3::Y * 0.875;
    let pieces = pipe_run_pieces(
        &[start, corner, corner + Vec3::X * 0.875],
        &[PipeNode::Bend { span: 1 }],
        CylinderDimensions::new(0.20, 0.10, 0.25).unwrap(),
        ConstructionMaterial::Steel,
    )
    .unwrap();
    let staged = super::stage_pipe_run_in_bounds(
        &graph,
        &pieces,
        PipeRunAttachment::Linear(super::LinearAttachment {
            rigid_targets: &targets,
            ..attachment
        }),
        PlacementBounds::Garage,
    )
    .unwrap();
    assert!(
        pieces
            .iter()
            .any(|piece| matches!(piece.spec, PartSpec::PipeBend(_)))
    );
    assert_eq!(staged.part_count(), graph.part_count() + pieces.len());
    assert_eq!(staged.weld_count(), pieces.len() - 1);
    let compiled = staged.compile().unwrap();
    assert_eq!(compiled.compounds.len(), 2);
    assert_eq!(compiled.bearings.len(), 1);
    assert!(matches!(
        compiled.bearings[0].kind,
        mechanic_core::BearingKind::Linear(_)
    ));
    assert_eq!(
        graph.part_count(),
        2,
        "staging must preserve its input graph"
    );
}

#[test]
fn staged_pipe_run_spawns_and_welds_every_piece_atomically() {
    let graph = ConstructionGraph::new();
    let dimensions = CylinderDimensions::new(0.25, 0.10, 0.25).unwrap();
    let pieces = pipe_run_pieces(
        &[Vec3::ZERO, Vec3::Y * 0.875, Vec3::new(0.875, 0.875, 0.0)],
        &[PipeNode::Bend { span: 2 }],
        dimensions,
        ConstructionMaterial::Steel,
    )
    .unwrap();
    let staged = stage_pipe_run(
        &graph,
        &pieces,
        PipeRunAttachment::AutoWeld {
            source: FaceOwner::Ground,
        },
    )
    .unwrap();
    assert_eq!(staged.part_count(), pieces.len());
    assert_eq!(staged.weld_count(), pieces.len());
    assert_eq!(
        graph.part_count(),
        0,
        "staging leaves the source graph untouched"
    );
    assert!(staged.compile().is_ok());
    assert!(pieces.iter().any(|piece| {
        piece
            .spec
            .as_pipe_bend()
            .is_some_and(|bend| (bend.dimensions.radius() - 0.375).abs() < 1.0e-5)
    }));
}

fn layer_test_graph(neighbour_x_ticks: Option<i32>) -> (ConstructionGraph, PartId) {
    let mut graph = ConstructionGraph::new();
    let pipe = CylinderSpec::new(
        CylinderDimensions::new(1.0, 0.5, 1.0).unwrap(),
        BuildPose::from_position_ticks([0, 400, 0].into(), GridRotation::default()),
    );
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::SpawnCylinder(pipe)).unwrap()
    else {
        panic!("spawning a cylinder reports its part");
    };
    if let Some(x) = neighbour_x_ticks {
        graph
            .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
                CylinderDimensions::new(0.5, 0.0, 1.0).unwrap(),
                BuildPose::from_position_ticks([x, 400, 0].into(), GridRotation::default()),
            )))
            .unwrap();
    }
    (graph, part)
}

fn surface_hit(part: PartId, point: Vec3, face: FaceKind) -> super::SurfaceHit {
    super::SurfaceHit {
        distance: 1.0,
        point,
        face: FaceRef::part(part, face),
    }
}

fn spawn_block(graph: &mut ConstructionGraph, x_ticks: i32) -> PartId {
    let block = CuboidSpec::new(
        [2, 2, 2],
        BuildPose::from_position_ticks([x_ticks, 100, 0].into(), GridRotation::default()),
    )
    .unwrap();
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(block)).unwrap() else {
        panic!("spawning a block reports its part");
    };
    part
}

#[test]
fn layer_host_accepts_faces_caps_walls_and_rejects_fillet_surfaces() {
    let (mut graph, pipe) = layer_test_graph(None);
    let target = |graph: &ConstructionGraph, part, point, face| {
        super::layer_target_from_hit(graph, surface_hit(part, point, face))
    };
    let outer = target(&graph, pipe, Vec3::new(0.5, 1.0, 0.0), FaceKind::PositiveX).unwrap();
    assert_eq!(outer.face, mechanic_core::LayerFace::OuterWall);
    assert!(outer.normal.abs_diff_eq(Vec3::X, 1.0e-5));
    let bore = target(&graph, pipe, Vec3::new(0.0, 1.2, 0.25), FaceKind::PositiveX).unwrap();
    assert_eq!(bore.face, mechanic_core::LayerFace::Bore);
    assert!(bore.normal.abs_diff_eq(Vec3::NEG_Z, 1.0e-5));
    let cap = target(&graph, pipe, Vec3::new(0.4, 1.5, 0.0), FaceKind::PositiveY).unwrap();
    assert_eq!(
        cap.face,
        mechanic_core::LayerFace::Face(FaceKind::PositiveY)
    );

    let block = spawn_block(&mut graph, 800);
    let top = target(&graph, block, Vec3::new(2.0, 0.5, 0.1), FaceKind::PositiveY).unwrap();
    assert_eq!(
        top.face,
        mechanic_core::LayerFace::Face(FaceKind::PositiveY)
    );
    assert!(top.normal.abs_diff_eq(Vec3::Y, 1.0e-5));

    // Round the block's top +X edge, then aim at the middle of the fillet.
    let owner = mechanic_core::SolidOwner::Part(block);
    let solid = graph.evaluated_solid(owner).unwrap();
    let edge = solid
        .logical_edges
        .iter()
        .find(|edge| {
            edge.half_edges.iter().all(|&half_edge| {
                let origin = solid.half_edges[half_edge as usize].origin;
                let position = solid.vertices[origin as usize].position;
                position.x > 2.2 && position.y > 0.45
            })
        })
        .unwrap()
        .key;
    graph
        .apply(BuildCommand::AddShapeFeature(
            mechanic_core::ShapeFeature::new(
                [mechanic_core::EdgeChainRef { owner, edge }],
                mechanic_core::EdgeTreatment::Fillet,
                40,
            ),
        ))
        .unwrap();
    let rounded = Vec3::new(2.15, 0.4, 0.0) + Vec3::new(1.0, 1.0, 0.0).normalize() * 0.1;
    assert_eq!(
        target(&graph, block, rounded, FaceKind::PositiveX),
        Err(PlacementError::NotLayerSurface)
    );
}

#[test]
fn dragging_out_thickens_and_back_thins_by_the_modifier_step() {
    let (graph, pipe) = layer_test_graph(None);
    let cap = super::layer_target_from_hit(
        &graph,
        surface_hit(pipe, Vec3::new(0.4, 1.5, 0.0), FaceKind::PositiveY),
    )
    .unwrap();
    // Looking sideways at the top cap, so pointer height is pull distance.
    let ray = |y: f32| (Vec3::new(0.4, y, 5.0), Vec3::NEG_Z);
    let (origin, direction) = ray(1.5);
    let mut drag = super::LayerDrag::begin(
        cap,
        ConstructionMaterial::Rubber,
        mechanic_core::MaterialAppearance::BAKED,
        0.25,
        origin,
        direction,
    );
    let mut step = |grid, y| {
        let (origin, direction) = ray(y);
        drag.update(grid, origin, direction);
        drag.thickness
    };
    assert!((step(super::PlacementGrid::Centimetres25, 1.8) - 0.5).abs() < 1.0e-4);
    assert!((step(super::PlacementGrid::Centimetres5, 1.62) - 0.35).abs() < 1.0e-4);
    assert!((step(super::PlacementGrid::Centimetres1, 1.543) - 0.29).abs() < 1.0e-4);
    assert!(
        (step(super::PlacementGrid::Centimetres25, 1.0) - 0.25).abs() < 1.0e-4,
        "never thinner than one step"
    );
}

#[test]
fn head_on_drag_advances_at_most_one_step_per_frame() {
    let (graph, pipe) = layer_test_graph(None);
    let cap = super::layer_target_from_hit(
        &graph,
        surface_hit(pipe, Vec3::new(0.4, 1.5, 0.0), FaceKind::PositiveY),
    )
    .unwrap();
    // Nearly straight down the cap normal: projected distance swings wildly.
    let direction = Vec3::new(0.0, -1.0, 0.05).normalize();
    let mut drag = super::LayerDrag::begin(
        cap,
        ConstructionMaterial::Rubber,
        mechanic_core::MaterialAppearance::BAKED,
        0.25,
        Vec3::new(0.4, 5.0, -1.0),
        direction,
    );
    drag.update(
        super::PlacementGrid::Centimetres25,
        Vec3::new(0.4, 5.0, -0.5),
        direction,
    );
    assert!((drag.thickness - 0.5).abs() < 1.0e-4);
}

#[test]
fn face_layer_that_would_hit_a_neighbour_is_refused() {
    let mut graph = ConstructionGraph::new();
    let block = spawn_block(&mut graph, 0);
    spawn_block(&mut graph, 300);
    let side = super::layer_target_from_hit(
        &graph,
        surface_hit(block, Vec3::new(0.25, 0.25, 0.0), FaceKind::PositiveX),
    )
    .unwrap();
    let rubber = |thickness| {
        super::stage_layer(
            &graph,
            &side,
            thickness,
            ConstructionMaterial::Rubber,
            mechanic_core::MaterialAppearance::BAKED,
            PlacementBounds::Garage,
        )
    };
    assert!(matches!(rubber(0.3), Err(PlacementError::OverlapsPart(_))));
    let (layered, parts) = rubber(0.1).unwrap();
    assert_eq!(parts.len(), 1);
    let (_, spec) = parts[0];
    assert_eq!(layered.part(block), Some(&spec));
    assert!((spec.as_cuboid().unwrap().size_meters().x - 0.6).abs() < 1.0e-5);
}

#[test]
fn layer_spreads_over_the_whole_uncovered_flat_surface_of_one_body() {
    let mut graph = ConstructionGraph::new();
    let mut unit = |x: i32, y: i32, z: i32| {
        let block = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_position_ticks([x, y, z].into(), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(block)).unwrap() else {
            panic!("spawning a block reports its part");
        };
        part
    };
    // A welded 2×2 slab with a block standing on one corner, and a loose
    // block lying flush against it.
    let corner = unit(0, 50, 0);
    let right = unit(100, 50, 0);
    let back = unit(0, 50, 100);
    let far = unit(100, 50, 100);
    let cover = unit(0, 150, 0);
    let loose = unit(200, 50, 0);
    for (first, face, second) in [
        (corner, FaceKind::PositiveX, right),
        (back, FaceKind::PositiveX, far),
        (corner, FaceKind::PositiveZ, back),
        (right, FaceKind::PositiveZ, far),
        (corner, FaceKind::PositiveY, cover),
    ] {
        graph
            .apply(BuildCommand::Weld(mechanic_core::WeldSpec {
                first: FaceRef::part(first, face),
                second: FaceRef::part(second, face.opposite()),
            }))
            .unwrap();
    }
    let top = super::layer_target_from_hit(
        &graph,
        surface_hit(right, Vec3::new(0.25, 0.25, 0.05), FaceKind::PositiveY),
    )
    .unwrap();
    let members = top
        .members
        .iter()
        .map(|member| member.part)
        .collect::<Vec<_>>();
    assert_eq!(members[0], right, "the picked block leads");
    let mut sorted = members.clone();
    sorted.sort();
    let mut expected = vec![right, back, far];
    expected.sort();
    assert_eq!(sorted, expected, "covered and loose blocks stay out");

    let (layered, parts) = super::stage_layer(
        &graph,
        &top,
        0.05,
        ConstructionMaterial::Rubber,
        mechanic_core::MaterialAppearance::BAKED,
        PlacementBounds::Garage,
    )
    .unwrap();
    assert_eq!(parts.len(), 3);
    for part in [right, back, far] {
        assert!(layered.part(part).unwrap().is_layered());
    }
    for part in [corner, cover, loose] {
        assert!(!layered.part(part).unwrap().is_layered());
    }
    assert_eq!(layered.weld_count(), graph.weld_count());
}
