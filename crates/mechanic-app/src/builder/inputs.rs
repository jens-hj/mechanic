//! Placement of authored controls with true sub-block envelopes.

use super::bounds::{CollisionBox, boxes_overlap, part_collision_boxes, validate_world_bounds};
use super::faces::try_face_geometry_from_ref;
use super::{PlacementBounds, PlacementError, PlacementGrid, SurfaceHit};
use bevy::prelude::*;
use mechanic_core::{
    BuildCommand, BuildOutcome, ConstructionFrame, ConstructionGraph, FaceKind, FaceRef, PartSpec,
    WeldSpec,
};

/// Align the mounting face with the picked plane, keeping rotation within that plane.
pub(crate) fn input_surface_transform(
    graph: &ConstructionGraph,
    spec: PartSpec,
    hit: SurfaceHit,
    yaw: Quat,
    grid: PlacementGrid,
) -> Option<Transform> {
    let face = try_face_geometry_from_ref(hit.face, Some(graph))?;
    let step = grid.step_meters();
    let relative = hit.point - face.center;
    let point = hit.point
        + face.tangent_u
            * ((relative.dot(face.tangent_u) / step).round() * step - relative.dot(face.tangent_u))
        + face.tangent_v
            * ((relative.dot(face.tangent_v) / step).round() * step - relative.dot(face.tangent_v));
    Some(
        Transform::from_translation(point + face.normal * spec.size_meters().y * 0.5)
            .with_rotation(Quat::from_rotation_arc(Vec3::Y, face.normal) * yaw),
    )
}

pub(crate) fn stage_input_part(
    graph: &ConstructionGraph,
    spec: PartSpec,
    transform: Transform,
    support: Option<FaceRef>,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    validate_input_part(graph, spec, transform, bounds)?;
    let mut staged = graph.clone();
    let command = match spec {
        PartSpec::Dial(spec) => BuildCommand::SpawnDial(spec),
        PartSpec::Button(spec) => BuildCommand::SpawnButton(spec),
        _ => unreachable!("input placement has a physical input spec"),
    };
    let fail = |error: mechanic_core::GraphError| PlacementError::Graph(error.to_string());
    let BuildOutcome::Spawned(part) = staged.apply(command).map_err(fail)? else {
        unreachable!()
    };
    let frame = ConstructionFrame::new(transform.translation, transform.rotation)
        .and_then(|frame| staged.add_construction_frame(frame))
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    staged
        .assign_part_frame(part, frame)
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    if let Some(support) = support {
        staged
            .apply(BuildCommand::Weld(WeldSpec {
                first: support,
                second: FaceRef::part(part, FaceKind::NegativeY),
            }))
            .map_err(fail)?;
    }
    Ok(staged)
}

pub(crate) fn validate_input_part(
    graph: &ConstructionGraph,
    spec: PartSpec,
    transform: Transform,
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    let candidate = CollisionBox {
        center: transform.translation,
        rotation: transform.rotation,
        half: spec.size_meters() * 0.5,
    };
    let extent = (candidate.rotation * Vec3::X).abs() * candidate.half.x
        + (candidate.rotation * Vec3::Y).abs() * candidate.half.y
        + (candidate.rotation * Vec3::Z).abs() * candidate.half.z;
    validate_world_bounds(candidate.center - extent, candidate.center + extent, bounds)?;
    for (id, existing) in graph.parts() {
        let frame = graph.part_frame(id).expect("part has frame");
        for shape in part_collision_boxes(*existing) {
            if boxes_overlap(
                candidate,
                CollisionBox {
                    center: frame.point(shape.center),
                    rotation: frame.rotation() * shape.rotation,
                    half: shape.half,
                },
            ) {
                return Err(PlacementError::OverlapsPart(id));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mechanic_core::{BuildPose, ButtonSpec, CreationDocument, DialSpec, InputSize};

    #[test]
    fn physical_inputs_mount_on_horizontal_vertical_and_oblique_faces() {
        for rotation in [
            Quat::IDENTITY,
            Quat::from_rotation_z(0.73),
            Quat::from_euler(EulerRot::XYZ, 0.4, -0.6, 0.9),
        ] {
            let mut graph = ConstructionGraph::new();
            let BuildOutcome::Spawned(support) = graph
                .apply(BuildCommand::Spawn(
                    mechanic_core::CuboidSpec::new([4; 3], BuildPose::default()).unwrap(),
                ))
                .unwrap()
            else {
                panic!("support")
            };
            let frame = graph
                .add_construction_frame(
                    ConstructionFrame::new(Vec3::Y * (crate::garage::BUILD_MIN_Y + 3.0), rotation)
                        .unwrap(),
                )
                .unwrap();
            graph.assign_part_frame(support, frame).unwrap();
            for kind in [
                FaceKind::PositiveY,
                FaceKind::NegativeY,
                FaceKind::PositiveX,
                FaceKind::NegativeZ,
            ] {
                let face = FaceRef::part(support, kind);
                let geometry = try_face_geometry_from_ref(face, Some(&graph)).unwrap();
                let hit = SurfaceHit {
                    point: geometry.center,
                    face,
                    distance: 1.0,
                };
                for size in InputSize::ALL {
                    for spec in [
                        PartSpec::Dial(DialSpec::new(size, BuildPose::default())),
                        PartSpec::Button(ButtonSpec::new(size, BuildPose::default())),
                    ] {
                        for turns in 0_u8..4 {
                            let transform = input_surface_transform(
                                &graph,
                                spec,
                                hit,
                                Quat::from_rotation_y(
                                    f32::from(turns) * std::f32::consts::FRAC_PI_2,
                                ),
                                PlacementGrid::default(),
                            )
                            .unwrap();
                            assert!(
                                (transform.rotation * Vec3::Y).abs_diff_eq(geometry.normal, 1e-5)
                            );
                            assert!(
                                (transform.translation
                                    - geometry.normal * spec.size_meters().y * 0.5)
                                    .abs_diff_eq(hit.point, 1e-5)
                            );
                            let staged = stage_input_part(
                                &graph,
                                spec,
                                transform,
                                Some(face),
                                PlacementBounds::GarageBuild,
                            )
                            .unwrap();
                            assert_eq!(staged.welds().count(), 1);
                            staged.compile().unwrap();
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn physical_inputs_snap_to_the_selected_surface_grid() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(support) = graph
            .apply(BuildCommand::Spawn(
                mechanic_core::CuboidSpec::new([4; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            panic!("support")
        };
        let frame = graph
            .add_construction_frame(
                ConstructionFrame::new(Vec3::new(1.3, 8.0, -0.7), Quat::from_rotation_z(0.73))
                    .unwrap(),
            )
            .unwrap();
        graph.assign_part_frame(support, frame).unwrap();
        for kind in [
            FaceKind::PositiveY,
            FaceKind::PositiveX,
            FaceKind::NegativeY,
        ] {
            let face = FaceRef::part(support, kind);
            let geometry = try_face_geometry_from_ref(face, Some(&graph)).unwrap();
            for grid in [
                PlacementGrid::Centimetres25,
                PlacementGrid::Centimetres5,
                PlacementGrid::Centimetres1,
            ] {
                let step = grid.step_meters();
                for size in InputSize::ALL {
                    for spec in [
                        PartSpec::Dial(DialSpec::new(size, BuildPose::default())),
                        PartSpec::Button(ButtonSpec::new(size, BuildPose::default())),
                    ] {
                        for offset in [0.1, 0.4] {
                            let hit = SurfaceHit {
                                face,
                                distance: 1.0,
                                point: geometry.center
                                    + geometry.tangent_u * (step * (2.0 + offset))
                                    - geometry.tangent_v * (step * (1.0 + offset)),
                            };
                            let transform =
                                input_surface_transform(&graph, spec, hit, Quat::IDENTITY, grid)
                                    .unwrap();
                            let mounting_point = transform.translation
                                - geometry.normal * spec.size_meters().y * 0.5;
                            let expected = geometry.center + geometry.tangent_u * (2.0 * step)
                                - geometry.tangent_v * step;
                            assert!(mounting_point.abs_diff_eq(expected, 1e-5));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn all_six_inputs_mount_flush_rotate_compile_and_round_trip_at_actual_size() {
        for size in InputSize::ALL {
            for spec in [
                PartSpec::Dial(DialSpec::new(size, BuildPose::default())),
                PartSpec::Button(ButtonSpec::new(size, BuildPose::default())),
            ] {
                for turns in 0_u8..4 {
                    let transform =
                        Transform::from_translation(Vec3::Y * spec.size_meters().y * 0.5)
                            .with_rotation(Quat::from_rotation_y(
                                f32::from(turns) * std::f32::consts::FRAC_PI_2,
                            ));
                    let graph = stage_input_part(
                        &ConstructionGraph::new(),
                        spec,
                        transform,
                        Some(FaceRef::ground()),
                        PlacementBounds::Garage,
                    )
                    .unwrap();
                    assert_eq!(graph.welds().count(), 1);
                    let creation = graph.compile().unwrap();
                    let collider = &creation.colliders[0];
                    let mechanic_core::ColliderShape::Cuboid { half_extents, .. } = collider.shape
                    else {
                        panic!("input envelope is a simple box")
                    };
                    assert!(half_extents.abs_diff_eq(spec.size_meters() * 0.5, 1e-6));
                    let document = CreationDocument::from_graph(&graph, "Input", &[]);
                    let loaded = document.clone().into_graph().unwrap();
                    assert_eq!(
                        CreationDocument::from_graph(&loaded.graph, "Input", &[]),
                        document
                    );
                }
            }
        }
    }
}
