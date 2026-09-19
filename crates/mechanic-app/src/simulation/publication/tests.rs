mod frames;

use bevy::prelude::{IVec3, Quat, Vec3};
use mechanic_core::{
    BuildCommand, BuildOutcome, BuildPose, CuboidSpec, DimensionLinkId, DimensionLinkSpec,
    FaceKind, FaceRef, GridRotation, WeldSpec,
};
use mechanic_gpu::GpuTransform;

use crate::editor::build_actions::editor_part_is_static_or_pending;
use crate::simulation::publication::{
    rebuilt_body_states, world_mechanism_self_collisions, world_physics_result_is_current,
};
use crate::simulation::state::AppSimulation;
use mechanic_core::ConstructionGraph;

#[test]
fn only_the_latest_graph_and_foundation_revision_can_publish_physics() {
    let desired = (42, 7);

    assert!(world_physics_result_is_current(desired, desired));
    assert!(!world_physics_result_is_current((41, 7), desired));
    assert!(!world_physics_result_is_current((42, 6), desired));
}

#[test]
fn linked_vehicle_preserves_internal_mechanism_contacts() {
    let mut graph = ConstructionGraph::new();

    assert!(world_mechanism_self_collisions(&graph));
    graph
        .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
            DimensionLinkId(1),
            BuildPose::default(),
        )))
        .unwrap();
    assert!(world_mechanism_self_collisions(&graph));
}

#[test]
fn pending_editor_parts_remain_buildable_before_physics_publication() {
    let mut published = ConstructionGraph::new();
    let BuildOutcome::Spawned(published_part) = published
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let mut graph = published.clone();
    let BuildOutcome::Spawned(pending_part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::new(IVec3::X, GridRotation::default())).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let simulation = AppSimulation {
        creation: Some(published.compile().unwrap()),
        published_graph: published,
        ..Default::default()
    };

    assert!(!editor_part_is_static_or_pending(
        &graph,
        &simulation,
        published_part
    ));
    assert!(editor_part_is_static_or_pending(
        &graph,
        &simulation,
        pending_part
    ));
}

#[test]
fn published_static_parts_remain_buildable_and_moving_parts_do_not() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(static_part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::Spawned(moving_part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::new(IVec3::X, GridRotation::default())).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let simulation = AppSimulation {
        creation: Some(graph.compile_with_static_parts([static_part]).unwrap()),
        published_graph: graph.clone(),
        ..Default::default()
    };

    assert!(editor_part_is_static_or_pending(
        &graph,
        &simulation,
        static_part
    ));
    assert!(!editor_part_is_static_or_pending(
        &graph,
        &simulation,
        moving_part
    ));
}

#[test]
fn deleting_dimension_link_preserves_the_live_body_pose() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(block) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [2, 1, 1],
                BuildPose::new(IVec3::new(0, 1, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::Spawned(link) = graph
        .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
            DimensionLinkId(7),
            BuildPose::new(IVec3::new(2, 1, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(block, FaceKind::PositiveX),
            second: FaceRef::part(link, FaceKind::NegativeX),
        }))
        .unwrap();
    let previous_creation = graph.compile().unwrap();
    let live_rotation = Quat::from_rotation_y(0.7);
    let live_position = Vec3::new(4.0, 5.0, 6.0);
    let live_transform = GpuTransform {
        position: live_position.extend(0.0).to_array(),
        rotation: live_rotation.to_array(),
    };
    let previous = AppSimulation {
        creation: Some(previous_creation.clone()),
        published_graph: graph.clone(),
        previous_transforms: vec![live_transform],
        transforms: vec![live_transform],
        previous_snapshot_tick: 1,
        snapshot_tick: 2,
        live_state: Some(crate::simulation::state::LivePhysicsState {
            tick: 3,
            transforms: vec![GpuTransform {
                position: (live_position + Vec3::Y).extend(0.0).to_array(),
                ..live_transform
            }],
            velocities: vec![mechanic_gpu::GpuVelocity {
                linear: [2.0, 3.0, 4.0, 0.0],
                angular: [0.0, 0.0, 2.0, 0.0],
            }],
            coordinates: Vec::new(),
        }),
        ..Default::default()
    };

    graph.apply(BuildCommand::Remove(link)).unwrap();
    let creation = graph.compile().unwrap();
    let root_delta =
        creation.compounds[0].root_translation - previous_creation.compounds[0].root_translation;
    let expected_position = live_position + Vec3::Y + live_rotation * root_delta;
    let (transforms, velocities) = rebuilt_body_states(&creation, &graph, &previous);
    let rebuilt = transforms[0];

    assert!(Vec3::from_slice(&rebuilt.position[..3]).abs_diff_eq(expected_position, 1.0e-5));
    assert!(Quat::from_array(rebuilt.rotation).abs_diff_eq(live_rotation, 1.0e-5));
    let expected_velocity =
        Vec3::new(2.0, 3.0, 4.0) + Vec3::new(0.0, 0.0, 2.0).cross(live_rotation * root_delta);
    assert!(Vec3::from_slice(&velocities[0].linear[..3]).abs_diff_eq(expected_velocity, 1.0e-5));
    assert_eq!(
        velocities[0].angular.map(f32::to_bits),
        [0.0_f32, 0.0, 2.0, 0.0].map(f32::to_bits)
    );
}
#[test]
fn surviving_joint_state_follows_bearing_identity_after_removal() {
    use mechanic_core::BearingSpec;
    use mechanic_gpu::GpuMechanismCoordinate;
    let mut graph = ConstructionGraph::new();
    let mut joints = Vec::new();
    for z in [0_i16, 8] {
        let mut parts = Vec::new();
        for x in [0, 4] {
            let BuildOutcome::Spawned(part) = graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4; 3],
                        BuildPose::new(IVec3::new(x, 2, i32::from(z)), GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap()
            else {
                unreachable!()
            };
            parts.push(part);
        }
        let BuildOutcome::BearingAdded(bearing) = graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(parts[0], FaceKind::PositiveX),
                FaceRef::part(parts[1], FaceKind::NegativeX),
                Vec3::new(0.5, 0.5, f32::from(z) * 0.25),
                Vec3::X,
            )))
            .unwrap()
        else {
            unreachable!()
        };
        joints.push(bearing);
    }
    let compiled = graph.compile().unwrap();
    let coordinates = vec![
        GpuMechanismCoordinate {
            position: 7.0,
            velocity: 2.0,
        },
        GpuMechanismCoordinate {
            position: -9.0,
            velocity: -3.0,
        },
    ];
    let surviving = coordinates[compiled.loop_topology.bearing_coordinates[&joints[1]] as usize];
    let previous = AppSimulation {
        creation: Some(compiled),
        live_state: Some(crate::simulation::state::LivePhysicsState {
            tick: 10,
            transforms: Vec::new(),
            velocities: Vec::new(),
            coordinates,
        }),
        ..Default::default()
    };
    graph.apply(BuildCommand::RemoveBearing(joints[0])).unwrap();
    let rebuilt = graph.compile().unwrap();
    assert_eq!(
        crate::simulation::publication::rebuilt_mechanism_coordinates(
            &rebuilt,
            &previous,
            &[],
            &[]
        ),
        vec![surviving]
    );
}
#[test]
fn additions_removals_and_splits_inherit_the_live_rigid_velocity_field() {
    use mechanic_gpu::GpuVelocity;
    let mut original = ConstructionGraph::new();
    let mut parts = Vec::new();
    for x in 0..3 {
        let BuildOutcome::Spawned(part) = original
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1; 3],
                    BuildPose::new(IVec3::new(x, 8, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        if let Some(&last) = parts.last() {
            original
                .apply(BuildCommand::Weld(WeldSpec {
                    first: FaceRef::part(last, FaceKind::PositiveX),
                    second: FaceRef::part(part, FaceKind::NegativeX),
                }))
                .unwrap();
        }
        parts.push(part);
    }
    let compiled = original.compile().unwrap();
    let old_root = compiled.compounds[0].root_translation;
    let live_position = Vec3::new(2.0, 4.0, 6.0);
    let live_rotation = Quat::from_rotation_y(0.8);
    let linear = Vec3::new(1.0, 2.0, 3.0);
    let angular = Vec3::new(0.0, 0.0, 2.0);
    let previous = AppSimulation {
        creation: Some(compiled),
        published_graph: original.clone(),
        live_state: Some(crate::simulation::state::LivePhysicsState {
            tick: 20,
            transforms: vec![GpuTransform {
                position: live_position.extend(0.0).to_array(),
                rotation: live_rotation.to_array(),
            }],
            velocities: vec![GpuVelocity {
                linear: linear.extend(0.0).to_array(),
                angular: angular.extend(0.0).to_array(),
            }],
            coordinates: Vec::new(),
        }),
        ..Default::default()
    };
    for edit in 0..3 {
        let mut graph = original.clone();
        match edit {
            0 => {
                graph.apply(BuildCommand::Remove(parts[1])).unwrap();
            }
            1 => {
                graph.apply(BuildCommand::Remove(parts[2])).unwrap();
            }
            _ => {
                let BuildOutcome::Spawned(part) = graph
                    .apply(BuildCommand::Spawn(
                        CuboidSpec::new(
                            [1; 3],
                            BuildPose::new(IVec3::new(3, 8, 0), GridRotation::default()),
                        )
                        .unwrap(),
                    ))
                    .unwrap()
                else {
                    unreachable!()
                };
                graph
                    .apply(BuildCommand::Weld(WeldSpec {
                        first: FaceRef::part(parts[2], FaceKind::PositiveX),
                        second: FaceRef::part(part, FaceKind::NegativeX),
                    }))
                    .unwrap();
            }
        }
        let rebuilt = graph.compile().unwrap();
        assert_eq!(rebuilt.compounds.len(), if edit == 0 { 2 } else { 1 });
        let (transforms, velocities) = rebuilt_body_states(&rebuilt, &graph, &previous);
        for (body, compound) in rebuilt.compounds.iter().enumerate() {
            let displacement = live_rotation * (compound.root_translation - old_root);
            assert!(
                Vec3::from_slice(&transforms[body].position[..3])
                    .abs_diff_eq(live_position + displacement, 1.0e-5)
            );
            assert!(
                Vec3::from_slice(&velocities[body].linear[..3])
                    .abs_diff_eq(linear + angular.cross(displacement), 1.0e-5)
            );
            assert_eq!(
                velocities[body].angular.map(f32::to_bits),
                angular.extend(0.0).to_array().map(f32::to_bits)
            );
        }
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "share the moving-frame fixture across both joint kinds"
)]
fn promoted_closure_recovers_motion_in_a_rotated_world_frame() {
    use mechanic_core::BearingSpec;
    use mechanic_gpu::GpuVelocity;
    let mut graph = ConstructionGraph::new();
    let mut parts = Vec::new();
    for x in [0, 4] {
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4; 3],
                    BuildPose::new(IVec3::new(x, 2, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        parts.push(part);
    }
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(parts[0], FaceKind::PositiveX),
            FaceRef::part(parts[1], FaceKind::NegativeX),
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::X,
        )))
        .unwrap();
    let mut compiled = graph.compile().unwrap();
    let mut old = compiled.clone();
    old.loop_topology.bearing_coordinates.clear();
    old.loop_topology.tree_bearings.clear();
    old.bearings[0].coordinate_index = None;
    let previous = AppSimulation {
        creation: Some(old),
        live_state: Some(crate::simulation::state::LivePhysicsState {
            tick: 9,
            transforms: Vec::new(),
            velocities: Vec::new(),
            coordinates: Vec::new(),
        }),
        ..Default::default()
    };
    let rotation_a = Quat::from_rotation_y(0.9);
    let origin = Vec3::new(4.0, 8.0, 3.0);
    let linear_a = Vec3::new(2.0, 3.0, 4.0);
    let angular_a = Vec3::new(0.3, 0.2, 0.1);
    for linear in [false, true] {
        if linear {
            compiled.bearings[0].kind =
                mechanic_core::JointKind::Linear(mechanic_core::LinearBearing {
                    dimensions: mechanic_core::LinearBearingDimensions::default(),
                    mount_normal: Vec3::Y,
                    face: mechanic_core::CarriageFace::Top,
                });
        }
        let bearing = compiled.bearings[0];
        let rotation_b = if linear {
            rotation_a
        } else {
            rotation_a * Quat::from_rotation_x(0.7)
        };
        let axis = rotation_a * Vec3::X;
        let arm_a = rotation_a * bearing.local_anchor_a;
        let arm_b = rotation_b * bearing.local_anchor_b;
        let separation = if linear { axis * 0.2 } else { Vec3::ZERO };
        let position_b = origin + arm_a - arm_b + separation;
        let angular_b = angular_a + if linear { Vec3::ZERO } else { axis * 1.2 };
        let linear_b = linear_a + angular_a.cross(arm_a + separation) - angular_b.cross(arm_b)
            + if linear { axis * 0.3 } else { Vec3::ZERO };
        let mut transforms = vec![
            GpuTransform {
                position: [0.0; 4],
                rotation: Quat::IDENTITY.to_array()
            };
            2
        ];
        let mut velocities = vec![
            GpuVelocity {
                linear: [0.0; 4],
                angular: [0.0; 4]
            };
            2
        ];
        for (body, position, rotation, velocity, angular) in [
            (bearing.compound_a, origin, rotation_a, linear_a, angular_a),
            (
                bearing.compound_b,
                position_b,
                rotation_b,
                linear_b,
                angular_b,
            ),
        ] {
            transforms[body as usize] = GpuTransform {
                position: position.extend(0.0).to_array(),
                rotation: rotation.to_array(),
            };
            velocities[body as usize] = GpuVelocity {
                linear: velocity.extend(0.0).to_array(),
                angular: angular.extend(0.0).to_array(),
            };
        }
        let coordinates = crate::simulation::publication::rebuilt_mechanism_coordinates(
            &compiled,
            &previous,
            &transforms,
            &velocities,
        );
        assert!((coordinates[0].position - if linear { 0.2 } else { 0.7 }).abs() < 1.0e-5);
        assert!((coordinates[0].velocity - if linear { 0.3 } else { 1.2 }).abs() < 1.0e-5);
    }
}
