#![allow(clippy::float_cmp)] // Holds and pure lifts preserve these rows exactly.

use super::*;
use bevy::math::DVec3;
use mechanic_core::{BuildCommand, BuildPose, ConstructionGraph, CuboidSpec, GridRotation};

fn creation() -> CompiledCreation {
    let mut graph = ConstructionGraph::new();
    for x in [0, 16, 32] {
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::new(IVec3::new(x, 8, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
    }
    graph.compile().unwrap()
}

fn pose(position: Vec3, rotation: Quat) -> GpuTransform {
    GpuTransform {
        position: position.extend(0.0).to_array(),
        rotation: rotation.to_array(),
    }
}

#[allow(clippy::unnecessary_wraps)]
fn flat_terrain(center: Vec3, radius: f32) -> Option<f32> {
    Some(radius - center.y)
}

#[test]
fn garage_pose_keeps_link_pivot_at_target_and_rotates_all_held_bodies_cardinally() {
    let creation = creation();
    // The link center is offset from its body's mass center.
    let pivot = creation.compounds[0].root_translation + Vec3::X;
    let target = Vec3::new(10.0, 4.25, -5.0);
    let unrelated = pose(Vec3::splat(50.0), Quat::from_rotation_x(0.3));
    let current = vec![unrelated; 3];
    let result = default_poses(&creation, &current, &[true, true, false], pivot, target, 1);
    let rotation = Quat::from_array(result[0].rotation);
    let link_center =
        position(result[0]) + rotation * (pivot - creation.compounds[0].root_translation);
    assert!(link_center.abs_diff_eq(target, 1.0e-5));
    assert!((rotation * Vec3::Y).abs_diff_eq(Vec3::Y, 1.0e-5));
    assert!((rotation * Vec3::X).abs_diff_eq(Vec3::NEG_Z, 1.0e-5));
    assert!((position(result[1]) - position(result[0])).abs_diff_eq(Vec3::NEG_Z * 4.0, 1.0e-5));
    assert_eq!(result[2], unrelated);
}

#[test]
fn swept_path_rejects_ceiling_and_wall_between_clear_endpoints() {
    let creation = creation();
    let held = [true, false, false];
    for direction in [Vec3::Y, Vec3::X] {
        let start = vec![pose(Vec3::Y, Quat::IDENTITY); 3];
        let end = vec![pose(Vec3::Y + direction * 2.0, Quat::IDENTITY); 3];
        let obstacle = Obb {
            center: Vec3::Y + direction,
            orientation: Quat::IDENTITY,
            half_extents: Vec3::splat(0.2),
        };
        assert!(endpoint_clear(
            &creation,
            &held,
            &start,
            &[obstacle],
            &flat_terrain
        ));
        assert!(endpoint_clear(
            &creation,
            &held,
            &end,
            &[obstacle],
            &flat_terrain
        ));
        assert!(!path_clear(
            &creation,
            &held,
            &start,
            &end,
            &[obstacle],
            &flat_terrain
        ));
        assert!(path_clear(
            &creation,
            &held,
            &start,
            &end,
            &[],
            &flat_terrain
        ));
    }
}

#[test]
fn overturned_offset_geometry_lifts_before_leveling() {
    let mut creation = creation();
    creation.colliders[0].local_center = Vec3::NEG_X;
    let held = [true, false, false];
    let start = vec![pose(Vec3::Y * 0.5, Quat::from_xyzw(0.0, 0.0, 1.0, 0.0)); 3];
    let end = vec![pose(Vec3::Y * 1.5, Quat::IDENTITY); 3];
    assert!(!path_clear(
        &creation,
        &held,
        &start,
        &end,
        &[],
        &flat_terrain
    ));
    let waypoints = plan(&creation, &held, &start, &end, &[], &flat_terrain).unwrap();
    assert_eq!(waypoints.len(), 2);
    assert_eq!(waypoints[0][0].rotation, start[0].rotation);
    assert!(position(waypoints[0][0]).y > position(start[0]).y);
    assert_eq!(waypoints[0][1], start[1]);
    assert_eq!(waypoints[1], end);
}

#[test]
fn lowering_into_ground_or_planning_without_terrain_is_refused() {
    let creation = creation();
    let held = [true, false, false];
    let start = vec![pose(Vec3::Y * 0.25, Quat::IDENTITY); 3];
    let below = vec![pose(Vec3::ZERO, Quat::IDENTITY); 3];
    assert!(endpoint_clear(&creation, &held, &start, &[], &flat_terrain));
    assert!(plan(&creation, &held, &start, &below, &[], &flat_terrain).is_none());
    assert!(plan(&creation, &held, &start, &start, &[], &|_, _| None).is_none());
}

#[test]
fn overlay_replaces_stale_held_readbacks_without_rewinding_neighbors() {
    let stale = pose(Vec3::Y, Quat::IDENTITY);
    let prescribed = pose(Vec3::Y * 3.0, Quat::from_rotation_y(0.5));
    let neighbor = pose(Vec3::X * 4.0, Quat::IDENTITY);
    let moving = mechanic_gpu::GpuVelocity {
        linear: [1.0, 2.0, 3.0, 0.0],
        angular: [0.0, 4.0, 0.0, 0.0],
    };
    let revision = Some((7, 2));
    let frozen = DimensionFreeze {
        record: Some(FrozenCreationDoc {
            link: DimensionLinkId(1),
            target: mechanic_world::WorldPosition::default(),
            heading: 0,
            construction_generation: 7,
        }),
        revision,
        held: vec![true, false],
        poses: vec![prescribed, stale],
        ..Default::default()
    };
    let mut simulation = AppSimulation {
        world_revision: revision,
        transforms: vec![stale, neighbor],
        live_state: Some(crate::LivePhysicsState {
            tick: 15,
            transforms: vec![stale, neighbor],
            velocities: vec![moving; 2],
            coordinates: Vec::new(),
        }),
        ..Default::default()
    };
    frozen.overlay(&mut simulation);
    assert_eq!(simulation.transforms, vec![prescribed, neighbor]);
    let live = simulation.live_state.as_ref().unwrap();
    assert_eq!(live.transforms, simulation.transforms);
    assert_eq!(live.velocities[0].linear, [0.0; 4]);
    assert_eq!(live.velocities[0].angular, [0.0; 4]);
    assert_eq!(live.velocities[1], moving);
    assert_eq!(live.tick, 15);
    assert_eq!(simulation.pose_revision, 1);
    assert!(simulation.render_dirty);
    frozen.overlay(&mut simulation);
    assert_eq!(simulation.pose_revision, 1);
    simulation.world_revision = Some((8, 2));
    simulation.transforms[0] = stale;
    frozen.overlay(&mut simulation);
    assert_eq!(simulation.transforms[0], stale);
}

fn linked_candidate() -> AppSimulation {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::SpawnDimensionLink(
            mechanic_core::DimensionLinkSpec::new(DimensionLinkId(1), BuildPose::default()),
        ))
        .unwrap();
    let mut creation = graph.compile().unwrap();
    creation.compounds[0].root_translation = Vec3::ZERO;
    creation.colliders[0].local_center = Vec3::NEG_X;
    creation.colliders[0].shape = ColliderShape::Cuboid {
        local_rotation: Quat::IDENTITY,
        half_extents: Vec3::splat(0.1),
    };
    AppSimulation {
        published_graph: graph,
        creation: Some(creation),
        world_revision: Some((8, 2)),
        transforms: vec![pose(Vec3::Y * 0.5, Quat::from_xyzw(0.0, 0.0, 1.0, 0.0))],
        ..Default::default()
    }
}

fn aligning_freeze(candidate: &AppSimulation) -> DimensionFreeze {
    DimensionFreeze {
        record: Some(FrozenCreationDoc {
            link: DimensionLinkId(1),
            target: mechanic_world::WorldPosition(DVec3::Y * 1.5),
            heading: 0,
            construction_generation: 7,
        }),
        revision: Some((7, 2)),
        held: vec![true],
        poses: candidate.transforms.clone(),
        waypoints: VecDeque::from([candidate.transforms.clone()]),
        ..Default::default()
    }
}

#[test]
fn rejected_publication_leaves_original_hold_and_repeat_progress_untouched() {
    let candidate = linked_candidate();
    let mut frozen = aligning_freeze(&candidate);
    assert_eq!(frozen.repeat.advance(0.0, true, false), 1);
    assert_eq!(frozen.repeat.advance(0.2, true, false), 0);
    let record = frozen.record;
    let poses = frozen.poses.clone();
    let waypoints = frozen.waypoints.clone();
    let mut expected_repeat = frozen.repeat.clone();
    assert!(
        frozen
            .prepare_with_environment(&candidate, None, &|p| p.0.as_vec3(), &|_, _| Some(1.0))
            .is_err()
    );
    assert_eq!(frozen.record, record);
    assert_eq!(frozen.revision, Some((7, 2)));
    assert_eq!(frozen.poses, poses);
    assert_eq!(frozen.waypoints, waypoints);
    assert_eq!(
        frozen.repeat.advance(0.1, true, false),
        expected_repeat.advance(0.1, true, false)
    );
}

#[test]
fn candidate_publication_replans_clearance_lift_before_leveling() {
    let candidate = linked_candidate();
    let frozen = aligning_freeze(&candidate);
    let prepared = frozen
        .prepare_with_environment(&candidate, None, &|p| p.0.as_vec3(), &flat_terrain)
        .unwrap();
    assert_eq!(prepared.revision, candidate.world_revision);
    assert_eq!(prepared.waypoints.len(), 2);
    assert_eq!(prepared.waypoints[0][0].rotation, frozen.poses[0].rotation);
    assert!(position(prepared.waypoints[0][0]).y > position(frozen.poses[0]).y);
    assert_eq!(frozen.revision, Some((7, 2)));
    assert_eq!(frozen.waypoints.len(), 1);
}

#[test]
fn saved_global_target_is_restored_exactly_before_first_tick() {
    let candidate = linked_candidate();
    let origin = DVec3::new(12_000.0, 500.0, -24_000.0);
    let record = FrozenCreationDoc {
        link: DimensionLinkId(1),
        target: mechanic_world::WorldPosition(origin + DVec3::new(10.0, 2.25, -4.0)),
        heading: 1,
        construction_generation: 9,
    };
    let prepared = DimensionFreeze::default()
        .prepare_with_environment(
            &candidate,
            Some(record),
            &|position| (position.0 - origin).as_vec3(),
            &flat_terrain,
        )
        .unwrap();
    assert_eq!(candidate.snapshot_tick, 0);
    assert!(prepared.waypoints.is_empty());
    assert!(position(prepared.poses[0]).abs_diff_eq(Vec3::new(10.0, 2.25, -4.0), 1.0e-5));
    assert!(
        (Quat::from_array(prepared.poses[0].rotation) * Vec3::X).abs_diff_eq(Vec3::NEG_Z, 1.0e-5)
    );
    assert_eq!(prepared.record, Some(record));
}

#[test]
fn held_bodies_with_clear_endpoints_cannot_pass_through_each_other() {
    let creation = creation();
    let held = [true, true, false];
    let start = vec![
        pose(Vec3::new(-1.0, 2.0, 0.0), Quat::IDENTITY),
        pose(Vec3::new(1.0, 2.0, 0.0), Quat::IDENTITY),
        pose(Vec3::splat(50.0), Quat::IDENTITY),
    ];
    let end = vec![start[1], start[0], start[2]];
    assert!(endpoint_clear(&creation, &held, &start, &[], &flat_terrain));
    assert!(endpoint_clear(&creation, &held, &end, &[], &flat_terrain));
    assert!(!path_clear(
        &creation,
        &held,
        &start,
        &end,
        &[],
        &flat_terrain
    ));
}

#[test]
fn held_default_overlap_is_refused_but_joint_collision_suppression_is_honored() {
    let mut creation = creation();
    let held = [true, true, false];
    let poses = vec![pose(Vec3::Y * 2.0, Quat::IDENTITY); 3];
    assert!(!endpoint_clear(
        &creation,
        &held,
        &poses,
        &[],
        &flat_terrain
    ));
    assert!(plan(&creation, &held, &poses, &poses, &[], &flat_terrain).is_none());
    creation.collision_suppression = vec![[0, 1]];
    assert!(endpoint_clear(&creation, &held, &poses, &[], &flat_terrain));
    assert!(path_clear(
        &creation,
        &held,
        &poses,
        &poses,
        &[],
        &flat_terrain
    ));
}

#[test]
fn close_held_bodies_translate_and_slide_without_bounding_sphere_false_blocks() {
    let creation = creation();
    let held = [true, true, false];
    let start = vec![
        pose(Vec3::Y * 2.0, Quat::IDENTITY),
        pose(Vec3::new(0.25, 2.0, 0.0), Quat::IDENTITY),
        pose(Vec3::splat(50.0), Quat::IDENTITY),
    ];
    let (a_center, a_radius) = sphere(&creation.colliders[0], start[0]);
    let (b_center, b_radius) = sphere(&creation.colliders[1], start[1]);
    assert!(a_center.distance(b_center) < a_radius + b_radius);
    assert!(endpoint_clear(&creation, &held, &start, &[], &flat_terrain));
    let mut sliding = start.clone();
    sliding[1].position[2] += 2.0;
    assert!(path_clear(
        &creation,
        &held,
        &start,
        &sliding,
        &[],
        &flat_terrain
    ));
    let mut translated = start.clone();
    for body in &mut translated[..2] {
        body.position[1] += 4.0;
    }
    assert!(path_clear(
        &creation,
        &held,
        &start,
        &translated,
        &[],
        &flat_terrain
    ));
}

#[test]
fn rotating_held_offset_geometry_cannot_sweep_through_another_held_body() {
    let mut creation = creation();
    creation.colliders[0].local_center = Vec3::X;
    let held = [true, true, false];
    let start = vec![
        pose(Vec3::Y * 2.0, Quat::IDENTITY),
        pose(Vec3::Y * 3.0, Quat::IDENTITY),
        pose(Vec3::splat(50.0), Quat::IDENTITY),
    ];
    let mut end = start.clone();
    end[0].rotation = Quat::from_xyzw(0.0, 0.0, 1.0, 0.0).to_array();
    assert!(endpoint_clear(&creation, &held, &start, &[], &flat_terrain));
    assert!(endpoint_clear(&creation, &held, &end, &[], &flat_terrain));
    assert!(!path_clear(
        &creation,
        &held,
        &start,
        &end,
        &[],
        &flat_terrain
    ));
}

#[test]
fn wide_plate_can_lower_to_five_centimetres_above_terrain() {
    let mut creation = creation();
    creation.colliders[0].shape = ColliderShape::Cuboid {
        local_rotation: Quat::IDENTITY,
        half_extents: Vec3::new(1.0, 0.125, 1.0),
    };
    creation.colliders[0].local_center = Vec3::ZERO;
    let held = [true, false, false];
    let start = vec![pose(Vec3::Y * 0.425, Quat::IDENTITY); 3];
    let end = vec![pose(Vec3::Y * 0.175, Quat::IDENTITY); 3];
    assert!(endpoint_clear(&creation, &held, &end, &[], &flat_terrain));
    assert!(path_clear(
        &creation,
        &held,
        &start,
        &end,
        &[],
        &flat_terrain
    ));
    let below = vec![pose(Vec3::Y * 0.17, Quat::IDENTITY); 3];
    assert!(!endpoint_clear(
        &creation,
        &held,
        &below,
        &[],
        &flat_terrain
    ));
}

#[test]
fn terrain_clearance_checks_face_interiors_and_rotated_lowest_points() {
    let mut creation = creation();
    let collider = &mut creation.colliders[0];
    collider.shape = ColliderShape::Cuboid {
        local_rotation: Quat::IDENTITY,
        half_extents: Vec3::new(1.0, 0.125, 1.0),
    };
    collider.local_center = Vec3::ZERO;
    let level = pose(Vec3::Y * 0.175, Quat::IDENTITY);
    let mound =
        |p: Vec3, radius: f32| Some(radius - p.y + (0.1 - p.x.abs().max(p.z.abs())).max(0.0));
    assert!(!terrain_clear(collider, level, 0.05, &mound));
    let tilted = pose(Vec3::Y * 0.175, Quat::from_rotation_z(0.1));
    assert!(!terrain_clear(collider, tilted, 0.05, &flat_terrain));
}

#[test]
fn last_downward_step_stops_at_clearance_between_block_heights() {
    let lift = minimum_clear_lift(0.25, |lift| -0.1 + lift >= 0.05).unwrap();
    assert!((lift - 0.15).abs() < 0.0001);
    assert!(-0.1 + lift >= 0.05);
    assert!(minimum_clear_lift(0.25, |lift| lift >= 0.3).is_none());
}

#[test]
fn triangle_clearance_replaces_dense_queries_for_a_wide_plate() {
    let mut creation = creation();
    let collider = &mut creation.colliders[0];
    collider.local_center = Vec3::ZERO;
    collider.shape = ColliderShape::Cuboid {
        local_rotation: Quat::IDENTITY,
        half_extents: Vec3::new(1.0, 0.125, 1.0),
    };
    let target = pose(Vec3::Y * 0.175, Quat::IDENTITY);
    let calls = std::cell::Cell::new(0);
    assert!(sampled_terrain_clear(collider, target, 0.05, &|p, r| {
        calls.set(calls.get() + 1);
        flat_terrain(p, r)
    }));
    let geometry = crate::live_weld::geometry(collider, target).unwrap();
    let triangles = [
        [
            Vec3::new(-2.0, 0.0, -2.0),
            Vec3::new(2.0, 0.0, -2.0),
            Vec3::new(2.0, 0.0, 2.0),
        ],
        [
            Vec3::new(-2.0, 0.0, -2.0),
            Vec3::new(2.0, 0.0, 2.0),
            Vec3::new(-2.0, 0.0, 2.0),
        ],
    ];
    assert!(
        triangles
            .iter()
            .all(|&t| triangle_clear(&geometry, t, 0.05))
    );
    assert!(calls.get() > 10_000, "old query count: {}", calls.get());
    eprintln!(
        "wide plate: {} old terrain searches replaced by 2 triangle tests",
        calls.get()
    );
    let mound = [
        Vec3::new(-0.1, 0.0, 0.0),
        Vec3::new(0.0, 0.02, 0.1),
        Vec3::new(0.1, 0.0, 0.0),
    ];
    assert!(!triangle_clear(&geometry, mound, 0.05));
}

#[test]
fn weld_destination_decides_the_hold_and_dimension_links_survive() {
    for (source_held, destination_held) in
        [(false, false), (true, false), (false, true), (true, true)]
    {
        let mut graph = ConstructionGraph::new();
        let mut parts = Vec::new();
        for (id, x) in [(1, 0), (2, 8)] {
            let mechanic_core::BuildOutcome::Spawned(part) = graph
                .apply(BuildCommand::SpawnDimensionLink(
                    mechanic_core::DimensionLinkSpec::new(
                        DimensionLinkId(id),
                        BuildPose::new(IVec3::new(x, 28, 0), GridRotation::default()),
                    ),
                ))
                .unwrap()
            else {
                panic!("spawn expected")
            };
            parts.push(part);
        }
        let creation = graph.compile().unwrap();
        let transforms = creation
            .compounds
            .iter()
            .map(|b| pose(b.root_translation, b.root_rotation))
            .collect::<Vec<_>>();
        let previous = AppSimulation {
            creation: Some(creation),
            published_graph: graph.clone(),
            transforms: transforms.clone(),
            world_revision: Some((0, 0)),
            ..default()
        };
        let held = previous
            .creation
            .as_ref()
            .unwrap()
            .compounds
            .iter()
            .map(|body| {
                if body.source_parts.contains(&parts[0]) {
                    source_held
                } else {
                    destination_held
                }
            })
            .collect();
        let frozen = DimensionFreeze {
            record: (source_held || destination_held).then_some(FrozenCreationDoc {
                link: DimensionLinkId(if destination_held { 2 } else { 1 }),
                target: mechanic_world::WorldPosition::default(),
                heading: 0,
                construction_generation: 0,
            }),
            held,
            poses: transforms,
            revision: previous.world_revision,
            ..default()
        };
        graph
            .apply(BuildCommand::RigidLink(mechanic_core::RigidLinkSpec {
                first: parts[0],
                second: parts[1],
            }))
            .unwrap();
        let creation = graph.compile().unwrap();
        let transforms = creation
            .compounds
            .iter()
            .map(|b| pose(b.root_translation, b.root_rotation))
            .collect();
        let replacement = AppSimulation {
            creation: Some(creation),
            published_graph: graph,
            transforms,
            world_revision: Some((1, 0)),
            ..default()
        };
        let result = frozen.for_weld(&previous, &replacement, parts[1], &[parts[0]]);
        assert_eq!(result.record.is_some(), destination_held);
        if destination_held {
            assert_eq!(result.held, vec![true]);
        }
        assert!(
            replacement
                .published_graph
                .dimension_link(DimensionLinkId(1))
                .is_some()
        );
        assert!(
            replacement
                .published_graph
                .dimension_link(DimensionLinkId(2))
                .is_some()
        );
    }
}
