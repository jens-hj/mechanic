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
fn swept_path_rejects_terrain_ceiling_and_wall_between_clear_endpoints() {
    let creation = creation();
    let held = [true, false, false];
    for direction in [Vec3::Y, Vec3::X] {
        let start = vec![pose(Vec3::Y, Quat::IDENTITY); 3];
        let end = vec![pose(Vec3::Y + direction * 2.0, Quat::IDENTITY); 3];
        // Flat ground with a 40 cm terrain block halfway along the path.
        let blocked = |center: Vec3, radius: f32| {
            let outside = ((center - (Vec3::Y + direction)).abs() - Vec3::splat(0.2))
                .max(Vec3::ZERO)
                .length();
            Some((radius - center.y).max(radius - outside))
        };
        assert!(endpoint_clear(&creation, &held, &start, &blocked));
        assert!(endpoint_clear(&creation, &held, &end, &blocked));
        assert!(!path_clear(&creation, &held, &start, &end, &blocked));
        assert!(path_clear(&creation, &held, &start, &end, &flat_terrain));
    }
}

#[test]
fn bodies_outside_the_frozen_creation_never_block_it() {
    let creation = creation();
    let held = [true, false, false];
    // An unheld body sits right on the held body's path and at its target.
    let start = vec![
        pose(Vec3::Y, Quat::IDENTITY),
        pose(Vec3::Y + Vec3::X, Quat::IDENTITY),
        pose(Vec3::Y + Vec3::X * 2.0, Quat::IDENTITY),
    ];
    let mut end = start.clone();
    end[0] = start[2];
    assert!(plan(&creation, &held, &start, &end, &flat_terrain).is_some());
}

#[test]
fn overturned_offset_geometry_lifts_before_leveling() {
    let mut creation = creation();
    creation.colliders[0].local_center = Vec3::NEG_X;
    let held = [true, false, false];
    let start = vec![pose(Vec3::Y * 0.5, Quat::from_xyzw(0.0, 0.0, 1.0, 0.0)); 3];
    let end = vec![pose(Vec3::Y * 1.5, Quat::IDENTITY); 3];
    assert!(!path_clear(&creation, &held, &start, &end, &flat_terrain));
    let waypoints = plan(&creation, &held, &start, &end, &flat_terrain).unwrap();
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
    assert!(endpoint_clear(&creation, &held, &start, &flat_terrain));
    assert!(plan(&creation, &held, &start, &below, &flat_terrain).is_none());
    assert!(plan(&creation, &held, &start, &start, &|_, _| None).is_none());
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
    assert!(endpoint_clear(&creation, &held, &start, &flat_terrain));
    assert!(endpoint_clear(&creation, &held, &end, &flat_terrain));
    assert!(!path_clear(&creation, &held, &start, &end, &flat_terrain));
}

#[test]
fn tangled_held_bodies_may_pass_through_each_other_to_come_apart() {
    let creation = creation();
    let held = [true, true, false];
    // The first cube starts buried in the second and ends on its far side.
    let start = vec![
        pose(Vec3::new(-0.1, 2.0, 0.0), Quat::IDENTITY),
        pose(Vec3::new(0.0, 2.0, 0.0), Quat::IDENTITY),
        pose(Vec3::splat(50.0), Quat::IDENTITY),
    ];
    let end = vec![
        pose(Vec3::new(1.0, 2.0, 0.0), Quat::IDENTITY),
        pose(Vec3::new(-1.0, 2.0, 0.0), Quat::IDENTITY),
        start[2],
    ];
    assert!(plan(&creation, &held, &start, &end, &flat_terrain).is_some());
}

#[test]
fn held_default_overlap_is_refused_but_joint_collision_suppression_is_honored() {
    let mut creation = creation();
    let held = [true, true, false];
    let poses = vec![pose(Vec3::Y * 2.0, Quat::IDENTITY); 3];
    assert!(!endpoint_clear(&creation, &held, &poses, &flat_terrain));
    assert!(plan(&creation, &held, &poses, &poses, &flat_terrain).is_none());
    creation.collision_suppression = vec![[0, 1]];
    assert!(endpoint_clear(&creation, &held, &poses, &flat_terrain));
    assert!(path_clear(&creation, &held, &poses, &poses, &flat_terrain));
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
    assert!(endpoint_clear(&creation, &held, &start, &flat_terrain));
    let mut sliding = start.clone();
    sliding[1].position[2] += 2.0;
    assert!(path_clear(
        &creation,
        &held,
        &start,
        &sliding,
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
    assert!(endpoint_clear(&creation, &held, &start, &flat_terrain));
    assert!(endpoint_clear(&creation, &held, &end, &flat_terrain));
    assert!(!path_clear(&creation, &held, &start, &end, &flat_terrain));
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
    assert!(endpoint_clear(&creation, &held, &end, &flat_terrain));
    assert!(path_clear(&creation, &held, &start, &end, &flat_terrain));
    let below = vec![pose(Vec3::Y * 0.17, Quat::IDENTITY); 3];
    assert!(!endpoint_clear(&creation, &held, &below, &flat_terrain));
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

#[test]
fn visual_snapshot_contains_only_held_bodies_and_tracks_prescribed_pose() {
    let creation = creation();
    let poses = vec![pose(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY); 3];
    let simulation = AppSimulation {
        creation: Some(creation),
        world_revision: Some((1, 2)),
        ..default()
    };
    let mut hold = DimensionFreeze {
        record: Some(FrozenCreationDoc {
            link: DimensionLinkId(1),
            target: mechanic_world::WorldPosition(DVec3::ZERO),
            heading: 0,
            construction_generation: 0,
        }),
        revision: Some((1, 2)),
        held: vec![true, false, false],
        poses,
        ..default()
    };
    let first = hold.visual_snapshot(&simulation).unwrap();
    assert_eq!(first.link, DimensionLinkId(1));
    assert!((first.max - first.min).abs_diff_eq(Vec3::splat(0.25), 1e-5));
    hold.poses[0].position[1] += 0.125;
    let second = hold.visual_snapshot(&simulation).unwrap();
    assert!((second.min - first.min).abs_diff_eq(Vec3::Y * 0.125, 1e-5));
    hold.reset();
    assert!(hold.visual_snapshot(&simulation).is_none());
}

fn exhaustive_pairs(creation: &CompiledCreation, held: &[bool]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (a, first) in creation.colliders.iter().enumerate() {
        if !held[first.compound_index as usize] {
            continue;
        }
        for (b, second) in creation.colliders.iter().enumerate().skip(a + 1) {
            if !held[second.compound_index as usize]
                || first.compound_index == second.compound_index
            {
                continue;
            }
            let pair = [
                first.compound_index.min(second.compound_index),
                first.compound_index.max(second.compound_index),
            ];
            if creation.collision_suppression.binary_search(&pair).is_err() {
                pairs.push((a, b));
            }
        }
    }
    pairs
}

fn exhaustive_endpoint_clear(
    creation: &CompiledCreation,
    held: &[bool],
    poses: &[GpuTransform],
) -> bool {
    exhaustive_pairs(creation, held).into_iter().all(|(a, b)| {
        let first = &creation.colliders[a];
        let second = &creation.colliders[b];
        let Ok(a) = crate::live_weld::geometry(first, poses[first.compound_index as usize]) else {
            return false;
        };
        let Ok(b) = crate::live_weld::geometry(second, poses[second.compound_index as usize])
        else {
            return false;
        };
        crate::live_weld::penetration(&a, &b) <= 1.0e-4
    })
}

fn exhaustive_path_clear(
    creation: &CompiledCreation,
    held: &[bool],
    start: &[GpuTransform],
    end: &[GpuTransform],
) -> bool {
    for (a_index, b_index) in exhaustive_pairs(creation, held) {
        let a = &creation.colliders[a_index];
        let b = &creation.colliders[b_index];
        let a_body = a.compound_index as usize;
        let b_body = b.compound_index as usize;
        let relative_translation = (position(end[b_body]) - position(start[b_body]))
            - (position(end[a_body]) - position(start[a_body]));
        let rotation_bound = rotation_travel(start[a_body], end[a_body]) * collider_body_radius(a)
            + rotation_travel(start[b_body], end[b_body]) * collider_body_radius(b);
        let Ok(initial_a) = crate::live_weld::geometry(a, start[a_body]) else {
            return false;
        };
        let Ok(initial_b) = crate::live_weld::geometry(b, start[b_body]) else {
            return false;
        };
        let overlap = crate::live_weld::penetration(&initial_a, &initial_b);
        // Parts already tangled into each other may pass through to come apart.
        if overlap > 1.0e-4 {
            continue;
        }
        let allowed = overlap.max(0.0) + 1.0e-4;
        let mut intervals = vec![(0.0, 1.0, 0_u8)];
        let mut evaluations = 0_usize;
        while let Some((low, high, depth)) = intervals.pop() {
            // A pair grazing along the whole path would otherwise split every
            // interval down to the depth limit.
            evaluations += 1;
            if evaluations > MAX_PAIR_EVALUATIONS {
                return false;
            }
            let midpoint = (low + high) * 0.5;
            let Ok(mid_a) =
                crate::live_weld::geometry(a, pose_at(start[a_body], end[a_body], midpoint))
            else {
                return false;
            };
            let Ok(mid_b) =
                crate::live_weld::geometry(b, pose_at(start[b_body], end[b_body], midpoint))
            else {
                return false;
            };
            if swept_pair_depth(
                &mid_a,
                &mid_b,
                relative_translation,
                rotation_bound,
                (high - low) * 0.5,
                allowed,
            ) <= allowed
            {
                continue;
            }
            if crate::live_weld::penetration(&mid_a, &mid_b) > allowed || depth >= 24 {
                return false;
            }
            intervals.push((midpoint, high, depth + 1));
            intervals.push((low, midpoint, depth + 1));
        }
    }
    true
}

#[test]
fn swept_cache_matches_exhaustive_checks_on_deterministic_poses() {
    let mut creation = creation();
    creation.colliders[0].local_center = Vec3::new(0.8, 0.1, -0.2);
    let held = [true; 3];
    let mut cache = ClearanceCache::new(&creation, &held);
    // Distinct translations and rotations include separated, crossing and tangled poses.
    for seed in 0..160_u32 {
        let value = |n: u32| {
            let bits = seed
                .wrapping_mul(747_796_405)
                .wrapping_add(n.wrapping_mul(2_891_336_453));
            f32::from(u16::try_from((bits ^ (bits >> 16)) & 0xffff).unwrap()) / 65535.0
        };
        let poses = |offset: u32| {
            (0..3_u32)
                .map(|body| {
                    let n = offset + body * 5;
                    pose(
                        Vec3::new(value(n) * 4.0 - 2.0, value(n + 1) * 2.0, value(n + 2) * 2.0),
                        Quat::from_rotation_z(value(n + 3) * 5.0)
                            * Quat::from_rotation_y(value(n + 4)),
                    )
                })
                .collect::<Vec<_>>()
        };
        let start = poses(0);
        let end = poses(17);
        assert_eq!(
            cache.endpoint_clear(&creation, &end),
            exhaustive_endpoint_clear(&creation, &held, &end),
            "endpoint seed {seed}"
        );
        assert_eq!(
            cache.path_clear(&creation, &start, &end),
            exhaustive_path_clear(&creation, &held, &start, &end),
            "sweep seed {seed}"
        );
    }
}

#[test]
fn common_height_interpolation_preserves_the_accepted_arrangement() {
    let held = [true, true, false];
    let start = vec![
        pose(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY),
        pose(Vec3::new(1.25, 3.0, 3.0), Quat::from_rotation_y(0.3)),
        pose(Vec3::splat(9.0), Quat::IDENTITY),
    ];
    let end = translated(&start, &held, 0.75);
    let (next, done) = translated_step(&start, &end, &held, 1.0 / 60.0);
    assert!(!done);
    assert_eq!(next[2], start[2]);
    assert_eq!(next[0].rotation, start[0].rotation);
    assert_eq!(next[1].rotation, start[1].rotation);
    assert!(
        (position(next[1]) - position(next[0]))
            .abs_diff_eq(position(start[1]) - position(start[0]), 1.0e-6)
    );
    assert!(next[0].position[1] > start[0].position[1]);
}

#[test]
fn publication_and_restore_replace_cached_geometry_and_reset_translation() {
    let candidate = linked_candidate();
    let mut frozen = aligning_freeze(&candidate);
    frozen.clearance = Some(ClearanceCache::new(&creation(), &[true; 3]));
    frozen.aligned = true;
    frozen.translating = true;
    let prepared = frozen
        .prepare_with_environment(&candidate, None, &|p| p.0.as_vec3(), &flat_terrain)
        .unwrap();
    assert!(!prepared.aligned);
    assert!(!prepared.translating);
    let mut cache = prepared.clearance.unwrap();
    // The replacement has one held body; a stale three-body cache would index past its poses.
    assert!(cache.endpoint_clear(candidate.creation.as_ref().unwrap(), &prepared.poses));
    let restored = frozen.restored_for_weld(&candidate);
    assert!(restored.clearance.is_none());
    assert!(!restored.aligned);
    assert!(!restored.translating);
    frozen.reset();
    assert!(frozen.clearance.is_none());
}

#[test]
fn arrows_during_alignment_still_reject_crossing_parts_and_keep_the_target() {
    let creation = creation();
    let start = vec![
        pose(Vec3::new(-1.0, 2.0, 0.0), Quat::IDENTITY),
        pose(Vec3::new(1.0, 2.0, 0.0), Quat::IDENTITY),
        pose(Vec3::splat(50.0), Quat::IDENTITY),
    ];
    let end = vec![start[1], start[0], start[2]];
    let mut frozen = DimensionFreeze {
        held: vec![true, true, false],
        poses: start.clone(),
        waypoints: VecDeque::from([start.clone()]),
        ..Default::default()
    };
    assert!(frozen.plan_height(&creation, &end, &flat_terrain).is_none());
    assert_eq!(frozen.poses, start);
    assert_eq!(frozen.waypoints.back(), Some(&start));
    // Once settled, height motion retains the validated arrangement and still queries terrain.
    frozen.aligned = true;
    let raised = translated(&start, &frozen.held, 0.25);
    assert!(
        frozen
            .plan_height(&creation, &raised, &flat_terrain)
            .is_some()
    );
    assert!(
        frozen
            .plan_height(&creation, &raised, &|_, _| Some(1.0))
            .is_none()
    );
    assert_eq!(frozen.poses, start);
}

#[test]
fn reused_cache_uses_current_poses_after_origin_translation() {
    let creation = creation();
    let held = [true, true, false];
    let mut cache = ClearanceCache::new(&creation, &held);
    let poses = vec![
        pose(Vec3::new(-1.0, 2.0, 0.0), Quat::IDENTITY),
        pose(Vec3::new(1.0, 2.0, 0.0), Quat::IDENTITY),
        pose(Vec3::splat(50.0), Quat::IDENTITY),
    ];
    assert!(cache.endpoint_clear(&creation, &poses));
    let mut shifted = poses.clone();
    for p in &mut shifted {
        p.position[0] += 1000.0;
        p.position[1] -= 250.0;
    }
    assert!(cache.endpoint_clear(&creation, &shifted));
    shifted[1] = shifted[0];
    assert!(!cache.endpoint_clear(&creation, &shifted));
    assert!(cache.endpoint_clear(&creation, &poses));
}

#[test]
fn hierarchical_terrain_queries_keep_walls_ceilings_and_new_terrain_obstructing() {
    let mut creation = creation();
    let original = creation.colliders[0].clone();
    creation.colliders.clear();
    for x in [-0.5, 0.0, 0.5] {
        let mut collider = original.clone();
        collider.compound_index = 0;
        collider.local_center = Vec3::new(x, 0.0, 0.0);
        creation.colliders.push(collider);
    }
    let held = [true, false, false];
    let cache = ClearanceCache::new(&creation, &held);
    let start = vec![pose(Vec3::Y * 2.0, Quat::IDENTITY); 3];
    let end = translated(&start, &held, 0.25);
    assert!(external_path_clear_cached(
        &cache,
        &creation,
        &start,
        &end,
        &flat_terrain
    ));
    let ceiling = |center: Vec3, radius: f32| Some(radius + center.y - 2.3);
    assert!(!external_path_clear_cached(
        &cache, &creation, &start, &end, &ceiling
    ));
    assert!(!external_endpoint_clear_cached(
        &cache, &creation, &end, &ceiling
    ));
    assert!(!external_path_clear_cached(
        &cache,
        &creation,
        &start,
        &end,
        &|_, _| None
    ));
    // The same cache observes changed terrain immediately.
    assert!(external_path_clear_cached(
        &cache,
        &creation,
        &start,
        &end,
        &flat_terrain
    ));
    let mut across = start.clone();
    across[0].position[0] = 3.0;
    let wall = |center: Vec3, radius: f32| Some(radius - (center.x - 1.5).abs());
    assert!(!external_path_clear_cached(
        &cache, &creation, &start, &across, &wall
    ));
}

#[test]
fn downward_clearance_preserves_clipped_steps_and_rejects_sub_tolerance_progress() {
    let maximum = 0.25;
    let clipped = minimum_clear_lift(maximum, |lift| lift >= 0.123).unwrap();
    assert!((clipped - 0.123).abs() < 2.0e-5);
    let already_at_floor = minimum_clear_lift(maximum, |lift| lift >= maximum - 5.0e-5).unwrap();
    assert!(already_at_floor >= maximum - 1.0e-4);
    assert!(minimum_clear_lift(maximum, |_| false).is_none());
}
