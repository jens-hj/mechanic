use super::*;
use bevy_math::DQuat;
use mechanic_core::{BuildCommand, BuildPose, ConstructionGraph, CuboidSpec};
use mechanic_world::{
    TerrainMaterial, TerrainTriangleGroupMask, TriangleBvh, TriangleBvhNode, TriangleBvhTriangle,
};

pub(crate) fn cube() -> (CompiledCreation, MachineCollisionGeometry, Vec<BodyPose>) {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([4, 4, 4], BuildPose::default()).unwrap(),
        ))
        .unwrap();
    let creation = graph.compile().unwrap();
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    (
        creation,
        geometry,
        vec![BodyPose {
            position: DVec3::Y * 0.499,
            rotation: DQuat::IDENTITY,
        }],
    )
}

// Two 1 m cubes with no bearing between them. They are authored apart so the
// build never joins them; tests place them by pose.
pub(crate) fn loose_cubes() -> CompiledCreation {
    use mechanic_core::GridRotation;
    let mut graph = ConstructionGraph::new();
    for ticks in [bevy_math::IVec3::ZERO, bevy_math::IVec3::X * 800] {
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 4, 4],
                    BuildPose::from_position_ticks(ticks, GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
    }
    let creation = graph.compile().unwrap();
    assert_eq!(creation.compounds.len(), 2);
    creation
}

pub(crate) fn pose(position: DVec3) -> BodyPose {
    BodyPose {
        position,
        rotation: DQuat::IDENTITY,
    }
}

#[test]
fn a_box_resting_on_another_box_is_supported_through_one_face_manifold() {
    let creation = loose_cubes();
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    let poses = vec![pose(DVec3::Y * 0.5), pose(DVec3::Y * 1.499)];
    let query = TerrainContactScene::default()
        .contacts(&geometry, &poses, DVec3::ZERO)
        .unwrap();
    assert_eq!(query.collider_pair_candidates, 1);
    // The upper box's sides also meet the lower box's top edges; only the one
    // separating face supplies points, so no horizontal support appears.
    assert_eq!(query.contacts.len(), 4);
    let model = crate::MachineDynamics::assemble(&creation, &poses, &[]).unwrap();
    let rows = creation.dynamics.elimination_parent.len();
    let mut rising = vec![0.0; rows];
    for range in &creation.dynamics.body_velocities {
        rising[range.start + 1] = 1.0;
    }
    for contact in &query.contacts {
        let opposing = contact.other_body.unwrap();
        assert_ne!(contact.body, opposing);
        assert!(matches!(
            contact.feature.obstacle,
            ContactObstacle::Collider(_)
        ));
        // The normal pushes the receiving body away from the opposing one.
        let away = poses[contact.body].position - poses[opposing].position;
        assert!(contact.normal.abs_diff_eq(away.normalize(), 1e-12));
        assert!((contact.depth - 0.001).abs() < 1e-9);
        assert!((contact.separation + 0.001).abs() < 1e-9);
        // Rows measure relative motion: rising together does not close the gap.
        let row = contact.point_row(&model, contact.normal).unwrap();
        let speed =
            |velocities: &[f64]| row.iter().zip(velocities).map(|(j, v)| j * v).sum::<f64>();
        assert!(speed(&rising).abs() < 1e-12);
        let mut receiving = vec![0.0; rows];
        receiving[creation.dynamics.body_velocities[contact.body].start + 1] = 1.0;
        assert!((speed(&receiving) - contact.normal.y).abs() < 1e-12);
    }
    let constraints = query
        .impact_constraints(&model, &vec![0.0; rows], 1.0, 1e-7)
        .unwrap();
    assert_eq!(constraints.blocks.len(), 1);
    assert_eq!(constraints.blocks[0].contacts.len(), 4);
}

#[test]
fn bodies_joined_by_a_bearing_never_collide_with_each_other() {
    use mechanic_core::{BearingSpec, BuildOutcome, FaceKind, FaceRef, GridRotation, PartId};
    let mut graph = ConstructionGraph::new();
    let mut spawn = |ticks: bevy_math::IVec3| {
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 4, 4],
                    BuildPose::from_position_ticks(ticks, GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            panic!("spawn expected");
        };
        part as PartId
    };
    let root = spawn(bevy_math::IVec3::ZERO);
    let tip = spawn(bevy_math::IVec3::X * 400);
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(root, FaceKind::PositiveX),
            FaceRef::part(tip, FaceKind::NegativeX),
            Vec3::X * 0.5,
            Vec3::X,
        )))
        .unwrap();
    let hinged = graph.compile().unwrap();
    // Share one face exactly, as a built hinge does.
    let poses = vec![pose(DVec3::ZERO), pose(DVec3::X)];
    let scene = TerrainContactScene::default();
    let geometry = MachineCollisionGeometry::new(&hinged, 1).unwrap();
    let query = scene.contacts(&geometry, &poses, DVec3::ZERO).unwrap();
    assert_eq!(query.collider_pair_candidates, 0);
    assert!(query.contacts.is_empty());
    // The same boxes without the bearing do touch there.
    let loose = MachineCollisionGeometry::new(&loose_cubes(), 1).unwrap();
    let query = scene.contacts(&loose, &poses, DVec3::ZERO).unwrap();
    assert_eq!(query.collider_pair_candidates, 1);
    assert!(!query.contacts.is_empty());
    assert!(
        query
            .contacts
            .iter()
            .all(|contact| contact.normal.abs().abs_diff_eq(DVec3::X, 1e-12))
    );
}

#[test]
fn bodies_of_one_mechanism_collide_only_when_they_were_built_apart() {
    use mechanic_core::{BearingSpec, BuildOutcome, FaceKind, FaceRef, GridRotation, PartId};
    // Three boxes in a row, each joined to the next. Root and tip are two joints
    // apart and built a metre from each other.
    let mut graph = ConstructionGraph::new();
    let mut spawn = |ticks: bevy_math::IVec3| {
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 4, 4],
                    BuildPose::from_position_ticks(ticks, GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            panic!("spawn expected");
        };
        part as PartId
    };
    let parts = [0, 400, 800].map(|x| spawn(bevy_math::IVec3::X * x));
    for (joint, anchor) in [0.5_f32, 1.5].into_iter().enumerate() {
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(parts[joint], FaceKind::PositiveX),
                FaceRef::part(parts[joint + 1], FaceKind::NegativeX),
                Vec3::X * anchor,
                Vec3::X,
            )))
            .unwrap();
    }
    let chain = graph.compile().unwrap();
    let geometry = MachineCollisionGeometry::new(&chain, 1).unwrap();
    // Fold the tip back onto the root: it rests 1 mm deep on the root's top.
    let poses = vec![pose(DVec3::ZERO), pose(DVec3::X), pose(DVec3::Y * 0.999)];
    let query = TerrainContactScene::default()
        .contacts(&geometry, &poses, DVec3::ZERO)
        .unwrap();
    let bodies = |contact: &TerrainContact| {
        let other = contact.other_body.unwrap();
        [contact.body.min(other), contact.body.max(other)]
    };
    assert_eq!(query.contacts.len(), 4);
    assert!(
        query
            .contacts
            .iter()
            .all(|contact| bodies(contact) == [0, 2])
    );
}

// The exact shapes the solver collides, paired with their bodies. A test that
// measures penetration from `creation.colliders` instead would read a solid
// cylinder's sixteen boxes, whose shared corners sit about 1e-8 m below the prism
// the solver actually uses.
pub(crate) fn collision_shapes(creation: &CompiledCreation) -> Vec<(usize, ContactPolytope)> {
    let mut shapes = Vec::new();
    let mut row = 0;
    while row < creation.colliders.len() {
        let source = &creation.colliders[row];
        let cylinder = creation
            .cylinders
            .iter()
            .find(|cylinder| cylinder.first_collider as usize == row);
        let shape = match cylinder {
            Some(cylinder) => ContactPolytope::from_convex(&cylinder.hull()),
            None => ContactPolytope::from_collider(source),
        }
        .expect("compiled collision geometry is valid");
        shapes.push((source.compound_index as usize, shape));
        row += cylinder.map_or(1, |_| mechanic_core::CYLINDER_COLLIDER_COUNT);
    }
    shapes
}

pub(crate) fn terrain(materials: [TerrainMaterial; 2]) -> Arc<TerrainCollisionChunk> {
    let bounds = WorldBounds {
        minimum: WorldPosition(DVec3::new(-1.0, 0.0, -1.0)),
        maximum: WorldPosition(DVec3::new(1.0, 0.0, 1.0)),
    };
    let mut weights = Vec::new();
    for material in materials {
        let mut row = [0.0; TerrainMaterial::COUNT];
        row[usize::from(material.code())] = 1.0;
        weights.extend([row; 3]);
    }
    Arc::new(TerrainCollisionChunk {
        node: TerrainNodeId::ROOT,
        vertices: vec![
            [-1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
        ],
        indices: vec![0, 1, 2, 3, 4, 5],
        material_weights: weights,
        bounds,
        generation: 3,
        triangle_bvh: TriangleBvh {
            bounds,
            triangles: vec![
                TriangleBvhTriangle {
                    indices: [0, 1, 2],
                    group_mask: TerrainTriangleGroupMask::REGULAR,
                },
                TriangleBvhTriangle {
                    indices: [3, 4, 5],
                    group_mask: TerrainTriangleGroupMask::REGULAR,
                },
            ],
            nodes: vec![TriangleBvhNode {
                bounds,
                first_triangle: 0,
                triangle_count: 2,
                group_mask: TerrainTriangleGroupMask::REGULAR,
                ..Default::default()
            }],
        },
        active_groups: TerrainTriangleGroupMask::REGULAR,
        ..Default::default()
    })
}

#[test]
fn coplanar_triangle_seam_reduces_to_four_supports_without_changing_materials() {
    let (_, geometry, poses) = cube();
    let mut scene = TerrainContactScene::default();
    scene
        .publish(1, &[terrain([TerrainMaterial::Rock; 2])], &[])
        .unwrap();
    let query = scene.contacts(&geometry, &poses, DVec3::ZERO).unwrap();
    assert_eq!(query.chunk_candidates, 1);
    assert_eq!(query.triangle_candidates, 2);
    assert_eq!(query.unreduced_points, 6);
    assert_eq!(query.contacts.len(), 4);
    for contact in query.contacts {
        assert!((contact.depth - 0.001).abs() < 1e-12);
        assert_eq!(contact.feature.topology_generation, 7);
        assert!(matches!(
            contact.feature.obstacle,
            ContactObstacle::Terrain {
                geometry_generation: 3,
                publication_generation: 1,
                ..
            }
        ));
    }
    scene
        .publish(
            2,
            &[terrain([TerrainMaterial::Rock, TerrainMaterial::Graphite])],
            &[],
        )
        .unwrap();
    let query = scene.contacts(&geometry, &poses, DVec3::ZERO).unwrap();
    assert_eq!(
        query.contacts.len(),
        6,
        "material boundary must retain each side's support"
    );
    assert!((query.contacts[0].response[0] - query.contacts[5].response[0]).abs() > 0.01);
}

#[test]
fn near_terrain_contacts_report_positive_gaps_and_survive_origin_rebasing() {
    let (creation, geometry, mut poses) = cube();
    poses[0].position.y = 0.500_001;
    let mut scene = TerrainContactScene::default();
    scene
        .publish(1, &[terrain([TerrainMaterial::Rock; 2])], &[])
        .unwrap();
    assert!(
        scene
            .contacts(&geometry, &poses, DVec3::ZERO)
            .unwrap()
            .contacts
            .is_empty()
    );
    let near = scene
        .proximity(&geometry, &poses, DVec3::ZERO, 2e-6)
        .unwrap();
    let model = crate::MachineDynamics::assemble(&creation, &poses, &[]).unwrap();
    assert!(matches!(
        near.impact_constraints(&model, &[0.0; 6], 1.0, 1e-7),
        Err(PhysicsError::InvalidConstraints)
    ));
    assert_eq!(near.contacts.len(), 4);
    for contact in &near.contacts {
        assert!((contact.separation - 1e-6).abs() < 1e-12);
        assert!(contact.depth.abs() < 1e-15);
    }
    let origin = DVec3::new(32.0, -16.0, 8.0);
    poses[0].position -= origin;
    let shifted = scene.proximity(&geometry, &poses, origin, 2e-6).unwrap();
    assert_eq!(near.contacts.len(), shifted.contacts.len());
    for (a, b) in near.contacts.iter().zip(&shifted.contacts) {
        assert_eq!(a.feature, b.feature);
        assert!((a.separation - b.separation).abs() < 1e-12);
        assert!(a.body_point.distance(b.body_point + origin) < 1e-12);
    }
}

#[test]
fn small_finite_triangles_remain_collidable_and_nonfinite_query_bounds_fail() {
    let (_, geometry, mut poses) = cube();
    let mut chunk = terrain([TerrainMaterial::Rock; 2]);
    for vertex in &mut Arc::make_mut(&mut chunk).vertices {
        vertex[0] *= 1e-4;
        vertex[2] *= 1e-4;
    }
    let mut scene = TerrainContactScene::default();
    scene.publish(1, &[chunk], &[]).unwrap();
    let query = scene.contacts(&geometry, &poses, DVec3::ZERO).unwrap();
    assert_eq!(
        query.contacts.len(),
        4,
        "finite surface must not be lost to an area cutoff"
    );
    poses[0].position = DVec3::splat(f64::MAX);
    assert!(matches!(
        scene.contacts(&geometry, &poses, DVec3::splat(f64::MAX)),
        Err(PhysicsError::InvalidCollision)
    ));
}

#[test]
fn origin_rebase_preserves_physical_support_and_feature_identity() {
    let (_, geometry, mut poses) = cube();
    let shift = DVec3::new(1e6, 2e6, -3e6);
    let mut chunk = terrain([TerrainMaterial::Rock; 2]);
    let changed = Arc::make_mut(&mut chunk);
    changed.origin.0 += shift;
    changed.bounds.minimum.0 += shift;
    changed.bounds.maximum.0 += shift;
    changed.triangle_bvh.bounds = changed.bounds;
    changed.triangle_bvh.nodes[0].bounds = changed.bounds;
    let mut scene = TerrainContactScene::default();
    scene.publish(1, &[chunk], &[]).unwrap();
    let local = scene.contacts(&geometry, &poses, shift).unwrap();
    poses[0].position += shift;
    let distant = scene.contacts(&geometry, &poses, DVec3::ZERO).unwrap();
    assert_eq!(local.contacts.len(), distant.contacts.len());
    for (a, b) in local.contacts.iter().zip(distant.contacts) {
        assert_eq!(a.feature, b.feature);
        assert!((a.body_point + shift).distance(b.body_point) < 1e-8);
        assert!((a.depth - b.depth).abs() < 1e-8);
    }
}

#[test]
fn malformed_publication_preserves_the_previous_collision_scene() {
    let (_, geometry, poses) = cube();
    let mut scene = TerrainContactScene::default();
    let chunk = terrain([TerrainMaterial::Rock; 2]);
    scene.publish(1, &[Arc::clone(&chunk)], &[]).unwrap();
    let expected = scene
        .contacts(&geometry, &poses, DVec3::ZERO)
        .unwrap()
        .contacts;
    let mut broken = chunk;
    Arc::make_mut(&mut broken).triangle_bvh.nodes[0].triangle_count = 3;
    assert_eq!(
        scene.publish(2, &[broken], &[]),
        Err(PhysicsError::InvalidCollision)
    );
    assert_eq!(scene.generation(), 1);
    assert_eq!(
        scene
            .contacts(&geometry, &poses, DVec3::ZERO)
            .unwrap()
            .contacts,
        expected
    );
    assert_eq!(
        scene.publish(1, &[], &[]),
        Err(PhysicsError::InvalidCollision)
    );
    scene.publish(2, &[], &[TerrainNodeId::ROOT]).unwrap();
    assert!(
        scene
            .contacts(&geometry, &poses, DVec3::ZERO)
            .unwrap()
            .contacts
            .is_empty()
    );
}

#[test]
fn changed_active_groups_and_materials_invalidate_only_changed_chunk_features() {
    let (_, geometry, poses) = cube();
    let mut scene = TerrainContactScene::default();
    let mut chunk = terrain([TerrainMaterial::Rock; 2]);
    scene.publish(1, &[Arc::clone(&chunk)], &[]).unwrap();
    scene.publish(2, &[], &[]).unwrap();
    assert!(
        scene
            .contacts(&geometry, &poses, DVec3::ZERO)
            .unwrap()
            .contacts
            .iter()
            .all(|c| matches!(
                c.feature.obstacle,
                ContactObstacle::Terrain {
                    publication_generation: 1,
                    ..
                }
            ))
    );
    Arc::make_mut(&mut chunk).active_groups = TerrainTriangleGroupMask::default();
    Arc::make_mut(&mut chunk).indices.clear();
    scene.publish(3, &[Arc::clone(&chunk)], &[]).unwrap();
    assert!(
        scene
            .contacts(&geometry, &poses, DVec3::ZERO)
            .unwrap()
            .contacts
            .is_empty()
    );
    Arc::make_mut(&mut chunk).active_groups = TerrainTriangleGroupMask::REGULAR;
    Arc::make_mut(&mut chunk).indices = vec![0, 1, 2, 3, 4, 5];
    scene.publish(4, &[chunk], &[]).unwrap();
    assert!(
        scene
            .contacts(&geometry, &poses, DVec3::ZERO)
            .unwrap()
            .contacts
            .iter()
            .all(|c| matches!(
                c.feature.obstacle,
                ContactObstacle::Terrain {
                    publication_generation: 4,
                    ..
                }
            ))
    );
}

#[test]
fn real_manifold_impulses_support_the_coupled_body_without_spurious_spin() {
    use crate::{ConstraintBlock, ImpulseBounds, MachineDynamics, solve_constraints};
    let (creation, geometry, poses) = cube();
    let mut scene = TerrainContactScene::default();
    scene
        .publish(1, &[terrain([TerrainMaterial::Rock; 2])], &[])
        .unwrap();
    let contacts = scene
        .contacts(&geometry, &poses, DVec3::ZERO)
        .unwrap()
        .contacts;
    let dynamics = MachineDynamics::assemble(&creation, &poses, &[]).unwrap();
    let factor = dynamics.factor(&[0.0; 6]).unwrap();
    let blocks = contacts
        .iter()
        .map(|contact| ConstraintBlock {
            jacobian: vec![
                dynamics
                    .point_row(contact.body, contact.body_point, contact.normal)
                    .unwrap(),
            ],
            target: vec![1.0],
            bounds: vec![ImpulseBounds {
                minimum: 0.0,
                maximum: f64::INFINITY,
            }],
            contacts: Vec::new(),
        })
        .collect::<Vec<_>>();
    let solution = solve_constraints(&factor, &blocks, 256, 1e-10).unwrap();
    assert!(solution.converged);
    assert!((solution.velocity_change[1] - 1.0).abs() < 1e-9);
    assert!(solution.velocity_change[3..].iter().all(|v| v.abs() < 1e-9));
    let mass = f64::from(creation.compounds[0].mass_properties.mass);
    assert!((solution.impulses.iter().sum::<f64>() - mass).abs() < mass * 1e-9);
}

#[test]
fn finite_manifold_restitution_uses_impact_speed_without_penetration_bounce() {
    use crate::{MachineDynamics, solve_constraints};
    let (creation, geometry, mut poses) = cube();
    let mut scene = TerrainContactScene::default();
    scene
        .publish(1, &[terrain([TerrainMaterial::Rock; 2])], &[])
        .unwrap();
    for height in [0.499, 0.49] {
        poses[0].position.y = height;
        let query = scene.contacts(&geometry, &poses, DVec3::ZERO).unwrap();
        let model = MachineDynamics::assemble(&creation, &poses, &[]).unwrap();
        let factor = model.factor(&[0.0; 6]).unwrap();
        for speed in [0.0, -0.1, -2.0] {
            let incoming = [0.0, speed, 0.0, 0.0, 0.0, 0.0];
            let constraints = query
                .impact_constraints(&model, &incoming, 1.0, 1e-7)
                .unwrap();
            let solution = solve_constraints(&factor, &constraints.blocks, 256, 1e-10).unwrap();
            assert!(solution.converged);
            let expected = if speed < -1.0 {
                -speed * query.contacts[0].response[2]
            } else {
                0.0
            };
            assert!((speed + solution.velocity_change[1] - expected).abs() < 1e-9);
            assert!(
                solution.velocity_change[3..]
                    .iter()
                    .all(|value| value.abs() < 1e-9)
            );
        }
    }
}

#[test]
fn generated_chunk_bvh_queries_match_direct_finite_triangle_queries() {
    use mechanic_world::{
        BrickCoord, TerrainField, TerrainMeshRequest, TerrainOctree, TerrainTransitionMask,
        WorldSeed, mesh_chunk,
    };
    let (creation, geometry, mut poses) = cube();
    let chunk = mesh_chunk(
        &TerrainField::new(WorldSeed(9)),
        &TerrainOctree::default().snapshot(),
        TerrainMeshRequest {
            node: TerrainNodeId::leaf(BrickCoord::new(0, 2, -1)),
            generation: 5,
            transition_mask: TerrainTransitionMask::NONE,
        },
    )
    .collision_chunk();
    assert!(!chunk.triangle_bvh.triangles.is_empty());
    let nominal = chunk.node.world_bounds();
    let outside = chunk
        .vertices
        .iter()
        .map(|v| {
            let point = chunk.origin.0 + Vec3::from_array(*v).as_dvec3();
            (nominal.minimum.0 - point)
                .max(point - nominal.maximum.0)
                .max_element()
                .max(0.0)
        })
        .fold(0.0, f64::max);
    eprintln!("generated_vertex_outside_nominal_node_m={outside:e}");
    let triangle = chunk
        .triangle_bvh
        .triangles
        .iter()
        .find(|triangle| triangle.group_mask.intersects(chunk.active_groups))
        .unwrap();
    poses[0].position = chunk.origin.0
        + triangle
            .indices
            .map(|index| Vec3::from_array(chunk.vertices[index as usize]).as_dvec3())
            .into_iter()
            .sum::<DVec3>()
            / 3.0;
    let shape = ContactPolytope::from_collider(&creation.colliders[0])
        .unwrap()
        .transformed(poses[0].position, poses[0].rotation)
        .unwrap();
    let direct = chunk
        .triangle_bvh
        .triangles
        .iter()
        .filter(|triangle| triangle.group_mask.intersects(chunk.active_groups))
        .map(|triangle| {
            let points = triangle.indices.map(|index| {
                chunk.origin.0 + Vec3::from_array(chunk.vertices[index as usize]).as_dvec3()
            });
            if (points[1] - points[0])
                .cross(points[2] - points[0])
                .try_normalize()
                .is_none()
            {
                0
            } else {
                shape.triangle_contacts(points).unwrap().len()
            }
        })
        .sum::<usize>();
    let mut scene = TerrainContactScene::default();
    scene.publish(1, &[Arc::new(chunk)], &[]).unwrap();
    let query = scene.contacts(&geometry, &poses, DVec3::ZERO).unwrap();
    assert!(direct > 0);
    assert_eq!(query.unreduced_points, direct);
    assert!(query.contacts.len() <= direct);
}

#[test]
fn builder_pipe_hierarchy_matches_exhaustive_pairs_at_rotated_poses() {
    let instance: mechanic_world::WorldCreationInstanceDoc = ron::from_str(include_str!(
        "../../../mechanic-bench/tests/fixtures/builder-world/generations/20/world.ron"
    ))
    .unwrap();
    let loaded = instance.creation.into_graph().unwrap();
    let creation = loaded
        .graph
        .compile_with_suspension_sockets([], &loaded.sockets)
        .unwrap();
    let geometry = MachineCollisionGeometry::new(&creation, 1).unwrap();
    let state = crate::MachineState::at_rest(&creation);
    for angle in [0.0, 0.4, 1.7] {
        let rotation = bevy_math::DQuat::from_rotation_z(angle);
        let poses = state
            .poses
            .iter()
            .map(|pose| BodyPose {
                position: rotation * pose.position,
                rotation: rotation * pose.rotation,
            })
            .collect::<Vec<_>>();
        let bounds = geometry
            .colliders
            .iter()
            .map(|collider| {
                let pose = poses[collider.body];
                let exact = collider
                    .local
                    .transformed(pose.position, pose.rotation)
                    .unwrap()
                    .bounds();
                assert_eq!(
                    exact,
                    collider
                        .local
                        .transformed_bounds(pose.position, pose.rotation)
                        .unwrap()
                );
                [exact[0] - DVec3::splat(0.01), exact[1] + DVec3::splat(0.01)]
            })
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        for (a, first) in geometry.colliders.iter().enumerate() {
            for (b, second) in geometry.colliders.iter().enumerate().skip(a + 1) {
                let bodies = [first.body.min(second.body), first.body.max(second.body)];
                if first.body != second.body
                    && (first.moving || second.moving)
                    && geometry.suppressed.binary_search(&bodies).is_err()
                    && super::overlaps(bounds[a], bounds[b])
                {
                    expected.push([a, b]);
                }
            }
        }
        assert_eq!(&*geometry.candidate_pairs(&bounds), expected);
        for [a, b] in expected.into_iter().take(50) {
            let a = geometry.colliders[a]
                .local
                .transformed(
                    poses[geometry.colliders[a].body].position,
                    poses[geometry.colliders[a].body].rotation,
                )
                .unwrap();
            let b = geometry.colliders[b]
                .local
                .transformed(
                    poses[geometry.colliders[b].body].position,
                    poses[geometry.colliders[b].body].rotation,
                )
                .unwrap();
            let full = a.convex_separation(&b).unwrap();
            for margin in [0.0, 0.01, 0.1] {
                let bounded = a.convex_separation_within(&b, margin).unwrap();
                if full.separation <= margin {
                    assert_eq!(bounded, Some(full));
                }
                if let Some(bounded) = bounded {
                    assert_eq!(bounded, full);
                }
            }
        }
    }
}

#[test]
fn rotated_pipe_opening_stays_open_while_its_annulus_and_solid_version_collide() {
    use mechanic_core::{GridRotation, PipeBendDimensions, PipeBendSpec};
    for (inner, offset, collides) in [(0.6, 0.0, false), (0.0, 0.0, true), (0.6, 0.375, true)] {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::SpawnPipeBend(PipeBendSpec::new(
                PipeBendDimensions::new(1.0, inner, 6).unwrap(),
                BuildPose::default(),
            )))
            .unwrap();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_half_grid(bevy_math::IVec3::splat(64), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
        let creation = graph.compile().unwrap();
        let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
        let scene = TerrainContactScene::default();
        let mut poses = crate::MachineState::at_rest(&creation).poses;
        poses[1].position = DVec3::new(offset, 1.0, 0.0);
        let rotation = DQuat::from_rotation_z(0.61);
        for pose in &mut poses {
            pose.position = rotation * pose.position;
            pose.rotation = rotation * pose.rotation;
        }
        assert_eq!(
            !scene
                .contacts(&geometry, &poses, DVec3::ZERO)
                .unwrap()
                .contacts
                .is_empty(),
            collides,
            "inner={inner}, offset={offset}"
        );
    }
}
