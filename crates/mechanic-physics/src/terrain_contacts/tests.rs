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
        assert_eq!(contact.feature.geometry_generation, 3);
        assert_eq!(contact.feature.publication_generation, 1);
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
            .all(|c| c.feature.publication_generation == 1)
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
            .all(|c| c.feature.publication_generation == 4)
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
