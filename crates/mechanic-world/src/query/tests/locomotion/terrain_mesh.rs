use super::*;
use crate::{TerrainMeshChunk, TriangleBvh, TriangleBvhNode, TriangleBvhTriangle, WorldBounds};

fn uneven_chunk() -> TerrainMeshChunk {
    let mut chunk = TerrainMeshChunk::default();
    let group = crate::TerrainTriangleGroupMask::REGULAR;
    for row in 0..25_u32 {
        let x = f64::from(row) * 0.4 - 4.0;
        let height = x * 35.0_f64.to_radians().tan() + (x * 3.0).sin() * 0.025;
        for z in [-4.0, 4.0] {
            chunk.vertices.push([x as f32, height as f32, z]);
            chunk.normals.push([0.0, 1.0, 0.0]);
            let mut weights = [0.0; TerrainMaterial::COUNT];
            weights[TerrainMaterial::Soil.code() as usize] = 1.0;
            chunk.material_weights.push(weights);
        }
        if row > 0 {
            let previous = (row - 1) * 2;
            chunk.index_groups.regular.extend_from_slice(&[
                previous,
                previous + 1,
                previous + 2,
                previous + 1,
                previous + 3,
                previous + 2,
            ]);
        }
    }
    chunk.bounds = WorldBounds {
        minimum: WorldPosition(DVec3::new(-4.0, -4.0, -4.0)),
        maximum: WorldPosition(DVec3::new(6.0, 5.0, 4.0)),
    };
    let triangles: Vec<_> = chunk
        .index_groups
        .regular
        .chunks_exact(3)
        .map(|row| TriangleBvhTriangle {
            indices: [row[0], row[1], row[2]],
            group_mask: group,
        })
        .collect();
    chunk.triangle_bvh = TriangleBvh {
        bounds: chunk.bounds,
        nodes: vec![TriangleBvhNode {
            bounds: chunk.bounds,
            first_triangle: 0,
            triangle_count: u32::try_from(triangles.len()).unwrap(),
            group_mask: group,
            ..TriangleBvhNode::default()
        }],
        triangles,
    };
    chunk
}

#[test]
fn uneven_mesh_backed_ground_supports_walking_and_jumps_with_construction_present() {
    let chunk = uneven_chunk();
    let mut spatial_index = TerrainSpatialIndex::default();
    spatial_index.insert_bounds(chunk.node, chunk.bounds);
    let chunks = BTreeMap::from([(chunk.node, chunk)]);
    let ready_faces = BTreeMap::new();
    let terrain = ActiveTerrainScene {
        chunks: &chunks,
        ready_faces: &ready_faces,
        spatial_index: &spatial_index,
    };
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [1; 3],
                BuildPose::from_position_ticks(
                    bevy_math::IVec3::splat(1000),
                    GridRotation::default(),
                ),
            )
            .unwrap(),
        ))
        .unwrap();
    let creation = graph.compile().unwrap();
    let mut construction = crate::ConstructionCollisionIndex::new(&creation);
    let mut scene = KinematicCollisionScene {
        terrain: &terrain,
        construction: Some(&mut construction),
        floating_origin: DVec3::ZERO,
    };
    let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::Y));
    for _ in 0..120 {
        capsule.tick(
            &mut scene,
            KinematicInput::default(),
            mechanic_core::TICK_SECONDS,
        );
    }
    assert!(capsule.grounded);
    for _ in 0..60 {
        capsule.tick(
            &mut scene,
            KinematicInput {
                movement: DVec2::X,
                ..KinematicInput::default()
            },
            mechanic_core::TICK_SECONDS,
        );
        assert!(capsule.grounded, "lost footing on mesh: {capsule:?}");
    }
    capsule.tick(
        &mut scene,
        KinematicInput::default(),
        mechanic_core::TICK_SECONDS,
    );
    let stopped = capsule.position.0;
    for _ in 0..60 {
        capsule.tick(
            &mut scene,
            KinematicInput::default(),
            mechanic_core::TICK_SECONDS,
        );
    }
    assert!(capsule.position.0.abs_diff_eq(stopped, 0.002));
    capsule.tick(
        &mut scene,
        KinematicInput {
            jump: true,
            ..KinematicInput::default()
        },
        mechanic_core::TICK_SECONDS,
    );
    assert!(capsule.velocity.y > 4.0 && !capsule.grounded);
}
