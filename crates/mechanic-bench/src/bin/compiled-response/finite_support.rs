//! Shared finite-floor fixture for measurements and physical regressions.

use super::CompiledCreation;
use bevy_math::DVec3;
use mechanic_core::ContactPolytope;
use mechanic_physics::BodyPose;
use mechanic_physics::{MachineCollisionGeometry, TerrainContactScene};
use mechanic_world::{
    TerrainCollisionChunk, TerrainMaterial, TerrainNodeId, TerrainTriangleGroupMask, TriangleBvh,
    TriangleBvhNode, TriangleBvhTriangle, WorldBounds, WorldPosition,
};
use std::error::Error;
use std::sync::Arc;

pub(super) fn scene(
    creation: &CompiledCreation,
    roots: &mut [BodyPose],
) -> Result<(MachineCollisionGeometry, TerrainContactScene), Box<dyn Error>> {
    let mut lowest = f64::INFINITY;
    for collider in &creation.colliders {
        let pose = roots[collider.compound_index as usize];
        lowest = lowest.min(
            ContactPolytope::from_collider(collider)?
                .transformed(pose.position, pose.rotation)?
                .bounds()[0]
                .y,
        );
    }
    for pose in roots {
        pose.position.y -= lowest + 0.001;
    }
    let geometry = MachineCollisionGeometry::new(creation, 1)?;
    let bounds = WorldBounds {
        minimum: WorldPosition(DVec3::new(-64.0, 0.0, -64.0)),
        maximum: WorldPosition(DVec3::new(64.0, 0.0, 64.0)),
    };
    let mut weights = [0.0; TerrainMaterial::COUNT];
    weights[usize::from(TerrainMaterial::Rock.code())] = 1.0;
    let mask = TerrainTriangleGroupMask::REGULAR;
    let chunk = TerrainCollisionChunk {
        node: TerrainNodeId::ROOT,
        vertices: vec![
            [-64.0, 0.0, -64.0],
            [-64.0, 0.0, 64.0],
            [64.0, 0.0, 64.0],
            [64.0, 0.0, -64.0],
        ],
        material_weights: vec![weights; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
        bounds,
        generation: 1,
        triangle_bvh: TriangleBvh {
            bounds,
            triangles: vec![
                TriangleBvhTriangle {
                    indices: [0, 1, 2],
                    group_mask: mask,
                },
                TriangleBvhTriangle {
                    indices: [0, 2, 3],
                    group_mask: mask,
                },
            ],
            nodes: vec![TriangleBvhNode {
                bounds,
                triangle_count: 2,
                group_mask: mask,
                ..Default::default()
            }],
        },
        active_groups: mask,
        ..Default::default()
    };
    let mut terrain = TerrainContactScene::default();
    terrain.publish(1, &[Arc::new(chunk)], &[])?;
    Ok((geometry, terrain))
}
