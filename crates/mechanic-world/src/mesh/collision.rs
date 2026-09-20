//! The collision form of a meshed chunk.

use super::WorldBounds;
use super::bvh::TriangleBvh;
use super::groups::{TerrainIndexGroups, TerrainTriangleGroupMask};
use crate::{TerrainMaterial, TerrainNodeId, WorldPosition};
use bevy_math::Vec3;

/// Physics-owned view of the same triangles rendered and queried by the app.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TerrainCollisionChunk {
    /// Owning octree node.
    pub node: TerrainNodeId,
    /// Global origin added to node-local vertices.
    pub origin: WorldPosition,
    /// Node-local triangle vertices.
    pub vertices: Vec<[f32; 3]>,
    /// Smooth normals.
    pub normals: Vec<[f32; 3]>,
    /// All generated index groups.
    pub index_groups: TerrainIndexGroups,
    /// Currently active sealed or final indices.
    pub indices: Vec<u32>,
    /// One weight per [`TerrainMaterial`] at each vertex.
    pub material_weights: Vec<[f32; TerrainMaterial::COUNT]>,
    /// Plastic compaction at each vertex, in compaction steps. A chunk that
    /// lists none is uncompacted ground.
    pub compaction: Vec<u8>,
    /// Global owning bounds.
    pub bounds: WorldBounds,
    /// Generation invalidating old manifolds after replacement.
    pub generation: u64,
    /// Triangle acceleration data.
    pub triangle_bvh: TriangleBvh,
    /// Geometry groups currently enabled for collision.
    pub active_groups: TerrainTriangleGroupMask,
}

impl TerrainCollisionChunk {
    /// Active triangle rows in the shared BVH whose leaf boxes overlap the query.
    /// Stable row order gives contact identities independent of traversal order.
    /// Leaf candidates still require an exact finite-triangle narrowphase.
    pub fn bounds_candidates(&self, bounds: WorldBounds) -> Vec<usize> {
        let mut result = Vec::new();
        if self.triangle_bvh.nodes.is_empty() {
            return result;
        }
        let mut stack = vec![0_usize];
        while let Some(index) = stack.pop() {
            let node = &self.triangle_bvh.nodes[index];
            if !node.group_mask.intersects(self.active_groups) || !node.bounds.intersects(bounds) {
                continue;
            }
            if node.triangle_count > 0 {
                let start = node.first_triangle as usize;
                result.extend(
                    (start..start + node.triangle_count as usize).filter(|&row| {
                        self.triangle_bvh.triangles[row]
                            .group_mask
                            .intersects(self.active_groups)
                    }),
                );
            } else {
                stack.extend(
                    [node.left_child, node.right_child]
                        .into_iter()
                        .flatten()
                        .map(|child| child as usize),
                );
            }
        }
        result.sort_unstable();
        result
    }
}

/// Result of a terrain triangle raycast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainRayHit {
    /// Global hit point.
    pub position: WorldPosition,
    /// Smooth density-gradient normal.
    pub normal: Vec3,
    /// Distance along the normalized ray.
    pub distance: f64,
    /// One weight per [`TerrainMaterial`] at the hit.
    pub material_weights: [f32; TerrainMaterial::COUNT],
    /// Chunk generation hit by the ray.
    pub chunk_generation: u64,
    /// Triangle number within the chunk.
    pub triangle: u32,
}
