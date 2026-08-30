//! Terrain-triangle physics scene kept outside the fixed construction collider ABI.

use std::collections::{BTreeMap, BTreeSet};

use bevy_math::{Quat, Vec3};
use mechanic_world::{TerrainCollisionChunk, TerrainNodeId, TriangleBvhTriangle};

/// Terrain-specific performance and failure diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TerrainStageMetrics {
    /// Active triangles traversed this tick.
    pub triangle_count: u64,
    /// Chunks waiting for a tick-boundary replacement.
    pub streaming_backlog: u32,
    /// Chunk generations replaced on this tick.
    pub remesh_count: u32,
    /// Candidate/contact capacity was exceeded.
    pub overflowed: bool,
    /// CPU submission/traversal time supplied by the runtime, in microseconds.
    pub terrain_stage_microseconds: u64,
    /// Allocated terrain vertex capacity.
    pub vertex_capacity: u64,
    /// Allocated terrain index capacity.
    pub index_capacity: u64,
    /// Allocated terrain BVH-node capacity.
    pub bvh_node_capacity: u64,
}

/// Device limits for independent terrain GPU buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerrainBufferLimits {
    /// Maximum terrain vertices.
    pub vertices: usize,
    /// Maximum terrain indices.
    pub indices: usize,
    /// Maximum terrain BVH nodes.
    pub bvh_nodes: usize,
}

impl Default for TerrainBufferLimits {
    fn default() -> Self {
        Self {
            vertices: usize::MAX,
            indices: usize::MAX,
            bvh_nodes: usize::MAX,
        }
    }
}

/// Independently streamed terrain collision scene.
#[derive(Clone, Debug)]
pub struct TerrainPhysicsScene {
    active: BTreeMap<TerrainNodeId, TerrainCollisionChunk>,
    pending: BTreeMap<TerrainNodeId, TerrainCollisionChunk>,
    pinned: BTreeSet<TerrainNodeId>,
    invalidated_generations: Vec<(TerrainNodeId, u64)>,
    metrics: TerrainStageMetrics,
    limits: TerrainBufferLimits,
    capacities: TerrainBufferLimits,
}

impl Default for TerrainPhysicsScene {
    fn default() -> Self {
        Self {
            active: BTreeMap::new(),
            pending: BTreeMap::new(),
            pinned: BTreeSet::new(),
            invalidated_generations: Vec::new(),
            metrics: TerrainStageMetrics::default(),
            limits: TerrainBufferLimits::default(),
            capacities: TerrainBufferLimits {
                vertices: 0,
                indices: 0,
                bvh_nodes: 0,
            },
        }
    }
}

impl TerrainPhysicsScene {
    /// Sets adapter/device limits used by geometric buffer growth.
    pub fn set_buffer_limits(&mut self, limits: TerrainBufferLimits) {
        self.limits = limits;
    }

    /// Queues a generated chunk for atomic replacement at the next safe tick boundary.
    pub fn queue_replacement(&mut self, id: TerrainNodeId, chunk: TerrainCollisionChunk) {
        self.pending.insert(id, chunk);
        self.metrics.streaming_backlog = u32::try_from(self.pending.len()).unwrap_or(u32::MAX);
    }

    /// Applies pending replacements and records chunks overlapped by active bodies.
    ///
    /// Old manifold generations are returned by [`Self::take_invalidated_generations`].
    pub fn begin_tick(&mut self, overlapping_chunks: impl IntoIterator<Item = TerrainNodeId>) {
        self.pinned.clear();
        self.pinned.extend(overlapping_chunks);
        self.metrics.remesh_count = 0;
        let mut prospective = self.active.clone();
        prospective.extend(self.pending.iter().map(|(&id, chunk)| (id, chunk.clone())));
        let required = buffer_usage(prospective.values());
        let Some(capacities) = grow_capacities(self.capacities, required, self.limits) else {
            self.metrics.overflowed = true;
            self.metrics.streaming_backlog = u32::try_from(self.pending.len()).unwrap_or(u32::MAX);
            return;
        };
        self.capacities = capacities;
        self.metrics.vertex_capacity = u64::try_from(capacities.vertices).unwrap_or(u64::MAX);
        self.metrics.index_capacity = u64::try_from(capacities.indices).unwrap_or(u64::MAX);
        self.metrics.bvh_node_capacity = u64::try_from(capacities.bvh_nodes).unwrap_or(u64::MAX);
        self.metrics.overflowed = false;
        let pending = std::mem::take(&mut self.pending);
        for (id, replacement) in pending {
            if let Some(previous) = self.active.insert(id, replacement) {
                self.invalidated_generations.push((id, previous.generation));
            }
            self.metrics.remesh_count = self.metrics.remesh_count.saturating_add(1);
        }
        self.metrics.streaming_backlog = 0;
        self.metrics.triangle_count = self
            .active
            .values()
            .map(|chunk| u64::try_from(chunk.indices.len() / 3).unwrap_or(u64::MAX))
            .sum();
    }

    /// Removes an unpinned streamed chunk. Overlapped chunks remain resident.
    pub fn unload(&mut self, id: TerrainNodeId) -> bool {
        if self.pinned.contains(&id) {
            return false;
        }
        self.active.remove(&id).is_some()
    }

    /// Active chunk, if resident.
    pub fn chunk(&self, id: TerrainNodeId) -> Option<&TerrainCollisionChunk> {
        self.active.get(&id)
    }

    /// Iterates resident collision chunks without consuming `MAX_COLLIDERS`.
    pub fn chunks(&self) -> impl Iterator<Item = (TerrainNodeId, &TerrainCollisionChunk)> {
        self.active.iter().map(|(&id, chunk)| (id, chunk))
    }

    /// Drains manifold generations made stale by a remesh.
    pub fn take_invalidated_generations(&mut self) -> Vec<(TerrainNodeId, u64)> {
        std::mem::take(&mut self.invalidated_generations)
    }

    /// Current terrain overlay metrics.
    pub const fn metrics(&self) -> TerrainStageMetrics {
        self.metrics
    }

    /// Records a stage duration and overflow result after dispatch/readback.
    pub fn record_stage(&mut self, microseconds: u64, overflowed: bool) {
        self.metrics.terrain_stage_microseconds = microseconds;
        self.metrics.overflowed |= overflowed;
    }
}

fn buffer_usage<'a>(
    chunks: impl IntoIterator<Item = &'a TerrainCollisionChunk>,
) -> TerrainBufferLimits {
    chunks.into_iter().fold(
        TerrainBufferLimits {
            vertices: 0,
            indices: 0,
            bvh_nodes: 0,
        },
        |mut usage, chunk| {
            usage.vertices = usage.vertices.saturating_add(chunk.vertices.len());
            usage.indices = usage.indices.saturating_add(chunk.indices.len());
            usage.bvh_nodes = usage
                .bvh_nodes
                .saturating_add(chunk.triangle_bvh.nodes.len());
            usage
        },
    )
}

fn grow_capacities(
    current: TerrainBufferLimits,
    required: TerrainBufferLimits,
    limits: TerrainBufferLimits,
) -> Option<TerrainBufferLimits> {
    let grow = |current: usize, required: usize, limit: usize| {
        if required > limit {
            return None;
        }
        if required <= current {
            return Some(current);
        }
        let capacity = required
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX)
            .min(limit);
        (capacity >= required).then_some(capacity)
    };
    Some(TerrainBufferLimits {
        vertices: grow(current.vertices, required.vertices, limits.vertices)?,
        indices: grow(current.indices, required.indices, limits.indices)?,
        bvh_nodes: grow(current.bvh_nodes, required.bvh_nodes, limits.bvh_nodes)?,
    })
}

/// Construction collider family tested against terrain triangles.
#[derive(Clone, Debug, PartialEq)]
pub enum TerrainContactShape {
    /// Oriented box.
    Cuboid {
        /// Local half extents.
        half_extents: Vec3,
    },
    /// Analytic wheel or cylinder along local Y.
    Cylinder {
        /// Outer radius.
        radius: f32,
        /// Half axial length.
        half_length: f32,
    },
    /// Convex shaped-part vertices in local space.
    Convex {
        /// Convex hull vertices.
        vertices: Vec<Vec3>,
    },
}

/// One shape-versus-terrain triangle contact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainContact {
    /// World contact point on the triangle.
    pub point: Vec3,
    /// Normal pointing from terrain toward the construction body.
    pub normal: Vec3,
    /// Positive overlap depth.
    pub penetration: f32,
    /// Triangle number in the chunk.
    pub triangle: u32,
    /// Chunk generation used to invalidate stale manifolds.
    pub chunk_generation: u64,
    /// One weight per terrain material at the contact.
    pub material_weights: [f32; mechanic_world::TerrainMaterial::COUNT],
}

/// Generates contacts against one chunk's triangle BVH contract.
pub fn terrain_contacts(
    shape: &TerrainContactShape,
    centre: Vec3,
    rotation: Quat,
    chunk: &TerrainCollisionChunk,
    maximum_contacts: usize,
) -> Vec<TerrainContact> {
    let mut contacts = Vec::new();
    for (triangle, candidate) in candidate_triangles(shape, centre, chunk) {
        if contacts.len() >= maximum_contacts {
            break;
        }
        let indices = &candidate.indices;
        let vertex = |index: u32| {
            usize::try_from(index)
                .ok()
                .and_then(|index| chunk.vertices.get(index))
                .copied()
                .map(|vertex| chunk.origin.0.as_vec3() + Vec3::from_array(vertex))
        };
        let (Some(first), Some(second), Some(third)) =
            (vertex(indices[0]), vertex(indices[1]), vertex(indices[2]))
        else {
            continue;
        };
        let raw_normal = (second - first).cross(third - first);
        let Some(mut normal) = raw_normal.try_normalize() else {
            continue;
        };
        let signed_distance = (centre - first).dot(normal);
        if signed_distance < 0.0 {
            normal = -normal;
        }
        let support = support_distance(shape, rotation, normal);
        if signed_distance.abs() >= support {
            continue;
        }
        let projected = centre - normal * signed_distance.abs();
        let point = closest_point_on_triangle(projected, first, second, third);
        let lateral = projected.distance(point);
        if lateral > support {
            continue;
        }
        let penetration = support - signed_distance.abs();
        if penetration > 0.0 {
            contacts.push(TerrainContact {
                point,
                normal,
                penetration,
                triangle,
                chunk_generation: chunk.generation,
                material_weights: triangle_material_weights(chunk, indices),
            });
        }
    }
    contacts
}

fn candidate_triangles(
    shape: &TerrainContactShape,
    centre: Vec3,
    chunk: &TerrainCollisionChunk,
) -> Vec<(u32, TriangleBvhTriangle)> {
    if chunk.triangle_bvh.nodes.is_empty() {
        return chunk
            .triangle_bvh
            .triangles
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, triangle)| triangle.group_mask.intersects(chunk.active_groups))
            .filter_map(|(index, triangle)| {
                u32::try_from(index).ok().map(|index| (index, triangle))
            })
            .collect();
    }
    let radius = match shape {
        TerrainContactShape::Cuboid { half_extents } => half_extents.length(),
        TerrainContactShape::Cylinder {
            radius,
            half_length,
        } => radius.hypot(*half_length),
        TerrainContactShape::Convex { vertices } => vertices
            .iter()
            .map(|vertex| vertex.length())
            .fold(0.0, f32::max),
    };
    let minimum = centre - Vec3::splat(radius);
    let maximum = centre + Vec3::splat(radius);
    let mut candidates = Vec::new();
    let mut stack = vec![0_usize];
    while let Some(index) = stack.pop() {
        let Some(node) = chunk.triangle_bvh.nodes.get(index) else {
            continue;
        };
        if !node.group_mask.intersects(chunk.active_groups) {
            continue;
        }
        let bounds_minimum = node.bounds.minimum.0.as_vec3();
        let bounds_maximum = node.bounds.maximum.0.as_vec3();
        if bounds_maximum.cmplt(minimum).any() || bounds_minimum.cmpgt(maximum).any() {
            continue;
        }
        if node.triangle_count != 0 {
            let Some(first) = usize::try_from(node.first_triangle).ok() else {
                continue;
            };
            let Some(count) = usize::try_from(node.triangle_count).ok() else {
                continue;
            };
            if let Some(triangles) = chunk
                .triangle_bvh
                .triangles
                .get(first..first.saturating_add(count))
            {
                candidates.extend(triangles.iter().copied().enumerate().filter_map(
                    |(offset, triangle)| {
                        triangle
                            .group_mask
                            .intersects(chunk.active_groups)
                            .then(|| {
                                u32::try_from(first + offset)
                                    .ok()
                                    .map(|index| (index, triangle))
                            })
                            .flatten()
                    },
                ));
            }
        } else {
            stack.extend(
                [node.left_child, node.right_child]
                    .into_iter()
                    .flatten()
                    .filter_map(|child| usize::try_from(child).ok()),
            );
        }
    }
    candidates
}

fn triangle_material_weights(
    chunk: &TerrainCollisionChunk,
    indices: &[u32],
) -> [f32; mechanic_world::TerrainMaterial::COUNT] {
    let mut weights = [0.0; mechanic_world::TerrainMaterial::COUNT];
    for &index in indices {
        let source = usize::try_from(index)
            .ok()
            .and_then(|index| chunk.material_weights.get(index))
            .copied()
            .unwrap_or_else(|| {
                let mut weights = [0.0; mechanic_world::TerrainMaterial::COUNT];
                weights[mechanic_world::TerrainMaterial::Rock.code() as usize] = 1.0;
                weights
            });
        for (target, source) in weights.iter_mut().zip(source) {
            *target += source / 3.0;
        }
    }
    weights
}

fn support_distance(shape: &TerrainContactShape, rotation: Quat, normal: Vec3) -> f32 {
    match shape {
        TerrainContactShape::Cuboid { half_extents } => {
            let local = rotation.inverse() * normal;
            local.abs().dot(*half_extents)
        }
        TerrainContactShape::Cylinder {
            radius,
            half_length,
        } => {
            let axis = rotation * Vec3::Y;
            let axial = axis.dot(normal).abs();
            half_length.mul_add(axial, radius * (1.0 - axial * axial).max(0.0).sqrt())
        }
        TerrainContactShape::Convex { vertices } => vertices
            .iter()
            .map(|vertex| (rotation * *vertex).dot(normal).abs())
            .fold(0.0, f32::max),
    }
}

// Real-Time Collision Detection, Christer Ericson, closest point on triangle.
fn closest_point_on_triangle(point: Vec3, first: Vec3, second: Vec3, third: Vec3) -> Vec3 {
    let ab = second - first;
    let ac = third - first;
    let ap = point - first;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return first;
    }
    let bp = point - second;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return second;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return first + ab * (d1 / (d1 - d3));
    }
    let cp = point - third;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return third;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return first + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return second + (third - second) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denominator = (va + vb + vc).recip();
    first + ab * (vb * denominator) + ac * (vc * denominator)
}

#[cfg(test)]
mod tests {
    use bevy_math::{Quat, Vec3};
    use mechanic_world::{
        TerrainCollisionChunk, TerrainTriangleGroupMask, TriangleBvh, TriangleBvhTriangle,
        WorldBounds,
    };

    use super::{
        TerrainBufferLimits, TerrainContactShape, TerrainNodeId, TerrainPhysicsScene,
        terrain_contacts,
    };

    fn flat_chunk(generation: u64) -> TerrainCollisionChunk {
        let bounds = WorldBounds::default();
        TerrainCollisionChunk {
            vertices: vec![[-5.0, 0.0, -5.0], [5.0, 0.0, -5.0], [0.0, 0.0, 5.0]],
            indices: vec![0, 2, 1],
            material_weights: vec![
                {
                    let mut weights = [0.0; mechanic_world::TerrainMaterial::COUNT];
                    weights[mechanic_world::TerrainMaterial::Rock.code() as usize] = 1.0;
                    weights
                };
                3
            ],
            bounds,
            generation,
            triangle_bvh: TriangleBvh {
                bounds,
                triangles: vec![TriangleBvhTriangle {
                    indices: [0, 2, 1],
                    group_mask: TerrainTriangleGroupMask::REGULAR,
                }],
                ..TriangleBvh::default()
            },
            active_groups: TerrainTriangleGroupMask::REGULAR,
            ..TerrainCollisionChunk::default()
        }
    }

    #[test]
    fn cuboid_convex_and_wheel_generate_triangle_contacts() {
        let chunk = flat_chunk(7);
        let shapes = [
            TerrainContactShape::Cuboid {
                half_extents: Vec3::splat(0.5),
            },
            TerrainContactShape::Cylinder {
                radius: 0.5,
                half_length: 0.5,
            },
            TerrainContactShape::Convex {
                vertices: vec![
                    Vec3::new(-0.5, -0.5, 0.0),
                    Vec3::new(0.5, -0.5, 0.0),
                    Vec3::Y,
                ],
            },
        ];
        for shape in shapes {
            let contacts =
                terrain_contacts(&shape, Vec3::new(0.0, 0.4, 0.0), Quat::IDENTITY, &chunk, 4);
            assert!(!contacts.is_empty());
            assert_eq!(contacts[0].chunk_generation, 7);
            assert!(contacts[0].normal.y > 0.9);
        }
    }

    #[test]
    fn replacement_at_tick_boundary_invalidates_old_generation_and_pins_overlap() {
        let id = TerrainNodeId::default();
        let mut scene = TerrainPhysicsScene::default();
        scene.queue_replacement(id, flat_chunk(1));
        scene.begin_tick([]);
        scene.queue_replacement(id, flat_chunk(2));
        scene.begin_tick([id]);
        assert_eq!(scene.chunk(id).unwrap().generation, 2);
        assert_eq!(scene.take_invalidated_generations(), vec![(id, 1)]);
        assert!(!scene.unload(id));
        scene.begin_tick([]);
        assert!(scene.unload(id));
    }

    #[test]
    fn device_limit_overflow_keeps_the_previous_generation_and_reports_backlog() {
        let id = TerrainNodeId::default();
        let mut scene = TerrainPhysicsScene::default();
        scene.queue_replacement(id, flat_chunk(1));
        scene.begin_tick([]);
        scene.set_buffer_limits(TerrainBufferLimits {
            vertices: 2,
            indices: 2,
            bvh_nodes: 0,
        });
        scene.queue_replacement(id, flat_chunk(2));
        scene.begin_tick([]);

        assert_eq!(scene.chunk(id).unwrap().generation, 1);
        assert!(scene.metrics().overflowed);
        assert_eq!(scene.metrics().streaming_backlog, 1);
        assert!(scene.take_invalidated_generations().is_empty());
    }
}
