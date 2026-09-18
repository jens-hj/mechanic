//! Smooth terrain mesh, collision, and spatial-query contracts.

#![expect(
    clippy::cast_possible_truncation,
    reason = "GPU vertices and barycentrics are explicitly f32"
)]

use std::{array, collections::HashMap, time::Instant};

use bevy_math::{DVec3, IVec3, Vec3};
use serde::{Deserialize, Serialize};

use crate::{
    BRICK_EDGE_CELLS, TERRAIN_CELL_METERS, TerrainBrick, TerrainFace, TerrainField,
    TerrainMaterial, TerrainNodeId, TerrainOctreeSnapshot, TerrainSample, TerrainTransitionMask,
    WorldCell, WorldPosition,
    transvoxel::tables::{
        REGULAR_CELL_CLASS, REGULAR_CELL_DATA, REGULAR_VERTEX_DATA, TRANSITION_CELL_CLASS,
        TRANSITION_CELL_DATA, TRANSITION_CORNER_DATA, TRANSITION_VERTEX_DATA,
    },
};

const CUBE_CORNERS: [[i32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

/// Global axis-aligned bounds in metres.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorldBounds {
    /// Inclusive minimum corner.
    pub minimum: WorldPosition,
    /// Inclusive maximum corner.
    pub maximum: WorldPosition,
}

impl WorldBounds {
    /// True when the point lies in the bounds.
    pub fn contains(self, point: WorldPosition) -> bool {
        point.0.cmpge(self.minimum.0).all() && point.0.cmple(self.maximum.0).all()
    }

    /// True when the two inclusive boxes share any point.
    pub fn intersects(self, other: Self) -> bool {
        self.minimum.0.cmple(other.maximum.0).all() && self.maximum.0.cmpge(other.minimum.0).all()
    }
}

impl TerrainNodeId {
    /// Inclusive global owning bounds, including shared boundary triangles.
    #[expect(
        clippy::cast_precision_loss,
        reason = "finite-world cell coordinates fit exactly in f64"
    )]
    pub fn world_bounds(self) -> WorldBounds {
        WorldBounds {
            minimum: WorldPosition(DVec3::from_array(
                self.minimum_cell_i64()
                    .map(|cell| cell as f64 * TERRAIN_CELL_METERS),
            )),
            maximum: WorldPosition(DVec3::from_array(
                self.maximum_cell_exclusive_i64()
                    .map(|cell| cell as f64 * TERRAIN_CELL_METERS),
            )),
        }
    }
}

/// Compact triangle acceleration structure owned by a terrain chunk.
///
/// One hierarchy contains regular, transition, and temporary-cap triangles;
/// traversal masks select the currently active geometry without rebuilding it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TriangleBvh {
    /// Root bounds.
    pub bounds: WorldBounds,
    /// Triangles from every geometry group in traversal order.
    pub triangles: Vec<TriangleBvhTriangle>,
    /// Binary hierarchy nodes in depth-first order.
    pub nodes: Vec<TriangleBvhNode>,
}

/// One node in a triangle BVH.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TriangleBvhNode {
    /// Bounds of every descendant triangle.
    pub bounds: WorldBounds,
    /// First triangle in [`TriangleBvh::triangles`] for a leaf.
    pub first_triangle: u32,
    /// Triangle count for a leaf; zero for a branch.
    pub triangle_count: u32,
    /// First child for a branch.
    pub left_child: Option<u32>,
    /// Second child for a branch.
    pub right_child: Option<u32>,
    /// Union of regular, transition, and cap groups below this node.
    pub group_mask: TerrainTriangleGroupMask,
}

/// One triangle in the shared all-groups BVH.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TriangleBvhTriangle {
    /// Vertex indices into the owning chunk.
    pub indices: [u32; 3],
    /// Geometry group controlling whether the triangle is active.
    pub group_mask: TerrainTriangleGroupMask,
}

/// Bit mask for regular, per-face transition, and per-face cap triangles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TerrainTriangleGroupMask(u16);

impl TerrainTriangleGroupMask {
    /// Regular modified-Marching-Cubes triangles.
    pub const REGULAR: Self = Self(1);

    const fn transition(face: TerrainFace) -> Self {
        Self(1 << (1 + face as u16))
    }

    const fn cap(face: TerrainFace) -> Self {
        Self(1 << (7 + face as u16))
    }

    /// True when the masks enable at least one common geometry group.
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// True when every group in `other` is included in this mask.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Separately activatable regular, transition, and temporary cap triangles.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TerrainIndexGroups {
    /// Ordinary modified-Marching-Cubes triangles.
    pub regular: Vec<u32>,
    /// Transvoxel seam triangles on each face.
    pub transitions: [Vec<u32>; 6],
    /// Marching-squares temporary closure triangles on each face.
    pub caps: [Vec<u32>; 6],
}

impl TerrainIndexGroups {
    /// Number of final indices without allocating the combined index vector.
    pub fn final_index_count(&self, transitions: TerrainTransitionMask) -> usize {
        self.regular.len()
            + TerrainFace::ALL
                .into_iter()
                .filter(|&face| transitions.contains(face))
                .map(|face| self.transitions[face.index()].len())
                .sum::<usize>()
    }

    /// Final indices with every requested transition active and no caps.
    pub fn final_indices(&self, transitions: TerrainTransitionMask) -> Vec<u32> {
        let mut indices = Vec::with_capacity(self.final_index_count(transitions));
        indices.extend_from_slice(&self.regular);
        for face in TerrainFace::ALL {
            if transitions.contains(face) {
                indices.extend_from_slice(&self.transitions[face.index()]);
            }
        }
        indices
    }

    /// Sealed indices, adding a cap beyond any requested seam or neighbor whose
    /// generation is not yet ready.
    pub fn sealed_indices(
        &self,
        requested_transitions: TerrainTransitionMask,
        ready_faces: TerrainTransitionMask,
    ) -> Vec<u32> {
        let mut indices =
            Vec::with_capacity(self.sealed_index_count(requested_transitions, ready_faces));
        indices.extend_from_slice(&self.regular);
        for face in TerrainFace::ALL {
            if requested_transitions.contains(face) {
                indices.extend_from_slice(&self.transitions[face.index()]);
            }
            if !ready_faces.contains(face) {
                indices.extend_from_slice(&self.caps[face.index()]);
            }
        }
        indices
    }

    /// Number of sealed indices without allocating the combined index vector.
    pub fn sealed_index_count(
        &self,
        requested_transitions: TerrainTransitionMask,
        ready_faces: TerrainTransitionMask,
    ) -> usize {
        self.regular.len()
            + TerrainFace::ALL
                .into_iter()
                .map(|face| {
                    usize::from(requested_transitions.contains(face))
                        * self.transitions[face.index()].len()
                        + usize::from(!ready_faces.contains(face)) * self.caps[face.index()].len()
                })
                .sum::<usize>()
    }

    fn final_group_mask(transitions: TerrainTransitionMask) -> TerrainTriangleGroupMask {
        let mut mask = TerrainTriangleGroupMask::REGULAR;
        for face in TerrainFace::ALL {
            if transitions.contains(face) {
                mask = mask.union(TerrainTriangleGroupMask::transition(face));
            }
        }
        mask
    }

    fn sealed_group_mask(
        requested_transitions: TerrainTransitionMask,
        ready_faces: TerrainTransitionMask,
    ) -> TerrainTriangleGroupMask {
        let mut mask = TerrainTriangleGroupMask::REGULAR;
        for face in TerrainFace::ALL {
            if requested_transitions.contains(face) {
                mask = mask.union(TerrainTriangleGroupMask::transition(face));
            }
            if !ready_faces.contains(face) {
                mask = mask.union(TerrainTriangleGroupMask::cap(face));
            }
        }
        mask
    }
}

/// One generated terrain chunk shared by rendering, picking, and collision.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TerrainMeshChunk {
    /// Owning octree node.
    pub node: TerrainNodeId,
    /// Global origin added to node-local vertices.
    pub origin: WorldPosition,
    /// Node-local vertex positions in metres.
    pub vertices: Vec<[f32; 3]>,
    /// Density-gradient normals corresponding one-to-one with vertices.
    pub normals: Vec<[f32; 3]>,
    /// Independently activatable triangle groups.
    pub index_groups: TerrainIndexGroups,
    /// One weight per [`TerrainMaterial`] at each vertex.
    pub material_weights: Vec<[f32; TerrainMaterial::COUNT]>,
    /// Owning global bounds.
    pub bounds: WorldBounds,
    /// Triangle query structure.
    pub triangle_bvh: TriangleBvh,
    /// Monotonic generation used to invalidate stale contacts.
    pub generation: u64,
    /// Sample spacing used for this LOD.
    pub sample_spacing_metres: f64,
    /// Transition faces requested for this generation.
    pub transition_mask: TerrainTransitionMask,
    #[doc(hidden)]
    pub vertex_cache: LatticeEdgeVertexCache,
}

type VertexKey = ([u32; 3], [u32; 3], [u32; TerrainMaterial::COUNT]);

#[derive(Clone, Debug, Default, PartialEq)]
/// Transient incremental cache used while emitting one chunk's lattice vertices.
pub struct LatticeEdgeVertexCache {
    vertices: HashMap<VertexKey, u32>,
}

impl TerrainMeshChunk {
    /// Creates the static collision view used by the physics runtime.
    ///
    /// Transition and temporary-cap triangles are render-only closures on the
    /// node face. Treating those face-aligned triangles as volume boundaries
    /// creates artificial collision walls when a controller crosses an LOD seam.
    pub fn collision_chunk(&self) -> TerrainCollisionChunk {
        self.collision_chunk_with_indices(
            &self.index_groups.regular,
            TerrainTriangleGroupMask::REGULAR,
        )
    }

    /// Creates collision while neighbor/transition generations are incomplete.
    /// Render-only face closures are intentionally excluded from collision.
    pub fn sealed_collision_chunk(
        &self,
        _ready_faces: TerrainTransitionMask,
    ) -> TerrainCollisionChunk {
        self.collision_chunk()
    }

    fn collision_chunk_with_indices(
        &self,
        indices: &[u32],
        active_groups: TerrainTriangleGroupMask,
    ) -> TerrainCollisionChunk {
        TerrainCollisionChunk {
            node: self.node,
            origin: self.origin,
            vertices: self.vertices.clone(),
            normals: self.normals.clone(),
            index_groups: self.index_groups.clone(),
            material_weights: self.material_weights.clone(),
            indices: indices.to_owned(),
            bounds: self.bounds,
            generation: self.generation,
            triangle_bvh: self.triangle_bvh.clone(),
            active_groups,
        }
    }

    /// Raycasts this chunk, returning its nearest triangle.
    pub fn raycast(
        &self,
        origin: WorldPosition,
        direction: DVec3,
        maximum_distance: f64,
    ) -> Option<TerrainRayHit> {
        self.raycast_groups(
            origin,
            direction,
            maximum_distance,
            TerrainIndexGroups::final_group_mask(self.transition_mask),
        )
    }

    pub(crate) fn raycast_sealed(
        &self,
        ready_faces: TerrainTransitionMask,
        origin: WorldPosition,
        direction: DVec3,
        maximum_distance: f64,
    ) -> Option<TerrainRayHit> {
        self.raycast_groups(
            origin,
            direction,
            maximum_distance,
            TerrainIndexGroups::sealed_group_mask(self.transition_mask, ready_faces),
        )
    }

    fn raycast_groups(
        &self,
        origin: WorldPosition,
        direction: DVec3,
        maximum_distance: f64,
        active_groups: TerrainTriangleGroupMask,
    ) -> Option<TerrainRayHit> {
        let direction = direction.try_normalize()?;
        let mut nearest = maximum_distance;
        let mut hit = None;
        let mut candidates = Vec::new();
        let mut stack = (!self.triangle_bvh.nodes.is_empty())
            .then_some(0_u32)
            .into_iter()
            .collect::<Vec<_>>();
        while let Some(node_index) = stack.pop() {
            let node = self.triangle_bvh.nodes[usize::try_from(node_index).ok()?];
            if !node.group_mask.intersects(active_groups)
                || !ray_intersects_bounds(origin.0, direction, node.bounds, nearest)
            {
                continue;
            }
            if node.triangle_count != 0 {
                let first = usize::try_from(node.first_triangle).ok()?;
                let count = usize::try_from(node.triangle_count).ok()?;
                candidates.extend(first..first + count);
            } else {
                if let Some(left) = node.left_child {
                    stack.push(left);
                }
                if let Some(right) = node.right_child {
                    stack.push(right);
                }
            }
        }
        for triangle_index in candidates {
            let triangle = self.triangle_bvh.triangles.get(triangle_index)?;
            if !triangle.group_mask.intersects(active_groups) {
                continue;
            }
            let indices = &triangle.indices;
            let first = self.origin.0
                + DVec3::from_array(
                    self.vertices[usize::try_from(indices[0]).ok()?].map(f64::from),
                );
            let second = self.origin.0
                + DVec3::from_array(
                    self.vertices[usize::try_from(indices[1]).ok()?].map(f64::from),
                );
            let third = self.origin.0
                + DVec3::from_array(
                    self.vertices[usize::try_from(indices[2]).ok()?].map(f64::from),
                );
            let Some((distance, barycentric)) =
                ray_triangle(origin.0, direction, first, second, third)
            else {
                continue;
            };
            if distance > nearest {
                continue;
            }
            nearest = distance;
            let first_normal = Vec3::from_array(self.normals[usize::try_from(indices[0]).ok()?]);
            let second_normal = Vec3::from_array(self.normals[usize::try_from(indices[1]).ok()?]);
            let third_normal = Vec3::from_array(self.normals[usize::try_from(indices[2]).ok()?]);
            let normal = (first_normal * barycentric.x
                + second_normal * barycentric.y
                + third_normal * barycentric.z)
                .normalize_or(Vec3::Y);
            let weights = weighted_materials(self, indices, barycentric);
            hit = Some(TerrainRayHit {
                position: WorldPosition(origin.0 + direction * distance),
                normal,
                distance,
                material_weights: weights,
                chunk_generation: self.generation,
                triangle: u32::try_from(triangle_index).ok()?,
            });
        }
        hit
    }

    pub(crate) fn nearest_sealed(
        &self,
        _ready_faces: TerrainTransitionMask,
        position: WorldPosition,
    ) -> Option<(f32, TerrainMaterial, f64)> {
        let active_groups = TerrainTriangleGroupMask::REGULAR;
        let query = position.0.as_vec3();
        let mut best_distance_squared = f32::INFINITY;
        let mut nearest = None;
        let mut stack = (!self.triangle_bvh.nodes.is_empty())
            .then_some(0_u32)
            .into_iter()
            .collect::<Vec<_>>();
        while let Some(node_index) = stack.pop() {
            let node = self.triangle_bvh.nodes[usize::try_from(node_index).ok()?];
            if !node.group_mask.intersects(active_groups)
                || point_bounds_distance_squared(position.0, node.bounds)
                    >= f64::from(best_distance_squared)
            {
                continue;
            }
            if node.triangle_count == 0 {
                stack.extend([node.left_child, node.right_child].into_iter().flatten());
                continue;
            }
            let first = usize::try_from(node.first_triangle).ok()?;
            let count = usize::try_from(node.triangle_count).ok()?;
            for triangle in &self.triangle_bvh.triangles[first..first + count] {
                if !triangle.group_mask.intersects(active_groups) {
                    continue;
                }
                let points = triangle.indices.map(|index| {
                    self.origin.0.as_vec3()
                        + Vec3::from_array(self.vertices[usize::try_from(index).expect("index")])
                });
                let closest = closest_point_on_triangle(query, points[0], points[1], points[2]);
                let distance_squared = query.distance_squared(closest);
                if distance_squared >= best_distance_squared {
                    continue;
                }
                let normal = (points[1] - points[0])
                    .cross(points[2] - points[0])
                    .normalize_or(Vec3::Y);
                let weights = self.material_weights
                    [usize::try_from(triangle.indices[0]).expect("mesh index fits usize")];
                let material_code = weights
                    .iter()
                    .enumerate()
                    .max_by(|first, second| first.1.total_cmp(second.1))
                    .map_or(2, |(index, _)| index as u8);
                best_distance_squared = distance_squared;
                nearest = Some((
                    -(query - closest).dot(normal),
                    TerrainMaterial::from_code(material_code).unwrap_or(TerrainMaterial::Rock),
                    f64::from(distance_squared.sqrt()),
                ));
            }
        }
        nearest
    }
}

fn point_bounds_distance_squared(point: DVec3, bounds: WorldBounds) -> f64 {
    (0..3)
        .map(|axis| {
            if point[axis] < bounds.minimum.0[axis] {
                bounds.minimum.0[axis] - point[axis]
            } else if point[axis] > bounds.maximum.0[axis] {
                point[axis] - bounds.maximum.0[axis]
            } else {
                0.0
            }
        })
        .map(|distance| distance * distance)
        .sum()
}

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
    let inverse = (va + vb + vc).recip();
    first + ab * (vb * inverse) + ac * (vc * inverse)
}

fn ray_intersects_bounds(
    origin: DVec3,
    direction: DVec3,
    bounds: WorldBounds,
    maximum_distance: f64,
) -> bool {
    let mut minimum_distance = 0.0_f64;
    let mut maximum = maximum_distance;
    for axis in 0..3 {
        if direction[axis].abs() <= f64::EPSILON {
            if origin[axis] < bounds.minimum.0[axis] || origin[axis] > bounds.maximum.0[axis] {
                return false;
            }
            continue;
        }
        let inverse = direction[axis].recip();
        let mut first = (bounds.minimum.0[axis] - origin[axis]) * inverse;
        let mut second = (bounds.maximum.0[axis] - origin[axis]) * inverse;
        if first > second {
            core::mem::swap(&mut first, &mut second);
        }
        minimum_distance = minimum_distance.max(first);
        maximum = maximum.min(second);
        if minimum_distance > maximum {
            return false;
        }
    }
    maximum >= 0.0
}

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

/// Parameters for one independently generated terrain chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerrainMeshRequest {
    /// Selected octree node.
    pub node: TerrainNodeId,
    /// New chunk generation.
    pub generation: u64,
    /// Faces requiring a transition to a node one level coarser.
    pub transition_mask: TerrainTransitionMask,
}

/// CPU timings for the independently measurable extraction stages.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TerrainMeshMetrics {
    /// Prepared-column and density-lattice sampling time.
    pub column_sampling_ms: f64,
    /// Regular-cell polygonization and vertex-cache time.
    pub polygonization_ms: f64,
    /// Transition and temporary-cap generation time.
    pub transitions_caps_ms: f64,
    /// Shared all-groups BVH construction time.
    pub bvh_construction_ms: f64,
}

/// Node-local view of promoted terrain used throughout one mesh job.
///
/// Preparing the view traverses the sparse octree once. Subsequent point and
/// range queries touch only the few promoted bricks overlapping the mesh halo,
/// avoiding a depth-27 root walk for every coarse lattice sample.
#[derive(Clone, Debug, Default)]
pub struct PreparedTerrainRegion<'a> {
    bricks: HashMap<crate::BrickCoord, &'a crate::TerrainBrick>,
}

impl<'a> PreparedTerrainRegion<'a> {
    /// Prepares the full sampling and transition halo for one mesh request.
    ///
    /// # Panics
    ///
    /// Panics if the requested streamed node lies outside `i32` cell space.
    pub fn for_mesh_request(
        terrain: &'a TerrainOctreeSnapshot,
        request: TerrainMeshRequest,
    ) -> Self {
        let stride = 1_i32 << request.node.level;
        let minimum = request
            .node
            .minimum_cell_i64()
            .map(|cell| i32::try_from(cell).expect("streamed node lies in i32 cell space"));
        let edge = BRICK_EDGE_CELLS * stride;
        // Coarse transition gradients reach four fine strides outside a node.
        // The extra cell covers the lower interpolation neighborhood.
        let padding = 4 * stride + 1;
        Self::between(
            terrain,
            WorldCell::new(
                minimum[0] - padding,
                minimum[1] - padding,
                minimum[2] - padding,
            ),
            WorldCell::new(
                minimum[0] + edge + padding,
                minimum[1] + edge + padding,
                minimum[2] + edge + padding,
            ),
        )
    }

    /// Prepares promoted bricks intersecting an inclusive cell range.
    pub fn between(
        terrain: &'a TerrainOctreeSnapshot,
        minimum: WorldCell,
        maximum: WorldCell,
    ) -> Self {
        debug_assert!(minimum.x <= maximum.x);
        debug_assert!(minimum.y <= maximum.y);
        debug_assert!(minimum.z <= maximum.z);
        let bricks = terrain
            .bricks_between(minimum.brick(), maximum.brick())
            .map(|brick| (brick.coordinate(), brick))
            .collect();
        Self { bricks }
    }

    /// Number of promoted bricks retained by this job-local view.
    pub fn promoted_brick_count(&self) -> usize {
        self.bricks.len()
    }

    fn is_empty(&self) -> bool {
        self.bricks.is_empty()
    }

    fn brick(&self, coordinate: crate::BrickCoord) -> Option<&TerrainBrick> {
        self.bricks.get(&coordinate).copied()
    }

    fn minimum_promoted_density_between(
        &self,
        minimum: WorldCell,
        maximum: WorldCell,
    ) -> Option<f32> {
        let minimum_brick = minimum.brick();
        let maximum_brick = maximum.brick();
        let mut result = f32::INFINITY;
        for z in minimum_brick.z..=maximum_brick.z {
            for y in minimum_brick.y..=maximum_brick.y {
                for x in minimum_brick.x..=maximum_brick.x {
                    let Some(brick) = self.brick(crate::BrickCoord::new(x, y, z)) else {
                        continue;
                    };
                    let brick_minimum = brick.coordinate().minimum_cell();
                    let first = IVec3::new(
                        minimum.x.max(brick_minimum.x) - brick_minimum.x,
                        minimum.y.max(brick_minimum.y) - brick_minimum.y,
                        minimum.z.max(brick_minimum.z) - brick_minimum.z,
                    );
                    let last = IVec3::new(
                        maximum.x.min(brick_minimum.x + BRICK_EDGE_CELLS - 1) - brick_minimum.x,
                        maximum.y.min(brick_minimum.y + BRICK_EDGE_CELLS - 1) - brick_minimum.y,
                        maximum.z.min(brick_minimum.z + BRICK_EDGE_CELLS - 1) - brick_minimum.z,
                    );
                    if first == IVec3::ZERO && last == IVec3::splat(BRICK_EDGE_CELLS - 1) {
                        result = result.min(brick.minimum_density());
                        continue;
                    }
                    for local_z in first.z..=last.z {
                        for local_y in first.y..=last.y {
                            for local_x in first.x..=last.x {
                                let density = brick
                                    .sample(IVec3::new(local_x, local_y, local_z))
                                    .expect("clamped coordinate is inside prepared brick")
                                    .density;
                                result = result.min(density);
                            }
                        }
                    }
                }
            }
        }
        result.is_finite().then_some(result)
    }
}

#[derive(Clone, Copy)]
struct LatticePoint {
    sample: TerrainSample,
    normal: Vec3,
    authored_material: Option<TerrainMaterial>,
}

#[derive(Clone, Copy)]
struct LatticeSample {
    sample: TerrainSample,
    authored_material: Option<TerrainMaterial>,
}

#[derive(Clone, Copy)]
struct MeshVertex {
    position: DVec3,
    normal: Vec3,
    material: TerrainMaterial,
}

/// Generates a smooth isosurface chunk.
///
/// Regular and transition cells use the official Transvoxel lookup tables over
/// one shared scalar lattice. Independently generated equal-LOD boundaries are
/// therefore byte-identical.
///
/// # Panics
///
/// Panics only if a requested chunk exceeds the `u32` mesh-index contract.
pub fn mesh_chunk(
    field: &TerrainField,
    edits: &TerrainOctreeSnapshot,
    request: TerrainMeshRequest,
) -> TerrainMeshChunk {
    mesh_chunk_profiled(field, edits, request).0
}

/// Generates a chunk and returns stage-level CPU timings for diagnostics.
///
/// # Panics
///
/// Panics if the request is outside streamed LOD levels zero through five or
/// if one chunk exceeds the `u32` mesh-index contract.
pub fn mesh_chunk_profiled(
    field: &TerrainField,
    edits: &TerrainOctreeSnapshot,
    request: TerrainMeshRequest,
) -> (TerrainMeshChunk, TerrainMeshMetrics) {
    let prepared = PreparedTerrainRegion::for_mesh_request(edits, request);
    mesh_chunk_profiled_prepared(field, &prepared, request)
}

/// Generates a chunk using an already prepared node-local edit view.
///
/// This entry point lets bounded workers prepare and account for job-local
/// terrain data before beginning expensive sampling.
///
/// # Panics
///
/// Panics if the request is outside streamed LOD levels zero through five or
/// if one chunk exceeds the `u32` mesh-index contract.
pub fn mesh_chunk_profiled_prepared(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    request: TerrainMeshRequest,
) -> (TerrainMeshChunk, TerrainMeshMetrics) {
    assert!(
        request.node.level <= 5,
        "streamed mesh LOD is level 0 through 5"
    );
    let stride = 1_i32 << request.node.level;
    let cubes = BRICK_EDGE_CELLS;
    let minimum_raw = request.node.minimum_cell_i64();
    let minimum = WorldCell::new(
        i32::try_from(minimum_raw[0]).expect("streamed node lies in i32 cell space"),
        i32::try_from(minimum_raw[1]).expect("streamed node lies in i32 cell space"),
        i32::try_from(minimum_raw[2]).expect("streamed node lies in i32 cell space"),
    );
    let minimum_position = minimum.centre().0 - DVec3::splat(TERRAIN_CELL_METERS * 0.5);
    let maximum_cell = WorldCell::new(
        minimum.x + cubes * stride,
        minimum.y + cubes * stride,
        minimum.z + cubes * stride,
    );
    let maximum_position = maximum_cell.centre().0 - DVec3::splat(TERRAIN_CELL_METERS * 0.5);
    // Vertices are part of the GPU-facing f32 contract. Bounds use the same
    // representable endpoints so a rounded boundary vertex remains inside.
    let bounds = WorldBounds {
        minimum: WorldPosition(minimum_position.as_vec3().as_dvec3() - DVec3::splat(1.0e-6)),
        maximum: WorldPosition(maximum_position.as_vec3().as_dvec3() + DVec3::splat(1.0e-6)),
    };
    let mut chunk = TerrainMeshChunk {
        node: request.node,
        origin: WorldPosition(minimum_position),
        bounds,
        generation: request.generation,
        sample_spacing_metres: f64::from(stride) * TERRAIN_CELL_METERS,
        transition_mask: request.transition_mask,
        ..TerrainMeshChunk::default()
    };

    let lattice_edge = usize::try_from(cubes + 1).expect("chunk edge is positive");
    let sampling_started = Instant::now();
    let halo = sample_halo(field, edits, request.node, minimum, cubes, stride);
    let mut lattice = lattice_from_halo(&halo, cubes);
    if request.node.level < 5 {
        synchronize_edited_boundary_lattice(
            field,
            edits,
            minimum,
            stride,
            lattice_edge,
            request.transition_mask,
            &mut lattice,
        );
    }
    let column_sampling_ms = sampling_started.elapsed().as_secs_f64() * 1_000.0;

    let polygonization_started = Instant::now();
    for z in 0..cubes {
        for y in 0..cubes {
            for x in 0..cubes {
                let cube_minimum = WorldCell::new(
                    minimum.x + x * stride,
                    minimum.y + y * stride,
                    minimum.z + z * stride,
                );
                let x = usize::try_from(x).expect("cube coordinate is positive");
                let y = usize::try_from(y).expect("cube coordinate is positive");
                let z = usize::try_from(z).expect("cube coordinate is positive");
                let lattice_index =
                    |x: usize, y: usize, z: usize| x + y * lattice_edge + z * lattice_edge.pow(2);
                let samples = CUBE_CORNERS.map(|offset| {
                    lattice[lattice_index(
                        x + usize::try_from(offset[0]).expect("corner is positive"),
                        y + usize::try_from(offset[1]).expect("corner is positive"),
                        z + usize::try_from(offset[2]).expect("corner is positive"),
                    )]
                });
                polygonise_cube(cube_minimum, stride, samples, &mut chunk);
            }
        }
    }
    let polygonization_ms = polygonization_started.elapsed().as_secs_f64() * 1_000.0;
    let transition_started = Instant::now();
    for face in TerrainFace::ALL {
        generate_face_cap(&lattice, lattice_edge, face, &mut chunk);
        if request.transition_mask.contains(face) {
            generate_transition_face(&lattice, lattice_edge, face, &mut chunk);
        }
    }
    let transitions_caps_ms = transition_started.elapsed().as_secs_f64() * 1_000.0;
    // The edge cache is extraction-only. `HashMap::clear` would retain its
    // potentially large allocation in every published chunk for the rest of
    // the world's lifetime.
    chunk.vertex_cache = LatticeEdgeVertexCache::default();
    release_growth_slack(&mut chunk);
    make_vertices_node_local(&mut chunk);
    let bvh_started = Instant::now();
    chunk.triangle_bvh = build_triangle_bvh(chunk.origin, &chunk.vertices, &chunk.index_groups);
    let bvh_construction_ms = bvh_started.elapsed().as_secs_f64() * 1_000.0;
    (
        chunk,
        TerrainMeshMetrics {
            column_sampling_ms,
            polygonization_ms,
            transitions_caps_ms,
            bvh_construction_ms,
        },
    )
}

/// Growth slack is retained for as long as a chunk stays published. Across a
/// streamed cut it was about 30% of the geometry vectors' capacity.
fn release_growth_slack(chunk: &mut TerrainMeshChunk) {
    chunk.vertices.shrink_to_fit();
    chunk.normals.shrink_to_fit();
    chunk.material_weights.shrink_to_fit();
    chunk.index_groups.regular.shrink_to_fit();
    for indices in chunk
        .index_groups
        .transitions
        .iter_mut()
        .chain(&mut chunk.index_groups.caps)
    {
        indices.shrink_to_fit();
    }
}

fn make_vertices_node_local(chunk: &mut TerrainMeshChunk) {
    let origin = chunk.origin.0.as_vec3();
    for vertex in &mut chunk.vertices {
        *vertex = (Vec3::from_array(*vertex) - origin).to_array();
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "fine and coarse halo preparation share indexing contracts"
)]
fn sample_halo(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    node: TerrainNodeId,
    minimum: WorldCell,
    cubes: i32,
    stride: i32,
) -> Vec<LatticeSample> {
    let lattice_edge = usize::try_from(cubes + 1).expect("chunk edge is positive");
    let halo_edge = lattice_edge + 2;
    let mut halo = Vec::with_capacity(halo_edge.pow(3));
    if stride == 1 {
        let cell_edge = halo_edge + 1;
        let columns = field.cached_mesh_columns(node, || {
            let mut columns = Vec::with_capacity(cell_edge.pow(2));
            for z in -2..=cubes + 1 {
                for x in -2..=cubes + 1 {
                    let position = WorldCell::new(minimum.x + x, minimum.y, minimum.z + z).centre();
                    columns.push(field.sample_column(position.0.x, position.0.z));
                }
            }
            columns
        });
        let column_index = |x: i32, z: i32| {
            usize::try_from(x + 2).expect("halo column is positive")
                + usize::try_from(z + 2).expect("halo column is positive") * cell_edge
        };
        let mut cells = Vec::with_capacity(cell_edge.pow(3));
        let mut edited_cells = Vec::with_capacity(cell_edge.pow(3));
        for z in -2..=cubes + 1 {
            for y in -2..=cubes + 1 {
                for x in -2..=cubes + 1 {
                    let cell = WorldCell::new(minimum.x + x, minimum.y + y, minimum.z + z);
                    let coordinate = cell.brick();
                    let generated = field.sample_cell_in_column(cell, columns[column_index(x, z)]);
                    let sample = if edits.is_empty() {
                        generated
                    } else {
                        edits
                            .brick(coordinate)
                            .and_then(|brick| brick.sample(cell.local_in_brick()))
                            .unwrap_or(generated)
                    };
                    cells.push(sample);
                    edited_cells.push(sample != generated);
                }
            }
        }
        let cell_index = |x: i32, y: i32, z: i32| {
            usize::try_from(x + 2).expect("halo cell is positive")
                + usize::try_from(y + 2).expect("halo cell is positive") * cell_edge
                + usize::try_from(z + 2).expect("halo cell is positive") * cell_edge.pow(2)
        };
        for z in -1..=cubes + 1 {
            for y in -1..=cubes + 1 {
                for x in -1..=cubes + 1 {
                    halo.push(lattice_sample_from_cells(
                        &cells,
                        &edited_cells,
                        cell_index,
                        x,
                        y,
                        z,
                    ));
                }
            }
        }
    } else {
        let prepared_edge = halo_edge * 2;
        let columns = field.cached_mesh_columns(node, || {
            let mut columns = Vec::with_capacity(prepared_edge.pow(2));
            for z in -1..=cubes + 1 {
                for z_offset in [-1, 0] {
                    for x in -1..=cubes + 1 {
                        for x_offset in [-1, 0] {
                            let cell = WorldCell::new(
                                minimum.x + x * stride + x_offset,
                                minimum.y,
                                minimum.z + z * stride + z_offset,
                            );
                            let position = cell.centre();
                            columns.push(field.sample_column(position.0.x, position.0.z));
                        }
                    }
                }
            }
            columns
        });
        let column_index = |x: i32, z: i32, x_offset: usize, z_offset: usize| {
            let x = usize::try_from(x + 1).expect("coarse halo column is positive") * 2 + x_offset;
            let z = usize::try_from(z + 1).expect("coarse halo column is positive") * 2 + z_offset;
            x + z * prepared_edge
        };
        for z in -1..=cubes + 1 {
            for y in -1..=cubes + 1 {
                for x in -1..=cubes + 1 {
                    let prepared = [
                        columns[column_index(x, z, 0, 0)],
                        columns[column_index(x, z, 1, 0)],
                        columns[column_index(x, z, 0, 1)],
                        columns[column_index(x, z, 1, 1)],
                    ];
                    halo.push(coarse_sample_in_columns(
                        field,
                        edits,
                        WorldCell::new(
                            minimum.x + x * stride,
                            minimum.y + y * stride,
                            minimum.z + z * stride,
                        ),
                        stride,
                        prepared,
                    ));
                }
            }
        }
    }
    halo
}

fn lattice_from_halo(halo: &[LatticeSample], cubes: i32) -> Vec<LatticePoint> {
    let lattice_edge = usize::try_from(cubes + 1).expect("chunk edge is positive");
    let halo_edge = lattice_edge + 2;
    let halo_index = |x: usize, y: usize, z: usize| x + y * halo_edge + z * halo_edge.pow(2);
    let mut lattice = Vec::with_capacity(lattice_edge.pow(3));
    for z in 0..=cubes {
        for y in 0..=cubes {
            for x in 0..=cubes {
                let x = usize::try_from(x + 1).expect("halo coordinate is positive");
                let y = usize::try_from(y + 1).expect("halo coordinate is positive");
                let z = usize::try_from(z + 1).expect("halo coordinate is positive");
                let sampled = halo[halo_index(x, y, z)];
                let gradient = Vec3::new(
                    halo[halo_index(x + 1, y, z)].sample.density
                        - halo[halo_index(x - 1, y, z)].sample.density,
                    halo[halo_index(x, y + 1, z)].sample.density
                        - halo[halo_index(x, y - 1, z)].sample.density,
                    halo[halo_index(x, y, z + 1)].sample.density
                        - halo[halo_index(x, y, z - 1)].sample.density,
                );
                lattice.push(LatticePoint {
                    sample: sampled.sample,
                    normal: (-gradient).normalize_or(Vec3::Y),
                    authored_material: sampled.authored_material,
                });
            }
        }
    }
    lattice
}

fn synchronize_edited_boundary_lattice(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    minimum: WorldCell,
    fine_stride: i32,
    lattice_edge: usize,
    transition_mask: TerrainTransitionMask,
    lattice: &mut [LatticePoint],
) {
    if edits.is_empty() {
        return;
    }
    let cubes = lattice_edge - 1;
    let coarse_stride = fine_stride * 2;
    let maximum = WorldCell::new(
        minimum.x
            + i32::try_from(cubes).expect("lattice edge fits i32") * fine_stride
            + coarse_stride
            - 1,
        minimum.y
            + i32::try_from(cubes).expect("lattice edge fits i32") * fine_stride
            + coarse_stride
            - 1,
        minimum.z
            + i32::try_from(cubes).expect("lattice edge fits i32") * fine_stride
            + coarse_stride
            - 1,
    );
    if edits
        .minimum_promoted_density_between(minimum, maximum)
        .is_none()
    {
        return;
    }
    let mut coarse_samples = HashMap::<WorldCell, LatticeSample>::new();
    let mut coarse_columns = HashMap::<(i32, i32), crate::generation::TerrainColumnSample>::new();
    for face in TerrainFace::ALL {
        for v in (0..=cubes).step_by(2) {
            for u in (0..=cubes).step_by(2) {
                let (x, y, z) = face_coordinate(face, u, v, cubes);
                let mut boundary_faces = 0_u8;
                for (coordinate, negative, positive) in [
                    (x, TerrainFace::NegativeX, TerrainFace::PositiveX),
                    (y, TerrainFace::NegativeY, TerrainFace::PositiveY),
                    (z, TerrainFace::NegativeZ, TerrainFace::PositiveZ),
                ] {
                    if coordinate == 0 {
                        boundary_faces |= 1 << negative as u8;
                    } else if coordinate == cubes {
                        boundary_faces |= 1 << positive as u8;
                    }
                }
                if !transition_mask.synchronizes_boundary_feature(boundary_faces) {
                    continue;
                }
                let cell = WorldCell::new(
                    minimum.x
                        + i32::try_from(x).expect("transition coordinate fits i32") * fine_stride,
                    minimum.y
                        + i32::try_from(y).expect("transition coordinate fits i32") * fine_stride,
                    minimum.z
                        + i32::try_from(z).expect("transition coordinate fits i32") * fine_stride,
                );
                let maximum = WorldCell::new(
                    cell.x + coarse_stride - 1,
                    cell.y + coarse_stride - 1,
                    cell.z + coarse_stride - 1,
                );
                if edits
                    .minimum_promoted_density_between(cell, maximum)
                    .is_none()
                {
                    continue;
                }
                lattice[x + y * lattice_edge + z * lattice_edge.pow(2)] =
                    transition_coarse_lattice_point(
                        field,
                        edits,
                        cell,
                        coarse_stride,
                        &mut coarse_samples,
                        &mut coarse_columns,
                    );
            }
        }
    }
}

fn polygonise_cube(
    minimum: WorldCell,
    stride: i32,
    samples: [LatticePoint; 8],
    chunk: &mut TerrainMeshChunk,
) {
    let mut positions = [DVec3::ZERO; 8];
    for (index, offset) in CUBE_CORNERS.into_iter().enumerate() {
        let cell = WorldCell::new(
            minimum.x + offset[0] * stride,
            minimum.y + offset[1] * stride,
            minimum.z + offset[2] * stride,
        );
        positions[index] = cell.centre().0 - DVec3::splat(TERRAIN_CELL_METERS * 0.5);
    }
    let case = samples
        .iter()
        .enumerate()
        .fold(0_u8, |case, (index, sample)| {
            case | if sample.sample.is_solid() {
                1 << index
            } else {
                0
            }
        });
    if case == 0 || case == u8::MAX {
        return;
    }
    let cell = REGULAR_CELL_DATA[usize::from(REGULAR_CELL_CLASS[usize::from(case)])];
    let vertex_count = usize::from(cell.geometry_counts >> 4);
    let triangle_count = usize::from(cell.geometry_counts & 0x0f);
    let mut vertices = Vec::with_capacity(vertex_count);
    for &data in &REGULAR_VERTEX_DATA[usize::from(case)][..vertex_count] {
        let edge = data & 0xff;
        let first = usize::from((edge >> 4) as u8);
        let second = usize::from((edge & 0x0f) as u8);
        let (solid, empty) = if samples[first].sample.is_solid() {
            (first, second)
        } else {
            (second, first)
        };
        let mut vertex = crossing(solid, empty, positions, samples);
        apply_transition_inset(&mut vertex, chunk);
        vertices.push(vertex);
    }
    for triangle in cell.vertex_index[..triangle_count * 3].chunks_exact(3) {
        emit_oriented_triangle(
            chunk,
            [
                vertices[usize::from(triangle[0])],
                vertices[usize::from(triangle[1])],
                vertices[usize::from(triangle[2])],
            ],
            Vec3::ZERO,
            IndexGroup::Regular,
        );
    }
}

fn crossing(
    solid: usize,
    empty: usize,
    positions: [DVec3; 8],
    samples: [LatticePoint; 8],
) -> MeshVertex {
    let solid_density = f64::from(samples[solid].sample.density);
    let empty_density = f64::from(samples[empty].sample.density);
    let along = solid_density / (solid_density - empty_density);
    let along = along.clamp(0.0, 1.0);
    MeshVertex {
        position: positions[solid].lerp(positions[empty], along),
        normal: samples[solid]
            .normal
            .lerp(samples[empty].normal, along as f32)
            .normalize_or(Vec3::Y),
        material: crossing_material(samples[solid], samples[empty]),
    }
}

fn crossing_material(first: LatticePoint, second: LatticePoint) -> TerrainMaterial {
    if let Some(material) = first.authored_material.or(second.authored_material) {
        material
    } else if first.sample.is_solid() {
        second.sample.material
    } else {
        first.sample.material
    }
}

fn apply_transition_inset(vertex: &mut MeshVertex, chunk: &TerrainMeshChunk) {
    if chunk.transition_mask == TerrainTransitionMask::NONE {
        return;
    }

    let spacing = chunk.sample_spacing_metres;
    let minimum = chunk.origin.0;
    let maximum = minimum + DVec3::splat(f64::from(BRICK_EDGE_CELLS) * spacing);
    let epsilon = spacing * 1.0e-6;

    // A vertex shared with a non-transition face must keep its primary
    // position so the equal-LOD neighbor remains byte-identical.
    for face in TerrainFace::ALL {
        if !chunk.transition_mask.contains(face)
            && vertex_on_face(vertex.position, minimum, maximum, face, epsilon)
        {
            return;
        }
    }

    let mut delta = DVec3::ZERO;
    for face in TerrainFace::ALL {
        if !chunk.transition_mask.contains(face) {
            continue;
        }
        let (distance, direction) = match face {
            TerrainFace::NegativeX => (vertex.position.x - minimum.x, DVec3::X),
            TerrainFace::PositiveX => (maximum.x - vertex.position.x, DVec3::NEG_X),
            TerrainFace::NegativeY => (vertex.position.y - minimum.y, DVec3::Y),
            TerrainFace::PositiveY => (maximum.y - vertex.position.y, DVec3::NEG_Y),
            TerrainFace::NegativeZ => (vertex.position.z - minimum.z, DVec3::Z),
            TerrainFace::PositiveZ => (maximum.z - vertex.position.z, DVec3::NEG_Z),
        };
        if distance <= spacing {
            let weight = (1.0 - distance / spacing).clamp(0.0, 1.0);
            delta += direction * (weight * spacing * 0.25);
        }
    }

    let normal = vertex.normal.as_dvec3();
    delta -= normal * delta.dot(normal);
    vertex.position += delta.clamp(DVec3::splat(-spacing), DVec3::splat(spacing));
}

fn vertex_on_face(
    position: DVec3,
    minimum: DVec3,
    maximum: DVec3,
    face: TerrainFace,
    epsilon: f64,
) -> bool {
    let distance = match face {
        TerrainFace::NegativeX => position.x - minimum.x,
        TerrainFace::PositiveX => maximum.x - position.x,
        TerrainFace::NegativeY => position.y - minimum.y,
        TerrainFace::PositiveY => maximum.y - position.y,
        TerrainFace::NegativeZ => position.z - minimum.z,
        TerrainFace::PositiveZ => maximum.z - position.z,
    };
    distance.abs() <= epsilon
}

fn emit_oriented_triangle(
    chunk: &mut TerrainMeshChunk,
    mut triangle: [MeshVertex; 3],
    fallback_outward: Vec3,
    group: IndexGroup,
) {
    let first = triangle[0].position.as_vec3();
    let geometric =
        (triangle[1].position.as_vec3() - first).cross(triangle[2].position.as_vec3() - first);
    let smooth_outward = triangle.iter().map(|vertex| vertex.normal).sum::<Vec3>();
    let expected_outward = if smooth_outward.length_squared() > 1.0e-12 {
        smooth_outward
    } else {
        fallback_outward
    };
    if geometric.dot(expected_outward) < 0.0 {
        triangle.swap(1, 2);
    }
    if let Some(indices) = append_triangle(chunk, triangle) {
        match group {
            IndexGroup::Regular => chunk.index_groups.regular.extend_from_slice(&indices),
            IndexGroup::Transition(face) => {
                chunk.index_groups.transitions[face.index()].extend_from_slice(&indices);
            }
            IndexGroup::Cap(face) => {
                chunk.index_groups.caps[face.index()].extend_from_slice(&indices);
            }
        }
    }
}

#[derive(Clone, Copy)]
enum IndexGroup {
    Regular,
    Transition(TerrainFace),
    Cap(TerrainFace),
}

fn append_triangle(chunk: &mut TerrainMeshChunk, triangle: [MeshVertex; 3]) -> Option<[u32; 3]> {
    if (triangle[1].position - triangle[0].position)
        .cross(triangle[2].position - triangle[0].position)
        .length_squared()
        <= 1.0e-20
    {
        return None;
    }
    let mut indices = [0_u32; 3];
    for (target, vertex) in indices.iter_mut().zip(triangle) {
        let position = vertex.position.as_vec3().to_array();
        let normal = vertex.normal.to_array();
        let mut weights = [0.0; TerrainMaterial::COUNT];
        weights[vertex.material.code() as usize] = 1.0;
        let key = (
            position.map(f32::to_bits),
            normal.map(f32::to_bits),
            weights.map(f32::to_bits),
        );
        *target = if let Some(&known) = chunk.vertex_cache.vertices.get(&key) {
            known
        } else {
            let index =
                u32::try_from(chunk.vertices.len()).expect("one chunk has fewer than u32 vertices");
            chunk.vertices.push(position);
            chunk.normals.push(normal);
            chunk.material_weights.push(weights);
            chunk.vertex_cache.vertices.insert(key, index);
            index
        };
    }
    Some(indices)
}

fn coarse_sample_in_columns(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    cell: WorldCell,
    stride: i32,
    columns: [crate::generation::TerrainColumnSample; 4],
) -> LatticeSample {
    let direct = lattice_sample_in_columns(field, edits, cell, columns);
    if edits.is_empty() {
        return direct;
    }
    let maximum = WorldCell::new(
        cell.x + stride - 1,
        cell.y + stride - 1,
        cell.z + stride - 1,
    );
    let Some(minimum_promoted) = edits.minimum_promoted_density_between(cell, maximum) else {
        return direct;
    };
    // A promoted negative sample must remain visible at coarser LODs even when
    // it lies between coarse lattice points.
    if minimum_promoted < direct.sample.density {
        LatticeSample {
            sample: TerrainSample {
                compaction: 0,
                density: minimum_promoted,
                material: direct.sample.material,
            },
            authored_material: direct.authored_material,
        }
    } else {
        direct
    }
}

fn lattice_sample_in_columns(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    upper_cell: WorldCell,
    columns: [crate::generation::TerrainColumnSample; 4],
) -> LatticeSample {
    // `WorldCell` values are cell-centred for exact removal accounting, while
    // the meshing lattice lies on their corners. Full signed distances retain
    // smooth procedural interpolation; only neighborhoods containing a real
    // edit are reconstructed as bounded cell occupancies.
    let mut samples = [TerrainSample {
        compaction: 0,
        density: 0.0,
        material: TerrainMaterial::Rock,
    }; 8];
    let mut edited = [false; 8];
    let mut sample_index = 0;
    for (column_index, (x, z)) in [(-1, -1), (0, -1), (-1, 0), (0, 0)].into_iter().enumerate() {
        for y in -1..=0 {
            let cell = WorldCell::new(upper_cell.x + x, upper_cell.y + y, upper_cell.z + z);
            let generated = field.sample_cell_in_column(cell, columns[column_index]);
            let sample = if edits.is_empty() {
                generated
            } else {
                edits
                    .brick(cell.brick())
                    .and_then(|brick| brick.sample(cell.local_in_brick()))
                    .unwrap_or(generated)
            };
            samples[sample_index] = sample;
            edited[sample_index] = sample != generated;
            sample_index += 1;
        }
    }
    blend_lattice_samples(samples, edited)
}

fn lattice_sample_from_cells(
    cells: &[TerrainSample],
    edited_cells: &[bool],
    cell_index: impl Fn(i32, i32, i32) -> usize,
    upper_x: i32,
    upper_y: i32,
    upper_z: i32,
) -> LatticeSample {
    let mut samples = [TerrainSample {
        compaction: 0,
        density: 0.0,
        material: TerrainMaterial::Rock,
    }; 8];
    let mut edited = [false; 8];
    let mut sample_index = 0;
    for z in -1..=0 {
        for y in -1..=0 {
            for x in -1..=0 {
                let index = cell_index(upper_x + x, upper_y + y, upper_z + z);
                samples[sample_index] = cells[index];
                edited[sample_index] = edited_cells[index];
                sample_index += 1;
            }
        }
    }
    blend_lattice_samples(samples, edited)
}

fn blend_lattice_samples(samples: [TerrainSample; 8], edited: [bool; 8]) -> LatticeSample {
    // Plastic compaction retains continuous signed distances. Binary brushes
    // need bounded occupancy reconstruction; applying it to a tiny soil load
    // would snap the previously procedural surface on its very first commit.
    let reconstructing_edit = samples
        .iter()
        .zip(edited)
        .any(|(sample, edited)| edited && sample.compaction == 0);
    let half_cell = TERRAIN_CELL_METERS as f32 * 0.5;
    let mut density = 0.0;
    let mut material = TerrainMaterial::Rock;
    let mut nearest_surface = f32::INFINITY;
    let mut authored_material = None;
    let mut nearest_authored_solid = f32::INFINITY;
    for (sample, edited) in samples.into_iter().zip(edited) {
        density += if reconstructing_edit {
            sample.density.clamp(-half_cell, half_cell)
        } else {
            sample.density
        };
        if sample.density.abs() < nearest_surface {
            nearest_surface = sample.density.abs();
            material = sample.material;
        }
        if edited && sample.is_solid() && sample.density.abs() < nearest_authored_solid {
            nearest_authored_solid = sample.density.abs();
            authored_material = Some(sample.material);
        }
    }
    LatticeSample {
        sample: TerrainSample {
            compaction: 0,
            density: density / 8.0,
            material,
        },
        authored_material,
    }
}

fn generate_face_cap(
    lattice: &[LatticePoint],
    lattice_edge: usize,
    face: TerrainFace,
    chunk: &mut TerrainMeshChunk,
) {
    let cubes = lattice_edge - 1;
    let outward = face_normal(face);
    for v in 0..cubes {
        for u in 0..cubes {
            let coordinates = [(u, v), (u + 1, v), (u + 1, v + 1), (u, v + 1)];
            let points = coordinates.map(|(u, v)| {
                let (x, y, z) = face_coordinate(face, u, v, cubes);
                let lattice_index = x + y * lattice_edge + z * lattice_edge.pow(2);
                let position = chunk.origin.0
                    + DVec3::new(
                        f64::from(u32::try_from(x).expect("cap coordinate fits u32")),
                        f64::from(u32::try_from(y).expect("cap coordinate fits u32")),
                        f64::from(u32::try_from(z).expect("cap coordinate fits u32")),
                    ) * chunk.sample_spacing_metres;
                (lattice[lattice_index], position)
            });
            let case = points
                .iter()
                .enumerate()
                .fold(0_u8, |case, (index, point)| {
                    case | if point.0.sample.is_solid() {
                        1 << index
                    } else {
                        0
                    }
                });
            if case == 0 {
                continue;
            }
            if case == 0b0101 || case == 0b1010 {
                for corner in (0..4).filter(|&corner| points[corner].0.sample.is_solid()) {
                    let previous = (corner + 3) % 4;
                    let next = (corner + 1) % 4;
                    emit_oriented_triangle(
                        chunk,
                        [
                            cap_crossing(points[corner], points[previous], outward),
                            cap_vertex(points[corner], outward),
                            cap_crossing(points[corner], points[next], outward),
                        ],
                        outward,
                        IndexGroup::Cap(face),
                    );
                }
            } else {
                let mut polygon = [cap_vertex(points[0], outward); 6];
                let mut polygon_length = 0;
                for current in 0..4 {
                    let next = (current + 1) % 4;
                    if points[current].0.sample.is_solid() {
                        polygon[polygon_length] = cap_vertex(points[current], outward);
                        polygon_length += 1;
                    }
                    if points[current].0.sample.is_solid() != points[next].0.sample.is_solid() {
                        polygon[polygon_length] =
                            cap_crossing(points[current], points[next], outward);
                        polygon_length += 1;
                    }
                }
                for index in 1..polygon_length.saturating_sub(1) {
                    emit_oriented_triangle(
                        chunk,
                        [polygon[0], polygon[index], polygon[index + 1]],
                        outward,
                        IndexGroup::Cap(face),
                    );
                }
            }
        }
    }
}

fn cap_vertex(point: (LatticePoint, DVec3), outward: Vec3) -> MeshVertex {
    MeshVertex {
        position: point.1,
        normal: outward,
        material: point.0.sample.material,
    }
}

fn cap_crossing(
    first: (LatticePoint, DVec3),
    second: (LatticePoint, DVec3),
    outward: Vec3,
) -> MeshVertex {
    let first_density = f64::from(first.0.sample.density);
    let second_density = f64::from(second.0.sample.density);
    let along = (first_density / (first_density - second_density)).clamp(0.0, 1.0);
    let material = crossing_material(first.0, second.0);
    MeshVertex {
        position: first.1.lerp(second.1, along),
        normal: outward,
        material,
    }
}

fn generate_transition_face(
    lattice: &[LatticePoint],
    lattice_edge: usize,
    face: TerrainFace,
    chunk: &mut TerrainMeshChunk,
) {
    let cubes = lattice_edge - 1;
    for v in (0..cubes).step_by(2) {
        for u in (0..cubes).step_by(2) {
            let fine = array::from_fn::<_, 9, _>(|index| {
                let du = index % 3;
                let dv = index / 3;
                let (x, y, z) = face_coordinate(face, u + du, v + dv, cubes);
                let point = lattice[x + y * lattice_edge + z * lattice_edge.pow(2)];
                let position = chunk.origin.0
                    + DVec3::new(
                        f64::from(u32::try_from(x).expect("transition coordinate fits u32")),
                        f64::from(u32::try_from(y).expect("transition coordinate fits u32")),
                        f64::from(u32::try_from(z).expect("transition coordinate fits u32")),
                    ) * chunk.sample_spacing_metres;
                (point, position)
            });
            // The official tables encode the eight perimeter samples clockwise,
            // followed by the centre, rather than the row-major order above.
            let case = [0, 1, 2, 5, 8, 7, 6, 3, 4].into_iter().enumerate().fold(
                0_u16,
                |case, (bit, point)| {
                    case | if fine[point].0.sample.is_solid() {
                        1 << bit
                    } else {
                        0
                    }
                },
            );
            if case == 0 || case == 0x1ff {
                continue;
            }
            let class = TRANSITION_CELL_CLASS[usize::from(case)];
            let reverse = class & 0x80 != 0;
            let cell = TRANSITION_CELL_DATA[usize::from(class & 0x7f)];
            let vertex_count = usize::from(cell.geometry_counts >> 4);
            let triangle_count = usize::from(cell.geometry_counts & 0x0f);
            let mut vertices = Vec::with_capacity(vertex_count);
            for &data in &TRANSITION_VERTEX_DATA[usize::from(case)][..vertex_count] {
                let edge = data & 0xff;
                let first_index = usize::from((edge >> 4) as u8);
                let second_index = usize::from((edge & 0x0f) as u8);
                let first = transition_point(first_index, &fine);
                let second = transition_point(second_index, &fine);
                let first_density = f64::from(first.0.sample.density);
                let second_density = f64::from(second.0.sample.density);
                let along = (first_density / (first_density - second_density)).clamp(0.0, 1.0);
                let mut vertex = MeshVertex {
                    position: first.1.lerp(second.1, along),
                    normal: first
                        .0
                        .normal
                        .lerp(second.0.normal, along as f32)
                        .normalize_or(face_normal(face)),
                    material: crossing_material(first.0, second.0),
                };
                if first_index < 9 || second_index < 9 {
                    apply_transition_inset(&mut vertex, chunk);
                }
                vertices.push(vertex);
            }
            for triangle in cell.vertex_index[..triangle_count * 3].chunks_exact(3) {
                let mut triangle = [
                    vertices[usize::from(triangle[0])],
                    vertices[usize::from(triangle[1])],
                    vertices[usize::from(triangle[2])],
                ];
                if reverse {
                    triangle.swap(1, 2);
                }
                emit_oriented_triangle(chunk, triangle, Vec3::ZERO, IndexGroup::Transition(face));
            }
        }
    }
}

fn transition_coarse_lattice_point(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    cell: WorldCell,
    stride: i32,
    samples: &mut HashMap<WorldCell, LatticeSample>,
    columns: &mut HashMap<(i32, i32), crate::generation::TerrainColumnSample>,
) -> LatticePoint {
    let sample = transition_coarse_sample(field, edits, cell, stride, samples, columns);
    let density =
        |offset: [i32; 3],
         samples: &mut HashMap<WorldCell, LatticeSample>,
         columns: &mut HashMap<(i32, i32), crate::generation::TerrainColumnSample>| {
            transition_coarse_sample(
                field,
                edits,
                WorldCell::new(
                    cell.x + offset[0] * stride,
                    cell.y + offset[1] * stride,
                    cell.z + offset[2] * stride,
                ),
                stride,
                samples,
                columns,
            )
            .sample
            .density
        };
    let gradient = Vec3::new(
        density([1, 0, 0], samples, columns) - density([-1, 0, 0], samples, columns),
        density([0, 1, 0], samples, columns) - density([0, -1, 0], samples, columns),
        density([0, 0, 1], samples, columns) - density([0, 0, -1], samples, columns),
    );
    LatticePoint {
        sample: sample.sample,
        normal: (-gradient).normalize_or(Vec3::Y),
        authored_material: sample.authored_material,
    }
}

fn transition_coarse_sample(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    cell: WorldCell,
    stride: i32,
    samples: &mut HashMap<WorldCell, LatticeSample>,
    columns: &mut HashMap<(i32, i32), crate::generation::TerrainColumnSample>,
) -> LatticeSample {
    if let Some(&sample) = samples.get(&cell) {
        return sample;
    }
    let prepared = [(-1, -1), (0, -1), (-1, 0), (0, 0)].map(|(x, z)| {
        let column_cell = WorldCell::new(cell.x + x, cell.y, cell.z + z);
        *columns
            .entry((column_cell.x, column_cell.z))
            .or_insert_with(|| {
                let position = column_cell.centre();
                field.sample_column(position.0.x, position.0.z)
            })
    });
    let sample = coarse_sample_in_columns(field, edits, cell, stride, prepared);
    samples.insert(cell, sample);
    sample
}

fn transition_point(index: usize, fine: &[(LatticePoint, DVec3); 9]) -> (LatticePoint, DVec3) {
    let reuse_data = TRANSITION_CORNER_DATA[index];
    debug_assert!(reuse_data <= 0x87);
    match index {
        0..=8 => fine[index],
        9 => fine[0],
        10 => fine[2],
        11 => fine[6],
        12 => fine[8],
        _ => unreachable!("official transition endpoint is 0 through C"),
    }
}

fn face_coordinate(face: TerrainFace, u: usize, v: usize, cubes: usize) -> (usize, usize, usize) {
    match face {
        TerrainFace::NegativeX => (0, u, v),
        TerrainFace::PositiveX => (cubes, u, v),
        TerrainFace::NegativeY => (u, 0, v),
        TerrainFace::PositiveY => (u, cubes, v),
        TerrainFace::NegativeZ => (u, v, 0),
        TerrainFace::PositiveZ => (u, v, cubes),
    }
}

fn face_normal(face: TerrainFace) -> Vec3 {
    match face {
        TerrainFace::NegativeX => Vec3::NEG_X,
        TerrainFace::PositiveX => Vec3::X,
        TerrainFace::NegativeY => Vec3::NEG_Y,
        TerrainFace::PositiveY => Vec3::Y,
        TerrainFace::NegativeZ => Vec3::NEG_Z,
        TerrainFace::PositiveZ => Vec3::Z,
    }
}

fn build_triangle_bvh(
    origin: WorldPosition,
    vertices: &[[f32; 3]],
    groups: &TerrainIndexGroups,
) -> TriangleBvh {
    let mut triangles = Vec::new();
    let mut append = |indices: &[u32], group_mask: TerrainTriangleGroupMask| {
        triangles.extend(indices.chunks_exact(3).map(|indices| {
            let triangle = TriangleBvhTriangle {
                indices: [indices[0], indices[1], indices[2]],
                group_mask,
            };
            let bounds = triangle_bounds(origin, vertices, triangle.indices);
            BuildTriangle {
                triangle,
                bounds,
                centroid: (bounds.minimum.0 + bounds.maximum.0) * 0.5,
            }
        }));
    };
    append(&groups.regular, TerrainTriangleGroupMask::REGULAR);
    for face in TerrainFace::ALL {
        append(
            &groups.transitions[face.index()],
            TerrainTriangleGroupMask::transition(face),
        );
        append(
            &groups.caps[face.index()],
            TerrainTriangleGroupMask::cap(face),
        );
    }
    if triangles.is_empty() {
        return TriangleBvh::default();
    }
    let triangle_count = triangles.len();
    let mut nodes = Vec::new();
    build_bvh_node(&mut triangles, &mut nodes, 0, triangle_count);
    TriangleBvh {
        bounds: nodes[0].bounds,
        triangles: triangles
            .into_iter()
            .map(|triangle| triangle.triangle)
            .collect(),
        nodes,
    }
}

#[derive(Clone, Copy)]
struct BuildTriangle {
    triangle: TriangleBvhTriangle,
    bounds: WorldBounds,
    centroid: DVec3,
}

fn build_bvh_node(
    triangles: &mut [BuildTriangle],
    nodes: &mut Vec<TriangleBvhNode>,
    first: usize,
    count: usize,
) -> u32 {
    let node_index = u32::try_from(nodes.len()).expect("BVH node count fits u32");
    nodes.push(TriangleBvhNode::default());
    let bounds = triangles[first..first + count]
        .iter()
        .map(|triangle| triangle.bounds)
        .reduce(union_bounds)
        .expect("a BVH node contains triangles");
    let group_mask = triangles[first..first + count]
        .iter()
        .fold(TerrainTriangleGroupMask::default(), |mask, triangle| {
            mask.union(triangle.triangle.group_mask)
        });
    if count <= 8 {
        nodes[usize::try_from(node_index).expect("node index fits usize")] = TriangleBvhNode {
            bounds,
            first_triangle: u32::try_from(first).expect("triangle offset fits u32"),
            triangle_count: u32::try_from(count).expect("leaf count fits u32"),
            left_child: None,
            right_child: None,
            group_mask,
        };
        return node_index;
    }
    let extent = bounds.maximum.0 - bounds.minimum.0;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };
    let left_count = count / 2;
    triangles[first..first + count].select_nth_unstable_by(left_count, |left, right| {
        left.centroid[axis].total_cmp(&right.centroid[axis])
    });
    let left = build_bvh_node(triangles, nodes, first, left_count);
    let right = build_bvh_node(triangles, nodes, first + left_count, count - left_count);
    nodes[usize::try_from(node_index).expect("node index fits usize")] = TriangleBvhNode {
        bounds,
        first_triangle: 0,
        triangle_count: 0,
        left_child: Some(left),
        right_child: Some(right),
        group_mask,
    };
    node_index
}

fn triangle_bounds(origin: WorldPosition, vertices: &[[f32; 3]], indices: [u32; 3]) -> WorldBounds {
    let points = indices.map(|index| {
        origin.0
            + DVec3::from_array(
                vertices[usize::try_from(index).expect("vertex fits usize")].map(f64::from),
            )
    });
    WorldBounds {
        minimum: WorldPosition(points.into_iter().reduce(DVec3::min).expect("three points")),
        maximum: WorldPosition(points.into_iter().reduce(DVec3::max).expect("three points")),
    }
}

fn union_bounds(first: WorldBounds, second: WorldBounds) -> WorldBounds {
    WorldBounds {
        minimum: WorldPosition(first.minimum.0.min(second.minimum.0)),
        maximum: WorldPosition(first.maximum.0.max(second.maximum.0)),
    }
}

fn ray_triangle(
    origin: DVec3,
    direction: DVec3,
    first: DVec3,
    second: DVec3,
    third: DVec3,
) -> Option<(f64, Vec3)> {
    let epsilon = 1.0e-9;
    let edge_1 = second - first;
    let edge_2 = third - first;
    let perpendicular = direction.cross(edge_2);
    let determinant = edge_1.dot(perpendicular);
    if determinant.abs() < epsilon {
        return None;
    }
    let inverse = determinant.recip();
    let from_first = origin - first;
    let u = from_first.dot(perpendicular) * inverse;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let cross = from_first.cross(edge_1);
    let v = direction.dot(cross) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let distance = edge_2.dot(cross) * inverse;
    (distance >= 0.0).then_some((
        distance,
        Vec3::new((1.0 - u - v) as f32, u as f32, v as f32),
    ))
}

fn weighted_materials(
    chunk: &TerrainMeshChunk,
    indices: &[u32],
    barycentric: Vec3,
) -> [f32; TerrainMaterial::COUNT] {
    let mut result = [0.0; TerrainMaterial::COUNT];
    for (corner, weight) in barycentric.to_array().into_iter().enumerate() {
        let source =
            chunk.material_weights[usize::try_from(indices[corner]).expect("index fits usize")];
        for material in 0..TerrainMaterial::COUNT {
            result[material] += source[material] * weight;
        }
    }
    result
}

#[cfg(test)]
mod tests;
