//! Smooth terrain mesh, collision, and spatial-query contracts.

#![allow(clippy::cast_possible_truncation)] // GPU vertices and barycentrics are explicitly f32.

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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
    /// Surface-cover, soil, and rock weights per vertex.
    pub material_weights: Vec<[f32; 3]>,
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

type VertexKey = ([u32; 3], [u32; 3], [u32; 3]);

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
    /// Surface-cover, soil, and rock weights.
    pub material_weights: Vec<[f32; 3]>,
    /// Global owning bounds.
    pub bounds: WorldBounds,
    /// Generation invalidating old manifolds after replacement.
    pub generation: u64,
    /// Triangle acceleration data.
    pub triangle_bvh: TriangleBvh,
    /// Geometry groups currently enabled for collision.
    pub active_groups: TerrainTriangleGroupMask,
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
    /// Surface-cover, soil, and rock weights.
    pub material_weights: [f32; 3],
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

fn make_vertices_node_local(chunk: &mut TerrainMeshChunk) {
    let origin = chunk.origin.0.as_vec3();
    for vertex in &mut chunk.vertices {
        *vertex = (Vec3::from_array(*vertex) - origin).to_array();
    }
}

#[allow(clippy::too_many_lines)] // Fine and coarse halo preparation share indexing contracts.
fn sample_halo(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    node: TerrainNodeId,
    minimum: WorldCell,
    cubes: i32,
    stride: i32,
) -> Vec<TerrainSample> {
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

fn lattice_from_halo(halo: &[TerrainSample], cubes: i32) -> Vec<LatticePoint> {
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
                let sample = halo[halo_index(x, y, z)];
                let gradient = Vec3::new(
                    halo[halo_index(x + 1, y, z)].density - halo[halo_index(x - 1, y, z)].density,
                    halo[halo_index(x, y + 1, z)].density - halo[halo_index(x, y - 1, z)].density,
                    halo[halo_index(x, y, z + 1)].density - halo[halo_index(x, y, z - 1)].density,
                );
                lattice.push(LatticePoint {
                    sample,
                    normal: (-gradient).normalize_or(Vec3::Y),
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
    let mut coarse_samples = HashMap::<WorldCell, TerrainSample>::new();
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
        material: crossing_material(samples[solid].sample, samples[empty].sample),
    }
}

fn crossing_material(first: TerrainSample, second: TerrainSample) -> TerrainMaterial {
    if first.is_solid() {
        second.material
    } else {
        first.material
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
        let mut weights = [0.0; 3];
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
) -> TerrainSample {
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
    if minimum_promoted < direct.density {
        TerrainSample {
            density: minimum_promoted,
            material: direct.material,
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
) -> TerrainSample {
    // `WorldCell` values are cell-centred for exact removal accounting, while
    // the meshing lattice lies on their corners. Full signed distances retain
    // smooth procedural interpolation; only neighborhoods containing a real
    // edit are reconstructed as bounded cell occupancies.
    let mut samples = [TerrainSample {
        density: 0.0,
        material: TerrainMaterial::Rock,
    }; 8];
    let mut edited = false;
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
            edited |= sample != generated;
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
) -> TerrainSample {
    let mut samples = [TerrainSample {
        density: 0.0,
        material: TerrainMaterial::Rock,
    }; 8];
    let mut edited = false;
    let mut sample_index = 0;
    for z in -1..=0 {
        for y in -1..=0 {
            for x in -1..=0 {
                let index = cell_index(upper_x + x, upper_y + y, upper_z + z);
                samples[sample_index] = cells[index];
                edited |= edited_cells[index];
                sample_index += 1;
            }
        }
    }
    blend_lattice_samples(samples, edited)
}

fn blend_lattice_samples(samples: [TerrainSample; 8], reconstructing_edit: bool) -> TerrainSample {
    let half_cell = TERRAIN_CELL_METERS as f32 * 0.5;
    let mut density = 0.0;
    let mut material = TerrainMaterial::Rock;
    let mut nearest_surface = f32::INFINITY;
    for sample in samples {
        density += if reconstructing_edit {
            sample.density.clamp(-half_cell, half_cell)
        } else {
            sample.density
        };
        if sample.density.abs() < nearest_surface {
            nearest_surface = sample.density.abs();
            material = sample.material;
        }
    }
    TerrainSample {
        density: density / 8.0,
        material,
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
    let material = crossing_material(first.0.sample, second.0.sample);
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
                    material: crossing_material(first.0.sample, second.0.sample),
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
    samples: &mut HashMap<WorldCell, TerrainSample>,
    columns: &mut HashMap<(i32, i32), crate::generation::TerrainColumnSample>,
) -> LatticePoint {
    let sample = transition_coarse_sample(field, edits, cell, stride, samples, columns);
    let density =
        |offset: [i32; 3],
         samples: &mut HashMap<WorldCell, TerrainSample>,
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
            .density
        };
    let gradient = Vec3::new(
        density([1, 0, 0], samples, columns) - density([-1, 0, 0], samples, columns),
        density([0, 1, 0], samples, columns) - density([0, -1, 0], samples, columns),
        density([0, 0, 1], samples, columns) - density([0, 0, -1], samples, columns),
    );
    LatticePoint {
        sample,
        normal: (-gradient).normalize_or(Vec3::Y),
    }
}

fn transition_coarse_sample(
    field: &TerrainField,
    edits: &PreparedTerrainRegion<'_>,
    cell: WorldCell,
    stride: i32,
    samples: &mut HashMap<WorldCell, TerrainSample>,
    columns: &mut HashMap<(i32, i32), crate::generation::TerrainColumnSample>,
) -> TerrainSample {
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

fn weighted_materials(chunk: &TerrainMeshChunk, indices: &[u32], barycentric: Vec3) -> [f32; 3] {
    let mut result = [0.0; 3];
    for (corner, weight) in barycentric.to_array().into_iter().enumerate() {
        let source =
            chunk.material_weights[usize::try_from(indices[corner]).expect("index fits usize")];
        for material in 0..3 {
            result[material] += source[material] * weight;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_possible_truncation, clippy::float_cmp)]
    use bevy_math::{DVec3, Vec3};

    use super::{PreparedTerrainRegion, TerrainIndexGroups, TerrainMeshRequest, mesh_chunk};
    use crate::{
        BRICK_EDGE_CELLS, BrickCoord, TERRAIN_CELL_METERS, TerrainFace, TerrainField,
        TerrainNodeId, TerrainOctree, TerrainTransitionMask, WorldCell, WorldPosition, WorldSeed,
    };

    fn surface_request(brick_x: i32) -> TerrainMeshRequest {
        TerrainMeshRequest {
            node: TerrainNodeId::leaf(BrickCoord::new(brick_x, 2, -1)),
            generation: 4,
            transition_mask: TerrainTransitionMask::NONE,
        }
    }

    fn final_indices(chunk: &super::TerrainMeshChunk) -> Vec<u32> {
        chunk.index_groups.final_indices(chunk.transition_mask)
    }

    #[test]
    fn prepared_region_matches_octree_range_queries_across_signed_bricks() {
        let field = TerrainField::new(WorldSeed(91));
        let mut terrain = TerrainOctree::default();
        for coordinate in [
            BrickCoord::new(-2, 1, -1),
            BrickCoord::new(-1, 1, -1),
            BrickCoord::new(0, 1, 0),
            BrickCoord::new(20, 1, 20),
        ] {
            terrain.promote(&field, coordinate);
        }
        let snapshot = terrain.snapshot();
        let minimum = WorldCell::new(-64, 32, -32);
        let maximum = WorldCell::new(31, 63, 31);
        let prepared = PreparedTerrainRegion::between(&snapshot, minimum, maximum);
        assert_eq!(prepared.promoted_brick_count(), 3);
        for (query_minimum, query_maximum) in [
            (minimum, maximum),
            (WorldCell::new(-33, 40, -2), WorldCell::new(-30, 45, 2)),
            (WorldCell::new(-1, 32, -1), WorldCell::new(1, 34, 1)),
            (WorldCell::new(16, 40, 16), WorldCell::new(20, 44, 20)),
        ] {
            assert_eq!(
                prepared.minimum_promoted_density_between(query_minimum, query_maximum),
                snapshot.minimum_promoted_density_between(query_minimum, query_maximum),
            );
        }
    }

    #[test]
    fn index_counts_match_combined_vectors() {
        let groups = TerrainIndexGroups {
            regular: vec![0, 1, 2],
            transitions: std::array::from_fn(|face| vec![face as u32; face * 3]),
            caps: std::array::from_fn(|face| vec![face as u32; (face + 1) * 3]),
        };
        let transitions = TerrainTransitionMask::from_bits(0b10_0101);
        let ready = TerrainTransitionMask::from_bits(0b11_0010);
        assert_eq!(
            groups.final_index_count(transitions),
            groups.final_indices(transitions).len(),
        );
        assert_eq!(
            groups.sealed_index_count(transitions, ready),
            groups.sealed_indices(transitions, ready).len(),
        );
    }

    #[test]
    fn generated_vertices_stay_in_owning_bounds_and_normals_are_finite() {
        let field = TerrainField::new(WorldSeed(2));
        let terrain = TerrainOctree::default().snapshot();
        let chunk = mesh_chunk(&field, &terrain, surface_request(-1));
        let indices = final_indices(&chunk);
        assert!(!indices.is_empty());
        assert_eq!(chunk.vertex_cache.vertices.capacity(), 0);
        assert!(chunk.vertices.len() < indices.len());
        let unique = chunk
            .vertices
            .iter()
            .zip(&chunk.normals)
            .zip(&chunk.material_weights)
            .map(|((position, normal), weights)| {
                (
                    position.map(f32::to_bits),
                    normal.map(f32::to_bits),
                    weights.map(f32::to_bits),
                )
            })
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), chunk.vertices.len());
        for (vertex, normal) in chunk.vertices.iter().zip(&chunk.normals) {
            let global = chunk.origin.0 + DVec3::from_array(vertex.map(f64::from));
            assert!(chunk.bounds.contains(WorldPosition(global)));
            assert!(normal.iter().all(|component| component.is_finite()));
        }
    }

    #[test]
    fn promoted_meshing_keeps_the_analytic_surface_height_and_outward_winding() {
        let field = TerrainField::new(WorldSeed(2));
        let terrain = TerrainOctree::default().snapshot();
        let chunk = mesh_chunk(&field, &terrain, surface_request(-1));
        let indices = final_indices(&chunk);
        assert!(!indices.is_empty());
        let regular_vertices = chunk
            .index_groups
            .regular
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        for index in regular_vertices {
            let vertex = chunk.vertices[index as usize];
            assert!((vertex[1] + chunk.origin.0.y as f32 - 4.0).abs() < 1.0e-5);
        }
        for triangle in indices.chunks_exact(3) {
            let first = Vec3::from_array(chunk.vertices[triangle[0] as usize]);
            let second = Vec3::from_array(chunk.vertices[triangle[1] as usize]);
            let third = Vec3::from_array(chunk.vertices[triangle[2] as usize]);
            let geometric = (second - first).cross(third - first);
            let smooth = Vec3::from_array(chunk.normals[triangle[0] as usize])
                + Vec3::from_array(chunk.normals[triangle[1] as usize])
                + Vec3::from_array(chunk.normals[triangle[2] as usize]);
            assert!(
                geometric.dot(smooth) > 0.0,
                "terrain triangle faces inward: {}",
                geometric.dot(smooth)
            );
        }
    }

    #[test]
    fn adjacent_equal_lod_boundaries_are_byte_identical() {
        let field = TerrainField::new(WorldSeed(77));
        let edits = TerrainOctree::default().snapshot();
        let left = mesh_chunk(&field, &edits, surface_request(-1));
        let right = mesh_chunk(&field, &edits, surface_request(0));
        let seam_x = 0.0_f32;
        let mut left_seam = left
            .vertices
            .iter()
            .map(|vertex| {
                [
                    vertex[0] + left.origin.0.x as f32,
                    vertex[1] + left.origin.0.y as f32,
                    vertex[2] + left.origin.0.z as f32,
                ]
            })
            .filter(|vertex| vertex[0] == seam_x)
            .collect::<Vec<_>>();
        let mut right_seam = right
            .vertices
            .iter()
            .map(|vertex| {
                [
                    vertex[0] + right.origin.0.x as f32,
                    vertex[1] + right.origin.0.y as f32,
                    vertex[2] + right.origin.0.z as f32,
                ]
            })
            .filter(|vertex| vertex[0] == seam_x)
            .collect::<Vec<_>>();
        left_seam.sort_by_key(|vertex| (vertex[1].to_bits(), vertex[2].to_bits()));
        right_seam.sort_by_key(|vertex| (vertex[1].to_bits(), vertex[2].to_bits()));
        left_seam.dedup();
        right_seam.dedup();
        assert_eq!(left_seam, right_seam);
    }

    #[test]
    fn untouched_surface_remains_smooth_and_covered_at_every_lod() {
        let field = TerrainField::new(WorldSeed(77));
        let edits = TerrainOctree::default().snapshot();
        let probe = WorldPosition(DVec3::new(500.0, field.surface_height(500.0, 500.0), 500.0));
        let brick = probe.cell().expect("probe is inside cell space").brick();
        for level in 0..=5 {
            let node = TerrainNodeId::containing(brick, level).expect("streamed LOD exists");
            let chunk = mesh_chunk(
                &field,
                &edits,
                TerrainMeshRequest {
                    node,
                    generation: 1,
                    transition_mask: TerrainTransitionMask::NONE,
                },
            );
            let regular_vertices = chunk
                .index_groups
                .regular
                .iter()
                .map(|&index| index as usize)
                .collect::<std::collections::BTreeSet<_>>();
            assert!(!regular_vertices.is_empty(), "LOD {level} has no surface");
            for index in regular_vertices {
                let local = DVec3::from_array(chunk.vertices[index].map(f64::from));
                let global = chunk.origin.0 + local;
                let expected_height = field.surface_height(global.x, global.z);
                assert!(
                    (global.y - expected_height).abs() < 0.1,
                    "LOD {level} surface error at {global:?}: expected {expected_height}"
                );
                assert_eq!(
                    chunk.material_weights[index],
                    [1.0, 0.0, 0.0],
                    "LOD {level} exposed a subsurface material"
                );
            }
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)] // The fixture assembles both sides of one complete LOD face.
    fn transition_surface_has_no_open_interior_edges() {
        type Point = [i64; 3];
        type Edge = (Point, Point);

        fn point(chunk: &super::TerrainMeshChunk, index: u32) -> Point {
            let local = DVec3::from_array(chunk.vertices[index as usize].map(f64::from));
            (chunk.origin.0 + local)
                .to_array()
                .map(|coordinate| (coordinate * 1_000.0).round() as i64)
        }

        fn add_face_edges(
            counts: &mut std::collections::BTreeMap<Edge, usize>,
            chunk: &super::TerrainMeshChunk,
            indices: &[u32],
            minimum_x: i64,
            seam_x: i64,
        ) {
            for triangle in indices.chunks_exact(3) {
                for edge in [
                    [triangle[0], triangle[1]],
                    [triangle[1], triangle[2]],
                    [triangle[2], triangle[0]],
                ] {
                    let first = point(chunk, edge[0]);
                    let second = point(chunk, edge[1]);
                    if first == second {
                        continue;
                    }
                    if first[0] < minimum_x
                        || second[0] < minimum_x
                        || first[0] > seam_x + 1
                        || second[0] > seam_x + 1
                    {
                        continue;
                    }
                    let edge = if first <= second {
                        (first, second)
                    } else {
                        (second, first)
                    };
                    *counts.entry(edge).or_default() += 1;
                }
            }
        }

        let field = TerrainField::new(WorldSeed(77));
        let edits = TerrainOctree::default().snapshot();
        let seam_brick_x = 320;
        let seam_brick_z = 320;
        let surface_brick =
            WorldPosition(DVec3::new(512.0, field.surface_height(512.0, 512.0), 512.0))
                .cell()
                .expect("probe is inside cell space")
                .brick();
        let coarse_node = TerrainNodeId::containing(
            BrickCoord::new(seam_brick_x, surface_brick.y, seam_brick_z),
            5,
        )
        .expect("streamed LOD exists");
        let coarse = mesh_chunk(
            &field,
            &edits,
            TerrainMeshRequest {
                node: coarse_node,
                generation: 1,
                transition_mask: TerrainTransitionMask::NONE,
            },
        );
        let transition_mask = TerrainTransitionMask::from_bits(1 << TerrainFace::PositiveX as u8);
        let fine = [0, 16]
            .into_iter()
            .flat_map(|y| [0, 16].map(move |z| (y, z)))
            .map(|(y, z)| {
                mesh_chunk(
                    &field,
                    &edits,
                    TerrainMeshRequest {
                        node: TerrainNodeId {
                            coordinates: BrickCoord::new(
                                coarse_node.coordinates.x - 16,
                                coarse_node.coordinates.y + y,
                                coarse_node.coordinates.z + z,
                            ),
                            level: 4,
                        },
                        generation: 1,
                        transition_mask,
                    },
                )
            })
            .collect::<Vec<_>>();

        let mut counts = std::collections::BTreeMap::new();
        let seam_x = i64::from(seam_brick_x) * 1_600;
        let transition_band_minimum = seam_x - 201;
        add_face_edges(
            &mut counts,
            &coarse,
            &coarse.index_groups.regular,
            transition_band_minimum,
            seam_x,
        );
        for chunk in &fine {
            add_face_edges(
                &mut counts,
                chunk,
                &chunk.index_groups.regular,
                transition_band_minimum,
                seam_x,
            );
            add_face_edges(
                &mut counts,
                chunk,
                &chunk.index_groups.transitions[TerrainFace::PositiveX.index()],
                transition_band_minimum,
                seam_x,
            );
        }
        let minimum_z = i64::from(coarse_node.coordinates.z) * 1_600;
        let maximum_z = minimum_z + 51_200;
        let unmatched = counts
            .iter()
            .filter(|((first, second), count)| {
                **count % 2 != 0
                    && first[2] != minimum_z
                    && second[2] != minimum_z
                    && first[2] != maximum_z
                    && second[2] != maximum_z
            })
            .collect::<Vec<_>>();
        assert!(!counts.is_empty());
        assert!(unmatched.is_empty(), "unmatched seam edges: {unmatched:?}");
    }

    #[test]
    fn transition_surface_occupies_an_inset_band() {
        let field = TerrainField::new(WorldSeed(77));
        let edits = TerrainOctree::default().snapshot();
        let seam_brick_x = 320;
        let seam_brick_z = 320;
        let surface_brick =
            WorldPosition(DVec3::new(512.0, field.surface_height(512.0, 512.0), 512.0))
                .cell()
                .expect("probe is inside cell space")
                .brick();
        let coarse_node = TerrainNodeId::containing(
            BrickCoord::new(seam_brick_x, surface_brick.y, seam_brick_z),
            5,
        )
        .expect("streamed LOD exists");
        let transition_mask = TerrainTransitionMask::from_bits(1 << TerrainFace::PositiveX as u8);
        let mut maximum_inset = 0.0_f64;
        let mut has_coarse_side_vertex = false;

        for (y, z) in [0, 16]
            .into_iter()
            .flat_map(|y| [0, 16].map(move |z| (y, z)))
        {
            let chunk = mesh_chunk(
                &field,
                &edits,
                TerrainMeshRequest {
                    node: TerrainNodeId {
                        coordinates: BrickCoord::new(
                            coarse_node.coordinates.x - 16,
                            coarse_node.coordinates.y + y,
                            coarse_node.coordinates.z + z,
                        ),
                        level: 4,
                    },
                    generation: 1,
                    transition_mask,
                },
            );
            let face_x =
                chunk.origin.0.x + f64::from(BRICK_EDGE_CELLS) * chunk.sample_spacing_metres;
            for &index in &chunk.index_groups.transitions[TerrainFace::PositiveX.index()] {
                let position = chunk.origin.0
                    + DVec3::from_array(chunk.vertices[index as usize].map(f64::from));
                let inset = face_x - position.x;
                assert!(
                    inset >= -1.0e-5,
                    "transition vertex escaped the fine chunk: {inset}"
                );
                maximum_inset = maximum_inset.max(inset);
                has_coarse_side_vertex |= inset.abs() <= 1.0e-5;
            }
        }

        assert!(
            has_coarse_side_vertex,
            "transition lost its coarse-side edge"
        );
        assert!(
            maximum_inset > 0.1 * f64::from(1_i32 << 4) * TERRAIN_CELL_METERS,
            "transition collapsed onto the chunk face: {maximum_inset}"
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)] // The fixture compares both complete sides of one edited LOD seam.
    fn edited_transition_coarse_edge_matches_coarse_regular_surface() {
        type Point = [i64; 3];
        type Edge = (Point, Point);

        fn point(chunk: &super::TerrainMeshChunk, index: u32) -> Point {
            let local = DVec3::from_array(chunk.vertices[index as usize].map(f64::from));
            (chunk.origin.0 + local)
                .to_array()
                .map(|coordinate| (coordinate * 1_000.0).round() as i64)
        }

        fn boundary_edges<'a>(
            groups: impl IntoIterator<Item = (&'a super::TerrainMeshChunk, &'a [u32])>,
            seam_x: i64,
        ) -> std::collections::BTreeSet<Edge> {
            let mut counts = std::collections::BTreeMap::<Edge, usize>::new();
            for (chunk, indices) in groups {
                for triangle in indices.chunks_exact(3) {
                    for pair in [
                        [triangle[0], triangle[1]],
                        [triangle[1], triangle[2]],
                        [triangle[2], triangle[0]],
                    ] {
                        let first = point(chunk, pair[0]);
                        let second = point(chunk, pair[1]);
                        if first[0] != seam_x || second[0] != seam_x || first == second {
                            continue;
                        }
                        let edge = if first <= second {
                            (first, second)
                        } else {
                            (second, first)
                        };
                        *counts.entry(edge).or_default() += 1;
                    }
                }
            }
            counts
                .into_iter()
                .filter_map(|(edge, count)| (count % 2 != 0).then_some(edge))
                .collect()
        }

        let field = TerrainField::new(WorldSeed(2_255_932_754_758_176_049));
        let surface = field.surface_height(0.3, 0.8);
        let mut edits = TerrainOctree::default();
        edits
            .excavate_sphere(
                &field,
                WorldPosition(DVec3::new(0.3, surface - 0.2, 0.8)),
                0.65,
            )
            .expect("edit is inside the world");
        let snapshot = edits.snapshot();
        let surface_brick = WorldPosition(DVec3::new(0.0, surface, 0.8))
            .cell()
            .expect("surface is in cell space")
            .brick();
        let coarse_node =
            TerrainNodeId::containing(BrickCoord::new(0, surface_brick.y, surface_brick.z), 1)
                .expect("coarse node");
        let transition_mask = TerrainTransitionMask::from_bits(1 << TerrainFace::PositiveX as u8);
        let coarse = mesh_chunk(
            &field,
            &snapshot,
            TerrainMeshRequest {
                node: coarse_node,
                generation: 1,
                transition_mask: TerrainTransitionMask::NONE,
            },
        );
        let fine = [0, 1]
            .into_iter()
            .flat_map(|y| [0, 1].map(move |z| (y, z)))
            .map(|(y, z)| {
                mesh_chunk(
                    &field,
                    &snapshot,
                    TerrainMeshRequest {
                        node: TerrainNodeId::leaf(BrickCoord::new(
                            coarse_node.coordinates.x - 1,
                            coarse_node.coordinates.y + y,
                            coarse_node.coordinates.z + z,
                        )),
                        generation: 1,
                        transition_mask,
                    },
                )
            })
            .collect::<Vec<_>>();
        let seam_x = (coarse.origin.0.x * 1_000.0).round() as i64;
        let mut coarse_edges =
            boundary_edges([(&coarse, coarse.index_groups.regular.as_slice())], seam_x);
        let mut transition_edges = boundary_edges(
            fine.iter().map(|chunk| {
                (
                    chunk,
                    chunk.index_groups.transitions[TerrainFace::PositiveX.index()].as_slice(),
                )
            }),
            seam_x,
        );
        let minimum = coarse
            .origin
            .0
            .to_array()
            .map(|value| (value * 1_000.0).round() as i64);
        let maximum = coarse
            .bounds
            .maximum
            .0
            .to_array()
            .map(|value| (value * 1_000.0).round() as i64);
        let on_perimeter = |edge: &Edge| {
            [1, 2].into_iter().any(|axis| {
                (edge.0[axis] == minimum[axis] && edge.1[axis] == minimum[axis])
                    || (edge.0[axis] == maximum[axis] && edge.1[axis] == maximum[axis])
            })
        };
        coarse_edges.retain(|edge| !on_perimeter(edge));
        transition_edges.retain(|edge| !on_perimeter(edge));

        assert!(
            !coarse_edges.is_empty(),
            "fixture missed the edited surface"
        );
        assert_eq!(transition_edges, coarse_edges);
    }

    #[test]
    fn official_transition_vertices_only_use_crossing_edges() {
        let row_major_point = |point: usize| match point {
            0..=8 => point,
            9 => 0,
            10 => 2,
            11 => 6,
            12 => 8,
            _ => unreachable!("transition point is 0 through C"),
        };
        for row_major_case in 0_u16..512 {
            let table_case = [0, 1, 2, 5, 8, 7, 6, 3, 4]
                .into_iter()
                .enumerate()
                .fold(0_u16, |case, (bit, point)| {
                    case | (((row_major_case >> point) & 1) << bit)
                });
            let class = super::TRANSITION_CELL_CLASS[usize::from(table_case)] & 0x7f;
            let vertex_count =
                usize::from(super::TRANSITION_CELL_DATA[usize::from(class)].geometry_counts >> 4);
            for &data in &super::TRANSITION_VERTEX_DATA[usize::from(table_case)][..vertex_count] {
                let edge = data & 0xff;
                let first = row_major_point(usize::from((edge >> 4) as u8));
                let second = row_major_point(usize::from((edge & 0x0f) as u8));
                assert_ne!(
                    (row_major_case >> first) & 1,
                    (row_major_case >> second) & 1,
                    "case {row_major_case:#05x} uses non-crossing edge {edge:#04x}"
                );
            }
        }
    }

    #[test]
    fn terrain_mesh_raycast_hits_the_surface() {
        let field = TerrainField::new(WorldSeed(9));
        let terrain = TerrainOctree::default().snapshot();
        let chunk = mesh_chunk(&field, &terrain, surface_request(-1));
        let hit = chunk
            .raycast(
                WorldPosition(DVec3::new(0.0, 10.0, 0.0)),
                DVec3::NEG_Y,
                20.0,
            )
            .expect("downward ray meets terrain");
        assert!(hit.normal.is_finite());
        assert_eq!(hit.chunk_generation, 4);
    }

    #[test]
    fn excavated_surface_triangles_keep_outward_winding() {
        let field = TerrainField::new(WorldSeed(33));
        let surface = field.surface_height(0.0, 0.0);
        let centre = WorldPosition(DVec3::new(0.0, surface - 0.3, 0.0));
        let mut edits = TerrainOctree::default();
        let outcome = edits
            .excavate_sphere(&field, centre, 0.75)
            .expect("excavation is valid");
        let snapshot = edits.snapshot();
        for coordinate in outcome.changed_brick_coordinates() {
            let chunk = mesh_chunk(
                &field,
                &snapshot,
                TerrainMeshRequest {
                    node: TerrainNodeId::leaf(*coordinate),
                    generation: 1,
                    transition_mask: TerrainTransitionMask::NONE,
                },
            );
            let indices = final_indices(&chunk);
            for triangle in indices.chunks_exact(3) {
                let first = Vec3::from_array(chunk.vertices[triangle[0] as usize]);
                let second = Vec3::from_array(chunk.vertices[triangle[1] as usize]);
                let third = Vec3::from_array(chunk.vertices[triangle[2] as usize]);
                let geometric = (second - first).cross(third - first);
                let smooth = triangle
                    .iter()
                    .map(|&index| Vec3::from_array(chunk.normals[index as usize]))
                    .sum::<Vec3>();
                let alignment = geometric.dot(smooth);
                assert!(
                    alignment >= -1.0e-12,
                    "excavated triangle faces inward by {alignment}"
                );
            }
        }
    }

    #[test]
    fn official_regular_and_transition_tables_cover_every_case() {
        for case in 0..256 {
            let class = super::REGULAR_CELL_CLASS[case];
            let cell = super::REGULAR_CELL_DATA[usize::from(class)];
            let vertices = usize::from(cell.geometry_counts >> 4);
            let triangles = usize::from(cell.geometry_counts & 0x0f);
            assert!(
                cell.vertex_index[..triangles * 3]
                    .iter()
                    .all(|&index| usize::from(index) < vertices)
            );
        }
        for case in 0..512 {
            let class = super::TRANSITION_CELL_CLASS[case] & 0x7f;
            let cell = super::TRANSITION_CELL_DATA[usize::from(class)];
            let vertices = usize::from(cell.geometry_counts >> 4);
            let triangles = usize::from(cell.geometry_counts & 0x0f);
            assert!(
                cell.vertex_index[..triangles * 3]
                    .iter()
                    .all(|&index| usize::from(index) < vertices)
            );
        }
    }

    #[test]
    fn cavity_generates_transitions_and_caps_on_all_six_faces() {
        let field = TerrainField::new(WorldSeed(90));
        let mut terrain = TerrainOctree::default();
        terrain
            .excavate_sphere(&field, WorldPosition(DVec3::new(0.8, 2.4, 0.8)), 1.0)
            .unwrap();
        let snapshot = terrain.snapshot();
        for face in crate::TerrainFace::ALL {
            let chunk = mesh_chunk(
                &field,
                &snapshot,
                TerrainMeshRequest {
                    node: TerrainNodeId::leaf(BrickCoord::new(0, 1, 0)),
                    generation: 1,
                    transition_mask: TerrainTransitionMask::from_bits(1 << face as u8),
                },
            );
            assert!(
                !chunk.index_groups.transitions[face.index()].is_empty(),
                "missing transition triangles on {face:?}"
            );
            assert!(
                !chunk.index_groups.caps[face.index()].is_empty(),
                "missing cap triangles on {face:?}"
            );
            assert!(
                chunk
                    .normals
                    .iter()
                    .flatten()
                    .all(|component| component.is_finite())
            );
            let ready_except_transition =
                TerrainTransitionMask::from_bits(0x3f & !(1 << face as u8));
            let sealed = chunk
                .index_groups
                .sealed_indices(chunk.transition_mask, ready_except_transition);
            assert_eq!(
                sealed.len(),
                chunk.index_groups.regular.len()
                    + chunk.index_groups.transitions[face.index()].len()
                    + chunk.index_groups.caps[face.index()].len(),
                "pending transition must bridge the inset mesh before its cap on {face:?}"
            );
            let collision = chunk.sealed_collision_chunk(TerrainTransitionMask::NONE);
            assert_eq!(collision.indices, chunk.index_groups.regular);
            assert_eq!(
                collision.active_groups,
                super::TerrainTriangleGroupMask::REGULAR
            );
        }
    }

    #[test]
    fn generated_triangle_bvh_has_real_branches_and_leaf_ranges() {
        let field = TerrainField::new(WorldSeed(91));
        let terrain = TerrainOctree::default().snapshot();
        let chunk = mesh_chunk(&field, &terrain, surface_request(-1));
        assert!(chunk.triangle_bvh.nodes.len() > 1);
        let all_group_triangles = chunk.index_groups.regular.len()
            + chunk
                .index_groups
                .transitions
                .iter()
                .map(Vec::len)
                .sum::<usize>()
            + chunk.index_groups.caps.iter().map(Vec::len).sum::<usize>();
        assert_eq!(chunk.triangle_bvh.triangles.len(), all_group_triangles / 3);
        assert_eq!(
            chunk
                .sealed_collision_chunk(TerrainTransitionMask::NONE)
                .triangle_bvh,
            chunk.triangle_bvh
        );
        assert_eq!(chunk.triangle_bvh.nodes[0].triangle_count, 0);
        assert!(
            chunk
                .triangle_bvh
                .nodes
                .iter()
                .filter(|node| node.triangle_count != 0)
                .all(|node| node.left_child.is_none() && node.right_child.is_none())
        );
    }
}
