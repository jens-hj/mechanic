//! Smooth terrain mesh, collision, and spatial-query contracts.

#![expect(
    clippy::cast_possible_truncation,
    reason = "GPU vertices and barycentrics are explicitly f32"
)]

mod bvh;
mod collision;
mod groups;
mod lattice;
mod polygonise;
mod transition;

pub use bvh::{TriangleBvh, TriangleBvhNode, TriangleBvhTriangle};
use bvh::{
    build_triangle_bvh, closest_point_on_triangle, point_bounds_distance_squared,
    ray_intersects_bounds, ray_triangle,
};
pub use collision::{TerrainCollisionChunk, TerrainRayHit};
pub use groups::{TerrainIndexGroups, TerrainTriangleGroupMask};
pub use lattice::PreparedTerrainRegion;
use lattice::{lattice_from_halo, sample_halo, synchronize_edited_boundary_lattice};
use polygonise::{CUBE_CORNERS, polygonise_cube, weighted_materials};
use transition::{generate_face_cap, generate_transition_face};

use std::{collections::HashMap, time::Instant};

use bevy_math::{DVec3, Vec3};
use serde::{Deserialize, Serialize};

use crate::{
    BRICK_EDGE_CELLS, TERRAIN_CELL_METERS, TerrainFace, TerrainField, TerrainMaterial,
    TerrainNodeId, TerrainOctreeSnapshot, TerrainTransitionMask, WorldCell, WorldPosition,
};

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

#[cfg(test)]
mod tests;
