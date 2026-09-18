//! Terrain scenes: exact density sampling and ray casts over the active terrain cut.

use super::capsule::density_normal;
use crate::{
    TERRAIN_CELL_METERS, TerrainField, TerrainMaterial, TerrainMeshChunk, TerrainNodeId,
    TerrainOctree, TerrainRayHit, TerrainTransitionMask, WorldPosition,
};
use bevy_math::DVec3;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

/// Read-only density source used by walking, raycasts, and support checks.
pub trait TerrainDensity {
    /// Density at a global point. Positive values are occupied.
    fn density(&self, position: WorldPosition) -> f32;
    /// Material at a global point.
    fn material(&self, position: WorldPosition) -> TerrainMaterial;
}

/// Procedural field with its sparse edits applied.
#[derive(Clone, Copy, Debug)]
pub struct TerrainScene<'a> {
    /// Untouched deterministic terrain.
    pub field: &'a TerrainField,
    /// Promoted edited bricks.
    pub edits: &'a TerrainOctree,
}

impl TerrainDensity for TerrainScene<'_> {
    fn density(&self, position: WorldPosition) -> f32 {
        self.edits.sample_position(self.field, position).density
    }

    fn material(&self, position: WorldPosition) -> TerrainMaterial {
        self.edits.sample_position(self.field, position).material
    }
}

/// Read-only view of the exact active node meshes used by rendering and physics.
#[derive(Clone, Copy, Debug)]
pub struct ActiveTerrainScene<'a> {
    /// Active generation-selected node bundles.
    pub chunks: &'a BTreeMap<TerrainNodeId, TerrainMeshChunk>,
    /// Faces whose adjacent active generation is ready.
    pub ready_faces: &'a BTreeMap<TerrainNodeId, TerrainTransitionMask>,
    /// Incrementally maintained active-node octree index.
    pub spatial_index: &'a TerrainSpatialIndex,
}

impl ActiveTerrainScene<'_> {
    /// Nearest active terrain triangle along a ray.
    pub fn raycast(
        self,
        origin: WorldPosition,
        direction: DVec3,
        maximum_distance: f64,
    ) -> Option<TerrainRayHit> {
        self.spatial_index
            .ray_candidates(origin, direction, maximum_distance)
            .into_iter()
            .filter_map(|id| {
                let chunk = self.chunks.get(&id)?;
                chunk.raycast_sealed(
                    self.ready_faces.get(&id).copied().unwrap_or_default(),
                    origin,
                    direction,
                    maximum_distance,
                )
            })
            .min_by(|first, second| first.distance.total_cmp(&second.distance))
    }

    pub(super) fn nearest(self, position: WorldPosition) -> Option<(f32, TerrainMaterial)> {
        self.spatial_index.nearest(position, |id| {
            let chunk = self.chunks.get(&id)?;
            let sample = chunk.nearest_sealed(
                self.ready_faces.get(&id).copied().unwrap_or_default(),
                position,
            )?;
            Some(((sample.0, sample.1), sample.2))
        })
    }
}

/// Incrementally maintained sparse octree index of active terrain nodes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TerrainSpatialIndex {
    pub(super) active: BTreeSet<TerrainNodeId>,
    pub(super) descendant_counts: BTreeMap<TerrainNodeId, u32>,
    pub(super) active_bounds: BTreeMap<TerrainNodeId, crate::WorldBounds>,
    pub(super) aggregate_bounds: BTreeMap<TerrainNodeId, crate::WorldBounds>,
}

impl TerrainSpatialIndex {
    /// Inserts an active node and updates its ancestor path.
    pub fn insert(&mut self, id: TerrainNodeId) {
        self.insert_bounds(id, id.world_bounds());
    }

    /// Inserts or refits one active node using its actual mesh bounds. This
    /// handles finite vertices rounded outside a nominal node boundary without
    /// clipping the mesh or losing contacts at a seam. Returns false for invalid
    /// bounds, leaving the index unchanged.
    pub fn insert_bounds(&mut self, id: TerrainNodeId, bounds: crate::WorldBounds) -> bool {
        if !bounds.minimum.0.is_finite()
            || !bounds.maximum.0.is_finite()
            || !bounds.minimum.0.cmple(bounds.maximum.0).all()
        {
            return false;
        }
        if self.active.insert(id) {
            let mut current = Some(id);
            while let Some(node) = current {
                *self.descendant_counts.entry(node).or_default() += 1;
                current = node.parent();
            }
        }
        self.active_bounds.insert(id, bounds);
        self.refit_bounds(id);
        true
    }

    pub(super) fn refit_bounds(&mut self, id: TerrainNodeId) {
        let mut current = Some(id);
        while let Some(node) = current {
            let mut bounds = self.active_bounds.get(&node).copied();
            if let Some(children) = node.children() {
                for child in children {
                    if let Some(&child) = self.aggregate_bounds.get(&child) {
                        bounds = Some(bounds.map_or(child, |old| crate::WorldBounds {
                            minimum: WorldPosition(old.minimum.0.min(child.minimum.0)),
                            maximum: WorldPosition(old.maximum.0.max(child.maximum.0)),
                        }));
                    }
                }
            }
            if let Some(bounds) = bounds {
                self.aggregate_bounds.insert(node, bounds);
            } else {
                self.aggregate_bounds.remove(&node);
            }
            current = node.parent();
        }
    }

    /// Removes an active node and prunes empty ancestor paths.
    pub fn remove(&mut self, id: TerrainNodeId) {
        if !self.active.remove(&id) {
            return;
        }
        self.active_bounds.remove(&id);
        let mut current = Some(id);
        while let Some(node) = current {
            let remove = if let Some(count) = self.descendant_counts.get_mut(&node) {
                *count = count.saturating_sub(1);
                *count == 0
            } else {
                false
            };
            if remove {
                self.descendant_counts.remove(&node);
            }
            current = node.parent();
        }
        self.refit_bounds(id);
    }

    /// True when the exact active node is indexed.
    pub fn contains(&self, id: TerrainNodeId) -> bool {
        self.active.contains(&id)
    }

    /// Active nodes whose owning boxes overlap inclusive global bounds, in stable
    /// node order. This reuses the incrementally maintained ancestor hierarchy.
    pub fn bounds_candidates(&self, bounds: crate::WorldBounds) -> Vec<TerrainNodeId> {
        let mut result = Vec::new();
        let mut stack = vec![TerrainNodeId::ROOT];
        while let Some(node) = stack.pop() {
            if !self.descendant_counts.contains_key(&node) {
                continue;
            }
            if !bounds.intersects(self.aggregate_bounds[&node]) {
                continue;
            }
            if self
                .active_bounds
                .get(&node)
                .is_some_and(|own| bounds.intersects(*own))
            {
                result.push(node);
            }
            if let Some(children) = node.children() {
                stack.extend(children);
            }
        }
        result.sort_unstable();
        result
    }

    pub(super) fn ray_candidates(
        &self,
        origin: WorldPosition,
        direction: DVec3,
        maximum_distance: f64,
    ) -> Vec<TerrainNodeId> {
        let Some(direction) = direction.try_normalize() else {
            return Vec::new();
        };
        let mut candidates = Vec::new();
        let mut stack = vec![TerrainNodeId::ROOT];
        while let Some(node) = stack.pop() {
            if !self.descendant_counts.contains_key(&node)
                || !ray_intersects_bounds(
                    origin.0,
                    direction,
                    self.aggregate_bounds[&node],
                    maximum_distance,
                )
            {
                continue;
            }
            if self.active_bounds.get(&node).is_some_and(|&bounds| {
                ray_intersects_bounds(origin.0, direction, bounds, maximum_distance)
            }) {
                candidates.push(node);
            }
            if let Some(children) = node.children() {
                stack.extend(children);
            }
        }
        candidates
    }

    pub(super) fn nearest<T>(
        &self,
        position: WorldPosition,
        mut visit: impl FnMut(TerrainNodeId) -> Option<(T, f64)>,
    ) -> Option<T> {
        let mut heap = BinaryHeap::new();
        let mut best_distance = f64::INFINITY;
        let mut nearest = None;
        if self.descendant_counts.contains_key(&TerrainNodeId::ROOT) {
            heap.push(DistanceNode::new(
                position,
                TerrainNodeId::ROOT,
                self.aggregate_bounds[&TerrainNodeId::ROOT],
            ));
        }
        while let Some(entry) = heap.pop() {
            if entry.distance_squared >= best_distance * best_distance {
                break;
            }
            if self.active.contains(&entry.node)
                && let Some((candidate, distance)) = visit(entry.node)
                && distance < best_distance
            {
                best_distance = distance;
                nearest = Some(candidate);
            }
            if let Some(children) = entry.node.children() {
                heap.extend(
                    children
                        .into_iter()
                        .filter(|child| self.descendant_counts.contains_key(child))
                        .map(|child| {
                            DistanceNode::new(position, child, self.aggregate_bounds[&child])
                        }),
                );
            }
        }
        nearest
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DistanceNode {
    pub(super) distance_squared: f64,
    pub(super) node: TerrainNodeId,
}

impl DistanceNode {
    pub(super) fn new(
        position: WorldPosition,
        node: TerrainNodeId,
        bounds: crate::WorldBounds,
    ) -> Self {
        let (minimum, maximum) = (bounds.minimum.0, bounds.maximum.0);
        let distance_squared = (0..3)
            .map(|axis| {
                if position.0[axis] < minimum[axis] {
                    minimum[axis] - position.0[axis]
                } else if position.0[axis] > maximum[axis] {
                    position.0[axis] - maximum[axis]
                } else {
                    0.0
                }
            })
            .map(|distance| distance * distance)
            .sum();
        Self {
            distance_squared,
            node,
        }
    }
}

impl PartialEq for DistanceNode {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.distance_squared == other.distance_squared
    }
}

impl Eq for DistanceNode {}

impl PartialOrd for DistanceNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistanceNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance_squared
            .total_cmp(&self.distance_squared)
            .then_with(|| other.node.cmp(&self.node))
    }
}

pub(super) fn ray_intersects_bounds(
    origin: DVec3,
    direction: DVec3,
    bounds: crate::WorldBounds,
    maximum_distance: f64,
) -> bool {
    let (minimum, maximum) = (bounds.minimum.0, bounds.maximum.0);
    let mut near = 0.0_f64;
    let mut far = maximum_distance;
    for axis in 0..3 {
        if direction[axis].abs() <= f64::EPSILON {
            if origin[axis] < minimum[axis] || origin[axis] > maximum[axis] {
                return false;
            }
            continue;
        }
        let inverse = direction[axis].recip();
        let mut first = (minimum[axis] - origin[axis]) * inverse;
        let mut second = (maximum[axis] - origin[axis]) * inverse;
        if first > second {
            core::mem::swap(&mut first, &mut second);
        }
        near = near.max(first);
        far = far.min(second);
        if near > far {
            return false;
        }
    }
    far >= 0.0
}

impl TerrainDensity for ActiveTerrainScene<'_> {
    fn density(&self, position: WorldPosition) -> f32 {
        (*self).nearest(position).map_or(-1.0, |sample| sample.0)
    }

    fn material(&self, position: WorldPosition) -> TerrainMaterial {
        (*self)
            .nearest(position)
            .map_or(TerrainMaterial::Rock, |sample| sample.1)
    }
}

/// Raycasts a density field without depending on a rendered chunk being ready.
pub fn raycast_density(
    terrain: &impl TerrainDensity,
    origin: WorldPosition,
    direction: DVec3,
    maximum_distance: f64,
) -> Option<TerrainRayHit> {
    let direction = direction.try_normalize()?;
    let step = TERRAIN_CELL_METERS * 0.5;
    let mut previous_distance = 0.0;
    let mut previous_density = terrain.density(origin);
    let mut distance = step;
    while distance <= maximum_distance {
        let position = WorldPosition(origin.0 + direction * distance);
        let density = terrain.density(position);
        if (density > 0.0) != (previous_density > 0.0) {
            let mut empty = previous_distance;
            let mut solid = distance;
            if previous_density > 0.0 {
                core::mem::swap(&mut empty, &mut solid);
            }
            for _ in 0..10 {
                let middle = (empty + solid) * 0.5;
                if terrain.density(WorldPosition(origin.0 + direction * middle)) > 0.0 {
                    solid = middle;
                } else {
                    empty = middle;
                }
            }
            let hit_distance = (empty + solid) * 0.5;
            let hit_position = WorldPosition(origin.0 + direction * hit_distance);
            let normal = density_normal(terrain, hit_position);
            let mut material_weights = [0.0; TerrainMaterial::COUNT];
            material_weights[terrain
                .material(WorldPosition(hit_position.0 - DVec3::from(normal) * step))
                .code() as usize] = 1.0;
            return Some(TerrainRayHit {
                position: hit_position,
                normal,
                distance: hit_distance,
                material_weights,
                chunk_generation: 0,
                triangle: 0,
            });
        }
        previous_distance = distance;
        previous_density = density;
        distance += step;
    }
    None
}
