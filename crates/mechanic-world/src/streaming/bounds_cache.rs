//! A bounded cache of procedural density bounds per node.

use crate::{BRICK_EDGE_METERS, TerrainField, TerrainNodeId};
use bevy_math::DVec3;
use std::collections::HashMap;

pub(super) const PROCEDURAL_BOUNDS_CACHE_BYTES: usize = 32 * 1024 * 1024;

pub(super) const PROCEDURAL_BOUNDS_ENTRY_BYTES: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct ProceduralBoundsKey {
    pub(super) level: u8,
    pub(super) x: i32,
    pub(super) z: i32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ProceduralBounds {
    pub(super) minimum_surface: f64,
    pub(super) maximum_surface: f64,
    pub(super) margin: f64,
}

/// Per-world cache of conservative procedural surface bounds used by octree
/// selection. Entries are independent of vertical coordinate and therefore
/// serve every y node in the same horizontal octree column.
#[derive(Clone, Debug, Default)]
pub struct TerrainBoundsCache {
    pub(super) entries: HashMap<ProceduralBoundsKey, ProceduralBounds>,
    pub(super) hits: u64,
    pub(super) misses: u64,
}

impl TerrainBoundsCache {
    /// Approximate retained cache memory, capped at 32 MiB.
    pub fn memory_bytes(&self) -> usize {
        self.entries
            .len()
            .saturating_mul(PROCEDURAL_BOUNDS_ENTRY_BYTES)
    }

    /// Cumulative cache hits and misses.
    pub const fn access_counts(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    pub(super) fn bounds(&mut self, field: &TerrainField, id: TerrainNodeId) -> ProceduralBounds {
        let key = ProceduralBoundsKey {
            level: id.level,
            x: id.coordinates.x,
            z: id.coordinates.z,
        };
        if let Some(bounds) = self.entries.get(&key).copied() {
            self.hits = self.hits.saturating_add(1);
            return bounds;
        }
        self.misses = self.misses.saturating_add(1);
        let minimum = id.minimum_cell_i64();
        let maximum = id.maximum_cell_exclusive_i64();
        let extent = maximum[0] - minimum[0] - 1;
        let x_coordinates = [
            minimum[0],
            minimum[0] + extent / 4,
            minimum[0] + extent / 2,
            minimum[0] + extent * 3 / 4,
            maximum[0] - 1,
        ];
        let extent = maximum[2] - minimum[2] - 1;
        let z_coordinates = [
            minimum[2],
            minimum[2] + extent / 4,
            minimum[2] + extent / 2,
            minimum[2] + extent * 3 / 4,
            maximum[2] - 1,
        ];
        let mut bounds = ProceduralBounds {
            minimum_surface: f64::INFINITY,
            maximum_surface: f64::NEG_INFINITY,
            margin: 0.0,
        };
        let mut surfaces = [[0.0_f64; 5]; 5];
        for (z_index, z) in z_coordinates.into_iter().enumerate() {
            for (x_index, x) in x_coordinates.into_iter().enumerate() {
                let x = x as f64 * crate::TERRAIN_CELL_METERS + crate::TERRAIN_CELL_METERS * 0.5;
                let z = z as f64 * crate::TERRAIN_CELL_METERS + crate::TERRAIN_CELL_METERS * 0.5;
                let surface = field.surface_height(x, z);
                surfaces[z_index][x_index] = surface;
                bounds.minimum_surface = bounds.minimum_surface.min(surface);
                bounds.maximum_surface = bounds.maximum_surface.max(surface);
            }
        }
        let mut maximum_step = 0.0_f64;
        for z in 0..5 {
            for x in 0..5 {
                if x + 1 < 5 {
                    maximum_step = maximum_step.max((surfaces[z][x + 1] - surfaces[z][x]).abs());
                }
                if z + 1 < 5 {
                    maximum_step = maximum_step.max((surfaces[z + 1][x] - surfaces[z][x]).abs());
                }
            }
        }
        let maximum_margin = id.edge_bricks() as f64 * BRICK_EDGE_METERS * 0.25;
        bounds.margin = maximum_step.mul_add(1.5, 0.25).min(maximum_margin);
        self.entries.insert(key, bounds);
        bounds
    }

    pub(super) fn evict_far_from(&mut self, focus: DVec3) {
        let maximum_entries = PROCEDURAL_BOUNDS_CACHE_BYTES / PROCEDURAL_BOUNDS_ENTRY_BYTES;
        if self.entries.len() <= maximum_entries {
            return;
        }
        let target = maximum_entries * 7 / 8;
        let mut keys = self.entries.keys().copied().collect::<Vec<_>>();
        keys.sort_by(|first, second| {
            let distance = |key: ProceduralBoundsKey| {
                let edge = (1_i64 << key.level) as f64 * BRICK_EDGE_METERS;
                let x = f64::from(key.x) * BRICK_EDGE_METERS + edge * 0.5;
                let z = f64::from(key.z) * BRICK_EDGE_METERS + edge * 0.5;
                (x - focus.x).mul_add(x - focus.x, (z - focus.z) * (z - focus.z))
            };
            distance(*second).total_cmp(&distance(*first))
        });
        for key in keys.into_iter().take(self.entries.len() - target) {
            self.entries.remove(&key);
        }
    }
}
