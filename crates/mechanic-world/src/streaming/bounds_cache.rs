//! A bounded cache of procedural density classifications per node.

use crate::{BRICK_EDGE_METERS, TerrainDensityClass, TerrainField, TerrainNodeId};
use bevy_math::DVec3;
use std::collections::HashMap;

pub(super) const PROCEDURAL_BOUNDS_CACHE_BYTES: usize = 32 * 1024 * 1024;

pub(super) const PROCEDURAL_BOUNDS_ENTRY_BYTES: usize = 48;

/// Per-world cache of conservative classifications of untouched terrain,
/// computed by interval evaluation over each node's box. Discard it whenever
/// the field changes.
#[derive(Clone, Debug, Default)]
pub struct TerrainBoundsCache {
    pub(super) entries: HashMap<(TerrainNodeId, bool), TerrainDensityClass>,
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

    pub(super) fn classify(
        &mut self,
        field: &TerrainField,
        id: TerrainNodeId,
        distant: bool,
    ) -> TerrainDensityClass {
        if let Some(class) = self.entries.get(&(id, distant)).copied() {
            self.hits = self.hits.saturating_add(1);
            return class;
        }
        self.misses = self.misses.saturating_add(1);
        let (minimum, maximum) = super::selection::node_bounds(id);
        // Lattice corners lie on the node's faces; a sliver of margin keeps
        // rounding on the boundary from ever deciding a class.
        let margin = DVec3::splat(crate::TERRAIN_CELL_METERS);
        let (minimum, maximum) = (minimum - margin, maximum + margin);
        let class = if distant {
            field.classify_distant(minimum, maximum)
        } else {
            field.classify(minimum, maximum)
        };
        self.entries.insert((id, distant), class);
        class
    }

    pub(super) fn evict_far_from(&mut self, focus: DVec3) {
        let maximum_entries = PROCEDURAL_BOUNDS_CACHE_BYTES / PROCEDURAL_BOUNDS_ENTRY_BYTES;
        if self.entries.len() <= maximum_entries {
            return;
        }
        let target = maximum_entries * 7 / 8;
        let mut keys = self.entries.keys().copied().collect::<Vec<_>>();
        keys.sort_by(|first, second| {
            let distance = |(key, _): (TerrainNodeId, bool)| {
                let edge = (1_i64 << key.level) as f64 * BRICK_EDGE_METERS;
                let x = f64::from(key.coordinates.x) * BRICK_EDGE_METERS + edge * 0.5;
                let z = f64::from(key.coordinates.z) * BRICK_EDGE_METERS + edge * 0.5;
                (x - focus.x).mul_add(x - focus.x, (z - focus.z) * (z - focus.z))
            };
            distance(*second).total_cmp(&distance(*first))
        });
        for key in keys.into_iter().take(self.entries.len() - target) {
            self.entries.remove(&key);
        }
    }
}
