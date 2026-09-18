//! Whether construction foundations still rest on terrain after an edit.

use super::scene::{TerrainDensity, raycast_density};
use crate::{BrickCoord, TERRAIN_CELL_METERS, TerrainRayHit, WorldPosition};
use bevy_math::DVec3;
use mechanic_core::PartId;
use std::collections::{BTreeMap, BTreeSet};

/// One 5 cm support sample under a terrain-anchored bottom face.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoundationSample {
    /// Global sample point on the construction's bottom face.
    pub position: WorldPosition,
    /// Whether terrain still reaches this sample.
    pub valid: bool,
}

/// Persistent terrain anchors for one static assembly.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FoundationSupport {
    /// Samples at exact 5 cm spacing.
    pub samples: Vec<FoundationSample>,
}

/// Result of refreshing only the support samples touched by terrain edits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FoundationRefresh {
    /// Samples whose terrain reachability was queried.
    pub sampled: usize,
    /// Previously valid samples that became invalid.
    pub anchors_changed: usize,
    /// No valid terrain anchor remains.
    pub detached: bool,
}

/// Incremental brick-to-foundation lookup used to avoid global anchor sweeps.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FoundationSpatialIndex {
    pub(super) by_brick: BTreeMap<BrickCoord, BTreeSet<PartId>>,
    pub(super) by_part: BTreeMap<PartId, BTreeSet<BrickCoord>>,
}

impl FoundationSpatialIndex {
    /// Adds or replaces the indexed footprint for one foundation.
    pub fn insert(&mut self, part: PartId, support: &FoundationSupport) {
        self.remove(part);
        let bricks = support.dependency_bricks();
        for &brick in &bricks {
            self.by_brick.entry(brick).or_default().insert(part);
        }
        if !bricks.is_empty() {
            self.by_part.insert(part, bricks);
        }
    }

    /// Merges a disjoint staged index without recomputing support footprints.
    pub fn append(&mut self, mut other: Self) {
        for (brick, mut parts) in other.by_brick {
            self.by_brick.entry(brick).or_default().append(&mut parts);
        }
        self.by_part.append(&mut other.by_part);
    }

    /// Removes one foundation footprint.
    pub fn remove(&mut self, part: PartId) {
        let Some(bricks) = self.by_part.remove(&part) else {
            return;
        };
        for brick in bricks {
            let remove_entry = self.by_brick.get_mut(&brick).is_some_and(|parts| {
                parts.remove(&part);
                parts.is_empty()
            });
            if remove_entry {
                self.by_brick.remove(&brick);
            }
        }
    }

    /// Foundations whose sample rays overlap at least one changed brick.
    pub fn candidates(&self, changed_bricks: &BTreeSet<BrickCoord>) -> BTreeSet<PartId> {
        changed_bricks
            .iter()
            .filter_map(|brick| self.by_brick.get(brick))
            .flatten()
            .copied()
            .collect()
    }

    /// Number of indexed foundations.
    pub fn len(&self) -> usize {
        self.by_part.len()
    }

    /// True when no foundation is indexed.
    pub fn is_empty(&self) -> bool {
        self.by_part.is_empty()
    }
}

impl FoundationSupport {
    /// Builds support samples for a world-up rectangular bottom face.
    pub fn rectangular(
        terrain: &impl TerrainDensity,
        hit: TerrainRayHit,
        size_x: f64,
        size_z: f64,
    ) -> Self {
        let snapped_x = snap_5_cm(hit.position.0.x);
        let snapped_z = snap_5_cm(hit.position.0.z);
        let count_x = (size_x / TERRAIN_CELL_METERS).round().max(1.0) as i32;
        let count_z = (size_z / TERRAIN_CELL_METERS).round().max(1.0) as i32;
        let minimum_x = snapped_x - (f64::from(count_x - 1) * TERRAIN_CELL_METERS * 0.5);
        let minimum_z = snapped_z - (f64::from(count_z - 1) * TERRAIN_CELL_METERS * 0.5);
        let mut samples = Vec::with_capacity(usize::try_from(count_x * count_z).unwrap_or(0));
        for z in 0..count_z {
            for x in 0..count_x {
                let position = WorldPosition(DVec3::new(
                    minimum_x + f64::from(x) * TERRAIN_CELL_METERS,
                    hit.position.0.y,
                    minimum_z + f64::from(z) * TERRAIN_CELL_METERS,
                ));
                samples.push(FoundationSample {
                    position,
                    valid: terrain_reaches(terrain, position),
                });
            }
        }
        Self { samples }
    }

    /// Refreshes anchors after an edit and returns true when the assembly must detach.
    pub fn refresh(&mut self, terrain: &impl TerrainDensity) -> bool {
        for sample in &mut self.samples {
            if sample.valid {
                sample.valid = terrain_reaches(terrain, sample.position);
            }
        }
        !self.has_valid_anchor()
    }

    /// Refreshes only valid samples whose short support ray overlaps an edited brick.
    pub fn refresh_changed(
        &mut self,
        terrain: &impl TerrainDensity,
        changed_bricks: &BTreeSet<BrickCoord>,
    ) -> FoundationRefresh {
        let mut refresh = FoundationRefresh::default();
        for sample in &mut self.samples {
            if !sample.valid || !sample_overlaps_bricks(sample.position, changed_bricks) {
                continue;
            }
            refresh.sampled += 1;
            sample.valid = terrain_reaches(terrain, sample.position);
            refresh.anchors_changed += usize::from(!sample.valid);
        }
        refresh.detached = !self.has_valid_anchor();
        refresh
    }

    /// Terrain bricks read by this support's short vertical sample rays.
    pub fn dependency_bricks(&self) -> BTreeSet<BrickCoord> {
        self.samples
            .iter()
            .flat_map(|sample| sample_dependency_bricks(sample.position))
            .collect()
    }

    /// True while at least one terrain anchor remains valid.
    pub fn has_valid_anchor(&self) -> bool {
        self.samples.iter().any(|sample| sample.valid)
    }

    /// Number of currently valid support samples.
    pub fn valid_count(&self) -> usize {
        self.samples.iter().filter(|sample| sample.valid).count()
    }

    /// Number of persistent terrain samples in this footprint.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

pub(super) fn sample_overlaps_bricks(
    position: WorldPosition,
    changed_bricks: &BTreeSet<BrickCoord>,
) -> bool {
    sample_dependency_bricks(position).any(|brick| changed_bricks.contains(&brick))
}

pub(super) fn sample_dependency_bricks(
    position: WorldPosition,
) -> impl Iterator<Item = BrickCoord> {
    let top = WorldPosition(position.0 + DVec3::Y * TERRAIN_CELL_METERS);
    let bottom = WorldPosition(position.0 - DVec3::Y * TERRAIN_CELL_METERS * 1.1);
    let top = top.cell().ok().map(crate::WorldCell::brick);
    let bottom = bottom.cell().ok().map(crate::WorldCell::brick);
    [top, bottom].into_iter().flatten()
}

pub(super) fn terrain_reaches(terrain: &impl TerrainDensity, position: WorldPosition) -> bool {
    raycast_density(
        terrain,
        WorldPosition(position.0 + DVec3::Y * TERRAIN_CELL_METERS),
        DVec3::NEG_Y,
        TERRAIN_CELL_METERS * 2.1,
    )
    .is_some()
}

pub(super) fn snap_5_cm(value: f64) -> f64 {
    (value / TERRAIN_CELL_METERS).round() * TERRAIN_CELL_METERS
}
