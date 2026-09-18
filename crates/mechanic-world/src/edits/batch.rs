//! Edit batches, their outcomes and errors, and the cell ranges a brush covers.

use crate::{BrickCoord, TERRAIN_CELL_METERS, TerrainMaterial, WorldCell, WorldPosition};
use bevy_math::DVec3;
use std::collections::BTreeSet;
use thiserror::Error;

/// Exact volume of one newly emptied terrain cell.
pub const REMOVED_CELL_CUBIC_METERS: f64 = 0.000_125;

/// Exact volume of one newly emptied terrain cell in litres.
pub const REMOVED_CELL_LITRES: f64 = 0.125;

/// Material-specific result of one terrain brush operation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TerrainEditOutcome {
    pub(super) removed_cells: [u64; TerrainMaterial::COUNT],
    pub(super) added_cells: [u64; TerrainMaterial::COUNT],
    /// Cells compressed, indexed by stable material code.
    pub compressed_cells: [u64; TerrainMaterial::COUNT],
    /// Sum of delivered cell displacements, not maximum surface rut depth.
    pub sunk_metres: f64,
    /// Number of 32³ bricks whose density changed.
    pub changed_bricks: usize,
    pub(super) changed_brick_coordinates: Vec<BrickCoord>,
}

/// Ordered terrain edit publication with the union of every changed brick.
///
/// A continuous brush stroke may contain many worker batches. Merging batches
/// retains only the newest exact edit generation while preserving the complete
/// footprint that foundation and render consumers must acknowledge.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TerrainEditBatch {
    /// Exact terrain generation after this batch was committed.
    pub generation: u64,
    /// Stable union of changed promoted bricks.
    pub changed_bricks: BTreeSet<BrickCoord>,
}

impl TerrainEditBatch {
    /// Creates one committed edit batch from its individual brush outcomes.
    pub fn from_outcomes(
        generation: u64,
        outcomes: impl IntoIterator<Item = TerrainEditOutcome>,
    ) -> Self {
        let mut batch = Self {
            generation,
            changed_bricks: BTreeSet::new(),
        };
        for outcome in outcomes {
            batch
                .changed_bricks
                .extend(outcome.changed_brick_coordinates);
        }
        batch
    }

    /// Merges a later ordered batch into the current continuous stroke.
    pub fn merge(&mut self, later: Self) {
        debug_assert!(later.generation >= self.generation);
        self.generation = self.generation.max(later.generation);
        self.changed_bricks.extend(later.changed_bricks);
    }

    /// True when no terrain cell changed.
    pub fn is_empty(&self) -> bool {
        self.changed_bricks.is_empty()
    }
}

impl TerrainEditOutcome {
    /// Number of cells newly removed from one material.
    pub const fn removed_cells(&self, material: TerrainMaterial) -> u64 {
        self.removed_cells[material.code() as usize]
    }

    /// Newly removed volume for one material, in cubic metres.
    pub fn cubic_metres(&self, material: TerrainMaterial) -> f64 {
        self.removed_cells(material) as f64 * REMOVED_CELL_CUBIC_METERS
    }

    /// Newly removed volume for one material, in litres.
    pub fn litres(&self, material: TerrainMaterial) -> f64 {
        self.removed_cells(material) as f64 * REMOVED_CELL_LITRES
    }

    /// Total number of newly removed cells across all materials.
    pub const fn total_removed_cells(&self) -> u64 {
        let mut total = 0;
        let mut index = 0;
        while index < TerrainMaterial::COUNT {
            total += self.removed_cells[index];
            index += 1;
        }
        total
    }

    /// Number of empty cells newly filled with one material.
    pub const fn added_cells(&self, material: TerrainMaterial) -> u64 {
        self.added_cells[material.code() as usize]
    }

    /// Total number of newly filled cells across all materials.
    pub const fn total_added_cells(&self) -> u64 {
        let mut total = 0;
        let mut index = 0;
        while index < TerrainMaterial::COUNT {
            total += self.added_cells[index];
            index += 1;
        }
        total
    }

    /// Total number of cells changed by this operation.
    pub const fn total_changed_cells(&self) -> u64 {
        let mut compressed = 0;
        let mut index = 0;
        while index < TerrainMaterial::COUNT {
            compressed += self.compressed_cells[index];
            index += 1;
        }
        self.total_removed_cells() + self.total_added_cells() + compressed
    }

    /// Stable coordinates of bricks whose samples changed in this operation.
    pub fn changed_brick_coordinates(&self) -> &[BrickCoord] {
        &self.changed_brick_coordinates
    }
}

/// Invalid terrain brush operation.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum TerrainEditError {
    /// Soil patch geometry, pressure or duration is invalid.
    #[error(
        "soil patch must have finite geometry, a unit normal, a 0.05–0.3 m radius and a duration within 0–1 s"
    )]
    InvalidSoilPatch,
    /// Radius must use the prototype's 0.10–2.00 m range.
    #[error("terrain brush radius {0} m is outside 0.10 through 2.00 m")]
    InvalidRadius(f64),
    /// The outer wall of the finite world is unbreakable.
    #[error("terrain edits cannot reach the unbreakable outer world boundary")]
    UnbreakableBoundary,
}

pub(super) fn cell_containing(position: DVec3) -> WorldCell {
    let scaled = (position / TERRAIN_CELL_METERS).floor();
    WorldCell::new(scaled.x as i32, scaled.y as i32, scaled.z as i32)
}

#[derive(Clone, Copy)]
pub(super) struct CellRange {
    pub(super) start: i32,
    pub(super) end: i32,
}

pub(super) fn sphere_y_cell_range(
    centre: WorldPosition,
    radius_squared: f64,
    x: f64,
    z: f64,
) -> Option<CellRange> {
    let horizontal_squared = (x - centre.0.x).powi(2) + (z - centre.0.z).powi(2);
    let half_height = (radius_squared - horizontal_squared).sqrt();
    if !half_height.is_finite() {
        return None;
    }
    let start = ((centre.0.y - half_height) / TERRAIN_CELL_METERS - 0.5).ceil() as i32;
    let end = ((centre.0.y + half_height) / TERRAIN_CELL_METERS - 0.5).floor() as i32;
    (start <= end).then_some(CellRange { start, end })
}
