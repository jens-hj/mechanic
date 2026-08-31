//! Stable global coordinates and floating-origin conversion.

#![allow(clippy::cast_possible_truncation)] // Range checks precede f64-to-cell conversion.

use bevy_math::{DVec3, IVec3, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Edge length of the hidden volumetric terrain cell.
pub const TERRAIN_CELL_METERS: f64 = 0.05;
/// Exact authored construction-position step.
pub const BUILD_POSITION_TICK_METERS: f64 = mechanic_core::POSITION_TICK_METERS as f64;
/// Half-width of the finite world from its centre.
pub const WORLD_HALF_EXTENT_METERS: f64 = 8_000.0;
/// Half-width of the finite world in terrain cells.
pub const WORLD_HALF_EXTENT_CELLS: i32 = 160_000;
/// Number of cells on one edge of a promoted brick.
pub const BRICK_EDGE_CELLS: i32 = 32;
/// Edge length of a promoted brick.
pub const BRICK_EDGE_METERS: f64 = 1.6;

/// Seed from which all untouched world data is regenerated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorldSeed(pub u64);

/// Version of the deterministic generation recipe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorldGeneratorVersion(pub u32);

impl WorldGeneratorVersion {
    /// Generator used by this build.
    pub const CURRENT: Self = Self(1);
}

/// Exact global coordinate of one terrain-cell centre.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct WorldCell {
    /// East/west cell coordinate.
    pub x: i32,
    /// Vertical cell coordinate.
    pub y: i32,
    /// North/south cell coordinate.
    pub z: i32,
}

impl WorldCell {
    /// Creates a cell coordinate.
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Returns the position at the centre of this cell.
    pub fn centre(self) -> WorldPosition {
        WorldPosition(DVec3::new(
            (f64::from(self.x) + 0.5) * TERRAIN_CELL_METERS,
            (f64::from(self.y) + 0.5) * TERRAIN_CELL_METERS,
            (f64::from(self.z) + 0.5) * TERRAIN_CELL_METERS,
        ))
    }

    /// True when the cell is inside the editable portion of the finite world.
    pub const fn is_editable(self) -> bool {
        self.x > -WORLD_HALF_EXTENT_CELLS
            && self.x < WORLD_HALF_EXTENT_CELLS - 1
            && self.z > -WORLD_HALF_EXTENT_CELLS
            && self.z < WORLD_HALF_EXTENT_CELLS - 1
    }

    /// Brick containing this cell, using Euclidean division for negative coordinates.
    pub fn brick(self) -> BrickCoord {
        BrickCoord::new(
            self.x.div_euclid(BRICK_EDGE_CELLS),
            self.y.div_euclid(BRICK_EDGE_CELLS),
            self.z.div_euclid(BRICK_EDGE_CELLS),
        )
    }

    /// Zero-based cell coordinate within its brick.
    pub fn local_in_brick(self) -> IVec3 {
        IVec3::new(
            self.x.rem_euclid(BRICK_EDGE_CELLS),
            self.y.rem_euclid(BRICK_EDGE_CELLS),
            self.z.rem_euclid(BRICK_EDGE_CELLS),
        )
    }
}

/// Continuous global position in metres, independent of floating-origin rebases.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorldPosition(pub DVec3);

impl WorldPosition {
    /// Converts to the cell containing this position.
    ///
    /// # Errors
    ///
    /// Returns [`CoordinateError::OutOfRange`] for non-finite positions or
    /// coordinates beyond the integer cell range.
    pub fn cell(self) -> Result<WorldCell, CoordinateError> {
        let scaled = (self.0 / TERRAIN_CELL_METERS).floor();
        if !scaled.is_finite()
            || scaled.cmplt(DVec3::splat(f64::from(i32::MIN))).any()
            || scaled.cmpgt(DVec3::splat(f64::from(i32::MAX))).any()
        {
            return Err(CoordinateError::OutOfRange(self));
        }
        Ok(WorldCell::new(
            scaled.x as i32,
            scaled.y as i32,
            scaled.z as i32,
        ))
    }

    /// True when the horizontal position lies inside the finite world.
    pub fn is_inside_world(self) -> bool {
        self.0.x.abs() < WORLD_HALF_EXTENT_METERS && self.0.z.abs() < WORLD_HALF_EXTENT_METERS
    }

    /// Converts this global position to a small local render/physics coordinate.
    pub fn relative_to(self, origin: FloatingOrigin) -> Vec3 {
        (self.0 - origin.0).as_vec3()
    }
}

/// Global point represented as local zero for rendering and physics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FloatingOrigin(pub DVec3);

impl FloatingOrigin {
    /// Rebases to whole metres near the supplied position when it is far enough away.
    pub fn rebase_for(&mut self, position: WorldPosition, threshold_metres: f64) -> Option<DVec3> {
        let offset = position.0 - self.0;
        if offset.x.abs().max(offset.y.abs()).max(offset.z.abs()) < threshold_metres {
            return None;
        }
        let previous = self.0;
        self.0 = position.0.round();
        Some(self.0 - previous)
    }
}

/// Integer coordinate of a promoted 32³-cell brick.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct BrickCoord {
    /// Brick x coordinate.
    pub x: i32,
    /// Brick y coordinate.
    pub y: i32,
    /// Brick z coordinate.
    pub z: i32,
}

impl BrickCoord {
    /// Creates a brick coordinate.
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Minimum cell corner belonging to this brick.
    pub const fn minimum_cell(self) -> WorldCell {
        WorldCell::new(
            self.x * BRICK_EDGE_CELLS,
            self.y * BRICK_EDGE_CELLS,
            self.z * BRICK_EDGE_CELLS,
        )
    }
}

/// Invalid conversion from a continuous coordinate.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum CoordinateError {
    /// Position cannot be represented by integer cell coordinates.
    #[error("world position {0:?} is outside the representable cell range")]
    OutOfRange(WorldPosition),
}

#[cfg(test)]
mod tests {
    use bevy_math::DVec3;

    use super::{BrickCoord, FloatingOrigin, WorldCell, WorldPosition};

    #[test]
    fn negative_cells_use_euclidean_bricks() {
        let cell = WorldCell::new(-1, -33, 32);
        assert_eq!(cell.brick(), BrickCoord::new(-1, -2, 1));
        assert_eq!(cell.local_in_brick().to_array(), [31, 31, 0]);
    }

    #[test]
    fn floating_origin_preserves_global_position() {
        let point = WorldPosition(DVec3::new(3_423.25, 91.5, -7_100.75));
        let mut origin = FloatingOrigin::default();
        let shift = origin.rebase_for(point, 512.0).expect("far point rebases");
        assert_eq!(shift, DVec3::new(3_423.0, 92.0, -7_101.0));
        assert!(point.relative_to(origin).length() < 1.0);
    }
}
