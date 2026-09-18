//! Octree node identity, summaries, and the node itself.

use super::brick::TerrainBrick;
use crate::{BRICK_EDGE_CELLS, BrickCoord};
use std::array;
use std::sync::Arc;

pub(super) const OCTREE_DEPTH: u8 = 27;

pub(super) const ROOT_MINIMUM_BRICK: i32 = -(1 << 26);

/// Stable identity of one sparse terrain-octree node.
///
/// Level zero is one 32³-cell (1.6 m) promoted leaf. Each higher level doubles
/// both the node edge and sample spacing. `coordinates` is the minimum level-zero
/// brick coordinate covered by the node, so the level-27 root starts at
/// `(-2²⁶, -2²⁶, -2²⁶)` and covers every signed `i32` cell.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TerrainNodeId {
    /// Level-zero brick coordinate of the node's minimum corner.
    pub coordinates: BrickCoord,
    /// Zero for a promoted leaf; 27 for the complete signed-coordinate root.
    pub level: u8,
}

impl TerrainNodeId {
    /// Complete signed-cell-coordinate root.
    pub const ROOT: Self = Self {
        coordinates: BrickCoord::new(ROOT_MINIMUM_BRICK, ROOT_MINIMUM_BRICK, ROOT_MINIMUM_BRICK),
        level: OCTREE_DEPTH,
    };

    /// Node identity for one promoted brick.
    pub const fn leaf(coordinates: BrickCoord) -> Self {
        Self {
            coordinates,
            level: 0,
        }
    }

    /// Aligned node at `level` containing a level-zero brick.
    pub fn containing(coordinates: BrickCoord, level: u8) -> Option<Self> {
        if level > OCTREE_DEPTH {
            return None;
        }
        let edge = 1_i64 << level;
        let align = |coordinate: i32| {
            let offset = i64::from(coordinate) - i64::from(ROOT_MINIMUM_BRICK);
            i32::try_from(i64::from(ROOT_MINIMUM_BRICK) + offset.div_euclid(edge) * edge).ok()
        };
        Some(Self {
            coordinates: BrickCoord::new(
                align(coordinates.x)?,
                align(coordinates.y)?,
                align(coordinates.z)?,
            ),
            level,
        })
    }

    /// Minimum cell coordinate, represented in `i64` at the root extremes.
    pub fn minimum_cell_i64(self) -> [i64; 3] {
        [
            i64::from(self.coordinates.x) * i64::from(BRICK_EDGE_CELLS),
            i64::from(self.coordinates.y) * i64::from(BRICK_EDGE_CELLS),
            i64::from(self.coordinates.z) * i64::from(BRICK_EDGE_CELLS),
        ]
    }

    /// Exclusive maximum cell coordinate, represented in `i64`.
    pub fn maximum_cell_exclusive_i64(self) -> [i64; 3] {
        let edge = self.edge_bricks() * i64::from(BRICK_EDGE_CELLS);
        self.minimum_cell_i64().map(|coordinate| coordinate + edge)
    }

    /// Number of level-zero bricks on one edge.
    pub const fn edge_bricks(self) -> i64 {
        1_i64 << self.level
    }

    /// Parent node, or `None` for the root.
    ///
    /// # Panics
    ///
    /// Panics only if an internally constructed node lies outside the signed
    /// depth-27 root, which public constructors prevent.
    pub fn parent(self) -> Option<Self> {
        if self.level >= OCTREE_DEPTH {
            return None;
        }
        let parent_level = self.level + 1;
        let edge = 1_i64 << parent_level;
        let align = |coordinate: i32| {
            let offset = i64::from(coordinate) - i64::from(ROOT_MINIMUM_BRICK);
            i32::try_from(i64::from(ROOT_MINIMUM_BRICK) + offset.div_euclid(edge) * edge)
                .expect("octree parent remains in the signed brick domain")
        };
        Some(Self {
            coordinates: BrickCoord::new(
                align(self.coordinates.x),
                align(self.coordinates.y),
                align(self.coordinates.z),
            ),
            level: parent_level,
        })
    }

    /// Eight children in x/y/z bit order.
    ///
    /// # Panics
    ///
    /// Panics only for a corrupt level whose child edge cannot fit `i32`.
    pub fn children(self) -> Option<[Self; 8]> {
        let child_level = self.level.checked_sub(1)?;
        let edge = i32::try_from(1_i64 << child_level).expect("child edge fits i32");
        Some(array::from_fn(|index| Self {
            coordinates: BrickCoord::new(
                self.coordinates.x + if index & 1 == 0 { 0 } else { edge },
                self.coordinates.y + if index & 2 == 0 { 0 } else { edge },
                self.coordinates.z + if index & 4 == 0 { 0 } else { edge },
            ),
            level: child_level,
        }))
    }

    pub(super) fn child_index_containing(self, leaf: BrickCoord) -> usize {
        debug_assert!(self.level > 0);
        let half = i32::try_from(1_i64 << (self.level - 1)).expect("child edge fits i32");
        usize::from(leaf.x >= self.coordinates.x + half)
            | (usize::from(leaf.y >= self.coordinates.y + half) << 1)
            | (usize::from(leaf.z >= self.coordinates.z + half) << 2)
    }
}

/// Conservative occupancy classification for octree traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainDensityClass {
    /// Every represented density is non-positive.
    Empty,
    /// Every represented density is positive.
    Solid,
    /// The node may contain an isosurface.
    Mixed,
}

/// Public immutable metadata for one allocated node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainNodeSummary {
    /// Node identity.
    pub id: TerrainNodeId,
    /// Number of promoted leaves below this node.
    pub promoted_descendants: u64,
    /// Minimum density among promoted descendants.
    pub minimum_density: f32,
    /// Maximum density among promoted descendants.
    pub maximum_density: f32,
    /// Latest descendant revision.
    pub latest_revision: u64,
    /// Bit mask of allocated children.
    pub child_mask: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct TerrainNode {
    pub(super) id: TerrainNodeId,
    pub(super) children: [Option<Arc<Self>>; 8],
    pub(super) brick: Option<Arc<TerrainBrick>>,
    pub(super) promoted_descendants: u64,
    pub(super) minimum_density: f32,
    pub(super) maximum_density: f32,
    pub(super) latest_revision: u64,
}

impl TerrainNode {
    pub(super) fn empty(id: TerrainNodeId) -> Self {
        Self {
            id,
            children: array::from_fn(|_| None),
            brick: None,
            promoted_descendants: 0,
            minimum_density: f32::INFINITY,
            maximum_density: f32::NEG_INFINITY,
            latest_revision: 0,
        }
    }

    pub(super) fn refresh(&mut self) {
        if let Some(brick) = &self.brick {
            self.promoted_descendants = 1;
            self.minimum_density = brick.minimum_density;
            self.maximum_density = brick.maximum_density;
            self.latest_revision = brick.revision;
            return;
        }
        self.promoted_descendants = self
            .children
            .iter()
            .flatten()
            .map(|child| child.promoted_descendants)
            .sum();
        self.minimum_density = self
            .children
            .iter()
            .flatten()
            .map(|child| child.minimum_density)
            .fold(f32::INFINITY, f32::min);
        self.maximum_density = self
            .children
            .iter()
            .flatten()
            .map(|child| child.maximum_density)
            .fold(f32::NEG_INFINITY, f32::max);
        self.latest_revision = self
            .children
            .iter()
            .flatten()
            .map(|child| child.latest_revision)
            .max()
            .unwrap_or(0);
    }

    pub(super) fn summary(&self) -> TerrainNodeSummary {
        let child_mask = self
            .children
            .iter()
            .enumerate()
            .fold(0_u8, |mask, (index, child)| {
                mask | if child.is_some() { 1 << index } else { 0 }
            });
        TerrainNodeSummary {
            id: self.id,
            promoted_descendants: self.promoted_descendants,
            minimum_density: self.minimum_density,
            maximum_density: self.maximum_density,
            latest_revision: self.latest_revision,
            child_mask,
        }
    }
}
