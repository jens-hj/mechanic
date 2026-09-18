//! Sparse promotion and persistent subtractive terrain edits.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    array,
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use bevy_math::{DVec3, IVec3};
use thiserror::Error;

use crate::{
    BRICK_EDGE_CELLS, BrickCoord, SoilCompression, SoilPatch, SoilResponse, TERRAIN_CELL_METERS,
    TerrainField, TerrainMaterial, TerrainSample, WORLD_HALF_EXTENT_METERS, WorldCell,
    WorldPosition,
    generation::TerrainColumnSample,
    soil::{COMPACTION_STEP_METRES, MAX_SOIL_DEPTH_METRES},
};

const BRICK_CELL_COUNT: usize = 32 * 32 * 32;
const EMPTY_DENSITY: f32 = -0.5 * TERRAIN_CELL_METERS as f32;
const BRICK_MAGIC: [u8; 4] = *b"MECB";
const BRICK_FORMAT_VERSION: u16 = 3;
const OCTREE_DEPTH: u8 = 27;
const ROOT_MINIMUM_BRICK: i32 = -(1 << 26);

/// Exact volume of one newly emptied terrain cell.
pub const REMOVED_CELL_CUBIC_METERS: f64 = 0.000_125;
/// Exact volume of one newly emptied terrain cell in litres.
pub const REMOVED_CELL_LITRES: f64 = 0.125;

/// Material-specific result of one terrain brush operation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TerrainEditOutcome {
    removed_cells: [u64; TerrainMaterial::COUNT],
    added_cells: [u64; TerrainMaterial::COUNT],
    /// Cells compressed, indexed by stable material code.
    pub compressed_cells: [u64; TerrainMaterial::COUNT],
    /// Sum of delivered cell displacements, not maximum surface rut depth.
    pub sunk_metres: f64,
    /// Number of 32³ bricks whose density changed.
    pub changed_bricks: usize,
    changed_brick_coordinates: Vec<BrickCoord>,
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

/// Fully promoted 32³-cell brick and its density acceleration bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainBrick {
    coordinate: BrickCoord,
    cells: Vec<TerrainSample>,
    minimum_density: f32,
    maximum_density: f32,
    revision: u64,
}

impl TerrainBrick {
    /// Brick coordinate.
    pub const fn coordinate(&self) -> BrickCoord {
        self.coordinate
    }

    /// Minimum density in the brick, used to skip known-empty hierarchy nodes.
    pub const fn minimum_density(&self) -> f32 {
        self.minimum_density
    }

    /// Maximum density in the brick, used to skip known-solid hierarchy nodes.
    pub const fn maximum_density(&self) -> f32 {
        self.maximum_density
    }

    /// Latest terrain revision represented by this leaf.
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Returns the promoted cell at a zero-based local coordinate.
    pub fn sample(&self, local: IVec3) -> Option<TerrainSample> {
        local_index(local).map(|index| self.cells[index])
    }

    fn promote(field: &TerrainField, coordinate: BrickCoord) -> Self {
        let minimum = coordinate.minimum_cell();
        let mut columns = Vec::with_capacity(usize::try_from(BRICK_EDGE_CELLS.pow(2)).unwrap());
        for z in 0..BRICK_EDGE_CELLS {
            for x in 0..BRICK_EDGE_CELLS {
                let position = WorldCell::new(minimum.x + x, minimum.y, minimum.z + z).centre();
                columns.push(field.sample_column(position.0.x, position.0.z));
            }
        }
        let mut cells = Vec::with_capacity(BRICK_CELL_COUNT);
        let mut minimum_density = f32::INFINITY;
        let mut maximum_density = f32::NEG_INFINITY;
        for z in 0..BRICK_EDGE_CELLS {
            for y in 0..BRICK_EDGE_CELLS {
                for x in 0..BRICK_EDGE_CELLS {
                    let cell = WorldCell::new(minimum.x + x, minimum.y + y, minimum.z + z);
                    let column = columns[usize::try_from(x + z * BRICK_EDGE_CELLS)
                        .expect("local index is positive")];
                    let sample = field.sample_cell_in_column(cell, column);
                    minimum_density = minimum_density.min(sample.density);
                    maximum_density = maximum_density.max(sample.density);
                    cells.push(sample);
                }
            }
        }
        Self {
            coordinate,
            cells,
            minimum_density,
            maximum_density,
            revision: 0,
        }
    }

    #[allow(clippy::cast_sign_loss)] // Depth is finite and positive before quantisation.
    fn compress(&mut self, local: IVec3, depth: f32) -> Option<f32> {
        let index = local_index(local)?;
        let sample = &mut self.cells[index];
        let before = sample.density;
        let bounded = before;
        let depth = depth
            .min(MAX_SOIL_DEPTH_METRES)
            .min(bounded - EMPTY_DENSITY);
        if !depth.is_finite() || depth < COMPACTION_STEP_METRES {
            return None;
        }
        let steps = (depth / COMPACTION_STEP_METRES).floor();
        let depth = steps * COMPACTION_STEP_METRES;
        sample.density = (bounded - depth).max(EMPTY_DENSITY);
        sample.compaction = sample.compaction.saturating_add(steps as u8);
        self.minimum_density = self.minimum_density.min(sample.density);
        if before >= self.maximum_density {
            self.maximum_density = self
                .cells
                .iter()
                .map(|cell| cell.density)
                .fold(f32::NEG_INFINITY, f32::max);
        }
        Some(depth)
    }

    fn set_empty(&mut self, local: IVec3) -> Option<TerrainMaterial> {
        let index = local_index(local)?;
        let sample = &mut self.cells[index];
        if !sample.is_solid() {
            return None;
        }
        let removed = sample.material;
        let removed_density = sample.density;
        sample.density = EMPTY_DENSITY;
        sample.compaction = 0;
        self.minimum_density = self.minimum_density.min(EMPTY_DENSITY);
        if removed_density >= self.maximum_density {
            self.maximum_density = self
                .cells
                .iter()
                .map(|cell| cell.density)
                .fold(f32::NEG_INFINITY, f32::max);
        }
        Some(removed)
    }

    fn set_solid(&mut self, local: IVec3, material: TerrainMaterial, density: f32) -> bool {
        let Some(index) = local_index(local) else {
            return false;
        };
        let sample = &mut self.cells[index];
        if sample.is_solid() {
            return false;
        }
        let previous_density = sample.density;
        *sample = TerrainSample {
            density,
            material,
            compaction: 0,
        };
        self.maximum_density = self.maximum_density.max(density);
        if previous_density <= self.minimum_density {
            self.minimum_density = self
                .cells
                .iter()
                .map(|cell| cell.density)
                .fold(f32::INFINITY, f32::min);
        }
        true
    }
}

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

    fn child_index_containing(self, leaf: BrickCoord) -> usize {
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
struct TerrainNode {
    id: TerrainNodeId,
    children: [Option<Arc<Self>>; 8],
    brick: Option<Arc<TerrainBrick>>,
    promoted_descendants: u64,
    minimum_density: f32,
    maximum_density: f32,
    latest_revision: u64,
}

impl TerrainNode {
    fn empty(id: TerrainNodeId) -> Self {
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

    fn refresh(&mut self) {
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

    fn summary(&self) -> TerrainNodeSummary {
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

/// Sparse depth-27 terrain octree with copy-on-write immutable snapshots.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainOctree {
    root: Arc<TerrainNode>,
    dirty: BTreeSet<TerrainNodeId>,
    next_revision: u64,
}

impl Default for TerrainOctree {
    fn default() -> Self {
        Self {
            root: Arc::new(TerrainNode::empty(TerrainNodeId::ROOT)),
            dirty: BTreeSet::new(),
            next_revision: 1,
        }
    }
}

/// Cheap immutable terrain input for background selection, meshing, and queries.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainOctreeSnapshot {
    root: Arc<TerrainNode>,
}

/// Common point-sampling contract for live and snapshotted octrees.
pub trait TerrainSource {
    /// Gets one promoted leaf.
    fn brick(&self, coordinate: BrickCoord) -> Option<&TerrainBrick>;

    /// Samples a cell, falling through to procedural generation when untouched.
    fn sample_cell(&self, field: &TerrainField, cell: WorldCell) -> TerrainSample {
        self.brick(cell.brick())
            .and_then(|brick| brick.sample(cell.local_in_brick()))
            .unwrap_or_else(|| field.sample_cell(cell))
    }

    /// Samples the cell containing a continuous position.
    fn sample_position(&self, field: &TerrainField, position: WorldPosition) -> TerrainSample {
        position.cell().map_or_else(
            |_| field.sample_position(position),
            |cell| self.sample_cell(field, cell),
        )
    }
}

impl TerrainOctree {
    /// Number of allocated 5 cm bricks.
    pub fn promoted_brick_count(&self) -> usize {
        usize::try_from(self.root.promoted_descendants).unwrap_or(usize::MAX)
    }

    /// Gets one promoted brick.
    pub fn brick(&self, coordinate: BrickCoord) -> Option<&TerrainBrick> {
        find_brick(&self.root, coordinate)
    }

    /// Captures a constant-time immutable root for background work.
    #[must_use]
    pub fn snapshot(&self) -> TerrainOctreeSnapshot {
        TerrainOctreeSnapshot {
            root: Arc::clone(&self.root),
        }
    }

    /// Iterates dirty leaf identities in stable order.
    pub fn dirty_leaves(&self) -> impl Iterator<Item = TerrainNodeId> + '_ {
        self.dirty.iter().copied()
    }

    /// Iterates every promoted leaf coordinate in stable order.
    pub fn brick_coordinates(&self) -> impl Iterator<Item = BrickCoord> {
        collect_bricks(&self.root)
            .into_iter()
            .map(TerrainBrick::coordinate)
    }

    /// Marks a brick clean after its atomic save succeeds.
    pub fn mark_saved(&mut self, coordinate: BrickCoord) {
        self.dirty.remove(&TerrainNodeId::leaf(coordinate));
    }

    /// Samples an edit when promoted and otherwise falls through to generation.
    pub fn sample_cell(&self, field: &TerrainField, cell: WorldCell) -> TerrainSample {
        TerrainSource::sample_cell(self, field, cell)
    }

    pub(crate) fn sample_cell_in_column(
        &self,
        field: &TerrainField,
        cell: WorldCell,
        column: TerrainColumnSample,
    ) -> TerrainSample {
        self.brick(cell.brick())
            .and_then(|brick| brick.sample(cell.local_in_brick()))
            .unwrap_or_else(|| field.sample_cell_in_column(cell, column))
    }

    /// Samples the cell containing a continuous position.
    pub fn sample_position(&self, field: &TerrainField, position: WorldPosition) -> TerrainSample {
        TerrainSource::sample_position(self, field, position)
    }

    /// Promotes a procedural brick to explicit 5 cm samples.
    ///
    /// # Panics
    ///
    /// Panics only if the internal insert path fails to retain the new leaf.
    pub fn promote(&mut self, field: &TerrainField, coordinate: BrickCoord) -> &TerrainBrick {
        if self.brick(coordinate).is_none() {
            self.insert_brick(TerrainBrick::promote(field, coordinate));
        }
        self.brick(coordinate).expect("promoted brick was inserted")
    }

    /// Applies a pressure patch to the exposed cells of upward-facing terrain.
    ///
    /// # Errors
    /// Rejects invalid patches and world-boundary overlap before promoting bricks.
    pub fn compress_patch(
        &mut self,
        field: &TerrainField,
        patch: SoilPatch,
    ) -> Result<TerrainEditOutcome, TerrainEditError> {
        let cells = self.soil_compressions(field, patch)?;
        Ok(self.compress_cells(field, &cells))
    }

    pub(crate) fn soil_compressions(
        &self,
        field: &TerrainField,
        patch: SoilPatch,
    ) -> Result<Vec<SoilCompression>, TerrainEditError> {
        patch.validate()?;
        let mut cells = Vec::new();
        if patch.normal.y < 0.25 {
            return Ok(cells);
        }
        // Include a short vertical search below a stale contact mesh. Each column
        // yields only its first exposed sample, never the whole supporting volume.
        let extent = DVec3::new(
            patch.radius,
            patch.radius + TERRAIN_CELL_METERS * 2.0,
            patch.radius,
        );
        let minimum = cell_containing(patch.centre.0 - extent);
        let maximum = cell_containing(patch.centre.0 + extent);
        for z in minimum.z..=maximum.z {
            for x in minimum.x..=maximum.x {
                let position = WorldCell::new(x, minimum.y, z).centre();
                let column = field.sample_column(position.0.x, position.0.z);
                for y in (minimum.y..=maximum.y).rev() {
                    let cell = WorldCell::new(x, y, z);
                    let sample = self.sample_cell_in_column(field, cell, column);
                    if sample.density <= EMPTY_DENSITY + COMPACTION_STEP_METRES
                        || (!sample.is_solid() && sample.compaction == 0)
                    {
                        continue;
                    }
                    let offset = cell.centre().0 - patch.centre.0;
                    let tangent = offset - patch.normal * offset.dot(patch.normal);
                    if tangent.length_squared() <= patch.radius * patch.radius {
                        let depth = SoilResponse::for_material(sample.material).depth(
                            sample.compaction,
                            patch.pressure_pa,
                            patch.seconds,
                        );
                        if depth > 0.0 {
                            cells.push(SoilCompression {
                                cell,
                                sample,
                                depth,
                            });
                        }
                    }
                    break;
                }
            }
        }
        Ok(cells)
    }

    /// Commits accumulated per-cell displacements, dropping loads for changed samples.
    /// This preserves async edit ordering without applying old loads to new material.
    pub fn compress_cells(
        &mut self,
        field: &TerrainField,
        cells: &[SoilCompression],
    ) -> TerrainEditOutcome {
        let mut by_leaf = BTreeMap::<BrickCoord, Vec<&SoilCompression>>::new();
        for cell in cells {
            if cell.depth.is_finite()
                && cell.depth >= COMPACTION_STEP_METRES
                && SoilResponse::for_material(cell.sample.material)
                    .bearing_capacity_pa
                    .is_finite()
                && cell.cell.centre().is_inside_world()
                && self.sample_cell(field, cell.cell) == cell.sample
            {
                by_leaf.entry(cell.cell.brick()).or_default().push(cell);
            }
        }
        let mut outcome = TerrainEditOutcome::default();
        for (coordinate, cells) in by_leaf {
            let mut brick = self
                .brick(coordinate)
                .cloned()
                .unwrap_or_else(|| TerrainBrick::promote(field, coordinate));
            let mut changed = false;
            for cell in cells {
                if brick.sample(cell.cell.local_in_brick()) != Some(cell.sample) {
                    continue;
                }
                if let Some(depth) = brick.compress(cell.cell.local_in_brick(), cell.depth) {
                    outcome.compressed_cells[cell.sample.material.code() as usize] += 1;
                    outcome.sunk_metres += f64::from(depth);
                    changed = true;
                }
            }
            if changed {
                brick.revision = self.next_revision;
                self.insert_brick(brick);
                self.dirty.insert(TerrainNodeId::leaf(coordinate));
                outcome.changed_brick_coordinates.push(coordinate);
            }
        }
        outcome.changed_bricks = outcome.changed_brick_coordinates.len();
        if outcome.changed_bricks > 0 {
            self.next_revision = self.next_revision.wrapping_add(1).max(1);
        }
        outcome
    }

    /// Removes exactly the authorized cells, or leaves the terrain unchanged if
    /// any source is stale, repeated, empty, or outside the editable world.
    /// The caller must publish the returned material and terrain together.
    pub fn extract_cells(
        &mut self,
        field: &TerrainField,
        cells: &[crate::ExtractionCell],
    ) -> Option<TerrainEditOutcome> {
        if !crate::breakage::unique_sources(cells)
            || cells.iter().any(|source| {
                !source.cell.is_editable()
                    || !source.sample.is_solid()
                    || self.sample_cell(field, source.cell) != source.sample
            })
        {
            return None;
        }
        let mut grouped = BTreeMap::<BrickCoord, Vec<WorldCell>>::new();
        for source in cells {
            grouped
                .entry(source.cell.brick())
                .or_default()
                .push(source.cell);
        }
        let mut outcome = TerrainEditOutcome::default();
        for (coordinate, cells) in grouped {
            let mut brick = self
                .brick(coordinate)
                .cloned()
                .unwrap_or_else(|| TerrainBrick::promote(field, coordinate));
            for cell in cells {
                if let Some(material) = brick.set_empty(cell.local_in_brick()) {
                    outcome.removed_cells[material.code() as usize] += 1;
                }
            }
            brick.revision = self.next_revision;
            self.insert_brick(brick);
            self.dirty.insert(TerrainNodeId::leaf(coordinate));
            outcome.changed_brick_coordinates.push(coordinate);
        }
        outcome.changed_bricks = outcome.changed_brick_coordinates.len();
        if outcome.changed_bricks > 0 {
            self.next_revision = self.next_revision.wrapping_add(1).max(1);
        }
        Some(outcome)
    }

    /// Deposits whole cells of loose material into empty terrain supported from
    /// below. Returns the applied edit and the unplaced quantity. Fractional
    /// cells remain owned by the clump; they are never rounded away.
    pub fn deposit_material(
        &mut self,
        field: &TerrainField,
        cells: &[WorldCell],
        material: TerrainMaterial,
        mut quanta: u64,
    ) -> (TerrainEditOutcome, u64) {
        let mut outcome = TerrainEditOutcome::default();
        if !crate::BreakageResponse::for_material(material).deposits {
            return (outcome, quanta);
        }
        let mut staged = BTreeMap::<BrickCoord, TerrainBrick>::new();
        for &cell in cells {
            if quanta < 510 {
                break;
            }
            if !cell.is_editable() || cell.y == i32::MIN {
                continue;
            }
            let sample_at = |cell: WorldCell| {
                staged
                    .get(&cell.brick())
                    .and_then(|brick| brick.sample(cell.local_in_brick()))
                    .unwrap_or_else(|| self.sample_cell(field, cell))
            };
            if sample_at(cell).is_solid()
                || !sample_at(WorldCell::new(cell.x, cell.y - 1, cell.z)).is_solid()
            {
                continue;
            }
            let coordinate = cell.brick();
            let brick = staged.entry(coordinate).or_insert_with(|| {
                self.brick(coordinate)
                    .cloned()
                    .unwrap_or_else(|| TerrainBrick::promote(field, coordinate))
            });
            if brick.set_solid(cell.local_in_brick(), material, -EMPTY_DENSITY) {
                quanta -= 510;
                outcome.added_cells[material.code() as usize] += 1;
                brick.revision = self.next_revision;
            }
        }
        for (coordinate, brick) in staged {
            self.insert_brick(brick);
            self.dirty.insert(TerrainNodeId::leaf(coordinate));
            outcome.changed_brick_coordinates.push(coordinate);
        }
        outcome.changed_bricks = outcome.changed_brick_coordinates.len();
        if outcome.changed_bricks > 0 {
            self.next_revision = self.next_revision.wrapping_add(1).max(1);
        }
        (outcome, quanta)
    }

    /// Subtracts a spherical brush and reports only cells that became empty.
    ///
    /// # Errors
    ///
    /// Refuses invalid radii and any brush that reaches the unbreakable outer
    /// world boundary. No brick is promoted when an edit is refused.
    pub fn excavate_sphere(
        &mut self,
        field: &TerrainField,
        centre: WorldPosition,
        radius_metres: f64,
    ) -> Result<TerrainEditOutcome, TerrainEditError> {
        self.excavate_sphere_delta(field, centre, radius_metres, None)
    }

    /// Subtracts only the part of a spherical brush not covered by the previous sample.
    ///
    /// Passing the last successfully applied brush sample makes continuous strokes
    /// proportional to their newly swept volume instead of rescanning the complete
    /// sphere every rendered frame. The result is identical to applying both full
    /// spheres because cells inside `previous` are already empty.
    ///
    /// # Errors
    ///
    /// Refuses invalid radii and any current brush that reaches the unbreakable
    /// outer world boundary. No brick is promoted when an edit is refused.
    pub fn excavate_sphere_delta(
        &mut self,
        field: &TerrainField,
        centre: WorldPosition,
        radius_metres: f64,
        previous: Option<(WorldPosition, f64)>,
    ) -> Result<TerrainEditOutcome, TerrainEditError> {
        if !radius_metres.is_finite() || !(0.10..=2.00).contains(&radius_metres) {
            return Err(TerrainEditError::InvalidRadius(radius_metres));
        }
        if centre.0.x.abs() + radius_metres >= WORLD_HALF_EXTENT_METERS
            || centre.0.z.abs() + radius_metres >= WORLD_HALF_EXTENT_METERS
        {
            return Err(TerrainEditError::UnbreakableBoundary);
        }

        let minimum = cell_containing(centre.0 - DVec3::splat(radius_metres));
        let maximum = cell_containing(centre.0 + DVec3::splat(radius_metres));
        let radius_squared = radius_metres * radius_metres;
        let mut cells_to_empty = Vec::new();
        let mut touched_bricks = BTreeSet::new();
        for z in minimum.z..=maximum.z {
            for x in minimum.x..=maximum.x {
                let column_position = WorldCell::new(x, minimum.y, z).centre();
                let Some(current_y) = sphere_y_cell_range(
                    centre,
                    radius_squared,
                    column_position.0.x,
                    column_position.0.z,
                ) else {
                    continue;
                };
                let column = field.sample_column(column_position.0.x, column_position.0.z);
                let previous_y = previous.and_then(|(previous_centre, previous_radius)| {
                    sphere_y_cell_range(
                        previous_centre,
                        previous_radius * previous_radius,
                        column_position.0.x,
                        column_position.0.z,
                    )
                });
                let mut sample_range = |first: i32, last: i32| {
                    for y in first..=last {
                        let cell = WorldCell::new(x, y, z);
                        if !self.sample_cell_in_column(field, cell, column).is_solid() {
                            continue;
                        }
                        cells_to_empty.push(cell);
                        touched_bricks.insert(cell.brick());
                    }
                };
                if let Some(previous_y) = previous_y {
                    sample_range(current_y.start, current_y.end.min(previous_y.start - 1));
                    sample_range(current_y.start.max(previous_y.end + 1), current_y.end);
                } else {
                    sample_range(current_y.start, current_y.end);
                }
            }
        }
        let mut outcome = TerrainEditOutcome::default();
        let mut changed = BTreeSet::new();
        let mut by_leaf = BTreeMap::<BrickCoord, Vec<WorldCell>>::new();
        for cell in cells_to_empty {
            by_leaf.entry(cell.brick()).or_default().push(cell);
        }
        let revision = self.next_revision;
        for coordinate in touched_bricks {
            let mut brick = self
                .brick(coordinate)
                .cloned()
                .unwrap_or_else(|| TerrainBrick::promote(field, coordinate));
            let Some(cells) = by_leaf.get(&coordinate) else {
                continue;
            };
            for cell in cells {
                if let Some(material) = brick.set_empty(cell.local_in_brick()) {
                    outcome.removed_cells[material.code() as usize] += 1;
                    changed.insert(coordinate);
                }
            }
            if changed.contains(&coordinate) {
                brick.revision = revision;
                self.insert_brick(brick);
            }
        }
        outcome.changed_bricks = changed.len();
        outcome.changed_brick_coordinates = changed.iter().copied().collect();
        if !changed.is_empty() {
            self.next_revision = self.next_revision.wrapping_add(1).max(1);
        }
        self.dirty
            .extend(changed.into_iter().map(TerrainNodeId::leaf));
        Ok(outcome)
    }

    /// Fills empty cells inside a spherical brush with `material`.
    ///
    /// Existing solid cells are never repainted.
    ///
    /// # Errors
    ///
    /// Refuses invalid radii and any brush that reaches the unbreakable outer
    /// world boundary. No brick is promoted when an edit is refused.
    pub fn add_sphere(
        &mut self,
        field: &TerrainField,
        centre: WorldPosition,
        radius_metres: f64,
        material: TerrainMaterial,
    ) -> Result<TerrainEditOutcome, TerrainEditError> {
        self.add_sphere_delta(field, centre, radius_metres, material, None)
    }

    /// Fills only the part of a spherical brush not covered by the previous sample.
    ///
    /// Passing the last successfully applied sample makes a continuous stroke
    /// proportional to its newly swept volume. Existing solids remain untouched.
    ///
    /// # Errors
    ///
    /// Refuses invalid radii and any current brush that reaches the unbreakable
    /// outer world boundary. No brick is promoted when an edit is refused.
    pub fn add_sphere_delta(
        &mut self,
        field: &TerrainField,
        centre: WorldPosition,
        radius_metres: f64,
        material: TerrainMaterial,
        previous: Option<(WorldPosition, f64)>,
    ) -> Result<TerrainEditOutcome, TerrainEditError> {
        if !radius_metres.is_finite() || !(0.10..=2.00).contains(&radius_metres) {
            return Err(TerrainEditError::InvalidRadius(radius_metres));
        }
        if centre.0.x.abs() + radius_metres >= WORLD_HALF_EXTENT_METERS
            || centre.0.z.abs() + radius_metres >= WORLD_HALF_EXTENT_METERS
        {
            return Err(TerrainEditError::UnbreakableBoundary);
        }

        let minimum = cell_containing(centre.0 - DVec3::splat(radius_metres));
        let maximum = cell_containing(centre.0 + DVec3::splat(radius_metres));
        let radius_squared = radius_metres * radius_metres;
        let mut cells_to_fill = Vec::new();
        let mut touched_bricks = BTreeSet::new();
        for z in minimum.z..=maximum.z {
            for x in minimum.x..=maximum.x {
                let column_position = WorldCell::new(x, minimum.y, z).centre();
                let Some(current_y) = sphere_y_cell_range(
                    centre,
                    radius_squared,
                    column_position.0.x,
                    column_position.0.z,
                ) else {
                    continue;
                };
                let column = field.sample_column(column_position.0.x, column_position.0.z);
                let previous_y = previous.and_then(|(previous_centre, previous_radius)| {
                    sphere_y_cell_range(
                        previous_centre,
                        previous_radius * previous_radius,
                        column_position.0.x,
                        column_position.0.z,
                    )
                });
                let mut sample_range = |first: i32, last: i32| {
                    for y in first..=last {
                        let cell = WorldCell::new(x, y, z);
                        if self.sample_cell_in_column(field, cell, column).is_solid() {
                            continue;
                        }
                        let distance = cell.centre().0.distance(centre.0);
                        let density =
                            (radius_metres - distance).max(TERRAIN_CELL_METERS * 0.5) as f32;
                        cells_to_fill.push((cell, density));
                        touched_bricks.insert(cell.brick());
                    }
                };
                if let Some(previous_y) = previous_y {
                    sample_range(current_y.start, current_y.end.min(previous_y.start - 1));
                    sample_range(current_y.start.max(previous_y.end + 1), current_y.end);
                } else {
                    sample_range(current_y.start, current_y.end);
                }
            }
        }

        let mut outcome = TerrainEditOutcome::default();
        let mut changed = BTreeSet::new();
        let mut by_leaf = BTreeMap::<BrickCoord, Vec<(WorldCell, f32)>>::new();
        for (cell, density) in cells_to_fill {
            by_leaf
                .entry(cell.brick())
                .or_default()
                .push((cell, density));
        }
        let revision = self.next_revision;
        for coordinate in touched_bricks {
            let mut brick = self
                .brick(coordinate)
                .cloned()
                .unwrap_or_else(|| TerrainBrick::promote(field, coordinate));
            let Some(cells) = by_leaf.get(&coordinate) else {
                continue;
            };
            for (cell, density) in cells {
                if brick.set_solid(cell.local_in_brick(), material, *density) {
                    outcome.added_cells[material.code() as usize] += 1;
                    changed.insert(coordinate);
                }
            }
            if changed.contains(&coordinate) {
                brick.revision = revision;
                self.insert_brick(brick);
            }
        }
        outcome.changed_bricks = changed.len();
        outcome.changed_brick_coordinates = changed.iter().copied().collect();
        if !changed.is_empty() {
            self.next_revision = self.next_revision.wrapping_add(1).max(1);
        }
        self.dirty
            .extend(changed.into_iter().map(TerrainNodeId::leaf));
        Ok(outcome)
    }

    /// Inserts a decoded saved brick, replacing only the same coordinate.
    pub fn insert_saved_brick(&mut self, brick: TerrainBrick) {
        let coordinate = brick.coordinate;
        self.next_revision = self.next_revision.max(brick.revision.wrapping_add(1));
        self.dirty.remove(&TerrainNodeId::leaf(coordinate));
        self.insert_brick(brick);
    }

    /// Gets allocated metadata for a node on the promoted path.
    pub fn node(&self, id: TerrainNodeId) -> Option<TerrainNodeSummary> {
        find_node(&self.root, id).map(TerrainNode::summary)
    }

    /// Traverses allocated nodes intersecting an inclusive leaf-coordinate region.
    pub fn nodes_between(
        &self,
        minimum: BrickCoord,
        maximum: BrickCoord,
    ) -> impl Iterator<Item = TerrainNodeSummary> {
        let mut nodes = Vec::new();
        collect_nodes_between(&self.root, minimum, maximum, &mut nodes);
        nodes.into_iter()
    }

    /// Conservative classification combining exact promoted bounds with the
    /// procedural generator for untouched space.
    pub fn classify(&self, field: &TerrainField, id: TerrainNodeId) -> TerrainDensityClass {
        classify_node(&self.root, field, id)
    }

    fn insert_brick(&mut self, brick: TerrainBrick) {
        insert_brick_node(&mut self.root, Arc::new(brick));
    }
}

impl TerrainOctreeSnapshot {
    /// Gets one promoted brick.
    pub fn brick(&self, coordinate: BrickCoord) -> Option<&TerrainBrick> {
        find_brick(&self.root, coordinate)
    }

    /// Samples a cell with the procedural field as fallback.
    pub fn sample_cell(&self, field: &TerrainField, cell: WorldCell) -> TerrainSample {
        TerrainSource::sample_cell(self, field, cell)
    }

    /// Samples a continuous position.
    pub fn sample_position(&self, field: &TerrainField, position: WorldPosition) -> TerrainSample {
        TerrainSource::sample_position(self, field, position)
    }

    /// Gets allocated metadata for a node.
    pub fn node(&self, id: TerrainNodeId) -> Option<TerrainNodeSummary> {
        find_node(&self.root, id).map(TerrainNode::summary)
    }

    /// Latest promoted revision in an inclusive brick-coordinate region.
    ///
    /// Fully covered octree nodes use their cached revision, so mesh dependency
    /// checks do not need to enumerate every promoted brick in the sampling halo.
    pub(crate) fn latest_revision_between(&self, minimum: BrickCoord, maximum: BrickCoord) -> u64 {
        latest_revision_between(&self.root, minimum, maximum)
    }

    /// Iterates promoted leaves in stable coordinate order.
    pub fn bricks(&self) -> impl Iterator<Item = &TerrainBrick> {
        collect_bricks(&self.root).into_iter()
    }

    /// Iterates promoted leaves intersecting an inclusive brick-coordinate region.
    ///
    /// Traversal prunes unrelated octree branches, allowing mesh jobs to prepare
    /// a compact local edit view without scanning every edit in the world.
    pub fn bricks_between(
        &self,
        minimum: BrickCoord,
        maximum: BrickCoord,
    ) -> impl Iterator<Item = &TerrainBrick> {
        let mut bricks = Vec::new();
        collect_bricks_between(&self.root, minimum, maximum, &mut bricks);
        bricks.sort_by_key(|brick| brick.coordinate);
        bricks.into_iter()
    }

    /// Conservative node classification.
    pub fn classify(&self, field: &TerrainField, id: TerrainNodeId) -> TerrainDensityClass {
        classify_node(&self.root, field, id)
    }

    /// Minimum promoted density in an inclusive cell range.
    ///
    /// Fully covered octree nodes use their cached range minimum. Only partial
    /// promoted leaves inspect individual cells, so coarse mesh sampling scales
    /// with the sparse edit hierarchy instead of the requested volume.
    pub fn minimum_promoted_density_between(
        &self,
        minimum: WorldCell,
        maximum: WorldCell,
    ) -> Option<f32> {
        minimum_promoted_density_between(&self.root, minimum, maximum)
    }
}

impl TerrainSource for TerrainOctree {
    fn brick(&self, coordinate: BrickCoord) -> Option<&TerrainBrick> {
        self.brick(coordinate)
    }
}

impl TerrainSource for TerrainOctreeSnapshot {
    fn brick(&self, coordinate: BrickCoord) -> Option<&TerrainBrick> {
        self.brick(coordinate)
    }
}

fn find_brick(node: &TerrainNode, coordinate: BrickCoord) -> Option<&TerrainBrick> {
    if node.id.level == 0 {
        return (node.id.coordinates == coordinate)
            .then_some(node.brick.as_deref())
            .flatten();
    }
    let index = node.id.child_index_containing(coordinate);
    node.children[index]
        .as_deref()
        .and_then(|child| find_brick(child, coordinate))
}

fn find_node(node: &TerrainNode, id: TerrainNodeId) -> Option<&TerrainNode> {
    if node.id == id {
        return Some(node);
    }
    if id.level >= node.id.level {
        return None;
    }
    let index = node.id.child_index_containing(id.coordinates);
    node.children[index]
        .as_deref()
        .and_then(|child| find_node(child, id))
}

fn insert_brick_node(node: &mut Arc<TerrainNode>, brick: Arc<TerrainBrick>) {
    let node = Arc::make_mut(node);
    if node.id.level == 0 {
        debug_assert_eq!(node.id.coordinates, brick.coordinate);
        node.brick = Some(brick);
        node.refresh();
        return;
    }
    let index = node.id.child_index_containing(brick.coordinate);
    let child_id = node.id.children().expect("a branch has children")[index];
    let child = node.children[index].get_or_insert_with(|| Arc::new(TerrainNode::empty(child_id)));
    insert_brick_node(child, brick);
    node.refresh();
}

fn collect_bricks(node: &TerrainNode) -> Vec<&TerrainBrick> {
    fn visit<'a>(node: &'a TerrainNode, bricks: &mut Vec<&'a TerrainBrick>) {
        if let Some(brick) = &node.brick {
            bricks.push(brick);
            return;
        }
        for child in node.children.iter().flatten() {
            visit(child, bricks);
        }
    }
    let mut bricks = Vec::with_capacity(usize::try_from(node.promoted_descendants).unwrap_or(0));
    visit(node, &mut bricks);
    bricks.sort_by_key(|brick| brick.coordinate);
    bricks
}

fn collect_bricks_between<'a>(
    node: &'a TerrainNode,
    minimum: BrickCoord,
    maximum: BrickCoord,
    bricks: &mut Vec<&'a TerrainBrick>,
) {
    if node.promoted_descendants == 0 || !node_intersects(node.id, minimum, maximum) {
        return;
    }
    if let Some(brick) = &node.brick {
        bricks.push(brick);
        return;
    }
    for child in node.children.iter().flatten() {
        collect_bricks_between(child, minimum, maximum, bricks);
    }
}

fn node_intersects(id: TerrainNodeId, minimum: BrickCoord, maximum: BrickCoord) -> bool {
    let edge = id.edge_bricks();
    let maximum_node = [id.coordinates.x, id.coordinates.y, id.coordinates.z]
        .map(|coordinate| i64::from(coordinate) + edge - 1);
    maximum_node[0] >= i64::from(minimum.x)
        && i64::from(id.coordinates.x) <= i64::from(maximum.x)
        && maximum_node[1] >= i64::from(minimum.y)
        && i64::from(id.coordinates.y) <= i64::from(maximum.y)
        && maximum_node[2] >= i64::from(minimum.z)
        && i64::from(id.coordinates.z) <= i64::from(maximum.z)
}

fn minimum_promoted_density_between(
    node: &TerrainNode,
    minimum: WorldCell,
    maximum: WorldCell,
) -> Option<f32> {
    if node.promoted_descendants == 0 {
        return None;
    }
    let node_minimum = node.id.minimum_cell_i64();
    let node_maximum = node.id.maximum_cell_exclusive_i64();
    let query_minimum = [minimum.x, minimum.y, minimum.z].map(i64::from);
    let query_maximum = [maximum.x, maximum.y, maximum.z].map(i64::from);
    if (0..3).any(|axis| {
        node_maximum[axis] <= query_minimum[axis] || node_minimum[axis] > query_maximum[axis]
    }) {
        return None;
    }
    if (0..3).all(|axis| {
        query_minimum[axis] <= node_minimum[axis] && query_maximum[axis] >= node_maximum[axis] - 1
    }) {
        return Some(node.minimum_density);
    }
    if let Some(brick) = &node.brick {
        let brick_minimum = brick.coordinate.minimum_cell();
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
        let mut result = f32::INFINITY;
        for z in first.z..=last.z {
            for y in first.y..=last.y {
                for x in first.x..=last.x {
                    result = result.min(
                        brick
                            .sample(IVec3::new(x, y, z))
                            .expect("clamped promoted coordinate is inside its brick")
                            .density,
                    );
                }
            }
        }
        return result.is_finite().then_some(result);
    }
    node.children
        .iter()
        .flatten()
        .filter_map(|child| minimum_promoted_density_between(child, minimum, maximum))
        .reduce(f32::min)
}

fn collect_nodes_between(
    node: &TerrainNode,
    minimum: BrickCoord,
    maximum: BrickCoord,
    nodes: &mut Vec<TerrainNodeSummary>,
) {
    if !node_intersects(node.id, minimum, maximum) {
        return;
    }
    nodes.push(node.summary());
    for child in node.children.iter().flatten() {
        collect_nodes_between(child, minimum, maximum, nodes);
    }
}

fn latest_revision_between(node: &TerrainNode, minimum: BrickCoord, maximum: BrickCoord) -> u64 {
    if node.promoted_descendants == 0 || !node_intersects(node.id, minimum, maximum) {
        return 0;
    }
    let node_minimum = node.id.coordinates;
    let node_maximum = node.id.edge_bricks() - 1;
    let node_maximum = BrickCoord::new(
        i32::try_from(i64::from(node_minimum.x) + node_maximum)
            .expect("octree node maximum fits brick coordinates"),
        i32::try_from(i64::from(node_minimum.y) + node_maximum)
            .expect("octree node maximum fits brick coordinates"),
        i32::try_from(i64::from(node_minimum.z) + node_maximum)
            .expect("octree node maximum fits brick coordinates"),
    );
    let fully_covered = minimum.x <= node_minimum.x
        && minimum.y <= node_minimum.y
        && minimum.z <= node_minimum.z
        && maximum.x >= node_maximum.x
        && maximum.y >= node_maximum.y
        && maximum.z >= node_maximum.z;
    if fully_covered || node.brick.is_some() {
        return node.latest_revision;
    }
    node.children
        .iter()
        .flatten()
        .map(|child| latest_revision_between(child, minimum, maximum))
        .max()
        .unwrap_or(0)
}

fn classify_node(
    root: &TerrainNode,
    field: &TerrainField,
    id: TerrainNodeId,
) -> TerrainDensityClass {
    if let Some(node) = find_node(root, id)
        && id
            .edge_bricks()
            .checked_pow(3)
            .and_then(|count| u64::try_from(count).ok())
            .is_some_and(|count| node.promoted_descendants == count)
    {
        return if node.maximum_density <= 0.0 {
            TerrainDensityClass::Empty
        } else if node.minimum_density > 0.0 {
            TerrainDensityClass::Solid
        } else {
            TerrainDensityClass::Mixed
        };
    }

    // Untouched regions are classified only when all eight corners and the
    // centre agree with a margin at least as wide as the node. Anything less
    // certain remains mixed, so traversal can never discard a procedural or
    // edited isosurface.
    let edge_cells = id.edge_bricks() * i64::from(BRICK_EDGE_CELLS);
    let minimum_cell = [id.coordinates.x, id.coordinates.y, id.coordinates.z]
        .map(|coordinate| i64::from(coordinate) * i64::from(BRICK_EDGE_CELLS));
    let maximum_cell = minimum_cell.map(|coordinate| coordinate + edge_cells - 1);
    let mut minimum_density = f32::INFINITY;
    let mut maximum_density = f32::NEG_INFINITY;
    for z in [minimum_cell[2], maximum_cell[2]] {
        for y in [minimum_cell[1], maximum_cell[1]] {
            for x in [minimum_cell[0], maximum_cell[0]] {
                let Ok(x) = i32::try_from(x) else {
                    return TerrainDensityClass::Mixed;
                };
                let Ok(y) = i32::try_from(y) else {
                    return TerrainDensityClass::Mixed;
                };
                let Ok(z) = i32::try_from(z) else {
                    return TerrainDensityClass::Mixed;
                };
                let density = field.sample_cell(WorldCell::new(x, y, z)).density;
                minimum_density = minimum_density.min(density);
                maximum_density = maximum_density.max(density);
            }
        }
    }
    let margin = (edge_cells as f64 * TERRAIN_CELL_METERS) as f32;
    if maximum_density < -margin {
        TerrainDensityClass::Empty
    } else if minimum_density > margin
        && find_node(root, id).is_none_or(|node| node.promoted_descendants == 0)
    {
        TerrainDensityClass::Solid
    } else {
        TerrainDensityClass::Mixed
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

/// Corrupt or unsupported edited-brick payload.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum BrickDecodeError {
    /// Header or version is not recognized.
    #[error("edited terrain brick has an unsupported header or version")]
    UnsupportedHeader,
    /// Payload ended before a complete record.
    #[error("edited terrain brick is truncated")]
    Truncated,
    /// Material byte is not a v1 material.
    #[error("edited terrain brick contains unknown material code {0}")]
    UnknownMaterial(u8),
    /// Runs did not expand to exactly 32³ cells.
    #[error("edited terrain brick expands to {0} cells instead of {BRICK_CELL_COUNT}")]
    InvalidCellCount(usize),
}

/// Encodes one promoted brick using versioned run-length compression.
pub fn encode_brick(brick: &TerrainBrick) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(BRICK_CELL_COUNT / 2);
    bytes.extend_from_slice(&BRICK_MAGIC);
    bytes.extend_from_slice(&BRICK_FORMAT_VERSION.to_le_bytes());
    for coordinate in [brick.coordinate.x, brick.coordinate.y, brick.coordinate.z] {
        bytes.extend_from_slice(&coordinate.to_le_bytes());
    }
    bytes.extend_from_slice(&brick.revision.to_le_bytes());
    let mut index = 0;
    while index < brick.cells.len() {
        let sample = brick.cells[index];
        let mut run = 1_usize;
        while index + run < brick.cells.len()
            && brick.cells[index + run] == sample
            && run < usize::from(u16::MAX)
        {
            run += 1;
        }
        bytes.extend_from_slice(&u16::try_from(run).unwrap_or(u16::MAX).to_le_bytes());
        bytes.extend_from_slice(&sample.density.to_bits().to_le_bytes());
        bytes.push(sample.material.code());
        bytes.push(sample.compaction);
        index += run;
    }
    bytes
}

/// Decodes a saved edited brick without falling back to procedural data.
///
/// # Errors
///
/// Returns a precise corruption error; callers must preserve the source file.
pub fn decode_brick(bytes: &[u8]) -> Result<TerrainBrick, BrickDecodeError> {
    if bytes.get(..4) != Some(&BRICK_MAGIC) || read_u16(bytes, 4)? != BRICK_FORMAT_VERSION {
        return Err(BrickDecodeError::UnsupportedHeader);
    }
    let coordinate = BrickCoord::new(
        read_i32(bytes, 6)?,
        read_i32(bytes, 10)?,
        read_i32(bytes, 14)?,
    );
    let mut cells = Vec::with_capacity(BRICK_CELL_COUNT);
    let revision = read_u64(bytes, 18)?;
    let mut cursor = 26;
    while cursor < bytes.len() {
        let run = usize::from(read_u16(bytes, cursor)?);
        let density = f32::from_bits(read_u32(bytes, cursor + 2)?);
        let compaction = *bytes.get(cursor + 7).ok_or(BrickDecodeError::Truncated)?;
        let code = *bytes.get(cursor + 6).ok_or(BrickDecodeError::Truncated)?;
        let material =
            TerrainMaterial::from_code(code).ok_or(BrickDecodeError::UnknownMaterial(code))?;
        if run == 0 || cells.len() + run > BRICK_CELL_COUNT {
            return Err(BrickDecodeError::InvalidCellCount(cells.len() + run));
        }
        cells.extend(std::iter::repeat_n(
            TerrainSample {
                density,
                material,
                compaction,
            },
            run,
        ));
        cursor += 8;
    }
    if cells.len() != BRICK_CELL_COUNT {
        return Err(BrickDecodeError::InvalidCellCount(cells.len()));
    }
    let minimum_density = cells
        .iter()
        .map(|cell| cell.density)
        .fold(f32::INFINITY, f32::min);
    let maximum_density = cells
        .iter()
        .map(|cell| cell.density)
        .fold(f32::NEG_INFINITY, f32::max);
    Ok(TerrainBrick {
        coordinate,
        cells,
        minimum_density,
        maximum_density,
        revision,
    })
}

fn cell_containing(position: DVec3) -> WorldCell {
    let scaled = (position / TERRAIN_CELL_METERS).floor();
    WorldCell::new(scaled.x as i32, scaled.y as i32, scaled.z as i32)
}

#[derive(Clone, Copy)]
struct CellRange {
    start: i32,
    end: i32,
}

fn sphere_y_cell_range(
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

fn local_index(local: IVec3) -> Option<usize> {
    if local.cmplt(IVec3::ZERO).any() || local.cmpge(IVec3::splat(BRICK_EDGE_CELLS)).any() {
        return None;
    }
    let index = local.x + local.y * BRICK_EDGE_CELLS + local.z * BRICK_EDGE_CELLS.pow(2);
    usize::try_from(index).ok()
}

fn read_u16(bytes: &[u8], cursor: usize) -> Result<u16, BrickDecodeError> {
    let raw = bytes
        .get(cursor..cursor + 2)
        .ok_or(BrickDecodeError::Truncated)?;
    Ok(u16::from_le_bytes([raw[0], raw[1]]))
}

fn read_u32(bytes: &[u8], cursor: usize) -> Result<u32, BrickDecodeError> {
    let raw = bytes
        .get(cursor..cursor + 4)
        .ok_or(BrickDecodeError::Truncated)?;
    Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

fn read_u64(bytes: &[u8], cursor: usize) -> Result<u64, BrickDecodeError> {
    let raw = bytes
        .get(cursor..cursor + 8)
        .ok_or(BrickDecodeError::Truncated)?;
    Ok(u64::from_le_bytes(
        raw.try_into().expect("slice length checked"),
    ))
}

fn read_i32(bytes: &[u8], cursor: usize) -> Result<i32, BrickDecodeError> {
    Ok(i32::from_le_bytes(read_u32(bytes, cursor)?.to_le_bytes()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)] // Exact litres-per-cell accounting is intentional.
    use bevy_math::DVec3;

    use super::{TerrainEditError, TerrainOctree, decode_brick, encode_brick};
    use crate::{BrickCoord, TerrainField, TerrainMaterial, WorldCell, WorldPosition, WorldSeed};

    #[test]
    fn depth_twenty_seven_root_covers_both_signed_cell_extremes() {
        let minimum = crate::WorldCell::new(i32::MIN, i32::MIN, i32::MIN).brick();
        let maximum = crate::WorldCell::new(i32::MAX, i32::MAX, i32::MAX).brick();
        assert_eq!(
            super::TerrainNodeId::containing(minimum, 27),
            Some(super::TerrainNodeId::ROOT)
        );
        assert_eq!(
            super::TerrainNodeId::containing(maximum, 27),
            Some(super::TerrainNodeId::ROOT)
        );
        assert_eq!(
            super::TerrainNodeId::ROOT.minimum_cell_i64(),
            [i64::from(i32::MIN); 3]
        );
        assert_eq!(
            super::TerrainNodeId::ROOT.maximum_cell_exclusive_i64(),
            [i64::from(i32::MAX) + 1; 3]
        );
    }

    #[test]
    fn untouched_space_allocates_nothing_and_promotion_is_sparse() {
        let field = TerrainField::new(WorldSeed(4));
        let mut edits = TerrainOctree::default();
        assert_eq!(edits.promoted_brick_count(), 0);
        edits.promote(&field, BrickCoord::new(0, 0, 0));
        assert_eq!(edits.promoted_brick_count(), 1);
        assert_eq!(
            edits
                .nodes_between(BrickCoord::new(0, 0, 0), BrickCoord::new(0, 0, 0))
                .count(),
            28
        );
    }

    #[test]
    fn ancestor_bounds_revisions_and_region_traversal_follow_promoted_leaves() {
        let field = TerrainField::new(WorldSeed(44));
        let mut terrain = TerrainOctree::default();
        let surface = field.surface_height(0.0, 0.0);
        let outcome = terrain
            .excavate_sphere(
                &field,
                WorldPosition(DVec3::new(0.0, surface - 0.1, 0.0)),
                0.25,
            )
            .unwrap();
        let leaf = super::TerrainNodeId::leaf(outcome.changed_brick_coordinates()[0]);
        let leaf_summary = terrain.node(leaf).unwrap();
        assert_eq!(leaf_summary.latest_revision, 1);
        assert!(leaf_summary.minimum_density <= leaf_summary.maximum_density);
        let mut ancestor = leaf;
        while let Some(parent) = ancestor.parent() {
            let summary = terrain.node(parent).unwrap();
            assert_eq!(summary.latest_revision, 1);
            assert!(summary.promoted_descendants >= 1);
            assert!(summary.minimum_density <= leaf_summary.minimum_density);
            assert!(summary.maximum_density >= leaf_summary.maximum_density);
            ancestor = parent;
        }
        let traversed = terrain
            .nodes_between(leaf.coordinates, leaf.coordinates)
            .collect::<Vec<_>>();
        assert_eq!(traversed.first().unwrap().id, super::TerrainNodeId::ROOT);
        assert_eq!(traversed.last().unwrap().id, leaf);
    }

    #[test]
    fn octree_range_minimum_matches_promoted_cell_scan() {
        let field = TerrainField::new(WorldSeed(45));
        let mut terrain = TerrainOctree::default();
        terrain
            .excavate_sphere(&field, WorldPosition(DVec3::new(0.8, 3.8, 0.8)), 0.7)
            .unwrap();
        let snapshot = terrain.snapshot();
        let minimum = WorldCell::new(-4, 60, -3);
        let maximum = WorldCell::new(37, 91, 35);
        let brute = snapshot
            .bricks()
            .flat_map(|brick| {
                let brick_minimum = brick.coordinate().minimum_cell();
                (0..super::BRICK_EDGE_CELLS).flat_map(move |z| {
                    (0..super::BRICK_EDGE_CELLS).flat_map(move |y| {
                        (0..super::BRICK_EDGE_CELLS).filter_map(move |x| {
                            let cell = WorldCell::new(
                                brick_minimum.x + x,
                                brick_minimum.y + y,
                                brick_minimum.z + z,
                            );
                            (cell.x >= minimum.x
                                && cell.y >= minimum.y
                                && cell.z >= minimum.z
                                && cell.x <= maximum.x
                                && cell.y <= maximum.y
                                && cell.z <= maximum.z)
                                .then(|| brick.sample(cell.local_in_brick()).unwrap().density)
                        })
                    })
                })
            })
            .reduce(f32::min);
        assert_eq!(
            snapshot.minimum_promoted_density_between(minimum, maximum),
            brute
        );
    }

    #[test]
    fn snapshot_isolation_is_copy_on_write() {
        let field = TerrainField::new(WorldSeed(55));
        let surface = field.surface_height(0.0, 0.0);
        let centre = WorldPosition(DVec3::new(0.0, surface - 0.1, 0.0));
        let cell = centre.cell().unwrap();
        let mut terrain = TerrainOctree::default();
        let snapshot = terrain.snapshot();
        assert!(snapshot.sample_cell(&field, cell).is_solid());
        terrain.excavate_sphere(&field, centre, 0.25).unwrap();
        assert!(snapshot.sample_cell(&field, cell).is_solid());
        assert!(!terrain.sample_cell(&field, cell).is_solid());
    }

    #[test]
    fn insertion_order_rebuilds_identical_hierarchy() {
        let field = TerrainField::new(WorldSeed(66));
        let coordinates = [
            BrickCoord::new(-12, 3, 7),
            BrickCoord::new(8, -2, 1),
            BrickCoord::new(0, 0, 0),
        ];
        let mut forward = TerrainOctree::default();
        for coordinate in coordinates {
            forward.promote(&field, coordinate);
        }
        let mut reverse = TerrainOctree::default();
        for coordinate in coordinates.into_iter().rev() {
            reverse.promote(&field, coordinate);
        }
        assert_eq!(forward, reverse);
    }

    #[test]
    fn excavating_untouched_air_does_not_promote_empty_bricks() {
        let field = TerrainField::new(WorldSeed(4));
        let mut edits = TerrainOctree::default();
        let centre = WorldPosition(DVec3::new(0.0, field.surface_height(0.0, 0.0) + 10.0, 0.0));
        let outcome = edits
            .excavate_sphere(&field, centre, 0.5)
            .expect("air brush lies inside the world");
        assert_eq!(outcome.total_removed_cells(), 0);
        assert_eq!(edits.promoted_brick_count(), 0);
    }

    #[test]
    fn excavation_is_idempotent_and_accounts_by_material() {
        let field = TerrainField::new(WorldSeed(123));
        let surface = field.surface_height(300.0, 300.0);
        let centre = WorldPosition(DVec3::new(300.0, surface - 1.4, 300.0));
        let mut edits = TerrainOctree::default();
        let first = edits
            .excavate_sphere(&field, centre, 1.5)
            .expect("valid edit");
        assert!(first.total_removed_cells() > 0);
        assert_eq!(
            first.changed_bricks,
            first.changed_brick_coordinates().len()
        );
        assert!(
            first
                .changed_brick_coordinates()
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
        assert!(first.removed_cells(TerrainMaterial::SurfaceCover) > 0);
        assert!(first.removed_cells(TerrainMaterial::Soil) > 0);
        assert!(first.removed_cells(TerrainMaterial::Rock) > 0);
        assert_eq!(
            first.litres(TerrainMaterial::Soil),
            first.removed_cells(TerrainMaterial::Soil) as f64 * 0.125
        );

        let second = edits
            .excavate_sphere(&field, centre, 1.5)
            .expect("repeat is valid");
        assert_eq!(second.total_removed_cells(), 0);
        assert_eq!(second.changed_bricks, 0);
        assert!(second.changed_brick_coordinates().is_empty());
    }

    #[test]
    fn delta_excavation_matches_two_complete_overlapping_spheres() {
        let field = TerrainField::new(WorldSeed(321));
        let surface = field.surface_height(300.0, 300.0);
        let first_centre = WorldPosition(DVec3::new(300.0, surface - 0.8, 300.0));
        let second_centre = WorldPosition(first_centre.0 + DVec3::new(0.05, 0.0, 0.0));

        let mut complete = TerrainOctree::default();
        complete
            .excavate_sphere(&field, first_centre, 1.0)
            .expect("first complete sphere is valid");
        let complete_outcome = complete
            .excavate_sphere(&field, second_centre, 1.0)
            .expect("second complete sphere is valid");

        let mut delta = TerrainOctree::default();
        delta
            .excavate_sphere(&field, first_centre, 1.0)
            .expect("first delta sphere is valid");
        let delta_outcome = delta
            .excavate_sphere_delta(&field, second_centre, 1.0, Some((first_centre, 1.0)))
            .expect("second delta sphere is valid");

        assert_eq!(delta, complete);
        assert_eq!(delta_outcome.removed_cells, complete_outcome.removed_cells);
    }

    #[test]
    fn addition_assigns_material_without_repainting_existing_solids() {
        let field = TerrainField::new(WorldSeed(456));
        let surface = field.surface_height(300.0, 300.0);
        let centre = WorldPosition(DVec3::new(300.0, surface + 0.3, 300.0));
        let existing_cell = super::cell_containing(DVec3::new(300.0, surface - 0.1, 300.0));
        let mut terrain = TerrainOctree::default();
        let existing = terrain.sample_cell(&field, existing_cell);
        assert!(existing.is_solid());

        let first = terrain
            .add_sphere(&field, centre, 0.5, TerrainMaterial::Rock)
            .expect("addition is valid");
        assert!(first.added_cells(TerrainMaterial::Rock) > 0);
        assert_eq!(first.total_added_cells(), first.total_changed_cells());
        assert_eq!(terrain.sample_cell(&field, existing_cell), existing);

        let added_cell = super::cell_containing(centre.0);
        let added = terrain.sample_cell(&field, added_cell);
        assert!(added.is_solid());
        assert_eq!(added.material, TerrainMaterial::Rock);

        let second = terrain
            .add_sphere(&field, centre, 0.5, TerrainMaterial::Soil)
            .expect("repeat addition is valid");
        assert_eq!(second.total_added_cells(), 0);
        assert_eq!(
            terrain.sample_cell(&field, added_cell).material,
            TerrainMaterial::Rock
        );
    }

    #[test]
    fn delta_addition_matches_two_complete_overlapping_spheres() {
        let field = TerrainField::new(WorldSeed(654));
        let surface = field.surface_height(300.0, 300.0);
        let first_centre = WorldPosition(DVec3::new(300.0, surface + 0.4, 300.0));
        let second_centre = WorldPosition(first_centre.0 + DVec3::new(0.05, 0.0, 0.0));

        let mut complete = TerrainOctree::default();
        complete
            .add_sphere(&field, first_centre, 0.6, TerrainMaterial::Soil)
            .unwrap();
        complete
            .add_sphere(&field, second_centre, 0.6, TerrainMaterial::Soil)
            .unwrap();

        let mut delta = TerrainOctree::default();
        delta
            .add_sphere(&field, first_centre, 0.6, TerrainMaterial::Soil)
            .unwrap();
        delta
            .add_sphere_delta(
                &field,
                second_centre,
                0.6,
                TerrainMaterial::Soil,
                Some((first_centre, 0.6)),
            )
            .unwrap();

        assert_eq!(delta, complete);
    }

    #[test]
    fn boundary_refusal_does_not_promote_bricks() {
        let field = TerrainField::new(WorldSeed(1));
        let mut edits = TerrainOctree::default();
        assert_eq!(
            edits.excavate_sphere(&field, WorldPosition(DVec3::new(7_999.5, 0.0, 0.0)), 1.0),
            Err(TerrainEditError::UnbreakableBoundary)
        );
        assert_eq!(edits.promoted_brick_count(), 0);
    }

    #[test]
    fn edited_brick_rle_round_trips_exactly() {
        let field = TerrainField::new(WorldSeed(8));
        let mut edits = TerrainOctree::default();
        let surface = field.surface_height(0.0, 0.0);
        edits
            .excavate_sphere(&field, WorldPosition(DVec3::new(0.0, surface, 0.0)), 0.25)
            .unwrap();
        let coordinate = edits
            .dirty_leaves()
            .next()
            .expect("the cut changes a brick")
            .coordinates;
        let mut original = edits.brick(coordinate).unwrap().clone();
        original.cells[0].compaction = 173;
        let original = &original;
        let decoded = decode_brick(&encode_brick(original)).expect("payload is valid");
        assert_eq!(&decoded, original);
    }

    #[test]
    fn added_material_rle_round_trips_exactly() {
        let field = TerrainField::new(WorldSeed(9));
        let surface = field.surface_height(0.0, 0.0);
        for material in TerrainMaterial::ALL {
            let mut edits = TerrainOctree::default();
            edits
                .add_sphere(
                    &field,
                    WorldPosition(DVec3::new(0.0, surface + 0.3, 0.0)),
                    0.5,
                    material,
                )
                .unwrap();
            assert!(edits.snapshot().bricks().next().is_some(), "{material:?}");
            for original in edits.snapshot().bricks() {
                let decoded = decode_brick(&encode_brick(original)).expect("payload is valid");
                assert_eq!(&decoded, original, "{material:?}");
            }
        }
    }
    fn soil_fixture(
        material: TerrainMaterial,
    ) -> (
        TerrainField,
        TerrainOctree,
        crate::SoilPatch,
        crate::WorldCell,
    ) {
        let field = TerrainField::new(WorldSeed(8));
        let coordinate = crate::BrickCoord::new(0, 100, 0);
        let mut brick = super::TerrainBrick::promote(&field, coordinate);
        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    brick.cells[super::local_index(bevy_math::IVec3::new(x, y, z)).unwrap()] =
                        crate::TerrainSample {
                            density: if y <= 16 {
                                -super::EMPTY_DENSITY
                            } else {
                                super::EMPTY_DENSITY
                            },
                            material,
                            compaction: 0,
                        };
                }
            }
        }
        brick.minimum_density = super::EMPTY_DENSITY;
        brick.maximum_density = -super::EMPTY_DENSITY;
        let mut terrain = TerrainOctree::default();
        terrain.insert_brick(brick);
        let cell = crate::WorldCell::new(8, 3216, 8);
        let patch = crate::SoilPatch {
            centre: crate::WorldPosition(cell.centre().0 + DVec3::Y * 0.025),
            normal: DVec3::Y,
            radius: 0.1,
            pressure_pa: 80_000.0,
            seconds: 0.1,
        };
        (field, terrain, patch, cell)
    }

    #[test]
    fn light_load_on_soil_leaves_the_surface_unchanged() {
        let (field, mut terrain, mut patch, cell) = soil_fixture(TerrainMaterial::Soil);
        let before = terrain.sample_cell(&field, cell);
        patch.pressure_pa = 20_000.0;
        assert_eq!(
            terrain
                .compress_patch(&field, patch)
                .unwrap()
                .total_changed_cells(),
            0
        );
        assert_eq!(terrain.sample_cell(&field, cell), before);
        assert_eq!(terrain.dirty_leaves().count(), 0);
    }

    #[test]
    fn extraction_is_atomic_and_preserves_compressed_material_quantity() {
        let (field, mut terrain, patch, cell) = soil_fixture(TerrainMaterial::Soil);
        terrain.compress_patch(&field, patch).unwrap();
        let source = crate::ExtractionCell {
            cell,
            sample: terrain.sample_cell(&field, cell),
        };
        let clumps = crate::ClumpCollection::default();
        assert!(
            clumps
                .prepare_extraction(&terrain, &field, &[source, source])
                .is_none()
        );
        assert_eq!(terrain.sample_cell(&field, cell), source.sample);
        let transfer = clumps
            .prepare_extraction(&terrain, &field, &[source])
            .unwrap();
        assert!(!transfer.terrain.sample_cell(&field, cell).is_solid());
        assert_eq!(
            u64::from(transfer.clumps.bodies[&1].quanta),
            source.material_quanta()
        );
        assert!(
            transfer
                .clumps
                .prepare_extraction(&transfer.terrain, &field, &[source])
                .is_none()
        );
        assert!(terrain.sample_cell(&field, cell).is_solid());
    }

    #[test]
    fn settled_soft_material_deposits_once_while_rock_stays_physical() {
        for material in [
            TerrainMaterial::Sand,
            TerrainMaterial::Soil,
            TerrainMaterial::Rock,
            TerrainMaterial::Iron,
        ] {
            let (field, terrain, _, cell) = soil_fixture(material);
            let source = crate::ExtractionCell {
                cell,
                sample: terrain.sample_cell(&field, cell),
            };
            let mut transfer = crate::ClumpCollection::default()
                .prepare_extraction(&terrain, &field, &[source])
                .unwrap();
            assert!(
                transfer
                    .clumps
                    .prepare_deposition(&transfer.terrain, &field, 1, &[cell])
                    .is_none()
            );
            transfer
                .clumps
                .bodies
                .get_mut(&1)
                .unwrap()
                .update_settling(true, 1.0);
            let deposited =
                transfer
                    .clumps
                    .prepare_deposition(&transfer.terrain, &field, 1, &[cell, cell]);
            if crate::BreakageResponse::for_material(material).deposits {
                let deposited = deposited.unwrap();
                assert!(deposited.clumps.bodies.is_empty());
                let sample = deposited.terrain.sample_cell(&field, cell);
                assert!(sample.is_solid());
                assert_eq!(sample.material, material);
                assert_eq!(sample.compaction, 0);
            } else {
                assert!(deposited.is_none());
            }
        }
    }

    #[test]
    fn breakage_requires_stress_and_work_at_the_exposed_contact() {
        let (field, terrain, soil, cell) = soil_fixture(TerrainMaterial::Rock);
        let mut damage = crate::BreakageAccumulator::default();
        let mut patch = crate::BreakagePatch {
            centre: soil.centre,
            normal: DVec3::Y,
            radius: 0.025,
            stress_pa: 1e9,
            work_j: 0.0,
        };
        damage.accumulate(&terrain, &field, patch);
        assert!(damage.ready(&terrain, &field, 256).is_empty());
        patch.stress_pa = 1.0;
        patch.work_j = 1e9;
        damage.accumulate(&terrain, &field, patch);
        assert!(damage.ready(&terrain, &field, 256).is_empty());
        patch.stress_pa = 1e9;
        damage.accumulate(&terrain, &field, patch);
        let ready = damage.ready(&terrain, &field, 256);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].cell, cell);
        assert!(damage.ready(&terrain, &field, 0).is_empty());
        assert_eq!(damage.ready(&terrain, &field, 1), ready);
        damage.committed(&ready);
        assert!(damage.ready(&terrain, &field, 1).is_empty());
    }

    #[test]
    fn sustained_overload_sinks_soil_and_then_stops_when_it_compacts() {
        let (field, mut terrain, patch, cell) = soil_fixture(TerrainMaterial::Soil);
        let before = terrain.sample_cell(&field, cell);
        for _ in 0..100 {
            terrain.compress_patch(&field, patch).unwrap();
        }
        let compacted = terrain.sample_cell(&field, cell);
        assert!(before.density - compacted.density > 0.001);
        for _ in 0..100 {
            terrain.compress_patch(&field, patch).unwrap();
        }
        assert_eq!(terrain.sample_cell(&field, cell), compacted);
    }

    #[test]
    fn rock_does_not_deform_under_any_pressure() {
        for material in [
            TerrainMaterial::Rock,
            TerrainMaterial::Iron,
            TerrainMaterial::Graphite,
        ] {
            let (field, mut terrain, mut patch, cell) = soil_fixture(material);
            let before = terrain.sample_cell(&field, cell);
            patch.pressure_pa = f32::MAX;
            let count = terrain.promoted_brick_count();
            assert_eq!(
                terrain
                    .compress_patch(&field, patch)
                    .unwrap()
                    .changed_bricks,
                0
            );
            assert_eq!(terrain.promoted_brick_count(), count);
            assert_eq!(terrain.sample_cell(&field, cell), before);
            assert_eq!(terrain.dirty_leaves().count(), 0);
        }
    }

    #[test]
    fn a_second_pass_over_compacted_soil_sinks_less_than_the_first() {
        let (field, mut terrain, patch, _) = soil_fixture(TerrainMaterial::Soil);
        let first = terrain.compress_patch(&field, patch).unwrap().sunk_metres;
        let second = terrain.compress_patch(&field, patch).unwrap().sunk_metres;
        assert!(first > second && second > 0.0);
    }

    #[test]
    fn a_fully_collapsed_cell_passes_further_sinking_to_the_cell_below() {
        let (field, mut terrain, mut patch, cell) = soil_fixture(TerrainMaterial::Soil);
        patch.pressure_pa = 1.0e9;
        for _ in 0..5 {
            terrain.compress_patch(&field, patch).unwrap();
        }
        assert!(
            terrain.sample_cell(&field, cell).density
                <= super::EMPTY_DENSITY + super::COMPACTION_STEP_METRES
        );
        terrain.compress_patch(&field, patch).unwrap();
        let below = crate::WorldCell::new(cell.x, cell.y - 1, cell.z);
        assert!(terrain.sample_cell(&field, below).compaction > 0);
    }

    #[test]
    fn compacted_brick_rle_round_trips_exactly() {
        let (field, mut terrain, patch, cell) = soil_fixture(TerrainMaterial::Soil);
        terrain.compress_patch(&field, patch).unwrap();
        let brick = terrain.brick(cell.brick()).unwrap();
        assert_eq!(decode_brick(&encode_brick(brick)).unwrap(), *brick);
        assert_eq!(std::mem::size_of::<crate::TerrainSample>(), 8);
        let mut old = encode_brick(brick);
        old[4..6].copy_from_slice(&2_u16.to_le_bytes());
        assert_eq!(
            decode_brick(&old),
            Err(super::BrickDecodeError::UnsupportedHeader)
        );
    }

    #[test]
    fn soil_accumulation_defers_edits_and_discards_stale_loads() {
        let (field, mut terrain, mut patch, cell) = soil_fixture(TerrainMaterial::Soil);
        patch.seconds = 1.0 / 60.0;
        let mut pending = crate::SoilAccumulator::default();
        pending.accumulate(&terrain, &field, patch).unwrap();
        assert!(pending.take_ready().is_empty());
        for _ in 0..15 {
            pending.accumulate(&terrain, &field, patch).unwrap();
        }
        let ready = pending.take_ready();
        assert!(!ready.is_empty());
        assert_eq!(terrain.sample_cell(&field, cell).compaction, 0);
        assert!(terrain.compress_cells(&field, &ready).changed_bricks > 0);
        assert_eq!(terrain.compress_cells(&field, &ready).changed_bricks, 0);
    }

    #[test]
    fn invalid_soil_pressure_does_not_promote_bricks() {
        let field = TerrainField::new(WorldSeed(8));
        let mut terrain = TerrainOctree::default();
        let patch = crate::SoilPatch {
            centre: WorldPosition(DVec3::ZERO),
            normal: DVec3::Y,
            radius: 0.1,
            pressure_pa: f32::NAN,
            seconds: 0.1,
        };
        assert_eq!(
            terrain.compress_patch(&field, patch),
            Err(TerrainEditError::InvalidSoilPatch)
        );
        assert_eq!(terrain.promoted_brick_count(), 0);
    }
    #[test]
    fn compression_lowers_the_meshed_surface_and_survives_reload() {
        let (field, mut terrain, patch, cell) = soil_fixture(TerrainMaterial::Soil);
        let height = |terrain: &TerrainOctree| {
            crate::mesh_chunk(
                &field,
                &terrain.snapshot(),
                crate::TerrainMeshRequest {
                    node: super::TerrainNodeId::leaf(cell.brick()),
                    generation: 1,
                    transition_mask: crate::TerrainTransitionMask::NONE,
                },
            )
            .raycast(
                crate::WorldPosition(cell.centre().0 + DVec3::Y * 0.2),
                -DVec3::Y,
                0.5,
            )
            .unwrap()
            .position
            .0
            .y
        };
        let before = height(&terrain);
        for _ in 0..100 {
            terrain.compress_patch(&field, patch).unwrap();
        }
        let after = height(&terrain);
        assert!(before - after > 0.0005, "{before} -> {after}");
        let mut loaded = TerrainOctree::default();
        loaded.insert_saved_brick(
            decode_brick(&encode_brick(terrain.brick(cell.brick()).unwrap())).unwrap(),
        );
        assert!((height(&loaded) - after).abs() < 1.0e-6);
    }
    #[test]
    fn a_small_soil_load_moves_procedural_surface_continuously_downward() {
        let field = TerrainField::new(WorldSeed(91));
        for x in [0.0, 0.13, 0.27, 0.41] {
            let mut terrain = TerrainOctree::default();
            let centre = WorldPosition(DVec3::new(x, field.surface_height(x, 0.0), 0.0));
            let node = super::TerrainNodeId::leaf(centre.cell().unwrap().brick());
            let height = |terrain: &TerrainOctree| {
                crate::mesh_chunk(
                    &field,
                    &terrain.snapshot(),
                    crate::TerrainMeshRequest {
                        node,
                        generation: 1,
                        transition_mask: crate::TerrainTransitionMask::NONE,
                    },
                )
                .raycast(WorldPosition(centre.0 + DVec3::Y * 0.2), -DVec3::Y, 0.5)
                .unwrap()
                .position
                .0
                .y
            };
            let before = height(&terrain);
            let outcome = terrain
                .compress_patch(
                    &field,
                    crate::SoilPatch {
                        centre,
                        normal: DVec3::Y,
                        radius: 0.15,
                        pressure_pa: 30_000.0,
                        seconds: 1.0 / 60.0,
                    },
                )
                .unwrap();
            assert!(outcome.total_changed_cells() > 0);
            let displacement = before - height(&terrain);
            assert!(
                (0.00001..0.0005).contains(&displacement),
                "x={x}: moved {displacement} m"
            );
        }
    }
}
