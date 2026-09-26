//! Sparse promotion and persistent subtractive terrain edits.

#![expect(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod batch;
mod brick;
mod node;
mod repose;
mod walk;

pub use batch::{
    REMOVED_CELL_CUBIC_METERS, REMOVED_CELL_LITRES, TerrainEditBatch, TerrainEditError,
    TerrainEditOutcome,
};
use batch::{cell_containing, sphere_y_cell_range};
use brick::EMPTY_DENSITY;
pub use brick::{BrickDecodeError, TerrainBrick, decode_brick, encode_brick};
use node::TerrainNode;
pub use node::{TerrainDensityClass, TerrainNodeId, TerrainNodeSummary};
pub(crate) use repose::SLIDING_LOOSENESS;
pub use repose::{Repose, SPOIL_LOOSENESS, SpoilSlump};
use walk::{
    classify_node, collect_bricks, collect_bricks_between, collect_nodes_between, find_brick,
    find_node, insert_brick_node, latest_revision_between, minimum_promoted_density_between,
};

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use bevy_math::DVec3;

use crate::{
    BrickCoord, SoilCompression, SoilPatch, SoilResponse, TERRAIN_CELL_METERS, TerrainField,
    TerrainMaterial, TerrainSample, WORLD_HALF_EXTENT_METERS, WorldCell, WorldPosition,
    soil::COMPACTION_STEP_METRES,
};

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

    /// Whether a cell holds ground, without painting its material.
    pub(crate) fn is_solid_cell(&self, field: &TerrainField, cell: WorldCell) -> bool {
        self.brick(cell.brick())
            .and_then(|brick| brick.sample(cell.local_in_brick()))
            .map_or_else(|| field.cell_density(cell) > 0.0, TerrainSample::is_solid)
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
        // Below the softest ground's strength nothing yields, however wide the load.
        let softest = TerrainMaterial::ALL
            .into_iter()
            .map(|material| SoilResponse::for_material(material).bearing_capacity_pa)
            .fold(f32::INFINITY, f32::min);
        if patch.normal.y < crate::GROUND_NORMAL_MIN_Y || patch.pressure_pa <= softest {
            return Ok(cells);
        }
        // Include a short vertical search below a stale contact mesh. Each column
        // yields only its first exposed sample, never the whole supporting volume.
        let reach = patch.footprint.reach();
        let extent = DVec3::new(reach, reach + TERRAIN_CELL_METERS * 2.0, reach);
        let minimum = cell_containing(patch.centre.0 - extent);
        let maximum = cell_containing(patch.centre.0 + extent);
        for z in minimum.z..=maximum.z {
            for x in minimum.x..=maximum.x {
                for y in (minimum.y..=maximum.y).rev() {
                    let cell = WorldCell::new(x, y, z);
                    let promoted = self
                        .brick(cell.brick())
                        .and_then(|brick| brick.sample(cell.local_in_brick()));
                    // Untouched ground is never compacted, so only its density
                    // decides whether to paint it.
                    if promoted.is_none()
                        && field.cell_density(cell) <= EMPTY_DENSITY + COMPACTION_STEP_METRES
                    {
                        continue;
                    }
                    let sample = promoted.unwrap_or_else(|| field.sample_cell(cell));
                    if sample.density <= EMPTY_DENSITY + COMPACTION_STEP_METRES
                        || (!sample.is_solid() && sample.compaction == 0)
                    {
                        continue;
                    }
                    let offset = cell.centre().0 - patch.centre.0;
                    if patch.footprint.contains(patch.normal, offset) {
                        let depth = SoilResponse::for_material(sample.material).depth(
                            sample,
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
                    // Flat the moment it stops being ground. It goes on sinking
                    // after that, but its material left once.
                    if cell.sample.is_solid()
                        && brick
                            .sample(cell.cell.local_in_brick())
                            .is_some_and(|pressed| !pressed.is_solid())
                    {
                        let quanta = crate::CELL_QUANTA - u32::from(cell.sample.looseness);
                        outcome.quanta_given_up += u64::from(quanta);
                        outcome
                            .pressed_out
                            .push((cell.cell, cell.sample.material, quanta));
                    }
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
        let mut outcome = TerrainEditOutcome {
            quanta_given_up: cells.iter().map(|source| source.material_quanta()).sum(),
            ..Default::default()
        };
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
                        if !self.is_solid_cell(field, cell) {
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
                        if self.is_solid_cell(field, cell) {
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
                if brick.set_solid(cell.local_in_brick(), material, *density, 0) {
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

#[cfg(test)]
mod tests;
