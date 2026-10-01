//! Sediment running water takes from the ground and lays back on it.
//!
//! A cell gives its material up a few quanta at a time: it loosens, and its
//! surface sinks with what it holds, until at half full it empties and gives
//! the rest. Laid sediment grows a loose cell the same way, and begins a new
//! one above it once that is as full as laid ground gets. So the bed lowers
//! and rises smoothly rather than a cell at a time, and the quanta taken and
//! laid are exactly those the ground lost and gained.

use std::collections::BTreeMap;

use super::{EMPTY_DENSITY, SLIDING_LOOSENESS, TerrainBrick, TerrainNodeId, TerrainOctree};
use crate::{
    BreakageResponse, BrickCoord, CELL_QUANTA, TERRAIN_CELL_METERS, TerrainEditOutcome,
    TerrainField, TerrainMaterial, TerrainSample, WorldCell,
};

/// Least a solid cell holds, in quanta: its looseness is a byte.
const LEAST_QUANTA: u32 = CELL_QUANTA - u8::MAX as u32;

/// Most laid sediment holds, in quanta: it stays loose.
const LAID_QUANTA: u32 = CELL_QUANTA - SLIDING_LOOSENESS as u32;

/// Density of a full cell, whose surface lies at its top.
const FULL_DENSITY: f32 = -EMPTY_DENSITY;

/// Density of a cell about to empty or just begun, whose surface lies near
/// its middle.
const THIN_DENSITY: f32 = 0.001;

/// Most quanta one cell gives or takes before the next column's turn, so a
/// column's cells sink and rise together.
const STEP_QUANTA: u32 = 8;

/// Cells above and below the given height searched for a column's top.
const SEARCH_CELLS: i32 = 6;

/// Sediment to take from or lay on the ground's top across a square of
/// terrain-cell columns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SedimentChange {
    /// The square's lowest column along x, in terrain cells.
    pub x: i32,
    /// The square's lowest column along z, in terrain cells.
    pub z: i32,
    /// Columns along each side of the square.
    pub edge: i32,
    /// Height near which the ground's top lies, in metres: the top within
    /// 30 cm of it is the one changed, not a cave's floor below.
    pub height: f64,
    /// Quanta to lay, or to take where negative.
    pub quanta: i64,
    /// Material to lay; what is taken is whatever soft ground is there.
    pub material: TerrainMaterial,
}

/// What one [`SedimentChange`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SedimentApplied {
    /// Quanta taken, by material code. Emptying a cell gives all it held,
    /// so this may exceed what was asked.
    pub taken: [u64; TerrainMaterial::COUNT],
    /// Quanta laid.
    pub laid: u64,
}

impl SedimentApplied {
    /// Quanta taken of every material.
    pub fn total_taken(&self) -> u64 {
        self.taken.iter().sum()
    }
}

/// How full a cell's surface shows it to be: the density of a cell holding
/// `quanta`, from its middle at the least to its top when full.
fn density_holding(quanta: u32) -> f32 {
    let full = (quanta.clamp(LEAST_QUANTA, CELL_QUANTA) - LEAST_QUANTA) as f32
        / (CELL_QUANTA - LEAST_QUANTA) as f32;
    THIN_DENSITY + (FULL_DENSITY - THIN_DENSITY) * full
}

/// Quanta a solid cell holds.
fn quanta_of(sample: TerrainSample) -> u32 {
    CELL_QUANTA - u32::from(sample.looseness)
}

/// Looseness of a cell holding `quanta`.
fn looseness_holding(quanta: u32) -> u8 {
    u8::try_from(CELL_QUANTA - quanta).expect("a solid cell holds at least half")
}

/// Bricks being changed, read through to the terrain where untouched.
struct Working<'a> {
    terrain: &'a TerrainOctree,
    field: &'a TerrainField,
    bricks: BTreeMap<BrickCoord, TerrainBrick>,
}

impl Working<'_> {
    fn sample(&self, cell: WorldCell) -> TerrainSample {
        self.bricks
            .get(&cell.brick())
            .and_then(|brick| brick.sample(cell.local_in_brick()))
            .unwrap_or_else(|| self.terrain.sample_cell(self.field, cell))
    }

    fn brick(&mut self, cell: WorldCell) -> &mut TerrainBrick {
        let coordinate = cell.brick();
        self.bricks.entry(coordinate).or_insert_with(|| {
            self.terrain
                .brick(coordinate)
                .cloned()
                .unwrap_or_else(|| TerrainBrick::promote(self.field, coordinate))
        })
    }

    /// The top solid cell of a column with open ground over it, near `from`.
    fn top(&self, x: i32, z: i32, from: i32) -> Option<WorldCell> {
        let mut above = self.sample(WorldCell::new(x, from + SEARCH_CELLS + 1, z));
        for y in (from - SEARCH_CELLS..=from + SEARCH_CELLS).rev() {
            let cell = WorldCell::new(x, y, z);
            let sample = self.sample(cell);
            if sample.is_solid() && !above.is_solid() {
                return Some(cell);
            }
            above = sample;
        }
        None
    }

    /// Where a column's drawn surface lies, in cells: its top cell's middle
    /// plus how far its density reaches above that.
    fn surface(&self, cell: WorldCell) -> f64 {
        f64::from(cell.y) + 0.5 + f64::from(self.sample(cell).density) / TERRAIN_CELL_METERS
    }
}

impl TerrainOctree {
    /// Takes sediment from and lays it on the ground's top, as running water
    /// asks: each change is worked out against the ground as it is now, so
    /// a change asked of an older snapshot does what still can be done.
    /// Only soft ground gives sediment up, and the highest column of a
    /// square gives first and the lowest takes first, so the bed evens out.
    pub fn exchange_sediment(
        &mut self,
        field: &TerrainField,
        changes: &[SedimentChange],
    ) -> (TerrainEditOutcome, Vec<SedimentApplied>) {
        let mut outcome = TerrainEditOutcome::default();
        let mut applied = Vec::with_capacity(changes.len());
        let mut working = Working {
            terrain: self,
            field,
            bricks: BTreeMap::new(),
        };
        for change in changes {
            let done = if change.quanta < 0 {
                take(&mut working, change, &mut outcome)
            } else {
                lay(&mut working, change, &mut outcome)
            };
            applied.push(done);
        }
        let bricks = std::mem::take(&mut working.bricks);
        for (coordinate, mut brick) in bricks {
            brick.revision = self.next_revision;
            self.insert_brick(brick);
            self.dirty.insert(TerrainNodeId::leaf(coordinate));
            outcome.changed_brick_coordinates.push(coordinate);
        }
        outcome.changed_bricks = outcome.changed_brick_coordinates.len();
        if outcome.changed_bricks > 0 {
            self.next_revision = self.next_revision.wrapping_add(1).max(1);
        }
        (outcome, applied)
    }
}

/// The columns of a change's square.
fn columns(change: &SedimentChange) -> impl Iterator<Item = (i32, i32)> {
    let (x, z, edge) = (change.x, change.z, change.edge.max(0));
    (0..edge).flat_map(move |dz| (0..edge).map(move |dx| (x + dx, z + dz)))
}

/// The cell height a change's height lies in.
#[expect(clippy::cast_possible_truncation, reason = "heights in the world")]
fn height_cell(change: &SedimentChange) -> i32 {
    (change.height / TERRAIN_CELL_METERS).floor() as i32
}

fn take(
    working: &mut Working<'_>,
    change: &SedimentChange,
    outcome: &mut TerrainEditOutcome,
) -> SedimentApplied {
    let mut done = SedimentApplied::default();
    let mut owed = change.quanta.unsigned_abs();
    let from = height_cell(change);
    let mut tops = columns(change)
        .map(|(x, z)| working.top(x, z, from))
        .collect::<Vec<_>>();
    while owed > 0 {
        let soft = |cell: WorldCell| {
            cell.is_editable() && BreakageResponse::for_material(working.sample(cell).material).soft
        };
        let Some((index, cell)) = tops
            .iter()
            .enumerate()
            .filter_map(|(index, top)| top.filter(|&cell| soft(cell)).map(|cell| (index, cell)))
            .max_by(|a, b| working.surface(a.1).total_cmp(&working.surface(b.1)))
        else {
            break;
        };
        let sample = working.sample(cell);
        let held = quanta_of(sample);
        let code = sample.material.code() as usize;
        if held > LEAST_QUANTA {
            let given = u32::try_from(owed)
                .unwrap_or(u32::MAX)
                .min(held - LEAST_QUANTA)
                .min(STEP_QUANTA);
            let left = held - given;
            let density = sample.density.min(density_holding(left));
            working
                .brick(cell)
                .reshape(cell.local_in_brick(), looseness_holding(left), density);
            done.taken[code] += u64::from(given);
            owed -= u64::from(given);
        } else {
            working.brick(cell).set_empty(cell.local_in_brick());
            outcome.removed_cells[code] += 1;
            done.taken[code] += u64::from(held);
            owed = owed.saturating_sub(u64::from(held));
            let below = WorldCell::new(cell.x, cell.y - 1, cell.z);
            tops[index] = working.sample(below).is_solid().then_some(below);
        }
        if outcome.sediment_cells.last() != Some(&cell) {
            outcome.sediment_cells.push(cell);
        }
    }
    outcome.quanta_given_up += done.total_taken();
    done
}

fn lay(
    working: &mut Working<'_>,
    change: &SedimentChange,
    outcome: &mut TerrainEditOutcome,
) -> SedimentApplied {
    let mut done = SedimentApplied::default();
    let material = change.material;
    if !BreakageResponse::for_material(material).soft {
        return done;
    }
    let mut owed = u64::try_from(change.quanta).unwrap_or(0);
    let from = height_cell(change);
    let mut tops = columns(change)
        .map(|(x, z)| working.top(x, z, from))
        .collect::<Vec<_>>();
    while owed > 0 {
        // A loose top of the same material fills; any other top takes a new
        // cell over it, if enough is owed to begin one.
        let fills = |cell: WorldCell| {
            let sample = working.sample(cell);
            sample.material == material
                && sample.looseness >= SLIDING_LOOSENESS
                && quanta_of(sample) < LAID_QUANTA
        };
        let begins = |cell: WorldCell| {
            let over = WorldCell::new(cell.x, cell.y + 1, cell.z);
            owed >= u64::from(LEAST_QUANTA)
                && over.is_editable()
                && over.centre().is_inside_world()
                && !working.sample(over).is_solid()
        };
        let Some((index, cell)) = tops
            .iter()
            .enumerate()
            .filter_map(|(index, top)| top.map(|cell| (index, cell)))
            .filter(|&(_, cell)| cell.is_editable() && (fills(cell) || begins(cell)))
            .min_by(|a, b| working.surface(a.1).total_cmp(&working.surface(b.1)))
        else {
            break;
        };
        if fills(cell) {
            let sample = working.sample(cell);
            let held = quanta_of(sample);
            let given = u32::try_from(owed)
                .unwrap_or(u32::MAX)
                .min(LAID_QUANTA - held)
                .min(STEP_QUANTA);
            let now = held + given;
            let density = sample.density.max(density_holding(now)).min(FULL_DENSITY);
            working
                .brick(cell)
                .reshape(cell.local_in_brick(), looseness_holding(now), density);
            done.laid += u64::from(given);
            owed -= u64::from(given);
            if outcome.sediment_cells.last() != Some(&cell) {
                outcome.sediment_cells.push(cell);
            }
        } else {
            let over = WorldCell::new(cell.x, cell.y + 1, cell.z);
            working.brick(over).set_solid(
                over.local_in_brick(),
                material,
                density_holding(LEAST_QUANTA),
                looseness_holding(LEAST_QUANTA),
            );
            outcome.added_cells[material.code() as usize] += 1;
            done.laid += u64::from(LEAST_QUANTA);
            owed -= u64::from(LEAST_QUANTA);
            tops[index] = Some(over);
            outcome.sediment_cells.push(over);
        }
    }
    outcome.quanta_taken_back += done.laid;
    done
}

#[cfg(test)]
mod tests {
    use bevy_math::IVec3;

    use super::{LAID_QUANTA, LEAST_QUANTA, SedimentChange, quanta_of};
    use crate::{
        BrickCoord, CELL_QUANTA, SurfaceId, TerrainField, TerrainMaterial, TerrainOctree,
        TerrainSample, WorldCell, WorldSeed,
        edits::{EMPTY_DENSITY, TerrainBrick, brick::local_index},
    };

    /// Flat ground of one material high in the sky, whose top cells are at
    /// y 32016, its surface at 1600.85 m.
    fn flat(material: TerrainMaterial) -> (TerrainField, TerrainOctree) {
        let field = TerrainField::new(WorldSeed(8));
        let coordinate = BrickCoord::new(0, 1000, 0);
        let mut brick = TerrainBrick::promote(&field, coordinate);
        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    brick.cells[local_index(IVec3::new(x, y, z)).unwrap()] = TerrainSample {
                        density: if y <= 16 {
                            -EMPTY_DENSITY
                        } else {
                            EMPTY_DENSITY
                        },
                        material,
                        surface: SurfaceId::plain(material),
                        compaction: 0,
                        looseness: 0,
                    };
                }
            }
        }
        brick.minimum_density = EMPTY_DENSITY;
        brick.maximum_density = -EMPTY_DENSITY;
        let mut terrain = TerrainOctree::default();
        terrain.insert_brick(brick);
        (field, terrain)
    }

    fn change(quanta: i64, material: TerrainMaterial) -> SedimentChange {
        SedimentChange {
            x: 8,
            z: 8,
            edge: 4,
            height: 1600.85,
            quanta,
            material,
        }
    }

    /// Quanta held in the square's columns, and its cells' densities.
    fn held(terrain: &TerrainOctree, field: &TerrainField) -> (u64, Vec<f32>) {
        let mut total = 0;
        let mut densities = Vec::new();
        for z in 8..12 {
            for x in 8..12 {
                for y in 32000..32031 {
                    let sample = terrain.sample_cell(field, WorldCell::new(x, y, z));
                    if sample.is_solid() {
                        total += u64::from(quanta_of(sample));
                        if y >= 32016 {
                            densities.push(sample.density);
                        }
                    }
                }
            }
        }
        (total, densities)
    }

    #[test]
    fn a_cell_shrinks_smoothly_then_empties_and_gives_up_all_it_held() {
        let (field, mut terrain) = flat(TerrainMaterial::Soil);
        let (before, _) = held(&terrain, &field);
        let top = WorldCell::new(8, 32016, 8);
        let mut last = terrain.sample_cell(&field, top).density;
        let mut taken = 0;
        // A little at a time across the square, as water takes it, until
        // the top layer is gone.
        for _ in 0..1_000 {
            if held(&terrain, &field).1.is_empty() {
                break;
            }
            let (outcome, applied) =
                terrain.exchange_sediment(&field, &[change(-40, TerrainMaterial::Soil)]);
            taken += applied[0].taken[TerrainMaterial::Soil.code() as usize];
            assert_eq!(outcome.quanta_given_up, applied[0].total_taken());
            let now = terrain.sample_cell(&field, top);
            if now.is_solid() {
                assert!(now.density <= last, "the surface rose as it gave");
                last = now.density;
            }
        }
        let (after, densities) = held(&terrain, &field);
        assert_eq!(before - after, taken, "material made or lost");
        // Every top cell emptied before any below gave: the square sank
        // evenly.
        assert!(densities.is_empty(), "{} top cells left", densities.len());
        assert_eq!(taken, 16 * u64::from(CELL_QUANTA));
    }

    #[test]
    fn laid_sediment_grows_a_loose_cell_and_starts_another_above_it() {
        let (field, mut terrain) = flat(TerrainMaterial::Soil);
        let (before, _) = held(&terrain, &field);
        // Too little to begin a cell lays nothing on undisturbed ground.
        let (_, applied) = terrain.exchange_sediment(&field, &[change(100, TerrainMaterial::Sand)]);
        assert_eq!(applied[0].laid, 0);
        let mut laid = 0;
        for _ in 0..40 {
            let (outcome, applied) =
                terrain.exchange_sediment(&field, &[change(300, TerrainMaterial::Sand)]);
            laid += applied[0].laid;
            assert_eq!(outcome.quanta_taken_back, applied[0].laid);
        }
        let (after, _) = held(&terrain, &field);
        assert_eq!(after - before, laid, "material made or lost");
        // Too little to begin a cell over full ones waits for more.
        assert!(laid > 40 * 300 * 9 / 10, "laid only {laid}");
        // Each column's first cell filled as far as laid ground goes before
        // a second began over it.
        let first = terrain.sample_cell(&field, WorldCell::new(8, 32017, 8));
        let second = terrain.sample_cell(&field, WorldCell::new(8, 32018, 8));
        assert_eq!(first.material, TerrainMaterial::Sand);
        assert_eq!(quanta_of(first), LAID_QUANTA);
        assert!(second.is_solid() && quanta_of(second) >= LEAST_QUANTA);
        assert!(second.density <= first.density);
    }

    #[test]
    fn rock_gives_nothing_and_a_change_finds_only_the_ground_near_it() {
        let (field, mut terrain) = flat(TerrainMaterial::Rock);
        let (outcome, applied) =
            terrain.exchange_sediment(&field, &[change(-500, TerrainMaterial::Soil)]);
        assert_eq!(applied[0].total_taken(), 0);
        assert_eq!(outcome.changed_bricks, 0);
        let (field, mut terrain) = flat(TerrainMaterial::Soil);
        let (before, _) = held(&terrain, &field);
        // Ground a metre above the height asked for is not this change's.
        let far = SedimentChange {
            height: 1599.85,
            ..change(-500, TerrainMaterial::Soil)
        };
        let (_, applied) = terrain.exchange_sediment(&field, &[far]);
        assert_eq!(applied[0].total_taken(), 0);
        assert_eq!(held(&terrain, &field).0, before);
    }
}
