//! How loose ground lies: it runs downhill to its angle of repose when it is
//! laid, and slides when it is undercut. Undisturbed ground stands as it is.
//! Game tuning, not measured soil data.

use std::collections::{BTreeMap, BTreeSet};

use super::{EMPTY_DENSITY, TerrainBrick, TerrainEditOutcome, TerrainNodeId, TerrainOctree};
use crate::{
    BrickCoord, CELL_QUANTA, ExtractionCell, TerrainField, TerrainMaterial, TerrainSample,
    WorldCell,
};

/// Looseness of spoil as it is laid: a quarter more room than the ground it
/// was dug from, as excavated soil takes.
pub const SPOIL_LOOSENESS: u8 = 102;
/// Ground at least this loose slides when it is left too steep.
const SLIDING_LOOSENESS: u8 = 51;
/// How far down a column is searched for its floor, in cells.
const FLOOR_SEARCH_CELLS: i32 = 40;
/// Columns a laid cell may run across before it stays where it is.
const RUN_COLUMNS: usize = 48;
/// How far around a changed cell loose ground is looked over again, in cells.
const DISTURBED_REACH: i32 = 3;

const DIRECTIONS: [(i32, i32); 8] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
];

/// The steepest a loose material lies: for each distance in cells, how many
/// cells lower the ground there may be.
#[derive(Clone, Copy, Debug)]
pub struct Repose(&'static [(i32, i32)]);

impl Repose {
    /// The slope of a material that settles back into ground, about 34° for
    /// soil and cover and 27° for sand. Minerals do not settle.
    pub const fn for_material(material: TerrainMaterial) -> Option<Self> {
        match material {
            TerrainMaterial::Soil | TerrainMaterial::SurfaceCover => Some(Self(&[(1, 1), (3, 2)])),
            TerrainMaterial::Sand => Some(Self(&[(1, 1), (2, 1)])),
            TerrainMaterial::Rock | TerrainMaterial::Iron | TerrainMaterial::Graphite => None,
        }
    }
}

/// Terrain with cells laid on it that are not committed yet.
struct Laying<'a> {
    terrain: &'a TerrainOctree,
    field: &'a TerrainField,
    bricks: BTreeMap<BrickCoord, TerrainBrick>,
}

impl Laying<'_> {
    fn sample(&self, cell: WorldCell) -> TerrainSample {
        self.bricks
            .get(&cell.brick())
            .and_then(|brick| brick.sample(cell.local_in_brick()))
            .unwrap_or_else(|| self.terrain.sample_cell(self.field, cell))
    }

    fn solid(&self, cell: WorldCell) -> bool {
        self.sample(cell).is_solid()
    }
}

// The free cell resting on ground in a column, at or below `from`. None when
// the column holds a machine, or no ground within reach: nothing is laid on a
// deck, which would leave it hanging when the deck moves.
fn floor(
    solid: &impl Fn(WorldCell) -> bool,
    occupied: &mut dyn FnMut(WorldCell) -> bool,
    from: WorldCell,
) -> Option<WorldCell> {
    let mut cell = from;
    for _ in 0..FLOOR_SEARCH_CELLS {
        if solid(cell) || !cell.is_editable() {
            return None;
        }
        let below = WorldCell::new(cell.x, cell.y - 1, cell.z);
        if solid(below) {
            return (!occupied(cell)).then_some(cell);
        }
        if occupied(below) {
            return None;
        }
        cell = below;
    }
    None
}

// Directions in which loose ground at `cell` would run: the next column is
// free beside it, and nothing within the repose holds it up. A steel block
// holds spoil up as a wall of ground does.
fn runs(
    solid: &impl Fn(WorldCell) -> bool,
    occupied: &mut dyn FnMut(WorldCell) -> bool,
    cell: WorldCell,
    repose: Repose,
) -> Vec<(i32, i32)> {
    let mut held = |cell| solid(cell) || occupied(cell);
    DIRECTIONS
        .into_iter()
        .filter(|&(x, z)| {
            if held(WorldCell::new(cell.x + x, cell.y, cell.z + z)) {
                return false;
            }
            repose.0.iter().any(|&(distance, drop)| {
                !(0..=drop).any(|lower| {
                    held(WorldCell::new(
                        cell.x + x * distance,
                        cell.y - lower,
                        cell.z + z * distance,
                    ))
                })
            })
        })
        .collect()
}

impl TerrainOctree {
    /// Lays loose material on the ground at or below `start`, each cell running
    /// downhill until it lies within its repose. Returns the edit and what was
    /// not laid: nothing, or enough to lay later. `occupied` names cells a
    /// machine fills, and `steps` is the running allowed, shared by the caller's
    /// other work and spent here.
    pub fn lay_spoil(
        &mut self,
        field: &TerrainField,
        start: WorldCell,
        material: TerrainMaterial,
        quanta: u32,
        occupied: &mut dyn FnMut(WorldCell) -> bool,
        steps: &mut usize,
    ) -> (TerrainEditOutcome, u32) {
        let mut outcome = TerrainEditOutcome::default();
        let Some(repose) = Repose::for_material(material) else {
            return (outcome, quanta);
        };
        if quanta < CELL_QUANTA - u32::from(u8::MAX) {
            return (outcome, quanta);
        }
        // Spoil takes more room than the ground it came from. Shared evenly,
        // every cell holds between half a cell and a whole one, and the last
        // quantum is laid: nothing is left over as a crumb.
        let loose = CELL_QUANTA - u32::from(SPOIL_LOOSENESS);
        let cells = ((quanta + loose / 2) / loose)
            .max(quanta.div_ceil(CELL_QUANTA))
            .max(1);
        let mut laying = Laying {
            terrain: self,
            field,
            bricks: BTreeMap::new(),
        };
        let mut left = quanta;
        for index in 0..cells {
            let share = quanta / cells + u32::from(index < quanta % cells);
            let Some(cell) = run_downhill(&laying, occupied, start, repose, steps) else {
                break;
            };
            let coordinate = cell.brick();
            let brick = laying.bricks.entry(coordinate).or_insert_with(|| {
                laying
                    .terrain
                    .brick(coordinate)
                    .cloned()
                    .unwrap_or_else(|| TerrainBrick::promote(field, coordinate))
            });
            let looseness = u8::try_from(CELL_QUANTA - share).unwrap_or(u8::MAX);
            if !brick.set_solid(cell.local_in_brick(), material, -EMPTY_DENSITY, looseness) {
                break;
            }
            left -= share;
            outcome.quanta_taken_back += u64::from(share);
            outcome.added_cells[material.code() as usize] += 1;
            outcome.laid_cells.push(cell);
        }
        let bricks = laying.bricks;
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
        (outcome, left)
    }
}

// Where one cell of spoil dropped at `start` comes to lie.
fn run_downhill(
    laying: &Laying<'_>,
    occupied: &mut dyn FnMut(WorldCell) -> bool,
    start: WorldCell,
    repose: Repose,
    steps: &mut usize,
) -> Option<WorldCell> {
    let solid = |cell| laying.solid(cell);
    // A clod lies with its centre above the ground, or just inside a heap.
    let mut cell = (0..4)
        .map(|up| WorldCell::new(start.x, start.y + up, start.z))
        .find(|&cell| !solid(cell))
        .and_then(|free| floor(&solid, occupied, free))?;
    let mut crossed = BTreeSet::from([(cell.x, cell.z)]);
    while crossed.len() <= RUN_COLUMNS && *steps > 0 {
        *steps -= 1;
        // Of the ways it would run, the one that takes it lowest.
        let next = runs(&solid, occupied, cell, repose)
            .into_iter()
            .filter(|&(x, z)| !crossed.contains(&(cell.x + x, cell.z + z)))
            .filter_map(|(x, z)| {
                floor(
                    &solid,
                    occupied,
                    WorldCell::new(cell.x + x, cell.y, cell.z + z),
                )
            })
            .min_by_key(|lower| lower.y);
        let Some(next) = next else {
            break;
        };
        crossed.insert((next.x, next.z));
        cell = next;
    }
    Some(cell)
}

/// Loose ground to look over again: columns beside cells that changed.
#[derive(Clone, Debug, Default)]
pub struct SpoilSlump {
    // Column to the highest cell worth looking at and how far below it to look.
    columns: BTreeMap<(i32, i32), (i32, i32)>,
}

impl SpoilSlump {
    /// Notes that ground changed at a cell: loose ground around it may have
    /// lost what held it up.
    pub fn disturb(&mut self, cell: WorldCell) {
        self.disturb_span(cell, 0);
    }

    /// Notes a change reaching `depth` cells below `top`, as a brush stroke does.
    pub fn disturb_span(&mut self, top: WorldCell, depth: i32) {
        for z in -DISTURBED_REACH..=DISTURBED_REACH {
            for x in -DISTURBED_REACH..=DISTURBED_REACH {
                let entry = self
                    .columns
                    .entry((top.x + x, top.z + z))
                    .or_insert((top.y, depth));
                let bottom = (entry.0 - entry.1).min(top.y - depth);
                entry.0 = entry.0.max(top.y);
                entry.1 = entry.0 - bottom;
            }
        }
    }

    /// Notes a change throughout a sphere, as a brush stroke makes.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a brush radius is a few metres"
    )]
    pub fn disturb_sphere(&mut self, centre: crate::WorldPosition, radius_metres: f64) {
        let Ok(cell) = centre.cell() else {
            return;
        };
        if !(radius_metres.is_finite() && (0.0..=64.0).contains(&radius_metres)) {
            return;
        }
        let reach = (radius_metres / crate::TERRAIN_CELL_METERS).ceil() as i32;
        let stride = usize::try_from(DISTURBED_REACH * 2 + 1).unwrap_or(1);
        for z in (-reach..=reach).step_by(stride) {
            for x in (-reach..=reach).step_by(stride) {
                self.disturb_span(
                    WorldCell::new(cell.x + x, cell.y + reach, cell.z + z),
                    reach * 2,
                );
            }
        }
    }

    /// Whether any column waits to be looked over.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Looks over up to `columns` noted columns and returns the loose cells at
    /// their tops that lie steeper than their repose, for the caller to break
    /// out and lay again.
    pub fn take_unstable(
        &mut self,
        terrain: &TerrainOctree,
        field: &TerrainField,
        occupied: &mut dyn FnMut(WorldCell) -> bool,
        columns: usize,
    ) -> Vec<ExtractionCell> {
        let solid = |cell| terrain.sample_cell(field, cell).is_solid();
        let mut unstable = Vec::new();
        for _ in 0..columns {
            let Some(((x, z), (top, depth))) = self.columns.pop_first() else {
                break;
            };
            // The top of the column's ground near where it changed.
            let Some(cell) = (top - depth - DISTURBED_REACH - 1..=top + DISTURBED_REACH + 1)
                .rev()
                .map(|y| WorldCell::new(x, y, z))
                .find(|&cell| solid(cell) && !solid(WorldCell::new(x, cell.y + 1, z)))
            else {
                continue;
            };
            let sample = terrain.sample_cell(field, cell);
            let Some(repose) = Repose::for_material(sample.material) else {
                continue;
            };
            // It slides only where it has somewhere to slide to.
            if sample.looseness >= SLIDING_LOOSENESS
                && cell.is_editable()
                && runs(&solid, occupied, cell, repose)
                    .into_iter()
                    .any(|(x, z)| {
                        floor(
                            &solid,
                            occupied,
                            WorldCell::new(cell.x + x, cell.y, cell.z + z),
                        )
                        .is_some()
                    })
            {
                unstable.push(ExtractionCell {
                    cell,
                    sample,
                    throw: bevy_math::DVec3::ZERO,
                });
            }
        }
        unstable
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ClumpCollection, TransferLimits, WorldSeed};
    use bevy_math::IVec3;

    const FLOOR: i32 = 8;

    // One brick high in the air: a slab of undisturbed ground up to `FLOOR`,
    // with `natural` cells standing on it.
    fn slab(
        material: TerrainMaterial,
        natural: &[IVec3],
    ) -> (TerrainField, TerrainOctree, WorldCell) {
        let field = TerrainField::new(WorldSeed(8));
        let coordinate = BrickCoord::new(0, 100, 0);
        let mut brick = TerrainBrick::promote(&field, coordinate);
        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    let local = IVec3::new(x, y, z);
                    let solid = y <= FLOOR || natural.contains(&local);
                    brick.cells[super::super::brick::local_index(local).unwrap()] = TerrainSample {
                        density: if solid { -EMPTY_DENSITY } else { EMPTY_DENSITY },
                        material,
                        compaction: 0,
                        looseness: 0,
                    };
                }
            }
        }
        brick.minimum_density = EMPTY_DENSITY;
        brick.maximum_density = -EMPTY_DENSITY;
        let mut terrain = TerrainOctree::default();
        terrain.insert_saved_brick(brick);
        (field, terrain, coordinate.minimum_cell())
    }

    fn at(origin: WorldCell, x: i32, y: i32, z: i32) -> WorldCell {
        WorldCell::new(origin.x + x, origin.y + y, origin.z + z)
    }

    // Height of the ground above the slab in a column, in cells.
    fn height(
        terrain: &TerrainOctree,
        field: &TerrainField,
        origin: WorldCell,
        x: i32,
        z: i32,
    ) -> i32 {
        (FLOOR + 1..32)
            .take_while(|&y| terrain.sample_cell(field, at(origin, x, y, z)).is_solid())
            .last()
            .map_or(0, |top| top - FLOOR)
    }

    fn pour(
        terrain: &mut TerrainOctree,
        field: &TerrainField,
        from: WorldCell,
        material: TerrainMaterial,
        cells: u32,
        occupied: &mut dyn FnMut(WorldCell) -> bool,
    ) -> Vec<WorldCell> {
        let mut laid = Vec::new();
        for _ in 0..cells {
            let mut steps = 10_000;
            let (outcome, left) =
                terrain.lay_spoil(field, from, material, CELL_QUANTA * 2, occupied, &mut steps);
            assert_eq!(left, 0);
            laid.extend(outcome.laid_cells);
        }
        laid
    }

    // The steepest drop between columns `distance` apart, over the brick's middle.
    fn steepest(
        terrain: &TerrainOctree,
        field: &TerrainField,
        origin: WorldCell,
        distance: i32,
    ) -> i32 {
        let mut steepest = 0;
        for z in 4..28 {
            for x in 4..28 {
                for (dx, dz) in DIRECTIONS {
                    let drop = height(terrain, field, origin, x, z)
                        - height(terrain, field, origin, x + dx * distance, z + dz * distance);
                    steepest = steepest.max(drop);
                }
            }
        }
        steepest
    }

    #[test]
    fn spoil_poured_on_one_spot_heaps_no_steeper_than_its_repose() {
        let (field, mut terrain, origin) = slab(TerrainMaterial::Soil, &[]);
        pour(
            &mut terrain,
            &field,
            at(origin, 16, 20, 16),
            TerrainMaterial::Soil,
            60,
            &mut |_| false,
        );
        assert!(height(&terrain, &field, origin, 16, 16) >= 3, "no heap");
        assert!(steepest(&terrain, &field, origin, 1) <= 1);
        assert!(steepest(&terrain, &field, origin, 3) <= 2);
    }

    #[test]
    fn sand_heaps_flatter_than_soil() {
        let heap = |material| {
            let (field, mut terrain, origin) = slab(material, &[]);
            pour(
                &mut terrain,
                &field,
                at(origin, 16, 20, 16),
                material,
                60,
                &mut |_| false,
            );
            height(&terrain, &field, origin, 16, 16)
        };
        assert!(heap(TerrainMaterial::Sand) < heap(TerrainMaterial::Soil));
    }

    #[test]
    fn spoil_landing_on_a_wall_runs_down_it() {
        let wall = (FLOOR + 1..FLOOR + 9)
            .map(|y| IVec3::new(16, y, 16))
            .collect::<Vec<_>>();
        let (field, mut terrain, origin) = slab(TerrainMaterial::Soil, &wall);
        let laid = pour(
            &mut terrain,
            &field,
            at(origin, 16, 20, 16),
            TerrainMaterial::Soil,
            4,
            &mut |_| false,
        );
        assert!(!laid.is_empty());
        assert!(
            laid.iter()
                .all(|cell| (cell.x, cell.z) != (origin.x + 16, origin.z + 16))
        );
        assert!(
            laid.iter().all(|cell| cell.y - origin.y <= FLOOR + 3),
            "the wall grew"
        );
    }

    #[test]
    fn a_steel_block_holds_spoil_up_like_a_wall() {
        let (field, mut terrain, origin) = slab(TerrainMaterial::Soil, &[]);
        let block = origin.x + 18;
        let laid = pour(
            &mut terrain,
            &field,
            at(origin, 16, 20, 16),
            TerrainMaterial::Soil,
            60,
            &mut |cell| cell.x >= block,
        );
        assert!(
            laid.iter().all(|cell| cell.x < block),
            "spoil was laid inside the block"
        );
        // Against the block spoil stands higher than on the open side.
        assert!(
            height(&terrain, &field, origin, 17, 16) > height(&terrain, &field, origin, 13, 16)
        );
    }

    #[test]
    fn spoil_takes_more_room_than_the_hole_it_left_and_gives_back_what_it_holds() {
        let (field, mut terrain, origin) = slab(TerrainMaterial::Soil, &[]);
        let hole = (0..8)
            .map(|index| {
                let cell = at(
                    origin,
                    15 + index % 2,
                    FLOOR - index / 2 % 2,
                    15 + index / 4,
                );
                ExtractionCell {
                    cell,
                    sample: terrain.sample_cell(&field, cell),
                    throw: bevy_math::DVec3::ZERO,
                }
            })
            .collect::<Vec<_>>();
        let mut clumps = ClumpCollection::default();
        clumps.extract(&mut terrain, &field, &hole, false).unwrap();
        let dug: u64 = clumps
            .bodies
            .values()
            .map(|body| u64::from(body.quanta))
            .sum();
        assert_eq!(dug, 8 * u64::from(CELL_QUANTA));
        let mut laid = Vec::new();
        for id in clumps.bodies.keys().copied().collect::<Vec<_>>() {
            clumps
                .bodies
                .get_mut(&id)
                .unwrap()
                .update_settling(true, 1.0);
            // Beside the hole, as thrown spoil lands.
            clumps.bodies.get_mut(&id).unwrap().position = at(origin, 24, 12, 16).centre();
            let mut steps = 10_000;
            let outcome = clumps
                .settle(&mut terrain, &field, id, &mut |_| false, &mut steps)
                .unwrap();
            laid.extend(outcome.laid_cells);
        }
        assert!(clumps.bodies.is_empty(), "a crumb was left over");
        assert_eq!(laid.len(), 10, "eight cells of ground lie as ten of spoil");
        let held: u64 = laid
            .iter()
            .map(|&cell| {
                ExtractionCell {
                    cell,
                    sample: terrain.sample_cell(&field, cell),
                    throw: bevy_math::DVec3::ZERO,
                }
                .material_quanta()
            })
            .sum();
        assert_eq!(held, dug, "nothing is made or lost");
    }

    #[test]
    fn an_undercut_spoil_heap_slides_but_natural_ground_stands() {
        let step = (FLOOR + 1..FLOOR + 5)
            .flat_map(|y| (4..8).flat_map(move |x| (4..8).map(move |z| IVec3::new(x, y, z))))
            .collect::<Vec<_>>();
        let (field, mut terrain, origin) = slab(TerrainMaterial::Soil, &step);
        pour(
            &mut terrain,
            &field,
            at(origin, 18, 20, 18),
            TerrainMaterial::Soil,
            80,
            &mut |_| false,
        );
        // Cut the heap in half with a vertical face, and note the cut.
        let mut clumps = ClumpCollection::default();
        let mut slump = SpoilSlump::default();
        let cut = (FLOOR + 1..32)
            .flat_map(|y| (18..30).flat_map(move |x| (4..30).map(move |z| (x, y, z))))
            .map(|(x, y, z)| at(origin, x, y, z))
            .filter(|&cell| terrain.sample_cell(&field, cell).is_solid())
            .map(|cell| ExtractionCell {
                cell,
                sample: terrain.sample_cell(&field, cell),
                throw: bevy_math::DVec3::ZERO,
            })
            .collect::<Vec<_>>();
        assert!(cut.len() > 20);
        terrain.extract_cells(&field, &cut).unwrap();
        for source in &cut {
            slump.disturb(source.cell);
        }
        assert!(
            steepest(&terrain, &field, origin, 1) > 1,
            "the cut is not steep"
        );
        let mut breakage = crate::BreakageAccumulator::default();
        for _ in 0..200 {
            for body in clumps.bodies.values_mut() {
                body.update_settling(true, 1.0);
            }
            clumps.transfer(
                &mut terrain,
                &field,
                &mut breakage,
                &mut slump,
                &mut |_| false,
                TransferLimits::default(),
            );
            if slump.is_empty() && clumps.bodies.is_empty() {
                break;
            }
        }
        assert!(
            slump.is_empty() && clumps.bodies.is_empty(),
            "the slide never came to rest"
        );
        // The heap has run out to its repose; the block of natural ground has not moved.
        for z in 8..30 {
            for x in 8..30 {
                for (dx, dz) in DIRECTIONS {
                    let drop = height(&terrain, &field, origin, x, z)
                        - height(&terrain, &field, origin, x + dx, z + dz);
                    assert!(drop <= 1, "spoil stands {drop} cells high at {x},{z}");
                }
            }
        }
        assert_eq!(height(&terrain, &field, origin, 7, 7), 4);
        assert_eq!(height(&terrain, &field, origin, 8, 7), 0);
    }
}
