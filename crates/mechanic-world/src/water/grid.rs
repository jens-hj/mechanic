//! Running water stored densely: square tiles of water-cell columns, each
//! column holding at most one sheet. Running water lies on the ground, so a
//! column rarely needs more than one, and dense tiles let the pipes run as
//! plain arithmetic over arrays instead of lookups in a map.

use super::WaterCell;
use super::cells::CellMap;
use super::sheet::Sheet;

/// Columns along one edge of a tile: 32 water cells, 6.4 m.
const TILE_EDGE: i32 = 32;

/// Columns in one tile.
const TILE_COLUMNS: usize = (TILE_EDGE * TILE_EDGE) as usize;

/// Where a column lives: its tile's slot and its index in the tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Slot {
    pub(super) tile: usize,
    pub(super) index: usize,
}

/// One tile of columns.
#[derive(Clone, Debug)]
struct Tile {
    key: (i32, i32),
    columns: Box<[Sheet]>,
    /// Columns holding a sheet.
    wet: usize,
}

/// Every sheet of running water.
#[derive(Clone, Debug, Default)]
pub(super) struct SheetGrid {
    tiles: Vec<Tile>,
    index: CellMap<(i32, i32), usize>,
    wet: usize,
}

const fn tile_key(x: i32, z: i32) -> (i32, i32) {
    (x.div_euclid(TILE_EDGE), z.div_euclid(TILE_EDGE))
}

#[expect(clippy::cast_sign_loss, reason = "remainders are non-negative")]
const fn column_index(x: i32, z: i32) -> usize {
    (x.rem_euclid(TILE_EDGE) + TILE_EDGE * z.rem_euclid(TILE_EDGE)) as usize
}

impl SheetGrid {
    /// Sheets held.
    pub(super) const fn len(&self) -> usize {
        self.wet
    }

    /// Where a column lives, if its tile exists.
    pub(super) fn slot(&self, x: i32, z: i32) -> Option<Slot> {
        self.index.get(&tile_key(x, z)).map(|&tile| Slot {
            tile,
            index: column_index(x, z),
        })
    }

    /// Where a column lives, making its tile if need be. Slots stay valid
    /// until [`Self::compact`].
    pub(super) fn slot_or_insert(&mut self, x: i32, z: i32) -> Slot {
        let key = tile_key(x, z);
        let tile = *self.index.entry(key).or_insert_with(|| {
            self.tiles.push(Tile {
                key,
                columns: vec![Sheet::default(); TILE_COLUMNS].into_boxed_slice(),
                wet: 0,
            });
            self.tiles.len() - 1
        });
        Slot {
            tile,
            index: column_index(x, z),
        }
    }

    /// The column in a slot.
    pub(super) fn at(&self, slot: Slot) -> &Sheet {
        &self.tiles[slot.tile].columns[slot.index]
    }

    /// The column in a slot, to change. A sheet made or taken away through
    /// it must go through [`Self::place`] or [`Self::remove_at`].
    pub(super) fn at_mut(&mut self, slot: Slot) -> &mut Sheet {
        &mut self.tiles[slot.tile].columns[slot.index]
    }

    /// The sheet in a cell, if its column holds one at that height.
    pub(super) fn get(&self, cell: WaterCell) -> Option<&Sheet> {
        let sheet = self.at(self.slot(cell.x, cell.z)?);
        (sheet.present && sheet.y == cell.y).then_some(sheet)
    }

    /// The sheet in a cell, to change.
    pub(super) fn get_mut(&mut self, cell: WaterCell) -> Option<&mut Sheet> {
        let slot = self.slot(cell.x, cell.z)?;
        let sheet = self.at_mut(slot);
        (sheet.present && sheet.y == cell.y).then_some(sheet)
    }

    /// Makes an empty column in a slot hold a dry sheet at height `y` over
    /// `floor`. A column already holding one keeps it.
    pub(super) fn place(&mut self, slot: Slot, y: i32, floor: f64) -> &mut Sheet {
        let tile = &mut self.tiles[slot.tile];
        let sheet = &mut tile.columns[slot.index];
        if !sheet.present {
            *sheet = Sheet::new(y, floor);
            tile.wet += 1;
            self.wet += 1;
        }
        sheet
    }

    /// Takes the sheet out of a slot.
    pub(super) fn remove_at(&mut self, slot: Slot) -> Option<Sheet> {
        let tile = &mut self.tiles[slot.tile];
        let sheet = tile.columns[slot.index];
        if !sheet.present {
            return None;
        }
        tile.columns[slot.index] = Sheet::default();
        tile.wet -= 1;
        self.wet -= 1;
        Some(sheet)
    }

    /// Takes the sheet out of a cell.
    pub(super) fn remove(&mut self, cell: WaterCell) -> Option<Sheet> {
        let slot = self.slot(cell.x, cell.z)?;
        (self.at(slot).y == cell.y).then(|| self.remove_at(slot))?
    }

    /// Every sheet's slot and cell.
    pub(super) fn wet(&self) -> Vec<(Slot, WaterCell)> {
        let mut wet = Vec::with_capacity(self.wet);
        for (tile, entry) in self.tiles.iter().enumerate() {
            if entry.wet == 0 {
                continue;
            }
            let (tx, tz) = entry.key;
            for (index, sheet) in entry.columns.iter().enumerate() {
                if sheet.present {
                    let index_i32 = i32::try_from(index).expect("a tile's index fits i32");
                    wet.push((
                        Slot { tile, index },
                        WaterCell::new(
                            tx * TILE_EDGE + index_i32 % TILE_EDGE,
                            sheet.y,
                            tz * TILE_EDGE + index_i32 / TILE_EDGE,
                        ),
                    ));
                }
            }
        }
        wet
    }

    /// Every sheet with its cell.
    pub(super) fn iter(&self) -> impl Iterator<Item = (WaterCell, &Sheet)> + '_ {
        self.tiles
            .iter()
            .filter(|tile| tile.wet > 0)
            .flat_map(|tile| {
                let (tx, tz) = tile.key;
                tile.columns
                    .iter()
                    .enumerate()
                    .filter(|(_, sheet)| sheet.present)
                    .map(move |(index, sheet)| {
                        let index = i32::try_from(index).expect("a tile's index fits i32");
                        (
                            WaterCell::new(
                                tx * TILE_EDGE + index % TILE_EDGE,
                                sheet.y,
                                tz * TILE_EDGE + index / TILE_EDGE,
                            ),
                            sheet,
                        )
                    })
            })
    }

    /// Every sheet, to change.
    pub(super) fn values_mut(&mut self) -> impl Iterator<Item = &mut Sheet> + '_ {
        self.tiles
            .iter_mut()
            .filter(|tile| tile.wet > 0)
            .flat_map(|tile| tile.columns.iter_mut().filter(|sheet| sheet.present))
    }

    /// Forgets the face routes of a column and its four neighbours, whose
    /// ground changed.
    pub(super) fn forget_routes(&mut self, x: i32, z: i32) {
        for (dx, dz) in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)] {
            if let Some(slot) = self.slot(x + dx, z + dz) {
                self.at_mut(slot).forget_routes();
            }
        }
    }

    /// Drops the tiles no sheet is left in. Slots taken before are no longer
    /// valid.
    pub(super) fn compact(&mut self) {
        if self.tiles.iter().all(|tile| tile.wet > 0) {
            return;
        }
        self.tiles.retain(|tile| tile.wet > 0);
        self.index = self
            .tiles
            .iter()
            .enumerate()
            .map(|(slot, tile)| (tile.key, slot))
            .collect();
    }
}
