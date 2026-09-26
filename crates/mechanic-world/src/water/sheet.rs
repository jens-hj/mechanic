//! Running water: thin sheets on the ground that run downhill.
//!
//! A sheet cell is a water cell whose water rests on its floor. Each holds a
//! volume, and pipes to its four horizontal neighbours carry water between
//! them: the virtual-pipe model of shallow water (O'Brien and Hodgins; Mei,
//! Decaudin and Hu). A pipe's flow gathers speed with the difference in
//! surface height across it and loses it to friction, and no cell sends
//! more than it holds, so water is conserved exactly and a sheet has a
//! current. A sheet climbs a step of one water cell and runs down one; a
//! bigger drop is a lip it pours over. Water reaching a pool or seed-derived
//! water joins it. Water that stops in a hollow deep enough becomes a pool.

use std::collections::{BTreeMap, BTreeSet};

use bevy_math::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use super::{End, FILM_METRES, WATER_CELL_METRES, WaterCell, WaterFall, WaterGround, WaterWorld};
use super::{floor_of, held_in};
use crate::WaterSurface;

/// Horizontal area of one water cell, in square metres.
const CELL_AREA_M2: f64 = WATER_CELL_METRES * WATER_CELL_METRES;

/// Pipe cross-section over its length, in metres: the pipes behave like
/// water 20 cm deep, so waves cross a cell in a tenth of a second.
const PIPE_METRES: f64 = WATER_CELL_METRES;

/// Gravity, in m/s².
const GRAVITY: f64 = 9.81;

/// How fast friction takes a pipe's flow, per second.
const FRICTION_PER_SECOND: f64 = 2.0;

/// Substeps per water step: pipes need shorter steps than pools.
const SUBSTEPS: u32 = 4;

/// Water shallower than this clings to the ground and does not run, in
/// metres.
const CLING_METRES: f64 = 0.002;

/// Least water a sheet cell keeps; less evaporates at once, in m³.
const DRY_SHEET_M3: f64 = 1.0e-7;

/// Steps a sheet cell must lie still in a hollow before it becomes a pool.
const STILL_STEPS: u8 = 10;

/// The four horizontal directions: -x, +x, -z, +z.
const DIRECTIONS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

/// Running water in one cell.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Sheet {
    /// Water held, in m³.
    pub(super) volume: f64,
    /// Height of the cell's floor, in metres.
    floor: f64,
    /// Flow out through each horizontal face, in m³/s.
    flux: [f64; 4],
    /// Steps it has lain still in a hollow.
    still: u8,
}

/// One sheet cell in a saved world.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SheetDoc {
    /// The cell.
    pub cell: WaterCell,
    /// Water held, in m³.
    pub volume_m3: f64,
}

impl Sheet {
    /// Its surface, where it is deep enough to run.
    pub(super) fn surface(&self) -> Option<WaterSurface> {
        let depth = self.volume / CELL_AREA_M2;
        (depth >= CLING_METRES).then(|| WaterSurface {
            level: self.floor + depth,
            body: crate::WaterBody::Running,
            flow: current(self, depth),
        })
    }
}

/// Where water leaving a sheet cell through one face goes.
#[derive(Clone, Copy, Debug)]
enum Target {
    /// Onto the floor of this cell.
    Cell(WaterCell),
    /// Into standing or seed-derived water with its surface here.
    Water(End, f64),
    /// Over a lip, pouring down from this cell.
    Lip(WaterCell),
    /// Nowhere: ground stands above the water.
    Wall,
}

impl WaterWorld {
    /// Water held in running sheets, in m³.
    pub fn running_m3(&self) -> f64 {
        self.sheets.values().map(|sheet| sheet.volume).sum()
    }

    /// Running water to save.
    pub(super) fn sheet_docs(&self) -> Vec<SheetDoc> {
        self.sheets
            .iter()
            .map(|(&cell, sheet)| SheetDoc {
                cell,
                volume_m3: sheet.volume,
            })
            .collect()
    }

    /// Adds running water to a cell.
    pub(super) fn add_sheet(&mut self, ground: &impl WaterGround, cell: WaterCell, volume: f64) {
        let floor = floor_of(cell, self.openings(ground, cell)).unwrap_or_else(|| cell.bottom());
        let sheet = self.sheets.entry(cell).or_insert(Sheet {
            floor,
            ..Sheet::default()
        });
        sheet.volume += volume;
    }

    /// Height of a cell's running surface and its depth, if it has a floor.
    fn sheet_surface(&mut self, ground: &impl WaterGround, cell: WaterCell) -> Option<(f64, f64)> {
        if let Some(sheet) = self.sheets.get(&cell) {
            let depth = sheet.volume / CELL_AREA_M2;
            return Some((sheet.floor + depth, depth));
        }
        floor_of(cell, self.openings(ground, cell)).map(|floor| (floor, 0.0))
    }

    /// Running water at a cell: its surface and current, if it runs there.
    pub(super) fn running(&self, cell: WaterCell) -> Option<WaterSurface> {
        self.sheets.get(&cell)?.surface()
    }

    /// Whether water landing in a cell would stand there rather than run.
    pub(super) fn stands(
        &mut self,
        ground: &impl WaterGround,
        cell: WaterCell,
        volume: f64,
    ) -> bool {
        let Some((height, _)) = self.sheet_surface(ground, cell) else {
            return true;
        };
        self.in_hollow(ground, cell, height + volume / CELL_AREA_M2)
    }

    /// Where water leaving `cell` towards a horizontal neighbour goes, with
    /// the water's surface at `height`.
    fn target(
        &mut self,
        ground: &impl WaterGround,
        cell: WaterCell,
        (dx, dz): (i32, i32),
        height: f64,
    ) -> Target {
        let beside = WaterCell::new(cell.x + dx, cell.y, cell.z + dz);
        if let Some(target) = self.standing(ground, beside) {
            return target;
        }
        if let Some(floor) = floor_of(beside, self.openings(ground, beside)) {
            if floor >= height {
                return Target::Wall;
            }
            if !self.falls(ground, beside, None) {
                return Target::Cell(beside);
            }
            // Its floor drops away: one cell down is a slope, more a lip.
            let below = beside.below();
            if let Some(target) = self.standing(ground, below) {
                return target;
            }
            let rests = floor_of(below, self.openings(ground, below)).is_some()
                && !self.falls(ground, below, None);
            return if rests {
                Target::Cell(below)
            } else {
                Target::Lip(beside)
            };
        }
        // Solid beside: the water may climb a step of one cell.
        let above = WaterCell::new(beside.x, beside.y + 1, beside.z);
        let open_above = floor_of(cell.up(), self.openings(ground, cell.up())).is_some();
        match floor_of(above, self.openings(ground, above)) {
            Some(floor) if floor < height && open_above => {
                self.standing(ground, above).unwrap_or(Target::Cell(above))
            }
            _ => Target::Wall,
        }
    }

    /// Standing water in a cell: a pool or seed-derived water.
    fn standing(&mut self, ground: &impl WaterGround, cell: WaterCell) -> Option<Target> {
        if let Some(&id) = self.owner.get(&cell) {
            let level = self.pools.get(&id)?.level;
            // A pool's surface is never below the floor it covers, even
            // before it has water to settle.
            let floor =
                floor_of(cell, self.openings(ground, cell)).unwrap_or_else(|| cell.bottom());
            return Some(Target::Water(End::Pool(id), level.max(floor)));
        }
        self.implicit(ground, cell)
            .map(|surface| Target::Water(End::Body(surface.body), surface.level))
    }

    /// Runs the sheets for `dt` seconds. Returns the water moved, the pools
    /// that received water, and the streams pouring over lips.
    pub(super) fn step_sheets(
        &mut self,
        ground: &impl WaterGround,
        dt: f64,
    ) -> (f64, BTreeSet<u32>, Vec<WaterFall>) {
        let mut moved = 0.0;
        let mut fed = BTreeSet::new();
        let mut lips = BTreeMap::<WaterCell, (WaterCell, f64)>::new();
        let sub = dt / f64::from(SUBSTEPS);
        for _ in 0..SUBSTEPS {
            let cells = self.sheets.keys().copied().collect::<Vec<_>>();
            // Every face's flow first, from the surfaces as they stand.
            let mut sends = Vec::with_capacity(cells.len());
            for &cell in &cells {
                let Some((height, depth)) = self.sheet_surface(ground, cell) else {
                    continue;
                };
                let mut sheet = self.sheets[&cell];
                let mut targets = [Target::Wall; 4];
                for (face, &direction) in DIRECTIONS.iter().enumerate() {
                    let target = if depth < CLING_METRES {
                        Target::Wall
                    } else {
                        self.target(ground, cell, direction, height)
                    };
                    let beyond = match target {
                        Target::Cell(other) => self.sheet_surface(ground, other).map(|(h, _)| h),
                        Target::Water(_, level) => Some(level),
                        Target::Lip(other) => Some(other.bottom()),
                        Target::Wall => None,
                    };
                    let flow = beyond.map_or(0.0, |beyond| {
                        let damped = sheet.flux[face] * (1.0 - FRICTION_PER_SECOND * sub).max(0.0);
                        (damped + sub * GRAVITY * PIPE_METRES * (height - beyond)).max(0.0)
                    });
                    sheet.flux[face] = flow;
                    targets[face] = target;
                }
                let out = sheet.flux.iter().sum::<f64>() * sub;
                if out > 0.0 && out > sheet.volume {
                    let scale = sheet.volume.max(0.0) / out;
                    for flux in &mut sheet.flux {
                        *flux *= scale;
                    }
                }
                self.sheets.insert(cell, sheet);
                sends.push((cell, targets));
            }
            // Then the water moves.
            for (cell, targets) in sends {
                let flux = self.sheets[&cell].flux;
                for (face, target) in targets.into_iter().enumerate() {
                    let volume = flux[face] * sub;
                    if volume <= 0.0 {
                        continue;
                    }
                    if let Some(sheet) = self.sheets.get_mut(&cell) {
                        sheet.volume -= volume;
                    }
                    moved += volume;
                    match target {
                        Target::Cell(other) => self.add_sheet(ground, other, volume),
                        Target::Water(end, _) => {
                            if let End::Pool(id) = end {
                                fed.insert(id);
                            }
                            self.deposit_end(ground, end, volume);
                        }
                        Target::Lip(over) => {
                            let to = self.landing(ground, over.below());
                            if let End::Pool(id) = to {
                                fed.insert(id);
                            }
                            self.deposit_end(ground, to, volume);
                            lips.entry(over).or_insert((cell, 0.0)).1 += volume;
                        }
                        Target::Wall => {}
                    }
                }
            }
        }
        let falls = lips
            .into_iter()
            .map(|(over, (from, volume))| {
                let centre = over.centre();
                WaterFall {
                    from: DVec3::new(centre.x, from.bottom() + WATER_CELL_METRES, centre.z),
                    to: DVec3::new(centre.x, over.bottom() - WATER_CELL_METRES, centre.z),
                    rate_m3_s: volume / dt,
                }
            })
            .collect();
        (moved, fed, falls)
    }

    /// Sheet cells that have lain still in a hollow, deep enough to stand,
    /// become pools; sheet cells that dried up evaporate.
    pub(super) fn settle_sheets(&mut self, ground: &impl WaterGround) {
        let cells = self.sheets.keys().copied().collect::<Vec<_>>();
        for cell in cells {
            let Some(sheet) = self.sheets.get(&cell).copied() else {
                continue;
            };
            if sheet.volume < DRY_SHEET_M3 {
                self.sheets.remove(&cell);
                self.cycle.evaporate(sheet.volume.max(0.0));
                continue;
            }
            let Some((height, depth)) = self.sheet_surface(ground, cell) else {
                continue;
            };
            let draining = sheet.flux.iter().sum::<f64>() > 0.01 * sheet.volume;
            let hollow = depth >= FILM_METRES && !draining && self.in_hollow(ground, cell, height);
            let still = if hollow { sheet.still + 1 } else { 0 };
            if still >= STILL_STEPS {
                self.sheets.remove(&cell);
                self.start_pool(ground, cell, sheet.volume);
            } else if let Some(sheet) = self.sheets.get_mut(&cell) {
                sheet.still = still;
            }
        }
    }

    /// Whether no neighbour of a sheet cell lies lower than its floor, so
    /// water there stands rather than runs.
    fn in_hollow(&mut self, ground: &impl WaterGround, cell: WaterCell, height: f64) -> bool {
        let Some(floor) = floor_of(cell, self.openings(ground, cell)) else {
            return false;
        };
        DIRECTIONS.iter().all(
            |&direction| match self.target(ground, cell, direction, height) {
                Target::Cell(other) => {
                    floor_of(other, self.openings(ground, other)).is_none_or(|other| other >= floor)
                }
                Target::Water(_, level) => level >= floor,
                Target::Lip(_) => false,
                Target::Wall => true,
            },
        )
    }

    /// Water evaporating from every sheet into the air.
    pub(super) fn evaporate_sheets(&mut self, rate_m_s: f64, dt: f64) {
        let mut risen = 0.0;
        for sheet in self.sheets.values_mut() {
            let lost = (CELL_AREA_M2 * rate_m_s * dt).min(sheet.volume);
            sheet.volume -= lost;
            risen += lost;
        }
        self.cycle.evaporate(risen);
    }

    /// A sheet cell taken in by a pool gives it its water.
    pub(super) fn absorb_sheet(&mut self, id: u32, cell: WaterCell) {
        if let Some(sheet) = self.sheets.remove(&cell)
            && let Some(pool) = self.pools.get_mut(&id)
        {
            pool.volume += sheet.volume;
        }
    }

    /// A sheet cell whose ground changed: water the ground now fills rises
    /// to the cell above.
    pub(super) fn remeasure_sheet(&mut self, ground: &impl WaterGround, cell: WaterCell) {
        let Some(sheet) = self.sheets.get(&cell).copied() else {
            return;
        };
        let openings = self.openings(ground, cell);
        match floor_of(cell, openings) {
            Some(floor) if held_in(cell, openings, f64::INFINITY) > 0.0 => {
                if let Some(sheet) = self.sheets.get_mut(&cell) {
                    sheet.floor = floor;
                }
            }
            _ => {
                self.sheets.remove(&cell);
                self.deposit_at(ground, cell.up(), sheet.volume);
            }
        }
    }
}

/// A sheet's current from the flow through its faces, in m/s.
fn current(sheet: &Sheet, depth: f64) -> DVec2 {
    let across = WATER_CELL_METRES * depth.max(CLING_METRES);
    DVec2::new(sheet.flux[1] - sheet.flux[0], sheet.flux[3] - sheet.flux[2]) / across
}
