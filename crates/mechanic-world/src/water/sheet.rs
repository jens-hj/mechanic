//! Running water: thin sheets on the ground that run downhill.
//!
//! A sheet is water resting on the floor of a water cell. Each holds a
//! volume, and pipes to its four horizontal neighbours carry water between
//! them: the virtual-pipe model of shallow water (O'Brien and Hodgins; Mei,
//! Decaudin and Hu). A pipe's flow gathers speed with the difference in
//! surface height across it and loses it to friction, and no sheet sends
//! more than it holds, so water is conserved exactly and a sheet has a
//! current. A sheet climbs a step of one water cell and runs down a drop of
//! up to a metre as a steep chute; a taller drop is a lip it pours over, and
//! the water lands at once wherever the drop leads.
//! Water reaching seed-derived water joins it; running water and a pool meet
//! through a pipe that runs either way, by the pool's level. Under open sky a
//! sheet may grow as deep as it likes: a pond is running water that has come
//! to rest, flat because the pipes have no other resting state, so a flood
//! fills hollows and spills on without breaking into flat pools that step
//! down the slope. Only water that rises against a roof, in a cave or a
//! tunnel, becomes a pool.
//!
//! Sheets live in dense tiles ([`super::grid`]); a tile whose water lies
//! still sleeps and costs nothing until water reaches it or its ground
//! changes. Where each face of a sheet leads is worked out once from the
//! ground, from the cell its surface is in, and kept until the ground
//! changes, so the pipes themselves are plain arithmetic.

use std::collections::BTreeSet;

use bevy_math::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use super::grid::Slot;
use super::sediment::SedimentLoad;
use super::{
    CLING_METRES, End, FILM_METRES, GRAVITY, Joined, MERGE_METRES, SPREAD_CELLS_PER_STEP,
    WATER_CELL_METRES, WaterCell, WaterGround, WaterWorld,
};
use super::{floor_of, held_in};
use crate::WaterSurface;

/// Horizontal area of one water cell, in square metres.
const CELL_AREA_M2: f64 = WATER_CELL_METRES * WATER_CELL_METRES;

/// Deepest water a pipe's drive counts, in metres: deeper water drives no
/// harder, which keeps waves slow enough for the substeps (3 m/s, a fifth
/// of a cell per substep).
const DRIVE_METRES: f64 = 1.0;

/// How fast friction takes a pipe's flow whatever its depth, per second.
const FRICTION_PER_SECOND: f64 = 0.5;

/// Manning's roughness of the ground under running water, in s/m^(1/3):
/// short grass and bare soil.
pub(super) const ROUGHNESS: f64 = 0.03;

/// Longest pipe substep, in seconds: pipes need shorter steps than pools,
/// and a longer one rocks water back and forth between neighbours.
const SUBSTEP_SECONDS: f64 = 0.0125;

/// Most water, as a depth in metres, a column may gain, lose or pass on over
/// a step while its tile counts as still, or stand above a sleeping
/// neighbour before that wakes.
const STILL_METRES: f64 = 2.0e-5;

/// Steps between the checks of whether running water has come under still
/// water or a roof, while its surface stays in one cell.
const SETTLE_EVERY: u32 = 16;

/// Least water a sheet keeps; less evaporates at once, in m³.
const DRY_SHEET_M3: f64 = 1.0e-7;

/// Deepest drop, in water cells, that running water takes as a steep chute
/// rather than pouring over as a fall.
const CHUTE_CELLS: i32 = 5;

/// The four horizontal directions: -x, +x, -z, +z.
const DIRECTIONS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

/// Where one face of a sheet leads, as far as the ground alone decides.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum Face {
    /// Ground stands above the water.
    #[default]
    Wall,
    /// Onto the floor of the neighbour column's cell at height `y`, if the
    /// water stands above `floor`. Standing water in the neighbour column
    /// from `above` down to `y` takes the water first.
    Onto { y: i32, floor: f64, above: i32 },
    /// Over a lip at height `y`: nothing rests for a metre below. Standing
    /// water in the neighbour column down to `lowest` takes the water first.
    Lip { y: i32, lowest: i32 },
}

/// Running water in one column.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Sheet {
    /// Whether the column holds a sheet.
    pub(super) present: bool,
    /// Height of its cell, in water cells.
    pub(super) y: i32,
    /// Water held, in m³.
    pub(super) volume: f64,
    /// Height of the cell's floor, in metres.
    floor: f64,
    /// Flow out through each horizontal face, in m³/s.
    flux: [f64; 4],
    /// Where each face leads, once worked out, and the height of the cell
    /// its surface was in then, in water cells.
    faces: Option<([Face; 4], i32)>,
    /// The height of the cell its surface was in, in water cells, when it
    /// was last found neither under still water nor under a roof.
    settled: Option<i32>,
    /// Sediment it carries.
    pub(super) load: SedimentLoad,
}

/// One sheet in a saved world.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SheetDoc {
    /// The cell.
    pub cell: WaterCell,
    /// Water held, in m³.
    pub volume_m3: f64,
    /// Sediment it carries.
    #[serde(default)]
    pub load: SedimentLoad,
}

impl Sheet {
    /// A dry sheet in the cell at height `y`, over `floor`.
    pub(super) fn new(y: i32, floor: f64) -> Self {
        Self {
            present: true,
            y,
            floor,
            ..Self::default()
        }
    }

    /// Height of the floor its water rests on.
    pub(super) const fn floor(&self) -> f64 {
        self.floor
    }

    /// Height of its water's surface, however shallow.
    pub(super) fn surface_height(&self) -> f64 {
        self.floor + self.volume / CELL_AREA_M2
    }

    /// The height of the cell its surface is in, in water cells: a pond
    /// deeper than its floor's cell reaches up into the cells above.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "heights are far inside i32"
    )]
    pub(super) fn top(&self) -> i32 {
        let top = (self.surface_height() / WATER_CELL_METRES).floor() as i32;
        top.max(self.y)
    }

    /// Whether its water reaches up into a cell of its column.
    pub(super) fn covers(&self, cell: WaterCell) -> bool {
        self.present && cell.y >= self.y && cell.y <= self.top()
    }

    /// Its surface, where it is deep enough to run.
    pub(super) fn surface(&self) -> Option<WaterSurface> {
        let depth = self.volume / CELL_AREA_M2;
        (depth >= CLING_METRES).then(|| WaterSurface {
            level: self.floor + depth,
            body: crate::WaterBody::Running,
            flow: current(self, depth),
        })
    }

    /// Forgets where its faces lead, since the ground around it changed.
    pub(super) fn forget_routes(&mut self) {
        self.faces = None;
        self.settled = None;
    }
}

/// Where water leaving a sheet through one face goes this step.
#[derive(Clone, Copy, Debug)]
enum Route {
    /// Nowhere.
    Wall,
    /// Onto the floor of a neighbour column, if the water stands above it.
    Onto { slot: Slot, y: i32, floor: f64 },
    /// Into seed-derived water with its surface here.
    Water(End, f64),
    /// To and from a pool, through its cell with this floor: the pipe runs
    /// either way, by the pool's level as it stands, so running water and
    /// the pool it meets come to one level.
    Pool { id: u32, floor: f64 },
    /// Over a lip, pouring down from this cell.
    Lip(WaterCell),
}

/// One cell of running water as it is drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunningView {
    /// The cell.
    pub cell: WaterCell,
    /// Height of its surface, in metres.
    pub level: f64,
    /// Depth of its water over its floor, in metres.
    pub depth: f64,
    /// Its current along x and z, in m/s.
    pub flow: DVec2,
}

impl WaterWorld {
    /// Every cell of running water deep enough to run, to draw.
    pub fn running_cells(&self) -> Vec<RunningView> {
        self.sheets
            .iter()
            .filter_map(|(cell, sheet)| {
                let surface = sheet.surface()?;
                Some(RunningView {
                    cell,
                    level: surface.level,
                    depth: surface.level - sheet.floor,
                    flow: surface.flow,
                })
            })
            .collect()
    }

    /// Water held in running sheets, in m³.
    pub fn running_m3(&self) -> f64 {
        self.sheets.iter().map(|(_, sheet)| sheet.volume).sum()
    }

    /// Running water to save.
    pub(super) fn sheet_docs(&self) -> Vec<SheetDoc> {
        self.sheets
            .iter()
            .map(|(cell, sheet)| SheetDoc {
                cell,
                volume_m3: sheet.volume,
                load: sheet.load,
            })
            .collect()
    }

    /// Adds running water to a cell. A column already running at another
    /// height takes it into its own sheet.
    pub(super) fn add_sheet(&mut self, ground: &impl WaterGround, cell: WaterCell, volume: f64) {
        let slot = self.sheets.slot_or_insert(cell.x, cell.z);
        if !self.sheets.at(slot).present {
            let floor = self
                .sheet_floor(ground, cell)
                .unwrap_or_else(|| cell.bottom());
            self.sheets.place(slot, cell.y, floor);
        }
        self.sheets.at_mut(slot).volume += volume;
        self.sheets.stir(slot, volume / CELL_AREA_M2);
    }

    /// Height of a cell's running surface and its depth, if it has a floor.
    pub(super) fn sheet_surface(
        &mut self,
        ground: &impl WaterGround,
        cell: WaterCell,
    ) -> Option<(f64, f64)> {
        // Running water with its floor in this cell or below it, however
        // deep it reaches.
        if let Some(slot) = self.sheets.slot(cell.x, cell.z)
            && let sheet = self.sheets.at(slot)
            && sheet.present
            && sheet.y <= cell.y
        {
            let depth = sheet.volume / CELL_AREA_M2;
            return Some((sheet.floor + depth, depth));
        }
        self.sheet_floor(ground, cell).map(|floor| (floor, 0.0))
    }

    /// Running water at a cell: its surface and current, if it runs there.
    pub(super) fn running(&self, cell: WaterCell) -> Option<WaterSurface> {
        self.sheets.covering(cell)?.surface()
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
        self.confined(ground, cell, height + volume / CELL_AREA_M2)
    }

    /// Where each face of a sheet leads, from the ground alone, for water
    /// whose surface is in `cell` and whose floor is in the cell at height
    /// `floor_y`: a pond looks over its banks from its top.
    fn faces(&mut self, ground: &impl WaterGround, cell: WaterCell, floor_y: i32) -> [Face; 4] {
        DIRECTIONS.map(|(dx, dz)| {
            let level = WaterCell::new(cell.x + dx, cell.y, cell.z + dz);
            // The highest opening beside the water, from its surface down to
            // its floor: a pond drains through a hole in its bank.
            let mut beside = level;
            let mut open = self.sheet_floor(ground, beside);
            while open.is_none() && beside.y > floor_y {
                beside = beside.below();
                open = self.sheet_floor(ground, beside);
            }
            if let Some(floor) = open {
                if !self.drops(ground, beside) {
                    return Face::Onto {
                        y: beside.y,
                        floor,
                        above: beside.y,
                    };
                }
                // Its floor drops away: down to a metre below this floor the
                // water runs on down a steep chute, further it pours over.
                let mut below = beside;
                for _ in 0..(cell.y - floor_y).max(0) + CHUTE_CELLS {
                    below = below.below();
                    let Some(floor) = self.sheet_floor(ground, below) else {
                        break;
                    };
                    if !self.drops(ground, below) {
                        return Face::Onto {
                            y: below.y,
                            floor,
                            above: beside.y,
                        };
                    }
                }
                return Face::Lip {
                    y: beside.y,
                    lowest: below.y,
                };
            }
            // Solid beside: the water may climb a step of one cell.
            let above = level.up();
            let open_above = floor_of(cell.up(), self.openings(ground, cell.up())).is_some();
            match self.sheet_floor(ground, above) {
                Some(floor) if open_above => Face::Onto {
                    y: above.y,
                    floor,
                    above: above.y,
                },
                _ => Face::Wall,
            }
        })
    }

    /// Whether the ground under a cell's floor is open: water there drops
    /// into the cell below, whatever water it may meet.
    fn drops(&mut self, ground: &impl WaterGround, cell: WaterCell) -> bool {
        self.openings(ground, cell)[0] > 0 && self.openings(ground, cell.below())[3] > 0
    }

    /// Where one face of a sheet in `cell` sends water this step: into
    /// standing water met on the way, or where the ground leads.
    fn route(
        &mut self,
        ground: &impl WaterGround,
        cell: WaterCell,
        (dx, dz): (i32, i32),
        face: Face,
    ) -> Route {
        let (x, z) = (cell.x + dx, cell.z + dz);
        // Water already running where the face leads stands in no pool:
        // pools take in the sheets they reach.
        if let Face::Onto { y, floor, .. } = face
            && let Some(slot) = self.sheets.slot(x, z)
            && self.sheets.at(slot).present
            && self.sheets.at(slot).y == y
        {
            return Route::Onto { slot, y, floor };
        }
        // Running water filling the drop beyond a lip up to it takes the
        // water as its own: a full pit is no fall.
        if let Face::Lip { y, .. } = face
            && let Some(slot) = self.sheets.slot(x, z)
            && let sheet = self.sheets.at(slot)
            && sheet.present
            && sheet.surface_height() >= WaterCell::new(x, y, z).bottom()
        {
            return Route::Onto {
                slot,
                y: sheet.y,
                floor: sheet.floor,
            };
        }
        let (top, bottom) = match face {
            Face::Wall => return Route::Wall,
            Face::Onto { y, above, .. } => (above, y),
            Face::Lip { y, lowest } => (y, lowest),
        };
        for y in (bottom..=top).rev() {
            let beside = WaterCell::new(x, y, z);
            if let Some(&id) = self.owner.get(&beside)
                && self.pools.contains_key(&id)
            {
                // A pool's surface is never below the floor it covers, even
                // before it has water to settle.
                let floor = floor_of(beside, self.openings(ground, beside))
                    .unwrap_or_else(|| beside.bottom());
                return Route::Pool { id, floor };
            }
            if let Some(surface) = self.implicit(ground, beside) {
                return Route::Water(End::Body(surface.body), surface.level);
            }
        }
        match face {
            Face::Wall => Route::Wall,
            Face::Onto { y, floor, .. } => Route::Onto {
                slot: self.sheets.slot_or_insert(x, z),
                y,
                floor,
            },
            Face::Lip { y, .. } => Route::Lip(WaterCell::new(x, y, z)),
        }
    }

    /// Where each face of every sheet leads this step: the ground's routes,
    /// worked out again where the ground changed, met by standing water.
    fn routes(&mut self, ground: &impl WaterGround) -> Vec<(Slot, [Route; 4])> {
        let wet = self.sheets.wet();
        let mut routes = Vec::with_capacity(wet.len());
        for (slot, cell) in wet {
            if self.sheets.asleep(slot.tile) {
                continue;
            }
            let sheet = *self.sheets.at(slot);
            if sheet.volume / CELL_AREA_M2 < CLING_METRES {
                routes.push((slot, [Route::Wall; 4]));
                continue;
            }
            let top = sheet.top();
            let faces = match sheet.faces {
                Some((faces, at)) if at == top => faces,
                _ => {
                    let faces = self.faces(ground, WaterCell::new(cell.x, top, cell.z), cell.y);
                    self.sheets.at_mut(slot).faces = Some((faces, top));
                    faces
                }
            };
            let mut face_routes = [Route::Wall; 4];
            for (route, (&direction, face)) in
                face_routes.iter_mut().zip(DIRECTIONS.iter().zip(faces))
            {
                *route = self.route(ground, cell, direction, face);
                // Sleeping water standing higher than this wakes to run
                // down into it.
                if let Route::Onto { slot: other, .. } = *route
                    && self.sheets.asleep(other.tile)
                    && self.sheets.at(other).present
                    && self.sheets.at(other).surface_height()
                        > sheet.surface_height() + STILL_METRES
                {
                    self.sheets.wake(cell.x + direction.0, cell.z + direction.1);
                }
            }
            routes.push((slot, face_routes));
        }
        routes
    }

    /// The surface beyond one face of a sheet, and the floor the water
    /// crosses it over, where water may cross.
    fn beyond(&self, sheet: &Sheet, route: Route) -> Option<(f64, f64)> {
        let height = sheet.surface_height();
        match route {
            Route::Wall => None,
            Route::Onto { floor, .. } if floor >= height => None,
            Route::Onto { slot, floor, .. } => {
                let other = self.sheets.at(slot);
                Some((
                    if other.present {
                        other.surface_height()
                    } else {
                        floor
                    },
                    floor.max(sheet.floor),
                ))
            }
            Route::Water(_, level) => Some((level, sheet.floor)),
            Route::Pool { id, floor } => {
                let sill = floor.max(sheet.floor);
                // A pool fuller than its cells takes nothing in, and presses
                // out no harder than their top: its surplus is no head of
                // water.
                self.pools
                    .get(&id)
                    .map(|pool| {
                        let level = pool.level.max(floor);
                        if height > level {
                            level
                        } else {
                            pool.level.min(pool.top()).max(floor).max(height)
                        }
                    })
                    .map(|level| (level, sill))
                    .filter(|&(level, sill)| height.max(level) > sill)
            }
            Route::Lip(over) => Some((over.bottom(), sheet.floor)),
        }
    }

    /// Moves what pools send into a sheet through its faces this substep,
    /// each at most the water it holds over the sill and never below its
    /// rim: water leaves a pool over the highest ground it crossed. Returns
    /// the water moved, and marks the pools it came from.
    fn take_from_pools(
        &mut self,
        (index, slot): (usize, Slot),
        faces: &[Route; 4],
        flow: &mut [f64; 4],
        sub: f64,
        (touched, drawn): (&mut BTreeSet<u32>, &mut Vec<(usize, u32, f64)>),
    ) -> f64 {
        let mut moved = 0.0;
        for (face, route) in faces.iter().enumerate() {
            let Route::Pool { id, floor } = *route else {
                continue;
            };
            if flow[face] >= 0.0 {
                continue;
            }
            let Some(pool) = self.pools.get_mut(&id) else {
                flow[face] = 0.0;
                continue;
            };
            let sill = floor.max(self.sheets.at(slot).floor).max(pool.rim);
            let given = (-flow[face] * sub)
                .min(pool.volume - pool.held_below(sill))
                .max(0.0);
            pool.volume -= given;
            touched.insert(id);
            if given > 0.0 {
                drawn.push((index, id, given));
            }
            flow[face] = -given / sub;
            self.sheets.at_mut(slot).volume += given;
            self.sheets.stir(slot, given / CELL_AREA_M2);
            moved += given;
        }
        moved
    }

    /// Every face's flow over the next substep of `sub` seconds, from the
    /// surfaces as they stand.
    fn pipe_flows(&self, routes: &[(Slot, [Route; 4])], flows: &mut [[f64; 4]], sub: f64) {
        for ((slot, faces), flow) in routes.iter().zip(flows.iter_mut()) {
            let sheet = self.sheets.at(*slot);
            let height = sheet.surface_height();
            let depth = (sheet.volume / CELL_AREA_M2).max(CLING_METRES);
            for (face, route) in faces.iter().enumerate() {
                let beyond = self.beyond(sheet, *route);
                let both_ways = matches!(route, Route::Pool { .. });
                flow[face] = beyond.map_or(0.0, |(beyond, sill)| {
                    // The pipe is as deep as the water over the face:
                    // a film is pushed as a film, not as a stream.
                    let across = (height.max(beyond) - sill).clamp(CLING_METRES, DRIVE_METRES);
                    let driven = sheet.flux[face] + sub * GRAVITY * across * (height - beyond);
                    let flow = driven / (1.0 + friction(sheet.flux[face], depth) * sub);
                    if both_ways { flow } else { flow.max(0.0) }
                });
            }
            // No sheet sends more than it holds; water a pool sends in
            // is the pool's to give.
            let out = flow.iter().map(|face| face.max(0.0)).sum::<f64>() * sub;
            if out > 0.0 && out > sheet.volume {
                let scale = sheet.volume.max(0.0) / out;
                for face in flow.iter_mut().filter(|face| **face > 0.0) {
                    *face *= scale;
                }
            }
        }
    }

    /// Runs the sheets for `dt` seconds. Returns the water moved and the
    /// pools that received water.
    pub(super) fn step_sheets(
        &mut self,
        ground: &impl WaterGround,
        dt: f64,
    ) -> (f64, BTreeSet<u32>) {
        let routes = self.routes(ground);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a positive count of substeps"
        )]
        let substeps = (dt / SUBSTEP_SECONDS).ceil().max(1.0) as u32;
        let sub = dt / f64::from(substeps);
        let mut moved = 0.0;
        let mut into = Vec::new();
        let mut pours = Vec::new();
        let mut flows = vec![[0.0; 4]; routes.len()];
        let mut fed = BTreeSet::new();
        // Water each face sent where sediment goes with it, and every face
        // sent anywhere, over the step, and what pools gave each sheet.
        let mut carried = vec![[0.0; 4]; routes.len()];
        let mut sent = vec![0.0; routes.len()];
        let mut drawn = Vec::new();
        for _ in 0..substeps {
            // Every face's flow first, from the surfaces as they stand.
            self.pipe_flows(&routes, &mut flows, sub);
            // Then the water moves.
            let mut touched = BTreeSet::new();
            for (index, ((slot, faces), flow)) in routes.iter().zip(&mut flows).enumerate() {
                moved += self.take_from_pools(
                    (index, *slot),
                    faces,
                    flow,
                    sub,
                    (&mut touched, &mut drawn),
                );
                self.sheets.at_mut(*slot).flux = *flow;
                for (face, route) in faces.iter().enumerate() {
                    let volume = flow[face] * sub;
                    if volume <= 0.0 {
                        continue;
                    }
                    self.sheets.at_mut(*slot).volume -= volume;
                    self.sheets.stir(*slot, volume / CELL_AREA_M2);
                    moved += volume;
                    sent[index] += volume;
                    if !matches!(route, Route::Water(..)) {
                        carried[index][face] += volume;
                    }
                    match *route {
                        Route::Onto { slot, y, floor } => {
                            self.sheets.place(slot, y, floor).volume += volume;
                            self.sheets.stir(slot, volume / CELL_AREA_M2);
                        }
                        Route::Water(end, _) => into.push((end, volume)),
                        Route::Pool { id, .. } => {
                            if let Some(pool) = self.pools.get_mut(&id) {
                                pool.volume += volume;
                                touched.insert(id);
                                fed.insert(id);
                            }
                        }
                        // It pours through the lip's cell, which may be a hole
                        // in a bank lower than its surface.
                        Route::Lip(over) => pours.push((over, volume, index, face)),
                        Route::Wall => {}
                    }
                }
            }
            // Each pool stands where its water now leaves it, for the next
            // substep's pipes.
            for id in touched {
                if let Some(pool) = self.pools.get_mut(&id) {
                    pool.settle();
                }
            }
        }
        let lips = self.carry_loads(&routes, &carried, &sent, &drawn);
        for (end, volume) in into {
            if let End::Pool(id) = end {
                fed.insert(id);
            }
            self.deposit_end(ground, end, volume);
        }
        // Water over a lip lands at once wherever the drop leads, with its
        // share of what its sheet carried over the lip.
        for (over, volume, index, face) in pours {
            let end = self.landing(ground, over);
            if let End::Pool(id) = end {
                fed.insert(id);
            }
            let placed = self.deposit_end(ground, end, volume);
            let load = lips
                .get(&(index, face))
                .map_or_else(SedimentLoad::default, |load| {
                    load.scaled(volume / carried[index][face])
                });
            self.give_load(placed, load, (over.x, over.z, over.bottom()));
        }
        self.sheets.rest(STILL_METRES * sub / dt);
        (moved, fed)
    }

    /// Sediment moving with the step's water, from what each sheet and pool
    /// carried as it began: each gives the share of its load that the water
    /// it sent on to running water or a pool was of all it held, as if well
    /// stirred; water it sent into seed-derived water leaves its sediment
    /// behind. Returns what each lip face carries over, for its pours.
    fn carry_loads(
        &mut self,
        routes: &[(Slot, [Route; 4])],
        carried: &[[f64; 4]],
        sent: &[f64],
        drawn: &[(usize, u32, f64)],
    ) -> std::collections::BTreeMap<(usize, usize), SedimentLoad> {
        let mut arriving = Vec::new();
        let mut given = std::collections::BTreeMap::<u32, f64>::new();
        for &(_, id, volume) in drawn {
            *given.entry(id).or_default() += volume;
        }
        let mut from_pools = std::collections::BTreeMap::<u32, SedimentLoad>::new();
        for (&id, &volume) in &given {
            if let Some(pool) = self.pools.get_mut(&id) {
                let held = pool.volume + volume;
                from_pools.insert(id, pool.load.part(volume, held));
            }
        }
        for &(index, id, volume) in drawn {
            if let Some(load) = from_pools.get(&id) {
                arriving.push((routes[index].0, load.scaled(volume / given[&id])));
            }
        }
        let mut lips = std::collections::BTreeMap::new();
        for (index, (slot, faces)) in routes.iter().enumerate() {
            let out = carried[index].iter().sum::<f64>();
            if out <= 0.0 || self.sheets.at(*slot).load.total() <= 0.0 {
                continue;
            }
            let sheet = self.sheets.at_mut(*slot);
            let held = sheet.volume.max(0.0) + sent[index];
            let leaving = sheet.load.part(out, held);
            for (face, route) in faces.iter().enumerate() {
                if carried[index][face] <= 0.0 {
                    continue;
                }
                let part = leaving.scaled(carried[index][face] / out);
                match *route {
                    Route::Onto { slot, .. } => arriving.push((slot, part)),
                    Route::Pool { id, .. } => match self.pools.get_mut(&id) {
                        Some(pool) => pool.load.add(part),
                        None => self.sheets.at_mut(*slot).load.add(part),
                    },
                    Route::Lip(_) => {
                        lips.insert((index, face), part);
                    }
                    Route::Wall | Route::Water(..) => self.sheets.at_mut(*slot).load.add(part),
                }
            }
        }
        for (slot, load) in arriving {
            self.sheets.at_mut(slot).load.add(load);
        }
        lips
    }

    /// Sheets risen against a roof become pools; sheets that dried up
    /// evaporate.
    pub(super) fn settle_sheets(&mut self, ground: &impl WaterGround) {
        for (slot, cell) in self.sheets.wet() {
            let sheet = *self.sheets.at(slot);
            if !sheet.present || sheet.y != cell.y {
                continue;
            }
            if sheet.volume < DRY_SHEET_M3 {
                self.sheets.remove_at(slot);
                self.cycle.evaporate(sheet.volume.max(0.0));
                self.settle_load((cell.x, cell.z), sheet.floor, sheet.load);
                continue;
            }
            // Sleeping water has settled where it lies, and water whose
            // surface has not left the cell it was last checked in is looked
            // at again only now and then.
            if self.sheets.asleep(slot.tile)
                || (sheet.settled == Some(sheet.top()) && !self.sheets.due(slot, SETTLE_EVERY))
            {
                continue;
            }
            if self.submerge(ground, cell, sheet) {
                continue;
            }
            // Under open sky a pond is running water that has stopped: only
            // water rising against a roof stands as a pool.
            if self.confined(ground, cell, sheet.surface_height()) {
                self.sheets.remove_at(slot);
                let id = self.start_pool(ground, cell, sheet.volume);
                if let Some(pool) = self.pools.get_mut(&id) {
                    pool.load.add(sheet.load);
                }
                self.flood(ground, id, SPREAD_CELLS_PER_STEP);
                self.shed_over(ground, id, sheet.surface_height());
            } else {
                self.sheets.at_mut(slot).settled = Some(sheet.top());
            }
        }
        self.sheets.compact();
    }

    /// A sheet that stands no higher than a pool beside it or over it, or
    /// seed-derived water over it, whose surface covers its floor, lies under
    /// that water: a pool floods it and takes its water, seed-derived water
    /// takes it in as a joined cell, or as a pool first where the sheet is
    /// deeper than its cell.
    /// Water running down past a pool's rim is not under the pool: it is the
    /// pool's spill.
    fn submerge(&mut self, ground: &impl WaterGround, cell: WaterCell, sheet: Sheet) -> bool {
        let height = sheet.surface_height();
        let under = |level: f64| level > sheet.floor + FILM_METRES && height < level + MERGE_METRES;
        // Still water beside it anywhere up its depth, or over it.
        let top = sheet.top();
        let mut around = (cell.y..=top)
            .flat_map(|y| DIRECTIONS.map(|(dx, dz)| WaterCell::new(cell.x + dx, y, cell.z + dz)))
            .collect::<Vec<_>>();
        let over = WaterCell::new(cell.x, top + 1, cell.z);
        around.push(over);
        // Under open sky running water beside a pool trades with it through
        // its pipes: taken in, it would spread the pool over running water.
        let mut roofed = None;
        for beside in around {
            // Running water beside it is no still water.
            if self.sheets.covering(beside).is_some() {
                continue;
            }
            if let Some(&id) = self.owner.get(&beside) {
                if beside != over && !*roofed.get_or_insert_with(|| self.roofed(ground, cell)) {
                    continue;
                }
                let Some(pool) = self.pools.get_mut(&id) else {
                    continue;
                };
                if under(pool.level) && sheet.floor >= pool.rim {
                    pool.volume += sheet.volume;
                    pool.load.add(sheet.load);
                    pool.queue(cell, sheet.floor);
                    self.sheets.remove(cell);
                    return true;
                }
                continue;
            }
            // Seed-derived water takes in only what lies under it: water
            // beside a lake at its level stays running water and trades with
            // it through its faces. Joining whatever touches the lake would
            // let the lake spread itself along its contour faster than any
            // water flows.
            if beside != over {
                continue;
            }
            let seed = match self.joined.get(&beside) {
                Some(joined) => Some(joined.surface),
                None => self.ground.implicit(ground, beside),
            };
            let Some(seed) = seed else {
                continue;
            };
            let level = self.drawn(ground, seed).level;
            if under(level) {
                // Every cell the water fills under the seed-derived water
                // joins it, and only those: water beside them stays free.
                self.sheets.remove(cell);
                let mut held = 0.0;
                for y in cell.y..=top {
                    let filled = WaterCell::new(cell.x, y, cell.z);
                    if self.joined.contains_key(&filled) || self.owner.contains_key(&filled) {
                        continue;
                    }
                    let holds = held_in(filled, self.openings(ground, filled), level);
                    held += holds;
                    self.joined.insert(
                        filled,
                        Joined {
                            surface: seed,
                            held: holds,
                        },
                    );
                }
                self.cycle.add(seed.body, sheet.volume - held);
                self.settle_load((cell.x, cell.z), sheet.floor, sheet.load);
                // Free cells around it now border the seed-derived water.
                for y in cell.y..=top {
                    for neighbour in WaterCell::new(cell.x, y, cell.z).neighbours() {
                        if !self.owner.contains_key(&neighbour)
                            && !self.joined.contains_key(&neighbour)
                        {
                            self.remeasure(ground, neighbour);
                        }
                    }
                }
                return true;
            }
        }
        false
    }

    /// Whether water in the column of `cell` with its surface at `height`
    /// has ground over it: the cell above its surface is mostly shut, so the
    /// water fills a cave or a tunnel rather than lying under the sky.
    fn confined(&mut self, ground: &impl WaterGround, cell: WaterCell, height: f64) -> bool {
        let top = WaterCell::containing(DVec3::new(cell.centre().x, height, cell.centre().z));
        let above = WaterCell::new(cell.x, top.y.max(cell.y) + 1, cell.z);
        let open = self
            .openings(ground, above)
            .iter()
            .map(|&open| u32::from(open))
            .sum::<u32>();
        let whole = u32::try_from(super::WATER_CELL_EDGE_CELLS.pow(3)).expect("64 cells");
        open * 2 < whole
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

    /// A sheet taken in by a pool gives it its water.
    pub(super) fn absorb_sheet(&mut self, id: u32, cell: WaterCell) {
        if let Some(sheet) = self.sheets.remove(cell) {
            match self.pools.get_mut(&id) {
                Some(pool) => {
                    pool.volume += sheet.volume;
                    pool.load.add(sheet.load);
                }
                None => self.settle_load((cell.x, cell.z), sheet.floor, sheet.load),
            }
        }
    }

    /// A cell whose ground changed: sheets around it find their routes
    /// again, and water the ground now fills in its own cell rises to the
    /// cell above.
    pub(super) fn remeasure_sheet(&mut self, ground: &impl WaterGround, cell: WaterCell) {
        self.sheets.forget_routes(cell.x, cell.z);
        let Some(sheet) = self.sheets.get(cell).copied() else {
            return;
        };
        let openings = self.openings(ground, cell);
        match self.sheet_floor(ground, cell) {
            Some(floor) if held_in(cell, openings, f64::INFINITY) > 0.0 => {
                if let Some(sheet) = self.sheets.get_mut(cell) {
                    sheet.floor = floor;
                }
            }
            _ => {
                self.sheets.remove(cell);
                let placed = self.deposit_at(ground, cell.up(), sheet.volume);
                self.give_load(placed, sheet.load, (cell.x, cell.z, sheet.floor));
            }
        }
    }
}

/// How fast friction takes a pipe's flow, per second: bed friction by
/// Manning's law, `g n² |v| / h^(4/3)`, grows as water thins, so a film
/// barely creeps while a stream runs, and running water gathers into rills
/// along the lowest ground instead of spreading evenly.
fn friction(flux: f64, depth: f64) -> f64 {
    let speed = flux.abs() / (WATER_CELL_METRES * depth);
    FRICTION_PER_SECOND + GRAVITY * ROUGHNESS * ROUGHNESS * speed / depth.powf(4.0 / 3.0)
}

/// A sheet's current from the flow through its faces, in m/s.
pub(super) fn current(sheet: &Sheet, depth: f64) -> DVec2 {
    let across = WATER_CELL_METRES * depth.max(CLING_METRES);
    DVec2::new(sheet.flux[1] - sheet.flux[0], sheet.flux[3] - sheet.flux[2]) / across
}
