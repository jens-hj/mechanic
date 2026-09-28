//! Running water: thin sheets on the ground that run downhill.
//!
//! A sheet is water resting on the floor of a water cell. Each holds a
//! volume, and pipes to its four horizontal neighbours carry water between
//! them: the virtual-pipe model of shallow water (O'Brien and Hodgins; Mei,
//! Decaudin and Hu). A pipe's flow gathers speed with the difference in
//! surface height across it and loses it to friction, and no sheet sends
//! more than it holds, so water is conserved exactly and a sheet has a
//! current. A sheet climbs a step of one water cell and runs down a drop of
//! up to a metre as a steep chute; a taller drop is a lip it pours over.
//! Water reaching a pool or seed-derived water joins it. Water that stops in
//! a hollow deep enough becomes a pool.
//!
//! Sheets live in dense tiles ([`super::grid`]), and where each face of a
//! sheet leads is worked out once from the ground and kept until the ground
//! changes, so the pipes themselves are plain arithmetic.

use std::collections::BTreeSet;

use bevy_math::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use super::grid::Slot;
use super::jet::Launch;
use super::{
    CLING_METRES, End, FILM_METRES, GRAVITY, Joined, MERGE_METRES, WATER_CELL_METRES, WaterCell,
    WaterGround, WaterWorld,
};
use super::{floor_of, ground_height, held_in};
use crate::WaterSurface;

/// Horizontal area of one water cell, in square metres.
const CELL_AREA_M2: f64 = WATER_CELL_METRES * WATER_CELL_METRES;

/// Pipe cross-section over its length, in metres: the pipes behave like
/// water 20 cm deep, so waves cross a cell in a tenth of a second.
const PIPE_METRES: f64 = WATER_CELL_METRES;

/// How fast friction takes a pipe's flow whatever its depth, per second.
const FRICTION_PER_SECOND: f64 = 0.5;

/// Manning's roughness of the ground under running water, in s/m^(1/3):
/// short grass and bare soil.
const ROUGHNESS: f64 = 0.03;

/// Substeps per water step: pipes need shorter steps than pools.
const SUBSTEPS: u32 = 4;

/// Least water a sheet keeps; less evaporates at once, in m³.
const DRY_SHEET_M3: f64 = 1.0e-7;

/// Deepest drop, in water cells, that running water takes as a steep chute
/// rather than pouring over as a fall.
const CHUTE_CELLS: i32 = 5;

/// Steps a sheet must lie still in a hollow before it becomes a pool.
const STILL_STEPS: u8 = 10;

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
    /// Steps it has lain still in a hollow.
    still: u8,
    /// Where each face leads, once worked out.
    faces: Option<[Face; 4]>,
}

/// One sheet in a saved world.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SheetDoc {
    /// The cell.
    pub cell: WaterCell,
    /// Water held, in m³.
    pub volume_m3: f64,
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
    }
}

/// Where water leaving a sheet through one face goes this step.
#[derive(Clone, Copy, Debug)]
enum Route {
    /// Nowhere.
    Wall,
    /// Onto the floor of a neighbour column, if the water stands above it.
    Onto { slot: Slot, y: i32, floor: f64 },
    /// Into standing or seed-derived water with its surface here.
    Water(End, f64),
    /// Over a lip, pouring down from this cell.
    Lip(WaterCell),
}

/// Where water leaving `cell` towards a horizontal neighbour goes, for water
/// not yet running there.
#[derive(Clone, Copy, Debug)]
enum Target {
    /// Onto the floor of this cell.
    Cell(WaterCell),
    /// Into standing or seed-derived water with its surface here.
    Water(End, f64),
    /// Over a lip.
    Lip,
    /// Nowhere: ground stands above the water.
    Wall,
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
            })
            .collect()
    }

    /// Adds running water to a cell. A column already running at another
    /// height takes it into its own sheet.
    pub(super) fn add_sheet(&mut self, ground: &impl WaterGround, cell: WaterCell, volume: f64) {
        let slot = self.sheets.slot_or_insert(cell.x, cell.z);
        if !self.sheets.at(slot).present {
            let floor =
                ground_height(cell, self.openings(ground, cell)).unwrap_or_else(|| cell.bottom());
            self.sheets.place(slot, cell.y, floor);
        }
        self.sheets.at_mut(slot).volume += volume;
    }

    /// Height of a cell's running surface and its depth, if it has a floor.
    pub(super) fn sheet_surface(
        &mut self,
        ground: &impl WaterGround,
        cell: WaterCell,
    ) -> Option<(f64, f64)> {
        if let Some(sheet) = self.sheets.get(cell) {
            let depth = sheet.volume / CELL_AREA_M2;
            return Some((sheet.floor + depth, depth));
        }
        ground_height(cell, self.openings(ground, cell)).map(|floor| (floor, 0.0))
    }

    /// Running water at a cell: its surface and current, if it runs there.
    pub(super) fn running(&self, cell: WaterCell) -> Option<WaterSurface> {
        self.sheets.get(cell)?.surface()
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

    /// Where each face of a sheet in `cell` leads, from the ground alone.
    fn faces(&mut self, ground: &impl WaterGround, cell: WaterCell) -> [Face; 4] {
        DIRECTIONS.map(|(dx, dz)| {
            let beside = WaterCell::new(cell.x + dx, cell.y, cell.z + dz);
            if let Some(floor) = ground_height(beside, self.openings(ground, beside)) {
                if !self.drops(ground, beside) {
                    return Face::Onto {
                        y: beside.y,
                        floor,
                        above: beside.y,
                    };
                }
                // Its floor drops away: down to a metre the water runs on
                // down a steep chute, further it pours over.
                let mut below = beside;
                for _ in 0..CHUTE_CELLS {
                    below = below.below();
                    let Some(floor) = ground_height(below, self.openings(ground, below)) else {
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
            let above = beside.up();
            let open_above = floor_of(cell.up(), self.openings(ground, cell.up())).is_some();
            match ground_height(above, self.openings(ground, above)) {
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
        let (top, bottom) = match face {
            Face::Wall => return Route::Wall,
            Face::Onto { y, above, .. } => (above, y),
            Face::Lip { y, lowest } => (y, lowest),
        };
        for y in (bottom..=top).rev() {
            if let Some(Target::Water(end, level)) = self.standing(ground, WaterCell::new(x, y, z))
            {
                return Route::Water(end, level);
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

    /// Where water leaving `cell` towards a horizontal neighbour goes, with
    /// the water's surface at `height`.
    fn target(
        &mut self,
        ground: &impl WaterGround,
        cell: WaterCell,
        face: usize,
        height: f64,
    ) -> Target {
        let direction = DIRECTIONS[face];
        let ground_face = self.faces(ground, cell)[face];
        match self.route(ground, cell, direction, ground_face) {
            Route::Wall => Target::Wall,
            Route::Onto { floor, .. } if floor >= height => Target::Wall,
            Route::Onto { y, .. } => Target::Cell(WaterCell::new(
                cell.x + direction.0,
                y,
                cell.z + direction.1,
            )),
            Route::Water(end, level) => Target::Water(end, level),
            Route::Lip(_) => Target::Lip,
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

    /// Where each face of every sheet leads this step: the ground's routes,
    /// worked out again where the ground changed, met by standing water.
    fn routes(&mut self, ground: &impl WaterGround) -> Vec<(Slot, [Route; 4])> {
        let wet = self.sheets.wet();
        let mut routes = Vec::with_capacity(wet.len());
        for (slot, cell) in wet {
            let sheet = *self.sheets.at(slot);
            if sheet.volume / CELL_AREA_M2 < CLING_METRES {
                routes.push((slot, [Route::Wall; 4]));
                continue;
            }
            let faces = if let Some(faces) = sheet.faces {
                faces
            } else {
                let faces = self.faces(ground, cell);
                self.sheets.at_mut(slot).faces = Some(faces);
                faces
            };
            let mut face_routes = [Route::Wall; 4];
            for (route, (&direction, face)) in
                face_routes.iter_mut().zip(DIRECTIONS.iter().zip(faces))
            {
                *route = self.route(ground, cell, direction, face);
            }
            routes.push((slot, face_routes));
        }
        routes
    }

    /// Runs the sheets for `dt` seconds. Returns the water moved and the
    /// pools that received water.
    pub(super) fn step_sheets(
        &mut self,
        ground: &impl WaterGround,
        dt: f64,
    ) -> (f64, BTreeSet<u32>) {
        let routes = self.routes(ground);
        let sub = dt / f64::from(SUBSTEPS);
        let mut moved = 0.0;
        let mut into = Vec::new();
        let mut pours = Vec::new();
        let mut flows = vec![[0.0; 4]; routes.len()];
        for _ in 0..SUBSTEPS {
            // Every face's flow first, from the surfaces as they stand.
            for ((slot, faces), flow) in routes.iter().zip(&mut flows) {
                let sheet = self.sheets.at(*slot);
                let height = sheet.surface_height();
                let depth = (sheet.volume / CELL_AREA_M2).max(CLING_METRES);
                for (face, route) in faces.iter().enumerate() {
                    let beyond = match *route {
                        Route::Wall => None,
                        Route::Onto { floor, .. } if floor >= height => None,
                        Route::Onto { slot, floor, .. } => {
                            let other = self.sheets.at(slot);
                            Some(if other.present {
                                other.surface_height()
                            } else {
                                floor
                            })
                        }
                        Route::Water(_, level) => Some(level),
                        Route::Lip(over) => Some(over.bottom()),
                    };
                    flow[face] = beyond.map_or(0.0, |beyond| {
                        let driven =
                            sheet.flux[face] + sub * GRAVITY * PIPE_METRES * (height - beyond);
                        (driven / (1.0 + friction(sheet.flux[face], depth) * sub)).max(0.0)
                    });
                }
                let out = flow.iter().sum::<f64>() * sub;
                if out > 0.0 && out > sheet.volume {
                    let scale = sheet.volume.max(0.0) / out;
                    for face in flow.iter_mut() {
                        *face *= scale;
                    }
                }
            }
            // Then the water moves.
            for ((slot, faces), flow) in routes.iter().zip(&flows) {
                let sheet = self.sheets.at_mut(*slot);
                sheet.flux = *flow;
                let floor = sheet.floor;
                let depth = (sheet.volume / CELL_AREA_M2).max(CLING_METRES);
                for (face, route) in faces.iter().enumerate() {
                    let volume = flow[face] * sub;
                    if volume <= 0.0 {
                        continue;
                    }
                    self.sheets.at_mut(*slot).volume -= volume;
                    moved += volume;
                    match *route {
                        Route::Onto { slot, y, floor } => {
                            self.sheets.place(slot, y, floor).volume += volume;
                        }
                        Route::Water(end, _) => into.push((end, volume)),
                        Route::Lip(over) => {
                            // It leaves the lip at the speed it ran at.
                            let (dx, dz) = DIRECTIONS[face];
                            let away = DVec3::new(f64::from(dx), 0.0, f64::from(dz));
                            let speed = flow[face] / (WATER_CELL_METRES * depth);
                            let centre = over.centre();
                            pours.push((
                                Launch {
                                    lip: over,
                                    from: DVec3::new(centre.x, floor + depth, centre.z)
                                        - away * 0.4 * WATER_CELL_METRES,
                                    velocity: away * speed,
                                },
                                volume,
                            ));
                        }
                        Route::Wall => {}
                    }
                }
            }
        }
        let mut fed = BTreeSet::new();
        for (end, volume) in into {
            if let End::Pool(id) = end {
                fed.insert(id);
            }
            self.deposit_end(ground, end, volume);
        }
        for (launch, volume) in pours {
            self.pour(launch, volume);
        }
        (moved, fed)
    }

    /// Sheets that have lain still in a hollow, deep enough to stand, become
    /// pools; sheets that dried up evaporate.
    pub(super) fn settle_sheets(&mut self, ground: &impl WaterGround) {
        for (slot, cell) in self.sheets.wet() {
            let sheet = *self.sheets.at(slot);
            if !sheet.present || sheet.y != cell.y {
                continue;
            }
            if sheet.volume < DRY_SHEET_M3 {
                self.sheets.remove_at(slot);
                self.cycle.evaporate(sheet.volume.max(0.0));
                continue;
            }
            if self.submerge(ground, cell, sheet) {
                continue;
            }
            let (height, depth) = (sheet.surface_height(), sheet.volume / CELL_AREA_M2);
            let draining = sheet.flux.iter().sum::<f64>() > 0.01 * sheet.volume;
            // Water brimming over its cell's top in a hollow fills a hole: it
            // stands there as a pool, however it sloshes.
            let over = height > cell.bottom() + WATER_CELL_METRES;
            let hollow = depth >= FILM_METRES
                && (over || !draining)
                && self.sheet_in_hollow(ground, cell, sheet);
            let still = if hollow && !draining {
                sheet.still + 1
            } else {
                0
            };
            let brimming = hollow && over;
            if still >= STILL_STEPS || brimming {
                self.sheets.remove_at(slot);
                self.start_pool(ground, cell, sheet.volume);
            } else {
                self.sheets.at_mut(slot).still = still;
            }
        }
        self.sheets.compact();
    }

    /// A sheet that stands no higher than still water beside it, whose
    /// surface covers its floor, lies under that water: a pool floods it and
    /// takes its water, seed-derived water takes it in as a joined cell.
    /// Water running down past a pool's rim is not under the pool: it is the
    /// pool's spill.
    fn submerge(&mut self, ground: &impl WaterGround, cell: WaterCell, sheet: Sheet) -> bool {
        let height = sheet.surface_height();
        let under = |level: f64| level > sheet.floor + FILM_METRES && height < level + MERGE_METRES;
        for (dx, dz) in DIRECTIONS {
            let beside = WaterCell::new(cell.x + dx, cell.y, cell.z + dz);
            // Running water beside it is no still water.
            if self.sheets.get(beside).is_some() {
                continue;
            }
            if let Some(&id) = self.owner.get(&beside) {
                let Some(pool) = self.pools.get_mut(&id) else {
                    continue;
                };
                if under(pool.level) && sheet.floor >= pool.rim {
                    pool.volume += sheet.volume;
                    pool.queue(cell, sheet.floor);
                    self.sheets.remove(cell);
                    return true;
                }
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
                self.sheets.remove(cell);
                let held = held_in(cell, self.openings(ground, cell), level);
                self.joined.insert(
                    cell,
                    Joined {
                        surface: seed,
                        held,
                    },
                );
                self.cycle.add(seed.body, sheet.volume - held);
                // Free cells around it now border the seed-derived water.
                for neighbour in cell.neighbours() {
                    if !self.owner.contains_key(&neighbour) && !self.joined.contains_key(&neighbour)
                    {
                        self.remeasure(ground, neighbour);
                    }
                }
                return true;
            }
        }
        false
    }

    /// Whether no face of a sheet leads lower than its floor.
    fn sheet_in_hollow(
        &mut self,
        ground: &impl WaterGround,
        cell: WaterCell,
        sheet: Sheet,
    ) -> bool {
        let Some(faces) = sheet.faces else {
            return self.in_hollow(ground, cell, sheet.surface_height());
        };
        DIRECTIONS.iter().zip(faces).all(|(&direction, face)| {
            match self.route(ground, cell, direction, face) {
                Route::Onto { floor, .. } => floor >= sheet.floor,
                Route::Water(_, level) => level >= sheet.floor,
                Route::Lip(_) => false,
                Route::Wall => true,
            }
        })
    }

    /// Whether no neighbour of a cell lies lower than its floor, so water
    /// there stands rather than runs.
    fn in_hollow(&mut self, ground: &impl WaterGround, cell: WaterCell, height: f64) -> bool {
        let Some(floor) = floor_of(cell, self.openings(ground, cell)) else {
            return false;
        };
        (0..DIRECTIONS.len()).all(|face| match self.target(ground, cell, face, height) {
            Target::Cell(other) => {
                floor_of(other, self.openings(ground, other)).is_none_or(|other| other >= floor)
            }
            Target::Water(_, level) => level >= floor,
            Target::Lip => false,
            Target::Wall => true,
        })
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
        if let Some(sheet) = self.sheets.remove(cell)
            && let Some(pool) = self.pools.get_mut(&id)
        {
            pool.volume += sheet.volume;
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
        match ground_height(cell, openings) {
            Some(floor) if held_in(cell, openings, f64::INFINITY) > 0.0 => {
                if let Some(sheet) = self.sheets.get_mut(cell) {
                    sheet.floor = floor;
                }
            }
            _ => {
                self.sheets.remove(cell);
                self.deposit_at(ground, cell.up(), sheet.volume);
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
fn current(sheet: &Sheet, depth: f64) -> DVec2 {
    let across = WATER_CELL_METRES * depth.max(CLING_METRES);
    DVec2::new(sheet.flux[1] - sheet.flux[0], sheet.flux[3] - sheet.flux[2]) / across
}
