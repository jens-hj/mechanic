//! Stored water in the world: stepping it, keeping it in step with the
//! ground, loading and saving it.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use bevy::prelude::*;
use bevy::tasks::futures::check_ready;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future};
use mechanic_world::{
    BrickCoord, SedimentApplied, SedimentChange, SurfaceTile, TerrainBrick, TerrainEditOutcome,
    TerrainField, TerrainOctree, TerrainWater, WaterBody, WaterCell, WaterLedger, WaterShift,
    WaterStep, WaterSurface, WaterSurfaces, WaterWorld, WetGround, WorldCell, WorldStore,
};

use super::{WorldListPhase, WorldListState, WorldRuntime};

/// Seconds of water per step: water moves at 20 Hz.
const WATER_STEP_SECONDS: f64 = 0.05;

/// Water steps one batch may run; water further behind than this slows down
/// rather than catch up.
const MAX_STEPS_PER_BATCH: u32 = 3;

/// Whether the world's water is switched on.
pub(crate) fn water_enabled() -> bool {
    crate::env::text(crate::env::WATER).as_deref() != Some("off")
}

/// Loads a world's stored water and finds where its seed-derived water
/// pours into edited ground.
///
/// # Errors
/// Returns the store's message when the water cannot be read.
pub(super) fn load_water(
    store: &WorldStore,
    world_name: &str,
    field: &TerrainField,
    edits: &TerrainOctree,
) -> Result<WaterWorld, String> {
    let doc = store
        .load_water(world_name)
        .map_err(|error| error.to_string())?;
    let ground = TerrainWater { field, edits };
    let mut water = WaterWorld::from_doc(&ground, &doc);
    let bricks = edits
        .snapshot()
        .bricks()
        .map(mechanic_world::TerrainBrick::coordinate)
        .collect::<Vec<_>>();
    water.terrain_changed(&ground, bricks);
    Ok(water)
}

/// Tells the water the ground changed in these bricks. The water takes the
/// change in before its next step.
pub(super) fn ground_changed(
    runtime: &mut WorldRuntime,
    bricks: impl IntoIterator<Item = BrickCoord>,
) {
    runtime.water.pending.extend(bricks);
}

/// What the water looked like after its last batch of steps, to draw.
#[derive(Default)]
pub(crate) struct WaterView {
    /// The stored water's surface, tile by tile: tiles unchanged since the
    /// batch before carry no mesh.
    pub(crate) surface: Vec<SurfaceTile>,
    /// Cells joined to seed-derived water, with its surface as it stands.
    pub(crate) joined: Vec<(WaterCell, WaterSurface)>,
    /// Columns the stored water's surface draws within a lake's reach,
    /// which the lake's own sheet leaves out.
    pub(crate) owned: Vec<(i32, i32)>,
    /// Wet ground.
    pub(crate) wet: Vec<WetGround>,
    /// Where each moved lake and river is drawn.
    pub(crate) shifts: BTreeMap<WaterBody, WaterShift>,
}

/// What the ground did with the changes the water asked for.
enum GroundAnswer {
    /// What each change did, in order.
    Applied(Vec<SedimentApplied>),
    /// Terrain editing is off: nothing changed.
    Refused,
}

/// One batch of water steps done on the worker.
struct WaterBatch {
    world: WaterWorld,
    view: WaterView,
    surfaces: WaterSurfaces,
    steps: Vec<(f64, WaterStep)>,
    ledger: WaterLedger,
    /// Changes to the ground the water asks for.
    asks: Vec<SedimentChange>,
    /// The untouched bricks those changes may touch, sampled on the worker.
    untouched: Vec<TerrainBrick>,
}

/// The world's stored water, stepped on a worker so a slow step never holds
/// up a frame: while a batch runs the worker owns the water, and the frame
/// draws what the last batch published.
pub(crate) struct WaterRunner {
    world: Option<WaterWorld>,
    task: Option<Task<WaterBatch>>,
    /// A batch finished while the water was waited for, not yet published.
    finished: Option<WaterBatch>,
    /// Bricks whose ground changed since the water last looked.
    pending: Vec<BrickCoord>,
    view: WaterView,
    /// What each surface tile showed when last meshed.
    drawn: HashMap<(i32, i32), u64>,
    /// Changes to the ground the water asked for, not yet made.
    asks: Vec<SedimentChange>,
    /// The untouched bricks they may touch, sampled ahead.
    untouched: Vec<TerrainBrick>,
    /// What the ground did with the water's last asks, until the water is
    /// home to hear it.
    answer: Option<GroundAnswer>,
    /// Terrain cells sediment changed since the water last looked.
    sediment_cells: Vec<WorldCell>,
    /// The ground changing as the water asked, on a worker.
    sediment_task: Option<Task<SedimentEdit>>,
}

/// The ground changed as the water asked: the terrain with the change, what
/// it did, and what each ask did.
type SedimentEdit = (TerrainOctree, TerrainEditOutcome, Vec<SedimentApplied>);

impl WaterRunner {
    /// Water to run, with its view as it stands.
    pub(crate) fn new(mut world: WaterWorld, field: &TerrainField, edits: &TerrainOctree) -> Self {
        let ground = TerrainWater { field, edits };
        let view = view(&mut world, &ground, &HashMap::new());
        let drawn = fingerprints(&view);
        Self {
            world: Some(world),
            task: None,
            finished: None,
            pending: Vec::new(),
            view,
            drawn,
            asks: Vec::new(),
            untouched: Vec::new(),
            answer: None,
            sediment_cells: Vec::new(),
            sediment_task: None,
        }
    }

    /// Whether the ground is changing as the water asked: no other edit
    /// may start meanwhile, or one would overwrite the other.
    pub(super) const fn sediment_busy(&self) -> bool {
        self.sediment_task.is_some()
    }

    /// Keeps what the ground did with the water's asks for the water, and
    /// tells it at once if it is home.
    fn take_answer(&mut self, outcome: &TerrainEditOutcome, applied: Vec<SedimentApplied>) {
        self.sediment_cells
            .extend_from_slice(&outcome.sediment_cells);
        self.answer = Some(GroundAnswer::Applied(applied));
        self.hear_answer();
    }

    /// Tells the water, if it is home, what the ground did with its asks.
    fn hear_answer(&mut self) {
        let Some(world) = self.world.as_mut() else {
            return;
        };
        match self.answer.take() {
            Some(GroundAnswer::Applied(applied)) => world.sediment_applied(&applied),
            Some(GroundAnswer::Refused) => world.sediment_refused(),
            None => {}
        }
    }

    /// The water itself, waiting for a running batch to finish.
    pub(crate) fn world_mut(&mut self) -> &mut WaterWorld {
        if let Some(task) = self.task.take() {
            let mut batch = block_on(task);
            self.world = Some(std::mem::take(&mut batch.world));
            self.finished = Some(batch);
        }
        // Material the ground gave up or took is in the water's books before
        // anything reads or saves them.
        self.hear_answer();
        self.world
            .as_mut()
            .expect("the water is home when no batch runs")
    }

    /// The view the last batch published.
    pub(crate) const fn view(&self) -> &WaterView {
        &self.view
    }
}

/// The view of the water to draw, meshing only the surface tiles that
/// changed since `drawn`.
fn view(
    world: &mut WaterWorld,
    ground: &impl mechanic_world::WaterGround,
    drawn: &HashMap<(i32, i32), u64>,
) -> WaterView {
    let surface = world.surface_tiles(ground, drawn);
    WaterView {
        surface: surface.tiles,
        joined: world.joined_cells(),
        owned: surface.owned,
        wet: world.wet_ground(),
        shifts: surface.shifts,
    }
}

/// What each surface tile of a view shows.
fn fingerprints(view: &WaterView) -> HashMap<(i32, i32), u64> {
    view.surface
        .iter()
        .map(|tile| (tile.key, tile.fingerprint))
        .collect()
}

/// Runs the world's stored water on a worker and publishes a view of it
/// when a batch finishes. Water that falls behind slows down; the frame
/// does not wait for it.
pub(super) fn step_water(
    mut runtime: ResMut<WorldRuntime>,
    list: Res<WorldListState>,
    time: Res<Time>,
) {
    if list.phase() != WorldListPhase::Playing || !water_enabled() {
        return;
    }
    let runtime = &mut *runtime;
    runtime.water_seconds = (runtime.water_seconds + time.delta_secs_f64())
        .min(f64::from(MAX_STEPS_PER_BATCH) * WATER_STEP_SECONDS);
    let done = runtime.water.finished.take().or_else(|| {
        let task = runtime.water.task.as_mut()?;
        let batch = block_on(future::poll_once(task))?;
        runtime.water.task = None;
        Some(batch)
    });
    if let Some(mut batch) = done {
        if runtime.water.world.is_none() {
            runtime.water.world = Some(std::mem::take(&mut batch.world));
        }
        publish(runtime, batch);
        runtime.water.hear_answer();
    }
    if runtime.water.task.is_some() || runtime.water_seconds < WATER_STEP_SECONDS {
        return;
    }
    let Some(mut world) = runtime.water.world.take() else {
        return;
    };
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a few steps"
    )]
    let steps = (runtime.water_seconds / WATER_STEP_SECONDS).floor() as u32;
    runtime.water_seconds -= f64::from(steps) * WATER_STEP_SECONDS;
    let field = runtime.field.clone();
    let edits = runtime.edits.snapshot();
    let bricks = std::mem::take(&mut runtime.water.pending);
    let cells = std::mem::take(&mut runtime.water.sediment_cells);
    let drawn = runtime.water.drawn.clone();
    runtime.water.task = Some(AsyncComputeTaskPool::get().spawn(async move {
        let ground = TerrainWater {
            field: &field,
            edits: &edits,
        };
        world.terrain_changed(&ground, bricks);
        world.ground_cells_changed(&ground, &cells);
        let mut done = Vec::with_capacity(steps as usize);
        for _ in 0..steps {
            let started = std::time::Instant::now();
            let step = world.step(&ground, WATER_STEP_SECONDS);
            done.push((started.elapsed().as_secs_f64() * 1000.0, step));
        }
        let asks = world.sediment_requests();
        // Sampling ground the seed made is most of the cost of changing it:
        // done here, it stays off the frame.
        let mut bricks = asks
            .iter()
            .flat_map(SedimentChange::bricks)
            .filter(|&brick| edits.brick(brick).is_none())
            .collect::<Vec<_>>();
        bricks.sort_unstable();
        bricks.dedup();
        WaterBatch {
            view: view(&mut world, &ground, &drawn),
            surfaces: world.surfaces(field.clone()),
            ledger: world.ledger(),
            untouched: bricks
                .into_iter()
                .map(|brick| TerrainBrick::untouched(&field, brick))
                .collect(),
            asks,
            world,
            steps: done,
        }
    }));
}

/// Shows what a finished batch did.
fn publish(runtime: &mut WorldRuntime, batch: WaterBatch) {
    let mut moved = false;
    for (duration_ms, step) in &batch.steps {
        moved |= step.moved_m3 > 0.0;
        crate::performance_capture::record("water_step", || {
            serde_json::json!({
                "duration_ms": duration_ms,
                "moved_m3": step.moved_m3,
                "pools": step.pools,
                "cells": step.cells,
                "sheet_cells": step.sheet_cells,
                "phases_ms": {
                    "flood": step.phases.flood_ms,
                    "exchange": step.phases.exchange_ms,
                    "sheets": step.phases.sheets_ms,
                    "joins": step.phases.joins_ms,
                    "settle": step.phases.settle_ms,
                },
                "ledger_m3": batch.ledger.total(),
            })
        });
    }
    if moved {
        runtime.autosave.mutate(runtime.clock);
    }
    runtime.water.asks.extend(batch.asks);
    runtime.water.untouched.extend(batch.untouched);
    runtime.water.drawn = fingerprints(&batch.view);
    runtime.water.view = batch.view;
    runtime.water_surfaces = Arc::new(batch.surfaces);
    runtime.water_revision = runtime.water_revision.wrapping_add(1);
}

impl WorldRuntime {
    /// Changes the ground as the water asked, as one ordinary terrain edit
    /// made on a worker: running water wears the ground and lays sediment
    /// back on it. The water measures again only the cells that changed,
    /// not their whole bricks.
    pub(super) fn lay_sediment(&mut self) {
        if let Some(done) = self.water.sediment_task.as_mut().and_then(check_ready) {
            self.water.sediment_task = None;
            self.commit_sediment(done);
        }
        if self.water.sediment_busy()
            || self.water.asks.is_empty()
            || self.terrain_edit_task.is_some()
        {
            return;
        }
        if self.terrain_edit_error.is_some() {
            self.water.asks.clear();
            self.water.untouched.clear();
            self.water.answer = Some(GroundAnswer::Refused);
            self.water.hear_answer();
            return;
        }
        let asks = std::mem::take(&mut self.water.asks);
        let untouched = std::mem::take(&mut self.water.untouched);
        let mut terrain = self.edits.clone();
        let field = Arc::clone(&self.field);
        self.water.sediment_task = Some(AsyncComputeTaskPool::get().spawn(async move {
            let (outcome, applied) = terrain.exchange_sediment(&field, &asks, untouched);
            (terrain, outcome, applied)
        }));
    }

    /// Waits for the ground to finish changing as the water asked.
    pub(super) fn finish_sediment(&mut self) {
        if let Some(task) = self.water.sediment_task.take() {
            let done = block_on(task);
            self.commit_sediment(done);
        }
    }

    fn commit_sediment(&mut self, (terrain, outcome, applied): SedimentEdit) {
        crate::performance_capture::record("sediment", || {
            serde_json::json!({
                "cells": outcome.sediment_cells.len(),
                "bricks": outcome.changed_bricks,
                "quanta_given_up": outcome.quanta_given_up,
                "quanta_taken_back": outcome.quanta_taken_back,
            })
        });
        // Loose ground sediment cut or laid slides like any other.
        for &cell in &outcome.sediment_cells {
            self.slump.disturb(cell);
        }
        self.water.take_answer(&outcome, applied);
        let pending = self.water.pending.len();
        super::brush::commit_terrain_edit_result(
            self,
            super::brush::TerrainEditTaskResult {
                terrain,
                outcomes: vec![outcome],
                strokes: Vec::new(),
                elapsed_ms: 0.0,
            },
        );
        self.water.pending.truncate(pending);
    }
}

#[cfg(test)]
mod tests {
    use bevy::math::DVec3;
    use mechanic_world::{
        ErosionConfig, TerrainField, TerrainOctree, TerrainWater, WaterWorld, WorldPosition,
        WorldSeed,
    };

    use super::WaterRunner;

    /// Changes the ground as the water asked, at once, as the app does on a
    /// worker. Returns the edit, if any.
    fn answer_asks(
        runner: &mut WaterRunner,
        terrain: &mut TerrainOctree,
        field: &TerrainField,
    ) -> Option<mechanic_world::TerrainEditOutcome> {
        if runner.asks.is_empty() {
            return None;
        }
        let asks = std::mem::take(&mut runner.asks);
        let untouched = std::mem::take(&mut runner.untouched);
        let (outcome, applied) = terrain.exchange_sediment(field, &asks, untouched);
        runner.take_answer(&outcome, applied);
        Some(outcome)
    }

    #[test]
    fn ground_the_water_wears_changes_and_the_books_balance() {
        let field = TerrainField::new(WorldSeed(42));
        let spawn = field.safe_spawn().0;
        let mut terrain = TerrainOctree::default();
        // A pit dug at spawn, water poured in hard enough to wear its floor.
        let floor = field
            .topmost_surface(spawn.x, spawn.z)
            .expect("ground at spawn");
        let centre = DVec3::new(spawn.x, floor - 0.3, spawn.z);
        let bricks = terrain
            .excavate_sphere(&field, WorldPosition(centre), 0.6)
            .expect("a pit")
            .changed_brick_coordinates()
            .to_vec();
        let mut world = WaterWorld::new();
        world.set_erosion(ErosionConfig { speed: 3_000.0 });
        world.terrain_changed(
            &TerrainWater {
                field: &field,
                edits: &terrain,
            },
            bricks,
        );
        let mut runner = WaterRunner::new(world, &field, &terrain);
        let before = terrain.clone();
        let mut changed = 0;
        for _ in 0..400 {
            let ground = TerrainWater {
                field: &field,
                edits: &terrain,
            };
            let world = runner.world_mut();
            world.deposit(&ground, centre + DVec3::new(0.3, 0.6, 0.0), 0.002);
            world.step(&ground, 0.05);
            let asks = world.sediment_requests();
            runner.asks.extend(asks);
            if let Some(outcome) = answer_asks(&mut runner, &mut terrain, &field) {
                changed += outcome.sediment_cells.len();
            }
            let books = runner.world_mut().sediment_ledger();
            assert!(
                books.unaccounted().abs() < 1.0e-6 * books.eroded.max(1.0),
                "sediment made or lost: {books:?}"
            );
        }
        let books = runner.world_mut().sediment_ledger();
        assert!(
            changed > 0 && books.eroded > 0.0,
            "the water wore nothing: {books:?}"
        );
        assert!(terrain != before, "the ground never changed");
        assert!(
            !runner.sediment_cells.is_empty(),
            "the water was told of no change"
        );
    }
}
