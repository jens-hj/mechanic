//! Stored water in the world: stepping it, keeping it in step with the
//! ground, loading and saving it.

use std::collections::HashMap;
use std::sync::Arc;

use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future};
use mechanic_world::{
    BrickCoord, SurfaceTile, TerrainField, TerrainOctree, TerrainWater, WaterCell, WaterFall,
    WaterLedger, WaterStep, WaterSurface, WaterSurfaces, WaterWorld, WetGround, WorldStore,
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
    /// Columns of running water at the level of the lake over them.
    pub(crate) meeting: Vec<(i32, i32)>,
    /// Wet ground.
    pub(crate) wet: Vec<WetGround>,
}

/// One batch of water steps done on the worker.
struct WaterBatch {
    world: WaterWorld,
    view: WaterView,
    surfaces: WaterSurfaces,
    falls: Vec<WaterFall>,
    steps: Vec<(f64, WaterStep)>,
    ledger: WaterLedger,
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
}

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
        }
    }

    /// The water itself, waiting for a running batch to finish.
    pub(crate) fn world_mut(&mut self) -> &mut WaterWorld {
        if let Some(task) = self.task.take() {
            let mut batch = block_on(task);
            self.world = Some(std::mem::take(&mut batch.world));
            self.finished = Some(batch);
        }
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
    WaterView {
        surface: world.surface_tiles(ground, drawn),
        joined: world.joined_cells(ground),
        meeting: world.meeting_columns(ground),
        wet: world.wet_ground(),
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
    let drawn = runtime.water.drawn.clone();
    runtime.water.task = Some(AsyncComputeTaskPool::get().spawn(async move {
        let ground = TerrainWater {
            field: &field,
            edits: &edits,
        };
        world.terrain_changed(&ground, bricks);
        let mut done = Vec::with_capacity(steps as usize);
        let mut falls = Vec::new();
        for _ in 0..steps {
            let started = std::time::Instant::now();
            let mut step = world.step(&ground, WATER_STEP_SECONDS);
            falls = std::mem::take(&mut step.falls);
            done.push((started.elapsed().as_secs_f64() * 1000.0, step));
        }
        WaterBatch {
            view: view(&mut world, &ground, &drawn),
            surfaces: world.surfaces(field.clone()),
            ledger: world.ledger(),
            world,
            falls,
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
                    "jets": step.phases.jets_ms,
                    "joins": step.phases.joins_ms,
                    "settle": step.phases.settle_ms,
                },
                "falls": batch.falls.len(),
                "falling_m3": batch.ledger.falling_m3,
                "ledger_m3": batch.ledger.total(),
            })
        });
    }
    if moved {
        runtime.autosave.mutate(runtime.clock);
    }
    runtime.water.drawn = fingerprints(&batch.view);
    runtime.water.view = batch.view;
    runtime.water_falls = batch.falls;
    runtime.water_surfaces = Arc::new(batch.surfaces);
    runtime.water_revision = runtime.water_revision.wrapping_add(1);
}
