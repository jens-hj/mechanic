//! Stored water in the world: stepping it, keeping it in step with the
//! ground, loading and saving it.

use std::sync::Arc;

use bevy::prelude::*;
use mechanic_world::{
    BrickCoord, TerrainField, TerrainOctree, TerrainWater, WaterWorld, WorldStore,
};

use super::{WorldListPhase, WorldListState, WorldRuntime};

/// Seconds of water per step: water moves at 20 Hz.
const WATER_STEP_SECONDS: f64 = 0.05;

/// Water steps one frame may run; a slower frame lets water fall behind
/// rather than slow further.
const MAX_STEPS_PER_FRAME: u32 = 3;

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

/// Tells the water the ground changed in these bricks.
pub(super) fn ground_changed(
    runtime: &mut WorldRuntime,
    bricks: impl IntoIterator<Item = BrickCoord>,
) {
    let ground = TerrainWater {
        field: &runtime.field,
        edits: &runtime.edits,
    };
    runtime.water.terrain_changed(&ground, bricks);
}

/// Runs the world's stored water and publishes a view of it.
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
        .min(f64::from(MAX_STEPS_PER_FRAME) * WATER_STEP_SECONDS);
    let mut stepped = false;
    while runtime.water_seconds >= WATER_STEP_SECONDS {
        runtime.water_seconds -= WATER_STEP_SECONDS;
        let started = std::time::Instant::now();
        let ground = TerrainWater {
            field: &runtime.field,
            edits: &runtime.edits,
        };
        let step = runtime.water.step(&ground, WATER_STEP_SECONDS);
        crate::performance_capture::record("water_step", || {
            serde_json::json!({
                "duration_ms": started.elapsed().as_secs_f64() * 1000.0,
                "moved_m3": step.moved_m3,
                "pools": step.pools,
                "cells": step.cells,
                "falls": step.falls.len(),
                "sheet_cells": step.sheet_cells,
                "phases_ms": {
                    "flood": step.phases.flood_ms,
                    "exchange": step.phases.exchange_ms,
                    "sheets": step.phases.sheets_ms,
                    "jets": step.phases.jets_ms,
                    "joins": step.phases.joins_ms,
                    "settle": step.phases.settle_ms,
                },
                "falling_m3": runtime.water.ledger().falling_m3,
                "ledger_m3": runtime.water.ledger().total(),
            })
        });
        if step.moved_m3 > 0.0 {
            runtime.autosave.mutate(runtime.clock);
        }
        runtime.water_falls = step.falls;
        stepped = true;
    }
    if stepped {
        runtime.water_surfaces = Arc::new(runtime.water.surfaces(runtime.field.clone()));
        runtime.water_revision = runtime.water_revision.wrapping_add(1);
    }
}
