//! The terrain brush: strokes batched into background edits and committed in order.

use super::list::{WorldListPhase, WorldListState};
use super::streaming::TerrainMeshTask;
use super::{
    ButtonInput, Component, GlobalTransform, Query, Res, ResMut, Result, Single, String, ToOwned,
    ToString, Transform, Vec, Vec3, Visibility, With, WorldDiagnostics, WorldRuntime, format, vec,
};
use crate::camera::MainCamera;
use crate::controls::GameAction;
use crate::editor::state::EditorState;
use crate::hotbar::{MainTool, MatterMode, SelectedTerrainMaterial, SelectedTool};
use bevy::tasks::futures::check_ready;
use bevy::tasks::{AsyncComputeTaskPool, block_on};
use mechanic_world::{
    ActiveTerrainScene, TerrainEditBatch, TerrainEditOutcome, TerrainField, TerrainMaterial,
    TerrainOctree, WorldPosition,
};
use std::collections::BTreeSet;
use std::sync::Arc;

pub(super) const MAX_PENDING_TERRAIN_EDITS: usize = 4_096;

pub(super) const TERRAIN_EDIT_BATCH_SIZE: usize = 64;

#[derive(Component)]
pub(super) struct BrushPreview;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct TerrainEditCommand {
    pub(super) centre: WorldPosition,
    pub(super) radius_metres: f64,
    pub(super) previous: Option<(WorldPosition, f64)>,
    pub(super) operation: TerrainEditOperation,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum TerrainEditOperation {
    Compress(mechanic_world::SoilCompression),
    Add(TerrainMaterial),
    Remove,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct TerrainStrokeSample {
    pub(super) centre: WorldPosition,
    pub(super) radius_metres: f64,
    pub(super) operation: TerrainEditOperation,
}

pub(super) struct TerrainEditTaskResult {
    pub(super) terrain: TerrainOctree,
    pub(super) outcomes: Vec<TerrainEditOutcome>,
    pub(super) elapsed_ms: f64,
}

pub(super) fn select_and_size_brush(
    actions: Res<ButtonInput<GameAction>>,
    mut runtime: ResMut<WorldRuntime>,
    mut editor: ResMut<EditorState>,
    selection: Res<SelectedTool>,
) {
    let active = selection.tool == Some(MainTool::MatterManipulator)
        && selection.matter_mode == MatterMode::Terrain;
    if !active {
        runtime.last_brush_edit = None;
        return;
    }
    let previous = runtime.brush_radius;
    if actions.just_pressed(GameAction::ZoomOut) {
        runtime.brush_radius = (runtime.brush_radius - 0.05).max(0.10);
    }
    if actions.just_pressed(GameAction::ZoomIn) {
        runtime.brush_radius = (runtime.brush_radius + 0.05).min(2.00);
    }
    if (runtime.brush_radius - previous).abs() > f64::EPSILON {
        editor.feedback = Some(format!(
            "Terrain brush — {:.2} m radius",
            runtime.brush_radius,
        ));
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "independent Bevy resources own brush input and output"
)]
pub(super) fn use_brush(
    actions: Res<ButtonInput<GameAction>>,
    camera: Single<&GlobalTransform, With<MainCamera>>,
    mut preview: Single<(&mut Transform, &mut Visibility), With<BrushPreview>>,
    list: Res<WorldListState>,
    mut runtime: ResMut<WorldRuntime>,
    mut editor: ResMut<EditorState>,
    selection: Res<SelectedTool>,
    material: Res<SelectedTerrainMaterial>,
) {
    if runtime.material_publication_pending() {
        return;
    }
    if list.phase() != WorldListPhase::Playing {
        runtime.last_brush_edit = None;
        *preview.1 = Visibility::Hidden;
        return;
    }
    if selection.tool != Some(MainTool::MatterManipulator)
        || selection.matter_mode != MatterMode::Terrain
    {
        runtime.last_brush_edit = None;
        *preview.1 = Visibility::Hidden;
        return;
    }
    if runtime.load_error.is_some() || runtime.terrain_edit_error.is_some() {
        runtime.last_brush_edit = None;
        *preview.1 = Visibility::Hidden;
        editor.feedback = runtime.load_error.as_ref().map_or_else(
            || runtime.terrain_edit_error.clone(),
            |error| Some(format!("Terrain editing disabled: {error}")),
        );
        return;
    }
    let global_origin = WorldPosition(runtime.floating_origin.0 + camera.translation().as_dvec3());
    let direction = camera.forward().as_vec3().as_dvec3();
    let scene = ActiveTerrainScene {
        chunks: &runtime.active_terrain,
        ready_faces: &runtime.active_terrain_ready_faces,
        spatial_index: &runtime.active_terrain_index,
    };
    let Some(hit) = scene.raycast(global_origin, direction, 24.0) else {
        runtime.last_brush_edit = None;
        *preview.1 = Visibility::Hidden;
        return;
    };
    preview.0.translation = hit.position.relative_to(runtime.floating_origin);
    preview.0.scale = Vec3::splat(runtime.brush_radius as f32);
    *preview.1 = Visibility::Visible;
    let operation = if actions.pressed(GameAction::Secondary) {
        Some(TerrainEditOperation::Remove)
    } else if actions.pressed(GameAction::Primary) {
        Some(TerrainEditOperation::Add(material.0))
    } else {
        None
    };
    let Some(operation) = operation else {
        runtime.last_brush_edit = None;
        return;
    };
    let radius = runtime.brush_radius;
    let previous = runtime
        .last_brush_edit
        .filter(|previous| previous.operation == operation);
    let commands = terrain_edit_commands(previous, hit.position, radius, operation);
    if runtime
        .pending_terrain_edits
        .len()
        .saturating_add(commands.len())
        > MAX_PENDING_TERRAIN_EDITS
    {
        let error = format!(
            "Terrain editing paused: the {MAX_PENDING_TERRAIN_EDITS}-sample brush queue is full"
        );
        runtime.last_brush_edit = None;
        runtime.terrain_edit_error = Some(error.clone());
        editor.feedback = Some(error);
        return;
    }
    runtime.pending_terrain_edits.extend(commands);
    runtime.last_brush_edit = Some(TerrainStrokeSample {
        centre: hit.position,
        radius_metres: radius,
        operation,
    });
}

pub(super) fn terrain_edit_commands(
    previous: Option<TerrainStrokeSample>,
    centre: WorldPosition,
    radius_metres: f64,
    operation: TerrainEditOperation,
) -> Vec<TerrainEditCommand> {
    const SAMPLE_INTERVAL_METRES: f64 = 0.05;
    let Some(previous) = previous else {
        return vec![TerrainEditCommand {
            centre,
            radius_metres,
            previous: None,
            operation,
        }];
    };
    let previous_centre = previous.centre;
    let previous_radius = previous.radius_metres;
    let distance = previous_centre.0.distance(centre.0);
    if distance <= f64::EPSILON && (previous_radius - radius_metres).abs() <= f64::EPSILON {
        return Vec::new();
    }
    let segment_count = usize::try_from((distance / SAMPLE_INTERVAL_METRES).ceil() as u64)
        .unwrap_or(usize::MAX)
        .max(1);
    let mut commands = Vec::with_capacity(segment_count);
    let mut last = (previous_centre, previous_radius);
    for index in 1..=segment_count {
        let amount = index as f64 / segment_count as f64;
        let sample = WorldPosition(previous_centre.0.lerp(centre.0, amount));
        let radius = previous_radius + (radius_metres - previous_radius) * amount;
        commands.push(TerrainEditCommand {
            centre: sample,
            radius_metres: radius,
            previous: Some(last),
            operation,
        });
        last = (sample, radius);
    }
    commands
}

pub(super) fn execute_terrain_edit_batch(
    mut terrain: TerrainOctree,
    field: &TerrainField,
    batch: Vec<TerrainEditCommand>,
) -> Result<TerrainEditTaskResult, String> {
    let started = std::time::Instant::now();
    let mut outcomes = Vec::with_capacity(batch.len());
    let mut commands = batch.into_iter().peekable();
    while let Some(command) = commands.next() {
        outcomes.push(match command.operation {
            TerrainEditOperation::Compress(first) => {
                let mut cells = vec![first];
                while let Some(TerrainEditCommand {
                    operation: TerrainEditOperation::Compress(next),
                    ..
                }) = commands.peek()
                {
                    cells.push(*next);
                    commands.next();
                }
                terrain.compress_cells(field, &cells)
            }
            TerrainEditOperation::Remove => terrain
                .excavate_sphere_delta(
                    field,
                    command.centre,
                    command.radius_metres,
                    command.previous,
                )
                .map_err(|error| error.to_string())?,
            TerrainEditOperation::Add(material) => terrain
                .add_sphere_delta(
                    field,
                    command.centre,
                    command.radius_metres,
                    material,
                    command.previous,
                )
                .map_err(|error| error.to_string())?,
        });
    }
    Ok(TerrainEditTaskResult {
        terrain,
        outcomes,
        elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
    })
}

pub(super) fn commit_terrain_edit_result(
    runtime: &mut WorldRuntime,
    result: TerrainEditTaskResult,
) -> (bool, u64) {
    let mut changed_bricks = 0_u64;
    let mut changed = false;
    let mut changed_brick_coordinates = BTreeSet::new();
    for outcome in result.outcomes {
        changed |= outcome.total_changed_cells() != 0;
        changed_brick_coordinates.extend(outcome.changed_brick_coordinates().iter().copied());
        changed_bricks = changed_bricks
            .saturating_add(u64::try_from(outcome.changed_bricks).unwrap_or(u64::MAX));
        for material in TerrainMaterial::ALL {
            let index = material.code() as usize;
            runtime.removed_cells[index] =
                runtime.removed_cells[index].saturating_add(outcome.removed_cells(material));
        }
    }
    runtime.edits = result.terrain;
    if changed {
        runtime.terrain_revision = runtime.terrain_revision.wrapping_add(1);
        runtime.terrain_acknowledgements.edit = runtime.terrain_revision;
        runtime.pending_foundation_edit.merge(TerrainEditBatch {
            generation: runtime.terrain_revision,
            changed_bricks: changed_brick_coordinates,
        });
        runtime.autosave.mutate(runtime.clock);
    }
    (changed, changed_bricks)
}

pub(super) fn coordinate_terrain_edits(
    mut runtime: ResMut<WorldRuntime>,
    mut editor: ResMut<EditorState>,
    mut diagnostics: ResMut<WorldDiagnostics>,
    list: Res<WorldListState>,
    tasks: Query<(), With<TerrainMeshTask>>,
) {
    if runtime.pending_material.is_some() {
        return;
    }
    // A transfer holds every staged mesh until its replacement cut is complete,
    // then publishes them all in one frame. Begun while loading, a saved clump
    // that is already settled would hold the loading screen indefinitely; begun
    // while terrain streams, that frame would carry the whole streamed cut.
    // Edit acknowledgements do not cover streaming, so check it here.
    if list.phase() == WorldListPhase::Playing
        && tasks.is_empty()
        && runtime.terrain_selection_task.is_none()
        && runtime.terrain_streamer.backlog() == 0
        && !runtime.terrain_streamer.has_dirty_publication()
    {
        runtime.begin_material_transfer();
    }
    if runtime.pending_material.is_some() {
        return;
    }
    let completed = runtime.terrain_edit_task.as_mut().and_then(check_ready);
    if let Some(completed) = completed {
        runtime.terrain_edit_task = None;
        match completed {
            Ok(result) => {
                diagnostics.terrain_stage_ms = result.elapsed_ms;
                let (changed, changed_bricks) = commit_terrain_edit_result(&mut runtime, result);
                if changed {
                    diagnostics.remesh_count =
                        diagnostics.remesh_count.saturating_add(changed_bricks);
                    editor.feedback = Some(format!(
                        "Removed: cover {:.3} L · soil {:.3} L · rock {:.3} L",
                        runtime.removed_cells[0] as f64 * 0.125,
                        runtime.removed_cells[1] as f64 * 0.125,
                        runtime.removed_cells[2] as f64 * 0.125,
                    ));
                }
            }
            Err(error) => {
                runtime.pending_terrain_edits.clear();
                runtime.pending_soil.clear();
                runtime.soil_ticks = 0;
                runtime.terrain_edit_error = Some(error.clone());
                editor.feedback = Some(format!("Terrain editing disabled: {error}"));
            }
        }
    }

    if runtime.terrain_edit_task.is_some()
        || runtime.pending_terrain_edits.is_empty()
        || runtime.terrain_edit_error.is_some()
    {
        return;
    }

    let batch_size = runtime
        .pending_terrain_edits
        .len()
        .min(TERRAIN_EDIT_BATCH_SIZE);
    let batch = runtime
        .pending_terrain_edits
        .drain(..batch_size)
        .collect::<Vec<_>>();
    let terrain = runtime.edits.clone();
    let field = Arc::clone(&runtime.field);
    runtime.terrain_edit_task = Some(AsyncComputeTaskPool::get().spawn(async move {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_terrain_edit_batch(terrain, &field, batch)
        }))
        .map_err(|_| "terrain edit worker panicked".to_owned())?
    }));
}

pub(super) fn finish_terrain_edits(runtime: &mut WorldRuntime) -> Result<(), String> {
    // Leaving a world cancels an unpublished ownership change, preserving the
    // last complete terrain/body pair. A future contact can retry extraction.
    if let Some(pending) = runtime.pending_material.take() {
        runtime.edits = pending.previous;
        runtime.terrain_revision = runtime.terrain_revision.wrapping_add(1);
        runtime.terrain_acknowledgements.edit = runtime.terrain_revision;
    }
    if let Some(task) = runtime.terrain_edit_task.take() {
        let result = block_on(task)?;
        commit_terrain_edit_result(runtime, result);
    }
    if runtime.pending_terrain_edits.is_empty() {
        return Ok(());
    }
    let batch = runtime.pending_terrain_edits.drain(..).collect::<Vec<_>>();
    let result = execute_terrain_edit_batch(runtime.edits.clone(), &runtime.field, batch)?;
    commit_terrain_edit_result(runtime, result);
    Ok(())
}
