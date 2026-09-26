//! Terrain selection, remesh scheduling, and publishing finished chunks within the frame budget.

use super::list::{WorldListPhase, WorldListState};
use super::terrain_render::{TerrainNodeRender, terrain_chunk_mesh, terrain_mesh_is_renderable};
use super::walking::{player_collision_nodes, terrain_chunk_has_collision_near};
use super::{
    Assets, Commands, Component, Entity, Mesh, Mesh3d, MeshMaterial3d, Name, Query, Res, ResMut,
    Result, String, ToOwned, Transform, Vec, Visibility, WorldDiagnostics, WorldOwned,
    WorldRuntime, format,
};
use crate::camera::{MainCamera, PlayerState};
use crate::editor::state::EditorGraph;
use crate::simulation::state::AppSimulation;
use bevy::camera::Projection;
use bevy::camera::primitives::Aabb;
use bevy::camera::visibility::NoAutoAabb;
use bevy::light::NotShadowCaster;
use bevy::prelude::{GlobalTransform, Time, With};
use bevy::tasks::futures::check_ready;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use mechanic_core::ConstructionGraph;
#[cfg(test)]
use mechanic_world::TerrainFace;
use mechanic_world::{
    ActiveTerrainNode, FloatingOrigin, MIN_TERRAIN_DETAIL_SCALE, TerrainBoundsCache,
    TerrainMeshChunk, TerrainMeshMetrics, TerrainMeshRequest, TerrainNodeId, TerrainSelection,
    TerrainView, WorldPosition, mesh_chunk_profiled, select_active_nodes_with_interests,
    terrain_loading_worker_count, terrain_worker_count,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[derive(Component)]
pub(super) struct TerrainMeshTask {
    pub(super) node: ActiveTerrainNode,
    pub(super) task: Task<Result<TerrainMeshResult, String>>,
}

pub(super) struct TerrainMeshResult {
    pub(super) chunk: TerrainMeshChunk,
    pub(super) elapsed_ms: f64,
    pub(super) metrics: TerrainMeshMetrics,
    pub(super) queue_wait_ms: f64,
}

pub(super) struct TerrainSelectionTaskResult {
    pub(super) selection: TerrainSelection,
    pub(super) bounds_cache: TerrainBoundsCache,
    pub(super) focus: WorldPosition,
    pub(super) terrain_revision: u64,
    pub(super) detail_steps: u8,
    pub(super) elapsed_ms: f64,
}

/// Resident terrain triangles the renderer aims to stay under; about what
/// the opaque pass drew at its measured baseline.
const TERRAIN_TRIANGLE_BUDGET: u64 = 6_000_000;

/// Shortest time between two detail changes, so a change can take effect
/// before it is judged.
const DETAIL_SETTLE_SECONDS: f32 = 3.0;

/// How finely terrain is selected, steered by the triangle budget: each
/// coarsening step shrinks every detail band by a fifth.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TerrainDetail {
    /// Coarsening steps the next selection uses.
    pub(super) steps: u8,
    /// Coarsening steps of the cut currently desired.
    pub(super) selected: u8,
    since_change: f32,
}

/// Coarsening steps the budget may take; the scale then sits at its floor.
const MAX_DETAIL_STEPS: u8 = 5;

impl Default for TerrainDetail {
    fn default() -> Self {
        Self {
            steps: 0,
            selected: 0,
            since_change: DETAIL_SETTLE_SECONDS,
        }
    }
}

impl TerrainDetail {
    /// Detail scale for a number of coarsening steps.
    pub(super) fn scale_of(steps: u8) -> f64 {
        0.8_f64.powi(i32::from(steps)).max(MIN_TERRAIN_DETAIL_SCALE)
    }

    /// Detail scale the next selection uses.
    pub(super) fn scale(self) -> f64 {
        Self::scale_of(self.steps)
    }

    /// Coarsens the cut when resident triangles pass the budget, and refines
    /// it again once they fall well below. `settled` means the desired cut is
    /// fully published, so the count is the cut's own.
    pub(super) fn steer(&mut self, triangles: u64, settled: bool, elapsed: f32) {
        self.since_change += elapsed;
        if self.since_change < DETAIL_SETTLE_SECONDS || self.selected != self.steps {
            return;
        }
        let over = triangles > TERRAIN_TRIANGLE_BUDGET * 3 / 2
            || (settled && triangles > TERRAIN_TRIANGLE_BUDGET);
        let under = settled && triangles < TERRAIN_TRIANGLE_BUDGET * 3 / 5;
        let steps = if over {
            (self.steps + 1).min(MAX_DETAIL_STEPS)
        } else if under {
            self.steps.saturating_sub(1)
        } else {
            self.steps
        };
        if steps != self.steps {
            self.steps = steps;
            self.since_change = 0.0;
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct TerrainAcknowledgements {
    pub(super) edit: u64,
    pub(super) mesh: u64,
    pub(super) upload: u64,
    pub(super) collision: u64,
}

impl TerrainAcknowledgements {
    pub(super) const fn completed(self, generation: u64) -> bool {
        self.edit == generation
            && self.mesh == generation
            && self.upload == generation
            && self.collision == generation
    }
}

pub(super) fn update_terrain_selection(
    runtime: &mut WorldRuntime,
    diagnostics: &mut WorldDiagnostics,
    focus: WorldPosition,
) {
    const RESELECT_DISTANCE_METRES: f64 = 8.0;

    let completed_selection = runtime
        .terrain_selection_task
        .as_mut()
        .and_then(check_ready);
    if let Some(completed) = completed_selection {
        runtime.terrain_selection_task = None;
        match completed {
            Ok(result) => {
                diagnostics.selection_ms = result.elapsed_ms;
                diagnostics.selection_count = diagnostics.selection_count.saturating_add(1);
                diagnostics.bounds_cache_bytes =
                    u64::try_from(result.selection.stats.cache_memory_bytes).unwrap_or(u64::MAX);
                runtime.terrain_bounds_cache = result.bounds_cache;
                // A continuous stroke can advance the edit revision while a
                // selection is still running. Publishing that stale cut would
                // immediately launch an obsolete horizon's mesh jobs and then
                // replace them again. Keep the procedural cache, but only let
                // the newest snapshot change desired terrain.
                if result.terrain_revision != runtime.terrain_revision {
                    return;
                }
                let cut = result.selection.nodes;
                let mut focus_capsule = runtime.capsule;
                focus_capsule.position = result.focus;
                let critical = startup_region_nodes(&cut, result.focus).collect::<Vec<_>>();
                runtime.terrain_streamer.set_pinned(
                    player_collision_nodes(&cut, &focus_capsule).chain(critical.iter().copied()),
                );
                runtime
                    .terrain_streamer
                    .set_critical_nodes(critical.iter().copied());
                runtime.terrain_streamer.set_desired(cut);
                runtime.selection_focus = Some(result.focus);
                runtime.selected_terrain_revision = result.terrain_revision;
                runtime.terrain_detail.selected = result.detail_steps;
            }
            Err(error) => runtime.load_error = Some(error),
        }
    }

    let needs_selection = runtime.selected_terrain_revision != runtime.terrain_revision
        || runtime.terrain_detail.selected != runtime.terrain_detail.steps
        || runtime
            .selection_focus
            .is_none_or(|previous| previous.0.distance(focus.0) >= RESELECT_DISTANCE_METRES);
    if needs_selection && runtime.terrain_selection_task.is_none() && runtime.load_error.is_none() {
        let field = Arc::clone(&runtime.field);
        let terrain = runtime.edits.snapshot();
        let terrain_revision = runtime.terrain_revision;
        let mut bounds_cache = core::mem::take(&mut runtime.terrain_bounds_cache);
        let detail_steps = runtime.terrain_detail.steps;
        let detail_scale = TerrainDetail::scale_of(detail_steps);
        runtime.terrain_selection_task = Some(AsyncComputeTaskPool::get().spawn(async move {
            let started = std::time::Instant::now();
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                select_active_nodes_with_interests(
                    &field,
                    &terrain,
                    focus,
                    &[],
                    detail_scale,
                    &mut bounds_cache,
                )
            }))
            .map(|selection| TerrainSelectionTaskResult {
                selection,
                bounds_cache,
                focus,
                terrain_revision,
                detail_steps,
                elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
            })
            .map_err(|_| "terrain selection worker panicked".to_owned())
        }));
    }
}

/// Mesh jobs kept in flight per terrain worker. Results are collected once a
/// frame, so a slow frame must not leave workers idle; jobs the cut stops
/// wanting are cancelled before they start.
const TERRAIN_JOBS_PER_WORKER: usize = 6;

pub(super) fn schedule_terrain_remeshes(
    mut commands: Commands,
    mut runtime: ResMut<WorldRuntime>,
    focus_sources: (Res<PlayerState>, Res<EditorGraph>, Res<AppSimulation>),
    list: Res<WorldListState>,
    tasks: Query<(Entity, &TerrainMeshTask)>,
    cameras: Query<(&GlobalTransform, &Projection), With<MainCamera>>,
    mut diagnostics: ResMut<WorldDiagnostics>,
) {
    let (player, graph, simulation) = focus_sources;
    let focus = terrain_streaming_focus(&player, &graph.0, &simulation, runtime.floating_origin);
    // A selection taken between two queued stroke batches is guaranteed to be
    // obsolete. Let the existing cut keep rendering/colliding and reconcile
    // once the ordered edit queue reaches a stable revision.
    if runtime.terrain_edit_task.is_none() && runtime.pending_terrain_edits.is_empty() {
        update_terrain_selection(&mut runtime, &mut diagnostics, focus);
    }
    runtime.terrain_streamer.set_view(
        cameras
            .single()
            .ok()
            .map(|(transform, projection)| terrain_view(transform, projection)),
    );

    // Dropping a task that has not started cancels it, so work the cut no
    // longer wants gives its slot to work it does.
    let mut in_flight = BTreeSet::new();
    for (entity, task) in &tasks {
        if runtime.terrain_streamer.wants(&task.node) {
            in_flight.insert(task.node.id);
        } else {
            commands.entity(entity).despawn();
        }
    }
    let worker_count = if list.phase() == WorldListPhase::Loading {
        terrain_loading_worker_count()
    } else {
        terrain_worker_count()
    };
    // Completions are collected once a frame, so keep a few jobs queued per
    // worker; otherwise throughput falls with the frame rate.
    let available = (worker_count * TERRAIN_JOBS_PER_WORKER).saturating_sub(in_flight.len());
    for _ in 0..available {
        let Some(node) = runtime.terrain_streamer.next_request(&in_flight, focus) else {
            break;
        };
        runtime.terrain_streamer.mark_started(node);
        in_flight.insert(node.id);
        let field = Arc::clone(&runtime.field);
        let terrain = runtime.edits.snapshot();
        let queued_at = std::time::Instant::now();
        let task = AsyncComputeTaskPool::get().spawn(async move {
            let started = std::time::Instant::now();
            let queue_wait_ms = queued_at.elapsed().as_secs_f64() * 1_000.0;
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                mesh_chunk_profiled(
                    &field,
                    &terrain,
                    TerrainMeshRequest {
                        node: node.id,
                        generation: node.generation,
                        transition_mask: node.transition_mask,
                    },
                )
            }))
            .map(|(chunk, metrics)| TerrainMeshResult {
                chunk,
                elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
                metrics,
                queue_wait_ms,
            })
            .map_err(|_| format!("terrain extraction panicked for node {:?}", node.id))
        });
        commands.spawn((
            Name::new(format!(
                "Terrain node task L{} {},{},{}",
                node.id.level, node.id.coordinates.x, node.id.coordinates.y, node.id.coordinates.z
            )),
            TerrainMeshTask { node, task },
            WorldOwned,
        ));
    }
    diagnostics.streaming_backlog = u32::try_from(
        runtime
            .terrain_streamer
            .backlog()
            .saturating_add(in_flight.len()),
    )
    .unwrap_or(u32::MAX);
    diagnostics.oldest_queue_age_ms =
        runtime.terrain_streamer.oldest_queue_age().as_secs_f64() * 1_000.0;
}

/// The camera's horizontal look direction and half field of view.
fn terrain_view(transform: &GlobalTransform, projection: &Projection) -> TerrainView {
    let forward = transform.forward().as_vec3().as_dvec3();
    let half_angle = match projection {
        Projection::Perspective(perspective) => {
            f64::from((perspective.fov * 0.5).tan() * perspective.aspect_ratio).atan()
        }
        _ => core::f64::consts::FRAC_PI_2,
    };
    TerrainView {
        forward,
        half_angle,
    }
}

/// Terrain selection follows the occupied seat while driving.
///
/// `PlayerState::position` remains at the point where the player entered a
/// seat, while the seat and camera move with physics. Selecting around that
/// stale point eventually leaves a vehicle outside the resident terrain cut.
pub(crate) fn terrain_streaming_focus(
    player: &PlayerState,
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    origin: FloatingOrigin,
) -> WorldPosition {
    let local = player
        .seat
        .and_then(|seat| crate::seat::seat_world_pose(graph, simulation, seat))
        .map_or(player.position, |(position, _)| position);
    WorldPosition(origin.0 + local.as_dvec3())
}

pub(super) fn startup_region_nodes(
    cut: &[ActiveTerrainNode],
    centre: WorldPosition,
) -> impl Iterator<Item = TerrainNodeId> + '_ {
    let minimum = centre.0 - bevy::math::DVec3::splat(16.0);
    let maximum = centre.0 + bevy::math::DVec3::splat(16.0);
    cut.iter()
        .map(|node| node.id)
        .filter(move |&id| node_overlaps(id, minimum, maximum))
}

pub(super) fn acknowledge_complete_terrain_pipeline(
    runtime: &mut WorldRuntime,
    workers_idle: bool,
) {
    if workers_idle
        && runtime.selected_terrain_revision == runtime.terrain_revision
        && runtime.terrain_streamer.backlog() == 0
        && !runtime.terrain_streamer.has_dirty_publication()
    {
        let generation = runtime.terrain_revision;
        runtime.terrain_acknowledgements.mesh = generation;
        runtime.terrain_acknowledgements.upload = generation;
        runtime.terrain_acknowledgements.collision = generation;
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "publication is one atomic frame-budgeted state transition"
)]
pub(super) fn integrate_terrain_remeshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut runtime: ResMut<WorldRuntime>,
    mut tasks: Query<(Entity, &mut TerrainMeshTask)>,
    mut diagnostics: ResMut<WorldDiagnostics>,
    mut list: ResMut<WorldListState>,
    time: Option<Res<Time>>,
) {
    const INTEGRATION_BUDGET_MS: f64 = 2.0;
    let started = std::time::Instant::now();
    diagnostics.column_sampling_ms = 0.0;
    diagnostics.polygonization_ms = 0.0;
    diagnostics.transitions_caps_ms = 0.0;
    diagnostics.bvh_construction_ms = 0.0;
    if tasks.is_empty() && !runtime.terrain_streamer.has_dirty_publication() {
        acknowledge_complete_terrain_pipeline(&mut runtime, true);
        diagnostics.terrain_stage_ms = 0.0;
        diagnostics.publication_ms = 0.0;
        return;
    }
    let mut maximum_stage_ms = 0.0_f64;
    for (task_entity, mut task) in &mut tasks {
        if started.elapsed().as_secs_f64() * 1_000.0 >= INTEGRATION_BUDGET_MS {
            break;
        }
        let Some(result) = check_ready(&mut task.task) else {
            continue;
        };
        commands.entity(task_entity).despawn();
        diagnostics.completed_mesh_jobs = diagnostics.completed_mesh_jobs.saturating_add(1);
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                runtime.load_error = Some(error.clone());
                list.notice = Some(format!("World loading failed: {error}"));
                continue;
            }
        };
        if !runtime.terrain_streamer.stage(task.node) {
            continue;
        }
        maximum_stage_ms = maximum_stage_ms.max(result.elapsed_ms);
        diagnostics.column_sampling_ms = diagnostics
            .column_sampling_ms
            .max(result.metrics.column_sampling_ms);
        diagnostics.polygonization_ms = diagnostics
            .polygonization_ms
            .max(result.metrics.polygonization_ms);
        diagnostics.transitions_caps_ms = diagnostics
            .transitions_caps_ms
            .max(result.metrics.transitions_caps_ms);
        diagnostics.bvh_construction_ms = diagnostics
            .bvh_construction_ms
            .max(result.metrics.bvh_construction_ms);
        diagnostics.oldest_queue_age_ms = diagnostics.oldest_queue_age_ms.max(result.queue_wait_ms);
        runtime.staged_terrain.insert(task.node.id, result);
        let activation = runtime.terrain_streamer.activate_replacing(task.node.id);
        let mut activated = Vec::with_capacity(activation.activated.len());
        for node in activation.activated {
            if let Some(result) = runtime.staged_terrain.remove(&node.id) {
                runtime.active_terrain.insert(node.id, result.chunk);
                runtime.terrain_cutovers.reactivate(node.id);
                activated.push(node.id);
            }
        }
        let retired = activation
            .retired
            .into_iter()
            .filter(|id| !activated.contains(id))
            .collect::<Vec<_>>();
        if retired.is_empty() {
            // Nothing is replaced, so the node joins the published cut as soon
            // as it is published.
            for &id in &activated {
                runtime.active_terrain_index.insert(id);
            }
        } else {
            for &id in &activated {
                runtime.active_terrain_index.remove(id);
            }
            runtime.terrain_cutovers.begin(activated, retired);
        }
    }

    // In-flight worker entities are deliberately long-lived. Their mere
    // presence must not make the main thread touch the complete active cut;
    // only an activation or readiness change produces publication work.
    if !runtime.terrain_streamer.has_dirty_publication() && !runtime.terrain_cutovers.is_waiting() {
        diagnostics.terrain_stage_ms = maximum_stage_ms;
        diagnostics.publication_ms = started.elapsed().as_secs_f64() * 1_000.0;
        diagnostics.streaming_backlog = u32::try_from(
            runtime
                .terrain_streamer
                .backlog()
                .saturating_add(tasks.iter().count()),
        )
        .unwrap_or(u32::MAX);
        return;
    }

    let publication_delta = runtime.terrain_streamer.take_publication_delta();
    for upsert in &publication_delta.upserts {
        if runtime.active_terrain.contains_key(&upsert.node.id) {
            runtime
                .active_terrain_ready_faces
                .insert(upsert.node.id, upsert.ready_faces);
        }
    }
    // Nodes the cut dropped without a replacement leave at once; replaced
    // nodes wait for their cutover.
    for &id in &publication_delta.removals {
        if runtime.active_terrain.contains_key(&id)
            && !runtime.terrain_streamer.is_active(id)
            && !runtime.terrain_cutovers.is_retiring(id)
        {
            retire_terrain_node(&mut runtime, &mut commands, &mut meshes, id);
        }
    }

    let material = runtime
        .terrain_material
        .as_ref()
        .expect("world terrain material exists")
        .clone();
    let mut deferred = Vec::new();
    // Publishing gets its own slice of the budget, checked before every
    // node, so a burst of activations spreads over several frames.
    let publishing = std::time::Instant::now();
    for upsert in publication_delta.upserts {
        let id = upsert.node.id;
        if publishing.elapsed().as_secs_f64() * 1_000.0 >= INTEGRATION_BUDGET_MS {
            deferred.push(id);
            continue;
        }
        let Some(chunk) = runtime.active_terrain.get(&id) else {
            continue;
        };
        let ready = runtime
            .active_terrain_ready_faces
            .get(&id)
            .copied()
            .unwrap_or_default();
        let indices = chunk
            .index_groups
            .sealed_indices(chunk.transition_mask, ready);
        if !terrain_mesh_is_renderable(chunk, indices.len()) {
            if let Some(entity) = runtime.terrain_entities.remove(&id) {
                commands.entity(entity).despawn();
            }
            if let Some(handle) = runtime.terrain_mesh_handles.remove(&id) {
                meshes.remove(handle.id());
            }
            runtime.terrain_cutovers.published(id, 0);
            continue;
        }
        let triangles = u64::try_from(indices.len() / 3).unwrap_or(u64::MAX);
        let transform =
            Transform::from_translation((chunk.origin.0 - runtime.floating_origin.0).as_vec3());
        let bounds = terrain_chunk_aabb(chunk);
        // The main world keeps no copy of an uploaded terrain mesh, so a
        // changed chunk always gets a new mesh asset.
        let handle = meshes.add(terrain_chunk_mesh(chunk, indices, runtime.field.palette()));
        let visibility = if runtime.terrain_cutovers.is_hidden(id) {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        if let Some(previous) = runtime.terrain_mesh_handles.insert(id, handle.clone()) {
            meshes.remove(previous.id());
        }
        if let Some(&entity) = runtime.terrain_entities.get(&id) {
            commands
                .entity(entity)
                .insert((Mesh3d(handle), bounds, transform, visibility));
        } else {
            let entity = commands
                .spawn((
                    Name::new(format!(
                        "Terrain node L{} {},{},{}",
                        id.level, id.coordinates.x, id.coordinates.y, id.coordinates.z
                    )),
                    Mesh3d(handle),
                    MeshMaterial3d(material.clone()),
                    bounds,
                    NoAutoAabb,
                    transform,
                    visibility,
                    TerrainNodeRender,
                    NotShadowCaster,
                    WorldOwned,
                ))
                .id();
            runtime.terrain_entities.insert(id, entity);
        }
        runtime.terrain_cutovers.published(id, triangles);
    }
    runtime.terrain_streamer.defer_publication(deferred);

    // A replacement shows, and what it replaced leaves, once every node of
    // the replacement is published.
    let runtime_state = &mut *runtime;
    let active_terrain = &runtime_state.active_terrain;
    let completed = runtime_state
        .terrain_cutovers
        .take_completed(|id| active_terrain.contains_key(&id));
    for cutover in completed {
        for id in cutover.new {
            if !runtime.active_terrain.contains_key(&id) {
                continue;
            }
            runtime.active_terrain_index.insert(id);
            if let Some(&entity) = runtime.terrain_entities.get(&id) {
                commands.entity(entity).insert(Visibility::Inherited);
            }
        }
        for id in cutover.old {
            if !runtime.terrain_streamer.is_active(id) {
                retire_terrain_node(&mut runtime, &mut commands, &mut meshes, id);
            }
        }
    }

    let readiness = runtime.terrain_streamer.local_readiness();
    diagnostics.local_resolved_nodes = u32::try_from(readiness.resolved).unwrap_or(u32::MAX);
    diagnostics.local_total_nodes = u32::try_from(readiness.total).unwrap_or(u32::MAX);
    if list.phase() == WorldListPhase::Loading {
        list.loading_progress = readiness;
        if let Some(error) = &runtime.load_error {
            list.notice = Some(format!("World loading failed: {error}"));
        } else if readiness.is_complete()
            && runtime.active_terrain.iter().any(|(&id, chunk)| {
                terrain_chunk_has_collision_near(
                    chunk,
                    runtime
                        .active_terrain_ready_faces
                        .get(&id)
                        .copied()
                        .unwrap_or_default(),
                    &runtime.capsule,
                )
            })
        {
            runtime.player_terrain_ready = true;
            list.phase = WorldListPhase::Playing;
            list.notice = None;
        }
    }

    diagnostics.terrain_stage_ms = maximum_stage_ms;
    diagnostics.publication_ms = started.elapsed().as_secs_f64() * 1_000.0;
    diagnostics.triangle_count = runtime.terrain_cutovers.triangles();
    let settled = runtime.terrain_streamer.backlog() == 0
        && !runtime.terrain_cutovers.is_waiting()
        && runtime.terrain_selection_task.is_none();
    let triangles = diagnostics.triangle_count;
    runtime.terrain_detail.steer(
        triangles,
        settled,
        time.map_or(0.0, |time| time.delta_secs()),
    );
    diagnostics.terrain_detail_scale = runtime.terrain_detail.scale();
    diagnostics.streaming_backlog = u32::try_from(
        runtime
            .terrain_streamer
            .backlog()
            .saturating_add(tasks.iter().count()),
    )
    .unwrap_or(u32::MAX);
    acknowledge_complete_terrain_pipeline(&mut runtime, tasks.is_empty());
}

/// Removes a node from every render and collision owner.
fn retire_terrain_node(
    runtime: &mut WorldRuntime,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    id: TerrainNodeId,
) {
    runtime.active_terrain.remove(&id);
    runtime.active_terrain_ready_faces.remove(&id);
    runtime.active_terrain_index.remove(id);
    runtime.terrain_cutovers.forget(id);
    if let Some(entity) = runtime.terrain_entities.remove(&id) {
        commands.entity(entity).despawn();
    }
    if let Some(handle) = runtime.terrain_mesh_handles.remove(&id) {
        meshes.remove(handle.id());
    }
}

/// Frustum-culling bounds of a chunk in its entity's local space.
fn terrain_chunk_aabb(chunk: &TerrainMeshChunk) -> Aabb {
    let minimum = (chunk.bounds.minimum.0 - chunk.origin.0).as_vec3();
    let maximum = (chunk.bounds.maximum.0 - chunk.origin.0).as_vec3();
    Aabb::from_min_max(minimum, maximum)
}

/// Replacements waiting to be published before they show, and the triangles
/// each published node holds.
#[derive(Debug, Default)]
pub(crate) struct TerrainCutovers {
    waiting: Vec<TerrainCutover>,
    published: BTreeMap<TerrainNodeId, u64>,
    triangles: u64,
}

/// New nodes hidden until all are published, and the old nodes they replace.
#[derive(Debug)]
pub(super) struct TerrainCutover {
    pub(super) new: Vec<TerrainNodeId>,
    pub(super) old: Vec<TerrainNodeId>,
}

impl TerrainCutovers {
    pub(super) fn begin(&mut self, new: Vec<TerrainNodeId>, old: Vec<TerrainNodeId>) {
        for id in &new {
            self.unpublish(*id);
        }
        self.waiting.push(TerrainCutover { new, old });
    }

    fn is_waiting(&self) -> bool {
        !self.waiting.is_empty()
    }

    /// A node activated again stops waiting to retire.
    fn reactivate(&mut self, id: TerrainNodeId) {
        for cutover in &mut self.waiting {
            cutover.old.retain(|old| *old != id);
        }
    }

    pub(super) fn is_retiring(&self, id: TerrainNodeId) -> bool {
        self.waiting.iter().any(|cutover| cutover.old.contains(&id))
    }

    pub(super) fn is_hidden(&self, id: TerrainNodeId) -> bool {
        self.waiting.iter().any(|cutover| cutover.new.contains(&id))
    }

    pub(super) fn published(&mut self, id: TerrainNodeId, triangles: u64) {
        let previous = self.published.insert(id, triangles).unwrap_or(0);
        self.triangles = self.triangles - previous.min(self.triangles) + triangles;
    }

    fn unpublish(&mut self, id: TerrainNodeId) {
        if let Some(previous) = self.published.remove(&id) {
            self.triangles -= previous.min(self.triangles);
        }
    }

    fn forget(&mut self, id: TerrainNodeId) {
        self.unpublish(id);
        for cutover in &mut self.waiting {
            cutover.old.retain(|old| *old != id);
        }
    }

    /// Triangles of every resident published node, hidden ones included.
    pub(super) fn triangles(&self) -> u64 {
        self.triangles
    }

    /// Cutovers whose new nodes are all published or gone.
    pub(super) fn take_completed(
        &mut self,
        present: impl Fn(TerrainNodeId) -> bool,
    ) -> Vec<TerrainCutover> {
        let (done, waiting) =
            core::mem::take(&mut self.waiting)
                .into_iter()
                .partition(|cutover: &TerrainCutover| {
                    cutover
                        .new
                        .iter()
                        .all(|&id| self.published.contains_key(&id) || !present(id))
                });
        self.waiting = waiting;
        done
    }
}

pub(super) fn node_overlaps(
    node: TerrainNodeId,
    minimum: bevy::math::DVec3,
    maximum: bevy::math::DVec3,
) -> bool {
    let node_minimum = bevy::math::DVec3::from_array(
        node.minimum_cell_i64()
            .map(|cell| cell as f64 * mechanic_world::TERRAIN_CELL_METERS),
    );
    let node_maximum = bevy::math::DVec3::from_array(
        node.maximum_cell_exclusive_i64()
            .map(|cell| cell as f64 * mechanic_world::TERRAIN_CELL_METERS),
    );
    bounds_overlap(node_minimum, node_maximum, minimum, maximum)
}

pub(super) fn bounds_overlap(
    first_minimum: bevy::math::DVec3,
    first_maximum: bevy::math::DVec3,
    second_minimum: bevy::math::DVec3,
    second_maximum: bevy::math::DVec3,
) -> bool {
    first_minimum.cmple(second_maximum).all() && second_minimum.cmple(first_maximum).all()
}

#[cfg(test)]
pub(super) fn nodes_touch_on_face(
    first: TerrainNodeId,
    second: TerrainNodeId,
    face: TerrainFace,
) -> bool {
    let first_min = first.minimum_cell_i64();
    let first_max = first.maximum_cell_exclusive_i64();
    let second_min = second.minimum_cell_i64();
    let second_max = second.maximum_cell_exclusive_i64();
    let overlaps =
        |axis: usize| first_min[axis] < second_max[axis] && second_min[axis] < first_max[axis];
    match face {
        TerrainFace::NegativeX => first_min[0] == second_max[0] && overlaps(1) && overlaps(2),
        TerrainFace::PositiveX => first_max[0] == second_min[0] && overlaps(1) && overlaps(2),
        TerrainFace::NegativeY => first_min[1] == second_max[1] && overlaps(0) && overlaps(2),
        TerrainFace::PositiveY => first_max[1] == second_min[1] && overlaps(0) && overlaps(2),
        TerrainFace::NegativeZ => first_min[2] == second_max[2] && overlaps(0) && overlaps(1),
        TerrainFace::PositiveZ => first_max[2] == second_min[2] && overlaps(0) && overlaps(1),
    }
}
