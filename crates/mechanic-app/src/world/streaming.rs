//! Terrain selection, remesh scheduling, and publishing finished chunks within the frame budget.

use super::list::{WorldListPhase, WorldListState};
use super::terrain_render::{TerrainNodeRender, terrain_chunk_mesh, terrain_mesh_is_renderable};
use super::walking::{player_collision_nodes, terrain_chunk_has_collision_near};
use super::{
    Assets, Commands, Component, Entity, Mesh, Mesh3d, MeshMaterial3d, Name, Query, Res, ResMut,
    Result, String, ToOwned, Transform, Vec, Visibility, WorldDiagnostics, WorldOwned,
    WorldRuntime, format,
};
use crate::camera::PlayerState;
use crate::editor::state::EditorGraph;
use crate::simulation::state::AppSimulation;
use bevy::light::NotShadowCaster;
use bevy::tasks::futures::check_ready;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use mechanic_core::ConstructionGraph;
#[cfg(test)]
use mechanic_world::TerrainFace;
use mechanic_world::{
    ActiveTerrainNode, FloatingOrigin, TerrainBoundsCache, TerrainMeshChunk, TerrainMeshMetrics,
    TerrainMeshRequest, TerrainNodeId, TerrainOctree, TerrainSelection, WorldPosition,
    mesh_chunk_profiled, select_active_nodes_with_interests, terrain_loading_worker_count,
    terrain_worker_count,
};
use std::collections::BTreeSet;
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
    pub(super) elapsed_ms: f64,
}

pub(super) struct PendingMaterialPublication {
    pub(super) previous: TerrainOctree,
    pub(super) clumps: mechanic_world::ClumpCollection,
    pub(super) sources: Vec<mechanic_world::ExtractionCell>,
    /// Lowest and highest corner of the terrain this transfer changes, with the
    /// neighbouring bricks whose seams and gradients read it.
    pub(super) region: [bevy::math::DVec3; 2],
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
            }
            Err(error) => runtime.load_error = Some(error),
        }
    }

    let interests = runtime
        .clumps
        .bodies
        .values()
        .filter(|body| !body.sleeping)
        .map(|body| body.position)
        .collect::<Vec<_>>();
    let clumps_moved = interests.len() != runtime.selection_clump_interests.len()
        || interests
            .iter()
            .zip(&runtime.selection_clump_interests)
            .any(|(a, b)| a.0.distance_squared(b.0) >= 64.0);
    let needs_selection = clumps_moved
        || runtime.selected_terrain_revision != runtime.terrain_revision
        || runtime
            .selection_focus
            .is_none_or(|previous| previous.0.distance(focus.0) >= RESELECT_DISTANCE_METRES);
    if needs_selection && runtime.terrain_selection_task.is_none() && runtime.load_error.is_none() {
        let field = Arc::clone(&runtime.field);
        let terrain = runtime.edits.snapshot();
        let terrain_revision = runtime.terrain_revision;
        let mut bounds_cache = core::mem::take(&mut runtime.terrain_bounds_cache);
        runtime.selection_clump_interests.clone_from(&interests);
        runtime.terrain_selection_task = Some(AsyncComputeTaskPool::get().spawn(async move {
            let started = std::time::Instant::now();
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                select_active_nodes_with_interests(
                    &field,
                    &terrain,
                    focus,
                    &interests,
                    &mut bounds_cache,
                )
            }))
            .map(|selection| TerrainSelectionTaskResult {
                selection,
                bounds_cache,
                focus,
                terrain_revision,
                elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
            })
            .map_err(|_| "terrain selection worker panicked".to_owned())
        }));
    }
}

pub(super) fn schedule_terrain_remeshes(
    mut commands: Commands,
    mut runtime: ResMut<WorldRuntime>,
    focus_sources: (Res<PlayerState>, Res<EditorGraph>, Res<AppSimulation>),
    list: Res<WorldListState>,
    tasks: Query<&TerrainMeshTask>,
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

    let mut in_flight = tasks
        .iter()
        .map(|task| task.node.id)
        .collect::<BTreeSet<_>>();
    let worker_count = if list.phase() == WorldListPhase::Loading {
        terrain_loading_worker_count()
    } else {
        terrain_worker_count()
    };
    let available = worker_count.saturating_sub(in_flight.len());
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
    mut simulation: ResMut<AppSimulation>,
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
        for activated in runtime.terrain_streamer.activate(task.node.id) {
            if let Some(result) = runtime.staged_terrain.remove(&activated.id) {
                runtime.active_terrain_index.insert(activated.id);
                runtime.active_terrain.insert(activated.id, result.chunk);
            }
        }
    }

    // Material ownership waits for the complete replacement cut. Staged meshes
    // remain invisible and physics retains the prior scene until this boundary.
    let material_cutover = runtime.pending_material.is_some();
    if material_cutover {
        let pending = runtime
            .pending_material
            .as_ref()
            .expect("pending material publication");
        // Only the ground the transfer changes has to be in place. Waiting for
        // the whole cut would hold digging, and everything staged meanwhile,
        // for as long as distant terrain keeps streaming.
        if !runtime.terrain_settled_within(pending.region) {
            return;
        }
        // Without a CPU scene there is no second copy to keep consistent; the
        // route reads terrain and clumps from the world when it starts.
        if let Some(cpu) = simulation.cpu.as_mut() {
            let active = runtime
                .terrain_streamer
                .current_active()
                .map(|node| node.id)
                .collect::<BTreeSet<_>>();
            if let Err(error) = cpu.publish_material(
                runtime
                    .active_terrain
                    .iter()
                    .filter(|(id, _)| active.contains(id))
                    .map(|(_, chunk)| chunk),
                &pending.clumps,
                runtime.floating_origin.0,
            ) {
                runtime.terrain_edit_error = Some(error.clone());
                list.notice = Some(format!("Material publication failed: {error}"));
                return;
            }
        }
    }

    // In-flight worker entities are deliberately long-lived. Their mere
    // presence must not make the main thread rebuild the complete active cut,
    // allocate comparison sets, and recount every triangle on every frame.
    // Only an activation/readiness change can produce publication work.
    if !runtime.terrain_streamer.has_dirty_publication() {
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

    let active = runtime
        .terrain_streamer
        .active()
        .map(|node| node.id)
        .collect::<BTreeSet<_>>();
    let removed = runtime
        .active_terrain
        .keys()
        .copied()
        .filter(|id| !active.contains(id))
        .collect::<BTreeSet<_>>();
    let publication_delta = runtime.terrain_streamer.take_publication_delta();
    let current_active = runtime
        .terrain_streamer
        .current_active()
        .map(|node| node.id)
        .collect::<BTreeSet<_>>();
    for upsert in &publication_delta.upserts {
        if runtime.active_terrain.contains_key(&upsert.node.id) {
            runtime
                .active_terrain_ready_faces
                .insert(upsert.node.id, upsert.ready_faces);
        }
    }
    for &id in &current_active {
        if removed
            .iter()
            .any(|&obsolete| terrain_node_regions_overlap(id, obsolete))
        {
            runtime.active_terrain_index.remove(id);
        }
    }

    let material = runtime
        .terrain_material
        .as_ref()
        .expect("world terrain material exists")
        .clone();
    let dirty = publication_delta
        .upserts
        .into_iter()
        .map(|upsert| upsert.node.id)
        .collect::<Vec<_>>();
    let mut deferred = Vec::new();
    // Publishing gets its own slice of the budget. Measured from the start of
    // the frame, the bookkeeping above exhausts it on a large cut, and every
    // frame then defers the same nodes without ever publishing one.
    let publishing = std::time::Instant::now();
    // A cutover shows its own ground in the frame its clumps appear; the rest of
    // the cut keeps to the budget.
    let cutover_region = runtime
        .pending_material
        .as_ref()
        .map(|pending| pending.region);
    for id in dirty.iter().copied() {
        let forced = cutover_region.is_some_and(|[low, high]| node_overlaps(id, low, high));
        if !forced && publishing.elapsed().as_secs_f64() * 1_000.0 >= INTEGRATION_BUDGET_MS {
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
            continue;
        }
        let transform =
            Transform::from_translation((chunk.origin.0 - runtime.floating_origin.0).as_vec3());
        let mesh = terrain_chunk_mesh(chunk, indices);
        let waits_for_cutover = removed
            .iter()
            .any(|&obsolete| terrain_node_regions_overlap(id, obsolete));
        let visibility = if waits_for_cutover {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        if let Some(handle) = runtime.terrain_mesh_handles.get(&id).cloned() {
            if let Some(mut existing) = meshes.get_mut(&handle) {
                *existing = mesh;
            }
            if let Some(&entity) = runtime.terrain_entities.get(&id) {
                commands.entity(entity).insert((transform, visibility));
            }
        } else {
            let handle = meshes.add(mesh);
            let entity = commands
                .spawn((
                    Name::new(format!(
                        "Terrain node L{} {},{},{}",
                        id.level, id.coordinates.x, id.coordinates.y, id.coordinates.z
                    )),
                    Mesh3d(handle.clone()),
                    MeshMaterial3d(material.clone()),
                    transform,
                    visibility,
                    TerrainNodeRender,
                    NotShadowCaster,
                    WorldOwned,
                ))
                .id();
            runtime.terrain_mesh_handles.insert(id, handle);
            runtime.terrain_entities.insert(id, entity);
        }
    }
    runtime.terrain_streamer.defer_publication(deferred);

    let published = current_active
        .iter()
        .copied()
        .filter(|id| {
            let Some(chunk) = runtime.active_terrain.get(id) else {
                return false;
            };
            let ready = runtime
                .active_terrain_ready_faces
                .get(id)
                .copied()
                .unwrap_or_default();
            let index_count = chunk
                .index_groups
                .sealed_index_count(chunk.transition_mask, ready);
            !terrain_mesh_is_renderable(chunk, index_count)
                || runtime.terrain_entities.contains_key(id)
        })
        .collect::<BTreeSet<_>>();
    let retired = ready_obsolete_nodes(&removed, &current_active, &published);
    for &id in &current_active {
        // Only a node held back by an obsolete owner is missing from the index.
        // Re-inserting refits its whole ancestor path and re-issues a visibility
        // command, which for every active node on every frame cost more than
        // the rest of publication combined.
        if runtime.active_terrain_index.contains(id) {
            continue;
        }
        let owners = removed
            .iter()
            .copied()
            .filter(|&obsolete| terrain_node_regions_overlap(id, obsolete))
            .collect::<BTreeSet<_>>();
        if owners.iter().all(|owner| retired.contains(owner)) {
            runtime.active_terrain_index.insert(id);
            if let Some(&entity) = runtime.terrain_entities.get(&id) {
                commands.entity(entity).insert(Visibility::Inherited);
            }
        }
    }
    for id in retired {
        runtime.active_terrain.remove(&id);
        runtime.active_terrain_ready_faces.remove(&id);
        runtime.active_terrain_index.remove(id);
        if let Some(entity) = runtime.terrain_entities.remove(&id) {
            commands.entity(entity).despawn();
        }
        if let Some(handle) = runtime.terrain_mesh_handles.remove(&id) {
            meshes.remove(handle.id());
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
    diagnostics.triangle_count = runtime
        .active_terrain
        .iter()
        .map(|(&id, chunk)| {
            // A chunk activated while a material cutover held publication has no
            // published faces yet.
            let ready = runtime
                .active_terrain_ready_faces
                .get(&id)
                .copied()
                .unwrap_or_default();
            u64::try_from(
                chunk
                    .index_groups
                    .sealed_index_count(chunk.transition_mask, ready)
                    / 3,
            )
            .unwrap_or(u64::MAX)
        })
        .sum();
    diagnostics.streaming_backlog = u32::try_from(
        runtime
            .terrain_streamer
            .backlog()
            .saturating_add(tasks.iter().count()),
    )
    .unwrap_or(u32::MAX);
    acknowledge_complete_terrain_pipeline(&mut runtime, tasks.is_empty());
    if material_cutover && let Some(pending) = runtime.pending_material.take() {
        runtime.clumps = pending.clumps;
        runtime.pending_breakage.committed(&pending.sources);
        let now = runtime.clock;
        runtime.autosave.mutate(now);
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

pub(super) fn terrain_node_regions_overlap(first: TerrainNodeId, second: TerrainNodeId) -> bool {
    let first_minimum = first.minimum_cell_i64();
    let first_maximum = first.maximum_cell_exclusive_i64();
    let second_minimum = second.minimum_cell_i64();
    let second_maximum = second.maximum_cell_exclusive_i64();
    (0..3).all(|axis| {
        first_minimum[axis] < second_maximum[axis] && second_minimum[axis] < first_maximum[axis]
    })
}

pub(super) fn ready_obsolete_nodes(
    obsolete: &BTreeSet<TerrainNodeId>,
    current: &BTreeSet<TerrainNodeId>,
    published: &BTreeSet<TerrainNodeId>,
) -> BTreeSet<TerrainNodeId> {
    obsolete
        .iter()
        .copied()
        .filter(|&old| {
            current
                .iter()
                .copied()
                .filter(|&replacement| terrain_node_regions_overlap(old, replacement))
                .all(|replacement| published.contains(&replacement))
        })
        .collect()
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
