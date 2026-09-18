//! Bounded procedural-world prototype state and playable terrain tools.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::Arc,
    time::Duration,
};

use bevy::{
    app::AppExit,
    asset::RenderAssetUsages,
    camera::Exposure,
    light::NotShadowCaster,
    math::{DVec2, DVec3},
    mesh::Indices,
    prelude::*,
    render::render_resource::{AsBindGroup, Face, PrimitiveTopology, TextureFormat},
    shader::ShaderRef,
    tasks::{AsyncComputeTaskPool, Task, block_on, futures::check_ready},
};
use mechanic_core::{
    BearingSocket, ConstructionEditDelta, ConstructionGraph, CreationDocument, DimensionLinkId,
    FaceOwnerDoc, PartId, PartSpec,
};
use mechanic_gpu::GpuExternalImpulse;
#[cfg(test)]
use mechanic_world::TerrainFace;
use mechanic_world::{
    ActiveTerrainNode, ActiveTerrainScene, AutosaveState, ConstructionBodyPose,
    ConstructionCollisionIndex, FloatingOrigin, FoundationSpatialIndex, FoundationSupport,
    KinematicCapsule, KinematicCollisionScene, KinematicInput, OpenWorldResult, SavedWorld,
    SavedWorldStatus, TerrainBoundsCache, TerrainDensity, TerrainEditBatch, TerrainEditOutcome,
    TerrainField, TerrainMaterial, TerrainMeshChunk, TerrainMeshMetrics, TerrainMeshRequest,
    TerrainNodeId, TerrainOctree, TerrainRayHit, TerrainReadiness, TerrainScene, TerrainSelection,
    TerrainSpatialIndex, TerrainStreamer, TerrainTransitionMask, WorldBounds,
    WorldCreationInstanceDoc, WorldDocument, WorldInstanceIndexDoc, WorldPoseDoc, WorldPosition,
    WorldSeed, WorldStore, mesh_chunk_profiled, raycast_density,
    select_active_nodes_with_interests, terrain_loading_worker_count, terrain_worker_count,
};

use crate::hotbar::{MainTool, MatterMode, SelectedTerrainMaterial, SelectedTool};
use crate::{
    AppSimulation, EditorGraph, EditorHistory, EditorState, PlacedBearing,
    builder::{
        GROUND_HALF_SIZE, PlacementSnapIndex, composed_part_world_bounds, part_world_bounds,
    },
    camera::{MainCamera, PlayerCamera, PlayerState},
    controls::GameAction,
    garage,
    ui::WorldAction,
};

/// Explicit prototype spaces with independent scenery and world simulation ownership.
#[derive(States, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) enum AppSpace {
    /// Physics-free authored construction room.
    #[default]
    Garage,
    /// Finite seeded destructible world.
    World,
}

impl AppSpace {
    pub(crate) const fn uses_garage_flight(self) -> bool {
        matches!(self, Self::Garage)
    }
}

fn exposure_for_space(space: AppSpace) -> Exposure {
    match space {
        AppSpace::Garage => garage::EXPOSURE,
        AppSpace::World => Exposure::OVERCAST,
    }
}

const MAX_PENDING_TERRAIN_EDITS: usize = 4_096;
mod clumps;
const TERRAIN_EDIT_BATCH_SIZE: usize = 64;
const MAX_CONTROLLER_TICKS_PER_FRAME: usize = 4;
const STEP_VISUAL_SMOOTHING_SECONDS: f32 = 0.08;

#[derive(Component)]
struct WorldOwned;

#[derive(Component)]
struct BrushPreview;

#[derive(Component)]
struct TerrainNodeRender;

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub(crate) struct TerrainRenderMaterial {
    #[texture(0)]
    #[sampler(15)]
    grass_base_color: Handle<Image>,
    #[texture(1)]
    dirt_base_color: Handle<Image>,
    #[texture(2)]
    stone_base_color: Handle<Image>,
    #[texture(3)]
    sand_base_color: Handle<Image>,
    #[texture(4)]
    iron_base_color: Handle<Image>,
    #[texture(5)]
    graphite_base_color: Handle<Image>,
    #[texture(6)]
    grass_normal: Handle<Image>,
    #[texture(7)]
    dirt_normal: Handle<Image>,
    #[texture(8)]
    stone_normal: Handle<Image>,
    #[texture(9)]
    grass_orm: Handle<Image>,
    #[texture(10)]
    dirt_orm: Handle<Image>,
    #[texture(11)]
    stone_orm: Handle<Image>,
    #[texture(12)]
    sand_orm: Handle<Image>,
    #[texture(13)]
    iron_orm: Handle<Image>,
    #[texture(14)]
    graphite_orm: Handle<Image>,
}

impl Material for TerrainRenderMaterial {
    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        _layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        descriptor.label = Some(
            format!(
                "mechanic_terrain:{}",
                descriptor.label.as_deref().unwrap_or_default()
            )
            .into(),
        );
        Ok(())
    }

    fn fragment_shader() -> ShaderRef {
        crate::render_experiments::current().terrain_shader().into()
    }
}

#[derive(Component)]
struct TerrainMeshTask {
    node: ActiveTerrainNode,
    task: Task<Result<TerrainMeshResult, String>>,
}

struct PlayerCollisionBuild {
    editor_revision: u64,
    task: Task<Result<ConstructionCollisionIndex, String>>,
}

struct TerrainMeshResult {
    chunk: TerrainMeshChunk,
    elapsed_ms: f64,
    metrics: TerrainMeshMetrics,
    queue_wait_ms: f64,
}

struct TerrainSelectionTaskResult {
    selection: TerrainSelection,
    bounds_cache: TerrainBoundsCache,
    focus: WorldPosition,
    terrain_revision: u64,
    elapsed_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TerrainEditCommand {
    centre: WorldPosition,
    radius_metres: f64,
    previous: Option<(WorldPosition, f64)>,
    operation: TerrainEditOperation,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum TerrainEditOperation {
    Compress(mechanic_world::SoilCompression),
    Add(TerrainMaterial),
    Remove,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TerrainStrokeSample {
    centre: WorldPosition,
    radius_metres: f64,
    operation: TerrainEditOperation,
}

struct TerrainEditTaskResult {
    terrain: TerrainOctree,
    outcomes: Vec<TerrainEditOutcome>,
    elapsed_ms: f64,
}

struct PendingMaterialPublication {
    previous: TerrainOctree,
    clumps: mechanic_world::ClumpCollection,
    sources: Vec<mechanic_world::ExtractionCell>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TerrainAcknowledgements {
    edit: u64,
    mesh: u64,
    upload: u64,
    collision: u64,
}

impl TerrainAcknowledgements {
    const fn completed(self, generation: u64) -> bool {
        self.edit == generation
            && self.mesh == generation
            && self.upload == generation
            && self.collision == generation
    }
}

#[derive(Default)]
struct SpaceEditorState {
    origin: FloatingOrigin,
    graph: ConstructionGraph,
    history: EditorHistory,
    placed_bearings: Vec<PlacedBearing>,
}

#[derive(Clone, Debug)]
struct TerrainFoundation {
    part: PartId,
    support: FoundationSupport,
}

struct PendingFoundationSync {
    editor_revision: u64,
    parts: BTreeMap<PartId, PartSpec>,
    frames: BTreeMap<PartId, mechanic_core::ConstructionFrame>,
    replaced_parts: BTreeSet<PartId>,
    new_parts: Vec<PartId>,
    next_part: usize,
    foundations: Vec<TerrainFoundation>,
    index: FoundationSpatialIndex,
}

/// Values contributed to the existing F3 overlay by the terrain pipeline.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub(crate) struct WorldDiagnostics {
    pub(crate) terrain_stage_ms: f64,
    pub(crate) selection_ms: f64,
    pub(crate) selection_count: u64,
    pub(crate) column_sampling_ms: f64,
    pub(crate) polygonization_ms: f64,
    pub(crate) transitions_caps_ms: f64,
    pub(crate) bvh_construction_ms: f64,
    pub(crate) publication_ms: f64,
    pub(crate) oldest_queue_age_ms: f64,
    pub(crate) bounds_cache_bytes: u64,
    pub(crate) local_resolved_nodes: u32,
    pub(crate) local_total_nodes: u32,
    pub(crate) triangle_count: u64,
    pub(crate) streaming_backlog: u32,
    pub(crate) remesh_count: u64,
    pub(crate) overflow_flags: u32,
    pub(crate) foundation_candidate_count: u64,
    pub(crate) foundation_sample_count: u64,
    pub(crate) foundation_refresh_ms: f64,
    pub(crate) player_collision_query_ms: f64,
    pub(crate) dynamic_collision_refit_ms: f64,
    pub(crate) player_collision_candidates: u32,
    pub(crate) player_collision_contacts: u32,
    pub(crate) player_reaction_impulses: u32,
}

/// State backing the full-window Mosaic world list.
#[derive(Resource)]
pub(crate) struct WorldListState {
    phase: WorldListPhase,
    entries: Vec<SavedWorld>,
    notice: Option<String>,
    loading_progress: TerrainReadiness,
    confirming_delete: Option<std::path::PathBuf>,
    requested: Option<WorldAction>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum WorldListPhase {
    #[default]
    Picking,
    Loading,
    Playing,
}

impl FromWorld for WorldListState {
    fn from_world(world: &mut World) -> Self {
        let entries = world.resource::<WorldRuntime>().store.list();
        Self {
            phase: WorldListPhase::Picking,
            entries,
            notice: None,
            loading_progress: TerrainReadiness::default(),
            confirming_delete: None,
            requested: None,
        }
    }
}

impl WorldListState {
    #[cfg(test)]
    pub(crate) fn empty_capture_garage() -> Self {
        Self {
            phase: WorldListPhase::Playing,
            entries: Vec::new(),
            notice: None,
            loading_progress: TerrainReadiness::default(),
            confirming_delete: None,
            requested: None,
        }
    }

    pub(crate) fn enter_capture_garage(&mut self) {
        self.phase = WorldListPhase::Playing;
    }

    pub(crate) const fn is_open(&self) -> bool {
        !matches!(self.phase, WorldListPhase::Playing)
    }

    pub(crate) const fn phase(&self) -> WorldListPhase {
        self.phase
    }

    pub(crate) const fn loading_progress(&self) -> TerrainReadiness {
        self.loading_progress
    }

    pub(crate) fn entries(&self) -> &[SavedWorld] {
        &self.entries
    }

    pub(crate) fn notice(&self) -> Option<&str> {
        self.notice.as_deref()
    }

    pub(crate) fn is_confirming_delete(&self, path: &std::path::Path) -> bool {
        self.confirming_delete.as_deref() == Some(path)
    }

    pub(crate) fn act(&mut self, action: WorldAction) {
        if self.phase == WorldListPhase::Picking
            || matches!(&action, WorldAction::ExitToSelector)
                && self.phase == WorldListPhase::Playing
        {
            self.requested = Some(action);
        }
    }

    fn refresh(&mut self, store: &WorldStore) {
        self.entries = store.list();
        self.confirming_delete = None;
    }
}

#[derive(Resource)]
pub(crate) struct WorldRuntime {
    store: WorldStore,
    document: WorldDocument,
    field: Arc<TerrainField>,
    edits: TerrainOctree,
    pub(crate) clumps: mechanic_world::ClumpCollection,
    pending_breakage: mechanic_world::BreakageAccumulator,
    pending_material: Option<PendingMaterialPublication>,
    capsule: KinematicCapsule,
    floating_origin: FloatingOrigin,
    autosave: AutosaveState,
    brush_radius: f64,
    last_brush_edit: Option<TerrainStrokeSample>,
    pending_terrain_edits: VecDeque<TerrainEditCommand>,
    pending_soil: mechanic_world::SoilAccumulator,
    soil_ticks: u8,
    terrain_edit_task: Option<Task<Result<TerrainEditTaskResult, String>>>,
    terrain_edit_error: Option<String>,
    removed_cells: [u64; TerrainMaterial::COUNT],
    clock: Duration,
    load_error: Option<String>,
    garage_editor: Option<SpaceEditorState>,
    pending_garage_editor: Option<SpaceEditorState>,
    world_editor: Option<SpaceEditorState>,
    known_world_parts: BTreeMap<PartId, PartSpec>,
    known_world_frames: BTreeMap<PartId, mechanic_core::ConstructionFrame>,
    foundations: Vec<TerrainFoundation>,
    foundation_index: FoundationSpatialIndex,
    pending_foundation_sync: Option<PendingFoundationSync>,
    foundation_revision: u64,
    terrain_revision: u64,
    terrain_acknowledgements: TerrainAcknowledgements,
    pending_foundation_edit: TerrainEditBatch,
    foundation_edit_acknowledgement: u64,
    synced_editor_revision: u64,
    terrain_streamer: TerrainStreamer,
    terrain_bounds_cache: TerrainBoundsCache,
    terrain_selection_task: Option<Task<Result<TerrainSelectionTaskResult, String>>>,
    staged_terrain: BTreeMap<TerrainNodeId, TerrainMeshResult>,
    active_terrain: BTreeMap<TerrainNodeId, TerrainMeshChunk>,
    active_terrain_ready_faces: BTreeMap<TerrainNodeId, TerrainTransitionMask>,
    active_terrain_index: TerrainSpatialIndex,
    terrain_entities: BTreeMap<TerrainNodeId, Entity>,
    terrain_mesh_handles: BTreeMap<TerrainNodeId, Handle<Mesh>>,
    player_terrain_ready: bool,
    terrain_material: Option<Handle<TerrainRenderMaterial>>,
    terrain_texture_mips_pending: Vec<Handle<Image>>,
    selection_focus: Option<WorldPosition>,
    selection_clump_interests: Vec<WorldPosition>,
    selected_terrain_revision: u64,
    construction_collision: Option<ConstructionCollisionIndex>,
    collision_revision: Option<(u64, u64)>,
    collision_editor_revision: Option<u64>,
    collision_build: Option<PlayerCollisionBuild>,
    collision_failed_revision: Option<u64>,
    collision_snapshot_tick: u64,
    collision_pose_revision: u64,
    /// Last construction revision validated against the frozen target.
    frozen_editor: Option<(ConstructionGraph, Vec<PlacedBearing>)>,
    collision_poses: Vec<ConstructionBodyPose>,
    controller_accumulator: f64,
    jump_queued: bool,
    step_visual_offset: f32,
    pending_player_reactions: Vec<GpuExternalImpulse>,
    walking_suspended: bool,
}

impl WorldRuntime {
    pub(crate) fn material_publication_pending(&self) -> bool {
        self.pending_material.is_some()
    }

    pub(crate) fn material_motion(&mut self) {
        self.autosave.mutate(self.clock);
    }

    pub(crate) fn accumulate_breakage(
        &mut self,
        patches: impl Iterator<Item = mechanic_world::BreakagePatch>,
    ) {
        if self.pending_material.is_some() || self.terrain_edit_error.is_some() {
            return;
        }
        self.pending_breakage
            .discard_stale(&self.edits, &self.field);
        for patch in patches {
            self.pending_breakage
                .accumulate(&self.edits, &self.field, patch);
        }
    }

    fn begin_material_transfer(&mut self) {
        if self.pending_material.is_some()
            || self.terrain_edit_task.is_some()
            || !self.pending_terrain_edits.is_empty()
            || self.terrain_edit_error.is_some()
            || !self
                .terrain_acknowledgements
                .completed(self.terrain_revision)
        {
            return;
        }
        let mut transfer = None;
        let mut sources = Vec::new();
        for body in self
            .clumps
            .bodies
            .values()
            .filter(|body| body.can_deposit())
        {
            let Ok(centre) = body.position.cell() else {
                continue;
            };
            let mut targets = Vec::new();
            for y in -3..=2 {
                for z in -2..=2 {
                    for x in -2..=2 {
                        targets.push(mechanic_world::WorldCell::new(
                            centre.x + x,
                            centre.y + y,
                            centre.z + z,
                        ));
                    }
                }
            }
            // Do not turn occupied bucket or fragment space into solid ground.
            let radius = mechanic_world::TERRAIN_CELL_METERS * 3.0_f64.sqrt() * 0.5;
            targets.retain(|cell| {
                self.clumps.bodies.values().all(|other| {
                    other.id == body.id
                        || other.position.0.distance(cell.centre().0)
                            > other.half_extents.length() + radius
                })
            });
            if let Some(collision) = self.construction_collision.as_mut() {
                let config = mechanic_world::KinematicCapsuleConfig {
                    radius,
                    standing_height: radius * 2.0,
                    step_height: 0.0,
                    ..Default::default()
                };
                let origin = self.floating_origin.0;
                targets.retain(|cell| {
                    collision
                        .cast_capsule(
                            (cell.centre().0 - origin - DVec3::Y * radius).as_vec3(),
                            Vec3::ZERO,
                            config,
                        )
                        .is_none()
                });
            }
            transfer = self
                .clumps
                .prepare_deposition(&self.edits, &self.field, body.id, &targets);
            if transfer.is_some() {
                break;
            }
        }
        if transfer.is_none() {
            sources =
                self.pending_breakage
                    .ready(&self.edits, &self.field, self.clumps.available());
            if sources.is_empty() {
                return;
            }
            transfer = self
                .clumps
                .prepare_extraction(&self.edits, &self.field, &sources);
        }
        let Some(transfer) = transfer else {
            return;
        };
        self.pending_material = Some(PendingMaterialPublication {
            previous: self.edits.clone(),
            clumps: transfer.clumps,
            sources,
        });
        commit_terrain_edit_result(
            self,
            TerrainEditTaskResult {
                terrain: transfer.terrain,
                outcomes: vec![transfer.outcome],
                elapsed_ms: 0.0,
            },
        );
        self.pending_soil.clear();
    }
    /// Accumulates global CPU loads, submitting thresholded cells at 10 Hz.
    pub(crate) fn accumulate_soil(
        &mut self,
        patches: impl Iterator<Item = mechanic_world::SoilPatch>,
    ) {
        if self.terrain_edit_error.is_some()
            || self.pending_terrain_edits.len() >= MAX_PENDING_TERRAIN_EDITS
        {
            self.pending_soil.clear();
            return;
        }
        for patch in patches {
            let _ = self
                .pending_soil
                .accumulate(&self.edits, &self.field, patch);
        }
        self.soil_ticks += 1;
        if self.soil_ticks < 6 {
            return;
        }
        self.soil_ticks = 0;
        let ready = self.pending_soil.take_ready();
        if self.pending_terrain_edits.len().saturating_add(ready.len()) > MAX_PENDING_TERRAIN_EDITS
        {
            self.pending_soil.clear();
            return;
        }
        self.pending_terrain_edits
            .extend(ready.into_iter().map(|compression| TerrainEditCommand {
                centre: compression.cell.centre(),
                radius_metres: 0.05,
                previous: None,
                operation: TerrainEditOperation::Compress(compression),
            }));
    }

    pub(crate) const fn frozen_creation(&self) -> Option<mechanic_world::FrozenCreationDoc> {
        self.document.frozen_creation
    }

    pub(crate) fn local_to_global(&self, position: Vec3) -> WorldPosition {
        WorldPosition(self.floating_origin.0 + position.as_dvec3())
    }

    pub(crate) fn global_to_local(&self, position: WorldPosition) -> Vec3 {
        position.relative_to(self.floating_origin)
    }

    pub(crate) fn persist_frozen_creation(
        &mut self,
        frozen: Option<mechanic_world::FrozenCreationDoc>,
        graph: &ConstructionGraph,
        editor: &EditorState,
    ) -> Result<(), String> {
        let previous = self.document.clone();
        let previous_editor = self.frozen_editor.clone();
        self.document.frozen_creation = frozen;
        self.frozen_editor = frozen.map(|_| (graph.clone(), editor.placed_bearings.clone()));
        if let Err(error) = save_world_instance(self, graph, editor) {
            self.document = previous;
            self.frozen_editor = previous_editor;
            return Err(error);
        }
        Ok(())
    }

    pub(crate) fn accept_weld_freeze(
        &mut self,
        record: Option<mechanic_world::FrozenCreationDoc>,
        graph: &ConstructionGraph,
        editor: &EditorState,
    ) {
        self.document.frozen_creation = record;
        self.accept_frozen_publication(graph, editor);
        self.autosave.mutate(self.clock);
    }

    pub(crate) fn accept_frozen_publication(
        &mut self,
        graph: &ConstructionGraph,
        editor: &EditorState,
    ) {
        self.frozen_editor = self
            .document
            .frozen_creation
            .map(|_| (graph.clone(), editor.placed_bearings.clone()));
    }

    pub(crate) fn set_frozen_target(&mut self, mut frozen: mechanic_world::FrozenCreationDoc) {
        frozen.construction_generation = self.document.construction_generation;
        self.document.frozen_creation = Some(frozen);
        self.autosave.mutate(self.clock);
    }

    pub(crate) fn freeze_sphere_penetration(&self, center: Vec3, radius: f32) -> Option<f32> {
        if self.active_terrain.is_empty() || !self.player_terrain_ready {
            return None;
        }
        let scene = ActiveTerrainScene {
            chunks: &self.active_terrain,
            ready_faces: &self.active_terrain_ready_faces,
            spatial_index: &self.active_terrain_index,
        };
        Some(scene.density(self.local_to_global(center)) + radius)
    }

    /// Visit only regular collision triangles whose BVH bounds overlap a local
    /// query box. Uses the same active mesh generations as terrain collision.
    pub(crate) fn freeze_triangles_clear(
        &self,
        low: Vec3,
        high: Vec3,
        mut clear: impl FnMut([Vec3; 3]) -> bool,
    ) -> bool {
        if self.active_terrain.is_empty() || !self.player_terrain_ready {
            return false;
        }
        let low = self.local_to_global(low).0;
        let high = self.local_to_global(high).0;
        let overlaps = |bounds: mechanic_world::WorldBounds| {
            low.cmple(bounds.maximum.0).all() && high.cmpge(bounds.minimum.0).all()
        };
        let regular = mechanic_world::TerrainTriangleGroupMask::REGULAR;
        let mut stack = Vec::new();
        for chunk in self.active_terrain.values() {
            if !overlaps(chunk.triangle_bvh.bounds) {
                continue;
            }
            stack.clear();
            if !chunk.triangle_bvh.nodes.is_empty() {
                stack.push(0_usize);
            }
            while let Some(index) = stack.pop() {
                let node = &chunk.triangle_bvh.nodes[index];
                if !node.group_mask.intersects(regular) || !overlaps(node.bounds) {
                    continue;
                }
                if node.triangle_count == 0 {
                    stack.extend(
                        [node.left_child, node.right_child]
                            .into_iter()
                            .flatten()
                            .map(|i| i as usize),
                    );
                    continue;
                }
                let start = node.first_triangle as usize;
                for triangle in
                    &chunk.triangle_bvh.triangles[start..start + node.triangle_count as usize]
                {
                    if !triangle.group_mask.intersects(regular) {
                        continue;
                    }
                    let points = triangle.indices.map(|index| {
                        self.global_to_local(WorldPosition(
                            chunk.origin.0
                                + Vec3::from_array(chunk.vertices[index as usize]).as_dvec3(),
                        ))
                    });
                    if !clear(points) {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub(crate) fn horizontal_origin(&self) -> DVec2 {
        DVec2::new(self.floating_origin.0.x, self.floating_origin.0.z)
    }

    pub(crate) fn raycast_terrain(
        &self,
        local_origin: Vec3,
        direction: Vec3,
        maximum_distance: f64,
    ) -> Option<(Vec3, f32)> {
        let scene = ActiveTerrainScene {
            chunks: &self.active_terrain,
            ready_faces: &self.active_terrain_ready_faces,
            spatial_index: &self.active_terrain_index,
        };
        let global_origin = WorldPosition(self.floating_origin.0 + local_origin.as_dvec3());
        let hit = scene.raycast(global_origin, direction.as_dvec3(), maximum_distance)?;
        Some((
            hit.position.relative_to(self.floating_origin),
            hit.distance as f32,
        ))
    }

    /// Collision meshes owned by the published spatial cut. Hidden replacement
    /// meshes remain excluded until their previous owners have retired.
    /// Active terrain chunks overlapping any region physics can reach.
    ///
    /// Publishing the whole resident cut hands the contact kernel every chunk out
    /// to the render horizon, which is orders of magnitude more geometry than the
    /// simulated bodies can touch.
    pub(crate) fn physics_terrain_near<'a>(
        &'a self,
        interest: &'a [WorldBounds],
    ) -> impl Iterator<Item = &'a TerrainMeshChunk> {
        self.active_terrain.iter().filter_map(move |(id, chunk)| {
            (self.active_terrain_index.contains(*id)
                && interest
                    .iter()
                    .any(|region| region.intersects(chunk.bounds)))
            .then_some(chunk)
        })
    }

    /// True once the terrain nodes around the current walking or driving focus
    /// have active current-generation meshes.
    pub(crate) fn physics_terrain_ready(&self) -> bool {
        let readiness = self.terrain_streamer.local_readiness();
        readiness.total > 0 && readiness.is_complete()
    }

    pub(crate) fn anchored_parts(&self) -> impl Iterator<Item = PartId> + '_ {
        self.foundations
            .iter()
            .filter(|foundation| foundation.support.has_valid_anchor())
            .map(|foundation| foundation.part)
    }

    pub(crate) const fn foundation_revision(&self) -> u64 {
        self.foundation_revision
    }

    pub(crate) fn pending_player_reactions(&self) -> &[GpuExternalImpulse] {
        &self.pending_player_reactions
    }

    pub(crate) fn clear_player_reactions(&mut self) {
        self.pending_player_reactions.clear();
    }

    pub(crate) fn queue_player_reaction(&mut self, impulse: GpuExternalImpulse) {
        self.pending_player_reactions.push(impulse);
    }

    #[cfg(test)]
    fn foundations_match_editor_revision(&self, editor_revision: u64) -> bool {
        self.synced_editor_revision == editor_revision && self.pending_foundation_sync.is_none()
    }

    pub(crate) fn static_parts_for_physics(&self, editor_revision: u64) -> Option<Vec<PartId>> {
        static_parts_for_physics(
            &self.known_world_parts,
            &self.foundations,
            self.pending_foundation_sync.as_ref(),
            self.synced_editor_revision,
            editor_revision,
        )
    }

    pub(crate) fn allocate_dimension_link_id(&mut self) -> DimensionLinkId {
        let id = DimensionLinkId(self.document.next_dimension_link_id);
        self.document.next_dimension_link_id =
            self.document.next_dimension_link_id.saturating_add(1);
        id
    }

    pub(crate) fn remap_imported_dimension_links(&mut self, document: &mut CreationDocument) {
        document.remap_dimension_links(&mut self.document.next_dimension_link_id);
    }

    pub(crate) const fn active_dimension_link(&self) -> Option<DimensionLinkId> {
        self.document.active_dimension_link
    }

    pub(crate) fn toggle_dimension_link(
        &mut self,
        space: AppSpace,
        graph: &ConstructionGraph,
        part: PartId,
    ) -> Result<Option<DimensionLinkId>, String> {
        let id = graph
            .dimension_link_id(part)
            .ok_or_else(|| "aimed part is not a Dimension Link".to_owned())?;
        if self.document.active_dimension_link == Some(id) {
            let previous = self.document.clone();
            self.document.active_dimension_link = None;
            self.document.frozen_creation = None;
            if let Err(error) = self.store.save_world(&self.document) {
                self.document = previous;
                return Err(error.to_string());
            }
            return Ok(None);
        }
        if space == AppSpace::World {
            let component = graph
                .structural_component(part, self.anchored_parts())
                .map_err(|error| error.to_string())?;
            if component.touches_authored_ground() {
                return Err(
                    "Dimension Link assembly is grounded and cannot enter the Garage".to_owned(),
                );
            }
            let mut minimum = Vec3::splat(f32::INFINITY);
            let mut maximum = Vec3::splat(f32::NEG_INFINITY);
            for member in component.parts() {
                if let Some((low, high)) = composed_part_world_bounds(graph, member) {
                    minimum = minimum.min(low);
                    maximum = maximum.max(high);
                }
            }
            let size = maximum - minimum;
            if size.x > GROUND_HALF_SIZE * 2.0 + 1.0e-4
                || size.z > GROUND_HALF_SIZE * 2.0 + 1.0e-4
                || size.y > garage::BUILD_MAX_Y - garage::BUILD_MIN_Y + 1.0e-4
            {
                return Err("Dimension Link assembly exceeds the Garage build volume".to_owned());
            }
            if self.garage_editor.as_ref().is_some_and(|garage| {
                garage
                    .graph
                    .parts()
                    .any(|(_, spec)| matches!(spec, PartSpec::DimensionLink(_)))
            }) {
                return Err("Another linked creation already occupies the Garage".to_owned());
            }
        }
        let previous = self.document.clone();
        self.document.active_dimension_link = Some(id);
        self.document.frozen_creation = None;
        if let Err(error) = self.store.save_world(&self.document) {
            self.document = previous;
            return Err(error.to_string());
        }
        Ok(Some(id))
    }

    pub(crate) fn clear_active_dimension_link_if(&mut self, id: DimensionLinkId) {
        if self.document.active_dimension_link == Some(id) {
            let previous = self.document.clone();
            self.document.active_dimension_link = None;
            self.document.frozen_creation = None;
            if let Err(error) = self.store.save_world(&self.document) {
                self.document = previous;
                warn!("could not clear active Dimension Link: {error}");
            }
        }
    }
}

fn editor_from_instance(instance: WorldCreationInstanceDoc) -> Result<SpaceEditorState, String> {
    let loaded = instance
        .creation
        .into_graph()
        .map_err(|error| error.to_string())?;
    Ok(SpaceEditorState {
        origin: FloatingOrigin(instance.root_pose.translation.0),
        graph: loaded.graph,
        placed_bearings: loaded
            .sockets
            .into_iter()
            .map(|socket| PlacedBearing {
                kind: socket.kind,
                axis: socket.axis,
                source: socket.source,
                anchor: socket.anchor,
                dimensions: socket.dimensions,
            })
            .collect(),
        ..SpaceEditorState::default()
    })
}

fn load_space_editors(
    store: &WorldStore,
    document: &WorldDocument,
) -> Result<(SpaceEditorState, SpaceEditorState), String> {
    let Some((world, garage)) = store
        .load_space_pair(document)
        .map_err(|error| error.to_string())?
    else {
        return Ok((SpaceEditorState::default(), SpaceEditorState::default()));
    };
    let world = editor_from_instance(world)?;
    let garage = editor_from_instance(garage)?;
    if let Some(frozen) = document.frozen_creation
        && world.graph.dimension_link(frozen.link).is_none()
    {
        return Err("Frozen Dimension Link must belong to the saved World construction".to_owned());
    }
    let mut ids = BTreeMap::<DimensionLinkId, usize>::new();
    for graph in [&world.graph, &garage.graph] {
        for (_, spec) in graph.parts() {
            if let PartSpec::DimensionLink(link) = spec {
                *ids.entry(link.id).or_default() += 1;
            }
        }
    }
    if let Some((&duplicate, _)) = ids.iter().find(|(_, count)| **count != 1) {
        return Err(format!(
            "Dimension Link ID {duplicate:?} exists more than once"
        ));
    }
    if let Some(active) = document.active_dimension_link
        && ids.get(&active) != Some(&1)
    {
        return Err(format!(
            "active Dimension Link {active:?} does not exist exactly once"
        ));
    }
    if let Some((&id, _)) = ids.last_key_value()
        && id.0 >= document.next_dimension_link_id
    {
        return Err(format!(
            "next Dimension Link ID {} does not follow existing ID {}",
            document.next_dimension_link_id, id.0
        ));
    }
    Ok((world, garage))
}

fn application_world_store() -> WorldStore {
    crate::automation::world_store().map_or_else(
        || {
            // Tests start from an empty store: a world left behind by a play
            // session would otherwise decide what every fixture contains.
            if cfg!(test) {
                WorldStore::new(
                    std::env::temp_dir()
                        .join(format!("mechanic-test-worlds-{}", std::process::id())),
                )
            } else {
                WorldStore::platform_default().unwrap_or_else(|| WorldStore::new("worlds"))
            }
        },
        WorldStore::new,
    )
}

impl FromWorld for WorldRuntime {
    #[allow(clippy::too_many_lines)] // Initialize one coherent world and its persisted material ownership.
    fn from_world(_world: &mut World) -> Self {
        let store = application_world_store();
        let loaded = store
            .list()
            .into_iter()
            .find(|saved| saved.status == SavedWorldStatus::Current)
            .and_then(|saved| store.load_world(&saved.path).ok());
        let (document, field) = loaded.map_or_else(
            || {
                let seed = WorldSeed(0x4d45_4348_414e_4943);
                let field = TerrainField::new(seed);
                let document = WorldDocument::new("Prototype Reach", seed, field.safe_spawn());
                (document, field)
            },
            |document| {
                let field = TerrainField::with_version(document.seed, document.generator_version);
                (document, field)
            },
        );
        let (edits, clumps, load_error) = match store.load_material_state(&document.name) {
            Ok((edits, clumps)) => (edits, clumps, None),
            Err(error) => (
                TerrainOctree::default(),
                mechanic_world::ClumpCollection::default(),
                Some(error.to_string()),
            ),
        };
        let ((world_editor, garage_editor), instance_error) =
            match load_space_editors(&store, &document) {
                Ok(editors) => (editors, None),
                Err(error) => (
                    (SpaceEditorState::default(), SpaceEditorState::default()),
                    Some(error),
                ),
            };
        let load_error = load_error.or(instance_error);
        let capsule = KinematicCapsule::new(document.player_pose.translation);
        let floating_origin = world_editor.origin;
        let frozen_editor = document.frozen_creation.map(|_| {
            (
                world_editor.graph.clone(),
                world_editor.placed_bearings.clone(),
            )
        });
        Self {
            store,
            document,
            field: Arc::new(field),
            edits,
            capsule,
            floating_origin,
            autosave: AutosaveState::default(),
            brush_radius: 0.5,
            last_brush_edit: None,
            pending_terrain_edits: VecDeque::new(),
            pending_soil: mechanic_world::SoilAccumulator::default(),
            clumps,
            pending_breakage: mechanic_world::BreakageAccumulator::default(),
            pending_material: None,
            soil_ticks: 0,
            terrain_edit_task: None,
            terrain_edit_error: None,
            removed_cells: [0; TerrainMaterial::COUNT],
            clock: Duration::ZERO,
            load_error,
            garage_editor: None,
            pending_garage_editor: Some(garage_editor),
            world_editor: Some(world_editor),
            known_world_parts: BTreeMap::new(),
            known_world_frames: BTreeMap::new(),
            foundations: Vec::new(),
            foundation_index: FoundationSpatialIndex::default(),
            pending_foundation_sync: None,
            foundation_revision: 0,
            terrain_revision: 0,
            terrain_acknowledgements: TerrainAcknowledgements::default(),
            pending_foundation_edit: TerrainEditBatch::default(),
            foundation_edit_acknowledgement: 0,
            synced_editor_revision: 0,
            terrain_streamer: TerrainStreamer::default(),
            terrain_bounds_cache: TerrainBoundsCache::default(),
            terrain_selection_task: None,
            staged_terrain: BTreeMap::new(),
            active_terrain: BTreeMap::new(),
            active_terrain_ready_faces: BTreeMap::new(),
            active_terrain_index: TerrainSpatialIndex::default(),
            terrain_entities: BTreeMap::new(),
            terrain_mesh_handles: BTreeMap::new(),
            player_terrain_ready: false,
            terrain_material: None,
            terrain_texture_mips_pending: Vec::new(),
            selection_focus: None,
            selection_clump_interests: Vec::new(),
            selected_terrain_revision: u64::MAX,
            construction_collision: None,
            collision_revision: None,
            collision_editor_revision: None,
            collision_build: None,
            collision_failed_revision: None,
            collision_snapshot_tick: 0,
            collision_pose_revision: 0,
            frozen_editor,
            collision_poses: Vec::new(),
            controller_accumulator: 0.0,
            jump_queued: false,
            step_visual_offset: 0.0,
            pending_player_reactions: Vec::with_capacity(64),
            walking_suspended: false,
        }
    }
}

/// Installs the temporary F6 world/garage loop without changing benchmark scenes.
pub(crate) struct WorldPrototypePlugin;

impl Plugin for WorldPrototypePlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<AppSpace>()
            .init_resource::<WorldRuntime>()
            .init_resource::<WorldListState>()
            .init_resource::<WorldDiagnostics>()
            .add_systems(Startup, restore_initial_garage)
            .add_systems(OnEnter(AppSpace::World), enter_world)
            .add_systems(OnExit(AppSpace::World), leave_world)
            .add_systems(
                Update,
                (
                    select_and_size_brush.after(crate::controls::update_action_state),
                    walk_world
                        .after(crate::camera::update_player_camera)
                        .after(crate::poll_simulation_readbacks)
                        .after(crate::freeze::update),
                    use_brush.after(walk_world),
                    coordinate_terrain_edits.after(use_brush),
                    prepare_terrain_texture_mips,
                    schedule_terrain_remeshes.after(coordinate_terrain_edits),
                    integrate_terrain_remeshes.after(schedule_terrain_remeshes),
                    clumps::sync_clump_rendering.after(integrate_terrain_remeshes),
                    sync_world_foundations
                        .after(integrate_terrain_remeshes)
                        .after(crate::handle_build_actions),
                    autosave_world.after(integrate_terrain_remeshes),
                    save_on_exit.after(autosave_world),
                )
                    .run_if(in_state(AppSpace::World)),
            )
            .add_systems(
                Update,
                toggle_space
                    .after(crate::controls::update_action_state)
                    .run_if(world_list_closed),
            )
            .add_systems(Update, handle_world_list.after(crate::handle_pause_request));
    }
}

fn world_list_closed(list: Res<WorldListState>) -> bool {
    !list.is_open()
}

fn restore_initial_garage(
    mut runtime: ResMut<WorldRuntime>,
    mut graph: ResMut<EditorGraph>,
    mut history: ResMut<EditorHistory>,
    mut editor: ResMut<EditorState>,
) {
    let Some(garage) = runtime.pending_garage_editor.take() else {
        return;
    };
    graph.0 = garage.graph;
    *history = garage.history;
    editor.placed_bearings = garage.placed_bearings;
    editor.construction_mesh_dirty = true;
}

pub(crate) fn world_playing(list: Res<WorldListState>) -> bool {
    list.phase() == WorldListPhase::Playing
}

fn handle_world_list(
    mut list: ResMut<WorldListState>,
    mut runtime: ResMut<WorldRuntime>,
    space: Res<State<AppSpace>>,
    graph: Res<EditorGraph>,
    mut editor: ResMut<EditorState>,
    mut next_space: ResMut<NextState<AppSpace>>,
) {
    let Some(action) = list.requested.take() else {
        return;
    };
    match action {
        WorldAction::Create { name, seed } => {
            let name = name.trim();
            if name.is_empty() {
                list.notice = Some("World name cannot be blank".to_owned());
                return;
            }
            let seed = if seed.trim().is_empty() {
                None
            } else if let Ok(seed) = seed.trim().parse::<u64>() {
                Some(seed)
            } else {
                list.notice = Some("Seed must be an unsigned whole number".to_owned());
                return;
            };
            match runtime.store.create_world(name, seed) {
                Ok(document) => match install_world(&mut runtime, document) {
                    Ok(()) => {
                        list.phase = WorldListPhase::Loading;
                        list.loading_progress = TerrainReadiness::default();
                        list.notice = None;
                        next_space.set(AppSpace::World);
                    }
                    Err(error) => list.notice = Some(error),
                },
                Err(error) => list.notice = Some(error.to_string()),
            }
        }
        WorldAction::Open(path) => {
            let Some(entry) = list
                .entries
                .iter()
                .find(|entry| entry.path == path)
                .cloned()
            else {
                list.notice = Some(format!("World entry disappeared: {}", path.display()));
                list.refresh(&runtime.store);
                return;
            };
            match runtime.store.open_entry(&entry) {
                Ok(OpenWorldResult::Opened(document)) => {
                    match install_world(&mut runtime, *document) {
                        Ok(()) => {
                            list.phase = WorldListPhase::Loading;
                            list.loading_progress = TerrainReadiness::default();
                            list.notice = None;
                            next_space.set(AppSpace::World);
                        }
                        Err(error) => list.notice = Some(error),
                    }
                }
                Ok(OpenWorldResult::OutdatedRemoved { path }) => {
                    list.notice = Some(format!(
                        "Incompatible world was removed: {}",
                        path.display()
                    ));
                    list.refresh(&runtime.store);
                }
                Err(error) => list.notice = Some(error.to_string()),
            }
        }
        WorldAction::Delete(path) => {
            if list.confirming_delete.as_deref() != Some(path.as_path()) {
                list.confirming_delete = Some(path);
                list.notice = Some("Press Delete again to confirm".to_owned());
                return;
            }
            match runtime.store.delete_world(&path) {
                Ok(()) => {
                    list.notice = Some(format!("Deleted world: {}", path.display()));
                    list.refresh(&runtime.store);
                }
                Err(error) => list.notice = Some(error.to_string()),
            }
        }
        WorldAction::ExitToSelector => {
            let active_space = *space.get();
            if let Err(error) =
                save_before_world_selector(&mut runtime, active_space, &graph.0, &editor)
            {
                editor.feedback = Some(format!("Could not exit to the world selector: {error}"));
                list.notice = Some(error);
                return;
            }
            list.phase = WorldListPhase::Picking;
            list.loading_progress = TerrainReadiness::default();
            list.notice = None;
            list.refresh(&runtime.store);
            if active_space == AppSpace::World {
                next_space.set(AppSpace::Garage);
            }
        }
    }
}

fn save_before_world_selector(
    runtime: &mut WorldRuntime,
    space: AppSpace,
    graph: &ConstructionGraph,
    editor: &EditorState,
) -> Result<(), String> {
    if space == AppSpace::World {
        runtime.document.return_anchor = Some(runtime.capsule.position);
    }
    finish_terrain_edits(runtime)?;
    save_all(runtime)?;
    match space {
        AppSpace::Garage => save_garage_instance(runtime, graph, editor),
        AppSpace::World => save_world_instance(runtime, graph, editor),
    }
}

fn install_world(runtime: &mut WorldRuntime, document: WorldDocument) -> Result<(), String> {
    let (terrain, clumps) = runtime
        .store
        .load_material_state(&document.name)
        .map_err(|error| error.to_string())?;
    let (world_editor, garage_editor) = load_space_editors(&runtime.store, &document)?;
    runtime.frozen_editor = document.frozen_creation.map(|_| {
        (
            world_editor.graph.clone(),
            world_editor.placed_bearings.clone(),
        )
    });
    runtime.field = Arc::new(TerrainField::with_version(
        document.seed,
        document.generator_version,
    ));
    runtime.capsule = KinematicCapsule::new(document.player_pose.translation);
    runtime.floating_origin = world_editor.origin;
    runtime.document = document;
    runtime.edits = terrain;
    runtime.clumps = clumps;
    runtime.pending_material = None;
    runtime.pending_breakage = mechanic_world::BreakageAccumulator::default();
    runtime.world_editor = Some(world_editor);
    runtime.pending_garage_editor = Some(garage_editor);
    runtime.known_world_parts.clear();
    runtime.known_world_frames.clear();
    runtime.foundations.clear();
    runtime.foundation_index = FoundationSpatialIndex::default();
    runtime.pending_foundation_sync = None;
    runtime.foundation_revision = 0;
    runtime.terrain_revision = 0;
    runtime.terrain_acknowledgements = TerrainAcknowledgements::default();
    runtime.pending_foundation_edit = TerrainEditBatch::default();
    runtime.foundation_edit_acknowledgement = 0;
    runtime.synced_editor_revision = 0;
    reset_player_collision_publication(runtime);
    runtime.autosave = AutosaveState::default();
    runtime.last_brush_edit = None;
    runtime.pending_terrain_edits.clear();
    runtime.pending_soil.clear();
    runtime.soil_ticks = 0;
    runtime.terrain_edit_task = None;
    runtime.terrain_edit_error = None;
    runtime.removed_cells = [0; TerrainMaterial::COUNT];
    runtime.selected_terrain_revision = u64::MAX;
    runtime.selection_focus = None;
    runtime.terrain_streamer = TerrainStreamer::default();
    runtime.terrain_bounds_cache = TerrainBoundsCache::default();
    runtime.terrain_selection_task = None;
    runtime.staged_terrain.clear();
    runtime.active_terrain.clear();
    runtime.active_terrain_ready_faces.clear();
    runtime.active_terrain_index = TerrainSpatialIndex::default();
    runtime.terrain_entities.clear();
    runtime.terrain_mesh_handles.clear();
    runtime.player_terrain_ready = false;
    runtime.terrain_texture_mips_pending.clear();
    runtime.load_error = None;
    Ok(())
}

fn graph_bounds(graph: &ConstructionGraph) -> Option<(Vec3, Vec3)> {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for (part, _) in graph.parts() {
        let (low, high) = composed_part_world_bounds(graph, part)?;
        minimum = minimum.min(low);
        maximum = maximum.max(high);
    }
    minimum.is_finite().then_some((minimum, maximum))
}

fn collision_free(
    candidate: &ConstructionGraph,
    destination: &ConstructionGraph,
    index: &PlacementSnapIndex,
) -> bool {
    let identity = mechanic_core::ConstructionFrame::IDENTITY;
    if candidate
        .parts()
        .all(|(part, _)| candidate.part_frame(part) == Some(identity))
        && destination
            .parts()
            .all(|(part, _)| destination.part_frame(part) == Some(identity))
    {
        return candidate
            .parts()
            .all(|(_, incoming)| !index.overlaps(*incoming));
    }
    candidate.parts().all(|(part, incoming)| {
        let frame = candidate
            .part_frame(part)
            .expect("candidate part has frame");
        destination.parts().all(|(other, existing)| {
            let other_frame = destination
                .part_frame(other)
                .expect("destination part has frame");
            if frame == identity && other_frame == identity {
                return !crate::builder::parts_overlap(*incoming, *existing);
            }
            // Cuboids retain their exact oriented box. Curved/featured parts use
            // a conservative authored box until transfer shares evaluated overlap.
            mechanic_gpu::obb_sat(
                transfer_part_box(*incoming, frame),
                transfer_part_box(*existing, other_frame),
            )
            .is_none_or(|contact| contact.penetration <= 1.0e-4)
        })
    })
}

fn transfer_part_box(spec: PartSpec, frame: mechanic_core::ConstructionFrame) -> mechanic_gpu::Obb {
    if let Some(cuboid) = spec.as_cuboid() {
        return mechanic_gpu::Obb {
            center: frame.point(cuboid.pose.translation()),
            orientation: frame.rotation() * cuboid.pose.rotation.quaternion(),
            half_extents: cuboid.size_meters() * 0.5,
        };
    }
    let (low, high) = part_world_bounds(spec);
    mechanic_gpu::Obb {
        center: frame.point((low + high) * 0.5),
        orientation: frame.rotation(),
        half_extents: (high - low) * 0.5,
    }
}

fn terrain_clear(
    candidate: &ConstructionGraph,
    terrain: &impl TerrainDensity,
    floating_origin: FloatingOrigin,
) -> bool {
    candidate.parts().all(|(part, _)| {
        let (low, high) =
            composed_part_world_bounds(candidate, part).expect("candidate part has frame");
        let inset_low = low + Vec3::splat(0.02);
        let inset_high = high - Vec3::splat(0.02);
        [0.0_f32, 0.5, 1.0].into_iter().all(|x| {
            [0.0_f32, 0.5, 1.0].into_iter().all(|y| {
                [0.0_f32, 0.5, 1.0].into_iter().all(|z| {
                    let local = inset_low + (inset_high - inset_low) * Vec3::new(x, y, z);
                    terrain.density(WorldPosition(floating_origin.0 + local.as_dvec3())) <= 0.0
                })
            })
        })
    })
}

fn foundation_clear(
    candidate: &ConstructionGraph,
    terrain: &impl TerrainDensity,
    floating_origin: FloatingOrigin,
) -> bool {
    candidate.parts().all(|(part, _)| {
        !bounds_foundation_support(
            terrain,
            composed_part_world_bounds(candidate, part).expect("candidate part has frame"),
            floating_origin,
        )
        .has_valid_anchor()
    })
}

fn framed_part_bounds(spec: PartSpec, frame: mechanic_core::ConstructionFrame) -> (Vec3, Vec3) {
    let (low, high) = part_world_bounds(spec);
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for x in [low.x, high.x] {
        for y in [low.y, high.y] {
            for z in [low.z, high.z] {
                let point = frame.point(Vec3::new(x, y, z));
                minimum = minimum.min(point);
                maximum = maximum.max(point);
            }
        }
    }
    (minimum, maximum)
}

fn bounds_foundation_support(
    terrain: &impl TerrainDensity,
    (minimum, maximum): (Vec3, Vec3),
    floating_origin: FloatingOrigin,
) -> FoundationSupport {
    FoundationSupport::rectangular(
        terrain,
        TerrainRayHit {
            position: WorldPosition(
                floating_origin.0
                    + Vec3::new(
                        (minimum.x + maximum.x) * 0.5,
                        minimum.y,
                        (minimum.z + maximum.z) * 0.5,
                    )
                    .as_dvec3(),
            ),
            normal: Vec3::Y,
            distance: 0.0,
            material_weights: [0.0; TerrainMaterial::COUNT],
            chunk_generation: 0,
            triangle: 0,
        },
        f64::from(maximum.x - minimum.x),
        f64::from(maximum.z - minimum.z),
    )
}

fn player_clear(candidate: &ConstructionGraph, feet: Vec3) -> bool {
    const TRANSFER_CLEARANCE: f32 = 2.0;
    candidate.parts().all(|(part, _)| {
        let (low, high) =
            composed_part_world_bounds(candidate, part).expect("candidate part has frame");
        let closest_x = feet.x.clamp(low.x, high.x);
        let closest_z = feet.z.clamp(low.z, high.z);
        Vec2::new(feet.x - closest_x, feet.z - closest_z).length_squared()
            > TRANSFER_CLEARANCE * TRANSFER_CLEARANCE
    })
}

fn deterministic_offsets(maximum_radius_cells: i32) -> Vec<IVec2> {
    let mut offsets = (-maximum_radius_cells..=maximum_radius_cells)
        .flat_map(|z| (-maximum_radius_cells..=maximum_radius_cells).map(move |x| IVec2::new(x, z)))
        .filter(|offset| offset.x * offset.x + offset.y * offset.y <= maximum_radius_cells.pow(2))
        .collect::<Vec<_>>();
    offsets.sort_by_key(|offset| {
        (
            offset.x * offset.x + offset.y * offset.y,
            offset.y,
            offset.x,
        )
    });
    offsets
}

fn component_document(
    graph: &ConstructionGraph,
    bearings: &[PlacedBearing],
    name: &str,
) -> CreationDocument {
    let sockets = bearings
        .iter()
        .map(|bearing| BearingSocket {
            kind: bearing.kind,
            axis: bearing.axis,
            source: bearing.source,
            anchor: bearing.anchor,
            dimensions: bearing.dimensions,
        })
        .collect::<Vec<_>>();
    CreationDocument::from_graph(graph, name, &sockets)
}

fn detach_authored_ground(document: &mut CreationDocument) {
    document.welds.retain(|weld| {
        !matches!(weld.first.owner, FaceOwnerDoc::Ground)
            && !matches!(weld.second.owner, FaceOwnerDoc::Ground)
    });
}

fn returned_component_parts(
    graph: &ConstructionGraph,
    active: DimensionLinkId,
) -> Result<BTreeSet<PartId>, String> {
    let link = graph
        .dimension_link(active)
        .ok_or_else(|| "returned Dimension Link is missing from the World".to_owned())?;
    graph
        .structural_component(link, [])
        .map(|component| component.parts().collect())
        .map_err(|error| error.to_string())
}

fn remove_cached_foundations(
    foundations: &mut Vec<TerrainFoundation>,
    index: &mut FoundationSpatialIndex,
    parts: &BTreeSet<PartId>,
) -> bool {
    let removed = foundations
        .iter()
        .any(|foundation| parts.contains(&foundation.part));
    for &part in parts {
        index.remove(part);
    }
    foundations.retain(|foundation| !parts.contains(&foundation.part));
    removed
}

fn static_parts_for_physics(
    known_parts: &BTreeMap<PartId, PartSpec>,
    foundations: &[TerrainFoundation],
    pending: Option<&PendingFoundationSync>,
    synced_editor_revision: u64,
    editor_revision: u64,
) -> Option<Vec<PartId>> {
    // Keep the previous physics publication until every new support has been
    // sampled. A partially reconciled cache must never release a ground weld.
    if synced_editor_revision != editor_revision || pending.is_some() {
        return None;
    }
    Some(
        foundations
            .iter()
            .filter(|foundation| {
                foundation.support.has_valid_anchor() && known_parts.contains_key(&foundation.part)
            })
            .map(|foundation| foundation.part)
            .collect(),
    )
}

fn merge_document(
    destination: &SpaceEditorState,
    incoming: CreationDocument,
) -> Result<SpaceEditorState, String> {
    let mut document = component_document(
        &destination.graph,
        &destination.placed_bearings,
        "Combined construction",
    );
    document
        .append(incoming)
        .map_err(|error| error.to_string())?;
    let loaded = document.into_graph().map_err(|error| error.to_string())?;
    Ok(SpaceEditorState {
        origin: destination.origin,
        graph: loaded.graph,
        history: EditorHistory::default(),
        placed_bearings: loaded
            .sockets
            .into_iter()
            .map(|socket| PlacedBearing {
                kind: socket.kind,
                axis: socket.axis,
                source: socket.source,
                anchor: socket.anchor,
                dimensions: socket.dimensions,
            })
            .collect(),
    })
}

/// Saved creations use the same editable volume as Dimension Link transfers.
pub(crate) fn place_loaded_creation_in_garage(
    loaded: mechanic_core::LoadedCreation,
) -> Result<mechanic_core::LoadedCreation, String> {
    if loaded.graph.part_count() == 0 {
        return Ok(loaded);
    }
    let bearings = loaded
        .sockets
        .into_iter()
        .map(|socket| PlacedBearing {
            kind: socket.kind,
            axis: socket.axis,
            source: socket.source,
            anchor: socket.anchor,
            dimensions: socket.dimensions,
        })
        .collect::<Vec<_>>();
    let placed = place_in_garage(&loaded.graph, &bearings, &SpaceEditorState::default())?;
    component_document(&placed.graph, &placed.placed_bearings, &loaded.name)
        .into_graph()
        .map_err(|error| error.to_string())
}

fn place_in_garage(
    component: &ConstructionGraph,
    bearings: &[PlacedBearing],
    destination: &SpaceEditorState,
) -> Result<SpaceEditorState, String> {
    let mut original = component_document(component, bearings, "Transferred construction");
    detach_authored_ground(&mut original);
    let mut destination_index = PlacementSnapIndex::default();
    destination_index.rebuild(&destination.graph);
    let offsets = deterministic_offsets(40);
    let mut dimension_overage = true;
    for yaw in 0_u8..4 {
        let mut rotated = original.clone();
        rotated.transform_cardinal(yaw, IVec3::ZERO);
        let rotated_loaded = rotated
            .clone()
            .into_graph()
            .map_err(|error| error.to_string())?;
        let Some((minimum, maximum)) = graph_bounds(&rotated_loaded.graph) else {
            continue;
        };
        let size = maximum - minimum;
        if size.x > GROUND_HALF_SIZE * 2.0 + 1.0e-4
            || size.z > GROUND_HALF_SIZE * 2.0 + 1.0e-4
            || size.y > garage::BUILD_MAX_Y - garage::BUILD_MIN_Y + 1.0e-4
        {
            continue;
        }
        dimension_overage = false;
        let center = (minimum + maximum) * 0.5;
        let base = IVec3::new(
            (-center.x / 0.125).round() as i32,
            // Rounding down can leave an off-grid creation below the build floor.
            ((garage::BUILD_MIN_Y - minimum.y) / 0.125).ceil() as i32,
            (-center.z / 0.125).round() as i32,
        );
        for offset in &offsets {
            let mut candidate = rotated.clone();
            candidate.transform_cardinal(0, base + IVec3::new(offset.x * 4, 0, offset.y * 4));
            let loaded = candidate
                .clone()
                .into_graph()
                .map_err(|error| error.to_string())?;
            let Some((low, high)) = graph_bounds(&loaded.graph) else {
                continue;
            };
            if low.x < -GROUND_HALF_SIZE - 1.0e-4
                || high.x > GROUND_HALF_SIZE + 1.0e-4
                || low.z < -GROUND_HALF_SIZE - 1.0e-4
                || high.z > GROUND_HALF_SIZE + 1.0e-4
                || low.y < garage::BUILD_MIN_Y - 1.0e-4
                || high.y > garage::BUILD_MAX_Y + 1.0e-4
                || !collision_free(&loaded.graph, &destination.graph, &destination_index)
            {
                continue;
            }
            return merge_document(destination, candidate);
        }
    }
    Err(if dimension_overage {
        "Linked assembly exceeds the Garage dimensions in every orientation".to_owned()
    } else {
        "Garage has no collision-free volume for the linked assembly".to_owned()
    })
}

fn place_in_world(
    component: &ConstructionGraph,
    bearings: &[PlacedBearing],
    destination: &SpaceEditorState,
    target: Vec3,
    terrain: &impl TerrainDensity,
    floating_origin: FloatingOrigin,
) -> Result<SpaceEditorState, String> {
    let mut original = component_document(component, bearings, "Returned construction");
    detach_authored_ground(&mut original);
    let loaded = original
        .clone()
        .into_graph()
        .map_err(|error| error.to_string())?;
    let (minimum, maximum) =
        graph_bounds(&loaded.graph).ok_or_else(|| "linked assembly is empty".to_owned())?;
    let mut destination_index = PlacementSnapIndex::default();
    destination_index.rebuild(&destination.graph);
    let center = (minimum + maximum) * 0.5;
    let horizontal_base = IVec2::new(
        ((target.x - center.x) / 0.125).round() as i32,
        ((target.z - center.z) / 0.125).round() as i32,
    );
    for offset in deterministic_offsets(40) {
        let horizontal = horizontal_base + offset * 4;
        let local_center = Vec3::new(
            center.x + horizontal.x as f32 * 0.125,
            target.y,
            center.z + horizontal.y as f32 * 0.125,
        );
        let ray_origin =
            WorldPosition(floating_origin.0 + (local_center + Vec3::Y * 20.0).as_dvec3());
        let Some(surface) = raycast_density(terrain, ray_origin, DVec3::NEG_Y, 40.0) else {
            continue;
        };
        let surface_y = surface.position.relative_to(floating_origin).y;
        // Keep the authored pose above the foundation probe; live physics owns the landing.
        let vertical = ((surface_y - minimum.y) / 0.125).ceil() as i32 + 1;
        for clearance in 0..=1 {
            let mut candidate = original.clone();
            candidate.transform_cardinal(
                0,
                IVec3::new(horizontal.x, vertical + clearance, horizontal.y),
            );
            let loaded = candidate
                .clone()
                .into_graph()
                .map_err(|error| error.to_string())?;
            if collision_free(&loaded.graph, &destination.graph, &destination_index)
                && terrain_clear(&loaded.graph, terrain, floating_origin)
                && foundation_clear(&loaded.graph, terrain, floating_origin)
                && player_clear(&loaded.graph, target)
            {
                return merge_document(destination, candidate);
            }
        }
    }
    Err("No terrain- and construction-safe return placement was found within 20 m".to_owned())
}

enum TransferAttempt {
    PlayerOnly,
    Transferred,
    Refused(String),
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn transfer_active_assembly(
    space: AppSpace,
    runtime: &mut WorldRuntime,
    graph: &mut ConstructionGraph,
    history: &mut EditorHistory,
    editor: &mut EditorState,
    simulation: &mut AppSimulation,
) -> TransferAttempt {
    let Some(active) = runtime.document.active_dimension_link else {
        return TransferAttempt::PlayerOnly;
    };
    let Some(link_part) = graph.dimension_link(active) else {
        return TransferAttempt::PlayerOnly;
    };
    let component = match graph.structural_component(link_part, runtime.anchored_parts()) {
        Ok(component) => component,
        Err(error) => return TransferAttempt::Refused(error.to_string()),
    };
    if space == AppSpace::World && component.touches_authored_ground() {
        return TransferAttempt::Refused(
            "Linked assembly is terrain-anchored or welded to ground".to_owned(),
        );
    }
    let partition = graph.partition(&component);
    let (component_bearings, remainder_bearings): (Vec<_>, Vec<_>) = editor
        .placed_bearings
        .iter()
        .copied()
        .partition(|bearing| {
            matches!(bearing.source.owner, mechanic_core::FaceOwner::Part(part) if component.contains(part))
        });
    let destination = match space {
        AppSpace::World => {
            let Some(garage) = runtime.garage_editor.as_ref() else {
                return TransferAttempt::Refused("Garage state is unavailable".to_owned());
            };
            match place_in_garage(&partition.component, &component_bearings, garage) {
                Ok(placed) => placed,
                Err(error) => return TransferAttempt::Refused(error),
            }
        }
        AppSpace::Garage => {
            let Some(world) = runtime.world_editor.as_ref() else {
                return TransferAttempt::Refused("World state is unavailable".to_owned());
            };
            let anchor = runtime
                .document
                .return_anchor
                .unwrap_or(runtime.document.player_pose.translation)
                .relative_to(runtime.floating_origin);
            let terrain = TerrainScene {
                field: &runtime.field,
                edits: &runtime.edits,
            };
            match place_in_world(
                &partition.component,
                &component_bearings,
                world,
                anchor,
                &terrain,
                runtime.floating_origin,
            ) {
                Ok(placed) => placed,
                Err(error) => return TransferAttempt::Refused(error),
            }
        }
    };
    let returned_world_parts = match space {
        AppSpace::World => BTreeSet::new(),
        AppSpace::Garage => match returned_component_parts(&destination.graph, active) {
            Ok(parts) => parts,
            Err(error) => return TransferAttempt::Refused(error),
        },
    };
    let remainder = SpaceEditorState {
        origin: runtime.floating_origin,
        graph: partition.remainder,
        history: EditorHistory::default(),
        placed_bearings: remainder_bearings,
    };
    let (world_state, garage_state) = match space {
        AppSpace::World => (&remainder, &destination),
        AppSpace::Garage => (&destination, &remainder),
    };
    let mut world_doc = space_instance(
        &world_state.graph,
        &world_state.placed_bearings,
        "World construction",
    );
    world_doc.root_pose.translation = WorldPosition(world_state.origin.0);
    let garage_doc = space_instance(
        &garage_state.graph,
        &garage_state.placed_bearings,
        "Garage construction",
    );
    let previous_document = runtime.document.clone();
    runtime.document.frozen_creation = None;
    runtime.document.instances = (world_state.graph.part_count() != 0)
        .then(|| WorldInstanceIndexDoc {
            id: 1,
            name: "World construction".to_owned(),
        })
        .into_iter()
        .collect();
    if let Err(error) =
        runtime
            .store
            .save_space_pair(&mut runtime.document, &world_doc, &garage_doc)
    {
        runtime.document = previous_document;
        return TransferAttempt::Refused(format!("Could not persist transfer: {error}"));
    }
    match space {
        AppSpace::World => runtime.garage_editor = Some(destination),
        AppSpace::Garage => {
            let removed = remove_cached_foundations(
                &mut runtime.foundations,
                &mut runtime.foundation_index,
                &returned_world_parts,
            );
            runtime.pending_foundation_sync = None;
            runtime.synced_editor_revision = destination.history.current_revision.wrapping_add(1);
            if removed {
                runtime.foundation_revision = runtime.foundation_revision.wrapping_add(1);
            }
            runtime.world_editor = Some(destination);
        }
    }
    graph.clone_from(&remainder.graph);
    *history = EditorHistory::default();
    editor.placed_bearings = remainder.placed_bearings;
    editor.construction_mesh_dirty = true;
    *simulation = AppSimulation::default();
    runtime.autosave.saved();
    TransferAttempt::Transferred
}

#[allow(clippy::too_many_arguments)]
fn toggle_space(
    actions: Res<ButtonInput<GameAction>>,
    space: Res<State<AppSpace>>,
    mut next: ResMut<NextState<AppSpace>>,
    mut runtime: ResMut<WorldRuntime>,
    player: Res<PlayerState>,
    mut graph: ResMut<EditorGraph>,
    mut history: ResMut<EditorHistory>,
    mut editor: ResMut<EditorState>,
    mut list: ResMut<WorldListState>,
    mut simulation: ResMut<AppSimulation>,
) {
    if !actions.just_pressed(GameAction::ToggleSpace) {
        return;
    }
    let current = *space.get();
    if current == AppSpace::World {
        if runtime.terrain_edit_task.is_some() || !runtime.pending_terrain_edits.is_empty() {
            editor.feedback = Some("Finishing queued terrain edits…".to_owned());
            return;
        }
        let global = WorldPosition(runtime.floating_origin.0 + player.position.as_dvec3());
        runtime.document.return_anchor = Some(global);
        runtime.document.player_pose.translation = global;
        if let Err(error) = save_all(&mut runtime) {
            editor.feedback = Some(error);
            return;
        }
    }
    let transferred = match transfer_active_assembly(
        current,
        &mut runtime,
        &mut graph.0,
        &mut history,
        &mut editor,
        &mut simulation,
    ) {
        TransferAttempt::Transferred => {
            editor.feedback = Some(match current {
                AppSpace::World => "Transferred linked assembly to the Garage".to_owned(),
                AppSpace::Garage => "Returned linked assembly to the World".to_owned(),
            });
            true
        }
        TransferAttempt::Refused(error) => {
            editor.feedback = Some(error);
            return;
        }
        TransferAttempt::PlayerOnly => false,
    };
    match current {
        AppSpace::Garage => {
            if !transferred
                && let Err(error) = save_garage_instance(&mut runtime, &graph.0, &editor)
            {
                editor.feedback = Some(error);
                return;
            }
            list.phase = WorldListPhase::Loading;
            list.loading_progress = TerrainReadiness::default();
            list.notice = None;
            next.set(AppSpace::World);
        }
        AppSpace::World => {
            if !transferred && let Err(error) = save_world_instance(&mut runtime, &graph.0, &editor)
            {
                editor.feedback = Some(error);
                return;
            }
            next.set(AppSpace::Garage);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn enter_world(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut terrain_materials: ResMut<Assets<TerrainRenderMaterial>>,
    mut clear: ResMut<ClearColor>,
    mut runtime: ResMut<WorldRuntime>,
    mut player: ResMut<PlayerState>,
    mut graph: ResMut<EditorGraph>,
    mut history: ResMut<EditorHistory>,
    mut editor: ResMut<EditorState>,
    mut named: Query<(&Name, &mut Visibility)>,
    camera: Single<(&mut DistanceFog, &mut Exposure), With<MainCamera>>,
    mut diagnostics: ResMut<WorldDiagnostics>,
) {
    for (name, mut visibility) in &mut named {
        if name.as_str().starts_with("Garage") {
            *visibility = Visibility::Hidden;
        }
    }
    clear.0 = Color::srgb_u8(69, 88, 102);
    let (mut fog, mut exposure) = camera.into_inner();
    *fog = DistanceFog {
        color: clear.0,
        falloff: FogFalloff::Exponential { density: 0.0022 },
        ..default()
    };
    *exposure = exposure_for_space(AppSpace::World);
    debug_assert!(runtime.garage_editor.is_none());
    runtime.garage_editor =
        Some(
            runtime
                .pending_garage_editor
                .take()
                .unwrap_or_else(|| SpaceEditorState {
                    origin: FloatingOrigin::default(),
                    graph: core::mem::take(&mut graph.0),
                    history: core::mem::take(&mut *history),
                    placed_bearings: core::mem::take(&mut editor.placed_bearings),
                }),
        );
    let world_editor = runtime.world_editor.take().unwrap_or_default();
    restore_world_player(&mut runtime, &mut player, world_editor.origin);
    graph.0 = world_editor.graph;
    *history = world_editor.history;
    editor.placed_bearings = world_editor.placed_bearings;
    crate::cancel_transient_editor_state(&mut graph.0, &mut editor);
    editor.construction_mesh_dirty = true;
    reset_player_collision_publication(&mut runtime);

    editor.feedback = runtime.load_error.clone().or_else(|| {
        Some("World — Shift sprint · Space jump · F6 Garage · Shift+4 terrain mode".to_owned())
    });

    spawn_world_terrain(
        &mut commands,
        &asset_server,
        &mut meshes,
        &mut materials,
        &mut terrain_materials,
        &mut runtime,
        &mut diagnostics,
    );
}

fn restore_world_player(
    runtime: &mut WorldRuntime,
    player: &mut PlayerState,
    origin: FloatingOrigin,
) {
    runtime.floating_origin = origin;
    let start = runtime
        .document
        .return_anchor
        .unwrap_or(runtime.document.player_pose.translation);
    runtime.capsule = KinematicCapsule::new(start);
    player.position = start.relative_to(runtime.floating_origin);
    player.seat = None;
}

fn spawn_world_terrain(
    commands: &mut Commands,
    asset_server: &AssetServer,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    terrain_materials: &mut Assets<TerrainRenderMaterial>,
    runtime: &mut WorldRuntime,
    diagnostics: &mut WorldDiagnostics,
) {
    runtime.terrain_streamer = TerrainStreamer::default();
    runtime.terrain_selection_task = None;
    runtime.pending_terrain_edits.clear();
    runtime.pending_soil.clear();
    runtime.soil_ticks = 0;
    runtime.terrain_edit_task = None;
    runtime.terrain_edit_error = None;
    runtime.last_brush_edit = None;
    runtime.staged_terrain.clear();
    runtime.active_terrain.clear();
    runtime.active_terrain_ready_faces.clear();
    runtime.active_terrain_index = TerrainSpatialIndex::default();
    runtime.terrain_entities.clear();
    runtime.terrain_mesh_handles.clear();
    runtime.player_terrain_ready = false;
    runtime.selection_focus = None;
    runtime.selected_terrain_revision = u64::MAX;
    let (terrain_material, pending_mips) = terrain_render_material(asset_server);
    runtime.terrain_texture_mips_pending = pending_mips;
    runtime.terrain_material = Some(terrain_materials.add(terrain_material));
    diagnostics.triangle_count = 0;

    let preview_mesh = Sphere::new(1.0)
        .mesh()
        .ico(3)
        .expect("valid sphere subdivision");
    let preview_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.25, 0.85, 1.0, 0.22),
        emissive: LinearRgba::rgb(0.05, 0.28, 0.42),
        alpha_mode: AlphaMode::Blend,
        cull_mode: Some(Face::Back),
        ..default()
    });
    commands.spawn((
        Name::new("Terrain brush preview"),
        Mesh3d(meshes.add(preview_mesh)),
        MeshMaterial3d(preview_material),
        Visibility::Hidden,
        BrushPreview,
        WorldOwned,
    ));
    commands.spawn((
        Name::new("World sun"),
        DirectionalLight {
            color: Color::srgb_u8(218, 204, 190),
            illuminance: 18_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, -0.55, 0.0)),
        WorldOwned,
    ));
}

fn terrain_render_material(
    asset_server: &AssetServer,
) -> (TerrainRenderMaterial, Vec<Handle<Image>>) {
    let texture = |path: &'static str, is_srgb: bool| {
        asset_server
            .load_builder()
            .with_settings(move |settings: &mut bevy::image::ImageLoaderSettings| {
                crate::configure_repeating_texture(settings, is_srgb);
            })
            .load(path)
    };
    let grass_base_color = texture("terrain/grass/grass_base_color.png", true);
    let dirt_base_color = texture("terrain/dirt/dirt_base_color.png", true);
    let stone_base_color = texture("terrain/stone/stone_base_color.png", true);
    let sand_base_color = texture("materials/sand/sand_base_color.png", true);
    let iron_base_color = texture("materials/iron/iron_base_color.png", true);
    let graphite_base_color = texture("materials/graphite/graphite_base_color.png", true);
    let grass_normal = texture("terrain/grass/grass_normal.png", false);
    let dirt_normal = texture("terrain/dirt/dirt_normal.png", false);
    let stone_normal = texture("terrain/stone/stone_normal.png", false);
    let grass_orm = texture("terrain/grass/grass_orm.png", false);
    let dirt_orm = texture("terrain/dirt/dirt_orm.png", false);
    let stone_orm = texture("terrain/stone/stone_orm.png", false);
    let sand_orm = texture("materials/sand/sand_orm.png", false);
    let iron_orm = texture("materials/iron/iron_orm.png", false);
    let graphite_orm = texture("materials/graphite/graphite_orm.png", false);
    let pending_mips = vec![
        grass_base_color.clone(),
        dirt_base_color.clone(),
        stone_base_color.clone(),
        sand_base_color.clone(),
        iron_base_color.clone(),
        graphite_base_color.clone(),
        grass_normal.clone(),
        dirt_normal.clone(),
        stone_normal.clone(),
        grass_orm.clone(),
        dirt_orm.clone(),
        stone_orm.clone(),
        sand_orm.clone(),
        iron_orm.clone(),
        graphite_orm.clone(),
    ];
    let material = TerrainRenderMaterial {
        grass_base_color,
        dirt_base_color,
        stone_base_color,
        sand_base_color,
        iron_base_color,
        graphite_base_color,
        grass_normal,
        dirt_normal,
        stone_normal,
        grass_orm,
        dirt_orm,
        stone_orm,
        sand_orm,
        iron_orm,
        graphite_orm,
    };
    (material, pending_mips)
}

fn prepare_terrain_texture_mips(
    mut images: ResMut<Assets<Image>>,
    mut runtime: ResMut<WorldRuntime>,
) {
    let Some(index) = runtime
        .terrain_texture_mips_pending
        .iter()
        .position(|handle| images.contains(handle.id()))
    else {
        return;
    };
    let handle = runtime.terrain_texture_mips_pending.swap_remove(index);
    let Some(mut image) = images.get_mut(&handle) else {
        return;
    };
    if let Err(error) = generate_rgba8_mip_chain(&mut image) {
        runtime.load_error = Some(error);
        runtime.terrain_texture_mips_pending.clear();
    }
}

pub(crate) fn generate_rgba8_mip_chain(image: &mut Image) -> Result<(), String> {
    if image.texture_descriptor.mip_level_count > 1 {
        return Ok(());
    }
    if !matches!(
        image.texture_descriptor.format,
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
    ) {
        return Err(format!(
            "texture has unsupported runtime format {:?}",
            image.texture_descriptor.format
        ));
    }
    let width = image.texture_descriptor.size.width;
    let height = image.texture_descriptor.size.height;
    if image.texture_descriptor.size.depth_or_array_layers != 1 {
        return Err("texture must be a single 2D image".to_owned());
    }
    let expected_top_bytes = usize::try_from(u64::from(width) * u64::from(height) * 4)
        .map_err(|_| "terrain texture dimensions overflow memory size".to_owned())?;
    let Some(top) = image.data.as_ref() else {
        return Err("texture has no CPU pixel data".to_owned());
    };
    if top.len() != expected_top_bytes {
        return Err(format!(
            "texture contains {} bytes, expected {expected_top_bytes}",
            top.len()
        ));
    }

    let level_count = 32 - width.max(height).leading_zeros();
    let mut chain = Vec::with_capacity(full_rgba8_mip_byte_count(width, height));
    chain.extend_from_slice(top);
    let mut previous = top.clone();
    let mut previous_width = width;
    let mut previous_height = height;
    while previous_width > 1 || previous_height > 1 {
        let next_width = (previous_width / 2).max(1);
        let next_height = (previous_height / 2).max(1);
        let mut next = vec![0_u8; (next_width * next_height * 4) as usize];
        for y in 0..next_height {
            for x in 0..next_width {
                let source_x = x * 2;
                let source_y = y * 2;
                let adjacent_x = (source_x + 1).min(previous_width - 1);
                let adjacent_y = (source_y + 1).min(previous_height - 1);
                for channel in 0..4_u32 {
                    let source = |sample_x: u32, sample_y: u32| {
                        previous[((sample_y * previous_width + sample_x) * 4 + channel) as usize]
                    };
                    let sum = u16::from(source(source_x, source_y))
                        + u16::from(source(adjacent_x, source_y))
                        + u16::from(source(source_x, adjacent_y))
                        + u16::from(source(adjacent_x, adjacent_y));
                    next[((y * next_width + x) * 4 + channel) as usize] =
                        u8::try_from((sum + 2) / 4).expect("four bytes average to one byte");
                }
            }
        }
        chain.extend_from_slice(&next);
        previous = next;
        previous_width = next_width;
        previous_height = next_height;
    }
    image.data = Some(chain);
    image.texture_descriptor.mip_level_count = level_count;
    Ok(())
}

fn full_rgba8_mip_byte_count(mut width: u32, mut height: u32) -> usize {
    let mut texel_count = 0_u64;
    loop {
        texel_count = texel_count.saturating_add(u64::from(width) * u64::from(height));
        if width == 1 && height == 1 {
            break;
        }
        width = (width / 2).max(1);
        height = (height / 2).max(1);
    }
    usize::try_from(texel_count.saturating_mul(4)).unwrap_or(usize::MAX)
}

#[allow(clippy::too_many_arguments)]
fn leave_world(
    mut commands: Commands,
    entities: Query<Entity, With<WorldOwned>>,
    mut named: Query<(&Name, &mut Visibility), Without<WorldOwned>>,
    mut clear: ResMut<ClearColor>,
    mut runtime: ResMut<WorldRuntime>,
    mut simulation: ResMut<AppSimulation>,
    mut graph: ResMut<EditorGraph>,
    mut history: ResMut<EditorHistory>,
    mut editor: ResMut<EditorState>,
    camera: Single<(&mut DistanceFog, &mut Exposure), With<MainCamera>>,
) {
    let _ = save_all(&mut runtime);
    let _ = save_world_instance(&mut runtime, &graph.0, &editor);
    *simulation = AppSimulation::default();
    runtime.world_editor = Some(SpaceEditorState {
        origin: runtime.floating_origin,
        graph: core::mem::take(&mut graph.0),
        history: core::mem::take(&mut *history),
        placed_bearings: core::mem::take(&mut editor.placed_bearings),
    });
    runtime.terrain_streamer = TerrainStreamer::default();
    runtime.terrain_selection_task = None;
    runtime.pending_terrain_edits.clear();
    runtime.pending_soil.clear();
    runtime.soil_ticks = 0;
    runtime.terrain_edit_task = None;
    runtime.terrain_edit_error = None;
    runtime.last_brush_edit = None;
    runtime.staged_terrain.clear();
    runtime.active_terrain.clear();
    runtime.active_terrain_ready_faces.clear();
    runtime.active_terrain_index = TerrainSpatialIndex::default();
    runtime.terrain_entities.clear();
    runtime.terrain_mesh_handles.clear();
    runtime.terrain_material = None;
    runtime.terrain_texture_mips_pending.clear();
    runtime.selection_focus = None;
    let garage_editor = runtime
        .garage_editor
        .take()
        .expect("entering World stores the Garage editor");
    graph.0 = garage_editor.graph;
    *history = garage_editor.history;
    editor.placed_bearings = garage_editor.placed_bearings;
    crate::cancel_transient_editor_state(&mut graph.0, &mut editor);
    editor.construction_mesh_dirty = true;
    for entity in &entities {
        commands.entity(entity).despawn();
    }
    for (name, mut visibility) in &mut named {
        if name.as_str().starts_with("Garage") {
            *visibility = Visibility::Inherited;
        }
    }
    clear.0 = garage::VOID_COLOR;
    let (mut fog, mut exposure) = camera.into_inner();
    *fog = garage::fog();
    *exposure = exposure_for_space(AppSpace::Garage);
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn walk_world(
    time: Res<Time>,
    actions: Res<ButtonInput<GameAction>>,
    mut camera: Single<&mut PlayerCamera, With<MainCamera>>,
    list: Res<WorldListState>,
    graph: Res<EditorGraph>,
    history: Res<EditorHistory>,
    simulation: Res<AppSimulation>,
    mut runtime: ResMut<WorldRuntime>,
    mut player: ResMut<PlayerState>,
    mut diagnostics: ResMut<WorldDiagnostics>,
) {
    if runtime.material_publication_pending() {
        return;
    }
    sync_player_construction_collision(
        &mut runtime,
        &simulation,
        &graph.0,
        history.current_revision,
    );
    if list.phase() != WorldListPhase::Playing {
        runtime.capsule.velocity = bevy::math::DVec3::ZERO;
        runtime.jump_queued = false;
        return;
    }
    let reactions_are_authoritative = simulation.is_running()
        && runtime.collision_revision == simulation.world_revision
        && runtime.collision_editor_revision == Some(history.current_revision);
    if player.seat.is_some() {
        runtime.capsule.velocity = DVec3::ZERO;
        runtime.capsule.clear_support();
        runtime.controller_accumulator = 0.0;
        runtime.jump_queued = false;
        runtime.step_visual_offset = 0.0;
        runtime.walking_suspended = true;
        return;
    }
    if runtime.walking_suspended {
        runtime.capsule.position =
            WorldPosition(runtime.floating_origin.0 + player.position.as_dvec3());
        runtime.capsule.velocity = DVec3::ZERO;
        runtime.capsule.clear_support();
        runtime.step_visual_offset = 0.0;
        runtime.walking_suspended = false;
    }
    let mut local = Vec2::ZERO;
    local.y += if actions.pressed(GameAction::MoveForward) {
        1.0
    } else {
        0.0
    };
    local.y -= if actions.pressed(GameAction::MoveBackward) {
        1.0
    } else {
        0.0
    };
    local.x += if actions.pressed(GameAction::MoveRight) {
        1.0
    } else {
        0.0
    };
    local.x -= if actions.pressed(GameAction::MoveLeft) {
        1.0
    } else {
        0.0
    };
    let forward = camera.look_rotation() * Vec3::NEG_Z;
    let right = camera.look_rotation() * Vec3::X;
    let movement = (right * local.x + forward * local.y)
        .with_y(0.0)
        .normalize_or_zero();
    if !runtime.player_terrain_ready {
        let ready = runtime.active_terrain.iter().any(|(&id, chunk)| {
            terrain_chunk_has_collision_near(
                chunk,
                runtime
                    .active_terrain_ready_faces
                    .get(&id)
                    .copied()
                    .unwrap_or_default(),
                &runtime.capsule,
            )
        });
        if !ready {
            runtime.capsule.velocity = bevy::math::DVec3::ZERO;
            player.position = runtime
                .capsule
                .position
                .relative_to(runtime.floating_origin);
            return;
        }
        runtime.player_terrain_ready = true;
    }
    let (controller_ticks, jump_queued) = {
        let WorldRuntime {
            controller_accumulator,
            jump_queued,
            ..
        } = &mut *runtime;
        advance_controller(
            controller_accumulator,
            jump_queued,
            f64::from(time.delta_secs()),
            actions.just_pressed(GameAction::Jump),
        )
    };
    if controller_ticks == 0 {
        return;
    }

    let mut collision_index = runtime.construction_collision.take();
    let mut pending_reactions = core::mem::take(&mut runtime.pending_player_reactions);
    let scene = ActiveTerrainScene {
        chunks: &runtime.active_terrain,
        ready_faces: &runtime.active_terrain_ready_faces,
        spatial_index: &runtime.active_terrain_index,
    };
    let mut capsule = runtime.capsule;
    let mut resolved_contacts = 0_u32;
    let mut queued_reactions = 0_u32;
    let mut stepped_height = 0.0_f32;
    for tick in 0..controller_ticks {
        let mut collision_scene = KinematicCollisionScene {
            terrain: &scene,
            construction: collision_index.as_mut(),
            floating_origin: runtime.floating_origin.0,
        };
        let result = capsule.tick(
            &mut collision_scene,
            KinematicInput {
                movement: bevy::math::DVec2::new(f64::from(movement.x), f64::from(movement.z)),
                sprint: actions.pressed(GameAction::Sprint),
                jump: tick == 0 && jump_queued,
                jump_held: actions.pressed(GameAction::Jump),
            },
            mechanic_core::TICK_SECONDS,
        );
        camera.yaw += result.support_yaw_delta;
        resolved_contacts = resolved_contacts.saturating_add(result.resolved_contacts);
        stepped_height += result.stepped_height;
        for reaction in result.reaction_impulses() {
            if reactions_are_authoritative {
                pending_reactions.push(GpuExternalImpulse::new(
                    reaction.compound_index,
                    reaction.world_point,
                    reaction.impulse,
                ));
                queued_reactions = queued_reactions.saturating_add(1);
            }
        }
    }
    if let Some(index) = collision_index.as_ref() {
        let metrics = index.metrics();
        diagnostics.player_collision_query_ms = metrics.query_ms;
        diagnostics.dynamic_collision_refit_ms = metrics.dynamic_refit_ms;
        diagnostics.player_collision_candidates = metrics.candidate_count;
    }
    diagnostics.player_collision_contacts = resolved_contacts;
    diagnostics.player_reaction_impulses = queued_reactions;
    runtime.pending_player_reactions = pending_reactions;
    runtime.construction_collision = collision_index;
    runtime.capsule = capsule;
    // Graph frames, bearing anchors, and physics bodies share this persisted origin.
    // Moving only terrain transforms would move construction relative to the world.
    let capsule_position = capsule.position.relative_to(runtime.floating_origin);
    runtime.step_visual_offset = smooth_step_visual_offset(
        runtime.step_visual_offset,
        stepped_height,
        time.delta_secs(),
    );
    player.position = capsule_position + Vec3::Y * runtime.step_visual_offset;
    runtime.document.player_pose.translation = capsule.position;
}

fn advance_controller(
    accumulator: &mut f64,
    jump_queued: &mut bool,
    delta_seconds: f64,
    jump_pressed: bool,
) -> (usize, bool) {
    *jump_queued |= jump_pressed;
    *accumulator += delta_seconds;
    let ticks =
        ((*accumulator / mechanic_core::TICK_SECONDS) as usize).min(MAX_CONTROLLER_TICKS_PER_FRAME);
    if ticks == 0 {
        return (0, false);
    }
    *accumulator -= mechanic_core::TICK_SECONDS * ticks as f64;
    (ticks, core::mem::take(jump_queued))
}

fn smooth_step_visual_offset(current: f32, stepped_height: f32, delta_seconds: f32) -> f32 {
    let offset = current - stepped_height;
    let smoothed = offset * (-delta_seconds / STEP_VISUAL_SMOOTHING_SECONDS).exp();
    if smoothed.abs() < 1.0e-4 {
        0.0
    } else {
        smoothed
    }
}

fn sync_player_construction_collision(
    runtime: &mut WorldRuntime,
    simulation: &AppSimulation,
    graph: &ConstructionGraph,
    editor_revision: u64,
) {
    let authoritative_revision = simulation
        .world_revision
        .filter(|revision| revision.0 == editor_revision);
    if authoritative_revision.is_some()
        && (runtime.collision_revision != authoritative_revision
            || runtime.collision_editor_revision != Some(editor_revision))
    {
        install_player_collision(
            runtime,
            simulation
                .creation
                .as_ref()
                .map(ConstructionCollisionIndex::new),
            editor_revision,
            authoritative_revision,
        );
        runtime.collision_build = None;
        runtime.collision_failed_revision = None;
    } else if authoritative_revision.is_none() {
        sync_provisional_player_collision(runtime, graph, editor_revision);
    }

    let Some(physics_revision) = runtime.collision_revision else {
        return;
    };
    if simulation.world_revision != Some(physics_revision) {
        return;
    }
    let Some(index) = runtime.construction_collision.as_mut() else {
        return;
    };
    if runtime.collision_snapshot_tick == simulation.snapshot_tick
        && runtime.collision_pose_revision == simulation.pose_revision
        && runtime.collision_poses.len() == simulation.transforms.len()
    {
        return;
    }
    let tick_delta = simulation
        .snapshot_tick
        .saturating_sub(runtime.collision_snapshot_tick);
    let elapsed = tick_delta.max(1) as f32 * mechanic_core::TICK_SECONDS_F32;
    runtime
        .collision_poses
        .resize(simulation.transforms.len(), ConstructionBodyPose::default());
    for (body, transform) in simulation.transforms.iter().copied().enumerate() {
        let translation = Vec3::from_slice(&transform.position[..3]);
        let rotation = Quat::from_array(transform.rotation).normalize();
        let previous = index.body_pose(body as u32).unwrap_or_default();
        let linear_velocity = if runtime.collision_snapshot_tick == 0 {
            Vec3::ZERO
        } else {
            (translation - previous.translation) / elapsed
        };
        let angular_velocity = if runtime.collision_snapshot_tick == 0 {
            Vec3::ZERO
        } else {
            angular_velocity(previous.rotation, rotation, elapsed)
        };
        runtime.collision_poses[body] = ConstructionBodyPose {
            translation,
            rotation,
            linear_velocity,
            angular_velocity,
        };
    }
    if index.refit_dynamic(&runtime.collision_poses) {
        runtime.collision_snapshot_tick = simulation.snapshot_tick;
        runtime.collision_pose_revision = simulation.pose_revision;
    }
}

fn sync_provisional_player_collision(
    runtime: &mut WorldRuntime,
    graph: &ConstructionGraph,
    editor_revision: u64,
) {
    if runtime
        .collision_build
        .as_ref()
        .is_some_and(|build| build.editor_revision != editor_revision)
    {
        runtime.collision_build = None;
    }
    let completed = runtime
        .collision_build
        .as_mut()
        .and_then(|build| check_ready(&mut build.task));
    if let Some(completed) = completed {
        let completed_revision = runtime
            .collision_build
            .take()
            .expect("completed collision build is still owned")
            .editor_revision;
        if completed_revision == editor_revision {
            match completed {
                Ok(index) => {
                    install_player_collision(runtime, Some(index), editor_revision, None);
                    runtime.collision_failed_revision = None;
                }
                Err(error) => {
                    runtime.collision_failed_revision = Some(editor_revision);
                    warn!("could not compile player construction collision: {error}");
                }
            }
        }
    }
    if runtime.collision_editor_revision != Some(editor_revision)
        && runtime.collision_build.is_none()
        && runtime.collision_failed_revision != Some(editor_revision)
    {
        runtime.pending_player_reactions.clear();
        runtime.capsule.clear_support();
        let graph = graph.clone();
        runtime.collision_build = Some(PlayerCollisionBuild {
            editor_revision,
            task: AsyncComputeTaskPool::get()
                .spawn(async move { compile_player_collision(&graph) }),
        });
    }
}

fn compile_player_collision(
    graph: &ConstructionGraph,
) -> Result<ConstructionCollisionIndex, String> {
    let creation = graph.compile().map_err(|error| error.to_string())?;
    Ok(ConstructionCollisionIndex::new(&creation))
}

fn install_player_collision(
    runtime: &mut WorldRuntime,
    collision: Option<ConstructionCollisionIndex>,
    editor_revision: u64,
    physics_revision: Option<(u64, u64)>,
) {
    runtime.pending_player_reactions.clear();
    runtime.capsule.clear_support();
    runtime.construction_collision = collision;
    runtime.collision_revision = physics_revision;
    runtime.collision_editor_revision = Some(editor_revision);
    runtime.collision_snapshot_tick = 0;
    runtime.collision_pose_revision = 0;
    runtime.collision_poses.clear();
}

fn reset_player_collision_publication(runtime: &mut WorldRuntime) {
    runtime.construction_collision = None;
    runtime.collision_revision = None;
    runtime.collision_editor_revision = None;
    runtime.collision_build = None;
    runtime.collision_failed_revision = None;
    runtime.collision_snapshot_tick = 0;
    runtime.collision_pose_revision = 0;
    runtime.collision_poses.clear();
    runtime.pending_player_reactions.clear();
    runtime.capsule.clear_support();
    runtime.jump_queued = false;
    runtime.step_visual_offset = 0.0;
}

fn angular_velocity(previous: Quat, current: Quat, elapsed: f32) -> Vec3 {
    let delta = (current * previous.inverse()).normalize();
    let (axis, angle) = delta.to_axis_angle();
    if angle.is_finite() && elapsed > 0.0 {
        axis * angle / elapsed
    } else {
        Vec3::ZERO
    }
}

fn select_and_size_brush(
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

#[allow(clippy::too_many_arguments)] // Independent Bevy resources own brush input and output.
fn use_brush(
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

fn terrain_edit_commands(
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

fn execute_terrain_edit_batch(
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

fn commit_terrain_edit_result(
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

fn coordinate_terrain_edits(
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

fn update_terrain_selection(
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

fn schedule_terrain_remeshes(
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
        .and_then(|seat| crate::seat_world_pose(graph, simulation, seat))
        .map_or(player.position, |(position, _)| position);
    WorldPosition(origin.0 + local.as_dvec3())
}

fn startup_region_nodes(
    cut: &[ActiveTerrainNode],
    centre: WorldPosition,
) -> impl Iterator<Item = TerrainNodeId> + '_ {
    let minimum = centre.0 - bevy::math::DVec3::splat(16.0);
    let maximum = centre.0 + bevy::math::DVec3::splat(16.0);
    cut.iter()
        .map(|node| node.id)
        .filter(move |&id| node_overlaps(id, minimum, maximum))
}

fn acknowledge_complete_terrain_pipeline(runtime: &mut WorldRuntime, workers_idle: bool) {
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

#[allow(clippy::too_many_lines)] // Publication is one atomic frame-budgeted state transition.
fn integrate_terrain_remeshes(
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
        if !tasks.is_empty()
            || runtime.terrain_streamer.backlog() != 0
            || runtime.selected_terrain_revision != runtime.terrain_revision
        {
            return;
        }
        let pending = runtime
            .pending_material
            .as_ref()
            .expect("pending material publication");
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
    for (offset, id) in dirty.iter().copied().enumerate() {
        if !material_cutover
            && publishing.elapsed().as_secs_f64() * 1_000.0 >= INTEGRATION_BUDGET_MS
        {
            deferred.extend_from_slice(&dirty[offset..]);
            break;
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
            let ready = runtime.active_terrain_ready_faces[&id];
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

fn terrain_mesh_is_renderable(chunk: &TerrainMeshChunk, index_count: usize) -> bool {
    !chunk.vertices.is_empty() && index_count != 0
}

fn player_collision_nodes<'a>(
    cut: &'a [ActiveTerrainNode],
    capsule: &KinematicCapsule,
) -> impl Iterator<Item = TerrainNodeId> + 'a {
    let (minimum, maximum) = capsule_loading_bounds(capsule);
    cut.iter()
        .map(|node| node.id)
        .filter(move |&id| node_overlaps(id, minimum, maximum))
}

fn terrain_chunk_has_collision_near(
    chunk: &TerrainMeshChunk,
    ready_faces: TerrainTransitionMask,
    capsule: &KinematicCapsule,
) -> bool {
    let (minimum, maximum) = capsule_loading_bounds(capsule);
    bounds_overlap(
        chunk.bounds.minimum.0,
        chunk.bounds.maximum.0,
        minimum,
        maximum,
    ) && chunk
        .index_groups
        .sealed_index_count(chunk.transition_mask, ready_faces)
        != 0
}

fn capsule_loading_bounds(capsule: &KinematicCapsule) -> (bevy::math::DVec3, bevy::math::DVec3) {
    let radius = capsule.config.radius;
    (
        capsule.position.0 - bevy::math::DVec3::new(radius, capsule.config.step_height, radius),
        capsule.position.0 + bevy::math::DVec3::new(radius, capsule.config.standing_height, radius),
    )
}

fn node_overlaps(
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

fn terrain_node_regions_overlap(first: TerrainNodeId, second: TerrainNodeId) -> bool {
    let first_minimum = first.minimum_cell_i64();
    let first_maximum = first.maximum_cell_exclusive_i64();
    let second_minimum = second.minimum_cell_i64();
    let second_maximum = second.maximum_cell_exclusive_i64();
    (0..3).all(|axis| {
        first_minimum[axis] < second_maximum[axis] && second_minimum[axis] < first_maximum[axis]
    })
}

fn ready_obsolete_nodes(
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

fn bounds_overlap(
    first_minimum: bevy::math::DVec3,
    first_maximum: bevy::math::DVec3,
    second_minimum: bevy::math::DVec3,
    second_maximum: bevy::math::DVec3,
) -> bool {
    first_minimum.cmple(second_maximum).all() && second_minimum.cmple(first_maximum).all()
}

#[cfg(test)]
fn nodes_touch_on_face(first: TerrainNodeId, second: TerrainNodeId, face: TerrainFace) -> bool {
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

fn foundation_edit_is_ready(
    acknowledgements: TerrainAcknowledgements,
    pending: &TerrainEditBatch,
    foundation_acknowledgement: u64,
    stroke_idle: bool,
) -> bool {
    !pending.is_empty()
        && foundation_acknowledgement != pending.generation
        && acknowledgements.completed(pending.generation)
        && stroke_idle
}

const FOUNDATION_SYNC_FRAME_BUDGET: Duration = Duration::from_millis(2);
const FOUNDATION_SYNC_MAX_PARTS_PER_FRAME: usize = 32;

#[allow(clippy::too_many_lines)] // Revision staging and bounded support sampling form one cutover.
pub(crate) fn sync_world_foundations(
    graph: Res<EditorGraph>,
    history: Res<EditorHistory>,
    mut runtime: ResMut<WorldRuntime>,
    list: Res<WorldListState>,
    mut editor: ResMut<EditorState>,
    mut diagnostics: ResMut<WorldDiagnostics>,
) {
    if list.phase() != WorldListPhase::Playing {
        return;
    }
    let frame_started = std::time::Instant::now();
    diagnostics.foundation_candidate_count = 0;
    diagnostics.foundation_sample_count = 0;
    diagnostics.foundation_refresh_ms = 0.0;

    let terrain_changed = foundation_edit_is_ready(
        runtime.terrain_acknowledgements,
        &runtime.pending_foundation_edit,
        runtime.foundation_edit_acknowledgement,
        runtime.terrain_edit_task.is_none()
            && runtime.pending_terrain_edits.is_empty()
            && runtime.last_brush_edit.is_none(),
    );
    if terrain_changed {
        // Partially sampled construction belongs to the previous terrain cut.
        // Discard it and restart from the newly published terrain below.
        runtime.pending_foundation_sync = None;
        refresh_foundations_after_terrain_edit(&mut runtime, &mut diagnostics, &mut editor);
    }

    let editor_changed = runtime.synced_editor_revision != history.current_revision;
    let needs_initial_sync = runtime.known_world_parts.is_empty()
        && graph.0.parts().next().is_some()
        && runtime.pending_foundation_sync.is_none();
    let pending_matches = runtime
        .pending_foundation_sync
        .as_ref()
        .is_some_and(|pending| pending.editor_revision == history.current_revision);
    if (editor_changed || needs_initial_sync) && !pending_matches {
        let current_parts = graph
            .0
            .parts()
            .map(|(part, spec)| (part, *spec))
            .collect::<BTreeMap<_, _>>();
        let delta =
            ConstructionEditDelta::between_parts(&runtime.known_world_parts, &current_parts);
        let current_frames = graph
            .0
            .parts()
            .map(|(part, _)| {
                (
                    part,
                    graph.0.part_frame(part).expect("world part has frame"),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let reframed_parts = current_frames
            .iter()
            .filter_map(|(part, frame)| {
                (runtime.known_world_frames.get(part) != Some(frame)).then_some(*part)
            })
            .collect::<BTreeSet<_>>();
        let replaced_parts = delta
            .removed
            .iter()
            .chain(&delta.modified)
            .copied()
            .chain(reframed_parts.iter().copied())
            .collect();
        let new_parts = delta
            .added
            .iter()
            .chain(&delta.modified)
            .copied()
            .chain(reframed_parts.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        runtime.pending_foundation_sync = Some(PendingFoundationSync {
            editor_revision: history.current_revision,
            parts: current_parts,
            frames: current_frames,
            replaced_parts,
            new_parts,
            next_part: 0,
            foundations: Vec::new(),
            index: FoundationSpatialIndex::default(),
        });
        let now = runtime.clock;
        runtime.autosave.mutate(now);
    }

    let Some(mut pending) = runtime.pending_foundation_sync.take() else {
        diagnostics.foundation_refresh_ms = frame_started.elapsed().as_secs_f64() * 1_000.0;
        return;
    };
    diagnostics.foundation_candidate_count =
        u64::try_from(pending.new_parts.len()).unwrap_or(u64::MAX);
    // Ground welds belong to the saved terrain, independently of mesh streaming.
    let scene = TerrainScene {
        field: &runtime.field,
        edits: &runtime.edits,
    };
    let mut processed = 0_usize;
    while pending.next_part < pending.new_parts.len()
        && processed < FOUNDATION_SYNC_MAX_PARTS_PER_FRAME
        && (processed == 0 || frame_started.elapsed() < FOUNDATION_SYNC_FRAME_BUDGET)
    {
        let part = pending.new_parts[pending.next_part];
        pending.next_part += 1;
        processed += 1;
        let Some(&spec) = pending.parts.get(&part) else {
            continue;
        };
        let frame = *pending.frames.get(&part).expect("pending part has frame");
        let support = bounds_foundation_support(
            &scene,
            framed_part_bounds(spec, frame),
            runtime.floating_origin,
        );
        diagnostics.foundation_sample_count = diagnostics
            .foundation_sample_count
            .saturating_add(u64::try_from(support.sample_count()).unwrap_or(u64::MAX));
        if support.has_valid_anchor() {
            pending.index.insert(part, &support);
            pending
                .foundations
                .push(TerrainFoundation { part, support });
        }
    }

    if pending.next_part < pending.new_parts.len() {
        runtime.pending_foundation_sync = Some(pending);
        diagnostics.foundation_refresh_ms = frame_started.elapsed().as_secs_f64() * 1_000.0;
        return;
    }

    let released_provisional_parts = !pending.new_parts.is_empty();
    let removed_foundation = runtime
        .foundations
        .iter()
        .any(|foundation| pending.replaced_parts.contains(&foundation.part));
    for &part in &pending.replaced_parts {
        runtime.foundation_index.remove(part);
    }
    runtime
        .foundations
        .retain(|foundation| !pending.replaced_parts.contains(&foundation.part));
    let added = !pending.foundations.is_empty();
    runtime.foundation_index.append(pending.index);
    runtime.foundations.append(&mut pending.foundations);
    runtime.known_world_parts = pending.parts;
    runtime.known_world_frames = pending.frames;
    runtime.synced_editor_revision = pending.editor_revision;
    if added || removed_foundation || released_provisional_parts {
        runtime.foundation_revision = runtime.foundation_revision.wrapping_add(1);
    }
    diagnostics.foundation_refresh_ms = frame_started.elapsed().as_secs_f64() * 1_000.0;
}

fn refresh_foundations_after_terrain_edit(
    runtime: &mut WorldRuntime,
    diagnostics: &mut WorldDiagnostics,
    editor: &mut EditorState,
) {
    let candidates = runtime
        .foundation_index
        .candidates(&runtime.pending_foundation_edit.changed_bricks);
    let changed_bricks = runtime.pending_foundation_edit.changed_bricks.clone();
    diagnostics.foundation_candidate_count = u64::try_from(candidates.len()).unwrap_or(u64::MAX);
    // Ground welds belong to the saved terrain, independently of mesh streaming.
    let scene = TerrainScene {
        field: &runtime.field,
        edits: &runtime.edits,
    };
    let mut detached = 0_u64;
    let mut anchors_changed = 0_u64;
    runtime.foundations.retain_mut(|foundation| {
        if !candidates.contains(&foundation.part) {
            return true;
        }
        let refresh = foundation.support.refresh_changed(&scene, &changed_bricks);
        diagnostics.foundation_sample_count = diagnostics
            .foundation_sample_count
            .saturating_add(u64::try_from(refresh.sampled).unwrap_or(u64::MAX));
        anchors_changed = anchors_changed
            .saturating_add(u64::try_from(refresh.anchors_changed).unwrap_or(u64::MAX));
        if refresh.detached {
            detached = detached.saturating_add(1);
            runtime.foundation_index.remove(foundation.part);
            false
        } else {
            true
        }
    });
    runtime.foundation_edit_acknowledgement = runtime.pending_foundation_edit.generation;
    runtime.pending_foundation_edit = TerrainEditBatch::default();
    if anchors_changed > 0 {
        runtime.foundation_revision = runtime.foundation_revision.wrapping_add(1);
    }
    if detached > 0 {
        editor.feedback = Some(if detached == 1 {
            "Foundation lost its last terrain anchor — construction released".to_owned()
        } else {
            format!(
                "{detached} foundations lost their last terrain anchors — constructions released"
            )
        });
    }
}

fn autosave_world(
    time: Res<Time>,
    mut runtime: ResMut<WorldRuntime>,
    mut editor: ResMut<EditorState>,
    graph: Res<EditorGraph>,
) {
    runtime.clock += time.delta();
    if runtime.autosave.due(runtime.clock)
        && let Err(error) = save_all(&mut runtime)
            .and_then(|()| save_world_instance(&mut runtime, &graph.0, &editor))
    {
        editor.feedback = Some(error);
    }
}

fn save_on_exit(
    mut exits: MessageReader<AppExit>,
    mut runtime: ResMut<WorldRuntime>,
    mut editor: ResMut<EditorState>,
    graph: Res<EditorGraph>,
) {
    if exits.read().next().is_some()
        && let Err(error) = finish_terrain_edits(&mut runtime)
            .and_then(|()| save_all(&mut runtime))
            .and_then(|()| save_world_instance(&mut runtime, &graph.0, &editor))
    {
        error!("failed to finish world save on exit: {error}");
        editor.feedback = Some(error);
    }
}

fn finish_terrain_edits(runtime: &mut WorldRuntime) -> Result<(), String> {
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

fn save_all(runtime: &mut WorldRuntime) -> Result<(), String> {
    runtime.document.last_played_unix_seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    runtime.document.player_pose = WorldPoseDoc {
        translation: runtime.capsule.position,
        ..runtime.document.player_pose
    };
    runtime
        .store
        .save_world(&runtime.document)
        .map_err(|error| error.to_string())?;
    runtime
        .store
        .save_material_state(
            &runtime.document.name,
            runtime
                .pending_material
                .as_ref()
                .map_or(&runtime.edits, |pending| &pending.previous),
            &runtime.clumps,
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn save_world_instance(
    runtime: &mut WorldRuntime,
    graph: &ConstructionGraph,
    editor: &EditorState,
) -> Result<(), String> {
    let (graph, bearings) = if runtime.document.frozen_creation.is_some() {
        let (graph, bearings) = runtime
            .frozen_editor
            .as_ref()
            .ok_or("Frozen construction is waiting for validated publication")?;
        (graph, bearings.as_slice())
    } else {
        (graph, editor.placed_bearings.as_slice())
    };
    let mut world = space_instance(graph, bearings, "World construction");
    world.root_pose.translation = WorldPosition(runtime.floating_origin.0);
    let garage = runtime.garage_editor.as_ref().map_or_else(
        || space_instance(&ConstructionGraph::new(), &[], "Garage construction"),
        |garage| {
            space_instance(
                &garage.graph,
                &garage.placed_bearings,
                "Garage construction",
            )
        },
    );
    runtime.document.instances = (graph.part_count() != 0)
        .then(|| WorldInstanceIndexDoc {
            id: 1,
            name: "World construction".to_owned(),
        })
        .into_iter()
        .collect();
    runtime
        .store
        .save_space_pair(&mut runtime.document, &world, &garage)
        .map_err(|error| error.to_string())?;
    runtime.autosave.saved();
    Ok(())
}

fn save_garage_instance(
    runtime: &mut WorldRuntime,
    graph: &ConstructionGraph,
    editor: &EditorState,
) -> Result<(), String> {
    let garage = space_instance(graph, &editor.placed_bearings, "Garage construction");
    let world = runtime.world_editor.as_ref().map_or_else(
        || space_instance(&ConstructionGraph::new(), &[], "World construction"),
        |world| {
            let mut instance =
                space_instance(&world.graph, &world.placed_bearings, "World construction");
            instance.root_pose.translation = WorldPosition(world.origin.0);
            instance
        },
    );
    runtime.document.instances = (!world.creation.parts.is_empty())
        .then(|| WorldInstanceIndexDoc {
            id: 1,
            name: "World construction".to_owned(),
        })
        .into_iter()
        .collect();
    runtime
        .store
        .save_space_pair(&mut runtime.document, &world, &garage)
        .map_err(|error| error.to_string())?;
    runtime.autosave.saved();
    Ok(())
}

fn space_instance(
    graph: &ConstructionGraph,
    bearings: &[PlacedBearing],
    name: &str,
) -> WorldCreationInstanceDoc {
    let sockets = bearings
        .iter()
        .map(|bearing| BearingSocket {
            kind: bearing.kind,
            axis: bearing.axis,
            source: bearing.source,
            anchor: bearing.anchor,
            dimensions: bearing.dimensions,
        })
        .collect::<Vec<_>>();
    WorldCreationInstanceDoc {
        id: 1,
        creation: CreationDocument::from_graph(graph, name, &sockets),
        root_pose: WorldPoseDoc::default(),
        joint_coordinates: Vec::new(),
    }
}

fn terrain_chunk_mesh(chunk: &TerrainMeshChunk, indices: Vec<u32>) -> Mesh {
    let colors = chunk
        .material_weights
        .iter()
        .copied()
        .map(|weights| [weights[0], weights[1], weights[2], weights[3]])
        .collect::<Vec<_>>();
    let uvs = chunk
        .vertices
        .iter()
        .map(|position| {
            [
                (chunk.origin.0.x + f64::from(position[0])) as f32 / 1.5,
                (chunk.origin.0.z + f64::from(position[2])) as f32 / 1.5,
            ]
        })
        .collect::<Vec<_>>();
    let vertical_uvs = chunk
        .vertices
        .iter()
        .zip(&chunk.material_weights)
        .map(|(position, weights)| {
            [
                (chunk.origin.0.y + f64::from(position[1])) as f32 / 1.5,
                weights[4],
            ]
        })
        .collect::<Vec<_>>();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, chunk.vertices.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, chunk.normals.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, vertical_uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

#[cfg(test)]
mod render_tests;

#[cfg(test)]
mod terrain_shader_tests;

/// Debug fixture isolation: keep visual captures off the player's saved worlds.
#[cfg(debug_assertions)]
pub(crate) fn prepare_fx_capture(
    runtime: &mut WorldRuntime,
    list: &mut WorldListState,
    directory: &std::path::Path,
) {
    runtime.store = WorldStore::new(directory.join("worlds"));
    list.phase = WorldListPhase::Playing;
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeSet,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use bevy::{
        asset::RenderAssetUsages,
        camera::Exposure,
        math::DVec3,
        mesh::VertexAttributeValues,
        prelude::{App, IVec3, Image, State, Update, Vec3},
        render::render_resource::{Extent3d, TextureDimension, TextureFormat},
        state::app::AppExtStates,
    };
    use mechanic_core::{
        BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, DimensionLinkId,
        DimensionLinkSpec, FaceKind, FaceOwner, FaceRef, GridRotation, WeldSpec,
    };
    use mechanic_world::{
        ActiveTerrainNode, BrickCoord, FloatingOrigin, FoundationSample, FoundationSpatialIndex,
        FoundationSupport, KinematicCapsule, TerrainDensity, TerrainEditBatch, TerrainFace,
        TerrainField, TerrainMaterial, TerrainMeshChunk, TerrainMeshRequest, TerrainNodeId,
        TerrainOctree, TerrainRayHit, TerrainReadiness, TerrainTransitionMask, WorldBounds,
        WorldPosition, WorldSeed, WorldStore, mesh_chunk, select_active_nodes,
    };

    use super::{
        AppSpace, SpaceEditorState, TerrainAcknowledgements, TerrainEditOperation,
        TerrainStrokeSample, WorldDiagnostics, WorldListPhase, WorldListState,
        WorldPrototypePlugin, WorldRuntime, advance_controller, compile_player_collision,
        exposure_for_space, foundation_edit_is_ready, full_rgba8_mip_byte_count,
        generate_rgba8_mip_chain, graph_bounds, handle_world_list, install_world,
        load_space_editors, nodes_touch_on_face, place_in_world, player_collision_nodes,
        ready_obsolete_nodes, remove_cached_foundations, returned_component_parts,
        smooth_step_visual_offset, static_parts_for_physics, sync_world_foundations,
        terrain_chunk_has_collision_near, terrain_chunk_mesh, terrain_edit_commands,
        terrain_mesh_is_renderable,
    };
    use super::{PendingFoundationSync, TerrainFoundation};
    use crate::{EditorGraph, EditorHistory, EditorState, garage, showcase};

    #[test]
    fn saved_floor_creation_is_centered_in_editable_garage_and_detached_from_ground() {
        let mut graph = mechanic_core::ConstructionGraph::new();
        let mechanic_core::BuildOutcome::Spawned(part) = graph
            .apply(mechanic_core::BuildCommand::Spawn(
                mechanic_core::CuboidSpec::new(
                    [4, 1, 4],
                    mechanic_core::BuildPose::from_half_grid(
                        IVec3::new(32, 1, -24),
                        mechanic_core::GridRotation::default(),
                    ),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(mechanic_core::BuildCommand::Weld(mechanic_core::WeldSpec {
                first: mechanic_core::FaceRef::ground(),
                second: mechanic_core::FaceRef::part(part, mechanic_core::FaceKind::NegativeY),
            }))
            .unwrap();
        let loaded = mechanic_core::CreationDocument::from_graph(&graph, "Floor example", &[])
            .into_graph()
            .unwrap();
        let placed = super::place_loaded_creation_in_garage(loaded).unwrap();
        let (low, high) = super::graph_bounds(&placed.graph).unwrap();
        assert!((low.y - garage::BUILD_MIN_Y).abs() < 1.0e-5);
        assert!((low.x + high.x).abs() < 1.0e-5);
        assert!((low.z + high.z).abs() < 1.0e-5);
        assert_eq!(placed.name, "Floor example");
        assert_eq!(placed.graph.weld_count(), 0);
        placed.graph.compile().unwrap();
    }

    struct TempWorldStore(std::path::PathBuf);

    static NEXT_TEMP_WORLD_STORE: AtomicUsize = AtomicUsize::new(0);

    impl TempWorldStore {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "mechanic-world-install-test-{}-{}",
                std::process::id(),
                NEXT_TEMP_WORLD_STORE.fetch_add(1, Ordering::Relaxed),
            ));
            let _ = std::fs::remove_dir_all(&path);
            Self(path)
        }
    }

    impl Drop for TempWorldStore {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    struct FlatTerrain(f64);

    impl TerrainDensity for FlatTerrain {
        fn density(&self, position: WorldPosition) -> f32 {
            (self.0 - position.0.y) as f32
        }

        fn material(&self, _position: WorldPosition) -> TerrainMaterial {
            TerrainMaterial::Soil
        }
    }

    struct SlopedTerrain;

    impl TerrainDensity for SlopedTerrain {
        fn density(&self, position: WorldPosition) -> f32 {
            (position.0.x * 0.25 - position.0.y) as f32
        }

        fn material(&self, _position: WorldPosition) -> TerrainMaterial {
            TerrainMaterial::Soil
        }
    }

    #[test]
    fn player_collision_compiles_before_foundation_classification() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 8, 8],
                    BuildPose::new(IVec3::new(4, 4, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();

        let mut collision = compile_player_collision(&graph).unwrap();
        assert!(
            collision
                .cast_capsule(
                    bevy::prelude::Vec3::ZERO,
                    bevy::prelude::Vec3::X * 2.0,
                    mechanic_world::KinematicCapsuleConfig::default(),
                )
                .is_some()
        );
    }

    #[test]
    fn automatic_step_rise_is_smoothed_without_changing_the_collision_height() {
        let mut offset = smooth_step_visual_offset(0.0, 0.25, 1.0 / 60.0);
        assert!(offset > -0.25 && offset < 0.0, "{offset}");
        let first = offset;
        for _ in 0..60 {
            offset = smooth_step_visual_offset(offset, 0.0, 1.0 / 60.0);
        }
        assert!(offset.abs() < 1.0e-4, "{offset}");
        assert!(first.abs() < 0.25);
    }

    #[test]
    fn jump_request_is_buffered_until_the_next_fixed_controller_tick() {
        let mut accumulator = 0.0;
        let mut jump_queued = false;

        assert_eq!(
            advance_controller(
                &mut accumulator,
                &mut jump_queued,
                mechanic_core::TICK_SECONDS * 0.5,
                true,
            ),
            (0, false)
        );
        assert_eq!(
            advance_controller(
                &mut accumulator,
                &mut jump_queued,
                mechanic_core::TICK_SECONDS * 0.5,
                false,
            ),
            (1, true)
        );
        assert_eq!(
            advance_controller(
                &mut accumulator,
                &mut jump_queued,
                mechanic_core::TICK_SECONDS,
                false,
            ),
            (1, false)
        );
    }

    #[test]
    fn prototype_starts_in_garage_space() {
        let mut app = App::new();
        app.add_plugins(bevy::state::app::StatesPlugin);
        app.add_plugins(WorldPrototypePlugin);
        assert_eq!(
            *app.world().resource::<State<AppSpace>>().get(),
            AppSpace::Garage
        );
    }

    #[test]
    fn transfer_collision_tests_composed_boxes_instead_of_local_overlap() {
        let spawn = |graph: &mut ConstructionGraph| {
            let BuildOutcome::Spawned(part) = graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new([4; 3], BuildPose::new(IVec3::ZERO, GridRotation::default()))
                        .unwrap(),
                ))
                .unwrap()
            else {
                unreachable!()
            };
            part
        };
        let mut candidate = ConstructionGraph::new();
        let part = spawn(&mut candidate);
        let mut destination = ConstructionGraph::new();
        let other = spawn(&mut destination);
        let far = mechanic_core::ConstructionFrame::new(
            Vec3::new(5.0, 0.0, 0.0),
            bevy::prelude::Quat::from_rotation_y(0.4),
        )
        .unwrap();
        candidate.reframe_parts([part], far).unwrap();
        let mut index = crate::builder::PlacementSnapIndex::default();
        index.rebuild(&destination);
        assert!(super::collision_free(&candidate, &destination, &index));
        destination.reframe_parts([other], far).unwrap();
        index.rebuild(&destination);
        assert!(!super::collision_free(&candidate, &destination, &index));
    }

    #[test]
    fn returned_framed_creation_accepts_blocks_in_its_local_grid() {
        use crate::builder::{self, PlacementBounds, PlacementGrid};
        use mechanic_core::ConstructionFrame;

        for rotation in [
            bevy::prelude::Quat::IDENTITY,
            bevy::prelude::Quat::from_rotation_z(0.4) * bevy::prelude::Quat::from_rotation_y(0.3),
        ] {
            let mut graph = ConstructionGraph::new();
            let BuildOutcome::Spawned(part) = graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [8, 2, 4],
                        BuildPose::new(IVec3::new(0, 80, 0), GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap()
            else {
                unreachable!()
            };
            graph
                .reframe_parts(
                    [part],
                    ConstructionFrame::new(Vec3::new(30.03, 4.07, -20.02), rotation).unwrap(),
                )
                .unwrap();
            // Repeat the user's Garage -> World -> Garage recovery path.
            for _ in 0..2 {
                let garage =
                    super::place_in_garage(&graph, &[], &SpaceEditorState::default()).unwrap();
                let part = garage.graph.parts().next().unwrap().0;
                let context = crate::live_edit::EditContext::resolve(
                    &garage.graph,
                    &crate::AppSimulation::default(),
                    part,
                )
                .unwrap();
                let local = garage.graph.in_edit_frame(context.frame).unwrap();
                let spec = local.part(part).unwrap().as_cuboid().unwrap();
                let (low, high) = builder::part_world_bounds(mechanic_core::PartSpec::Cuboid(spec));
                let hit = builder::raycast_construction_with_ground(
                    &local,
                    Vec3::new((low.x + high.x) * 0.5, high.y + 1.0, (low.z + high.z) * 0.5),
                    Vec3::NEG_Y,
                    None,
                )
                .unwrap();
                let bounds = PlacementBounds::GarageBuild.in_edit_frame(context.frame_to_world);
                let candidate = builder::candidate_from_hit_with_grid(
                    &local,
                    hit,
                    PlacementGrid::Centimetres25,
                    bounds,
                );
                let mut index = builder::PlacementSnapIndex::default();
                index.rebuild(&local);
                assert_eq!(
                    builder::validate_indexed_block_batch_in_bounds(
                        &index,
                        candidate,
                        &[candidate.spec],
                        PlacementBounds::GarageBuild,
                    ),
                    Err(builder::PlacementError::OutsidePlatform),
                );
                builder::validate_indexed_block_batch_in_bounds(
                    &index,
                    candidate,
                    &[candidate.spec],
                    bounds,
                )
                .unwrap();
                let edited = builder::stage_block_batch_in_bounds(
                    &local,
                    candidate,
                    &[candidate.spec],
                    bounds,
                )
                .unwrap()
                .canonicalized();
                assert_eq!(edited.parts().count(), garage.graph.parts().count() + 1);
                edited.compile().unwrap();
                graph = place_in_world(
                    &garage.graph,
                    &[],
                    &SpaceEditorState::default(),
                    Vec3::ZERO,
                    &FlatTerrain(0.0),
                    FloatingOrigin::default(),
                )
                .unwrap()
                .graph;
            }
        }
    }

    #[test]
    fn framed_creation_transfers_preserve_orientation_and_composed_size() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [8, 2, 4],
                    BuildPose::new(IVec3::new(0, 4, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let frame = mechanic_core::ConstructionFrame::new(
            Vec3::new(30.0, 4.0, -20.0),
            bevy::prelude::Quat::from_rotation_z(0.4) * bevy::prelude::Quat::from_rotation_y(0.3),
        )
        .unwrap();
        graph.reframe_parts([part], frame).unwrap();
        let (low, high) = graph_bounds(&graph).unwrap();
        let size = high - low;
        let original_rotation = graph.part_rotation(part).unwrap();
        let garage = super::place_in_garage(&graph, &[], &SpaceEditorState::default()).unwrap();
        let (garage_low, garage_high) = graph_bounds(&garage.graph).unwrap();
        assert!((garage_high - garage_low).distance(size) < 1.0e-4);
        assert!(garage_low.y >= crate::garage::BUILD_MIN_Y - 1.0e-4);
        let garage_part = garage.graph.parts().next().unwrap().0;
        assert!(
            garage
                .graph
                .part_rotation(garage_part)
                .unwrap()
                .angle_between(original_rotation)
                < 1.0e-3
        );
        let world = place_in_world(
            &garage.graph,
            &[],
            &SpaceEditorState::default(),
            Vec3::ZERO,
            &FlatTerrain(0.0),
            FloatingOrigin::default(),
        )
        .unwrap();
        let (world_low, world_high) = graph_bounds(&world.graph).unwrap();
        assert!((world_high - world_low).distance(size) < 1.0e-4);
        assert!(world_low.y >= 0.125 - 1.0e-4);
        let world_part = world.graph.parts().next().unwrap().0;
        assert!(
            world
                .graph
                .part_rotation(world_part)
                .unwrap()
                .angle_between(original_rotation)
                < 1.0e-3
        );
    }

    #[test]
    fn framed_foundation_sampling_uses_global_composed_bottom_bounds() {
        let spec = mechanic_core::PartSpec::Cuboid(
            CuboidSpec::new(
                [4, 2, 2],
                BuildPose::new(IVec3::ZERO, GridRotation::default()),
            )
            .unwrap(),
        );
        let frame = mechanic_core::ConstructionFrame::new(
            Vec3::new(5.0, 0.5, -3.0),
            bevy::prelude::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        )
        .unwrap();
        let bounds = super::framed_part_bounds(spec, frame);
        let origin = FloatingOrigin(DVec3::new(100.0, 10.0, 200.0));
        let support = super::bounds_foundation_support(&FlatTerrain(10.0), bounds, origin);
        assert!(support.has_valid_anchor());
        assert!(support.samples.iter().all(|sample| {
            sample.position.0.x > 104.0
                && sample.position.0.x < 106.0
                && sample.position.0.z > 196.0
                && sample.position.0.z < 198.0
        }));
        assert!(
            super::bounds_foundation_support(
                &FlatTerrain(10.0),
                crate::builder::part_world_bounds(spec),
                origin
            )
            .samples
            .iter()
            .all(|sample| sample.position.0.x < 101.0)
        );
    }

    #[test]
    fn frame_only_edit_invalidates_foundation_cache_with_unchanged_part_spec() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [2; 3],
                    BuildPose::new(IVec3::new(0, 1, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let original_spec = *graph.part(part).unwrap();
        let mut app = App::new();
        app.init_resource::<WorldRuntime>();
        app.init_resource::<WorldListState>();
        app.init_resource::<EditorState>();
        app.init_resource::<WorldDiagnostics>();
        {
            let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
            runtime.known_world_parts.insert(part, original_spec);
            runtime
                .known_world_frames
                .insert(part, mechanic_core::ConstructionFrame::IDENTITY);
            runtime.synced_editor_revision = 1;
        }
        let frame = mechanic_core::ConstructionFrame::new(
            Vec3::new(5.0, 0.0, 0.0),
            bevy::prelude::Quat::from_rotation_y(0.4),
        )
        .unwrap();
        graph.reframe_parts([part], frame).unwrap();
        assert_eq!(*graph.part(part).unwrap(), original_spec);
        app.insert_resource(EditorGraph(graph));
        app.insert_resource(EditorHistory {
            current_revision: 2,
            next_revision: 2,
            ..Default::default()
        });
        app.world_mut().resource_mut::<WorldListState>().phase = WorldListPhase::Playing;
        app.add_systems(Update, sync_world_foundations);
        app.update();
        let runtime = app.world().resource::<WorldRuntime>();
        assert_eq!(runtime.known_world_frames.get(&part), Some(&frame));
        assert!(runtime.foundations_match_editor_revision(2));
        assert_eq!(
            app.world()
                .resource::<WorldDiagnostics>()
                .foundation_candidate_count,
            1
        );
        assert!(
            app.world()
                .resource::<WorldDiagnostics>()
                .foundation_sample_count
                > 0
        );
    }

    #[test]
    fn garage_creation_enters_world_detached_and_clear_of_terrain() {
        const TERRAIN_HEIGHT: f64 = 0.06;

        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [2; 3],
                    BuildPose::new(IVec3::new(0, 1, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(part, FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();

        let placed = place_in_world(
            &graph,
            &[],
            &SpaceEditorState::default(),
            Vec3::ZERO,
            &FlatTerrain(TERRAIN_HEIGHT),
            FloatingOrigin::default(),
        )
        .unwrap();

        assert!(placed.graph.welds().all(|(_, weld)| {
            !matches!(weld.first.owner, FaceOwner::Ground)
                && !matches!(weld.second.owner, FaceOwner::Ground)
        }));
        let (minimum, maximum) = graph_bounds(&placed.graph).unwrap();
        assert!(f64::from(minimum.y) - TERRAIN_HEIGHT >= 0.125 - 1.0e-6);
        let closest_x = 0.0_f32.clamp(minimum.x, maximum.x);
        let closest_z = 0.0_f32.clamp(minimum.z, maximum.z);
        assert!(closest_x.mul_add(closest_x, closest_z * closest_z) > 4.0);
        let support = FoundationSupport::rectangular(
            &FlatTerrain(TERRAIN_HEIGHT),
            TerrainRayHit {
                position: WorldPosition(
                    Vec3::new(
                        (minimum.x + maximum.x) * 0.5,
                        minimum.y,
                        (minimum.z + maximum.z) * 0.5,
                    )
                    .as_dvec3(),
                ),
                normal: Vec3::Y,
                distance: 0.0,
                material_weights: [0.0; TerrainMaterial::COUNT],
                chunk_generation: 0,
                triangle: 0,
            },
            f64::from(maximum.x - minimum.x),
            f64::from(maximum.z - minimum.z),
        );
        assert!(!support.has_valid_anchor());
    }

    #[test]
    fn garage_creation_enters_sloped_world_without_becoming_a_foundation() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([8, 2, 2], BuildPose::new(IVec3::Y, GridRotation::default()))
                    .unwrap(),
            ))
            .unwrap();

        let placed = place_in_world(
            &graph,
            &[],
            &SpaceEditorState::default(),
            Vec3::new(0.0, -10.0, 0.0),
            &SlopedTerrain,
            FloatingOrigin::default(),
        )
        .unwrap();

        for (_, part) in placed.graph.parts() {
            let (minimum, maximum) = crate::builder::part_world_bounds(*part);
            let support = FoundationSupport::rectangular(
                &SlopedTerrain,
                TerrainRayHit {
                    position: WorldPosition(
                        Vec3::new(
                            (minimum.x + maximum.x) * 0.5,
                            minimum.y,
                            (minimum.z + maximum.z) * 0.5,
                        )
                        .as_dvec3(),
                    ),
                    normal: Vec3::Y,
                    distance: 0.0,
                    material_weights: [0.0; TerrainMaterial::COUNT],
                    chunk_generation: 0,
                    triangle: 0,
                },
                f64::from(maximum.x - minimum.x),
                f64::from(maximum.z - minimum.z),
            );
            assert!(!support.has_valid_anchor());
        }
    }

    #[test]
    fn returning_component_discards_foundations_cached_under_reused_part_ids() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(chassis) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([2, 1, 1], BuildPose::new(IVec3::Y, GridRotation::default()))
                    .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(link) = graph
            .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
                DimensionLinkId(4),
                BuildPose::new(IVec3::new(2, 1, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(chassis, FaceKind::PositiveX),
                second: FaceRef::part(link, FaceKind::NegativeX),
            }))
            .unwrap();
        let BuildOutcome::Spawned(unrelated) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1; 3],
                    BuildPose::new(IVec3::new(20, 0, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let support = || FoundationSupport {
            samples: vec![FoundationSample {
                position: WorldPosition::default(),
                valid: true,
            }],
        };
        let mut foundations = vec![
            TerrainFoundation {
                part: chassis,
                support: support(),
            },
            TerrainFoundation {
                part: unrelated,
                support: support(),
            },
        ];
        let mut index = FoundationSpatialIndex::default();
        for foundation in &foundations {
            index.insert(foundation.part, &foundation.support);
        }

        let returned = returned_component_parts(&graph, DimensionLinkId(4)).unwrap();
        assert!(remove_cached_foundations(
            &mut foundations,
            &mut index,
            &returned,
        ));

        assert_eq!(foundations.len(), 1);
        assert_eq!(foundations[0].part, unrelated);
        assert_eq!(index.len(), 1);
        assert!(!returned.contains(&unrelated));
        let compiled = graph
            .compile_with_static_parts(foundations.iter().map(|foundation| foundation.part))
            .unwrap();
        let body_for = |part| {
            compiled
                .part_to_compound
                .iter()
                .find_map(|(candidate, body)| (*candidate == part).then_some(*body as usize))
                .unwrap()
        };
        assert!(!compiled.compounds[body_for(chassis)].is_static);
        assert!(compiled.compounds[body_for(unrelated)].is_static);
    }

    #[test]
    fn dimension_link_keeps_ground_anchors_after_foundation_sync() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(chassis) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([2, 1, 1], BuildPose::new(IVec3::Y, GridRotation::default()))
                    .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(link) = graph
            .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
                DimensionLinkId(7),
                BuildPose::new(IVec3::new(2, 1, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(chassis, FaceKind::PositiveX),
                second: FaceRef::part(link, FaceKind::NegativeX),
            }))
            .unwrap();
        let BuildOutcome::Spawned(unrelated) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1; 3],
                    BuildPose::new(IVec3::new(20, 0, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let valid_support = || FoundationSupport {
            samples: vec![FoundationSample {
                position: WorldPosition::default(),
                valid: true,
            }],
        };
        let foundations = vec![
            TerrainFoundation {
                part: chassis,
                support: valid_support(),
            },
            TerrainFoundation {
                part: unrelated,
                support: valid_support(),
            },
        ];
        let pending = PendingFoundationSync {
            editor_revision: 8,
            parts: graph.parts().map(|(part, spec)| (part, *spec)).collect(),
            frames: graph
                .parts()
                .map(|(part, _)| (part, graph.part_frame(part).unwrap()))
                .collect(),
            replaced_parts: BTreeSet::new(),
            new_parts: graph.parts().map(|(part, _)| part).collect(),
            next_part: 32,
            foundations: Vec::new(),
            index: FoundationSpatialIndex::default(),
        };

        let static_parts = static_parts_for_physics(
            &pending.parts,
            &foundations,
            None,
            pending.editor_revision,
            pending.editor_revision,
        )
        .expect("completed support sampling permits publication");
        assert_eq!(static_parts, vec![chassis, unrelated]);

        let compiled = graph.compile_with_static_parts(static_parts).unwrap();
        let body_for = |part| {
            compiled
                .part_to_compound
                .iter()
                .find_map(|(candidate, body)| (*candidate == part).then_some(*body as usize))
                .unwrap()
        };
        assert!(compiled.compounds[body_for(chassis)].is_static);
        assert!(compiled.compounds[body_for(link)].is_static);
        assert!(compiled.compounds[body_for(unrelated)].is_static);
    }

    #[test]
    fn linked_creation_does_not_mobilize_unrelated_world_construction() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(chassis) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([2, 1, 1], BuildPose::new(IVec3::Y, GridRotation::default()))
                    .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(link) = graph
            .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
                DimensionLinkId(12),
                BuildPose::new(IVec3::new(2, 1, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(chassis, FaceKind::PositiveX),
                second: FaceRef::part(link, FaceKind::NegativeX),
            }))
            .unwrap();
        let BuildOutcome::Spawned(unrelated) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1; 3],
                    BuildPose::new(IVec3::new(20, 0, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let known_parts = graph.parts().map(|(part, spec)| (part, *spec)).collect();

        let foundations = [TerrainFoundation {
            part: unrelated,
            support: FoundationSupport {
                samples: vec![FoundationSample {
                    position: WorldPosition::default(),
                    valid: true,
                }],
            },
        }];
        let static_parts =
            static_parts_for_physics(&known_parts, &foundations, None, 3, 3).unwrap();
        assert_eq!(static_parts, vec![unrelated]);

        let compiled = graph.compile_with_static_parts(static_parts).unwrap();
        let body_for = |part| {
            compiled
                .part_to_compound
                .iter()
                .find_map(|(candidate, body)| (*candidate == part).then_some(*body as usize))
                .unwrap()
        };
        assert!(!compiled.compounds[body_for(chassis)].is_static);
        assert!(!compiled.compounds[body_for(link)].is_static);
        assert!(compiled.compounds[body_for(unrelated)].is_static);
    }

    #[test]
    #[ignore = "requires a real GPU adapter"]
    #[allow(clippy::too_many_lines)] // Exercise the app owner and GPU publication together.
    fn real_gpu_terrain_publication_preserves_motion_and_rejects_failed_replacements() {
        use mechanic_world::{TerrainTriangleGroupMask, TriangleBvh, TriangleBvhTriangle};
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("terrain publication requires a real adapter");
        eprintln!("Terrain publication adapter: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let mut app = App::new();
        app.init_resource::<WorldRuntime>();
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        runtime.floating_origin = FloatingOrigin(DVec3::splat(1000.0));
        let mut weights = [0.0; TerrainMaterial::COUNT];
        weights[usize::from(TerrainMaterial::Rock.code())] = 1.0;
        let mut chunk = TerrainMeshChunk {
            origin: WorldPosition(runtime.floating_origin.0),
            generation: 1,
            vertices: vec![[-5.0, 0.0, -5.0], [0.0, 0.0, 5.0], [5.0, 0.0, -5.0]],
            material_weights: vec![weights; 3],
            triangle_bvh: TriangleBvh {
                triangles: vec![TriangleBvhTriangle {
                    indices: [0, 1, 2],
                    group_mask: TerrainTriangleGroupMask::REGULAR,
                }],
                ..Default::default()
            },
            ..Default::default()
        };
        chunk.index_groups.regular = vec![0, 1, 2];
        let node = chunk.node;
        runtime.active_terrain_index.insert(node);
        runtime.active_terrain.insert(node, chunk);
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4; 3],
                    BuildPose::from_half_grid(IVec3::new(0, 5, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
        let mut creation = graph.compile().unwrap();
        for collider in &mut creation.colliders {
            collider.material_properties.restitution = 0.0;
            collider.material_properties.youngs_modulus_pa = 200.0e9;
        }
        let gpu = crate::GpuPhysics::new_with_config(
            &device,
            &queue,
            &creation,
            crate::GpuPhysicsConfig {
                ground_plane_enabled: false,
                ..Default::default()
            },
        )
        .unwrap();
        gpu.enable_async_readback();
        gpu.apply_impulse(
            &device,
            &queue,
            0,
            creation.compounds[0].root_translation,
            Vec3::NEG_Y * 20.0 * creation.compounds[0].mass_properties.mass,
        )
        .unwrap();
        let mut simulation = crate::AppSimulation {
            gpu: Some(gpu),
            ..Default::default()
        };
        let publish = |simulation: &mut crate::AppSimulation, runtime: &WorldRuntime| {
            crate::terrain_publication::publish(simulation, runtime, &device, &queue)
        };
        assert!(publish(&mut simulation, &runtime).unwrap());
        assert!(!publish(&mut simulation, &runtime).unwrap());
        let gpu = simulation.gpu.as_ref().unwrap();
        gpu.dispatch_tick(&device, &queue, 1);
        // Replace the scene before waiting for the submitted tick. Its captured
        // buffers and readback must still represent the previously accepted cut.

        let chunk = runtime.active_terrain.get_mut(&node).unwrap();
        chunk.generation = 2;
        chunk.triangle_bvh.triangles[0].indices[0] = u32::MAX;
        assert!(publish(&mut simulation, &runtime).is_err());
        let chunk = runtime.active_terrain.get_mut(&node).unwrap();
        chunk.generation = 1;
        chunk.triangle_bvh.triangles[0].indices[0] = 0;
        assert!(!publish(&mut simulation, &runtime).unwrap());

        // Retirement removes contacts without rebuilding or resetting bodies.
        runtime.active_terrain_index.remove(node);
        assert!(publish(&mut simulation, &runtime).unwrap());
        let gpu = simulation.gpu.as_ref().unwrap();
        gpu.dispatch_tick(&device, &queue, 2);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let first = gpu.poll_tick_readback(&device).unwrap().unwrap();
        assert_eq!(first.diagnostics.error_flags, 0);
        assert_eq!(first.diagnostics.contact_count, 4);
        assert!((first.transforms[0].position[1] - 0.5).abs() <= 0.005);
        assert!(first.velocities[0].linear[1] <= 0.001);
        let retired = gpu.poll_tick_readback(&device).unwrap().unwrap();
        assert_eq!(retired.diagnostics.error_flags, 0);
        assert_eq!(retired.diagnostics.contact_count, 0);
        assert!(retired.transforms[0].position[1] < first.transforms[0].position[1]);
        assert!(retired.velocities[0].linear[1] < first.velocities[0].linear[1]);

        // The same chunk generation must be re-uploaded in a new local frame.
        runtime.active_terrain_index.insert(node);
        runtime.floating_origin.0 += DVec3::X * 20.0;
        assert!(publish(&mut simulation, &runtime).unwrap());
        let gpu = simulation.gpu.as_ref().unwrap();
        gpu.dispatch_tick(&device, &queue, 3);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let shifted = gpu.poll_tick_readback(&device).unwrap().unwrap();
        assert_eq!(shifted.diagnostics.error_flags, 0);
        assert_eq!(shifted.diagnostics.contact_count, 0);
        runtime.floating_origin.0 -= DVec3::X * 20.0;
        assert!(publish(&mut simulation, &runtime).unwrap());
        let gpu = simulation.gpu.as_ref().unwrap();
        gpu.dispatch_tick(&device, &queue, 4);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let restored = gpu.poll_tick_readback(&device).unwrap().unwrap();
        assert_eq!(restored.diagnostics.error_flags, 0);
        assert_eq!(restored.diagnostics.contact_count, 4);
    }

    #[test]
    fn physics_terrain_excludes_hidden_replacements_and_tracks_retirement() {
        let mut app = App::new();
        app.init_resource::<WorldRuntime>();
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        let parent = TerrainNodeId::ROOT;
        let child = parent.children().unwrap()[0];
        for (node, generation) in [(parent, 1), (child, 2)] {
            runtime.active_terrain.insert(
                node,
                TerrainMeshChunk {
                    node,
                    generation,
                    ..Default::default()
                },
            );
        }
        let everywhere = [WorldBounds {
            minimum: WorldPosition(DVec3::splat(-1.0e9)),
            maximum: WorldPosition(DVec3::splat(1.0e9)),
        }];
        runtime.active_terrain_index.insert(parent);
        assert_eq!(
            runtime
                .physics_terrain_near(&everywhere)
                .map(|chunk| chunk.node)
                .collect::<Vec<_>>(),
            vec![parent]
        );
        runtime.active_terrain_index.remove(parent);
        runtime.active_terrain_index.insert(child);
        assert_eq!(
            runtime
                .physics_terrain_near(&everywhere)
                .map(|chunk| (chunk.node, chunk.generation))
                .collect::<Vec<_>>(),
            vec![(child, 2)]
        );
        runtime.active_terrain_index.remove(child);
        assert_eq!(runtime.physics_terrain_near(&everywhere).count(), 0);

        // A region that no chunk reaches publishes nothing at all.
        runtime.active_terrain_index.insert(child);
        let elsewhere = [WorldBounds {
            minimum: WorldPosition(DVec3::splat(1.0e8)),
            maximum: WorldPosition(DVec3::splat(1.0e8 + 1.0)),
        }];
        assert_eq!(runtime.physics_terrain_near(&elsewhere).count(), 0);
    }

    #[test]
    fn physics_waits_until_local_terrain_is_current() {
        let mut app = App::new();
        app.init_resource::<WorldRuntime>();
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        let id = TerrainNodeId::default();
        let node = ActiveTerrainNode {
            id,
            generation: 1,
            transition_mask: TerrainTransitionMask::NONE,
        };
        assert!(
            !runtime.physics_terrain_ready(),
            "no selected region is unsafe"
        );
        runtime.terrain_streamer.set_critical_nodes([id]);
        runtime.terrain_streamer.set_desired([node]);
        assert!(
            !runtime.physics_terrain_ready(),
            "pending terrain is unsafe"
        );
        runtime.terrain_streamer.mark_started(node);
        assert!(runtime.terrain_streamer.stage(node));
        assert_eq!(runtime.terrain_streamer.activate(id), vec![node]);
        assert!(
            runtime.physics_terrain_ready(),
            "active current terrain is safe"
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)] // Exercise edits through the foundation publication system.
    fn bearing_and_upper_block_edits_preserve_ground_weld_until_last_foot_is_removed() {
        let mut graph = ConstructionGraph::new();
        let mut parts = Vec::new();
        for (size, position) in [
            ([1; 3], IVec3::ZERO),
            ([1; 3], IVec3::X),
            ([2, 1, 1], IVec3::Y),
            ([1; 3], IVec3::Y * 2),
        ] {
            let BuildOutcome::Spawned(part) = graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(size, BuildPose::new(position, GridRotation::default()))
                        .unwrap(),
                ))
                .unwrap()
            else {
                unreachable!()
            };
            parts.push(part);
        }
        let [first_foot, second_foot, platform, upper] = parts.try_into().unwrap();
        for (below, above) in [
            (first_foot, platform),
            (second_foot, platform),
            (platform, upper),
        ] {
            graph
                .apply(BuildCommand::Weld(WeldSpec {
                    first: FaceRef::part(below, FaceKind::PositiveY),
                    second: FaceRef::part(above, FaceKind::NegativeY),
                }))
                .unwrap();
        }
        let mut app = App::new();
        app.init_resource::<WorldRuntime>()
            .init_resource::<WorldListState>()
            .init_resource::<EditorState>()
            .init_resource::<WorldDiagnostics>()
            .init_resource::<EditorHistory>()
            .insert_resource(EditorGraph(graph.clone()))
            .add_systems(Update, sync_world_foundations);
        app.world_mut().resource_mut::<WorldListState>().phase = WorldListPhase::Playing;
        {
            let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
            runtime.known_world_parts = graph.parts().map(|(part, spec)| (part, *spec)).collect();
            runtime.known_world_frames = graph
                .parts()
                .map(|(part, _)| (part, graph.part_frame(part).unwrap()))
                .collect();
            for part in [first_foot, second_foot] {
                let support = FoundationSupport {
                    samples: vec![FoundationSample {
                        position: WorldPosition::default(),
                        valid: true,
                    }],
                };
                runtime.foundation_index.insert(part, &support);
                runtime
                    .foundations
                    .push(TerrainFoundation { part, support });
            }
        }
        // Placing a bearing socket changes editor history without changing part geometry.
        app.world_mut()
            .resource_mut::<EditorState>()
            .placed_bearings
            .push(crate::PlacedBearing {
                kind: mechanic_core::BearingKind::Rotational,
                axis: Vec3::ZERO,
                source: FaceRef::part(platform, FaceKind::PositiveY),
                anchor: Vec3::new(0.25, 0.5, 0.125),
                dimensions: mechanic_core::BearingDimensions::default(),
            });
        for (revision, removed, expected_anchors) in [
            (1, None, vec![first_foot, second_foot]),
            (2, Some(upper), vec![first_foot, second_foot]),
            (3, Some(first_foot), vec![second_foot]),
            (4, Some(second_foot), vec![]),
        ] {
            if let Some(part) = removed {
                app.world_mut()
                    .resource_mut::<EditorGraph>()
                    .0
                    .apply(BuildCommand::Remove(part))
                    .unwrap();
            }
            app.world_mut()
                .resource_mut::<EditorHistory>()
                .current_revision = revision;
            app.update();
            let runtime = app.world().resource::<WorldRuntime>();
            let anchors = runtime.static_parts_for_physics(revision).unwrap();
            assert_eq!(anchors, expected_anchors);
            let graph = &app.world().resource::<EditorGraph>().0;
            let compiled = graph.compile_with_static_parts(anchors).unwrap();
            let body = compiled
                .compounds
                .iter()
                .find(|body| body.source_parts.contains(&platform))
                .unwrap();
            assert_eq!(body.is_static, !expected_anchors.is_empty());
        }
    }

    #[test]
    fn ground_weld_sampling_does_not_depend_on_streamed_meshes() {
        let mut app = App::new();
        app.init_resource::<WorldRuntime>()
            .init_resource::<WorldListState>()
            .init_resource::<EditorState>()
            .init_resource::<WorldDiagnostics>()
            .init_resource::<EditorHistory>()
            .add_systems(Update, sync_world_foundations);
        {
            let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
            let spawn = runtime.field.safe_spawn();
            let scene = super::TerrainScene {
                field: &runtime.field,
                edits: &runtime.edits,
            };
            let hit = mechanic_world::raycast_density(
                &scene,
                WorldPosition(spawn.0 + DVec3::Y * 64.0),
                -DVec3::Y,
                128.0,
            )
            .unwrap();
            runtime.floating_origin = FloatingOrigin(hit.position.0 + DVec3::Y * 0.125);
            assert!(runtime.active_terrain.is_empty());
        }
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        app.insert_resource(EditorGraph(graph));
        app.world_mut().resource_mut::<WorldListState>().phase = WorldListPhase::Playing;
        app.update();
        assert_eq!(
            app.world()
                .resource::<WorldRuntime>()
                .static_parts_for_physics(0),
            Some(vec![part])
        );
    }

    #[test]
    fn world_physics_waits_for_foundation_reconciliation() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let pending = PendingFoundationSync {
            editor_revision: 2,
            parts: graph.parts().map(|(part, spec)| (part, *spec)).collect(),
            frames: graph
                .parts()
                .map(|(part, _)| (part, graph.part_frame(part).unwrap()))
                .collect(),
            replaced_parts: BTreeSet::new(),
            new_parts: vec![part],
            next_part: 0,
            foundations: Vec::new(),
            index: FoundationSpatialIndex::default(),
        };

        assert_eq!(
            static_parts_for_physics(&pending.parts, &[], Some(&pending), 1, 2,),
            None
        );
    }

    #[test]
    fn loading_world_ignores_picker_actions_until_playing() {
        let mut state = WorldListState {
            phase: WorldListPhase::Loading,
            entries: Vec::new(),
            notice: None,
            loading_progress: TerrainReadiness::default(),
            confirming_delete: None,
            requested: None,
        };
        state.act(crate::ui::WorldAction::Create {
            name: "ignored".to_owned(),
            seed: String::new(),
        });
        assert!(state.requested.is_none());
        assert!(state.is_open());
        state.phase = WorldListPhase::Playing;
        assert!(!state.is_open());
        state.act(crate::ui::WorldAction::ExitToSelector);
        assert_eq!(
            state.requested,
            Some(crate::ui::WorldAction::ExitToSelector)
        );
    }

    #[test]
    fn exiting_to_the_selector_saves_and_allows_the_world_to_reload() {
        let temporary = TempWorldStore::new();
        let store = WorldStore::new(&temporary.0);
        let document = store.create_world("Reloadable", Some(7)).unwrap();
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::new(IVec3::ZERO, GridRotation::default()))
                    .unwrap(),
            ))
            .unwrap();

        let mut app = App::new();
        app.add_plugins(bevy::state::app::StatesPlugin);
        app.init_state::<AppSpace>();
        app.init_resource::<WorldRuntime>();
        {
            let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
            runtime.store = store;
            install_world(&mut runtime, document).unwrap();
        }
        app.init_resource::<WorldListState>();
        app.insert_resource(EditorGraph(graph));
        app.init_resource::<EditorState>();
        app.add_systems(Update, handle_world_list);
        {
            let mut list = app.world_mut().resource_mut::<WorldListState>();
            list.phase = WorldListPhase::Playing;
            list.act(crate::ui::WorldAction::ExitToSelector);
        }

        app.update();

        let path = {
            let list = app.world().resource::<WorldListState>();
            assert_eq!(list.phase(), WorldListPhase::Picking);
            list.entries()[0].path.clone()
        };
        app.world_mut()
            .resource_mut::<WorldListState>()
            .act(crate::ui::WorldAction::Open(path));

        app.update();

        assert_eq!(
            app.world().resource::<WorldListState>().phase(),
            WorldListPhase::Loading
        );
        assert_eq!(
            app.world()
                .resource::<WorldRuntime>()
                .pending_garage_editor
                .as_ref()
                .unwrap()
                .graph
                .part_count(),
            1
        );
    }

    fn frozen_save_fixture() -> (TempWorldStore, App, ConstructionGraph) {
        let temporary = TempWorldStore::new();
        let store = WorldStore::new(&temporary.0);
        let document = store.create_world("Frozen publication", Some(7)).unwrap();
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
                DimensionLinkId(1),
                BuildPose::new(IVec3::new(0, 4, 0), GridRotation::default()),
            )))
            .unwrap();
        let mut app = App::new();
        app.init_resource::<WorldRuntime>();
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        runtime.store = store;
        install_world(&mut runtime, document).unwrap();
        runtime.document.active_dimension_link = Some(DimensionLinkId(1));
        runtime.document.next_dimension_link_id = 2;
        let construction_generation = runtime.document.construction_generation;
        runtime
            .persist_frozen_creation(
                Some(mechanic_world::FrozenCreationDoc {
                    link: DimensionLinkId(1),
                    target: WorldPosition(DVec3::new(20.0, 8.0, 30.0)),
                    heading: 2,
                    construction_generation,
                }),
                &graph,
                &EditorState::default(),
            )
            .unwrap();
        (temporary, app, graph)
    }

    #[test]
    fn toggling_active_dimension_link_off_clears_and_persists_its_frozen_hold() {
        let (_temporary, mut app, graph) = frozen_save_fixture();
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        let link = graph.dimension_link(DimensionLinkId(1)).unwrap();
        // Deactivation must work even when activation would reject an occupied Garage.
        runtime.garage_editor = Some(SpaceEditorState {
            graph: graph.clone(),
            ..Default::default()
        });
        assert_eq!(
            runtime.toggle_dimension_link(AppSpace::World, &graph, link),
            Ok(None)
        );
        assert_eq!(runtime.active_dimension_link(), None);
        assert!(runtime.document.frozen_creation.is_none());
        let saved = runtime
            .store
            .load_world(&runtime.store.directory_for(&runtime.document.name))
            .unwrap();
        assert_eq!(saved.active_dimension_link, None);
        assert!(saved.frozen_creation.is_none());
        assert_eq!(
            runtime.toggle_dimension_link(AppSpace::Garage, &graph, link),
            Ok(Some(DimensionLinkId(1)))
        );
        assert_eq!(runtime.active_dimension_link(), Some(DimensionLinkId(1)));
    }

    fn spawn_unaccepted_frozen_edit(graph: &mut ConstructionGraph) {
        // This block occupies the held target, so it has not passed freeze clearance.
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4; 3],
                    BuildPose::new(IVec3::new(80, 32, 120), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
    }

    #[test]
    fn leaving_world_discards_body_state_and_keeps_the_construction_frame() {
        let temporary = TempWorldStore::new();
        let mut app = App::new();
        app.init_resource::<WorldRuntime>();
        app.init_resource::<EditorGraph>();
        app.init_resource::<EditorHistory>();
        app.init_resource::<EditorState>();
        app.init_resource::<bevy::prelude::ClearColor>();
        app.world_mut().spawn((
            crate::MainCamera,
            bevy::prelude::DistanceFog::default(),
            super::Exposure::default(),
        ));
        let graph = showcase::build_preset(showcase::CreationPreset::PendulumGarden256).unwrap();
        app.insert_resource(crate::AppSimulation {
            creation: Some(graph.compile().unwrap()),
            published_graph: graph.clone(),
            world_revision: Some((1, 1)),
            transforms: vec![mechanic_gpu::GpuTransform {
                position: [1.0; 4],
                rotation: [0.0, 0.0, 0.0, 1.0],
            }],
            ..Default::default()
        });
        app.insert_resource(EditorGraph(graph));
        let origin = FloatingOrigin(DVec3::new(12.0, 34.0, 56.0));
        {
            let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
            runtime.store = WorldStore::new(&temporary.0);
            runtime.document = runtime.store.create_world("Leave", Some(42)).unwrap();
            runtime.floating_origin = origin;
            runtime.garage_editor = Some(SpaceEditorState::default());
        }
        app.add_systems(Update, super::leave_world);
        app.update();
        let simulation = app.world().resource::<crate::AppSimulation>();
        assert!(simulation.creation.is_none());
        assert!(simulation.transforms.is_empty());
        assert!(simulation.live_state.is_none());
        assert_eq!(simulation.published_graph.part_count(), 0);
        let runtime = app.world().resource::<WorldRuntime>();
        assert_eq!(runtime.world_editor.as_ref().unwrap().origin, origin);
        assert!(runtime.world_editor.as_ref().unwrap().graph.part_count() > 0);
    }

    #[test]
    fn world_reload_preserves_construction_origin_after_player_moves() {
        let temporary = TempWorldStore::new();
        let mut app = App::new();
        app.init_resource::<WorldRuntime>();
        app.init_resource::<WorldListState>();
        app.init_resource::<EditorState>();
        app.init_resource::<WorldDiagnostics>();
        app.init_resource::<EditorHistory>();
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4; 3],
                    BuildPose::new(IVec3::new(8, 12, 16), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
        let origin = FloatingOrigin(DVec3::new(123.0, 45.0, -678.0));
        let bounds = graph_bounds(&graph).unwrap();
        {
            let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
            runtime.store = WorldStore::new(&temporary.0);
            runtime.document = runtime.store.create_world("Origin", Some(42)).unwrap();
            runtime.floating_origin = origin;
            runtime.capsule.position = WorldPosition(DVec3::new(-89.0, 123.0, 456.0));
            runtime.document.return_anchor = Some(runtime.capsule.position);
            super::save_all(&mut runtime).unwrap();
            super::save_world_instance(&mut runtime, &graph, &EditorState::default()).unwrap();
            let saved = runtime
                .store
                .load_world(&runtime.store.directory_for("Origin"))
                .unwrap();
            let (world, _) = load_space_editors(&runtime.store, &saved).unwrap();
            assert_eq!(world.origin, origin);
            assert_eq!(graph_bounds(&world.graph).unwrap(), bounds);
            // Saving again from the Garage must preserve the inactive World's root.
            runtime.world_editor = Some(world);
            super::save_garage_instance(
                &mut runtime,
                &ConstructionGraph::new(),
                &EditorState::default(),
            )
            .unwrap();
            let saved = runtime
                .store
                .load_world(&runtime.store.directory_for("Origin"))
                .unwrap();
            install_world(&mut runtime, saved).unwrap();
            assert_eq!(runtime.floating_origin, origin);
            let world = runtime.world_editor.take().unwrap();
            assert_eq!(world.origin, origin);
            assert_eq!(graph_bounds(&world.graph).unwrap(), bounds);
            let mut player = crate::camera::PlayerState::default();
            super::restore_world_player(&mut runtime, &mut player, world.origin);
            assert_eq!(runtime.floating_origin, origin);
            assert_eq!(
                runtime.local_to_global(player.position),
                runtime.document.return_anchor.unwrap()
            );
            graph = world.graph;
        }
        app.insert_resource(EditorGraph(graph));
        app.world_mut().resource_mut::<WorldListState>().phase = WorldListPhase::Playing;
        app.add_systems(Update, sync_world_foundations);
        app.update();
        let runtime = app.world().resource::<WorldRuntime>();
        let graph = app.world().resource::<EditorGraph>();
        let compiled = graph
            .0
            .compile_with_static_parts(runtime.static_parts_for_physics(0).unwrap())
            .unwrap();
        // This fixture has no terrain contact; loading must not invent a ground weld.
        assert!(compiled.compounds.iter().all(|body| !body.is_static));
        assert_eq!(
            runtime.local_to_global(bounds.0),
            WorldPosition(origin.0 + bounds.0.as_dvec3())
        );
    }

    #[test]
    fn frozen_save_pairs_target_with_only_the_last_accepted_construction() {
        let (_temporary, mut app, mut graph) = frozen_save_fixture();
        let editor = EditorState::default();
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        let target = runtime.frozen_creation().unwrap().target;
        spawn_unaccepted_frozen_edit(&mut graph);
        super::save_world_instance(&mut runtime, &graph, &editor).unwrap();
        let saved = runtime
            .store
            .load_world(&runtime.store.directory_for(&runtime.document.name))
            .unwrap();
        let (world, _) = load_space_editors(&runtime.store, &saved).unwrap();
        assert_eq!(world.graph.part_count(), 1);
        assert!(world.graph.dimension_link(DimensionLinkId(1)).is_some());
        let hold = saved.frozen_creation.unwrap();
        assert_eq!(hold.target, target);
        assert_eq!(hold.construction_generation, saved.construction_generation);
        let previous_generation = saved.construction_generation;

        // Publication acceptance is the explicit gate; save itself never validates geometry.
        runtime.accept_frozen_publication(&graph, &editor);
        super::save_world_instance(&mut runtime, &graph, &editor).unwrap();
        let saved = runtime
            .store
            .load_world(&runtime.store.directory_for(&runtime.document.name))
            .unwrap();
        let (world, _) = load_space_editors(&runtime.store, &saved).unwrap();
        assert_eq!(world.graph.part_count(), 2);
        assert!(saved.construction_generation > previous_generation);
        let hold = saved.frozen_creation.unwrap();
        assert_eq!(hold.target, target);
        assert_eq!(hold.construction_generation, saved.construction_generation);
    }

    #[test]
    fn frozen_global_target_survives_origin_change_and_release_saves_current_edits() {
        let (_temporary, mut app, mut graph) = frozen_save_fixture();
        let editor = EditorState::default();
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        runtime.floating_origin = FloatingOrigin(DVec3::new(1000.0, 0.0, -2000.0));
        let mut hold = runtime.frozen_creation().unwrap();
        hold.target = runtime.local_to_global(Vec3::new(3.0, 8.25, -4.0));
        let target = hold.target;
        runtime.set_frozen_target(hold);
        runtime.floating_origin = FloatingOrigin(DVec3::new(-500.0, 0.0, 1000.0));
        spawn_unaccepted_frozen_edit(&mut graph);
        super::save_world_instance(&mut runtime, &graph, &editor).unwrap();
        let saved = runtime
            .store
            .load_world(&runtime.store.directory_for(&runtime.document.name))
            .unwrap();
        assert_eq!(saved.frozen_creation.unwrap().target, target);
        assert_eq!(
            load_space_editors(&runtime.store, &saved)
                .unwrap()
                .0
                .graph
                .part_count(),
            1
        );
        install_world(&mut runtime, saved).unwrap();
        assert_eq!(runtime.frozen_creation().unwrap().target, target);
        assert_eq!(runtime.frozen_editor.as_ref().unwrap().0.part_count(), 1);

        runtime
            .persist_frozen_creation(None, &graph, &editor)
            .unwrap();
        assert!(runtime.frozen_editor.is_none());
        let saved = runtime
            .store
            .load_world(&runtime.store.directory_for(&runtime.document.name))
            .unwrap();
        assert!(saved.frozen_creation.is_none());
        assert_eq!(
            load_space_editors(&runtime.store, &saved)
                .unwrap()
                .0
                .graph
                .part_count(),
            2
        );
    }

    #[test]
    fn installing_a_new_world_replaces_the_previous_world_editor() {
        let mut editor = super::SpaceEditorState {
            graph: showcase::build_preset(showcase::CreationPreset::PendulumGarden256).unwrap(),
            ..super::SpaceEditorState::default()
        };
        assert!(editor.graph.part_count() > 0);
        let temporary = TempWorldStore::new();
        let store = WorldStore::new(&temporary.0);
        let document = store.create_world("Fresh", Some(42)).unwrap();

        editor = load_space_editors(&store, &document).unwrap().0;

        assert_eq!(editor.graph.part_count(), 0);
        assert!(editor.placed_bearings.is_empty());
    }

    #[test]
    fn brush_paths_preserve_every_five_centimetre_sample() {
        let start = WorldPosition(DVec3::new(-1.0, 2.0, 3.0));
        let end = WorldPosition(DVec3::new(-0.77, 2.0, 3.0));
        let operation = TerrainEditOperation::Remove;
        let previous = TerrainStrokeSample {
            centre: start,
            radius_metres: 0.5,
            operation,
        };
        let commands = terrain_edit_commands(Some(previous), end, 0.5, operation);

        assert_eq!(commands.len(), 5);
        let mut previous = start;
        for command in &commands {
            assert_eq!(command.previous, Some((previous, 0.5)));
            assert!(previous.0.distance(command.centre.0) <= 0.05 + 1.0e-12);
            previous = command.centre;
        }
        assert_eq!(previous, end);
        assert!(
            terrain_edit_commands(
                Some(TerrainStrokeSample {
                    centre: end,
                    radius_metres: 0.5,
                    operation,
                }),
                end,
                0.5,
                operation,
            )
            .is_empty()
        );
        assert_eq!(terrain_edit_commands(None, end, 0.5, operation).len(), 1);
    }

    #[test]
    fn publication_waves_do_not_reinvalidate_an_acknowledged_foundation_edit() {
        let pending = TerrainEditBatch {
            generation: 7,
            changed_bricks: BTreeSet::from([BrickCoord::new(1, 2, 3)]),
        };
        let acknowledgements = TerrainAcknowledgements {
            edit: 7,
            mesh: 7,
            upload: 7,
            collision: 7,
        };
        assert!(foundation_edit_is_ready(
            acknowledgements,
            &pending,
            6,
            true
        ));
        assert!(!foundation_edit_is_ready(
            acknowledgements,
            &pending,
            7,
            true
        ));
        assert!(!foundation_edit_is_ready(
            TerrainAcknowledgements {
                upload: 6,
                ..acknowledgements
            },
            &pending,
            6,
            true
        ));
    }

    #[test]
    fn continuous_stroke_waits_for_the_final_acknowledged_generation() {
        let pending = TerrainEditBatch {
            generation: 9,
            changed_bricks: BTreeSet::from([BrickCoord::new(0, 0, 0), BrickCoord::new(1, 0, 0)]),
        };
        let acknowledgements = TerrainAcknowledgements {
            edit: 9,
            mesh: 9,
            upload: 9,
            collision: 9,
        };
        assert!(!foundation_edit_is_ready(
            acknowledgements,
            &pending,
            8,
            false
        ));
        assert!(foundation_edit_is_ready(
            acknowledgements,
            &pending,
            8,
            true
        ));
    }

    #[test]
    fn large_construction_foundations_publish_over_bounded_frames() {
        let mut graph = ConstructionGraph::new();
        let mut edit = graph.begin_edit();
        edit.reserve_parts_and_welds(4_096, 0);
        edit.spawn_cuboids((0..64).flat_map(|x| {
            (0..64).map(move |z| {
                CuboidSpec::new(
                    [1; 3],
                    BuildPose::new(IVec3::new(x, 0, z), GridRotation::default()),
                )
                .unwrap()
            })
        }));
        graph = edit.finish();

        let mut app = App::new();
        app.init_resource::<WorldRuntime>();
        app.init_resource::<WorldListState>();
        app.init_resource::<EditorState>();
        app.init_resource::<WorldDiagnostics>();
        app.insert_resource(EditorGraph(graph));
        let history = EditorHistory {
            current_revision: 1,
            next_revision: 1,
            ..Default::default()
        };
        app.insert_resource(history);
        app.world_mut().resource_mut::<WorldListState>().phase = WorldListPhase::Playing;
        app.add_systems(Update, sync_world_foundations);

        app.update();
        let runtime = app.world().resource::<WorldRuntime>();
        let pending = runtime
            .pending_foundation_sync
            .as_ref()
            .expect("the first frame leaves bounded foundation work pending");
        assert!((1..=super::FOUNDATION_SYNC_MAX_PARTS_PER_FRAME).contains(&pending.next_part));
        assert!(!runtime.foundations_match_editor_revision(1));

        for _ in 0..4_096 {
            if app
                .world()
                .resource::<WorldRuntime>()
                .foundations_match_editor_revision(1)
            {
                break;
            }
            app.update();
        }
        let runtime = app.world().resource::<WorldRuntime>();
        assert!(runtime.foundations_match_editor_revision(1));
        assert_eq!(runtime.known_world_parts.len(), 4_096);
    }

    #[test]
    fn linked_creation_waits_for_bounded_world_foundation_sampling() {
        let mut graph = ConstructionGraph::new();
        let mut edit = graph.begin_edit();
        edit.reserve_parts_and_welds(129, 0);
        let _unrelated = edit.spawn_cuboids((0..128).map(|x| {
            CuboidSpec::new(
                [1; 3],
                BuildPose::new(IVec3::new(x, 0, 0), GridRotation::default()),
            )
            .unwrap()
        }));
        graph = edit.finish();
        let BuildOutcome::Spawned(_link) = graph
            .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
                DimensionLinkId(11),
                BuildPose::new(IVec3::new(300, 4, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };

        let mut app = App::new();
        app.init_resource::<WorldRuntime>();
        app.init_resource::<WorldListState>();
        app.init_resource::<EditorState>();
        app.init_resource::<WorldDiagnostics>();
        app.insert_resource(EditorGraph(graph));
        app.insert_resource(EditorHistory {
            current_revision: 1,
            next_revision: 1,
            ..Default::default()
        });
        app.world_mut().resource_mut::<WorldListState>().phase = WorldListPhase::Playing;
        app.add_systems(Update, sync_world_foundations);

        app.update();

        let runtime = app.world().resource::<WorldRuntime>();
        let pending = runtime.pending_foundation_sync.as_ref().unwrap();
        assert!(pending.next_part > 0);
        assert_eq!(runtime.static_parts_for_physics(1), None);
    }

    #[test]
    fn adding_dimension_link_preserves_cached_ground_anchors() {
        let mut previous_graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(chassis) = previous_graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([2, 1, 1], BuildPose::new(IVec3::Y, GridRotation::default()))
                    .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(unrelated) = previous_graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1; 3],
                    BuildPose::new(IVec3::new(20, 0, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let mut graph = previous_graph.clone();
        let BuildOutcome::Spawned(link) = graph
            .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
                DimensionLinkId(12),
                BuildPose::new(IVec3::new(2, 1, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(chassis, FaceKind::PositiveX),
                second: FaceRef::part(link, FaceKind::NegativeX),
            }))
            .unwrap();
        let valid_support = || FoundationSupport {
            samples: vec![FoundationSample {
                position: WorldPosition::default(),
                valid: true,
            }],
        };

        let mut app = App::new();
        app.init_resource::<WorldRuntime>();
        app.init_resource::<WorldListState>();
        app.init_resource::<EditorState>();
        app.init_resource::<WorldDiagnostics>();
        app.insert_resource(EditorGraph(graph));
        app.insert_resource(EditorHistory {
            current_revision: 2,
            next_revision: 2,
            ..Default::default()
        });
        app.world_mut().resource_mut::<WorldListState>().phase = WorldListPhase::Playing;
        {
            let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
            runtime.known_world_parts = previous_graph
                .parts()
                .map(|(part, spec)| (part, *spec))
                .collect();
            runtime.known_world_frames = previous_graph
                .parts()
                .map(|(part, _)| (part, previous_graph.part_frame(part).unwrap()))
                .collect();
            runtime.synced_editor_revision = 1;
            for part in [chassis, unrelated] {
                let support = valid_support();
                runtime.foundation_index.insert(part, &support);
                runtime
                    .foundations
                    .push(TerrainFoundation { part, support });
            }
        }
        app.add_systems(Update, sync_world_foundations);

        app.update();

        let runtime = app.world().resource::<WorldRuntime>();
        assert!(runtime.foundations_match_editor_revision(2));
        assert_eq!(
            runtime.anchored_parts().collect::<Vec<_>>(),
            vec![chassis, unrelated]
        );
        assert_eq!(
            runtime.static_parts_for_physics(2),
            Some(vec![chassis, unrelated])
        );
    }

    #[test]
    fn empty_terrain_chunks_are_not_published_to_bevys_mesh_allocator() {
        let mut chunk = TerrainMeshChunk::default();
        assert!(!terrain_mesh_is_renderable(&chunk, 0));

        chunk.vertices = vec![[0.0; 3]; 3];
        assert!(!terrain_mesh_is_renderable(&chunk, 0));
        assert!(terrain_mesh_is_renderable(&chunk, 3));
    }

    #[test]
    fn freeze_triangle_query_prunes_distant_geometry_and_converts_global_coordinates() {
        use mechanic_world::{
            TerrainTriangleGroupMask, TriangleBvh, TriangleBvhNode, TriangleBvhTriangle,
            WorldBounds,
        };
        let mut app = App::new();
        app.init_resource::<WorldRuntime>();
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        runtime.floating_origin = FloatingOrigin(DVec3::splat(1000.0));
        runtime.player_terrain_ready = true;
        let bounds = WorldBounds {
            minimum: WorldPosition(DVec3::splat(1000.0)),
            maximum: WorldPosition(DVec3::splat(1002.0)),
        };
        let group_mask = TerrainTriangleGroupMask::REGULAR;
        let chunk = TerrainMeshChunk {
            origin: WorldPosition(DVec3::splat(1000.0)),
            vertices: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
            triangle_bvh: TriangleBvh {
                bounds,
                nodes: vec![TriangleBvhNode {
                    bounds,
                    first_triangle: 0,
                    triangle_count: 1,
                    group_mask,
                    ..Default::default()
                }],
                triangles: vec![TriangleBvhTriangle {
                    indices: [0, 1, 2],
                    group_mask,
                }],
            },
            ..Default::default()
        };
        runtime.active_terrain.insert(chunk.node, chunk);
        let mut visits = 0;
        assert!(
            runtime.freeze_triangles_clear(Vec3::ZERO, Vec3::ONE, |points| {
                visits += 1;
                assert_eq!(points[0], Vec3::ZERO);
                assert_eq!(points[1], Vec3::X * 2.0);
                true
            })
        );
        assert_eq!(visits, 1);
        assert!(
            runtime.freeze_triangles_clear(Vec3::splat(20.0), Vec3::splat(21.0), |_| panic!(
                "distant triangles must be pruned"
            ))
        );
        assert!(!runtime.freeze_triangles_clear(Vec3::ZERO, Vec3::ONE, |_| false));
    }

    #[test]
    fn terrain_texture_coordinates_and_weights_are_chunk_seam_stable() {
        let chunk = TerrainMeshChunk {
            origin: WorldPosition(DVec3::new(15.0, 30.0, 45.0)),
            vertices: vec![[1.5, 3.0, 4.5]],
            normals: vec![[0.0, 1.0, 0.0]],
            material_weights: vec![[0.1, 0.2, 0.3, 0.15, 0.1, 0.15]],
            ..TerrainMeshChunk::default()
        };
        let mesh = terrain_chunk_mesh(&chunk, Vec::new());
        let Some(VertexAttributeValues::Float32x2(horizontal)) =
            mesh.attribute(bevy::mesh::Mesh::ATTRIBUTE_UV_0)
        else {
            panic!("terrain mesh must have horizontal texture coordinates")
        };
        let Some(VertexAttributeValues::Float32x2(vertical)) =
            mesh.attribute(bevy::mesh::Mesh::ATTRIBUTE_UV_1)
        else {
            panic!("terrain mesh must have vertical texture coordinates")
        };
        let Some(VertexAttributeValues::Float32x4(weights)) =
            mesh.attribute(bevy::mesh::Mesh::ATTRIBUTE_COLOR)
        else {
            panic!("terrain mesh must carry material weights as vertex colors")
        };
        assert_eq!(horizontal, &[[11.0, 33.0]]);
        assert_eq!(vertical, &[[22.0, 0.1]]);
        assert_eq!(weights, &[[0.1, 0.2, 0.3, 0.15]]);
    }

    #[test]
    fn terrain_pbr_maps_keep_the_authored_1536_pixel_top_mip() {
        let maps: [&[u8]; 9] = [
            include_bytes!("../assets/terrain/grass/grass_base_color.png"),
            include_bytes!("../assets/terrain/grass/grass_normal.png"),
            include_bytes!("../assets/terrain/grass/grass_orm.png"),
            include_bytes!("../assets/terrain/dirt/dirt_base_color.png"),
            include_bytes!("../assets/terrain/dirt/dirt_normal.png"),
            include_bytes!("../assets/terrain/dirt/dirt_orm.png"),
            include_bytes!("../assets/terrain/stone/stone_base_color.png"),
            include_bytes!("../assets/terrain/stone/stone_normal.png"),
            include_bytes!("../assets/terrain/stone/stone_orm.png"),
        ];
        for png in maps {
            assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
            assert_eq!(u32::from_be_bytes(png[16..20].try_into().unwrap()), 1536);
            assert_eq!(u32::from_be_bytes(png[20..24].try_into().unwrap()), 1536);
        }
    }

    #[test]
    fn terrain_runtime_textures_receive_a_complete_mip_chain() {
        let pixel = [40_u8, 80, 120, 255];
        let top = pixel.repeat(8);
        let mut image = Image::new(
            Extent3d {
                width: 4,
                height: 2,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            top.clone(),
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::MAIN_WORLD,
        );

        generate_rgba8_mip_chain(&mut image).unwrap();

        assert_eq!(image.texture_descriptor.mip_level_count, 3);
        assert_eq!(
            image.data.as_ref().unwrap().len(),
            full_rgba8_mip_byte_count(4, 2)
        );
        assert_eq!(&image.data.as_ref().unwrap()[..top.len()], top);
        assert!(
            image
                .data
                .as_ref()
                .unwrap()
                .chunks_exact(4)
                .all(|sample| sample == pixel)
        );
    }

    #[test]
    fn player_waits_for_collision_geometry_near_the_capsule() {
        let capsule = KinematicCapsule::new(WorldPosition(DVec3::new(0.0, 4.05, 0.0)));
        let mut chunk = TerrainMeshChunk {
            bounds: WorldBounds {
                minimum: WorldPosition(DVec3::new(-1.0, 3.0, -1.0)),
                maximum: WorldPosition(DVec3::new(1.0, 6.0, 1.0)),
            },
            ..TerrainMeshChunk::default()
        };
        assert!(!terrain_chunk_has_collision_near(
            &chunk,
            TerrainTransitionMask::NONE,
            &capsule,
        ));

        chunk.index_groups.regular = vec![0, 1, 2];
        assert!(terrain_chunk_has_collision_near(
            &chunk,
            TerrainTransitionMask::NONE,
            &capsule,
        ));
        chunk.bounds.minimum.0.x = 20.0;
        chunk.bounds.maximum.0.x = 25.0;
        assert!(!terrain_chunk_has_collision_near(
            &chunk,
            TerrainTransitionMask::NONE,
            &capsule,
        ));
    }

    #[test]
    fn capsule_overlapping_nodes_are_streamed_first() {
        let capsule = KinematicCapsule::new(WorldPosition(DVec3::new(0.0, 4.05, 0.0)));
        let local = ActiveTerrainNode {
            id: TerrainNodeId::containing(BrickCoord::new(0, 2, 0), 2).unwrap(),
            generation: 0,
            transition_mask: TerrainTransitionMask::NONE,
        };
        let far = ActiveTerrainNode {
            id: TerrainNodeId::containing(BrickCoord::new(100, 100, 100), 2).unwrap(),
            ..local
        };
        assert_eq!(
            player_collision_nodes(&[far, local], &capsule).collect::<Vec<_>>(),
            vec![local.id],
        );
    }

    #[test]
    fn generated_spawn_cut_contains_collision_pins() {
        let field = TerrainField::new(WorldSeed(7));
        let spawn = field.safe_spawn();
        let terrain = TerrainOctree::default().snapshot();
        let cut = select_active_nodes(&field, &terrain, spawn);
        let capsule = KinematicCapsule::new(spawn);
        let pins = player_collision_nodes(&cut, &capsule).collect::<Vec<_>>();
        assert!(!pins.is_empty());
        assert!(pins.into_iter().any(|id| {
            let node = cut
                .iter()
                .find(|node| node.id == id)
                .expect("a pin belongs to the selected cut");
            let chunk = mesh_chunk(
                &field,
                &terrain,
                TerrainMeshRequest {
                    node: id,
                    generation: node.generation,
                    transition_mask: node.transition_mask,
                },
            );
            terrain_chunk_has_collision_near(&chunk, TerrainTransitionMask::NONE, &capsule)
        }));
    }

    #[test]
    fn each_space_uses_exposure_matched_to_its_lighting() {
        assert!(
            (exposure_for_space(AppSpace::World).ev100 - Exposure::OVERCAST.ev100).abs()
                < f32::EPSILON
        );
        assert!(
            (exposure_for_space(AppSpace::Garage).ev100 - garage::EXPOSURE.ev100).abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn equal_and_two_to_one_nodes_share_the_expected_face() {
        let fine = TerrainNodeId::leaf(BrickCoord::new(1, 0, 0));
        let equal = TerrainNodeId::leaf(BrickCoord::new(2, 0, 0));
        let coarse = TerrainNodeId::containing(BrickCoord::new(2, 0, 0), 1).unwrap();
        assert!(nodes_touch_on_face(fine, equal, TerrainFace::PositiveX));
        assert!(nodes_touch_on_face(fine, coarse, TerrainFace::PositiveX));
    }

    #[test]
    fn old_lod_waits_until_every_visible_replacement_is_published() {
        let parent = TerrainNodeId::containing(BrickCoord::new(0, 0, 0), 1).unwrap();
        let children = BTreeSet::from(parent.children().unwrap());
        let obsolete = BTreeSet::from([parent]);
        let mut published = children.clone();
        let missing = *published.first().unwrap();
        published.remove(&missing);

        assert!(ready_obsolete_nodes(&obsolete, &children, &published).is_empty());
        published.insert(missing);
        assert_eq!(
            ready_obsolete_nodes(&obsolete, &children, &published),
            obsolete
        );
    }
    #[test]
    fn soil_commits_on_sixth_tick_and_survives_world_reload() {
        let temporary = TempWorldStore::new();
        let store = WorldStore::new(&temporary.0);
        let document = store.create_world("Soil", Some(91)).unwrap();
        let mut app = App::new();
        app.init_resource::<WorldRuntime>();
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        runtime.store = store;
        install_world(&mut runtime, document).unwrap();
        let patch = mechanic_world::SoilPatch {
            centre: WorldPosition(DVec3::new(0.0, runtime.field.surface_height(0.0, 0.0), 0.0)),
            normal: DVec3::Y,
            radius: 0.1,
            pressure_pa: 1.0e6,
            seconds: 1.0 / 60.0,
        };
        for _ in 0..5 {
            runtime.accumulate_soil(std::iter::once(patch));
        }
        assert!(runtime.pending_terrain_edits.is_empty());
        runtime.accumulate_soil(std::iter::once(patch));
        assert!(!runtime.pending_terrain_edits.is_empty());
        let commands = runtime.pending_terrain_edits.drain(..).collect();
        let result =
            super::execute_terrain_edit_batch(runtime.edits.clone(), &runtime.field, commands)
                .unwrap();
        assert!(super::commit_terrain_edit_result(&mut runtime, result).0);
        assert!(!runtime.pending_foundation_edit.is_empty());
        let store = WorldStore::new(&temporary.0);
        let name = runtime.document.name.clone();
        store.save_dirty_leaves(&name, &mut runtime.edits).unwrap();
        let reloaded = store.load_octree(&name).unwrap();
        for brick in runtime.edits.snapshot().bricks() {
            assert_eq!(Some(brick), reloaded.brick(brick.coordinate()));
        }
        let command = super::TerrainEditCommand {
            centre: patch.centre,
            radius_metres: 0.1,
            previous: None,
            operation: TerrainEditOperation::Remove,
        };
        runtime.pending_terrain_edits =
            std::iter::repeat_n(command, super::MAX_PENDING_TERRAIN_EDITS).collect();
        runtime.accumulate_soil(std::iter::once(patch));
        assert_eq!(
            runtime.pending_terrain_edits.len(),
            super::MAX_PENDING_TERRAIN_EDITS
        );
        assert!(runtime.pending_soil.take_ready().is_empty());
    }

    #[test]
    fn saving_an_unpublished_transfer_keeps_the_previous_material_owner() {
        let temporary = TempWorldStore::new();
        let store = WorldStore::new(&temporary.0);
        let document = store.create_world("Material", Some(91)).unwrap();
        let mut app = App::new();
        app.init_resource::<WorldRuntime>();
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        runtime.store = store;
        install_world(&mut runtime, document).unwrap();
        let centre = WorldPosition(DVec3::new(0.025, 200.025, 0.025));
        let field = runtime.field.clone();
        runtime
            .edits
            .add_sphere(&field, centre, 0.1, TerrainMaterial::Rock)
            .unwrap();
        let cell = centre.cell().unwrap();
        let source = mechanic_world::ExtractionCell {
            cell,
            sample: runtime.edits.sample_cell(&field, cell),
        };
        let transfer = runtime
            .clumps
            .prepare_extraction(&runtime.edits, &field, &[source])
            .unwrap();
        runtime.pending_material = Some(super::PendingMaterialPublication {
            previous: runtime.edits.clone(),
            clumps: transfer.clumps,
            sources: vec![source],
        });
        super::commit_terrain_edit_result(
            &mut runtime,
            super::TerrainEditTaskResult {
                terrain: transfer.terrain,
                outcomes: vec![transfer.outcome],
                elapsed_ms: 0.0,
            },
        );
        super::save_all(&mut runtime).unwrap();
        let (saved, clumps) = runtime.store.load_material_state("Material").unwrap();
        assert!(saved.sample_cell(&field, cell).is_solid());
        assert!(clumps.bodies.is_empty());
        super::finish_terrain_edits(&mut runtime).unwrap();
        assert!(runtime.edits.sample_cell(&field, cell).is_solid());
        assert!(!runtime.material_publication_pending());
    }

    /// Updates until `done`, failing instead of waiting forever on a stalled pipeline.
    fn update_until(app: &mut App, what: &str, done: impl Fn(&App) -> bool) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_mins(2);
        while !done(app) {
            assert!(std::time::Instant::now() < deadline, "timed out: {what}");
            app.update();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    #[test]
    fn a_settled_saved_clump_neither_blocks_loading_nor_waits_for_physics_to_deposit() {
        use bevy::prelude::IntoScheduleConfigs;

        bevy::tasks::AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::new);
        let temporary = TempWorldStore::new();
        let store = WorldStore::new(&temporary.0);
        let document = store.create_world("Settled", Some(91)).unwrap();
        let mut app = App::new();
        app.init_resource::<WorldRuntime>()
            .init_resource::<WorldListState>()
            .init_resource::<WorldDiagnostics>()
            .init_resource::<EditorState>()
            .init_resource::<EditorGraph>()
            .init_resource::<super::AppSimulation>()
            .init_resource::<bevy::prelude::Assets<bevy::prelude::Mesh>>()
            .add_systems(
                Update,
                (
                    super::coordinate_terrain_edits,
                    super::schedule_terrain_remeshes,
                    super::integrate_terrain_remeshes,
                )
                    .chain(),
            );
        let player = {
            let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
            runtime.store = store;
            install_world(&mut runtime, document).unwrap();
            runtime.terrain_material = Some(bevy::prelude::Handle::default());
            let spawn = runtime.capsule.position.0;
            let surface = runtime.field.surface_height(spawn.x, spawn.z);
            // Saved exactly as a play session leaves it: soft, at rest, and
            // already past the deposit delay, with no physics scene yet.
            let clump = mechanic_world::MaterialClump {
                id: 1,
                material: TerrainMaterial::Soil,
                quanta: 510 * 8,
                half_extents: DVec3::splat(0.05),
                position: WorldPosition(DVec3::new(spawn.x, surface + 0.05, spawn.z)),
                rotation: bevy::math::DQuat::IDENTITY,
                linear_velocity: DVec3::ZERO,
                angular_velocity: DVec3::ZERO,
                settled_seconds: 300.0,
                sleeping: false,
            };
            assert!(clump.is_valid() && clump.can_deposit());
            runtime.clumps.bodies.insert(clump.id, clump);
            runtime.clumps.next_id = 2;
            (spawn - runtime.floating_origin.0).as_vec3()
        };
        app.insert_resource(super::PlayerState {
            position: player,
            ..Default::default()
        });
        app.world_mut().resource_mut::<WorldListState>().phase = WorldListPhase::Loading;

        update_until(&mut app, "the world never finished loading", |app| {
            let playing =
                app.world().resource::<WorldListState>().phase() == WorldListPhase::Playing;
            // A transfer holds terrain publication, and with it loading progress.
            assert!(
                playing
                    || !app
                        .world()
                        .resource::<WorldRuntime>()
                        .material_publication_pending()
            );
            playing
        });
        assert!(app.world().resource::<super::AppSimulation>().cpu.is_none());

        update_until(&mut app, "the clump never deposited", |app| {
            let runtime = app.world().resource::<WorldRuntime>();
            runtime.clumps.bodies.is_empty() && !runtime.material_publication_pending()
        });
    }
}
