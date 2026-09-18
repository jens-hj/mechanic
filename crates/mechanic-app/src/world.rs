//! Bounded procedural-world prototype state and playable terrain tools.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod brush;
mod clumps;
mod foundations;
mod list;
mod saving;
mod space;
pub(crate) mod streaming;
mod terrain_render;
mod transfer;
mod walking;

use brush::{
    MAX_PENDING_TERRAIN_EDITS, TerrainEditCommand, TerrainEditOperation, TerrainEditTaskResult,
    TerrainStrokeSample, commit_terrain_edit_result, coordinate_terrain_edits,
    select_and_size_brush, use_brush,
};
pub(crate) use foundations::sync_world_foundations;
use foundations::{PendingFoundationSync, TerrainFoundation};
pub(crate) use list::{WorldListPhase, WorldListState};
use list::{handle_world_list, world_list_closed};
use saving::{autosave_world, save_on_exit, save_world_instance};
use space::{
    application_world_store, enter_world, leave_world, load_space_editors, restore_initial_garage,
    toggle_space,
};
use streaming::{
    PendingMaterialPublication, TerrainAcknowledgements, TerrainMeshResult,
    TerrainSelectionTaskResult, integrate_terrain_remeshes, schedule_terrain_remeshes,
};
use terrain_render::prepare_terrain_texture_mips;
pub(crate) use terrain_render::{TerrainRenderMaterial, generate_rgba8_mip_chain};
pub(crate) use transfer::place_loaded_creation_in_garage;
use transfer::static_parts_for_physics;
use walking::{PlayerCollisionBuild, walk_world};

use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
    time::Duration,
};

use bevy::{
    camera::Exposure,
    math::{DVec2, DVec3},
    prelude::*,
    tasks::Task,
};
use mechanic_core::{ConstructionGraph, CreationDocument, DimensionLinkId, PartId, PartSpec};
use mechanic_gpu::GpuExternalImpulse;
use mechanic_world::{
    ActiveTerrainScene, AutosaveState, ConstructionBodyPose, ConstructionCollisionIndex,
    FloatingOrigin, FoundationSpatialIndex, KinematicCapsule, SavedWorldStatus, TerrainBoundsCache,
    TerrainDensity, TerrainEditBatch, TerrainField, TerrainMaterial, TerrainMeshChunk,
    TerrainNodeId, TerrainOctree, TerrainSpatialIndex, TerrainStreamer, TerrainTransitionMask,
    WorldBounds, WorldDocument, WorldPosition, WorldSeed, WorldStore,
};

use crate::editor::build_actions::PlacedBearing;
use crate::editor::history::EditorHistory;
use crate::editor::state::EditorState;
use crate::schedule::FrameSet;
use crate::{
    builder::{GROUND_HALF_SIZE, composed_part_world_bounds},
    garage,
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

#[derive(Component)]
struct WorldOwned;

#[derive(Default)]
struct SpaceEditorState {
    origin: FloatingOrigin,
    graph: ConstructionGraph,
    history: EditorHistory,
    placed_bearings: Vec<PlacedBearing>,
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

impl FromWorld for WorldRuntime {
    #[expect(
        clippy::too_many_lines,
        reason = "initialize one coherent world and its persisted material ownership"
    )]
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
                    select_and_size_brush.after(FrameSet::Input),
                    walk_world.after(FrameSet::Readback),
                    use_brush.after(walk_world),
                    coordinate_terrain_edits.after(use_brush),
                    prepare_terrain_texture_mips,
                    schedule_terrain_remeshes.after(coordinate_terrain_edits),
                    integrate_terrain_remeshes.after(schedule_terrain_remeshes),
                    clumps::sync_clump_rendering.after(integrate_terrain_remeshes),
                    sync_world_foundations
                        .after(integrate_terrain_remeshes)
                        .after(FrameSet::Build)
                        .before(FrameSet::Simulation),
                    autosave_world.after(integrate_terrain_remeshes),
                    save_on_exit.after(autosave_world),
                )
                    .run_if(in_state(AppSpace::World)),
            )
            .add_systems(
                Update,
                toggle_space
                    .after(FrameSet::Input)
                    .run_if(world_list_closed),
            )
            .add_systems(Update, handle_world_list.after(FrameSet::Commands));
    }
}

pub(crate) fn world_playing(list: Res<WorldListState>) -> bool {
    list.phase() == WorldListPhase::Playing
}

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
mod tests;
