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
    math::DVec2,
    mesh::Indices,
    prelude::*,
    render::render_resource::{AsBindGroup, Face, PrimitiveTopology, TextureFormat},
    shader::ShaderRef,
    tasks::{AsyncComputeTaskPool, Task, block_on, futures::check_ready},
};
use mechanic_core::{ConstructionEditDelta, ConstructionGraph, PartId, PartSpec};
#[cfg(test)]
use mechanic_world::TerrainFace;
use mechanic_world::{
    ActiveTerrainNode, ActiveTerrainScene, AutosaveState, FloatingOrigin, FoundationSpatialIndex,
    FoundationSupport, KinematicCapsule, KinematicInput, OpenWorldResult, SavedWorld,
    SavedWorldStatus, TerrainBoundsCache, TerrainEditBatch, TerrainEditOutcome, TerrainField,
    TerrainMaterial, TerrainMeshChunk, TerrainMeshMetrics, TerrainMeshRequest, TerrainNodeId,
    TerrainOctree, TerrainRayHit, TerrainReadiness, TerrainSelection, TerrainSpatialIndex,
    TerrainStreamer, TerrainTransitionMask, WorldCreationInstanceDoc, WorldDocument,
    WorldInstanceIndexDoc, WorldPoseDoc, WorldPosition, WorldSeed, WorldStore, mesh_chunk_profiled,
    select_active_nodes_cached, terrain_loading_worker_count, terrain_worker_count,
};

use crate::{
    AppSimulation, EditorGraph, EditorHistory, EditorState, PlacedBearing,
    builder::part_world_bounds,
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
    pub(crate) const fn uses_bounded_garage_walking(self) -> bool {
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
const TERRAIN_EDIT_BATCH_SIZE: usize = 64;

#[derive(Component)]
struct WorldOwned;

#[derive(Component)]
struct BrushPreview;

#[derive(Component)]
struct TerrainNodeRender;

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub(crate) struct TerrainRenderMaterial {
    #[texture(0)]
    #[sampler(9)]
    grass_base_color: Handle<Image>,
    #[texture(1)]
    dirt_base_color: Handle<Image>,
    #[texture(2)]
    stone_base_color: Handle<Image>,
    #[texture(3)]
    grass_normal: Handle<Image>,
    #[texture(4)]
    dirt_normal: Handle<Image>,
    #[texture(5)]
    stone_normal: Handle<Image>,
    #[texture(6)]
    grass_orm: Handle<Image>,
    #[texture(7)]
    dirt_orm: Handle<Image>,
    #[texture(8)]
    stone_orm: Handle<Image>,
}

impl Material for TerrainRenderMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/terrain_material.wgsl".into()
    }
}

#[derive(Component)]
struct TerrainMeshTask {
    node: ActiveTerrainNode,
    task: Task<Result<TerrainMeshResult, String>>,
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
}

struct TerrainEditTaskResult {
    terrain: TerrainOctree,
    outcomes: Vec<TerrainEditOutcome>,
    elapsed_ms: f64,
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
    graph: ConstructionGraph,
    history: EditorHistory,
    placed_bearings: Vec<PlacedBearing>,
}

#[derive(Clone, Debug)]
struct TerrainFoundation {
    part: PartId,
    support: FoundationSupport,
}

/// Values contributed to the existing F3 overlay by the terrain pipeline.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub(crate) struct WorldDiagnostics {
    pub(crate) terrain_stage_ms: f64,
    pub(crate) selection_ms: f64,
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
        if self.phase == WorldListPhase::Picking {
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
    capsule: KinematicCapsule,
    floating_origin: FloatingOrigin,
    autosave: AutosaveState,
    brush_radius: f64,
    brush_selected: bool,
    last_brush_edit: Option<(WorldPosition, f64)>,
    pending_terrain_edits: VecDeque<TerrainEditCommand>,
    terrain_edit_task: Option<Task<Result<TerrainEditTaskResult, String>>>,
    terrain_edit_error: Option<String>,
    removed_cells: [u64; 3],
    clock: Duration,
    load_error: Option<String>,
    garage_editor: Option<SpaceEditorState>,
    world_editor: Option<SpaceEditorState>,
    known_world_parts: BTreeMap<PartId, PartSpec>,
    foundations: Vec<TerrainFoundation>,
    foundation_index: FoundationSpatialIndex,
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
    selected_terrain_revision: u64,
}

impl WorldRuntime {
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

    pub(crate) fn anchored_parts(&self) -> impl Iterator<Item = PartId> + '_ {
        self.foundations
            .iter()
            .filter(|foundation| foundation.support.has_valid_anchor())
            .map(|foundation| foundation.part)
    }

    pub(crate) const fn foundation_revision(&self) -> u64 {
        self.foundation_revision
    }
}

fn load_world_editor(
    store: &WorldStore,
    document: &WorldDocument,
) -> Result<SpaceEditorState, String> {
    let Some(index) = document.instances.first() else {
        return Ok(SpaceEditorState::default());
    };
    let path = store
        .directory_for(&document.name)
        .join("instances")
        .join(format!("{}.ron", index.id));
    let instance = store
        .load_instance(&path)
        .map_err(|error| error.to_string())?;
    let loaded = instance
        .creation
        .into_graph()
        .map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(SpaceEditorState {
        graph: loaded.graph,
        placed_bearings: loaded
            .sockets
            .into_iter()
            .map(|socket| PlacedBearing {
                source: socket.source,
                anchor: socket.anchor,
                dimensions: socket.dimensions,
            })
            .collect(),
        ..SpaceEditorState::default()
    })
}

impl FromWorld for WorldRuntime {
    fn from_world(_world: &mut World) -> Self {
        let store = WorldStore::platform_default().unwrap_or_else(|| WorldStore::new("worlds"));
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
        let (edits, load_error) = match store.load_octree(&document.name) {
            Ok(edits) => (edits, None),
            Err(error) => (TerrainOctree::default(), Some(error.to_string())),
        };
        let (world_editor, instance_error) = match load_world_editor(&store, &document) {
            Ok(editor) => (editor, None),
            Err(error) => (SpaceEditorState::default(), Some(error)),
        };
        let load_error = load_error.or(instance_error);
        let capsule = KinematicCapsule::new(document.player_pose.translation);
        Self {
            store,
            document,
            field: Arc::new(field),
            edits,
            capsule,
            floating_origin: FloatingOrigin::default(),
            autosave: AutosaveState::default(),
            brush_radius: 0.5,
            brush_selected: false,
            last_brush_edit: None,
            pending_terrain_edits: VecDeque::new(),
            terrain_edit_task: None,
            terrain_edit_error: None,
            removed_cells: [0; 3],
            clock: Duration::ZERO,
            load_error,
            garage_editor: None,
            world_editor: Some(world_editor),
            known_world_parts: BTreeMap::new(),
            foundations: Vec::new(),
            foundation_index: FoundationSpatialIndex::default(),
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
            selected_terrain_revision: u64::MAX,
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
            .add_systems(OnEnter(AppSpace::World), enter_world)
            .add_systems(OnExit(AppSpace::World), leave_world)
            .add_systems(
                Update,
                (
                    select_and_size_brush.after(crate::controls::update_action_state),
                    walk_world.after(crate::camera::update_player_camera),
                    use_brush.after(walk_world),
                    coordinate_terrain_edits.after(use_brush),
                    prepare_terrain_texture_mips,
                    schedule_terrain_remeshes.after(coordinate_terrain_edits),
                    integrate_terrain_remeshes.after(schedule_terrain_remeshes),
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
            .add_systems(Update, handle_world_list);
    }
}

fn world_list_closed(list: Res<WorldListState>) -> bool {
    !list.is_open()
}

pub(crate) fn world_playing(list: Res<WorldListState>) -> bool {
    list.phase() == WorldListPhase::Playing
}

fn handle_world_list(
    mut list: ResMut<WorldListState>,
    mut runtime: ResMut<WorldRuntime>,
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
                    match install_world(&mut runtime, document) {
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
    }
}

fn install_world(runtime: &mut WorldRuntime, document: WorldDocument) -> Result<(), String> {
    let terrain = runtime
        .store
        .load_octree(&document.name)
        .map_err(|error| error.to_string())?;
    let world_editor = load_world_editor(&runtime.store, &document)?;
    runtime.field = Arc::new(TerrainField::with_version(
        document.seed,
        document.generator_version,
    ));
    runtime.capsule = KinematicCapsule::new(document.player_pose.translation);
    runtime.document = document;
    runtime.edits = terrain;
    runtime.world_editor = Some(world_editor);
    runtime.known_world_parts.clear();
    runtime.foundations.clear();
    runtime.foundation_index = FoundationSpatialIndex::default();
    runtime.foundation_revision = 0;
    runtime.terrain_revision = 0;
    runtime.terrain_acknowledgements = TerrainAcknowledgements::default();
    runtime.pending_foundation_edit = TerrainEditBatch::default();
    runtime.foundation_edit_acknowledgement = 0;
    runtime.synced_editor_revision = 0;
    runtime.autosave = AutosaveState::default();
    runtime.last_brush_edit = None;
    runtime.pending_terrain_edits.clear();
    runtime.terrain_edit_task = None;
    runtime.terrain_edit_error = None;
    runtime.removed_cells = [0; 3];
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

#[allow(clippy::too_many_arguments)]
fn toggle_space(
    actions: Res<ButtonInput<GameAction>>,
    space: Res<State<AppSpace>>,
    mut next: ResMut<NextState<AppSpace>>,
    mut runtime: ResMut<WorldRuntime>,
    player: Res<PlayerState>,
    graph: Res<EditorGraph>,
    mut editor: ResMut<EditorState>,
    mut list: ResMut<WorldListState>,
) {
    if !actions.just_pressed(GameAction::ToggleSpace) {
        return;
    }
    match space.get() {
        AppSpace::Garage => {
            list.phase = WorldListPhase::Loading;
            list.loading_progress = TerrainReadiness::default();
            list.notice = None;
            next.set(AppSpace::World);
        }
        AppSpace::World => {
            if runtime.terrain_edit_task.is_some() || !runtime.pending_terrain_edits.is_empty() {
                editor.feedback = Some("Finishing queued terrain edits…".to_owned());
                return;
            }
            let global = WorldPosition(runtime.floating_origin.0 + player.position.as_dvec3());
            runtime.document.return_anchor = Some(global);
            runtime.document.player_pose.translation = global;
            let _ = save_all(&mut runtime);
            let _ = save_world_instance(&mut runtime, &graph.0, &editor);
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
    runtime.garage_editor = Some(SpaceEditorState {
        graph: core::mem::take(&mut graph.0),
        history: core::mem::take(&mut *history),
        placed_bearings: core::mem::take(&mut editor.placed_bearings),
    });
    let world_editor = runtime.world_editor.take().unwrap_or_default();
    graph.0 = world_editor.graph;
    *history = world_editor.history;
    editor.placed_bearings = world_editor.placed_bearings;
    crate::cancel_transient_editor_state(&mut graph.0, &mut editor);
    editor.construction_mesh_dirty = true;

    let start = runtime
        .document
        .return_anchor
        .unwrap_or(runtime.document.player_pose.translation);
    runtime.capsule = KinematicCapsule::new(start);
    runtime.floating_origin.0 = start.0.round();
    player.position = start.relative_to(runtime.floating_origin);
    player.seat = None;
    editor.feedback = runtime.load_error.clone().or_else(|| {
        Some("World — Shift sprint · Space jump · F6 Garage · T terrain brush".to_owned())
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
    let grass_normal = texture("terrain/grass/grass_normal.png", false);
    let dirt_normal = texture("terrain/dirt/dirt_normal.png", false);
    let stone_normal = texture("terrain/stone/stone_normal.png", false);
    let grass_orm = texture("terrain/grass/grass_orm.png", false);
    let dirt_orm = texture("terrain/dirt/dirt_orm.png", false);
    let stone_orm = texture("terrain/stone/stone_orm.png", false);
    runtime.terrain_texture_mips_pending = vec![
        grass_base_color.clone(),
        dirt_base_color.clone(),
        stone_base_color.clone(),
        grass_normal.clone(),
        dirt_normal.clone(),
        stone_normal.clone(),
        grass_orm.clone(),
        dirt_orm.clone(),
        stone_orm.clone(),
    ];
    runtime.terrain_material = Some(terrain_materials.add(TerrainRenderMaterial {
        grass_base_color,
        dirt_base_color,
        stone_base_color,
        grass_normal,
        dirt_normal,
        stone_normal,
        grass_orm,
        dirt_orm,
        stone_orm,
    }));
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

fn generate_rgba8_mip_chain(image: &mut Image) -> Result<(), String> {
    if image.texture_descriptor.mip_level_count > 1 {
        return Ok(());
    }
    if !matches!(
        image.texture_descriptor.format,
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
    ) {
        return Err(format!(
            "terrain texture has unsupported runtime format {:?}",
            image.texture_descriptor.format
        ));
    }
    let width = image.texture_descriptor.size.width;
    let height = image.texture_descriptor.size.height;
    if image.texture_descriptor.size.depth_or_array_layers != 1 {
        return Err("terrain texture must be a single 2D image".to_owned());
    }
    let expected_top_bytes = usize::try_from(u64::from(width) * u64::from(height) * 4)
        .map_err(|_| "terrain texture dimensions overflow memory size".to_owned())?;
    let Some(top) = image.data.as_ref() else {
        return Err("terrain texture has no CPU pixel data".to_owned());
    };
    if top.len() != expected_top_bytes {
        return Err(format!(
            "terrain texture contains {} bytes, expected {expected_top_bytes}",
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
    simulation.gpu = None;
    simulation.world_revision = None;
    runtime.world_editor = Some(SpaceEditorState {
        graph: core::mem::take(&mut graph.0),
        history: core::mem::take(&mut *history),
        placed_bearings: core::mem::take(&mut editor.placed_bearings),
    });
    runtime.terrain_streamer = TerrainStreamer::default();
    runtime.terrain_selection_task = None;
    runtime.pending_terrain_edits.clear();
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

fn walk_world(
    time: Res<Time>,
    actions: Res<ButtonInput<GameAction>>,
    camera: Single<&PlayerCamera, With<MainCamera>>,
    list: Res<WorldListState>,
    mut runtime: ResMut<WorldRuntime>,
    mut player: ResMut<PlayerState>,
    mut world_owned: Query<&mut Transform, With<WorldOwned>>,
) {
    if list.phase() != WorldListPhase::Playing {
        runtime.capsule.velocity = bevy::math::DVec3::ZERO;
        return;
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
    let mut capsule = runtime.capsule;
    if !runtime.player_terrain_ready {
        let ready = runtime.active_terrain.iter().any(|(&id, chunk)| {
            terrain_chunk_has_collision_near(
                chunk,
                runtime
                    .active_terrain_ready_faces
                    .get(&id)
                    .copied()
                    .unwrap_or_default(),
                capsule,
            )
        });
        if !ready {
            capsule.velocity = bevy::math::DVec3::ZERO;
            runtime.capsule = capsule;
            player.position = capsule.position.relative_to(runtime.floating_origin);
            return;
        }
        runtime.player_terrain_ready = true;
    }
    let scene = ActiveTerrainScene {
        chunks: &runtime.active_terrain,
        ready_faces: &runtime.active_terrain_ready_faces,
        spatial_index: &runtime.active_terrain_index,
    };
    capsule.tick(
        &scene,
        KinematicInput {
            movement: bevy::math::DVec2::new(f64::from(movement.x), f64::from(movement.z)),
            sprint: actions.pressed(GameAction::Sprint),
            jump: actions.just_pressed(GameAction::Jump),
        },
        f64::from(time.delta_secs().min(1.0 / 30.0)),
    );
    runtime.capsule = capsule;
    // The active World construction graph is still authored in this local frame. Keep that
    // frame stable across the finite prototype world until instances gain their own root entity.
    if let Some(shift) = runtime
        .floating_origin
        .rebase_for(capsule.position, 9_000.0)
    {
        for mut transform in &mut world_owned {
            transform.translation -= shift.as_vec3();
        }
        player.position -= shift.as_vec3();
    }
    player.position = capsule.position.relative_to(runtime.floating_origin);
    runtime.document.player_pose.translation = capsule.position;
}

fn select_and_size_brush(
    actions: Res<ButtonInput<GameAction>>,
    mut runtime: ResMut<WorldRuntime>,
    mut editor: ResMut<EditorState>,
    mut selection: ResMut<crate::hotbar::SelectedTool>,
) {
    if runtime.brush_selected
        && (selection.is_changed() && selection.0.is_some()
            || GameAction::TOOL_ACTIONS
                .into_iter()
                .any(|(action, _)| actions.just_pressed(action)))
    {
        runtime.brush_selected = false;
    }
    if actions.just_pressed(GameAction::ToolTerrainBrush) {
        runtime.brush_selected = !runtime.brush_selected;
        selection.0 = if runtime.brush_selected {
            None
        } else {
            Some(crate::hotbar::Tool::Block)
        };
    }
    if actions.just_pressed(GameAction::TerrainBrushDecrease) {
        runtime.brush_radius = (runtime.brush_radius - 0.05).max(0.10);
    }
    if actions.just_pressed(GameAction::TerrainBrushIncrease) {
        runtime.brush_radius = (runtime.brush_radius + 0.05).min(2.00);
    }
    if actions.just_pressed(GameAction::ToolTerrainBrush)
        || actions.just_pressed(GameAction::TerrainBrushDecrease)
        || actions.just_pressed(GameAction::TerrainBrushIncrease)
    {
        editor.feedback = Some(format!(
            "Terrain brush {} — {:.2} m radius",
            if runtime.brush_selected {
                "selected"
            } else {
                "stowed"
            },
            runtime.brush_radius
        ));
    }
}

fn use_brush(
    actions: Res<ButtonInput<GameAction>>,
    camera: Single<&GlobalTransform, With<MainCamera>>,
    mut preview: Single<(&mut Transform, &mut Visibility), With<BrushPreview>>,
    list: Res<WorldListState>,
    mut runtime: ResMut<WorldRuntime>,
    mut editor: ResMut<EditorState>,
) {
    if list.phase() != WorldListPhase::Playing {
        runtime.last_brush_edit = None;
        *preview.1 = Visibility::Hidden;
        return;
    }
    if !runtime.brush_selected {
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
    if !actions.pressed(GameAction::Primary) {
        runtime.last_brush_edit = None;
        return;
    }
    let radius = runtime.brush_radius;
    let previous = runtime.last_brush_edit;
    let commands = terrain_edit_commands(previous, hit.position, radius);
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
    runtime.last_brush_edit = Some((hit.position, radius));
}

fn terrain_edit_commands(
    previous: Option<(WorldPosition, f64)>,
    centre: WorldPosition,
    radius_metres: f64,
) -> Vec<TerrainEditCommand> {
    const SAMPLE_INTERVAL_METRES: f64 = 0.05;
    let Some((previous_centre, previous_radius)) = previous else {
        return vec![TerrainEditCommand {
            centre,
            radius_metres,
            previous: None,
        }];
    };
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
    for command in batch {
        outcomes.push(
            terrain
                .excavate_sphere_delta(
                    field,
                    command.centre,
                    command.radius_metres,
                    command.previous,
                )
                .map_err(|error| error.to_string())?,
        );
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
        changed |= outcome.total_removed_cells() != 0;
        changed_brick_coordinates.extend(outcome.changed_brick_coordinates().iter().copied());
        changed_bricks = changed_bricks
            .saturating_add(u64::try_from(outcome.changed_bricks).unwrap_or(u64::MAX));
        for material in [
            TerrainMaterial::SurfaceCover,
            TerrainMaterial::Soil,
            TerrainMaterial::Rock,
        ] {
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
) {
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
                let capsule = runtime.capsule;
                let critical = startup_region_nodes(&cut, capsule.position).collect::<Vec<_>>();
                runtime.terrain_streamer.set_pinned(
                    player_collision_nodes(&cut, capsule).chain(critical.iter().copied()),
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

    let needs_selection = runtime.selected_terrain_revision != runtime.terrain_revision
        || runtime
            .selection_focus
            .is_none_or(|previous| previous.0.distance(focus.0) >= RESELECT_DISTANCE_METRES);
    if needs_selection && runtime.terrain_selection_task.is_none() && runtime.load_error.is_none() {
        let field = Arc::clone(&runtime.field);
        let terrain = runtime.edits.snapshot();
        let terrain_revision = runtime.terrain_revision;
        let mut bounds_cache = core::mem::take(&mut runtime.terrain_bounds_cache);
        runtime.terrain_selection_task = Some(AsyncComputeTaskPool::get().spawn(async move {
            let started = std::time::Instant::now();
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                select_active_nodes_cached(&field, &terrain, focus, &mut bounds_cache)
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
    player: Res<PlayerState>,
    list: Res<WorldListState>,
    tasks: Query<&TerrainMeshTask>,
    mut diagnostics: ResMut<WorldDiagnostics>,
) {
    let focus = WorldPosition(runtime.floating_origin.0 + player.position.as_dvec3());
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
    for (offset, id) in dirty.iter().copied().enumerate() {
        if started.elapsed().as_secs_f64() * 1_000.0 >= INTEGRATION_BUDGET_MS {
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
                    runtime.capsule,
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
}

fn terrain_mesh_is_renderable(chunk: &TerrainMeshChunk, index_count: usize) -> bool {
    !chunk.vertices.is_empty() && index_count != 0
}

fn player_collision_nodes(
    cut: &[ActiveTerrainNode],
    capsule: KinematicCapsule,
) -> impl Iterator<Item = TerrainNodeId> + '_ {
    let (minimum, maximum) = capsule_loading_bounds(capsule);
    cut.iter()
        .map(|node| node.id)
        .filter(move |&id| node_overlaps(id, minimum, maximum))
}

fn terrain_chunk_has_collision_near(
    chunk: &TerrainMeshChunk,
    ready_faces: TerrainTransitionMask,
    capsule: KinematicCapsule,
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

fn capsule_loading_bounds(capsule: KinematicCapsule) -> (bevy::math::DVec3, bevy::math::DVec3) {
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

#[allow(clippy::too_many_lines)] // Incremental ownership and anchor refresh form one cutover.
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
    let editor_changed = runtime.synced_editor_revision != history.current_revision;
    let terrain_changed = foundation_edit_is_ready(
        runtime.terrain_acknowledgements,
        &runtime.pending_foundation_edit,
        runtime.foundation_edit_acknowledgement,
        runtime.terrain_edit_task.is_none()
            && runtime.pending_terrain_edits.is_empty()
            && runtime.last_brush_edit.is_none(),
    );
    let needs_initial_sync =
        runtime.known_world_parts.is_empty() && graph.0.parts().next().is_some();
    if !editor_changed && !terrain_changed && !needs_initial_sync {
        return;
    }
    diagnostics.foundation_candidate_count = 0;
    diagnostics.foundation_sample_count = 0;
    diagnostics.foundation_refresh_ms = 0.0;
    if editor_changed {
        runtime.synced_editor_revision = history.current_revision;
        let now = runtime.clock;
        runtime.autosave.mutate(now);
    }
    let current_parts = graph
        .0
        .parts()
        .map(|(part, spec)| (part, *spec))
        .collect::<BTreeMap<_, _>>();
    let construction_delta =
        ConstructionEditDelta::between_parts(&runtime.known_world_parts, &current_parts);
    let replaced_parts = construction_delta
        .removed
        .iter()
        .chain(&construction_delta.modified)
        .copied()
        .collect::<BTreeSet<_>>();
    let removed_foundation = runtime
        .foundations
        .iter()
        .any(|foundation| replaced_parts.contains(&foundation.part));
    for &part in &replaced_parts {
        runtime.foundation_index.remove(part);
    }
    runtime
        .foundations
        .retain(|foundation| !replaced_parts.contains(&foundation.part));
    let new_parts = construction_delta
        .added
        .iter()
        .chain(&construction_delta.modified)
        .copied()
        .collect::<Vec<_>>();
    let foundation_candidates = terrain_changed.then(|| {
        runtime
            .foundation_index
            .candidates(&runtime.pending_foundation_edit.changed_bricks)
    });
    let changed_bricks = runtime.pending_foundation_edit.changed_bricks.clone();
    let mut foundations = core::mem::take(&mut runtime.foundations);
    let mut foundation_index = core::mem::take(&mut runtime.foundation_index);

    let scene = ActiveTerrainScene {
        chunks: &runtime.active_terrain,
        ready_faces: &runtime.active_terrain_ready_faces,
        spatial_index: &runtime.active_terrain_index,
    };
    let mut added = 0_u64;
    for part in new_parts {
        let Some(spec) = graph.0.part(part).copied() else {
            continue;
        };
        let (minimum, maximum) = part_world_bounds(spec);
        let position = WorldPosition(
            runtime.floating_origin.0
                + Vec3::new(
                    (minimum.x + maximum.x) * 0.5,
                    minimum.y,
                    (minimum.z + maximum.z) * 0.5,
                )
                .as_dvec3(),
        );
        let support = FoundationSupport::rectangular(
            &scene,
            TerrainRayHit {
                position,
                normal: Vec3::Y,
                distance: 0.0,
                material_weights: [0.0; 3],
                chunk_generation: 0,
                triangle: 0,
            },
            f64::from(maximum.x - minimum.x),
            f64::from(maximum.z - minimum.z),
        );
        if support.has_valid_anchor() {
            foundation_index.insert(part, &support);
            foundations.push(TerrainFoundation { part, support });
            added = added.saturating_add(1);
        }
    }
    let mut detached = 0_u64;
    let mut anchors_changed = 0_u64;
    let refresh_started = std::time::Instant::now();
    if let Some(candidates) = &foundation_candidates {
        diagnostics.foundation_candidate_count =
            u64::try_from(candidates.len()).unwrap_or(u64::MAX);
        foundations.retain_mut(|foundation| {
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
                foundation_index.remove(foundation.part);
                false
            } else {
                true
            }
        });
        runtime.foundation_edit_acknowledgement = runtime.pending_foundation_edit.generation;
        runtime.pending_foundation_edit = TerrainEditBatch::default();
    }
    diagnostics.foundation_refresh_ms = refresh_started.elapsed().as_secs_f64() * 1_000.0;
    runtime.foundations = foundations;
    runtime.foundation_index = foundation_index;
    runtime.known_world_parts = current_parts;
    if added > 0 || anchors_changed > 0 || removed_foundation {
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
        .save_dirty_leaves(&runtime.document.name, &mut runtime.edits)
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn save_world_instance(
    runtime: &mut WorldRuntime,
    graph: &ConstructionGraph,
    editor: &EditorState,
) -> Result<(), String> {
    const INSTANCE_ID: u64 = 1;
    if graph.part_count() == 0 {
        runtime.autosave.saved();
        return Ok(());
    }
    if runtime
        .document
        .instances
        .iter()
        .all(|entry| entry.id != INSTANCE_ID)
    {
        runtime.document.instances.push(WorldInstanceIndexDoc {
            id: INSTANCE_ID,
            name: "World construction".to_owned(),
        });
        runtime
            .store
            .save_world(&runtime.document)
            .map_err(|error| error.to_string())?;
    }
    let instance = WorldCreationInstanceDoc {
        id: INSTANCE_ID,
        creation: crate::capture_creation(graph, editor, "World construction"),
        root_pose: WorldPoseDoc::default(),
        joint_coordinates: Vec::new(),
    };
    runtime
        .store
        .save_instance(&runtime.document.name, &instance)
        .map_err(|error| error.to_string())?;
    runtime.autosave.saved();
    Ok(())
}

fn terrain_chunk_mesh(chunk: &TerrainMeshChunk, indices: Vec<u32>) -> Mesh {
    let colors = chunk
        .material_weights
        .iter()
        .copied()
        .map(|weights| [weights[0], weights[1], weights[2], 1.0])
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
        .map(|position| {
            [
                (chunk.origin.0.y + f64::from(position[1])) as f32 / 1.5,
                0.0,
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
mod tests {
    use std::collections::BTreeSet;

    use bevy::{
        asset::RenderAssetUsages,
        camera::Exposure,
        math::DVec3,
        mesh::VertexAttributeValues,
        prelude::{App, Image, State},
        render::render_resource::{Extent3d, TextureDimension, TextureFormat},
    };
    use mechanic_world::{
        ActiveTerrainNode, BrickCoord, KinematicCapsule, TerrainEditBatch, TerrainFace,
        TerrainField, TerrainMeshChunk, TerrainMeshRequest, TerrainNodeId, TerrainOctree,
        TerrainReadiness, TerrainTransitionMask, WorldBounds, WorldPosition, WorldSeed, WorldStore,
        mesh_chunk, select_active_nodes,
    };

    use super::{
        AppSpace, TerrainAcknowledgements, WorldListPhase, WorldListState, WorldPrototypePlugin,
        exposure_for_space, foundation_edit_is_ready, full_rgba8_mip_byte_count,
        generate_rgba8_mip_chain, load_world_editor, nodes_touch_on_face, player_collision_nodes,
        ready_obsolete_nodes, terrain_chunk_has_collision_near, terrain_chunk_mesh,
        terrain_edit_commands, terrain_mesh_is_renderable,
    };
    use crate::{garage, showcase};

    struct TempWorldStore(std::path::PathBuf);

    impl TempWorldStore {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "mechanic-world-install-test-{}",
                std::process::id()
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

        editor = load_world_editor(&store, &document).unwrap();

        assert_eq!(editor.graph.part_count(), 0);
        assert!(editor.placed_bearings.is_empty());
    }

    #[test]
    fn brush_paths_preserve_every_five_centimetre_sample() {
        let start = WorldPosition(DVec3::new(-1.0, 2.0, 3.0));
        let end = WorldPosition(DVec3::new(-0.77, 2.0, 3.0));
        let commands = terrain_edit_commands(Some((start, 0.5)), end, 0.5);

        assert_eq!(commands.len(), 5);
        let mut previous = start;
        for command in &commands {
            assert_eq!(command.previous, Some((previous, 0.5)));
            assert!(previous.0.distance(command.centre.0) <= 0.05 + 1.0e-12);
            previous = command.centre;
        }
        assert_eq!(previous, end);
        assert!(terrain_edit_commands(Some((end, 0.5)), end, 0.5).is_empty());
        assert_eq!(terrain_edit_commands(None, end, 0.5).len(), 1);
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
    fn empty_terrain_chunks_are_not_published_to_bevys_mesh_allocator() {
        let mut chunk = TerrainMeshChunk::default();
        assert!(!terrain_mesh_is_renderable(&chunk, 0));

        chunk.vertices = vec![[0.0; 3]; 3];
        assert!(!terrain_mesh_is_renderable(&chunk, 0));
        assert!(terrain_mesh_is_renderable(&chunk, 3));
    }

    #[test]
    fn terrain_texture_coordinates_and_weights_are_chunk_seam_stable() {
        let chunk = TerrainMeshChunk {
            origin: WorldPosition(DVec3::new(15.0, 30.0, 45.0)),
            vertices: vec![[1.5, 3.0, 4.5]],
            normals: vec![[0.0, 1.0, 0.0]],
            material_weights: vec![[0.2, 0.3, 0.5]],
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
        assert_eq!(vertical, &[[22.0, 0.0]]);
        assert_eq!(weights, &[[0.2, 0.3, 0.5, 1.0]]);
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
            capsule,
        ));

        chunk.index_groups.regular = vec![0, 1, 2];
        assert!(terrain_chunk_has_collision_near(
            &chunk,
            TerrainTransitionMask::NONE,
            capsule,
        ));
        chunk.bounds.minimum.0.x = 20.0;
        chunk.bounds.maximum.0.x = 25.0;
        assert!(!terrain_chunk_has_collision_near(
            &chunk,
            TerrainTransitionMask::NONE,
            capsule,
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
            player_collision_nodes(&[far, local], capsule).collect::<Vec<_>>(),
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
        let pins = player_collision_nodes(&cut, capsule).collect::<Vec<_>>();
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
            terrain_chunk_has_collision_near(&chunk, TerrainTransitionMask::NONE, capsule)
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
}
