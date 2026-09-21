//! The world list: choosing, creating, and opening saved worlds.

use super::brush::finish_terrain_edits;
use super::saving::{save_all, save_garage_instance, save_world_instance};
use super::space::load_space_editors;
use super::streaming::TerrainAcknowledgements;
use super::walking::reset_player_collision_publication;
use super::{
    AppSpace, FromWorld, NextState, Res, ResMut, Resource, Result, State, String, ToOwned,
    ToString, Vec, World, WorldRuntime, format,
};
use crate::editor::state::{EditorGraph, EditorState};
use crate::ui::WorldAction;
use mechanic_core::ConstructionGraph;
use mechanic_world::{
    AutosaveState, FoundationSpatialIndex, KinematicCapsule, OpenWorldOutcome, SavedWorld,
    TerrainBoundsCache, TerrainEditBatch, TerrainField, TerrainMaterial, TerrainReadiness,
    TerrainSpatialIndex, TerrainStreamer, WorldDocument, WorldStore,
};
use std::sync::Arc;

/// State backing the full-window Mosaic world list.
#[derive(Resource)]
pub(crate) struct WorldListState {
    pub(super) phase: WorldListPhase,
    pub(super) entries: Vec<SavedWorld>,
    pub(super) notice: Option<String>,
    pub(super) loading_progress: TerrainReadiness,
    pub(super) confirming_delete: Option<std::path::PathBuf>,
    pub(super) requested: Option<WorldAction>,
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

    pub(super) fn refresh(&mut self, store: &WorldStore) {
        self.entries = store.list();
        self.confirming_delete = None;
    }
}

pub(super) fn world_list_closed(list: Res<WorldListState>) -> bool {
    !list.is_open()
}

pub(super) fn handle_world_list(
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
                Ok(OpenWorldOutcome::Opened(document)) => {
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
                Ok(OpenWorldOutcome::OutdatedRemoved { path }) => {
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

pub(super) fn save_before_world_selector(
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

pub(super) fn install_world(
    runtime: &mut WorldRuntime,
    document: WorldDocument,
) -> Result<(), String> {
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
    runtime.spoil.reset();
    runtime.slump = mechanic_world::SpoilSlump::default();
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
