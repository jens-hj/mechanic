//! Entering and leaving the world, and the editors each space keeps.

use super::list::{WorldListPhase, WorldListState};
use super::saving::{save_all, save_garage_instance, save_world_instance};
use super::terrain_render::{TerrainRenderMaterial, spawn_world_terrain};
use super::transfer::{TransferAttempt, transfer_active_assembly};
use super::walking::reset_player_collision_publication;
use super::{
    AppSpace, AssetServer, Assets, ButtonInput, ClearColor, Color, Commands, DistanceFog, Entity,
    FogFalloff, Mesh, Name, NextState, Query, Res, ResMut, Result, Single, SpaceEditorState,
    StandardMaterial, State, String, ToOwned, ToString, Visibility, With, Without,
    WorldDiagnostics, WorldOwned, WorldRuntime, default, exposure_for_space, format,
};
use crate::camera::{MainCamera, PlayerState};
use crate::controls::GameAction;
use crate::editor::build_actions::PlacedBearing;
use crate::editor::history::EditorHistory;
use crate::editor::state::{EditorGraph, EditorState};
use crate::garage;
use crate::simulation::state::AppSimulation;
use bevy::camera::Exposure;
use mechanic_core::{DimensionLinkId, PartSpec};
use mechanic_world::{
    FloatingOrigin, KinematicCapsule, TerrainReadiness, TerrainSpatialIndex, TerrainStreamer,
    WorldCreationInstanceDoc, WorldDocument, WorldPosition, WorldStore,
};
use std::collections::BTreeMap;

pub(super) fn editor_from_instance(
    instance: WorldCreationInstanceDoc,
) -> Result<SpaceEditorState, String> {
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

pub(super) fn load_space_editors(
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

pub(super) fn application_world_store() -> WorldStore {
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

pub(super) fn restore_initial_garage(
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

#[expect(clippy::too_many_arguments)]
pub(super) fn toggle_space(
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

#[expect(clippy::too_many_arguments)]
pub(super) fn enter_world(
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
    crate::editor::history::cancel_transient_editor_state(&mut graph.0, &mut editor);
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

pub(super) fn restore_world_player(
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

#[expect(clippy::too_many_arguments)]
pub(super) fn leave_world(
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
    crate::editor::history::cancel_transient_editor_state(&mut graph.0, &mut editor);
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
