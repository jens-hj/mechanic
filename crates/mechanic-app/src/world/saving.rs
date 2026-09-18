//! Autosave, save on exit, and writing the garage and world instances.

use super::brush::finish_terrain_edits;
use super::{
    MessageReader, Res, ResMut, Result, String, Time, ToOwned, ToString, Vec, WorldRuntime, error,
};
use crate::editor::build_actions::PlacedBearing;
use crate::editor::state::{EditorGraph, EditorState};
use bevy::app::AppExit;
use mechanic_core::{BearingSocket, ConstructionGraph, CreationDocument};
use mechanic_world::{
    WorldCreationInstanceDoc, WorldInstanceIndexDoc, WorldPoseDoc, WorldPosition,
};

pub(super) fn autosave_world(
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

pub(super) fn save_on_exit(
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

pub(super) fn save_all(runtime: &mut WorldRuntime) -> Result<(), String> {
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

pub(super) fn save_world_instance(
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

pub(super) fn save_garage_instance(
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

pub(super) fn space_instance(
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
