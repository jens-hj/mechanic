//! Authoring hot reload: with `MECHANIC_WORLDGEN_DIR` set, the world is
//! generated from that directory's RON files and regenerates whenever they
//! change. Saved edits are kept as they are, so an edited region may no
//! longer match the new ground around it; the save itself is never touched.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use bevy::prelude::*;
use bevy::render::storage::ShaderBuffer;
use mechanic_world::{
    TerrainBoundsCache, TerrainField, TerrainSpatialIndex, TerrainStreamer, WorldSeed, WorldgenSpec,
};

use super::WorldRuntime;
use super::terrain_render::{TerrainRenderMaterial, refresh_terrain_surfaces};

const CHECK_INTERVAL: Duration = Duration::from_millis(500);

/// The authored definition directory and when it last changed.
#[derive(Resource, Default)]
pub(crate) struct WorldgenWatch {
    stamp: Option<SystemTime>,
    since_check: Duration,
}

fn directory() -> Option<PathBuf> {
    crate::env::path(crate::env::WORLDGEN_DIR)
}

/// The field for a world: from the authored directory when one is set,
/// otherwise the definition compiled into the build.
pub(super) fn world_field(seed: WorldSeed) -> Result<TerrainField, String> {
    let Some(directory) = directory() else {
        return Ok(TerrainField::new(seed));
    };
    let spec = WorldgenSpec::from_dir(&directory).map_err(|error| error.to_string())?;
    TerrainField::from_spec(seed, Arc::new(spec)).map_err(|error| error.to_string())
}

/// Latest modification time of any file under `directory`.
fn newest_change(directory: &Path) -> Option<SystemTime> {
    let mut newest = None;
    let mut pending = vec![directory.to_owned()];
    while let Some(folder) = pending.pop() {
        for entry in std::fs::read_dir(&folder).ok()?.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if let Ok(modified) = entry.metadata().and_then(|meta| meta.modified()) {
                newest = Some(newest.map_or(modified, |known: SystemTime| known.max(modified)));
            }
        }
    }
    newest
}

pub(super) fn reload_worldgen(
    time: Res<Time>,
    mut watch: ResMut<WorldgenWatch>,
    mut runtime: ResMut<WorldRuntime>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut surface_buffers: ResMut<Assets<ShaderBuffer>>,
    materials: Res<Assets<TerrainRenderMaterial>>,
) {
    let Some(directory) = directory() else {
        return;
    };
    watch.since_check += time.delta();
    if watch.since_check < CHECK_INTERVAL {
        return;
    }
    watch.since_check = Duration::ZERO;
    let stamp = newest_change(&directory);
    if watch.stamp.is_none() {
        watch.stamp = stamp;
        return;
    }
    if stamp == watch.stamp {
        return;
    }
    watch.stamp = stamp;
    let field = match world_field(runtime.document.seed) {
        Ok(field) => field,
        Err(error) => {
            warn!("worldgen reload failed: {error}");
            runtime.load_error = Some(format!("worldgen: {error}"));
            return;
        }
    };
    info!("worldgen reloaded from {}", directory.display());
    runtime.load_error = None;
    runtime.field = Arc::new(field);
    for (_, entity) in std::mem::take(&mut runtime.terrain_entities) {
        commands.entity(entity).despawn();
    }
    for (_, handle) in std::mem::take(&mut runtime.terrain_mesh_handles) {
        meshes.remove(handle.id());
    }
    runtime.terrain_streamer = TerrainStreamer::default();
    runtime.terrain_bounds_cache = TerrainBoundsCache::default();
    runtime.terrain_selection_task = None;
    runtime.staged_terrain.clear();
    runtime.terrain_cutovers = super::streaming::TerrainCutovers::default();
    runtime.active_terrain.clear();
    runtime.active_terrain_ready_faces.clear();
    runtime.active_terrain_index = TerrainSpatialIndex::default();
    runtime.selected_terrain_revision = u64::MAX;
    runtime.selection_focus = None;
    let luma = runtime.terrain_layer_luma;
    refresh_terrain_surfaces(&mut surface_buffers, &materials, &runtime, &luma);
}
