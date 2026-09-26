//! Water surfaces: a quadtree of sheet tiles around the camera, meshed on
//! worker threads and drawn with the water material.
//!
//! Tiles are square and hold the same number of grid cells at every level,
//! so their spacing doubles with each level out from the camera. A sheet
//! runs on under the shore, where the terrain hides it, so tiles need no
//! seams: shorelines are where the terrain crosses the water at any detail.

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::math::DVec3;
use bevy::mesh::{Indices, MeshVertexAttribute, VertexAttributeValues};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, PrimitiveTopology, VertexFormat};
use bevy::shader::ShaderRef;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future};
use mechanic_world::{WaterSheet, WaterTile, water_sheet};

use super::{WorldOwned, WorldRuntime};
use crate::camera::MainCamera;

/// The water's vertex stage and full material.
const WATER_SHADER: &str = "shaders/water_material.wgsl";

/// Edge of the coarsest tiles, in metres.
const ROOT_EDGE_METRES: f64 = 2_048.0;

/// Levels below the root; the finest tiles are 64 m across.
const FINEST_LEVEL: u8 = 5;

/// Grid cells along each tile edge: 1 m apart in the finest tiles.
const TILE_CELLS: u32 = 64;

/// A tile splits while the camera is nearer than this many of its edges.
const SPLIT_DISTANCE_EDGES: f64 = 1.25;

/// Water is drawn out to this distance from the camera, in metres.
const WATER_REACH_METRES: f64 = 4_096.0;

/// Tiles meshing at once.
const TILES_IN_FLIGHT: usize = 8;

/// Depth of water under a vertex, then its surface current along x and z.
pub(crate) const ATTRIBUTE_WATER: MeshVertexAttribute =
    MeshVertexAttribute::new("Water", 0x6d65_6368_0010, VertexFormat::Float32x3);

/// Colours and scale of the water surface.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub(crate) struct WaterRenderMaterial {
    /// Linear colour of shallow water over pale ground.
    #[uniform(0)]
    pub(crate) shallow: LinearRgba,
    /// Linear colour of deep water.
    #[uniform(1)]
    pub(crate) deep: LinearRgba,
}

impl Default for WaterRenderMaterial {
    fn default() -> Self {
        Self {
            shallow: LinearRgba::rgb(0.05, 0.28, 0.3),
            deep: LinearRgba::rgb(0.005, 0.03, 0.08),
        }
    }
}

impl Material for WaterRenderMaterial {
    fn vertex_shader() -> ShaderRef {
        WATER_SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        WATER_SHADER.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        descriptor.vertex.buffers = vec![layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            ATTRIBUTE_WATER.at_shader_location(8),
        ])?];
        // Both faces: the surface is seen from below when swimming.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// One tile of the quadtree: its level and position among that level's tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TileKey {
    level: u8,
    x: i32,
    z: i32,
}

impl TileKey {
    fn edge(self) -> f64 {
        ROOT_EDGE_METRES / f64::from(1_u32 << self.level)
    }

    fn minimum(self) -> [f64; 2] {
        let edge = self.edge();
        [f64::from(self.x) * edge, f64::from(self.z) * edge]
    }

    fn distance_to(self, point: DVec3) -> f64 {
        let [x0, z0] = self.minimum();
        let edge = self.edge();
        let dx = (x0 - point.x).max(point.x - (x0 + edge)).max(0.0);
        let dz = (z0 - point.z).max(point.z - (z0 + edge)).max(0.0);
        dx.hypot(dz)
    }

    fn children(self) -> [Self; 4] {
        let (x, z, level) = (self.x * 2, self.z * 2, self.level + 1);
        [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(dx, dz)| Self {
            level,
            x: x + dx,
            z: z + dz,
        })
    }

    fn tile(self) -> WaterTile {
        WaterTile {
            minimum: self.minimum(),
            edge: self.edge(),
            cells: TILE_CELLS,
        }
    }
}

enum TileState {
    Meshing(Task<Option<WaterSheet>>),
    Shown(Option<Entity>),
}

/// The water tiles around the camera.
#[derive(Resource, Default)]
pub(crate) struct WaterTiles {
    material: Option<Handle<WaterRenderMaterial>>,
    tiles: HashMap<TileKey, TileState>,
    /// Floating origin the shown tiles are placed against.
    origin: Option<DVec3>,
}

/// Forgets every tile; the entities go with the world.
pub(crate) fn clear_water_tiles(mut tiles: ResMut<WaterTiles>) {
    *tiles = WaterTiles::default();
}

/// The tiles that cover the water around `camera`, finest nearest.
fn wanted_tiles(camera: DVec3) -> Vec<TileKey> {
    let lowest = |value: f64| ((value - WATER_REACH_METRES) / ROOT_EDGE_METRES).floor() as i32;
    let highest = |value: f64| ((value + WATER_REACH_METRES) / ROOT_EDGE_METRES).floor() as i32;
    let mut stack = Vec::new();
    for z in lowest(camera.z)..=highest(camera.z) {
        for x in lowest(camera.x)..=highest(camera.x) {
            stack.push(TileKey { level: 0, x, z });
        }
    }
    let mut wanted = Vec::new();
    while let Some(key) = stack.pop() {
        let distance = key.distance_to(camera);
        if distance > WATER_REACH_METRES {
            continue;
        }
        if key.level < FINEST_LEVEL && distance < key.edge() * SPLIT_DISTANCE_EDGES {
            stack.extend(key.children());
        } else {
            wanted.push(key);
        }
    }
    wanted.sort_by(|first, second| {
        first
            .distance_to(camera)
            .total_cmp(&second.distance_to(camera))
    });
    wanted
}

fn water_mesh(sheet: WaterSheet) -> Mesh {
    let attributes = sheet
        .depths
        .iter()
        .zip(&sheet.flows)
        .map(|(depth, flow)| [*depth, flow[0], flow[1]])
        .collect::<Vec<_>>();
    let normals = vec![[0.0, 1.0, 0.0]; sheet.vertices.len()];
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, sheet.vertices);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(
        ATTRIBUTE_WATER,
        VertexAttributeValues::Float32x3(attributes),
    );
    mesh.insert_indices(Indices::U32(sheet.indices));
    mesh
}

/// Keeps the tiles around the camera meshed and shown. A tile the camera
/// has left stays until every tile that replaces it is ready, so the water
/// never flickers out while detail changes.
pub(crate) fn stream_water(
    mut commands: Commands,
    mut tiles: ResMut<WaterTiles>,
    runtime: Res<WorldRuntime>,
    camera: Query<&GlobalTransform, With<MainCamera>>,
    mut meshes: ResMut<Assets<Mesh>>,
    materials: Option<ResMut<Assets<WaterRenderMaterial>>>,
) {
    let Some(mut materials) = materials else {
        return;
    };
    if crate::env::text(crate::env::WATER).as_deref() == Some("off") {
        return;
    }
    let Ok(camera) = camera.single() else {
        return;
    };
    let origin = runtime.floating_origin.0;
    if tiles.origin != Some(origin) {
        for state in tiles.tiles.values() {
            if let TileState::Shown(Some(entity)) = state {
                commands.entity(*entity).despawn();
            }
        }
        tiles.tiles.clear();
        tiles.origin = Some(origin);
    }
    let material = tiles
        .material
        .get_or_insert_with(|| materials.add(WaterRenderMaterial::default()))
        .clone();
    let camera = origin + camera.translation().as_dvec3();
    let wanted = wanted_tiles(camera);

    let mut in_flight = tiles
        .tiles
        .values()
        .filter(|state| matches!(state, TileState::Meshing(_)))
        .count();
    for key in &wanted {
        if in_flight >= TILES_IN_FLIGHT {
            break;
        }
        if tiles.tiles.contains_key(key) {
            continue;
        }
        let field = runtime.field.clone();
        let edits = runtime.edits.snapshot();
        let tile = key.tile();
        let task =
            AsyncComputeTaskPool::get().spawn(async move { water_sheet(&field, &edits, tile) });
        tiles.tiles.insert(*key, TileState::Meshing(task));
        in_flight += 1;
    }

    for (key, state) in &mut tiles.tiles {
        let TileState::Meshing(task) = state else {
            continue;
        };
        let Some(sheet) = block_on(future::poll_once(task)) else {
            continue;
        };
        let entity = sheet.map(|sheet| {
            let translation = (sheet.origin.0 - origin).as_vec3();
            commands
                .spawn((
                    Name::new(format!("Water tile {}:{},{}", key.level, key.x, key.z)),
                    Mesh3d(meshes.add(water_mesh(sheet))),
                    MeshMaterial3d(material.clone()),
                    Transform::from_translation(translation),
                    // Shadow cascades cost more than the rest of the
                    // water's shading together, over half the screen.
                    bevy::light::NotShadowReceiver,
                    bevy::light::NotShadowCaster,
                    WorldOwned,
                ))
                .id()
        });
        *state = TileState::Shown(entity);
    }

    let settled = wanted
        .iter()
        .all(|key| matches!(tiles.tiles.get(key), Some(TileState::Shown(_))));
    if settled {
        let wanted = wanted.into_iter().collect::<std::collections::HashSet<_>>();
        tiles.tiles.retain(|key, state| {
            let keep = wanted.contains(key);
            if !keep && let TileState::Shown(Some(entity)) = state {
                commands.entity(*entity).despawn();
            }
            keep
        });
    }
}

#[cfg(test)]
mod tests {
    use bevy::math::DVec3;

    use super::{FINEST_LEVEL, TileKey, WATER_REACH_METRES, wanted_tiles};

    #[test]
    fn tiles_cover_the_reach_once_with_the_finest_nearest() {
        let camera = DVec3::new(130.0, 5.0, -70.0);
        let wanted = wanted_tiles(camera);
        let nearest = wanted[0];
        assert_eq!(nearest.level, FINEST_LEVEL);
        assert!(nearest.distance_to(camera) <= 0.0);
        // Sample points in the reach are each covered by exactly one tile.
        for step in 0..64 {
            let angle = f64::from(step) * 0.7;
            let radius = f64::from(step) / 64.0 * WATER_REACH_METRES * 0.9;
            let point = camera + DVec3::new(angle.cos() * radius, 0.0, angle.sin() * radius);
            let covering = wanted
                .iter()
                .filter(|key: &&TileKey| key.distance_to(point) <= 0.0 && inside(**key, point))
                .count();
            assert_eq!(covering, 1, "point {point} is covered {covering} times");
        }
    }

    fn inside(key: TileKey, point: DVec3) -> bool {
        let [x0, z0] = key.minimum();
        (x0..x0 + key.edge()).contains(&point.x) && (z0..z0 + key.edge()).contains(&point.z)
    }
}
