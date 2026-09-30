//! Water surfaces: a quadtree of sheet tiles around the camera, meshed on
//! worker threads and drawn with the water material.
//!
//! Tiles are square and hold the same number of grid cells at every level,
//! so their spacing doubles with each level out from the camera. A sheet
//! runs on under the shore, where the terrain hides it, so tiles need no
//! seams: shorelines are where the terrain crosses the water at any detail.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::math::DVec3;
use bevy::mesh::{Indices, MeshVertexAttribute, VertexAttributeValues};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, PrimitiveTopology, VertexFormat};
use bevy::shader::ShaderRef;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future};
use mechanic_world::{
    WATER_CELL_METRES, WaterBody, WaterCell, WaterSheet, WaterShift, WaterSurface, WaterTile,
    joined_water_sheet,
};

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

    /// Whether the tile's grid reaches a column: its own square and one
    /// grid cell around it.
    fn reaches(self, [x, z]: [f64; 2]) -> bool {
        let [x0, z0] = self.minimum();
        let (edge, margin) = (self.edge(), self.edge() / f64::from(TILE_CELLS));
        (x0 - margin..=x0 + edge + margin).contains(&x)
            && (z0 - margin..=z0 + edge + margin).contains(&z)
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
    /// Shown, and meshing again since the water under it changed; the old
    /// mesh stays until the new one is ready.
    Refreshing(Option<Entity>, Task<Option<WaterSheet>>),
}

impl TileState {
    fn shown(&self) -> Option<Entity> {
        match self {
            Self::Shown(entity) | Self::Refreshing(entity, _) => *entity,
            Self::Meshing(_) => None,
        }
    }
}

/// The water tiles around the camera.
#[derive(Resource, Default)]
pub(crate) struct WaterTiles {
    material: Option<Handle<WaterRenderMaterial>>,
    tiles: HashMap<TileKey, TileState>,
    /// Floating origin the shown tiles are placed against.
    origin: Option<DVec3>,
    /// Where each moved lake and river stood when the tiles were meshed.
    shifts: std::collections::BTreeMap<WaterBody, WaterShift>,
    /// Each tile of the stored water's surface: its entity, what it shows,
    /// and where it is placed from.
    surface: HashMap<(i32, i32), (Entity, u64, DVec3)>,
    /// Floating origin the surface tiles are placed against.
    surface_origin: DVec3,
    /// The wetness map's world corner and base height, once drawn.
    wet_window: Option<DVec3>,
    /// Floating origin the wetness map was placed against.
    wet_origin: DVec3,
    /// Water revision the wetness map was drawn at.
    wet_revision: Option<u64>,
    /// Water revision the stored water was drawn at.
    drawn_revision: Option<u64>,
    /// Cells joined to seed-derived water, which the tiles draw as that
    /// water.
    joined_cells: Arc<HashSet<WaterCell>>,
    /// Water-cell columns the stored water's surface draws within a lake's
    /// reach, which the tiles leave out.
    owned_columns: Arc<HashSet<(i32, i32)>>,
    /// Columns whose joined cells or stored water changed since the tiles
    /// over them were meshed.
    stale: Vec<[f64; 2]>,
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
    if !super::water::water_enabled() {
        return;
    }
    let Ok(camera) = camera.single() else {
        return;
    };
    let origin = runtime.floating_origin.0;
    // Lakes and rivers are drawn where the stored water meeting them is,
    // which moves them only once they move 2 cm.
    let shifts = &runtime.water.view().shifts;
    if *shifts != tiles.shifts {
        tiles.shifts = shifts.clone();
        tiles.origin = None;
    }
    if tiles.origin != Some(origin) {
        for state in tiles.tiles.values() {
            if let Some(entity) = state.shown() {
                commands.entity(entity).despawn();
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
    let mesh = |key: TileKey, tiles: &WaterTiles| {
        let field = runtime.field.clone();
        let edits = runtime.edits.snapshot();
        let (shifts, joined, owned) = (
            tiles.shifts.clone(),
            tiles.joined_cells.clone(),
            tiles.owned_columns.clone(),
        );
        AsyncComputeTaskPool::get().spawn(async move {
            joined_water_sheet(&field, &edits, key.tile(), &shifts, &joined, &owned)
        })
    };
    refresh_stale(&mut tiles, mesh);

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
        let task = mesh(*key, &tiles);
        tiles.tiles.insert(*key, TileState::Meshing(task));
        in_flight += 1;
    }

    for (key, state) in &mut tiles.tiles {
        let (TileState::Meshing(task) | TileState::Refreshing(_, task)) = state else {
            continue;
        };
        let Some(sheet) = block_on(future::poll_once(task)) else {
            continue;
        };
        if let Some(old) = state.shown() {
            commands.entity(old).despawn();
        }
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

    retire_tiles(&mut commands, &mut tiles, wanted);
}

/// Once every wanted tile shows, drops the tiles no longer wanted.
fn retire_tiles(commands: &mut Commands, tiles: &mut WaterTiles, wanted: Vec<TileKey>) {
    let settled = wanted.iter().all(|key| {
        matches!(
            tiles.tiles.get(key),
            Some(TileState::Shown(_) | TileState::Refreshing(..))
        )
    });
    if settled {
        let wanted = wanted.into_iter().collect::<HashSet<_>>();
        tiles.tiles.retain(|key, state| {
            let keep = wanted.contains(key);
            if !keep && let Some(entity) = state.shown() {
                commands.entity(entity).despawn();
            }
            keep
        });
    }
}

/// Meshes again the tiles over water that joined or left seed-derived water
/// since they were meshed, showing the old mesh until the new one is ready.
fn refresh_stale(
    tiles: &mut WaterTiles,
    mesh: impl Fn(TileKey, &WaterTiles) -> Task<Option<WaterSheet>>,
) {
    let stale = std::mem::take(&mut tiles.stale);
    if stale.is_empty() {
        return;
    }
    let keys = tiles
        .tiles
        .keys()
        .copied()
        .filter(|key| stale.iter().any(|&column| key.reaches(column)))
        .collect::<Vec<_>>();
    for key in keys {
        let task = mesh(key, tiles);
        if let Some(old) = tiles.tiles.remove(&key) {
            let refreshed = match old {
                TileState::Meshing(_) => TileState::Meshing(task),
                TileState::Shown(entity) | TileState::Refreshing(entity, _) => {
                    TileState::Refreshing(entity, task)
                }
            };
            tiles.tiles.insert(key, refreshed);
        }
    }
}

fn surface_mesh(
    positions: Vec<[f32; 3]>,
    normals: Option<Vec<[f32; 3]>>,
    attributes: Vec<[f32; 3]>,
    indices: Vec<u32>,
) -> Mesh {
    let normals = normals.unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(
        ATTRIBUTE_WATER,
        VertexAttributeValues::Float32x3(attributes),
    );
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Draws the stored water's surface after each water batch: a surface tile is meshed again only when what it shows
/// changes, and follows the floating origin.
pub(crate) fn draw_stored_water(
    mut commands: Commands,
    mut tiles: ResMut<WaterTiles>,
    runtime: Res<WorldRuntime>,
    mut meshes: ResMut<Assets<Mesh>>,
    materials: Option<ResMut<Assets<WaterRenderMaterial>>>,
    mut placed: Query<&mut Transform>,
) {
    let Some(mut materials) = materials else {
        return;
    };
    if !super::water::water_enabled() {
        return;
    }
    let origin = runtime.floating_origin.0;
    if tiles.surface_origin != origin {
        tiles.surface_origin = origin;
        for &(entity, _, from) in tiles.surface.values() {
            if let Ok(mut transform) = placed.get_mut(entity) {
                transform.translation = (from - origin).as_vec3();
            }
        }
    }
    if tiles.drawn_revision == Some(runtime.water_revision) {
        return;
    }
    tiles.drawn_revision = Some(runtime.water_revision);
    let material = tiles
        .material
        .get_or_insert_with(|| materials.add(WaterRenderMaterial::default()))
        .clone();
    let spawn = |commands: &mut Commands,
                 meshes: &mut Assets<Mesh>,
                 name: String,
                 mesh: Mesh,
                 translation: Vec3| {
        commands
            .spawn((
                Name::new(name),
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(material.clone()),
                Transform::from_translation(translation),
                // Shadow cascades cost more than the rest of the water's
                // shading together.
                bevy::light::NotShadowReceiver,
                bevy::light::NotShadowCaster,
                WorldOwned,
            ))
            .id()
    };
    let view = runtime.water.view();
    let mut seen = HashSet::with_capacity(view.surface.len());
    for tile in &view.surface {
        seen.insert(tile.key);
        if tile.positions.is_empty() {
            // Unchanged since it was last meshed.
            continue;
        }
        if let Some((entity, ..)) = tiles.surface.remove(&tile.key) {
            commands.entity(entity).despawn();
        }
        let mesh = surface_mesh(
            tile.positions.clone(),
            Some(tile.normals.clone()),
            tile.attributes.clone(),
            tile.indices.clone(),
        );
        let entity = spawn(
            &mut commands,
            &mut meshes,
            format!("Stored water {},{}", tile.key.0, tile.key.1),
            mesh,
            (tile.origin - origin).as_vec3(),
        );
        tiles
            .surface
            .insert(tile.key, (entity, tile.fingerprint, tile.origin));
    }
    tiles.surface.retain(|key, (entity, ..)| {
        let keep = seen.contains(key);
        if !keep {
            commands.entity(*entity).despawn();
        }
        keep
    });
    note_joined(&mut tiles, &view.joined, &view.owned);
}

/// Texels along one edge of the wetness map: 20 cm each, 51.2 m in all.
const WET_TEXELS: u32 = 256;

/// How far the camera may stray from the wetness map's centre before the
/// map moves with it, in metres.
const WET_RECENTRE_METRES: f64 = 10.0;

/// A wetness map: texels of fill, then ground height, `edge` along a side.
pub(crate) fn wetness_image(edge: u32, texels: Vec<f32>) -> Image {
    let data = texels
        .chunks(2)
        .flat_map(|texel| [texel[0], texel[1], 0.0, 1.0])
        .flat_map(|value| half_bits(value).to_le_bytes())
        .collect::<Vec<_>>();
    let mut image = Image::new(
        bevy::render::render_resource::Extent3d {
            width: edge,
            height: edge,
            depth_or_array_layers: 1,
        },
        bevy::render::render_resource::TextureDimension::D2,
        data,
        bevy::render::render_resource::TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = bevy::image::ImageSampler::linear();
    image
}

/// A float's bits as a half float, rounded towards zero, for the wetness
/// map: fills from 0 to 1 and heights of a few metres.
fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = u16::try_from((bits >> 16) & 0x8000).expect("one bit");
    let exponent = i32::try_from((bits >> 23) & 0xff).expect("eight bits") - 127 + 15;
    if exponent <= 0 {
        return sign;
    }
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    let exponent = u16::try_from(exponent).expect("under 31");
    let mantissa = u16::try_from((bits >> 13) & 0x03ff).expect("ten bits");
    sign | (exponent << 10) | mantissa
}

/// Draws how wet the ground is around the camera into the terrain's
/// wetness map, after each water batch, moving the map with the camera.
pub(crate) fn draw_wet_ground(
    mut tiles: ResMut<WaterTiles>,
    runtime: Res<WorldRuntime>,
    camera: Query<&GlobalTransform, With<MainCamera>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<super::terrain_render::TerrainRenderMaterial>>,
) {
    let (Some(handle), Ok(camera)) = (runtime.terrain_material.as_ref(), camera.single()) else {
        return;
    };
    let origin = runtime.floating_origin.0;
    let camera = origin + camera.translation().as_dvec3();
    let size = f64::from(WET_TEXELS) * WATER_CELL_METRES;
    let moved = tiles.wet_window.is_none_or(|window| {
        (window.x + 0.5 * size - camera.x).abs() > WET_RECENTRE_METRES
            || (window.z + 0.5 * size - camera.z).abs() > WET_RECENTRE_METRES
            || (window.y - camera.y).abs() > WET_RECENTRE_METRES
    });
    if !moved && tiles.wet_origin == origin && tiles.wet_revision == Some(runtime.water_revision) {
        return;
    }
    tiles.wet_revision = Some(runtime.water_revision);
    tiles.wet_origin = origin;
    if moved {
        let snap = |value: f64| (value / WATER_CELL_METRES).round() * WATER_CELL_METRES;
        tiles.wet_window = Some(DVec3::new(
            snap(camera.x - 0.5 * size),
            camera.y,
            snap(camera.z - 0.5 * size),
        ));
    }
    let Some(window) = tiles.wet_window else {
        return;
    };
    let edge = WET_TEXELS as usize;
    let mut texels = vec![0.0_f32; edge * edge * 2];
    #[expect(clippy::cast_possible_truncation, reason = "texels within the map")]
    let first = (
        (window.x / WATER_CELL_METRES).round() as i32,
        (window.z / WATER_CELL_METRES).round() as i32,
    );
    for wet in &runtime.water.view().wet {
        let (x, z) = (wet.column.0 - first.0, wet.column.1 - first.1);
        let (Ok(x), Ok(z)) = (usize::try_from(x), usize::try_from(z)) else {
            continue;
        };
        if x >= edge || z >= edge {
            continue;
        }
        let texel = (x + z * edge) * 2;
        // A few millimetres soaked in already darken the ground.
        #[expect(clippy::cast_possible_truncation, reason = "shader data is f32")]
        {
            texels[texel] = (1.0 - (-wet.soaked / 0.003).exp()) as f32;
            texels[texel + 1] = (wet.top - window.y) as f32;
        }
    }
    let Some(material) = materials.get(handle) else {
        return;
    };
    // The map keeps one image, written again in place, so the terrain's
    // material only changes when the map moves.
    let image = material.wetness.clone();
    let fresh = wetness_image(WET_TEXELS, texels);
    let sized = images
        .get(&image)
        .is_some_and(|current| current.texture_descriptor.size.width == WET_TEXELS);
    if sized && let Some(mut current) = images.get_mut(&image) {
        current.data = fresh.data;
    } else {
        let image = images.add(fresh);
        if let Some(mut material) = materials.get_mut(handle) {
            material.wetness = image;
        }
    }
    // The map's texels are columns, centred half a texel in.
    let corner = window - origin;
    #[expect(clippy::cast_possible_truncation, reason = "shader data is f32")]
    let placed = Vec4::new(
        corner.x as f32,
        corner.z as f32,
        size as f32,
        (window.y - origin.y) as f32,
    );
    if materials
        .get(handle)
        .is_some_and(|material| material.wet_window != placed)
        && let Some(mut material) = materials.get_mut(handle)
    {
        material.wet_window = placed;
    }
}

/// Keeps the cells joined to seed-derived water and the columns the stored
/// water draws within its reach, and marks the lake tiles over any that came
/// or went to be meshed again.
fn note_joined(tiles: &mut WaterTiles, cells: &[(WaterCell, WaterSurface)], owned: &[(i32, i32)]) {
    let set = cells.iter().map(|(cell, _)| *cell).collect::<HashSet<_>>();
    if set != *tiles.joined_cells {
        let stale = set
            .symmetric_difference(&tiles.joined_cells)
            .map(|cell| [cell.centre().x, cell.centre().z])
            .collect::<Vec<_>>();
        tiles.stale.extend(stale);
        tiles.joined_cells = Arc::new(set);
    }
    let owned = owned.iter().copied().collect::<HashSet<_>>();
    if owned != *tiles.owned_columns {
        let stale = owned
            .symmetric_difference(&tiles.owned_columns)
            .map(|&(x, z)| {
                let centre = WaterCell::new(x, 0, z).centre();
                [centre.x, centre.z]
            })
            .collect::<Vec<_>>();
        tiles.stale.extend(stale);
        tiles.owned_columns = Arc::new(owned);
    }
}

#[cfg(test)]
mod tests {
    use bevy::math::DVec3;

    use super::{FINEST_LEVEL, TileKey, WATER_REACH_METRES, half_bits, wanted_tiles};

    #[test]
    fn wetness_values_become_half_floats() {
        assert_eq!(half_bits(1.0), 0x3c00);
        assert_eq!(half_bits(0.5), 0x3800);
        assert_eq!(half_bits(-2.0), 0xc000);
        assert_eq!(half_bits(0.0), 0);
    }

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
