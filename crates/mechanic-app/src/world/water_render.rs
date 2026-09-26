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
use mechanic_world::{
    PoolView, TerrainField, WATER_CELL_METRES, WaterBody, WaterCell, WaterFall, WaterSheet,
    WaterShift, WaterSurface, WaterTile, water_sheet,
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
    /// Where each moved lake and river stood when the tiles were meshed.
    shifts: std::collections::BTreeMap<WaterBody, WaterShift>,
    /// Each stored pool's entity, and the level and cells it was drawn at.
    pools: HashMap<u32, (Entity, f64, usize)>,
    /// The falling streams' entity.
    falls: Option<Entity>,
    /// Cells joined to seed-derived water: their entity, and a fingerprint of
    /// what it shows.
    joined: Option<(Entity, (usize, u64))>,
    /// Water revision the pools were drawn at.
    drawn_revision: Option<u64>,
}

/// A lake that has dropped this much further than its tiles show is meshed
/// again, in metres.
const REMESH_DROP_METRES: f64 = 0.02;

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
    let shifts = runtime.water_surfaces.shifts();
    let drop = |shifts: &std::collections::BTreeMap<WaterBody, WaterShift>, body| {
        shifts
            .get(body)
            .map_or(0.0, |shift: &WaterShift| shift.drop)
    };
    let moved = shifts
        .keys()
        .chain(tiles.shifts.keys())
        .any(|body| (drop(shifts, body) - drop(&tiles.shifts, body)).abs() > REMESH_DROP_METRES);
    if moved {
        tiles.shifts = shifts.clone();
        tiles.origin = None;
    }
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
        let shifts = tiles.shifts.clone();
        let task = AsyncComputeTaskPool::get()
            .spawn(async move { water_sheet(&field, &edits, tile, &shifts) });
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

/// One column of stored water to draw: its top cell, surface and depth.
struct SurfaceColumn {
    cell: WaterCell,
    level: f64,
    depth: f64,
}

/// The columns of a pool whose surface lies open to the air: a pool filling
/// under a lake has no surface of its own.
fn open_columns(pool: &PoolView, field: &TerrainField) -> Vec<SurfaceColumn> {
    pool.surface_cells
        .iter()
        .zip(&pool.depths)
        .filter(|(cell, _)| {
            let centre = cell.centre();
            !field.is_water(DVec3::new(centre.x, pool.level + 0.05, centre.z))
        })
        .map(|(&cell, &depth)| SurfaceColumn {
            cell,
            // A sealed pool pressed higher than its cells shows at their top.
            level: pool.level.min(cell.bottom() + WATER_CELL_METRES),
            depth,
        })
        .collect()
}

/// The columns of cells that joined seed-derived water where that water's
/// own sheet does not show: ground dug beside a lake, not under it.
fn joined_columns(
    joined: &[(WaterCell, WaterSurface)],
    field: &TerrainField,
) -> Vec<SurfaceColumn> {
    let mut tops = std::collections::BTreeMap::<(i32, i32), (WaterCell, f64)>::new();
    for &(cell, surface) in joined {
        if cell.bottom() >= surface.level {
            continue;
        }
        let top = tops
            .entry((cell.x, cell.z))
            .or_insert((cell, surface.level));
        if cell.y > top.0.y {
            *top = (cell, surface.level);
        }
    }
    tops.into_values()
        .filter(|&(cell, level)| {
            let centre = cell.centre();
            !field.is_water(DVec3::new(centre.x, level - 0.02, centre.z))
        })
        .map(|(cell, level)| SurfaceColumn {
            cell,
            level,
            depth: 1.0,
        })
        .collect()
}

/// What a set of columns shows, to the millimetre: drawn again only when
/// this changes.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a millimetre fingerprint of the drawn levels"
)]
fn fingerprint(columns: &[SurfaceColumn]) -> (usize, u64) {
    (
        columns.len(),
        columns
            .iter()
            .map(|column| (column.level * 1000.0).round() as i64 as u64)
            .fold(0_u64, u64::wrapping_add),
    )
}

/// A flat quad per column of stored water, placed against `origin`.
fn columns_mesh(columns: &[SurfaceColumn], origin: DVec3) -> Mesh {
    let edge = WATER_CELL_METRES;
    let mut positions = Vec::with_capacity(columns.len() * 4);
    let mut attributes = Vec::with_capacity(columns.len() * 4);
    let mut indices = Vec::with_capacity(columns.len() * 6);
    for &SurfaceColumn { cell, level, depth } in columns {
        let base = u32::try_from(positions.len()).expect("a pool mesh fits u32 indices");
        let corner = DVec3::new(f64::from(cell.x) * edge, level, f64::from(cell.z) * edge);
        for offset in [
            DVec3::ZERO,
            DVec3::new(0.0, 0.0, edge),
            DVec3::new(edge, 0.0, 0.0),
            DVec3::new(edge, 0.0, edge),
        ] {
            positions.push((corner + offset - origin).as_vec3().to_array());
            attributes.push([depth as f32, 0.0, 0.0]);
        }
        indices.extend([base, base + 1, base + 2, base + 2, base + 1, base + 3]);
    }
    surface_mesh(positions, attributes, indices)
}

/// Two crossed ribbons along each stream's arc, placed against `origin`,
/// wider as more water pours.
fn falls_mesh(falls: &[WaterFall], origin: DVec3) -> Mesh {
    let mut positions = Vec::new();
    let mut attributes = Vec::new();
    let mut indices = Vec::new();
    for fall in falls {
        // A stream of a litre a second is a finger wide; ten litres a hand.
        let half = (fall.rate_m3_s * 400.0)
            .sqrt()
            .mul_add(0.02, 0.02)
            .min(WATER_CELL_METRES);
        for pair in fall.points.windows(2) {
            let along = (pair[1] - pair[0]).normalize_or_zero();
            let side = along.cross(DVec3::Y).normalize_or(DVec3::X);
            for across in [side * half, along.cross(side).normalize_or_zero() * half] {
                let base = u32::try_from(positions.len()).expect("a falls mesh fits u32 indices");
                for point in [
                    pair[0] - across,
                    pair[0] + across,
                    pair[1] - across,
                    pair[1] + across,
                ] {
                    positions.push((point - origin).as_vec3().to_array());
                    attributes.push([0.6, 0.0, 0.0]);
                }
                indices.extend([base, base + 2, base + 1, base + 1, base + 2, base + 3]);
            }
        }
    }
    surface_mesh(positions, attributes, indices)
}

fn surface_mesh(positions: Vec<[f32; 3]>, attributes: Vec<[f32; 3]>, indices: Vec<u32>) -> Mesh {
    let normals = vec![[0.0, 1.0, 0.0]; positions.len()];
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

/// Draws the stored pools and falling streams after each water step. A pool
/// is drawn again when its level moves by more than 5 mm or its surface
/// changes shape.
pub(crate) fn draw_stored_water(
    mut commands: Commands,
    mut tiles: ResMut<WaterTiles>,
    runtime: Res<WorldRuntime>,
    mut meshes: ResMut<Assets<Mesh>>,
    materials: Option<ResMut<Assets<WaterRenderMaterial>>>,
) {
    let Some(mut materials) = materials else {
        return;
    };
    if !super::water::water_enabled() || tiles.drawn_revision == Some(runtime.water_revision) {
        return;
    }
    tiles.drawn_revision = Some(runtime.water_revision);
    let origin = runtime.floating_origin.0;
    let field = &runtime.field;
    let material = tiles
        .material
        .get_or_insert_with(|| materials.add(WaterRenderMaterial::default()))
        .clone();
    let spawn = |commands: &mut Commands, meshes: &mut Assets<Mesh>, name: String, mesh: Mesh| {
        commands
            .spawn((
                Name::new(name),
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(material.clone()),
                Transform::IDENTITY,
                // Shadow cascades cost more than the rest of the water's
                // shading together.
                bevy::light::NotShadowReceiver,
                bevy::light::NotShadowCaster,
                WorldOwned,
            ))
            .id()
    };
    let mut seen = std::collections::HashSet::new();
    for pool in runtime.water.pools() {
        seen.insert(pool.id);
        let columns = open_columns(&pool, field);
        let cells = columns.len();
        if let Some(&(entity, level, drawn)) = tiles.pools.get(&pool.id) {
            if (level - pool.level).abs() < 0.005 && drawn == cells {
                continue;
            }
            commands.entity(entity).despawn();
            tiles.pools.remove(&pool.id);
        }
        if cells == 0 {
            continue;
        }
        let entity = spawn(
            &mut commands,
            &mut meshes,
            format!("Water pool {}", pool.id),
            columns_mesh(&columns, origin),
        );
        tiles.pools.insert(pool.id, (entity, pool.level, cells));
    }
    tiles.pools.retain(|id, (entity, ..)| {
        let keep = seen.contains(id);
        if !keep {
            commands.entity(*entity).despawn();
        }
        keep
    });
    let ground = mechanic_world::TerrainWater {
        field,
        edits: &runtime.edits,
    };
    let joined = joined_columns(&runtime.water.joined_cells(&ground), field);
    let key = fingerprint(&joined);
    if tiles.joined.is_none_or(|(_, drawn)| drawn != key) {
        if let Some((entity, _)) = tiles.joined.take() {
            commands.entity(entity).despawn();
        }
        if !joined.is_empty() {
            let entity = spawn(
                &mut commands,
                &mut meshes,
                "Water joined to a lake".to_owned(),
                columns_mesh(&joined, origin),
            );
            tiles.joined = Some((entity, key));
        }
    }
    if let Some(entity) = tiles.falls.take() {
        commands.entity(entity).despawn();
    }
    // A stream falling under a lake falls through lake water: nothing shows.
    let falls = runtime
        .water_falls
        .iter()
        .filter(|fall| {
            fall.points
                .first()
                .is_some_and(|&top| !field.is_water(top + DVec3::Y * 0.05))
        })
        .cloned()
        .collect::<Vec<_>>();
    if !falls.is_empty() {
        tiles.falls = Some(spawn(
            &mut commands,
            &mut meshes,
            "Falling water".to_owned(),
            falls_mesh(&falls, origin),
        ));
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
