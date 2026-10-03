//! A world-space cache of the procedural terrain fields around the camera.
//!
//! Grass, dirt and stone are drawn from fields (cell borders, cell ids and
//! noise) that a compute shader generates in world space, so the ground never
//! repeats. The fields live in camera-centred clipmaps: [`FIELD_PLANES`]
//! projection planes, each with [`FIELD_LEVELS`] levels of [`FIELD_EDGE`]²
//! texels whose size doubles from [`FIELD_TEXEL_METRES`]. Every level is
//! stored toroidally, so a camera move regenerates only the strips it exposes.

use std::borrow::Cow;

use bevy::{
    asset::{RenderAssetUsages, uuid_handle},
    core_pipeline::schedule::camera_driver,
    math::{DVec2, DVec3},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
            DynamicUniformBuffer, Extent3d, PipelineCache, ShaderStages, ShaderType,
            StorageTextureAccess, TextureDimension, TextureFormat, TextureUsages,
            TextureViewDescriptor, TextureViewDimension,
            binding_types::{texture_storage_2d_array, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue},
        storage::{GpuShaderBuffer, ShaderBuffer},
        texture::GpuImage,
    },
};

use super::{TerrainRenderMaterial, WorldRuntime};
use crate::{camera::MainCamera, settings::AppSettings};

/// Texels along each edge of one cache level. Mirrors `FIELD_EDGE` in the
/// field shaders.
pub(crate) const FIELD_EDGE: u32 = 512;
/// Levels per plane, each with texels twice the size of the last.
pub(crate) const FIELD_LEVELS: u32 = 10;
/// Projection planes, in the terrain shader's projection order: x (yz),
/// y (xz) and z (xy).
pub(crate) const FIELD_PLANES: u32 = 3;
const FIELD_LAYERS: usize = (FIELD_PLANES * FIELD_LEVELS) as usize;
/// Edge of a finest-level texel. Mirrors `FIELD_TEXEL` in the field shaders.
pub(crate) const FIELD_TEXEL_METRES: f64 = 0.003;
/// Windows move in steps of this many texels, so a slow camera regenerates
/// strips of useful width rather than a sliver every frame.
const FIELD_SNAP: i32 = 32;
/// Texels generated per frame at most. A whole layer is always allowed, so a
/// level larger than the remainder waits for the next frame.
const TEXEL_BUDGET: u64 = 1 << 20;
const WORKGROUP: u32 = 8;
const FIELD_SHADER: &str = "shaders/terrain_field_cache.wgsl";

/// Where the cache is centred: the camera, in absolute world metres. `None`
/// while procedural ground is switched off, which leaves the cache idle and
/// the terrain drawn from its textures.
#[derive(Resource, Clone, Copy, Debug, Default, ExtractResource)]
pub(crate) struct TerrainCacheFocus(pub(crate) Option<DVec3>);

/// Stone fields: slab border, chip border, slab id, chip id.
pub(crate) const STONE_FIELDS: Handle<Image> = uuid_handle!("729a24bd-2e99-40c2-85c8-b6acce64102a");
/// Soil fields: band noise, pebble border, pebble id.
pub(crate) const SOIL_FIELDS: Handle<Image> = uuid_handle!("54190aa2-d3a7-4d9e-ba78-35527c381669");
/// Grass fields: stroke noise, and tone shared by every recipe.
pub(crate) const GRASS_FIELDS: Handle<Image> = uuid_handle!("aa2ad5b2-5c2f-4a4c-84a9-520c7ff7f9fd");
/// Fine grain under the fields, too small for the cache to hold far from the
/// camera: grass flecks, dirt grit and stone speckle. A short tile of pure
/// noise, mipmapped like any texture; the unique fields above it keep it
/// from reading as a repeat.
pub(crate) const GRAIN: Handle<Image> = uuid_handle!("dcaa1176-566d-488a-b1ec-c9136fe8e534");
/// Texels along each edge of the grain tile.
const GRAIN_EDGE: u32 = 512;
/// Lattice cells across the tile for each grain, so each repeats exactly
/// with it. The tile spans `GRAIN_METRES` in `terrain_material.wgsl`, 0.75 m:
/// flecks 5 by 1.2 cm, grit 1.2 cm, speckle 6 mm.
const FLECK_CELLS: [u32; 2] = [15, 62];
const GRIT_CELLS: [u32; 2] = [62, 62];
const SPECKLE_CELLS: [u32; 2] = [125, 125];

/// Per layer: the window's lower corner in texels, and whether it is filled.
pub(crate) const FIELD_WINDOWS: Handle<ShaderBuffer> =
    uuid_handle!("8e54c926-5260-43d3-b73f-50ae9c239be9");

/// The terrain material together with the field cache it samples.
pub(crate) struct TerrainRenderPlugin;

impl Plugin for TerrainRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            MaterialPlugin::<TerrainRenderMaterial>::default(),
            ExtractResourcePlugin::<TerrainCacheFocus>::default(),
        ))
        .init_resource::<TerrainCacheFocus>();
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<FieldUpdates>()
            .add_systems(RenderStartup, init_field_pipeline)
            .add_systems(
                Render,
                plan_field_updates.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(RenderGraph, generate_fields.before(camera_driver));
    }

    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        if let Some(mut images) = world.get_resource_mut::<Assets<Image>>() {
            for handle in [STONE_FIELDS, SOIL_FIELDS, GRASS_FIELDS] {
                images
                    .insert(&handle, field_image())
                    .expect("field cache handles are fixed ids");
            }
            images
                .insert(&GRAIN, grain_image())
                .expect("field cache handles are fixed ids");
        }
        if let Some(mut buffers) = world.get_resource_mut::<Assets<ShaderBuffer>>() {
            buffers
                .insert(
                    &FIELD_WINDOWS,
                    ShaderBuffer::from(vec![IVec4::ZERO; FIELD_LAYERS]),
                )
                .expect("field cache handles are fixed ids");
        }
    }
}

/// Centres the cache on the main camera, in absolute world metres, while the
/// player has procedural ground switched on.
pub(super) fn follow_main_camera(
    camera: Query<&GlobalTransform, With<MainCamera>>,
    runtime: Res<WorldRuntime>,
    settings: Res<AppSettings>,
    mut focus: ResMut<TerrainCacheFocus>,
) {
    let position = camera
        .single()
        .ok()
        .filter(|_| settings.procedural_ground())
        .map(|camera| runtime.floating_origin.0 + camera.translation().as_dvec3());
    if focus.0 != position {
        focus.0 = position;
    }
}

/// Builds the world's terrain shader with or without the procedural path, as
/// the player prefers.
pub(super) fn apply_procedural_ground(
    settings: Res<AppSettings>,
    runtime: Res<WorldRuntime>,
    mut materials: ResMut<Assets<TerrainRenderMaterial>>,
) {
    let wanted = settings.procedural_ground();
    if let Some(handle) = runtime.terrain_material.as_ref()
        && materials
            .get(handle)
            .is_some_and(|material| material.procedural_ground != wanted)
        && let Some(mut material) = materials.get_mut(handle)
    {
        material.procedural_ground = wanted;
    }
}

fn field_image() -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: FIELD_EDGE,
            height: FIELD_EDGE,
            depth_or_array_layers: FIELD_PLANES * FIELD_LEVELS,
        },
        TextureDimension::D2,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    image
}

fn hash(cell: [u32; 2], seed: u32) -> f32 {
    let mut state = cell[0]
        .wrapping_mul(0x8da6_b343)
        .wrapping_add(cell[1].wrapping_mul(0xd816_3841))
        .wrapping_add(seed.wrapping_mul(0xcb1a_b31f));
    state ^= state >> 16;
    state = state.wrapping_mul(0x7feb_352d);
    state ^= state >> 15;
    state = state.wrapping_mul(0x846c_a68b);
    state ^= state >> 16;
    #[expect(clippy::cast_precision_loss, reason = "24 bits fit an f32 exactly")]
    let unit = (state >> 8) as f32 / 16_777_216.0;
    unit
}

/// Value noise in [0, 1] at `point` in [0, 1)², over `cells` lattice cells
/// that wrap at the tile's edge.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "lattice indices of a small tile"
)]
fn tiled_noise(point: Vec2, cells: [u32; 2], seed: u32) -> f32 {
    let scaled = point * Vec2::new(cells[0] as f32, cells[1] as f32);
    let base = scaled.floor();
    let f = scaled - base;
    let u = f * f * (Vec2::splat(3.0) - 2.0 * f);
    let corner = |dx: u32, dy: u32| {
        let x = (base.x as u32 + dx) % cells[0];
        let y = (base.y as u32 + dy) % cells[1];
        hash([x, y], seed)
    };
    let low = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * u.x;
    let high = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * u.x;
    low + (high - low) * u.y
}

fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A speck lighter or darker than its surroundings, around one half.
fn speck(noise: f32) -> f32 {
    0.5 + 0.5 * (smoothstep(0.82, 0.88, noise) - 1.0 + smoothstep(0.12, 0.18, noise))
}

/// The grain tile: fleck coverage, grit and speckle around one half. Stored
/// as what each texel shows, so mips average to the right mean.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "texel indices and unit bytes of a small tile"
)]
fn grain_image() -> Image {
    let mut top = Vec::with_capacity((GRAIN_EDGE * GRAIN_EDGE * 4) as usize);
    for y in 0..GRAIN_EDGE {
        for x in 0..GRAIN_EDGE {
            let point = (Vec2::new(x as f32, y as f32) + 0.5) / GRAIN_EDGE as f32;
            let fleck = smoothstep(0.74, 0.79, tiled_noise(point, FLECK_CELLS, 1));
            let grit = speck(tiled_noise(point, GRIT_CELLS, 2));
            let speckle = speck(tiled_noise(point, SPECKLE_CELLS, 3));
            for value in [fleck, grit, speckle, 0.5] {
                top.push((value * 255.0).round() as u8);
            }
        }
    }
    super::terrain_render::texture_array(
        vec![super::terrain_render::full_mip_chain(
            &top, GRAIN_EDGE, GRAIN_EDGE,
        )],
        GRAIN_EDGE,
        TextureFormat::Rgba8Unorm,
    )
}

/// One rectangle of a layer to generate, in world texels of that layer.
/// Mirrors `Region` in the field cache shader.
#[derive(Clone, Copy, Debug, PartialEq, ShaderType)]
pub(crate) struct FieldRegion {
    pub(crate) origin: IVec2,
    pub(crate) size: UVec2,
    pub(crate) layer: u32,
    pub(crate) plane: u32,
    pub(crate) texel: f32,
}

impl FieldRegion {
    fn texels(&self) -> u64 {
        u64::from(self.size.x) * u64::from(self.size.y)
    }
}

/// The focus as one plane sees it.
const fn plane_point(focus: DVec3, plane: u32) -> DVec2 {
    match plane {
        0 => DVec2::new(focus.y, focus.z),
        1 => DVec2::new(focus.x, focus.z),
        _ => DVec2::new(focus.x, focus.y),
    }
}

fn level_texel(level: u32) -> f64 {
    FIELD_TEXEL_METRES * f64::from(1_u32 << level)
}

/// The snapped lower corner of a layer's window around the focus, in texels.
#[expect(
    clippy::cast_possible_truncation,
    reason = "texel indices of a bounded world fit an i32"
)]
fn window_origin(focus: DVec3, plane: u32, level: u32) -> IVec2 {
    let centre = (plane_point(focus, plane) / level_texel(level)).floor();
    let half = f64::from(FIELD_EDGE / 2);
    let corner = IVec2::new((centre.x - half) as i32, (centre.y - half) as i32);
    IVec2::new(
        corner.x.div_euclid(FIELD_SNAP) * FIELD_SNAP,
        corner.y.div_euclid(FIELD_SNAP) * FIELD_SNAP,
    )
}

/// The rectangles a window exposes moving from `old` to `new`: all of it
/// when nothing overlaps, otherwise the strips beyond the old window.
fn exposed(old: Option<IVec2>, new: IVec2) -> Vec<(IVec2, UVec2)> {
    let edge = FIELD_EDGE.cast_signed();
    let whole = vec![(new, UVec2::splat(FIELD_EDGE))];
    let Some(old) = old else {
        return whole;
    };
    if old == new {
        return Vec::new();
    }
    let low = old.max(new);
    let high = (old + edge).min(new + edge);
    if low.x >= high.x || low.y >= high.y {
        return whole;
    }
    let mut rects = Vec::new();
    // Columns outside the kept range, over the new window's full height.
    let columns = if new.x < old.x {
        (new.x, old.x)
    } else {
        (old.x + edge, new.x + edge)
    };
    if columns.0 < columns.1 {
        rects.push((
            IVec2::new(columns.0, new.y),
            UVec2::new((columns.1 - columns.0).cast_unsigned(), FIELD_EDGE),
        ));
    }
    // Rows outside the kept range, over the kept columns only.
    let rows = if new.y < old.y {
        (new.y, old.y)
    } else {
        (old.y + edge, new.y + edge)
    };
    if rows.0 < rows.1 {
        rects.push((
            IVec2::new(low.x, rows.0),
            UVec2::new(
                (high.x - low.x).cast_unsigned(),
                (rows.1 - rows.0).cast_unsigned(),
            ),
        ));
    }
    rects
}

/// Moves every layer's window toward the focus within the texel budget,
/// coarse levels first, and returns the regions to generate. A layer moves
/// whole or not at all, so its window always describes what it holds.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a texel edge in metres is small"
)]
pub(crate) fn plan_regions(
    windows: &mut [Option<IVec2>; FIELD_LAYERS],
    focus: DVec3,
    budget: u64,
) -> Vec<FieldRegion> {
    let mut regions = Vec::new();
    let mut spent = 0;
    for level in (0..FIELD_LEVELS).rev() {
        // The ground plane first: it is most of what is on screen.
        for plane in [1, 0, 2] {
            let layer = plane * FIELD_LEVELS + level;
            let target = window_origin(focus, plane, level);
            let rects: Vec<FieldRegion> = exposed(windows[layer as usize], target)
                .into_iter()
                .map(|(origin, size)| FieldRegion {
                    origin,
                    size,
                    layer,
                    plane,
                    texel: level_texel(level) as f32,
                })
                .collect();
            let cost: u64 = rects.iter().map(FieldRegion::texels).sum();
            if cost == 0 || (spent > 0 && spent + cost > budget) {
                continue;
            }
            spent += cost;
            windows[layer as usize] = Some(target);
            regions.extend(rects);
        }
    }
    regions
}

#[derive(Resource)]
struct FieldPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

/// The render world's view of the cache: what each layer holds and what
/// this frame generates.
#[derive(Resource)]
struct FieldUpdates {
    windows: [Option<IVec2>; FIELD_LAYERS],
    regions: DynamicUniformBuffer<FieldRegion>,
    dispatches: Vec<(u32, UVec2)>,
    bind_group: Option<BindGroup>,
}

impl Default for FieldUpdates {
    fn default() -> Self {
        Self {
            windows: [None; FIELD_LAYERS],
            regions: DynamicUniformBuffer::default(),
            dispatches: Vec::new(),
            bind_group: None,
        }
    }
}

fn init_field_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let storage =
        || texture_storage_2d_array(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly);
    let layout = BindGroupLayoutDescriptor::new(
        "terrain_field_cache",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage(),
                storage(),
                storage(),
                uniform_buffer::<FieldRegion>(true),
            ),
        ),
    );
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("terrain_field_cache".into()),
        layout: vec![layout.clone()],
        shader: asset_server.load(FIELD_SHADER),
        entry_point: Some(Cow::from("generate")),
        ..default()
    });
    commands.insert_resource(FieldPipeline { layout, pipeline });
}

#[expect(
    clippy::too_many_arguments,
    reason = "a render-world system reading each resource it needs"
)]
fn plan_field_updates(
    mut updates: ResMut<FieldUpdates>,
    pipeline: Option<Res<FieldPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    focus: Option<Res<TerrainCacheFocus>>,
    images: Res<RenderAssets<GpuImage>>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let updates = &mut *updates;
    updates.dispatches.clear();
    updates.bind_group = None;
    let (Some(pipeline), Some(focus)) = (pipeline, focus) else {
        return;
    };
    if pipeline_cache
        .get_compute_pipeline(pipeline.pipeline)
        .is_none()
    {
        return;
    }
    let (Some(stone), Some(soil), Some(grass), Some(windows)) = (
        images.get(&STONE_FIELDS),
        images.get(&SOIL_FIELDS),
        images.get(&GRASS_FIELDS),
        buffers.get(&FIELD_WINDOWS),
    ) else {
        return;
    };
    let regions = focus
        .0
        .map(|position| plan_regions(&mut updates.windows, position, TEXEL_BUDGET))
        .unwrap_or_default();
    let table: Vec<u8> = updates
        .windows
        .iter()
        .flat_map(|window| match window {
            Some(origin) => [origin.x, origin.y, 1, 0],
            None => [0; 4],
        })
        .flat_map(i32::to_le_bytes)
        .collect();
    queue.write_buffer(&windows.buffer, 0, &table);
    if regions.is_empty() {
        return;
    }
    updates.regions.clear();
    for region in &regions {
        let offset = updates.regions.push(region);
        updates.dispatches.push((offset, region.size));
    }
    updates.regions.write_buffer(&device, &queue);
    let Some(binding) = updates.regions.binding() else {
        updates.dispatches.clear();
        return;
    };
    updates.bind_group = Some(device.create_bind_group(
        "terrain_field_cache",
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            &stone.texture_view,
            &soil.texture_view,
            &grass.texture_view,
            binding,
        )),
    ));
}

fn generate_fields(
    mut render_context: RenderContext,
    updates: Res<FieldUpdates>,
    pipeline: Option<Res<FieldPipeline>>,
    pipeline_cache: Res<PipelineCache>,
) {
    let (Some(pipeline), Some(bind_group)) = (pipeline, updates.bind_group.as_ref()) else {
        return;
    };
    let Some(compute) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };
    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("terrain_field_cache"),
            ..default()
        });
    pass.set_pipeline(compute);
    for &(offset, size) in &updates.dispatches {
        pass.set_bind_group(0, bind_group, &[offset]);
        pass.dispatch_workgroups(size.x.div_ceil(WORKGROUP), size.y.div_ceil(WORKGROUP), 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn covered(regions: &[FieldRegion], layer: u32) -> u64 {
        regions
            .iter()
            .filter(|region| region.layer == layer)
            .map(FieldRegion::texels)
            .sum()
    }

    #[test]
    fn first_frames_fill_every_layer_coarse_first_within_the_budget() {
        let mut windows = [None; FIELD_LAYERS];
        let layer_texels = u64::from(FIELD_EDGE * FIELD_EDGE);
        let first = plan_regions(&mut windows, DVec3::ZERO, 4 * layer_texels);
        assert_eq!(first.len(), 4);
        assert!(
            first
                .iter()
                .all(|region| region.size == UVec2::splat(FIELD_EDGE))
        );
        // The coarsest ground level is the first thing generated.
        assert_eq!(first[0].layer, FIELD_LEVELS + FIELD_LEVELS - 1);
        let mut frames = 1;
        while windows.iter().any(Option::is_none) {
            plan_regions(&mut windows, DVec3::ZERO, 4 * layer_texels);
            frames += 1;
        }
        assert_eq!(frames, FIELD_LAYERS.div_ceil(4));
        assert!(plan_regions(&mut windows, DVec3::ZERO, TEXEL_BUDGET).is_empty());
    }

    #[test]
    fn a_short_move_regenerates_only_the_exposed_strips() {
        let mut windows = [None; FIELD_LAYERS];
        while windows.iter().any(Option::is_none) {
            plan_regions(&mut windows, DVec3::ZERO, u64::MAX);
        }
        let ground_fine = FIELD_LEVELS as usize;
        let before = windows[ground_fine].unwrap();
        // 20 cm along x and 10 cm along z: whole snaps at 3 mm texels.
        let regions = plan_regions(&mut windows, DVec3::new(0.2, 0.0, 0.1), u64::MAX);
        let after = windows[ground_fine].unwrap();
        let moved = (after - before).abs();
        assert!(moved.x > 0 && moved.y > 0);
        let edge = u64::from(FIELD_EDGE);
        let expected = u64::from(moved.x.cast_unsigned()) * edge
            + u64::from(moved.y.cast_unsigned()) * (edge - u64::from(moved.x.cast_unsigned()));
        assert_eq!(covered(&regions, FIELD_LEVELS), expected);
        // Coarse levels have not reached a new snap and stay as they are.
        assert_eq!(covered(&regions, FIELD_LEVELS + FIELD_LEVELS - 1), 0);
    }

    #[test]
    fn exposed_strips_and_the_kept_window_tile_the_new_window_exactly() {
        let edge = FIELD_EDGE.cast_signed();
        let old = IVec2::new(64, -96);
        for new in [
            IVec2::new(96, -96),
            IVec2::new(32, -64),
            IVec2::new(-32, 0),
            IVec2::new(64 + edge, -96),
        ] {
            let mut hits = vec![0_u8; (FIELD_EDGE * FIELD_EDGE) as usize];
            let mut mark = |x: i32, y: i32| {
                let x = usize::try_from(x - new.x).unwrap();
                let y = usize::try_from(y - new.y).unwrap();
                hits[y * FIELD_EDGE as usize + x] += 1;
            };
            for (origin, size) in exposed(Some(old), new) {
                for y in 0..size.y.cast_signed() {
                    for x in 0..size.x.cast_signed() {
                        mark(origin.x + x, origin.y + y);
                    }
                }
            }
            for y in new.y..new.y + edge {
                for x in new.x..new.x + edge {
                    let kept =
                        (old.x..old.x + edge).contains(&x) && (old.y..old.y + edge).contains(&y);
                    if kept {
                        mark(x, y);
                    }
                }
            }
            assert!(hits.iter().all(|&count| count == 1), "{new}");
        }
    }

    #[test]
    fn grain_wraps_without_a_seam() {
        for cells in [FLECK_CELLS, GRIT_CELLS, SPECKLE_CELLS] {
            for step in 0..64_u8 {
                let along = f32::from(step) / 64.0;
                let edge = 1.0e-5;
                let across_x = tiled_noise(Vec2::new(1.0 - edge, along), cells, 4)
                    - tiled_noise(Vec2::new(0.0, along), cells, 4);
                let across_y = tiled_noise(Vec2::new(along, 1.0 - edge), cells, 4)
                    - tiled_noise(Vec2::new(along, 0.0), cells, 4);
                assert!(across_x.abs() < 1.0e-2 && across_y.abs() < 1.0e-2);
            }
        }
    }

    #[test]
    fn windows_centre_on_the_focus_in_every_plane() {
        let focus = DVec3::new(120.0, -40.0, 3_000.0);
        for plane in 0..FIELD_PLANES {
            for level in 0..FIELD_LEVELS {
                let texel = level_texel(level);
                let origin = window_origin(focus, plane, level).as_dvec2() * texel;
                let centre = origin + DVec2::splat(f64::from(FIELD_EDGE / 2) * texel);
                let offset = (centre - plane_point(focus, plane)).abs();
                let slack = f64::from(FIELD_SNAP + 1) * texel;
                assert!(offset.max_element() <= slack, "plane {plane} level {level}");
            }
        }
    }
}
