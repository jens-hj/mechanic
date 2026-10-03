//! Terrain materials, texture mip chains, and chunk meshes.

use super::brush::BrushPreview;
use super::{
    AlphaMode, Asset, AssetServer, Assets, Color, Commands, Component, DirectionalLight, EulerRot,
    Handle, Image, LinearRgba, Material, Mesh, Mesh3d, MeshMaterial3d, Meshable, Name, Quat,
    Reflect, ResMut, Result, Sphere, StandardMaterial, String, ToOwned, ToString, Transform, Vec,
    Visibility, WorldDiagnostics, WorldOwned, WorldRuntime, default, format, vec,
};
use bevy::asset::RenderAssetUsages;
use bevy::math::Vec4;
use bevy::mesh::{Indices, MeshVertexAttribute, VertexAttributeValues};
use bevy::render::render_resource::{
    AsBindGroup, Face, PrimitiveTopology, ShaderType, TextureFormat, VertexFormat,
};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::{ShaderDefVal, ShaderRef};
use bevy::tasks::futures::check_ready;
use bevy::tasks::{AsyncComputeTaskPool, Task, TaskPool};
use mechanic_world::{
    SurfaceId, SurfacePalette, TREE_TEXTURE_LUMA, TerrainMeshChunk, TerrainSpatialIndex,
    TerrainStreamer, TextureSet, TreeTexture,
};

/// The terrain's vertex stage and full material.
const TERRAIN_SHADER: &str = "shaders/terrain_material.wgsl";

#[derive(Component)]
pub(super) struct TerrainNodeRender;

/// Per-vertex weights of a chunk's first four surface slots.
pub(crate) const ATTRIBUTE_TERRAIN_WEIGHTS_LOW: MeshVertexAttribute = MeshVertexAttribute::new(
    "TerrainWeightsLow",
    0x6d65_6368_0001,
    VertexFormat::Unorm8x4,
);
/// Per-vertex weights of a chunk's last four surface slots.
pub(crate) const ATTRIBUTE_TERRAIN_WEIGHTS_HIGH: MeshVertexAttribute = MeshVertexAttribute::new(
    "TerrainWeightsHigh",
    0x6d65_6368_0002,
    VertexFormat::Unorm8x4,
);
/// A chunk's eight palette surfaces, two 16-bit ids per word, on every vertex.
pub(crate) const ATTRIBUTE_TERRAIN_SLOTS: MeshVertexAttribute =
    MeshVertexAttribute::new("TerrainSlots", 0x6d65_6368_0003, VertexFormat::Uint32x4);

/// Surfaces one chunk can show.
const SURFACE_SLOTS: usize = 8;
/// Edge of one colour, normal, or surface texture layer.
const TERRAIN_LAYER_EDGE: u32 = 1_536;
/// Edge of one tint-mask layer; masks are soft and need less detail.
const TERRAIN_MASK_EDGE: u32 = 768;
/// Edge of one procedural tree texture layer: a 1.5 m repeat at 3 mm.
const TREE_LAYER_EDGE: u32 = 512;
/// Layer numbers from here on name tree texture layers. Mirrors
/// `TREE_LAYER_BASE` in the terrain shader.
const TREE_LAYER_BASE: u32 = 64;
/// Map kinds a tree texture has: base colour, normal, and ORM. Trees tint
/// everywhere, so they need no mask.
const TREE_MAP_KINDS: usize = 3;

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone, PartialEq)]
#[bind_group_data(TerrainMaterialKey)]
pub(crate) struct TerrainRenderMaterial {
    #[texture(0, dimension = "2d_array")]
    #[sampler(4)]
    pub(super) base_color: Handle<Image>,
    #[texture(1, dimension = "2d_array")]
    pub(super) normal: Handle<Image>,
    #[texture(2, dimension = "2d_array")]
    pub(super) orm: Handle<Image>,
    #[texture(3, dimension = "2d_array")]
    pub(super) tint_mask: Handle<Image>,
    #[storage(5, read_only)]
    pub(super) surfaces: Handle<ShaderBuffer>,
    /// How wet the ground is around the camera: its fill, then the height of
    /// the ground it was measured at over `wet_window.w`.
    #[texture(6)]
    #[sampler(7)]
    pub(crate) wetness: Handle<Image>,
    /// Where the wetness map lies: its lower x and z corner and edge in
    /// metres, and the height its ground heights are measured from, all
    /// against the floating origin.
    #[uniform(8)]
    pub(crate) wet_window: Vec4,
    /// Diagnostic transfer amounts and bed height, sampled without filtering.
    #[texture(9, filterable = false)]
    pub(crate) erosion_map: Handle<Image>,
    /// Lower x/z corner, edge length, and base height relative to the origin.
    /// A zero edge disables diagnostics.
    #[uniform(10)]
    pub(crate) erosion_window: Vec4,
    /// Procedural bark and leaf maps, one layer per palette tree texture.
    #[texture(11, dimension = "2d_array")]
    pub(super) tree_base_color: Handle<Image>,
    #[texture(12, dimension = "2d_array")]
    pub(super) tree_normal: Handle<Image>,
    #[texture(13, dimension = "2d_array")]
    pub(super) tree_orm: Handle<Image>,
    /// The procedural field cache around the camera; see
    /// [`super::terrain_cache`].
    #[texture(14, dimension = "2d_array")]
    pub(super) stone_fields: Handle<Image>,
    #[texture(15, dimension = "2d_array")]
    pub(super) soil_fields: Handle<Image>,
    #[texture(16, dimension = "2d_array")]
    pub(super) grass_fields: Handle<Image>,
    #[storage(17, read_only)]
    pub(super) field_windows: Handle<ShaderBuffer>,
    /// Fine grain under the procedural fields; see [`super::terrain_cache`].
    #[texture(18, dimension = "2d_array")]
    pub(super) grain: Handle<Image>,
    /// Whether grass, dirt and stone are drawn procedurally. Switched off,
    /// the shader is built without the procedural path and costs nothing
    /// for it.
    pub(crate) procedural_ground: bool,
}

/// Which terrain shader a material needs: with the procedural path or not.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TerrainMaterialKey {
    procedural_ground: bool,
}

impl From<&TerrainRenderMaterial> for TerrainMaterialKey {
    fn from(material: &TerrainRenderMaterial) -> Self {
        Self {
            procedural_ground: material.procedural_ground,
        }
    }
}

/// A wetness map of dry ground, for terrain drawn before water has run.
pub(crate) fn dry_ground(images: &mut Assets<Image>) -> Handle<Image> {
    images.add(super::water_render::wetness_image(1, vec![0.0; 3]))
}

/// One palette surface as the terrain shader reads it.
#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub(crate) struct TerrainSurfaceGpu {
    /// Linear tint; w above one half recolours instead of multiplying.
    tint: Vec4,
    /// Texture layer, tint-masked flag, roughness and repeat multipliers.
    params: Vec4,
    /// Mean luminance of the layer's base colour, and the procedural recipe
    /// that draws the layer instead of its textures, or zero.
    shade: Vec4,
}

/// Storage buffer of every palette surface, in [`mechanic_world::SurfaceId`] order.
pub(crate) fn terrain_surface_buffer(
    palette: &SurfacePalette,
    layer_luma: &[f32; TextureSet::ALL.len()],
) -> ShaderBuffer {
    ShaderBuffer::from(terrain_surfaces(palette, layer_luma))
}

/// Every palette surface as the shader reads it, in id order.
fn terrain_surfaces(
    palette: &SurfacePalette,
    layer_luma: &[f32; TextureSet::ALL.len()],
) -> Vec<TerrainSurfaceGpu> {
    palette
        .looks()
        .iter()
        .map(|look| {
            if let Some(tree) = look.tree_texture {
                // Tree textures are drawn to a shared mean brightness.
                return TerrainSurfaceGpu {
                    tint: Vec4::new(
                        look.tint[0],
                        look.tint[1],
                        look.tint[2],
                        if look.recolor { 1.0 } else { 0.0 },
                    ),
                    params: Vec4::new(
                        (TREE_LAYER_BASE + u32::from(tree)) as f32,
                        0.0,
                        look.roughness,
                        look.scale,
                    ),
                    shade: Vec4::new(TREE_TEXTURE_LUMA, 0.0, 0.0, 0.0),
                };
            }
            let layer = look.texture.layer();
            TerrainSurfaceGpu {
                tint: Vec4::new(
                    look.tint[0],
                    look.tint[1],
                    look.tint[2],
                    if look.recolor { 1.0 } else { 0.0 },
                ),
                params: Vec4::new(
                    layer as f32,
                    if look.masked { 1.0 } else { 0.0 },
                    look.roughness,
                    look.scale,
                ),
                shade: Vec4::new(
                    layer_luma[layer as usize],
                    f32::from(procedural_recipe(look.texture)),
                    0.0,
                    0.0,
                ),
            }
        })
        .collect()
}

/// Vertex attributes the terrain shader's own vertex stage reads. Every mesh
/// drawn with [`TerrainRenderMaterial`] must carry all of them.
pub(super) fn terrain_vertex_attributes() -> [bevy::mesh::VertexAttributeDescriptor; 7] {
    [
        Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
        Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
        Mesh::ATTRIBUTE_UV_0.at_shader_location(2),
        Mesh::ATTRIBUTE_UV_1.at_shader_location(3),
        ATTRIBUTE_TERRAIN_WEIGHTS_LOW.at_shader_location(8),
        ATTRIBUTE_TERRAIN_WEIGHTS_HIGH.at_shader_location(9),
        ATTRIBUTE_TERRAIN_SLOTS.at_shader_location(10),
    ]
}

impl Material for TerrainRenderMaterial {
    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        if key.bind_group_data.procedural_ground
            && let Some(fragment) = descriptor.fragment.as_mut()
        {
            fragment.shader_defs.push("PROCEDURAL_GROUND".into());
        }
        // The prepass keeps Bevy's own vertex stage and layout.
        let prepass = descriptor
            .vertex
            .shader_defs
            .iter()
            .any(|definition| {
                matches!(definition, ShaderDefVal::Bool(name, true) if name == "PREPASS_PIPELINE")
            });
        if !prepass {
            descriptor.vertex.buffers = vec![layout.0.get_layout(&terrain_vertex_attributes())?];
        }
        descriptor.label = Some(
            format!(
                "mechanic_terrain:{}",
                descriptor.label.as_deref().unwrap_or_default()
            )
            .into(),
        );
        Ok(())
    }

    fn vertex_shader() -> ShaderRef {
        TERRAIN_SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        crate::render_experiments::current().terrain_shader().into()
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "entering the world touches each terrain asset store once"
)]
pub(super) fn spawn_world_terrain(
    commands: &mut Commands,
    asset_server: &AssetServer,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    terrain_materials: &mut Assets<TerrainRenderMaterial>,
    images: &mut Assets<Image>,
    surface_buffers: &mut Assets<ShaderBuffer>,
    runtime: &mut WorldRuntime,
    diagnostics: &mut WorldDiagnostics,
) {
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
    runtime.terrain_cutovers = super::streaming::TerrainCutovers::default();
    runtime.player_terrain_ready = false;
    runtime.selection_focus = None;
    runtime.selected_terrain_revision = u64::MAX;
    let (terrain_material, build) = terrain_render_material(
        asset_server,
        images,
        surface_buffers,
        runtime.field.palette(),
    );
    runtime.terrain_textures = Some(build);
    runtime.terrain_material = Some(terrain_materials.add(terrain_material));
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
        crate::sky::Sun,
        bevy::light::SunDisk::EARTH,
        DirectionalLight {
            illuminance: bevy::light::light_consts::lux::RAW_SUNLIGHT,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, -0.55, 0.0)),
        WorldOwned,
    ));
}

/// Source maps still being folded into the terrain texture arrays. The
/// arrays are assembled a layer per frame so loading a world never stalls.
pub(crate) struct TerrainTextureBuild {
    /// Source images per texture set and map kind, in array-layer order.
    sources: Vec<[Option<Handle<Image>>; MAP_KINDS]>,
    /// Finished layers, with mip chains, per map kind.
    layers: [Vec<Vec<u8>>; MAP_KINDS],
    /// Array images the material already binds.
    targets: [Handle<Image>; MAP_KINDS],
    /// Mean base-colour luminance per layer.
    luma: [f32; TextureSet::ALL.len()],
    surfaces: Handle<ShaderBuffer>,
    /// Tree layers being drawn off the main thread.
    tree_task: Option<Task<TreeLayers>>,
    /// Tree arrays the material already binds.
    tree_targets: [Handle<Image>; TREE_MAP_KINDS],
    /// Whether the tree arrays are published.
    trees_done: bool,
}

/// Finished tree layers, with mip chains, per map kind.
type TreeLayers = [Vec<Vec<u8>>; TREE_MAP_KINDS];

const MAP_KINDS: usize = 4;
const MAP_NAMES: [&str; MAP_KINDS] = ["base_color", "normal", "orm", "tint"];

fn texture_directory(set: TextureSet) -> &'static str {
    match set {
        TextureSet::Grass => "terrain/grass/grass",
        TextureSet::Dirt => "terrain/dirt/dirt",
        TextureSet::Stone => "terrain/stone/stone",
        TextureSet::Sand => "materials/sand/sand",
        TextureSet::Iron => "materials/iron/iron",
        TextureSet::Graphite => "materials/graphite/graphite",
        TextureSet::Copper => "materials/copper/copper",
        TextureSet::Wood => "materials/wood/wood",
    }
}

/// The terrain shader's procedural recipe for a set, or zero for a set only
/// ever drawn from its textures. Sets with a recipe keep their textures for
/// when procedural ground is switched off. Mirrors the `RECIPE_*` constants
/// in the shader.
pub(crate) const fn procedural_recipe(set: TextureSet) -> u8 {
    match set {
        TextureSet::Grass => 1,
        TextureSet::Dirt => 2,
        TextureSet::Stone => 3,
        TextureSet::Sand
        | TextureSet::Iron
        | TextureSet::Graphite
        | TextureSet::Copper
        | TextureSet::Wood => 0,
    }
}

/// Sets whose textures come with a tint mask; the rest tint everywhere.
const fn has_tint_mask(set: TextureSet) -> bool {
    matches!(
        set,
        TextureSet::Grass | TextureSet::Dirt | TextureSet::Copper
    )
}

pub(crate) fn terrain_render_material(
    asset_server: &AssetServer,
    images: &mut Assets<Image>,
    surface_buffers: &mut Assets<ShaderBuffer>,
    palette: &SurfacePalette,
) -> (TerrainRenderMaterial, TerrainTextureBuild) {
    let sources = TextureSet::ALL
        .iter()
        .map(|&set| {
            core::array::from_fn(|kind| {
                if kind == 3 && !has_tint_mask(set) {
                    return None;
                }
                let is_srgb = kind == 0;
                let path = format!("{}_{}.png", texture_directory(set), MAP_NAMES[kind]);
                Some(
                    asset_server
                        .load_builder()
                        .with_settings(move |settings: &mut bevy::image::ImageLoaderSettings| {
                            crate::render::materials::configure_repeating_texture(
                                settings, is_srgb,
                            );
                        })
                        .load(path),
                )
            })
        })
        .collect();
    let targets: [Handle<Image>; MAP_KINDS] = core::array::from_fn(|_| images.reserve_handle());
    let tree_targets: [Handle<Image>; TREE_MAP_KINDS] =
        core::array::from_fn(|_| images.reserve_handle());
    let luma = [0.5; TextureSet::ALL.len()];
    let surfaces = surface_buffers.add(terrain_surface_buffer(palette, &luma));
    let material = TerrainRenderMaterial {
        base_color: targets[0].clone(),
        normal: targets[1].clone(),
        orm: targets[2].clone(),
        tint_mask: targets[3].clone(),
        surfaces: surfaces.clone(),
        wetness: dry_ground(images),
        wet_window: Vec4::ZERO,
        erosion_map: super::erosion_overlay::empty_map(images),
        erosion_window: Vec4::ZERO,
        tree_base_color: tree_targets[0].clone(),
        tree_normal: tree_targets[1].clone(),
        tree_orm: tree_targets[2].clone(),
        stone_fields: super::terrain_cache::STONE_FIELDS,
        soil_fields: super::terrain_cache::SOIL_FIELDS,
        grass_fields: super::terrain_cache::GRASS_FIELDS,
        field_windows: super::terrain_cache::FIELD_WINDOWS,
        grain: super::terrain_cache::GRAIN,
        procedural_ground: false,
    };
    (
        material,
        TerrainTextureBuild {
            sources,
            layers: Default::default(),
            targets,
            luma,
            surfaces,
            tree_task: Some(draw_tree_layers(palette.tree_textures().to_vec())),
            tree_targets,
            trees_done: false,
        },
    )
}

/// Folds one loaded source map per frame into its array layer, then
/// publishes the finished arrays and the palette's shading.
pub(super) fn prepare_terrain_textures(
    mut images: ResMut<Assets<Image>>,
    mut surface_buffers: ResMut<Assets<ShaderBuffer>>,
    mut runtime: ResMut<WorldRuntime>,
) {
    let runtime = &mut *runtime;
    let Some(build) = runtime.terrain_textures.as_mut() else {
        return;
    };
    match advance_terrain_textures(build, &mut images) {
        Ok(false) => {}
        Ok(true) => {
            if let Some(mut buffer) = surface_buffers.get_mut(&build.surfaces) {
                *buffer = terrain_surface_buffer(runtime.field.palette(), &build.luma);
            }
            runtime.terrain_layer_luma = build.luma;
            runtime.terrain_textures = None;
        }
        Err(error) => {
            runtime.load_error = Some(error);
            runtime.terrain_textures = None;
        }
    }
}

/// Produces the next array layer once its source has loaded, publishing an
/// array when its last layer is done. True once every array is published.
pub(crate) fn advance_terrain_textures(
    build: &mut TerrainTextureBuild,
    images: &mut Assets<Image>,
) -> Result<bool, String> {
    if !build.trees_done {
        advance_tree_textures(build, images)?;
    }
    let layer_count = TextureSet::ALL.len();
    // Layers are produced strictly in order so each array stays layer-major.
    let next = (0..MAP_KINDS).find_map(|kind| {
        let layer = build.layers[kind].len();
        (layer < layer_count).then_some((kind, layer))
    });
    let Some((kind, layer)) = next else {
        return Ok(build.trees_done);
    };
    let edge = if kind == 3 {
        TERRAIN_MASK_EDGE
    } else {
        TERRAIN_LAYER_EDGE
    };
    let chain = match &build.sources[layer][kind] {
        None => full_mip_chain(&vec![255_u8; (edge * edge * 4) as usize], edge, edge),
        Some(handle) => {
            let Some(image) = images.get(handle) else {
                return Ok(false);
            };
            let top = texture_layer(image, edge).map_err(|error| {
                format!(
                    "terrain texture {}_{}: {error}",
                    texture_directory(TextureSet::ALL[layer]),
                    MAP_NAMES[kind]
                )
            })?;
            if kind == 0 {
                build.luma[layer] = mean_luma(&top);
            }
            full_mip_chain(&top, edge, edge)
        }
    };
    build.layers[kind].push(chain);
    // Construction materials share some source maps and keep their handles
    // across world entries. Let Bevy unload only images with no remaining users.
    build.sources[layer][kind] = None;
    if build.layers[kind].len() == layer_count {
        let format = if kind == 0 {
            TextureFormat::Rgba8UnormSrgb
        } else {
            TextureFormat::Rgba8Unorm
        };
        let layers = core::mem::take(&mut build.layers[kind]);
        images
            .insert(
                build.targets[kind].id(),
                texture_array(layers, edge, format),
            )
            .map_err(|error| format!("terrain texture array: {error}"))?;
        // Keep the published kind counted as done.
        build.layers[kind] = vec![Vec::new(); layer_count];
    }
    Ok(build.trees_done
        && build
            .layers
            .iter()
            .all(|layers| layers.len() == layer_count))
}

/// Starts drawing every tree texture's maps off the main thread, each on its
/// own thread: a species' leaves take up to a few hundred milliseconds. A
/// world without trees gets one blank layer, as an array needs one.
fn draw_tree_layers(textures: Vec<TreeTexture>) -> Task<TreeLayers> {
    AsyncComputeTaskPool::get_or_init(TaskPool::new).spawn(async move {
        let edge = TREE_LAYER_EDGE;
        let drawn: Vec<[Vec<u8>; TREE_MAP_KINDS]> = std::thread::scope(|scope| {
            let threads = textures
                .iter()
                .map(|texture| {
                    scope.spawn(move || {
                        let maps = texture.maps(edge);
                        [maps.base_color, maps.normal, maps.orm]
                            .map(|top| full_mip_chain(&top, edge, edge))
                    })
                })
                .collect::<Vec<_>>();
            threads
                .into_iter()
                .map(|thread| thread.join().expect("drawing a tree texture panicked"))
                .collect()
        });
        let mut layers = TreeLayers::default();
        if drawn.is_empty() {
            let blank = [
                [255_u8, 255, 255, 255],
                [128, 128, 255, 255],
                [255, 230, 0, 255],
            ];
            for (kind, pixel) in blank.iter().enumerate() {
                let top = pixel.repeat((edge * edge) as usize);
                layers[kind].push(full_mip_chain(&top, edge, edge));
            }
        }
        for maps in drawn {
            for (kind, chain) in maps.into_iter().enumerate() {
                layers[kind].push(chain);
            }
        }
        layers
    })
}

/// Publishes the tree arrays once their maps are drawn.
fn advance_tree_textures(
    build: &mut TerrainTextureBuild,
    images: &mut Assets<Image>,
) -> Result<(), String> {
    let Some(layers) = build.tree_task.as_mut().and_then(check_ready) else {
        return Ok(());
    };
    build.tree_task = None;
    for (kind, layers) in layers.into_iter().enumerate() {
        let format = if kind == 0 {
            TextureFormat::Rgba8UnormSrgb
        } else {
            TextureFormat::Rgba8Unorm
        };
        images
            .insert(
                build.tree_targets[kind].id(),
                texture_array(layers, TREE_LAYER_EDGE, format),
            )
            .map_err(|error| format!("tree texture array: {error}"))?;
    }
    build.trees_done = true;
    Ok(())
}

/// Rewrites the palette buffer after the definition changed.
#[cfg(debug_assertions)]
pub(super) fn refresh_terrain_surfaces(
    surface_buffers: &mut Assets<ShaderBuffer>,
    materials: &Assets<TerrainRenderMaterial>,
    runtime: &WorldRuntime,
    luma: &[f32; TextureSet::ALL.len()],
) {
    let Some(material) = runtime
        .terrain_material
        .as_ref()
        .and_then(|handle| materials.get(handle))
    else {
        return;
    };
    if let Some(mut buffer) = surface_buffers.get_mut(&material.surfaces) {
        *buffer = terrain_surface_buffer(runtime.field.palette(), luma);
    }
}

/// The top level of one array layer: the source's RGBA8 pixels box-filtered
/// down to `edge` when it is a power-of-two multiple of it.
fn texture_layer(image: &Image, edge: u32) -> Result<Vec<u8>, String> {
    if !matches!(
        image.texture_descriptor.format,
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
    ) {
        return Err(format!(
            "unsupported format {:?}",
            image.texture_descriptor.format
        ));
    }
    let width = image.texture_descriptor.size.width;
    let height = image.texture_descriptor.size.height;
    if width != height
        || width < edge
        || !(width / edge).is_power_of_two()
        || !width.is_multiple_of(edge)
    {
        return Err(format!(
            "is {width}×{height}; expected a multiple of {edge} square"
        ));
    }
    let Some(data) = image.data.as_ref() else {
        return Err("has no CPU pixel data".to_owned());
    };
    let mut level = data[..(width * width * 4) as usize].to_vec();
    let mut level_edge = width;
    while level_edge > edge {
        level = halve_rgba8(&level, level_edge, level_edge).0;
        level_edge /= 2;
    }
    Ok(level)
}

fn mean_luma(pixels: &[u8]) -> f32 {
    let mut sum = 0.0_f64;
    for pixel in pixels.chunks_exact(4) {
        let linear = |value: u8| {
            let value = f64::from(value) / 255.0;
            if value <= 0.040_45 {
                value / 12.92
            } else {
                ((value + 0.055) / 1.055).powf(2.4)
            }
        };
        sum += 0.2126 * linear(pixel[0]) + 0.7152 * linear(pixel[1]) + 0.0722 * linear(pixel[2]);
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "a mean over a few million pixels"
    )]
    let mean = (sum / (pixels.len() / 4).max(1) as f64) as f32;
    mean
}

pub(super) fn full_mip_chain(top: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut chain = Vec::with_capacity(full_rgba8_mip_byte_count(width, height));
    chain.extend_from_slice(top);
    let mut level = top.to_vec();
    let (mut width, mut height) = (width, height);
    while width > 1 || height > 1 {
        let (next, next_width, next_height) = halve_rgba8(&level, width, height);
        chain.extend_from_slice(&next);
        level = next;
        width = next_width;
        height = next_height;
    }
    chain
}

pub(super) fn texture_array(layers: Vec<Vec<u8>>, edge: u32, format: TextureFormat) -> Image {
    let layer_count = u32::try_from(layers.len()).expect("a handful of layers");
    let mut image = Image::new_uninit(
        bevy::render::render_resource::Extent3d {
            width: edge,
            height: edge,
            depth_or_array_layers: layer_count,
        },
        bevy::render::render_resource::TextureDimension::D2,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(layers.concat());
    image.data_order = bevy::render::render_resource::TextureDataOrder::LayerMajor;
    image.texture_descriptor.mip_level_count = 32 - edge.leading_zeros();
    image.texture_view_descriptor = Some(bevy::render::render_resource::TextureViewDescriptor {
        dimension: Some(bevy::render::render_resource::TextureViewDimension::D2Array),
        ..default()
    });
    image.sampler = bevy::image::ImageSampler::Descriptor(bevy::image::ImageSamplerDescriptor {
        address_mode_u: bevy::image::ImageAddressMode::Repeat,
        address_mode_v: bevy::image::ImageAddressMode::Repeat,
        mag_filter: bevy::image::ImageFilterMode::Linear,
        min_filter: bevy::image::ImageFilterMode::Linear,
        mipmap_filter: bevy::image::ImageFilterMode::Linear,
        anisotropy_clamp: 8,
        ..default()
    });
    image
}

/// One 2×2 box-filter step of RGBA8 pixels.
fn halve_rgba8(previous: &[u8], width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    let next_width = (width / 2).max(1);
    let next_height = (height / 2).max(1);
    let mut next = vec![0_u8; (next_width * next_height * 4) as usize];
    for y in 0..next_height {
        for x in 0..next_width {
            let source_x = x * 2;
            let source_y = y * 2;
            let adjacent_x = (source_x + 1).min(width - 1);
            let adjacent_y = (source_y + 1).min(height - 1);
            for channel in 0..4_u32 {
                let source = |sample_x: u32, sample_y: u32| {
                    previous[((sample_y * width + sample_x) * 4 + channel) as usize]
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
    (next, next_width, next_height)
}

pub(crate) fn generate_rgba8_mip_chain(image: &mut Image) -> Result<(), String> {
    if image.texture_descriptor.mip_level_count > 1 {
        return Ok(());
    }
    if !matches!(
        image.texture_descriptor.format,
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
    ) {
        return Err(format!(
            "texture has unsupported runtime format {:?}",
            image.texture_descriptor.format
        ));
    }
    let width = image.texture_descriptor.size.width;
    let height = image.texture_descriptor.size.height;
    if image.texture_descriptor.size.depth_or_array_layers != 1 {
        return Err("texture must be a single 2D image".to_owned());
    }
    let expected_top_bytes = usize::try_from(u64::from(width) * u64::from(height) * 4)
        .map_err(|_| "terrain texture dimensions overflow memory size".to_owned())?;
    let Some(top) = image.data.as_ref() else {
        return Err("texture has no CPU pixel data".to_owned());
    };
    if top.len() != expected_top_bytes {
        return Err(format!(
            "texture contains {} bytes, expected {expected_top_bytes}",
            top.len()
        ));
    }

    let level_count = 32 - width.max(height).leading_zeros();
    let chain = full_mip_chain(top, width, height);
    image.data = Some(chain);
    image.texture_descriptor.mip_level_count = level_count;
    Ok(())
}

pub(super) fn full_rgba8_mip_byte_count(mut width: u32, mut height: u32) -> usize {
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

pub(super) fn terrain_mesh_is_renderable(chunk: &TerrainMeshChunk, index_count: usize) -> bool {
    !chunk.vertices.is_empty() && index_count != 0
}

/// Chooses up to [`SURFACE_SLOTS`] surfaces for a chunk and maps every vertex
/// surface to one of them. Rare extras merge into the most common kept
/// surface that draws the same texture, or failing that one of the same
/// material, then one from the same texture set, then the most common one:
/// a distant crown stays leaves rather than turning into the meadow below.
pub(super) fn surface_slots(
    surfaces: &[SurfaceId],
    palette: &SurfacePalette,
) -> (Vec<SurfaceId>, std::collections::BTreeMap<SurfaceId, usize>) {
    let mut counts = std::collections::BTreeMap::<SurfaceId, usize>::new();
    for surface in surfaces {
        *counts.entry(*surface).or_default() += 1;
    }
    let mut ranked = counts.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|first, second| second.1.cmp(&first.1).then(first.0.cmp(&second.0)));
    let kept = ranked
        .iter()
        .take(SURFACE_SLOTS)
        .map(|(surface, _)| *surface)
        .collect::<Vec<_>>();
    let mut slots = kept
        .iter()
        .enumerate()
        .map(|(slot, surface)| (*surface, slot))
        .collect::<std::collections::BTreeMap<_, _>>();
    for (surface, _) in ranked.iter().skip(SURFACE_SLOTS) {
        let look = palette.look(*surface);
        let kept_looks = kept.iter().map(|candidate| palette.look(*candidate));
        let draws = |candidate: &mechanic_world::SurfaceLook| {
            candidate.tree_texture == look.tree_texture && candidate.texture == look.texture
        };
        let slot = kept_looks
            .clone()
            .position(|candidate| draws(&candidate))
            .or_else(|| {
                kept_looks
                    .clone()
                    .position(|candidate| candidate.material == look.material)
            })
            .or_else(|| {
                kept_looks
                    .clone()
                    .position(|candidate| candidate.texture == look.texture)
            })
            .unwrap_or(0);
        slots.insert(*surface, slot);
    }
    (kept, slots)
}

pub(super) fn terrain_chunk_mesh(
    chunk: &TerrainMeshChunk,
    indices: Vec<u32>,
    palette: &SurfacePalette,
) -> Mesh {
    let (kept, slots) = surface_slots(&chunk.surfaces, palette);
    let mut table = [u32::MAX; 4];
    for (slot, surface) in kept.iter().enumerate() {
        let word = &mut table[slot / 2];
        let shift = (slot % 2) * 16;
        *word = (*word & !(0xffff << shift)) | (u32::from(surface.0) << shift);
    }
    let (low, high): (Vec<[u8; 4]>, Vec<[u8; 4]>) = chunk
        .surfaces
        .iter()
        .map(|surface| {
            let slot = slots[surface];
            let mut weights = [[0_u8; 4]; 2];
            weights[slot / 4][slot % 4] = u8::MAX;
            (weights[0], weights[1])
        })
        .unzip();
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
    // Collision and queries read the chunk itself, so the uploaded mesh needs
    // no main-world copy.
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, chunk.vertices.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, chunk.normals.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, vertical_uvs);
    mesh.insert_attribute(
        ATTRIBUTE_TERRAIN_WEIGHTS_LOW,
        VertexAttributeValues::Unorm8x4(low),
    );
    mesh.insert_attribute(
        ATTRIBUTE_TERRAIN_WEIGHTS_HIGH,
        VertexAttributeValues::Unorm8x4(high),
    );
    mesh.insert_attribute(
        ATTRIBUTE_TERRAIN_SLOTS,
        VertexAttributeValues::Uint32x4(vec![table; chunk.vertices.len()]),
    );
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::render::render_resource::{Extent3d, TextureDimension};

    #[test]
    fn shared_terrain_source_can_be_used_again_after_leaving_world() {
        let mut images = Assets::<Image>::default();
        // Construction materials keep a strong handle to shared source maps
        // across world transitions.
        let construction_texture = images.add(Image::new_fill(
            Extent3d {
                width: TERRAIN_LAYER_EDGE,
                height: TERRAIN_LAYER_EDGE,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[128, 128, 128, 255],
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        ));
        for _ in 0..2 {
            let mut build = TerrainTextureBuild {
                sources: vec![[Some(construction_texture.clone()), None, None, None]],
                layers: Default::default(),
                targets: core::array::from_fn(|_| images.reserve_handle()),
                luma: [0.5; TextureSet::ALL.len()],
                surfaces: Handle::default(),
                tree_task: None,
                tree_targets: core::array::from_fn(|_| images.reserve_handle()),
                trees_done: true,
            };
            advance_terrain_textures(&mut build, &mut images).unwrap();
            assert_eq!(build.layers[0].len(), 1, "source must produce a layer");
            assert!(build.sources[0][0].is_none());
            assert!(
                images.get(&construction_texture).is_some(),
                "terrain must release its handle without deleting a shared image"
            );
            // Leaving the world drops the build before the next entry.
        }
    }

    #[test]
    fn grass_dirt_and_stone_surfaces_name_their_procedural_recipe() {
        for set in TextureSet::ALL {
            assert_eq!(
                procedural_recipe(set) != 0,
                matches!(
                    set,
                    TextureSet::Grass | TextureSet::Dirt | TextureSet::Stone
                ),
                "{set:?}"
            );
        }
        let palette = mechanic_world::TerrainField::new(mechanic_world::WorldSeed(42))
            .palette()
            .clone();
        let surfaces = terrain_surfaces(&palette, &[0.5; TextureSet::ALL.len()]);
        for (look, gpu) in palette.looks().iter().zip(&surfaces) {
            if look.tree_texture.is_none() {
                let recipe = f32::from(procedural_recipe(look.texture));
                assert!((gpu.shade.y - recipe).abs() < f32::EPSILON);
            }
        }
    }

    #[test]
    fn terrain_textures_are_ready_only_once_the_tree_arrays_are() {
        let palette = mechanic_world::TerrainField::new(mechanic_world::WorldSeed(42))
            .palette()
            .clone();
        let mut images = Assets::<Image>::default();
        // Blank ground layers finish at once; the trees take far longer.
        let mut build = TerrainTextureBuild {
            sources: vec![[None, None, None, None]; TextureSet::ALL.len()],
            layers: Default::default(),
            targets: core::array::from_fn(|_| images.reserve_handle()),
            luma: [0.5; TextureSet::ALL.len()],
            surfaces: Handle::default(),
            tree_task: Some(draw_tree_layers(palette.tree_textures().to_vec())),
            tree_targets: core::array::from_fn(|_| images.reserve_handle()),
            trees_done: false,
        };
        let started = std::time::Instant::now();
        let mut ready = false;
        while !ready && started.elapsed() < std::time::Duration::from_mins(2) {
            ready = advance_terrain_textures(&mut build, &mut images).unwrap();
            std::thread::yield_now();
        }
        assert!(ready, "terrain textures never finished");
        for target in &build.tree_targets {
            assert!(images.get(target).is_some(), "ready before the tree arrays");
        }
    }

    #[test]
    fn every_grown_species_gets_its_own_bark_and_leaf_layers() {
        let palette = mechanic_world::TerrainField::new(mechanic_world::WorldSeed(42))
            .palette()
            .clone();
        let trees = palette.tree_textures().len();
        assert!(trees >= 2, "{trees} tree textures in the default world");
        let mut images = Assets::<Image>::default();
        let mut build = TerrainTextureBuild {
            sources: Vec::new(),
            layers: Default::default(),
            targets: core::array::from_fn(|_| images.reserve_handle()),
            luma: [0.5; TextureSet::ALL.len()],
            surfaces: Handle::default(),
            tree_task: Some(draw_tree_layers(palette.tree_textures().to_vec())),
            tree_targets: core::array::from_fn(|_| images.reserve_handle()),
            trees_done: false,
        };
        let started = std::time::Instant::now();
        while !build.trees_done && started.elapsed() < std::time::Duration::from_mins(2) {
            advance_tree_textures(&mut build, &mut images).unwrap();
            std::thread::yield_now();
        }
        assert!(build.trees_done, "tree textures never finished");
        for target in &build.tree_targets {
            let array = images.get(target).expect("tree array published");
            assert_eq!(
                array.texture_descriptor.size.depth_or_array_layers as usize,
                trees
            );
        }
        // Each tree look samples its own layer, at the shared brightness.
        let surfaces = terrain_surfaces(&palette, &[0.5; TextureSet::ALL.len()]);
        let mut layers = palette
            .looks()
            .iter()
            .zip(&surfaces)
            .filter_map(|(look, gpu)| look.tree_texture.map(|tree| (tree, gpu)))
            .map(|(tree, gpu)| {
                assert!((gpu.shade.x - TREE_TEXTURE_LUMA).abs() < 1.0e-6);
                (gpu.params.x as u32, TREE_LAYER_BASE + u32::from(tree))
            })
            .collect::<Vec<_>>();
        layers.dedup();
        assert_eq!(layers.len(), trees);
        assert!(layers.iter().all(|(sampled, expected)| sampled == expected));
    }
}
