//! Terrain materials, texture mip chains, and chunk meshes.

use super::brush::BrushPreview;
use super::{
    AlphaMode, Asset, AssetServer, Assets, Color, Commands, Component, DirectionalLight, EulerRot,
    Handle, Image, LinearRgba, Material, Mesh, Mesh3d, MeshMaterial3d, Meshable, Name, Quat,
    Reflect, ResMut, Result, Sphere, StandardMaterial, String, ToOwned, ToString, Transform, Vec,
    Visibility, WorldDiagnostics, WorldOwned, WorldRuntime, default, format, vec,
};
use bevy::asset::RenderAssetUsages;
use bevy::mesh::Indices;
use bevy::render::render_resource::{AsBindGroup, Face, PrimitiveTopology, TextureFormat};
use bevy::shader::ShaderRef;
use mechanic_world::{TerrainMeshChunk, TerrainSpatialIndex, TerrainStreamer};

#[derive(Component)]
pub(super) struct TerrainNodeRender;

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub(crate) struct TerrainRenderMaterial {
    #[texture(0)]
    #[sampler(15)]
    pub(super) grass_base_color: Handle<Image>,
    #[texture(1)]
    pub(super) dirt_base_color: Handle<Image>,
    #[texture(2)]
    pub(super) stone_base_color: Handle<Image>,
    #[texture(3)]
    pub(super) sand_base_color: Handle<Image>,
    #[texture(4)]
    pub(super) iron_base_color: Handle<Image>,
    #[texture(5)]
    pub(super) graphite_base_color: Handle<Image>,
    #[texture(6)]
    pub(super) grass_normal: Handle<Image>,
    #[texture(7)]
    pub(super) dirt_normal: Handle<Image>,
    #[texture(8)]
    pub(super) stone_normal: Handle<Image>,
    #[texture(9)]
    pub(super) grass_orm: Handle<Image>,
    #[texture(10)]
    pub(super) dirt_orm: Handle<Image>,
    #[texture(11)]
    pub(super) stone_orm: Handle<Image>,
    #[texture(12)]
    pub(super) sand_orm: Handle<Image>,
    #[texture(13)]
    pub(super) iron_orm: Handle<Image>,
    #[texture(14)]
    pub(super) graphite_orm: Handle<Image>,
}

impl Material for TerrainRenderMaterial {
    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        _layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        descriptor.label = Some(
            format!(
                "mechanic_terrain:{}",
                descriptor.label.as_deref().unwrap_or_default()
            )
            .into(),
        );
        Ok(())
    }

    fn fragment_shader() -> ShaderRef {
        crate::render_experiments::current().terrain_shader().into()
    }
}

pub(super) fn spawn_world_terrain(
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
    runtime.player_terrain_ready = false;
    runtime.selection_focus = None;
    runtime.selected_terrain_revision = u64::MAX;
    let (terrain_material, pending_mips) = terrain_render_material(asset_server);
    runtime.terrain_texture_mips_pending = pending_mips;
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

pub(super) fn terrain_render_material(
    asset_server: &AssetServer,
) -> (TerrainRenderMaterial, Vec<Handle<Image>>) {
    let texture = |path: &'static str, is_srgb: bool| {
        asset_server
            .load_builder()
            .with_settings(move |settings: &mut bevy::image::ImageLoaderSettings| {
                crate::render::materials::configure_repeating_texture(settings, is_srgb);
            })
            .load(path)
    };
    let grass_base_color = texture("terrain/grass/grass_base_color.png", true);
    let dirt_base_color = texture("terrain/dirt/dirt_base_color.png", true);
    let stone_base_color = texture("terrain/stone/stone_base_color.png", true);
    let sand_base_color = texture("materials/sand/sand_base_color.png", true);
    let iron_base_color = texture("materials/iron/iron_base_color.png", true);
    let graphite_base_color = texture("materials/graphite/graphite_base_color.png", true);
    let grass_normal = texture("terrain/grass/grass_normal.png", false);
    let dirt_normal = texture("terrain/dirt/dirt_normal.png", false);
    let stone_normal = texture("terrain/stone/stone_normal.png", false);
    let grass_orm = texture("terrain/grass/grass_orm.png", false);
    let dirt_orm = texture("terrain/dirt/dirt_orm.png", false);
    let stone_orm = texture("terrain/stone/stone_orm.png", false);
    let sand_orm = texture("materials/sand/sand_orm.png", false);
    let iron_orm = texture("materials/iron/iron_orm.png", false);
    let graphite_orm = texture("materials/graphite/graphite_orm.png", false);
    let pending_mips = vec![
        grass_base_color.clone(),
        dirt_base_color.clone(),
        stone_base_color.clone(),
        sand_base_color.clone(),
        iron_base_color.clone(),
        graphite_base_color.clone(),
        grass_normal.clone(),
        dirt_normal.clone(),
        stone_normal.clone(),
        grass_orm.clone(),
        dirt_orm.clone(),
        stone_orm.clone(),
        sand_orm.clone(),
        iron_orm.clone(),
        graphite_orm.clone(),
    ];
    let material = TerrainRenderMaterial {
        grass_base_color,
        dirt_base_color,
        stone_base_color,
        sand_base_color,
        iron_base_color,
        graphite_base_color,
        grass_normal,
        dirt_normal,
        stone_normal,
        grass_orm,
        dirt_orm,
        stone_orm,
        sand_orm,
        iron_orm,
        graphite_orm,
    };
    (material, pending_mips)
}

pub(super) fn prepare_terrain_texture_mips(
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

pub(super) fn terrain_chunk_mesh(chunk: &TerrainMeshChunk, indices: Vec<u32>) -> Mesh {
    let colors = chunk
        .material_weights
        .iter()
        .copied()
        .map(|weights| [weights[0], weights[1], weights[2], weights[3]])
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
        .zip(&chunk.material_weights)
        .map(|(position, weights)| {
            [
                (chunk.origin.0.y + f64::from(position[1])) as f32 / 1.5,
                weights[4],
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
