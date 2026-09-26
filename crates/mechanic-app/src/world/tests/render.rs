//! Real-renderer proof for each launch-only terrain experiment.

use bevy::{
    asset::RenderAssetUsages,
    camera::RenderTarget,
    prelude::*,
    render::{
        RenderApp, RenderPlugin,
        gpu_readback::{Readback, ReadbackComplete},
        pipelined_rendering::PipelinedRenderingPlugin,
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
        renderer::{RenderAdapterInfo, RenderDevice},
    },
    window::ExitCondition,
};

use crate::world::TerrainRenderMaterial;

#[derive(Resource, Default)]
struct Pixels(Vec<u8>);

fn render_frame(app: &mut App) {
    app.update();
    app.sub_app(RenderApp)
        .world()
        .resource::<RenderDevice>()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
}

#[test]
#[ignore = "requires a real GPU; run separately for each MECHANIC_RENDER_EXPERIMENT"]
#[expect(
    clippy::too_many_lines,
    reason = "keep offscreen setup and pixel proof in one fixture"
)]
fn terrain_experiment_renders_pixels_with_the_real_material() {
    let mode = crate::render_experiments::current();
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(AssetPlugin {
                file_path: format!("{}/assets", env!("CARGO_MANIFEST_DIR")),
                ..default()
            })
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .set(RenderPlugin {
                synchronous_pipeline_compilation: true,
                ..default()
            })
            .disable::<bevy::winit::WinitPlugin>()
            .disable::<PipelinedRenderingPlugin>(),
    )
    .add_plugins(MaterialPlugin::<TerrainRenderMaterial>::default())
    .init_resource::<Pixels>();
    app.finish();
    app.cleanup();
    eprintln!(
        "Terrain experiment {}: {:?}",
        mode.label(),
        app.sub_app(RenderApp)
            .world()
            .resource::<RenderAdapterInfo>()
            .0
    );
    let mut target = Image::new_target_texture(64, 64, TextureFormat::Rgba8UnormSrgb, None);
    target.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(target);
    app.world_mut()
        .spawn(Readback::texture(target.clone()))
        .observe(|event: On<ReadbackComplete>, mut pixels: ResMut<Pixels>| {
            pixels.0.clone_from(&event.data);
        });
    app.world_mut().spawn((
        Camera3d::default(),
        // Match the app: the default tonemapper requires LUTs we do not enable.
        bevy::core_pipeline::tonemapping::Tonemapping::SomewhatBoringDisplayTransform,
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        RenderTarget::Image(target.into()),
        mode.msaa(),
        Transform::from_xyz(0.0, 0.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    // Tonemapping can lift a black clear color above zero. Capture the empty
    // scene first so a clear-only frame cannot masquerade as a material draw.
    for _ in 0..12 {
        render_frame(&mut app);
    }
    let center = (32 * 64 + 32) * 4;
    let background = &app.world().resource::<Pixels>().0;
    assert_eq!(background.len(), 64 * 64 * 4);
    let background = background[center..center + 4].to_vec();
    eprintln!("Empty target pixel: {background:?}");
    let mut layer = |pixel: [u8; 4]| {
        let mut image = Image::new_fill(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 7,
            },
            TextureDimension::D2,
            &pixel,
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::default(),
        );
        image.texture_view_descriptor =
            Some(bevy::render::render_resource::TextureViewDescriptor {
                dimension: Some(bevy::render::render_resource::TextureViewDimension::D2Array),
                ..default()
            });
        app.world_mut().resource_mut::<Assets<Image>>().add(image)
    };
    let (base_color, normal, orm, tint_mask) = (
        layer([128, 128, 128, 255]),
        layer([128, 128, 255, 255]),
        layer([255, 200, 0, 255]),
        layer([255, 255, 255, 255]),
    );
    let palette = mechanic_world::TerrainField::new(mechanic_world::WorldSeed(1))
        .palette()
        .clone();
    let surfaces = app
        .world_mut()
        .resource_mut::<Assets<bevy::render::storage::ShaderBuffer>>()
        .add(crate::world::terrain_render::terrain_surface_buffer(
            &palette,
            &[0.5; mechanic_world::TextureSet::ALL.len()],
        ));
    let material = TerrainRenderMaterial {
        base_color,
        normal,
        orm,
        tint_mask,
        surfaces,
    };
    let material = app
        .world_mut()
        .resource_mut::<Assets<TerrainRenderMaterial>>()
        .add(material);
    let mut mesh = Mesh::from(Cuboid::default());
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_UV_1,
        vec![[0.0, 0.0]; mesh.count_vertices()],
    );
    let count = mesh.count_vertices();
    mesh.insert_attribute(
        crate::world::terrain_render::ATTRIBUTE_TERRAIN_WEIGHTS_LOW,
        bevy::mesh::VertexAttributeValues::Unorm8x4(vec![[255, 0, 0, 0]; count]),
    );
    mesh.insert_attribute(
        crate::world::terrain_render::ATTRIBUTE_TERRAIN_WEIGHTS_HIGH,
        bevy::mesh::VertexAttributeValues::Unorm8x4(vec![[0; 4]; count]),
    );
    mesh.insert_attribute(
        crate::world::terrain_render::ATTRIBUTE_TERRAIN_SLOTS,
        bevy::mesh::VertexAttributeValues::Uint32x4(vec![[0, u32::MAX, u32::MAX, u32::MAX]; count]),
    );
    let mesh = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
    app.world_mut()
        .spawn((Mesh3d(mesh), MeshMaterial3d(material)));
    app.world_mut().spawn((
        DirectionalLight {
            illuminance: 18_000.0,
            ..default()
        },
        Transform::from_xyz(0.0, 1.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    // Width 64 gives 256-byte aligned RGBA rows, so no row padding to strip.
    for _ in 0..60 {
        render_frame(&mut app);
    }
    let pixels = &app.world().resource::<Pixels>().0;
    assert_eq!(pixels.len(), 64 * 64 * 4);
    let pixel = &pixels[center..center + 4];
    eprintln!("Settled terrain pixel: {pixel:?}");
    assert!(
        pixel[..3]
            .iter()
            .zip(&background)
            .any(|(byte, clear)| byte.abs_diff(*clear) > 8),
        "terrain did not draw over the empty target"
    );
    assert!(
        pixel[1] > 8,
        "terrain is still using the magenta loading/error material"
    );
    if mode == crate::render_experiments::RenderExperiment::SimpleTerrain {
        assert!(
            pixel[1] > pixel[0] && pixel[0] > pixel[2],
            "expected the diagnostic green terrain material"
        );
    }
}
