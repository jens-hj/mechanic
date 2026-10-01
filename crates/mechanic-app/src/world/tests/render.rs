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
    let wetness = crate::world::terrain_render::dry_ground(
        &mut app.world_mut().resource_mut::<Assets<Image>>(),
    );
    let material = TerrainRenderMaterial {
        base_color,
        normal,
        orm,
        tint_mask,
        surfaces,
        wetness,
        wet_window: Vec4::ZERO,
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

/// Water over pale ground, looked at from three metres up and away.
#[derive(Clone, Copy)]
struct WaterScene {
    /// Depth of the water, in metres.
    depth: f32,
    /// Its current, in m/s.
    flow: [f32; 2],
    /// How cloudy with sediment it is, or none for water that carries none.
    murk: Option<f32>,
    /// How white it churns, or none for water that carries no churn.
    churn: Option<f32>,
    /// Pixels along each edge of the picture.
    size: u32,
}

impl Default for WaterScene {
    fn default() -> Self {
        Self {
            depth: 2.0,
            flow: [0.3, 0.0],
            murk: None,
            churn: None,
            size: 64,
        }
    }
}

/// The pixel at the middle of water two metres deep over pale ground, as
/// cloudy with sediment as `murk` says, or clear without it.
fn water_pixel(murk: Option<f32>) -> Vec<u8> {
    let pixels = water_frame(WaterScene {
        murk,
        ..WaterScene::default()
    });
    let center = (32 * 64 + 32) * 4;
    pixels[center..center + 4].to_vec()
}

/// The luminance of each pixel in the middle half of a picture of `scene`,
/// from 0 to 1.
fn water_luminance(scene: WaterScene) -> Vec<f32> {
    let size = scene.size as usize;
    let pixels = water_frame(scene);
    let mut luminance = Vec::new();
    for y in size / 4..size * 3 / 4 {
        for x in size / 4..size * 3 / 4 {
            let pixel = &pixels[(y * size + x) * 4..][..3];
            luminance.push(
                (0.2126 * f32::from(pixel[0])
                    + 0.7152 * f32::from(pixel[1])
                    + 0.0722 * f32::from(pixel[2]))
                    / 255.0,
            );
        }
    }
    luminance
}

fn mean(values: &[f32]) -> f32 {
    #[expect(clippy::cast_precision_loss, reason = "a few thousand pixels")]
    let count = values.len() as f32;
    values.iter().sum::<f32>() / count
}

/// How grainy a picture of `scene` is: the mean difference in luminance
/// between pixels side by side in the middle half of it, so the light
/// falling off across the picture counts for nothing.
fn grain(scene: WaterScene) -> f32 {
    let width = scene.size as usize / 2;
    let luminance = water_luminance(scene);
    let steps = luminance
        .chunks(width)
        .flat_map(|row| row.windows(2).map(|pair| (pair[1] - pair[0]).abs()))
        .collect::<Vec<_>>();
    mean(&steps)
}

/// The water surface of `scene`: a flat 8 m square.
fn water_plane(scene: WaterScene) -> Mesh {
    use crate::world::water_render::{ATTRIBUTE_CHURN, ATTRIBUTE_MURK, ATTRIBUTE_WATER};

    let mut water = Mesh::from(Plane3d::default().mesh().size(8.0, 8.0));
    let count = water.count_vertices();
    water.insert_attribute(
        ATTRIBUTE_WATER,
        bevy::mesh::VertexAttributeValues::Float32x3(vec![
            [
                scene.depth,
                scene.flow[0],
                scene.flow[1]
            ];
            count
        ]),
    );
    if let Some(murk) = scene.murk {
        water.insert_attribute(
            ATTRIBUTE_MURK,
            bevy::mesh::VertexAttributeValues::Float32(vec![murk; count]),
        );
    }
    if let Some(churn) = scene.churn {
        water.insert_attribute(
            ATTRIBUTE_CHURN,
            bevy::mesh::VertexAttributeValues::Float32(vec![churn; count]),
        );
    }
    water.remove_attribute(Mesh::ATTRIBUTE_UV_0);
    water
}

/// A picture of `scene`, as RGBA bytes row by row.
fn water_frame(scene: WaterScene) -> Vec<u8> {
    use crate::world::WaterRenderMaterial;

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
    .add_plugins(MaterialPlugin::<WaterRenderMaterial>::default())
    .init_resource::<Pixels>();
    app.finish();
    app.cleanup();
    let mut target =
        Image::new_target_texture(scene.size, scene.size, TextureFormat::Rgba8UnormSrgb, None);
    target.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(target);
    app.world_mut()
        .spawn(Readback::texture(target.clone()))
        .observe(|event: On<ReadbackComplete>, mut pixels: ResMut<Pixels>| {
            pixels.0.clone_from(&event.data);
        });
    app.world_mut().spawn((
        Camera3d::default(),
        bevy::core_pipeline::tonemapping::Tonemapping::SomewhatBoringDisplayTransform,
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        RenderTarget::Image(target.into()),
        Transform::from_xyz(0.0, 3.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    // Pale ground under the water.
    let ground = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::from(Color::srgb(0.8, 0.75, 0.6)));
    let plane = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Plane3d::default().mesh().size(8.0, 8.0));
    app.world_mut().spawn((
        Mesh3d(plane),
        MeshMaterial3d(ground),
        Transform::from_xyz(0.0, -2.0, 0.0),
    ));
    let water = water_plane(scene);
    let water = app.world_mut().resource_mut::<Assets<Mesh>>().add(water);
    let material =
        WaterRenderMaterial::with_noise(&mut app.world_mut().resource_mut::<Assets<Image>>());
    let material = app
        .world_mut()
        .resource_mut::<Assets<WaterRenderMaterial>>()
        .add(material);
    app.world_mut()
        .spawn((Mesh3d(water), MeshMaterial3d(material)));
    app.world_mut().spawn((
        DirectionalLight {
            illuminance: 18_000.0,
            ..default()
        },
        Transform::from_xyz(0.0, 1.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    for _ in 0..60 {
        render_frame(&mut app);
    }
    app.world_mut()
        .remove_resource::<Pixels>()
        .unwrap_or_default()
        .0
}

#[test]
#[ignore = "requires a real GPU"]
fn water_draws_a_translucent_blue_surface_over_the_ground() {
    let pixel = water_pixel(None);
    eprintln!("Water pixel: {pixel:?}");
    assert!(pixel[1] > 8, "water is still the magenta error material");
    assert!(
        pixel[2] > pixel[0],
        "water over pale ground should read blue: {pixel:?}"
    );
}

#[test]
#[ignore = "requires a real GPU"]
fn water_thick_with_sediment_draws_silty_brown() {
    let clear = water_pixel(Some(0.0));
    let muddy = water_pixel(Some(1.0));
    eprintln!("Clear {clear:?}, muddy {muddy:?}");
    assert!(
        clear[2] > clear[0],
        "clear water should read blue: {clear:?}"
    );
    assert!(
        muddy[0] > muddy[2] && muddy[1] > muddy[2],
        "muddy water should read brown: {muddy:?}"
    );
}

#[test]
#[ignore = "requires a real GPU"]
fn fast_shallow_water_breaks_white() {
    // A breach: water 8 cm deep at 1.5 m/s, against the same water at a
    // walking pace's tenth.
    let shallow = |flow| WaterScene {
        depth: 0.08,
        flow: [flow, 0.0],
        size: 256,
        ..WaterScene::default()
    };
    let slow = mean(&water_luminance(shallow(0.15)));
    let fast = mean(&water_luminance(shallow(1.5)));
    eprintln!("Slow {slow:.3}, fast {fast:.3}");
    // The pale ground already shows bright through shallow water.
    assert!(
        fast > slow + 0.04,
        "rapids should break white: {fast:.3} against {slow:.3}"
    );
}

#[test]
#[ignore = "requires a real GPU"]
fn churned_water_draws_foam() {
    let pool = |churn| WaterScene {
        depth: 0.5,
        flow: [0.0, 0.0],
        churn: Some(churn),
        size: 256,
        ..WaterScene::default()
    };
    let calm = mean(&water_luminance(pool(0.0)));
    let churned = mean(&water_luminance(pool(1.0)));
    eprintln!("Calm {calm:.3}, churned {churned:.3}");
    assert!(
        churned > calm + 0.2,
        "churned water should foam white: {churned:.3} against {calm:.3}"
    );
}

#[test]
#[ignore = "requires a real GPU"]
fn still_shallow_water_is_smoother_than_running_water() {
    let pond = |flow| WaterScene {
        depth: 0.1,
        flow: [flow, 0.0],
        size: 256,
        ..WaterScene::default()
    };
    let still = grain(pond(0.0));
    let running = grain(pond(0.5));
    eprintln!("Still {still:.4}, running {running:.4}");
    assert!(
        running > 2.0 * still,
        "running water should wrinkle where still water lies glassy: {running:.4} against {still:.4}"
    );
}
