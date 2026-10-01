//! Opt-in offscreen GPU captures. Player worlds are never loaded or saved.

mod performance;
mod scene;

use super::super::*;
use bevy::camera::RenderTarget;
use bevy::render::{
    RenderApp, RenderPlugin,
    gpu_readback::{Readback, ReadbackComplete},
    pipelined_rendering::PipelinedRenderingPlugin,
    render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
    renderer::{RenderAdapterInfo, RenderDevice},
};
use bevy::window::ExitCondition;
use std::time::Instant;

const WIDTH: u32 = 768;
const HEIGHT: u32 = 512;

#[derive(Resource, Default)]
struct Pixels(Vec<u8>);

fn frame(app: &mut App) {
    app.update();
    app.sub_app(RenderApp)
        .world()
        .resource::<RenderDevice>()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
}

fn timings(app: &mut App, readback: Entity) -> (f64, f64) {
    let request = app
        .world_mut()
        .entity_mut(readback)
        .take::<Readback>()
        .unwrap();
    for _ in 0..12 {
        frame(app);
    }
    let mut samples = Vec::new();
    for _ in 0..60 {
        let start = Instant::now();
        frame(app);
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    app.world_mut().entity_mut(readback).insert(request);
    samples.sort_by(f64::total_cmp);
    (samples.iter().sum::<f64>() / 60.0, samples[56])
}

fn save_image(app: &App, directory: &std::path::Path, name: &str) {
    let pixels = &app.world().resource::<Pixels>().0;
    assert_eq!(pixels.len(), (WIDTH * HEIGHT * 4) as usize);
    Image::new(
        Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixels.clone(),
        TextureFormat::Rgba8UnormSrgb,
        default(),
    )
    .try_into_dynamic()
    .unwrap()
    .save(directory.join(format!("{name}.png")))
    .unwrap();
}

#[test]
#[ignore = "real GPU atmosphere capture and synchronized frame timing comparison"]
#[expect(
    clippy::too_many_lines,
    reason = "self-contained offscreen GPU fixture"
)]
fn atmosphere_captures_four_times_and_reports_cost() {
    let mut app = App::new();
    let mut dev = DevTools::default();
    dev.cycle_paused = true;
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
    .add_plugins((
        SkyPlugin,
        MaterialPlugin::<crate::world::TerrainRenderMaterial>::default(),
        MaterialPlugin::<crate::world::WaterRenderMaterial>::default(),
        crate::render::environment::OneShotEnvironmentMapPlugin,
    ))
    .insert_resource(ClearColor(Color::srgb_u8(69, 88, 102)))
    .init_resource::<Pixels>()
    .init_resource::<WorldRuntime>()
    .insert_resource(WorldListState::empty_capture_garage())
    .init_resource::<PauseMenuState>()
    .insert_resource(dev)
    .insert_resource(State::new(AppSpace::Garage))
    .insert_resource(GlobalAmbientLight {
        color: Color::srgb_u8(43, 60, 76),
        brightness: 20.0,
        ..default()
    })
    .configure_sets(Update, (FrameSet::SkyClock, FrameSet::Sky).chain());
    performance::install(&mut app);
    app.finish();
    app.cleanup();
    let adapter = format!(
        "{:?}",
        app.sub_app(RenderApp)
            .world()
            .resource::<RenderAdapterInfo>()
            .0
    );
    let directory = std::env::temp_dir().join(format!("mechanic-sky-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    eprintln!(
        "Sky adapter: {adapter}\nSky captures: {}",
        directory.display()
    );
    let mut target = Image::new_target_texture(WIDTH, HEIGHT, TextureFormat::Rgba8UnormSrgb, None);
    target.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(target);
    let readback = app
        .world_mut()
        .spawn(Readback::texture(target.clone()))
        .observe(|event: On<ReadbackComplete>, mut pixels: ResMut<Pixels>| {
            pixels.0.clone_from(&event.data);
        })
        .id();
    let environment = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(crate::render::environment::sky_cubemap(64));
    let camera = app
        .world_mut()
        .spawn((
            Camera3d::default(),
            MainCamera,
            bevy::core_pipeline::tonemapping::Tonemapping::SomewhatBoringDisplayTransform,
            RenderTarget::Image(target.into()),
            Msaa::Sample4,
            Exposure::OVERCAST,
            crate::tool_fx::bloom(),
            DistanceFog {
                color: Color::srgb_u8(69, 88, 102),
                falloff: FogFalloff::Exponential { density: 0.0022 },
                ..default()
            },
            StaticEnvironmentMap,
            GeneratedEnvironmentMapLight {
                environment_map: environment,
                intensity: crate::render::environment::SKY_ENVIRONMENT_INTENSITY,
                ..default()
            },
            Transform::from_xyz(0.0, 3.0, 10.0).looking_at(Vec3::new(0.0, 2.5, 0.0), Vec3::Y),
        ))
        .id();
    app.world_mut().spawn((
        Sun,
        DirectionalLight {
            color: Color::srgb_u8(218, 204, 190),
            illuminance: 18_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, -0.55, 0.0)),
    ));
    scene::ground(&mut app);
    let sphere = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Sphere::new(1.0).mesh().uv(32, 16));
    for (x, metallic, roughness) in [(-2.5, 0.0, 0.8), (0.0, 1.0, 0.1), (2.5, 1.0, 0.45)] {
        let material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                base_color: Color::srgb(0.55, 0.55, 0.55),
                metallic,
                perceptual_roughness: roughness,
                ..default()
            });
        app.world_mut().spawn((
            Mesh3d(sphere.clone()),
            MeshMaterial3d(material),
            Transform::from_xyz(x, 1.0, 0.0),
        ));
    }
    scene::water(&mut app);
    for _ in 0..40 {
        frame(&mut app);
    }
    save_image(&app, &directory, "baseline");
    let baseline = timings(&mut app, readback);
    app.insert_resource(State::new(AppSpace::World));
    for mut light in app
        .world_mut()
        .query_filtered::<&mut DirectionalLight, With<Sun>>()
        .iter_mut(app.world_mut())
    {
        light.color = Color::WHITE;
    }
    app.world_mut().entity_mut(camera).insert(DistanceFog {
        color: Color::NONE,
        falloff: FogFalloff::Exponential { density: 0.0 },
        ..default()
    });
    let mut reports = Vec::new();
    let mut previous = Vec::new();
    for (name, hours) in [
        ("sunrise", 6.0),
        ("noon", 12.0),
        ("sunset", 18.0),
        ("midnight", 0.0),
    ] {
        app.world_mut().resource_mut::<SkyState>().fixed_seconds = Some(hours * 3600.0);
        for _ in 0..40 {
            frame(&mut app);
        }
        save_image(&app, &directory, name);
        let pixels = app.world().resource::<Pixels>().0.clone();
        assert_ne!(
            pixels, previous,
            "solar time must change the rendered image"
        );
        previous = pixels;
        let original = *app.world().entity(camera).get::<Transform>().unwrap();
        let direction = celestial_rotation(hours * 3600.0)
            * if name == "midnight" {
                Vec3::NEG_X
            } else {
                Vec3::X
            };
        app.world_mut()
            .entity_mut(camera)
            .insert(original.looking_to(direction, Vec3::Y));
        for _ in 0..8 {
            frame(&mut app);
        }
        save_image(&app, &directory, &format!("{name}-disk"));
        app.world_mut().entity_mut(camera).insert(original);
        let (mean, p95) = timings(&mut app, readback);
        reports.push(serde_json::json!({"time": name, "mean_ms": mean, "p95_ms": p95}));
        assert!(
            app.world()
                .entity(camera)
                .contains::<GeneratedEnvironmentMapLight>()
        );
    }
    let report = serde_json::json!({"adapter": adapter, "resolution": [WIDTH, HEIGHT],
        "measurement": "synchronized offscreen CPU+GPU frame latency, readback disabled; not interactive throughput or isolated GPU pass time",
        "baseline": {"mean_ms": baseline.0, "p95_ms": baseline.1}, "atmosphere": reports});
    std::fs::write(
        directory.join("timings.json"),
        serde_json::to_string_pretty(&report).unwrap(),
    )
    .unwrap();
    eprintln!("{report}");
    performance::compare(&mut app, camera, readback, &directory);
}
