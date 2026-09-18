//! Paired real-material GPU timings and pixel comparison, with a frozen reference.

use bevy::{
    camera::RenderTarget,
    mesh::Indices,
    prelude::*,
    render::{
        RenderApp, RenderPlugin,
        gpu_readback::{Readback, ReadbackComplete},
        pipelined_rendering::PipelinedRenderingPlugin,
        render_resource::{
            Extent3d, PrimitiveTopology, TextureDimension, TextureFormat, TextureUsages,
        },
        renderer::{RenderAdapterInfo, RenderDevice},
    },
    shader::Shader,
    window::ExitCondition,
};

use crate::render_diagnostics::{ProfiledCamera, RenderTimings, RenderTimingsPlugin};
use crate::world::{TerrainRenderMaterial, generate_rgba8_mip_chain, terrain_render_material};

const WIDTH: u32 = 4096;
const HEIGHT: u32 = 2524;
const SHADER_PATH: &str = "shaders/terrain_material.wgsl";
const REFERENCE: &str = include_str!("../fixtures/terrain_material_reference.wgsl");
const CANDIDATE: &str = include_str!("../../../assets/shaders/terrain_material.wgsl");

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

fn fixture() -> (App, Handle<Shader>, Entity, Entity) {
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
    .add_plugins((
        MaterialPlugin::<TerrainRenderMaterial>::default(),
        RenderTimingsPlugin,
    ))
    .init_resource::<Pixels>();
    app.finish();
    app.cleanup();
    eprintln!(
        "Terrain shader comparison adapter: {:?}",
        app.sub_app(RenderApp)
            .world()
            .resource::<RenderAdapterInfo>()
            .0
    );
    let (material, mut pending) = terrain_render_material(app.world().resource::<AssetServer>());
    let shader = app.world().resource::<AssetServer>().load(SHADER_PATH);
    for _ in 0..600 {
        frame(&mut app);
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        pending.retain(|handle| {
            if let Some(mut image) = images.get_mut(handle) {
                generate_rgba8_mip_chain(&mut image).unwrap();
                false
            } else {
                true
            }
        });
        if pending.is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(pending.is_empty(), "authored terrain textures did not load");
    let material = app
        .world_mut()
        .resource_mut::<Assets<TerrainRenderMaterial>>()
        .add(material);
    let mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(terrain_patch(false, 33));
    let terrain = app
        .world_mut()
        .spawn((Mesh3d(mesh), MeshMaterial3d(material)))
        .id();
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
    app.world_mut().spawn((
        Camera3d::default(),
        bevy::core_pipeline::tonemapping::Tonemapping::SomewhatBoringDisplayTransform,
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        RenderTarget::Image(target.into()),
        ProfiledCamera,
        Msaa::Sample4,
        Transform::from_xyz(0.0, 4.0, 8.0).looking_at(Vec3::new(0.0, 0.0, -8.0), Vec3::Y),
    ));
    app.world_mut().spawn((
        DirectionalLight {
            illuminance: 18_000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, -0.55, 0.0)),
    ));
    app.world().resource::<RenderTimings>().set_enabled(true);
    (app, shader, terrain, readback)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "small, bounded procedural fixture grid"
)]
fn terrain_patch(blended: bool, side: u32) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uv = Vec::new();
    let mut uv1 = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();
    for z in 0..side {
        for x in 0..side {
            let px = x as f32 / (side - 1) as f32 * 96.0 - 48.0;
            let pz = 8.0 - z as f32 / (side - 1) as f32 * 192.0;
            let py = if blended { (px * 0.3).sin() * 2.0 } else { 0.0 };
            let slope = if blended { (px * 0.3).cos() * 0.6 } else { 0.0 };
            positions.push([px, py, pz]);
            normals.push(Vec3::new(-slope, 1.0, 0.0).normalize().to_array());
            uv.push([px / 1.5, pz / 1.5]);
            let channel = if blended { x % 6 } else { 0 };
            let mut color = [0.0; 4];
            if channel < 4 {
                color[channel as usize] = 1.0;
            }
            colors.push(color);
            uv1.push([py / 1.5, if channel == 4 { 1.0 } else { 0.0 }]);
            if x + 1 < side && z + 1 < side {
                let a = z * side + x;
                indices.extend_from_slice(&[a, a + 1, a + side, a + 1, a + side + 1, a + side]);
            }
        }
    }
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, uv1);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn measure(
    app: &mut App,
    shader: &Handle<Shader>,
    source: &str,
    readback: Entity,
) -> (f64, Vec<u8>) {
    app.world_mut()
        .resource_mut::<Assets<Shader>>()
        .insert(
            shader.id(),
            Shader::from_wgsl(source.to_owned(), SHADER_PATH),
        )
        .unwrap();
    // Exclude compilation, shader replacement and old async samples from timing.
    for _ in 0..30 {
        frame(app);
    }
    let pixels = app.world().resource::<Pixels>().0.clone();
    let request = app
        .world_mut()
        .entity_mut(readback)
        .take::<Readback>()
        .unwrap();
    for _ in 0..6 {
        frame(app);
    }
    let mut samples = Vec::new();
    for _ in 0..80 {
        frame(app);
        let snapshot = app.world().resource::<RenderTimings>().snapshot();
        samples.push(
            snapshot
                .breakdown
                .opaque_ms
                .expect("opaque GPU timestamps unavailable"),
        );
    }
    app.world_mut().entity_mut(readback).insert(request);
    samples.sort_by(f64::total_cmp);
    (samples[samples.len() / 2], pixels)
}

fn save_pixels(pixels: &[u8], name: &str) {
    let directory =
        std::env::temp_dir().join(format!("mechanic-terrain-shader-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join(format!("{name}.png"));
    Image::new(
        Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixels.to_vec(),
        TextureFormat::Rgba8UnormSrgb,
        default(),
    )
    .try_into_dynamic()
    .unwrap()
    .save(&path)
    .unwrap();
    eprintln!("Terrain comparison image: {}", path.display());
}

#[test]
#[ignore = "real GPU paired benchmark; run in release with baseline render experiment"]
fn terrain_shader_preserves_pixels_and_measures_gpu_cost() {
    assert_eq!(
        crate::render_experiments::current(),
        crate::render_experiments::RenderExperiment::Baseline
    );
    // Explicit paths let each experiment compare against its actual starting
    // shader without replacing the frozen reference for earlier work.
    let source = |variable, fallback: &str| {
        std::env::var_os(variable).map_or_else(
            || fallback.to_owned(),
            |path| std::fs::read_to_string(path).expect("benchmark shader source"),
        )
    };
    let reference_source = source("MECHANIC_TERRAIN_REFERENCE_SHADER", REFERENCE);
    let candidate_source = source("MECHANIC_TERRAIN_CANDIDATE_SHADER", CANDIDATE);
    let (mut app, shader, terrain, readback) = fixture();
    for (blended, side) in [(false, 33), (false, 513), (true, 33)] {
        let mesh = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(terrain_patch(blended, side));
        app.world_mut().entity_mut(terrain).insert(Mesh3d(mesh));
        // A/B/B/A order helps expose warm-up or clock drift.
        let (before, reference) = measure(&mut app, &shader, &reference_source, readback);
        let (after, candidate) = measure(&mut app, &shader, &candidate_source, readback);
        let (after_repeat, _) = measure(&mut app, &shader, &candidate_source, readback);
        let (before_repeat, _) = measure(&mut app, &shader, &reference_source, readback);
        assert_eq!(reference.len(), (WIDTH * HEIGHT * 4) as usize);
        assert_eq!(candidate.len(), reference.len());
        assert!(
            reference.chunks_exact(4).filter(|p| p[1] > 8).count() > reference.len() / 16,
            "reference terrain must cover a substantial part of the image"
        );
        let differences: Vec<_> = reference
            .iter()
            .zip(&candidate)
            .map(|(a, b)| a.abs_diff(*b))
            .collect();
        let max = differences.iter().copied().max().unwrap();
        let changed = differences
            .iter()
            .filter(|difference| **difference > 2)
            .count();
        eprintln!(
            "Terrain blended={blended} grid={side} {WIDTH}x{HEIGHT} MSAA4: opaque median reference={before:.3}/{before_repeat:.3} ms candidate={after:.3}/{after_repeat:.3} ms; max channel delta={max}, channels over 2={changed}/{}",
            differences.len()
        );
        save_pixels(&reference, &format!("reference-{blended}-{side}"));
        save_pixels(&candidate, &format!("candidate-{blended}-{side}"));
        assert!(
            max <= 2,
            "terrain appearance changed: maximum channel delta {max}"
        );
    }
}

#[test]
#[ignore = "native GPU pixel check; enable MECHANIC_PERF_TERRAIN_PASSES=1 and MECHANIC_PERF_CAPTURE_DIR"]
fn terrain_pass_partition_preserves_pixels_and_reports_separate_timing() {
    assert!(crate::render_diagnostics::terrain_passes_enabled());
    let (mut app, _, _, _) = fixture();
    let cube = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Cuboid::new(2.0, 2.0, 2.0));
    let material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::srgb(0.7, 0.15, 0.1),
            ..default()
        });
    app.world_mut().spawn((
        Mesh3d(cube),
        MeshMaterial3d(material),
        Transform::from_xyz(0.0, 1.0, -5.0),
    ));
    for _ in 0..40 {
        frame(&mut app);
    }
    let split = app.world().resource::<Pixels>().0.clone();
    let timing = app.world().resource::<RenderTimings>().snapshot();
    assert!(timing.breakdown.terrain_ms.is_some_and(|ms| ms > 0.0));
    assert!(timing.breakdown.opaque_other_ms.is_some_and(|ms| ms > 0.0));
    assert_eq!(timing.breakdown.opaque_ms, None);
    app.sub_app_mut(RenderApp)
        .world_mut()
        .remove_resource::<bevy::core_pipeline::core_3d::OpaquePassPartition>()
        .unwrap();
    for _ in 0..30 {
        frame(&mut app);
    }
    let normal = &app.world().resource::<Pixels>().0;
    assert_eq!(split.len(), (WIDTH * HEIGHT * 4) as usize);
    assert_eq!(split.len(), normal.len());
    let changed = split.iter().zip(normal).filter(|(a, b)| a != b).count();
    eprintln!(
        "Terrain pass partition pixel differences: {changed}/{}",
        split.len()
    );
    assert_eq!(
        &split, normal,
        "partition must preserve pixels, including other opaque draws"
    );
    assert!(
        app.world()
            .resource::<RenderTimings>()
            .snapshot()
            .breakdown
            .opaque_ms
            .is_some()
    );
}
