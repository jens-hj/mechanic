//! Opt-in screenshots of normally generated, streamed terrain in an isolated world.

use super::{AppSpace, WorldDiagnostics, WorldListState, WorldRuntime};
use crate::{
    camera::{EYE_HEIGHT, MainCamera, PlayerState},
    dev_tools::{DevMode, DevTools},
    schedule::FrameSet,
};
use bevy::math::DVec3;
use bevy::{
    app::AppExit,
    camera::RenderTarget,
    input::{
        InputSystems,
        mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    },
    prelude::*,
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
    },
    winit::WinitSettings,
};
use mechanic_world::{KinematicCapsule, WorldPosition};
use serde::Deserialize;
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

#[derive(Deserialize)]
struct CaptureConfig {
    seed: u64,
    directory: PathBuf,
    views: Vec<View>,
}

#[derive(Deserialize)]
struct View {
    name: String,
    eye: [f64; 3],
    target: [f64; 3],
}

#[derive(Resource)]
struct Capture {
    config: CaptureConfig,
    view: usize,
    requested: bool,
    readback: bool,
    settled: u32,
    started: Instant,
    target: Option<Handle<Image>>,
    reported: u64,
}

pub(super) fn install(app: &mut App) {
    let Some(path) = crate::env::raw(crate::env::TERRAIN_CAPTURE) else {
        return;
    };
    let config: CaptureConfig =
        serde_json::from_slice(&std::fs::read(path).expect("read terrain capture config"))
            .expect("parse terrain capture config");
    assert!(!config.views.is_empty(), "terrain capture needs views");
    for view in &config.views {
        assert!(
            DVec3::from_array(view.eye).distance_squared(DVec3::from_array(view.target)) > 1.0e-12
        );
        assert!(view.eye.iter().chain(&view.target).all(|v| v.is_finite()));
        assert!(
            view.name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-')
        );
    }
    std::fs::create_dir_all(&config.directory).expect("create terrain capture directory");
    app.insert_resource(WinitSettings::continuous())
        .insert_resource(Capture {
            config,
            view: 0,
            requested: false,
            readback: false,
            settled: 0,
            started: Instant::now(),
            target: None,
            reported: 0,
        })
        .add_systems(First, stop_repeated_readback)
        .add_systems(PreUpdate, suppress_input.after(InputSystems))
        .add_systems(Update, prepare_target.before(FrameSet::Camera))
        .add_systems(
            Update,
            position.after(FrameSet::WorldList).before(FrameSet::Camera),
        )
        .add_systems(Update, aim.after(FrameSet::Camera).before(FrameSet::Hover))
        .add_systems(Last, capture);
}

// Read the application cameras through an owned texture instead of relying
// on capture of a background window surface.
const CAPTURE_WIDTH: u32 = 1280;
const CAPTURE_HEIGHT: u32 = 720;

#[derive(Component)]
struct CaptureCamera;

fn prepare_target(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut capture: ResMut<Capture>,
    cameras: Query<Entity, (With<Camera>, Without<CaptureCamera>)>,
) {
    let target = capture.target.get_or_insert_with(|| {
        let mut image = Image::new_target_texture(
            CAPTURE_WIDTH,
            CAPTURE_HEIGHT,
            TextureFormat::Rgba8UnormSrgb,
            None,
        );
        image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        images.add(image)
    });
    for entity in &cameras {
        commands
            .entity(entity)
            .insert((RenderTarget::Image(target.clone().into()), CaptureCamera));
    }
}

// Readback normally copies every frame. One request is enough; keep its entity
// for the completion observer while the submitted GPU copy finishes.
#[derive(Component)]
struct CaptureReadback;

fn stop_repeated_readback(
    mut commands: Commands,
    pending: Query<Entity, (With<Readback>, With<CaptureReadback>)>,
) {
    for entity in &pending {
        commands.entity(entity).remove::<Readback>();
    }
}

fn suppress_input(
    mut keyboard: ResMut<ButtonInput<KeyCode>>,
    mut mouse: ResMut<ButtonInput<MouseButton>>,
    mut motion: ResMut<AccumulatedMouseMotion>,
    mut scroll: ResMut<AccumulatedMouseScroll>,
    mut selection: ResMut<crate::hotbar::SelectedTool>,
) {
    keyboard.reset_all();
    mouse.reset_all();
    motion.delta = Vec2::ZERO;
    scroll.delta = Vec2::ZERO;
    selection.clear();
}

fn position(
    mut capture: ResMut<Capture>,
    mut list: ResMut<WorldListState>,
    mut runtime: ResMut<WorldRuntime>,
    mut player: ResMut<PlayerState>,
    mut dev: ResMut<DevTools>,
    space: Res<State<AppSpace>>,
) {
    if !capture.requested {
        list.act(crate::ui::WorldAction::Create {
            name: "Terrain capture".into(),
            seed: capture.config.seed.to_string(),
        });
        capture.requested = true;
    }
    let eye = DVec3::from_array(capture.config.views[capture.view].eye);
    // Create at the view's horizontal location, avoiding a complete spawn cut
    // followed by an immediate teleport and a second overlapping cut.
    if *space.get() == AppSpace::Garage && list.phase() == super::WorldListPhase::Loading {
        let ground = runtime
            .field
            .topmost_surface(eye.x, eye.z)
            .expect("capture ground");
        runtime.document.player_pose.translation =
            WorldPosition(DVec3::new(eye.x, ground + 0.1, eye.z));
    }
    if *space.get() != AppSpace::World || list.is_open() {
        return;
    }
    dev.enabled = true;
    dev.mode = DevMode::Noclip;
    let foot = WorldPosition(eye - DVec3::Y * f64::from(EYE_HEIGHT));
    runtime.capsule = KinematicCapsule::new(foot);
    player.position = runtime.global_to_local(foot);
}

fn aim(
    capture: Res<Capture>,
    runtime: Res<WorldRuntime>,
    mut camera: Query<(&mut Transform, &mut GlobalTransform), With<MainCamera>>,
) {
    let view = &capture.config.views[capture.view];
    let pose = Transform::from_translation(
        runtime.global_to_local(WorldPosition(DVec3::from_array(view.eye))),
    )
    .looking_at(
        runtime.global_to_local(WorldPosition(DVec3::from_array(view.target))),
        Vec3::Y,
    );
    for (mut local, mut global) in &mut camera {
        *local = pose;
        *global = GlobalTransform::from(pose);
    }
}

fn capture(
    mut commands: Commands,
    mut capture: ResMut<Capture>,
    runtime: Res<WorldRuntime>,
    list: Res<WorldListState>,
    diagnostics: Res<WorldDiagnostics>,
    settings: Res<crate::settings::AppSettings>,
    mut exit: MessageWriter<AppExit>,
) {
    let period = capture.started.elapsed().as_secs() / 15;
    if period != capture.reported {
        capture.reported = period;
        info!(
            "Terrain capture waiting: phase={:?} backlog={} local={}/{} textures={} settled={}",
            list.phase(),
            diagnostics.streaming_backlog,
            diagnostics.local_resolved_nodes,
            diagnostics.local_total_nodes,
            runtime.terrain_textures.is_some(),
            capture.settled
        );
    }
    if capture.started.elapsed() > Duration::from_mins(10) {
        error!(
            "Terrain capture timed out: backlog={} local={}/{}",
            diagnostics.streaming_backlog,
            diagnostics.local_resolved_nodes,
            diagnostics.local_total_nodes
        );
        exit.write(AppExit::error());
        return;
    }
    if capture.readback {
        return;
    }
    let ready = !list.is_open()
        && runtime.terrain_textures.is_none()
        && diagnostics.local_total_nodes > 0
        && diagnostics.local_resolved_nodes == diagnostics.local_total_nodes
        && capture.started.elapsed() >= Duration::from_secs(30);
    capture.settled = if ready { capture.settled + 1 } else { 0 };
    if capture.settled < 3 {
        return;
    }
    let path = capture
        .config
        .directory
        .join(format!("{}.png", capture.config.views[capture.view].name));
    let view = &capture.config.views[capture.view];
    let metrics = serde_json::json!({
        "seed": capture.config.seed, "eye": view.eye, "target": view.target,
        "app_version": env!("CARGO_PKG_VERSION"),
        "procedural_ground": settings.procedural_ground(),
        "triangles": diagnostics.triangle_count, "detail_scale": diagnostics.terrain_detail_scale,
        "streaming_backlog": diagnostics.streaming_backlog, "frame_sample_phase": "local terrain ready",
        "settle_seconds": capture.started.elapsed().as_secs_f64(),
    });
    std::fs::write(
        path.with_extension("json"),
        serde_json::to_vec_pretty(&metrics).expect("capture metrics"),
    )
    .expect("save capture metrics");
    info!(
        "Terrain capture {}: triangles={} detail={} terrain_ms={}",
        path.display(),
        diagnostics.triangle_count,
        diagnostics.terrain_detail_scale,
        diagnostics.terrain_stage_ms
    );
    commands
        .spawn((
            Readback::texture(capture.target.clone().expect("capture target")),
            CaptureReadback,
        ))
        .observe(save_capture);
    capture.readback = true;
}

fn save_capture(
    event: On<ReadbackComplete>,
    mut commands: Commands,
    mut capture: ResMut<Capture>,
    mut exit: MessageWriter<AppExit>,
) {
    if !capture.readback {
        return;
    }
    commands.entity(event.entity).despawn();
    capture.readback = false;
    if !event
        .data
        .chunks_exact(4)
        .any(|pixel| pixel[..3].iter().any(|&channel| channel != 0))
    {
        error!("Terrain capture returned a black frame; refusing visual evidence");
        exit.write(AppExit::error());
        return;
    }
    let path = capture
        .config
        .directory
        .join(format!("{}.png", capture.config.views[capture.view].name));
    Image::new(
        Extent3d {
            width: CAPTURE_WIDTH,
            height: CAPTURE_HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        event.data.clone(),
        TextureFormat::Rgba8UnormSrgb,
        default(),
    )
    .try_into_dynamic()
    .expect("capture image")
    .to_rgb8()
    .save(&path)
    .expect("save terrain screenshot");
    info!("Saved terrain capture {}", path.display());
    if capture.view + 1 == capture.config.views.len() {
        exit.write(AppExit::Success);
    } else {
        capture.view += 1;
        capture.settled = 0;
        capture.started = Instant::now();
    }
}
