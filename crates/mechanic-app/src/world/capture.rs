//! Opt-in screenshots of normally generated, streamed terrain in an isolated world.
//!
//! With `timing_frames`, each view also records that many frames of GPU pass
//! timings once its terrain is ready, and writes their medians beside the
//! screenshot, so a change's rendering cost can be compared view by view.

use super::{AppSpace, WorldDiagnostics, WorldListState, WorldRuntime};
use crate::{
    camera::{EYE_HEIGHT, MainCamera, PlayerState},
    dev_tools::{DevMode, DevTools},
    render_diagnostics::{RenderTimings, Snapshot},
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
    /// Width and height of the screenshots, and of the frames timed.
    #[serde(default = "default_size")]
    size: [u32; 2],
    /// Frames of GPU timings recorded per view before its screenshot.
    #[serde(default)]
    timing_frames: usize,
    /// Minutes a view may take to stream, settle and be timed.
    #[serde(default = "default_timeout_minutes")]
    timeout_minutes: u64,
}

const fn default_timeout_minutes() -> u64 {
    10
}

const fn default_size() -> [u32; 2] {
    [CAPTURE_WIDTH, CAPTURE_HEIGHT]
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
    /// Frame time and GPU timings of the frames timed for this view.
    timed: Vec<(f64, Snapshot)>,
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
            timed: Vec::new(),
        })
        .add_systems(First, stop_repeated_readback)
        .add_systems(PreUpdate, suppress_input.after(InputSystems))
        .add_systems(Update, prepare_target.before(FrameSet::Camera))
        .add_systems(
            Update,
            position.after(FrameSet::WorldList).before(FrameSet::Camera),
        )
        .add_systems(Update, aim.after(FrameSet::Camera).before(FrameSet::Hover))
        .add_systems(Last, (time_frames, capture).chain());
}

// Read the application cameras through an owned texture instead of relying
// on capture of a background window surface, by default this large.
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
    let [width, height] = capture.config.size;
    let target = capture.target.get_or_insert_with(|| {
        let mut image =
            Image::new_target_texture(width, height, TextureFormat::Rgba8UnormSrgb, None);
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

/// Records a ready view's frames for its timings, unpaced, so frame times
/// show the work rather than the display.
fn time_frames(
    mut capture: ResMut<Capture>,
    timings: Res<RenderTimings>,
    time: Res<Time<Real>>,
    mut window: Single<&mut Window, With<bevy::window::PrimaryWindow>>,
) {
    if capture.settled < 3 || capture.timed.len() >= capture.config.timing_frames {
        return;
    }
    timings.set_enabled(true);
    window.present_mode = bevy::window::PresentMode::AutoNoVsync;
    // The first frames after enabling carry no complete sample yet.
    if capture.settled > 10 {
        capture
            .timed
            .push((time.delta_secs_f64() * 1_000.0, timings.snapshot()));
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
    if capture.started.elapsed() > Duration::from_mins(capture.config.timeout_minutes) {
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
    // A view high above the ground has no local terrain; it is ready once
    // everything it streams is drawn. A timed view always waits for that, so
    // what is timed is the whole scene.
    let drained = diagnostics.streaming_backlog == 0;
    let ready = !list.is_open()
        && runtime.terrain_textures.is_none()
        && (diagnostics.local_total_nodes > 0 || drained)
        && (capture.config.timing_frames == 0 || drained)
        && diagnostics.local_resolved_nodes == diagnostics.local_total_nodes
        && capture.started.elapsed() >= Duration::from_secs(30);
    capture.settled = if ready { capture.settled + 1 } else { 0 };
    if capture.settled < 3 {
        capture.timed.clear();
        return;
    }
    if capture.timed.len() < capture.config.timing_frames {
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
        "size": capture.config.size, "timing": timing_medians(&capture.timed),
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
            width: capture.config.size[0],
            height: capture.config.size[1],
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        tight_rows(&event.data, capture.config.size),
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
        capture.timed.clear();
    }
}

/// Medians of the timed frames: frame time, the world's whole GPU span, and
/// its opaque and transparent passes, each over the frames that measured it.
fn timing_medians(timed: &[(f64, Snapshot)]) -> serde_json::Value {
    let median = |values: Vec<f64>| {
        let mut values = values;
        values.sort_by(f64::total_cmp);
        values.get(values.len() / 2).copied()
    };
    let of = |pick: fn(&Snapshot) -> Option<f64>| {
        let values = timed
            .iter()
            .filter_map(|(_, snapshot)| pick(snapshot))
            .collect::<Vec<_>>();
        serde_json::json!({ "median_ms": median(values.clone()), "frames": values.len() })
    };
    serde_json::json!({
        "frame": { "median_ms": median(timed.iter().map(|(frame, _)| *frame).collect()), "frames": timed.len() },
        "gpu": of(|snapshot| snapshot.render_gpu_ms),
        "opaque": of(|snapshot| snapshot.breakdown.opaque_ms),
        "transparent": of(|snapshot| snapshot.breakdown.transparent_ms),
    })
}

/// Readback rows padded to the copy alignment, packed tightly again.
fn tight_rows(data: &[u8], [width, height]: [u32; 2]) -> Vec<u8> {
    let row = width as usize * 4;
    let rows = height as usize;
    let stride = data.len() / rows.max(1);
    if stride == row {
        return data.to_vec();
    }
    data.chunks_exact(stride)
        .take(rows)
        .flat_map(|padded| &padded[..row])
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::tight_rows;

    #[test]
    fn padded_readback_rows_are_packed_into_the_image() {
        // Three pixels wide: 12 bytes a row, padded to 16 as copies align them.
        let padded = (0..2_u8)
            .flat_map(|row| (0..12).map(move |byte| row * 100 + byte).chain([0xee; 4]))
            .collect::<Vec<_>>();
        let packed = tight_rows(&padded, [3, 2]);
        assert_eq!(packed.len(), 24);
        assert_eq!(&packed[..12], &(0..12).collect::<Vec<u8>>()[..]);
        assert_eq!(&packed[12..], &(100..112).collect::<Vec<u8>>()[..]);
        // Unpadded rows pass through untouched.
        assert_eq!(tight_rows(&packed, [3, 2]), packed);
    }
}
