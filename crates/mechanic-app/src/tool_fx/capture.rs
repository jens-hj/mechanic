//! Opt-in debug visual fixture. Does not open or mutate a saved world.
use super::{
    EmitterFrame, InstanceMaterialData, Kind, Request, ToolEmitter, ToolFx, VisualSnapshot,
};
use crate::{camera::MainCamera, hotbar::SelectedTool};
use bevy::prelude::*;
use bevy::{
    app::AppExit,
    render::view::screenshot::{Screenshot, ScreenshotCaptured},
    winit::WinitSettings,
};
use std::{path::PathBuf, sync::OnceLock};
fn directory() -> Option<&'static PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| std::env::var_os("MECHANIC_FX_CAPTURE_DIR").map(PathBuf::from))
        .as_ref()
}
pub(super) fn enabled() -> bool {
    directory().is_some()
}
pub(super) fn install(app: &mut App) {
    if enabled() {
        app.insert_resource(WinitSettings::continuous())
            .insert_resource(Capture {
                elapsed: -4.0,
                ..default()
            })
            .add_systems(Startup, setup.after(crate::setup))
            .add_systems(
                PostUpdate,
                advance
                    .before(super::update)
                    .after(super::collect_plate_emitters)
                    .after(bevy::transform::TransformSystems::Propagate),
            );
    }
}
#[derive(Resource, Default)]
pub(super) struct Capture {
    pub emitter: Option<EmitterFrame>,
    pub snapshot: Option<VisualSnapshot>,
    pub movement: f32,
    stage: usize,
    elapsed: f32,
    fired: bool,
    shot: bool,
    samples: Vec<f32>,
    max_particle_draws: usize,
    configured: bool,
}
#[derive(Component)]
struct Backdrop;
#[derive(Component)]
struct Link;
#[allow(clippy::too_many_arguments)]
fn setup(
    mut commands: Commands,
    visuals: Res<crate::EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    camera: Single<Entity, With<MainCamera>>,
    mut runtime: ResMut<crate::world::WorldRuntime>,
    mut worlds: ResMut<crate::world::WorldListState>,
    mut settings: ResMut<crate::settings::AppSettings>,
    mut window: Single<&mut Window>,
) {
    window.present_mode = bevy::window::PresentMode::AutoNoVsync;
    std::fs::create_dir_all(directory().unwrap()).expect("capture directory");
    crate::world::prepare_fx_capture(&mut runtime, &mut worlds, directory().unwrap());
    settings
        .set_camera_fov_degrees(75.0)
        .expect("isolated capture settings");
    commands.entity(*camera).with_children(|parent| {
        parent.spawn((
            Backdrop,
            Mesh3d(meshes.add(Cuboid::new(20.0, 20.0, 0.1))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.015, 0.015, 0.025),
                unlit: true,
                ..default()
            })),
            Transform::from_xyz(0.0, 0.0, -4.0),
        ));
        parent.spawn((
            Link,
            Mesh3d(meshes.add(crate::single_authored_part_mesh(
                crate::AuthoredPart::DimensionLinkEnabled,
            ))),
            MeshMaterial3d(
                visuals.authored_materials[crate::AuthoredPart::DimensionLinkEnabled.index()]
                    .clone(),
            ),
            Transform::from_xyz(0.0, -0.10, -2.0).with_scale(Vec3::new(0.5, 0.25, 0.25)),
        ));
    });
}
const NAMES: [&str; 7] = [
    "baseline",
    "sledge",
    "matter",
    "welder",
    "connector",
    "freeze",
    "lift_edit",
];
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn advance(
    time: Res<Time>,
    mut capture: ResMut<Capture>,
    mut fx: ResMut<ToolFx>,
    camera: Single<(Entity, &GlobalTransform), With<MainCamera>>,
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    backdrop: Single<&MeshMaterial3d<StandardMaterial>, With<Backdrop>>,
    mut link: Single<&mut Transform, With<Link>>,
    mut exit: MessageWriter<AppExit>,
    batches: Query<&InstanceMaterialData>,
    emitter: Res<ToolEmitter>,
    mut selection: ResMut<SelectedTool>,
) {
    if capture.stage >= 14 {
        return;
    }
    let dt = time.delta_secs();
    capture.elapsed += dt;
    if capture.elapsed > 1.0 {
        capture.max_particle_draws = capture.max_particle_draws.max(
            batches
                .iter()
                .filter(|batch| !batch.data.is_empty())
                .count(),
        );
    }
    let stage = capture.stage;
    let name = NAMES[stage % 7];
    let bright = stage >= 7;
    if let Some(mut material) = materials.get_mut(&backdrop.0) {
        material.base_color = if bright {
            Color::srgb(0.82, 0.82, 0.82)
        } else {
            Color::srgb(0.015, 0.015, 0.025)
        };
    }
    if !capture.configured {
        capture.configured = true;
        if stage.is_multiple_of(7) {
            commands
                .entity(camera.0)
                .remove::<bevy::post_process::bloom::Bloom>();
        } else {
            commands.entity(camera.0).insert(super::bloom());
        }
        *selection = SelectedTool::from_editor_tool(match stage % 7 {
            1 | 5 => crate::Tool::Hammer,
            3 => crate::Tool::Weld,
            4 => crate::Tool::Connector,
            _ => crate::Tool::Block,
        });
    }
    // Baseline keeps the same HDR target format even when bloom is removed.
    let camera_transform = camera.1.compute_transform();
    let offset = if stage % 7 == 6 {
        (capture.elapsed * 2.0).sin() * 0.12
    } else {
        0.0
    };
    let previous = link.translation.y;
    link.translation.y = -0.10 + offset;
    capture.movement = link.translation.y - previous;
    let hit = camera_transform.transform_point(link.translation + Vec3::Z * 0.126);
    let origin = camera_transform.mul_transform(emitter.local);
    let normal = camera_transform.rotation * Vec3::Z;
    let kind = match stage % 7 {
        2 | 6 => Some(Kind::Matter),
        3 => Some(Kind::Welder),
        4 => Some(Kind::Connector),
        _ => None,
    };
    capture.emitter = kind.map(|tool| EmitterFrame {
        tool,
        origin,
        target: hit,
        normal,
        connector_phase: emitter.connector_phase,
        connector_plates: emitter.connector_plates,
    });
    capture.snapshot = if stage % 7 >= 5 {
        Some(VisualSnapshot {
            link: mechanic_core::DimensionLinkId(0),
            min: hit - Vec3::new(0.25, 0.125, 0.25),
            max: hit + Vec3::new(0.25, 0.125, 0.0),
        })
    } else {
        None
    };
    // Warm the actual pipelines, then measure stationary frames for each fixture.
    if capture.elapsed > 1.0 {
        capture.samples.push(dt * 1000.0);
    }
    if capture.elapsed > 2.0 && !capture.fired {
        capture.fired = true;

        let request = match stage % 7 {
            1 => Some(Request::Sledge { hit, normal }),
            2 | 6 => Some(Request::Matter { hit }),
            5 => Some(Request::Freeze {
                center: hit,
                radius: 0.4,
            }),
            _ => None,
        };
        if let Some(request) = request {
            fx.push(request);
        }
    }
    if capture.elapsed > 2.07 && !capture.shot {
        capture.shot = true;
        let path = directory().unwrap().join(format!(
            "{name}-{}.png",
            if bright { "bright" } else { "dark" }
        ));
        commands.spawn(Screenshot::primary_window()).observe(
            move |event: On<ScreenshotCaptured>| {
                event
                    .image
                    .clone()
                    .try_into_dynamic()
                    .expect("capture image")
                    .to_rgb8()
                    .save(&path)
                    .expect("write capture");
            },
        );
    }
    if capture.elapsed > 3.0 {
        let mut samples = std::mem::take(&mut capture.samples);
        samples.sort_by(f32::total_cmp);
        let n = samples.len();
        #[allow(clippy::cast_precision_loss)]
        let mean = samples.iter().sum::<f32>() / n.max(1) as f32;
        let record = serde_json::json!({"effect":name,"background":if bright {"bright"}else{"dark"},"frames":n,"mean_ms":mean,"p95_ms":samples.get(n*95/100),"particle_draws":capture.max_particle_draws,"bloom_passes":if stage.is_multiple_of(7) {0}else{2*(super::bloom().max_mip_dimension.ilog2().max(2)-1)}});
        let path = directory().unwrap().join(format!(
            "{name}-{}.json",
            if bright { "bright" } else { "dark" }
        ));
        std::fs::write(path, serde_json::to_string_pretty(&record).unwrap())
            .expect("capture timings");
        info!("FX capture: {record}");
        fx.clear();
        capture.max_particle_draws = 0;
        capture.stage += 1;
        capture.elapsed = 0.0;
        capture.configured = false;
        capture.fired = false;
        capture.shot = false;
        if capture.stage == 14 {
            exit.write(AppExit::Success);
        }
    }
}
