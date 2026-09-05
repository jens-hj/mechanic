//! Opt-in stationary capture runner. Never synthesizes OS input or requests focus.
use std::{
    sync::OnceLock,
    time::{Duration, Instant},
};

use bevy::{
    app::AppExit,
    input::{
        InputSystems,
        mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    },
    prelude::*,
    render::view::screenshot::{Screenshot, ScreenshotCaptured},
    window::PrimaryWindow,
    winit::WinitSettings,
};

#[derive(Debug)]
struct Config {
    world: String,
}

fn config() -> Option<&'static Config> {
    static CONFIG: OnceLock<Option<Config>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            std::env::var("MECHANIC_AUTO_WORLD").ok().map(|world| {
                assert!(
                    !world.trim().is_empty(),
                    "MECHANIC_AUTO_WORLD must name a test-world copy"
                );
                let _directory = std::env::var_os("MECHANIC_PERF_CAPTURE_DIR")
                    .filter(|value| !value.is_empty())
                    .expect("MECHANIC_AUTO_WORLD requires MECHANIC_PERF_CAPTURE_DIR");
                Config { world }
            })
        })
        .as_ref()
}

pub(crate) fn enabled() -> bool {
    config().is_some()
}

pub(crate) struct AutomationPlugin;
impl Plugin for AutomationPlugin {
    fn build(&self, app: &mut App) {
        if !enabled() {
            return;
        }
        app.insert_resource(WinitSettings::continuous())
            .init_resource::<Run>()
            .add_systems(PreUpdate, suppress_input.after(InputSystems))
            .add_systems(Update, advance.after(crate::performance_capture::sample));
    }
}

#[derive(Default)]
enum Stage {
    #[default]
    Load,
    Capture,
    Screenshot,
    Done,
}

#[derive(Resource)]
struct Run {
    stage: Stage,
    started: Instant,
}
impl Default for Run {
    fn default() -> Self {
        Self {
            stage: Stage::Load,
            started: Instant::now(),
        }
    }
}

fn suppress_input(
    mut keyboard: ResMut<ButtonInput<KeyCode>>,
    mut mouse: ResMut<ButtonInput<MouseButton>>,
    mut motion: ResMut<AccumulatedMouseMotion>,
    mut scroll: ResMut<AccumulatedMouseScroll>,
) {
    keyboard.reset_all();
    mouse.reset_all();
    motion.delta = Vec2::ZERO;
    scroll.delta = Vec2::ZERO;
}

#[allow(clippy::too_many_arguments)]
fn advance(
    mut commands: Commands,
    mut run: ResMut<Run>,
    mut worlds: ResMut<crate::world::WorldListState>,
    mut recorder: ResMut<crate::performance_capture::Recorder>,
    mut metrics: ResMut<crate::performance::PerformanceMetrics>,
    timings: Res<crate::render_diagnostics::RenderTimings>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
    mut exit: MessageWriter<AppExit>,
) {
    let config = config().expect("automation enabled");
    if !matches!(run.stage, Stage::Done) && run.started.elapsed() > Duration::from_mins(5) {
        error!("Automatic capture timed out (loading, readiness, capture or screenshot)");
        run.stage = Stage::Done;
        exit.write(AppExit::error());
        return;
    }
    match run.stage {
        Stage::Load => {
            let found = worlds
                .entries()
                .iter()
                .find(|entry| entry.name.as_deref() == Some(&config.world));
            let Some(entry) =
                found.filter(|entry| entry.status == mechanic_world::SavedWorldStatus::Current)
            else {
                error!(
                    "Automatic capture world missing or not current: {}",
                    config.world
                );
                run.stage = Stage::Done;
                exit.write(AppExit::error());
                return;
            };
            let path = entry.path.clone();
            worlds.act(crate::ui::WorldAction::Open(path));
            if !metrics.snapshot().open {
                metrics.toggle();
            }
            timings.set_enabled(true);
            window.present_mode = bevy::window::PresentMode::AutoNoVsync;
            recorder.arm();
            info!(
                "Automatic background capture: {} (not a controlled foreground benchmark)",
                config.world
            );
            run.stage = Stage::Capture;
        }
        Stage::Capture => {
            if let Some(result) = recorder.completed.take() {
                match result {
                    Ok(capture) => {
                        let path = capture.with_extension("png");
                        commands.spawn(Screenshot::primary_window()).observe(
                            move |event: On<ScreenshotCaptured>,
                                  mut run: ResMut<Run>,
                                  mut exit: MessageWriter<AppExit>| {
                                let result = event
                                    .image
                                    .clone()
                                    .try_into_dynamic()
                                    .map_err(|error| format!("{error:?}"))
                                    .and_then(|image| {
                                        image
                                            .to_rgb8()
                                            .save(&path)
                                            .map_err(|error| error.to_string())
                                    });
                                let status = match result {
                                    Ok(()) => {
                                        info!("Automatic screenshot saved: {}", path.display());
                                        AppExit::Success
                                    }
                                    Err(error) => {
                                        error!("Automatic screenshot failed: {error}");
                                        AppExit::error()
                                    }
                                };
                                run.stage = Stage::Done;
                                exit.write(status);
                            },
                        );
                        run.stage = Stage::Screenshot;
                    }
                    Err(error) => {
                        error!("Automatic capture failed: {error}");
                        run.stage = Stage::Done;
                        exit.write(AppExit::error());
                    }
                }
            }
        }
        Stage::Screenshot | Stage::Done => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automation_discards_keyboard_mouse_and_camera_motion() {
        let mut app = App::new();
        app.init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<ButtonInput<MouseButton>>()
            .init_resource::<AccumulatedMouseMotion>()
            .init_resource::<AccumulatedMouseScroll>()
            .add_systems(Update, suppress_input);
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyW);
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Left);
        app.world_mut()
            .resource_mut::<AccumulatedMouseMotion>()
            .delta = Vec2::ONE;
        app.world_mut()
            .resource_mut::<AccumulatedMouseScroll>()
            .delta = Vec2::ONE;
        app.update();
        assert!(
            !app.world()
                .resource::<ButtonInput<KeyCode>>()
                .pressed(KeyCode::KeyW)
        );
        assert!(
            !app.world()
                .resource::<ButtonInput<MouseButton>>()
                .pressed(MouseButton::Left)
        );
        assert_eq!(
            app.world().resource::<AccumulatedMouseMotion>().delta,
            Vec2::ZERO
        );
        assert_eq!(
            app.world().resource::<AccumulatedMouseScroll>().delta,
            Vec2::ZERO
        );
    }
}
