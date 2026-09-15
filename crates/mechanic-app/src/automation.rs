//! Opt-in captures with scripted input. Foreground captures explicitly request focus.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DrivingPattern {
    Straight,
    Steering,
}

#[derive(Debug)]
struct Config {
    world: String,
    world_store: Option<std::path::PathBuf>,
    foreground: bool,
    replay_ticks: Option<u64>,
    driving: Option<DrivingPattern>,
    demonstration: bool,
    placement_interval: Option<Duration>,
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
                let placement_interval = std::env::var("MECHANIC_AUTO_PLACE")
                    .ok()
                    .filter(|value| !value.is_empty())
                    .map(|value| {
                        let seconds: f64 = value
                            .parse()
                            .expect("MECHANIC_AUTO_PLACE must be a seconds interval");
                        assert!(seconds > 0.0, "MECHANIC_AUTO_PLACE must be positive");
                        Duration::from_secs_f64(seconds)
                    });
                let foreground = std::env::var("MECHANIC_AUTO_FOREGROUND").as_deref() == Ok("1");
                if foreground {
                    assert!(
                        placement_interval.is_none(),
                        "foreground comparison forbids placement diagnostics"
                    );
                    assert!(
                        std::env::var_os("MECHANIC_PERF_CAPTURE_FROM_START")
                            .is_none_or(|v| v.is_empty()),
                        "foreground comparison requires settled streaming"
                    );
                    assert!(
                        std::env::var("MECHANIC_AUTO_DRIVING_FRAMES").as_deref() != Ok("1"),
                        "foreground comparison forbids demonstration screenshots"
                    );
                }
                let replay_ticks = std::env::var("MECHANIC_AUTO_REPLAY_TICKS")
                    .ok()
                    .map(|value| {
                        let ticks = value
                            .parse::<u64>()
                            .expect("replay ticks must be an integer");
                        assert!(
                            foreground && (1..=3600).contains(&ticks),
                            "replay requires foreground mode and 1..=3600 ticks"
                        );
                        ticks
                    });
                Config {
                    world,
                    world_store: std::env::var_os("MECHANIC_AUTO_WORLD_STORE").map(Into::into),
                    foreground,
                    replay_ticks,
                    driving: (std::env::var("MECHANIC_AUTO_DRIVE").as_deref() == Ok("1")).then(
                        || {
                            if std::env::var("MECHANIC_AUTO_DRIVE_STRAIGHT").as_deref() == Ok("1") {
                                DrivingPattern::Straight
                            } else {
                                DrivingPattern::Steering
                            }
                        },
                    ),
                    demonstration: std::env::var("MECHANIC_AUTO_DRIVING_FRAMES").as_deref()
                        == Ok("1"),
                    placement_interval,
                }
            })
        })
        .as_ref()
}

pub(crate) fn enabled() -> bool {
    config().is_some()
}

pub(crate) fn foreground() -> bool {
    config().is_some_and(|config| config.foreground)
}

pub(crate) fn replay_ticks() -> Option<u64> {
    config().and_then(|config| config.replay_ticks)
}

pub(crate) fn background() -> bool {
    enabled() && !foreground()
}

pub(crate) fn world_store() -> Option<std::path::PathBuf> {
    config().and_then(|config| config.world_store.clone())
}

pub(crate) fn driving_enabled() -> bool {
    config().is_some_and(|config| config.driving.is_some())
}

pub(crate) fn driving_seat(simulation: &crate::AppSimulation) -> Option<mechanic_core::PartId> {
    let mut seats = simulation
        .published_graph
        .parts()
        .map(|(part, _)| part)
        .filter(|part| {
            simulation.published_graph.seat_input(*part).is_some()
                && simulation.published_graph.seat_controller(*part).is_some()
        });
    let seat = seats.next()?;
    seats.next().is_none().then_some(seat)
}

#[derive(Default)]
pub(crate) struct Driving {
    start_tick: Option<u64>,
    previous: Vec<char>,
}

fn driving_keys(tick: u64, straight: bool) -> Vec<char> {
    if tick < 180 {
        return Vec::new();
    }
    if straight {
        return vec!['W'];
    }
    match ((tick - 180) / 180) % 4 {
        1 => vec!['W', 'A'],
        3 => vec!['W', 'D'],
        _ => vec!['W'],
    }
}

/// Drives the sole input-linked seat in an explicitly selected disposable world.
pub(crate) fn drive_input(
    simulation: &crate::AppSimulation,
    state: &mut Driving,
    tick: u64,
) -> Option<(crate::sequencer::DriveKeyState, mechanic_core::PartId)> {
    if !driving_enabled() || !crate::performance_capture::is_active() {
        return None;
    }
    let seat = driving_seat(simulation)?;
    let start = *state.start_tick.get_or_insert(tick);
    let held = driving_keys(
        tick.saturating_sub(start),
        config().is_some_and(|config| config.driving == Some(DrivingPattern::Straight)),
    );
    let keys = crate::sequencer::DriveKeyState::scripted(&held, &state.previous);
    crate::performance_capture::record(
        "driving_input",
        || serde_json::json!({"tick":tick, "script_tick":tick.saturating_sub(start), "held": held}),
    );
    state.previous = held;
    Some((keys, seat))
}

pub(crate) struct AutomationPlugin;
impl Plugin for AutomationPlugin {
    fn build(&self, app: &mut App) {
        if !enabled() {
            return;
        }
        app.insert_resource(WinitSettings::continuous())
            .init_resource::<Run>()
            .init_resource::<ScriptedPlacement>()
            .add_systems(PreUpdate, suppress_input.after(InputSystems))
            .add_systems(
                Update,
                (advance, scripted_placement).after(crate::performance_capture::sample),
            );
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
    next_driving_frame: Option<Instant>,
    next_progress_log: Option<Instant>,
    driving_frames: u32,
}
impl Default for Run {
    fn default() -> Self {
        Self {
            stage: Stage::Load,
            started: Instant::now(),
            next_driving_frame: None,
            next_progress_log: None,
            driving_frames: 0,
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
    simulation: Res<crate::AppSimulation>,
    mut metrics: ResMut<crate::performance::PerformanceMetrics>,
    diagnostics: Res<crate::world::WorldDiagnostics>,
    timings: Res<crate::render_diagnostics::RenderTimings>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
    mut exit: MessageWriter<AppExit>,
) {
    let config = config().expect("automation enabled");
    if matches!(run.stage, Stage::Capture)
        && let Some(error) = &simulation.failure
    {
        error!("Automatic capture stopped with physics failure: {error}");
        run.stage = Stage::Done;
        exit.write(AppExit::error());
        return;
    }
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
                "Automatic capture: {} (foreground requested: {})",
                config.world, config.foreground
            );
            run.stage = Stage::Capture;
        }
        Stage::Capture => {
            log_capture_readiness(&mut run, &worlds, &simulation, &diagnostics);
            capture_driving_frame(&mut commands, &mut run, config);
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

/// Scripted block placements, so a capture contains the editing stall itself.
#[derive(Resource, Default)]
struct ScriptedPlacement {
    next: Option<Instant>,
    placed: u32,
}

/// Commits one disposable cuboid above the construction on the scripted cadence.
fn scripted_placement(
    mut placement: ResMut<ScriptedPlacement>,
    mut graph: ResMut<crate::EditorGraph>,
    mut history: ResMut<crate::EditorHistory>,
    mut state: ResMut<crate::EditorState>,
    list: Res<crate::world::WorldListState>,
) {
    let Some(interval) = config().and_then(|config| config.placement_interval) else {
        return;
    };
    // Placements also run before the capture arms, so a world whose construction
    // is entirely static still reaches a running simulation.
    if list.phase() != crate::world::WorldListPhase::Playing {
        return;
    }
    let now = Instant::now();
    if *placement.next.get_or_insert(now) > now {
        return;
    }
    placement.next = Some(now + interval);
    let anchor = graph
        .0
        .parts()
        .map(|(_, spec)| spec.pose().translation_units)
        .fold(IVec3::new(0, 0, 0), |highest, translation| {
            IVec3::new(
                highest.x.min(translation.x),
                highest.y.max(translation.y),
                highest.z.min(translation.z),
            )
        });
    let index = i32::try_from(placement.placed).unwrap_or(0);
    let pose = mechanic_core::BuildPose::new(
        anchor + IVec3::new(2 * index, 12 + 4 * index, 0),
        mechanic_core::GridRotation::default(),
    );
    let volume: i32 = std::env::var("MECHANIC_AUTO_PLACE_VOLUME")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    let previous = crate::EditorSnapshot::capture(&graph.0, &state);
    let mut outcome = Err(mechanic_core::GraphError::LinearCarriageOccupied);
    for x in 0..volume {
        for z in 0..volume {
            let cell = mechanic_core::BuildPose::new(
                pose.translation_units + IVec3::new(2 * x, 0, 2 * z),
                mechanic_core::GridRotation::default(),
            );
            let spec = mechanic_core::CuboidSpec::new([2, 2, 2], cell)
                .expect("scripted cuboid dimensions");
            outcome = graph.0.apply(mechanic_core::BuildCommand::Spawn(spec));
        }
    }
    match outcome {
        Ok(_) => {
            history.commit(previous);
            state.construction_mesh_dirty = true;
            placement.placed += 1;
            info!(
                "Scripted placement {} at {:?}",
                placement.placed, pose.translation_units
            );
            crate::performance_capture::record("scripted_placement", || {
                serde_json::json!({
                    "placement": placement.placed,
                    "translation_units": pose.translation_units.to_array(),
                    "part_count": graph.0.part_count(),
                })
            });
        }
        Err(error) => error!("Scripted placement failed: {error}"),
    }
}

/// Reports why a capture has not started yet, so a stuck run explains itself.
fn log_capture_readiness(
    run: &mut Run,
    worlds: &crate::world::WorldListState,
    simulation: &crate::AppSimulation,
    diagnostics: &crate::world::WorldDiagnostics,
) {
    if crate::performance_capture::is_active()
        || run
            .next_progress_log
            .is_some_and(|next| Instant::now() < next)
    {
        return;
    }
    run.next_progress_log = Some(Instant::now() + Duration::from_secs(2));
    info!(
        "Automation waiting: phase {:?} notice {:?} progress {:?} running {} backlog {} resolved {}/{}",
        worlds.phase(),
        worlds.notice(),
        worlds.loading_progress(),
        simulation.is_running(),
        diagnostics.streaming_backlog,
        diagnostics.local_resolved_nodes,
        diagnostics.local_total_nodes,
    );
}

fn capture_driving_frame(commands: &mut Commands, run: &mut Run, config: &Config) {
    if config.driving.is_some()
        && config.demonstration
        && crate::performance_capture::is_active()
        && run
            .next_driving_frame
            .is_none_or(|next| Instant::now() >= next)
    {
        let directory =
            std::path::PathBuf::from(std::env::var_os("MECHANIC_PERF_CAPTURE_DIR").unwrap());
        // Capture I/O is explicitly opt-in and is visible in frame timings.
        let _ = std::fs::create_dir_all(&directory);
        let path = directory.join(format!("driving-{:03}.png", run.driving_frames));
        commands
            .spawn(Screenshot::primary_window())
            .observe(bevy::render::view::screenshot::save_to_disk(path));
        run.driving_frames += 1;
        run.next_driving_frame = Some(Instant::now() + Duration::from_secs(5));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driving_sequence_settles_accelerates_and_steers_both_directions() {
        assert!(driving_keys(179, false).is_empty());
        assert_eq!(driving_keys(180, false), ['W']);
        assert_eq!(driving_keys(360, false), ['W', 'A']);
        assert_eq!(driving_keys(540, false), ['W']);
        assert_eq!(driving_keys(720, false), ['W', 'D']);
        assert_eq!(driving_keys(900, false), ['W']);
        assert!(driving_keys(179, true).is_empty());
        assert_eq!(driving_keys(180, true), ['W']);
        assert_eq!(driving_keys(900, true), ['W']);
    }

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

/// Disposable-world actions pass through the normal freeze transaction and repeat handler.
#[derive(Default)]
pub(crate) struct FreezeSequence {
    started: Option<Instant>,
    entered: bool,
    released: bool,
}
impl FreezeSequence {
    pub(crate) fn advance(&mut self) -> Option<(bool, bool, bool)> {
        if !config().is_some_and(|_| std::env::var("MECHANIC_AUTO_FREEZE").as_deref() == Ok("1"))
            || !crate::performance_capture::is_active()
        {
            return None;
        }
        let elapsed = self
            .started
            .get_or_insert_with(Instant::now)
            .elapsed()
            .as_secs_f32();
        let toggle = if elapsed >= 2.0 && !self.entered {
            self.entered = true;
            true
        } else if elapsed >= 40.0 && !self.released {
            self.released = true;
            true
        } else {
            false
        };
        let up = (2.1..2.3).contains(&elapsed) || (10.0..14.0).contains(&elapsed);
        let down = (20.0..26.0).contains(&elapsed);
        if toggle || up || down {
            crate::performance_capture::record("freeze_input", || {
                serde_json::json!({
                    "toggle": toggle, "up": up, "down": down
                })
            });
        }
        Some((toggle, up, down))
    }
}
