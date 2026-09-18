//! Opt-in native snapshots of prescribed suspension geometry, without OS input.
use crate::ConstructionRenderMaterial;
use crate::editor::preview::EditorVisuals;
use crate::render::materials::material_index;
use bevy::{
    app::AppExit,
    input::{
        InputSystems,
        mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    },
    prelude::*,
    render::view::screenshot::{Screenshot, ScreenshotCaptured},
    winit::WinitSettings,
};
use mechanic_core::{
    BumpStopSpec, SUSPENSION_FINISHES, ShockBodyEnd, ShockSpec, SpringSpec, SuspensionSpec,
    suspension_meshes,
};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

pub(crate) struct SuspensionCapturePlugin;
impl Plugin for SuspensionCapturePlugin {
    fn build(&self, app: &mut App) {
        let Some(directory) = std::env::var_os("MECHANIC_SUSPENSION_CAPTURE_DIR") else {
            return;
        };
        let directory = PathBuf::from(directory);
        std::fs::create_dir_all(&directory).expect("create suspension capture directory");
        std::fs::write(directory.join("README.txt"), "Native Bevy screenshots of prescribed static geometry separations, not simulated equilibrium.\nCombined assemblies: extension, intermediate, actual gland contact, and 55% rubber crush limit; both shock body orientations.\n").expect("write capture description");
        app.insert_resource(WinitSettings::continuous())
            .insert_resource(Capture {
                directory,
                index: 0,
                started: Instant::now(),
                stage: Stage::Initialize,
                entities: Vec::new(),
                frames: Vec::new(),
            })
            .add_systems(PreUpdate, suppress_input.after(InputSystems))
            .add_systems(Last, advance)
            .add_systems(
                Update,
                exercise
                    .after(crate::schedule::FrameSet::Build)
                    .before(crate::schedule::FrameSet::Simulation),
            )
            .add_systems(
                Update,
                aim_camera
                    .after(crate::schedule::FrameSet::Camera)
                    .before(crate::schedule::FrameSet::Hover),
            );
    }
}
#[derive(Clone, Copy)]
enum Stage {
    Initialize,
    Wait,
    Readback,
    Done,
}
#[derive(Resource)]
struct Capture {
    directory: PathBuf,
    index: usize,
    started: Instant,
    stage: Stage,
    entities: Vec<Entity>,
    frames: Vec<(f64, u64)>,
}
fn suppress_input(
    mut keyboard: ResMut<ButtonInput<KeyCode>>,
    mut mouse: ResMut<ButtonInput<MouseButton>>,
    mut motion: ResMut<AccumulatedMouseMotion>,
    mut scroll: ResMut<AccumulatedMouseScroll>,
    mut selection: ResMut<crate::hotbar::SelectedTool>,
    capture: Res<Capture>,
) {
    if capture.index >= 17 {
        if selection.active_editor_tool() != Some(crate::Tool::Connector) {
            *selection = crate::hotbar::SelectedTool::from_editor_tool(crate::Tool::Connector);
        }
    } else if selection.active_editor_tool().is_some() {
        selection.clear();
    }
    keyboard.reset_all();
    mouse.reset_all();
    motion.delta = Vec2::ZERO;
    scroll.delta = Vec2::ZERO;
}
fn aim_camera(
    mut camera: Query<(&mut Transform, &mut GlobalTransform), With<crate::MainCamera>>,
    capture: Res<Capture>,
    editor: Res<crate::editor::state::EditorState>,
) {
    let transform = if capture.index >= 22 {
        let target = editor
            .suspension
            .controls
            .selected
            .map_or(Vec3::new(0.0, 5.5, 0.0), |t| t.0.anchor + Vec3::Y * 0.35);
        let wobble = if capture.index == 22 {
            capture.started.elapsed().as_secs_f32().sin() * 0.15
        } else {
            0.0
        };
        Transform::from_translation(target + Vec3::new(1.4 + wobble, 0.5, 2.3))
            .looking_at(target, Vec3::Y)
    } else {
        Transform::from_xyz(0.72, 1.75, 1.0).looking_at(Vec3::new(0.0, 1.47, 0.0), Vec3::Y)
    };
    for (mut local, mut global) in &mut camera {
        *local = transform;
        *global = GlobalTransform::from(transform);
    }
}
fn sample(index: usize) -> Option<(String, SuspensionSpec, f32)> {
    if (17..=25).contains(&index) {
        let (_, spec, _) = sample(if index == 20 { 13 } else { 9 }).expect("combined reference");
        let name = [
            "spring-controls",
            "shock-controls",
            "rubber-controls",
            "reversed-controls",
            "invalid-fit",
            "showcase-camera",
            "showcase-ghost",
            "showcase-damping",
            "locked-spacing",
        ][index - 17];
        return Some((format!("{index:02}-{name}"), spec, 0.0));
    }
    let spring = SpringSpec::default();
    if index < 3 {
        let spec = SuspensionSpec::new(Some(spring), None, None).unwrap();
        let (label, fraction) = [
            ("extension", 0.0),
            ("intermediate", 0.5),
            ("bottom-out", 1.0),
        ][index];
        return Some((
            format!("{index:02}-spring-{label}"),
            spec,
            fraction * spec.compression_limit().0,
        ));
    }
    if index < 9 {
        let orientation = if index < 6 {
            ShockBodyEnd::Source
        } else {
            ShockBodyEnd::Opposite
        };
        let shock = ShockSpec::new(0.5, 0.1, orientation, 0.0, 1.0, 1.6).unwrap();
        let spec = SuspensionSpec::new(None, Some(shock), None).unwrap();
        let (label, fraction) = [
            ("extension", 0.0),
            ("intermediate", 0.5),
            ("bottom-out", 1.0),
        ][(index - 3) % 3];
        return Some((
            format!("{index:02}-shock-{orientation:?}-{label}"),
            spec,
            fraction * spec.compression_limit().0,
        ));
    }
    if index >= 17 {
        return None;
    }
    let orientation = if index < 13 {
        ShockBodyEnd::Source
    } else {
        ShockBodyEnd::Opposite
    };
    let shock = ShockSpec::new(0.5, 0.1, orientation, 0.0, 1.0, 1.6).unwrap();
    let spec = SuspensionSpec::new(
        Some(spring),
        Some(shock),
        Some(BumpStopSpec::new(0.05, 0.06).unwrap()),
    )
    .unwrap();
    let stage = (index - 9) % 4;
    let (label, compression) = match stage {
        0 => ("extension", 0.0),
        1 => ("intermediate", spec.compression_limit().0 / 2.0),
        2 => ("bump-contact", spec.bump_contact().unwrap()),
        _ => ("bottom-out", spec.compression_limit().0),
    };
    Some((
        format!("{index:02}-combined-{orientation:?}-{label}"),
        spec,
        compression,
    ))
}
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "one opt-in capture state machine"
)]
fn advance(
    mut commands: Commands,
    mut capture: ResMut<Capture>,
    mut worlds: ResMut<crate::world::WorldListState>,
    mut graph: ResMut<crate::editor::state::EditorGraph>,
    mut editor: ResMut<crate::editor::state::EditorState>,
    visuals: Res<EditorVisuals>,
    mut materials: ResMut<Assets<ConstructionRenderMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut exit: MessageWriter<AppExit>,
    time: Res<Time>,
) {
    if !matches!(capture.stage, Stage::Done) && capture.started.elapsed() > Duration::from_mins(2) {
        error!("Suspension screenshot timed out");
        capture.stage = Stage::Done;
        exit.write(AppExit::error());
        return;
    }
    if matches!(capture.stage, Stage::Wait) && capture.started.elapsed() > Duration::from_secs(1) {
        capture.frames.push((
            time.delta_secs_f64() * 1000.0,
            crate::suspension_render::geometry_builds(),
        ));
    }
    match capture.stage {
        Stage::Initialize => {
            if capture.started.elapsed() < Duration::from_secs(8) {
                return;
            }
            worlds.enter_capture_garage();
            graph.0 = mechanic_core::ConstructionGraph::new();
            editor.placed_bearings.clear();
            editor.suspension.controls = crate::suspension_controls::Controls::default();
            capture.frames.clear();
            let Some((name, spec, compression)) = sample(capture.index) else {
                capture.stage = Stage::Done;
                exit.write(AppExit::Success);
                return;
            };
            for entity in capture.entities.drain(..) {
                commands.entity(entity).despawn();
            }
            let geometry = if capture.index >= 22 {
                let document = crate::creation_store::read_document(std::path::Path::new(
                    "creations/suspension-playground.mech",
                ))
                .expect("showcase document");
                graph.0 = document.into_graph().expect("showcase graph").graph;
                crate::suspension_editor::sync_sockets(&graph.0, &mut editor);
                let socket = *editor.placed_bearings.iter().find(|s| matches!(s.kind, mechanic_core::BearingKind::Suspension(s) if s.spring().is_some() && s.shock().is_some())).expect("combined station");
                editor.suspension.controls.select(socket, 1);
                editor.construction_mesh_dirty = true;
                Vec::new()
            } else if capture.index >= 17 {
                let mechanic_core::BuildOutcome::Spawned(part) = graph
                    .0
                    .apply(mechanic_core::BuildCommand::Spawn(
                        mechanic_core::CuboidSpec::new(
                            [1, 1, 1],
                            mechanic_core::BuildPose::new(
                                IVec3::new(0, 4, 0),
                                mechanic_core::GridRotation::default(),
                            ),
                        )
                        .unwrap(),
                    ))
                    .unwrap()
                else {
                    unreachable!("spawned fixture cube");
                };
                let source = mechanic_core::FaceRef::part(part, mechanic_core::FaceKind::PositiveY);
                let anchor = crate::face_geometry_from_ref(source, Some(&graph.0)).center;
                editor
                    .placed_bearings
                    .push(crate::editor::build_actions::PlacedBearing {
                        kind: mechanic_core::BearingKind::Suspension(spec),
                        axis: Vec3::Y,
                        source,
                        anchor,
                        dimensions: mechanic_core::BearingDimensions::default(),
                    });
                let socket = editor.placed_bearings[0];
                editor.suspension.controls.select(
                    socket,
                    if capture.index == 17 {
                        0
                    } else if capture.index == 19 {
                        2
                    } else {
                        1
                    },
                );
                Vec::new()
            } else {
                suspension_meshes(spec, compression)
            };
            for chunk in geometry {
                let finish = SUSPENSION_FINISHES[chunk.finish];
                let Some(base) =
                    materials.get(&visuals.construction_materials[material_index(finish.material)])
                else {
                    return;
                };
                let material = crate::suspension_render::finish_material(base, finish);
                let mesh = meshes.add(crate::suspension_render::render_mesh(
                    &chunk,
                    Some(mechanic_core::MaterialAppearance::BAKED),
                ));
                let material = materials.add(material);
                capture.entities.push(
                    commands
                        .spawn((
                            Name::new("Suspension capture sample"),
                            Mesh3d(mesh),
                            MeshMaterial3d(material),
                            Transform::from_xyz(0.0, 1.25, 0.0),
                        ))
                        .id(),
                );
            }
            info!("Suspension capture sample {name}: static compression {compression:.6} m");
            capture.started = Instant::now();
            capture.stage = Stage::Wait;
        }
        Stage::Wait if capture.started.elapsed() > Duration::from_secs(3) => {
            let name = sample(capture.index).unwrap().0;
            if capture.index >= 22 {
                let mut timings: Vec<_> = capture.frames.iter().map(|(ms, _)| *ms).collect();
                timings.sort_by(f64::total_cmp);
                if !timings.is_empty() {
                    let p95 = timings[(timings.len() * 95 / 100).min(timings.len() - 1)];
                    #[expect(clippy::cast_precision_loss)]
                    let mean = timings.iter().sum::<f64>() / timings.len() as f64;
                    let builds =
                        capture.frames.last().unwrap().1 - capture.frames.first().unwrap().1;
                    std::fs::write(capture.directory.join(format!("{name}.json")), serde_json::json!({"scenario":name,"frames":timings.len(),"mean_ms":mean,"p95_ms":p95,"geometry_builds_after_warmup":builds,"physics_running":false}).to_string()).expect("capture timings");
                }
            }
            let path = capture.directory.join(format!("{name}.png"));
            commands.spawn(Screenshot::primary_window()).observe(
                move |event: On<ScreenshotCaptured>,
                      mut run: ResMut<Capture>,
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
                    match result {
                        Ok(()) => {
                            info!("Suspension capture saved {}", path.display());
                            run.index += 1;
                            run.stage = Stage::Initialize;
                            run.started = Instant::now()
                                .checked_sub(Duration::from_secs(8))
                                .unwrap_or_else(Instant::now);
                        }
                        Err(error) => {
                            error!("Suspension capture failed: {error}");
                            run.stage = Stage::Done;
                            exit.write(AppExit::error());
                        }
                    }
                },
            );
            capture.stage = Stage::Readback;
        }
        Stage::Wait | Stage::Readback | Stage::Done => {}
    }
}

/// Native fixture input uses the same transaction handler as captured mouse motion.
#[expect(clippy::needless_pass_by_value)]
fn exercise(
    capture: Res<Capture>,
    mut graph: ResMut<crate::editor::state::EditorGraph>,
    mut editor: ResMut<crate::editor::state::EditorState>,
    mut history: ResMut<crate::editor::history::EditorHistory>,
) {
    if !matches!(capture.stage, Stage::Wait) {
        return;
    }
    if (17..=20).contains(&capture.index)
        && editor.suspension.controls.selected.is_none()
        && let Some(socket) = editor.placed_bearings.first().copied()
    {
        editor.suspension.controls.dismissed = None;
        editor.suspension.controls.select(
            socket,
            match capture.index {
                17 => 0,
                19 => 2,
                _ => 1,
            },
        );
    }
    if capture.index == 23 {
        let Some(target) = editor.suspension.controls.selected else {
            return;
        };
        let mut socket = target.0;
        socket.anchor += Vec3::X * (0.5 + capture.started.elapsed().as_secs_f32().sin() * 0.1);
        editor.suspension.preview = Some(socket);
        editor.preview_error = None;
    }
    if matches!(capture.index, 21 | 24 | 25) {
        use crate::suspension_controls::{Aim, Control, Parameter};
        let parameter = match capture.index {
            21 => Parameter::ShockOd,
            25 => Parameter::StartingCompression,
            _ => Parameter::Compression,
        };
        let mut actions = ButtonInput::default();
        if editor.suspension.controls.gesture.is_none() {
            editor.suspension.controls.aim = Some(Aim {
                control: Control::Parameter(parameter),
                direction: Vec2::X,
                pixels_per_step: 6.0,
            });
            editor.suspension.controls.consume_until_release = false;
            actions.press(crate::GameAction::Primary);
        }
        let delta = if capture.index == 24 {
            0.5
        } else if editor
            .suspension
            .controls
            .gesture
            .as_ref()
            .is_some_and(|g| (g.value - parameter.value(g.original)).abs() < 1e-6)
        {
            if capture.index == 25 { 6.0 } else { 120.0 }
        } else {
            0.0
        };
        crate::suspension_controls::actions(
            &mut graph,
            &mut editor,
            &mut history,
            &actions,
            Vec2::X * delta,
            Some(crate::Tool::Connector),
            false,
        );
    }
}
