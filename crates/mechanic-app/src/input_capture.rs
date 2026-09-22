//! Opt-in native physical-input geometry screenshots, with transient visual poses.

use bevy::{
    app::AppExit,
    asset::RenderAssetUsages,
    input::{
        InputSystems,
        mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    },
    mesh::Indices,
    prelude::*,
    render::{
        render_resource::PrimitiveTopology,
        view::screenshot::{Screenshot, ScreenshotCaptured},
    },
    winit::WinitSettings,
};
use mechanic_core::{
    ButtonFeedback, DriveKey, INPUT_FINISHES, InputKind, InputMeshOwner, InputSize, input_meshes,
};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

pub(crate) struct InputCapturePlugin;

impl Plugin for InputCapturePlugin {
    fn build(&self, app: &mut App) {
        let Some(directory) = crate::env::raw(crate::env::INPUT_CAPTURE_DIR) else {
            return;
        };
        let directory = PathBuf::from(directory);
        std::fs::create_dir_all(&directory).expect("create input capture directory");
        std::fs::write(directory.join("README.txt"), "Native Bevy static geometry showcase. Columns: 5 cm / 10 cm / 25 cm. Back row: dials. Front row: buttons W / 8 / A. Successive screenshots show released and depressed illuminated buttons. The overview is followed by released/held closeups of each button size the controller dial-assignment editor, and a Connector button-configuration overlay. These prescribed poses verify rendering, not controller interaction.\n").expect("capture description");
        app.insert_resource(WinitSettings::continuous())
            .insert_resource(Capture {
                directory,
                started: Instant::now(),
                stage: Stage::Initialize,
                on: false,
                view: 0,
                entities: Vec::new(),
            })
            .add_systems(PreUpdate, suppress_input.after(InputSystems))
            .add_systems(
                Update,
                aim_camera
                    .after(crate::schedule::FrameSet::Camera)
                    .before(crate::schedule::FrameSet::Hover),
            )
            .add_systems(Last, advance);
    }
}

#[derive(Resource)]
struct Capture {
    directory: PathBuf,
    started: Instant,
    stage: Stage,
    on: bool,
    view: usize,
    entities: Vec<Entity>,
}

#[derive(Clone, Copy)]
enum Stage {
    Initialize,
    Wait,
    Readback,
    Done,
}

fn suppress_input(
    mut keyboard: ResMut<ButtonInput<KeyCode>>,
    mut mouse: ResMut<ButtonInput<MouseButton>>,
    mut motion: ResMut<AccumulatedMouseMotion>,
    mut scroll: ResMut<AccumulatedMouseScroll>,
    mut selection: ResMut<crate::hotbar::SelectedTool>,
    capture: Res<Capture>,
) {
    keyboard.reset_all();
    mouse.reset_all();
    motion.delta = Vec2::ZERO;
    scroll.delta = Vec2::ZERO;
    if capture.view == 5 {
        *selection = crate::hotbar::SelectedTool::from_editor_tool(crate::hotbar::Tool::Connector);
    } else {
        selection.clear();
    }
}

fn aim_camera(
    capture: Res<Capture>,
    mut camera: Query<(&mut Transform, &mut GlobalTransform), With<crate::MainCamera>>,
) {
    let pose = if capture.view == 5 {
        Transform::from_xyz(0.0, 1.85, 0.3).looking_at(Vec3::new(0.0, 1.5, 0.0), Vec3::Y)
    } else if capture.view == 4 {
        Transform::from_xyz(1.0, 1.6, 2.0).looking_at(Vec3::new(0.0, 0.6, 0.0), Vec3::Y)
    } else if capture.view == 0 {
        Transform::from_xyz(0.0, 2.45, 0.85).looking_at(Vec3::new(0.0, 1.5, 0.0), Vec3::Y)
    } else {
        let index = capture.view - 1;
        let size = InputSize::ALL[index];
        let distance = (size.meters() * 2.2).max(0.22);
        let target = Vec3::new([-0.34, 0.0, 0.34][index], 1.5, 0.18);
        Transform::from_translation(target + Vec3::new(0.0, distance * 0.8, distance * 0.6))
            .looking_at(target, Vec3::Y)
    };
    for (mut local, mut global) in &mut camera {
        *local = pose;
        *global = GlobalTransform::from(pose);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "opt-in capture owns geometry and screenshot lifecycle"
)]
fn advance(
    mut commands: Commands,
    mut capture: ResMut<Capture>,
    mut worlds: ResMut<crate::world::WorldListState>,
    mut graph: ResMut<crate::editor::state::EditorGraph>,
    mut editor: ResMut<crate::editor::state::EditorState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut exit: MessageWriter<AppExit>,
    mut panel: ResMut<crate::control_panel::ControlPanelState>,
    mut assignments: ResMut<crate::dial_assignment::DialAssignments>,
    mut button_config: ResMut<crate::button_config::ButtonConfiguration>,
) {
    if !matches!(capture.stage, Stage::Done) && capture.started.elapsed() > Duration::from_secs(90)
    {
        error!("Physical-input screenshot timed out");
        capture.stage = Stage::Done;
        exit.write(AppExit::error());
        return;
    }
    match capture.stage {
        Stage::Initialize => {
            if capture.started.elapsed() < Duration::from_secs(8) {
                return;
            }
            worlds.enter_capture_garage();
            graph.0 = mechanic_core::ConstructionGraph::new();
            editor.placed_bearings.clear();
            editor.construction_mesh_dirty = true;
            for entity in capture.entities.drain(..) {
                commands.entity(entity).despawn();
            }
            if capture.view == 5 {
                let (fixture, button) = button_fixture();
                graph.0 = fixture;
                panel.close();
                assignments.draft = None;
                button_config.selected = Some(button);
                button_config.capturing = false;
            } else if capture.view == 4 {
                let (fixture, controller, draft) = assignment_fixture();
                graph.0 = fixture;
                panel.open(controller);
                assignments.draft = Some(draft);
            } else {
                capture.entities = showcase(&mut commands, &mut meshes, &mut materials, capture.on);
            }
            capture.started = Instant::now();
            capture.stage = Stage::Wait;
        }
        Stage::Wait if capture.started.elapsed() > Duration::from_secs(3) => {
            let name = if capture.view == 5 {
                "09-button-connector-ui.png".to_owned()
            } else if capture.view == 4 {
                "08-dial-assignment-ui.png".to_owned()
            } else {
                let state = if capture.on { "held" } else { "released" };
                let subject = [
                    "inputs",
                    "button-panel",
                    "button-utility",
                    "button-industrial",
                ][capture.view];
                let index = capture.view * 2 + usize::from(capture.on);
                format!("{index:02}-{subject}-{state}.png")
            };
            let path = capture.directory.join(name);
            commands.spawn(Screenshot::primary_window()).observe(
                move |event: On<ScreenshotCaptured>,
                      mut capture: ResMut<Capture>,
                      mut exit: MessageWriter<AppExit>| {
                    let saved = event
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
                    if let Err(error) = saved {
                        error!("Physical-input capture failed: {error}");
                        capture.stage = Stage::Done;
                        exit.write(AppExit::error());
                    } else if capture.view == 5 {
                        info!(
                            "Physical-input captures saved in {}",
                            capture.directory.display()
                        );
                        capture.stage = Stage::Done;
                        exit.write(AppExit::Success);
                    } else {
                        if capture.on || capture.view >= 4 {
                            capture.view += 1;
                        }
                        capture.on = !capture.on;
                        capture.stage = Stage::Initialize;
                        capture.started = Instant::now()
                            .checked_sub(Duration::from_secs(8))
                            .unwrap_or_else(Instant::now);
                    }
                },
            );
            capture.stage = Stage::Readback;
        }
        Stage::Wait | Stage::Readback | Stage::Done => {}
    }
}

fn showcase(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    on: bool,
) -> Vec<Entity> {
    let mut entities = Vec::new();
    entities.push(
        commands
            .spawn((
                PointLight {
                    intensity: 120.0,
                    shadow_maps_enabled: true,
                    ..default()
                },
                Transform::from_xyz(-0.4, 2.5, 0.4),
            ))
            .id(),
    );
    entities.push(
        commands
            .spawn((
                Mesh3d(meshes.add(Cuboid::new(1.05, 0.025, 0.75))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.13, 0.16, 0.19),
                    perceptual_roughness: 0.8,
                    ..default()
                })),
                Transform::from_xyz(0.0, 1.37, 0.0),
            ))
            .id(),
    );
    for ((size, x), symbol) in InputSize::ALL
        .into_iter()
        .zip([-0.34, 0.0, 0.34])
        .zip(['W', '8', 'A'])
    {
        for (kind, z) in [(InputKind::Button, 0.18), (InputKind::Dial, -0.18)] {
            let mut chunks = input_meshes(kind, size)
                .into_iter()
                .map(|chunk| {
                    let mesh = Mesh::new(
                        PrimitiveTopology::TriangleList,
                        RenderAssetUsages::default(),
                    )
                    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, chunk.positions)
                    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, chunk.normals)
                    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, chunk.uvs)
                    .with_inserted_indices(Indices::U32(chunk.indices));
                    (chunk.owner, chunk.finish, mesh)
                })
                .collect::<Vec<_>>();
            if kind == InputKind::Button {
                chunks.push((
                    InputMeshOwner::Cap,
                    5,
                    crate::input_label::mesh(DriveKey::new(symbol).unwrap(), size),
                ));
            }
            for (owner, finish, mesh) in chunks {
                if matches!(owner, InputMeshOwner::Spring(pose) if pose != if on { ButtonFeedback::On } else { ButtonFeedback::Off })
                {
                    continue;
                }
                let finish = INPUT_FINISHES[finish];
                let signal = finish.name == "signal";
                let mut material = StandardMaterial {
                    base_color: Color::srgb_u8(finish.color[0], finish.color[1], finish.color[2]),
                    perceptual_roughness: finish.roughness,
                    metallic: finish.metalness,
                    ..default()
                };
                if signal {
                    material.base_color = if on {
                        Color::srgb(0.05, 1.0, 0.55)
                    } else {
                        Color::srgb(0.08, 0.12, 0.14)
                    };
                    material.emissive = if on {
                        LinearRgba::new(0.05, 1.0, 0.55, 1.0)
                    } else {
                        LinearRgba::BLACK
                    };
                }
                let depression = if on && owner == InputMeshOwner::Cap {
                    size.button_travel()
                } else {
                    0.0
                };
                entities.push(
                    commands
                        .spawn((
                            Name::new(format!("{kind:?} {size:?} {owner:?}")),
                            Mesh3d(meshes.add(mesh)),
                            MeshMaterial3d(materials.add(material)),
                            Transform::from_xyz(x, 1.5 - depression, z),
                        ))
                        .id(),
                );
            }
        }
    }
    entities
}

fn assignment_fixture() -> (
    mechanic_core::ConstructionGraph,
    mechanic_core::PartId,
    crate::dial_assignment::Draft,
) {
    use mechanic_core::{
        AnalogMapping, AnalogRange, BearingSpec, BuildCommand, BuildOutcome, BuildPose,
        ConstructionGraph, ControllerSpec, CuboidSpec, DialSpec, DriveLinkSpec, DriveParameter,
        FaceKind, FaceRef, GridRotation, InputConfiguration, NumericParameter,
    };
    let mut graph = ConstructionGraph::new();
    let mut spawn = |command| {
        let BuildOutcome::Spawned(part) = graph.apply(command).expect("capture part") else {
            panic!("capture spawn outcome");
        };
        part
    };
    let cuboid = |dimensions, units| {
        CuboidSpec::new(dimensions, BuildPose::new(units, GridRotation::default())).unwrap()
    };
    let base = spawn(BuildCommand::Spawn(cuboid([4, 2, 4], IVec3::new(0, 1, 0))));
    let rotor = spawn(BuildCommand::Spawn(cuboid([2, 2, 2], IVec3::new(0, 3, 0))));
    let controller = spawn(BuildCommand::SpawnController(ControllerSpec::new(
        BuildPose::from_half_grid(IVec3::new(2, 5, 0), GridRotation::default()),
    )));
    let dial = spawn(BuildCommand::SpawnDial(DialSpec::new(
        InputSize::Panel,
        BuildPose::new(IVec3::new(3, 2, 0), GridRotation::default()),
    )));
    let BuildOutcome::BearingAdded(bearing) = graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(base, FaceKind::PositiveY),
            FaceRef::part(rotor, FaceKind::NegativeY),
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::Y,
        )))
        .unwrap()
    else {
        panic!("capture bearing");
    };
    let BuildOutcome::DriveLinked(link) = graph
        .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
            controller, bearing,
        )))
        .unwrap()
    else {
        panic!("capture drive");
    };
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input: dial,
            configuration: InputConfiguration {
                controller: Some(controller),
                name: "Dial 1".into(),
                ..default()
            },
        })
        .unwrap();
    let target = NumericParameter::Drive {
        link,
        parameter: DriveParameter::AngularSpeed(0),
    };
    graph
        .assign_dial(
            dial,
            &[AnalogMapping {
                target,
                range: AnalogRange::new([-10.0, 10.0], false).unwrap(),
            }],
        )
        .unwrap();
    let factor = mechanic_core::rad_s_to_rpm(1.0);
    let draft = crate::dial_assignment::Draft {
        targets: vec![target],
        name: "Joint 1 · State 1 speed".into(),
        current: graph.numeric_value(target).unwrap() * factor,
        dial: Some(dial),
        minimum: -10.0 * factor,
        maximum: 10.0 * factor,
        reverse: false,
        factor,
        unit: "rpm".into(),
        integer: false,
        error: None,
    };
    (graph, controller, draft)
}

fn button_fixture() -> (mechanic_core::ConstructionGraph, mechanic_core::PartId) {
    use mechanic_core::{
        BuildCommand, BuildOutcome, BuildPose, ButtonSpec, ConstructionGraph, ControllerSpec,
        GridRotation, InputConfiguration,
    };
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(button) = graph
        .apply(BuildCommand::SpawnButton(ButtonSpec::new(
            InputSize::Utility,
            BuildPose::new(IVec3::new(0, 6, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        panic!("capture button");
    };
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::new(IVec3::new(6, 6, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        panic!("capture controller");
    };
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input: button,
            configuration: InputConfiguration {
                controller: Some(controller),
                key: DriveKey::new('W'),
                ..default()
            },
        })
        .unwrap();
    (graph, button)
}
