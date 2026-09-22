//! End-to-end physical interaction against the live construction raycast.

mod dials;

use crate::camera::{MainCamera, MaterialWheelState, PlayerCamera, PlayerState};
use crate::controls::GameAction;
use crate::editor::state::{EditorGraph, EditorState};
use crate::physical_controls::{PhysicalControls, interact};
use crate::simulation::state::AppSimulation;
use bevy::{input::mouse::AccumulatedMouseMotion, prelude::*};
use mechanic_core::{
    BuildCommand, BuildOutcome, BuildPose, ButtonSpec, ConstructionGraph, ControllerSpec,
    CuboidSpec, DriveKey, GridRotation, InputConfiguration, InputSize, PartId, SeatSpec,
};
use mechanic_gpu::{GpuPhysics, GpuPhysicsConfig, GpuTransform};

fn spawn(graph: &mut ConstructionGraph, command: BuildCommand) -> PartId {
    let BuildOutcome::Spawned(part) = graph.apply(command).unwrap() else {
        panic!("expected spawned fixture part");
    };
    part
}

#[expect(
    clippy::too_many_lines,
    reason = "standalone live interaction fixture owns graph, residency and both input systems"
)]
fn fixture(occluded: bool, seated: bool) -> (App, PartId, Entity) {
    let mut graph = ConstructionGraph::new();
    let button = spawn(
        &mut graph,
        BuildCommand::SpawnButton(ButtonSpec::new(InputSize::Panel, BuildPose::default())),
    );
    let controller = spawn(
        &mut graph,
        BuildCommand::SpawnController(ControllerSpec::new(BuildPose::new(
            IVec3::new(10, 0, 0),
            GridRotation::default(),
        ))),
    );
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
    if occluded {
        spawn(
            &mut graph,
            BuildCommand::Spawn(
                CuboidSpec::new(
                    [2; 3],
                    BuildPose::new(IVec3::new(0, 0, 4), GridRotation::default()),
                )
                .unwrap(),
            ),
        );
    }
    let seat = seated.then(|| {
        spawn(
            &mut graph,
            BuildCommand::SpawnSeat(SeatSpec::new(BuildPose::new(
                IVec3::new(0, 0, 8),
                GridRotation::default(),
            ))),
        )
    });
    let creation = graph.compile().unwrap();
    let transforms = creation
        .compounds
        .iter()
        .map(|body| GpuTransform {
            position: body.root_translation.extend(0.0).to_array(),
            rotation: body.root_rotation.to_array(),
        })
        .collect();
    // No simulation tick is submitted: the no-op device supplies residency,
    // while raycasts and virtual key routing exercise the production systems.
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let gpu = GpuPhysics::new_with_config(&device, &queue, &creation, GpuPhysicsConfig::default())
        .unwrap();
    let mut app = App::new();
    app.insert_resource(AppSimulation {
        gpu: Some(gpu),
        creation: Some(creation),
        transforms,
        published_graph: graph.clone(),
        ..default()
    })
    .insert_resource(EditorGraph(graph))
    .insert_resource(PlayerState {
        position: Vec3::new(0.0, -crate::camera::EYE_HEIGHT, 2.0),
        seat,
        input_captured: true,
        ..default()
    })
    .insert_resource(State::new(crate::world::AppSpace::World))
    .init_resource::<MaterialWheelState>()
    .init_resource::<PhysicalControls>()
    .init_resource::<ButtonInput<GameAction>>()
    .init_resource::<ButtonInput<KeyCode>>()
    .init_resource::<AccumulatedMouseMotion>()
    .init_resource::<EditorState>()
    .init_resource::<crate::world::WorldRuntime>()
    .init_resource::<crate::freeze::DimensionFreeze>()
    .init_resource::<crate::pause_menu::PauseMenuState>()
    .init_resource::<crate::ui::UiInput>()
    .init_resource::<crate::button_config::ButtonConfiguration>()
    .add_systems(
        Update,
        (interact, crate::seat::handle_seat_interaction).chain(),
    );
    let camera = seat.map_or(Transform::default(), |seat| {
        let graph = app.world().resource::<EditorGraph>();
        let simulation = app.world().resource::<AppSimulation>();
        let (position, rotation) =
            crate::seat::seat_world_pose(&graph.0, simulation, seat).unwrap();
        Transform::from_translation(
            position + rotation * Vec3::Y * crate::camera::SEATED_EYE_HEIGHT,
        )
        .looking_at(Vec3::ZERO, Vec3::Y)
    });
    app.world_mut().spawn((
        MainCamera,
        Camera::default(),
        PlayerCamera::default(),
        camera,
        GlobalTransform::from(camera),
    ));
    let window = app
        .world_mut()
        .spawn((
            Window {
                focused: true,
                ..default()
            },
            bevy::window::PrimaryWindow,
        ))
        .id();
    (app, controller, window)
}

fn press(app: &mut App) {
    app.world_mut()
        .resource_mut::<ButtonInput<GameAction>>()
        .press(GameAction::Interact);
    app.update();
}

fn held(app: &App, controller: PartId) -> bool {
    app.world()
        .resource::<PhysicalControls>()
        .keys
        .held(controller, DriveKey::new('W').unwrap())
}

#[test]
fn reachable_button_consumes_interact_and_holds_until_released() {
    let (mut app, controller, _) = fixture(false, false);
    press(&mut app);
    assert!(held(&app, controller));
    assert!(
        !app.world()
            .resource::<ButtonInput<GameAction>>()
            .just_pressed(GameAction::Interact)
    );
    assert!(
        app.world()
            .resource::<ButtonInput<KeyCode>>()
            .get_pressed()
            .next()
            .is_none()
    );
    app.update();
    assert!(held(&app, controller));
    app.world_mut()
        .resource_mut::<ButtonInput<GameAction>>()
        .release(GameAction::Interact);
    app.update();
    assert!(!held(&app, controller));
    assert!(
        app.world()
            .resource::<PhysicalControls>()
            .keys
            .released(controller, DriveKey::new('W').unwrap())
    );
}

#[test]
fn losing_reach_focus_or_pausing_releases_a_held_momentary_button() {
    for cancellation in 0..3 {
        let (mut app, controller, window) = fixture(false, false);
        press(&mut app);
        assert!(held(&app, controller));
        match cancellation {
            0 => app.world_mut().resource_mut::<PlayerState>().position.z = 5.0,
            1 => {
                app.world_mut()
                    .entity_mut(window)
                    .get_mut::<Window>()
                    .unwrap()
                    .focused = false;
            }
            _ => app
                .world_mut()
                .resource_mut::<crate::pause_menu::PauseMenuState>()
                .open(),
        }
        app.update();
        assert!(!held(&app, controller), "cancellation {cancellation}");
    }
}

#[test]
fn intervening_construction_prevents_button_operation() {
    let (mut app, controller, _) = fixture(true, false);
    press(&mut app);
    assert!(!held(&app, controller));
    assert!(
        app.world()
            .resource::<ButtonInput<GameAction>>()
            .just_pressed(GameAction::Interact)
    );
}

#[test]
fn seated_button_consumes_interact_before_the_seat_exit_handler() {
    let (mut app, controller, _) = fixture(false, true);
    let seat = app.world().resource::<PlayerState>().seat;
    assert!(seat.is_some());
    press(&mut app);
    assert!(held(&app, controller));
    assert_eq!(app.world().resource::<PlayerState>().seat, seat);
    assert!(
        !app.world()
            .resource::<ButtonInput<GameAction>>()
            .just_pressed(GameAction::Interact)
    );
}
