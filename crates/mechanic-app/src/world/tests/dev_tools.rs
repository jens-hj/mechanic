//! Dev overrides preserve the normal controller and its persisted player pose.

use bevy::math::DVec3;
use bevy::prelude::*;
use mechanic_world::{KinematicCapsule, WorldPosition, WorldStore};

use crate::camera::{MainCamera, PlayerCamera, PlayerState};
use crate::controls::GameAction;
use crate::dev_tools::{DevMode, DevTools};
use crate::editor::history::EditorHistory;
use crate::editor::state::EditorGraph;
use crate::simulation::state::AppSimulation;
use crate::world::{WorldDiagnostics, WorldListState, WorldRuntime};

fn walking_app(speed: f32) -> App {
    let mut app = App::new();
    let mut dev = DevTools::default();
    dev.enabled = true;
    dev.speed = speed;
    app.add_plugins(bevy::app::TaskPoolPlugin::default())
        .init_resource::<Time>()
        .init_resource::<WorldRuntime>()
        .init_resource::<WorldListState>()
        .init_resource::<WorldDiagnostics>()
        .init_resource::<EditorGraph>()
        .init_resource::<EditorHistory>()
        .init_resource::<AppSimulation>()
        .init_resource::<PlayerState>()
        .init_resource::<ButtonInput<GameAction>>()
        .insert_resource(dev)
        .add_systems(Update, super::super::walking::walk_world);
    app.world_mut().spawn((MainCamera, PlayerCamera::default()));
    app.world_mut()
        .resource_mut::<WorldListState>()
        .enter_capture_garage();
    app.world_mut()
        .resource_mut::<Time>()
        .advance_by(std::time::Duration::from_secs_f32(1.0 / 60.0));
    let position = WorldPosition(DVec3::new(0.0, 100.0, 0.0));
    {
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        runtime.capsule = KinematicCapsule::new(position);
        runtime.floating_origin = mechanic_world::FloatingOrigin::default();
        runtime.player_terrain_ready = true;
    }
    app.world_mut().resource_mut::<PlayerState>().position = position.0.as_vec3();
    app.world_mut()
        .resource_mut::<ButtonInput<GameAction>>()
        .press(GameAction::MoveForward);
    app
}

#[test]
fn dev_speed_changes_horizontal_travel_without_changing_fall_time() {
    let mut normal = walking_app(1.0);
    let mut faster = walking_app(2.0);
    for _ in 0..240 {
        normal.update();
        faster.update();
    }
    let normal = normal.world().resource::<PlayerState>().position;
    let faster = faster.world().resource::<PlayerState>().position;
    assert!(
        faster.z > normal.z * 1.6,
        "normal {normal:?}, faster {faster:?}"
    );
    assert!((normal.y - faster.y).abs() < 0.001);
}

#[test]
fn dev_noclip_suspends_gravity_and_saves_the_return_location() {
    let mut app = walking_app(1.0);
    app.world_mut().resource_mut::<DevTools>().mode = DevMode::Noclip;
    let start = app.world().resource::<PlayerState>().position;
    let flying = Vec3::new(400.0, -100.0, 800.0);
    app.world_mut().resource_mut::<PlayerState>().position = flying;
    for _ in 0..10 {
        app.update();
    }
    assert_eq!(app.world().resource::<PlayerState>().position, flying);
    let temporary = crate::testing::TempDir::created("dev-noclip-save");
    let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
    runtime.store = WorldStore::new(&temporary.0);
    super::super::saving::save_all(&mut runtime).unwrap();
    let saved = runtime
        .store
        .load_world(&runtime.store.directory_for(&runtime.document.name))
        .unwrap();
    assert_eq!(saved.player_pose.translation.0.as_vec3(), start);
}

#[test]
fn dev_spectator_keeps_player_gravity_live_with_neutral_controls() {
    let mut app = walking_app(1.0);
    app.world_mut().resource_mut::<DevTools>().mode = DevMode::Spectator;
    app.world_mut()
        .resource_mut::<ButtonInput<GameAction>>()
        .reset_all();
    let start = app.world().resource::<PlayerState>().position;
    for _ in 0..30 {
        app.update();
    }
    let current = app.world().resource::<PlayerState>().position;
    assert!(current.y < start.y - 0.5);
    assert!((current.x - start.x).abs() < 0.001);
    assert!((current.z - start.z).abs() < 0.001);
}
