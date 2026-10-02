use super::*;

#[test]
fn dev_navigation_is_disabled_by_default() {
    let dev = DevTools::default();
    assert!(!dev.enabled);
    assert!(!dev.noclip());
    assert!(!dev.spectator());
    assert!((dev.multiplier() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn speed_changes_are_bounded_and_resettable() {
    let mut dev = DevTools {
        enabled: true,
        ..default()
    };
    for _ in 0..20 {
        dev.adjust_speed(false, true, false);
    }
    assert!((dev.speed - 32.0).abs() < f32::EPSILON);
    for _ in 0..20 {
        dev.adjust_speed(true, false, false);
    }
    assert!((dev.speed - 0.125).abs() < f32::EPSILON);
    dev.adjust_speed(false, false, true);
    assert!((dev.speed - 1.0).abs() < f32::EPSILON);
    dev.adjust_speed(true, true, false);
    assert!((dev.speed - 1.0).abs() < f32::EPSILON);
}

#[test]
fn noclip_returns_player_and_view_to_start() {
    let mut dev = DevTools {
        enabled: true,
        ..default()
    };
    let mut player = PlayerState::default();
    let mut view = PlayerCamera::default();
    let original = player.position;
    dev.enter(DevMode::Noclip, &player, view, Transform::IDENTITY);
    player.position = Vec3::splat(100.0);
    view.yaw = 1.0;
    assert_eq!(dev.return_position(), Some(original));
    assert!(dev.exit(&mut player, &mut view));
    assert_eq!(player.position, original);
    assert_eq!(view, PlayerCamera::default());
}

#[test]
fn spectator_does_not_restore_an_outdated_player_position() {
    let mut dev = DevTools {
        enabled: true,
        ..default()
    };
    let mut player = PlayerState::default();
    let mut view = PlayerCamera::default();
    dev.enter(DevMode::Spectator, &player, view, Transform::IDENTITY);
    player.position = Vec3::splat(10.0);
    assert!(!dev.exit(&mut player, &mut view));
    assert_eq!(player.position, Vec3::splat(10.0));
}

#[test]
fn flight_speed_scales_without_diagonal_boost() {
    let mut dev = DevTools {
        enabled: true,
        movement: Vec2::Y,
        ..default()
    };
    let straight = dev.step(Quat::IDENTITY, 1.0).length();
    dev.movement = Vec2::ONE.normalize();
    dev.vertical = 1.0;
    assert!((dev.step(Quat::IDENTITY, 1.0).length() - straight).abs() < 0.0001);
    dev.speed = 4.0;
    assert!((dev.step(Quat::IDENTITY, 1.0).length() - straight * 4.0).abs() < 0.0001);
}

fn input_app() -> App {
    let mut app = App::new();
    app.insert_resource(DevTools {
        enabled: true,
        ..default()
    })
    .init_resource::<ButtonInput<GameAction>>()
    .init_resource::<PlayerState>()
    .init_resource::<crate::pause_menu::PauseMenuState>()
    .init_resource::<crate::creation_menu::CreationMenuState>()
    .init_resource::<crate::control_panel::ControlPanelState>()
    .init_resource::<crate::camera::MaterialWheelState>()
    .init_resource::<crate::ui::UiInput>()
    .init_resource::<crate::physical_controls::PhysicalControls>()
    .init_resource::<crate::editor::state::EditorState>()
    .init_resource::<WorldRuntime>()
    .init_resource::<WorldListState>()
    .insert_resource(State::new(AppSpace::World))
    .add_systems(Update, input);
    app.world_mut()
        .resource_mut::<WorldListState>()
        .enter_capture_garage();
    app.world_mut().spawn((
        Window {
            focused: true,
            ..default()
        },
        PrimaryWindow,
    ));
    app.world_mut().spawn((
        MainCamera,
        PlayerCamera::default(),
        Transform::IDENTITY,
        GlobalTransform::IDENTITY,
    ));
    app
}

fn press(app: &mut App, actions: &[GameAction]) {
    let mut input = app.world_mut().resource_mut::<ButtonInput<GameAction>>();
    input.reset_all();
    for action in actions {
        input.press(*action);
    }
}

#[test]
fn spectator_owns_navigation_without_exposing_gameplay_actions() {
    let mut app = input_app();
    press(
        &mut app,
        &[
            GameAction::DevSpectator,
            GameAction::MoveForward,
            GameAction::Jump,
            GameAction::Primary,
            GameAction::Interact,
        ],
    );
    app.update();
    let dev = app.world().resource::<DevTools>();
    assert!(dev.spectator());
    assert_eq!(dev.movement, Vec2::Y);
    assert!((dev.vertical - 1.0).abs() < f32::EPSILON);
    let actions = app.world().resource::<ButtonInput<GameAction>>();
    for action in [
        GameAction::MoveForward,
        GameAction::Jump,
        GameAction::Primary,
        GameAction::Interact,
    ] {
        assert!(!actions.pressed(action));
        assert!(!actions.just_released(action));
    }
}

#[test]
fn noclip_keeps_editing_and_mode_switch_restores_start() {
    let mut app = input_app();
    let start = app.world().resource::<PlayerState>().position;
    press(&mut app, &[GameAction::DevNoclip, GameAction::Primary]);
    app.update();
    assert!(app.world().resource::<DevTools>().noclip());
    assert!(
        app.world()
            .resource::<ButtonInput<GameAction>>()
            .pressed(GameAction::Primary)
    );
    app.world_mut().resource_mut::<PlayerState>().position = Vec3::splat(200.0);
    press(&mut app, &[GameAction::DevSpectator]);
    app.update();
    assert!(app.world().resource::<DevTools>().spectator());
    assert_eq!(app.world().resource::<PlayerState>().position, start);
    let dev = app.world().resource::<DevTools>();
    assert!(
        dev.spectator
            .translation
            .distance(start + Vec3::Y * crate::camera::EYE_HEIGHT)
            < 0.001
    );
}

#[test]
fn dev_input_respects_focus_pause_and_disabled_opt_in() {
    let mut app = input_app();
    app.world_mut().resource_mut::<DevTools>().enabled = false;
    press(
        &mut app,
        &[GameAction::DevNoclip, GameAction::DevSpeedIncrease],
    );
    app.update();
    assert_eq!(app.world().resource::<DevTools>().mode, DevMode::Normal);
    assert!((app.world().resource::<DevTools>().speed - 1.0).abs() < f32::EPSILON);
    app.world_mut().resource_mut::<DevTools>().enabled = true;
    app.world_mut()
        .resource_mut::<crate::pause_menu::PauseMenuState>()
        .open();
    app.update();
    assert_eq!(app.world().resource::<DevTools>().mode, DevMode::Normal);
    app.world_mut()
        .resource_mut::<crate::pause_menu::PauseMenuState>()
        .close();
    app.world_mut()
        .query::<&mut Window>()
        .single_mut(app.world_mut())
        .unwrap()
        .focused = false;
    app.update();
    assert_eq!(app.world().resource::<DevTools>().mode, DevMode::Normal);
    assert!((app.world().resource::<DevTools>().speed - 1.0).abs() < f32::EPSILON);
}

#[test]
fn space_transition_restores_noclip_before_world_commands() {
    let mut app = input_app();
    let start = app.world().resource::<PlayerState>().position;
    press(&mut app, &[GameAction::DevNoclip]);
    app.update();
    app.world_mut().resource_mut::<PlayerState>().position = Vec3::splat(200.0);
    press(&mut app, &[GameAction::ToggleSpace]);
    app.update();
    assert_eq!(app.world().resource::<DevTools>().mode, DevMode::Normal);
    assert_eq!(app.world().resource::<PlayerState>().position, start);
    assert!(
        app.world()
            .resource::<ButtonInput<GameAction>>()
            .just_pressed(GameAction::ToggleSpace)
    );
}

#[test]
fn detached_camera_moves_without_moving_the_player_or_player_look() {
    let mut dev = DevTools {
        enabled: true,
        ..default()
    };
    let mut player = PlayerState::default();
    let mut view = PlayerCamera::default();
    let original = player;
    let initial = Transform::from_xyz(1.0, 2.0, 3.0);
    dev.enter(DevMode::Spectator, &player, view, initial);
    dev.input_active = true;
    dev.movement = Vec2::Y;
    let mut transform = initial;
    dev.update_camera(
        &mut player,
        &mut view,
        &mut transform,
        Vec2::new(50.0, 20.0),
        1.0,
        true,
    );
    assert!(transform.translation.distance(initial.translation) > 3.9);
    assert_eq!(player, original);
    assert_eq!(view, PlayerCamera::default());
    let stopped = transform;
    dev.update_camera(
        &mut player,
        &mut view,
        &mut transform,
        Vec2::splat(40.0),
        1.0,
        false,
    );
    assert_eq!(transform, stopped);
}

#[test]
fn streaming_follows_dev_view_and_retains_the_live_player_or_return_point() {
    use mechanic_world::{FloatingOrigin, WorldPosition};
    let origin = FloatingOrigin(bevy::math::DVec3::splat(1000.0));
    let mut dev = DevTools {
        enabled: true,
        ..default()
    };
    let mut player = PlayerState::default();
    let start = WorldPosition(origin.0 + player.position.as_dvec3());
    let live_seat = WorldPosition(origin.0 + bevy::math::DVec3::new(100.0, 5.0, 200.0));
    let mut view = PlayerCamera::default();
    let spectator = Transform::from_xyz(300.0, 100.0, 400.0);
    assert_eq!(
        dev.streaming_focus(&player, live_seat, origin),
        (live_seat, None)
    );
    dev.enter(DevMode::Spectator, &player, view, spectator);
    assert_eq!(
        dev.streaming_focus(&player, live_seat, origin),
        (
            WorldPosition(origin.0 + spectator.translation.as_dvec3()),
            Some(live_seat)
        )
    );
    dev.exit(&mut player, &mut view);
    dev.enter(DevMode::Noclip, &player, view, spectator);
    player.position = Vec3::splat(400.0);
    let flying = WorldPosition(origin.0 + player.position.as_dvec3());
    assert_eq!(
        dev.streaming_focus(&player, flying, origin),
        (flying, Some(start))
    );
}

#[test]
#[expect(
    clippy::float_cmp,
    reason = "exact integral clock steps and lossless persistence"
)]
fn time_controls_obey_developer_focus_menu_and_space_gates() {
    for gate in 0..5 {
        let mut app = input_app();
        let initial = app.world().resource::<WorldRuntime>().time_of_day_seconds();
        match gate {
            0 => app.world_mut().resource_mut::<DevTools>().enabled = false,
            1 => app
                .world_mut()
                .resource_mut::<crate::pause_menu::PauseMenuState>()
                .open(),
            2 => {
                let mut query = app.world_mut().query::<&mut Window>();
                query.single_mut(app.world_mut()).unwrap().focused = false;
            }
            3 => {
                app.insert_resource(State::new(AppSpace::Garage));
            }
            _ => {}
        }
        press(
            &mut app,
            &[GameAction::DevTimeLater, GameAction::DevTimePause],
        );
        app.update();
        let delta = if gate == 4 { 3600.0 } else { 0.0 };
        assert_eq!(
            app.world().resource::<WorldRuntime>().time_of_day_seconds(),
            initial + delta
        );
        assert_eq!(app.world().resource::<DevTools>().cycle_paused, gate == 4);
    }
}

#[test]
#[expect(clippy::float_cmp, reason = "exact powers of ten")]
fn erosion_cycles_up_to_a_thousandfold_and_runs_fast_only_with_dev_tools() {
    let mut app = input_app();
    for expected in [10.0, 100.0, 1000.0, 1.0, 10.0] {
        press(&mut app, &[GameAction::DevErosion]);
        app.update();
        assert_eq!(app.world().resource::<DevTools>().erosion_speed(), expected);
    }
    app.world_mut().resource_mut::<DevTools>().enabled = false;
    assert_eq!(app.world().resource::<DevTools>().erosion_speed(), 1.0);
}
