use super::*;
use bevy::input::mouse::MouseScrollUnit;

#[test]
fn cursor_options_change_only_when_capture_state_changes() {
    #[derive(Resource, Default)]
    struct PanelOpen(bool);
    let mut app = App::new();
    app.init_resource::<PanelOpen>().add_systems(
        Update,
        |panel: Res<PanelOpen>, mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>| {
            update_cursor_capture(panel.0, &mut cursor);
        },
    );
    let window = app
        .world_mut()
        .spawn((PrimaryWindow, CursorOptions::default()))
        .id();
    app.update();
    let cursor = app.world().get::<CursorOptions>(window).unwrap();
    assert_eq!(cursor.grab_mode, CursorGrabMode::Locked);
    assert!(!cursor.visible);
    let changed = app
        .world()
        .entity(window)
        .get_ref::<CursorOptions>()
        .unwrap()
        .last_changed();
    app.update();
    assert_eq!(
        app.world()
            .entity(window)
            .get_ref::<CursorOptions>()
            .unwrap()
            .last_changed(),
        changed
    );
    app.world_mut().resource_mut::<PanelOpen>().0 = true;
    app.update();
    let cursor = app.world().get::<CursorOptions>(window).unwrap();
    assert_eq!(cursor.grab_mode, CursorGrabMode::None);
    assert!(cursor.visible);
    app.world_mut().resource_mut::<PanelOpen>().0 = false;
    app.update();
    assert_eq!(
        app.world().get::<CursorOptions>(window).unwrap().grab_mode,
        CursorGrabMode::Locked
    );
    // External release must be noticed even if the desired state is unchanged.
    app.world_mut()
        .get_mut::<CursorOptions>(window)
        .unwrap()
        .grab_mode = CursorGrabMode::None;
    app.update();
    assert_eq!(
        app.world().get::<CursorOptions>(window).unwrap().grab_mode,
        CursorGrabMode::Locked
    );
}

#[test]
fn contextual_wheel_adjustments_suppress_zoom_only_when_consumed() {
    assert_eq!(contextual_zoom_delta(true, false, false), 1.0);
    assert_eq!(contextual_zoom_delta(false, true, false), -1.0);
    assert_eq!(contextual_zoom_delta(true, false, true), 0.0);
}

#[test]
fn pullback_is_clamped_and_zoom_memories_are_independent() {
    let mut camera = PlayerCamera::default();
    camera.set_target_pullback(false, 99.0);
    camera.set_target_pullback(true, -2.0);
    assert_eq!(camera.target_pullback(false), MAX_PULLBACK);
    assert_eq!(camera.target_pullback(true), 0.0);
}

#[test]
fn world_picker_releases_character_controls() {
    assert!(player_controls_blocked([false, false, false, true]));
    assert!(!player_controls_blocked([false; 4]));
}

#[test]
fn line_and_pixel_scroll_are_normalized() {
    assert_eq!(
        normalized_scroll(Vec2::new(0.0, 2.0), MouseScrollUnit::Line),
        2.0
    );
    assert_eq!(
        normalized_scroll(Vec2::new(0.0, 100.0), MouseScrollUnit::Pixel),
        2.0
    );
}

#[test]
fn damping_is_frame_rate_independent() {
    let one_step = damp(0.0, 10.0, 0.120);
    let two_steps = damp(damp(0.0, 10.0, 0.060), 10.0, 0.060);
    assert!((one_step - two_steps).abs() < 1.0e-5);
}

#[test]
fn zero_pullback_preserves_seated_first_person_transform() {
    let rotation = seated_view_rotation(Quat::from_rotation_y(0.4), 0.2, -0.1);
    let eye = Vec3::new(2.0, 3.0, 4.0);
    let transform = camera_transform(eye, rotation, 0.0);
    assert_eq!(transform.translation, eye);
    assert_eq!(transform.rotation, rotation);
}

#[test]
fn diagonal_camera_relative_movement_has_unit_speed() {
    let movement = camera_relative_movement(Vec2::ONE.normalize(), 0.0);
    assert!((movement.length() - 1.0).abs() < 1.0e-6);
}

#[test]
fn a_and_d_strafe_to_the_cameras_left_and_right() {
    let camera = PlayerCamera {
        yaw: 0.7,
        ..default()
    };
    let camera_right = camera.look_rotation() * Vec3::X;
    let mut keyboard = ButtonInput::default();

    keyboard.press(GameAction::MoveRight);
    let right = camera_relative_movement(movement_axis(&keyboard), camera.yaw);
    assert!(right.abs_diff_eq(camera_right, 1.0e-6));

    keyboard.release(GameAction::MoveRight);
    keyboard.press(GameAction::MoveLeft);
    let left = camera_relative_movement(movement_axis(&keyboard), camera.yaw);
    assert!(left.abs_diff_eq(-camera_right, 1.0e-6));
}

#[test]
fn walking_step_remains_available_to_world_gestures() {
    let step = walking_step(Vec2::Y, 0.0, 0.25, false);
    assert!((step - Vec3::Z).length() < 1.0e-6);
}

#[test]
fn sprinting_uses_the_faster_movement_speed() {
    let walking = walking_step(Vec2::Y, 0.0, 1.0, false);
    let sprinting = walking_step(Vec2::Y, 0.0, 1.0, true);
    assert!((walking.length() - WALK_SPEED).abs() < 1.0e-6);
    assert!((sprinting.length() - SPRINT_SPEED).abs() < 1.0e-6);
}

#[test]
fn platform_clamp_keeps_feet_on_ground() {
    assert_eq!(
        clamp_to_platform(Vec3::new(30.0, 8.0, -40.0)),
        Vec3::new(GROUND_HALF_SIZE, 0.0, -GROUND_HALF_SIZE)
    );
}

#[test]
fn player_lifecycle_starts_standing_and_exit_clamps_garage_horizontal_position() {
    let mut player = PlayerState::default();
    assert!(player.seat.is_none());
    assert_eq!(player.position.y, 5.0);
    player.leave_seat_at(Vec3::new(30.0, 8.0, -40.0), AppSpace::Garage);
    assert_eq!(
        player.position,
        Vec3::new(GROUND_HALF_SIZE, 8.0, -GROUND_HALF_SIZE)
    );
    assert!(player.seat.is_none());
}

#[test]
fn leaving_a_world_seat_stays_with_a_moved_creation() {
    let mut player = PlayerState::default();
    let beside_creation = Vec3::new(42.0, 8.0, -37.0);

    player.leave_seat_at(beside_creation, AppSpace::World);

    assert_eq!(player.position, beside_creation);
    assert!(player.seat.is_none());
}

#[test]
fn loaded_bounds_place_player_outside_and_facing_the_creation() {
    let mut player = PlayerState::default();
    let mut camera = PlayerCamera::default();
    player.place_outside_bounds(
        &mut camera,
        Vec3::new(-2.0, 0.0, -2.0),
        Vec3::new(2.0, 3.0, 2.0),
    );
    assert_eq!(player.position.y, 0.0);
    let forward = camera_relative_movement(Vec2::Y, camera.yaw);
    let toward = (Vec3::ZERO - player.position).normalize_or_zero();
    assert!(forward.dot(toward) > 0.99);
}

#[test]
#[allow(clippy::cast_precision_loss)] // Test sector counts are small.
fn material_sectors_run_clockwise_from_the_top_and_centre_cancels() {
    let context = Some(WheelChoice::ConstructionMaterial(
        ConstructionMaterial::Steel,
    ));
    assert_eq!(choice_at_selector(Vec2::ZERO, context), None);
    for (index, material) in ConstructionMaterial::ALL.into_iter().enumerate() {
        let angle = TAU * index as f32 / ConstructionMaterial::ALL.len() as f32;
        let direction = Vec2::new(angle.sin(), -angle.cos()) * 80.0;
        assert_eq!(
            choice_at_selector(direction, context),
            Some(WheelChoice::ConstructionMaterial(material))
        );
    }
}

#[test]
#[allow(clippy::cast_precision_loss)] // Test sector counts are small.
fn item_and_terrain_selectors_cover_every_contextual_sector() {
    for (index, item) in PlaceableItem::ALL.into_iter().enumerate() {
        let angle = TAU * index as f32 / PlaceableItem::ALL.len() as f32;
        let direction = Vec2::new(angle.sin(), -angle.cos()) * 80.0;
        assert_eq!(
            choice_at_selector(direction, Some(WheelChoice::Item(PlaceableItem::Bearing))),
            Some(WheelChoice::Item(item)),
        );
    }
    for (index, mode) in crate::shape_tool::ShapeEditMode::ALL
        .into_iter()
        .enumerate()
    {
        let angle = TAU * index as f32 / crate::shape_tool::ShapeEditMode::ALL.len() as f32;
        let direction = Vec2::new(angle.sin(), -angle.cos()) * 80.0;
        assert_eq!(
            choice_at_selector(
                direction,
                Some(WheelChoice::ShapeMode(
                    crate::shape_tool::ShapeEditMode::Vertex,
                )),
            ),
            Some(WheelChoice::ShapeMode(mode)),
        );
    }
    let materials = TerrainMaterial::ALL;
    for (index, material) in materials.into_iter().enumerate() {
        let angle = TAU * index as f32 / materials.len() as f32;
        let direction = Vec2::new(angle.sin(), -angle.cos()) * 80.0;
        assert_eq!(
            choice_at_selector(
                direction,
                Some(WheelChoice::TerrainMaterial(TerrainMaterial::Soil)),
            ),
            Some(WheelChoice::TerrainMaterial(material)),
        );
    }
}

#[test]
fn avatar_fades_between_first_and_third_person_thresholds() {
    assert_eq!(avatar_alpha(AVATAR_HIDDEN_PULLBACK), 0.0);
    assert_eq!(avatar_alpha(AVATAR_OPAQUE_PULLBACK), 1.0);
    assert!((avatar_alpha(0.675) - 0.5).abs() < 1.0e-5);
}

#[test]
fn reticle_ray_uses_the_viewport_centre() {
    assert_eq!(
        viewport_center(Vec2::new(1600.0, 900.0)),
        Vec2::new(800.0, 450.0)
    );
}

#[test]
fn drag_dead_zone_is_angular() {
    let start = Vec3::Z;
    assert!(!ray_drag_started(
        start,
        Quat::from_rotation_y(0.002) * start
    ));
    assert!(ray_drag_started(
        start,
        Quat::from_rotation_y(0.008) * start
    ));
}

#[test]
fn seat_entry_requires_a_seat_inside_three_metres() {
    assert!(seat_entry_allowed(3.0, true));
    assert!(!seat_entry_allowed(3.001, true));
    assert!(!seat_entry_allowed(1.0, false));
}

#[test]
fn material_wheel_respects_input_ownership_and_commits_only_on_release() {
    assert!(material_wheel_may_open(false, false, false));
    assert!(!material_wheel_may_open(true, false, false));
    assert!(!material_wheel_may_open(false, true, false));
    assert!(!material_wheel_may_open(false, false, true));
    assert_eq!(
        committed_choice(
            true,
            Some(WheelChoice::ConstructionMaterial(
                ConstructionMaterial::Rubber
            ))
        ),
        Some(WheelChoice::ConstructionMaterial(
            ConstructionMaterial::Rubber
        ))
    );
    assert_eq!(
        committed_choice(
            false,
            Some(WheelChoice::ConstructionMaterial(
                ConstructionMaterial::Rubber
            ))
        ),
        None
    );
    assert_eq!(committed_choice(true, None), None);
}

#[test]
fn material_wheel_opens_on_and_retains_the_current_material_in_its_dead_zone() {
    let mut wheel = MaterialWheelState::default();
    wheel.open(WheelChoice::ConstructionMaterial(
        ConstructionMaterial::Wood,
    ));

    assert!(wheel.open);
    assert_eq!(
        wheel.highlighted,
        Some(WheelChoice::ConstructionMaterial(
            ConstructionMaterial::Wood
        ))
    );

    wheel.move_selector(Vec2::new(1.0, 1.0));
    assert_eq!(
        wheel.highlighted,
        Some(WheelChoice::ConstructionMaterial(
            ConstructionMaterial::Wood
        ))
    );

    wheel.move_selector(Vec2::new(0.0, -100.0));
    assert_eq!(
        wheel.highlighted,
        Some(WheelChoice::ConstructionMaterial(
            ConstructionMaterial::Aluminium
        ))
    );
    wheel.move_selector(Vec2::new(-1.0, 99.0));
    assert_eq!(
        wheel.highlighted,
        Some(WheelChoice::ConstructionMaterial(
            ConstructionMaterial::Aluminium
        ))
    );
}

#[test]
fn tapping_a_two_choice_selector_toggles_and_aiming_commits_the_highlight() {
    use crate::hotbar::WeldMode;
    let mut wheel = MaterialWheelState::default();
    wheel.open(WheelChoice::WeldMode(WeldMode::Join));
    wheel.move_selector(Vec2::new(2.0, -3.0));
    assert_eq!(
        wheel.released_choice(),
        Some(WheelChoice::WeldMode(WeldMode::Place))
    );

    wheel.open(WheelChoice::WeldMode(WeldMode::Place));
    wheel.move_selector(Vec2::new(0.0, -80.0));
    wheel.move_selector(Vec2::new(0.0, 78.0));
    assert_eq!(
        wheel.released_choice(),
        Some(WheelChoice::WeldMode(WeldMode::Join))
    );

    wheel.open(WheelChoice::ShapeMode(
        crate::shape_tool::ShapeEditMode::Chamfer,
    ));
    assert_eq!(
        wheel.released_choice(),
        Some(WheelChoice::ShapeMode(
            crate::shape_tool::ShapeEditMode::Chamfer
        ))
    );
}

#[test]
fn chroma_configuration_uses_the_selector_without_a_radial_choice() {
    let mut wheel = MaterialWheelState::default();
    wheel.open_chroma_config();

    assert!(wheel.open);
    assert!(wheel.chroma_config);
    assert_eq!(wheel.highlighted, None);
    assert_eq!(wheel.selector, Vec2::ZERO);

    wheel.close();
    assert_eq!(wheel, MaterialWheelState::default());
}

#[test]
fn chroma_configuration_toggles_on_press_instead_of_release() {
    assert!(!chroma_config_should_close(false, false, true));
    assert!(chroma_config_should_close(true, false, true));
    assert!(chroma_config_should_close(false, false, false));
}
