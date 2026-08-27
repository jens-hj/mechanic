use std::f32::consts::{FRAC_PI_2, PI, TAU};

use bevy::{
    input::mouse::AccumulatedMouseMotion,
    prelude::*,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use mechanic_core::{ConstructionMaterial, PartId};

use crate::{
    AppSimulation, EditorState, builder::GROUND_HALF_SIZE, control_panel::ControlPanelState,
    controls::GameAction, creation_menu::CreationMenuState, hotbar::SelectedMaterial,
    pause_menu::PauseMenuState, ui::UiInput,
};

pub(crate) const EYE_HEIGHT: f32 = 1.65;
pub(crate) const SEATED_EYE_HEIGHT: f32 = 0.475;
pub(crate) const MAX_PULLBACK: f32 = 12.0;
pub(crate) const AVATAR_HIDDEN_PULLBACK: f32 = 0.35;
pub(crate) const AVATAR_OPAQUE_PULLBACK: f32 = 1.0;
pub(crate) const MAX_PITCH: f32 = FRAC_PI_2 - 0.08;
pub(crate) const MIN_PITCH: f32 = -MAX_PITCH;
pub(crate) const WALK_SPEED: f32 = 4.0;
pub(crate) const MOUSE_SENSITIVITY: f32 = 0.0025;
pub(crate) const PULLBACK_DAMPING_SECONDS: f32 = 0.120;
pub(crate) const MAX_CAMERA_LIFT: f32 = 0.35;
pub(crate) const MATERIAL_WHEEL_RADIUS: f32 = 104.0;
pub(crate) const MATERIAL_WHEEL_DEAD_ZONE: f32 = 24.0;

#[derive(Component, Debug, Default)]
pub(crate) struct MainCamera;

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PlayerState {
    pub(crate) position: Vec3,
    pub(crate) seat: Option<PartId>,
    pub(crate) input_captured: bool,
}

impl Default for PlayerState {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, -6.0),
            seat: None,
            input_captured: false,
        }
    }
}

impl PlayerState {
    pub(crate) const fn world_input_active(&self) -> bool {
        self.input_captured
    }

    pub(crate) fn leave_seat_at(&mut self, position: Vec3) {
        self.position = clamp_to_platform(Vec3::new(position.x, 0.0, position.z));
        self.seat = None;
    }

    pub(crate) fn place_outside_bounds(
        &mut self,
        camera: &mut PlayerCamera,
        minimum: Vec3,
        maximum: Vec3,
    ) {
        let centre = (minimum + maximum) * 0.5;
        let positive_z = maximum.z + 2.0;
        let negative_z = minimum.z - 2.0;
        let z = if positive_z <= GROUND_HALF_SIZE {
            positive_z
        } else {
            negative_z.max(-GROUND_HALF_SIZE)
        };
        self.position = clamp_to_platform(Vec3::new(
            centre.x.clamp(-GROUND_HALF_SIZE, GROUND_HALF_SIZE),
            0.0,
            z,
        ));
        let toward = centre - self.position;
        camera.yaw = toward.x.atan2(toward.z);
        camera.pitch = 0.0;
        self.seat = None;
    }
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PlayerCamera {
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
    foot_pullback: f32,
    seated_pullback: f32,
    current_pullback: f32,
}

impl Default for PlayerCamera {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            foot_pullback: 0.0,
            seated_pullback: 4.0,
            current_pullback: 0.0,
        }
    }
}

impl PlayerCamera {
    pub(crate) fn target_pullback(&self, seated: bool) -> f32 {
        if seated {
            self.seated_pullback
        } else {
            self.foot_pullback
        }
    }

    pub(crate) const fn current_pullback(&self) -> f32 {
        self.current_pullback
    }

    pub(crate) fn set_target_pullback(&mut self, seated: bool, pullback: f32) {
        let value = clamp_pullback(pullback);
        if seated {
            self.seated_pullback = value;
        } else {
            self.foot_pullback = value;
        }
    }

    pub(crate) fn add_scroll(&mut self, seated: bool, scroll: f32) {
        self.set_target_pullback(seated, self.target_pullback(seated) - scroll);
    }

    pub(crate) fn damp_pullback(&mut self, seated: bool, delta_seconds: f32) {
        self.current_pullback = damp(
            self.current_pullback,
            self.target_pullback(seated),
            delta_seconds,
        );
    }

    pub(crate) fn look_rotation(&self) -> Quat {
        free_look_rotation(Quat::IDENTITY, self.yaw, self.pitch)
    }

    pub(crate) fn apply_pullback(&self, eye: Vec3, rotation: Quat) -> Transform {
        camera_transform(eye, rotation, self.current_pullback)
    }
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct MaterialWheelState {
    pub(crate) open: bool,
    pub(crate) selector: Vec2,
    pub(crate) highlighted: Option<ConstructionMaterial>,
    selection_started: bool,
}

impl MaterialWheelState {
    pub(crate) fn open(&mut self, current_material: ConstructionMaterial) {
        self.open = true;
        self.selector = Vec2::ZERO;
        self.highlighted = Some(current_material);
        self.selection_started = false;
    }

    fn move_selector(&mut self, delta: Vec2) {
        self.selector = (self.selector + delta).clamp_length_max(MATERIAL_WHEEL_RADIUS);
        if let Some(material) = material_at_selector(self.selector) {
            self.highlighted = Some(material);
            self.selection_started = true;
        } else if self.selection_started {
            self.highlighted = None;
        }
    }

    pub(crate) fn close(&mut self) {
        *self = Self::default();
    }
}

pub(crate) fn seated_view_rotation(seat_rotation: Quat, yaw: f32, pitch: f32) -> Quat {
    free_look_rotation(seat_rotation, yaw, pitch)
}

pub(crate) fn free_look_rotation(base: Quat, yaw: f32, pitch: f32) -> Quat {
    base * Quat::from_rotation_y(PI) * Quat::from_rotation_y(yaw) * Quat::from_rotation_x(pitch)
}

pub(crate) const fn clamp_pullback(value: f32) -> f32 {
    value.clamp(0.0, MAX_PULLBACK)
}

#[cfg(test)]
pub(crate) fn normalized_scroll(delta: Vec2, unit: bevy::input::mouse::MouseScrollUnit) -> f32 {
    delta.y
        * match unit {
            bevy::input::mouse::MouseScrollUnit::Line => 1.0,
            bevy::input::mouse::MouseScrollUnit::Pixel => 0.02,
        }
}

pub(crate) fn damp(current: f32, target: f32, delta_seconds: f32) -> f32 {
    let blend = 1.0 - (-delta_seconds.max(0.0) / PULLBACK_DAMPING_SECONDS).exp();
    current + (target - current) * blend
}

pub(crate) fn camera_transform(eye: Vec3, rotation: Quat, pullback: f32) -> Transform {
    let pullback = clamp_pullback(pullback);
    let lift = MAX_CAMERA_LIFT * pullback / MAX_PULLBACK;
    let forward = rotation * -Vec3::Z;
    Transform::from_translation(eye - forward * pullback + Vec3::Y * lift).with_rotation(rotation)
}

pub(crate) fn viewport_center(size: Vec2) -> Vec2 {
    size * 0.5
}

pub(crate) fn ray_drag_started(start: Vec3, current: Vec3) -> bool {
    start.angle_between(current) > crate::DRAG_DEAD_ZONE_RADIANS
}

pub(crate) fn seat_entry_allowed(hit_distance: f32, is_seat: bool) -> bool {
    is_seat && hit_distance <= 3.0
}

pub(crate) fn movement_axis(actions: &ButtonInput<GameAction>) -> Vec2 {
    let mut axis = Vec2::ZERO;
    if actions.pressed(GameAction::MoveLeft) {
        axis.x -= 1.0;
    }
    if actions.pressed(GameAction::MoveRight) {
        axis.x += 1.0;
    }
    if actions.pressed(GameAction::MoveForward) {
        axis.y += 1.0;
    }
    if actions.pressed(GameAction::MoveBackward) {
        axis.y -= 1.0;
    }
    axis.normalize_or_zero()
}

pub(crate) fn camera_relative_movement(axis: Vec2, yaw: f32) -> Vec3 {
    let forward = Vec3::new(yaw.sin(), 0.0, yaw.cos());
    let right = Vec3::new(-forward.z, 0.0, forward.x);
    (right * axis.x + forward * axis.y).normalize_or_zero()
}

pub(crate) fn walking_step(axis: Vec2, yaw: f32, delta_seconds: f32) -> Vec3 {
    camera_relative_movement(axis, yaw) * WALK_SPEED * delta_seconds
}

pub(crate) fn clamp_to_platform(position: Vec3) -> Vec3 {
    Vec3::new(
        position.x.clamp(-GROUND_HALF_SIZE, GROUND_HALF_SIZE),
        0.0,
        position.z.clamp(-GROUND_HALF_SIZE, GROUND_HALF_SIZE),
    )
}

pub(crate) fn avatar_alpha(pullback: f32) -> f32 {
    ((pullback - AVATAR_HIDDEN_PULLBACK) / (AVATAR_OPAQUE_PULLBACK - AVATAR_HIDDEN_PULLBACK))
        .clamp(0.0, 1.0)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub(crate) fn material_at_selector(selector: Vec2) -> Option<ConstructionMaterial> {
    if selector.length() < MATERIAL_WHEEL_DEAD_ZONE {
        return None;
    }
    let angle = selector.x.atan2(-selector.y).rem_euclid(TAU);
    let sector = TAU / ConstructionMaterial::ALL.len() as f32;
    let centred = (angle + sector * 0.5).rem_euclid(TAU);
    let index = (centred / sector).floor() as usize % ConstructionMaterial::ALL.len();
    Some(ConstructionMaterial::ALL[index])
}

pub(crate) const fn material_wheel_may_open(
    simulating: bool,
    interactive_panel: bool,
    world_drag: bool,
) -> bool {
    !simulating && !interactive_panel && !world_drag
}

pub(crate) const fn committed_material(
    tab_released: bool,
    highlighted: Option<ConstructionMaterial>,
) -> Option<ConstructionMaterial> {
    if tab_released { highlighted } else { None }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn update_material_wheel(
    actions: Res<ButtonInput<GameAction>>,
    motion: Res<AccumulatedMouseMotion>,
    simulation: Res<AppSimulation>,
    menu: Res<CreationMenuState>,
    panel: Res<ControlPanelState>,
    pause: Res<PauseMenuState>,
    overlay: Res<UiInput>,
    state: Res<EditorState>,
    mut wheel: ResMut<MaterialWheelState>,
    mut material: ResMut<SelectedMaterial>,
) {
    if wheel.open {
        wheel.move_selector(motion.delta);
        if actions.just_released(GameAction::MaterialWheel)
            || simulation.is_running()
            || menu.is_open()
            || panel.is_open()
            || pause.blocks_world_input()
        {
            if let Some(highlighted) = committed_material(
                actions.just_released(GameAction::MaterialWheel),
                wheel.highlighted,
            ) {
                material.0 = highlighted;
            }
            wheel.close();
        }
        return;
    }
    let interactive_panel =
        menu.is_open() || panel.is_open() || overlay.blocks_pointer() || overlay.blocks_keyboard();
    if actions.just_pressed(GameAction::MaterialWheel)
        && material_wheel_may_open(
            simulation.is_running(),
            interactive_panel,
            state.world_drag_active(),
        )
    {
        wheel.open(material.0);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn update_player_camera(
    time: Res<Time>,
    actions: Res<ButtonInput<GameAction>>,
    motion: Res<AccumulatedMouseMotion>,
    menu: Res<CreationMenuState>,
    panel: Res<ControlPanelState>,
    pause: Res<PauseMenuState>,
    wheel: Res<MaterialWheelState>,
    editor: Res<EditorState>,
    mut player: ResMut<PlayerState>,
    mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>,
    mut camera: Single<(&mut PlayerCamera, &mut Transform, &mut GlobalTransform), With<MainCamera>>,
) {
    let panel_open = menu.is_open() || panel.is_open() || pause.is_open();
    cursor.grab_mode = if panel_open {
        CursorGrabMode::None
    } else {
        CursorGrabMode::Locked
    };
    cursor.visible = panel_open;
    player.input_captured = !panel_open && !pause.blocks_world_input();

    let (view, transform, global) = &mut *camera;
    let world_active = player.input_captured && !wheel.open;
    if world_active {
        view.yaw -= motion.delta.x * MOUSE_SENSITIVITY;
        view.pitch = (view.pitch - motion.delta.y * MOUSE_SENSITIVITY).clamp(MIN_PITCH, MAX_PITCH);
        let zoom = if editor.pipe_bend_active() {
            0.0
        } else {
            f32::from(actions.just_pressed(GameAction::ZoomIn))
                - f32::from(actions.just_pressed(GameAction::ZoomOut))
        };
        view.add_scroll(player.seat.is_some(), zoom);
    }
    view.damp_pullback(player.seat.is_some(), time.delta_secs());
    if player.seat.is_none() {
        if world_active {
            player.position += walking_step(movement_axis(&actions), view.yaw, time.delta_secs());
            player.position = clamp_to_platform(player.position);
        }
        **transform =
            view.apply_pullback(player.position + Vec3::Y * EYE_HEIGHT, view.look_rotation());
        **global = GlobalTransform::from(**transform);
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use bevy::input::mouse::MouseScrollUnit;

    #[test]
    fn pullback_is_clamped_and_zoom_memories_are_independent() {
        let mut camera = PlayerCamera::default();
        camera.set_target_pullback(false, 99.0);
        camera.set_target_pullback(true, -2.0);
        assert_eq!(camera.target_pullback(false), MAX_PULLBACK);
        assert_eq!(camera.target_pullback(true), 0.0);
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
        let step = walking_step(Vec2::Y, 0.0, 0.25);
        assert!((step - Vec3::Z).length() < 1.0e-6);
    }

    #[test]
    fn platform_clamp_keeps_feet_on_ground() {
        assert_eq!(
            clamp_to_platform(Vec3::new(30.0, 8.0, -40.0)),
            Vec3::new(GROUND_HALF_SIZE, 0.0, -GROUND_HALF_SIZE)
        );
    }

    #[test]
    fn player_lifecycle_starts_standing_and_exit_returns_to_safe_ground() {
        let mut player = PlayerState::default();
        assert!(player.seat.is_none());
        assert_eq!(player.position.y, 0.0);
        player.leave_seat_at(Vec3::new(30.0, 8.0, -40.0));
        assert_eq!(
            player.position,
            Vec3::new(GROUND_HALF_SIZE, 0.0, -GROUND_HALF_SIZE)
        );
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
    fn material_sectors_run_clockwise_from_the_top_and_centre_cancels() {
        assert_eq!(material_at_selector(Vec2::ZERO), None);
        let directions = [
            Vec2::new(0.0, -80.0),
            Vec2::new(47.0, -65.0),
            Vec2::new(76.0, -25.0),
            Vec2::new(76.0, 25.0),
            Vec2::new(47.0, 65.0),
            Vec2::new(0.0, 80.0),
            Vec2::new(-47.0, 65.0),
            Vec2::new(-76.0, 25.0),
            Vec2::new(-76.0, -25.0),
            Vec2::new(-47.0, -65.0),
        ];
        for (direction, material) in directions.into_iter().zip(ConstructionMaterial::ALL) {
            assert_eq!(material_at_selector(direction), Some(material));
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
            committed_material(true, Some(ConstructionMaterial::Rubber)),
            Some(ConstructionMaterial::Rubber)
        );
        assert_eq!(
            committed_material(false, Some(ConstructionMaterial::Rubber)),
            None
        );
        assert_eq!(committed_material(true, None), None);
    }

    #[test]
    fn material_wheel_opens_on_and_retains_the_current_material_in_its_dead_zone() {
        let mut wheel = MaterialWheelState::default();
        wheel.open(ConstructionMaterial::Wood);

        assert!(wheel.open);
        assert_eq!(wheel.highlighted, Some(ConstructionMaterial::Wood));

        wheel.move_selector(Vec2::new(1.0, 1.0));
        assert_eq!(wheel.highlighted, Some(ConstructionMaterial::Wood));

        wheel.move_selector(Vec2::new(0.0, -100.0));
        assert_eq!(wheel.highlighted, Some(ConstructionMaterial::Aluminium));
        wheel.move_selector(Vec2::new(-1.0, 99.0));
        assert_eq!(wheel.highlighted, None);
    }
}
