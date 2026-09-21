use crate::settings::AppSettings;
use std::f32::consts::{FRAC_PI_2, PI, TAU};

use bevy::{
    input::mouse::AccumulatedMouseMotion,
    prelude::*,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
#[cfg(test)]
use mechanic_core::ConstructionMaterial;
use mechanic_core::PartId;
#[cfg(test)]
use mechanic_world::TerrainMaterial;

use crate::editor::state::EditorState;
#[cfg(test)]
use crate::hotbar::PlaceableItem;
use crate::{
    builder::GROUND_HALF_SIZE,
    control_panel::ControlPanelState,
    controls::GameAction,
    creation_menu::CreationMenuState,
    garage,
    hotbar::{
        MainTool, MatterMode, SelectedMaterial, SelectedTerrainMaterial, SelectedTool, WheelChoice,
    },
    pause_menu::PauseMenuState,
    ui::UiInput,
    world::{AppSpace, WorldListState},
};

pub(crate) const EYE_HEIGHT: f32 = 1.65;
pub(crate) const CROUCHED_EYE_HEIGHT: f32 = 0.92;
pub(crate) const SEATED_EYE_HEIGHT: f32 = 0.475;
pub(crate) const MAX_PULLBACK: f32 = 12.0;
pub(crate) const AVATAR_HIDDEN_PULLBACK: f32 = 0.35;
pub(crate) const AVATAR_OPAQUE_PULLBACK: f32 = 1.0;
pub(crate) const MAX_PITCH: f32 = FRAC_PI_2 - 0.08;
pub(crate) const MIN_PITCH: f32 = -MAX_PITCH;
pub(crate) const WALK_SPEED: f32 = 4.0;
pub(crate) const SPRINT_SPEED: f32 = 7.0;
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
    /// How far the player is crouched, `0.0` standing and `1.0` fully crouched.
    pub(crate) crouch: f32,
}

impl Default for PlayerState {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, garage::BUILD_MIN_Y, -6.0),
            seat: None,
            input_captured: false,
            crouch: 0.0,
        }
    }
}

impl PlayerState {
    pub(crate) const fn world_input_active(&self) -> bool {
        self.input_captured
    }

    pub(crate) fn leave_seat_at(&mut self, position: Vec3, space: AppSpace) {
        self.position = match space {
            AppSpace::Garage => clamp_to_platform_horizontal(position),
            AppSpace::World => position,
        };
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
    pub(crate) chroma_config: bool,
    pub(crate) selector: Vec2,
    pub(crate) highlighted: Option<WheelChoice>,
    context: Option<WheelChoice>,
    aimed: bool,
}

impl MaterialWheelState {
    pub(crate) fn open(&mut self, current: WheelChoice) {
        self.open = true;
        self.chroma_config = false;
        self.selector = Vec2::ZERO;
        self.highlighted = Some(current);
        self.context = Some(current);
        self.aimed = false;
    }

    /// The choice a Tab release commits. A two-choice selector released
    /// without aiming switches to the other choice, so a tap toggles.
    pub(crate) fn released_choice(&self) -> Option<WheelChoice> {
        if !self.aimed
            && let Some(current) = self.context
            && current.context().count() == 2
        {
            return current
                .context()
                .choices()
                .find(|&choice| choice != current);
        }
        self.highlighted
    }

    pub(crate) fn open_chroma_config(&mut self) {
        self.open = true;
        self.chroma_config = true;
        self.selector = Vec2::ZERO;
        self.highlighted = None;
        self.context = None;
    }

    fn move_selector(&mut self, delta: Vec2) {
        self.selector = (self.selector + delta).clamp_length_max(MATERIAL_WHEEL_RADIUS);
        self.aimed |= self.selector.length() >= MATERIAL_WHEEL_DEAD_ZONE;
        if let Some(choice) = choice_at_selector(self.selector, self.context) {
            self.highlighted = Some(choice);
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

#[cfg(test)]
pub(crate) fn camera_relative_movement(axis: Vec2, yaw: f32) -> Vec3 {
    let forward = Vec3::new(yaw.sin(), 0.0, yaw.cos());
    let right = Vec3::new(-forward.z, 0.0, forward.x);
    (right * axis.x + forward * axis.y).normalize_or_zero()
}

#[cfg(test)]
pub(crate) fn walking_step(axis: Vec2, yaw: f32, delta_seconds: f32, sprinting: bool) -> Vec3 {
    let speed = if sprinting { SPRINT_SPEED } else { WALK_SPEED };
    camera_relative_movement(axis, yaw) * speed * delta_seconds
}

pub(crate) fn flight_step(
    axis: Vec2,
    look_rotation: Quat,
    vertical: f32,
    delta_seconds: f32,
    sprinting: bool,
) -> Vec3 {
    let speed = if sprinting { SPRINT_SPEED } else { WALK_SPEED };
    let forward = look_rotation * Vec3::NEG_Z;
    let right = look_rotation * Vec3::X;
    (right * axis.x + forward * axis.y + Vec3::Y * vertical).normalize_or_zero()
        * speed
        * delta_seconds
}

pub(crate) fn clamp_to_garage(position: Vec3) -> Vec3 {
    let half = garage::SIDE_LENGTH * 0.5;
    Vec3::new(
        position.x.clamp(-half, half),
        position.y.clamp(0.0, garage::HEIGHT - EYE_HEIGHT),
        position.z.clamp(-half, half),
    )
}

pub(crate) fn clamp_to_platform(position: Vec3) -> Vec3 {
    Vec3::new(
        position.x.clamp(-GROUND_HALF_SIZE, GROUND_HALF_SIZE),
        0.0,
        position.z.clamp(-GROUND_HALF_SIZE, GROUND_HALF_SIZE),
    )
}

fn clamp_to_platform_horizontal(position: Vec3) -> Vec3 {
    let clamped = clamp_to_platform(position);
    Vec3::new(clamped.x, position.y, clamped.z)
}

/// Eye height for a stance, dropping with the capsule as the player crouches.
pub(crate) fn eye_height(crouch: f32) -> f32 {
    EYE_HEIGHT.lerp(CROUCHED_EYE_HEIGHT, crouch.clamp(0.0, 1.0))
}

pub(crate) fn avatar_alpha(pullback: f32) -> f32 {
    ((pullback - AVATAR_HIDDEN_PULLBACK) / (AVATAR_OPAQUE_PULLBACK - AVATAR_HIDDEN_PULLBACK))
        .clamp(0.0, 1.0)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub(crate) fn choice_at_selector(
    selector: Vec2,
    current: Option<WheelChoice>,
) -> Option<WheelChoice> {
    if selector.length() < MATERIAL_WHEEL_DEAD_ZONE {
        return None;
    }
    let context = current?.context();
    let count = context.count();
    let angle = selector.x.atan2(-selector.y).rem_euclid(TAU);
    let sector = TAU / count as f32;
    let centred = (angle + sector * 0.5).rem_euclid(TAU);
    let index = (centred / sector).floor() as usize % count;
    context.choice(index)
}

pub(crate) const fn material_wheel_may_open(
    simulating: bool,
    interactive_panel: bool,
    world_drag: bool,
) -> bool {
    !simulating && !interactive_panel && !world_drag
}

pub(crate) const fn committed_choice(
    tab_released: bool,
    highlighted: Option<WheelChoice>,
) -> Option<WheelChoice> {
    if tab_released { highlighted } else { None }
}

const fn chroma_config_should_close(
    selector_pressed: bool,
    another_panel_open: bool,
    chroma_active: bool,
) -> bool {
    selector_pressed || another_panel_open || !chroma_active
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn update_material_wheel(
    actions: Res<ButtonInput<GameAction>>,
    motion: Res<AccumulatedMouseMotion>,
    menu: Res<CreationMenuState>,
    panel: Res<ControlPanelState>,
    pause: Res<PauseMenuState>,
    overlay: Res<UiInput>,
    state: Res<EditorState>,
    mut wheel: ResMut<MaterialWheelState>,
    mut material: ResMut<SelectedMaterial>,
    mut terrain_material: ResMut<SelectedTerrainMaterial>,
    mut selection: ResMut<SelectedTool>,
    mut shape_mode: ResMut<crate::shape_tool::ShapeEditMode>,
) {
    if wheel.open {
        let chroma_active = selection.tool == Some(MainTool::MatterManipulator)
            && selection.matter_mode == MatterMode::Chroma;
        if wheel.chroma_config {
            if chroma_config_should_close(
                actions.just_pressed(GameAction::MaterialWheel),
                menu.is_open() || panel.is_open() || pause.blocks_world_input(),
                chroma_active,
            ) {
                wheel.close();
            }
            return;
        }
        wheel.move_selector(motion.delta);
        if actions.just_released(GameAction::MaterialWheel)
            || menu.is_open()
            || panel.is_open()
            || pause.blocks_world_input()
        {
            if let Some(highlighted) = committed_choice(
                actions.just_released(GameAction::MaterialWheel),
                wheel.released_choice(),
            ) {
                match highlighted {
                    WheelChoice::ConstructionMaterial(next) => material.0 = next,
                    WheelChoice::Item(next) => selection.select_item(next),
                    WheelChoice::TerrainMaterial(next) => terrain_material.0 = next,
                    WheelChoice::ShapeMode(next) => *shape_mode = next,
                    WheelChoice::WeldMode(next) => selection.weld_mode = next,
                }
            }
            wheel.close();
        }
        return;
    }
    let interactive_panel =
        menu.is_open() || panel.is_open() || overlay.blocks_pointer() || overlay.blocks_keyboard();
    let current = match (selection.tool, selection.matter_mode) {
        (
            Some(MainTool::MatterManipulator),
            MatterMode::Block | MatterMode::Cylinder | MatterMode::Layer,
        ) => Some(Some(WheelChoice::ConstructionMaterial(material.0))),
        (Some(MainTool::MatterManipulator), MatterMode::Item) => {
            Some(Some(WheelChoice::Item(selection.item)))
        }
        (Some(MainTool::MatterManipulator), MatterMode::Terrain) => {
            Some(Some(WheelChoice::TerrainMaterial(terrain_material.0)))
        }
        (Some(MainTool::MatterManipulator), MatterMode::Chroma) => Some(None),
        (Some(MainTool::MatterManipulator), MatterMode::Manipulate) => {
            Some(Some(WheelChoice::ShapeMode(*shape_mode)))
        }
        (Some(MainTool::Welder), _) => Some(Some(WheelChoice::WeldMode(selection.weld_mode))),
        _ => None,
    };
    if actions.just_pressed(GameAction::MaterialWheel)
        && material_wheel_may_open(
            false,
            interactive_panel,
            state.contextual_selector_blocked(),
        )
        && let Some(current) = current
    {
        if let Some(choice) = current {
            wheel.open(choice);
        } else {
            wheel.open_chroma_config();
        }
    }
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn update_player_camera(
    time: Res<Time>,
    actions: Res<ButtonInput<GameAction>>,
    motion: Res<AccumulatedMouseMotion>,
    menu: Res<CreationMenuState>,
    panel: Res<ControlPanelState>,
    pause: Res<PauseMenuState>,
    worlds: Res<WorldListState>,
    wheel: Res<MaterialWheelState>,
    editor: Res<EditorState>,
    selection: Res<SelectedTool>,
    space: Res<State<AppSpace>>,
    mut player: ResMut<PlayerState>,
    mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>,
    mut camera: Single<(&mut PlayerCamera, &mut Transform, &mut GlobalTransform), With<MainCamera>>,
) {
    let panel_open = crate::automation::enabled()
        || wheel.chroma_config
        || player_controls_blocked([
            menu.is_open(),
            panel.is_open(),
            pause.is_open(),
            worlds.is_open(),
        ]);
    update_cursor_capture(panel_open, &mut cursor);
    player.input_captured = !panel_open && !pause.blocks_world_input();

    let (view, transform, global) = &mut *camera;
    let world_active = player.input_captured && !wheel.open;
    if world_active
        && editor.suspension.controls.gesture.is_none()
        && !(editor.suspension.controls.aim.is_some() && actions.just_pressed(GameAction::Primary))
    {
        view.yaw -= motion.delta.x * MOUSE_SENSITIVITY;
        view.pitch = (view.pitch - motion.delta.y * MOUSE_SENSITIVITY).clamp(MIN_PITCH, MAX_PITCH);
        // The terrain brush and the Spiral tool's taper take the wheel.
        let terrain_mode = selection.tool == Some(MainTool::MatterManipulator)
            && matches!(
                selection.matter_mode,
                MatterMode::Terrain | MatterMode::Spiral
            );
        let zoom = contextual_zoom_delta(
            actions.just_pressed(GameAction::ZoomIn),
            actions.just_pressed(GameAction::ZoomOut),
            editor.pipe_bend_active()
                || terrain_mode
                || editor.smart_snap.range_adjusted_this_frame
                || editor.free_placement.range_adjusted_this_frame,
        );
        view.add_scroll(player.seat.is_some(), zoom);
    }
    view.damp_pullback(player.seat.is_some(), time.delta_secs());
    if player.seat.is_none() {
        if world_active && space.get().uses_garage_flight() {
            let vertical = f32::from(actions.pressed(GameAction::Jump))
                - f32::from(actions.pressed(GameAction::Descend));
            player.position += flight_step(
                movement_axis(&actions),
                view.look_rotation(),
                vertical,
                time.delta_secs(),
                actions.pressed(GameAction::Sprint),
            );
            player.position = clamp_to_garage(player.position);
        }
        **transform = view.apply_pullback(
            player.position + Vec3::Y * eye_height(player.crouch),
            view.look_rotation(),
        );
        **global = GlobalTransform::from(**transform);
    }
}

fn contextual_zoom_delta(zoom_in: bool, zoom_out: bool, wheel_consumed: bool) -> f32 {
    if wheel_consumed {
        0.0
    } else {
        f32::from(zoom_in) - f32::from(zoom_out)
    }
}

fn player_controls_blocked(open_panels: [bool; 4]) -> bool {
    open_panels.into_iter().any(core::convert::identity)
}

fn update_cursor_capture(
    panel_open: bool,
    cursor: &mut Single<&mut CursorOptions, With<PrimaryWindow>>,
) {
    let grab_mode = if panel_open {
        CursorGrabMode::None
    } else {
        CursorGrabMode::Locked
    };
    // Changed<CursorOptions> makes winit call the OS even for identical values.
    // Read the actual component so external releases are still recaptured.
    if cursor.grab_mode != grab_mode {
        cursor.grab_mode = grab_mode;
    }
    if cursor.visible != panel_open {
        cursor.visible = panel_open;
    }
}

#[cfg(test)]
#[expect(clippy::float_cmp)]
mod tests;
#[derive(Component)]
pub(crate) struct FovCamera;

pub(crate) fn apply_camera_fov(
    settings: Res<AppSettings>,
    mut cameras: Query<&mut Projection, With<FovCamera>>,
) {
    if !settings.is_changed() {
        return;
    }
    let fov = settings.camera_fov_degrees().to_radians();
    for mut projection in &mut cameras {
        set_projection_fov(&mut projection, fov);
    }
}

pub(crate) fn set_projection_fov(projection: &mut Projection, fov_radians: f32) {
    if let Projection::Perspective(perspective) = projection {
        perspective.fov = fov_radians;
    }
}
