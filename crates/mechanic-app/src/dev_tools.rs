//! Launch-opt-in developer navigation. None of these modes are player abilities.

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::camera::{MainCamera, PlayerCamera, PlayerState};
use crate::controls::GameAction;
use crate::world::{AppSpace, WorldListState, WorldRuntime};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum DevMode {
    #[default]
    Normal,
    Noclip,
    Spectator,
}

#[derive(Clone, Copy, Debug)]
struct ReturnPose {
    position: Vec3,
    view: PlayerCamera,
}

#[expect(
    clippy::struct_excessive_bools,
    reason = "independent navigation input and day-cycle switches"
)]
#[derive(Resource, Debug)]
pub(crate) struct DevTools {
    pub(crate) enabled: bool,
    pub(crate) mode: DevMode,
    pub(crate) speed: f32,
    pub(crate) cycle_paused: bool,
    pub(crate) notice: &'static str,
    return_pose: Option<ReturnPose>,
    spectator: Transform,
    spectator_view: PlayerCamera,
    movement: Vec2,
    vertical: f32,
    sprint: bool,
    input_active: bool,
}

impl Default for DevTools {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: DevMode::Normal,
            speed: 1.0,
            cycle_paused: false,
            notice: "",
            return_pose: None,
            spectator: Transform::IDENTITY,
            spectator_view: PlayerCamera::default(),
            movement: Vec2::ZERO,
            vertical: 0.0,
            sprint: false,
            input_active: false,
        }
    }
}

impl DevTools {
    pub(crate) fn from_env() -> Self {
        Self {
            enabled: crate::env::flag(crate::env::DEV_TOOLS),
            ..Self::default()
        }
    }

    pub(crate) fn spectator(&self) -> bool {
        self.enabled && self.mode == DevMode::Spectator
    }

    pub(crate) fn noclip(&self) -> bool {
        self.enabled && self.mode == DevMode::Noclip
    }

    pub(crate) fn multiplier(&self) -> f32 {
        if self.enabled { self.speed } else { 1.0 }
    }

    pub(crate) fn focus(&self, player: &PlayerState) -> Option<Vec3> {
        match self.mode {
            DevMode::Normal => None,
            DevMode::Noclip => Some(player.position),
            DevMode::Spectator => Some(self.spectator.translation),
        }
        .filter(|_| self.enabled)
    }

    /// Follow the dev view while retaining terrain for the player or noclip return.
    pub(crate) fn streaming_focus(
        &self,
        player: &PlayerState,
        live_player: mechanic_world::WorldPosition,
        origin: mechanic_world::FloatingOrigin,
    ) -> (
        mechanic_world::WorldPosition,
        Option<mechanic_world::WorldPosition>,
    ) {
        let Some(local) = self.focus(player) else {
            return (live_player, None);
        };
        let interest = self.return_position().map_or(live_player, |position| {
            mechanic_world::WorldPosition(origin.0 + position.as_dvec3())
        });
        (
            mechanic_world::WorldPosition(origin.0 + local.as_dvec3()),
            Some(interest),
        )
    }

    pub(crate) fn return_position(&self) -> Option<Vec3> {
        self.return_pose
            .filter(|_| self.noclip())
            .map(|pose| pose.position)
    }

    fn adjust_speed(&mut self, decrease: bool, increase: bool, reset: bool) {
        if reset {
            self.speed = 1.0;
        } else if decrease != increase {
            self.speed = (self.speed * if increase { 2.0 } else { 0.5 }).clamp(0.125, 32.0);
        }
    }

    /// Restore the player before saving or replacing a space. Spectator never moves it.
    pub(crate) fn exit(&mut self, player: &mut PlayerState, view: &mut PlayerCamera) -> bool {
        let noclip = self.noclip();
        if let Some(pose) = self.return_pose.take() {
            if noclip {
                player.position = pose.position;
            }
            *view = pose.view;
        }
        self.mode = DevMode::Normal;
        self.input_active = false;
        self.notice = "";
        noclip
    }

    fn enter(
        &mut self,
        mode: DevMode,
        player: &PlayerState,
        view: PlayerCamera,
        transform: Transform,
    ) {
        self.return_pose = Some(ReturnPose {
            position: player.position,
            view,
        });
        self.mode = mode;
        self.spectator = transform;
        // Derive world yaw/pitch even when entering from a rotated seat.
        let forward = transform.rotation * Vec3::NEG_Z;
        self.spectator_view = PlayerCamera::default();
        self.spectator_view.yaw = forward.x.atan2(forward.z);
        self.spectator_view.pitch = forward
            .y
            .asin()
            .clamp(crate::camera::MIN_PITCH, crate::camera::MAX_PITCH);
    }

    /// Owns only the developer camera path; ordinary camera/garage movement stays in camera.rs.
    pub(crate) fn update_camera(
        &mut self,
        player: &mut PlayerState,
        view: &mut PlayerCamera,
        transform: &mut Transform,
        motion: Vec2,
        seconds: f32,
        active: bool,
    ) {
        let active = active && self.input_active;
        if self.spectator() {
            if active {
                self.spectator_view.yaw -= motion.x * crate::camera::MOUSE_SENSITIVITY;
                self.spectator_view.pitch = (self.spectator_view.pitch
                    - motion.y * crate::camera::MOUSE_SENSITIVITY)
                    .clamp(crate::camera::MIN_PITCH, crate::camera::MAX_PITCH);
                self.spectator.rotation = self.spectator_view.look_rotation();
                self.spectator.translation += self.step(self.spectator.rotation, seconds);
            }
            *transform = self.spectator;
        } else if self.noclip() {
            if active {
                player.position += self.step(view.look_rotation(), seconds);
            }
            player.crouch = 0.0;
            *transform = view.apply_pullback(
                player.position + Vec3::Y * crate::camera::EYE_HEIGHT,
                view.look_rotation(),
            );
        }
    }

    fn step(&self, rotation: Quat, seconds: f32) -> Vec3 {
        crate::camera::flight_step(self.movement, rotation, self.vertical, seconds, self.sprint)
            * self.multiplier()
    }
}

#[derive(SystemParam)]
pub(crate) struct DevInputContext<'w, 's> {
    pause: Res<'w, crate::pause_menu::PauseMenuState>,
    menu: Res<'w, crate::creation_menu::CreationMenuState>,
    panel: Res<'w, crate::control_panel::ControlPanelState>,
    wheel: Res<'w, crate::camera::MaterialWheelState>,
    overlay: Res<'w, crate::ui::UiInput>,
    worlds: Res<'w, WorldListState>,
    windows: Query<'w, 's, &'static Window, With<PrimaryWindow>>,
    physical: Res<'w, crate::physical_controls::PhysicalControls>,
    editor: Res<'w, crate::editor::state::EditorState>,
}

impl DevInputContext<'_, '_> {
    fn active(&self) -> bool {
        !self.pause.blocks_world_input()
            && self.pause.binding_capture().is_none()
            && !self.menu.is_open()
            && !self.panel.is_open()
            && !self.wheel.open
            && !self.overlay.blocks_keyboard()
            && !self.worlds.is_open()
            && !self.physical.captures_pointer()
            && !crate::automation::enabled()
            && self.windows.iter().any(|window| window.focused)
    }
}

/// Resolve modes before world actions and keep spectator input out of gameplay.
pub(crate) fn input(
    mut dev: ResMut<DevTools>,
    mut actions: ResMut<ButtonInput<GameAction>>,
    context: DevInputContext,
    mut player: ResMut<PlayerState>,
    mut camera: Single<(&mut PlayerCamera, &mut Transform, &mut GlobalTransform), With<MainCamera>>,
    mut runtime: ResMut<WorldRuntime>,
    space: Res<State<AppSpace>>,
) {
    if !dev.enabled {
        return;
    }
    let (view, transform, global) = &mut *camera;
    if actions.just_pressed(GameAction::ToggleSpace) || context.worlds.is_open() {
        if dev.exit(&mut player, view) {
            runtime.reset_dev_motion();
        }
        return;
    }
    let active = context.active();
    if active {
        if *space.get() == AppSpace::World {
            if actions.just_pressed(GameAction::DevTimePause) {
                dev.cycle_paused = !dev.cycle_paused;
            }
            let hours = i32::from(actions.just_pressed(GameAction::DevTimeLater))
                - i32::from(actions.just_pressed(GameAction::DevTimeEarlier));
            runtime.advance_day(f64::from(hours) * 3600.0);
        }
        dev.adjust_speed(
            actions.just_pressed(GameAction::DevSpeedDecrease),
            actions.just_pressed(GameAction::DevSpeedIncrease),
            actions.just_pressed(GameAction::DevSpeedReset),
        );
        let requested = if actions.just_pressed(GameAction::DevNoclip) {
            Some(DevMode::Noclip)
        } else if actions.just_pressed(GameAction::DevSpectator) {
            Some(DevMode::Spectator)
        } else {
            None
        };
        if let Some(requested) = requested.filter(|_| !context.editor.contextual_selector_blocked())
        {
            if requested == DevMode::Noclip && player.seat.is_some() {
                dev.notice = "Leave the seat before enabling noclip";
            } else {
                let previous = dev.mode;
                if dev.exit(&mut player, view) {
                    runtime.reset_dev_motion();
                }
                if previous != DevMode::Normal && player.seat.is_none() {
                    **transform = view.apply_pullback(
                        player.position + Vec3::Y * crate::camera::eye_height(player.crouch),
                        view.look_rotation(),
                    );
                    **global = GlobalTransform::from(**transform);
                }
                if previous != requested {
                    dev.enter(requested, &player, **view, **transform);
                    if requested == DevMode::Noclip && *space.get() == AppSpace::World {
                        runtime.reset_dev_motion();
                    }
                }
            }
        }
    }
    dev.input_active = active;
    dev.movement = crate::camera::movement_axis(&actions);
    dev.vertical = f32::from(actions.pressed(GameAction::Jump))
        - f32::from(actions.pressed(GameAction::Descend));
    dev.sprint = actions.pressed(GameAction::Sprint);
    if dev.spectator() {
        for action in GameAction::ALL {
            if !action.is_dev() && action != GameAction::ToggleHelp {
                actions.reset(action);
            }
        }
    }
}

pub(crate) fn gameplay_enabled(dev: Option<Res<DevTools>>) -> bool {
    !dev.is_some_and(|dev| dev.spectator())
}

/// Clear a mode after the world selector opens, before the next state transition.
pub(crate) fn leave_for_world_selector(
    worlds: Res<WorldListState>,
    mut dev: ResMut<DevTools>,
    mut player: ResMut<PlayerState>,
    mut view: Single<&mut PlayerCamera, With<MainCamera>>,
    mut runtime: ResMut<WorldRuntime>,
) {
    if worlds.is_open() && dev.exit(&mut player, &mut view) {
        runtime.reset_dev_motion();
    }
}

#[cfg(test)]
mod tests;
