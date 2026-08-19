use std::f32::consts::FRAC_PI_2;

use bevy::{
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit},
    prelude::*,
};

use crate::hotbar::HotbarPointerCapture;

const MIN_PITCH: f32 = 0.08;
const MAX_PITCH: f32 = FRAC_PI_2 - 0.08;
const MIN_RADIUS: f32 = 3.0;
const MAX_RADIUS: f32 = 60.0;

#[derive(Component, Debug)]
pub(crate) struct OrbitCamera {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    radius: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::new(0.0, 1.0, 0.0),
            yaw: 0.75,
            pitch: 0.55,
            radius: 16.0,
        }
    }
}

impl OrbitCamera {
    fn apply_input(&mut self, mouse_delta: Vec2, scroll: f32, rotating: bool, panning: bool) {
        if rotating {
            self.yaw -= mouse_delta.x * 0.006;
            self.pitch = (self.pitch + mouse_delta.y * 0.006).clamp(MIN_PITCH, MAX_PITCH);
        } else if panning {
            let scale = self.radius * 0.0015;
            let right = Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin());
            let forward = Vec3::new(-self.yaw.sin(), 0.0, -self.yaw.cos());
            self.target += (-right * mouse_delta.x + forward * mouse_delta.y) * scale;
        }
        self.radius = (self.radius * (-scroll * 0.12).exp()).clamp(MIN_RADIUS, MAX_RADIUS);
    }

    pub(crate) fn transform(&self) -> Transform {
        let horizontal = self.radius * self.pitch.cos();
        let offset = Vec3::new(
            horizontal * self.yaw.sin(),
            self.radius * self.pitch.sin(),
            horizontal * self.yaw.cos(),
        );
        Transform::from_translation(self.target + offset).looking_at(self.target, Vec3::Y)
    }

    pub(crate) fn frame_bounds(&mut self, minimum: Vec3, maximum: Vec3) {
        self.target = (minimum + maximum) * 0.5;
        let half_extent = (maximum - minimum) * 0.5;
        self.radius = (half_extent.length() * 1.45).clamp(MIN_RADIUS, MAX_RADIUS);
    }
}

pub(crate) fn update_orbit_camera(
    mut camera: Single<(&mut OrbitCamera, &mut Transform)>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    hotbar_capture: Res<HotbarPointerCapture>,
) {
    let (orbit, transform) = &mut *camera;
    let scroll_scale = match scroll.unit {
        MouseScrollUnit::Line => 1.0,
        MouseScrollUnit::Pixel => 0.02,
    };
    orbit.apply_input(
        motion.delta,
        if hotbar_capture.active() {
            0.0
        } else {
            scroll.delta.y * scroll_scale
        },
        !hotbar_capture.active() && orbit_input_active(&mouse_buttons, &keyboard),
        !hotbar_capture.active() && pan_input_active(&mouse_buttons, &keyboard),
    );
    **transform = orbit.transform();
}

pub(crate) fn orbit_input_active(
    mouse_buttons: &ButtonInput<MouseButton>,
    keyboard: &ButtonInput<KeyCode>,
) -> bool {
    mouse_buttons.pressed(MouseButton::Middle)
        || (mouse_buttons.pressed(MouseButton::Left)
            && keyboard.any_pressed([KeyCode::AltLeft, KeyCode::AltRight])
            && !keyboard.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]))
}

pub(crate) fn pan_input_active(
    mouse_buttons: &ButtonInput<MouseButton>,
    keyboard: &ButtonInput<KeyCode>,
) -> bool {
    mouse_buttons.pressed(MouseButton::Left)
        && keyboard.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight])
}

pub(crate) fn camera_input_active(
    mouse_buttons: &ButtonInput<MouseButton>,
    keyboard: &ButtonInput<KeyCode>,
) -> bool {
    orbit_input_active(mouse_buttons, keyboard) || pan_input_active(mouse_buttons, keyboard)
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_PITCH, MAX_RADIUS, MIN_PITCH, MIN_RADIUS, OrbitCamera, camera_input_active,
        pan_input_active,
    };
    use bevy::prelude::{ButtonInput, KeyCode, MouseButton, Vec2, Vec3};

    use super::orbit_input_active;

    #[test]
    fn option_left_drag_activates_orbit_without_removing_middle_drag() {
        let mut mouse = ButtonInput::default();
        let mut keyboard = ButtonInput::default();
        mouse.press(MouseButton::Left);
        assert!(!orbit_input_active(&mouse, &keyboard));

        keyboard.press(KeyCode::AltLeft);
        assert!(orbit_input_active(&mouse, &keyboard));

        mouse.release(MouseButton::Left);
        mouse.press(MouseButton::Middle);
        assert!(orbit_input_active(&mouse, &keyboard));
    }

    #[test]
    fn orbit_pitch_and_zoom_stay_in_bounds() {
        let mut camera = OrbitCamera::default();
        camera.apply_input(Vec2::new(0.0, 10_000.0), 10_000.0, true, false);
        assert!((camera.pitch - MAX_PITCH).abs() < f32::EPSILON);
        assert!((camera.radius - MIN_RADIUS).abs() < f32::EPSILON);

        camera.apply_input(Vec2::new(0.0, -10_000.0), -10_000.0, true, false);
        assert!((camera.pitch - MIN_PITCH).abs() < f32::EPSILON);
        assert!((camera.radius - MAX_RADIUS).abs() < f32::EPSILON);
    }

    #[test]
    fn downward_drag_raises_orbit_pitch() {
        let mut camera = OrbitCamera::default();
        let before = camera.pitch;
        camera.apply_input(Vec2::new(0.0, 10.0), 0.0, true, false);
        assert!(camera.pitch > before);
    }

    #[test]
    fn orbit_transform_keeps_looking_at_target() {
        let camera = OrbitCamera::default();
        let transform = camera.transform();
        let toward_target = (camera.target - transform.translation).normalize();
        assert!(
            transform
                .forward()
                .as_vec3()
                .abs_diff_eq(toward_target, 1.0e-5)
        );
        assert!(transform.translation.y > camera.target.y);
        assert_ne!(transform.translation, Vec3::ZERO);
    }

    #[test]
    fn framing_centres_the_camera_and_respects_zoom_limits() {
        let mut camera = OrbitCamera::default();
        camera.frame_bounds(Vec3::new(-10.0, 0.0, -4.0), Vec3::new(10.0, 8.0, 4.0));

        assert_eq!(camera.target, Vec3::new(0.0, 4.0, 0.0));
        assert!((MIN_RADIUS..=MAX_RADIUS).contains(&camera.radius));
    }

    #[test]
    fn shift_left_drag_pans_only_across_the_ground_plane() {
        let mut mouse = ButtonInput::default();
        let mut keyboard = ButtonInput::default();
        mouse.press(MouseButton::Left);
        keyboard.press(KeyCode::ShiftLeft);
        assert!(pan_input_active(&mouse, &keyboard));
        assert!(camera_input_active(&mouse, &keyboard));
        assert!(!orbit_input_active(&mouse, &keyboard));

        let mut camera = OrbitCamera::default();
        let before = camera.target;
        camera.apply_input(Vec2::new(40.0, -20.0), 0.0, false, true);
        assert_ne!(camera.target, before);
        assert!((camera.target.y - before.y).abs() < f32::EPSILON);
        assert!((camera.yaw - 0.75).abs() < f32::EPSILON);
        assert!((camera.pitch - 0.55).abs() < f32::EPSILON);
    }
}
