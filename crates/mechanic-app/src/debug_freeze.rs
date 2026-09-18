//! A debug key that freezes every frame update so a single frame can be inspected.

use bevy::prelude::{
    ButtonInput, KeyCode, MouseButton, Res, ResMut, Resource, Single, Time, Virtual, With,
};
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

pub(crate) const DEBUG_FRAME_FREEZE_KEY: KeyCode = KeyCode::F8;

#[derive(Resource, Debug, Default, PartialEq, Eq)]
pub(crate) struct DebugFrameFreeze {
    pub(crate) active: bool,
    pub(crate) resume_pending: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DebugFrameFreezeEffect {
    None,
    PauseTime,
    ResumeTime,
}

impl DebugFrameFreeze {
    pub(crate) fn advance(
        &mut self,
        freeze_pressed: bool,
        primary_pressed: bool,
    ) -> DebugFrameFreezeEffect {
        if self.resume_pending {
            self.resume_pending = false;
            return DebugFrameFreezeEffect::ResumeTime;
        }
        if self.active && primary_pressed {
            self.active = false;
            self.resume_pending = true;
            return DebugFrameFreezeEffect::None;
        }
        if !self.active && freeze_pressed {
            self.active = true;
            return DebugFrameFreezeEffect::PauseTime;
        }
        DebugFrameFreezeEffect::None
    }

    pub(crate) const fn blocks_updates(&self) -> bool {
        self.active || self.resume_pending
    }
}

pub(crate) fn update_debug_frame_freeze(
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut freeze: ResMut<DebugFrameFreeze>,
    mut virtual_time: ResMut<Time<Virtual>>,
    mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>,
) {
    match freeze.advance(
        keyboard.just_pressed(DEBUG_FRAME_FREEZE_KEY),
        mouse.just_pressed(MouseButton::Left),
    ) {
        DebugFrameFreezeEffect::PauseTime => virtual_time.pause(),
        DebugFrameFreezeEffect::ResumeTime => virtual_time.unpause(),
        DebugFrameFreezeEffect::None => {}
    }
    if freeze.blocks_updates() {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
    }
}

pub(crate) fn debug_frame_updates_enabled(freeze: Res<DebugFrameFreeze>) -> bool {
    !freeze.blocks_updates()
}

#[cfg(test)]
mod tests {
    use crate::debug_freeze::{DebugFrameFreeze, DebugFrameFreezeEffect};

    #[test]
    fn freeze_stays_active_until_a_click_is_consumed() {
        let mut freeze = DebugFrameFreeze::default();
        assert_eq!(
            freeze.advance(true, false),
            DebugFrameFreezeEffect::PauseTime
        );
        assert!(freeze.blocks_updates());

        assert_eq!(freeze.advance(false, true), DebugFrameFreezeEffect::None);
        assert!(freeze.blocks_updates());

        assert_eq!(
            freeze.advance(false, false),
            DebugFrameFreezeEffect::ResumeTime
        );
        assert!(!freeze.blocks_updates());
    }
}
