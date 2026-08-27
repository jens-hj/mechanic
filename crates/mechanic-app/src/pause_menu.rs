//! Pause-menu state and the requests produced by its Mosaic view.

use bevy::prelude::Resource;

use crate::controls::{GameAction, InputChord};
use crate::ui::PauseAction;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BindingCapture {
    pub(crate) action: GameAction,
    pub(crate) slot: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PauseRequest {
    Continue,
    OpenOptions,
    OpenControls,
    Back,
    SetCameraFov(f32),
    BeginBindingCapture(BindingCapture),
    ClearBinding(GameAction, usize),
    ResetControls,
    ReturnToBuild,
    Exit,
    CancelExit,
    ExitWithoutSaving,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum PausePage {
    #[default]
    Main,
    Options,
    Controls,
    ExitConfirmation,
}

/// Modal state plus a one-frame barrier that keeps closing input out of the world.
#[derive(Resource, Debug, Default)]
pub(crate) struct PauseMenuState {
    open: bool,
    page: PausePage,
    blocks_for_frame: bool,
    requested: Option<PauseRequest>,
    capturing: Option<BindingCapture>,
}

impl PauseMenuState {
    pub(crate) const fn is_open(&self) -> bool {
        self.open
    }

    pub(crate) const fn page(&self) -> PausePage {
        self.page
    }

    pub(crate) const fn is_in_submenu(&self) -> bool {
        !matches!(self.page, PausePage::Main)
    }

    pub(crate) const fn blocks_world_input(&self) -> bool {
        self.open || self.blocks_for_frame
    }

    pub(crate) fn begin_frame(&mut self) {
        self.blocks_for_frame = false;
    }

    pub(crate) fn consume_frame(&mut self) {
        self.blocks_for_frame = true;
    }

    pub(crate) fn open(&mut self) {
        self.open = true;
        self.page = PausePage::Main;
        self.consume_frame();
    }

    pub(crate) fn close(&mut self) {
        self.open = false;
        self.page = PausePage::Main;
        self.capturing = None;
        self.consume_frame();
    }

    pub(crate) fn return_to_main(&mut self) {
        self.page = PausePage::Main;
        self.capturing = None;
        self.consume_frame();
    }

    pub(crate) fn open_options(&mut self) {
        self.page = PausePage::Options;
        self.consume_frame();
    }

    pub(crate) fn open_controls(&mut self) {
        self.page = PausePage::Controls;
        self.capturing = None;
        self.consume_frame();
    }

    pub(crate) const fn binding_capture(&self) -> Option<BindingCapture> {
        self.capturing
    }

    pub(crate) fn cancel_binding_capture(&mut self) {
        self.capturing = None;
        self.consume_frame();
    }

    pub(crate) fn finish_binding_capture(&mut self, _chord: InputChord) {
        self.capturing = None;
        self.consume_frame();
    }

    pub(crate) fn confirm_exit(&mut self) {
        self.page = PausePage::ExitConfirmation;
        self.consume_frame();
    }

    pub(crate) fn act(&mut self, action: PauseAction) {
        self.requested = Some(match action {
            PauseAction::Continue => PauseRequest::Continue,
            PauseAction::OpenOptions => PauseRequest::OpenOptions,
            PauseAction::OpenControls => PauseRequest::OpenControls,
            PauseAction::Back => PauseRequest::Back,
            PauseAction::SetCameraFov(value) => PauseRequest::SetCameraFov(value),
            PauseAction::BeginBindingCapture(action, slot) => {
                self.capturing = Some(BindingCapture { action, slot });
                PauseRequest::BeginBindingCapture(BindingCapture { action, slot })
            }
            PauseAction::ClearBinding(action, slot) => PauseRequest::ClearBinding(action, slot),
            PauseAction::ResetControls => PauseRequest::ResetControls,
            PauseAction::ReturnToBuild => PauseRequest::ReturnToBuild,
            PauseAction::Exit => PauseRequest::Exit,
            PauseAction::CancelExit => PauseRequest::CancelExit,
            PauseAction::ExitWithoutSaving => PauseRequest::ExitWithoutSaving,
        });
    }

    pub(crate) fn take_request(&mut self) -> Option<PauseRequest> {
        self.requested.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closing_keeps_input_blocked_for_the_rest_of_the_frame() {
        let mut menu = PauseMenuState::default();
        menu.open();
        menu.begin_frame();
        menu.close();
        assert!(!menu.is_open());
        assert!(menu.blocks_world_input());
        menu.begin_frame();
        assert!(!menu.blocks_world_input());
    }

    #[test]
    fn confirmation_cancels_before_the_menu_closes() {
        let mut menu = PauseMenuState::default();
        menu.open();
        menu.confirm_exit();
        menu.return_to_main();
        assert!(menu.is_open());
        assert_eq!(menu.page(), PausePage::Main);
    }

    #[test]
    fn binding_capture_can_cancel_and_clear_either_slot() {
        let mut menu = PauseMenuState::default();
        menu.open();
        menu.open_controls();

        menu.act(PauseAction::BeginBindingCapture(GameAction::Rotate, 1));
        let capture = BindingCapture {
            action: GameAction::Rotate,
            slot: 1,
        };
        assert_eq!(menu.binding_capture(), Some(capture));
        assert_eq!(
            menu.take_request(),
            Some(PauseRequest::BeginBindingCapture(capture))
        );

        menu.cancel_binding_capture();
        assert_eq!(menu.binding_capture(), None);

        menu.act(PauseAction::ClearBinding(GameAction::Rotate, 1));
        assert_eq!(
            menu.take_request(),
            Some(PauseRequest::ClearBinding(GameAction::Rotate, 1))
        );
    }
}
