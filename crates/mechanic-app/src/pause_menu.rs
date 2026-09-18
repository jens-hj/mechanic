//! Pause-menu state and the requests produced by its Mosaic view.

use crate::camera::MaterialWheelState;
use crate::control_panel::ControlPanelState;
use crate::controls::BindingInput;
use crate::controls::Modifiers;
use crate::controls::WheelDirection;
use crate::creation_menu::CreationMenuState;
use crate::editor::history::EditorHistory;
use crate::editor::hover::clear_hover;
use crate::editor::shape_actions::shape_tool_is_busy;
use crate::editor::state::EditorGraph;
use crate::editor::state::EditorState;
use crate::hotbar::SelectedTool;
use crate::settings::AppSettings;
use crate::{ui, world};
use bevy::app::AppExit;
use bevy::input::mouse::AccumulatedMouseScroll;
use bevy::prelude::ButtonInput;
use bevy::prelude::KeyCode;
use bevy::prelude::MessageWriter;
use bevy::prelude::MouseButton;
use bevy::prelude::Res;
use bevy::prelude::ResMut;
use bevy::prelude::Resource;
use bevy::prelude::State;
use bevy::prelude::warn;
use mechanic_core::BuildCommand;
use mechanic_core::ConstructionGraph;

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
    ExitToWorldSelector,
    OpenOptions,
    OpenControls,
    Back,
    SetCameraFov(f32),
    BeginBindingCapture(BindingCapture),
    ClearBinding(GameAction, usize),
    ResetControls,
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
            PauseAction::ExitToWorldSelector => PauseRequest::ExitToWorldSelector,
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
            PauseAction::Exit => PauseRequest::Exit,
            PauseAction::CancelExit => PauseRequest::CancelExit,
            PauseAction::ExitWithoutSaving => PauseRequest::ExitWithoutSaving,
        });
    }

    pub(crate) fn take_request(&mut self) -> Option<PauseRequest> {
        self.requested.take()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EscapeTarget {
    PauseSubmenu,
    PauseMenu,
    ExistingUi,
    ControlPanel,
    WorldState,
    OpenPause,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExitDisposition {
    Exit,
    ConfirmUnsaved,
}

pub(crate) const fn exit_disposition(construction_dirty: bool) -> ExitDisposition {
    if construction_dirty {
        ExitDisposition::ConfirmUnsaved
    } else {
        ExitDisposition::Exit
    }
}

#[expect(clippy::fn_params_excessive_bools)]
// The booleans are independent, ordered input owners; the return value is their priority.
pub(crate) const fn escape_target(
    pause_open: bool,
    submenu_open: bool,
    existing_ui: bool,
    panel_open: bool,
    world_state: bool,
) -> EscapeTarget {
    if pause_open && submenu_open {
        EscapeTarget::PauseSubmenu
    } else if pause_open {
        EscapeTarget::PauseMenu
    } else if existing_ui {
        EscapeTarget::ExistingUi
    } else if panel_open {
        EscapeTarget::ControlPanel
    } else if world_state {
        EscapeTarget::WorldState
    } else {
        EscapeTarget::OpenPause
    }
}

pub(crate) fn begin_pause_frame(mut pause: ResMut<PauseMenuState>) {
    pause.begin_frame();
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn handle_pause_escape(
    keyboard: Res<ButtonInput<KeyCode>>,
    overlay: Res<ui::UiInput>,
    menu: Res<CreationMenuState>,
    mut panel: ResMut<ControlPanelState>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    selection: Res<SelectedTool>,
    mut wheel: ResMut<MaterialWheelState>,
    mut pause: ResMut<PauseMenuState>,
    worlds: Res<world::WorldListState>,
) {
    if worlds.is_open() || !keyboard.just_pressed(KeyCode::Escape) {
        return;
    }
    if state.suspension.controls.selected.is_some() || state.suspension.drag.is_some() {
        state.suspension.controls.dismiss();
        state.suspension.drag = None;
        state.suspension.preview = None;
        pause.consume_frame();
        return;
    }
    if pause.binding_capture().is_some() {
        pause.cancel_binding_capture();
        return;
    }
    let world_state = state.block_drag.is_some()
        || state.pipe_drag.is_some()
        || state.delete_drag.is_some()
        || graph.0.pending().is_some()
        || state.weld.busy()
        || state.wire_drag.is_some()
        || shape_tool_is_busy(selection.active_editor_tool(), &state);
    let target = escape_target(
        pause.is_open(),
        pause.is_in_submenu(),
        menu.is_open() || overlay.escape_is_consumed(),
        panel.is_open(),
        world_state,
    );
    pause.consume_frame();
    match target {
        EscapeTarget::PauseSubmenu => pause.return_to_main(),
        EscapeTarget::PauseMenu => pause.close(),
        EscapeTarget::ExistingUi => {}
        EscapeTarget::ControlPanel => {
            panel.close();
            state.feedback = Some("Control block panel closed".to_owned());
        }
        EscapeTarget::WorldState => cancel_one_world_escape_owner(&mut graph.0, &mut state),
        EscapeTarget::OpenPause => {
            wheel.close();
            pause.open();
        }
    }
}

pub(crate) fn cancel_one_world_escape_owner(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
) {
    if state.weld.busy() {
        state.weld.cancel();
        state.feedback = Some("Weld cancelled".to_owned());
        return;
    }
    if state.block_drag.take().is_some() {
        clear_hover(state);
        state.feedback = Some("Block drag cancelled".to_owned());
    } else if state.pipe_drag.take().is_some() {
        clear_hover(state);
        state.feedback = Some("Pipe run cancelled".to_owned());
    } else if state.layer_drag.take().is_some() {
        clear_hover(state);
        state.feedback = Some("Layer cancelled".to_owned());
    } else if state.delete_drag.take().is_some() {
        clear_hover(state);
        state.feedback = Some("Delete drag cancelled".to_owned());
    } else if graph.pending().is_some() {
        let _ = graph.apply(BuildCommand::CancelPending);
        state.feedback = Some("Selection cancelled".to_owned());
    } else if state.wire_drag.take().is_some() {
        state.feedback = Some("Wire drag cancelled".to_owned());
    } else if state.region_drag.take().is_some() {
        state.feedback = Some("Area selection cancelled".to_owned());
    } else if state.vertex_drag.take().is_some() {
        state.construction_mesh_dirty = true;
        state.feedback = Some("Shape drag cancelled".to_owned());
    } else if state.paint_selecting || !state.selected_vertices.is_empty() {
        state.paint_selecting = false;
        state.selected_vertices.clear();
        state.feedback = Some("Selection cleared".to_owned());
    } else if state.active_region.take().is_some() {
        state.construction_mesh_dirty = true;
        state.feedback = Some("Left the region".to_owned());
    }
}

pub(crate) fn handle_pause_request(
    mut pause: ResMut<PauseMenuState>,
    mut settings: ResMut<AppSettings>,
    history: Res<EditorHistory>,
    space: Res<State<world::AppSpace>>,
    mut worlds: ResMut<world::WorldListState>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(request) = pause.take_request() else {
        return;
    };
    match request {
        PauseRequest::Continue => pause.close(),
        PauseRequest::ExitToWorldSelector => {
            worlds.act(ui::WorldAction::ExitToSelector);
            pause.close();
        }
        PauseRequest::OpenOptions => pause.open_options(),
        PauseRequest::OpenControls => pause.open_controls(),
        PauseRequest::Back | PauseRequest::CancelExit => pause.return_to_main(),
        PauseRequest::SetCameraFov(degrees) => {
            if let Err(error) = settings.set_camera_fov_degrees(degrees) {
                warn!("could not save settings: {error}");
            }
        }
        PauseRequest::BeginBindingCapture(_) => {}
        PauseRequest::ClearBinding(action, slot) => {
            if let Err(error) = settings.set_binding(action, slot, None) {
                warn!("could not save settings: {error}");
            }
        }
        PauseRequest::ResetControls => {
            if let Err(error) = settings.reset_controls() {
                warn!("could not save settings: {error}");
            }
        }
        PauseRequest::Exit => {
            match exit_disposition(*space.get() == world::AppSpace::Garage && history.is_dirty()) {
                ExitDisposition::Exit => {
                    exit.write(AppExit::Success);
                }
                ExitDisposition::ConfirmUnsaved => pause.confirm_exit(),
            }
        }
        PauseRequest::ExitWithoutSaving => {
            exit.write(AppExit::Success);
        }
    }
}

pub(crate) fn capture_control_binding(
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    scroll: Res<AccumulatedMouseScroll>,
    mut pause: ResMut<PauseMenuState>,
    mut settings: ResMut<AppSettings>,
) {
    let Some(capture) = pause.binding_capture() else {
        return;
    };
    if keyboard.just_pressed(KeyCode::Escape) {
        return;
    }
    let modifiers = Modifiers::from_keyboard(&keyboard);
    let modifier_key = |key| {
        matches!(
            key,
            KeyCode::ShiftLeft
                | KeyCode::ShiftRight
                | KeyCode::ControlLeft
                | KeyCode::ControlRight
                | KeyCode::AltLeft
                | KeyCode::AltRight
                | KeyCode::SuperLeft
                | KeyCode::SuperRight
        )
    };
    let input = keyboard
        .get_just_pressed()
        .copied()
        .find(|key| !modifier_key(*key))
        .map(BindingInput::Key)
        .or_else(|| {
            mouse
                .get_just_pressed()
                .next()
                .copied()
                .map(BindingInput::Mouse)
        })
        .or_else(|| {
            if scroll.delta.y > 0.0 {
                Some(BindingInput::Wheel(WheelDirection::Up))
            } else if scroll.delta.y < 0.0 {
                Some(BindingInput::Wheel(WheelDirection::Down))
            } else if scroll.delta.x < 0.0 {
                Some(BindingInput::Wheel(WheelDirection::Left))
            } else if scroll.delta.x > 0.0 {
                Some(BindingInput::Wheel(WheelDirection::Right))
            } else {
                None
            }
        });
    let Some(input) = input else {
        return;
    };
    if matches!(input, BindingInput::Wheel(_)) && !capture.action.instantaneous() {
        return;
    }
    let chord = InputChord { input, modifiers };
    if let Err(error) = settings.set_binding(capture.action, capture.slot, Some(chord)) {
        warn!("could not save settings: {error}");
    }
    pause.finish_binding_capture(chord);
}

#[cfg(test)]
mod tests {
    mod escape;

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
