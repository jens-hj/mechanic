//! Translating Bevy's input messages into Mosaic's input events.
//!
//! The mapping is deliberately a near-port of `mosaic-runtime`'s winit
//! translation rather than a fresh one: the behaviors that matter here are not
//! in the type correspondence but in the timing. Moves are coalesced, a line of
//! scroll is a fixed number of logical pixels, and modifiers ride on every
//! event because Mosaic has no modifiers-changed event of its own.

use bevy::ecs::system::SystemParam;
use bevy::input::ButtonState;
use bevy::input::keyboard::{Key as BevyKey, KeyCode, KeyboardInput};
use bevy::input::mouse::{MouseButton, MouseButtonInput, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::window::{CursorLeft, CursorMoved, Ime};
use mosaic_core::Vector2;
use mosaic_widgets::Ui;
use mosaic_widgets::input::{
    ImeEvent, Key, KeyEvent, KeyEventKind, Modifiers, PointerButton, PointerEvent,
    PointerEventKind, PointerType,
};

use core::time::Duration;

/// One line of wheel scroll, in logical pixels.
///
/// Bevy reports line-based wheel deltas in lines, and the only thing that can
/// turn those into a distance is a convention. This is Mosaic's, so a trackpad
/// (which reports pixels) and a wheel (which reports lines) agree about how far
/// a gesture scrolls.
const LINE_HEIGHT_PX: f32 = 32.0;

/// Pointer and keyboard state carried between Bevy messages.
///
/// Mosaic's events are self-contained — each one names its position and its
/// modifiers — so somebody has to remember what Bevy reports only as changes.
/// That is this.
#[derive(Debug, Default)]
pub(crate) struct InputState {
    /// The last position the pointer was seen at, in logical pixels. `None`
    /// once the pointer leaves the window.
    position: Option<Vector2>,
    /// A move that has arrived but not yet been dispatched. See
    /// [`flush_moves`](Self::flush_moves).
    pending_move: Option<Vector2>,
    modifiers: Modifiers,
    /// Monotonic time since the app started, which is what Mosaic derives
    /// double- and triple-click from.
    elapsed: Duration,
    /// Whether the last key Mosaic saw was consumed by the tree.
    keyboard_consumed: bool,
}

impl InputState {
    /// The pointer's last known position, in logical pixels.
    pub(crate) fn position(&self) -> Option<Vector2> {
        self.position
    }

    /// Whether the tree took the last key event it was offered.
    pub(crate) fn keyboard_consumed(&self) -> bool {
        self.keyboard_consumed
    }

    /// Dispatch the coalesced move, if there is one.
    ///
    /// Every dispatched move costs a full hit test, and in a scene with
    /// pointer-lit surfaces it invalidates their shape caches — so a burst of
    /// moves inside one frame becomes one dispatch. Anything that reads the
    /// hover chain (a press, a wheel) has to flush first, or it resolves
    /// against a stale position.
    pub(crate) fn flush_moves(&mut self, ui: &Ui) {
        let Some(position) = self.pending_move.take() else {
            return;
        };
        self.dispatch_pointer(ui, PointerEventKind::Move, position);
    }

    /// Record a move without dispatching it. See [`flush_moves`](Self::flush_moves).
    pub(crate) fn note_move(&mut self, position: Vector2) {
        self.position = Some(position);
        self.pending_move = Some(position);
    }

    /// The pointer left the window: nothing pending, nothing hovered.
    pub(crate) fn note_gone(&mut self, ui: &Ui) {
        self.pending_move = None;
        self.position = None;
        ui.pointer_gone();
    }

    /// Dispatch a button transition at the pointer's last known position,
    /// flushing the move that put it there first.
    pub(crate) fn note_button(&mut self, ui: &Ui, button: PointerButton, pressed: bool) {
        let Some(position) = self.position else {
            return;
        };
        self.flush_moves(ui);
        let kind = if pressed {
            PointerEventKind::Down(button)
        } else {
            PointerEventKind::Up(button)
        };
        self.dispatch_pointer(ui, kind, position);
    }

    /// Dispatch a wheel delta at the pointer's last known position.
    pub(crate) fn note_wheel(&mut self, ui: &Ui, delta: Vector2) {
        let Some(position) = self.position else {
            return;
        };
        self.flush_moves(ui);
        self.dispatch_pointer(ui, PointerEventKind::Wheel { delta }, position);
    }

    fn dispatch_pointer(&self, ui: &Ui, kind: PointerEventKind, position: Vector2) {
        ui.dispatch_pointer(PointerEvent {
            kind,
            position,
            pointer_type: PointerType::Mouse,
            modifiers: self.modifiers,
            timestamp: self.elapsed,
        });
    }
}

/// Everything the translation reads out of Bevy in one frame.
///
/// A `SystemParam` rather than a long argument list, so the plugin's system
/// signature stays readable and the readers can be reused by tests.
#[derive(SystemParam)]
pub(crate) struct BevyInput<'w, 's> {
    pub(crate) cursor_moved: MessageReader<'w, 's, CursorMoved>,
    pub(crate) cursor_left: MessageReader<'w, 's, CursorLeft>,
    pub(crate) mouse_button: MessageReader<'w, 's, MouseButtonInput>,
    pub(crate) mouse_wheel: MessageReader<'w, 's, MouseWheel>,
    pub(crate) keyboard: MessageReader<'w, 's, KeyboardInput>,
    pub(crate) ime: MessageReader<'w, 's, Ime>,
    pub(crate) keys: Res<'w, ButtonInput<KeyCode>>,
}

impl InputState {
    /// Feed one frame of Bevy input into the tree.
    ///
    /// `window` filters the messages: every Bevy input message names the window
    /// it landed in, and a context drives exactly one of them.
    pub(crate) fn drive(
        &mut self,
        ui: &Ui,
        window: Entity,
        elapsed: Duration,
        input: &mut BevyInput,
    ) {
        self.elapsed = elapsed;
        self.modifiers = modifiers_from(&input.keys);

        for moved in input.cursor_moved.read() {
            if moved.window != window {
                continue;
            }
            // Entering the window is a position becoming known, which the tree
            // has to see as a move before it can see anything else.
            self.note_move(Vector2::new(moved.position.x, moved.position.y));
        }

        for left in input.cursor_left.read() {
            if left.window != window {
                continue;
            }
            self.note_gone(ui);
        }

        for button in input.mouse_button.read() {
            if button.window != window {
                continue;
            }
            let Some(pointer_button) = pointer_button_from(button.button) else {
                continue;
            };
            // A press resolves against the hover chain, so the move that put
            // the pointer where it is has to land first.
            self.note_button(ui, pointer_button, button.state == ButtonState::Pressed);
        }

        for wheel in input.mouse_wheel.read() {
            if wheel.window != window {
                continue;
            }
            self.note_wheel(ui, wheel_delta(wheel));
        }

        for key in input.keyboard.read() {
            if key.window != window {
                continue;
            }
            let Some(mapped) = key_from(key) else {
                // Mosaic has no vocabulary for this key; dispatching a
                // placeholder would only teach the tree to swallow it.
                continue;
            };
            let kind = match key.state {
                ButtonState::Pressed => KeyEventKind::Down { repeat: key.repeat },
                ButtonState::Released => KeyEventKind::Up,
            };
            let consumed = ui.dispatch_key(KeyEvent {
                kind,
                key: mapped,
                modifiers: self.modifiers,
            });
            if matches!(key.state, ButtonState::Pressed) {
                self.keyboard_consumed = consumed;
            }
        }

        for ime in input.ime.read() {
            match ime {
                Ime::Preedit {
                    window: w,
                    value,
                    cursor,
                    ..
                } if *w == window => ui.dispatch_ime(ImeEvent::Preedit {
                    text: value.clone(),
                    cursor: *cursor,
                }),
                Ime::Commit { window: w, value } if *w == window => {
                    ui.dispatch_ime(ImeEvent::Commit(value.clone()));
                }
                // Enabled/Disabled are the platform acknowledging the request
                // this crate already made through the window; the tree does not
                // need to hear about them.
                _ => {}
            }
        }
    }
}

/// Mosaic carries modifiers on every event, so they are read fresh each frame
/// from Bevy's held-key state rather than tracked through press/release pairs.
fn modifiers_from(keys: &ButtonInput<KeyCode>) -> Modifiers {
    Modifiers {
        shift: keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]),
        ctrl: keys.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]),
        alt: keys.any_pressed([KeyCode::AltLeft, KeyCode::AltRight]),
        meta: keys.any_pressed([KeyCode::SuperLeft, KeyCode::SuperRight]),
    }
}

/// Mosaic routes three buttons; the rest have no meaning to it.
fn pointer_button_from(button: MouseButton) -> Option<PointerButton> {
    match button {
        MouseButton::Left => Some(PointerButton::Primary),
        MouseButton::Right => Some(PointerButton::Secondary),
        MouseButton::Middle => Some(PointerButton::Middle),
        _ => None,
    }
}

/// Wheel deltas in logical pixels, whichever unit the device reports in.
///
/// Bevy and Mosaic both describe the direction the content should move. The
/// scroll widget converts that movement into its own opposite-growing offset.
fn wheel_delta(wheel: &MouseWheel) -> Vector2 {
    let scale = match wheel.unit {
        MouseScrollUnit::Line => LINE_HEIGHT_PX,
        MouseScrollUnit::Pixel => 1.0,
    };
    Vector2::new(wheel.x * scale, wheel.y * scale)
}

/// Bevy's logical key as one of Mosaic's, or `None` for a key Mosaic has no
/// name for. An unmapped key is never dispatched: the tree would have no way to
/// tell it apart from any other, and reporting it as consumed would swallow a
/// shortcut the app wanted.
fn key_from(input: &KeyboardInput) -> Option<Key> {
    if let Some(number) = function_key(&input.logical_key) {
        return Some(Key::Function(number));
    }
    match &input.logical_key {
        BevyKey::Character(text) => Some(Key::Character(text.to_string())),
        BevyKey::Enter => Some(Key::Enter),
        BevyKey::Escape => Some(Key::Escape),
        BevyKey::Backspace => Some(Key::Backspace),
        BevyKey::Delete => Some(Key::Delete),
        BevyKey::Tab => Some(Key::Tab),
        BevyKey::Space => Some(Key::Space),
        BevyKey::ArrowUp => Some(Key::ArrowUp),
        BevyKey::ArrowDown => Some(Key::ArrowDown),
        BevyKey::ArrowLeft => Some(Key::ArrowLeft),
        BevyKey::ArrowRight => Some(Key::ArrowRight),
        BevyKey::Home => Some(Key::Home),
        BevyKey::End => Some(Key::End),
        _ => None,
    }
}

/// Bevy spells the function keys as distinct variants and Mosaic numbers them;
/// this is the whole of the correspondence, written once.
macro_rules! function_keys {
    ($($variant:ident => $number:literal),* $(,)?) => {
        fn function_key(key: &BevyKey) -> Option<u8> {
            match key {
                $(BevyKey::$variant => Some($number),)*
                _ => None,
            }
        }
    };
}

function_keys! {
    F1 => 1, F2 => 2, F3 => 3, F4 => 4, F5 => 5, F6 => 6,
    F7 => 7, F8 => 8, F9 => 9, F10 => 10, F11 => 11, F12 => 12,
    F13 => 13, F14 => 14, F15 => 15, F16 => 16, F17 => 17, F18 => 18,
    F19 => 19, F20 => 20, F21 => 21, F22 => 22, F23 => 23, F24 => 24,
}

#[cfg(test)]
// The wheel deltas compared here are products of small integers and a power of
// two, which f32 represents exactly; an epsilon would only hide a wrong scale.
#[allow(clippy::float_cmp)]
mod tests {
    use super::{InputState, LINE_HEIGHT_PX, key_from, pointer_button_from, wheel_delta};
    use bevy::input::ButtonState;
    use bevy::input::keyboard::{Key as BevyKey, KeyCode, KeyboardInput};
    use bevy::input::mouse::{MouseButton, MouseScrollUnit, MouseWheel};
    use bevy::prelude::Entity;
    use bevy::window::SystemCursorIcon;
    use core::cell::Cell;
    use mosaic_core::{Size, Vector2};
    use mosaic_layout::{Dimension, Style};
    use mosaic_text::FontContext;
    use mosaic_widgets::Ui;
    use mosaic_widgets::input::{Key, PointerButton, PointerEventKind};
    use std::rc::Rc;

    /// A tree filling the viewport, counting the pointer events that reach it.
    ///
    /// `embedded_only` fonts keep it hermetic: no system font scan, and the
    /// same metrics on every machine.
    fn counting_tree() -> (Ui, Rc<Cell<usize>>, Rc<Cell<usize>>) {
        let ui = Ui::new();
        ui.set_fonts(FontContext::embedded_only());
        let root = ui.root();
        root.style(Style::column().size(Dimension::Fill, Dimension::Fill));

        let moves = Rc::new(Cell::new(0));
        let downs = Rc::new(Cell::new(0));
        {
            let moves = moves.clone();
            let downs = downs.clone();
            root.on_pointer(move |event, _| match event.kind {
                PointerEventKind::Move => moves.set(moves.get() + 1),
                PointerEventKind::Down(_) => downs.set(downs.get() + 1),
                _ => {}
            });
        }
        ui.frame(Size::new(200.0, 200.0), 1.0);
        (ui, moves, downs)
    }

    #[test]
    fn a_burst_of_moves_in_one_frame_dispatches_once_at_the_latest_position() {
        let (ui, moves, _) = counting_tree();
        let mut state = InputState::default();

        state.note_move(Vector2::new(10.0, 10.0));
        state.note_move(Vector2::new(20.0, 20.0));
        state.note_move(Vector2::new(30.0, 30.0));
        assert_eq!(moves.get(), 0, "a noted move must not dispatch on its own");

        state.flush_moves(&ui);
        assert_eq!(
            moves.get(),
            1,
            "three moves in one frame are one hit test, not three",
        );
        assert_eq!(
            state.position(),
            Some(Vector2::new(30.0, 30.0)),
            "the pointer is wherever it was last seen, not where it was first seen",
        );
    }

    #[test]
    fn flushing_with_nothing_pending_dispatches_nothing() {
        let (ui, moves, _) = counting_tree();
        let mut state = InputState::default();

        state.flush_moves(&ui);
        state.note_move(Vector2::new(10.0, 10.0));
        state.flush_moves(&ui);
        state.flush_moves(&ui);

        assert_eq!(
            moves.get(),
            1,
            "an idle frame re-dispatches nothing; only a new move costs a hit test",
        );
    }

    #[test]
    fn a_button_press_flushes_the_move_that_put_the_pointer_there() {
        let (ui, moves, downs) = counting_tree();
        let mut state = InputState::default();

        state.note_move(Vector2::new(40.0, 40.0));
        state.note_button(&ui, PointerButton::Primary, true);

        assert_eq!(
            moves.get(),
            1,
            "a press resolves against the hover chain, so the move lands first",
        );
        assert_eq!(downs.get(), 1, "the press itself reaches the tree");
    }

    #[test]
    fn a_press_before_the_pointer_is_ever_seen_is_dropped() {
        let (ui, _, downs) = counting_tree();
        let mut state = InputState::default();

        state.note_button(&ui, PointerButton::Primary, true);

        assert_eq!(
            downs.get(),
            0,
            "with no position there is nowhere to route the press to",
        );
    }

    #[test]
    fn the_pointer_leaving_drops_the_move_it_had_not_dispatched_yet() {
        let (ui, moves, _) = counting_tree();
        let mut state = InputState::default();

        state.note_move(Vector2::new(50.0, 50.0));
        state.note_gone(&ui);
        state.flush_moves(&ui);

        assert_eq!(
            moves.get(),
            0,
            "a move the pointer already left behind must not be delivered after it",
        );
        assert_eq!(state.position(), None);
    }

    fn wheel(unit: MouseScrollUnit, x: f32, y: f32) -> MouseWheel {
        MouseWheel {
            unit,
            x,
            y,
            window: Entity::PLACEHOLDER,
            phase: bevy::input::touch::TouchPhase::Moved,
        }
    }

    #[test]
    fn a_line_of_scroll_is_a_fixed_number_of_logical_pixels() {
        let delta = wheel_delta(&wheel(MouseScrollUnit::Line, 0.0, 1.0));
        assert_eq!(
            delta.y, LINE_HEIGHT_PX,
            "a wheel reports lines, and only a convention turns those into a distance",
        );
    }

    #[test]
    fn pixel_scroll_passes_through_at_its_own_scale() {
        let delta = wheel_delta(&wheel(MouseScrollUnit::Pixel, 3.0, 7.0));
        assert_eq!((delta.x, delta.y), (3.0, 7.0));
    }

    #[test]
    fn scrolling_the_wheel_up_moves_content_down() {
        // Mosaic subtracts this content-motion delta from its scroll offset,
        // so a positive upward wheel moves the content down toward its start.
        let delta = wheel_delta(&wheel(MouseScrollUnit::Pixel, 0.0, 1.0));
        assert!(delta.y > 0.0, "an upward wheel is positive content motion");
    }

    fn keyboard(logical_key: BevyKey) -> KeyboardInput {
        KeyboardInput {
            key_code: KeyCode::KeyA,
            logical_key,
            state: ButtonState::Pressed,
            text: None,
            repeat: false,
            window: Entity::PLACEHOLDER,
        }
    }

    #[test]
    fn typed_text_arrives_as_a_character() {
        let mapped = key_from(&keyboard(BevyKey::Character("q".into())));
        assert_eq!(mapped, Some(Key::Character("q".to_string())));
    }

    #[test]
    fn the_named_keys_a_text_field_needs_all_map() {
        for (bevy_key, expected) in [
            (BevyKey::Enter, Key::Enter),
            (BevyKey::Escape, Key::Escape),
            (BevyKey::Backspace, Key::Backspace),
            (BevyKey::Delete, Key::Delete),
            (BevyKey::Tab, Key::Tab),
            (BevyKey::Space, Key::Space),
            (BevyKey::ArrowLeft, Key::ArrowLeft),
            (BevyKey::Home, Key::Home),
            (BevyKey::End, Key::End),
            (BevyKey::F5, Key::Function(5)),
        ] {
            assert_eq!(
                key_from(&keyboard(bevy_key.clone())),
                Some(expected),
                "{bevy_key:?} has a Mosaic spelling and must reach the tree",
            );
        }
    }

    #[test]
    fn a_key_mosaic_cannot_name_is_never_dispatched() {
        // Reporting it as handled would swallow a shortcut the app wanted, and
        // there is no event the tree could tell it apart by.
        assert_eq!(key_from(&keyboard(BevyKey::Alt)), None);
        assert_eq!(key_from(&keyboard(BevyKey::F35)), None);
    }

    #[test]
    fn only_the_three_buttons_mosaic_routes_are_translated() {
        assert_eq!(
            pointer_button_from(MouseButton::Left),
            Some(PointerButton::Primary),
        );
        assert_eq!(
            pointer_button_from(MouseButton::Right),
            Some(PointerButton::Secondary),
        );
        assert_eq!(
            pointer_button_from(MouseButton::Middle),
            Some(PointerButton::Middle),
        );
        assert_eq!(pointer_button_from(MouseButton::Back), None);
    }

    #[test]
    fn every_mosaic_cursor_has_a_platform_spelling() {
        // A `_` arm in the mapping would silently turn a new Mosaic cursor into
        // the default arrow; this fails instead.
        use mosaic_widgets::CursorIcon as MosaicCursor;
        for icon in [
            MosaicCursor::Default,
            MosaicCursor::Pointer,
            MosaicCursor::ResizeHorizontal,
            MosaicCursor::ResizeVertical,
            MosaicCursor::ResizeDiagonalDown,
            MosaicCursor::ResizeDiagonalUp,
        ] {
            let mapped = crate::frame::system_cursor(icon);
            if icon != MosaicCursor::Default {
                assert_ne!(
                    mapped,
                    SystemCursorIcon::Default,
                    "{icon:?} must not collapse onto the default arrow",
                );
            }
        }
    }
}
