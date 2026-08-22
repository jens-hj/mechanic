//! Driving one Mosaic frame per Bevy frame.
//!
//! The order here is the same one `mosaic-runtime` uses on its redraw, for the
//! same reasons: animation drivers run before the reactive flush so what they
//! write is settled, the flush runs before assembly so assembly sees a settled
//! graph, and the window is told about the cursor and the input method only
//! after assembly has decided what is focused.

use std::sync::Arc;

use bevy::prelude::*;
use bevy::window::{CursorIcon, SystemCursorIcon, Window};
use mosaic_core::{Size, reactive};
use mosaic_widgets::CursorIcon as MosaicCursor;

use crate::context::{MosaicContext, MosaicFrame};
use crate::input::BevyInput;

/// Push this frame's Bevy input into the tree.
pub(crate) fn process_input(
    context: Option<NonSendMut<MosaicContext>>,
    time: Res<Time>,
    mut input: BevyInput,
) {
    let Some(mut context) = context else {
        return;
    };
    let window = context.window();
    let elapsed = time.elapsed();
    let (ui, state) = context.parts_mut();
    state.drive(ui, window, elapsed, &mut input);
}

/// Settle the tree, assemble a scene, and tell the window what the tree wants.
pub(crate) fn assemble_frame(
    context: Option<NonSendMut<MosaicContext>>,
    mut frame: ResMut<MosaicFrame>,
    time: Res<Time>,
    // The cursor icon is optional: Bevy only puts a `CursorIcon` on a window
    // once something asks for one, so requiring it would match no window at all
    // until the first request — and there would never be a first request.
    mut windows: Query<(&mut Window, Option<&mut CursorIcon>)>,
    mut commands: Commands,
) {
    let Some(mut context) = context else {
        return;
    };
    let window_entity = context.window();
    let Ok((mut window, cursor)) = windows.get_mut(window_entity) else {
        return;
    };

    let scale = window.resolution.scale_factor();
    let size = Size::new(window.resolution.width(), window.resolution.height());

    {
        let (ui, state) = context.parts_mut();
        // A move that arrived this frame and was never forced out by a press
        // still has to reach the tree before anything reads the hover chain.
        state.flush_moves(ui);

        if ui.is_animating() {
            ui.tick(time.delta());
        }
        // The embedder flushes the process-wide reactive graph once a frame;
        // effects armed by input run here, before anything they wrote is read.
        reactive::flush();

        if ui.frame(size, scale).is_some() {
            let scene = ui.scene().clone();
            frame.scene = Some(Arc::new(scene));
            frame.revision = context.bump_revision();
        }
    }
    frame.scale = scale;

    let ui = context.ui();
    let wanted = system_cursor(ui.cursor_icon());
    match cursor {
        // Writing unconditionally would mark the component changed every frame,
        // and the windowing backend re-issues the request when it does.
        Some(mut cursor) if cursor.as_system() != Some(&wanted) => {
            *cursor = CursorIcon::System(wanted);
        }
        Some(_) => {}
        None => {
            commands
                .entity(window_entity)
                .insert(CursorIcon::System(wanted));
        }
    }

    // The input method is enabled only while something takes text, and its
    // candidate window follows the caret. Both are written through `Window`,
    // so guard them: an unconditional write marks the component changed every
    // frame and makes the windowing backend re-issue the request.
    let wants_text = ui.wants_text_input();
    if window.ime_enabled != wants_text {
        window.ime_enabled = wants_text;
    }
    if let Some(area) = ui.ime_cursor_area() {
        let position = Vec2::new(area.origin.x, area.origin.y + area.size.height);
        if window.ime_position != position {
            window.ime_position = position;
        }
    }
}

/// Mosaic's cursor vocabulary as the platform's.
pub(crate) fn system_cursor(icon: MosaicCursor) -> SystemCursorIcon {
    match icon {
        MosaicCursor::Default => SystemCursorIcon::Default,
        MosaicCursor::Pointer => SystemCursorIcon::Pointer,
        MosaicCursor::ResizeHorizontal => SystemCursorIcon::EwResize,
        MosaicCursor::ResizeVertical => SystemCursorIcon::NsResize,
        MosaicCursor::ResizeDiagonalDown => SystemCursorIcon::NwseResize,
        MosaicCursor::ResizeDiagonalUp => SystemCursorIcon::NeswResize,
    }
}
