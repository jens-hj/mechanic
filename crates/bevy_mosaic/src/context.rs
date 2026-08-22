//! The tree, and the frame it hands to the render world.
//!
//! Mosaic's reactive graph lives in a thread-local and every handle into it is
//! `!Send` — one reactive world per UI thread, no locks. Bevy's answer to that
//! is a non-send resource, which pins every system touching it to the main
//! thread. What crosses into the render world is the assembled [`Scene`], which
//! is `Send + Sync` because every shared pointer inside it is an `Arc`.

use std::sync::Arc;

use bevy::prelude::*;
use mosaic_core::reactive::Scope;
use mosaic_render::Scene;
use mosaic_text::FontContext;
use mosaic_widgets::Ui;

use crate::input::InputState;

/// The Mosaic tree driving one window, and the state carried between frames.
///
/// Reach it with `NonSendMut<MosaicContext>` — the `!Send` tree is what makes
/// this a non-send resource, and that is what keeps every system that touches
/// it on the main thread.
///
/// One window is supported. The context records which one so input messages
/// for other windows are ignored; a second tree would be additive, but nothing
/// here pretends to offer it yet.
pub struct MosaicContext {
    /// The retained tree. Cheap to clone — it is a handle to shared storage.
    ui: Ui,
    /// Owns the tree and everything the app builds into it. Lives as long as
    /// the context, and is never explicitly disposed.
    _scope: Scope,
    /// The window this tree draws into and takes input from.
    window: Entity,
    input: InputState,
    /// Bumped whenever a frame assembles a new scene, which is how the extract
    /// system knows a clone is worth making.
    revision: u64,
}

impl MosaicContext {
    /// Build a tree for `window`, with system fonts installed.
    ///
    /// Fonts go in before anything can create text, which is what
    /// [`Ui::set_fonts`] requires.
    pub(crate) fn new(window: Entity) -> Self {
        Self::with_fonts(window, FontContext::new())
    }

    /// Build a tree with a font context the caller chose. Tests use
    /// [`FontContext::embedded_only`] so text metrics do not depend on which
    /// fonts the machine happens to have.
    pub(crate) fn with_fonts(window: Entity, fonts: FontContext) -> Self {
        // Mosaic's own runtime registers these before any tree is built, and
        // the widget defaults are written against them: without this, every
        // `mocha.*` token in a `view!` resolves to nothing.
        mosaic_core::builtins::install();

        let scope = Scope::new(|| {});
        let ui = scope.run(Ui::new);
        ui.set_fonts(fonts);
        MosaicContext {
            ui,
            _scope: scope,
            window,
            input: InputState::default(),
            revision: 0,
        }
    }

    /// The tree. Build into it with [`Ui::mount`], and drive it thereafter by
    /// writing the reactive state its bindings read.
    pub fn ui(&self) -> &Ui {
        &self.ui
    }

    /// The window this context draws into.
    pub fn window(&self) -> Entity {
        self.window
    }

    /// Whether the pointer is over something the overlay will react to.
    ///
    /// The gate an app puts in front of its own pointer handling, so a click on
    /// a panel does not also reach the world behind it. False whenever the
    /// pointer has left the window.
    ///
    /// An overlay's root is normally a full-window container holding panels
    /// with gaps between them, and a hit on that container is a hit on the gap
    /// — the world showing through. So the root itself does not count as
    /// wanting the pointer; anything inside it does. An app that mounts an
    /// interactive element *as* the root should wrap it in a container, or the
    /// pointer over it reads as pointer over background.
    pub fn wants_pointer(&self) -> bool {
        let Some(position) = self.input.position() else {
            return false;
        };
        self.ui
            .hit_test(position)
            .is_some_and(|hit| hit != self.ui.root())
    }

    /// Whether Mosaic is using the keyboard.
    ///
    /// True while a focused element takes text — a text field mid-edit — and
    /// for a key the tree just consumed, so an app's shortcuts do not fire
    /// through a panel that acted on the same key.
    pub fn wants_keyboard(&self) -> bool {
        self.ui.wants_text_input() || self.input.keyboard_consumed()
    }

    pub(crate) fn parts_mut(&mut self) -> (&Ui, &mut InputState) {
        (&self.ui, &mut self.input)
    }

    pub(crate) fn bump_revision(&mut self) -> u64 {
        self.revision += 1;
        self.revision
    }
}

/// The last frame Mosaic assembled, waiting to be extracted.
///
/// A plain `Send` resource standing between the non-send tree and the render
/// world. The scene is behind an `Arc` so extraction is a pointer copy rather
/// than a second deep clone of the command list.
#[derive(Resource, Default)]
pub(crate) struct MosaicFrame {
    pub(crate) scene: Option<Arc<Scene>>,
    /// Matches [`MosaicContext`]'s revision at the time the scene was taken.
    pub(crate) revision: u64,
    /// The window's DPI scale factor, which the renderer needs to turn the
    /// scene's logical coordinates into pixels.
    pub(crate) scale: f32,
}

#[cfg(test)]
mod tests {
    use super::MosaicContext;
    use bevy::prelude::Entity;
    use mosaic_core::{Size, Vector2};
    use mosaic_layout::{Dimension, Style};
    use mosaic_text::FontContext;

    /// A context whose tree is one 100×50 panel in the top-left corner.
    fn context_with_a_panel() -> MosaicContext {
        let context = MosaicContext::with_fonts(Entity::PLACEHOLDER, FontContext::embedded_only());
        let ui = context.ui();
        let root = ui.root();
        root.style(Style::column().size(Dimension::Fill, Dimension::Fill));
        let panel = root.child(Style::column().size(Dimension::Px(100.0), Dimension::Px(50.0)));
        panel.hit_testable(true);
        ui.frame(Size::new(400.0, 300.0), 1.0);
        context
    }

    #[test]
    fn a_pointer_that_never_entered_the_window_wants_nothing() {
        let context = context_with_a_panel();
        assert!(
            !context.wants_pointer(),
            "with no pointer position there is nothing to be over",
        );
    }

    #[test]
    fn a_pointer_over_a_panel_wants_the_pointer() {
        let mut context = context_with_a_panel();
        context.parts_mut().1.note_move(Vector2::new(20.0, 20.0));
        assert!(
            context.wants_pointer(),
            "the app must not also act on a click that landed on the overlay",
        );
    }

    #[test]
    fn a_pointer_past_the_panel_leaves_the_pointer_to_the_app() {
        let mut context = context_with_a_panel();
        context.parts_mut().1.note_move(Vector2::new(300.0, 200.0));
        assert!(
            !context.wants_pointer(),
            "empty space over the world is the world's to handle",
        );
    }

    #[test]
    fn an_idle_tree_does_not_hold_the_keyboard() {
        let context = context_with_a_panel();
        assert!(
            !context.wants_keyboard(),
            "with nothing focused, every shortcut belongs to the app",
        );
    }
}
