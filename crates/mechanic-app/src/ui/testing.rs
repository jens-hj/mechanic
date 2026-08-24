//! A mounted overlay to point at, shared by every panel's tests.
//!
//! The whole shell rather than one panel: where a panel sits, whether it takes
//! the pointer, and whether the world still shows through beside it are all
//! questions about the tree as a whole, and a panel mounted on its own would
//! answer them differently from the one that ships.

use std::cell::RefCell;

use bevy_mosaic::ui::PaintCmd;
use mosaic_core::{Rect, Scope, Size, Vector2};
use mosaic_widgets::Ui;
use mosaic_widgets::input::{
    Modifiers, PointerButton, PointerEvent, PointerEventKind, PointerType,
};

use super::{Handles, UiIntent, shell, theme};

/// The window the overlay is laid out in for these tests.
pub(crate) const VIEWPORT: Size = Size {
    width: 1600.0,
    height: 900.0,
};

/// How far apart two points are.
pub(crate) fn away(from: Vector2, to: Vector2) -> f32 {
    (from.x - to.x).hypot(from.y - to.y)
}

/// The overlay, mounted and laid out.
pub(crate) struct Overlay {
    ui: Ui,
    /// Everything the tree reads, for a test to write.
    pub(crate) handles: Handles,
    /// What the last frame that painted drew. Kept rather than asked for on
    /// demand: a frame reports only what changed, and a settled tree paints
    /// nothing at all.
    ink: RefCell<Vec<Rect>>,
    _scope: Scope,
}

impl Overlay {
    /// Mounts the whole overlay, as the app does.
    pub(crate) fn mount() -> Self {
        mosaic_core::builtins::install();
        let scope = Scope::new(|| {});
        let ui = scope.run(Ui::new);
        theme::install();

        let handles = Handles::new();
        handles.viewport.set(VIEWPORT);
        let tree = {
            let _ambient = ui.enter();
            shell(&handles)
        };
        ui.mount(&tree);
        let overlay = Overlay {
            ui,
            handles,
            ink: RefCell::new(Vec::new()),
            _scope: scope,
        };
        overlay.settle();
        overlay
    }

    /// Lays the overlay out until it stops moving, so observers that place
    /// content from a resolved rect have taken effect.
    pub(crate) fn settle(&self) {
        for _ in 0..4 {
            mosaic_core::reactive::flush();
            if let Some(scene) = self.ui.frame(VIEWPORT, 1.0) {
                *self.ink.borrow_mut() = scene
                    .cmds
                    .iter()
                    .filter_map(|cmd| match cmd {
                        PaintCmd::Shape(shape) => Some(painted(shape)),
                        _ => None,
                    })
                    .collect();
            }
        }
    }

    /// The box each painted mark covers, stroke included.
    ///
    /// A shape's rect is its own outline's extent, not the element it was
    /// written inside — which is the only way to ask where a drawing actually
    /// landed, since every shape in a canvas lays out as the whole canvas.
    pub(crate) fn ink(&self) -> Vec<Rect> {
        self.ink.borrow().clone()
    }

    /// How many elements the tree holds.
    pub(crate) fn element_count(&self) -> usize {
        self.ui.element_count()
    }

    /// What the overlay has asked for, taken.
    pub(crate) fn intents(&self) -> Vec<UiIntent> {
        self.handles.intents.borrow_mut().drain(..).collect()
    }

    /// Whether a point lands on a panel rather than on the world behind it.
    ///
    /// The same rule the app gates its own input on: a hit on the root is a hit
    /// on the gap between panels.
    pub(crate) fn wants_pointer_at(&self, at: Vector2) -> bool {
        self.ui
            .hit_test(at)
            .is_some_and(|hit| hit != self.ui.root())
    }

    /// Every element in the tree, depth first, as `(depth, rect)`.
    ///
    /// Depth first means a node's descendants are exactly the entries that
    /// follow it until the depth drops back, which is how the assertions say
    /// "everything drawn inside this box".
    pub(crate) fn rects(&self) -> Vec<(usize, Rect)> {
        fn walk(element: &mosaic_widgets::Element, depth: usize, out: &mut Vec<(usize, Rect)>) {
            out.push((depth, element.layout_rect()));
            for child in element.__hot_children() {
                walk(&child, depth + 1, out);
            }
        }
        let mut out = Vec::new();
        walk(&self.ui.root(), 0, &mut out);
        out
    }

    /// Every box the pointer can land on, swept over the whole window.
    ///
    /// Sweeping rather than naming coordinates: what matters is that the
    /// controls are reachable at all, and a sweep says so without pinning the
    /// test to where the layout happens to put them.
    pub(crate) fn reachable_boxes(&self) -> Vec<Rect> {
        let mut found: Vec<Rect> = Vec::new();
        let mut y = 0.0;
        while y < VIEWPORT.height {
            let mut x = 0.0;
            while x < VIEWPORT.width {
                if let Some(element) = self.ui.hit_test(Vector2::new(x, y))
                    && element != self.ui.root()
                {
                    let rect = element.layout_rect();
                    if !found.contains(&rect) {
                        found.push(rect);
                    }
                }
                x += 3.0;
            }
            y += 3.0;
        }
        found
    }

    /// Whether the pointer can land on a box of exactly this size.
    pub(crate) fn reaches_box(&self, boxes: &[Rect], width: f32, height: f32) -> bool {
        let _ = self;
        boxes.iter().any(|rect| {
            (rect.size.width - width).abs() < 0.5 && (rect.size.height - height).abs() < 0.5
        })
    }

    /// Presses and releases the pointer at a point, which is a click.
    pub(crate) fn click(&self, at: Vector2) {
        for kind in [
            PointerEventKind::Down(PointerButton::Primary),
            PointerEventKind::Up(PointerButton::Primary),
        ] {
            self.dispatch(kind, at);
        }
    }

    /// Presses, moves and releases: a drag from one point to another.
    ///
    /// Two moves rather than one, because the first is what carries the gesture
    /// past its slop and starts the drag at all.
    pub(crate) fn drag(&self, from: Vector2, to: Vector2) {
        let midpoint = Vector2::new(f32::midpoint(from.x, to.x), f32::midpoint(from.y, to.y));
        self.dispatch(PointerEventKind::Down(PointerButton::Primary), from);
        for at in [midpoint, to] {
            self.dispatch(PointerEventKind::Move, at);
        }
    }

    /// Sends one pointer event and lets the tree settle.
    pub(crate) fn dispatch(&self, kind: PointerEventKind, at: Vector2) {
        self.ui.dispatch_pointer(PointerEvent {
            kind,
            position: at,
            pointer_type: PointerType::Mouse,
            modifiers: Modifiers::default(),
            timestamp: std::time::Duration::ZERO,
        });
        self.settle();
    }
}

/// What a shape covers once it is painted: its outline, grown by the stroke
/// drawn around it.
///
/// The outline alone is not what is on screen. A stroke straddles the path it
/// follows and a round cap reaches half a width past each end, so a bar drawn
/// as a line covers a stroke width more than the line measures — which is
/// exactly the difference a drawing that overhangs its box is made of.
fn painted(shape: &bevy_mosaic::ui::Shape) -> Rect {
    let widest = shape
        .strokes
        .iter()
        .map(|stroke| stroke.width)
        .fold(0.0_f32, f32::max);
    let out = widest / 2.0;
    Rect::from_xywh(
        shape.rect.origin.x - out,
        shape.rect.origin.y - out,
        shape.rect.size.width + widest,
        shape.rect.size.height + widest,
    )
}
