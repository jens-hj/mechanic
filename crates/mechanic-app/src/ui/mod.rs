//! Mechanic's overlay: every panel, in one Mosaic tree.
//!
//! There is one tree per window — [`mosaic_widgets::Ui::mount`] replaces the
//! root rather than adding to it — so the panels are not independent overlays
//! but siblings under one shell. That is what makes the awkward parts go away:
//! stacking is tree order rather than a fight between two renderers, and
//! whether the pointer belongs to the world or to a panel is one question with
//! one answer ([`bevy_mosaic::MosaicContext::wants_pointer`]).
//!
//! The world stays the truth. Each panel renders a snapshot pushed in from the
//! ECS, and hands back [`UiIntent`]s that a Bevy system folds into the world a
//! moment later. Nothing in a view touches a resource, and nothing in the world
//! knows a view exists.

mod control_block;
mod creations;
mod help;
mod hotbar;
pub(crate) mod markers;
#[cfg(test)]
mod testing;
pub(crate) mod theme;

use std::cell::RefCell;
use std::rc::Rc;

use bevy::input::keyboard::Key;
use bevy::prelude::{
    ButtonInput, Camera, GlobalTransform, KeyCode, NonSend, NonSendMut, Query, Res, ResMut,
    Resource, Single, Window, With, World, info, warn,
};
use bevy_mosaic::MosaicContext;
use mosaic_core::{Size as MosaicSize, State as MosaicState};

use crate::control_panel::ControlPanelState;
use crate::creation_menu::CreationMenuState;
use crate::hotbar::{SelectedTool, Tool};
use crate::showcase::CreationPreset;
use crate::{AppSimulation, EditorGraph, OrbitCamera};

pub(crate) use control_block::{EditTarget, LocatedJoint};

#[allow(clippy::wildcard_imports)] // Mosaic's authoring vocabulary is meant to be globbed.
use bevy_mosaic::ui::*;
use mosaic_macros::view;

/// What the overlay is asking the world to do.
///
/// One queue for every panel rather than one each: the order two panels acted
/// in is the order they are applied in, and there is a single place where the
/// overlay is allowed to touch the world.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum UiIntent {
    /// Pick up a tool.
    Tool(Tool),
    /// Something in the creation picker.
    Creations(CreationsAction),
    /// Change a joint's drive program.
    Drive(control_block::Intent),
}

/// What was asked of the creation picker.
///
/// The decisions themselves — whether a save needs confirming, whether a delete
/// has been asked for twice — stay in [`CreationMenuState`]; this only says what
/// the person did.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CreationsAction {
    /// The name field now reads this.
    Name(String),
    /// Save under the typed name.
    Save,
    /// Open this saved creation.
    Load(std::path::PathBuf),
    /// Remove this saved creation.
    Delete(std::path::PathBuf),
    /// Open a built-in scene.
    Preset(CreationPreset),
    /// Back out.
    Cancel,
}

/// Every piece of reactive state the overlay reads.
///
/// Handed around by value: each field is a `Copy` handle into the reactive
/// graph or a shared queue, so cloning one costs a pointer.
#[derive(Clone)]
pub(crate) struct Handles {
    /// The window, so a panel can place itself against an edge it cannot see.
    viewport: MosaicState<MosaicSize>,
    /// What the help panel says.
    help: MosaicState<help::Model>,
    /// Whether the help panel is showing.
    help_open: MosaicState<bool>,
    /// Which tool is in hand.
    hotbar: MosaicState<Tool>,
    /// Which tool the pointer is over, if any.
    hovered: MosaicState<Option<Tool>>,
    /// What the creation picker shows.
    creations: MosaicState<creations::Model>,
    /// Where each driven joint's number sits on screen.
    markers: MosaicState<Vec<markers::Marker>>,
    /// The control block's own state.
    block: control_block::Handles,
    /// What the overlay is asking for.
    intents: Rc<RefCell<Vec<UiIntent>>>,
}

impl Handles {
    /// Builds a fresh, unmounted set of handles.
    fn new() -> Self {
        let intents = Rc::new(RefCell::new(Vec::new()));
        Handles {
            viewport: MosaicState::new(MosaicSize::ZERO),
            help: MosaicState::new(help::Model::default()),
            help_open: MosaicState::new(false),
            hotbar: MosaicState::new(Tool::default()),
            hovered: MosaicState::new(None),
            creations: MosaicState::new(creations::Model::default()),
            markers: MosaicState::new(Vec::new()),
            block: control_block::Handles {
                model: MosaicState::new(control_block::PanelModel::default()),
                selected: MosaicState::new(None),
                located: MosaicState::new(None),
                capturing: MosaicState::new(None),
                intents: Rc::clone(&intents),
            },
            intents,
        }
    }

    /// Queues one intent.
    fn ask(&self, intent: UiIntent) {
        self.intents.borrow_mut().push(intent);
    }
}

/// The overlay's live tree and the state driving it.
///
/// Non-send because Mosaic's reactive graph is: every handle in here is a
/// pointer into a thread-local arena.
pub(crate) struct AppUi {
    handles: Handles,
    /// What was last pushed, so an unchanged world does not re-render.
    pushed: Pushed,
}

/// The last snapshot handed to each panel.
#[derive(Default)]
struct Pushed {
    help: help::Model,
    creations: creations::Model,
    markers: Vec<markers::Marker>,
    block: control_block::PanelModel,
}

/// What the world is allowed to do with this frame's input.
///
/// A plain resource mirroring the overlay's answer, so the systems that drive
/// the scene read a `bool` instead of being pinned to the main thread by the
/// non-send context.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub(crate) struct UiInput {
    /// The pointer is over a panel.
    pointer: bool,
    /// A panel is using the keyboard.
    keyboard: bool,
    /// Escape belongs to a panel this frame.
    escape: bool,
}

impl UiInput {
    /// Whether the pointer landed on the overlay rather than the world.
    pub(crate) const fn blocks_pointer(self) -> bool {
        self.pointer
    }

    /// Whether the overlay is taking the keyboard.
    pub(crate) const fn blocks_keyboard(self) -> bool {
        self.keyboard
    }

    /// Whether Escape has already been spoken for.
    pub(crate) const fn escape_is_consumed(self) -> bool {
        self.escape
    }
}

/// Where the design's two typefaces are looked for.
const FONT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/fonts");

/// The typeface the design sets its chrome in.
const CHROME_FAMILY: &str = "Chakra Petch";

/// The typeface the design sets every number in.
const NUMERIC_FAMILY: &str = "JetBrains Mono";

/// Builds the overlay and hands it to Mosaic.
///
/// Exclusive because the tree is `!Send`, and inserting a non-send resource
/// needs the world itself.
pub(crate) fn mount(world: &mut World) {
    let Some(mosaic) = world.get_non_send::<MosaicContext>() else {
        warn!("no Mosaic context; the overlay has nothing to draw into");
        return;
    };
    let ui = mosaic.ui().clone();
    load_fonts(&ui);
    theme::install();

    let handles = Handles::new();
    let tree = {
        let _ambient = ui.enter();
        shell(&handles)
    };
    ui.mount(&tree);

    world.insert_non_send(AppUi {
        handles,
        pushed: Pushed::default(),
    });
    world.init_resource::<LocatedJoint>();
    world.init_resource::<UiInput>();
}

/// The whole overlay.
///
/// A stack, so every panel is placed from the same corner and puts itself where
/// it belongs with `translate:`. The root is the one element allowed to fill the
/// window: a hit on it reads as a hit on the world behind, and a hit on anything
/// inside it does not — so every panel hugs its own contents, and the two that
/// deliberately cover the window (the block, the picker) are the two that are
/// meant to take the pointer with them.
fn shell(handles: &Handles) -> Element {
    let block_open = handles.block.model;
    let creations_open = handles.creations;
    let help_open = handles.help_open;
    let help_panel = handles.clone();
    let hotbar_panel = handles.clone();
    let markers_panel = handles.clone();
    let block_panel = handles.clone();
    let creations_panel = handles.clone();
    view! {
        stack width:fill height:fill align:start justify:start {
            (markers::view(&markers_panel))
            if $help_open {
                (help::view(&help_panel))
            }
            (hotbar::view(&hotbar_panel))
            if block_open.with(control_block::PanelModel::is_open) {
                (control_block::panel(&block_panel.block))
            }
            if creations_open.with(|model| model.open) {
                (creations::view(&creations_panel))
            }
        }
    }
}

/// Installs the design's typefaces, when they are on hand.
///
/// Any font file dropped into `assets/fonts` is loaded, and the two the design
/// names become the overlay's sans and monospace defaults if they turn up —
/// either from there or from the system. Neither is required: the overlay is
/// legible in whatever the machine offers, so a missing file is worth a line in
/// the log rather than a failure to start.
fn load_fonts(ui: &mosaic_widgets::Ui) {
    let fonts = ui.fonts();
    if let Ok(entries) = std::fs::read_dir(FONT_DIR) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path
                .extension()
                .is_some_and(|extension| extension == "ttf" || extension == "otf")
            {
                continue;
            }
            match std::fs::read(&path) {
                Ok(data) => fonts.borrow_mut().load_font_data(data),
                Err(error) => warn!("overlay: cannot read {}: {error}", path.display()),
            }
        }
    }

    let mut fonts = fonts.borrow_mut();
    for (family, install) in [(CHROME_FAMILY, true), (NUMERIC_FAMILY, false)] {
        if !fonts.has_family(family) {
            info!("overlay: {family} is not installed; falling back");
            continue;
        }
        if install {
            fonts.set_sans_serif_family(family);
        } else {
            fonts.set_monospace_family(family);
        }
    }
}

/// Applies everything the overlay asked for.
///
/// Runs before the pushes, so an edit made this frame is already in the snapshot
/// the panels get: a dial under the pointer never lags the pointer by a frame.
#[allow(clippy::needless_pass_by_value)] // Bevy system parameters are value-typed wrappers.
pub(crate) fn drain(
    ui: Option<NonSendMut<AppUi>>,
    panel: Res<ControlPanelState>,
    mut menu: ResMut<CreationMenuState>,
    mut selection: ResMut<SelectedTool>,
    mut target: EditTarget,
    keyboard: Res<ButtonInput<KeyCode>>,
    typed: Res<ButtonInput<Key>>,
) {
    let Some(ui) = ui else {
        return;
    };
    control_block::capture_key(&ui.handles.block, &keyboard);
    // `?` belongs to the help panel unless something else is taking letters.
    if !menu.is_open() && !panel.blocks_keyboard() && crate::help_toggle_requested(&typed) {
        let open = ui.handles.help_open.get_untracked();
        ui.handles.help_open.set(!open);
    }
    let intents: Vec<UiIntent> = ui.handles.intents.borrow_mut().drain(..).collect();
    for intent in intents {
        match intent {
            UiIntent::Tool(tool) => selection.0 = tool,
            UiIntent::Creations(action) => menu.act(action),
            UiIntent::Drive(edit) => control_block::write_joint(&panel, &mut target, &edit),
        }
    }
}

/// Shows the panels what the world now says.
#[allow(clippy::needless_pass_by_value)] // Bevy system parameters are value-typed wrappers.
pub(crate) fn push(
    ui: Option<NonSendMut<AppUi>>,
    panel: Res<ControlPanelState>,
    menu: Res<CreationMenuState>,
    selection: Res<SelectedTool>,
    graph: Res<EditorGraph>,
    mut located: ResMut<LocatedJoint>,
) {
    let Some(mut ui) = ui else {
        return;
    };
    ui.handles.hotbar.set(selection.0);
    let block = control_block::capture(&panel, &graph);
    if block != ui.pushed.block {
        ui.handles.block.model.set(block.clone());
        ui.pushed.block = block;
    }
    let creations = creations::capture(&menu);
    if creations != ui.pushed.creations {
        ui.handles.creations.set(creations.clone());
        ui.pushed.creations = creations;
    }
    located.0 = ui.handles.block.located.get_untracked();
}

/// Rebuilds the help panel's lines.
///
/// Its own system because what it says is drawn from most of the editor at
/// once, and grouping those reads anywhere else would drag them into a system
/// that has no other use for them.
#[allow(clippy::needless_pass_by_value)] // Bevy system parameters are value-typed wrappers.
pub(crate) fn push_help(ui: Option<NonSendMut<AppUi>>, sources: help::Sources) {
    let Some(mut ui) = ui else {
        return;
    };
    let next = help::capture(&sources);
    if next != ui.pushed.help {
        ui.handles.help.set(next.clone());
        ui.pushed.help = next;
    }
}

/// Projects each driven joint's number onto the screen.
#[allow(clippy::needless_pass_by_value)] // Bevy system parameters are value-typed wrappers.
pub(crate) fn push_markers(
    ui: Option<NonSendMut<AppUi>>,
    graph: Res<EditorGraph>,
    simulation: Res<AppSimulation>,
    selection: Res<SelectedTool>,
    camera: Single<(&Camera, &GlobalTransform), With<OrbitCamera>>,
) {
    let Some(mut ui) = ui else {
        return;
    };
    let next = markers::capture(&graph, &simulation, selection.0, &camera);
    if next != ui.pushed.markers {
        ui.handles.markers.set(next.clone());
        ui.pushed.markers = next;
    }
}

/// Mirrors the overlay's claim on this frame's input into a plain resource.
///
/// Last of the overlay's systems, because it reports on a tree the others have
/// finished changing.
#[allow(clippy::needless_pass_by_value)] // Bevy system parameters are value-typed wrappers.
pub(crate) fn sync_input(
    mosaic: Option<NonSend<MosaicContext>>,
    ui: Option<NonSend<AppUi>>,
    menu: Res<CreationMenuState>,
    panel: Res<ControlPanelState>,
    windows: Query<&Window>,
    mut input: ResMut<UiInput>,
) {
    let (Some(mosaic), Some(ui)) = (mosaic, ui) else {
        return;
    };
    // The same size the overlay is laid out at, read from the same window it
    // draws into: a panel placing itself against an edge has to agree with the
    // renderer about where that edge is.
    if let Ok(window) = windows.get(mosaic.window()) {
        let size = MosaicSize::new(window.width(), window.height());
        if ui.handles.viewport.get_untracked() != size {
            ui.handles.viewport.set(size);
        }
    }
    // A panel holds the keyboard for as long as it is open, not only while a
    // field has the focus: a digit typed into a name must never also pick up a
    // tool behind it.
    let typing = mosaic.wants_keyboard() || menu.is_open() || panel.blocks_keyboard();
    *input = UiInput {
        pointer: mosaic.wants_pointer(),
        keyboard: typing,
        escape: typing || ui.handles.block.capturing.get_untracked().is_some(),
    };
}

#[cfg(test)]
mod tests {
    use mosaic_core::Vector2;

    use super::creations;
    use super::testing::{Overlay, VIEWPORT};

    /// The bug this guards against: a `stack` child fills its parent unless it
    /// is told not to, and one full-bleed panel makes every point in the window
    /// read as "over the overlay" — the machine behind it stops taking the
    /// pointer at all, everywhere, with nothing on screen to explain why.
    #[test]
    fn the_world_shows_through_the_gaps_between_panels() {
        let overlay = Overlay::mount();
        // The middle of the window, where nothing is drawn: the help panel is
        // in the corner and hidden, and the hotbar is along the bottom.
        for at in [
            Vector2::new(VIEWPORT.width / 2.0, VIEWPORT.height / 2.0),
            Vector2::new(VIEWPORT.width - 8.0, 8.0),
            Vector2::new(8.0, VIEWPORT.height / 2.0),
        ] {
            assert!(
                !overlay.wants_pointer_at(at),
                "the world must keep the pointer at {at:?}",
            );
        }
    }

    /// And the panels themselves do take it, or a click would fall through the
    /// thing it landed on.
    #[test]
    fn a_panel_takes_the_pointer_over_itself() {
        let overlay = Overlay::mount();
        let bar = overlay
            .reachable_boxes()
            .into_iter()
            .find(|rect| (rect.size.width - 64.0).abs() < 0.5)
            .expect("the hotbar is on screen");
        assert!(overlay.wants_pointer_at(bar.center()));
    }

    /// The picker is the one panel that is meant to cover the window: while it
    /// is up, a click anywhere is a click on it.
    #[test]
    fn the_open_picker_covers_the_window() {
        let overlay = Overlay::mount();
        overlay.handles.creations.update(|model| model.open = true);
        overlay.settle();
        for at in [
            Vector2::new(VIEWPORT.width / 2.0, VIEWPORT.height / 2.0),
            Vector2::new(4.0, 4.0),
            Vector2::new(VIEWPORT.width - 4.0, VIEWPORT.height - 4.0),
        ] {
            assert!(
                overlay.wants_pointer_at(at),
                "a modal that lets clicks through at {at:?} is not a modal",
            );
        }
    }

    /// A `scroll` fills its parent whatever size is written on it — the size
    /// styles the content it holds — so a sheet that *is* the scroll cannot be
    /// centred by its veil, and lands in the corner instead.
    #[test]
    fn the_picker_sits_in_the_middle_of_the_window() {
        let overlay = Overlay::mount();
        overlay.handles.creations.update(|model| model.open = true);
        overlay.settle();

        let sheet = overlay
            .rects()
            .into_iter()
            .map(|(_, rect)| rect)
            .find(|rect| (rect.size.width - creations::SHEET).abs() < 0.5)
            .expect("the picker is on screen");
        let (sits, middle) = (
            sheet.center(),
            Vector2::new(VIEWPORT.width / 2.0, VIEWPORT.height / 2.0),
        );
        assert!(
            (sits.x - middle.x).abs() < 0.5 && (sits.y - middle.y).abs() < 0.5,
            "the picker sits at {sits:?} in a window centred on {middle:?}",
        );
    }
}
