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

mod components;
mod control_block;
mod creations;
mod dimensions;
mod help;
mod hotbar;
pub(crate) mod markers;
mod material_wheel;
mod pause;
mod performance;
mod reticle;
mod styles;
#[cfg(test)]
mod testing;
pub(crate) mod theme;
mod worlds;

use std::cell::RefCell;
use std::rc::Rc;

use bevy::prelude::{
    ButtonInput, Camera, GlobalTransform, KeyCode, NonSend, NonSendMut, Query, Res, ResMut,
    Resource, Single, Window, With, World, info, warn,
};
use bevy_mosaic::MosaicContext;
use mosaic_core::{Size as MosaicSize, State as MosaicState};

use crate::camera::{MainCamera, MaterialWheelState};
use crate::control_panel::ControlPanelState;
use crate::controls::{Controls, GameAction};
use crate::creation_menu::CreationMenuState;
use crate::hotbar::{
    MainTool, MatterMode, SelectedMaterial, SelectedTerrainMaterial, SelectedTool,
};
use crate::pause_menu::PauseMenuState;
use crate::settings::AppSettings;
use crate::showcase::CreationPreset;
use crate::{AppSimulation, EditorGraph, EditorState};
use mechanic_core::ConstructionMaterial;
use mechanic_world::TerrainMaterial;

pub(crate) use control_block::{EditTarget, LocatedJoint};

#[allow(clippy::wildcard_imports)] // Mosaic's authoring vocabulary is meant to be globbed.
use bevy_mosaic::ui::*;
use mosaic_macros::{component, view};

use control_block::{ControlPanel, ControlPanelProps};
use creations::{CreationPicker, CreationPickerProps};
use dimensions::{DimensionOverlay, DimensionOverlayProps};
use help::{HelpPanel, HelpPanelProps};
use hotbar::{Hotbar, HotbarProps};
use markers::{MarkerOverlay, MarkerOverlayProps};
use material_wheel::{RadialSelector, RadialSelectorProps};
use pause::{PauseMenu, PauseMenuProps};
use performance::{PerformanceOverlay, PerformanceOverlayProps};
use reticle::{WorldReticle, WorldReticleProps};
#[allow(unused_imports, clippy::wildcard_imports)]
// Style constants are consumed by `view!` expansion.
use styles::*;
use worlds::{WorldList, WorldListProps};

/// What the overlay is asking the world to do.
///
/// One queue for every panel rather than one each: the order two panels acted
/// in is the order they are applied in, and there is a single place where the
/// overlay is allowed to touch the world.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum UiIntent {
    /// Pick up a tool.
    Tool(MainTool),
    /// Pick a Matter Manipulator mode, activating the tool if necessary.
    MatterMode(MatterMode),
    /// Pick a material and the material-capable tool that owns its menu.
    MaterialMode(ConstructionMaterial, MatterMode),
    /// Something in the creation picker.
    Creations(CreationsAction),
    /// Change a joint's drive program.
    Drive(control_block::Intent),
    /// Change one engine lane's gearbox settings.
    Gearbox(control_block::GearboxIntent),
    /// An action in the full-screen pause menu.
    Pause(PauseAction),
    /// World-list creation, opening, or deletion.
    Worlds(WorldAction),
    /// Put away the control-block panel.
    CloseControlPanel,
}

/// Action requested by the world-list modal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum WorldAction {
    Create { name: String, seed: String },
    Open(std::path::PathBuf),
    Delete(std::path::PathBuf),
}

/// What was changed or requested in the pause menu.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PauseAction {
    Continue,
    OpenOptions,
    OpenControls,
    Back,
    SetCameraFov(f32),
    BeginBindingCapture(GameAction, usize),
    ClearBinding(GameAction, usize),
    ResetControls,
    Exit,
    CancelExit,
    ExitWithoutSaving,
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
    hotbar: MosaicState<SelectedTool>,
    /// Current shortcuts, used by hotbar labels and other guidance.
    controls: MosaicState<Controls>,
    /// Shared material used by both ordinary shape tools.
    material: MosaicState<ConstructionMaterial>,
    /// Material used by terrain additions.
    terrain_material: MosaicState<TerrainMaterial>,
    /// Material tool whose press-and-drag menu is open.
    material_menu: MosaicState<Option<MatterMode>>,
    /// Material row currently under the captured pointer.
    material_hover: MosaicState<Option<ConstructionMaterial>>,
    /// Open/highlight state of the keyboard-owned radial material selector.
    material_wheel: MosaicState<material_wheel::Model>,
    /// Which tool the pointer is over, if any.
    hovered: MosaicState<Option<hotbar::HoverTarget>>,
    /// What the creation picker shows.
    creations: MosaicState<creations::Model>,
    /// Current world-list modal.
    worlds: MosaicState<worlds::Model>,
    /// Where each driven joint's number sits on screen.
    markers: MosaicState<Vec<markers::Marker>>,
    /// Dimensions for the live block-sheet preview.
    dimensions: MosaicState<dimensions::Model>,
    /// What the pause modal shows.
    pause: MosaicState<pause::Model>,
    /// Slider-owned FOV value, kept separate so the native widget can bind it.
    pause_fov: MosaicState<f32>,
    /// The control block's own state.
    block: control_block::Handles,
    /// Opt-in frame, renderer, and physics diagnostics.
    performance: MosaicState<performance::Model>,
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
            hotbar: MosaicState::new(SelectedTool::default()),
            controls: MosaicState::new(Controls::default()),
            material: MosaicState::new(ConstructionMaterial::Steel),
            terrain_material: MosaicState::new(TerrainMaterial::Soil),
            material_menu: MosaicState::new(None),
            material_hover: MosaicState::new(None),
            material_wheel: MosaicState::new(material_wheel::Model::default()),
            hovered: MosaicState::new(None),
            creations: MosaicState::new(creations::Model::default()),
            worlds: MosaicState::new(worlds::Model::default()),
            markers: MosaicState::new(Vec::new()),
            dimensions: MosaicState::new(dimensions::Model::default()),
            pause: MosaicState::new(pause::Model::default()),
            pause_fov: MosaicState::new(crate::settings::DEFAULT_CAMERA_FOV_DEGREES),
            performance: MosaicState::new(performance::Model::default()),
            block: control_block::Handles {
                model: MosaicState::new(control_block::PanelModel::default()),
                selected: MosaicState::new(None),
                located: MosaicState::new(None),
                capturing: MosaicState::new(None),
                gearbox_capturing: MosaicState::new(None),
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
    worlds: worlds::Model,
    markers: Vec<markers::Marker>,
    dimensions: dimensions::Model,
    pause: pause::Model,
    material_wheel: material_wheel::Model,
    performance: performance::Model,
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
        OverlayShell(
            OverlayShellProps::builder()
                .handles(handles.clone())
                .build(),
        )
        .root()
        .clone()
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
#[component]
pub(crate) fn OverlayShell(handles: Handles) -> Element {
    let block_open = handles.block.model;
    let creations_open = handles.creations;
    let material_wheel_model = handles.material_wheel;
    let help_open = handles.help_open;
    let help_panel = handles.clone();
    let hotbar_panel = handles.clone();
    let markers_panel = handles.clone();
    let dimensions_panel = handles.clone();
    let block_panel = handles.clone();
    let creations_panel = handles.clone();
    let worlds_model = handles.worlds;
    let worlds_panel = handles.clone();
    let pause_model = handles.pause;
    let pause_panel = handles.clone();
    let performance_model = handles.performance;
    let performance_viewport = handles.viewport;
    view! {
        stack #mechanic.overlay width:fill height:fill align:start justify:start {
            if !worlds_model.with(|model| model.open) {
                MarkerOverlay handles:(markers_panel.clone())
            }
            if !worlds_model.with(|model| model.open) {
                DimensionOverlay handles:(dimensions_panel.clone())
            }
            if material_wheel_model.with(|model| !model.open)
                && !pause_model.with(|model| model.open)
                && !worlds_model.with(|model| model.open) {
                WorldReticle
            }
            if $help_open && !worlds_model.with(|model| model.open) {
                HelpPanel handles:(help_panel.clone())
            }
            if !worlds_model.with(|model| model.open) {
                Hotbar handles:(hotbar_panel.clone())
            }
            if material_wheel_model.with(|model| model.open)
                && !worlds_model.with(|model| model.open) {
                RadialSelector model:(material_wheel_model)
            }
            if block_open.with(control_block::PanelModel::is_open)
                && !worlds_model.with(|model| model.open) {
                ControlPanel handles:(block_panel.block.clone())
            }
            if creations_open.with(|model| model.open)
                && !worlds_model.with(|model| model.open) {
                CreationPicker handles:(creations_panel.clone())
            }
            if pause_model.with(|model| model.open)
                && !worlds_model.with(|model| model.open) {
                PauseMenu handles:(pause_panel.clone())
            }
            if worlds_model.with(|model| model.open) {
                WorldList handles:(worlds_panel.clone())
            }
            if performance_model.with(performance::Model::is_open)
                && !worlds_model.with(|model| model.open) {
                PerformanceOverlay model:(performance_model) viewport:(performance_viewport)
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
    if let Ok(entries) = std::fs::read_dir(theme::FONT_DIR) {
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
    for (family, install_as_sans) in [(theme::DISPLAY_FAMILY, true), (theme::BODY_FAMILY, false)] {
        if !fonts.has_family(family) {
            info!("overlay: {family} is not installed; falling back");
            continue;
        }
        if install_as_sans {
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
#[derive(bevy::ecs::system::SystemParam)]
pub(crate) struct ToolSelection<'w> {
    tool: ResMut<'w, SelectedTool>,
    material: ResMut<'w, SelectedMaterial>,
}

#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
// Bevy system parameters are value-typed wrappers and independent resources.
pub(crate) fn drain(
    ui: Option<NonSendMut<AppUi>>,
    mut panel: ResMut<ControlPanelState>,
    mut menu: ResMut<CreationMenuState>,
    mut pause: ResMut<PauseMenuState>,
    mut worlds: ResMut<crate::world::WorldListState>,
    mut selection: ToolSelection,
    mut target: EditTarget,
    keyboard: Res<ButtonInput<KeyCode>>,
    actions: Res<ButtonInput<GameAction>>,
) {
    let Some(ui) = ui else {
        return;
    };
    control_block::capture_key(&ui.handles.block, &keyboard);
    // `?` belongs to the help panel unless something else is taking letters.
    if !menu.is_open() && !panel.blocks_keyboard() && actions.just_pressed(GameAction::ToggleHelp) {
        let open = ui.handles.help_open.get_untracked();
        ui.handles.help_open.set(!open);
    }
    let intents: Vec<UiIntent> = ui.handles.intents.borrow_mut().drain(..).collect();
    for intent in intents {
        match intent {
            UiIntent::Tool(tool) => selection.tool.select_tool(tool),
            UiIntent::MatterMode(mode) => selection.tool.select_mode(mode),
            UiIntent::MaterialMode(next, mode) => {
                selection.material.0 = next;
                selection.tool.select_mode(mode);
            }
            UiIntent::Creations(action) => menu.act(action),
            UiIntent::Drive(edit) => control_block::write_joint(&mut panel, &mut target, &edit),
            UiIntent::Gearbox(edit) => control_block::write_gearbox(&panel, &mut target, &edit),
            UiIntent::CloseControlPanel => {
                panel.close();
                ui.handles.block.capturing.set(None);
                ui.handles.block.gearbox_capturing.set(None);
                ui.handles.block.located.set(None);
            }
            UiIntent::Pause(action) => pause.act(action),
            UiIntent::Worlds(action) => worlds.act(action),
        }
    }
}

/// Shows the panels what the world now says.
#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
// Bevy system parameters are value-typed wrappers and independent resources.
pub(crate) fn push(
    ui: Option<NonSendMut<AppUi>>,
    panel: Res<ControlPanelState>,
    menu: Res<CreationMenuState>,
    selection: Res<SelectedTool>,
    material: Res<SelectedMaterial>,
    terrain_material: Res<SelectedTerrainMaterial>,
    graph: Res<EditorGraph>,
    pause: Res<PauseMenuState>,
    worlds_state: Res<crate::world::WorldListState>,
    settings: Res<AppSettings>,
    gearboxes: Res<crate::sequencer::GearboxRuntime>,
    mut located: ResMut<LocatedJoint>,
) {
    let Some(mut ui) = ui else {
        return;
    };
    ui.handles.hotbar.set(*selection);
    ui.handles.controls.set(settings.controls().clone());
    ui.handles.material.set(material.0);
    ui.handles.terrain_material.set(terrain_material.0);
    let block = control_block::capture(&panel, &graph, &gearboxes, settings.controls());
    if block != ui.pushed.block {
        ui.handles.block.model.set(block.clone());
        ui.pushed.block = block;
    }
    let creations = creations::capture(&menu);
    if creations != ui.pushed.creations {
        ui.handles.creations.set(creations.clone());
        ui.pushed.creations = creations;
    }
    let worlds = worlds::capture(&worlds_state);
    if worlds != ui.pushed.worlds {
        ui.handles.worlds.set(worlds.clone());
        ui.pushed.worlds = worlds;
    }
    let pause_model = pause::Model {
        open: pause.is_open(),
        page: pause.page(),
        camera_fov_degrees: settings.camera_fov_degrees(),
        controls: settings.controls().clone(),
        capture: pause.binding_capture(),
        vehicle_conflicts: GameAction::ALL
            .into_iter()
            .filter(|action| {
                settings
                    .controls()
                    .conflicts_with_vehicle(&graph.0, *action)
            })
            .collect(),
    };
    if pause_model != ui.pushed.pause {
        ui.handles.pause.set(pause_model.clone());
        if (ui.handles.pause_fov.get_untracked() - settings.camera_fov_degrees()).abs()
            > f32::EPSILON
        {
            ui.handles.pause_fov.set(settings.camera_fov_degrees());
        }
        ui.pushed.pause = pause_model;
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
    camera: Single<(&Camera, &GlobalTransform), With<MainCamera>>,
) {
    let Some(mut ui) = ui else {
        return;
    };
    let next = markers::capture(&graph, &simulation, selection.active_editor_tool(), &camera);
    if next != ui.pushed.markers {
        ui.handles.markers.set(next.clone());
        ui.pushed.markers = next;
    }
}

/// Projects live block-sheet dimensions onto the screen.
#[allow(clippy::needless_pass_by_value)] // Bevy system parameters are value-typed wrappers.
pub(crate) fn push_dimensions(
    ui: Option<NonSendMut<AppUi>>,
    state: Res<EditorState>,
    simulation: Res<AppSimulation>,
    camera: Single<(&Camera, &GlobalTransform), With<MainCamera>>,
) {
    let Some(mut ui) = ui else {
        return;
    };
    let next = dimensions::capture(&state, &simulation, &camera);
    if next != ui.pushed.dimensions {
        ui.handles.dimensions.set(next.clone());
        ui.pushed.dimensions = next;
    }
}

/// Mirrors the transient world selector into the typed overlay components.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn push_player(ui: Option<NonSendMut<AppUi>>, wheel: Res<MaterialWheelState>) {
    let Some(mut ui) = ui else {
        return;
    };
    let next = material_wheel::Model {
        open: wheel.open,
        highlighted: wheel.highlighted,
    };
    if next != ui.pushed.material_wheel {
        ui.handles.material_wheel.set(next);
        ui.pushed.material_wheel = next;
    }
}

/// Publishes the throttled performance snapshot into the retained overlay.
#[allow(clippy::needless_pass_by_value)] // Bevy system parameters are value-typed wrappers.
pub(crate) fn push_performance(
    ui: Option<NonSendMut<AppUi>>,
    metrics: Res<crate::performance::PerformanceMetrics>,
) {
    let Some(mut ui) = ui else {
        return;
    };
    let next = performance::capture(&metrics.snapshot());
    if next != ui.pushed.performance {
        ui.handles.performance.set(next.clone());
        ui.pushed.performance = next;
    }
}

/// Mirrors the overlay's claim on this frame's input into a plain resource.
///
/// Last of the overlay's systems, because it reports on a tree the others have
/// finished changing.
#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)] // Bevy system parameters are value-typed wrappers.
pub(crate) fn sync_input(
    mosaic: Option<NonSend<MosaicContext>>,
    ui: Option<NonSend<AppUi>>,
    menu: Res<CreationMenuState>,
    panel: Res<ControlPanelState>,
    pause: Res<PauseMenuState>,
    worlds: Res<crate::world::WorldListState>,
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
    let mosaic_keyboard = mosaic.wants_keyboard();
    let menu_open = menu.is_open();
    let panel_keyboard = panel.blocks_keyboard();
    let capturing = ui.handles.block.capturing.get_untracked().is_some();
    *input = UiInput {
        pointer: mosaic.wants_pointer() || pause.blocks_world_input() || worlds.is_open(),
        keyboard: mosaic_keyboard
            || menu_open
            || panel_keyboard
            || pause.blocks_world_input()
            || worlds.is_open(),
        escape: escape_is_consumed(mosaic_keyboard, menu_open, capturing)
            || pause.blocks_world_input()
            || worlds.is_open(),
    };
}

fn escape_is_consumed(mosaic_keyboard: bool, menu_open: bool, capturing: bool) -> bool {
    // An open control panel owns ordinary keys, but Escape is its way out.
    // Only an active field, modal, or key capture gets the first Escape.
    mosaic_keyboard || menu_open || capturing
}

#[cfg(test)]
mod tests {
    use bevy_mosaic::ui::{Color, FontFamily, Length};
    use mechanic_core::ConstructionMaterial;
    use mosaic_core::Vector2;
    use mosaic_core::theme::{color, install as install_theme, typed};
    use mosaic_widgets::Ui;

    use super::material_wheel;
    use super::testing::{Overlay, VIEWPORT, away};
    use super::theme::{BODY_FAMILY, DISPLAY_FAMILY, accent, metrics, palette, typeface};
    use super::{creations, escape_is_consumed, load_fonts, theme};

    #[test]
    fn bundled_fonts_cover_the_authored_type_roles() {
        let ui = Ui::new();
        load_fonts(&ui);
        theme::install();
        let fonts = ui.fonts();
        let fonts = fonts.borrow();

        assert!(fonts.has_family(DISPLAY_FAMILY));
        assert!(fonts.has_family(BODY_FAMILY));
        assert_eq!(
            typed(typeface.display, FontFamily::default),
            FontFamily::Named(DISPLAY_FAMILY.into()),
        );
        assert_eq!(
            typed(typeface.body, FontFamily::default),
            FontFamily::Named(BODY_FAMILY.into()),
        );
    }

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

    #[test]
    fn typed_boundaries_preserve_panel_and_world_pointer_ownership() {
        let overlay = Overlay::mount();
        overlay.handles.help_open.set(true);
        overlay.settle();

        assert!(overlay.wants_pointer_at(Vector2::new(32.0, 32.0)));
        assert!(!overlay.wants_pointer_at(Vector2::new(900.0, 300.0)));
    }

    #[test]
    fn reticle_is_centred_and_never_takes_world_input() {
        let overlay = Overlay::mount();
        let centre = Vector2::new(VIEWPORT.width / 2.0, VIEWPORT.height / 2.0);
        assert!(
            overlay
                .ink()
                .into_iter()
                .any(|rect| away(rect.center(), centre) < 1.0 && rect.size.width < 40.0),
            "the compact reticle is painted at viewport centre",
        );
        assert!(!overlay.wants_pointer_at(centre));
    }

    #[test]
    fn material_wheel_is_large_textured_and_paints_the_highlight_last() {
        let overlay = Overlay::mount();
        overlay.handles.material_wheel.set(material_wheel::Model {
            open: true,
            highlighted: Some(crate::hotbar::WheelChoice::ConstructionMaterial(
                ConstructionMaterial::Concrete,
            )),
        });
        overlay.settle();
        let centre = Vector2::new(VIEWPORT.width / 2.0, VIEWPORT.height / 2.0);
        let sectors = overlay
            .shapes()
            .into_iter()
            .filter(|shape| {
                away(shape.rect.center(), centre) < 1.0
                    && shape
                        .strokes
                        .first()
                        .is_some_and(|stroke| (stroke.width - 72.0).abs() < f32::EPSILON)
            })
            .collect::<Vec<_>>();
        assert_eq!(sectors.len(), ConstructionMaterial::ALL.len());
        assert_eq!(
            sectors
                .last()
                .and_then(|shape| shape.strokes.first())
                .and_then(|stroke| stroke.color.as_solid()),
            Some(color(accent.key)),
        );
        assert!(sectors.iter().all(|shape| shape.rect.size.width > 250.0));

        let last_sector_paint = overlay
            .indexed_shapes()
            .into_iter()
            .filter(|(_, shape)| {
                shape
                    .strokes
                    .first()
                    .is_some_and(|stroke| (stroke.width - 72.0).abs() < f32::EPSILON)
            })
            .map(|(index, _)| index)
            .max()
            .expect("material sector paint");
        let first_block_paint = overlay
            .indexed_images()
            .into_iter()
            .filter(|(_, intrinsic, _)| *intrinsic == (96, 106))
            .map(|(index, _, _)| index)
            .min()
            .expect("material block paint");
        assert!(
            last_sector_paint < first_block_paint,
            "all sector backgrounds paint beneath every material block"
        );

        let block_previews = overlay
            .images()
            .into_iter()
            .filter(|(intrinsic, destination)| {
                *intrinsic == (96, 106)
                    && (destination.size.width - 54.0).abs() < f32::EPSILON
                    && destination.size.height > 59.0
                    && destination.size.height <= 60.0
            })
            .count();
        assert_eq!(block_previews, ConstructionMaterial::ALL.len());

        let label_texture = overlay
            .images()
            .into_iter()
            .find(|(intrinsic, destination)| {
                *intrinsic == (3072, 3072)
                    && (destination.size.width - 236.0).abs() < f32::EPSILON
                    && (destination.size.height - 54.0).abs() < f32::EPSILON
            })
            .expect("the selected material textures the high-resolution swatch");
        assert!((label_texture.1.center().x - centre.x).abs() < f32::EPSILON);
        assert!(
            label_texture.1.origin.y > centre.y + 190.0,
            "the material swatch is spaced below the wheel"
        );

        assert!(overlay.shapes().into_iter().any(|shape| {
            (shape.rect.size.width - 236.0).abs() < f32::EPSILON
                && (shape.rect.size.height - 54.0).abs() < f32::EPSILON
                && (shape.radii.tl - 5.0).abs() < f32::EPSILON
        }));

        assert!(!overlay.wants_pointer_at(centre));
    }

    #[test]
    fn alternate_palette_and_metric_themes_update_the_mounted_tree() {
        let overlay = Overlay::mount();
        overlay.handles.help_open.set(true);
        overlay.settle();
        let elements = overlay.element_count();

        let switched_shell = Color::from_rgb_hex(0x0017_304A);
        let mut switched_palette = palette();
        switched_palette.shell = switched_shell;
        let mut switched_metrics = metrics();
        switched_metrics.radius.panel = Length::px(23.0);
        install_theme(&switched_palette);
        install_theme(&switched_metrics);
        overlay.settle();

        let panel = overlay
            .shapes()
            .into_iter()
            .find(|shape| (shape.rect.size.width - 720.0).abs() < 0.5)
            .expect("the already-mounted help panel repaints");
        assert_eq!(panel.fill.as_solid(), Some(switched_shell));
        assert!((panel.radii.tl - 23.0).abs() < f32::EPSILON);
        assert_eq!(
            overlay.element_count(),
            elements,
            "theme changes do not rebuild"
        );

        theme::install();
    }

    #[test]
    fn escape_is_reserved_only_for_active_panel_editing() {
        assert!(!escape_is_consumed(false, false, false));
        assert!(escape_is_consumed(true, false, false));
        assert!(escape_is_consumed(false, true, false));
        assert!(escape_is_consumed(false, false, true));
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

    #[test]
    fn world_picker_is_an_opaque_standalone_launch_screen() {
        let overlay = Overlay::mount();
        overlay.handles.worlds.update(|model| model.open = true);
        overlay.settle();

        let screen = overlay
            .shapes()
            .into_iter()
            .find(|shape| {
                (shape.rect.size.width - VIEWPORT.width).abs() < 0.5
                    && (shape.rect.size.height - VIEWPORT.height).abs() < 0.5
                    && shape.fill.as_solid() == Some(color(theme::picker.screen))
            })
            .expect("the world picker paints the full window");
        assert_eq!(screen.fill.as_solid(), Some(color(theme::picker.screen)));

        let centre = Vector2::new(VIEWPORT.width / 2.0, VIEWPORT.height / 2.0);
        assert!(overlay.wants_pointer_at(centre));
        assert!(
            !overlay
                .ink()
                .into_iter()
                .any(|rect| away(rect.center(), centre) < 1.0 && rect.size.width < 40.0),
            "the in-game reticle is not mounted behind the world picker",
        );
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
