//! Run the [Mosaic](https://gitlab.com/unincorporated/mosaic) GUI framework
//! inside a Bevy app, the way `bevy_egui` does for egui.
//!
//! # What this replaces
//!
//! Mosaic normally owns the whole stack: `mosaic-runtime` drives a `winit`
//! event loop, owns the window, and builds a `WgpuRenderer` around its own
//! surface. None of that can coexist with Bevy, which owns all three. This
//! crate stands in for `mosaic-runtime` and nothing else — the tree, the layout
//! engine, the widgets, and the wgpu backend are the real ones.
//!
//! # How the halves divide
//!
//! Mosaic's reactive graph is thread-local and `!Send` by design, so the tree
//! lives in the main world as a non-send [`MosaicContext`]. What it produces —
//! a `Scene`, which is `Send + Sync` — is what crosses into the render world.
//! That split is Bevy's own, and Mosaic happens to be cut along the same line.
//!
//! Bevy and Mosaic must resolve to the same `wgpu`, because the renderer is
//! handed Bevy's device and queue directly. There is no bridging layer and no
//! second GPU context.
//!
//! Any crate that writes a `view!` also depends on `mosaic-core`,
//! `mosaic-widgets` and `mosaic-macros` directly: the macro resolves its paths
//! through the calling crate's manifest, so it cannot borrow this crate's.
//!
//! # Using it
//!
//! Add the plugin, mark the camera the overlay belongs in front of, and mount a
//! tree once at startup. Mosaic is retained and reactive: the tree is built
//! once, and afterwards the app drives it by writing the reactive state its
//! bindings read, not by rebuilding.
//!
//! ```no_run
//! use bevy::prelude::*;
//! use bevy_mosaic::prelude::*;
//!
//! fn main() {
//!     App::new()
//!         .add_plugins((DefaultPlugins, MosaicPlugin))
//!         .add_systems(Startup, setup)
//!         .run();
//! }
//!
//! fn setup(mut commands: Commands, mosaic: NonSend<MosaicContext>) {
//!     commands.spawn((Camera3d::default(), MosaicCamera));
//!
//!     let ui = mosaic.ui();
//!     let view = panel::build(ui);
//!     ui.mount(&view);
//! }
//!
//! // The UI goes in its own module so it can glob Mosaic's vocabulary without
//! // fighting Bevy's prelude over `State`, `Children` and `Interaction`.
//! mod panel {
//!     use bevy_mosaic::ui::*;
//!
//!     pub fn build(ui: &Ui) -> Element {
//!         let count: State<i64> = State::new(0);
//!         let _ambient = ui.enter();
//!         view! {
//!             col pad:24px gap:8px height:min-content {
//!                 text font-color:mocha.text "Hello from Mosaic"
//!                 button @click:{ $count += 1 } { format!("clicked {} times", $count) }
//!             }
//!         }
//!     }
//! }
//! ```
//!
//! # Not here yet
//!
//! One window per app; the OS clipboard (Mosaic's in-process one is installed,
//! so copy and paste work within the app but not across it); accessibility,
//! which Mosaic exposes as an AccessKit tree that nothing here forwards yet;
//! and touch, which Mosaic's own runtime recognizes into pointer and pinch
//! gestures. Backdrop filters sample Mosaic's own root rather than the Bevy
//! scene behind them, so a glass panel reads as glass over nothing.

#![allow(clippy::needless_pass_by_value)] // Bevy system parameters are value-typed wrappers.

mod context;
mod frame;
mod input;
mod render;

pub use context::MosaicContext;
pub use render::MosaicCamera;

use bevy::input::InputSystems;
use bevy::log::warn;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::context::MosaicFrame;

/// This crate's own API, and the `view!` macro.
///
/// Deliberately narrow. Mosaic and Bevy both have a `State`, a `Children`, and
/// an `Interaction`, so a prelude that re-exported Mosaic's widget vocabulary
/// would collide with `bevy::prelude::*` on all three the moment anyone glob
/// imported both. UI-authoring code imports what it needs from `mosaic_core`
/// and `mosaic_widgets` by name instead — which it depends on directly anyway,
/// because that is how `view!` resolves its paths.
pub mod prelude {
    pub use crate::{MosaicCamera, MosaicContext, MosaicPlugin, MosaicSystems};
    pub use mosaic_macros::view;
}

/// Mosaic's authoring vocabulary: widgets, layout, paint, reactive state, the
/// theme palettes, and the `view!` macro.
///
/// This mirrors `mosaic::prelude` with the windowing items removed — `App`,
/// `WindowConfig`, `AppContext` and friends belong to Mosaic's own runtime,
/// which a Bevy app replaces. Skipping them is also what keeps `winit`,
/// `arboard` and `accesskit_winit` out of the dependency graph.
///
/// Glob import it in modules that build UI — under clippy's pedantic lints
/// that needs an `#[allow(clippy::wildcard_imports)]` on the module, which is
/// the trade for `view!` reading the way it does everywhere else. It is kept
/// separate from
/// [`prelude`] because Mosaic and Bevy both define `State`, `Children` and
/// `Interaction`: importing both preludes into one module is ambiguous on all
/// three, which is a decision only the calling module can make.
pub mod ui {
    pub use mosaic_core::{
        Catppuccin, CatppuccinFlavor, Color, ColorToken, Derived, DisplayP3, Easing, Effect,
        ElementToken, FRAPPE, Hsl, Hwb, IntoFontLength, LATTE, Lab, Lch, LengthToken, MACCHIATO,
        MOCHA, Motion, Oklab, Oklch, PaintToken, ReadState, Rect, ScalarToken, Scope, Size, Spring,
        Srgb, State, StateSender, SvgToken, SystemScheme, ThemeToken, Vector2, ViewBinding,
        ViewBuildGuard, batch, col, ease, frappe, install_theme, latte, macchiato, mocha,
        on_cleanup, resolve_theme_value, spring, state_channel, untracked,
    };
    pub use mosaic_layout::{
        Align, AxisPair, Dimension, Direction, Edges, Grid, GridPacking, GridTrack, GridTracks,
        Inherit, Justify, LayoutMode, Length, SizeBound, Style, Translate,
    };
    pub use mosaic_macros::{
        component, displacement, preview, scheme, style, surface, theme, view,
    };
    pub use mosaic_render::{
        Anchor, ArcSpec, BackdropFilter, BackdropSample, BoxPoint, BoxSize, CIRCULAR_EXPONENT,
        ColorEdit, CornerSpec, DirectionalLight, DisplacementField, Extend, FieldProgram,
        FieldUniform, FilterChain, FilterLength, FilterScale, GaussianPlan, GeometrySpec,
        GradientSpec, GradientStop, InterpSpace, KindSpec, LayerSpec, LightKindSpec, LightPosition,
        LightRecordBase, LightSpec, LightSpill, LightTargets, LineCap, LineJoin, MarkerEnd,
        MarkerShape, MarkerSpec, MaskComposite, MaskMode, MeshSpec, PaintSpec, PlannedFilter,
        PointLight, PointSpec, Radius, Reach, Refraction, ShadowSpec, SpotLight, StopSpec,
        StrokeEdges, StrokeSpec, SurfaceProfile, Theme, ThemedColor, ThemedF32, ThemedVec2,
        even_stops, plan_gaussian,
    };
    pub use mosaic_text::{
        FontFamily, FontStretch, FontStyle, IntoLineHeight, LineHeight, TextStyle, TextTransform,
        TextWrap,
    };
    pub use mosaic_widgets::input::{
        Clipboard, ImeEvent, Key, KeyEvent, KeyEventKind, Modifiers, PointerButton, PointerEvent,
        PointerEventKind, PointerType, TextInputContents,
    };
    pub use mosaic_widgets::{
        Animated, ButtonStyle, Checkbox, CheckboxStyle, Children, ComponentHandle, DragAxis,
        DragEvent, DragOptions, DragPhase, DragRelease, EditorState, Element, ElementPatch,
        ElementSpec, FindBar, FindBarStyle, FocusStyle, Fx, IconPaints, IconPart, IconPartStyle,
        IconSource, IconStroke, IconStyle, IconValue, ImageSource, ImgStyle, InspectionAttribute,
        InspectionAttributeOrigin, InspectionBoundary, InspectionBoundaryKind, InspectionDetails,
        InspectionDetailsSeed, InspectionNode, InspectionSnapshot, InspectionTreeId, Interaction,
        IntoIconColor, IntoIconStroke, IntoSemanticRole, IntoSemanticText, IntoTextContent,
        LayoutSize, MaskSpec, ObjectFit, OverlayAlign, OverlayCollision, OverlayPlacement,
        OverlayPoint, OverlayPosition, OverlaySide, Progress, ProgressStyle, Prop, Radii, Radio,
        RadioStyle, ReorderAxis, ReorderEvent, ReorderLocation, ReorderMode, ReorderOptions,
        ResizeEdges, ResizeEvent, ResizeOptions, ResizePhase, Role, Scroll, Select, SelectStyle,
        Semantics, Slider, SliderStyle, SpanBackground, Stepper, StepperStyle, StyleCtx, StylePart,
        StyleSet, SvgSpec, TextAreaStyle, TextContent, TextEditor, TextEditorStyle,
        TextInputOptions, TextInputStyle, TextSpan, ThemeAsset, Toggle, ToggleStyle, Tooltip,
        TooltipOptions, TooltipTrigger, Transition, Ui, VirtualList, Visual, bind_anchored_overlay,
        button, button_container, button_container_styled, button_styled, canvas_fit, checkbox,
        checkbox_labeled, checkbox_styled, fade, find_bar, find_bar_styled, fly, icon, icon_dyn,
        icon_element, img, img_dyn, peek_icon_element, peek_svg, progress, progress_dyn,
        progress_dyn_styled, progress_styled, radio, radio_labeled, radio_styled,
        resolve_icon_color, resolve_icon_stroke, scroll, select, select_styled, set_icon_element,
        set_svg, shape, shape_content, shape_filled, slide, slide_x, slider, slider_styled,
        stepper, stepper_styled, style_text_tooltip_root, text, text_area, text_area_styled,
        text_area_styled_with_options, text_area_with_options, text_dyn, text_dyn_styled,
        text_editor, text_editor_styled, text_input, text_input_styled,
        text_input_styled_with_options, text_input_with_options, text_tooltip, toggle,
        toggle_styled, tooltip, virtual_list, virtual_list_measured,
    };
}

/// Where this crate's work sits in the frame, so an app can order against it.
///
/// An app's own systems belong between the two: input has reached the tree by
/// the end of [`ProcessInput`](MosaicSystems::ProcessInput), and whatever they
/// write to reactive state is picked up by
/// [`AssembleFrame`](MosaicSystems::AssembleFrame).
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MosaicSystems {
    /// Bevy's input messages are translated and dispatched into the tree.
    /// Runs in `PreUpdate`, after Bevy's own input handling.
    ProcessInput,
    /// Animations tick, the reactive graph settles, and a scene is assembled.
    /// Runs in `PostUpdate`.
    AssembleFrame,
}

/// Installs Mosaic into a Bevy app.
///
/// Creates one [`MosaicContext`] for the primary window in `PreStartup`, so a
/// `Startup` system can mount a tree into it.
pub struct MosaicPlugin;

impl Plugin for MosaicPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MosaicFrame>()
            .add_plugins(render::MosaicRenderPlugin)
            .add_systems(PreStartup, create_context)
            .add_systems(
                PreUpdate,
                frame::process_input
                    .in_set(MosaicSystems::ProcessInput)
                    .after(InputSystems),
            )
            .add_systems(
                PostUpdate,
                frame::assemble_frame.in_set(MosaicSystems::AssembleFrame),
            );
    }
}

/// Build the tree for the primary window.
///
/// An exclusive system because the context is `!Send`, and inserting a non-send
/// resource needs the world itself.
fn create_context(world: &mut World) {
    let mut windows = world.query_filtered::<Entity, With<PrimaryWindow>>();
    let Some(window) = windows.iter(world).next() else {
        warn!("no primary window; Mosaic has nothing to draw into");
        return;
    };
    let context = MosaicContext::new(window);
    world.insert_non_send(context);
}
