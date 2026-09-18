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

#![expect(
    clippy::needless_pass_by_value,
    reason = "bevy system parameters are value-typed wrappers"
)]

mod context;
mod frame;
mod input;
mod plugin;
pub mod prelude;
mod render;
pub mod ui;

pub use context::MosaicContext;
pub use plugin::{MosaicPlugin, MosaicSystems};
pub use render::MosaicCamera;
