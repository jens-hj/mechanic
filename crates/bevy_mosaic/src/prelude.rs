//! This crate's own API, and the `view!` macro.
//!
//! Deliberately narrow. Mosaic and Bevy both have a `State`, a `Children`, and
//! an `Interaction`, so a prelude that re-exported Mosaic's widget vocabulary
//! would collide with `bevy::prelude::*` on all three the moment anyone glob
//! imported both. UI-authoring code imports what it needs from `mosaic_core`
//! and `mosaic_widgets` by name instead — which it depends on directly anyway,
//! because that is how `view!` resolves its paths.

pub use crate::{MosaicCamera, MosaicContext, MosaicPlugin, MosaicSystems};
pub use mosaic_macros::view;
