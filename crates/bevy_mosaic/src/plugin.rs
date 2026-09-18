//! The plugin that installs Mosaic into a Bevy app, and the sets it runs in.

use crate::context::{MosaicContext, MosaicFrame};
use crate::{frame, render};
use bevy::input::InputSystems;
use bevy::log::warn;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

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
