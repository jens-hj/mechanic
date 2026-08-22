//! Painting the assembled scene into a Bevy view.
//!
//! Mechanic's Bevy and Mosaic resolve to the same `wgpu`, so there is no bridge
//! here in the usual sense: the renderer is handed Bevy's own device and queue
//! and asked for a command buffer, which goes into Bevy's submission with
//! everything else. The overlay loads the view rather than clearing it, so the
//! scene Bevy already drew survives underneath.

use std::sync::Arc;

use bevy::core_pipeline::upscaling::upscaling;
use bevy::core_pipeline::{Core2d, Core2dSystems, Core3d, Core3dSystems};
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::render_resource::TextureFormat;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::sync_component::SyncComponent;
use bevy::render::view::ViewTarget;
use bevy::render::{Extract, ExtractSchedule, RenderApp};
use mosaic_core::Size;
use mosaic_render::{Renderer, Scene};
use mosaic_render_wgpu::{TargetLoad, WgpuRenderer};

use crate::context::MosaicFrame;

/// Marks the camera whose view the Mosaic overlay paints into.
///
/// Required rather than inferred. An app with more than one camera on the same
/// window — a scene camera and an x-ray pass, say — has no obviously correct
/// default, and picking the wrong one puts the UI under geometry or on a view
/// that never reaches the screen. Put this on the camera the overlay belongs
/// in front of.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct MosaicCamera;

/// Removing the marker in the main world removes it in the render world too,
/// so a camera that stops carrying the overlay stops painting it.
impl SyncComponent for MosaicCamera {
    type Target = Self;
}

impl ExtractComponent for MosaicCamera {
    type QueryData = &'static Self;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(
        _item: bevy::ecs::query::QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self> {
        Some(MosaicCamera)
    }
}

/// The render world's copy of the last assembled frame.
///
/// A render-world resource persists across frames, so a frame in which nothing
/// changed re-paints the scene already here instead of forcing the main world
/// to re-assemble one.
#[derive(Resource, Default)]
struct RenderScene {
    scene: Option<Arc<Scene>>,
    revision: u64,
    scale: f32,
}

/// The renderer, built on Bevy's device the first time a view is painted.
///
/// It cannot be built any earlier: the pipelines bake against the view's
/// texture format, which is not known until there is a view.
#[derive(Resource, Default)]
struct MosaicRenderer {
    renderer: Option<WgpuRenderer>,
    format: Option<TextureFormat>,
}

pub(crate) struct MosaicRenderPlugin;

impl Plugin for MosaicRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<MosaicCamera>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<RenderScene>()
            .init_resource::<MosaicRenderer>()
            .add_systems(ExtractSchedule, extract_scene);

        render_app
            .add_systems(
                Core2d,
                mosaic_pass
                    .after(Core2dSystems::PostProcess)
                    .before(upscaling),
            )
            .add_systems(
                Core3d,
                mosaic_pass
                    .after(Core3dSystems::PostProcess)
                    .before(upscaling),
            );
    }
}

/// Copy the frame into the render world, cloning only when it changed.
fn extract_scene(mut target: ResMut<RenderScene>, source: Extract<Res<MosaicFrame>>) {
    if target.revision != source.revision {
        // An `Arc` clone: the command list itself was copied once, when the
        // main world took it off the tree.
        target.scene.clone_from(&source.scene);
        target.revision = source.revision;
    }
    target.scale = source.scale;
}

/// Paint the overlay into the current view.
fn mosaic_pass(
    view: ViewQuery<&'static ViewTarget, With<MosaicCamera>>,
    frame: Res<RenderScene>,
    mut state: ResMut<MosaicRenderer>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut ctx: RenderContext,
) {
    let Some(scene) = frame.scene.as_deref() else {
        return;
    };
    let target = view.into_inner();

    let format = target.main_texture_format();
    let texture = target.main_texture();
    // Extents are pixel counts, and f32 holds every integer below 2^24
    // exactly; no target is 16 million pixels on a side.
    #[allow(clippy::cast_precision_loss)]
    let size = Size::new(texture.width() as f32, texture.height() as f32);

    // The pipelines are baked against the target format, so a format change is
    // a new renderer rather than a reconfiguration.
    if state.format != Some(format) {
        state.renderer = None;
        state.format = Some(format);
    }
    let renderer = state.renderer.get_or_insert_with(|| {
        WgpuRenderer::from_device(
            device.wgpu_device().clone(),
            wgpu::Queue::clone(&queue),
            format,
            size,
        )
    });

    // `resize` is a no-op unless the extent actually moved; it rebuilds the
    // pooled layer textures when it does.
    renderer.resize(size);
    // A scale of zero would divide the viewport by nothing; before the first
    // assembled frame there is no reported scale at all.
    let scale = if frame.scale > 0.0 { frame.scale } else { 1.0 };
    renderer.set_scale_factor(f64::from(scale));

    let cmd = renderer.encode_into(scene, target.main_texture_view(), TargetLoad::Load);
    ctx.add_command_buffer(cmd);
}
