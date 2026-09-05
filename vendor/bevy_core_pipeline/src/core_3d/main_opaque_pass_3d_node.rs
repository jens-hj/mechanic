use crate::{
    core_3d::Opaque3d,
    skybox::{SkyboxBindGroup, SkyboxPipelineId},
};
use bevy_camera::{MainPassResolutionOverride, Viewport};
use bevy_ecs::prelude::*;
use bevy_log::error;
#[cfg(feature = "trace")]
use bevy_log::info_span;
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    render_phase::ViewBinnedRenderPhases,
    render_resource::{PipelineCache, RenderPassDescriptor, StoreOp},
    renderer::{RenderContext, ViewQuery},
    view::{ExtractedView, ViewDepthTexture, ViewTarget, ViewUniformOffset},
};

use super::{AlphaMask3d, Opaque3dBatchSetKey};

/// Opt-in diagnostic partition of opaque draws into actual pass boundaries.
/// Runs preserve draw order and attachment contents, but add store/load overhead.
#[derive(Resource)]
pub struct OpaquePassPartition(
    pub  std::sync::Arc<
        dyn Fn(&World, Entity, &Opaque3dBatchSetKey, &PipelineCache) -> Option<&'static str>
            + Send
            + Sync,
    >,
);

fn contiguous_runs(labels: &[&'static str]) -> Vec<(core::ops::Range<usize>, &'static str)> {
    let mut runs = Vec::new();
    let mut start = 0;
    while start < labels.len() {
        let label = labels[start];
        let mut end = start + 1;
        while end < labels.len() && labels[end] == label {
            end += 1;
        }
        runs.push((start..end, label));
        start = end;
    }
    runs
}

pub fn main_opaque_pass_3d(
    world: &World,
    view: ViewQuery<(
        &ExtractedCamera,
        &ExtractedView,
        &ViewTarget,
        &ViewDepthTexture,
        Option<&SkyboxPipelineId>,
        Option<&SkyboxBindGroup>,
        &ViewUniformOffset,
        Option<&MainPassResolutionOverride>,
    )>,
    opaque_phases: Res<ViewBinnedRenderPhases<Opaque3d>>,
    alpha_mask_phases: Res<ViewBinnedRenderPhases<AlphaMask3d>>,
    pipeline_cache: Res<PipelineCache>,
    partition: Option<Res<OpaquePassPartition>>,
    mut ctx: RenderContext,
) {
    let view_entity = view.entity();

    let (
        camera,
        extracted_view,
        target,
        depth,
        skybox_pipeline,
        skybox_bind_group,
        view_uniform_offset,
        resolution_override,
    ) = view.into_inner();

    let (Some(opaque_phase), Some(alpha_mask_phase)) = (
        opaque_phases.get(&extracted_view.retained_view_entity),
        alpha_mask_phases.get(&extracted_view.retained_view_entity),
    ) else {
        return;
    };

    #[cfg(feature = "trace")]
    let _main_opaque_pass_3d_span = info_span!("main_opaque_pass_3d").entered();

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    let labels = partition
        .as_ref()
        .map(|partition| {
            opaque_phase
                .draw_batch_keys()
                .into_iter()
                .map(|key| (partition.0)(world, view_entity, key, &pipeline_cache))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let split = labels.iter().any(Option::is_some);
    if split {
        let labels: Vec<_> = labels
            .into_iter()
            .map(|label| label.unwrap_or("main_opaque_other_3d"))
            .collect();
        for (range, label) in contiguous_runs(&labels) {
            let color_attachments = [Some(target.get_color_attachment())];
            let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some(label),
                color_attachments: &color_attachments,
                depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some(viewport) =
                Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
            {
                pass.set_camera_viewport(&viewport);
            }
            let mut index = 0;
            if let Err(err) =
                opaque_phase.render_filtered(&mut pass, world, view_entity, &mut |_| {
                    let selected = range.contains(&index);
                    index += 1;
                    selected
                })
            {
                error!("Error rendering diagnostic opaque partition {err:?}");
            }
            debug_assert_eq!(index, labels.len(), "opaque partition traversal changed");
        }
    }

    // Avoid an empty trailing pass: Metal may not produce an end timestamp for it.
    if split
        && alpha_mask_phase.is_empty()
        && !skybox_pipeline
            .zip(skybox_bind_group)
            .is_some_and(|(pipeline, _)| pipeline_cache.get_render_pipeline(pipeline.0).is_some())
    {
        return;
    }

    let color_attachments = [Some(target.get_color_attachment())];
    let depth_stencil_attachment = Some(depth.get_attachment(StoreOp::Store));

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some(if split {
            "main_opaque_other_3d"
        } else {
            "main_opaque_pass_3d"
        }),
        color_attachments: &color_attachments,
        depth_stencil_attachment,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let pass_span = diagnostics.pass_span(&mut render_pass, "main_opaque_pass_3d");

    if let Some(viewport) =
        Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
    {
        render_pass.set_camera_viewport(&viewport);
    }

    if !split && !opaque_phase.is_empty() {
        #[cfg(feature = "trace")]
        let _opaque_main_pass_3d_span = info_span!("opaque_main_pass_3d").entered();
        if let Err(err) = opaque_phase.render(&mut render_pass, world, view_entity) {
            error!("Error encountered while rendering the opaque phase {err:?}");
        }
    }

    if !alpha_mask_phase.is_empty() {
        #[cfg(feature = "trace")]
        let _alpha_mask_main_pass_3d_span = info_span!("alpha_mask_main_pass_3d").entered();
        if let Err(err) = alpha_mask_phase.render(&mut render_pass, world, view_entity) {
            error!("Error encountered while rendering the alpha mask phase {err:?}");
        }
    }

    if let (Some(skybox_pipeline), Some(SkyboxBindGroup(skybox_bind_group))) =
        (skybox_pipeline, skybox_bind_group)
        && let Some(pipeline) = pipeline_cache.get_render_pipeline(skybox_pipeline.0)
    {
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(
            0,
            &skybox_bind_group.0,
            &[view_uniform_offset.offset, skybox_bind_group.1],
        );
        render_pass.draw(0..3, 0..1);
    }

    pass_span.end(&mut render_pass);
}

#[cfg(test)]
mod partition_tests {
    #[test]
    fn contiguous_partitions_preserve_interleaved_draw_order() {
        assert_eq!(
            super::contiguous_runs(&["terrain", "terrain", "other", "terrain"]),
            vec![(0..2, "terrain"), (2..3, "other"), (3..4, "terrain")]
        );
        assert!(super::contiguous_runs(&[]).is_empty());
    }
}
