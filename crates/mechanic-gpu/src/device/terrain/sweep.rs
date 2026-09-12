//! Conservative pose clamping for unjointed bodies; physical velocities stay live.

use crate::device::{
    GpuPhysics, GpuPhysicsPipelines, bind_group, compute_pipeline, create_sized_buffer,
    direct_compute_pass, entry,
};

#[derive(Debug)]
pub(in crate::device) struct RotationalSweep {
    pub(super) fractions: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    pub(super) bindings: Option<wgpu::BindGroup>,
    apply: wgpu::ComputePipeline,
    apply_bindings: wgpu::BindGroup,
}

impl RotationalSweep {
    pub(super) fn new(
        gpu: &GpuPhysics,
        device: &wgpu::Device,
        pipelines: &GpuPhysicsPipelines,
        shader: &wgpu::ShaderModule,
        previous: &wgpu::Buffer,
    ) -> Self {
        let fractions = create_sized_buffer(
            device,
            "mechanic terrain sweep fractions",
            gpu.body_count.max(1) as usize * 4,
            wgpu::BufferUsages::STORAGE,
        );
        let pipeline = compute_pipeline(
            pipelines,
            device,
            "mechanic rotational terrain sweep",
            shader,
            "sweep_rotating_terrain_colliders",
        );
        let apply = compute_pipeline(
            pipelines,
            device,
            "mechanic apply rotational terrain sweep",
            shader,
            "apply_terrain_sweep",
        );
        let apply_bindings = bind_group(
            device,
            "mechanic apply terrain sweep bindings",
            &apply,
            &[
                entry(0, &gpu.config),
                entry(1, &gpu.positions),
                entry(2, &gpu.rotations),
                entry(32, previous),
                entry(33, &fractions),
            ],
        );
        Self {
            fractions,
            pipeline,
            bindings: None,
            apply,
            apply_bindings,
        }
    }

    pub(super) fn bind(
        &self,
        gpu: &GpuPhysics,
        device: &wgpu::Device,
        terrain: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        bind_group(
            device,
            "mechanic rotational terrain bindings",
            &self.pipeline,
            &[
                entry(0, &gpu.config),
                entry(1, &gpu.positions),
                entry(2, &gpu.rotations),
                entry(6, &gpu.colliders),
                entry(28, &gpu.convex_shapes),
                entry(31, terrain),
                entry(
                    32,
                    &gpu.terrain_recovery
                        .as_ref()
                        .expect("terrain recovery initialized")
                        .previous_poses,
                ),
                entry(33, &self.fractions),
                entry(34, &gpu.terrain_free_bodies),
            ],
        )
    }

    pub(in crate::device) fn encode(&self, gpu: &GpuPhysics, encoder: &mut wgpu::CommandEncoder) {
        direct_compute_pass(
            encoder,
            "mechanic sweep rotating terrain colliders",
            &self.pipeline,
            self.bindings.as_ref().expect("terrain sweep is bound"),
            gpu.collider_count.div_ceil(256),
            None,
        );
        direct_compute_pass(
            encoder,
            "mechanic clamp terrain sweep poses",
            &self.apply,
            &self.apply_bindings,
            gpu.body_count.div_ceil(256),
            None,
        );
    }
}
