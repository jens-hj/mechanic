//! Split positional impulses. Scratch buffers cannot alias physical velocities.

use crate::device::{
    GpuPhysics, GpuPhysicsPipelines, bind_group, compute_pipeline, create_sized_buffer,
    direct_compute_pass, entry, indirect_compute_pass, indirect_dispatch_in_pass, shader_module,
    timestamp_writes,
};

#[derive(Debug)]
pub(in crate::device) struct PositionRecovery {
    pub(in crate::device) previous_poses: wgpu::Buffer,
    pub(in crate::device) sweep: super::sweep::RotationalSweep,
    capture: wgpu::ComputePipeline,
    capture_bindings: wgpu::BindGroup,
    gate: wgpu::Buffer,
    prepare: wgpu::ComputePipeline,
    prepare_bindings: wgpu::BindGroup,
    clear: wgpu::ComputePipeline,
    clear_bindings: wgpu::BindGroup,
    linear: wgpu::Buffer,
    angular: wgpu::Buffer,
    solve: wgpu::ComputePipeline,
    solve_bindings: wgpu::BindGroup,
    pub(in crate::device) geometry: wgpu::ComputePipeline,
    roots: wgpu::ComputePipeline,
    root_bindings: wgpu::BindGroup,
    coordinates: wgpu::ComputePipeline,
    coordinate_bindings: wgpu::BindGroup,
}

impl PositionRecovery {
    #[allow(clippy::too_many_lines)] // Keep the three scratch-only binding contracts together.
    pub(in crate::device) fn new(gpu: &GpuPhysics, device: &wgpu::Device) -> Self {
        let gate = create_sized_buffer(
            device,
            "mechanic recovery dispatch gate",
            (12 + gpu.pair_capacity as usize
                + gpu.bearing_count as usize
                + gpu.body_count as usize)
                * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        );
        let size = gpu.body_count.max(1) as usize * 16;
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let linear = create_sized_buffer(
            device,
            "mechanic position-only linear correction",
            size,
            usage,
        );
        let angular = create_sized_buffer(
            device,
            "mechanic position-only angular correction",
            size,
            usage,
        );
        let pipelines = GpuPhysicsPipelines::new();
        let collision = shader_module(
            &pipelines,
            device,
            "mechanic split collision kernel",
            include_str!("../../kernels/collision.wgsl"),
        );
        let previous_poses =
            create_sized_buffer(device, "mechanic terrain previous poses", size * 2, usage);
        let sweep = super::sweep::RotationalSweep::new(
            gpu,
            device,
            &pipelines,
            &collision,
            &previous_poses,
        );
        let capture = compute_pipeline(
            &pipelines,
            device,
            "mechanic capture terrain sweep poses",
            &collision,
            "capture_terrain_poses",
        );
        let capture_bindings = bind_group(
            device,
            "mechanic terrain sweep pose bindings",
            &capture,
            &[
                entry(0, &gpu.config),
                entry(1, &gpu.positions),
                entry(2, &gpu.rotations),
                entry(32, &previous_poses),
                entry(33, &sweep.fractions),
            ],
        );
        let prepare = compute_pipeline(
            &pipelines,
            device,
            "mechanic prepare recovery dispatch",
            &collision,
            "prepare_terrain_recovery",
        );
        let prepare_bindings = bind_group(
            device,
            "mechanic prepare recovery bindings",
            &prepare,
            &[
                entry(0, &gpu.config),
                entry(5, &gpu.diagnostics),
                entry(10, &gpu.collision.contacts),
                entry(16, &gpu.collision.active_contacts),
                entry(35, &gate),
                entry(27, &gpu.collision.body_components),
                entry(30, &gpu.mechanism.drive_constraints),
            ],
        );
        let solve = compute_pipeline(
            &pipelines,
            device,
            "mechanic split position solve",
            &collision,
            "solve_terrain_positions",
        );
        let geometry = compute_pipeline(
            &pipelines,
            device,
            "mechanic refresh terrain correction geometry",
            &collision,
            "update_terrain_position_geometry",
        );
        let solve_bindings = bind_group(
            device,
            "mechanic split position bindings",
            &solve,
            &[
                entry(0, &gpu.config),
                entry(2, &gpu.rotations),
                entry(3, &linear),
                entry(5, &gpu.diagnostics),
                entry(10, &gpu.collision.contacts),
                entry(36, &gate),
                entry(25, &angular),
                entry(26, &gpu.collision.world_masses),
                entry(30, &gpu.mechanism.drive_constraints),
            ],
        );
        let physics = shader_module(
            &pipelines,
            device,
            "mechanic split root kernel",
            include_str!("../../kernels/physics.wgsl"),
        );
        let clear = compute_pipeline(
            &pipelines,
            device,
            "mechanic clear position corrections",
            &physics,
            "clear_position_corrections",
        );
        let clear_bindings = bind_group(
            device,
            "mechanic clear position correction bindings",
            &clear,
            &[entry(0, &gpu.config), entry(3, &linear), entry(4, &angular)],
        );
        let roots = compute_pipeline(
            &pipelines,
            device,
            "mechanic split root correction",
            &physics,
            "apply_position_correction",
        );
        let root_bindings = bind_group(
            device,
            "mechanic split root bindings",
            &roots,
            &[
                entry(0, &gpu.config),
                entry(1, &gpu.positions),
                entry(2, &gpu.rotations),
                entry(3, &linear),
                entry(4, &angular),
                entry(5, &gpu.inverse_masses),
                entry(6, &gpu.diagnostics),
                entry(9, &gpu.mechanism.root_flags),
            ],
        );
        let articulated = shader_module(
            &pipelines,
            device,
            "mechanic split joint kernel",
            include_str!("../../kernels/articulated.wgsl"),
        );
        let coordinates = compute_pipeline(
            &pipelines,
            device,
            "mechanic split joint correction",
            &articulated,
            "correct_coordinate_positions",
        );
        let coordinate_bindings = bind_group(
            device,
            "mechanic split joint bindings",
            &coordinates,
            &[
                entry(0, &gpu.config),
                entry(1, &gpu.positions),
                entry(2, &gpu.rotations),
                entry(3, &linear),
                entry(4, &angular),
                entry(6, &gpu.bearings),
                entry(7, &gpu.mechanism.bodies),
                entry(8, &gpu.mechanism.coordinates),
                entry(12, &gpu.mechanism.drives),
            ],
        );
        Self {
            previous_poses,
            sweep,
            capture,
            capture_bindings,
            gate,
            prepare,
            prepare_bindings,
            clear,
            clear_bindings,
            linear,
            angular,
            solve,
            solve_bindings,
            geometry,
            roots,
            root_bindings,
            coordinates,
            coordinate_bindings,
        }
    }

    pub(in crate::device) fn capture(&self, gpu: &GpuPhysics, encoder: &mut wgpu::CommandEncoder) {
        direct_compute_pass(
            encoder,
            "mechanic capture terrain sweep poses",
            &self.capture,
            &self.capture_bindings,
            gpu.body_count.div_ceil(256),
            None,
        );
    }

    pub(in crate::device) fn encode(&self, gpu: &GpuPhysics, encoder: &mut wgpu::CommandEncoder) {
        if gpu.body_count <= 256
            && gpu.bearing_count <= 256
            && gpu.mechanism.active
            && gpu.mechanism.closure_count == 0
            // Keep moving roots on the established separated route. The merged
            // route has no dynamic-root terrain regression and preceded a
            // reported driven-car fall through terrain.
            && !gpu.mechanism.has_dynamic_root
        {
            self.encode_small_tree(gpu, encoder);
            return;
        }
        for iteration in 0..3 {
            if iteration != 0 {
                direct_compute_pass(
                    encoder,
                    "mechanic refresh correction mass frames",
                    &gpu.collision.update_world_masses_pipeline,
                    &gpu.collision.update_world_masses_bind_group,
                    gpu.body_count.div_ceil(256),
                    None,
                );
                indirect_compute_pass(
                    encoder,
                    "mechanic refresh terrain correction geometry",
                    &self.geometry,
                    gpu.collision
                        .terrain_geometry_bind_group
                        .as_ref()
                        .expect("terrain geometry is bound"),
                    &gpu.collision.indirect_args,
                    12,
                    None,
                );
            }
            encoder.clear_buffer(&self.linear, 0, None);
            encoder.clear_buffer(&self.angular, 0, None);
            direct_compute_pass(
                encoder,
                "mechanic prepare recovery dispatch",
                &self.prepare,
                &self.prepare_bindings,
                1,
                None,
            );
            indirect_compute_pass(
                encoder,
                "mechanic split position solve",
                &self.solve,
                &self.solve_bindings,
                &self.gate,
                12,
                None,
            );
            if gpu.mechanism.active {
                indirect_compute_pass(
                    encoder,
                    "mechanic correct joint positions",
                    &self.coordinates,
                    &self.coordinate_bindings,
                    &self.gate,
                    0,
                    None,
                );
            }
            indirect_compute_pass(
                encoder,
                "mechanic correct root positions",
                &self.roots,
                &self.root_bindings,
                &self.gate,
                0,
                None,
            );
            if gpu.mechanism.active {
                gpu.encode_terrain_timestamp(encoder, 20 + iteration * 2);
                gpu.encode_recovery_pose_projection(encoder, &self.gate);
                gpu.encode_terrain_timestamp(encoder, 21 + iteration * 2);
            }
        }
    }

    fn encode_small_tree(&self, gpu: &GpuPhysics, encoder: &mut wgpu::CommandEncoder) {
        for iteration in 0..3 {
            if iteration != 0 {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mechanic refresh small tree terrain recovery"),
                    timestamp_writes: None,
                });
                indirect_dispatch_in_pass(
                    &mut pass,
                    &gpu.collision.update_world_masses_pipeline,
                    &gpu.collision.update_world_masses_bind_group,
                    &gpu.collision.indirect_args,
                    36,
                );
                indirect_dispatch_in_pass(
                    &mut pass,
                    &self.geometry,
                    gpu.collision
                        .terrain_geometry_bind_group
                        .as_ref()
                        .expect("terrain geometry is bound"),
                    &gpu.collision.indirect_args,
                    12,
                );
            }
            direct_compute_pass(
                encoder,
                "mechanic prepare recovery dispatch",
                &self.prepare,
                &self.prepare_bindings,
                1,
                None,
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mechanic small tree terrain correction"),
                    timestamp_writes: None,
                });
                for (pipeline, bindings, offset) in [
                    (&self.clear, &self.clear_bindings, 36),
                    (&self.solve, &self.solve_bindings, 48),
                    (&self.coordinates, &self.coordinate_bindings, 48),
                    (&self.roots, &self.root_bindings, 36),
                ] {
                    indirect_dispatch_in_pass(
                        &mut pass,
                        pipeline,
                        bindings,
                        &gpu.collision.indirect_args,
                        offset,
                    );
                }
            }
            Self::encode_small_tree_pose(gpu, encoder, iteration);
        }
    }

    fn encode_small_tree_pose(
        gpu: &GpuPhysics,
        encoder: &mut wgpu::CommandEncoder,
        iteration: u32,
    ) {
        let mechanism = &gpu.mechanism;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mechanic small tree recovery pose projection"),
            timestamp_writes: timestamp_writes(
                gpu.timestamps.as_ref(),
                Some(20 + iteration * 2),
                Some(21 + iteration * 2),
            ),
        });
        if let Some(timestamps) = &gpu.timestamps {
            pass.set_pipeline(&timestamps.boundary);
            pass.set_bind_group(0, &timestamps.boundary_bindings, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        indirect_dispatch_in_pass(
            &mut pass,
            &mechanism.prepare_pipeline,
            &mechanism.prepare_bind_group,
            &gpu.collision.indirect_args,
            36,
        );
        let mut final_is_a = true;
        for _ in 0..mechanism.pointer_jump_rounds {
            let (pipeline, bindings) = if final_is_a {
                (
                    &mechanism.jump_a_to_b_pipeline,
                    &mechanism.jump_a_to_b_bind_group,
                )
            } else {
                (
                    &mechanism.jump_b_to_a_pipeline,
                    &mechanism.jump_b_to_a_bind_group,
                )
            };
            indirect_dispatch_in_pass(
                &mut pass,
                pipeline,
                bindings,
                &gpu.collision.indirect_args,
                36,
            );
            final_is_a = !final_is_a;
        }
        let (pipeline, bindings) = if mechanism.final_is_a {
            (
                &mechanism.publish_a_pipeline,
                &mechanism.publish_a_bind_group,
            )
        } else {
            (
                &mechanism.publish_b_pipeline,
                &mechanism.publish_b_bind_group,
            )
        };
        indirect_dispatch_in_pass(
            &mut pass,
            pipeline,
            bindings,
            &gpu.collision.indirect_args,
            36,
        );
    }
}
