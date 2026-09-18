//! Compute-pass encoding for the mechanism, collision, and solver stages.

use super::wgpu_util::{
    direct_compute_pass, indirect_compute_pass, indirect_dispatch_in_pass, timestamp_writes,
};
use super::{
    GpuPhysics, GpuSolverRoute, uses_fused_contact_schedule, uses_fused_velocity_schedule,
};

impl GpuPhysics {
    pub(super) fn encode_mechanism_passes(&self, encoder: &mut wgpu::CommandEncoder) {
        let mechanism = &self.mechanism;
        let workgroups = self.body_count.div_ceil(crate::abi::WORKGROUP_SIZE);
        direct_compute_pass(
            encoder,
            "mechanic prepare drive constraints",
            &mechanism.prepare_drives_pipeline,
            &mechanism.prepare_drives_bind_group,
            self.bearing_count.div_ceil(crate::abi::WORKGROUP_SIZE),
            None,
        );
        self.encode_bearing_velocity_projection(encoder, true);
        direct_compute_pass(
            encoder,
            "mechanic advance reduced coordinates",
            &mechanism.advance_coordinates_pipeline,
            &mechanism.advance_coordinates_bind_group,
            workgroups,
            None,
        );
        self.encode_mechanism_pose_projection(encoder);
        direct_compute_pass(
            encoder,
            "mechanic reconstruct body velocities",
            &mechanism.reconstruct_velocities_pipeline,
            &mechanism.reconstruct_velocities_bind_group,
            1,
            timestamp_writes(self.timestamps.as_ref(), None, Some(3)),
        );
    }

    // Metal drops timestamps on empty passes. A bit-preserving atomic access
    // orders markers against pose readers/writers, including zero-work dispatches.
    pub(super) fn encode_terrain_timestamp(&self, encoder: &mut wgpu::CommandEncoder, index: u32) {
        if let Some(timestamps) = &self.timestamps {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mechanic terrain stage boundary"),
                timestamp_writes: timestamp_writes(self.timestamps.as_ref(), Some(index), None),
            });
            pass.set_pipeline(&timestamps.boundary);
            pass.set_bind_group(0, &timestamps.boundary_bindings, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    pub(super) fn encode_recovery_pose_projection(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gate: &wgpu::Buffer,
    ) {
        self.encode_mechanism_pose_projection_gated(encoder, Some(gate));
    }

    pub(super) fn encode_mechanism_pose_projection(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_mechanism_pose_projection_gated(encoder, None);
    }

    pub(super) fn encode_mechanism_pose_projection_gated(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gate: Option<&wgpu::Buffer>,
    ) {
        let mechanism = &self.mechanism;
        let workgroups = self.body_count.div_ceil(crate::abi::WORKGROUP_SIZE);
        self.encode_mechanism_forward_kinematics(encoder, gate, 0);
        if mechanism.closure_count > 0 {
            const CLOSURE_CORRECTION_STEPS: u32 = 12;
            for step in 0..CLOSURE_CORRECTION_STEPS {
                encoder.clear_buffer(&mechanism.closure_accumulators, 0, None);
                encoder.clear_buffer(&mechanism.closure_state, 0, None);
                if step == 0
                    && let Some(gate) = gate
                {
                    indirect_compute_pass(
                        encoder,
                        "mechanic evaluate recovery closures",
                        &mechanism.evaluate_closures_pipeline,
                        &mechanism.evaluate_closures_bind_group,
                        gate,
                        32,
                        None,
                    );
                } else if step == 0 {
                    direct_compute_pass(
                        encoder,
                        "mechanic evaluate closures",
                        &mechanism.evaluate_closures_pipeline,
                        &mechanism.evaluate_closures_bind_group,
                        self.bearing_count.div_ceil(crate::abi::WORKGROUP_SIZE),
                        None,
                    );
                } else {
                    indirect_compute_pass(
                        encoder,
                        "mechanic evaluate closures",
                        &mechanism.evaluate_closures_pipeline,
                        &mechanism.evaluate_closures_bind_group,
                        &mechanism.closure_indirect_args,
                        0,
                        None,
                    );
                }
                direct_compute_pass(
                    encoder,
                    "mechanic finalize closures",
                    &mechanism.finalize_closures_pipeline,
                    &mechanism.finalize_closures_bind_group,
                    1,
                    None,
                );
                indirect_compute_pass(
                    encoder,
                    "mechanic closure Newton PCG step",
                    &mechanism.apply_closure_step_pipeline,
                    &mechanism.apply_closure_step_bind_group,
                    &mechanism.closure_indirect_args,
                    12,
                    None,
                );
                self.encode_mechanism_forward_kinematics(
                    encoder,
                    Some(&mechanism.closure_indirect_args),
                    12,
                );
            }
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
        if let Some(gate) = gate {
            indirect_compute_pass(
                encoder,
                "mechanic publish recovered mechanism poses",
                pipeline,
                bindings,
                gate,
                0,
                None,
            );
            return;
        }
        direct_compute_pass(
            encoder,
            "mechanic publish mechanism poses",
            pipeline,
            bindings,
            workgroups,
            None,
        );
    }

    pub(super) fn encode_bearing_velocity_projection(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_start: bool,
    ) {
        if uses_fused_velocity_schedule(self.bearing_count, self.body_count) {
            direct_compute_pass(
                encoder,
                "mechanic fused small bearing velocity projection",
                &self.mechanism.project_small_velocity_pipeline,
                &self.mechanism.project_small_velocity_bind_group,
                1,
                if timestamp_start {
                    timestamp_writes(self.timestamps.as_ref(), Some(2), None)
                } else {
                    None
                },
            );
            return;
        }
        for iteration in 0..self.pipeline_config.solver_iterations.max(1) {
            self.encode_bearing_velocity_projection_iteration(
                encoder,
                timestamp_start && iteration == 0,
                false,
            );
        }
    }

    pub(super) fn encode_bearing_velocity_projection_iteration(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_start: bool,
        serial: bool,
    ) {
        let mechanism = &self.mechanism;
        if serial {
            direct_compute_pass(
                encoder,
                "mechanic project bearing velocities serially",
                &mechanism.project_velocity_serial_pipeline,
                &mechanism.project_velocity_serial_bind_group,
                1,
                if timestamp_start {
                    timestamp_writes(self.timestamps.as_ref(), Some(2), None)
                } else {
                    None
                },
            );
            return;
        }
        direct_compute_pass(
            encoder,
            "mechanic project bearing velocities",
            &mechanism.project_velocity_pipeline,
            &mechanism.project_velocity_bind_group,
            self.bearing_count.div_ceil(crate::abi::WORKGROUP_SIZE),
            if timestamp_start {
                timestamp_writes(self.timestamps.as_ref(), Some(2), None)
            } else {
                None
            },
        );
        direct_compute_pass(
            encoder,
            "mechanic apply bearing velocity deltas",
            &mechanism.apply_velocity_pipeline,
            &mechanism.apply_velocity_bind_group,
            self.body_count.div_ceil(crate::abi::WORKGROUP_SIZE),
            None,
        );
    }

    pub(super) fn encode_contact_bearing_velocity_projection_iteration(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        serial: bool,
    ) {
        if serial {
            indirect_compute_pass(
                encoder,
                "mechanic project contact bearing velocities serially",
                &self.mechanism.project_velocity_serial_pipeline,
                &self.mechanism.project_velocity_serial_bind_group,
                &self.collision.indirect_args,
                48,
                None,
            );
        } else {
            self.encode_bearing_velocity_projection_iteration(encoder, false, false);
        }
    }

    pub(super) fn encode_post_contact_mechanism(&self, encoder: &mut wgpu::CommandEncoder) {
        let mechanism = &self.mechanism;
        if !mechanism.has_dynamic_root {
            self.encode_bearing_velocity_projection(encoder, false);
        }
        direct_compute_pass(
            encoder,
            "mechanic capture reduced velocities",
            &mechanism.capture_coordinates_pipeline,
            &mechanism.capture_coordinates_bind_group,
            self.body_count.div_ceil(crate::abi::WORKGROUP_SIZE),
            if mechanism.has_dynamic_root {
                timestamp_writes(self.timestamps.as_ref(), None, Some(9))
            } else {
                None
            },
        );
        if !mechanism.has_dynamic_root {
            direct_compute_pass(
                encoder,
                "mechanic reconstruct grounded post-contact velocities",
                &mechanism.reconstruct_velocities_pipeline,
                &mechanism.reconstruct_velocities_bind_group,
                1,
                timestamp_writes(self.timestamps.as_ref(), None, Some(9)),
            );
        }
    }

    pub(super) fn encode_mechanism_forward_kinematics(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        indirect: Option<&wgpu::Buffer>,
        indirect_offset: u64,
    ) {
        let mechanism = &self.mechanism;
        let workgroups = self.body_count.div_ceil(crate::abi::WORKGROUP_SIZE);
        let mut dispatch = |label: &str,
                            pipeline: &wgpu::ComputePipeline,
                            bindings: &wgpu::BindGroup,
                            timestamp_writes| {
            if let Some(indirect) = indirect {
                indirect_compute_pass(
                    encoder,
                    label,
                    pipeline,
                    bindings,
                    indirect,
                    indirect_offset,
                    timestamp_writes,
                );
            } else {
                direct_compute_pass(
                    encoder,
                    label,
                    pipeline,
                    bindings,
                    workgroups,
                    timestamp_writes,
                );
            }
        };
        dispatch(
            "mechanic prepare mechanism links",
            &mechanism.prepare_pipeline,
            &mechanism.prepare_bind_group,
            None,
        );
        let mut final_is_a = true;
        for _ in 0..mechanism.pointer_jump_rounds {
            if final_is_a {
                dispatch(
                    "mechanic mechanism jump A to B",
                    &mechanism.jump_a_to_b_pipeline,
                    &mechanism.jump_a_to_b_bind_group,
                    None,
                );
            } else {
                dispatch(
                    "mechanic mechanism jump B to A",
                    &mechanism.jump_b_to_a_pipeline,
                    &mechanism.jump_b_to_a_bind_group,
                    None,
                );
            }
            final_is_a = !final_is_a;
        }
        debug_assert_eq!(final_is_a, mechanism.final_is_a);
    }

    #[expect(clippy::too_many_lines)]
    pub(super) fn encode_collision_passes(&self, encoder: &mut wgpu::CommandEncoder) {
        let collision = &self.collision;
        let lbvh = &collision.lbvh;
        let sort_workgroups = lbvh.sort_count.div_ceil(crate::abi::WORKGROUP_SIZE);
        direct_compute_pass(
            encoder,
            "mechanic world inverse inertias",
            &collision.update_world_masses_pipeline,
            &collision.update_world_masses_bind_group,
            self.body_count.div_ceil(crate::abi::WORKGROUP_SIZE),
            timestamp_writes(self.timestamps.as_ref(), Some(4), None),
        );
        direct_compute_pass(
            encoder,
            "mechanic LBVH Morton codes",
            &lbvh.compute_morton_pipeline,
            &lbvh.compute_morton_bind_group,
            sort_workgroups,
            None,
        );
        direct_compute_pass(
            encoder,
            "mechanic LBVH local sort",
            &lbvh.sort_local_initial_pipeline,
            &lbvh.sort_local_initial_bind_group,
            sort_workgroups,
            None,
        );
        for &(parameter_offset, local_merge) in &lbvh.sort_steps {
            encoder.copy_buffer_to_buffer(
                &lbvh.sort_params_upload,
                parameter_offset,
                &lbvh.sort_params,
                0,
                16,
            );
            let (pipeline, bindings, label) = if local_merge {
                (
                    &lbvh.sort_local_merge_pipeline,
                    &lbvh.sort_local_merge_bind_group,
                    "mechanic LBVH local merge",
                )
            } else {
                (
                    &lbvh.sort_global_pipeline,
                    &lbvh.sort_global_bind_group,
                    "mechanic LBVH global merge",
                )
            };
            direct_compute_pass(encoder, label, pipeline, bindings, sort_workgroups, None);
        }
        direct_compute_pass(
            encoder,
            "mechanic LBVH topology",
            &lbvh.build_topology_pipeline,
            &lbvh.build_topology_bind_group,
            self.collider_count
                .saturating_sub(1)
                .max(1)
                .div_ceil(crate::abi::WORKGROUP_SIZE),
            None,
        );
        direct_compute_pass(
            encoder,
            "mechanic LBVH leaves",
            &lbvh.prepare_leaves_pipeline,
            &lbvh.prepare_leaves_bind_group,
            self.collider_count.div_ceil(crate::abi::WORKGROUP_SIZE),
            None,
        );
        direct_compute_pass(
            encoder,
            "mechanic LBVH bounds",
            &lbvh.build_bounds_pipeline,
            &lbvh.build_bounds_bind_group,
            self.collider_count.div_ceil(crate::abi::WORKGROUP_SIZE),
            None,
        );
        direct_compute_pass(
            encoder,
            "mechanic LBVH traversal",
            &lbvh.traverse_pipeline,
            &lbvh.traverse_bind_group,
            self.collider_count.div_ceil(crate::abi::WORKGROUP_SIZE),
            None,
        );
        direct_compute_pass(
            encoder,
            "mechanic finalize pairs",
            &lbvh.finalize_pairs_pipeline,
            &lbvh.finalize_pairs_bind_group,
            1,
            timestamp_writes(self.timestamps.as_ref(), None, Some(5)),
        );
        indirect_compute_pass(
            encoder,
            "mechanic OBB SAT",
            &collision.narrowphase_pipeline,
            &collision.narrowphase_bind_group,
            &collision.indirect_args,
            0,
            timestamp_writes(self.timestamps.as_ref(), Some(6), None),
        );
        if self.pipeline_config.ground_plane_enabled {
            direct_compute_pass(
                encoder,
                "mechanic ground contacts",
                &collision.ground_contacts_pipeline,
                &collision.ground_contacts_bind_group,
                self.collider_count.div_ceil(crate::abi::WORKGROUP_SIZE),
                None,
            );
        }
        if let Some(bindings) = &collision.terrain_bind_group
            && collision.terrain_triangle_count > 0
        {
            direct_compute_pass(
                encoder,
                "mechanic terrain BVH contacts",
                &collision.terrain_pipeline,
                bindings,
                self.collider_count.div_ceil(crate::abi::WORKGROUP_SIZE),
                timestamp_writes(self.timestamps.as_ref(), Some(14), Some(15)),
            );
        }
        direct_compute_pass(
            encoder,
            "mechanic finalize contacts",
            &collision.finalize_contacts_pipeline,
            &collision.finalize_contacts_bind_group,
            1,
            timestamp_writes(self.timestamps.as_ref(), None, Some(7)),
        );
        indirect_compute_pass(
            encoder,
            "mechanic prepare persistent contacts",
            &collision.select_active_pipeline,
            &collision.select_active_bind_group,
            &collision.indirect_args,
            12,
            timestamp_writes(self.timestamps.as_ref(), Some(8), None),
        );
        direct_compute_pass(
            encoder,
            "mechanic finalize active contacts",
            &collision.finalize_active_pipeline,
            &collision.finalize_active_bind_group,
            1,
            None,
        );
        if self.mechanism.active
            && !self.mechanism.has_dynamic_root
            && uses_fused_velocity_schedule(self.bearing_count, self.body_count)
        {
            self.encode_grounded_contact_solver_pass(encoder);
            return;
        }
        if self.solver_route() == GpuSolverRoute::General || collision.terrain_triangle_count > 0 {
            indirect_compute_pass(
                encoder,
                "mechanic count body contacts",
                &collision.count_body_contacts_pipeline,
                &collision.count_body_contacts_bind_group,
                &collision.indirect_args,
                24,
                None,
            );
        }
        indirect_compute_pass(
            encoder,
            "mechanic contact warm start",
            &collision.warm_start_pipeline,
            &collision.warm_start_bind_group,
            &collision.indirect_args,
            24,
            None,
        );
        indirect_compute_pass(
            encoder,
            "mechanic contact warm start apply",
            &collision.solve_apply_pipeline,
            &collision.solve_apply_bind_group,
            &collision.indirect_args,
            36,
            None,
        );
        let serial_mechanism = self.mechanism.active
            && self.mechanism.has_dynamic_root
            && uses_fused_contact_schedule(
                self.bearing_count,
                self.pipeline_config.ground_plane_enabled,
            );
        if serial_mechanism {
            direct_compute_pass(
                encoder,
                "mechanic fused small mechanism contacts",
                &collision.solve_small_mechanism_pipeline,
                &collision.solve_small_mechanism_bind_group,
                1,
                None,
            );
        } else if self.mechanism.active && self.mechanism.has_dynamic_root {
            self.encode_contact_bearing_velocity_projection_iteration(encoder, serial_mechanism);
        }
        let iterations = self.pipeline_config.solver_iterations.max(1);
        if !serial_mechanism {
            for _ in 1..iterations {
                indirect_compute_pass(
                    encoder,
                    "mechanic contact projection",
                    &collision.solve_accumulate_pipeline,
                    &collision.solve_accumulate_bind_group,
                    &collision.indirect_args,
                    24,
                    None,
                );
                indirect_compute_pass(
                    encoder,
                    "mechanic contact apply",
                    &collision.solve_apply_pipeline,
                    &collision.solve_apply_bind_group,
                    &collision.indirect_args,
                    36,
                    None,
                );
                if self.mechanism.active && self.mechanism.has_dynamic_root {
                    self.encode_contact_bearing_velocity_projection_iteration(encoder, false);
                }
            }
        }
        indirect_compute_pass(
            encoder,
            "mechanic persist contact manifolds",
            &collision.persist_contacts_pipeline,
            &collision.persist_contacts_bind_group,
            &collision.indirect_args,
            24,
            if self.mechanism.active {
                None
            } else {
                timestamp_writes(self.timestamps.as_ref(), None, Some(9))
            },
        );
    }

    /// Records the small grounded mechanism's contact reconciliation in one
    /// Metal compute pass. Dispatch order and iteration count match the general
    /// schedule; eliminating empty pass boundaries matters when no contacts exist.
    pub(super) fn encode_grounded_contact_solver_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let collision = &self.collision;
        let mechanism = &self.mechanism;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mechanic grounded contact solver"),
            timestamp_writes: timestamp_writes(self.timestamps.as_ref(), None, Some(9)),
        });
        indirect_dispatch_in_pass(
            &mut pass,
            &collision.count_body_contacts_pipeline,
            &collision.count_body_contacts_bind_group,
            &collision.indirect_args,
            24,
        );
        indirect_dispatch_in_pass(
            &mut pass,
            &collision.warm_start_pipeline,
            &collision.warm_start_bind_group,
            &collision.indirect_args,
            24,
        );
        indirect_dispatch_in_pass(
            &mut pass,
            &collision.solve_apply_pipeline,
            &collision.solve_apply_bind_group,
            &collision.indirect_args,
            36,
        );
        for _ in 1..self.pipeline_config.solver_iterations.max(1) {
            indirect_dispatch_in_pass(
                &mut pass,
                &collision.solve_accumulate_pipeline,
                &collision.solve_accumulate_bind_group,
                &collision.indirect_args,
                24,
            );
            indirect_dispatch_in_pass(
                &mut pass,
                &collision.solve_apply_pipeline,
                &collision.solve_apply_bind_group,
                &collision.indirect_args,
                36,
            );
        }
        indirect_dispatch_in_pass(
            &mut pass,
            &collision.persist_contacts_pipeline,
            &collision.persist_contacts_bind_group,
            &collision.indirect_args,
            24,
        );
        indirect_dispatch_in_pass(
            &mut pass,
            &mechanism.project_small_velocity_pipeline,
            &mechanism.project_small_velocity_bind_group,
            &collision.indirect_args,
            48,
        );
        indirect_dispatch_in_pass(
            &mut pass,
            &mechanism.capture_coordinates_pipeline,
            &mechanism.capture_coordinates_bind_group,
            &collision.indirect_args,
            36,
        );
        indirect_dispatch_in_pass(
            &mut pass,
            &mechanism.reconstruct_velocities_pipeline,
            &mechanism.reconstruct_velocities_bind_group,
            &collision.indirect_args,
            48,
        );
        if let Some(timestamps) = &self.timestamps {
            pass.set_pipeline(&timestamps.boundary);
            pass.set_bind_group(0, &timestamps.boundary_bindings, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }
}
