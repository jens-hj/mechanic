//! Tick submission: uniform upload, pass ordering, and queue submission.

use std::sync::atomic::Ordering;
use std::time::Instant;

use bytemuck::bytes_of;
use mechanic_core::{STANDARD_GRAVITY_M_S2_F32, TICK_SECONDS_F32};

use super::readback::{begin_async_mapping, create_async_readback_slot};
use super::wgpu_util::{direct_compute_pass, timestamp_writes, wrapping_u32};
use super::{
    ASYNC_READBACK_RING_SIZE, EXTERNAL_IMPULSE_BATCH_CAPACITY, GpuExternalImpulse, GpuImpulseError,
    GpuPhysics, GpuSolverRoute, GpuSubmissionTimings, GpuTickSubmission,
};
use crate::{BROADPHASE_HASH_CAPACITY, GpuDiagnostics, GpuTickConfig};

impl GpuPhysics {
    /// Encodes and submits one 60 Hz integration/publication pass.
    pub fn dispatch_tick(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tick_index: u64,
    ) -> GpuTickSubmission {
        self.encode_and_submit_tick(device, queue, tick_index)
    }

    /// Applies all pending impulses in ordered serial batches, then dispatches a tick.
    ///
    /// Every row is validated before the first queue write. Sets larger than the fixed
    /// staging buffer are split without dropping contacts.
    ///
    /// # Errors
    ///
    /// Returns [`GpuImpulseError`] when any pending row has an invalid body index or
    /// non-finite point/vector.
    pub fn dispatch_tick_with_impulses(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tick_index: u64,
        impulses: &[GpuExternalImpulse],
    ) -> Result<GpuTickSubmission, GpuImpulseError> {
        self.validate_impulses(impulses)?;
        for batch in impulses.chunks(EXTERNAL_IMPULSE_BATCH_CAPACITY) {
            self.apply_impulses(device, queue, batch)?;
        }
        Ok(self.encode_and_submit_tick(device, queue, tick_index))
    }

    #[expect(clippy::too_many_lines)]
    pub(super) fn encode_and_submit_tick(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tick_index: u64,
    ) -> GpuTickSubmission {
        let submission_sequence = self.submission_sequence.fetch_add(1, Ordering::Relaxed) + 1;
        let encoding_started = Instant::now();
        let snapshot_slot = u8::try_from(tick_index % 3).unwrap_or(0);
        let config = GpuTickConfig {
            body_count: self.body_count,
            tick_index: wrapping_u32(tick_index),
            snapshot_slot: u32::from(snapshot_slot),
            collider_count: self.collider_count,
            delta_seconds: TICK_SECONDS_F32,
            gravity_y: -STANDARD_GRAVITY_M_S2_F32,
            linear_damping: 0.999,
            angular_damping: 0.98,
            bearing_count: self.bearing_count,
            suppression_count: self.suppression_count,
            pair_capacity: self.pair_capacity,
            flags: u32::from(self.pipeline_config.collisions_enabled)
                | (u32::from(self.solver_route() == GpuSolverRoute::FusedSmallMechanism) << 1),
            hash_capacity: u32::try_from(BROADPHASE_HASH_CAPACITY).unwrap_or(u32::MAX),
            solver_iterations: self.pipeline_config.solver_iterations.max(1),
            sort_count: self.collision.lbvh.sort_count,
            coordinate_count: self.mechanism.coordinate_count,
        };
        queue.write_buffer(&self.config, 0, bytes_of(&config));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mechanic physics tick"),
        });
        // Error flags are a persistent GPU failure latch. Per-tick counters are
        // cleared independently, so a delayed CPU readback can never erase a
        // terminal failure reported by an earlier submission.
        encoder.clear_buffer(&self.diagnostics, 4, None);
        let run_collisions = self.pipeline_config.collisions_enabled && self.collider_count > 0;
        let run_bearings = self.bearing_count > 0;
        if run_collisions {
            encoder.clear_buffer(&self.collision.lbvh.node_parents, 0, None);
            encoder.clear_buffer(&self.collision.lbvh.node_visits, 0, None);
            encoder.clear_buffer(&self.collision.velocity_deltas, 0, None);
        }
        if self.mechanism.active {
            encoder.clear_buffer(&self.mechanism.velocity_deltas, 0, None);
        }
        if run_collisions
            && self.collision.terrain_triangle_count > 0
            && let Some(recovery) = &self.terrain_recovery
        {
            recovery.capture(self, &mut encoder);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mechanic integrate pass"),
                timestamp_writes: timestamp_writes(self.timestamps.as_ref(), Some(0), Some(1)),
            });
            pass.set_pipeline(&self.integration_pipeline);
            pass.set_bind_group(0, &self.bind_groups[usize::from(snapshot_slot)], &[]);
            pass.dispatch_workgroups(self.body_count.div_ceil(crate::abi::WORKGROUP_SIZE), 1, 1);
        }
        if self.mechanism.active {
            self.encode_mechanism_passes(&mut encoder);
        }
        if run_collisions
            && self.collision.terrain_triangle_count > 0
            && let Some(recovery) = &self.terrain_recovery
        {
            self.encode_terrain_timestamp(&mut encoder, 16);
            if self.terrain_has_free_bodies {
                recovery.sweep.encode(self, &mut encoder);
            }
            self.encode_terrain_timestamp(&mut encoder, 17);
        }
        if run_collisions {
            self.encode_collision_passes(&mut encoder);
        }
        if self.mechanism.active && run_collisions {
            self.encode_post_contact_mechanism(&mut encoder);
        }
        if run_collisions
            && self.collision.terrain_triangle_count > 0
            && let Some(recovery) = &self.terrain_recovery
        {
            self.encode_terrain_timestamp(&mut encoder, 18);
            recovery.encode(self, &mut encoder);
            self.encode_terrain_timestamp(&mut encoder, 19);
        }
        if self.mechanism.active {
            direct_compute_pass(
                &mut encoder,
                "mechanic validate articulated state",
                &self.mechanism.validate_state_pipeline,
                &self.mechanism.validate_state_bind_group,
                self.body_count.div_ceil(crate::abi::WORKGROUP_SIZE),
                None,
            );
        }
        if run_bearings {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mechanic bearing validation"),
                timestamp_writes: timestamp_writes(self.timestamps.as_ref(), Some(10), Some(11)),
            });
            pass.set_pipeline(&self.bearing_pipeline);
            pass.set_bind_group(0, &self.bearing_bind_group, &[]);
            pass.dispatch_workgroups(
                self.bearing_count.div_ceil(crate::abi::WORKGROUP_SIZE),
                1,
                1,
            );
        }
        direct_compute_pass(
            &mut encoder,
            "mechanic snapshot publication",
            &self.snapshot_pipeline,
            &self.snapshot_bind_groups[usize::from(snapshot_slot)],
            self.body_count.div_ceil(crate::abi::WORKGROUP_SIZE),
            timestamp_writes(self.timestamps.as_ref(), Some(12), Some(13)),
        );
        encoder.copy_buffer_to_buffer(
            &self.diagnostics,
            0,
            &self.diagnostics_readback,
            0,
            u64::try_from(size_of::<GpuDiagnostics>()).unwrap_or(32),
        );
        if let Some(timestamps) = &self.timestamps {
            encoder.resolve_query_set(&timestamps.query_set, 0..28, &timestamps.resolve, 0);
            encoder.copy_buffer_to_buffer(&timestamps.resolve, 0, &timestamps.readback, 0, 224);
        }
        let mut async_readbacks = self
            .async_readback_enabled
            .load(Ordering::Acquire)
            .then(|| {
                self.async_readbacks
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
            });
        let async_slot_index = async_readbacks.as_mut().and_then(|slots| {
            let index = if let Some(index) = slots.iter().position(|slot| slot.pending.is_none()) {
                index
            } else if slots.len() < ASYNC_READBACK_RING_SIZE {
                let index = slots.len();
                slots.push(create_async_readback_slot(
                    device,
                    self.body_count,
                    self.mechanism.coordinate_count,
                    self.timestamps.is_some(),
                    index,
                ));
                index
            } else {
                return None;
            };
            let slot = &slots[index];
            let diagnostics_size = u64::try_from(size_of::<GpuDiagnostics>()).unwrap_or(32);
            let snapshot_size = u64::from(self.body_count) * 16;
            encoder.copy_buffer_to_buffer(
                &self.diagnostics,
                0,
                &slot.diagnostics,
                0,
                diagnostics_size,
            );
            if let (Some(timestamps), Some(readback)) = (&self.timestamps, slot.timestamps.as_ref())
            {
                encoder.copy_buffer_to_buffer(&timestamps.resolve, 0, readback, 0, 224);
            }
            let snapshot = &self.snapshots[usize::from(snapshot_slot)];
            encoder.copy_buffer_to_buffer(
                snapshot.positions(),
                0,
                &slot.positions,
                0,
                snapshot_size,
            );
            encoder.copy_buffer_to_buffer(
                snapshot.rotations(),
                0,
                &slot.rotations,
                0,
                snapshot_size,
            );
            // Copies share the tick command encoder, before any subsequent tick can
            // mutate the source buffers. Zero-coordinate scenes retain a dummy row.
            for (source, destination, size) in [
                (
                    &self.linear_velocities,
                    &slot.linear_velocities,
                    snapshot_size,
                ),
                (
                    &self.angular_velocities,
                    &slot.angular_velocities,
                    snapshot_size,
                ),
                (
                    &self.mechanism.coordinates,
                    &slot.coordinates,
                    u64::from(self.mechanism.coordinate_count) * 8,
                ),
            ] {
                if size != 0 {
                    encoder.copy_buffer_to_buffer(source, 0, destination, 0, size);
                }
            }
            Some(index)
        });
        let encoding_ms = encoding_started.elapsed().as_secs_f64() * 1_000.0;
        let finalization_started = Instant::now();
        let command_buffer = encoder.finish();
        let finalization_ms = finalization_started.elapsed().as_secs_f64() * 1_000.0;
        let submission_started = Instant::now();
        let submission_index = queue.submit([command_buffer]);
        let submission_finished = Instant::now();
        let submission_ms = submission_finished
            .duration_since(submission_started)
            .as_secs_f64()
            * 1_000.0;
        let readback_setup_started = Instant::now();
        if let (Some(slots), Some(index)) = (&mut async_readbacks, async_slot_index) {
            begin_async_mapping(
                &mut slots[index],
                submission_sequence,
                tick_index,
                snapshot_slot,
                submission_finished,
                self.readback_timing_enabled.load(Ordering::Acquire),
            );
        }
        let readback_setup_ms = readback_setup_started.elapsed().as_secs_f64() * 1_000.0;
        GpuTickSubmission {
            submission_sequence,
            tick_index,
            snapshot_slot,
            submission_index,
            cpu_timings: GpuSubmissionTimings {
                encoding_ms,
                finalization_ms,
                submission_ms,
                readback_setup_ms,
            },
        }
    }
}
