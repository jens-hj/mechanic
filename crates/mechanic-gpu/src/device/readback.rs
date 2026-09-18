//! Synchronous and asynchronous tick readback.

use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Instant;

use bytemuck::{Pod, cast_slice};

use super::wgpu_util::{diagnostic_units, timestamp_milliseconds};
use super::{
    AsyncReadbackSlot, GpuCompletedTickReadback, GpuExecutionEvidence, GpuKernelTimings,
    GpuPhysics, GpuReadbackError, GpuTickReadback, PendingAsyncReadback,
};
use crate::{GpuDiagnostics, GpuMechanismCoordinate, GpuTransform, GpuVelocity};

impl GpuPhysics {
    /// Polls asynchronous telemetry and prototype-render staging without
    /// waiting for queue completion.
    ///
    /// Completed ticks are returned in submission order. `None` means the GPU
    /// has not completed another staged tick yet.
    ///
    /// # Errors
    ///
    /// Returns [`GpuReadbackError`] if device polling or a completed mapping
    /// failed. The caller should latch that as a terminal simulation failure.
    pub fn poll_tick_readback(
        &self,
        device: &wgpu::Device,
    ) -> Result<Option<GpuCompletedTickReadback>, GpuReadbackError> {
        let poll_started = self
            .readback_timing_enabled
            .load(Ordering::Acquire)
            .then(Instant::now);
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| GpuReadbackError::DevicePoll(error.to_string()))?;
        let mut slots = self
            .async_readbacks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        for slot in &mut *slots {
            let Some(pending) = &mut slot.pending else {
                continue;
            };
            while pending.remaining_callbacks > 0 {
                match pending.receiver.try_recv() {
                    Ok((Ok(()), observed_at)) => {
                        pending.remaining_callbacks -= 1;
                        pending.callbacks_completed_at =
                            pending.callbacks_completed_at.max(observed_at);
                    }
                    Ok((Err(error), _)) => {
                        unmap_async_slot(slot);
                        slot.pending = None;
                        return Err(GpuReadbackError::BufferMap(error));
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        unmap_async_slot(slot);
                        slot.pending = None;
                        return Err(GpuReadbackError::CallbackLost);
                    }
                }
            }
        }

        let Some(slot_index) =
            oldest_completed_readback(slots.iter().enumerate().filter_map(|(index, slot)| {
                let pending = slot.pending.as_ref()?;
                Some((
                    index,
                    pending.submission_sequence,
                    pending.remaining_callbacks,
                ))
            }))
        else {
            return Ok(None);
        };
        let slot = &mut slots[slot_index];
        let Some(pending) = slot.pending.take() else {
            return Ok(None);
        };
        let diagnostics = {
            let bytes = slot
                .diagnostics
                .get_mapped_range(0..u64::try_from(size_of::<GpuDiagnostics>()).unwrap_or(32));
            bytemuck::pod_read_unaligned::<GpuDiagnostics>(&bytes)
        };
        let timestamp_values = slot.timestamps.as_ref().map(|buffer| {
            bytemuck::pod_read_unaligned::<[u64; 28]>(&buffer.get_mapped_range(0..224))
        });
        let positions = mapped_rows::<[f32; 4]>(&slot.positions, self.body_count);
        let rotations = mapped_rows::<[f32; 4]>(&slot.rotations, self.body_count);
        let linear = mapped_rows::<[f32; 4]>(&slot.linear_velocities, self.body_count);
        let angular = mapped_rows::<[f32; 4]>(&slot.angular_velocities, self.body_count);
        let coordinates = mapped_rows::<GpuMechanismCoordinate>(
            &slot.coordinates,
            self.mechanism.coordinate_count,
        );
        unmap_async_slot(slot);

        Ok(Some(GpuCompletedTickReadback {
            submission_sequence: pending.submission_sequence,
            tick_index: pending.tick_index,
            snapshot_slot: pending.snapshot_slot,
            submission_to_readback_ms: pending.submitted_at.elapsed().as_secs_f64() * 1_000.0,
            callbacks_completed_at: pending.callbacks_completed_at,
            submission_to_callbacks_ms: pending
                .callbacks_completed_at
                .map(|at| at.duration_since(pending.submitted_at).as_secs_f64() * 1_000.0),
            callbacks_during_poll: pending
                .callbacks_completed_at
                .zip(poll_started)
                .map(|(at, start)| at >= start),
            diagnostics: self.decode_tick_readback(diagnostics, timestamp_values),
            velocities: linear
                .into_iter()
                .zip(angular)
                .map(|(linear, angular)| GpuVelocity { linear, angular })
                .collect(),
            coordinates,
            transforms: positions
                .into_iter()
                .zip(rotations)
                .map(|(position, rotation)| GpuTransform { position, rotation })
                .collect(),
        }))
    }

    pub(super) fn decode_tick_readback(
        &self,
        diagnostics: GpuDiagnostics,
        timestamp_values: Option<[u64; 28]>,
    ) -> GpuTickReadback {
        let timestamp_readback =
            timestamp_values
                .zip(self.timestamps.as_ref())
                .map(|(values, timestamps)| {
                    let elapsed = |start, end| {
                        timestamp_milliseconds(
                            values[start],
                            values[end],
                            timestamps.period_nanoseconds,
                        )
                    };
                    let timings = GpuKernelTimings {
                        integration_ms: elapsed(0, 1),
                        terrain_traversal_ms: if self.pipeline_config.collisions_enabled
                            && self.collision.terrain_triangle_count > 0
                        {
                            elapsed(14, 15)
                        } else {
                            0.0
                        },
                        rotational_sweep_ms: if self.pipeline_config.collisions_enabled
                            && self.collision.terrain_triangle_count > 0
                        {
                            elapsed(16, 17)
                        } else {
                            0.0
                        },
                        terrain_recovery_ms: if self.pipeline_config.collisions_enabled
                            && self.collision.terrain_triangle_count > 0
                        {
                            elapsed(18, 19)
                        } else {
                            0.0
                        },
                        recovery_projection_ms: if self.pipeline_config.collisions_enabled
                            && self.collision.terrain_triangle_count > 0
                            && self.mechanism.active
                        {
                            elapsed(20, 21) + elapsed(22, 23) + elapsed(24, 25)
                        } else {
                            0.0
                        },
                        mechanism_ms: if self.mechanism.active {
                            elapsed(2, 3)
                        } else {
                            0.0
                        },
                        broadphase_ms: if self.pipeline_config.collisions_enabled {
                            elapsed(4, 5)
                        } else {
                            0.0
                        },
                        narrowphase_ms: if self.pipeline_config.collisions_enabled {
                            elapsed(6, 7)
                        } else {
                            0.0
                        },
                        contact_solver_ms: if self.pipeline_config.collisions_enabled {
                            elapsed(8, 9)
                        } else {
                            0.0
                        },
                        bearings_ms: if self.bearing_count > 0 {
                            elapsed(10, 11)
                        } else {
                            0.0
                        },
                        snapshot_ms: elapsed(12, 13),
                    };
                    let total = timings.rotational_sweep_ms
                        + timings.terrain_recovery_ms
                        + timings.integration_ms
                        + timings.mechanism_ms
                        + timings.broadphase_ms
                        + timings.narrowphase_ms
                        + timings.contact_solver_ms
                        + timings.bearings_ms
                        + timings.snapshot_ms;
                    (total, timings)
                });
        GpuTickReadback {
            execution: GpuExecutionEvidence {
                stage_mask: diagnostics.executed_stage_mask,
                integrated_bodies: diagnostics.integrated_bodies,
                published_bodies: diagnostics.published_bodies,
                validated_bearings: diagnostics.validated_bearings,
            },
            gpu_tick_ms: timestamp_readback.map(|(total, _)| total),
            kernel_timings: timestamp_readback.map(|(_, timings)| timings),
            error_flags: diagnostics.error_flags,
            pair_count: diagnostics.pair_count,
            contact_count: diagnostics.contact_count,
            active_contact_count: diagnostics.active_contact_count,
            planned_solver_sweeps: diagnostics.planned_solver_sweeps,
            executed_solver_sweeps: diagnostics.executed_solver_sweeps,
            anchor_residual_meters: diagnostic_units(diagnostics.max_anchor_micrometers),
            axis_residual_degrees: diagnostic_units(diagnostics.max_axis_microdegrees),
        }
    }

    /// Reads only stage timestamps and fixed-size diagnostics after a completed
    /// submission. Authoritative body state remains GPU-resident.
    ///
    /// # Errors
    ///
    /// Returns [`GpuReadbackError`] if device polling or either fixed-size map fails.
    pub fn read_last_tick(
        &self,
        device: &wgpu::Device,
    ) -> Result<GpuTickReadback, GpuReadbackError> {
        map_for_read(device, &self.diagnostics_readback)?;
        let diagnostics = {
            let bytes = self
                .diagnostics_readback
                .get_mapped_range(0..u64::try_from(size_of::<GpuDiagnostics>()).unwrap_or(32));
            bytemuck::pod_read_unaligned::<GpuDiagnostics>(&bytes)
        };
        self.diagnostics_readback.unmap();

        let timestamp_values = self
            .timestamps
            .as_ref()
            .map(|timestamps| {
                map_for_read(device, &timestamps.readback)?;
                let values = bytemuck::pod_read_unaligned::<[u64; 28]>(
                    &timestamps.readback.get_mapped_range(0..224),
                );
                timestamps.readback.unmap();
                Ok::<_, GpuReadbackError>(values)
            })
            .transpose()?;
        Ok(self.decode_tick_readback(diagnostics, timestamp_values))
    }

    /// Copies one published snapshot to CPU memory for prototype renderers.
    ///
    /// Production rendering should consume [`SnapshotBuffers`](crate::SnapshotBuffers) directly and avoid
    /// this synchronous readback.
    ///
    /// # Errors
    ///
    /// Returns [`GpuReadbackError`] when `slot` is invalid or GPU mapping fails.
    pub fn read_snapshot_transforms(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: u8,
    ) -> Result<Vec<GpuTransform>, GpuReadbackError> {
        let snapshot = self
            .snapshot(slot)
            .ok_or(GpuReadbackError::InvalidSnapshotSlot(slot))?;
        let byte_len = u64::from(self.body_count) * 16;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mechanic snapshot readback"),
        });
        encoder.copy_buffer_to_buffer(
            snapshot.positions(),
            0,
            &self.snapshot_positions_readback,
            0,
            byte_len,
        );
        encoder.copy_buffer_to_buffer(
            snapshot.rotations(),
            0,
            &self.snapshot_rotations_readback,
            0,
            byte_len,
        );
        queue.submit([encoder.finish()]);

        let positions = read_vec4_buffer(device, &self.snapshot_positions_readback, byte_len)?;
        let rotations = read_vec4_buffer(device, &self.snapshot_rotations_readback, byte_len)?;
        Ok(positions
            .into_iter()
            .zip(rotations)
            .map(|(position, rotation)| GpuTransform { position, rotation })
            .collect())
    }
}

pub(super) fn map_for_read(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
) -> Result<(), GpuReadbackError> {
    let (sender, receiver) = mpsc::sync_channel(1);
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| GpuReadbackError::DevicePoll(error.to_string()))?;
    receiver
        .recv()
        .map_err(|_| GpuReadbackError::CallbackLost)?
        .map_err(|error| GpuReadbackError::BufferMap(error.to_string()))
}

pub(super) fn create_async_readback_slot(
    device: &wgpu::Device,
    body_count: u32,
    coordinate_count: u32,
    timestamps_enabled: bool,
    index: usize,
) -> AsyncReadbackSlot {
    let readback_buffer = |label: String, size: u64| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&label),
            size: size.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    };
    let snapshot_size = u64::from(body_count) * 16;
    AsyncReadbackSlot {
        diagnostics: readback_buffer(
            format!("mechanic async diagnostics {index}"),
            u64::try_from(size_of::<GpuDiagnostics>()).unwrap_or(32),
        ),
        timestamps: timestamps_enabled
            .then(|| readback_buffer(format!("mechanic async timestamps {index}"), 224)),
        positions: readback_buffer(
            format!("mechanic async snapshot positions {index}"),
            snapshot_size,
        ),
        rotations: readback_buffer(
            format!("mechanic async snapshot rotations {index}"),
            snapshot_size,
        ),
        linear_velocities: readback_buffer(
            format!("mechanic async linear velocities {index}"),
            snapshot_size,
        ),
        angular_velocities: readback_buffer(
            format!("mechanic async angular velocities {index}"),
            snapshot_size,
        ),
        coordinates: readback_buffer(
            format!("mechanic async coordinates {index}"),
            u64::from(coordinate_count) * 8,
        ),
        pending: None,
    }
}

pub(super) fn mapped_rows<T: Pod>(buffer: &wgpu::Buffer, count: u32) -> Vec<T> {
    if count == 0 {
        return Vec::new();
    }
    let bytes = buffer.get_mapped_range(..u64::from(count) * size_of::<T>() as u64);
    bytes
        .chunks_exact(size_of::<T>())
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

// A later mapping callback must not overtake an earlier incomplete snapshot.
pub(super) fn oldest_completed_readback(
    pending: impl Iterator<Item = (usize, u64, u8)>,
) -> Option<usize> {
    pending
        .min_by_key(|&(_, sequence, _)| sequence)
        .and_then(|(index, _, remaining)| (remaining == 0).then_some(index))
}

pub(super) fn begin_async_mapping(
    slot: &mut AsyncReadbackSlot,
    submission_sequence: u64,
    tick_index: u64,
    snapshot_slot: u8,
    submitted_at: Instant,
    timing: bool,
) {
    let callback_count = 6 + u8::from(slot.timestamps.is_some());
    let (sender, receiver) = mpsc::sync_channel(usize::from(callback_count));
    let map = |buffer: &wgpu::Buffer,
               sender: mpsc::SyncSender<(Result<(), String>, Option<Instant>)>| {
        buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            let observed_at = timing.then(Instant::now);
            let _ = sender.send((result.map_err(|error| error.to_string()), observed_at));
        });
    };
    map(&slot.diagnostics, sender.clone());
    if let Some(timestamps) = &slot.timestamps {
        map(timestamps, sender.clone());
    }
    map(&slot.positions, sender.clone());
    map(&slot.rotations, sender.clone());
    map(&slot.linear_velocities, sender.clone());
    map(&slot.angular_velocities, sender.clone());
    map(&slot.coordinates, sender);
    slot.pending = Some(PendingAsyncReadback {
        submission_sequence,
        tick_index,
        snapshot_slot,
        receiver,
        remaining_callbacks: callback_count,
        submitted_at,
        callbacks_completed_at: None,
    });
}

pub(super) fn unmap_async_slot(slot: &AsyncReadbackSlot) {
    slot.diagnostics.unmap();
    if let Some(timestamps) = &slot.timestamps {
        timestamps.unmap();
    }
    slot.positions.unmap();
    slot.rotations.unmap();
    slot.linear_velocities.unmap();
    slot.angular_velocities.unmap();
    slot.coordinates.unmap();
}

pub(super) fn read_vec4_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    byte_len: u64,
) -> Result<Vec<[f32; 4]>, GpuReadbackError> {
    map_for_read(device, buffer)?;
    let values = {
        let bytes = buffer.get_mapped_range(0..byte_len);
        cast_slice::<u8, [f32; 4]>(&bytes).to_vec()
    };
    buffer.unmap();
    Ok(values)
}
