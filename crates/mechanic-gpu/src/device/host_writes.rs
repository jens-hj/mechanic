//! CPU writes into live GPU state: impulses, body states, ground planes, and drives.

use std::sync::atomic::Ordering;

use bevy_math::Vec3;
use bytemuck::{Zeroable, bytes_of, cast_slice};
use mechanic_core::ConstructionMaterial;

use super::wgpu_util::direct_compute_pass;
use super::{
    ASYNC_READBACK_RING_SIZE, EXTERNAL_IMPULSE_BATCH_CAPACITY, GpuBodyStateError,
    GpuDriveConstraint, GpuExternalImpulse, GpuExternalImpulseBatch, GpuGroundPlane,
    GpuGroundPlaneError, GpuImpulseError, GpuPhysics, GpuPhysicsError,
};
use crate::{
    GpuBearing, GpuGroundSurface, GpuMechanismCoordinate, GpuMechanismDrive, GpuTransform,
    GpuVelocity,
};

impl GpuPhysics {
    /// Adds a world-space impulse at a world-space point on one compound body.
    ///
    /// Static bodies ignore the impulse. The submission is ordered before later
    /// fixed ticks submitted to the same queue.
    ///
    /// # Errors
    ///
    /// Returns [`GpuImpulseError`] for an invalid body row or non-finite input.
    pub fn apply_impulse(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        body_index: u32,
        world_point: Vec3,
        impulse: Vec3,
    ) -> Result<wgpu::SubmissionIndex, GpuImpulseError> {
        self.apply_impulses(
            device,
            queue,
            &[GpuExternalImpulse::new(body_index, world_point, impulse)],
        )
    }

    /// Validates and applies at most one fixed staging batch as a serial pass.
    ///
    /// Validation covers every row before the queue is modified, so invalid input
    /// can never produce a partial submission.
    ///
    /// # Errors
    ///
    /// Returns [`GpuImpulseError`] when the batch exceeds staging capacity or any
    /// row has an invalid body index or non-finite point/vector.
    pub fn apply_impulses(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        impulses: &[GpuExternalImpulse],
    ) -> Result<wgpu::SubmissionIndex, GpuImpulseError> {
        if impulses.len() > EXTERNAL_IMPULSE_BATCH_CAPACITY {
            return Err(GpuImpulseError::BatchCapacity {
                provided: impulses.len(),
                capacity: EXTERNAL_IMPULSE_BATCH_CAPACITY,
            });
        }
        self.validate_impulses(impulses)?;
        let mut batch = GpuExternalImpulseBatch::zeroed();
        batch.metadata[0] = u32::try_from(impulses.len()).unwrap_or(u32::MAX);
        batch.rows[..impulses.len()].copy_from_slice(impulses);
        queue.write_buffer(&self.external_impulse, 0, bytes_of(&batch));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mechanic external impulse"),
        });
        direct_compute_pass(
            &mut encoder,
            "mechanic apply external impulse",
            &self.external_impulse_pipeline,
            &self.external_impulse_bind_group,
            1,
            None,
        );
        Ok(queue.submit([encoder.finish()]))
    }

    pub(super) fn validate_impulses(
        &self,
        impulses: &[GpuExternalImpulse],
    ) -> Result<(), GpuImpulseError> {
        for row in impulses {
            let body_index = row.body_index();
            if body_index >= self.body_count {
                return Err(GpuImpulseError::BodyIndexOutOfRange {
                    body_index,
                    body_count: self.body_count,
                });
            }
            if !row.is_finite() {
                return Err(GpuImpulseError::NonFinite);
            }
        }
        Ok(())
    }

    /// Replaces all authoritative body transforms and velocities at a safe app-owned boundary.
    ///
    /// This is used when a live world rebuilds its static construction topology after an edit;
    /// unchanged moving compounds keep their latest pose and motion.
    ///
    /// # Errors
    ///
    /// Returns [`GpuBodyStateError`] when row counts differ or any lane is non-finite.
    pub fn write_body_states(
        &self,
        queue: &wgpu::Queue,
        transforms: &[GpuTransform],
        velocities: &[GpuVelocity],
    ) -> Result<(), GpuBodyStateError> {
        let expected = self.body_count;
        if transforms.len() != expected as usize || velocities.len() != expected as usize {
            return Err(GpuBodyStateError::BodyCount {
                provided: transforms.len().max(velocities.len()),
                expected,
            });
        }
        let finite = transforms.iter().all(|state| {
            state.position.into_iter().all(f32::is_finite)
                && state.rotation.into_iter().all(f32::is_finite)
        }) && velocities.iter().all(|state| {
            state.linear.into_iter().all(f32::is_finite)
                && state.angular.into_iter().all(f32::is_finite)
        });
        if !finite {
            return Err(GpuBodyStateError::NonFinite);
        }
        let positions = transforms
            .iter()
            .map(|state| state.position)
            .collect::<Vec<_>>();
        let rotations = transforms
            .iter()
            .map(|state| state.rotation)
            .collect::<Vec<_>>();
        let linear = velocities
            .iter()
            .map(|state| state.linear)
            .collect::<Vec<_>>();
        let angular = velocities
            .iter()
            .map(|state| state.angular)
            .collect::<Vec<_>>();
        queue.write_buffer(&self.positions, 0, cast_slice(&positions));
        queue.write_buffer(&self.rotations, 0, cast_slice(&rotations));
        queue.write_buffer(&self.linear_velocities, 0, cast_slice(&linear));
        queue.write_buffer(&self.angular_velocities, 0, cast_slice(&angular));
        Ok(())
    }

    /// Assigns the same explicit flat collision plane to every collider.
    pub fn write_ground_plane(&self, queue: &wgpu::Queue, normal: Vec3, offset: f32) {
        let length = normal.length();
        let plane = if length > 0.0 {
            GpuGroundPlane {
                normal: normal / length,
                offset: offset / length,
            }
        } else {
            GpuGroundPlane::DISABLED
        };
        let planes = vec![plane; self.collider_count as usize];
        // This method constructs exactly one finite row per uploaded collider.
        let _ = self.write_ground_planes(queue, &planes);
    }

    /// Replaces the terrain-support plane independently for every collider.
    ///
    /// Per-collider planes let a large mechanism rest on the streamed terrain
    /// beneath each of its parts without pretending the entire world is one
    /// moving infinite plane.
    ///
    /// # Errors
    ///
    /// Returns [`GpuGroundPlaneError`] when the row count differs from the
    /// uploaded collider count or a row is not finite.
    pub fn write_ground_planes(
        &self,
        queue: &wgpu::Queue,
        planes: &[GpuGroundPlane],
    ) -> Result<(), GpuGroundPlaneError> {
        if planes.len() != self.collider_count as usize {
            return Err(GpuGroundPlaneError::PlaneCount {
                provided: planes.len(),
                expected: self.collider_count,
            });
        }
        if planes
            .iter()
            .any(|plane| !plane.normal.is_finite() || !plane.offset.is_finite())
        {
            return Err(GpuGroundPlaneError::NonFinite);
        }
        let concrete = ConstructionMaterial::Concrete.properties();
        let rows = planes
            .iter()
            .map(|plane| {
                let length = plane.normal.length();
                let (normal, offset) = if length > 0.0 {
                    (plane.normal / length, plane.offset / length)
                } else {
                    (Vec3::ZERO, 0.0)
                };
                GpuGroundSurface {
                    response: [
                        concrete.static_friction,
                        concrete.dynamic_friction,
                        concrete.restitution,
                        concrete.rolling_resistance,
                    ],
                    elasticity: [
                        concrete.nominal_block_compliance(),
                        concrete.youngs_modulus_pa,
                        0.0,
                        0.0,
                    ],
                    plane: [normal.x, normal.y, normal.z, offset],
                }
            })
            .collect::<Vec<_>>();
        queue.write_buffer(&self.collision.ground_surfaces, 0, cast_slice(&rows));
        Ok(())
    }

    /// Enables non-blocking per-tick telemetry and prototype snapshot staging.
    ///
    /// This is opt-in because correctness tests and headless benchmarks use
    /// explicit synchronous sampling and should not allocate an application
    /// readback ring for unobserved warm-up ticks.
    pub fn enable_async_readback(&self) {
        self.async_readback_enabled.store(true, Ordering::Release);
    }

    /// Enables callback timestamps for subsequently submitted asynchronous ticks.
    /// Adds no polling, waits, queue work, or readback slots.
    pub fn enable_readback_timing(&self) {
        self.readback_timing_enabled.store(true, Ordering::Release);
    }

    /// Number of ticks that can be staged without waiting or allocating.
    ///
    /// The application uses this as its in-flight submission budget. Logical
    /// scheduler ticks remain in the CPU backlog until a fixed ring slot is
    /// available, preventing a slow GPU from turning into an unbounded queue.
    pub fn async_readback_slots_available(&self) -> usize {
        if !self.async_readback_enabled.load(Ordering::Acquire) {
            return usize::MAX;
        }
        let slots = self
            .async_readbacks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        slots.iter().filter(|slot| slot.pending.is_none()).count()
            + ASYNC_READBACK_RING_SIZE.saturating_sub(slots.len())
    }

    /// Replaces the drive parameters of every mechanism coordinate.
    ///
    /// This is the one write permitted while the simulation is running: it
    /// changes no topology, mass, or buffer size, so compiled row indices stay
    /// valid and a control block can be retuned without recompiling.
    ///
    /// # Errors
    ///
    /// Returns [`GpuPhysicsError::DriveStateCount`] unless one row is supplied
    /// per compiled tree bearing.
    pub fn write_mechanism_drives(
        &self,
        queue: &wgpu::Queue,
        drives: &[GpuMechanismDrive],
    ) -> Result<(), GpuPhysicsError> {
        let required = usize::try_from(self.mechanism.coordinate_count).unwrap_or(usize::MAX);
        if drives.len() != required {
            return Err(GpuPhysicsError::DriveStateCount {
                provided: drives.len(),
                required,
            });
        }
        if !drives.is_empty() {
            queue.write_buffer(&self.mechanism.drives, 0, cast_slice(drives));
            for (coordinate, drive) in drives.iter().enumerate() {
                let row = self.mechanism.drive_constraint_rows[coordinate];
                let offset = u64::from(row)
                    * u64::try_from(size_of::<GpuDriveConstraint>()).unwrap_or(u64::MAX)
                    + u64::try_from(size_of::<GpuBearing>()).unwrap_or(u64::MAX);
                queue.write_buffer(&self.mechanism.drive_constraints, offset, bytes_of(drive));
            }
        }
        Ok(())
    }

    /// Replaces the permitted bearing-coordinate state at a paused/load boundary.
    ///
    /// # Errors
    ///
    /// Returns [`GpuPhysicsError::CoordinateStateCount`] unless one row is
    /// supplied for every tree bearing in the compiled mechanism forest.
    pub fn initialize_mechanism_coordinates(
        &self,
        queue: &wgpu::Queue,
        coordinates: &[GpuMechanismCoordinate],
    ) -> Result<(), GpuPhysicsError> {
        let required = usize::try_from(self.mechanism.coordinate_count).unwrap_or(usize::MAX);
        if coordinates.len() != required {
            return Err(GpuPhysicsError::CoordinateStateCount {
                provided: coordinates.len(),
                required,
            });
        }
        if !coordinates.is_empty() {
            queue.write_buffer(&self.mechanism.coordinates, 0, cast_slice(coordinates));
        }
        Ok(())
    }
}
