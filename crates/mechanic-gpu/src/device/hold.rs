use std::collections::BTreeMap;

use bytemuck::{Zeroable, bytes_of};
use thiserror::Error;

use super::{GpuBearing, GpuDriveConstraint, GpuMass, GpuPhysics, GpuTransform};

const SUSPENDED_BEARING: u32 = 2;

/// Invalid component hold or prescribed pose update.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum GpuHoldError {
    /// A hold mask or pose array must describe every body in this scene.
    #[error("hold state requires {expected} bodies, received {provided}")]
    BodyCount {
        /// Scene body count.
        expected: usize,
        /// Supplied row count.
        provided: usize,
    },
    /// All bodies joined by bearings must enter and leave hold together.
    #[error("component {0} cannot be partially held")]
    PartialComponent(u32),
    /// A prescribed position or quaternion is invalid.
    #[error("held pose {0} must be finite with a unit quaternion")]
    InvalidPose(usize),
}

#[derive(Debug)]
pub(super) struct HoldResources {
    masses: Vec<GpuMass>,
    bearings: Vec<GpuBearing>,
    components: Vec<u32>,
    held: Vec<bool>,
}

impl HoldResources {
    pub(super) fn new(
        masses: Vec<GpuMass>,
        bearings: Vec<GpuBearing>,
        components: Vec<u32>,
    ) -> Self {
        let held = vec![false; masses.len()];
        Self {
            masses,
            bearings,
            components,
            held,
        }
    }

    fn validate(&self, held: &[bool]) -> Result<(), GpuHoldError> {
        if held.len() != self.held.len() {
            return Err(GpuHoldError::BodyCount {
                expected: self.held.len(),
                provided: held.len(),
            });
        }
        let mut components = BTreeMap::new();
        for (&component, &holding) in self.components.iter().zip(held) {
            if components
                .insert(component, holding)
                .is_some_and(|prior| prior != holding)
            {
                return Err(GpuHoldError::PartialComponent(component));
            }
        }
        Ok(())
    }
}

impl GpuPhysics {
    /// Holds complete articulated components as collidable prescribed bodies.
    ///
    /// Changes are ordered on the supplied queue before subsequent ticks. Both
    /// entering hold and releasing discard body velocities and joint coordinates.
    /// Before release, callers must prescribe a valid default joint pose. Other
    /// components retain their state. Repeating an unchanged mask does no work.
    ///
    /// # Errors
    /// Returns [`GpuHoldError`] for a wrong row count or partial component mask.
    pub fn set_body_holds(&self, queue: &wgpu::Queue, held: &[bool]) -> Result<(), GpuHoldError> {
        let mut state = self
            .holds
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.validate(held)?;
        for (body, (&holding, &previous)) in held.iter().zip(&state.held).enumerate() {
            if holding == previous {
                continue;
            }
            let mass = if holding {
                GpuMass::zeroed()
            } else {
                state.masses[body]
            };
            queue.write_buffer(
                &self.masses,
                (body * size_of::<GpuMass>()) as u64,
                bytes_of(&mass),
            );
            queue.write_buffer(
                &self.inverse_masses,
                (body * 4) as u64,
                bytes_of(&mass.inverse_mass[0]),
            );
            let mut topology = self.mechanism.body_rows[body];
            if holding {
                // Independent roots bypass forward kinematics and coordinate
                // reconstruction while preserving all collider identities.
                topology.metadata = [u32::try_from(body).unwrap_or(u32::MAX), u32::MAX, 0, 1];
            }
            queue.write_buffer(
                &self.mechanism.bodies,
                (body * size_of::<super::GpuMechanismBody>()) as u64,
                bytes_of(&topology),
            );
            queue.write_buffer(
                &self.linear_velocities,
                (body * 16) as u64,
                bytes_of(&[0.0_f32; 4]),
            );
            queue.write_buffer(
                &self.angular_velocities,
                (body * 16) as u64,
                bytes_of(&[0.0_f32; 4]),
            );
        }
        for (index, original) in state.bearings.iter().enumerate() {
            let body = original.metadata[0] as usize;
            if held[body] == state.held[body] {
                continue;
            }
            let mut bearing = *original;
            if held[body] {
                bearing.metadata[3] |= SUSPENDED_BEARING;
            }
            queue.write_buffer(
                &self.bearings,
                (index * size_of::<GpuBearing>()) as u64,
                bytes_of(&bearing),
            );
            // Preserve the independently updated drive configuration and state.
            queue.write_buffer(
                &self.mechanism.drive_constraints,
                (index * size_of::<GpuDriveConstraint>()) as u64,
                bytes_of(&bearing),
            );
            if bearing.metadata[2] != u32::MAX {
                queue.write_buffer(
                    &self.mechanism.coordinates,
                    u64::from(bearing.metadata[2]) * 8,
                    bytes_of(&[0.0_f32; 2]),
                );
            }
        }
        state.held.copy_from_slice(held);
        Ok(())
    }

    /// Updates only held bodies, leaving the live state of other bodies alone.
    ///
    /// The pose array uses compiled body order. Its held rows must be finite and
    /// have unit quaternions. Validation precedes all writes.
    ///
    /// # Errors
    /// Returns [`GpuHoldError`] for an invalid row count or held pose.
    pub fn prescribe_held_poses(
        &self,
        queue: &wgpu::Queue,
        poses: &[GpuTransform],
    ) -> Result<(), GpuHoldError> {
        let state = self
            .holds
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if poses.len() != state.held.len() {
            return Err(GpuHoldError::BodyCount {
                expected: state.held.len(),
                provided: poses.len(),
            });
        }
        for (body, (pose, &holding)) in poses.iter().zip(&state.held).enumerate() {
            if holding
                && (!pose
                    .position
                    .iter()
                    .chain(&pose.rotation)
                    .all(|v| v.is_finite())
                    || (pose.rotation.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() > 1.0e-3)
            {
                return Err(GpuHoldError::InvalidPose(body));
            }
        }
        for (body, (pose, &holding)) in poses.iter().zip(&state.held).enumerate() {
            if holding {
                queue.write_buffer(
                    &self.positions,
                    (body * 16) as u64,
                    bytes_of(&pose.position),
                );
                queue.write_buffer(
                    &self.rotations,
                    (body * 16) as u64,
                    bytes_of(&pose.rotation),
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hold_masks_require_complete_components_and_exact_body_counts() {
        let state = HoldResources::new(vec![GpuMass::zeroed(); 3], vec![], vec![4, 4, 9]);
        assert_eq!(state.validate(&[true, true, false]), Ok(()));
        assert_eq!(state.validate(&[false, false, true]), Ok(()));
        assert_eq!(
            state.validate(&[true, false, false]),
            Err(GpuHoldError::PartialComponent(4))
        );
        assert_eq!(
            state.validate(&[true, true]),
            Err(GpuHoldError::BodyCount {
                expected: 3,
                provided: 2
            })
        );
        assert_eq!(state.held, vec![false; 3]);
    }
}
