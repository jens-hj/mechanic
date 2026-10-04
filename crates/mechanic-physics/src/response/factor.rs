//! Factorizations of the effective dynamics matrix.

use super::articulated;
use crate::PhysicsError;

/// Explicit numerical factor selection for matched CPU experiments.
/// Both choices solve the same effective dynamics; arithmetic ordering differs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DynamicsFactorization {
    /// Frozen dense reference. Kept authoritative until complete tick gates pass.
    #[default]
    DenseReference,
    /// Articulated tree factor with component-local impulse responses.
    Articulated,
}

impl DynamicsFactorization {
    pub(crate) fn factor(
        self,
        creation: &mechanic_core::CompiledCreation,
        model: &crate::MachineDynamics,
        coordinates: &[f64],
        diagonal: &[f64],
    ) -> Result<DynamicsFactor, PhysicsError> {
        match self {
            Self::DenseReference => model.factor(diagonal),
            Self::Articulated => {
                DynamicsFactor::articulated(creation, &model.poses, coordinates, diagonal)
            }
        }
    }
}

/// Positive-definite effective dynamics factored once per substep.
#[derive(Clone, Debug)]
pub struct DynamicsFactor {
    pub(super) size: usize,
    pub(super) storage: FactorStorage,
}

#[derive(Clone, Debug)]
pub(super) enum FactorStorage {
    Dense(Vec<f64>),
    Articulated(articulated::ArticulatedFactor),
}

impl DynamicsFactor {
    /// Factors a symmetric row-major effective mass matrix. No inverse is formed.
    ///
    /// # Errors
    /// Rejects malformed, non-finite, asymmetric, or non-positive dynamics.
    pub fn new(matrix: &[f64], size: usize) -> Result<Self, PhysicsError> {
        if size.checked_mul(size) != Some(matrix.len()) || matrix.iter().any(|x| !x.is_finite()) {
            return Err(PhysicsError::InvalidDynamics);
        }
        let mut lower = vec![0.0; matrix.len()];
        for row in 0..size {
            for column in 0..=row {
                let a = matrix[row * size + column];
                let b = matrix[column * size + row];
                if (a - b).abs() > 1e-12 * a.abs().max(b.abs()).max(1.0) {
                    return Err(PhysicsError::InvalidDynamics);
                }
                let value = a
                    - (0..column)
                        .map(|k| lower[row * size + k] * lower[column * size + k])
                        .sum::<f64>();
                lower[row * size + column] = if row == column {
                    if value <= 0.0 || !value.is_finite() {
                        return Err(PhysicsError::InvalidDynamics);
                    }
                    value.sqrt()
                } else {
                    value / lower[column * size + column]
                };
            }
        }
        Ok(Self {
            size,
            storage: FactorStorage::Dense(lower),
        })
    }

    /// Factors tree dynamics directly, with linear body storage and work. Root
    /// rows use world linear/angular velocities; scalar joints use compiled order.
    /// Numerical factors belong to this exact pose and implicit diagonal. Loop
    /// equations remain separate constraint rows; this does not enforce them.
    ///
    /// # Errors
    /// Rejects invalid poses, diagonal rows, and non-positive effective inertia.
    pub fn articulated(
        creation: &mechanic_core::CompiledCreation,
        roots: &[crate::BodyPose],
        coordinates: &[f64],
        implicit_diagonal: &[f64],
    ) -> Result<Self, PhysicsError> {
        Ok(Self {
            size: creation.dynamics.elimination_parent.len(),
            storage: FactorStorage::Articulated(articulated::ArticulatedFactor::new(
                creation,
                roots,
                coordinates,
                implicit_diagonal,
            )?),
        })
    }

    pub(crate) fn articulated_from_poses(
        creation: &mechanic_core::CompiledCreation,
        poses: &[crate::BodyPose],
        diagonal: &[f64],
    ) -> Result<Self, PhysicsError> {
        Ok(Self {
            size: creation.dynamics.elimination_parent.len(),
            storage: FactorStorage::Articulated(articulated::ArticulatedFactor::from_poses(
                creation, poses, diagonal,
            )?),
        })
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        match &self.storage {
            FactorStorage::Dense(values) => values.capacity() * size_of::<f64>(),
            FactorStorage::Articulated(factor) => factor.retained_bytes(),
        }
    }

    pub(crate) fn refit_articulated(
        &mut self,
        creation: &mechanic_core::CompiledCreation,
        poses: &[crate::BodyPose],
        diagonal: &[f64],
    ) -> Result<(), PhysicsError> {
        match &mut self.storage {
            FactorStorage::Articulated(factor) => {
                factor.refit(creation, poses, diagonal)?;
                self.size = creation.dynamics.elimination_parent.len();
            }
            FactorStorage::Dense(_) => {
                *self = Self::articulated_from_poses(creation, poses, diagonal)?;
            }
        }
        Ok(())
    }

    pub(crate) fn solve_ranges(
        &self,
        values: &mut [f64],
        ranges: &[std::ops::Range<usize>],
    ) -> Result<(), PhysicsError> {
        if values.len() != self.size {
            return Err(PhysicsError::InvalidDynamics);
        }
        match &self.storage {
            FactorStorage::Articulated(factor) => factor.solve_ranges(values, ranges),
            FactorStorage::Dense(_) => {
                for (row, value) in values.iter_mut().enumerate() {
                    if !ranges.iter().any(|range| range.contains(&row)) {
                        *value = 0.0;
                    }
                }
                self.solve(values)
            }
        }
    }

    /// [`Self::solve_ranges`] for several right-hand sides, each solved exactly
    /// as it would be alone.
    pub(crate) fn solve_ranges_many(
        &self,
        values: &mut [Vec<f64>],
        ranges: &[std::ops::Range<usize>],
    ) -> Result<(), PhysicsError> {
        if values.iter().any(|values| values.len() != self.size) {
            return Err(PhysicsError::InvalidDynamics);
        }
        match &self.storage {
            FactorStorage::Articulated(factor) => factor.solve_ranges_many(values, ranges),
            FactorStorage::Dense(_) => values
                .iter_mut()
                .try_for_each(|values| self.solve_ranges(values, ranges)),
        }
    }

    /// The component of an articulated factor whose velocities are exactly
    /// `range`, which [`Self::solve_lanes`] can solve.
    pub(crate) fn lane_component(&self, range: &std::ops::Range<usize>) -> Option<usize> {
        match &self.storage {
            FactorStorage::Articulated(factor) => factor.component_of(range),
            FactorStorage::Dense(_) => None,
        }
    }

    /// The velocities of a component [`Self::lane_component`] named.
    pub(crate) fn lane_velocities(&self, component: usize) -> std::ops::Range<usize> {
        match &self.storage {
            FactorStorage::Articulated(factor) => factor.component_velocities(component),
            FactorStorage::Dense(_) => 0..0,
        }
    }

    /// Solves many right-hand sides confined to one component of an
    /// articulated factor, each exactly as [`Self::solve_ranges`] solves a
    /// lone one. `values` holds each lane's slice of the component's
    /// velocities in turn; `finite` receives whether each lane's solve would
    /// have succeeded.
    pub(crate) fn solve_lanes(
        &self,
        component: usize,
        values: &mut [f64],
        finite: &mut Vec<bool>,
        scratch: &mut articulated::LaneScratch,
    ) {
        if let FactorStorage::Articulated(factor) = &self.storage {
            factor.solve_lanes(component, values, finite, scratch);
        }
    }

    /// Applies inverse dynamics in-place to a generalized impulse.
    ///
    /// # Errors
    /// Rejects incorrect row counts or non-finite inputs/results.
    pub fn solve(&self, values: &mut [f64]) -> Result<(), PhysicsError> {
        if values.len() != self.size || values.iter().any(|x| !x.is_finite()) {
            return Err(PhysicsError::InvalidDynamics);
        }
        let lower = match &self.storage {
            FactorStorage::Dense(lower) => lower,
            FactorStorage::Articulated(factor) => return factor.solve(values),
        };
        for row in 0..self.size {
            let previous = (0..row)
                .map(|k| lower[row * self.size + k] * values[k])
                .sum::<f64>();
            values[row] = (values[row] - previous) / lower[row * self.size + row];
        }
        for row in (0..self.size).rev() {
            let next = (row + 1..self.size)
                .map(|k| lower[k * self.size + row] * values[k])
                .sum::<f64>();
            values[row] = (values[row] - next) / lower[row * self.size + row];
        }
        if values.iter().any(|x| !x.is_finite()) {
            return Err(PhysicsError::InvalidDynamics);
        }
        Ok(())
    }
}
