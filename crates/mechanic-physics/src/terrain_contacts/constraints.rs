//! Contact manifold rows for the coupled machine response.

use super::{ContactObstacle, TerrainContact, TerrainContactQuery};
use crate::{ConstraintBlock, ContactFriction, ImpulseBounds, MachineDynamics, PhysicsError};
use bevy_math::DVec3;
use std::collections::BTreeMap;

impl TerrainContact {
    /// Generalized row of the relative point velocity along `direction`: the
    /// receiving body's material point minus the opposing body's, if any.
    ///
    /// # Errors
    /// Rejects unknown bodies or non-finite points/directions.
    pub fn point_row(
        &self,
        model: &MachineDynamics,
        direction: DVec3,
    ) -> Result<Vec<f64>, PhysicsError> {
        let mut row = model.point_row(self.body, self.body_point, direction)?;
        if let Some(other) = self.other_body {
            let opposing = model.point_row(other, self.terrain_point, direction)?;
            for (value, theirs) in row.iter_mut().zip(opposing) {
                *value -= theirs;
            }
        }
        Ok(row)
    }

    /// Generalized row of the relative angular velocity along `direction`.
    ///
    /// # Errors
    /// Rejects unknown bodies or a non-finite direction.
    pub fn angular_row(
        &self,
        model: &MachineDynamics,
        direction: DVec3,
    ) -> Result<Vec<f64>, PhysicsError> {
        let mut row = model.angular_row(self.body, direction)?;
        if let Some(other) = self.other_body {
            let opposing = model.angular_row(other, direction)?;
            for (value, theirs) in row.iter_mut().zip(opposing) {
                *value -= theirs;
            }
        }
        Ok(row)
    }
}

/// Coupled impact blocks and their geometric source point order.
pub struct TerrainImpactConstraints {
    /// One block per retained collider/surface manifold.
    pub blocks: Vec<ConstraintBlock>,
    /// Indices into the source query's contacts, in block/contact-point order.
    /// Each point owns three rows, plus two when rolling resistance is enabled.
    pub point_indices: Vec<usize>,
}

impl TerrainContactQuery {
    /// Builds an instantaneous impact solve from current incoming velocities.
    /// Each manifold couples its normal, two tangential, and two optional rolling
    /// directions through the machine's generalized response. Normal targets use
    /// restitution only above the fixed velocity threshold. Penetration is never
    /// added to these physical velocity targets; recovery is a separate solve.
    ///
    /// # Errors
    /// Rejects invalid velocities, geometry, nonnegative threshold settings, or
    /// separated proximity points. A positive gap needs continuous activation;
    /// it cannot silently become an instantaneous supporting contact.
    pub fn impact_constraints(
        &self,
        model: &MachineDynamics,
        incoming: &[f64],
        restitution_threshold: f64,
        stiction_threshold: f64,
    ) -> Result<TerrainImpactConstraints, PhysicsError> {
        model.body_motions(incoming)?;
        if !restitution_threshold.is_finite()
            || restitution_threshold < 0.0
            || !stiction_threshold.is_finite()
            || stiction_threshold < 0.0
        {
            return Err(PhysicsError::InvalidConstraints);
        }
        let mut groups = BTreeMap::<(usize, Option<usize>, usize), Vec<usize>>::new();
        for (index, contact) in self.contacts.iter().enumerate() {
            if !contact.separation.is_finite()
                || contact.separation > contact.feature.obstacle.target().activation_distance()
            {
                return Err(PhysicsError::InvalidConstraints);
            }
            // Terrain manifolds share one numbering per collider; each collider
            // pair numbers its own.
            let opposing = match contact.feature.obstacle {
                ContactObstacle::Terrain { .. } => None,
                ContactObstacle::Collider(collider) => Some(collider),
            };
            groups
                .entry((contact.feature.collider, opposing, contact.manifold))
                .or_default()
                .push(index);
        }
        let mut result = TerrainImpactConstraints {
            blocks: Vec::new(),
            point_indices: Vec::new(),
        };
        for indices in groups.values() {
            let mut block = ConstraintBlock {
                jacobian: Vec::new(),
                target: Vec::new(),
                bounds: Vec::new(),
                contacts: Vec::new(),
            };
            for &index in indices {
                let contact = &self.contacts[index];
                let reference = if contact.normal.y.abs() > 0.9 {
                    DVec3::X
                } else {
                    DVec3::Y
                };
                let u = reference.cross(contact.normal).normalize();
                let v = contact.normal.cross(u);
                let first = block.jacobian.len();
                for direction in [contact.normal, u, v] {
                    let row = contact.point_row(model, direction)?;
                    let speed = row
                        .iter()
                        .zip(incoming)
                        .map(|(j, value)| j * value)
                        .sum::<f64>();
                    block.jacobian.push(row);
                    block.target.push(-speed);
                    block.bounds.push(ImpulseBounds {
                        minimum: f64::NEG_INFINITY,
                        maximum: f64::INFINITY,
                    });
                }
                block.bounds[first].minimum = 0.0;
                if block.target[first] > restitution_threshold {
                    block.target[first] *= 1.0 + contact.response[2];
                }
                let sliding =
                    block.target[first + 1].hypot(block.target[first + 2]) > stiction_threshold;
                let radius = contact
                    .body_point
                    .distance(model.poses[contact.body].position)
                    .max(1e-3);
                let rolling_length =
                    (contact.response[3] > 0.0).then_some(contact.response[3] * radius);
                if rolling_length.is_some() {
                    for direction in [u, v] {
                        let row = contact.angular_row(model, direction)?;
                        let speed = row
                            .iter()
                            .zip(incoming)
                            .map(|(j, value)| j * value)
                            .sum::<f64>();
                        block.jacobian.push(row);
                        block.target.push(-speed);
                        block.bounds.push(ImpulseBounds {
                            minimum: f64::NEG_INFINITY,
                            maximum: f64::INFINITY,
                        });
                    }
                }
                block.contacts.push(ContactFriction {
                    static_coefficient: contact.response[0],
                    kinetic_coefficient: contact.response[1],
                    sliding,
                    rolling_length,
                });
                result.point_indices.push(index);
            }
            result.blocks.push(block);
        }
        Ok(result)
    }
}
