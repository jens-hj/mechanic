//! Pose-local articulated elimination of H = M + diagonal, in world axes.
//!
//! Spatial vectors use [linear, angular] at body origins. Eliminating a joint
//! with motion S leaves I - (I S)(I S)^T / (S^T I S + diagonal). The two RHS
//! passes reuse those factors; no generalized matrix or inverse is formed.

use super::PhysicsError;
use crate::{BodyPose, MachineDynamics};
use bevy_math::{DMat3, DVec3};
use mechanic_core::CompiledCreation;

type Vector = [f64; 6];
type Matrix = [[f64; 6]; 6];

#[derive(Clone, Debug)]
enum Joint {
    Fixed,
    Floating {
        lower: Matrix,
    },
    Scalar {
        motion: Vector,
        projected: Vector,
        pivot: f64,
    },
}

#[derive(Clone, Debug)]
struct Body {
    parent: Option<usize>,
    row: usize,
    arm: DVec3,
    joint: Joint,
}

#[derive(Clone, Debug)]
pub(super) struct ArticulatedFactor {
    bodies: Vec<Body>,
    inertia: Vec<Matrix>,
    preorder: Vec<usize>,
    components: Vec<mechanic_core::DynamicsComponent>,
    scratch: std::sync::Arc<std::sync::Mutex<Vec<Vector>>>,
}

impl ArticulatedFactor {
    pub(super) fn retained_bytes(&self) -> usize {
        self.bodies.capacity() * size_of::<Body>()
            + self.inertia.capacity() * size_of::<Matrix>()
            + self.preorder.capacity() * size_of::<usize>()
            + self.components.capacity() * size_of::<mechanic_core::DynamicsComponent>()
            + self.scratch.lock().expect("factor scratch lock").capacity() * size_of::<Vector>()
    }

    pub(super) fn new(
        creation: &CompiledCreation,
        roots: &[BodyPose],
        coordinates: &[f64],
        diagonal: &[f64],
    ) -> Result<Self, PhysicsError> {
        let poses = MachineDynamics::reconstruct_poses(creation, roots, coordinates)?;
        Self::from_poses(creation, &poses, diagonal)
    }

    pub(super) fn from_poses(
        creation: &CompiledCreation,
        poses: &[BodyPose],
        diagonal: &[f64],
    ) -> Result<Self, PhysicsError> {
        let mut factor = Self {
            bodies: Vec::new(),
            inertia: Vec::new(),
            preorder: Vec::new(),
            components: Vec::new(),
            scratch: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        };
        factor.refit(creation, poses, diagonal)?;
        Ok(factor)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "rebuild numeric values in retained body/factor arenas"
    )]
    pub(super) fn refit(
        &mut self,
        creation: &CompiledCreation,
        poses: &[BodyPose],
        diagonal: &[f64],
    ) -> Result<(), PhysicsError> {
        let dynamics = &creation.dynamics;
        if diagonal.len() != dynamics.elimination_parent.len()
            || diagonal.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(PhysicsError::InvalidDynamics);
        }
        let mut inertia = std::mem::take(&mut self.inertia);
        inertia.clear();
        inertia.resize(poses.len(), [[0.0; 6]; 6]);
        let mut bodies = std::mem::take(&mut self.bodies);
        bodies.clear();
        for (body, pose) in poses.iter().enumerate() {
            let topology = creation.loop_topology.body_parents[body];
            let parent = (!topology.is_root).then_some(topology.parent_body as usize);
            let rows = dynamics.body_velocities[body].clone();
            let arm = parent.map_or(DVec3::ZERO, |p| pose.position - poses[p].position);
            let joint = if let Some(parent) = parent {
                let bearing = creation.bearings
                    [dynamics.body_bearings[body].ok_or(PhysicsError::InvalidDynamics)?];
                let (anchor, axis, sign) = if topology.bearing_direction == 0 {
                    (bearing.local_anchor_a, bearing.local_axis_a, 1.0)
                } else {
                    (bearing.local_anchor_b, bearing.local_axis_b, -1.0)
                };
                let axis = poses[parent].rotation * axis.as_dvec3().normalize() * sign;
                let motion = if bearing.kind.is_translational() {
                    spatial(axis, DVec3::ZERO)
                } else {
                    let anchor =
                        poses[parent].position + poses[parent].rotation * anchor.as_dvec3();
                    spatial(axis.cross(pose.position - anchor), axis)
                };
                Joint::Scalar {
                    motion,
                    projected: [0.0; 6],
                    pivot: 0.0,
                }
            } else if rows.is_empty() {
                Joint::Fixed
            } else {
                Joint::Floating {
                    lower: [[0.0; 6]; 6],
                }
            };
            bodies.push(Body {
                parent,
                row: rows.start,
                arm,
                joint,
            });
            if !creation.compounds[body].is_static {
                let local = dynamics.inertias[body];
                let rotation = DMat3::from_quat(pose.rotation);
                let rotational = rotation * local.rotational.as_dmat3() * rotation.transpose();
                let center = pose.rotation * local.center.as_dvec3();
                let mass = f64::from(local.mass);
                for column in 0..6 {
                    let mut unit = [0.0; 6];
                    unit[column] = 1.0;
                    let (linear, angular) = parts(unit);
                    let force = mass * (linear + angular.cross(center));
                    let wrench = spatial(force, center.cross(force) + rotational * angular);
                    for row in 0..6 {
                        inertia[body][row][column] = wrench[row];
                    }
                }
                if inertia[body].iter().flatten().any(|v| !v.is_finite()) {
                    return Err(PhysicsError::InvalidDynamics);
                }
            }
        }
        for &body in &dynamics.postorder {
            let mut reduced = inertia[body];
            let row = bodies[body].row;
            match &mut bodies[body].joint {
                Joint::Fixed => {}
                Joint::Floating { lower } => {
                    for axis in 0..6 {
                        reduced[axis][axis] += diagonal[row + axis];
                    }
                    *lower = cholesky(&reduced)?;
                }
                Joint::Scalar {
                    motion,
                    projected,
                    pivot,
                } => {
                    *projected = multiply(&reduced, *motion);
                    *pivot = dot(*motion, *projected) + diagonal[row];
                    if !pivot.is_finite() || *pivot <= 0.0 {
                        return Err(PhysicsError::InvalidDynamics);
                    }
                    for r in 0..6 {
                        for c in 0..6 {
                            reduced[r][c] -= projected[r] * projected[c] / *pivot;
                        }
                    }
                }
            }
            if let Some(parent) = bodies[body].parent {
                // X maps parent-origin motion to this origin; X^T maps wrenches
                // back. All axes are world aligned, so only the lever arm changes.
                let arm = bodies[body].arm;
                for column in 0..6 {
                    let mut unit = [0.0; 6];
                    unit[column] = 1.0;
                    let value = shift_force(multiply(&reduced, shift_motion(unit, arm)), arm);
                    for (row, entry) in value.into_iter().enumerate() {
                        inertia[parent][row][column] += entry;
                    }
                }
            }
        }
        self.bodies = bodies;
        self.inertia = inertia;
        self.preorder.clone_from(&dynamics.preorder);
        self.components.clone_from(&dynamics.components);
        self.scratch
            .lock()
            .map_err(|_| PhysicsError::InvalidDynamics)?
            .resize(poses.len(), [0.0; 6]);
        Ok(())
    }

    pub(super) fn solve(&self, values: &mut [f64]) -> Result<(), PhysicsError> {
        let active = self
            .components
            .iter()
            .filter(|component| {
                values[component.velocities.clone()]
                    .iter()
                    .any(|v| *v != 0.0)
            })
            .collect::<Vec<_>>();
        self.solve_active(values, active.into_iter())
    }

    pub(super) fn solve_ranges(
        &self,
        values: &mut [f64],
        ranges: &[std::ops::Range<usize>],
    ) -> Result<(), PhysicsError> {
        let active = self.components.iter().filter(|component| {
            ranges
                .iter()
                .any(|range| !range.is_empty() && *range == component.velocities)
        });
        self.solve_active(values, active)
    }

    fn solve_active<'a>(
        &self,
        values: &mut [f64],
        active: impl DoubleEndedIterator<Item = &'a mechanic_core::DynamicsComponent> + Clone,
    ) -> Result<(), PhysicsError> {
        // Only the affected components need their body scratch cleared.
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| PhysicsError::InvalidDynamics)?;
        for component in active.clone() {
            if values[component.velocities.clone()]
                .iter()
                .any(|v| !v.is_finite())
            {
                return Err(PhysicsError::InvalidDynamics);
            }
            for &body in &self.preorder[component.bodies.clone()] {
                scratch[body] = [0.0; 6];
            }
        }
        for &index in active
            .clone()
            .rev()
            .flat_map(|component| self.preorder[component.bodies.clone()].iter().rev())
        {
            let body = &self.bodies[index];
            if let Joint::Scalar {
                motion,
                projected,
                pivot,
            } = body.joint
            {
                let rhs = values[body.row] - dot(motion, scratch[index]);
                values[body.row] = rhs;
                let mut wrench = scratch[index];
                for axis in 0..6 {
                    wrench[axis] += projected[axis] * (rhs / pivot);
                }
                if let Some(parent) = body.parent {
                    let shifted = shift_force(wrench, body.arm);
                    for (entry, value) in scratch[parent].iter_mut().zip(shifted) {
                        *entry += value;
                    }
                }
            }
        }
        for &index in active
            .clone()
            .flat_map(|component| &self.preorder[component.bodies.clone()])
        {
            let body = &self.bodies[index];
            scratch[index] = match body.joint {
                Joint::Fixed => [0.0; 6],
                Joint::Floating { lower } => {
                    let mut rhs = [0.0; 6];
                    for axis in 0..6 {
                        rhs[axis] = values[body.row + axis] - scratch[index][axis];
                    }
                    solve_root(&lower, &mut rhs);
                    values[body.row..body.row + 6].copy_from_slice(&rhs);
                    rhs
                }
                Joint::Scalar {
                    motion,
                    projected,
                    pivot,
                } => {
                    let mut inherited =
                        shift_motion(scratch[body.parent.expect("joint has parent")], body.arm);
                    let speed = (values[body.row] - dot(projected, inherited)) / pivot;
                    values[body.row] = speed;
                    for axis in 0..6 {
                        inherited[axis] += motion[axis] * speed;
                    }
                    inherited
                }
            };
        }
        if active
            .flat_map(|component| &values[component.velocities.clone()])
            .any(|v| !v.is_finite())
        {
            return Err(PhysicsError::InvalidDynamics);
        }
        Ok(())
    }
}

fn spatial(linear: DVec3, angular: DVec3) -> Vector {
    [
        linear.x, linear.y, linear.z, angular.x, angular.y, angular.z,
    ]
}

fn parts(v: Vector) -> (DVec3, DVec3) {
    (DVec3::new(v[0], v[1], v[2]), DVec3::new(v[3], v[4], v[5]))
}

fn dot(a: Vector, b: Vector) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}

fn multiply(matrix: &Matrix, vector: Vector) -> Vector {
    matrix.map(|row| dot(row, vector))
}

fn shift_motion(value: Vector, arm: DVec3) -> Vector {
    let (linear, angular) = parts(value);
    spatial(linear + angular.cross(arm), angular)
}

fn shift_force(value: Vector, arm: DVec3) -> Vector {
    let (force, torque) = parts(value);
    spatial(force, torque + arm.cross(force))
}

fn cholesky(matrix: &Matrix) -> Result<Matrix, PhysicsError> {
    // Spatial congruences and Schur updates construct a symmetric inertia.
    // Their independently accumulated mirrored entries can differ by round-off,
    // particularly for a long chassis. Use the same lower triangle as the dense
    // factor without its arbitrary-input symmetry test; do not shift pivots.
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(PhysicsError::InvalidDynamics);
    }
    let mut lower = [[0.0; 6]; 6];
    for row in 0..6 {
        for column in 0..=row {
            let value = matrix[row][column]
                - (0..column)
                    .map(|k| lower[row][k] * lower[column][k])
                    .sum::<f64>();
            if !value.is_finite() || (row == column && value <= 0.0) {
                return Err(PhysicsError::InvalidDynamics);
            }
            lower[row][column] = if row == column {
                value.sqrt()
            } else {
                value / lower[column][column]
            };
        }
    }
    Ok(lower)
}

fn solve_root(lower: &Matrix, values: &mut Vector) {
    for row in 0..6 {
        let previous: f64 = (0..row)
            .map(|column| lower[row][column] * values[column])
            .sum();
        values[row] = (values[row] - previous) / lower[row][row];
    }
    for row in (0..6).rev() {
        let next: f64 = (row + 1..6)
            .map(|column| lower[column][row] * values[column])
            .sum();
        values[row] = (values[row] - next) / lower[row][row];
    }
}

#[cfg(test)]
mod tests;
