//! Pose-local articulated elimination of H = M + diagonal, in world axes.
//!
//! Spatial vectors use [linear, angular] at body origins. Eliminating a joint
//! with motion S leaves I - (I S)(I S)^T / (S^T I S + diagonal). The two RHS
//! passes reuse those factors; no generalized matrix or inverse is formed.

use super::{DynamicsFactor, PhysicsError};
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
    preorder: Vec<usize>,
    postorder: Vec<usize>,
}

impl ArticulatedFactor {
    #[allow(clippy::too_many_lines)] // Ordered spatial inertia construction and joint elimination.
    pub(super) fn new(
        creation: &CompiledCreation,
        roots: &[BodyPose],
        coordinates: &[f64],
        diagonal: &[f64],
    ) -> Result<Self, PhysicsError> {
        let dynamics = &creation.dynamics;
        if diagonal.len() != dynamics.elimination_parent.len()
            || diagonal.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(PhysicsError::InvalidDynamics);
        }
        let poses = MachineDynamics::reconstruct_poses(creation, roots, coordinates)?;
        let mut inertia = vec![[[0.0; 6]; 6]; poses.len()];
        let mut bodies = Vec::with_capacity(poses.len());
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
        Ok(Self {
            bodies,
            preorder: dynamics.preorder.clone(),
            postorder: dynamics.postorder.clone(),
        })
    }

    pub(super) fn solve(&self, values: &mut [f64]) -> Result<(), PhysicsError> {
        // One body-indexed arena serves as backward wrench and forward motion.
        let mut scratch = vec![[0.0; 6]; self.bodies.len()];
        for &index in &self.postorder {
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
        for &index in &self.preorder {
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
        if values.iter().any(|v| !v.is_finite()) {
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
    let flat: Vec<_> = matrix.iter().flatten().copied().collect();
    let factor = DynamicsFactor::new(&flat, 6)?;
    let super::FactorStorage::Dense(lower) = factor.storage else {
        unreachable!()
    };
    Ok(std::array::from_fn(|row| {
        std::array::from_fn(|column| lower[row * 6 + column])
    }))
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
