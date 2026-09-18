//! Eliminate contact-space Newton motion into a bounded generalized system.
//!
//! With projection derivative P, row steps S and shift s, the Newton matrix is
//! A + P S J H^-1 J^T, where A = (1+s)I-P is point-block diagonal.
//! Solve (I + H^-1 J^T A^-1 P S J) dq = H^-1 J^T A^-1 rhs,
//! then recover every contact impulse. No global contact response is stored.

use super::{
    ConstraintBlock, DynamicsFactor, NewtonWork, apply_lu, factor_dense,
    projection::PointDerivative,
};

/// Bounds quadratic storage by generalized dimension; larger machines use Krylov.
pub(super) const MAXIMUM_COORDINATES: usize = 64;

pub(super) fn direction(
    factor: &DynamicsFactor,
    blocks: &[ConstraintBlock],
    derivatives: &[PointDerivative],
    steps: &[f64],
    rhs: &[f64],
    shift: f64,
) -> NewtonWork {
    let mut work = NewtonWork::default();
    work.direction = solve(factor, blocks, derivatives, steps, rhs, shift, &mut work);
    work
}

fn solve(
    factor: &DynamicsFactor,
    blocks: &[ConstraintBlock],
    derivatives: &[PointDerivative],
    steps: &[f64],
    rhs: &[f64],
    shift: f64,
    work: &mut NewtonWork,
) -> Option<Vec<f64>> {
    let size = factor.size;
    let rows = blocks
        .iter()
        .flat_map(|block| &block.jacobian)
        .collect::<Vec<_>>();
    let mut local_rhs = rhs.to_vec();
    let mut response = vec![0.0; rhs.len() * size];
    work.reduced_storage = response.len() + size * size;
    for point in derivatives {
        let width = point.size;
        work.local_factorizations += 1;
        let inverse = point.shifted_inverse(shift)?;
        inverse.solve(&mut local_rhs[point.first..point.first + width])?;
        for column in 0..size {
            let mut input = [0.0; 5];
            let mut output = [0.0; 5];
            for row in 0..width {
                input[row] = steps[point.first + row] * rows[point.first + row][column];
            }
            point.apply(&input[..width], &mut output[..width]);
            inverse.solve(&mut output[..width])?;
            for row in 0..width {
                response[(point.first + row) * size + column] = output[row];
            }
        }
    }
    let mut matrix = vec![0.0; size * size];
    let mut scratch = vec![0.0; size];
    for column in 0..size {
        scratch.fill(0.0);
        for (row, jacobian) in rows.iter().enumerate() {
            for (out, j) in scratch.iter_mut().zip(*jacobian) {
                *out += j * response[row * size + column];
            }
        }
        work.factor_solves += 1;
        factor.solve(&mut scratch).ok()?;
        for row in 0..size {
            matrix[row * size + column] = scratch[row] + f64::from(row == column);
        }
    }
    scratch.fill(0.0);
    for (row, value) in rows.iter().zip(&local_rhs) {
        for (out, j) in scratch.iter_mut().zip(*row) {
            *out += j * value;
        }
    }
    work.factor_solves += 1;
    factor.solve(&mut scratch).ok()?;
    work.generalized_factorizations += 1;
    let (matrix, pivots) = factor_dense(matrix, size)?;
    apply_lu(&matrix, &pivots, &mut scratch)?;
    for (row, out) in local_rhs.iter_mut().enumerate() {
        *out -= response[row * size..(row + 1) * size]
            .iter()
            .zip(&scratch)
            .map(|(a, b)| a * b)
            .sum::<f64>();
    }
    local_rhs
        .iter()
        .all(|value| value.is_finite())
        .then_some(local_rhs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContactFriction, ImpulseBounds, PreparedConstraints};

    #[test]
    fn recovered_reduced_direction_satisfies_every_full_newton_row() {
        let factor = DynamicsFactor::new(&[2.0, 0.2, 0.2, 1.0], 2).unwrap();
        let mut block = ConstraintBlock {
            jacobian: vec![
                vec![1.0, 0.2],
                vec![0.3, 1.0],
                vec![-0.4, 0.2],
                vec![0.2, -0.1],
                vec![-0.1, 0.7],
            ],
            target: vec![0.0; 5],
            bounds: vec![
                ImpulseBounds {
                    minimum: f64::NEG_INFINITY,
                    maximum: f64::INFINITY
                };
                5
            ],
            contacts: vec![ContactFriction {
                static_coefficient: 0.8,
                kinetic_coefficient: 0.5,
                sliding: true,
                rolling_length: Some(0.1),
            }],
        };
        block.bounds[0].minimum = 0.0;
        let blocks = [block];
        let prepared = PreparedConstraints::new(&factor, &blocks).unwrap();
        let steps = [prepared.layout[0].scale.recip(); 5];
        let at = [2.0, 3.0, 4.0, 5.0, -7.0];
        let derivatives =
            super::super::projection::prepare(&blocks, &prepared.layout, &[vec![true]], &at);
        let rhs = [1.0, -0.3, 0.7, 0.2, -0.5];
        let shift = 1e-3;
        let work = direction(&factor, &blocks, &derivatives, &steps, &rhs, shift);
        assert_eq!(work.factor_solves, 3);
        assert_eq!(work.generalized_factorizations, 1);
        assert_eq!(work.reduced_storage, 14);
        let delta = work.direction.unwrap();
        let mut velocity = vec![0.0; 2];
        for (row, impulse) in blocks[0].jacobian.iter().zip(&delta) {
            for (out, j) in velocity.iter_mut().zip(row) {
                *out += j * impulse;
            }
        }
        factor.solve(&mut velocity).unwrap();
        let input = blocks[0]
            .jacobian
            .iter()
            .zip(&delta)
            .zip(steps)
            .map(|((row, change), step)| {
                change - step * row.iter().zip(&velocity).map(|(j, v)| j * v).sum::<f64>()
            })
            .collect::<Vec<_>>();
        let mut projection = [0.0; 5];
        derivatives[0].apply(&input, &mut projection);
        for row in 0..5 {
            let residual = (1.0 + shift) * delta[row] - projection[row] - rhs[row];
            assert!(residual.abs() < 1e-10, "row={row} residual={residual}");
        }
    }
}
