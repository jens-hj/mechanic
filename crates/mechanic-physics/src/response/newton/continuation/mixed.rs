//! Eliminate well-conditioned local search rows; retain weak rows in a bounded
//! mixed generalized/contact system. No local diagonal shift changes the equation.

use super::{DynamicsFactor, Evaluation, NewtonWork, dot};

const MAXIMUM_MIXED_ROWS: usize = 128;
const LOCAL_PIVOT_THRESHOLD: f64 = 1e-8;

struct Point {
    first: usize,
    size: usize,
    rank: usize,
    retained: usize,
    permutation: Vec<usize>,
    matrix: Vec<f64>,
    rhs: Vec<f64>,
    coupling: Vec<f64>,
}

#[allow(clippy::too_many_lines)] // Complete local pivoting transforms all companion rows together.
fn prepare(
    at: &super::smoothing::Point,
    rows: &[&Vec<f64>],
    scales: &[f64],
    rhs: &[f64],
    size: usize,
    retained: usize,
) -> Point {
    let width = at.size;
    let mut point = Point {
        first: at.first,
        size: width,
        rank: 0,
        retained,
        permutation: (0..width).collect(),
        matrix: at.derivative.iter().map(|p| -p).collect(),
        rhs: rhs[at.first..at.first + width].to_vec(),
        coupling: vec![0.0; width * size],
    };
    for row in 0..width {
        point.matrix[row * width + row] += 1.0;
        for (column, _) in rows[at.first].iter().enumerate() {
            point.coupling[row * size + column] = (0..width)
                .map(|k| {
                    at.derivative[row * width + k] * rows[at.first + k][column]
                        / scales[at.first + k]
                })
                .sum();
        }
    }
    for column in 0..width {
        let mut pivot = (column, column);
        for row in column..width {
            for index in column..width {
                if point.matrix[row * width + index].abs()
                    > point.matrix[pivot.0 * width + pivot.1].abs()
                {
                    pivot = (row, index);
                }
            }
        }
        // This threshold chooses which unknowns to retain. It never drops a row:
        // the exact remaining Schur block is included in the mixed matrix.
        if point.matrix[pivot.0 * width + pivot.1].abs() <= LOCAL_PIVOT_THRESHOLD {
            break;
        }
        for index in 0..width {
            point
                .matrix
                .swap(column * width + index, pivot.0 * width + index);
        }
        point.rhs.swap(column, pivot.0);
        for index in 0..size {
            point
                .coupling
                .swap(column * size + index, pivot.0 * size + index);
        }
        for row in 0..width {
            point
                .matrix
                .swap(row * width + column, row * width + pivot.1);
        }
        point.permutation.swap(column, pivot.1);
        for row in column + 1..width {
            let multiplier =
                point.matrix[row * width + column] / point.matrix[column * width + column];
            point.matrix[row * width + column] = 0.0;
            for index in column + 1..width {
                point.matrix[row * width + index] -=
                    multiplier * point.matrix[column * width + index];
            }
            point.rhs[row] -= multiplier * point.rhs[column];
            for index in 0..size {
                point.coupling[row * size + index] -=
                    multiplier * point.coupling[column * size + index];
            }
        }
        point.rank = column + 1;
    }
    // Back substitution removes the strong upper block. Each strong equation
    // becomes lambda_i = rhs_i - B_i*dq - C_i*lambda_retained.
    for row in (0..point.rank).rev() {
        let diagonal = point.matrix[row * width + row];
        for next in row + 1..point.rank {
            let multiplier = point.matrix[row * width + next];
            point.rhs[row] -= multiplier * point.rhs[next];
            for index in 0..size {
                point.coupling[row * size + index] -=
                    multiplier * point.coupling[next * size + index];
            }
            for index in point.rank..width {
                point.matrix[row * width + index] -=
                    multiplier * point.matrix[next * width + index];
            }
        }
        point.rhs[row] /= diagonal;
        for index in 0..size {
            point.coupling[row * size + index] /= diagonal;
        }
        for index in point.rank..width {
            point.matrix[row * width + index] /= diagonal;
        }
    }
    point
}

#[allow(clippy::too_many_lines)] // Assemble, solve, and reconstruct the same bounded Schur system.
pub(super) fn direction(
    factor: &DynamicsFactor,
    rows: &[&Vec<f64>],
    scales: &[f64],
    at: &Evaluation,
    work: &mut NewtonWork,
) -> Option<Vec<f64>> {
    let size = factor.size;
    let mut retained = 0;
    let mut points = Vec::new();
    for point in &at.points {
        work.local_factorizations += 1;
        let point = prepare(point, rows, scales, &at.rhs, size, retained);
        retained += point.size - point.rank;
        if size + retained > MAXIMUM_MIXED_ROWS {
            return None;
        }
        points.push(point);
    }
    let count = size + retained;
    let mut matrix = vec![0.0; count * count];
    let mut rhs = vec![0.0; count];
    let mut dynamic = vec![0.0; size * count];
    let mut impulse = vec![0.0; size];
    for point in &points {
        for row in 0..point.rank {
            let jacobian = rows[point.first + point.permutation[row]];
            for (axis, &j) in jacobian.iter().enumerate() {
                impulse[axis] += j * point.rhs[row];
                for column in 0..size {
                    dynamic[axis * count + column] += j * point.coupling[row * size + column];
                }
                for column in point.rank..point.size {
                    dynamic[axis * count + size + point.retained + column - point.rank] +=
                        j * point.matrix[row * point.size + column];
                }
            }
        }
        for local in point.rank..point.size {
            let row = size + point.retained + local - point.rank;
            rhs[row] = point.rhs[local];
            for column in 0..size {
                matrix[row * count + column] = point.coupling[local * size + column];
            }
            for column in point.rank..point.size {
                matrix[row * count + size + point.retained + column - point.rank] =
                    point.matrix[local * point.size + column];
            }
            for (axis, &j) in rows[point.first + point.permutation[local]]
                .iter()
                .enumerate()
            {
                dynamic[axis * count + row] -= j;
            }
        }
    }
    work.factor_solves += 1;
    factor.solve(&mut impulse).ok()?;
    rhs[..size].copy_from_slice(&impulse);
    for column in 0..count {
        let mut values = (0..size)
            .map(|row| dynamic[row * count + column])
            .collect::<Vec<_>>();
        work.factor_solves += 1;
        factor.solve(&mut values).ok()?;
        for (row, value) in values.into_iter().enumerate() {
            matrix[row * count + column] = value + f64::from(row == column);
        }
    }
    // Include original and mutable rank-factor matrices, the rectangular dynamics
    // assembly, and both original/transformed local projection matrices.
    let local = points
        .iter()
        .map(|p| 2 * p.size * p.size + p.size * size)
        .sum::<usize>();
    let matrix_storage = 2 * count * count + dynamic.len() + local;
    work.reduced_storage = work.reduced_storage.max(matrix_storage);
    work.mixed_rows = work.mixed_rows.max(count);
    work.generalized_factorizations += 1;
    let solution = equilibrated_rank_solve(&matrix, &rhs, matrix_storage, work)?;
    let mut result = vec![0.0; rows.len()];
    for point in &points {
        let retained =
            &solution[size + point.retained..size + point.retained + point.size - point.rank];
        for row in 0..point.rank {
            result[point.first + point.permutation[row]] = point.rhs[row]
                - dot(
                    &point.coupling[row * size..(row + 1) * size],
                    &solution[..size],
                )
                - dot(
                    &point.matrix[row * point.size + point.rank..(row + 1) * point.size],
                    retained,
                );
        }
        for (row, &value) in retained.iter().enumerate() {
            result[point.first + point.permutation[point.rank + row]] = value;
        }
    }
    result.iter().all(|v| v.is_finite()).then_some(result)
}

fn equilibrated_rank_solve(
    matrix: &[f64],
    rhs: &[f64],
    storage: usize,
    work: &mut NewtonWork,
) -> Option<Vec<f64>> {
    let size = rhs.len();
    let mut scaled = matrix.to_vec();
    let mut target = rhs.to_vec();
    for (row, value) in scaled.chunks_exact_mut(size).zip(&mut target) {
        let scale = row.iter().map(|v| v.abs()).fold(0.0, f64::max);
        if scale > 0.0 {
            for v in row {
                *v /= scale;
            }
            *value /= scale;
        }
    }
    let result = super::rank::least_squares(&scaled, &target, storage, work)?;
    result.iter().all(|v| v.is_finite()).then_some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConstraintBlock, ContactFriction, ImpulseBounds, PreparedConstraints};

    fn problem(count: usize) -> (DynamicsFactor, Vec<ConstraintBlock>, Vec<f64>) {
        let factor =
            DynamicsFactor::new(&[2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0], 3).unwrap();
        let block = ConstraintBlock {
            jacobian: vec![
                vec![0.0, 1.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 1.0],
                vec![1.0, 0.0, -1.0],
            ],
            target: vec![0.0; 5],
            bounds: std::iter::once(ImpulseBounds {
                minimum: 0.0,
                maximum: f64::INFINITY,
            })
            .chain(std::iter::repeat_n(
                ImpulseBounds {
                    minimum: f64::NEG_INFINITY,
                    maximum: f64::INFINITY,
                },
                4,
            ))
            .collect(),
            contacts: vec![ContactFriction {
                static_coefficient: 0.8,
                kinetic_coefficient: 0.5,
                sliding: true,
                rolling_length: Some(0.05),
            }],
        };
        (
            factor,
            vec![block; count],
            [2.0, 0.4, 0.1, 0.3, 0.4].repeat(count),
        )
    }

    #[test]
    fn mixed_search_reconstructs_every_original_newton_row_above_the_dense_response_boundary() {
        let (factor, blocks, impulses) = problem(35);
        let prepared = PreparedConstraints::new(&factor, &blocks).unwrap();
        assert!(prepared.response.is_empty());
        let modes = vec![vec![true]; blocks.len()];
        let at = super::super::smoothing::evaluate(
            &blocks,
            &prepared.layout,
            &modes,
            &vec![None; blocks.len()],
            &impulses,
            &vec![0.0; impulses.len()],
            1e-6,
        )
        .unwrap();
        let rows = blocks.iter().flat_map(|b| &b.jacobian).collect::<Vec<_>>();
        let scales = prepared
            .layout
            .iter()
            .flat_map(|info| [info.scale; 5])
            .collect::<Vec<_>>();
        let mut work = NewtonWork::default();
        let delta = direction(&factor, &rows, &scales, &at, &mut work).unwrap();
        assert!(work.mixed_rows > factor.size && work.mixed_rows <= 128);
        let (_, motion) = crate::response::response_motion(&factor, &rows, &delta).unwrap();
        let input = delta
            .iter()
            .zip(motion)
            .zip(&scales)
            .map(|((lambda, v), scale)| lambda - v / scale)
            .collect::<Vec<_>>();
        for point in &at.points {
            for local in 0..point.size {
                let row = point.first + local;
                let response = dot(
                    &point.derivative[local * point.size..(local + 1) * point.size],
                    &input[point.first..point.first + point.size],
                );
                let error = delta[row] - response - at.rhs[row];
                let magnitude = delta[row].abs() + response.abs() + at.rhs[row].abs();
                assert!(
                    error.abs() <= 1e-8 * magnitude.max(1.0),
                    "row={row} error={error:e}"
                );
            }
        }
    }

    #[test]
    fn too_many_weak_rows_reject_the_search_without_allocating_a_large_contact_matrix() {
        let (factor, blocks, impulses) = problem(50);
        let prepared = PreparedConstraints::new(&factor, &blocks).unwrap();
        let at = super::super::smoothing::evaluate(
            &blocks,
            &prepared.layout,
            &vec![vec![true]; 50],
            &vec![None; 50],
            &impulses,
            &vec![0.0; impulses.len()],
            1e-6,
        )
        .unwrap();
        let rows = blocks.iter().flat_map(|b| &b.jacobian).collect::<Vec<_>>();
        let scales = prepared
            .layout
            .iter()
            .flat_map(|info| [info.scale; 5])
            .collect::<Vec<_>>();
        let mut work = NewtonWork::default();
        assert!(direction(&factor, &rows, &scales, &at, &mut work).is_none());
        assert_eq!(work.mixed_rows, 0);
        assert_eq!(work.generalized_factorizations, 0);
        assert_eq!(work.factor_solves, 0);
    }
}
