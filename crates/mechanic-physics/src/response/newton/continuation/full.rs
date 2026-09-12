//! Search-only dense direction over every contact row of a small machine.
//!
//! Support transitions leave hundreds of weak, nearly dependent contact rows.
//! No reduced system under the 128-row response bound truncates them, so this
//! search deliberately stores its full matrix. Original contact laws alone
//! accept a proposal.

use super::{DynamicsFactor, Evaluation, NewtonWork, dot};

/// Largest contact row count searched in full space.
pub(super) const MAXIMUM_FULL_ROWS: usize = 256;

/// Pivot columns weaker than this fraction of the leading pivot are dropped.
/// Retaining nearly dependent support rows yields proposals the line search
/// cannot accept; an absolute round-off cutoff keeps them. Offline, 1e-13 fails
/// the support transition and 1e-10 later settling. 1e-12 helps captured impacts
/// but regresses the endpoint-policy drop from tick 94 to tick 61.
const RELATIVE_RANK: f64 = 1e-11;

/// Pose-local `I - W/s`, prepared once for every direction of one search.
pub(super) struct Coupling {
    size: usize,
    matrix: Vec<f64>,
}

impl Coupling {
    pub(super) fn new(
        factor: &DynamicsFactor,
        rows: &[&Vec<f64>],
        scales: &[f64],
        work: &mut NewtonWork,
    ) -> Option<Self> {
        let size = rows.len();
        if size == 0 || size > MAXIMUM_FULL_ROWS {
            return None;
        }
        let mut matrix = vec![0.0; size * size];
        for column in 0..size {
            let mut response = rows[column].clone();
            work.factor_solves += 1;
            factor.solve(&mut response).ok()?;
            for (row, jacobian) in rows.iter().enumerate() {
                matrix[row * size + column] =
                    f64::from(row == column) - dot(jacobian, &response) / scales[row];
            }
        }
        Some(Self { size, matrix })
    }
}

/// Solves the Newton rows `(I - P (I - W/s)) d = rhs`, with `P` block diagonal
/// over contact points, as a rank-truncated minimum-norm least-squares problem.
pub(super) fn direction(
    coupling: &Coupling,
    at: &Evaluation,
    work: &mut NewtonWork,
) -> Option<Vec<f64>> {
    let size = coupling.size;
    if at.rhs.len() != size {
        return None;
    }
    let mut matrix = vec![0.0; size * size];
    for value in matrix.iter_mut().step_by(size + 1) {
        *value = 1.0;
    }
    for point in &at.points {
        for row in 0..point.size {
            let target = (point.first + row) * size;
            for local in 0..point.size {
                let derivative = point.derivative[row * point.size + local];
                let source = (point.first + local) * size;
                for column in 0..size {
                    matrix[target + column] -= derivative * coupling.matrix[source + column];
                }
            }
        }
    }
    let mut rhs = at.rhs.clone();
    for (row, value) in matrix.chunks_exact_mut(size).zip(&mut rhs) {
        if !row.iter().all(|entry| entry.is_finite()) {
            return None;
        }
        let scale = row.iter().fold(0.0_f64, |largest, v| largest.max(v.abs()));
        if scale > 0.0 {
            for entry in row.iter_mut() {
                *entry /= scale;
            }
            *value /= scale;
        }
    }
    work.full_rows = work.full_rows.max(size);
    work.generalized_factorizations += 1;
    let (solution, rank) = truncated_solve(&mut matrix, &mut rhs, size)?;
    // The coupling, factored search matrix and minimum-norm row basis coexist.
    work.reduced_storage = work
        .reduced_storage
        .max(2 * size * size + rank * size + rank * rank);
    Some(solution)
}

// Column-pivoted Householder QR with downdated column norms, truncated at a
// relative pivot, then the minimum-norm solve of its retained rows.
fn truncated_solve(a: &mut [f64], b: &mut [f64], size: usize) -> Option<(Vec<f64>, usize)> {
    let column_norm = |a: &[f64], column: usize, first: usize| {
        (first..size).fold(0.0_f64, |sum, row| sum.hypot(a[row * size + column]))
    };
    let mut permutation = (0..size).collect::<Vec<_>>();
    let mut norms = (0..size)
        .map(|column| column_norm(a, column, 0))
        .collect::<Vec<_>>();
    let mut reference = norms.clone();
    let downdate_limit = f64::EPSILON.sqrt();
    let mut leading = 0.0;
    let mut rank = size;
    for column in 0..size {
        let mut pivot = column;
        for candidate in column + 1..size {
            if norms[candidate] > norms[pivot] {
                pivot = candidate;
            }
        }
        if pivot != column {
            for row in 0..size {
                a.swap(row * size + column, row * size + pivot);
            }
            norms.swap(column, pivot);
            reference.swap(column, pivot);
            permutation.swap(column, pivot);
        }
        let largest = column_norm(a, column, column);
        if !largest.is_finite() {
            return None;
        }
        if column == 0 {
            leading = largest;
        }
        if largest <= 0.0 || largest <= RELATIVE_RANK * leading {
            rank = column;
            break;
        }
        let mut reflection = (column..size)
            .map(|row| a[row * size + column])
            .collect::<Vec<_>>();
        reflection[0] += largest.copysign(reflection[0]);
        let length = reflection.iter().fold(0.0_f64, |sum, v| sum.hypot(*v));
        if length <= 0.0 || !length.is_finite() {
            return None;
        }
        for value in &mut reflection {
            *value /= length;
        }
        for index in column..size {
            let component = 2.0
                * reflection
                    .iter()
                    .enumerate()
                    .map(|(local, v)| v * a[(column + local) * size + index])
                    .sum::<f64>();
            for (local, v) in reflection.iter().enumerate() {
                a[(column + local) * size + index] -= component * v;
            }
        }
        let component = 2.0 * dot(&reflection, &b[column..]);
        for (value, v) in b[column..].iter_mut().zip(&reflection) {
            *value -= component * v;
        }
        for row in column + 1..size {
            a[row * size + column] = 0.0;
        }
        for index in column + 1..size {
            if norms[index] > 0.0 {
                let ratio = a[column * size + index].abs() / norms[index];
                let remaining = (1.0 - ratio * ratio).max(0.0);
                if remaining * (norms[index] / reference[index]).powi(2) <= downdate_limit {
                    norms[index] = column_norm(a, index, column + 1);
                    reference[index] = norms[index];
                } else {
                    norms[index] *= remaining.sqrt();
                }
            }
        }
    }
    let solution = super::rank::minimum_norm_rows(a, b, size, rank)?;
    let mut output = vec![0.0; size];
    for (column, value) in solution.into_iter().enumerate() {
        output[permutation[column]] = value;
    }
    output
        .iter()
        .all(|value| value.is_finite())
        .then_some((output, rank))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConstraintBlock, ContactFriction, ImpulseBounds, PreparedConstraints};

    #[test]
    fn truncated_solve_drops_a_nearly_dependent_direction_and_returns_the_minimum_norm_step() {
        // Keeping the 1e-14 pivot would give the large step [7, -5, 1].
        let mut matrix = [1.0, 1.0, 0.0, 0.0, 1e-14, 0.0, 0.0, 0.0, 1.0];
        let mut rhs = [2.0, -5e-14, 1.0];
        let (solution, rank) = truncated_solve(&mut matrix, &mut rhs, 3).unwrap();
        assert_eq!(rank, 2);
        for (actual, expected) in solution.iter().zip([1.0, 1.0, 1.0]) {
            assert!((actual - expected).abs() < 1e-9, "{solution:?}");
        }
    }

    #[test]
    fn truncated_solve_keeps_a_weak_independent_direction_exactly() {
        let mut matrix = [1.0, 1.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 0.0, 1.0];
        let mut rhs = [2.0, -5e-6, 1.0];
        let (solution, rank) = truncated_solve(&mut matrix, &mut rhs, 3).unwrap();
        assert_eq!(rank, 3);
        for (actual, expected) in solution.iter().zip([7.0, -5.0, 1.0]) {
            assert!((actual - expected).abs() < 1e-9, "{solution:?}");
        }
    }

    #[test]
    fn full_space_direction_satisfies_every_original_newton_row() {
        #[rustfmt::skip]
        let mass = [
            2.0, 0.3, 0.0, 0.1, 0.0,
            0.3, 3.0, 0.2, 0.0, 0.0,
            0.0, 0.2, 4.0, 0.0, 0.1,
            0.1, 0.0, 0.0, 1.5, 0.0,
            0.0, 0.0, 0.1, 0.0, 2.5,
        ];
        let factor = DynamicsFactor::new(&mass, 5).unwrap();
        let unbounded = ImpulseBounds {
            minimum: f64::NEG_INFINITY,
            maximum: f64::INFINITY,
        };
        let blocks = [ConstraintBlock {
            jacobian: (0..5)
                .map(|row| (0..5).map(|column| f64::from(row == column)).collect())
                .collect(),
            target: vec![0.2, -0.1, 0.3, 0.05, -0.02],
            bounds: std::iter::once(ImpulseBounds {
                minimum: 0.0,
                maximum: f64::INFINITY,
            })
            .chain(std::iter::repeat_n(unbounded, 4))
            .collect(),
            contacts: vec![ContactFriction {
                static_coefficient: 0.8,
                kinetic_coefficient: 0.5,
                sliding: true,
                rolling_length: Some(0.05),
            }],
        }];
        let prepared = PreparedConstraints::new(&factor, &blocks).unwrap();
        let rows = blocks[0].jacobian.iter().collect::<Vec<_>>();
        let scales = vec![prepared.layout[0].scale; 5];
        let impulses = [2.0, 0.9, 0.3, 0.3, 0.4];
        let (_, motion) = crate::response::response_motion(&factor, &rows, &impulses).unwrap();
        let at = super::super::smoothing::evaluate(
            &blocks,
            &prepared.layout,
            &[vec![true]],
            &[None],
            &impulses,
            &motion,
            1e-3,
        )
        .unwrap();
        let mut work = NewtonWork::default();
        let coupling = Coupling::new(&factor, &rows, &scales, &mut work).unwrap();
        let delta = direction(&coupling, &at, &mut work).unwrap();
        assert_eq!(work.full_rows, 5);
        let (_, change) = crate::response::response_motion(&factor, &rows, &delta).unwrap();
        let input = delta
            .iter()
            .zip(change)
            .zip(&scales)
            .map(|((lambda, velocity), scale)| lambda - velocity / scale)
            .collect::<Vec<_>>();
        let rows = delta
            .iter()
            .zip(&at.rhs)
            .zip(at.points[0].derivative.chunks_exact(5));
        for (row, ((value, rhs), derivative)) in rows.enumerate() {
            let projected = dot(derivative, &input);
            let error = value - projected - rhs;
            let magnitude = value.abs() + projected.abs() + rhs.abs();
            assert!(
                error.abs() <= 1e-10 * magnitude.max(1.0),
                "row={row} error={error:e}"
            );
        }
    }
}
