//! Column-pivoted Householder QR and a minimum-norm solve of its retained rows.
//! Only bounded Newton search matrices use this least-squares approximation.

use super::dot;
fn norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |sum, value| sum.hypot(*value))
}

#[allow(clippy::too_many_lines)]
pub(super) fn least_squares(
    matrix: &[f64],
    rhs: &[f64],
    storage: usize,
    work: &mut super::NewtonWork,
) -> Option<Vec<f64>> {
    let size = rhs.len();
    let mut a = matrix.to_vec();
    let mut b = rhs.to_vec();
    let mut permutation = (0..size).collect::<Vec<_>>();
    let scale = matrix.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let mut rank = size;
    for column in 0..size {
        let mut pivot = column;
        let mut largest = 0.0;
        for candidate in column..size {
            let length =
                (column..size).fold(0.0_f64, |sum, row| sum.hypot(a[row * size + candidate]));
            if length > largest {
                largest = length;
                pivot = candidate;
            }
        }
        if largest <= 32.0 * f64::EPSILON * scale {
            rank = column;
            break;
        }
        for row in 0..size {
            a.swap(row * size + column, row * size + pivot);
        }
        permutation.swap(column, pivot);
        let mut reflection = (column..size)
            .map(|row| a[row * size + column])
            .collect::<Vec<_>>();
        reflection[0] += largest.copysign(reflection[0]);
        let length = norm(&reflection);
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
    }
    work.reduced_storage = work
        .reduced_storage
        .max(storage + size * size + rank * size + rank * rank);
    let solution = minimum_norm_rows(&a, &b, size, rank)?;
    let mut output = vec![0.0; size];
    for (column, value) in solution.into_iter().enumerate() {
        output[permutation[column]] = value;
    }
    output.iter().all(|v| v.is_finite()).then_some(output)
}
pub(super) fn minimum_norm_rows(
    matrix: &[f64],
    rhs: &[f64],
    size: usize,
    rank: usize,
) -> Option<Vec<f64>> {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut triangular = vec![0.0; rank * rank];
    for column in 0..rank {
        let mut vector = matrix[column * size..(column + 1) * size].to_vec();
        for _ in 0..2 {
            for (row, axis) in basis.iter().enumerate() {
                let component = dot(axis, &vector);
                triangular[row * rank + column] += component;
                for (out, value) in vector.iter_mut().zip(axis) {
                    *out -= component * value;
                }
            }
        }
        let length = norm(&vector);
        if length <= 0.0 || !length.is_finite() {
            return None;
        }
        triangular[column * rank + column] = length;
        for value in &mut vector {
            *value /= length;
        }
        basis.push(vector);
    }
    let mut coefficients = vec![0.0; rank];
    for row in 0..rank {
        coefficients[row] = (rhs[row]
            - (0..row)
                .map(|k| triangular[k * rank + row] * coefficients[k])
                .sum::<f64>())
            / triangular[row * rank + row];
    }
    let mut result = vec![0.0; size];
    for (column, coefficient) in basis.iter().zip(coefficients) {
        for (out, value) in result.iter_mut().zip(column) {
            *out += coefficient * value;
        }
    }
    result.iter().all(|v| v.is_finite()).then_some(result)
}

#[cfg(test)]
mod tests {
    use super::super::NewtonWork;
    use super::*;

    #[test]
    fn redundant_search_equations_use_the_minimum_norm_least_squares_direction() {
        let mut work = NewtonWork::default();
        let result = least_squares(&[1.0, 1.0, 2.0, 2.0], &[1.0, 3.0], 0, &mut work).unwrap();
        // Orthogonal projection onto span([1,2]) requires x+y=7/5.
        // The minimum norm distribution divides that sum equally.
        assert!((result[0] - 0.7).abs() < 1e-14);
        assert!((result[1] - 0.7).abs() < 1e-14);
        assert_eq!(work.reduced_storage, 7);
    }

    #[test]
    fn weak_independent_search_rows_are_kept_and_unavoidable_residual_is_not_biased_to_pivots() {
        let epsilon = 1e-8;
        let matrix = [1.0, 1.0, 0.0, 0.0, epsilon, 1.0, 0.0, 0.0, 0.0];
        let rhs = [2.0, 1.0, 1.0];
        let run = || least_squares(&matrix, &rhs, 0, &mut NewtonWork::default()).unwrap();
        let result = run();
        let expected_y = (2.0 + epsilon) / (2.0 + epsilon * epsilon);
        for (a, b) in result
            .iter()
            .zip([2.0 - expected_y, expected_y, 1.0 - epsilon * expected_y])
        {
            assert!((a - b).abs() < 1e-14);
        }
        assert_eq!(result, run());
        // The third zero row has residual -1. Search permits that least-squares
        // residual; this utility is never the physical constraint validator.
        assert!((result[0] + result[1] - 2.0).abs() < 1e-14);
        assert!((epsilon * result[1] + result[2] - 1.0).abs() < 1e-14);
    }
}
