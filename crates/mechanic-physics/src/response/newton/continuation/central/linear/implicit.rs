//! Matrix-free direction when weak contact equations exceed the dense search cap.

use super::{ArcRow, NewtonWork, Point, Problem, dot};

pub(super) fn direction(
    problem: &Problem<'_>,
    points: &[Point],
    count: usize,
    arc: Option<ArcRow<'_>>,
    local_storage: usize,
    work: &mut NewtonWork,
) -> Option<Vec<f64>> {
    // The bordered path retains one normal per point and is bounded before
    // entry. Extra weak tangent equations arise only in the finishing solve.
    if arc.is_some() {
        return None;
    }
    let axes = problem.size;
    let zero = vec![0.0; count];
    let constant = expand(problem, points, &zero, true);
    let mut rhs = vec![0.0; count];
    for (response, value) in problem.response.iter().zip(&constant) {
        for (out, coefficient) in rhs[..axes].iter_mut().zip(response) {
            *out += coefficient * value;
        }
    }
    for point in points {
        for row in point.rank..point.size {
            rhs[axes + point.offset + row - point.rank] = point.rhs[row];
        }
    }
    // Compute row equilibration one column at a time, without storing a square
    // contact or mixed matrix. The solve retains at most 128 orthogonal rows.
    let mut scales = vec![0.0_f64; count];
    for column in 0..count {
        let mut unit = vec![0.0; count];
        unit[column] = 1.0;
        let image = apply(problem, points, &unit);
        work.applications += 1;
        for (scale, coefficient) in scales.iter_mut().zip(image) {
            *scale = scale.max(coefficient.abs());
        }
    }
    for (value, scale) in rhs.iter_mut().zip(&mut scales) {
        if !scale.is_finite() {
            return None;
        }
        if *scale == 0.0 {
            *scale = 1.0;
        }
        *value /= *scale;
    }
    work.mixed_rows = work.mixed_rows.max(count);
    work.generalized_factorizations += 1;
    // QR basis, retained rows, and the minimum-norm basis coexist. Their
    // rectangular storage grows linearly with equation count at fixed rank.
    let rank = 128.min(count);
    work.reduced_storage = work
        .reduced_storage
        .max(local_storage + 3 * rank * count + rank * rank);
    let solution = least_squares(
        &rhs,
        |column| {
            let mut unit = vec![0.0; count];
            unit[column] = 1.0;
            let mut image = apply(problem, points, &unit);
            for (value, scale) in image.iter_mut().zip(&scales) {
                *value /= scale;
            }
            image
        },
        |value| {
            let weighted = value
                .iter()
                .zip(&scales)
                .map(|(value, scale)| value / scale)
                .collect::<Vec<_>>();
            transpose(problem, points, &weighted)
        },
        work,
    )?;
    let result = expand(problem, points, &solution, true);
    result
        .iter()
        .all(|value| value.is_finite())
        .then_some(result)
}

fn expand(problem: &Problem<'_>, points: &[Point], value: &[f64], constant: bool) -> Vec<f64> {
    let axes = problem.size;
    let mut result = vec![0.0; problem.rows.len()];
    for point in points {
        for row in 0..point.rank {
            let mut delta = if constant { point.rhs[row] } else { 0.0 }
                - dot(
                    &point.coupling[row * axes..(row + 1) * axes],
                    &value[..axes],
                );
            for column in point.rank..point.size {
                delta -= point.matrix[row * point.size + column]
                    * problem.units[point.first + point.permutation[column]]
                    * value[axes + point.offset + column - point.rank];
            }
            result[point.first + point.permutation[row]] = delta;
        }
        for row in point.rank..point.size {
            result[point.first + point.permutation[row]] = problem.units
                [point.first + point.permutation[row]]
                * value[axes + point.offset + row - point.rank];
        }
    }
    result
}

fn apply(problem: &Problem<'_>, points: &[Point], value: &[f64]) -> Vec<f64> {
    let axes = problem.size;
    let impulses = expand(problem, points, value, false);
    let mut result = value.to_vec();
    for (response, impulse) in problem.response.iter().zip(impulses) {
        for (out, coefficient) in result[..axes].iter_mut().zip(response) {
            *out -= coefficient * impulse;
        }
    }
    for point in points {
        for row in point.rank..point.size {
            let mut equation = dot(
                &point.coupling[row * axes..(row + 1) * axes],
                &value[..axes],
            );
            for column in point.rank..point.size {
                equation += point.matrix[row * point.size + column]
                    * problem.units[point.first + point.permutation[column]]
                    * value[axes + point.offset + column - point.rank];
            }
            result[axes + point.offset + row - point.rank] = equation;
        }
    }
    result
}

fn magnitude(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |sum, value| sum.hypot(*value))
}

fn transpose(problem: &Problem<'_>, points: &[Point], value: &[f64]) -> Vec<f64> {
    let axes = problem.size;
    let mut result = vec![0.0; value.len()];
    result[..axes].copy_from_slice(&value[..axes]);
    for point in points {
        for row in 0..point.rank {
            let adjoint = -dot(
                &problem.response[point.first + point.permutation[row]],
                &value[..axes],
            );
            for (axis, out) in result[..axes].iter_mut().enumerate() {
                *out -= point.coupling[row * axes + axis] * adjoint;
            }
            for column in point.rank..point.size {
                result[axes + point.offset + column - point.rank] -= point.matrix
                    [row * point.size + column]
                    * problem.units[point.first + point.permutation[column]]
                    * adjoint;
            }
        }
        for row in point.rank..point.size {
            let equation = axes + point.offset + row - point.rank;
            result[equation] -= problem.units[point.first + point.permutation[row]]
                * dot(
                    &problem.response[point.first + point.permutation[row]],
                    &value[..axes],
                );
            for (axis, out) in result[..axes].iter_mut().enumerate() {
                *out += point.coupling[row * axes + axis] * value[equation];
            }
            for column in point.rank..point.size {
                result[axes + point.offset + column - point.rank] += point.matrix
                    [row * point.size + column]
                    * problem.units[point.first + point.permutation[column]]
                    * value[equation];
            }
        }
    }
    result
}

fn residual_column(mut column: Vec<f64>, basis: &[Vec<f64>]) -> Vec<f64> {
    for _ in 0..2 {
        for axis in basis {
            let component = dot(&column, axis);
            for (value, coefficient) in column.iter_mut().zip(axis) {
                *value -= component * coefficient;
            }
        }
    }
    column
}

fn least_squares(
    rhs: &[f64],
    mut column: impl FnMut(usize) -> Vec<f64>,
    mut transpose: impl FnMut(&[f64]) -> Vec<f64>,
    work: &mut NewtonWork,
) -> Option<Vec<f64>> {
    let size = rhs.len();
    let mut norms = (0..size)
        .map(|index| {
            work.applications += 1;
            magnitude(&column(index))
        })
        .collect::<Vec<_>>();
    let scale = norms.iter().copied().fold(0.0, f64::max);
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut rows = Vec::new();
    let mut targets = Vec::new();
    let mut selected = vec![false; size];
    for _ in 0..128.min(size) {
        let mut chosen = None;
        for _ in 0..size {
            let index = (0..size)
                .filter(|i| !selected[*i])
                .max_by(|a, b| norms[*a].total_cmp(&norms[*b]).then_with(|| b.cmp(a)))?;
            work.applications += 1;
            let vector = residual_column(column(index), &basis);
            let length = magnitude(&vector);
            norms[index] = length;
            let other = (0..size)
                .filter(|i| !selected[*i] && *i != index)
                .map(|i| norms[i])
                .fold(0.0, f64::max);
            if length >= other * (1.0 - 1e-12) {
                chosen = Some((index, vector, length));
                break;
            }
        }
        let (index, mut vector, length) = chosen?;
        if length <= 32.0 * f64::EPSILON * scale {
            break;
        }
        selected[index] = true;
        for value in &mut vector {
            *value /= length;
        }
        work.applications += 1;
        let row = transpose(&vector);
        targets.push(dot(&vector, rhs));
        for (norm, value) in norms.iter_mut().zip(&row) {
            *norm = (norm.powi(2) - value.powi(2)).max(0.0).sqrt();
        }
        rows.extend(row);
        basis.push(vector);
        if norms.iter().copied().fold(0.0, f64::max) <= scale * 1e-7 {
            for index in 0..size {
                if !selected[index] {
                    work.applications += 1;
                    norms[index] = magnitude(&residual_column(column(index), &basis));
                }
            }
        }
    }
    super::super::super::rank::minimum_norm_rows(&rows, &targets, size, basis.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn implicit_mixed_transpose_matches_every_independent_matrix_entry() {
        let rows = vec![vec![0.2, 0.1]; 6];
        let problem = Problem {
            size: 2,
            rows: rows.iter().collect(),
            response: rows.clone(),
            targets: vec![0.0; 6],
            scales: vec![1.0; 6],
            units: vec![2.0; 6],
            points: vec![],
        };
        let points = [
            Point {
                first: 0,
                size: 3,
                rank: 1,
                offset: 0,
                permutation: vec![1, 0, 2],
                matrix: vec![1.0, 0.2, -0.1, 0.0, 0.3, 0.4, 0.0, 0.1, 0.2],
                coupling: vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.2],
                rhs: vec![0.1; 3],
            },
            Point {
                first: 3,
                size: 3,
                rank: 2,
                offset: 2,
                permutation: vec![2, 1, 0],
                matrix: vec![1.0, 0.0, 0.1, 0.0, 1.0, 0.2, 0.0, 0.0, 0.3],
                coupling: vec![0.3, -0.1, 0.2, 0.6, -0.5, 0.7],
                rhs: vec![-0.2; 3],
            },
        ];
        for column in 0..5 {
            let mut unit = vec![0.0; 5];
            unit[column] = 1.0;
            let image = apply(&problem, &points, &unit);
            for row in 0..5 {
                let mut unit = vec![0.0; 5];
                unit[row] = 1.0;
                let transposed = transpose(&problem, &points, &unit);
                assert!((image[row] - transposed[column]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn streamed_rank_revealing_solve_preserves_redundant_equations_and_minimum_norm() {
        let rhs = (0..144)
            .map(|row| if row < 48 { 1.0 } else { 3.0 })
            .collect::<Vec<_>>();
        let result = least_squares(
            &rhs,
            |column| {
                (0..144)
                    .map(|row| f64::from((row < 48) == (column < 48)))
                    .collect()
            },
            |value| {
                let first = value[..48].iter().sum::<f64>();
                let second = value[48..].iter().sum::<f64>();
                (0..144)
                    .map(|row| if row < 48 { first } else { second })
                    .collect()
            },
            &mut NewtonWork::default(),
        )
        .unwrap();
        for (index, value) in result.iter().enumerate() {
            let expected = if index < 48 { 1.0 / 48.0 } else { 3.0 / 96.0 };
            assert!((value - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn streamed_full_rank_search_retains_only_its_fixed_128_directions() {
        let result = least_squares(
            &[1.0; 144],
            |column| (0..144).map(|row| f64::from(row == column)).collect(),
            <[f64]>::to_vec,
            &mut NewtonWork::default(),
        )
        .unwrap();
        assert_eq!(&result[..128], &[1.0; 128]);
        assert_eq!(&result[128..], &[0.0; 16]);
    }
}
