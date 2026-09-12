//! Bounded local elimination and a bordered machine system for path following.

use super::super::super::NewtonWork;
use super::{ArcRow, Problem, dot, equations::Evaluation};

mod implicit;

struct Point {
    first: usize,
    size: usize,
    rank: usize,
    offset: usize,
    permutation: Vec<usize>,
    matrix: Vec<f64>,
    coupling: Vec<f64>,
    rhs: Vec<f64>,
}

#[allow(clippy::too_many_lines)]
pub(super) fn direction(
    problem: &Problem<'_>,
    at: &Evaluation,
    stable: bool,
    arc: Option<ArcRow<'_>>,
    work: &mut NewtonWork,
) -> Option<Vec<f64>> {
    let axes = problem.size + usize::from(arc.is_some());
    let mut retained = 0;
    let mut points = Vec::new();
    for local in &at.locals {
        work.local_factorizations += 1;
        let width = local.size;
        let mut point = Point {
            first: local.first,
            size: width,
            rank: 0,
            offset: retained,
            permutation: (0..width).collect(),
            matrix: local.matrix.clone(),
            coupling: vec![0.0; width * axes],
            rhs: local.rhs.clone(),
        };
        if arc.is_some_and(|arc| arc.homogeneous) {
            point.rhs.fill(0.0);
        }
        for row in 0..width {
            point.coupling[row * axes..row * axes + problem.size]
                .copy_from_slice(&local.velocity[row * problem.size..(row + 1) * problem.size]);
            if arc.is_some() {
                point.coupling[row * axes + problem.size] = local.parameter[row];
            }
            let scale = point.matrix[row * width..(row + 1) * width]
                .iter()
                .chain(&point.coupling[row * axes..(row + 1) * axes])
                .map(|v| v.abs())
                .fold(0.0, f64::max);
            if scale <= 0.0 || !scale.is_finite() {
                return None;
            }
            for value in &mut point.matrix[row * width..(row + 1) * width] {
                *value /= scale;
            }
            for value in &mut point.coupling[row * axes..(row + 1) * axes] {
                *value /= scale;
            }
            point.rhs[row] /= scale;
        }
        for column in 0..width {
            let mut pivot = None;
            let mut largest = 0.0;
            for candidate in column..width {
                // Retaining each normal avoids eliminating the small barrier
                // normal diagonal while following folds in the central path.
                if !stable && point.permutation[candidate] == 0 {
                    continue;
                }
                for row in column..width {
                    let value = point.matrix[row * width + candidate].abs();
                    if value > largest {
                        largest = value;
                        pivot = Some((row, candidate));
                    }
                }
            }
            if largest <= 1e-8 {
                break;
            }
            let (pivot_row, pivot_column) = pivot?;
            for index in 0..width {
                point
                    .matrix
                    .swap(column * width + index, pivot_row * width + index);
            }
            for index in 0..axes {
                point
                    .coupling
                    .swap(column * axes + index, pivot_row * axes + index);
            }
            point.rhs.swap(column, pivot_row);
            for row in 0..width {
                point
                    .matrix
                    .swap(row * width + column, row * width + pivot_column);
            }
            point.permutation.swap(column, pivot_column);
            for row in column + 1..width {
                let multiplier =
                    point.matrix[row * width + column] / point.matrix[column * width + column];
                point.matrix[row * width + column] = 0.0;
                for index in column + 1..width {
                    point.matrix[row * width + index] -=
                        multiplier * point.matrix[column * width + index];
                }
                for index in 0..axes {
                    point.coupling[row * axes + index] -=
                        multiplier * point.coupling[column * axes + index];
                }
                point.rhs[row] -= multiplier * point.rhs[column];
            }
            point.rank = column + 1;
        }
        for row in (0..point.rank).rev() {
            for next in row + 1..point.rank {
                let multiplier = point.matrix[row * width + next];
                point.rhs[row] -= multiplier * point.rhs[next];
                for index in 0..axes {
                    point.coupling[row * axes + index] -=
                        multiplier * point.coupling[next * axes + index];
                }
                for index in point.rank..width {
                    point.matrix[row * width + index] -=
                        multiplier * point.matrix[next * width + index];
                }
            }
            let diagonal = point.matrix[row * width + row];
            point.rhs[row] /= diagonal;
            for index in 0..axes {
                point.coupling[row * axes + index] /= diagonal;
            }
            for index in point.rank..width {
                point.matrix[row * width + index] /= diagonal;
            }
        }
        retained += width - point.rank;
        points.push(point);
    }
    let count = axes + retained;
    // The evaluation and its eliminated copy coexist with the cached inverse
    // dynamics columns throughout either linear solve.
    let local_storage = problem.response.iter().map(Vec::len).sum::<usize>()
        + at.locals
            .iter()
            .map(|local| local.matrix.len() + local.velocity.len())
            .sum::<usize>()
        + points
            .iter()
            .map(|point| point.matrix.len() + point.coupling.len())
            .sum::<usize>();
    if count > 128 {
        return implicit::direction(problem, &points, count, arc, local_storage, work);
    }
    let mut matrix = vec![0.0; count * count];
    let mut rhs = vec![0.0; count];
    let mut expansion = vec![0.0; problem.rows.len() * count];
    let mut constant = vec![0.0; problem.rows.len()];
    for point in &points {
        for row in 0..point.rank {
            let original = point.first + point.permutation[row];
            constant[original] = point.rhs[row];
            for axis in 0..axes {
                expansion[original * count + axis] = -point.coupling[row * axes + axis];
            }
            for column in point.rank..point.size {
                let retained_column = axes + point.offset + column - point.rank;
                let unit = problem.units[point.first + point.permutation[column]];
                expansion[original * count + retained_column] =
                    -point.matrix[row * point.size + column] * unit;
            }
        }
        for row in point.rank..point.size {
            let original = point.first + point.permutation[row];
            let equation = axes + point.offset + row - point.rank;
            expansion[original * count + equation] = problem.units[original];
            rhs[equation] = point.rhs[row];
            for axis in 0..axes {
                matrix[equation * count + axis] = point.coupling[row * axes + axis];
            }
            for column in point.rank..point.size {
                matrix[equation * count + axes + point.offset + column - point.rank] = point.matrix
                    [row * point.size + column]
                    * problem.units[point.first + point.permutation[column]];
            }
        }
    }
    for axis in 0..problem.size {
        matrix[axis * count + axis] = 1.0;
        for (row, response) in problem.response.iter().enumerate() {
            rhs[axis] += response[axis] * constant[row];
            for column in 0..count {
                matrix[axis * count + column] -= response[axis] * expansion[row * count + column];
            }
        }
    }
    if let Some(arc) = arc {
        let equation = problem.size;
        rhs[equation] = arc.rhs;
        matrix[equation * count + equation] = arc.tangent[problem.rows.len()];
        for row in 0..problem.rows.len() {
            let component = arc.tangent[row] / problem.units[row];
            rhs[equation] -= component * constant[row];
            for column in 0..count {
                matrix[equation * count + column] += component * expansion[row * count + column];
            }
        }
    }
    for row in 0..count {
        let scale = matrix[row * count..(row + 1) * count]
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        if scale <= 0.0 || !scale.is_finite() {
            return None;
        }
        for value in &mut matrix[row * count..(row + 1) * count] {
            *value /= scale;
        }
        rhs[row] /= scale;
    }
    let storage = matrix.len() + expansion.len() + local_storage;
    work.mixed_rows = work.mixed_rows.max(count);
    work.generalized_factorizations += 1;
    let solution = super::super::rank::least_squares(&matrix, &rhs, storage, work)?;
    let mut result = constant
        .iter()
        .enumerate()
        .map(|(row, value)| value + dot(&expansion[row * count..(row + 1) * count], &solution))
        .collect::<Vec<_>>();
    if arc.is_some() {
        for (value, unit) in result.iter_mut().zip(&problem.units) {
            *value /= unit;
        }
        result.push(solution[problem.size]);
    }
    result.iter().all(|v| v.is_finite()).then_some(result)
}
