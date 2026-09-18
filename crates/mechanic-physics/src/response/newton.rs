//! Bounded Newton proposals for the unchanged projected contact equations.
//!
//! A dimensionless diagonal shift stabilizes only the search direction when
//! redundant support rows make its Jacobian singular. Acceptance and final
//! convergence always use the original unshifted contact equations.

use super::{BlockLayout, ConstraintBlock, DynamicsFactor, project};

pub(super) mod continuation;
mod projection;
mod reduced;

const KRYLOV_ROWS: usize = 32;

#[derive(Default)]
pub(super) struct NewtonWork {
    pub direction: Option<Vec<f64>>,
    pub applications: usize,
    pub local_factorizations: usize,
    pub generalized_factorizations: usize,
    pub factor_solves: usize,
    pub reduced_storage: usize,
    pub mixed_rows: usize,
    pub full_rows: usize,
    pub contact_factorizations: usize,
    pub contact_storage: usize,
}

#[expect(
    clippy::too_many_arguments,
    reason = "the same immutable Newton linearization for both bounded execution routes"
)]
pub(super) fn direction(
    factor: &DynamicsFactor,
    blocks: &[ConstraintBlock],
    layout: &[BlockLayout],
    impulses: &[f64],
    row_change: &[f64],
    modes: &[Vec<bool>],
    shift: f64,
    unshifted: bool,
    mut response: impl FnMut(&[f64]) -> Option<Vec<f64>>,
) -> NewtonWork {
    let mut at = impulses.to_vec();
    let mut steps = vec![0.0; impulses.len()];
    for (block, info) in blocks.iter().zip(layout) {
        let step = info.scale.recip();
        for row in 0..block.jacobian.len() {
            let index = info.first + row;
            steps[index] = step;
            at[index] += step * (block.target[row] - row_change[index]);
        }
    }
    let mut projected = at.clone();
    for ((block, info), modes) in blocks.iter().zip(layout).zip(modes) {
        project(
            block,
            modes,
            &mut projected[info.first..info.first + block.jacobian.len()],
        );
    }
    let rhs = projected
        .iter()
        .zip(impulses)
        .map(|(value, old)| value - old)
        .collect::<Vec<_>>();
    let derivatives = projection::prepare(blocks, layout, modes, &at);
    if impulses.len() <= super::DENSE_CONTACT_ROWS {
        return contact_direction(
            &derivatives,
            &steps,
            &rhs,
            if unshifted { 0.0 } else { shift },
            &mut response,
        );
    }
    if factor.size <= reduced::MAXIMUM_COORDINATES {
        return reduced::direction(factor, blocks, &derivatives, &steps, &rhs, shift);
    }
    let preconditioner = point_preconditioner(blocks, layout, &derivatives, &steps, shift);
    let mut applications = 0;
    let direction = gmres(&rhs, |krylov| {
        applications += 1;
        let mut trial = precondition(&preconditioner, krylov)?;
        let mut input = response(&trial)?;
        for ((out, value), step) in input.iter_mut().zip(&trial).zip(&steps) {
            *out = value - step * *out;
        }
        for point in &derivatives {
            let mut derivative = [0.0; 5];
            point.apply(
                &input[point.first..point.first + point.size],
                &mut derivative[..point.size],
            );
            for (row, value) in derivative[..point.size].iter().enumerate() {
                let index = point.first + row;
                trial[index] = (1.0 + shift) * trial[index] - value;
            }
        }
        trial.iter().all(|v| v.is_finite()).then_some(trial)
    })
    .and_then(|value| precondition(&preconditioner, &value));
    NewtonWork {
        direction,
        applications,
        local_factorizations: preconditioner.len(),
        ..NewtonWork::default()
    }
}

// Small contact systems already have bounded explicit response storage. Solve
// their Newton rows directly, avoiding cancellation through A^-1 when the
// projection derivative has unit eigenvalues. Complete pivoting handles
// dependent rows without adding a shift to these small systems.
fn contact_direction(
    derivatives: &[projection::PointDerivative],
    steps: &[f64],
    rhs: &[f64],
    shift: f64,
    response: &mut impl FnMut(&[f64]) -> Option<Vec<f64>>,
) -> NewtonWork {
    let size = rhs.len();
    let mut work = NewtonWork {
        contact_storage: size * size,
        ..Default::default()
    };
    work.direction = (|| {
        let mut matrix = vec![0.0; size * size];
        let mut unit = vec![0.0; size];
        for column in 0..size {
            unit[column] = 1.0;
            let mut input = response(&unit)?;
            for row in 0..size {
                input[row] = unit[row] - steps[row] * input[row];
            }
            for point in derivatives {
                let mut projected = [0.0; 5];
                point.apply(
                    &input[point.first..point.first + point.size],
                    &mut projected[..point.size],
                );
                for (local, value) in projected[..point.size].iter().enumerate() {
                    let row = point.first + local;
                    matrix[row * size + column] = (1.0 + shift) * unit[row] - value;
                }
            }
            unit[column] = 0.0;
        }
        work.contact_factorizations += 1;
        if shift == 0.0 {
            // Complete pivoting retains the original matrix for an independent
            // linear residual check while eliminating its mutable copy.
            work.contact_storage = 2 * size * size;
            rank_direction(&matrix, rhs)
        } else {
            let (matrix, pivots) = factor_dense(matrix, size)?;
            let mut direction = rhs.to_vec();
            apply_lu(&matrix, &pivots, &mut direction)?;
            Some(direction)
        }
    })();
    work
}

// Complete pivoting keeps weak but independent contact directions while removing
// dependent rows. This is only a Newton proposal; the original projected physical
// residual still controls acceptance and publication.
fn rank_direction(matrix: &[f64], rhs: &[f64]) -> Option<Vec<f64>> {
    let size = rhs.len();
    let mut a = matrix.to_vec();
    let mut b = rhs.to_vec();
    let mut permutation = (0..size).collect::<Vec<_>>();
    let scale = a.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let mut rank = size;
    for column in 0..size {
        let mut pivot = (column, column);
        for row in column..size {
            for index in column..size {
                if a[row * size + index].abs() > a[pivot.0 * size + pivot.1].abs() {
                    pivot = (row, index);
                }
            }
        }
        if a[pivot.0 * size + pivot.1].abs() <= 32.0 * f64::EPSILON * scale {
            rank = column;
            break;
        }
        for index in 0..size {
            a.swap(column * size + index, pivot.0 * size + index);
        }
        b.swap(column, pivot.0);
        for row in 0..size {
            a.swap(row * size + column, row * size + pivot.1);
        }
        permutation.swap(column, pivot.1);
        for row in column + 1..size {
            let multiplier = a[row * size + column] / a[column * size + column];
            for index in column + 1..size {
                a[row * size + index] -= multiplier * a[column * size + index];
            }
            b[row] -= multiplier * b[column];
            a[row * size + column] = 0.0;
        }
    }
    let mut solution = vec![0.0; size];
    for row in (0..rank).rev() {
        solution[row] = (b[row]
            - (row + 1..rank)
                .map(|c| a[row * size + c] * solution[c])
                .sum::<f64>())
            / a[row * size + row];
    }
    let mut output = vec![0.0; size];
    for row in 0..size {
        output[permutation[row]] = solution[row];
    }
    let rhs_scale = rhs.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let residual = matrix
        .chunks_exact(size)
        .zip(rhs)
        .map(|(row, value)| {
            (row.iter().zip(&output).map(|(a, b)| a * b).sum::<f64>() - value).abs()
        })
        .fold(0.0, f64::max);
    (output.iter().all(|v| v.is_finite()) && residual <= 1e-8 * rhs_scale.max(1e-30))
        .then_some(output)
}

struct PointFactor {
    first: usize,
    size: usize,
    lu: Option<(Vec<f64>, Vec<usize>)>,
}

fn point_preconditioner(
    blocks: &[ConstraintBlock],
    layout: &[BlockLayout],
    derivatives: &[projection::PointDerivative],
    steps: &[f64],
    shift: f64,
) -> Vec<PointFactor> {
    let mut result = Vec::with_capacity(derivatives.len());
    for point in derivatives {
        let info = &layout[point.block];
        let size = blocks[point.block].jacobian.len();
        let first = point.first - info.first;
        let width = point.size;
        let mut matrix = vec![0.0; width * width];
        for column in 0..width {
            let mut input = [0.0; 5];
            let mut projected = [0.0; 5];
            for (row, value) in input[..width].iter_mut().enumerate() {
                *value = f64::from(row == column)
                    - steps[point.first + row]
                        * info.diagonal[(first + row) * size + first + column];
            }
            point.apply(&input[..width], &mut projected[..width]);
            for row in 0..width {
                matrix[row * width + column] =
                    (1.0 + shift) * f64::from(row == column) - projected[row];
            }
        }
        result.push(PointFactor {
            first: point.first,
            size: width,
            lu: factor_dense(matrix, width),
        });
    }
    result
}

// Pivoted LU for local diagonals and the bounded generalized Newton system.
// Singular point factors use identity only for Krylov preconditioning; exact
// generalized elimination rejects any singular factor.
fn factor_dense(mut matrix: Vec<f64>, size: usize) -> Option<(Vec<f64>, Vec<usize>)> {
    let scale = matrix.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let mut pivots = Vec::with_capacity(size);
    for column in 0..size {
        let mut pivot = column;
        for row in column + 1..size {
            if matrix[row * size + column].abs() > matrix[pivot * size + column].abs() {
                pivot = row;
            }
        }
        if matrix[pivot * size + column].abs() <= scale * 1e-12 {
            return None;
        }
        pivots.push(pivot);
        for index in 0..size {
            matrix.swap(column * size + index, pivot * size + index);
        }
        for row in column + 1..size {
            matrix[row * size + column] /= matrix[column * size + column];
            for index in column + 1..size {
                matrix[row * size + index] -=
                    matrix[row * size + column] * matrix[column * size + index];
            }
        }
    }
    matrix
        .iter()
        .all(|value| value.is_finite())
        .then_some((matrix, pivots))
}

fn precondition(factors: &[PointFactor], rhs: &[f64]) -> Option<Vec<f64>> {
    let mut result = rhs.to_vec();
    for factor in factors {
        let Some((matrix, pivots)) = &factor.lu else {
            continue;
        };
        let size = factor.size;
        let values = &mut result[factor.first..factor.first + size];
        apply_lu(matrix, pivots, values)?;
    }
    result
        .iter()
        .all(|value| value.is_finite())
        .then_some(result)
}

fn apply_lu(matrix: &[f64], pivots: &[usize], values: &mut [f64]) -> Option<()> {
    let size = values.len();
    for (row, &pivot) in pivots.iter().enumerate() {
        values.swap(row, pivot);
    }
    for row in 0..size {
        for column in 0..row {
            values[row] -= matrix[row * size + column] * values[column];
        }
    }
    for row in (0..size).rev() {
        for column in row + 1..size {
            values[row] -= matrix[row * size + column] * values[column];
        }
        values[row] /= matrix[row * size + row];
    }
    values.iter().all(|value| value.is_finite()).then_some(())
}

#[cfg(test)]
fn projection_derivative(
    block: &ConstraintBlock,
    modes: &[bool],
    at: &[f64],
    input: &[f64],
    output: &mut [f64],
) {
    for (row, bounds) in block.bounds.iter().enumerate() {
        output[row] = if at[row] > bounds.minimum && at[row] < bounds.maximum {
            input[row]
        } else {
            0.0
        };
    }
    let mut first = 0;
    for (law, &sliding) in block.contacts.iter().zip(modes) {
        let normal = at[first].clamp(block.bounds[first].minimum, block.bounds[first].maximum);
        let normal_derivative = output[first];
        let coefficient = if sliding {
            law.kinetic_coefficient
        } else {
            law.static_coefficient
        };
        disk_derivative(
            &at[first + 1..first + 3],
            &input[first + 1..first + 3],
            coefficient * normal,
            coefficient * normal_derivative,
            &mut output[first + 1..first + 3],
        );
        if let Some(length) = law.rolling_length {
            disk_derivative(
                &at[first + 3..first + 5],
                &input[first + 3..first + 5],
                length * normal,
                length * normal_derivative,
                &mut output[first + 3..first + 5],
            );
        }
        first += law.rows();
    }
}

#[cfg(test)]
fn disk_derivative(
    at: &[f64],
    input: &[f64],
    radius: f64,
    radius_derivative: f64,
    output: &mut [f64],
) {
    let length = at[0].hypot(at[1]);
    if length < radius {
        output.copy_from_slice(input);
    } else if length > 0.0 {
        let unit = [at[0] / length, at[1] / length];
        let radial = unit[0] * input[0] + unit[1] * input[1];
        for row in 0..2 {
            output[row] =
                radius / length * (input[row] - unit[row] * radial) + unit[row] * radius_derivative;
        }
    } else {
        output.fill(0.0);
    }
}

// Restart-free bounded GMRES with two-pass Arnoldi orthogonalization and Givens
// rotations. Workspace is O(32 * constraint_rows), not a dense contact matrix.
fn gmres(rhs: &[f64], mut apply: impl FnMut(&[f64]) -> Option<Vec<f64>>) -> Option<Vec<f64>> {
    let length = norm(rhs);
    if length == 0.0 || !length.is_finite() {
        return None;
    }
    let maximum = KRYLOV_ROWS.min(rhs.len());
    let mut basis = vec![rhs.iter().map(|value| value / length).collect::<Vec<_>>()];
    let mut hessenberg = vec![0.0; (maximum + 1) * maximum];
    let mut cosines = vec![0.0; maximum];
    let mut sines = vec![0.0; maximum];
    let mut transformed = vec![0.0; maximum + 1];
    transformed[0] = length;
    let mut count = 0;
    for column in 0..maximum {
        let mut vector = apply(&basis[column])?;
        for _ in 0..2 {
            for row in 0..=column {
                let component = dot(&basis[row], &vector);
                hessenberg[row * maximum + column] += component;
                for (value, axis) in vector.iter_mut().zip(&basis[row]) {
                    *value -= component * axis;
                }
            }
        }
        let remainder = norm(&vector);
        hessenberg[(column + 1) * maximum + column] = remainder;
        for row in 0..column {
            let a = hessenberg[row * maximum + column];
            let b = hessenberg[(row + 1) * maximum + column];
            hessenberg[row * maximum + column] = cosines[row] * a + sines[row] * b;
            hessenberg[(row + 1) * maximum + column] = -sines[row] * a + cosines[row] * b;
        }
        let diagonal = hessenberg[column * maximum + column];
        let hypotenuse = diagonal.hypot(hessenberg[(column + 1) * maximum + column]);
        if hypotenuse <= 1e-14 || !hypotenuse.is_finite() {
            break;
        }
        cosines[column] = diagonal / hypotenuse;
        sines[column] = hessenberg[(column + 1) * maximum + column] / hypotenuse;
        hessenberg[column * maximum + column] = hypotenuse;
        transformed[column + 1] = -sines[column] * transformed[column];
        transformed[column] *= cosines[column];
        count = column + 1;
        if transformed[column + 1].abs() <= length * 1e-4 || remainder <= 1e-14 {
            break;
        }
        basis.push(vector.iter().map(|value| value / remainder).collect());
    }
    if count == 0 {
        return None;
    }
    let mut coefficients = vec![0.0; count];
    for row in (0..count).rev() {
        let later = (row + 1..count)
            .map(|column| hessenberg[row * maximum + column] * coefficients[column])
            .sum::<f64>();
        coefficients[row] = (transformed[row] - later) / hessenberg[row * maximum + row];
    }
    let mut result = vec![0.0; rhs.len()];
    for (column, coefficient) in basis.iter().zip(coefficients) {
        for (out, value) in result.iter_mut().zip(column) {
            *out += coefficient * value;
        }
    }
    result.iter().all(|v| v.is_finite()).then_some(result)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(values: &[f64]) -> f64 {
    let mut scale = 0.0_f64;
    for value in values {
        if !value.is_finite() {
            return f64::NAN;
        }
        scale = scale.max(value.abs());
    }
    if scale == 0.0 {
        return 0.0;
    }
    // One scaled sum and square root preserves overflow/underflow protection
    // without a scalar libm hypot call for every row of every Krylov vector.
    scale
        * values
            .iter()
            .map(|value| (value / scale).powi(2))
            .sum::<f64>()
            .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContactFriction, ImpulseBounds};

    #[test]
    fn krylov_norm_handles_large_small_and_zero_vectors_without_overflow() {
        for values in [
            [3.0_f64, 4.0],
            [3e200, 4e200],
            [3e-200, 4e-200],
            [1e300, 1e-300],
        ] {
            let expected = values[0].hypot(values[1]);
            assert!((norm(&values) / expected - 1.0).abs() < 1e-15);
        }
        assert_eq!(norm(&[0.0; 3]).to_bits(), 0.0_f64.to_bits());
        assert!(!norm(&[f64::NAN]).is_finite());
        assert!(!norm(&[f64::INFINITY]).is_finite());
    }

    #[test]
    fn local_pivoted_factor_preserves_a_nonsymmetric_response_and_singular_rows() {
        let matrix = [0.0, 2.0, 1.0, 3.0, 1.0, -1.0, 1.0, -2.0, 4.0];
        let factors = [
            PointFactor {
                first: 0,
                size: 3,
                lu: factor_dense(matrix.to_vec(), 3),
            },
            PointFactor {
                first: 3,
                size: 1,
                lu: factor_dense(vec![0.0], 1),
            },
        ];
        assert!(factors[0].lu.is_some());
        assert!(factors[1].lu.is_none());
        let rhs = [4.0, 1.0, -2.0, 7.0];
        let solution = precondition(&factors, &rhs).unwrap();
        for (row, target) in matrix.chunks_exact(3).zip(rhs) {
            assert!((dot(row, &solution[..3]) - target).abs() < 1e-12);
        }
        assert_eq!(solution[3].to_bits(), 7.0_f64.to_bits());
    }

    #[test]
    fn cone_derivative_matches_differentiated_normal_tangent_and_rolling_projection() {
        let block = ConstraintBlock {
            jacobian: vec![vec![0.0]; 5],
            target: vec![0.0; 5],
            bounds: vec![
                ImpulseBounds {
                    minimum: 0.0,
                    maximum: f64::INFINITY,
                },
                ImpulseBounds {
                    minimum: f64::NEG_INFINITY,
                    maximum: f64::INFINITY,
                },
                ImpulseBounds {
                    minimum: f64::NEG_INFINITY,
                    maximum: f64::INFINITY,
                },
                ImpulseBounds {
                    minimum: f64::NEG_INFINITY,
                    maximum: f64::INFINITY,
                },
                ImpulseBounds {
                    minimum: f64::NEG_INFINITY,
                    maximum: f64::INFINITY,
                },
            ],
            contacts: vec![ContactFriction {
                static_coefficient: 0.8,
                kinetic_coefficient: 0.5,
                sliding: true,
                rolling_length: Some(0.1),
            }],
        };
        let at = [2.0, 3.0, 4.0, 5.0, -7.0];
        let direction = [0.3, -0.7, 0.2, 0.1, -0.5];
        let mut analytic = [0.0; 5];
        let layout = [BlockLayout {
            first: 0,
            diagonal: Vec::new(),
            scale: 1.0,
        }];
        projection::prepare(std::slice::from_ref(&block), &layout, &[vec![true]], &at)[0]
            .apply(&direction, &mut analytic);
        let epsilon = 1e-6;
        let mut plus = std::array::from_fn::<_, 5, _>(|row| at[row] + epsilon * direction[row]);
        let mut minus = std::array::from_fn::<_, 5, _>(|row| at[row] - epsilon * direction[row]);
        project(&block, &[true], &mut plus);
        project(&block, &[true], &mut minus);
        for row in 0..5 {
            assert!((analytic[row] - (plus[row] - minus[row]) / (2.0 * epsilon)).abs() < 1e-9);
        }
        for at in [
            at,
            [2.0, 0.1, 0.2, 0.01, 0.02],
            [0.0; 5],
            [-1.0, 1.0, 2.0, 3.0, 4.0],
        ] {
            for mode in [false, true] {
                let mut expected = [0.0; 5];
                projection_derivative(&block, &[mode], &at, &direction, &mut expected);
                let cached =
                    projection::prepare(std::slice::from_ref(&block), &layout, &[vec![mode]], &at);
                cached[0].apply(&direction, &mut analytic);
                assert_eq!(analytic.map(f64::to_bits), expected.map(f64::to_bits));
            }
        }
    }

    #[test]
    fn matrix_free_gmres_solves_a_nonsymmetric_system() {
        let matrix = [[3.0, 2.0, -1.0], [0.0, 1.0, 0.5], [1.0, 0.0, 2.0]];
        let rhs = [4.0, 1.0, -2.0];
        let solution = gmres(&rhs, |value| {
            Some(matrix.iter().map(|row| dot(row, value)).collect())
        })
        .unwrap();
        for (row, target) in matrix.iter().zip(rhs) {
            assert!((dot(row, &solution) - target).abs() < 1e-10);
        }
    }

    mod dense {
        use super::super::*;

        #[test]
        fn contact_newton_preserves_a_weak_direction_and_dependent_rows() {
            let matrix = [1.0, 1.0, 0.0, 1.0, 1.0 + 1e-10, 0.0, 2.0, 2.0, 0.0];
            let expected = [2.0, -3.0, 0.0];
            let rhs = matrix
                .chunks_exact(3)
                .map(|row| row.iter().zip(expected).map(|(a, b)| a * b).sum())
                .collect::<Vec<_>>();
            let actual = rank_direction(&matrix, &rhs).unwrap();
            assert!((actual[0] - expected[0]).abs() < 1e-5);
            assert!((actual[1] - expected[1]).abs() < 1e-5);
            let mut inconsistent = rhs;
            inconsistent[2] += 1.0;
            assert!(rank_direction(&matrix, &inconsistent).is_none());
        }
    }
}
