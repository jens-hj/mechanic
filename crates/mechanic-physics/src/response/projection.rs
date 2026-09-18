//! Projections onto bounds and friction disks, residuals, and rank-revealing solves.

use super::constraints::{BlockLayout, ConstraintBlock};
use super::factor::DynamicsFactor;
use crate::PhysicsError;

pub(super) fn response_motion(
    factor: &DynamicsFactor,
    rows: &[&Vec<f64>],
    impulses: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), PhysicsError> {
    let mut change = vec![0.0; factor.size];
    for (row, &impulse) in rows.iter().zip(impulses) {
        for (out, &j) in change.iter_mut().zip(*row) {
            *out += j * impulse;
        }
    }
    factor.solve(&mut change)?;
    let row_change = rows.iter().map(|row| dot(row, &change)).collect();
    Ok((change, row_change))
}

pub(super) fn stalled_contact(
    blocks: &[ConstraintBlock],
    layout: &[BlockLayout],
    row_change: &[f64],
    impulses: &[f64],
) -> Option<(usize, usize)> {
    let mut selected: Option<(usize, usize, f64)> = None;
    for (index, (block, info)) in blocks.iter().zip(layout).enumerate() {
        let mut local = 0;
        for contact in &block.contacts {
            let row = info.first + local;
            // Positive slack with load violates normal complementarity. Ties
            // retain the first feature in the deterministic constraint ordering.
            // A positive slack below the convergence tolerance can still cause a
            // larger coupled violation elsewhere. It may propose an inactive row;
            // only the unchanged residual over all original rows may accept it.
            let slack = row_change[row] - block.target[local];
            if impulses[row] > 0.0
                && slack > 0.0
                && selected.is_none_or(|(_, _, previous)| slack > previous)
            {
                selected = Some((index, local, slack));
            }
            local += contact.rows();
        }
    }
    selected.map(|(block, row, _)| (block, row))
}

pub(super) fn projected_residual(
    blocks: &[ConstraintBlock],
    layout: &[BlockLayout],
    row_change: &[f64],
    impulses: &[f64],
    sliding: &[Vec<bool>],
    tolerance: f64,
) -> Result<f64, PhysicsError> {
    let mut residual = 0.0_f64;
    for ((block, info), modes) in blocks.iter().zip(layout).zip(sliding) {
        let size = block.jacobian.len();
        let rhs = (0..size)
            .map(|row| block.target[row] - row_change[info.first + row])
            .collect::<Vec<_>>();
        let delta = block_delta(block, &info.diagonal, info.scale, &rhs, tolerance)?;
        let mut candidate = (0..size)
            .map(|row| impulses[info.first + row] + delta[row])
            .collect::<Vec<_>>();
        project(block, modes, &mut candidate);
        for row in 0..size {
            let error = if is_bilateral(block) {
                rhs[row]
            } else {
                info.scale * (candidate[row] - impulses[info.first + row])
            };
            if !error.is_finite() {
                return Err(PhysicsError::InvalidDynamics);
            }
            residual = residual.max(error.abs());
        }
    }
    Ok(residual)
}

pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

pub(super) fn is_bilateral(block: &ConstraintBlock) -> bool {
    block.contacts.is_empty()
        && block
            .bounds
            .iter()
            .all(|bounds| bounds.minimum == f64::NEG_INFINITY && bounds.maximum == f64::INFINITY)
}

pub(super) fn block_scale(matrix: &[f64], size: usize) -> f64 {
    matrix
        .chunks_exact(size)
        .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0, f64::max)
        .max(1e-30)
}

pub(super) fn block_delta(
    block: &ConstraintBlock,
    matrix: &[f64],
    scale: f64,
    rhs: &[f64],
    tolerance: f64,
) -> Result<Vec<f64>, PhysicsError> {
    if is_bilateral(block) {
        return rank_solve(matrix, rhs, tolerance);
    }
    // Reuse the pose-local spectral-radius bound. Clipping an unconstrained
    // W^-1 solution is not a bounded block solve: off-diagonal coupling can
    // falsely report convergence. The contact projection is non-associated
    // Coulomb (normal clamp followed by disks), not an associated cone QP.
    Ok(rhs.iter().map(|value| value / scale).collect())
}

pub(super) fn valid_contacts(block: &ConstraintBlock) -> bool {
    if block.contacts.is_empty() {
        return true;
    }
    let mut row = 0;
    for law in &block.contacts {
        if !law.static_coefficient.is_finite()
            || !law.kinetic_coefficient.is_finite()
            || law.kinetic_coefficient < 0.0
            || law.static_coefficient < law.kinetic_coefficient
            || law
                .rolling_length
                .is_some_and(|length| !length.is_finite() || length < 0.0)
        {
            return false;
        }
        let count = law.rows();
        let Some(bounds) = block.bounds.get(row..row + count) else {
            return false;
        };
        if bounds[0].minimum != 0.0
            || bounds[1..]
                .iter()
                .any(|b| b.minimum != f64::NEG_INFINITY || b.maximum != f64::INFINITY)
        {
            return false;
        }
        row += count;
    }
    row == block.jacobian.len()
}

pub(super) fn promote_sliding(
    blocks: &[ConstraintBlock],
    layout: &[BlockLayout],
    row_change: &[f64],
    impulses: &[f64],
    modes: &mut [Vec<bool>],
    tolerance: f64,
) -> usize {
    let mut changed = 0;
    for ((block, info), modes) in blocks.iter().zip(layout).zip(modes) {
        let mut row = 0;
        for (law, sliding) in block.contacts.iter().zip(modes) {
            let remaining_u = block.target[row + 1] - row_change[info.first + row + 1];
            let remaining_v = block.target[row + 2] - row_change[info.first + row + 2];
            if !*sliding
                && impulses[info.first + row] > 0.0
                && remaining_u.hypot(remaining_v) > tolerance * 4.0
            {
                *sliding = true;
                changed += 1;
            }
            row += law.rows();
        }
    }
    changed
}

pub(super) fn project(block: &ConstraintBlock, sliding: &[bool], values: &mut [f64]) {
    for (value, bounds) in values.iter_mut().zip(&block.bounds) {
        *value = value.clamp(bounds.minimum, bounds.maximum);
    }
    let mut row = 0;
    for (law, &sliding) in block.contacts.iter().zip(sliding) {
        let point = &mut values[row..row + law.rows()];
        let coefficient = if sliding {
            law.kinetic_coefficient
        } else {
            law.static_coefficient
        };
        let normal = point[0].max(0.0);
        project_disk(&mut point[1..3], coefficient * normal);
        if let Some(length) = law.rolling_length {
            project_disk(&mut point[3..5], length * normal);
        }
        row += law.rows();
    }
}

pub(super) fn project_disk(pair: &mut [f64], radius: f64) {
    let length = pair[0].hypot(pair[1]);
    if length > radius {
        pair[0] *= radius / length;
        pair[1] *= radius / length;
    }
}

// Deterministic complete diagonal pivoting for a positive-semidefinite block.
// Dependent rows are retained and checked for consistency, never regularized.
pub(super) fn rank_solve(
    matrix: &[f64],
    rhs: &[f64],
    tolerance: f64,
) -> Result<Vec<f64>, PhysicsError> {
    let n = rhs.len();
    if n == 1 {
        if matrix[0] > 0.0 {
            return Ok(vec![rhs[0] / matrix[0]]);
        }
        if rhs[0].abs() <= tolerance {
            return Ok(vec![0.0]);
        }
        return Err(PhysicsError::InconsistentConstraints);
    }
    let mut a = matrix.to_vec();
    let mut b = rhs.to_vec();
    let mut permutation = (0..n).collect::<Vec<_>>();
    let scale = (0..n).map(|i| a[i * n + i].abs()).fold(0.0, f64::max);
    let mut rank = n;
    for k in 0..n {
        let mut pivot = k;
        for i in k + 1..n {
            if a[i * n + i] > a[pivot * n + pivot] {
                pivot = i;
            }
        }
        if a[pivot * n + pivot] <= scale * 1e-12 {
            rank = k;
            if b[k..].iter().any(|x| x.abs() > tolerance) {
                return Err(PhysicsError::InconsistentConstraints);
            }
            break;
        }
        for j in 0..n {
            a.swap(k * n + j, pivot * n + j);
        }
        for i in 0..n {
            a.swap(i * n + k, i * n + pivot);
        }
        b.swap(k, pivot);
        permutation.swap(k, pivot);
        for i in k + 1..n {
            let multiplier = a[i * n + k] / a[k * n + k];
            for j in k + 1..n {
                a[i * n + j] -= multiplier * a[k * n + j];
            }
            b[i] -= multiplier * b[k];
            a[i * n + k] = 0.0;
        }
    }
    let mut solution = vec![0.0; n];
    for i in (0..rank).rev() {
        solution[i] = (b[i]
            - (i + 1..rank)
                .map(|j| a[i * n + j] * solution[j])
                .sum::<f64>())
            / a[i * n + i];
    }
    let mut output = vec![0.0; n];
    for i in 0..n {
        output[permutation[i]] = solution[i];
    }
    Ok(output)
}
