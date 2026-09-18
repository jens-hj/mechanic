//! Bounded dual active-set seed for normal contact inequalities. Friction remains
//! in the original solver and must independently converge before publication.

use super::{ConstraintBlock, DynamicsFactor, PhysicsError, dot};

pub(crate) struct Seed {
    pub impulses: Option<Vec<f64>>,
    pub pivots: usize,
    pub factor_solves: usize,
    pub basis_factorizations: usize,
    pub basis_storage: usize,
}

#[expect(
    clippy::too_many_lines,
    reason = "ordered primal/dual steps with one bounded active basis"
)]
pub(crate) fn seed(
    factor: &DynamicsFactor,
    blocks: &[ConstraintBlock],
    maximum_pivots: usize,
    tolerance: f64,
) -> Result<Seed, PhysicsError> {
    let mut result = Seed {
        impulses: None,
        pivots: 0,
        factor_solves: 0,
        basis_factorizations: 0,
        basis_storage: 0,
    };
    if factor.size > 64 {
        return Ok(result);
    }
    let mut rows = Vec::new();
    let mut first = 0;
    for block in blocks {
        let mut local = 0;
        for contact in &block.contacts {
            rows.push((first + local, &block.jacobian[local], block.target[local]));
            local += contact.rows();
        }
        if block.contacts.is_empty() {
            for (index, bounds) in block.bounds.iter().enumerate() {
                if bounds.minimum == 0.0 && bounds.maximum == f64::INFINITY {
                    rows.push((first + index, &block.jacobian[index], block.target[index]));
                }
            }
        }
        first += block.target.len();
    }
    if rows.is_empty() {
        return Ok(result);
    }
    let mut multipliers = vec![0.0; rows.len()];
    let mut velocity = vec![0.0; factor.size];
    let mut active: Vec<(usize, Vec<f64>)> = Vec::new();
    loop {
        let violated = rows
            .iter()
            .enumerate()
            .map(|(index, (_, row, target))| (index, target - dot(row, &velocity)))
            .filter(|(_, violation)| *violation > tolerance)
            .max_by(|a, b| a.1.total_cmp(&b.1).then_with(|| b.0.cmp(&a.0)));
        let Some((selected, _)) = violated else {
            let mut impulses = vec![0.0; first];
            for ((index, _, _), value) in rows.iter().zip(&multipliers) {
                impulses[*index] = *value;
            }
            // Reconstruct from the final multipliers rather than trusting the
            // incrementally updated primal velocity used for active-set search.
            let mut actual = vec![0.0; factor.size];
            for ((_, row, _), lambda) in rows.iter().zip(&multipliers) {
                for (out, j) in actual.iter_mut().zip(*row) {
                    *out += j * lambda;
                }
            }
            result.factor_solves += 1;
            factor.solve(&mut actual)?;
            if rows
                .iter()
                .zip(&multipliers)
                .all(|((_, row, target), lambda)| {
                    let slack = dot(row, &actual) - target;
                    lambda.is_finite()
                        && *lambda >= 0.0
                        && slack.is_finite()
                        && slack >= -tolerance
                        && (*lambda == 0.0 || slack <= tolerance)
                })
            {
                result.impulses = Some(impulses);
            }
            return Ok(result);
        };
        if active.iter().any(|(index, _)| *index == selected) {
            return Ok(result);
        }
        let mut image = rows[selected].1.clone();
        factor.solve(&mut image)?;
        result.factor_solves += 1;
        let scale = dot(rows[selected].1, &image);
        loop {
            if result.pivots == maximum_pivots {
                return Ok(result);
            }
            result.pivots += 1;
            let count = active.len();
            let mut dual = active
                .iter()
                .map(|(index, _)| dot(rows[*index].1, &image))
                .collect::<Vec<_>>();
            if count > 0 {
                let matrix = active
                    .iter()
                    .flat_map(|(index, _)| {
                        active.iter().map(|(_, image)| dot(rows[*index].1, image))
                    })
                    .collect::<Vec<_>>();
                result.basis_storage = result.basis_storage.max(2 * matrix.len());
                result.basis_factorizations += 1;
                let Ok(basis) = DynamicsFactor::new(&matrix, count) else {
                    return Ok(result);
                };
                basis.solve(&mut dual)?;
            }
            let mut direction = image.clone();
            for ((_, image), coefficient) in active.iter().zip(&dual) {
                for (out, value) in direction.iter_mut().zip(image) {
                    *out -= coefficient * value;
                }
            }
            let denominator = dot(rows[selected].1, &direction);
            let primal = if denominator > 32.0 * f64::EPSILON * scale {
                (rows[selected].2 - dot(rows[selected].1, &velocity)) / denominator
            } else {
                f64::INFINITY
            };
            let limiting = active
                .iter()
                .zip(&dual)
                .enumerate()
                .filter(|(_, (_, rate))| **rate > 0.0)
                .map(|(position, ((index, _), rate))| (position, multipliers[*index] / rate))
                .min_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            let step = primal.min(limiting.map_or(f64::INFINITY, |(_, step)| step));
            if !step.is_finite() || step < 0.0 {
                return Ok(result);
            }
            for (out, direction) in velocity.iter_mut().zip(&direction) {
                *out += step * direction;
            }
            for ((index, _), rate) in active.iter().zip(&dual) {
                multipliers[*index] = (multipliers[*index] - step * rate).max(0.0);
            }
            multipliers[selected] += step;
            if primal <= limiting.map_or(f64::INFINITY, |(_, step)| step) {
                if active.len() == factor.size {
                    return Ok(result);
                }
                active.push((selected, image));
                break;
            }
            let Some((limiting, _)) = limiting else {
                return Ok(result);
            };
            multipliers[active[limiting].0] = 0.0;
            active.remove(limiting);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImpulseBounds;

    fn rows(jacobian: &[[f64; 2]], target: &[f64]) -> Vec<ConstraintBlock> {
        jacobian
            .iter()
            .zip(target)
            .map(|(row, target)| ConstraintBlock {
                jacobian: vec![row.to_vec()],
                target: vec![*target],
                bounds: vec![ImpulseBounds {
                    minimum: 0.0,
                    maximum: f64::INFINITY,
                }],
                contacts: Vec::new(),
            })
            .collect()
    }

    #[test]
    fn normal_basis_releases_a_weaker_constraint_and_preserves_momentum_response() {
        let factor = DynamicsFactor::new(&[1.0, 0.0, 0.0, 1.0], 2).unwrap();
        let blocks = rows(&[[1.0, 0.0], [2.0, 1.0]], &[2.0, 3.0]);
        let result = seed(&factor, &blocks, 16, 1e-12).unwrap();
        let impulses = result.impulses.unwrap();
        assert!((impulses[0] - 2.0).abs() < 1e-12);
        assert!(impulses[1].abs() < 1e-12);
        assert!(result.pivots >= 3 && result.pivots <= 16);
        assert_eq!(
            impulses,
            seed(&factor, &blocks, 16, 1e-12).unwrap().impulses.unwrap()
        );
    }

    #[test]
    fn normal_basis_keeps_the_strongest_dependent_row_and_rejects_inconsistent_rows() {
        let factor = DynamicsFactor::new(&[2.0, 0.0, 0.0, 1.0], 2).unwrap();
        let blocks = rows(&[[1.0, 0.0], [1.0, 0.0]], &[1.0, 2.0]);
        let impulses = seed(&factor, &blocks, 8, 1e-12).unwrap().impulses.unwrap();
        assert!(impulses[0].abs() < 1e-12);
        assert!((impulses[1] - 4.0).abs() < 1e-12);
        let impossible = rows(&[[1.0, 0.0], [-1.0, 0.0]], &[1.0, 1.0]);
        let rejected = seed(&factor, &impossible, 8, 1e-12).unwrap();
        assert!(rejected.impulses.is_none());
        assert!(rejected.pivots <= 8);
        assert!(seed(&factor, &blocks, 0, 1e-12).unwrap().impulses.is_none());
    }
}
