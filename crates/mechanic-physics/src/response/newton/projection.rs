//! Derivatives valid only for one fixed Newton linearization and friction mode.

use super::{BlockLayout, ConstraintBlock};
use crate::{ContactFriction, ImpulseBounds};

pub(super) struct PointDerivative {
    pub block: usize,
    pub first: usize,
    pub size: usize,
    normal_active: bool,
    tangent: Option<DiskDerivative>,
    rolling: Option<DiskDerivative>,
}

impl PointDerivative {
    fn new(
        block: usize,
        first: usize,
        bounds: ImpulseBounds,
        contact: Option<(ContactFriction, bool)>,
        at: &[f64],
    ) -> Self {
        let normal = at[0].clamp(bounds.minimum, bounds.maximum);
        let normal_active = at[0] > bounds.minimum && at[0] < bounds.maximum;
        let Some((law, sliding)) = contact else {
            return Self {
                block,
                first,
                size: 1,
                normal_active,
                tangent: None,
                rolling: None,
            };
        };
        let coefficient = if sliding {
            law.kinetic_coefficient
        } else {
            law.static_coefficient
        };
        Self {
            block,
            first,
            size: law.rows(),
            normal_active,
            tangent: Some(DiskDerivative::new(
                &at[1..3],
                coefficient * normal,
                coefficient,
            )),
            rolling: law
                .rolling_length
                .map(|length| DiskDerivative::new(&at[3..5], length * normal, length)),
        }
    }

    pub fn apply(&self, input: &[f64], output: &mut [f64]) {
        let normal = if self.normal_active { input[0] } else { 0.0 };
        output[0] = normal;
        if let Some(tangent) = &self.tangent {
            tangent.apply(&input[1..3], normal, &mut output[1..3]);
        }
        if let Some(rolling) = &self.rolling {
            rolling.apply(&input[3..5], normal, &mut output[3..5]);
        }
    }
}

enum DiskDerivative {
    Inside,
    Outside {
        unit: [f64; 2],
        ratio: f64,
        coefficient: f64,
    },
    Apex,
}

impl DiskDerivative {
    fn new(at: &[f64], radius: f64, coefficient: f64) -> Self {
        let length = at[0].hypot(at[1]);
        if length < radius {
            Self::Inside
        } else if length > 0.0 {
            Self::Outside {
                unit: [at[0] / length, at[1] / length],
                ratio: radius / length,
                coefficient,
            }
        } else {
            Self::Apex
        }
    }

    fn apply(&self, input: &[f64], normal_derivative: f64, output: &mut [f64]) {
        match self {
            Self::Inside => output.copy_from_slice(input),
            Self::Outside {
                unit,
                ratio,
                coefficient,
            } => {
                let radius_derivative = coefficient * normal_derivative;
                let radial = unit[0] * input[0] + unit[1] * input[1];
                for row in 0..2 {
                    output[row] =
                        ratio * (input[row] - unit[row] * radial) + unit[row] * radius_derivative;
                }
            }
            Self::Apex => output.fill(0.0),
        }
    }
}

pub(super) fn prepare(
    blocks: &[ConstraintBlock],
    layout: &[BlockLayout],
    modes: &[Vec<bool>],
    at: &[f64],
) -> Vec<PointDerivative> {
    let mut result = Vec::new();
    for (block_index, ((block, info), modes)) in blocks.iter().zip(layout).zip(modes).enumerate() {
        let count = if block.contacts.is_empty() {
            block.jacobian.len()
        } else {
            block.contacts.len()
        };
        let mut first = 0;
        for point in 0..count {
            let law = block
                .contacts
                .get(point)
                .copied()
                .zip(modes.get(point).copied());
            let derivative = PointDerivative::new(
                block_index,
                info.first + first,
                block.bounds[first],
                law,
                &at[info.first + first..],
            );
            first += derivative.size;
            result.push(derivative);
        }
    }
    result
}

/// Factor of (1 + shift) I - P for one fixed projection derivative. The normal
/// row is scalar; disk rows couple only to that normal and their radial axis.
pub(super) struct PointInverse {
    normal: f64,
    normal_active: bool,
    tangent: Option<DiskInverse>,
    rolling: Option<DiskInverse>,
}

enum DiskInverse {
    Diagonal(f64),
    Outside {
        unit: [f64; 2],
        radial: f64,
        tangent: f64,
        coefficient: f64,
    },
}

impl PointDerivative {
    pub(super) fn shifted_inverse(&self, shift: f64) -> Option<PointInverse> {
        let alpha = 1.0 + shift;
        if !shift.is_finite() || shift <= 0.0 {
            return None;
        }
        let normal = positive_reciprocal(alpha - f64::from(self.normal_active))?;
        Some(PointInverse {
            normal,
            normal_active: self.normal_active,
            tangent: match &self.tangent {
                Some(disk) => Some(disk.inverse(alpha)?),
                None => None,
            },
            rolling: match &self.rolling {
                Some(disk) => Some(disk.inverse(alpha)?),
                None => None,
            },
        })
    }
}

impl DiskDerivative {
    fn inverse(&self, alpha: f64) -> Option<DiskInverse> {
        Some(match *self {
            Self::Inside => DiskInverse::Diagonal(positive_reciprocal(alpha - 1.0)?),
            Self::Apex => DiskInverse::Diagonal(positive_reciprocal(alpha)?),
            Self::Outside {
                unit,
                ratio,
                coefficient,
            } => {
                let norm = unit[0] * unit[0] + unit[1] * unit[1];
                let tangent = alpha - ratio;
                // Keep the rounded direction's actual norm; assuming exactly one
                // would change the matrix at small shifts. Radial and orthogonal
                // components avoid cancellation in a Sherman-Morrison subtraction.
                DiskInverse::Outside {
                    unit,
                    radial: positive_reciprocal(norm * (tangent + ratio * norm))?,
                    tangent: positive_reciprocal(norm * tangent)?,
                    coefficient: coefficient * norm,
                }
            }
        })
    }
}

impl PointInverse {
    pub(super) fn solve(&self, values: &mut [f64]) -> Option<()> {
        values[0] *= self.normal;
        let normal = if self.normal_active { values[0] } else { 0.0 };
        if let Some(disk) = &self.tangent {
            disk.solve(&mut values[1..3], normal);
        }
        if let Some(disk) = &self.rolling {
            disk.solve(&mut values[3..5], normal);
        }
        values.iter().all(|value| value.is_finite()).then_some(())
    }
}

impl DiskInverse {
    fn solve(&self, values: &mut [f64], normal: f64) {
        match *self {
            Self::Diagonal(inverse) => {
                values[0] *= inverse;
                values[1] *= inverse;
            }
            Self::Outside {
                unit,
                radial,
                tangent,
                coefficient,
            } => {
                let along =
                    (unit[0] * values[0] + unit[1] * values[1] + coefficient * normal) * radial;
                let across = (-unit[1] * values[0] + unit[0] * values[1]) * tangent;
                values[0] = unit[0] * along - unit[1] * across;
                values[1] = unit[1] * along + unit[0] * across;
            }
        }
    }
}

fn positive_reciprocal(value: f64) -> Option<f64> {
    let inverse = value.recip();
    (value > 0.0 && inverse.is_finite()).then_some(inverse)
}

#[cfg(test)]
mod tests {
    use super::super::{apply_lu, factor_dense};
    use super::*;

    fn check(point: &PointDerivative) {
        for shift in [1e-2, 1e-4, 1e-6, 1e-8] {
            let width = point.size;
            let mut matrix = vec![0.0; width * width];
            for column in 0..width {
                let mut input = [0.0; 5];
                input[column] = 1.0;
                let mut projected = [0.0; 5];
                point.apply(&input[..width], &mut projected[..width]);
                for row in 0..width {
                    matrix[row * width + column] = (1.0 + shift) * input[row] - projected[row];
                }
            }
            let (lu, pivots) = factor_dense(matrix.clone(), width).unwrap();
            let inverse = point.shifted_inverse(shift).unwrap();
            for column in 0..width {
                let mut rhs = vec![0.0; width];
                rhs[column] = 1.0;
                let mut expected = rhs.clone();
                let mut actual = rhs.clone();
                apply_lu(&lu, &pivots, &mut expected).unwrap();
                inverse.solve(&mut actual).unwrap();
                let scale = expected.iter().map(|v| v.abs()).fold(1.0, f64::max);
                let difference = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                assert!(
                    difference <= 5e-8 * scale,
                    "shift={shift} difference={difference} scale={scale}"
                );
                // Independent original row residual, scaled by the actual sum of
                // magnitudes to expose a wrong coupling in ill-conditioned rows.
                for (row, target) in matrix.chunks_exact(width).zip(&rhs) {
                    let residual =
                        row.iter().zip(&actual).map(|(a, b)| a * b).sum::<f64>() - target;
                    let scale = row
                        .iter()
                        .zip(&actual)
                        .map(|(a, b)| (a * b).abs())
                        .sum::<f64>()
                        + target.abs();
                    assert!(
                        residual.abs() <= 1e-14 * scale.max(1.0),
                        "residual={residual} scale={scale}"
                    );
                }
            }
        }
    }

    #[test]
    fn analytic_point_inverse_matches_lu_and_every_original_row() {
        let bounds = ImpulseBounds {
            minimum: 0.0,
            maximum: 4.0,
        };
        for normal in [-1.0, 0.0, 2.0, 4.0, 5.0] {
            check(&PointDerivative::new(0, 0, bounds, None, &[normal]));
            for tangent in [[0.0, 0.0], [0.3, -0.4], [0.6, 0.8], [3.0, 4.0]] {
                for rolling in [[0.0, 0.0], [0.03, 0.04], [0.6, -0.8]] {
                    for length in [None, Some(0.1)] {
                        let law = ContactFriction {
                            static_coefficient: 0.8,
                            kinetic_coefficient: 0.5,
                            sliding: true,
                            rolling_length: length,
                        };
                        for sliding in [false, true] {
                            check(&PointDerivative::new(
                                0,
                                0,
                                bounds,
                                Some((law, sliding)),
                                &[normal, tangent[0], tangent[1], rolling[0], rolling[1]],
                            ));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn analytic_point_inverse_rejects_invalid_or_unshifted_singular_systems() {
        let point = PointDerivative::new(
            0,
            0,
            ImpulseBounds {
                minimum: f64::NEG_INFINITY,
                maximum: f64::INFINITY,
            },
            None,
            &[1.0],
        );
        for shift in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(point.shifted_inverse(shift).is_none());
        }
        assert!(
            point
                .shifted_inverse(1e-3)
                .unwrap()
                .solve(&mut [f64::NAN])
                .is_none()
        );
    }
}
