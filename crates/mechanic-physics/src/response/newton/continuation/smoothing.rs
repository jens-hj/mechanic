//! Smooth search equations only. Original nonsmooth laws decide acceptance.

use super::super::super::{BlockLayout, ConstraintBlock};

pub(super) struct Point {
    pub first: usize,
    pub size: usize,
    pub derivative: Vec<f64>,
}

pub(super) struct Evaluation {
    pub rhs: Vec<f64>,
    pub points: Vec<Point>,
    pub norm: f64,
    pub maximum: f64,
}

pub(super) fn evaluate(
    blocks: &[ConstraintBlock],
    layout: &[BlockLayout],
    modes: &[Vec<bool>],
    rolling: &[Option<Vec<f64>>],
    impulses: &[f64],
    motion: &[f64],
    epsilon: f64,
) -> Option<Evaluation> {
    if rolling.len() != blocks.len() {
        return None;
    }
    let mut rhs = vec![0.0; impulses.len()];
    let mut points = Vec::new();
    let mut norm = 0.0_f64;
    let mut maximum = 0.0_f64;
    for (((block, info), modes), rolling) in blocks.iter().zip(layout).zip(modes).zip(rolling) {
        let mut first = info.first;
        let width = epsilon / info.scale;
        if rolling.is_some() || block.contacts.is_empty() {
            if let Some(lengths) = rolling {
                points.push(manifold_point(
                    block, info, modes, lengths, impulses, motion, width, &mut rhs,
                ));
            } else {
                points.extend(scalar_points(
                    block, info, impulses, motion, width, &mut rhs,
                )?);
            }
            for value in &rhs[info.first..info.first + block.target.len()] {
                let residual = value * info.scale;
                maximum = maximum.max(residual.abs());
                norm = norm.hypot(residual);
            }
            continue;
        }
        for (contact, &sliding) in block.contacts.iter().zip(modes) {
            let size = contact.rows();
            let mut at = [0.0; 5];
            for row in 0..size {
                at[row] = impulses[first + row]
                    + (block.target[first + row - info.first] - motion[first + row]) / info.scale;
            }
            let mut projection = [0.0; 5];
            let mut derivative = vec![0.0; size * size];
            let root = at[0].hypot(width);
            // Rationalized negative branch avoids cancellation at released points.
            projection[0] = if at[0] < 0.0 {
                0.5 * width * (width / (root - at[0]))
            } else {
                0.5 * (at[0] + root)
            };
            derivative[0] = projection[0] / root;
            let coefficient = if sliding {
                contact.kinetic_coefficient
            } else {
                contact.static_coefficient
            };
            for (offset, coefficient) in std::iter::once((1, coefficient))
                .chain(contact.rolling_length.map(|length| (3, length)))
            {
                let values = [at[offset], at[offset + 1]];
                let length = values[0].hypot(values[1]);
                let radius = coefficient * projection[0];
                let root = (length - radius).hypot(width);
                let denominator = 0.5 * (radius + length + root);
                let length_derivative = 0.5 * (1.0 + (length - radius) / root);
                let radius_derivative = 1.0 - length_derivative;
                let ratio = radius / denominator;
                for row in 0..2 {
                    projection[offset + row] = values[row] * ratio;
                    derivative[(offset + row) * size] =
                        values[row] * coefficient * (1.0 - ratio * radius_derivative) / denominator
                            * derivative[0];
                    for column in 0..2 {
                        derivative[(offset + row) * size + offset + column] = ratio
                            * (f64::from(row == column)
                                - length_derivative * values[row] / denominator
                                    * if length > 0.0 {
                                        values[column] / length
                                    } else {
                                        0.0
                                    });
                    }
                }
            }
            for row in 0..size {
                rhs[first + row] = projection[row] - impulses[first + row];
                let residual = rhs[first + row] * info.scale;
                maximum = maximum.max(residual.abs());
                norm = norm.hypot(residual);
            }
            points.push(Point {
                first,
                size,
                derivative,
            });
            first += size;
        }
    }
    (norm.is_finite() && rhs.iter().all(|value| value.is_finite())).then_some(Evaluation {
        rhs,
        points,
        norm,
        maximum,
    })
}

// A merged manifold is one point: its rolling disk radius depends on every
// smoothed normal in the block, so the derivative couples all of its rows.
#[allow(clippy::too_many_arguments)] // One block's rows, laws, iterate and smoothing width.
fn manifold_point(
    block: &ConstraintBlock,
    info: &BlockLayout,
    modes: &[bool],
    lengths: &[f64],
    impulses: &[f64],
    motion: &[f64],
    width: f64,
    rhs: &mut [f64],
) -> Point {
    let size = block.target.len();
    let at = (0..size)
        .map(|row| {
            impulses[info.first + row] + (block.target[row] - motion[info.first + row]) / info.scale
        })
        .collect::<Vec<_>>();
    let mut projection = vec![0.0; size];
    let mut derivative = vec![0.0; size * size];
    let mut radius = 0.0;
    let mut radius_terms = Vec::with_capacity(lengths.len());
    for (index, ((contact, &sliding), length)) in
        block.contacts.iter().zip(modes).zip(lengths).enumerate()
    {
        let normal = 3 * index;
        let (value, slope) = smooth_plus(at[normal], width);
        projection[normal] = value;
        derivative[normal * size + normal] = slope;
        let coefficient = if sliding {
            contact.kinetic_coefficient
        } else {
            contact.static_coefficient
        };
        smoothed_disk(
            &at,
            &mut projection,
            &mut derivative,
            size,
            normal + 1,
            coefficient * value,
            &[(normal, coefficient * slope)],
            width,
        );
        radius += length * value;
        radius_terms.push((normal, length * slope));
    }
    smoothed_disk(
        &at,
        &mut projection,
        &mut derivative,
        size,
        3 * lengths.len(),
        radius,
        &radius_terms,
        width,
    );
    for (row, value) in projection.iter().enumerate() {
        rhs[info.first + row] = value - impulses[info.first + row];
    }
    Point {
        first: info.first,
        size,
        derivative,
    }
}

// Smoothed projection of two rows onto a disk whose radius depends on smoothed
// normals; `radius_terms` holds each `(row, d radius / d at[row])`.
#[allow(clippy::too_many_arguments)] // Shared block-wide output buffers and the disk definition.
fn smoothed_disk(
    at: &[f64],
    projection: &mut [f64],
    derivative: &mut [f64],
    size: usize,
    offset: usize,
    radius: f64,
    radius_terms: &[(usize, f64)],
    width: f64,
) {
    let values = [at[offset], at[offset + 1]];
    let length = values[0].hypot(values[1]);
    let root = (length - radius).hypot(width);
    let denominator = 0.5 * (radius + length + root);
    let length_derivative = 0.5 * (1.0 + (length - radius) / root);
    let radius_derivative = 1.0 - length_derivative;
    let ratio = radius / denominator;
    for row in 0..2 {
        projection[offset + row] = values[row] * ratio;
        let radius_slope = values[row] * (1.0 - ratio * radius_derivative) / denominator;
        for &(column, slope) in radius_terms {
            derivative[(offset + row) * size + column] += radius_slope * slope;
        }
        for column in 0..2 {
            let unit = if length > 0.0 {
                values[column] / length
            } else {
                0.0
            };
            derivative[(offset + row) * size + offset + column] = ratio
                * (f64::from(row == column) - length_derivative * values[row] / denominator * unit);
        }
    }
}

// One-row points for scalar stop and drive rows. A row without a smoothed
// interior (bilateral or fixed) rejects the evaluation.
fn scalar_points(
    block: &ConstraintBlock,
    info: &BlockLayout,
    impulses: &[f64],
    motion: &[f64],
    width: f64,
    rhs: &mut [f64],
) -> Option<Vec<Point>> {
    block
        .bounds
        .iter()
        .enumerate()
        .map(|(row, bounds)| {
            let index = info.first + row;
            let at = impulses[index] + (block.target[row] - motion[index]) / info.scale;
            let (projection, derivative) = scalar_projection(*bounds, at, width)?;
            rhs[index] = projection - impulses[index];
            Some(Point {
                first: index,
                size: 1,
                derivative: vec![derivative],
            })
        })
        .collect()
}

// Scalar stop and drive rows use the same smoothed clamp as a contact normal,
// from each finite bound. Bilateral and fixed rows have no interior and stay
// on the bordered path.
fn scalar_projection(bounds: crate::ImpulseBounds, at: f64, width: f64) -> Option<(f64, f64)> {
    match (bounds.minimum.is_finite(), bounds.maximum.is_finite()) {
        (true, true) if bounds.minimum < bounds.maximum => {
            let (low, low_derivative) = smooth_plus(at - bounds.minimum, width);
            let (high, high_derivative) = smooth_plus(at - bounds.maximum, width);
            Some((
                bounds.minimum + low - high,
                low_derivative - high_derivative,
            ))
        }
        (true, false) => {
            let (plus, derivative) = smooth_plus(at - bounds.minimum, width);
            Some((bounds.minimum + plus, derivative))
        }
        (false, true) => {
            let (plus, derivative) = smooth_plus(bounds.maximum - at, width);
            Some((bounds.maximum - plus, derivative))
        }
        _ => None,
    }
}

// Smoothed max(gap, 0) and its derivative. The rationalized negative branch
// avoids cancellation far outside the bound.
fn smooth_plus(gap: f64, width: f64) -> (f64, f64) {
    let root = gap.hypot(width);
    let value = if gap < 0.0 {
        0.5 * width * (width / (root - gap))
    } else {
        0.5 * (gap + root)
    };
    (value, value / root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContactFriction, ImpulseBounds};

    #[test]
    fn merged_manifold_rolling_derivative_matches_central_differences() {
        // Two contacts sharing one rolling pair: [n, t, t, n, t, t, r, r].
        let size = 8;
        let unbounded = ImpulseBounds {
            minimum: f64::NEG_INFINITY,
            maximum: f64::INFINITY,
        };
        let mut bounds = vec![unbounded; size];
        bounds[0].minimum = 0.0;
        bounds[3].minimum = 0.0;
        let contact = ContactFriction {
            static_coefficient: 0.8,
            kinetic_coefficient: 0.5,
            sliding: true,
            rolling_length: None,
        };
        let blocks = [ConstraintBlock {
            jacobian: vec![vec![0.0]; size],
            target: vec![0.0; size],
            bounds,
            contacts: vec![contact; 2],
        }];
        let layout = [BlockLayout {
            first: 0,
            diagonal: Vec::new(),
            scale: 1.0,
        }];
        let rolling = [Some(vec![0.05, 0.07])];
        let eval = |v: &[f64]| {
            evaluate(
                &blocks,
                &layout,
                &[vec![true, true]],
                &rolling,
                v,
                &[0.0; 8],
                0.01,
            )
            .unwrap()
        };
        for at in [
            [1.0, 0.2, -0.1, 0.5, 0.05, 0.3, 0.02, -0.05],
            [0.3, 0.01, -0.02, -0.2, 0.4, 0.4, 0.2, 0.1],
            [2.0, 3.0, -4.0, 1.5, 0.1, 0.1, 0.01, 0.02],
        ] {
            let original = eval(&at);
            assert_eq!(original.points.len(), 1);
            for column in 0..size {
                let step = 1e-6;
                let mut plus = at;
                let mut minus = at;
                plus[column] += step;
                minus[column] -= step;
                let a = eval(&plus);
                let b = eval(&minus);
                for row in 0..size {
                    let measured =
                        (a.rhs[row] + plus[row] - b.rhs[row] - minus[row]) / (2.0 * step);
                    let expected = original.points[0].derivative[row * size + column];
                    assert!(
                        (measured - expected).abs() < 2e-7,
                        "row={row} column={column} measured={measured} expected={expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn smooth_scalar_stop_and_drive_rows_stay_in_bounds_and_match_central_differences() {
        for bounds in [
            ImpulseBounds {
                minimum: 0.0,
                maximum: f64::INFINITY,
            },
            ImpulseBounds {
                minimum: -0.25,
                maximum: 0.4,
            },
            ImpulseBounds {
                minimum: -0.5,
                maximum: f64::INFINITY,
            },
            ImpulseBounds {
                minimum: f64::NEG_INFINITY,
                maximum: 0.3,
            },
        ] {
            let blocks = [ConstraintBlock {
                jacobian: vec![vec![0.0]],
                target: vec![0.0],
                bounds: vec![bounds],
                contacts: Vec::new(),
            }];
            let layout = [BlockLayout {
                first: 0,
                diagonal: vec![0.0],
                scale: 1.0,
            }];
            let eval = |impulse: f64| {
                evaluate(
                    &blocks,
                    &layout,
                    &[Vec::new()],
                    &[None],
                    &[impulse],
                    &[0.0],
                    0.01,
                )
                .unwrap()
            };
            let projected = |impulse: f64| eval(impulse).rhs[0] + impulse;
            for at in [-2.0, -0.4, 0.0, 0.25, 3.0] {
                let value = projected(at);
                assert!(value >= bounds.minimum && value <= bounds.maximum);
                let step = 1e-6;
                let measured = (projected(at + step) - projected(at - step)) / (2.0 * step);
                let expected = eval(at).points[0].derivative[0];
                assert!(
                    (measured - expected).abs() < 1e-7,
                    "bounds={bounds:?} at={at} measured={measured} expected={expected}"
                );
            }
        }
        let bilateral = [ConstraintBlock {
            jacobian: vec![vec![0.0]],
            target: vec![0.0],
            bounds: vec![ImpulseBounds {
                minimum: f64::NEG_INFINITY,
                maximum: f64::INFINITY,
            }],
            contacts: Vec::new(),
        }];
        let layout = [BlockLayout {
            first: 0,
            diagonal: vec![0.0],
            scale: 1.0,
        }];
        assert!(
            evaluate(
                &bilateral,
                &layout,
                &[Vec::new()],
                &[None],
                &[0.0],
                &[0.0],
                0.01
            )
            .is_none()
        );
    }

    #[test]
    fn smooth_contact_derivative_matches_central_differences_through_contact_transitions() {
        for rolling in [None, Some(0.1)] {
            let size = if rolling.is_some() { 5 } else { 3 };
            let mut bounds = vec![
                ImpulseBounds {
                    minimum: f64::NEG_INFINITY,
                    maximum: f64::INFINITY
                };
                size
            ];
            bounds[0].minimum = 0.0;
            let blocks = [ConstraintBlock {
                jacobian: vec![vec![0.0]; size],
                target: vec![0.0; size],
                bounds,
                contacts: vec![ContactFriction {
                    static_coefficient: 0.8,
                    kinetic_coefficient: 0.5,
                    sliding: true,
                    rolling_length: rolling,
                }],
            }];
            let layout = [BlockLayout {
                first: 0,
                diagonal: vec![0.0; size * size],
                scale: 1.0,
            }];
            for normal in [-2.0, 0.0, 1.0, 3.0] {
                for tangent in [[0.0, 0.0], [0.3, 0.4], [3.0, -4.0]] {
                    for mode in [false, true] {
                        let at = [normal, tangent[0], tangent[1], 0.06, -0.08];
                        let at = &at[..size];
                        let modes = [vec![mode]];
                        let eval = |v: &[f64]| {
                            evaluate(&blocks, &layout, &modes, &[None], v, &vec![0.0; size], 0.01)
                                .unwrap()
                        };
                        let original = eval(at);
                        for column in 0..size {
                            let mut plus = at.to_vec();
                            let mut minus = at.to_vec();
                            // The radial map is C1 at a zero tangent. Its central
                            // quotient there has first-order truncation error.
                            let step = if (column == 1 || column == 2) && tangent == [0.0, 0.0] {
                                1e-9
                            } else {
                                1e-6
                            };
                            plus[column] += step;
                            minus[column] -= step;
                            let a = eval(&plus);
                            let b = eval(&minus);
                            for row in 0..size {
                                let actual = (a.rhs[row] + plus[row] - b.rhs[row] - minus[row])
                                    / (2.0 * step);
                                let expected = original.points[0].derivative[row * size + column];
                                assert!(
                                    (actual - expected).abs() < 2e-7,
                                    "row={row} column={column} normal={normal} actual={actual} expected={expected}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
