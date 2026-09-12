//! Exactly duplicated rolling rows in a contact manifold share one disk.
//!
//! Every point of a manifold constrains the same body angular rolling rows, so
//! the per-point disks `|r_i| <= L_i n_i` on identical axes add to one disk
//! `|R| <= sum_i L_i n_i`. The search runs on the merged rows. `expand` splits R
//! in proportion to `L_i n_i`, which satisfies every original per-point law, and
//! the original contact laws alone still accept the result.

use super::super::super::{BlockLayout, ConstraintBlock, ContactFriction, project, project_disk};

/// Relative agreement required before rolling rows count as the same axes.
const DUPLICATE_TOLERANCE: f64 = 1e-12;

struct Span {
    original: usize,
    reduced: usize,
    size: usize,
}

/// Search-only view with each duplicated manifold rolling pair merged.
pub(super) struct Manifolds {
    pub blocks: Vec<ConstraintBlock>,
    pub layout: Vec<BlockLayout>,
    /// Per block: the original per-point rolling lengths when merged.
    pub rolling: Vec<Option<Vec<f64>>>,
    spans: Vec<Span>,
    reduced_rows: usize,
    original_rows: usize,
}

impl Manifolds {
    pub(super) fn new(blocks: &[ConstraintBlock], layout: &[BlockLayout]) -> Self {
        let mut result = Self {
            blocks: Vec::with_capacity(blocks.len()),
            layout: Vec::with_capacity(blocks.len()),
            rolling: Vec::with_capacity(blocks.len()),
            spans: Vec::with_capacity(blocks.len()),
            reduced_rows: 0,
            original_rows: 0,
        };
        for (block, info) in blocks.iter().zip(layout) {
            let lengths = duplicated_rolling(block);
            let reduced = if lengths.is_some() {
                merge(block)
            } else {
                block.clone()
            };
            result.layout.push(BlockLayout {
                first: result.reduced_rows,
                diagonal: if lengths.is_some() {
                    Vec::new()
                } else {
                    info.diagonal.clone()
                },
                scale: info.scale,
            });
            result.spans.push(Span {
                original: info.first,
                reduced: result.reduced_rows,
                size: block.target.len(),
            });
            result.reduced_rows += reduced.target.len();
            result.original_rows += block.target.len();
            result.blocks.push(reduced);
            result.rolling.push(lengths);
        }
        result
    }

    /// Sums each merged manifold's per-point rolling impulses.
    pub(super) fn reduce(&self, impulses: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.reduced_rows];
        for (span, lengths) in self.spans.iter().zip(&self.rolling) {
            let Some(lengths) = lengths else {
                result[span.reduced..span.reduced + span.size]
                    .copy_from_slice(&impulses[span.original..span.original + span.size]);
                continue;
            };
            let rolling = span.reduced + 3 * lengths.len();
            for index in 0..lengths.len() {
                let from = span.original + 5 * index;
                let to = span.reduced + 3 * index;
                result[to..to + 3].copy_from_slice(&impulses[from..from + 3]);
                result[rolling] += impulses[from + 3];
                result[rolling + 1] += impulses[from + 4];
            }
        }
        result
    }

    /// Splits merged rolling impulses in proportion to `L_i n_i`. Apply only to
    /// a projected iterate, whose merged disk radius is `sum_i L_i n_i`.
    pub(super) fn expand(&self, reduced: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.original_rows];
        for (span, lengths) in self.spans.iter().zip(&self.rolling) {
            let Some(lengths) = lengths else {
                result[span.original..span.original + span.size]
                    .copy_from_slice(&reduced[span.reduced..span.reduced + span.size]);
                continue;
            };
            let rolling = span.reduced + 3 * lengths.len();
            let total = lengths
                .iter()
                .enumerate()
                .map(|(index, length)| length * reduced[span.reduced + 3 * index].max(0.0))
                .sum::<f64>();
            for (index, length) in lengths.iter().enumerate() {
                let from = span.reduced + 3 * index;
                let to = span.original + 5 * index;
                result[to..to + 3].copy_from_slice(&reduced[from..from + 3]);
                if total > 0.0 {
                    let share = length * reduced[from].max(0.0) / total;
                    result[to + 3] = reduced[rolling] * share;
                    result[to + 4] = reduced[rolling + 1] * share;
                }
            }
        }
        result
    }

    /// Projects merged-layout impulses onto their bounds, cones and merged disks.
    pub(super) fn project(&self, reduced: &mut [f64], modes: &[Vec<bool>]) {
        for (((block, info), lengths), mode) in self
            .blocks
            .iter()
            .zip(&self.layout)
            .zip(&self.rolling)
            .zip(modes)
        {
            let values = &mut reduced[info.first..info.first + block.target.len()];
            project(block, mode, values);
            if let Some(lengths) = lengths {
                let radius = lengths
                    .iter()
                    .enumerate()
                    .map(|(index, length)| length * values[3 * index].max(0.0))
                    .sum::<f64>();
                let rolling = 3 * lengths.len();
                project_disk(&mut values[rolling..rolling + 2], radius);
            }
        }
    }
}

// Rolling lengths when every point shares bitwise-close rolling rows and targets.
fn duplicated_rolling(block: &ConstraintBlock) -> Option<Vec<f64>> {
    if block.contacts.len() < 2
        || block
            .contacts
            .iter()
            .any(|contact| contact.rolling_length.is_none())
    {
        return None;
    }
    let reference = &block.jacobian[3..5];
    let targets = &block.target[3..5];
    let row_scale = reference
        .iter()
        .flatten()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    let target_scale = targets
        .iter()
        .fold(1.0_f64, |largest, value| largest.max(value.abs()));
    for index in 1..block.contacts.len() {
        let first = 5 * index;
        for row in 0..2 {
            let rows_match = block.jacobian[first + 3 + row]
                .iter()
                .zip(&reference[row])
                .all(|(a, b)| (a - b).abs() <= DUPLICATE_TOLERANCE * row_scale);
            if !rows_match
                || (block.target[first + 3 + row] - targets[row]).abs()
                    > DUPLICATE_TOLERANCE * target_scale
            {
                return None;
            }
        }
    }
    block
        .contacts
        .iter()
        .map(|contact| contact.rolling_length)
        .collect()
}

// Keeps each point's normal and tangent rows, then one shared rolling pair.
fn merge(block: &ConstraintBlock) -> ConstraintBlock {
    let mut merged = ConstraintBlock {
        jacobian: Vec::with_capacity(3 * block.contacts.len() + 2),
        target: Vec::with_capacity(3 * block.contacts.len() + 2),
        bounds: Vec::with_capacity(3 * block.contacts.len() + 2),
        contacts: Vec::with_capacity(block.contacts.len()),
    };
    for (index, contact) in block.contacts.iter().enumerate() {
        let first = 5 * index;
        merged
            .jacobian
            .extend_from_slice(&block.jacobian[first..first + 3]);
        merged
            .target
            .extend_from_slice(&block.target[first..first + 3]);
        merged
            .bounds
            .extend_from_slice(&block.bounds[first..first + 3]);
        merged.contacts.push(ContactFriction {
            rolling_length: None,
            ..*contact
        });
    }
    merged.jacobian.extend_from_slice(&block.jacobian[3..5]);
    merged.target.extend_from_slice(&block.target[3..5]);
    merged.bounds.extend_from_slice(&block.bounds[3..5]);
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImpulseBounds;

    fn manifold(second_rolling: [f64; 3]) -> (ConstraintBlock, BlockLayout) {
        let unbounded = ImpulseBounds {
            minimum: f64::NEG_INFINITY,
            maximum: f64::INFINITY,
        };
        let normal = ImpulseBounds {
            minimum: 0.0,
            maximum: f64::INFINITY,
        };
        let block = ConstraintBlock {
            jacobian: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![0.0, 0.5, 0.0],
                vec![0.0, 0.0, 0.5],
                vec![1.0, 0.1, 0.0],
                vec![0.0, 1.0, 0.1],
                vec![0.1, 0.0, 1.0],
                second_rolling.to_vec(),
                vec![0.0, 0.0, 0.5],
            ],
            target: vec![0.0; 10],
            bounds: [normal, unbounded, unbounded, unbounded, unbounded].repeat(2),
            contacts: [0.02, 0.03]
                .map(|length| ContactFriction {
                    static_coefficient: 0.8,
                    kinetic_coefficient: 0.5,
                    sliding: false,
                    rolling_length: Some(length),
                })
                .to_vec(),
        };
        let layout = BlockLayout {
            first: 0,
            diagonal: vec![0.0; 100],
            scale: 2.0,
        };
        (block, layout)
    }

    #[test]
    fn identical_manifold_rolling_merges_and_splits_within_every_point_law() {
        let (block, layout) = manifold([0.0, 0.5, 0.0]);
        let merged = Manifolds::new(std::slice::from_ref(&block), &[layout]);
        assert_eq!(merged.blocks[0].target.len(), 8);
        assert_eq!(merged.rolling[0].as_deref(), Some(&[0.02, 0.03][..]));
        let original = [2.0, 0.1, 0.2, 0.01, -0.02, 1.0, 0.0, 0.1, 0.03, 0.04];
        let reduced = merged.reduce(&original);
        assert_eq!(&reduced[..3], &original[..3]);
        assert_eq!(&reduced[3..6], &original[5..8]);
        assert!((reduced[6] - 0.04).abs() < 1e-15 && (reduced[7] - 0.02).abs() < 1e-15);

        // A saturated merged disk of radius 0.02*2 + 0.03*1 splits into two
        // saturated point disks with the same direction.
        let mut saturated = reduced.clone();
        saturated[6] = 1.0;
        saturated[7] = 0.0;
        merged.project(&mut saturated, &[vec![false, false]]);
        assert!((saturated[6] - 0.07).abs() < 1e-15 && saturated[7].abs() < 1e-15);
        let split = merged.expand(&saturated);
        assert!((split[3] - 0.04).abs() < 1e-15 && split[4].abs() < 1e-15);
        assert!((split[8] - 0.03).abs() < 1e-15 && split[9].abs() < 1e-15);
        assert!((split[3] + split[8] - saturated[6]).abs() < 1e-15);
    }

    #[test]
    fn distinct_rolling_rows_keep_the_original_layout() {
        let (block, layout) = manifold([0.0, 0.5, 1e-6]);
        let merged = Manifolds::new(std::slice::from_ref(&block), &[layout]);
        assert_eq!(merged.blocks[0].target.len(), 10);
        assert!(merged.rolling[0].is_none());
        let original = [2.0, 0.1, 0.2, 0.01, -0.02, 1.0, 0.0, 0.1, 0.03, 0.04];
        assert_eq!(merged.expand(&merged.reduce(&original)), original);
    }
}
