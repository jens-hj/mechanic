use bevy_math::DVec3;

use super::super::{GenomeSweep, TreeMetrics, grow_tree};
use super::{mean, preset};

const STEPS: usize = 7;
const SWEEP_SEEDS: u64 = 8;

/// Mean of every sweep's measure at each step of one field's sweep:
/// `[step][measure]`.
fn responses(field: &GenomeSweep) -> Vec<Vec<f64>> {
    let base = preset("oak");
    (0..STEPS)
        .map(|step| {
            #[expect(clippy::cast_precision_loss, reason = "a handful of steps")]
            let t = step as f64 / (STEPS - 1) as f64;
            let mut species = base.clone();
            (field.apply)(&mut species, t);
            let trees = (0..SWEEP_SEEDS)
                .map(|seed| {
                    let tree = grow_tree(&species, seed * 104_729 + 11, DVec3::ZERO);
                    let metrics = TreeMetrics::measure(&tree);
                    (tree, metrics)
                })
                .collect::<Vec<_>>();
            GenomeSweep::ALL
                .iter()
                .map(|measure| mean(trees.iter().map(|(tree, m)| (measure.metric)(tree, m))))
                .collect()
        })
        .collect()
}

fn ranks(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|value| {
            let below = values.iter().filter(|other| *other < value).count();
            let equal = values
                .iter()
                .filter(|other| other.total_cmp(value).is_eq())
                .count();
            #[expect(clippy::cast_precision_loss, reason = "a handful of steps")]
            let rank = below as f64 + (equal as f64 - 1.0) / 2.0;
            rank
        })
        .collect()
}

/// Spearman rank correlation of a series against its step index.
fn spearman(values: &[f64]) -> f64 {
    let ranked = ranks(values);
    #[expect(clippy::cast_precision_loss, reason = "a handful of steps")]
    let steps = (0..values.len())
        .map(|step| step as f64)
        .collect::<Vec<_>>();
    let mean_rank = mean(ranked.iter().copied());
    let mean_step = mean(steps.iter().copied());
    let (mut covariance, mut rank_spread, mut step_spread) = (0.0, 0.0, 0.0);
    for (rank, step) in ranked.iter().zip(&steps) {
        covariance += (rank - mean_rank) * (step - mean_step);
        rank_spread += (rank - mean_rank).powi(2);
        step_spread += (step - mean_step).powi(2);
    }
    if rank_spread == 0.0 {
        return 0.0;
    }
    covariance / (rank_spread * step_spread).sqrt()
}

fn spread(values: impl Iterator<Item = f64>) -> f64 {
    let (low, high) = values.fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
        (low.min(value), high.max(value))
    });
    high - low
}

/// Every field moves its own measure monotonically, and no two fields do the
/// same thing: their effects on all measures, each scaled by how far its own
/// field moves it, must not point the same way.
#[test]
fn every_genome_field_moves_its_own_metric() {
    let all = GenomeSweep::ALL.iter().map(responses).collect::<Vec<_>>();
    let count = GenomeSweep::ALL.len();
    let scale = (0..count)
        .map(|measure| spread(all[measure].iter().map(|step| step[measure])).max(1.0e-9))
        .collect::<Vec<_>>();
    let effects = all
        .iter()
        .map(|steps| {
            (0..count)
                .map(|measure| (steps[STEPS - 1][measure] - steps[0][measure]) / scale[measure])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut failures = Vec::new();
    for (owner, sweep) in GenomeSweep::ALL.iter().enumerate() {
        let own = all[owner]
            .iter()
            .map(|step| step[owner])
            .collect::<Vec<_>>();
        let rho = spearman(&own) * sweep.direction;
        if rho < 0.9 {
            failures.push(format!(
                "{}: own measure rho = {rho:.2} over {own:.3?}",
                sweep.field
            ));
        }
        for other in owner + 1..count {
            let dot: f64 = effects[owner]
                .iter()
                .zip(&effects[other])
                .map(|(a, b)| a * b)
                .sum();
            let norm =
                |effect: &[f64]| effect.iter().map(|value| value * value).sum::<f64>().sqrt();
            let cosine = dot / (norm(&effects[owner]) * norm(&effects[other])).max(1.0e-12);
            if cosine.abs() >= 0.9 {
                failures.push(format!(
                    "{} and {} have the same effect (cosine {cosine:.2})",
                    sweep.field,
                    GenomeSweep::ALL[other].field
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
