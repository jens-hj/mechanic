//! One sweep per genome field: how to vary it, and the measure it must move.

use super::metrics::TreeMetrics;
use super::model::TreeModel;
use super::species::SpeciesSpec;

/// Samples per tree when a sweep measures foliage fill.
const FILL_SAMPLES: u32 = 3_000;

/// Varies one genome field from its smallest to its largest useful value.
#[derive(Clone, Copy, Debug)]
pub struct GenomeSweep {
    /// Field name as written in `flora.ron`.
    pub field: &'static str,
    /// Sets the field for `t` in `[0, 1]`.
    pub apply: fn(&mut SpeciesSpec, f64),
    /// The measure this field owns.
    pub metric: fn(&TreeModel, &TreeMetrics) -> f64,
    /// Whether the measure rises (+1) or falls (-1) as the field rises.
    pub direction: f64,
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "small positive counts"
)]
fn count(lo: f64, hi: f64, t: f64) -> u32 {
    (hi - lo).mul_add(t, lo).round() as u32
}

impl GenomeSweep {
    /// Every genome field, in `flora.ron` order. `bark` and the looks are
    /// appearance only and have no sweep.
    pub const ALL: [Self; 15] = [
        Self {
            field: "height",
            apply: |spec, t| {
                let height = 32.0f64.mul_add(t, 4.0);
                spec.height = (height, height * 1.1);
            },
            metric: |_, metrics| metrics.height,
            direction: 1.0,
        },
        Self {
            field: "width",
            apply: |spec, t| spec.width = 1.5f64.mul_add(t, 0.1),
            metric: |_, metrics| metrics.width_ratio,
            direction: 1.0,
        },
        Self {
            field: "girth",
            apply: |spec, t| spec.girth = 0.09f64.mul_add(t, 0.01),
            metric: |_, metrics| metrics.girth_ratio,
            direction: 1.0,
        },
        Self {
            field: "stems",
            apply: |spec, t| {
                let stems = count(1.0, 13.0, t);
                spec.stems = (stems, stems);
            },
            metric: |_, metrics| f64::from(metrics.stems),
            direction: 1.0,
        },
        Self {
            field: "crown_base",
            apply: |spec, t| spec.crown_base = 0.8 * t,
            metric: |_, metrics| metrics.lowest_foliage,
            direction: 1.0,
        },
        Self {
            field: "dominance",
            apply: |spec, t| spec.dominance = t,
            metric: |_, metrics| metrics.leader_reach,
            direction: 1.0,
        },
        Self {
            field: "split_chance",
            apply: |spec, t| spec.split_chance = 0.95f64.mul_add(t, 0.05),
            metric: |_, metrics| metrics.split_density,
            direction: 1.0,
        },
        Self {
            field: "split_count",
            apply: |spec, t| {
                let children = count(1.0, 7.0, t);
                spec.split_count = (children, children);
            },
            metric: |_, metrics| metrics.children_per_split,
            direction: 1.0,
        },
        Self {
            field: "split_angle",
            apply: |spec, t| spec.split_angle = 80.0f64.mul_add(t, 10.0).to_radians(),
            metric: |_, metrics| metrics.branch_angle,
            direction: 1.0,
        },
        Self {
            field: "tropism",
            apply: |spec, t| spec.tropism = 2.0f64.mul_add(t, -1.0),
            metric: |_, metrics| metrics.terminal_rise,
            direction: 1.0,
        },
        Self {
            field: "wobble",
            apply: |spec, t| spec.wobble = t,
            metric: |_, metrics| metrics.tortuosity,
            direction: 1.0,
        },
        Self {
            field: "foliage.size",
            apply: |spec, t| spec.foliage.size = 1.4f64.mul_add(t, 0.1),
            metric: |_, metrics| metrics.foliage_volume,
            direction: 1.0,
        },
        Self {
            field: "foliage.density",
            apply: |spec, t| spec.foliage.density = 0.9f64.mul_add(t, 0.1),
            metric: |tree, _| TreeMetrics::filled_fraction(tree, FILL_SAMPLES),
            direction: 1.0,
        },
        Self {
            field: "roots.spread",
            apply: |spec, t| spec.roots.spread = 1.8f64.mul_add(t, 0.2),
            metric: |_, metrics| metrics.root_radius,
            direction: 1.0,
        },
        Self {
            field: "roots.depth",
            apply: |spec, t| spec.roots.depth = 2.8f64.mul_add(t, 0.2),
            metric: |_, metrics| metrics.root_depth,
            direction: 1.0,
        },
    ];
}
