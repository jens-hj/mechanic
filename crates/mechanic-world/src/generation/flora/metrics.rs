//! Observable measures of a grown tree, shared by tests and the gallery.

use bevy_math::DVec3;

use super::super::scatter::mix;
use super::TWIG_RADIUS;
use super::model::{Part, TreeModel};

/// Height bands used to find where the crown is widest.
const BANDS: usize = 24;

/// Shape measures of one tree. Ratios are relative to its measured height.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TreeMetrics {
    /// Top of wood or foliage above the origin, in metres.
    pub height: f64,
    /// Widest horizontal extent of wood and foliage above ground, in metres.
    pub crown_width: f64,
    /// `crown_width / height`.
    pub width_ratio: f64,
    /// Trunk base diameter (all stems' area together) over height.
    pub girth_ratio: f64,
    /// Stems rising from the base.
    pub stems: u32,
    /// Bottom of the lowest foliage over height.
    pub lowest_foliage: f64,
    /// Height of the widest crown band over height.
    pub widest_at: f64,
    /// Wood and root segments.
    pub segments: usize,
    /// Foliage blobs.
    pub foliage_blobs: usize,
    /// Mean children per split of wood axes.
    pub children_per_split: f64,
    /// Mean angle of first-order branches from vertical where they leave
    /// their parent, in degrees.
    pub branch_angle: f64,
    /// Length-weighted mean upward component of unbranched twigs' direction.
    pub terminal_rise: f64,
    /// Length-weighted mean upward component of first-order branches.
    pub lateral_rise: f64,
    /// Length-weighted mean of path length over chord for stems.
    pub tortuosity: f64,
    /// Fraction of nodes that split, on laterals thick enough to split.
    pub split_density: f64,
    /// Summed volume of foliage blobs, in cubic metres.
    pub foliage_volume: f64,
    /// Farthest horizontal reach of the roots, in metres.
    pub root_radius: f64,
    /// Deepest root below the origin, in metres.
    pub root_depth: f64,
    /// Fraction of wood length on order-0 axes.
    pub trunk_share: f64,
    /// Largest angle of a stem from vertical, in degrees.
    pub max_stem_lean: f64,
    /// Highest point the original stems reach before forking, over height.
    pub leader_reach: f64,
}

impl TreeMetrics {
    /// Measures a tree.
    #[must_use]
    #[expect(clippy::too_many_lines, reason = "one pass per measure, kept together")]
    pub fn measure(tree: &TreeModel) -> Self {
        let origin = tree.origin;
        let mut top: f64 = 0.0;
        let (mut low, mut high) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
        let mut grow_bounds = |point: DVec3, radius: f64| {
            low = low.min(point - radius);
            high = high.max(point + radius);
        };
        for segment in tree.segments.iter().filter(|s| s.part == Part::Wood) {
            for (point, radius) in [(segment.a, segment.ra), (segment.b, segment.rb)] {
                if point.y >= origin.y {
                    top = top.max(point.y + radius - origin.y);
                    grow_bounds(point, radius);
                }
            }
        }
        let mut lowest = f64::INFINITY;
        let mut foliage_volume = 0.0;
        for blob in &tree.foliage {
            for point in [blob.a, blob.b] {
                top = top.max(point.y + blob.radius - origin.y);
                grow_bounds(point, blob.radius);
                lowest = lowest.min(point.y - blob.radius - origin.y);
            }
            let radius = blob.radius;
            foliage_volume += std::f64::consts::PI
                * radius
                * radius
                * (4.0 / 3.0f64).mul_add(radius, blob.a.distance(blob.b));
        }
        let height = top.max(1.0e-6);
        let crown_width = if low.x.is_finite() {
            (high.x - low.x).max(high.z - low.z)
        } else {
            0.0
        };

        // Widest band: horizontal reach from the base axis per height band.
        let mut bands = [0.0_f64; BANDS];
        let mut mark = |point: DVec3, radius: f64| {
            let reach = (point - origin).with_y(0.0).length() + radius;
            let from = band(point.y - radius - origin.y, height);
            let to = band(point.y + radius - origin.y, height);
            for value in &mut bands[from..=to] {
                *value = value.max(reach);
            }
        };
        for blob in &tree.foliage {
            mark(blob.a, blob.radius);
            mark(blob.b, blob.radius);
        }
        for segment in tree.segments.iter().filter(|s| s.part == Part::Wood) {
            mark(segment.b, segment.rb);
        }
        let widest = bands.iter().copied().fold(0.0, f64::max);
        let (mut sum, mut count) = (0.0, 0.0);
        for (index, value) in bands.iter().enumerate() {
            if *value >= 0.95 * widest && widest > 0.0 {
                #[expect(clippy::cast_precision_loss, reason = "a handful of bands")]
                let centre = (index as f64 + 0.5) / BANDS as f64;
                sum += centre;
                count += 1.0;
            }
        }
        let widest_at = if count > 0.0 { sum / count } else { 0.0 };

        let mut stems = 0;
        let mut stem_area = 0.0;
        let mut max_stem_lean: f64 = 0.0;
        let mut leader_top: f64 = 0.0;
        let (mut splits, mut children) = (0, 0);
        let (mut angle_sum, mut angle_count) = (0.0, 0.0);
        let (mut terminal_sum, mut terminal_weight) = (0.0, 0.0);
        let (mut lateral_sum, mut lateral_weight) = (0.0, 0.0);
        let (mut tortuous_sum, mut tortuous_weight) = (0.0, 0.0);
        let (mut trunk_length, mut wood_length) = (0.0, 0.0);
        let (mut lateral_splits, mut lateral_nodes) = (0.0, 0.0);
        for axis in &tree.axes {
            let segments = &tree.segments[axis.segments.0 as usize..axis.segments.1 as usize];
            let (Some(first), Some(last)) = (segments.first(), segments.last()) else {
                continue;
            };
            if axis.part != Part::Wood {
                continue;
            }
            splits += axis.splits;
            children += axis.children;
            let length: f64 = segments.iter().map(|s| s.a.distance(s.b)).sum();
            wood_length += length;
            if axis.order == 0 {
                trunk_length += length;
            } else if first.ra > TWIG_RADIUS * 1.01 && segments.len() > 1 {
                lateral_splits += f64::from(axis.splits);
                #[expect(clippy::cast_precision_loss, reason = "segment counts are small")]
                let nodes = (segments.len() - 1) as f64;
                lateral_nodes += nodes;
            }
            if axis.stem {
                stems += 1;
                stem_area += first.ra * first.ra;
                let dir = (first.b - first.a).normalize_or_zero();
                max_stem_lean = max_stem_lean.max(dir.y.clamp(-1.0, 1.0).acos().to_degrees());
                leader_top = leader_top.max(last.b.y - origin.y);
            }
            if axis.order == 1 {
                let dir = (first.b - first.a).normalize_or_zero();
                angle_sum += dir.y.clamp(-1.0, 1.0).acos().to_degrees();
                angle_count += 1.0;
                for segment in segments {
                    let span = segment.b - segment.a;
                    lateral_sum += span.y;
                    lateral_weight += span.length();
                }
            }
            if axis.order >= 1 && axis.splits == 0 {
                for segment in segments {
                    let span = segment.b - segment.a;
                    terminal_sum += span.y;
                    terminal_weight += span.length();
                }
            }
            if axis.stem && segments.len() >= 3 {
                let chord = first.a.distance(last.b);
                if chord > 1.0e-9 {
                    tortuous_sum += length / chord * length;
                    tortuous_weight += length;
                }
            }
        }
        let mut root_radius: f64 = 0.0;
        let mut root_depth: f64 = 0.0;
        for segment in tree.segments.iter().filter(|s| s.part == Part::Root) {
            root_radius = root_radius.max((segment.b - origin).with_y(0.0).length());
            root_depth = root_depth.max(origin.y - segment.b.y);
        }
        let ratio = |sum: f64, weight: f64| if weight > 0.0 { sum / weight } else { 0.0 };
        Self {
            height,
            crown_width,
            width_ratio: crown_width / height,
            girth_ratio: 2.0 * stem_area.sqrt() / height,
            stems,
            lowest_foliage: if lowest.is_finite() {
                lowest.max(0.0) / height
            } else {
                1.0
            },
            widest_at,
            segments: tree.segments.len(),
            foliage_blobs: tree.foliage.len(),
            children_per_split: ratio(f64::from(children), f64::from(splits)),
            branch_angle: ratio(angle_sum, angle_count),
            terminal_rise: ratio(terminal_sum, terminal_weight),
            lateral_rise: ratio(lateral_sum, lateral_weight),
            tortuosity: ratio(tortuous_sum, tortuous_weight),
            split_density: ratio(lateral_splits, lateral_nodes),
            foliage_volume,
            root_radius,
            root_depth,
            trunk_share: ratio(trunk_length, wood_length),
            max_stem_lean,
            leader_reach: leader_top / height,
        }
    }

    /// Fraction of points inside foliage blobs that are solid foliage,
    /// estimated from `samples` points spread through the blobs by volume.
    #[must_use]
    pub fn filled_fraction(tree: &TreeModel, samples: u32) -> f64 {
        if tree.foliage.is_empty() {
            return 0.0;
        }
        let mut state = mix(0xf111_ed00 ^ tree.segments.len() as u64);
        let mut unit = || {
            state = mix(state.wrapping_add(0x9e37_79b9_7f4a_7c15));
            #[expect(
                clippy::cast_precision_loss,
                reason = "53 random bits map exactly onto the unit interval"
            )]
            let value = (state >> 11) as f64 / (1_u64 << 53) as f64;
            value
        };
        let (mut filled, mut total) = (0_u32, 0_u32);
        for _ in 0..samples {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss,
                reason = "an index below the blob count"
            )]
            let blob = tree.foliage
                [((unit() * tree.foliage.len() as f64) as usize).min(tree.foliage.len() - 1)];
            // A point in the blob: along its axis, then within its radius.
            let along = blob.a.lerp(blob.b, unit());
            let offset = DVec3::new(unit() * 2.0 - 1.0, unit() * 2.0 - 1.0, unit() * 2.0 - 1.0);
            if offset.length_squared() > 1.0 {
                continue;
            }
            let point = along + offset * blob.radius;
            total += 1;
            if tree
                .sample(point)
                .is_some_and(|(density, part)| density > 0.0 && part == Part::Foliage)
            {
                filled += 1;
            }
        }
        if total == 0 {
            0.0
        } else {
            f64::from(filled) / f64::from(total)
        }
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "clamped to the band count"
)]
fn band(height: f64, total: f64) -> usize {
    ((height / total * BANDS as f64).floor().max(0.0) as usize).min(BANDS - 1)
}
