//! River networks traced once per world over the blended biome heights.
//!
//! Depressions are filled with priority-flood (Barnes et al. 2014) so every
//! cell drains to the sea or the world's edge, flow follows the steepest
//! filled descent, and cells whose upstream area passes a threshold become
//! channel segments. The field then carves a valley around each segment.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::interval::Interval;
use super::noise::NoiseGen;
use super::spec::{Dims, Fractal, NoiseDoc, NoiseKind, RiversDoc};
use super::tape::smoothstep;
use crate::WORLD_HALF_EXTENT_METERS;

/// Grid spacing of the drainage analysis.
pub(crate) const DRAINAGE_CELL_METRES: f64 = 32.0;

/// Horizontal distance beyond a river's bank at which its valley stops.
const VALLEY_REACH_METRES: f64 = 140.0;

/// Width over which a valley's far edge fades back to untouched ground.
const VALLEY_FADE_METRES: f64 = 40.0;

/// Height added to a valley where rivers are disallowed or faded out.
pub(crate) const RIVER_LIFT_METRES: f64 = 2_000.0;

const BUCKET_METRES: f64 = 128.0;

#[derive(Clone, Copy, Debug)]
struct Segment {
    from: [f64; 2],
    to: [f64; 2],
    /// Water level at each end.
    level: [f64; 2],
    half_width: [f64; 2],
    depth: [f64; 2],
}

/// Carved valley under one column.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Valley {
    /// Ground may not rise above this height.
    pub(crate) height: f64,
    /// Horizontal distance to the nearest centre line.
    pub(crate) distance: f64,
}

#[derive(Debug)]
pub(crate) struct RiverNetwork {
    segments: Vec<Segment>,
    buckets: Vec<Vec<u32>>,
    bucket_count: usize,
    bank_slope: f64,
    meander: f64,
    meander_x: NoiseGen,
    meander_z: NoiseGen,
    reach: f64,
}

#[derive(Clone, Copy, PartialEq)]
struct Pending {
    height: f64,
    index: usize,
}

impl Eq for Pending {}

impl PartialOrd for Pending {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Pending {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .height
            .total_cmp(&self.height)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl RiverNetwork {
    /// A network with no rivers, used before tracing.
    pub(crate) fn none() -> Self {
        let dummy = || {
            NoiseGen::new(
                &NoiseDoc {
                    kind: NoiseKind::Simplex,
                    fractal: Fractal::Fbm,
                    octaves: 1,
                    freq: 1.0,
                    lacunarity: 2.0,
                    gain: 0.5,
                    amp: 0.0,
                    offset: 0.0,
                    dims: Dims::Two,
                    seed: 0,
                },
                0,
            )
        };
        Self {
            segments: Vec::new(),
            buckets: vec![Vec::new()],
            bucket_count: 1,
            bank_slope: 1.0,
            meander: 0.0,
            meander_x: dummy(),
            meander_z: dummy(),
            reach: 0.0,
        }
    }

    /// Traces rivers over `heights`, sampled on the drainage grid with
    /// `side` points per row starting at the world's lower corner.
    #[expect(clippy::cast_precision_loss, reason = "grid indices are a few hundred")]
    #[expect(
        clippy::too_many_lines,
        reason = "fill, flow, accumulation, and segmenting share the grid"
    )]
    pub(crate) fn trace(doc: &RiversDoc, sea_level: f64, heights: &[f64], seed: i32) -> Self {
        let side = drainage_side();
        debug_assert_eq!(heights.len(), side * side);
        let position =
            |index: usize| (index as f64).mul_add(DRAINAGE_CELL_METRES, -WORLD_HALF_EXTENT_METERS);
        let neighbours = |index: usize| {
            let (column, row) = (index % side, index / side);
            let mut found = [None; 8];
            let mut count = 0;
            for z in row.saturating_sub(1)..=(row + 1).min(side - 1) {
                for x in column.saturating_sub(1)..=(column + 1).min(side - 1) {
                    if (x, z) != (column, row) {
                        found[count] = Some(x + z * side);
                        count += 1;
                    }
                }
            }
            found
        };

        // Priority-flood with a tiny gradient so filled flats still drain.
        let mut filled = vec![f64::INFINITY; heights.len()];
        let mut queue = BinaryHeap::new();
        for (index, &height) in heights.iter().enumerate() {
            let (column, row) = (index % side, index / side);
            let edge = column == 0 || row == 0 || column == side - 1 || row == side - 1;
            if edge || height < sea_level {
                filled[index] = height;
                queue.push(Pending { height, index });
            }
        }
        let mut order = Vec::with_capacity(heights.len());
        while let Some(Pending { height, index }) = queue.pop() {
            if height > filled[index] {
                continue;
            }
            order.push(index);
            for neighbour in neighbours(index).into_iter().flatten() {
                let raised = heights[neighbour].max(height + 1.0e-3);
                if raised < filled[neighbour] && filled[neighbour].is_infinite() {
                    filled[neighbour] = raised;
                    queue.push(Pending {
                        height: raised,
                        index: neighbour,
                    });
                }
            }
        }

        let mut downstream = vec![usize::MAX; heights.len()];
        for index in 0..heights.len() {
            let mut lowest = filled[index];
            for neighbour in neighbours(index).into_iter().flatten() {
                if filled[neighbour] < lowest {
                    lowest = filled[neighbour];
                    downstream[index] = neighbour;
                }
            }
        }
        let mut area = vec![1.0_f64; heights.len()];
        for &index in order.iter().rev() {
            if downstream[index] != usize::MAX {
                area[downstream[index]] += area[index];
            }
        }

        let cell_area_km2 = DRAINAGE_CELL_METRES * DRAINAGE_CELL_METRES / 1.0e6;
        let source = (doc.source_area / cell_area_km2).max(1.0);
        let largest = area.iter().copied().fold(source, f64::max);
        let span = (largest / source).ln().max(f64::EPSILON);
        let magnitude = |cells: f64| ((cells / source).ln() / span).clamp(0.0, 1.0);
        let lerp = |range: (f64, f64), t: f64| (range.1 - range.0).mul_add(t, range.0);
        let mut segments = Vec::new();
        for index in 0..heights.len() {
            let next = downstream[index];
            if area[index] < source || next == usize::MAX || heights[index] < sea_level {
                continue;
            }
            let (t0, t1) = (magnitude(area[index]), magnitude(area[next]));
            segments.push(Segment {
                from: [position(index % side), position(index / side)],
                to: [position(next % side), position(next / side)],
                level: [filled[index], filled[next].max(sea_level)],
                half_width: [lerp(doc.half_width, t0), lerp(doc.half_width, t1)],
                depth: [lerp(doc.depth, t0), lerp(doc.depth, t1)],
            });
        }

        let bucket_count = bucket_of(WORLD_HALF_EXTENT_METERS, usize::MAX) + 1;
        let widest = doc.half_width.0.max(doc.half_width.1);
        let reach = widest + VALLEY_REACH_METRES + doc.meander.abs();
        let mut buckets = vec![Vec::new(); bucket_count * bucket_count];
        for (index, segment) in segments.iter().enumerate() {
            let lo = [
                segment.from[0].min(segment.to[0]) - reach,
                segment.from[1].min(segment.to[1]) - reach,
            ];
            let hi = [
                segment.from[0].max(segment.to[0]) + reach,
                segment.from[1].max(segment.to[1]) + reach,
            ];
            let (x0, z0) = (
                bucket_of(lo[0], bucket_count),
                bucket_of(lo[1], bucket_count),
            );
            let (x1, z1) = (
                bucket_of(hi[0], bucket_count),
                bucket_of(hi[1], bucket_count),
            );
            for z in z0..=z1 {
                for x in x0..=x1 {
                    buckets[x + z * bucket_count]
                        .push(u32::try_from(index).expect("segment count fits u32"));
                }
            }
        }
        let meander_noise = |salt: i32| {
            NoiseGen::new(
                &NoiseDoc {
                    kind: NoiseKind::SimplexSmooth,
                    fractal: Fractal::Fbm,
                    octaves: 2,
                    freq: 1.0 / 260.0,
                    lacunarity: 2.3,
                    gain: 0.45,
                    amp: 1.0,
                    offset: 0.0,
                    dims: Dims::Two,
                    seed: 0,
                },
                seed.wrapping_add(salt),
            )
        };
        Self {
            segments,
            buckets,
            bucket_count,
            bank_slope: doc.bank_slope,
            meander: doc.meander,
            meander_x: meander_noise(101),
            meander_z: meander_noise(211),
            reach,
        }
    }

    pub(crate) fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Valley height at a column, if a river is close enough to shape it.
    pub(crate) fn valley(&self, x: f64, z: f64) -> Option<Valley> {
        let bucket = &self.buckets
            [bucket_of(x, self.bucket_count) + bucket_of(z, self.bucket_count) * self.bucket_count];
        if bucket.is_empty() {
            return None;
        }
        let point = [
            self.meander.mul_add(self.meander_x.sample(x, 0.0, z), x),
            self.meander.mul_add(self.meander_z.sample(x, 0.0, z), z),
        ];
        let mut result: Option<Valley> = None;
        for &index in bucket {
            let segment = &self.segments[index as usize];
            let (distance, along) = distance_to_segment(point, segment.from, segment.to);
            let mix = |pair: [f64; 2]| (pair[1] - pair[0]).mul_add(along, pair[0]);
            let (level, half_width, depth) = (
                mix(segment.level),
                mix(segment.half_width),
                mix(segment.depth),
            );
            let mut height = if distance < half_width {
                let across = distance / half_width;
                depth.mul_add(-(1.0 - across * across), level)
            } else {
                (distance - half_width).mul_add(self.bank_slope, level)
            };
            let outer = half_width + VALLEY_REACH_METRES;
            height += smoothstep(outer - VALLEY_FADE_METRES, outer, distance) * RIVER_LIFT_METRES;
            if result.is_none_or(|best| height < best.height) {
                result = Some(Valley {
                    height,
                    distance: result.map_or(distance, |best| best.distance.min(distance)),
                });
            } else if let Some(best) = &mut result {
                best.distance = best.distance.min(distance);
            }
        }
        result
    }

    /// Lowest valley height any column in the box could carve to, if any.
    ///
    /// A valley only rises with distance from its centre line, so each
    /// segment is bounded at the box's nearest point, moved closer by the
    /// largest meander.
    pub(crate) fn lowest_valley(&self, x: Interval, z: Interval) -> Option<f64> {
        if !(x.lo.is_finite() && x.hi.is_finite() && z.lo.is_finite() && z.hi.is_finite()) {
            return None;
        }
        let (x0, x1) = (
            bucket_of(x.lo, self.bucket_count),
            bucket_of(x.hi, self.bucket_count),
        );
        let (z0, z1) = (
            bucket_of(z.lo, self.bucket_count),
            bucket_of(z.hi, self.bucket_count),
        );
        let box_centre = [(x.lo + x.hi) * 0.5, (z.lo + z.hi) * 0.5];
        let box_radius = (x.hi - x.lo).hypot(z.hi - z.lo) * 0.5;
        let mut lowest: Option<f64> = None;
        for bucket_z in z0..=z1 {
            for bucket_x in x0..=x1 {
                for &index in &self.buckets[bucket_x + bucket_z * self.bucket_count] {
                    let segment = &self.segments[index as usize];
                    let (to_centre, _) = distance_to_segment(box_centre, segment.from, segment.to);
                    let nearest =
                        (to_centre - box_radius - self.meander.abs() * std::f64::consts::SQRT_2)
                            .max(0.0);
                    if nearest > self.reach {
                        continue;
                    }
                    let level = segment.level[0].min(segment.level[1]);
                    let half_width = segment.half_width[0].max(segment.half_width[1]);
                    let depth = segment.depth[0].max(segment.depth[1]);
                    let floor = if nearest < half_width {
                        level - depth
                    } else {
                        (nearest - half_width).mul_add(self.bank_slope, level)
                    };
                    lowest = Some(lowest.map_or(floor, |value| value.min(floor)));
                }
            }
        }
        lowest
    }
}

/// Points per side of the drainage grid.
pub(crate) fn drainage_side() -> usize {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the world is a few hundred drainage cells across"
    )]
    let side = (2.0 * WORLD_HALF_EXTENT_METERS / DRAINAGE_CELL_METRES).ceil() as usize + 1;
    side
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "clamped to the bucket grid"
)]
fn bucket_of(coordinate: f64, count: usize) -> usize {
    (((coordinate + WORLD_HALF_EXTENT_METERS) / BUCKET_METRES)
        .floor()
        .max(0.0) as usize)
        .min(count - 1)
}

fn distance_to_segment(point: [f64; 2], from: [f64; 2], to: [f64; 2]) -> (f64, f64) {
    let segment = [to[0] - from[0], to[1] - from[1]];
    let offset = [point[0] - from[0], point[1] - from[1]];
    let length_squared = segment[0].mul_add(segment[0], segment[1] * segment[1]);
    let along = if length_squared <= f64::EPSILON {
        0.0
    } else {
        (offset[0].mul_add(segment[0], offset[1] * segment[1]) / length_squared).clamp(0.0, 1.0)
    };
    (
        (offset[0] - segment[0] * along).hypot(offset[1] - segment[1] * along),
        along,
    )
}

#[cfg(test)]
mod tests {
    use super::{RiverNetwork, drainage_side};
    use crate::generation::spec::RiversDoc;

    fn doc() -> RiversDoc {
        RiversDoc {
            source_area: 0.2,
            half_width: (3.0, 20.0),
            depth: (1.0, 6.0),
            bank_slope: 0.5,
            meander: 0.0,
        }
    }

    #[test]
    fn a_tilted_plane_drains_downhill_to_the_sea() {
        let side = drainage_side();
        let heights: Vec<f64> = (0..side * side)
            .map(|index| {
                let column = index % side;
                #[expect(clippy::cast_precision_loss, reason = "small grid")]
                let height = column as f64 * 0.8 - 20.0;
                height
            })
            .collect();
        let network = RiverNetwork::trace(&doc(), 0.0, &heights, 5);
        assert!(network.segment_count() > 0);
        for segment in &network.segments {
            assert!(segment.level[1] <= segment.level[0] + 1.0e-9);
            assert!(segment.to[0] <= segment.from[0]);
        }
    }
}
