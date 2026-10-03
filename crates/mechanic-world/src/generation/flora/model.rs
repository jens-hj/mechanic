//! A grown tree as primitives, and its signed density.

use bevy_math::DVec3;

use super::forest::MAX_GROWN_STRIDE;
use super::noise::foliage_noise;
use super::{FOLIAGE_DENT_SLACK, FOLIAGE_FILL_BIAS, FOLIAGE_NOISE_SLOPE};
use crate::TERRAIN_CELL_METERS;

/// Distance beyond a primitive's surface over which its density stays exact;
/// farther points report nothing. It spans a whole lattice edge at the
/// coarsest stride that samples grown trees, so every crossing's open
/// corner knows the tree it meets.
pub(crate) const SAMPLE_MARGIN: f64 = 0.3;

/// Thinnest wood a lattice holds in one piece, as a fraction of its spacing.
/// No point on a branch's axis lies farther than half a cube diagonal, 0.87
/// spacings, from a lattice corner, so a branch at least this thick puts a
/// solid corner beside every point of its axis; and as the point moves on,
/// its nearest corner steps to a neighbour along a cube edge, so those solid
/// corners join edge to edge, and the surface meshed over them is one piece.
const RESOLVED_RADIUS: f64 = 0.9;

/// Thinnest wood, in metres, that a lattice of `stride` cells holds in one
/// piece. Thinner branches are sampled this thick, so they never break up
/// into floating shards.
pub(crate) fn wood_floor(stride: i32) -> f64 {
    f64::from(stride) * TERRAIN_CELL_METERS * RESOLVED_RADIUS
}

/// Edge of the buckets that index primitives for point queries.
const BUCKET_METRES: f64 = 1.0;

/// What a solid point of a tree is made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    /// Trunk, branch, and twig wood.
    Wood,
    /// Wood below the base.
    Root,
    /// Leaves or needles.
    Foliage,
}

/// Which parts of a tree a sample sees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Parts {
    /// Everything, roots included: what the ground is made of.
    All,
    /// What the tree adds to the ground: wood and leaves. Roots only run
    /// through ground that is already there.
    Solid,
    /// Wood alone, as water sees the tree: it runs through leaves.
    Wood,
}

/// A tapered capsule of wood.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    /// Start, nearer the base.
    pub a: DVec3,
    /// End, nearer the tip.
    pub b: DVec3,
    /// Radius at `a`.
    pub ra: f64,
    /// Radius at `b`.
    pub rb: f64,
    /// [`Part::Wood`] or [`Part::Root`].
    pub part: Part,
    /// Index of the axis this segment belongs to.
    pub axis: u32,
}

/// One unbranched run of segments: a stem, a branch, a twig, or a root.
#[derive(Clone, Debug, PartialEq)]
pub struct Axis {
    /// 0 for stems and their co-dominant forks, one more per lateral.
    pub order: u8,
    /// [`Part::Wood`] or [`Part::Root`].
    pub part: Part,
    /// Whether the axis rises from the base as a stem.
    pub stem: bool,
    /// Segment whose end this axis starts from, if any.
    pub parent_segment: Option<u32>,
    /// First segment and one past the last.
    pub segments: (u32, u32),
    /// Splits along this axis.
    pub splits: u32,
    /// Children created by those splits.
    pub children: u32,
}

/// A capsule of foliage; a ball when both ends coincide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoliageBlob {
    /// One end.
    pub a: DVec3,
    /// Other end.
    pub b: DVec3,
    /// Radius in metres.
    pub radius: f64,
}

/// A grown tree: wood and root segments, foliage blobs, and an index for
/// point queries.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeModel {
    /// Base of the tree, where the stems leave the ground.
    pub origin: DVec3,
    /// Height drawn for this tree, in metres.
    pub height: f64,
    /// Crown radius aimed for, in metres.
    pub crown_radius: f64,
    /// Radius of the whole trunk at its base, before the flare.
    pub base_radius: f64,
    /// Wood and roots.
    pub segments: Vec<Segment>,
    /// Unbranched runs of segments.
    pub axes: Vec<Axis>,
    /// Foliage.
    pub foliage: Vec<FoliageBlob>,
    /// Fraction of foliage blobs that is filled.
    pub foliage_density: f64,
    /// Lowest corner of every primitive's bounds, wood as thick as the
    /// coarsest lattice that samples grown trees draws it.
    pub min: DVec3,
    /// Highest corner of every primitive's bounds, likewise.
    pub max: DVec3,
    /// Bounds of the wood and leaves alone: what the tree adds to the ground.
    solid: (DVec3, DVec3),
    /// Radius of the largest foliage blob: how deep holes can cut.
    foliage_depth: f64,
    noise_seed: u64,
    buckets: Buckets,
}

impl TreeModel {
    #[expect(clippy::too_many_arguments, reason = "every part of a grown tree")]
    pub(super) fn new(
        origin: DVec3,
        height: f64,
        crown_radius: f64,
        base_radius: f64,
        segments: Vec<Segment>,
        axes: Vec<Axis>,
        foliage: Vec<FoliageBlob>,
        foliage_density: f64,
        noise_seed: u64,
    ) -> Self {
        // Bucket items are 16-bit; no preset comes near the limit.
        let mut segments = segments;
        let mut foliage = foliage;
        let limit = usize::from(u16::MAX);
        segments.truncate(limit);
        foliage.truncate(limit - segments.len());
        let bounds = |a: DVec3, b: DVec3, radius: f64| (a.min(b) - radius, a.max(b) + radius);
        // Wood as thick as the coarsest lattice that samples it draws it.
        let thickest = wood_floor(MAX_GROWN_STRIDE);
        let primitives = segments
            .iter()
            .map(|segment| {
                bounds(
                    segment.a,
                    segment.b,
                    segment.ra.max(segment.rb).max(thickest),
                )
            })
            .chain(
                foliage
                    .iter()
                    .map(|blob| bounds(blob.a, blob.b, blob.radius)),
            )
            .collect::<Vec<_>>();
        let (min, max) = primitives.iter().fold(
            (origin, origin),
            |(low, high), &(primitive_low, primitive_high)| {
                (low.min(primitive_low), high.max(primitive_high))
            },
        );
        let solid = primitives
            .iter()
            .zip(
                segments
                    .iter()
                    .map(|segment| segment.part != Part::Root)
                    .chain(foliage.iter().map(|_| true)),
            )
            .filter(|(_, solid)| *solid)
            .fold(
                (origin, origin),
                |(low, high), (&(primitive_low, primitive_high), _)| {
                    (low.min(primitive_low), high.max(primitive_high))
                },
            );
        let buckets = Buckets::new(min - SAMPLE_MARGIN, max + SAMPLE_MARGIN, &primitives);
        let foliage_depth = foliage
            .iter()
            .fold(0.0, |deepest: f64, blob| deepest.max(blob.radius));
        Self {
            origin,
            height,
            crown_radius,
            base_radius,
            segments,
            axes,
            foliage,
            foliage_density,
            min,
            max,
            solid,
            foliage_depth,
            noise_seed,
            buckets,
        }
    }

    /// Bounds of the wood and leaves: where [`Parts::Solid`] can be positive
    /// or within the exact range.
    pub(crate) fn solid_bounds(&self) -> (DVec3, DVec3) {
        (self.solid.0 - SAMPLE_MARGIN, self.solid.1 + SAMPLE_MARGIN)
    }

    /// Heap and inline bytes this model holds.
    pub fn memory_bytes(&self) -> usize {
        size_of::<Self>()
            + self.segments.capacity() * size_of::<Segment>()
            + self.axes.capacity() * size_of::<Axis>()
            + self.foliage.capacity() * size_of::<FoliageBlob>()
            + self.buckets.starts.capacity() * size_of::<u32>()
            + self.buckets.items.capacity() * size_of::<u16>()
    }

    /// Signed density at a point as the finest terrain lattice holds it,
    /// positive inside, with the part that dominates there. `None` farther
    /// than about 0.3 m from every primitive.
    pub fn sample(&self, point: DVec3) -> Option<(f32, Part)> {
        self.sample_at_stride(point, 1)
    }

    /// [`Self::sample`] as a lattice of `stride` cells holds the tree: wood
    /// thinner than that lattice can draw in one piece is thickened to it.
    pub fn sample_at_stride(&self, point: DVec3, stride: i32) -> Option<(f32, Part)> {
        self.sample_parts(point, Parts::All, wood_floor(stride))
    }

    /// [`Self::sample`] over only some of the tree's parts, with wood at
    /// least `floor` metres thick.
    pub(crate) fn sample_parts(
        &self,
        point: DVec3,
        parts: Parts,
        floor: f64,
    ) -> Option<(f32, Part)> {
        let items = self.buckets.items_at(point)?;
        // Only true densities within the exact range count: a floor here
        // would lift the ground's own density wherever a bucket overlaps it.
        let mut best = f64::NEG_INFINITY;
        let mut part = None;
        // Deepest point inside any foliage blob: the leaves before holes.
        let mut shell = f64::NEG_INFINITY;
        let segment_count = self.segments.len();
        for &item in items {
            let item = item as usize;
            if item < segment_count {
                let segment = &self.segments[item];
                if segment.part == Part::Root && parts != Parts::All {
                    continue;
                }
                let density = capsule_density(
                    point,
                    segment.a,
                    segment.b,
                    segment.ra.max(floor),
                    segment.rb.max(floor),
                );
                if density > best {
                    best = density;
                    part = Some(segment.part);
                }
            } else if parts != Parts::Wood {
                let blob = &self.foliage[item - segment_count];
                shell = shell.max(capsule_density(
                    point,
                    blob.a,
                    blob.b,
                    blob.radius,
                    blob.radius,
                ));
            }
        }
        // Holes only ever take leaves away.
        if shell > best {
            let leaves = shell - self.foliage_dent(point);
            if leaves > best {
                best = leaves;
                part = Some(Part::Foliage);
            }
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "terrain stores densities as f32"
        )]
        part.filter(|_| best > -SAMPLE_MARGIN)
            .map(|part| (best as f32, part))
    }

    /// How deep holes cut into the foliage at a point: nothing where the
    /// noise falls well below the fill target, the whole depth of the leaves
    /// where it rises well above it. The noise's wavelength scales with that
    /// depth so the dent never changes faster than distance does; walking
    /// from any leaf toward its twig, the foliage only grows denser, so no
    /// clump floats free of its twig.
    fn foliage_dent(&self, point: DVec3) -> f64 {
        let wavelength = FOLIAGE_DENT_SLACK * FOLIAGE_NOISE_SLOPE * self.foliage_depth;
        let noise = foliage_noise(self.noise_seed, point / wavelength);
        // Most of a blob's volume lies near its surface, where the shallowest
        // dents already reach, so the target is biased up to keep about
        // `foliage_density` of the foliage.
        let fill = (0.5 + FOLIAGE_FILL_BIAS + self.foliage_density - noise).clamp(0.0, 1.0);
        (1.0 - fill) * self.foliage_depth
    }
}

/// Signed distance-like density of a tapered capsule, positive inside.
fn capsule_density(point: DVec3, a: DVec3, b: DVec3, ra: f64, rb: f64) -> f64 {
    let axis = b - a;
    let length_squared = axis.length_squared();
    let t = if length_squared > 0.0 {
        ((point - a).dot(axis) / length_squared).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let radius = (rb - ra).mul_add(t, ra);
    radius - point.distance(axis.mul_add(DVec3::splat(t), a))
}

/// Primitive indices per cube of space, in compressed rows.
#[derive(Clone, Debug, PartialEq)]
struct Buckets {
    min: DVec3,
    dims: [usize; 3],
    starts: Vec<u32>,
    /// Primitive indices; a tree holds fewer than 2^16 primitives.
    items: Vec<u16>,
}

impl Buckets {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "bucket counts are small and positive"
    )]
    fn new(min: DVec3, max: DVec3, primitives: &[(DVec3, DVec3)]) -> Self {
        let span = ((max - min) / BUCKET_METRES).ceil().max(DVec3::ONE);
        let dims = [span.x as usize, span.y as usize, span.z as usize];
        let mut buckets = Self {
            min,
            dims,
            starts: vec![0; dims[0] * dims[1] * dims[2] + 1],
            items: Vec::new(),
        };
        // Count, then fill: two passes over the same covered ranges.
        let mut counts = vec![0_u32; buckets.starts.len() - 1];
        for (low, high) in primitives {
            let (from, to) = buckets.range(*low - SAMPLE_MARGIN, *high + SAMPLE_MARGIN);
            for z in from[2]..=to[2] {
                for y in from[1]..=to[1] {
                    for x in from[0]..=to[0] {
                        counts[buckets.flat([x, y, z])] += 1;
                    }
                }
            }
        }
        let mut running = 0;
        for (start, count) in buckets.starts.iter_mut().zip(&counts) {
            *start = running;
            running += count;
        }
        *buckets.starts.last_mut().unwrap_or(&mut 0) = running;
        buckets.items = vec![0; running as usize];
        let mut cursor = buckets.starts.clone();
        for (index, (low, high)) in primitives.iter().enumerate() {
            let (from, to) = buckets.range(*low - SAMPLE_MARGIN, *high + SAMPLE_MARGIN);
            for z in from[2]..=to[2] {
                for y in from[1]..=to[1] {
                    for x in from[0]..=to[0] {
                        let flat = buckets.flat([x, y, z]);
                        buckets.items[cursor[flat] as usize] = index as u16;
                        cursor[flat] += 1;
                    }
                }
            }
        }
        buckets
    }

    const fn flat(&self, [x, y, z]: [usize; 3]) -> usize {
        (z * self.dims[1] + y) * self.dims[0] + x
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to the bucket grid"
    )]
    fn range(&self, low: DVec3, high: DVec3) -> ([usize; 3], [usize; 3]) {
        let cell = |value: f64, axis: usize| {
            ((value / BUCKET_METRES).floor().max(0.0) as usize).min(self.dims[axis] - 1)
        };
        let low = low - self.min;
        let high = high - self.min;
        (
            [cell(low.x, 0), cell(low.y, 1), cell(low.z, 2)],
            [cell(high.x, 0), cell(high.y, 1), cell(high.z, 2)],
        )
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "checked against the bucket grid first"
    )]
    fn items_at(&self, point: DVec3) -> Option<&[u16]> {
        let local = (point - self.min) / BUCKET_METRES;
        if local.x < 0.0 || local.y < 0.0 || local.z < 0.0 {
            return None;
        }
        let cell = [local.x as usize, local.y as usize, local.z as usize];
        if cell.iter().zip(self.dims).any(|(&value, dim)| value >= dim) {
            return None;
        }
        let flat = self.flat(cell);
        let items = &self.items[self.starts[flat] as usize..self.starts[flat + 1] as usize];
        (!items.is_empty()).then_some(items)
    }
}
