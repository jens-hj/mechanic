//! A grown tree as an octree of how full each cell is, for terrain sampled
//! too coarsely for twigs.
//!
//! The finest level holds, for every 40 cm cell around the tree, how much of
//! it is wood and how much is leaves. Each coarser level merges eight cells
//! of the level below, as the terrain octree's levels do. A lattice reads the
//! finest level whose cells are wide enough for it to hold: twigs too thin
//! for it average away, while the crown keeps the shape the tree's own
//! branches and leaf clumps give it.
//!
//! Stems are drawn apart from the octree, as capsules no thinner than the
//! lattice holds in one piece, so every crown stands on its trunk.

use std::collections::VecDeque;

use bevy_math::DVec3;

use super::model::{Parts, RESOLVED_RADIUS, capsule_density};
use super::{Part, SpeciesSpec, TreeModel};
use crate::TERRAIN_CELL_METERS;

use super::forest::MAX_GROWN_STRIDE;

/// Levels in a tree's octree: cells 40 cm to 6.4 m wide.
const LEVELS: usize = 5;

/// Fullness at which a cell's surface is drawn. Below half, as a canopy with
/// holes in it reads as solid from afar.
const DRAWN_FILL: f64 = 0.4;

/// Lattice spacings across the cells a lattice reads. A full cell among
/// empty ones is drawn `1 - DRAWN_FILL` of a cell from its centre, so at this
/// width a branch one cell thick is as thick as the lattice holds in one
/// piece ([`RESOLVED_RADIUS`] spacings).
const CELL_SPACINGS: f64 = RESOLVED_RADIUS / (1.0 - DRAWN_FILL);

/// Layers of cells each merge looks through, from the finest level up: the
/// 80 cm level is a plain mean, coarser ones fill a porous crown.
const MERGE_LAYERS: [i32; 3] = [1, 2, 3];

/// Fullness stored per cell: 0 to this.
const FULL: f64 = 255.0;

/// Edge of the finest cells, in metres: twice the coarsest lattice spacing
/// that samples grown trees, the finest that samples this octree.
fn finest_cell() -> f64 {
    f64::from(MAX_GROWN_STRIDE * 2) * TERRAIN_CELL_METERS
}

/// One level of a tree's octree.
#[derive(Clone, Debug, PartialEq)]
struct Level {
    /// Edge of a cell, in metres.
    cell: f64,
    dims: [usize; 3],
    /// How much of each cell is wood, and how much is leaves, from 0 to
    /// [`FULL`].
    fill: Vec<[u8; 2]>,
    /// Stems the level draws: those below a cell into its crown, which
    /// hides them above.
    stems: Vec<Stem>,
    /// Bounds of those stems at their own thickness.
    stem_bounds: (DVec3, DVec3),
}

impl Level {
    const fn index(&self, [x, y, z]: [usize; 3]) -> usize {
        x + self.dims[0] * (y + self.dims[1] * z)
    }

    /// Wood and leaves at a cell, nothing beyond the level's edges.
    fn at(&self, [x, y, z]: [i64; 3]) -> [f64; 2] {
        let inside = |value: i64, axis: usize| {
            usize::try_from(value)
                .ok()
                .filter(|&value| value < self.dims[axis])
        };
        match (inside(x, 0), inside(y, 1), inside(z, 2)) {
            (Some(x), Some(y), Some(z)) => {
                let [wood, leaves] = self.fill[self.index([x, y, z])];
                [f64::from(wood) / FULL, f64::from(leaves) / FULL]
            }
            _ => [0.0; 2],
        }
    }

    /// The level above. Each cell holds what its eight cells below hold, as
    /// seen through `layers` layers of them: with more than one, a crown with
    /// gaps between its clumps reads as solid from afar, while a lone twig
    /// still averages away.
    fn merged(&self, layers: i32) -> Self {
        let dims = self.dims.map(|dim| dim.div_ceil(2));
        let mut fill = vec![[0_u8; 2]; dims[0] * dims[1] * dims[2]];
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let mut sum = [0_u32; 2];
                    for child in 0..8 {
                        let at = [
                            2 * x + (child & 1),
                            2 * y + (child >> 1 & 1),
                            2 * z + (child >> 2),
                        ];
                        if at.iter().zip(self.dims).all(|(&value, dim)| value < dim) {
                            let [wood, leaves] = self.fill[self.index(at)];
                            sum[0] += u32::from(wood);
                            sum[1] += u32::from(leaves);
                        }
                    }
                    let mean = sum.map(|total| f64::from(total) / (8.0 * FULL));
                    let total = mean[0] + mean[1];
                    let seen = 1.0 - (1.0 - total.min(1.0)).powi(layers);
                    let gain = if total > 0.0 { seen / total } else { 0.0 };
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "shares of a full cell are bytes"
                    )]
                    let merged = mean.map(|share| (share * gain * FULL).round().min(FULL) as u8);
                    fill[x + dims[0] * (y + dims[1] * z)] = merged;
                }
            }
        }
        Self {
            cell: self.cell * 2.0,
            dims,
            fill,
            stems: Vec::new(),
            stem_bounds: (DVec3::ZERO, DVec3::ZERO),
        }
    }

    /// Draws the stems that start below `top`, metres above the tree's base.
    fn keep_stems(&mut self, stems: &[Stem], top: f64) {
        self.stems = stems
            .iter()
            .filter(|stem| stem.a.y.min(stem.b.y) < top)
            .copied()
            .collect();
        self.stem_bounds = self.stems.iter().fold(
            (DVec3::INFINITY, DVec3::NEG_INFINITY),
            |(low, high), stem| {
                let radius = stem.ra.max(stem.rb);
                (
                    low.min(stem.a.min(stem.b) - radius),
                    high.max(stem.a.max(stem.b) + radius),
                )
            },
        );
    }

    /// Height above the tree's base of the lowest leaves the level draws.
    fn crown_bottom(&self, min: DVec3) -> Option<f64> {
        let layer = self.dims[0] * self.dims[1];
        (0..self.dims[1]).find_map(|y| {
            let drawn = (0..self.dims[2]).any(|z| {
                (0..self.dims[0]).any(|x| {
                    let [wood, leaves] = self.fill[x + self.dims[0] * y + layer * z];
                    leaves >= wood
                        && f64::from(u16::from(wood) + u16::from(leaves)) / FULL >= DRAWN_FILL
                })
            });
            #[expect(clippy::cast_precision_loss, reason = "a few hundred cells")]
            drawn.then_some(min.y + y as f64 * self.cell)
        })
    }

    /// Keeps only what is joined to a stem, and draws it as whole cells.
    /// Cells drawn but cut off from every stem are thinned below the drawn
    /// fullness; cells joined only through wood too thin to draw have that
    /// wood filled in, so no clump of leaves floats however coarse the level.
    fn hold_up(&mut self, min: DVec3) {
        const NONE: u32 = u32::MAX;
        let held_by = std::mem::take(&mut self.stems);
        let drawn = |fill: [u8; 2]| f64::from(u16::from(fill[0]) + u16::from(fill[1])) / FULL;
        let mut from = vec![None::<u32>; self.fill.len()];
        let mut queue = VecDeque::new();
        for stem in &held_by {
            let length = stem.a.distance(stem.b);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a stem is a few hundred cells long at most"
            )]
            let steps = (length / (self.cell * 0.25)).ceil() as usize + 1;
            for step in 0..=steps {
                #[expect(clippy::cast_precision_loss, reason = "a few hundred steps")]
                let point = stem.a.lerp(stem.b, step as f64 / steps as f64);
                if let Some(at) = self.cell_of(min, point) {
                    let index = self.index(at);
                    if from[index].is_none() {
                        from[index] = Some(NONE);
                        queue.push_back(at);
                    }
                }
            }
        }
        // Breadth first through what is drawn and through any wood at all,
        // so each cell remembers its shortest way back to a stem.
        while let Some(at @ [x, y, z]) = queue.pop_front() {
            #[expect(clippy::cast_possible_truncation, reason = "a few million cells")]
            let here = self.index(at) as u32;
            let steps = [
                [x.wrapping_sub(1), y, z],
                [x + 1, y, z],
                [x, y.wrapping_sub(1), z],
                [x, y + 1, z],
                [x, y, z.wrapping_sub(1)],
                [x, y, z + 1],
            ];
            for next in steps {
                if next.iter().zip(self.dims).all(|(&value, dim)| value < dim) {
                    let index = self.index(next);
                    let fill = self.fill[index];
                    if from[index].is_none() && (fill[0] > 0 || drawn(fill) >= DRAWN_FILL) {
                        from[index] = Some(here);
                        queue.push_back(next);
                    }
                }
            }
        }
        let below = DRAWN_FILL * 0.75;
        for index in 0..self.fill.len() {
            let total = drawn(self.fill[index]);
            if total < DRAWN_FILL {
                continue;
            }
            match from[index] {
                None => {
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "thinned bytes stay bytes"
                    )]
                    let thin = |value: u8| (f64::from(value) * below / total).floor() as u8;
                    self.fill[index] = self.fill[index].map(thin);
                }
                Some(mut back) => {
                    // Drawn cells are made full, keeping their share of wood
                    // and leaves: a cell barely over the drawn fullness would
                    // join its neighbours by a neck thinner than any lattice
                    // holds.
                    let [wood, _] = self.fill[index];
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "a share of a full cell is a byte"
                    )]
                    let wood = (f64::from(wood) / total).round().min(FULL) as u8;
                    self.fill[index] = [wood, u8::MAX - wood];
                    while back != NONE && drawn(self.fill[back as usize]) < DRAWN_FILL {
                        let [_, leaves] = self.fill[back as usize];
                        self.fill[back as usize] = [u8::MAX - leaves, leaves];
                        back = from[back as usize].unwrap_or(NONE);
                    }
                }
            }
        }
        self.stems = held_by;
    }

    /// The cell holding a point, if the level reaches it.
    fn cell_of(&self, min: DVec3, point: DVec3) -> Option<[usize; 3]> {
        let local = (point - min) / self.cell;
        let at = |value: f64, axis: usize| {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "checked against the level first"
            )]
            let cell = value.floor() as usize;
            (value >= 0.0 && cell < self.dims[axis]).then_some(cell)
        };
        Some([at(local.x, 0)?, at(local.y, 1)?, at(local.z, 2)?])
    }

    /// Wood and leaves at a point, blended between cell centres.
    fn blend(&self, min: DVec3, point: DVec3) -> [f64; 2] {
        let local = (point - min) / self.cell - 0.5;
        let base = local.floor();
        let t = local - base;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a point near the tree is near its cells"
        )]
        let [bx, by, bz] = [base.x as i64, base.y as i64, base.z as i64];
        let mut blended = [0.0; 2];
        for corner in 0..8_i64 {
            let (dx, dy, dz) = (corner & 1, corner >> 1 & 1, corner >> 2);
            let weight = (if dx == 1 { t.x } else { 1.0 - t.x })
                * (if dy == 1 { t.y } else { 1.0 - t.y })
                * (if dz == 1 { t.z } else { 1.0 - t.z });
            if weight > 0.0 {
                let [wood, leaves] = self.at([bx + dx, by + dy, bz + dz]);
                blended[0] += weight * wood;
                blended[1] += weight * leaves;
            }
        }
        blended
    }
}

/// A stem below the middle of the crown, drawn whatever the level.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Stem {
    a: DVec3,
    b: DVec3,
    ra: f64,
    rb: f64,
}

/// A grown tree as an octree of how full its cells are: what terrain too
/// coarse for twigs draws in its place.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeLod {
    /// Height of the tree it was made from, in metres.
    height: f64,
    /// Lowest corner of every level, relative to the tree's base.
    min: DVec3,
    /// Highest corner of what any level or stem draws, relative to the base.
    max: DVec3,
    /// Farthest anything reaches sideways from the base.
    reach: f64,
    /// Finest first.
    levels: Vec<Level>,
}

impl TreeLod {
    /// The octree of a tree grown as `species`.
    #[must_use]
    pub fn new(model: &TreeModel, species: &SpeciesSpec) -> Self {
        let origin = model.origin;
        let finest = finest_cell();
        let (low, high) = model.solid_bounds();
        let min = low - origin;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a tree is a few hundred cells across"
        )]
        let dims = ((high - low) / finest)
            .ceil()
            .max(DVec3::ONE)
            .to_array()
            .map(|dim| dim as usize);
        let mut fill = vec![[0_u8; 2]; dims[0] * dims[1] * dims[2]];
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    #[expect(clippy::cast_precision_loss, reason = "a few hundred cells")]
                    let centre = low + (DVec3::new(x as f64, y as f64, z as f64) + 0.5) * finest;
                    fill[x + dims[0] * (y + dims[1] * z)] = cell_fill(model, centre, finest);
                }
            }
        }

        // Stems stand at most up to the middle of the crown; each level draws
        // them only a cell into the leaves it draws.
        let crown_from = species.crown_base * model.height;
        let top = crown_from + 0.5 * (model.height - crown_from);
        let stems = model
            .axes
            .iter()
            .filter(|axis| axis.stem && axis.part == Part::Wood)
            .flat_map(|axis| {
                let (first, last) = axis.segments;
                model.segments[first as usize..last as usize].iter()
            })
            .filter(|segment| segment.a.y.min(segment.b.y) - origin.y < top)
            .map(|segment| Stem {
                a: segment.a - origin,
                b: segment.b - origin,
                ra: segment.ra,
                rb: segment.rb,
            })
            .collect::<Vec<_>>();

        let mut levels = Vec::with_capacity(LEVELS);
        // Each level merges the one below as grown, before any is held up.
        let mut grown = Level {
            cell: finest,
            dims,
            fill,
            stems: Vec::new(),
            stem_bounds: (DVec3::ZERO, DVec3::ZERO),
        };
        for level in 0..LEVELS {
            // Near levels keep the crown's own volume, so neighbouring crowns
            // stay apart; only levels far enough for gaps to vanish fill them.
            let next = grown.merged(MERGE_LAYERS[level.min(MERGE_LAYERS.len() - 1)]);
            let mut held = grown;
            let stem_top = held
                .crown_bottom(min)
                .map_or(top, |bottom| (bottom + held.cell).min(top));
            held.keep_stems(&stems, stem_top);
            held.hold_up(min);
            levels.push(held);
            grown = next;
        }

        let coarsest = &levels[levels.len() - 1];
        #[expect(clippy::cast_precision_loss, reason = "a few cells")]
        let mut max = min
            + DVec3::new(
                coarsest.dims[0] as f64,
                coarsest.dims[1] as f64,
                coarsest.dims[2] as f64,
            ) * coarsest.cell;
        for stem in &stems {
            max = max.max(stem.a.max(stem.b) + stem.ra.max(stem.rb));
        }
        let reach = [min.x, max.x]
            .into_iter()
            .flat_map(|x| [min.z, max.z].map(|z| x.hypot(z)))
            .fold(0.0, f64::max);
        Self {
            height: model.height,
            min,
            max,
            reach,
            levels,
        }
    }

    /// Height of the tree it was made from, in metres.
    #[must_use]
    pub const fn height(&self) -> f64 {
        self.height
    }

    /// Bounds of what it draws relative to the tree's base, turned any way
    /// about the vertical, with stems at least `floor` metres thick.
    #[must_use]
    pub fn bounds(&self, floor: f64) -> (DVec3, DVec3) {
        let side = self.reach + floor;
        (
            DVec3::new(-side, self.min.y - floor, -side),
            DVec3::new(side, self.max.y + floor, side),
        )
    }

    /// Density at a point relative to the tree's base, positive inside, as a
    /// lattice `spacing` metres apart draws it, with stems at least `floor`
    /// metres thick, and the part that dominates there. `None` beyond
    /// everything it draws.
    #[must_use]
    pub fn sample(&self, point: DVec3, spacing: f64, floor: f64) -> Option<(f64, Part)> {
        self.sample_parts(point, spacing, floor, Parts::Solid)
    }

    /// The finest level whose cells a lattice `spacing` metres apart holds.
    fn level_for(&self, spacing: f64) -> &Level {
        self.levels
            .iter()
            .find(|level| level.cell >= spacing * CELL_SPACINGS)
            .unwrap_or(&self.levels[self.levels.len() - 1])
    }

    pub(crate) fn sample_parts(
        &self,
        point: DVec3,
        spacing: f64,
        floor: f64,
        parts: Parts,
    ) -> Option<(f64, Part)> {
        let (low, high) = self.bounds(floor);
        if point.cmplt(low).any() || point.cmpgt(high).any() {
            return None;
        }
        let level = self.level_for(spacing);
        let [wood, leaves] = level.blend(self.min, point);
        let full = match parts {
            Parts::Wood => wood,
            Parts::All | Parts::Solid => wood + leaves,
        };
        let mut best = (full - DRAWN_FILL) * level.cell;
        let mut part = if wood >= leaves || parts == Parts::Wood {
            Part::Wood
        } else {
            Part::Foliage
        };
        let (stems_low, stems_high) = level.stem_bounds;
        if point.cmpge(stems_low - floor).all() && point.cmple(stems_high + floor).all() {
            for stem in &level.stems {
                let density = capsule_density(
                    point,
                    stem.a,
                    stem.b,
                    stem.ra.max(floor),
                    stem.rb.max(floor),
                );
                if density >= best {
                    best = density;
                    part = Part::Wood;
                }
            }
        }
        Some((best, part))
    }
}

/// How much of a cell centred at `centre` is wood, and how much is leaves:
/// cells the surface misses are judged from their centre, the rest from
/// eight samples, each blending across its own width.
fn cell_fill(model: &TreeModel, centre: DVec3, cell: f64) -> [u8; 2] {
    let half_diagonal = cell * 0.5 * 3.0_f64.sqrt();
    let fill = |share: [f64; 2]| {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "shares are clamped to 0..=1"
        )]
        share.map(|value| (value.clamp(0.0, 1.0) * FULL).round() as u8)
    };
    let split = |part: Part, amount: f64| match part {
        Part::Wood | Part::Root => [amount, 0.0],
        Part::Foliage => [0.0, amount],
    };
    let Some((density, part)) = model.sample_parts(centre, Parts::Solid, 0.0) else {
        return [0; 2];
    };
    let density = f64::from(density);
    if density >= half_diagonal {
        return fill(split(part, 1.0));
    }
    if density <= -half_diagonal {
        return [0; 2];
    }
    let quarter = cell * 0.25;
    let mut share = [0.0; 2];
    for corner in 0..8 {
        let offset = DVec3::new(
            if corner & 1 == 0 { -quarter } else { quarter },
            if corner >> 1 & 1 == 0 {
                -quarter
            } else {
                quarter
            },
            if corner >> 2 == 0 { -quarter } else { quarter },
        );
        if let Some((density, part)) = model.sample_parts(centre + offset, Parts::Solid, 0.0) {
            let amount = (0.5 + f64::from(density) / (2.0 * quarter)).clamp(0.0, 1.0) / 8.0;
            let [wood, leaves] = split(part, amount);
            share[0] += wood;
            share[1] += leaves;
        }
    }
    fill(share)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::flora::grow_tree;

    fn preset(name: &str) -> SpeciesSpec {
        SpeciesSpec::embedded()
            .into_iter()
            .find(|species| species.name == name)
            .expect("a preset")
    }

    /// Points `spacing` apart over a box where `solid` holds.
    fn solid_points(
        low: DVec3,
        high: DVec3,
        spacing: f64,
        solid: impl Fn(DVec3) -> bool,
    ) -> Vec<DVec3> {
        let mut points = Vec::new();
        let mut z = low.z;
        while z <= high.z {
            let mut y = low.y;
            while y <= high.y {
                let mut x = low.x;
                while x <= high.x {
                    let point = DVec3::new(x, y, z);
                    if solid(point) {
                        points.push(point);
                    }
                    x += spacing;
                }
                y += spacing;
            }
            z += spacing;
        }
        points
    }

    #[test]
    fn every_level_keeps_the_crown_and_stays_within_a_cell_of_it() {
        for name in ["oak", "spruce", "birch", "willow"] {
            let species = preset(name);
            let model = grow_tree(&species, 11, DVec3::new(3.0, 7.0, -2.0));
            let lod = TreeLod::new(&model, &species);
            let (low, high) = lod.bounds(0.0);
            let grown = solid_points(low, high, 0.2, |point| {
                model
                    .sample(point + model.origin)
                    .is_some_and(|(density, part)| density > 0.0 && part == Part::Foliage)
            });
            let crown = grown.iter().fold(
                (DVec3::INFINITY, DVec3::NEG_INFINITY),
                |(low, high), &point| (low.min(point), high.max(point)),
            );
            #[expect(clippy::cast_precision_loss, reason = "a few thousand points")]
            let grown_volume = grown.len() as f64 * 0.2_f64.powi(3);
            for spacing in [0.4, 0.8, 1.6, 3.2] {
                let drawn = solid_points(low, high, spacing, |point| {
                    lod.sample(point, spacing, spacing * RESOLVED_RADIUS)
                        .is_some_and(|(density, part)| density > 0.0 && part == Part::Foliage)
                });
                #[expect(clippy::cast_precision_loss, reason = "a few thousand points")]
                let drawn_volume = drawn.len() as f64 * spacing.powi(3);
                // Coarser levels fill the gaps between clumps, as a crown
                // reads from afar, so they draw more than the leaves alone,
                // but never past the crown by more than a cell.
                assert!(
                    drawn_volume > 0.5 * grown_volume,
                    "{name} draws {drawn_volume:.1} m³ of leaves at {spacing} m, \
                     grown {grown_volume:.1} m³"
                );
                let cell = lod.level_for(spacing).cell;
                let outside = drawn
                    .iter()
                    .filter(|point| {
                        point.cmplt(crown.0 - cell).any() || point.cmpgt(crown.1 + cell).any()
                    })
                    .count();
                assert_eq!(
                    outside, 0,
                    "{name} draws leaves past its crown at {spacing} m"
                );
            }
        }
    }

    #[test]
    fn nothing_is_drawn_above_the_tree_and_its_trunk_stands_at_every_level() {
        let species = preset("oak");
        let model = grow_tree(&species, 11, DVec3::new(3.0, 7.0, -2.0));
        let lod = TreeLod::new(&model, &species);
        for spacing in [0.4, 0.8, 1.6, 3.2] {
            let floor = spacing * RESOLVED_RADIUS;
            let above = DVec3::Y * (model.height + 2.0 * spacing + 2.0);
            assert!(
                lod.sample(above, spacing, floor)
                    .is_none_or(|(density, _)| density < 0.0),
                "solid above the tree at {spacing} m"
            );
            let (density, part) = lod
                .sample(DVec3::Y * 0.5, spacing, floor)
                .expect("the trunk is inside the octree");
            assert!(
                density > 0.0 && part == Part::Wood,
                "no trunk at {spacing} m"
            );
        }
    }
}
