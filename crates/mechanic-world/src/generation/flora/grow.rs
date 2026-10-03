//! The one branching process every species grows by.

use std::collections::VecDeque;
use std::f64::consts::{PI, TAU};

use bevy_math::DVec3;

use super::super::grid::Hash;
use super::super::scatter::mix;
use super::model::{Axis, FoliageBlob, Part, Segment, TreeModel};
use super::species::SpeciesSpec;
use super::{
    FLARE_HEIGHT, FLARE_WIDENING, GOLDEN_ANGLE, INTERNODES, MAX_ORDER, MAX_SEGMENTS, ROOT_COUNT,
    TROPISM_RATE, TWIG_RADIUS, WOBBLE_RATE,
};

/// How an axis branches: the species' genome for wood, fixed for roots.
#[derive(Clone, Copy, Debug)]
struct Habit {
    split_chance: f64,
    split_count: (u32, u32),
    split_angle: f64,
    tropism: f64,
    wobble: f64,
    dominance: f64,
}

const ROOT_HABIT: Habit = Habit {
    split_chance: 0.5,
    split_count: (2, 3),
    split_angle: 35.0 * PI / 180.0,
    tropism: 0.0,
    wobble: 0.4,
    dominance: 0.6,
};

/// Largest tilt of co-dominant forks away from each other's axis.
const MAX_FORK_ANGLE: f64 = 30.0 * PI / 180.0;

/// Length of a lateral of a lateral, as a fraction of what remains of its
/// parent beyond the split.
const SUBLATERAL_LENGTH: f64 = 0.6;

/// Foliage ball at an axis tip, relative to the sleeve radius.
const TIP_BALL: f64 = 1.3;

/// Longest branch per metre of its base radius: thin wood stays short.
const MAX_SLENDERNESS: f64 = 240.0;

/// Inward turn of a fork that meets the crown's edge, per segment.
const FORK_RETURN: f64 = 0.3;

/// Room beyond the envelope before a branch is pruned, in metres.
const ENVELOPE_SLACK: f64 = 0.25;

/// Wood ends this far above the base rather than reach the ground.
const GROUND_CLEARANCE: f64 = 0.4;

/// Axes shorter than this are not grown.
const MIN_AXIS_LENGTH: f64 = 0.1;

/// Axis waiting to be grown.
#[derive(Clone, Copy, Debug)]
struct Bud {
    start: DVec3,
    dir: DVec3,
    length: f64,
    radius: f64,
    order: u8,
    part: Part,
    stem: bool,
    /// Whether the axis may split at all.
    splits: bool,
    parent_segment: Option<u32>,
    /// Height above the origin at which this stem's crown starts.
    crown_from: f64,
}

/// Grows one tree of `species` at `origin`. Equal seeds grow equal trees.
#[must_use]
pub fn grow_tree(species: &SpeciesSpec, seed: u64, origin: DVec3) -> TreeModel {
    let mut random = Hash::new(seed, 0, 0);
    let height = drawn_height(species, &mut random);
    let crown_radius = species.width * height * 0.5;
    let base_radius = (species.girth * height * 0.5).max(TWIG_RADIUS);
    let stems = random.integer(species.stems.0, species.stems.1);
    let mut grower = Grower {
        species,
        random,
        origin,
        height,
        crown_radius,
        base_radius,
        segments: Vec::new(),
        axes: Vec::new(),
        foliage: Vec::new(),
        buds: VecDeque::new(),
    };
    grower.plant_stems(stems);
    grower.plant_roots();
    while let Some(bud) = grower.buds.pop_front() {
        grower.grow_axis(bud);
    }
    let noise_seed = mix(seed ^ 0xf011_a6e5_5eed_0001);
    TreeModel::new(
        origin,
        height,
        crown_radius,
        base_radius,
        grower.segments,
        grower.axes,
        grower.foliage,
        species.foliage.density,
        noise_seed,
    )
}

/// The height a tree of `seed` grows to, without growing it.
pub(crate) fn tree_height(species: &SpeciesSpec, seed: u64) -> f64 {
    drawn_height(species, &mut Hash::new(seed, 0, 0))
}

fn drawn_height(species: &SpeciesSpec, random: &mut Hash) -> f64 {
    random.between(species.height.0, species.height.1)
}

struct Grower<'a> {
    species: &'a SpeciesSpec,
    random: Hash,
    origin: DVec3,
    height: f64,
    crown_radius: f64,
    base_radius: f64,
    segments: Vec<Segment>,
    axes: Vec<Axis>,
    foliage: Vec<FoliageBlob>,
    buds: VecDeque<Bud>,
}

impl Grower<'_> {
    fn habit(&self) -> Habit {
        Habit {
            split_chance: self.species.split_chance,
            split_count: self.species.split_count,
            split_angle: self.species.split_angle,
            tropism: self.species.tropism,
            wobble: self.species.wobble,
            dominance: self.species.dominance,
        }
    }

    fn plant_stems(&mut self, stems: u32) {
        let count = f64::from(stems);
        let radius = (self.base_radius / count.sqrt()).max(TWIG_RADIUS);
        let disc = if stems > 1 {
            1.5 * self.base_radius * count.sqrt()
        } else {
            0.0
        };
        let lean = if stems > 1 {
            self.species.split_angle * (1.0 - self.species.dominance) * 0.5
        } else {
            0.0
        };
        let phase = self.random.unit() * TAU;
        for index in 0..stems {
            let index = f64::from(index);
            let azimuth = index.mul_add(GOLDEN_ANGLE, phase);
            let outward = DVec3::new(azimuth.cos(), 0.0, azimuth.sin());
            let start = self.origin + outward * disc * ((index + 0.5) / count).sqrt();
            let length = if index == 0.0 {
                self.height
            } else {
                self.height * self.random.between(0.8, 1.0)
            };
            let dir = DVec3::Y * lean.cos() + outward * lean.sin();
            // The flare: a short, wider cone the stem rises out of.
            let flare_top = start + dir * FLARE_HEIGHT;
            let axis = u32::try_from(self.axes.len()).unwrap_or(u32::MAX);
            self.segments.push(Segment {
                a: start - DVec3::Y * FLARE_HEIGHT,
                b: flare_top,
                ra: radius * FLARE_WIDENING,
                rb: radius,
                part: Part::Wood,
                axis,
            });
            let flare = u32::try_from(self.segments.len() - 1).unwrap_or(u32::MAX);
            self.axes.push(Axis {
                order: 0,
                part: Part::Wood,
                stem: false,
                parent_segment: None,
                segments: (flare, flare + 1),
                splits: 0,
                children: 0,
            });
            self.buds.push_back(Bud {
                start: flare_top,
                dir,
                length: length - FLARE_HEIGHT,
                radius,
                order: 0,
                part: Part::Wood,
                stem: true,
                splits: true,
                parent_segment: Some(flare),
                crown_from: self.species.crown_base * length,
            });
        }
    }

    fn plant_roots(&mut self) {
        let roots = self.species.roots;
        let reach = roots.spread * self.crown_radius;
        if reach <= 0.0 && roots.depth <= 0.0 {
            return;
        }
        let pitch = roots.depth.atan2(reach);
        let length = reach.hypot(roots.depth);
        let phase = self.random.unit() * TAU;
        for index in 0..ROOT_COUNT {
            let azimuth = f64::from(index).mul_add(GOLDEN_ANGLE, phase);
            let outward = DVec3::new(azimuth.cos(), 0.0, azimuth.sin());
            self.buds.push_back(Bud {
                start: self.origin,
                dir: outward * pitch.cos() - DVec3::Y * pitch.sin(),
                length,
                radius: (self.base_radius * 0.5).max(TWIG_RADIUS),
                order: 1,
                part: Part::Root,
                stem: false,
                splits: true,
                parent_segment: None,
                crown_from: 0.0,
            });
        }
        if roots.depth > 0.5 * reach {
            self.buds.push_back(Bud {
                start: self.origin,
                dir: -DVec3::Y,
                length: roots.depth,
                radius: (self.base_radius * 0.6).max(TWIG_RADIUS),
                order: 1,
                part: Part::Root,
                stem: false,
                splits: true,
                parent_segment: None,
                crown_from: 0.0,
            });
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "internode counts are small and positive"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "an axis grows in one pass: bend, prune, lay a segment, split"
    )]
    fn grow_axis(&mut self, bud: Bud) {
        if bud.length < MIN_AXIS_LENGTH {
            return;
        }
        let habit = if bud.part == Part::Root {
            ROOT_HABIT
        } else {
            self.habit()
        };
        let nodes = (bud.length * INTERNODES / self.height)
            .round()
            .clamp(2.0, INTERNODES) as u32;
        let step = bud.length / f64::from(nodes);
        let axis_index = u32::try_from(self.axes.len()).unwrap_or(u32::MAX);
        let first = u32::try_from(self.segments.len()).unwrap_or(u32::MAX);
        let mut axis = Axis {
            order: bud.order,
            part: bud.part,
            stem: bud.stem,
            parent_segment: bud.parent_segment,
            segments: (first, first),
            splits: 0,
            children: 0,
        };
        let mut position = bud.start;
        let mut dir = bud.dir;
        let mut radius = bud.radius;
        let mut travelled = 0.0;
        // Stems spiral their branches round; laterals branch flat, side to
        // side in their own horizontal plane.
        let flat = bud.order > 0;
        let mut phase = if flat { 0.0 } else { self.random.unit() * TAU };
        let mut last_split = None;
        let can_split = bud.splits && bud.order < MAX_ORDER && radius > TWIG_RADIUS;
        for node in 0..nodes {
            // Leaders hold their line; laterals leave at their split angle,
            // then bend by how thin they are.
            if bud.order > 0 && node > 0 {
                let flex = (1.0 - radius / self.base_radius).clamp(0.0, 1.0).powi(2);
                dir += DVec3::Y * (habit.tropism * TROPISM_RATE * flex);
            }
            dir += self.random.unit_vector() * (habit.wobble * WOBBLE_RATE);
            if bud.part == Part::Root {
                // Roots never climb, and level off at their depth.
                let floor = position.y < self.origin.y - self.species.roots.depth;
                dir.y = if floor { 0.0 } else { dir.y.min(0.0) };
            }
            dir = dir.normalize_or(DVec3::Y);
            let tip_radius = radius + (TWIG_RADIUS - radius) / f64::from(nodes - node);
            let mut end = position + dir * step;
            if bud.order == 0 && !bud.stem && self.outside_envelope(end, bud) {
                // Co-dominant forks are leaders: rather than be pruned, they
                // turn back in along the crown's edge.
                let outward = (end - self.origin).with_y(0.0).normalize_or_zero();
                dir -= outward * (dir.dot(outward).max(0.0) + FORK_RETURN);
                dir = dir.normalize_or(DVec3::Y);
                end = position + dir * step;
            }
            if bud.part == Part::Wood && end.y < self.origin.y + GROUND_CLEARANCE {
                // Hanging wood stops short of the ground.
                break;
            }
            if bud.order > 0 && self.outside_envelope(end, bud) {
                // Pruned at the crown or root envelope: the genome's width and
                // shape bound every branch.
                break;
            }
            self.segments.push(Segment {
                a: position,
                b: end,
                ra: radius,
                rb: tip_radius,
                part: bud.part,
                axis: axis_index,
            });
            position = end;
            radius = tip_radius;
            travelled += step;
            let at_tip = node + 1 == nodes;
            let height_here = position.y - self.origin.y;
            if at_tip
                || !can_split
                || (bud.stem && height_here < bud.crown_from)
                || self.segments.len() >= MAX_SEGMENTS
                || self.random.unit() >= habit.split_chance
            {
                continue;
            }
            let parent_segment = u32::try_from(self.segments.len() - 1).ok();
            let mut count = self
                .random
                .integer(habit.split_count.0, habit.split_count.1);
            if flat {
                // A flat spray has one side shoot to each side.
                count = count.min(2);
            }
            let remaining = bud.length - travelled;
            axis.splits += 1;
            last_split = Some(self.segments.len());
            if self.random.unit() < habit.dominance {
                let child_radius = radius * lerp(0.7, 0.35, habit.dominance);
                let length = if bud.order == 0 && bud.part == Part::Wood {
                    self.lateral_length(height_here, bud.crown_from, habit)
                } else {
                    SUBLATERAL_LENGTH * remaining
                };
                for child in 0..count {
                    let azimuth = f64::from(child).mul_add(TAU / f64::from(count), phase);
                    self.bud(
                        &bud,
                        position,
                        tilt(dir, habit.split_angle, azimuth),
                        length,
                        child_radius,
                        bud.order + 1,
                        parent_segment,
                    );
                }
                axis.children += count;
                phase += if flat { PI } else { GOLDEN_ANGLE };
                let kept = radius.mul_add(radius, -f64::from(count) * child_radius * child_radius);
                radius = kept.max(0.25 * radius * radius).sqrt();
            } else {
                let count = count.max(2);
                let child_radius = radius / f64::from(count).sqrt();
                for child in 0..count {
                    let azimuth = f64::from(child).mul_add(TAU / f64::from(count), phase);
                    self.bud(
                        &bud,
                        position,
                        tilt(dir, (habit.split_angle * 0.5).min(MAX_FORK_ANGLE), azimuth),
                        remaining,
                        child_radius,
                        bud.order,
                        parent_segment,
                    );
                }
                axis.children += count;
                axis.segments.1 = u32::try_from(self.segments.len()).unwrap_or(u32::MAX);
                self.axes.push(axis);
                return;
            }
        }
        axis.segments.1 = u32::try_from(self.segments.len()).unwrap_or(u32::MAX);
        if axis.segments.0 == axis.segments.1 {
            // Pruned before its first segment.
            return;
        }
        if bud.part == Part::Wood {
            self.clothe(&axis, last_split, bud);
        }
        self.axes.push(axis);
    }

    /// Crown radius at `height` above the origin: a blend of a dome and a
    /// cone by dominance, zero above the tree.
    fn envelope(&self, height: f64, crown_from: f64) -> f64 {
        let crown = (self.height - crown_from).max(1.0e-6);
        let t = (height - crown_from) / crown;
        if t > 1.0 {
            return 0.0;
        }
        crown_shape(t.max(0.0), self.species.dominance) * self.crown_radius
    }

    /// Whether a point lies beyond the envelope that bounds `bud`: the crown,
    /// less the foliage that will clothe it, or the root spread.
    fn outside_envelope(&self, point: DVec3, bud: Bud) -> bool {
        let reach = (point - self.origin).with_y(0.0).length();
        let bound = if bud.part == Part::Root {
            self.species.roots.spread * self.crown_radius
        } else {
            let crown = self.envelope(point.y - self.origin.y, bud.crown_from);
            (crown - self.species.foliage.size).max(ENVELOPE_SLACK)
        };
        reach > bound + ENVELOPE_SLACK
    }

    /// Length of a lateral leaving a stem at `height_here`, chosen so its tip
    /// lands on the crown envelope: the laterals together trace the crown.
    fn lateral_length(&self, height_here: f64, crown_from: f64, habit: Habit) -> f64 {
        let (rise, reach) = (habit.split_angle.cos(), habit.split_angle.sin().max(0.3));
        let envelope = |length: f64| {
            let crown = self.envelope(length.mul_add(rise, height_here), crown_from);
            (crown - self.species.foliage.size).max(ENVELOPE_SLACK)
        };
        // The reach grows with length and the envelope narrows above: bisect
        // for the length whose tip meets it.
        let (mut short, mut long) = (0.0, self.crown_radius / reach);
        if long * reach <= envelope(long) {
            return long;
        }
        for _ in 0..24 {
            let middle = 0.5 * (short + long);
            if middle * reach < envelope(middle) {
                short = middle;
            } else {
                long = middle;
            }
        }
        short
    }

    #[expect(clippy::too_many_arguments, reason = "a bud is all of these")]
    fn bud(
        &mut self,
        parent: &Bud,
        start: DVec3,
        dir: DVec3,
        length: f64,
        radius: f64,
        order: u8,
        parent_segment: Option<u32>,
    ) {
        let terminal = radius < TWIG_RADIUS;
        self.buds.push_back(Bud {
            start,
            dir,
            length: if order > 0 {
                length.min(MAX_SLENDERNESS * radius.max(TWIG_RADIUS))
            } else {
                length
            },
            radius: radius.max(TWIG_RADIUS),
            order,
            part: parent.part,
            stem: false,
            splits: !terminal,
            parent_segment,
            crown_from: parent.crown_from,
        });
    }

    /// Foliage on the axis beyond its last split, and a ball at its tip.
    fn clothe(&mut self, axis: &Axis, last_split: Option<usize>, bud: Bud) {
        let size = self.species.foliage.size;
        if size <= 0.0 {
            return;
        }
        let (first, end) = (axis.segments.0 as usize, axis.segments.1 as usize);
        let carries = !axis.stem || self.species.split_chance <= 0.0;
        if carries {
            let from = last_split.unwrap_or(first);
            for index in from..end {
                let segment = self.segments[index];
                if axis.order == 0 && segment.b.y - self.origin.y < bud.crown_from {
                    continue;
                }
                self.foliage.push(FoliageBlob {
                    a: segment.a,
                    b: segment.b,
                    radius: size,
                });
            }
        }
        if let Some(tip) = self.segments[first..end].last()
            && (axis.order > 0 || tip.b.y - self.origin.y >= bud.crown_from)
        {
            self.foliage.push(FoliageBlob {
                a: tip.b,
                b: tip.b,
                radius: size * TIP_BALL,
            });
        }
    }
}

impl Hash {
    fn integer(&mut self, lo: u32, hi: u32) -> u32 {
        let span = f64::from(hi - lo + 1);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "below span, which fits in u32"
        )]
        let offset = (self.unit() * span).floor() as u32;
        lo + offset.min(hi - lo)
    }

    fn unit_vector(&mut self) -> DVec3 {
        let z = self.unit().mul_add(2.0, -1.0);
        let azimuth = self.unit() * TAU;
        let ring = (1.0 - z * z).max(0.0).sqrt();
        DVec3::new(ring * azimuth.cos(), z, ring * azimuth.sin())
    }
}

/// `dir` turned by `angle` toward `azimuth` around itself.
fn tilt(dir: DVec3, angle: f64, azimuth: f64) -> DVec3 {
    let reference = if dir.y.abs() < 0.99 {
        DVec3::Y
    } else {
        DVec3::X
    };
    let across = dir.cross(reference).normalize();
    let other = dir.cross(across);
    let side = across * azimuth.cos() + other * azimuth.sin();
    (dir * angle.cos() + side * angle.sin()).normalize_or(dir)
}

/// Crown radius as a fraction of the widest, at `t` from the crown's base
/// (0) to the tree's top (1): a dome at dominance 0, a cone at 1.
pub(crate) fn crown_shape(t: f64, dominance: f64) -> f64 {
    let cone = 0.95f64.mul_add(-t, 1.0);
    let dome = (PI * 0.85f64.mul_add(t, 0.15)).sin().max(0.0).sqrt();
    lerp(dome, cone, dominance)
}

fn lerp(from: f64, to: f64, t: f64) -> f64 {
    (to - from).mul_add(t, from)
}
