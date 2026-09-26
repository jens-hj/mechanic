//! Standing water on the drainage grid: the sea, and lakes in the hollows the
//! priority-flood fill raises to where they spill. Rivers carry their own
//! levels along their segments.
//!
//! This water is implicit: a level per column and a rule, nothing stored. A
//! point is water where the ground is open, the point lies below the level,
//! and the open space belongs to the surface rather than to a buried void.
//! So that no water stands against air, every shore and bank is raised a
//! margin above its water in a band just outside it, and carve layers keep a
//! sealing roof of rock under lakes, rivers and shores. The sea is the one
//! body whose surface-breaking carves flood: its trenches are part of it.

use std::collections::VecDeque;

use bevy_math::DVec2;

use super::interval::Interval;
use super::rivers::{DRAINAGE_CELL_METRES, Reach, Valley, drainage_side};
use super::spec::WaterDoc;
use crate::WORLD_HALF_EXTENT_METERS;

/// Rock every carve layer keeps above its voids under lakes, rivers and
/// shores, in metres of biome density.
pub(crate) const SEALED_ROOF_METRES: f64 = 4.0;

/// Drainage cells a sea or lake reaches past its own points, over ground at
/// or above its level. Within that reach the terrain draws the shore; only
/// beyond it does a shore have to be raised.
const SPREAD_CELLS: usize = 2;

/// How far past its channel a river's water reaches over its banks, in metres.
const BANK_SPREAD_METRES: f64 = 8.0;

/// Fill depth above which a drainage point belongs to a hollow, in metres.
const HOLLOW_METRES: f64 = 0.05;

/// A grid body is wet where its blend weight reaches one half. Its shore
/// stands at full height over the weights from this up to one half.
const CREST_WEIGHT: f64 = 0.4;

/// Weights beyond which a grid body's shore raises nothing: far outside it,
/// and deep inside it.
const OUTER_WEIGHT: f64 = 0.1;
const INNER_WEIGHT: f64 = 0.8;

/// Width of a river bank's crest beyond the channel, in metres.
const BANK_CREST_METRES: f64 = 2.0;

/// How far past its crest a bank still raises ground, in metres.
const BANK_REACH_METRES: f64 = 12.0;

/// Farthest a bank raises ground beyond its channel, in metres.
pub(crate) const BANK_TOTAL_METRES: f64 =
    BANK_SPREAD_METRES + BANK_CREST_METRES + BANK_REACH_METRES;

/// Slope under a river's own water: steep, so a carved channel stays as
/// carved and only the lip at the bank is raised.
const CHANNEL_SLOPE: f64 = 4.0;

/// Body number of the sea on the drainage grid; lakes follow it.
const SEA: u32 = 1;

/// Rain on land at neutral humidity, in millimetres a year. Game-wet, so a
/// small stream still carries enough to fill a trench in minutes. Every
/// river and lake is fed by it; the sea gives it back by evaporation.
const RAIN_MM_PER_YEAR: f64 = 4_000.0;

/// Width a lake spills over where no river leaves it, in metres.
const SPILL_WIDTH_METRES: f64 = 4.0;

/// Rain falling on one drainage cell at a humidity in `[-1, 1]`, in m³/s.
pub(crate) fn cell_rain(humidity: f64) -> f64 {
    const SECONDS_PER_YEAR: f64 = 365.25 * 86_400.0;
    let rate = RAIN_MM_PER_YEAR / 1_000.0 / SECONDS_PER_YEAR * (1.0 + humidity).clamp(0.1, 2.0);
    rate * DRAINAGE_CELL_METRES * DRAINAGE_CELL_METRES
}

/// Where the water leaving a river reach or a lake goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outflow {
    /// Into the next river reach, by number.
    River(u32),
    /// Into a lake, by number.
    Lake(u32),
    /// Into the sea, or off the world's edge into the ocean beyond it.
    Sea,
}

/// One river reach as the water cycle sees it: what it carries in the
/// untouched world and where that goes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiverReach {
    /// Water it carries in the untouched world, in m³/s: the rain on its
    /// catchment.
    pub discharge_m3_s: f64,
    /// Time its water takes to run its length, in seconds.
    pub travel_seconds: f64,
    /// Depth of its channel, in metres: how far its surface falls when it
    /// runs dry.
    pub depth: f64,
    /// Where its water goes.
    pub outflow: Outflow,
}

/// One lake as the water cycle sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LakeBasin {
    /// Surface area of its hollow, in square metres.
    pub area_m2: f64,
    /// Water it holds at its seed level, in m³.
    pub volume_m3: f64,
    /// Water spilling from it in the untouched world, in m³/s: as much as
    /// its rivers and rain bring in.
    pub discharge_m3_s: f64,
    /// Width of the rim it spills over, in metres.
    pub spill_width: f64,
    /// Where the water it spills goes.
    pub outflow: Outflow,
}

/// Which water a column holds.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum WaterBody {
    /// The sea, at the world's sea level.
    Sea,
    /// A lake, numbered in the order the world traced it.
    Lake(u32),
    /// A river channel, by the number of its reach.
    River(u32),
    /// Water stored in the world rather than derived from the seed: a pool,
    /// by its number in the world's water.
    Pool(u32),
    /// Water running over the ground, stored in the world.
    Running,
}

/// The water over one column.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSurface {
    /// Height of the surface, in metres.
    pub level: f64,
    /// Which water it is.
    pub body: WaterBody,
    /// Horizontal surface current along x and z, in m/s.
    pub flow: DVec2,
}

/// What the water does to one column of ground.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ColumnWater {
    /// The water over the column, where it lies in a body of water.
    pub(crate) surface: Option<WaterSurface>,
    /// Ground is raised to at least this height: the shore seal.
    pub(crate) floor: f64,
    /// Below this height carve layers keep [`SEALED_ROOF_METRES`] of rock,
    /// so no void opens under the water or beside it below its level.
    pub(crate) seal_below: f64,
}

impl ColumnWater {
    pub(crate) const DRY: Self = Self {
        surface: None,
        floor: f64::NEG_INFINITY,
        seal_below: f64::NEG_INFINITY,
    };
}

/// The sea and lakes on the drainage grid.
#[derive(Debug)]
pub(crate) struct WaterBodies {
    /// Body at each drainage point: 0 dry, [`SEA`], then one per lake. A
    /// body covers its own points and the higher ground it spreads over.
    body: Vec<u32>,
    /// Level of each body, by the same numbering.
    levels: Vec<f64>,
    /// Each lake as the water cycle sees it, by lake number.
    lakes: Vec<LakeBasin>,
    /// Each river reach as the water cycle sees it, by number.
    reaches: Vec<RiverReach>,
    /// The lake a reach runs inside of, whose water it is.
    reach_lakes: Vec<Option<u32>>,
    /// Surface area of the sea, in square metres.
    sea_area: f64,
    margin: f64,
    slope: f64,
}

impl WaterBodies {
    /// No water, used before tracing.
    pub(crate) fn none() -> Self {
        Self {
            body: Vec::new(),
            levels: Vec::new(),
            lakes: Vec::new(),
            reaches: Vec::new(),
            reach_lakes: Vec::new(),
            sea_area: 0.0,
            margin: 0.0,
            slope: 1.0,
        }
    }

    /// Finds the sea and lakes from blended heights on the drainage grid,
    /// the heights after priority-flood filling, and the sea points, and
    /// joins them to the river `reaches` into one drainage network carrying
    /// the water that drains through each point.
    pub(crate) fn trace(
        doc: &WaterDoc,
        sea_level: f64,
        [heights, filled, discharge]: [&[f64]; 3],
        sea: &[bool],
        reaches: &[Reach],
    ) -> Self {
        let mut body = sea
            .iter()
            .map(|&sea| if sea { SEA } else { 0 })
            .collect::<Vec<_>>();
        let mut levels = vec![f64::NAN, sea_level];
        let mut lakes = Vec::new();
        let cell_area = DRAINAGE_CELL_METRES * DRAINAGE_CELL_METRES;
        let hollow = |index: usize| !sea[index] && filled[index] - heights[index] > HOLLOW_METRES;
        let mut seen = vec![false; heights.len()];
        let mut members = Vec::new();
        for start in 0..heights.len() {
            if seen[start] || !hollow(start) {
                continue;
            }
            members.clear();
            flood(start, &mut seen, &mut members, hollow);
            let level = members
                .iter()
                .map(|&index| filled[index])
                .fold(f64::INFINITY, f64::min);
            let depth = members
                .iter()
                .map(|&index| level - heights[index])
                .fold(f64::NEG_INFINITY, f64::max);
            if depth < doc.lake_depth {
                continue;
            }
            let id = u32::try_from(levels.len()).expect("lake count fits u32");
            levels.push(level);
            #[expect(
                clippy::cast_precision_loss,
                reason = "a lake spans few drainage cells"
            )]
            let area_m2 = members.len() as f64 * cell_area;
            lakes.push(LakeBasin {
                area_m2,
                volume_m3: members
                    .iter()
                    .map(|&index| (level - heights[index]).max(0.0) * cell_area)
                    .sum(),
                discharge_m3_s: members
                    .iter()
                    .map(|&index| discharge[index])
                    .fold(0.0, f64::max),
                spill_width: SPILL_WIDTH_METRES,
                outflow: Outflow::Sea,
            });
            for &index in &members {
                body[index] = id;
            }
        }
        spread(&mut body, &levels, heights);
        let (reaches, reach_lakes) = link_reaches(&body, sea, &mut lakes, reaches);
        #[expect(clippy::cast_precision_loss, reason = "a few hundred thousand points")]
        let sea_area = sea.iter().filter(|&&sea| sea).count() as f64 * cell_area;
        Self {
            body,
            levels,
            lakes,
            reaches,
            reach_lakes,
            sea_area,
            margin: doc.shore_margin,
            slope: doc.shore_slope,
        }
    }

    /// Number of lakes.
    pub(crate) fn lake_count(&self) -> usize {
        self.levels.len().saturating_sub(2)
    }

    /// A lake as the water cycle sees it.
    pub(crate) fn lake(&self, lake: u32) -> Option<LakeBasin> {
        self.lakes.get(lake as usize).copied()
    }

    /// A river reach as the water cycle sees it.
    pub(crate) fn reach(&self, reach: u32) -> Option<RiverReach> {
        self.reaches.get(reach as usize).copied()
    }

    /// Surface area of the sea, in square metres.
    pub(crate) const fn sea_area(&self) -> f64 {
        self.sea_area
    }

    /// Each body's bilinear weight at a column, over the four drainage
    /// points around it, and how many bodies there are.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        reason = "clamped to the drainage grid"
    )]
    fn weights(&self, x: f64, z: f64) -> ([(u32, f64); 4], usize) {
        let mut grid = [(0_u32, 0.0_f64); 4];
        let mut bodies = 0;
        if !self.body.is_empty() {
            let side = drainage_side();
            let last = (side - 1) as f64;
            let gx = ((x + WORLD_HALF_EXTENT_METERS) / DRAINAGE_CELL_METRES).clamp(0.0, last);
            let gz = ((z + WORLD_HALF_EXTENT_METERS) / DRAINAGE_CELL_METRES).clamp(0.0, last);
            let (i, k) = ((gx as usize).min(side - 2), (gz as usize).min(side - 2));
            let (fx, fz) = (gx - i as f64, gz - k as f64);
            for (index, weight) in [
                (i + k * side, (1.0 - fx) * (1.0 - fz)),
                (i + 1 + k * side, fx * (1.0 - fz)),
                (i + (k + 1) * side, (1.0 - fx) * fz),
                (i + 1 + (k + 1) * side, fx * fz),
            ] {
                let id = self.body[index];
                if id == 0 {
                    continue;
                }
                if let Some(entry) = grid[..bodies].iter_mut().find(|entry| entry.0 == id) {
                    entry.1 += weight;
                } else {
                    grid[bodies] = (id, weight);
                    bodies += 1;
                }
            }
        }
        (grid, bodies)
    }

    /// The water over a column and what it does to the ground there. `river`
    /// is the column's valley where a river may run.
    pub(crate) fn column(&self, x: f64, z: f64, river: Option<Valley>) -> ColumnWater {
        let (grid, bodies) = self.weights(x, z);
        let grid = &grid[..bodies];
        let river_wet =
            river.filter(|valley| valley.channel_distance < valley.half_width + BANK_SPREAD_METRES);

        let mut surface: Option<WaterSurface> = None;
        let mut offer = |candidate: WaterSurface| {
            if surface.is_none_or(|best| candidate.level > best.level) {
                surface = Some(candidate);
            }
        };
        for &(id, weight) in grid {
            if weight >= 0.5 {
                offer(WaterSurface {
                    level: self.levels[id as usize],
                    body: if id == SEA {
                        WaterBody::Sea
                    } else {
                        WaterBody::Lake(id - SEA - 1)
                    },
                    flow: DVec2::ZERO,
                });
            }
        }
        if let Some(valley) = river_wet {
            // A reach running through a lake is that lake's water.
            let lake = self
                .reach_lakes
                .get(valley.segment as usize)
                .copied()
                .flatten();
            offer(WaterSurface {
                level: valley.level,
                body: lake.map_or(WaterBody::River(valley.segment), WaterBody::Lake),
                flow: DVec2::from_array(valley.flow),
            });
        }

        let mut floor = f64::NEG_INFINITY;
        let mut seal_below = surface
            .filter(|surface| surface.body != WaterBody::Sea)
            .map_or(f64::NEG_INFINITY, |surface| surface.level + self.margin);
        for &(id, weight) in grid {
            let wet_elsewhere = river_wet.is_some()
                || grid
                    .iter()
                    .any(|&(other, other_weight)| other != id && other_weight >= 0.5);
            if wet_elsewhere || !(OUTER_WEIGHT..=INNER_WEIGHT).contains(&weight) {
                continue;
            }
            let distance = if weight < CREST_WEIGHT {
                (CREST_WEIGHT - weight) * DRAINAGE_CELL_METRES
            } else if weight < 0.5 {
                0.0
            } else {
                (weight - 0.5) * DRAINAGE_CELL_METRES
            };
            let crest = self.levels[id as usize] + self.margin;
            if weight < 0.5 {
                seal_below = seal_below.max(crest);
            }
            floor = floor.max(crest - distance * self.slope);
        }
        // A river's banks stand wherever no lake or sea at its level holds
        // its water: a lake far below a river crossing its shore does not.
        if let Some(valley) = river
            && !grid.iter().any(|&(id, weight)| {
                weight >= 0.5 && self.levels[id as usize] >= valley.level - self.margin
            })
        {
            let beyond = valley.channel_distance - valley.half_width - BANK_SPREAD_METRES;
            let drop = if beyond < 0.0 {
                -beyond * CHANNEL_SLOPE
            } else if beyond < BANK_CREST_METRES {
                0.0
            } else {
                (beyond - BANK_CREST_METRES) * self.slope
            };
            if beyond < BANK_CREST_METRES + BANK_REACH_METRES {
                seal_below = seal_below.max(valley.level + self.margin);
                floor = floor.max(valley.level + self.margin - drop);
            }
        }
        ColumnWater {
            surface,
            floor,
            seal_below,
        }
    }

    /// Lowest and highest level of any sea or lake whose water or shore may
    /// reach the columns of a box, if one may.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        reason = "clamped to the drainage grid"
    )]
    pub(crate) fn level_range(&self, x: Interval, z: Interval) -> Option<(f64, f64)> {
        if self.body.is_empty() {
            return None;
        }
        let side = drainage_side();
        let last = (side - 1) as f64;
        let index_range = |range: Interval| {
            let lo = ((range.lo + WORLD_HALF_EXTENT_METERS) / DRAINAGE_CELL_METRES).floor() - 1.0;
            let hi = ((range.hi + WORLD_HALF_EXTENT_METERS) / DRAINAGE_CELL_METRES).ceil() + 1.0;
            (lo.clamp(0.0, last) as usize, hi.clamp(0.0, last) as usize)
        };
        let (x0, x1) = index_range(x);
        let (z0, z1) = index_range(z);
        let mut range: Option<(f64, f64)> = None;
        for k in z0..=z1 {
            for i in x0..=x1 {
                let id = self.body[i + k * side];
                if id != 0 {
                    let level = self.levels[id as usize];
                    range = Some(
                        range.map_or((level, level), |(lo, hi)| (lo.min(level), hi.max(level))),
                    );
                }
            }
        }
        range
    }

    /// Highest ground any shore may raise within a box, if one may.
    pub(crate) fn floor_bound(&self, x: Interval, z: Interval) -> Option<f64> {
        self.level_range(x, z).map(|(_, hi)| hi + self.margin)
    }

    /// Height every shore keeps above its water.
    pub(crate) const fn margin(&self) -> f64 {
        self.margin
    }
}

/// Joins river reaches to the sea and lakes on the drainage grid: where each
/// reach's water goes, which lake each reach runs inside of, and which
/// reach each lake spills into.
fn link_reaches(
    body: &[u32],
    sea: &[bool],
    lakes: &mut [LakeBasin],
    reaches: &[Reach],
) -> (Vec<RiverReach>, Vec<Option<u32>>) {
    let lake_at = |index: usize| body[index].checked_sub(SEA + 1);
    let starting = reaches
        .iter()
        .enumerate()
        .map(|(number, reach)| (reach.points[0], number))
        .collect::<std::collections::HashMap<_, _>>();
    let mut reach_lakes = Vec::with_capacity(reaches.len());
    let reaches = reaches
        .iter()
        .enumerate()
        .map(|(number, reach)| {
            let [from, to] = reach.points;
            let (from_lake, to_lake) = (lake_at(from), lake_at(to));
            reach_lakes.push(from_lake.filter(|&lake| to_lake == Some(lake)));
            let outflow = if sea[to] || body[to] == SEA {
                Outflow::Sea
            } else if let Some(lake) = to_lake.filter(|&lake| from_lake != Some(lake)) {
                Outflow::Lake(lake)
            } else if let Some(&next) = starting.get(&to) {
                Outflow::River(u32::try_from(next).expect("reach count fits u32"))
            } else {
                Outflow::Sea
            };
            // A river leaving a lake is where the lake spills.
            if let Some(lake) = from_lake.filter(|&lake| to_lake != Some(lake))
                && let Some(basin) = lakes.get_mut(lake as usize)
                && basin.outflow == Outflow::Sea
            {
                basin.outflow = Outflow::River(u32::try_from(number).expect("fits u32"));
                basin.spill_width = 2.0 * reach.half_width;
                basin.discharge_m3_s = basin.discharge_m3_s.max(reach.discharge);
            }
            RiverReach {
                discharge_m3_s: reach.discharge,
                travel_seconds: reach.length / reach.speed.max(0.1),
                depth: reach.depth,
                outflow,
            }
        })
        .collect::<Vec<_>>();
    (reaches, reach_lakes)
}

/// Spreads every body over the dry drainage points around it whose ground
/// stands at or above its level, [`SPREAD_CELLS`] rings deep.
fn spread(body: &mut [u32], levels: &[f64], heights: &[f64]) {
    let side = drainage_side();
    let mut frontier = (0..body.len())
        .filter(|&index| body[index] != 0)
        .collect::<Vec<_>>();
    for _ in 0..SPREAD_CELLS {
        let mut next = Vec::new();
        for &index in &frontier {
            let id = body[index];
            let (column, row) = (index % side, index / side);
            for z in row.saturating_sub(1)..=(row + 1).min(side - 1) {
                for x in column.saturating_sub(1)..=(column + 1).min(side - 1) {
                    let neighbour = x + z * side;
                    if body[neighbour] == 0 && heights[neighbour] >= levels[id as usize] {
                        body[neighbour] = id;
                        next.push(neighbour);
                    }
                }
            }
        }
        frontier = next;
    }
}

/// Drainage points of the sea: connected ground below sea level that covers
/// at least `sea_area` square kilometres.
pub(crate) fn sea_points(heights: &[f64], sea_level: f64, sea_area: f64) -> Vec<bool> {
    let cell_area_km2 = DRAINAGE_CELL_METRES * DRAINAGE_CELL_METRES / 1.0e6;
    let below = |index: usize| heights[index] < sea_level;
    let mut sea = vec![false; heights.len()];
    let mut seen = vec![false; heights.len()];
    let mut members = Vec::new();
    for start in 0..heights.len() {
        if seen[start] || !below(start) {
            continue;
        }
        members.clear();
        flood(start, &mut seen, &mut members, below);
        #[expect(clippy::cast_precision_loss, reason = "a few hundred thousand points")]
        let area = members.len() as f64 * cell_area_km2;
        if area >= sea_area {
            for &index in &members {
                sea[index] = true;
            }
        }
    }
    sea
}

/// Collects the 8-connected drainage points from `start` that `inside` holds.
fn flood(
    start: usize,
    seen: &mut [bool],
    members: &mut Vec<usize>,
    inside: impl Fn(usize) -> bool,
) {
    let side = drainage_side();
    let mut queue = VecDeque::from([start]);
    seen[start] = true;
    while let Some(index) = queue.pop_front() {
        members.push(index);
        let (column, row) = (index % side, index / side);
        for z in row.saturating_sub(1)..=(row + 1).min(side - 1) {
            for x in column.saturating_sub(1)..=(column + 1).min(side - 1) {
                let neighbour = x + z * side;
                if !seen[neighbour] && inside(neighbour) {
                    seen[neighbour] = true;
                    queue.push_back(neighbour);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{WaterBodies, WaterBody, sea_points};
    use crate::WORLD_HALF_EXTENT_METERS;
    use crate::generation::rivers::{DRAINAGE_CELL_METRES, drainage_side};
    use crate::generation::spec::WaterDoc;

    fn doc() -> WaterDoc {
        WaterDoc {
            sea_area: 2.0,
            lake_depth: 0.6,
            shore_margin: 0.2,
            shore_slope: 0.6,
        }
    }

    fn flow(heights: &[f64]) -> Vec<f64> {
        vec![0.0; heights.len()]
    }

    #[expect(clippy::cast_precision_loss, reason = "small grid")]
    fn position(index: usize) -> f64 {
        index as f64 * DRAINAGE_CELL_METRES - WORLD_HALF_EXTENT_METERS
    }

    /// Flat ground at 10 m with one square hollow 4 m deep, filled to 10 m.
    fn hollow_world() -> (Vec<f64>, Vec<f64>, (usize, usize)) {
        let side = drainage_side();
        let centre = (side / 2, side / 2);
        let heights = (0..side * side)
            .map(|index| {
                let (x, z) = (index % side, index / side);
                if x.abs_diff(centre.0) <= 2 && z.abs_diff(centre.1) <= 2 {
                    6.0
                } else {
                    10.0
                }
            })
            .collect::<Vec<_>>();
        let filled = heights
            .iter()
            .map(|height: &f64| height.max(10.0))
            .collect();
        (heights, filled, centre)
    }

    #[test]
    fn small_ground_below_sea_level_is_not_sea() {
        let side = drainage_side();
        let heights = (0..side * side)
            .map(|index| if index == side * side / 2 { -5.0 } else { 3.0 })
            .collect::<Vec<_>>();
        assert!(!sea_points(&heights, 0.0, 2.0).iter().any(|&sea| sea));
        let low = vec![-5.0; side * side];
        assert!(sea_points(&low, 0.0, 2.0).iter().all(|&sea| sea));
    }

    #[test]
    fn a_filled_hollow_is_a_lake_at_its_spill_level() {
        let (heights, filled, centre) = hollow_world();
        let sea = vec![false; heights.len()];
        let bodies =
            WaterBodies::trace(&doc(), 0.0, [&heights, &filled, &flow(&heights)], &sea, &[]);
        assert_eq!(bodies.lake_count(), 1);
        let water = bodies.column(position(centre.0), position(centre.1), None);
        let surface = water.surface.expect("the hollow's centre is wet");
        assert_eq!(surface.body, WaterBody::Lake(0));
        assert!((surface.level - 10.0).abs() < 1.0e-9);
        assert!(water.seal_below > 10.0, "carves stay sealed under a lake");
    }

    #[test]
    fn a_lake_reaches_over_higher_ground_and_raises_a_shore_only_past_that() {
        let (heights, filled, centre) = hollow_world();
        let sea = vec![false; heights.len()];
        let bodies =
            WaterBodies::trace(&doc(), 0.0, [&heights, &filled, &flow(&heights)], &sea, &[]);
        let (x, z) = (position(centre.0), position(centre.1));
        // Along a row out of the lake: the hollow and the two rings of
        // higher ground around it are wet, with nothing raised; past them a
        // crest stands above the water, and far away the ground is untouched.
        let spread = bodies.column(x + 4.0 * DRAINAGE_CELL_METRES, z, None);
        assert!(
            spread
                .surface
                .is_some_and(|surface| (surface.level - 10.0).abs() < 1.0e-9)
        );
        assert!(spread.floor < 10.0);
        let edge = x + 4.5 * DRAINAGE_CELL_METRES;
        let crest = bodies.column(edge + 0.05 * DRAINAGE_CELL_METRES, z, None);
        assert!(crest.surface.is_none());
        assert!((crest.floor - 10.2).abs() < 1.0e-9);
        let far = bodies.column(x + 10.0 * DRAINAGE_CELL_METRES, z, None);
        assert!(far.surface.is_none() && far.floor.is_infinite() && far.seal_below.is_infinite());
    }

    #[test]
    fn a_lake_does_not_reach_over_ground_below_its_level() {
        let (mut heights, mut filled, centre) = hollow_world();
        let side = drainage_side();
        // Ground falls away east of the hollow: the lake's outlet.
        for index in 0..heights.len() {
            if index % side > centre.0 + 2 {
                heights[index] = 8.0;
                filled[index] = 8.0;
            }
        }
        let sea = vec![false; heights.len()];
        let bodies =
            WaterBodies::trace(&doc(), 0.0, [&heights, &filled, &flow(&heights)], &sea, &[]);
        let (x, z) = (position(centre.0), position(centre.1));
        let outside = bodies.column(x + 3.5 * DRAINAGE_CELL_METRES, z, None);
        assert!(outside.surface.is_none());
        let west = bodies.column(x - 4.0 * DRAINAGE_CELL_METRES, z, None);
        assert!(west.surface.is_some());
    }
}
