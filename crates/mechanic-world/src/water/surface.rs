//! One water surface over all stored water: running sheets, the open tops of
//! pools, and cells joined to seed-derived water beyond that water's own
//! sheet, meshed together in square tiles as a heightfield.
//!
//! Every column of visible water is a quad, and each quad corner stands at
//! the depth-weighted mean level of the columns around it that belong to the
//! same water: within a chute's drop of each other. Neighbouring columns
//! share their corners, so running water down a slope is one smooth ramp,
//! a pool meets the stream feeding it on one edge, and no gaps open between
//! columns at different heights. A corner beside dry ground takes no depth,
//! and a ring of empty edge columns carries the surface one column on, so
//! the water fades out across it along its depth rather than stopping at its
//! columns' edges in steps.

use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

use bevy_math::{DVec2, DVec3};

use super::cells::{CellHasher, CellMap};
use super::{WATER_CELL_METRES, WaterCell, WaterGround, WaterShift, WaterWorld};
use crate::WaterBody;

/// Columns along one edge of a surface tile: 32 water cells, 6.4 m.
pub const SURFACE_TILE_COLUMNS: i32 = 32;

/// Levels further apart than this belong to different water: a fall, not a
/// ramp, lies between them.
const JOINS_METRES: f64 = 1.0;

/// Shallowest running water drawn as water, in metres.
const VISIBLE_METRES: f64 = 0.003;

/// Shallowest water that weighs in on a corner's level, in metres.
const LEAST_WEIGHT_METRES: f64 = 0.002;

/// One column of visible water.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Column {
    level: f64,
    depth: f64,
    flow: DVec2,
    /// Seed-derived water beside stored water, drawn with it at the seed
    /// water's level, depth and current: the seed water's own sheet leaves
    /// the column out, so the two surfaces meet in the calm water one column
    /// out, where they agree.
    anchor: bool,
    /// Past the water's edge: an empty column around visible water, drawn
    /// so the water fades out across it along its depth rather than
    /// stopping at a column's edge.
    edge: bool,
}

/// Least depth an anchor weighs in with on a corner, in metres: as deep
/// water, so stored water meets a lake at the lake's level.
const ANCHOR_METRES: f64 = 1.0;

/// Water shallower than this, in metres, lies draped over the drawn ground:
/// each corner stands at the ground there plus the water's depth.
const DRAPED_METRES: f64 = 0.02;

/// How far over a corner's water the search for the ground under it starts,
/// in metres: water sunk into the ground lies at most this deep in it.
const TOP_ABOVE_METRES: f64 = 0.25;

/// How far down the search for the ground under a corner looks, in metres.
const TOP_REACH_METRES: f64 = 1.5;

/// Water deeper than this, in metres, lies level: its corners stand at the
/// mean level of the water around them. Between the two it blends.
const LEVEL_METRES: f64 = 0.1;

/// The stored water's surface, and the columns of seed-derived water's
/// reach it draws, which that water's own sheet leaves out.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StoredSurface {
    /// The surface, tile by tile.
    pub tiles: Vec<SurfaceTile>,
    /// Columns drawn here that lie in a lake's or river's reach.
    pub owned: Vec<(i32, i32)>,
    /// Where each moved lake and river is drawn, here and by its own sheet.
    pub shifts: BTreeMap<WaterBody, WaterShift>,
}

/// One tile of the stored water's surface, placed at `origin`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SurfaceTile {
    /// The tile, in tile coordinates.
    pub key: (i32, i32),
    /// What the tile shows, to the millimetre: a tile is meshed again only
    /// when this changes.
    pub fingerprint: u64,
    /// Where vertices are placed from: the tile's lower corner at height
    /// zero.
    pub origin: DVec3,
    /// Vertex positions relative to `origin`.
    pub positions: Vec<[f32; 3]>,
    /// Upward vertex normals of the surface.
    pub normals: Vec<[f32; 3]>,
    /// Depth of water under each vertex, then its current along x and z.
    pub attributes: Vec<[f32; 3]>,
    /// Upward-facing triangles.
    pub indices: Vec<u32>,
}

impl WaterWorld {
    /// The surface of the stored water, tile by tile, and the columns of
    /// seed-derived water's reach it draws. Tiles whose fingerprint matches
    /// `drawn` come back without a mesh: they have not changed.
    pub fn surface_tiles(
        &mut self,
        ground: &impl WaterGround,
        drawn: &HashMap<(i32, i32), u64>,
    ) -> StoredSurface {
        self.show_shifts(ground);
        let mut columns = self.visible_columns(ground);
        self.sound_anchors(ground, &mut columns);
        let reaches = &mut self.reaches;
        let mut tiles_reached = CellMap::<(i32, i32), bool>::default();
        let owned = columns
            .keys()
            .copied()
            // Any of the column the lake's sheet reaches, which ends along a
            // line through columns, not between them.
            .filter(|&(x, z)| {
                let edge = WATER_CELL_METRES;
                let tile = (
                    x.div_euclid(SURFACE_TILE_COLUMNS),
                    z.div_euclid(SURFACE_TILE_COLUMNS),
                );
                let reachable = *tiles_reached.entry(tile).or_insert_with(|| {
                    let span = f64::from(SURFACE_TILE_COLUMNS) * edge;
                    let low = [f64::from(tile.0) * span, f64::from(tile.1) * span];
                    ground.may_reach(low, [low[0] + span, low[1] + span])
                });
                reachable
                    && *reaches.entry((x, z)).or_insert_with(|| {
                        let (low, high) = (f64::from(x) * edge, f64::from(z) * edge);
                        [
                            (0.5_f64, 0.5_f64),
                            (0.0, 0.0),
                            (1.0, 0.0),
                            (0.0, 1.0),
                            (1.0, 1.0),
                        ]
                        .iter()
                        .any(|&(dx, dz)| {
                            ground
                                .surface(dx.mul_add(edge, low), dz.mul_add(edge, high))
                                .is_some()
                        })
                    })
            })
            .collect();
        let mut tiles = CellMap::<(i32, i32), Vec<(i32, i32)>>::default();
        for &(x, z) in columns.keys() {
            tiles
                .entry((
                    x.div_euclid(SURFACE_TILE_COLUMNS),
                    z.div_euclid(SURFACE_TILE_COLUMNS),
                ))
                .or_default()
                .push((x, z));
        }
        let mut out = Vec::with_capacity(tiles.len());
        for (key, mut members) in tiles {
            members.sort_unstable();
            let fingerprint = fingerprint(&columns, &members);
            if drawn.get(&key) == Some(&fingerprint) {
                out.push(SurfaceTile {
                    key,
                    fingerprint,
                    ..SurfaceTile::default()
                });
                continue;
            }
            let tops = &mut self.tops;
            let mut top = |x: i32, z: i32, near: f64| {
                let y = (near / WATER_CELL_METRES).floor();
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "heights are far inside i32"
                )]
                let key = (x, y as i32, z);
                *tops.entry(key).or_insert_with(|| {
                    let edge = WATER_CELL_METRES;
                    ground.ground_top(
                        f64::from(x) * edge,
                        f64::from(z) * edge,
                        near + TOP_ABOVE_METRES,
                        TOP_REACH_METRES,
                    )
                })
            };
            out.push(mesh_tile(&columns, key, &members, fingerprint, &mut top));
        }
        out.sort_unstable_by_key(|tile| tile.key);
        StoredSurface {
            tiles: out,
            owned,
            shifts: self.shown_shifts().clone(),
        }
    }

    /// Gives each anchor the depth of the seed-derived water there, as that
    /// water's own sheet measures it, so the two surfaces show one colour
    /// where they meet. Depths are kept per column until the ground there is
    /// edited.
    fn sound_anchors(
        &mut self,
        ground: &impl WaterGround,
        columns: &mut CellMap<(i32, i32), Column>,
    ) {
        for (&(x, z), column) in columns.iter_mut().filter(|(_, column)| column.anchor) {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "heights are far inside i32"
            )]
            let key = (x, (column.level / WATER_CELL_METRES).floor() as i32, z);
            let floor = *self.floors.entry(key).or_insert_with(|| {
                // From over the water, as a column at a bank's edge may
                // stand in the bank at its own level.
                let centre = WaterCell::new(x, 0, z).centre();
                ground.ground_top(
                    centre.x,
                    centre.z,
                    column.level + TOP_ABOVE_METRES,
                    TOP_ABOVE_METRES + crate::mesh::DEEPEST_METRES,
                )
            });
            column.depth = floor.map_or(crate::mesh::DEEPEST_METRES, |floor| {
                (column.level - floor).clamp(0.0, crate::mesh::DEEPEST_METRES)
            });
        }
    }

    /// Every column of visible stored water, at its top.
    fn visible_columns(&self, ground: &impl WaterGround) -> CellMap<(i32, i32), Column> {
        let mut columns = CellMap::<(i32, i32), Column>::default();
        let mut offer = |x: i32, z: i32, column: Column| {
            let entry = columns.entry((x, z)).or_insert(column);
            if column.level > entry.level {
                *entry = column;
            }
        };
        for view in self.running_cells() {
            // A film shows as wet ground, not as water.
            if view.depth < VISIBLE_METRES {
                continue;
            }
            offer(
                view.cell.x,
                view.cell.z,
                Column {
                    level: view.level,
                    depth: view.depth,
                    flow: view.flow,
                    anchor: false,
                    edge: false,
                },
            );
        }
        for pool in self.pools() {
            for (&cell, &depth) in pool.surface_cells.iter().zip(&pool.depths) {
                // A pool sealed higher than its cells shows at their top, and
                // one under seed-derived water has no surface of its own.
                let level = pool.level.min(cell.bottom() + WATER_CELL_METRES);
                let centre = cell.centre();
                if ground
                    .implicit(DVec3::new(centre.x, level + 0.05, centre.z))
                    .is_some()
                {
                    continue;
                }
                offer(
                    cell.x,
                    cell.z,
                    Column {
                        level,
                        depth,
                        flow: DVec2::ZERO,
                        anchor: false,
                        edge: false,
                    },
                );
            }
        }
        // Cells joined to seed-derived water where its own sheet, which
        // draws the joined cells within its reach, does not run.
        let mut joined = CellMap::<(i32, i32), (WaterCell, f64, f64)>::default();
        for (cell, surface) in self.joined_cells() {
            if cell.bottom() >= surface.level {
                continue;
            }
            let top =
                joined
                    .entry((cell.x, cell.z))
                    .or_insert((cell, surface.level, cell.bottom()));
            if cell.y > top.0.y {
                top.0 = cell;
            }
            top.2 = top.2.min(cell.bottom());
        }
        for ((x, z), (cell, level, bottom)) in joined {
            let centre = cell.centre();
            if ground.surface(centre.x, centre.z).is_none() {
                offer(
                    x,
                    z,
                    Column {
                        level,
                        depth: level - bottom,
                        flow: DVec2::ZERO,
                        anchor: false,
                        edge: false,
                    },
                );
            }
        }
        self.anchor_to_seed_water(ground, &mut columns);
        add_edges(&mut columns);
        columns
    }

    /// Adds seed-derived water beside the stored water as anchors, so the
    /// stored water's edge meets it at its own level.
    fn anchor_to_seed_water(
        &self,
        ground: &impl WaterGround,
        columns: &mut CellMap<(i32, i32), Column>,
    ) {
        let mut anchors = CellMap::<(i32, i32), Column>::default();
        for (&(x, z), column) in columns.iter() {
            for dz in -1..=1 {
                for dx in -1..=1 {
                    let key = (x + dx, z + dz);
                    if columns.contains_key(&key) || anchors.contains_key(&key) {
                        continue;
                    }
                    let centre = WaterCell::new(key.0, 0, key.1).centre();
                    let Some(seed) = ground.surface(centre.x, centre.z) else {
                        continue;
                    };
                    let drawn = super::cycle::shifted(self.shown_shifts(), seed);
                    let level = drawn.level;
                    // Only where the seed-derived water shows: its sheet runs
                    // on under the bank, and an edge pulled down to its level
                    // there sinks into the ground in teeth. It is looked for
                    // under its seed level too: a lake standing over that
                    // level has no seed-derived water at its own.
                    let probe = DVec3::new(centre.x, level.min(seed.level) - 0.02, centre.z);
                    let shows = ground.implicit(probe).is_some()
                        || self.joined.contains_key(&WaterCell::containing(probe));
                    if shows && (level - column.level).abs() <= JOINS_METRES {
                        anchors.insert(
                            key,
                            Column {
                                level,
                                // Sounded once every anchor is known.
                                depth: ANCHOR_METRES,
                                flow: drawn.flow,
                                anchor: true,
                                edge: false,
                            },
                        );
                    }
                }
            }
        }
        columns.extend(anchors);
    }
}

/// Adds a ring of empty edge columns around the visible stored water, each
/// at the level of the highest water beside it. Without it the water's
/// outline is its columns' outline, a staircase of 20 cm steps.
fn add_edges(columns: &mut CellMap<(i32, i32), Column>) {
    let mut edges = CellMap::<(i32, i32), Column>::default();
    for (&(x, z), column) in columns.iter().filter(|(_, column)| !column.anchor) {
        for dz in -1..=1 {
            for dx in -1..=1 {
                let key = (x + dx, z + dz);
                if columns.contains_key(&key) {
                    continue;
                }
                let edge = edges.entry(key).or_insert(Column {
                    depth: 0.0,
                    edge: true,
                    ..*column
                });
                if column.level > edge.level {
                    edge.level = column.level;
                    edge.flow = column.flow;
                }
            }
        }
    }
    columns.extend(edges);
}

/// What a tile shows, to the millimetre.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a millimetre fingerprint of the drawn levels"
)]
fn fingerprint(columns: &CellMap<(i32, i32), Column>, members: &[(i32, i32)]) -> u64 {
    let mut hasher = CellHasher::default();
    for key in members {
        let column = columns[key];
        key.hash(&mut hasher);
        ((column.level * 500.0).round() as i64).hash(&mut hasher);
        ((column.depth * 500.0).round() as i64).hash(&mut hasher);
        ((column.flow.x * 10.0).round() as i64).hash(&mut hasher);
        ((column.flow.y * 10.0).round() as i64).hash(&mut hasher);
    }
    hasher.finish()
}

/// Meshes one tile: a quad per column, its corners shared with every
/// neighbour of the same water.
fn mesh_tile(
    columns: &CellMap<(i32, i32), Column>,
    key: (i32, i32),
    members: &[(i32, i32)],
    fingerprint: u64,
    top: &mut impl FnMut(i32, i32, f64) -> Option<f64>,
) -> SurfaceTile {
    let edge = WATER_CELL_METRES;
    let origin = DVec3::new(
        f64::from(key.0 * SURFACE_TILE_COLUMNS) * edge,
        0.0,
        f64::from(key.1 * SURFACE_TILE_COLUMNS) * edge,
    );
    let mut tile = SurfaceTile {
        key,
        fingerprint,
        origin,
        ..SurfaceTile::default()
    };
    // Corners met again by a neighbour of the same water, found by where
    // they stand and how high, to the tenth of a millimetre.
    let mut shared = CellMap::<(i32, i32, i64), u32>::default();
    for &(x, z) in members {
        let column = columns[&(x, z)];
        let mut quad = [0_u32; 4];
        for (slot, (cx, cz)) in quad.iter_mut().zip([(0, 0), (0, 1), (1, 0), (1, 1)]) {
            let (corner_x, corner_z) = (x + cx, z + cz);
            let corner = corner(columns, corner_x, corner_z, column, top);
            #[expect(clippy::cast_possible_truncation, reason = "a tenth of a millimetre")]
            let height = (corner.level * 10_000.0).round() as i64;
            *slot = *shared
                .entry((corner_x, corner_z, height))
                .or_insert_with(|| {
                    let index = u32::try_from(tile.positions.len())
                        .expect("a surface tile fits u32 indices");
                    let position = DVec3::new(
                        f64::from(corner_x) * edge,
                        corner.level,
                        f64::from(corner_z) * edge,
                    ) - origin;
                    tile.positions.push(position.as_vec3().to_array());
                    tile.normals.push(corner.normal.as_vec3().to_array());
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "shader attributes are f32"
                    )]
                    tile.attributes.push([
                        corner.depth as f32,
                        corner.flow.x as f32,
                        corner.flow.y as f32,
                    ]);
                    index
                });
        }
        tile.indices
            .extend([quad[0], quad[1], quad[2], quad[2], quad[1], quad[3]]);
    }
    tile
}

/// One quad corner: its level, the depth it shows, the water's current and
/// the surface's normal there.
struct Corner {
    level: f64,
    depth: f64,
    flow: DVec2,
    normal: DVec3,
}

/// The corner at `(x, z)` of a column of water `own`, from the four columns
/// around it that belong to the same water. Shallow water lies over the
/// drawn ground, `top` giving its height at a corner near a level: a mean
/// level over columns at different heights would sink a film between two
/// deeper rills into the ground, and the ground would show through it in
/// teeth running down the slope.
fn corner(
    columns: &CellMap<(i32, i32), Column>,
    x: i32,
    z: i32,
    own: Column,
    top: &mut impl FnMut(i32, i32, f64) -> Option<f64>,
) -> Corner {
    let around = [(-1, -1), (-1, 0), (0, -1), (0, 0)]
        .map(|(dx, dz)| columns.get(&(x + dx, z + dz)).copied());
    // Edge columns hold no water: they weigh in on no corner that has water
    // of its own around it.
    let same = |column: &Column| !column.edge && (column.level - own.level).abs() <= JOINS_METRES;
    let (mut level, mut weight, mut depth, mut flow) = (0.0, 0.0, 0.0, DVec2::ZERO);
    for column in around.iter().flatten().filter(|column| same(column)) {
        let w = weighs(column);
        level += column.level * w;
        weight += w;
        depth += column.depth;
        flow += column.flow * w;
    }
    if weight == 0.0 {
        return beyond(x, z, own, top);
    }
    // Dry ground beside a corner counts as no depth, so the water fades out
    // at its edges; beside seed-derived water the columns missing around it
    // are that water's, drawn by its own sheet as deep as the columns here.
    let present = around.iter().flatten().filter(|column| same(column));
    let shared = if present.clone().any(|column| column.anchor) {
        f64::from(u8::try_from(present.count()).expect("four columns at most"))
    } else {
        4.0
    };
    // Dry ground beside the corner shows no depth: the water fades out.
    let level_at = |column: Option<Column>| column.filter(same).map(|column| column.level);
    let height = |dx: usize, dz: usize| level_at(around[dx * 2 + dz]);
    let slope = |a: Option<f64>, b: Option<f64>| match (a, b) {
        (Some(a), Some(b)) => b - a,
        _ => 0.0,
    };
    let edge = WATER_CELL_METRES;
    let dx = 0.5 * (slope(height(0, 0), height(1, 0)) + slope(height(0, 1), height(1, 1))) / edge;
    let dz = 0.5 * (slope(height(0, 0), height(0, 1)) + slope(height(1, 0), height(1, 1))) / edge;
    let mut corner = Corner {
        level: level / weight,
        depth: depth / shared,
        flow: flow / weight,
        normal: DVec3::new(-dx, 1.0, -dz).normalize(),
    };
    drape(
        &around.map(|column| column.filter(same)),
        x,
        z,
        &mut corner,
        top,
    );
    corner
}

/// How much a column weighs in on the corners around it: by its depth, and
/// seed-derived water as deep water at least, so stored water meets it at its
/// level.
fn weighs(column: &Column) -> f64 {
    if column.anchor {
        column.depth.max(ANCHOR_METRES)
    } else {
        column.depth.max(LEAST_WEIGHT_METRES)
    }
}

/// The outer corner of an edge column, with no water around it: no depth, at
/// the water's level where the ground rises through it, so the terrain draws
/// the shore, and down on the ground where it falls away.
fn beyond(
    x: i32,
    z: i32,
    own: Column,
    top: &mut impl FnMut(i32, i32, f64) -> Option<f64>,
) -> Corner {
    Corner {
        level: top(x, z, own.level).map_or(own.level, |ground| ground.min(own.level)),
        depth: 0.0,
        flow: own.flow,
        normal: DVec3::Y,
    }
}

/// Lays a corner of shallow water over the drawn ground: each column around
/// it counts as the ground at the corner plus its own depth while shallow,
/// and at its own level once deep, and the surface's normal follows the
/// ground as far as the water lies on it. Water tilts along its current,
/// never across it: it lies on the ground only as far as the ground is gentle
/// across the current, or all round where it rests, so water beside a bank
/// stays level and the terrain draws its shore rather than the water
/// climbing the bank.
fn drape(
    around: &[Option<Column>; 4],
    x: i32,
    z: i32,
    corner: &mut Corner,
    top: &mut impl FnMut(i32, i32, f64) -> Option<f64>,
) {
    // Seed-derived water lies at its own level, as that water's sheet does.
    let level = |column: &Column| {
        if column.anchor {
            1.0
        } else {
            smoothstep(DRAPED_METRES, LEVEL_METRES, column.depth)
        }
    };
    if around.iter().flatten().all(|column| level(column) >= 1.0) {
        return;
    }
    let near = corner.level;
    let Some(ground) = top(x, z, near) else {
        return;
    };
    let edge = WATER_CELL_METRES;
    let slope = |a: Option<f64>, b: Option<f64>| a.zip(b).map(|(a, b)| (b - a) / (2.0 * edge));
    let gradient = slope(top(x - 1, z, near), top(x + 1, z, near))
        .zip(slope(top(x, z - 1, near), top(x, z + 1, near)))
        .map(|(dx, dz)| DVec2::new(dx, dz));
    let follows = gradient.map_or(1.0, |gradient| {
        let speed = corner.flow.length();
        let across = if speed > 0.0 {
            gradient.perp_dot(corner.flow / speed).abs()
        } else {
            0.0
        };
        let steep = (across - gradient.length()).mul_add(
            smoothstep(0.0, CURRENT_METRES_PER_SECOND, speed),
            gradient.length(),
        );
        1.0 - smoothstep(GENTLE_SLOPE, STEEP_SLOPE, steep)
    });
    let (mut height, mut lying, mut weight) = (0.0, 0.0, 0.0);
    for column in around.iter().flatten() {
        let w = weighs(column);
        let t = level(column);
        height += w * (ground + column.depth).mul_add(1.0 - t, column.level * t);
        lying += w * (1.0 - t);
        weight += w;
    }
    // Draping only ever lifts water the averaging sank into the ground: water
    // lying level over the ground, as at a lake's shallow edge, stays level.
    let lift = (height / weight - corner.level) * follows;
    if lift <= 0.0 {
        return;
    }
    corner.level += lift;
    let lying = lying / weight * smoothstep(0.0, LIFTED_METRES, lift);
    if let Some(gradient) = gradient {
        let ground = DVec3::new(-gradient.x, 1.0, -gradient.y).normalize();
        corner.normal = corner.normal.lerp(ground, lying).normalize();
    }
}

/// Ground slope across the current, rise over run, up to which shallow
/// water lies on it. Films run down gentler ground than this; water resting
/// against a bank or running along one meets steeper.
const GENTLE_SLOPE: f64 = 0.3;

/// Ground slope across the current, rise over run, from which shallow water
/// lies level over it rather than on it.
const STEEP_SLOPE: f64 = 0.45;

/// Current, in m/s, from which water tilts along it: slower water lies level
/// across ground steep in any direction.
const CURRENT_METRES_PER_SECOND: f64 = 0.05;

/// Lift over which a draped corner's normal turns to follow the ground, in
/// metres.
const LIFTED_METRES: f64 = 0.005;

/// 0 below `low`, 1 above `high`, and a smooth step between.
fn smoothstep(low: f64, high: f64, value: f64) -> f64 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * 2.0_f64.mul_add(-t, 3.0)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use bevy_math::{DVec2, DVec3};

    use super::super::cells::CellMap;
    use super::{Column, mesh_tile};

    /// Water running down a 1-in-3 slope, 8 columns long and 4 wide.
    fn ramp() -> CellMap<(i32, i32), Column> {
        let mut columns = CellMap::default();
        for x in 0..8 {
            for z in 0..4 {
                columns.insert(
                    (x, z),
                    Column {
                        level: -f64::from(x) * 0.2 / 3.0 + 0.05,
                        depth: 0.05,
                        flow: DVec2::new(1.0, 0.0),
                        anchor: false,
                        edge: false,
                    },
                );
            }
        }
        columns
    }

    #[test]
    fn water_down_a_slope_is_one_watertight_surface() {
        let columns = ramp();
        let mut members = columns.keys().copied().collect::<Vec<_>>();
        members.sort_unstable();
        let tile = mesh_tile(&columns, (0, 0), &members, 0, &mut |_, _, _| None);
        // Every interior edge is shared by two triangles and every boundary
        // edge by one: no gaps between neighbouring columns.
        let mut edges = HashMap::<(u32, u32), u32>::new();
        for triangle in tile.indices.chunks(3) {
            for (a, b) in [(0, 1), (1, 2), (2, 0)] {
                let (a, b) = (triangle[a], triangle[b]);
                *edges.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        let boundary = edges.values().filter(|&&count| count == 1).count();
        assert!(edges.values().all(|&count| count <= 2));
        // A 8 × 4 block of quads has 2 × (8 + 4) boundary edges.
        assert_eq!(boundary, 24, "the surface has holes or seams");
        // One vertex per corner: (8 + 1) × (4 + 1).
        assert_eq!(tile.positions.len(), 45);
        // The surface falls with the ground, facing up and downhill.
        let normal = DVec3::from_array(tile.normals[20].map(f64::from));
        assert!(normal.y > 0.9 && normal.x > 0.1, "normal {normal}");
    }

    #[test]
    fn water_far_below_other_water_keeps_its_own_edge() {
        let mut columns = CellMap::default();
        for (x, level) in [(0, 2.0), (1, 2.0), (2, 0.0), (3, 0.0)] {
            columns.insert(
                (x, 0),
                Column {
                    level,
                    depth: 0.1,
                    flow: DVec2::ZERO,
                    anchor: false,
                    edge: false,
                },
            );
        }
        let members = [(0, 0), (1, 0), (2, 0), (3, 0)];
        let tile = mesh_tile(&columns, (0, 0), &members, 0, &mut |_, _, _| None);
        let heights = tile
            .positions
            .iter()
            .map(|position| f64::from(position[1]))
            .collect::<Vec<_>>();
        assert!(
            heights
                .iter()
                .all(|&height| (height - 2.0).abs() < 1.0e-3 || height.abs() < 1.0e-3),
            "a ramp joined two separate waters: {heights:?}"
        );
    }

    #[test]
    fn shallow_water_meets_a_lake_beside_it_at_the_lake_level() {
        let mut columns = CellMap::default();
        columns.insert(
            (0, 0),
            Column {
                level: 0.7,
                depth: 0.05,
                flow: DVec2::ZERO,
                anchor: false,
                edge: false,
            },
        );
        columns.insert(
            (-1, 0),
            Column {
                level: 0.8,
                depth: super::ANCHOR_METRES,
                flow: DVec2::ZERO,
                anchor: true,
                edge: false,
            },
        );
        let tile = mesh_tile(&columns, (0, 0), &[(0, 0)], 0, &mut |_, _, _| None);
        // Only the stored water is drawn; its edge on the lake's side stands
        // at nearly the lake's level, the far edge at its own.
        assert_eq!(tile.positions.len(), 4);
        for position in &tile.positions {
            let level = f64::from(position[1]);
            let expected = if position[0] == 0.0 { 0.8 } else { 0.7 };
            assert!(
                (level - expected).abs() < 0.01,
                "a corner at x {} stands at {level:.3} m",
                position[0]
            );
        }
    }

    /// Ground falling 1 in 5 along x, with a groove 4 cm deep along z = 2.
    fn lumpy(x: f64, z: f64) -> f64 {
        let groove = (-(z - 0.5).powi(2) / 0.01).exp();
        -0.2 * x - 0.04 * groove + 0.01 * (x * 17.0).sin()
    }

    /// A film 5 mm deep down the slope, with a rill 5 cm deep in the groove.
    fn film_and_rill() -> CellMap<(i32, i32), Column> {
        let mut columns = CellMap::default();
        for x in 0..8 {
            for z in 0..5 {
                let (cx, cz) = ((f64::from(x) + 0.5) * 0.2, (f64::from(z) + 0.5) * 0.2);
                let depth = if z == 2 { 0.05 } else { 0.005 };
                columns.insert(
                    (x, z),
                    Column {
                        level: lumpy(cx, cz) + depth,
                        depth,
                        flow: DVec2::new(0.5, 0.0),
                        anchor: false,
                        edge: false,
                    },
                );
            }
        }
        columns
    }

    /// How far the lowest vertex over the film's columns lies under the
    /// ground, in metres.
    fn deepest_under(tile: &super::SurfaceTile) -> f64 {
        tile.positions
            .iter()
            .map(|position| {
                let [x, y, z] = position.map(f64::from);
                lumpy(x, z) - y
            })
            .fold(f64::NEG_INFINITY, f64::max)
    }

    #[test]
    fn a_film_beside_a_rill_lies_over_the_ground_it_runs_on() {
        let columns = film_and_rill();
        let mut members = columns.keys().copied().collect::<Vec<_>>();
        members.sort_unstable();
        // Levels averaged over the film and the rill in its groove sink the
        // film's corners into the ground beside the groove.
        let level = mesh_tile(&columns, (0, 0), &members, 0, &mut |_, _, _| None);
        assert!(
            deepest_under(&level) > 0.005,
            "the averaged surface never dips under the ground"
        );
        let edge = super::WATER_CELL_METRES;
        let draped = mesh_tile(&columns, (0, 0), &members, 0, &mut |x, z, _| {
            Some(lumpy(f64::from(x) * edge, f64::from(z) * edge))
        });
        let under = deepest_under(&draped);
        assert!(
            under < -super::VISIBLE_METRES + 1.0e-9,
            "the film dips {under:.4} m under the ground"
        );
    }

    #[test]
    fn a_lake_shelving_onto_its_shore_lies_level() {
        // The drawn ground lies a little under each column's floor, as the
        // terrain mesh does, and rises towards the shore at x = 8.
        let ground = |x: f64| 0.8 + 0.15 * x / 8.0;
        let mut columns = CellMap::default();
        for x in 0..8 {
            for z in 0..4 {
                let floor = ground(f64::from(x) + 0.5) + 0.025;
                columns.insert(
                    (x, z),
                    Column {
                        level: 1.0,
                        depth: 1.0 - floor,
                        flow: DVec2::ZERO,
                        anchor: false,
                        edge: false,
                    },
                );
            }
        }
        let mut members = columns.keys().copied().collect::<Vec<_>>();
        members.sort_unstable();
        let tile = mesh_tile(&columns, (0, 0), &members, 0, &mut |x, _, _| {
            Some(ground(f64::from(x)))
        });
        assert!(
            tile.positions
                .iter()
                .all(|position| (f64::from(position[1]) - 1.0).abs() < 1.0e-3),
            "the lake's shallow edge sinks towards the ground"
        );
        assert!(
            tile.normals
                .iter()
                .all(|normal| f64::from(normal[1]) > 0.999),
            "the lake's shallow edge is lit as a slope"
        );
    }

    /// Water 5 cm deep over flat ground in the columns on one side of a
    /// diagonal, with its ring of edge columns.
    fn diagonal() -> CellMap<(i32, i32), Column> {
        let mut columns = CellMap::default();
        for x in 0..16 {
            for z in 0..16 - x {
                columns.insert(
                    (x, z),
                    Column {
                        level: 1.0,
                        depth: 0.05,
                        flow: DVec2::ZERO,
                        anchor: false,
                        edge: false,
                    },
                );
            }
        }
        super::add_edges(&mut columns);
        columns
    }

    /// How opaque the drawn water is at a point, as the shader fades it out
    /// by the depth under it.
    fn drawn(tile: &super::SurfaceTile, point: DVec2) -> f64 {
        let mut alpha: f64 = 0.0;
        for triangle in tile.indices.chunks(3) {
            let [first, second, third] = [0, 1, 2].map(|corner| {
                let [x, _, z] = tile.positions[triangle[corner] as usize].map(f64::from);
                DVec2::new(x, z)
            });
            let area = (second - first).perp_dot(third - first);
            let (along_second, along_third) = (
                (point - first).perp_dot(third - first) / area,
                (second - first).perp_dot(point - first) / area,
            );
            if along_second < -1.0e-9
                || along_third < -1.0e-9
                || along_second + along_third > 1.0 + 1.0e-9
            {
                continue;
            }
            let depth =
                [0, 1, 2].map(|corner| f64::from(tile.attributes[triangle[corner] as usize][0]));
            let depth = depth[2].mul_add(
                along_third,
                depth[1].mul_add(along_second, (1.0 - along_second - along_third) * depth[0]),
            );
            alpha = alpha.max(super::smoothstep(0.003, 0.012, depth));
        }
        alpha
    }

    #[test]
    fn a_diagonal_edge_of_water_is_a_straight_line_not_steps() {
        let columns = diagonal();
        let mut members = columns.keys().copied().collect::<Vec<_>>();
        members.sort_unstable();
        let tile = mesh_tile(&columns, (0, 0), &members, 0, &mut |_, _, _| Some(0.95));
        // Where the water ends along lines across the edge, a centimetre
        // apart along it.
        let edge = super::WATER_CELL_METRES;
        let across = DVec2::new(1.0, 1.0).normalize();
        let ends = (0..80)
            .map(|step| {
                let along = DVec2::new(8.0 * edge, 8.0 * edge)
                    + DVec2::new(1.0, -1.0).normalize() * (f64::from(step) * 0.01 - 0.4);
                let mut reach = -0.5;
                while drawn(&tile, along + across * reach) >= 0.5 {
                    reach += 0.002;
                }
                reach
            })
            .collect::<Vec<_>>();
        let spread = ends.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - ends.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        assert!(spread < 0.02, "the edge steps by {spread:.3} m");
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "the water's own corners are exactly as they were"
    )]
    fn edge_columns_leave_the_water_and_meet_a_bank() {
        let mut columns = CellMap::default();
        for x in 2..6 {
            for z in 2..6 {
                columns.insert(
                    (x, z),
                    Column {
                        level: 1.0,
                        depth: 0.3,
                        flow: DVec2::ZERO,
                        anchor: false,
                        edge: false,
                    },
                );
            }
        }
        let mut members = columns.keys().copied().collect::<Vec<_>>();
        members.sort_unstable();
        // A bank over the water on the low x side, a floor under it on the
        // high x side.
        let ground = |x: i32| if x <= 2 { 1.2 } else { 0.9 };
        let mut top = |x: i32, _: i32, _: f64| Some(ground(x));
        let without = mesh_tile(&columns, (0, 0), &members, 0, &mut top);
        super::add_edges(&mut columns);
        let mut members = columns.keys().copied().collect::<Vec<_>>();
        members.sort_unstable();
        let with = mesh_tile(&columns, (0, 0), &members, 0, &mut top);
        // The water's own corners are as they were.
        for (position, attributes) in without.positions.iter().zip(&without.attributes) {
            assert!(
                with.positions
                    .iter()
                    .zip(&with.attributes)
                    .any(|(other, others)| other == position && others == attributes),
                "the corner at {position:?} moved"
            );
        }
        // Past them the surface runs on into the bank, and down onto the floor.
        let edge = super::WATER_CELL_METRES;
        for (position, attributes) in with.positions.iter().zip(&with.attributes) {
            if attributes[0] > 0.0 {
                continue;
            }
            #[expect(clippy::cast_possible_truncation, reason = "a corner of the grid")]
            let x = (f64::from(position[0]) / edge).round() as i32;
            let expected = ground(x).min(1.0);
            assert!(
                (f64::from(position[1]) - expected).abs() < 1.0e-6,
                "an outer corner at {position:?} stands at {} m, not {expected} m",
                position[1]
            );
        }
    }

    /// Water 1 cm deep over flat ground, 4 columns by 4, with `flow`, beside
    /// a bank rising 1 in 1 from inside its last column along x, meshed over
    /// the ground under it.
    fn beside_a_bank(flow: DVec2) -> super::SurfaceTile {
        let mut columns = CellMap::default();
        for x in 0..4 {
            for z in 0..4 {
                columns.insert(
                    (x, z),
                    Column {
                        level: 1.0,
                        depth: 0.01,
                        flow,
                        anchor: false,
                        edge: false,
                    },
                );
            }
        }
        let mut members = columns.keys().copied().collect::<Vec<_>>();
        members.sort_unstable();
        let edge = super::WATER_CELL_METRES;
        mesh_tile(&columns, (0, 0), &members, 0, &mut |x, _, _| {
            Some(0.99 + (f64::from(x) * edge - 0.7).max(0.0))
        })
    }

    #[test]
    fn shallow_water_against_a_steep_bank_stays_level() {
        // At rest, and running along the bank.
        for flow in [DVec2::ZERO, DVec2::new(0.0, 0.5)] {
            let tile = beside_a_bank(flow);
            for position in &tile.positions {
                assert!(
                    (f64::from(position[1]) - 1.0).abs() < 0.005,
                    "water running at {flow} climbs the bank to {} m at x {}",
                    position[1],
                    position[0]
                );
            }
        }
    }

    #[test]
    fn a_film_down_a_steep_hill_lies_on_it() {
        // A film 5 mm deep running down ground falling 1 in 2, with a groove
        // across the current; the ground rises across the film's upper edge.
        let edge = super::WATER_CELL_METRES;
        let ground = |x: f64, z: f64| -0.5 * x - 0.03 * (-(z - 0.5).powi(2) / 0.01).exp();
        let mut columns = CellMap::default();
        for x in 0..8 {
            for z in 0..5 {
                let (cx, cz) = ((f64::from(x) + 0.5) * edge, (f64::from(z) + 0.5) * edge);
                let depth = if z == 2 { 0.05 } else { 0.005 };
                columns.insert(
                    (x, z),
                    Column {
                        level: ground(cx, cz) + depth,
                        depth,
                        flow: DVec2::new(1.0, 0.0),
                        anchor: false,
                        edge: false,
                    },
                );
            }
        }
        let mut members = columns.keys().copied().collect::<Vec<_>>();
        members.sort_unstable();
        let tile = mesh_tile(&columns, (0, 0), &members, 0, &mut |x, z, _| {
            Some(ground(f64::from(x) * edge, f64::from(z) * edge))
        });
        for position in &tile.positions {
            let [x, y, z] = position.map(f64::from);
            assert!(
                y - ground(x, z) > super::VISIBLE_METRES - 1.0e-9,
                "the film at ({x:.1}, {z:.1}) sinks {:.4} m into the hill",
                ground(x, z) - y
            );
        }
    }

    #[test]
    fn seed_water_beside_stored_water_shows_its_own_depth_to_its_edge() {
        let mut columns = CellMap::default();
        columns.insert(
            (0, 0),
            Column {
                level: 0.95,
                depth: 0.3,
                flow: DVec2::new(1.0, 0.0),
                anchor: false,
                edge: false,
            },
        );
        columns.insert(
            (-1, 0),
            Column {
                level: 1.0,
                depth: 2.0,
                flow: DVec2::ZERO,
                anchor: true,
                edge: false,
            },
        );
        let tile = mesh_tile(&columns, (-1, -1), &[(-1, 0), (0, 0)], 0, &mut |_, _, _| {
            None
        });
        // The lake's column is drawn with the running water, and its far
        // edge, where the lake's own sheet takes over, stands at the lake's
        // level and depth: the lake beyond it is no dry ground.
        let edge = super::WATER_CELL_METRES;
        let far = tile
            .positions
            .iter()
            .zip(&tile.attributes)
            .filter(|(position, _)| (f64::from(position[0]) + tile.origin.x + edge).abs() < 1.0e-4)
            .collect::<Vec<_>>();
        assert_eq!(far.len(), 2, "the lake's column is not drawn");
        for (position, attributes) in far {
            assert!(
                (f64::from(position[1]) - 1.0).abs() < 1.0e-6,
                "the edge stands at {}",
                position[1]
            );
            assert!(
                (f64::from(attributes[0]) - 2.0).abs() < 1.0e-6,
                "the edge shows {} m of water",
                attributes[0]
            );
        }
    }

    #[test]
    fn a_pond_over_lumpy_ground_lies_level() {
        let mut columns = CellMap::default();
        for x in 0..6 {
            for z in 0..6 {
                columns.insert(
                    (x, z),
                    Column {
                        level: 1.0,
                        depth: 0.3,
                        flow: DVec2::ZERO,
                        anchor: false,
                        edge: false,
                    },
                );
            }
        }
        let mut members = columns.keys().copied().collect::<Vec<_>>();
        members.sort_unstable();
        let edge = super::WATER_CELL_METRES;
        let tile = mesh_tile(&columns, (0, 0), &members, 0, &mut |x, z, _| {
            Some(0.7 + lumpy(f64::from(x) * edge, f64::from(z) * edge))
        });
        assert!(
            tile.positions
                .iter()
                .all(|position| (f64::from(position[1]) - 1.0).abs() < 1.0e-3),
            "the pond follows the ground under it"
        );
    }
}
