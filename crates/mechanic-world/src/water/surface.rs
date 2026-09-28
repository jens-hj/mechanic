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
//! so the water fades out at its edges.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use bevy_math::{DVec2, DVec3};

use super::cells::{CellHasher, CellMap};
use super::{WATER_CELL_METRES, WaterCell, WaterGround, WaterWorld};

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
    /// Seed-derived water beside stored water, which draws itself: it only
    /// holds the corners it shares with stored water at its level.
    anchor: bool,
}

/// Depth an anchor weighs in with on a corner, in metres: as deep water, so
/// stored water meets a lake at the lake's level and colour.
const ANCHOR_METRES: f64 = 1.0;

/// Running water this near a lake's level, in metres, meets the lake's own
/// sheet, which runs on over it.
const MEETS_METRES: f64 = 0.1;

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
    /// The surface of the stored water, tile by tile. Tiles whose
    /// fingerprint matches `drawn` come back without a mesh: they have not
    /// changed.
    pub fn surface_tiles(
        &self,
        ground: &impl WaterGround,
        drawn: &HashMap<(i32, i32), u64>,
    ) -> Vec<SurfaceTile> {
        let columns = self.visible_columns(ground);
        let mut tiles = CellMap::<(i32, i32), Vec<(i32, i32)>>::default();
        for (&(x, z), _) in columns.iter().filter(|(_, column)| !column.anchor) {
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
            out.push(mesh_tile(&columns, key, &members, fingerprint));
        }
        out.sort_unstable_by_key(|tile| tile.key);
        out
    }

    /// Columns of running water standing at the level of the lake or river
    /// whose sheet reaches over them, as in a channel dug from a lake: the
    /// sheet runs on over them to meet the running water's surface.
    pub fn meeting_columns(&self, ground: &impl WaterGround) -> Vec<(i32, i32)> {
        self.running_cells()
            .into_iter()
            .filter(|view| view.depth >= VISIBLE_METRES)
            .filter(|view| {
                let centre = view.cell.centre();
                ground
                    .surface(centre.x, centre.z)
                    .is_some_and(|seed| view.level >= self.drawn(ground, seed).level - MEETS_METRES)
            })
            .map(|view| (view.cell.x, view.cell.z))
            .collect()
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
                    },
                );
            }
        }
        // Cells joined to seed-derived water where its own sheet, which
        // draws the joined cells within its reach, does not run.
        let mut joined = CellMap::<(i32, i32), (WaterCell, f64, f64)>::default();
        for (cell, surface) in self.joined_cells(ground) {
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
                    },
                );
            }
        }
        self.anchor_to_seed_water(ground, &mut columns);
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
                    let level = self.drawn(ground, seed).level;
                    // Only where the seed-derived water shows: its sheet runs
                    // on under the bank, and an edge pulled down to its level
                    // there sinks into the ground in teeth.
                    let probe = DVec3::new(centre.x, level - 0.02, centre.z);
                    let shows = ground.implicit(probe).is_some()
                        || self.joined.contains_key(&WaterCell::containing(probe));
                    if shows && (level - column.level).abs() <= JOINS_METRES {
                        anchors.insert(
                            key,
                            Column {
                                level,
                                depth: ANCHOR_METRES,
                                flow: DVec2::ZERO,
                                anchor: true,
                            },
                        );
                    }
                }
            }
        }
        columns.extend(anchors);
    }
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
            let corner = corner(columns, corner_x, corner_z, column);
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
/// around it that belong to the same water.
fn corner(columns: &CellMap<(i32, i32), Column>, x: i32, z: i32, own: Column) -> Corner {
    let around = [(-1, -1), (-1, 0), (0, -1), (0, 0)]
        .map(|(dx, dz)| columns.get(&(x + dx, z + dz)).copied());
    let same = |column: &Column| (column.level - own.level).abs() <= JOINS_METRES;
    let (mut level, mut weight, mut depth, mut flow) = (0.0, 0.0, 0.0, DVec2::ZERO);
    for column in around.iter().flatten().filter(|column| same(column)) {
        let w = column.depth.max(LEAST_WEIGHT_METRES);
        level += column.level * w;
        weight += w;
        depth += column.depth;
        flow += column.flow * w;
    }
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
    Corner {
        level: level / weight,
        depth: depth / 4.0,
        flow: flow / weight,
        normal: DVec3::new(-dx, 1.0, -dz).normalize(),
    }
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
        let tile = mesh_tile(&columns, (0, 0), &members, 0);
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
                },
            );
        }
        let members = [(0, 0), (1, 0), (2, 0), (3, 0)];
        let tile = mesh_tile(&columns, (0, 0), &members, 0);
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
            },
        );
        columns.insert(
            (-1, 0),
            Column {
                level: 0.8,
                depth: super::ANCHOR_METRES,
                flow: DVec2::ZERO,
                anchor: true,
            },
        );
        let tile = mesh_tile(&columns, (0, 0), &[(0, 0)], 0);
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
}
