//! Water surface sheets: one tile of the sea, lakes and rivers as a grid at
//! each column's water level.
//!
//! The sheet is not clipped to the shore. It runs on under ground that
//! stands above the water, where the terrain hides it, so a shoreline is
//! exactly where the terrain mesh crosses the surface at any level of detail.
//! It is cut only where open ground at the surface is not water: a dry void
//! under a lake, though not ground dug away beside one.
//!
//! Where stored water's own surface draws columns of the lake, or the lake
//! ends in open air beyond its reach, as at a channel dug from it, a grid
//! square is cut into 20 cm columns: the sheet draws every column of its
//! reach the stored water does not, so the two surfaces meet column to column,
//! neither fading out nor overlapping.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::BuildHasher;

use bevy_math::DVec3;

use crate::{
    TerrainField, TerrainSource, WATER_CELL_METRES, WaterBody, WaterCell, WaterShift, WorldPosition,
};

/// How far below its level the surface is probed, in metres.
const SURFACE_PROBE_METRES: f64 = 0.02;

/// Deepest water a sheet measures under a vertex, in metres.
pub(crate) const DEEPEST_METRES: f64 = 16.0;

/// Coarsest grid, in metres, whose squares are cut into columns: coarser
/// tiles lie hundreds of metres out, where a column is under a pixel.
const CUT_SPACING_METRES: f64 = 2.0;

/// One square tile of water surface to mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterTile {
    /// Lower x and z corner, in global metres.
    pub minimum: [f64; 2],
    /// Edge length in metres.
    pub edge: f64,
    /// Grid cells per edge.
    pub cells: u32,
}

/// A tile's water surface.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WaterSheet {
    /// Global position vertices are relative to: the tile's lower corner at
    /// height zero.
    pub origin: WorldPosition,
    /// Vertex positions relative to `origin`.
    pub vertices: Vec<[f32; 3]>,
    /// Depth of water under each vertex, in metres, capped at 16 m; zero
    /// where the surface runs under ground.
    pub depths: Vec<f32>,
    /// Surface current at each vertex along x and z, in m/s.
    pub flows: Vec<[f32; 2]>,
    /// Upward-facing triangles.
    pub indices: Vec<u32>,
}

/// What the surface does at one grid vertex.
#[derive(Clone, Copy)]
struct Vertex {
    level: f64,
    depth: f64,
    flow: [f64; 2],
    /// Open water at the surface.
    open: bool,
    /// Under ground at the surface, hidden by the terrain, or at the edge
    /// of the water where the sheet ends.
    buried: bool,
}

/// Meshes one tile of water surface over untouched and edited ground, or
/// `None` where the tile holds no water. Each lake and river stands where
/// `shifts` moved it from its seed surface.
///
/// # Panics
///
/// Panics only if a tile exceeds the `u32` mesh-index contract.
pub fn water_sheet(
    field: &TerrainField,
    edits: &impl TerrainSource,
    tile: WaterTile,
    shifts: &BTreeMap<WaterBody, WaterShift>,
) -> Option<WaterSheet> {
    joined_water_sheet(field, edits, tile, shifts, &HashSet::new(), &HashSet::new())
}

/// Meshes one tile of water surface as [`water_sheet`] does, with the cells
/// dug out beside or under seed-derived water and `joined` to it counted as
/// that water: one surface runs on over them, as deep as the dug ground.
/// The water-cell columns in `owned` are drawn by the stored water's surface:
/// the sheet runs on to them as open water and leaves each out, cutting the
/// grid squares they lie in into columns.
///
/// # Panics
///
/// Panics only if a tile exceeds the `u32` mesh-index contract.
pub fn joined_water_sheet<S: BuildHasher, T: BuildHasher>(
    field: &TerrainField,
    edits: &impl TerrainSource,
    tile: WaterTile,
    shifts: &BTreeMap<WaterBody, WaterShift>,
    joined: &HashSet<WaterCell, S>,
    owned: &HashSet<(i32, i32), T>,
) -> Option<WaterSheet> {
    let [x0, z0] = tile.minimum;
    field.water_level_range(
        DVec3::new(x0, 0.0, z0),
        DVec3::new(x0 + tile.edge, 0.0, z0 + tile.edge),
    )?;
    let meshing = Meshing {
        field,
        edits,
        shifts,
        joined,
        owned,
        minimum: tile.minimum,
        side: tile.cells as usize + 1,
        spacing: tile.edge / f64::from(tile.cells),
    };
    let mut grid = meshing.grid();
    let mut cut = meshing.cut(&grid);
    meshing.fill(&mut grid, &mut cut);
    meshing.assemble(&grid, &cut)
}

/// The water at a point of a sheet's surface.
#[derive(Clone, Copy)]
struct Water {
    surface: crate::WaterSurface,
    /// The ground stands over the surface there.
    under_ground: bool,
    /// The seed's water, or water joined to it, lies under the surface.
    wet: bool,
    /// The ground there was edited.
    dug: bool,
}

/// One tile of water surface being meshed: the ground and water it lies
/// over, and its grid.
struct Meshing<'a, E, S, T> {
    field: &'a TerrainField,
    edits: &'a E,
    shifts: &'a BTreeMap<WaterBody, WaterShift>,
    joined: &'a HashSet<WaterCell, S>,
    owned: &'a HashSet<(i32, i32), T>,
    minimum: [f64; 2],
    /// Grid vertices along each edge.
    side: usize,
    spacing: f64,
}

impl<E: TerrainSource, S: BuildHasher, T: BuildHasher> Meshing<'_, E, S, T> {
    /// Grid squares along each edge.
    const fn squares(&self) -> usize {
        self.side - 1
    }

    /// Where a grid vertex stands.
    #[expect(clippy::cast_precision_loss, reason = "a tile is a few hundred cells")]
    fn at(&self, index: usize) -> (f64, f64) {
        (
            (index % self.side) as f64 * self.spacing + self.minimum[0],
            (index / self.side) as f64 * self.spacing + self.minimum[1],
        )
    }

    /// Density of edited ground at a point, where its brick was edited.
    fn edited(&self, point: DVec3) -> Option<f64> {
        WorldPosition(point).cell().ok().and_then(|cell| {
            self.edits
                .brick(cell.brick())
                .and_then(|brick| brick.sample(cell.local_in_brick()))
                .map(|sample| f64::from(sample.density))
        })
    }

    fn density(&self, point: DVec3) -> f64 {
        self.edited(point)
            .unwrap_or_else(|| self.field.density(point))
    }

    /// Depth of open water under a point of the surface at a level.
    fn depth_under(&self, x: f64, z: f64, level: f64) -> f64 {
        let mut y = level;
        while level - y < DEEPEST_METRES {
            let value = self.density(DVec3::new(x, y, z));
            if value > 0.0 {
                break;
            }
            y -= (-value * 0.9).clamp(0.05, 4.0);
        }
        (level - y).min(DEEPEST_METRES)
    }

    /// Whether the stored water draws any of the four columns around a grid
    /// vertex.
    fn handed(&self, x: f64, z: f64) -> bool {
        #[expect(clippy::cast_possible_truncation, reason = "far inside i32")]
        let (column, row) = (
            (x / WATER_CELL_METRES).round() as i32,
            (z / WATER_CELL_METRES).round() as i32,
        );
        [(-1, -1), (-1, 0), (0, -1), (0, 0)]
            .iter()
            .any(|(dx, dz)| self.owned.contains(&(column + dx, row + dz)))
    }

    /// The water at a point of the surface, if a lake or river reaches it.
    fn water_at(&self, x: f64, z: f64) -> Option<Water> {
        let seed = self.field.water_surface(x, z)?;
        let surface = self
            .shifts
            .get(&seed.body)
            .map_or(seed, |shift| shift.apply(seed));
        let probe = DVec3::new(x, surface.level - SURFACE_PROBE_METRES, z);
        let dug = self.edited(probe);
        let under_ground = dug.unwrap_or_else(|| self.field.density(probe)) > 0.0;
        // The seed's water under the surface: a lake standing over its seed
        // level also covers the bank that stood above that level.
        let seeded = DVec3::new(x, seed.level.min(surface.level) - SURFACE_PROBE_METRES, z);
        let seed_water = self.field.is_water(seeded)
            || (surface.level > seed.level && self.density(seeded) > 0.0);
        Some(Water {
            surface,
            under_ground,
            wet: seed_water || self.joined.contains(&WaterCell::containing(probe)),
            dug: dug.is_some(),
        })
    }

    /// The corners of every grid square holding joined water: the sheet
    /// draws the joined cells, so it shows over all of each such square even
    /// where a corner stands in running water or dug ground.
    fn joining(&self) -> HashSet<usize> {
        let (side, cells) = (self.side, self.squares());
        let mut joining = HashSet::new();
        for cell in self.joined {
            let centre = cell.centre();
            let (column, row) = (
                (centre.x - self.minimum[0]) / self.spacing,
                (centre.z - self.minimum[1]) / self.spacing,
            );
            #[expect(clippy::cast_precision_loss, reason = "a tile is a few hundred cells")]
            let inside =
                (0.0..cells as f64).contains(&column) && (0.0..cells as f64).contains(&row);
            if inside {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "inside the tile"
                )]
                let corner = column as usize + row as usize * side;
                joining.extend([corner, corner + 1, corner + side, corner + side + 1]);
            }
        }
        joining
    }

    /// What the surface does at each grid vertex.
    fn grid(&self) -> Vec<Option<Vertex>> {
        let joining = self.joining();
        let grid = (0..self.side * self.side)
            .map(|index| {
                let (x, z) = self.at(index);
                let at = self.water_at(x, z)?;
                // Beside stored water drawing the lake's columns the sheet
                // stays open, as deep as the water there, so it never fades
                // out towards the stored water's surface.
                let open =
                    !at.under_ground && (at.wet || joining.contains(&index) || self.handed(x, z));
                // Ground dug away beside the water, not yet joined to it, is
                // where the sheet ends: cutting it there would leave the grid
                // squares around, and the water in them, undrawn.
                let buried = !open && (at.under_ground || at.dug);
                Some(Vertex {
                    level: at.surface.level,
                    depth: if open {
                        self.depth_under(x, z, at.surface.level)
                    } else {
                        0.0
                    },
                    flow: at.surface.flow.to_array(),
                    open,
                    buried,
                })
            })
            .collect::<Vec<_>>();
        // A dry vertex beside the water continues the surface under ground
        // that stands above it, so the sheet ends inside the bank rather
        // than at the grid, and the terrain draws the shoreline.
        (0..grid.len())
            .map(|index| {
                grid[index].or_else(|| {
                    let neighbour = self.highest_beside(&grid, index, |vertex| vertex.open)?;
                    let (x, z) = self.at(index);
                    let probe = DVec3::new(x, neighbour.level - SURFACE_PROBE_METRES, z);
                    (self.density(probe) > 0.0).then_some(Vertex {
                        depth: 0.0,
                        open: false,
                        buried: true,
                        ..neighbour
                    })
                })
            })
            .collect()
    }

    /// The highest vertex around a grid vertex that `counts`.
    fn highest_beside(
        &self,
        grid: &[Option<Vertex>],
        index: usize,
        counts: impl Fn(&Vertex) -> bool,
    ) -> Option<Vertex> {
        let side = self.side;
        let (column, row) = (index % side, index / side);
        (row.saturating_sub(1)..=(row + 1).min(side - 1))
            .flat_map(|z| {
                (column.saturating_sub(1)..=(column + 1).min(side - 1)).map(move |x| x + z * side)
            })
            .filter_map(|other| grid[other].filter(|vertex| counts(vertex)))
            .max_by(|first, second| first.level.total_cmp(&second.level))
    }

    /// Grid squares cut into columns: those showing open water that hold a
    /// column the stored water draws, or a corner the sheet cannot keep: one
    /// beyond the lake's reach, as where it ends over a channel dug from it,
    /// or over open ground that is not water. The columns of such a square
    /// that hold no water stay cut out.
    fn cut(&self, grid: &[Option<Vertex>]) -> Vec<bool> {
        let squares = self.squares();
        let mut cut = vec![false; squares * squares];
        if self.spacing > CUT_SPACING_METRES {
            return cut;
        }
        for &(column, row) in self.owned {
            let (along, down) = (
                ((f64::from(column) + 0.5).mul_add(WATER_CELL_METRES, -self.minimum[0])
                    / self.spacing)
                    .floor(),
                ((f64::from(row) + 0.5).mul_add(WATER_CELL_METRES, -self.minimum[1])
                    / self.spacing)
                    .floor(),
            );
            #[expect(clippy::cast_precision_loss, reason = "a tile is a few hundred cells")]
            let inside =
                (0.0..squares as f64).contains(&along) && (0.0..squares as f64).contains(&down);
            if inside {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "inside the tile"
                )]
                let square = along as usize + down as usize * squares;
                cut[square] = true;
            }
        }
        for (square, cut) in cut.iter_mut().enumerate() {
            let corners = corners(square, self.side);
            let reaches = corners.iter().any(|&corner| grid[corner].is_some());
            let shows = corners
                .iter()
                .any(|&corner| grid[corner].is_some_and(|vertex| vertex.open));
            let unkept = corners
                .iter()
                .any(|&corner| !grid[corner].is_some_and(|vertex| vertex.open || vertex.buried));
            *cut = (*cut && shows) || (unkept && reaches && (shows || self.holds_water(square)));
        }
        cut
    }

    /// The lower corner of a square, in water-cell columns, and how many
    /// columns it spans along each edge.
    fn columns_of(&self, square: usize) -> (i32, i32, i32) {
        let (x, z) = self.at(corners(square, self.side)[0]);
        #[expect(clippy::cast_possible_truncation, reason = "far inside i32")]
        let columns = (
            (x / WATER_CELL_METRES).round() as i32,
            (z / WATER_CELL_METRES).round() as i32,
            (self.spacing / WATER_CELL_METRES).round() as i32,
        );
        columns
    }

    /// Whether a column of a square holds the lake's water at its surface.
    fn holds_water(&self, square: usize) -> bool {
        let (column, row, across) = self.columns_of(square);
        (0..across).any(|down| {
            (0..across).any(|along| {
                let (x, z) = centre(column + along, row + down);
                self.water_at(x, z)
                    .is_some_and(|at| at.wet && !at.under_ground)
            })
        })
    }

    /// Continues the water beside each corner of a cut square beyond the
    /// reach, so the square's columns interpolate its surface; a square with
    /// a corner that finds no water beside it stays whole.
    fn fill(&self, grid: &mut [Option<Vertex>], cut: &mut [bool]) {
        let side = self.side;
        let filled = (0..grid.len())
            .filter(|&index| grid[index].is_none())
            .filter(|&index| squares_at(index, side).any(|square| cut[square]))
            .filter_map(|index| {
                let neighbour = self.highest_beside(grid, index, |_| true)?;
                let (x, z) = self.at(index);
                let open =
                    self.density(DVec3::new(x, neighbour.level - SURFACE_PROBE_METRES, z)) <= 0.0;
                Some((
                    index,
                    Vertex {
                        depth: if open {
                            self.depth_under(x, z, neighbour.level)
                        } else {
                            0.0
                        },
                        open: false,
                        buried: false,
                        ..neighbour
                    },
                ))
            })
            .collect::<Vec<_>>();
        for (index, vertex) in filled {
            grid[index] = Some(vertex);
        }
        for (square, cut) in cut.iter_mut().enumerate() {
            *cut &= corners(square, side)
                .iter()
                .all(|&corner| grid[corner].is_some());
        }
    }

    /// Whether a cut square draws a column: one the stored water does not,
    /// holding the lake's water or under ground, where the terrain hides the
    /// sheet; not open ground dug beside the lake before its water runs in.
    fn keeps(&self, column: i32, row: i32) -> bool {
        let (x, z) = centre(column, row);
        !self.owned.contains(&(column, row))
            && self
                .water_at(x, z)
                .is_some_and(|at| at.under_ground || at.wet)
    }

    /// Triangles over every grid square whose corners all hold water or lie
    /// under ground, and show some open water; a cut square draws the
    /// columns the sheet keeps.
    fn assemble(&self, grid: &[Option<Vertex>], cut: &[bool]) -> Option<WaterSheet> {
        let [x0, z0] = self.minimum;
        let mut sheet = WaterSheet {
            origin: WorldPosition(DVec3::new(x0, 0.0, z0)),
            ..WaterSheet::default()
        };
        let mut remap = vec![u32::MAX; grid.len()];
        let mut shared = HashMap::new();
        for (square, &cut) in cut.iter().enumerate() {
            if cut {
                self.cut_square(grid, square, &mut sheet, &mut shared);
                continue;
            }
            let [first, along, down, across] = corners(square, self.side);
            for triangle in [[first, down, along], [along, down, across]] {
                let corners = triangle.map(|index| grid[index]);
                let kept = corners
                    .iter()
                    .all(|corner| corner.is_some_and(|corner| corner.open || corner.buried));
                let shows = corners.iter().flatten().any(|corner| corner.open);
                if kept && shows {
                    for index in triangle {
                        if remap[index] == u32::MAX {
                            let point = grid[index].expect("only kept vertices are emitted");
                            let (x, z) = self.at(index);
                            remap[index] = push(&mut sheet, [x - x0, z - z0], point);
                        }
                        sheet.indices.push(remap[index]);
                    }
                }
            }
        }
        (!sheet.indices.is_empty()).then_some(sheet)
    }

    /// Draws the columns a cut square keeps, each corner interpolating the
    /// square's corners. Corners are shared by column corner in `shared`.
    fn cut_square(
        &self,
        grid: &[Option<Vertex>],
        square: usize,
        sheet: &mut WaterSheet,
        shared: &mut HashMap<(i32, i32), u32>,
    ) {
        let corners =
            corners(square, self.side).map(|index| grid[index].expect("a cut square is filled"));
        let (column, row, across) = self.columns_of(square);
        let kept = |corner: usize| corners[corner].open || corners[corner].buried;
        let mut vertex = |along: i32, down: i32, sheet: &mut WaterSheet| {
            *shared
                .entry((column + along, row + down))
                .or_insert_with(|| {
                    let (u, v) = (
                        f64::from(along) / f64::from(across),
                        f64::from(down) / f64::from(across),
                    );
                    let mix = |value: fn(&Vertex) -> f64| {
                        let near = value(&corners[0]).mul_add(1.0 - u, value(&corners[1]) * u);
                        let far = value(&corners[2]).mul_add(1.0 - u, value(&corners[3]) * u);
                        near.mul_add(1.0 - v, far * v)
                    };
                    let level = mix(|vertex| vertex.level);
                    let (x, z) = (
                        f64::from(column + along) * WATER_CELL_METRES,
                        f64::from(row + down) * WATER_CELL_METRES,
                    );
                    // Each column corner is sounded, as the stored water's
                    // columns are, so the two show one colour where they meet,
                    // save on an edge between two kept corners, where it follows
                    // the square beside it.
                    let follows = [
                        (along == 0, 0, 2),
                        (along == across, 1, 3),
                        (down == 0, 0, 1),
                        (down == across, 2, 3),
                    ]
                    .iter()
                    .any(|&(on, first, second)| on && kept(first) && kept(second));
                    let depth = if follows {
                        mix(|vertex| vertex.depth)
                    } else {
                        self.depth_under(x, z, level)
                    };
                    let point = Vertex {
                        level,
                        depth,
                        flow: [mix(|vertex| vertex.flow[0]), mix(|vertex| vertex.flow[1])],
                        open: true,
                        buried: false,
                    };
                    push(sheet, [x - self.minimum[0], z - self.minimum[1]], point)
                })
        };
        for down in 0..across {
            for along in 0..across {
                if !self.keeps(column + along, row + down) {
                    continue;
                }
                let quad = [(0, 0), (1, 0), (0, 1), (1, 1)]
                    .map(|(dx, dz)| vertex(along + dx, down + dz, sheet));
                sheet
                    .indices
                    .extend([quad[0], quad[2], quad[1], quad[1], quad[2], quad[3]]);
            }
        }
    }
}

/// The centre of a water-cell column.
fn centre(column: i32, row: i32) -> (f64, f64) {
    (
        (f64::from(column) + 0.5) * WATER_CELL_METRES,
        (f64::from(row) + 0.5) * WATER_CELL_METRES,
    )
}

/// Adds a vertex to a sheet at `[x, z]` from its origin, returning its index.
#[expect(clippy::cast_possible_truncation, reason = "mesh attributes are f32")]
fn push(sheet: &mut WaterSheet, [x, z]: [f64; 2], point: Vertex) -> u32 {
    let index = u32::try_from(sheet.vertices.len()).expect("a water tile fits u32 indices");
    sheet
        .vertices
        .push([x as f32, point.level as f32, z as f32]);
    sheet.depths.push(point.depth as f32);
    sheet.flows.push(point.flow.map(|value| value as f32));
    index
}

/// The grid indices of a square's corners: lower x and z, then along x,
/// along z, and both.
const fn corners(square: usize, side: usize) -> [usize; 4] {
    let squares = side - 1;
    let first = square % squares + square / squares * side;
    [first, first + 1, first + side, first + side + 1]
}

/// The squares a grid vertex is a corner of.
fn squares_at(index: usize, side: usize) -> impl Iterator<Item = usize> {
    let squares = side - 1;
    let (column, row) = (index % side, index / side);
    [(0, 0), (1, 0), (0, 1), (1, 1)]
        .into_iter()
        .filter_map(move |(dx, dz)| {
            let (x, z) = (column.checked_sub(dx)?, row.checked_sub(dz)?);
            (x < squares && z < squares).then_some(x + z * squares)
        })
}
