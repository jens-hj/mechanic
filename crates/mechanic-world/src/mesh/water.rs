//! Water surface sheets: one tile of the sea, lakes and rivers as a grid at
//! each column's water level.
//!
//! The sheet is not clipped to the shore. It runs on under ground that
//! stands above the water, where the terrain hides it, so a shoreline is
//! exactly where the terrain mesh crosses the surface at any level of detail.
//! It is cut only where open ground at the surface is not water: a dry void
//! under a lake.

use std::collections::HashSet;

use bevy_math::DVec3;

use crate::{TerrainField, TerrainSource, WaterCell, WorldPosition};

/// How far below its level the surface is probed, in metres.
const SURFACE_PROBE_METRES: f64 = 0.02;

/// Deepest water a sheet measures under a vertex, in metres.
const DEEPEST_METRES: f64 = 16.0;

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
    /// Under ground at the surface, hidden by the terrain.
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
    shifts: &std::collections::BTreeMap<crate::WaterBody, crate::WaterShift>,
) -> Option<WaterSheet> {
    joined_water_sheet(field, edits, tile, shifts, &HashSet::new(), &HashSet::new())
}

/// Meshes one tile of water surface as [`water_sheet`] does, with the cells
/// dug out beside or under seed-derived water and `joined` to it counted as
/// that water: one surface runs on over them, as deep as the dug ground.
/// Over the water-cell columns in `meeting`, where running water stands at
/// the lake's level, the sheet runs on and fades out, so it meets the
/// running water's own surface rather than stopping a grid square short.
///
/// # Panics
///
/// Panics only if a tile exceeds the `u32` mesh-index contract.
pub fn joined_water_sheet<S: std::hash::BuildHasher, T: std::hash::BuildHasher>(
    field: &TerrainField,
    edits: &impl TerrainSource,
    tile: WaterTile,
    shifts: &std::collections::BTreeMap<crate::WaterBody, crate::WaterShift>,
    joined: &HashSet<WaterCell, S>,
    meeting: &HashSet<(i32, i32), T>,
) -> Option<WaterSheet> {
    let [x0, z0] = tile.minimum;
    field.water_level_range(
        DVec3::new(x0, 0.0, z0),
        DVec3::new(x0 + tile.edge, 0.0, z0 + tile.edge),
    )?;
    let side = tile.cells as usize + 1;
    let spacing = tile.edge / f64::from(tile.cells);
    let density = |point: DVec3| {
        WorldPosition(point).cell().map_or(-1.0, |cell| {
            edits
                .brick(cell.brick())
                .and_then(|brick| brick.sample(cell.local_in_brick()))
                .map_or_else(|| field.density(point), |sample| f64::from(sample.density))
        })
    };
    let grid = (0..side * side)
        .map(|index| {
            #[expect(clippy::cast_precision_loss, reason = "a tile is a few hundred cells")]
            let (x, z) = (
                (index % side) as f64 * spacing + x0,
                (index / side) as f64 * spacing + z0,
            );
            let seed = field.water_surface(x, z)?;
            let surface = shifts
                .get(&seed.body)
                .map_or(seed, |shift| shift.apply(seed));
            let probe = DVec3::new(x, surface.level - SURFACE_PROBE_METRES, z);
            let cell = WaterCell::containing(probe);
            let under_ground = density(probe) > 0.0;
            let open = !under_ground && (field.is_water(probe) || joined.contains(&cell));
            // Running water at the lake's level hides the sheet as ground
            // over it would.
            let buried = !open && (under_ground || meeting.contains(&(cell.x, cell.z)));
            let mut depth = 0.0;
            if open {
                let mut y = surface.level;
                while surface.level - y < DEEPEST_METRES {
                    let value = density(DVec3::new(x, y, z));
                    if value > 0.0 {
                        break;
                    }
                    y -= (-value * 0.9).clamp(0.05, 4.0);
                }
                depth = (surface.level - y).min(DEEPEST_METRES);
            }
            Some(Vertex {
                level: surface.level,
                depth,
                flow: surface.flow.to_array(),
                open,
                buried,
            })
        })
        .collect::<Vec<_>>();
    // A dry vertex beside the water continues the surface under ground that
    // stands above it, so the sheet ends inside the bank rather than at the
    // grid, and the terrain draws the shoreline.
    let grid = (0..grid.len())
        .map(|index| {
            grid[index].or_else(|| {
                let (column, row) = (index % side, index / side);
                let neighbour = (row.saturating_sub(1)..=(row + 1).min(side - 1))
                    .flat_map(|z| {
                        (column.saturating_sub(1)..=(column + 1).min(side - 1))
                            .map(move |x| x + z * side)
                    })
                    .filter_map(|other| grid[other].filter(|vertex| vertex.open))
                    .max_by(|first, second| first.level.total_cmp(&second.level))?;
                #[expect(clippy::cast_precision_loss, reason = "a tile is a few hundred cells")]
                let (x, z) = (column as f64 * spacing + x0, row as f64 * spacing + z0);
                let probe = DVec3::new(x, neighbour.level - SURFACE_PROBE_METRES, z);
                (density(probe) > 0.0).then_some(Vertex {
                    depth: 0.0,
                    open: false,
                    buried: true,
                    ..neighbour
                })
            })
        })
        .collect::<Vec<_>>();

    assemble(&grid, side, spacing, [x0, z0])
}

/// Triangles over every grid square whose corners all hold water or lie
/// under ground, and show some open water.
fn assemble(
    grid: &[Option<Vertex>],
    side: usize,
    spacing: f64,
    [x0, z0]: [f64; 2],
) -> Option<WaterSheet> {
    let mut sheet = WaterSheet {
        origin: WorldPosition(DVec3::new(x0, 0.0, z0)),
        ..WaterSheet::default()
    };
    let mut remap = vec![u32::MAX; grid.len()];
    let mut vertex = |index: usize, sheet: &mut WaterSheet| {
        if remap[index] == u32::MAX {
            let point = grid[index].expect("only kept vertices are emitted");
            #[expect(clippy::cast_precision_loss, reason = "a tile is a few hundred cells")]
            let (x, z) = (
                (index % side) as f64 * spacing,
                (index / side) as f64 * spacing,
            );
            remap[index] =
                u32::try_from(sheet.vertices.len()).expect("a water tile fits u32 indices");
            sheet
                .vertices
                .push([x as f32, point.level as f32, z as f32]);
            sheet.depths.push(point.depth as f32);
            sheet.flows.push(point.flow.map(|value| value as f32));
        }
        remap[index]
    };
    for row in 0..side - 1 {
        for column in 0..side - 1 {
            let a = column + row * side;
            let (b, c, d) = (a + 1, a + side, a + side + 1);
            for triangle in [[a, c, b], [b, c, d]] {
                let corners = triangle.map(|index| grid[index]);
                let kept = corners
                    .iter()
                    .all(|corner| corner.is_some_and(|corner| corner.open || corner.buried));
                let shows = corners.iter().flatten().any(|corner| corner.open);
                if kept && shows {
                    for index in triangle {
                        let index = vertex(index, &mut sheet);
                        sheet.indices.push(index);
                    }
                }
            }
        }
    }
    (!sheet.indices.is_empty()).then_some(sheet)
}
