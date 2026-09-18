//! Decomposition of a construction part into convex pieces.
//!
//! This is the single source of a shaped part's geometry. The compiler builds
//! colliders and mass properties from it, the app builds its render mesh from
//! it, and the editor raycasts against it. Because all three consume the same
//! pieces, the hitbox, the visible surface, and the cursor cannot drift apart.
//!
//! A part occupies a range of construction cells. Cells whose eight corners all
//! rest on the grid are *plain* and are covered by as few axis-aligned boxes as
//! possible, so an unshaped part still compiles to exactly the one box it always
//! did. Cells with a displaced corner are *shaped*: they are split into
//! tetrahedra by the Freudenthal scheme and then greedily fused back into the
//! largest convex pieces that exactly reproduce their union.
//!
//! # Why Freudenthal
//!
//! Watertightness between neighbouring cells is a property of the split, not of
//! a tolerance. Freudenthal labels a cell's corners by grid parity, so `v0` is
//! always the cell's minimum corner and `v7` its maximum. A face shared by two
//! cells is the first cell's maximum face and the second's minimum face; the
//! first triangulates it along the diagonal leaving `v7`, the second along the
//! diagonal arriving at `v0`, and those are the two ends of the same diagonal.
//! Both sides therefore emit identical triangles. Fusing pieces only ever
//! removes faces interior to a cell, so it cannot disturb this.

use bevy_math::{IVec3, Quat, Vec3};

use crate::geometry::{CuboidSpec, FaceKind, GridRotation};
use crate::{POSITION_TICK_METERS, POSITION_TICKS_PER_HALF_GRID_UNIT};

/// Converts a position in integer steps to metres.
pub fn steps_to_meters(steps: IVec3) -> Vec3 {
    steps.as_vec3() * POSITION_TICK_METERS
}

/// Largest number of vertices one convex piece can carry. A piece is fused only
/// from tetrahedra of a single cell, so its vertices are a subset of that cell's
/// eight corners.
pub const MAX_PIECE_VERTICES: usize = 8;

/// Largest number of distinct face planes one convex piece can carry.
pub const MAX_PIECE_FACES: usize = 12;

/// Largest number of distinct edge directions one convex piece can carry.
pub const MAX_PIECE_EDGES: usize = 18;

/// Corner offsets of one cell, indexed so bit 0 is x, bit 1 is y, and bit 2 is
/// z. This matches the vertex numbering the collision kernels already use.
const CELL_CORNERS: [IVec3; 8] = [
    IVec3::new(0, 0, 0),
    IVec3::new(1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(1, 1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(1, 0, 1),
    IVec3::new(0, 1, 1),
    IVec3::new(1, 1, 1),
];

/// The six Freudenthal tetrahedra of one cell, each running from the cell's
/// minimum corner to its maximum along one permutation of the axes.
///
/// Each is listed so an undisplaced cell gives it a positive signed volume,
/// which is what lets a negative volume mean "this cell has been turned inside
/// out". The vertex *set* is what fixes the face triangulation, so ordering
/// them this way costs the watertightness argument nothing.
const FREUDENTHAL_TETRAHEDRA: [[usize; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 5, 1, 7],
    [0, 3, 2, 7],
    [0, 2, 6, 7],
    [0, 4, 5, 7],
    [0, 6, 4, 7],
];

/// Which grid face of which cell a piece face lies on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridFace {
    /// Cell index within the part, in world cell coordinates.
    pub cell: IVec3,
    /// Which of the cell's six faces this is.
    pub face: FaceKind,
}

/// One planar face of a convex piece, wound counter-clockwise seen from
/// outside.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvexFace {
    /// Outward unit normal.
    pub normal: Vec3,
    /// Plane offset, so the plane is `dot(normal, x) == offset`.
    pub offset: f32,
    /// Indices into the piece's vertex list.
    pub indices: Vec<u32>,
    /// The grid face this lies on, when it is on the cell boundary rather than
    /// interior to it. Placement keeps working on grid coordinates through this.
    pub grid_face: Option<GridFace>,
}

/// One convex piece of a decomposed part, in build space.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvexPiece {
    /// Distinct corner positions in metres.
    pub vertices: Vec<Vec3>,
    /// Planar faces, with coplanar triangles already fused.
    pub faces: Vec<ConvexFace>,
    /// Distinct edge directions, deduplicated so antiparallel counts once.
    pub edge_directions: Vec<Vec3>,
    /// Volume centroid in metres.
    pub centroid: Vec3,
    /// Volume in cubic metres.
    pub volume: f32,
}

/// One piece of a decomposed part.
#[derive(Clone, Debug, PartialEq)]
pub enum PartPiece {
    /// An unshaped run of cells, kept as a box so the fast collision path and
    /// the six-quad mesh path are preserved exactly.
    Cuboid {
        /// Centre in build space.
        center: Vec3,
        /// Half extents before rotation.
        half_extents: Vec3,
        /// Orientation.
        rotation: Quat,
        /// Lowest cell this box covers, in part cell coordinates.
        cell_min: IVec3,
        /// Cell counts this box covers.
        cell_span: IVec3,
    },
    /// A shaped cell, or part of one.
    Convex(ConvexPiece),
}

/// A grid of cells, given by where its dividing planes sit on each axis.
///
/// A plain part's planes are evenly spaced one cell apart. A shape region's are
/// its control cage, which subdivision can space unevenly. Both decompose
/// through the same code because both are just a grid of hexahedra.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellGrid {
    /// Plane positions in half-grid units, ascending, one list per axis. Each
    /// list holds `cells + 1` entries.
    planes_half_units: [Vec<i32>; 3],
    /// Translation not representable by the half-grid planes.
    offset_steps: IVec3,
}

impl CellGrid {
    /// A grid whose planes sit one cell apart, starting at `min_half_units`.
    ///
    /// # Panics
    ///
    /// Never in practice: counts come from validated grid dimensions.
    pub fn uniform(min_half_units: IVec3, counts: IVec3) -> Self {
        let planes_half_units = core::array::from_fn(|axis| {
            (0..=counts[axis])
                .map(|step| min_half_units[axis] + step * 2)
                .collect()
        });
        Self {
            planes_half_units,
            offset_steps: IVec3::ZERO,
        }
    }

    /// A grid from explicit plane positions.
    pub fn from_planes(planes_half_units: [Vec<i32>; 3]) -> Self {
        Self {
            planes_half_units,
            offset_steps: IVec3::ZERO,
        }
    }

    /// A grid whose cell-relative planes begin at an exact shape-step origin.
    pub(crate) fn from_cell_planes(origin_steps: IVec3, planes_cells: &[Vec<i32>; 3]) -> Self {
        let origin_half_units =
            origin_steps.div_euclid(IVec3::splat(POSITION_TICKS_PER_HALF_GRID_UNIT));
        let planes_half_units = core::array::from_fn(|axis| {
            planes_cells[axis]
                .iter()
                .map(|cells| origin_half_units[axis] + cells * 2)
                .collect()
        });
        Self {
            planes_half_units,
            offset_steps: origin_steps.rem_euclid(IVec3::splat(POSITION_TICKS_PER_HALF_GRID_UNIT)),
        }
    }

    /// Cell counts along each axis.
    pub fn counts(&self) -> IVec3 {
        IVec3::new(
            i32::try_from(self.planes_half_units[0].len()).unwrap_or(1) - 1,
            i32::try_from(self.planes_half_units[1].len()).unwrap_or(1) - 1,
            i32::try_from(self.planes_half_units[2].len()).unwrap_or(1) - 1,
        )
    }

    /// Plane positions along one axis.
    pub fn planes(&self, axis: usize) -> &[i32] {
        &self.planes_half_units[axis]
    }

    /// Half-grid coordinate of one cell corner. `cell` indexes the cell and
    /// `corner` selects one of its eight corners.
    /// # Panics
    ///
    /// Never in practice: cell indices come from this grid's own counts.
    pub fn corner_half_units(&self, cell: IVec3, corner: usize) -> IVec3 {
        let offset = CELL_CORNERS[corner];
        let plane = |axis: usize, index: i32| {
            let index = usize::try_from(index).expect("cell indices are inside the grid");
            self.planes_half_units[axis][index]
        };
        IVec3::new(
            plane(0, cell.x + offset.x),
            plane(1, cell.y + offset.y),
            plane(2, cell.z + offset.z),
        )
    }

    /// Exact shape-step coordinate of one cell corner, including a precision offset.
    pub fn corner_steps(&self, cell: IVec3, corner: usize) -> IVec3 {
        self.corner_half_units(cell, corner) * POSITION_TICKS_PER_HALF_GRID_UNIT + self.offset_steps
    }

    /// Whether a cell index lies inside the grid.
    pub fn contains(&self, cell: IVec3) -> bool {
        cell.cmpge(IVec3::ZERO).all() && cell.cmplt(self.counts()).all()
    }
}

/// Cell extent covered by a cuboid.
///
/// A cuboid centred on integer half-grid units with integer grid dimensions has
/// corners at `centre ± dimensions`, so every corner is an integer half-grid
/// coordinate.
pub fn part_cells(spec: CuboidSpec) -> CellGrid {
    // Material layers sit off the grid; only the core has cells.
    let spec = spec.without_layers();
    let world_dimensions = world_grid_dimensions(spec);
    let minimum_ticks = spec.pose.translation_position_ticks()
        - world_dimensions * POSITION_TICKS_PER_HALF_GRID_UNIT;
    let mut minimum_half_units = IVec3::ZERO;
    let mut offset_steps = IVec3::ZERO;
    for axis in 0..3 {
        minimum_half_units[axis] =
            minimum_ticks[axis].div_euclid(POSITION_TICKS_PER_HALF_GRID_UNIT);
        offset_steps[axis] = minimum_ticks[axis].rem_euclid(POSITION_TICKS_PER_HALF_GRID_UNIT);
    }
    let mut grid = CellGrid::uniform(minimum_half_units, world_dimensions);
    grid.offset_steps = offset_steps;
    grid
}

/// Cuboid side lengths in grid units, permuted into world axes.
fn world_grid_dimensions(spec: CuboidSpec) -> IVec3 {
    let rotation = spec.pose.rotation;
    let mut world = IVec3::ZERO;
    for (local_axis, direction) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
        let world_axis = cardinal_axis(rotation, direction);
        world[world_axis] = i32::from(spec.dimensions[local_axis].units());
    }
    world
}

fn cardinal_axis(rotation: GridRotation, direction: Vec3) -> usize {
    let rotated = rotation.quaternion() * direction;
    let absolute = rotated.abs();
    if absolute.x >= absolute.y && absolute.x >= absolute.z {
        0
    } else if absolute.y >= absolute.z {
        1
    } else {
        2
    }
}

/// Whether any cell of a grid has been turned inside out.
///
/// A control vertex can be dragged through the far side of its own cell, which
/// produces self-intersecting geometry with no meaningful volume. Collapsing a
/// cell flat is fine — that is how a wedge is made — so only strictly negative
/// tetrahedra count as inverted.
pub fn has_inverted_cell(grid: &CellGrid, corner_steps: &dyn Fn(IVec3, usize) -> IVec3) -> bool {
    cell_indices(grid.counts()).any(|cell| {
        let corners: [IVec3; 8] = core::array::from_fn(|corner| corner_steps(cell, corner));
        FREUDENTHAL_TETRAHEDRA.iter().any(|tetrahedron| {
            signed_volume_six(
                corners[tetrahedron[0]],
                corners[tetrahedron[1]],
                corners[tetrahedron[2]],
                corners[tetrahedron[3]],
            ) < 0
        })
    })
}

/// Where a cell corner sits when nothing has moved it.
pub fn undisplaced_steps(grid: &CellGrid, cell: IVec3, corner: usize) -> IVec3 {
    grid.corner_steps(cell, corner)
}

fn cell_is_shaped(
    grid: &CellGrid,
    cell: IVec3,
    corner_steps: &dyn Fn(IVec3, usize) -> IVec3,
) -> bool {
    (0..8).any(|corner| corner_steps(cell, corner) != undisplaced_steps(grid, cell, corner))
}

fn cell_indices(counts: IVec3) -> impl Iterator<Item = IVec3> {
    (0..counts.z).flat_map(move |z| {
        (0..counts.y).flat_map(move |y| (0..counts.x).map(move |x| IVec3::new(x, y, z)))
    })
}

/// Splits a grid of cells into the convex pieces that represent it exactly.
///
/// `corner_steps` gives each cell corner's position in lattice steps. A grid
/// whose corners all sit undisplaced yields exactly one [`PartPiece::Cuboid`],
/// which is what keeps an unshaped part compiling to the single box it always
/// did.
///
/// # Panics
///
/// Never in practice: the cell counts come from validated grid dimensions, and
/// a piece is fused only from one cell's eight corners.
pub fn decompose(grid: &CellGrid, corner_steps: &dyn Fn(IVec3, usize) -> IVec3) -> Vec<PartPiece> {
    let counts = grid.counts();
    let shaped = cell_indices(counts)
        .filter(|&cell| cell_is_shaped(grid, cell, corner_steps))
        .collect::<Vec<_>>();

    let mut pieces = Vec::new();
    let cell_count = usize::try_from(counts.x * counts.y * counts.z)
        .expect("validated grid dimensions give a non-negative cell count");
    let mut plain = vec![true; cell_count];
    for &cell in &shaped {
        plain[cell_slot(counts, cell)] = false;
    }
    append_box_cover(grid, &plain, &mut pieces);
    for &cell in &shaped {
        append_shaped_cell(grid, cell, corner_steps, &mut pieces);
    }
    pieces
}

/// Splits an unshaped part, which is always the one box it always was.
pub fn decompose_part(spec: CuboidSpec) -> Vec<PartPiece> {
    vec![PartPiece::Cuboid {
        center: spec.pose.translation(),
        half_extents: spec.size_meters() * 0.5,
        rotation: spec.pose.rotation.quaternion(),
        cell_min: IVec3::ZERO,
        cell_span: part_cells(spec).counts(),
    }]
}

fn cell_slot(counts: IVec3, cell: IVec3) -> usize {
    usize::try_from((cell.z * counts.y + cell.y) * counts.x + cell.x)
        .expect("cell indices are inside the part")
}

/// Covers every plain cell with as few axis-aligned boxes as possible by
/// greedily growing each run along x, then y, then z.
fn append_box_cover(grid: &CellGrid, plain: &[bool], pieces: &mut Vec<PartPiece>) {
    let counts = grid.counts();
    let mut used = vec![false; plain.len()];
    for cell in cell_indices(counts) {
        let slot = cell_slot(counts, cell);
        if !plain[slot] || used[slot] {
            continue;
        }
        let mut span = IVec3::ONE;
        while cell.x + span.x < counts.x
            && run_available(
                counts,
                plain,
                &used,
                cell,
                IVec3::new(span.x, 0, 0),
                span.with_x(1),
            )
        {
            span.x += 1;
        }
        while cell.y + span.y < counts.y
            && run_available(
                counts,
                plain,
                &used,
                cell,
                IVec3::new(0, span.y, 0),
                span.with_y(1),
            )
        {
            span.y += 1;
        }
        while cell.z + span.z < counts.z
            && run_available(
                counts,
                plain,
                &used,
                cell,
                IVec3::new(0, 0, span.z),
                span.with_z(1),
            )
        {
            span.z += 1;
        }
        for member in cell_indices(span) {
            used[cell_slot(counts, cell + member)] = true;
        }
        let min_meters = steps_to_meters(grid.corner_steps(cell, 0));
        let max_meters = steps_to_meters(grid.corner_steps(cell + span - IVec3::ONE, 7));
        pieces.push(PartPiece::Cuboid {
            center: (min_meters + max_meters) * 0.5,
            half_extents: (max_meters - min_meters) * 0.5,
            rotation: Quat::IDENTITY,
            cell_min: cell,
            cell_span: span,
        });
    }
}

/// Whether the slab at `offset` with size `span` is entirely plain and unused.
fn run_available(
    counts: IVec3,
    plain: &[bool],
    used: &[bool],
    origin: IVec3,
    offset: IVec3,
    span: IVec3,
) -> bool {
    cell_indices(span).all(|member| {
        let cell = origin + offset + member;
        let slot = cell_slot(counts, cell);
        plain[slot] && !used[slot]
    })
}

/// Splits one shaped cell into tetrahedra, then fuses them back into the
/// largest convex pieces that reproduce their union exactly.
fn append_shaped_cell(
    grid: &CellGrid,
    cell: IVec3,
    corner_steps: &dyn Fn(IVec3, usize) -> IVec3,
    pieces: &mut Vec<PartPiece>,
) {
    let _ = grid;
    let corners: [IVec3; 8] = core::array::from_fn(|corner| corner_steps(cell, corner));

    let mut parts: Vec<Vec<usize>> = Vec::new();
    for tetrahedron in FREUDENTHAL_TETRAHEDRA {
        if signed_volume_six(
            corners[tetrahedron[0]],
            corners[tetrahedron[1]],
            corners[tetrahedron[2]],
            corners[tetrahedron[3]],
        ) == 0
        {
            // A merged corner collapses this tetrahedron. The test is on
            // integer coordinates, so it is exact rather than tolerance-based.
            continue;
        }
        parts.push(tetrahedron.to_vec());
    }

    fuse_convex(&corners, &mut parts);

    for corner_indices in parts {
        if let Some(piece) = build_piece(&corners, &corner_indices, cell) {
            pieces.push(PartPiece::Convex(piece));
        }
    }
}

/// Greedily fuses pieces whose union is convex.
///
/// Two interior-disjoint pieces have a convex union exactly when the volume of
/// the convex hull of their combined corners equals the sum of their volumes.
/// Every quantity here is an exact integer, so no tolerance decides the result.
///
/// Convexity alone is not enough. Fusing rebuilds the piece from its convex
/// hull, and a hull re-triangulates a non-planar boundary quad along whichever
/// diagonal keeps *this* piece convex — which is the opposite diagonal from the
/// one the neighbouring cell picks, splitting the shared surface open. Freudenthal
/// already fixes a consistent diagonal on every grid face, so a fusion is allowed
/// only where it cannot disturb one: see [`preserves_grid_faces`].
fn fuse_convex(corners: &[IVec3; 8], parts: &mut Vec<Vec<usize>>) {
    loop {
        let mut fused = None;
        'search: for first in 0..parts.len() {
            for second in (first + 1)..parts.len() {
                let mut combined = parts[first].clone();
                combined.extend_from_slice(&parts[second]);
                combined.sort_unstable();
                combined.dedup();
                if !preserves_grid_faces(corners, &combined) {
                    continue;
                }
                let volume = hull_volume_six(corners, &combined);
                if volume != 0
                    && volume
                        == hull_volume_six(corners, &parts[first])
                            + hull_volume_six(corners, &parts[second])
                {
                    fused = Some((first, second, combined));
                    break 'search;
                }
            }
        }
        let Some((first, second, combined)) = fused else {
            return;
        };
        parts[first] = combined;
        parts.remove(second);
    }
}

/// Whether fusing this set of corners leaves every grid face triangulated the
/// way Freudenthal triangulated it.
///
/// A grid face is only at risk when the fused piece spans all four of its
/// corners, because that is when the hull gets to choose a diagonal. If those
/// four corners are coplanar both diagonals describe the same surface and the
/// choice cannot matter. If they are not, the fold direction is real geometry
/// and only Freudenthal's diagonal agrees with the neighbouring cell, so the
/// tetrahedra must stay apart.
fn preserves_grid_faces(corners: &[IVec3; 8], indices: &[usize]) -> bool {
    for axis in 0..3 {
        for side in 0..2 {
            let quad: Vec<usize> = (0..8)
                .filter(|corner| (corner >> axis) & 1 == side)
                .collect();
            if !quad.iter().all(|corner| indices.contains(corner)) {
                continue;
            }
            let points = distinct_points(corners, &quad);
            if points.len() < 4 {
                continue;
            }
            let normal = (points[1] - points[0])
                .as_i64vec3()
                .cross((points[2] - points[0]).as_i64vec3());
            let offset = normal.dot(points[0].as_i64vec3());
            if points
                .iter()
                .any(|point| normal.dot(point.as_i64vec3()) != offset)
            {
                return false;
            }
        }
    }
    true
}

/// Six times the signed volume of a tetrahedron, exact in integer coordinates.
fn signed_volume_six(a: IVec3, b: IVec3, c: IVec3, d: IVec3) -> i64 {
    let ba = (b - a).as_i64vec3();
    let ca = (c - a).as_i64vec3();
    let da = (d - a).as_i64vec3();
    ba.dot(ca.cross(da))
}

/// Six times the volume of the convex hull of the given corners.
fn hull_volume_six(corners: &[IVec3; 8], indices: &[usize]) -> i64 {
    let points = distinct_points(corners, indices);
    if points.len() < 4 {
        return 0;
    }
    let origin = points[0];
    hull_faces(&points)
        .into_iter()
        .map(|face| {
            let polygon = &face.polygon;
            (1..polygon.len() - 1)
                .map(|index| {
                    signed_volume_six(
                        origin,
                        points[polygon[0]],
                        points[polygon[index]],
                        points[polygon[index + 1]],
                    )
                })
                .sum::<i64>()
        })
        .sum::<i64>()
        .abs()
}

fn distinct_points(corners: &[IVec3; 8], indices: &[usize]) -> Vec<IVec3> {
    let mut points: Vec<IVec3> = Vec::with_capacity(indices.len());
    for &index in indices {
        let point = corners[index];
        if !points.contains(&point) {
            points.push(point);
        }
    }
    points
}

/// One planar face of an integer convex hull.
struct HullFace {
    /// Outward normal, reduced by its greatest common divisor.
    normal: IVec3,
    /// Plane offset for the reduced normal.
    offset: i64,
    /// Indices into the point list, wound counter-clockwise seen from outside.
    polygon: Vec<usize>,
}

/// Faces of the convex hull of up to eight integer points.
///
/// The point count is tiny, so every triple is tested directly against every
/// other point. This is exact, needs no tolerance, and cannot produce the
/// degenerate output an incremental hull can.
fn hull_faces(points: &[IVec3]) -> Vec<HullFace> {
    let mut faces: Vec<HullFace> = Vec::new();
    for first in 0..points.len() {
        for second in (first + 1)..points.len() {
            for third in (second + 1)..points.len() {
                let normal = (points[second] - points[first])
                    .as_i64vec3()
                    .cross((points[third] - points[first]).as_i64vec3());
                if normal == bevy_math::I64Vec3::ZERO {
                    continue;
                }
                let offset = normal.dot(points[first].as_i64vec3());
                let mut positive = false;
                let mut negative = false;
                for point in points {
                    let side = normal.dot(point.as_i64vec3()) - offset;
                    if side > 0 {
                        positive = true;
                    } else if side < 0 {
                        negative = true;
                    }
                }
                if positive && negative {
                    continue;
                }
                // Every other point is on one side, so this plane supports the
                // hull. Orient the normal outward.
                let (normal, offset) = if positive {
                    (-normal, -offset)
                } else {
                    (normal, offset)
                };
                let (normal, offset) = reduce_plane(normal, offset);
                if faces
                    .iter()
                    .any(|face| face.normal == normal && face.offset == offset)
                {
                    continue;
                }
                let on_plane = (0..points.len())
                    .filter(|&index| normal.as_i64vec3().dot(points[index].as_i64vec3()) == offset)
                    .collect::<Vec<_>>();
                let polygon = order_polygon(points, &on_plane, normal);
                faces.push(HullFace {
                    normal,
                    offset,
                    polygon,
                });
            }
        }
    }
    faces
}

/// Divides a plane through by the greatest common divisor of its normal so
/// coplanar faces compare equal exactly.
fn reduce_plane(normal: bevy_math::I64Vec3, offset: i64) -> (IVec3, i64) {
    let divisor = gcd(gcd(normal.x.abs(), normal.y.abs()), normal.z.abs()).max(1);
    let reduced = normal / divisor;
    let component =
        |value: i64| i32::try_from(value).expect("a plane normal reduced by its gcd stays small");
    (
        IVec3::new(
            component(reduced.x),
            component(reduced.y),
            component(reduced.z),
        ),
        offset / divisor,
    )
}

const fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

/// Orders coplanar points counter-clockwise seen from along the outward normal.
///
/// The ordering is only used to wind a polygon whose vertices are at least one
/// lattice step apart, so float angles are ample.
#[expect(
    clippy::cast_precision_loss,
    reason = "at most eight coplanar hull vertices"
)]
fn order_polygon(points: &[IVec3], on_plane: &[usize], normal: IVec3) -> Vec<usize> {
    if on_plane.len() < 3 {
        return on_plane.to_vec();
    }
    let normal = normal.as_vec3().normalize();
    let center = on_plane
        .iter()
        .map(|&index| points[index].as_vec3())
        .sum::<Vec3>()
        / on_plane.len() as f32;
    let reference = (points[on_plane[0]].as_vec3() - center).normalize();
    let tangent = normal.cross(reference);
    let mut ordered = on_plane.to_vec();
    ordered.sort_by(|&left, &right| {
        let angle = |index: usize| {
            let offset = points[index].as_vec3() - center;
            f32::atan2(offset.dot(tangent), offset.dot(reference))
        };
        angle(left)
            .partial_cmp(&angle(right))
            .expect("hull vertices are finite")
    });
    ordered
}

/// Builds the exported piece for one fused set of cell corners.
///
/// Volumes and centroids are accumulated as exact integers and converted once
/// at the end; the magnitudes involved are a handful of lattice steps cubed.
#[expect(clippy::cast_precision_loss)]
fn build_piece(corners: &[IVec3; 8], indices: &[usize], cell: IVec3) -> Option<ConvexPiece> {
    let points = distinct_points(corners, indices);
    if points.len() < 4 {
        return None;
    }
    let faces = hull_faces(&points);
    if faces.is_empty() {
        return None;
    }

    let vertices = points.iter().map(|&point| steps_to_meters(point)).collect();
    let mut exported = Vec::with_capacity(faces.len());
    let mut edges: Vec<IVec3> = Vec::new();
    let mut volume_six = 0_i64;
    let mut centroid_accumulator = Vec3::ZERO;
    let origin = points[0];

    for face in &faces {
        for window in 0..face.polygon.len() {
            let start = points[face.polygon[window]];
            let end = points[face.polygon[(window + 1) % face.polygon.len()]];
            let direction = reduce_direction(end - start);
            if !edges.contains(&direction) {
                edges.push(direction);
            }
        }
        for index in 1..face.polygon.len() - 1 {
            let a = points[face.polygon[0]];
            let b = points[face.polygon[index]];
            let c = points[face.polygon[index + 1]];
            let tetrahedron = signed_volume_six(origin, a, b, c);
            volume_six += tetrahedron;
            centroid_accumulator += (origin + a + b + c).as_vec3() * (tetrahedron as f32);
        }

        let normal = face.normal.as_vec3().normalize();
        exported.push(ConvexFace {
            normal,
            offset: normal.dot(steps_to_meters(points[face.polygon[0]])),
            indices: face
                .polygon
                .iter()
                .map(|&index| u32::try_from(index).expect("a piece has at most eight vertices"))
                .collect(),
            grid_face: grid_face_of(corners, &points, &face.polygon, cell),
        });
    }

    if volume_six == 0 {
        return None;
    }
    let scale = POSITION_TICK_METERS;
    let volume = (volume_six.abs() as f32) / 6.0 * scale * scale * scale;
    let centroid = centroid_accumulator / (4.0 * volume_six as f32) * scale;

    Some(ConvexPiece {
        vertices,
        faces: exported,
        edge_directions: edges
            .into_iter()
            .map(|edge| edge.as_vec3().normalize())
            .collect(),
        centroid,
        volume,
    })
}

/// Reduces an edge vector to a canonical direction so antiparallel edges
/// deduplicate to one separating axis.
fn reduce_direction(edge: IVec3) -> IVec3 {
    let divisor = gcd(
        gcd(i64::from(edge.x).abs(), i64::from(edge.y).abs()),
        i64::from(edge.z).abs(),
    )
    .max(1);
    let divisor = i32::try_from(divisor).expect("an edge gcd divides lattice-step components");
    let reduced = edge / divisor;
    let leading = if reduced.x != 0 {
        reduced.x
    } else if reduced.y != 0 {
        reduced.y
    } else {
        reduced.z
    };
    if leading < 0 { -reduced } else { reduced }
}

/// Which grid face this piece face lies on, when it is on the cell boundary.
///
/// A face is on the cell's positive face along an axis when every one of its
/// vertices came from a cell corner with that axis bit set, and on the negative
/// face when every bit is clear. Interior faces match neither.
fn grid_face_of(
    corners: &[IVec3; 8],
    points: &[IVec3],
    polygon: &[usize],
    cell: IVec3,
) -> Option<GridFace> {
    let mut shared_set = 0b111_u8;
    let mut shared_clear = 0b111_u8;
    for &index in polygon {
        let point = points[index];
        let mut set = 0_u8;
        let mut clear = 0_u8;
        for (corner, &position) in corners.iter().enumerate() {
            if position != point {
                continue;
            }
            for axis in 0..3 {
                if corner & (1 << axis) == 0 {
                    clear |= 1 << axis;
                } else {
                    set |= 1 << axis;
                }
            }
        }
        shared_set &= set;
        shared_clear &= clear;
    }
    for axis in 0..3 {
        let bit = 1 << axis;
        if shared_set & bit != 0 {
            return Some(GridFace {
                cell,
                face: positive_face(axis),
            });
        }
        if shared_clear & bit != 0 {
            return Some(GridFace {
                cell,
                face: positive_face(axis).opposite(),
            });
        }
    }
    None
}

/// Cell offset across one grid face.
pub const fn face_neighbour_offset(face: FaceKind) -> IVec3 {
    match face {
        FaceKind::PositiveX => IVec3::new(1, 0, 0),
        FaceKind::NegativeX => IVec3::new(-1, 0, 0),
        FaceKind::PositiveY => IVec3::new(0, 1, 0),
        FaceKind::NegativeY => IVec3::new(0, -1, 0),
        FaceKind::PositiveZ => IVec3::new(0, 0, 1),
        FaceKind::NegativeZ => IVec3::new(0, 0, -1),
    }
}

const fn positive_face(axis: usize) -> FaceKind {
    match axis {
        0 => FaceKind::PositiveX,
        1 => FaceKind::PositiveY,
        _ => FaceKind::PositiveZ,
    }
}

#[cfg(test)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::type_complexity,
    reason = "tests quantise geometry back to lattice steps to compare it exactly"
)]
mod tests;
