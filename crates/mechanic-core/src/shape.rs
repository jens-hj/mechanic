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

use crate::GRID_UNIT_METERS;
use crate::geometry::{CuboidSpec, FaceKind, GridRotation};

/// Displacement steps spanning one half-grid unit (0.125 m), so one step is
/// 12.5 mm. This is the resolution every control vertex moves in.
pub const STEPS_PER_HALF_UNIT: i32 = 10;

/// Steps spanning one construction cell.
pub const STEPS_PER_CELL: i32 = 2 * STEPS_PER_HALF_UNIT;

/// Length of one displacement step, in metres.
pub const STEP_METERS: f32 = GRID_UNIT_METERS * 0.05;

/// Converts a position in integer steps to metres.
pub fn steps_to_meters(steps: IVec3) -> Vec3 {
    steps.as_vec3() * STEP_METERS
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
        Self { planes_half_units }
    }

    /// A grid from explicit plane positions.
    pub fn from_planes(planes_half_units: [Vec<i32>; 3]) -> Self {
        Self { planes_half_units }
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
    let world_dimensions = world_grid_dimensions(spec);
    CellGrid::uniform(
        spec.pose.translation_half_units() - world_dimensions,
        world_dimensions,
    )
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
    grid.corner_half_units(cell, corner) * STEPS_PER_HALF_UNIT
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
        let min = grid.corner_half_units(cell, 0);
        let max = grid.corner_half_units(cell + span - IVec3::ONE, 7);
        let min_meters = half_units_to_meters(min);
        let max_meters = half_units_to_meters(max);
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

fn half_units_to_meters(half_units: IVec3) -> Vec3 {
    steps_to_meters(half_units * STEPS_PER_HALF_UNIT)
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
#[allow(clippy::cast_precision_loss)] // At most eight coplanar hull vertices.
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
#[allow(clippy::cast_precision_loss)]
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
    let scale = STEP_METERS;
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
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::type_complexity
)] // Tests quantise geometry back to lattice steps to compare it exactly.
mod tests {
    use super::{
        CellGrid, ConvexPiece, GridFace, PartPiece, STEP_METERS, STEPS_PER_CELL, decompose,
        has_inverted_cell, undisplaced_steps,
    };
    use crate::geometry::{BuildPose, CuboidSpec, FaceKind, GridRotation};
    use bevy_math::{IVec3, Vec3};
    use std::collections::BTreeMap;

    /// A grid of cells whose corners can be displaced, standing in for a
    /// region's control cage.
    struct Cage {
        grid: CellGrid,
        offsets: BTreeMap<[i32; 3], IVec3>,
    }

    impl Cage {
        fn of_size(counts: IVec3) -> Self {
            Self {
                grid: CellGrid::uniform(IVec3::ZERO, counts),
                offsets: BTreeMap::new(),
            }
        }

        fn unit() -> Self {
            Self::of_size(IVec3::ONE)
        }

        /// Displaces the corner shared at `cell`'s `corner`, in steps.
        fn displace(&mut self, cell: IVec3, corner: usize, offset: IVec3) {
            let key = self.grid.corner_half_units(cell, corner).to_array();
            self.offsets.insert(key, offset);
        }

        fn steps(&self, cell: IVec3, corner: usize) -> IVec3 {
            let key = self.grid.corner_half_units(cell, corner).to_array();
            undisplaced_steps(&self.grid, cell, corner)
                + self.offsets.get(&key).copied().unwrap_or(IVec3::ZERO)
        }

        fn pieces(&self) -> Vec<PartPiece> {
            decompose(&self.grid, &|cell, corner| self.steps(cell, corner))
        }

        fn inverted(&self) -> bool {
            has_inverted_cell(&self.grid, &|cell, corner| self.steps(cell, corner))
        }
    }

    fn convex_pieces(pieces: &[PartPiece]) -> Vec<&ConvexPiece> {
        pieces
            .iter()
            .filter_map(|piece| match piece {
                PartPiece::Convex(convex) => Some(convex),
                PartPiece::Cuboid { .. } => None,
            })
            .collect()
    }

    fn total_volume(pieces: &[PartPiece]) -> f32 {
        pieces
            .iter()
            .map(|piece| match piece {
                PartPiece::Cuboid { half_extents, .. } => {
                    8.0 * half_extents.x * half_extents.y * half_extents.z
                }
                PartPiece::Convex(convex) => convex.volume,
            })
            .sum()
    }

    fn quantise(point: Vec3) -> [i32; 3] {
        let steps = point / STEP_METERS;
        [
            steps.x.round() as i32,
            steps.y.round() as i32,
            steps.z.round() as i32,
        ]
    }

    /// Boundary polygons on one grid face of one cell, each a sorted vertex set.
    fn boundary_polygons(piece: &ConvexPiece, wanted: GridFace) -> Vec<Vec<[i32; 3]>> {
        let mut polygons = Vec::new();
        for face in &piece.faces {
            if face.grid_face != Some(wanted) {
                continue;
            }
            let mut polygon = face
                .indices
                .iter()
                .map(|&index| quantise(piece.vertices[index as usize]))
                .collect::<Vec<_>>();
            polygon.sort_unstable();
            polygons.push(polygon);
        }
        polygons.sort_unstable();
        polygons
    }

    fn shared_face_polygons(
        pieces: &[PartPiece],
        left_cell: IVec3,
        right_cell: IVec3,
        axis: FaceKind,
    ) -> (Vec<Vec<[i32; 3]>>, Vec<Vec<[i32; 3]>>) {
        let mut left = Vec::new();
        let mut right = Vec::new();
        for piece in convex_pieces(pieces) {
            left.extend(boundary_polygons(
                piece,
                GridFace {
                    cell: left_cell,
                    face: axis,
                },
            ));
            right.extend(boundary_polygons(
                piece,
                GridFace {
                    cell: right_cell,
                    face: axis.opposite(),
                },
            ));
        }
        left.sort_unstable();
        right.sort_unstable();
        (left, right)
    }

    fn distinct_vertices(polygons: &[Vec<[i32; 3]>]) -> Vec<[i32; 3]> {
        let mut vertices = polygons.iter().flatten().copied().collect::<Vec<_>>();
        vertices.sort_unstable();
        vertices.dedup();
        vertices
    }

    fn polygon_area(polygons: &[Vec<[i32; 3]>]) -> f64 {
        polygons
            .iter()
            .map(|polygon| {
                let point = |index: usize| {
                    let [x, y, z] = polygon[index];
                    [f64::from(x), f64::from(y), f64::from(z)]
                };
                (1..polygon.len().saturating_sub(1))
                    .map(|index| {
                        let a = point(0);
                        let b = point(index);
                        let c = point(index + 1);
                        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                        let cross = [
                            u[1] * v[2] - u[2] * v[1],
                            u[2] * v[0] - u[0] * v[2],
                            u[0] * v[1] - u[1] * v[0],
                        ];
                        0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2])
                            .sqrt()
                    })
                    .sum::<f64>()
            })
            .sum()
    }

    #[test]
    fn an_unshaped_part_compiles_to_exactly_one_cuboid_piece() {
        let spec = CuboidSpec::new(
            [4, 2, 3],
            BuildPose::new(IVec3::new(1, 2, 3), GridRotation::default()),
        )
        .unwrap();
        assert_eq!(
            super::decompose_part(spec),
            vec![PartPiece::Cuboid {
                center: spec.pose.translation(),
                half_extents: spec.size_meters() * 0.5,
                rotation: spec.pose.rotation.quaternion(),
                cell_min: IVec3::ZERO,
                cell_span: IVec3::new(4, 2, 3),
            }],
            "an unshaped part must compile to the single box it always did"
        );
    }

    #[test]
    fn an_undisplaced_cage_covers_itself_with_one_box() {
        let pieces = Cage::of_size(IVec3::new(3, 2, 4)).pieces();
        assert_eq!(
            pieces.len(),
            1,
            "a cage nobody has shaped is still just a box"
        );
        assert!(matches!(pieces[0], PartPiece::Cuboid { .. }));
    }

    #[test]
    fn a_sheared_cell_fuses_to_one_piece_with_three_face_normals() {
        // Slide all four top corners along +x by the same amount. The cell stays
        // a parallelepiped, so it must fuse back into a single convex piece
        // whose separating axes cost exactly what a box costs.
        let mut cage = Cage::unit();
        for corner in [2, 3, 6, 7] {
            cage.displace(IVec3::ZERO, corner, IVec3::new(5, 0, 0));
        }
        let pieces = cage.pieces();
        let convex = convex_pieces(&pieces);
        assert_eq!(convex.len(), 1, "a parallelepiped is one convex piece");
        assert_eq!(convex[0].vertices.len(), 8);
        assert_eq!(
            convex[0].faces.len(),
            6,
            "coplanar triangles must fuse into six planar faces"
        );
        assert_eq!(
            convex[0].edge_directions.len(),
            3,
            "a parallelepiped has three distinct edge directions, like a box"
        );
        let expected = 0.25_f32.powi(3);
        assert!(
            (total_volume(&pieces) - expected).abs() < 1.0e-9,
            "shearing moves mass sideways without adding or removing any"
        );
    }

    #[test]
    fn collapsing_an_edge_culls_degenerate_tetrahedra_and_makes_a_wedge() {
        // Drive the two top corners on +z down a whole cell onto the corners
        // beneath them: the plain single-slope wedge, half a cell of material.
        let mut cage = Cage::unit();
        for corner in [6, 7] {
            cage.displace(IVec3::ZERO, corner, IVec3::new(0, -STEPS_PER_CELL, 0));
        }
        let pieces = cage.pieces();
        let convex = convex_pieces(&pieces);
        assert_eq!(convex.len(), 1, "a wedge is convex, so it is one piece");
        assert_eq!(
            convex[0].vertices.len(),
            6,
            "the collapsed corners must deduplicate to six distinct vertices"
        );
        let expected = 0.25_f32.powi(3) * 0.5;
        assert!(
            (total_volume(&pieces) - expected).abs() < 1.0e-9,
            "a wedge holds half a cell; got {}",
            total_volume(&pieces)
        );
        assert!(!cage.inverted(), "collapsing an edge is not an inversion");
    }

    #[test]
    fn a_single_displaced_corner_shapes_only_its_own_cell() {
        let mut cage = Cage::of_size(IVec3::new(2, 1, 1));
        cage.displace(IVec3::ZERO, 0, IVec3::new(0, 4, 0));
        let pieces = cage.pieces();
        let boxes = pieces
            .iter()
            .filter(|piece| matches!(piece, PartPiece::Cuboid { .. }))
            .count();
        assert_eq!(boxes, 1, "the untouched cell stays a single box");
        assert!(!convex_pieces(&pieces).is_empty());
    }

    /// A two-cell cage whose shared corner is displaced by `offset`.
    fn two_cell_cage(offset: IVec3) -> Vec<PartPiece> {
        let mut cage = Cage::of_size(IVec3::new(2, 1, 1));
        // Corner 1 of cell 0 is corner 0 of cell 1: the node they share.
        cage.displace(IVec3::ZERO, 1, offset);
        cage.pieces()
    }

    #[test]
    fn cells_sharing_an_in_plane_displaced_node_cover_the_shared_face_identically() {
        // The node moves within the shared plane, so that plane stays flat and
        // both cells may fuse freely across it.
        let pieces = two_cell_cage(IVec3::new(0, 6, 3));
        let (left, right) = shared_face_polygons(
            &pieces,
            IVec3::new(0, 0, 0),
            IVec3::new(1, 0, 0),
            FaceKind::PositiveX,
        );
        assert!(!left.is_empty(), "the shared plane must produce polygons");
        assert_eq!(
            distinct_vertices(&left),
            distinct_vertices(&right),
            "both cells must span the same corners of the shared face"
        );
        assert!(
            (polygon_area(&left) - polygon_area(&right)).abs() < 1.0e-9,
            "both cells must cover the same area of the shared face"
        );
    }

    #[test]
    fn cells_sharing_an_out_of_plane_displaced_node_cover_the_shared_face_identically() {
        // The node leaves the shared plane, so the shared quad is genuinely
        // non-planar and its diagonal is real geometry. This is the case that
        // cracks open if fusion is allowed to re-triangulate a grid face.
        let pieces = two_cell_cage(IVec3::new(4, 6, 3));
        let (left, right) = shared_face_polygons(
            &pieces,
            IVec3::new(0, 0, 0),
            IVec3::new(1, 0, 0),
            FaceKind::PositiveX,
        );
        assert!(
            left.len() >= 2,
            "a folded quad must stay split into its two Freudenthal triangles"
        );
        assert_eq!(
            left, right,
            "both cells must fold the shared face the same way, or the surface cracks"
        );
    }

    #[test]
    fn every_convex_piece_stays_within_the_declared_caps() {
        let mut cage = Cage::unit();
        let offsets = [
            [3, -2, 1],
            [-4, 5, 2],
            [1, 3, -5],
            [2, -1, 4],
            [-3, 2, 3],
            [5, 4, -2],
            [-1, -3, 5],
            [4, 1, 2],
        ];
        for (corner, offset) in offsets.into_iter().enumerate() {
            cage.displace(IVec3::ZERO, corner, IVec3::from_array(offset));
        }
        let pieces = cage.pieces();
        for piece in convex_pieces(&pieces) {
            assert!(piece.vertices.len() <= super::MAX_PIECE_VERTICES);
            assert!(piece.faces.len() <= super::MAX_PIECE_FACES);
            assert!(piece.edge_directions.len() <= super::MAX_PIECE_EDGES);
            assert!(piece.volume > 0.0, "a piece must enclose volume");
        }

        let corners: [IVec3; 8] = core::array::from_fn(|corner| cage.steps(IVec3::ZERO, corner));
        let split_volume = super::FREUDENTHAL_TETRAHEDRA
            .iter()
            .map(|tetrahedron| {
                super::signed_volume_six(
                    corners[tetrahedron[0]],
                    corners[tetrahedron[1]],
                    corners[tetrahedron[2]],
                    corners[tetrahedron[3]],
                )
                .abs()
            })
            .sum::<i64>() as f32
            / 6.0
            * STEP_METERS.powi(3);
        assert!(
            (total_volume(&pieces) - split_volume).abs() <= split_volume * 1.0e-5,
            "fusing must neither lose nor duplicate volume"
        );
    }

    #[test]
    fn a_shaped_cages_decomposition_is_a_closed_surface() {
        // Every directed edge of the whole complex must be matched by its
        // reverse. An unmatched edge is a crack.
        let mut cage = Cage::of_size(IVec3::new(2, 2, 1));
        for (index, cell) in [IVec3::new(0, 0, 0), IVec3::new(1, 1, 0)]
            .into_iter()
            .enumerate()
        {
            for corner in 0..8 {
                let offset = IVec3::new(
                    ((corner * 3 + index) % 7) as i32 - 3,
                    ((corner * 5 + index) % 7) as i32 - 3,
                    ((corner * 2 + index) % 7) as i32 - 3,
                );
                cage.displace(cell, corner, offset);
            }
        }
        let pieces = cage.pieces();

        let mut edges: std::collections::HashMap<([i32; 3], [i32; 3]), i32> =
            std::collections::HashMap::new();
        for piece in convex_pieces(&pieces) {
            for face in &piece.faces {
                for step in 0..face.indices.len() {
                    let from = quantise(piece.vertices[face.indices[step] as usize]);
                    let to = quantise(
                        piece.vertices[face.indices[(step + 1) % face.indices.len()] as usize],
                    );
                    *edges.entry((from, to)).or_default() += 1;
                }
            }
        }
        let unmatched = edges
            .iter()
            .filter(|&(&(from, to), &count)| count != edges.get(&(to, from)).copied().unwrap_or(0))
            .count();
        assert_eq!(
            unmatched, 0,
            "every directed edge must be matched by its reverse; the surface has a crack"
        );
    }

    #[test]
    fn every_reference_tetrahedron_is_positively_oriented() {
        // A negative signed volume has to mean "inverted", so an undisplaced
        // cell must not contain one.
        let corners: [IVec3; 8] = core::array::from_fn(|corner| super::CELL_CORNERS[corner] * 20);
        for tetrahedron in super::FREUDENTHAL_TETRAHEDRA {
            let volume = super::signed_volume_six(
                corners[tetrahedron[0]],
                corners[tetrahedron[1]],
                corners[tetrahedron[2]],
                corners[tetrahedron[3]],
            );
            assert!(
                volume > 0,
                "tetrahedron {tetrahedron:?} has volume {volume}"
            );
        }
    }

    #[test]
    fn swapping_two_neighbouring_corners_is_an_inversion() {
        // One vertex alone reaches its neighbour but never passes it, so a cell
        // can only be turned inside out by driving two through each other.
        let mut cage = Cage::unit();
        cage.displace(IVec3::ZERO, 0, IVec3::new(STEPS_PER_CELL, 0, 0));
        cage.displace(IVec3::ZERO, 1, IVec3::new(-STEPS_PER_CELL, 0, 0));
        assert!(
            cage.inverted(),
            "two vertices driven through each other must be rejected"
        );
    }
}
