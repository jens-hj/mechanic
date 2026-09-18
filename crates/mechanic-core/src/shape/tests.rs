use super::{
    CellGrid, ConvexPiece, GridFace, POSITION_TICK_METERS, PartPiece, decompose, has_inverted_cell,
    undisplaced_steps,
};
use crate::POSITION_TICKS_PER_GRID_UNIT;
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
    let steps = point / POSITION_TICK_METERS;
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
                    0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
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
        cage.displace(
            IVec3::ZERO,
            corner,
            IVec3::new(0, -POSITION_TICKS_PER_GRID_UNIT, 0),
        );
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
        * POSITION_TICK_METERS.powi(3);
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
    cage.displace(
        IVec3::ZERO,
        0,
        IVec3::new(POSITION_TICKS_PER_GRID_UNIT, 0, 0),
    );
    cage.displace(
        IVec3::ZERO,
        1,
        IVec3::new(-POSITION_TICKS_PER_GRID_UNIT, 0, 0),
    );
    assert!(
        cage.inverted(),
        "two vertices driven through each other must be rejected"
    );
}
