//! Treads drawn into the surfaces they are cut into.
//!
//! A part is drawn as usual first. Each treaded surface's triangles are then
//! dropped and replaced by its relief: lug tops on the surface, groove floors
//! the tread's depth below it, and walls where the two meet. A groove that runs
//! to the surface's edge ends against the neighbouring face, which stays whole.

use super::construction::{append_block_texture_coordinates, append_cylinder_texture_coordinates};
use crate::chroma;
use bevy::prelude::{Quat, Vec3};
use mechanic_core::{
    Axis, ConstructionMaterial, FaceKind, LayerFace, PartSpec, SurfaceTreads, TREAD_CELL_METERS,
    TREAD_TILE_CELLS, TREAD_TILE_METERS, TreadMask,
};
use std::f32::consts::TAU;

/// Fewest tiles around a curved surface, so a small wheel still reads round.
const MIN_TILES_AROUND: u32 = 4;

/// Slack for a drawn vertex to count as lying on a surface.
const ON_SURFACE_METERS: f32 = 1.0e-3;

/// Least alignment of a drawn triangle's normal with a surface's own.
const ON_SURFACE_ALIGNMENT: f32 = 0.5;

/// Where the part sits in the mesh being built.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PartPlacement {
    pub(crate) translation: Vec3,
    pub(crate) rotation: Quat,
}

impl PartPlacement {
    fn point(self, local: Vec3) -> Vec3 {
        self.translation + self.rotation * local
    }

    fn local(self, point: Vec3) -> Vec3 {
        self.rotation.inverse() * (point - self.translation)
    }
}

/// A treaded surface in part-local coordinates.
#[derive(Clone, Copy, Debug)]
enum Relief {
    /// A flat rectangle, its cells counted along `u` then `v`.
    Plane {
        center: Vec3,
        u: Vec3,
        v: Vec3,
        outward: Vec3,
        half: [f32; 2],
    },
    /// A wall about local Y: the outer wall facing out, or a bore facing in.
    Wall {
        radius: f32,
        half_length: f32,
        facing: f32,
    },
    /// An end cap: an annulus at `y`, facing `facing` along Y.
    Cap {
        y: f32,
        facing: f32,
        inner: f32,
        outer: f32,
    },
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "cell counts of surfaces at most 8 m across"
)]
fn cells_along(length: f32) -> u32 {
    ((length / TREAD_CELL_METERS).round() as u32).max(1)
}

/// Cells around a circle: whole tiles, so the pattern closes on itself.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "tile counts of circles at most 8 m across"
)]
fn cells_around(radius: f32) -> u32 {
    TREAD_TILE_CELLS * ((TAU * radius / TREAD_TILE_METERS).round() as u32).max(MIN_TILES_AROUND)
}

fn radial(angle: f32) -> Vec3 {
    Vec3::new(angle.cos(), 0.0, angle.sin())
}

impl Relief {
    fn of(spec: PartSpec, surface: LayerFace) -> Option<Self> {
        match (spec, surface) {
            (PartSpec::Cuboid(cuboid), LayerFace::Face(face)) => {
                let half = cuboid.size_meters() * 0.5;
                let (u, v) = match face.axis() {
                    Axis::X => (Vec3::Y, Vec3::Z),
                    Axis::Y => (Vec3::X, Vec3::Z),
                    Axis::Z => (Vec3::X, Vec3::Y),
                };
                let outward = face.axis().unit() * face.sign();
                Some(Self::Plane {
                    center: outward * half.dot(outward.abs()),
                    u,
                    v,
                    outward,
                    half: [half.dot(u), half.dot(v)],
                })
            }
            (PartSpec::Cylinder(cylinder), surface) => {
                let outer = cylinder.dimensions.outer_diameter() * 0.5;
                let inner = cylinder.dimensions.inner_diameter() * 0.5;
                let half_length = cylinder.dimensions.axial_length() * 0.5;
                match surface {
                    LayerFace::OuterWall => Some(Self::Wall {
                        radius: outer,
                        half_length,
                        facing: 1.0,
                    }),
                    LayerFace::Bore if inner > 0.0 => Some(Self::Wall {
                        radius: inner,
                        half_length,
                        facing: -1.0,
                    }),
                    LayerFace::Face(face @ (FaceKind::PositiveY | FaceKind::NegativeY)) => {
                        Some(Self::Cap {
                            y: face.sign() * half_length,
                            facing: face.sign(),
                            inner,
                            outer,
                        })
                    }
                    LayerFace::Face(_) | LayerFace::Bore => None,
                }
            }
            _ => None,
        }
    }

    /// Cells along the surface's first and second directions.
    fn cells(self) -> [u32; 2] {
        match self {
            Self::Plane { half, .. } => [cells_along(2.0 * half[0]), cells_along(2.0 * half[1])],
            Self::Wall {
                radius,
                half_length,
                ..
            } => [cells_around(radius), cells_along(2.0 * half_length)],
            Self::Cap { inner, outer, .. } => [cells_around(outer), cells_along(outer - inner)],
        }
    }

    /// Whether the first direction closes on itself.
    const fn wraps(self) -> bool {
        !matches!(self, Self::Plane { .. })
    }

    /// The point at fractional cell coordinates, `height` out along the
    /// surface's outward normal.
    #[expect(clippy::cast_precision_loss, reason = "a few thousand cells")]
    fn point(self, cells: [u32; 2], column: f32, row: f32, height: f32) -> Vec3 {
        let [columns, rows] = cells.map(|count| count as f32);
        match self {
            Self::Plane {
                center,
                u,
                v,
                outward,
                half,
            } => {
                center
                    + u * half[0] * (2.0 * column / columns - 1.0)
                    + v * half[1] * (2.0 * row / rows - 1.0)
                    + outward * height
            }
            Self::Wall {
                radius,
                half_length,
                facing,
            } => {
                radial(TAU * column / columns) * (radius + facing * height)
                    + Vec3::Y * half_length * (2.0 * row / rows - 1.0)
            }
            Self::Cap {
                y,
                facing,
                inner,
                outer,
            } => {
                radial(TAU * column / columns) * (inner + (outer - inner) * row / rows)
                    + Vec3::Y * (y + facing * height)
            }
        }
    }

    /// The surface's outward normal at a fractional column.
    #[expect(clippy::cast_precision_loss, reason = "a few thousand cells")]
    fn outward(self, cells: [u32; 2], column: f32) -> Vec3 {
        match self {
            Self::Plane { outward, .. } => outward,
            Self::Wall { facing, .. } => radial(TAU * column / cells[0] as f32) * facing,
            Self::Cap { facing, .. } => Vec3::Y * facing,
        }
    }

    /// Whether a drawn triangle, in part-local coordinates with its averaged
    /// vertex normal, lies on this surface.
    fn holds(self, triangle: [Vec3; 3], normal: Vec3) -> bool {
        match self {
            Self::Plane {
                center, outward, ..
            } => {
                normal.dot(outward) > ON_SURFACE_ALIGNMENT
                    && triangle
                        .iter()
                        .all(|&point| (point - center).dot(outward).abs() < ON_SURFACE_METERS)
            }
            Self::Wall {
                radius,
                half_length,
                facing,
            } => {
                let middle = (triangle[0] + triangle[1] + triangle[2]) / 3.0;
                let out = Vec3::new(middle.x, 0.0, middle.z).normalize_or_zero() * facing;
                normal.dot(out) > ON_SURFACE_ALIGNMENT
                    && triangle.iter().all(|point| {
                        (point.x.hypot(point.z) - radius).abs() < ON_SURFACE_METERS
                            && point.y.abs() < half_length + ON_SURFACE_METERS
                    })
            }
            Self::Cap { y, facing, .. } => {
                normal.y * facing > ON_SURFACE_ALIGNMENT
                    && triangle
                        .iter()
                        .all(|point| (point.y - y).abs() < ON_SURFACE_METERS)
            }
        }
    }

    /// A point just inside the middle of the surface, to ask which material
    /// band it belongs to.
    fn inside(self) -> Vec3 {
        let cells = self.cells();
        let [column, row] = cells.map(|count| count / 2);
        #[expect(clippy::cast_precision_loss, reason = "a few thousand cells")]
        let (column, row) = (column as f32 + 0.5, row as f32 + 0.5);
        self.point(cells, column, row, -1.0e-3)
    }
}

/// Collects the relief's triangles in the mesh's own frame.
struct Sink<'a> {
    placement: PartPlacement,
    positions: &'a mut Vec<[f32; 3]>,
    normals: &'a mut Vec<[f32; 3]>,
    indices: &'a mut Vec<u32>,
}

impl Sink<'_> {
    /// One quad facing its corner normals, whichever way its corners run.
    fn quad(&mut self, corners: [Vec3; 4], normals: [Vec3; 4]) {
        let base = u32::try_from(self.positions.len()).expect("mesh vertex count fits u32");
        for (corner, normal) in corners.into_iter().zip(normals) {
            self.positions.push(self.placement.point(corner).to_array());
            self.normals
                .push((self.placement.rotation * normal).to_array());
        }
        let facing = normals.into_iter().sum::<Vec3>();
        let turn = (corners[1] - corners[0]).cross(corners[2] - corners[0]);
        let order = if turn.dot(facing) >= 0.0 {
            [0, 1, 2, 0, 2, 3]
        } else {
            [0, 2, 1, 0, 3, 2]
        };
        self.indices.extend(order.map(|offset| base + offset));
    }

    /// One groove wall, facing `toward` the groove.
    fn wall(&mut self, corners: [Vec3; 4], toward: Vec3) {
        let normal = (corners[1] - corners[0])
            .cross(corners[3] - corners[0])
            .normalize_or_zero();
        let normal = if normal.dot(toward) < 0.0 {
            -normal
        } else {
            normal
        };
        self.quad(corners, [normal; 4]);
    }
}

/// Lug tops, groove floors, and the walls between them for one surface.
#[expect(clippy::cast_precision_loss, reason = "a few thousand cells")]
fn append_relief(relief: Relief, mask: TreadMask, depth: f32, sink: &mut Sink) {
    let cells = relief.cells();
    let height = |raised: bool| if raised { 0.0 } else { -depth };
    for row in 0..cells[1] {
        for column in 0..cells[0] {
            let level = height(mask.raised(column, row));
            let (left, right) = (column as f32, column as f32 + 1.0);
            let (near, far) = (row as f32, row as f32 + 1.0);
            sink.quad(
                [(left, near), (right, near), (right, far), (left, far)]
                    .map(|(column, row)| relief.point(cells, column, row, level)),
                [left, right, right, left].map(|column| relief.outward(cells, column)),
            );
        }
    }
    // A wall stands wherever a lug meets a groove: between neighbouring
    // columns, around the seam of a closed surface too, and between rows.
    let middle = |column: u32, row: u32| {
        relief.point(cells, column as f32 + 0.5, row as f32 + 0.5, -0.5 * depth)
    };
    let columns = if relief.wraps() {
        cells[0]
    } else {
        cells[0] - 1
    };
    for row in 0..cells[1] {
        for column in 0..columns {
            let next = (column + 1) % cells[0];
            let raised = mask.raised(column, row);
            if raised == mask.raised(next, row) {
                continue;
            }
            let edge = column as f32 + 1.0;
            let (near, far) = (row as f32, row as f32 + 1.0);
            let toward = middle(next, row) - middle(column, row);
            sink.wall(
                [
                    relief.point(cells, edge, near, 0.0),
                    relief.point(cells, edge, far, 0.0),
                    relief.point(cells, edge, far, -depth),
                    relief.point(cells, edge, near, -depth),
                ],
                if raised { toward } else { -toward },
            );
        }
    }
    for row in 0..cells[1] - 1 {
        for column in 0..cells[0] {
            let raised = mask.raised(column, row);
            if raised == mask.raised(column, row + 1) {
                continue;
            }
            let edge = row as f32 + 1.0;
            let (left, right) = (column as f32, column as f32 + 1.0);
            let toward = middle(column, row + 1) - middle(column, row);
            sink.wall(
                [
                    relief.point(cells, left, edge, 0.0),
                    relief.point(cells, right, edge, 0.0),
                    relief.point(cells, right, edge, -depth),
                    relief.point(cells, left, edge, -depth),
                ],
                if raised { toward } else { -toward },
            );
        }
    }
}

/// Replaces each treaded surface of a part, drawn into these buffers from
/// `first_index` on, with its relief. Only surfaces whose material band is
/// `material`, when one is given, gain a relief: the others belong to another
/// material's mesh.
#[expect(clippy::too_many_arguments)]
pub(crate) fn append_part_treads(
    spec: PartSpec,
    treads: SurfaceTreads,
    placement: PartPlacement,
    material: Option<ConstructionMaterial>,
    first_index: usize,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let surfaces = treads
        .iter()
        .filter_map(|(surface, tread)| Relief::of(spec, surface).map(|relief| (relief, tread)))
        .collect::<Vec<_>>();
    if surfaces.is_empty() {
        return;
    }
    let vertex = |index: u32| {
        let index = index as usize;
        (
            placement.local(Vec3::from_array(positions[index])),
            placement.rotation.inverse() * Vec3::from_array(normals[index]),
        )
    };
    let kept = indices[first_index..]
        .chunks_exact(3)
        .filter(|triangle| {
            let [(a, na), (b, nb), (c, nc)] = [0, 1, 2].map(|corner| vertex(triangle[corner]));
            let normal = (na + nb + nc).normalize_or_zero();
            !surfaces
                .iter()
                .any(|(relief, _)| relief.holds([a, b, c], normal))
        })
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    indices.truncate(first_index);
    indices.extend(kept);

    for (relief, tread) in surfaces {
        let Some((band_material, appearance)) =
            spec.band(spec.band_at_local_point(relief.inside()))
        else {
            continue;
        };
        if material.is_some_and(|wanted| wanted != band_material) {
            continue;
        }
        let first = positions.len();
        append_relief(
            relief,
            tread.pattern().mask(),
            tread.depth_meters(),
            &mut Sink {
                placement,
                positions,
                normals,
                indices,
            },
        );
        if let PartSpec::Cylinder(_) = spec {
            append_cylinder_texture_coordinates(
                placement.translation,
                placement.rotation,
                first,
                positions,
                normals,
                0.0,
                uvs,
                tangents,
            );
        } else {
            let frame_rotation = placement.rotation * spec.pose().rotation.quaternion().inverse();
            append_block_texture_coordinates(
                frame_rotation,
                placement.translation - frame_rotation * spec.pose().translation(),
                first,
                positions,
                normals,
                uvs,
                tangents,
            );
        }
        colors.extend(std::iter::repeat_n(
            chroma::encode_appearance(appearance),
            positions.len() - first,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::mesh::construction::combined_construction_mesh;
    use bevy::mesh::{Indices, Mesh, VertexAttributeValues};
    use mechanic_core::{
        BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, CylinderDimensions,
        CylinderSpec, TreadPattern, TreadSpec,
    };

    fn triangles(mesh: &Mesh) -> Vec<([Vec3; 3], Vec3)> {
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("positions")
        };
        let Some(VertexAttributeValues::Float32x3(normals)) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("normals")
        };
        let Some(Indices::U32(indices)) = mesh.indices() else {
            panic!("indices")
        };
        indices
            .chunks_exact(3)
            .map(|triangle| {
                let corner = |index: u32| Vec3::from_array(positions[index as usize]);
                let normal = triangle
                    .iter()
                    .map(|&index| Vec3::from_array(normals[index as usize]))
                    .sum::<Vec3>();
                (
                    [
                        corner(triangle[0]),
                        corner(triangle[1]),
                        corner(triangle[2]),
                    ],
                    normal.normalize_or_zero(),
                )
            })
            .collect()
    }

    fn area(triangle: [Vec3; 3]) -> f32 {
        0.5 * (triangle[1] - triangle[0])
            .cross(triangle[2] - triangle[0])
            .length()
    }

    fn tread(pattern: TreadPattern) -> TreadSpec {
        TreadSpec::new(pattern, 20).unwrap()
    }

    fn cut(
        graph: &mut ConstructionGraph,
        part: mechanic_core::PartId,
        cuts: &[(LayerFace, TreadPattern)],
    ) {
        for &(surface, pattern) in cuts {
            graph
                .apply(BuildCommand::SetTread {
                    part,
                    surface,
                    tread: Some(tread(pattern)),
                })
                .unwrap();
        }
    }

    fn spawned(outcome: BuildOutcome) -> mechanic_core::PartId {
        let BuildOutcome::Spawned(part) = outcome else {
            panic!("spawning reports the part")
        };
        part
    }

    fn treaded_block() -> ConstructionGraph {
        let mut graph = ConstructionGraph::new();
        let block = CuboidSpec::new([4, 4, 4], BuildPose::default()).unwrap();
        let block = spawned(graph.apply(BuildCommand::Spawn(block)).unwrap());
        cut(
            &mut graph,
            block,
            &[(LayerFace::Face(FaceKind::NegativeY), TreadPattern::Block)],
        );
        graph
    }

    fn treaded_wheel() -> (ConstructionGraph, CylinderSpec) {
        let mut graph = ConstructionGraph::new();
        let wheel = CylinderSpec::new(
            CylinderDimensions::new(1.0, 0.5, 0.5).unwrap(),
            BuildPose::default(),
        );
        let part = spawned(graph.apply(BuildCommand::SpawnCylinder(wheel)).unwrap());
        cut(
            &mut graph,
            part,
            &[
                (LayerFace::OuterWall, TreadPattern::Chevron),
                (LayerFace::Bore, TreadPattern::Ribbed),
                (LayerFace::Face(FaceKind::PositiveY), TreadPattern::Studded),
            ],
        );
        (graph, wheel)
    }

    #[test]
    fn lug_tops_keep_the_face_and_groove_floors_sit_a_depth_below() {
        let mesh = combined_construction_mesh(&treaded_block());
        let triangles = triangles(&mesh);
        let bottom = triangles
            .iter()
            .flat_map(|(corners, _)| corners.map(|corner| corner.y))
            .fold(f32::INFINITY, f32::min);
        let facing_down_at = |height: f32| {
            triangles
                .iter()
                .filter(|(corners, normal)| {
                    normal.y < -0.9
                        && corners
                            .iter()
                            .all(|corner| (corner.y - height).abs() < 1.0e-4)
                })
                .map(|(corners, _)| area(*corners))
                .sum::<f32>()
        };
        let lugs = TreadPattern::Block.mask().contact_ratio();
        assert!((facing_down_at(bottom) - lugs).abs() < 1.0e-3);
        assert!((facing_down_at(bottom + 0.02) - (1.0 - lugs)).abs() < 1.0e-3);
    }

    #[test]
    fn every_relief_triangle_faces_out_of_the_part() {
        let (wheel_graph, _) = treaded_wheel();
        for graph in [treaded_block(), wheel_graph] {
            for (corners, normal) in triangles(&combined_construction_mesh(&graph)) {
                let turn = (corners[1] - corners[0]).cross(corners[2] - corners[0]);
                assert!(
                    turn.dot(normal) > 0.0,
                    "{corners:?} winds against {normal:?}"
                );
            }
        }
    }

    #[test]
    fn a_wheel_tread_closes_around_its_rim_and_bore() {
        let (graph, wheel) = treaded_wheel();
        let triangles = triangles(&combined_construction_mesh(&graph));
        let outer = wheel.dimensions.outer_diameter() * 0.5;
        let inner = wheel.dimensions.inner_diameter() * 0.5;
        let at_radius = |radius: f32| {
            triangles.iter().any(|(corners, _)| {
                corners
                    .iter()
                    .all(|corner| (corner.x.hypot(corner.z) - radius).abs() < 1.0e-4)
            })
        };
        assert!(at_radius(outer));
        assert!(at_radius(outer - 0.02));
        assert!(at_radius(inner));
        assert!(at_radius(inner + 0.02));
        assert_eq!(
            Relief::of(PartSpec::Cylinder(wheel), LayerFace::OuterWall)
                .unwrap()
                .cells()[0]
                % TREAD_TILE_CELLS,
            0
        );
        // The studded cap is cut into the top only.
        let top = wheel.dimensions.axial_length() * 0.5;
        assert!(triangles.iter().any(|(corners, normal)| {
            normal.y > 0.9
                && corners
                    .iter()
                    .all(|corner| (corner.y - (top - 0.02)).abs() < 1.0e-4)
        }));
        assert!(!triangles.iter().any(|(corners, normal)| {
            normal.y < -0.9
                && corners
                    .iter()
                    .all(|corner| (corner.y + top - 0.02).abs() < 1.0e-4)
        }));
    }
}
