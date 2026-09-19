//! Procedural linear-slide profiles, ported from the supplied linear-slide asset.
//!
//! Local X is travel, Y is the mounting normal, and Z spans the width. Meshes
//! have a centred carriage build pose; the renderer moves carriage chunks only.

use bevy_math::{Vec2, Vec3};

use crate::LinearBearingDimensions;
use crate::hardware_mesh::triangulate;

/// A material slot using the existing aluminium or steel texture family.
#[derive(Clone, Copy, Debug)]
pub struct LinearFinish {
    /// Stable asset material name.
    pub name: &'static str,
    /// Base colour in sRGB bytes.
    pub color: [u8; 3],
    /// Perceptual roughness.
    pub roughness: f32,
    /// Metallic reflectance weight.
    pub metalness: f32,
    /// Selects aluminium textures; otherwise use steel textures.
    pub aluminium: bool,
}

/// The eight original material slots. The steel field slot is reserved by the
/// source asset; its currently ground/machined carriage faces use other slots.
pub const LINEAR_FINISHES: [LinearFinish; 8] = [
    finish("alu", [0x5a, 0x6b, 0x76], 0.28, true),
    finish("aluMachined", [0x61, 0x72, 0x7d], 0.25, true),
    finish("aluBright", [0x68, 0x79, 0x84], 0.23, true),
    finish("steel", [0x28, 0x32, 0x3c], 0.34, false),
    finish("machined", [0x36, 0x41, 0x4d], 0.26, false),
    finish("ground", [0x47, 0x53, 0x5f], 0.15, false),
    finish("bright", [0x5c, 0x6a, 0x76], 0.20, false),
    finish("dark", [0x1e, 0x26, 0x2e], 0.45, false),
];

const fn finish(
    name: &'static str,
    color: [u8; 3],
    roughness: f32,
    aluminium: bool,
) -> LinearFinish {
    LinearFinish {
        name,
        color,
        roughness,
        metalness: if aluminium { 0.95 } else { 0.94 },
        aluminium,
    }
}

/// Compound transform that owns a mesh chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearMeshOwner {
    /// Fixed way, stops, and mounting fixings.
    Rail,
    /// Moving carriage and carriage fixings.
    Carriage,
}

/// Renderer-independent indexed triangle mesh with one material and owner.
#[derive(Clone, Debug)]
pub struct LinearMeshChunk {
    /// Compound transform to apply.
    pub owner: LinearMeshOwner,
    /// Index into [`LINEAR_FINISHES`].
    pub finish: usize,
    /// Positions in rail-local metres.
    pub positions: Vec<[f32; 3]>,
    /// Unit outward normals; profile edges remain hard.
    pub normals: Vec<[f32; 3]>,
    /// Texture coordinates at the construction texture scale (1.5 m/repeat).
    pub uvs: Vec<[f32; 2]>,
    /// Counterclockwise triangle indices.
    pub indices: Vec<u32>,
}

impl LinearMeshChunk {
    fn new(owner: LinearMeshOwner, finish: usize) -> Self {
        Self {
            owner,
            finish,
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
        }
    }

    fn triangle(&mut self, points: [Vec3; 3], normal: Vec3) {
        let start = u32::try_from(self.positions.len()).expect("bounded linear mesh vertex count");
        let tangent = if normal.x.abs() > 0.9 {
            Vec3::Z
        } else {
            Vec3::X
        };
        let bitangent = normal.cross(tangent).normalize();
        for point in points {
            self.positions.push(point.to_array());
            self.normals.push(normal.to_array());
            self.uvs
                .push([point.dot(tangent) / 1.5, point.dot(bitangent) / 1.5]);
        }
        self.indices.extend([start, start + 1, start + 2]);
    }
}

/// Builds the aluminium way, steel carriage, end stops and recessed fixings.
/// Only the carriage's outer half-width is snapped; its inner profile retains
/// the original 2 mm lateral and 2.5 mm ceiling clearances.
#[must_use]
#[expect(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "bounded dimension-derived fixing counts and literal asset profiles"
)]
pub fn linear_bearing_meshes(dimensions: LinearBearingDimensions) -> Vec<LinearMeshChunk> {
    let mut chunks: Vec<_> = [LinearMeshOwner::Rail, LinearMeshOwner::Carriage]
        .into_iter()
        .flat_map(|owner| (0..8).map(move |finish| LinearMeshChunk::new(owner, finish)))
        .collect();
    let hw = dimensions.width() / 2.0;
    let inner = hw + 0.002;
    let outer = dimensions.carriage_half_width();
    let way = [
        [-hw, 0.003],
        [-hw + 0.003, 0.0],
        [hw - 0.003, 0.0],
        [hw, 0.003],
        [hw, 0.0235],
        [hw - 0.0115, 0.031],
        [hw, 0.0385],
        [hw, 0.059],
        [hw - 0.003, 0.062],
        [-hw + 0.003, 0.062],
        [-hw, 0.059],
        [-hw, 0.0385],
        [-hw + 0.0115, 0.031],
        [-hw, 0.0235],
    ];
    profile(
        &mut chunks[..8],
        &way,
        dimensions.length() - 0.030,
        2,
        &[2, 1, 2, 0, 5, 5, 0, 2, 2, 2, 0, 5, 5, 0],
        0.0,
    );
    for sign in [-1.0, 1.0] {
        let half = hw + 0.004;
        profile(
            &mut chunks[..8],
            &[[-half, 0.0], [half, 0.0], [half, 0.062], [-half, 0.062]],
            0.015,
            6,
            &[6; 4],
            sign * (dimensions.length() / 2.0 - 0.0075),
        );
    }
    let rows = ((dimensions.length() / 0.25).round() as u16 + 1).max(2);
    let lateral_rows = if dimensions.width() >= 0.15 {
        vec![-(hw - 0.020), hw - 0.020]
    } else {
        vec![0.0]
    };
    for row in 0..rows {
        let x = -dimensions.length() / 2.0
            + 0.030
            + f32::from(row) * (dimensions.length() - 0.060) / f32::from(rows - 1);
        for &z in &lateral_rows {
            fixing(&mut chunks[7], Vec3::new(x, 0.0614, z), 0.008, Vec3::Y);
        }
    }
    let car = [
        [-outer, 0.010],
        [-outer, 0.097],
        [-outer + 0.003, 0.100],
        [outer - 0.003, 0.100],
        [outer, 0.097],
        [outer, 0.010],
        [inner, 0.010],
        [inner, 0.0235],
        [inner - 0.0115, 0.031],
        [inner, 0.0385],
        [inner, 0.0645],
        [-inner, 0.0645],
        [-inner, 0.0385],
        [-inner + 0.0115, 0.031],
        [-inner, 0.0235],
        [-inner, 0.010],
    ];
    profile(
        &mut chunks[8..],
        &car,
        0.120,
        5,
        &[5, 6, 5, 6, 5, 4, 4, 5, 5, 4, 5, 4, 5, 5, 4, 4],
        0.0,
    );
    let top_rows = ((outer * 2.0 / 0.050).round() as u16).max(2);
    for row in 0..top_rows {
        let z = -outer + (outer * 2.0 / f32::from(top_rows)) * (f32::from(row) + 0.5);
        for sign in [-1.0, 1.0] {
            fixing(
                &mut chunks[15],
                Vec3::new(sign * 0.0375, 0.0994, z),
                0.005,
                Vec3::Y,
            );
        }
    }
    for sign in [-1.0, 1.0] {
        for other in [-1.0, 1.0] {
            fixing(
                &mut chunks[15],
                Vec3::new(other * 0.030, 0.042, sign * (outer - 0.0006)),
                0.005,
                Vec3::Z * sign,
            );
            fixing(
                &mut chunks[15],
                Vec3::new(sign * 0.0594, 0.042, other * (outer - 0.022)),
                0.005,
                Vec3::X * sign,
            );
        }
    }
    chunks.retain(|chunk| !chunk.indices.is_empty());
    chunks
}

fn profile(
    chunks: &mut [LinearMeshChunk],
    points: &[[f32; 2]],
    length: f32,
    cap: usize,
    edges: &[usize],
    offset: f32,
) {
    // The source rotates a profile (u, y) around Y: its local Z becomes -u.
    let points: Vec<_> = points.iter().map(|p| Vec2::new(p[0], p[1])).collect();
    let area: f32 = points
        .iter()
        .enumerate()
        .map(|(i, p)| p.perp_dot(points[(i + 1) % points.len()]))
        .sum();
    let orientation = area.signum();
    let vertex = |p: Vec2, x: f32| Vec3::new(x + offset, p.y, -p.x);
    for (i, a) in points.iter().copied().enumerate() {
        let b = points[(i + 1) % points.len()];
        let p = [
            vertex(a, -length / 2.0),
            vertex(b, -length / 2.0),
            vertex(b, length / 2.0),
            vertex(a, length / 2.0),
        ];
        let normal = (p[1] - p[0]).cross(p[2] - p[0]).normalize() * orientation;
        let mesh = &mut chunks[edges[i]];
        if orientation > 0.0 {
            mesh.triangle([p[0], p[1], p[2]], normal);
            mesh.triangle([p[0], p[2], p[3]], normal);
        } else {
            mesh.triangle([p[0], p[2], p[1]], normal);
            mesh.triangle([p[0], p[3], p[2]], normal);
        }
    }
    for triangle in triangulate(&points, orientation) {
        for sign in [-1.0, 1.0] {
            let mut p = triangle.map(|index| vertex(points[index], sign * length / 2.0));
            if orientation * sign < 0.0 {
                p.swap(1, 2);
            }
            chunks[cap].triangle(p, Vec3::X * sign);
        }
    }
}

fn fixing(mesh: &mut LinearMeshChunk, centre: Vec3, radius: f32, normal: Vec3) {
    let tangent = if normal.x.abs() > 0.9 {
        Vec3::Z
    } else {
        Vec3::X
    };
    let bitangent = normal.cross(tangent);
    let point = |step: u8| {
        let angle = f32::from(step) * core::f32::consts::TAU / 8.0;
        centre + radius * (tangent * angle.cos() + bitangent * angle.sin())
    };
    for step in 0..8 {
        mesh.triangle([centre, point(step), point(step + 1)], normal);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extreme_and_default_meshes_preserve_bounds_winding_and_finishes() {
        assert_eq!(LINEAR_FINISHES.len(), 8);
        assert_eq!(LINEAR_FINISHES.iter().filter(|f| f.aluminium).count(), 3);
        for (length, width) in [(0.25, 0.05), (1.0, 0.1), (8.0, 0.4)] {
            let dimensions = LinearBearingDimensions::new(length, width).unwrap();
            let meshes = linear_bearing_meshes(dimensions);
            assert!(meshes.iter().any(|m| m.owner == LinearMeshOwner::Carriage));
            for mesh in meshes {
                assert!(mesh.finish < 8);
                assert_eq!(mesh.positions.len(), mesh.normals.len());
                assert_eq!(mesh.positions.len(), mesh.uvs.len());
                for (&position, &normal) in mesh.positions.iter().zip(&mesh.normals) {
                    let p = Vec3::from_array(position);
                    assert!(p.is_finite());
                    assert!(p.x.abs() <= length / 2.0 + 1.0e-6);
                    assert!((-1.0e-6..=0.100_001).contains(&p.y));
                    assert!(p.z.abs() <= dimensions.carriage_half_width() + 1.0e-6);
                    assert!((Vec3::from_array(normal).length() - 1.0).abs() < 1.0e-5);
                }
                for triangle in mesh.indices.chunks_exact(3) {
                    let p = triangle
                        .iter()
                        .map(|&index| Vec3::from_array(mesh.positions[index as usize]))
                        .collect::<Vec<_>>();
                    let cross = (p[1] - p[0]).cross(p[2] - p[0]);
                    assert!(cross.length() > 1.0e-10);
                    assert!(
                        cross
                            .normalize()
                            .dot(Vec3::from_array(mesh.normals[triangle[0] as usize]))
                            > 0.999
                    );
                }
                assert!(mesh.uvs.iter().flatten().all(|value| value.is_finite()));
            }
        }
    }

    #[test]
    fn exterior_faces_and_recessed_fixings_have_outward_winding() {
        for (length, width) in [(0.25, 0.05), (1.0, 0.1), (8.0, 0.4)] {
            let dimensions = LinearBearingDimensions::new(length, width).unwrap();
            let mut checks = [0_u32; 5];
            for mesh in linear_bearing_meshes(dimensions) {
                for triangle in mesh.indices.chunks_exact(3) {
                    let points =
                        [0, 1, 2].map(|i| Vec3::from_array(mesh.positions[triangle[i] as usize]));
                    let normal = (points[1] - points[0])
                        .cross(points[2] - points[0])
                        .normalize();
                    let centre = (points[0] + points[1] + points[2]) / 3.0;
                    let plane = |axis: usize, coordinate: f32| {
                        points.iter().all(|p| (p[axis] - coordinate).abs() < 1.0e-6)
                    };
                    let expected = if mesh.finish == 7 {
                        checks[0] += 1;
                        if mesh.owner == LinearMeshOwner::Rail || plane(1, 0.0994) {
                            // The source discs are deliberately recessed 0.6 mm.
                            assert!(plane(
                                1,
                                if mesh.owner == LinearMeshOwner::Rail {
                                    0.0614
                                } else {
                                    0.0994
                                }
                            ));
                            Vec3::Y
                        } else if plane(0, centre.x.signum() * 0.0594) {
                            Vec3::X * centre.x.signum()
                        } else {
                            assert!(plane(
                                2,
                                centre.z.signum() * (dimensions.carriage_half_width() - 0.0006)
                            ));
                            Vec3::Z * centre.z.signum()
                        }
                    } else if mesh.owner == LinearMeshOwner::Carriage && plane(1, 0.1) {
                        checks[1] += 1;
                        Vec3::Y
                    } else if mesh.owner == LinearMeshOwner::Carriage
                        && plane(2, centre.z.signum() * dimensions.carriage_half_width())
                    {
                        checks[2] += 1;
                        Vec3::Z * centre.z.signum()
                    } else if mesh.owner == LinearMeshOwner::Carriage
                        && plane(0, centre.x.signum() * 0.06)
                    {
                        checks[3] += 1;
                        Vec3::X * centre.x.signum()
                    } else if mesh.owner == LinearMeshOwner::Carriage && mesh.finish == 6 {
                        checks[4] += 1;
                        // Each top chamfer slopes equally outwards and upwards.
                        Vec3::new(0.0, 1.0, centre.z.signum()).normalize()
                    } else {
                        continue;
                    };
                    assert!(
                        normal.dot(expected) > 0.9999,
                        "{:?} finish {} at {centre:?}: {normal:?} points away from {expected:?}",
                        mesh.owner,
                        mesh.finish
                    );
                }
            }
            assert!(
                checks.into_iter().all(|count| count > 0),
                "every exterior face class must be tested"
            );
        }
    }

    #[test]
    fn snapped_outer_faces_keep_original_inner_clearance() {
        for width in [0.05, 0.1, 0.4] {
            let dimensions = LinearBearingDimensions::new(1.0, width).unwrap();
            let positions: Vec<_> = linear_bearing_meshes(dimensions)
                .into_iter()
                .filter(|mesh| mesh.owner == LinearMeshOwner::Carriage)
                .flat_map(|mesh| mesh.positions)
                .collect();
            assert!(
                positions
                    .iter()
                    .any(|p| (p[2].abs() - dimensions.carriage_half_width()).abs() < 1.0e-6)
            );
            assert!(
                positions
                    .iter()
                    .any(|p| (p[2].abs() - (width / 2.0 + 0.002)).abs() < 1.0e-6
                        && (p[1] - 0.0645).abs() < 1.0e-6)
            );
        }
    }
}
