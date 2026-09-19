//! Renderer-independent piston meshes. Local +Y is travel and the base mount
//! face is at Y = 0; a side mount's supporting face is towards local −Z.
//!
//! The mesh is built once. Extension translates each stage's chunks along Y by
//! [`crate::PistonDimensions::stage_offsets`]; nothing is rebuilt.
//!
//! Detail is built, never cut, and abutting parts interfere by a few tenths of a
//! millimetre rather than share a plane.

use crate::hardware_mesh::{self, MeshSink, lathe, triangulate};
use crate::{ConstructionMaterial, HardwareFinish, Piston, PistonDimensions, PistonMount};
use bevy_math::{Quat, Vec2, Vec3};
use core::f32::consts::{FRAC_PI_2, PI, TAU};

const STEEL: ConstructionMaterial = ConstructionMaterial::Steel;
const ALUMINIUM: ConstructionMaterial = ConstructionMaterial::Aluminium;

const BARREL: usize = 0;
const BAND: usize = 1;
const PAD: usize = 2;
const GLAND: usize = 3;
const CROWN: usize = 4;
const CROWN_BRIGHT: usize = 5;
const SEAL: usize = 6;
const FIXING: usize = 7;
const RECESS: usize = 8;
const ACCENT: usize = 9;
const SADDLE: usize = 10;
const FIRST_STAGE: usize = 11;
const CHROME: usize = 16;

/// Guide finishes on the existing construction textures, without an atlas.
///
/// Stage skins walk from ground steel to chrome across the stack, which is
/// what makes an extended telescope legible at distance. A player dye belongs
/// on `barrel` and `band` only.
pub const PISTON_FINISHES: [HardwareFinish; 17] = [
    HardwareFinish::new("barrel", STEEL, [0x28, 0x32, 0x3c], 0.34, 0.94),
    HardwareFinish::new("band", STEEL, [0x5c, 0x6a, 0x76], 0.20, 0.94),
    HardwareFinish::new("pad", STEEL, [0x47, 0x53, 0x5f], 0.15, 0.94),
    HardwareFinish::new("gland", STEEL, [0x36, 0x41, 0x4d], 0.26, 0.94),
    HardwareFinish::new("crown", ALUMINIUM, [0x5a, 0x6b, 0x76], 0.28, 0.95),
    HardwareFinish::new("crownBright", ALUMINIUM, [0x68, 0x79, 0x84], 0.23, 0.95),
    HardwareFinish::new(
        "seal",
        ConstructionMaterial::Rubber,
        [0x19, 0x1d, 0x21],
        0.95,
        0.0,
    ),
    HardwareFinish::new("fixing", STEEL, [0x1e, 0x26, 0x2e], 0.45, 0.94),
    HardwareFinish::new("recess", STEEL, [0x16, 0x1d, 0x24], 0.72, 0.55),
    HardwareFinish::new("accent", STEEL, [0x4d, 0x9e, 0xa8], 0.32, 0.70),
    HardwareFinish::new("saddle", ALUMINIUM, [0x5a, 0x6b, 0x76], 0.28, 0.95),
    HardwareFinish::new("stage1", STEEL, [0x47, 0x53, 0x5f], 0.150, 0.95),
    HardwareFinish::new("stage2", STEEL, [0x4f, 0x5b, 0x67], 0.138, 0.95),
    HardwareFinish::new("stage3", STEEL, [0x57, 0x64, 0x6f], 0.126, 0.95),
    HardwareFinish::new("stage4", STEEL, [0x5e, 0x6c, 0x78], 0.114, 0.95),
    HardwareFinish::new("stage5", STEEL, [0x66, 0x75, 0x80], 0.102, 0.95),
    HardwareFinish::new("chrome", STEEL, [0x6e, 0x7d, 0x88], 0.09, 0.97),
];

/// Radial segments of every full-section lathed part.
const SEGMENTS: u16 = 24;
/// Full-section end band on the body, in metres.
const END_BAND: f32 = 0.032;
/// How far the body field sits under the bands, in metres.
const FIELD_SINK: f32 = 0.0015;
/// Collar at a mouth, closing the annulus from above, in metres.
const GLAND_HEIGHT: f32 = 0.020;
/// Root stop ring, which seats on the parent's gland crown at full draw, in metres.
const STOP: f32 = 0.012;
/// Shadow gap above a stop ring, in metres.
const SEAM: f32 = 0.003;
/// A mount field sits this far inside its rim, in metres.
const RELIEF: f32 = 0.0008;
/// Flat-to-flat gap between saddle lobes, in metres.
const SADDLE_GAP: f32 = 0.052;
/// Saddle bore radius, sized for the body field, in metres.
const SADDLE_BORE: f32 = 0.1240;
/// Strap bridging two lobes, in metres.
const SADDLE_WEB: f32 = 0.0025;
/// Outer corner chamfer of a lobe, in metres.
const SADDLE_CHAMFER: f32 = 0.008;

/// Rigid group that carries a mesh chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PistonMeshOwner {
    /// Barrel, body gland and, on a side mount, the saddle brackets.
    Body,
    /// One telescoping stage, counted from one; the last carries the head crown.
    Stage(u8),
}

/// Indexed, counterclockwise triangle mesh with one finish and owner.
#[derive(Clone, Debug)]
pub struct PistonMeshChunk {
    /// Rigid group to move this chunk with.
    pub owner: PistonMeshOwner,
    /// Index into [`PISTON_FINISHES`].
    pub finish: usize,
    /// Positions in piston-local metres at the collapsed pose.
    pub positions: Vec<[f32; 3]>,
    /// Unit outward normals.
    pub normals: Vec<[f32; 3]>,
    /// Texture coordinates (1.5 m per repeat).
    pub uvs: Vec<[f32; 2]>,
    /// Triangle indices.
    pub indices: Vec<u32>,
}

impl MeshSink for PistonMeshChunk {
    fn vertex_count(&self) -> u32 {
        u32::try_from(self.positions.len()).expect("bounded piston mesh vertex count")
    }

    fn vertex(&mut self, position: Vec3, normal: Vec3, uv: [f32; 2]) {
        self.positions.push(position.to_array());
        self.normals.push(normal.to_array());
        self.uvs.push(uv);
    }

    fn triangles(&mut self, indices: &[u32]) {
        self.indices.extend_from_slice(indices);
    }
}

impl PistonMeshChunk {
    const fn new(owner: PistonMeshOwner, finish: usize) -> Self {
        Self {
            owner,
            finish,
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// A flat triangle, wound to face along `normal`.
    fn triangle(&mut self, mut points: [Vec3; 3], normal: Vec3) {
        if (points[1] - points[0])
            .cross(points[2] - points[0])
            .dot(normal)
            < 0.0
        {
            points.swap(1, 2);
        }
        let start = self.vertex_count();
        let tangent = if normal.x.abs() > 0.9 {
            Vec3::Z
        } else {
            Vec3::X
        };
        let bitangent = normal.cross(tangent).normalize();
        for point in points {
            self.vertex(
                point,
                normal,
                [point.dot(tangent) / 1.5, point.dot(bitangent) / 1.5],
            );
        }
        self.triangles(&[start, start + 1, start + 2]);
    }

    fn quad(&mut self, points: [Vec3; 4], normal: Vec3) {
        self.triangle([points[0], points[1], points[2]], normal);
        self.triangle([points[0], points[2], points[3]], normal);
    }

    fn cuboid(&mut self, size: Vec3, centre: Vec3) {
        for axis in 0..3 {
            for sign in [-1.0, 1.0] {
                let mut normal = Vec3::ZERO;
                normal[axis] = sign;
                let mut u = Vec3::ZERO;
                u[(axis + 1) % 3] = size[(axis + 1) % 3] / 2.0;
                let mut v = Vec3::ZERO;
                v[(axis + 2) % 3] = size[(axis + 2) % 3] / 2.0;
                let face = centre + normal * size[axis] / 2.0;
                self.quad(
                    [face - u - v, face + u - v, face + u + v, face - u + v],
                    normal,
                );
            }
        }
    }

    /// An annular ring standing on `y`; a zero inner radius makes it solid.
    fn ring(&mut self, radii: [f32; 2], height: f32, y: f32) {
        hardware_mesh::ring(self, radii, height, y, SEGMENTS, false, Vec3::ZERO);
    }

    /// A small solid cylinder of any axis, centred on `centre`.
    fn disc(&mut self, radius: f32, thickness: f32, segments: u16, axis: Vec3, centre: Vec3) {
        let mut disc = Self::new(self.owner, self.finish);
        hardware_mesh::ring(
            &mut disc,
            [radius, 0.0],
            thickness,
            -thickness / 2.0,
            segments,
            false,
            Vec3::ZERO,
        );
        let rotation = Quat::from_rotation_arc(Vec3::Y, axis);
        let start = self.vertex_count();
        for ((position, normal), uv) in disc.positions.iter().zip(&disc.normals).zip(&disc.uvs) {
            self.vertex(
                rotation * Vec3::from_array(*position) + centre,
                rotation * Vec3::from_array(*normal),
                *uv,
            );
        }
        self.indices
            .extend(disc.indices.iter().map(|index| start + index));
    }
}

/// The chunks of one rigid group, one per finish in use.
struct Group {
    owner: PistonMeshOwner,
    chunks: Vec<PistonMeshChunk>,
}

impl Group {
    const fn new(owner: PistonMeshOwner) -> Self {
        Self {
            owner,
            chunks: Vec::new(),
        }
    }

    fn finish(&mut self, finish: usize) -> &mut PistonMeshChunk {
        let index = self
            .chunks
            .iter()
            .position(|chunk| chunk.finish == finish)
            .unwrap_or_else(|| {
                self.chunks.push(PistonMeshChunk::new(self.owner, finish));
                self.chunks.len() - 1
            });
        &mut self.chunks[index]
    }

    /// The collar at a mouth. It belongs to the part whose mouth it is, and its
    /// crown is flush with that part's top face, because the crown is what the
    /// child's stop ring seats on at full draw.
    fn gland(&mut self, outer: f32, child: f32, top: f32) {
        let h = GLAND_HEIGHT;
        let ro = outer - PistonDimensions::WALL;
        let rc = child + PistonDimensions::CLEARANCE;
        let y = top - h;
        lathe(
            self.finish(GLAND),
            &[
                [rc + 0.0068, 0.0],
                [ro, 0.0],
                [ro, h * 0.34],
                [ro - 0.004, h * 0.46],
                [ro - 0.004, h * 0.84],
                [ro, h * 0.90],
                [ro, h - 0.004],
                [rc + 0.0068, h - 0.004],
            ],
            SEGMENTS,
            Vec3::Y * y,
            false,
        );
        self.finish(BAND)
            .ring([ro, rc + 0.0068], 0.004, top - 0.004);
        self.finish(SEAL).ring([rc + 0.007, rc], h, y);
    }
}

fn body(dimensions: PistonDimensions) -> Group {
    let mut group = Group::new(PistonMeshOwner::Body);
    let length = dimensions.closed();
    let radius = PistonDimensions::SECTION / 2.0;
    let bore = radius - PistonDimensions::WALL;
    let field = length - 2.0 * END_BAND;

    // Two full-section end bands and a field sunk under them.
    group.finish(BAND).ring([radius, bore], END_BAND, 0.0);
    group
        .finish(BARREL)
        .ring([radius - FIELD_SINK, bore], field, END_BAND);
    group
        .finish(BAND)
        .ring([radius, bore], END_BAND, length - END_BAND);

    // Skin pads and their tapped lattice.
    let pad = field - 0.060;
    let rows = (pad / 0.05).floor();
    let first_row = END_BAND + 0.030 + (pad - (rows - 1.0) * 0.05) / 2.0;
    for side in [1.0, -1.0] {
        group.finish(PAD).cuboid(
            Vec3::new(0.070, pad, 0.008),
            Vec3::new(0.0, END_BAND + 0.030 + pad / 2.0, side * (radius - 0.007)),
        );
        let mut y = first_row;
        while y < END_BAND + 0.030 + pad {
            for x in [-0.0175, 0.0175] {
                group.finish(RECESS).disc(
                    0.0052,
                    0.0014,
                    8,
                    Vec3::Z,
                    Vec3::new(x, y, side * (radius - 0.0032)),
                );
            }
            y += 0.05;
        }
    }

    // Band bolt circles, standing on the field so no head is on a mount plane.
    for y in [END_BAND + 0.012, length - END_BAND - 0.012] {
        for step in 0..8_u8 {
            let angle = f32::from(step) / 8.0 * TAU + PI / 8.0;
            let outward = Vec3::new(angle.cos(), 0.0, angle.sin());
            for (finish, disc_radius, thickness, sink) in [
                (BAND, 0.0085, 0.0012, 0.0021),
                (FIXING, 0.0062, 0.0014, 0.0014),
            ] {
                group.finish(finish).disc(
                    disc_radius,
                    thickness,
                    10,
                    outward,
                    outward * (radius - sink) + Vec3::Y * y,
                );
            }
        }
    }

    // Feed and return, sunk in the field. One accent: the line a player plumbs.
    // The pack stands these half a millimetre proud of the section; they sit
    // just inside it here so a side-mounted piston stays within its block.
    for (side, finish) in [(1.0, ACCENT), (-1.0, RECESS)] {
        let y = END_BAND + 0.070;
        group.finish(GLAND).disc(
            0.021,
            0.004,
            16,
            Vec3::X,
            Vec3::new(side * (radius - 0.0035), y, 0.0),
        );
        group.finish(finish).disc(
            0.012,
            0.004,
            14,
            Vec3::X,
            Vec3::new(side * (radius - 0.0022), y, 0.0),
        );
    }

    group.gland(radius, dimensions.section(1) / 2.0, length);
    group
}

fn stage(dimensions: PistonDimensions, stage: u8) -> Group {
    let mut group = Group::new(PistonMeshOwner::Stage(stage));
    let length = dimensions.closed();
    let outer = dimensions.section(stage) / 2.0;
    let last = stage == dimensions.stages();
    let skin = if last {
        CHROME
    } else {
        FIRST_STAGE + usize::from(stage - 1)
    };
    if last {
        head(&mut group, skin, outer, length);
    } else {
        group
            .finish(skin)
            .ring([outer, outer - PistonDimensions::WALL], length, 0.0);
        group.gland(outer, dimensions.section(stage + 1) / 2.0, length);
    }

    // Root stop ring: fills the clearance annulus out to the parent's bore and
    // seats on the parent's gland crown at full draw.
    let parent_bore = dimensions.section(stage - 1) / 2.0 - PistonDimensions::WALL - 0.0004;
    group.finish(BAND).ring([parent_bore, outer], STOP, 0.0);
    group.finish(RECESS).ring([parent_bore, outer], SEAM, STOP);
    // Witness ring, and the key every stage carries because a cylinder turns in its bore.
    group
        .finish(BAND)
        .ring([outer + 0.0006, outer - 0.005], 0.007, 0.024);
    let key = length - GLAND_HEIGHT - 0.034;
    group.finish(GLAND).cuboid(
        Vec3::new(0.011, key, 0.005),
        Vec3::new(0.0, 0.034 + key / 2.0, outer - 0.0018),
    );
    group
}

/// The solid last stage and its crown: rim on the true plane, field relieved
/// and tapped on the lattice. Rim and field interfere rather than meet.
fn head(group: &mut Group, skin: usize, outer: f32, length: f32) {
    let crown = PistonDimensions::CROWN;
    let chamfer = 0.004;
    group.finish(skin).ring([outer, 0.0], length - crown, 0.0);
    lathe(
        group.finish(CROWN),
        &[
            [outer, 0.0],
            [outer, crown - chamfer],
            [outer - chamfer, crown],
            [outer - 0.0135, crown - chamfer],
            [outer - 0.0135, 0.0],
        ],
        SEGMENTS,
        Vec3::Y * (length - crown),
        false,
    );
    let field = crown + 0.002 - RELIEF;
    group
        .finish(CROWN_BRIGHT)
        .ring([outer - 0.0128, 0.0], field, length - RELIEF - field);
    for step in 0..10_u8 {
        let angle = f32::from(step) / 10.0 * TAU + PI / 10.0;
        let centre = Vec3::new(angle.cos(), 0.0, angle.sin()) * (outer - 0.024);
        group.finish(BAND).disc(
            0.0085,
            0.003,
            10,
            Vec3::Y,
            centre + Vec3::Y * (length - RELIEF - 0.0013),
        );
        group.finish(FIXING).disc(
            0.0062,
            0.0034,
            10,
            Vec3::Y,
            centre + Vec3::Y * (length - RELIEF - 0.0013),
        );
    }
    let pitch = PistonDimensions::ATTACHMENT_PITCH * 2.0;
    let count = (outer * 1.4 / pitch).floor();
    let start = -(count - 1.0) * pitch / 2.0;
    let mut x = start;
    while x < -start + pitch / 2.0 {
        let mut z = start;
        while z < -start + pitch / 2.0 {
            if x.hypot(z) <= outer - 0.044 {
                group.finish(RECESS).disc(
                    0.0052,
                    0.0028,
                    8,
                    Vec3::Y,
                    Vec3::new(x, length - RELIEF - 0.0011, z),
                );
            }
            z += pitch;
        }
        x += pitch;
    }
}

/// A saddle bracket: four corner lobes living in the corners the round section
/// wastes, so a bracketed piston is still exactly one block square.
fn saddle(group: &mut Group, centre: f32) {
    let radius = PistonDimensions::SECTION / 2.0;
    let length = PistonDimensions::SADDLE_LENGTH;
    let start = (SADDLE_GAP / SADDLE_BORE).clamp(0.0, 0.95).asin();
    let end = FRAC_PI_2 - start;
    for sz in [1.0, -1.0] {
        for sx in [1.0, -1.0] {
            let arc = |angle: f32| {
                Vec2::new(
                    angle.cos() * SADDLE_BORE * sx,
                    angle.sin() * SADDLE_BORE * sz,
                )
            };
            let mut outline = (0..=10_u8)
                .map(|step| arc(start + (end - start) * f32::from(step) / 10.0))
                .collect::<Vec<_>>();
            outline.extend([
                Vec2::new(arc(end).x, sz * radius),
                Vec2::new(sx * (radius - SADDLE_CHAMFER), sz * radius),
                Vec2::new(sx * radius, sz * (radius - SADDLE_CHAMFER)),
                Vec2::new(sx * radius, arc(start).y),
            ]);
            extrude(group.finish(SADDLE), &outline, centre, length);
        }
        // The bridging web, sunk half a millimetre: the face a block lands on
        // is the lobes, never this.
        group.finish(GLAND).cuboid(
            Vec3::new(0.100, length, SADDLE_WEB),
            Vec3::new(0.0, centre, sz * (radius - SADDLE_WEB / 2.0 - 0.0005)),
        );
        for sx in [1.0, -1.0] {
            for dy in [-length * 0.26, length * 0.26] {
                let at = |depth: f32| {
                    Vec3::new(sx * (radius - 0.0115), centre + dy, sz * (radius - depth))
                };
                group
                    .finish(BAND)
                    .disc(0.0085, 0.004, 10, Vec3::Z, at(0.004));
                group
                    .finish(FIXING)
                    .disc(0.0062, 0.005, 10, Vec3::Z, at(0.0055));
            }
        }
    }
    for sx in [1.0, -1.0] {
        group.finish(RECESS).cuboid(
            Vec3::new(0.0016, length, 0.0075),
            Vec3::new(sx * (radius - 0.0008), centre, 0.0),
        );
    }
}

/// Extrudes an outline in the local XZ plane along Y, centred on `centre`.
fn extrude(chunk: &mut PistonMeshChunk, outline: &[Vec2], centre: f32, length: f32) {
    let orientation = outline
        .iter()
        .enumerate()
        .map(|(index, point)| point.perp_dot(outline[(index + 1) % outline.len()]))
        .sum::<f32>()
        .signum();
    let lift = |point: Vec2, y: f32| Vec3::new(point.x, y, point.y);
    let (bottom, top) = (centre - length / 2.0, centre + length / 2.0);
    for (index, &a) in outline.iter().enumerate() {
        let b = outline[(index + 1) % outline.len()];
        let edge = b - a;
        let normal = Vec3::new(edge.y, 0.0, -edge.x).normalize() * orientation;
        chunk.quad(
            [lift(a, bottom), lift(b, bottom), lift(b, top), lift(a, top)],
            normal,
        );
    }
    for triangle in triangulate(outline, orientation) {
        for (y, normal) in [(bottom, Vec3::NEG_Y), (top, Vec3::Y)] {
            chunk.triangle(triangle.map(|index| lift(outline[index], y)), normal);
        }
    }
}

/// Builds the body, every stage, and a side mount's two saddle brackets.
#[must_use]
pub fn piston_meshes(piston: Piston) -> Vec<PistonMeshChunk> {
    let dimensions = piston.dimensions;
    let mut body = body(dimensions);
    if matches!(piston.mount, PistonMount::Side { .. }) {
        for centre in dimensions.saddle_positions() {
            saddle(&mut body, centre);
        }
    }
    let mut chunks = body.chunks;
    for index in 1..=dimensions.stages() {
        chunks.extend(stage(dimensions, index).chunks);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piston(blocks: u8, stages: u8, side: bool) -> Piston {
        Piston {
            dimensions: PistonDimensions::new(blocks, stages).unwrap(),
            mount: if side {
                PistonMount::Side {
                    mount_normal: Vec3::Y,
                }
            } else {
                PistonMount::End
            },
        }
    }

    #[test]
    fn every_chunk_stays_inside_the_collapsed_envelope_with_outward_winding() {
        for (blocks, stages, side) in [(2, 1, false), (2, 4, true), (8, 6, true)] {
            let piston = piston(blocks, stages, side);
            let half = PistonDimensions::SECTION / 2.0 + 1.0e-5;
            for chunk in piston_meshes(piston) {
                assert!(chunk.finish < PISTON_FINISHES.len());
                assert_eq!(chunk.positions.len(), chunk.normals.len());
                assert_eq!(chunk.positions.len(), chunk.uvs.len());
                for (&position, &normal) in chunk.positions.iter().zip(&chunk.normals) {
                    let p = Vec3::from_array(position);
                    assert!(p.is_finite());
                    assert!(
                        p.x.abs() <= half && p.z.abs() <= half,
                        "{:?} {} leaves the section at {p}",
                        chunk.owner,
                        PISTON_FINISHES[chunk.finish].name
                    );
                    assert!((-1.0e-5..=piston.dimensions.closed() + 1.0e-5).contains(&p.y));
                    assert!((Vec3::from_array(normal).length() - 1.0).abs() < 1.0e-4);
                }
                for triangle in chunk.indices.chunks_exact(3) {
                    let [a, b, c] = [0, 1, 2]
                        .map(|corner| Vec3::from_array(chunk.positions[triangle[corner] as usize]));
                    let face = (b - a).cross(c - a);
                    if face.length() < 1.0e-10 {
                        continue;
                    }
                    let normal = Vec3::from_array(chunk.normals[triangle[0] as usize]);
                    assert!(
                        face.normalize().dot(normal) > 0.5,
                        "{:?} {} winds inward",
                        chunk.owner,
                        PISTON_FINISHES[chunk.finish].name
                    );
                }
            }
        }
    }

    #[test]
    fn each_stage_is_its_own_rigid_group_and_only_a_side_mount_has_saddles() {
        let chunks = piston_meshes(piston(2, 4, false));
        for index in 1..=4 {
            assert!(
                chunks
                    .iter()
                    .any(|chunk| chunk.owner == PistonMeshOwner::Stage(index))
            );
        }
        assert!(!chunks.iter().any(|chunk| chunk.finish == SADDLE));
        assert!(
            chunks
                .iter()
                .any(|chunk| chunk.owner == PistonMeshOwner::Stage(4) && chunk.finish == CHROME)
        );
        let saddled = piston_meshes(piston(2, 4, true));
        assert!(
            saddled
                .iter()
                .any(|chunk| chunk.owner == PistonMeshOwner::Body && chunk.finish == SADDLE)
        );
    }

    #[test]
    fn the_head_crown_rim_is_the_top_of_the_collapsed_piston() {
        let piston = piston(3, 2, false);
        let top = piston_meshes(piston)
            .iter()
            .filter(|chunk| chunk.finish == CROWN)
            .flat_map(|chunk| chunk.positions.iter().map(|position| position[1]))
            .fold(f32::NEG_INFINITY, f32::max);
        assert!((top - piston.dimensions.closed()).abs() < 1.0e-6);
    }
}
