//! Renderer-independent suspension meshes. Local +Y runs from source to opposite
//! mounting face. Compression is in metres, measured from maximum extension.
//! Topology and UVs are cached; deformation updates positions and normals only.

use crate::hardware_mesh::{self, MeshSink, lathe};
use crate::{ConstructionMaterial, HardwareFinish, ShockBodyEnd, SpringSpec, SuspensionSpec};
use bevy_math::Vec3;
use core::f32::consts::{PI, TAU};

const fn finish(
    name: &'static str,
    material: ConstructionMaterial,
    color: [u8; 3],
    roughness: f32,
    metalness: f32,
) -> HardwareFinish {
    HardwareFinish::new(name, material, color, roughness, metalness)
}
/// Original guide finish contrast, using existing textures without an atlas.
pub const SUSPENSION_FINISHES: [HardwareFinish; 12] = [
    finish(
        "plate",
        ConstructionMaterial::Aluminium,
        [0x5a, 0x6b, 0x76],
        0.28,
        0.95,
    ),
    finish(
        "plateBright",
        ConstructionMaterial::Aluminium,
        [0x68, 0x79, 0x84],
        0.23,
        0.95,
    ),
    finish(
        "spring",
        ConstructionMaterial::Steel,
        [0x2c, 0x37, 0x42],
        0.38,
        0.92,
    ),
    finish(
        "springBright",
        ConstructionMaterial::Steel,
        [0x4a, 0x58, 0x66],
        0.22,
        0.94,
    ),
    finish(
        "body",
        ConstructionMaterial::Steel,
        [0x28, 0x32, 0x3c],
        0.34,
        0.94,
    ),
    finish(
        "bodyMachined",
        ConstructionMaterial::Steel,
        [0x36, 0x41, 0x4d],
        0.26,
        0.94,
    ),
    finish(
        "bodyBright",
        ConstructionMaterial::Steel,
        [0x5c, 0x6a, 0x76],
        0.20,
        0.94,
    ),
    finish(
        "shaft",
        ConstructionMaterial::Steel,
        [0x6e, 0x7d, 0x88],
        0.09,
        0.97,
    ),
    finish(
        "rubber",
        ConstructionMaterial::Rubber,
        [0x19, 0x1d, 0x21],
        0.95,
        0.0,
    ),
    finish(
        "fixing",
        ConstructionMaterial::Steel,
        [0x1e, 0x26, 0x2e],
        0.45,
        0.94,
    ),
    finish(
        "fixingBright",
        ConstructionMaterial::Steel,
        [0x6e, 0x7e, 0x89],
        0.22,
        0.95,
    ),
    finish(
        "accent",
        ConstructionMaterial::Steel,
        [0x4d, 0x9e, 0xa8],
        0.32,
        0.70,
    ),
];
/// Component or rigid mount owning a chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuspensionMeshOwner {
    /// Source plate and its rigid shock hardware.
    Source,
    /// Opposite plate and its rigid shock hardware.
    Opposite,
    /// Deforming spring wire and its end faces.
    Spring,
    /// Rubber and its shaft-mounted clamp.
    BumpStop,
}
#[derive(Clone, Debug)]
enum Deformation {
    Rigid { opposite: bool },
    Coil { samples: Vec<[f32; 3]> },
    Rubber { opposite: bool },
}
/// Indexed, counterclockwise triangle mesh, in source-local SI coordinates.
#[derive(Clone, Debug)]
pub struct SuspensionMeshChunk {
    /// Component ownership for picking and removal.
    pub owner: SuspensionMeshOwner,
    /// Index into [`SUSPENSION_FINISHES`].
    pub finish: usize,
    /// Positions in metres.
    pub positions: Vec<[f32; 3]>,
    /// Unit outward normals.
    pub normals: Vec<[f32; 3]>,
    /// Texture coordinates (1.5 m per repeat).
    pub uvs: Vec<[f32; 2]>,
    /// Triangle indices.
    pub indices: Vec<u32>,
    spec: SuspensionSpec,
    rest_positions: Vec<[f32; 3]>,
    rest_normals: Vec<[f32; 3]>,
    deformation: Deformation,
}
impl MeshSink for SuspensionMeshChunk {
    fn vertex_count(&self) -> u32 {
        u32::try_from(self.positions.len()).expect("bounded mesh")
    }
    fn vertex(&mut self, p: Vec3, n: Vec3, uv: [f32; 2]) {
        self.positions.push(p.to_array());
        self.normals.push(n.to_array());
        self.uvs.push(uv);
    }
    fn triangles(&mut self, indices: &[u32]) {
        self.indices.extend_from_slice(indices);
    }
}
impl SuspensionMeshChunk {
    fn new(
        spec: SuspensionSpec,
        owner: SuspensionMeshOwner,
        finish: usize,
        deformation: Deformation,
    ) -> Self {
        Self {
            owner,
            finish,
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            spec,
            rest_positions: Vec::new(),
            rest_normals: Vec::new(),
            deformation,
        }
    }
    fn freeze(&mut self) {
        self.rest_positions.clone_from(&self.positions);
        self.rest_normals.clone_from(&self.normals);
    }
    /// Updates cached vertices without changing topology, allocations or UVs.
    /// Nonfinite compression uses full extension; finite values clamp to travel.
    #[expect(
        clippy::missing_panics_doc,
        reason = "private cached state guarantees matching component specs"
    )]
    pub fn update_deformation(&mut self, compression: f32) {
        let compression = if compression.is_finite() {
            compression.clamp(0.0, self.spec.compression_limit().0)
        } else {
            0.0
        };
        let length = self.spec.extended_length() - compression;
        match &self.deformation {
            Deformation::Rigid { opposite } => {
                let offset = if *opposite { -compression } else { 0.0 };
                for (p, rest) in self.positions.iter_mut().zip(&self.rest_positions) {
                    *p = *rest;
                    p[1] += offset;
                }
            }
            Deformation::Coil { samples } => {
                let spring = self.spec.spring().expect("coil host");
                let plate = self.spec.plates().thickness;
                for ((p, n), sample) in self
                    .positions
                    .iter_mut()
                    .zip(&mut self.normals)
                    .zip(samples)
                {
                    let (position, normal) =
                        coil_vertex(spring, length - 2.0 * plate, plate, *sample);
                    *p = position.to_array();
                    *n = normal.to_array();
                }
            }
            Deformation::Rubber { opposite } => {
                let stop = self.spec.bump_stop().expect("stop host");
                let crush = (compression - self.spec.bump_contact().expect("stop contact"))
                    .clamp(0.0, stop.max_crush());
                let scale = 1.0 - crush / stop.length();
                let anchor = if *opposite {
                    self.spec.extended_length() - self.spec.plates().thickness
                } else {
                    self.spec.plates().thickness
                };
                for (((p, n), rest), normal) in self
                    .positions
                    .iter_mut()
                    .zip(&mut self.normals)
                    .zip(&self.rest_positions)
                    .zip(&self.rest_normals)
                {
                    *p = *rest;
                    p[1] = anchor + (rest[1] - anchor) * scale
                        - if *opposite { compression } else { 0.0 };
                    *n = Vec3::new(normal[0], normal[1] / scale, normal[2])
                        .normalize()
                        .to_array();
                }
            }
        }
    }
}

fn ring(
    chunk: &mut SuspensionMeshChunk,
    outer: f32,
    inner: f32,
    height: f32,
    y: f32,
    flip: bool,
    offset: Vec3,
) {
    hardware_mesh::ring(chunk, [outer, inner], height, y, 28, flip, offset);
}
fn rigid(spec: SuspensionSpec, opposite: bool, finish: usize) -> SuspensionMeshChunk {
    SuspensionMeshChunk::new(
        spec,
        if opposite {
            SuspensionMeshOwner::Opposite
        } else {
            SuspensionMeshOwner::Source
        },
        finish,
        Deformation::Rigid { opposite },
    )
}
fn plate_meshes(spec: SuspensionSpec, opposite: bool, chunks: &mut Vec<SuspensionMeshChunk>) {
    let plate = spec.plates();
    let r = plate.diameter / 2.0;
    let t = plate.thickness;
    let c = (t * 0.32).min(0.006);
    let offset = Vec3::Y
        * if opposite {
            spec.extended_length()
        } else {
            0.0
        };
    let mut plate_mesh = rigid(spec, opposite, 0);
    lathe(
        &mut plate_mesh,
        &[
            [0.0, 0.0],
            [r - c, 0.0],
            [r, c],
            [r, t - c],
            [r - c, t],
            [0.0, t],
        ],
        44,
        offset,
        opposite,
    );
    chunks.push(plate_mesh);
    let mut bright = rigid(spec, opposite, 1);
    if let Some(spring) = spec.spring() {
        ring(
            &mut bright,
            spring.od() / 2.0 + spring.wire() * 0.15,
            (spring.id() / 2.0 - spring.wire() * 0.15).max(0.004),
            spring.wire() * 0.34,
            t - 0.0004,
            opposite,
            offset,
        );
    }
    chunks.push(bright);
    // Guide fasteners face upward on both plates: inward on the source,
    // outward on the opposite plate. Keep their caps 0.1/0.3 mm proud.
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let bolts = ((PI * (plate.diameter - 0.036) / 0.055 / 2.0).round() as u16 * 2).max(6);
    for (finish, radius, height, proud) in [(10, 0.011, 0.0016, 0.0001), (9, 0.008, 0.0018, 0.0003)]
    {
        let mut mesh = rigid(spec, opposite, finish);
        for i in 0..bolts {
            let angle = TAU * f32::from(i) / f32::from(bolts) + PI / f32::from(bolts);
            let br = r - 0.014_f32.max(t * 0.9);
            let bolt_offset = offset + Vec3::new(angle.cos() * br, 0.0, angle.sin() * br);
            ring(
                &mut mesh,
                radius,
                0.0,
                height,
                if opposite { -proud } else { t + proud - height },
                opposite,
                bolt_offset,
            );
        }
        chunks.push(mesh);
    }
}
fn coil_vertex(spring: SpringSpec, gap: f32, plate: f32, sample: [f32; 3]) -> (Vec3, Vec3) {
    let [t, angle, cap] = sample;
    let turns = f32::from(spring.coils() + 2);
    let flat = 0.5 / turns;
    let f = ((t - flat) / (1.0 - 2.0 * flat)).clamp(0.0, 1.0);
    let theta = TAU * turns * t;
    let (s, c) = theta.sin_cos();
    let radius = spring.mean_diameter() / 2.0;
    let slope = if t > flat && t < 1.0 - flat {
        (gap - spring.wire()) / (1.0 - 2.0 * flat)
    } else {
        0.0
    };
    let tangent = Vec3::new(-s * radius * TAU * turns, slope, c * radius * TAU * turns).normalize();
    let radial = Vec3::new(c, 0.0, s);
    let binormal = tangent.cross(radial).normalize();
    let normal = radial * angle.cos() + binormal * angle.sin();
    let center = Vec3::new(
        c * radius,
        plate + spring.wire() / 2.0 + f * (gap - spring.wire()),
        s * radius,
    );
    (
        center + normal * spring.wire() / 2.0,
        if cap == 0.0 { normal } else { tangent * cap },
    )
}
fn coil_meshes(spec: SuspensionSpec, chunks: &mut Vec<SuspensionMeshChunk>) {
    let spring = spec.spring().expect("spring host");
    let steps = u16::from(spring.coils() + 2) * 28;
    let mut mesh = SuspensionMeshChunk::new(
        spec,
        SuspensionMeshOwner::Spring,
        2,
        Deformation::Coil {
            samples: Vec::new(),
        },
    );
    let mut samples = Vec::new();
    for i in 0..=steps {
        for j in 0..=8_u16 {
            let t = f32::from(i) / f32::from(steps);
            let angle = TAU * f32::from(j) / 8.0;
            samples.push([t, angle, 0.0]);
            let (p, n) = coil_vertex(
                spring,
                spec.extended_length() - 2.0 * spec.plates().thickness,
                spec.plates().thickness,
                [t, angle, 0.0],
            );
            mesh.vertex(
                p,
                n,
                [
                    t * spring.mean_diameter() * PI * f32::from(spring.coils() + 2) / 1.5,
                    angle * spring.wire() / 3.0,
                ],
            );
        }
    }
    for i in 0..u32::from(steps) {
        for j in 0..8 {
            let a = i * 9 + j;
            mesh.indices
                .extend_from_slice(&[a, a + 1, a + 9, a + 1, a + 10, a + 9]);
        }
    }
    mesh.deformation = Deformation::Coil { samples };
    chunks.push(mesh);
    let mut caps = SuspensionMeshChunk::new(
        spec,
        SuspensionMeshOwner::Spring,
        3,
        Deformation::Coil {
            samples: Vec::new(),
        },
    );
    let mut samples = Vec::new();
    for (t, sign) in [(0.0, -1.0), (1.0, 1.0)] {
        let base = u32::try_from(samples.len()).expect("two caps");
        for j in 0..8_u16 {
            let sample = [t, TAU * f32::from(j) / 8.0, sign];
            samples.push(sample);
            let (p, n) = coil_vertex(
                spring,
                spec.extended_length() - 2.0 * spec.plates().thickness,
                spec.plates().thickness,
                sample,
            );
            caps.vertex(
                p,
                n,
                [
                    sample[1].cos() * spring.wire() / 3.0,
                    sample[1].sin() * spring.wire() / 3.0,
                ],
            );
        }
        for j in 1..7 {
            if sign > 0.0 {
                caps.indices
                    .extend_from_slice(&[base, base + j, base + j + 1]);
            } else {
                caps.indices
                    .extend_from_slice(&[base, base + j + 1, base + j]);
            }
        }
    }
    caps.deformation = Deformation::Coil { samples };
    chunks.push(caps);
}
#[expect(
    clippy::too_many_lines,
    reason = "one rigid assembly follows the supplied hardware profile"
)]
fn shock_meshes(spec: SuspensionSpec, chunks: &mut Vec<SuspensionMeshChunk>) {
    let shock = spec.shock().expect("shock host");
    let plate = spec.plates();
    let g = shock.geometry(plate).expect("validated packaging");
    let opposite = shock.body_end() == ShockBodyEnd::Opposite;
    let offset = Vec3::Y
        * if opposite {
            spec.extended_length() - plate.thickness
        } else {
            plate.thickness
        };
    let rb = shock.od() / 2.0;
    let rs = shock.shaft_diameter() / 2.0;
    let mut body = rigid(spec, opposite, 4);
    ring(&mut body, rb, 0.0, g.body_length, 0.0, opposite, offset);
    chunks.push(body);
    for (finish, ro, ri, h, y) in [
        (
            5,
            rb * 1.03,
            rb * 0.98,
            g.body_length * 0.12,
            g.body_length * 0.14,
        ),
        (
            6,
            rb * 1.02,
            rb * 0.98,
            g.body_length * 0.05,
            g.body_length * 0.815,
        ),
        (
            11,
            rb * 1.10,
            rb * 0.99,
            rb * 0.30,
            g.body_length * 0.10 - rb * 0.15,
        ),
        (1, rb * 0.62, rb * 0.5, plate.thickness * 0.5, -0.0004),
    ] {
        let mut mesh = rigid(spec, opposite, finish);
        ring(&mut mesh, ro, ri, h, y, opposite, offset);
        chunks.push(mesh);
    }
    let mut gland = rigid(spec, opposite, 5);
    lathe(
        &mut gland,
        &[
            [rs * 1.05, g.body_length],
            [rb * 0.92, g.body_length],
            [rb * 0.96, g.body_length + rb * 0.10],
            [rb * 0.96, g.body_length + rb * 0.30],
            [rb * 0.80, g.body_length + g.gland_height],
            [rs * 1.05, g.body_length + g.gland_height],
            [rs * 1.05, g.body_length],
        ],
        30,
        offset,
        opposite,
    );
    chunks.push(gland);
    // A fixed-length shaft belongs to the other mount and slides inside the body.
    let shaft_offset = Vec3::Y
        * if opposite {
            plate.thickness
        } else {
            spec.extended_length() - plate.thickness
        };
    let mut shaft = rigid(spec, !opposite, 7);
    ring(
        &mut shaft,
        rs,
        0.0,
        g.shaft_length,
        0.0,
        !opposite,
        shaft_offset,
    );
    chunks.push(shaft);
    let mut seat = rigid(spec, !opposite, 1);
    ring(
        &mut seat,
        rs * 1.9,
        rs * 1.02,
        plate.thickness * 0.5,
        -0.0004,
        !opposite,
        shaft_offset,
    );
    chunks.push(seat);
    if let Some(stop) = spec.bump_stop() {
        let mut rubber = SuspensionMeshChunk::new(
            spec,
            SuspensionMeshOwner::BumpStop,
            8,
            Deformation::Rubber {
                opposite: !opposite,
            },
        );
        let base = stop.od() / 2.0;
        let tip = (rs * 1.25).max(base * 0.52);
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let ribs = (stop.length() / (stop.od() * 0.55)).round().clamp(3.0, 6.0) as u16;
        let mut profile = vec![[rs * 1.02, 0.0], [base, 0.0]];
        for i in 0..ribs {
            let a = (f32::from(i) + 0.55) / f32::from(ribs);
            let b = f32::from(i + 1) / f32::from(ribs);
            let ra = base + (tip - base) * a;
            let rb = base + (tip - base) * b;
            profile.extend_from_slice(&[
                [ra * 1.02, stop.length() * a],
                [rb * 0.88, stop.length() * b * 0.995],
                [if i == ribs - 1 { tip } else { rb }, stop.length() * b],
            ]);
        }
        profile.extend_from_slice(&[[rs * 1.02, stop.length()], [rs * 1.02, 0.0]]);
        lathe(&mut rubber, &profile, 26, shaft_offset, !opposite);
        chunks.push(rubber);
        let mut clamp = rigid(spec, !opposite, 9);
        clamp.owner = SuspensionMeshOwner::BumpStop;
        ring(
            &mut clamp,
            rs * 1.55,
            rs * 1.01,
            shock.shaft_diameter() * 0.35,
            -0.0004,
            !opposite,
            shaft_offset,
        );
        chunks.push(clamp);
    }
}
/// Builds standalone or combined suspension hardware once. Both plates are
/// counted once; rigid chunks translate, coils compress and rubber crushes at
/// the actual gland contact. Call each chunk's `update_deformation` to animate.
pub fn suspension_meshes(spec: SuspensionSpec, compression: f32) -> Vec<SuspensionMeshChunk> {
    let mut chunks = Vec::new();
    plate_meshes(spec, false, &mut chunks);
    plate_meshes(spec, true, &mut chunks);
    if spec.spring().is_some() {
        coil_meshes(spec, &mut chunks);
    }
    if spec.shock().is_some() {
        shock_meshes(spec, &mut chunks);
    }
    chunks.retain(|chunk| !chunk.indices.is_empty());
    for chunk in &mut chunks {
        chunk.freeze();
        chunk.update_deformation(compression);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BumpStopSpec, ShockSpec};
    fn assembly(end: ShockBodyEnd) -> SuspensionSpec {
        SuspensionSpec::new(
            Some(SpringSpec::default()),
            Some(ShockSpec::new(0.5, 0.1, end, 0.0, 1.0, 1.6).unwrap()),
            Some(BumpStopSpec::new(0.05, 0.06).unwrap()),
        )
        .unwrap()
    }
    #[test]
    fn both_orientations_have_outward_winding_and_finite_normals() {
        for end in [ShockBodyEnd::Source, ShockBodyEnd::Opposite] {
            let spec = assembly(end);
            for compression in [
                0.0,
                spec.bump_contact().unwrap(),
                spec.compression_limit().0,
            ] {
                for chunk in suspension_meshes(spec, compression) {
                    for n in &chunk.normals {
                        assert!(Vec3::from_array(*n).is_normalized());
                    }
                    for triangle in chunk.indices.chunks_exact(3) {
                        let a = triangle[0] as usize;
                        let b = triangle[1] as usize;
                        let c = triangle[2] as usize;
                        let geometric = (Vec3::from_array(chunk.positions[b])
                            - Vec3::from_array(chunk.positions[a]))
                        .cross(
                            Vec3::from_array(chunk.positions[c])
                                - Vec3::from_array(chunk.positions[a]),
                        );
                        let normal = Vec3::from_array(chunk.normals[a])
                            + Vec3::from_array(chunk.normals[b])
                            + Vec3::from_array(chunk.normals[c]);
                        assert!(
                            geometric.dot(normal) >= -1e-9,
                            "finish {}, owner {:?}",
                            chunk.finish,
                            chunk.owner
                        );
                    }
                }
            }
        }
    }
    fn bounds(mesh: &SuspensionMeshChunk) -> [f32; 2] {
        mesh.positions
            .iter()
            .fold([f32::INFINITY, f32::NEG_INFINITY], |[lo, hi], p| {
                [lo.min(p[1]), hi.max(p[1])]
            })
    }
    #[test]
    fn cached_deformation_preserves_topology_and_rigid_body_and_shaft_lengths() {
        for end in [ShockBodyEnd::Source, ShockBodyEnd::Opposite] {
            let spec = assembly(end);
            let mut chunks = suspension_meshes(spec, 0.0);
            for chunk in &mut chunks {
                let indices = chunk.indices.clone();
                let uvs = chunk.uvs.clone();
                let original = bounds(chunk);
                let ptr = chunk.positions.as_ptr();
                chunk.update_deformation(spec.compression_limit().0);
                assert_eq!(ptr, chunk.positions.as_ptr());
                assert_eq!(indices, chunk.indices);
                assert_eq!(uvs, chunk.uvs);
                let compressed = bounds(chunk);
                if chunk.finish == 4 || chunk.finish == 7 {
                    assert!(
                        ((original[1] - original[0]) - (compressed[1] - compressed[0])).abs()
                            < 1e-6
                    );
                }
                if chunk.finish == 8 {
                    assert!(
                        ((compressed[1] - compressed[0]) / (original[1] - original[0]) - 0.45)
                            .abs()
                            < 1e-5
                    );
                }
                chunk.update_deformation(0.0);
                assert!(
                    original
                        .into_iter()
                        .zip(bounds(chunk))
                        .all(|(a, b)| (a - b).abs() < 1e-7)
                );
            }
        }
    }
    #[test]
    fn upper_plate_fixings_are_visible_above_outside_mount_face() {
        let spec = SuspensionSpec::new(Some(SpringSpec::default()), None, None).unwrap();
        for compression in [0.0, spec.compression_limit().0] {
            let meshes = suspension_meshes(spec, compression);
            let fixings = meshes
                .iter()
                .filter(|mesh| mesh.owner == SuspensionMeshOwner::Opposite && mesh.finish == 9)
                .collect::<Vec<_>>();
            assert_eq!(fixings.len(), 1);
            assert!(
                (bounds(fixings[0])[1] - (spec.extended_length() - compression + 0.0003)).abs()
                    < 1e-6
            );
        }
    }
    #[test]
    fn standalone_parts_and_shared_plates_have_correct_owners() {
        for (spring, shock) in [
            (Some(SpringSpec::default()), None),
            (None, Some(ShockSpec::default())),
            (Some(SpringSpec::default()), Some(ShockSpec::default())),
        ] {
            let spec = SuspensionSpec::new(spring, shock, None).unwrap();
            let meshes = suspension_meshes(spec, 0.0);
            assert_eq!(meshes.iter().filter(|m| m.finish == 0).count(), 2);
            assert_eq!(
                meshes
                    .iter()
                    .any(|m| m.owner == SuspensionMeshOwner::Spring),
                spring.is_some()
            );
            assert_eq!(meshes.iter().any(|m| m.finish == 4), shock.is_some());
        }
    }
}
