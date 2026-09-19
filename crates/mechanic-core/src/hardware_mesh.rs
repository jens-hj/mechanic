//! Mesh-building primitives shared by the authored bearing, suspension, and piston assets.

use crate::ConstructionMaterial;
use bevy_math::{Vec2, Vec3};
use core::f32::consts::TAU;

/// Existing construction texture and procedural finish modulation of authored hardware.
#[derive(Clone, Copy, Debug)]
pub struct HardwareFinish {
    /// Stable finish name.
    pub name: &'static str,
    /// Construction texture family.
    pub material: ConstructionMaterial,
    /// sRGB base colour.
    pub color: [u8; 3],
    /// Perceptual roughness.
    pub roughness: f32,
    /// Metallic reflectance.
    pub metalness: f32,
}

impl HardwareFinish {
    pub(crate) const fn new(
        name: &'static str,
        material: ConstructionMaterial,
        color: [u8; 3],
        roughness: f32,
        metalness: f32,
    ) -> Self {
        Self {
            name,
            material,
            color,
            roughness,
            metalness,
        }
    }
}

/// An indexed triangle mesh under construction.
pub(crate) trait MeshSink {
    /// Number of vertices written so far.
    fn vertex_count(&self) -> u32;
    /// Appends one vertex.
    fn vertex(&mut self, position: Vec3, normal: Vec3, uv: [f32; 2]);
    /// Appends counterclockwise triangle indices.
    fn triangles(&mut self, indices: &[u32]);
}

/// Revolves a profile of `[radius, y]` points about Y.
///
/// Each profile edge has its own vertices, retaining hard machined corners. A
/// profile runs outward along its underside and back inward along its top.
pub(crate) fn lathe(
    sink: &mut impl MeshSink,
    profile: &[[f32; 2]],
    segments: u16,
    offset: Vec3,
    flip: bool,
) {
    for edge in profile.windows(2) {
        let [r0, y0] = edge[0];
        let [r1, y1] = edge[1];
        let normal = Vec3::new(y1 - y0, r0 - r1, 0.0).normalize_or_zero();
        let base = sink.vertex_count();
        for i in 0..=segments {
            let theta = TAU * f32::from(i) / f32::from(segments);
            let (s, c) = theta.sin_cos();
            for [r, y] in edge {
                let mut p = Vec3::new(r * c, *y, r * s);
                let mut n = Vec3::new(normal.x * c, normal.y, normal.x * s);
                if flip {
                    p.x = -p.x;
                    p.y = -p.y;
                    n.x = -n.x;
                    n.y = -n.y;
                }
                let uv = if (y1 - y0).abs() < 1e-7 {
                    [r * c / 1.5, r * s / 1.5]
                } else {
                    [theta * r / 1.5, y / 1.5]
                };
                sink.vertex(p + offset, n, uv);
            }
        }
        for i in 0..u32::from(segments) {
            let a = base + 2 * i;
            // Y tangent cross angular tangent points outward. An end on the
            // axis collapses one triangle of the quad, which is left out.
            if r0 > 0.0 {
                sink.triangles(&[a, a + 1, a + 2]);
            }
            if r1 > 0.0 {
                sink.triangles(&[a + 2, a + 1, a + 3]);
            }
        }
    }
}

/// A closed annular ring standing on `y`; a zero inner radius makes a solid cylinder.
pub(crate) fn ring(
    sink: &mut impl MeshSink,
    [outer, inner]: [f32; 2],
    height: f32,
    y: f32,
    segments: u16,
    flip: bool,
    offset: Vec3,
) {
    let mut profile = vec![
        [inner, y],
        [outer, y],
        [outer, y + height],
        [inner, y + height],
    ];
    if inner > 0.0 {
        profile.push([inner, y]);
    }
    lathe(sink, &profile, segments, offset, flip);
}

/// Ear-clips a simple polygon whose signed area has the given orientation.
pub(crate) fn triangulate(points: &[Vec2], orientation: f32) -> Vec<[usize; 3]> {
    let mut remaining: Vec<_> = (0..points.len()).collect();
    let mut triangles = Vec::with_capacity(points.len() - 2);
    while remaining.len() > 3 {
        let ear = (0..remaining.len())
            .find(|&i| {
                let indices = [
                    remaining[(i + remaining.len() - 1) % remaining.len()],
                    remaining[i],
                    remaining[(i + 1) % remaining.len()],
                ];
                let [a, b, c] = indices.map(|index| points[index]);
                if (b - a).perp_dot(c - b) * orientation <= 0.0 {
                    return false;
                }
                !remaining.iter().any(|index| {
                    if indices.contains(index) {
                        return false;
                    }
                    let p = points[*index];
                    [
                        (b - a).perp_dot(p - a),
                        (c - b).perp_dot(p - b),
                        (a - c).perp_dot(p - c),
                    ]
                    .iter()
                    .all(|cross| cross * orientation >= -1.0e-10)
                })
            })
            .expect("authored profiles are simple polygons");
        triangles.push([
            remaining[(ear + remaining.len() - 1) % remaining.len()],
            remaining[ear],
            remaining[(ear + 1) % remaining.len()],
        ]);
        remaining.remove(ear);
    }
    triangles.push([remaining[0], remaining[1], remaining[2]]);
    triangles
}
