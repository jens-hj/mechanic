//! Convex endpoint validation shared by default and runtime weld candidates.

use crate::{ColliderShape, ConstructionFrame, LocalCollider};
use bevy_math::Vec3;

/// World-space convex collider geometry for separating-axis validation.
#[derive(Clone, Debug)]
pub struct WeldCollider {
    /// Convex hull vertices.
    pub vertices: Vec<Vec3>,
    /// Face normals; both signs of each separating axis are equivalent.
    pub normals: Vec<Vec3>,
    /// Straight edge directions used to generate cross-product axes.
    pub edges: Vec<Vec3>,
}

impl WeldCollider {
    /// Evaluates a compiled collider at a supplied body transform.
    pub fn new(collider: &LocalCollider, transform: ConstructionFrame) -> Self {
        match &collider.shape {
            ColliderShape::Cuboid {
                local_rotation,
                half_extents,
            } => {
                let rotation = transform.rotation() * *local_rotation;
                let center = transform.point(collider.local_center);
                let axes = [rotation * Vec3::X, rotation * Vec3::Y, rotation * Vec3::Z];
                let mut vertices = Vec::with_capacity(8);
                for x in [-1.0, 1.0] {
                    for y in [-1.0, 1.0] {
                        for z in [-1.0, 1.0] {
                            vertices.push(center + rotation * (*half_extents * Vec3::new(x, y, z)));
                        }
                    }
                }
                Self {
                    vertices,
                    normals: axes.to_vec(),
                    edges: axes.to_vec(),
                }
            }
            ColliderShape::Convex(shape) => Self {
                vertices: shape.vertices.iter().map(|&v| transform.point(v)).collect(),
                normals: shape
                    .face_planes
                    .iter()
                    .map(|p| transform.vector(p.truncate()))
                    .collect(),
                edges: shape
                    .edge_directions
                    .iter()
                    .map(|&e| transform.vector(e))
                    .collect(),
            },
        }
    }

    /// Negative values mean separation; positive values mean penetration.
    /// Touching is zero within floating-point tolerance.
    pub fn penetration(&self, other: &Self) -> f32 {
        let mut minimum = f32::INFINITY;
        for axis in self
            .normals
            .iter()
            .copied()
            .chain(other.normals.iter().copied())
            .chain(
                self.edges
                    .iter()
                    .flat_map(|&a| other.edges.iter().map(move |&b| a.cross(b))),
            )
        {
            let Some(axis) = axis.try_normalize() else {
                continue;
            };
            let interval = |vertices: &[Vec3]| {
                vertices
                    .iter()
                    .map(|v| v.dot(axis))
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), value| {
                        (low.min(value), high.max(value))
                    })
            };
            let (a_low, a_high) = interval(&self.vertices);
            let (b_low, b_high) = interval(&other.vertices);
            minimum = minimum.min((a_high - b_low).min(b_high - a_low));
        }
        minimum
    }
}
