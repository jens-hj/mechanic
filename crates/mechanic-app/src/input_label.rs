//! Extruded button key glyphs from the UI font's reusable outline geometry.

use bevy::{
    asset::RenderAssetUsages, mesh::Indices, prelude::*, render::render_resource::PrimitiveTopology,
};
use lyon_tessellation::{
    BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, VertexBuffers,
    math::{Point, point},
    path::{Path, iterator::PathIterator},
};
use mechanic_core::{DriveKey, InputKind, InputMeshOwner, InputSize, input_meshes};
use mosaic_text::{FontContext, OutlineCommand, TextStyle};

pub(crate) fn mesh(key: DriveKey, size: InputSize) -> Mesh {
    let mut fonts = FontContext::new();
    let shaped = fonts.shape(&key.to_string(), &TextStyle::new(64.0), None);
    let mut builder = Path::builder();
    for outline in shaped.outlines(&mut fonts, mosaic_core::Vector2::ZERO) {
        for command in outline.commands {
            match command {
                OutlineCommand::MoveTo(p) => {
                    builder.begin(point(p.x, p.y));
                }
                OutlineCommand::LineTo(p) => {
                    builder.line_to(point(p.x, p.y));
                }
                OutlineCommand::QuadTo(c, p) => {
                    builder.quadratic_bezier_to(point(c.x, c.y), point(p.x, p.y));
                }
                OutlineCommand::CurveTo(a, b, p) => {
                    builder.cubic_bezier_to(point(a.x, a.y), point(b.x, b.y), point(p.x, p.y));
                }
                OutlineCommand::Close => {
                    builder.close();
                }
            }
        }
    }
    let path = builder.build();
    let mut triangles: VertexBuffers<Point, u32> = VertexBuffers::new();
    FillTessellator::new()
        .tessellate_path(
            &path,
            &FillOptions::default()
                .with_fill_rule(FillRule::EvenOdd)
                .with_tolerance(0.1),
            &mut BuffersBuilder::new(&mut triangles, |vertex: FillVertex<'_>| vertex.position()),
        )
        .expect("supported key outlines tessellate");
    extrude(&path, &triangles, size)
}

#[derive(Default)]
struct Geometry {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

impl Geometry {
    fn triangle(&mut self, mut vertices: [Vec3; 3], normal: Vec3) {
        if (vertices[1] - vertices[0])
            .cross(vertices[2] - vertices[0])
            .dot(normal)
            < 0.0
        {
            vertices.swap(1, 2);
        }
        for vertex in vertices {
            self.indices
                .push(u32::try_from(self.positions.len()).expect("small glyph mesh"));
            self.positions.push(vertex.to_array());
            self.normals.push(normal.to_array());
            self.uvs.push([vertex.x, vertex.z]);
        }
    }

    fn finish(self) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs)
        .with_inserted_indices(Indices::U32(self.indices))
    }
}

fn contours(path: &Path) -> Vec<Vec<Point>> {
    use lyon_tessellation::path::Event;
    let mut contours = Vec::new();
    let mut current = Vec::new();
    for event in path.iter().flattened(0.1) {
        match event {
            Event::Begin { at } => current.push(at),
            Event::Line { to, .. } => current.push(to),
            Event::End { first, .. } => {
                current.push(first);
                contours.push(std::mem::take(&mut current));
            }
            _ => unreachable!("flattened paths contain straight segments"),
        }
    }
    contours
}

fn extrude(path: &Path, triangles: &VertexBuffers<Point, u32>, size: InputSize) -> Mesh {
    let mut geometry = Geometry::default();
    if triangles.vertices.is_empty() {
        return geometry.finish();
    }
    let mut minimum = Vec2::splat(f32::INFINITY);
    let mut maximum = Vec2::splat(f32::NEG_INFINITY);
    for vertex in &triangles.vertices {
        let vertex = Vec2::new(vertex.x, vertex.y);
        minimum = minimum.min(vertex);
        maximum = maximum.max(vertex);
    }
    let scale = size.meters() * 0.35 / (maximum - minimum).max_element();
    let center = (minimum + maximum) * 0.5;
    let bottom = input_meshes(InputKind::Button, size)
        .into_iter()
        .filter(|chunk| chunk.owner == InputMeshOwner::Cap)
        .flat_map(|chunk| chunk.positions)
        .map(|position| position[1])
        .fold(f32::NEG_INFINITY, f32::max)
        + size.meters() * 0.0005;
    let top = bottom + size.meters() * 0.005;
    let position =
        |p: Point, height| Vec3::new((p.x - center.x) * scale, height, (p.y - center.y) * scale);
    for indices in triangles.indices.chunks_exact(3) {
        let points =
            [indices[0], indices[1], indices[2]].map(|index| triangles.vertices[index as usize]);
        geometry.triangle(points.map(|point| position(point, top)), Vec3::Y);
        geometry.triangle(points.map(|point| position(point, bottom)), Vec3::NEG_Y);
    }
    let contours = contours(path);
    let winding = contours
        .iter()
        .map(|contour| {
            contour
                .windows(2)
                .map(|edge| edge[0].x * edge[1].y - edge[1].x * edge[0].y)
                .sum::<f32>()
        })
        .max_by(|a, b| a.abs().total_cmp(&b.abs()))
        .unwrap_or(1.0)
        .signum();
    for contour in contours {
        for edge in contour.windows(2) {
            let a = position(edge[0], bottom);
            let b = position(edge[1], bottom);
            let normal = Vec3::new(b.z - a.z, 0.0, a.x - b.x).normalize_or_zero() * winding;
            if normal == Vec3::ZERO {
                continue;
            }
            let c = position(edge[1], top);
            let d = position(edge[0], top);
            geometry.triangle([a, b, c], normal);
            geometry.triangle([a, c, d], normal);
        }
    }
    geometry.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::mesh::VertexAttributeValues;

    #[test]
    fn every_supported_key_has_finite_extruded_geometry_at_every_size() {
        for symbol in ('A'..='Z').chain('0'..='9') {
            for size in InputSize::ALL {
                let mesh = mesh(DriveKey::new(symbol).unwrap(), size);
                let Some(VertexAttributeValues::Float32x3(vertices)) =
                    mesh.attribute(Mesh::ATTRIBUTE_POSITION)
                else {
                    panic!("positions");
                };
                assert!(!vertices.is_empty(), "{symbol} {size:?}");
                assert!(vertices.iter().flatten().all(|value| value.is_finite()));
                assert!(
                    vertices
                        .iter()
                        .any(|v| (v[1] - vertices[0][1]).abs() > f32::EPSILON)
                );
                assert!(mesh.indices().unwrap().len() >= 6);
            }
        }
    }

    #[test]
    fn o_counter_remains_open_in_the_top_surface() {
        let mesh = mesh(DriveKey::new('O').unwrap(), InputSize::Panel);
        let Some(VertexAttributeValues::Float32x3(vertices)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("positions");
        };
        let Some(VertexAttributeValues::Float32x3(normals)) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("normals");
        };
        for (triangle, normal) in vertices.chunks_exact(3).zip(normals.chunks_exact(3)) {
            if normal[0][1] < 0.5 {
                continue;
            }
            let sides = [0, 1, 2].map(|index| {
                let a = triangle[index];
                let b = triangle[(index + 1) % 3];
                a[0] * b[2] - b[0] * a[2]
            });
            assert!(
                !(sides.iter().all(|side| *side >= 0.0) || sides.iter().all(|side| *side <= 0.0)),
                "a top triangle fills the counter"
            );
        }
    }
}
