//! The triangle bounding-volume hierarchy of a collision chunk, with its ray and distance primitives.

use super::WorldBounds;
use super::groups::{TerrainIndexGroups, TerrainTriangleGroupMask};
use crate::{TerrainFace, WorldPosition};
use bevy_math::{DVec3, Vec3};

/// Compact triangle acceleration structure owned by a terrain chunk.
///
/// One hierarchy contains regular, transition, and temporary-cap triangles;
/// traversal masks select the currently active geometry without rebuilding it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TriangleBvh {
    /// Root bounds.
    pub bounds: WorldBounds,
    /// Triangles from every geometry group in traversal order.
    pub triangles: Vec<TriangleBvhTriangle>,
    /// Binary hierarchy nodes in depth-first order.
    pub nodes: Vec<TriangleBvhNode>,
}

/// One node in a triangle BVH.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TriangleBvhNode {
    /// Bounds of every descendant triangle.
    pub bounds: WorldBounds,
    /// First triangle in [`TriangleBvh::triangles`] for a leaf.
    pub first_triangle: u32,
    /// Triangle count for a leaf; zero for a branch.
    pub triangle_count: u32,
    /// First child for a branch.
    pub left_child: Option<u32>,
    /// Second child for a branch.
    pub right_child: Option<u32>,
    /// Union of regular, transition, and cap groups below this node.
    pub group_mask: TerrainTriangleGroupMask,
}

/// One triangle in the shared all-groups BVH.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TriangleBvhTriangle {
    /// Vertex indices into the owning chunk.
    pub indices: [u32; 3],
    /// Geometry group controlling whether the triangle is active.
    pub group_mask: TerrainTriangleGroupMask,
}

pub(super) fn point_bounds_distance_squared(point: DVec3, bounds: WorldBounds) -> f64 {
    (0..3)
        .map(|axis| {
            if point[axis] < bounds.minimum.0[axis] {
                bounds.minimum.0[axis] - point[axis]
            } else if point[axis] > bounds.maximum.0[axis] {
                point[axis] - bounds.maximum.0[axis]
            } else {
                0.0
            }
        })
        .map(|distance| distance * distance)
        .sum()
}

pub(super) fn closest_point_on_triangle(
    point: Vec3,
    first: Vec3,
    second: Vec3,
    third: Vec3,
) -> Vec3 {
    let ab = second - first;
    let ac = third - first;
    let ap = point - first;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return first;
    }
    let bp = point - second;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return second;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return first + ab * (d1 / (d1 - d3));
    }
    let cp = point - third;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return third;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return first + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return second + (third - second) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let inverse = (va + vb + vc).recip();
    first + ab * (vb * inverse) + ac * (vc * inverse)
}

pub(super) fn ray_intersects_bounds(
    origin: DVec3,
    direction: DVec3,
    bounds: WorldBounds,
    maximum_distance: f64,
) -> bool {
    let mut minimum_distance = 0.0_f64;
    let mut maximum = maximum_distance;
    for axis in 0..3 {
        if direction[axis].abs() <= f64::EPSILON {
            if origin[axis] < bounds.minimum.0[axis] || origin[axis] > bounds.maximum.0[axis] {
                return false;
            }
            continue;
        }
        let inverse = direction[axis].recip();
        let mut first = (bounds.minimum.0[axis] - origin[axis]) * inverse;
        let mut second = (bounds.maximum.0[axis] - origin[axis]) * inverse;
        if first > second {
            core::mem::swap(&mut first, &mut second);
        }
        minimum_distance = minimum_distance.max(first);
        maximum = maximum.min(second);
        if minimum_distance > maximum {
            return false;
        }
    }
    maximum >= 0.0
}

pub(super) fn build_triangle_bvh(
    origin: WorldPosition,
    vertices: &[[f32; 3]],
    groups: &TerrainIndexGroups,
) -> TriangleBvh {
    let mut triangles = Vec::new();
    let mut append = |indices: &[u32], group_mask: TerrainTriangleGroupMask| {
        triangles.extend(indices.chunks_exact(3).map(|indices| {
            let triangle = TriangleBvhTriangle {
                indices: [indices[0], indices[1], indices[2]],
                group_mask,
            };
            let bounds = triangle_bounds(origin, vertices, triangle.indices);
            BuildTriangle {
                triangle,
                bounds,
                centroid: (bounds.minimum.0 + bounds.maximum.0) * 0.5,
            }
        }));
    };
    append(&groups.regular, TerrainTriangleGroupMask::REGULAR);
    for face in TerrainFace::ALL {
        append(
            &groups.transitions[face.index()],
            TerrainTriangleGroupMask::transition(face),
        );
        append(
            &groups.caps[face.index()],
            TerrainTriangleGroupMask::cap(face),
        );
    }
    if triangles.is_empty() {
        return TriangleBvh::default();
    }
    let triangle_count = triangles.len();
    let mut nodes = Vec::new();
    build_bvh_node(&mut triangles, &mut nodes, 0, triangle_count);
    TriangleBvh {
        bounds: nodes[0].bounds,
        triangles: triangles
            .into_iter()
            .map(|triangle| triangle.triangle)
            .collect(),
        nodes,
    }
}

#[derive(Clone, Copy)]
pub(super) struct BuildTriangle {
    pub(super) triangle: TriangleBvhTriangle,
    pub(super) bounds: WorldBounds,
    pub(super) centroid: DVec3,
}

pub(super) fn build_bvh_node(
    triangles: &mut [BuildTriangle],
    nodes: &mut Vec<TriangleBvhNode>,
    first: usize,
    count: usize,
) -> u32 {
    let node_index = u32::try_from(nodes.len()).expect("BVH node count fits u32");
    nodes.push(TriangleBvhNode::default());
    let bounds = triangles[first..first + count]
        .iter()
        .map(|triangle| triangle.bounds)
        .reduce(union_bounds)
        .expect("a BVH node contains triangles");
    let group_mask = triangles[first..first + count]
        .iter()
        .fold(TerrainTriangleGroupMask::default(), |mask, triangle| {
            mask.union(triangle.triangle.group_mask)
        });
    if count <= 8 {
        nodes[usize::try_from(node_index).expect("node index fits usize")] = TriangleBvhNode {
            bounds,
            first_triangle: u32::try_from(first).expect("triangle offset fits u32"),
            triangle_count: u32::try_from(count).expect("leaf count fits u32"),
            left_child: None,
            right_child: None,
            group_mask,
        };
        return node_index;
    }
    let extent = bounds.maximum.0 - bounds.minimum.0;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };
    let left_count = count / 2;
    triangles[first..first + count].select_nth_unstable_by(left_count, |left, right| {
        left.centroid[axis].total_cmp(&right.centroid[axis])
    });
    let left = build_bvh_node(triangles, nodes, first, left_count);
    let right = build_bvh_node(triangles, nodes, first + left_count, count - left_count);
    nodes[usize::try_from(node_index).expect("node index fits usize")] = TriangleBvhNode {
        bounds,
        first_triangle: 0,
        triangle_count: 0,
        left_child: Some(left),
        right_child: Some(right),
        group_mask,
    };
    node_index
}

pub(super) fn triangle_bounds(
    origin: WorldPosition,
    vertices: &[[f32; 3]],
    indices: [u32; 3],
) -> WorldBounds {
    let points = indices.map(|index| {
        origin.0
            + DVec3::from_array(
                vertices[usize::try_from(index).expect("vertex fits usize")].map(f64::from),
            )
    });
    WorldBounds {
        minimum: WorldPosition(points.into_iter().reduce(DVec3::min).expect("three points")),
        maximum: WorldPosition(points.into_iter().reduce(DVec3::max).expect("three points")),
    }
}

pub(super) fn union_bounds(first: WorldBounds, second: WorldBounds) -> WorldBounds {
    WorldBounds {
        minimum: WorldPosition(first.minimum.0.min(second.minimum.0)),
        maximum: WorldPosition(first.maximum.0.max(second.maximum.0)),
    }
}

pub(super) fn ray_triangle(
    origin: DVec3,
    direction: DVec3,
    first: DVec3,
    second: DVec3,
    third: DVec3,
) -> Option<(f64, Vec3)> {
    let epsilon = 1.0e-9;
    let edge_1 = second - first;
    let edge_2 = third - first;
    let perpendicular = direction.cross(edge_2);
    let determinant = edge_1.dot(perpendicular);
    if determinant.abs() < epsilon {
        return None;
    }
    let inverse = determinant.recip();
    let from_first = origin - first;
    let u = from_first.dot(perpendicular) * inverse;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let cross = from_first.cross(edge_1);
    let v = direction.dot(cross) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let distance = edge_2.dot(cross) * inverse;
    (distance >= 0.0).then_some((
        distance,
        Vec3::new((1.0 - u - v) as f32, u as f32, v as f32),
    ))
}
