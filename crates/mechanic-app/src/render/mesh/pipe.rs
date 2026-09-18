//! Cylinder, pipe-bend, and pipe-junction surfaces with their end faces and texture frames.

use crate::render::mesh::bearing::append_bearing_face_ring;
use crate::render::mesh::primitives::{
    append_mesh_quad, append_mesh_quad_with_normals, append_mesh_triangle,
};
use bevy::prelude::{Quat, Vec3};
use mechanic_core::{
    ConstructionGraph, CylinderDimensions, FaceKind, FaceOwner, FaceRef, PartId, PartSpec,
    PipeBendDimensions,
};
use std::collections::{HashMap, HashSet, VecDeque};

pub(crate) fn pipe_endpoint_texture_u(spec: PartSpec, face: FaceKind) -> Option<f32> {
    match (spec, face) {
        (PartSpec::Cylinder(cylinder), FaceKind::NegativeY) => {
            Some(-cylinder.dimensions.axial_length() * 0.5)
        }
        (PartSpec::Cylinder(cylinder), FaceKind::PositiveY) => {
            Some(cylinder.dimensions.axial_length() * 0.5)
        }
        (PartSpec::PipeBend(bend), FaceKind::NegativeX) => {
            Some(-std::f32::consts::FRAC_PI_2 * bend.dimensions.radius())
        }
        (PartSpec::PipeBend(_), FaceKind::PositiveY) => Some(0.0),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PipeTextureOffset {
    pub(crate) u: f32,
    pub(crate) v_angle: f32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PipeEndpointTextureFrame {
    pub(crate) direction: Vec3,
    pub(crate) radial_zero: Vec3,
}

pub(crate) fn pipe_endpoint_texture_frame(
    spec: PartSpec,
    face: FaceKind,
) -> Option<PipeEndpointTextureFrame> {
    let rotation = spec.pose().rotation.quaternion();
    match (spec, face) {
        (PartSpec::Cylinder(_), FaceKind::NegativeY | FaceKind::PositiveY) => {
            Some(PipeEndpointTextureFrame {
                direction: rotation * Vec3::Y,
                radial_zero: rotation * Vec3::X,
            })
        }
        (PartSpec::PipeBend(_), FaceKind::NegativeX) => Some(PipeEndpointTextureFrame {
            direction: rotation * Vec3::X,
            radial_zero: rotation * Vec3::NEG_Y,
        }),
        (PartSpec::PipeBend(_), FaceKind::PositiveY) => Some(PipeEndpointTextureFrame {
            direction: rotation * Vec3::Y,
            radial_zero: rotation * Vec3::X,
        }),
        _ => None,
    }
}

/// Carries both lengthwise and circumferential texture phase across welded pipe ends.
pub(crate) fn pipe_texture_offsets(
    graph: &ConstructionGraph,
) -> HashMap<PartId, PipeTextureOffset> {
    let mut neighbours = HashMap::<PartId, Vec<(PartId, PipeTextureOffset)>>::new();
    for (_, weld) in graph.welds() {
        let (FaceOwner::Part(first), FaceOwner::Part(second)) =
            (weld.first.owner, weld.second.owner)
        else {
            continue;
        };
        let Some(first_u) = graph
            .part(first)
            .copied()
            .and_then(|spec| pipe_endpoint_texture_u(spec, weld.first.face))
        else {
            continue;
        };
        let Some(second_u) = graph
            .part(second)
            .copied()
            .and_then(|spec| pipe_endpoint_texture_u(spec, weld.second.face))
        else {
            continue;
        };
        let first_spec = graph
            .part(first)
            .copied()
            .expect("welded part remains in graph");
        let second_spec = graph
            .part(second)
            .copied()
            .expect("welded part remains in graph");
        let mut first_frame = pipe_endpoint_texture_frame(first_spec, weld.first.face)
            .expect("pipe endpoint exposes a texture frame");
        let mut second_frame = pipe_endpoint_texture_frame(second_spec, weld.second.face)
            .expect("pipe endpoint exposes a texture frame");
        let first_rotation = graph
            .part_frame(first)
            .expect("welded part has a frame")
            .rotation();
        let second_rotation = graph
            .part_frame(second)
            .expect("welded part has a frame")
            .rotation();
        first_frame.direction = first_rotation * first_frame.direction;
        first_frame.radial_zero = first_rotation * first_frame.radial_zero;
        second_frame.direction = second_rotation * second_frame.direction;
        second_frame.radial_zero = second_rotation * second_frame.radial_zero;
        let v_angle = if first_frame.direction.dot(second_frame.direction) > 1.0 - 1.0e-4 {
            let second_angular = -second_frame.direction.cross(second_frame.radial_zero);
            -first_frame
                .radial_zero
                .dot(second_angular)
                .atan2(first_frame.radial_zero.dot(second_frame.radial_zero))
        } else {
            0.0
        };
        let second_from_first = PipeTextureOffset {
            u: first_u - second_u,
            v_angle,
        };
        neighbours
            .entry(first)
            .or_default()
            .push((second, second_from_first));
        neighbours.entry(second).or_default().push((
            first,
            PipeTextureOffset {
                u: -second_from_first.u,
                v_angle: -second_from_first.v_angle,
            },
        ));
    }

    let mut offsets = HashMap::new();
    let mut pending = VecDeque::new();
    for (root, spec) in graph.parts() {
        if !matches!(spec, PartSpec::Cylinder(_) | PartSpec::PipeBend(_))
            || offsets.contains_key(&root)
        {
            continue;
        }
        offsets.insert(root, PipeTextureOffset::default());
        pending.push_back(root);
        while let Some(part) = pending.pop_front() {
            let offset = offsets[&part];
            for &(neighbour, delta) in neighbours.get(&part).into_iter().flatten() {
                if let std::collections::hash_map::Entry::Vacant(entry) = offsets.entry(neighbour) {
                    entry.insert(PipeTextureOffset {
                        u: offset.u + delta.u,
                        v_angle: offset.v_angle + delta.v_angle,
                    });
                    pending.push_back(neighbour);
                }
            }
        }
    }
    offsets
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PipeEndFaces {
    pub(crate) inlet: bool,
    pub(crate) outlet: bool,
}

impl PipeEndFaces {
    pub(crate) const ALL: Self = Self {
        inlet: true,
        outlet: true,
    };
}

/// Welded pipe end caps hidden because the mating end covers them completely.
///
/// Equal cross-sections hide both caps; a thinner pipe welded to a wider one
/// hides only its own cap, since the wider cap still shows around it.
pub(crate) fn welded_pipe_ends(graph: &ConstructionGraph) -> HashSet<FaceRef> {
    let mut ends = HashSet::new();
    for (_, weld) in graph.welds() {
        let (Some(first), Some(second)) = (
            pipe_end_section(graph, weld.first),
            pipe_end_section(graph, weld.second),
        ) else {
            continue;
        };
        if first.center.distance(second.center) > PIPE_END_COVER_TOLERANCE {
            continue;
        }
        if second.covers(first) {
            ends.insert(weld.first);
        }
        if first.covers(second) {
            ends.insert(weld.second);
        }
    }
    ends
}

pub(crate) const PIPE_END_COVER_TOLERANCE: f32 = 1.0e-4;

/// Annular cross-section of a pipe endpoint in world space.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PipeEndSection {
    pub(crate) center: Vec3,
    pub(crate) inner_radius: f32,
    pub(crate) outer_radius: f32,
    pub(crate) sweep_degrees: u16,
    /// Direction the retained sector is centred on.
    pub(crate) sector_axis: Vec3,
}

impl PipeEndSection {
    pub(crate) fn covers(self, other: Self) -> bool {
        let radial = self.outer_radius >= other.outer_radius - PIPE_END_COVER_TOLERANCE
            && self.inner_radius <= other.inner_radius + PIPE_END_COVER_TOLERANCE;
        let angular = self.sweep_degrees >= 360
            || (self.sweep_degrees >= other.sweep_degrees
                && self.sector_axis.dot(other.sector_axis) > 1.0 - PIPE_END_COVER_TOLERANCE);
        radial && angular
    }
}

pub(crate) fn pipe_end_section(graph: &ConstructionGraph, face: FaceRef) -> Option<PipeEndSection> {
    let FaceOwner::Part(part) = face.owner else {
        return None;
    };
    let spec = graph.part(part).copied()?;
    pipe_endpoint_texture_u(spec, face.face)?;
    let (outer_diameter, inner_diameter, sweep_degrees) = match spec {
        PartSpec::Cylinder(cylinder) => (
            cylinder.dimensions.outer_diameter(),
            cylinder.dimensions.inner_diameter(),
            cylinder.dimensions.sweep_angle_degrees(),
        ),
        PartSpec::PipeBend(bend) => (
            bend.dimensions.outer_diameter(),
            bend.dimensions.inner_diameter(),
            360,
        ),
        _ => return None,
    };
    let rotation = graph.part_frame(part)?.rotation() * spec.pose().rotation.quaternion();
    Some(PipeEndSection {
        center: crate::builder::try_face_geometry_from_ref(face, Some(graph))?.center,
        inner_radius: inner_diameter * 0.5,
        outer_radius: outer_diameter * 0.5,
        sweep_degrees,
        sector_axis: rotation * Vec3::X,
    })
}

pub(crate) fn pipe_end_faces(part: PartId, welded_ends: &HashSet<FaceRef>) -> PipeEndFaces {
    let hidden = |face| welded_ends.contains(&FaceRef::part(part, face));
    PipeEndFaces {
        inlet: !hidden(FaceKind::NegativeY) && !hidden(FaceKind::NegativeX),
        outlet: !hidden(FaceKind::PositiveY),
    }
}

pub(crate) fn append_cylinder_shape(
    center: Vec3,
    rotation: Quat,
    dimensions: CylinderDimensions,
    scale: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    append_cylinder_shape_with_end_faces(
        center,
        rotation,
        dimensions,
        scale,
        PipeEndFaces::ALL,
        0.0,
        positions,
        normals,
        indices,
    );
}

#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn append_cylinder_shape_with_end_faces(
    center: Vec3,
    rotation: Quat,
    dimensions: CylinderDimensions,
    scale: f32,
    end_faces: PipeEndFaces,
    v_angle_offset: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    if dimensions.sweep_angle_degrees() == 360 {
        append_annular_cylinder_with_end_faces(
            center,
            rotation * Vec3::Y,
            dimensions.outer_diameter() * scale,
            dimensions.inner_diameter() * scale,
            dimensions.axial_length() * scale,
            end_faces,
            true,
            Some(rotation * (Vec3::X * v_angle_offset.cos() - Vec3::Z * v_angle_offset.sin())),
            positions,
            normals,
            indices,
        );
        return;
    }

    let axis = rotation * Vec3::Y;
    let tangent_u = rotation * Vec3::X;
    let tangent_v = rotation * Vec3::Z;
    let outer = dimensions.outer_diameter() * scale * 0.5;
    let inner = dimensions.inner_diameter() * scale * 0.5;
    let half_length = dimensions.axial_length() * scale * 0.5;
    let sweep = dimensions.sweep_angle_radians();
    let segment_count = dimensions.sweep_angle_degrees() / 15;
    let lower = center - axis * half_length;
    let upper = center + axis * half_length;
    let radial = |angle: f32| tangent_u * angle.cos() + tangent_v * angle.sin();
    let angular = |angle: f32| -tangent_u * angle.sin() + tangent_v * angle.cos();

    for segment in 0..segment_count {
        let first_angle = -sweep * 0.5 + sweep * f32::from(segment) / f32::from(segment_count);
        let second_angle = -sweep * 0.5 + sweep * f32::from(segment + 1) / f32::from(segment_count);
        let first = radial(first_angle);
        let second = radial(second_angle);
        append_mesh_quad(
            [
                lower + first * outer,
                upper + first * outer,
                upper + second * outer,
                lower + second * outer,
            ],
            (first + second).normalize(),
            positions,
            normals,
            indices,
        );
        if inner > 0.0 {
            append_mesh_quad(
                [
                    lower + second * inner,
                    upper + second * inner,
                    upper + first * inner,
                    lower + first * inner,
                ],
                -(first + second).normalize(),
                positions,
                normals,
                indices,
            );
            append_mesh_quad(
                [
                    lower + first * inner,
                    lower + first * outer,
                    lower + second * outer,
                    lower + second * inner,
                ],
                -axis,
                positions,
                normals,
                indices,
            );
            append_mesh_quad(
                [
                    upper + second * inner,
                    upper + second * outer,
                    upper + first * outer,
                    upper + first * inner,
                ],
                axis,
                positions,
                normals,
                indices,
            );
        } else {
            append_mesh_triangle(
                [lower, lower + first * outer, lower + second * outer],
                -axis,
                positions,
                normals,
                indices,
            );
            append_mesh_triangle(
                [upper, upper + second * outer, upper + first * outer],
                axis,
                positions,
                normals,
                indices,
            );
        }
    }

    for (angle, normal, reverse) in [
        (-sweep * 0.5, -angular(-sweep * 0.5), false),
        (sweep * 0.5, angular(sweep * 0.5), true),
    ] {
        let direction = radial(angle);
        let inner_lower = lower + direction * inner;
        let inner_upper = upper + direction * inner;
        let outer_lower = lower + direction * outer;
        let outer_upper = upper + direction * outer;
        let vertices = if reverse {
            [inner_lower, outer_lower, outer_upper, inner_upper]
        } else {
            [inner_lower, inner_upper, outer_upper, outer_lower]
        };
        append_mesh_quad(vertices, normal, positions, normals, indices);
    }
}

pub(crate) fn append_pipe_bend_shape(
    corner: Vec3,
    rotation: Quat,
    dimensions: PipeBendDimensions,
    scale: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    append_pipe_bend_shape_with_end_faces(
        corner,
        rotation,
        dimensions,
        scale,
        PipeEndFaces::ALL,
        0.0,
        positions,
        normals,
        indices,
    );
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn append_pipe_bend_shape_with_end_faces(
    corner: Vec3,
    rotation: Quat,
    dimensions: PipeBendDimensions,
    scale: f32,
    end_faces: PipeEndFaces,
    v_angle_offset: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    const ARC_SLICES: u16 = mechanic_core::PIPE_BEND_ARC_SLICES;
    const RADIAL_SIDES: u16 = mechanic_core::PIPE_BEND_RADIAL_SIDES;
    let radius = dimensions.radius() * scale;
    let outer = dimensions.outer_diameter() * scale * 0.5;
    let inner = dimensions.inner_diameter() * scale * 0.5;
    let curve_center = Vec3::new(-radius, radius, 0.0);
    let point = |theta: f32, phi: f32, tube_radius: f32| {
        let radial = Vec3::new(theta.cos(), theta.sin(), 0.0);
        curve_center
            + radial * (radius + tube_radius * phi.cos())
            + Vec3::Z * (tube_radius * phi.sin())
    };
    let surface_normal = |theta: f32, phi: f32| {
        let radial = Vec3::new(theta.cos(), theta.sin(), 0.0);
        rotation * (radial * phi.cos() + Vec3::Z * phi.sin())
    };
    let world = |local: Vec3| corner + rotation * local;

    for arc in 0..ARC_SLICES {
        let theta0 = -std::f32::consts::FRAC_PI_2
            + std::f32::consts::FRAC_PI_2 * f32::from(arc) / f32::from(ARC_SLICES);
        let theta1 = -std::f32::consts::FRAC_PI_2
            + std::f32::consts::FRAC_PI_2 * f32::from(arc + 1) / f32::from(ARC_SLICES);
        for side in 0..RADIAL_SIDES {
            let phi0 =
                std::f32::consts::TAU * f32::from(side) / f32::from(RADIAL_SIDES) - v_angle_offset;
            let phi1 = std::f32::consts::TAU * f32::from(side + 1) / f32::from(RADIAL_SIDES)
                - v_angle_offset;
            append_mesh_quad_with_normals(
                [
                    world(point(theta0, phi0, outer)),
                    world(point(theta1, phi0, outer)),
                    world(point(theta1, phi1, outer)),
                    world(point(theta0, phi1, outer)),
                ],
                [
                    surface_normal(theta0, phi0),
                    surface_normal(theta1, phi0),
                    surface_normal(theta1, phi1),
                    surface_normal(theta0, phi1),
                ],
                positions,
                normals,
                indices,
            );
            if inner > 0.0 {
                append_mesh_quad_with_normals(
                    [
                        world(point(theta0, phi1, inner)),
                        world(point(theta1, phi1, inner)),
                        world(point(theta1, phi0, inner)),
                        world(point(theta0, phi0, inner)),
                    ],
                    [
                        -surface_normal(theta0, phi1),
                        -surface_normal(theta1, phi1),
                        -surface_normal(theta1, phi0),
                        -surface_normal(theta0, phi0),
                    ],
                    positions,
                    normals,
                    indices,
                );
            }
        }
    }

    append_pipe_bend_end_faces(
        corner,
        rotation,
        dimensions,
        scale,
        end_faces,
        v_angle_offset,
        positions,
        normals,
        indices,
    );
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn append_pipe_bend_end_faces(
    corner: Vec3,
    rotation: Quat,
    dimensions: PipeBendDimensions,
    scale: f32,
    end_faces: PipeEndFaces,
    v_angle_offset: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    const RADIAL_SIDES: u16 = 24;
    let radius = dimensions.radius() * scale;
    let outer = dimensions.outer_diameter() * scale * 0.5;
    let inner = dimensions.inner_diameter() * scale * 0.5;
    let curve_center = Vec3::new(-radius, radius, 0.0);
    let point = |theta: f32, phi: f32, tube_radius: f32| {
        let radial = Vec3::new(theta.cos(), theta.sin(), 0.0);
        curve_center
            + radial * (radius + tube_radius * phi.cos())
            + Vec3::Z * (tube_radius * phi.sin())
    };
    let world = |local: Vec3| corner + rotation * local;

    for (theta, end_normal, reverse, visible) in [
        (
            -std::f32::consts::FRAC_PI_2,
            Vec3::NEG_X,
            false,
            end_faces.inlet,
        ),
        (0.0, Vec3::Y, true, end_faces.outlet),
    ] {
        if !visible {
            continue;
        }
        let centerline = point(theta, 0.0, 0.0);
        for side in 0..RADIAL_SIDES {
            let phi0 =
                std::f32::consts::TAU * f32::from(side) / f32::from(RADIAL_SIDES) - v_angle_offset;
            let phi1 = std::f32::consts::TAU * f32::from(side + 1) / f32::from(RADIAL_SIDES)
                - v_angle_offset;
            if inner > 0.0 {
                let vertices = if reverse {
                    [
                        world(point(theta, phi1, inner)),
                        world(point(theta, phi1, outer)),
                        world(point(theta, phi0, outer)),
                        world(point(theta, phi0, inner)),
                    ]
                } else {
                    [
                        world(point(theta, phi0, inner)),
                        world(point(theta, phi0, outer)),
                        world(point(theta, phi1, outer)),
                        world(point(theta, phi1, inner)),
                    ]
                };
                append_mesh_quad(vertices, rotation * end_normal, positions, normals, indices);
            } else {
                let vertices = if reverse {
                    [
                        world(centerline),
                        world(point(theta, phi1, outer)),
                        world(point(theta, phi0, outer)),
                    ]
                } else {
                    [
                        world(centerline),
                        world(point(theta, phi0, outer)),
                        world(point(theta, phi1, outer)),
                    ]
                };
                append_mesh_triangle(vertices, rotation * end_normal, positions, normals, indices);
            }
        }
    }
}

#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn append_annular_cylinder_with_end_faces(
    anchor: Vec3,
    axis: Vec3,
    outer_diameter: f32,
    inner_diameter: f32,
    axial_length: f32,
    end_faces: PipeEndFaces,
    duplicate_side_seam: bool,
    circumference_start: Option<Vec3>,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    const SEGMENTS: u16 = 24;
    let axis = axis.normalize();
    let (tangent_u, tangent_v, positive_circumference) = if let Some(start) = circumference_start {
        let tangent_u = start.normalize();
        (tangent_u, -axis.cross(tangent_u), true)
    } else {
        let tangent_u = if axis.y.abs() < 0.9 {
            axis.cross(Vec3::Y).normalize()
        } else {
            axis.cross(Vec3::X).normalize()
        };
        (tangent_u, axis.cross(tangent_u), false)
    };
    let outer_radius = outer_diameter * 0.5;
    let inner_radius = inner_diameter * 0.5;
    let half_depth = axial_length * 0.5;
    let lower = anchor - axis * half_depth;
    let upper = anchor + axis * half_depth;
    let base = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");

    let side_ring_vertices = if duplicate_side_seam {
        SEGMENTS + 1
    } else {
        SEGMENTS
    };
    for segment in 0..side_ring_vertices {
        let angle = std::f32::consts::TAU * f32::from(segment) / f32::from(SEGMENTS);
        let radial = tangent_u * angle.cos() + tangent_v * angle.sin();
        positions.push((lower + radial * outer_radius).to_array());
        positions.push((upper + radial * outer_radius).to_array());
        normals.push(radial.to_array());
        normals.push(radial.to_array());
    }
    for segment in 0..SEGMENTS {
        let next = if duplicate_side_seam {
            segment + 1
        } else {
            (segment + 1) % SEGMENTS
        };
        let lower_current = base + u32::from(segment) * 2;
        let upper_current = lower_current + 1;
        let lower_next = base + u32::from(next) * 2;
        let upper_next = lower_next + 1;
        if positive_circumference {
            indices.extend([
                lower_current,
                upper_current,
                lower_next,
                upper_current,
                upper_next,
                lower_next,
            ]);
        } else {
            indices.extend([
                lower_current,
                lower_next,
                upper_current,
                upper_current,
                lower_next,
                upper_next,
            ]);
        }
    }

    if inner_radius == 0.0 {
        for (center, normal, visible, reverse) in [
            (lower, -axis, end_faces.inlet, !positive_circumference),
            (upper, axis, end_faces.outlet, positive_circumference),
        ] {
            if !visible {
                continue;
            }
            let center_index = u32::try_from(positions.len()).unwrap();
            positions.push(center.to_array());
            normals.push(normal.to_array());
            let ring = append_bearing_face_ring(
                center,
                normal,
                outer_radius,
                tangent_u,
                tangent_v,
                positions,
                normals,
            );
            for segment in 0..SEGMENTS {
                let next = (segment + 1) % SEGMENTS;
                let current = u32::from(segment);
                let next = u32::from(next);
                if reverse {
                    indices.extend([center_index, ring + next, ring + current]);
                } else {
                    indices.extend([center_index, ring + current, ring + next]);
                }
            }
        }
        return;
    }

    let inner_side = u32::try_from(positions.len()).unwrap();
    for segment in 0..side_ring_vertices {
        let angle = std::f32::consts::TAU * f32::from(segment) / f32::from(SEGMENTS);
        let radial = tangent_u * angle.cos() + tangent_v * angle.sin();
        positions.push((lower + radial * inner_radius).to_array());
        positions.push((upper + radial * inner_radius).to_array());
        normals.push((-radial).to_array());
        normals.push((-radial).to_array());
    }
    for segment in 0..SEGMENTS {
        let next = if duplicate_side_seam {
            segment + 1
        } else {
            (segment + 1) % SEGMENTS
        };
        let lower_current = inner_side + u32::from(segment) * 2;
        let upper_current = lower_current + 1;
        let lower_next = inner_side + u32::from(next) * 2;
        let upper_next = lower_next + 1;
        if positive_circumference {
            indices.extend([
                lower_current,
                lower_next,
                upper_current,
                upper_current,
                lower_next,
                upper_next,
            ]);
        } else {
            indices.extend([
                lower_current,
                upper_current,
                lower_next,
                upper_current,
                upper_next,
                lower_next,
            ]);
        }
    }

    for (center, normal, visible, reverse) in [
        (lower, -axis, end_faces.inlet, !positive_circumference),
        (upper, axis, end_faces.outlet, positive_circumference),
    ] {
        if !visible {
            continue;
        }
        let outer = append_bearing_face_ring(
            center,
            normal,
            outer_radius,
            tangent_u,
            tangent_v,
            positions,
            normals,
        );
        let inner = append_bearing_face_ring(
            center,
            normal,
            inner_radius,
            tangent_u,
            tangent_v,
            positions,
            normals,
        );
        for segment in 0..SEGMENTS {
            let next = (segment + 1) % SEGMENTS;
            let current = u32::from(segment);
            let next = u32::from(next);
            if reverse {
                indices.extend([
                    outer + current,
                    inner + current,
                    outer + next,
                    outer + next,
                    inner + current,
                    inner + next,
                ]);
            } else {
                indices.extend([
                    outer + current,
                    outer + next,
                    inner + current,
                    outer + next,
                    inner + next,
                    inner + current,
                ]);
            }
        }
    }
}

/// Junction surface triangles as `(local position, local normal)` corners:
/// the outside, then the bore facing inward.
pub(crate) fn pipe_junction_mesh_corners(
    junction: mechanic_core::PipeJunctionSpec,
) -> Vec<[(Vec3, Vec3); 3]> {
    let mut corners = Vec::new();
    for triangle in mechanic_core::pipe_junction_triangles(junction) {
        corners.push(
            triangle
                .outer
                .map(|point| (point, triangle.outer_normal(point))),
        );
        if triangle.inner_surface.is_some() {
            let [a, b, c] = triangle.inner;
            corners.push(
                [c, b, a].map(|point| (point, triangle.inner_normal(point).unwrap_or_default())),
            );
        }
    }
    corners
}

/// Pipe-shaped junction with exact arm normals, scaled about its centre.
pub(crate) fn append_pipe_junction_shape(
    translation: Vec3,
    rotation: Quat,
    junction: mechanic_core::PipeJunctionSpec,
    scale: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    for triangle in pipe_junction_mesh_corners(junction) {
        let base = u32::try_from(positions.len()).expect("construction mesh fits 32-bit indices");
        for (position, normal) in triangle {
            positions.push((translation + rotation * (position * scale)).to_array());
            normals.push((rotation * normal).to_array());
        }
        indices.extend([base, base + 1, base + 2]);
    }
}
