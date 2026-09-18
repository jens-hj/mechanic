//! The convex cells each part kind starts from, split by material layer.

use super::model::{SurfacePatchKey, TopologySource};
use super::polygon::{
    ClipPlane, EPSILON, PolyCell, PolyFace, clip_cell, polygon_normal, without_repeated_vertices,
};
use crate::{
    CYLINDER_SWEEP_STEP_DEGREES, ConvexPiece, FaceKind, PartPiece, PartSpec, PipeBendSpec,
};
use bevy_math::{DVec3, Quat, Vec3};

pub(super) fn pieces_to_cells(pieces: Vec<PartPiece>) -> Vec<PolyCell> {
    pieces
        .into_iter()
        .map(|piece| match piece {
            PartPiece::Cuboid {
                center,
                half_extents,
                rotation,
                ..
            } => cuboid_cell(center, half_extents, rotation),
            PartPiece::Convex(piece) => convex_piece_cell(piece),
        })
        .collect()
}

pub(super) fn cuboid_cell(center: Vec3, half: Vec3, rotation: Quat) -> PolyCell {
    let point = |x: f32, y: f32, z: f32| DVec3::from(center + rotation * Vec3::new(x, y, z));
    let vertices = [
        point(-half.x, -half.y, -half.z),
        point(half.x, -half.y, -half.z),
        point(-half.x, half.y, -half.z),
        point(half.x, half.y, -half.z),
        point(-half.x, -half.y, half.z),
        point(half.x, -half.y, half.z),
        point(-half.x, half.y, half.z),
        point(half.x, half.y, half.z),
    ];
    let loops = [
        [0, 4, 6, 2],
        [1, 3, 7, 5],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 2, 3, 1],
        [4, 5, 7, 6],
    ];
    PolyCell {
        band: 0,
        faces: loops
            .into_iter()
            .enumerate()
            .map(|(index, face)| {
                base_face(face.map(|vertex| vertices[vertex]).to_vec(), index as u32)
            })
            .collect(),
    }
}

pub(super) fn convex_piece_cell(piece: ConvexPiece) -> PolyCell {
    PolyCell {
        band: 0,
        faces: piece
            .faces
            .into_iter()
            .enumerate()
            .map(|(index, face)| {
                base_face(
                    face.indices
                        .into_iter()
                        .map(|vertex| DVec3::from(piece.vertices[vertex as usize]))
                        .collect(),
                    face.grid_face
                        .map_or(index as u32, |grid| grid_patch(grid.face)),
                )
            })
            .collect(),
    }
}

pub(super) const fn grid_patch(face: FaceKind) -> u32 {
    match face {
        FaceKind::NegativeX => 0,
        FaceKind::PositiveX => 1,
        FaceKind::NegativeY => 2,
        FaceKind::PositiveY => 3,
        FaceKind::NegativeZ => 4,
        FaceKind::PositiveZ => 5,
    }
}

pub(super) fn base_face(vertices: Vec<DVec3>, local: u32) -> PolyFace {
    let key = SurfacePatchKey {
        source: TopologySource::Base,
        local,
    };
    PolyFace {
        vertices,
        patch: key,
        family: key,
        smoothing_group: 0,
        smooth_with: Vec::new(),
        uv_provenance: key,
    }
}

pub(super) fn cylinder_cells(spec: crate::CylinderSpec) -> Vec<PolyCell> {
    let outer = f64::from(spec.dimensions.outer_diameter()) * 0.5;
    let inner = f64::from(spec.dimensions.inner_diameter()) * 0.5;
    let half_length = f64::from(spec.dimensions.axial_length()) * 0.5;
    let segments = usize::from(spec.dimensions.sweep_angle_degrees() / CYLINDER_SWEEP_STEP_DEGREES);
    let sweep = f64::from(spec.dimensions.sweep_angle_radians());
    let start = -sweep * 0.5;
    let transform = |point: DVec3| {
        DVec3::from(spec.pose.translation())
            + DVec3::from(spec.pose.rotation.quaternion() * point.as_vec3())
    };
    (0..segments)
        .map(|segment| {
            let a = start + sweep * segment as f64 / segments as f64;
            let b = start + sweep * (segment + 1) as f64 / segments as f64;
            let radial = |radius: f64, angle: f64, y: f64| {
                transform(DVec3::new(radius * angle.cos(), y, radius * angle.sin()))
            };
            let mut cross = if inner > EPSILON {
                vec![
                    radial(inner, a, -half_length),
                    radial(outer, a, -half_length),
                    radial(outer, b, -half_length),
                    radial(inner, b, -half_length),
                ]
            } else {
                vec![
                    transform(DVec3::new(0.0, -half_length, 0.0)),
                    radial(outer, a, -half_length),
                    radial(outer, b, -half_length),
                ]
            };
            let bottom = cross.clone();
            for point in &mut cross {
                let local = spec.pose.rotation.quaternion().inverse()
                    * (point.as_vec3() - spec.pose.translation());
                *point = transform(DVec3::new(
                    f64::from(local.x),
                    half_length,
                    f64::from(local.z),
                ));
            }
            prism_cell(bottom, cross, 0, 2)
        })
        .collect()
}

/// Divides a layered part's replayed envelope cells into material bands.
///
/// Features replay on the whole envelope first, so a chamfer or fillet cuts
/// through every layer it reaches. Each layer then claims whatever lies beyond
/// the envelope it was laid on, later layers winning. A flat layer's boundary
/// is one plane. Inside one 15-degree wedge a wall or bore boundary is a single
/// chord plane, the same polygon the wedge walls use. Either way both halves
/// of a split share an identical face and stitch away as interior.
#[expect(
    clippy::too_many_lines,
    reason = "one split per layer reads better whole"
)]
pub(super) fn partition_layers(cells: Vec<PolyCell>, spec: PartSpec) -> Vec<PolyCell> {
    const SPLIT_TOLERANCE: f64 = 1.0e-6;
    const BAND_INTERFACE_PATCH: u32 = 0x4000_0000;
    let regions = spec.layer_regions();
    if regions.is_empty() {
        return cells;
    }
    let pose = spec.pose();
    let rotation = pose.rotation.quaternion().as_dquat();
    let center = DVec3::from(pose.translation());
    let wedges = spec.as_cylinder().map(|cylinder| {
        let segments =
            f64::from(cylinder.dimensions.sweep_angle_degrees() / CYLINDER_SWEEP_STEP_DEGREES);
        let sweep = f64::from(cylinder.dimensions.sweep_angle_radians());
        (segments, sweep / segments, -sweep * 0.5)
    });
    let mut banded = Vec::with_capacity(cells.len() * (regions.len() + 1));
    for cell in cells {
        // The radial direction through the middle of this cell's wedge.
        let wedge = wedges.map(|(segments, step, start)| {
            let vertices = cell.faces.iter().flat_map(|face| face.vertices.iter());
            let count = vertices.clone().count().max(1) as f64;
            let local = rotation.inverse() * (vertices.copied().sum::<DVec3>() / count - center);
            let segment = ((local.z.atan2(local.x) - start).rem_euclid(core::f64::consts::TAU)
                / step)
                .floor()
                .clamp(0.0, segments - 1.0);
            let middle = start + step * (segment + 0.5);
            (
                rotation * DVec3::new(middle.cos(), 0.0, middle.sin()),
                (step * 0.5).cos(),
            )
        });
        let chord = |radius: f32| {
            let (radial, chord_scale) = wedge.expect("only cylinders have wall or bore layers");
            (radial, radial.dot(center) + f64::from(radius) * chord_scale)
        };
        let mut pieces = vec![cell];
        for (index, region) in regions.iter().enumerate() {
            let band = index as u8 + 1;
            // The kept side of this plane, `normal · x <= offset`, lies
            // outside the layer's region.
            let (normal, offset) = match *region {
                crate::LayerRegion::Beyond {
                    axis,
                    sign,
                    distance,
                } => {
                    let normal =
                        rotation * ([DVec3::X, DVec3::Y, DVec3::Z][axis] * f64::from(sign));
                    (normal, normal.dot(center) + f64::from(distance))
                }
                crate::LayerRegion::OutsideRadius(radius) => chord(radius),
                crate::LayerRegion::InsideRadius(radius) => {
                    let (radial, offset) = chord(radius);
                    (-radial, -offset)
                }
            };
            let key = SurfacePatchKey {
                source: TopologySource::Base,
                local: BAND_INTERFACE_PATCH + index as u32,
            };
            let outside = ClipPlane {
                normal,
                offset,
                patch: key,
                family: key,
                smoothing_group: 0,
                smooth_with: Vec::new(),
                uv_provenance: key,
            };
            let mut split = Vec::with_capacity(pieces.len() + 1);
            for piece in pieces {
                let (nearest, farthest) = piece
                    .faces
                    .iter()
                    .flat_map(|face| face.vertices.iter())
                    .map(|vertex| normal.dot(*vertex) - offset)
                    .fold(
                        (f64::INFINITY, f64::NEG_INFINITY),
                        |(low, high), distance| (low.min(distance), high.max(distance)),
                    );
                if farthest <= SPLIT_TOLERANCE {
                    split.push(piece);
                    continue;
                }
                if nearest >= -SPLIT_TOLERANCE {
                    split.push(PolyCell { band, ..piece });
                    continue;
                }
                let beyond = ClipPlane {
                    normal: -normal,
                    offset: -offset,
                    ..outside.clone()
                };
                match (
                    clip_cell(&piece, outside.clone()),
                    clip_cell(&piece, beyond),
                ) {
                    (Some(kept), Some(claimed)) => {
                        split.push(kept);
                        split.push(PolyCell { band, ..claimed });
                    }
                    (Some(_), None) => split.push(piece),
                    (None, _) => split.push(PolyCell { band, ..piece }),
                }
            }
            pieces = split;
        }
        banded.extend(pieces);
    }
    banded
}

pub(super) fn pipe_bend_cells(spec: PipeBendSpec) -> Vec<PolyCell> {
    let outer = f64::from(spec.dimensions.outer_diameter()) * 0.5;
    let inner = f64::from(spec.dimensions.inner_diameter()) * 0.5;
    let radius = f64::from(spec.dimensions.radius());
    let transform = |point: DVec3| {
        DVec3::from(spec.pose.translation())
            + DVec3::from(spec.pose.rotation.quaternion() * point.as_vec3())
    };
    let mut cells = Vec::new();
    for arc in 0..12 {
        let theta_a =
            -core::f64::consts::FRAC_PI_2 + core::f64::consts::FRAC_PI_2 * f64::from(arc) / 12.0;
        let theta_b = -core::f64::consts::FRAC_PI_2
            + core::f64::consts::FRAC_PI_2 * f64::from(arc + 1) / 12.0;
        for radial_index in 0..24 {
            let phi_a = core::f64::consts::TAU * f64::from(radial_index) / 24.0;
            let phi_b = core::f64::consts::TAU * f64::from(radial_index + 1) / 24.0;
            let point = |theta: f64, phi: f64, tube: f64| {
                let radial = DVec3::new(theta.cos(), theta.sin(), 0.0);
                transform(
                    DVec3::new(-radius, radius, 0.0)
                        + radial * (radius + tube * phi.cos())
                        + DVec3::Z * (tube * phi.sin()),
                )
            };
            let tube_inner = if inner > EPSILON { inner } else { 0.0 };
            let bottom = if inner > EPSILON {
                vec![
                    point(theta_a, phi_a, tube_inner),
                    point(theta_a, phi_a, outer),
                    point(theta_a, phi_b, outer),
                    point(theta_a, phi_b, tube_inner),
                ]
            } else {
                vec![
                    point(theta_a, phi_a, 0.0),
                    point(theta_a, phi_a, outer),
                    point(theta_a, phi_b, outer),
                ]
            };
            let top = if inner > EPSILON {
                vec![
                    point(theta_b, phi_a, tube_inner),
                    point(theta_b, phi_a, outer),
                    point(theta_b, phi_b, outer),
                    point(theta_b, phi_b, tube_inner),
                ]
            } else {
                vec![
                    point(theta_b, phi_a, 0.0),
                    point(theta_b, phi_a, outer),
                    point(theta_b, phi_b, outer),
                ]
            };
            cells.push(prism_cell(bottom, top, 0, 2));
        }
    }
    cells
}

/// One convex cell per junction ray triangle: a pyramid to the centre when
/// solid, or the shell between the bore and the outside when hollow.
pub(super) fn pipe_junction_cells(spec: crate::PipeJunctionSpec) -> Vec<PolyCell> {
    let transform = |point: DVec3| {
        DVec3::from(spec.pose.translation())
            + DVec3::from(spec.pose.rotation.quaternion() * point.as_vec3())
    };
    let patch = |(face, surface): (crate::FaceKind, crate::PipeJunctionSurface)| {
        junction_face_patch(face) + 6 * surface.index()
    };
    crate::pipe_junction::ray_triangles(spec)
        .into_iter()
        .map(|triangle| {
            let outer_patch = patch(triangle.outer_surface);
            let inner_patch = triangle
                .inner_surface
                .map_or(outer_patch, |surface| 18 + patch(surface));
            shell_cell(
                triangle.outer.map(transform).to_vec(),
                triangle.inner.map(transform).to_vec(),
                [outer_patch, inner_patch, 36],
            )
        })
        .collect()
}

pub(super) const fn junction_face_patch(face: crate::FaceKind) -> u32 {
    match face {
        crate::FaceKind::NegativeX => 0,
        crate::FaceKind::PositiveX => 1,
        crate::FaceKind::NegativeY => 2,
        crate::FaceKind::PositiveY => 3,
        crate::FaceKind::NegativeZ => 4,
        crate::FaceKind::PositiveZ => 5,
    }
}

/// Builds a convex cell between an outer and an inner polygon with matching
/// vertex order. Coincident vertices collapse prisms into wedges or pyramids.
/// Patches are `[outer, inner, sides]`.
pub(super) fn shell_cell(
    mut outer: Vec<DVec3>,
    mut inner: Vec<DVec3>,
    patches: [u32; 3],
) -> PolyCell {
    if polygon_normal(&outer).dot(inner[0] - outer[0]) > 0.0 {
        outer.reverse();
        inner.reverse();
    }
    let mut inner_face = inner.clone();
    inner_face.reverse();
    let mut faces = Vec::with_capacity(outer.len() + 2);
    for (vertices, patch) in [(outer.clone(), patches[0]), (inner_face, patches[1])] {
        let vertices = without_repeated_vertices(vertices);
        if vertices.len() >= 3 {
            faces.push(base_face(vertices, patch));
        }
    }
    for index in 0..outer.len() {
        let next = (index + 1) % outer.len();
        let face =
            without_repeated_vertices(vec![outer[next], outer[index], inner[index], inner[next]]);
        if face.len() >= 3 {
            faces.push(base_face(face, patches[2]));
        }
    }
    PolyCell { faces, band: 0 }
}

pub(super) fn prism_cell(
    mut bottom: Vec<DVec3>,
    mut top: Vec<DVec3>,
    cap_patch: u32,
    side_patch: u32,
) -> PolyCell {
    if polygon_normal(&bottom).dot(top[0] - bottom[0]) > 0.0 {
        bottom.reverse();
        top.reverse();
    }
    let mut top_face = top.clone();
    top_face.reverse();
    let mut faces = vec![
        base_face(bottom.clone(), cap_patch),
        base_face(top_face, cap_patch + 1),
    ];
    for index in 0..bottom.len() {
        let next = (index + 1) % bottom.len();
        let local = if index == 1 {
            side_patch
        } else if index + 1 == bottom.len() {
            side_patch + 1
        } else {
            side_patch + 2 + index as u32
        };
        let face =
            without_repeated_vertices(vec![bottom[next], bottom[index], top[index], top[next]]);
        if face.len() >= 3 {
            faces.push(base_face(face, local));
        }
    }
    PolyCell { faces, band: 0 }
}
