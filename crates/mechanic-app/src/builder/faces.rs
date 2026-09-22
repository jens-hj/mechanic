//! Face outlines in world space and the overlap tests between them.

use super::grid::{cardinal_axis, snap_cardinal};
use super::{CONTACT_EPSILON, Vec, Vec2, Vec3, vec};
use mechanic_core::{
    ConstructionGraph, CuboidSpec, CylinderSpec, FaceKind, FaceOwner, FaceRef, PartSpec,
    PipeBendSpec, PipeJunctionSpec,
};

#[derive(Clone, Debug)]
pub(crate) struct FaceGeometry {
    pub(crate) center: Vec3,
    pub(crate) normal: Vec3,
    pub(crate) tangent_u: Vec3,
    pub(crate) tangent_v: Vec3,
    pub(super) profile: FaceProfile,
}

#[derive(Clone, Debug)]
pub(super) enum FaceProfile {
    Rectangle {
        half_u: f32,
        half_v: f32,
    },
    Annulus {
        inner_radius: f32,
        outer_radius: f32,
    },
    AnnularSector {
        inner_radius: f32,
        outer_radius: f32,
        half_angle: f32,
    },
    Polygon {
        vertices: Vec<Vec2>,
    },
    Ground,
}

pub(super) fn faces_share_plane_and_normal(first: &FaceGeometry, second: &FaceGeometry) -> bool {
    first.normal.dot(second.normal) > 1.0 - CONTACT_EPSILON
        && (first.center - second.center).dot(first.normal).abs() <= CONTACT_EPSILON
}

pub(crate) fn face_geometry_from_ref(
    face: FaceRef,
    graph: Option<&ConstructionGraph>,
) -> FaceGeometry {
    try_face_geometry_from_ref(face, graph).expect("face reference must expose flat geometry")
}

pub(crate) fn try_face_geometry_from_ref(
    face: FaceRef,
    graph: Option<&ConstructionGraph>,
) -> Option<FaceGeometry> {
    try_face_geometries_from_ref(face, graph).into_iter().next()
}

pub(super) fn try_face_geometries_from_ref(
    face: FaceRef,
    graph: Option<&ConstructionGraph>,
) -> Vec<FaceGeometry> {
    match face.owner {
        FaceOwner::Ground => vec![FaceGeometry {
            center: Vec3::ZERO,
            normal: Vec3::Y,
            tangent_u: Vec3::X,
            tangent_v: Vec3::Z,
            profile: FaceProfile::Ground,
        }],
        FaceOwner::Part(part) => {
            let graph = graph.expect("live face references have a graph");
            let spec = graph
                .part(part)
                .copied()
                .expect("live face references have a part");
            let owner = graph.region_of(part).map_or(
                mechanic_core::SolidOwner::Part(part),
                mechanic_core::SolidOwner::Region,
            );
            let patch = face.patch.or_else(|| {
                graph
                    .owner_has_shape_features(owner)
                    .then(|| primitive_surface_patch(spec, face.face))
            });
            let Some(patch) = patch else {
                let frame = graph
                    .part_frame(part)
                    .expect("live parts have construction frames");
                return part_face_geometry(spec, face.face)
                    .map(|mut geometry| {
                        geometry.center = frame.point(geometry.center);
                        geometry.normal = frame.vector(geometry.normal);
                        geometry.tangent_u = frame.vector(geometry.tangent_u);
                        geometry.tangent_v = frame.vector(geometry.tangent_v);
                        geometry
                    })
                    .into_iter()
                    .collect();
            };
            let Ok(solid) = graph.evaluated_solid_shared(owner) else {
                return Vec::new();
            };
            solid
                .surfaces
                .iter()
                .filter(|surface| surface.key == patch)
                .filter_map(|surface| evaluated_surface_geometry(&solid, surface))
                .collect()
        }
    }
}

pub(crate) const fn primitive_surface_patch(
    spec: PartSpec,
    face: FaceKind,
) -> mechanic_core::SurfacePatchKey {
    let local = match spec {
        PartSpec::Cylinder(_) => match face {
            FaceKind::NegativeY => 0,
            FaceKind::PositiveY => 1,
            _ => u32::MAX,
        },
        PartSpec::PipeBend(_) => match face {
            FaceKind::NegativeX => 0,
            FaceKind::PositiveY => 1,
            _ => u32::MAX,
        },
        _ => match face {
            FaceKind::NegativeX => 0,
            FaceKind::PositiveX => 1,
            FaceKind::NegativeY => 2,
            FaceKind::PositiveY => 3,
            FaceKind::NegativeZ => 4,
            FaceKind::PositiveZ => 5,
        },
    };
    mechanic_core::SurfacePatchKey {
        source: mechanic_core::TopologySource::Base,
        local,
    }
}

pub(super) fn evaluated_surface_geometry(
    solid: &mechanic_core::EvaluatedSolid,
    surface: &mechanic_core::SurfacePatch,
) -> Option<FaceGeometry> {
    let mut points = Vec::new();
    let mut edge = surface.half_edge;
    loop {
        let half_edge = *solid.half_edges.get(edge as usize)?;
        points.push(solid.vertices.get(half_edge.origin as usize)?.position);
        edge = half_edge.next;
        if edge == surface.half_edge {
            break;
        }
    }
    if points.len() < 3 {
        return None;
    }
    let count = f32::from(u16::try_from(points.len()).ok()?);
    let center = points.iter().copied().sum::<Vec3>() / count;
    let normal = surface.normal.normalize_or_zero();
    let tangent_u = normal.any_orthonormal_vector();
    let tangent_v = normal.cross(tangent_u);
    let vertices = points
        .into_iter()
        .map(|point| {
            let offset = point - center;
            Vec2::new(offset.dot(tangent_u), offset.dot(tangent_v))
        })
        .collect();
    Some(FaceGeometry {
        center,
        normal,
        tangent_u,
        tangent_v,
        profile: FaceProfile::Polygon { vertices },
    })
}

pub(crate) fn face_geometry(spec: CuboidSpec, face: FaceKind) -> FaceGeometry {
    envelope_face_geometry(spec.pose, spec.size_meters(), face)
}

pub(crate) fn envelope_face_geometry(
    pose: mechanic_core::BuildPose,
    size: Vec3,
    face: FaceKind,
) -> FaceGeometry {
    let rotation = pose.rotation.quaternion();
    let (normal, tangent_u, tangent_v, normal_extent, half_u, half_v) = match face {
        FaceKind::PositiveX => (Vec3::X, Vec3::Y, Vec3::Z, size.x, size.y, size.z),
        FaceKind::NegativeX => (-Vec3::X, Vec3::Y, Vec3::Z, size.x, size.y, size.z),
        FaceKind::PositiveY => (Vec3::Y, Vec3::X, Vec3::Z, size.y, size.x, size.z),
        FaceKind::NegativeY => (-Vec3::Y, Vec3::X, Vec3::Z, size.y, size.x, size.z),
        FaceKind::PositiveZ => (Vec3::Z, Vec3::X, Vec3::Y, size.z, size.x, size.y),
        FaceKind::NegativeZ => (-Vec3::Z, Vec3::X, Vec3::Y, size.z, size.x, size.y),
    };
    let normal = snap_cardinal(rotation * normal);
    FaceGeometry {
        center: pose.translation() + normal * normal_extent * 0.5,
        normal,
        tangent_u: snap_cardinal(rotation * tangent_u),
        tangent_v: snap_cardinal(rotation * tangent_v),
        profile: FaceProfile::Rectangle {
            half_u: half_u * 0.5,
            half_v: half_v * 0.5,
        },
    }
}

pub(super) fn cylinder_face_geometry(spec: CylinderSpec, face: FaceKind) -> Option<FaceGeometry> {
    if !matches!(face, FaceKind::PositiveY | FaceKind::NegativeY) {
        return None;
    }
    let rotation = spec.pose.rotation.quaternion();
    let local_normal = if face == FaceKind::PositiveY {
        Vec3::Y
    } else {
        Vec3::NEG_Y
    };
    let normal = snap_cardinal(rotation * local_normal);
    let profile = if spec.dimensions.sweep_angle_degrees() == 360 {
        FaceProfile::Annulus {
            inner_radius: spec.dimensions.inner_diameter() * 0.5,
            outer_radius: spec.dimensions.outer_diameter() * 0.5,
        }
    } else {
        FaceProfile::AnnularSector {
            inner_radius: spec.dimensions.inner_diameter() * 0.5,
            outer_radius: spec.dimensions.outer_diameter() * 0.5,
            half_angle: spec.dimensions.sweep_angle_radians() * 0.5,
        }
    };
    Some(FaceGeometry {
        center: spec.pose.translation() + normal * spec.dimensions.axial_length() * 0.5,
        normal,
        tangent_u: snap_cardinal(rotation * Vec3::X),
        tangent_v: snap_cardinal(rotation * Vec3::Z),
        profile,
    })
}

pub(super) fn pipe_bend_face_geometry(spec: PipeBendSpec, face: FaceKind) -> Option<FaceGeometry> {
    let radius = spec.dimensions.radius();
    let (local_center, local_normal, local_u, local_v) = match face {
        FaceKind::NegativeX => (Vec3::new(-radius, 0.0, 0.0), Vec3::NEG_X, Vec3::Y, Vec3::Z),
        FaceKind::PositiveY => (Vec3::new(0.0, radius, 0.0), Vec3::Y, Vec3::X, Vec3::Z),
        _ => return None,
    };
    let rotation = spec.pose.rotation.quaternion();
    Some(FaceGeometry {
        center: spec.pose.translation() + rotation * local_center,
        normal: snap_cardinal(rotation * local_normal),
        tangent_u: snap_cardinal(rotation * local_u),
        tangent_v: snap_cardinal(rotation * local_v),
        profile: FaceProfile::Annulus {
            inner_radius: spec.dimensions.inner_diameter() * 0.5,
            outer_radius: spec.dimensions.outer_diameter() * 0.5,
        },
    })
}

pub(super) fn part_face_geometry(spec: PartSpec, face: FaceKind) -> Option<FaceGeometry> {
    match spec {
        PartSpec::Cuboid(spec) => Some(face_geometry(spec, face)),
        PartSpec::Controller(spec) => Some(face_geometry(spec.cuboid(), face)),
        PartSpec::Engine(spec) => Some(face_geometry(spec.cuboid(), face)),
        PartSpec::Transmission(spec) => Some(face_geometry(spec.cuboid(), face)),
        PartSpec::Servo(spec) => Some(face_geometry(spec.cuboid(), face)),
        PartSpec::Seat(spec) => Some(face_geometry(spec.cuboid(), face)),
        spec @ (PartSpec::Dial(_) | PartSpec::Button(_)) => Some(envelope_face_geometry(
            spec.pose(),
            spec.size_meters(),
            face,
        )),
        PartSpec::Input(spec) => Some(face_geometry(spec.cuboid(), face)),
        PartSpec::DimensionLink(spec) => Some(face_geometry(spec.cuboid(), face)),
        PartSpec::Cylinder(spec) => cylinder_face_geometry(spec, face),
        PartSpec::PipeBend(spec) => pipe_bend_face_geometry(spec, face),
        PartSpec::PipeJunction(spec) => pipe_junction_face_geometry(spec, face),
    }
}

/// Open arm end of a junction; closed faces are not connection faces.
pub(super) fn pipe_junction_face_geometry(
    spec: PipeJunctionSpec,
    face: FaceKind,
) -> Option<FaceGeometry> {
    if !spec.arms.contains(face) {
        return None;
    }
    let rotation = spec.pose.rotation.quaternion();
    let (tangent_u, tangent_v) = match face {
        FaceKind::PositiveX | FaceKind::NegativeX => (Vec3::Y, Vec3::Z),
        FaceKind::PositiveY | FaceKind::NegativeY => (Vec3::X, Vec3::Z),
        FaceKind::PositiveZ | FaceKind::NegativeZ => (Vec3::X, Vec3::Y),
    };
    let normal = snap_cardinal(rotation * face_normal(face));
    Some(FaceGeometry {
        center: spec.pose.translation() + normal * spec.dimensions.half_side(),
        normal,
        tangent_u: snap_cardinal(rotation * tangent_u),
        tangent_v: snap_cardinal(rotation * tangent_v),
        profile: FaceProfile::Annulus {
            inner_radius: spec.dimensions.inner_diameter() * 0.5,
            outer_radius: spec.dimensions.outer_diameter() * 0.5,
        },
    })
}

/// Whether something can be mounted on this face.
///
/// Only genuinely flat surfaces take a new part: a shaped face is no longer an
/// axis-aligned rectangle, so a grid-aligned block could not sit flush on it.
/// Flattening the face back onto the grid makes it placeable again.
pub(crate) fn face_is_flat(graph: &ConstructionGraph, face: FaceRef) -> bool {
    let FaceOwner::Part(part) = face.owner else {
        return true;
    };
    if face
        .patch
        .is_some_and(|patch| matches!(patch.source, mechanic_core::TopologySource::Feature(_)))
    {
        return false;
    }
    let Some(id) = graph.region_of(part) else {
        if face.patch.is_none() {
            return true;
        }
        let Some(spec) = graph.part(part).copied() else {
            return false;
        };
        let normal = face_normal(face.face);
        return match spec {
            PartSpec::Cuboid(_) => true,
            PartSpec::PipeJunction(spec) => spec.arms.faces().any(|arm| {
                normal.dot(spec.pose.rotation.quaternion() * face_normal(arm))
                    >= 1.0 - CONTACT_EPSILON
            }),
            PartSpec::Cylinder(spec) => {
                normal.dot(spec.pose.rotation.quaternion() * Vec3::Y).abs() >= 1.0 - CONTACT_EPSILON
            }
            PartSpec::PipeBend(spec) => {
                let rotation = spec.pose.rotation.quaternion();
                [rotation * Vec3::NEG_X, rotation * Vec3::Y]
                    .into_iter()
                    .any(|end| normal.dot(end) >= 1.0 - CONTACT_EPSILON)
            }
            PartSpec::Controller(_)
            | PartSpec::Engine(_)
            | PartSpec::Transmission(_)
            | PartSpec::Servo(_)
            | PartSpec::Seat(_)
            | PartSpec::Dial(_)
            | PartSpec::Button(_)
            | PartSpec::Input(_)
            | PartSpec::DimensionLink(_) => false,
        };
    };
    let Some(region) = graph.region(id) else {
        return true;
    };
    let Some(spec) = graph.part(part).and_then(|spec| spec.as_cuboid()) else {
        return true;
    };
    // The face is named in the part's own frame, so rotate it into the world
    // before asking the region, whose cage is world-aligned.
    let normal = spec.pose.rotation.quaternion() * face_normal(face.face);
    let (axis, sign) = cardinal_axis(normal);
    region.face_is_flat(axis, sign > 0)
}

/// Outward normal of one local face.
pub(super) const fn face_normal(face: FaceKind) -> Vec3 {
    match face {
        FaceKind::PositiveX => Vec3::X,
        FaceKind::NegativeX => Vec3::NEG_X,
        FaceKind::PositiveY => Vec3::Y,
        FaceKind::NegativeY => Vec3::NEG_Y,
        FaceKind::PositiveZ => Vec3::Z,
        FaceKind::NegativeZ => Vec3::NEG_Z,
    }
}

pub(super) fn overlap_center(first: &FaceGeometry, second: &FaceGeometry) -> Option<Vec3> {
    if first.normal.dot(second.normal) > -1.0 + CONTACT_EPSILON
        || (first.center - second.center).dot(first.normal).abs() > CONTACT_EPSILON
    {
        return None;
    }
    profiles_overlap(first, second).then_some((first.center + second.center) * 0.5)
}

pub(super) fn point_in_profile(u: f32, v: f32, profile: &FaceProfile) -> bool {
    match profile {
        FaceProfile::Rectangle { half_u, half_v } => {
            u.abs() <= *half_u + CONTACT_EPSILON && v.abs() <= *half_v + CONTACT_EPSILON
        }
        FaceProfile::Annulus {
            inner_radius,
            outer_radius,
        } => {
            let squared = u.mul_add(u, v * v);
            squared >= (*inner_radius - CONTACT_EPSILON).max(0.0).powi(2)
                && squared <= (*outer_radius + CONTACT_EPSILON).powi(2)
        }
        FaceProfile::AnnularSector {
            inner_radius,
            outer_radius,
            half_angle,
        } => {
            let squared = u.mul_add(u, v * v);
            squared >= (*inner_radius - CONTACT_EPSILON).max(0.0).powi(2)
                && squared <= (*outer_radius + CONTACT_EPSILON).powi(2)
                && v.atan2(u).abs() <= *half_angle + CONTACT_EPSILON
        }
        FaceProfile::Polygon { vertices } => point_in_convex_polygon(Vec2::new(u, v), vertices),
        FaceProfile::Ground => true,
    }
}

pub(super) fn profiles_overlap(first: &FaceGeometry, second: &FaceGeometry) -> bool {
    match (&first.profile, &second.profile) {
        (FaceProfile::Ground, _) | (_, FaceProfile::Ground) => true,
        (FaceProfile::Rectangle { half_u, half_v }, FaceProfile::Rectangle { .. }) => {
            positive_rect_overlap(first, second, first.tangent_u, *half_u)
                && positive_rect_overlap(first, second, first.tangent_v, *half_v)
        }
        (FaceProfile::Annulus { .. }, FaceProfile::Rectangle { .. }) => {
            annulus_rectangle_overlap(first, second)
        }
        (FaceProfile::Rectangle { .. }, FaceProfile::Annulus { .. }) => {
            annulus_rectangle_overlap(second, first)
        }
        (FaceProfile::Annulus { .. }, FaceProfile::Annulus { .. }) => {
            let FaceProfile::Annulus {
                inner_radius: inner_a,
                outer_radius: outer_a,
            } = &first.profile
            else {
                unreachable!()
            };
            let FaceProfile::Annulus {
                inner_radius: inner_b,
                outer_radius: outer_b,
            } = &second.profile
            else {
                unreachable!()
            };
            let offset = second.center - first.center;
            let distance =
                Vec2::new(offset.dot(first.tangent_u), offset.dot(first.tangent_v)).length();
            distance < *outer_a + *outer_b - CONTACT_EPSILON
                && distance + *outer_a > *inner_b + CONTACT_EPSILON
                && distance + *outer_b > *inner_a + CONTACT_EPSILON
        }
        (FaceProfile::AnnularSector { .. } | FaceProfile::Polygon { .. }, _)
        | (_, FaceProfile::AnnularSector { .. } | FaceProfile::Polygon { .. }) => {
            sector_profiles_overlap(first, second)
        }
    }
}

pub(super) fn sector_profiles_overlap(first: &FaceGeometry, second: &FaceGeometry) -> bool {
    let first_cells = profile_cells(first, first.center, first.tangent_u, first.tangent_v);
    let second_cells = profile_cells(second, first.center, first.tangent_u, first.tangent_v);
    first_cells.iter().any(|first| {
        second_cells
            .iter()
            .any(|second| convex_polygons_overlap(first, second))
    })
}

pub(super) fn profile_cells(
    face: &FaceGeometry,
    origin: Vec3,
    plane_u: Vec3,
    plane_v: Vec3,
) -> Vec<Vec<Vec2>> {
    let project = |point: Vec3| {
        let offset = point - origin;
        Vec2::new(offset.dot(plane_u), offset.dot(plane_v))
    };
    match &face.profile {
        FaceProfile::Rectangle { half_u, half_v } => vec![vec![
            project(face.center - face.tangent_u * *half_u - face.tangent_v * *half_v),
            project(face.center + face.tangent_u * *half_u - face.tangent_v * *half_v),
            project(face.center + face.tangent_u * *half_u + face.tangent_v * *half_v),
            project(face.center - face.tangent_u * *half_u + face.tangent_v * *half_v),
        ]],
        FaceProfile::Annulus {
            inner_radius,
            outer_radius,
        } => annular_profile_cells(
            face,
            origin,
            plane_u,
            plane_v,
            *inner_radius,
            *outer_radius,
            std::f32::consts::PI,
        ),
        FaceProfile::AnnularSector {
            inner_radius,
            outer_radius,
            half_angle,
        } => annular_profile_cells(
            face,
            origin,
            plane_u,
            plane_v,
            *inner_radius,
            *outer_radius,
            *half_angle,
        ),
        FaceProfile::Polygon { vertices } => vec![
            vertices
                .iter()
                .map(|vertex| {
                    project(face.center + face.tangent_u * vertex.x + face.tangent_v * vertex.y)
                })
                .collect(),
        ],
        FaceProfile::Ground => Vec::new(),
    }
}

pub(super) fn annular_profile_cells(
    face: &FaceGeometry,
    origin: Vec3,
    plane_u: Vec3,
    plane_v: Vec3,
    inner_radius: f32,
    outer_radius: f32,
    half_angle: f32,
) -> Vec<Vec<Vec2>> {
    let sweep = half_angle * 2.0;
    let segment_count = (1_u16..=24)
        .find(|&count| (f32::from(count) * (std::f32::consts::PI / 12.0) - sweep).abs() < 1.0e-4)
        .expect("annular profiles use 15-degree increments");
    let project = |point: Vec3| {
        let offset = point - origin;
        Vec2::new(offset.dot(plane_u), offset.dot(plane_v))
    };
    (0..segment_count)
        .map(|segment| {
            let first_angle = -half_angle + sweep * f32::from(segment) / f32::from(segment_count);
            let second_angle =
                -half_angle + sweep * f32::from(segment + 1) / f32::from(segment_count);
            let radial = |angle: f32| face.tangent_u * angle.cos() + face.tangent_v * angle.sin();
            let outer_first = project(face.center + radial(first_angle) * outer_radius);
            let outer_second = project(face.center + radial(second_angle) * outer_radius);
            if inner_radius == 0.0 {
                vec![project(face.center), outer_first, outer_second]
            } else {
                vec![
                    project(face.center + radial(first_angle) * inner_radius),
                    outer_first,
                    outer_second,
                    project(face.center + radial(second_angle) * inner_radius),
                ]
            }
        })
        .collect()
}

pub(super) fn point_in_convex_polygon(point: Vec2, vertices: &[Vec2]) -> bool {
    if vertices.len() < 3 {
        return false;
    }
    let mut sign = 0.0_f32;
    for index in 0..vertices.len() {
        let edge = vertices[(index + 1) % vertices.len()] - vertices[index];
        let cross = edge.perp_dot(point - vertices[index]);
        if cross.abs() <= CONTACT_EPSILON {
            continue;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if sign * cross < 0.0 {
            return false;
        }
    }
    true
}

pub(super) fn convex_polygons_overlap(first: &[Vec2], second: &[Vec2]) -> bool {
    first
        .iter()
        .zip(first.iter().cycle().skip(1))
        .chain(second.iter().zip(second.iter().cycle().skip(1)))
        .all(|(start, end)| {
            let edge = *end - *start;
            let axis = Vec2::new(-edge.y, edge.x).normalize();
            let project = |polygon: &[Vec2]| {
                polygon.iter().fold(
                    (f32::INFINITY, f32::NEG_INFINITY),
                    |(minimum, maximum), point| {
                        let value = point.dot(axis);
                        (minimum.min(value), maximum.max(value))
                    },
                )
            };
            let (first_minimum, first_maximum) = project(first);
            let (second_minimum, second_maximum) = project(second);
            first_maximum.min(second_maximum) - first_minimum.max(second_minimum) > CONTACT_EPSILON
        })
}

pub(super) fn positive_rect_overlap(
    first: &FaceGeometry,
    second: &FaceGeometry,
    axis: Vec3,
    first_half: f32,
) -> bool {
    let FaceProfile::Rectangle { half_u, half_v } = &second.profile else {
        unreachable!()
    };
    let second_half =
        second.tangent_u.dot(axis).abs() * *half_u + second.tangent_v.dot(axis).abs() * *half_v;
    first_half + second_half - (second.center - first.center).dot(axis).abs() > CONTACT_EPSILON
}

pub(super) fn annulus_rectangle_overlap(annulus: &FaceGeometry, rectangle: &FaceGeometry) -> bool {
    let FaceProfile::Annulus {
        inner_radius,
        outer_radius,
    } = &annulus.profile
    else {
        unreachable!()
    };
    let FaceProfile::Rectangle { half_u, half_v } = &rectangle.profile else {
        unreachable!()
    };
    let offset = annulus.center - rectangle.center;
    let center_u = offset.dot(rectangle.tangent_u).abs();
    let center_v = offset.dot(rectangle.tangent_v).abs();
    let nearest_u = (center_u - *half_u).max(0.0);
    let nearest_v = (center_v - *half_v).max(0.0);
    let nearest_squared = nearest_u.mul_add(nearest_u, nearest_v * nearest_v);
    let farthest_u = center_u + *half_u;
    let farthest_v = center_v + *half_v;
    let farthest_squared = farthest_u.mul_add(farthest_u, farthest_v * farthest_v);
    nearest_squared < (*outer_radius - CONTACT_EPSILON).max(0.0).powi(2)
        && farthest_squared > (*inner_radius + CONTACT_EPSILON).powi(2)
}
