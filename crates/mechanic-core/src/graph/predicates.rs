//! Geometric predicates over faces, profiles, and bearing rings.

use super::specs::{BearingDimensions, BearingSpec, RigidLinkSpec, WeldSpec};
use crate::geometry::{FaceGeometry, FaceProfile};
use crate::{
    ANCHOR_TOLERANCE_METERS, AXIS_TOLERANCE_DEGREES, CuboidSpec, FaceKind, FaceOwner, FaceRef,
    PartId, PartSpec,
};
use bevy_math::{Vec2, Vec3};

pub(super) fn face_references(face: FaceRef, part: PartId) -> bool {
    face.owner == FaceOwner::Part(part)
}

pub(super) const fn primitive_surface_patch(
    spec: PartSpec,
    face: FaceKind,
) -> crate::SurfacePatchKey {
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
    crate::SurfacePatchKey {
        source: crate::TopologySource::Base,
        local,
    }
}

pub(super) fn weld_references(weld: WeldSpec, part: PartId) -> bool {
    face_references(weld.first, part) || face_references(weld.second, part)
}

pub(super) fn simple_grid_faces_touch(
    first: CuboidSpec,
    first_face: FaceKind,
    second: CuboidSpec,
    second_face: FaceKind,
) -> bool {
    let (axis, first_sign) = simple_face_axis(first_face);
    let (second_axis, second_sign) = simple_face_axis(second_face);
    if axis != second_axis || first_sign == second_sign {
        return false;
    }
    let first_center = first.pose.translation_position_ticks();
    let second_center = second.pose.translation_position_ticks();
    let first_half =
        i32::from(first.dimensions[axis].units()) * crate::POSITION_TICKS_PER_HALF_GRID_UNIT;
    let second_half =
        i32::from(second.dimensions[axis].units()) * crate::POSITION_TICKS_PER_HALF_GRID_UNIT;
    if first_center[axis] + first_sign * first_half
        != second_center[axis] + second_sign * second_half
    {
        return false;
    }
    (0..3).filter(|&tangent| tangent != axis).all(|tangent| {
        let distance = (first_center[tangent] - second_center[tangent]).abs();
        let combined_half = (i32::from(first.dimensions[tangent].units())
            + i32::from(second.dimensions[tangent].units()))
            * crate::POSITION_TICKS_PER_HALF_GRID_UNIT;
        distance < combined_half
    })
}

pub(super) fn simple_grid_face_on_ground(cuboid: CuboidSpec, face: FaceKind) -> bool {
    face == FaceKind::NegativeY
        && cuboid.pose.translation_position_ticks().y
            == i32::from(cuboid.dimensions[1].units()) * crate::POSITION_TICKS_PER_HALF_GRID_UNIT
}

pub(super) const fn simple_face_axis(face: FaceKind) -> (usize, i32) {
    match face {
        FaceKind::PositiveX => (0, 1),
        FaceKind::NegativeX => (0, -1),
        FaceKind::PositiveY => (1, 1),
        FaceKind::NegativeY => (1, -1),
        FaceKind::PositiveZ => (2, 1),
        FaceKind::NegativeZ => (2, -1),
    }
}

pub(super) fn rigid_link_references(link: RigidLinkSpec, part: PartId) -> bool {
    link.first == part || link.second == part
}

pub(super) fn bearing_references(bearing: BearingSpec, part: PartId) -> bool {
    face_references(bearing.source, part) || face_references(bearing.target, part)
}

pub(super) fn point_on_face(point: Vec3, face: &FaceGeometry) -> bool {
    let offset = point - face.center;
    if offset.dot(face.normal).abs() > ANCHOR_TOLERANCE_METERS {
        return false;
    }
    point_in_profile(
        offset.dot(face.tangent_u),
        offset.dot(face.tangent_v),
        &face.profile,
    )
}

pub(super) fn bearing_ring_overlaps_face(
    anchor: Vec3,
    dimensions: BearingDimensions,
    face: &FaceGeometry,
) -> bool {
    let offset = anchor - face.center;
    if offset.dot(face.normal).abs() > ANCHOR_TOLERANCE_METERS
        || matches!(face.profile, FaceProfile::Ground)
    {
        return false;
    }
    let ring = FaceGeometry {
        center: anchor,
        normal: face.normal,
        tangent_u: face.tangent_u,
        tangent_v: face.tangent_v,
        profile: FaceProfile::Annulus {
            inner_radius: dimensions.inner_diameter() * 0.5,
            outer_radius: dimensions.outer_diameter() * 0.5,
        },
    };
    profiles_overlap(&ring, face)
}

pub(super) fn faces_touch(first: &FaceGeometry, second: &FaceGeometry) -> bool {
    if first.normal.dot(second.normal) > -1.0 + axis_cosine_tolerance() {
        return false;
    }
    let separation = (second.center - first.center).dot(first.normal).abs();
    if separation > ANCHOR_TOLERANCE_METERS {
        return false;
    }
    if matches!(&first.profile, FaceProfile::Ground)
        || matches!(&second.profile, FaceProfile::Ground)
    {
        return true;
    }
    profiles_overlap(first, second)
}

pub(super) fn point_in_profile(u: f32, v: f32, profile: &FaceProfile) -> bool {
    match profile {
        FaceProfile::Rectangle { half_u, half_v } => {
            u.abs() <= half_u + ANCHOR_TOLERANCE_METERS
                && v.abs() <= half_v + ANCHOR_TOLERANCE_METERS
        }
        FaceProfile::Annulus {
            inner_radius,
            outer_radius,
        } => {
            let radius_squared = u.mul_add(u, v * v);
            radius_squared >= (inner_radius - ANCHOR_TOLERANCE_METERS).max(0.0).powi(2)
                && radius_squared <= (outer_radius + ANCHOR_TOLERANCE_METERS).powi(2)
        }
        FaceProfile::AnnularSector {
            inner_radius,
            outer_radius,
            half_angle,
        } => {
            let radius_squared = u.mul_add(u, v * v);
            radius_squared >= (inner_radius - ANCHOR_TOLERANCE_METERS).max(0.0).powi(2)
                && radius_squared <= (outer_radius + ANCHOR_TOLERANCE_METERS).powi(2)
                && v.atan2(u).abs() <= half_angle + ANCHOR_TOLERANCE_METERS
        }
        FaceProfile::Polygon { vertices } => point_in_convex_polygon(Vec2::new(u, v), vertices),
        FaceProfile::Ground => true,
    }
}

pub(super) fn profiles_overlap(first: &FaceGeometry, second: &FaceGeometry) -> bool {
    match (&first.profile, &second.profile) {
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
        (FaceProfile::Annulus { .. }, FaceProfile::Annulus { .. }) => annuli_overlap(first, second),
        (FaceProfile::Ground, _) | (_, FaceProfile::Ground) => true,
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
            core::f32::consts::PI,
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

pub(super) fn point_in_convex_polygon(point: Vec2, vertices: &[Vec2]) -> bool {
    if vertices.len() < 3 {
        return false;
    }
    let mut sign = 0.0_f32;
    for index in 0..vertices.len() {
        let edge = vertices[(index + 1) % vertices.len()] - vertices[index];
        let offset = point - vertices[index];
        let cross = edge.perp_dot(offset);
        if cross.abs() <= ANCHOR_TOLERANCE_METERS {
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
        .find(|&count| (f32::from(count) * (core::f32::consts::PI / 12.0) - sweep).abs() < 1.0e-4)
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
            first_maximum.min(second_maximum) - first_minimum.max(second_minimum)
                > ANCHOR_TOLERANCE_METERS
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
    let centre_distance = (second.center - first.center).dot(axis).abs();
    first_half + second_half - centre_distance > ANCHOR_TOLERANCE_METERS
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
    nearest_squared < (*outer_radius - ANCHOR_TOLERANCE_METERS).max(0.0).powi(2)
        && farthest_squared > (*inner_radius + ANCHOR_TOLERANCE_METERS).powi(2)
}

pub(super) fn annuli_overlap(first: &FaceGeometry, second: &FaceGeometry) -> bool {
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
    let distance = Vec3::new(
        offset.dot(first.tangent_u),
        offset.dot(first.tangent_v),
        0.0,
    )
    .length();
    distance < *outer_a + *outer_b - ANCHOR_TOLERANCE_METERS
        && distance + *outer_a > *inner_b + ANCHOR_TOLERANCE_METERS
        && distance + *outer_b > *inner_a + ANCHOR_TOLERANCE_METERS
}

pub(super) fn axis_cosine_tolerance() -> f32 {
    1.0 - AXIS_TOLERANCE_DEGREES.to_radians().cos()
}
