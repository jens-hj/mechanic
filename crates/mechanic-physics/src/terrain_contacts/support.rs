//! Contact points on a surface and their reduction to a stable support set.

use super::QueryKind;
use super::model::{PAIR_ACTIVATION_DISTANCE, TerrainContact};
use crate::PhysicsError;
use bevy_math::DVec3;
use mechanic_core::{
    ContactCylinder, ContactPolytope, ConvexFeature, ConvexSeparation, TriangleContactPoint,
};

// Finite points of one convex against one triangle for the requested query.
pub(super) fn surface_points(
    shape: &ContactPolytope,
    triangle: [DVec3; 3],
    kind: QueryKind,
    margin: f64,
    window: f64,
) -> Result<Vec<TriangleContactPoint>, PhysicsError> {
    surface_points_with_scratch(
        shape,
        triangle,
        kind,
        margin,
        window,
        &mut mechanic_core::TriangleClipScratch::default(),
    )
}

pub(super) fn surface_points_with_scratch(
    shape: &ContactPolytope,
    triangle: [DVec3; 3],
    kind: QueryKind,
    margin: f64,
    window: f64,
    scratch: &mut mechanic_core::TriangleClipScratch,
) -> Result<Vec<TriangleContactPoint>, PhysicsError> {
    match kind {
        QueryKind::Activation => activation_points_with_scratch(shape, triangle, window, scratch),
        QueryKind::Surface => shape
            .triangle_proximity_with_scratch(triangle, margin, scratch)
            .map_err(|_| PhysicsError::InvalidCollision),
        QueryKind::BuriedVertices => shape
            .triangle_buried_vertices_with_scratch(triangle, scratch)
            .map_err(|_| PhysicsError::InvalidCollision),
        QueryKind::Recovery => {
            let points = shape
                .triangle_recovery_contacts_with_scratch(triangle, scratch)
                .map_err(|_| PhysicsError::InvalidCollision)?;
            if points.is_empty() {
                activation_points_with_scratch(shape, triangle, window, scratch)
            } else {
                Ok(points)
            }
        }
    }
}

// Points between two colliders from the feature realizing their separating
// axis, as (receiving collider, opposing collider, points). A face clips the
// other solid against that face's triangles, so its outward normal pushes the
// other collider away; crossed edges touch at their single closest pair.
pub(super) fn pair_points(
    shapes: [&ContactPolytope; 2],
    [first, second]: [usize; 2],
    separation: ConvexSeparation,
    kind: QueryKind,
    margin: f64,
) -> Result<(usize, usize, Vec<TriangleContactPoint>), PhysicsError> {
    let (receiving, opposing, plane) = match separation.feature {
        ConvexFeature::OtherFace(plane) => (first, second, plane),
        ConvexFeature::OwnFace(plane) => (second, first, plane),
        ConvexFeature::Edges([own, other]) => {
            let point = TriangleContactPoint {
                triangle_point: other,
                body_point: own,
                normal: separation.axis,
                depth: (-separation.separation).max(0.0),
            };
            return Ok((first, second, vec![point]));
        }
    };
    let shape = |row| shapes[usize::from(row != first)];
    let mut points = Vec::new();
    for triangle in shape(opposing)
        .face_triangles(plane)
        .map_err(|_| PhysicsError::InvalidCollision)?
    {
        points.extend(surface_points(
            shape(receiving),
            triangle,
            kind,
            margin,
            PAIR_ACTIVATION_DISTANCE,
        )?);
    }
    Ok((receiving, opposing, points))
}

pub(super) struct SupportGroup {
    pub(super) normal: DVec3,
    pub(super) distance: f64,
    pub(super) response: [f64; 4],
    pub(super) directions: [DVec3; 5],
    pub(super) supports: [TerrainContact; 5],
    pub(super) curved: bool,
    // A cylinder's deepest point, which a tipped cap's lowest rim point can be
    // without winning any corner.
    pub(super) deepest: Option<TerrainContact>,
}

impl SupportGroup {
    pub(super) fn append_unique(self, manifold: usize, output: &mut Vec<TerrainContact>) {
        // Points along a level line are one depth; only a clearly deeper point
        // adds a row.
        const DEEPER: f64 = 1e-6;
        for corner in 0..if self.curved { 5 } else { 4 } {
            let mut support = self.supports[corner];
            support.manifold = manifold;
            if self.supports[..corner]
                .iter()
                .all(|previous| previous.terrain_point.distance(support.terrain_point) >= 1e-5)
            {
                output.push(support);
            }
        }
        let corners = &self.supports[..if self.curved { 5 } else { 4 }];
        if let Some(mut deepest) = self.deepest
            && corners
                .iter()
                .all(|support| deepest.separation < support.separation - DEEPER)
        {
            deepest.manifold = manifold;
            output.push(deepest);
        }
    }
}

// The receiving solid, which names the crown direction of a new support group.
#[derive(Clone, Copy)]
pub(super) enum Opposing<'a> {
    Polytope(&'a ContactPolytope),
    Cylinder(&'a ContactCylinder),
}

impl Opposing<'_> {
    pub(super) fn normal(self, surface_normal: DVec3) -> DVec3 {
        match self {
            Self::Polytope(shape) => shape.opposing_normal(surface_normal),
            Self::Cylinder(cylinder) => cylinder.opposing_normal(surface_normal),
        }
    }
}

// A cylinder side touches a surface along its lowest line, but a triangle beside
// that line still reports its own nearest point higher up the flank. Such a
// point would win a group corner and hold the wheel ahead of or behind its
// axle, braking it. Keep a flank point only where no lowest-line point of its
// group lies at or below it, as at a kerb or in a crease.
pub(super) fn rolling_supports(
    points: &[TerrainContact],
    cylinder: &ContactCylinder,
    center: DVec3,
) -> Vec<TerrainContact> {
    let on_line = points
        .iter()
        .map(|point| on_lowest_line(cylinder, point.normal, point.body_point))
        .collect::<Vec<_>>();
    points
        .iter()
        .zip(&on_line)
        .filter(|&(point, &line)| {
            line || !points.iter().zip(&on_line).any(|(other, &other_line)| {
                other_line
                    && other.separation <= point.separation
                    && same_support(Surface::of(other), Surface::of(point), center).is_some()
            })
        })
        .map(|(point, _)| *point)
        .collect()
}

// Whether a point lies on a cylinder side's lowest line toward a surface, or
// the cylinder stands on an end.
pub(super) fn on_lowest_line(cylinder: &ContactCylinder, normal: DVec3, point: DVec3) -> bool {
    const ON_LINE: f64 = 1e-6;
    cylinder
        .lowest_line_distance(normal, point)
        .is_none_or(|distance| distance <= ON_LINE)
}

// A contact's surface plane and response.
#[derive(Clone, Copy)]
pub(super) struct Surface {
    pub(super) normal: DVec3,
    pub(super) distance: f64,
    pub(super) response: [f64; 4],
}

impl Surface {
    pub(super) fn of(contact: &TerrainContact) -> Self {
        Self {
            normal: contact.normal,
            distance: contact.normal.dot(contact.terrain_point),
            response: contact.response,
        }
    }
}

// Whether a contact on `surface` joins the support group of `group`, and if so
// whether only because the surface curves.
pub(super) fn same_support(group: Surface, surface: Surface, center: DVec3) -> Option<bool> {
    let (normal, distance) = (group.normal, group.distance);
    let own = surface.distance;
    let parallel = (normal - surface.normal).abs().max_element() < 1e-6;
    let separation = ((surface.normal - normal).dot(center) - own + distance).abs();
    let nearby = !parallel && normal.dot(surface.normal) > 0.995 && separation < 0.025;
    (((parallel && (own - distance).abs() < 1e-5) || nearby)
        && group.response.map(f64::to_bits) == surface.response.map(f64::to_bits))
    .then_some(nearby)
}

pub(super) fn reduce_support(
    groups: &mut Vec<SupportGroup>,
    overflow: &mut Vec<TerrainContact>,
    mut contact: TerrainContact,
    shape: Opposing<'_>,
    center: DVec3,
) {
    let distance = contact.normal.dot(contact.terrain_point);
    for group in groups.iter_mut() {
        let own = Surface {
            normal: group.normal,
            distance: group.distance,
            response: group.response,
        };
        if let Some(nearby) = same_support(own, Surface::of(&contact), center) {
            group.curved |= nearby;
            if let Some(deepest) = &mut group.deepest
                && contact.separation < deepest.separation
            {
                *deepest = contact;
            }
            for (support, direction) in group.supports.iter_mut().zip(group.directions) {
                if contact.terrain_point.dot(direction) > support.terrain_point.dot(direction) {
                    *support = contact;
                }
            }
            return;
        }
    }
    // Keep the existing 16-group/four-corner-plus-crown rule. Extra groups are
    // emitted unreduced; they must never silently disappear at a capacity bound.
    if groups.len() == 16 {
        contact.manifold = 16 + overflow.len();
        overflow.push(contact);
        return;
    }
    let reference = if contact.normal.y.abs() > 0.9 {
        DVec3::X
    } else {
        DVec3::Y
    };
    let u = reference.cross(contact.normal).normalize();
    let v = contact.normal.cross(u);
    groups.push(SupportGroup {
        normal: contact.normal,
        distance,
        response: contact.response,
        directions: [u + v, u - v, -u - v, -u + v, -shape.normal(contact.normal)],
        supports: [contact; 5],
        curved: false,
        deepest: matches!(shape, Opposing::Cylinder(_)).then_some(contact),
    });
}

// Existing finite intersections retain their established four-corner rule.
// Extrusion is only for a pair without an intersection, not a perturbation of
// every supporting manifold. Out-of-tolerance/invalid results still fail.
pub(super) fn activation_points_with_scratch(
    shape: &ContactPolytope,
    triangle: [DVec3; 3],
    window: f64,
    scratch: &mut mechanic_core::TriangleClipScratch,
) -> Result<Vec<mechanic_core::TriangleContactPoint>, PhysicsError> {
    let points = shape
        .triangle_activation_contacts_with_scratch(triangle, window, scratch)
        .map_err(|_| PhysicsError::NotConverged)?;
    if points.iter().any(|point| {
        let gap = (point.body_point - point.triangle_point).dot(point.normal);
        !gap.is_finite() || gap > window
    }) {
        return Err(PhysicsError::NotConverged);
    }
    Ok(points)
}
