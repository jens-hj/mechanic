//! Feature-constrained placement of an authored assembly.
//!
//! Callers resolve picks against evaluated topology in an unchanged graph revision.
//! All source points are authored/default points, never live articulated points.

use bevy_math::{Quat, Vec2, Vec3};
use thiserror::Error;

use crate::ConstructionFrame;

mod collision;
pub use collision::WeldCollider;
mod placement;
pub use placement::WeldPlacement;

const EPSILON: f32 = 1.0e-5;

/// Geometric incidence retained throughout a weld gesture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeldFeature {
    /// A point in a planar material patch; bounded material is validated separately.
    Face,
    /// A complete straight logical edge, with finite endpoints.
    Edge([Vec3; 2]),
    /// A real corner, excluding tessellation vertices.
    Vertex(Vec3),
}

/// An evaluated feature and the adjacent planar surface approached by the pointer.
#[derive(Clone, Copy, Debug)]
pub struct WeldSelection {
    /// Selected geometric feature in the same frame as the mating plane.
    pub feature: WeldFeature,
    /// Picked point on the feature, used to seed alignment.
    pub point: Vec3,
    /// Unit outward normal of the retained mating surface.
    pub normal: Vec3,
    /// Unit tangent of that surface.
    pub tangent: Vec3,
}

/// Reasons a feature alignment cannot be installed.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum WeldError {
    /// Non-finite, degenerate, or nonincident selection data.
    #[error("The selected feature is missing or degenerate")]
    InvalidFeature,
    /// Translation or rotation violates the selected finite features.
    #[error("Placement exceeds the selected feature boundaries")]
    Incidence,
    /// The mating material cannot contain the required square.
    #[error("Weld contact must contain a continuous 5 × 5 cm square")]
    InsufficientContact,
}

/// Allowed displacement from the initial alignment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeldConstraint {
    /// Vertex-to-vertex placement cannot slide.
    Fixed,
    /// Slides on a finite straight edge in destination space.
    Line {
        /// Unit direction of permitted motion.
        direction: Vec3,
    },
    /// Slides in the destination mating plane.
    Plane,
}

/// One rigid transform applied to every body of the default source assembly.
#[derive(Clone, Copy, Debug)]
pub struct WeldAlignment {
    source: WeldSelection,
    destination: WeldSelection,
    initial: ConstructionFrame,
    /// Translation constraint of this feature pair.
    pub constraint: WeldConstraint,
}

impl WeldSelection {
    fn validate(self) -> Result<(), WeldError> {
        if !self.point.is_finite()
            || !self.normal.is_normalized()
            || !self.tangent.is_normalized()
            || self.normal.dot(self.tangent).abs() > EPSILON
        {
            return Err(WeldError::InvalidFeature);
        }
        match self.feature {
            WeldFeature::Face => (),
            WeldFeature::Vertex(point) if point.abs_diff_eq(self.point, EPSILON) => (),
            WeldFeature::Edge([a, b])
                if a.is_finite()
                    && b.is_finite()
                    && a.distance(b) > EPSILON
                    && (b - a).normalize().dot(self.normal).abs() <= EPSILON
                    && point_on_segment(self.point, a, b) => {}
            _ => return Err(WeldError::InvalidFeature),
        }
        Ok(())
    }
}

impl WeldAlignment {
    /// Opposes mating normals and selects the smallest compatible rotation.
    ///
    /// # Errors
    /// Returns `InvalidFeature` for degenerate or nonincident selection data.
    pub fn new(source: WeldSelection, destination: WeldSelection) -> Result<Self, WeldError> {
        source.validate()?;
        destination.validate()?;
        let mut rotation = Quat::from_rotation_arc(source.normal, -destination.normal);
        if let (WeldFeature::Edge(a), WeldFeature::Edge(b)) = (source.feature, destination.feature)
        {
            let from = rotation * (a[1] - a[0]).normalize();
            let mut to = (b[1] - b[0]).normalize();
            if from.dot(to) < 0.0 {
                to = -to;
            }
            let angle = destination.normal.dot(from.cross(to)).atan2(from.dot(to));
            rotation = Quat::from_axis_angle(destination.normal, angle) * rotation;
        }
        let initial = ConstructionFrame::new(destination.point - rotation * source.point, rotation)
            .map_err(|_| WeldError::InvalidFeature)?;
        let constraint = match (source.feature, destination.feature) {
            (WeldFeature::Vertex(_), WeldFeature::Vertex(_)) => WeldConstraint::Fixed,
            (WeldFeature::Face, _) | (_, WeldFeature::Face) => WeldConstraint::Plane,
            (_, WeldFeature::Edge([a, b])) => WeldConstraint::Line {
                direction: (b - a).normalize(),
            },
            (WeldFeature::Edge([a, b]), _) => WeldConstraint::Line {
                direction: rotation * (b - a).normalize(),
            },
        };
        Ok(Self {
            source,
            destination,
            initial,
            constraint,
        })
    }

    /// Aligns the two surface grids with the nearest quarter-turn correspondence.
    /// Fixed and line-constrained feature pairs retain their required orientation.
    ///
    /// # Errors
    /// Returns `InvalidFeature` if the aligned transform is not finite.
    pub fn align_tangent_grids(mut self) -> Result<Self, WeldError> {
        if self.constraint == WeldConstraint::Plane {
            let from = self.initial.vector(self.source.tangent);
            let to = self.destination.tangent;
            let angle = self
                .destination
                .normal
                .dot(from.cross(to))
                .atan2(from.dot(to));
            let quarter = std::f32::consts::FRAC_PI_2;
            let correction = angle - (angle / quarter).round() * quarter;
            let rotation = Quat::from_axis_angle(self.destination.normal, correction)
                * self.initial.rotation();
            self.initial = ConstructionFrame::new(
                self.destination.point - rotation * self.source.point,
                rotation,
            )
            .map_err(|_| WeldError::InvalidFeature)?;
        }
        Ok(self)
    }

    /// Returns the starting transform, before drag or rotation.
    pub fn initial(self) -> ConstructionFrame {
        self.initial
    }

    /// Projects pointer displacement into the allowed translation space.
    pub fn displacement(self, value: Vec3) -> Vec3 {
        match self.constraint {
            WeldConstraint::Fixed => Vec3::ZERO,
            WeldConstraint::Line { direction } => direction * value.dot(direction),
            WeldConstraint::Plane => {
                value - self.destination.normal * value.dot(self.destination.normal)
            }
        }
    }

    /// Evaluates an endpoint; angles are multiples of 15 degrees about the mating anchor.
    ///
    /// # Errors
    /// Returns `Incidence` if translation or rotation breaks a finite feature constraint.
    pub fn place(self, displacement: Vec3, steps: u8) -> Result<ConstructionFrame, WeldError> {
        if !displacement.is_finite()
            || !self
                .displacement(displacement)
                .abs_diff_eq(displacement, EPSILON)
        {
            return Err(WeldError::Incidence);
        }
        let spin = Quat::from_axis_angle(
            self.destination.normal,
            f32::from(steps % 24) * std::f32::consts::PI / 12.0,
        );
        let rotation = spin * self.initial.rotation();
        let anchor = self.destination.point + displacement;
        let frame = ConstructionFrame::new(anchor - rotation * self.source.point, rotation)
            .map_err(|_| WeldError::InvalidFeature)?;
        let valid = match (self.source.feature, self.destination.feature) {
            (WeldFeature::Vertex(a), WeldFeature::Vertex(b)) => {
                frame.point(a).abs_diff_eq(b, EPSILON)
            }
            (WeldFeature::Vertex(p), WeldFeature::Edge([a, b])) => {
                point_on_segment(frame.point(p), a, b)
            }
            (WeldFeature::Edge([a, b]), WeldFeature::Vertex(p)) => {
                point_on_segment(p, frame.point(a), frame.point(b))
            }
            (WeldFeature::Edge(a), WeldFeature::Edge(b)) => {
                let a = a.map(|p| frame.point(p));
                let direction = (b[1] - b[0]).normalize();
                let collinear = a
                    .iter()
                    .all(|p| (*p - b[0]).cross(direction).length() <= EPSILON);
                let start = (a[0] - b[0]).dot(direction);
                let end = (a[1] - b[0]).dot(direction);
                collinear
                    && start.min(end) <= b[0].distance(b[1]) + EPSILON
                    && start.max(end) >= -EPSILON
            }
            _ => true,
        };
        valid.then_some(frame).ok_or(WeldError::Incidence)
    }

    /// Advances to the next constraint-compatible orientation, ignoring obstacles.
    pub fn next_rotation(self, displacement: Vec3, current: u8) -> Option<u8> {
        (1..=24)
            .map(|delta| (current % 24 + delta) % 24)
            .find(|&step| self.place(displacement, step).is_ok())
    }
}

fn point_on_segment(point: Vec3, a: Vec3, b: Vec3) -> bool {
    let edge = b - a;
    let length = edge.length();
    if length <= EPSILON {
        return false;
    }
    let direction = edge / length;
    let offset = point - a;
    offset.cross(direction).length() <= EPSILON
        && offset.dot(direction) >= -EPSILON
        && offset.dot(direction) <= length + EPSILON
}

/// Stateful displacement snapping which preserves the current position on Shift changes.
#[derive(Clone, Copy, Debug, Default)]
pub struct WeldSnap {
    input_origin: Vec2,
    output_origin: Vec2,
    output: Vec2,
    fine: Option<bool>,
}

impl WeldSnap {
    /// Snaps relative drag coordinates to 25 cm, or 5 cm with Shift.
    pub fn update(&mut self, input: Vec2, fine: bool) -> Vec2 {
        if self.fine.is_some_and(|previous| previous != fine) {
            self.input_origin = input;
            self.output_origin = self.output;
        }
        self.fine = Some(fine);
        let step = if fine { 0.05 } else { 0.25 };
        self.output = self.output_origin + ((input - self.input_origin) / step).round() * step;
        self.output
    }
}

/// Convex coplanar material polygon in destination tangent coordinates, in metres.
/// Polygons can adjoin; omitted material represents holes and gaps.
pub type WeldMaterialPatch = Vec<Vec2>;

/// Finds a continuous 5 cm square covered by both unions of coplanar material.
/// The square stays aligned with destination tangent axes. This checks containment,
/// rather than total area or four corner samples (which would overlook holes).
///
/// # Errors
/// Returns `InsufficientContact` when no covered square exists.
pub fn weld_contact_square(
    source: &[WeldMaterialPatch],
    destination: &[WeldMaterialPatch],
) -> Result<Vec2, WeldError> {
    let half = 0.025;
    let mut lines = Vec::<(Vec2, f32)>::new();
    let mut candidates = Vec::new();
    for polygon in source.iter().chain(destination) {
        if polygon.len() < 3 || polygon.iter().any(|p| !p.is_finite()) {
            continue;
        }
        for (&a, &b) in polygon.iter().zip(polygon.iter().cycle().skip(1)) {
            candidates.push(a);
            for sign in [-1.0, 1.0] {
                lines.push((Vec2::X, a.x + sign * half));
                lines.push((Vec2::Y, a.y + sign * half));
            }
            if let Some(normal) = (b - a).perp().try_normalize() {
                let support = half * (normal.x.abs() + normal.y.abs());
                for sign in [-1.0, 1.0] {
                    lines.push((normal, normal.dot(a) + sign * support));
                }
            }
        }
    }
    let contains = |center: Vec2| {
        let square = [
            center + Vec2::new(-half, -half),
            center + Vec2::new(half, -half),
            center + Vec2::new(half, half),
            center + Vec2::new(-half, half),
        ];
        covered(&square, source) && covered(&square, destination)
    };
    if let Some(center) = candidates.into_iter().find(|&center| contains(center)) {
        return Ok(center);
    }
    // Coplanar tiles repeat boundary lines. Remove those duplicates before the
    // arrangement search and test candidates lazily instead of allocating O(n²) points.
    let mut unique = Vec::<(Vec2, f32)>::new();
    for (mut normal, mut distance) in lines {
        if normal.x < 0.0 || (normal.x == 0.0 && normal.y < 0.0) {
            normal = -normal;
            distance = -distance;
        }
        if !unique
            .iter()
            .any(|&(n, d)| n.abs_diff_eq(normal, 1.0e-7) && (d - distance).abs() < 1.0e-7)
        {
            unique.push((normal, distance));
        }
    }
    for (i, &(a, da)) in unique.iter().enumerate() {
        for &(b, db) in &unique[i + 1..] {
            let determinant = a.perp_dot(b);
            if determinant.abs() > EPSILON {
                let center = Vec2::new(da * b.y - a.y * db, a.x * db - da * b.x) / determinant;
                if contains(center) {
                    return Ok(center);
                }
            }
        }
    }
    Err(WeldError::InsufficientContact)
}

fn area(polygon: &[Vec2]) -> f32 {
    polygon
        .iter()
        .zip(polygon.iter().cycle().skip(1))
        .map(|(a, b)| a.perp_dot(*b))
        .sum::<f32>()
        * 0.5
}

fn clip(polygon: &[Vec2], normal: Vec2, distance: f32) -> Vec<Vec2> {
    let mut result = Vec::new();
    for (&a, &b) in polygon.iter().zip(polygon.iter().cycle().skip(1)) {
        let da = normal.dot(a) - distance;
        let db = normal.dot(b) - distance;
        if da >= 0.0 {
            result.push(a);
        }
        if (da < 0.0) != (db < 0.0) {
            result.push(a + (b - a) * (da / (da - db)));
        }
    }
    result
}

fn covered(square: &[Vec2], material: &[WeldMaterialPatch]) -> bool {
    let mut uncovered = vec![square.to_vec()];
    for polygon in material {
        if polygon.len() < 3 || polygon.iter().any(|point| !point.is_finite()) {
            continue;
        }
        let signed_area = area(polygon);
        if !signed_area.is_finite() {
            continue;
        }
        let winding = signed_area.signum();
        if winding == 0.0 {
            continue;
        }
        let mut next = Vec::new();
        for mut remaining in uncovered {
            for (&a, &b) in polygon.iter().zip(polygon.iter().cycle().skip(1)) {
                let Some(normal) = ((b - a).perp() * winding).try_normalize() else {
                    continue;
                };
                let distance = normal.dot(a) - 1.0e-7;
                let outside = clip(&remaining, -normal, -distance);
                if area(&outside).abs() > 1.0e-10 {
                    next.push(outside);
                }
                remaining = clip(&remaining, normal, distance);
                if remaining.len() < 3 {
                    break;
                }
            }
        }
        uncovered = next;
        if uncovered.is_empty() {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests;

/// Evaluated topology identity. Vertex indices are valid only in the captured revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeldFeatureRef {
    /// Flat logical surface, including flat cylinder ends.
    Face(crate::SurfacePatchKey),
    /// Straight logical edge, excluding curved chains and tessellation seams.
    Edge(crate::TopologyKey),
    /// Real boundary corner with at least three straight incident logical edges.
    Vertex(u32),
}

/// A feature reference guarded by its originating graph revision.
#[derive(Clone, Debug)]
pub struct WeldPick {
    revision: crate::ConstructionGraph,
    /// Solid carrying the selection.
    pub owner: crate::SolidOwner,
    /// Selected logical topology.
    pub feature: WeldFeatureRef,
    /// Adjacent planar surface retained for the gesture.
    pub mating: crate::SurfacePatchKey,
    point: Vec3,
}

impl WeldPick {
    /// Selects a nearby real corner or straight logical edge on the approached face.
    ///
    /// # Errors
    /// Rejects curved, missing, or nonincident mating surfaces.
    pub fn nearest(
        graph: &crate::ConstructionGraph,
        owner: crate::SolidOwner,
        mating: crate::SurfacePatchKey,
        point: Vec3,
        radius: f32,
    ) -> Result<Self, WeldError> {
        let face = Self::new(graph, owner, WeldFeatureRef::Face(mating), mating, point)?;
        let solid = graph
            .evaluated_solid(owner)
            .map_err(|_| WeldError::InvalidFeature)?;
        let mut vertices = solid
            .vertices
            .iter()
            .enumerate()
            .filter(|(_, vertex)| vertex.position.distance(point) <= radius)
            .collect::<Vec<_>>();
        vertices.sort_by(|(_, a), (_, b)| {
            a.position
                .distance_squared(point)
                .total_cmp(&b.position.distance_squared(point))
        });
        for (index, vertex) in vertices {
            if let Ok(index) = u32::try_from(index)
                && let Ok(pick) = Self::new(
                    graph,
                    owner,
                    WeldFeatureRef::Vertex(index),
                    mating,
                    vertex.position,
                )
            {
                return Ok(pick);
            }
        }
        let mut best = None;
        for edge in &solid.logical_edges {
            let Some([a, b]) = straight_edge(&solid, edge) else {
                continue;
            };
            let delta = b - a;
            let nearest =
                a + delta * ((point - a).dot(delta) / delta.length_squared()).clamp(0.0, 1.0);
            let distance = nearest.distance(point);
            if distance <= radius
                && best.as_ref().is_none_or(|(prior, _)| distance < *prior)
                && let Ok(pick) = Self::new(
                    graph,
                    owner,
                    WeldFeatureRef::Edge(edge.key),
                    mating,
                    nearest,
                )
            {
                best = Some((distance, pick));
            }
        }
        Ok(best.map_or(face, |(_, pick)| pick))
    }

    /// Captures a feature on evaluated default geometry, retaining its mating plane.
    ///
    /// # Errors
    /// Returns `InvalidFeature` for missing, curved, nonincident, or degenerate topology.
    pub fn new(
        graph: &crate::ConstructionGraph,
        owner: crate::SolidOwner,
        feature: WeldFeatureRef,
        mating: crate::SurfacePatchKey,
        point: Vec3,
    ) -> Result<Self, WeldError> {
        let pick = Self {
            revision: graph.clone(),
            owner,
            feature,
            mating,
            point,
        };
        pick.resolve(graph)?;
        Ok(pick)
    }

    /// Resolves default geometry, rejecting any intervening graph edit.
    ///
    /// # Errors
    /// Returns `InvalidFeature` if the revision or selected topology no longer matches.
    pub fn resolve(&self, graph: &crate::ConstructionGraph) -> Result<WeldSelection, WeldError> {
        if !graph.shares_revision(&self.revision) {
            return Err(WeldError::InvalidFeature);
        }
        let solid = graph
            .evaluated_solid(self.owner)
            .map_err(|_| WeldError::InvalidFeature)?;
        let surface = solid
            .surfaces
            .iter()
            .find(|surface| surface.key == self.mating && surface.smoothing_group == 0)
            .ok_or(WeldError::InvalidFeature)?;
        let normal = surface.normal.normalize();
        let plane_point =
            solid.vertices[solid.half_edges[surface.half_edge as usize].origin as usize].position;
        if solid
            .surfaces
            .iter()
            .filter(|s| s.key == self.mating)
            .any(|s| !s.normal.abs_diff_eq(normal, EPSILON))
        {
            return Err(WeldError::InvalidFeature);
        }
        if (self.point - plane_point).dot(normal).abs() > EPSILON {
            return Err(WeldError::InvalidFeature);
        }
        if !on_patch(&solid, self.mating, self.point, normal) {
            return Err(WeldError::InvalidFeature);
        }
        let feature = match self.feature {
            WeldFeatureRef::Face(key) if key == self.mating => WeldFeature::Face,
            WeldFeatureRef::Face(_) => return Err(WeldError::InvalidFeature),
            WeldFeatureRef::Edge(key) => {
                let edge = solid.logical_edge(key).ok_or(WeldError::InvalidFeature)?;
                if !edge.half_edges.iter().any(|&index| {
                    let half = solid.half_edges[index as usize];
                    solid.surfaces[half.face as usize].key == self.mating
                        || solid.surfaces[solid.half_edges[half.twin as usize].face as usize].key
                            == self.mating
                }) {
                    return Err(WeldError::InvalidFeature);
                }
                WeldFeature::Edge(straight_edge(&solid, edge).ok_or(WeldError::InvalidFeature)?)
            }
            WeldFeatureRef::Vertex(index) => {
                let point = solid
                    .vertices
                    .get(index as usize)
                    .ok_or(WeldError::InvalidFeature)?
                    .position;
                let count = solid
                    .logical_edges
                    .iter()
                    .filter(|edge| {
                        straight_edge(&solid, edge)
                            .is_some_and(|ends| ends.iter().any(|p| p.abs_diff_eq(point, EPSILON)))
                    })
                    .count();
                let incident = solid.half_edges.iter().any(|edge| {
                    edge.origin == index && solid.surfaces[edge.face as usize].key == self.mating
                });
                if count < 3 || !incident {
                    return Err(WeldError::InvalidFeature);
                }
                WeldFeature::Vertex(point)
            }
        };
        let selection = WeldSelection {
            feature,
            point: self.point,
            normal,
            tangent: normal.any_orthonormal_vector(),
        };
        selection.validate()?;
        Ok(selection)
    }
}

fn straight_edge(solid: &crate::EvaluatedSolid, edge: &crate::LogicalEdge) -> Option<[Vec3; 2]> {
    if edge.closed {
        return None;
    }
    let points = edge
        .half_edges
        .iter()
        .flat_map(|&index| {
            let half = solid.half_edges[index as usize];
            [
                solid.vertices[half.origin as usize].position,
                solid.vertices[solid.half_edges[half.next as usize].origin as usize].position,
            ]
        })
        .collect::<Vec<_>>();
    let origin = *points.first()?;
    let direction = (*points.iter().find(|p| p.distance(origin) > EPSILON)? - origin).normalize();
    if points
        .iter()
        .any(|p| (*p - origin).cross(direction).length() > EPSILON)
    {
        return None;
    }
    let (low, high) = points
        .iter()
        .map(|p| (*p - origin).dot(direction))
        .fold((0.0_f32, 0.0_f32), |(a, b), x| (a.min(x), b.max(x)));
    Some([origin + direction * low, origin + direction * high])
}

fn on_patch(
    solid: &crate::EvaluatedSolid,
    mating: crate::SurfacePatchKey,
    point: Vec3,
    normal: Vec3,
) -> bool {
    solid
        .surfaces
        .iter()
        .filter(|s| s.key == mating)
        .any(|surface| {
            let mut edge = surface.half_edge;
            let mut side = 0.0_f32;
            loop {
                let half = solid.half_edges[edge as usize];
                let a = solid.vertices[half.origin as usize].position;
                let b =
                    solid.vertices[solid.half_edges[half.next as usize].origin as usize].position;
                let cross = (b - a).cross(point - a).dot(normal);
                if cross.abs() > EPSILON {
                    if side * cross < 0.0 {
                        return false;
                    }
                    side = cross.signum();
                }
                edge = half.next;
                if edge == surface.half_edge {
                    return true;
                }
            }
        })
}
