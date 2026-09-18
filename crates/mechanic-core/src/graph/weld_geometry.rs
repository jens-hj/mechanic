//! Where two parts may weld: mating faces, feature welds, and contact squares.

use super::ConstructionGraph;
use super::error::GraphError;
use super::predicates::{primitive_surface_patch, profile_cells};
use crate::geometry::{
    FaceGeometry, FaceProfile, cuboid_face, cylinder_face, ground_face, pipe_bend_face,
    pipe_junction_face,
};
use crate::{ANCHOR_TOLERANCE_METERS, FaceKind, FaceOwner, FaceRef, PartSpec, SolidOwner};
use bevy_math::{Vec2, Vec3};
use std::collections::BTreeSet;

impl ConstructionGraph {
    /// Collects adjoining material on the selected rigid body's mating plane.
    ///
    /// # Errors
    /// Rejects missing faces or a construction that cannot compile.
    pub fn weld_mating_faces(&self, selected: FaceRef) -> Result<Vec<FaceRef>, crate::WeldError> {
        let reject = crate::WeldError::InvalidFeature;
        let target = self.face_geometry(selected).map_err(|_| reject)?;
        let FaceOwner::Part(part) = selected.owner else {
            return Ok(vec![selected]);
        };
        let creation = self.compile().map_err(|_| reject)?;
        let body = creation
            .part_to_compound
            .iter()
            .find_map(|&(id, body)| (id == part).then_some(body))
            .ok_or(reject)?;
        let mut faces = vec![selected];
        let mut seen = BTreeSet::new();
        for &part in &creation.compounds[body as usize].source_parts {
            let owner = self
                .region_of(part)
                .map_or(SolidOwner::Part(part), SolidOwner::Region);
            if !seen.insert(owner) {
                continue;
            }
            let candidates = if let Ok(solid) = self.evaluated_solid_shared(owner) {
                solid
                    .surfaces
                    .iter()
                    .filter(|surface| surface.smoothing_group == 0)
                    .map(|surface| FaceRef::patch(part, FaceKind::PositiveY, surface.key))
                    .collect::<Vec<_>>()
            } else {
                [
                    FaceKind::NegativeX,
                    FaceKind::PositiveX,
                    FaceKind::NegativeY,
                    FaceKind::PositiveY,
                    FaceKind::NegativeZ,
                    FaceKind::PositiveZ,
                ]
                .into_iter()
                .map(|kind| FaceRef::part(part, kind))
                .collect()
            };
            for face in candidates {
                if faces.contains(&face) {
                    continue;
                }
                if let Ok(geometry) = self.face_geometry(face)
                    && geometry.normal.abs_diff_eq(target.normal, 1.0e-5)
                    && (geometry.center - target.center).dot(target.normal).abs()
                        <= ANCHOR_TOLERANCE_METERS
                {
                    faces.push(face);
                }
            }
        }
        Ok(faces)
    }

    /// Checks that a transformed corner or finite edge remains incident to mating material.
    pub fn weld_feature_on_faces(
        &self,
        faces: &[FaceRef],
        feature: crate::WeldFeature,
        transform: crate::ConstructionFrame,
    ) -> bool {
        let ends = match feature {
            crate::WeldFeature::Face => return true,
            crate::WeldFeature::Vertex(point) => [transform.point(point); 2],
            crate::WeldFeature::Edge(ends) => ends.map(|point| transform.point(point)),
        };
        faces.iter().any(|&face| {
            let Ok(geometry) = self.face_geometry(face) else {
                return false;
            };
            if ends.iter().any(|point| {
                (*point - geometry.center).dot(geometry.normal).abs() > ANCHOR_TOLERANCE_METERS
            }) {
                return false;
            }
            if matches!(geometry.profile, FaceProfile::Ground) {
                return true;
            }
            let points = ends.map(|point| {
                let offset = point - geometry.center;
                Vec2::new(
                    offset.dot(geometry.tangent_u),
                    offset.dot(geometry.tangent_v),
                )
            });
            self.weld_material_cells(
                face,
                &geometry,
                crate::ConstructionFrame::IDENTITY,
                geometry.normal,
            )
            .iter()
            .any(|polygon| {
                let winding = polygon
                    .iter()
                    .zip(polygon.iter().cycle().skip(1))
                    .map(|(a, b)| a.perp_dot(*b))
                    .sum::<f32>()
                    .signum();
                let mut low = 0.0_f32;
                let mut high = 1.0_f32;
                for (a, b) in polygon.iter().zip(polygon.iter().cycle().skip(1)) {
                    let normal = (*b - *a).perp() * winding;
                    let start = (points[0] - *a).dot(normal);
                    let delta = (points[1] - points[0]).dot(normal);
                    if delta.abs() < 1.0e-10 {
                        if start < -1.0e-7 {
                            return false;
                        }
                    } else {
                        let boundary = -start / delta;
                        if delta > 0.0 {
                            low = low.max(boundary);
                        } else {
                            high = high.min(boundary);
                        }
                    }
                }
                low <= high + 1.0e-5
            })
        })
    }

    /// Validates continuous mating material containing a 5 × 5 cm square.
    /// Lists may include adjoining coplanar faces on the same respective rigid body.
    /// The first destination face defines the square's tangent axes and plane.
    ///
    /// # Errors
    /// Rejects missing faces, noncoplanar mating planes, and insufficient continuous material.
    pub fn weld_contact_square(
        &self,
        source: &[FaceRef],
        destination: &[FaceRef],
    ) -> Result<Vec3, crate::WeldError> {
        self.weld_contact_square_transformed(
            source,
            destination,
            crate::ConstructionFrame::IDENTITY,
            crate::ConstructionFrame::IDENTITY,
        )
    }

    /// Validates mating material under two independent body motions.
    ///
    /// # Errors
    /// Rejects missing faces, noncoplanar surfaces, or insufficient continuous contact.
    pub fn weld_contact_square_transformed(
        &self,
        source: &[FaceRef],
        destination: &[FaceRef],
        source_motion: crate::ConstructionFrame,
        destination_motion: crate::ConstructionFrame,
    ) -> Result<Vec3, crate::WeldError> {
        let rejection = crate::WeldError::InsufficientContact;
        let mut target = self
            .face_geometry(*destination.first().ok_or(rejection)?)
            .map_err(|_| crate::WeldError::InvalidFeature)?;
        target.center = destination_motion.point(target.center);
        target.normal = destination_motion.vector(target.normal);
        target.tangent_u = destination_motion.vector(target.tangent_u);
        target.tangent_v = destination_motion.vector(target.tangent_v);
        let collect = |faces: &[FaceRef], opposed: bool| {
            let motion = if opposed {
                source_motion
            } else {
                destination_motion
            };
            let expected = if opposed {
                -target.normal
            } else {
                target.normal
            };
            faces
                .iter()
                .flat_map(|&face| self.weld_material_cells(face, &target, motion, expected))
                .collect::<Vec<_>>()
        };
        let mut source_cells = collect(source, true);
        let mut destination_cells = collect(destination, false);
        // An infinite ground plane contributes no finite material boundary.
        if source.iter().any(|face| face.owner == FaceOwner::Ground)
            && target
                .normal
                .abs_diff_eq(-source_motion.vector(Vec3::Y), 1.0e-5)
        {
            source_cells.clone_from(&destination_cells);
        }
        if destination
            .iter()
            .any(|face| face.owner == FaceOwner::Ground)
        {
            destination_cells.clone_from(&source_cells);
        }
        let center = crate::weld_contact_square(&source_cells, &destination_cells)?;
        Ok(target.center + target.tangent_u * center.x + target.tangent_v * center.y)
    }

    pub(super) fn weld_material_cells(
        &self,
        face: FaceRef,
        target: &FaceGeometry,
        motion: crate::ConstructionFrame,
        expected: Vec3,
    ) -> Vec<Vec<Vec2>> {
        let Ok(mut geometry) = self.face_geometry(face) else {
            return Vec::new();
        };
        geometry.center = motion.point(geometry.center);
        geometry.normal = motion.vector(geometry.normal);
        geometry.tangent_u = motion.vector(geometry.tangent_u);
        geometry.tangent_v = motion.vector(geometry.tangent_v);
        if !geometry.normal.abs_diff_eq(expected, 1.0e-5)
            || (geometry.center - target.center).dot(target.normal).abs() > ANCHOR_TOLERANCE_METERS
        {
            return Vec::new();
        }
        if let FaceOwner::Part(part) = face.owner {
            let owner = self
                .region_of(part)
                .map_or(SolidOwner::Part(part), SolidOwner::Region);
            if let Ok(solid) = self.evaluated_solid_shared(owner) {
                return solid
                    .surfaces
                    .iter()
                    .filter_map(|surface| {
                        if !motion.vector(surface.normal).abs_diff_eq(expected, 1.0e-5) {
                            return None;
                        }
                        let mut points = Vec::new();
                        let mut edge = surface.half_edge;
                        loop {
                            let half = solid.half_edges[edge as usize];
                            let offset = motion
                                .point(solid.vertices[half.origin as usize].position)
                                - target.center;
                            if offset.dot(target.normal).abs() > ANCHOR_TOLERANCE_METERS {
                                return None;
                            }
                            points.push(Vec2::new(
                                offset.dot(target.tangent_u),
                                offset.dot(target.tangent_v),
                            ));
                            edge = half.next;
                            if edge == surface.half_edge {
                                break;
                            }
                        }
                        Some(points)
                    })
                    .collect();
            }
        }
        profile_cells(&geometry, target.center, target.tangent_u, target.tangent_v)
    }

    pub(crate) fn face_geometry(&self, face: FaceRef) -> Result<FaceGeometry, GraphError> {
        match face.owner {
            FaceOwner::Part(part) => {
                let spec = self
                    .parts
                    .get(part)
                    .copied()
                    .ok_or(GraphError::MissingPart(part))?;
                let owner = self
                    .region_of(part)
                    .map_or(SolidOwner::Part(part), SolidOwner::Region);
                if face.patch.is_some() || self.owner_has_shape_features(owner) {
                    let patch = face
                        .patch
                        .unwrap_or_else(|| primitive_surface_patch(spec, face.face));
                    return self.evaluated_patch_geometry(owner, patch);
                }
                match spec {
                    PartSpec::Cuboid(spec) => Ok(cuboid_face(spec, face.face)),
                    PartSpec::Controller(spec) => Ok(cuboid_face(spec.cuboid(), face.face)),
                    PartSpec::Engine(spec) => Ok(cuboid_face(spec.cuboid(), face.face)),
                    PartSpec::Transmission(spec) => Ok(cuboid_face(spec.cuboid(), face.face)),
                    PartSpec::Servo(spec) => Ok(cuboid_face(spec.cuboid(), face.face)),
                    PartSpec::Seat(spec) => Ok(cuboid_face(spec.cuboid(), face.face)),
                    PartSpec::Input(spec) => Ok(cuboid_face(spec.cuboid(), face.face)),
                    PartSpec::DimensionLink(spec) => Ok(cuboid_face(spec.cuboid(), face.face)),
                    PartSpec::Cylinder(spec) => {
                        cylinder_face(spec, face.face).ok_or(GraphError::InvalidCylinderFace)
                    }
                    PartSpec::PipeBend(spec) => {
                        pipe_bend_face(spec, face.face).ok_or(GraphError::InvalidPipeBendFace)
                    }
                    PartSpec::PipeJunction(spec) => pipe_junction_face(spec, face.face)
                        .ok_or(GraphError::InvalidPipeJunctionFace),
                }
                .map(|mut geometry| {
                    let frame = self.part_frame(part).expect("face part exists");
                    geometry.center = frame.point(geometry.center);
                    geometry.normal = frame.vector(geometry.normal);
                    geometry.tangent_u = frame.vector(geometry.tangent_u);
                    geometry.tangent_v = frame.vector(geometry.tangent_v);
                    geometry
                })
            }
            FaceOwner::Ground if face.face == FaceKind::PositiveY => Ok(ground_face()),
            FaceOwner::Ground => Err(GraphError::InvalidGroundFace),
        }
    }

    pub(super) fn evaluated_patch_geometry(
        &self,
        owner: SolidOwner,
        patch: crate::SurfacePatchKey,
    ) -> Result<FaceGeometry, GraphError> {
        let solid = self.evaluated_solid_shared(owner)?;
        let surface = solid
            .surfaces
            .iter()
            .find(|surface| surface.key == patch)
            .ok_or(GraphError::MissingSurfacePatch { owner, patch })?;
        let mut points = Vec::<Vec3>::new();
        let mut edge = surface.half_edge;
        loop {
            let half_edge = solid.half_edges[edge as usize];
            points.push(solid.vertices[half_edge.origin as usize].position);
            edge = half_edge.next;
            if edge == surface.half_edge {
                break;
            }
        }
        if points.len() < 3 {
            return Err(GraphError::MissingSurfacePatch { owner, patch });
        }
        let point_count = f32::from(
            u16::try_from(points.len())
                .map_err(|_| GraphError::MissingSurfacePatch { owner, patch })?,
        );
        let center = points.iter().copied().sum::<Vec3>() / point_count;
        let normal = surface.normal.normalize_or_zero();
        let tangent_u = normal.any_orthonormal_vector();
        let tangent_v = normal.cross(tangent_u);
        let vertices = points
            .iter()
            .map(|point| {
                let offset = *point - center;
                Vec2::new(offset.dot(tangent_u), offset.dot(tangent_v))
            })
            .collect();
        Ok(FaceGeometry {
            center,
            normal,
            tangent_u,
            tangent_v,
            profile: FaceProfile::Polygon { vertices },
        })
    }
}
