//! Parametric construction-solid evaluation.
//!
//! Ordinary construction geometry is represented twice: a manifold boundary
//! for rendering and selection, and disjoint convex cells for mass and
//! collision.  Feature references name logical edges rather than tessellation
//! segments, so a rounded cylinder rim remains one selectable chain.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::collections::{BTreeMap, BTreeSet};

use bevy_math::{DVec3, Quat, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    CYLINDER_SWEEP_STEP_DEGREES, ConvexFace, ConvexPiece, FaceKind, PartPiece, PartSpec,
    PipeBendSpec, RegionId, ShapeFeatureId, ShapeRegion, decompose, decompose_part,
};

const EPSILON: f64 = 1.0e-8;
const KEY_SCALE: f64 = 1_000_000.0;
const FILLET_MAX_FACET_DEGREES: f64 = 7.5;

/// A construction solid which may own feature targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SolidOwner {
    /// One ordinary construction part.
    Part(crate::PartId),
    /// One Shape region, whose member blocks share geometry.
    Region(RegionId),
}

/// Where a stable topology key originated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TopologySource {
    /// Topology emitted by the owner's base generator.
    Base,
    /// Topology introduced by an earlier feature.
    Feature(ShapeFeatureId),
}

/// Stable key for a logical curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TopologyKey {
    /// Base or generating-feature provenance.
    pub source: TopologySource,
    /// Deterministic identity within that provenance.
    pub local: u32,
}

/// Stable key for one logical surface patch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SurfacePatchKey {
    /// Base or generating-feature provenance.
    pub source: TopologySource,
    /// Deterministic identity within that provenance.
    pub local: u32,
}

/// Reference to a complete tangent-continuous logical edge chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeChainRef {
    /// Solid carrying the chain.
    pub owner: SolidOwner,
    /// Stable logical-curve key.
    pub edge: TopologyKey,
}

/// Constant profile applied to selected edge chains.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeTreatment {
    /// Symmetric equal-setback planar cut.
    Chamfer,
    /// Constant-radius polygonal round, at no more than 7.5 degrees per facet.
    Fillet,
}

/// One ordered parametric edge feature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeFeature {
    /// Logical chains treated together.
    pub targets: Vec<EdgeChainRef>,
    /// Chamfer or fillet profile.
    pub treatment: EdgeTreatment,
    /// Equal setback or radius in exact 2.5 mm position ticks.
    pub amount_ticks: u32,
}

impl ShapeFeature {
    /// Creates a feature record. Graph insertion validates owners, topology,
    /// and the positive amount transactionally.
    pub fn new(
        targets: impl IntoIterator<Item = EdgeChainRef>,
        treatment: EdgeTreatment,
        amount_ticks: u32,
    ) -> Self {
        Self {
            targets: targets.into_iter().collect(),
            treatment,
            amount_ticks,
        }
    }
}

/// One boundary vertex.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryVertex {
    /// Build-space position in metres.
    pub position: Vec3,
    /// One outgoing half-edge, when the boundary is non-empty.
    pub outgoing: Option<u32>,
}

/// One directed side of a manifold boundary edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundaryHalfEdge {
    /// Origin vertex index.
    pub origin: u32,
    /// Oppositely directed half-edge index.
    pub twin: u32,
    /// Next half-edge around the face.
    pub next: u32,
    /// Surface-patch polygon index.
    pub face: u32,
    /// Logical chain, absent on tessellation seams within one patch.
    pub logical_edge: Option<TopologyKey>,
}

/// One polygon belonging to a logical surface patch.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfacePatch {
    /// Stable patch provenance.
    pub key: SurfacePatchKey,
    /// Outward polygon normal.
    pub normal: Vec3,
    /// Boundary half-edge at which this loop begins.
    pub half_edge: u32,
    /// Smooth shading group. Zero denotes a hard planar patch.
    pub smoothing_group: u32,
    /// Base patch whose texture projection this surface continues.
    pub uv_provenance: SurfacePatchKey,
}

/// A complete logical edge and all of its tessellated boundary segments.
#[derive(Clone, Debug, PartialEq)]
pub struct LogicalEdge {
    /// Stable chain key.
    pub key: TopologyKey,
    /// Half-edges forming this logical curve.
    pub half_edges: Vec<u32>,
    /// Whether the chain closes on itself.
    pub closed: bool,
    /// Whether its profile is convex and can be treated by V1.
    pub convex: bool,
}

/// One disjoint convex volume used by mass integration and collision.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvexVolumeCell {
    /// Convex polyhedron in build space.
    pub piece: ConvexPiece,
}

/// Evaluated result of a base solid plus its ordered features.
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatedSolid {
    /// Manifold boundary vertices.
    pub vertices: Vec<BoundaryVertex>,
    /// Manifold half-edges.
    pub half_edges: Vec<BoundaryHalfEdge>,
    /// Boundary surface polygons.
    pub surfaces: Vec<SurfacePatch>,
    /// Selectable logical curves, excluding tessellation seams.
    pub logical_edges: Vec<LogicalEdge>,
    /// Positive, pairwise interior-disjoint convex cells.
    pub cells: Vec<ConvexVolumeCell>,
}

impl EvaluatedSolid {
    /// Finds one logical chain by stable key.
    pub fn logical_edge(&self, key: TopologyKey) -> Option<&LogicalEdge> {
        self.logical_edges.iter().find(|edge| edge.key == key)
    }

    /// Total represented volume in cubic metres.
    pub fn volume(&self) -> f64 {
        self.cells
            .iter()
            .map(|cell| f64::from(cell.piece.volume))
            .sum()
    }
}

/// A base generator or ordered feature could not produce valid solid geometry.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SolidError {
    /// Authored machine geometry is fixed.
    #[error("authored machine parts do not support Shape features")]
    AuthoredPart,
    /// Feature amounts are positive integer position ticks.
    #[error("a chamfer or fillet amount must be positive")]
    ZeroAmount,
    /// The feature no longer names topology produced by the preceding replay.
    #[error("feature {feature:?} references missing edge {edge:?}")]
    MissingEdge {
        /// Feature which failed to replay.
        feature: ShapeFeatureId,
        /// Missing logical curve.
        edge: TopologyKey,
    },
    /// The selected chain is concave or otherwise unsupported by subtractive clipping.
    #[error("edge {0:?} is not a convex feature edge")]
    NonConvexEdge(TopologyKey),
    /// The requested amount consumes a cell or produces collapsed topology.
    #[error("feature {0:?} is too large for its target geometry")]
    AmountTooLarge(ShapeFeatureId),
    /// Boundary stitching found an open or multiply-used edge.
    #[error("evaluated solid is not a closed two-manifold")]
    NonManifold,
    /// No positive-volume cell survived evaluation.
    #[error("evaluated solid has no positive volume")]
    ZeroVolume,
}

#[derive(Clone, Debug)]
struct PolyFace {
    vertices: Vec<DVec3>,
    patch: SurfacePatchKey,
    smoothing_group: u32,
    uv_provenance: SurfacePatchKey,
}

#[derive(Clone, Debug)]
struct PolyCell {
    faces: Vec<PolyFace>,
}

#[derive(Clone, Copy)]
struct ClipPlane {
    normal: DVec3,
    offset: f64,
    patch: SurfacePatchKey,
    smoothing_group: u32,
    uv_provenance: SurfacePatchKey,
}

#[derive(Clone)]
struct EdgeSegment {
    key: TopologyKey,
    half_edge: u32,
    a: DVec3,
    b: DVec3,
    first_normal: DVec3,
    second_normal: DVec3,
    first_patch: SurfacePatchKey,
    cell: usize,
    convex: bool,
}

/// Evaluates one ordinary construction part with features already filtered to
/// that owner and supplied in global order.
///
/// # Errors
///
/// Returns an error when the part is authored rather than construction geometry,
/// its base boundary is invalid, or an ordered feature cannot be replayed.
pub fn evaluate_part_solid(
    spec: PartSpec,
    features: impl IntoIterator<Item = (ShapeFeatureId, ShapeFeature)>,
) -> Result<EvaluatedSolid, SolidError> {
    let cells = match spec {
        PartSpec::Cuboid(cuboid) => pieces_to_cells(decompose_part(cuboid)),
        PartSpec::Cylinder(cylinder) => cylinder_cells(cylinder),
        PartSpec::PipeBend(bend) => pipe_bend_cells(bend),
        PartSpec::Controller(_)
        | PartSpec::Engine(_)
        | PartSpec::Transmission(_)
        | PartSpec::Servo(_)
        | PartSpec::Seat(_)
        | PartSpec::Input(_)
        | PartSpec::DimensionLink(_) => return Err(SolidError::AuthoredPart),
    };
    evaluate(cells, features)
}

/// Evaluates one Shape region with features already filtered to that owner and
/// supplied in global order.
///
/// # Errors
///
/// Returns an error when the region boundary is invalid or an ordered feature
/// cannot be replayed.
pub fn evaluate_region_solid(
    region: &ShapeRegion,
    features: impl IntoIterator<Item = (ShapeFeatureId, ShapeFeature)>,
) -> Result<EvaluatedSolid, SolidError> {
    let grid = region.grid();
    let pieces = decompose(&grid, &|cell, corner| region.corner_steps(cell, corner));
    evaluate(pieces_to_cells(pieces), features)
}

fn evaluate(
    mut cells: Vec<PolyCell>,
    features: impl IntoIterator<Item = (ShapeFeatureId, ShapeFeature)>,
) -> Result<EvaluatedSolid, SolidError> {
    for (feature_id, feature) in features {
        if feature.amount_ticks == 0 {
            return Err(SolidError::ZeroAmount);
        }
        let (_, segments) = stitch(&cells)?;
        let selected = feature
            .targets
            .iter()
            .map(|target| target.edge)
            .collect::<BTreeSet<_>>();
        for key in &selected {
            if !segments.iter().any(|segment| segment.key == *key) {
                return Err(SolidError::MissingEdge {
                    feature: feature_id,
                    edge: *key,
                });
            }
            if segments
                .iter()
                .any(|segment| segment.key == *key && !segment.convex)
            {
                return Err(SolidError::NonConvexEdge(*key));
            }
        }
        let amount = f64::from(feature.amount_ticks) * f64::from(crate::POSITION_TICK_METERS);
        validate_feature_clearance(
            &cells,
            &segments,
            &selected,
            feature.treatment,
            amount,
            feature_id,
        )?;
        let mut planes_by_cell = BTreeMap::<usize, Vec<ClipPlane>>::new();
        for segment in segments
            .iter()
            .filter(|segment| selected.contains(&segment.key))
        {
            let dot = segment
                .first_normal
                .dot(segment.second_normal)
                .clamp(-1.0, 1.0);
            let angle = dot.acos();
            if !(1.0e-6..=core::f64::consts::PI - 1.0e-6).contains(&angle) {
                continue;
            }
            let steps = match feature.treatment {
                EdgeTreatment::Chamfer => 1,
                EdgeTreatment::Fillet => {
                    ((angle.to_degrees() / FILLET_MAX_FACET_DEGREES).ceil() as usize).max(1)
                }
            };
            for step in 0..steps {
                let step = f64::from(u32::try_from(step).unwrap_or(u32::MAX));
                let steps = f64::from(u32::try_from(steps).unwrap_or(u32::MAX));
                let fraction = (step + 0.5) / steps;
                let normal = slerp_unit(segment.first_normal, segment.second_normal, fraction);
                let edge_offset = normal.dot(segment.a);
                let offset = match feature.treatment {
                    EdgeTreatment::Chamfer => {
                        let sine = (angle * 0.5).sin().max(EPSILON);
                        edge_offset - amount * sine
                    }
                    EdgeTreatment::Fillet => {
                        fillet_chord_offset(segment, normal, amount, angle / steps, edge_offset)
                    }
                };
                let patch = generated_patch_key(feature_id, segment, step as usize + 1);
                planes_by_cell
                    .entry(segment.cell)
                    .or_default()
                    .push(ClipPlane {
                        normal,
                        offset,
                        patch,
                        smoothing_group: u32::from(matches!(
                            feature.treatment,
                            EdgeTreatment::Fillet
                        )) * (feature_id.index().saturating_add(1)),
                        uv_provenance: segment.first_patch,
                    });
            }
        }
        if feature.treatment == EdgeTreatment::Fillet {
            append_fillet_junction_planes(
                feature_id,
                amount,
                &segments,
                &selected,
                &mut planes_by_cell,
            );
        }
        clip_feature_cells(&mut cells, planes_by_cell, feature_id)?;
        cells.retain(|cell| cell_volume(cell) > EPSILON);
        if cells.is_empty() {
            return Err(SolidError::AmountTooLarge(feature_id));
        }
    }
    build_evaluated(&cells)
}

fn append_fillet_junction_planes(
    feature: ShapeFeatureId,
    radius: f64,
    segments: &[EdgeSegment],
    selected: &BTreeSet<TopologyKey>,
    planes_by_cell: &mut BTreeMap<usize, Vec<ClipPlane>>,
) {
    let mut incident = BTreeMap::<(usize, PointKey), Vec<&EdgeSegment>>::new();
    for segment in segments
        .iter()
        .filter(|segment| selected.contains(&segment.key))
    {
        for point in [segment.a, segment.b] {
            incident
                .entry((segment.cell, point_key(point)))
                .or_default()
                .push(segment);
        }
    }
    for ((cell, vertex_key), incident_edges) in incident {
        let edge_count = incident_edges
            .iter()
            .map(|edge| edge.key)
            .collect::<BTreeSet<_>>()
            .len();
        if edge_count < 3 {
            continue;
        }
        let mut normals = BTreeMap::<PointKey, DVec3>::new();
        for edge in &incident_edges {
            for normal in [edge.first_normal, edge.second_normal] {
                normals.entry(point_key(normal)).or_insert(normal);
            }
        }
        let normals = normals.into_values().collect::<Vec<_>>();
        if normals.len() != 3 {
            continue;
        }
        let vertex = incident_edges[0].a;
        let vertex = if point_key(vertex) == vertex_key {
            vertex
        } else {
            incident_edges[0].b
        };
        let [first, second, third] = [normals[0], normals[1], normals[2]];
        let determinant = first.dot(second.cross(third));
        if determinant.abs() <= EPSILON {
            continue;
        }
        let distances = [
            first.dot(vertex) - radius,
            second.dot(vertex) - radius,
            third.dot(vertex) - radius,
        ];
        let centre = (distances[0] * second.cross(third)
            + distances[1] * third.cross(first)
            + distances[2] * first.cross(second))
            / determinant;
        let maximum_angle = first
            .dot(second)
            .min(second.dot(third))
            .min(third.dot(first))
            .clamp(-1.0, 1.0)
            .acos();
        let steps =
            ((maximum_angle.to_degrees() / FILLET_MAX_FACET_DEGREES).ceil() as usize).max(1);
        let direction = |first_weight: usize, second_weight: usize| {
            let third_weight = steps - first_weight - second_weight;
            (first * first_weight as f64
                + second * second_weight as f64
                + third * third_weight as f64)
                .normalize()
        };
        let mut ordinal = 0_usize;
        for first_weight in 0..steps {
            for second_weight in 0..steps - first_weight {
                let a = direction(first_weight, second_weight);
                let b = direction(first_weight + 1, second_weight);
                let c = direction(first_weight, second_weight + 1);
                push_fillet_junction_plane(
                    feature,
                    vertex,
                    centre,
                    radius,
                    [a, b, c],
                    ordinal,
                    incident_edges[0].first_patch,
                    planes_by_cell.entry(cell).or_default(),
                );
                ordinal += 1;
                if first_weight + second_weight + 1 < steps {
                    let d = direction(first_weight + 1, second_weight + 1);
                    push_fillet_junction_plane(
                        feature,
                        vertex,
                        centre,
                        radius,
                        [b, d, c],
                        ordinal,
                        incident_edges[0].first_patch,
                        planes_by_cell.entry(cell).or_default(),
                    );
                    ordinal += 1;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_fillet_junction_plane(
    feature: ShapeFeatureId,
    vertex: DVec3,
    centre: DVec3,
    radius: f64,
    directions: [DVec3; 3],
    ordinal: usize,
    uv_provenance: SurfacePatchKey,
    planes: &mut Vec<ClipPlane>,
) {
    let points = directions.map(|direction| centre + direction * radius);
    let mut normal = (points[1] - points[0])
        .cross(points[2] - points[0])
        .normalize();
    if normal.dot(directions.into_iter().sum()) < 0.0 {
        normal = -normal;
    }
    planes.push(ClipPlane {
        normal,
        offset: normal.dot(points[0]),
        patch: generated_junction_patch_key(feature, vertex, ordinal),
        smoothing_group: feature.index().saturating_add(1),
        uv_provenance,
    });
}

fn generated_junction_patch_key(
    feature: ShapeFeatureId,
    vertex: DVec3,
    ordinal: usize,
) -> SurfacePatchKey {
    let mut hash = 2_166_136_261_u32;
    for value in [
        0x4a55_4e43,
        point_word(point_key(vertex)),
        u32::try_from(ordinal).unwrap_or(u32::MAX),
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(16_777_619);
    }
    SurfacePatchKey {
        source: TopologySource::Feature(feature),
        local: hash,
    }
}

fn clip_feature_cells(
    cells: &mut [PolyCell],
    planes_by_cell: BTreeMap<usize, Vec<ClipPlane>>,
    feature_id: ShapeFeatureId,
) -> Result<(), SolidError> {
    for (cell_index, planes) in planes_by_cell {
        let Some(cell) = cells.get_mut(cell_index) else {
            continue;
        };
        for plane in planes {
            *cell = clip_cell(cell, plane).ok_or(SolidError::AmountTooLarge(feature_id))?;
        }
    }
    Ok(())
}

fn validate_feature_clearance(
    cells: &[PolyCell],
    segments: &[EdgeSegment],
    selected: &BTreeSet<TopologyKey>,
    treatment: EdgeTreatment,
    amount: f64,
    feature: ShapeFeatureId,
) -> Result<(), SolidError> {
    for segment in segments
        .iter()
        .filter(|segment| selected.contains(&segment.key))
    {
        let dot = segment
            .first_normal
            .dot(segment.second_normal)
            .clamp(-1.0, 1.0);
        let angle = dot.acos();
        if !(1.0e-6..=core::f64::consts::PI - 1.0e-6).contains(&angle) {
            continue;
        }
        let setback = match treatment {
            EdgeTreatment::Chamfer => amount,
            EdgeTreatment::Fillet => amount * (angle * 0.5).tan(),
        };
        let first_inward = -(segment.second_normal - segment.first_normal * dot).normalize();
        let second_inward = -(segment.first_normal - segment.second_normal * dot).normalize();
        let Some(cell) = cells.get(segment.cell) else {
            return Err(SolidError::AmountTooLarge(feature));
        };
        let first_clearance = face_clearance(cell, segment, segment.first_normal, first_inward);
        let second_clearance = face_clearance(cell, segment, segment.second_normal, second_inward);
        if setback > first_clearance + EPSILON || setback > second_clearance + EPSILON {
            return Err(SolidError::AmountTooLarge(feature));
        }
    }
    Ok(())
}

fn face_clearance(cell: &PolyCell, segment: &EdgeSegment, normal: DVec3, inward: DVec3) -> f64 {
    cell.faces
        .iter()
        .filter(|face| polygon_normal(&face.vertices).dot(normal) >= 1.0 - 1.0e-6)
        .filter(|face| {
            face.vertices
                .iter()
                .all(|vertex| normal.dot(*vertex - segment.a).abs() <= 1.0e-6)
        })
        .flat_map(|face| face.vertices.iter())
        .map(|vertex| inward.dot(*vertex - segment.a))
        .fold(0.0, f64::max)
}

/// Plane through one chord of the ideal circular fillet. Sampling the chord
/// midpoint for its normal keeps each facet symmetric, while the chord itself
/// includes the true tangent points at the two ends of the complete profile.
fn fillet_chord_offset(
    segment: &EdgeSegment,
    normal: DVec3,
    radius: f64,
    facet_angle: f64,
    edge_offset: f64,
) -> f64 {
    let normal_dot = segment
        .first_normal
        .dot(segment.second_normal)
        .clamp(-1.0, 1.0);
    let centre_projection = -radius * normal.dot(segment.first_normal + segment.second_normal)
        / (1.0 + normal_dot).max(EPSILON);
    edge_offset + centre_projection + radius * (facet_angle * 0.5).cos()
}

fn generated_patch_key(
    feature: ShapeFeatureId,
    segment: &EdgeSegment,
    step: usize,
) -> SurfacePatchKey {
    let mut hash = 2_166_136_261_u32;
    for value in [
        segment.key.local,
        u32::try_from(step).unwrap_or(u32::MAX),
        point_word(point_key(segment.a)),
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(16_777_619);
    }
    SurfacePatchKey {
        source: TopologySource::Feature(feature),
        local: hash,
    }
}

fn point_word(point: PointKey) -> u32 {
    point.0.into_iter().fold(0_u32, |hash, value| {
        let bytes = value.cast_unsigned().to_le_bytes();
        let low = u32::from_le_bytes(bytes[..4].try_into().expect("four low bytes"));
        let high = u32::from_le_bytes(bytes[4..].try_into().expect("four high bytes"));
        hash.rotate_left(5) ^ low ^ high
    })
}

fn slerp_unit(a: DVec3, b: DVec3, fraction: f64) -> DVec3 {
    let angle = a.dot(b).clamp(-1.0, 1.0).acos();
    if angle < EPSILON {
        return a;
    }
    ((a * ((1.0 - fraction) * angle).sin() + b * (fraction * angle).sin()) / angle.sin())
        .normalize()
}

fn clip_cell(cell: &PolyCell, plane: ClipPlane) -> Option<PolyCell> {
    let mut faces = Vec::new();
    let mut cap = Vec::<DVec3>::new();
    for face in &cell.faces {
        let mut polygon = Vec::new();
        for index in 0..face.vertices.len() {
            let current = face.vertices[index];
            let next = face.vertices[(index + 1) % face.vertices.len()];
            let current_distance = plane.normal.dot(current) - plane.offset;
            let next_distance = plane.normal.dot(next) - plane.offset;
            let current_inside = current_distance <= EPSILON;
            let next_inside = next_distance <= EPSILON;
            if current_inside {
                push_unique(&mut polygon, current);
            }
            if current_inside != next_inside {
                let fraction = current_distance / (current_distance - next_distance);
                let intersection = current.lerp(next, fraction);
                push_unique(&mut polygon, intersection);
                push_unique_global(&mut cap, intersection);
            }
        }
        clean_polygon(&mut polygon);
        if polygon.len() >= 3 {
            faces.push(PolyFace {
                vertices: polygon,
                patch: face.patch,
                smoothing_group: face.smoothing_group,
                uv_provenance: face.uv_provenance,
            });
        }
    }
    if cap.len() >= 3 {
        let center = cap.iter().copied().sum::<DVec3>() / cap.len() as f64;
        let tangent = plane.normal.any_orthonormal_vector();
        let bitangent = plane.normal.cross(tangent);
        cap.sort_by(|left, right| {
            let l = *left - center;
            let r = *right - center;
            l.dot(bitangent)
                .atan2(l.dot(tangent))
                .total_cmp(&r.dot(bitangent).atan2(r.dot(tangent)))
        });
        if polygon_normal(&cap).dot(plane.normal) < 0.0 {
            cap.reverse();
        }
        faces.push(PolyFace {
            vertices: cap,
            patch: plane.patch,
            smoothing_group: plane.smoothing_group,
            uv_provenance: plane.uv_provenance,
        });
    }
    let result = PolyCell { faces };
    (result.faces.len() >= 4 && cell_volume(&result) > EPSILON).then_some(result)
}

fn build_evaluated(cells: &[PolyCell]) -> Result<EvaluatedSolid, SolidError> {
    let (stitched, _) = stitch(cells)?;
    let volume_cells = cells
        .iter()
        .filter_map(poly_cell_to_convex)
        .map(|piece| ConvexVolumeCell { piece })
        .collect::<Vec<_>>();
    if volume_cells.is_empty() {
        return Err(SolidError::ZeroVolume);
    }
    Ok(EvaluatedSolid {
        vertices: stitched.vertices,
        half_edges: stitched.half_edges,
        surfaces: stitched.surfaces,
        logical_edges: stitched.logical_edges,
        cells: volume_cells,
    })
}

struct Stitched {
    vertices: Vec<BoundaryVertex>,
    half_edges: Vec<BoundaryHalfEdge>,
    surfaces: Vec<SurfacePatch>,
    logical_edges: Vec<LogicalEdge>,
}

#[allow(clippy::too_many_lines)]
fn stitch(cells: &[PolyCell]) -> Result<(Stitched, Vec<EdgeSegment>), SolidError> {
    let mut occurrences = BTreeMap::<FaceSignature, Vec<(usize, usize)>>::new();
    for (cell_index, cell) in cells.iter().enumerate() {
        for (face_index, face) in cell.faces.iter().enumerate() {
            occurrences
                .entry(face_signature(&face.vertices))
                .or_default()
                .push((cell_index, face_index));
        }
    }
    let boundary = occurrences
        .values()
        .filter(|uses| uses.len() == 1)
        .map(|uses| uses[0])
        .collect::<Vec<_>>();
    let mut vertex_map = BTreeMap::<PointKey, u32>::new();
    let mut vertices = Vec::<BoundaryVertex>::new();
    let mut half_edges = Vec::<BoundaryHalfEdge>::new();
    let mut surfaces = Vec::<SurfacePatch>::new();
    let mut directed = BTreeMap::<(u32, u32), u32>::new();
    let mut half_edge_cells = Vec::<usize>::new();
    for (cell_index, face_index) in boundary {
        let face = &cells[cell_index].faces[face_index];
        let surface_index = u32::try_from(surfaces.len()).map_err(|_| SolidError::NonManifold)?;
        let first_edge = u32::try_from(half_edges.len()).map_err(|_| SolidError::NonManifold)?;
        let indices = face
            .vertices
            .iter()
            .map(|&point| {
                let key = point_key(point);
                *vertex_map.entry(key).or_insert_with(|| {
                    let index = u32::try_from(vertices.len()).unwrap_or(u32::MAX);
                    vertices.push(BoundaryVertex {
                        position: point.as_vec3(),
                        outgoing: None,
                    });
                    index
                })
            })
            .collect::<Vec<_>>();
        for index in 0..indices.len() {
            let origin = indices[index];
            let destination = indices[(index + 1) % indices.len()];
            let edge = u32::try_from(half_edges.len()).map_err(|_| SolidError::NonManifold)?;
            vertices[origin as usize].outgoing.get_or_insert(edge);
            half_edges.push(BoundaryHalfEdge {
                origin,
                twin: u32::MAX,
                next: first_edge
                    + u32::try_from((index + 1) % indices.len())
                        .map_err(|_| SolidError::NonManifold)?,
                face: surface_index,
                logical_edge: None,
            });
            half_edge_cells.push(cell_index);
            if directed.insert((origin, destination), edge).is_some() {
                return Err(SolidError::NonManifold);
            }
        }
        surfaces.push(SurfacePatch {
            key: face.patch,
            normal: polygon_normal(&face.vertices).as_vec3(),
            half_edge: first_edge,
            smoothing_group: face.smoothing_group,
            uv_provenance: face.uv_provenance,
        });
    }
    for (&(origin, destination), &edge) in &directed {
        let Some(&twin) = directed.get(&(destination, origin)) else {
            return Err(SolidError::NonManifold);
        };
        half_edges[edge as usize].twin = twin;
    }
    let mut logical = BTreeMap::<TopologyKey, Vec<u32>>::new();
    let mut segments = Vec::new();
    for edge_index in 0..half_edges.len() {
        let edge = half_edges[edge_index];
        if edge_index as u32 > edge.twin {
            continue;
        }
        let first_surface = &surfaces[edge.face as usize];
        let second_surface = &surfaces[half_edges[edge.twin as usize].face as usize];
        let first = first_surface.key;
        let second = second_surface.key;
        if first == second
            || (first_surface.smoothing_group != 0
                && first_surface.smoothing_group == second_surface.smoothing_group)
        {
            continue;
        }
        let key = topology_key(first, second);
        logical.entry(key).or_default().push(edge_index as u32);
        let a = DVec3::from(vertices[edge.origin as usize].position);
        let b = DVec3::from(vertices[half_edges[edge.next as usize].origin as usize].position);
        let first_normal = DVec3::from(surfaces[edge.face as usize].normal);
        let second_normal =
            DVec3::from(surfaces[half_edges[edge.twin as usize].face as usize].normal);
        let tangent = (b - a).normalize();
        let convex = first_normal.cross(second_normal).dot(tangent) > EPSILON;
        segments.push(EdgeSegment {
            key,
            half_edge: edge_index as u32,
            a,
            b,
            first_normal,
            second_normal,
            first_patch: first,
            cell: half_edge_cells[edge_index],
            convex,
        });
    }
    let mut logical_edges = Vec::new();
    for (candidate_key, half_edges_for_key) in logical {
        let chains = split_logical_chains(&half_edges, half_edges_for_key);
        let split = chains.len() > 1;
        for (ordinal, half_edges_for_key) in chains.into_iter().enumerate() {
            let key = if split {
                split_topology_key(candidate_key, ordinal)
            } else {
                candidate_key
            };
            let mut degree = BTreeMap::<u32, usize>::new();
            for &edge in &half_edges_for_key {
                half_edges[edge as usize].logical_edge = Some(key);
                let twin = half_edges[edge as usize].twin;
                half_edges[twin as usize].logical_edge = Some(key);
                let start = half_edges[edge as usize].origin;
                let end = half_edges[half_edges[edge as usize].next as usize].origin;
                *degree.entry(start).or_default() += 1;
                *degree.entry(end).or_default() += 1;
                if let Some(segment) = segments
                    .iter_mut()
                    .find(|segment| segment.half_edge == edge)
                {
                    segment.key = key;
                }
            }
            let convex = half_edges_for_key.iter().all(|edge| {
                segments
                    .iter()
                    .find(|segment| segment.half_edge == *edge)
                    .is_none_or(|segment| segment.convex)
            });
            logical_edges.push(LogicalEdge {
                key,
                half_edges: half_edges_for_key,
                closed: !degree.is_empty() && degree.values().all(|degree| *degree == 2),
                convex,
            });
        }
    }
    logical_edges.sort_by_key(|edge| edge.key);
    Ok((
        Stitched {
            vertices,
            half_edges,
            surfaces,
            logical_edges,
        },
        segments,
    ))
}

fn split_logical_chains(
    half_edges: &[BoundaryHalfEdge],
    half_edges_for_key: Vec<u32>,
) -> Vec<Vec<u32>> {
    let endpoints = |edge: u32| {
        let start = half_edges[edge as usize].origin;
        let end = half_edges[half_edges[edge as usize].next as usize].origin;
        (start, end)
    };
    let mut incident = BTreeMap::<u32, Vec<u32>>::new();
    for &edge in &half_edges_for_key {
        let (start, end) = endpoints(edge);
        incident.entry(start).or_default().push(edge);
        incident.entry(end).or_default().push(edge);
    }
    let mut unvisited = half_edges_for_key.into_iter().collect::<BTreeSet<_>>();
    let mut chains = Vec::new();

    for (&vertex, edges) in &incident {
        if edges.len() == 2 {
            continue;
        }
        for &edge in edges {
            if !unvisited.remove(&edge) {
                continue;
            }
            chains.push(walk_logical_chain(
                half_edges,
                &incident,
                &mut unvisited,
                edge,
                vertex,
            ));
        }
    }
    while let Some(&edge) = unvisited.first() {
        unvisited.remove(&edge);
        let (start, _) = endpoints(edge);
        chains.push(walk_logical_chain(
            half_edges,
            &incident,
            &mut unvisited,
            edge,
            start,
        ));
    }
    chains
}

fn walk_logical_chain(
    half_edges: &[BoundaryHalfEdge],
    incident: &BTreeMap<u32, Vec<u32>>,
    unvisited: &mut BTreeSet<u32>,
    first_edge: u32,
    start_vertex: u32,
) -> Vec<u32> {
    let mut chain = vec![first_edge];
    let first = half_edges[first_edge as usize];
    let first_end = half_edges[first.next as usize].origin;
    let mut vertex = if first.origin == start_vertex {
        first_end
    } else {
        first.origin
    };
    while let Some(edges) = incident.get(&vertex)
        && edges.len() == 2
        && let Some(&next) = edges.iter().find(|edge| unvisited.contains(edge))
    {
        unvisited.remove(&next);
        chain.push(next);
        let edge = half_edges[next as usize];
        let end = half_edges[edge.next as usize].origin;
        vertex = if edge.origin == vertex {
            end
        } else {
            edge.origin
        };
    }
    chain
}

fn split_topology_key(candidate: TopologyKey, ordinal: usize) -> TopologyKey {
    let mut hash = 2_166_136_261_u32;
    for value in [
        0x5350_4c54,
        candidate.local,
        u32::try_from(ordinal).unwrap_or(u32::MAX),
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(16_777_619);
    }
    TopologyKey {
        source: candidate.source,
        local: hash,
    }
}

fn topology_key(first: SurfacePatchKey, second: SurfacePatchKey) -> TopologyKey {
    let (a, b) = if first <= second {
        (first, second)
    } else {
        (second, first)
    };
    let source = match (a.source, b.source) {
        (TopologySource::Base, TopologySource::Base) => TopologySource::Base,
        (TopologySource::Feature(feature), TopologySource::Base)
        | (TopologySource::Base, TopologySource::Feature(feature)) => {
            TopologySource::Feature(feature)
        }
        (TopologySource::Feature(first), TopologySource::Feature(second)) => {
            TopologySource::Feature(first.max(second))
        }
    };
    let mut hash = 2_166_136_261_u32;
    for value in [patch_word(a), a.local, patch_word(b), b.local] {
        hash ^= value;
        hash = hash.wrapping_mul(16_777_619);
    }
    TopologyKey {
        source,
        local: hash,
    }
}

const fn patch_word(key: SurfacePatchKey) -> u32 {
    match key.source {
        TopologySource::Base => 0,
        TopologySource::Feature(feature) => feature.index().wrapping_add(1),
    }
}

fn pieces_to_cells(pieces: Vec<PartPiece>) -> Vec<PolyCell> {
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

fn cuboid_cell(center: Vec3, half: Vec3, rotation: Quat) -> PolyCell {
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
        faces: loops
            .into_iter()
            .enumerate()
            .map(|(index, face)| {
                base_face(face.map(|vertex| vertices[vertex]).to_vec(), index as u32)
            })
            .collect(),
    }
}

fn convex_piece_cell(piece: ConvexPiece) -> PolyCell {
    PolyCell {
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

const fn grid_patch(face: FaceKind) -> u32 {
    match face {
        FaceKind::NegativeX => 0,
        FaceKind::PositiveX => 1,
        FaceKind::NegativeY => 2,
        FaceKind::PositiveY => 3,
        FaceKind::NegativeZ => 4,
        FaceKind::PositiveZ => 5,
    }
}

fn base_face(vertices: Vec<DVec3>, local: u32) -> PolyFace {
    let key = SurfacePatchKey {
        source: TopologySource::Base,
        local,
    };
    PolyFace {
        vertices,
        patch: key,
        smoothing_group: 0,
        uv_provenance: key,
    }
}

fn cylinder_cells(spec: crate::CylinderSpec) -> Vec<PolyCell> {
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

fn pipe_bend_cells(spec: PipeBendSpec) -> Vec<PolyCell> {
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

fn prism_cell(
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
        let face = vec![bottom[next], bottom[index], top[index], top[next]];
        faces.push(base_face(face, local));
    }
    PolyCell { faces }
}

fn poly_cell_to_convex(cell: &PolyCell) -> Option<ConvexPiece> {
    let mut vertices = Vec::<Vec3>::new();
    let mut index_by_key = BTreeMap::<PointKey, u32>::new();
    let mut faces = Vec::new();
    for face in &cell.faces {
        let indices = face
            .vertices
            .iter()
            .map(|&point| {
                *index_by_key.entry(point_key(point)).or_insert_with(|| {
                    let index = u32::try_from(vertices.len()).unwrap_or(u32::MAX);
                    vertices.push(point.as_vec3());
                    index
                })
            })
            .collect::<Vec<_>>();
        let normal = polygon_normal(&face.vertices).as_vec3().normalize_or_zero();
        faces.push(ConvexFace {
            normal,
            offset: normal.dot(vertices[indices[0] as usize]),
            indices,
            grid_face: None,
        });
    }
    let mut edges = Vec::<Vec3>::new();
    for face in &faces {
        for index in 0..face.indices.len() {
            let a = vertices[face.indices[index] as usize];
            let b = vertices[face.indices[(index + 1) % face.indices.len()] as usize];
            let mut direction = (b - a).normalize_or_zero();
            if direction.x < -1.0e-6
                || (direction.x.abs() <= 1.0e-6
                    && (direction.y < -1.0e-6
                        || (direction.y.abs() <= 1.0e-6 && direction.z < 0.0)))
            {
                direction = -direction;
            }
            if direction != Vec3::ZERO
                && !edges
                    .iter()
                    .any(|other| other.abs_diff_eq(direction, 1.0e-5))
            {
                edges.push(direction);
            }
        }
    }
    let volume = cell_volume(cell);
    let centroid = cell_centroid(cell);
    (volume > EPSILON).then_some(ConvexPiece {
        vertices,
        faces,
        edge_directions: edges,
        centroid: centroid.as_vec3(),
        volume: volume as f32,
    })
}

fn cell_volume(cell: &PolyCell) -> f64 {
    cell.faces
        .iter()
        .map(|face| {
            let anchor = face.vertices[0];
            (1..face.vertices.len() - 1)
                .map(|index| anchor.dot(face.vertices[index].cross(face.vertices[index + 1])) / 6.0)
                .sum::<f64>()
        })
        .sum::<f64>()
        .abs()
}

fn cell_centroid(cell: &PolyCell) -> DVec3 {
    let reference = cell
        .faces
        .first()
        .and_then(|face| face.vertices.first())
        .copied()
        .unwrap_or(DVec3::ZERO);
    let mut volume = 0.0;
    let mut moment = DVec3::ZERO;
    for face in &cell.faces {
        for index in 1..face.vertices.len() - 1 {
            let a = face.vertices[0];
            let b = face.vertices[index];
            let c = face.vertices[index + 1];
            let signed = (a - reference).dot((b - reference).cross(c - reference)) / 6.0;
            volume += signed;
            moment += (reference + a + b + c) * (signed / 4.0);
        }
    }
    if volume.abs() > EPSILON {
        moment / volume
    } else {
        reference
    }
}

fn polygon_normal(vertices: &[DVec3]) -> DVec3 {
    let mut normal = DVec3::ZERO;
    for index in 0..vertices.len() {
        let current = vertices[index];
        let next = vertices[(index + 1) % vertices.len()];
        normal += current.cross(next);
    }
    normal.normalize_or_zero()
}

fn clean_polygon(polygon: &mut Vec<DVec3>) {
    if polygon.len() > 1
        && polygon[0].distance_squared(*polygon.last().unwrap()) <= EPSILON * EPSILON
    {
        polygon.pop();
    }
    let mut index = 0;
    while polygon.len() >= 3 && index < polygon.len() {
        let previous = polygon[(index + polygon.len() - 1) % polygon.len()];
        let current = polygon[index];
        let next = polygon[(index + 1) % polygon.len()];
        if (current - previous).cross(next - current).length_squared() <= EPSILON * EPSILON {
            polygon.remove(index);
        } else {
            index += 1;
        }
    }
}

fn push_unique(points: &mut Vec<DVec3>, point: DVec3) {
    if points
        .last()
        .is_none_or(|last| last.distance_squared(point) > EPSILON * EPSILON)
    {
        points.push(point);
    }
}

fn push_unique_global(points: &mut Vec<DVec3>, point: DVec3) {
    if !points
        .iter()
        .any(|other| other.distance_squared(point) <= EPSILON * EPSILON)
    {
        points.push(point);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PointKey([i64; 3]);

fn point_key(point: DVec3) -> PointKey {
    PointKey([
        (point.x * KEY_SCALE).round() as i64,
        (point.y * KEY_SCALE).round() as i64,
        (point.z * KEY_SCALE).round() as i64,
    ])
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct FaceSignature(Vec<PointKey>);

fn face_signature(vertices: &[DVec3]) -> FaceSignature {
    let mut keys = vertices.iter().copied().map(point_key).collect::<Vec<_>>();
    keys.sort_unstable();
    FaceSignature(keys)
}

#[cfg(test)]
mod tests {
    use bevy_math::IVec3;

    use super::*;
    use crate::id::Handle;
    use crate::{
        BuildPose, ConstructionMaterial, CuboidSpec, CylinderDimensions, CylinderSpec,
        GridRotation, PipeBendDimensions, PipeBendSpec,
    };

    fn cube() -> PartSpec {
        PartSpec::Cuboid(
            CuboidSpec::new(
                [1, 1, 1],
                BuildPose::new(IVec3::ZERO, GridRotation::default()),
            )
            .unwrap()
            .with_material(ConstructionMaterial::Steel),
        )
    }

    #[test]
    fn cuboid_boundary_has_six_patches_and_twelve_logical_edges() {
        let solid = evaluate_part_solid(cube(), []).unwrap();
        assert_eq!(solid.surfaces.len(), 6);
        assert_eq!(solid.logical_edges.len(), 12);
        assert_eq!(solid.cells.len(), 1);
        assert!((solid.volume() - 0.25_f64.powi(3)).abs() < 1.0e-8);
    }

    #[test]
    fn chamfer_replaces_one_edge_with_a_generated_patch() {
        let base = evaluate_part_solid(cube(), []).unwrap();
        let edge = base.logical_edges[0].key;
        let feature_id = ShapeFeatureId::from_parts(0, 0);
        let feature = ShapeFeature::new(
            [EdgeChainRef {
                owner: SolidOwner::Part(crate::PartId::from_parts(0, 0)),
                edge,
            }],
            EdgeTreatment::Chamfer,
            10,
        );
        let rounded = evaluate_part_solid(cube(), [(feature_id, feature)]).unwrap();
        assert!(
            rounded
                .surfaces
                .iter()
                .any(|surface| surface.key.source == TopologySource::Feature(feature_id))
        );
        assert!(rounded.volume() < base.volume());
    }

    #[test]
    fn fillet_chords_end_at_the_exact_face_tangencies() {
        let patch = SurfacePatchKey {
            source: TopologySource::Base,
            local: 0,
        };
        let segment = EdgeSegment {
            key: TopologyKey {
                source: TopologySource::Base,
                local: 0,
            },
            half_edge: 0,
            a: DVec3::ZERO,
            b: DVec3::Z,
            first_normal: DVec3::X,
            second_normal: DVec3::Y,
            first_patch: patch,
            cell: 0,
            convex: true,
        };
        let radius = 0.05;
        let angle = core::f64::consts::FRAC_PI_2;
        let facets = (angle.to_degrees() / FILLET_MAX_FACET_DEGREES).ceil() as usize;
        let facet_angle = angle / facets as f64;
        let centre = DVec3::new(-radius, -radius, 0.0);
        let first_tangent = centre + DVec3::X * radius;
        let last_tangent = centre + DVec3::Y * radius;

        let first_normal = slerp_unit(DVec3::X, DVec3::Y, 0.5 / facets as f64);
        let first_offset = fillet_chord_offset(&segment, first_normal, radius, facet_angle, 0.0);
        let last_normal = slerp_unit(DVec3::X, DVec3::Y, (facets as f64 - 0.5) / facets as f64);
        let last_offset = fillet_chord_offset(&segment, last_normal, radius, facet_angle, 0.0);

        assert!((first_normal.dot(first_tangent) - first_offset).abs() < EPSILON);
        assert!((last_normal.dot(last_tangent) - last_offset).abs() < EPSILON);
        assert_eq!(facets, 12);
    }

    #[test]
    fn cuboid_fillet_accepts_sub_block_radii_in_five_centimetre_steps() {
        let base = evaluate_part_solid(cube(), []).unwrap();
        let edge = base.logical_edges[0].key;
        let feature_id = ShapeFeatureId::from_parts(0, 0);
        for amount_ticks in [20, 40, 60, 80] {
            let feature = ShapeFeature::new(
                [EdgeChainRef {
                    owner: SolidOwner::Part(crate::PartId::from_parts(0, 0)),
                    edge,
                }],
                EdgeTreatment::Fillet,
                amount_ticks,
            );
            let filleted = evaluate_part_solid(cube(), [(feature_id, feature)])
                .unwrap_or_else(|error| panic!("{amount_ticks} ticks was rejected: {error}"));
            if amount_ticks == 20 {
                let radius = f64::from(amount_ticks) * f64::from(crate::POSITION_TICK_METERS);
                let expected =
                    0.25_f64.powi(3) - 0.25 * radius.powi(2) * (1.0 - core::f64::consts::FRAC_PI_4);
                assert!(
                    (filleted.volume() - expected).abs() < 2.0e-6,
                    "five centimetres produced volume {}, expected {expected}",
                    filleted.volume()
                );
            }
        }
    }

    #[test]
    fn cuboid_treatments_accept_a_full_block_amount_but_not_more() {
        let base = evaluate_part_solid(cube(), []).unwrap();
        let edge = base.logical_edges[0].key;
        let feature_id = ShapeFeatureId::from_parts(0, 0);
        for treatment in [EdgeTreatment::Fillet, EdgeTreatment::Chamfer] {
            let target = || EdgeChainRef {
                owner: SolidOwner::Part(crate::PartId::from_parts(0, 0)),
                edge,
            };
            evaluate_part_solid(
                cube(),
                [(feature_id, ShapeFeature::new([target()], treatment, 100))],
            )
            .unwrap_or_else(|error| panic!("full-block {treatment:?} failed: {error}"));

            assert_eq!(
                evaluate_part_solid(
                    cube(),
                    [(feature_id, ShapeFeature::new([target()], treatment, 120),)],
                ),
                Err(SolidError::AmountTooLarge(feature_id))
            );
        }
    }

    #[test]
    fn larger_cuboids_accept_fillets_and_chamfers_past_one_block() {
        let spec = PartSpec::Cuboid(
            CuboidSpec::new(
                [2, 2, 2],
                BuildPose::new(IVec3::ZERO, GridRotation::default()),
            )
            .unwrap()
            .with_material(ConstructionMaterial::Steel),
        );
        let base = evaluate_part_solid(spec, []).unwrap();
        let edge = base.logical_edges[0].key;
        let feature_id = ShapeFeatureId::from_parts(0, 0);
        for treatment in [EdgeTreatment::Fillet, EdgeTreatment::Chamfer] {
            let feature = ShapeFeature::new(
                [EdgeChainRef {
                    owner: SolidOwner::Part(crate::PartId::from_parts(0, 0)),
                    edge,
                }],
                treatment,
                120,
            );
            evaluate_part_solid(spec, [(feature_id, feature)])
                .unwrap_or_else(|error| panic!("{treatment:?} past one block failed: {error}"));
        }
    }

    #[test]
    fn three_incident_fillet_edges_round_their_shared_corner() {
        let base = evaluate_part_solid(cube(), []).unwrap();
        let corner = Vec3::splat(0.125);
        let edges = base
            .logical_edges
            .iter()
            .filter(|logical| {
                logical.half_edges.iter().any(|&edge_index| {
                    let edge = base.half_edges[edge_index as usize];
                    let next = base.half_edges[edge.next as usize];
                    base.vertices[edge.origin as usize]
                        .position
                        .abs_diff_eq(corner, 1.0e-6)
                        || base.vertices[next.origin as usize]
                            .position
                            .abs_diff_eq(corner, 1.0e-6)
                })
            })
            .map(|logical| EdgeChainRef {
                owner: SolidOwner::Part(crate::PartId::from_parts(0, 0)),
                edge: logical.key,
            })
            .collect::<Vec<_>>();
        assert_eq!(edges.len(), 3);
        let feature_id = ShapeFeatureId::from_parts(0, 0);
        let filleted = evaluate_part_solid(
            cube(),
            [(
                feature_id,
                ShapeFeature::new(edges, EdgeTreatment::Fillet, 20),
            )],
        )
        .unwrap();

        let junction = filleted
            .surfaces
            .iter()
            .filter(|surface| {
                surface.key.source == TopologySource::Feature(feature_id)
                    && surface.normal.x > 0.1
                    && surface.normal.y > 0.1
                    && surface.normal.z > 0.1
            })
            .collect::<Vec<_>>();
        assert!(!junction.is_empty());
        let sphere_centre = Vec3::splat(0.075);
        for surface in junction {
            let mut edge = surface.half_edge;
            loop {
                let half_edge = filleted.half_edges[edge as usize];
                let position = filleted.vertices[half_edge.origin as usize].position;
                assert!((position.distance(sphere_centre) - 0.05).abs() < 1.0e-5);
                edge = half_edge.next;
                if edge == surface.half_edge {
                    break;
                }
            }
        }
        assert!(filleted.logical_edges.iter().all(|logical| {
            logical.half_edges.iter().all(|&edge_index| {
                let edge = filleted.half_edges[edge_index as usize];
                let twin = filleted.half_edges[edge.twin as usize];
                let first = filleted.surfaces[edge.face as usize].smoothing_group;
                let second = filleted.surfaces[twin.face as usize].smoothing_group;
                first == 0 || first != second
            })
        }));
    }

    #[test]
    fn one_logical_edge_does_not_join_matching_edges_on_separate_cells() {
        let half = Vec3::splat(0.125);
        let cells = vec![
            cuboid_cell(Vec3::ZERO, half, Quat::IDENTITY),
            cuboid_cell(Vec3::X * 0.5, half, Quat::IDENTITY),
        ];
        let base = evaluate(cells.clone(), []).unwrap();
        assert_eq!(base.logical_edges.len(), 24);

        let edge = base.logical_edges[0].key;
        let feature_id = ShapeFeatureId::from_parts(0, 0);
        let feature = ShapeFeature::new(
            [EdgeChainRef {
                owner: SolidOwner::Region(RegionId::from_parts(0, 0)),
                edge,
            }],
            EdgeTreatment::Fillet,
            20,
        );
        let filleted = evaluate(cells, [(feature_id, feature)]).unwrap();
        let radius = 0.05_f64;
        let expected =
            2.0 * 0.25_f64.powi(3) - 0.25 * radius.powi(2) * (1.0 - core::f64::consts::FRAC_PI_4);
        assert!((filleted.volume() - expected).abs() < 2.0e-6);
    }

    #[test]
    fn concave_region_edges_are_not_offered_as_subtractive_fillet_targets() {
        let half = Vec3::splat(0.125);
        let cells = vec![
            cuboid_cell(Vec3::ZERO, half, Quat::IDENTITY),
            cuboid_cell(Vec3::X * 0.25, half, Quat::IDENTITY),
            cuboid_cell(Vec3::Y * 0.25, half, Quat::IDENTITY),
        ];

        let solid = evaluate(cells, []).unwrap();
        assert!(
            solid.logical_edges.iter().any(|edge| !edge.convex),
            "the inside corner of an L-shaped region is concave"
        );
    }

    #[test]
    fn five_centimetre_fillet_is_available_on_a_sloped_region() {
        let mut region = ShapeRegion::new(
            IVec3::ZERO,
            IVec3::new(6, 3, 2),
            ConstructionMaterial::Steel,
        )
        .unwrap();
        region.set_offset([0, 1, 1], [0, -15, 0]).unwrap();
        region.set_offset([1, 1, 1], [0, -15, 0]).unwrap();
        let base = evaluate_region_solid(&region, []).unwrap();
        let base_volume = base.volume();
        let owner = SolidOwner::Region(RegionId::from_parts(0, 0));
        for (index, logical) in base
            .logical_edges
            .iter()
            .filter(|edge| edge.convex)
            .enumerate()
        {
            let feature_id = ShapeFeatureId::from_parts(0, 0);
            let feature = ShapeFeature::new(
                [EdgeChainRef {
                    owner,
                    edge: logical.key,
                }],
                EdgeTreatment::Fillet,
                20,
            );
            let filleted = evaluate_region_solid(&region, [(feature_id, feature)])
                .unwrap_or_else(|error| panic!("sloped edge {index} rejected 5 cm: {error}"));
            assert!(
                base_volume - filleted.volume() < 0.01,
                "sloped edge {index} removed {} cubic metres at 5 cm",
                base_volume - filleted.volume()
            );
        }
    }

    #[test]
    fn region_fillets_accept_a_full_block_radius() {
        let region = ShapeRegion::new(
            IVec3::ZERO,
            IVec3::new(4, 4, 4),
            ConstructionMaterial::Steel,
        )
        .unwrap();
        let base = evaluate_region_solid(&region, []).unwrap();
        let owner = SolidOwner::Region(RegionId::from_parts(0, 0));
        for (index, logical) in base
            .logical_edges
            .iter()
            .filter(|edge| edge.convex)
            .enumerate()
        {
            let feature_id = ShapeFeatureId::from_parts(0, 0);
            let feature = ShapeFeature::new(
                [EdgeChainRef {
                    owner,
                    edge: logical.key,
                }],
                EdgeTreatment::Fillet,
                100,
            );
            evaluate_region_solid(&region, [(feature_id, feature)])
                .unwrap_or_else(|error| panic!("region edge {index} rejected 25 cm: {error}"));
        }
    }

    #[test]
    fn larger_regions_accept_fillets_and_chamfers_past_one_block() {
        let region = ShapeRegion::new(
            IVec3::ZERO,
            IVec3::new(4, 4, 4),
            ConstructionMaterial::Steel,
        )
        .unwrap();
        let base = evaluate_region_solid(&region, []).unwrap();
        let owner = SolidOwner::Region(RegionId::from_parts(0, 0));
        let edge = base
            .logical_edges
            .iter()
            .find(|edge| edge.convex)
            .unwrap()
            .key;
        let feature_id = ShapeFeatureId::from_parts(0, 0);
        for treatment in [EdgeTreatment::Fillet, EdgeTreatment::Chamfer] {
            let feature = ShapeFeature::new([EdgeChainRef { owner, edge }], treatment, 120);
            evaluate_region_solid(&region, [(feature_id, feature)]).unwrap_or_else(|error| {
                panic!("region {treatment:?} past one block failed: {error}")
            });
        }
    }

    #[test]
    fn shaped_region_boundary_excludes_internal_convex_decomposition_faces() {
        let mut region =
            ShapeRegion::new(IVec3::ZERO, IVec3::ONE, ConstructionMaterial::Steel).unwrap();
        region.set_offset([1, 1, 1], [-20, -20, -20]).unwrap();
        let pieces = decompose(&region.grid(), &|cell, corner| {
            region.corner_steps(cell, corner)
        });
        let expected_boundary_faces = pieces
            .iter()
            .map(|piece| match piece {
                PartPiece::Cuboid { .. } => 6,
                PartPiece::Convex(piece) => piece
                    .faces
                    .iter()
                    .filter(|face| face.grid_face.is_some())
                    .count(),
            })
            .sum::<usize>();

        let solid = evaluate_region_solid(&region, []).unwrap();
        assert_eq!(
            solid.surfaces.len(),
            expected_boundary_faces,
            "the feature boundary must not expose internal convex-cell faces"
        );
    }

    #[test]
    fn five_centimetre_fillet_is_available_on_a_generated_chamfer_edge() {
        let spec = PartSpec::Cuboid(
            CuboidSpec::new(
                [4, 4, 4],
                BuildPose::new(IVec3::ZERO, GridRotation::default()),
            )
            .unwrap(),
        );
        let owner = SolidOwner::Part(crate::PartId::from_parts(0, 0));
        let base = evaluate_part_solid(spec, []).unwrap();
        let chamfer_id = ShapeFeatureId::from_parts(0, 0);
        let chamfer = ShapeFeature::new(
            [EdgeChainRef {
                owner,
                edge: base.logical_edges[0].key,
            }],
            EdgeTreatment::Chamfer,
            100,
        );
        let chamfered = evaluate_part_solid(spec, [(chamfer_id, chamfer.clone())]).unwrap();
        let generated = chamfered
            .logical_edges
            .iter()
            .find(|edge| edge.key.source == TopologySource::Feature(chamfer_id) && edge.convex)
            .unwrap();
        let fillet_id = ShapeFeatureId::from_parts(1, 0);
        let fillet = ShapeFeature::new(
            [EdgeChainRef {
                owner,
                edge: generated.key,
            }],
            EdgeTreatment::Fillet,
            20,
        );

        evaluate_part_solid(spec, [(chamfer_id, chamfer), (fillet_id, fillet)])
            .expect("a generated chamfer edge accepts a 5 cm fillet");
    }

    #[test]
    fn cylinder_rims_are_closed_logical_chains_not_tessellation_edges() {
        let spec = PartSpec::Cylinder(CylinderSpec::new(
            CylinderDimensions::default(),
            BuildPose::default(),
        ));
        let solid = evaluate_part_solid(spec, []).unwrap();
        assert!(
            solid
                .logical_edges
                .iter()
                .filter(|edge| edge.closed)
                .count()
                >= 2
        );
        assert!(solid.logical_edges.len() < 24);
        assert!(solid.volume() > 0.0);
    }

    #[test]
    fn pipe_bend_generator_is_manifold_for_solid_and_hollow_profiles() {
        for dimensions in [
            PipeBendDimensions::new(0.25, 0.0, 0.25).unwrap(),
            PipeBendDimensions::default(),
        ] {
            let solid = evaluate_part_solid(
                PartSpec::PipeBend(PipeBendSpec::new(dimensions, BuildPose::default())),
                [],
            )
            .unwrap();
            assert!(!solid.cells.is_empty());
            assert!(solid.logical_edges.iter().any(|edge| edge.closed));
        }
    }
}
