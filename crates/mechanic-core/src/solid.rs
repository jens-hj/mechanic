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
// Independently clipped neighboring cells can differ below float render
// precision. Ten-micrometre keys stitch those seams without approaching the
// 2.5 mm authored position grid.
const KEY_SCALE: f64 = 100_000.0;
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
    /// Radial material band of a layered cylinder; zero for every other solid.
    pub band: u8,
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
    /// Radial material band of a layered cylinder; zero for every other solid.
    pub band: u8,
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
    // Facet identity remains distinct for picking and UV provenance.
    patch: SurfacePatchKey,
    // Family identity joins tessellated facets into one logical boundary.
    family: SurfacePatchKey,
    smoothing_group: u32,
    // Families joined tangentially to this facet are not sharp edges.
    smooth_with: Vec<SurfacePatchKey>,
    uv_provenance: SurfacePatchKey,
}

#[derive(Clone, Debug)]
struct PolyCell {
    faces: Vec<PolyFace>,
    // Radial material band of a layered cylinder; zero everywhere else.
    band: u8,
}

#[derive(Clone)]
struct ClipPlane {
    normal: DVec3,
    offset: f64,
    patch: SurfacePatchKey,
    family: SurfacePatchKey,
    smoothing_group: u32,
    smooth_with: Vec<SurfacePatchKey>,
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
    first_family: SurfacePatchKey,
    second_family: SurfacePatchKey,
    uv_provenance: SurfacePatchKey,
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
        PartSpec::PipeJunction(junction) => pipe_junction_cells(junction),
        PartSpec::Controller(_)
        | PartSpec::Engine(_)
        | PartSpec::Transmission(_)
        | PartSpec::Servo(_)
        | PartSpec::Seat(_)
        | PartSpec::Input(_)
        | PartSpec::DimensionLink(_) => return Err(SolidError::AuthoredPart),
    };
    if spec.is_layered() {
        return build_evaluated(&partition_layers(replay_features(cells, features)?, spec));
    }
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
    cells: Vec<PolyCell>,
    features: impl IntoIterator<Item = (ShapeFeatureId, ShapeFeature)>,
) -> Result<EvaluatedSolid, SolidError> {
    build_evaluated(&replay_features(cells, features)?)
}

fn replay_features(
    mut cells: Vec<PolyCell>,
    features: impl IntoIterator<Item = (ShapeFeatureId, ShapeFeature)>,
) -> Result<Vec<PolyCell>, SolidError> {
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
        let profiles = vertex_profiles(feature.treatment, amount, &segments, &selected);
        for segment in segments
            .iter()
            .filter(|segment| selected.contains(&segment.key))
        {
            append_edge_profile_planes(
                feature_id,
                feature.treatment,
                &profiles,
                segment,
                &mut planes_by_cell,
            );
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
        snap_to_profile_points(&mut cells, &profiles);
        cells.retain(|cell| cell_volume(cell) > EPSILON);
        if cells.is_empty() {
            return Err(SolidError::AmountTooLarge(feature_id));
        }
    }
    Ok(cells)
}

/// Treatment cross-sections shared by every segment meeting at a chain vertex,
/// keyed by segment half-edge and vertex.
type VertexProfiles = BTreeMap<(u32, PointKey), Vec<DVec3>>;

/// Computes each selected segment's cross-section at both of its endpoints.
///
/// As in Blender's bevel, boundary points are placed once per vertex and both
/// strips meeting there reuse them. A chain crossing convex-cell seams, such as
/// a cylinder rim with one wedge cell per segment, is then cut on either side
/// of each seam along the same polyline, so the interior seam faces still
/// cancel when the boundary is stitched. Sampling the profile per segment
/// instead left each side with its own arc wherever exact symmetry was lost.
fn vertex_profiles(
    treatment: EdgeTreatment,
    amount: f64,
    segments: &[EdgeSegment],
    selected: &BTreeSet<TopologyKey>,
) -> VertexProfiles {
    let mut incident = BTreeMap::<(TopologyKey, PointKey), Vec<(&EdgeSegment, DVec3)>>::new();
    for segment in segments
        .iter()
        .filter(|segment| selected.contains(&segment.key))
    {
        for point in [segment.a, segment.b] {
            incident
                .entry((segment.key, point_key(point)))
                .or_default()
                .push((segment, point));
        }
    }
    // One facet count per chain. Posed parts measure a right-angle rim a hair
    // either side of 90°, and vertices that disagreed on the count could not
    // be joined by strip facets.
    let mut chain_facets = BTreeMap::<TopologyKey, usize>::new();
    for segment in segments
        .iter()
        .filter(|segment| selected.contains(&segment.key))
    {
        let angle = segment
            .first_normal
            .dot(segment.second_normal)
            .clamp(-1.0, 1.0)
            .acos();
        let count = chain_facets.entry(segment.key).or_insert(1);
        *count = (*count).max(fillet_facets(angle));
    }
    // Inside a chain a vertex takes the mean of both segments' face normals,
    // which on a faceted rim is the true surface normal there.
    let mut chain_normals = BTreeMap::<(TopologyKey, PointKey), (DVec3, DVec3)>::new();
    for (&(key, vertex_key), uses) in &incident {
        if let [(first, _), (second, _)] = uses.as_slice() {
            let first_normal = (first.first_normal + second.first_normal).normalize_or_zero();
            let second_normal = (first.second_normal + second.second_normal).normalize_or_zero();
            if first_normal != DVec3::ZERO && second_normal != DVec3::ZERO {
                chain_normals.insert((key, vertex_key), (first_normal, second_normal));
            }
        }
    }
    let mut profiles = VertexProfiles::new();
    for ((key, vertex_key), uses) in incident {
        let facets = chain_facets.get(&key).copied().unwrap_or(1);
        for (segment, vertex) in uses {
            let (first_normal, second_normal) = chain_normals
                .get(&(key, vertex_key))
                .copied()
                .or_else(|| {
                    // An open chain ends at a vertex only one segment reaches.
                    // Continuing the normals' turn past that segment ends a
                    // curved rim on the surface normal at its last vertex, as
                    // a whole rim would, instead of on its last facet's normal.
                    let other = if point_key(segment.a) == vertex_key {
                        segment.b
                    } else {
                        segment.a
                    };
                    chain_normals
                        .get(&(key, point_key(other)))
                        .map(|&(first, second)| {
                            (
                                slerp_unit(first, segment.first_normal, 2.0),
                                slerp_unit(second, segment.second_normal, 2.0),
                            )
                        })
                })
                .unwrap_or((segment.first_normal, segment.second_normal));
            if let Some(profile) = vertex_profile(
                treatment,
                amount,
                facets,
                vertex,
                first_normal,
                second_normal,
            ) {
                profiles.insert((segment.half_edge, vertex_key), profile);
            }
        }
    }
    profiles
}

/// Cross-section of a treatment at `vertex`, from the first face to the second.
///
/// A fillet samples its circular arc from one face tangency to the other; a
/// chamfer is the straight cut between the two setback points.
/// `facets` is shared by the whole chain so neighbouring profiles pair up
/// step for step; chamfers ignore it.
fn vertex_profile(
    treatment: EdgeTreatment,
    amount: f64,
    facets: usize,
    vertex: DVec3,
    first_normal: DVec3,
    second_normal: DVec3,
) -> Option<Vec<DVec3>> {
    let dot = first_normal.dot(second_normal).clamp(-1.0, 1.0);
    let angle = dot.acos();
    if !(1.0e-6..=core::f64::consts::PI - 1.0e-6).contains(&angle) {
        return None;
    }
    Some(match treatment {
        EdgeTreatment::Chamfer => {
            let first_inward = -(second_normal - first_normal * dot).normalize();
            let second_inward = -(first_normal - second_normal * dot).normalize();
            vec![
                vertex + first_inward * amount,
                vertex + second_inward * amount,
            ]
        }
        EdgeTreatment::Fillet => {
            let facets = facets.max(1);
            let centre = vertex - (first_normal + second_normal) * (amount / (1.0 + dot));
            (0..=facets)
                .map(|step| {
                    centre
                        + slerp_unit(first_normal, second_normal, step as f64 / facets as f64)
                            * amount
                })
                .collect()
        }
    })
}

/// Facets a fillet needs across a dihedral `angle`, tolerating float noise
/// so a right-angle edge always gets the same count.
fn fillet_facets(angle: f64) -> usize {
    ((angle.to_degrees() / FILLET_MAX_FACET_DEGREES - 1.0e-6).ceil() as usize).max(1)
}

fn append_edge_profile_planes(
    feature: ShapeFeatureId,
    treatment: EdgeTreatment,
    profiles: &VertexProfiles,
    segment: &EdgeSegment,
    planes_by_cell: &mut BTreeMap<usize, Vec<ClipPlane>>,
) {
    let (Some(start), Some(end)) = (
        profiles.get(&(segment.half_edge, point_key(segment.a))),
        profiles.get(&(segment.half_edge, point_key(segment.b))),
    ) else {
        return;
    };
    if start.len() != end.len() {
        return;
    }
    let steps = start.len() - 1;
    let outward = segment.first_normal + segment.second_normal;
    let planes = planes_by_cell.entry(segment.cell).or_default();
    for step_index in 0..steps {
        let profile_step = step_index + 1;
        let smooth_with: Vec<_> = if treatment == EdgeTreatment::Fillet {
            [
                (step_index == 0).then_some(segment.first_family),
                (step_index + 1 == steps).then_some(segment.second_family),
            ]
            .into_iter()
            .flatten()
            .collect()
        } else {
            Vec::new()
        };
        let facet = [
            start[step_index],
            start[step_index + 1],
            end[step_index + 1],
            end[step_index],
        ];
        for (normal, offset) in strip_facet_planes(facet, outward) {
            planes.push(ClipPlane {
                normal,
                offset,
                patch: generated_patch_key(feature, segment, profile_step),
                family: generated_patch_family_key(feature, segment.key, profile_step),
                smoothing_group: u32::from(treatment == EdgeTreatment::Fillet)
                    * feature.index().saturating_add(1),
                smooth_with: smooth_with.clone(),
                uv_provenance: segment.uv_provenance,
            });
        }
    }
}

/// Outward clip planes through one strip quad between two vertex profiles.
///
/// Rotational and straight chains give planar quads. Otherwise the quad is
/// split along the diagonal that keeps the pair convex seen from outside, so
/// the solid remains the intersection of the half-spaces.
fn strip_facet_planes(facet: [DVec3; 4], outward: DVec3) -> Vec<(DVec3, f64)> {
    let plane = |points: [DVec3; 3]| {
        let normal = (points[1] - points[0])
            .cross(points[2] - points[0])
            .normalize_or_zero();
        let normal = if normal.dot(outward) < 0.0 {
            -normal
        } else {
            normal
        };
        (normal, normal.dot(points[0]))
    };
    let newell = (0..4).fold(DVec3::ZERO, |sum, index| {
        let current = facet[index];
        let next = facet[(index + 1) % 4];
        sum + DVec3::new(
            (current.y - next.y) * (current.z + next.z),
            (current.z - next.z) * (current.x + next.x),
            (current.x - next.x) * (current.y + next.y),
        )
    });
    if let Some(normal) = newell.try_normalize() {
        let normal = if normal.dot(outward) < 0.0 {
            -normal
        } else {
            normal
        };
        let offset = facet.iter().map(|point| normal.dot(*point)).sum::<f64>() * 0.25;
        // Well under one stitch key, so a plane this far from a corner still
        // meets the neighbouring cell's cut at the same key.
        if facet
            .iter()
            .all(|point| (normal.dot(*point) - offset).abs() <= 1.0e-6)
        {
            return vec![(normal, offset)];
        }
    }
    for [first, second] in [[[0, 1, 2], [0, 2, 3]], [[0, 1, 3], [1, 2, 3]]] {
        let triangles = [first, second].map(|indices| plane(indices.map(|index| facet[index])));
        let opposite = [
            facet[(0..4).find(|index| !first.contains(index)).unwrap_or(0)],
            facet[(0..4).find(|index| !second.contains(index)).unwrap_or(0)],
        ];
        if triangles
            .iter()
            .zip(opposite)
            .all(|((normal, offset), point)| normal.dot(point) - offset <= 1.0e-9)
        {
            return triangles
                .into_iter()
                .filter(|(normal, _)| *normal != DVec3::ZERO)
                .collect();
        }
    }
    Vec::new()
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
        if distinct_edge_count(&incident_edges) < 3 {
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
        let smooth_with = incident_surface_families(&incident_edges);
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
                    incident_edges[0].uv_provenance,
                    &smooth_with,
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
                        incident_edges[0].uv_provenance,
                        &smooth_with,
                        planes_by_cell.entry(cell).or_default(),
                    );
                    ordinal += 1;
                }
            }
        }
    }
}

fn distinct_edge_count(segments: &[&EdgeSegment]) -> usize {
    segments
        .iter()
        .map(|edge| edge.key)
        .collect::<BTreeSet<_>>()
        .len()
}

fn incident_surface_families(segments: &[&EdgeSegment]) -> Vec<SurfacePatchKey> {
    segments
        .iter()
        .flat_map(|edge| [edge.first_family, edge.second_family])
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
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
    smooth_with: &[SurfacePatchKey],
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
        family: generated_junction_family_key(feature, vertex),
        smoothing_group: feature.index().saturating_add(1),
        smooth_with: smooth_with.to_vec(),
        uv_provenance,
    });
}

fn generated_junction_family_key(feature: ShapeFeatureId, vertex: DVec3) -> SurfacePatchKey {
    let mut hash = 2_166_136_261_u32;
    for value in [0x4a46_414d, point_word(point_key(vertex))] {
        hash ^= value;
        hash = hash.wrapping_mul(16_777_619);
    }
    SurfacePatchKey {
        source: TopologySource::Feature(feature),
        local: hash,
    }
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

/// Welds clipped vertices onto the vertex-profile points they were cut through.
///
/// Neighbouring cells reach the same profile point through different plane
/// intersections, which can land either side of a stitch-key boundary. Blender
/// avoids this by reusing one vertex; snapping both copies onto the shared
/// point does the same here.
fn snap_to_profile_points(cells: &mut [PolyCell], profiles: &VertexProfiles) {
    const SNAP_DISTANCE: f64 = 1.0e-6;
    let mut points = BTreeMap::<PointKey, Vec<DVec3>>::new();
    for &point in profiles.values().flatten() {
        let bucket = points.entry(point_key(point)).or_default();
        if !bucket
            .iter()
            .any(|existing| existing.distance_squared(point) <= EPSILON * EPSILON)
        {
            bucket.push(point);
        }
    }
    for vertex in cells
        .iter_mut()
        .flat_map(|cell| cell.faces.iter_mut())
        .flat_map(|face| face.vertices.iter_mut())
    {
        let [x, y, z] = point_key(*vertex).0;
        let nearest = (-1..=1)
            .flat_map(|dx| (-1..=1).flat_map(move |dy| (-1..=1).map(move |dz| [dx, dy, dz])))
            .filter_map(|[dx, dy, dz]| points.get(&PointKey([x + dx, y + dy, z + dz])))
            .flatten()
            .copied()
            .filter(|point| point.distance_squared(*vertex) <= SNAP_DISTANCE * SNAP_DISTANCE)
            .min_by(|left, right| {
                left.distance_squared(*vertex)
                    .total_cmp(&right.distance_squared(*vertex))
            });
        if let Some(point) = nearest {
            *vertex = point;
        }
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

fn generated_patch_family_key(
    feature: ShapeFeatureId,
    target: TopologyKey,
    step: usize,
) -> SurfacePatchKey {
    let mut hash = 2_166_136_261_u32;
    for value in [
        0x4641_4d49,
        topology_word(target),
        target.local,
        u32::try_from(step).unwrap_or(u32::MAX),
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
                family: face.family,
                smoothing_group: face.smoothing_group,
                smooth_with: face.smooth_with.clone(),
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
            family: plane.family,
            smoothing_group: plane.smoothing_group,
            smooth_with: plane.smooth_with,
            uv_provenance: plane.uv_provenance,
        });
    }
    let result = PolyCell {
        faces,
        band: cell.band,
    };
    (result.faces.len() >= 4 && cell_volume(&result) > EPSILON).then_some(result)
}

fn build_evaluated(cells: &[PolyCell]) -> Result<EvaluatedSolid, SolidError> {
    let (stitched, _) = stitch(cells)?;
    let volume_cells = cells
        .iter()
        .filter_map(|cell| {
            poly_cell_to_convex(cell).map(|piece| ConvexVolumeCell {
                piece,
                band: cell.band,
            })
        })
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
    let mut surface_families = Vec::<SurfacePatchKey>::new();
    let mut surface_continuity = Vec::<Vec<SurfacePatchKey>>::new();
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
            band: cells[cell_index].band,
        });
        surface_families.push(face.family);
        surface_continuity.push(face.smooth_with.clone());
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
        let twin_index = edge.twin;
        let first_face = edge.face as usize;
        let second_face = half_edges[twin_index as usize].face as usize;
        let first_surface = &surfaces[first_face];
        let second_surface = &surfaces[second_face];
        let first_family = surface_families[first_face];
        let second_family = surface_families[second_face];
        if first_family == second_family
            || (first_surface.smoothing_group != 0
                && first_surface.smoothing_group == second_surface.smoothing_group)
            || surface_continuity[first_face].contains(&second_family)
            || surface_continuity[second_face].contains(&first_family)
        {
            continue;
        }
        let key = topology_key(first_family, second_family);
        let (canonical_edge_index, canonical_twin_index) =
            if (first_family, first_surface.key) <= (second_family, second_surface.key) {
                (edge_index as u32, twin_index)
            } else {
                (twin_index, edge_index as u32)
            };
        logical.entry(key).or_default().push(canonical_edge_index);
        let canonical_edge = half_edges[canonical_edge_index as usize];
        let canonical_twin = half_edges[canonical_twin_index as usize];
        let canonical_first = &surfaces[canonical_edge.face as usize];
        let canonical_second = &surfaces[canonical_twin.face as usize];
        let a = DVec3::from(vertices[canonical_edge.origin as usize].position);
        let b = DVec3::from(
            vertices[half_edges[canonical_edge.next as usize].origin as usize].position,
        );
        let first_normal = DVec3::from(canonical_first.normal);
        let second_normal = DVec3::from(canonical_second.normal);
        let tangent = (b - a).normalize();
        let convex = first_normal.cross(second_normal).dot(tangent) > EPSILON;
        segments.push(EdgeSegment {
            key,
            half_edge: canonical_edge_index,
            a,
            b,
            first_normal,
            second_normal,
            first_family: surface_families[canonical_edge.face as usize],
            second_family: surface_families[canonical_twin.face as usize],
            uv_provenance: canonical_first.uv_provenance,
            cell: half_edge_cells[canonical_edge_index as usize],
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

const fn topology_word(key: TopologyKey) -> u32 {
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

fn convex_piece_cell(piece: ConvexPiece) -> PolyCell {
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
        family: key,
        smoothing_group: 0,
        smooth_with: Vec::new(),
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

/// Divides a layered part's replayed envelope cells into material bands.
///
/// Features replay on the whole envelope first, so a chamfer or fillet cuts
/// through every layer it reaches. Each layer then claims whatever lies beyond
/// the envelope it was laid on, later layers winning. A flat layer's boundary
/// is one plane. Inside one 15-degree wedge a wall or bore boundary is a single
/// chord plane, the same polygon the wedge walls use. Either way both halves
/// of a split share an identical face and stitch away as interior.
#[allow(clippy::too_many_lines)] // One split per layer reads better whole.
fn partition_layers(cells: Vec<PolyCell>, spec: PartSpec) -> Vec<PolyCell> {
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

/// One convex cell per junction ray triangle: a pyramid to the centre when
/// solid, or the shell between the bore and the outside when hollow.
fn pipe_junction_cells(spec: crate::PipeJunctionSpec) -> Vec<PolyCell> {
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

const fn junction_face_patch(face: crate::FaceKind) -> u32 {
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
fn shell_cell(mut outer: Vec<DVec3>, mut inner: Vec<DVec3>, patches: [u32; 3]) -> PolyCell {
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
        let face =
            without_repeated_vertices(vec![bottom[next], bottom[index], top[index], top[next]]);
        if face.len() >= 3 {
            faces.push(base_face(face, local));
        }
    }
    PolyCell { faces, band: 0 }
}

/// Collapses vertices shared by a pinched profile, such as the crease of a
/// pipe bend whose inner wall meets at the centre of curvature.
fn without_repeated_vertices(mut face: Vec<DVec3>) -> Vec<DVec3> {
    face.dedup_by(|next, previous| next.distance_squared(*previous) <= EPSILON * EPSILON);
    if face.len() > 1 && face[0].distance_squared(face[face.len() - 1]) <= EPSILON * EPSILON {
        face.pop();
    }
    face
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
    fn vertex_profile_endpoints_are_face_tangencies() {
        let radius = 0.05;
        let profile = vertex_profile(
            EdgeTreatment::Fillet,
            radius,
            fillet_facets(core::f64::consts::FRAC_PI_2),
            DVec3::ZERO,
            DVec3::X,
            DVec3::Y,
        )
        .unwrap();
        let centre = DVec3::new(-radius, -radius, 0.0);

        assert_eq!(profile.len(), 13, "a right-angle fillet has twelve facets");
        assert!(profile[0].abs_diff_eq(centre + DVec3::X * radius, EPSILON));
        assert!(profile[12].abs_diff_eq(centre + DVec3::Y * radius, EPSILON));
        assert!(
            profile
                .iter()
                .all(|point| (point.distance(centre) - radius).abs() < EPSILON)
        );
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

    fn cylinder_spec(inner_diameter: f32, sweep_degrees: u16) -> PartSpec {
        let dimensions = CylinderDimensions::new(0.5, inner_diameter, 0.5)
            .unwrap()
            .with_sweep_angle_degrees(sweep_degrees)
            .unwrap();
        PartSpec::Cylinder(CylinderSpec::new(dimensions, BuildPose::default()))
    }

    fn target(owner: SolidOwner, edge: TopologyKey) -> EdgeChainRef {
        EdgeChainRef { owner, edge }
    }

    #[test]
    fn solid_and_hollow_cylinder_rims_accept_five_centimetre_treatments() {
        let owner = SolidOwner::Part(crate::PartId::from_parts(0, 0));
        for spec in [cylinder_spec(0.0, 360), cylinder_spec(0.25, 360)] {
            let base = evaluate_part_solid(spec, []).unwrap();
            let rims = base
                .logical_edges
                .iter()
                .filter(|edge| edge.closed && edge.convex)
                .map(|edge| edge.key)
                .collect::<Vec<_>>();
            assert_eq!(
                rims.len(),
                if matches!(spec, PartSpec::Cylinder(cylinder) if cylinder.dimensions.inner_diameter() > 0.0)
                {
                    4
                } else {
                    2
                }
            );
            assert!(rims.iter().all(|key| {
                base.logical_edge(*key)
                    .is_some_and(|edge| edge.half_edges.len() == 24)
            }));
            for (edge_index, edge) in rims.into_iter().enumerate() {
                for treatment in [EdgeTreatment::Chamfer, EdgeTreatment::Fillet] {
                    let feature = ShapeFeatureId::from_parts(edge_index as u32, 0);
                    evaluate_part_solid(
                        spec,
                        [(
                            feature,
                            ShapeFeature::new([target(owner, edge)], treatment, 20),
                        )],
                    )
                    .unwrap_or_else(|error| {
                        panic!("{treatment:?} rejected cylinder rim {edge_index}: {error}")
                    });
                }
            }
        }
    }

    #[test]
    fn translated_and_rotated_cylinder_rims_accept_treatments() {
        let owner = SolidOwner::Part(crate::PartId::from_parts(0, 0));
        let poses = [
            BuildPose::new(IVec3::new(1, 4, 0), GridRotation::default()),
            BuildPose::new(IVec3::ZERO, GridRotation::new(0, 0, 3)),
            BuildPose::new(IVec3::new(-3, 2, 7), GridRotation::new(1, 2, 0)),
        ];
        for pose in poses {
            for (inner_diameter, sweep_degrees) in [(0.0, 360), (0.25, 360), (0.25, 90)] {
                let dimensions = CylinderDimensions::new(0.5, inner_diameter, 0.5)
                    .unwrap()
                    .with_sweep_angle_degrees(sweep_degrees)
                    .unwrap();
                let spec = PartSpec::Cylinder(CylinderSpec::new(dimensions, pose));
                let base = evaluate_part_solid(spec, []).unwrap();
                let rims = base
                    .logical_edges
                    .iter()
                    .filter(|edge| edge.convex && edge.half_edges.len() > 1)
                    .map(|edge| edge.key)
                    .collect::<Vec<_>>();
                assert!(!rims.is_empty());
                for edge in rims {
                    for treatment in [EdgeTreatment::Chamfer, EdgeTreatment::Fillet] {
                        let treated = evaluate_part_solid(
                            spec,
                            [(
                                ShapeFeatureId::from_parts(0, 0),
                                ShapeFeature::new([target(owner, edge)], treatment, 20),
                            )],
                        )
                        .unwrap_or_else(|error| {
                            panic!("{treatment:?} rejected a rim of {dimensions:?} at {pose:?}: {error}")
                        });
                        assert!(treated.volume() > 0.0 && treated.volume() < base.volume());
                    }
                }
            }
        }
    }

    #[test]
    fn posed_cylinder_fillet_vertices_lie_on_the_rim_torus() {
        let owner = SolidOwner::Part(crate::PartId::from_parts(0, 0));
        let spec = PartSpec::Cylinder(CylinderSpec::new(
            CylinderDimensions::new(0.5, 0.0, 0.5).unwrap(),
            BuildPose::new(IVec3::new(1, 4, 0), GridRotation::default()),
        ));
        let axis = DVec3::new(0.25, 1.0, 0.0);
        let (outer_radius, half_length, fillet_radius) = (0.25, 0.25, 0.05);
        let base = evaluate_part_solid(spec, []).unwrap();
        for rim in base
            .logical_edges
            .iter()
            .filter(|edge| edge.closed && edge.convex)
        {
            let feature = ShapeFeatureId::from_parts(0, 0);
            let filleted = evaluate_part_solid(
                spec,
                [(
                    feature,
                    ShapeFeature::new([target(owner, rim.key)], EdgeTreatment::Fillet, 20),
                )],
            )
            .unwrap();
            let mut checked = 0;
            for surface in filleted
                .surfaces
                .iter()
                .filter(|surface| surface.key.source == TopologySource::Feature(feature))
            {
                let mut edge = surface.half_edge;
                loop {
                    let half_edge = filleted.half_edges[edge as usize];
                    let offset = filleted.vertices[half_edge.origin as usize]
                        .position
                        .as_dvec3()
                        - axis;
                    let radial = offset.x.hypot(offset.z) - (outer_radius - fillet_radius);
                    let axial = offset.y.abs() - (half_length - fillet_radius);
                    assert!(
                        (radial.hypot(axial) - fillet_radius).abs() < 1.0e-5,
                        "fillet vertex {offset:?} is off the rim torus"
                    );
                    checked += 1;
                    edge = half_edge.next;
                    if edge == surface.half_edge {
                        break;
                    }
                }
            }
            assert!(checked > 0);
        }
    }

    #[test]
    fn hollow_sector_curved_rims_and_axial_cut_edges_accept_treatments() {
        let spec = cylinder_spec(0.25, 90);
        let owner = SolidOwner::Part(crate::PartId::from_parts(0, 0));
        let base = evaluate_part_solid(spec, []).unwrap();
        let curved_rims = base
            .logical_edges
            .iter()
            .filter(|edge| !edge.closed && edge.convex && edge.half_edges.len() > 1)
            .map(|edge| edge.key)
            .collect::<Vec<_>>();
        assert_eq!(curved_rims.len(), 4);
        let axial_cuts = base
            .logical_edges
            .iter()
            .filter(|edge| {
                edge.convex && edge.half_edges.len() == 1 && {
                    let half_edge = base.half_edges[edge.half_edges[0] as usize];
                    let next = base.half_edges[half_edge.next as usize];
                    let a = base.vertices[half_edge.origin as usize].position;
                    let b = base.vertices[next.origin as usize].position;
                    (a.y - b.y).abs() > 0.49
                }
            })
            .map(|edge| edge.key)
            .collect::<Vec<_>>();
        assert_eq!(axial_cuts.len(), 4);

        for (index, edge) in curved_rims.iter().copied().enumerate() {
            for treatment in [EdgeTreatment::Chamfer, EdgeTreatment::Fillet] {
                let feature = ShapeFeatureId::from_parts(index as u32, 0);
                evaluate_part_solid(
                    spec,
                    [(
                        feature,
                        ShapeFeature::new([target(owner, edge)], treatment, 20),
                    )],
                )
                .unwrap_or_else(|error| panic!("sector {treatment:?} failed: {error}"));
            }
        }
        for (index, edge) in axial_cuts.iter().copied().enumerate() {
            for treatment in [EdgeTreatment::Chamfer, EdgeTreatment::Fillet] {
                let feature = ShapeFeatureId::from_parts(index as u32 + 4, 0);
                evaluate_part_solid(
                    spec,
                    [(
                        feature,
                        ShapeFeature::new([target(owner, edge)], treatment, 5),
                    )],
                )
                .unwrap_or_else(|error| panic!("sector axial {treatment:?} failed: {error}"));
            }
        }

        let fillet_id = ShapeFeatureId::from_parts(8, 0);
        let filleted = evaluate_part_solid(
            spec,
            [(
                fillet_id,
                ShapeFeature::new([target(owner, curved_rims[0])], EdgeTreatment::Fillet, 20),
            )],
        )
        .unwrap();
        assert!(filleted.logical_edges.iter().any(|edge| {
            edge.key.source == TopologySource::Feature(fillet_id) && edge.convex && !edge.closed
        }));
    }

    #[test]
    fn chamfered_cylinder_rim_produces_two_treatable_closed_chains() {
        let spec = cylinder_spec(0.0, 360);
        let owner = SolidOwner::Part(crate::PartId::from_parts(0, 0));
        let base = evaluate_part_solid(spec, []).unwrap();
        let rim = base
            .logical_edges
            .iter()
            .find(|edge| edge.closed && edge.convex)
            .unwrap()
            .key;
        let chamfer_id = ShapeFeatureId::from_parts(0, 0);
        let chamfer = ShapeFeature::new([target(owner, rim)], EdgeTreatment::Chamfer, 20);
        let chamfered = evaluate_part_solid(spec, [(chamfer_id, chamfer.clone())]).unwrap();
        let generated = chamfered
            .logical_edges
            .iter()
            .filter(|edge| {
                edge.key.source == TopologySource::Feature(chamfer_id) && edge.closed && edge.convex
            })
            .map(|edge| edge.key)
            .collect::<Vec<_>>();
        assert_eq!(generated.len(), 2);
        assert!(generated.iter().all(|key| {
            chamfered
                .logical_edge(*key)
                .is_some_and(|edge| edge.half_edges.len() == 24)
        }));

        for edge in generated {
            for treatment in [EdgeTreatment::Chamfer, EdgeTreatment::Fillet] {
                let follow_up_id = ShapeFeatureId::from_parts(1, 0);
                let follow_up = ShapeFeature::new([target(owner, edge)], treatment, 20);
                evaluate_part_solid(
                    spec,
                    [(chamfer_id, chamfer.clone()), (follow_up_id, follow_up)],
                )
                .unwrap_or_else(|error| panic!("follow-up {treatment:?} failed: {error}"));
            }
        }
    }

    #[test]
    fn cylinder_fillet_tangencies_are_not_selectable_edges() {
        let spec = cylinder_spec(0.0, 360);
        let owner = SolidOwner::Part(crate::PartId::from_parts(0, 0));
        let base = evaluate_part_solid(spec, []).unwrap();
        let rim = base
            .logical_edges
            .iter()
            .find(|edge| edge.closed && edge.convex)
            .unwrap()
            .key;
        let feature = ShapeFeatureId::from_parts(0, 0);
        let filleted = evaluate_part_solid(
            spec,
            [(
                feature,
                ShapeFeature::new([target(owner, rim)], EdgeTreatment::Fillet, 20),
            )],
        )
        .unwrap();

        assert!(
            filleted
                .logical_edges
                .iter()
                .all(|edge| { edge.key.source != TopologySource::Feature(feature) })
        );
        assert!(
            filleted
                .logical_edges
                .iter()
                .any(|edge| edge.closed && edge.convex)
        );
    }

    #[test]
    fn pipe_bend_generator_is_manifold_for_solid_and_hollow_profiles() {
        for dimensions in [
            PipeBendDimensions::new(0.20, 0.0, 2).unwrap(),
            PipeBendDimensions::new(0.25, 0.0, 1).unwrap(),
            PipeBendDimensions::new(0.25, 0.10, 1).unwrap(),
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

    #[test]
    fn pipe_junction_generator_evaluates_every_arm_set_solid_and_hollow() {
        use crate::{PipeArms, PipeJunctionDimensions, PipeJunctionSpec};
        let (radius, reach) = (0.10_f64, 0.125_f64);
        for inner in [0.0, 0.10] {
            let bore = f64::from(inner) * 0.5;
            for bits in [0b00_0001, 0b00_0011, 0b01_0011, 0b11_1111] {
                let arms = PipeArms::from_bits(bits).unwrap();
                let spec = PipeJunctionSpec::new(
                    PipeJunctionDimensions::new(0.20, inner).unwrap(),
                    arms,
                    BuildPose::default(),
                );
                let solid = evaluate_part_solid(PartSpec::PipeJunction(spec), [])
                    .unwrap_or_else(|error| panic!("bits {bits:06b}, inner {inner}: {error}"));
                let volume = solid
                    .cells
                    .iter()
                    .map(|cell| f64::from(cell.piece.volume))
                    .sum::<f64>();
                // A lone arm is a pipe capped by half the centre ball; opposite
                // arms make one straight pipe across the cell.
                let expected = match bits {
                    0b00_0001 => {
                        core::f64::consts::PI
                            * ((radius * radius - bore * bore) * reach
                                + (radius.powi(3) - bore.powi(3)) * 2.0 / 3.0)
                    }
                    0b00_0011 => {
                        core::f64::consts::PI * (radius * radius - bore * bore) * 2.0 * reach
                    }
                    _ => continue,
                };
                assert!(
                    (volume - expected).abs() < expected * 0.03,
                    "bits {bits:06b}, inner {inner}: volume {volume} vs {expected}"
                );
            }
        }
    }

    fn layered_wheel() -> (PartSpec, PartSpec) {
        let core = CylinderSpec::new(
            CylinderDimensions::new(1.0, 0.0, 0.5).unwrap(),
            BuildPose::default(),
        );
        let layered = core
            .with_layer(
                crate::LayerFace::OuterWall,
                0.25,
                ConstructionMaterial::Rubber,
                crate::MaterialAppearance::BAKED,
            )
            .unwrap();
        let envelope = CylinderSpec::new(layered.dimensions, BuildPose::default());
        (PartSpec::Cylinder(layered), PartSpec::Cylinder(envelope))
    }

    #[test]
    fn layered_cylinder_band_interfaces_stitch_away() {
        let (layered, envelope) = layered_wheel();
        let plain = evaluate_part_solid(envelope, []).unwrap();
        let banded = evaluate_part_solid(layered, []).unwrap();
        assert!((banded.volume() - plain.volume()).abs() < 1.0e-4 * plain.volume());
        assert_eq!(banded.logical_edges.len(), plain.logical_edges.len());
        assert!(banded.cells.iter().any(|cell| cell.band == 0));
        assert!(banded.cells.iter().any(|cell| cell.band == 1));
    }

    #[test]
    fn fillet_deeper_than_the_outer_layer_cuts_through_both_bands() {
        let owner = SolidOwner::Part(crate::PartId::from_parts(0, 0));
        let (layered, envelope) = layered_wheel();
        let rim = evaluate_part_solid(envelope, [])
            .unwrap()
            .logical_edges
            .iter()
            .find(|edge| edge.closed && edge.convex)
            .unwrap()
            .key;
        let fillet = [(
            ShapeFeatureId::from_parts(0, 0),
            ShapeFeature::new([target(owner, rim)], EdgeTreatment::Fillet, 120),
        )];
        let plain = evaluate_part_solid(envelope, fillet.clone()).unwrap();
        let banded = evaluate_part_solid(layered, fillet).unwrap();
        assert!((banded.volume() - plain.volume()).abs() < 1.0e-4 * plain.volume());
        assert_eq!(banded.logical_edges.len(), plain.logical_edges.len());
        let rounded = |band| {
            banded
                .surfaces
                .iter()
                .any(|surface| surface.smoothing_group != 0 && surface.band == band)
        };
        assert!(rounded(0), "a 30 cm fillet reaches the steel core");
        assert!(rounded(1), "the fillet also rounds the rubber");
    }

    fn rubber(spec: PartSpec, face: crate::LayerFace, thickness: f32) -> PartSpec {
        spec.with_layer(
            face,
            thickness,
            ConstructionMaterial::Rubber,
            crate::MaterialAppearance::BAKED,
        )
        .unwrap()
    }

    #[test]
    fn cap_and_wall_layer_interfaces_stitch_away() {
        let core = PartSpec::Cylinder(CylinderSpec::new(
            CylinderDimensions::new(1.0, 0.5, 0.5).unwrap(),
            BuildPose::default(),
        ));
        let layered = rubber(
            rubber(
                rubber(core, crate::LayerFace::Face(FaceKind::PositiveY), 0.1),
                crate::LayerFace::OuterWall,
                0.25,
            ),
            crate::LayerFace::Bore,
            0.05,
        );
        let cylinder = layered.as_cylinder().unwrap();
        let envelope = PartSpec::Cylinder(CylinderSpec::new(cylinder.dimensions, cylinder.pose));
        let plain = evaluate_part_solid(envelope, []).unwrap();
        let banded = evaluate_part_solid(layered, []).unwrap();
        assert!((banded.volume() - plain.volume()).abs() < 1.0e-4 * plain.volume());
        assert_eq!(banded.logical_edges.len(), plain.logical_edges.len());
        for band in 0..=3 {
            assert!(
                banded.cells.iter().any(|cell| cell.band == band),
                "band {band} has cells"
            );
        }
    }

    #[test]
    fn fillet_deeper_than_a_face_layer_cuts_through_both_bands() {
        let owner = SolidOwner::Part(crate::PartId::from_parts(0, 0));
        let core = PartSpec::Cuboid(CuboidSpec::new([4, 4, 4], BuildPose::default()).unwrap());
        let layered = rubber(core, crate::LayerFace::Face(FaceKind::PositiveY), 0.25);
        let envelope = PartSpec::Cuboid(CuboidSpec::new([4, 5, 4], layered.pose()).unwrap());
        let plain_base = evaluate_part_solid(envelope, []).unwrap();
        // The top edge along x on the +z side.
        let top_edge = plain_base
            .logical_edges
            .iter()
            .find(|edge| {
                edge.half_edges.iter().all(|&half_edge| {
                    let origin = plain_base.half_edges[half_edge as usize].origin;
                    let position = plain_base.vertices[origin as usize].position;
                    position.y > 0.6 && position.z > 0.4
                })
            })
            .unwrap()
            .key;
        let fillet = [(
            ShapeFeatureId::from_parts(0, 0),
            ShapeFeature::new([target(owner, top_edge)], EdgeTreatment::Fillet, 120),
        )];
        let plain = evaluate_part_solid(envelope, fillet.clone()).unwrap();
        let banded = evaluate_part_solid(layered, fillet).unwrap();
        assert!((banded.volume() - plain.volume()).abs() < 1.0e-4 * plain.volume());
        assert_eq!(banded.logical_edges.len(), plain.logical_edges.len());
        let rounded = |band| {
            banded
                .surfaces
                .iter()
                .any(|surface| surface.smoothing_group != 0 && surface.band == band)
        };
        assert!(
            rounded(0),
            "a 30 cm fillet reaches the core under a 25 cm layer"
        );
        assert!(rounded(1), "the fillet also rounds the layer");
    }
}
