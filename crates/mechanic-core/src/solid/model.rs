//! The evaluated solid: topology keys, features, boundary, patches, and volume cells.

use crate::{ConvexPiece, RegionId, ShapeFeatureId};
use bevy_math::Vec3;
use serde::{Deserialize, Serialize};
use thiserror::Error;

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
