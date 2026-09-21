//! Parametric construction-solid evaluation.
//!
//! Ordinary construction geometry is represented twice: a manifold boundary
//! for rendering and selection, and disjoint convex cells for mass and
//! collision.  Feature references name logical edges rather than tessellation
//! segments, so a rounded cylinder rim remains one selectable chain.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod cells;
mod features;
mod model;
mod polygon;
mod spiral;
mod stitch;

use cells::{
    cylinder_cells, partition_layers, pieces_to_cells, pipe_bend_cells, pipe_junction_cells,
};
use features::replay_features;
pub use model::{
    BoundaryHalfEdge, BoundaryVertex, ConvexVolumeCell, EdgeChainRef, EdgeTreatment,
    EvaluatedSolid, LogicalEdge, ShapeFeature, SolidError, SolidOwner, SurfacePatch,
    SurfacePatchKey, TopologyKey, TopologySource,
};
use polygon::PolyCell;
pub use spiral::{SpiralCore, spiral_core, spiral_pieces};
use stitch::build_evaluated;

use crate::{PartSpec, ShapeFeatureId, ShapeRegion, decompose, decompose_part};

/// Evaluates one ordinary construction part with features already filtered to
/// that owner and supplied in global order.
///
/// # Errors
///
/// Returns an error when the part is authored rather than construction geometry
/// or carries a spiral, its base boundary is invalid, or an ordered feature cannot be replayed.
pub fn evaluate_part_solid(
    spec: PartSpec,
    features: impl IntoIterator<Item = (ShapeFeatureId, ShapeFeature)>,
) -> Result<EvaluatedSolid, SolidError> {
    let cells = match spec {
        PartSpec::Cuboid(cuboid) => pieces_to_cells(decompose_part(cuboid)),
        PartSpec::Cylinder(cylinder) if cylinder.spiral().is_some() => {
            return Err(SolidError::SpiralPart);
        }
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

#[cfg(test)]
mod tests;
