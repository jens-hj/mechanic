//! Trees grown from a species genome.
//!
//! Every tree, from a bamboo clump to a weeping willow, comes from one
//! recursive branching process steered by a [`SpeciesSpec`]: how tall and wide
//! it grows, how thick, how many stems, where the crown starts, how strongly
//! the leader dominates, how often and how many ways it splits and at what
//! angle, whether its twigs seek the sun or hang, how much it wanders, how its
//! foliage clothes the twigs, and how its roots spread. Radii follow the pipe
//! model, so recursion ends by itself once a branch is thinner than a terrain
//! cell. See `docs/flora.md`.

mod forest;
mod grow;
mod metrics;
mod model;
mod noise;
mod species;
mod sweep;

pub(crate) use self::forest::{
    Forest, ForestLayer, ForestSpecies, TREE_DENSITY_CEILING, Tree, TreeHit, TreeInstance,
};
pub use self::grow::grow_tree;
pub use self::metrics::TreeMetrics;
pub(crate) use self::model::Parts;
pub use self::model::{Axis, FoliageBlob, Part, Segment, TreeModel};
pub use self::species::{FoliageSpec, RootsSpec, SpeciesSpec};
pub use self::sweep::GenomeSweep;

use crate::TERRAIN_CELL_METERS;

/// Nodes along a full-height stem; shorter axes get proportionally fewer.
const INTERNODES: f64 = 20.0;

/// Azimuth step between successive splits on one axis.
const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;

/// Thinnest branch: half a terrain cell, the finest wood the field can hold.
const TWIG_RADIUS: f64 = TERRAIN_CELL_METERS * 0.5;

/// Branching depth beyond which no axis splits again.
const MAX_ORDER: u8 = 6;

/// Segment budget per tree; splitting stops once it is spent.
const MAX_SEGMENTS: usize = 20_000;

/// Direction change per node of a fully flexible branch at tropism 1.
const TROPISM_RATE: f64 = 0.45;

/// Direction change per segment at wobble 1.
const WOBBLE_RATE: f64 = 0.35;

/// Main roots leaving the base.
const ROOT_COUNT: u32 = 5;

/// Wavelength of the holes that thin foliage, in metres: leaf clumps rather
/// than a sponge, whose surface would cost far more triangles.
const FOLIAGE_NOISE_SCALE: f64 = 0.4;

/// Root flare: how much wider the trunk is at the ground, and over what height.
const FLARE_WIDENING: f64 = 1.4;
const FLARE_HEIGHT: f64 = 0.3;

#[cfg(test)]
mod tests;
