//! Density queries, kinematic walking, and terrain-foundation support.

#![expect(
    clippy::cast_possible_truncation,
    reason = "validated prototype dimensions become grid counts"
)]

mod capsule;
mod foundation;
mod scene;

pub use capsule::{
    KinematicCapsule, KinematicCapsuleConfig, KinematicContactReaction, KinematicInput,
    KinematicSupport, KinematicTickOutcome,
};
pub use foundation::{
    FoundationRefresh, FoundationSample, FoundationSpatialIndex, FoundationSupport,
};
pub use scene::{
    ActiveTerrainScene, TerrainDensity, TerrainScene, TerrainSpatialIndex, raycast_density,
};

#[cfg(test)]
mod tests;
