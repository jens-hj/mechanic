//! Fixtures shared by tests across modules.

use std::sync::Arc;

use crate::{TerrainField, WorldSeed, WorldgenSpec};

/// The embedded world without trees, for tests of the ground's own shape.
pub(crate) fn treeless_field(seed: WorldSeed) -> TerrainField {
    let spec = WorldgenSpec::embedded().without_flora();
    TerrainField::from_spec(seed, Arc::new(spec)).expect("the embedded world compiles")
}
