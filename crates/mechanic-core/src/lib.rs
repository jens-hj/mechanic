//! Editable construction data and deterministic compilation into physics rows.

mod compile;
mod geometry;
mod graph;
mod id;

pub use compile::{
    CompiledBearing, CompiledCompound, CompiledCreation, LocalCuboidCollider, LoopTopology,
    MassProperties, MechanismBodyTopology, TopologyError,
};
pub use geometry::{
    Axis, BuildPose, CuboidSpec, DimensionError, FaceKind, FaceOwner, FaceRef, GRID_UNIT_METERS,
    GridDimension, GridRotation, MAX_GRID_UNITS, snap_world_to_grid,
};
pub use graph::{
    BearingSpec, BuildCommand, BuildOutcome, ConstructionGraph, GraphError, PendingOperation,
    WeldSpec,
};
pub use id::{BearingId, PartId, WeldId};

/// Fixed density used for every cuboid in this milestone, in kg/m³.
pub const CUBOID_DENSITY_KG_M3: f32 = 500.0;

/// Maximum acceptable derived bearing-anchor separation, in metres.
pub const ANCHOR_TOLERANCE_METERS: f32 = 0.000_01;

/// Maximum acceptable derived bearing-axis separation, in degrees.
pub const AXIS_TOLERANCE_DEGREES: f32 = 0.001;
