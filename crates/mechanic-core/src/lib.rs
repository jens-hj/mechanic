//! Editable construction data and deterministic compilation into physics rows.

mod compile;
mod creation;
mod drive;
mod geometry;
mod graph;
mod id;

pub use compile::{
    CYLINDER_COLLIDER_COUNT, CompiledBearing, CompiledCompound, CompiledCreation, CoordinateDrive,
    DriveMode, LocalCuboidCollider, LoopTopology, MassProperties, MechanismBodyTopology,
    TopologyError,
};
pub use creation::{
    BearingDoc, BearingSocket, BearingSocketDoc, CREATION_FORMAT_VERSION, CreationDocument,
    CreationError, DriveDwellDoc, DriveLimitsDoc, DriveLinkDoc, DriveProgramDoc, DriveStateDoc,
    DriveTriggerDoc, FaceOwnerDoc, FaceRefDoc, InputSeatLinkDoc, LoadedCreation, PartDoc, PoseDoc,
    RigidLinkDoc, SeatControllerLinkDoc, WeldDoc,
};
pub use drive::{
    ActuatorAssignment, ActuatorPercentageError, DriveDwell, DriveKey, DriveLimits,
    DriveLimitsError, DriveName, DriveProgram, DriveProgramError, DriveRelease, DriveState,
    DriveTarget, DriveTrigger, MAX_DRIVE_DWELL_SECONDS, MAX_DRIVE_LIMIT_RADIANS,
    MAX_DRIVE_NAME_BYTES, MAX_DRIVE_SPEED_RAD_S, MAX_DRIVE_STATES,
};
pub use geometry::{
    Axis, BuildPose, CYLINDER_SWEEP_STEP_DEGREES, ControllerSpec, CuboidSpec,
    CylinderDimensionError, CylinderDimensions, CylinderSpec, DimensionError, EngineKind,
    EngineSpec, FaceKind, FaceOwner, FaceRef, GRID_UNIT_METERS, GridDimension, GridRotation,
    InputSpec, MAX_CYLINDER_OUTER_DIAMETER, MAX_CYLINDER_SWEEP_DEGREES, MAX_GRID_UNITS,
    MIN_CYLINDER_DIAMETER_GAP, MIN_CYLINDER_OUTER_DIAMETER, MIN_CYLINDER_SWEEP_DEGREES, PartSpec,
    SeatSpec, ServoSpec, snap_world_to_grid,
};
pub use graph::{
    ActuatorInventory, BearingDimensionError, BearingDimensions, BearingSpec, BuildCommand,
    BuildOutcome, ConstructionGraph, DriveLinkSpec, GraphError, InputSeatLinkSpec,
    MAX_BEARING_OUTER_DIAMETER, MIN_BEARING_DIAMETER_GAP, MIN_BEARING_OUTER_DIAMETER,
    PendingOperation, RigidLinkSpec, SeatControllerLinkSpec, WeldSpec,
};
pub use id::{
    BearingId, DriveLinkId, InputSeatLinkId, PartId, RigidLinkId, SeatControllerLinkId, WeldId,
};

/// Fixed construction-material density, in kg/m³.
pub const CUBOID_DENSITY_KG_M3: f32 = 500.0;

/// Maximum acceptable derived bearing-anchor separation, in metres.
pub const ANCHOR_TOLERANCE_METERS: f32 = 0.000_01;

/// Maximum acceptable derived bearing-axis separation, in degrees.
pub const AXIS_TOLERANCE_DEGREES: f32 = 0.001;
