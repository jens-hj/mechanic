//! Editable construction data and deterministic compilation into physics rows.

mod appearance;
mod compile;
mod creation;
mod drive;
mod edit;
mod gearbox;
mod geometry;
mod graph;
mod id;
mod region;
mod shape;
mod solid;

pub use compile::{
    CYLINDER_COLLIDER_COUNT, ColliderShape, CompiledBearing, CompiledCompound, CompiledConvex,
    CompiledCreation, CoordinateDrive, DriveMode, GearSelection, LocalCollider, LoopTopology,
    MAX_COMPILED_COLLIDERS, MassProperties, MechanismBodyTopology, PIPE_BEND_COLLIDER_COUNT,
    TopologyError,
};
pub use creation::{
    BearingDoc, BearingSocket, BearingSocketDoc, CREATION_FORMAT_VERSION, CreationDocument,
    CreationError, DriveDwellDoc, DriveLimitsDoc, DriveLinkDoc, DriveProgramDoc, DriveStateDoc,
    DriveTriggerDoc, EdgeChainRefDoc, FaceOwnerDoc, FaceRefDoc, GearboxConfigDoc, InputSeatLinkDoc,
    LoadedCreation, PartDoc, PoseDoc, RegionDoc, RigidLinkDoc, SeatControllerLinkDoc,
    ShapeFeatureDoc, SolidOwnerDoc, TopologyKeyDoc, TopologySourceDoc, WeldDoc,
};
pub use drive::{
    ActuatorAssignment, ActuatorPercentageError, DriveDwell, DriveKey, DriveLimits,
    DriveLimitsError, DriveName, DriveProgram, DriveProgramError, DriveRelease, DriveState,
    DriveTarget, DriveTrigger, MAX_DRIVE_DWELL_SECONDS, MAX_DRIVE_LIMIT_RADIANS,
    MAX_DRIVE_NAME_BYTES, MAX_DRIVE_SPEED_RAD_S, MAX_DRIVE_STATES,
};
pub use edit::{
    CONSTRUCTION_PAGE_MAX_PARTS, CONSTRUCTION_PAGE_MAX_VERTICES, ConstructionEditDelta,
    ConstructionGeometryOwner, ConstructionPageKey, ConstructionRenderDelta,
    ConstructionRenderPage,
};
pub use gearbox::{
    GearKey, GearKeyChord, GearboxConfig, GearboxError, MAX_GEAR_RATIO, MAX_GEARS, MIN_GEAR_RATIO,
    ShiftMode,
};
pub use geometry::{
    Axis, BuildPose, CYLINDER_SWEEP_STEP_DEGREES, ConstructionMaterial, ControllerSpec, CuboidSpec,
    CylinderDimensionError, CylinderDimensions, CylinderSpec, DimensionError, DimensionLinkId,
    DimensionLinkSpec, EngineKind, EngineSpec, FaceKind, FaceOwner, FaceRef, GRID_UNIT_METERS,
    GridDimension, GridRotation, InputSpec, MAX_CYLINDER_OUTER_DIAMETER,
    MAX_CYLINDER_SWEEP_DEGREES, MAX_GRID_UNITS, MAX_PIPE_BEND_RADIUS, MIN_CYLINDER_DIAMETER_GAP,
    MIN_CYLINDER_OUTER_DIAMETER, MIN_CYLINDER_SWEEP_DEGREES, MIN_PIPE_BEND_RADIUS,
    MaterialProperties, PIPE_BEND_ARC_SLICES, PIPE_BEND_RADIAL_SIDES, POSITION_TICK_METERS,
    POSITION_TICKS_PER_GRID_UNIT, POSITION_TICKS_PER_HALF_GRID_UNIT, PartSpec,
    PipeBendDimensionError, PipeBendDimensions, PipeBendSpec, SeatSpec, ServoSpec,
    TransmissionSpec, snap_world_to_grid,
};
pub use graph::{
    ActuatorInventory, AppearanceTarget, BearingDimensionError, BearingDimensions, BearingSpec,
    BuildCommand, BuildOutcome, ConstructionGraph, ConstructionGraphEdit, DriveLinkSpec,
    GraphError, GraphPartition, InputSeatLinkSpec, MAX_BEARING_OUTER_DIAMETER,
    MIN_BEARING_DIAMETER_GAP, MIN_BEARING_OUTER_DIAMETER, PendingOperation, RigidLinkSpec,
    SeatControllerLinkSpec, StructuralComponent, WeldSpec,
};
pub use id::{
    BearingId, DriveLinkId, InputSeatLinkId, PartId, RegionId, RigidLinkId, SeatControllerLinkId,
    ShapeFeatureId, WeldId,
};
pub use region::{CageIndex, RegionError, ShapeRegion};
pub use shape::{
    CellGrid, ConvexFace, ConvexPiece, GridFace, MAX_PIECE_EDGES, MAX_PIECE_FACES,
    MAX_PIECE_VERTICES, PartPiece, STEP_METERS, STEPS_PER_CELL, STEPS_PER_HALF_UNIT, decompose,
    decompose_part, face_neighbour_offset, has_inverted_cell, part_cells, steps_to_meters,
    undisplaced_steps,
};
pub use solid::{
    BoundaryHalfEdge, BoundaryVertex, ConvexVolumeCell, EdgeChainRef, EdgeTreatment,
    EvaluatedSolid, LogicalEdge, ShapeFeature, SolidError, SolidOwner, SurfacePatch,
    SurfacePatchKey, TopologyKey, TopologySource, evaluate_part_solid, evaluate_region_solid,
};

/// Legacy authored-machine density, in kg/m³.
pub const CUBOID_DENSITY_KG_M3: f32 = 500.0;

/// Maximum acceptable derived bearing-anchor separation, in metres.
pub const ANCHOR_TOLERANCE_METERS: f32 = 0.000_01;

/// Maximum acceptable derived bearing-axis separation, in degrees.
pub const AXIS_TOLERANCE_DEGREES: f32 = 0.001;
pub use appearance::{
    AppearanceError, MaterialAppearance, MaterialColor, MaterialDye, MaterialFinish, MaterialShift,
};
