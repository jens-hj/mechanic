//! Editable construction data and deterministic compilation into physics rows.

mod appearance;
mod compile;
mod contact_geometry;
mod creation;
mod drive;
mod dynamics;
mod edit;
mod frame;
mod gearbox;
mod geometry;
mod graph;
mod hardware_mesh;
mod id;
mod joint;
mod linear;
mod linear_geometry;
mod obb;
mod pipe_junction;
mod piston;
mod piston_geometry;
mod region;
mod runtime_bodies;
mod shape;
mod solid;
mod suspension;
mod suspension_geometry;
mod units;
mod weld;

pub use appearance::{
    AppearanceError, MaterialAppearance, MaterialColor, MaterialDye, MaterialFinish, MaterialShift,
};
pub use compile::{
    CYLINDER_COLLIDER_COUNT, ColliderShape, CompiledBearing, CompiledCompound, CompiledConvex,
    CompiledCreation, CompiledCylinder, CoordinateDrive, DriveMode, GearSelection, LocalCollider,
    LoopTopology, MAX_COMPILED_COLLIDERS, MassProperties, MechanismBodyTopology,
    PIPE_BEND_COLLIDER_COUNT, TopologyError, cylinder_collider_count,
};
pub use contact_geometry::{
    ContactCylinder, ContactGeometryError, ContactPolytope, ContactVelocity, ConvexFeature,
    ConvexSeparation, CylinderAnchor, RigidContactSweep, SweepOutcome, TriangleClipScratch,
    TriangleContactPoint, TriangleSupport,
};
pub use creation::{
    BearingDoc, BearingSocket, BearingSocketDoc, CREATION_FORMAT_VERSION, ConstructionFrameDoc,
    CreationDocument, CreationError, DriveDwellDoc, DriveLimitsDoc, DriveLinkDoc, DriveProgramDoc,
    DriveStateDoc, DriveTriggerDoc, EdgeChainRefDoc, FaceOwnerDoc, FaceRefDoc, GearboxConfigDoc,
    InputSeatLinkDoc, LoadedCreation, MaterialLayerDoc, PartDoc, PoseDoc, RegionDoc, RigidLinkDoc,
    SeatControllerLinkDoc, ShapeFeatureDoc, SolidOwnerDoc, SpiralDoc, SpiralTaperDoc,
    TopologyKeyDoc, TopologySourceDoc, WeldDoc,
};
pub use drive::{
    ActuatorAssignment, ActuatorPercentageError, DriveDwell, DriveKey, DriveLimits,
    DriveLimitsError, DriveName, DriveProgram, DriveProgramError, DriveRelease, DriveState,
    DriveTarget, DriveTrigger, LinearDriveLimits, MAX_DRIVE_DWELL_SECONDS, MAX_DRIVE_LIMIT_RADIANS,
    MAX_DRIVE_NAME_BYTES, MAX_DRIVE_SPEED_RAD_S, MAX_DRIVE_STATES, MAX_LINEAR_TRAVEL_METERS,
};
pub use dynamics::{CompiledDynamics, DynamicsComponent, LoopConstraintPattern, SpatialInertia};
pub use edit::ConstructionEditDelta;
pub use frame::{ConstructionFrame, ConstructionFrameId, FrameError};
pub use gearbox::{
    GearKey, GearKeyChord, GearboxConfig, GearboxError, MAX_GEAR_RATIO, MAX_GEARS, MIN_GEAR_RATIO,
    ShiftMode,
};
pub use geometry::{
    Axis, BuildPose, CYLINDER_SWEEP_STEP_DEGREES, ConstructionMaterial, ControllerSpec, CuboidSpec,
    CylinderDimensionError, CylinderDimensions, CylinderSpec, DimensionError, DimensionLinkId,
    DimensionLinkSpec, EngineKind, EngineSpec, FaceKind, FaceOwner, FaceRef, GRID_UNIT_METERS,
    GridDimension, GridRotation, InputSpec, LayerError, LayerFace, LayerRegion,
    MAX_CYLINDER_OUTER_DIAMETER, MAX_CYLINDER_SWEEP_DEGREES, MAX_GRID_UNITS, MAX_PART_LAYERS,
    MAX_SPIRAL_PITCH_TICKS, MAX_SPIRAL_PROFILE_POINTS, MAX_SPIRAL_RIDGE_COLLIDERS,
    MAX_SPIRAL_STARTS, MIN_CYLINDER_DIAMETER_GAP, MIN_CYLINDER_OUTER_DIAMETER,
    MIN_CYLINDER_SWEEP_DEGREES, MIN_LAYER_THICKNESS_METERS, MIN_SPIRAL_COLLIDER_STEPS_PER_TURN,
    MIN_SPIRAL_PITCH_TICKS, MIN_SPIRAL_TIP_DIAMETER_TICKS, MaterialLayer, MaterialLayers,
    MaterialProperties, PIPE_BEND_ARC_SLICES, PIPE_BEND_RADIAL_SIDES, POSITION_TICK_METERS,
    POSITION_TICKS_PER_GRID_UNIT, POSITION_TICKS_PER_HALF_GRID_UNIT, PartSpec, PipeArms,
    PipeBendDimensionError, PipeBendDimensions, PipeBendSpec, PipeJunctionDimensions,
    PipeJunctionError, PipeJunctionSpec, SPIRAL_COLLIDER_STEPS_PER_TURN, SPIRAL_PROFILE_STEP_TICKS,
    SeatSpec, ServoSpec, SpiralEnd, SpiralError, SpiralHand, SpiralPoint, SpiralProfile,
    SpiralSpec, SpiralTaper, SurfaceResponse, TransmissionSpec, snap_world_to_grid,
};
pub use graph::{
    ActuatorInventory, AppearanceTarget, BearingDimensionError, BearingDimensions, BearingSpec,
    BuildCommand, BuildOutcome, ConstructionGraph, ConstructionGraphEdit, DriveLinkSpec,
    GraphError, GraphPartition, InputSeatLinkSpec, MAX_BEARING_OUTER_DIAMETER,
    MIN_BEARING_DIAMETER_GAP, MIN_BEARING_OUTER_DIAMETER, PendingOperation, RigidLinkSpec,
    SeatControllerLinkSpec, StructuralComponent, WeldSpec,
};
pub use hardware_mesh::HardwareFinish;
pub use id::{
    BearingId, DriveLinkId, InputSeatLinkId, PartId, RegionId, RigidLinkId, SeatControllerLinkId,
    ShapeFeatureId, WeldId,
};
pub use joint::{JointKind, JointMassElement};
pub use linear::{
    CarriageFace, LINEAR_METERS_PER_RADIAN, LINEAR_METERS_PER_REVOLUTION, LinearBearing,
    LinearBearingDimensions, LinearBearingError,
};
pub use linear_geometry::{
    LINEAR_FINISHES, LinearFinish, LinearMeshChunk, LinearMeshOwner, linear_bearing_meshes,
};
pub use obb::{Obb, ObbContact, obb_sat};
pub use pipe_junction::{
    PipeJunctionBox, PipeJunctionSurface, PipeJunctionTriangle, pipe_junction_triangles,
    pipe_junction_wall_boxes,
};
pub use piston::{Piston, PistonDimensions, PistonError, PistonMount};
pub use piston_geometry::{PISTON_FINISHES, PistonMeshChunk, PistonMeshOwner, piston_meshes};
pub use region::{CageIndex, RegionError, ShapeRegion};
pub use runtime_bodies::RuntimeBox;
pub use shape::{
    CellGrid, ConvexFace, ConvexPiece, GridFace, MAX_PIECE_EDGES, MAX_PIECE_FACES,
    MAX_PIECE_VERTICES, PartPiece, decompose, decompose_part, face_neighbour_offset,
    has_inverted_cell, part_cells, steps_to_meters, undisplaced_steps,
};
pub use solid::{
    BoundaryHalfEdge, BoundaryVertex, ConvexVolumeCell, EdgeChainRef, EdgeTreatment,
    EvaluatedSolid, LogicalEdge, ShapeFeature, SolidError, SolidOwner, SpiralCore, SurfacePatch,
    SurfacePatchKey, TopologyKey, TopologySource, evaluate_part_solid, evaluate_region_solid,
    spiral_core, spiral_pieces,
};
pub use suspension::{
    BumpStopSpec, CompressionLimit, MountPlates, ShockBodyEnd, ShockGeometry, ShockSpec,
    SpringSpec, SuspensionError, SuspensionSpec,
};
pub use suspension_geometry::{
    SUSPENSION_FINISHES, SuspensionMeshChunk, SuspensionMeshOwner, suspension_meshes,
};
pub use units::{
    ANCHOR_TOLERANCE_METERS, AXIS_TOLERANCE_DEGREES, GRAVITY, MACHINE_PART_DENSITY_KG_M3,
    STANDARD_GRAVITY_M_S2, STANDARD_GRAVITY_M_S2_F32, TICK_RATE_HZ, TICK_SECONDS, TICK_SECONDS_F32,
    rad_s_to_rpm, rpm_to_rad_s,
};
pub use weld::{
    WeldAlignment, WeldCollider, WeldConstraint, WeldError, WeldFeature, WeldFeatureRef,
    WeldMaterialPatch, WeldPick, WeldPlacement, WeldSelection, WeldSnap, weld_contact_square,
};
