//! GPU physics ABI, fixed-capacity runtime state, and custom compute dispatch.

mod abi;
mod collision;
mod device;
mod limits;
mod render;
mod runtime;
mod scheduler;
mod terrain;

pub use abi::{
    COLLIDER_SHAPE_CONVEX, COLLIDER_SHAPE_CUBOID, CONSTRAINT_NON_CONVERGENCE_FLAG,
    DRIVE_MODE_ANGLE, DRIVE_MODE_PASSIVE, DRIVE_MODE_SPEED, GpuBearing, GpuCollider, GpuContact,
    GpuContractionNode, GpuDiagnostics, GpuGroundSurface, GpuLinkState, GpuMass, GpuMechanismBody,
    GpuMechanismCoordinate, GpuMechanismDrive, GpuPair, GpuPersistentManifold, GpuSpatialInertia,
    GpuTickConfig, GpuTransform, GpuVelocity, INVALID_NUMERIC_FLAG, MANIFOLD_OVERFLOW_FLAG,
    PAIR_OVERFLOW_FLAG, pack_convex_counts,
};
pub use collision::{
    ContactManifold, ContactPoint, Obb, SatContact, obb_contact_manifold, obb_sat,
};
pub use device::{
    EXTERNAL_IMPULSE_BATCH_CAPACITY, GpuBodyStateError, GpuCompletedTickReadback,
    GpuExecutionEvidence, GpuExternalImpulse, GpuGroundPlane, GpuGroundPlaneError, GpuHoldError,
    GpuImpulseError, GpuKernelTimings, GpuPhysics, GpuPhysicsConfig, GpuPhysicsError,
    GpuPhysicsPipelines, GpuReadbackError, GpuSolverRoute, GpuSubmissionTimings, GpuTerrainError,
    GpuTickReadback, GpuTickSubmission, PreparedTerrainUpdate, SnapshotBuffers,
    TerrainPreparationCache, TerrainPreparationRequest, TerrainResidency, TerrainUploadStats,
};
pub use limits::{
    BROADPHASE_HASH_CAPACITY, MAX_BEARINGS, MAX_BODIES, MAX_COLLIDERS, MAX_CONTACT_PAIRS,
    MAX_CONVEX_SHAPE_SLOTS, SNAPSHOT_RING_SIZE,
};
pub use render::{
    TerrainRenderAcknowledgement, TerrainRenderArena, TerrainRenderArenaLimits, TerrainRenderChunk,
    TerrainRenderDelta, TerrainRenderDirtyRanges, TerrainRenderError,
};
pub use runtime::{
    CapacityKind, FailureStatus, PhysicsRuntime, PublishedGpuState, SimulationStatus,
    TickStatistics,
};
pub use scheduler::{FixedStepScheduler, ScheduledTicks};
pub use terrain::{
    TerrainBufferLimits, TerrainContact, TerrainContactShape, TerrainPhysicsScene,
    TerrainStageMetrics, terrain_contacts,
};
