//! GPU physics ABI, fixed-capacity runtime state, and custom compute dispatch.

mod abi;
mod device;
mod limits;
mod shaders;

pub use abi::{
    COLLIDER_SHAPE_CONVEX, COLLIDER_SHAPE_CUBOID, CONSTRAINT_NON_CONVERGENCE_FLAG,
    DRIVE_MODE_ANGLE, DRIVE_MODE_PASSIVE, DRIVE_MODE_SPEED, GpuBearing, GpuCollider, GpuContact,
    GpuContractionNode, GpuDiagnostics, GpuGroundSurface, GpuLinkState, GpuMass, GpuMechanismBody,
    GpuMechanismCoordinate, GpuMechanismDrive, GpuPair, GpuPersistentManifold, GpuSpatialInertia,
    GpuTickConfig, GpuTransform, GpuVelocity, INVALID_NUMERIC_FLAG, MANIFOLD_OVERFLOW_FLAG,
    PAIR_OVERFLOW_FLAG, pack_convex_counts,
};
pub use device::{
    EXTERNAL_IMPULSE_BATCH_CAPACITY, GpuBodyStateError, GpuCompletedTickReadback, GpuDispatchError,
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
