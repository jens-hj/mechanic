//! GPU physics ABI, fixed-capacity runtime state, and custom compute dispatch.

mod abi;
mod collision;
mod device;
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
    GpuBodyStateError, GpuCompletedTickReadback, GpuImpulseError, GpuKernelTimings, GpuPhysics,
    GpuPhysicsConfig, GpuPhysicsError, GpuReadbackError, GpuTickReadback, GpuTickSubmission,
    SnapshotBuffers,
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

/// Fixed 60 Hz physics frequency.
pub const PHYSICS_TPS: u32 = 60;

/// Fixed physics step duration in seconds.
pub const FIXED_DT_SECONDS: f32 = 1.0 / 60.0;

/// Maximum number of compound bodies accepted by the milestone runtime.
pub const MAX_BODIES: usize = 131_072;

/// Maximum number of passive bearings accepted by the milestone runtime.
pub const MAX_BEARINGS: usize = 262_144;

/// Maximum uploaded collider rows.
pub const MAX_COLLIDERS: usize = 131_072;

/// Maximum `vec4` slots in the packed convex-shape buffer.
///
/// One shaped piece needs at most eight vertices, twelve face planes, and
/// eighteen edge directions, so this holds a large shaped creation while
/// staying a fixed allocation like every other buffer here.
pub const MAX_CONVEX_SHAPE_SLOTS: usize = 1_048_576;

/// Fixed candidate/contact capacity. Overflow blocks publication.
pub const MAX_CONTACT_PAIRS: usize = 2_097_152;

/// Power-of-two spatial broadphase table capacity.
pub const BROADPHASE_HASH_CAPACITY: usize = 262_144;

/// Number of published snapshots retained entirely on the GPU.
pub const SNAPSHOT_RING_SIZE: usize = 3;
