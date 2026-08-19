//! GPU physics ABI, fixed-capacity runtime state, and custom compute dispatch.

mod abi;
mod collision;
mod device;
mod runtime;
mod scheduler;

pub use abi::{
    CONSTRAINT_NON_CONVERGENCE_FLAG, GpuBearing, GpuCollider, GpuContact, GpuContractionNode,
    GpuDiagnostics, GpuLinkState, GpuMass, GpuMechanismBody, GpuMechanismCoordinate, GpuPair,
    GpuPersistentManifold, GpuSpatialInertia, GpuTickConfig, GpuTransform, GpuVelocity,
    INVALID_NUMERIC_FLAG, MANIFOLD_OVERFLOW_FLAG, PAIR_OVERFLOW_FLAG,
};
pub use collision::{
    ContactManifold, ContactPoint, Obb, SatContact, obb_contact_manifold, obb_sat,
};
pub use device::{
    GpuImpulseError, GpuKernelTimings, GpuPhysics, GpuPhysicsConfig, GpuPhysicsError,
    GpuReadbackError, GpuTickReadback, GpuTickSubmission, SnapshotBuffers,
};
pub use runtime::{
    CapacityKind, FailureStatus, PhysicsRuntime, PublishedGpuState, SimulationStatus,
    TickStatistics,
};
pub use scheduler::{FixedStepScheduler, ScheduledTicks};

/// Fixed 60 Hz physics frequency.
pub const PHYSICS_TPS: u32 = 60;

/// Fixed physics step duration in seconds.
pub const FIXED_DT_SECONDS: f32 = 1.0 / 60.0;

/// Maximum number of compound bodies accepted by the milestone runtime.
pub const MAX_BODIES: usize = 131_072;

/// Maximum number of passive bearings accepted by the milestone runtime.
pub const MAX_BEARINGS: usize = 262_144;

/// Maximum uploaded cuboid collider rows.
pub const MAX_COLLIDERS: usize = 131_072;

/// Fixed candidate/contact capacity. Overflow blocks publication.
pub const MAX_CONTACT_PAIRS: usize = 2_097_152;

/// Power-of-two spatial broadphase table capacity.
pub const BROADPHASE_HASH_CAPACITY: usize = 262_144;

/// Number of published snapshots retained entirely on the GPU.
pub const SNAPSHOT_RING_SIZE: usize = 3;
