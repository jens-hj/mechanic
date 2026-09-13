//! CPU experiments for compiled machine dynamics.
//!
//! Numerical factors, pose-dependent response, and transactional free-motion
//! reference ticks and implicit midpoint joint ticks with bounded drives, stops,
//! and suspension. This is not yet an authoritative world runtime: complete
//! collision integration, loops, and cross-backend acceptance remain open.

mod free_motion;
mod joint_forces;
mod joint_machine;
mod machine;
mod motion_path;
mod response;
mod terrain_contacts;

pub use free_motion::{CpuFreeMotion, CpuSnapshot, ExternalImpulse, MachineState, TICK_SECONDS};
pub use joint_machine::{
    CpuJointMachine, DriveCommand, JointAttemptFailure, JointFailureStage, JointTickDiagnostics,
    JointTickSettings, TerrainIntegration, TerrainSubstep,
};
pub use machine::{BodyPose, MachineDynamics, SpatialMotion};
pub use motion_path::{MachineMotion, MotionBound};
pub use response::{
    ConstraintBlock, ConstraintSolution, ContactFriction, DENSE_CONTACT_ROWS, DynamicsFactor,
    DynamicsFactorization, ImpulseBounds, PreparedConstraints, solve_constraints,
};
pub use terrain_contacts::{
    ContactObstacle, ContactTarget, MachineCollisionGeometry, TerrainContact,
    TerrainContactFeature, TerrainContactQuery, TerrainContactScene, TerrainImpactConstraints,
    TerrainPathFailure, TerrainPathOutcome, TerrainPathQuery, TerrainSweepHit, TerrainSweepOutcome,
    TerrainSweepQuery,
};

/// Explicit failures from the compiled dynamics experiment. Failed results must
/// never be published as a completed physics tick.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum PhysicsError {
    /// Invalid finite collision geometry, hierarchy, or generation publication.
    #[error("invalid collision geometry or terrain publication")]
    InvalidCollision,
    /// A bounded implicit/constraint solve failed its residual or travel gate.
    #[error("joint tick did not converge within the configured bounds")]
    NotConverged,
    /// Joint-only ticks cannot omit closed-loop equations.
    #[error("joint-only CPU runtime does not yet support closed loops")]
    UnsupportedJointLoops,
    /// A command targets another tick/topology or contains invalid drive/impulse data.
    #[error("invalid tick-indexed physics command")]
    InvalidCommand,
    /// The free-motion reference cannot discard authored constraints or forces.
    #[error("free-motion reference requires passive unbounded revolute trees without loops")]
    UnsupportedFreeMotion,
    /// Malformed or non-positive effective dynamics.
    #[error("effective dynamics are non-finite, asymmetric, or not positive definite")]
    InvalidDynamics,
    /// Invalid constraint dimensions, bounds, or solver settings.
    #[error("invalid constraint rows or solver settings")]
    InvalidConstraints,
    /// Rank-revealing elimination found incompatible dependent equations.
    #[error("dependent constraint rows have inconsistent targets")]
    InconsistentConstraints,
    /// The reference experiment deliberately bounds its dense generalized matrix.
    #[error("dense reference experiment supports at most 512 generalized velocities")]
    ReferenceCapacity,
}
