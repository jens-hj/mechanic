//! CPU physics for compiled machines.
//!
//! [`CpuMachine`] is the soft-step game solver the app runs: every valid tick
//! publishes. [`reference`] holds the exact solver, which publishes only when the
//! original contact laws hold and serves as an offline checker. Both share the
//! reduced-coordinate machine model, joint force laws and collision geometry.
//! The soft-step solver runs closed mechanism loops; the exact reference does not.

mod free_motion;
mod joint_forces;
mod joint_machine;
mod machine;
mod motion_path;
mod response;
mod soft_step;
mod terrain_contacts;

/// The exact reference solver: implicit midpoint ticks with event search and
/// original-law contact solves. A tick it cannot certify is rejected unchanged.
pub mod reference {
    pub use crate::joint_machine::{
        CpuJointMachine, JointAttemptFailure, JointFailureStage, JointTickDiagnostics,
        JointTickSettings, TerrainIntegration, TerrainSubstep,
    };
}

pub use free_motion::{CpuFreeMotion, CpuSnapshot, ExternalImpulse, MachineState, TICK_SECONDS};
pub use joint_machine::DriveCommand;
pub use machine::{BodyPose, MachineDynamics, SpatialMotion};
pub use motion_path::{MachineMotion, MotionBound};
pub use response::{
    ConstraintBlock, ConstraintSolution, ContactFriction, DENSE_CONTACT_ROWS, DynamicsFactor,
    DynamicsFactorization, ImpulseBounds, PreparedConstraints, solve_constraints,
};
pub use soft_step::{CpuMachine, SoftStepDiagnostics, SoftStepSettings, SoftStepTerrain};
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
