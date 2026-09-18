//! CPU physics for compiled machines.
//!
//! [`CpuMachine`] is the soft-step game solver the app runs: every valid tick
//! publishes. [`CpuJointMachine`] is the exact reference solver: implicit midpoint
//! ticks with event search and original-law contact solves. It publishes only
//! when the original contact laws hold, leaves a tick it cannot certify
//! unchanged, and serves as an offline checker. Both share the
//! reduced-coordinate machine model, joint force laws and collision geometry.
//! The soft-step solver runs closed mechanism loops; the exact reference does not.

mod clumps;
mod error;
mod free_motion;
mod joint_forces;
mod joint_machine;
mod kinematics;
mod machine;
mod motion_path;
mod response;
mod soft_step;
mod terrain_contacts;

pub use clumps::PreparedClumpBodies;
pub use error::PhysicsError;
pub use free_motion::{CpuFreeMotion, CpuSnapshot, ExternalImpulse, MachineState};
pub use joint_machine::{
    CpuJointMachine, DriveCommand, JointAttemptFailure, JointFailureStage, JointTickConfig,
    JointTickDiagnostics, TerrainIntegration, TerrainSubstep,
};
pub use kinematics::MachineKinematics;
pub use machine::{BodyPose, MachineDynamics, SpatialMotion};
pub use motion_path::{MachineMotion, MotionBound};
pub use response::{
    ConstraintBlock, ConstraintSolution, ContactFriction, DENSE_CONTACT_ROWS, DynamicsFactor,
    DynamicsFactorization, ImpulseBounds, PreparedConstraints, solve_constraints,
};
pub use soft_step::{
    CpuMachine, SoftStepConfig, SoftStepDiagnostics, SoftStepTerrain, TerrainLoad,
};
pub use terrain_contacts::{
    ContactObstacle, ContactTarget, MachineCollisionGeometry, TerrainContact,
    TerrainContactFeature, TerrainContactQuery, TerrainContactScene, TerrainImpactConstraints,
    TerrainPathFailure, TerrainPathOutcome, TerrainPathQuery, TerrainSweepHit, TerrainSweepOutcome,
    TerrainSweepQuery,
};
