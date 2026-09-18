//! The failure type shared by every CPU solver.

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
