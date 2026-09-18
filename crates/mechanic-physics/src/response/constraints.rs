//! Constraint rows, their bounds and friction, and the solved impulses.

/// Largest scalar constraint count for explicit contact-response storage.
pub const DENSE_CONTACT_ROWS: usize = 128;

/// Bounds on one scalar row. Bilateral rows use infinite limits.
#[derive(Clone, Copy, Debug)]
pub struct ImpulseBounds {
    /// Minimum allowed impulse.
    pub minimum: f64,
    /// Maximum allowed impulse.
    pub maximum: f64,
}

/// Coulomb/rolling law for one point in a coupled contact manifold.
#[derive(Clone, Copy, Debug)]
pub struct ContactFriction {
    /// Coefficient available when a contact can remain at rest.
    pub static_coefficient: f64,
    /// Coefficient while sliding, no greater than the static coefficient.
    pub kinetic_coefficient: f64,
    /// Whether the contact is already sliding at the beginning of this solve.
    pub sliding: bool,
    /// Optional rolling-resistance coefficient times effective radius, in metres.
    /// Adds two angular rows after the normal and two tangential rows.
    pub rolling_length: Option<f64>,
}

impl ContactFriction {
    pub(super) fn rows(self) -> usize {
        if self.rolling_length.is_some() { 5 } else { 3 }
    }
}

/// One coupled contact manifold, drive, stop, or loop block.
#[derive(Clone, Debug)]
pub struct ConstraintBlock {
    /// Generalized Jacobian rows. All rows must match the dynamics dimension.
    pub jacobian: Vec<Vec<f64>>,
    /// Required change in row velocity before constraint impulses.
    pub target: Vec<f64>,
    /// Per-row impulse bounds.
    pub bounds: Vec<ImpulseBounds>,
    /// Ordered contact points, each with normal/tangent/tangent and optionally
    /// two rolling rows. Empty for drives, stops, and bilateral loop blocks.
    /// Normal lower bounds are zero; tangent/rolling bounds are infinite.
    pub contacts: Vec<ContactFriction>,
}

/// Result of a deterministic block projected solve.
#[derive(Clone, Debug)]
pub struct ConstraintSolution {
    /// Safeguarded Newton proposals, separate from block sweeps.
    pub newton_attempts: usize,
    /// Newton proposals accepted within the bounded physical-residual history.
    pub newton_accepts: usize,
    /// Newton/search operator applications, including cached dynamics responses.
    pub newton_applications: usize,
    /// Attempted 1/3/5-row factors (analytic projection blocks or pivoted LU).
    pub newton_local_factorizations: usize,
    /// Bounded reduced or mixed Newton matrix factorizations.
    pub newton_generalized_factorizations: usize,
    /// Peak scalar matrix storage for reduced searches, including rectangular
    /// factors when an implicit mixed system exceeds 128 equations.
    pub newton_reduced_storage: usize,
    /// Bounded dense contact-space Newton factorizations.
    pub newton_contact_factorizations: usize,
    /// Peak scalar storage of the dense contact Newton matrix.
    pub newton_contact_storage: usize,
    /// Candidate residual evaluations during bounded Newton backtracking.
    pub newton_line_searches: usize,
    /// Bounded inactive-contact hypotheses, validated against every original row.
    pub active_set_trials: usize,
    /// Hypotheses discarded because original constraints or subset solving failed.
    pub active_set_trial_rejections: usize,
    /// Extra scalar matrix storage for a hypothesis: W, local blocks and Jacobian.
    /// Excludes vectors/metadata; numerical H is borrowed, never copied.
    pub active_set_trial_matrix_storage: usize,
    /// Stalled supplied warm iterates discarded within the original sweep budget.
    pub warm_start_restarts: usize,
    /// Bounded smoothing searches; candidates must pass the original contact laws.
    pub continuation_trials: usize,
    /// Search Newton directions charged to the same total iteration budget.
    pub continuation_iterations: usize,
    /// Smoothing stages evaluated during candidate search.
    pub continuation_stages: usize,
    /// Smoothed residual evaluations, including bounded backtracking.
    pub continuation_evaluations: usize,
    /// Peak mixed search dimension, bounded to 128 including retained contact rows.
    pub continuation_mixed_rows: usize,
    /// Peak full-space search dimension. Small machines deliberately store this
    /// search matrix beyond the 128-row response bound, up to 256 contact rows.
    pub continuation_full_rows: usize,
    /// Final sliding state in block/contact-point order.
    pub sliding: Vec<bool>,
    /// Static contacts whose required impulse exceeded the static cone.
    pub friction_transitions: usize,
    /// Generalized velocity change; apply only after checking convergence.
    pub velocity_change: Vec<f64>,
    /// Scalar impulses in block/row order.
    pub impulses: Vec<f64>,
    /// Maximum projected velocity residual.
    pub residual: f64,
    /// Total completed block sweeps and continuation Newton directions.
    pub iterations: usize,
    /// Whether the residual reached the requested threshold.
    pub converged: bool,
    /// Scalar slots allocated for the explicit response matrix (zero above 128 rows).
    pub response_storage: usize,
    /// Inverse-dynamics applications, including response preparation and final motion.
    pub factor_solves: usize,
}

#[derive(Clone)]
pub(super) struct BlockLayout {
    pub(super) first: usize,
    pub(super) diagonal: Vec<f64>,
    pub(super) scale: f64,
}
