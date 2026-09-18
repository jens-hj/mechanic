//! Coupled, implicit joint-only ticks; collision and loop integration remain open.

mod contact_kinematics;
mod events;
mod impact;
mod recovery;
mod stops;

use bevy_math::DVec3;
use mechanic_core::{CompiledCreation, CoordinateDrive, DriveMode, TICK_SECONDS};

use crate::{
    ConstraintBlock, CpuSnapshot, ExternalImpulse, ImpulseBounds, MachineDynamics, MachineState,
    PhysicsError, PreparedConstraints,
    free_motion::{advance_positions, apply_external_impulses},
    joint_forces::{PassiveForce, drive_budget, drive_target, validate_drive},
};

/// Tick-indexed replacement of one resolved actuator row (including gearing).
#[derive(Clone, Copy, Debug)]
pub struct DriveCommand {
    /// External tick that consumes the command.
    pub tick: u64,
    /// Generation identifying the coordinate rows.
    pub topology_generation: u64,
    /// Compiled joint coordinate.
    pub coordinate: usize,
    /// Resolved drive target, effort envelope, and authored coordinate limits.
    pub drive: CoordinateDrive,
}

/// Fixed quality bounds. Retry uses successively finer subdivisions, never frame time.
#[derive(Clone, Copy, Debug)]
pub struct JointTickSettings {
    /// Numerical factor for complete CPU experiments; quality bounds are unchanged.
    pub factorization: crate::DynamicsFactorization,
    /// Initial subdivisions of the external 60 Hz tick: 1, 2, 4, or 8.
    pub substeps: u32,
    /// Maximum subdivisions permitted during unpublished retries.
    pub maximum_substeps: u32,
    /// Bound on nonlinear midpoint inertia/force updates per substep.
    pub force_iterations: usize,
    /// Bound on coupled projected constraint sweeps per force update.
    pub constraint_iterations: usize,
    /// Generalized velocity residual tolerance in SI coordinate units per second.
    pub tolerance: f64,
}

impl Default for JointTickSettings {
    fn default() -> Self {
        Self {
            factorization: crate::DynamicsFactorization::DenseReference,
            substeps: 1,
            maximum_substeps: 8,
            force_iterations: 128,
            constraint_iterations: 256,
            tolerance: 1e-8,
        }
    }
}

/// Operation that rejected a tick or one of its subdivision attempts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointFailureStage {
    /// Tick, generation, settings, and authored command validation.
    InputValidation,
    /// Applying the external impulses before subdivision retries.
    ExternalImpulse,
    /// Coupled finite-duration force, constraint, or travel integration.
    FiniteStep,
    /// Gathering finite terrain contact geometry.
    TerrainQuery,
    /// Continuous collision or penetration certification.
    TerrainPath,
    /// Split terrain position recovery.
    TerrainRecovery,
    /// Instantaneous coupled impact and active stops.
    Impact,
    /// Event localization or progress exhausted its fixed bound.
    EventSearch,
}

/// One rejected whole-tick attempt; at most four subdivision policies are tried.
#[derive(Clone, Debug, PartialEq)]
pub struct JointAttemptFailure {
    /// Subdivisions used by this rejected attempt.
    pub substeps: u32,
    /// Operation that rejected it, separate from cumulative work/residual maxima.
    pub stage: JointFailureStage,
    /// Error returned by that operation.
    pub error: PhysicsError,
    /// Validated time in the unpublished candidate; never published as a partial tick.
    pub accepted_seconds: f64,
}

/// Work and convergence for the most recent attempted external joint tick.
#[derive(Clone, Debug, Default)]
pub struct JointTickDiagnostics {
    /// Numerical factor used for the attempted tick (dense assembly remains).
    pub factorization: crate::DynamicsFactorization,
    /// Whether the candidate passed and was published.
    pub published: bool,
    /// Number of attempted subdivision policies, including the successful one.
    pub attempts: u32,
    /// Last attempted (or accepted) subdivision count.
    pub substeps: u32,
    /// Final rejection stage; absent after successful publication.
    pub failure_stage: Option<JointFailureStage>,
    /// Rejections in subdivision order, retained even if a finer attempt succeeds.
    pub attempt_failures: Vec<JointAttemptFailure>,
    /// Numerical factorizations, including external impulse preparation and retries.
    pub factorizations: usize,
    /// Pose-dependent matrix/Jacobian assemblies, including residual evaluation.
    pub dynamics_assemblies: usize,
    /// Reintegrations reusing the unchanged interval start's assembled dynamics.
    pub initial_dynamics_cache_hits: usize,
    /// Response preparation applications across all attempts.
    pub response_preparation_solves: usize,
    /// Factor solves during constraint iteration/reconstruction across all attempts.
    pub constraint_factor_solves: usize,
    /// RHS and physical-force residual factor applications, including retries.
    pub force_factor_solves: usize,
    /// Factor applications for split travel-limit and terrain position correction.
    pub position_factor_solves: usize,
    /// Number of nonlinear force updates, including rejected attempts.
    pub force_iterations: usize,
    /// Total constraint sweeps, including rejected attempts.
    pub constraint_iterations: usize,
    /// Scalar constraint rows prepared across all substeps and attempts.
    pub constraint_rows_prepared: usize,
    /// Ordered point-acceleration bias traversals for finite contact forces.
    pub contact_bias_traversals: usize,
    /// Pose reconstructions for material-point endpoint velocity conditions.
    pub contact_kinematic_poses: usize,
    /// Linear tree velocity passes for contact support and release.
    pub contact_velocity_traversals: usize,
    /// Maximum nonlinear contact-target mismatch across rejected and accepted updates.
    pub contact_kinematic_residual: f64,
    /// Finite surface points supplied to numerical substeps across all attempts.
    pub surface_points: usize,
    /// Coupled finite surface manifolds prepared across all attempts.
    pub surface_manifolds: usize,
    /// Discrete finite terrain queries across all attempts.
    pub terrain_contact_queries: usize,
    /// Reintegrations reusing contact geometry at the unchanged interval start.
    pub terrain_contact_cache_hits: usize,
    /// Discrete contact queries that returned an error instead of a manifold.
    pub terrain_contact_query_failures: usize,
    /// Physical and correction path validations across all attempts.
    pub terrain_path_queries: usize,
    /// Split position paths checked separately from physical motion.
    pub terrain_correction_paths: usize,
    /// Finite geometry queries for split penetration recovery.
    pub terrain_recovery_queries: usize,
    /// Accepted split correction updates, with no physical elapsed time.
    pub terrain_recovery_passes: usize,
    /// Scalar recovery/stop rows prepared for split correction.
    pub terrain_recovery_rows: usize,
    /// Pose reconstructions after split terrain correction.
    pub terrain_recovery_poses: usize,
    /// Whole-tree pose reconstructions during terrain path validation.
    pub terrain_path_poses: usize,
    /// Exact midpoint pose reuses within immutable penetration queries.
    pub terrain_pose_cache_hits: usize,
    /// Interval envelopes evaluated across accepted and rejected paths.
    pub terrain_envelopes: usize,
    /// Certified intervals across all candidate paths.
    pub terrain_certified_intervals: usize,
    /// Terrain chunk candidates across contact and path queries.
    pub terrain_chunk_candidates: usize,
    /// Finite triangle candidates across contact and path queries.
    pub terrain_triangle_candidates: usize,
    /// Candidate paths refused for depth, new impacts, or inconclusive work.
    pub terrain_path_rejections: usize,
    /// Continuous queries for newly approached pairs on candidate paths.
    pub terrain_sweep_queries: usize,
    /// Continuous queries that failed before returning work/outcome data.
    pub terrain_sweep_query_failures: usize,
    /// Penetration queries that failed before returning work/outcome data.
    pub terrain_envelope_query_failures: usize,
    /// Exact SAT evaluations during those queries.
    pub terrain_separation_evaluations: usize,
    /// Initial finite contact tests identifying support candidates.
    pub terrain_initial_contact_evaluations: usize,
    /// Exact linear SAT interval queries, without iterative advancement.
    pub terrain_linear_sweep_evaluations: usize,
    /// Per-vertex quadratic clearance certificates in terrain CCD.
    pub terrain_quadratic_sweep_evaluations: usize,
    /// Whole-tree spatial derivative traversals in terrain CCD.
    pub terrain_velocity_evaluations: usize,
    /// Slow collider CCD deferrals; each still requires the full depth certificate.
    pub terrain_slow_contact_deferrals: usize,
    /// Motion reconstructions establishing the endpoint-speed deferral bound.
    pub terrain_policy_poses: usize,
    /// Event/progress bounds exhausted while resolving new impacts.
    pub terrain_impact_holds: usize,
    /// Finite-duration event trials, including rejected intervals.
    pub event_trials: usize,
    /// Previously validated prefixes committed after a non-nested reintegration forecast.
    pub event_prefix_commits: usize,
    /// Intervals reintegrated from their unchanged start after CCD found a hit.
    pub event_refinements: usize,
    /// Clear reintegrated trials discarded while locating an impending impact.
    pub event_localizations: usize,
    /// Reintegration trials locating a released point's normal-velocity reversal.
    pub release_refinements: usize,
    /// Clear reintegrated trials extended to locate an actual velocity reversal.
    pub release_localizations: usize,
    /// Trials reintegrated to the first crossed joint travel bound.
    pub joint_stop_refinements: usize,
    /// Last coordinate whose travel bound requested a shorter trial.
    pub last_joint_stop_coordinate: Option<usize>,
    /// Initially separating points excluded from finite support force rows.
    pub released_surface_points: usize,
    /// Accepted physical intervals within the last whole-tick attempt.
    pub accepted_intervals: usize,
    /// Accepted elapsed time within the last whole-tick attempt, in seconds.
    pub accepted_seconds: f64,
    /// Instantaneous coupled impact attempts, including failed solves.
    pub impact_attempts: usize,
    /// Converged instantaneous velocity updates within unpublished attempts.
    pub impact_events: usize,
    /// Contact and active-stop rows prepared for instantaneous impact.
    pub impact_rows_prepared: usize,
    /// Inverse-mass applications during instantaneous impact solving.
    pub impact_factor_solves: usize,
    /// Projected solver sweeps during instantaneous impact attempts.
    pub impact_iterations: usize,
    /// Newton proposals during velocity and position constraints.
    pub newton_attempts: usize,
    /// Accepted Newton proposals across all attempts.
    pub newton_accepts: usize,
    /// Krylov response applications across all attempts.
    pub newton_applications: usize,
    /// Small numerical preconditioner factorizations across all attempts.
    pub newton_local_factorizations: usize,
    /// Generalized Newton matrix factorizations across all attempts.
    pub newton_generalized_factorizations: usize,
    /// Peak scalar storage of reduced Newton matrices.
    pub newton_reduced_storage: usize,
    /// Bounded dense contact-space Newton factorizations.
    pub newton_contact_factorizations: usize,
    /// Peak scalar storage of a dense contact Newton matrix.
    pub newton_contact_storage: usize,
    /// Newton candidate residual evaluations across all attempts.
    pub newton_line_searches: usize,
    /// Inactive-contact hypotheses within the original constraint iteration budgets.
    pub constraint_active_set_trials: usize,
    /// Rejected inactive-contact hypotheses, with original iterates retained.
    pub constraint_active_set_trial_rejections: usize,
    /// Stalled warm impulse guesses restarted within their original solve budgets.
    pub constraint_warm_start_restarts: usize,
    /// Peak extra scalar matrices retained by an inactive-contact hypothesis.
    pub constraint_active_set_trial_matrix_storage: usize,
    /// Maximum final residual within the last attempt, in generalized velocity units.
    pub residual: f64,
    /// Maximum implicit-force equation error after applying inverse dynamics.
    pub force_velocity_residual: f64,
    /// Maximum finite-duration projected constraint velocity residual.
    pub constraint_velocity_residual: f64,
    /// Maximum instantaneous-impact projected velocity residual (tighter solve).
    pub impact_velocity_residual: f64,
    /// Ordered active-set pivots in split position recovery.
    pub recovery_active_set_pivots: usize,
    /// Active normal-basis factors, separate from physical dynamics factors.
    pub recovery_active_set_factorizations: usize,
    /// Peak scalar normal-basis matrix storage during recovery.
    pub recovery_active_set_storage: usize,
    /// Bounded smoothing contact searches, all checked against original laws.
    pub continuation_trials: usize,
    /// Smoothing Newton directions included in constraint iteration counts.
    pub continuation_iterations: usize,
    /// Smoothing stages attempted across all numerical solves.
    pub continuation_stages: usize,
    /// Smoothed residual evaluations, including backtracking.
    pub continuation_evaluations: usize,
    /// Peak mixed contact search dimension (at most 128).
    pub continuation_mixed_rows: usize,
    /// Signed drive impulses summed over the last attempt, in N·s or N·m·s.
    pub drive_impulses: Vec<f64>,
}

impl JointTickDiagnostics {
    fn record_newton(&mut self, solution: &crate::ConstraintSolution) {
        self.continuation_trials += solution.continuation_trials;
        self.continuation_iterations += solution.continuation_iterations;
        self.continuation_stages += solution.continuation_stages;
        self.continuation_evaluations += solution.continuation_evaluations;
        self.continuation_mixed_rows = self
            .continuation_mixed_rows
            .max(solution.continuation_mixed_rows);
        self.newton_attempts += solution.newton_attempts;
        self.newton_accepts += solution.newton_accepts;
        self.newton_applications += solution.newton_applications;
        self.newton_local_factorizations += solution.newton_local_factorizations;
        self.newton_generalized_factorizations += solution.newton_generalized_factorizations;
        self.newton_contact_factorizations += solution.newton_contact_factorizations;
        self.newton_contact_storage = self
            .newton_contact_storage
            .max(solution.newton_contact_storage);
        self.newton_reduced_storage = self
            .newton_reduced_storage
            .max(solution.newton_reduced_storage);
        self.newton_line_searches += solution.newton_line_searches;
        self.constraint_active_set_trials += solution.active_set_trials;
        self.constraint_active_set_trial_rejections += solution.active_set_trial_rejections;
        self.constraint_warm_start_restarts += solution.warm_start_restarts;
        self.constraint_active_set_trial_matrix_storage = self
            .constraint_active_set_trial_matrix_storage
            .max(solution.active_set_trial_matrix_storage);
    }
}

/// CPU joint runtime with coupled inertia, implicit passive forces, bounded
/// drives, and timed inelastic stops. Tree joints are exact by reconstruction.
/// This API evaluates no terrain or body contacts and currently rejects loops;
/// it must not be selected as the application's complete physics backend.
pub struct CpuJointMachine {
    creation: CompiledCreation,
    passive: Vec<PassiveForce>,
    drives: Vec<CoordinateDrive>,
    completed: CpuSnapshot,
    diagnostics: JointTickDiagnostics,
}

impl CpuJointMachine {
    /// Loads compiled joint forces without changing authored construction state.
    ///
    /// # Errors
    /// Rejects loop topology, invalid numerical state/drives, out-of-travel initial
    /// coordinates, and the dense reference's capacity limit.
    pub fn new(
        creation: CompiledCreation,
        topology_generation: u64,
        mut state: MachineState,
    ) -> Result<Self, PhysicsError> {
        if !creation.dynamics.loops.is_empty() {
            return Err(PhysicsError::UnsupportedJointLoops);
        }
        let drives = creation.coordinate_drives.clone();
        if drives.len() != creation.dynamics.coordinate_bearings.len() {
            return Err(PhysicsError::InvalidDynamics);
        }
        for &drive in &drives {
            validate_drive(drive)?;
        }
        let model = MachineDynamics::assemble(&creation, &state.poses, &state.coordinates)?;
        model.body_motions(&state.velocities)?;
        model.factor(&vec![0.0; state.velocities.len()])?;
        validate_positions(&creation, &drives, &state)?;
        let passive = creation
            .dynamics
            .coordinate_bearings
            .iter()
            .map(|&row| PassiveForce::from_kind(creation.bearings[row].kind))
            .collect();
        state.poses = model.poses;
        Ok(Self {
            creation,
            passive,
            drives,
            completed: CpuSnapshot {
                tick: 0,
                topology_generation,
                state,
            },
            diagnostics: JointTickDiagnostics::default(),
        })
    }

    /// Last complete state; a failed tick never mutates it.
    pub fn snapshot(&self) -> &CpuSnapshot {
        &self.completed
    }

    /// Last attempted tick's work, including unpublished retries.
    pub fn diagnostics(&self) -> &JointTickDiagnostics {
        &self.diagnostics
    }

    /// Advances one external tick, retrying only numerical non-convergence with
    /// bounded finer substeps. Every attempt restarts from the same post-command
    /// state. Resolved drive changes commit only with the completed snapshot.
    ///
    /// # Errors
    /// Rejects wrong tick/generation, duplicate drive coordinates, invalid settings,
    /// invalid drives/impulses, incompatible limits, and unconverged final attempts.
    pub fn step(
        &mut self,
        gravity: DVec3,
        settings: JointTickSettings,
        impulses: &[ExternalImpulse],
        commands: &[DriveCommand],
    ) -> Result<&CpuSnapshot, PhysicsError> {
        self.step_candidate(gravity, settings, impulses, commands, None)
    }

    /// Advances one external tick against a published terrain scene.
    ///
    /// Experimental. Split terrain recovery, impact coverage and cross-backend
    /// publication semantics are incomplete, so some states still fail. A failed
    /// tick changes nothing: the caller must keep the previous snapshot, surface
    /// [`Self::diagnostics`], and never publish the attempt as a completed tick.
    ///
    /// # Errors
    /// Rejects wrong tick/generation, duplicate drive coordinates, invalid
    /// settings, invalid drives/impulses, terrain publications that disagree with
    /// the geometry's generation, and unconverged final attempts.
    pub fn step_with_terrain(
        &mut self,
        gravity: DVec3,
        settings: JointTickSettings,
        impulses: &[ExternalImpulse],
        commands: &[DriveCommand],
        terrain: &TerrainSubstep<'_>,
    ) -> Result<&CpuSnapshot, PhysicsError> {
        self.step_candidate(gravity, settings, impulses, commands, Some(terrain))
    }

    #[allow(clippy::too_many_lines)] // Keep command validation, bounded retries, and the sole commit point together.
    fn step_candidate(
        &mut self,
        gravity: DVec3,
        settings: JointTickSettings,
        impulses: &[ExternalImpulse],
        commands: &[DriveCommand],
        terrain: Option<&TerrainSubstep<'_>>,
    ) -> Result<&CpuSnapshot, PhysicsError> {
        self.diagnostics = JointTickDiagnostics {
            factorization: settings.factorization,
            failure_stage: Some(JointFailureStage::InputValidation),
            ..Default::default()
        };
        let tick = self
            .completed
            .tick
            .checked_add(1)
            .ok_or(PhysicsError::InvalidCommand)?;
        if !gravity.is_finite()
            || !matches!(settings.substeps, 1 | 2 | 4 | 8)
            || !matches!(settings.maximum_substeps, 1 | 2 | 4 | 8)
            || settings.maximum_substeps < settings.substeps
            || settings.force_iterations == 0
            || settings.constraint_iterations == 0
            || !settings.tolerance.is_finite()
            || settings.tolerance <= 0.0
        {
            return Err(PhysicsError::InvalidConstraints);
        }
        if terrain.is_some_and(|terrain| {
            terrain.topology_generation != self.completed.topology_generation
        }) {
            return Err(PhysicsError::InvalidCollision);
        }
        if impulses.iter().any(|c| {
            c.tick != tick
                || c.topology_generation != self.completed.topology_generation
                || c.body >= self.creation.compounds.len()
                || !c.point.is_finite()
                || !c.impulse.is_finite()
        }) {
            return Err(PhysicsError::InvalidCommand);
        }
        let mut drives = self.drives.clone();
        let mut changed = vec![false; drives.len()];
        for command in commands {
            if command.tick != tick
                || command.topology_generation != self.completed.topology_generation
                || command.coordinate >= drives.len()
                || changed[command.coordinate]
            {
                return Err(PhysicsError::InvalidCommand);
            }
            validate_drive(command.drive)?;
            changed[command.coordinate] = true;
            drives[command.coordinate] = command.drive;
        }
        validate_positions(&self.creation, &drives, &self.completed.state)?;
        let mut initial = self.completed.state.clone();
        self.diagnostics.failure_stage = Some(JointFailureStage::ExternalImpulse);
        apply_external_impulses(
            &self.creation,
            &mut initial,
            impulses,
            settings.factorization,
        )?;
        self.diagnostics.factorizations += usize::from(!impulses.is_empty());
        let mut subdivisions = settings.substeps;
        loop {
            self.diagnostics.failure_stage = None;
            self.diagnostics.attempts += 1;
            self.diagnostics.substeps = subdivisions;
            self.diagnostics.residual = 0.0;
            self.diagnostics.accepted_intervals = 0;
            self.diagnostics.accepted_seconds = 0.0;
            self.diagnostics.drive_impulses = vec![0.0; drives.len()];
            let mut candidate = initial.clone();
            let mut result = Ok(());
            for _ in 0..subdivisions {
                result = events::advance_interval(
                    &self.creation,
                    &self.passive,
                    &drives,
                    &mut candidate,
                    gravity,
                    TICK_SECONDS / f64::from(subdivisions),
                    settings,
                    terrain,
                    &mut self.diagnostics,
                );
                if result.is_err() {
                    break;
                }
            }
            if let Err(error) = &result {
                let stage = *self
                    .diagnostics
                    .failure_stage
                    .get_or_insert(JointFailureStage::FiniteStep);
                self.diagnostics.attempt_failures.push(JointAttemptFailure {
                    substeps: subdivisions,
                    stage,
                    error: error.clone(),
                    accepted_seconds: self.diagnostics.accepted_seconds,
                });
            }
            match result {
                Ok(()) => {
                    self.completed = CpuSnapshot {
                        tick,
                        topology_generation: self.completed.topology_generation,
                        state: candidate,
                    };
                    self.drives = drives;
                    self.diagnostics.published = true;
                    self.diagnostics.failure_stage = None;
                    return Ok(&self.completed);
                }
                Err(PhysicsError::NotConverged) if subdivisions < settings.maximum_substeps => {
                    subdivisions *= 2;
                }
                Err(error) => return Err(error),
            }
        }
    }
}

pub(crate) fn bounds(
    creation: &CompiledCreation,
    drives: &[CoordinateDrive],
    coordinate: usize,
) -> [f64; 2] {
    let physical = creation.bearings[creation.dynamics.coordinate_bearings[coordinate]]
        .kind
        .bounds();
    [
        f64::from(physical[0].max(drives[coordinate].min_angle)),
        f64::from(physical[1].min(drives[coordinate].max_angle)),
    ]
}

fn validate_positions(
    creation: &CompiledCreation,
    drives: &[CoordinateDrive],
    state: &MachineState,
) -> Result<(), PhysicsError> {
    for (coordinate, &q) in state.coordinates.iter().enumerate() {
        let [lower, upper] = bounds(creation, drives, coordinate);
        if lower > upper || !q.is_finite() || q < lower - 1e-10 || q > upper + 1e-10 {
            return Err(PhysicsError::InvalidConstraints);
        }
    }
    Ok(())
}

// Pose-local surface inputs to the numerical substep. The public joint-only
// runtime supplies None until continuous collision and publication are complete.
#[derive(Clone, Copy)]
struct SubstepContacts<'a> {
    query: &'a crate::TerrainContactQuery,
    restitution_threshold: f64,
    stiction_threshold: f64,
}

/// How a tick resolves the terrain contacts it meets inside a substep.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TerrainIntegration {
    /// Split every substep at each arrival and resolve it there. Strictest, and
    /// the policy whose trial budget a chattering contact can still exhaust.
    #[default]
    EventResolved,
    /// Defer contacts whose collider moves slowly to the substep endpoint. The
    /// endpoint impulse is bounded by the same independent depth certificate.
    SlowContactsAtEndpoint,
}

/// One immutable terrain publication, held through all of a tick's unpublished
/// retries. Generations, origin and policy bounds cannot change within a tick.
#[derive(Clone, Copy)]
pub struct TerrainSubstep<'a> {
    /// Contact event policy for this tick.
    pub integration: TerrainIntegration,
    /// Published collision scene; its generation must not change within a tick.
    pub scene: &'a crate::TerrainContactScene,
    /// Immutable collider geometry compiled for `topology_generation`.
    pub geometry: &'a crate::MachineCollisionGeometry,
    /// Construction generation the geometry and state belong to.
    pub topology_generation: u64,
    /// Floating-origin offset applied to published terrain, in metres.
    pub origin: DVec3,
    /// Largest certified penetration an accepted path may reach, in metres.
    pub maximum_depth: f64,
    /// Bound on separating-axis advancement evaluations per triangle.
    pub maximum_evaluations: usize,
    /// Bound on event-localization trials per interval.
    pub maximum_event_trials: usize,
    /// Impact speed above which material restitution applies, in metres/second.
    pub restitution_threshold: f64,
    /// Tangential speed below which a contact is treated as sticking.
    pub stiction_threshold: f64,
}

impl TerrainSubstep<'_> {
    fn contacts(
        &self,
        state: &MachineState,
        diagnostics: &mut JointTickDiagnostics,
    ) -> Result<crate::TerrainContactQuery, PhysicsError> {
        diagnostics.terrain_contact_queries += 1;
        let query = self
            .scene
            .activation_contacts(self.geometry, &state.poses, self.origin)
            .inspect_err(|_| {
                diagnostics.terrain_contact_query_failures += 1;
                diagnostics.failure_stage = Some(JointFailureStage::TerrainQuery);
            })?;
        diagnostics.terrain_chunk_candidates += query.chunk_candidates;
        diagnostics.terrain_triangle_candidates += query.triangle_candidates;
        Ok(query)
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)] // Direct numerical probes prepare exactly one initial pose model.
fn substep(
    creation: &CompiledCreation,
    passive: &[PassiveForce],
    drives: &[CoordinateDrive],
    state: &mut MachineState,
    gravity: DVec3,
    dt: f64,
    settings: JointTickSettings,
    contacts: Option<SubstepContacts<'_>>,
    terrain: Option<&TerrainSubstep<'_>>,
    diagnostics: &mut JointTickDiagnostics,
) -> Result<events::TrialOutcome, PhysicsError> {
    let model = MachineDynamics::assemble(creation, &state.poses, &state.coordinates)?;
    diagnostics.dynamics_assemblies += 1;
    integrate_substep(
        creation,
        passive,
        drives,
        state,
        &model,
        gravity,
        dt,
        settings,
        contacts,
        terrain,
        diagnostics,
    )
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // One ordered numerical substep with explicit inputs and work accounting.
fn integrate_substep(
    creation: &CompiledCreation,
    passive: &[PassiveForce],
    drives: &[CoordinateDrive],
    state: &mut MachineState,
    model: &MachineDynamics,
    gravity: DVec3,
    dt: f64,
    settings: JointTickSettings,
    contacts: Option<SubstepContacts<'_>>,
    terrain: Option<&TerrainSubstep<'_>>,
    diagnostics: &mut JointTickDiagnostics,
) -> Result<events::TrialOutcome, PhysicsError> {
    let surface = contacts
        .as_ref()
        .map(|contacts| {
            contacts.query.impact_constraints(
                model,
                &state.velocities,
                contacts.restitution_threshold,
                contacts.stiction_threshold,
            )
        })
        .transpose()?;
    let empty = crate::TerrainImpactConstraints {
        blocks: Vec::new(),
        point_indices: Vec::new(),
    };
    impact::activate(
        creation,
        model,
        drives,
        state,
        surface.as_ref().unwrap_or(&empty),
        settings,
        diagnostics,
    )?;
    // Rebuild from outgoing motion so restitution is applied once. Separating
    // points release: they cannot supply a finite force while moving away.
    let sustaining = contacts
        .as_ref()
        .map(|contacts| {
            events::SustainingSurface::new(
                contacts.query,
                model,
                &state.velocities,
                settings.tolerance,
            )
        })
        .transpose()?;
    let surface = sustaining
        .as_ref()
        .zip(contacts.as_ref())
        .map(|(surface, contacts)| {
            diagnostics.released_surface_points +=
                contacts.query.contacts.len() - surface.query.contacts.len();
            surface.query.impact_constraints(
                model,
                &state.velocities,
                contacts.restitution_threshold,
                contacts.stiction_threshold,
            )
        })
        .transpose()?;
    let size = state.velocities.len();
    let mut diagonal = vec![0.0; size];
    for (coordinate, &row) in creation.dynamics.coordinate_velocities.iter().enumerate() {
        diagonal[row] = passive[coordinate].implicit_diagonal(dt * 0.5);
    }
    let factor = settings
        .factorization
        .factor(creation, model, &state.coordinates, &diagonal)?;
    diagnostics.factorizations += 1;
    let mut blocks = Vec::new();
    let mut desired = Vec::new();
    let mut drive_indices = Vec::new();
    let mut contact_kinematics = Vec::new();
    for (coordinate, &row) in creation.dynamics.coordinate_velocities.iter().enumerate() {
        let drive = drives[coordinate];
        let position = state.coordinates[coordinate];
        let [lower, upper] = bounds(creation, drives, coordinate);
        let mut add = |sign: f64, target: f64, minimum: f64, maximum: f64| {
            let mut jacobian = vec![0.0; size];
            jacobian[row] = sign;
            blocks.push(ConstraintBlock {
                jacobian: vec![jacobian],
                target: vec![0.0],
                bounds: vec![ImpulseBounds { minimum, maximum }],
                contacts: Vec::new(),
            });
            desired.push(target);
        };
        if lower.is_finite()
            && position <= lower + stops::POSITION_TOLERANCE
            && state.velocities[row] <= settings.tolerance
        {
            add(1.0, 0.0, 0.0, f64::INFINITY);
        }
        if upper.is_finite()
            && position >= upper - stops::POSITION_TOLERANCE
            && state.velocities[row] >= -settings.tolerance
        {
            add(-1.0, 0.0, 0.0, f64::INFINITY);
        }
        if drive.mode != DriveMode::Passive {
            let target = drive_target(drive, position);
            let capacity = drive_budget(
                drive,
                f64::from(creation.loop_topology.coordinate_axis_inertia[coordinate]),
                state.velocities[row],
                target,
                dt,
            );
            if capacity.is_nan() || capacity < 0.0 {
                return Err(PhysicsError::InvalidDynamics);
            }
            if capacity > 0.0 {
                add(1.0, target, -capacity, capacity);
                drive_indices.push((coordinate, blocks.len() - 1));
            }
        }
    }
    if let Some((contacts, impact)) = sustaining.as_ref().zip(surface) {
        diagnostics.surface_points += contacts.query.contacts.len();
        diagnostics.surface_manifolds += impact.blocks.len();
        let mut point_index = 0;
        for block in impact.blocks {
            let first = desired.len();
            // Store absolute desired row velocity. Every nonlinear free-force
            // iterate then supplies the flattened required velocity change.
            desired.extend(
                block
                    .jacobian
                    .iter()
                    .zip(&block.target)
                    .map(|(row, target)| {
                        target
                            + row
                                .iter()
                                .zip(&state.velocities)
                                .map(|(j, v)| j * v)
                                .sum::<f64>()
                    }),
            );
            // Follow the material point through the force-dependent end pose;
            // freezing J omits its centripetal/Coriolis normal acceleration.
            let mut row = first;
            for law in &block.contacts {
                let point = contacts.query.contacts[impact.point_indices[point_index]];
                let local = |body: usize, world: DVec3| {
                    model.poses[body].rotation.inverse() * (world - model.poses[body].position)
                };
                contact_kinematics.push(contact_kinematics::ContactPoint {
                    row,
                    body: point.body,
                    local_point: local(point.body, point.body_point),
                    normal: point.normal,
                    other: point
                        .other_body
                        .map(|body| (body, local(body, point.terrain_point))),
                });
                point_index += 1;
                row += if law.rolling_length.is_some() { 5 } else { 3 };
            }
            blocks.push(block);
        }
    }
    diagnostics.constraint_rows_prepared +=
        blocks.iter().map(|block| block.target.len()).sum::<usize>();
    let mut response = PreparedConstraints::new(&factor, &blocks)?;
    diagnostics.response_preparation_solves += response.preparation_factor_solves();
    let mut velocity = state.velocities.clone();
    let mut adjusted_desired = desired.clone();
    if !contact_kinematics.is_empty() {
        let world = |body: usize, local: DVec3| {
            let pose = model.poses[body];
            (body, pose.position + pose.rotation * local)
        };
        let points = contact_kinematics
            .iter()
            .flat_map(|point| {
                std::iter::once(world(point.body, point.local_point))
                    .chain(point.other.map(|(body, local)| world(body, local)))
            })
            .collect::<Vec<_>>();
        let bias = model.point_acceleration_bias(creation, &state.velocities, &points)?;
        diagnostics.contact_bias_traversals += 1;
        let mut bias = bias.into_iter();
        for point in &contact_kinematics {
            let mut relative = bias.next().ok_or(PhysicsError::InvalidDynamics)?;
            if point.other.is_some() {
                relative -= bias.next().ok_or(PhysicsError::InvalidDynamics)?;
            }
            adjusted_desired[point.row] -= dt * point.normal.dot(relative);
        }
    }
    let mut last_residual = f64::INFINITY;
    let mut warm_impulses: Option<Vec<f64>> = None;
    for _ in 0..settings.force_iterations {
        diagnostics.force_iterations += 1;
        let rhs = midpoint_rhs(
            creation, passive, state, model, &diagonal, &velocity, gravity, dt,
        )?;
        diagnostics.dynamics_assemblies += 1;
        let mut free = rhs.clone();
        factor.solve(&mut free)?;
        diagnostics.force_factor_solves += 1;
        let targets = blocks
            .iter()
            .flat_map(|block| &block.jacobian)
            .zip(&adjusted_desired)
            .map(|(row, target)| target - row.iter().zip(&free).map(|(j, v)| j * v).sum::<f64>())
            .collect::<Vec<_>>();
        let solution = response.solve_from(
            &targets,
            warm_impulses.as_deref(),
            settings.constraint_iterations,
            settings.tolerance * 0.1,
        )?;
        diagnostics.constraint_factor_solves += solution.factor_solves;
        diagnostics.constraint_iterations += solution.iterations;
        diagnostics.record_newton(&solution);
        diagnostics.constraint_velocity_residual = diagnostics
            .constraint_velocity_residual
            .max(solution.residual);
        if !solution.converged {
            #[cfg(test)]
            if let Some(path) = std::env::var_os("MECHANIC_FORCE_CAPTURE") {
                let mut mass = model.mass_matrix.clone();
                for i in 0..size {
                    mass[i * size + i] += diagonal[i];
                }
                let mut cursor = 0;
                let rows = blocks
                    .iter()
                    .map(|block| {
                        let target = targets[cursor..cursor + block.target.len()].to_vec();
                        cursor += block.target.len();
                        (
                            &block.jacobian,
                            target,
                            block
                                .bounds
                                .iter()
                                .map(|b| (b.minimum, b.maximum))
                                .collect::<Vec<_>>(),
                            block
                                .contacts
                                .iter()
                                .map(|c| {
                                    (
                                        c.static_coefficient,
                                        c.kinetic_coefficient,
                                        c.sliding,
                                        c.rolling_length,
                                    )
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>();
                std::fs::write(path, ron::to_string(&(mass, rows, &warm_impulses)).unwrap())
                    .unwrap();
            }
            diagnostics.residual = diagnostics.residual.max(solution.residual);
            return Err(PhysicsError::NotConverged);
        }
        warm_impulses = Some(solution.impulses.clone());
        let next = free
            .iter()
            .zip(&solution.velocity_change)
            .map(|(v, d)| v + d)
            .collect::<Vec<_>>();
        // Verify the full nonlinear midpoint equation, including pose-dependent
        // inertia and gyroscopic forces. Constraint impulses cancel in this difference.
        let actual_rhs = midpoint_rhs(
            creation, passive, state, model, &diagonal, &next, gravity, dt,
        )?;
        diagnostics.dynamics_assemblies += 1;
        let mut error = rhs
            .iter()
            .zip(actual_rhs)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        factor.solve(&mut error)?;
        diagnostics.force_factor_solves += 1;
        let force_residual = error.iter().map(|v| v.abs()).fold(0.0, f64::max);
        diagnostics.force_velocity_residual =
            diagnostics.force_velocity_residual.max(force_residual);
        let next_desired = contact_kinematics::endpoint_targets(
            creation,
            state,
            &next,
            dt,
            &blocks,
            &desired,
            &contact_kinematics,
            diagnostics,
        )?;
        let kinematic_residual = adjusted_desired
            .iter()
            .zip(&next_desired)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        diagnostics.contact_kinematic_residual = diagnostics
            .contact_kinematic_residual
            .max(kinematic_residual);
        let residual = force_residual
            .max(solution.residual)
            .max(kinematic_residual);
        velocity = next;
        if residual <= settings.tolerance {
            diagnostics.residual = diagnostics.residual.max(residual);
            if terrain.is_some()
                && let Some(fraction) = sustaining
                    .as_ref()
                    .map(|surface| {
                        surface.reversal(
                            creation,
                            state,
                            &velocity,
                            dt,
                            settings.tolerance,
                            terrain
                                .filter(|terrain| {
                                    terrain.integration
                                        == TerrainIntegration::SlowContactsAtEndpoint
                                })
                                .map_or(0.0, |terrain| terrain.restitution_threshold),
                            diagnostics,
                        )
                    })
                    .transpose()?
                    .flatten()
            {
                return Ok(events::TrialOutcome::Release(fraction));
            }
            if let Some(fraction) =
                stops::reversal(creation, drives, state, &velocity, settings.tolerance)
            {
                return Ok(events::TrialOutcome::Release(fraction));
            }
            let incoming = state.velocities.clone();
            for (old, &new) in state.velocities.iter_mut().zip(&velocity) {
                *old = 0.5 * (*old + new);
            }
            if let Some(hit) = stops::first_crossing(creation, drives, state, dt) {
                return Ok(events::TrialOutcome::JointStop(hit));
            }
            let path = events::validate_path(
                creation,
                state,
                dt,
                terrain,
                false,
                Some((&incoming, &velocity)),
                diagnostics,
            )?;
            if let events::PathOutcome::Refine(hit) = path {
                return Ok(events::TrialOutcome::Refine(hit));
            }
            advance_positions(creation, state, dt);
            state.velocities = velocity;
            if validate_positions(creation, drives, state).is_err() {
                correct_joint_positions(
                    creation,
                    drives,
                    state,
                    &factor,
                    settings,
                    terrain,
                    diagnostics,
                )?;
            }
            let final_model =
                MachineDynamics::assemble(creation, &state.poses, &state.coordinates)?;
            diagnostics.dynamics_assemblies += 1;
            final_model.body_motions(&state.velocities)?;
            state.poses.clone_from(&final_model.poses);
            if let events::PathOutcome::Activate(hit) = path {
                events::activate_endpoint(
                    creation,
                    drives,
                    state,
                    &final_model,
                    terrain.ok_or(PhysicsError::InvalidCollision)?,
                    hit,
                    settings,
                    diagnostics,
                )?;
            }
            let stop_arrival = stops::closing(creation, drives, state, settings.tolerance);
            if stop_arrival {
                let query = terrain
                    .map(|terrain| terrain.contacts(state, diagnostics))
                    .transpose()?
                    .unwrap_or_default();
                let constraints = query.impact_constraints(
                    &final_model,
                    &state.velocities,
                    terrain.map_or(0.0, |t| t.restitution_threshold),
                    terrain.map_or(0.0, |t| t.stiction_threshold),
                )?;
                impact::activate(
                    creation,
                    &final_model,
                    drives,
                    state,
                    &constraints,
                    settings,
                    diagnostics,
                )?;
            }
            return Ok(events::TrialOutcome::Complete {
                arrived: stop_arrival || matches!(path, events::PathOutcome::Activate(_)),
                drive_impulses: drive_indices
                    .into_iter()
                    .map(|(coordinate, index)| (coordinate, solution.impulses[index]))
                    .collect(),
            });
        }
        adjusted_desired = next_desired;
        last_residual = residual;
    }
    diagnostics.residual = diagnostics.residual.max(last_residual);
    Err(PhysicsError::NotConverged)
}

#[allow(clippy::too_many_arguments)] // Explicit immutable numerical inputs; no hidden runtime state.
fn midpoint_rhs(
    creation: &CompiledCreation,
    passive: &[PassiveForce],
    initial: &MachineState,
    base: &MachineDynamics,
    diagonal: &[f64],
    candidate: &[f64],
    gravity: DVec3,
    dt: f64,
) -> Result<Vec<f64>, PhysicsError> {
    let mut midpoint = initial.clone();
    for (v, &next) in midpoint.velocities.iter_mut().zip(candidate) {
        *v = 0.5 * (*v + next);
    }
    advance_positions(creation, &mut midpoint, dt * 0.5);
    let model = MachineDynamics::assemble(creation, &midpoint.poses, &midpoint.coordinates)?;
    let gravity = model.gravity_force(creation, gravity)?;
    let bias = model.inertial_bias(creation, &midpoint.velocities)?;
    let size = candidate.len();
    let mut rhs = (0..size)
        .map(|row| {
            diagonal[row] * candidate[row]
                + dt * (gravity[row] - bias[row])
                + (0..size)
                    .map(|column| {
                        base.mass_matrix[row * size + column] * candidate[column]
                            - model.mass_matrix[row * size + column]
                                * (candidate[column] - initial.velocities[column])
                    })
                    .sum::<f64>()
        })
        .collect::<Vec<_>>();
    for (coordinate, &row) in creation.dynamics.coordinate_velocities.iter().enumerate() {
        rhs[row] += dt
            * passive[coordinate].force(midpoint.coordinates[coordinate], midpoint.velocities[row]);
    }
    Ok(rhs)
}

fn correct_joint_positions(
    creation: &CompiledCreation,
    drives: &[CoordinateDrive],
    state: &mut MachineState,
    factor: &crate::DynamicsFactor,
    settings: JointTickSettings,
    terrain: Option<&TerrainSubstep<'_>>,
    diagnostics: &mut JointTickDiagnostics,
) -> Result<(), PhysicsError> {
    let mut blocks = Vec::new();
    for (coordinate, &row) in creation.dynamics.coordinate_velocities.iter().enumerate() {
        let [lower, upper] = bounds(creation, drives, coordinate);
        for (sign, gap) in [
            (1.0, lower - state.coordinates[coordinate]),
            (-1.0, state.coordinates[coordinate] - upper),
        ] {
            if !gap.is_finite() {
                continue;
            }
            let mut jacobian = vec![0.0; state.velocities.len()];
            jacobian[row] = sign;
            blocks.push(ConstraintBlock {
                jacobian: vec![jacobian],
                target: vec![gap],
                bounds: vec![ImpulseBounds {
                    minimum: 0.0,
                    maximum: f64::INFINITY,
                }],
                contacts: Vec::new(),
            });
        }
    }
    let correction =
        crate::solve_constraints(factor, &blocks, settings.constraint_iterations, 1e-12)?;
    diagnostics.position_factor_solves += correction.factor_solves;
    diagnostics.record_newton(&correction);
    if !correction.converged {
        return Err(PhysicsError::NotConverged);
    }
    // Only position is corrected. These scratch rates never become physical velocity.
    let physical = std::mem::replace(&mut state.velocities, correction.velocity_change);
    let validation = validate_terrain_path(creation, state, 1.0, terrain, true, diagnostics);
    if validation.is_ok() {
        advance_positions(creation, state, 1.0);
    }
    state.velocities = physical;
    validation?;
    validate_positions(creation, drives, state).map_err(|_| PhysicsError::NotConverged)
}

fn validate_terrain_path(
    creation: &CompiledCreation,
    state: &MachineState,
    dt: f64,
    terrain: Option<&TerrainSubstep<'_>>,
    correction: bool,
    diagnostics: &mut JointTickDiagnostics,
) -> Result<(), PhysicsError> {
    match events::validate_path(creation, state, dt, terrain, correction, None, diagnostics)? {
        events::PathOutcome::Clear => Ok(()),
        _ => Err(PhysicsError::NotConverged),
    }
}

#[cfg(test)]
mod tests;
