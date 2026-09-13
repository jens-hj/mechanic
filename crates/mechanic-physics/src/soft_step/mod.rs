//! Soft-step game solver: every valid tick publishes.
//!
//! The `Box2D` v3 soft step on the reduced-coordinate machine model. Each tick queries
//! contacts once with a speculative margin, then runs fixed substeps: integrate
//! forces, warm start, a biased projected Gauss–Seidel pass over drive, joint-limit
//! and contact rows, integrate positions, and an unbiased relaxing pass. A
//! restitution pass follows the substeps. Quality is reported in diagnostics; only
//! invalid input returns an error, and numerical trouble marks the tick degraded.

mod solve;
#[cfg(test)]
mod tests;

use std::{collections::BTreeMap, time::Instant};

use bevy_math::DVec3;
use mechanic_core::{CompiledCreation, CoordinateDrive};

use crate::{
    CpuSnapshot, DriveCommand, DynamicsFactorization, ExternalImpulse, MachineCollisionGeometry,
    MachineDynamics, MachineState, PhysicsError, TICK_SECONDS, TerrainContactFeature,
    TerrainContactScene,
    free_motion::apply_external_impulses,
    joint_forces::{PassiveForce, validate_drive},
    joint_machine::bounds,
};
use solve::{Contact, JointImpulses, Machine};

/// Offset separating submerged-vertex features from clipped manifold corners,
/// so their warm-start impulses never share a key.
const SUBMERGED_CORNERS: usize = 1 << 20;

/// Fixed solver parameters. Defaults suit 60 Hz game ticks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftStepSettings {
    /// Numerical factor for the effective dynamics.
    pub factorization: DynamicsFactorization,
    /// Substeps per external tick.
    pub substeps: u32,
    /// Biased solver passes per substep.
    pub iterations: u32,
    /// Unbiased relaxing passes per substep, after positions advance.
    pub relax_iterations: u32,
    /// Contact and joint-limit stiffness in hertz, capped at a quarter of the
    /// substep rate.
    pub contact_hertz: f64,
    /// Contact damping ratio; large values keep corrections from overshooting.
    pub damping_ratio: f64,
    /// Largest speed used to push penetrating contacts apart, in metres/second.
    pub push_out: f64,
    /// Penetration left uncorrected at rest, in metres.
    pub slop: f64,
    /// Gap within which contacts and joint limits act before arrival, in metres
    /// (radians for revolute limits).
    pub speculative: f64,
    /// Approach speed above which material restitution applies, in metres/second.
    pub restitution_threshold: f64,
    /// Tangential speed below which static friction applies, in metres/second.
    pub stiction_speed: f64,
    /// Bound on every generalized velocity after a substep.
    pub maximum_speed: f64,
}

impl Default for SoftStepSettings {
    fn default() -> Self {
        Self {
            factorization: DynamicsFactorization::Articulated,
            substeps: 4,
            iterations: 1,
            relax_iterations: 1,
            contact_hertz: 30.0,
            damping_ratio: 10.0,
            push_out: 3.0,
            slop: 0.001,
            speculative: 0.02,
            restitution_threshold: 1.0,
            stiction_speed: 0.05,
            maximum_speed: 500.0,
        }
    }
}

impl SoftStepSettings {
    fn is_valid(&self) -> bool {
        let nonnegative = [
            self.contact_hertz,
            self.damping_ratio,
            self.push_out,
            self.slop,
            self.speculative,
            self.restitution_threshold,
            self.stiction_speed,
        ];
        (1..=64).contains(&self.substeps)
            && self.iterations > 0
            && nonnegative
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
            && self.maximum_speed.is_finite()
            && self.maximum_speed > 0.0
    }
}

/// The published collision scene one tick collides with.
#[derive(Clone, Copy)]
pub struct SoftStepTerrain<'a> {
    /// Published terrain.
    pub scene: &'a TerrainContactScene,
    /// Collider geometry compiled for `topology_generation`.
    pub geometry: &'a MachineCollisionGeometry,
    /// Construction generation the geometry and state belong to.
    pub topology_generation: u64,
    /// Floating-origin offset applied to published terrain, in metres.
    pub origin: DVec3,
}

/// Work and quality of the most recent tick.
#[derive(Clone, Debug, Default)]
pub struct SoftStepDiagnostics {
    /// Contact points solved this tick.
    pub contacts: usize,
    /// Scalar rows in the last substep, across contacts, drives and limits.
    pub rows: usize,
    /// Deepest contact overlap seen at any substep, in metres.
    pub maximum_penetration: f64,
    /// Largest remaining approach speed at a loaded contact after the tick.
    pub velocity_error: f64,
    /// Whether numerical trouble forced a fallback this tick.
    pub degraded: bool,
    /// First reason the tick degraded.
    pub degraded_reason: Option<&'static str>,
    /// Contact query time, in milliseconds.
    pub query_ms: f64,
    /// Integration and solve time, in milliseconds.
    pub solve_ms: f64,
    /// Signed drive impulses summed over the tick, in N·s or N·m·s.
    pub drive_impulses: Vec<f64>,
}

impl SoftStepDiagnostics {
    fn degrade(&mut self, reason: &'static str) {
        self.degraded = true;
        self.degraded_reason.get_or_insert(reason);
    }
}

/// CPU machine stepped by the soft-step solver. Tree joints are exact by
/// reconstruction; contacts, drives and joint limits are soft rows.
pub struct CpuMachine {
    creation: CompiledCreation,
    passive: Vec<PassiveForce>,
    drives: Vec<CoordinateDrive>,
    completed: CpuSnapshot,
    diagnostics: SoftStepDiagnostics,
    warm: BTreeMap<TerrainContactFeature, [f64; 5]>,
}

impl CpuMachine {
    /// Loads a compiled creation. Joint coordinates outside their travel are
    /// clamped into it.
    ///
    /// # Errors
    /// Rejects closed loops, mismatched state or drive rows, invalid drives, and
    /// non-finite or non-positive dynamics.
    pub fn new(
        creation: CompiledCreation,
        topology_generation: u64,
        mut state: MachineState,
    ) -> Result<Self, PhysicsError> {
        if !creation.dynamics.loops.is_empty() {
            return Err(PhysicsError::UnsupportedJointLoops);
        }
        let drives = creation.coordinate_drives.clone();
        if drives.len() != creation.dynamics.coordinate_bearings.len()
            || state.coordinates.len() != drives.len()
            || state.velocities.len() != creation.dynamics.elimination_parent.len()
        {
            return Err(PhysicsError::InvalidDynamics);
        }
        for &drive in &drives {
            validate_drive(drive)?;
        }
        clamp_coordinates(&creation, &drives, &mut state);
        let model = MachineDynamics::assemble(&creation, &state.poses, &state.coordinates)?;
        model.body_motions(&state.velocities)?;
        state.poses = model.poses;
        let passive = creation
            .dynamics
            .coordinate_bearings
            .iter()
            .map(|&row| PassiveForce::from_kind(creation.bearings[row].kind))
            .collect();
        Ok(Self {
            creation,
            passive,
            drives,
            completed: CpuSnapshot {
                tick: 0,
                topology_generation,
                state,
            },
            diagnostics: SoftStepDiagnostics::default(),
            warm: BTreeMap::new(),
        })
    }

    /// Last published state.
    pub fn snapshot(&self) -> &CpuSnapshot {
        &self.completed
    }

    /// Work and quality of the last tick.
    pub fn diagnostics(&self) -> &SoftStepDiagnostics {
        &self.diagnostics
    }

    /// Advances and publishes one external tick.
    ///
    /// # Errors
    /// Only invalid input: a wrong tick or generation, duplicate or invalid drive
    /// commands, invalid impulses or settings, or terrain geometry compiled for
    /// another topology. Invalid input changes nothing.
    #[allow(clippy::too_many_lines)] // Validation, the substep loop and the sole publication stay together.
    pub fn step(
        &mut self,
        gravity: DVec3,
        settings: &SoftStepSettings,
        impulses: &[ExternalImpulse],
        commands: &[DriveCommand],
        terrain: Option<SoftStepTerrain<'_>>,
    ) -> Result<&CpuSnapshot, PhysicsError> {
        let tick = self
            .completed
            .tick
            .checked_add(1)
            .ok_or(PhysicsError::InvalidCommand)?;
        if !gravity.is_finite() || !settings.is_valid() {
            return Err(PhysicsError::InvalidConstraints);
        }
        let generation = self.completed.topology_generation;
        if terrain.is_some_and(|terrain| terrain.topology_generation != generation) {
            return Err(PhysicsError::InvalidCollision);
        }
        if impulses.iter().any(|impulse| {
            impulse.tick != tick
                || impulse.topology_generation != generation
                || impulse.body >= self.creation.compounds.len()
                || !impulse.point.is_finite()
                || !impulse.impulse.is_finite()
        }) {
            return Err(PhysicsError::InvalidCommand);
        }
        let mut drives = self.drives.clone();
        let mut changed = vec![false; drives.len()];
        for command in commands {
            if command.tick != tick
                || command.topology_generation != generation
                || command.coordinate >= drives.len()
                || changed[command.coordinate]
            {
                return Err(PhysicsError::InvalidCommand);
            }
            validate_drive(command.drive)?;
            changed[command.coordinate] = true;
            drives[command.coordinate] = command.drive;
        }

        // Everything below publishes, degraded if necessary.
        self.drives = drives;
        let started = Instant::now();
        let mut diagnostics = SoftStepDiagnostics {
            drive_impulses: vec![0.0; self.drives.len()],
            ..SoftStepDiagnostics::default()
        };
        let mut state = self.completed.state.clone();
        if apply_external_impulses(&self.creation, &mut state, impulses, settings.factorization)
            .is_err()
        {
            diagnostics.degrade("external impulse");
        }
        let mut contacts = terrain.map_or_else(Vec::new, |terrain| {
            self.contacts(terrain, &state, settings, &mut diagnostics)
        });
        diagnostics.contacts = contacts.len();
        diagnostics.query_ms = started.elapsed().as_secs_f64() * 1000.0;

        let machine = Machine {
            creation: &self.creation,
            passive: &self.passive,
            drives: &self.drives,
        };
        let mut joints = JointImpulses::new(self.drives.len());
        let dt = TICK_SECONDS / f64::from(settings.substeps);
        let mut last = None;
        for index in 0..settings.substeps {
            let before = state.clone();
            let result = solve::substep(
                &machine,
                &mut state,
                &mut contacts,
                &mut joints,
                gravity,
                dt,
                settings,
                index == 0,
                &mut diagnostics,
            );
            match result {
                Ok(points) if sane(&mut state, settings, &mut diagnostics) => last = Some(points),
                Ok(_) | Err(_) => {
                    state = before;
                    state.velocities.fill(0.0);
                    last = None;
                    diagnostics.degrade("numerical substep");
                    break;
                }
            }
        }
        if let Some(points) = &last {
            diagnostics.velocity_error =
                solve::restitution(&mut contacts, points, &mut state.velocities, settings);
        }
        match MachineDynamics::assemble(&self.creation, &state.poses, &state.coordinates) {
            Ok(model) if sane(&mut state, settings, &mut diagnostics) => state.poses = model.poses,
            Ok(_) | Err(_) => {
                state = self.completed.state.clone();
                state.velocities.fill(0.0);
                diagnostics.degrade("final pose");
            }
        }

        self.warm = contacts
            .iter()
            .map(|contact| (contact.source.feature, contact.impulses))
            .collect();
        diagnostics.solve_ms = started.elapsed().as_secs_f64() * 1000.0 - diagnostics.query_ms;
        self.diagnostics = diagnostics;
        self.completed = CpuSnapshot {
            tick,
            topology_generation: generation,
            state,
        };
        Ok(&self.completed)
    }

    fn contacts(
        &self,
        terrain: SoftStepTerrain<'_>,
        state: &MachineState,
        settings: &SoftStepSettings,
        diagnostics: &mut SoftStepDiagnostics,
    ) -> Vec<Contact> {
        // One query serves the whole tick, so it must reach as far as any body can
        // travel in it. Angular reach assumes a one-metre lever.
        let reach = MachineDynamics::assemble(&self.creation, &state.poses, &state.coordinates)
            .and_then(|model| model.body_motions(&state.velocities))
            .map_or(0.0, |motions| {
                motions
                    .iter()
                    .map(|motion| motion.linear.length() + motion.angular.length())
                    .fold(0.0, f64::max)
            });
        let margin = (settings.speculative + TICK_SECONDS * reach).min(1.0);
        // Clipping nearly parallel faces within a wide margin can refuse a query;
        // narrower queries still keep the bodies apart.
        let Some(query) = [margin, settings.speculative, 0.0]
            .into_iter()
            .find_map(|margin| {
                terrain
                    .scene
                    .proximity(terrain.geometry, &state.poses, terrain.origin, margin)
                    .ok()
            })
        else {
            diagnostics.degrade("contact query");
            return Vec::new();
        };
        let mut points = query.contacts;
        // A clipped manifold only holds points where the collider crosses the
        // surface, all at zero gap; a tilted body's submerged corner lies inside
        // that polygon. Add those vertices with their real depth.
        if let Ok(recovery) =
            terrain
                .scene
                .recovery_contacts(terrain.geometry, &state.poses, terrain.origin)
        {
            for mut vertex in recovery.contacts {
                if vertex.depth > 0.0
                    && points
                        .iter()
                        .all(|point| point.body_point.distance(vertex.body_point) > 1e-4)
                {
                    vertex.feature.corner += SUBMERGED_CORNERS;
                    points.push(vertex);
                }
            }
        }
        points
            .into_iter()
            .map(|point| Contact::new(point, &state.poses, self.warm.get(&point.feature)))
            .collect()
    }
}

fn clamp_coordinates(
    creation: &CompiledCreation,
    drives: &[CoordinateDrive],
    state: &mut MachineState,
) {
    for (coordinate, value) in state.coordinates.iter_mut().enumerate() {
        let [lower, upper] = bounds(creation, drives, coordinate);
        if lower <= upper {
            *value = value.clamp(lower, upper);
        }
    }
}

// Rejects non-finite state and bounds runaway speeds, which count as degraded.
fn sane(
    state: &mut MachineState,
    settings: &SoftStepSettings,
    diagnostics: &mut SoftStepDiagnostics,
) -> bool {
    let finite = state
        .velocities
        .iter()
        .chain(&state.coordinates)
        .all(|value| value.is_finite())
        && state
            .poses
            .iter()
            .all(|pose| pose.position.is_finite() && pose.rotation.is_finite());
    if !finite {
        return false;
    }
    let maximum = settings.maximum_speed;
    if state.velocities.iter().any(|value| value.abs() > maximum) {
        diagnostics.degrade("speed limit");
        for value in &mut state.velocities {
            *value = value.clamp(-maximum, maximum);
        }
    }
    true
}
