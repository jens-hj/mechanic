//! Soft-step game solver: every valid tick publishes.
//!
//! The `Box2D` v3 soft step on the reduced-coordinate machine model. Each tick queries
//! contacts with small per-collider speculative margins, then runs fixed substeps:
//! integrate forces, warm start, a biased projected Gauss–Seidel pass over drive,
//! joint-limit and contact rows, a continuous sweep of fast colliders, integrate
//! positions, and an unbiased relaxing pass. Contacts are queried again when a
//! collider outruns its margin. A restitution pass follows the substeps. Quality is
//! reported in diagnostics; only invalid input returns an error, and numerical
//! trouble marks the tick degraded.

mod solve;
#[cfg(test)]
mod tests;

use std::{collections::BTreeMap, time::Instant};

use bevy_math::DVec3;
use mechanic_core::{CompiledCreation, CoordinateDrive, TICK_SECONDS};

use crate::{
    BodyPose, CpuSnapshot, DriveCommand, DynamicsFactorization, ExternalImpulse,
    MachineCollisionGeometry, MachineKinematics, MachineMotion, MachineState, PhysicsError,
    TerrainContactFeature, TerrainContactScene,
    free_motion::apply_external_impulses,
    joint_forces::{PassiveForce, validate_drive},
    joint_machine::bounds,
};
use solve::{Contact, JointImpulses, Machine, closure_errors};

/// Offset separating submerged-vertex features from clipped manifold corners,
/// so their warm-start impulses never share a key.
const SUBMERGED_CORNERS: usize = 1 << 20;

/// Fixed solver parameters. Defaults suit 60 Hz game ticks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftStepConfig {
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
    /// Whether fast substeps are swept for collisions the contact margins miss.
    pub continuous: bool,
    /// Collider travel within one substep above which it is swept, in metres.
    pub continuous_travel: f64,
    /// Gap at which a swept collider counts as arriving, in metres.
    pub continuous_tolerance: f64,
    /// Conservative-advancement steps per swept triangle or collider pair.
    pub continuous_evaluations: usize,
    /// Penetration at the end of a swept substep left to the contact rows, in
    /// metres; deeper or missed arrivals cut the substep short.
    pub continuous_depth: f64,
    /// Body rotation since the last contact query, in radians, after which
    /// contacts are queried again. Anchors are fixed on the body, so a turning
    /// body's queried points move away from the surface they measure. A
    /// cylinder's terrain contacts stay put as it turns about its own axis, so
    /// that turn doesn't count against terrain.
    pub requery_angle: f64,
    /// Biased and relaxing passes for substeps holding a contact that arrived
    /// faster than `continuous_travel` per substep.
    pub impact_iterations: u32,
    /// Stiffness of loop-closing bearings in hertz, capped at a quarter of the
    /// substep rate.
    pub joint_hertz: f64,
    /// Damping ratio of loop-closing bearings.
    pub joint_damping_ratio: f64,
}

impl Default for SoftStepConfig {
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
            continuous: true,
            continuous_travel: 0.05,
            continuous_tolerance: 1e-3,
            continuous_evaluations: 32,
            continuous_depth: 0.01,
            requery_angle: 0.25,
            impact_iterations: 8,
            joint_hertz: 60.0,
            joint_damping_ratio: 2.0,
        }
    }
}

impl SoftStepConfig {
    fn is_valid(&self) -> bool {
        let nonnegative = [
            self.contact_hertz,
            self.damping_ratio,
            self.push_out,
            self.slop,
            self.speculative,
            self.restitution_threshold,
            self.stiction_speed,
            self.continuous_travel,
            self.continuous_depth,
            self.joint_hertz,
            self.joint_damping_ratio,
        ];
        (1..=64).contains(&self.substeps)
            && self.iterations > 0
            && nonnegative
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
            && self.maximum_speed.is_finite()
            && self.maximum_speed > 0.0
            && self.continuous_tolerance.is_finite()
            && self.continuous_tolerance > 0.0
            && self.continuous_evaluations > 0
            && self.requery_angle.is_finite()
            && self.requery_angle > 0.0
            && self.impact_iterations > 0
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
    /// Whether the tick fell back to an earlier state. Its contact loads then
    /// describe motion that never published. A clamped runaway body degrades
    /// the tick without rolling it back, and everything else's loads stand.
    pub rolled_back: bool,
    /// Contact query time, in milliseconds.
    pub query_ms: f64,
    /// Integration and solve time, in milliseconds.
    pub solve_ms: f64,
    /// Kinematics, factorization and free-force response time.
    pub dynamics_ms: f64,
    /// Contact and joint constraint row preparation time.
    pub rows_ms: f64,
    /// Iterative constraint solve and warm-start time.
    pub constraints_ms: f64,
    /// Continuous collision query time.
    pub continuous_ms: f64,
    /// Continuous collision shape transformations.
    pub continuous_shape_transformations: usize,
    /// Continuous collision shape cache hits.
    pub continuous_shape_cache_hits: usize,
    /// Continuous collision hierarchy node pair tests.
    pub continuous_hierarchy_node_pair_tests: usize,
    /// Continuous collision pose evaluations.
    pub continuous_pose_evaluations: usize,
    /// Continuous collision velocity evaluations.
    pub continuous_velocity_evaluations: usize,
    /// Continuous collision separation evaluations.
    pub continuous_separation_evaluations: usize,
    /// Continuous collision collider pair candidates.
    pub continuous_collider_pair_candidates: usize,
    /// Continuous collision triangle candidates.
    pub continuous_triangle_candidates: usize,
    /// Finite terrain candidates across proximity and recovery queries.
    pub triangle_candidates: usize,
    /// Collider candidates across proximity and recovery queries.
    pub collider_pair_candidates: usize,
    /// Retained contact-row and articulated-factor arena capacity (not total allocation).
    pub solver_scratch_bytes: usize,
    /// Net growth of those retained arenas during this tick, in bytes.
    pub solver_scratch_growth_bytes: usize,
    /// Signed drive impulses summed over the tick, in N·s or N·m·s.
    pub drive_impulses: Vec<f64>,
    /// Contact queries repeated within the tick because a collider outran its
    /// margin or a continuous hit cut a substep short.
    pub requeries: usize,
    /// Empty contact queries reused inside a conservatively cleared region.
    pub empty_contact_reuses: usize,
    /// Contact ownership groups refreshed during this tick.
    pub refreshed_contact_groups: usize,
    /// Contact ownership groups retained across substeps.
    pub reused_contact_groups: usize,
    /// Detailed collider paths prepared after coarse rejection.
    pub detailed_sweep_preparations: usize,
    /// Initial finite supports reused at unchanged queried poses.
    pub continuous_cached_supports: usize,
    /// Tick-local empty-region proofs that failed validation.
    pub clearance_certificate_failures: usize,
    /// Substeps swept for collisions the contact margins could miss.
    pub continuous_sweeps: usize,
    /// Swept substeps cut short before a collision the contacts would miss.
    pub continuous_hits: usize,
    /// Bearings closing mechanism loops, solved as soft rows.
    pub closures: usize,
    /// Largest distance a loop-closing bearing has pulled apart after the tick,
    /// in metres, across its constrained directions.
    pub closure_position_error: f64,
    /// Largest misalignment of a loop-closing bearing after the tick, in radians.
    pub closure_angle_error: f64,
}

impl SoftStepDiagnostics {
    fn degrade(&mut self, reason: &'static str) {
        self.degraded = true;
        self.degraded_reason.get_or_insert(reason);
    }
}

/// One terrain manifold's load integrated over the last accepted simulation tick.
#[derive(Clone, Copy, Debug)]
pub struct TerrainLoad {
    /// Source body in the published simulation topology.
    pub body: usize,
    /// Footprint centre relative to the terrain query's floating origin.
    pub point: DVec3,
    /// Outward unit terrain normal.
    pub normal: DVec3,
    /// Normal impulse summed across the tick's accepted substeps, in N s.
    pub normal_impulse: f64,
    /// Integrated world-space tangential impulse in N s.
    pub tangent_impulse: DVec3,
    /// Physical work dissipated by accepted substeps, in joules.
    pub work_j: f64,
    /// Velocity of the body's surface over the ground, averaged by the work it
    /// did there, in m/s. Zero for a load that did no work.
    pub slip_velocity: DVec3,
    /// Integrated resultant of the normal and tangential load, in N s.
    pub footprint_impulse: f64,
    /// Ground the manifold presses on.
    pub footprint: mechanic_world::LoadFootprint,
}

impl TerrainLoad {
    /// Pressure this load put on the ground over one tick, for compaction.
    /// `origin` is the terrain query's floating origin.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "world density and pressure use f32"
    )]
    pub fn soil_patch(&self, origin: DVec3) -> mechanic_world::SoilPatch {
        mechanic_world::SoilPatch {
            centre: mechanic_world::WorldPosition(origin + self.point),
            normal: self.normal,
            footprint: self.footprint,
            pressure_pa: (self.normal_impulse / (TICK_SECONDS * self.footprint.area())) as f32,
            seconds: TICK_SECONDS as f32,
        }
    }

    /// Stress, work and sideways crushing this load delivered over one tick,
    /// for breakage. `origin` is the terrain query's floating origin.
    pub fn breakage_patch(&self, origin: DVec3) -> mechanic_world::BreakagePatch {
        let area = TICK_SECONDS * self.footprint.area();
        let sideways = (1.0 - self.normal.y * self.normal.y).max(0.0).sqrt();
        mechanic_world::BreakagePatch {
            centre: mechanic_world::WorldPosition(origin + self.point),
            normal: self.normal,
            footprint: self.footprint,
            stress_pa: self.footprint_impulse / area,
            work_j: self.work_j,
            crush_pa: self.normal_impulse * sideways / area,
            throw: self.slip_velocity,
            seconds: TICK_SECONDS,
        }
    }
}

/// CPU machine stepped by the soft-step solver. Tree joints are exact by
/// reconstruction; contacts, drives, joint limits and loop-closing bearings are
/// soft rows.
pub struct CpuMachine {
    creation: CompiledCreation,
    passive: Vec<PassiveForce>,
    drives: Vec<CoordinateDrive>,
    completed: CpuSnapshot,
    candidate: MachineState,
    rollback: MachineState,
    scratch: solve::Scratch,
    contact_groups: Option<crate::terrain_contacts::ContactGroups>,
    diagnostics: SoftStepDiagnostics,
    warm: BTreeMap<TerrainContactFeature, [f64; 5]>,
    terrain_loads: Vec<TerrainLoad>,
    supported: Vec<bool>,
    load_features: Vec<(LoadKey, usize)>,
    load_order: Vec<usize>,
    footprint_points: Vec<DVec3>,
    failing_colliders: Vec<usize>,
    /// Suspension laws of loop-closing bearings, in `dynamics.loops` order.
    closure_passive: Vec<PassiveForce>,
    /// Loop-closure impulses carried into the next tick.
    closure_warm: Vec<[f64; 8]>,
    /// Bodies held at their published pose, by body.
    held: Vec<bool>,
    /// Generalized velocity rows owned by held bodies.
    held_rows: Vec<bool>,
}

impl CpuMachine {
    /// Loads a compiled creation. Joint coordinates outside their travel are
    /// clamped into it.
    ///
    /// # Errors
    /// Rejects mismatched state or drive rows, invalid drives, and non-finite or
    /// non-positive dynamics.
    pub fn new(
        creation: CompiledCreation,
        topology_generation: u64,
        mut state: MachineState,
    ) -> Result<Self, PhysicsError> {
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
        let model = MachineKinematics::assemble(&creation, &state.poses, &state.coordinates)?;
        model.body_motions(&state.velocities)?;
        state.poses = model.poses;
        let passive = creation
            .dynamics
            .coordinate_bearings
            .iter()
            .map(|&row| PassiveForce::from_kind(creation.bearings[row].kind))
            .collect();
        let closure_passive = creation
            .dynamics
            .loops
            .iter()
            .map(|pattern| PassiveForce::from_kind(creation.bearings[pattern.bearing].kind))
            .collect::<Vec<_>>();
        Ok(Self {
            supported: vec![false; creation.compounds.len()],
            held: vec![false; creation.compounds.len()],
            held_rows: vec![false; state.velocities.len()],
            closure_warm: vec![[0.0; 8]; closure_passive.len()],
            closure_passive,
            creation,
            passive,
            drives,
            candidate: state.clone(),
            rollback: state.clone(),
            scratch: solve::Scratch::default(),
            contact_groups: None,
            completed: CpuSnapshot {
                tick: 0,
                topology_generation,
                state,
            },
            diagnostics: SoftStepDiagnostics::default(),
            warm: BTreeMap::new(),
            terrain_loads: Vec::new(),
            load_features: Vec::new(),
            footprint_points: Vec::new(),
            failing_colliders: Vec::new(),
            load_order: Vec::new(),
        })
    }

    /// Terrain loads for the last accepted tick; empty after a degraded tick.
    pub fn terrain_loads(&self) -> &[TerrainLoad] {
        &self.terrain_loads
    }

    /// Whether the last accepted tick loaded an upward support contact for a
    /// body, including support from another body rather than terrain.
    pub fn body_supported(&self, body: usize) -> bool {
        self.supported.get(body).copied().unwrap_or(false)
    }

    /// Last published state.
    pub fn snapshot(&self) -> &CpuSnapshot {
        &self.completed
    }

    /// Work and quality of the last tick.
    pub fn diagnostics(&self) -> &SoftStepDiagnostics {
        &self.diagnostics
    }

    /// Holds whole mechanisms at prescribed poses. Held bodies stay at rest with
    /// their joints at zero, and other bodies collide with them as immovable.
    /// The mask replaces the previous one; a released body starts at rest from
    /// its last held pose. Poses of bodies not held are ignored.
    ///
    /// # Errors
    /// Rejects a wrong row count, a mask splitting one mechanism, or a held pose
    /// that is not finite. Invalid input changes nothing.
    pub fn hold(&mut self, held: &[bool], poses: &[BodyPose]) -> Result<(), PhysicsError> {
        let bodies = self.creation.compounds.len();
        if held.len() != bodies || poses.len() != bodies {
            return Err(PhysicsError::InvalidCommand);
        }
        let mut components = BTreeMap::new();
        for (parent, &holding) in self.creation.loop_topology.body_parents.iter().zip(held) {
            if components
                .insert(parent.component_index, holding)
                .is_some_and(|prior| prior != holding)
            {
                return Err(PhysicsError::InvalidCommand);
            }
        }
        if poses.iter().zip(held).any(|(pose, &holding)| {
            holding
                && !(pose.position.is_finite()
                    && pose.rotation.is_finite()
                    && pose.rotation.length_squared() > 0.0)
        }) {
            return Err(PhysicsError::InvalidCommand);
        }

        let dynamics = &self.creation.dynamics;
        let state = &mut self.completed.state;
        let mut rows = vec![false; state.velocities.len()];
        for (body, (&holding, &was)) in held.iter().zip(&self.held).enumerate() {
            if holding {
                state.poses[body] = BodyPose {
                    position: poses[body].position,
                    rotation: poses[body].rotation.normalize(),
                };
            }
            if holding || was {
                for row in dynamics.body_velocities[body].clone() {
                    rows[row] = holding;
                    state.velocities[row] = 0.0;
                }
            }
        }
        for (coordinate, &bearing) in dynamics.coordinate_bearings.iter().enumerate() {
            if held[self.creation.bearings[bearing].compound_a as usize] {
                state.coordinates[coordinate] = 0.0;
            }
        }
        if held != self.held.as_slice() {
            self.closure_warm.fill([0.0; 8]);
        }
        self.held = held.to_vec();
        self.held_rows = rows;
        Ok(())
    }

    // Held bodies keep their prescribed pose and joint coordinates, at rest.
    fn pin(&self, state: &mut MachineState) {
        if !self.held.contains(&true) {
            return;
        }
        let prescribed = &self.completed.state;
        for (body, _) in self.held.iter().enumerate().filter(|(_, held)| **held) {
            state.poses[body] = prescribed.poses[body];
        }
        for (coordinate, &bearing) in self
            .creation
            .dynamics
            .coordinate_bearings
            .iter()
            .enumerate()
        {
            if self.held[self.creation.bearings[bearing].compound_a as usize] {
                state.coordinates[coordinate] = prescribed.coordinates[coordinate];
            }
        }
        for (velocity, _) in state
            .velocities
            .iter_mut()
            .zip(&self.held_rows)
            .filter(|(_, held)| **held)
        {
            *velocity = 0.0;
        }
    }

    /// Advances and publishes one external tick.
    ///
    /// # Errors
    /// Only invalid input: a wrong tick or generation, duplicate or invalid drive
    /// commands, invalid impulses or settings, or terrain geometry compiled for
    /// another topology. Invalid input changes nothing.
    #[expect(
        clippy::too_many_lines,
        reason = "validation, the substep loop and the sole publication stay together"
    )]
    pub fn step(
        &mut self,
        gravity: DVec3,
        settings: &SoftStepConfig,
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

        self.terrain_loads.clear();
        self.load_features.clear();
        // Everything below publishes, degraded if necessary.
        self.drives = drives;
        let started = Instant::now();
        let scratch_before = self.scratch.retained_bytes();
        let mut diagnostics = SoftStepDiagnostics {
            drive_impulses: vec![0.0; self.drives.len()],
            ..SoftStepDiagnostics::default()
        };
        let mut state = std::mem::replace(
            &mut self.candidate,
            MachineState {
                poses: Vec::new(),
                coordinates: Vec::new(),
                velocities: Vec::new(),
            },
        );
        state.clone_from(&self.completed.state);
        if apply_external_impulses(&self.creation, &mut state, impulses, settings.factorization)
            .is_err()
        {
            diagnostics.degrade("external impulse");
        }
        self.pin(&mut state);
        let (mut contacts, margins) = terrain.map_or_else(
            || (Vec::new(), Vec::new()),
            |terrain| {
                self.contacts(
                    terrain,
                    &state,
                    settings,
                    gravity,
                    &self.warm,
                    None,
                    &mut diagnostics,
                )
            },
        );
        assign_footprints(
            &mut contacts,
            &mut self.load_order,
            &mut self.footprint_points,
        );
        diagnostics.contacts = contacts.len();

        let machine = Machine {
            creation: &self.creation,
            passive: &self.passive,
            drives: &self.drives,
            closure_passive: &self.closure_passive,
            held: &self.held_rows,
        };
        let mut joints = JointImpulses::new(self.drives.len(), self.closure_warm.clone());
        let dt = TICK_SECONDS / f64::from(settings.substeps);
        let mut last = None;
        let group_count = terrain.map_or(0, |t| t.geometry.assembly_count);
        let mut groups = if let Some(mut groups) = self.contact_groups.take() {
            if let Some(terrain) = terrain {
                groups.reset_moving(terrain.geometry, &margins, settings.speculative);
            } else {
                groups.reset(0, &margins, settings.speculative);
            }
            groups
        } else {
            crate::terrain_contacts::ContactGroups::new(group_count, &margins, settings.speculative)
        };
        let mut regions = (0..terrain.map_or(0, |t| t.geometry.assembly_count))
            .map(|_| None)
            .collect::<Vec<_>>();
        let mut tried_regions = vec![false; regions.len()];
        diagnostics.refreshed_contact_groups = groups.len();
        for substep in 0..settings.substeps {
            if let Some(terrain) = terrain.filter(|_| groups.invalid_count() > 0) {
                match MachineKinematics::reconstruct_poses(
                    &self.creation,
                    &state.poses,
                    &state.coordinates,
                ) {
                    Ok(poses) => {
                        state.poses = poses;
                        if (0..regions.len()).any(|assembly| {
                            groups.needs_refresh(assembly)
                                && (regions[assembly].is_some() || !tried_regions[assembly])
                                && !contacts.iter().any(|c| {
                                    terrain.geometry.assemblies[c.source.body] == assembly
                                        || c.source.other_body.is_some_and(|b| {
                                            terrain.geometry.assemblies[b] == assembly
                                        })
                                })
                        }) {
                            let clearance_started = Instant::now();
                            let displacement = state
                                .velocities
                                .iter()
                                .map(|v| v * TICK_SECONDS)
                                .collect::<Vec<_>>();
                            let zero = vec![0.0; displacement.len()];
                            if let (Ok(predicted), Ok(current)) = (
                                MachineMotion::new(
                                    &self.creation,
                                    terrain.topology_generation,
                                    &state,
                                    &displacement,
                                ),
                                MachineMotion::new(
                                    &self.creation,
                                    terrain.topology_generation,
                                    &state,
                                    &zero,
                                ),
                            ) {
                                let padding =
                                    margins.iter().copied().fold(settings.speculative, f64::max);
                                for assembly in 0..regions.len() {
                                    if !groups.needs_refresh(assembly) {
                                        continue;
                                    }
                                    if contacts.iter().any(|c| {
                                        terrain.geometry.assemblies[c.source.body] == assembly
                                            || c.source.other_body.is_some_and(|b| {
                                                terrain.geometry.assemblies[b] == assembly
                                            })
                                    }) {
                                        continue;
                                    }
                                    if !tried_regions[assembly] {
                                        tried_regions[assembly] = true;
                                        regions[assembly] = terrain.scene.assembly_clearance(
                                            terrain.geometry,
                                            &predicted,
                                            terrain.origin,
                                            padding + settings.continuous_travel,
                                            assembly,
                                        );
                                    }
                                    if let Some(region) = &regions[assembly] {
                                        if region.contains(
                                            terrain.scene,
                                            terrain.geometry,
                                            &current,
                                            terrain.origin,
                                            padding,
                                        ) {
                                            let reused = groups.reuse_assembly(assembly);
                                            diagnostics.empty_contact_reuses +=
                                                usize::from(reused > 0);
                                        } else {
                                            diagnostics.clearance_certificate_failures += 1;
                                            regions[assembly] = None;
                                        }
                                    }
                                }
                            }
                            diagnostics.query_ms +=
                                clearance_started.elapsed().as_secs_f64() * 1000.0;
                        }
                        let invalid = groups.invalid_count();
                        diagnostics.reused_contact_groups += groups.len() - invalid;
                        diagnostics.refreshed_contact_groups += invalid;
                        if invalid > 0 {
                            diagnostics.requeries += 1;
                            let warm = contacts
                                .iter()
                                .map(|contact| (contact.source.feature, contact.impulses))
                                .collect();
                            let (refreshed, new_margins) = self.contacts(
                                terrain,
                                &state,
                                settings,
                                gravity,
                                &warm,
                                Some((&groups, &margins)),
                                &mut diagnostics,
                            );
                            self.failing_colliders.clear();
                            self.failing_colliders.extend(
                                contacts
                                    .iter()
                                    .filter(|contact| contact.is_failing())
                                    .map(|contact| contact.source.feature.collider),
                            );
                            self.failing_colliders.sort_unstable();
                            contacts.retain(|contact| {
                                !groups.includes(
                                    terrain.geometry,
                                    contact.source.body,
                                    contact.source.other_body,
                                )
                            });
                            let failing = &self.failing_colliders;
                            contacts.extend(refreshed.into_iter().map(|mut contact| {
                                if failing
                                    .binary_search(&contact.source.feature.collider)
                                    .is_ok()
                                {
                                    contact.inherit_failing();
                                }
                                contact
                            }));
                            contacts.sort_by_key(|contact| {
                                (
                                    contact.source.feature.corner >= SUBMERGED_CORNERS,
                                    contact.source.feature,
                                )
                            });
                            assign_footprints(
                                &mut contacts,
                                &mut self.load_order,
                                &mut self.footprint_points,
                            );
                            groups.refreshed(
                                terrain.geometry,
                                &margins,
                                &new_margins,
                                settings.speculative,
                            );
                            diagnostics.contacts = diagnostics.contacts.max(contacts.len());
                        }
                    }
                    Err(_) => diagnostics.degrade("contact query"),
                }
            } else {
                diagnostics.reused_contact_groups += groups.len();
            }
            self.rollback.clone_from(&state);
            let result = solve::substep(
                &machine,
                &mut state,
                &mut contacts,
                &mut joints,
                gravity,
                dt,
                settings,
                terrain,
                &groups,
                &mut diagnostics,
                &mut self.scratch,
            );
            match result {
                Ok(outcome) if sane(&mut state, settings, &mut diagnostics) => {
                    if outcome.rewound {
                        for region in &mut regions {
                            *region = None;
                        }
                        tried_regions.fill(true);
                    }
                    if let Some(terrain) = terrain {
                        groups.advance_measured(
                            terrain.geometry,
                            &outcome.motion,
                            settings.requery_angle,
                            outcome.rewound,
                        );
                    }
                    // The last substep also carries the tick's restitution impulse.
                    if substep + 1 < settings.substeps {
                        collect_terrain_loads(
                            &contacts,
                            &self.held,
                            &mut self.terrain_loads,
                            &mut self.load_features,
                            &mut self.load_order,
                        );
                    }
                    last = Some(outcome.point_count);
                }
                Ok(_) | Err(_) => {
                    state.clone_from(&self.rollback);
                    state.velocities.fill(0.0);
                    last = None;
                    diagnostics.degrade("numerical substep");
                    diagnostics.rolled_back = true;
                    self.pin(&mut state);
                    break;
                }
            }
            self.pin(&mut state);
        }
        if let Some(count) = last {
            diagnostics.velocity_error = solve::restitution(
                &mut contacts,
                &self.scratch.points[..count],
                &mut state.velocities,
                settings,
            );
            collect_terrain_loads(
                &contacts,
                &self.held,
                &mut self.terrain_loads,
                &mut self.load_features,
                &mut self.load_order,
            );
        }
        match MachineKinematics::reconstruct_poses(&self.creation, &state.poses, &state.coordinates)
        {
            Ok(poses) if sane(&mut state, settings, &mut diagnostics) => state.poses = poses,
            Ok(_) | Err(_) => {
                state.clone_from(&self.completed.state);
                state.velocities.fill(0.0);
                diagnostics.degrade("final pose");
                diagnostics.rolled_back = true;
            }
        }
        self.pin(&mut state);
        diagnostics.closures = self.closure_warm.len();
        (
            diagnostics.closure_position_error,
            diagnostics.closure_angle_error,
        ) = closure_errors(&self.creation, &state.poses);
        // Impulses from a rewound substep describe a state that never published.
        self.closure_warm = if diagnostics.degraded {
            vec![[0.0; 8]; self.closure_warm.len()]
        } else {
            joints.closures
        };

        if diagnostics.rolled_back {
            self.terrain_loads.clear();
        }
        self.supported.fill(false);
        if !diagnostics.rolled_back {
            for contact in &contacts {
                if contact.impulses[0] <= 0.0 {
                    continue;
                }
                if contact.source.normal.y >= mechanic_world::GROUND_NORMAL_MIN_Y {
                    self.supported[contact.source.body] = true;
                }
                if contact.source.normal.y <= -mechanic_world::GROUND_NORMAL_MIN_Y
                    && let Some(body) = contact.source.other_body
                {
                    self.supported[body] = true;
                }
            }
        }
        for load in &mut self.terrain_loads {
            load.work_j = load.work_j.max(0.0);
        }
        self.warm = contacts
            .iter()
            .map(|contact| (contact.source.feature, contact.impulses))
            .collect();
        diagnostics.solve_ms = started.elapsed().as_secs_f64() * 1000.0 - diagnostics.query_ms;
        diagnostics.solver_scratch_bytes = self.scratch.retained_bytes();
        diagnostics.solver_scratch_growth_bytes = diagnostics
            .solver_scratch_bytes
            .saturating_sub(scratch_before);
        self.contact_groups = Some(groups);
        self.diagnostics = diagnostics;
        self.candidate = std::mem::replace(&mut self.completed.state, state);
        self.completed.tick = tick;
        self.completed.topology_generation = generation;
        Ok(&self.completed)
    }

    // Contacts at the current pose, and the margin each collider row was queried
    // with.
    #[expect(
        clippy::too_many_arguments,
        reason = "query context and per-group selection share one fallback policy"
    )]
    fn contacts(
        &self,
        terrain: SoftStepTerrain<'_>,
        state: &MachineState,
        settings: &SoftStepConfig,
        gravity: DVec3,
        warm: &BTreeMap<TerrainContactFeature, [f64; 5]>,
        selection: Option<(&crate::terrain_contacts::ContactGroups, &[f64])>,
        diagnostics: &mut SoftStepDiagnostics,
    ) -> (Vec<Contact>, Vec<f64>) {
        // A query serves until a collider outruns its margin. Each collider
        // reaches as far as its body carries it in a tick, ancestor rotation and
        // suspension travel included, with a quarter more for speed gained within
        // the tick and the fall under gravity. The reach stops at the travel the
        // continuous sweep takes over from: a wide margin measures a tilted
        // collider's gap up its side faces, so most of a far manifold's points
        // would sit at the margin instead of on the corners that arrive.
        let groups = selection.map(|(groups, _)| groups);
        let query_started = Instant::now();
        let colliders = terrain.geometry.collider_reach().len();
        let fall = 0.5 * gravity.length() * TICK_SECONDS * TICK_SECONDS;
        let displacement = state
            .velocities
            .iter()
            .map(|velocity| 1.25 * TICK_SECONDS * velocity)
            .collect::<Vec<_>>();
        let reach = if let Some((_, margins)) = selection {
            margins.to_vec()
        } else if let Ok(motion) = MachineMotion::new(
            &self.creation,
            terrain.topology_generation,
            state,
            &displacement,
        ) {
            terrain
                .geometry
                .collider_reach()
                .map(|(body, radius)| {
                    settings.speculative
                        + fall
                        + motion.bounds()[body]
                            .point_speed(radius)
                            .min(settings.continuous_travel)
                })
                .collect()
        } else {
            diagnostics.degrade("contact reach");
            vec![settings.speculative; colliders]
        };
        // Clipping nearly parallel faces within a wide margin can refuse a query;
        // narrower queries still keep the bodies apart.
        let Some((query, margins)) = [
            reach,
            vec![settings.speculative; colliders],
            vec![settings.speculative * 0.5; colliders],
            vec![0.0; colliders],
        ]
        .into_iter()
        .find_map(|margins| {
            terrain
                .scene
                .proximity_groups(
                    terrain.geometry,
                    &state.poses,
                    terrain.origin,
                    &margins,
                    groups,
                )
                .ok()
                .map(|query| (query, margins))
        }) else {
            diagnostics.degrade("contact query");
            return (Vec::new(), vec![0.0; colliders]);
        };
        diagnostics.triangle_candidates += query.triangle_candidates;
        diagnostics.collider_pair_candidates += query.collider_pair_candidates;
        let mut points = query.contacts;
        // A clipped manifold only holds points where the collider crosses the
        // surface, all at zero gap; a tilted body's submerged corner lies inside
        // that polygon. Add those vertices with their real depth.
        if let Ok(recovery) =
            terrain
                .scene
                .buried_groups(terrain.geometry, &state.poses, terrain.origin, groups)
        {
            diagnostics.triangle_candidates += recovery.triangle_candidates;
            diagnostics.collider_pair_candidates += recovery.collider_pair_candidates;
            for mut vertex in recovery.contacts {
                if vertex.depth <= 0.0 {
                    continue;
                }
                // Deep in overlap, a clipped point reports the gap at its clipping
                // boundary rather than the overlap; the buried vertex beside it
                // carries the real depth and takes its place.
                if let Some(point) = points.iter_mut().find(|point| {
                    point.body == vertex.body
                        && point.body_point.distance(vertex.body_point) <= 1e-4
                }) {
                    if point.separation > vertex.separation {
                        vertex.feature = point.feature;
                        *point = vertex;
                    }
                } else {
                    vertex.feature.corner += SUBMERGED_CORNERS;
                    points.push(vertex);
                }
            }
        }
        points.sort_by_key(|point| (point.feature.corner >= SUBMERGED_CORNERS, point.feature));
        let contacts = points
            .into_iter()
            .map(|point| {
                Contact::new(
                    point,
                    &state.poses,
                    warm.get(&point.feature),
                    terrain.geometry,
                )
            })
            .collect();
        diagnostics.query_ms += query_started.elapsed().as_secs_f64() * 1000.0;
        (contacts, margins)
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
    settings: &SoftStepConfig,
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

// Body, collider and manifold: manifold numbers are local to each collider, so
// the collider is part of the key even when several colliders share a body.
type LoadKey = (usize, usize, usize);

fn load_key(source: &crate::TerrainContact) -> LoadKey {
    (source.body, source.feature.collider, source.manifold)
}

// Every terrain contact learns the ground its manifold presses on. Called
// whenever the contact list is rebuilt, before the solver needs the area.
fn assign_footprints(contacts: &mut [Contact], order: &mut Vec<usize>, points: &mut Vec<DVec3>) {
    order.clear();
    order.extend(
        contacts
            .iter()
            .enumerate()
            .filter(|(_, contact)| {
                matches!(
                    contact.source.feature.obstacle,
                    crate::ContactObstacle::Terrain { .. }
                )
            })
            .map(|(index, _)| index),
    );
    order.sort_unstable_by_key(|&index| load_key(&contacts[index].source));
    let mut start = 0;
    while start < order.len() {
        let key = load_key(&contacts[order[start]].source);
        let end = start
            + order[start..]
                .iter()
                .take_while(|&&index| load_key(&contacts[index].source) == key)
                .count();
        points.clear();
        points.extend(
            order[start..end]
                .iter()
                .map(|&index| contacts[index].source.terrain_point),
        );
        let (centre, footprint) =
            mechanic_world::LoadFootprint::spanning(points, contacts[order[start]].source.normal);
        for &index in &order[start..end] {
            contacts[index].footprint = solve::Footprint {
                centre,
                shape: footprint,
                points: end - start,
            };
        }
        start = end;
    }
}

// Reuse all three vectors.
fn collect_terrain_loads(
    contacts: &[Contact],
    held: &[bool],
    loads: &mut Vec<TerrainLoad>,
    keys: &mut Vec<(LoadKey, usize)>,
    order: &mut Vec<usize>,
) {
    order.clear();
    order.extend(
        contacts
            .iter()
            .enumerate()
            .filter(|(_, contact)| {
                matches!(
                    contact.source.feature.obstacle,
                    crate::ContactObstacle::Terrain { .. }
                ) && !held[contact.source.body]
            })
            .map(|(index, _)| index),
    );
    order.sort_unstable_by_key(|&index| load_key(&contacts[index].source));
    let mut start = 0;
    while start < order.len() {
        let first = &contacts[order[start]];
        let key = load_key(&first.source);
        let end = start
            + order[start..]
                .iter()
                .take_while(|&&index| load_key(&contacts[index].source) == key)
                .count();
        let group = &order[start..end];
        start = end;
        let mut normal_impulse = 0.0;
        let mut tangent_impulse = DVec3::ZERO;
        let mut work_j = 0.0;
        let mut slip = DVec3::ZERO;
        for &index in group {
            let contact = &contacts[index];
            if contact.impulses[0].is_finite() && contact.impulses[0] > 0.0 {
                normal_impulse += contact.impulses[0];
                tangent_impulse += contact.tangent_impulse;
                work_j += contact.work_j;
                slip += contact.slip_velocity * contact.work_j;
            }
        }
        if normal_impulse <= 0.0 {
            continue;
        }
        let footprint_impulse = normal_impulse.hypot(tangent_impulse.length());
        match keys.binary_search_by_key(&key, |&(key, _)| key) {
            Ok(index) => {
                let load = &mut loads[keys[index].1];
                load.normal_impulse += normal_impulse;
                load.tangent_impulse += tangent_impulse;
                let total = load.work_j + work_j;
                if total > 0.0 {
                    load.slip_velocity = (load.slip_velocity * load.work_j + slip) / total;
                }
                load.work_j = total;
                load.footprint_impulse += footprint_impulse;
            }
            Err(index) => {
                keys.insert(index, (key, loads.len()));
                loads.push(TerrainLoad {
                    body: first.source.body,
                    point: first.footprint.centre,
                    normal: first.source.normal,
                    normal_impulse,
                    tangent_impulse,
                    work_j,
                    slip_velocity: if work_j > 0.0 {
                        slip / work_j
                    } else {
                        DVec3::ZERO
                    },
                    footprint_impulse,
                    footprint: first.footprint.shape,
                });
            }
        }
    }
}
