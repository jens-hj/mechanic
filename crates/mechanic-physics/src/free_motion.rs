//! Transactional, collision-free reference integration for passive tree machines.

use bevy_math::{DQuat, DVec3};
use mechanic_core::{CompiledCreation, CoordinateDrive, JointKind, TICK_SECONDS};

use crate::{BodyPose, MachineDynamics, PhysicsError};

/// Reduced-coordinate state. Root velocities are world-space origin linear
/// velocity then angular velocity; each tree joint owns one coordinate rate.
#[derive(Clone, Debug, PartialEq)]
pub struct MachineState {
    /// Body-indexed poses; only canonical roots are inputs to reconstruction.
    pub poses: Vec<BodyPose>,
    /// Joint positions, in compiled coordinate order (radians or metres).
    pub coordinates: Vec<f64>,
    /// Generalized velocities in compiled dynamics order.
    pub velocities: Vec<f64>,
}

impl MachineState {
    /// Authored pose with zero joint displacement and velocity.
    pub fn at_rest(creation: &CompiledCreation) -> Self {
        Self {
            poses: MachineDynamics::initial_roots(creation),
            coordinates: vec![0.0; creation.dynamics.coordinate_bearings.len()],
            velocities: vec![0.0; creation.dynamics.elimination_parent.len()],
        }
    }
}

/// One external impulse, applied exactly once at the beginning of its tick.
#[derive(Clone, Copy, Debug)]
pub struct ExternalImpulse {
    /// Completed tick this command should produce (first tick is one).
    pub tick: u64,
    /// Topology generation identifying the body rows.
    pub topology_generation: u64,
    /// Compiled body receiving the impulse.
    pub body: usize,
    /// World-space application point at the beginning of the tick, in metres.
    pub point: DVec3,
    /// World-space impulse in newton-seconds.
    pub impulse: DVec3,
}

/// Completed reference state; never contains a partially integrated tick.
#[derive(Clone, Debug, PartialEq)]
pub struct CpuSnapshot {
    /// Last completed external tick, starting at zero.
    pub tick: u64,
    /// Generation of the immutable compiled topology.
    pub topology_generation: u64,
    /// Reconstructed body poses and reduced-coordinate motion.
    pub state: MachineState,
}

impl CpuSnapshot {
    /// Stable bitwise identity in explicit row/component order. Repeatability is
    /// required within a build/backend, not between different architectures.
    pub fn state_hash(&self) -> u64 {
        let words = [self.tick, self.topology_generation].into_iter().chain(
            self.state
                .poses
                .iter()
                .flat_map(|pose| {
                    pose.position
                        .to_array()
                        .into_iter()
                        .chain(pose.rotation.to_array())
                })
                .chain(self.state.coordinates.iter().copied())
                .chain(self.state.velocities.iter().copied())
                .map(f64::to_bits),
        );
        words
            .flat_map(u64::to_le_bytes)
            .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
            })
    }
}

/// Collision-free CPU reference for free bodies and passive, unbounded revolute
/// trees. This is a foundation for the world runtime, not an authoritative scene
/// backend: no terrain or body collision is performed. Unsupported physical
/// features are rejected instead of silently dropping their forces/constraints.
/// Dense RK4 integration deliberately provides a convergence reference; it does
/// not meet the eventual once-per-substep factorization or performance target.
pub struct CpuFreeMotion {
    creation: CompiledCreation,
    completed: CpuSnapshot,
}

impl CpuFreeMotion {
    /// Creates a reference machine and validates its initial numerical dynamics.
    ///
    /// # Errors
    /// Rejects loops, linear/suspension joints, drives/stops, invalid state, and
    /// scenes beyond the dense reference capacity. Collision geometry is retained
    /// in the creation but is deliberately not evaluated by this free-motion API.
    pub fn new(
        creation: CompiledCreation,
        topology_generation: u64,
        mut state: MachineState,
    ) -> Result<Self, PhysicsError> {
        if !creation.dynamics.loops.is_empty()
            || creation
                .bearings
                .iter()
                .any(|b| b.kind != JointKind::Rotational)
            || creation
                .coordinate_drives
                .iter()
                .any(|d| *d != CoordinateDrive::PASSIVE)
        {
            return Err(PhysicsError::UnsupportedFreeMotion);
        }
        let model = MachineDynamics::assemble(&creation, &state.poses, &state.coordinates)?;
        model.body_motions(&state.velocities)?;
        model.factor(&vec![0.0; state.velocities.len()])?;
        state.poses = model.poses;
        Ok(Self {
            creation,
            completed: CpuSnapshot {
                tick: 0,
                topology_generation,
                state,
            },
        })
    }

    /// Last fully completed state. Failed ticks leave this snapshot unchanged.
    pub fn snapshot(&self) -> &CpuSnapshot {
        &self.completed
    }

    /// Advances one 60 Hz tick using a fixed 1/2/4/8 subdivision policy. Commands
    /// are applied in supplied order before integration and are never replayed
    /// during internal stages. The entire candidate remains unpublished until
    /// final reconstruction and numerical validation succeed.
    ///
    /// # Errors
    /// Rejects a stale/wrong tick or topology, invalid impulse, invalid subdivision
    /// policy, and non-finite or singular numerical results. This reference only
    /// validates free-motion numerics; it cannot validate collision penetration.
    pub fn step(
        &mut self,
        gravity: DVec3,
        substeps: u32,
        commands: &[ExternalImpulse],
    ) -> Result<&CpuSnapshot, PhysicsError> {
        let tick = self
            .completed
            .tick
            .checked_add(1)
            .ok_or(PhysicsError::InvalidCommand)?;
        if !matches!(substeps, 1 | 2 | 4 | 8) || !gravity.is_finite() {
            return Err(PhysicsError::InvalidDynamics);
        }
        if commands.iter().any(|c| {
            c.tick != tick
                || c.topology_generation != self.completed.topology_generation
                || c.body >= self.creation.compounds.len()
                || !c.point.is_finite()
                || !c.impulse.is_finite()
        }) {
            return Err(PhysicsError::InvalidCommand);
        }
        let mut candidate = self.completed.state.clone();
        apply_external_impulses(
            &self.creation,
            &mut candidate,
            commands,
            crate::DynamicsFactorization::DenseReference,
        )?;
        let dt = TICK_SECONDS / f64::from(substeps);
        for _ in 0..substeps {
            candidate = rk4(&self.creation, &candidate, gravity, dt)?;
        }
        let model =
            MachineDynamics::assemble(&self.creation, &candidate.poses, &candidate.coordinates)?;
        model.body_motions(&candidate.velocities)?;
        model.factor(&vec![0.0; candidate.velocities.len()])?;
        candidate.poses = model.poses;
        self.completed = CpuSnapshot {
            tick,
            topology_generation: self.completed.topology_generation,
            state: candidate,
        };
        Ok(&self.completed)
    }
}

pub(crate) fn apply_external_impulses(
    creation: &CompiledCreation,
    candidate: &mut MachineState,
    commands: &[ExternalImpulse],
    factorization: crate::DynamicsFactorization,
) -> Result<(), PhysicsError> {
    if !commands.is_empty() {
        let model =
            crate::MachineKinematics::assemble(creation, &candidate.poses, &candidate.coordinates)?;
        let mut impulse = vec![0.0; candidate.velocities.len()];
        for command in commands {
            let row = model.point_row(command.body, command.point, command.impulse)?;
            for (sum, value) in impulse.iter_mut().zip(row) {
                *sum += value;
            }
        }
        model
            .factor(
                factorization,
                &candidate.coordinates,
                &vec![0.0; impulse.len()],
            )?
            .solve(&mut impulse)?;
        for (v, delta) in candidate.velocities.iter_mut().zip(impulse) {
            *v += delta;
        }
    }
    Ok(())
}

pub(crate) fn advance_positions(creation: &CompiledCreation, state: &mut MachineState, dt: f64) {
    for &body in &creation.dynamics.preorder {
        if !creation.loop_topology.body_parents[body].is_root {
            continue;
        }
        let rows = creation.dynamics.body_velocities[body].clone();
        if rows.is_empty() {
            continue;
        }
        let v = &state.velocities[rows];
        state.poses[body].position += DVec3::new(v[0], v[1], v[2]) * dt;
        state.poses[body].rotation = (DQuat::from_scaled_axis(DVec3::new(v[3], v[4], v[5]) * dt)
            * state.poses[body].rotation)
            .normalize();
    }
    for (coordinate, &velocity) in creation.dynamics.coordinate_velocities.iter().enumerate() {
        state.coordinates[coordinate] += dt * state.velocities[velocity];
    }
}

// Derivatives share the state shape; quaternion entries are derivatives, not poses.
fn derivative(
    creation: &CompiledCreation,
    state: &MachineState,
    gravity: DVec3,
) -> Result<MachineState, PhysicsError> {
    let model = MachineDynamics::assemble(creation, &state.poses, &state.coordinates)?;
    let mut acceleration = model.gravity_force(creation, gravity)?;
    let bias = model.inertial_bias(creation, &state.velocities)?;
    for (value, bias) in acceleration.iter_mut().zip(bias) {
        *value -= bias;
    }
    model
        .factor(&vec![0.0; acceleration.len()])?
        .solve(&mut acceleration)?;
    let mut rate = MachineState {
        poses: vec![
            BodyPose {
                position: DVec3::ZERO,
                rotation: DQuat::from_xyzw(0.0, 0.0, 0.0, 0.0)
            };
            state.poses.len()
        ],
        coordinates: vec![0.0; state.coordinates.len()],
        velocities: acceleration,
    };
    for &body in &creation.dynamics.preorder {
        let rows = creation.dynamics.body_velocities[body].clone();
        if rows.is_empty() {
            continue;
        }
        if creation.loop_topology.body_parents[body].is_root {
            let v = &state.velocities[rows];
            rate.poses[body] = BodyPose {
                position: DVec3::new(v[0], v[1], v[2]),
                rotation: (DQuat::from_xyzw(v[3], v[4], v[5], 0.0) * state.poses[body].rotation)
                    * 0.5,
            };
        } else {
            let bearing =
                creation.dynamics.body_bearings[body].ok_or(PhysicsError::InvalidDynamics)?;
            let coordinate = creation.bearings[bearing]
                .coordinate_index
                .ok_or(PhysicsError::InvalidDynamics)? as usize;
            rate.coordinates[coordinate] = state.velocities[rows.start];
        }
    }
    Ok(rate)
}

fn shifted(state: &MachineState, rates: &[(&MachineState, f64)], dt: f64) -> MachineState {
    let mut result = state.clone();
    for (rate, weight) in rates {
        let scale = dt * weight;
        for (pose, rate) in result.poses.iter_mut().zip(&rate.poses) {
            pose.position += rate.position * scale;
            pose.rotation += rate.rotation * scale;
        }
        for (q, rate) in result.coordinates.iter_mut().zip(&rate.coordinates) {
            *q += scale * rate;
        }
        for (v, rate) in result.velocities.iter_mut().zip(&rate.velocities) {
            *v += scale * rate;
        }
    }
    for pose in &mut result.poses {
        pose.rotation = pose.rotation.normalize();
    }
    result
}

fn rk4(
    creation: &CompiledCreation,
    state: &MachineState,
    gravity: DVec3,
    dt: f64,
) -> Result<MachineState, PhysicsError> {
    let a = derivative(creation, state, gravity)?;
    let b = derivative(creation, &shifted(state, &[(&a, 0.5)], dt), gravity)?;
    let c = derivative(creation, &shifted(state, &[(&b, 0.5)], dt), gravity)?;
    let d = derivative(creation, &shifted(state, &[(&c, 1.0)], dt), gravity)?;
    Ok(shifted(
        state,
        &[
            (&a, 1.0 / 6.0),
            (&b, 1.0 / 3.0),
            (&c, 1.0 / 3.0),
            (&d, 1.0 / 6.0),
        ],
        dt,
    ))
}

#[cfg(test)]
mod tests;
