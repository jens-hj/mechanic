//! Loose material in the world: stepping it, breaking it out and laying it down.

use std::time::Duration;

use bevy::math::{DQuat, DVec3};
use bevy::prelude::*;
use mechanic_gpu::GpuExternalImpulse;
use mechanic_physics::{BodyPose, SpatialMotion, SpoilMachine};

use super::brush::{TerrainEditTaskResult, commit_terrain_edit_result};
use super::{WorldListPhase, WorldListState, WorldRuntime};
use crate::simulation::state::AppSimulation;

/// Spoil ticks one frame may run; a slower frame lets spoil fall behind rather
/// than slow further.
const MAX_TICKS_PER_FRAME: u32 = 3;
/// Least time between material transfers: the cadence soil compaction commits at.
const TRANSFER_INTERVAL: Duration = Duration::from_millis(100);

/// Steps loose material against the ground and the machine's last published
/// state, and hands the machine what the spoil did to it.
pub(super) fn step_spoil(
    mut runtime: ResMut<WorldRuntime>,
    simulation: Res<AppSimulation>,
    list: Res<WorldListState>,
    time: Res<Time>,
) {
    if list.phase() != WorldListPhase::Playing {
        return;
    }
    let runtime = &mut *runtime;
    runtime.spoil_machine = machine(&simulation, runtime.floating_origin.0);
    if runtime.clumps.bodies.is_empty() {
        runtime.spoil_seconds = 0.0;
        return;
    }
    let started = std::time::Instant::now();
    runtime.spoil_seconds = (runtime.spoil_seconds + time.delta_secs_f64())
        .min(f64::from(MAX_TICKS_PER_FRAME) * mechanic_core::TICK_SECONDS);
    let mut awake = 0;
    while runtime.spoil_seconds >= mechanic_core::TICK_SECONDS {
        runtime.spoil_seconds -= mechanic_core::TICK_SECONDS;
        let step = runtime.spoil.step(
            &mut runtime.clumps,
            &runtime.edits,
            &runtime.field,
            &runtime.spoil_machine,
            crate::cpu_physics::gravity(),
            mechanic_core::TICK_SECONDS,
        );
        awake = step.awake;
        if simulation.is_running() {
            for reaction in step.reactions {
                let Ok(body) = u32::try_from(reaction.body) else {
                    continue;
                };
                runtime
                    .pending_player_reactions
                    .push(GpuExternalImpulse::new(
                        body,
                        (reaction.point - runtime.floating_origin.0).as_vec3(),
                        reaction.impulse.as_vec3(),
                    ));
            }
        }
    }
    if awake > 0 {
        runtime.material_motion();
    }
    crate::performance_capture::record(
        "spoil_step",
        || serde_json::json!({"duration_ms": started.elapsed().as_secs_f64() * 1000.0, "clumps": runtime.clumps.bodies.len(), "awake": awake}),
    );
}

// The running creation as spoil meets it, or nothing when none runs.
fn machine(simulation: &AppSimulation, origin: DVec3) -> SpoilMachine {
    let (Some(creation), Some(live)) =
        (simulation.creation.as_ref(), simulation.live_state.as_ref())
    else {
        return SpoilMachine::default();
    };
    if !simulation.is_running() {
        return SpoilMachine::default();
    }
    let poses = live
        .transforms
        .iter()
        .map(|transform| BodyPose {
            position: Vec3::from_slice(&transform.position[..3]).as_dvec3(),
            rotation: DQuat::from_array(transform.rotation.map(f64::from)).normalize(),
        })
        .collect::<Vec<_>>();
    let motions = live
        .velocities
        .iter()
        .map(|velocity| SpatialMotion {
            linear: Vec3::from_slice(&velocity.linear[..3]).as_dvec3(),
            angular: Vec3::from_slice(&velocity.angular[..3]).as_dvec3(),
        })
        .collect::<Vec<_>>();
    SpoilMachine::new(creation, &poses, &motions, origin)
}

impl WorldRuntime {
    /// Breaks ready ground out into clumps and lays settled spoil back down, as
    /// one ordinary terrain edit. Nothing waits for it: spoil reads the voxels,
    /// and the mesh follows like after any other edit.
    pub(super) fn transfer_material(&mut self) {
        if self.terrain_edit_task.is_some()
            || self.terrain_edit_error.is_some()
            || self
                .last_material_transfer
                .is_some_and(|last| self.clock.saturating_sub(last) < TRANSFER_INTERVAL)
        {
            return;
        }
        let started = std::time::Instant::now();
        let mut terrain = self.edits.clone();
        let machine = &self.spoil_machine;
        let outcomes = self.clumps.transfer(
            &mut terrain,
            &self.field,
            &mut self.pending_breakage,
            &mut self.slump,
            &mut |cell| machine.keeps_clear(cell.centre().0),
            mechanic_world::TransferLimits::default(),
        );
        if outcomes.is_empty() {
            return;
        }
        self.last_material_transfer = Some(self.clock);
        crate::performance_capture::record(
            "material_transfer",
            || serde_json::json!({"duration_ms": started.elapsed().as_secs_f64() * 1000.0, "laid_cells": outcomes.iter().map(|outcome| outcome.laid_cells.len()).sum::<usize>(), "removed_cells": outcomes.iter().map(mechanic_world::TerrainEditOutcome::total_removed_cells).sum::<u64>(), "quanta_given_up": outcomes.iter().map(|outcome| outcome.quanta_given_up).sum::<u64>(), "quanta_taken_back": outcomes.iter().map(|outcome| outcome.quanta_taken_back).sum::<u64>(), "loose_quanta": self.clumps.bodies.values().map(|body| u64::from(body.quanta)).sum::<u64>(), "clumps": self.clumps.bodies.len()}),
        );
        commit_terrain_edit_result(
            self,
            TerrainEditTaskResult {
                terrain,
                outcomes,
                strokes: Vec::new(),
                elapsed_ms: 0.0,
            },
        );
    }
}
