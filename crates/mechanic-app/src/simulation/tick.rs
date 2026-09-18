//! Dispatching physics ticks and collecting their readbacks.

use crate::editor::hammer::{HammerInteraction, pending_hammer_impulse};
use crate::editor::preview::{BearingVisual, ConstructionVisual, EditorVisuals};
use crate::editor::state::EditorState;
use crate::hotbar::SelectedTool;
use crate::sequencer::{
    DriveSequencer, GearboxRuntime, geared_gpu_drive_rows, step_drive_programs,
};
use crate::simulation::publication::WorldPhysicsPublication;
use crate::simulation::state::{
    AppSimulation, LivePhysicsState, MAXIMUM_TICKS_PER_FRAME, next_simulation_ticks,
    stop_failed_simulation,
};
use crate::simulation::visuals::{refresh_published_construction_visuals, visual_snapshot_is_due};
use crate::{
    automation, cpu_physics, freeze, performance_capture, terrain_publication, weld_publication,
    world,
};
use bevy::prelude::{
    Assets, ButtonInput, Local, Mesh, Query, Real, Res, ResMut, Time, Visibility, Without, format,
    warn,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};

/// Polls completed physics snapshots without dispatching new work.
///
/// World walking runs after this system, so collision and rendered machinery consume
/// the same newest valid transform publication.
#[expect(
    clippy::too_many_lines,
    reason = "keep capture and validated publication in visible order"
)]
pub(crate) fn poll_simulation_readbacks(
    mut simulation: ResMut<AppSimulation>,
    frozen: Res<freeze::DimensionFreeze>,
    mut state: ResMut<EditorState>,
    render_device: Res<RenderDevice>,
) {
    if !simulation.is_running() || simulation.cpu.is_some() {
        // The CPU route publishes each tick as it completes it; there is no queue.
        return;
    }
    loop {
        let poll_started = performance_capture::is_recording().then(std::time::Instant::now);
        let completed = simulation
            .gpu
            .as_ref()
            .expect("running simulation has GPU state")
            .poll_tick_readback(render_device.wgpu_device());
        if let Some(start) = poll_started {
            performance_capture::record("physics_poll", || {
                serde_json::json!({
                    "duration_ms": start.elapsed().as_secs_f64()*1000.0,
                    "outcome": match &completed { Ok(Some(_)) => "completed", Ok(None) => "empty", Err(_) => "error" }
                })
            });
        }
        if let Ok(Some(completed)) = &completed {
            performance_capture::record("physics_readback", || {
                let stages = completed.diagnostics.kernel_timings;
                serde_json::json!({"tick":completed.tick_index, "sequence":completed.submission_sequence, "latency_ms":completed.submission_to_readback_ms, "submission_to_callbacks_ms":completed.submission_to_callbacks_ms, "callbacks_during_poll":completed.callbacks_during_poll, "gpu_tick_ms":completed.diagnostics.gpu_tick_ms, "error_flags":completed.diagnostics.error_flags,
                    "terrain_ms": stages.map(|s| s.terrain_traversal_ms),
                    "rotational_sweep_ms": stages.map(|s| s.rotational_sweep_ms),
                    "terrain_recovery_ms": stages.map(|s| s.terrain_recovery_ms),
                    "recovery_projection_ms": stages.map(|s| s.recovery_projection_ms),
                    "integration_ms": stages.map(|s| s.integration_ms),
                    "mechanism_ms": stages.map(|s| s.mechanism_ms),
                    "broadphase_ms": stages.map(|s| s.broadphase_ms),
                    "narrowphase_ms": stages.map(|s| s.narrowphase_ms),
                    "contact_solver_ms": stages.map(|s| s.contact_solver_ms),
                    "bearings_ms": stages.map(|s| s.bearings_ms),
                    "snapshot_ms": stages.map(|s| s.snapshot_ms),
                    "contacts": completed.diagnostics.contact_count,
                    "active_contacts": completed.diagnostics.active_contact_count,
                    "executed_stage_mask": completed.diagnostics.execution.stage_mask,
                    "integrated_bodies": completed.diagnostics.execution.integrated_bodies,
                    "published_bodies": completed.diagnostics.execution.published_bodies,
                    "validated_bearings": completed.diagnostics.execution.validated_bearings,
                    "planned_solver_sweeps": completed.diagnostics.planned_solver_sweeps,
                    "executed_solver_sweeps": completed.diagnostics.executed_solver_sweeps,
                    "anchor_residual_m": completed.diagnostics.anchor_residual_meters,
                    "axis_residual_deg": completed.diagnostics.axis_residual_degrees,
                    "solver_route": simulation.gpu.as_ref().map(|gpu| format!("{:?}", gpu.solver_route())),
                })
            });
        }
        match completed {
            Ok(Some(completed)) if completed.diagnostics.error_flags == 0 => {
                if completed.tick_index <= simulation.completed_tick
                    || simulation
                        .live_state
                        .as_ref()
                        .is_some_and(|state| completed.tick_index <= state.tick)
                {
                    continue;
                }
                simulation.completed_tick = completed.tick_index;
                performance_capture::record("physics_publication", || {
                    serde_json::json!({
                        "tick":completed.tick_index,
                        "state_hash":performance_capture::state_hash(&completed.transforms, &completed.velocities, &completed.coordinates),
                        "callback_to_publication_ms":completed.callbacks_completed_at.map(|at| at.elapsed().as_secs_f64()*1000.0)
                    })
                });
                simulation.last_tick_readback = Some(completed.diagnostics);
                simulation.submission_to_readback_ms = Some(completed.submission_to_readback_ms);
                simulation.live_state = Some(LivePhysicsState {
                    tick: completed.tick_index,
                    transforms: completed.transforms.clone(),
                    velocities: completed.velocities,
                    coordinates: completed.coordinates,
                });
                if visual_snapshot_is_due(simulation.snapshot_tick, completed.tick_index) {
                    simulation.previous_transforms =
                        core::mem::replace(&mut simulation.transforms, completed.transforms);
                    simulation.previous_snapshot_tick = simulation.snapshot_tick;
                    simulation.snapshot_tick = completed.tick_index;
                    simulation.pose_revision = simulation.pose_revision.wrapping_add(1);
                    simulation.render_dirty = true;
                }
            }
            Ok(Some(completed)) => {
                stop_failed_simulation(
                    &mut simulation,
                    &mut state,
                    format!(
                        "physics tick {} reported flags {}",
                        completed.tick_index, completed.diagnostics.error_flags
                    ),
                );
                return;
            }
            Ok(None) => break,
            Err(error) => {
                stop_failed_simulation(&mut simulation, &mut state, error.to_string());
                return;
            }
        }
    }
    frozen.overlay(&mut simulation);
}

#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn advance_simulation(
    time: (Res<Time>, Res<Time<Real>>),
    mut world_runtime: ResMut<world::WorldRuntime>,
    publication: Res<WorldPhysicsPublication>,
    mut sequencer: ResMut<DriveSequencer>,
    mut gearboxes: ResMut<GearboxRuntime>,
    frozen: Res<freeze::DimensionFreeze>,
    mut automated: Local<automation::Driving>,
    selection: Res<SelectedTool>,
    mut state: ResMut<EditorState>,
    mut simulation: ResMut<AppSimulation>,
    mut hammer: ResMut<HammerInteraction>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut construction_visuals: Query<(&ConstructionVisual, &mut Visibility), Without<BearingVisual>>,
) {
    if !(simulation.is_running()
        || (simulation.cpu.is_some()
            && !world_runtime.clumps.bodies.is_empty()
            && simulation.failure.is_none()))
    {
        return;
    }
    let published_graph = simulation.published_graph.clone();
    simulation.ticks_submitted_per_frame = 0;

    if state.drive_rows_dirty {
        state.drive_rows_dirty = false;
        if let (Some(gpu), Some(creation)) = (simulation.gpu.as_ref(), simulation.creation.as_ref())
            && let Err(error) = gpu.write_mechanism_drives(
                &render_queue,
                &geared_gpu_drive_rows(creation, &published_graph, &sequencer, &gearboxes),
            )
        {
            stop_failed_simulation(&mut simulation, &mut state, error.to_string());
            return;
        }
    }

    // Construction visuals follow the published scene, never the terrain cut.
    // The gate below now only holds ticks before the first cut and across a
    // floating-origin rebase, but returning there still must not leave a newly
    // published ground-welded block solid but undrawn until the Garage rebuilt
    // the editor meshes.
    refresh_published_construction_visuals(
        &mut simulation,
        &published_graph,
        &state,
        *selection,
        &sequencer,
        &visuals,
        &mut meshes,
        &mut construction_visuals,
    );

    if performance_capture::is_draining() {
        return;
    }
    let replay_end = performance_capture::replay_end_tick();
    if performance_capture::is_active() && replay_end.is_some() {
        // Count time even when terrain readiness prevents this frame's submissions.
        let due = simulation.scheduler.advance(time.1.delta()).count();
        simulation.tick_backlog = simulation.tick_backlog.saturating_add(due);
    }

    match terrain_publication::publish(
        &mut simulation,
        &world_runtime,
        render_device.wgpu_device(),
        &render_queue,
    ) {
        Ok(true) => {}
        Ok(false) => {
            performance_capture::record(
                "physics_terrain_hold",
                || serde_json::json!({"tick":simulation.next_tick}),
            );
            return;
        }
        Err(error) => {
            stop_failed_simulation(&mut simulation, &mut state, error);
            return;
        }
    }

    // A foreground comparison starts from the loaded state, independent of
    // shader compilation and terrain preparation wall time. Warm-up renders
    // without advancing physics; capture readiness still requires a running scene.
    if automation::foreground() && !performance_capture::is_active() {
        return;
    }

    let ticks = {
        let available = if simulation.cpu.is_some() {
            MAXIMUM_TICKS_PER_FRAME
        } else {
            simulation
                .gpu
                .as_ref()
                .expect("running simulation has GPU state")
                .async_readback_slots_available()
                .min(MAXIMUM_TICKS_PER_FRAME)
        };
        let AppSimulation {
            scheduler,
            next_tick,
            tick_backlog,
            dropped_ticks,
            ..
        } = &mut *simulation;
        if let Some(end) = replay_end {
            let paused = publication.ready.is_some()
                || publication
                    .placement
                    .as_ref()
                    .is_some_and(weld_publication::Publication::ready);
            performance_capture::replay_batch(
                next_tick,
                tick_backlog,
                if paused {
                    0
                } else {
                    u64::try_from(available).unwrap_or(0)
                },
                end,
            )
        } else {
            next_simulation_ticks(
                scheduler,
                next_tick,
                tick_backlog,
                dropped_ticks,
                time.0.delta(),
                publication.ready.is_some()
                    || publication
                        .placement
                        .as_ref()
                        .is_some_and(weld_publication::Publication::ready),
                u64::try_from(available).unwrap_or(u64::MAX),
            )
        }
    };
    if !ticks.is_empty() {
        let tick_count =
            usize::try_from(ticks.end.saturating_sub(ticks.start)).unwrap_or(usize::MAX);
        let physics_started = std::time::Instant::now();
        let mut cpu_timings = mechanic_gpu::GpuSubmissionTimings::default();
        // The CPU route owns the tick itself. Taking it out of the
        // resource keeps the published state, drives and impulses borrowable while
        // it steps; it is restored before returning, including on failure.
        let mut cpu_route = simulation.cpu.take();
        for tick in ticks {
            if let Some((keys, seat)) = automation::drive_input(&simulation, &mut automated, tick) {
                let controller = published_graph.seat_controller(seat);
                step_drive_programs(
                    &simulation,
                    &frozen,
                    &mut sequencer,
                    &mut gearboxes,
                    &mut state,
                    &ButtonInput::default(),
                    &keys,
                    controller,
                    None,
                    tick,
                );
                let creation = simulation
                    .creation
                    .as_ref()
                    .expect("running simulation has creation");
                let drive_rows =
                    geared_gpu_drive_rows(creation, &published_graph, &sequencer, &gearboxes);
                performance_capture::record("physics_drive_rows", || {
                    use std::hash::{DefaultHasher, Hash, Hasher};
                    let mut hash = DefaultHasher::new();
                    bytemuck::cast_slice::<_, u8>(&drive_rows).hash(&mut hash);
                    serde_json::json!({"tick":tick, "hash":format!("{:016x}",hash.finish()), "rows":drive_rows.len()})
                });
                if state.drive_rows_dirty {
                    let gpu = simulation
                        .gpu
                        .as_ref()
                        .expect("running simulation has GPU state");
                    if let Err(error) = gpu.write_mechanism_drives(&render_queue, &drive_rows) {
                        stop_failed_simulation(&mut simulation, &mut state, error.to_string());
                        return;
                    }
                    state.drive_rows_dirty = false;
                }
            }
            match pending_hammer_impulse(&simulation, &mut hammer, tick) {
                Ok(Some(impulse)) => world_runtime.queue_player_reaction(impulse),
                Ok(None) => {}
                Err(error) => {
                    hammer.pending = None;
                    stop_failed_simulation(&mut simulation, &mut state, error);
                    return;
                }
            }
            if let Some(cpu) = cpu_route.as_mut() {
                let cpu_tick_started = std::time::Instant::now();
                if !cpu.is_ready() {
                    // No terrain cut reached the CPU scene yet, so a tick would
                    // drop every body through the world.
                    break;
                }
                let creation = simulation
                    .creation
                    .as_ref()
                    .expect("running simulation has creation");
                let drive_rows =
                    geared_gpu_drive_rows(creation, &published_graph, &sequencer, &gearboxes);
                let stepped = cpu
                    .prepare_clump_tick(&mut world_runtime.clumps)
                    .and_then(|()| {
                        cpu.step(
                            tick,
                            cpu_physics::gravity(),
                            &drive_rows,
                            world_runtime.pending_player_reactions(),
                        )
                    });
                world_runtime.clear_player_reactions();
                match stepped {
                    Ok(completed) => {
                        cpu.update_clumps(&mut world_runtime);
                        cpu.accumulate_soil(&mut world_runtime);
                        let publication_started = std::time::Instant::now();
                        let sequence = completed.sequence;
                        performance_capture::record(
                            "physics_submit",
                            || serde_json::json!({"tick":tick,"sequence":sequence,"route":"cpu"}),
                        );
                        simulation.publish_cpu_tick(tick, completed);
                        performance_capture::record(
                            "physics_readback",
                            || serde_json::json!({"tick":tick,"sequence":sequence,"route":"cpu","error_flags":0,"publication_ms":publication_started.elapsed().as_secs_f64()*1000.0,"complete_cpu_tick_ms":cpu_tick_started.elapsed().as_secs_f64()*1000.0}),
                        );
                        continue;
                    }
                    Err(message) => {
                        if !world_runtime.clumps.bodies.is_empty() {
                            stop_failed_simulation(&mut simulation, &mut state, message);
                            simulation.cpu = cpu_route;
                            return;
                        }
                        // Physics in the world never pauses. The GPU runtime stays
                        // resident, so it takes over from the last CPU publication
                        // and runs this same tick; the next construction publication
                        // builds a fresh CPU route.
                        let gpu = simulation
                            .gpu
                            .as_ref()
                            .expect("running simulation has GPU state");
                        let handoff = simulation.live_state.as_ref().map_or(Ok(()), |live| {
                            gpu.write_body_states(
                                &render_queue,
                                &live.transforms,
                                &live.velocities,
                            )
                            .map_err(|error| error.to_string())?;
                            gpu.initialize_mechanism_coordinates(&render_queue, &live.coordinates)
                                .map_err(|error| error.to_string())
                        });
                        if let Err(error) = handoff {
                            stop_failed_simulation(
                                &mut simulation,
                                &mut state,
                                format!("{message} Handing the state to the GPU failed: {error}"),
                            );
                            return;
                        }
                        warn!("{message} Continuing on the GPU solver.");
                        state.feedback = Some(format!(
                            "CPU physics fell back to the GPU solver: {message}"
                        ));
                        state.drive_rows_dirty = true;
                        cpu_route = None;
                    }
                }
            }
            let dispatch = simulation
                .gpu
                .as_ref()
                .expect("running simulation has GPU state")
                .dispatch_tick_with_impulses(
                    render_device.wgpu_device(),
                    &render_queue,
                    tick,
                    world_runtime.pending_player_reactions(),
                );
            let submission = match dispatch {
                Ok(submission) => submission,
                Err(error) => {
                    world_runtime.clear_player_reactions();
                    stop_failed_simulation(&mut simulation, &mut state, error.to_string());
                    return;
                }
            };
            performance_capture::record(
                "physics_submit",
                || serde_json::json!({"tick":tick, "sequence":submission.submission_sequence, "encoding_ms":submission.cpu_timings.encoding_ms, "finalization_ms":submission.cpu_timings.finalization_ms, "submission_ms":submission.cpu_timings.submission_ms, "readback_setup_ms":submission.cpu_timings.readback_setup_ms}),
            );
            cpu_timings.encoding_ms += submission.cpu_timings.encoding_ms;
            cpu_timings.finalization_ms += submission.cpu_timings.finalization_ms;
            cpu_timings.submission_ms += submission.cpu_timings.submission_ms;
            cpu_timings.readback_setup_ms += submission.cpu_timings.readback_setup_ms;
            world_runtime.clear_player_reactions();
        }
        simulation.cpu = cpu_route;
        simulation.record_performance(physics_started.elapsed(), tick_count, cpu_timings, None);
    }
    simulation.in_flight_tick_count = simulation.gpu.as_ref().map_or(0, |gpu| {
        u32::try_from(gpu.in_flight_tick_count()).unwrap_or(u32::MAX)
    });
}
