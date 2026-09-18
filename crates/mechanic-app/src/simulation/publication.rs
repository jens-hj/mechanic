//! Building and swapping the compiled world simulation while physics keeps running.

use crate::editor::hammer::HammerInteraction;
use crate::editor::history::{EditorHistory, EditorSnapshot, cancel_transient_editor_state};
use crate::editor::state::{EditorGraph, EditorState};
use crate::scheduler::FixedStepScheduler;
use crate::simulation::state::{AppSimulation, LivePhysicsState, stop_failed_simulation};
use crate::{
    cpu_physics, freeze, performance_capture, showcase, suspension_editor, terrain_publication,
    weld_publication, world,
};
use bevy::prelude::{Mat3, Quat, Res, ResMut, Resource, State, Vec3, default, format, vec};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::tasks::futures::check_ready;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use mechanic_core::{BearingSocket, CompiledCreation, ConstructionGraph, PartId};
use mechanic_gpu::{GpuPhysics, GpuPhysicsConfig, GpuPhysicsPipelines, GpuTransform, GpuVelocity};
use std::sync::Arc;

pub(crate) type WorldPhysicsRevision = (u64, u64);

pub(crate) struct PreparedWorldPhysics {
    pub(crate) graph: ConstructionGraph,
    pub(crate) creation: CompiledCreation,
    pub(crate) gpu: Option<GpuPhysics>,
    pub(crate) cpu: Option<cpu_physics::PreparedRoute>,
}

pub(crate) struct WorldPhysicsTask {
    pub(crate) revision: WorldPhysicsRevision,
    pub(crate) task: Task<Result<PreparedWorldPhysics, String>>,
}

/// Owns compilation and upload work that must never block the editor frame.
#[derive(Resource)]
pub(crate) struct WorldPhysicsPublication {
    pub(crate) pipelines: Arc<GpuPhysicsPipelines>,
    pub(crate) pending: Option<WorldPhysicsTask>,
    pub(crate) ready: Option<(WorldPhysicsRevision, PreparedWorldPhysics)>,
    pub(crate) failed_revision: Option<WorldPhysicsRevision>,
    pub(crate) accepted_editor: Option<(EditorSnapshot, EditorHistory)>,
    pub(crate) placement: Option<weld_publication::Publication>,
}

impl Default for WorldPhysicsPublication {
    fn default() -> Self {
        Self {
            pipelines: Arc::new(GpuPhysicsPipelines::new()),
            pending: None,
            ready: None,
            failed_revision: None,
            accepted_editor: None,
            placement: None,
        }
    }
}

// Keep the asynchronous scene lifecycle and its publication boundary together.
#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn maintain_space_simulation(
    space: Res<State<world::AppSpace>>,
    worlds: Res<world::WorldListState>,
    mut graph: ResMut<EditorGraph>,
    mut history: ResMut<EditorHistory>,
    mut runtime: ResMut<world::WorldRuntime>,
    mut state: ResMut<EditorState>,
    mut simulation: ResMut<AppSimulation>,
    mut hammer: ResMut<HammerInteraction>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut publication: ResMut<WorldPhysicsPublication>,
    mut frozen: ResMut<freeze::DimensionFreeze>,
) {
    if *space.get() == world::AppSpace::Garage {
        frozen.reset();
        publication.placement = None;
        publication.accepted_editor = None;
        publication.pending = None;
        publication.ready = None;
        publication.failed_revision = None;
        if simulation.gpu.is_some() || simulation.cpu.is_some() {
            simulation.gpu = None;
            simulation.cpu = None;
            simulation.world_revision = None;
            *hammer = HammerInteraction::default();
            state.construction_mesh_dirty = true;
        }
        return;
    }
    if worlds.is_open() {
        frozen.reset();
        publication.placement = None;
        publication.accepted_editor = None;
        publication.pending = None;
        publication.ready = None;
        publication.failed_revision = None;
        *simulation = AppSimulation::default();
        return;
    }

    if !cpu_physics::selected() && !runtime.clumps.bodies.is_empty() {
        stop_failed_simulation(
            &mut simulation,
            &mut state,
            "This world contains loose material and requires CPU physics".to_owned(),
        );
        return;
    }

    if weld_publication::maintain(
        &mut graph,
        &mut state,
        &mut history,
        &mut simulation,
        &mut frozen,
        &mut runtime,
        &mut publication,
        &render_device,
        &render_queue,
    ) {
        return;
    }

    let revision = (history.current_revision, runtime.foundation_revision());
    let completed = publication
        .pending
        .as_mut()
        .and_then(|pending| check_ready(&mut pending.task));
    if let Some(completed) = completed {
        let completed_revision = publication
            .pending
            .take()
            .expect("completed physics task is still owned")
            .revision;
        if world_physics_result_is_current(completed_revision, revision) {
            match completed {
                Ok(prepared) => publication.ready = Some((revision, prepared)),
                Err(error) => {
                    publication.failed_revision = Some(revision);
                    state.weld_restore = None;
                    state.feedback = Some(format!("Cannot update live world physics: {error}"));
                }
            }
        }
    }

    if publication
        .ready
        .as_ref()
        .is_some_and(|(prepared_revision, _)| {
            !world_physics_result_is_current(*prepared_revision, revision)
        })
    {
        publication.ready = None;
    }
    if publication.ready.is_some() {
        if simulation.is_running() && simulation.completed_tick + 1 < simulation.next_tick {
            // advance_simulation stops submitting while ready is owned here;
            // polling continues until the final old tick is authoritative.
            return;
        }
        let install_started = std::time::Instant::now();
        let (_, prepared) = publication.ready.take().expect("ready scene exists");
        let replacement = if state.weld_restore.is_some() {
            replacement_simulation_with_transfer(
                prepared,
                &simulation,
                revision,
                &render_queue,
                None,
                state.weld_restore.as_ref(),
            )
        } else {
            replacement_simulation(prepared, &simulation, revision, &render_queue)
        }
        .and_then(|mut replacement| {
            let candidate = if let Some(restore) = &state.weld_restore {
                restore.hold(&replacement, &frozen)
            } else {
                frozen.prepare_publication(&replacement, &runtime)?
            };
            candidate.install_publication(&mut replacement, &render_queue)?;
            Ok((replacement, candidate))
        });
        match replacement {
            Ok((mut replacement, candidate)) => {
                inherit_terrain_residency(&mut simulation, &mut replacement, &render_device);
                *simulation = replacement;
                *frozen = candidate;
                if state.weld_restore.take().is_some() {
                    runtime.accept_weld_freeze(frozen.saved_record(), &graph.0, &state);
                }
                runtime.accept_frozen_publication(&graph.0, &state);
                publication.accepted_editor =
                    Some((EditorSnapshot::capture(&graph.0, &state), history.clone()));
                publication.failed_revision = None;
                *hammer = HammerInteraction::default();
                performance_capture::record("world_physics_install", || {
                    let creation = simulation.creation.as_ref();
                    serde_json::json!({
                        "graph_revision": revision.0,
                        "foundation_revision": revision.1,
                        "install_ms": install_started.elapsed().as_secs_f64() * 1000.0,
                        "bodies": creation.map(|c| c.compounds.len()),
                        "static_bodies": creation
                            .map(|c| c.compounds.iter().filter(|body| body.is_static).count()),
                        "colliders": creation.map(|c| c.colliders.len()),
                        "static_colliders": creation.map(|c| {
                            c.compounds
                                .iter()
                                .filter(|body| body.is_static)
                                .map(|body| body.collider_range.len())
                                .sum::<usize>()
                        }),
                        "bearings": creation.map(|c| c.bearings.len()),
                        "closures": creation.map(|c| c.loop_topology.closure_bearings.len()),
                    })
                });
            }
            Err(error) => {
                publication.failed_revision = Some(revision);
                state.weld_restore = None;
                state.feedback = Some(format!("Cannot update live world physics: {error}"));
                if let Some((snapshot, accepted_history)) = &publication.accepted_editor
                    && history.current_revision != accepted_history.current_revision
                {
                    graph.0 = (*snapshot.graph).clone();
                    state.placed_bearings.clone_from(&snapshot.placed_bearings);
                    *history = accepted_history.clone();
                    cancel_transient_editor_state(&mut graph.0, &mut state);
                    state.construction_mesh_dirty = true;
                    state.feedback = Some(format!("Construction edit rejected: {error}"));
                }
                return;
            }
        }
    }

    if simulation.world_revision == Some(revision) {
        return;
    }
    let Some(static_parts) = runtime.static_parts_for_physics(history.current_revision) else {
        return;
    };
    if graph.0.part_count() == 0 && runtime.clumps.bodies.is_empty() {
        publication.pending = None;
        publication.ready = None;
        publication.failed_revision = None;
        *simulation = AppSimulation {
            world_revision: Some(revision),
            ..default()
        };
        return;
    }
    if publication.pending.is_some() || publication.failed_revision == Some(revision) {
        return;
    }

    let graph = graph.0.clone();
    let suspension_sockets = suspension_editor::sockets(&state.placed_bearings);
    let physics_config = GpuPhysicsConfig {
        ground_plane_enabled: false,
        mechanism_self_collisions: world_mechanism_self_collisions(&graph),
        ..GpuPhysicsConfig::default()
    };
    let device = render_device.clone();
    let queue = render_queue.clone();
    let pipelines = Arc::clone(&publication.pipelines);
    performance_capture::record("world_physics_request", || {
        serde_json::json!({
            "graph_revision": revision.0,
            "foundation_revision": revision.1,
            "part_count": graph.part_count(),
        })
    });
    publication.failed_revision = None;
    publication.pending = Some(WorldPhysicsTask {
        revision,
        task: AsyncComputeTaskPool::get().spawn(async move {
            prepare_world_physics(
                graph,
                revision.0,
                suspension_sockets,
                static_parts,
                physics_config,
                device,
                queue,
                pipelines,
            )
        }),
    });
}

/// Moves terrain collision geometry from a retired scene to its replacement.
///
/// Only the construction changes when a block is placed, so the packed chunks
/// and their uploaded device buffer stay valid. Without this the replacement
/// scene re-packs and re-uploads the whole terrain cut, which stalls the main
/// thread for seconds after every edit.
pub(crate) fn inherit_terrain_residency(
    previous: &mut AppSimulation,
    replacement: &mut AppSimulation,
    render_device: &RenderDevice,
) {
    if let (Some(retired), Some(cpu)) = (previous.cpu.as_mut(), replacement.cpu.as_mut()) {
        cpu.inherit_terrain(retired);
    }
    let mut resident = false;
    if let (Some(retired), Some(gpu)) = (previous.gpu.as_mut(), replacement.gpu.as_mut()) {
        gpu.adopt_terrain_residency(
            render_device.wgpu_device(),
            retired.take_terrain_residency(),
        );
        resident = true;
    }
    replacement
        .terrain_publication
        .inherit(&mut previous.terrain_publication, resident);
}

pub(crate) const fn world_physics_result_is_current(
    completed: WorldPhysicsRevision,
    desired: WorldPhysicsRevision,
) -> bool {
    completed.0 == desired.0 && completed.1 == desired.1
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn prepare_world_physics(
    graph: ConstructionGraph,
    generation: u64,
    suspension_sockets: Vec<BearingSocket>,
    anchored: Vec<PartId>,
    physics_config: GpuPhysicsConfig,
    render_device: RenderDevice,
    render_queue: RenderQueue,
    pipelines: Arc<GpuPhysicsPipelines>,
) -> Result<PreparedWorldPhysics, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let compile_started = std::time::Instant::now();
        let creation = if graph.part_count() == 0 {
            CompiledCreation::default()
        } else {
            graph
                .compile_with_suspension_sockets(anchored, &suspension_sockets)
                .map_err(|error| error.to_string())?
        };
        let compile_ms = compile_started.elapsed().as_secs_f64() * 1000.0;
        let cpu_started = std::time::Instant::now();
        let cpu = cpu_physics::selected()
            .then(|| cpu_physics::PreparedRoute::new(&creation, generation))
            .transpose()?;
        let cpu_prepare_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;
        let scene_started = std::time::Instant::now();
        let gpu = creation_requires_live_physics(&creation)
            .then(|| {
                let mut gpu = GpuPhysics::new_with_pipelines(
                    render_device.wgpu_device(),
                    &render_queue,
                    &creation,
                    physics_config,
                    &pipelines,
                )
                .map_err(|error| error.to_string())?;
                // Compile recovery pipelines and allocate their scratch state on
                // the worker, before the first terrain upload on the main thread.
                gpu.write_terrain_chunks(
                    render_device.wgpu_device(),
                    &render_queue,
                    [],
                    bevy::math::DVec3::ZERO,
                )
                .map_err(|error| error.to_string())?;
                Ok::<_, String>(gpu)
            })
            .transpose()?;
        performance_capture::record("world_physics_prepare", || {
            serde_json::json!({
                "compile_ms": compile_ms,
                "cpu_prepare_ms": cpu_prepare_ms,
                "scene_ms": scene_started.elapsed().as_secs_f64() * 1000.0,
                "body_count": creation.compounds.len(),
                "collider_count": creation.colliders.len(),
            })
        });
        Ok(PreparedWorldPhysics {
            graph,
            creation,
            gpu,
            cpu,
        })
    }))
    .map_err(|_| "physics compilation worker panicked".to_owned())?
}

pub(crate) fn replacement_simulation(
    prepared: PreparedWorldPhysics,
    previous: &AppSimulation,
    revision: WorldPhysicsRevision,
    render_queue: &RenderQueue,
) -> Result<AppSimulation, String> {
    replacement_simulation_for_weld(prepared, previous, revision, render_queue, None)
}

pub(crate) fn replacement_simulation_for_weld(
    prepared: PreparedWorldPhysics,
    previous: &AppSimulation,
    revision: WorldPhysicsRevision,
    render_queue: &RenderQueue,
    placement: Option<&weld_publication::Intent>,
) -> Result<AppSimulation, String> {
    replacement_simulation_with_transfer(
        prepared,
        previous,
        revision,
        render_queue,
        placement,
        None,
    )
}

pub(crate) fn replacement_simulation_with_transfer(
    prepared: PreparedWorldPhysics,
    previous: &AppSimulation,
    revision: WorldPhysicsRevision,
    render_queue: &RenderQueue,
    placement: Option<&weld_publication::Intent>,
    restore: Option<&weld_publication::Restore>,
) -> Result<AppSimulation, String> {
    let PreparedWorldPhysics {
        graph,
        creation,
        gpu,
        cpu,
    } = prepared;
    let (transforms, velocities, coordinates) = if let Some(restore) = restore {
        restore.states(&creation, &graph, previous)?
    } else if let Some(placement) = placement {
        placement.states(&creation, &graph, previous)?
    } else {
        validate_merged_body_poses(&creation, &graph, previous)?;
        let (transforms, velocities) = rebuilt_body_states(&creation, &graph, previous);
        let coordinates =
            rebuilt_mechanism_coordinates(&creation, previous, &transforms, &velocities);
        (transforms, velocities, coordinates)
    };
    let next_tick = previous.next_tick.max(1);
    let live_state = Some(LivePhysicsState {
        tick: next_tick.saturating_sub(1),
        transforms: transforms.clone(),
        velocities: velocities.clone(),
        coordinates: coordinates.clone(),
    });
    let cpu = cpu
        .map(|prepared| {
            prepared
                .install(
                    revision.0,
                    next_tick.saturating_sub(1),
                    &transforms,
                    &velocities,
                    &coordinates,
                )
                .map(Box::new)
        })
        .transpose()?;
    let Some(gpu) = gpu else {
        return Ok(AppSimulation {
            cpu,
            creation: Some(creation),
            published_graph: graph,
            live_state,
            next_tick,
            previous_transforms: transforms.clone(),
            transforms,
            previous_snapshot_tick: next_tick.saturating_sub(1),
            snapshot_tick: next_tick,
            world_revision: Some(revision),
            ..default()
        });
    };
    gpu.enable_async_readback();
    if crate::env::is_set(crate::env::PERF_CAPTURE_DIR) {
        gpu.enable_readback_timing();
    }
    gpu.write_body_states(render_queue, &transforms, &velocities)
        .map_err(|error| format!("cannot preserve live body state: {error}"))?;
    gpu.initialize_mechanism_coordinates(render_queue, &coordinates)
        .map_err(|error| format!("cannot preserve live joint state: {error}"))?;
    Ok(AppSimulation {
        live_state,
        gpu: Some(gpu),
        cpu,
        creation: Some(creation),
        published_graph: graph,
        scheduler: FixedStepScheduler::new(),
        next_tick,
        tick_backlog: 0,
        dropped_ticks: previous.dropped_ticks,
        completed_tick: next_tick.saturating_sub(1),
        previous_transforms: transforms.clone(),
        transforms,
        previous_snapshot_tick: next_tick.saturating_sub(2),
        snapshot_tick: next_tick.saturating_sub(1),
        pose_revision: previous.pose_revision.wrapping_add(1),
        static_mesh_dirty: true,
        rendered_feature_preview: None,
        render_dirty: true,
        physics_cpu_ms: None,
        physics_submission_timings: None,
        ticks_submitted_per_frame: 0,
        in_flight_tick_count: 0,
        submission_to_readback_ms: None,
        visual_update_ms: None,
        last_tick_readback: None,
        failure: None,
        world_revision: Some(revision),
        terrain_publication: terrain_publication::TerrainPublication::default(),
    })
}

pub(crate) fn creation_requires_live_physics(creation: &CompiledCreation) -> bool {
    creation
        .compounds
        .iter()
        .any(|compound| !compound.is_static)
}

pub(crate) fn world_mechanism_self_collisions(graph: &ConstructionGraph) -> bool {
    // Dimension-link activation must preserve contacts between moving parts of
    // one mechanism. Compilation already excludes directly jointed bodies.
    !showcase::uses_reduced_collision_mode(graph)
}

pub(crate) fn rebuilt_body_states(
    creation: &CompiledCreation,
    graph: &ConstructionGraph,
    previous: &AppSimulation,
) -> (Vec<GpuTransform>, Vec<GpuVelocity>) {
    let mut transforms = creation
        .compounds
        .iter()
        .map(|compound| GpuTransform {
            position: compound.root_translation.extend(0.0).to_array(),
            rotation: compound.root_rotation.to_array(),
        })
        .collect::<Vec<_>>();
    let mut velocities = vec![
        GpuVelocity {
            linear: [0.0; 4],
            angular: [0.0; 4]
        };
        creation.compounds.len()
    ];
    let Some(previous_creation) = previous.creation.as_ref() else {
        return (transforms, velocities);
    };
    let old_bodies = previous_creation
        .part_to_compound
        .iter()
        .copied()
        .collect::<std::collections::BTreeMap<_, _>>();
    let source_transforms = previous
        .live_state
        .as_ref()
        .map_or(previous.transforms.as_slice(), |state| {
            state.transforms.as_slice()
        });
    for (new_index, compound) in creation.compounds.iter().enumerate() {
        if compound.is_static {
            continue;
        }
        let mut ancestors = edit_body_ancestors(compound, graph, &old_bodies);
        ancestors.retain(|body, _| !previous_creation.compounds[*body].is_static);
        let Some((&old_index, &(part, source))) = ancestors.first_key_value() else {
            continue;
        };
        let Some(&current) = source_transforms.get(old_index) else {
            continue;
        };
        let Some(remapped) =
            body_pose_from_edit_ancestor(compound, graph, previous, old_index, part, source)
        else {
            continue;
        };
        let position = Vec3::from_slice(&remapped.position[..3]);
        transforms[new_index] = remapped;
        if ancestors.len() == 1 {
            if let Some(velocity) = previous
                .live_state
                .as_ref()
                .and_then(|state| state.velocities.get(old_index))
            {
                let angular = Vec3::from_slice(&velocity.angular[..3]);
                let displacement = position - Vec3::from_slice(&current.position[..3]);
                velocities[new_index] = GpuVelocity {
                    linear: (Vec3::from_slice(&velocity.linear[..3]) + angular.cross(displacement))
                        .extend(0.0)
                        .to_array(),
                    angular: velocity.angular,
                };
            }
        } else {
            velocities[new_index] = merged_body_velocity(
                compound,
                transforms[new_index],
                ancestors.keys().copied(),
                previous,
            );
        }
    }
    (transforms, velocities)
}

/// The source body identities represented by this compound, each counted once.
pub(crate) fn edit_body_ancestors(
    compound: &mechanic_core::CompiledCompound,
    graph: &ConstructionGraph,
    old_bodies: &std::collections::BTreeMap<PartId, u32>,
) -> std::collections::BTreeMap<usize, (PartId, PartId)> {
    let mut ancestors = std::collections::BTreeMap::new();
    for &part in &compound.source_parts {
        let mut source = Some(part);
        while source.is_some_and(|source| !old_bodies.contains_key(&source)) {
            source = source.and_then(|source| graph.edit_source(source));
        }
        if let Some(source) = source
            && let Some(&body) = old_bodies.get(&source)
        {
            ancestors.entry(body as usize).or_insert((part, source));
        }
    }
    ancestors
}

pub(crate) fn body_pose_from_edit_ancestor(
    compound: &mechanic_core::CompiledCompound,
    graph: &ConstructionGraph,
    previous: &AppSimulation,
    body: usize,
    part: PartId,
    source: PartId,
) -> Option<GpuTransform> {
    let old = previous.creation.as_ref()?.compounds.get(body)?;
    let source_transforms = previous
        .live_state
        .as_ref()
        .map_or(previous.transforms.as_slice(), |state| {
            state.transforms.as_slice()
        });
    let current = source_transforms.get(body)?;
    let old_frame = previous.published_graph.part_frame(source)?;
    let new_frame = graph.part_frame(part)?;
    let authored_remap = old_frame.compose(new_frame.inverse());
    let world_from_old_build = Quat::from_array(current.rotation) * old.root_rotation.conjugate();
    let position = Vec3::from_slice(&current.position[..3])
        + world_from_old_build
            * (authored_remap.point(compound.root_translation) - old.root_translation);
    let rotation =
        (world_from_old_build * authored_remap.rotation() * compound.root_rotation).normalize();
    Some(GpuTransform {
        position: position.extend(0.0).to_array(),
        rotation: rotation.to_array(),
    })
}

/// Rejects a stale weld before any graph, renderer, or GPU state is published.
pub(crate) fn validate_merged_body_poses(
    creation: &CompiledCreation,
    graph: &ConstructionGraph,
    previous: &AppSimulation,
) -> Result<(), String> {
    let Some(old) = previous.creation.as_ref() else {
        return Ok(());
    };
    let old_bodies = old.part_to_compound.iter().copied().collect();
    for compound in &creation.compounds {
        let ancestors = edit_body_ancestors(compound, graph, &old_bodies);
        if ancestors.len() < 2 {
            continue;
        }
        let mut expected = compound.is_static.then_some(GpuTransform {
            position: compound.root_translation.extend(0.0).to_array(),
            rotation: compound.root_rotation.to_array(),
        });
        for (&body, &(part, source)) in &ancestors {
            let candidate =
                body_pose_from_edit_ancestor(compound, graph, previous, body, part, source)
                    .ok_or_else(|| {
                        "Cannot weld: a source creation has no authoritative pose".to_owned()
                    })?;
            if let Some(expected) = expected {
                let separation = Vec3::from_slice(&candidate.position[..3])
                    .distance(Vec3::from_slice(&expected.position[..3]));
                let relative = Quat::from_array(expected.rotation).conjugate()
                    * Quat::from_array(candidate.rotation);
                // One millimetre and 0.001 radians allow roundoff, not another
                // animation step. Quaternion sign does not affect this test.
                if !separation.is_finite()
                    || separation > 0.001
                    || !relative.is_finite()
                    || relative.xyz().length() > (0.001_f32 * 0.5).sin()
                {
                    return Err("Cannot weld: the creations moved apart while preparing the edit; try again".to_owned());
                }
            } else {
                expected = Some(candidate);
            }
        }
    }
    Ok(())
}

/// Conserves each source body's momentum once, measured about the new COM.
pub(crate) fn merged_body_velocity(
    compound: &mechanic_core::CompiledCompound,
    transform: GpuTransform,
    sources: impl Iterator<Item = usize>,
    previous: &AppSimulation,
) -> GpuVelocity {
    let mut result = GpuVelocity {
        linear: [0.0; 4],
        angular: [0.0; 4],
    };
    let Some((creation, state)) = previous.creation.as_ref().zip(previous.live_state.as_ref())
    else {
        return result;
    };
    let center = Vec3::from_slice(&transform.position[..3]);
    let mut momentum = Vec3::ZERO;
    let mut angular_momentum = Vec3::ZERO;
    for body in sources {
        let Some((pose, velocity)) = state.transforms.get(body).zip(state.velocities.get(body))
        else {
            continue;
        };
        let mass = creation.compounds[body].mass_properties;
        let linear = Vec3::from_slice(&velocity.linear[..3]);
        let angular = Vec3::from_slice(&velocity.angular[..3]);
        let basis = Mat3::from_quat(Quat::from_array(pose.rotation));
        let body_momentum = mass.mass * linear;
        momentum += body_momentum;
        angular_momentum += basis * mass.inertia * basis.transpose() * angular
            + (Vec3::from_slice(&pose.position[..3]) - center).cross(body_momentum);
    }
    let basis = Mat3::from_quat(Quat::from_array(transform.rotation));
    let inverse_inertia = basis * compound.mass_properties.inverse_inertia * basis.transpose();
    result.linear = (momentum / compound.mass_properties.mass)
        .extend(0.0)
        .to_array();
    result.angular = (inverse_inertia * angular_momentum).extend(0.0).to_array();
    result
}

pub(crate) fn rebuilt_mechanism_coordinates(
    creation: &CompiledCreation,
    previous: &AppSimulation,
    transforms: &[GpuTransform],
    velocities: &[GpuVelocity],
) -> Vec<mechanic_gpu::GpuMechanismCoordinate> {
    let old = previous.creation.as_ref().zip(previous.live_state.as_ref());
    creation
        .loop_topology
        .tree_bearings
        .iter()
        .map(|bearing| {
            old.and_then(|(compiled, state)| {
                if let Some(&index) = compiled.loop_topology.bearing_coordinates.get(bearing) {
                    return state.coordinates.get(index as usize).copied();
                }
                // Closure-only joints have no stored scalar coordinate. If one is
                // promoted into the tree, recover its state from the same body tick.
                compiled
                    .bearings
                    .iter()
                    .find(|row| row.source_bearing == *bearing)?;
                let row = creation
                    .bearings
                    .iter()
                    .find(|row| row.source_bearing == *bearing)?;
                coordinate_from_body_states(creation, row, transforms, velocities)
            })
            .unwrap_or(mechanic_gpu::GpuMechanismCoordinate {
                position: 0.0,
                velocity: 0.0,
            })
        })
        .collect()
}

pub(crate) fn coordinate_from_body_states(
    creation: &CompiledCreation,
    bearing: &mechanic_core::CompiledBearing,
    transforms: &[GpuTransform],
    velocities: &[GpuVelocity],
) -> Option<mechanic_gpu::GpuMechanismCoordinate> {
    let a = bearing.compound_a as usize;
    let b = bearing.compound_b as usize;
    let pose_a = transforms.get(a)?;
    let pose_b = transforms.get(b)?;
    let velocity_a = velocities.get(a)?;
    let velocity_b = velocities.get(b)?;
    let rotation_a = Quat::from_array(pose_a.rotation).normalize();
    let rotation_b = Quat::from_array(pose_b.rotation).normalize();
    let angular_a = Vec3::from_slice(&velocity_a.angular[..3]);
    let angular_b = Vec3::from_slice(&velocity_b.angular[..3]);
    let axis = rotation_a * bearing.local_axis_a;
    let (position, velocity) = match bearing.kind {
        mechanic_core::BearingKind::Rotational => {
            let initial = creation.compounds[a].root_rotation.conjugate()
                * creation.compounds[b].root_rotation;
            let delta = (rotation_a.conjugate() * rotation_b * initial.conjugate()).normalize();
            let position = 2.0 * delta.xyz().dot(bearing.local_axis_a).atan2(delta.w);
            // A closure has no winding counter. Choose its equivalent principal
            // angle; surviving tree coordinates retain their full winding above.
            let position = (position + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
                - std::f32::consts::PI;
            (position, (angular_b - angular_a).dot(axis))
        }
        mechanic_core::BearingKind::Linear(_) | mechanic_core::BearingKind::Suspension(_) => {
            let arm_a = rotation_a * bearing.local_anchor_a;
            let arm_b = rotation_b * bearing.local_anchor_b;
            let separation = Vec3::from_slice(&pose_b.position[..3]) + arm_b
                - Vec3::from_slice(&pose_a.position[..3])
                - arm_a;
            let relative_velocity = Vec3::from_slice(&velocity_b.linear[..3])
                + angular_b.cross(arm_b)
                - Vec3::from_slice(&velocity_a.linear[..3])
                - angular_a.cross(arm_a);
            (
                separation.dot(axis),
                relative_velocity.dot(axis) + separation.dot(angular_a.cross(axis)),
            )
        }
    };
    Some(mechanic_gpu::GpuMechanismCoordinate { position, velocity })
}

#[cfg(test)]
mod tests;
