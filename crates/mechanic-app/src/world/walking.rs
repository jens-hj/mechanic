//! The player's kinematic walk and the construction collision it stands on.

use super::list::{WorldListPhase, WorldListState};
use super::streaming::{bounds_overlap, node_overlaps};
use super::{
    ButtonInput, Quat, Res, ResMut, Result, Single, String, Time, ToString, Vec2, Vec3, With,
    WorldDiagnostics, WorldRuntime, warn,
};
use crate::camera::{MainCamera, PlayerCamera, PlayerState};
use crate::controls::GameAction;
use crate::editor::history::EditorHistory;
use crate::editor::state::EditorGraph;
use crate::simulation::state::AppSimulation;
use bevy::math::DVec3;
use bevy::tasks::futures::check_ready;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use mechanic_core::ConstructionGraph;
use mechanic_gpu::GpuExternalImpulse;
use mechanic_world::{
    ActiveTerrainNode, ActiveTerrainScene, ConstructionBodyPose, ConstructionCollisionIndex,
    KinematicCapsule, KinematicCollisionScene, KinematicInput, TerrainMeshChunk, TerrainNodeId,
    TerrainTransitionMask, WorldPosition,
};

pub(super) const MAX_CONTROLLER_TICKS_PER_FRAME: usize = 4;

pub(super) const STEP_VISUAL_SMOOTHING_SECONDS: f32 = 0.08;

pub(super) struct PlayerCollisionBuild {
    pub(super) editor_revision: u64,
    pub(super) task: Task<Result<ConstructionCollisionIndex, String>>,
}

#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn walk_world(
    time: Res<Time>,
    actions: Res<ButtonInput<GameAction>>,
    mut camera: Single<&mut PlayerCamera, With<MainCamera>>,
    list: Res<WorldListState>,
    graph: Res<EditorGraph>,
    history: Res<EditorHistory>,
    simulation: Res<AppSimulation>,
    mut runtime: ResMut<WorldRuntime>,
    mut player: ResMut<PlayerState>,
    mut diagnostics: ResMut<WorldDiagnostics>,
) {
    if runtime.material_publication_pending() {
        return;
    }
    sync_player_construction_collision(
        &mut runtime,
        &simulation,
        &graph.0,
        history.current_revision,
    );
    if list.phase() != WorldListPhase::Playing {
        runtime.capsule.velocity = bevy::math::DVec3::ZERO;
        runtime.jump_queued = false;
        return;
    }
    let reactions_are_authoritative = simulation.is_running()
        && runtime.collision_revision == simulation.world_revision
        && runtime.collision_editor_revision == Some(history.current_revision);
    if player.seat.is_some() {
        runtime.capsule.velocity = DVec3::ZERO;
        runtime.capsule.clear_support();
        runtime.controller_accumulator = 0.0;
        runtime.jump_queued = false;
        runtime.step_visual_offset = 0.0;
        runtime.walking_suspended = true;
        return;
    }
    if runtime.walking_suspended {
        runtime.capsule.position =
            WorldPosition(runtime.floating_origin.0 + player.position.as_dvec3());
        runtime.capsule.velocity = DVec3::ZERO;
        runtime.capsule.clear_support();
        runtime.step_visual_offset = 0.0;
        runtime.walking_suspended = false;
    }
    let mut local = Vec2::ZERO;
    local.y += if actions.pressed(GameAction::MoveForward) {
        1.0
    } else {
        0.0
    };
    local.y -= if actions.pressed(GameAction::MoveBackward) {
        1.0
    } else {
        0.0
    };
    local.x += if actions.pressed(GameAction::MoveRight) {
        1.0
    } else {
        0.0
    };
    local.x -= if actions.pressed(GameAction::MoveLeft) {
        1.0
    } else {
        0.0
    };
    let forward = camera.look_rotation() * Vec3::NEG_Z;
    let right = camera.look_rotation() * Vec3::X;
    let movement = (right * local.x + forward * local.y)
        .with_y(0.0)
        .normalize_or_zero();
    if !runtime.player_terrain_ready {
        let ready = runtime.active_terrain.iter().any(|(&id, chunk)| {
            terrain_chunk_has_collision_near(
                chunk,
                runtime
                    .active_terrain_ready_faces
                    .get(&id)
                    .copied()
                    .unwrap_or_default(),
                &runtime.capsule,
            )
        });
        if !ready {
            runtime.capsule.velocity = bevy::math::DVec3::ZERO;
            player.position = runtime
                .capsule
                .position
                .relative_to(runtime.floating_origin);
            return;
        }
        runtime.player_terrain_ready = true;
    }
    let (controller_ticks, jump_queued) = {
        let WorldRuntime {
            controller_accumulator,
            jump_queued,
            ..
        } = &mut *runtime;
        advance_controller(
            controller_accumulator,
            jump_queued,
            f64::from(time.delta_secs()),
            actions.just_pressed(GameAction::Jump),
        )
    };
    if controller_ticks == 0 {
        return;
    }

    let mut collision_index = runtime.construction_collision.take();
    let mut pending_reactions = core::mem::take(&mut runtime.pending_player_reactions);
    let scene = ActiveTerrainScene {
        chunks: &runtime.active_terrain,
        ready_faces: &runtime.active_terrain_ready_faces,
        spatial_index: &runtime.active_terrain_index,
    };
    let mut capsule = runtime.capsule;
    let mut resolved_contacts = 0_u32;
    let mut queued_reactions = 0_u32;
    let mut stepped_height = 0.0_f32;
    for tick in 0..controller_ticks {
        let mut collision_scene = KinematicCollisionScene {
            terrain: &scene,
            construction: collision_index.as_mut(),
            floating_origin: runtime.floating_origin.0,
        };
        let result = capsule.tick(
            &mut collision_scene,
            KinematicInput {
                movement: bevy::math::DVec2::new(f64::from(movement.x), f64::from(movement.z)),
                sprint: actions.pressed(GameAction::Sprint),
                jump: tick == 0 && jump_queued,
                jump_held: actions.pressed(GameAction::Jump),
            },
            mechanic_core::TICK_SECONDS,
        );
        camera.yaw += result.support_yaw_delta;
        resolved_contacts = resolved_contacts.saturating_add(result.resolved_contacts);
        stepped_height += result.stepped_height;
        for reaction in result.reaction_impulses() {
            if reactions_are_authoritative {
                pending_reactions.push(GpuExternalImpulse::new(
                    reaction.compound_index,
                    reaction.world_point,
                    reaction.impulse,
                ));
                queued_reactions = queued_reactions.saturating_add(1);
            }
        }
    }
    if let Some(index) = collision_index.as_ref() {
        let metrics = index.metrics();
        diagnostics.player_collision_query_ms = metrics.query_ms;
        diagnostics.dynamic_collision_refit_ms = metrics.dynamic_refit_ms;
        diagnostics.player_collision_candidates = metrics.candidate_count;
    }
    diagnostics.player_collision_contacts = resolved_contacts;
    diagnostics.player_reaction_impulses = queued_reactions;
    runtime.pending_player_reactions = pending_reactions;
    runtime.construction_collision = collision_index;
    runtime.capsule = capsule;
    // Graph frames, bearing anchors, and physics bodies share this persisted origin.
    // Moving only terrain transforms would move construction relative to the world.
    let capsule_position = capsule.position.relative_to(runtime.floating_origin);
    runtime.step_visual_offset = smooth_step_visual_offset(
        runtime.step_visual_offset,
        stepped_height,
        time.delta_secs(),
    );
    player.position = capsule_position + Vec3::Y * runtime.step_visual_offset;
    runtime.document.player_pose.translation = capsule.position;
}

pub(super) fn advance_controller(
    accumulator: &mut f64,
    jump_queued: &mut bool,
    delta_seconds: f64,
    jump_pressed: bool,
) -> (usize, bool) {
    *jump_queued |= jump_pressed;
    *accumulator += delta_seconds;
    let ticks =
        ((*accumulator / mechanic_core::TICK_SECONDS) as usize).min(MAX_CONTROLLER_TICKS_PER_FRAME);
    if ticks == 0 {
        return (0, false);
    }
    *accumulator -= mechanic_core::TICK_SECONDS * ticks as f64;
    (ticks, core::mem::take(jump_queued))
}

pub(super) fn smooth_step_visual_offset(
    current: f32,
    stepped_height: f32,
    delta_seconds: f32,
) -> f32 {
    let offset = current - stepped_height;
    let smoothed = offset * (-delta_seconds / STEP_VISUAL_SMOOTHING_SECONDS).exp();
    if smoothed.abs() < 1.0e-4 {
        0.0
    } else {
        smoothed
    }
}

pub(super) fn sync_player_construction_collision(
    runtime: &mut WorldRuntime,
    simulation: &AppSimulation,
    graph: &ConstructionGraph,
    editor_revision: u64,
) {
    let authoritative_revision = simulation
        .world_revision
        .filter(|revision| revision.0 == editor_revision);
    if authoritative_revision.is_some()
        && (runtime.collision_revision != authoritative_revision
            || runtime.collision_editor_revision != Some(editor_revision))
    {
        install_player_collision(
            runtime,
            simulation
                .creation
                .as_ref()
                .map(ConstructionCollisionIndex::new),
            editor_revision,
            authoritative_revision,
        );
        runtime.collision_build = None;
        runtime.collision_failed_revision = None;
    } else if authoritative_revision.is_none() {
        sync_provisional_player_collision(runtime, graph, editor_revision);
    }

    let Some(physics_revision) = runtime.collision_revision else {
        return;
    };
    if simulation.world_revision != Some(physics_revision) {
        return;
    }
    let Some(index) = runtime.construction_collision.as_mut() else {
        return;
    };
    if runtime.collision_snapshot_tick == simulation.snapshot_tick
        && runtime.collision_pose_revision == simulation.pose_revision
        && runtime.collision_poses.len() == simulation.transforms.len()
    {
        return;
    }
    let tick_delta = simulation
        .snapshot_tick
        .saturating_sub(runtime.collision_snapshot_tick);
    let elapsed = tick_delta.max(1) as f32 * mechanic_core::TICK_SECONDS_F32;
    runtime
        .collision_poses
        .resize(simulation.transforms.len(), ConstructionBodyPose::default());
    for (body, transform) in simulation.transforms.iter().copied().enumerate() {
        let translation = Vec3::from_slice(&transform.position[..3]);
        let rotation = Quat::from_array(transform.rotation).normalize();
        let previous = index.body_pose(body as u32).unwrap_or_default();
        let linear_velocity = if runtime.collision_snapshot_tick == 0 {
            Vec3::ZERO
        } else {
            (translation - previous.translation) / elapsed
        };
        let angular_velocity = if runtime.collision_snapshot_tick == 0 {
            Vec3::ZERO
        } else {
            angular_velocity(previous.rotation, rotation, elapsed)
        };
        runtime.collision_poses[body] = ConstructionBodyPose {
            translation,
            rotation,
            linear_velocity,
            angular_velocity,
        };
    }
    if index.refit_dynamic(&runtime.collision_poses) {
        runtime.collision_snapshot_tick = simulation.snapshot_tick;
        runtime.collision_pose_revision = simulation.pose_revision;
    }
}

pub(super) fn sync_provisional_player_collision(
    runtime: &mut WorldRuntime,
    graph: &ConstructionGraph,
    editor_revision: u64,
) {
    if runtime
        .collision_build
        .as_ref()
        .is_some_and(|build| build.editor_revision != editor_revision)
    {
        runtime.collision_build = None;
    }
    let completed = runtime
        .collision_build
        .as_mut()
        .and_then(|build| check_ready(&mut build.task));
    if let Some(completed) = completed {
        let completed_revision = runtime
            .collision_build
            .take()
            .expect("completed collision build is still owned")
            .editor_revision;
        if completed_revision == editor_revision {
            match completed {
                Ok(index) => {
                    install_player_collision(runtime, Some(index), editor_revision, None);
                    runtime.collision_failed_revision = None;
                }
                Err(error) => {
                    runtime.collision_failed_revision = Some(editor_revision);
                    warn!("could not compile player construction collision: {error}");
                }
            }
        }
    }
    if runtime.collision_editor_revision != Some(editor_revision)
        && runtime.collision_build.is_none()
        && runtime.collision_failed_revision != Some(editor_revision)
    {
        runtime.pending_player_reactions.clear();
        runtime.capsule.clear_support();
        let graph = graph.clone();
        runtime.collision_build = Some(PlayerCollisionBuild {
            editor_revision,
            task: AsyncComputeTaskPool::get()
                .spawn(async move { compile_player_collision(&graph) }),
        });
    }
}

pub(super) fn compile_player_collision(
    graph: &ConstructionGraph,
) -> Result<ConstructionCollisionIndex, String> {
    let creation = graph.compile().map_err(|error| error.to_string())?;
    Ok(ConstructionCollisionIndex::new(&creation))
}

pub(super) fn install_player_collision(
    runtime: &mut WorldRuntime,
    collision: Option<ConstructionCollisionIndex>,
    editor_revision: u64,
    physics_revision: Option<(u64, u64)>,
) {
    runtime.pending_player_reactions.clear();
    runtime.capsule.clear_support();
    runtime.construction_collision = collision;
    runtime.collision_revision = physics_revision;
    runtime.collision_editor_revision = Some(editor_revision);
    runtime.collision_snapshot_tick = 0;
    runtime.collision_pose_revision = 0;
    runtime.collision_poses.clear();
}

pub(super) fn reset_player_collision_publication(runtime: &mut WorldRuntime) {
    runtime.construction_collision = None;
    runtime.collision_revision = None;
    runtime.collision_editor_revision = None;
    runtime.collision_build = None;
    runtime.collision_failed_revision = None;
    runtime.collision_snapshot_tick = 0;
    runtime.collision_pose_revision = 0;
    runtime.collision_poses.clear();
    runtime.pending_player_reactions.clear();
    runtime.capsule.clear_support();
    runtime.jump_queued = false;
    runtime.step_visual_offset = 0.0;
}

pub(super) fn angular_velocity(previous: Quat, current: Quat, elapsed: f32) -> Vec3 {
    let delta = (current * previous.inverse()).normalize();
    let (axis, angle) = delta.to_axis_angle();
    if angle.is_finite() && elapsed > 0.0 {
        axis * angle / elapsed
    } else {
        Vec3::ZERO
    }
}

pub(super) fn player_collision_nodes<'a>(
    cut: &'a [ActiveTerrainNode],
    capsule: &KinematicCapsule,
) -> impl Iterator<Item = TerrainNodeId> + 'a {
    let (minimum, maximum) = capsule_loading_bounds(capsule);
    cut.iter()
        .map(|node| node.id)
        .filter(move |&id| node_overlaps(id, minimum, maximum))
}

pub(super) fn terrain_chunk_has_collision_near(
    chunk: &TerrainMeshChunk,
    ready_faces: TerrainTransitionMask,
    capsule: &KinematicCapsule,
) -> bool {
    let (minimum, maximum) = capsule_loading_bounds(capsule);
    bounds_overlap(
        chunk.bounds.minimum.0,
        chunk.bounds.maximum.0,
        minimum,
        maximum,
    ) && chunk
        .index_groups
        .sealed_index_count(chunk.transition_mask, ready_faces)
        != 0
}

pub(super) fn capsule_loading_bounds(
    capsule: &KinematicCapsule,
) -> (bevy::math::DVec3, bevy::math::DVec3) {
    let radius = capsule.config.radius;
    (
        capsule.position.0 - bevy::math::DVec3::new(radius, capsule.config.step_height, radius),
        capsule.position.0 + bevy::math::DVec3::new(radius, capsule.config.standing_height, radius),
    )
}
