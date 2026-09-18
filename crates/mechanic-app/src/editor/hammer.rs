//! The hammer tool: charge, impulse sizing, and delivery over several ticks.

use crate::camera::{MainCamera, MaterialWheelState, PlayerState};
use crate::controls::GameAction;
use crate::editor::raycast::raycast_simulation;
use crate::hotbar::{SelectedTool, Tool};
use crate::{AppSimulation, EditorState, automation, camera, tool_fx, ui};
use bevy::prelude::{
    ButtonInput, Camera, GlobalTransform, Quat, Res, ResMut, Resource, Single, Time, Vec2, Vec3,
    Window, With, format,
};
use mechanic_core::{CompiledCreation, TICK_SECONDS_F32};
use mechanic_gpu::{GpuExternalImpulse, GpuTransform};

pub(crate) const HAMMER_CHARGE_SECONDS: f32 = 1.5;

pub(crate) const HAMMER_MIN_IMPULSE: f32 = 25.0;

pub(crate) const HAMMER_MAX_IMPULSE: f32 = 4_000.0;

pub(crate) const HAMMER_MAX_POINT_TRAVEL_PER_TICK: f32 = 0.05;

pub(crate) const HAMMER_MAX_DELIVERY_TICKS: u16 = 12;

#[derive(Resource, Default)]
pub(crate) struct HammerInteraction {
    pub(crate) charging: Option<HammerCharge>,
    pub(crate) pending: Option<HammerImpact>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct HammerCharge {
    pub(crate) body_index: u32,
    pub(crate) local_point: Vec3,
    pub(crate) local_normal: Vec3,
    pub(crate) direction: Vec3,
    pub(crate) elapsed_seconds: f32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct HammerImpact {
    pub(crate) body_index: u32,
    pub(crate) local_point: Vec3,
    pub(crate) impulse_per_tick: Vec3,
    pub(crate) remaining_ticks: u16,
}

#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn handle_hammer_actions(
    actions: Res<ButtonInput<GameAction>>,
    time: Res<Time>,
    window: Single<&Window>,
    camera: Single<(&Camera, &GlobalTransform), With<MainCamera>>,
    simulation: Res<AppSimulation>,
    mut hammer: ResMut<HammerInteraction>,
    mut state: ResMut<EditorState>,
    selection: Res<SelectedTool>,
    overlay: Res<ui::UiInput>,
    player: Res<PlayerState>,
    wheel: Res<MaterialWheelState>,
    mut fx: Option<ResMut<tool_fx::ToolFx>>,
) {
    if !simulation.is_running() {
        hammer.charging = None;
        hammer.pending = None;
        return;
    }
    if !window.focused || overlay.blocks_pointer() || !player.world_input_active() || wheel.open {
        hammer.charging = None;
        hammer.pending = None;
        return;
    }
    let Some(tool) = selection.active_editor_tool() else {
        hammer.charging = None;
        return;
    };
    if tool != Tool::Hammer {
        hammer.charging = None;
        return;
    }
    if overlay.blocks_pointer() && hammer.charging.is_none() {
        return;
    }
    if actions.just_pressed(GameAction::Primary) {
        hammer.charging = None;
        let cursor = camera::viewport_center(Vec2::new(window.width(), window.height()));
        let hit = {
            let (camera, camera_transform) = *camera;
            camera.viewport_to_world(camera_transform, cursor).ok()
        }
        .and_then(|ray| {
            let creation = simulation
                .creation
                .as_ref()
                .expect("running simulation has compiled creation");
            raycast_simulation(
                &simulation.published_graph,
                creation,
                &simulation.transforms,
                ray.origin,
                ray.direction.as_vec3(),
            )
            .map(|hit| (hit, ray.direction.as_vec3()))
        });
        match hit {
            Some((hit, direction)) => {
                let creation = simulation
                    .creation
                    .as_ref()
                    .expect("running simulation has compiled creation");
                if creation.compounds[hit.body_index as usize].is_static {
                    state.feedback = Some("The fixed structure cannot be struck loose".to_owned());
                } else {
                    let transform = simulation.transforms[hit.body_index as usize];
                    let position = Vec3::from_slice(&transform.position[..3]);
                    let rotation = Quat::from_array(transform.rotation);
                    hammer.charging = Some(HammerCharge {
                        body_index: hit.body_index,
                        local_point: rotation.inverse() * (hit.point - position),
                        local_normal: rotation.inverse() * hit.normal,
                        direction: direction.normalize(),
                        elapsed_seconds: 0.0,
                    });
                    state.feedback =
                        Some("Charging hammer — release left mouse to strike".to_owned());
                }
            }
            None => state.feedback = Some("Point at a moving cuboid to use the hammer".to_owned()),
        }
    }

    if actions.pressed(GameAction::Primary)
        && let Some(charge) = hammer.charging.as_mut()
    {
        charge.elapsed_seconds =
            (charge.elapsed_seconds + time.delta_secs()).min(HAMMER_CHARGE_SECONDS);
    }

    if !actions.just_released(GameAction::Primary) {
        return;
    }
    let Some(charge) = hammer.charging.take() else {
        return;
    };
    let Some(&transform) = simulation.transforms.get(charge.body_index as usize) else {
        return;
    };
    let magnitude = hammer_impulse_magnitude(charge.elapsed_seconds);
    let impulse = charge.direction * magnitude;
    let (delivery_ticks, impulse_per_tick) = hammer_delivery(
        simulation
            .creation
            .as_ref()
            .expect("running simulation has compiled creation"),
        transform,
        charge.body_index,
        charge.local_point,
        impulse,
    );
    hammer.pending = Some(HammerImpact {
        body_index: charge.body_index,
        local_point: charge.local_point,
        impulse_per_tick,
        remaining_ticks: delivery_ticks,
    });
    if let Some(fx) = fx.as_deref_mut() {
        fx.push(tool_fx::Request::Sledge {
            hit: Vec3::from_slice(&transform.position[..3])
                + Quat::from_array(transform.rotation) * charge.local_point,
            normal: Quat::from_array(transform.rotation) * charge.local_normal,
        });
    }
    let delivered_magnitude = impulse_per_tick.length() * f32::from(delivery_ticks);
    state.feedback = Some(if delivered_magnitude + f32::EPSILON < magnitude {
        format!("Hammer strike: {delivered_magnitude:.0} N·s (stability limited)")
    } else {
        format!("Hammer strike: {magnitude:.0} N·s")
    });
}

pub(crate) fn pending_hammer_impulse(
    simulation: &AppSimulation,
    hammer: &mut HammerInteraction,
    tick: u64,
) -> Result<Option<GpuExternalImpulse>, String> {
    if hammer.pending.is_none() {
        hammer.pending = automation::hammer_impact(simulation, tick);
    }
    let Some(impact) = hammer.pending.as_mut() else {
        return Ok(None);
    };
    let transform = simulation
        .transforms
        .get(impact.body_index as usize)
        .ok_or_else(|| "hammer target no longer exists".to_owned())?;
    let position = Vec3::from_slice(&transform.position[..3]);
    let rotation = Quat::from_array(transform.rotation);
    let world_point = position + rotation * impact.local_point;
    let impulse = GpuExternalImpulse::new(impact.body_index, world_point, impact.impulse_per_tick);
    impact.remaining_ticks -= 1;
    if impact.remaining_ticks == 0 {
        hammer.pending = None;
    }
    Ok(Some(impulse))
}

pub(crate) fn hammer_impulse_magnitude(elapsed_seconds: f32) -> f32 {
    let charge = (elapsed_seconds / HAMMER_CHARGE_SECONDS).clamp(0.0, 1.0);
    HAMMER_MIN_IMPULSE + (HAMMER_MAX_IMPULSE - HAMMER_MIN_IMPULSE) * charge * charge
}

pub(crate) fn hammer_delivery(
    creation: &CompiledCreation,
    transform: GpuTransform,
    body_index: u32,
    local_point: Vec3,
    impulse: Vec3,
) -> (u16, Vec3) {
    let point_travel = hammer_point_travel(creation, transform, body_index, local_point, impulse);
    let maximum_travel = HAMMER_MAX_POINT_TRAVEL_PER_TICK * f32::from(HAMMER_MAX_DELIVERY_TICKS);
    let delivered_impulse = if point_travel > maximum_travel {
        impulse * (maximum_travel / point_travel)
    } else {
        impulse
    };
    let delivered_travel = point_travel.min(maximum_travel);
    let mut ticks = 1_u16;
    while delivered_travel > HAMMER_MAX_POINT_TRAVEL_PER_TICK * f32::from(ticks)
        && ticks < HAMMER_MAX_DELIVERY_TICKS
    {
        ticks += 1;
    }
    (ticks, delivered_impulse / f32::from(ticks))
}

pub(crate) fn hammer_point_travel(
    creation: &CompiledCreation,
    transform: GpuTransform,
    body_index: u32,
    local_point: Vec3,
    impulse: Vec3,
) -> f32 {
    let compound = &creation.compounds[body_index as usize];
    let mass = &compound.mass_properties;
    let rotation = Quat::from_array(transform.rotation);
    let arm = rotation * local_point;
    let local_torque = rotation.inverse() * arm.cross(impulse);
    let angular_delta = rotation * (mass.inverse_inertia * local_torque);
    let linear_delta = impulse * mass.inverse_mass;
    let maximum_radius = creation
        .colliders
        .iter()
        .filter(|collider| collider.compound_index == body_index)
        .map(|collider| collider.local_center.length() + collider_reach(collider))
        .fold(0.0_f32, f32::max);
    (linear_delta.length() + angular_delta.length() * maximum_radius) * TICK_SECONDS_F32
}

/// How far one collider extends from its own centre.
pub(crate) fn collider_reach(collider: &mechanic_core::LocalCollider) -> f32 {
    match &collider.shape {
        mechanic_core::ColliderShape::Cuboid { half_extents, .. } => half_extents.length(),
        mechanic_core::ColliderShape::Convex(convex) => convex
            .vertices
            .iter()
            .map(|vertex| (*vertex - collider.local_center).length())
            .fold(0.0_f32, f32::max),
    }
}
