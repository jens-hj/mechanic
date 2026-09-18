//! App-owned bounded tool feedback, independent of simulation suspension.
mod aperture;
#[cfg(debug_assertions)]
pub(crate) mod capture;
mod particles;
mod render;
#[cfg(test)]
mod tests;

use crate::{
    AppSimulation, EditorGraph, EditorHistory, EditorState, MaterialWheelState, PlayerState,
    camera::MainCamera,
    controls::GameAction,
    freeze::{DimensionFreeze, VisualSnapshot},
    hotbar::{MainTool, SelectedTool},
};
use bevy::{
    camera::visibility::NoFrustumCulling,
    light::{NotShadowCaster, NotShadowReceiver},
    prelude::*,
};
use particles::{DASHES, Particles, SHARDS, TRACES, color, halo};
pub(crate) use particles::{EmitterFrame, Kind, Request};
pub(crate) use render::FxCamera;
use render::{InstanceData, InstanceMaterialData};

pub(crate) struct ToolFxPlugin;
impl Plugin for ToolFxPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(debug_assertions)]
        capture::install(app);
        app.init_resource::<ToolFx>()
            .init_resource::<ToolEmitter>()
            .add_plugins((render::CustomMaterialPlugin, aperture::AperturePlugin))
            .add_systems(Startup, spawn.after(crate::setup))
            .add_systems(
                PostUpdate,
                (collect_plate_emitters, update)
                    .chain()
                    .after(bevy::transform::TransformSystems::Propagate),
            );
    }
}
#[derive(Resource, Default)]
pub(crate) struct ToolEmitter {
    pub local: Transform,
    pub deployed: bool,
    pub connector_phase: f32,
    pub active_end: usize,
    pub connector_plates: [Option<Vec3>; 6],
}
/// Marker on each connector plate's animated tip lamp.
#[derive(Component)]
pub(crate) struct ConnectorPlateEmitter {
    pub end: usize,
    pub panel: usize,
}
fn collect_plate_emitters(
    mut emitter: ResMut<ToolEmitter>,
    plates: Query<(&ConnectorPlateEmitter, &GlobalTransform)>,
) {
    emitter.connector_plates.fill(None);
    for (plate, transform) in &plates {
        if plate.end == emitter.active_end {
            emitter.connector_plates[plate.panel] = Some(transform.translation());
        }
    }
}
type BoundsKey = (Option<(u64, u64)>, u64, mechanic_core::DimensionLinkId);

#[derive(Resource, Default)]
pub(crate) struct ToolFx {
    particles: Particles,
    space_key: Option<(crate::world::AppSpace, bevy::math::DVec2)>,
    requests: [Option<Request>; 32],
    gesture: Option<(u64, Vec3)>,
    bounds_key: Option<BoundsKey>,
    snapshot: Option<VisualSnapshot>,
    pub aperture_multiplier: f32,
}
impl ToolFx {
    pub(crate) fn push(&mut self, request: Request) {
        if let Some(slot) = self.requests.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(request);
        }
    }
    pub(crate) fn clear(&mut self) {
        self.particles.clear();
        self.requests.fill(None);
        self.gesture = None;
        self.bounds_key = None;
        self.snapshot = None;
        self.aperture_multiplier = 1.0;
    }
}
#[derive(Component)]
struct Batch(u8);
fn spawn(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    // A shared mesh supplies Bevy's view pipeline layout; vertices come from the shader.
    let mesh = meshes.add(Triangle3d::new(Vec3::ZERO, Vec3::X, Vec3::Y));
    for (index, capacity) in [SHARDS, TRACES, DASHES + 12].into_iter().enumerate() {
        commands.spawn((
            Name::new("Tool FX batch"),
            Mesh3d(mesh.clone()),
            Transform::default(),
            Visibility::Visible,
            NoFrustumCulling,
            bevy::render::batching::NoAutomaticBatching,
            NotShadowCaster,
            NotShadowReceiver,
            Batch(u8::try_from(index).unwrap()),
            InstanceMaterialData {
                data: Vec::with_capacity(capacity),
                lines: index != 0,
                capacity,
            },
        ));
    }
}

/// Capture before the gesture mutates hover/context or publishes a replacement.
#[expect(clippy::too_many_arguments)]
pub(crate) fn capture_gesture(
    mut fx: ResMut<ToolFx>,
    history: Res<EditorHistory>,
    state: Res<EditorState>,
    actions: Res<ButtonInput<GameAction>>,
    selected: Res<SelectedTool>,
    player: Res<PlayerState>,
    overlay: Res<crate::ui::UiInput>,
    wheel: Res<MaterialWheelState>,
    windows: Query<&Window>,
) {
    fx.gesture = None;
    if selected.tool != Some(MainTool::MatterManipulator)
        || !player.world_input_active()
        || overlay.blocks_pointer()
        || wheel.open
        || !windows.iter().any(|w| w.focused)
        || !(actions.just_released(GameAction::Primary)
            || actions.just_released(GameAction::Secondary))
    {
        return;
    }
    let point = state
        .hovered
        .map(|hit| hit.point)
        .or_else(|| state.preview.map(|preview| preview.spec.pose.translation()))
        .or_else(|| {
            state
                .block_drag
                .as_ref()
                .map(|drag| drag.start.spec.pose.translation())
        })
        .or_else(|| {
            state
                .delete_drag
                .as_ref()
                .map(|drag| drag.start.pose.translation())
        });
    if let Some(mut point) = point {
        if let Some(context) = state.edit_context {
            point = context.frame_to_world.point(point);
        }
        fx.gesture = Some((history.current_revision, point));
    } else if let Some(hit) = state.hovered_simulation {
        fx.gesture = Some((history.current_revision, hit.point));
    }
}
pub(crate) fn finish_gesture(mut fx: ResMut<ToolFx>, history: Res<EditorHistory>) {
    if let Some((revision, hit)) = fx.gesture.take()
        && history.current_revision != revision
    {
        fx.push(Request::Matter { hit });
    }
}

#[derive(bevy::ecs::system::SystemParam)]
struct FxInput<'w, 's> {
    space: Res<'w, State<crate::world::AppSpace>>,
    worlds: Res<'w, crate::world::WorldListState>,
    world: Res<'w, crate::world::WorldRuntime>,
    graph: Res<'w, EditorGraph>,
    state: Res<'w, EditorState>,
    selected: Res<'w, SelectedTool>,
    player: Res<'w, PlayerState>,
    overlay: Res<'w, crate::ui::UiInput>,
    wheel: Res<'w, MaterialWheelState>,
    windows: Query<'w, 's, &'static Window>,
    camera: Query<'w, 's, &'static GlobalTransform, With<MainCamera>>,
}
#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
fn update(
    time: Res<Time>,
    mut fx: ResMut<ToolFx>,
    emitter: Res<ToolEmitter>,
    input: FxInput,
    simulation: Res<AppSimulation>,
    frozen: Res<DimensionFreeze>,
    mut batches: Query<(&Batch, &mut InstanceMaterialData)>,
    #[cfg(debug_assertions)] capture: Option<Res<capture::Capture>>,
) {
    let space_key = (*input.space.get(), input.world.horizontal_origin());
    if fx.space_key.is_some_and(|previous| previous != space_key) || input.worlds.is_open() {
        fx.clear();
    }
    fx.space_key = Some(space_key);
    let active = !input.worlds.is_open()
        && input.player.world_input_active()
        && !input.overlay.blocks_pointer()
        && !input.wheel.open
        && input.windows.iter().any(|w| w.focused)
        && emitter.deployed;
    let origin = input.camera.single().map_or(Transform::IDENTITY, |camera| {
        camera.compute_transform().mul_transform(emitter.local)
    });
    let frame = if active {
        match input.selected.tool {
            Some(MainTool::MatterManipulator) => Some(EmitterFrame {
                tool: Kind::Matter,
                origin,
                target: origin.translation,
                normal: Vec3::Y,
                connector_phase: 0.0,
                connector_plates: [None; 6],
            }),
            Some(MainTool::Welder) => {
                input
                    .state
                    .weld
                    .effects_target(&simulation)
                    .map(|(target, normal)| EmitterFrame {
                        tool: Kind::Welder,
                        origin,
                        target,
                        normal,
                        connector_phase: 0.0,
                        connector_plates: [None; 6],
                    })
            }
            Some(MainTool::Connector) => {
                crate::wire_drag_endpoints(&input.graph.0, &input.state, &simulation).map(
                    |(_, target)| EmitterFrame {
                        tool: Kind::Connector,
                        origin,
                        target,
                        normal: Vec3::Y,
                        connector_phase: emitter.connector_phase,
                        connector_plates: emitter.connector_plates,
                    },
                )
            }
            _ => None,
        }
    } else {
        None
    };
    #[cfg(debug_assertions)]
    let origin = capture
        .as_deref()
        .and_then(|c| c.emitter)
        .map_or(origin, |f| f.origin);
    let dt = time.delta_secs().clamp(0.0, 0.05);
    #[cfg(debug_assertions)]
    let frame = capture.as_deref().map_or(frame, |capture| capture.emitter);
    fx.particles.advance(dt, frame);
    for request in std::mem::take(&mut fx.requests).into_iter().flatten() {
        if active || matches!(request, Request::Freeze { .. }) || capture_active() {
            fx.particles.request(request, origin.translation);
        }
    }
    let key = frozen
        .saved_record()
        .map(|r| (simulation.world_revision, simulation.pose_revision, r.link));
    if fx.bounds_key.map(|key| key.2) != key.map(|key| key.2) && fx.bounds_key.is_some() {
        fx.particles.clear_freeze();
    }
    if fx.bounds_key != key {
        fx.snapshot = frozen.visual_snapshot(&simulation);
        fx.bounds_key = key;
    }
    #[cfg(debug_assertions)]
    if let Some(capture) = capture.as_deref() {
        fx.snapshot = capture.snapshot;
    }
    if let Some(snapshot) = fx.snapshot {
        let center = (snapshot.min + snapshot.max) * 0.5;
        debug_assert!(fx.bounds_key.is_none_or(|key| key.2 == snapshot.link));
        let movement = frozen.vertical_movement();
        #[cfg(debug_assertions)]
        let movement = capture.as_deref().map_or(movement, |c| c.movement);
        fx.particles.wake(
            center - Vec3::Y * movement,
            center,
            (snapshot.max - snapshot.min).length() * 0.5,
            dt,
        );
        fx.aperture_multiplier = particles::pulse(fx.particles.clock);
    } else {
        fx.aperture_multiplier = 1.0;
    }
    for (batch, mut data) in &mut batches {
        data.data.clear();
        match batch.0 {
            0 => {
                for s in fx.particles.shards.slots.iter().flatten() {
                    let tone = s.tone();
                    let k = s.size * [0.55, 0.775, 1.0][tone];
                    data.data.push(InstanceData {
                        position_size: s.p.extend(k).to_array(),
                        rotation: s.rotation().to_array(),
                        color: color(s.kind, tone),
                        endpoint: [0.0; 4],
                    });
                }
            }
            1 => {
                for s in fx.particles.traces.slots.iter().flatten() {
                    data.data.push(line(s.a, s.b, s.color()));
                }
            }
            _ => {
                if let Some(s) = fx.snapshot {
                    halo(s.min, s.max, fx.particles.clock, |a, b, inner| {
                        let mut c = color(Kind::Freeze, usize::from(!inner));
                        let strength = if inner { 0.5 } else { 0.95 };
                        for channel in &mut c[..3] {
                            *channel *= strength;
                        }
                        data.data.push(line(a, b, c));
                    });
                }
            }
        }
    }
}
fn line(a: Vec3, b: Vec3, color: [f32; 4]) -> InstanceData {
    InstanceData {
        position_size: a.extend(0.0).to_array(),
        endpoint: b.extend(0.0).to_array(),
        rotation: Quat::IDENTITY.to_array(),
        color,
    }
}
pub(crate) fn bloom() -> bevy::post_process::bloom::Bloom {
    use bevy::post_process::bloom::{Bloom, BloomCompositeMode, BloomPrefilter};
    Bloom {
        intensity: 0.035,
        low_frequency_boost: 0.0,
        prefilter: BloomPrefilter {
            threshold: 0.65,
            threshold_softness: 0.2,
        },
        composite_mode: BloomCompositeMode::Additive,
        ..Bloom::default()
    }
}

#[cfg(debug_assertions)]
pub(crate) fn capture_active() -> bool {
    capture::enabled()
}
#[cfg(not(debug_assertions))]
pub(crate) const fn capture_active() -> bool {
    false
}
