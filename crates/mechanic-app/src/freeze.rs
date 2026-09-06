//! Prescribed dimension-link holds and their validated animation targets.

use std::collections::{BTreeSet, VecDeque};

use bevy::{
    ecs::system::SystemParam, input::mouse::AccumulatedMouseScroll, prelude::*,
    render::renderer::RenderQueue,
};
use mechanic_core::{ColliderShape, CompiledCreation, DimensionLinkId, LocalCollider, PartId};
use mechanic_gpu::{GpuTransform, Obb};
use mechanic_world::FrozenCreationDoc;

use crate::{
    AppSimulation, EditorGraph, EditorHistory, EditorState, MaterialWheelState, PlayerState,
    controls::{ActionInput, GameAction},
    freeze_motion::{HeightRepeat, cardinal_heading, grid_ceiling, smooth_pose},
    hotbar::{MainTool, SelectedTool},
    settings::AppSettings,
    ui,
    world::WorldRuntime,
};

#[derive(Resource, Clone, Default)]
pub(crate) struct DimensionFreeze {
    record: Option<FrozenCreationDoc>,
    revision: Option<(u64, u64)>,
    held: Vec<bool>,
    poses: Vec<GpuTransform>,
    waypoints: VecDeque<Vec<GpuTransform>>,
    release_requested: bool,
    repeat: HeightRepeat,
    vertical_movement: f32,
}

impl DimensionFreeze {
    /// Current held geometry in world space; callers cache by construction and pose revision.
    pub(crate) fn visual_snapshot(&self, simulation: &AppSimulation) -> Option<VisualSnapshot> {
        let record = self.record?;
        if self.revision != simulation.world_revision {
            return None;
        }
        let creation = simulation.creation.as_ref()?;
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for collider in &creation.colliders {
            let body = collider.compound_index as usize;
            if self.held.get(body) != Some(&true) {
                continue;
            }
            let pose = self.poses.get(body)?;
            let rotation = Quat::from_array(pose.rotation);
            let mut include = |v: Vec3| {
                let p = position(*pose) + rotation * v;
                min = min.min(p);
                max = max.max(p);
            };
            match &collider.shape {
                ColliderShape::Cuboid {
                    local_rotation,
                    half_extents,
                } => {
                    for x in [-1.0, 1.0] {
                        for y in [-1.0, 1.0] {
                            for z in [-1.0, 1.0] {
                                include(
                                    collider.local_center
                                        + *local_rotation * (*half_extents * Vec3::new(x, y, z)),
                                );
                            }
                        }
                    }
                }
                ColliderShape::Convex(shape) => {
                    for &v in &shape.vertices {
                        include(v);
                    }
                }
            }
        }
        min.is_finite().then_some(VisualSnapshot {
            link: record.link,
            min,
            max,
        })
    }

    pub(crate) const fn vertical_movement(&self) -> f32 {
        self.vertical_movement
    }

    pub(crate) fn suspended_bearings(
        &self,
        simulation: &AppSimulation,
    ) -> BTreeSet<mechanic_core::BearingId> {
        if self.record.is_none() || self.revision != simulation.world_revision {
            return BTreeSet::new();
        }
        simulation
            .creation
            .as_ref()
            .map_or_else(BTreeSet::new, |creation| {
                creation
                    .bearings
                    .iter()
                    .filter_map(|bearing| {
                        (self.held.get(bearing.compound_a as usize) == Some(&true)
                            || self.held.get(bearing.compound_b as usize) == Some(&true))
                        .then_some(bearing.source_bearing)
                    })
                    .collect()
            })
    }

    pub(crate) fn suspended_controllers(&self, simulation: &AppSimulation) -> BTreeSet<PartId> {
        if self.record.is_none() || self.revision != simulation.world_revision {
            return BTreeSet::new();
        }
        simulation
            .creation
            .as_ref()
            .map_or_else(BTreeSet::new, |creation| {
                creation
                    .part_to_compound
                    .iter()
                    .filter_map(|&(part, body)| {
                        (self.held.get(body as usize) == Some(&true)
                            && simulation.published_graph.is_controller(part))
                        .then_some(part)
                    })
                    .collect()
            })
    }

    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn for_weld(
        &self,
        previous: &AppSimulation,
        replacement: &AppSimulation,
        destination: PartId,
        source: &[PartId],
    ) -> Self {
        let destination_held = previous
            .creation
            .as_ref()
            .and_then(|creation| {
                creation
                    .part_to_compound
                    .iter()
                    .find(|(part, _)| *part == destination)
            })
            .is_some_and(|(_, body)| self.held.get(*body as usize) == Some(&true));
        let source_held = previous.creation.as_ref().is_some_and(|creation| {
            creation.part_to_compound.iter().any(|(part, body)| {
                source.contains(part) && self.held.get(*body as usize) == Some(&true)
            })
        });
        if !destination_held && source_held {
            return Self::default();
        }
        let Some(record) = self.record else {
            return Self::default();
        };
        let Some((held, _)) = replacement
            .creation
            .as_ref()
            .and_then(|creation| component(creation, &replacement.published_graph, record.link))
        else {
            return Self::default();
        };
        Self {
            record: Some(record),
            held,
            poses: replacement.transforms.clone(),
            revision: replacement.world_revision,
            ..Self::default()
        }
    }

    pub(crate) fn restored_for_weld(&self, replacement: &AppSimulation) -> Self {
        let Some(record) = self.record else {
            return Self::default();
        };
        let Some((held, _)) = replacement
            .creation
            .as_ref()
            .and_then(|creation| component(creation, &replacement.published_graph, record.link))
        else {
            return Self::default();
        };
        Self {
            held,
            poses: replacement.transforms.clone(),
            revision: replacement.world_revision,
            waypoints: VecDeque::new(),
            ..self.clone()
        }
    }

    pub(crate) const fn saved_record(&self) -> Option<FrozenCreationDoc> {
        self.record
    }

    /// Validate a replacement without changing the published hold or GPU scene.
    pub(crate) fn prepare_publication(
        &self,
        simulation: &AppSimulation,
        world: &WorldRuntime,
    ) -> Result<Self, String> {
        self.prepare_with_environment(
            simulation,
            world.frozen_creation(),
            &|position| world.global_to_local(position),
            world,
        )
    }

    fn prepare_with_environment(
        &self,
        simulation: &AppSimulation,
        saved: Option<FrozenCreationDoc>,
        global_to_local: &impl Fn(mechanic_world::WorldPosition) -> Vec3,
        terrain: &impl TerrainProbe,
    ) -> Result<Self, String> {
        if self.revision == simulation.world_revision {
            return Ok(self.clone());
        }
        let Some(record) = self.record.or(saved) else {
            return Ok(self.clone());
        };
        let Some(creation) = simulation.creation.as_ref() else {
            return Ok(self.clone());
        };
        let Some((held, pivot)) = component(creation, &simulation.published_graph, record.link)
        else {
            return Ok(Self::default());
        };
        let restoring = self.record.is_none();
        let target = default_poses(
            creation,
            &simulation.transforms,
            &held,
            pivot,
            global_to_local(record.target),
            record.heading,
        );
        let poses = if restoring {
            target.clone()
        } else {
            simulation.transforms.clone()
        };
        let blockers = obstacles(creation, &simulation.transforms, &held);
        if !endpoint_clear(creation, &held, &target, &blockers, terrain) {
            return Err("Frozen target is obstructed in this construction generation".to_owned());
        }
        let waypoints = if restoring {
            VecDeque::new()
        } else {
            plan(creation, &held, &poses, &target, &blockers, terrain)
                .ok_or("Construction edit leaves no safe path to the frozen target")?
        };
        Ok(Self {
            poses,
            waypoints,
            held,
            record: Some(record),
            revision: simulation.world_revision,
            ..self.clone()
        })
    }

    /// Apply a prepared hold only to its candidate scene, before publishing it.
    pub(crate) fn install_publication(
        &self,
        simulation: &mut AppSimulation,
        queue: &RenderQueue,
    ) -> Result<(), String> {
        if self.record.is_none() || self.revision != simulation.world_revision {
            return Ok(());
        }
        if let Some(gpu) = &simulation.gpu {
            gpu.set_body_holds(queue, &self.held)
                .map_err(|e| e.to_string())?;
            gpu.prescribe_held_poses(queue, &self.poses)
                .map_err(|e| e.to_string())?;
        }
        self.overlay(simulation);
        Ok(())
    }

    /// Install a saved hold before the replacement scene can submit a tick.
    pub(crate) fn rebind(
        &mut self,
        simulation: &mut AppSimulation,
        world: &WorldRuntime,
        queue: &RenderQueue,
    ) -> Result<(), String> {
        if self.revision == simulation.world_revision {
            return Ok(());
        }
        let prepared = self.prepare_publication(simulation, world)?;
        prepared.install_publication(simulation, queue)?;
        *self = prepared;
        Ok(())
    }

    /// Async readbacks must not replace newer prescribed animation poses.
    pub(crate) fn overlay(&self, simulation: &mut AppSimulation) {
        if self.record.is_none() || self.revision != simulation.world_revision {
            return;
        }
        let mut changed = false;
        for (body, &held) in self.held.iter().enumerate() {
            if !held {
                continue;
            }
            if let (Some(pose), Some(destination)) =
                (self.poses.get(body), simulation.transforms.get_mut(body))
            {
                changed |= *destination != *pose;
                *destination = *pose;
                if let Some(live) = simulation.live_state.as_mut() {
                    if let Some(row) = live.transforms.get_mut(body) {
                        *row = *pose;
                    }
                    if let Some(row) = live.velocities.get_mut(body) {
                        row.linear = [0.0; 4];
                        row.angular = [0.0; 4];
                    }
                }
            }
        }
        if let (Some(creation), Some(live)) = (&simulation.creation, &mut simulation.live_state) {
            for bearing in &creation.bearings {
                if self.held[bearing.compound_a as usize]
                    && let Some(coordinate) = bearing.coordinate_index
                    && let Some(row) = live.coordinates.get_mut(coordinate as usize)
                {
                    row.position = 0.0;
                    row.velocity = 0.0;
                }
            }
        }
        if changed {
            simulation.pose_revision = simulation.pose_revision.wrapping_add(1);
            simulation.render_dirty = true;
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct VisualSnapshot {
    pub link: DimensionLinkId,
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(SystemParam)]
pub(crate) struct FreezeInput<'w, 's> {
    keyboard: Res<'w, ButtonInput<KeyCode>>,
    mouse: Res<'w, ButtonInput<MouseButton>>,
    scroll: Res<'w, AccumulatedMouseScroll>,
    settings: Res<'w, AppSettings>,
    selection: Res<'w, SelectedTool>,
    player: Res<'w, PlayerState>,
    overlay: Res<'w, ui::UiInput>,
    wheel: Res<'w, MaterialWheelState>,
    windows: Query<'w, 's, &'static Window>,
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn update(
    time: Res<Time>,
    mut frozen: ResMut<DimensionFreeze>,
    mut simulation: ResMut<AppSimulation>,
    mut world: ResMut<WorldRuntime>,
    graph: Res<EditorGraph>,
    history: Res<EditorHistory>,
    mut editor: ResMut<EditorState>,
    queue: Res<RenderQueue>,
    input: FreezeInput,
    mut fx: Option<ResMut<crate::tool_fx::ToolFx>>,
) {
    frozen.vertical_movement = 0.0;
    if !simulation.is_running() {
        frozen.repeat.reset();
        return;
    }
    if let Err(error) = frozen.rebind(&mut simulation, &world, &queue) {
        editor.feedback = Some(error);
        return;
    }
    if frozen
        .record
        .is_some_and(|r| world.active_dimension_link() != Some(r.link))
    {
        frozen.release_requested = true;
    }
    let accepts = input.selection.tool == Some(MainTool::Hammer)
        && input.player.world_input_active()
        && !input.overlay.blocks_keyboard()
        && !input.wheel.open
        && input.windows.iter().any(|w| w.focused);
    let actions = ActionInput::new(
        input.settings.controls(),
        &input.keyboard,
        &input.mouse,
        &input.scroll,
    );
    if accepts && actions.just_pressed_for_tool(GameAction::FreezeCreation, MainTool::Hammer) {
        if frozen.record.is_some() {
            frozen.release_requested = true;
            frozen.repeat.reset();
        } else if simulation.world_revision
            != Some((history.current_revision, world.foundation_revision()))
        {
            editor.feedback = Some("Wait for construction changes to finish publishing".to_owned());
        } else {
            let result = begin(
                &mut frozen,
                &mut simulation,
                &mut world,
                &graph,
                &editor,
                &queue,
            );
            if result.is_ok()
                && let Some(snapshot) = frozen.visual_snapshot(&simulation)
                && let Some(fx) = fx.as_deref_mut()
            {
                fx.push(crate::tool_fx::Request::Freeze {
                    center: (snapshot.min + snapshot.max) * 0.5,
                    radius: (snapshot.max - snapshot.min).length() * 0.5,
                });
            }
            editor.feedback = Some(result.map_or_else(
                |error| error,
                |()| "Linked creation frozen — arrows adjust height".to_owned(),
            ));
        }
    }
    let steps = if accepts && !frozen.release_requested {
        frozen.repeat.advance(
            time.delta_secs(),
            actions.pressed_for_tool(GameAction::RaiseFrozenCreation, MainTool::Hammer),
            actions.pressed_for_tool(GameAction::LowerFrozenCreation, MainTool::Hammer),
        )
    } else {
        frozen.repeat.reset();
        0
    };
    if frozen.record.is_some() {
        for _ in 0..steps.unsigned_abs() {
            if let Err(error) = change_height(
                &mut frozen,
                &simulation,
                &mut world,
                if steps > 0 { 0.25 } else { -0.25 },
            ) {
                editor.feedback = Some(error);
                break;
            }
        }
    }
    if let Some(target) = frozen.waypoints.front() {
        let mut settled = true;
        let mut next = frozen.poses.clone();
        for (body, &held) in frozen.held.iter().enumerate() {
            if held {
                let (pose, done) = smooth_pose(next[body], target[body], time.delta_secs());
                next[body] = pose;
                settled &= done;
            }
        }
        if let Some(creation) = simulation.creation.as_ref() {
            let obstacles = obstacles(creation, &simulation.transforms, &frozen.held);
            if path_clear(
                creation,
                &frozen.held,
                &frozen.poses,
                &next,
                &obstacles,
                &*world,
            ) {
                if let Some(gpu) = &simulation.gpu
                    && let Err(error) = gpu.prescribe_held_poses(&queue, &next)
                {
                    editor.feedback = Some(error.to_string());
                    return;
                }
                if let Some(body) = frozen.held.iter().position(|held| *held) {
                    frozen.vertical_movement =
                        next[body].position[1] - frozen.poses[body].position[1];
                }
                frozen.poses = next;
                if settled {
                    frozen.waypoints.pop_front();
                }
            } else {
                frozen.repeat.reset();
                editor.feedback = Some("Freeze movement is obstructed".to_owned());
            }
        }
    }
    frozen.overlay(&mut simulation);
    if frozen.release_requested && frozen.waypoints.is_empty() {
        let saved = if world.frozen_creation().is_some() {
            world.persist_frozen_creation(None, &graph.0, &editor)
        } else {
            Ok(())
        };
        if let Err(error) = saved {
            editor.feedback = Some(error);
            return;
        }
        if let Some(gpu) = &simulation.gpu
            && let Err(error) = gpu.set_body_holds(&queue, &vec![false; frozen.held.len()])
        {
            editor.feedback = Some(error.to_string());
            return;
        }
        frozen.reset();
        editor.feedback = Some("Linked creation released from rest".to_owned());
    }
}

fn begin(
    frozen: &mut DimensionFreeze,
    simulation: &mut AppSimulation,
    world: &mut WorldRuntime,
    graph: &EditorGraph,
    editor: &EditorState,
    queue: &RenderQueue,
) -> Result<(), String> {
    let link = world
        .active_dimension_link()
        .ok_or("Activate a Dimension Link first")?;
    let creation = simulation
        .creation
        .as_ref()
        .ok_or("Wait for world physics")?;
    let (held, pivot) = component(creation, &simulation.published_graph, link)
        .ok_or("Active link is not in this world")?;
    if creation
        .compounds
        .iter()
        .zip(&held)
        .any(|(body, &held)| held && body.is_static)
    {
        return Err("Grounded creations cannot be frozen".to_owned());
    }
    let part = simulation
        .published_graph
        .dimension_link(link)
        .ok_or("Dimension Link was removed")?;
    let reference = creation
        .part_to_compound
        .iter()
        .find(|(p, _)| *p == part)
        .ok_or("Link body is unavailable")?
        .1 as usize;
    let poses = simulation
        .live_state
        .as_ref()
        .map_or(&simulation.transforms, |s| &s.transforms);
    let rotation = Quat::from_array(poses[reference].rotation)
        * creation.compounds[reference].root_rotation.conjugate();
    let center = position(poses[reference])
        + rotation * (pivot - creation.compounds[reference].root_translation);
    let heading = cardinal_heading(rotation);
    let obstacles = obstacles(creation, &simulation.transforms, &held);
    let base = world.local_to_global(center);
    for blocks in 0..=80 {
        let mut target = base;
        target.0.y = grid_ceiling(base.0.y) + f64::from(blocks) * 0.25;
        if target.0.y - base.0.y > 20.0 + 1.0e-6 {
            break;
        }
        let endpoint = default_poses(
            creation,
            poses,
            &held,
            pivot,
            world.global_to_local(target),
            heading,
        );
        let terrain = &*world;
        let Some(waypoints) = plan(creation, &held, poses, &endpoint, &obstacles, terrain) else {
            continue;
        };
        let record = FrozenCreationDoc {
            link,
            target,
            heading,
            construction_generation: 0,
        };
        world.persist_frozen_creation(Some(record), &graph.0, editor)?;
        if let Some(gpu) = &simulation.gpu {
            gpu.set_body_holds(queue, &held)
                .map_err(|e| e.to_string())?;
            gpu.prescribe_held_poses(queue, poses)
                .map_err(|e| e.to_string())?;
        }
        frozen.record = world.frozen_creation();
        frozen.revision = simulation.world_revision;
        frozen.held = held;
        frozen.poses = poses.clone();
        frozen.waypoints = waypoints;
        frozen.overlay(simulation);
        return Ok(());
    }
    Err("No safe freeze alignment path within 20 m of upward clearance".to_owned())
}

fn change_height(
    frozen: &mut DimensionFreeze,
    simulation: &AppSimulation,
    world: &mut WorldRuntime,
    delta: f32,
) -> Result<(), String> {
    let mut record = frozen.record.ok_or("No frozen creation")?;
    record.target.0.y += f64::from(delta);
    let creation = simulation
        .creation
        .as_ref()
        .ok_or("Wait for world physics")?;
    let (_, pivot) = component(creation, &simulation.published_graph, record.link)
        .ok_or("Dimension Link was removed")?;
    let mut endpoint = default_poses(
        creation,
        &frozen.poses,
        &frozen.held,
        pivot,
        world.global_to_local(record.target),
        record.heading,
    );
    let terrain = &*world;
    let ground_clear = |poses: &[GpuTransform]| {
        creation
            .colliders
            .iter()
            .filter(|c| frozen.held[c.compound_index as usize])
            .all(|c| terrain_clear(c, poses[c.compound_index as usize], 0.05, terrain))
    };
    if delta < 0.0 && !ground_clear(&endpoint) {
        let raised = |lift: f32| {
            endpoint
                .iter()
                .enumerate()
                .map(|(body, pose)| {
                    let mut pose = *pose;
                    if frozen.held[body] {
                        pose.position[1] += lift;
                    }
                    pose
                })
                .collect::<Vec<_>>()
        };
        let clear = minimum_clear_lift(-delta, |lift| ground_clear(&raised(lift)))
            .ok_or("Height step is obstructed")?;
        if clear >= -delta - 1.0e-4 {
            return Err("Creation is already 5 cm above the terrain".to_owned());
        }
        let adjusted = raised(clear);
        record.target.0.y += f64::from(clear);
        endpoint = adjusted;
    }
    let obstacles = obstacles(creation, &simulation.transforms, &frozen.held);
    let waypoints = plan(
        creation,
        &frozen.held,
        &frozen.poses,
        &endpoint,
        &obstacles,
        &*world,
    )
    .ok_or("Height step is obstructed")?;
    world.set_frozen_target(record);
    frozen.record = world.frozen_creation();
    frozen.waypoints = waypoints;
    Ok(())
}

/// Clip a blocked downward step to the first terrain-clear height.
fn minimum_clear_lift(maximum: f32, clear: impl Fn(f32) -> bool) -> Option<f32> {
    if !clear(maximum) {
        return None;
    }
    let (mut low, mut high) = (0.0, maximum);
    for _ in 0..14 {
        let middle = (low + high) * 0.5;
        if clear(middle) {
            high = middle;
        } else {
            low = middle;
        }
    }
    Some(high)
}

fn component(
    creation: &CompiledCreation,
    graph: &mechanic_core::ConstructionGraph,
    link: DimensionLinkId,
) -> Option<(Vec<bool>, Vec3)> {
    let part = graph.dimension_link(link)?;
    let body = creation
        .part_to_compound
        .iter()
        .find(|(p, _)| *p == part)?
        .1 as usize;
    let id = creation.loop_topology.body_parents[body].component_index;
    Some((
        creation
            .loop_topology
            .body_parents
            .iter()
            .map(|b| b.component_index == id)
            .collect(),
        graph.part_position(part)?,
    ))
}

fn default_poses(
    creation: &CompiledCreation,
    current: &[GpuTransform],
    held: &[bool],
    pivot: Vec3,
    target: Vec3,
    heading: u8,
) -> Vec<GpuTransform> {
    let rotation = Quat::from_rotation_y(f32::from(heading) * std::f32::consts::FRAC_PI_2);
    creation
        .compounds
        .iter()
        .enumerate()
        .map(|(body, compound)| {
            if !held[body] {
                return current[body];
            }
            GpuTransform {
                position: (target + rotation * (compound.root_translation - pivot))
                    .extend(0.0)
                    .to_array(),
                rotation: (rotation * compound.root_rotation).to_array(),
            }
        })
        .collect()
}

fn position(pose: GpuTransform) -> Vec3 {
    Vec3::from_slice(&pose.position[..3])
}

fn sphere(collider: &LocalCollider, pose: GpuTransform) -> (Vec3, f32) {
    let radius = match &collider.shape {
        ColliderShape::Cuboid { half_extents, .. } => half_extents.length(),
        ColliderShape::Convex(shape) => shape
            .vertices
            .iter()
            .map(|v| (*v - collider.local_center).length())
            .fold(0.0, f32::max),
    };
    (
        position(pose) + Quat::from_array(pose.rotation) * collider.local_center,
        radius,
    )
}

fn obstacles(creation: &CompiledCreation, poses: &[GpuTransform], held: &[bool]) -> Vec<Obb> {
    creation
        .colliders
        .iter()
        .filter(|c| !held[c.compound_index as usize])
        .map(|c| {
            let pose = poses[c.compound_index as usize];
            let rotation = Quat::from_array(pose.rotation);
            match &c.shape {
                ColliderShape::Cuboid {
                    local_rotation,
                    half_extents,
                } => Obb {
                    center: position(pose) + rotation * c.local_center,
                    orientation: rotation * *local_rotation,
                    half_extents: *half_extents,
                },
                ColliderShape::Convex(shape) => {
                    let (low, high) = shape.vertices.iter().fold(
                        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                        |(low, high), &v| (low.min(v), high.max(v)),
                    );
                    Obb {
                        center: position(pose) + rotation * ((low + high) * 0.5),
                        orientation: rotation,
                        half_extents: (high - low) * 0.5,
                    }
                }
            }
        })
        .collect()
}

fn penetration(center: Vec3, radius: f32, obstacle: Obb) -> f32 {
    let local = obstacle.orientation.conjugate() * (center - obstacle.center);
    let outside = (local.abs() - obstacle.half_extents)
        .max(Vec3::ZERO)
        .length();
    radius - outside
}

/// Candidate internal collider pairs obey the same collision exclusions as physics.
fn held_pairs(creation: &CompiledCreation, held: &[bool]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (a, first) in creation.colliders.iter().enumerate() {
        if !held[first.compound_index as usize] {
            continue;
        }
        for (b, second) in creation.colliders.iter().enumerate().skip(a + 1) {
            if !held[second.compound_index as usize]
                || first.compound_index == second.compound_index
            {
                continue;
            }
            let pair = [
                first.compound_index.min(second.compound_index),
                first.compound_index.max(second.compound_index),
            ];
            if creation.collision_suppression.binary_search(&pair).is_err() {
                pairs.push((a, b));
            }
        }
    }
    pairs
}

fn held_endpoint_clear(creation: &CompiledCreation, held: &[bool], poses: &[GpuTransform]) -> bool {
    held_pairs(creation, held).into_iter().all(|(a, b)| {
        let first = &creation.colliders[a];
        let second = &creation.colliders[b];
        let Ok(a) = crate::live_weld::geometry(first, poses[first.compound_index as usize]) else {
            return false;
        };
        let Ok(b) = crate::live_weld::geometry(second, poses[second.compound_index as usize])
        else {
            return false;
        };
        crate::live_weld::penetration(&a, &b) <= 1.0e-4
    })
}

fn collider_body_radius(collider: &LocalCollider) -> f32 {
    match &collider.shape {
        ColliderShape::Cuboid { half_extents, .. } => {
            collider.local_center.length() + half_extents.length()
        }
        ColliderShape::Convex(shape) => shape
            .vertices
            .iter()
            .map(|vertex| vertex.length())
            .fold(0.0, f32::max),
    }
}

fn pose_at(start: GpuTransform, end: GpuTransform, amount: f32) -> GpuTransform {
    GpuTransform {
        position: position(start)
            .lerp(position(end), amount)
            .extend(0.0)
            .to_array(),
        rotation: Quat::from_array(start.rotation)
            .slerp(Quat::from_array(end.rotation), amount)
            .normalize()
            .to_array(),
    }
}

fn rotation_travel(start: GpuTransform, end: GpuTransform) -> f32 {
    let delta = Quat::from_array(end.rotation) * Quat::from_array(start.rotation).conjugate();
    2.0 * Vec3::new(delta.x, delta.y, delta.z)
        .length()
        .atan2(delta.w.abs())
}

/// Any SAT axis separating these expanded midpoint shapes proves the entire
/// interval clear. Common translation cancels and tangential sliding contributes
/// no expansion on its contact normal; close mechanisms need no bounding spheres.
fn swept_pair_depth(
    a: &crate::live_weld::ConvexGeometry,
    b: &crate::live_weld::ConvexGeometry,
    relative_translation: Vec3,
    rotation_bound: f32,
    half_interval: f32,
) -> f32 {
    let mut minimum = f32::INFINITY;
    for axis in a
        .normals
        .iter()
        .copied()
        .chain(b.normals.iter().copied())
        .chain(
            a.edges
                .iter()
                .flat_map(|&a| b.edges.iter().map(move |&b| a.cross(b))),
        )
    {
        let Some(axis) = axis.try_normalize() else {
            continue;
        };
        let interval = |vertices: &[Vec3]| {
            vertices
                .iter()
                .map(|v| v.dot(axis))
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), value| {
                    (low.min(value), high.max(value))
                })
        };
        let (a_low, a_high) = interval(&a.vertices);
        let (b_low, b_high) = interval(&b.vertices);
        let expansion = (relative_translation.dot(axis).abs() + rotation_bound) * half_interval;
        minimum = minimum.min((a_high - b_low).min(b_high - a_low) + expansion);
    }
    minimum
}

fn held_path_clear(
    creation: &CompiledCreation,
    held: &[bool],
    start: &[GpuTransform],
    end: &[GpuTransform],
) -> bool {
    for (a_index, b_index) in held_pairs(creation, held) {
        let a = &creation.colliders[a_index];
        let b = &creation.colliders[b_index];
        let a_body = a.compound_index as usize;
        let b_body = b.compound_index as usize;
        let relative_translation = (position(end[b_body]) - position(start[b_body]))
            - (position(end[a_body]) - position(start[a_body]));
        let rotation_bound = rotation_travel(start[a_body], end[a_body]) * collider_body_radius(a)
            + rotation_travel(start[b_body], end[b_body]) * collider_body_radius(b);
        let Ok(initial_a) = crate::live_weld::geometry(a, start[a_body]) else {
            return false;
        };
        let Ok(initial_b) = crate::live_weld::geometry(b, start[b_body]) else {
            return false;
        };
        let allowed = crate::live_weld::penetration(&initial_a, &initial_b).max(0.0) + 1.0e-4;
        let mut intervals = vec![(0.0, 1.0, 0_u8)];
        while let Some((low, high, depth)) = intervals.pop() {
            let midpoint = (low + high) * 0.5;
            let Ok(mid_a) =
                crate::live_weld::geometry(a, pose_at(start[a_body], end[a_body], midpoint))
            else {
                return false;
            };
            let Ok(mid_b) =
                crate::live_weld::geometry(b, pose_at(start[b_body], end[b_body], midpoint))
            else {
                return false;
            };
            if swept_pair_depth(
                &mid_a,
                &mid_b,
                relative_translation,
                rotation_bound,
                (high - low) * 0.5,
            ) <= allowed
            {
                continue;
            }
            if crate::live_weld::penetration(&mid_a, &mid_b) > allowed || depth >= 24 {
                return false;
            }
            intervals.push((midpoint, high, depth + 1));
            intervals.push((low, midpoint, depth + 1));
        }
    }
    true
}

trait TerrainProbe {
    fn penetration(&self, center: Vec3, radius: f32) -> Option<f32>;
    fn collider_clear(&self, collider: &LocalCollider, pose: GpuTransform, padding: f32) -> bool;
}

#[cfg(test)]
impl<F: Fn(Vec3, f32) -> Option<f32>> TerrainProbe for F {
    fn penetration(&self, center: Vec3, radius: f32) -> Option<f32> {
        self(center, radius)
    }
    fn collider_clear(&self, collider: &LocalCollider, pose: GpuTransform, padding: f32) -> bool {
        sampled_terrain_clear(collider, pose, padding, self)
    }
}

impl TerrainProbe for WorldRuntime {
    fn penetration(&self, center: Vec3, radius: f32) -> Option<f32> {
        self.freeze_sphere_penetration(center, radius)
    }

    fn collider_clear(&self, collider: &LocalCollider, pose: GpuTransform, padding: f32) -> bool {
        let (center, radius) = sphere(collider, pose);
        let Some(depth) = self.penetration(center, 0.0) else {
            return false;
        };
        if depth + radius + padding <= 1.0e-4 {
            return true;
        }
        // A surface intersection query alone cannot detect a fully buried body.
        if depth > 0.0 {
            return false;
        }
        let Ok(geometry) = crate::live_weld::geometry(collider, pose) else {
            return false;
        };
        let (low, high) = geometry.vertices.iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(low, high), &v| (low.min(v), high.max(v)),
        );
        self.freeze_triangles_clear(
            low - Vec3::splat(padding),
            high + Vec3::splat(padding),
            |points| triangle_clear(&geometry, points, padding),
        )
    }
}

fn triangle_clear(
    geometry: &crate::live_weld::ConvexGeometry,
    points: [Vec3; 3],
    padding: f32,
) -> bool {
    let edges = [
        points[1] - points[0],
        points[2] - points[1],
        points[0] - points[2],
    ];
    let triangle = crate::live_weld::ConvexGeometry {
        vertices: points.to_vec(),
        normals: vec![edges[0].cross(edges[1]).normalize_or_zero()],
        edges: edges.to_vec(),
    };
    crate::live_weld::penetration(geometry, &triangle) < -padding + 1.0e-4
}

fn terrain_clear(
    collider: &LocalCollider,
    pose: GpuTransform,
    padding: f32,
    terrain: &impl TerrainProbe,
) -> bool {
    terrain.collider_clear(collider, pose, padding)
}

/// Terrain samples follow collider faces, so a wide plate's width does not
/// become an artificial vertical clearance radius. Face interiors are sampled
/// at 5 cm intervals as well as edges and vertices.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
#[cfg(test)]
fn sampled_terrain_clear(
    collider: &LocalCollider,
    pose: GpuTransform,
    padding: f32,
    terrain: &impl TerrainProbe,
) -> bool {
    let (center, radius) = sphere(collider, pose);
    if terrain
        .penetration(center, radius + padding)
        .is_some_and(|depth| depth <= 1.0e-4)
    {
        return true;
    }
    let Ok(geometry) = crate::live_weld::geometry(collider, pose) else {
        return false;
    };
    let clear = |point| {
        terrain
            .penetration(point, padding)
            .is_some_and(|depth| depth <= 1.0e-4)
    };
    for normal in &geometry.normals {
        for normal in [*normal, -*normal] {
            let offset = geometry
                .vertices
                .iter()
                .map(|v| v.dot(normal))
                .fold(f32::NEG_INFINITY, f32::max);
            let mut face: Vec<_> = geometry
                .vertices
                .iter()
                .copied()
                .filter(|v| (v.dot(normal) - offset).abs() < 1.0e-4)
                .collect();
            if face.len() < 3 {
                continue;
            }
            let center = face.iter().copied().sum::<Vec3>() / face.len() as f32;
            let u = (face[0] - center).normalize();
            let v = normal.cross(u);
            face.sort_by(|a, b| {
                let angle = |point: Vec3| (point - center).dot(v).atan2((point - center).dot(u));
                angle(*a).total_cmp(&angle(*b))
            });
            for edge in 0..face.len() {
                let a = face[edge];
                let b = face[(edge + 1) % face.len()];
                let steps = (a
                    .distance(b)
                    .max(a.distance(center))
                    .max(b.distance(center))
                    / 0.05)
                    .ceil()
                    .max(1.0) as u32;
                for i in 0..=steps {
                    for j in 0..=steps - i {
                        let point = center
                            + (a - center) * (i as f32 / steps as f32)
                            + (b - center) * (j as f32 / steps as f32);
                        if !clear(point) {
                            return false;
                        }
                    }
                }
            }
        }
    }
    geometry.vertices.iter().copied().all(clear)
}

fn endpoint_clear(
    creation: &CompiledCreation,
    held: &[bool],
    poses: &[GpuTransform],
    obstacles: &[Obb],
    terrain: &impl TerrainProbe,
) -> bool {
    if !held_endpoint_clear(creation, held, poses) {
        return false;
    }
    creation
        .colliders
        .iter()
        .filter(|c| held[c.compound_index as usize])
        .all(|c| {
            let (center, radius) = sphere(c, poses[c.compound_index as usize]);
            terrain_clear(c, poses[c.compound_index as usize], 0.05, terrain)
                && obstacles
                    .iter()
                    .all(|&o| penetration(center, radius, o) <= 1.0e-4)
        })
}

/// Midpoint spheres inflated by a bound on translation and rotation cover the
/// complete subsegment. Existing conservative overlaps may only decrease.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn path_clear(
    creation: &CompiledCreation,
    held: &[bool],
    start: &[GpuTransform],
    end: &[GpuTransform],
    obstacles: &[Obb],
    terrain: &impl TerrainProbe,
) -> bool {
    if !held_path_clear(creation, held, start, end) {
        return false;
    }
    for c in creation
        .colliders
        .iter()
        .filter(|c| held[c.compound_index as usize])
    {
        let body = c.compound_index as usize;
        let a = start[body];
        let b = end[body];
        let qa = Quat::from_array(a.rotation);
        let qb = Quat::from_array(b.rotation);
        let rotating = qa.dot(qb).abs() < 1.0 - 1.0e-7;
        let travel =
            position(a).distance(position(b)) + qa.angle_between(qb) * c.local_center.length();
        let steps = (travel / 0.05).ceil().max(1.0) as u32;
        let margin = travel / (2.0 * steps as f32);
        let mut prior = sphere(c, a);
        let initially_clear = terrain_clear(c, a, 0.05, terrain);
        for step in 0..steps {
            let amount = (step as f32 + 0.5) / steps as f32;
            let midpoint = GpuTransform {
                position: position(a).lerp(position(b), amount).extend(0.0).to_array(),
                rotation: qa.slerp(qb, amount).to_array(),
            };
            let (center, radius) = sphere(c, midpoint);
            let Some(initial_depth) = terrain.penetration(prior.0, prior.1) else {
                return false;
            };
            let allowed = if rotating {
                0.0
            } else {
                initial_depth.max(0.0)
            };
            let terrain_safe = if initially_clear {
                terrain_clear(c, midpoint, 0.05 + margin, terrain)
            } else {
                terrain
                    .penetration(center, radius + margin)
                    .is_some_and(|depth| depth <= allowed + 1.0e-4)
            };
            if !terrain_safe
                || obstacles.iter().any(|&o| {
                    penetration(center, radius + margin, o)
                        > (if rotating {
                            0.0
                        } else {
                            penetration(prior.0, prior.1, o).max(0.0)
                        }) + 1.0e-4
                })
            {
                return false;
            }
            let amount = (step + 1) as f32 / steps as f32;
            prior = sphere(
                c,
                GpuTransform {
                    position: position(a).lerp(position(b), amount).extend(0.0).to_array(),
                    rotation: qa.slerp(qb, amount).to_array(),
                },
            );
        }
    }
    true
}

fn plan(
    creation: &CompiledCreation,
    held: &[bool],
    start: &[GpuTransform],
    end: &[GpuTransform],
    obstacles: &[Obb],
    terrain: &impl TerrainProbe,
) -> Option<VecDeque<Vec<GpuTransform>>> {
    if !endpoint_clear(creation, held, end, obstacles, terrain) {
        return None;
    }
    if path_clear(creation, held, start, end, obstacles, terrain) {
        return Some(VecDeque::from([end.to_vec()]));
    }
    // Lift without changing orientation before attempting to level the creation.
    let lift = start
        .iter()
        .zip(end)
        .zip(held)
        .filter(|(_, held)| **held)
        .map(|((a, b), _)| b.position[1] - a.position[1])
        .fold(0.0, f32::max);
    if lift > 20.0 {
        return None;
    }
    let raised = start
        .iter()
        .zip(held)
        .map(|(pose, &held)| {
            let mut pose = *pose;
            if held {
                pose.position[1] += lift;
            }
            pose
        })
        .collect::<Vec<_>>();
    (path_clear(creation, held, start, &raised, obstacles, terrain)
        && endpoint_clear(creation, held, &raised, obstacles, terrain)
        && path_clear(creation, held, &raised, end, obstacles, terrain))
    .then(|| VecDeque::from([raised, end.to_vec()]))
}

#[cfg(test)]
#[path = "freeze_tests.rs"]
mod tests;

pub(crate) fn weld_terrain_clear(
    world: &WorldRuntime,
    collider: &LocalCollider,
    pose: GpuTransform,
) -> bool {
    TerrainProbe::collider_clear(world, collider, pose, -0.001)
}
