//! Proximity alignment, controller activation, and rigid publication of live couplers.

use crate::simulation::state::AppSimulation;
use bevy::prelude::*;
use mechanic_core::{ConstructionGraph, GRID_UNIT_METERS, PartId, PartSpec, TICK_SECONDS_F32};
use mechanic_gpu::GpuExternalImpulse;
use std::collections::BTreeSet;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_8, PI};

const CAPTURE_DISTANCE: f32 = 0.07;
const RELEASE_DISTANCE: f32 = 0.12;
const LOCK_SECONDS: f32 = 1.2;

#[derive(Clone, Copy)]
struct End {
    part: PartId,
    body: u32,
    center: Vec3,
    rotation: Quat,
    velocity: Vec3,
    angular: Vec3,
    mass: f32,
    inertia: f32,
}

impl End {
    fn face(self) -> Vec3 {
        self.center + self.rotation * Vec3::Y * (GRID_UNIT_METERS * 0.25)
    }
}

#[derive(Clone, Copy)]
struct Pair {
    first: PartId,
    second: PartId,
    quarter: f32,
    gripping: Option<u64>,
}

impl Pair {
    fn target(self, first: End, tick: u64) -> Quat {
        let progress = self
            .gripping
            .map_or(0.0, |start| elapsed(tick, start) / LOCK_SECONDS)
            .clamp(0.0, 1.0);
        let smooth = progress * progress * (3.0 - 2.0 * progress);
        first.rotation
            * Quat::from_rotation_y(self.quarter + FRAC_PI_8 * (1.0 - smooth))
            * Quat::from_rotation_x(PI)
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "short differences of simulation ticks"
)]
fn elapsed(tick: u64, start: u64) -> f32 {
    tick.saturating_sub(start) as f32 * TICK_SECONDS_F32
}

#[derive(Resource, Default)]
pub(crate) struct Couplers {
    pairs: Vec<Pair>,
    parts: Vec<PartId>,
    pending: Option<(PartId, PartId)>,
    revision: Option<(u64, u64)>,
}

impl Couplers {
    fn track(&mut self, ends: &[End], graph: &ConstructionGraph, tick: u64) -> bool {
        let end = |part| ends.iter().copied().find(|e| e.part == part);
        let mut acquired = false;
        self.pairs.retain(|pair| {
            end(pair.first)
                .zip(end(pair.second))
                .is_some_and(|(a, b)| compatible(a, b, RELEASE_DISTANCE))
                && pair.gripping.is_none_or(|start| elapsed(tick, start) < 8.0)
        });
        let mut occupied: BTreeSet<_> = self
            .pairs
            .iter()
            .flat_map(|p| [p.first, p.second])
            .collect();
        for (index, &a) in ends.iter().enumerate() {
            if occupied.contains(&a.part) {
                continue;
            }
            for &b in &ends[index + 1..] {
                if occupied.contains(&b.part) || !compatible(a, b, CAPTURE_DISTANCE) {
                    continue;
                }
                // Couplers join separate creations; closing an articulated loop is a weld-tool operation.
                if graph
                    .structural_component(a.part, [])
                    .is_ok_and(|c| c.contains(b.part))
                {
                    continue;
                }
                let (a, b) = if graph
                    .structural_component(b.part, [])
                    .is_ok_and(|c| c.touches_authored_ground())
                {
                    (b, a)
                } else {
                    (a, b)
                };
                self.pairs.push(Pair {
                    first: a.part,
                    second: b.part,
                    quarter: nearest_quarter(a, b),
                    gripping: None,
                });
                occupied.extend([a.part, b.part]);
                acquired = true;
                break;
            }
        }
        acquired
    }
}

fn endpoints(simulation: &AppSimulation, parts: &[PartId]) -> Vec<End> {
    let Some(creation) = &simulation.creation else {
        return Vec::new();
    };
    let Some(live) = &simulation.live_state else {
        return Vec::new();
    };
    parts
        .iter()
        .copied()
        .filter_map(|part| {
            if !matches!(
                simulation.effective_graph().part(part),
                Some(PartSpec::Coupler(_))
            ) {
                return None;
            }
            let body = creation
                .part_to_compound
                .iter()
                .find_map(|&(id, body)| (id == part).then_some(body))?;
            let compound = &creation.compounds[body as usize];
            let pose = live.transforms.get(body as usize)?;
            let velocity = live.velocities.get(body as usize)?;
            let motion =
                crate::live_weld::world_from_build(creation, &live.transforms, body as usize)
                    .ok()?;
            let graph = simulation.effective_graph();
            let center = motion.point(graph.part_position(part)?);
            let rotation = motion.rotation() * graph.part_rotation(part)?;
            let angular = Vec3::from_slice(&velocity.angular[..3]);
            let linear = Vec3::from_slice(&velocity.linear[..3]);
            let properties = &compound.mass_properties;
            Some(End {
                part,
                body,
                center,
                rotation,
                angular,
                velocity: linear + angular.cross(center - Vec3::from_slice(&pose.position[..3])),
                mass: if compound.is_static {
                    0.0
                } else {
                    properties.mass
                },
                inertia: if compound.is_static {
                    0.0
                } else {
                    (properties.inertia.x_axis.x
                        + properties.inertia.y_axis.y
                        + properties.inertia.z_axis.z)
                        / 3.0
                },
            })
        })
        .collect()
}

fn compatible(a: End, b: End, distance: f32) -> bool {
    a.body != b.body
        && (a.mass > 0.0 || b.mass > 0.0)
        && a.face().distance(b.face()) <= distance
        && (a.rotation * Vec3::Y).dot(b.rotation * Vec3::Y) < -0.85
}

fn nearest_quarter(a: End, b: End) -> f32 {
    let relative = a.rotation.inverse() * b.rotation * Quat::from_rotation_x(-PI);
    let direction = relative * Vec3::X;
    ((-direction.z).atan2(direction.x) / FRAC_PI_2).round() * FRAC_PI_2
}

fn activated(
    graph: &ConstructionGraph,
    keys: &mechanic_core::ControllerKeys,
    part: PartId,
) -> bool {
    graph
        .input_configuration(part)
        .and_then(|c| c.controller.zip(c.key))
        .is_some_and(|(controller, key)| keys.pressed(controller, key))
}

#[expect(
    clippy::too_many_arguments,
    reason = "coupler publication uses editor history and authoritative physics"
)]
pub(crate) fn update(
    mut couplers: ResMut<Couplers>,
    simulation: Res<AppSimulation>,
    controls: Res<crate::physical_controls::PhysicalControls>,
    frozen: Res<crate::freeze::DimensionFreeze>,
    pause: Res<crate::pause_menu::PauseMenuState>,
    mut graph: ResMut<crate::editor::state::EditorGraph>,
    mut state: ResMut<crate::editor::state::EditorState>,
    mut history: ResMut<crate::editor::history::EditorHistory>,
) {
    if !simulation.is_running() {
        *couplers = Couplers::default();
        return;
    }
    if simulation.world_revision != couplers.revision {
        couplers.pairs.clear();
        couplers.revision = simulation.world_revision;
        couplers.parts = simulation
            .effective_graph()
            .parts()
            .filter_map(|(id, spec)| matches!(spec, PartSpec::Coupler(_)).then_some(id))
            .collect();
    }
    if pause.blocks_world_input() {
        return;
    }
    if let Some((first, second)) = couplers.pending {
        let linked = |graph: &ConstructionGraph| {
            graph.rigid_links().any(|(_, link)| {
                (link.first == first && link.second == second)
                    || (link.first == second && link.second == first)
            })
        };
        if linked(&simulation.published_graph) {
            couplers.pending = None;
            couplers.pairs.clear();
            state.feedback = Some("Couplers locked into one rigid creation".into());
        } else if linked(&graph.0) {
            // Keep the grip while asynchronous compilation prepares the merge.
            return;
        } else {
            couplers.pending = None;
            couplers.pairs.clear();
        }
    }
    if !graph.0.shares_revision(&simulation.published_graph) {
        couplers.pairs.clear();
        return;
    }
    let ends: Vec<_> = endpoints(&simulation, &couplers.parts)
        .into_iter()
        .filter(|end| !frozen.holds_body(end.body))
        .collect();
    let end = |part| ends.iter().copied().find(|e| e.part == part);
    if couplers.track(&ends, &graph.0, simulation.completed_tick) {
        state.feedback =
            Some("Couplers aligning — press an assigned Controller key to grip".into());
    }
    for pair in &mut couplers.pairs {
        let (Some(a), Some(b)) = (end(pair.first), end(pair.second)) else {
            continue;
        };
        if pair.gripping.is_none()
            && (activated(simulation.effective_graph(), &controls.keys, a.part)
                || activated(simulation.effective_graph(), &controls.keys, b.part))
        {
            pair.gripping = Some(simulation.completed_tick);
            state.feedback = Some("Coupler gripping — twisting to lock".into());
        }
        if pair
            .gripping
            .is_some_and(|start| elapsed(simulation.completed_tick, start) >= LOCK_SECONDS)
            && a.face().distance(b.face()) < 0.0005
            && pair
                .target(a, simulation.completed_tick)
                .angle_between(b.rotation)
                < 0.002
            && (a.velocity - b.velocity).length() < 0.03
        {
            match crate::live_weld::stage_coupler(&graph.0, &simulation, pair.first, pair.second) {
                Ok(staged) => {
                    let previous =
                        crate::editor::history::EditorSnapshot::capture(&graph.0, &state);
                    graph.0 = staged;
                    history.commit(previous);
                    state.construction_mesh_dirty = true;
                    state.feedback = Some("Couplers gripping — publishing rigid lock".into());
                    let pending = (pair.first, pair.second);
                    couplers.pending = Some(pending);
                    return;
                }
                Err(error) => state.feedback = Some(format!("Coupler cannot lock: {error}")),
            }
        }
    }
}

fn reduced(a: f32, b: f32) -> f32 {
    match (a > 0.0, b > 0.0) {
        (true, true) => a * b / (a + b),
        (true, false) => a,
        (false, true) => b,
        (false, false) => 0.0,
    }
}

/// Equal and opposite attraction and torque, applied by either physics backend each tick.
pub(crate) fn impulses(
    couplers: &Couplers,
    simulation: &AppSimulation,
    tick: u64,
) -> Vec<GpuExternalImpulse> {
    if couplers.pairs.is_empty() {
        return Vec::new();
    }
    let ends = endpoints(simulation, &couplers.parts);
    let mut impulses = Vec::new();
    for &pair in &couplers.pairs {
        let end = |part| ends.iter().copied().find(|e| e.part == part);
        let Some((a, b)) = end(pair.first).zip(end(pair.second)) else {
            continue;
        };
        if !compatible(a, b, RELEASE_DISTANCE) {
            continue;
        }
        let separation = b.face() - a.face();
        let force = (separation * 36.0 + (b.velocity - a.velocity) * 12.0).clamp_length_max(4.0)
            * reduced(a.mass, b.mass)
            * TICK_SECONDS_F32;
        impulses.push(GpuExternalImpulse::new(a.body, a.center, force));
        impulses.push(GpuExternalImpulse::new(b.body, b.center, -force));
        let mut delta = pair.target(a, tick) * b.rotation.inverse();
        if delta.w < 0.0 {
            delta = -delta;
        }
        let torque = (delta.to_scaled_axis() * 36.0 + (a.angular - b.angular) * 12.0)
            .clamp_length_max(4.0)
            * reduced(a.inertia, b.inertia)
            * TICK_SECONDS_F32;
        // A force couple contributes pure torque without changing linear momentum.
        for (end, torque) in [(a, -torque), (b, torque)] {
            if torque.length_squared() < 1e-16 {
                continue;
            }
            let arm = torque.normalize().any_orthonormal_vector() * (GRID_UNIT_METERS * 0.5);
            let force = torque.cross(arm) / (2.0 * arm.length_squared());
            impulses.push(GpuExternalImpulse::new(end.body, end.center + arm, force));
            impulses.push(GpuExternalImpulse::new(end.body, end.center - arm, -force));
        }
    }
    impulses
}

#[cfg(test)]
mod tests;
