//! Independent suspension placement through the common bearing socket workflow.
use super::{
    BearingDimensions, BearingSocket, BuildCommand, ButtonInput, ConstructionGraph,
    ConstructionMaterial, CylinderDimensions, EditorHistory, EditorSnapshot, EditorState,
    FaceOwner, GameAction, MaterialAppearance, PlacedBearing, PlacementError, Tool,
    bearing_anchor_from_hit_with_grid, bearing_location_occupied, bearing_socket_targets,
    bearing_uses_socket, builder, suspension_render, try_face_geometry_from_ref,
};
use mechanic_core::{BumpStopSpec, ShockSpec, SpringSpec, SuspensionSpec};

/// Imported creations may describe attached assemblies only as graph bearings.
/// Expose their source sockets to the same picking and editing path as new hardware.
pub(crate) fn sync_sockets(graph: &ConstructionGraph, state: &mut EditorState) {
    for (_, bearing) in graph.bearings() {
        if !matches!(bearing.kind, mechanic_core::BearingKind::Suspension(_)) {
            continue;
        }
        let socket = PlacedBearing {
            source: bearing.source,
            anchor: bearing.shared_anchor,
            axis: bearing.axis,
            dimensions: bearing.dimensions,
            kind: bearing.kind,
        };
        if let Some(existing) = state.placed_bearings.iter_mut().find(|s| {
            crate::suspension_controls::Target(**s) == crate::suspension_controls::Target(socket)
        }) {
            *existing = socket;
        } else {
            state.placed_bearings.push(socket);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct SuspensionToolState {
    pub drag: Option<SuspensionDrag>,
    pub controls: crate::suspension_controls::Controls,
    pub spring: SpringSpec,
    pub shock: ShockSpec,
    pub stop: BumpStopSpec,
    pub picked_component: Option<(usize, usize)>,
    pub preview: Option<PlacedBearing>,
    pub insertion: Option<usize>,
    pub attachment: Option<PlacedBearing>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SuspensionDrag {
    socket: PlacedBearing,
    insertion: Option<PlacedBearing>,
    direction: bevy::math::Vec3,
    spring: SpringSpec,
    shock: ShockSpec,
    stop: BumpStopSpec,
    mode: usize,
}

pub(super) fn cycle_drag(state: &mut EditorState, tool: Tool) -> Option<String> {
    let drag = state.suspension.drag.as_mut()?;
    drag.mode = (drag.mode + 1) % if tool == Tool::Spring { 4 } else { 2 };
    drag.direction = state.pointer_ray?.1;
    drag.spring = state.suspension.spring;
    drag.shock = state.suspension.shock;
    drag.stop = state.suspension.stop;
    Some(format!(
        "Suspension: {} — move mouse; release to place",
        dimension_label(drag.mode)
    ))
}

fn dimension_label(mode: usize) -> &'static str {
    ["length", "outer diameter", "inner diameter", "coil count"][mode]
}

fn placement_parameter(tool: Tool, mode: usize) -> crate::suspension_controls::Parameter {
    use crate::suspension_controls::Parameter as P;
    match tool {
        Tool::Spring => [P::SpringLength, P::SpringOd, P::SpringId, P::Coils][mode],
        Tool::Shock => [P::ShockLength, P::ShockOd][mode],
        _ => [P::StopLength, P::StopOd][mode],
    }
}
fn resized(
    drag: &SuspensionDrag,
    tool: Tool,
    steps: f32,
) -> Result<(SpringSpec, ShockSpec, BumpStopSpec), String> {
    let original = match tool {
        Tool::Spring => SuspensionSpec::new(Some(drag.spring), None, None),
        Tool::Shock => SuspensionSpec::new(None, Some(drag.shock), None),
        _ => SuspensionSpec::new(None, Some(drag.shock), Some(drag.stop)),
    }
    .map_err(|e| e.to_string())?;
    let parameter = placement_parameter(tool, drag.mode);
    let spec = parameter.edit(
        original,
        parameter.value(original) + steps * parameter.step(),
        false,
    )?;
    Ok((
        spec.spring().unwrap_or(drag.spring),
        spec.shock().unwrap_or(drag.shock),
        spec.bump_stop().unwrap_or(drag.stop),
    ))
}

pub(super) fn drag_actions(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    actions: &ButtonInput<GameAction>,
) {
    if actions.just_pressed(GameAction::Secondary) {
        state.suspension.drag = None;
        state.suspension.preview = None;
        return;
    }
    if actions.just_pressed(GameAction::Primary)
        && state.preview_error.is_none()
        && let Some(socket) = state.suspension.preview
        && let Some((_, direction)) = state.pointer_ray
    {
        if let mechanic_core::BearingKind::Suspension(spec) = socket.kind {
            if let Some(spring) = spec.spring() {
                state.suspension.spring = spring;
            }
            if let Some(shock) = spec.shock() {
                state.suspension.shock = shock;
            }
            if let Some(stop) = spec.bump_stop() {
                state.suspension.stop = stop;
            }
        }
        state.suspension.drag = Some(SuspensionDrag {
            socket,
            insertion: state
                .suspension
                .insertion
                .and_then(|i| state.placed_bearings.get(i).copied()),
            direction,
            spring: state.suspension.spring,
            shock: state.suspension.shock,
            stop: state.suspension.stop,
            mode: 0,
        });
        state.feedback = Some("Drag length; R cycles dimensions; release to place".into());
    }
    if actions.just_released(GameAction::Primary) && state.suspension.drag.take().is_some() {
        place(graph, state, history);
    }
}

fn assembly(
    tool: Tool,
    state: &EditorState,
    host: Option<SuspensionSpec>,
) -> Result<SuspensionSpec, String> {
    let spring = if tool == Tool::Spring {
        let s = state.suspension.spring;
        let length = host.map_or(s.length(), SuspensionSpec::extended_length);
        let id = host
            .and_then(SuspensionSpec::shock)
            .map_or(s.id(), |shock| {
                s.id()
                    .max(((shock.hardware_od() + 0.004) * 400.0).ceil() / 400.0)
            });
        let od = s.od().max(id + 2.0 * s.wire());
        Some(SpringSpec::new(length, od, id, s.coils(), s.preload()).map_err(|e| e.to_string())?)
    } else {
        host.and_then(SuspensionSpec::spring)
    };
    let shock = if tool == Tool::Shock {
        let s = state.suspension.shock;
        let length = host.map_or(s.length(), SuspensionSpec::extended_length);
        let od = host
            .and_then(SuspensionSpec::spring)
            .map_or(s.od(), |spring| {
                s.od()
                    .min((((spring.id() - 0.004) / 1.1) * 400.0).floor() / 400.0)
            });
        let base = ShockSpec::new(
            s.length(),
            s.od(),
            s.body_end(),
            s.starting_compression(),
            1.0,
            1.0,
        )
        .map_err(|e| e.to_string())?;
        Some(
            ShockSpec::new(
                length,
                od,
                s.body_end(),
                s.starting_compression(),
                s.damping(true) / base.damping(true),
                s.damping(false) / base.damping(false),
            )
            .map_err(|e| e.to_string())?,
        )
    } else {
        host.and_then(SuspensionSpec::shock)
    };
    host.map_or_else(
        || SuspensionSpec::new(spring, shock, None),
        |host| host.with_components(spring, shock, host.bump_stop(), false),
    )
    .map_err(|e| e.to_string())
}

#[expect(
    clippy::too_many_lines,
    reason = "one preview transaction covers component insertion and opposite-plate construction"
)]
pub(super) fn refresh(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    tool: Tool,
    material: ConstructionMaterial,
    cylinder_dimensions: CylinderDimensions,
) -> bool {
    if let Some(drag) = state.suspension.drag {
        if drag.insertion.is_some_and(|host| {
            !crate::suspension_controls::Target(host)
                .resolve(state)
                .is_some_and(|i| state.placed_bearings[i].kind == host.kind)
        }) {
            state.suspension.drag = None;
            state.suspension.preview = None;
            state.suspension.controls.consume_until_release = true;
            state.feedback = Some("Suspension changed; placement cancelled".into());
            return true;
        }
        let steps = state.pointer_ray.map_or(0.0, |(_, ray)| {
            (super::pipe_pointer_delta(drag.direction, ray) / 0.005).round()
        });
        let result = resized(&drag, tool, steps).and_then(|(spring, shock, stop)| {
            state.suspension.spring = spring;
            state.suspension.shock = shock;
            state.suspension.stop = stop;
            let host = drag
                .insertion
                .and_then(|socket| crate::suspension_controls::Target(socket).resolve(state))
                .and_then(|i| state.placed_bearings.get(i))
                .and_then(|s| {
                    if let mechanic_core::BearingKind::Suspension(spec) = s.kind {
                        Some(spec)
                    } else {
                        None
                    }
                });
            if drag.insertion.is_some() && tool != Tool::Cylinder && drag.mode == 0 && steps != 0.0
            {
                return Err(
                    "Shared mount length is fixed; edit the installed assembly with Connector"
                        .into(),
                );
            }
            let replacement = if let Some(host) = host {
                host.with_components(
                    if tool == Tool::Spring {
                        Some(spring)
                    } else {
                        host.spring()
                    },
                    if tool == Tool::Shock {
                        Some(shock)
                    } else {
                        host.shock()
                    },
                    if tool == Tool::Cylinder {
                        Some(stop)
                    } else {
                        host.bump_stop()
                    },
                    !bearing_socket_targets(graph, drag.socket).is_empty(),
                )
                .map_err(|e| e.to_string())?
            } else {
                assembly(tool, state, None)?
            };
            if let Some(host) = host {
                host.validate_edit(
                    replacement,
                    !bearing_socket_targets(graph, drag.socket).is_empty(),
                )
                .map_err(|e| e.to_string())?;
            }
            Ok(replacement)
        });
        state.suspension.insertion = drag
            .insertion
            .and_then(|socket| crate::suspension_controls::Target(socket).resolve(state));
        match result {
            Ok(spec) => {
                let socket = PlacedBearing {
                    kind: mechanic_core::BearingKind::Suspension(spec),
                    ..drag.socket
                };
                let half = spec.plates().diameter / 2.0;
                let center = socket.anchor + socket.axis * spec.extended_length() / 2.0;
                let extent = socket.axis.abs() * (spec.extended_length() / 2.0)
                    + (bevy::math::Vec3::ONE - socket.axis.abs()) * half;
                state.preview_error = builder::validate_world_bounds(
                    center - extent,
                    center + extent,
                    state.placement_bounds,
                )
                .err();
                state.suspension.preview = Some(socket);
                let dimensions = if tool == Tool::Spring {
                    let spring = spec.spring().expect("spring tool");
                    format!(
                        "L {:.1}, OD {:.1}, ID {:.1} mm, {} coils",
                        spring.length() * 1000.0,
                        spring.od() * 1000.0,
                        spring.id() * 1000.0,
                        spring.coils()
                    )
                } else if tool == Tool::Cylinder {
                    let stop = spec.bump_stop().expect("rubber tool");
                    format!(
                        "L {:.1}, OD {:.1} mm",
                        stop.length() * 1000.0,
                        stop.od() * 1000.0
                    )
                } else {
                    let shock = spec.shock().expect("shock tool");
                    format!(
                        "L {:.1}, OD {:.1} mm",
                        shock.length() * 1000.0,
                        shock.od() * 1000.0
                    )
                };
                state.feedback = Some(format!(
                    "{}: {dimensions} — R cycles; release to place",
                    dimension_label(drag.mode)
                ));
            }
            Err(message) => {
                state.preview_error = Some(PlacementError::Graph(message.clone()));
                state.feedback = Some(message);
            }
        }
        return true;
    }
    state.suspension.preview = None;
    state.suspension.insertion = None;
    state.suspension.attachment = None;
    let host = state
        .hovered_bearing
        .and_then(|i| state.placed_bearings.get(i).copied().map(|s| (i, s)))
        .filter(|(_, s)| matches!(s.kind, mechanic_core::BearingKind::Suspension(_)));
    let rubber_target = tool == Tool::Cylinder && material == ConstructionMaterial::Rubber
        && host.is_some_and(|(_,s)| matches!(s.kind, mechanic_core::BearingKind::Suspension(spec) if spec.shock().is_some()));
    if let Some((_, socket)) = host
        && matches!(tool, Tool::Block | Tool::Cylinder)
        && !rubber_target
    {
        state.suspension.attachment = Some(socket);
        state.preview_error = None;
        if tool == Tool::Block {
            match builder::suspension_block_candidate(socket) {
                Ok(mut candidate) => {
                    candidate.spec = candidate.spec.with_material(material);
                    state.preview = Some(candidate);
                }
                Err(e) => state.preview_error = Some(e),
            }
        } else {
            match builder::suspension_cylinder_candidate(socket, cylinder_dimensions) {
                Ok(mut candidate) => {
                    candidate.spec = candidate.spec.with_material(material);
                    state.cylinder_preview = Some(candidate);
                }
                Err(e) => state.preview_error = Some(e),
            }
        }
        return true;
    }
    if !(matches!(tool, Tool::Spring | Tool::Shock) || rubber_target) {
        return false;
    }
    let result = (|| -> Result<PlacedBearing, String> {
        if let Some((index, socket)) = host {
            let mechanic_core::BearingKind::Suspension(spec) = socket.kind else {
                unreachable!()
            };
            state.suspension.insertion = Some(index);
            let replacement = if tool == Tool::Cylinder {
                let stop = state.suspension.stop;
                spec.with_components(
                    spec.spring(),
                    spec.shock(),
                    Some(stop),
                    !bearing_socket_targets(graph, socket).is_empty(),
                )
                .map_err(|e| e.to_string())?
            } else {
                if (tool == Tool::Spring && spec.spring().is_some())
                    || (tool == Tool::Shock && spec.shock().is_some())
                {
                    return Err(
                        "This component is already installed; equip Connector to edit it".into(),
                    );
                }
                assembly(tool, state, Some(spec))?
            };
            spec.validate_edit(
                replacement,
                !bearing_socket_targets(graph, socket).is_empty(),
            )
            .map_err(|e| e.to_string())?;
            return Ok(PlacedBearing {
                kind: mechanic_core::BearingKind::Suspension(replacement),
                ..socket
            });
        }
        let hit = state.hovered.ok_or("Point at a flat construction face")?;
        if !matches!(hit.face.owner, FaceOwner::Part(_)) {
            return Err("Mount suspension on construction".into());
        }
        let face = try_face_geometry_from_ref(hit.face, Some(graph))
            .ok_or("Choose a flat construction face")?;
        let anchor = bearing_anchor_from_hit_with_grid(
            graph,
            hit,
            state.placement_grid,
            state.placement_bounds,
        )
        .map_err(|e| e.to_string())?;
        if bearing_location_occupied(graph, &state.placed_bearings, hit.face, anchor) {
            return Err("A mounting assembly already occupies this point".into());
        }
        let spec = assembly(tool, state, None)?;
        let dimensions =
            BearingDimensions::new(spec.plates().diameter, 0.0).map_err(|e| e.to_string())?;
        Ok(PlacedBearing {
            source: hit.face,
            anchor,
            axis: face.normal,
            dimensions,
            kind: mechanic_core::BearingKind::Suspension(spec),
        })
    })();
    match result {
        Ok(mut socket) => {
            let mechanic_core::BearingKind::Suspension(spec) = socket.kind else {
                unreachable!()
            };
            socket.dimensions =
                BearingDimensions::new(spec.plates().diameter, 0.0).expect("validated plates");
            let half = spec.plates().diameter / 2.0;
            let center = socket.anchor + socket.axis * spec.extended_length() / 2.0;
            let axis = socket.axis.abs();
            let extent =
                axis * (spec.extended_length() / 2.0) + (bevy::math::Vec3::ONE - axis) * half;
            state.preview_error = builder::validate_world_bounds(
                center - extent,
                center + extent,
                state.placement_bounds,
            )
            .err();
            state.suspension.preview = Some(socket);
        }
        Err(message) => {
            state.preview_error = Some(PlacementError::Graph(message.clone()));
            state.feedback = Some(message);
        }
    }
    true
}

pub(super) fn place(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
) {
    if state.preview_error.is_some() {
        return;
    }
    let Some(mut socket) = state.suspension.preview else {
        return;
    };
    if let mechanic_core::BearingKind::Suspension(spec) = socket.kind {
        socket.dimensions =
            BearingDimensions::new(spec.plates().diameter, 0.0).expect("validated mount plates");
    }
    let previous = EditorSnapshot::capture(graph, state);
    if let Some(index) = state.suspension.insertion {
        let old = state.placed_bearings[index];
        let joint = graph
            .bearings()
            .find(|(_, b)| bearing_uses_socket(b, old))
            .map(|(id, _)| id);
        if let Some(bearing) = joint {
            let mechanic_core::BearingKind::Suspension(spec) = socket.kind else {
                return;
            };
            if let Err(e) = graph.apply(BuildCommand::SetSuspension { bearing, spec }) {
                state.feedback = Some(e.to_string());
                return;
            }
        }
        state.placed_bearings[index] = socket;
    } else {
        state.placed_bearings.push(socket);
    }
    history.commit(previous);
    state.construction_mesh_dirty = true;
    state.suspension.preview = None;
    state.feedback = Some("Suspension placed — attach construction to the opposite plate".into());
}

pub(super) fn attach(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    tool: Tool,
    actions: &ButtonInput<GameAction>,
) -> bool {
    let Some(socket) = state.suspension.attachment else {
        return false;
    };
    if actions.just_pressed(GameAction::Primary) {
        let previous = EditorSnapshot::capture(graph, state);
        let targets = bearing_socket_targets(graph, socket);
        let result = if tool == Tool::Block {
            state
                .preview
                .ok_or(PlacementError::BearingOutsideFace)
                .and_then(|candidate| {
                    builder::stage_suspension_block(
                        graph,
                        socket,
                        candidate,
                        &targets,
                        state.placement_bounds,
                    )
                })
        } else {
            state
                .cylinder_preview
                .ok_or(PlacementError::BearingOutsideFace)
                .and_then(|candidate| {
                    builder::stage_suspension_cylinder(
                        graph,
                        socket,
                        candidate,
                        &targets,
                        state.placement_bounds,
                    )
                })
        };
        match result {
            Ok(staged) => {
                *graph = staged;
                history.commit(previous);
                state.construction_mesh_dirty = true;
            }
            Err(e) => state.feedback = Some(e.to_string()),
        }
    }
    true
}

pub(crate) fn apply_settings(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    index: usize,
    replacement: SuspensionSpec,
) {
    let Some(socket) = state.placed_bearings.get(index).copied() else {
        return;
    };
    let mechanic_core::BearingKind::Suspension(current) = socket.kind else {
        return;
    };
    let replacement = match current.with_components(
        replacement.spring(),
        replacement.shock(),
        replacement.bump_stop(),
        !bearing_socket_targets(graph, socket).is_empty(),
    ) {
        Ok(s) => s,
        Err(e) => {
            state.feedback = Some(e.to_string());
            return;
        }
    };
    let half = replacement.plates().diameter / 2.0;
    let center = socket.anchor + socket.axis * replacement.extended_length() / 2.0;
    let axis = socket.axis.abs();
    let extent = axis * replacement.extended_length() / 2.0 + (bevy::math::Vec3::ONE - axis) * half;
    if let Err(e) =
        builder::validate_world_bounds(center - extent, center + extent, state.placement_bounds)
    {
        state.feedback = Some(e.to_string());
        return;
    }
    state.suspension.preview = Some(PlacedBearing {
        kind: mechanic_core::BearingKind::Suspension(replacement),
        ..socket
    });
    state.suspension.insertion = Some(index);
    state.preview_error = None;
    place(graph, state, history);
    if state.suspension.preview.is_none() {
        state.feedback = Some("Suspension updated".into());
    }
}

fn selected_component(
    graph: &ConstructionGraph,
    state: &EditorState,
    socket: PlacedBearing,
) -> usize {
    let mechanic_core::BearingKind::Suspension(spec) = socket.kind else {
        return 0;
    };
    if let Some((index, component)) = state.suspension.picked_component
        && state
            .placed_bearings
            .get(index)
            .is_some_and(|picked| picked.source == socket.source && picked.kind == socket.kind)
    {
        return component;
    }

    let owner = state
        .pointer_ray
        .and_then(|(origin, direction)| {
            suspension_render::raycast_scene_component(graph, None, &[socket], origin, direction)
        })
        .map(|(_, _, owner)| owner);
    match owner {
        Some(mechanic_core::SuspensionMeshOwner::Spring) => 0,
        Some(mechanic_core::SuspensionMeshOwner::BumpStop) => 2,
        _ => usize::from(spec.shock().is_some()),
    }
}

pub(super) fn remove_component(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    index: usize,
) -> bool {
    let Some(socket) = state.placed_bearings.get(index).copied() else {
        return false;
    };
    let mechanic_core::BearingKind::Suspension(spec) = socket.kind else {
        return false;
    };
    let replacement = match selected_component(graph, state, socket) {
        0 => spec.without_spring(),
        2 => spec
            .with_components(spec.spring(), spec.shock(), None, true)
            .map(Some),
        _ => spec.without_shock(),
    };
    match replacement {
        Ok(Some(spec)) => {
            state.suspension.preview = Some(PlacedBearing {
                kind: mechanic_core::BearingKind::Suspension(spec),
                ..socket
            });
            state.suspension.insertion = Some(index);
            state.preview_error = None;
            place(graph, state, history);
            true
        }
        Ok(None) => false,
        Err(e) => {
            state.feedback = Some(e.to_string());
            true
        }
    }
}

pub(super) fn paint(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    brush: MaterialAppearance,
    actions: &ButtonInput<GameAction>,
) -> bool {
    let Some(index) = state.hovered_bearing else {
        return false;
    };
    let Some(socket) = state.placed_bearings.get(index).copied() else {
        return false;
    };
    let mechanic_core::BearingKind::Suspension(spec) = socket.kind else {
        return false;
    };
    if actions.just_pressed(GameAction::Primary) || actions.just_pressed(GameAction::Secondary) {
        let mut appearances = spec.appearances();
        appearances[selected_component(graph, state, socket)] =
            if actions.just_pressed(GameAction::Secondary) {
                MaterialAppearance::BAKED
            } else {
                brush
            };
        state.suspension.preview = Some(PlacedBearing {
            kind: mechanic_core::BearingKind::Suspension(spec.with_appearances(appearances)),
            ..socket
        });
        state.suspension.insertion = Some(index);
        state.preview_error = None;
        place(graph, state, history);
    }
    true
}

pub(super) fn sockets(placed: &[PlacedBearing]) -> Vec<BearingSocket> {
    placed
        .iter()
        .map(|s| BearingSocket {
            kind: s.kind,
            axis: s.axis,
            source: s.source,
            anchor: s.anchor,
            dimensions: s.dimensions,
        })
        .collect()
}

pub(super) fn component_index(
    spec: SuspensionSpec,
    owner: mechanic_core::SuspensionMeshOwner,
) -> usize {
    match owner {
        mechanic_core::SuspensionMeshOwner::Spring => 0,
        mechanic_core::SuspensionMeshOwner::BumpStop => 2,
        _ => usize::from(spec.shock().is_some()),
    }
}
pub(super) fn sample_appearance(
    graph: &ConstructionGraph,
    state: &EditorState,
) -> Option<MaterialAppearance> {
    let socket = *state.placed_bearings.get(state.hovered_bearing?)?;
    let mechanic_core::BearingKind::Suspension(spec) = socket.kind else {
        return None;
    };
    Some(spec.appearances()[selected_component(graph, state, socket)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::{IVec3, Vec3};
    use mechanic_core::{
        BearingKind, BuildOutcome, BuildPose, CuboidSpec, FaceKind, FaceRef, GridRotation,
    };

    fn fixture(spec: SuspensionSpec) -> (ConstructionGraph, EditorState) {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(base) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 1, 4],
                    BuildPose::from_position_ticks(IVec3::new(0, 50, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            panic!("base")
        };
        let socket = PlacedBearing {
            source: FaceRef::part(base, FaceKind::PositiveY),
            anchor: Vec3::Y * 0.25,
            axis: Vec3::Y,
            dimensions: BearingDimensions::new(spec.plates().diameter, 0.0).unwrap(),
            kind: BearingKind::Suspension(spec),
        };
        let state = EditorState {
            placed_bearings: vec![socket],
            hovered_bearing: Some(0),
            ..Default::default()
        };
        (graph, state)
    }
    #[test]
    fn drag_resizes_validated_dimensions_and_rejects_impossible_spring() {
        let spec = SuspensionSpec::new(Some(SpringSpec::default()), None, None).unwrap();
        let (_, state) = fixture(spec);
        let mut drag = SuspensionDrag {
            socket: state.placed_bearings[0],
            insertion: None,
            direction: Vec3::Z,
            spring: SpringSpec::default(),
            shock: ShockSpec::default(),
            stop: BumpStopSpec::default(),
            mode: 0,
        };
        let (spring, _, _) = resized(&drag, Tool::Spring, 1.0).unwrap();
        assert!((spring.length() - 0.5025).abs() < 1e-6);
        drag.mode = 2;
        assert!(resized(&drag, Tool::Spring, 100.0).is_err());
        drag.mode = 3;
        assert_eq!(resized(&drag, Tool::Spring, 1.0).unwrap().0.coils(), 7);
        drag.mode = 1;
        let shock = resized(&drag, Tool::Shock, 1.0).unwrap().1;
        assert!((shock.od() - 0.1025).abs() < 1e-6);
        assert!((shock.damping(false) / shock.damping(true) - 1.6).abs() < 1e-6);
    }

    #[test]
    fn drag_places_on_release_and_secondary_cancels() {
        let spec = SuspensionSpec::new(Some(SpringSpec::default()), None, None).unwrap();
        let (mut graph, mut state) = fixture(spec);
        state.suspension.preview = Some(state.placed_bearings.remove(0));
        state.pointer_ray = Some((Vec3::ZERO, Vec3::Z));
        let mut history = EditorHistory::default();
        let mut actions = ButtonInput::default();
        actions.press(GameAction::Primary);
        drag_actions(&mut graph, &mut state, &mut history, &actions);
        assert!(state.placed_bearings.is_empty());
        assert!(state.suspension.drag.is_some());
        actions.clear();
        actions.release(GameAction::Primary);
        drag_actions(&mut graph, &mut state, &mut history, &actions);
        assert_eq!(state.placed_bearings.len(), 1);
        state.suspension.preview = Some(state.placed_bearings[0]);
        actions.clear();
        actions.press(GameAction::Primary);
        drag_actions(&mut graph, &mut state, &mut history, &actions);
        actions.clear();
        actions.press(GameAction::Secondary);
        drag_actions(&mut graph, &mut state, &mut history, &actions);
        actions.clear();
        actions.release(GameAction::Primary);
        drag_actions(&mut graph, &mut state, &mut history, &actions);
        assert_eq!(state.placed_bearings.len(), 1);
        assert!(state.suspension.drag.is_none());
    }

    #[test]
    fn opposite_plate_accepts_construction_and_live_settings_retain_joint_id() {
        let spec = SuspensionSpec::new(
            Some(SpringSpec::default()),
            Some(ShockSpec::default()),
            None,
        )
        .unwrap();
        let (mut graph, mut state) = fixture(spec);
        let socket = state.placed_bearings[0];
        let candidate = builder::suspension_block_candidate(socket).unwrap();
        graph =
            builder::stage_suspension_block(&graph, socket, candidate, &[], state.placement_bounds)
                .unwrap();
        assert_eq!(graph.bearing_count(), 1);
        assert_eq!(graph.compile().unwrap().bearings.len(), 1);
        let id = graph.bearings().next().unwrap().0;
        let mut history = EditorHistory::default();
        let spring = SpringSpec::new(0.5, 0.16, 0.12, 6, 0.025).unwrap();
        let updated = spec
            .with_components(Some(spring), spec.shock(), None, true)
            .unwrap();
        apply_settings(&mut graph, &mut state, &mut history, 0, updated);
        assert_eq!(graph.bearings().next().unwrap().0, id);
        assert_eq!(
            graph.bearing(id).unwrap().kind,
            BearingKind::Suspension(updated)
        );
        assert!(crate::apply_history_action(
            crate::HistoryAction::Undo,
            &mut graph,
            &mut state,
            &mut history
        ));
        assert_eq!(
            graph.bearing(id).unwrap().kind,
            BearingKind::Suspension(spec)
        );
        assert!(crate::apply_history_action(
            crate::HistoryAction::Redo,
            &mut graph,
            &mut state,
            &mut history
        ));
        assert_eq!(
            graph.bearing(id).unwrap().kind,
            BearingKind::Suspension(updated)
        );
    }
    #[test]
    fn insertion_previews_both_orders_and_leaves_host_dimensions_untouched() {
        for tool in [Tool::Spring, Tool::Shock] {
            let host = if tool == Tool::Spring {
                SuspensionSpec::new(None, Some(ShockSpec::default()), None)
            } else {
                SuspensionSpec::new(Some(SpringSpec::default()), None, None)
            }
            .unwrap();
            let (graph, mut state) = fixture(host);
            assert!(refresh(
                &graph,
                &mut state,
                tool,
                ConstructionMaterial::Steel,
                CylinderDimensions::default()
            ));
            assert!(state.preview_error.is_none(), "{:?}", state.preview_error);
            let BearingKind::Suspension(preview) = state.suspension.preview.unwrap().kind else {
                panic!("preview")
            };
            if tool == Tool::Spring {
                assert_eq!(preview.shock(), host.shock());
            } else {
                assert_eq!(preview.spring(), host.spring());
            }
            assert_eq!(state.placed_bearings[0].kind, BearingKind::Suspension(host));
        }
    }
    #[test]
    fn invalid_rubber_on_shock_cannot_fall_through_to_pipe_placement() {
        let spec = SuspensionSpec::new(None, Some(ShockSpec::default()), None).unwrap();
        let (graph, mut state) = fixture(spec);
        state.suspension.stop = BumpStopSpec::new(0.5, 0.06).unwrap();
        let dimensions = CylinderDimensions::default();
        assert!(refresh(
            &graph,
            &mut state,
            Tool::Cylinder,
            ConstructionMaterial::Rubber,
            dimensions
        ));
        assert!(state.preview_error.is_some());
        assert!(state.cylinder_preview.is_none());
        assert_eq!(state.suspension.insertion, Some(0));
        assert!(state.suspension.attachment.is_none());
        state.suspension.stop = BumpStopSpec::default();
        let valid = CylinderDimensions::default();
        assert!(refresh(
            &graph,
            &mut state,
            Tool::Cylinder,
            ConstructionMaterial::Rubber,
            valid
        ));
        assert!(state.preview_error.is_none());
        let BearingKind::Suspension(preview) = state.suspension.preview.unwrap().kind else {
            panic!("stop")
        };
        assert_eq!(
            preview.bump_stop(),
            Some(BumpStopSpec::new(0.05, 0.06).unwrap())
        );
    }
    #[test]
    fn rubber_pipe_on_spring_only_uses_normal_construction_attachment() {
        let spec = SuspensionSpec::new(Some(SpringSpec::default()), None, None).unwrap();
        let (graph, mut state) = fixture(spec);
        assert!(refresh(
            &graph,
            &mut state,
            Tool::Cylinder,
            ConstructionMaterial::Rubber,
            CylinderDimensions::default()
        ));
        assert!(state.suspension.attachment.is_some());
        assert!(state.suspension.insertion.is_none());
        assert!(state.cylinder_preview.is_some());
    }
    #[test]
    fn component_removal_ignores_duplicate_placement_feedback_and_preserves_shared_mounts() {
        let spec = SuspensionSpec::new(
            Some(SpringSpec::default()),
            Some(ShockSpec::default()),
            None,
        )
        .unwrap();
        let (mut graph, mut state) = fixture(spec);
        state.suspension.picked_component = Some((0, 0));
        state.preview_error = Some(PlacementError::Graph("component already installed".into()));
        let mut history = EditorHistory::default();
        assert!(remove_component(&mut graph, &mut state, &mut history, 0));
        let BearingKind::Suspension(remaining) = state.placed_bearings[0].kind else {
            panic!("mounts")
        };
        assert!(remaining.spring().is_none());
        assert_eq!(remaining.shock(), spec.shock());
        assert_eq!(remaining.plates(), spec.plates());
    }
    #[test]
    fn rubber_placement_cycles_free_length_and_diameter_and_commits_on_release() {
        let spec = SuspensionSpec::new(None, Some(ShockSpec::default()), None).unwrap();
        let (mut graph, mut state) = fixture(spec);
        state.pointer_ray = Some((Vec3::ZERO, Vec3::Z));
        refresh(
            &graph,
            &mut state,
            Tool::Cylinder,
            ConstructionMaterial::Rubber,
            CylinderDimensions::default(),
        );
        let mut actions = ButtonInput::default();
        actions.press(GameAction::Primary);
        let mut history = EditorHistory::default();
        drag_actions(&mut graph, &mut state, &mut history, &actions);
        assert!(state.suspension.drag.is_some());
        assert!(history.undo.is_empty());
        let drag = state.suspension.drag.as_ref().unwrap();
        assert!((resized(drag, Tool::Cylinder, 1.0).unwrap().2.length() - 0.0525).abs() < 1e-6);
        cycle_drag(&mut state, Tool::Cylinder).unwrap();
        let drag = state.suspension.drag.as_ref().unwrap();
        assert!((resized(drag, Tool::Cylinder, 1.0).unwrap().2.od() - 0.0625).abs() < 1e-6);
        refresh(
            &graph,
            &mut state,
            Tool::Cylinder,
            ConstructionMaterial::Rubber,
            CylinderDimensions::default(),
        );
        actions.clear();
        actions.release(GameAction::Primary);
        drag_actions(&mut graph, &mut state, &mut history, &actions);
        assert_eq!(history.undo.len(), 1);
        let BearingKind::Suspension(spec) = state.placed_bearings[0].kind else {
            panic!("suspension");
        };
        assert!(spec.bump_stop().is_some());
    }
}
