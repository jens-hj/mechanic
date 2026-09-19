//! Drive wires and control links: dragging, connecting, and their hover previews.

use crate::controls::GameAction;
use crate::editor::build_actions::{PlacedBearing, socket_bearings};
use crate::editor::history::{EditorHistory, EditorSnapshot};
use crate::editor::preview::EditorVisuals;
use crate::editor::raycast::hovered_part;
use crate::editor::state::{EditorGraph, EditorState};
use crate::hotbar::{SelectedTool, Tool};
use crate::pose::live_placed_bearing_pose;
use crate::render::mesh::bearing::single_bearing_mesh;
use crate::render::mesh::drive::wire_drag_preview_mesh;
use crate::render::mesh::primitives::degenerate_overlay_mesh;
use crate::simulation::state::AppSimulation;
use crate::{linear_render, piston_render};
use bevy::prelude::{
    Assets, ButtonInput, Component, Cuboid, Local, Mesh, Quat, Res, ResMut, Single, Transform,
    Vec3, With, format,
};
use mechanic_core::{
    ActuatorAssignment, BuildCommand, ConstructionGraph, DriveLinkSpec, InputSeatLinkSpec, PartId,
    PartSpec, SeatControllerLinkSpec,
};

/// The wire the pointer is currently dragging between a block and a bearing.
#[derive(Component)]
pub(crate) struct WireDragVisual;

/// The joint or control block the pointer would wire, drawn oversized.
#[derive(Component)]
pub(crate) struct WireHoverVisual;

/// One-line description of a wire's envelope and its first state.
pub(crate) fn drive_summary(spec: &DriveLinkSpec) -> String {
    let actuator = match spec.actuator {
        ActuatorAssignment::Unpowered => "unpowered".to_owned(),
        ActuatorAssignment::Servo => "Servo".to_owned(),
        ActuatorAssignment::Motor {
            electric_percent,
            gas_percent,
        } => format!("motor E{electric_percent}% / G{gas_percent}%"),
    };
    let states = spec.program.len();
    format!(
        "{actuator}, {states} state{}",
        if states == 1 { "" } else { "s" }
    )
}

/// Advances the two-click connector. Returns the feedback line to display.
/// One end of a drive wire while it is being dragged out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WireEnd {
    Controller(PartId),
    Input(PartId),
    Seat(PartId),
    /// Index into [`EditorState::placed_bearings`].
    Bearing(usize),
}

impl WireEnd {
    /// Resolves a supported logical connection from two ends, in either order.
    pub(crate) const fn paired_with(self, other: Self) -> Option<WireConnection> {
        match (self, other) {
            (Self::Controller(controller), Self::Bearing(bearing))
            | (Self::Bearing(bearing), Self::Controller(controller)) => {
                Some(WireConnection::Drive {
                    controller,
                    bearing,
                })
            }
            (Self::Input(input), Self::Seat(seat)) | (Self::Seat(seat), Self::Input(input)) => {
                Some(WireConnection::InputSeat { input, seat })
            }
            (Self::Seat(seat), Self::Controller(controller))
            | (Self::Controller(controller), Self::Seat(seat)) => {
                Some(WireConnection::SeatController { seat, controller })
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WireConnection {
    Drive { controller: PartId, bearing: usize },
    InputSeat { input: PartId, seat: PartId },
    SeatController { seat: PartId, controller: PartId },
}

/// A drive wire the pointer is dragging out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WireDrag {
    pub(crate) from: WireEnd,
    /// The pointer was released back on `from`, so the wire is waiting for a
    /// second click instead of a drag.
    pub(crate) armed: bool,
}

/// What one pointer press or release does to a wire drag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WireDragStep {
    /// Nothing in progress and nothing to start.
    Idle,
    /// Nothing wirable under the pointer.
    Miss,
    Begin(WireEnd),
    Connect(WireConnection),
    /// Keep the started end, so a plain click can be finished by a second one.
    Arm,
    Cancel,
}

/// Wiring is symmetric: press either end and release on the other. Pressing and
/// releasing on the same end leaves the wire armed, so click-then-click works
/// as well as drag-and-drop.
pub(crate) fn wire_drag_step(
    drag: Option<WireDrag>,
    under: Option<WireEnd>,
    pressed: bool,
) -> WireDragStep {
    let Some(drag) = drag else {
        if !pressed {
            return WireDragStep::Idle;
        }
        return under.map_or(WireDragStep::Miss, WireDragStep::Begin);
    };
    if let Some(under) = under
        && let Some(connection) = drag.from.paired_with(under)
    {
        return WireDragStep::Connect(connection);
    }
    if pressed {
        // A press somewhere else restarts the wire there, or drops it.
        return under.map_or(WireDragStep::Cancel, WireDragStep::Begin);
    }
    if under == Some(drag.from) {
        WireDragStep::Arm
    } else {
        WireDragStep::Cancel
    }
}

/// The wire end the pointer is over, if any. A bearing wins over the block
/// behind it, which is what the hover raycast already resolves.
pub(crate) fn wire_end_under_cursor(
    graph: &ConstructionGraph,
    state: &EditorState,
) -> Option<WireEnd> {
    if let Some(index) = state.hovered_bearing {
        return Some(WireEnd::Bearing(index));
    }
    let part =
        hovered_part(state.hovered).or_else(|| state.hovered_simulation.map(|hit| hit.part))?;
    match graph.part(part) {
        Some(PartSpec::Controller(_)) => Some(WireEnd::Controller(part)),
        Some(PartSpec::Input(_)) => Some(WireEnd::Input(part)),
        Some(PartSpec::Seat(_)) => Some(WireEnd::Seat(part)),
        _ => None,
    }
}

pub(crate) fn wire_end_position(
    graph: &ConstructionGraph,
    state: &EditorState,
    simulation: &AppSimulation,
    end: WireEnd,
) -> Option<Vec3> {
    match end {
        WireEnd::Controller(part) | WireEnd::Input(part) | WireEnd::Seat(part) => {
            simulation.live_part_pose(graph, part).map(|pose| pose.0)
        }
        WireEnd::Bearing(index) => {
            live_placed_bearing_pose(graph, simulation, *state.placed_bearings.get(index)?)
                .map(|pose| pose.0)
        }
    }
}

/// Both ends of the wire being dragged: where it started, and either the joint
/// it would land on or the pointer itself.
pub(crate) fn wire_drag_endpoints(
    graph: &ConstructionGraph,
    state: &EditorState,
    simulation: &AppSimulation,
) -> Option<(Vec3, Vec3)> {
    let drag = state.wire_drag?;
    let from = wire_end_position(graph, state, simulation, drag.from)?;
    let target = wire_end_under_cursor(graph, state)
        .filter(|end| drag.from.paired_with(*end).is_some())
        .and_then(|end| wire_end_position(graph, state, simulation, end));
    if let Some(target) = target {
        return Some((from, target));
    }
    // No target yet, so the loose end follows the pointer at the depth the
    // wire started from.
    let (mut origin, mut direction) = state.pointer_ray?;
    if let Some(context) = state.edit_context {
        origin = context.frame_to_world.point(origin);
        direction = context.frame_to_world.vector(direction);
    }
    let direction = direction.normalize_or_zero();
    if direction == Vec3::ZERO {
        return None;
    }
    Some((
        from,
        origin + direction * (from - origin).dot(direction).max(0.1),
    ))
}

/// Press-and-drag wiring for the Connector tool.
pub(crate) fn handle_connector_actions(
    actions: &ButtonInput<GameAction>,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
) {
    let pressed = actions.just_pressed(GameAction::Primary);
    if !pressed && !actions.just_released(GameAction::Primary) {
        return;
    }
    let under = wire_end_under_cursor(graph, state);
    match wire_drag_step(state.wire_drag, under, pressed) {
        WireDragStep::Idle => {}
        WireDragStep::Miss => {
            state.feedback =
                Some("Drag Controller↔Bearing, Input↔Seat, or Seat↔Controller".to_owned());
        }
        WireDragStep::Begin(from) => {
            state.wire_drag = Some(WireDrag { from, armed: false });
            state.feedback = Some(match from {
                WireEnd::Controller(controller) => {
                    state.selected_controller = Some(controller);
                    "Drag to a bearing or Seat".to_owned()
                }
                WireEnd::Bearing(_) => "Drag to a control block to wire it".to_owned(),
                WireEnd::Input(_) => "Drag to a Seat".to_owned(),
                WireEnd::Seat(_) => "Drag to an Input or Controller".to_owned(),
            });
        }
        WireDragStep::Connect(connection) => {
            state.wire_drag = None;
            state.feedback = Some(match connection {
                WireConnection::Drive {
                    controller,
                    bearing,
                } => connect_drive_wire(graph, state, history, controller, bearing),
                WireConnection::InputSeat { input, seat } => connect_control_link(
                    graph,
                    state,
                    history,
                    BuildCommand::AddInputSeatLink(InputSeatLinkSpec { input, seat }),
                    "Linked Input to Seat",
                ),
                WireConnection::SeatController { seat, controller } => connect_control_link(
                    graph,
                    state,
                    history,
                    BuildCommand::AddSeatControllerLink(SeatControllerLinkSpec {
                        seat,
                        controller,
                    }),
                    "Linked Seat to Controller",
                ),
            });
        }
        WireDragStep::Arm => {
            if let Some(drag) = state.wire_drag.as_mut() {
                drag.armed = true;
            }
            state.feedback = Some("Now click the other end to finish the wire".to_owned());
        }
        WireDragStep::Cancel => {
            state.wire_drag = None;
            state.feedback = Some("Drive wire cancelled".to_owned());
        }
    }
}

pub(crate) fn connect_control_link(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    command: BuildCommand,
    success: &str,
) -> String {
    let removal = match &command {
        BuildCommand::AddInputSeatLink(spec) => graph.input_seat_links().find_map(|(id, link)| {
            (*link == *spec).then_some((
                BuildCommand::RemoveInputSeatLink(id),
                "Removed Input-to-Seat link",
            ))
        }),
        BuildCommand::AddSeatControllerLink(spec) => {
            graph.seat_controller_links().find_map(|(id, link)| {
                (*link == *spec).then_some((
                    BuildCommand::RemoveSeatControllerLink(id),
                    "Removed Seat-to-Controller link",
                ))
            })
        }
        _ => None,
    };
    let (command, feedback) = removal.unwrap_or((command, success));
    let previous = EditorSnapshot::capture(graph, state);
    match graph.apply(command) {
        Ok(_) => {
            history.commit(previous);
            // The changed link is a line in the drive overlay, and that overlay
            // is only rebuilt on request.
            state.construction_mesh_dirty = true;
            feedback.to_owned()
        }
        Err(error) => error.to_string(),
    }
}

/// Wires `controller` to every bearing row of one placed socket, or removes
/// those wires when the pair is already connected. Returns the feedback line.
pub(crate) fn connect_drive_wire(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    controller: PartId,
    socket_index: usize,
) -> String {
    let Some(socket) = state.placed_bearings.get(socket_index).copied() else {
        return "That bearing is no longer there".to_owned();
    };
    let bearings = socket_bearings(graph, socket);
    if bearings.is_empty() {
        return "Cannot wire an empty bearing — attach a part through it first".to_owned();
    }

    let existing = graph
        .drive_links()
        .filter(|(_, link)| link.controller == controller && bearings.contains(&link.bearing))
        .map(|(id, link)| (id, *link))
        .collect::<Vec<_>>();
    let previous = EditorSnapshot::capture(graph, state);
    let removing = !existing.is_empty();
    let commands = if removing {
        existing
            .iter()
            .map(|&(id, _)| BuildCommand::RemoveDriveLink(id))
            .collect::<Vec<_>>()
    } else {
        bearings
            .iter()
            .map(|&bearing| {
                BuildCommand::AddDriveLink(
                    match graph.bearing(bearing).expect("live bearing").kind {
                        kind if kind.is_translational() && kind.accepts_drive() => {
                            DriveLinkSpec::new_linear(controller, bearing, kind.bounds())
                        }
                        _ => DriveLinkSpec::new(controller, bearing),
                    },
                )
            })
            .collect::<Vec<_>>()
    };

    let mut staged = graph.begin_edit();
    match staged.apply_batch(commands) {
        Ok(_) => {
            *graph = staged.finish();
            history.commit(previous);
            state.selected_controller = Some(controller);
            state.construction_mesh_dirty = true;
            if removing {
                format!("Removed drive wire from {} bearing row(s)", existing.len())
            } else {
                let summary = graph
                    .bearing_drive_link(bearings[0])
                    .map_or_else(String::new, |(_, link)| {
                        format!(" — {}", drive_summary(link))
                    });
                format!(
                    "Wired {} bearing row(s){summary}. Press E to program it",
                    bearings.len()
                )
            }
        }
        Err(error) => error.to_string(),
    }
}

/// Removes Input-chain links from the hovered endpoint. Bearing drive reversal
/// is handled first by [`reverse_drive_wires`].
pub(crate) fn disconnect_connector_links(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
) -> String {
    if graph.pending().is_some() {
        let _ = graph.apply(BuildCommand::CancelPending);
        return "Drive wire cancelled".to_owned();
    }
    if state.hovered_bearing.is_some() {
        return "That bearing is not wired to a control block".to_owned();
    }
    let Some(part) = hovered_part(state.hovered) else {
        return "Right click a linked bearing, Input, Seat, or Controller".to_owned();
    };
    let commands = graph
        .input_seat_links()
        .filter_map(|(id, link)| {
            (link.input == part || link.seat == part)
                .then_some(BuildCommand::RemoveInputSeatLink(id))
        })
        .chain(graph.seat_controller_links().filter_map(|(id, link)| {
            (link.seat == part || link.controller == part)
                .then_some(BuildCommand::RemoveSeatControllerLink(id))
        }))
        .collect::<Vec<_>>();
    if commands.is_empty() {
        return "That part has no Input-chain links".to_owned();
    }
    let previous = EditorSnapshot::capture(graph, state);
    match graph.apply_batch(commands) {
        Ok(_) => {
            history.commit(previous);
            state.construction_mesh_dirty = true;
            "Removed Input-chain link(s)".to_owned()
        }
        Err(error) => error.to_string(),
    }
}

/// Flips every drive wire on one placed socket, preserving its controller and
/// program. `None` means the socket is not driven and the normal secondary
/// action should continue.
pub(crate) fn reverse_drive_wires(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    socket: PlacedBearing,
) -> Option<String> {
    let bearings = socket_bearings(graph, socket);
    let links = graph
        .drive_links()
        .filter_map(|(id, link)| bearings.contains(&link.bearing).then_some((id, *link)))
        .collect::<Vec<_>>();
    if links.is_empty() {
        return None;
    }
    if socket.kind.is_one_sided() {
        return Some(
            "A piston extends one way from collapsed; its wire has no direction".to_owned(),
        );
    }
    let previous = EditorSnapshot::capture(graph, state);
    let mut staged = graph.begin_edit();
    let commands = links
        .iter()
        .map(|&(id, _)| BuildCommand::RemoveDriveLink(id))
        .chain(links.iter().map(|&(_, link)| {
            BuildCommand::AddDriveLink(DriveLinkSpec {
                reversed: !link.reversed,
                ..link
            })
        }));
    Some(match staged.apply_batch(commands) {
        Ok(_) => {
            *graph = staged.finish();
            history.commit(previous);
            state.construction_mesh_dirty = true;
            "Changed this bearing's default direction".to_owned()
        }
        Err(error) => error.to_string(),
    })
}

/// How much bigger a wirable joint or block is drawn while the pointer is on
/// it. The ring is thin, so it needs more than the solid block does.
pub(crate) const WIRE_HOVER_BEARING_SCALE: f32 = 1.3;

pub(crate) const WIRE_HOVER_BLOCK_SCALE: f32 = 1.14;

/// Draws the joint or control block the pointer is over, slightly oversized, so
/// what a wire would land on is visible before the button goes down.
#[expect(clippy::too_many_arguments)]
pub(crate) fn update_wire_hover_preview(
    graph: Res<EditorGraph>,
    state: Res<EditorState>,
    selection: Res<SelectedTool>,
    simulation: Res<AppSimulation>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut drawn: Local<Option<WireEnd>>,
    mut transform: Single<&mut Transform, With<WireHoverVisual>>,
) {
    let hovered = if selection.active_editor_tool() == Some(Tool::Connector) {
        wire_end_under_cursor(&graph.0, &state)
    } else {
        None
    };
    let placement = match hovered {
        Some(WireEnd::Bearing(index)) => state.placed_bearings.get(index).and_then(|&socket| {
            if matches!(socket.kind, mechanic_core::BearingKind::Suspension(_)) {
                return None;
            }
            if let mechanic_core::BearingKind::Linear(rail) = socket.kind {
                let (_, carriage) = linear_render::socket_transforms(&graph.0, &simulation, socket);
                return Some(carriage.mul_transform(
                    Transform::from_translation(Vec3::new(0.0, 0.055, 0.0)).with_scale(
                        Vec3::new(
                            mechanic_core::LinearBearingDimensions::CARRIAGE_LENGTH,
                            0.09,
                            rail.dimensions.carriage_half_width() * 2.0,
                        ) * WIRE_HOVER_BLOCK_SCALE,
                    ),
                ));
            }
            if let mechanic_core::BearingKind::Piston(piston) = socket.kind {
                let (body, _) = piston_render::socket_transforms(&graph.0, &simulation, socket);
                let closed = piston.dimensions.closed();
                let section = mechanic_core::PistonDimensions::SECTION;
                return Some(
                    body.mul_transform(
                        Transform::from_translation(Vec3::Y * (closed / 2.0)).with_scale(
                            Vec3::new(section, closed, section) * WIRE_HOVER_BLOCK_SCALE,
                        ),
                    ),
                );
            }
            let (anchor, axis) = live_placed_bearing_pose(&graph.0, &simulation, socket)?;
            Some(
                Transform::from_translation(anchor)
                    .with_rotation(Quat::from_rotation_arc(Vec3::Y, axis))
                    .with_scale(Vec3::splat(WIRE_HOVER_BEARING_SCALE)),
            )
        }),
        Some(WireEnd::Controller(part) | WireEnd::Input(part) | WireEnd::Seat(part)) => graph
            .0
            .part(part)
            .and_then(|spec| spec.as_cuboid())
            .and_then(|block| {
                let (translation, rotation) = simulation.live_part_pose(&graph.0, part)?;
                Some(
                    Transform::from_translation(translation)
                        .with_rotation(rotation)
                        .with_scale(block.size_meters() * WIRE_HOVER_BLOCK_SCALE),
                )
            }),
        None => None,
    };
    let hovered = placement.is_some().then_some(hovered).flatten();
    if *drawn != hovered
        && let Some(mut mesh) = meshes.get_mut(&visuals.wire_hover_mesh)
    {
        *mesh =
            match hovered {
                Some(WireEnd::Bearing(index)) => state.placed_bearings.get(index).map_or_else(
                    degenerate_overlay_mesh,
                    |socket| match socket.kind {
                        mechanic_core::BearingKind::Rotational => {
                            single_bearing_mesh(socket.dimensions)
                        }
                        mechanic_core::BearingKind::Linear(_)
                        | mechanic_core::BearingKind::Suspension(_)
                        | mechanic_core::BearingKind::Piston(_) => Cuboid::default().into(),
                    },
                ),
                Some(WireEnd::Controller(_) | WireEnd::Input(_) | WireEnd::Seat(_)) => {
                    Cuboid::default().into()
                }
                None => degenerate_overlay_mesh(),
            };
    }
    *drawn = hovered;
    **transform = placement.unwrap_or_default();
}

/// Draws the wire from the end it was started on to the pointer, snapping to a
/// joint or block once the pointer is over one that would complete it.
pub(crate) fn update_wire_drag_preview(
    graph: Res<EditorGraph>,
    actions: Res<ButtonInput<GameAction>>,
    mut state: ResMut<EditorState>,
    simulation: Res<AppSimulation>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut drawn: Local<bool>,
) {
    // The release that ends a drag is swallowed when it lands on the camera or
    // the hotbar, so a wire that is neither held nor armed is dropped here
    // rather than left following the pointer forever.
    if state
        .wire_drag
        .is_some_and(|drag| !drag.armed && !actions.pressed(GameAction::Primary))
    {
        state.wire_drag = None;
    }
    let endpoints = wire_drag_endpoints(&graph.0, &state, &simulation);
    if endpoints.is_none() && !*drawn {
        return;
    }
    let (from, to) = endpoints.unwrap_or((Vec3::ZERO, Vec3::ZERO));
    if let Some(mut mesh) = meshes.get_mut(&visuals.wire_drag_mesh) {
        *mesh = wire_drag_preview_mesh(from, to);
    }
    *drawn = endpoints.is_some();
}
