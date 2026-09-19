//! Piston placement, sizing, and picking in the shared bearing workflow.
use crate::builder::PlacementError;
use crate::builder::bearings::bearing_anchor_from_hit_with_grid;
use crate::builder::faces::try_face_geometry_from_ref;
use crate::controls::GameAction;
use crate::creation_menu::CreationMenuState;
use crate::editor::build_actions::{PlacedBearing, bearing_location_occupied};
use crate::editor::history::{EditorHistory, EditorSnapshot};
use crate::editor::pipe::pipe_pointer_delta;
use crate::editor::state::{EditorGraph, EditorState};
use crate::hotbar::{SelectedTool, Tool};
use crate::render::mesh::drive::axis_tangents;
use crate::simulation::state::AppSimulation;
use crate::{builder, piston_render};
use bevy::prelude::{ButtonInput, Res, ResMut, Transform, Vec3};
use mechanic_core::{
    BearingDimensions, BearingKind, BearingSocket, BearingSpec, BuildCommand, ConstructionGraph,
    FaceOwner, Piston, PistonDimensions, PistonMount,
};

/// Pointer travel, in radians, that steps a dragged count by one.
const DRAG_RADIANS_PER_STEP: f32 = 0.06;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PistonToolState {
    pub dimensions: PistonDimensions,
    /// Zero is an end mount; one to four are the side mount's travel directions.
    pub orientation: u8,
    pub drag: Option<PistonDrag>,
}

/// A held placement: the socket stays where it was pressed while the pointer sizes it.
#[derive(Clone, Copy, Debug)]
pub(super) struct PistonDrag {
    socket: PlacedBearing,
    direction: Vec3,
    dimensions: PistonDimensions,
    stages: bool,
}

pub(super) fn controls(
    mut graph: ResMut<EditorGraph>,
    actions: Res<ButtonInput<GameAction>>,
    selection: Res<SelectedTool>,
    menu: Res<CreationMenuState>,
    mut state: ResMut<EditorState>,
) {
    if selection.active_editor_tool() != Some(Tool::Piston) || menu.blocks_keyboard() {
        return;
    }
    let mut view = super::live_edit::EditorView::new(&mut graph, &mut state);
    let (_, state) = view.parts();
    if actions.just_pressed(GameAction::Rotate) {
        let ray = state.pointer_ray.map(|(_, direction)| direction);
        let dimensions = state.piston.dimensions;
        if let Some(drag) = &mut state.piston.drag {
            // Re-anchor so the knob just left keeps the value it reached.
            drag.stages = !drag.stages;
            drag.dimensions = dimensions;
            drag.direction = ray.unwrap_or(drag.direction);
        } else {
            state.piston.orientation = (state.piston.orientation + 1) % 5;
        }
    }
    let dimensions = state.piston.dimensions;
    let mount = if state.piston.orientation == 0 {
        "end mount".to_owned()
    } else {
        format!("side mount {} / 4", state.piston.orientation)
    };
    let hint = match state.piston.drag {
        Some(PistonDrag { stages: false, .. }) => "drag closed length · R stages · release places",
        Some(PistonDrag { stages: true, .. }) => "drag stages · R closed length · release places",
        None => "R mount · hold to size",
    };
    state.feedback = Some(format!(
        "Piston — {} blocks closed · {} stages · {:.2} m stroke · {} blocks extended · {mount} — {hint}",
        dimensions.blocks(),
        dimensions.stages(),
        dimensions.stroke(),
        dimensions.extended_blocks(),
    ));
}

fn resized(drag: PistonDrag, steps: f32) -> PistonDimensions {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to a single-digit count"
    )]
    let stepped = |value: u8, minimum: u8, maximum: u8| {
        (f32::from(value) + steps).clamp(f32::from(minimum), f32::from(maximum)) as u8
    };
    let (blocks, stages) = if drag.stages {
        (
            drag.dimensions.blocks(),
            stepped(
                drag.dimensions.stages(),
                PistonDimensions::MIN_STAGES,
                PistonDimensions::MAX_STAGES,
            ),
        )
    } else {
        (
            stepped(
                drag.dimensions.blocks(),
                PistonDimensions::MIN_BLOCKS,
                PistonDimensions::MAX_BLOCKS,
            ),
            drag.dimensions.stages(),
        )
    };
    PistonDimensions::new(blocks, stages).expect("clamped piston counts")
}

/// Travel directions a side mount cycles through on a face.
fn side_axes(mount_normal: Vec3) -> [Vec3; 4] {
    let (u, v) = axis_tangents(mount_normal);
    [u, v, -u, -v]
}

impl PistonToolState {
    /// Takes up a placed piston's configuration, returning the tool that places it.
    pub(super) fn adopt(&mut self, piston: Piston, axis: Vec3) -> Tool {
        self.dimensions = piston.dimensions;
        self.orientation = match piston.mount {
            PistonMount::End => 0,
            PistonMount::Side { mount_normal } => side_axes(mount_normal)
                .iter()
                .position(|candidate| candidate.abs_diff_eq(axis, 1.0e-5))
                .and_then(|index| u8::try_from(index + 1).ok())
                .unwrap_or(1),
        };
        Tool::Piston
    }
}

fn piston_of(socket: PlacedBearing) -> Option<Piston> {
    match socket.kind {
        BearingKind::Piston(piston) => Some(piston),
        _ => None,
    }
}

pub(super) fn preview_socket(
    graph: &ConstructionGraph,
    state: &EditorState,
) -> Option<PlacedBearing> {
    if let Some(drag) = state.piston.drag {
        let piston = piston_of(drag.socket)?;
        return Some(PlacedBearing {
            kind: BearingKind::Piston(Piston {
                dimensions: state.piston.dimensions,
                ..piston
            }),
            ..drag.socket
        });
    }
    let hit = state.hovered?;
    let face = try_face_geometry_from_ref(hit.face, Some(graph))?;
    if !matches!(hit.face.owner, FaceOwner::Part(_)) {
        return None;
    }
    let anchor =
        bearing_anchor_from_hit_with_grid(graph, hit, state.placement_grid, state.placement_bounds)
            .ok()?;
    let (mount, axis) = match state.piston.orientation {
        0 => (PistonMount::End, face.normal),
        turns => (
            PistonMount::Side {
                mount_normal: face.normal,
            },
            side_axes(face.normal)[usize::from(turns - 1) % 4],
        ),
    };
    Some(PlacedBearing {
        source: hit.face,
        anchor,
        axis,
        dimensions: BearingDimensions::default(),
        kind: BearingKind::Piston(Piston {
            dimensions: state.piston.dimensions,
            mount,
        }),
    })
}

/// The collapsed envelope stays in the build area; extension is the simulation's business.
fn validate_socket(
    graph: &ConstructionGraph,
    socket: PlacedBearing,
    bounds: builder::PlacementBounds,
) -> Result<(), PlacementError> {
    let Some(piston) = piston_of(socket) else {
        return Ok(());
    };
    let graph_error = |error: &dyn std::fmt::Display| PlacementError::Graph(error.to_string());
    let rotation = piston
        .rotation(socket.axis)
        .map_err(|error| graph_error(&error))?;
    let closed = piston.dimensions.closed();
    let half_section = PistonDimensions::SECTION / 2.0;
    let center = piston.base_center(socket.anchor, socket.axis) + socket.axis * (closed / 2.0);
    let half = (rotation * Vec3::X).abs() * half_section
        + (rotation * Vec3::Y).abs() * (closed / 2.0)
        + (rotation * Vec3::Z).abs() * half_section;
    builder::validate_world_bounds(center - half, center + half, bounds)?;
    graph
        .validate_socket(BearingSocket {
            kind: socket.kind,
            axis: socket.axis,
            source: socket.source,
            anchor: socket.anchor,
            dimensions: socket.dimensions,
        })
        .map_err(|_| {
            PlacementError::Graph(
                match piston.mount {
                    PistonMount::End => "The piston base must sit on a flat supporting face",
                    PistonMount::Side { .. } => {
                        "The piston body must lie on a flat supporting face"
                    }
                }
                .to_owned(),
            )
        })
}

pub(super) fn refresh(graph: &ConstructionGraph, state: &mut EditorState) {
    if let Some(drag) = state.piston.drag {
        let steps = state.pointer_ray.map_or(0.0, |(_, ray)| {
            (pipe_pointer_delta(drag.direction, ray) / DRAG_RADIANS_PER_STEP).round()
        });
        state.piston.dimensions = resized(drag, steps);
    }
    let socket = preview_socket(graph, state);
    state.bearing_preview_anchor = socket.map(|socket| socket.anchor);
    state.preview_error =
        socket.and_then(|socket| validate_socket(graph, socket, state.placement_bounds).err());
}

/// Places the piston together with its joint: the head is a body of its own,
/// so the piston can be wired and run before anything is built on it.
fn place(graph: &mut ConstructionGraph, state: &mut EditorState, history: &mut EditorHistory) {
    let Some(socket) = preview_socket(graph, state) else {
        return;
    };
    if let Err(error) = validate_socket(graph, socket, state.placement_bounds) {
        state.feedback = Some(error.to_string());
        return;
    }
    if bearing_location_occupied(graph, &state.placed_bearings, socket.source, socket.anchor) {
        state.feedback = Some("A bearing is already placed here".to_owned());
        return;
    }
    let previous = EditorSnapshot::capture(graph, state);
    if let Err(error) = graph.apply(BuildCommand::AddBearing(BearingSpec::bare(
        socket.source,
        socket.anchor,
        socket.axis,
        socket.kind,
    ))) {
        state.feedback = Some(error.to_string());
        return;
    }
    state.placed_bearings.push(socket);
    history.commit(previous);
    state.construction_mesh_dirty = true;
}

/// Press holds the piston where it is aimed, the pointer sizes it, release places it.
pub(super) fn drag_actions(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    actions: &ButtonInput<GameAction>,
) {
    if actions.just_pressed(GameAction::Secondary) {
        state.piston.drag = None;
        return;
    }
    if actions.just_pressed(GameAction::Primary)
        && state.preview_error.is_none()
        && let Some(socket) = preview_socket(graph, state)
        && let Some((_, direction)) = state.pointer_ray
    {
        state.piston.drag = Some(PistonDrag {
            socket,
            direction,
            dimensions: state.piston.dimensions,
            stages: false,
        });
    }
    if actions.just_released(GameAction::Primary) && state.piston.drag.is_some() {
        place(graph, state, history);
        state.piston.drag = None;
    }
}

fn ray_box(origin: Vec3, direction: Vec3, minimum: Vec3, maximum: Vec3) -> Option<f32> {
    let mut near = 0.0_f32;
    let mut far = f32::INFINITY;
    for axis in 0..3 {
        if direction[axis].abs() < 1.0e-7 {
            if origin[axis] < minimum[axis] || origin[axis] > maximum[axis] {
                return None;
            }
        } else {
            let a = (minimum[axis] - origin[axis]) / direction[axis];
            let b = (maximum[axis] - origin[axis]) / direction[axis];
            near = near.max(a.min(b));
            far = far.min(a.max(b));
            if far < near {
                return None;
            }
        }
    }
    Some(near)
}

/// Nearest hit on the body or on the drawn stage column under the head.
pub(super) fn raycast(
    socket: PlacedBearing,
    body_pose: Transform,
    head_pose: Transform,
    origin: Vec3,
    direction: Vec3,
) -> Option<f32> {
    let piston = piston_of(socket)?;
    let closed = piston.dimensions.closed();
    let half = PistonDimensions::SECTION / 2.0;
    let head = piston.dimensions.head_radius();
    let extension = (head_pose.translation - body_pose.translation)
        .dot(body_pose.rotation * Vec3::Y)
        .max(0.0);
    let pick = |pose: Transform, minimum, maximum| {
        let inverse = pose.rotation.inverse();
        ray_box(
            inverse * (origin - pose.translation),
            inverse * direction,
            minimum,
            maximum,
        )
    };
    let body = pick(
        body_pose,
        Vec3::new(-half, 0.0, -half),
        Vec3::new(half, closed, half),
    );
    let column = pick(
        head_pose,
        Vec3::new(-head, closed - extension, -head),
        Vec3::new(head, closed, head),
    );
    match (body, column) {
        (Some(body), Some(column)) => Some(body.min(column)),
        (body, column) => body.or(column),
    }
}

/// Body and head poses of the piston-local frame at the collapsed build pose.
pub(super) fn build_poses(socket: PlacedBearing) -> Option<(Transform, Transform)> {
    let piston = piston_of(socket)?;
    let pose = Transform::from_translation(piston.base_center(socket.anchor, socket.axis))
        .with_rotation(piston.rotation(socket.axis).ok()?);
    Some((pose, pose))
}

/// Picks the nearest piston using the poses used by rendering.
pub(super) fn raycast_scene(
    graph: &ConstructionGraph,
    simulation: Option<&AppSimulation>,
    sockets: &[PlacedBearing],
    origin: Vec3,
    direction: Vec3,
) -> Option<(usize, f32)> {
    if !origin.is_finite() || !direction.is_finite() || direction.length_squared() < f32::EPSILON {
        return None;
    }
    let direction = direction.normalize();
    sockets
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, socket)| {
            let (body, head) = build_poses(socket)?;
            let (body, head) = simulation.map_or((body, head), |simulation| {
                piston_render::socket_transforms(graph, simulation, socket)
            });
            raycast(socket, body, head, origin, direction).map(|distance| (index, distance))
        })
        .min_by(|first, second| first.1.total_cmp(&second.1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mechanic_core::{BuildCommand, BuildOutcome, BuildPose, CuboidSpec, FaceKind, FaceRef};

    fn socket(mount: PistonMount, axis: Vec3) -> (ConstructionGraph, PlacedBearing) {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            panic!("expected construction part");
        };
        let top =
            try_face_geometry_from_ref(FaceRef::part(part, FaceKind::PositiveY), Some(&graph))
                .unwrap();
        (
            graph,
            PlacedBearing {
                kind: BearingKind::Piston(Piston {
                    dimensions: PistonDimensions::default(),
                    mount,
                }),
                axis,
                source: FaceRef::part(part, FaceKind::PositiveY),
                anchor: top.center,
                dimensions: BearingDimensions::default(),
            },
        )
    }

    #[test]
    fn end_and_side_mounts_are_accepted_on_a_block_and_rejected_off_its_face() {
        for (mount, axis) in [
            (PistonMount::End, Vec3::Y),
            (
                PistonMount::Side {
                    mount_normal: Vec3::Y,
                },
                Vec3::X,
            ),
        ] {
            let (graph, socket) = socket(mount, axis);
            assert!(
                validate_socket(
                    &graph,
                    socket,
                    builder::PlacementBounds::World {
                        origin: bevy::math::DVec2::ZERO
                    }
                )
                .is_ok()
            );
            let adrift = PlacedBearing {
                anchor: socket.anchor + Vec3::Z * 2.0,
                ..socket
            };
            assert!(
                validate_socket(
                    &graph,
                    adrift,
                    builder::PlacementBounds::World {
                        origin: bevy::math::DVec2::ZERO
                    }
                )
                .is_err()
            );
        }
    }

    #[test]
    fn a_bare_piston_wires_to_a_controller_and_keeps_its_wire_when_built_on() {
        for (mount, axis) in [
            (PistonMount::End, Vec3::Y),
            (
                PistonMount::Side {
                    mount_normal: Vec3::Y,
                },
                Vec3::X,
            ),
        ] {
            let (mut graph, socket) = socket(mount, axis);
            let mut state = EditorState::default();
            state.piston.drag = Some(PistonDrag {
                socket,
                direction: Vec3::Z,
                dimensions: PistonDimensions::default(),
                stages: false,
            });
            let mut history = EditorHistory::default();
            place(&mut graph, &mut state, &mut history);
            assert_eq!(state.placed_bearings, vec![socket]);
            assert_eq!(graph.bearings().count(), 1);

            let BuildOutcome::Spawned(controller) = graph
                .apply(BuildCommand::SpawnController(
                    mechanic_core::ControllerSpec::new(BuildPose::new(
                        bevy::math::IVec3::new(0, 40, 0),
                        mechanic_core::GridRotation::default(),
                    )),
                ))
                .unwrap()
            else {
                panic!("expected controller");
            };
            let message = crate::editor::wiring::connect_drive_wire(
                &mut graph,
                &mut state,
                &mut history,
                controller,
                0,
            );
            assert_eq!(graph.drive_link_count(), 1, "{message}");
            let bare = graph.compile().unwrap();
            assert_eq!(bare.bearings.len(), 1);
            assert_eq!(bare.coordinate_drives.len(), 1);

            let candidate = builder::plate_block_candidate(socket).unwrap();
            let graph = builder::stage_plate_block(
                &graph,
                socket,
                candidate,
                &[],
                builder::PlacementBounds::World {
                    origin: bevy::math::DVec2::ZERO,
                },
            )
            .unwrap();
            let built = graph.compile().unwrap();
            assert_eq!(built.bearings.len(), 1);
            assert_eq!(built.coordinate_drives.len(), 1);
            assert_eq!(built.compounds.len(), bare.compounds.len());
        }
    }

    #[test]
    fn dragging_steps_one_count_at_a_time_and_stops_at_the_pack_limits() {
        let (_, socket) = socket(PistonMount::End, Vec3::Y);
        let drag = PistonDrag {
            socket,
            direction: Vec3::Z,
            dimensions: PistonDimensions::default(),
            stages: false,
        };
        assert_eq!(resized(drag, 1.0).blocks(), 3);
        assert_eq!(resized(drag, 40.0).blocks(), PistonDimensions::MAX_BLOCKS);
        assert_eq!(resized(drag, -40.0).blocks(), PistonDimensions::MIN_BLOCKS);
        let stages = PistonDrag {
            stages: true,
            ..drag
        };
        assert_eq!(resized(stages, -1.0).stages(), 3);
        assert_eq!(resized(stages, 1.0).blocks(), 2);
    }

    #[test]
    fn a_ray_finds_the_extended_stage_column_beyond_the_body() {
        let (_, socket) = socket(PistonMount::End, Vec3::Y);
        let (body, head) = build_poses(socket).unwrap();
        let above = socket.anchor + Vec3::new(1.0, 1.0, 0.0);
        assert!(raycast(socket, body, head, above, Vec3::NEG_X).is_none());
        let extended = Transform {
            translation: head.translation + Vec3::Y,
            ..head
        };
        assert!(raycast(socket, body, extended, above, Vec3::NEG_X).is_some());
    }
}
