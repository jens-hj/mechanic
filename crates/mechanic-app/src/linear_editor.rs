//! Linear rail placement and carriage picking in the shared bearing workflow.
use crate::builder::PlacementError;
use crate::builder::bearings::bearing_anchor_from_hit_with_grid;
use crate::builder::faces::try_face_geometry_from_ref;
use crate::controls::GameAction;
use crate::creation_menu::CreationMenuState;
use crate::editor::build_actions::{PlacedBearing, bearing_location_occupied};
use crate::editor::history::{EditorHistory, EditorSnapshot};
use crate::editor::state::{EditorGraph, EditorState};
use crate::hotbar::{SelectedTool, Tool};
use crate::render::mesh::drive::axis_tangents;
use crate::simulation::state::AppSimulation;
use crate::{builder, linear_render};
use bevy::prelude::{ButtonInput, Res, ResMut, Transform, Vec3};
use mechanic_core::{
    BearingDimensions, BearingKind, CarriageFace, ConstructionGraph, FaceOwner, LinearBearing,
    LinearBearingDimensions, PartId,
};

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct LinearToolState {
    pub dimensions: LinearBearingDimensions,
    pub turns: u8,
}

pub(super) fn controls(
    mut graph: ResMut<EditorGraph>,
    actions: Res<ButtonInput<GameAction>>,
    selection: Res<SelectedTool>,
    menu: Res<CreationMenuState>,
    mut state: ResMut<EditorState>,
) {
    use GameAction as A;
    if selection.active_editor_tool() != Some(Tool::LinearBearing) || menu.blocks_keyboard() {
        return;
    }
    let mut view = super::live_edit::EditorView::new(&mut graph, &mut state);
    let (graph, state) = view.parts();
    for (action, width, step) in [
        (A::LinearLengthDecrease, false, -0.25),
        (A::LinearLengthIncrease, false, 0.25),
        (A::LinearWidthDecrease, true, -0.025),
        (A::LinearWidthIncrease, true, 0.025),
        (A::LinearLengthFineDecrease, false, -0.0025),
        (A::LinearLengthFineIncrease, false, 0.0025),
        (A::LinearWidthFineDecrease, true, -0.0025),
        (A::LinearWidthFineIncrease, true, 0.0025),
    ] {
        if actions.just_pressed(action) {
            state.linear.dimensions = resized(state.linear.dimensions, width, step);
        }
    }
    if actions.just_pressed(A::Rotate) {
        state.linear.turns = (state.linear.turns + 1) % 4;
    }
    let dimensions = state.linear.dimensions;
    let axis = preview_socket(&graph.0, state).map_or("—", |socket| axis_label(socket.axis));
    state.feedback = Some(format!(
        "Linear Bearing — {:.1} × {:.1} mm · {:.1} mm travel · axis {axis} · direction {} / 4",
        dimensions.length() * 1000.0,
        dimensions.width() * 1000.0,
        dimensions.travel() * 1000.0,
        state.linear.turns + 1
    ));
}

fn axis_label(axis: Vec3) -> &'static str {
    if axis.x.abs() > 0.5 {
        if axis.x > 0.0 { "+X" } else { "−X" }
    } else if axis.y.abs() > 0.5 {
        if axis.y > 0.0 { "+Y" } else { "−Y" }
    } else if axis.z > 0.0 {
        "+Z"
    } else {
        "−Z"
    }
}

fn resized(dimensions: LinearBearingDimensions, width: bool, step: f32) -> LinearBearingDimensions {
    let (length, width) = if width {
        (
            dimensions.length(),
            ((dimensions.width() + step) * 400.0)
                .round()
                .clamp(20.0, 160.0)
                / 400.0,
        )
    } else {
        (
            ((dimensions.length() + step) * 400.0)
                .round()
                .clamp(100.0, 3200.0)
                / 400.0,
            dimensions.width(),
        )
    };
    LinearBearingDimensions::new(length, width).expect("clamped rail dimensions")
}

fn travel_axis(normal: Vec3, turns: u8) -> Vec3 {
    let (u, v) = axis_tangents(normal);
    [u, v, -u, -v][usize::from(turns % 4)]
}

pub(super) fn preview_socket(
    graph: &ConstructionGraph,
    state: &EditorState,
) -> Option<PlacedBearing> {
    let hit = state.hovered?;
    let face = try_face_geometry_from_ref(hit.face, Some(graph))?;
    if !matches!(hit.face.owner, FaceOwner::Part(_)) {
        return None;
    }
    let anchor =
        bearing_anchor_from_hit_with_grid(graph, hit, state.placement_grid, state.placement_bounds)
            .ok()?;
    let axis = travel_axis(face.normal, state.linear.turns);
    Some(PlacedBearing {
        source: hit.face,
        anchor,
        axis,
        dimensions: BearingDimensions::default(),
        kind: BearingKind::Linear(LinearBearing {
            dimensions: state.linear.dimensions,
            mount_normal: face.normal,
            face: CarriageFace::Top,
        }),
    })
}

pub(super) fn refresh(graph: &ConstructionGraph, state: &mut EditorState) {
    let socket = preview_socket(graph, state);
    state.bearing_preview_anchor = socket.map(|socket| socket.anchor);
    state.preview_error = socket.and_then(|socket| {
        if let Err(error) = validate_socket_bounds(socket, state.placement_bounds) {
            return Some(error);
        }
        let BearingKind::Linear(rail) = socket.kind else {
            unreachable!()
        };
        (!builder::linear_mount_overlaps_face(
            graph,
            socket.source,
            socket.anchor,
            rail,
            socket.axis,
        ))
        .then(|| {
            PlacementError::Graph(
                "The rail underside must overlap a flat supporting face".to_owned(),
            )
        })
    });
}

fn validate_socket_bounds(
    socket: PlacedBearing,
    bounds: builder::PlacementBounds,
) -> Result<(), PlacementError> {
    let BearingKind::Linear(rail) = socket.kind else {
        return Ok(());
    };
    let rotation = rail
        .rotation(socket.axis)
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    let half_height = LinearBearingDimensions::HEIGHT * 0.5;
    let center = socket.anchor + rotation * Vec3::Y * half_height;
    // Include both end stops and the wider carriage; the rail can overhang
    // its support, but its complete travel envelope stays in the build area.
    let half = (rotation * Vec3::X).abs() * (rail.dimensions.length() * 0.5)
        + (rotation * Vec3::Y).abs() * half_height
        + (rotation * Vec3::Z).abs() * rail.dimensions.carriage_half_width();
    builder::validate_world_bounds(center - half, center + half, bounds)
}

pub(super) fn place(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
) {
    let Some(socket) = preview_socket(graph, state) else {
        return;
    };
    if let Err(error) = validate_socket_bounds(socket, state.placement_bounds) {
        state.feedback = Some(error.to_string());
        return;
    }
    let BearingKind::Linear(rail) = socket.kind else {
        return;
    };
    if !builder::linear_mount_overlaps_face(graph, socket.source, socket.anchor, rail, socket.axis)
    {
        return;
    }
    if bearing_location_occupied(graph, &state.placed_bearings, socket.source, socket.anchor) {
        state.feedback = Some("A bearing is already placed here".to_owned());
        return;
    }
    let previous = EditorSnapshot::capture(graph, state);
    state.placed_bearings.push(socket);
    history.commit(previous);
    state.construction_mesh_dirty = true;
    state.feedback =
        Some("Rail placed — attach construction to the carriage top or either side".to_owned());
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

pub(super) fn raycast(
    socket: PlacedBearing,
    rail_pose: Transform,
    carriage_pose: Transform,
    origin: Vec3,
    direction: Vec3,
) -> Option<f32> {
    let BearingKind::Linear(rail) = socket.kind else {
        return None;
    };
    let dims = rail.dimensions;
    let pick = |pose: Transform, min, max| {
        let inverse = pose.rotation.inverse();
        ray_box(
            inverse * (origin - pose.translation),
            inverse * direction,
            min,
            max,
        )
    };
    let a = pick(
        rail_pose,
        Vec3::new(-dims.length() / 2.0, 0.0, -dims.width() / 2.0 - 0.004),
        Vec3::new(dims.length() / 2.0, 0.062, dims.width() / 2.0 + 0.004),
    );
    let b = pick(
        carriage_pose,
        Vec3::new(-0.06, 0.01, -dims.carriage_half_width()),
        Vec3::new(0.06, 0.1, dims.carriage_half_width()),
    );
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

pub(super) fn build_poses(socket: PlacedBearing) -> Option<(Transform, Transform)> {
    let BearingKind::Linear(rail) = socket.kind else {
        return None;
    };
    let pose =
        Transform::from_translation(socket.anchor).with_rotation(rail.rotation(socket.axis).ok()?);
    Some((pose, pose))
}

/// Picks the nearest rail or carriage using the poses used by rendering.
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
            let (rail, carriage) = build_poses(socket)?;
            let (rail, carriage) = simulation.map_or((rail, carriage), |simulation| {
                linear_render::socket_transforms(graph, simulation, socket)
            });
            raycast(socket, rail, carriage, origin, direction).map(|distance| (index, distance))
        })
        .min_by(|first, second| first.1.total_cmp(&second.1))
}

#[cfg(test)]
pub(super) fn selected_socket(state: &EditorState, index: usize) -> Option<(PlacedBearing, Vec3)> {
    selected_socket_on_face(state, index, None)
}

/// Restricts further direct attachments to the carriage face already in use.
pub(super) fn selected_socket_on_face(
    state: &EditorState,
    index: usize,
    occupied: Option<CarriageFace>,
) -> Option<(PlacedBearing, Vec3)> {
    let mut socket = *state.placed_bearings.get(index)?;
    let BearingKind::Linear(mut rail) = socket.kind else {
        return None;
    };
    let (origin, direction) = state.pointer_ray?;
    let rotation = rail.rotation(socket.axis).ok()?;
    let mut nearest: Option<(CarriageFace, f32)> = None;
    for face in [
        CarriageFace::Top,
        CarriageFace::PositiveSide,
        CarriageFace::NegativeSide,
    ] {
        if occupied.is_some_and(|occupied| occupied != face) {
            continue;
        }
        let normal = rotation * face.normal();
        let denominator = direction.dot(normal);
        if denominator >= -1.0e-6 {
            continue;
        }
        let center = socket.anchor + rotation * face.origin(rail.dimensions);
        let distance = (center - origin).dot(normal) / denominator;
        if distance < 0.0 {
            continue;
        }
        let offset = origin + direction * distance - center;
        let size = face.size(rail.dimensions);
        if offset.dot(socket.axis).abs() <= size.x / 2.0 + 1.0e-5
            && offset.dot(socket.axis.cross(normal)).abs() <= size.y / 2.0 + 1.0e-5
            && nearest.is_none_or(|(_, old)| distance < old)
        {
            nearest = Some((face, distance));
        }
    }
    let (face, distance) = nearest?;
    rail.face = face;
    socket.kind = BearingKind::Linear(rail);
    Some((socket, origin + direction * distance))
}

pub(super) fn attachment(
    socket: PlacedBearing,
    targets: &[PartId],
) -> builder::LinearAttachment<'_> {
    let BearingKind::Linear(rail) = socket.kind else {
        unreachable!()
    };
    builder::LinearAttachment {
        source: socket.source,
        anchor: socket.anchor,
        rail,
        axis: socket.axis,
        rigid_targets: targets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mechanic_core::{BuildCommand, BuildOutcome, BuildPose};

    fn socket() -> (ConstructionGraph, PlacedBearing) {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(base) = graph
            .apply(BuildCommand::Spawn(
                mechanic_core::CuboidSpec::new([1, 1, 1], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            panic!("expected block")
        };
        (
            graph,
            PlacedBearing {
                source: mechanic_core::FaceRef::part(base, mechanic_core::FaceKind::PositiveY),
                anchor: Vec3::Y * 0.125,
                dimensions: BearingDimensions::default(),
                axis: Vec3::X,
                kind: BearingKind::Linear(LinearBearing {
                    dimensions: LinearBearingDimensions::default(),
                    mount_normal: Vec3::Y,
                    face: CarriageFace::Top,
                }),
            },
        )
    }

    #[test]
    fn placed_linear_rail_round_trips_through_undo_and_redo_without_duplicates() {
        let (mut graph, template) = socket();
        let mut state = EditorState {
            hovered: Some(crate::SurfaceHit {
                face: template.source,
                point: template.anchor,
                distance: 1.0,
            }),
            placement_bounds: builder::PlacementBounds::Garage,
            linear: LinearToolState {
                dimensions: LinearBearingDimensions::new(1.25, 0.125).unwrap(),
                turns: 3,
            },
            ..Default::default()
        };
        let mut history = EditorHistory::default();
        place(&graph, &mut state, &mut history);
        assert_eq!(state.placed_bearings.len(), 1);
        let placed = state.placed_bearings[0];
        place(&graph, &mut state, &mut history);
        assert_eq!(state.placed_bearings.len(), 1);
        assert!(crate::editor::history::apply_history_action(
            crate::editor::history::HistoryAction::Undo,
            &mut graph,
            &mut state,
            &mut history
        ));
        assert!(state.placed_bearings.is_empty());
        assert!(crate::editor::history::apply_history_action(
            crate::editor::history::HistoryAction::Redo,
            &mut graph,
            &mut state,
            &mut history
        ));
        assert_eq!(state.placed_bearings, vec![placed]);
    }

    #[test]
    fn resizing_uses_exact_coarse_and_fine_steps_and_clamps_dimensions() {
        let defaults = LinearBearingDimensions::default();
        assert_eq!(
            resized(defaults, false, 0.25),
            LinearBearingDimensions::new(1.25, 0.1).unwrap()
        );
        assert_eq!(
            resized(defaults, true, 0.025),
            LinearBearingDimensions::new(1.0, 0.125).unwrap()
        );
        assert_eq!(
            resized(defaults, false, -0.0025),
            LinearBearingDimensions::new(0.9975, 0.1).unwrap()
        );
        assert_eq!(
            resized(defaults, true, 0.0025),
            LinearBearingDimensions::new(1.0, 0.1025).unwrap()
        );
        let minimum = LinearBearingDimensions::new(0.25, 0.05).unwrap();
        let maximum = LinearBearingDimensions::new(8.0, 0.4).unwrap();
        for width in [false, true] {
            assert_eq!(resized(minimum, width, -0.25), minimum);
            assert_eq!(resized(maximum, width, 0.25), maximum);
        }
    }

    #[test]
    fn rail_overhang_is_allowed_only_within_the_complete_construction_bounds() {
        let (graph, socket) = socket();
        let BearingKind::Linear(rail) = socket.kind else {
            unreachable!()
        };
        assert!(builder::linear_mount_overlaps_face(
            &graph,
            socket.source,
            socket.anchor,
            rail,
            socket.axis
        ));
        assert!(validate_socket_bounds(socket, builder::PlacementBounds::Garage).is_ok());
        let edge = PlacedBearing {
            anchor: Vec3::new(9.5, 1.0, 0.0),
            ..socket
        };
        assert!(validate_socket_bounds(edge, builder::PlacementBounds::Garage).is_ok());
        let beyond = PlacedBearing {
            anchor: edge.anchor + Vec3::X * 0.0025,
            ..edge
        };
        assert!(matches!(
            validate_socket_bounds(beyond, builder::PlacementBounds::Garage),
            Err(PlacementError::OutsidePlatform)
        ));
        let carriage_edge = PlacedBearing {
            anchor: Vec3::new(0.0, 1.0, 9.94),
            ..socket
        };
        assert!(
            matches!(
                validate_socket_bounds(carriage_edge, builder::PlacementBounds::Garage),
                Err(PlacementError::OutsidePlatform)
            ),
            "the 130 mm carriage, not just the 100 mm rail, must fit"
        );
        let ceiling = PlacedBearing {
            anchor: Vec3::new(0.0, crate::garage::BUILD_MAX_Y - 0.05, 0.0),
            ..socket
        };
        assert!(matches!(
            validate_socket_bounds(ceiling, builder::PlacementBounds::GarageBuild),
            Err(PlacementError::OutsidePlatform)
        ));
        let origin = bevy::math::DVec2::new(mechanic_world::WORLD_HALF_EXTENT_METERS - 0.25, 0.0);
        assert!(
            matches!(
                validate_socket_bounds(socket, builder::PlacementBounds::World { origin }),
                Err(PlacementError::OutsidePlatform)
            ),
            "world placement includes the floating origin and rail overhang"
        );
    }

    #[test]
    fn rotate_selects_four_distinct_in_plane_directions_on_every_mounting_face() {
        for normal in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::NEG_X,
            Vec3::NEG_Y,
            Vec3::NEG_Z,
        ] {
            let axes = [0, 1, 2, 3].map(|turn| travel_axis(normal, turn));
            for (index, axis) in axes.iter().enumerate() {
                assert!(axis.dot(normal).abs() < 1.0e-6);
                assert!((axis.length() - 1.0).abs() < 1.0e-6);
                assert!(
                    axes.iter()
                        .skip(index + 1)
                        .all(|other| axis.distance(*other) > 0.5)
                );
            }
            assert!(travel_axis(normal, 4).distance(axes[0]) < 1.0e-6);
        }
    }

    #[test]
    fn carriage_attachment_picks_top_and_sides_but_not_end_faces() {
        let (_, socket) = socket();
        let BearingKind::Linear(rail) = socket.kind else {
            unreachable!()
        };
        let mut state = EditorState {
            placed_bearings: vec![socket],
            ..Default::default()
        };
        for face in [
            CarriageFace::Top,
            CarriageFace::PositiveSide,
            CarriageFace::NegativeSide,
        ] {
            let center = socket.anchor + face.origin(rail.dimensions);
            state.pointer_ray = Some((center + face.normal(), -face.normal()));
            let (picked, point) = selected_socket(&state, 0).unwrap();
            let BearingKind::Linear(picked) = picked.kind else {
                unreachable!()
            };
            assert_eq!(picked.face, face);
            assert!(point.distance(center) < 1.0e-6);
            assert!(selected_socket_on_face(&state, 0, Some(face)).is_some());
            let other = if face == CarriageFace::Top {
                CarriageFace::PositiveSide
            } else {
                CarriageFace::Top
            };
            assert!(selected_socket_on_face(&state, 0, Some(other)).is_none());
        }
        for direction in [Vec3::X, Vec3::NEG_X] {
            state.pointer_ray = Some((socket.anchor + Vec3::Y * 0.08 + direction, -direction));
            assert!(selected_socket(&state, 0).is_none());
        }
    }

    #[test]
    fn scene_picking_preserves_socket_indices_and_rejects_invalid_rays() {
        let (graph, first) = socket();
        let second = PlacedBearing {
            anchor: first.anchor + Vec3::X * 2.0,
            ..first
        };
        let origin = second.anchor + Vec3::Y;
        let picked =
            raycast_scene(&graph, None, &[first, second], origin, Vec3::NEG_Y * 3.0).unwrap();
        assert_eq!(picked.0, 1);
        assert!((picked.1 - 0.9).abs() < 1.0e-6);
        assert!(raycast_scene(&graph, None, &[first], origin, Vec3::ZERO).is_none());
        assert!(raycast_scene(&graph, None, &[first], Vec3::NAN, Vec3::Y).is_none());
        assert!(raycast_scene(&graph, None, &[first], origin, Vec3::NAN).is_none());
    }

    #[test]
    fn carriage_picking_follows_its_own_transform_along_the_rail() {
        let (_, socket) = socket();
        let (rail, mut carriage) = build_poses(socket).unwrap();
        carriage.translation += Vec3::X * 0.3;
        let origin = socket.anchor + Vec3::new(0.3, 1.0, 0.0);
        let distance = raycast(socket, rail, carriage, origin, Vec3::NEG_Y).unwrap();
        assert!((distance - 0.9).abs() < 1.0e-6);
        let old_position =
            raycast(socket, rail, carriage, socket.anchor + Vec3::Y, Vec3::NEG_Y).unwrap();
        assert!(old_position > distance + 0.03);
    }
}
