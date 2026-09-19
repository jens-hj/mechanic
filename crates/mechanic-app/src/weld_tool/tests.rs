use super::*;
use mechanic_core::{BuildCommand, BuildOutcome, BuildPose, CuboidSpec, GridRotation};

fn scene() -> (ConstructionGraph, PartId, PartId) {
    let mut graph = ConstructionGraph::new();
    let mut parts = Vec::new();
    for x in [0, 8] {
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [2; 3],
                    BuildPose::new(IVec3::new(x, 28, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            panic!("spawn expected")
        };
        parts.push(part);
    }
    (graph, parts[0], parts[1])
}
fn ray(x: f32) -> Ray3d {
    Ray3d::new(Vec3::new(x, 9.0, 0.0), Dir3::NEG_Y)
}

#[test]
fn garage_press_drag_release_commits_one_default_pose_placement() {
    let (mut graph, source, destination) = scene();
    let simulation = AppSimulation::default();
    let world = crate::world::WorldRuntime::from_world(&mut World::new());
    let mut state = EditorState {
        placement_bounds: builder::PlacementBounds::GarageBuild,
        ..default()
    };
    let mut history = crate::editor::history::EditorHistory::default();
    let mut input = ButtonInput::default();
    hover(&graph, &simulation, &mut state, ray(0.0), &input, &world);
    input.press(GameAction::Primary);
    actions(
        &mut graph,
        &simulation,
        &mut state,
        &mut history,
        &input,
        false,
    );
    input.clear();
    input.release(GameAction::Primary);
    actions(
        &mut graph,
        &simulation,
        &mut state,
        &mut history,
        &input,
        false,
    );
    assert!(state.weld.busy());
    assert_eq!(graph.weld_count(), 0);
    input.clear();
    hover(&graph, &simulation, &mut state, ray(2.0), &input, &world);
    assert!(state.weld.preview.is_some());
    assert!(state.weld.error.is_none(), "{:?}", state.weld.error);
    input.press(GameAction::Primary);
    actions(
        &mut graph,
        &simulation,
        &mut state,
        &mut history,
        &input,
        false,
    );
    input.clear();
    hover(&graph, &simulation, &mut state, ray(2.0), &input, &world);
    assert!(state.weld.effects_target(&simulation).is_some());
    input.release(GameAction::Primary);
    actions(
        &mut graph,
        &simulation,
        &mut state,
        &mut history,
        &input,
        false,
    );
    assert!(state.weld.effects_target(&simulation).is_none());
    assert_eq!(graph.weld_count(), 1, "{:?}", state.feedback);
    assert_eq!(history.undo.len(), 1);
    assert!(!state.weld.busy());
    assert!(
        graph
            .part_position(source)
            .unwrap()
            .abs_diff_eq(Vec3::new(2.0, 7.5, 0.0), 1.0e-5)
    );
    assert!(
        graph
            .part_position(destination)
            .unwrap()
            .abs_diff_eq(Vec3::new(2.0, 7.0, 0.0), 1.0e-5)
    );
    assert!(crate::editor::history::apply_history_action(
        crate::editor::history::HistoryAction::Undo,
        &mut graph,
        &mut state,
        &mut history
    ));
    assert!(
        graph
            .part_position(source)
            .unwrap()
            .abs_diff_eq(Vec3::Y * 7.0, 1.0e-5)
    );
    assert!(crate::editor::history::apply_history_action(
        crate::editor::history::HistoryAction::Redo,
        &mut graph,
        &mut state,
        &mut history
    ));
    assert_eq!(graph.weld_count(), 1);
}

#[test]
fn invalid_release_retains_source_and_escape_or_secondary_cancels_without_editing() {
    for escape in [false, true] {
        let (mut graph, _, _) = scene();
        let original = graph.clone();
        let simulation = AppSimulation::default();
        let world = crate::world::WorldRuntime::from_world(&mut World::new());
        let mut state = EditorState {
            placement_bounds: builder::PlacementBounds::GarageBuild,
            ..default()
        };
        let mut history = crate::editor::history::EditorHistory::default();
        let mut input = ButtonInput::default();
        hover(&graph, &simulation, &mut state, ray(0.0), &input, &world);
        input.press(GameAction::Primary);
        actions(
            &mut graph,
            &simulation,
            &mut state,
            &mut history,
            &input,
            false,
        );
        input.reset_all();
        hover(&graph, &simulation, &mut state, ray(2.0), &input, &world);
        input.press(GameAction::Primary);
        actions(
            &mut graph,
            &simulation,
            &mut state,
            &mut history,
            &input,
            false,
        );
        input.clear();
        hover(&graph, &simulation, &mut state, ray(4.0), &input, &world);
        assert!(state.weld.error.is_some());
        assert!(state.weld.effects_target(&simulation).is_none());
        assert!(state.weld.preview.is_some());
        input.release(GameAction::Primary);
        actions(
            &mut graph,
            &simulation,
            &mut state,
            &mut history,
            &input,
            false,
        );
        assert!(state.weld.source.is_some());
        assert!(history.undo.is_empty());
        assert!(graph.shares_revision(&original));
        if escape {
            crate::pause_menu::cancel_one_world_escape_owner(&mut graph, &mut state);
        } else {
            input.press(GameAction::Secondary);
            actions(
                &mut graph,
                &simulation,
                &mut state,
                &mut history,
                &input,
                false,
            );
        }
        assert!(!state.weld.busy());
        assert!(graph.shares_revision(&original));
    }
}

#[test]
fn a_locked_destination_tracks_motion_without_a_stationary_drag_jump() {
    let (mut graph, _, destination) = scene();
    let creation = graph.compile().unwrap();
    let transforms = creation
        .compounds
        .iter()
        .map(|b| mechanic_gpu::GpuTransform {
            position: b.root_translation.extend(0.0).to_array(),
            rotation: b.root_rotation.to_array(),
        })
        .collect::<Vec<_>>();
    let mut simulation = AppSimulation {
        published_graph: graph.clone(),
        creation: Some(creation),
        transforms: transforms.clone(),
        live_state: Some(crate::simulation::state::LivePhysicsState {
            tick: 1,
            transforms,
            velocities: vec![
                mechanic_gpu::GpuVelocity {
                    linear: [0.0; 4],
                    angular: [0.0; 4]
                };
                2
            ],
            coordinates: Vec::new(),
        }),
        world_revision: Some((0, 0)),
        ..default()
    };
    let world = crate::world::WorldRuntime::from_world(&mut World::new());
    let mut state = EditorState {
        placement_bounds: builder::PlacementBounds::GarageBuild,
        ..default()
    };
    let mut history = crate::editor::history::EditorHistory::default();
    let mut input = ButtonInput::default();
    hover(&graph, &simulation, &mut state, ray(0.0), &input, &world);
    input.press(GameAction::Primary);
    actions(
        &mut graph,
        &simulation,
        &mut state,
        &mut history,
        &input,
        false,
    );
    input.reset_all();
    hover(&graph, &simulation, &mut state, ray(2.0), &input, &world);
    input.press(GameAction::Primary);
    actions(
        &mut graph,
        &simulation,
        &mut state,
        &mut history,
        &input,
        false,
    );
    input.clear();
    let before = state.weld.preview.as_ref().unwrap().2;
    let body = simulation
        .creation
        .as_ref()
        .unwrap()
        .part_to_compound
        .iter()
        .find(|(part, _)| *part == destination)
        .unwrap()
        .1 as usize;
    simulation.transforms[body].position[0] += 0.3;
    simulation.live_state.as_mut().unwrap().transforms[body] = simulation.transforms[body];
    hover(&graph, &simulation, &mut state, ray(2.0), &input, &world);
    let after = state.weld.preview.as_ref().unwrap().2;
    assert!((after.translation() - before.translation()).abs_diff_eq(Vec3::X * 0.3, 1.0e-5));
    assert!(state.weld.drag.as_ref().unwrap().displacement.length() < 1.0e-6);
    assert!(state.weld.error.is_none(), "{:?}", state.weld.error);
}

#[test]
#[expect(clippy::too_many_lines)]
fn initial_hover_snaps_both_axes_and_drag_keeps_the_snapped_alignment() {
    let (mut graph, _, _) = scene();
    let simulation = AppSimulation::default();
    let world = crate::world::WorldRuntime::from_world(&mut World::new());
    let mut state = EditorState {
        placement_bounds: builder::PlacementBounds::GarageBuild,
        ..default()
    };
    let mut history = crate::editor::history::EditorHistory::default();
    let mut input = ButtonInput::default();
    let pointer = |x, z| Ray3d::new(Vec3::new(x, 9.0, z), Dir3::NEG_Y);
    hover(
        &graph,
        &simulation,
        &mut state,
        pointer(0.03, 0.04),
        &input,
        &world,
    );
    input.press(GameAction::Primary);
    actions(
        &mut graph,
        &simulation,
        &mut state,
        &mut history,
        &input,
        false,
    );
    input.reset_all();

    let mut placements = Vec::new();
    for fine in [false, true] {
        if fine {
            input.press(GameAction::FinePlacement);
        }
        hover(
            &graph,
            &simulation,
            &mut state,
            pointer(2.11, 0.16),
            &input,
            &world,
        );
        assert!(state.weld.error.is_none(), "{:?}", state.weld.error);
        let placement = state.weld.preview.as_ref().unwrap().2;
        let step = if fine { 0.05 } else { 0.25 };
        for coordinate in [placement.translation().x, placement.translation().z] {
            assert!((coordinate / step - (coordinate / step).round()).abs() < 1.0e-4);
        }
        // Pointer movement inside the same grid cell must leave the ghost still.
        hover(
            &graph,
            &simulation,
            &mut state,
            pointer(2.112, 0.162),
            &input,
            &world,
        );
        assert!(
            state
                .weld
                .preview
                .as_ref()
                .unwrap()
                .2
                .translation()
                .abs_diff_eq(placement.translation(), 1.0e-5)
        );
        placements.push(placement);
    }
    assert!(
        !placements[0]
            .translation()
            .abs_diff_eq(placements[1].translation(), 0.01)
    );
    input.press(GameAction::Primary);
    actions(
        &mut graph,
        &simulation,
        &mut state,
        &mut history,
        &input,
        false,
    );
    input.clear();
    hover(
        &graph,
        &simulation,
        &mut state,
        pointer(2.112, 0.162),
        &input,
        &world,
    );
    assert!(
        state
            .weld
            .preview
            .as_ref()
            .unwrap()
            .2
            .translation()
            .abs_diff_eq(placements[1].translation(), 1.0e-5)
    );
    input.release(GameAction::FinePlacement);
    hover(
        &graph,
        &simulation,
        &mut state,
        pointer(2.112, 0.162),
        &input,
        &world,
    );
    assert!(
        state
            .weld
            .preview
            .as_ref()
            .unwrap()
            .2
            .translation()
            .abs_diff_eq(placements[1].translation(), 1.0e-5)
    );
}

#[test]
fn hover_matches_face_grids_on_independently_rotated_and_offset_parts() {
    let (mut graph, source, destination) = scene();
    let source_frame =
        ConstructionFrame::new(Vec3::new(0.037, 0.0, 0.021), Quat::from_rotation_y(0.31)).unwrap();
    let destination_frame = ConstructionFrame::new(
        Vec3::new(0.073, 0.0, 0.019),
        Quat::from_rotation_z(0.19) * Quat::from_rotation_y(0.67),
    )
    .unwrap();
    graph.reframe_parts([source], source_frame).unwrap();
    graph
        .reframe_parts([destination], destination_frame)
        .unwrap();
    let simulation = AppSimulation::default();
    let surface_ray = |frame: ConstructionFrame, x: f32| {
        Ray3d::new(
            frame.point(Vec3::new(x, 9.0, 0.09)),
            Dir3::new(frame.vector(Vec3::NEG_Y)).unwrap(),
        )
    };
    let source_pick = pick(&graph, &simulation, surface_ray(source_frame, 0.03)).unwrap();
    let destination_pick = pick(&graph, &simulation, surface_ray(destination_frame, 2.11)).unwrap();
    let alignment = WeldAlignment::new(source_pick.selection, destination_pick.selection)
        .unwrap()
        .align_tangent_grids()
        .unwrap();
    for fine in [false, true] {
        let displacement =
            initial_displacement(&graph, &source_pick, &destination_pick, alignment, fine);
        let placement = alignment.place(displacement, 0).unwrap();
        // Actual face corners, independent of the helper's chosen lattice origin.
        let source_corner = source_frame.point(Vec3::new(-0.25, 7.25, -0.25));
        let destination_corner = destination_frame.point(Vec3::new(1.75, 7.25, -0.25));
        let offset = placement.point(source_corner) - destination_corner;
        let step = if fine { 0.05 } else { 0.25 };
        for axis in [Vec3::X, Vec3::Z] {
            let coordinate = offset.dot(destination_frame.vector(axis)) / step;
            assert!((coordinate - coordinate.round()).abs() < 1.0e-4);
        }
        let source_axis = placement.vector(source_frame.vector(Vec3::X));
        let destination_axis = destination_frame.vector(Vec3::X);
        let correspondence = source_axis.dot(destination_axis).abs();
        assert!(correspondence < 1.0e-5 || (correspondence - 1.0).abs() < 1.0e-5);
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "complete pointer gesture and history round trip"
)]
fn socket_gesture(kind: mechanic_core::JointKind) {
    let (mut graph, source, destination) = scene();
    let simulation = AppSimulation::default();
    let world = crate::world::WorldRuntime::from_world(&mut World::new());
    let mut state = EditorState {
        placement_bounds: builder::PlacementBounds::GarageBuild,
        ..default()
    };
    let face = FaceRef::part(destination, mechanic_core::FaceKind::PositiveY);
    let socket = crate::editor::build_actions::PlacedBearing {
        source: face,
        anchor: builder::face_geometry_from_ref(face, Some(&graph)).center,
        axis: if matches!(kind, mechanic_core::JointKind::Rotational) {
            Vec3::Y
        } else {
            Vec3::X
        },
        dimensions: mechanic_core::BearingDimensions::default(),
        kind,
    };
    state.placed_bearings.push(socket);
    let destination_ray = if matches!(kind, mechanic_core::JointKind::Rotational) {
        ray(2.08)
    } else {
        ray(2.0)
    };
    let mut history = crate::editor::history::EditorHistory::default();
    let mut input = ButtonInput::default();
    hover(&graph, &simulation, &mut state, ray(0.0), &input, &world);
    input.press(GameAction::Primary);
    actions(
        &mut graph,
        &simulation,
        &mut state,
        &mut history,
        &input,
        false,
    );
    input.clear();
    input.release(GameAction::Primary);
    actions(
        &mut graph,
        &simulation,
        &mut state,
        &mut history,
        &input,
        false,
    );
    assert!(state.weld.busy());
    assert_eq!(graph.weld_count(), 0);
    input.clear();
    hover(
        &graph,
        &simulation,
        &mut state,
        destination_ray,
        &input,
        &world,
    );
    assert!(state.weld.hovered.as_ref().unwrap().socket.is_some());
    assert!(state.weld.preview.is_some());
    assert!(state.weld.error.is_none(), "{:?}", state.weld.error);
    input.press(GameAction::Primary);
    actions(
        &mut graph,
        &simulation,
        &mut state,
        &mut history,
        &input,
        false,
    );
    input.clear();
    hover(
        &graph,
        &simulation,
        &mut state,
        destination_ray,
        &input,
        &world,
    );
    assert!(state.weld.effects_target(&simulation).is_some());
    input.release(GameAction::Primary);
    actions(
        &mut graph,
        &simulation,
        &mut state,
        &mut history,
        &input,
        false,
    );
    assert!(state.weld.effects_target(&simulation).is_none());
    assert_eq!(graph.bearing_count(), 1, "{:?}", state.feedback);
    assert_eq!(history.undo.len(), 1);
    assert!(!state.weld.busy());
    let compiled = graph.compile().unwrap();
    assert_eq!(
        compiled.compounds.len(),
        2,
        "attachment must preserve a moving body"
    );
    assert_eq!(compiled.bearings.len(), 1);
    assert_eq!(graph.bearings().next().unwrap().1.kind, kind);
    assert!(crate::editor::history::apply_history_action(
        crate::editor::history::HistoryAction::Undo,
        &mut graph,
        &mut state,
        &mut history
    ));
    assert!(
        graph
            .part_position(source)
            .unwrap()
            .abs_diff_eq(Vec3::Y * 7.0, 1.0e-5)
    );
    assert!(crate::editor::history::apply_history_action(
        crate::editor::history::HistoryAction::Redo,
        &mut graph,
        &mut state,
        &mut history
    ));
    assert_eq!(graph.bearing_count(), 1);
}

#[test]
fn weld_gesture_attaches_to_rotational_bearing_and_undoes() {
    socket_gesture(mechanic_core::JointKind::Rotational);
}

#[test]
fn weld_gesture_attaches_to_linear_carriage_and_undoes() {
    socket_gesture(mechanic_core::JointKind::Linear(
        mechanic_core::LinearBearing {
            dimensions: mechanic_core::LinearBearingDimensions::default(),
            mount_normal: Vec3::Y,
            face: mechanic_core::CarriageFace::Top,
        },
    ));
}

fn blocks(placements: &[([u8; 3], IVec3)]) -> (ConstructionGraph, Vec<PartId>) {
    let mut graph = ConstructionGraph::new();
    let parts = placements
        .iter()
        .map(|&(size, position)| {
            let BuildOutcome::Spawned(part) = graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(size, BuildPose::new(position, GridRotation::default()))
                        .unwrap(),
                ))
                .unwrap()
            else {
                panic!("spawn expected")
            };
            part
        })
        .collect();
    (graph, parts)
}

/// A base carrying two bearing-mounted arms whose facing sides touch.
fn bearing_loop() -> (ConstructionGraph, PartId, PartId, PartId) {
    let (mut graph, parts) = blocks(&[
        ([6, 2, 2], IVec3::new(0, 28, 0)),
        ([2; 3], IVec3::new(-1, 30, 0)),
        ([2; 3], IVec3::new(1, 30, 0)),
    ]);
    for (arm, x) in [(parts[1], -0.25), (parts[2], 0.25)] {
        graph
            .apply(BuildCommand::AddBearing(mechanic_core::BearingSpec::new(
                FaceRef::part(parts[0], mechanic_core::FaceKind::PositiveY),
                FaceRef::part(arm, mechanic_core::FaceKind::NegativeY),
                Vec3::new(x, 7.25, 0.0),
                Vec3::Y,
            )))
            .unwrap();
    }
    (graph, parts[0], parts[1], parts[2])
}

fn click(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut crate::editor::history::EditorHistory,
    x: f32,
) {
    join_hover(graph, &AppSimulation::default(), state, ray(x));
    let mut input = ButtonInput::default();
    input.press(GameAction::Primary);
    join_actions(graph, state, history, &input, false);
}

#[test]
fn join_welds_two_touching_garage_parts_in_place_with_one_history_entry() {
    let (mut graph, parts) = blocks(&[
        ([2; 3], IVec3::new(0, 28, 0)),
        ([2; 3], IVec3::new(2, 28, 0)),
    ]);
    let frames = parts
        .iter()
        .map(|&part| graph.part_frame(part))
        .collect::<Vec<_>>();
    let mut state = EditorState::default();
    let mut history = crate::editor::history::EditorHistory::default();
    click(&mut graph, &mut state, &mut history, 0.0);
    assert_eq!(state.weld.join_first(), Some(parts[0]));
    join_hover(&graph, &AppSimulation::default(), &mut state, ray(0.5));
    assert_eq!(state.weld.join_valid(), Some(true), "{:?}", state.feedback);
    click(&mut graph, &mut state, &mut history, 0.5);
    assert_eq!(graph.weld_count(), 1, "{:?}", state.feedback);
    assert_eq!(history.undo.len(), 1);
    assert!(!state.weld.busy());
    assert_eq!(
        parts
            .iter()
            .map(|&part| graph.part_frame(part))
            .collect::<Vec<_>>(),
        frames
    );
    assert!(crate::editor::history::apply_history_action(
        crate::editor::history::HistoryAction::Undo,
        &mut graph,
        &mut state,
        &mut history
    ));
    assert_eq!(graph.weld_count(), 0);
}

#[test]
fn join_closes_a_loop_through_bearings() {
    let (mut graph, _, left, right) = bearing_loop();
    let mut state = EditorState::default();
    let mut history = crate::editor::history::EditorHistory::default();
    click(&mut graph, &mut state, &mut history, -0.25);
    click(&mut graph, &mut state, &mut history, 0.25);
    assert_eq!(graph.weld_count(), 1, "{:?}", state.feedback);
    assert_eq!(graph.bearing_count(), 2);
    let creation = graph.compile().unwrap();
    let body = |part| {
        creation
            .part_to_compound
            .iter()
            .find_map(|&(id, body)| (id == part).then_some(body))
    };
    assert_eq!(body(left), body(right));
}

#[test]
fn join_refuses_separated_bodies_and_the_same_body() {
    let (mut graph, _) = blocks(&[
        ([2; 3], IVec3::new(0, 28, 0)),
        ([2; 3], IVec3::new(4, 28, 0)),
    ]);
    let mut state = EditorState::default();
    let mut history = crate::editor::history::EditorHistory::default();
    click(&mut graph, &mut state, &mut history, 0.0);
    for x in [1.0, 0.0] {
        click(&mut graph, &mut state, &mut history, x);
        assert_eq!(state.weld.join_valid(), None);
        assert!(state.weld.busy());
        join_hover(&graph, &AppSimulation::default(), &mut state, ray(x));
        assert_eq!(state.weld.join_valid(), Some(false));
    }
    assert_eq!(graph.weld_count(), 0);
    assert!(history.undo.is_empty());
}

#[test]
fn changing_weld_mode_cancels_the_gesture() {
    let (mut graph, _) = blocks(&[
        ([2; 3], IVec3::new(0, 28, 0)),
        ([2; 3], IVec3::new(2, 28, 0)),
    ]);
    let mut state = EditorState::default();
    let mut history = crate::editor::history::EditorHistory::default();
    sync_mode(&mut state, WeldMode::Join);
    click(&mut graph, &mut state, &mut history, 0.0);
    assert!(state.weld.busy());
    sync_mode(&mut state, WeldMode::Place);
    assert!(!state.weld.busy());
    sync_mode(&mut state, WeldMode::Place);
    assert_eq!(state.feedback.as_deref(), Some("Welder · Place"));
}

#[test]
fn place_mode_refuses_a_destination_in_the_source_creation() {
    let (mut graph, _, _, _) = bearing_loop();
    let simulation = AppSimulation::default();
    let world = crate::world::WorldRuntime::from_world(&mut World::new());
    let mut state = EditorState {
        placement_bounds: builder::PlacementBounds::GarageBuild,
        ..default()
    };
    let mut history = crate::editor::history::EditorHistory::default();
    let mut input = ButtonInput::default();
    hover(&graph, &simulation, &mut state, ray(-0.25), &input, &world);
    input.press(GameAction::Primary);
    actions(
        &mut graph,
        &simulation,
        &mut state,
        &mut history,
        &input,
        false,
    );
    input.clear();
    input.release(GameAction::Primary);
    actions(
        &mut graph,
        &simulation,
        &mut state,
        &mut history,
        &input,
        false,
    );
    input.clear();
    hover(&graph, &simulation, &mut state, ray(0.25), &input, &world);
    let error = state.weld.error.clone().unwrap_or_default();
    assert!(error.contains("Join"), "{error:?}");
    assert_eq!(graph.weld_count(), 0);
}
