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
    let mut history = crate::EditorHistory::default();
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
    input.release(GameAction::Primary);
    actions(
        &mut graph,
        &simulation,
        &mut state,
        &mut history,
        &input,
        false,
    );
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
    assert!(crate::apply_history_action(
        crate::HistoryAction::Undo,
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
    assert!(crate::apply_history_action(
        crate::HistoryAction::Redo,
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
        let mut history = crate::EditorHistory::default();
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
            crate::cancel_one_world_escape_owner(&mut graph, &mut state);
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
        live_state: Some(crate::LivePhysicsState {
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
    let mut history = crate::EditorHistory::default();
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
#[allow(clippy::too_many_lines)]
fn initial_hover_snaps_both_axes_and_drag_keeps_the_snapped_alignment() {
    let (mut graph, _, _) = scene();
    let simulation = AppSimulation::default();
    let world = crate::world::WorldRuntime::from_world(&mut World::new());
    let mut state = EditorState {
        placement_bounds: builder::PlacementBounds::GarageBuild,
        ..default()
    };
    let mut history = crate::EditorHistory::default();
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
