use std::time::Duration;

use crate::scheduler::FixedStepScheduler;
use bevy::prelude::{IVec3, Quat, Vec3};
use mechanic_core::{
    BuildCommand, BuildOutcome, BuildPose, CuboidSpec, GridRotation, PartId, SeatSpec,
    TopologyError,
};
use mechanic_gpu::GpuTransform;

use crate::editor::creation::install_editor_graph;
use crate::editor::history::{EditorHistory, EditorSnapshot, HistoryAction, apply_history_action};
use crate::editor::state::EditorState;
use crate::seat::{
    raycast_seat_interaction, seat_exit_position, seat_surface_distance, seat_world_pose,
};
use crate::showcase;
use crate::simulation::publication::creation_requires_live_physics;
use crate::simulation::state::{AppSimulation, next_simulation_ticks};
use crate::simulation::visuals::visual_snapshot_is_due;
use mechanic_core::ConstructionGraph;

fn graph_with_seat() -> (ConstructionGraph, PartId) {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(seat) = graph
        .apply(BuildCommand::SpawnSeat(SeatSpec::new(BuildPose::default())))
        .unwrap()
    else {
        unreachable!()
    };
    (graph, seat)
}

#[test]
fn terrain_anchored_construction_does_not_start_live_physics() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([2; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };

    assert!(creation_requires_live_physics(&graph.compile().unwrap()));
    assert!(!creation_requires_live_physics(
        &graph.compile_with_static_parts([part]).unwrap()
    ));
}

#[test]
fn stopped_published_construction_supports_seat_interaction() {
    let (graph, seat) = graph_with_seat();
    let creation = graph.compile_with_static_parts([seat]).unwrap();
    let simulation = AppSimulation {
        transforms: vec![GpuTransform {
            position: creation.compounds[0]
                .root_translation
                .extend(0.0)
                .to_array(),
            rotation: Quat::IDENTITY.to_array(),
        }],
        creation: Some(creation),
        published_graph: graph.clone(),
        failure: Some("stopped for test".to_owned()),
        ..AppSimulation::default()
    };

    assert!(!simulation.is_running());
    assert_eq!(
        raycast_seat_interaction(&graph, &simulation, Vec3::Z * 2.0, Vec3::NEG_Z).map(|hit| hit.0),
        Some(seat)
    );
}

#[test]
fn authored_seat_is_interactive_before_physics_publication() {
    let (graph, seat) = graph_with_seat();
    let simulation = AppSimulation::default();

    assert_eq!(
        raycast_seat_interaction(&graph, &simulation, Vec3::Z * 2.0, Vec3::NEG_Z).map(|hit| hit.0),
        Some(seat)
    );
    let (position, rotation) = seat_world_pose(&graph, &simulation, seat).unwrap();
    assert!(position.abs_diff_eq(Vec3::ZERO, 1.0e-6));
    assert!(rotation.abs_diff_eq(Quat::IDENTITY, 1.0e-6));
}

#[test]
fn seat_exit_position_is_just_above_the_seat_surface() {
    let (graph, seat) = graph_with_seat();

    let exit = seat_exit_position(&graph, &AppSimulation::default(), seat).unwrap();

    assert!(exit.abs_diff_eq(Vec3::new(0.0, 0.135, 0.0), 1.0e-6));
}

#[test]
fn framed_seat_pose_matches_authored_and_stopped_published_geometry() {
    let (mut graph, seat) = graph_with_seat();
    let frame = mechanic_core::ConstructionFrame::new(
        Vec3::new(5.0, 3.0, -2.0),
        Quat::from_rotation_y(0.7) * Quat::from_rotation_z(0.2),
    )
    .unwrap();
    graph.reframe_parts([seat], frame).unwrap();
    let (position, rotation) = seat_world_pose(&graph, &AppSimulation::default(), seat).unwrap();
    assert!(position.distance(frame.translation()) < 1.0e-5);
    assert!(rotation.angle_between(frame.rotation()) < 1.0e-3);
    let creation = graph.compile().unwrap();
    let transforms = creation
        .compounds
        .iter()
        .map(|body| GpuTransform {
            position: body.root_translation.extend(0.0).to_array(),
            rotation: body.root_rotation.to_array(),
        })
        .collect();
    let mut simulation = AppSimulation {
        creation: Some(creation),
        transforms,
        published_graph: graph.clone(),
        failure: Some("stopped".to_owned()),
        ..Default::default()
    };
    let (published_position, published_rotation) =
        seat_world_pose(&graph, &simulation, seat).unwrap();
    assert!(published_position.distance(position) < 1.0e-5);
    assert!(published_rotation.angle_between(rotation) < 1.0e-3);
    simulation.transforms[0].position[0] += 2.0;
    assert!(
        seat_world_pose(&graph, &simulation, seat)
            .unwrap()
            .0
            .distance(position + Vec3::X * 2.0)
            < 1.0e-5
    );
    let streaming_focus = crate::world::terrain_streaming_focus(
        &crate::camera::PlayerState {
            position: Vec3::new(-100.0, 0.0, -100.0),
            seat: Some(seat),
            input_captured: true,
        },
        &graph,
        &simulation,
        mechanic_world::FloatingOrigin::default(),
    );
    assert!(
        streaming_focus
            .0
            .distance((position + Vec3::X * 2.0).as_dvec3())
            < 1.0e-5,
        "terrain follows the moving seat instead of its entry point",
    );
    let standing = crate::camera::PlayerState {
        position: Vec3::new(7.0, 2.0, -3.0),
        ..Default::default()
    };
    assert_eq!(
        crate::world::terrain_streaming_focus(
            &standing,
            &graph,
            &simulation,
            mechanic_world::FloatingOrigin::default(),
        ),
        mechanic_world::WorldPosition(standing.position.as_dvec3()),
        "walking terrain still follows player position",
    );
    assert!(
        crate::editor::wiring::wire_end_position(
            &graph,
            &EditorState::default(),
            &simulation,
            crate::editor::wiring::WireEnd::Seat(seat)
        )
        .unwrap()
        .distance(position + Vec3::X * 2.0)
            < 1.0e-5
    );
}

#[test]
fn wire_drag_maps_the_local_pointer_ray_to_world_coordinates() {
    let (mut graph, seat) = graph_with_seat();
    let frame = mechanic_core::ConstructionFrame::new(
        Vec3::new(5.0, 3.0, -2.0),
        Quat::from_rotation_y(0.7),
    )
    .unwrap();
    graph.reframe_parts([seat], frame).unwrap();
    let state = EditorState {
        edit_context: Some(crate::live_edit::EditContext {
            anchor: seat,
            frame: graph.part_frame_id(seat).unwrap(),
            frame_to_world: frame,
        }),
        wire_drag: Some(crate::editor::wiring::WireDrag {
            from: crate::editor::wiring::WireEnd::Seat(seat),
            armed: true,
        }),
        pointer_ray: Some((Vec3::new(1.0, 0.0, 5.0), Vec3::NEG_Z)),
        ..Default::default()
    };
    let (from, to) =
        crate::editor::wiring::wire_drag_endpoints(&graph, &state, &AppSimulation::default())
            .unwrap();
    assert!(from.distance(frame.translation()) < 1.0e-5);
    assert!(to.distance(frame.point(Vec3::X)) < 1.0e-5);
}

#[test]
fn seat_range_is_measured_from_the_player_in_third_person() {
    let (graph, seat) = graph_with_seat();
    let simulation = AppSimulation::default();
    let camera_hit =
        raycast_seat_interaction(&graph, &simulation, Vec3::Z * 6.0, Vec3::NEG_Z).unwrap();
    let player_distance = seat_surface_distance(&graph, &simulation, seat, Vec3::Z * 2.0).unwrap();

    assert!(camera_hit.1 > 3.0);
    assert!(player_distance < 3.0);
}

#[test]
fn app_simulation_stages_catch_up_ticks_up_to_the_backlog_cap() {
    let mut scheduler = FixedStepScheduler::new();
    let mut next_tick = 1;
    let mut backlog = 0;
    let mut dropped = 0;

    // A one-second hitch owes sixty ticks. Only the cap is kept, and the
    // discarded ticks advance the index so simulated time stays a fixed
    // distance behind wall time instead of an ever-growing one.
    assert_eq!(
        next_simulation_ticks(
            &mut scheduler,
            &mut next_tick,
            &mut backlog,
            &mut dropped,
            Duration::from_secs(1),
            false,
            3,
        ),
        31..34
    );
    assert_eq!(scheduler.next_tick(), 61);
    assert_eq!(dropped, 30);
    assert_eq!(backlog, 27);
    assert_eq!(
        next_simulation_ticks(
            &mut scheduler,
            &mut next_tick,
            &mut backlog,
            &mut dropped,
            Duration::from_millis(17),
            false,
            3,
        ),
        34..37
    );
    assert_eq!(backlog, 25);
    assert_eq!(
        next_simulation_ticks(
            &mut scheduler,
            &mut next_tick,
            &mut backlog,
            &mut dropped,
            Duration::ZERO,
            false,
            u64::MAX,
        ),
        37..62
    );
    assert_eq!(next_tick, 62);
    assert_eq!(backlog, 0);
    assert_eq!(dropped, 30, "nothing is dropped once the batch keeps up");
}

#[test]
fn a_simulation_that_keeps_up_drops_no_ticks() {
    let mut scheduler = FixedStepScheduler::new();
    let mut next_tick = 1;
    let mut backlog = 0;
    let mut dropped = 0;
    for _ in 0..600 {
        next_simulation_ticks(
            &mut scheduler,
            &mut next_tick,
            &mut backlog,
            &mut dropped,
            Duration::from_millis(16),
            false,
            u64::MAX,
        );
    }
    assert_eq!(dropped, 0);
    assert_eq!(backlog, 0);
}

#[test]
fn paused_simulation_does_not_advance_or_accumulate_time() {
    let mut scheduler = FixedStepScheduler::new();
    let mut next_tick = 7;
    let mut backlog = 5;
    let mut dropped = 0;
    let scheduler_tick = scheduler.next_tick();

    assert_eq!(
        next_simulation_ticks(
            &mut scheduler,
            &mut next_tick,
            &mut backlog,
            &mut dropped,
            Duration::from_secs(10),
            true,
            3,
        ),
        7..7
    );
    assert_eq!(next_tick, 7);
    assert_eq!(backlog, 5);
    assert_eq!(dropped, 0);
    assert_eq!(scheduler.next_tick(), scheduler_tick);
}

#[test]
fn prototype_meshes_publish_every_second_completed_physics_tick() {
    assert!(!visual_snapshot_is_due(10, 10));
    assert!(!visual_snapshot_is_due(10, 11));
    assert!(visual_snapshot_is_due(10, 12));
    assert!(visual_snapshot_is_due(10, 14));
}

#[test]
fn selected_creation_replaces_editor_and_round_trips_history() {
    let mut graph = ConstructionGraph::new();
    let mut state = EditorState::default();
    let mut history = EditorHistory::default();
    let previous = EditorSnapshot::capture(&graph, &state);
    let preset = showcase::CreationPreset::PendulumGarden256;
    let creation =
        install_editor_graph(&mut graph, showcase::build_preset(preset).unwrap()).unwrap();
    history.commit(previous);
    assert_eq!(graph.part_count(), preset.part_count());
    assert_eq!(creation.compounds.len(), preset.part_count());
    assert!(preset.matches(&graph));

    apply_history_action(HistoryAction::Undo, &mut graph, &mut state, &mut history);
    assert_eq!(graph.part_count(), 0);
    apply_history_action(HistoryAction::Redo, &mut graph, &mut state, &mut history);
    assert!(preset.matches(&graph));
}

#[test]
fn failed_install_preserves_the_current_graph() {
    let mut graph = ConstructionGraph::new();
    let spec = CuboidSpec::new(
        [2, 2, 2],
        BuildPose::new(IVec3::new(0, 1, 0), GridRotation::default()),
    )
    .unwrap();
    graph.apply(BuildCommand::Spawn(spec)).unwrap();

    let result = install_editor_graph(&mut graph, ConstructionGraph::new());
    assert!(matches!(result, Err(TopologyError::EmptyConstruction)));
    assert_eq!(graph.part_count(), 1);
    assert_eq!(graph.parts().next().unwrap().1, &spec);
}
