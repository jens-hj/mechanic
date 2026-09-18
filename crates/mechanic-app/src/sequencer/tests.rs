mod transmission_speed;

use super::{
    DriveKeyState, RowCursor, automatic_shift_destination, chord_just_pressed, initial_gear,
    reversal_is_safe, stepped_cursor,
};
use mechanic_core::{
    DriveDwell, DriveKey, DriveProgram, DriveRelease, DriveState, DriveTarget, DriveTrigger,
    EngineKind, GearKey, GearKeyChord, GearboxConfig,
};

fn key(symbol: char) -> DriveKey {
    DriveKey::new(symbol).expect("test keys are letters")
}

fn keys(held: &[char], pressed: &[char]) -> DriveKeyState {
    DriveKeyState {
        held: held.iter().copied().map(key).collect(),
        pressed: pressed.iter().copied().map(key).collect(),
    }
}

fn angle_state(degrees: f32) -> DriveState {
    DriveState::new(DriveTarget::Angle(degrees.to_radians())).expect("test angle is in range")
}

fn row() -> RowCursor {
    RowCursor::default()
}

/// `S1 0°` · `S2 30° hold A` · `S3 -30° hold D`, both reverting to S1.
fn steering() -> DriveProgram {
    let held = |degrees: f32, symbol: char| {
        angle_state(degrees).with_trigger(Some(DriveTrigger::new(
            key(symbol),
            DriveRelease::RevertTo(0),
        )))
    };
    DriveProgram::new(
        &[angle_state(0.0), held(30.0, 'A'), held(-30.0, 'D')],
        false,
    )
    .expect("steering program is valid")
}

#[test]
fn held_key_reverts_to_its_default_state_on_release() {
    let program = steering();
    let start = row();

    let pressed = stepped_cursor(start, &program, &keys(&['A'], &['A']), 10);
    assert_eq!(pressed.active, 1);
    assert_eq!(pressed.entered_tick, 10);

    // Still held: the row stays put rather than re-entering and resetting
    // its dwell clock.
    let holding = stepped_cursor(pressed, &program, &keys(&['A'], &[]), 30);
    assert_eq!(holding, pressed);

    let released = stepped_cursor(holding, &program, &keys(&[], &[]), 45);
    assert_eq!(
        released.active, 0,
        "letting go returns to the default state"
    );
    assert_eq!(released.entered_tick, 45);
}

#[test]
fn steering_the_other_way_switches_states_without_passing_through_neutral() {
    let program = steering();
    let left = stepped_cursor(row(), &program, &keys(&['A'], &['A']), 5);
    assert_eq!(left.active, 1);

    let right = stepped_cursor(left, &program, &keys(&['D'], &['D']), 6);
    assert_eq!(right.active, 2, "a fresh press wins over the held state");
}

#[test]
fn latched_pose_keys_stay_until_another_key() {
    let latched = |degrees: f32, symbol: char| {
        angle_state(degrees).with_trigger(Some(DriveTrigger::new(key(symbol), DriveRelease::Latch)))
    };
    let program = DriveProgram::new(
        &[latched(30.0, 'Q'), latched(40.0, 'W'), latched(80.0, 'E')],
        false,
    )
    .expect("arm program is valid");

    let pressed = stepped_cursor(row(), &program, &keys(&['W'], &['W']), 4);
    assert_eq!(pressed.active, 1);

    // Released, and many ticks later: a latched state has no dwell and no
    // revert, so it holds.
    let later = stepped_cursor(pressed, &program, &keys(&[], &[]), 4_000);
    assert_eq!(later, pressed);

    let moved = stepped_cursor(later, &program, &keys(&['E'], &['E']), 4_100);
    assert_eq!(moved.active, 2);
}

#[test]
fn dwell_advances_to_its_named_state_and_cycles() {
    // S1 0° key R · S2 90° key S, 2 s -> S3 · S3 -90°, 4 s -> S2.
    let reset =
        angle_state(0.0).with_trigger(Some(DriveTrigger::new(key('R'), DriveRelease::Latch)));
    let forward = angle_state(90.0)
        .with_trigger(Some(DriveTrigger::new(key('S'), DriveRelease::Latch)))
        .with_dwell(Some(
            DriveDwell::new(2.0, Some(2)).expect("dwell is in range"),
        ));
    let back = angle_state(-90.0).with_dwell(Some(
        DriveDwell::new(4.0, Some(1)).expect("dwell is in range"),
    ));
    let program = DriveProgram::new(&[reset, forward, back], false).expect("procedure is valid");

    let started = stepped_cursor(row(), &program, &keys(&['S'], &['S']), 100);
    assert_eq!(started.active, 1);

    // 2 s is 120 ticks: one tick early it holds, on the boundary it hands off.
    let waiting = stepped_cursor(started, &program, &keys(&[], &[]), 219);
    assert_eq!(waiting.active, 1);
    let advanced = stepped_cursor(waiting, &program, &keys(&[], &[]), 220);
    assert_eq!(advanced.active, 2);
    assert_eq!(advanced.entered_tick, 220);

    // 4 s is 240 ticks, and S3 names S2, so the pair cycles forever.
    let cycled = stepped_cursor(advanced, &program, &keys(&[], &[]), 460);
    assert_eq!(cycled.active, 1);
}

#[test]
fn linear_position_sequence_preserves_signed_targets_and_dwell_reversal() {
    let forward = DriveState::new(DriveTarget::LinearPosition(0.2))
        .unwrap()
        .with_trigger(Some(DriveTrigger::new(key('S'), DriveRelease::Latch)))
        .with_dwell(Some(DriveDwell::new(1.0, Some(1)).unwrap()));
    let reverse = DriveState::new(DriveTarget::LinearPosition(-0.2))
        .unwrap()
        .with_dwell(Some(DriveDwell::new(1.0, Some(0)).unwrap()));
    let program = DriveProgram::new(&[forward, reverse], false).unwrap();
    let started = stepped_cursor(row(), &program, &keys(&['S'], &['S']), 10);
    let reversed = stepped_cursor(started, &program, &keys(&[], &[]), 70);
    assert_eq!(reversed.active, 1);
    assert_eq!(
        program.states()[reversed.active as usize].target(),
        DriveTarget::LinearPosition(-0.2)
    );
    assert_eq!(
        stepped_cursor(reversed, &program, &keys(&[], &[]), 130).active,
        0
    );
    assert_eq!(
        DriveTarget::LinearSpeed(0.125).reversed(),
        DriveTarget::LinearSpeed(-0.125)
    );
}

#[test]
fn reset_key_interrupts_a_running_procedure() {
    let reset =
        angle_state(0.0).with_trigger(Some(DriveTrigger::new(key('R'), DriveRelease::Latch)));
    let forward = angle_state(90.0)
        .with_trigger(Some(DriveTrigger::new(key('S'), DriveRelease::Latch)))
        .with_dwell(Some(
            DriveDwell::new(2.0, Some(2)).expect("dwell is in range"),
        ));
    let back = angle_state(-90.0).with_dwell(Some(
        DriveDwell::new(4.0, Some(1)).expect("dwell is in range"),
    ));
    let program = DriveProgram::new(&[reset, forward, back], false).expect("procedure is valid");

    let running = RowCursor {
        active: 2,
        entered_tick: 500,
    };
    let stopped = stepped_cursor(running, &program, &keys(&['R'], &['R']), 520);
    assert_eq!(stopped.active, 0);

    // State 0 has no dwell, so the procedure stays stopped.
    let still = stepped_cursor(stopped, &program, &keys(&[], &[]), 5_000);
    assert_eq!(still, stopped);
}

#[test]
fn a_dwell_without_a_target_walks_the_program_and_wraps_only_when_it_loops() {
    let timed = |degrees: f32| {
        angle_state(degrees)
            .with_dwell(Some(DriveDwell::new(1.0, None).expect("dwell is in range")))
    };
    let states = [timed(0.0), timed(45.0)];

    let once = DriveProgram::new(&states, false).expect("program is valid");
    let advanced = stepped_cursor(row(), &once, &keys(&[], &[]), 60);
    assert_eq!(advanced.active, 1);
    let held = stepped_cursor(advanced, &once, &keys(&[], &[]), 600);
    assert_eq!(held, advanced, "the last state of a one-shot program holds");

    let looping = DriveProgram::new(&states, true).expect("program is valid");
    let wrapped = stepped_cursor(advanced, &looping, &keys(&[], &[]), 600);
    assert_eq!(wrapped.active, 0);
}

/// Grounded base, one hinged arm, and a control block wired to it.
fn driven_arm(
    program: DriveProgram,
) -> (
    mechanic_core::ConstructionGraph,
    mechanic_core::CompiledCreation,
    mechanic_core::PartId,
) {
    use bevy::prelude::{IVec3, Vec3};
    use mechanic_core::{
        BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, ControllerSpec,
        CuboidSpec, DriveLinkSpec, FaceKind, FaceRef, GridRotation, WeldSpec,
    };

    let mut graph = ConstructionGraph::new();
    let mut spawn = |units| {
        let spec =
            CuboidSpec::new([4, 4, 4], BuildPose::new(units, GridRotation::default())).unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        part
    };
    let base = spawn(IVec3::new(0, 2, 0));
    let arm = spawn(IVec3::new(4, 2, 0));
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(base, FaceKind::NegativeY),
            second: FaceRef::ground(),
        }))
        .unwrap();
    let BuildOutcome::BearingAdded(bearing) = graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(base, FaceKind::PositiveX),
            FaceRef::part(arm, FaceKind::NegativeX),
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::X,
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::Spawned(controller) = graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::new(IVec3::new(0, 40, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let mut link = DriveLinkSpec::new(controller, bearing);
    link.program = program;
    graph.apply(BuildCommand::AddDriveLink(link)).unwrap();
    let creation = graph.compile().unwrap();
    (graph, creation, controller)
}

fn gas_drive(
    speed: f32,
    reversed: bool,
) -> (
    mechanic_core::ConstructionGraph,
    mechanic_core::CompiledCreation,
    super::DriveSequencer,
    super::GearboxRuntime,
) {
    use bevy::prelude::IVec3;
    use mechanic_core::{
        ActuatorAssignment, BuildCommand, BuildOutcome, BuildPose, EngineSpec, FaceKind, FaceRef,
        GridRotation, WeldSpec,
    };

    let program = DriveProgram::new(
        &[DriveState::new(DriveTarget::Speed(speed)).unwrap()],
        false,
    )
    .unwrap();
    let (mut graph, _, controller) = driven_arm(program);
    let (link_id, mut link) = graph
        .drive_links()
        .next()
        .map(|(id, link)| (id, *link))
        .unwrap();
    graph.apply(BuildCommand::RemoveDriveLink(link_id)).unwrap();
    link.actuator = ActuatorAssignment::motor(0, 100).unwrap();
    link.reversed = reversed;
    graph.apply(BuildCommand::AddDriveLink(link)).unwrap();
    let BuildOutcome::Spawned(engine) = graph
        .apply(BuildCommand::SpawnEngine(EngineSpec::new(
            EngineKind::Gas,
            BuildPose::new(IVec3::new(0, 42, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(controller, FaceKind::PositiveY),
            second: FaceRef::part(engine, FaceKind::NegativeY),
        }))
        .unwrap();
    let first_spec = graph.next_transmission_spec(engine).unwrap();
    let BuildOutcome::Spawned(first) = graph
        .apply(BuildCommand::AttachTransmission {
            parent: engine,
            spec: first_spec,
        })
        .unwrap()
    else {
        unreachable!()
    };
    let second_spec = graph.next_transmission_spec(first).unwrap();
    graph
        .apply(BuildCommand::AttachTransmission {
            parent: first,
            spec: second_spec,
        })
        .unwrap();

    let creation = graph.compile().unwrap();
    let mut sequencer = super::DriveSequencer::default();
    sequencer.start(&creation, &graph, Some((7, 3)));
    let mut gearboxes = super::GearboxRuntime::default();
    gearboxes.start(&graph, &sequencer);
    (graph, creation, sequencer, gearboxes)
}

#[test]
fn wire_reversal_does_not_select_the_opposite_gas_gear_bank() {
    for reversed in [false, true] {
        let (graph, creation, sequencer, mut gearboxes) = gas_drive(2.0, reversed);
        let controller = graph
            .drive_link(sequencer.rows()[0].link)
            .unwrap()
            .controller;
        let config = graph.gearbox_config(controller, EngineKind::Gas).unwrap();
        assert_eq!(config.ratios(), [4.0, 3.0, 1.0]);
        let keyboard = bevy::input::ButtonInput::default();
        for (tick, measured_speed, expected_gear) in [(21, 10.0, 2), (42, 5.0, 1), (63, 10.0, 2)] {
            gearboxes.step(
                &graph,
                &sequencer,
                &keyboard,
                None,
                tick,
                &[(controller, EngineKind::Gas, measured_speed)],
                false,
            );
            assert_eq!(
                gearboxes.active_gear(controller, EngineKind::Gas),
                Some(expected_gear)
            );
            assert!(expected_gear >= usize::from(config.reverse_gears()));
            let rows = super::geared_gpu_drive_rows(&creation, &graph, &sequencer, &gearboxes);
            assert!(rows[0].source_b_max_acceleration > 0.0);
            assert_eq!(rows[0].target_speed.is_sign_negative(), reversed);
        }
    }
}

#[test]
fn zero_speed_explicitly_disengages_gas_torque() {
    let (graph, creation, sequencer, gearboxes) = gas_drive(0.0, false);

    let rows = super::geared_gpu_drive_rows(&creation, &graph, &sequencer, &gearboxes);
    assert!(rows.iter().all(|row| row.source_b_max_acceleration == 0.0));
}

#[test]
fn frozen_program_retains_remaining_dwell_while_another_controller_advances() {
    use bevy::prelude::{IVec3, Vec3};
    use mechanic_core::{
        BearingSpec, BuildCommand, BuildOutcome, BuildPose, ControllerSpec, CuboidSpec,
        DriveLinkSpec, FaceKind, FaceRef, GridRotation,
    };
    let program = DriveProgram::new(
        &[
            angle_state(0.0).with_dwell(Some(DriveDwell::new(1.0, None).unwrap())),
            angle_state(45.0).with_trigger(Some(DriveTrigger::new(key('W'), DriveRelease::Latch))),
        ],
        false,
    )
    .unwrap();
    let (mut graph, _, frozen) = driven_arm(program);
    let first_bearing = *graph.bearings().next().unwrap().1;
    let mechanic_core::FaceOwner::Part(base) = first_bearing.source.owner else {
        unreachable!()
    };
    let BuildOutcome::Spawned(arm) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [4, 4, 4],
                BuildPose::new(IVec3::new(0, 2, 4), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::BearingAdded(bearing) = graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(base, FaceKind::PositiveZ),
            FaceRef::part(arm, FaceKind::NegativeZ),
            Vec3::new(0.0, 0.5, 0.5),
            Vec3::Z,
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::Spawned(other) = graph
        .apply(BuildCommand::SpawnController(ControllerSpec::new(
            BuildPose::new(IVec3::new(0, 50, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let mut link = DriveLinkSpec::new(other, bearing);
    link.program = program;
    graph.apply(BuildCommand::AddDriveLink(link)).unwrap();
    let creation = graph.compile().unwrap();
    let mut sequencer = super::DriveSequencer::default();
    sequencer.start(&creation, &graph, None);
    assert!(!sequencer.step(&graph, &keys(&[], &[]), None, 20));
    let suspended = std::collections::BTreeSet::from([frozen]);
    assert!(sequencer.step_with_suspension(
        &graph,
        &keys(&['W'], &['W']),
        Some(frozen),
        1020,
        &suspended
    ));
    let frozen_link = graph
        .drive_links()
        .find(|(_, spec)| spec.controller == frozen)
        .unwrap()
        .0;
    let other_link = graph
        .drive_links()
        .find(|(_, spec)| spec.controller == other)
        .unwrap()
        .0;
    assert_eq!(sequencer.active_state(frozen_link), Some(0));
    assert_eq!(sequencer.active_state(other_link), Some(1));
    sequencer.sync_publication(&creation, &graph, Some((2, 1)), 1020);
    assert!(!sequencer.step_with_suspension(&graph, &keys(&[], &[]), None, 2020, &suspended));
    assert!(!sequencer.step(&graph, &keys(&[], &[]), None, 2059));
    assert!(sequencer.step(&graph, &keys(&[], &[]), None, 2060));
    assert_eq!(sequencer.active_state(frozen_link), Some(1));
    sequencer.stop();
    assert_eq!(sequencer.last_step_tick, 0);
}

#[test]
fn external_controller_pauses_only_rows_driving_held_bearings() {
    use bevy::prelude::{IVec3, Vec3};
    use mechanic_core::{
        BearingSpec, BuildCommand, BuildOutcome, BuildPose, CuboidSpec, DriveLinkSpec, FaceKind,
        FaceRef, GridRotation,
    };
    let program = DriveProgram::new(
        &[
            angle_state(0.0).with_dwell(Some(DriveDwell::new(1.0, None).unwrap())),
            angle_state(45.0).with_trigger(Some(DriveTrigger::new(key('W'), DriveRelease::Latch))),
        ],
        false,
    )
    .unwrap();
    let (mut graph, _, controller) = driven_arm(program);
    let (held, first) = graph
        .bearings()
        .next()
        .map(|(id, spec)| (id, *spec))
        .unwrap();
    let mechanic_core::FaceOwner::Part(base) = first.source.owner else {
        unreachable!()
    };
    let BuildOutcome::Spawned(arm) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [4; 3],
                BuildPose::new(IVec3::new(0, 2, 4), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::BearingAdded(other) = graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(base, FaceKind::PositiveZ),
            FaceRef::part(arm, FaceKind::NegativeZ),
            Vec3::new(0.0, 0.5, 0.5),
            Vec3::Z,
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let mut link = DriveLinkSpec::new(controller, other);
    link.program = program;
    graph.apply(BuildCommand::AddDriveLink(link)).unwrap();
    let creation = graph.compile().unwrap();
    let mut sequencer = super::DriveSequencer::default();
    sequencer.start(&creation, &graph, None);
    let held_link = graph
        .drive_links()
        .find(|(_, spec)| spec.bearing == held)
        .unwrap()
        .0;
    let free_link = graph
        .drive_links()
        .find(|(_, spec)| spec.bearing == other)
        .unwrap()
        .0;
    assert!(!sequencer.step(&graph, &keys(&[], &[]), Some(controller), 20));
    let held_bearings = std::collections::BTreeSet::from([held]);
    let no_controllers = std::collections::BTreeSet::new();
    assert!(sequencer.step_with_held_bearings(
        &graph,
        &keys(&['W'], &['W']),
        Some(controller),
        1020,
        &no_controllers,
        &held_bearings
    ));
    assert_eq!(sequencer.active_state(held_link), Some(0));
    assert_eq!(sequencer.active_state(free_link), Some(1));
    sequencer.sync_publication(&creation, &graph, Some((2, 1)), 1020);
    assert!(!sequencer.step_with_held_bearings(
        &graph,
        &keys(&[], &[]),
        Some(controller),
        2020,
        &no_controllers,
        &held_bearings
    ));
    assert!(!sequencer.step(&graph, &keys(&[], &[]), Some(controller), 2059));
    assert!(sequencer.step(&graph, &keys(&[], &[]), Some(controller), 2060));
    assert_eq!(sequencer.active_state(held_link), Some(1));
}

#[test]
fn frozen_gearbox_preserves_cooldown_across_publication_and_resumes() {
    let (graph, _, sequencer, mut gearboxes) = gas_drive(2.0, false);
    let controller = graph
        .drive_link(sequencer.rows()[0].link)
        .unwrap()
        .controller;
    let keyboard = bevy::input::ButtonInput::default();
    let speeds = [(controller, EngineKind::Gas, 10.0)];
    assert!(!gearboxes.step(&graph, &sequencer, &keyboard, None, 10, &speeds, false));
    let suspended = std::collections::BTreeSet::from([controller]);
    assert!(!gearboxes.step_with_suspension(
        &graph, &sequencer, &keyboard, None, 1010, &speeds, false, &suspended
    ));
    assert_eq!(gearboxes.active_gear(controller, EngineKind::Gas), Some(1));
    gearboxes.sync_publication(&graph, &sequencer);
    assert!(!gearboxes.step_with_suspension(
        &graph, &sequencer, &keyboard, None, 2010, &speeds, false, &suspended
    ));
    assert!(!gearboxes.step(&graph, &sequencer, &keyboard, None, 2020, &speeds, false));
    assert!(gearboxes.step(&graph, &sequencer, &keyboard, None, 2021, &speeds, false));
    assert_eq!(gearboxes.active_gear(controller, EngineKind::Gas), Some(2));
    gearboxes.stop();
    assert_eq!(gearboxes.last_step_tick, 0);
}

#[test]
fn frozen_gearbox_does_not_complete_a_pending_reversal() {
    let (graph, _, sequencer, mut gearboxes) = gas_drive(2.0, false);
    let row = gearboxes
        .rows
        .iter_mut()
        .find(|row| row.kind == EngineKind::Gas)
        .unwrap();
    row.gear = None;
    row.pending = Some(0);
    let controller = row.controller;
    let keyboard = bevy::input::ButtonInput::default();
    let suspended = std::collections::BTreeSet::from([controller]);
    assert!(!gearboxes.step_with_suspension(
        &graph,
        &sequencer,
        &keyboard,
        None,
        1000,
        &[],
        false,
        &suspended
    ));
    let row = gearboxes
        .rows
        .iter()
        .find(|row| row.kind == EngineKind::Gas)
        .unwrap();
    assert_eq!(row.gear, None);
    assert_eq!(row.pending, Some(0));
    assert!(gearboxes.step(&graph, &sequencer, &keyboard, None, 1001, &[], false));
    assert_eq!(gearboxes.active_gear(controller, EngineKind::Gas), Some(0));
}

#[test]
fn live_publication_preserves_wire_progress_when_coordinates_change() {
    let (graph, mut creation, controller) = driven_arm(steering());
    let mut sequencer = super::DriveSequencer::default();
    sequencer.start(&creation, &graph, Some((1, 1)));
    sequencer.step(&graph, &keys(&['A'], &['A']), Some(controller), 37);
    // Keep a nonzero dwell epoch independently of the program's bindings.
    sequencer.rows[0].cursor = RowCursor {
        active: 1,
        entered_tick: 37,
    };
    let old = sequencer.rows[0];
    for coordinate in creation.loop_topology.bearing_coordinates.values_mut() {
        *coordinate += 4;
    }
    sequencer.sync_publication(&creation, &graph, Some((2, 1)), 50);
    assert_eq!(sequencer.rows[0].link, old.link);
    assert_eq!(sequencer.rows[0].cursor, old.cursor);
    assert_eq!(sequencer.rows[0].coordinate, old.coordinate + 4);
    assert!(sequencer.is_started_for(Some((2, 1))));
    sequencer.start(&creation, &graph, Some((2, 1)));
    assert_eq!(sequencer.rows[0].cursor, RowCursor::default());
}

#[test]
fn renaming_and_tuning_a_wire_preserves_its_program_progress() {
    let (mut graph, creation, _) = driven_arm(steering());
    let mut sequencer = super::DriveSequencer::default();
    sequencer.start(&creation, &graph, Some((1, 1)));
    let cursor = RowCursor {
        active: 1,
        entered_tick: 37,
    };
    sequencer.rows[0].cursor = cursor;
    let link = sequencer.rows[0].link;
    let spec = *graph.drive_link(link).unwrap();
    graph
        .apply(mechanic_core::BuildCommand::SetDriveLink {
            link,
            limits: mechanic_core::DriveLimits::new(2.0, 12.0, None).unwrap(),
            program: spec.program,
            name: mechanic_core::DriveName::new("Front steering"),
            actuator: spec.actuator,
        })
        .unwrap();
    sequencer.sync_publication(&creation, &graph, Some((2, 1)), 50);
    assert_eq!(sequencer.rows[0].cursor, cursor);
}

#[test]
fn changing_a_wire_program_resets_its_cursor() {
    let (mut graph, creation, _) = driven_arm(steering());
    let mut sequencer = super::DriveSequencer::default();
    sequencer.start(&creation, &graph, Some((1, 1)));
    sequencer.rows[0].cursor = RowCursor {
        active: 1,
        entered_tick: 37,
    };
    let link = sequencer.rows[0].link;
    let spec = *graph.drive_link(link).unwrap();
    graph
        .apply(mechanic_core::BuildCommand::SetDriveLink {
            link,
            limits: spec.limits,
            program: DriveProgram::default(),
            name: spec.name,
            actuator: spec.actuator,
        })
        .unwrap();
    sequencer.sync_publication(&creation, &graph, Some((2, 1)), 50);
    assert_eq!(
        sequencer.rows[0].cursor,
        RowCursor {
            active: 0,
            entered_tick: 50
        }
    );
}

#[test]
fn live_publication_preserves_gear_and_pending_shift_until_config_changes() {
    let (mut graph, _, sequencer, mut gearboxes) = gas_drive(2.0, false);
    let row = gearboxes
        .rows
        .iter_mut()
        .find(|row| row.kind == EngineKind::Gas)
        .unwrap();
    row.gear = Some(2);
    row.pending = Some(1);
    row.last_shift_tick = 57;
    let previous = *row;
    gearboxes.sync_publication(&graph, &sequencer);
    assert_eq!(
        *gearboxes
            .rows
            .iter()
            .find(|row| row.kind == EngineKind::Gas)
            .unwrap(),
        previous
    );
    graph
        .apply(mechanic_core::BuildCommand::SetGearboxMode {
            controller: previous.controller,
            kind: EngineKind::Gas,
            mode: super::ShiftMode::Manual,
        })
        .unwrap();
    gearboxes.sync_publication(&graph, &sequencer);
    let row = gearboxes
        .rows
        .iter()
        .find(|row| row.kind == EngineKind::Gas)
        .unwrap();
    assert_eq!(row.pending, None);
    assert_eq!(row.last_shift_tick, 0);
    assert_eq!(
        row.gear,
        Some(usize::from(
            graph
                .gearbox_config(previous.controller, EngineKind::Gas)
                .unwrap()
                .reverse_gears()
        ))
    );
    gearboxes
        .rows
        .iter_mut()
        .find(|row| row.kind == EngineKind::Gas)
        .unwrap()
        .last_shift_tick = 99;
    gearboxes.start(&graph, &sequencer);
    assert!(gearboxes.rows.iter().all(|row| row.last_shift_tick == 0));
}

#[test]
fn sequencer_rows_are_scoped_to_the_physics_publication() {
    let program = DriveProgram::default();
    let (graph, creation, _) = driven_arm(program);
    let mut sequencer = super::DriveSequencer::default();

    sequencer.start(&creation, &graph, Some((4, 8)));
    assert!(sequencer.is_started_for(Some((4, 8))));
    assert!(!sequencer.is_started_for(Some((5, 8))));
    assert!(!sequencer.is_started_for(Some((4, 9))));
}

#[test]
fn a_key_press_reaches_the_gpu_row_for_that_bearings_coordinate() {
    use mechanic_core::{DriveTarget, DriveTrigger};

    let latched = |speed: f32, symbol: char| {
        DriveState::new(DriveTarget::Speed(speed))
            .expect("speed is in range")
            .with_trigger(Some(DriveTrigger::new(key(symbol), DriveRelease::Latch)))
    };
    let program = DriveProgram::new(&[latched(0.0, 'N'), latched(2.0, 'W')], false)
        .expect("driving program is valid");
    let (graph, creation, controller) = driven_arm(program);

    let mut sequencer = super::DriveSequencer::default();
    sequencer.start(&creation, &graph, None);
    assert_eq!(sequencer.rows().len(), 1, "one driven bearing, one row");

    // State 0 holds the arm still.
    let idle = super::gpu_drive_rows(&creation, &graph, &sequencer);
    assert_eq!(idle.len(), creation.loop_topology.tree_bearings.len());
    assert!(idle[0].target_speed.abs() < f32::EPSILON);

    // A key is ignored until an occupied Seat's Input route selects this
    // controller.
    assert!(!sequencer.step(&graph, &keys(&['W'], &['W']), None, 29));

    // Pressing W enters state 1, and that target reaches the same row.
    assert!(sequencer.step(&graph, &keys(&['W'], &['W']), Some(controller), 30));
    let driving = super::gpu_drive_rows(&creation, &graph, &sequencer);
    assert!((driving[0].target_speed - 2.0).abs() < 1.0e-5);
    assert_eq!(driving[0].mode, mechanic_gpu::DRIVE_MODE_SPEED);

    // Holding the same key changes nothing, so no needless GPU write.
    assert!(!sequencer.step(&graph, &keys(&['W'], &[]), Some(controller), 31));
}

#[test]
fn a_reversed_wire_flips_the_row_the_sequencer_uploads() {
    use mechanic_core::{BuildCommand, DriveTarget};

    let program =
        DriveProgram::new(&[DriveState::new(DriveTarget::Speed(2.0)).unwrap()], false).unwrap();
    let (mut graph, creation, _) = driven_arm(program);
    let (link, spec) = graph
        .drive_links()
        .map(|(id, spec)| (id, *spec))
        .next()
        .unwrap();

    let mut sequencer = super::DriveSequencer::default();
    sequencer.start(&creation, &graph, None);
    let forward = super::gpu_drive_rows(&creation, &graph, &sequencer);
    assert!(forward[0].target_speed > 0.0);

    graph.apply(BuildCommand::RemoveDriveLink(link)).unwrap();
    let mut reversed = spec;
    reversed.reversed = true;
    graph.apply(BuildCommand::AddDriveLink(reversed)).unwrap();
    let mut sequencer = super::DriveSequencer::default();
    sequencer.start(&creation, &graph, None);
    let backward = super::gpu_drive_rows(&creation, &graph, &sequencer);
    assert!((backward[0].target_speed + forward[0].target_speed).abs() < 1.0e-5);
}

#[test]
fn the_panel_key_is_bindable_with_gameplay_conflicts_allowed() {
    assert!(super::drive_key(bevy::prelude::KeyCode::KeyE).is_some());
    assert!(super::drive_key(bevy::prelude::KeyCode::KeyD).is_some());
    assert!(super::drive_key(bevy::prelude::KeyCode::KeyF).is_some());
    assert_eq!(
        super::gear_key(bevy::prelude::KeyCode::KeyE),
        Some(GearKey::Letter('E'))
    );
    assert_eq!(
        super::gear_key(bevy::prelude::KeyCode::Space),
        Some(GearKey::Space)
    );
    assert_eq!(
        super::gear_key(bevy::prelude::KeyCode::PageDown),
        Some(GearKey::PageDown)
    );
}

#[test]
fn a_blocked_keyboard_drives_nothing() {
    let program = steering();
    let mut keyboard = bevy::input::ButtonInput::<bevy::prelude::KeyCode>::default();
    keyboard.press(bevy::prelude::KeyCode::KeyA);

    let typing = DriveKeyState::from_keyboard(&keyboard, true);
    assert_eq!(stepped_cursor(row(), &program, &typing, 10).active, 0);

    let playing = DriveKeyState::from_keyboard(&keyboard, false);
    assert_eq!(stepped_cursor(row(), &program, &playing, 10).active, 1);
}

#[test]
fn automatic_shift_thresholds_have_hysteresis() {
    let config = GearboxConfig::for_depth(3, false);
    let output_speed =
        |rpm: f32, gear: usize| mechanic_core::rpm_to_rad_s(rpm) / config.ratios()[gear];
    assert_eq!(
        automatic_shift_destination(
            EngineKind::Electric,
            &config,
            0,
            output_speed(EngineKind::Electric.no_load_rpm() * 0.85, 0),
        ),
        1,
    );
    assert_eq!(
        automatic_shift_destination(
            EngineKind::Electric,
            &config,
            1,
            output_speed(EngineKind::Electric.no_load_rpm() * 0.40, 1),
        ),
        0,
    );
    assert_eq!(
        automatic_shift_destination(
            EngineKind::Electric,
            &config,
            1,
            output_speed(EngineKind::Electric.no_load_rpm() * 0.60, 1),
        ),
        1,
    );
}

#[test]
fn gas_direction_banks_and_reversal_gate_cover_missing_and_safe_destinations() {
    let mut config = GearboxConfig::for_depth(1, true);
    assert_eq!(initial_gear(&config, EngineKind::Gas, -1.0), Some(0));
    assert_eq!(initial_gear(&config, EngineKind::Gas, 1.0), Some(1));
    assert!(!reversal_is_safe(EngineKind::Gas, &config, 1, 30.0));
    assert!(reversal_is_safe(EngineKind::Gas, &config, 1, 0.01));
    config = mechanic_core::GearboxConfig::new(
        mechanic_core::ShiftMode::Auto,
        config.ratios().to_vec(),
        0,
        config.gear_up(),
        config.gear_down(),
    )
    .unwrap();
    assert_eq!(initial_gear(&config, EngineKind::Gas, -1.0), None);
}

#[test]
fn shift_chords_accept_space_page_keys_and_modifiers() {
    let mut keyboard = bevy::input::ButtonInput::<bevy::prelude::KeyCode>::default();
    keyboard.press(bevy::prelude::KeyCode::ShiftLeft);
    keyboard.press(bevy::prelude::KeyCode::Space);
    let chord = GearKeyChord {
        shift: true,
        ..GearKeyChord::new(GearKey::Space)
    };
    assert!(chord_just_pressed(&keyboard, chord));
    assert!(!chord_just_pressed(
        &keyboard,
        GearKeyChord::new(GearKey::Space)
    ));
}
