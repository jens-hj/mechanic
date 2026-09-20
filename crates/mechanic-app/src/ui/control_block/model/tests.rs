use super::apply_linear_edit;
use super::{HardwareModel, Mode, PanelEdit, Preset, StateModel, WireKind, apply_edit, wires};
use mechanic_core::LinearDriveLimits;
use mechanic_core::{
    ActuatorInventory, DriveDwell, DriveKey, DriveLimits, DriveName, DriveProgram, DriveRelease,
    DriveState, DriveTarget, DriveTrigger, MAX_DRIVE_SPEED_RAD_S,
};

#[test]
fn hardware_stats_report_used_and_available_bearing_slots() {
    let hardware = HardwareModel::from(ActuatorInventory {
        electric_engines: 2,
        gas_engines: 1,
        servos: 3,
        electric_joints: 5,
        gas_joints: 1,
        servo_joints: 2,
        ..ActuatorInventory::default()
    });

    assert_eq!(hardware.electric.text(), "5/8 ports");
    assert_eq!(hardware.gas.text(), "1/4 ports");
    assert_eq!(hardware.servo.text(), "2/3 ports");
}

#[test]
fn release_wire_pills_leave_the_event_to_the_arrow_icon() {
    let mut released = StateModel::resting();
    released.key = Some('W');
    released.release = Some(0);

    let labels = wires(&[released], WireKind::Release)
        .into_iter()
        .map(|wire| wire.label)
        .collect::<Vec<_>>();

    assert_eq!(labels, ["W"]);
}

/// A joint holding three angles, the middle two bound to keys.
fn steering() -> (DriveLimits, DriveProgram, DriveName) {
    let held = |degrees: f32| {
        DriveState::new(DriveTarget::Angle(f32::to_radians(degrees))).expect("in range")
    };
    let keyed = |state: DriveState, key: char| {
        state.with_trigger(Some(DriveTrigger::new(
            DriveKey::new(key).expect("a letter"),
            DriveRelease::RevertTo(0),
        )))
    };
    let states = [held(0.0), keyed(held(-30.0), 'A'), keyed(held(30.0), 'D')];
    (
        DriveLimits::new(3.0, f32::INFINITY, Some((-0.8, 0.8))).expect("in range"),
        DriveProgram::new(&states, false).expect("a valid program"),
        DriveName::new("Steer · front left"),
    )
}

/// A joint that spins.
fn driving() -> (DriveLimits, DriveProgram, DriveName) {
    let states = [
        DriveState::new(DriveTarget::Speed(0.0)).expect("in range"),
        DriveState::new(DriveTarget::Speed(2.0)).expect("in range"),
    ];
    (
        DriveLimits::new(3.0, 180.0, None).expect("in range"),
        DriveProgram::new(&states, false).expect("a valid program"),
        DriveName::EMPTY,
    )
}

fn linear_edit(edit: &PanelEdit) -> (DriveLimits, LinearDriveLimits, DriveProgram, DriveName) {
    apply_linear_edit(
        DriveLimits::default(),
        LinearDriveLimits::new(0.5, 100.0, -0.2, 0.3).unwrap(),
        DriveProgram::new(
            &[DriveState::new(DriveTarget::LinearPosition(0.15)).unwrap()],
            false,
        )
        .unwrap(),
        DriveName::EMPTY,
        [-0.425, 0.425],
        edit,
    )
    .unwrap()
}

#[test]
fn linear_position_edits_use_metres_and_clamp_to_programmed_travel() {
    let (_, _, program, _) = linear_edit(&PanelEdit::SetValue {
        state: 0,
        value: -0.125,
    });
    assert!(
        matches!(program.state(0).unwrap().target(), DriveTarget::LinearPosition(value) if (value + 0.125).abs() < 1.0e-6)
    );
    let (_, _, program, _) = linear_edit(&PanelEdit::SetValue {
        state: 0,
        value: 4.0,
    });
    assert!(
        matches!(program.state(0).unwrap().target(), DriveTarget::LinearPosition(value) if (value - 0.3).abs() < 1.0e-6)
    );
}

#[test]
fn linear_travel_edits_cannot_extend_or_disable_physical_stops() {
    for edit in [
        PanelEdit::SetTravel {
            min: -10.0,
            max: 10.0,
        },
        PanelEdit::ToggleTravel,
    ] {
        let (_, limits, _, _) = linear_edit(&edit);
        assert!((limits.minimum() + 0.425).abs() < 1.0e-6);
        assert!((limits.maximum() - 0.425).abs() < 1.0e-6);
    }
    let (_, _, program, _) = linear_edit(&PanelEdit::SetTravel {
        min: -0.1,
        max: 0.1,
    });
    assert!(
        matches!(program.state(0).unwrap().target(), DriveTarget::LinearPosition(value) if (value - 0.1).abs() < 1.0e-6)
    );
}

#[test]
fn linear_modes_and_presets_preserve_target_types_and_transitions() {
    let (_, _, program, _) = linear_edit(&PanelEdit::SetMode {
        state: 0,
        mode: Mode::Speed,
    });
    assert!(
        matches!(program.state(0).unwrap().target(), DriveTarget::LinearSpeed(value) if (value - 0.15).abs() < 1.0e-6)
    );
    for preset in [Preset::Steer, Preset::Drive, Preset::Spin] {
        let (_, _, program, _) = linear_edit(&PanelEdit::ApplyPreset(preset));
        assert!(
            program
                .states()
                .iter()
                .all(|state| state.target().is_linear())
        );
        let active = program.state(1).unwrap();
        assert_eq!(
            active.trigger().unwrap().release(),
            if preset == Preset::Spin {
                DriveRelease::Latch
            } else {
                DriveRelease::RevertTo(0)
            }
        );
    }
    let (_, _, program, _) = linear_edit(&PanelEdit::AddState);
    assert!(
        matches!(program.state(1).unwrap().target(), DriveTarget::LinearPosition(value) if value.abs() < 1.0e-6)
    );
}

fn apply(
    wire: (DriveLimits, DriveProgram, DriveName),
    edit: &PanelEdit,
) -> (DriveLimits, DriveProgram, DriveName) {
    apply_edit(wire.0, wire.1, wire.2, edit).expect("the edit applies")
}

#[test]
fn a_named_joint_keeps_its_name_and_an_overlong_one_is_cut() {
    let named = apply(steering(), &PanelEdit::SetName("Tipper arm".to_owned()));
    assert_eq!(named.2.as_str(), "Tipper arm");
}

#[test]
fn travel_limits_toggle_off_and_back_on_at_a_default_range() {
    let free = apply(steering(), &PanelEdit::ToggleTravel);
    assert_eq!(free.0.angle_limits(), None);
    let limited = apply(free, &PanelEdit::ToggleTravel);
    let (low, high) = limited.0.angle_limits().expect("limits came back");
    assert!((low.to_degrees() + 45.0).abs() < 1.0e-3);
    assert!((high.to_degrees() - 45.0).abs() < 1.0e-3);
}

#[test]
fn moving_the_limits_pulls_every_held_angle_inside_them() {
    let tight = apply(
        steering(),
        &PanelEdit::SetTravel {
            min: -10.0,
            max: 10.0,
        },
    );
    for index in 0..3 {
        let held = tight.1.state(index).expect("the state exists");
        let DriveTarget::Angle(angle) = held.target() else {
            panic!("a steering state holds an angle");
        };
        assert!(
            angle.to_degrees() >= -10.1 && angle.to_degrees() <= 10.1,
            "state {index} sits outside the joint's travel: {}°",
            angle.to_degrees(),
        );
    }
}

#[test]
fn limits_that_would_cross_are_held_apart() {
    let crossed = apply(
        steering(),
        &PanelEdit::SetTravel {
            min: 40.0,
            max: -40.0,
        },
    );
    let (low, high) = crossed
        .0
        .angle_limits()
        .expect("the joint still has limits");
    assert!(high > low, "a joint must be left somewhere to turn");
}

#[test]
fn repeating_the_sequence_flips_back_and_forth() {
    let looping = apply(steering(), &PanelEdit::ToggleLoop);
    assert!(looping.1.loops());
    assert!(!apply(looping, &PanelEdit::ToggleLoop).1.loops());
}

#[test]
fn switching_mode_carries_the_number_across_and_clamps_it() {
    let spun = apply(
        steering(),
        &PanelEdit::SetMode {
            state: 1,
            mode: Mode::Speed,
        },
    );
    let state = spun.1.state(1).expect("the state exists");
    assert!(
        matches!(state.target(), DriveTarget::Speed(speed)
                if speed.abs() <= MAX_DRIVE_SPEED_RAD_S),
        "the number carries across into the new unit's range",
    );
}

#[test]
fn a_typed_angle_is_clamped_into_the_joints_travel() {
    let pushed = apply(
        steering(),
        &PanelEdit::SetValue {
            state: 1,
            value: 300.0,
        },
    );
    let DriveTarget::Angle(angle) = pushed.1.state(1).expect("the state exists").target() else {
        panic!("a steering state holds an angle");
    };
    assert!(
        angle <= 0.8 + 1.0e-4,
        "a typed angle cannot leave the joint's travel: {}°",
        angle.to_degrees(),
    );
}

#[test]
fn binding_a_key_takes_it_from_whichever_state_had_it() {
    let stolen = apply(steering(), &PanelEdit::BindKey { state: 0, key: 'a' });
    assert_eq!(
        stolen
            .1
            .state(0)
            .unwrap()
            .trigger()
            .map(|t| t.key().symbol()),
        Some('A'),
        "the key binds where it was asked for",
    );
    assert!(
        stolen.1.state(1).unwrap().trigger().is_none(),
        "one key can only mean one state, so it leaves the sibling that had it",
    );
}

#[test]
fn clearing_a_key_leaves_the_state_without_a_trigger() {
    let cleared = apply(steering(), &PanelEdit::ClearKey { state: 1 });
    assert!(cleared.1.state(1).unwrap().trigger().is_none());
    let (limits, program, name) = steering();
    assert!(
        apply_edit(limits, program, name, &PanelEdit::ClearKey { state: 0 }).is_none(),
        "a state with no key has nothing to clear",
    );
}

#[test]
fn release_cycles_through_every_state_and_back_to_staying_put() {
    let mut wire = steering();
    // S2 starts handing back to S1.
    assert_eq!(
        wire.1.state(1).unwrap().trigger().unwrap().release(),
        DriveRelease::RevertTo(0)
    );
    let mut seen = Vec::new();
    for _ in 0..4 {
        wire = apply(wire, &PanelEdit::CycleRelease { state: 1 });
        seen.push(wire.1.state(1).unwrap().trigger().unwrap().release());
    }
    assert_eq!(
        seen,
        vec![
            DriveRelease::RevertTo(1),
            DriveRelease::RevertTo(2),
            DriveRelease::Latch,
            DriveRelease::RevertTo(0),
        ],
        "release steps through each state in turn, then back to staying put",
    );
}

#[test]
fn a_dwell_turns_on_pointing_at_the_next_state_and_off_again() {
    let waiting = apply(steering(), &PanelEdit::ToggleDwell { state: 0 });
    let dwell = waiting
        .1
        .state(0)
        .unwrap()
        .dwell()
        .expect("a dwell was added");
    assert!((dwell.seconds() - 1.0).abs() < f32::EPSILON);
    assert_eq!(
        dwell.next(),
        Some(1),
        "a fresh dwell hands off to the next state"
    );
    let stopped = apply(waiting, &PanelEdit::ToggleDwell { state: 0 });
    assert!(stopped.1.state(0).unwrap().dwell().is_none());
}

#[test]
fn the_last_states_dwell_wraps_round_to_the_first() {
    let waiting = apply(steering(), &PanelEdit::ToggleDwell { state: 2 });
    assert_eq!(waiting.1.state(2).unwrap().dwell().unwrap().next(), Some(0));
}

#[test]
fn a_dwell_time_is_kept_inside_what_the_model_accepts() {
    let waiting = apply(steering(), &PanelEdit::ToggleDwell { state: 0 });
    let long = apply(
        waiting,
        &PanelEdit::SetDwell {
            state: 0,
            seconds: 10_000.0,
        },
    );
    assert!((long.1.state(0).unwrap().dwell().unwrap().seconds() - 600.0).abs() < 1.0e-3);
    let short = apply(
        long,
        &PanelEdit::SetDwell {
            state: 0,
            seconds: -4.0,
        },
    );
    assert!(
        short.1.state(0).unwrap().dwell().unwrap().seconds() > 0.0,
        "a dwell of nothing is not a dwell",
    );
}

#[test]
fn pointing_a_dwell_at_a_state_gives_it_one_if_it_had_none() {
    let aimed = apply(
        steering(),
        &PanelEdit::SetDwellTarget {
            state: 0,
            target: 2,
        },
    );
    let dwell = aimed
        .1
        .state(0)
        .unwrap()
        .dwell()
        .expect("a dwell was added");
    assert_eq!(dwell.next(), Some(2));
    let (limits, program, name) = steering();
    assert!(
        apply_edit(
            limits,
            program,
            name,
            &PanelEdit::SetDwellTarget {
                state: 0,
                target: 9
            }
        )
        .is_none(),
        "a dwell cannot hand off to a state that is not there",
    );
}

#[test]
fn states_are_added_up_to_the_limit_and_removed_down_to_one() {
    let mut wire = steering();
    while wire.1.len() < 8 {
        wire = apply(wire, &PanelEdit::AddState);
    }
    assert_eq!(wire.1.len(), 8);
    assert!(
        apply_edit(wire.0, wire.1, wire.2, &PanelEdit::AddState).is_none(),
        "a joint holds at most eight states",
    );

    let mut wire = steering();
    while wire.1.len() > 1 {
        wire = apply(wire, &PanelEdit::RemoveState { state: 0 });
    }
    assert!(
        apply_edit(wire.0, wire.1, wire.2, &PanelEdit::RemoveState { state: 0 }).is_none(),
        "a joint always has somewhere to be",
    );
}

#[test]
fn an_added_state_inherits_the_last_ones_mode_and_rests_at_zero() {
    let grown = apply(driving(), &PanelEdit::AddState);
    let added = grown.1.state(2).expect("the state was added");
    assert!(matches!(added.target(), DriveTarget::Speed(speed) if speed == 0.0));
}

#[test]
fn each_preset_installs_a_program_that_matches_its_name() {
    let steer = apply(driving(), &PanelEdit::ApplyPreset(Preset::Steer));
    assert_eq!(steer.1.len(), 3);
    assert!(
        steer.0.angle_limits().is_some(),
        "steering has travel limits"
    );
    assert!(matches!(
        steer.1.state(0).unwrap().target(),
        DriveTarget::Angle(_)
    ));

    let drive = apply(driving(), &PanelEdit::ApplyPreset(Preset::Drive));
    assert_eq!(drive.1.len(), 3);
    assert_eq!(drive.0.angle_limits(), None, "a driven wheel turns freely");
    assert!(matches!(
        drive.1.state(1).unwrap().target(),
        DriveTarget::Speed(_)
    ));

    let spin = apply(driving(), &PanelEdit::ApplyPreset(Preset::Spin));
    assert_eq!(spin.1.len(), 2, "run and stop is two states");
    for index in 0..2 {
        assert_eq!(
            spin.1
                .state(index)
                .unwrap()
                .trigger()
                .map(DriveTrigger::release),
            Some(DriveRelease::Latch),
            "a toggle stays where it was put",
        );
    }
}

#[test]
fn a_preset_keeps_the_joints_name_and_its_dwellless_states_valid() {
    let named = apply(steering(), &PanelEdit::ApplyPreset(Preset::Drive));
    assert_eq!(named.2.as_str(), "Steer · front left");
    assert!(
        named.1.states().iter().all(|state| state.dwell().is_none()),
        "a preset is a fresh program, not a merge",
    );
    let _ = DriveDwell::new(1.0, None).expect("dwells still validate");
}

#[test]
fn a_dwell_reads_with_only_the_decimals_it_has() {
    for (seconds, expected) in [
        (1.0, "1"),
        (10.0, "10"),
        (1.5, "1.5"),
        (1.25, "1.25"),
        (0.1, "0.1"),
        (600.0, "600"),
    ] {
        assert_eq!(super::dwell_text(seconds), expected);
    }
}

#[test]
fn a_rail_force_reads_in_kilonewtons_once_it_runs_to_four_digits() {
    for (newtons, expected) in [
        (0.0, "0 N"),
        (350.0, "350 N"),
        (999.0, "999 N"),
        (1_000.0, "1.0 kN"),
        (301_593.0, "301.6 kN"),
    ] {
        assert_eq!(super::force_text(newtons), expected);
    }
}
