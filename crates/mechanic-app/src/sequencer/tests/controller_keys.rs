use super::{driven_arm, keys, steering};
use crate::sequencer::{DriveKeyState, DriveSequencer};
use mechanic_core::{
    BuildCommand, BuildOutcome, BuildPose, ButtonSpec, ControllerKeys, DriveKey,
    InputConfiguration, InputSize,
};

#[test]
fn a_physical_key_matches_seated_keyboard_and_releasing_one_source_keeps_the_state() {
    let (mut graph, _, controller) = driven_arm(steering());
    let BuildOutcome::Spawned(button) = graph
        .apply(BuildCommand::SpawnButton(ButtonSpec::new(
            InputSize::Panel,
            BuildPose::default(),
        )))
        .unwrap()
    else {
        panic!("button")
    };
    let key = DriveKey::new('A').unwrap();
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input: button,
            configuration: InputConfiguration {
                controller: Some(controller),
                key: Some(key),
                ..Default::default()
            },
        })
        .unwrap();
    let creation = graph.compile().unwrap();
    let mut physical = DriveSequencer::default();
    let mut keyboard = DriveSequencer::default();
    physical.start(&creation, &graph, None);
    keyboard.start(&creation, &graph, None);
    let mut combined = ControllerKeys::default();
    combined.press_button(&graph, button);
    combined.update(&graph, []);
    let routed = |combined: &ControllerKeys| DriveKeyState {
        routed: Some(combined.clone()),
        ..Default::default()
    };
    physical.step(&graph, &routed(&combined), None, 1);
    keyboard.step(&graph, &keys(&['A'], &['A']), Some(controller), 1);
    assert_eq!(physical.rows(), keyboard.rows());
    assert_eq!(physical.rows()[0].cursor.active, 1);
    combined.update(&graph, [(controller, key)]);
    physical.step(&graph, &routed(&combined), None, 2);
    combined.release_button(button);
    combined.update(&graph, [(controller, key)]);
    physical.step(&graph, &routed(&combined), None, 3);
    assert_eq!(physical.rows()[0].cursor.active, 1);
    combined.update(&graph, []);
    physical.step(&graph, &routed(&combined), None, 4);
    assert_eq!(physical.rows()[0].cursor.active, 0);
}

#[test]
fn physical_keys_shift_only_matching_unmodified_gearbox_bindings() {
    use mechanic_core::{EngineKind, GearKey, GearKeyChord, ShiftMode};

    for modified in [false, true] {
        let (mut graph, _, mut sequencer, mut gearboxes) = super::gas_drive(2.0, false);
        let controller = graph
            .drive_link(sequencer.rows()[0].link)
            .unwrap()
            .controller;
        graph
            .apply(BuildCommand::SetGearboxMode {
                controller,
                kind: EngineKind::Gas,
                mode: ShiftMode::Manual,
            })
            .unwrap();
        let mut up = GearKeyChord::new(GearKey::Letter('W'));
        up.shift = modified;
        graph
            .apply(BuildCommand::SetGearboxBindings {
                controller,
                kind: EngineKind::Gas,
                up,
                down: GearKeyChord::new(GearKey::Letter('S')),
            })
            .unwrap();
        gearboxes.sync_publication(&graph, &sequencer);
        let BuildOutcome::Spawned(button) = graph
            .apply(BuildCommand::SpawnButton(ButtonSpec::new(
                InputSize::Panel,
                BuildPose::default(),
            )))
            .unwrap()
        else {
            panic!("expected button")
        };
        graph
            .apply(BuildCommand::SetInputConfiguration {
                input: button,
                configuration: InputConfiguration {
                    controller: Some(controller),
                    key: DriveKey::new('W'),
                    ..Default::default()
                },
            })
            .unwrap();
        let mut combined = ControllerKeys::default();
        combined.press_button(&graph, button);
        combined.update(&graph, []);
        sequencer.step(
            &graph,
            &DriveKeyState {
                routed: Some(combined),
                ..Default::default()
            },
            None,
            100,
        );
        let before = gearboxes.active_gear(controller, EngineKind::Gas).unwrap();
        gearboxes.step(
            &graph,
            &sequencer,
            &bevy::input::ButtonInput::default(),
            None,
            100,
            &[],
            false,
        );
        assert_eq!(
            gearboxes.active_gear(controller, EngineKind::Gas),
            Some(before + usize::from(!modified))
        );
    }
}

#[test]
fn seated_modified_keys_do_not_become_unmodified_gearbox_presses() {
    use mechanic_core::{EngineKind, GearKey, GearKeyChord, ShiftMode};
    for shifted in [false, true] {
        let (mut graph, _, mut sequencer, mut gearboxes) = super::gas_drive(2.0, false);
        let controller = graph
            .drive_link(sequencer.rows()[0].link)
            .unwrap()
            .controller;
        graph
            .apply(BuildCommand::SetGearboxMode {
                controller,
                kind: EngineKind::Gas,
                mode: ShiftMode::Manual,
            })
            .unwrap();
        graph
            .apply(BuildCommand::SetGearboxBindings {
                controller,
                kind: EngineKind::Gas,
                up: GearKeyChord::new(GearKey::Letter('W')),
                down: GearKeyChord::new(GearKey::Letter('S')),
            })
            .unwrap();
        gearboxes.sync_publication(&graph, &sequencer);
        let mut keyboard = bevy::input::ButtonInput::default();
        keyboard.press(bevy::prelude::KeyCode::KeyW);
        if shifted {
            keyboard.press(bevy::prelude::KeyCode::ShiftLeft);
        }
        let mut combined = ControllerKeys::default();
        combined.update(&graph, [(controller, DriveKey::new('W').unwrap())]);
        sequencer.step(
            &graph,
            &DriveKeyState {
                routed: Some(combined),
                ..Default::default()
            },
            Some(controller),
            100,
        );
        let before = gearboxes.active_gear(controller, EngineKind::Gas).unwrap();
        gearboxes.step(
            &graph,
            &sequencer,
            &keyboard,
            Some(controller),
            100,
            &[],
            false,
        );
        assert_eq!(
            gearboxes.active_gear(controller, EngineKind::Gas),
            Some(before + usize::from(!shifted))
        );
    }
}
