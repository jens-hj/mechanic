use super::{controller_at, hinged_pair, spawn_controller};
use crate::{
    AnalogMapping, AnalogRange, BuildCommand, BuildOutcome, BuildPose, ButtonMode, ButtonSpec,
    ConstructionGraph, CreationDocument, DialSpec, DriveKey, DriveLinkId, DriveLinkSpec,
    DriveParameter, InputConfiguration, InputSize, NumericParameter, PartId,
};

mod numeric;

fn fixture(button: bool) -> (ConstructionGraph, PartId, PartId, DriveLinkId) {
    let mut graph = ConstructionGraph::new();
    let bearing = hinged_pair(&mut graph);
    let controller = spawn_controller(&mut graph, controller_at(20));
    let BuildOutcome::DriveLinked(link) = graph
        .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
            controller, bearing,
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let command = if button {
        BuildCommand::SpawnButton(ButtonSpec::new(InputSize::Panel, BuildPose::default()))
    } else {
        BuildCommand::SpawnDial(DialSpec::new(InputSize::Panel, BuildPose::default()))
    };
    let BuildOutcome::Spawned(input) = graph.apply(command).unwrap() else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input,
            configuration: InputConfiguration {
                controller: Some(controller),
                ..InputConfiguration::default()
            },
        })
        .unwrap();
    (graph, input, controller, link)
}

#[test]
fn keys_links_and_modes_round_trip_without_runtime_state() {
    let (mut graph, input, controller, _link) = fixture(true);
    let config = InputConfiguration {
        name: "Launch".to_owned(),
        controller: Some(controller),
        button_mode: ButtonMode::Toggle,
        key: DriveKey::new('w'),
        analog: Vec::new(),
    };
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input,
            configuration: config,
        })
        .unwrap();
    let doc = CreationDocument::from_graph(&graph, "Inputs", &[]);
    let json = ron::to_string(&doc).unwrap();
    let loaded = ron::from_str::<CreationDocument>(&json)
        .unwrap()
        .into_graph()
        .unwrap();
    assert_eq!(
        CreationDocument::from_graph(&loaded.graph, "Inputs", &[]),
        doc
    );
    let (_, config) = loaded.graph.physical_inputs().next().unwrap();
    assert_eq!(config.key, DriveKey::new('W'));
    assert!(!json.contains("label"));
    assert!(!json.contains("pointer"));
    assert!(!json.contains("active_state"));
}

#[test]
fn changing_controller_preserves_key_and_deletion_disconnects() {
    let (mut graph, input, controller, _link) = fixture(true);
    let config = InputConfiguration {
        controller: Some(controller),
        key: DriveKey::new('W'),
        ..InputConfiguration::default()
    };
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input,
            configuration: config.clone(),
        })
        .unwrap();
    let previous = graph.clone();
    let other = spawn_controller(&mut graph, controller_at(40));
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input,
            configuration: InputConfiguration {
                controller: Some(other),
                ..config
            },
        })
        .unwrap();
    assert_eq!(
        graph.input_configuration(input).unwrap().key,
        DriveKey::new('W')
    );
    assert_eq!(
        previous.input_configuration(input).unwrap().controller,
        Some(controller)
    );
    graph.apply(BuildCommand::Remove(other)).unwrap();
    assert_eq!(graph.input_configuration(input).unwrap().controller, None);
}

#[test]
fn each_parameter_has_one_dial_and_failed_bindings_are_atomic() {
    let (mut graph, input, controller, link) = fixture(false);
    let config = InputConfiguration {
        controller: Some(controller),
        analog: vec![AnalogMapping {
            target: NumericParameter::Drive {
                link,
                parameter: DriveParameter::AngularSpeed(0),
            },
            range: AnalogRange::new([0.0, 100.0], false).unwrap(),
        }],
        ..InputConfiguration::default()
    };
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input,
            configuration: config.clone(),
        })
        .unwrap();
    let BuildOutcome::Spawned(other) = graph
        .apply(BuildCommand::SpawnDial(DialSpec::new(
            InputSize::Industrial,
            BuildPose::default(),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input: other,
            configuration: InputConfiguration {
                controller: Some(controller),
                ..InputConfiguration::default()
            },
        })
        .unwrap();
    let before = CreationDocument::from_graph(&graph, "Inputs", &[]);
    assert!(
        graph
            .apply(BuildCommand::SetInputConfiguration {
                input: other,
                configuration: config
            })
            .is_err()
    );
    assert_eq!(CreationDocument::from_graph(&graph, "Inputs", &[]), before);
    graph.apply(BuildCommand::RemoveDriveLink(link)).unwrap();
    assert!(graph.input_configuration(input).unwrap().analog.is_empty());
}

#[test]
fn appending_a_document_remaps_controllers_and_preserves_keys() {
    let (mut graph, input, controller, _link) = fixture(true);
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input,
            configuration: InputConfiguration {
                controller: Some(controller),
                key: DriveKey::new('W'),
                ..InputConfiguration::default()
            },
        })
        .unwrap();
    let mut document = CreationDocument::from_graph(&graph, "Inputs", &[]);
    document.append(document.clone()).unwrap();
    let loaded = document.into_graph().unwrap();
    let configs: Vec<_> = loaded.graph.physical_inputs().collect();
    assert_eq!(configs.len(), 2);
    assert_ne!(configs[0].1.controller, configs[1].1.controller);
    for (_, config) in configs {
        assert_eq!(config.key, DriveKey::new('W'));
        assert!(loaded.graph.is_controller(config.controller.unwrap()));
    }
}

#[test]
fn a_zero_contribution_dial_cannot_reserve_absent_hardware() {
    let (mut graph, input, controller, link) = fixture(false);
    let previous = CreationDocument::from_graph(&graph, "Inputs", &[]);
    let configuration = InputConfiguration {
        controller: Some(controller),
        analog: vec![AnalogMapping {
            target: NumericParameter::Drive {
                link,
                parameter: DriveParameter::ElectricContribution,
            },
            range: AnalogRange::new([0.0, 100.0], false).unwrap(),
        }],
        ..InputConfiguration::default()
    };
    assert_eq!(
        graph.apply(BuildCommand::SetInputConfiguration {
            input,
            configuration
        }),
        Err(crate::GraphError::InputBinding(
            crate::InputBindingError::InsufficientCapacity
        ))
    );
    assert_eq!(
        CreationDocument::from_graph(&graph, "Inputs", &[]),
        previous
    );
}

#[test]
fn dial_writes_all_targets_and_reports_clamping_without_changing_topology() {
    let (mut graph, input, _, link) = fixture(false);
    let target = NumericParameter::Drive {
        link,
        parameter: DriveParameter::AngularSpeed(0),
    };
    let mut config = graph.input_configuration(input).unwrap().clone();
    config.analog.push(AnalogMapping {
        target,
        range: AnalogRange::new([-10.0, 30.0], false).unwrap(),
    });
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input,
            configuration: config,
        })
        .unwrap();
    let before = graph.clone();
    assert!(!graph.operate_dial(input, 0.25).unwrap());
    assert_eq!(graph.numeric_value(target), Some(0.0));
    assert!(!graph.operate_dial(input, 1.0).unwrap());
    assert_eq!(graph.numeric_value(target), Some(30.0));
    assert_eq!(
        graph.parts().collect::<Vec<_>>(),
        before.parts().collect::<Vec<_>>()
    );
    assert_eq!(
        graph.bearings().collect::<Vec<_>>(),
        before.bearings().collect::<Vec<_>>()
    );
    assert!(graph.apply_numeric_values(&[(target, f32::MAX)]).unwrap());
    assert_eq!(
        graph.numeric_value(target),
        Some(crate::MAX_DRIVE_SPEED_RAD_S)
    );
}

#[test]
fn coupled_numeric_edits_validate_final_values_and_reject_without_partial_application() {
    let (mut graph, _, _, link) = fixture(false);
    let spec = *graph.drive_link(link).unwrap();
    graph
        .apply(BuildCommand::SetDriveLink {
            link,
            limits: spec.limits.with_angle_limits(Some((-1.0, 1.0))).unwrap(),
            program: spec.program,
            name: spec.name,
            actuator: spec.actuator,
        })
        .unwrap();
    let target = |parameter| NumericParameter::Drive { link, parameter };
    // Neither endpoint is validated against the old opposite endpoint.
    graph
        .apply_numeric_values(&[
            (target(DriveParameter::TravelMinimum), 2.0),
            (target(DriveParameter::TravelMaximum), 3.0),
        ])
        .unwrap();
    let before = CreationDocument::from_graph(&graph, "atomic", &[]);
    assert!(
        graph
            .apply_numeric_values(&[
                (target(DriveParameter::AngularSpeed(0)), 10.0),
                (target(DriveParameter::TravelMinimum), 4.0),
            ])
            .is_err()
    );
    assert_eq!(CreationDocument::from_graph(&graph, "atomic", &[]), before);
    assert!(
        graph
            .apply_numeric_values(&[(target(DriveParameter::AngularSpeed(0)), f32::NAN)])
            .is_err()
    );
}

#[test]
fn numeric_edits_leave_unedited_narrow_travel_limits_usable() {
    let (mut graph, _, _, link) = fixture(false);
    let spec = *graph.drive_link(link).unwrap();
    graph
        .apply(BuildCommand::SetDriveLink {
            link,
            limits: spec.limits.with_angle_limits(Some((0.0, 0.01))).unwrap(),
            program: spec.program,
            name: spec.name,
            actuator: spec.actuator,
        })
        .unwrap();
    let target = NumericParameter::Drive {
        link,
        parameter: DriveParameter::AngularSpeed(0),
    };
    graph.apply_numeric_values(&[(target, 5.0)]).unwrap();
    assert_eq!(graph.numeric_value(target), Some(5.0));
}

#[test]
fn assignment_replacement_unlink_and_failed_group_leave_values_unchanged() {
    let (mut graph, first, controller, link) = fixture(false);
    let BuildOutcome::Spawned(second) = graph
        .apply(BuildCommand::SpawnDial(DialSpec::new(
            InputSize::Panel,
            BuildPose::default(),
        )))
        .unwrap()
    else {
        panic!("dial")
    };
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input: second,
            configuration: InputConfiguration {
                controller: Some(controller),
                ..Default::default()
            },
        })
        .unwrap();
    let target = NumericParameter::Drive {
        link,
        parameter: DriveParameter::AngularSpeed(0),
    };
    let mapping = AnalogMapping {
        target,
        range: AnalogRange::new([-10.0, 10.0], true).unwrap(),
    };
    let value = graph.numeric_value(target);
    graph.assign_dial(first, &[mapping]).unwrap();
    graph.assign_dial(second, &[mapping]).unwrap();
    assert!(graph.input_configuration(first).unwrap().analog.is_empty());
    assert_eq!(graph.input_configuration(second).unwrap().analog, [mapping]);
    assert_eq!(graph.numeric_value(target), value);
    let before = CreationDocument::from_graph(&graph, "atomic", &[]);
    assert!(
        graph
            .assign_dial(
                first,
                &[
                    mapping,
                    AnalogMapping {
                        target: NumericParameter::Drive {
                            link,
                            parameter: DriveParameter::Dwell(0)
                        },
                        ..mapping
                    }
                ]
            )
            .is_err()
    );
    assert_eq!(CreationDocument::from_graph(&graph, "atomic", &[]), before);
    graph.unlink_dial_targets(&[target]);
    assert!(graph.input_configuration(second).unwrap().analog.is_empty());
    assert_eq!(graph.numeric_value(target), value);
}

#[test]
fn button_key_can_be_configured_before_connecting() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(button) = graph
        .apply(BuildCommand::SpawnButton(ButtonSpec::new(
            InputSize::Panel,
            BuildPose::default(),
        )))
        .unwrap()
    else {
        panic!("button")
    };
    assert_eq!(graph.input_configuration(button).unwrap().key, None);
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input: button,
            configuration: InputConfiguration {
                key: DriveKey::new('a'),
                ..Default::default()
            },
        })
        .unwrap();
    assert_eq!(
        graph
            .input_configuration(button)
            .unwrap()
            .key
            .unwrap()
            .symbol(),
        'A'
    );
}
