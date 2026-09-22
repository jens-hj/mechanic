use super::fixture;
use crate::{
    AnalogMapping, AnalogRange, BuildCommand, CreationDocument, DriveDwell, DriveParameter,
    InputBindingError, NumericParameter,
};

#[test]
fn reversed_dial_updates_speed_and_dwell_together_preserving_handoff() {
    let (mut graph, dial, _, link) = fixture(false);
    let spec = *graph.drive_link(link).unwrap();
    let state = spec
        .program
        .state(0)
        .unwrap()
        .with_dwell(Some(DriveDwell::new(2.0, Some(0)).unwrap()));
    graph
        .apply(BuildCommand::SetDriveLink {
            link,
            limits: spec.limits,
            program: spec.program.with_state(0, state).unwrap(),
            name: spec.name,
            actuator: spec.actuator,
        })
        .unwrap();
    let speed = NumericParameter::Drive {
        link,
        parameter: DriveParameter::AngularSpeed(0),
    };
    let dwell = NumericParameter::Drive {
        link,
        parameter: DriveParameter::Dwell(0),
    };
    let mappings = [
        AnalogMapping {
            target: speed,
            range: AnalogRange::new([-4.0, 4.0], true).unwrap(),
        },
        AnalogMapping {
            target: dwell,
            range: AnalogRange::new([1.0, 5.0], true).unwrap(),
        },
    ];
    let before = graph.clone();
    graph.assign_dial(dial, &mappings).unwrap();
    assert_eq!(graph.numeric_value(speed), before.numeric_value(speed));
    assert_eq!(graph.numeric_value(dwell), Some(2.0));
    for (position, expected_speed, expected_dwell) in
        [(0.0, 4.0, 5.0), (0.25, 2.0, 4.0), (1.0, -4.0, 1.0)]
    {
        assert!(!graph.operate_dial(dial, position).unwrap());
        assert_eq!(graph.numeric_value(speed), Some(expected_speed));
        assert_eq!(graph.numeric_value(dwell), Some(expected_dwell));
        assert_eq!(
            graph
                .drive_link(link)
                .unwrap()
                .program
                .state(0)
                .unwrap()
                .dwell()
                .unwrap()
                .next(),
            Some(0)
        );
    }
    assert_eq!(graph.input_configuration(dial).unwrap().analog, mappings);
    assert_eq!(
        graph.parts().collect::<Vec<_>>(),
        before.parts().collect::<Vec<_>>()
    );
    assert_eq!(
        graph.bearings().collect::<Vec<_>>(),
        before.bearings().collect::<Vec<_>>()
    );
}

#[test]
fn invalid_coupled_dial_travel_rejects_dwell_and_speed_without_partial_writes() {
    let (mut graph, dial, _, link) = fixture(false);
    let spec = *graph.drive_link(link).unwrap();
    let state = spec
        .program
        .state(0)
        .unwrap()
        .with_dwell(Some(DriveDwell::new(2.0, None).unwrap()));
    graph
        .apply(BuildCommand::SetDriveLink {
            link,
            limits: spec.limits.with_angle_limits(Some((-1.0, 1.0))).unwrap(),
            program: spec.program.with_state(0, state).unwrap(),
            name: spec.name,
            actuator: spec.actuator,
        })
        .unwrap();
    let target = |parameter| NumericParameter::Drive { link, parameter };
    graph
        .assign_dial(
            dial,
            &[
                AnalogMapping {
                    target: target(DriveParameter::Dwell(0)),
                    range: AnalogRange::new([1.0, 5.0], false).unwrap(),
                },
                AnalogMapping {
                    target: target(DriveParameter::AngularSpeed(0)),
                    range: AnalogRange::new([0.0, 4.0], false).unwrap(),
                },
                AnalogMapping {
                    target: target(DriveParameter::TravelMinimum),
                    range: AnalogRange::new([-0.5, 0.5], false).unwrap(),
                },
                AnalogMapping {
                    target: target(DriveParameter::TravelMaximum),
                    range: AnalogRange::new([-0.5, 0.5], true).unwrap(),
                },
            ],
        )
        .unwrap();
    assert!(!graph.operate_dial(dial, 0.0).unwrap());
    let before = CreationDocument::from_graph(&graph, "dial", &[]);
    assert!(graph.operate_dial(dial, 1.0).is_err());
    assert_eq!(CreationDocument::from_graph(&graph, "dial", &[]), before);
    assert_eq!(
        graph.numeric_value(target(DriveParameter::Dwell(0))),
        Some(1.0)
    );
    assert_eq!(
        graph.numeric_value(target(DriveParameter::AngularSpeed(0))),
        Some(0.0)
    );
}

#[test]
fn fractional_contribution_endpoints_are_rejected_before_replacing_assignments() {
    let (mut graph, dial, _, link) = fixture(false);
    let target = NumericParameter::Drive {
        link,
        parameter: DriveParameter::ElectricContribution,
    };
    assert_eq!(
        graph.numeric_metadata(target).unwrap().integer_step,
        Some(1.0)
    );
    let before = CreationDocument::from_graph(&graph, "dial", &[]);
    let result = graph.assign_dial(
        dial,
        &[AnalogMapping {
            target,
            range: AnalogRange::new([0.0, 0.5], false).unwrap(),
        }],
    );
    assert_eq!(result, Err(InputBindingError::InvalidRange.into()));
    assert_eq!(CreationDocument::from_graph(&graph, "dial", &[]), before);
}
