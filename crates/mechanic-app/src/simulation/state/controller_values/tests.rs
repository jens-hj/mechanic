mod linear;

use super::*;
use bevy::math::{IVec3, Vec3};
use mechanic_core::{
    AnalogMapping, AnalogRange, BearingSpec, BuildCommand, BuildOutcome, BuildPose, ControllerSpec,
    CreationDocument, CuboidSpec, DialSpec, DriveLinkId, DriveLinkSpec, DriveParameter, FaceKind,
    FaceRef, GridRotation, InputConfiguration, InputSize, NumericParameter,
};

fn wired() -> (ConstructionGraph, DriveLinkId) {
    let mut graph = ConstructionGraph::new();
    let cuboid = |dimensions: [u8; 3], units: IVec3| {
        CuboidSpec::new(dimensions, BuildPose::new(units, GridRotation::default()))
            .expect("test dimensions are in range")
    };
    let spawned = |outcome: BuildOutcome| match outcome {
        BuildOutcome::Spawned(part) => part,
        other => panic!("expected a spawn, got {other:?}"),
    };
    let base = spawned(
        graph
            .apply(BuildCommand::Spawn(cuboid([4, 2, 4], IVec3::new(0, 1, 0))))
            .expect("the base spawns"),
    );
    let rotor = spawned(
        graph
            .apply(BuildCommand::Spawn(cuboid([2, 2, 2], IVec3::new(0, 3, 0))))
            .expect("the rotor spawns"),
    );
    let controller = spawned(
        graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::from_half_grid(IVec3::new(2, 5, 0), GridRotation::default()),
            )))
            .expect("the control block spawns"),
    );
    let BuildOutcome::BearingAdded(bearing) = graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(base, FaceKind::PositiveY),
            FaceRef::part(rotor, FaceKind::NegativeY),
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::Y,
        )))
        .expect("the bearing is added")
    else {
        panic!("expected a bearing outcome");
    };
    graph
        .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
            controller, bearing,
        )))
        .expect("the wire is added");
    let link = graph
        .controller_links(controller)
        .next()
        .expect("the wire is there")
        .0;
    (graph, link)
}

fn mapped() -> (AppSimulation, PartId, NumericParameter) {
    let (mut graph, link) = wired();
    let controller = graph.drive_link(link).unwrap().controller;
    let BuildOutcome::Spawned(dial) = graph
        .apply(BuildCommand::SpawnDial(DialSpec::new(
            InputSize::Panel,
            BuildPose::default(),
        )))
        .unwrap()
    else {
        panic!("expected a dial");
    };
    let target = NumericParameter::Drive {
        link,
        parameter: DriveParameter::AngularSpeed(0),
    };
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input: dial,
            configuration: InputConfiguration {
                controller: Some(controller),
                ..InputConfiguration::default()
            },
        })
        .unwrap();
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input: dial,
            configuration: InputConfiguration {
                controller: Some(controller),
                analog: vec![AnalogMapping {
                    target,
                    range: AnalogRange::new([0.0, 20.0], false).unwrap(),
                }],
                ..InputConfiguration::default()
            },
        })
        .unwrap();
    (
        AppSimulation {
            published_graph: graph,
            ..AppSimulation::default()
        },
        dial,
        target,
    )
}

#[test]
fn dial_changes_runtime_without_changing_authored_values() {
    let (mut simulation, dial, target) = mapped();
    let authored = CreationDocument::from_graph(&simulation.published_graph, "authored", &[]);
    simulation.operate_dial(dial, 1.0).unwrap();
    assert_eq!(
        simulation.effective_graph().numeric_value(target),
        Some(20.0)
    );
    assert_eq!(
        CreationDocument::from_graph(&simulation.published_graph, "authored", &[]),
        authored
    );
    simulation.controller_values = None;
    simulation.controller_overrides.clear();
    assert_eq!(
        simulation.effective_graph().numeric_value(target),
        Some(0.0)
    );
}

#[test]
fn invalid_dial_operation_preserves_all_runtime_values() {
    let (mut simulation, dial, _) = mapped();
    simulation.operate_dial(dial, 0.5).unwrap();
    let previous = CreationDocument::from_graph(simulation.effective_graph(), "runtime", &[]);
    assert!(simulation.operate_dial(dial, f32::NAN).is_err());
    assert_eq!(
        CreationDocument::from_graph(simulation.effective_graph(), "runtime", &[]),
        previous
    );
}

#[test]
fn authored_edits_replace_only_changed_targets() {
    let (mut simulation, dial, target) = mapped();
    simulation.operate_dial(dial, 1.0).unwrap();
    let before = simulation.published_graph.clone();
    simulation.reconcile_controller_values(&before, &before);
    assert_eq!(
        simulation.effective_graph().numeric_value(target),
        Some(20.0)
    );
    let mut after = before.clone();
    after.apply_numeric_values(&[(target, 5.0)]).unwrap();
    simulation.reconcile_controller_values(&before, &after);
    assert_eq!(
        simulation.effective_graph().numeric_value(target),
        Some(5.0)
    );
    assert_eq!(simulation.published_graph.numeric_value(target), Some(0.0));
}

#[test]
fn incompatible_manual_edit_preserves_all_runtime_values() {
    let (mut simulation, dial, speed) = mapped();
    let NumericParameter::Drive { link, .. } = speed else {
        unreachable!();
    };
    let graph = &mut simulation.published_graph;
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
    let minimum = NumericParameter::Drive {
        link,
        parameter: DriveParameter::TravelMinimum,
    };
    let maximum = NumericParameter::Drive {
        link,
        parameter: DriveParameter::TravelMaximum,
    };
    let mut configuration = graph.input_configuration(dial).unwrap().clone();
    configuration.analog.push(AnalogMapping {
        target: minimum,
        range: AnalogRange::new([0.0, 2.0], false).unwrap(),
    });
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input: dial,
            configuration,
        })
        .unwrap();
    simulation.operate_dial(dial, 0.25).unwrap();
    assert_eq!(
        simulation.effective_graph().numeric_value(minimum),
        Some(0.5)
    );
    let before = simulation.published_graph.clone();
    let mut after = before.clone();
    after.apply_numeric_values(&[(maximum, 0.25)]).unwrap();
    let previous = CreationDocument::from_graph(simulation.effective_graph(), "runtime", &[]);
    let overrides = simulation.controller_overrides.clone();
    assert!(
        simulation
            .reconcile_controller_edit(&before, &after, &[maximum])
            .is_err()
    );
    assert_eq!(
        CreationDocument::from_graph(simulation.effective_graph(), "runtime", &[]),
        previous,
    );
    assert_eq!(simulation.controller_overrides, overrides);
    assert_eq!(
        simulation.effective_graph().numeric_value(minimum),
        Some(0.5)
    );
    assert_eq!(simulation.effective_graph().numeric_value(speed), Some(5.0));
    assert_eq!(
        simulation.effective_graph().numeric_value(maximum),
        Some(1.0)
    );
}

#[test]
fn dial_updates_shared_solver_drive_rows_without_recompiling() {
    use crate::sequencer::{DriveSequencer, GearboxRuntime, geared_gpu_drive_rows};
    use mechanic_core::{ActuatorAssignment, EngineKind, EngineSpec, WeldSpec};

    let (mut simulation, dial, target) = mapped();
    let NumericParameter::Drive { link, .. } = target else {
        unreachable!();
    };
    let graph = &mut simulation.published_graph;
    let spec = *graph.drive_link(link).unwrap();
    let BuildOutcome::Spawned(engine) = graph
        .apply(BuildCommand::SpawnEngine(EngineSpec::new(
            EngineKind::Electric,
            BuildPose::from_half_grid(IVec3::new(2, 9, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        panic!("engine");
    };
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(spec.controller, FaceKind::PositiveY),
            second: FaceRef::part(engine, FaceKind::NegativeY),
        }))
        .unwrap();
    graph
        .apply(BuildCommand::SetDriveLink {
            link,
            limits: spec.limits,
            program: spec.program,
            name: spec.name,
            actuator: ActuatorAssignment::motor(100, 0).unwrap(),
        })
        .unwrap();
    let creation = graph.compile().unwrap();
    let authored = CreationDocument::from_graph(graph, "authored", &[]);
    let mut sequencer = DriveSequencer::default();
    sequencer.start(&creation, graph, None);
    let mut gearboxes = GearboxRuntime::default();
    gearboxes.start(graph, &sequencer);
    let before = geared_gpu_drive_rows(&creation, graph, &sequencer, &gearboxes);
    simulation.operate_dial(dial, 0.05).unwrap();
    // The CPU step and GPU upload both consume this same row builder, using
    // the original compiled coordinates throughout the numeric operation.
    let after = geared_gpu_drive_rows(
        &creation,
        simulation.effective_graph(),
        &sequencer,
        &gearboxes,
    );
    let coordinate = sequencer.rows()[0].coordinate as usize;
    assert_eq!(before.len(), after.len());
    assert!(before[coordinate].target_speed.abs() < f32::EPSILON);
    assert!((after[coordinate].target_speed - 1.0).abs() < 1e-5);
    assert_eq!(
        CreationDocument::from_graph(&simulation.published_graph, "authored", &[]),
        authored
    );
}
#[test]
fn manual_edit_to_authored_value_clears_runtime_override() {
    let (mut simulation, dial, target) = mapped();
    simulation.operate_dial(dial, 1.0).unwrap();
    let authored = simulation.published_graph.clone();
    simulation
        .reconcile_controller_edit(&authored, &authored, &[target])
        .unwrap();
    assert_eq!(
        simulation.effective_graph().numeric_value(target),
        Some(0.0)
    );
    assert!(!simulation.controller_overrides.contains_key(&target));
}

#[test]
fn unlink_then_unrelated_rename_preserves_runtime_value() {
    let (mut simulation, dial, target) = mapped();
    simulation.operate_dial(dial, 1.0).unwrap();
    let before = simulation.published_graph.clone();
    let mut unlinked = before.clone();
    let mut configuration = unlinked.input_configuration(dial).unwrap().clone();
    configuration.analog.clear();
    unlinked
        .apply(BuildCommand::SetInputConfiguration {
            input: dial,
            configuration,
        })
        .unwrap();
    simulation.reconcile_controller_values(&before, &unlinked);
    assert_eq!(
        simulation.effective_graph().numeric_value(target),
        Some(20.0)
    );
    let mut renamed = unlinked.clone();
    let mut configuration = renamed.input_configuration(dial).unwrap().clone();
    configuration.name = "Renamed dial".into();
    renamed
        .apply(BuildCommand::SetInputConfiguration {
            input: dial,
            configuration,
        })
        .unwrap();
    simulation.reconcile_controller_values(&unlinked, &renamed);
    assert_eq!(
        simulation.effective_graph().numeric_value(target),
        Some(20.0)
    );
    assert_eq!(simulation.controller_overrides.get(&target), Some(&20.0));
    assert_eq!(renamed.numeric_value(target), Some(0.0));
}
