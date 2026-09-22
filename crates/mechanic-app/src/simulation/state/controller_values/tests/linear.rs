use super::{AppSimulation, ConstructionGraph};
use crate::sequencer::{DriveSequencer, GearboxRuntime, geared_gpu_drive_rows};
use bevy::math::{IVec3, Vec3};
use mechanic_core::{
    ActuatorAssignment, AnalogMapping, AnalogRange, BearingSpec, BuildCommand, BuildOutcome,
    BuildPose, CarriageFace, ControllerSpec, CreationDocument, CuboidSpec, DialSpec, DriveLinkSpec,
    DriveParameter, DriveProgram, DriveState, DriveTarget, FaceKind, FaceRef, GridRotation,
    InputConfiguration, InputSize, JointKind, LinearBearing, LinearBearingDimensions,
    NumericParameter, PartId, ServoSpec, WeldSpec,
};

fn spawn(graph: &mut ConstructionGraph, command: BuildCommand) -> PartId {
    let BuildOutcome::Spawned(part) = graph.apply(command).unwrap() else {
        panic!("expected part");
    };
    part
}

fn mapped_rail() -> (AppSimulation, PartId) {
    let mut graph = ConstructionGraph::new();
    let mut block = |y| {
        spawn(
            &mut graph,
            BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_position_ticks(IVec3::new(0, y, 0), GridRotation::default()),
                )
                .unwrap(),
            ),
        )
    };
    let base = block(850);
    let carriage = block(990);
    let rail = LinearBearing {
        dimensions: LinearBearingDimensions::default(),
        mount_normal: Vec3::Y,
        face: CarriageFace::Top,
    };
    let bearing = BearingSpec::new(
        FaceRef::part(base, FaceKind::PositiveY),
        FaceRef::part(carriage, FaceKind::NegativeY),
        Vec3::new(0.0, 2.25, 0.0),
        Vec3::X,
    )
    .with_kind(JointKind::Linear(rail));
    let BuildOutcome::BearingAdded(bearing) =
        graph.apply(BuildCommand::AddBearing(bearing)).unwrap()
    else {
        panic!("expected bearing")
    };
    let pose = |y| BuildPose::from_half_grid(IVec3::new(40, y, 0), GridRotation::default());
    let controller = spawn(
        &mut graph,
        BuildCommand::SpawnController(ControllerSpec::new(pose(0))),
    );
    let servo = spawn(
        &mut graph,
        BuildCommand::SpawnServo(ServoSpec::new(pose(3))),
    );
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(controller, FaceKind::PositiveY),
            second: FaceRef::part(servo, FaceKind::NegativeY),
        }))
        .unwrap();
    let mut spec = DriveLinkSpec::new_linear(controller, bearing, rail.dimensions.bounds());
    spec.actuator = ActuatorAssignment::Servo;
    spec.program = DriveProgram::new(
        &[DriveState::new(DriveTarget::LinearPosition(0.0)).unwrap()],
        false,
    )
    .unwrap();
    let BuildOutcome::DriveLinked(link) = graph.apply(BuildCommand::AddDriveLink(spec)).unwrap()
    else {
        panic!("expected drive")
    };
    let dial = spawn(
        &mut graph,
        BuildCommand::SpawnDial(DialSpec::new(InputSize::Panel, pose(8))),
    );
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input: dial,
            configuration: InputConfiguration {
                controller: Some(controller),
                ..Default::default()
            },
        })
        .unwrap();
    let mappings = [
        (DriveParameter::LinearPosition(0), [-0.2, 0.2]),
        (DriveParameter::TravelMinimum, [-0.4, -0.25]),
        (DriveParameter::TravelMaximum, [0.25, 0.4]),
    ]
    .map(|(parameter, endpoints)| AnalogMapping {
        target: NumericParameter::Drive { link, parameter },
        range: AnalogRange::new(endpoints, false).unwrap(),
    });
    graph.assign_dial(dial, &mappings).unwrap();
    (
        AppSimulation {
            published_graph: graph,
            ..Default::default()
        },
        dial,
    )
}

#[test]
fn dial_updates_signed_linear_target_and_travel_in_shared_solver_rows_without_recompiling() {
    let (mut simulation, dial) = mapped_rail();
    let graph = &simulation.published_graph;
    let creation = graph.compile().unwrap();
    let authored = CreationDocument::from_graph(graph, "linear", &[]);
    let mut sequencer = DriveSequencer::default();
    sequencer.start(&creation, graph, None);
    let mut gearboxes = GearboxRuntime::default();
    gearboxes.start(graph, &sequencer);
    let coordinate = sequencer.rows()[0].coordinate as usize;
    let before = geared_gpu_drive_rows(&creation, graph, &sequencer, &gearboxes);
    assert!(before[coordinate].target_angle.abs() < f32::EPSILON);
    assert!(before[coordinate].max_acceleration > 0.0);
    // Both solvers consume these rows. Keep the same compiled creation while
    // changing signed metre values and the envelope through the runtime dial.
    for (position, target, minimum, maximum) in [(0.0, -0.2, -0.4, 0.25), (1.0, 0.2, -0.25, 0.4)] {
        assert!(!simulation.operate_dial(dial, position).unwrap());
        let rows = geared_gpu_drive_rows(
            &creation,
            simulation.effective_graph(),
            &sequencer,
            &gearboxes,
        );
        assert_eq!(rows.len(), before.len());
        let row = &rows[coordinate];
        assert!((row.target_angle - target).abs() < 1.0e-5);
        assert!((row.min_angle - minimum).abs() < 1.0e-5);
        assert!((row.max_angle - maximum).abs() < 1.0e-5);
        assert!(row.max_acceleration > 0.0);
    }
    assert_eq!(
        CreationDocument::from_graph(&simulation.published_graph, "linear", &[]),
        authored
    );
}
