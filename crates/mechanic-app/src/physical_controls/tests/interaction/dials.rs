//! Physical dial drag and mixed-target selection through the production systems.

use super::*;
use mechanic_core::{
    AnalogMapping, AnalogRange, BearingSpec, CreationDocument, DialSpec, DriveLinkSpec,
    DriveParameter, FaceKind, FaceRef, NumericParameter,
};

fn dial_graph(mixed: bool) -> (ConstructionGraph, NumericParameter) {
    let mut graph = ConstructionGraph::new();
    let cuboid = |dimensions, units| {
        CuboidSpec::new(dimensions, BuildPose::new(units, GridRotation::default())).unwrap()
    };
    // All machinery sits beside the line of sight to the dial at the origin.
    let base = spawn(
        &mut graph,
        BuildCommand::Spawn(cuboid([4, 2, 4], IVec3::new(8, 1, 0))),
    );
    let rotor = spawn(
        &mut graph,
        BuildCommand::Spawn(cuboid([2, 2, 2], IVec3::new(8, 3, 0))),
    );
    let controller = spawn(
        &mut graph,
        BuildCommand::SpawnController(ControllerSpec::new(BuildPose::new(
            IVec3::new(12, 2, 0),
            GridRotation::default(),
        ))),
    );
    let dial = spawn(
        &mut graph,
        BuildCommand::SpawnDial(DialSpec::new(InputSize::Panel, BuildPose::default())),
    );
    let BuildOutcome::BearingAdded(bearing) = graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(base, FaceKind::PositiveY),
            FaceRef::part(rotor, FaceKind::NegativeY),
            Vec3::new(2.0, 0.5, 0.0),
            Vec3::Y,
        )))
        .unwrap()
    else {
        panic!("bearing");
    };
    let BuildOutcome::DriveLinked(link) = graph
        .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
            controller, bearing,
        )))
        .unwrap()
    else {
        panic!("drive");
    };
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
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input: dial,
            configuration: InputConfiguration {
                controller: Some(controller),
                ..default()
            },
        })
        .unwrap();
    let speed = NumericParameter::Drive {
        link,
        parameter: DriveParameter::AngularSpeed(0),
    };
    let mut configuration = graph.input_configuration(dial).unwrap().clone();
    configuration.analog.push(AnalogMapping {
        target: speed,
        range: AnalogRange::new([-10.0, 10.0], false).unwrap(),
    });
    if mixed {
        configuration.analog.push(AnalogMapping {
            target: NumericParameter::Drive {
                link,
                parameter: DriveParameter::TravelMinimum,
            },
            range: AnalogRange::new([-1.0, 1.0], false).unwrap(),
        });
    }
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input: dial,
            configuration,
        })
        .unwrap();
    (graph, speed)
}

fn fixture(mixed: bool) -> (App, NumericParameter, Entity) {
    let (mut app, _, window) = super::fixture(false, false);
    let (graph, speed) = dial_graph(mixed);
    let creation = graph.compile().unwrap();
    let transforms = creation
        .compounds
        .iter()
        .map(|body| GpuTransform {
            position: body.root_translation.extend(0.0).to_array(),
            rotation: body.root_rotation.to_array(),
        })
        .collect();
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let gpu = GpuPhysics::new_with_config(&device, &queue, &creation, GpuPhysicsConfig::default())
        .unwrap();
    app.insert_resource(AppSimulation {
        gpu: Some(gpu),
        creation: Some(creation),
        transforms,
        published_graph: graph.clone(),
        ..default()
    })
    .insert_resource(EditorGraph(graph))
    .add_systems(PreUpdate, crate::physical_controls::capture);
    (app, speed, window)
}

fn move_pointer(app: &mut App, x: f32) {
    app.world_mut()
        .resource_mut::<AccumulatedMouseMotion>()
        .delta = Vec2::new(x, 0.0);
    app.update();
}

fn value(app: &App, target: NumericParameter) -> f32 {
    app.world()
        .resource::<AppSimulation>()
        .effective_graph()
        .numeric_value(target)
        .unwrap()
}

#[test]
fn physical_drag_updates_runtime_without_writing_authored_values() {
    let (mut app, speed, _) = fixture(false);
    press(&mut app);
    assert!(
        app.world()
            .resource::<PhysicalControls>()
            .captures_pointer()
    );
    move_pointer(&mut app, 50.0);
    assert!((value(&app, speed) - 2.0).abs() < 1e-5);
    assert!(app.world().resource::<EditorState>().drive_rows_dirty);
    assert_eq!(
        app.world().resource::<EditorGraph>().0.numeric_value(speed),
        Some(0.0)
    );
    app.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .press(KeyCode::ShiftLeft);
    move_pointer(&mut app, 50.0);
    assert!((value(&app, speed) - 2.2).abs() < 1e-5);
}

#[test]
fn mixed_target_selection_consumes_key_and_writes_only_after_later_movement() {
    let (mut app, speed, _) = fixture(true);
    let before = CreationDocument::from_graph(
        app.world().resource::<AppSimulation>().effective_graph(),
        "runtime",
        &[],
    );
    press(&mut app);
    assert!(
        app.world()
            .resource::<PhysicalControls>()
            .choices()
            .is_some()
    );
    move_pointer(&mut app, 50.0);
    assert_eq!(
        CreationDocument::from_graph(
            app.world().resource::<AppSimulation>().effective_graph(),
            "runtime",
            &[]
        ),
        before
    );
    app.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .press(KeyCode::Enter);
    app.update();
    assert!(
        app.world()
            .resource::<PhysicalControls>()
            .choices()
            .is_none()
    );
    assert!(
        !app.world()
            .resource::<ButtonInput<KeyCode>>()
            .pressed(KeyCode::Enter)
    );
    assert_eq!(
        CreationDocument::from_graph(
            app.world().resource::<AppSimulation>().effective_graph(),
            "runtime",
            &[]
        ),
        before
    );
    move_pointer(&mut app, 50.0);
    assert!((value(&app, speed) - 2.0).abs() < 1e-5);
    let NumericParameter::Drive { link, .. } = speed else {
        unreachable!();
    };
    assert!(
        (value(
            &app,
            NumericParameter::Drive {
                link,
                parameter: DriveParameter::TravelMinimum
            }
        ) - 0.2)
            .abs()
            < 1e-5
    );
}

#[test]
fn released_or_unreachable_dial_stops_capturing_and_ignores_motion() {
    for cancellation in 0..4 {
        let (mut app, speed, window) = fixture(false);
        press(&mut app);
        move_pointer(&mut app, 50.0);
        let previous = value(&app, speed);
        match cancellation {
            0 => app
                .world_mut()
                .resource_mut::<ButtonInput<GameAction>>()
                .release(GameAction::Interact),
            1 => app.world_mut().resource_mut::<PlayerState>().position.z = 5.0,
            2 => {
                app.world_mut()
                    .entity_mut(window)
                    .get_mut::<Window>()
                    .unwrap()
                    .focused = false;
            }
            _ => app
                .world_mut()
                .resource_mut::<crate::pause_menu::PauseMenuState>()
                .open(),
        }
        move_pointer(&mut app, 100.0);
        assert!(
            !app.world()
                .resource::<PhysicalControls>()
                .captures_pointer()
        );
        assert!((value(&app, speed) - previous).abs() < f32::EPSILON);
    }
}
