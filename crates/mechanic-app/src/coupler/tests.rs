use super::*;
use crate::simulation::state::LivePhysicsState;
use mechanic_core::{BuildCommand, BuildOutcome, BuildPose, CouplerSpec, GridRotation};
use mechanic_gpu::GpuVelocity;

fn fixture(gap: f32) -> (AppSimulation, Couplers) {
    let mut graph = ConstructionGraph::new();
    let mut spawn = |center| {
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::SpawnCoupler(CouplerSpec::new(
                BuildPose::new(center, GridRotation::default()),
            )))
            .unwrap()
        else {
            panic!("spawn")
        };
        part
    };
    let a = spawn(IVec3::ZERO);
    let b = spawn(IVec3::Y * 8);
    let creation = graph.compile().unwrap();
    let poses = vec![
        crate::pose::gpu_transform(Vec3::ZERO, Quat::IDENTITY),
        crate::pose::gpu_transform(
            Vec3::new(0.004, GRID_UNIT_METERS * 0.5 + gap, 0.0),
            Quat::from_rotation_y(0.1) * Quat::from_rotation_x(PI),
        ),
    ];
    let count = poses.len();
    let simulation = AppSimulation {
        creation: Some(creation),
        published_graph: graph,
        live_state: Some(LivePhysicsState {
            tick: 0,
            transforms: poses,
            velocities: vec![
                GpuVelocity {
                    linear: [0.0; 4],
                    angular: [0.0; 4]
                };
                count
            ],
            coordinates: Vec::new(),
        }),
        ..Default::default()
    };
    let couplers = Couplers {
        pairs: vec![Pair {
            first: a,
            second: b,
            quarter: 0.0,
            gripping: None,
        }],
        revision: None,
        pending: None,
        parts: vec![a, b],
    };
    (simulation, couplers)
}

#[test]
fn distant_same_body_and_back_facing_couplers_do_not_capture() {
    let (simulation, _) = fixture(0.04);
    let ends = endpoints(
        &simulation,
        &simulation
            .published_graph
            .parts()
            .filter_map(|(id, spec)| matches!(spec, PartSpec::Coupler(_)).then_some(id))
            .collect::<Vec<_>>(),
    );
    let (a, mut b) = (ends[0], ends[1]);
    assert!(compatible(a, b, CAPTURE_DISTANCE));
    b.body = a.body;
    assert!(!compatible(a, b, CAPTURE_DISTANCE));
    b = ends[1];
    b.rotation = a.rotation;
    assert!(!compatible(a, b, CAPTURE_DISTANCE));
    b = ends[1];
    b.center += Vec3::Y;
    assert!(!compatible(a, b, CAPTURE_DISTANCE));
}

#[test]
fn activation_uses_controller_local_seat_or_button_edges() {
    use mechanic_core::{
        ButtonSpec, ControllerKeys, ControllerSpec, DriveKey, InputConfiguration, InputSize,
    };
    let (simulation, couplers) = fixture(0.04);
    let mut graph = simulation.published_graph;
    let mut spawn = |command| {
        let BuildOutcome::Spawned(id) = graph.apply(command).unwrap() else {
            panic!("spawn")
        };
        id
    };
    let controller = spawn(BuildCommand::SpawnController(ControllerSpec::new(
        BuildPose::default(),
    )));
    let other = spawn(BuildCommand::SpawnController(ControllerSpec::new(
        BuildPose::default(),
    )));
    let button = spawn(BuildCommand::SpawnButton(ButtonSpec::new(
        InputSize::Panel,
        BuildPose::default(),
    )));
    let part = couplers.pairs[0].first;
    let key = DriveKey::new('C').unwrap();
    for input in [part, button] {
        graph
            .apply(BuildCommand::SetInputConfiguration {
                input,
                configuration: InputConfiguration {
                    controller: Some(controller),
                    key: Some(key),
                    ..Default::default()
                },
            })
            .unwrap();
    }
    let mut keys = ControllerKeys::default();
    keys.update(&graph, [(other, key)]);
    assert!(!activated(&graph, &keys, part));
    keys.update(&graph, [(controller, key)]);
    assert!(activated(&graph, &keys, part));
    keys.update(&graph, []);
    keys.press_button(&graph, button);
    keys.update(&graph, []);
    assert!(activated(&graph, &keys, part));
    keys.update(&graph, []);
    assert!(!activated(&graph, &keys, part));
}

#[test]
fn grip_rotates_gradually_from_interleaved_to_square() {
    let (simulation, mut runtime) = fixture(0.04);
    let a = endpoints(
        &simulation,
        &simulation
            .published_graph
            .parts()
            .filter_map(|(id, spec)| matches!(spec, PartSpec::Coupler(_)).then_some(id))
            .collect::<Vec<_>>(),
    )[0];
    let pair = &mut runtime.pairs[0];
    let aligned = pair.target(a, 0);
    pair.gripping = Some(0);
    let middle = pair.target(a, 36);
    let locked = pair.target(a, 1000);
    assert!((aligned.angle_between(locked) - FRAC_PI_8).abs() < 1e-4);
    assert!(aligned.angle_between(middle) > 0.01);
    assert!(middle.angle_between(locked) > 0.01);
    assert!(locked.angle_between(Quat::from_rotation_x(PI)) < 1e-4);
}

#[test]
fn attraction_and_torque_conserve_net_linear_impulse() {
    let (simulation, runtime) = fixture(0.04);
    let rows = impulses(&runtime, &simulation, 0);
    let total: Vec3 = rows.iter().map(|r| Vec3::from_slice(&r.impulse[..3])).sum();
    assert!(total.length() < 1e-6);
    assert!(rows.iter().all(|r| r.impulse.iter().all(|v| v.is_finite())));
}

#[test]
fn cpu_alignment_grip_and_publication_produce_one_rigid_body() {
    let (mut simulation, mut runtime) = fixture(0.04);
    let creation = simulation.creation.as_ref().unwrap();
    let live = simulation.live_state.as_ref().unwrap();
    let mut cpu = crate::cpu_physics::PreparedRoute::new(creation, 1)
        .unwrap()
        .install(1, 0, &live.transforms, &live.velocities, &[])
        .unwrap();
    for tick in 1..=900 {
        if tick == 450 {
            runtime.pairs[0].gripping = Some(tick);
        }
        let rows = impulses(&runtime, &simulation, tick);
        let completed = cpu.step(tick, bevy::math::DVec3::ZERO, &[], &rows).unwrap();
        simulation.live_state = Some(LivePhysicsState {
            tick,
            transforms: completed.transforms,
            velocities: completed.velocities,
            coordinates: completed.coordinates,
        });
    }
    let ends = endpoints(
        &simulation,
        &simulation
            .published_graph
            .parts()
            .filter_map(|(id, spec)| matches!(spec, PartSpec::Coupler(_)).then_some(id))
            .collect::<Vec<_>>(),
    );
    assert!(
        ends[0].face().distance(ends[1].face()) < 0.0005,
        "gap {}",
        ends[0].face().distance(ends[1].face())
    );
    assert!(
        runtime.pairs[0]
            .target(ends[0], 900)
            .angle_between(ends[1].rotation)
            < 0.002
    );
    let pair = runtime.pairs[0];
    let graph = crate::live_weld::stage_coupler(
        &simulation.published_graph,
        &simulation,
        pair.first,
        pair.second,
    )
    .unwrap();
    assert_eq!(graph.compile().unwrap().compounds.len(), 1);
    assert_eq!(graph.rigid_links().count(), 1);
}

#[test]
fn gpu_coupler_impulses_reduce_gap_and_angular_error() {
    use mechanic_gpu::{GpuPhysics, GpuPhysicsConfig};
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Ok(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    else {
        eprintln!("Coupler GPU test skipped: no adapter");
        return;
    };
    eprintln!("Coupler GPU test adapter: {:?}", adapter.get_info());
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let (mut simulation, runtime) = fixture(0.04);
    let creation = simulation.creation.as_ref().unwrap();
    let gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        creation,
        GpuPhysicsConfig {
            collisions_enabled: true,
            ground_plane_enabled: false,
            mechanism_self_collisions: true,
            solver_iterations: 16,
        },
    )
    .unwrap();
    let live = simulation.live_state.as_ref().unwrap();
    gpu.write_body_states(&queue, &live.transforms, &live.velocities)
        .unwrap();
    let before = endpoints(&simulation, &runtime.parts);
    let rows = impulses(&runtime, &simulation, 1);
    gpu.dispatch_tick_with_impulses(&device, &queue, 1, &rows)
        .unwrap();
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
    simulation.live_state.as_mut().unwrap().transforms =
        gpu.read_snapshot_transforms(&device, &queue, 1).unwrap();
    let after = endpoints(&simulation, &runtime.parts);
    assert!(
        after[0].face().distance(after[1].face()) < before[0].face().distance(before[1].face())
    );
    assert!(
        runtime.pairs[0]
            .target(after[0], 1)
            .angle_between(after[1].rotation)
            < runtime.pairs[0]
                .target(before[0], 1)
                .angle_between(before[1].rotation)
    );
}

#[test]
fn pairing_is_exclusive_and_drops_missing_or_separated_endpoints() {
    let (simulation, mut runtime) = fixture(0.04);
    let mut ends = endpoints(&simulation, &runtime.parts);
    runtime.pairs.clear();
    assert!(runtime.track(&ends, &simulation.published_graph, 0));
    assert!(!runtime.track(&ends, &simulation.published_graph, 1));
    assert_eq!(runtime.pairs.len(), 1);
    ends[1].center += Vec3::Y;
    runtime.track(&ends, &simulation.published_graph, 2);
    assert!(runtime.pairs.is_empty());
    ends[1].center -= Vec3::Y;
    runtime.track(&ends, &simulation.published_graph, 3);
    runtime.track(&ends[..1], &simulation.published_graph, 4);
    assert!(runtime.pairs.is_empty());
}
