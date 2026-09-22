//! Meshes: gears, racks, worms and a differential turn at their tooth ratios.

use super::{World, spawn};
use bevy_math::{DVec3, IVec3, Vec3};
use mechanic_core::{
    BearingId, BearingSpec, BuildCommand, BuildOutcome, BuildPose, CompiledCreation,
    ConstructionGraph, CoordinateDrive, CuboidSpec, CylinderDimensions, CylinderSpec, DriveMode,
    FaceKind, FaceRef, GearKind, GearLinkSpec, GearSpec, GridRotation, PartId, RackSpec,
    RigidLinkSpec, SpiralEnd,
};

use crate::MachineState;

/// Local Y to world Z.
const Y_TO_Z: GridRotation = GridRotation::new(1, 0, 0);
/// Local Y to world −Z.
const Y_TO_NEG_Z: GridRotation = GridRotation::new(3, 0, 0);
/// Local Y to world X.
const Y_TO_X: GridRotation = GridRotation::new(0, 0, 3);
/// Local Y to world −X.
const Y_TO_NEG_X: GridRotation = GridRotation::new(0, 0, 1);

fn spawned(outcome: BuildOutcome) -> PartId {
    let BuildOutcome::Spawned(part) = outcome else {
        panic!("spawn expected, got {outcome:?}")
    };
    part
}

/// A toothed cylinder one block long whose tips are its outer wall.
fn gear(
    graph: &mut ConstructionGraph,
    spec: GearSpec,
    ticks: IVec3,
    rotation: GridRotation,
) -> PartId {
    let (outer, inner) = if spec.is_internal() {
        (spec.tip_diameter() + 0.1, spec.tip_diameter())
    } else {
        (spec.tip_diameter(), 0.0)
    };
    let cylinder = CylinderSpec::new(
        CylinderDimensions::new(outer, inner, 0.25).unwrap(),
        BuildPose::from_position_ticks(ticks, rotation),
    )
    .with_gear(spec)
    .unwrap();
    spawned(graph.apply(BuildCommand::SpawnCylinder(cylinder)).unwrap())
}

fn spur(module_ticks: u8, teeth: u16) -> GearSpec {
    GearSpec::new(module_ticks, teeth, GearKind::Spur).unwrap()
}

/// A hinge whose axis is the source face's normal.
fn hinge(
    graph: &mut ConstructionGraph,
    source: FaceRef,
    target: FaceRef,
    anchor: Vec3,
    axis: Vec3,
) -> BearingId {
    // A small ring, so it also sits on the 10 cm worm's end.
    let dimensions = mechanic_core::BearingDimensions::new(0.08, 0.0).unwrap();
    match graph
        .apply(BuildCommand::AddBearing(
            BearingSpec::new(source, target, anchor, axis).with_dimensions(dimensions),
        ))
        .unwrap()
    {
        BuildOutcome::BearingAdded(id) => id,
        other => panic!("bearing expected, got {other:?}"),
    }
}

fn mesh(graph: &mut ConstructionGraph, first: PartId, second: PartId) {
    graph
        .apply(BuildCommand::AddGearLink(GearLinkSpec { first, second }))
        .unwrap();
}

fn weld(graph: &mut ConstructionGraph, first: PartId, second: PartId) {
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec { first, second }))
        .unwrap();
}

fn coordinate(creation: &CompiledCreation, bearing: BearingId) -> usize {
    creation.loop_topology.bearing_coordinates[&bearing] as usize
}

fn spin(
    creation: &CompiledCreation,
    coordinate: usize,
    effort: f32,
    target: f32,
) -> CoordinateDrive {
    let acceleration = effort / creation.loop_topology.coordinate_axis_inertia[coordinate];
    CoordinateDrive {
        mode: DriveMode::Speed,
        target_speed: target,
        max_speed: 100.0,
        max_acceleration: acceleration,
        source_a_max_acceleration: acceleration,
        source_a_no_load_speed: 100.0,
        ..CoordinateDrive::PASSIVE
    }
}

fn speed(world: &World, coordinate: usize) -> f64 {
    let row = world.creation().dynamics.coordinate_velocities[coordinate];
    world.machine.snapshot().state.velocities[row]
}

/// Runs `ticks` without gravity, driving `coordinate` at `target` rad/s.
fn drive(world: &mut World, drives: &[(usize, CoordinateDrive)], ticks: usize) {
    let commands = drives
        .iter()
        .map(|&(coordinate, drive)| world.command(coordinate, drive))
        .collect::<Vec<_>>();
    world.tick_with(DVec3::ZERO, &commands);
    for _ in 1..ticks {
        world.tick(DVec3::ZERO);
    }
}

/// A 24-tooth pinion and a 36-tooth wheel, module 1 cm, on hinges to a static
/// plate: the pinion's and wheel's coordinates.
fn gear_train() -> (CompiledCreation, [usize; 2]) {
    let mut graph = ConstructionGraph::new();
    let plate = spawn(&mut graph, IVec3::new(0, 400, 0), [8, 1, 4]);
    let pinion = gear(
        &mut graph,
        spur(4, 24),
        IVec3::new(0, 500, 0),
        GridRotation::default(),
    );
    let wheel = gear(
        &mut graph,
        spur(4, 36),
        IVec3::new(120, 500, 0),
        GridRotation::default(),
    );
    let top = |x: f32| Vec3::new(x, 1.125, 0.0);
    let pinion_hinge = hinge(
        &mut graph,
        FaceRef::part(plate, FaceKind::PositiveY),
        FaceRef::part(pinion, FaceKind::NegativeY),
        top(0.0),
        Vec3::Y,
    );
    let wheel_hinge = hinge(
        &mut graph,
        FaceRef::part(plate, FaceKind::PositiveY),
        FaceRef::part(wheel, FaceKind::NegativeY),
        top(0.3),
        Vec3::Y,
    );
    mesh(&mut graph, pinion, wheel);
    let creation = graph.compile_with_static_parts(vec![plate]).unwrap();
    assert_eq!(creation.gear_links.len(), 1);
    let coordinates = [pinion_hinge, wheel_hinge].map(|id| coordinate(&creation, id));
    (creation, coordinates)
}

#[test]
fn meshing_gears_turn_at_their_tooth_ratio_and_stay_phased() {
    let (creation, [pinion, wheel]) = gear_train();
    let motor = spin(&creation, pinion, 50.0, 6.0);
    let mut world = World::new(creation, MachineState::at_rest(&gear_train().0));
    drive(&mut world, &[(pinion, motor)], 120);
    let (pinion_speed, wheel_speed) = (speed(&world, pinion), speed(&world, wheel));
    assert!(pinion_speed > 5.0, "{pinion_speed}");
    // External gears on parallel axes turn opposite ways at the inverse of
    // their tooth counts.
    assert!(
        (wheel_speed / pinion_speed + 2.0 / 3.0).abs() < 0.01,
        "{pinion_speed} against {wheel_speed}"
    );
    let state = &world.machine.snapshot().state;
    assert!(
        (state.coordinates[wheel] / state.coordinates[pinion] + 2.0 / 3.0).abs() < 0.01,
        "{state:?}"
    );
    let diagnostics = world.machine.diagnostics();
    assert_eq!(diagnostics.meshes, 1);
    assert!(diagnostics.mesh_slip < 0.002, "{}", diagnostics.mesh_slip);
}

#[test]
fn a_geared_machine_repeats_exactly_from_identical_inputs() {
    let run = || {
        let (creation, [pinion, _]) = gear_train();
        let motor = spin(&creation, pinion, 50.0, 6.0);
        let state = MachineState::at_rest(&creation);
        let mut world = World::new(creation, state);
        drive(&mut world, &[(pinion, motor)], 60);
        world.machine.snapshot().state_hash()
    };
    assert_eq!(run(), run());
}

#[test]
fn a_compound_train_multiplies_its_ratios() {
    let mut graph = ConstructionGraph::new();
    let plate = spawn(&mut graph, IVec3::new(0, 400, 0), [8, 1, 4]);
    let pinion = gear(
        &mut graph,
        spur(4, 24),
        IVec3::new(0, 500, 0),
        GridRotation::default(),
    );
    let wheel = gear(
        &mut graph,
        spur(4, 36),
        IVec3::new(120, 500, 0),
        GridRotation::default(),
    );
    // A 12-tooth pinion on the wheel's shaft, two blocks up, meshing a second
    // 36-tooth wheel on a post.
    let second_pinion = gear(
        &mut graph,
        spur(4, 12),
        IVec3::new(120, 700, 0),
        GridRotation::default(),
    );
    let second_wheel = gear(
        &mut graph,
        spur(4, 36),
        IVec3::new(216, 700, 0),
        GridRotation::default(),
    );
    let post = spawn(&mut graph, IVec3::new(216, 550, 0), [1, 2, 1]);
    weld(&mut graph, wheel, second_pinion);
    weld(&mut graph, post, plate);
    let pinion_hinge = hinge(
        &mut graph,
        FaceRef::part(plate, FaceKind::PositiveY),
        FaceRef::part(pinion, FaceKind::NegativeY),
        Vec3::new(0.0, 1.125, 0.0),
        Vec3::Y,
    );
    hinge(
        &mut graph,
        FaceRef::part(plate, FaceKind::PositiveY),
        FaceRef::part(wheel, FaceKind::NegativeY),
        Vec3::new(0.3, 1.125, 0.0),
        Vec3::Y,
    );
    let output_hinge = hinge(
        &mut graph,
        FaceRef::part(post, FaceKind::PositiveY),
        FaceRef::part(second_wheel, FaceKind::NegativeY),
        Vec3::new(0.54, 1.625, 0.0),
        Vec3::Y,
    );
    mesh(&mut graph, pinion, wheel);
    mesh(&mut graph, second_pinion, second_wheel);
    let creation = graph.compile_with_static_parts(vec![plate]).unwrap();
    assert_eq!(creation.gear_links.len(), 2);
    let input = coordinate(&creation, pinion_hinge);
    let output = coordinate(&creation, output_hinge);
    let motor = spin(&creation, input, 50.0, 6.0);
    let state = MachineState::at_rest(&creation);
    let mut world = World::new(creation, state);
    drive(&mut world, &[(input, motor)], 120);
    let (input_speed, output_speed) = (speed(&world, input), speed(&world, output));
    assert!(input_speed > 5.0, "{input_speed}");
    // 24:36 then 12:36, each stage reversing the direction.
    assert!(
        (output_speed / input_speed - 2.0 / 9.0).abs() < 0.02 * 2.0 / 9.0,
        "{input_speed} against {output_speed}"
    );
}

#[test]
fn a_planetary_set_turns_its_carrier_at_the_textbook_ratio() {
    let mut graph = ConstructionGraph::new();
    let plate = spawn(&mut graph, IVec3::new(0, 400, 0), [8, 1, 8]);
    let carrier = spawn(&mut graph, IVec3::new(0, 500, 0), [4, 1, 4]);
    let sun = gear(
        &mut graph,
        spur(4, 24),
        IVec3::new(0, 600, 0),
        GridRotation::default(),
    );
    let ring = gear(
        &mut graph,
        GearSpec::new(4, 72, GearKind::Internal).unwrap(),
        IVec3::new(0, 600, 0),
        GridRotation::default(),
    );
    weld(&mut graph, ring, plate);
    let carrier_hinge = hinge(
        &mut graph,
        FaceRef::part(plate, FaceKind::PositiveY),
        FaceRef::part(carrier, FaceKind::NegativeY),
        Vec3::new(0.0, 1.125, 0.0),
        Vec3::Y,
    );
    let sun_hinge = hinge(
        &mut graph,
        FaceRef::part(carrier, FaceKind::PositiveY),
        FaceRef::part(sun, FaceKind::NegativeY),
        Vec3::new(0.0, 1.375, 0.0),
        Vec3::Y,
    );
    // Three planets 24 cm out, 120 degrees apart, within a tick of the grid.
    for [x, z] in [[96_i16, 0], [-48, 83], [-48, -83]] {
        let planet = gear(
            &mut graph,
            spur(4, 24),
            IVec3::new(x.into(), 600, z.into()),
            GridRotation::default(),
        );
        hinge(
            &mut graph,
            FaceRef::part(carrier, FaceKind::PositiveY),
            FaceRef::part(planet, FaceKind::NegativeY),
            Vec3::new(f32::from(x) * 0.0025, 1.375, f32::from(z) * 0.0025),
            Vec3::Y,
        );
        mesh(&mut graph, sun, planet);
        mesh(&mut graph, ring, planet);
    }
    let creation = graph.compile_with_static_parts(vec![plate]).unwrap();
    assert_eq!(creation.gear_links.len(), 6);
    let carrier = coordinate(&creation, carrier_hinge);
    let sun = coordinate(&creation, sun_hinge);
    // The sun turns on the carrier, so its drive sets the speed between them.
    let motor = spin(&creation, sun, 300.0, 8.0);
    let state = MachineState::at_rest(&creation);
    let mut world = World::new(creation, state);
    drive(&mut world, &[(sun, motor)], 180);
    let (relative, carrier_speed) = (speed(&world, sun), speed(&world, carrier));
    assert!(relative > 7.0, "{relative}");
    // With the ring fixed the carrier turns at N_s / (N_s + N_r) = 1/4 of the
    // sun, so the sun leads the carrier by three carrier turns.
    let expected = relative / 3.0;
    assert!(
        (carrier_speed - expected).abs() < 0.02 * expected,
        "{carrier_speed} against {expected}"
    );
    assert!(world.machine.diagnostics().mesh_slip < 0.003);
}

#[test]
fn a_rack_advances_at_its_pinions_pitch_speed() {
    let mut graph = ConstructionGraph::new();
    let post = spawn(&mut graph, IVec3::new(0, 494, -100), [1, 1, 1]);
    let pinion = gear(&mut graph, spur(4, 24), IVec3::new(0, 494, 0), Y_TO_Z);
    let bar = spawned(
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [8, 1, 1],
                    BuildPose::from_position_ticks(IVec3::new(0, 400, 0), GridRotation::default()),
                )
                .unwrap()
                .with_rack(RackSpec::new(4, FaceKind::PositiveY, mechanic_core::Axis::X).unwrap())
                .unwrap(),
            ))
            .unwrap(),
    );
    let pinion_hinge = hinge(
        &mut graph,
        FaceRef::part(post, FaceKind::PositiveZ),
        FaceRef::part(pinion, FaceKind::NegativeY),
        Vec3::new(0.0, 1.235, -0.125),
        Vec3::Z,
    );
    mesh(&mut graph, pinion, bar);
    let creation = graph.compile_with_static_parts(vec![post]).unwrap();
    let pinion = coordinate(&creation, pinion_hinge);
    let bar_body = creation
        .part_to_compound
        .iter()
        .find(|(part, _)| *part == bar)
        .unwrap()
        .1 as usize;
    let rows = creation.dynamics.body_velocities[bar_body].clone();
    let motor = spin(&creation, pinion, 500.0, 5.0);
    let state = MachineState::at_rest(&creation);
    let mut world = World::new(creation, state);
    drive(&mut world, &[(pinion, motor)], 120);
    let pinion_speed = speed(&world, pinion);
    assert!(pinion_speed > 4.0, "{pinion_speed}");
    // The free bar also pitches about the tooth force, so measure its
    // material point under the pinion, on the pitch plane 1 cm below the face
    // wherever the bar has turned it to.
    let state = &world.machine.snapshot().state;
    let velocity = &state.velocities[rows];
    let linear = DVec3::new(velocity[0], velocity[1], velocity[2]);
    let angular = DVec3::new(velocity[3], velocity[4], velocity[5]);
    let bar_pose = state.poses[bar_body];
    let normal = bar_pose.rotation * DVec3::Y;
    let pitch_center = bar_pose.position + normal * 0.115;
    let pinion_center = DVec3::new(0.0, 1.235, 0.0);
    let down = -(normal - DVec3::Z * normal.z).normalize();
    let pitch_point = pinion_center + down * 0.12;
    let bar_point = pitch_point - normal * (pitch_point - pitch_center).dot(normal);
    let tangent = DVec3::Z.cross(down);
    let along = (linear + angular.cross(bar_point - bar_pose.position))
        .dot(tangent)
        .abs();
    assert!(
        (along - 0.12 * pinion_speed).abs() < 0.02 * 0.12 * pinion_speed,
        "{velocity:?} against {pinion_speed}"
    );
}

#[test]
fn a_rack_of_many_blocks_runs_under_its_pinion_past_where_it_was_built() {
    let mut graph = ConstructionGraph::new();
    let post = spawn(&mut graph, IVec3::new(0, 494, -100), [1, 1, 1]);
    let pinion = gear(&mut graph, spur(4, 24), IVec3::new(0, 494, 0), Y_TO_Z);
    // Eight racked blocks welded into one 2 m bar, the pinion over the joint
    // between the fourth and fifth: the others were built clear of it.
    let blocks = (0..8)
        .map(|index| {
            spawned(
                graph
                    .apply(BuildCommand::Spawn(
                        CuboidSpec::new(
                            [1, 1, 1],
                            BuildPose::from_position_ticks(
                                IVec3::new(-350 + 100 * index, 400, 0),
                                GridRotation::default(),
                            ),
                        )
                        .unwrap()
                        .with_rack(
                            RackSpec::new(4, FaceKind::PositiveY, mechanic_core::Axis::X).unwrap(),
                        )
                        .unwrap(),
                    ))
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    for pair in blocks.windows(2) {
        weld(&mut graph, pair[0], pair[1]);
    }
    let pinion_hinge = hinge(
        &mut graph,
        FaceRef::part(post, FaceKind::PositiveZ),
        FaceRef::part(pinion, FaceKind::NegativeY),
        Vec3::new(0.0, 1.235, -0.125),
        Vec3::Z,
    );
    mesh(&mut graph, pinion, blocks[3]);
    let creation = graph.compile_with_static_parts(vec![post]).unwrap();
    let pinion = coordinate(&creation, pinion_hinge);
    let bar_body = creation
        .part_to_compound
        .iter()
        .find(|(part, _)| *part == blocks[0])
        .unwrap()
        .1 as usize;
    let motor = spin(&creation, pinion, 500.0, 5.0);
    let state = MachineState::at_rest(&creation);
    let start = state.poses[bar_body].position;
    let mut world = World::new(creation, state);
    drive(&mut world, &[(pinion, motor)], 120);
    let travelled = (world.machine.snapshot().state.poses[bar_body].position - start).length();
    assert!(travelled > 0.5, "the bar went {travelled} m");
}

#[test]
fn a_worm_turns_its_wheel_one_lead_per_turn() {
    let mut graph = ConstructionGraph::new();
    let wheel_post = spawn(&mut graph, IVec3::new(0, 500, 0), [1, 1, 1]);
    let worm_post = spawn(&mut graph, IVec3::new(90, 600, -150), [1, 1, 1]);
    weld(&mut graph, wheel_post, worm_post);
    let wheel = gear(
        &mut graph,
        spur(6, 24),
        IVec3::new(0, 600, 0),
        GridRotation::default(),
    );
    let worm = spawned(
        graph
            .apply(BuildCommand::SpawnCylinder(
                CylinderSpec::new(
                    CylinderDimensions::new(0.1, 0.0, 0.5).unwrap(),
                    BuildPose::from_position_ticks(IVec3::new(90, 600, 0), Y_TO_Z),
                )
                .with_spiral(
                    mechanic_core::SpiralSpec::new(
                        20,
                        1,
                        mechanic_core::SpiralHand::Right,
                        mechanic_core::SpiralProfile::square(10, 5).unwrap(),
                        mechanic_core::SpiralProfile::PLAIN,
                        None,
                    )
                    .unwrap(),
                )
                .unwrap(),
            ))
            .unwrap(),
    );
    let wheel_hinge = hinge(
        &mut graph,
        FaceRef::part(wheel_post, FaceKind::PositiveY),
        FaceRef::part(wheel, FaceKind::NegativeY),
        Vec3::new(0.0, 1.375, 0.0),
        Vec3::Y,
    );
    let worm_hinge = hinge(
        &mut graph,
        FaceRef::part(worm_post, FaceKind::PositiveZ),
        FaceRef::part(worm, FaceKind::NegativeY),
        Vec3::new(0.225, 1.5, -0.25),
        Vec3::Z,
    );
    mesh(&mut graph, wheel, worm);
    let creation = graph.compile_with_static_parts(vec![wheel_post]).unwrap();
    assert_eq!(creation.gear_links.len(), 1);
    let wheel = coordinate(&creation, wheel_hinge);
    let worm = coordinate(&creation, worm_hinge);
    let motor = spin(&creation, worm, 20.0, 20.0);
    let state = MachineState::at_rest(&creation);
    let mut world = World::new(creation, state);
    drive(&mut world, &[(worm, motor)], 120);
    let (worm_speed, wheel_speed) = (speed(&world, worm), speed(&world, wheel));
    assert!(worm_speed > 18.0, "{worm_speed}");
    // One worm turn moves the wheel's 18 cm pitch circle one 5 cm lead.
    let expected = worm_speed * 0.05 / (core::f64::consts::TAU * 0.18);
    assert!(
        (wheel_speed.abs() - expected).abs() < 0.02 * expected,
        "{wheel_speed} against {expected}"
    );
}

/// Two 24-tooth mitre gears on the housing's X axis, two on the carrier's Z
/// axis, the carrier on a hinge to the housing: the housing's static post, and
/// the coordinates of the carrier, side A and side B.
fn differential() -> (CompiledCreation, [usize; 3]) {
    let mut graph = ConstructionGraph::new();
    let bevel = GearSpec::new(
        4,
        24,
        GearKind::Bevel {
            cone_angle_degrees: 45,
            large_end: SpiralEnd::PositiveY,
        },
    )
    .unwrap();
    let post_a = spawn(&mut graph, IVec3::new(-198, 600, 0), [1, 1, 1]);
    let post_b = spawn(&mut graph, IVec3::new(198, 600, 0), [1, 1, 1]);
    let post_c = spawn(&mut graph, IVec3::new(398, 600, 0), [1, 1, 1]);
    weld(&mut graph, post_a, post_b);
    weld(&mut graph, post_a, post_c);
    let side_a = gear(&mut graph, bevel, IVec3::new(-98, 600, 0), Y_TO_X);
    let side_b = gear(&mut graph, bevel, IVec3::new(98, 600, 0), Y_TO_NEG_X);
    let spider_a = gear(&mut graph, bevel, IVec3::new(0, 600, -98), Y_TO_Z);
    let spider_b = gear(&mut graph, bevel, IVec3::new(0, 600, 98), Y_TO_NEG_Z);
    let arm_a = spawn(&mut graph, IVec3::new(0, 600, -198), [1, 1, 1]);
    let arm_b = spawn(&mut graph, IVec3::new(0, 600, 198), [1, 1, 1]);
    let hub = spawn(&mut graph, IVec3::new(298, 600, 0), [1, 1, 1]);
    weld(&mut graph, hub, arm_a);
    weld(&mut graph, hub, arm_b);
    let hinge_a = hinge(
        &mut graph,
        FaceRef::part(post_a, FaceKind::PositiveX),
        FaceRef::part(side_a, FaceKind::NegativeY),
        Vec3::new(-0.37, 1.5, 0.0),
        Vec3::X,
    );
    let hinge_b = hinge(
        &mut graph,
        FaceRef::part(post_b, FaceKind::NegativeX),
        FaceRef::part(side_b, FaceKind::NegativeY),
        Vec3::new(0.37, 1.5, 0.0),
        Vec3::NEG_X,
    );
    hinge(
        &mut graph,
        FaceRef::part(arm_a, FaceKind::PositiveZ),
        FaceRef::part(spider_a, FaceKind::NegativeY),
        Vec3::new(0.0, 1.5, -0.37),
        Vec3::Z,
    );
    hinge(
        &mut graph,
        FaceRef::part(arm_b, FaceKind::NegativeZ),
        FaceRef::part(spider_b, FaceKind::NegativeY),
        Vec3::new(0.0, 1.5, 0.37),
        Vec3::NEG_Z,
    );
    let carrier_hinge = hinge(
        &mut graph,
        FaceRef::part(post_c, FaceKind::NegativeX),
        FaceRef::part(hub, FaceKind::PositiveX),
        Vec3::new(0.87, 1.5, 0.0),
        Vec3::NEG_X,
    );
    for side in [side_a, side_b] {
        for spider in [spider_a, spider_b] {
            mesh(&mut graph, side, spider);
        }
    }
    let creation = graph.compile_with_static_parts(vec![post_a]).unwrap();
    assert_eq!(creation.gear_links.len(), 4);
    let coordinates = [carrier_hinge, hinge_a, hinge_b].map(|id| coordinate(&creation, id));
    (creation, coordinates)
}

#[test]
fn a_differential_splits_one_input_between_two_outputs() {
    let (creation, [carrier, side_a, side_b]) = differential();
    let motor = spin(&creation, carrier, 1000.0, 5.0);
    let brake = spin(&creation, side_a, 2000.0, 0.0);
    let state = MachineState::at_rest(&creation);
    let mut world = World::new(creation, state);
    // Every coordinate here turns about the world X axis, and the carrier's
    // and side B's hinges are measured about −X.
    let about_x = |world: &World| {
        [
            -speed(world, carrier),
            speed(world, side_a),
            -speed(world, side_b),
        ]
    };
    drive(&mut world, &[(carrier, motor)], 120);
    let [carrier_speed, a, b] = about_x(&world);
    assert!(carrier_speed.abs() > 4.5, "{carrier_speed}");
    // Unloaded, the two outputs average the carrier.
    assert!(
        (a + b - 2.0 * carrier_speed).abs() < 0.02 * carrier_speed.abs(),
        "{a} + {b} against {carrier_speed}"
    );
    drive(&mut world, &[(side_a, brake)], 180);
    let [carrier_speed, a, b] = about_x(&world);
    assert!(a.abs() < 0.05 * carrier_speed.abs(), "{a}");
    assert!(
        (b - 2.0 * carrier_speed).abs() < 0.02 * carrier_speed.abs(),
        "{b} against {carrier_speed}"
    );
}
