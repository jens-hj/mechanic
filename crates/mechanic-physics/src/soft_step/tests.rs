//! Behaviour of the soft-step solver, with tolerances suited to a game.

use super::*;
use crate::{
    DriveCommand,
    terrain_contacts::tests::{
        collision_shapes, collision_solids, cube, loose_cubes, pose, terrain,
    },
};
use bevy_math::{DQuat, IVec3, Vec3};
use mechanic_core::{
    BearingKind, BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec,
    DriveMode, FaceKind, FaceRef, GridRotation, PartId, ShockBodyEnd, ShockSpec, SpringSpec,
    SuspensionSpec,
};
use mechanic_world::TerrainMaterial;

const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);
const GENERATION: u64 = 7;

// The 128 m rock floor, optionally with a 128 m wall at `wall` metres facing −X.
fn ground(wall: Option<f32>) -> std::sync::Arc<mechanic_world::TerrainCollisionChunk> {
    let mut chunk = terrain([TerrainMaterial::Rock; 2]);
    let expanded = std::sync::Arc::make_mut(&mut chunk);
    for vertex in &mut expanded.vertices {
        vertex[0] *= 64.0;
        vertex[2] *= 64.0;
    }
    expanded.bounds.minimum.0 *= 64.0;
    expanded.bounds.maximum.0 *= 64.0;
    if let Some(x) = wall {
        // Mapping (x, y, z) to (wall, x, z) keeps the winding, so the floor's
        // upward normal becomes −X.
        let offset = u32::try_from(expanded.vertices.len()).unwrap();
        let floor = expanded.vertices.clone();
        expanded
            .vertices
            .extend(floor.iter().map(|vertex| [x, vertex[0], vertex[2]]));
        let weights = expanded.material_weights.clone();
        expanded.material_weights.extend(weights);
        let indices = expanded.indices.clone();
        expanded
            .indices
            .extend(indices.iter().map(|index| index + offset));
        let triangles = expanded.triangle_bvh.triangles.clone();
        expanded
            .triangle_bvh
            .triangles
            .extend(triangles.into_iter().map(|mut triangle| {
                triangle.indices = triangle.indices.map(|index| index + offset);
                triangle
            }));
        expanded.bounds.minimum.0.y = -64.0;
        expanded.bounds.maximum.0.y = 64.0;
    }
    expanded.triangle_bvh.bounds = expanded.bounds;
    expanded.triangle_bvh.nodes[0].bounds = expanded.bounds;
    expanded.triangle_bvh.nodes[0].triangle_count =
        expanded.triangle_bvh.triangles.len().try_into().unwrap();
    chunk
}

// Rock floor with a kerb along Z: its face at `face` metres faces −X, and its
// top is `height` metres up.
fn kerb(face: f32, height: f32) -> std::sync::Arc<mechanic_world::TerrainCollisionChunk> {
    use mechanic_world::{
        TerrainCollisionChunk, TerrainNodeId, TerrainTriangleGroupMask, TriangleBvh,
        TriangleBvhNode, TriangleBvhTriangle, WorldBounds, WorldPosition,
    };
    // Rectangles as [corner, across, diagonal, along]; each splits into two
    // triangles wound toward the kerb's open side.
    let rectangles = [
        [
            [-64.0, 0.0, -64.0],
            [-64.0, 0.0, 64.0],
            [face, 0.0, 64.0],
            [face, 0.0, -64.0],
        ],
        [
            [face, 0.0, -64.0],
            [face, 0.0, 64.0],
            [face, height, 64.0],
            [face, height, -64.0],
        ],
        [
            [face, height, -64.0],
            [face, height, 64.0],
            [64.0, height, 64.0],
            [64.0, height, -64.0],
        ],
    ];
    let vertices = rectangles.concat();
    let triangles = (0..3_u32)
        .flat_map(|rectangle| {
            let base = rectangle * 4;
            [[base, base + 1, base + 2], [base, base + 2, base + 3]]
        })
        .map(|indices| TriangleBvhTriangle {
            indices,
            group_mask: TerrainTriangleGroupMask::REGULAR,
        })
        .collect::<Vec<_>>();
    let bounds = WorldBounds {
        minimum: WorldPosition(DVec3::new(-64.0, 0.0, -64.0)),
        maximum: WorldPosition(DVec3::new(64.0, f64::from(height), 64.0)),
    };
    let mut weights = [0.0; TerrainMaterial::COUNT];
    weights[usize::from(TerrainMaterial::Rock.code())] = 1.0;
    std::sync::Arc::new(TerrainCollisionChunk {
        node: TerrainNodeId::ROOT,
        material_weights: vec![weights; vertices.len()],
        indices: triangles.iter().flat_map(|t| t.indices).collect(),
        vertices,
        bounds,
        generation: 1,
        triangle_bvh: TriangleBvh {
            bounds,
            nodes: vec![TriangleBvhNode {
                bounds,
                triangle_count: 6,
                group_mask: TerrainTriangleGroupMask::REGULAR,
                ..Default::default()
            }],
            triangles,
        },
        active_groups: TerrainTriangleGroupMask::REGULAR,
        ..Default::default()
    })
}

// Corners of the box around every collider.
fn extent(creation: &CompiledCreation, state: &MachineState) -> [DVec3; 2] {
    collision_shapes(creation).into_iter().fold(
        [DVec3::INFINITY, DVec3::NEG_INFINITY],
        |[low, high], (body, shape)| {
            let pose = state.poses[body];
            let [minimum, maximum] = shape
                .transformed(pose.position, pose.rotation)
                .unwrap()
                .bounds();
            [low.min(minimum), high.max(maximum)]
        },
    )
}

// One cuboid of the given block dimensions.
fn block(dimensions: [u8; 3]) -> CompiledCreation {
    let mut graph = ConstructionGraph::new();
    spawn(&mut graph, IVec3::ZERO, dimensions);
    graph.compile().unwrap()
}

// A creation at rest with its lowest point `height` above the floor.
fn lifted(creation: &CompiledCreation, height: f64) -> MachineState {
    let mut state = MachineState::at_rest(creation);
    let lowest = clearance(creation, &state);
    for pose in &mut state.poses {
        pose.position.y += height - lowest;
    }
    state
}

fn launch(
    creation: &CompiledCreation,
    state: &mut MachineState,
    body: usize,
    linear: DVec3,
    angular: DVec3,
) {
    let rows = creation.dynamics.body_velocities[body].clone();
    state.velocities[rows].copy_from_slice(&[
        linear.x, linear.y, linear.z, angular.x, angular.y, angular.z,
    ]);
}

#[test]
fn a_very_fast_cube_never_ends_below_the_floor() {
    for speed in [30.0, 60.0, 120.0, 250.0] {
        let creation = block([1, 1, 1]);
        let mut state = lifted(&creation, 2.0);
        launch(&creation, &mut state, 0, DVec3::NEG_Y * speed, DVec3::ZERO);
        let mut world = World::new(creation, state);
        for tick in 1..=60 {
            let state = world.tick(GRAVITY);
            let clearance = clearance(world.creation(), &state);
            assert!(
                clearance > -0.05,
                "{speed} m/s, tick {tick}: clearance {clearance}"
            );
        }
    }
}

#[test]
fn a_fast_cube_grazing_the_floor_stays_above_it() {
    let creation = block([1, 1, 1]);
    let mut state = lifted(&creation, 1.0);
    let angle = 20.0_f64.to_radians();
    let velocity = DVec3::new(angle.cos(), -angle.sin(), 0.0) * 120.0;
    launch(&creation, &mut state, 0, velocity, DVec3::ZERO);
    let mut world = World::new(creation, state);
    for tick in 1..=25 {
        let state = world.tick(GRAVITY);
        let clearance = clearance(world.creation(), &state);
        assert!(clearance > -0.05, "tick {tick}: clearance {clearance}");
    }
}

#[test]
fn a_fast_cube_does_not_pass_through_a_steep_wall() {
    let creation = block([1, 1, 1]);
    let mut state = lifted(&creation, 0.5);
    launch(&creation, &mut state, 0, DVec3::X * 100.0, DVec3::ZERO);
    let mut world = World::with_wall(creation, state, 5.0);
    for tick in 1..=60 {
        let state = world.tick(GRAVITY);
        let front = extent(world.creation(), &state)[1].x;
        assert!(front < 5.05, "tick {tick}: front {front}");
    }
}

#[test]
fn a_fast_spinning_bar_does_not_pass_through_the_floor() {
    let creation = block([8, 1, 1]);
    let mut state = lifted(&creation, 0.5);
    launch(
        &creation,
        &mut state,
        0,
        DVec3::NEG_Y * 5.0,
        DVec3::Z * 60.0,
    );
    let mut world = World::new(creation, state);
    for tick in 1..=60 {
        let state = world.tick(GRAVITY);
        let clearance = clearance(world.creation(), &state);
        assert!(clearance > -0.05, "tick {tick}: clearance {clearance}");
    }
}

#[test]
fn a_fast_projectile_neither_passes_through_nor_pushes_its_target_through_the_floor() {
    let creation = loose_cubes();
    let mut state = MachineState {
        poses: vec![pose(DVec3::Y * 0.501), pose(DVec3::Y * 4.0)],
        ..MachineState::at_rest(&creation)
    };
    launch(&creation, &mut state, 1, DVec3::NEG_Y * 80.0, DVec3::ZERO);
    let mut world = World::new(creation, state);
    for tick in 1..=60 {
        let state = world.tick(GRAVITY);
        let clearance = clearance(world.creation(), &state);
        let gap = state.poses[1].position.y - state.poses[0].position.y;
        assert!(
            clearance > -0.05 && gap > 0.95,
            "tick {tick}: clearance {clearance}, gap {gap}"
        );
    }
}

#[test]
fn a_car_crashing_into_a_wall_stays_in_front_of_it() {
    let (creation, mut state) = saved_car();
    let front = extent(&creation, &state)[1].x;
    for body in 0..creation.compounds.len() {
        if creation.loop_topology.body_parents[body].is_root {
            launch(&creation, &mut state, body, DVec3::X * 40.0, DVec3::ZERO);
        }
    }
    #[allow(clippy::cast_possible_truncation)] // A wall a few metres from the origin.
    let wall = (front + 6.0) as f32;
    let mut world = World::with_wall(creation, state, wall);
    for tick in 1..=90 {
        let state = world.tick(GRAVITY);
        let front = extent(world.creation(), &state)[1].x;
        assert!(
            front < f64::from(wall) + 0.05,
            "tick {tick}: front {front}, wall {wall}"
        );
    }
}

/// A machine on a 128 m rock floor, optionally facing a wall.
struct World {
    scene: TerrainContactScene,
    geometry: MachineCollisionGeometry,
    machine: CpuMachine,
    settings: SoftStepSettings,
}

impl World {
    fn new(creation: CompiledCreation, state: MachineState) -> Self {
        Self::on(creation, state, None)
    }

    fn with_wall(creation: CompiledCreation, state: MachineState, wall: f32) -> Self {
        Self::on(creation, state, Some(wall))
    }

    fn on(creation: CompiledCreation, state: MachineState, wall: Option<f32>) -> Self {
        Self::over(creation, state, ground(wall))
    }

    fn over(
        creation: CompiledCreation,
        state: MachineState,
        terrain: std::sync::Arc<mechanic_world::TerrainCollisionChunk>,
    ) -> Self {
        let mut scene = TerrainContactScene::default();
        scene.publish(1, &[terrain], &[]).unwrap();
        Self {
            scene,
            geometry: MachineCollisionGeometry::new(&creation, GENERATION).unwrap(),
            machine: CpuMachine::new(creation, GENERATION, state).unwrap(),
            settings: SoftStepSettings::default(),
        }
    }

    fn creation(&self) -> &CompiledCreation {
        &self.machine.creation
    }

    fn next_tick(&self) -> u64 {
        self.machine.snapshot().tick + 1
    }

    fn tick_with(&mut self, gravity: DVec3, commands: &[DriveCommand]) -> MachineState {
        let terrain = SoftStepTerrain {
            scene: &self.scene,
            geometry: &self.geometry,
            topology_generation: GENERATION,
            origin: DVec3::ZERO,
        };
        let tick = self.next_tick();
        self.machine
            .step(gravity, &self.settings, &[], commands, Some(terrain))
            .unwrap();
        let diagnostics = self.machine.diagnostics();
        assert!(
            !diagnostics.degraded,
            "tick {tick} degraded: {:?}",
            diagnostics.degraded_reason
        );
        self.machine.snapshot().state.clone()
    }

    fn tick(&mut self, gravity: DVec3) -> MachineState {
        self.tick_with(gravity, &[])
    }

    fn command(&self, coordinate: usize, drive: CoordinateDrive) -> DriveCommand {
        DriveCommand {
            tick: self.next_tick(),
            topology_generation: GENERATION,
            coordinate,
            drive,
        }
    }
}

// Each body's lowest collider point, measured from the colliders themselves.
// Lowest point of each body, taking a cylinder as the circle it rolls on.
fn lowest_points(creation: &CompiledCreation, state: &MachineState) -> Vec<f64> {
    let mut lowest = vec![f64::INFINITY; state.poses.len()];
    for (body, shape, round) in collision_solids(creation) {
        let pose = state.poses[body];
        let bottom = match round {
            Some(cylinder) => {
                cylinder
                    .transformed(pose.position, pose.rotation)
                    .unwrap()
                    .support(DVec3::NEG_Y)
                    .y
            }
            None => {
                shape
                    .transformed(pose.position, pose.rotation)
                    .unwrap()
                    .bounds()[0]
                    .y
            }
        };
        lowest[body] = lowest[body].min(bottom);
    }
    lowest
}

// Lowest collider point above the floor; negative values are penetration.
fn clearance(creation: &CompiledCreation, state: &MachineState) -> f64 {
    lowest_points(creation, state)
        .into_iter()
        .fold(f64::INFINITY, f64::min)
}

fn fastest(state: &MachineState) -> f64 {
    state
        .velocities
        .iter()
        .fold(0.0_f64, |fastest, value| fastest.max(value.abs()))
}

// The cube's half extent is 0.5 m, so `height` is its initial clearance.
fn box_above_floor(height: f64, vertical: f64) -> World {
    let (creation, _, _) = cube();
    let mut state = MachineState::at_rest(&creation);
    state.poses[0].position.y = 0.5 + height;
    state.velocities[1] = vertical;
    World::new(creation, state)
}

fn spawn(graph: &mut ConstructionGraph, ticks: IVec3, dimensions: [u8; 3]) -> PartId {
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                dimensions,
                BuildPose::from_position_ticks(ticks, GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        panic!("spawn expected");
    };
    part
}

// A 1 m block with a small block sprung above it.
fn suspension(spec: SuspensionSpec, anchored: bool) -> CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let root = spawn(&mut graph, IVec3::ZERO, [4, 4, 4]);
    #[allow(clippy::cast_possible_truncation)] // Bounded fixture spacing on the lattice.
    let spacing = ((spec.initial_length() + 0.625) / 0.0025).round() as i32;
    let tip = spawn(&mut graph, IVec3::Y * spacing, [1, 1, 1]);
    graph
        .apply(BuildCommand::AddBearing(
            BearingSpec::new(
                FaceRef::part(root, FaceKind::PositiveY),
                FaceRef::part(tip, FaceKind::NegativeY),
                Vec3::Y * 0.5,
                Vec3::Y,
            )
            .with_kind(BearingKind::Suspension(spec)),
        ))
        .unwrap();
    graph
        .compile_with_static_parts(if anchored { vec![root] } else { vec![] })
        .unwrap()
}

// A planar parallelogram 2 m above the floor: a 2 m ground bar, two 1.5 m cranks
// on Z bearings and a coupler whose second bearing closes the loop.
fn four_bar(anchored: bool) -> CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let height = IVec3::Y * 800;
    let ground = spawn(&mut graph, height, [8, 1, 1]);
    let left = spawn(&mut graph, height + IVec3::new(-350, 250, 100), [1, 6, 1]);
    let right = spawn(&mut graph, height + IVec3::new(350, 250, 100), [1, 6, 1]);
    let coupler = spawn(&mut graph, height + IVec3::new(0, 500, 200), [8, 1, 1]);
    let hinge = |source, target, anchor: Vec3| {
        BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(source, FaceKind::PositiveZ),
            FaceRef::part(target, FaceKind::NegativeZ),
            anchor + Vec3::Y * 2.0,
            Vec3::Z,
        ))
    };
    graph
        .apply_batch([
            hinge(ground, left, Vec3::new(-0.875, 0.0, 0.125)),
            hinge(ground, right, Vec3::new(0.875, 0.0, 0.125)),
            hinge(left, coupler, Vec3::new(-0.875, 1.25, 0.375)),
            hinge(right, coupler, Vec3::new(0.875, 1.25, 0.375)),
        ])
        .unwrap();
    let creation = graph
        .compile_with_static_parts(if anchored { vec![ground] } else { vec![] })
        .unwrap();
    assert_eq!(creation.dynamics.loops.len(), 1);
    creation
}

// A 1 m block carrying a 1 m plate on two parallel suspensions, one of which
// closes a loop.
fn twin_suspension(spec: SuspensionSpec) -> CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let root = spawn(&mut graph, IVec3::ZERO, [4, 4, 4]);
    #[allow(clippy::cast_possible_truncation)] // Bounded fixture spacing on the lattice.
    let spacing = ((spec.initial_length() + 0.625) / 0.0025).round() as i32;
    let plate = spawn(&mut graph, IVec3::Y * spacing, [4, 1, 4]);
    let strut = |x: f32| {
        BuildCommand::AddBearing(
            BearingSpec::new(
                FaceRef::part(root, FaceKind::PositiveY),
                FaceRef::part(plate, FaceKind::NegativeY),
                Vec3::new(x, 0.5, 0.0),
                Vec3::Y,
            )
            .with_kind(BearingKind::Suspension(spec)),
        )
    };
    graph.apply_batch([strut(-0.25), strut(0.25)]).unwrap();
    let creation = graph.compile_with_static_parts(vec![root]).unwrap();
    assert_eq!(creation.dynamics.loops.len(), 1);
    creation
}

// Runs `ticks` and returns the widest closure gap and misalignment seen.
fn worst_closure(world: &mut World, ticks: usize, mut each: impl FnMut(&World)) -> (f64, f64) {
    let mut worst = (0.0_f64, 0.0_f64);
    for _ in 0..ticks {
        world.tick(GRAVITY);
        let diagnostics = world.machine.diagnostics();
        worst = (
            worst.0.max(diagnostics.closure_position_error),
            worst.1.max(diagnostics.closure_angle_error),
        );
        each(world);
    }
    worst
}

#[test]
fn a_four_bar_swings_under_gravity_and_stays_closed() {
    let creation = four_bar(true);
    let mut state = MachineState::at_rest(&creation);
    state.velocities[creation.dynamics.coordinate_velocities[0]] = 0.5;
    let mut world = World::new(creation, state);
    let start = world.machine.snapshot().state.poses.clone();

    // A swing through the bottom can return near its start, so track the peak.
    let mut moved = 0.0_f64;
    let worst = worst_closure(&mut world, 180, |world| {
        for (now, then) in world.machine.snapshot().state.poses.iter().zip(&start) {
            moved = moved.max(now.position.distance(then.position));
        }
    });

    assert!(moved > 0.3, "the linkage barely moved: {moved} m");
    assert!(
        worst.0 < 0.005 && worst.1 < 1_f64.to_radians(),
        "the loop opened: {worst:?}"
    );
}

#[test]
fn a_free_four_bar_lands_on_the_floor_and_stays_closed() {
    let creation = four_bar(false);
    let state = lifted(&creation, 0.3);
    let mut world = World::new(creation, state);

    let worst = worst_closure(&mut world, 240, |world| {
        let state = &world.machine.snapshot().state;
        let clearance = clearance(world.creation(), state);
        assert!(clearance > -0.02, "sank {clearance} m into the floor");
    });

    assert!(
        worst.0 < 0.005 && worst.1 < 1_f64.to_radians(),
        "the loop opened: {worst:?}"
    );
}

#[test]
fn a_loop_seeded_open_closes_without_flying_apart() {
    let creation = four_bar(true);
    let mut state = MachineState::at_rest(&creation);
    state.coordinates[0] = 0.05;
    let mut world = World::new(creation, state);

    let mut fastest_seen = 0.0_f64;
    worst_closure(&mut world, 60, |world| {
        fastest_seen = fastest_seen.max(fastest(&world.machine.snapshot().state));
    });

    let settled = world.machine.diagnostics().closure_position_error;
    assert!(settled < 0.005, "still {settled} m open after a second");
    assert!(fastest_seen < 10.0, "closing flung it at {fastest_seen}");
}

#[test]
fn two_parallel_suspensions_share_their_load() {
    let spring = SpringSpec::new(0.5, 0.16, 0.12, 6, 0.0).unwrap();
    let shock = ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.0, 20.0, 20.0).unwrap();
    let spec = SuspensionSpec::new(Some(spring), Some(shock), None).unwrap();
    let creation = twin_suspension(spec);
    // Both springs carry the plate, so each compresses half as far as one would.
    let expected = f64::from(spec.passive_rows()[0][1])
        - f64::from(creation.compounds[1].mass_properties.mass) * 9.81
            / (2.0 * f64::from(spring.rate()));
    let state = MachineState::at_rest(&creation);
    let mut world = World::new(creation, state);

    let worst = worst_closure(&mut world, 360, |_| {});

    let state = &world.machine.snapshot().state;
    assert!(
        (state.coordinates[0] - expected).abs() < 0.005,
        "{} expected {expected}",
        state.coordinates[0]
    );
    assert!(fastest(state) < 0.05, "still moving at {}", fastest(state));
    assert!(
        worst.0 < 0.005 && worst.1 < 1_f64.to_radians(),
        "the struts parted: {worst:?}"
    );
}

#[test]
fn a_looped_machine_repeats_exactly_from_identical_inputs() {
    let run = || {
        let creation = four_bar(false);
        let state = lifted(&creation, 0.3);
        let mut world = World::new(creation, state);
        for _ in 0..90 {
            world.tick(GRAVITY);
        }
        world.machine.snapshot().state_hash()
    };
    assert_eq!(run(), run());
}

// Two 1 m blocks joined by a revolute bearing along X.
fn rotor(anchored: bool) -> CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let root = spawn(&mut graph, IVec3::ZERO, [4, 4, 4]);
    let tip = spawn(&mut graph, IVec3::X * 400, [4, 4, 4]);
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(root, FaceKind::PositiveX),
            FaceRef::part(tip, FaceKind::NegativeX),
            Vec3::X * 0.5,
            Vec3::X,
        )))
        .unwrap();
    graph
        .compile_with_static_parts(if anchored { vec![root] } else { vec![] })
        .unwrap()
}

fn motor(creation: &CompiledCreation, effort: f32, target: f32) -> CoordinateDrive {
    let acceleration = effort / creation.loop_topology.coordinate_axis_inertia[0];
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

// A saved world creation, lowered to rest 1 mm above the floor.
fn captured(instance: &str) -> (CompiledCreation, MachineState) {
    let instance: mechanic_world::WorldCreationInstanceDoc = ron::from_str(instance).unwrap();
    let loaded = instance.creation.into_graph().unwrap();
    let creation = loaded
        .graph
        .compile_with_suspension_sockets([], &loaded.sockets)
        .unwrap();
    let mut state = MachineState::at_rest(&creation);
    let lowest = clearance(&creation, &state);
    for pose in &mut state.poses {
        pose.position.y -= lowest - 0.001;
    }
    (creation, state)
}

fn saved_car() -> (CompiledCreation, MachineState) {
    captured(include_str!(
        "../../../mechanic-bench/tests/fixtures/driven_car_instance.ron"
    ))
}

#[test]
fn a_box_dropped_on_the_floor_comes_to_rest_on_it() {
    let mut world = box_above_floor(0.2, 0.0);
    for tick in 1..=120 {
        let state = world.tick(GRAVITY);
        let clearance = clearance(world.creation(), &state);
        assert!(clearance > -0.005, "tick {tick}: clearance {clearance}");
    }
    let state = &world.machine.snapshot().state;
    let clearance = clearance(world.creation(), state);
    assert!((-0.005..0.002).contains(&clearance), "{clearance}");
    assert!(fastest(state) < 1e-2, "{state:?}");
}

#[test]
fn a_fast_impact_rebounds_near_the_material_restitution() {
    // Steel block on rock: restitution 0.2 above the 1 m/s threshold.
    let mut world = box_above_floor(0.05, -4.0);
    let mut incoming = 4.0;
    for tick in 1..=30 {
        let state = world.tick(GRAVITY);
        if state.velocities[1] > 0.0 {
            let rebound = state.velocities[1];
            let expected = 0.2 * incoming;
            assert!(
                (0.5 * expected..1.5 * expected).contains(&rebound),
                "tick {tick}: rebound {rebound}, expected about {expected}"
            );
            return;
        }
        incoming = -state.velocities[1];
    }
    panic!("the box never rebounded");
}

#[test]
fn a_slow_impact_does_not_bounce() {
    let mut world = box_above_floor(0.002, -0.5);
    for tick in 1..=30 {
        let state = world.tick(GRAVITY);
        assert!(
            state.velocities[1] < 0.05,
            "tick {tick}: rebound {}",
            state.velocities[1]
        );
    }
    assert!(world.machine.snapshot().state.velocities[1].abs() < 1e-2);
}

#[test]
fn a_box_leaving_the_floor_is_not_pulled_back() {
    let mut world = box_above_floor(0.0, 0.5);
    let state = world.tick(GRAVITY);
    let expected = 0.5 - 9.81 * TICK_SECONDS;
    assert!(
        (state.velocities[1] - expected).abs() < 1e-9,
        "{} expected {expected}",
        state.velocities[1]
    );
}

#[test]
fn a_box_dropped_on_a_resting_box_comes_to_rest_on_top_of_it() {
    let creation = loose_cubes();
    let state = MachineState {
        poses: vec![pose(DVec3::Y * 0.5), pose(DVec3::Y * 1.52)],
        ..MachineState::at_rest(&creation)
    };
    let mut world = World::new(creation, state);
    for tick in 1..=120 {
        let state = world.tick(GRAVITY);
        let gap = state.poses[1].position.y - state.poses[0].position.y;
        assert!(gap > 0.99, "tick {tick}: boxes {gap} apart");
    }
    let state = &world.machine.snapshot().state;
    assert!((state.poses[0].position.y - 0.5).abs() < 0.005, "{state:?}");
    assert!((state.poses[1].position.y - 1.5).abs() < 0.01, "{state:?}");
    assert!(fastest(state) < 2e-2, "{state:?}");
}

#[test]
fn a_held_box_stays_put_under_a_dropped_box_and_falls_once_released() {
    let creation = loose_cubes();
    let state = MachineState {
        poses: vec![pose(DVec3::Y * 2.0), pose(DVec3::Y * 3.2)],
        ..MachineState::at_rest(&creation)
    };
    let mut world = World::new(creation, state);
    let held = world.machine.snapshot().state.poses.clone();
    world.machine.hold(&[true, false], &held).unwrap();
    for tick in 1..=120 {
        let state = world.tick(GRAVITY);
        let drift = state.poses[0].position.distance(held[0].position);
        assert!(drift < 1e-6, "tick {tick}: the held box moved {drift} m");
        let gap = state.poses[1].position.y - state.poses[0].position.y;
        assert!(gap > 0.99, "tick {tick}: boxes {gap} apart");
    }
    let state = &world.machine.snapshot().state;
    assert!((state.poses[1].position.y - 3.0).abs() < 0.01, "{state:?}");

    world.machine.hold(&[false, false], &held).unwrap();
    for _ in 0..30 {
        world.tick(GRAVITY);
    }
    let fallen = 2.0 - world.machine.snapshot().state.poses[0].position.y;
    assert!(fallen > 0.5, "the released box fell only {fallen} m");
}

#[test]
fn a_sliding_box_pushes_a_floating_box_and_keeps_momentum() {
    let creation = loose_cubes();
    let x = |body: usize| creation.dynamics.body_velocities[body].start;
    let (first, second) = (x(0), x(1));
    let mut state = MachineState {
        poses: vec![
            pose(DVec3::new(0.0, 5.0, 0.0)),
            pose(DVec3::new(1.2, 5.0, 0.0)),
        ],
        ..MachineState::at_rest(&creation)
    };
    state.velocities[first] = 1.0;
    let mut world = World::new(creation, state);
    for tick in 1..=60 {
        let state = world.tick(DVec3::ZERO);
        let gap = state.poses[1].position.x - state.poses[0].position.x;
        assert!(gap > 0.99, "tick {tick}: boxes {gap} apart");
        let momentum = state.velocities[first] + state.velocities[second];
        assert!((momentum - 1.0).abs() < 1e-9, "tick {tick}: {momentum}");
    }
    let state = &world.machine.snapshot().state;
    assert!(state.velocities[second] > 0.4, "{state:?}");
}

#[test]
fn a_sprung_block_settles_at_its_spring_equilibrium() {
    let spring = SpringSpec::default();
    let shock = ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.0, 20.0, 20.0).unwrap();
    let spec = SuspensionSpec::new(Some(spring), Some(shock), None).unwrap();
    let creation = suspension(spec, false);
    let mut state = MachineState::at_rest(&creation);
    for pose in &mut state.poses {
        pose.position.y += 0.502;
    }
    let equilibrium =
        -f64::from(creation.compounds[1].mass_properties.mass) * 9.81 / f64::from(spring.rate());
    let mut world = World::new(creation, state);
    for _ in 0..240 {
        world.tick(GRAVITY);
    }
    let state = &world.machine.snapshot().state;
    assert!(
        (state.coordinates[0] - equilibrium).abs() < 0.005,
        "{} expected {equilibrium}",
        state.coordinates[0]
    );
    assert!(clearance(world.creation(), state) > -0.005);
}

#[test]
fn a_preloaded_spring_reaches_its_authored_equilibrium() {
    for preload in [0.0, 0.005] {
        let spring = SpringSpec::new(0.5, 0.16, 0.12, 6, preload).unwrap();
        let shock = ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.0, 20.0, 20.0).unwrap();
        let spec = SuspensionSpec::new(Some(spring), Some(shock), None).unwrap();
        let creation = suspension(spec, true);
        let expected = f64::from(spec.passive_rows()[0][1])
            - f64::from(creation.compounds[1].mass_properties.mass) * 9.81
                / f64::from(spring.rate());
        let state = MachineState::at_rest(&creation);
        let mut world = World::new(creation, state);
        for _ in 0..360 {
            world.tick(GRAVITY);
        }
        let state = &world.machine.snapshot().state;
        assert!(
            (state.coordinates[0] - expected).abs() < 1e-3,
            "preload {preload}: {} expected {expected}",
            state.coordinates[0]
        );
        assert!(state.velocities[0].abs() < 1e-2);
    }
}

#[test]
fn a_motor_spins_its_rotor_and_back_drives_the_floating_mount() {
    let creation = rotor(false);
    let row = creation.dynamics.coordinate_velocities[0];
    let drive = motor(&creation, 120.0, 10.0);
    let state = MachineState::at_rest(&creation);
    let mut world = World::new(creation, state);
    // Far above the floor, without gravity: only the motor acts.
    for pose in &mut world.machine.completed.state.poses {
        pose.position.y += 10.0;
    }
    let command = world.command(0, drive);
    let state = world.tick_with(DVec3::ZERO, &[command]);
    assert!(state.velocities[row] > 0.0, "{state:?}");
    assert!(state.velocities[3] < 0.0, "{state:?}");
    let budget = 120.0 * TICK_SECONDS;
    let applied = world.machine.diagnostics().drive_impulses[0];
    assert!(applied > 0.0 && applied <= budget + 1e-9, "{applied}");
}

#[test]
fn a_joint_limit_holds_against_a_stalled_motor() {
    let creation = rotor(true);
    let row = creation.dynamics.coordinate_velocities[0];
    let mut drive = motor(&creation, 120.0, 10.0);
    drive.max_angle = 0.1;
    let limit = f64::from(drive.max_angle);
    let mut state = MachineState::at_rest(&creation);
    state.coordinates[0] = limit;
    let mut world = World::new(creation, state);
    for pose in &mut world.machine.completed.state.poses {
        pose.position.y += 10.0;
    }
    let command = world.command(0, drive);
    world.tick_with(DVec3::ZERO, &[command]);
    for _ in 0..30 {
        let state = world.tick(DVec3::ZERO);
        assert!(state.coordinates[0] <= limit + 1e-9, "{state:?}");
    }
    let state = &world.machine.snapshot().state;
    assert!(state.velocities[row].abs() < 0.05, "{state:?}");
}

#[test]
fn a_long_multi_collider_body_settles_on_a_redundant_manifold() {
    use mechanic_core::RigidLinkSpec;
    let mut graph = ConstructionGraph::new();
    let parts = (0..12)
        .map(|index| spawn(&mut graph, IVec3::X * (600 * index), [4, 4, 4]))
        .collect::<Vec<_>>();
    for pair in parts.windows(2) {
        graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: pair[0],
                second: pair[1],
            }))
            .unwrap();
    }
    let creation = graph.compile().unwrap();
    let mut state = MachineState::at_rest(&creation);
    for pose in &mut state.poses {
        pose.position.y += 0.502;
    }
    let mut world = World::new(creation, state);
    for tick in 1..=60 {
        let state = world.tick(GRAVITY);
        let clearance = clearance(world.creation(), &state);
        assert!(clearance > -0.005, "tick {tick}: clearance {clearance}");
    }
    assert!(fastest(&world.machine.snapshot().state) < 1e-2);
}

const WHEEL_RADIUS: f64 = 0.475;

// A free 0.95 m steel wheel resting on the floor, rolling along +X on its Z axle.
fn rolling_wheel(speed: f64) -> (CompiledCreation, MachineState) {
    use mechanic_core::{CylinderDimensions, CylinderSpec};
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
            CylinderDimensions::new(0.95, 0.0, 0.25).unwrap(),
            BuildPose::from_position_ticks(IVec3::Y * 300, GridRotation::new(1, 0, 0)),
        )))
        .unwrap();
    let creation = graph.compile().unwrap();
    let mut state = MachineState::at_rest(&creation);
    state.poses[0].position.y -= clearance(&creation, &state);
    state.velocities[0] = speed;
    state.velocities[5] = -speed / WHEEL_RADIUS;
    (creation, state)
}

#[test]
fn a_rolling_wheel_keeps_its_axle_height() {
    let (creation, state) = rolling_wheel(2.0);
    let mut world = World::new(creation, state);
    let [mut lowest, mut highest] = [f64::INFINITY, f64::NEG_INFINITY];
    for tick in 1..=120 {
        let height = world.tick(GRAVITY).poses[0].position.y;
        // The first ticks settle onto the soft contact.
        if tick > 10 {
            lowest = lowest.min(height);
            highest = highest.max(height);
        }
    }
    // A sixteen-sided wheel rose and fell 9 mm with every facet.
    assert!(highest - lowest < 0.001, "axle moved {}", highest - lowest);
}

#[test]
fn a_rolling_wheel_keeps_its_speed() {
    let (creation, state) = rolling_wheel(2.0);
    let mut world = World::new(creation, state);
    for _ in 0..60 {
        world.tick(GRAVITY);
    }
    let state = &world.machine.snapshot().state;
    // Rolling resistance alone takes about 0.1 m/s in a second.
    assert!(state.velocities[0] > 1.8, "{:?}", &state.velocities[0..6]);
    assert!(
        state.poses[0].position.z.abs() < 0.001,
        "{:?}",
        state.poses[0]
    );
}

#[test]
fn a_wheel_rolled_into_a_low_kerb_never_passes_through_it() {
    const FACE: f64 = 1.5;
    const HEIGHT: f64 = 0.1;
    let (creation, state) = rolling_wheel(3.0);
    #[allow(clippy::cast_possible_truncation)] // Exact small constants.
    let mut world = World::over(creation, state, kerb(FACE as f32, HEIGHT as f32));
    for tick in 1..=120 {
        let axle = world.tick(GRAVITY).poses[0].position;
        // Overlap of the wheel's circle with the kerb's solid corner region.
        let nearest = DVec3::new(axle.x.max(FACE), axle.y.min(HEIGHT), axle.z);
        let overlap = if axle.x >= FACE && axle.y <= HEIGHT {
            WHEEL_RADIUS + HEIGHT - axle.y
        } else {
            WHEEL_RADIUS - axle.distance(nearest)
        };
        assert!(
            overlap < 0.01,
            "tick {tick}: {overlap} m into the kerb at {axle}"
        );
    }
}

// The rolling wheel laid on an end cap, tipped 0.03 rad short of flat and
// spinning about its axis at `spin` rad/s, with its lowest point on the floor.
fn face_down_wheel(spin: f64) -> World {
    let (creation, mut state) = rolling_wheel(0.0);
    let local = creation.cylinders[0].local_rotation.as_dquat() * DVec3::Y;
    let pose = &mut state.poses[0];
    let axis = pose.rotation * local;
    pose.rotation =
        DQuat::from_rotation_x(0.03) * DQuat::from_rotation_arc(axis, DVec3::Y) * pose.rotation;
    let spin = spin * (pose.rotation * local);
    state.velocities[3..6].copy_from_slice(&spin.to_array());
    state.poses[0].position.y -= clearance(&creation, &state);
    World::new(creation, state)
}

// The deepest clearance of the wheel's lowest point over `ticks`.
fn deepest_clearance(world: &mut World, ticks: usize) -> f64 {
    (0..ticks).fold(f64::INFINITY, |deepest, _| {
        let state = world.tick(GRAVITY);
        deepest.min(clearance(world.creation(), &state))
    })
}

#[test]
fn a_wheel_tipping_onto_its_face_stays_on_the_floor() {
    let deepest = deepest_clearance(&mut face_down_wheel(0.0), 60);
    // It rests about 2 mm deep. When its lowest rim point lost every manifold
    // corner, it sank 5.8 mm.
    assert!(deepest > -0.004, "lowest point {deepest} m below the floor");
}

#[test]
fn a_wheel_spinning_on_its_face_stays_on_the_floor() {
    // Spinning 60 rad/s, it may hop as it slaps flat, but never sinks.
    let deepest = deepest_clearance(&mut face_down_wheel(60.0), 60);
    assert!(deepest > -0.004, "lowest point {deepest} m below the floor");
}

#[test]
fn a_fast_spinning_wheel_dropped_on_the_floor_lands_on_it() {
    let (creation, mut state) = rolling_wheel(0.0);
    state.poses[0].position.y += 0.3;
    state.velocities[1] = -15.0;
    state.velocities[5] = -40.0;
    let mut world = World::new(creation, state);
    for tick in 1..=30 {
        let state = world.tick(GRAVITY);
        let clearance = clearance(world.creation(), &state);
        assert!(
            clearance > -0.01,
            "tick {tick}: {clearance} m below the floor"
        );
    }
}

#[test]
fn a_rolling_wheel_does_not_gain_energy() {
    let (creation, state) = rolling_wheel(1.0);
    let properties = creation.compounds[0].mass_properties;
    let mass = f64::from(properties.mass);
    let inertia = properties.inertia.as_dmat3();
    let energy = |state: &MachineState| {
        let linear = DVec3::from_slice(&state.velocities[0..3]);
        let angular = DVec3::from_slice(&state.velocities[3..6]);
        let rotation = bevy_math::DMat3::from_quat(state.poses[0].rotation);
        let world_inertia = rotation * inertia * rotation.transpose();
        0.5 * mass * linear.length_squared()
            + 0.5 * angular.dot(world_inertia * angular)
            + mass * 9.81 * state.poses[0].position.y
    };
    let initial = energy(&state);
    let mut world = World::new(creation, state);
    for tick in 1..=120 {
        let state = world.tick(GRAVITY);
        let current = energy(&state);
        // A millimetre of lift is the most a soft contact may add.
        assert!(
            current <= initial + mass * 9.81 * 0.001,
            "tick {tick}: energy {current} from {initial}"
        );
        assert!(state.velocities[0] > -1e-3, "tick {tick}: rolled backwards");
    }
}

// Deepest overlap of any collider with the floor or another body, from the
// recovery query's buried points.
fn deepest_overlap(world: &World, state: &MachineState) -> f64 {
    world
        .scene
        .recovery_groups(&world.geometry, &state.poses, DVec3::ZERO, None)
        .unwrap()
        .contacts
        .iter()
        .fold(0.0_f64, |deepest, contact| deepest.max(contact.depth))
}

// Blocks captured from app saves, which once froze the exact solver. A chaotic
// landing may come to rest in more than one pose; none may bury a block.
fn captured_blocks_come_to_rest(instance: &str) {
    let (creation, state) = captured(instance);
    let mut world = World::new(creation, state);
    for _ in 0..600 {
        world.tick(GRAVITY);
    }
    let state = world.machine.snapshot().state.clone();
    let overlap = deepest_overlap(&world, &state);
    assert!(overlap < 0.005, "deepest overlap {overlap}");
    assert!(clearance(world.creation(), &state) > -0.005);
    assert!(fastest(&state) < 0.2, "{state:?}");
}

#[test]
fn captured_blocks_dropped_across_a_ledge_come_to_rest() {
    captured_blocks_come_to_rest(include_str!(
        "../joint_machine/tests/terrain_ticks/fixtures/ledge_blocks_instance.ron"
    ));
}

#[test]
fn captured_blocks_landing_on_each_other_come_to_rest() {
    captured_blocks_come_to_rest(include_str!(
        "../joint_machine/tests/terrain_ticks/fixtures/leaning_blocks_instance.ron"
    ));
}

#[test]
fn the_saved_car_survives_a_cold_drop_and_settles() {
    let (creation, mut state) = saved_car();
    for pose in &mut state.poses {
        pose.position.y += 0.051;
    }
    state.velocities[1] = -4.0;
    let mut world = World::new(creation, state);
    let (mut deepest, mut settled) = (0.0_f64, 0.0_f64);
    for tick in 1..=600 {
        let state = world.tick(GRAVITY);
        let depth = -clearance(world.creation(), &state);
        deepest = deepest.max(depth);
        if tick > 120 {
            settled = settled.max(depth);
        }
    }
    // A 4 m/s landing compresses the soft contacts briefly; resting depth is the slop.
    assert!(deepest < 0.025, "deepest penetration {deepest}");
    assert!(settled < 0.003, "settled penetration {settled}");
    let state = &world.machine.snapshot().state;
    assert!(
        state.velocities[..6].iter().all(|v| v.abs() < 5e-2),
        "{state:?}"
    );
}

#[test]
fn the_saved_car_drives_forward_on_its_speed_drives() {
    let (creation, state) = saved_car();
    let driven = creation
        .coordinate_drives
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, drive)| drive.mode == DriveMode::Speed)
        .collect::<Vec<_>>();
    assert!(!driven.is_empty(), "the fixture keeps its authored drives");
    let start = state.poses[0].position;
    let mut world = World::new(creation, state);
    // Let the car settle on its suspension first.
    for _ in 0..60 {
        world.tick(GRAVITY);
    }
    let commands = driven
        .iter()
        .map(|&(coordinate, mut drive)| {
            drive.target_speed = 8.0;
            world.command(coordinate, drive)
        })
        .collect::<Vec<_>>();
    world.tick_with(GRAVITY, &commands);
    for _ in 0..180 {
        world.tick(GRAVITY);
    }
    let end = world.machine.snapshot().state.poses[0].position;
    let travelled = (end - start).with_y(0.0).length();
    assert!(travelled > 0.5, "the car moved {travelled} m");
    assert!(clearance(world.creation(), &world.machine.snapshot().state) > -0.01);
}

#[test]
fn identical_inputs_repeat_exactly() {
    let run = || {
        let (creation, mut state) = saved_car();
        state.velocities[1] = -2.0;
        let mut world = World::new(creation, state);
        for _ in 0..120 {
            world.tick(GRAVITY);
        }
        world.machine.snapshot().state_hash()
    };
    assert_eq!(run(), run());
}

#[test]
fn invalid_input_is_refused_without_changing_state() {
    let mut world = box_above_floor(0.0, 0.0);
    let before = world.machine.snapshot().clone();
    let stale = DriveCommand {
        tick: 5,
        topology_generation: GENERATION,
        coordinate: 0,
        drive: CoordinateDrive::PASSIVE,
    };
    assert_eq!(
        world
            .machine
            .step(GRAVITY, &world.settings, &[], &[stale], None)
            .map(|_| ()),
        Err(PhysicsError::InvalidCommand)
    );
    assert_eq!(world.machine.snapshot(), &before);
}

#[test]
fn a_two_wheeled_cart_tips_its_tail_onto_the_floor() {
    // The builder's cart balances its chassis, tail behind the axle, on two
    // wheels hung from double wishbones closed by loaded suspension struts.
    let (creation, state) = captured(include_str!(
        "../../../mechanic-bench/tests/fixtures/builder-world/generations/20/world.ron"
    ));
    let height = |creation: &CompiledCreation, state: &MachineState| {
        let (mass, moment) = state.poses.iter().zip(&creation.dynamics.inertias).fold(
            (0.0, 0.0),
            |(mass, moment), (pose, inertia)| {
                let center = pose.position + pose.rotation * inertia.center.as_dvec3();
                let body = f64::from(inertia.mass);
                (mass + body, moment + body * center.y)
            },
        );
        moment / mass
    };
    let start = height(&creation, &state);
    let mut world = World::new(creation, state);
    for _ in 0..300 {
        world.tick(GRAVITY);
    }
    let settled = height(world.creation(), &world.machine.snapshot().state);
    assert!(
        settled < start - 0.1,
        "the tail stayed up: centre of mass {settled} from {start}"
    );
    for tick in 301..=600 {
        let state = world.tick(GRAVITY);
        let current = height(world.creation(), &state);
        assert!(
            current < settled + 0.005,
            "tick {tick}: the tail climbed to {current} from {settled}"
        );
    }
}

#[test]
fn fast_free_rotor_keeps_spinning_while_its_static_base_stays_fixed() {
    let creation = rotor(true);
    let row = creation.dynamics.coordinate_velocities[0];
    let mut initial = MachineState::at_rest(&creation);
    for pose in &mut initial.poses {
        pose.position.y += 3.0;
    }
    initial.velocities[row] = 400.0;
    let root = creation
        .compounds
        .iter()
        .position(|body| body.is_static)
        .unwrap();
    let fixed = initial.poses[root];
    let mut world = World::new(creation, initial);
    for _ in 0..30 {
        let state = world.tick(DVec3::ZERO);
        assert_eq!(state.poses[root], fixed);
        assert!((state.velocities[row] - 400.0).abs() < 1e-6);
        assert!(!world.machine.diagnostics().degraded);
    }
}

#[test]
fn spinning_pipe_preserves_distant_wishbone_support_and_free_rotation() {
    use mechanic_core::{PipeBendDimensions, PipeBendSpec};
    let instance: mechanic_world::WorldCreationInstanceDoc = ron::from_str(include_str!(
        "../../../mechanic-bench/tests/fixtures/driven_car_instance.ron"
    ))
    .unwrap();
    let mut loaded = instance.creation.into_graph().unwrap();
    let root = spawn(&mut loaded.graph, IVec3::new(8160, 5000, 0), [4, 4, 4]);
    let BuildOutcome::Spawned(pipe) = loaded
        .graph
        .apply(BuildCommand::SpawnPipeBend(PipeBendSpec::new(
            PipeBendDimensions::new(1.0, 0.6, 6).unwrap(),
            BuildPose::from_position_ticks(IVec3::new(8000, 4400, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        panic!("pipe expected")
    };
    loaded
        .graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(root, FaceKind::NegativeY),
            FaceRef::part(pipe, FaceKind::PositiveY),
            Vec3::new(20.4, 12.0, 0.0),
            Vec3::NEG_Y,
        )))
        .unwrap();
    let creation = loaded
        .graph
        .compile_with_suspension_sockets([root], &loaded.sockets)
        .unwrap();
    let pipe_body = creation
        .colliders
        .iter()
        .find(|c| c.source_part == pipe)
        .unwrap()
        .compound_index;
    let coordinate = creation
        .bearings
        .iter()
        .find(|b| b.compound_a == pipe_body || b.compound_b == pipe_body)
        .unwrap()
        .coordinate_index
        .unwrap() as usize;
    let row = creation.dynamics.coordinate_velocities[coordinate];
    let initial = lifted(&creation, 0.001);
    let mut settled = World::new(creation.clone(), initial);
    for _ in 0..120 {
        settled.tick(GRAVITY);
    }
    let initial = settled.machine.snapshot().state.clone();
    let mut spinning = initial.clone();
    spinning.velocities[row] = 200.0;
    let mut slow = World::new(creation.clone(), initial);
    let mut fast = World::new(creation, spinning);
    for _ in 0..120 {
        let expected = slow.tick(GRAVITY);
        let actual = fast.tick(GRAVITY);
        assert!((actual.velocities[row] - 200.0).abs() < 1e-5);
        for (a, b) in actual
            .poses
            .iter()
            .zip(&expected.poses)
            .filter(|(a, _)| a.position.x < 10.0)
        {
            assert!(
                a.position.distance(b.position) < 0.02,
                "distant support changed"
            );
        }
        assert!(clearance(fast.creation(), &actual) > -0.03);
    }
}

#[test]
fn a_resting_body_reports_terrain_normal_load_matching_its_weight() {
    for substeps in [2, 8] {
        let mut world = box_above_floor(0.01, 0.0);
        world.settings.substeps = substeps;
        for _ in 0..120 {
            world.tick(GRAVITY);
        }
        let expected = world
            .creation()
            .compounds
            .iter()
            .map(|body| f64::from(body.mass_properties.mass))
            .sum::<f64>()
            * 9.81
            * crate::TICK_SECONDS;
        let impulse = world
            .machine
            .terrain_loads()
            .iter()
            .map(|load| load.normal_impulse)
            .sum::<f64>();
        assert!(
            (impulse / expected - 1.0).abs() < 0.1,
            "substeps={substeps}: {impulse} vs {expected}"
        );
        assert!(
            world
                .machine
                .terrain_loads()
                .iter()
                .all(|load| (0.05..=0.3).contains(&load.patch_radius) && load.normal.y > 0.99)
        );
        world
            .machine
            .step(DVec3::ZERO, &world.settings, &[], &[], None)
            .unwrap();
        assert!(world.machine.terrain_loads().is_empty());
    }
}

#[test]
fn reported_impact_load_includes_the_rebound_impulse() {
    let mut world = box_above_floor(0.05, -4.0);
    let mass = f64::from(world.creation().compounds[0].mass_properties.mass);
    for _ in 0..30 {
        let before = world.machine.snapshot().state.velocities[1];
        let after = world.tick(GRAVITY).velocities[1];
        if after > 0.0 {
            let expected = mass * (after - before - GRAVITY.y * crate::TICK_SECONDS);
            let impulse = world
                .machine
                .terrain_loads()
                .iter()
                .map(|load| load.normal_impulse)
                .sum::<f64>();
            assert!(
                (impulse / expected - 1.0).abs() < 0.1,
                "{impulse} vs {expected}"
            );
            return;
        }
    }
    panic!("the body never rebounded");
}
