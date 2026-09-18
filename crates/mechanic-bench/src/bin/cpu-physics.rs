//! CPU physics quality report. Prints JSONL and always exits successfully:
//! solver quality is tracked here, not gated in `cargo test`.

use std::{collections::BTreeMap, error::Error, fs, path::Path, sync::Arc, time::Instant};

use bevy_math::{DVec3, IVec3};
use mechanic_core::{
    BuildCommand, BuildPose, CompiledCreation, ConstructionGraph, ContactCylinder, ContactPolytope,
    CuboidSpec, DriveMode, GRAVITY, GridRotation,
};
use mechanic_physics::{
    ConstraintBlock, ConstraintSolution, ContactFriction, CpuMachine, DriveCommand, DynamicsFactor,
    ImpulseBounds, MachineCollisionGeometry, MachineState, PreparedConstraints, SoftStepSettings,
    SoftStepTerrain, TerrainContactScene, solve_constraints,
};
use mechanic_world::{
    TerrainCollisionChunk, TerrainMaterial, TerrainNodeId, TerrainTriangleGroupMask, TriangleBvh,
    TriangleBvhNode, TriangleBvhTriangle, WorldBounds, WorldPosition,
};
use serde_json::json;

/// Position of the wall faced by `fast-impacts` cases, along +X.
const WALL: f32 = 5.0;

#[path = "compiled-response/finite_support.rs"]
mod finite_support;

#[path = "cpu-physics/scale.rs"]
mod scale;

#[path = "cpu-physics/motion.rs"]
mod motion;

#[path = "cpu-physics/pipe_motion.rs"]
mod pipe_motion;

#[path = "cpu-physics/rolling.rs"]
mod rolling;

#[path = "cpu-physics/world_drive.rs"]
mod world_drive;

const ITERATIONS: usize = 256;
const TOLERANCE: f64 = 1e-9;

type RecordedBlock = (
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<(f64, f64)>,
    Vec<(f64, f64, bool, Option<f64>)>,
);

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut scenario = "reference-fixtures".to_owned();
    let mut instance = None;
    let mut background = None;
    let mut soil = false;
    let mut block_width = 8;
    let mut scale = scale::Options {
        copies: 1,
        connected: false,
        warmup: 600,
        ticks: 3600,
        floor: false,
        hold: false,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--scenario" => scenario = args.next().ok_or("--scenario needs a value")?,
            "--instance" => instance = Some(args.next().ok_or("--instance needs a value")?),
            "--background" => background = Some(args.next().ok_or("--background needs a value")?),
            "--copies" => {
                scale.copies = args.next().ok_or("--copies needs a value")?.parse()?;
                if ![1, 2, 5, 10].contains(&scale.copies) {
                    return Err("copies must be 1, 2, 5 or 10".into());
                }
            }
            "--connected" => scale.connected = true,
            "--block-width" => {
                block_width = args.next().ok_or("--block-width needs a value")?.parse()?;
            }
            "--soil" => soil = true,
            "--hold" => scale.hold = true,
            "--floor" => scale.floor = true,
            "--warmup" => scale.warmup = args.next().ok_or("--warmup needs a value")?.parse()?,
            "--ticks" => {
                scale.ticks = args.next().ok_or("--ticks needs a value")?.parse()?;
                if scale.ticks == 0 {
                    return Err("ticks must be positive".into());
                }
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if soil && !matches!(scenario.as_str(), "world-drive" | "large-surface") {
        return Err("--soil requires world-drive or large-surface".into());
    }
    match scenario.as_str() {
        "builder-scale" => scale::run(&scale),
        "fast-motion" => motion::run(&scale),
        "pipe-motion" => pipe_motion::run(
            instance.as_deref().ok_or("pipe-motion needs --instance")?,
            &scale,
        ),
        "pipe-scene" => pipe_motion::scene_ticks(
            instance.as_deref().ok_or("pipe-scene needs --instance")?,
            background.as_deref(),
            &scale,
        ),
        "reference-fixtures" => reference_fixtures(),
        "car-drop" => car(false),
        "car-drive" => car(true),
        "block-pile" => block_pile(),
        "fast-impacts" => fast_impacts(),
        "four-bar" => four_bar(),
        "wheel-roll" => rolling::run(),
        "large-surface" => world_drive::large_surface(&scale, soil, block_width),
        "world-drive" => world_drive::run(
            instance
                .as_deref()
                .ok_or("world-drive needs --instance <world directory>")?,
            &scale,
            soil,
        ),
        other => Err(format!(
            "unknown scenario {other}; expected reference-fixtures, car-drop, car-drive, \
             block-pile, fast-impacts, four-bar, wheel-roll, large-surface or world-drive"
        )
        .into()),
    }
}

// A free planar parallelogram, whose coupler's second bearing closes a loop,
// dropped 30 cm onto the floor and left to topple and settle.
fn four_bar() -> Result<(), Box<dyn Error>> {
    use mechanic_core::{BearingSpec, FaceKind, FaceRef, PartId};
    let mut graph = ConstructionGraph::new();
    let mut spawn = |ticks: IVec3, dimensions: [u8; 3]| -> Result<PartId, Box<dyn Error>> {
        match graph.apply(BuildCommand::Spawn(CuboidSpec::new(
            dimensions,
            BuildPose::from_position_ticks(ticks, GridRotation::default()),
        )?))? {
            mechanic_core::BuildOutcome::Spawned(part) => Ok(part),
            _ => Err("spawn expected".into()),
        }
    };
    let height = IVec3::Y * 800;
    let ground = spawn(height, [8, 1, 1])?;
    let left = spawn(height + IVec3::new(-350, 250, 100), [1, 6, 1])?;
    let right = spawn(height + IVec3::new(350, 250, 100), [1, 6, 1])?;
    let coupler = spawn(height + IVec3::new(0, 500, 200), [8, 1, 1])?;
    for (source, target, x, y, z) in [
        (ground, left, -0.875, 0.0, 0.125),
        (ground, right, 0.875, 0.0, 0.125),
        (left, coupler, -0.875, 1.25, 0.375),
        (right, coupler, 0.875, 1.25, 0.375),
    ] {
        graph.apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(source, FaceKind::PositiveZ),
            FaceRef::part(target, FaceKind::NegativeZ),
            bevy_math::Vec3::new(x, y + 2.0, z),
            bevy_math::Vec3::Z,
        )))?;
    }
    let creation = graph.compile()?;
    let mut state = MachineState::at_rest(&creation);
    let lowest = lowest_point(&creation, &state)?;
    for pose in &mut state.poses {
        pose.position.y += 0.3 - lowest;
    }
    run("four-bar", &creation, state, 600, |_| Vec::new())
}

// The saved car dropped at 4 m/s, or settled and then driven on its speed drives.
fn car(drive: bool) -> Result<(), Box<dyn Error>> {
    let creation = saved_car()?;
    let mut state = MachineState::at_rest(&creation);
    if !drive {
        for pose in &mut state.poses {
            pose.position.y += 0.051;
        }
        state.velocities[1] = -4.0;
    }
    let commands = if drive {
        creation
            .coordinate_drives
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, drive)| drive.mode == DriveMode::Speed)
            .map(|(coordinate, mut drive)| {
                drive.target_speed = 8.0;
                (coordinate, drive)
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let name = if drive { "car-drive" } else { "car-drop" };
    run(name, &creation, state, 600, |tick| {
        if tick == 60 {
            commands
                .iter()
                .map(|&(coordinate, drive)| DriveCommand {
                    tick,
                    topology_generation: 1,
                    coordinate,
                    drive,
                })
                .collect()
        } else {
            Vec::new()
        }
    })
}

// Twenty-seven loose 1 m blocks in a 3 × 3 × 3 stack with 5 cm gaps.
fn block_pile() -> Result<(), Box<dyn Error>> {
    let mut graph = ConstructionGraph::new();
    for x in 0..3 {
        for y in 0..3 {
            for z in 0..3 {
                graph.apply(BuildCommand::Spawn(CuboidSpec::new(
                    [4, 4, 4],
                    BuildPose::from_position_ticks(
                        IVec3::new(x, y, z) * 420,
                        GridRotation::default(),
                    ),
                )?))?;
            }
        }
    }
    let creation = graph.compile()?;
    let state = MachineState::at_rest(&creation);
    run("block-pile", &creation, state, 600, |_| Vec::new())
}

fn run(
    name: &str,
    creation: &CompiledCreation,
    mut state: MachineState,
    ticks: u64,
    commands: impl Fn(u64) -> Vec<DriveCommand>,
) -> Result<(), Box<dyn Error>> {
    let mut roots = state.poses.clone();
    let (geometry, scene) = finite_support::scene(creation, &mut roots)?;
    state.poses = roots;
    let start = state.poses[0].position;
    let settings = SoftStepSettings::default();
    let mut machine = CpuMachine::new(creation.clone(), 1, state)?;
    let mut samples = Vec::new();
    let (mut deepest, mut settled, mut degraded) = (0.0_f64, 0.0_f64, 0_u64);
    let (mut closure_gap, mut closure_angle) = (0.0_f64, 0.0_f64);
    let (mut continuous, mut sweeps) = (0.0, 0);
    let mut reasons = std::collections::BTreeMap::<&str, u64>::new();
    for tick in 1..=ticks {
        let terrain = SoftStepTerrain {
            scene: &scene,
            geometry: &geometry,
            topology_generation: 1,
            origin: DVec3::ZERO,
        };
        let started = Instant::now();
        machine.step(GRAVITY, &settings, &[], &commands(tick), Some(terrain))?;
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
        let diagnostics = machine.diagnostics();
        degraded += u64::from(diagnostics.degraded);
        continuous += diagnostics.continuous_ms;
        sweeps += diagnostics.continuous_sweeps;
        closure_gap = closure_gap.max(diagnostics.closure_position_error);
        closure_angle = closure_angle.max(diagnostics.closure_angle_error);
        if let Some(reason) = diagnostics.degraded_reason {
            *reasons.entry(reason).or_default() += 1;
        }
        let depth = -lowest_point(creation, &machine.snapshot().state)?;
        deepest = deepest.max(depth);
        if tick + 120 > ticks {
            settled = settled.max(depth);
        }
    }
    samples.sort_by(f64::total_cmp);
    let percentile =
        |fraction: usize| samples[(samples.len() * fraction / 100).min(samples.len() - 1)];
    let state = &machine.snapshot().state;
    let record = json!({
        "scenario": name,
        "ticks": ticks,
        "p50_ms": percentile(50),
        "p95_ms": percentile(95),
        "continuous_mean_ms": continuous / f64::from(u32::try_from(ticks)?),
        "continuous_sweeps": sweeps,
        "deepest_m": deepest,
        "settled_m": settled,
        "degraded_ticks": degraded,
        "degraded_reasons": reasons,
        "closures": creation.dynamics.loops.len(),
        "closure_gap_m": closure_gap,
        "closure_angle_rad": closure_angle,
        "travelled_m": (state.poses[0].position - start).with_y(0.0).length(),
        "fastest_final": state.velocities.iter().fold(0.0_f64, |m, v| m.max(v.abs())),
    });
    println!("{record}");
    Ok(())
}

// Lowest collider point above the floor; negative values are penetration.
fn lowest_point(creation: &CompiledCreation, state: &MachineState) -> Result<f64, Box<dyn Error>> {
    Ok(extent(creation, state)?[0].y)
}

// Corners of the box around every collider, taking a solid cylinder as the
// circle the CPU route rolls it on rather than its sixteen boxes.
fn extent(creation: &CompiledCreation, state: &MachineState) -> Result<[DVec3; 2], Box<dyn Error>> {
    let mut extent = [DVec3::INFINITY, DVec3::NEG_INFINITY];
    for cylinder in &creation.cylinders {
        let pose = state.poses[cylinder.compound_index as usize];
        let world =
            ContactCylinder::from_compiled(cylinder)?.transformed(pose.position, pose.rotation)?;
        for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
            extent[0] = extent[0].min(world.support(-axis));
            extent[1] = extent[1].max(world.support(axis));
        }
    }
    for (row, collider) in creation.colliders.iter().enumerate() {
        let rolled = creation.cylinders.iter().any(|cylinder| {
            let first = cylinder.first_collider as usize;
            (first..first + mechanic_core::CYLINDER_COLLIDER_COUNT).contains(&row)
        });
        if rolled {
            continue;
        }
        let pose = state.poses[collider.compound_index as usize];
        let [minimum, maximum] = ContactPolytope::from_collider(collider)?
            .transformed(pose.position, pose.rotation)?
            .bounds();
        extent = [extent[0].min(minimum), extent[1].max(maximum)];
    }
    Ok(extent)
}

fn saved_car() -> Result<CompiledCreation, Box<dyn Error>> {
    let instance: mechanic_world::WorldCreationInstanceDoc =
        ron::from_str(include_str!("../../tests/fixtures/driven_car_instance.ron"))?;
    let loaded = instance.creation.into_graph()?;
    Ok(loaded
        .graph
        .compile_with_suspension_sockets([], &loaded.sockets)?)
}

// Cuboids of the given block dimensions at lattice positions.
fn cuboids(parts: &[([u8; 3], IVec3)]) -> Result<CompiledCreation, Box<dyn Error>> {
    let mut graph = ConstructionGraph::new();
    for &(dimensions, ticks) in parts {
        graph.apply(BuildCommand::Spawn(CuboidSpec::new(
            dimensions,
            BuildPose::from_position_ticks(ticks, GridRotation::default()),
        )?))?;
    }
    Ok(graph.compile()?)
}

// A creation with its lowest point `clearance` above the floor, and root bodies
// launched with (body, linear, angular) velocities.
fn launched(
    creation: &CompiledCreation,
    clearance: f64,
    launches: &[(usize, DVec3, DVec3)],
) -> Result<MachineState, Box<dyn Error>> {
    let mut state = MachineState::at_rest(creation);
    let lowest = lowest_point(creation, &state)?;
    for pose in &mut state.poses {
        pose.position.y += clearance - lowest;
    }
    for &(body, linear, angular) in launches {
        let rows = creation.dynamics.body_velocities[body].clone();
        state.velocities[rows].copy_from_slice(&[
            linear.x, linear.y, linear.z, angular.x, angular.y, angular.z,
        ]);
    }
    Ok(state)
}

// Fast bodies against the floor, a wall and each other. A case has tunnelled
// when a collider ends up more than one block past a surface.
fn fast_impacts() -> Result<(), Box<dyn Error>> {
    let cube = || cuboids(&[([1, 1, 1], IVec3::ZERO)]);
    for speed in [30.0, 60.0, 120.0, 250.0] {
        let creation = cube()?;
        let state = launched(&creation, 2.0, &[(0, DVec3::NEG_Y * speed, DVec3::ZERO)])?;
        impact(&format!("drop-{speed}"), &creation, state, false)?;
    }
    let creation = cube()?;
    let angle = 20.0_f64.to_radians();
    let graze = DVec3::new(angle.cos(), -angle.sin(), 0.0) * 120.0;
    let state = launched(&creation, 1.0, &[(0, graze, DVec3::ZERO)])?;
    impact("graze-120", &creation, state, false)?;
    let creation = cube()?;
    let state = launched(&creation, 0.5, &[(0, DVec3::X * 100.0, DVec3::ZERO)])?;
    impact("wall-100", &creation, state, true)?;
    let creation = cuboids(&[([8, 1, 1], IVec3::ZERO)])?;
    let state = launched(&creation, 0.5, &[(0, DVec3::NEG_Y * 5.0, DVec3::Z * 60.0)])?;
    impact("spinning-bar-60", &creation, state, false)?;
    let creation = cuboids(&[([4, 4, 4], IVec3::ZERO), ([4, 4, 4], IVec3::X * 800)])?;
    let mut state = launched(&creation, 0.001, &[(1, DVec3::NEG_Y * 80.0, DVec3::ZERO)])?;
    state.poses[1].position = state.poses[0].position + DVec3::Y * 3.5;
    impact("projectile-80", &creation, state, false)?;
    let creation = saved_car()?;
    let roots = (0..creation.compounds.len())
        .filter(|&body| creation.loop_topology.body_parents[body].is_root)
        .map(|body| (body, DVec3::X * 40.0, DVec3::ZERO))
        .collect::<Vec<_>>();
    let state = launched(&creation, 0.001, &roots)?;
    impact("car-crash-40", &creation, state, true)
}

fn impact(
    name: &str,
    creation: &CompiledCreation,
    mut state: MachineState,
    wall: bool,
) -> Result<(), Box<dyn Error>> {
    const TICKS: u64 = 90;
    if wall {
        // Five metres short of the wall.
        let front = extent(creation, &state)?[1].x;
        for pose in &mut state.poses {
            pose.position.x += f64::from(WALL) - 5.0 - front;
        }
    }
    let geometry = MachineCollisionGeometry::new(creation, 1)?;
    let mut scene = TerrainContactScene::default();
    scene.publish(1, &[ground(wall)], &[])?;
    let settings = SoftStepSettings::default();
    let mut machine = CpuMachine::new(creation.clone(), 1, state)?;
    let (mut samples, mut deepest, mut degraded) = (Vec::new(), 0.0_f64, 0_u64);
    let (mut requeries, mut hits) = (0_usize, 0_usize);
    let mut reasons = BTreeMap::<&str, u64>::new();
    for _ in 1..=TICKS {
        let terrain = SoftStepTerrain {
            scene: &scene,
            geometry: &geometry,
            topology_generation: 1,
            origin: DVec3::ZERO,
        };
        let started = Instant::now();
        machine.step(GRAVITY, &settings, &[], &[], Some(terrain))?;
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
        let diagnostics = machine.diagnostics();
        degraded += u64::from(diagnostics.degraded);
        requeries += diagnostics.requeries;
        hits += diagnostics.continuous_hits;
        if let Some(reason) = diagnostics.degraded_reason {
            *reasons.entry(reason).or_default() += 1;
        }
        let [minimum, maximum] = extent(creation, &machine.snapshot().state)?;
        // The floor only counts while the bodies are over it.
        let over_floor =
            minimum.x > -64.0 && maximum.x < 64.0 && minimum.z > -64.0 && maximum.z < 64.0;
        let mut depth = if over_floor { -minimum.y } else { 0.0 };
        if wall {
            depth = depth.max(maximum.x - f64::from(WALL));
        }
        deepest = deepest.max(depth);
    }
    samples.sort_by(f64::total_cmp);
    let percentile =
        |fraction: usize| samples[(samples.len() * fraction / 100).min(samples.len() - 1)];
    let record = json!({
        "scenario": "fast-impacts",
        "case": name,
        "ticks": TICKS,
        "p50_ms": percentile(50),
        "p95_ms": percentile(95),
        "deepest_m": deepest,
        "tunnelled": deepest > 0.25,
        "degraded_ticks": degraded,
        "degraded_reasons": reasons,
        "requeries": requeries,
        "continuous_hits": hits,
    });
    println!("{record}");
    Ok(())
}

// The 128 m rock floor, optionally with a wall at `WALL` facing −X.
fn ground(wall: bool) -> Arc<TerrainCollisionChunk> {
    let mut weights = [0.0; TerrainMaterial::COUNT];
    weights[usize::from(TerrainMaterial::Rock.code())] = 1.0;
    let mask = TerrainTriangleGroupMask::REGULAR;
    let mut vertices = vec![
        [-64.0, 0.0, -64.0],
        [-64.0, 0.0, 64.0],
        [64.0, 0.0, 64.0],
        [64.0, 0.0, -64.0],
    ];
    let mut indices = vec![0, 1, 2, 0, 2, 3];
    let mut bounds = WorldBounds {
        minimum: WorldPosition(DVec3::new(-64.0, 0.0, -64.0)),
        maximum: WorldPosition(DVec3::new(64.0, 0.0, 64.0)),
    };
    if wall {
        // Mapping (x, y, z) to (wall, x, z) keeps the winding, so the floor's
        // upward normal becomes −X.
        let floor = vertices.clone();
        vertices.extend(floor.into_iter().map(|[x, _, z]| [WALL, x, z]));
        indices.extend([4, 5, 6, 4, 6, 7]);
        bounds.minimum.0.y = -64.0;
        bounds.maximum.0.y = 64.0;
    }
    let triangles = indices
        .chunks(3)
        .map(|corners| TriangleBvhTriangle {
            indices: [corners[0], corners[1], corners[2]],
            group_mask: mask,
        })
        .collect::<Vec<_>>();
    Arc::new(TerrainCollisionChunk {
        node: TerrainNodeId::ROOT,
        material_weights: vec![weights; vertices.len()],
        vertices,
        indices,
        bounds,
        generation: 1,
        triangle_bvh: TriangleBvh {
            bounds,
            nodes: vec![TriangleBvhNode {
                bounds,
                triangle_count: if wall { 4 } else { 2 },
                group_mask: mask,
                ..Default::default()
            }],
            triangles,
        },
        active_groups: mask,
        ..Default::default()
    })
}

// Captured solves the exact reference solver once failed or narrowly passed.
fn reference_fixtures() -> Result<(), Box<dyn Error>> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/cpu-reference");
    let mut names = fs::read_dir(&directory)?
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| {
            std::path::Path::new(name)
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("ron"))
                && !name.ends_with("_reference.ron")
        })
        .collect::<Vec<_>>();
    names.sort();
    for name in names {
        let text = fs::read_to_string(directory.join(&name))?;
        let (mass, recorded, initial) = parse(&text)?;
        let blocks = blocks(recorded);
        let size = blocks
            .first()
            .and_then(|block| block.jacobian.first())
            .map_or(0, Vec::len);
        let started = Instant::now();
        let solution = solve(&mass, size, &blocks, initial.as_deref());
        let milliseconds = started.elapsed().as_secs_f64() * 1000.0;
        let record = match solution {
            Ok(solution) => {
                let reference = name.strip_suffix("_impact.ron").and_then(|stem| {
                    fs::read_to_string(directory.join(format!("{stem}_reference.ron"))).ok()
                });
                let reference_error = reference
                    .map(|text| -> Result<f64, Box<dyn Error>> {
                        let (_, velocity): (Vec<f64>, Vec<f64>) = ron::from_str(&text)?;
                        Ok(solution
                            .velocity_change
                            .iter()
                            .zip(&velocity)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0, f64::max))
                    })
                    .transpose()?;
                json!({
                    "fixture": name,
                    "rows": solution.impulses.len(),
                    "converged": solution.converged,
                    "residual": solution.residual,
                    "iterations": solution.iterations,
                    "ms": milliseconds,
                    "law_violation": law_violation(&blocks, &solution),
                    "reference_velocity_error": reference_error,
                })
            }
            Err(error) => json!({
                "fixture": name,
                "error": error.to_string(),
                "ms": milliseconds,
            }),
        };
        println!("{record}");
    }
    Ok(())
}

type Parsed = (Vec<f64>, Vec<RecordedBlock>, Option<Vec<f64>>);

fn parse(text: &str) -> Result<Parsed, Box<dyn Error>> {
    if let Ok(parsed) = ron::from_str::<Parsed>(text) {
        return Ok(parsed);
    }
    let (mass, blocks): (Vec<f64>, Vec<RecordedBlock>) = ron::from_str(text)?;
    Ok((mass, blocks, None))
}

fn solve(
    mass: &[f64],
    size: usize,
    blocks: &[ConstraintBlock],
    initial: Option<&[f64]>,
) -> Result<ConstraintSolution, Box<dyn Error>> {
    let factor = DynamicsFactor::new(mass, size)?;
    Ok(match initial {
        Some(initial) => {
            let targets = blocks
                .iter()
                .flat_map(|block| block.target.iter().copied())
                .collect::<Vec<_>>();
            PreparedConstraints::new(&factor, blocks)?.solve_from(
                &targets,
                Some(initial),
                ITERATIONS,
                TOLERANCE,
            )?
        }
        None => solve_constraints(&factor, blocks, ITERATIONS, TOLERANCE)?,
    })
}

fn blocks(recorded: Vec<RecordedBlock>) -> Vec<ConstraintBlock> {
    recorded
        .into_iter()
        .map(|(jacobian, target, bounds, contacts)| ConstraintBlock {
            jacobian,
            target,
            bounds: bounds
                .into_iter()
                .map(|(minimum, maximum)| ImpulseBounds { minimum, maximum })
                .collect(),
            contacts: contacts
                .into_iter()
                .map(
                    |(static_coefficient, kinetic_coefficient, sliding, rolling_length)| {
                        ContactFriction {
                            static_coefficient,
                            kinetic_coefficient,
                            sliding,
                            rolling_length,
                        }
                    },
                )
                .collect(),
        })
        .collect()
}

// Largest violation of bounds, normal complementarity, and friction/rolling disks,
// measured in physical row velocities and impulses. Zero is an exact solution.
fn law_violation(blocks: &[ConstraintBlock], solution: &ConstraintSolution) -> f64 {
    let impulses = &solution.impulses;
    let mut worst = 0.0_f64;
    let mut first = 0;
    let mut point = 0;
    for block in blocks {
        let slack = block
            .jacobian
            .iter()
            .zip(&block.target)
            .map(|(row, target)| {
                row.iter()
                    .zip(&solution.velocity_change)
                    .map(|(j, v)| j * v)
                    .sum::<f64>()
                    - target
            })
            .collect::<Vec<_>>();
        for (local, bounds) in block.bounds.iter().enumerate() {
            let impulse = impulses[first + local];
            worst = worst
                .max(bounds.minimum - impulse)
                .max(impulse - bounds.maximum);
            if block.contacts.is_empty() {
                if impulse > bounds.minimum {
                    worst = worst.max(slack[local]);
                }
                if impulse < bounds.maximum {
                    worst = worst.max(-slack[local]);
                }
            }
        }
        let mut local = 0;
        for law in &block.contacts {
            let normal = impulses[first + local];
            worst = worst.max(-normal).max(-slack[local]);
            if normal > 0.0 {
                worst = worst.max(slack[local].abs());
            }
            let sliding = solution.sliding.get(point).copied().unwrap_or(law.sliding);
            let coefficient = if sliding {
                law.kinetic_coefficient
            } else {
                law.static_coefficient
            };
            let tangent = impulses[first + local + 1].hypot(impulses[first + local + 2]);
            worst = worst.max(tangent - coefficient * normal);
            if let Some(length) = law.rolling_length {
                let rolling = impulses[first + local + 3].hypot(impulses[first + local + 4]);
                worst = worst.max(rolling - length * normal);
                local += 5;
            } else {
                local += 3;
            }
            point += 1;
        }
        first += block.target.len();
    }
    worst
}
