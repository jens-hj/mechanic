//! CPU physics quality report. Prints JSONL and always exits successfully:
//! solver quality is tracked here, not gated in `cargo test`.

use std::{error::Error, fs, path::Path, time::Instant};

use bevy_math::{DVec3, IVec3};
use mechanic_core::{
    BuildCommand, BuildPose, CompiledCreation, ConstructionGraph, ContactPolytope, CuboidSpec,
    DriveMode, GridRotation,
};
use mechanic_physics::{
    ConstraintBlock, ConstraintSolution, ContactFriction, CpuMachine, DriveCommand, DynamicsFactor,
    ImpulseBounds, MachineState, PreparedConstraints, SoftStepSettings, SoftStepTerrain,
    solve_constraints,
};
use serde_json::json;

#[path = "compiled-response/finite_support.rs"]
mod finite_support;

const ITERATIONS: usize = 256;
const TOLERANCE: f64 = 1e-9;
const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);

type RecordedBlock = (
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<(f64, f64)>,
    Vec<(f64, f64, bool, Option<f64>)>,
);

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut scenario = "reference-fixtures".to_owned();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--scenario" => scenario = args.next().ok_or("--scenario needs a value")?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    match scenario.as_str() {
        "reference-fixtures" => reference_fixtures(),
        "car-drop" => car(false),
        "car-drive" => car(true),
        "block-pile" => block_pile(),
        other => Err(format!(
            "unknown scenario {other}; expected reference-fixtures, car-drop, car-drive or block-pile"
        )
        .into()),
    }
}

// The saved car dropped at 4 m/s, or settled and then driven on its speed drives.
fn car(drive: bool) -> Result<(), Box<dyn Error>> {
    let instance: mechanic_world::WorldCreationInstanceDoc =
        ron::from_str(include_str!("../../tests/fixtures/driven_car_instance.ron"))?;
    let loaded = instance.creation.into_graph()?;
    let creation = loaded
        .graph
        .compile_with_suspension_sockets([], &loaded.sockets)?;
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
        "deepest_m": deepest,
        "settled_m": settled,
        "degraded_ticks": degraded,
        "degraded_reasons": reasons,
        "travelled_m": (state.poses[0].position - start).with_y(0.0).length(),
        "fastest_final": state.velocities.iter().fold(0.0_f64, |m, v| m.max(v.abs())),
    });
    println!("{record}");
    Ok(())
}

// Lowest collider point above the floor; negative values are penetration.
fn lowest_point(creation: &CompiledCreation, state: &MachineState) -> Result<f64, Box<dyn Error>> {
    let mut lowest = f64::INFINITY;
    for collider in &creation.colliders {
        let pose = state.poses[collider.compound_index as usize];
        lowest = lowest.min(
            ContactPolytope::from_collider(collider)?
                .transformed(pose.position, pose.rotation)?
                .bounds()[0]
                .y,
        );
    }
    Ok(lowest)
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
