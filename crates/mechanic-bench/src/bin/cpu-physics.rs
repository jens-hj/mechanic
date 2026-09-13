//! CPU physics quality report. Prints JSONL and always exits successfully:
//! solver quality is tracked here, not gated in `cargo test`.

use std::{error::Error, fs, path::Path, time::Instant};

use mechanic_physics::{
    ConstraintBlock, ConstraintSolution, ContactFriction, DynamicsFactor, ImpulseBounds,
    PreparedConstraints, solve_constraints,
};
use serde_json::json;

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
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--scenario" => scenario = args.next().ok_or("--scenario needs a value")?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    match scenario.as_str() {
        "reference-fixtures" => reference_fixtures(),
        other => Err(format!("unknown scenario {other}").into()),
    }
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
