//! Matched pose-to-factor and repeated inverse-dynamics costs, not a tick gate.

use mechanic_core::CompiledCreation;
use mechanic_physics::{BodyPose, DynamicsFactor, MachineDynamics};
use std::{error::Error, hint::black_box, time::Instant};

const WARMUP: usize = 100;
const SAMPLES: usize = 1000;
const IMPULSES: usize = 32;

type Timing = [f64; 3];

struct Sample {
    timing: Timing,
    values: Vec<Vec<f64>>,
}

struct Experiment {
    creation: CompiledCreation,
    roots: Vec<BodyPose>,
    coordinates: Vec<f64>,
    diagonal: Vec<f64>,
    impulses: Vec<Vec<f64>>,
    reference: Vec<Vec<f64>>,
}

impl Experiment {
    fn new(creation: CompiledCreation) -> Result<Self, Box<dyn Error>> {
        let roots = MachineDynamics::initial_roots(&creation);
        let coordinates = vec![0.0; creation.dynamics.coordinate_bearings.len()];
        let size = creation.dynamics.elimination_parent.len();
        if size == 0 {
            return Err("experiment requires dynamic coordinates".into());
        }
        let diagonal = vec![0.01; size];
        let model = MachineDynamics::assemble(&creation, &roots, &coordinates)?;
        let factor = model.factor(&diagonal)?;
        let impulses: Vec<_> = (0..IMPULSES)
            .map(|index| {
                let mut rhs = vec![0.0; size];
                rhs[index * (size - 1) / (IMPULSES - 1)] = 1.0;
                rhs
            })
            .collect();
        let mut reference = impulses.clone();
        for rhs in &mut reference {
            factor.solve(rhs)?;
        }
        Ok(Self {
            creation,
            roots,
            coordinates,
            diagonal,
            impulses,
            reference,
        })
    }

    fn sample(&self, articulated: bool) -> Result<Sample, Box<dyn Error>> {
        let started = Instant::now();
        let factor = if articulated {
            DynamicsFactor::articulated(
                black_box(&self.creation),
                &self.roots,
                &self.coordinates,
                &self.diagonal,
            )?
        } else {
            MachineDynamics::assemble(black_box(&self.creation), &self.roots, &self.coordinates)?
                .factor(&self.diagonal)?
        };
        let factored = Instant::now();
        let mut values = self.impulses.clone();
        for rhs in &mut values {
            factor.solve(black_box(rhs))?;
        }
        let solved = Instant::now();
        Ok(Sample {
            timing: [
                (factored - started).as_secs_f64() * 1000.0,
                (solved - factored).as_secs_f64() * 1000.0,
                (solved - started).as_secs_f64() * 1000.0,
            ],
            values,
        })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.len() != 2 {
        return Err(
            "usage: compiled-inertia WORLD_INSTANCE.ron|chain_256 dense|articulated".into(),
        );
    }
    let articulated = match arguments[1].as_str() {
        "dense" => false,
        "articulated" => true,
        _ => return Err("factor must be dense or articulated".into()),
    };
    let creation = if arguments[0] == "chain_256" {
        mechanic_bench::scenarios::build_bearing_chain(256)?
    } else {
        let source = std::fs::read_to_string(&arguments[0])?;
        let doc: mechanic_world::WorldCreationInstanceDoc = ron::from_str(&source)?;
        let loaded = doc.creation.into_graph()?;
        loaded
            .graph
            .compile_with_suspension_sockets([], &loaded.sockets)?
    };
    measure(&Experiment::new(creation)?, &arguments[0], articulated)
}

fn measure(
    experiment: &Experiment,
    fixture: &str,
    articulated: bool,
) -> Result<(), Box<dyn Error>> {
    let mut samples = Vec::with_capacity(SAMPLES);
    let mut hash = None;
    let mut maximum_relative_error = 0.0_f64;
    for index in 0..WARMUP + SAMPLES {
        let Sample { timing, values } = experiment.sample(articulated)?;
        for (actual, expected) in values.iter().zip(&experiment.reference) {
            let scale = expected.iter().map(|v| v.abs()).fold(1.0_f64, f64::max);
            let error = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
                / scale;
            if !error.is_finite() || error > 2e-8 {
                return Err(format!("inverse response mismatch: {error:e}").into());
            }
            maximum_relative_error = maximum_relative_error.max(error);
        }
        let current = values
            .iter()
            .flatten()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325_u64, |h, byte| {
                (h ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
            });
        if hash.is_some_and(|previous| previous != current) {
            return Err("inverse response changed state hash".into());
        }
        hash = Some(current);
        if index >= WARMUP {
            samples.push(timing);
        }
    }
    let p95: Vec<_> = (0..3)
        .map(|stage| {
            let mut values: Vec<_> = samples.iter().map(|sample| sample[stage]).collect();
            values.sort_by(f64::total_cmp);
            values[(SAMPLES * 95).div_ceil(100) - 1]
        })
        .collect();
    println!(
        "{}",
        serde_json::json!({
            "type": "compiled_inertia_experiment", "fixture": fixture,
            "backend": "cpu_f64", "factorization": if articulated { "articulated" } else { "dense_reference" },
            "bodies": experiment.creation.compounds.len(), "generalized_velocities": experiment.diagonal.len(),
            "implicit_diagonal": 0.01, "impulses_per_factor": IMPULSES,
            "warmup_samples": WARMUP, "measured_samples": SAMPLES,
            "stage_order": ["pose_to_factor", "32_impulses_including_rhs_copy_and_scratch", "total"],
            "samples_ms": samples, "p95_ms": p95,
            "reference_max_relative_error": maximum_relative_error,
            "state_hash": format!("{:016x}", hash.ok_or("no samples")?), "repeatable": true,
            "simulated_duration_seconds": 0, "collision_performed": false, "publication_performed": false,
            "gpu_execution_ms": 0, "transferred_bytes": 0, "physics_tick_gate_passed": false
        })
    );
    Ok(())
}
