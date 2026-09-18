//! Repeated fixed-pose costs, with no simulation/publication performance claim.

use super::{CompiledCreation, MachineDynamics, finite_support, support_probes};
use bevy_math::DVec3;
use mechanic_physics::{
    BodyPose, ConstraintSolution, DynamicsFactor, MachineCollisionGeometry, TerrainContactScene,
    solve_constraints,
};
use std::{error::Error, time::Instant};

#[allow(clippy::too_many_lines)] // Matched acquisition, repeatability checks and complete result record.
pub(super) fn run() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let path = arguments.first().ok_or(
        "usage: compiled-response WORLD_INSTANCE.ron [MAX_SWEEPS] [synthetic|finite-terrain] [TOLERANCE] [dense|articulated]",
    )?;
    let maximum = arguments
        .get(1)
        .map_or(Ok(256), |value| value.parse::<usize>())?;
    let finite = match arguments.get(2).map(String::as_str) {
        None | Some("synthetic") => false,
        Some("finite-terrain") => true,
        _ => return Err("support must be synthetic or finite-terrain".into()),
    };
    let tolerance = arguments
        .get(3)
        .map_or(Ok(1e-8), |value| value.parse::<f64>())?;
    let articulated = match arguments.get(4).map(String::as_str) {
        None | Some("dense") => false,
        Some("articulated") => true,
        _ => return Err("factor must be dense or articulated".into()),
    };
    if arguments.len() > 5 {
        return Err("too many experiment arguments".into());
    }
    let source = std::fs::read_to_string(path)?;
    let instance: mechanic_world::WorldCreationInstanceDoc = ron::from_str(&source)?;
    let loaded = instance.creation.into_graph()?;
    let creation = loaded
        .graph
        .compile_with_suspension_sockets([], &loaded.sockets)?;
    let mut roots = MachineDynamics::initial_roots(&creation);
    let surface = if finite {
        Some(finite_support::scene(&creation, &mut roots)?)
    } else {
        None
    };
    let mut durations = Vec::with_capacity(1000);
    let mut hash = None;
    let mut last = None;
    for index in 0..1100 {
        let (timing, solution) = sample(
            &creation,
            &roots,
            surface.as_ref(),
            maximum,
            tolerance,
            articulated,
        )?;
        let current = solution
            .velocity_change
            .iter()
            .chain(&solution.impulses)
            .flat_map(|value| value.to_bits().to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
            });
        if hash.is_some_and(|previous| previous != current) {
            return Err("fixed-pose response changed state hash".into());
        }
        if !solution.converged {
            report_failure(&solution, timing, index, maximum, tolerance);
            return Err(format!("fixed-pose solve failed: residual {}", solution.residual).into());
        }
        hash = Some(current);
        last = Some(solution);
        if index >= 100 {
            durations.push(timing);
        }
    }
    let solution = last.ok_or("no experiment samples")?;
    let p95 = (0..5)
        .map(|stage| {
            let mut values = durations
                .iter()
                .map(|sample| sample[stage])
                .collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            values[949]
        })
        .collect::<Vec<_>>();
    let mut record = serde_json::json!({
        "type": "compiled_response_experiment", "backend": "cpu_f64",
        "factorization": if articulated { "articulated" } else { "dense_reference" },
        "support": if finite { "finite_two_triangle_rock_floor" } else { "synthetic_horizontal_probes" },
        "pose": if finite { "authored_pose_translated_to_1mm_overlap" } else { "authored_bind_pose" },
        "bodies": creation.compounds.len(), "colliders": creation.colliders.len(),
        "generalized_velocities": creation.dynamics.elimination_parent.len(),
        "constraint_rows": solution.impulses.len(), "response_storage_scalars": solution.response_storage,
        "max_iterations": maximum, "tolerance": tolerance, "iterations": solution.iterations, "factor_solves": solution.factor_solves,
        "newton_attempts": solution.newton_attempts, "newton_accepts": solution.newton_accepts,
        "newton_applications": solution.newton_applications, "newton_local_factorizations": solution.newton_local_factorizations,
        "newton_line_searches": solution.newton_line_searches,
        "newton_generalized_factorizations": solution.newton_generalized_factorizations,
        "newton_reduced_storage": solution.newton_reduced_storage,
        "newton_contact_factorizations": solution.newton_contact_factorizations,
        "newton_contact_storage": solution.newton_contact_storage,
        "active_set_trials": solution.active_set_trials,
        "active_set_trial_rejections": solution.active_set_trial_rejections,
        "warm_start_restarts": solution.warm_start_restarts,
        "active_set_trial_matrix_storage": solution.active_set_trial_matrix_storage,
        "warmup_samples": 100, "measured_samples": 1000,
        "stage_order": ["assembly", "factor_and_free_velocity", "collision_and_constraint_rows", "response_preparation_and_solve", "total"],
        "p95_ms": p95, "samples_ms": durations,
        "state_hash": format!("{:016x}", hash.ok_or("no state hash")?),
        "repeatable": true, "converged": solution.converged, "residual": solution.residual,
        "simulated_duration_seconds": 0, "gpu_execution_ms": 0, "transferred_bytes": 0,
        "publication_performed": false, "physics_tick_gate_passed": false
    });
    for (name, value) in [
        ("continuation_trials", solution.continuation_trials),
        ("continuation_iterations", solution.continuation_iterations),
        ("continuation_stages", solution.continuation_stages),
        (
            "continuation_evaluations",
            solution.continuation_evaluations,
        ),
        ("continuation_mixed_rows", solution.continuation_mixed_rows),
        ("continuation_full_rows", solution.continuation_full_rows),
    ] {
        record[name] = serde_json::json!(value);
    }
    record["reduced_newton_point_factor"] = serde_json::json!("analytic_radial_tangent");
    println!("{record}");
    Ok(())
}

fn report_failure(
    solution: &ConstraintSolution,
    timing: [f64; 5],
    sample_index: usize,
    maximum: usize,
    tolerance: f64,
) {
    println!(
        "{}",
        serde_json::json!({
            "type": "compiled_response_experiment_failure", "backend": "cpu_f64",
            "sample_index": sample_index, "during_warmup": sample_index < 100,
            "max_iterations": maximum, "tolerance": tolerance, "residual": solution.residual,
            "constraint_rows": solution.impulses.len(), "iterations": solution.iterations,
            "factor_solves": solution.factor_solves, "newton_attempts": solution.newton_attempts,
            "newton_applications": solution.newton_applications,
            "newton_local_factorizations": solution.newton_local_factorizations,
            "newton_generalized_factorizations": solution.newton_generalized_factorizations,
            "newton_reduced_storage": solution.newton_reduced_storage,
            "newton_contact_factorizations": solution.newton_contact_factorizations,
            "newton_contact_storage": solution.newton_contact_storage,
            "continuation": { "trials": solution.continuation_trials,
                "iterations": solution.continuation_iterations, "stages": solution.continuation_stages,
                "evaluations": solution.continuation_evaluations, "mixed_rows": solution.continuation_mixed_rows,
                "full_rows": solution.continuation_full_rows },
            "active_set_trials": solution.active_set_trials,
            "active_set_trial_rejections": solution.active_set_trial_rejections,
            "warm_start_restarts": solution.warm_start_restarts,
            "active_set_trial_matrix_storage": solution.active_set_trial_matrix_storage,
            "newton_line_searches": solution.newton_line_searches, "sample_ms": timing,
            "p95_ms": null, "converged": false, "publication_performed": false,
            "physics_tick_gate_passed": false
        })
    );
}

fn sample(
    creation: &CompiledCreation,
    roots: &[BodyPose],
    surface: Option<&(MachineCollisionGeometry, TerrainContactScene)>,
    maximum: usize,
    tolerance: f64,
    articulated: bool,
) -> Result<([f64; 5], ConstraintSolution), Box<dyn Error>> {
    let started = Instant::now();
    let model = MachineDynamics::assemble(
        creation,
        roots,
        &vec![0.0; creation.dynamics.coordinate_bearings.len()],
    )?;
    let assembled = Instant::now();
    let diagonal = vec![0.0; creation.dynamics.elimination_parent.len()];
    let factor = if articulated {
        DynamicsFactor::articulated(
            creation,
            roots,
            &vec![0.0; creation.dynamics.coordinate_bearings.len()],
            &diagonal,
        )?
    } else {
        model.factor(&diagonal)?
    };
    let mut incoming = model.gravity_force(
        creation,
        mechanic_core::GRAVITY / f64::from(mechanic_core::TICK_RATE_HZ),
    )?;
    factor.solve(&mut incoming)?;
    let factored = Instant::now();
    let blocks = if let Some((geometry, terrain)) = surface {
        let root = creation
            .dynamics
            .body_velocities
            .iter()
            .find(|rows| rows.len() == 6)
            .ok_or("finite-floor experiment requires a floating root")?;
        incoming[root.start] = 0.3;
        terrain
            .contacts(geometry, &model.poses, DVec3::ZERO)?
            .impact_constraints(&model, &incoming, 1.0, 1e-7)?
            .blocks
    } else {
        support_probes(creation, &model, &incoming)?
    };
    let collided = Instant::now();
    let solution = solve_constraints(&factor, &blocks, maximum, tolerance)?;
    let solved = Instant::now();
    Ok((
        [
            (assembled - started).as_secs_f64() * 1000.0,
            (factored - assembled).as_secs_f64() * 1000.0,
            (collided - factored).as_secs_f64() * 1000.0,
            (solved - collided).as_secs_f64() * 1000.0,
            (solved - started).as_secs_f64() * 1000.0,
        ],
        solution,
    ))
}
