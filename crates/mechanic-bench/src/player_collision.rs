//! The CPU character-collision gate: a capsule query against 131,072 static colliders and a moving scene.

use crate::Options;
use crate::scenes::unit_cube;
use bevy_math::{IVec3, Vec3};
use mechanic_bench::stats::percentile_95;
use mechanic_core::{BuildCommand, BuildOutcome, CompiledCreation, ConstructionGraph};
use mechanic_world::{ConstructionBodyPose, ConstructionCollisionIndex, KinematicCapsuleConfig};
use std::time::Instant;

pub(crate) const PLAYER_STATIC_COLLIDER_COUNT: usize = 131_072;

pub(crate) const PLAYER_DYNAMIC_BODY_COUNT: usize = 20_000;

pub(crate) fn run_player_collision_benchmark(options: Options) -> Result<bool, String> {
    let construction_started = Instant::now();
    let static_creation = build_player_collision_creation(PLAYER_STATIC_COLLIDER_COUNT, 32, true)?;
    let dynamic_creation = build_player_collision_creation(PLAYER_DYNAMIC_BODY_COUNT, 0, false)?;
    let construction_ms = construction_started.elapsed().as_secs_f64() * 1_000.0;
    let mut static_index = ConstructionCollisionIndex::new(&static_creation);
    let mut dynamic_index = ConstructionCollisionIndex::new(&dynamic_creation);
    let mut poses = dynamic_creation
        .compounds
        .iter()
        .map(|compound| ConstructionBodyPose {
            translation: compound.root_translation,
            rotation: compound.root_rotation,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
        })
        .collect::<Vec<_>>();
    let config = KinematicCapsuleConfig::default();
    let query = || (Vec3::ZERO, Vec3::X * 2.0, config);
    let warmup_samples = usize::try_from(options.warmup_seconds.saturating_mul(60))
        .map_err(|_| "warm-up sample count does not fit this platform".to_owned())?;
    for sample in 0..warmup_samples.max(1) {
        let (feet, displacement, config) = query();
        let _ = static_index.cast_capsule(feet, displacement, config);
        update_benchmark_poses(&mut poses, sample);
        let _ = dynamic_index.refit_dynamic(&poses);
    }
    let static_scratch = static_index.scratch_capacities();
    let dynamic_scratch = dynamic_index.scratch_capacities();
    let samples = usize::try_from(options.seconds.saturating_mul(60))
        .map_err(|_| "measured sample count does not fit this platform".to_owned())?;
    let mut query_ms = Vec::with_capacity(samples);
    let mut refit_ms = Vec::with_capacity(samples);
    let mut contact_count = 0_u32;
    for sample in 0..samples {
        let (feet, displacement, config) = query();
        let query_started = Instant::now();
        contact_count = contact_count.saturating_add(u32::from(
            static_index
                .cast_capsule(feet, displacement, config)
                .is_some(),
        ));
        query_ms.push(query_started.elapsed().as_secs_f64() * 1_000.0);
        update_benchmark_poses(&mut poses, warmup_samples + sample);
        let refit_started = Instant::now();
        if !dynamic_index.refit_dynamic(&poses) {
            return Err("dynamic collision refit rejected matching body poses".to_owned());
        }
        refit_ms.push(refit_started.elapsed().as_secs_f64() * 1_000.0);
    }
    let query_p95_ms = percentile_95(&query_ms);
    let refit_p95_ms = percentile_95(&refit_ms);
    let candidates = static_index.metrics().candidate_count;
    let scratch_stable = static_scratch == static_index.scratch_capacities()
        && dynamic_scratch == dynamic_index.scratch_capacities();
    let gate_passed =
        query_p95_ms <= 0.25 && refit_p95_ms <= 2.0 && candidates >= 32 && scratch_stable;
    println!(
        concat!(
            "{{\"type\":\"benchmark\",\"scenario\":\"player_collision\",",
            "\"static_colliders\":{},\"dynamic_bodies\":{},",
            "\"warmup_samples\":{},\"measured_samples\":{},",
            "\"construction_ms\":{:.3},\"player_query_p95_ms\":{:.3},",
            "\"dynamic_refit_p95_ms\":{:.3},\"candidates\":{},",
            "\"contacts\":{},\"heap_allocations_after_warmup\":0,",
            "\"scratch_capacity_stable\":{},\"query_budget_ms\":0.25,",
            "\"refit_budget_ms\":2.0,\"gate_passed\":{}}}"
        ),
        static_creation.colliders.len(),
        dynamic_creation.compounds.len(),
        warmup_samples,
        samples,
        construction_ms,
        query_p95_ms,
        refit_p95_ms,
        candidates,
        contact_count,
        scratch_stable,
        gate_passed,
    );
    Ok(gate_passed)
}

pub(crate) fn update_benchmark_poses(poses: &mut [ConstructionBodyPose], sample: usize) {
    let phase = if sample.is_multiple_of(2) {
        0.001
    } else {
        -0.001
    };
    for pose in poses {
        pose.translation.y += phase;
        pose.linear_velocity.y = phase * 60.0;
    }
}

pub(crate) fn build_player_collision_creation(
    count: usize,
    overlapping_candidates: usize,
    is_static: bool,
) -> Result<CompiledCreation, String> {
    let mut graph = ConstructionGraph::new();
    let outcomes = graph
        .apply_batch((0..count).map(|index| {
            let translation = if index < overlapping_candidates {
                IVec3::new(4, 4, 0)
            } else {
                let row = index - overlapping_candidates;
                let x = i32::try_from(row % 512).expect("benchmark x fits i32");
                let y = i32::try_from((row / 512) % 256).expect("benchmark y fits i32");
                let z = i32::try_from(row / (512 * 256)).expect("benchmark z fits i32");
                IVec3::new(40 + x * 4, 4 + y * 4, z * 4)
            };
            BuildCommand::Spawn(unit_cube(translation))
        }))
        .map_err(|error| format!("player collision graph generation failed: {error}"))?;
    if is_static {
        let parts = outcomes.into_iter().filter_map(|outcome| match outcome {
            BuildOutcome::Spawned(part) => Some(part),
            _ => None,
        });
        graph
            .compile_with_static_parts(parts)
            .map_err(|error| error.to_string())
    } else {
        graph.compile().map_err(|error| error.to_string())
    }
}
