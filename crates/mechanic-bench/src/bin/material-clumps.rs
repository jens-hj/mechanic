//! Reproducible loose-material replay: clumps of every material poured onto
//! generated ground, all awake. JSONL includes poses for the replay viewer.

use bevy_math::{DQuat, DVec3};
use mechanic_physics::{SpoilMachine, SpoilSolver, spoil_radius};
use mechanic_world::{
    ClumpCollection, MATERIAL_QUANTUM_M3, MaterialClump, TerrainField, TerrainMaterial,
    TerrainOctree, WorldPosition, WorldSeed,
};
use std::{error::Error, time::Instant};

#[expect(
    clippy::too_many_lines,
    reason = "one explicit fixture, replay and result record"
)]
fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let value = |name: &str| {
        args.iter()
            .position(|arg| arg == name)
            .and_then(|index| args.get(index + 1))
    };
    let ticks: u32 = value("--ticks").map_or(Ok(360), |value| value.parse())?;
    let bodies: u32 = value("--bodies").map_or(Ok(256), |value| value.parse())?;
    let trace = args.iter().any(|arg| arg == "--trace");
    let field = TerrainField::new(WorldSeed(84));
    let terrain = TerrainOctree::default();
    let spawn = field.safe_spawn().0;
    let surface = field.surface_height(spawn.x, spawn.z);
    let mut clumps = ClumpCollection::default();
    for id in 1..=bodies {
        let index = id - 1;
        let quanta = 510 * (1 + index % 8);
        let body = MaterialClump {
            id: u64::from(id),
            material: TerrainMaterial::ALL[index as usize % TerrainMaterial::COUNT],
            quanta,
            half_extents: DVec3::splat((f64::from(quanta) * MATERIAL_QUANTUM_M3).cbrt() * 0.5),
            position: WorldPosition(DVec3::new(
                spawn.x + f64::from(index % 8) * 0.13 - 0.45,
                surface + 0.3 + f64::from(index / 64) * 0.16,
                spawn.z + f64::from((index / 8) % 8) * 0.13 - 0.45,
            )),
            rotation: DQuat::IDENTITY,
            linear_velocity: DVec3::ZERO,
            angular_velocity: DVec3::ZERO,
            settled_seconds: 0.0,
            sleeping: false,
        };
        clumps.bodies.insert(body.id, body);
    }
    clumps.next_id = u64::from(bodies) + 1;
    let quantity: u64 = clumps
        .bodies
        .values()
        .map(|body| u64::from(body.quanta))
        .sum();
    let machine = SpoilMachine::default();
    let mut solver = SpoilSolver::default();
    let mut timings = Vec::new();
    let mut awake = 0;
    for tick in 0..ticks {
        // Every body stays awake: the measured case is the worst one.
        for body in clumps.bodies.values_mut() {
            body.sleeping = false;
        }
        let started = Instant::now();
        let step = solver.step(
            &mut clumps,
            &terrain,
            &field,
            &machine,
            mechanic_core::GRAVITY,
            mechanic_core::TICK_SECONDS,
        );
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        timings.push(elapsed_ms);
        awake = step.awake;
        let mut record = serde_json::json!({ "scenario":"material-clumps", "tick":tick, "bodies":bodies,
            "awake":step.awake, "material_quanta":quantity, "physics_ms":elapsed_ms,
            "total_tick_ms":elapsed_ms, "kernel_coverage_complete":false });
        if trace && tick % 3 == 0 {
            record["poses"] = serde_json::json!(
                clumps
                    .bodies
                    .values()
                    .map(|body| [
                        body.position.0.x - spawn.x,
                        body.position.0.y - surface,
                        body.position.0.z - spawn.z,
                        body.rotation.x,
                        body.rotation.y,
                        body.rotation.z,
                        body.rotation.w
                    ])
                    .collect::<Vec<_>>()
            );
        }
        println!("{record}");
    }
    let sunk = clumps
        .bodies
        .values()
        .filter(|body| {
            body.position.0.y + spoil_radius(body.quanta)
                < field.surface_height(body.position.0.x, body.position.0.z)
        })
        .count();
    let kept: u64 = clumps
        .bodies
        .values()
        .map(|body| u64::from(body.quanta))
        .sum();
    let p95 = mechanic_bench::stats::percentile_95_or_zero(&timings);
    println!(
        "{}",
        serde_json::json!({"scenario":"material-clumps-summary", "bodies":bodies, "awake":awake, "ticks":ticks,
        "physics_p95_ms":p95, "total_tick_p95_ms":p95, "sunk":sunk,
        "material_quanta":kept, "kernel_coverage_complete":false,
        "scope":"spoil solver on generated ground; all bodies remain awake; extraction, deposition and app remeshing measured separately"})
    );
    if sunk > 0 || kept != quantity {
        return Err("clump replay lost material or let it under the ground".into());
    }
    Ok(())
}
