//! Reproducible CPU loose-material replay. JSONL includes actual body poses so
//! the accompanying browser viewer can replay measured physics.

use bevy_math::{DQuat, DVec3};
use mechanic_core::CompiledCreation;
use mechanic_physics::{
    CpuMachine, MachineState, PreparedClumpBodies, SoftStepSettings, SoftStepTerrain,
    TerrainContactScene,
};
use mechanic_world::{
    ClumpCollection, MaterialClump, TerrainCollisionChunk, TerrainMaterial, TerrainNodeId,
    TerrainTriangleGroupMask, TriangleBvh, TriangleBvhNode, TriangleBvhTriangle, WorldBounds,
    WorldPosition,
};
use std::{error::Error, sync::Arc, time::Instant};

#[allow(clippy::too_many_lines)] // One explicit fixture, replay and result record.
fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let ticks: u32 = args
        .iter()
        .position(|arg| arg == "--ticks")
        .and_then(|i| args.get(i + 1))
        .map_or(Ok(360), |value| value.parse())?;
    let trace = args.iter().any(|arg| arg == "--trace");
    let base = CompiledCreation::default();
    let mut clumps = ClumpCollection::default();
    for id in 1..=256_u32 {
        let index = id - 1;
        let body = MaterialClump {
            id: u64::from(id),
            material: TerrainMaterial::ALL[index as usize % TerrainMaterial::COUNT],
            quanta: 510 * 8,
            half_extents: DVec3::splat(0.05),
            position: WorldPosition(DVec3::new(
                f64::from(index % 8) * 0.115 - 0.4,
                0.15 + f64::from(index / 64) * 0.16,
                f64::from((index / 8) % 8) * 0.115 - 0.4,
            )),
            rotation: DQuat::IDENTITY,
            linear_velocity: DVec3::ZERO,
            angular_velocity: DVec3::ZERO,
            settled_seconds: 0.0,
            sleeping: false,
        };
        clumps.bodies.insert(body.id, body);
    }
    clumps.next_id = 257;
    let quantity: u64 = clumps
        .bodies
        .values()
        .map(|body| u64::from(body.quanta))
        .sum();
    let prepared = PreparedClumpBodies::new(
        &base,
        &MachineState::at_rest(&base),
        &clumps,
        DVec3::ZERO,
        1,
    )?;
    let mut machine = CpuMachine::new(prepared.creation, 1, prepared.state)?;
    let mut scene = TerrainContactScene::default();
    scene.publish(1, &[floor()], &[])?;
    let settings = SoftStepSettings::default();
    let mut timings = Vec::new();
    let mut degraded = 0_u32;
    let mut minimum_y = f64::INFINITY;
    for tick in 0..ticks {
        let started = Instant::now();
        machine.step(
            mechanic_core::GRAVITY,
            &settings,
            &[],
            &[],
            Some(SoftStepTerrain {
                scene: &scene,
                geometry: &prepared.geometry,
                topology_generation: 1,
                origin: DVec3::ZERO,
            }),
        )?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        timings.push(elapsed_ms);
        degraded += u32::from(machine.diagnostics().degraded);
        for pose in &machine.snapshot().state.poses {
            minimum_y = minimum_y.min(pose.position.y);
        }
        let mut record = serde_json::json!({ "scenario":"material-clumps", "tick":tick, "bodies":256,
            "material_quanta":quantity, "physics_ms":elapsed_ms, "total_tick_ms":elapsed_ms,
            "contacts":machine.diagnostics().contacts, "degraded":machine.diagnostics().degraded,
            "remesh_ms":0, "publication_ms":0, "kernel_coverage_complete":false });
        if trace && tick % 3 == 0 {
            record["poses"] = serde_json::json!(
                machine
                    .snapshot()
                    .state
                    .poses
                    .iter()
                    .map(|pose| [
                        pose.position.x,
                        pose.position.y,
                        pose.position.z,
                        pose.rotation.x,
                        pose.rotation.y,
                        pose.rotation.z,
                        pose.rotation.w
                    ])
                    .collect::<Vec<_>>()
            );
        }
        println!("{record}");
    }
    timings.sort_by(f64::total_cmp);
    let p95 = timings
        .get(timings.len().saturating_mul(95) / 100)
        .copied()
        .unwrap_or_default();
    println!(
        "{}",
        serde_json::json!({"scenario":"material-clumps-summary", "bodies":256, "ticks":ticks,
        "physics_p95_ms":p95, "total_tick_p95_ms":p95, "degraded_ticks":degraded,
        "minimum_centre_y":minimum_y, "material_quanta":quantity, "kernel_coverage_complete":false,
        "scope":"CPU clump contacts; all bodies remain active; extraction, deposition and app remeshing measured separately"})
    );
    if degraded > 0 || minimum_y < 0.035 {
        return Err("clump replay failed its collision/solver check".into());
    }
    Ok(())
}

fn floor() -> Arc<TerrainCollisionChunk> {
    let bounds = WorldBounds {
        minimum: WorldPosition(DVec3::new(-16.0, 0.0, -16.0)),
        maximum: WorldPosition(DVec3::new(16.0, 0.0, 16.0)),
    };
    let mut weights = [0.0; TerrainMaterial::COUNT];
    weights[TerrainMaterial::Rock.code() as usize] = 1.0;
    let triangles = [[0, 1, 2], [0, 2, 3]]
        .into_iter()
        .map(|indices| TriangleBvhTriangle {
            indices,
            group_mask: TerrainTriangleGroupMask::REGULAR,
        })
        .collect();
    Arc::new(TerrainCollisionChunk {
        node: TerrainNodeId::ROOT,
        generation: 1,
        vertices: vec![
            [-16.0, 0.0, -16.0],
            [-16.0, 0.0, 16.0],
            [16.0, 0.0, 16.0],
            [16.0, 0.0, -16.0],
        ],
        indices: vec![0, 1, 2, 0, 2, 3],
        material_weights: vec![weights; 4],
        bounds,
        triangle_bvh: TriangleBvh {
            bounds,
            triangles,
            nodes: vec![TriangleBvhNode {
                bounds,
                first_triangle: 0,
                triangle_count: 2,
                group_mask: TerrainTriangleGroupMask::REGULAR,
                ..Default::default()
            }],
        },
        active_groups: TerrainTriangleGroupMask::REGULAR,
        ..Default::default()
    })
}
