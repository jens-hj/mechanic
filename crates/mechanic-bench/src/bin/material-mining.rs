//! Powered rotary-contact fixture on production-meshed editable terrain.
//! The guide and motor use bounded external impulses; no designated mining
//! tool or synthetic damage is supplied to the breakage system.

use bevy_math::{DVec3, Vec3};
use mechanic_core::{CompiledCreation, MaterialProperties, RuntimeBox};
use mechanic_physics::{
    CpuMachine, ExternalImpulse, MachineState, PreparedClumpBodies, SoftStepSettings,
    SoftStepTerrain, TICK_SECONDS, TerrainContactScene,
};
use mechanic_world::{
    BreakageAccumulator, BreakagePatch, BrickCoord, ClumpCollection, TerrainField, TerrainMaterial,
    TerrainMeshRequest, TerrainNodeId, TerrainOctree, TerrainTransitionMask, WorldPosition,
    WorldSeed, mesh_chunk,
};
use std::{error::Error, sync::Arc, time::Instant};

#[allow(clippy::too_many_lines)] // Keep the benchmark's ordered measurement stages visible together.
fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let name = args.get(1).map_or("rock", String::as_str);
    let material = match name {
        "sand" => TerrainMaterial::Sand,
        "soil" => TerrainMaterial::Soil,
        "rock" => TerrainMaterial::Rock,
        "iron" => TerrainMaterial::Iron,
        "graphite" => TerrainMaterial::Graphite,
        "cover" => TerrainMaterial::SurfaceCover,
        _ => return Err("material must be sand, soil, rock, iron, graphite or cover".into()),
    };
    let ticks: u32 = args.get(2).map_or(Ok(180), |value| value.parse())?;
    let settle = args.iter().any(|arg| arg == "--settle");
    let centre = DVec3::new(0.8, 200.8, 0.8);
    let field = TerrainField::new(WorldSeed(84));
    let mut terrain = TerrainOctree::default();
    terrain.add_sphere(&field, WorldPosition(centre), 0.55, material)?;
    let node = TerrainNodeId::leaf(BrickCoord::new(0, 125, 0));
    let make_chunk = |terrain: &TerrainOctree, generation| {
        mesh_chunk(
            &field,
            &terrain.snapshot(),
            TerrainMeshRequest {
                node,
                generation,
                transition_mask: TerrainTransitionMask::NONE,
            },
        )
    };
    let mut chunk = make_chunk(&terrain, 1);
    let surface = material.surface_response();
    // Soft ground uses a light cutter; the mineral fixture supplies the much
    // higher feed load and motor torque needed by an industrial drilling rig.
    let tool_mass = match material {
        TerrainMaterial::Iron => 20_000.0_f32,
        TerrainMaterial::Graphite => 1_600.0,
        TerrainMaterial::Rock => 8_000.0,
        _ => 80.0,
    };
    let base = CompiledCreation::default().with_runtime_boxes(&[RuntimeBox {
        half_extents: Vec3::new(0.025, 0.05, 0.1),
        mass: tool_mass,
        material: MaterialProperties {
            density_kg_m3: 7_800.0,
            static_friction: surface.static_friction,
            dynamic_friction: surface.dynamic_friction,
            restitution: 0.0,
            rolling_resistance: 0.0,
            youngs_modulus_pa: 2e11,
        },
    }])?;
    let mut state = MachineState::at_rest(&base);
    state.poses[0].position = centre + DVec3::Y * 0.602;
    let mut clumps = ClumpCollection::default();
    let prepared = PreparedClumpBodies::new(&base, &state, &clumps, DVec3::ZERO, 1)?;
    let mut geometry = prepared.geometry;
    let mut machine = CpuMachine::new(prepared.creation, 1, prepared.state)?;
    let mut scene = TerrainContactScene::default();
    scene.publish(1, &[Arc::new(chunk.collision_chunk())], &[])?;
    let mut damage = BreakageAccumulator::default();
    let settings = SoftStepSettings::default();
    let mut generation = 1;
    let mut removed = 0_u64;
    let mut deposited = 0_u64;
    let mut degraded = 0_u32;
    let mut totals = Vec::new();
    let mut solves = Vec::new();
    let mut remesh_total = 0.0;
    let mut publication_total = 0.0;
    for tick in 1..=ticks {
        let started = Instant::now();
        let snapshot = machine.snapshot();
        let pose = snapshot.state.poses[0];
        let velocities = &snapshot.state.velocities;
        let maximum_torque = f64::from(tool_mass) * 0.5;
        let mut torque = ((3.0 - velocities[4]) * f64::from(tool_mass) * 0.25)
            .clamp(-maximum_torque, maximum_torque);
        if settle && tick > 90 {
            torque = 0.0;
        }
        let spin = DVec3::Z * (-torque * TICK_SECONDS / 0.05);
        let mut guide = DVec3::new(
            (-velocities[0] - (pose.position.x - centre.x) * 10.0) * f64::from(tool_mass),
            0.0,
            (-velocities[2] - (pose.position.z - centre.z) * 10.0) * f64::from(tool_mass),
        )
        .clamp_length_max(f64::from(tool_mass) * 0.25);
        if settle && tick > 90 {
            guide.y = (0.5 - velocities[1] + 9.81 * TICK_SECONDS) * f64::from(tool_mass);
        }
        let command = |point, impulse| ExternalImpulse {
            tick: snapshot.tick + 1,
            topology_generation: 1,
            body: 0,
            point,
            impulse,
        };
        let impulses = [
            command(pose.position + DVec3::X * 0.025, spin),
            command(pose.position - DVec3::X * 0.025, -spin),
            command(pose.position, guide),
        ];
        let solve_started = Instant::now();
        machine.step(
            -DVec3::Y * 9.81,
            &settings,
            &impulses,
            &[],
            Some(SoftStepTerrain {
                scene: &scene,
                geometry: &geometry,
                topology_generation: 1,
                origin: DVec3::ZERO,
            }),
        )?;
        let solve_ms = solve_started.elapsed().as_secs_f64() * 1000.0;
        solves.push(solve_ms);
        degraded += u32::from(machine.diagnostics().degraded);
        for (offset, body) in clumps.bodies.values_mut().enumerate() {
            let row = offset + 1;
            let state = &machine.snapshot().state;
            body.position = WorldPosition(state.poses[row].position);
            body.rotation = state.poses[row].rotation;
            let first = 6 + offset * 6;
            body.linear_velocity = DVec3::from_slice(&state.velocities[first..first + 3]);
            body.angular_velocity = DVec3::from_slice(&state.velocities[first + 3..first + 6]);
            body.update_settling(
                machine.terrain_loads().iter().any(|load| {
                    load.body == row && load.normal.y > 0.25 && load.normal_impulse > 0.0
                }),
                TICK_SECONDS,
            );
        }
        for load in machine.terrain_loads().iter().filter(|load| load.body == 0) {
            damage.accumulate(
                &terrain,
                &field,
                BreakagePatch {
                    centre: WorldPosition(load.point),
                    normal: load.normal,
                    radius: load.patch_radius,
                    stress_pa: load.footprint_impulse
                        / (TICK_SECONDS * std::f64::consts::PI * load.patch_radius.powi(2)),
                    work_j: load.work_j,
                },
            );
        }
        let mut remesh_ms = 0.0;
        let mut publication_ms = 0.0;
        if tick % 6 == 0 {
            let mut sources = Vec::new();
            let mut transfer = None;
            for body in clumps.bodies.values().filter(|body| body.can_deposit()) {
                let cell = body.position.cell()?;
                let mut targets = Vec::new();
                for y in -3..=2 {
                    for z in -2..=2 {
                        for x in -2..=2 {
                            targets.push(mechanic_world::WorldCell::new(
                                cell.x + x,
                                cell.y + y,
                                cell.z + z,
                            ));
                        }
                    }
                }
                transfer = clumps.prepare_deposition(&terrain, &field, body.id, &targets);
                if transfer.is_some() {
                    break;
                }
            }
            if transfer.is_none() && (!settle || tick <= 90) {
                sources = damage.ready(&terrain, &field, clumps.available());
                if !sources.is_empty() {
                    transfer = clumps.prepare_extraction(&terrain, &field, &sources);
                }
            }
            if let Some(transfer) = transfer {
                let remesh_started = Instant::now();
                generation += 1;
                chunk = make_chunk(&transfer.terrain, generation);
                remesh_ms = remesh_started.elapsed().as_secs_f64() * 1000.0;
                let publication_started = Instant::now();
                let prepared = PreparedClumpBodies::new(
                    &base,
                    &machine.snapshot().state,
                    &transfer.clumps,
                    DVec3::ZERO,
                    1,
                )?;
                let mut next_scene = TerrainContactScene::default();
                next_scene.publish(generation, &[Arc::new(chunk.collision_chunk())], &[])?;
                machine.replace_bodies(prepared.creation, prepared.state, 1)?;
                geometry = prepared.geometry;
                scene = next_scene;
                terrain = transfer.terrain;
                clumps = transfer.clumps;
                removed += transfer.outcome.total_removed_cells();
                deposited += transfer.outcome.total_added_cells();
                damage.committed(&sources);
                damage.discard_stale(&terrain, &field);
                publication_ms = publication_started.elapsed().as_secs_f64() * 1000.0;
            }
        }
        remesh_total += remesh_ms;
        publication_total += publication_ms;
        let total_ms = started.elapsed().as_secs_f64() * 1000.0;
        totals.push(total_ms);
        let mut record = serde_json::json!({"scenario":"material-mining", "material":name, "tick":tick,
            "physics_ms":solve_ms, "total_tick_ms":total_ms, "remesh_ms":remesh_ms, "publication_ms":publication_ms,
            "removed_cells":removed, "deposited_cells":deposited, "clumps":clumps.bodies.len(), "degraded":machine.diagnostics().degraded,
            "degraded_reason":machine.diagnostics().degraded_reason,
            "tool_position":machine.snapshot().state.poses[0].position.to_array(),
            "tool_rotation":machine.snapshot().state.poses[0].rotation.to_array(), "motor_torque_nm":torque, "tool_mass_kg":tool_mass,
            "poses":clumps.bodies.values().map(|body| [body.position.0.x, body.position.0.y - 200.0, body.position.0.z, body.rotation.x, body.rotation.y, body.rotation.z, body.rotation.w, body.half_extents.x, body.half_extents.y, body.half_extents.z]).collect::<Vec<_>>(),
            "kernel_coverage_complete":false});
        if tick == 1 || remesh_ms > 0.0 {
            record["terrain"] = serde_json::json!({"origin":[chunk.origin.0.x, chunk.origin.0.y - 200.0, chunk.origin.0.z], "vertices":chunk.vertices, "indices":chunk.index_groups.regular});
        }
        println!("{record}");
    }
    solves.sort_by(f64::total_cmp);
    totals.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({"scenario":"material-mining-summary", "material":name, "ticks":ticks,
        "removed_cells":removed, "deposited_cells":deposited, "clumps":clumps.bodies.len(), "degraded_ticks":degraded,
        "physics_p95_ms":solves.get(solves.len() * 95 / 100), "total_tick_p95_ms":totals.get(totals.len() * 95 / 100),
        "remesh_ms":remesh_total, "publication_ms":publication_total, "material_quanta":clumps.bodies.values().map(|body| u64::from(body.quanta)).sum::<u64>(),
        "kernel_coverage_complete":false})
    );
    if removed == 0 || degraded > 0 || (settle && deposited == 0) {
        return Err("mining replay failed: no extraction or degraded physics".into());
    }
    Ok(())
}
