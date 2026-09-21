//! Powered rotary-contact fixture on production-meshed editable terrain.
//! The guide and motor use bounded external impulses; no designated mining
//! tool or synthetic damage is supplied to the breakage system.

use bevy_math::{DVec3, Vec3};
use mechanic_core::{CompiledCreation, MaterialProperties, RuntimeBox, TICK_SECONDS};
use mechanic_physics::{
    CpuMachine, ExternalImpulse, MachineState, SoftStepConfig, SoftStepTerrain, TerrainContactScene,
};
use mechanic_world::{
    BreakageAccumulator, BrickCoord, ClumpCollection, TerrainField, TerrainMaterial,
    TerrainMeshRequest, TerrainNodeId, TerrainOctree, TerrainTransitionMask, WorldPosition,
    WorldSeed, mesh_chunk,
};
use std::{error::Error, sync::Arc, time::Instant};

#[expect(
    clippy::too_many_lines,
    reason = "keep the benchmark's ordered measurement stages visible together"
)]
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
    let geometry = mechanic_physics::MachineCollisionGeometry::new(&base, 1)?;
    let mut machine = CpuMachine::new(base.clone(), 1, state)?;
    let mut spoil = mechanic_physics::SpoilSolver::default();
    let mut slump = mechanic_world::SpoilSlump::default();
    let mut scene = TerrainContactScene::default();
    scene.publish(1, &[Arc::new(chunk.collision_chunk())], &[])?;
    let mut damage = BreakageAccumulator::default();
    let settings = SoftStepConfig::default();
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
            guide.y = (0.5 - velocities[1] + mechanic_core::STANDARD_GRAVITY_M_S2 * TICK_SECONDS)
                * f64::from(tool_mass);
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
            mechanic_core::GRAVITY,
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
        let snapshot = machine.snapshot();
        let motions = mechanic_physics::MachineKinematics::published_motions(
            &base,
            &snapshot.state.poses,
            &snapshot.state.velocities,
        )?;
        let tool = mechanic_physics::SpoilMachine::new(
            &base,
            &snapshot.state.poses,
            &motions,
            DVec3::ZERO,
        );
        spoil.step(
            &mut clumps,
            &terrain,
            &field,
            &tool,
            mechanic_core::GRAVITY,
            TICK_SECONDS,
        );
        for load in machine.terrain_loads().iter().filter(|load| load.body == 0) {
            damage.accumulate(&terrain, &field, load.breakage_patch(DVec3::ZERO));
        }
        let mut remesh_ms = 0.0;
        let mut publication_ms = 0.0;
        if tick % 6 == 0 {
            if settle && tick > 90 {
                // Nothing more is dug while the spoil settles.
                damage = BreakageAccumulator::default();
            }
            let outcomes = clumps.transfer(
                &mut terrain,
                &field,
                &mut damage,
                &mut slump,
                &mut |cell| tool.keeps_clear(cell.centre().0),
                mechanic_world::TransferLimits::default(),
            );
            if !outcomes.is_empty() {
                let bricks = outcomes
                    .iter()
                    .flat_map(mechanic_world::TerrainEditOutcome::changed_brick_coordinates)
                    .copied()
                    .collect::<Vec<_>>();
                spoil.ground_changed(bricks);
                let remesh_started = Instant::now();
                generation += 1;
                chunk = make_chunk(&terrain, generation);
                remesh_ms = remesh_started.elapsed().as_secs_f64() * 1000.0;
                let publication_started = Instant::now();
                scene.publish(generation, &[Arc::new(chunk.collision_chunk())], &[])?;
                for outcome in &outcomes {
                    removed += outcome.total_removed_cells();
                    deposited += outcome.total_added_cells();
                }
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
        "physics_p95_ms":(!solves.is_empty()).then(|| mechanic_bench::stats::percentile_95(&solves)), "total_tick_p95_ms":(!totals.is_empty()).then(|| mechanic_bench::stats::percentile_95(&totals)),
        "remesh_ms":remesh_total, "publication_ms":publication_total, "material_quanta":clumps.bodies.values().map(|body| u64::from(body.quanta)).sum::<u64>(),
        "kernel_coverage_complete":false})
    );
    if removed == 0 || degraded > 0 || (settle && deposited == 0) {
        return Err("mining replay failed: no extraction or degraded physics".into());
    }
    Ok(())
}
