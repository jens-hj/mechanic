//! Deterministic collision-cost comparisons with identical geometry at each speed.
use super::{
    BuildCommand, BuildPose, ConstructionGraph, ContactPolytope, CpuMachine, CuboidSpec, DVec3,
    DriveCommand, DriveMode, Error, GRAVITY, GridRotation, IVec3, Instant, MachineState,
    SoftStepSettings, SoftStepTerrain, finite_support, json, scale,
};
use bevy_math::Vec3;
use mechanic_core::{BearingSpec, BuildOutcome, FaceKind, FaceRef, PartId, WeldSpec};

fn spawn(graph: &mut ConstructionGraph, ticks: IVec3) -> Result<PartId, Box<dyn Error>> {
    let BuildOutcome::Spawned(id) = graph.apply(BuildCommand::Spawn(CuboidSpec::new(
        [4, 4, 4],
        BuildPose::from_position_ticks(ticks, GridRotation::default()),
    )?))?
    else {
        unreachable!()
    };
    Ok(id)
}

#[allow(clippy::too_many_lines)] // Keep the paired fixture and measurement protocol together.
pub(super) fn run(options: &scale::Options) -> Result<(), Box<dyn Error>> {
    let settings = SoftStepSettings::default();
    for rotor in [true, false] {
        for unrelated in [0, 32] {
            let mut graph = ConstructionGraph::new();
            let root = spawn(&mut graph, IVec3::ZERO)?;
            let mut fixed = Vec::new();
            if rotor {
                let base = spawn(&mut graph, IVec3::NEG_X * 400)?;
                graph.apply(BuildCommand::Weld(WeldSpec {
                    first: FaceRef::part(base, FaceKind::PositiveX),
                    second: FaceRef::part(root, FaceKind::NegativeX),
                }))?;
                let tip = spawn(&mut graph, IVec3::X * 400)?;
                graph.apply(BuildCommand::AddBearing(BearingSpec::new(
                    FaceRef::part(root, FaceKind::PositiveX),
                    FaceRef::part(tip, FaceKind::NegativeX),
                    Vec3::X * 0.5,
                    Vec3::X,
                )))?;
                fixed.push(root);
            }
            for index in 0..unrelated {
                fixed.push(spawn(
                    &mut graph,
                    IVec3::new((index % 8) * 800, -400, (index / 8 + 1) * 800),
                )?);
            }
            let creation = graph.compile_with_static_parts(fixed)?;
            let initial = MachineState::at_rest(&creation);
            let geometry = super::MachineCollisionGeometry::new(&creation, 1)?;
            let scene = wide_floor()?;
            let row = if rotor {
                creation.dynamics.coordinate_velocities[0]
            } else {
                creation.dynamics.body_velocities[0].start
            };
            let mut unit = vec![0.0; initial.velocities.len()];
            unit[row] = 1.0;
            let path = mechanic_physics::MachineMotion::new(&creation, 1, &initial, &unit)?;
            let reach = creation
                .colliders
                .iter()
                .filter(|c| !creation.compounds[c.compound_index as usize].is_static)
                .map(|c| {
                    Ok(path.bounds()[c.compound_index as usize]
                        .point_speed(ContactPolytope::from_collider(c)?.conservative_radius()?))
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?
                .into_iter()
                .fold(0.0_f64, f64::max);
            let threshold = settings.continuous_travel * f64::from(settings.substeps)
                / mechanic_core::TICK_SECONDS
                / reach;
            for speed in [0.0, 1.0, threshold * 0.99, threshold * 1.01, 500.0] {
                let mut state = initial.clone();
                for pose in &mut state.poses {
                    pose.position.y += 2.0;
                }
                let row = if rotor {
                    creation.dynamics.coordinate_velocities[0]
                } else {
                    creation.dynamics.body_velocities[0].start
                };
                state.velocities[row] = speed;
                let mut machine = CpuMachine::new(creation.clone(), 1, state)?;
                let mut samples = Vec::new();
                let (mut query, mut continuous) = (0.0, 0.0);
                let mut empty_reuses = 0;
                let (mut candidates, mut requeries, mut sweeps, mut degraded) = (0, 0, 0, 0);
                let mut groups = [0_usize; 4];
                for tick in 0..options.warmup + options.ticks {
                    let terrain = SoftStepTerrain {
                        scene: &scene,
                        geometry: &geometry,
                        topology_generation: 1,
                        origin: DVec3::ZERO,
                    };
                    let started = Instant::now();
                    machine.step(DVec3::ZERO, &settings, &[], &[], Some(terrain))?;
                    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                    if tick < options.warmup {
                        continue;
                    }
                    let d = machine.diagnostics();
                    samples.push(elapsed);
                    query += d.query_ms;
                    continuous += d.continuous_ms;
                    candidates +=
                        d.continuous_triangle_candidates + d.continuous_collider_pair_candidates;
                    groups[0] += d.refreshed_contact_groups;
                    groups[1] += d.reused_contact_groups;
                    groups[2] += d.detailed_sweep_preparations;
                    groups[3] += d.clearance_certificate_failures;
                    requeries += d.requeries;
                    empty_reuses += d.empty_contact_reuses;
                    sweeps += d.continuous_sweeps;
                    degraded += usize::from(d.degraded);
                }
                samples.sort_by(f64::total_cmp);
                println!(
                    "{}",
                    json!({"scenario":if rotor {"rotor"} else {"translation"}, "unrelated":unrelated,"speed":speed,"threshold_speed":threshold,"ticks":options.ticks,"p50_ms":samples[samples.len()/2],"p95_ms":samples[samples.len()*95/100],"refreshed_contact_groups":groups[0],"reused_contact_groups":groups[1],"detailed_sweep_preparations":groups[2],"clearance_certificate_failures":groups[3],"query_ms":query,"continuous_ms":continuous,"continuous_candidates":candidates,"requeries":requeries,"empty_contact_reuses":empty_reuses,"sweeps":sweeps,"degraded_ticks":degraded,"state_hash":machine.snapshot().state_hash()})
                );
            }
        }
    }
    vehicle(options)
}

// Four driven cuboid wheels and a chassis: no saved-world or terrain assets.
#[allow(clippy::too_many_lines)] // Keep paired vehicle setup and reported measurements together.
fn vehicle(options: &scale::Options) -> Result<(), Box<dyn Error>> {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(chassis) = graph.apply(BuildCommand::Spawn(CuboidSpec::new(
        [4, 2, 8],
        BuildPose::default(),
    )?))?
    else {
        unreachable!()
    };
    for x in [-1, 1] {
        for z in [-1, 1] {
            let BuildOutcome::Spawned(wheel) =
                graph.apply(BuildCommand::Spawn(CuboidSpec::new(
                    [2, 4, 2],
                    BuildPose::from_position_ticks(
                        IVec3::new(x * 300, 0, z * 200),
                        GridRotation::default(),
                    ),
                )?))?
            else {
                unreachable!()
            };
            let (first, second) = if x > 0 {
                (FaceKind::PositiveX, FaceKind::NegativeX)
            } else {
                (FaceKind::NegativeX, FaceKind::PositiveX)
            };
            graph.apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(chassis, first),
                FaceRef::part(wheel, second),
                IVec3::new(x, 0, z).as_vec3() * 0.5,
                IVec3::new(x, 0, 0).as_vec3(),
            )))?;
        }
    }
    let creation = graph.compile()?;
    let mut initial = MachineState::at_rest(&creation);
    let (geometry, _) = finite_support::scene(&creation, &mut initial.poses)?;
    let scene = wide_floor()?;
    for speed in [0.0, 1.0, 100.0] {
        let mut machine = CpuMachine::new(creation.clone(), 1, initial.clone())?;
        let commands = (0..creation.coordinate_drives.len())
            .map(|coordinate| DriveCommand {
                tick: 1,
                topology_generation: 1,
                coordinate,
                drive: mechanic_core::CoordinateDrive {
                    mode: DriveMode::Speed,
                    target_speed: if coordinate < 2 { -speed } else { speed },
                    max_speed: 500.0,
                    max_acceleration: 100.0,
                    source_a_max_acceleration: 100.0,
                    source_a_no_load_speed: 500.0,
                    ..mechanic_core::CoordinateDrive::PASSIVE
                },
            })
            .collect::<Vec<_>>();
        let settings = SoftStepSettings::default();
        let mut samples = Vec::new();
        let (mut query, mut continuous) = (0.0, 0.0);
        let (mut candidates, mut requeries, mut sweeps, mut degraded) = (0, 0, 0, 0);
        let mut groups = [0_usize; 4];
        for tick in 0..options.warmup + options.ticks {
            let terrain = SoftStepTerrain {
                scene: &scene,
                geometry: &geometry,
                topology_generation: 1,
                origin: DVec3::ZERO,
            };
            let start = Instant::now();
            machine.step(
                GRAVITY,
                &settings,
                &[],
                if tick == 0 { &commands } else { &[] },
                Some(terrain),
            )?;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            if tick < options.warmup {
                continue;
            }
            samples.push(elapsed);
            let d = machine.diagnostics();
            query += d.query_ms;
            continuous += d.continuous_ms;
            candidates += d.continuous_triangle_candidates + d.continuous_collider_pair_candidates;
            groups[0] += d.refreshed_contact_groups;
            groups[1] += d.reused_contact_groups;
            groups[2] += d.detailed_sweep_preparations;
            groups[3] += d.clearance_certificate_failures;
            requeries += d.requeries;
            sweeps += d.continuous_sweeps;
            degraded += usize::from(d.degraded);
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "{}",
            json!({"scenario":"vehicle", "speed":speed, "unrelated":0,
            "ticks":options.ticks,"p50_ms":samples[samples.len()/2],"p95_ms":samples[samples.len()*95/100],
            "refreshed_contact_groups":groups[0],"reused_contact_groups":groups[1],"detailed_sweep_preparations":groups[2],"clearance_certificate_failures":groups[3],"query_ms":query,"continuous_ms":continuous,"continuous_candidates":candidates,"requeries":requeries,
            "sweeps":sweeps,"degraded_ticks":degraded,"state_hash":machine.snapshot().state_hash()})
        );
    }
    Ok(())
}

// Keep even the fastest translation over the same floor throughout the run.
fn wide_floor() -> Result<super::TerrainContactScene, Box<dyn Error>> {
    let mut chunk = super::ground(false);
    let floor = std::sync::Arc::make_mut(&mut chunk);
    for vertex in &mut floor.vertices {
        vertex[0] *= 16_384.0;
        vertex[2] *= 16_384.0;
    }
    floor.bounds.minimum.0 *= 16_384.0;
    floor.bounds.maximum.0 *= 16_384.0;
    floor.triangle_bvh.bounds = floor.bounds;
    floor.triangle_bvh.nodes[0].bounds = floor.bounds;
    let mut scene = super::TerrainContactScene::default();
    scene.publish(1, &[chunk], &[])?;
    Ok(scene)
}
