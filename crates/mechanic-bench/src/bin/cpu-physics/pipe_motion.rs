//! Diagnose individual bearing paths in a read-only saved construction snapshot.
use super::{DVec3, Error, Instant, MachineCollisionGeometry, MachineState, json, scale};
use mechanic_physics::{MachineMotion, TerrainContactScene};

pub(super) fn run(path: &str, options: &scale::Options) -> Result<(), Box<dyn Error>> {
    let instance: mechanic_world::WorldCreationInstanceDoc =
        ron::from_str(&std::fs::read_to_string(path)?)?;
    let loaded = instance.creation.into_graph()?;
    let creation = loaded
        .graph
        .compile_with_suspension_sockets([], &loaded.sockets)?;
    let state = MachineState::at_rest(&creation);
    let geometry = MachineCollisionGeometry::new(&creation, 1)?;
    let scene = TerrainContactScene::default();
    println!(
        "{}",
        json!({"kind":"metadata", "bodies":creation.compounds.len(), "colliders":creation.colliders.len(), "coordinates":creation.dynamics.coordinate_velocities.len(), "terrain":"none", "protocol":"repeat identical single-bearing substep paths at authored poses; all other displacements zero"})
    );
    for bearing in &creation.bearings {
        let Some(coordinate) = bearing.coordinate_index else {
            continue;
        };
        for speed in [0.0, 1.0, 10.0, 40.0] {
            let mut displacement = vec![0.0; state.velocities.len()];
            displacement[creation.dynamics.coordinate_velocities[coordinate as usize]] =
                speed / 240.0;
            let motion = MachineMotion::new(&creation, 1, &state, &displacement)?;
            let mut samples = Vec::new();
            let mut last = None;
            for tick in 0..options.warmup + options.ticks {
                let start = Instant::now();
                let query = scene.sweep(&geometry, &motion, DVec3::ZERO, 0.001, 64)?;
                if tick >= options.warmup {
                    samples.push(start.elapsed().as_secs_f64() * 1000.0);
                }
                last = Some(query);
            }
            samples.sort_by(f64::total_cmp);
            let q = last.unwrap();
            println!(
                "{}",
                json!({"coordinate":coordinate,"speed":speed,"anchor":(creation.compounds[bearing.compound_a as usize].root_translation + bearing.local_anchor_a).to_array(),"bodies":[bearing.compound_a,bearing.compound_b],"p50_ms":samples[samples.len()/2],"p95_ms":samples[samples.len()*95/100],"pairs":q.collider_pair_candidates,"separations":q.separation_evaluations,"poses":q.pose_evaluations,"outcome":format!("{:?}",q.outcome)})
            );
        }
    }
    Ok(())
}

// Keep the same saved pipe 20 m away from optional unrelated builder geometry.
// A diagnostic plane at authored y=4 isolates scene amplification from meshing.
#[expect(clippy::too_many_lines)]
pub(super) fn scene_ticks(
    path: &str,
    background: Option<&str>,
    options: &scale::Options,
) -> Result<(), Box<dyn Error>> {
    use super::{CpuMachine, GRAVITY, SoftStepConfig, SoftStepTerrain};
    let load = |path: &str| -> Result<mechanic_core::CreationDocument, Box<dyn Error>> {
        let instance: mechanic_world::WorldCreationInstanceDoc =
            ron::from_str(&std::fs::read_to_string(path)?)?;
        Ok(instance.creation)
    };
    let mut pipe = load(path)?;
    pipe.transform_cardinal(0, bevy_math::IVec3::X * 80);
    let mut document = match background {
        Some(path) => load(path)?,
        None => mechanic_core::CreationDocument::from_graph(
            &mechanic_core::ConstructionGraph::new(),
            "Pipe scene",
            &[],
        ),
    };
    document.append(pipe)?;
    let loaded = document.into_graph()?;
    let unfixed = loaded
        .graph
        .compile_with_suspension_sockets([], &loaded.sockets)?;
    let mut fixed = Vec::new();
    for collider in &unfixed.colliders {
        let body = &unfixed.compounds[collider.compound_index as usize];
        let shape = mechanic_core::ContactPolytope::from_collider(collider)?;
        let bounds = shape.transformed_bounds(
            body.root_translation.as_dvec3(),
            body.root_rotation.as_dquat(),
        )?;
        if bounds[0].y <= 4.001 {
            fixed.push(collider.source_part);
        }
    }
    fixed.sort();
    fixed.dedup();
    let creation = loaded
        .graph
        .compile_with_suspension_sockets(fixed, &loaded.sockets)?;
    let bearing = creation
        .bearings
        .iter()
        .filter(|b| b.coordinate_index.is_some())
        .max_by(|a, b| {
            let x = |b: &mechanic_core::CompiledBearing| {
                (creation.compounds[b.compound_a as usize].root_translation + b.local_anchor_a).x
            };
            x(a).total_cmp(&x(b))
        })
        .ok_or("missing pipe bearing")?;
    let row = creation.dynamics.coordinate_velocities[bearing.coordinate_index.unwrap() as usize];
    let mut initial = MachineState::at_rest(&creation);
    for pose in &mut initial.poses {
        pose.position.y -= 4.0;
    }
    let geometry = MachineCollisionGeometry::new(&creation, 1)?;
    let mut scene = TerrainContactScene::default();
    scene.publish(1, &[super::ground(false)], &[])?;
    let settings = SoftStepConfig::default();
    let terrain = || SoftStepTerrain {
        scene: &scene,
        geometry: &geometry,
        topology_generation: 1,
        origin: DVec3::ZERO,
    };
    let mut settled = CpuMachine::new(creation.clone(), 1, initial)?;
    for _ in 0..options.warmup {
        settled.step(GRAVITY, &settings, &[], &[], Some(terrain()))?;
    }
    let mut unit = vec![0.0; settled.snapshot().state.velocities.len()];
    unit[row] = 1.0;
    let unit_motion =
        mechanic_physics::MachineMotion::new(&creation, 1, &settled.snapshot().state, &unit)?;
    let unit_travel = creation
        .colliders
        .iter()
        .map(|collider| {
            Ok(
                unit_motion.bounds()[collider.compound_index as usize].point_speed(
                    super::ContactPolytope::from_collider(collider)?.conservative_radius()?,
                ),
            )
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?
        .into_iter()
        .fold(0.0_f64, f64::max);
    let threshold_speed = settings.continuous_travel * f64::from(settings.substeps)
        / mechanic_core::TICK_SECONDS
        / unit_travel;
    println!(
        "{}",
        json!({"kind":"metadata","threshold_speed":threshold_speed,"background":background,"colliders":creation.colliders.len(),"bodies":creation.compounds.len(),"static_bodies":creation.compounds.iter().filter(|body| body.is_static).count(),"static_colliders":creation.compounds.iter().filter(|body| body.is_static).map(|body| body.collider_range.len()).sum::<usize>(),"pipe_coordinate":bearing.coordinate_index,"warmup":options.warmup,"terrain":"diagnostic y=4 plane; intersecting authored parts anchored"})
    );
    let speeds = std::env::var("MECHANIC_PIPE_SPEEDS").map_or_else(
        |_| Ok(vec![0.0, 1.0, 10.0, 40.0]),
        |value| {
            value
                .split(',')
                .map(str::parse::<f64>)
                .collect::<Result<Vec<_>, _>>()
        },
    )?;
    for speed in speeds {
        let mut state = settled.snapshot().state.clone();
        state.velocities[row] = speed;
        let mut machine = CpuMachine::new(creation.clone(), 1, state)?;
        let mut samples = Vec::new();
        let (mut query, mut continuous, mut dynamics, mut constraints) = (0.0, 0.0, 0.0, 0.0);
        let (mut sweeps, mut pairs, mut triangles, mut requeries, mut degraded) = (0, 0, 0, 0, 0);
        let (mut refreshed, mut reused, mut detailed, mut failures, mut cached_supports) =
            (0, 0, 0, 0, 0);
        let (mut query_pairs, mut query_triangles, mut contacts) = (0, 0, 0);
        for _ in 0..options.ticks {
            let start = Instant::now();
            machine.step(GRAVITY, &settings, &[], &[], Some(terrain()))?;
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
            let d = machine.diagnostics();
            cached_supports += d.continuous_cached_supports;
            refreshed += d.refreshed_contact_groups;
            reused += d.reused_contact_groups;
            detailed += d.detailed_sweep_preparations;
            failures += d.clearance_certificate_failures;
            query_pairs += d.collider_pair_candidates;
            query_triangles += d.triangle_candidates;
            contacts += d.contacts;
            query += d.query_ms;
            continuous += d.continuous_ms;
            dynamics += d.dynamics_ms;
            constraints += d.constraints_ms;
            sweeps += d.continuous_sweeps;
            pairs += d.continuous_collider_pair_candidates;
            triangles += d.continuous_triangle_candidates;
            requeries += d.requeries;
            degraded += usize::from(d.degraded);
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "{}",
            json!({"continuous_cached_supports":cached_supports,"refreshed_contact_groups":refreshed,"reused_contact_groups":reused,"detailed_sweep_preparations":detailed,"clearance_certificate_failures":failures,"speed":speed,"final_speed":machine.snapshot().state.velocities[row],"ticks":options.ticks,"p50_ms":samples[samples.len()/2],"p95_ms":samples[samples.len()*95/100],"query_pairs":query_pairs,"query_triangles":query_triangles,"contacts":contacts,"query_ms":query,"continuous_ms":continuous,"dynamics_ms":dynamics,"constraints_ms":constraints,"sweeps":sweeps,"continuous_pairs":pairs,"continuous_triangles":triangles,"requeries":requeries,"degraded_ticks":degraded,"state_hash":machine.snapshot().state_hash()})
        );
    }
    Ok(())
}
