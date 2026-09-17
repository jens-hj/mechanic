//! Drive a saved world's construction on its own terrain with W held.
//!
//! Replays what the app publishes to the CPU solver: the paired world
//! construction with its terrain foundations anchored, and leaf terrain chunks
//! around every moving body. Prints one JSONL record per simulated second.
use std::{collections::BTreeMap, path::Path, sync::Arc};

use super::{
    CpuMachine, DVec3, DriveCommand, Error, GRAVITY, Instant, MachineCollisionGeometry,
    MachineState, SoftStepSettings, SoftStepTerrain, TerrainContactScene, json, scale,
};
use mechanic_core::{ContactPolytope, DriveKey, DriveMode, DriveTarget};
use mechanic_physics::MachineKinematics;
use mechanic_world::{
    BrickCoord, FoundationSupport, TerrainField, TerrainMaterial, TerrainMeshRequest,
    TerrainNodeId, TerrainRayHit, TerrainScene, TerrainTransitionMask, WorldPosition, WorldStore,
    mesh_chunk,
};

/// Terrain meshed around each moving body, in bricks.
const REACH_BRICKS: i32 = 2;

#[allow(clippy::too_many_lines)]
pub(super) fn run(directory: &str, options: &scale::Options) -> Result<(), Box<dyn Error>> {
    let directory = Path::new(directory);
    let store = WorldStore::new(directory.parent().ok_or("world directory has no parent")?);
    let world = store.load_world(directory)?;
    let (instance, _) = store
        .load_space_pair(&world)?
        .ok_or("world has no construction")?;
    let loaded = instance.creation.into_graph()?;
    let edits = store.load_octree(&world.name)?;
    let field = TerrainField::new(world.seed);
    let terrain = TerrainScene {
        field: &field,
        edits: &edits,
    };

    let loose = loaded
        .graph
        .compile_with_suspension_sockets([], &loaded.sockets)?;
    let mut bounds = BTreeMap::new();
    for collider in &loose.colliders {
        let body = &loose.compounds[collider.compound_index as usize];
        let [low, high] = ContactPolytope::from_collider(collider)?.transformed_bounds(
            body.root_translation.as_dvec3(),
            body.root_rotation.as_dquat(),
        )?;
        let entry = bounds
            .entry(collider.source_part)
            .or_insert([DVec3::INFINITY, DVec3::NEG_INFINITY]);
        *entry = [entry[0].min(low), entry[1].max(high)];
    }
    let anchored = bounds
        .iter()
        .filter(|(_, [low, high])| {
            FoundationSupport::rectangular(
                &terrain,
                TerrainRayHit {
                    position: WorldPosition(DVec3::new(
                        (low.x + high.x) * 0.5,
                        low.y,
                        (low.z + high.z) * 0.5,
                    )),
                    normal: bevy_math::Vec3::Y,
                    distance: 0.0,
                    material_weights: [0.0; TerrainMaterial::COUNT],
                    chunk_generation: 0,
                    triangle: 0,
                },
                high.x - low.x,
                high.z - low.z,
            )
            .has_valid_anchor()
        })
        .map(|(&part, _)| part)
        .collect::<Vec<_>>();
    let creation = loaded
        .graph
        .compile_with_suspension_sockets(anchored.iter().copied(), &loaded.sockets)?;
    let geometry = MachineCollisionGeometry::new(&creation, 1)?;

    let throttle = DriveKey::new('W').ok_or("invalid key")?;
    let mut drives = creation.resolve_coordinate_drives(&loaded.graph);
    let mut throttled = 0;
    for (_, link) in loaded.graph.drive_links() {
        let Some(coordinate) = creation
            .bearings
            .iter()
            .find(|bearing| bearing.source_bearing == link.bearing)
            .and_then(|bearing| bearing.coordinate_index)
        else {
            continue;
        };
        let Some(state) = link
            .program
            .states()
            .iter()
            .position(|state| state.trigger().is_some_and(|t| t.key() == throttle))
        else {
            continue;
        };
        let drive = &mut drives[coordinate as usize];
        if let Some(DriveTarget::Speed(speed)) = link.resolved_target(u8::try_from(state)?) {
            drive.mode = DriveMode::Speed;
            drive.target_speed = speed.clamp(-drive.max_speed, drive.max_speed);
            drive.target_angle = 0.0;
            throttled += 1;
        }
    }

    let dynamic = creation
        .compounds
        .iter()
        .enumerate()
        .filter(|(_, body)| !body.is_static)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    println!(
        "{}",
        json!({"kind":"metadata","world":world.name,"generation":world.construction_generation,"parts":bounds.len(),"anchored_parts":anchored.len(),"bodies":creation.compounds.len(),"dynamic_bodies":dynamic.len(),"colliders":creation.colliders.len(),"cylinders":creation.cylinders.len(),"bearings":creation.bearings.len(),"throttle_drives":throttled,"warmup":options.warmup})
    );

    let mut scene = TerrainContactScene::default();
    let mut meshed = BTreeMap::<BrickCoord, Arc<_>>::new();
    let mut publication = 0;
    let settings = SoftStepSettings::default();
    let mut machine = CpuMachine::new(creation.clone(), 1, MachineState::at_rest(&creation))?;
    let mut window = Window::default();
    for tick in 1..=options.warmup + options.ticks {
        let state = &machine.snapshot().state;
        let wanted = dynamic
            .iter()
            .flat_map(|&body| {
                let centre = WorldPosition(state.poses[body].position)
                    .cell()
                    .map(mechanic_world::WorldCell::brick);
                centre.into_iter().flat_map(|brick| {
                    let span = -REACH_BRICKS..=REACH_BRICKS;
                    span.clone().flat_map(move |x| {
                        let span = span.clone();
                        span.clone().flat_map(move |y| {
                            span.clone().map(move |z| {
                                BrickCoord::new(brick.x + x, brick.y + y, brick.z + z)
                            })
                        })
                    })
                })
            })
            .collect::<std::collections::BTreeSet<_>>();
        let removed = meshed
            .keys()
            .filter(|brick| !wanted.contains(brick))
            .copied()
            .collect::<Vec<_>>();
        let missing = wanted
            .iter()
            .filter(|brick| !meshed.contains_key(brick))
            .copied()
            .collect::<Vec<_>>();
        if !missing.is_empty() || !removed.is_empty() {
            publication += 1;
            let snapshot = edits.snapshot();
            let mut upserts = Vec::new();
            for brick in missing {
                let chunk = Arc::new(
                    mesh_chunk(
                        &field,
                        &snapshot,
                        TerrainMeshRequest {
                            node: TerrainNodeId::leaf(brick),
                            generation: publication,
                            transition_mask: TerrainTransitionMask::NONE,
                        },
                    )
                    .collision_chunk(),
                );
                meshed.insert(brick, Arc::clone(&chunk));
                upserts.push(chunk);
            }
            for brick in &removed {
                meshed.remove(brick);
            }
            scene.publish(
                publication,
                &upserts,
                &removed
                    .iter()
                    .map(|&brick| TerrainNodeId::leaf(brick))
                    .collect::<Vec<_>>(),
            )?;
        }

        let commands = if tick == options.warmup + 1 {
            drives
                .iter()
                .enumerate()
                .map(|(coordinate, &drive)| DriveCommand {
                    tick,
                    topology_generation: 1,
                    coordinate,
                    drive,
                })
                .collect()
        } else {
            Vec::new()
        };
        let step = SoftStepTerrain {
            scene: &scene,
            geometry: &geometry,
            topology_generation: 1,
            origin: DVec3::ZERO,
        };
        let started = Instant::now();
        machine.step(GRAVITY, &settings, &[], &commands, Some(step))?;
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        window.record(elapsed, machine.diagnostics());
        if tick % 60 == 0 {
            let state = &machine.snapshot().state;
            let motions =
                MachineKinematics::published_motions(&creation, &state.poses, &state.velocities)?;
            let speed = dynamic
                .iter()
                .map(|&body| motions[body].linear.length())
                .fold(0.0, f64::max);
            println!(
                "{}",
                window.report(tick, speed, meshed.len(), tick > options.warmup)
            );
            window = Window::default();
        }
    }
    Ok(())
}

/// Tick timings and solver work accumulated over one reporting window.
#[derive(Default)]
struct Window {
    samples: Vec<f64>,
    work: BTreeMap<&'static str, f64>,
    degraded: u64,
}

impl Window {
    fn record(&mut self, elapsed: f64, d: &mechanic_physics::SoftStepDiagnostics) {
        self.samples.push(elapsed);
        self.degraded += u64::from(d.degraded);
        #[allow(clippy::cast_precision_loss)]
        for (name, value) in [
            ("query_ms", d.query_ms),
            ("continuous_ms", d.continuous_ms),
            ("dynamics_ms", d.dynamics_ms),
            ("rows_ms", d.rows_ms),
            ("constraints_ms", d.constraints_ms),
            ("solve_ms", d.solve_ms),
            ("requeries", d.requeries as f64),
            ("empty_contact_reuses", d.empty_contact_reuses as f64),
            (
                "refreshed_contact_groups",
                d.refreshed_contact_groups as f64,
            ),
            ("reused_contact_groups", d.reused_contact_groups as f64),
            ("sweeps", d.continuous_sweeps as f64),
            (
                "continuous_pairs",
                d.continuous_collider_pair_candidates as f64,
            ),
            (
                "continuous_triangles",
                d.continuous_triangle_candidates as f64,
            ),
            (
                "detailed_sweep_preparations",
                d.detailed_sweep_preparations as f64,
            ),
            ("query_pairs", d.collider_pair_candidates as f64),
            ("query_triangles", d.triangle_candidates as f64),
            ("contacts", d.contacts as f64),
            ("rows", d.rows as f64),
        ] {
            *self.work.entry(name).or_default() += value;
        }
    }

    fn report(&mut self, tick: u64, speed: f64, chunks: usize, driving: bool) -> serde_json::Value {
        self.samples.sort_by(f64::total_cmp);
        #[allow(clippy::cast_precision_loss)]
        let count = self.samples.len() as f64;
        let mut record = json!({
            "tick": tick,
            "driving": driving,
            "fastest_body_m_s": speed,
            "terrain_chunks": chunks,
            "p50_ms": self.samples[self.samples.len() / 2],
            "max_ms": self.samples.last(),
            "degraded_ticks": self.degraded,
        });
        for (name, total) in &self.work {
            record[format!("mean_{name}")] = json!(total / count);
        }
        record
    }
}
