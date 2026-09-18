//! Drive a saved world's construction on its own terrain with W held.
//!
//! Replays what the app publishes to the CPU solver: the paired world
//! construction with its terrain foundations anchored, and leaf terrain chunks
//! around every moving body. Prints one JSONL record per simulated second.
use std::{collections::BTreeMap, path::Path, sync::Arc};

use super::{
    CpuMachine, DVec3, DriveCommand, Error, GRAVITY, Instant, MachineCollisionGeometry,
    MachineState, SoftStepConfig, SoftStepTerrain, TerrainContactScene, json, scale,
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

#[allow(clippy::too_many_lines, clippy::cast_possible_truncation)] // Replay protocol; terrain samples use f32.
pub(super) fn run(
    directory: &str,
    options: &scale::Options,
    soil: bool,
) -> Result<(), Box<dyn Error>> {
    let directory = Path::new(directory);
    let store = WorldStore::new(directory.parent().ok_or("world directory has no parent")?);
    let world = store.load_world(directory)?;
    let (instance, _) = store
        .load_space_pair(&world)?
        .ok_or("world has no construction")?;
    let origin = instance.root_pose.translation.0;
    let rotation = instance.root_pose.rotation;
    if bevy_math::Quat::from_array(rotation)
        .angle_between(bevy_math::Quat::IDENTITY)
        .abs()
        > 1.0e-6
        || !instance.joint_coordinates.is_empty()
    {
        return Err(
            "world-drive currently requires an unrotated instance with default joint coordinates"
                .into(),
        );
    }
    let loaded = instance.creation.into_graph()?;
    let (mut edits, clumps) = store.load_material_state(&world.name)?;
    if !clumps.bodies.is_empty() {
        return Err(
            "world-drive does not replay saved clumps; use the material-clumps benchmark".into(),
        );
    }
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
            body.root_translation.as_dvec3() + origin,
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
        json!({"kind":"metadata","soil":soil,"kernel_coverage_complete":false,"world":world.name,"generation":world.construction_generation,"parts":bounds.len(),"anchored_parts":anchored.len(),"bodies":creation.compounds.len(),"dynamic_bodies":dynamic.len(),"colliders":creation.colliders.len(),"cylinders":creation.cylinders.len(),"bearings":creation.bearings.len(),"throttle_drives":throttled,"warmup":options.warmup})
    );

    let mut scene = TerrainContactScene::default();
    let mut meshed = BTreeMap::<BrickCoord, Arc<_>>::new();
    let mut publication = 0;
    let mut pending_soil = mechanic_world::SoilAccumulator::default();
    let mut invalidated = std::collections::BTreeSet::new();
    let mut probes = BTreeMap::<mechanic_world::WorldCell, f64>::new();
    let mut remeshes = 0_u64;
    let mut remesh_ms = 0.0;
    let mut sunk_metres = 0.0;
    let mut maximum_rut = 0.0_f64;
    let settings = SoftStepConfig::default();
    let mut machine = CpuMachine::new(creation.clone(), 1, MachineState::at_rest(&creation))?;
    let mut window = Window::default();
    let mut measured = Window::default();
    let mut measured_remeshes = 0_u64;
    for tick in 1..=options.warmup + options.ticks {
        let tick_started = Instant::now();
        let state = &machine.snapshot().state;
        let wanted = dynamic
            .iter()
            .flat_map(|&body| {
                let centre = WorldPosition(origin + state.poses[body].position)
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
            .filter(|brick| !meshed.contains_key(brick) || invalidated.contains(*brick))
            .copied()
            .collect::<Vec<_>>();
        if !missing.is_empty() || !removed.is_empty() {
            publication += 1;
            let snapshot = edits.snapshot();
            let mut upserts = Vec::new();
            for brick in missing {
                let started = Instant::now();
                let remesh = invalidated.remove(&brick);
                let chunk = Arc::new(mesh_chunk(
                    &field,
                    &snapshot,
                    TerrainMeshRequest {
                        node: TerrainNodeId::leaf(brick),
                        generation: publication,
                        transition_mask: TerrainTransitionMask::NONE,
                    },
                ));
                meshed.insert(brick, Arc::clone(&chunk));
                upserts.push(Arc::new(chunk.collision_chunk()));
                if remesh {
                    remeshes += 1;
                    measured_remeshes += u64::from(tick > options.warmup);
                    remesh_ms += started.elapsed().as_secs_f64() * 1000.0;
                }
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
            origin,
        };
        let started = Instant::now();
        machine.step(GRAVITY, &settings, &[], &commands, Some(step))?;
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        window.record(elapsed, machine.diagnostics());
        let soil_started = Instant::now();
        if soil {
            for load in machine.terrain_loads() {
                pending_soil.accumulate(
                    &edits,
                    &field,
                    mechanic_world::SoilPatch {
                        centre: WorldPosition(origin + load.point),
                        normal: load.normal,
                        radius: load.patch_radius,
                        pressure_pa: (load.normal_impulse
                            / (mechanic_core::TICK_SECONDS
                                * std::f64::consts::PI
                                * load.patch_radius.powi(2)))
                            as f32,
                        seconds: mechanic_core::TICK_SECONDS as f32,
                    },
                )?;
            }
            if tick % 6 == 0 {
                let ready = pending_soil.take_ready();
                for compression in &ready {
                    probes.entry(compression.cell).or_insert_with(|| {
                        meshed
                            .get(&compression.cell.brick())
                            .and_then(|mesh| {
                                mesh.raycast(
                                    WorldPosition(compression.cell.centre().0 + DVec3::Y * 0.2),
                                    -DVec3::Y,
                                    0.5,
                                )
                            })
                            .map_or(f64::NAN, |hit| hit.position.0.y)
                    });
                }
                let outcome = edits.compress_cells(&field, &ready);
                sunk_metres += outcome.sunk_metres;
                // Match streaming: leaf sampling/gradient halos include adjacent bricks.
                for brick in outcome.changed_brick_coordinates() {
                    for z in -1..=1 {
                        for y in -1..=1 {
                            for x in -1..=1 {
                                let neighbor =
                                    BrickCoord::new(brick.x + x, brick.y + y, brick.z + z);
                                if meshed.contains_key(&neighbor) {
                                    invalidated.insert(neighbor);
                                }
                            }
                        }
                    }
                }
            }
        }
        let soil_ms = soil_started.elapsed().as_secs_f64() * 1000.0;
        *window.work.entry("soil_ms").or_default() += soil_ms;
        let total_elapsed = tick_started.elapsed().as_secs_f64() * 1000.0;
        window.total_samples.push(total_elapsed);
        if tick > options.warmup {
            measured.record(elapsed, machine.diagnostics());
            *measured.work.entry("soil_ms").or_default() += soil_ms;
            measured.total_samples.push(total_elapsed);
        }
        if tick % 60 == 0 {
            let state = &machine.snapshot().state;
            let motions =
                MachineKinematics::published_motions(&creation, &state.poses, &state.velocities)?;
            let speed = dynamic
                .iter()
                .map(|&body| motions[body].linear.length())
                .fold(0.0, f64::max);
            for (cell, baseline) in &probes {
                if let Some(hit) = meshed.get(&cell.brick()).and_then(|mesh| {
                    mesh.raycast(
                        WorldPosition(cell.centre().0 + DVec3::Y * 0.2),
                        -DVec3::Y,
                        0.5,
                    )
                }) {
                    maximum_rut = maximum_rut.max(baseline - hit.position.0.y);
                }
            }
            let mut report = window.report(tick, speed, meshed.len(), tick > options.warmup);
            report["soil"] = json!(soil);
            report["maximum_rut_depth_m"] = json!(maximum_rut);
            report["summed_cell_displacement_m"] = json!(sunk_metres);
            report["remesh_count"] = json!(remeshes);
            report["remesh_ms"] = json!(remesh_ms);
            report["kernel_coverage_complete"] = json!(false);
            println!("{report}");
            remeshes = 0;
            remesh_ms = 0.0;
            window = Window::default();
        }
    }
    let mut summary = measured.report(options.warmup + options.ticks, 0.0, meshed.len(), true);
    summary
        .as_object_mut()
        .expect("report is an object")
        .remove("fastest_body_m_s");
    summary["kind"] = json!("summary");
    summary["measured_ticks"] = json!(options.ticks);
    summary["soil"] = json!(soil);
    summary["maximum_rut_depth_m"] = json!(maximum_rut);
    summary["remesh_count"] = json!(measured_remeshes);
    summary["kernel_coverage_complete"] = json!(false);
    println!("{summary}");
    Ok(())
}

/// Tick timings and solver work accumulated over one reporting window.
#[derive(Default)]
struct Window {
    samples: Vec<f64>,
    total_samples: Vec<f64>,
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
        self.total_samples.sort_by(f64::total_cmp);
        #[allow(clippy::cast_precision_loss)]
        let count = self.samples.len() as f64;
        let mut record = json!({
            "tick": tick,
            "driving": driving,
            "fastest_body_m_s": speed,
            "terrain_chunks": chunks,
            "p95_ms": self.samples[(self.samples.len() * 95 / 100).min(self.samples.len() - 1)],
            "total_tick_p95_ms": self.total_samples[(self.total_samples.len() * 95 / 100).min(self.total_samples.len() - 1)],
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

/// Large steel footprint on generated ground, using the normal world replay.
pub(super) fn large_surface(
    options: &scale::Options,
    soil: bool,
    block_width: u8,
) -> Result<(), Box<dyn Error>> {
    use mechanic_core::{
        BuildCommand, BuildPose, ConstructionGraph, ConstructionMaterial, CreationDocument,
        CuboidSpec,
    };
    use mechanic_world::{WorldCreationInstanceDoc, WorldDocument, WorldPoseDoc, WorldSeed};
    let root = std::env::temp_dir().join(format!("mechanic-large-surface-{}", std::process::id()));
    let store = WorldStore::new(&root);
    let field = TerrainField::new(WorldSeed(91));
    let mut graph = ConstructionGraph::new();
    graph.apply(BuildCommand::Spawn(
        CuboidSpec::new([block_width; 3], BuildPose::default())?
            .with_material(ConstructionMaterial::Steel),
    ))?;
    let mut world = WorldDocument::new("Large surface", WorldSeed(91), field.safe_spawn());
    let instance = WorldCreationInstanceDoc {
        id: 1,
        creation: CreationDocument::from_graph(&graph, "Steel cube", &[]),
        root_pose: WorldPoseDoc {
            translation: WorldPosition(DVec3::new(
                0.0,
                field.surface_height(0.0, 0.0) + f64::from(block_width) * 0.125 + 0.2,
                0.0,
            )),
            ..WorldPoseDoc::default()
        },
        joint_coordinates: Vec::new(),
    };
    let garage = WorldCreationInstanceDoc {
        id: 2,
        creation: CreationDocument::from_graph(&ConstructionGraph::new(), "Garage", &[]),
        root_pose: WorldPoseDoc::default(),
        joint_coordinates: Vec::new(),
    };
    store.save_space_pair(&mut world, &instance, &garage)?;
    run(
        store
            .directory_for(&world.name)
            .to_str()
            .ok_or("invalid temporary path")?,
        options,
        soil,
    )
}
