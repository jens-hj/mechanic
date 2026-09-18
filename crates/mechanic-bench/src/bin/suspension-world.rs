//! Reproducible installed-car physics benchmark; stdout is JSONL.
use std::{error::Error, time::Instant};

use bevy_math::{DVec3, Quat, Vec3};
use mechanic_core::CreationDocument;
use mechanic_gpu::{GpuMechanismDrive, GpuPhysics, GpuPhysicsConfig, TerrainPreparationCache};
use mechanic_world::{
    BrickCoord, TerrainCollisionChunk, TerrainField, TerrainMeshRequest, TerrainNodeId,
    TerrainOctree, TerrainTransitionMask, WorldPosition, WorldSeed, mesh_chunk,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn terrain(field: &TerrainField, position: DVec3, generation: u64) -> Vec<TerrainCollisionChunk> {
    let brick = WorldPosition(position).cell().unwrap().brick();
    let edits = TerrainOctree::default().snapshot();
    let mut chunks = Vec::new();
    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                chunks.push(
                    mesh_chunk(
                        field,
                        &edits,
                        TerrainMeshRequest {
                            node: TerrainNodeId::leaf(BrickCoord::new(
                                brick.x + x,
                                brick.y + y,
                                brick.z + z,
                            )),
                            generation,
                            transition_mask: TerrainTransitionMask::NONE,
                        },
                    )
                    .collision_chunk(),
                );
            }
        }
    }
    chunks
}

#[expect(clippy::too_many_lines)]
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let plane = args.iter().any(|arg| arg == "--plane");
    let seconds: u32 = args
        .windows(2)
        .find(|pair| pair[0] == "--seconds")
        .map_or(Ok(60), |pair| pair[1].parse())?;
    if seconds == 0 {
        return Err("--seconds must be positive".into());
    }
    let field = TerrainField::new(WorldSeed(91));
    let origin = DVec3::new(0.0, field.surface_height(0.0, 0.0), 0.0);
    let doc: CreationDocument =
        ron::from_str(include_str!("../../../../creations/suspension-car.mech"))?;
    if let Some([_, root]) = args.windows(2).find(|pair| pair[0] == "--write-world") {
        write_world(root, &doc, origin)?;
        return Ok(());
    }
    let creation = doc.clone().into_graph()?.graph.compile()?;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    eprintln!("Adapter: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
        ..Default::default()
    }))?;
    let mut gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        &creation,
        GpuPhysicsConfig {
            ground_plane_enabled: plane,
            ..Default::default()
        },
    )?;
    gpu.enable_async_readback();
    let mut drives: Vec<_> = creation
        .coordinate_drives
        .iter()
        .copied()
        .map(GpuMechanismDrive::from)
        .collect();
    let mut position = Vec3::ZERO;
    let mut published_brick = None;
    let mut terrain_cache = TerrainPreparationCache::default();
    let mut minimum_up = 1.0_f32;
    let warmup = 180;
    let mut samples = Vec::new();
    let mut measurement = Instant::now();
    for tick in 1..=warmup + seconds * 60 {
        if tick == warmup + 1 {
            measurement = Instant::now();
        }
        let start = Instant::now();
        let mut preparation_ms = 0.0;
        let mut publication_ms = 0.0;
        let mut uploaded_bytes = 0;
        let mut reused_chunks = 0;
        let mut triangles = 0;
        let brick = WorldPosition(origin + position.as_dvec3())
            .cell()
            .unwrap()
            .brick();
        // Replace on brick crossings and force a same-cut remesh every ten seconds.
        if !plane && (published_brick != Some(brick) || tick % 600 == 0) {
            let prepare = Instant::now();
            let chunks = terrain(&field, origin + position.as_dvec3(), u64::from(tick));
            let prepared = terrain_cache.prepare(&chunks, origin, u64::from(tick))?;
            preparation_ms = prepare.elapsed().as_secs_f64() * 1000.0;
            triangles = chunks
                .iter()
                .map(|chunk| chunk.indices.len() / 3)
                .sum::<usize>();
            let publish = Instant::now();
            let stats =
                gpu.publish_prepared_terrain(&device, &queue, &prepared, u64::from(tick), origin)?;
            uploaded_bytes = stats.uploaded_bytes;
            reused_chunks = stats.reused_chunks;
            publication_ms = publish.elapsed().as_secs_f64() * 1000.0;
            published_brick = Some(brick);
        }
        let driving_tick = tick.saturating_sub(warmup);
        let phase = (driving_tick / 180) % 4;
        for link in &doc.drive_links {
            let coordinate = creation.bearings[link.bearing as usize]
                .coordinate_index
                .unwrap() as usize;
            if link.name == "AWD" {
                drives[coordinate].target_speed = if tick <= warmup {
                    0.0
                } else if link.reversed {
                    -40.0
                } else {
                    40.0
                };
            } else if link.name == "Steering" {
                drives[coordinate].target_angle = match phase {
                    1 => -0.20,
                    3 => 0.20,
                    _ => 0.0,
                };
            }
        }
        gpu.write_mechanism_drives(&queue, &drives)?;
        let submission = gpu.dispatch_tick(&device, &queue, u64::from(tick));
        device.poll(wgpu::PollType::wait_indefinitely())?;
        let state = gpu
            .poll_tick_readback(&device)?
            .ok_or("missing completed tick")?;
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        let d = state.diagnostics;
        if d.error_flags != 0
            || state
                .transforms
                .iter()
                .any(|p| p.position.iter().chain(&p.rotation).any(|v| !v.is_finite()))
        {
            return Err(format!("invalid state at tick {tick}: flags {}", d.error_flags).into());
        }
        position = Vec3::from_slice(&state.transforms[0].position[..3]);
        let up = (Quat::from_array(state.transforms[0].rotation) * Vec3::Y).y;
        if tick > warmup {
            minimum_up = minimum_up.min(up);
            samples.push(wall_ms);
            let k = d.kernel_timings.unwrap_or_default();
            println!(
                "{{\"type\":\"tick\",\"plane\":{plane},\"tick\":{tick},\"wall_ms\":{wall_ms},\"preparation_ms\":{preparation_ms},\"publication_ms\":{publication_ms},\"published_triangles\":{triangles},\"uploaded_bytes\":{uploaded_bytes},\"reused_chunks\":{reused_chunks},\"gpu_ms\":{},\"terrain_ms\":{},\"rotation_ms\":{},\"recovery_ms\":{},\"recovery_projection_ms\":{},\"solver_ms\":{},\"encode_ms\":{},\"contacts\":{},\"route\":\"{:?}\",\"failure_flags\":{},\"backlog\":0,\"up_y\":{up},\"x\":{},\"y\":{},\"z\":{}}}",
                d.gpu_tick_ms.unwrap_or_default(),
                k.terrain_traversal_ms,
                k.rotational_sweep_ms,
                k.terrain_recovery_ms,
                k.recovery_projection_ms,
                k.contact_solver_ms,
                submission.cpu_timings.encoding_ms,
                d.contact_count,
                gpu.solver_route(),
                d.error_flags,
                position.x,
                position.y,
                position.z
            );
        }
    }
    let elapsed = measurement.elapsed().as_secs_f64();
    samples.sort_by(f64::total_cmp);
    let percentile = |percent: usize| mechanic_bench::stats::percentile(&samples, percent);
    println!(
        "{{\"type\":\"summary\",\"plane\":{plane},\"samples\":{},\"completed_tps\":{},\"p50_ms\":{},\"p95_ms\":{},\"p99_ms\":{},\"minimum_up_y\":{minimum_up},\"stable\":{},\"serialized_readback\":true}}",
        samples.len(),
        f64::from(seconds * 60) / elapsed,
        percentile(50),
        percentile(95),
        percentile(99),
        minimum_up > 0.8
    );
    Ok(())
}

fn write_world(root: &str, doc: &CreationDocument, origin: DVec3) -> Result<()> {
    use mechanic_world::{WorldCreationInstanceDoc, WorldDocument, WorldPoseDoc, WorldStore};
    let store = WorldStore::new(root);
    let mut world = WorldDocument::new(
        "Suspension performance",
        WorldSeed(91),
        WorldPosition(DVec3::new(
            0.0,
            TerrainField::new(WorldSeed(91)).surface_height(0.0, 8.0) + 0.05,
            8.0,
        )),
    );
    if store.directory_for(&world.name).exists() {
        return Err("test world already exists".into());
    }
    let space = WorldCreationInstanceDoc {
        id: 1,
        creation: doc.clone(),
        root_pose: WorldPoseDoc {
            translation: WorldPosition(origin),
            ..WorldPoseDoc::default()
        },
        joint_coordinates: Vec::new(),
    };
    let garage = WorldCreationInstanceDoc {
        id: 2,
        creation: CreationDocument::from_graph(
            &mechanic_core::ConstructionGraph::new(),
            "Garage",
            &[],
        ),
        root_pose: WorldPoseDoc::default(),
        joint_coordinates: Vec::new(),
    };
    store.save_space_pair(&mut world, &space, &garage)?;
    println!("{}", store.directory_for(&world.name).display());
    Ok(())
}
