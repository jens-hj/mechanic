//! Headless correctness smoke and hard scale-gate runner.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    env,
    process::ExitCode,
    sync::{Arc, mpsc},
    thread,
    time::{Duration, Instant},
};

use bevy_math::{IVec3, Vec3};
use mechanic_core::{
    BearingSpec, BuildCommand, BuildOutcome, BuildPose, CompiledCreation, ConstructionGraph,
    CuboidSpec, FaceKind, FaceRef, GridRotation, PartId,
};
use mechanic_gpu::{
    CONSTRAINT_NON_CONVERGENCE_FLAG, GpuMechanismCoordinate, GpuPhysics, GpuPhysicsConfig,
};
use mechanic_world::{
    ActiveTerrainNode, TerrainBoundsCache, TerrainField, TerrainMeshChunk, TerrainMeshMetrics,
    TerrainMeshRequest, TerrainNodeId, TerrainOctree, TerrainOctreeSnapshot, TerrainStreamer,
    WorldPosition, WorldSeed, mesh_chunk_profiled, select_active_nodes_cached,
    terrain_loading_worker_count, terrain_worker_count,
};

const SCALE_BODY_COUNT: usize = 100_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    Smoke,
    OpenBearing,
    FourBearingContact,
    Bearings16,
    Bearings64,
    Bearings65,
    Bearings256,
    FourBar,
    InvalidLoop,
    Dense100k,
    Loops100k,
    TerrainStream,
    TerrainDig,
}

impl Scenario {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "smoke" => Some(Self::Smoke),
            "open_bearing" | "bearings_1" => Some(Self::OpenBearing),
            "four_bearing_contact" | "bearings_4" => Some(Self::FourBearingContact),
            "bearings_16" => Some(Self::Bearings16),
            "bearings_64" => Some(Self::Bearings64),
            "bearings_65" => Some(Self::Bearings65),
            "bearings_256" => Some(Self::Bearings256),
            "four_bar" => Some(Self::FourBar),
            "invalid_loop" => Some(Self::InvalidLoop),
            "dense_100k" => Some(Self::Dense100k),
            "loops_100k" => Some(Self::Loops100k),
            "terrain_stream" => Some(Self::TerrainStream),
            "terrain_dig" => Some(Self::TerrainDig),
            _ => None,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::OpenBearing => "open_bearing",
            Self::FourBearingContact => "four_bearing_contact",
            Self::Bearings16 => "bearings_16",
            Self::Bearings64 => "bearings_64",
            Self::Bearings65 => "bearings_65",
            Self::Bearings256 => "bearings_256",
            Self::FourBar => "four_bar",
            Self::InvalidLoop => "invalid_loop",
            Self::Dense100k => "dense_100k",
            Self::Loops100k => "loops_100k",
            Self::TerrainStream => "terrain_stream",
            Self::TerrainDig => "terrain_dig",
        }
    }

    const fn bearing_count(self) -> Option<usize> {
        match self {
            Self::OpenBearing => Some(1),
            Self::FourBearingContact => Some(4),
            Self::Bearings16 => Some(16),
            Self::Bearings64 => Some(64),
            Self::Bearings65 => Some(65),
            Self::Bearings256 => Some(256),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Options {
    scenario: Scenario,
    seconds: u64,
    warmup_seconds: u64,
}

fn main() -> ExitCode {
    match run() {
        Ok(gate_passed) if gate_passed => ExitCode::SUCCESS,
        Ok(_) => ExitCode::from(2),
        Err(error) => {
            eprintln!(
                "{{\"type\":\"error\",\"message\":\"{}\"}}",
                json_escape(&error)
            );
            ExitCode::FAILURE
        }
    }
}

#[allow(clippy::too_many_lines)]
fn run() -> Result<bool, String> {
    let options = parse_options()?;
    if matches!(
        options.scenario,
        Scenario::TerrainStream | Scenario::TerrainDig
    ) {
        return run_terrain_benchmark(options);
    }
    let construction_start = Instant::now();
    let creation = build_scenario(options.scenario)?;
    let construction_ms = construction_start.elapsed().as_secs_f64() * 1000.0;
    let expected_bodies = match options.scenario {
        Scenario::Smoke => 1_024,
        scenario @ (Scenario::OpenBearing
        | Scenario::FourBearingContact
        | Scenario::Bearings16
        | Scenario::Bearings64
        | Scenario::Bearings65
        | Scenario::Bearings256) => scenario.bearing_count().unwrap_or_default() + 1,
        Scenario::FourBar | Scenario::InvalidLoop => 4,
        Scenario::Dense100k | Scenario::Loops100k => SCALE_BODY_COUNT,
        Scenario::TerrainStream | Scenario::TerrainDig => unreachable!(),
    };
    if creation.compounds.len() != expected_bodies {
        return Err(format!(
            "scenario generated {} bodies; expected {expected_bodies}",
            creation.compounds.len()
        ));
    }

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|error| format!("no compatible compute adapter: {error}"))?;
    let adapter_info = adapter.get_info();
    let timestamp_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("mechanic benchmark device"),
        required_features: timestamp_features,
        ..Default::default()
    }))
    .map_err(|error| format!("could not request compute device: {error}"))?;
    let gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        &creation,
        GpuPhysicsConfig {
            collisions_enabled: matches!(options.scenario, Scenario::Smoke | Scenario::Dense100k)
                || options.scenario.bearing_count().is_some(),
            ground_plane_enabled: true,
            mechanism_self_collisions: true,
            solver_iterations: 8,
        },
    )
    .map_err(|error| error.to_string())?;
    if options.scenario == Scenario::FourBar {
        gpu.initialize_mechanism_coordinates(
            &queue,
            &[
                GpuMechanismCoordinate {
                    angle: 0.001,
                    angular_velocity: 0.0,
                },
                GpuMechanismCoordinate {
                    angle: 0.0,
                    angular_velocity: 0.0,
                },
                GpuMechanismCoordinate {
                    angle: 0.0,
                    angular_velocity: 0.0,
                },
            ],
        )
        .map_err(|error| error.to_string())?;
    }

    let warmup_ticks = options.warmup_seconds * 60;
    for tick in 1..=warmup_ticks {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| format!("device failed during warm-up: {error}"))?;

    let measured_ticks = options.seconds * 60;
    let measured_capacity = usize::try_from(measured_ticks)
        .map_err(|_| "measured tick count does not fit this platform".to_owned())?;
    let mut engine_tick_costs_ms = Vec::with_capacity(measured_capacity);
    let mut blocking_wait_costs_ms = Vec::with_capacity(measured_capacity);
    let mut gpu_tick_costs_ms = Vec::with_capacity(measured_capacity);
    let mut kernel_costs_ms: [Vec<f64>; 7] =
        core::array::from_fn(|_| Vec::with_capacity(measured_capacity));
    let mut error_flags = 0_u32;
    let mut pair_count = 0_u32;
    let mut contact_count = 0_u32;
    let mut active_contact_count = 0_u32;
    let mut anchor_residual_meters = 0.0_f32;
    let mut axis_residual_degrees = 0.0_f32;
    for tick in 1..=measured_ticks {
        let start = Instant::now();
        gpu.dispatch_tick(&device, &queue, warmup_ticks + tick);
        let blocking_wait_started = Instant::now();
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| format!("device failed during measured tick: {error}"))?;
        blocking_wait_costs_ms.push(blocking_wait_started.elapsed().as_secs_f64() * 1_000.0);
        let readback = gpu
            .read_last_tick(&device)
            .map_err(|error| format!("tick diagnostic readback failed: {error}"))?;
        error_flags |= readback.error_flags;
        pair_count = pair_count.max(readback.pair_count);
        contact_count = contact_count.max(readback.contact_count);
        active_contact_count = active_contact_count.max(readback.active_contact_count);
        anchor_residual_meters = anchor_residual_meters.max(readback.anchor_residual_meters);
        axis_residual_degrees = axis_residual_degrees.max(readback.axis_residual_degrees);
        if let Some(gpu_tick_ms) = readback.gpu_tick_ms {
            gpu_tick_costs_ms.push(gpu_tick_ms);
        }
        if let Some(timings) = readback.kernel_timings {
            for (samples, value) in kernel_costs_ms.iter_mut().zip([
                timings.integration_ms,
                timings.mechanism_ms,
                timings.broadphase_ms,
                timings.narrowphase_ms,
                timings.contact_solver_ms,
                timings.bearings_ms,
                timings.snapshot_ms,
            ]) {
                samples.push(value);
            }
        }
        engine_tick_costs_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    engine_tick_costs_ms.sort_by(f64::total_cmp);
    blocking_wait_costs_ms.sort_by(f64::total_cmp);
    gpu_tick_costs_ms.sort_by(f64::total_cmp);
    for costs in &mut kernel_costs_ms {
        costs.sort_by(f64::total_cmp);
    }
    let engine_p95_ms = percentile_95(&engine_tick_costs_ms);
    let engine_p50_ms = percentile(&engine_tick_costs_ms, 50);
    let engine_p99_ms = percentile(&engine_tick_costs_ms, 99);
    let sample_count = u32::try_from(engine_tick_costs_ms.len()).unwrap_or(u32::MAX);
    let engine_mean_ms = engine_tick_costs_ms.iter().sum::<f64>() / f64::from(sample_count);
    let gpu_p95_ms = (!gpu_tick_costs_ms.is_empty()).then(|| percentile_95(&gpu_tick_costs_ms));
    let gpu_p50_ms = (!gpu_tick_costs_ms.is_empty()).then(|| percentile(&gpu_tick_costs_ms, 50));
    let gpu_p99_ms = (!gpu_tick_costs_ms.is_empty()).then(|| percentile(&gpu_tick_costs_ms, 99));
    let blocking_wait_p95_ms = percentile_95(&blocking_wait_costs_ms);
    let diagnostics_bytes_per_tick = core::mem::size_of::<mechanic_gpu::GpuDiagnostics>()
        + usize::from(gpu.has_gpu_timestamps()) * 14 * core::mem::size_of::<u64>();
    let mapped_bytes = measured_capacity.saturating_mul(diagnostics_bytes_per_tick);
    let achieved_tps = 1000.0 / engine_mean_ms;
    let timing_source = if gpu.has_gpu_timestamps() {
        "gpu_timestamp"
    } else {
        "unavailable"
    };

    // The smoke kernel proves shared-buffer integration and publication. The
    // scale scenarios remain hard-failed until all declared collision or
    // articulation passes are dispatched and timestamped.
    let kernel_coverage_complete = matches!(
        options.scenario,
        Scenario::Smoke | Scenario::FourBar | Scenario::InvalidLoop
    ) || options.scenario.bearing_count().is_some();
    let expected_constraint_failure = options.scenario == Scenario::InvalidLoop;
    let correctness_passed = if expected_constraint_failure {
        error_flags & CONSTRAINT_NON_CONVERGENCE_FLAG != 0
    } else {
        error_flags == 0
    };
    let gpu_budget_ms = if options
        .scenario
        .bearing_count()
        .is_some_and(|count| count <= 64)
    {
        4.0
    } else {
        16.67
    };
    let budget_passed =
        achieved_tps >= 60.0 && gpu_p95_ms.is_some_and(|cost| cost <= gpu_budget_ms);
    let gate_passed = kernel_coverage_complete && correctness_passed && budget_passed;
    println!(
        concat!(
            "{{\"type\":\"benchmark\",\"scenario\":\"{}\",",
            "\"adapter\":\"{}\",\"backend\":\"{:?}\",",
            "\"bodies\":{},\"colliders\":{},\"bearings\":{},",
            "\"warmup_ticks\":{},\"measured_ticks\":{},",
            "\"construction_ms\":{:.3},\"mean_engine_tick_ms\":{:.3},",
            "\"p50_engine_tick_ms\":{:.3},\"p95_engine_tick_ms\":{:.3},",
            "\"p99_engine_tick_ms\":{:.3},",
            "\"p50_gpu_tick_ms\":{},\"p95_gpu_tick_ms\":{},\"p99_gpu_tick_ms\":{},",
            "\"submission_count\":{},\"blocking_wait_p95_ms\":{:.3},",
            "\"mapped_bytes\":{},\"bulk_snapshot_readback_bytes\":0,",
            "\"dynamic_mesh_upload_bytes\":0,\"tick_backlog\":0,",
            "\"kernel_pipeline_p95_ms\":{},\"physics_tps\":{:.2},",
            "\"kernel_integration_p95_ms\":{},\"kernel_mechanism_p95_ms\":{},",
            "\"kernel_broadphase_p95_ms\":{},\"kernel_narrowphase_p95_ms\":{},",
            "\"kernel_contact_solver_p95_ms\":{},\"kernel_bearings_p95_ms\":{},",
            "\"kernel_snapshot_p95_ms\":{},",
            "\"pairs\":{},\"contacts\":{},\"active_contacts\":{},",
            "\"anchor_residual_m\":{:.8},",
            "\"axis_residual_deg\":{:.8},\"error_flags\":{},",
            "\"timing_source\":\"{}\",",
            "\"kernel_coverage_complete\":{},\"correctness_passed\":{},",
            "\"budget_passed\":{},",
            "\"gate_passed\":{}}}"
        ),
        options.scenario.name(),
        json_escape(&adapter_info.name),
        adapter_info.backend,
        creation.compounds.len(),
        creation.colliders.len(),
        creation.bearings.len(),
        warmup_ticks,
        measured_ticks,
        construction_ms,
        engine_mean_ms,
        engine_p50_ms,
        engine_p95_ms,
        engine_p99_ms,
        gpu_p50_ms.map_or_else(|| "null".to_owned(), |value| format!("{value:.3}")),
        gpu_p95_ms.map_or_else(|| "null".to_owned(), |value| format!("{value:.3}")),
        gpu_p99_ms.map_or_else(|| "null".to_owned(), |value| format!("{value:.3}")),
        measured_ticks,
        blocking_wait_p95_ms,
        mapped_bytes,
        gpu_p95_ms.map_or_else(|| "null".to_owned(), |value| format!("{value:.3}")),
        achieved_tps,
        optional_percentile_95(&kernel_costs_ms[0]),
        optional_percentile_95(&kernel_costs_ms[1]),
        optional_percentile_95(&kernel_costs_ms[2]),
        optional_percentile_95(&kernel_costs_ms[3]),
        optional_percentile_95(&kernel_costs_ms[4]),
        optional_percentile_95(&kernel_costs_ms[5]),
        optional_percentile_95(&kernel_costs_ms[6]),
        pair_count,
        contact_count,
        active_contact_count,
        anchor_residual_meters,
        axis_residual_degrees,
        error_flags,
        timing_source,
        kernel_coverage_complete,
        correctness_passed,
        budget_passed,
        gate_passed,
    );
    Ok(gate_passed)
}

fn optional_percentile_95(samples: &[f64]) -> String {
    if samples.is_empty() {
        "null".to_owned()
    } else {
        format!("{:.3}", percentile_95(samples))
    }
}

fn parse_options() -> Result<Options, String> {
    let mut scenario = None;
    let mut seconds = None;
    let mut warmup_seconds = None;
    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--scenario" => {
                index += 1;
                scenario = args.get(index).and_then(|value| Scenario::parse(value));
                if scenario.is_none() {
                    return Err(
                        "--scenario must be smoke, open_bearing, four_bearing_contact, bearings_16, bearings_64, bearings_65, bearings_256, four_bar, invalid_loop, dense_100k, loops_100k, terrain_stream, or terrain_dig"
                            .to_owned(),
                    );
                }
            }
            "--seconds" => {
                index += 1;
                seconds = Some(parse_positive(args.get(index), "--seconds")?);
            }
            "--warmup" => {
                index += 1;
                warmup_seconds = Some(parse_nonnegative(args.get(index), "--warmup")?);
            }
            "--help" | "-h" => {
                return Err(
                    "usage: mechanic-bench --scenario smoke|open_bearing|four_bearing_contact|bearings_16|bearings_64|bearings_65|bearings_256|four_bar|invalid_loop|dense_100k|loops_100k|terrain_stream|terrain_dig [--seconds N] [--warmup N]"
                        .to_owned(),
                );
            }
            other => return Err(format!("unknown argument {other}")),
        }
        index += 1;
    }
    let scenario = scenario.unwrap_or(Scenario::Smoke);
    Ok(Options {
        scenario,
        seconds: seconds.unwrap_or(
            if matches!(
                scenario,
                Scenario::Smoke
                    | Scenario::OpenBearing
                    | Scenario::FourBearingContact
                    | Scenario::Bearings16
                    | Scenario::Bearings64
                    | Scenario::Bearings65
                    | Scenario::Bearings256
                    | Scenario::FourBar
                    | Scenario::InvalidLoop
                    | Scenario::TerrainStream
                    | Scenario::TerrainDig
            ) {
                1
            } else {
                30
            },
        ),
        warmup_seconds: warmup_seconds.unwrap_or(
            if matches!(
                scenario,
                Scenario::Smoke
                    | Scenario::OpenBearing
                    | Scenario::FourBearingContact
                    | Scenario::Bearings16
                    | Scenario::Bearings64
                    | Scenario::Bearings65
                    | Scenario::Bearings256
                    | Scenario::FourBar
                    | Scenario::InvalidLoop
                    | Scenario::TerrainStream
                    | Scenario::TerrainDig
            ) {
                0
            } else {
                5
            },
        ),
    })
}

fn parse_positive(value: Option<&String>, flag: &str) -> Result<u64, String> {
    let number = parse_nonnegative(value, flag)?;
    if number == 0 {
        Err(format!("{flag} must be positive"))
    } else {
        Ok(number)
    }
}

fn parse_nonnegative(value: Option<&String>, flag: &str) -> Result<u64, String> {
    value
        .ok_or_else(|| format!("{flag} requires an integer"))?
        .parse()
        .map_err(|_| format!("{flag} requires an integer"))
}

fn build_scenario(scenario: Scenario) -> Result<CompiledCreation, String> {
    match scenario {
        Scenario::Smoke => build_dense(1_024),
        scenario @ (Scenario::OpenBearing
        | Scenario::FourBearingContact
        | Scenario::Bearings16
        | Scenario::Bearings64
        | Scenario::Bearings65
        | Scenario::Bearings256) => {
            build_bearing_chain(scenario.bearing_count().unwrap_or_default())
        }
        Scenario::FourBar => build_four_bar(false),
        Scenario::InvalidLoop => build_four_bar(true),
        Scenario::Dense100k => build_dense(SCALE_BODY_COUNT),
        Scenario::Loops100k => build_loops_100k(),
        Scenario::TerrainStream | Scenario::TerrainDig => {
            Err("terrain scenarios do not build construction bodies".to_owned())
        }
    }
}

struct TerrainBenchJob {
    node: ActiveTerrainNode,
    terrain: TerrainOctreeSnapshot,
    queued_at: Instant,
}

struct TerrainBenchResult {
    node: ActiveTerrainNode,
    chunk: TerrainMeshChunk,
    metrics: TerrainMeshMetrics,
    queue_wait_ms: f64,
}

struct TerrainBenchWorkers {
    senders: Vec<mpsc::Sender<TerrainBenchJob>>,
    results: mpsc::Receiver<TerrainBenchResult>,
    threads: Vec<thread::JoinHandle<()>>,
    next_worker: usize,
}

impl TerrainBenchWorkers {
    fn new(count: usize, seed: WorldSeed) -> Self {
        let (result_sender, results) = mpsc::channel();
        let field = Arc::new(TerrainField::new(seed));
        let mut senders = Vec::with_capacity(count);
        let mut threads = Vec::with_capacity(count);
        for index in 0..count {
            let (sender, receiver) = mpsc::channel::<TerrainBenchJob>();
            let result_sender = result_sender.clone();
            let field = Arc::clone(&field);
            let worker = thread::Builder::new()
                .name(format!("terrain-bench-{index}"))
                .spawn(move || {
                    while let Ok(job) = receiver.recv() {
                        let queue_wait_ms = job.queued_at.elapsed().as_secs_f64() * 1_000.0;
                        let (chunk, metrics) = mesh_chunk_profiled(
                            &field,
                            &job.terrain,
                            TerrainMeshRequest {
                                node: job.node.id,
                                generation: job.node.generation,
                                transition_mask: job.node.transition_mask,
                            },
                        );
                        if result_sender
                            .send(TerrainBenchResult {
                                node: job.node,
                                chunk,
                                metrics,
                                queue_wait_ms,
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                })
                .expect("terrain benchmark worker starts");
            senders.push(sender);
            threads.push(worker);
        }
        Self {
            senders,
            results,
            threads,
            next_worker: 0,
        }
    }

    fn submit(&mut self, job: TerrainBenchJob) -> Result<(), String> {
        let worker = self.next_worker % self.senders.len();
        self.next_worker = self.next_worker.wrapping_add(1);
        self.senders[worker]
            .send(job)
            .map_err(|_| "terrain worker queue closed unexpectedly".to_owned())
    }
}

impl Drop for TerrainBenchWorkers {
    fn drop(&mut self) {
        self.senders.clear();
        for worker in self.threads.drain(..) {
            let _ = worker.join();
        }
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Bounded benchmark coordinates become terrain cells.
#[allow(clippy::too_many_lines)] // Keeps one benchmark sample loop and its report together.
fn run_terrain_benchmark(options: Options) -> Result<bool, String> {
    const PUBLISH_BUDGET_MS: f64 = 2.0;
    let seed = WorldSeed(0x0054_4552_5241_494e);
    let field = TerrainField::new(seed);
    let mut edits = TerrainOctree::default();
    let mut streamer = TerrainStreamer::default();
    let mut bounds_cache = TerrainBoundsCache::default();
    let mut active = BTreeMap::new();
    let mut staged_chunks = BTreeMap::new();
    let worker_count = if options.scenario == Scenario::TerrainStream {
        terrain_loading_worker_count()
    } else {
        terrain_worker_count()
    };
    let mut workers = TerrainBenchWorkers::new(worker_count, seed);
    let mut in_flight = BTreeMap::<TerrainNodeId, Instant>::new();
    let mut completed = VecDeque::<TerrainBenchResult>::new();
    let samples = usize::try_from(options.seconds.saturating_mul(60))
        .map_err(|_| "terrain sample count does not fit this platform".to_owned())?;
    let benchmark_started = Instant::now();
    let mut frame_ms = Vec::with_capacity(samples);
    let mut cold_selection_ms = None;
    let mut cached_selection_ms = Vec::new();
    let mut column_sampling_ms = Vec::new();
    let mut polygonization_ms = Vec::new();
    let mut transitions_caps_ms = Vec::new();
    let mut bvh_construction_ms = Vec::new();
    let mut extraction_ms = Vec::new();
    let mut extraction_by_lod: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::new());
    let mut queue_wait_ms = Vec::new();
    let mut publication_ms = Vec::new();
    let mut vertex_count = 0_usize;
    let mut remesh_count = 0_u64;
    let mut removed_cells = 0_u64;
    let mut previous_brush = None;
    let mut maximum_backlog = 0_usize;
    let mut oldest_queue_age_ms = 0.0_f64;
    let mut empty_completed_jobs = 0_u64;
    let mut completed_jobs = 0_u64;
    let mut active_by_lod = [0_usize; 6];
    let mut selected_by_lod = [0_usize; 6];
    let mut rejected_empty = 0_usize;
    let mut rejected_solid = 0_usize;
    let mut cache_memory_bytes = 0_usize;
    let mut local_ready_ms = None;
    let mut horizon_completion_ms = None;
    let mut backlog_drain_ms = None;
    let mut current_selection_started = None;
    for index in 0..samples {
        let frame_started = Instant::now();
        let index_u32 = u32::try_from(index).unwrap_or(u32::MAX);
        let x = f64::from(index_u32) * 0.20;
        let surface = field.surface_height(x, 0.0);
        let edited = match options.scenario {
            Scenario::TerrainStream => false,
            Scenario::TerrainDig => {
                let centre = WorldPosition(bevy_math::DVec3::new(x, surface - 0.35, 0.0));
                let outcome = edits
                    .excavate_sphere_delta(&field, centre, 0.35, previous_brush)
                    .map_err(|error| error.to_string())?;
                previous_brush = Some((centre, 0.35));
                removed_cells = removed_cells.saturating_add(outcome.total_removed_cells());
                remesh_count = remesh_count.saturating_add(outcome.changed_bricks as u64);
                outcome.total_removed_cells() != 0
            }
            _ => unreachable!(),
        };
        let focus = WorldPosition(bevy_math::DVec3::new(x, surface + 1.8, 0.0));
        if edited || index.is_multiple_of(40) {
            let selection_started = Instant::now();
            let selection =
                select_active_nodes_cached(&field, &edits.snapshot(), focus, &mut bounds_cache);
            let elapsed_ms = selection_started.elapsed().as_secs_f64() * 1_000.0;
            if cold_selection_ms.is_none() {
                cold_selection_ms = Some(elapsed_ms);
            } else {
                cached_selection_ms.push(elapsed_ms);
            }
            selected_by_lod = selection.stats.selected_by_lod;
            rejected_empty = selection.stats.rejected_empty;
            rejected_solid = selection.stats.rejected_solid;
            cache_memory_bytes = selection.stats.cache_memory_bytes;
            let critical = selection
                .nodes
                .iter()
                .map(|node| node.id)
                .filter(|&node| node_overlaps_region(node, focus, 16.0))
                .collect::<Vec<_>>();
            streamer.set_pinned(critical.iter().copied());
            streamer.set_critical_nodes(critical);
            streamer.set_desired(selection.nodes);
            current_selection_started = Some(Instant::now());
            horizon_completion_ms = None;
        }

        while in_flight.len() < worker_count {
            let in_flight_ids = in_flight.keys().copied().collect::<BTreeSet<_>>();
            let Some(node) = streamer.next_request(&in_flight_ids, focus) else {
                break;
            };
            streamer.mark_started(node);
            let queued_at = Instant::now();
            workers.submit(TerrainBenchJob {
                node,
                terrain: edits.snapshot(),
                queued_at,
            })?;
            in_flight.insert(node.id, queued_at);
        }
        for result in workers.results.try_iter() {
            in_flight.remove(&result.node.id);
            completed.push_back(result);
        }

        let publication_started = Instant::now();
        while publication_started.elapsed().as_secs_f64() * 1_000.0 < PUBLISH_BUDGET_MS {
            let Some(result) = completed.pop_front() else {
                break;
            };
            let extraction = result.metrics.column_sampling_ms
                + result.metrics.polygonization_ms
                + result.metrics.transitions_caps_ms
                + result.metrics.bvh_construction_ms;
            column_sampling_ms.push(result.metrics.column_sampling_ms);
            polygonization_ms.push(result.metrics.polygonization_ms);
            transitions_caps_ms.push(result.metrics.transitions_caps_ms);
            bvh_construction_ms.push(result.metrics.bvh_construction_ms);
            extraction_ms.push(extraction);
            extraction_by_lod[usize::from(result.node.id.level)].push(extraction);
            queue_wait_ms.push(result.queue_wait_ms);
            oldest_queue_age_ms = oldest_queue_age_ms.max(result.queue_wait_ms);
            completed_jobs = completed_jobs.saturating_add(1);
            vertex_count = vertex_count.max(result.chunk.vertices.len());
            if result
                .chunk
                .index_groups
                .final_index_count(result.chunk.transition_mask)
                == 0
            {
                empty_completed_jobs = empty_completed_jobs.saturating_add(1);
            }
            if streamer.stage(result.node) {
                let id = result.node.id;
                staged_chunks.insert(id, result.chunk);
                for activated in streamer.activate(id) {
                    if let Some(chunk) = staged_chunks.remove(&activated.id) {
                        active.insert(activated.id, chunk);
                        remesh_count = remesh_count.saturating_add(1);
                    }
                }
            }
        }
        publication_ms.push(publication_started.elapsed().as_secs_f64() * 1_000.0);

        let current_active = streamer
            .active()
            .map(|node| node.id)
            .collect::<BTreeSet<_>>();
        active.retain(|id, _| current_active.contains(id));
        if local_ready_ms.is_none()
            && streamer.local_readiness().is_complete()
            && active
                .values()
                .any(|chunk| chunk.index_groups.final_index_count(chunk.transition_mask) != 0)
        {
            local_ready_ms =
                current_selection_started.map(|started| started.elapsed().as_secs_f64() * 1_000.0);
        }
        if horizon_completion_ms.is_none()
            && streamer.backlog() == 0
            && in_flight.is_empty()
            && completed.is_empty()
        {
            horizon_completion_ms =
                current_selection_started.map(|started| started.elapsed().as_secs_f64() * 1_000.0);
        }
        maximum_backlog = maximum_backlog.max(
            streamer
                .backlog()
                .saturating_add(in_flight.len())
                .saturating_add(completed.len()),
        );
        oldest_queue_age_ms =
            oldest_queue_age_ms.max(streamer.oldest_queue_age().as_secs_f64() * 1_000.0);
        frame_ms.push(frame_started.elapsed().as_secs_f64() * 1_000.0);
        thread::yield_now();
    }

    let drain_started = Instant::now();
    while drain_started.elapsed() < Duration::from_secs(2)
        && (streamer.backlog() != 0 || !in_flight.is_empty() || !completed.is_empty())
    {
        let focus = WorldPosition(bevy_math::DVec3::new(
            0.0,
            field.safe_spawn().0.y + 1.8,
            0.0,
        ));
        while in_flight.len() < worker_count {
            let ids = in_flight.keys().copied().collect::<BTreeSet<_>>();
            let Some(node) = streamer.next_request(&ids, focus) else {
                break;
            };
            streamer.mark_started(node);
            let queued_at = Instant::now();
            workers.submit(TerrainBenchJob {
                node,
                terrain: edits.snapshot(),
                queued_at,
            })?;
            in_flight.insert(node.id, queued_at);
        }
        if let Ok(result) = workers.results.recv_timeout(Duration::from_millis(1)) {
            in_flight.remove(&result.node.id);
            completed.push_back(result);
        }
        for result in workers.results.try_iter() {
            in_flight.remove(&result.node.id);
            completed.push_back(result);
        }
        let publication_started = Instant::now();
        while publication_started.elapsed().as_secs_f64() * 1_000.0 < PUBLISH_BUDGET_MS {
            let Some(result) = completed.pop_front() else {
                break;
            };
            let extraction = result.metrics.column_sampling_ms
                + result.metrics.polygonization_ms
                + result.metrics.transitions_caps_ms
                + result.metrics.bvh_construction_ms;
            column_sampling_ms.push(result.metrics.column_sampling_ms);
            polygonization_ms.push(result.metrics.polygonization_ms);
            transitions_caps_ms.push(result.metrics.transitions_caps_ms);
            bvh_construction_ms.push(result.metrics.bvh_construction_ms);
            extraction_ms.push(extraction);
            extraction_by_lod[usize::from(result.node.id.level)].push(extraction);
            queue_wait_ms.push(result.queue_wait_ms);
            completed_jobs = completed_jobs.saturating_add(1);
            vertex_count = vertex_count.max(result.chunk.vertices.len());
            if result
                .chunk
                .index_groups
                .final_index_count(result.chunk.transition_mask)
                == 0
            {
                empty_completed_jobs = empty_completed_jobs.saturating_add(1);
            }
            if streamer.stage(result.node) {
                let id = result.node.id;
                staged_chunks.insert(id, result.chunk);
                for activated in streamer.activate(id) {
                    if let Some(chunk) = staged_chunks.remove(&activated.id) {
                        active.insert(activated.id, chunk);
                        remesh_count = remesh_count.saturating_add(1);
                    }
                }
            }
        }
        publication_ms.push(publication_started.elapsed().as_secs_f64() * 1_000.0);
        if local_ready_ms.is_none()
            && streamer.local_readiness().is_complete()
            && active
                .values()
                .any(|chunk| chunk.index_groups.final_index_count(chunk.transition_mask) != 0)
        {
            local_ready_ms =
                current_selection_started.map(|started| started.elapsed().as_secs_f64() * 1_000.0);
        }
    }
    if horizon_completion_ms.is_none()
        && streamer.backlog() == 0
        && in_flight.is_empty()
        && completed.is_empty()
    {
        let elapsed_ms =
            current_selection_started.map(|started| started.elapsed().as_secs_f64() * 1_000.0);
        horizon_completion_ms = elapsed_ms;
        backlog_drain_ms = Some(drain_started.elapsed().as_secs_f64() * 1_000.0);
    }
    let triangle_count: usize = active
        .values()
        .map(|chunk: &TerrainMeshChunk| {
            chunk.index_groups.final_index_count(chunk.transition_mask) / 3
        })
        .sum();
    for id in active.keys() {
        active_by_lod[usize::from(id.level)] += 1;
    }
    frame_ms.sort_by(f64::total_cmp);
    let p50_ms = frame_ms
        .get(frame_ms.len() / 2)
        .copied()
        .unwrap_or_default();
    let p95_ms = percentile_95_or_zero(&frame_ms);
    let sample_count = u32::try_from(frame_ms.len()).unwrap_or(u32::MAX).max(1);
    let mean_ms = frame_ms.iter().sum::<f64>() / f64::from(sample_count);
    let memory_bytes = edits
        .promoted_brick_count()
        .saturating_mul(32 * 32 * 32 * 8);
    let elapsed_seconds = benchmark_started.elapsed().as_secs_f64().max(0.001);
    let worker_busy_ms: f64 = extraction_ms.iter().sum();
    let cpu_utilization = worker_busy_ms
        / (elapsed_seconds * 1_000.0 * f64::from(u32::try_from(worker_count).unwrap_or(1)))
        * 100.0;
    let jobs_per_second =
        f64::from(u32::try_from(completed_jobs).unwrap_or(u32::MAX)) / elapsed_seconds;
    let publication_p95_ms = percentile_95_or_zero(&publication_ms);
    let cached_selection_p95_ms = percentile_95_or_zero(&cached_selection_ms);
    let extraction_p95_ms = percentile_95_or_zero(&extraction_ms);
    let extraction_p95_by_lod = extraction_by_lod
        .each_ref()
        .map(|samples| percentile_95_or_zero(samples));
    let queue_wait_p95_ms = percentile_95_or_zero(&queue_wait_ms);
    let final_readiness = streamer.local_readiness();
    let remaining_backlog = streamer
        .backlog()
        .saturating_add(in_flight.len())
        .saturating_add(completed.len());
    let cold_selection_passed = cold_selection_ms.is_some_and(|elapsed| elapsed <= 100.0);
    let cached_selection_passed = cached_selection_p95_ms <= 8.0;
    let local_ready_passed = local_ready_ms.is_some_and(|elapsed| elapsed <= 250.0);
    let horizon_passed = horizon_completion_ms.is_some_and(|elapsed| elapsed <= 2_000.0);
    let extraction_passed = match options.scenario {
        Scenario::TerrainStream => extraction_p95_by_lod[2..=5]
            .iter()
            .all(|&elapsed| elapsed <= 4.0),
        Scenario::TerrainDig => extraction_p95_by_lod[0] <= 8.0,
        _ => unreachable!(),
    };
    let edit_latency_passed = options.scenario != Scenario::TerrainDig
        || queue_wait_p95_ms + extraction_p95_ms + publication_p95_ms <= 100.0;
    let backlog_drain_passed = options.scenario != Scenario::TerrainDig
        || backlog_drain_ms.is_some_and(|elapsed| elapsed <= 250.0);
    let budget_passed = cold_selection_passed
        && cached_selection_passed
        && local_ready_passed
        && horizon_passed
        && extraction_passed
        && p95_ms <= 16.67
        && publication_p95_ms <= PUBLISH_BUDGET_MS
        && edit_latency_passed
        && backlog_drain_passed;
    let local_ready_json = optional_milliseconds(local_ready_ms);
    let horizon_completion_json = optional_milliseconds(horizon_completion_ms);
    let backlog_drain_json = optional_milliseconds(backlog_drain_ms);
    println!(
        concat!(
            "{{\"type\":\"benchmark\",\"scenario\":\"{}\",",
            "\"ticks\":{},\"physics_tps\":60.0,\"uncapped_fps\":{:.2},",
            "\"terrain_stage_p50_ms\":{:.3},\"terrain_stage_p95_ms\":{:.3},",
            "\"cold_selection_ms\":{:.3},\"cached_selection_p95_ms\":{:.3},",
            "\"column_sampling_p95_ms\":{:.3},",
            "\"polygonization_p95_ms\":{:.3},\"transitions_caps_p95_ms\":{:.3},",
            "\"bvh_construction_p95_ms\":{:.3},\"extraction_p95_ms\":{:.3},",
            "\"extraction_p95_ms_by_lod\":{:?},",
            "\"queue_wait_p95_ms\":{:.3},\"publication_p95_ms\":{:.3},",
            "\"local_ready_ms\":{},\"horizon_completion_ms\":{},",
            "\"backlog_drain_ms\":{},",
            "\"local_resolved_nodes\":{},\"local_total_nodes\":{},",
            "\"memory_bytes\":{},\"triangle_count\":{},\"vertex_count\":{},",
            "\"streaming_backlog\":{},\"maximum_streaming_backlog\":{},",
            "\"selected_nodes_by_lod\":{:?},",
            "\"active_nodes_by_lod\":{:?},",
            "\"rejected_solid_nodes\":{},\"rejected_empty_nodes\":{},",
            "\"empty_completed_jobs\":{},\"jobs_per_second\":{:.2},",
            "\"oldest_queue_age_ms\":{:.3},\"bounds_cache_bytes\":{},",
            "\"terrain_worker_count\":{},\"cpu_utilization_percent\":{:.2},",
            "\"remesh_count\":{},\"removed_cells\":{},\"error_flags\":0,",
            "\"overflow_flags\":0,\"prototype_exception\":true,",
            "\"budget_passed\":{},\"gate_passed\":{}}}"
        ),
        options.scenario.name(),
        samples,
        1_000.0 / mean_ms.max(0.001),
        p50_ms,
        p95_ms,
        cold_selection_ms.unwrap_or_default(),
        cached_selection_p95_ms,
        percentile_95_or_zero(&column_sampling_ms),
        percentile_95_or_zero(&polygonization_ms),
        percentile_95_or_zero(&transitions_caps_ms),
        percentile_95_or_zero(&bvh_construction_ms),
        extraction_p95_ms,
        extraction_p95_by_lod,
        queue_wait_p95_ms,
        publication_p95_ms,
        local_ready_json,
        horizon_completion_json,
        backlog_drain_json,
        final_readiness.resolved,
        final_readiness.total,
        memory_bytes,
        triangle_count,
        vertex_count,
        remaining_backlog,
        maximum_backlog,
        selected_by_lod,
        active_by_lod,
        rejected_solid,
        rejected_empty,
        empty_completed_jobs,
        jobs_per_second,
        oldest_queue_age_ms,
        cache_memory_bytes,
        worker_count,
        cpu_utilization,
        remesh_count,
        removed_cells,
        budget_passed,
        budget_passed,
    );
    Ok(budget_passed)
}

fn optional_milliseconds(value: Option<f64>) -> String {
    value.map_or_else(
        || "null".to_owned(),
        |milliseconds| format!("{milliseconds:.3}"),
    )
}

#[allow(clippy::cast_precision_loss)]
fn node_overlaps_region(node: TerrainNodeId, centre: WorldPosition, radius: f64) -> bool {
    let minimum = bevy_math::DVec3::from_array(
        node.minimum_cell_i64()
            .map(|cell| cell as f64 * mechanic_world::TERRAIN_CELL_METERS),
    );
    let maximum = bevy_math::DVec3::from_array(
        node.maximum_cell_exclusive_i64()
            .map(|cell| cell as f64 * mechanic_world::TERRAIN_CELL_METERS),
    );
    let region_minimum = centre.0 - bevy_math::DVec3::splat(radius);
    let region_maximum = centre.0 + bevy_math::DVec3::splat(radius);
    minimum.cmple(region_maximum).all() && region_minimum.cmple(maximum).all()
}

fn percentile_95_or_zero(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        0.0
    } else {
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        percentile_95(&sorted)
    }
}

fn build_four_bar(invalid: bool) -> Result<CompiledCreation, String> {
    let mut graph = ConstructionGraph::new();
    let outcomes = graph
        .apply_batch([
            BuildCommand::Spawn(unit_cube(IVec3::ZERO)),
            BuildCommand::Spawn(unit_cube(IVec3::new(4, 0, 0))),
            BuildCommand::Spawn(unit_cube(IVec3::new(4, 4, 0))),
            BuildCommand::Spawn(unit_cube(IVec3::new(0, 4, 0))),
        ])
        .map_err(|error| format!("four-bar part generation failed: {error}"))?;
    let parts = outcomes
        .into_iter()
        .map(|outcome| match outcome {
            BuildOutcome::Spawned(part) => part,
            _ => unreachable!("batch contains only spawn commands"),
        })
        .collect::<Vec<_>>();
    let edges = [
        (
            parts[0],
            FaceKind::PositiveX,
            parts[1],
            FaceKind::NegativeX,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::X,
        ),
        (
            parts[1],
            FaceKind::PositiveY,
            parts[2],
            FaceKind::NegativeY,
            Vec3::new(1.0, 0.5, 0.0),
            Vec3::Y,
        ),
        (
            parts[2],
            FaceKind::NegativeX,
            parts[3],
            FaceKind::PositiveX,
            Vec3::new(0.5, 1.0, 0.0),
            Vec3::NEG_X,
        ),
        (
            parts[3],
            FaceKind::NegativeY,
            parts[0],
            FaceKind::PositiveY,
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::NEG_Y,
        ),
    ];
    graph
        .apply_batch(
            edges.map(|(source, source_face, target, target_face, anchor, axis)| {
                bearing_command(source, source_face, target, target_face, anchor, axis)
            }),
        )
        .map_err(|error| format!("four-bar bearing generation failed: {error}"))?;
    let mut creation = graph.compile().map_err(|error| error.to_string())?;
    if invalid {
        let closure = creation
            .bearings
            .iter_mut()
            .find(|bearing| bearing.coordinate_index.is_none())
            .ok_or_else(|| "four-bar did not compile a closure edge".to_owned())?;
        closure.local_anchor_b += Vec3::new(0.2, 0.13, 0.07);
    }
    Ok(creation)
}

fn build_bearing_chain(bearing_count: usize) -> Result<CompiledCreation, String> {
    let mut graph = ConstructionGraph::new();
    let outcomes = graph
        .apply_batch((0..=bearing_count).map(|index| {
            let x = i32::try_from(index.saturating_mul(4)).expect("chain coordinate fits i32");
            BuildCommand::Spawn(unit_cube(IVec3::new(x, 2, 0)))
        }))
        .map_err(|error| format!("bearing-chain part generation failed: {error}"))?;
    let parts = outcomes
        .into_iter()
        .map(|outcome| match outcome {
            BuildOutcome::Spawned(part) => part,
            _ => unreachable!("batch contains only spawn commands"),
        })
        .collect::<Vec<_>>();
    graph
        .apply_batch((0..bearing_count).map(|index| {
            bearing_command(
                parts[index],
                FaceKind::PositiveX,
                parts[index + 1],
                FaceKind::NegativeX,
                Vec3::new(grid_f32(index) + 0.5, 0.5, 0.0),
                Vec3::X,
            )
        }))
        .map_err(|error| format!("bearing-chain joint generation failed: {error}"))?;
    graph.compile().map_err(|error| error.to_string())
}

fn build_dense(count: usize) -> Result<CompiledCreation, String> {
    let mut graph = ConstructionGraph::new();
    let commands = (0..count).map(|index| {
        let x = i32::try_from(index % 50).expect("x fits i32");
        let y = i32::try_from((index / 50) % 40).expect("y fits i32");
        let z = i32::try_from(index / 2_000).expect("z fits i32");
        BuildCommand::Spawn(unit_cube(IVec3::new(x * 4, y * 4 + 2, z * 4)))
    });
    graph
        .apply_batch(commands)
        .map_err(|error| format!("dense graph generation failed: {error}"))?;
    graph.compile().map_err(|error| error.to_string())
}

fn build_loops_100k() -> Result<CompiledCreation, String> {
    const WIDTH: usize = 100;
    const HEIGHT: usize = 100;
    const DEPTH: usize = 10;
    let mut graph = ConstructionGraph::new();
    let outcomes = graph
        .apply_batch((0..DEPTH).flat_map(|z| {
            (0..HEIGHT).flat_map(move |y| {
                (0..WIDTH).map(move |x| {
                    BuildCommand::Spawn(unit_cube(IVec3::new(
                        i32::try_from(x * 4).expect("x fits i32"),
                        i32::try_from(y * 4).expect("y fits i32"),
                        i32::try_from(z * 4).expect("z fits i32"),
                    )))
                })
            })
        }))
        .map_err(|error| format!("lattice part generation failed: {error}"))?;
    let parts = outcomes
        .into_iter()
        .map(|outcome| match outcome {
            BuildOutcome::Spawned(part) => part,
            _ => unreachable!("batch contains only spawn commands"),
        })
        .collect::<Vec<_>>();
    let part =
        |x: usize, y: usize, z: usize| -> PartId { parts[z * WIDTH * HEIGHT + y * WIDTH + x] };
    let mut bearings = Vec::with_capacity(198_009);
    for z in 0..DEPTH {
        for y in 0..HEIGHT {
            for x in 0..WIDTH - 1 {
                bearings.push(bearing_command(
                    part(x, y, z),
                    FaceKind::PositiveX,
                    part(x + 1, y, z),
                    FaceKind::NegativeX,
                    Vec3::new(grid_f32(x) + 0.5, grid_f32(y), grid_f32(z)),
                    Vec3::X,
                ));
            }
        }
        for y in 0..HEIGHT - 1 {
            for x in 0..WIDTH {
                bearings.push(bearing_command(
                    part(x, y, z),
                    FaceKind::PositiveY,
                    part(x, y + 1, z),
                    FaceKind::NegativeY,
                    Vec3::new(grid_f32(x), grid_f32(y) + 0.5, grid_f32(z)),
                    Vec3::Y,
                ));
            }
        }
    }
    for z in 0..DEPTH - 1 {
        bearings.push(bearing_command(
            part(0, 0, z),
            FaceKind::PositiveZ,
            part(0, 0, z + 1),
            FaceKind::NegativeZ,
            Vec3::new(0.0, 0.0, grid_f32(z) + 0.5),
            Vec3::Z,
        ));
    }
    graph
        .apply_batch(bearings)
        .map_err(|error| format!("lattice bearing generation failed: {error}"))?;
    graph.compile().map_err(|error| error.to_string())
}

fn unit_cube(units: IVec3) -> CuboidSpec {
    CuboidSpec::new([4, 4, 4], BuildPose::new(units, GridRotation::default()))
        .expect("one-metre cube is in range")
}

fn bearing_command(
    source: PartId,
    source_face: FaceKind,
    target: PartId,
    target_face: FaceKind,
    anchor: Vec3,
    axis: Vec3,
) -> BuildCommand {
    BuildCommand::AddBearing(BearingSpec::new(
        FaceRef::part(source, source_face),
        FaceRef::part(target, target_face),
        anchor,
        axis,
    ))
}

fn percentile_95(sorted: &[f64]) -> f64 {
    percentile(sorted, 95)
}

fn percentile(sorted: &[f64], percentage: usize) -> f64 {
    let rank = sorted.len().saturating_mul(percentage).div_ceil(100);
    sorted[rank.saturating_sub(1)]
}

fn grid_f32(value: usize) -> f32 {
    f32::from(u16::try_from(value).unwrap_or(u16::MAX))
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::{Scenario, build_scenario};

    #[test]
    fn smoke_scene_has_expected_rows() {
        let creation = build_scenario(Scenario::Smoke).unwrap();
        assert_eq!(creation.compounds.len(), 1_024);
        assert_eq!(creation.colliders.len(), 1_024);
        assert!(creation.bearings.is_empty());
    }

    #[test]
    fn four_bar_scenarios_have_one_closure() {
        for scenario in [Scenario::FourBar, Scenario::InvalidLoop] {
            let creation = build_scenario(scenario).unwrap();
            assert_eq!(creation.compounds.len(), 4);
            assert_eq!(creation.loop_topology.tree_bearings.len(), 3);
            assert_eq!(creation.loop_topology.closure_bearings.len(), 1);
        }
    }

    #[test]
    fn bearing_sweep_preserves_the_64_65_topology_boundary() {
        for (scenario, bearings) in [
            (Scenario::OpenBearing, 1),
            (Scenario::FourBearingContact, 4),
            (Scenario::Bearings16, 16),
            (Scenario::Bearings64, 64),
            (Scenario::Bearings65, 65),
            (Scenario::Bearings256, 256),
        ] {
            let creation = build_scenario(scenario).unwrap();
            assert_eq!(creation.compounds.len(), bearings + 1);
            assert_eq!(creation.bearings.len(), bearings);
            assert_eq!(creation.loop_topology.tree_bearings.len(), bearings);
            assert!(creation.loop_topology.closure_bearings.is_empty());
        }
    }
}
