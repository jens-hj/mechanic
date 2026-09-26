//! Terrain streaming and digging benchmarks over a pool of meshing workers.

use crate::{Options, Scenario};
use mechanic_bench::stats::percentile_95_or_zero;
use mechanic_world::{
    ActiveTerrainNode, STREAMED_LEVELS, TerrainBoundsCache, TerrainField, TerrainMeshChunk,
    TerrainMeshMetrics, TerrainMeshRequest, TerrainNodeId, TerrainOctree, TerrainOctreeSnapshot,
    TerrainStreamer, WorldPosition, WorldSeed, mesh_chunk_profiled, select_active_nodes_cached,
    terrain_loading_worker_count, terrain_worker_count,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

pub(crate) struct TerrainBenchJob {
    pub(crate) node: ActiveTerrainNode,
    pub(crate) terrain: TerrainOctreeSnapshot,
    pub(crate) queued_at: Instant,
}

pub(crate) struct TerrainBenchResult {
    pub(crate) node: ActiveTerrainNode,
    pub(crate) chunk: TerrainMeshChunk,
    pub(crate) metrics: TerrainMeshMetrics,
    pub(crate) queue_wait_ms: f64,
}

pub(crate) struct TerrainBenchWorkers {
    pub(crate) senders: Vec<mpsc::Sender<TerrainBenchJob>>,
    pub(crate) results: mpsc::Receiver<TerrainBenchResult>,
    pub(crate) threads: Vec<thread::JoinHandle<()>>,
    pub(crate) next_worker: usize,
}

impl TerrainBenchWorkers {
    pub(crate) fn new(count: usize, seed: WorldSeed) -> Self {
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

    pub(crate) fn submit(&mut self, job: TerrainBenchJob) -> Result<(), String> {
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

#[expect(
    clippy::too_many_lines,
    reason = "keeps one benchmark sample loop and its report together"
)]
pub(crate) fn run_terrain_benchmark(options: Options) -> Result<bool, String> {
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
    let mut extraction_by_lod: [Vec<f64>; STREAMED_LEVELS] = std::array::from_fn(|_| Vec::new());
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
    let mut active_by_lod = [0_usize; STREAMED_LEVELS];
    let mut selected_by_lod = [0_usize; STREAMED_LEVELS];
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
        Scenario::TerrainStream => extraction_p95_by_lod[2..]
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

pub(crate) fn optional_milliseconds(value: Option<f64>) -> String {
    value.map_or_else(
        || "null".to_owned(),
        |milliseconds| format!("{milliseconds:.3}"),
    )
}

#[expect(clippy::cast_precision_loss)]
pub(crate) fn node_overlaps_region(
    node: TerrainNodeId,
    centre: WorldPosition,
    radius: f64,
) -> bool {
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
