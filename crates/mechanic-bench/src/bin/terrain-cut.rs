//! The full streaming cut around a point in every biome: what selection
//! picks, what meshing it costs, and what the GPU would hold. One JSONL line
//! per biome and a total, so selection, sampling, and triangle budgets can be
//! compared before and after a change.
//!
//! `cargo run -p mechanic-bench --release --bin terrain-cut -- --seed 42`
//! with optional `--biome <name>` to measure one biome and `--worldgen <dir>`
//! to measure an authored definition.

use std::error::Error;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use bevy_math::DVec3;
use mechanic_world::{
    ActiveTerrainNode, TerrainBoundsCache, TerrainField, TerrainMeshRequest, TerrainOctree,
    WorldPosition, WorldSeed, WorldgenSpec, mesh_chunk_profiled, select_active_nodes_cached,
};

/// Spacing of the search grid for biome hearts, and how far it reaches.
const SEARCH_STEP: f64 = 64.0;
const SEARCH_REACH: f64 = 4_000.0;
/// A heart is a point whose neighbours this far away share its biome.
const HEART_RADIUS: f64 = 192.0;

struct CutReport {
    selected_by_level: Vec<usize>,
    triangles: usize,
    triangles_by_level: Vec<usize>,
    empty_jobs: usize,
    empty_by_level: Vec<usize>,
    selection_ms: f64,
    wall_ms: f64,
    sampling_cpu_ms: f64,
    extraction_cpu_ms: f64,
    sampling_p95_by_level: Vec<f64>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let value = |name: &str| {
        args.iter()
            .position(|arg| arg == name)
            .and_then(|index| args.get(index + 1))
            .cloned()
    };
    let seed: u64 = value("--seed").map_or(Ok(42), |value| value.parse())?;
    let only = value("--biome");
    let spec = match value("--worldgen") {
        Some(directory) => Arc::new(WorldgenSpec::from_dir(Path::new(&directory))?),
        None => WorldgenSpec::embedded(),
    };
    let field = TerrainField::from_spec(WorldSeed(seed), spec)?;
    let names = field
        .spec()
        .biome_names()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let workers = std::thread::available_parallelism().map_or(8, usize::from);

    let mut total_triangles = 0;
    let mut total_cpu_ms = 0.0;
    for name in &names {
        if only.as_ref().is_some_and(|only| only != name) {
            continue;
        }
        let Some((x, z)) = biome_heart(&field, name) else {
            println!("{{\"type\":\"terrain_cut\",\"biome\":\"{name}\",\"found\":false}}");
            continue;
        };
        let ground = field.topmost_surface(x, z).unwrap_or(0.0);
        let focus = WorldPosition(DVec3::new(x, ground + 1.8, z));
        let report = measure_cut(&field, focus, workers);
        total_triangles += report.triangles;
        total_cpu_ms += report.sampling_cpu_ms + report.extraction_cpu_ms;
        println!(
            concat!(
                "{{\"type\":\"terrain_cut\",\"biome\":\"{}\",\"x\":{:.0},\"z\":{:.0},",
                "\"nodes\":{},\"selected_by_level\":{:?},\"triangles\":{},",
                "\"triangles_by_level\":{:?},\"empty_jobs\":{},\"empty_by_level\":{:?},",
                "\"selection_ms\":{:.1},\"wall_ms\":{:.1},\"workers\":{},",
                "\"sampling_cpu_ms\":{:.1},\"extraction_cpu_ms\":{:.1},",
                "\"sampling_p95_ms_by_level\":{:?}}}"
            ),
            name,
            x,
            z,
            report.selected_by_level.iter().sum::<usize>(),
            report.selected_by_level,
            report.triangles,
            report.triangles_by_level,
            report.empty_jobs,
            report.empty_by_level,
            report.selection_ms,
            report.wall_ms,
            workers,
            report.sampling_cpu_ms,
            report.extraction_cpu_ms,
            report
                .sampling_p95_by_level
                .iter()
                .map(|value| (value * 100.0).round() / 100.0)
                .collect::<Vec<_>>(),
        );
    }
    println!(
        "{{\"type\":\"terrain_cut_total\",\"seed\":{seed},\"worldgen\":{},\"triangles\":{total_triangles},\"cpu_ms\":{total_cpu_ms:.1}}}",
        field.spec().hash()
    );
    Ok(())
}

/// The search-grid point nearest the origin whose surroundings all belong
/// to `name`.
fn biome_heart(field: &TerrainField, name: &str) -> Option<(f64, f64)> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a search grid of a few hundred steps"
    )]
    let steps = (SEARCH_REACH / SEARCH_STEP) as i32;
    let mut candidates = Vec::new();
    for i in -steps..=steps {
        for k in -steps..=steps {
            let (x, z) = (f64::from(i) * SEARCH_STEP, f64::from(k) * SEARCH_STEP);
            candidates.push((x.hypot(z), x, z));
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    candidates.into_iter().find_map(|(_, x, z)| {
        let inside = |dx: f64, dz: f64| field.biome_at(x + dx, z + dz) == name;
        (inside(0.0, 0.0)
            && inside(HEART_RADIUS, 0.0)
            && inside(-HEART_RADIUS, 0.0)
            && inside(0.0, HEART_RADIUS)
            && inside(0.0, -HEART_RADIUS))
        .then_some((x, z))
    })
}

/// One meshed node's contribution to a cut.
struct Measured {
    level: usize,
    triangles: usize,
    sampling_ms: f64,
    extraction_ms: f64,
}

fn measure_cut(field: &TerrainField, focus: WorldPosition, workers: usize) -> CutReport {
    let edits = TerrainOctree::default();
    let snapshot = edits.snapshot();
    let mut cache = TerrainBoundsCache::default();
    let started = Instant::now();
    let selection = select_active_nodes_cached(field, &snapshot, focus, &mut cache);
    let selection_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let nodes: Vec<ActiveTerrainNode> = selection.nodes;
    let levels = nodes
        .iter()
        .map(|node| usize::from(node.id.level) + 1)
        .max()
        .unwrap_or(0);
    let mut selected_by_level = vec![0; levels];
    for node in &nodes {
        selected_by_level[usize::from(node.id.level)] += 1;
    }

    let next = AtomicUsize::new(0);
    let results = Mutex::new(Vec::with_capacity(nodes.len()));
    let started = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                let mut local = Vec::new();
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(node) = nodes.get(index) else {
                        break;
                    };
                    let (chunk, metrics) = mesh_chunk_profiled(
                        field,
                        &snapshot,
                        TerrainMeshRequest {
                            node: node.id,
                            generation: node.generation,
                            transition_mask: node.transition_mask,
                        },
                    );
                    local.push(Measured {
                        level: usize::from(node.id.level),
                        triangles: chunk.index_groups.final_index_count(chunk.transition_mask) / 3,
                        sampling_ms: metrics.column_sampling_ms,
                        extraction_ms: metrics.polygonization_ms
                            + metrics.transitions_caps_ms
                            + metrics.bvh_construction_ms,
                    });
                }
                results.lock().expect("results lock").extend(local);
            });
        }
    });
    let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let results = results.into_inner().expect("results lock");
    let mut triangles_by_level = vec![0; levels];
    let mut sampling_by_level = vec![Vec::new(); levels];
    let mut empty_by_level = vec![0; levels];
    for measured in &results {
        if measured.triangles == 0 {
            empty_by_level[measured.level] += 1;
        }
        triangles_by_level[measured.level] += measured.triangles;
        sampling_by_level[measured.level].push(measured.sampling_ms);
    }
    CutReport {
        selected_by_level,
        triangles: triangles_by_level.iter().sum(),
        triangles_by_level,
        empty_jobs: results.iter().filter(|m| m.triangles == 0).count(),
        empty_by_level,
        selection_ms,
        wall_ms,
        sampling_cpu_ms: results.iter().map(|m| m.sampling_ms).sum(),
        extraction_cpu_ms: results.iter().map(|m| m.extraction_ms).sum(),
        sampling_p95_by_level: sampling_by_level
            .iter_mut()
            .map(|samples| {
                samples.sort_by(f64::total_cmp);
                samples
                    .get(samples.len() * 95 / 100)
                    .copied()
                    .unwrap_or_default()
            })
            .collect(),
    }
}
