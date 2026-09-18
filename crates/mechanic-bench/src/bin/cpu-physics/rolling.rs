//! `wheel-roll`: one rubber-tyred 1 m wheel rolled across finely tessellated
//! terrain. Reports how steadily the axle rides and how much speed survives.

use std::{error::Error, sync::Arc, time::Instant};

use bevy_math::{DVec3, IVec3};
use mechanic_core::{
    BuildCommand, BuildPose, CompiledCreation, ConstructionGraph, ConstructionMaterial,
    CylinderDimensions, CylinderSpec, GridRotation, LayerFace, MaterialAppearance,
};
use mechanic_physics::{
    CpuMachine, MachineCollisionGeometry, MachineState, SoftStepConfig, SoftStepTerrain,
    TerrainContactScene,
};
use mechanic_world::{
    TerrainCollisionChunk, TerrainMaterial, TerrainNodeId, TerrainTriangleGroupMask, TriangleBvh,
    TriangleBvhNode, TriangleBvhTriangle, WorldBounds, WorldPosition,
};
use serde_json::json;

use super::GRAVITY;

const RADIUS: f64 = 0.5;
const TICKS: usize = 600;
// Ticks left for the wheel to settle onto the floor before the ride is measured.
const SETTLE: usize = 30;
const CELL: f64 = 0.5;

pub(super) fn run() -> Result<(), Box<dyn Error>> {
    let flat = |_: f64| 0.0;
    // Gentle 5 cm swells, 7.9 m apart: every triangle edge is a slight crest or valley.
    let wavy = |x: f64| 0.05 * (0.8 * x).sin();
    for (floor, height, speed) in [
        ("flat", &flat as &dyn Fn(f64) -> f64, 1.0),
        ("flat", &flat, 5.0),
        ("flat", &flat, 15.0),
        ("wavy", &wavy, 5.0),
    ] {
        roll(floor, height, speed)?;
    }
    Ok(())
}

fn wheel() -> Result<CompiledCreation, Box<dyn Error>> {
    let spec = CylinderSpec::new(
        CylinderDimensions::new(0.5, 0.0, 0.25)?,
        BuildPose::from_position_ticks(IVec3::Y * 300, GridRotation::new(1, 0, 0)),
    )
    .with_material(ConstructionMaterial::Aluminium)
    .with_layer(
        LayerFace::OuterWall,
        0.25,
        ConstructionMaterial::Rubber,
        MaterialAppearance::BAKED,
    )?;
    let mut graph = ConstructionGraph::new();
    graph.apply(BuildCommand::SpawnCylinder(spec))?;
    Ok(graph.compile()?)
}

#[allow(clippy::cast_precision_loss)] // Tick counts are far below 2^52.
fn roll(floor: &str, height: &dyn Fn(f64) -> f64, speed: f64) -> Result<(), Box<dyn Error>> {
    let creation = wheel()?;
    let axle = creation.cylinders[0].local_rotation.as_dquat() * DVec3::Y;
    let mut state = MachineState::at_rest(&creation);
    state.poses[0].position = DVec3::new(0.0, height(0.0) + RADIUS + 0.001, 0.0);
    let rows = creation.dynamics.body_velocities[0].clone();
    // Rolling along +X about the wheel's Z axle.
    state.velocities[rows].copy_from_slice(&[speed, 0.0, 0.0, 0.0, 0.0, -speed / RADIUS]);
    let geometry = MachineCollisionGeometry::new(&creation, 1)?;
    let mut scene = TerrainContactScene::default();
    scene.publish(1, &[tessellated(height)], &[])?;
    let settings = SoftStepConfig::default();
    let mut machine = CpuMachine::new(creation, 1, state)?;
    let (mut ticks, mut queries) = (Vec::new(), Vec::new());
    let (mut lowest, mut highest) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut triangles, mut contacts, mut requeries, mut hits, mut degraded) = (0, 0, 0, 0, 0);
    let (mut sweeps, mut continuous, mut dynamics, mut constraints) = (0, 0.0, 0.0, 0.0);
    for tick in 1..=TICKS {
        let terrain = SoftStepTerrain {
            scene: &scene,
            geometry: &geometry,
            topology_generation: 1,
            origin: DVec3::ZERO,
        };
        let started = Instant::now();
        machine.step(GRAVITY, &settings, &[], &[], Some(terrain))?;
        ticks.push(started.elapsed().as_secs_f64() * 1000.0);
        let diagnostics = machine.diagnostics();
        queries.push(diagnostics.query_ms);
        triangles += diagnostics.triangle_candidates;
        contacts += diagnostics.contacts;
        requeries += diagnostics.requeries;
        hits += diagnostics.continuous_hits;
        degraded += usize::from(diagnostics.degraded);
        sweeps += diagnostics.continuous_sweeps;
        continuous += diagnostics.continuous_ms;
        dynamics += diagnostics.dynamics_ms;
        constraints += diagnostics.constraints_ms;
        if tick > SETTLE {
            let axle = machine.snapshot().state.poses[0].position;
            let ride = axle.y - height(axle.x);
            lowest = lowest.min(ride);
            highest = highest.max(ride);
        }
    }
    let state = &machine.snapshot().state;
    let final_speed = DVec3::from_slice(&state.velocities[0..3]).length();
    let record = json!({
        "scenario": "wheel-roll",
        "floor": floor,
        "speed": speed,
        "ticks": TICKS,
        "axle_height_range_m": highest - lowest,
        "lowest_axle_m": lowest,
        "speed_retained": final_speed / speed,
        "travelled_m": state.poses[0].position.x,
        "lateral_m": state.poses[0].position.z,
        // Angle between the axle and the floor plane; 90° lies on its side.
        "axle_tilt_deg": (state.poses[0].rotation * axle).y.abs().asin().to_degrees(),
        "p50_ms": percentile(&mut ticks, 50),
        "p95_ms": percentile(&mut ticks, 95),
        "query_mean_ms": queries.iter().sum::<f64>() / TICKS as f64,
        "query_p95_ms": percentile(&mut queries, 95),
        "continuous_mean_ms": continuous / TICKS as f64,
        "dynamics_mean_ms": dynamics / TICKS as f64,
        "constraints_mean_ms": constraints / TICKS as f64,
        "continuous_sweeps": sweeps,
        "triangle_candidates": triangles,
        "mean_contacts": contacts as f64 / TICKS as f64,
        "requeries": requeries,
        "continuous_hits": hits,
        "degraded_ticks": degraded,
    });
    println!("{record}");
    Ok(())
}

fn percentile(samples: &mut [f64], fraction: usize) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[(samples.len() * fraction / 100).min(samples.len() - 1)]
}

// A strip of `CELL`-sized square cells along +X, two triangles each, with a
// balanced hierarchy over consecutive cells.
#[allow(clippy::cast_possible_truncation)] // Metre coordinates well inside f32.
fn tessellated(height: &dyn Fn(f64) -> f64) -> Arc<TerrainCollisionChunk> {
    const COLUMNS: u32 = 400;
    const ROWS: u32 = 32;
    let [x0, z0] = [-4.0, -8.0];
    let mut weights = [0.0; TerrainMaterial::COUNT];
    weights[usize::from(TerrainMaterial::Rock.code())] = 1.0;
    let mask = TerrainTriangleGroupMask::REGULAR;
    let mut vertices = Vec::new();
    for column in 0..=COLUMNS {
        for row in 0..=ROWS {
            let x = x0 + f64::from(column) * CELL;
            let z = z0 + f64::from(row) * CELL;
            vertices.push([x as f32, height(x) as f32, z as f32]);
        }
    }
    let vertex = |column: u32, row: u32| column * (ROWS + 1) + row;
    let mut triangles = Vec::new();
    for column in 0..COLUMNS {
        for row in 0..ROWS {
            let [a, b, c, d] = [
                vertex(column, row),
                vertex(column, row + 1),
                vertex(column + 1, row + 1),
                vertex(column + 1, row),
            ];
            // Wound so the normal points up.
            for indices in [[a, b, c], [a, c, d]] {
                triangles.push(TriangleBvhTriangle {
                    indices,
                    group_mask: mask,
                });
            }
        }
    }
    let bounds_of = |range: std::ops::Range<usize>| {
        let [minimum, maximum] = triangles[range].iter().flat_map(|t| t.indices).fold(
            [DVec3::INFINITY, DVec3::NEG_INFINITY],
            |[lo, hi], index| {
                let point = bevy_math::Vec3::from_array(vertices[index as usize]).as_dvec3();
                [lo.min(point), hi.max(point)]
            },
        );
        WorldBounds {
            minimum: WorldPosition(minimum),
            maximum: WorldPosition(maximum),
        }
    };
    let mut nodes = Vec::new();
    build(&mut nodes, 0..triangles.len(), &bounds_of, mask);
    let bounds = nodes[0].bounds;
    let indices = triangles.iter().flat_map(|t| t.indices).collect();
    Arc::new(TerrainCollisionChunk {
        node: TerrainNodeId::ROOT,
        material_weights: vec![weights; vertices.len()],
        vertices,
        indices,
        bounds,
        generation: 1,
        triangle_bvh: TriangleBvh {
            bounds,
            nodes,
            triangles,
        },
        active_groups: mask,
        ..Default::default()
    })
}

// Depth-first, so every child follows its parent.
fn build(
    nodes: &mut Vec<TriangleBvhNode>,
    range: std::ops::Range<usize>,
    bounds_of: &dyn Fn(std::ops::Range<usize>) -> WorldBounds,
    mask: TerrainTriangleGroupMask,
) -> u32 {
    let index = u32::try_from(nodes.len()).expect("node count fits u32");
    nodes.push(TriangleBvhNode {
        bounds: bounds_of(range.clone()),
        group_mask: mask,
        ..Default::default()
    });
    if range.len() <= 16 {
        let node = &mut nodes[index as usize];
        node.first_triangle = u32::try_from(range.start).expect("triangle row fits u32");
        node.triangle_count = u32::try_from(range.len()).expect("triangle count fits u32");
    } else {
        let middle = range.start + range.len() / 2;
        let left = build(nodes, range.start..middle, bounds_of, mask);
        let right = build(nodes, middle..range.end, bounds_of, mask);
        nodes[index as usize].left_child = Some(left);
        nodes[index as usize].right_child = Some(right);
    }
    index
}
