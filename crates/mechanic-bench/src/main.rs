//! Headless correctness smoke and hard scale-gate runner.

use std::{env, process::ExitCode, time::Instant};

use bevy_math::{IVec3, Vec3};
use mechanic_core::{
    BearingSpec, BuildCommand, BuildOutcome, BuildPose, CompiledCreation, ConstructionGraph,
    CuboidSpec, FaceKind, FaceRef, GridRotation, PartId,
};
use mechanic_gpu::{
    CONSTRAINT_NON_CONVERGENCE_FLAG, GpuMechanismCoordinate, GpuPhysics, GpuPhysicsConfig,
};

const SCALE_BODY_COUNT: usize = 100_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    Smoke,
    FourBar,
    InvalidLoop,
    Dense100k,
    Loops100k,
}

impl Scenario {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "smoke" => Some(Self::Smoke),
            "four_bar" => Some(Self::FourBar),
            "invalid_loop" => Some(Self::InvalidLoop),
            "dense_100k" => Some(Self::Dense100k),
            "loops_100k" => Some(Self::Loops100k),
            _ => None,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::FourBar => "four_bar",
            Self::InvalidLoop => "invalid_loop",
            Self::Dense100k => "dense_100k",
            Self::Loops100k => "loops_100k",
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
    let construction_start = Instant::now();
    let creation = build_scenario(options.scenario)?;
    let construction_ms = construction_start.elapsed().as_secs_f64() * 1000.0;
    let expected_bodies = match options.scenario {
        Scenario::Smoke => 1_024,
        Scenario::FourBar | Scenario::InvalidLoop => 4,
        Scenario::Dense100k | Scenario::Loops100k => SCALE_BODY_COUNT,
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
            collisions_enabled: matches!(options.scenario, Scenario::Smoke | Scenario::Dense100k),
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
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| format!("device failed during measured tick: {error}"))?;
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
    gpu_tick_costs_ms.sort_by(f64::total_cmp);
    for costs in &mut kernel_costs_ms {
        costs.sort_by(f64::total_cmp);
    }
    let engine_p95_ms = percentile_95(&engine_tick_costs_ms);
    let sample_count = u32::try_from(engine_tick_costs_ms.len()).unwrap_or(u32::MAX);
    let engine_mean_ms = engine_tick_costs_ms.iter().sum::<f64>() / f64::from(sample_count);
    let gpu_p95_ms = (!gpu_tick_costs_ms.is_empty()).then(|| percentile_95(&gpu_tick_costs_ms));
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
    );
    let expected_constraint_failure = options.scenario == Scenario::InvalidLoop;
    let correctness_passed = if expected_constraint_failure {
        error_flags & CONSTRAINT_NON_CONVERGENCE_FLAG != 0
    } else {
        error_flags == 0
    };
    let budget_passed = achieved_tps >= 60.0 && gpu_p95_ms.is_some_and(|cost| cost <= 16.67);
    let gate_passed = kernel_coverage_complete && correctness_passed && budget_passed;
    println!(
        concat!(
            "{{\"type\":\"benchmark\",\"scenario\":\"{}\",",
            "\"adapter\":\"{}\",\"backend\":\"{:?}\",",
            "\"bodies\":{},\"colliders\":{},\"bearings\":{},",
            "\"warmup_ticks\":{},\"measured_ticks\":{},",
            "\"construction_ms\":{:.3},\"mean_engine_tick_ms\":{:.3},",
            "\"p95_engine_tick_ms\":{:.3},\"p95_gpu_tick_ms\":{},",
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
        engine_p95_ms,
        gpu_p95_ms.map_or_else(|| "null".to_owned(), |value| format!("{value:.3}")),
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
                        "--scenario must be smoke, four_bar, invalid_loop, dense_100k, or loops_100k"
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
                    "usage: mechanic-bench --scenario smoke|four_bar|invalid_loop|dense_100k|loops_100k [--seconds N] [--warmup N]"
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
                Scenario::Smoke | Scenario::FourBar | Scenario::InvalidLoop
            ) {
                1
            } else {
                30
            },
        ),
        warmup_seconds: warmup_seconds.unwrap_or(
            if matches!(
                scenario,
                Scenario::Smoke | Scenario::FourBar | Scenario::InvalidLoop
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
        Scenario::FourBar => build_four_bar(false),
        Scenario::InvalidLoop => build_four_bar(true),
        Scenario::Dense100k => build_dense(SCALE_BODY_COUNT),
        Scenario::Loops100k => build_loops_100k(),
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
    let rank = sorted.len().saturating_mul(95).div_ceil(100);
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
}
