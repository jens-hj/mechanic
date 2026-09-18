//! Headless correctness smoke and hard scale-gate runner.

mod player_collision;
mod scenes;
mod terrain_benchmark;

use player_collision::run_player_collision_benchmark;
use scenes::{
    SCALE_BODY_COUNT, build_dense, build_four_bar, build_loops_100k, build_suspension_one,
    build_test2_car, test2_phase_drives,
};
use terrain_benchmark::run_terrain_benchmark;

use std::{env, process::ExitCode, time::Instant};

use bevy_math::Vec3;
use mechanic_core::CompiledCreation;
use mechanic_gpu::{
    CONSTRAINT_NON_CONVERGENCE_FLAG, GpuGroundPlane, GpuMechanismCoordinate, GpuMechanismDrive,
    GpuPhysics, GpuPhysicsConfig, GpuSolverRoute,
};

use mechanic_bench::scenarios::build_bearing_chain;
use mechanic_bench::stats::{percentile, percentile_95};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    Smoke,
    OpenBearing,
    SuspensionOne,
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
    PlayerCollision,
    Test2Car,
}

impl Scenario {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "smoke" => Some(Self::Smoke),
            "open_bearing" | "bearings_1" => Some(Self::OpenBearing),
            "suspension_1" => Some(Self::SuspensionOne),
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
            "player_collision" => Some(Self::PlayerCollision),
            "test2_car" => Some(Self::Test2Car),
            _ => None,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::OpenBearing => "open_bearing",
            Self::SuspensionOne => "suspension_1",
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
            Self::PlayerCollision => "player_collision",
            Self::Test2Car => "test2_car",
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

#[expect(clippy::too_many_lines)]
fn run() -> Result<bool, String> {
    let options = parse_options()?;
    if matches!(
        options.scenario,
        Scenario::TerrainStream | Scenario::TerrainDig
    ) {
        return run_terrain_benchmark(options);
    }
    if options.scenario == Scenario::PlayerCollision {
        return run_player_collision_benchmark(options);
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
        Scenario::SuspensionOne => 2,
        Scenario::FourBar | Scenario::InvalidLoop => 4,
        Scenario::Dense100k | Scenario::Loops100k => SCALE_BODY_COUNT,
        Scenario::Test2Car => 9,
        Scenario::TerrainStream | Scenario::TerrainDig | Scenario::PlayerCollision => {
            unreachable!()
        }
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
            collisions_enabled: matches!(
                options.scenario,
                Scenario::Smoke
                    | Scenario::SuspensionOne
                    | Scenario::Dense100k
                    | Scenario::Test2Car
            ) || options.scenario.bearing_count().is_some(),
            ground_plane_enabled: options.scenario != Scenario::SuspensionOne,
            mechanism_self_collisions: options.scenario != Scenario::Test2Car,
            solver_iterations: 8,
        },
    )
    .map_err(|error| error.to_string())?;
    if options.scenario == Scenario::FourBar {
        gpu.initialize_mechanism_coordinates(
            &queue,
            &[
                GpuMechanismCoordinate {
                    position: 0.001,
                    velocity: 0.0,
                },
                GpuMechanismCoordinate {
                    position: 0.0,
                    velocity: 0.0,
                },
                GpuMechanismCoordinate {
                    position: 0.0,
                    velocity: 0.0,
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
    let mut encoding_costs_ms = Vec::with_capacity(measured_capacity);
    let mut blocking_wait_costs_ms = Vec::with_capacity(measured_capacity);
    let mut gpu_tick_costs_ms = Vec::with_capacity(measured_capacity);
    let mut kernel_costs_ms: [Vec<f64>; 7] =
        core::array::from_fn(|_| Vec::with_capacity(measured_capacity));
    let mut observed_stage_mask = u32::MAX;
    let mut minimum_integrated_bodies = u32::MAX;
    let mut minimum_published_bodies = u32::MAX;
    let mut minimum_validated_bearings = u32::MAX;
    let mut error_flags = 0_u32;
    let mut pair_count = 0_u32;
    let mut contact_count = 0_u32;
    let mut active_contact_count = 0_u32;
    let mut planned_solver_sweeps = 0_u32;
    let mut executed_solver_sweeps = 0_u32;
    let mut anchor_residual_meters = 0.0_f32;
    let mut axis_residual_degrees = 0.0_f32;
    let mut test2_phase = usize::MAX;
    let test2_base_drives = (options.scenario == Scenario::Test2Car).then(|| {
        creation
            .coordinate_drives
            .iter()
            .copied()
            .map(GpuMechanismDrive::from)
            .collect::<Vec<_>>()
    });
    for tick in 1..=measured_ticks {
        if let Some(base_drives) = &test2_base_drives {
            let phase =
                usize::try_from((tick.saturating_sub(1) * 6 / measured_ticks.max(1)).min(5))
                    .unwrap_or(5);
            if phase != test2_phase {
                let drives = test2_phase_drives(base_drives, &creation, phase);
                gpu.write_mechanism_drives(&queue, &drives)
                    .map_err(|error| error.to_string())?;
                if phase == 5 {
                    let normal = Vec3::new(0.08, 1.0, 0.04).normalize();
                    let planes = creation
                        .colliders
                        .iter()
                        .map(|collider| {
                            let point = Vec3::new(
                                0.0,
                                if collider.compound_index.is_multiple_of(2) {
                                    0.04
                                } else {
                                    -0.02
                                },
                                0.0,
                            );
                            GpuGroundPlane::through_point(normal, point)
                        })
                        .collect::<Vec<_>>();
                    gpu.write_ground_planes(&queue, &planes)
                        .map_err(|error| error.to_string())?;
                }
                test2_phase = phase;
            }
        }
        let start = Instant::now();
        let encoding_started = Instant::now();
        gpu.dispatch_tick(&device, &queue, warmup_ticks + tick);
        encoding_costs_ms.push(encoding_started.elapsed().as_secs_f64() * 1_000.0);
        let blocking_wait_started = Instant::now();
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| format!("device failed during measured tick: {error}"))?;
        blocking_wait_costs_ms.push(blocking_wait_started.elapsed().as_secs_f64() * 1_000.0);
        let readback = gpu
            .read_last_tick(&device)
            .map_err(|error| format!("tick diagnostic readback failed: {error}"))?;
        observed_stage_mask &= readback.execution.stage_mask;
        minimum_integrated_bodies =
            minimum_integrated_bodies.min(readback.execution.integrated_bodies);
        minimum_published_bodies =
            minimum_published_bodies.min(readback.execution.published_bodies);
        minimum_validated_bearings =
            minimum_validated_bearings.min(readback.execution.validated_bearings);
        error_flags |= readback.error_flags;
        pair_count = pair_count.max(readback.pair_count);
        contact_count = contact_count.max(readback.contact_count);
        active_contact_count = active_contact_count.max(readback.active_contact_count);
        planned_solver_sweeps = planned_solver_sweeps.max(readback.planned_solver_sweeps);
        executed_solver_sweeps = executed_solver_sweeps.max(readback.executed_solver_sweeps);
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
    encoding_costs_ms.sort_by(f64::total_cmp);
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
    let encoding_p95_ms = percentile_95(&encoding_costs_ms);
    let diagnostics_bytes_per_tick = core::mem::size_of::<mechanic_gpu::GpuDiagnostics>()
        + usize::from(gpu.has_gpu_timestamps()) * 28 * core::mem::size_of::<u64>();
    let mapped_bytes = measured_capacity.saturating_mul(diagnostics_bytes_per_tick);
    let achieved_tps = 1000.0 / engine_mean_ms;
    let timing_source = if gpu.has_gpu_timestamps() {
        "gpu_timestamp"
    } else {
        "unavailable"
    };
    let phases_json = if options.scenario == Scenario::Test2Car {
        "[\"idle\",\"drop_settle\",\"straight_acceleration\",\"steering_under_power\",\"coasting\",\"uneven_terrain\"]"
    } else {
        "[]"
    };

    // Observed stage entry is evidence, but does not prove all internal kernels
    // executed. Keep full coverage unproven until every required kernel has a
    // device-written marker; a scenario allowlist must never open a scale gate.
    let kernel_coverage_complete = false;
    let expected_constraint_failure = options.scenario == Scenario::InvalidLoop;
    let base_correctness_passed = if expected_constraint_failure {
        error_flags & CONSTRAINT_NON_CONVERGENCE_FLAG != 0
    } else {
        error_flags == 0
    };
    let test2_correctness_passed = options.scenario != Scenario::Test2Car
        || (active_contact_count == 4
            && gpu.solver_route() == GpuSolverRoute::FusedSmallMechanism
            && planned_solver_sweeps == 8
            && executed_solver_sweeps == 8
            && anchor_residual_meters <= mechanic_core::ANCHOR_TOLERANCE_METERS
            && axis_residual_degrees <= mechanic_core::AXIS_TOLERANCE_DEGREES);
    let correctness_passed = base_correctness_passed && test2_correctness_passed;
    let gpu_budget_ms = if options.scenario == Scenario::Test2Car {
        8.3
    } else if options
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
            "\"cpu_encoding_per_tick_p95_ms\":{:.3},",
            "\"p50_engine_tick_ms\":{:.3},\"p95_engine_tick_ms\":{:.3},",
            "\"p99_engine_tick_ms\":{:.3},",
            "\"p50_gpu_tick_ms\":{},\"p95_gpu_tick_ms\":{},\"p99_gpu_tick_ms\":{},",
            "\"submission_count\":{},\"blocking_wait_p95_ms\":{:.3},",
            "\"mapped_bytes\":{},\"bulk_snapshot_readback_bytes\":0,",
            "\"dynamic_mesh_upload_bytes\":0,\"tick_backlog\":0,",
            "\"ticks_submitted_per_frame\":1,\"in_flight_slots\":1,",
            "\"submission_to_readback_p95_ms\":{:.3},\"visual_update_p95_ms\":0.000,",
            "\"solver_route\":\"{}\",\"solver_iterations\":{},",
            "\"planned_solver_sweeps\":{},\"executed_solver_sweeps\":{},",
            "\"phases\":{},",
            "\"kernel_pipeline_p95_ms\":{},\"physics_tps\":{:.2},",
            "\"kernel_integration_p95_ms\":{},\"kernel_mechanism_p95_ms\":{},",
            "\"kernel_broadphase_p95_ms\":{},\"kernel_narrowphase_p95_ms\":{},",
            "\"kernel_contact_solver_p95_ms\":{},\"kernel_bearings_p95_ms\":{},",
            "\"kernel_snapshot_p95_ms\":{},",
            "\"pairs\":{},\"contacts\":{},\"active_contacts\":{},",
            "\"expected_ground_contacts\":{},\"ground_contacts_passed\":{},",
            "\"anchor_residual_m\":{:.8},",
            "\"axis_residual_deg\":{:.8},\"error_flags\":{},",
            "\"timing_source\":\"{}\",",
            "\"observed_stage_mask_every_tick\":{},",
            "\"minimum_integrated_bodies\":{},\"minimum_published_bodies\":{},",
            "\"minimum_validated_bearings\":{},",
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
        encoding_p95_ms,
        engine_p50_ms,
        engine_p95_ms,
        engine_p99_ms,
        gpu_p50_ms.map_or_else(|| "null".to_owned(), |value| format!("{value:.3}")),
        gpu_p95_ms.map_or_else(|| "null".to_owned(), |value| format!("{value:.3}")),
        gpu_p99_ms.map_or_else(|| "null".to_owned(), |value| format!("{value:.3}")),
        measured_ticks,
        blocking_wait_p95_ms,
        mapped_bytes,
        engine_p95_ms,
        gpu.solver_route().name(),
        8,
        planned_solver_sweeps,
        executed_solver_sweeps,
        phases_json,
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
        if options.scenario == Scenario::Test2Car {
            4
        } else {
            0
        },
        options.scenario != Scenario::Test2Car || active_contact_count == 4,
        anchor_residual_meters,
        axis_residual_degrees,
        error_flags,
        timing_source,
        observed_stage_mask,
        minimum_integrated_bodies,
        minimum_published_bodies,
        minimum_validated_bearings,
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
                        "--scenario must be smoke, open_bearing, suspension_1, four_bearing_contact, bearings_16, bearings_64, bearings_65, bearings_256, four_bar, invalid_loop, dense_100k, loops_100k, terrain_stream, terrain_dig, player_collision, or test2_car"
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
                    "usage: mechanic-bench --scenario smoke|open_bearing|suspension_1|four_bearing_contact|bearings_16|bearings_64|bearings_65|bearings_256|four_bar|invalid_loop|dense_100k|loops_100k|terrain_stream|terrain_dig|player_collision|test2_car [--seconds N] [--warmup N]"
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
        Scenario::SuspensionOne => build_suspension_one(),
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
        Scenario::Test2Car => build_test2_car(),
        Scenario::TerrainStream | Scenario::TerrainDig | Scenario::PlayerCollision => {
            Err("terrain scenarios do not build construction bodies".to_owned())
        }
    }
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::{Scenario, build_scenario};

    #[test]
    fn test2_car_preserves_captured_topology() {
        let creation = build_scenario(Scenario::Test2Car).unwrap();
        assert_eq!(creation.part_to_compound.len(), 94);
        assert_eq!(creation.compounds.len(), 9);
        assert_eq!(creation.colliders.len(), 842);
        assert_eq!(creation.bearings.len(), 8);
    }

    #[test]
    fn smoke_scene_has_expected_rows() {
        let creation = build_scenario(Scenario::Smoke).unwrap();
        assert_eq!(creation.compounds.len(), 1_024);
        assert_eq!(creation.colliders.len(), 1_024);
        assert!(creation.bearings.is_empty());
    }

    #[test]
    fn single_suspension_scene_has_one_grounded_mechanism() {
        let creation = build_scenario(Scenario::SuspensionOne).unwrap();
        assert_eq!(creation.compounds.len(), 2);
        assert_eq!(
            creation
                .compounds
                .iter()
                .filter(|body| body.is_static)
                .count(),
            1
        );
        assert_eq!(creation.bearings.len(), 1);
        assert_eq!(creation.loop_topology.tree_bearings.len(), 1);
        assert!(creation.loop_topology.closure_bearings.is_empty());
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
