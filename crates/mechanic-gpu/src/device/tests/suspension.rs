//! Suspension springs, dampers, bump stops, and holds.

use super::*;

pub(super) fn suspension_test_creation(
    spec: mechanic_core::SuspensionSpec,
    vertical: bool,
    copies: i32,
) -> mechanic_core::CompiledCreation {
    suspension_test_creation_with_anchor(spec, vertical, copies, true)
}

pub(super) fn suspension_test_creation_with_anchor(
    spec: mechanic_core::SuspensionSpec,
    vertical: bool,
    copies: i32,
    grounded: bool,
) -> mechanic_core::CompiledCreation {
    let mut graph = ConstructionGraph::new();
    let axis = if vertical { Vec3::Y } else { Vec3::X };
    let source_face = if vertical {
        FaceKind::PositiveY
    } else {
        FaceKind::PositiveX
    };
    let target_face = if vertical {
        FaceKind::NegativeY
    } else {
        FaceKind::NegativeX
    };
    for copy in 0..copies {
        let base_ticks = IVec3::new(0, 200, copy * 1600);
        #[expect(clippy::cast_possible_truncation)]
        let spacing_ticks = ((spec.initial_length() + 0.625) / 0.0025).round() as i32;
        let offset = if vertical { IVec3::Y } else { IVec3::X } * spacing_ticks;
        let mut spawn = |ticks, dimensions| {
            spawned_part(
                graph
                    .apply(BuildCommand::Spawn(
                        CuboidSpec::new(
                            dimensions,
                            BuildPose::from_position_ticks(ticks, GridRotation::default()),
                        )
                        .unwrap(),
                    ))
                    .unwrap(),
            )
        };
        let base = spawn(base_ticks, [4, 4, 4]);
        let tip = spawn(base_ticks + offset, [1, 1, 1]);
        if grounded {
            graph
                .apply(BuildCommand::Weld(WeldSpec {
                    first: FaceRef::part(base, FaceKind::NegativeY),
                    second: FaceRef::ground(),
                }))
                .unwrap();
        }
        graph
            .apply(BuildCommand::AddBearing(
                BearingSpec::new(
                    FaceRef::part(base, source_face),
                    FaceRef::part(tip, target_face),
                    base_ticks.as_vec3() * 0.0025 + axis * 0.5,
                    axis,
                )
                .with_kind(mechanic_core::BearingKind::Suspension(spec)),
            ))
            .unwrap();
    }
    graph.compile().unwrap()
}

pub(super) fn suspension_test_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    creation: &mechanic_core::CompiledCreation,
) -> GpuPhysics {
    GpuPhysics::new_with_config(
        device,
        queue,
        creation,
        GpuPhysicsConfig {
            collisions_enabled: false,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
pub(super) fn suspension_loaded_spring_equilibrium_in_small_and_parallel_routes() {
    use mechanic_core::{ShockSpec, SpringSpec, SuspensionSpec};
    let (device, queue) = test_device().expect("suspension regression requires a GPU");
    let spring = SpringSpec::default();
    let spec = SuspensionSpec::new(Some(spring), Some(ShockSpec::default()), None).unwrap();
    for copies in [1, 65] {
        let creation = suspension_test_creation(spec, true, copies);
        assert_eq!(
            uses_fused_velocity_schedule(
                u32::try_from(creation.bearings.len()).unwrap(),
                u32::try_from(creation.compounds.len()).unwrap()
            ),
            copies == 1
        );
        let bearing = creation.bearings[0];
        let expected = creation.compounds[bearing.compound_b as usize]
            .mass_properties
            .mass
            * mechanic_core::STANDARD_GRAVITY_M_S2_F32
            / spring.rate();
        let gpu = suspension_test_gpu(&device, &queue, &creation);
        for tick in 1..=360 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        let (displacement, _) = linear_test_snapshot(&gpu, &device, &queue, &creation, 360);
        assert!(
            (-displacement - expected).abs() < expected * 0.1 + 0.0001,
            "copies {copies}: compression {}, equilibrium {expected}",
            -displacement
        );
        assert_mixed_linear_constraints(&gpu, &device, &queue, &creation, 360);
        eprintln!(
            "suspension copies {copies}: {:?}",
            gpu.read_last_tick(&device).unwrap()
        );
    }
}

#[test]
pub(super) fn suspension_damper_starting_stroke_is_force_free_and_rebound_is_stronger() {
    use mechanic_core::{ShockBodyEnd, ShockSpec, SuspensionSpec};
    let (device, queue) = test_device().expect("suspension regression requires a GPU");
    let spec = SuspensionSpec::new(
        None,
        Some(ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.075, 1.0, 1.6).unwrap()),
        None,
    )
    .unwrap();
    let creation = suspension_test_creation(spec, false, 1);
    let mut distances = Vec::new();
    for speed in [0.0_f32, -0.1, 0.1] {
        let gpu = suspension_test_gpu(&device, &queue, &creation);
        let bearing = creation.bearings[0];
        let body = &creation.compounds[bearing.compound_b as usize];
        if speed != 0.0 {
            gpu.apply_impulse(
                &device,
                &queue,
                bearing.compound_b,
                body.root_translation,
                Vec3::X * speed * body.mass_properties.mass,
            )
            .unwrap();
        }
        for tick in 1..=60 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        let (position, _) = linear_test_snapshot(&gpu, &device, &queue, &creation, 60);
        distances.push(position.abs());
        if speed == 0.0 {
            assert!(
                position.abs() < 0.00001,
                "stationary damper moved {position}"
            );
        } else {
            assert!(
                position * speed > 0.0
                    && position.abs() < speed.abs() * 60.0 * mechanic_core::TICK_SECONDS_F32 * 0.9,
                "damper failed to decay: speed {speed}, travel {position}"
            );
        }
    }
    assert!(
        distances[2] < distances[1] * 0.9,
        "compression/rebound travel {distances:?}"
    );
}

#[test]
pub(super) fn suspension_spring_oscillates_and_preload_reduces_loaded_compression() {
    use mechanic_core::{SpringSpec, SuspensionSpec};
    let (device, queue) = test_device().expect("suspension regression requires a GPU");
    let mut compressions = Vec::new();
    for preload in [0.0, 0.025] {
        let spring = SpringSpec::new(0.5, 0.16, 0.12, 6, preload).unwrap();
        let spec = SuspensionSpec::new(Some(spring), None, None).unwrap();
        let creation = suspension_test_creation(spec, true, 1);
        let gpu = suspension_test_gpu(&device, &queue, &creation);
        let mut samples = Vec::new();
        for tick in 1..=60 {
            gpu.dispatch_tick(&device, &queue, tick);
            samples.push(-linear_test_snapshot(&gpu, &device, &queue, &creation, tick).0);
        }
        if preload == 0.0 {
            assert!(samples.windows(2).any(|s| s[1] > s[0] + 0.00001));
            assert!(
                samples.windows(2).any(|s| s[1] < s[0] - 0.00001),
                "no spring rebound: {samples:?}"
            );
        }
        compressions.push(samples.iter().copied().fold(0.0_f32, f32::max));
    }
    assert!(
        compressions[1] < compressions[0] * 0.1,
        "preload had no effect: {compressions:?}"
    );
}

#[test]
pub(super) fn suspension_shock_bottoms_and_rubber_supports_load_in_both_body_orientations() {
    use mechanic_core::{BumpStopSpec, ShockBodyEnd, ShockSpec, SuspensionSpec};
    let (device, queue) = test_device().expect("suspension regression requires a GPU");
    let mut masses = Vec::new();
    for body_end in [ShockBodyEnd::Source, ShockBodyEnd::Opposite] {
        for with_stop in [false, true] {
            let shock = ShockSpec::new(0.5, 0.1, body_end, 0.0, 1.0, 1.6).unwrap();
            let stop = with_stop.then(|| BumpStopSpec::new(0.05, 0.06).unwrap());
            let spec = SuspensionSpec::new(None, Some(shock), stop).unwrap();
            let creation = suspension_test_creation(spec, true, 1);
            let bearing = creation.bearings[0];
            let mass = creation.compounds[bearing.compound_b as usize]
                .mass_properties
                .mass;
            if !with_stop {
                masses.push(mass);
            }
            let gpu = suspension_test_gpu(&device, &queue, &creation);
            for tick in 1..=360 {
                gpu.dispatch_tick(&device, &queue, tick);
            }
            let (position, _) = linear_test_snapshot(&gpu, &device, &queue, &creation, 360);
            let compression = -position;
            if with_stop {
                assert!(
                    compression > spec.bump_contact().unwrap(),
                    "rubber never contacted: {compression}"
                );
                assert!(
                    compression < spec.compression_limit().0,
                    "rubber reached crush limit under small load"
                );
                let force = spec.elastic_force(compression);
                assert!(
                    (force - mass * mechanic_core::STANDARD_GRAVITY_M_S2_F32).abs()
                        < mass * mechanic_core::STANDARD_GRAVITY_M_S2_F32 * 0.15,
                    "rubber load mismatch: force {force}, load {}",
                    mass * mechanic_core::STANDARD_GRAVITY_M_S2_F32
                );
            } else {
                assert!(
                    (compression - spec.compression_limit().0).abs() < 0.0001,
                    "shock failed to bottom: {compression}, limit {}",
                    spec.compression_limit().0
                );
            }
        }
    }
    assert!(
        masses[1] > masses[0],
        "body-end reversal did not move body mass: {masses:?}"
    );
}

#[test]
pub(super) fn suspension_passive_forces_preserve_mixed_loop_closures_in_both_routes() {
    use mechanic_core::{BearingKind, ShockSpec, SpringSpec, SuspensionSpec};
    let (device, queue) = test_device().expect("suspension regression requires a GPU");
    let spec = SuspensionSpec::new(
        Some(SpringSpec::default()),
        Some(ShockSpec::default()),
        None,
    )
    .unwrap();
    for copies in [1, 22] {
        // Reuse compiled mixed-loop topology to isolate the passive solver.
        // Graph placement and suspension mass ownership have separate fixtures.
        let mut creation = mixed_linear_creation(copies, true);
        for bearing in &mut creation.bearings {
            if bearing.kind.is_translational() {
                bearing.kind = BearingKind::Suspension(spec);
            }
        }
        assert_eq!(
            creation.loop_topology.closure_bearings.len(),
            usize::try_from(copies).unwrap()
        );
        let gpu = suspension_test_gpu(&device, &queue, &creation);
        let bearing = creation.bearings[0];
        let body = &creation.compounds[bearing.compound_b as usize];
        gpu.apply_impulse(
            &device,
            &queue,
            bearing.compound_b,
            body.root_translation,
            Vec3::NEG_Z * body.mass_properties.mass,
        )
        .unwrap();
        let mut minimum = 0.0_f32;
        for tick in 1..=90 {
            gpu.dispatch_tick(&device, &queue, tick);
            if tick % 5 == 0 {
                assert_mixed_linear_constraints(&gpu, &device, &queue, &creation, tick);
                minimum =
                    minimum.min(linear_test_snapshot(&gpu, &device, &queue, &creation, tick).0);
            }
        }
        assert!(
            minimum < -0.001,
            "loop suspension failed to compress: {minimum}"
        );
    }
}

#[test]
pub(super) fn suspension_floating_mount_receives_bottom_out_reaction() {
    use mechanic_core::{ShockBodyEnd, ShockSpec, SuspensionSpec};
    let (device, queue) = test_device().expect("suspension regression requires a GPU");
    let spec = SuspensionSpec::new(
        None,
        Some(ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.075, 0.0, 0.0).unwrap()),
        None,
    )
    .unwrap();
    for copies in [1, 65] {
        let creation = suspension_test_creation_with_anchor(spec, false, copies, false);
        let gpu = suspension_test_gpu(&device, &queue, &creation);
        let bearing = creation.bearings[0];
        let tip = &creation.compounds[bearing.compound_b as usize];
        let base = &creation.compounds[bearing.compound_a as usize];
        gpu.apply_impulse(
            &device,
            &queue,
            bearing.compound_b,
            tip.root_translation,
            Vec3::NEG_X * 20.0 * tip.mass_properties.mass,
        )
        .unwrap();
        for tick in 1..=15 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        let (position, snapshot) = linear_test_snapshot(&gpu, &device, &queue, &creation, 15);
        assert!(
            (position - spec.bounds()[0]).abs() < 0.0001,
            "floating shock did not bottom: {position}"
        );
        let movement =
            transform_position(snapshot[bearing.compound_a as usize]) - base.root_translation;
        assert!(
            movement.x < -0.0001,
            "floating mount received no reaction: {movement:?}"
        );
    }
}

#[test]
#[expect(clippy::float_cmp, reason = "held velocities must be exactly zero")]
pub(super) fn suspension_holds_suppress_passive_forces_and_release_restores_spring_motion() {
    use mechanic_core::{ShockBodyEnd, ShockSpec, SpringSpec, SuspensionSpec};
    let (device, queue) = test_device().expect("suspension regression requires a GPU");
    let spec = SuspensionSpec::new(
        Some(SpringSpec::default()),
        Some(ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.05, 1.0, 1.6).unwrap()),
        None,
    )
    .unwrap();
    for copies in [1, 65] {
        let creation = suspension_test_creation_with_anchor(spec, false, copies, false);
        let gpu = suspension_test_gpu(&device, &queue, &creation);
        gpu.enable_async_readback();
        let poses = creation
            .compounds
            .iter()
            .map(|body| crate::GpuTransform {
                position: body.root_translation.extend(0.0).to_array(),
                rotation: body.root_rotation.to_array(),
            })
            .collect::<Vec<_>>();
        let mut holds = vec![true; poses.len()];
        gpu.set_body_holds(&queue, &holds).unwrap();
        gpu.prescribe_held_poses(&queue, &poses).unwrap();
        for tick in 1..=3 {
            gpu.dispatch_tick(&device, &queue, tick);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let state = gpu.poll_tick_readback(&device).unwrap().unwrap();
            assert_eq!(state.diagnostics.error_flags, 0);
            assert_eq!(state.transforms, poses);
            assert!(
                state
                    .velocities
                    .iter()
                    .all(|v| v.linear == [0.0; 4] && v.angular == [0.0; 4])
            );
        }
        let bearing = creation.bearings[0];
        holds[bearing.compound_a as usize] = false;
        holds[bearing.compound_b as usize] = false;
        gpu.set_body_holds(&queue, &holds).unwrap();
        let mut last = None;
        for tick in 4..=33 {
            gpu.dispatch_tick(&device, &queue, tick);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let state = gpu.poll_tick_readback(&device).unwrap().unwrap();
            assert_eq!(state.diagnostics.error_flags, 0);
            for (index, held) in holds.iter().enumerate() {
                if *held {
                    assert_eq!(state.transforms[index], poses[index]);
                }
            }
            last = Some(state);
        }
        let state = last.unwrap();
        let base = bearing.compound_a as usize;
        assert!(
            state.transforms[base].position[0] < poses[base].position[0] - 0.00001,
            "released mount received no spring reaction"
        );
        let tip = bearing.compound_b as usize;
        assert!(
            state.transforms[tip].position[0] > poses[tip].position[0] + 0.02,
            "released spring did not extend from initial compression"
        );
    }
}

#[test]
pub(super) fn suspension_rubber_crush_limit_resists_sustained_overload_in_both_routes() {
    use mechanic_core::{BumpStopSpec, ShockSpec, SuspensionSpec};
    let (device, queue) = test_device().expect("suspension regression requires a GPU");
    let spec = SuspensionSpec::new(
        None,
        Some(ShockSpec::default()),
        Some(BumpStopSpec::new(0.05, 0.06).unwrap()),
    )
    .unwrap();
    let force = spec.elastic_force(spec.compression_limit().0) * 4.0;
    for copies in [1, 65] {
        let creation = suspension_test_creation(spec, false, copies);
        let gpu = suspension_test_gpu(&device, &queue, &creation);
        let bearing = creation.bearings[0];
        let body = &creation.compounds[bearing.compound_b as usize];
        let mut contact_point = body.root_translation;
        let mut position = 0.0;
        for tick in 1..=60 {
            gpu.apply_impulse(
                &device,
                &queue,
                bearing.compound_b,
                contact_point,
                Vec3::NEG_X * force * mechanic_core::TICK_SECONDS_F32,
            )
            .unwrap();
            gpu.dispatch_tick(&device, &queue, tick);
            let (displacement, snapshot) =
                linear_test_snapshot(&gpu, &device, &queue, &creation, tick);
            position = displacement;
            contact_point = transform_position(snapshot[bearing.compound_b as usize]);
        }
        assert!(
            (-position - spec.compression_limit().0).abs() < 0.0001,
            "rubber failed to stop at 55% crush: {} vs {}",
            -position,
            spec.compression_limit().0
        );
    }
}
