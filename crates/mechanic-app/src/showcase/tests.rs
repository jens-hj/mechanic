use bevy::prelude::{Quat, Vec3};

use super::{
    BEARING_COUNT, COMPOUND_COUNT, CreationPreset, PART_COUNT, WELD_COUNT, build, build_preset,
    uses_reduced_collision_mode,
};
use mechanic_core::BearingDimensions;
use mechanic_gpu::{GpuPhysics, GpuPhysicsConfig, MAX_BEARINGS, MAX_BODIES, MAX_COLLIDERS};

#[test]
fn showcase_has_exact_deterministic_topology() {
    const COMPACTED_COLLIDER_COUNT: usize = 5_609;
    let graph = build().unwrap();
    let creation = graph.compile().unwrap();

    assert_eq!(graph.part_count(), PART_COUNT);
    assert_eq!(graph.weld_count(), WELD_COUNT);
    assert_eq!(graph.bearing_count(), BEARING_COUNT);
    assert_eq!(creation.colliders.len(), COMPACTED_COLLIDER_COUNT);
    assert_eq!(creation.compounds.len(), COMPOUND_COUNT);
    assert_eq!(creation.bearings.len(), BEARING_COUNT);
    assert!(
        graph
            .bearings()
            .all(|(_, bearing)| bearing.dimensions == BearingDimensions::default())
    );
    assert_eq!(creation.loop_topology.tree_bearings.len(), BEARING_COUNT);
    assert!(creation.loop_topology.closure_bearings.is_empty());
    assert!(graph.pending().is_none());
    assert_eq!(
        creation
            .loop_topology
            .mechanism_components
            .iter()
            .filter(|component| component.len() > 1)
            .count(),
        1
    );
    assert!(
        creation
            .compounds
            .iter()
            .any(|compound| { compound.is_static && compound.source_parts.len() == 14_416 })
    );
    assert!(creation.compounds.iter().all(|compound| {
        let mass = compound.mass_properties;
        mass.mass.is_finite()
            && mass.center_of_mass.is_finite()
            && mass.inertia.is_finite()
            && mass.inverse_inertia.is_finite()
    }));
    assert!(creation.compounds.len() <= MAX_BODIES);
    assert!(creation.colliders.len() <= MAX_COLLIDERS);
    assert!(creation.bearings.len() <= MAX_BEARINGS);
    assert!(uses_reduced_collision_mode(&graph));
    assert_no_initial_intersections(&graph);
}

#[test]
fn smaller_creation_presets_have_exact_valid_topology() {
    for preset in CreationPreset::ALL[..3].iter().copied() {
        let graph = build_preset(preset).unwrap();
        let creation = graph.compile().unwrap();
        let (component_count, component_size, closure_count) = match preset {
            CreationPreset::PendulumGarden256 => (64, 4, 0),
            CreationPreset::MobileWorkshop1024 => (128, 5, 0),
            CreationPreset::ClosureLab4096 => (512, 4, 512),
            CreationPreset::KineticShowcase20000 => unreachable!(),
        };

        assert_eq!(graph.part_count(), preset.part_count());
        assert_eq!(graph.weld_count(), preset.weld_count());
        assert_eq!(graph.bearing_count(), preset.bearing_count());
        assert_eq!(creation.compounds.len(), preset.body_count());
        assert_eq!(creation.bearings.len(), preset.bearing_count());
        assert_eq!(
            creation.loop_topology.tree_bearings.len(),
            preset.bearing_count() - closure_count
        );
        assert_eq!(creation.loop_topology.closure_bearings.len(), closure_count);
        assert_eq!(
            creation
                .loop_topology
                .mechanism_components
                .iter()
                .filter(|component| component.len() == component_size)
                .count(),
            component_count
        );
        assert_bearings_are_centered_on_face_overlaps(&graph);
        assert!(!uses_reduced_collision_mode(&graph));
        assert_no_initial_intersections(&graph);
    }
}

#[test]
fn smaller_creation_presets_run_120_gpu_ticks_without_failure() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Some(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()
    else {
        return;
    };
    let Some((device, queue)) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mechanic creation preset test device"),
            ..Default::default()
        }))
        .ok()
    else {
        return;
    };

    for preset in CreationPreset::ALL[..3].iter().copied() {
        let creation = build_preset(preset).unwrap().compile().unwrap();
        let gpu = GpuPhysics::new_with_config(
            &device,
            &queue,
            &creation,
            GpuPhysicsConfig {
                mechanism_self_collisions: true,
                ..GpuPhysicsConfig::default()
            },
        )
        .unwrap();
        for tick in 1..=120 {
            gpu.dispatch_tick(&device, &queue, tick);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let diagnostics = gpu.read_last_tick(&device).unwrap();
            assert_eq!(
                diagnostics.error_flags,
                0,
                "{} tick {tick} diagnostics: {diagnostics:?}",
                preset.label()
            );
        }
        let snapshot = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
        let moved_body_count = snapshot
            .iter()
            .zip(&creation.compounds)
            .filter(|(transform, compound)| {
                let position = Vec3::from_slice(&transform.position[..3]);
                let rotation = Quat::from_array(transform.rotation);
                position.distance(compound.root_translation) > 1.0e-4
                    || rotation.angle_between(compound.root_rotation) > 1.0e-4
            })
            .count();
        assert!(
            moved_body_count > 0,
            "{} did not produce any simulated motion",
            preset.label()
        );
    }
}

#[test]
fn dynamic_presets_do_not_gain_unbounded_spin() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Some(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()
    else {
        return;
    };
    let Some((device, queue)) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    else {
        return;
    };

    for preset in [
        CreationPreset::MobileWorkshop1024,
        CreationPreset::ClosureLab4096,
    ] {
        let creation = build_preset(preset).unwrap().compile().unwrap();
        let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
        for tick in 1..=600 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let current = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
        let previous = gpu.read_snapshot_transforms(&device, &queue, 2).unwrap();
        let maximum_angular_speed = current
            .iter()
            .zip(previous)
            .map(|(current, previous)| {
                Quat::from_array(current.rotation)
                    .angle_between(Quat::from_array(previous.rotation))
                    * 60.0
            })
            .fold(0.0_f32, f32::max);
        let diagnostics = gpu.read_last_tick(&device).unwrap();
        assert_eq!(diagnostics.error_flags, 0, "{}", preset.label());
        assert!(
            maximum_angular_speed < 3.0,
            "{} reached {maximum_angular_speed:.3} rad/s",
            preset.label()
        );
    }
}

fn assert_bearings_are_centered_on_face_overlaps(graph: &mechanic_core::ConstructionGraph) {
    for (_, bearing) in graph.bearings() {
        let source = match bearing.source.owner {
            mechanic_core::FaceOwner::Part(part) => graph.part(part).unwrap(),
            mechanic_core::FaceOwner::Ground => unreachable!(),
        };
        let target = match bearing.target.owner {
            mechanic_core::FaceOwner::Part(part) => graph.part(part).unwrap(),
            mechanic_core::FaceOwner::Ground => unreachable!(),
        };
        let source_center = source.pose().translation();
        let target_center = target.pose().translation();
        let source_half = source.size_meters() * 0.5;
        let target_half = target.size_meters() * 0.5;
        let source_minimum = source_center - source_half;
        let source_maximum = source_center + source_half;
        let target_minimum = target_center - target_half;
        let target_maximum = target_center + target_half;
        let normal_axis = (0..3).find(|&axis| bearing.axis[axis].abs() > 0.5).unwrap();
        let mut expected = source_center + bearing.axis * source_half[normal_axis];
        for axis in 0..3 {
            if axis != normal_axis {
                expected[axis] = (source_minimum[axis].max(target_minimum[axis])
                    + source_maximum[axis].min(target_maximum[axis]))
                    * 0.5;
            }
        }
        assert!(
            bearing.shared_anchor.abs_diff_eq(expected, 1.0e-5),
            "bearing anchor {:?} is not centered at {:?}",
            bearing.shared_anchor,
            expected
        );
    }
}

#[test]
fn showcase_runs_1_200_gpu_ticks_without_failure_or_blowup() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Some(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()
    else {
        return;
    };
    let Some((device, queue)) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mechanic showcase test device"),
            ..Default::default()
        }))
        .ok()
    else {
        return;
    };
    let creation = build().unwrap().compile().unwrap();
    let gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        &creation,
        GpuPhysicsConfig {
            mechanism_self_collisions: false,
            ..GpuPhysicsConfig::default()
        },
    )
    .unwrap();

    for tick in 1..=1_200 {
        gpu.dispatch_tick(&device, &queue, tick);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let diagnostics = gpu.read_last_tick(&device).unwrap();
        assert_eq!(
            diagnostics.error_flags, 0,
            "showcase tick {tick} diagnostics: {diagnostics:?}"
        );
    }
    let snapshot = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
    let previous_snapshot = gpu.read_snapshot_transforms(&device, &queue, 2).unwrap();
    let mut maximum_displacement = 0.0_f32;
    let mut maximum_displacement_body = 0_usize;
    let mut maximum_speed = 0.0_f32;
    let mut maximum_speed_body = 0_usize;
    for (body, (transform, compound)) in snapshot.iter().zip(&creation.compounds).enumerate() {
        if compound.is_static || creation.loop_topology.body_parents[body].is_root {
            continue;
        }
        let position = Vec3::from_slice(&transform.position[..3]);
        let displacement = position.distance(compound.root_translation);
        if displacement > maximum_displacement {
            maximum_displacement = displacement;
            maximum_displacement_body = body;
        }
        let previous_position = Vec3::from_slice(&previous_snapshot[body].position[..3]);
        let speed = position.distance(previous_position) * 60.0;
        if speed > maximum_speed {
            maximum_speed = speed;
            maximum_speed_body = body;
        }
        assert!(
            position.is_finite(),
            "showcase body {body} has invalid position {position:?}"
        );
    }
    assert!(
        maximum_displacement < 5.0,
        "showcase body {maximum_displacement_body} moved {maximum_displacement} metres"
    );
    assert!(
        maximum_speed < 2.0,
        "showcase body {maximum_speed_body} reached {maximum_speed} m/s"
    );
}

fn assert_no_initial_intersections(graph: &mechanic_core::ConstructionGraph) {
    const EPSILON: f32 = 1.0e-5;

    let mut bounds = graph
        .parts()
        .map(|(part, spec)| {
            let half = spec.size_meters() * 0.5;
            (
                part,
                spec.pose().translation() - half,
                spec.pose().translation() + half,
            )
        })
        .collect::<Vec<_>>();
    bounds.sort_unstable_by(|left, right| left.1.x.total_cmp(&right.1.x));

    for (index, &(part, minimum, maximum)) in bounds.iter().enumerate() {
        for &(other_part, other_minimum, other_maximum) in &bounds[index + 1..] {
            if other_minimum.x >= maximum.x - EPSILON {
                break;
            }
            let overlap = minimum.cmplt(other_maximum - Vec3::splat(EPSILON)).all()
                && other_minimum.cmplt(maximum - Vec3::splat(EPSILON)).all();
            assert!(
                !overlap,
                "showcase parts {part:?} and {other_part:?} initially intersect"
            );
        }
    }
}
