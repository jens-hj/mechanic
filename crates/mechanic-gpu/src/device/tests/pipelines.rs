//! Schedules, buffer sizing, pipeline reuse, and timestamp ordering.

use super::*;

#[test]
pub(super) fn fused_small_mechanism_schedule_has_explicit_size_and_scene_boundaries() {
    assert!(uses_fused_velocity_schedule(64, 256));
    assert!(!uses_fused_velocity_schedule(65, 256));
    assert!(!uses_fused_velocity_schedule(64, 257));

    assert!(uses_fused_contact_schedule(4, true));
    assert!(uses_fused_contact_schedule(64, true));
    assert!(!uses_fused_contact_schedule(65, true));
    assert!(uses_fused_contact_schedule(64, false));
    assert!(!uses_fused_contact_schedule(65, false));
}

#[test]
pub(super) fn collision_buffers_scale_with_the_scene_up_to_the_hard_limit() {
    assert_eq!(contact_pair_capacity(0), 256);
    assert_eq!(contact_pair_capacity(1), 256);
    assert_eq!(contact_pair_capacity(16), 256);
    assert_eq!(contact_pair_capacity(1_024), 1_048_576);
    assert_eq!(
        contact_pair_capacity(crate::MAX_COLLIDERS),
        u32::try_from(crate::MAX_CONTACT_PAIRS).unwrap()
    );
}

#[test]
pub(super) fn replacement_scene_reuses_every_compiled_shader_and_pipeline() {
    let Some((device, queue)) = test_device() else {
        return;
    };
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap();
    let creation = graph.compile().unwrap();
    let pipelines = GpuPhysicsPipelines::new();
    let first = GpuPhysics::new_with_pipelines(
        &device,
        &queue,
        &creation,
        GpuPhysicsConfig::default(),
        &pipelines,
    )
    .unwrap();
    let shader_count = pipelines.shaders.lock().unwrap().len();
    let pipeline_count = pipelines.pipelines.lock().unwrap().len();
    drop(first);

    let _replacement = GpuPhysics::new_with_pipelines(
        &device,
        &queue,
        &creation,
        GpuPhysicsConfig::default(),
        &pipelines,
    )
    .unwrap();

    assert!(shader_count > 0);
    assert!(pipeline_count > shader_count);
    assert_eq!(pipelines.shaders.lock().unwrap().len(), shader_count);
    assert_eq!(pipelines.pipelines.lock().unwrap().len(), pipeline_count);
}

#[test]
pub(super) fn gated_recovery_timestamps_remain_ordered() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .unwrap();
    if !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        return;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::TIMESTAMP_QUERY,
        ..Default::default()
    }))
    .unwrap();
    let mut gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        &pendulum_creation(false),
        GpuPhysicsConfig {
            ground_plane_enabled: false,
            ..Default::default()
        },
    )
    .unwrap();
    let mut chunk = super::super::terrain::tests::rigid_chunk();
    chunk.origin.0.y -= 100.0;
    gpu.write_terrain_chunks(&device, &queue, [&chunk], bevy_math::DVec3::ZERO)
        .unwrap();
    for tick in 1..=30 {
        gpu.dispatch_tick(&device, &queue, tick);
        let result = gpu.read_last_tick(&device).unwrap();
        assert_eq!(result.error_flags, 0);
        assert_eq!(result.contact_count, 0);
        let stages = result.kernel_timings.unwrap();
        assert!(stages.terrain_recovery_ms > 0.0);
        assert!(
            stages.recovery_projection_ms <= stages.terrain_recovery_ms,
            "invalid nested span: {stages:?}"
        );
        assert!(stages.terrain_recovery_ms <= result.gpu_tick_ms.unwrap());
    }
}

#[test]
pub(super) fn gpu_pipelines_construct_on_noop_backend() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
        label: Some("mechanic pipeline validation device"),
        ..Default::default()
    });
    let creation = pendulum_creation(true);
    for mechanism_self_collisions in [true, false] {
        GpuPhysics::new_with_config(
            &device,
            &queue,
            &creation,
            GpuPhysicsConfig {
                collisions_enabled: true,
                ground_plane_enabled: true,
                mechanism_self_collisions,
                solver_iterations: 8,
            },
        )
        .unwrap();
    }
}
