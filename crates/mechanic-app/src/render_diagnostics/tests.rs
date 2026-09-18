use super::*;
use bevy::core_pipeline::Core3d;
use bevy::render::renderer::{CurrentView, RenderContext};

#[derive(Resource)]
struct Raster {
    pipeline: wgpu::RenderPipeline,
    target: wgpu::TextureView,
    resolved: wgpu::TextureView,
    explicit: Option<wgpu::QuerySet>,
}

impl Raster {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raster timing reference"),
            source: wgpu::ShaderSource::Wgsl(
                r"
                @vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                    let p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
                    return vec4(p[i], 0.0, 1.0);
                }
                @fragment fn fragment(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                    var v = p.xy * 0.001;
                    for (var i = 0u; i < 128u; i++) { v = sin(v.yx * 1.01 + vec2(0.13, 0.17)); }
                    return vec4(v, 0.5, 1.0);
                }
            "
                .into(),
            ),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raster timing reference"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 4,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let texture = |samples| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("raster timing target"),
                    size: wgpu::Extent3d {
                        width: 1024,
                        height: 1024,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let target = texture(4);
        let resolved = texture(1);

        Self {
            pipeline,
            target,
            resolved,
            explicit: None,
        }
    }
}

#[derive(Resource)]
struct TestViews(Vec<Entity>);

fn test_views(world: &mut World) {
    for entity in world.resource::<TestViews>().0.clone() {
        world.insert_resource(CurrentView(entity));
        world.run_schedule(Core3d);
    }
}

fn raster_work(raster: Res<Raster>, mut ctx: RenderContext) {
    let mut pass = ctx.begin_tracked_render_pass(wgpu::RenderPassDescriptor {
        label: Some("main_opaque_pass_3d"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &raster.target,
            depth_slice: None,
            resolve_target: Some(&raster.resolved),
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Discard,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: raster
            .explicit
            .as_ref()
            .map(|queries| wgpu::RenderPassTimestampWrites {
                query_set: queries,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.wgpu_pass().set_pipeline(&raster.pipeline);
    pass.draw(0..3, 0..1);
}

fn timestamp_adapter() -> Option<wgpu::Adapter> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Ok(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    else {
        eprintln!("SKIPPED render timing test: no adapter");
        return None;
    };
    if !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        eprintln!("SKIPPED render timing test: timestamps unsupported");
        return None;
    }
    eprintln!("Render timing adapter: {:?}", adapter.get_info());
    Some(adapter)
}

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let adapter = timestamp_adapter()?;
    Some(
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            ..Default::default()
        }))
        .unwrap(),
    )
}

fn test_app(device: &wgpu::Device, queue: &wgpu::Queue) -> App {
    let mut app = App::new();
    let mut render = SubApp::new();
    render.add_schedule(RenderGraph::base_schedule());
    render.add_schedule(Core3d::base_schedule());
    let world = render.world_mut().spawn(ProfiledCamera).id();
    let xray = render.world_mut().spawn(MosaicCamera).id();
    render.insert_resource(TestViews(vec![world, xray]));
    render.insert_resource(Raster::new(device));
    render.insert_resource(RenderDevice::from(device.clone()));
    render.insert_resource(RenderQueue(Arc::new(
        bevy::render::renderer::WgpuWrapper::new(queue.clone()),
    )));
    render.init_resource::<PendingCommandBuffers>();
    render.add_systems(RenderGraph, test_views.in_set(RenderGraphSystems::Render));
    render.add_systems(Core3d, raster_work);
    app.insert_sub_app(RenderApp, render);
    app.add_plugins(RenderTimingsPlugin);
    app
}

fn submit_and_wait(render: &mut SubApp, device: &wgpu::Device, queue: &wgpu::Queue) {
    queue.submit(
        render
            .world_mut()
            .resource_mut::<PendingCommandBuffers>()
            .take(),
    );
    // Blocking polls are test-only. Production uses asynchronous readback callbacks.
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

#[test]
#[allow(clippy::too_many_lines)] // Keep the real renderer setup and regression together.
fn bevy_pbr_keeps_opaque_timings_when_a_shadow_timestamp_is_missing() {
    use bevy::camera::RenderTarget;
    use bevy::render::{RenderPlugin, pipelined_rendering::PipelinedRenderingPlugin};
    use bevy::window::ExitCondition;

    let Some(_adapter) = timestamp_adapter() else {
        return;
    };

    // Exercise real extraction, PBR passes and diagnostics, not a hand-built graph.
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .set(RenderPlugin {
                synchronous_pipeline_compilation: true,
                ..default()
            })
            .disable::<bevy::winit::WinitPlugin>()
            .disable::<PipelinedRenderingPlugin>()
            .disable::<bevy::log::LogPlugin>(),
    )
    .add_plugins((
        bevy::render::diagnostic::RenderDiagnosticsPlugin,
        RenderTimingsPlugin,
    ));
    app.finish();
    app.cleanup();
    let target = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(Image::new_target_texture(
            256,
            256,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            None,
        ));
    app.world_mut().spawn((
        Camera3d::default(),
        RenderTarget::Image(target.into()),
        ProfiledCamera,
        Msaa::Sample4,
        Transform::from_xyz(0.0, 2.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    let mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Cuboid::default());
    let material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    app.world_mut()
        .spawn((Mesh3d(mesh), MeshMaterial3d(material)));
    app.world_mut().spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(3.0, 4.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    app.world().resource::<RenderTimings>().set_enabled(true);
    for _ in 0..12 {
        app.update();
        let render = app.sub_app_mut(RenderApp);
        let device = render.world().resource::<RenderDevice>();
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    eprintln!(
        "PBR adapter: {:?}",
        app.sub_app(RenderApp)
            .world()
            .resource::<bevy::render::renderer::RenderAdapterInfo>()
            .0
    );
    let snapshot = app.world().resource::<RenderTimings>().snapshot();
    eprintln!("PBR snapshot: {snapshot:?}");
    assert!(
        snapshot.breakdown.opaque_ms.is_some(),
        "PBR opaque timings unavailable"
    );
    assert!(
        snapshot.health.total_pairs >= 5,
        "real shadow cascades must be recorded"
    );
    if snapshot.health.valid_pairs < snapshot.health.total_pairs {
        // Metal can leave the end timestamp unwritten on an empty cascade.
        assert!(matches!(
            snapshot.health.status,
            GpuSampleStatus::Partial(_)
        ));
        assert_eq!(snapshot.breakdown.prepass_ms, None);
        assert_eq!(snapshot.render_gpu_ms, None);
    } else {
        // Backends that timestamp empty passes can publish the entire sample.
        assert_eq!(snapshot.health.status, GpuSampleStatus::Complete);
        assert!(snapshot.breakdown.prepass_ms.is_some());
        assert!(snapshot.render_gpu_ms.is_some());
    }
}

#[test]
fn real_msaa_passes_are_timed_with_bounded_slots_and_no_stale_view_samples() {
    let Some((device, queue)) = gpu() else {
        return;
    };
    let mut app = test_app(&device, &queue);
    let render = app.sub_app_mut(RenderApp);
    render.world_mut().run_schedule(RenderGraph);
    assert!(render.world().resource::<Probe>().slots.is_empty());
    render.world().resource::<RenderTimings>().set_enabled(true);
    for _ in 0..=SLOTS {
        render.world_mut().run_schedule(RenderGraph);
    }
    let probe = render.world().resource::<Probe>();
    assert_eq!(probe.slots.len(), SLOTS);
    assert_eq!(probe.next_frame, SLOTS as u64);
    assert!(probe.active.is_none());
    for slot in &probe.slots {
        assert_eq!(slot.groups, [PassGroup::Opaque, PassGroup::Xray]);
        assert!(!slot.overflow);
    }
    assert!(
        render
            .world()
            .resource::<RenderTimings>()
            .snapshot()
            .render_gpu_ms
            .is_none()
    );
    submit_and_wait(render, &device, &queue);
    render
        .world()
        .resource::<RenderTimings>()
        .set_enabled(false);
    render.world_mut().run_schedule(RenderGraph);
    assert_eq!(render.world().resource::<Probe>().next_frame, SLOTS as u64);
    let snapshot = render.world().resource::<RenderTimings>().snapshot();
    eprintln!("Actual 1024x1024 4x MSAA raster pass timings: {snapshot:?}");
    assert!(snapshot.breakdown.opaque_ms.is_some_and(|ms| ms > 0.0));
    assert!(snapshot.breakdown.xray_ms.is_some_and(|ms| ms > 0.0));
    assert!(snapshot.render_gpu_ms.unwrap() >= snapshot.breakdown.opaque_ms.unwrap());
    assert_eq!(snapshot.breakdown.ui_ms, None);
    assert_eq!(snapshot.breakdown.post_ms, None);
    assert_eq!(snapshot.breakdown.other_ms, None);

    // Query-set contents persist, but a new frame without views must publish N/A.
    render.world_mut().resource_mut::<TestViews>().0.clear();
    render.world().resource::<RenderTimings>().set_enabled(true);
    render.world_mut().run_schedule(RenderGraph);
    let snapshot = render.world().resource::<RenderTimings>().snapshot();
    assert_eq!(snapshot.render_gpu_ms, None);
    assert_eq!(snapshot.breakdown, GpuBreakdown::default());
    submit_and_wait(render, &device, &queue);
}

#[test]
fn existing_descriptor_timestamps_are_preserved_and_allocator_is_bounded() {
    let Some((device, queue)) = gpu() else {
        return;
    };
    let mut app = test_app(&device, &queue);
    let render = app.sub_app_mut(RenderApp);
    let reference = Slot::new(&device);
    render.world_mut().resource_mut::<TestViews>().0.truncate(1);
    render.world_mut().resource_mut::<Raster>().explicit = Some(reference.queries.clone());
    render.world().resource::<RenderTimings>().set_enabled(true);
    render.world_mut().run_schedule(RenderGraph);
    assert!(
        render
            .world()
            .resource::<Probe>()
            .slots
            .iter()
            .all(|slot| slot.groups.is_empty())
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.resolve_query_set(&reference.queries, 0..2, &reference.resolve, 0);
    encoder.copy_buffer_to_buffer(&reference.resolve, 0, &reference.readback, 0, 16);
    encoder.map_buffer_on_submit(&reference.readback, wgpu::MapMode::Read, .., |result| {
        result.unwrap();
    });
    render
        .world_mut()
        .resource_mut::<PendingCommandBuffers>()
        .push([encoder.finish()]);
    submit_and_wait(render, &device, &queue);
    {
        let bytes = reference.readback.slice(..).get_mapped_range();
        assert!(
            query_interval(&bytes, 0).is_ok(),
            "explicit descriptor query pair was replaced"
        );
    }
    reference.readback.unmap();

    let recording = render.world().resource::<Recording>();
    assert!(recording.allocate(None, None).is_none());
    *recording.0.lock().unwrap() = Some(ActiveRecording {
        queries: reference.queries.clone(),
        groups: Vec::new(),
        views: Vec::new(),
        overflow: false,
    });
    for index in 0..QUERY_COUNT / 2 {
        let allocation = recording.allocate(None, None).unwrap();
        assert_eq!(allocation.beginning, index * 2);
        assert_eq!(allocation.end, index * 2 + 1);
    }
    assert!(recording.allocate(None, None).is_none());
    assert!(recording.0.lock().unwrap().as_ref().unwrap().overflow);
}

#[test]
fn breakdown_unions_overlapping_passes_and_does_not_invent_unmeasured_work() {
    let bytes: Vec<_> = [1_u64, 11, 6, 16, 30, 40]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect();
    let (span, breakdown, health) = read_breakdown(
        &bytes,
        &[PassGroup::Opaque, PassGroup::Opaque, PassGroup::Xray],
        1_000_000.0,
    );
    assert_eq!(span, Some(39.0));
    assert_eq!(
        health,
        GpuSampleHealth {
            status: GpuSampleStatus::Complete,
            valid_pairs: 3,
            total_pairs: 3
        }
    );
    assert_eq!(breakdown.opaque_ms, Some(15.0));
    assert_eq!(breakdown.xray_ms, Some(10.0));
    assert_eq!(breakdown.other_ms, None); // The gap is not another measured pass.
    assert_eq!(breakdown.ui_ms, None);
    assert_eq!(breakdown.world_output_ms, None);
    assert_eq!(
        read_breakdown(&bytes, &[], 1.0),
        unavailable_sample(GpuSampleStatus::NoPasses, 0)
    );
    let (_, single, _) = read_breakdown(&bytes, &[PassGroup::Opaque], 1_000_000.0);
    assert_eq!(single.opaque_ms, Some(10.0));
    assert_eq!(single.xray_ms, None); // Ignore stale trailing query contents.
}

#[test]
fn invalid_or_incomplete_samples_are_unavailable() {
    for (start, end, error) in [
        (0_u64, 10_u64, TimestampError::MissingStart),
        (10, 0, TimestampError::MissingEnd),
        (11, 10, TimestampError::Reversed),
    ] {
        let bytes: Vec<_> = [start, end]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect();
        let (span, breakdown, health) = read_breakdown(&bytes, &[PassGroup::Opaque], 1.0);
        assert_eq!(span, None);
        assert_eq!(breakdown, GpuBreakdown::default());
        assert_eq!(health.status, GpuSampleStatus::Partial(error));
        assert_eq!((health.valid_pairs, health.total_pairs), (0, 1));
    }
    let valid: Vec<_> = [1_u64, 2].into_iter().flat_map(u64::to_le_bytes).collect();
    for period in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            read_breakdown(&valid, &[PassGroup::Opaque], period),
            unavailable_sample(GpuSampleStatus::InvalidPeriod, 1)
        );
    }
    assert_eq!(
        read_breakdown(&valid, &[PassGroup::Opaque, PassGroup::Xray], 1.0).0,
        None
    );
    assert_eq!(
        read_breakdown(&valid[..8], &[PassGroup::Opaque], 1.0).0,
        None
    );
    assert_eq!(
        query_interval(&valid[..8], 0),
        Err(TimestampError::Truncated)
    );
}

#[test]
fn missing_shadow_end_invalidates_only_its_group_and_the_total_span() {
    let bytes: Vec<_> = [1_u64, 10, 12, 0, 20, 50, 60, 60]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect();
    let (span, breakdown, health) = read_breakdown(
        &bytes,
        &[
            PassGroup::Prepass,
            PassGroup::Prepass,
            PassGroup::Opaque,
            PassGroup::Xray,
        ],
        1_000_000.0,
    );
    assert_eq!(span, None);
    assert_eq!(breakdown.prepass_ms, None); // Not the partial 9ms sum.
    assert_eq!(breakdown.opaque_ms, Some(30.0));
    assert_eq!(breakdown.xray_ms, Some(0.0)); // Equal nonzero timestamps are valid.
    assert_eq!(
        health,
        GpuSampleHealth {
            status: GpuSampleStatus::Partial(TimestampError::MissingEnd),
            valid_pairs: 3,
            total_pairs: 4,
        }
    );

    // A query from a previous use of this slot must not survive a missing end.
    let stale: Vec<_> = [100_u64, 10, 120, 150]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect();
    let (_, breakdown, health) = read_breakdown(
        &stale,
        &[PassGroup::Prepass, PassGroup::Opaque],
        1_000_000.0,
    );
    assert_eq!(breakdown.prepass_ms, None);
    assert_eq!(breakdown.opaque_ms, Some(30.0));
    assert_eq!(
        health.status,
        GpuSampleStatus::Partial(TimestampError::Reversed)
    );
}

#[test]
fn camera_and_pass_labels_keep_world_xray_and_shadows_separate() {
    assert_eq!(
        pass_group("main_opaque_pass_3d", true, false),
        PassGroup::Opaque
    );
    assert_eq!(
        pass_group("main_opaque_pass_3d", false, true),
        PassGroup::Xray
    );
    assert_eq!(
        pass_group("main_opaque_pass_3d", false, false),
        PassGroup::Other
    );
    assert_eq!(
        pass_group("shadow_spot_light_0", false, false),
        PassGroup::Prepass
    );
    assert_eq!(pass_group("early_prepass", true, false), PassGroup::Prepass);
    assert_eq!(
        pass_group("main_transparent_pass_3d", true, false),
        PassGroup::Transparent
    );
}
