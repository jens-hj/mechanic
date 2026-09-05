//! GPU timestamps attached to actual tracked graphics passes; CPU presentation spans.
//! Raw passes remain unmeasured. No marker draws, extra submissions, or CPU waits.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU8, Ordering},
};
use std::time::Instant;

use bevy::prelude::*;
use bevy::render::{
    Render, RenderApp, RenderSystems,
    camera::ExtractedCamera,
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    renderer::{
        PendingCommandBuffers, RenderDevice, RenderGraph, RenderGraphSystems,
        RenderPassTimestampAllocation, RenderPassTimestampHook, RenderQueue, render_system,
    },
    view::{ViewTarget, window::prepare_windows},
};
use bevy_mosaic::MosaicCamera;

const SLOTS: usize = 3;
const QUERY_COUNT: u32 = 256;
const QUERY_BYTES: u64 = QUERY_COUNT as u64 * 8;
const IDLE: u8 = 0;
const PENDING: u8 = 1;
const READY: u8 = 2;
const FAILED: u8 = 3;

/// Explicitly selects the world camera; the x-ray/UI camera is measured separately.
#[derive(Component, Clone, ExtractComponent)]
pub(crate) struct ProfiledCamera;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[allow(clippy::struct_field_names)] // Milliseconds throughout the diagnostic boundary.
pub(crate) struct GpuBreakdown {
    pub(crate) prepass_ms: Option<f64>,
    pub(crate) opaque_ms: Option<f64>,
    pub(crate) terrain_ms: Option<f64>,
    pub(crate) opaque_other_ms: Option<f64>,
    pub(crate) post_ms: Option<f64>,
    pub(crate) ui_ms: Option<f64>,
    pub(crate) transparent_ms: Option<f64>,
    pub(crate) world_output_ms: Option<f64>,
    pub(crate) xray_ms: Option<f64>,
    pub(crate) xray_output_ms: Option<f64>,
    pub(crate) other_ms: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RenderExtent {
    pub(crate) target: UVec2,
    pub(crate) viewport: UVec2,
    pub(crate) samples: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum GpuSampleStatus {
    #[default]
    Waiting,
    Unsupported,
    NoPasses,
    CapacityExceeded,
    ReadbackFailed,
    InvalidPeriod,
    Complete,
    Partial(TimestampError),
}

impl GpuSampleStatus {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Waiting => "Waiting",
            Self::Unsupported => "Unsupported",
            Self::NoPasses => "No tracked passes",
            Self::CapacityExceeded => "Query limit exceeded",
            Self::ReadbackFailed => "Readback failed",
            Self::InvalidPeriod => "Invalid clock period",
            Self::Complete => "Complete",
            Self::Partial(TimestampError::Truncated) => "Partial: short data",
            Self::Partial(TimestampError::MissingStart) => "Partial: no start",
            Self::Partial(TimestampError::MissingEnd) => "Partial: no end",
            Self::Partial(TimestampError::Reversed) => "Partial: reversed",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TimestampError {
    Truncated,
    MissingStart,
    MissingEnd,
    Reversed,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GpuSampleHealth {
    pub(crate) status: GpuSampleStatus,
    pub(crate) valid_pairs: u32,
    pub(crate) total_pairs: u32,
}

type GpuSample = (Option<f64>, GpuBreakdown, GpuSampleHealth);

fn unavailable_sample(status: GpuSampleStatus, total_pairs: u32) -> GpuSample {
    (
        None,
        GpuBreakdown::default(),
        GpuSampleHealth {
            status,
            total_pairs,
            valid_pairs: 0,
        },
    )
}

#[derive(Clone, Copy, Debug, Default)]
#[allow(clippy::struct_field_names)] // Keep units explicit at the UI boundary.
pub(crate) struct Snapshot {
    pub(crate) acquire_cpu_ms: Option<f64>,
    pub(crate) render_cpu_ms: Option<f64>,
    pub(crate) render_gpu_ms: Option<f64>,
    pub(crate) breakdown: GpuBreakdown,
    pub(crate) health: GpuSampleHealth,
    pub(crate) extent: Option<RenderExtent>,
}

#[derive(Resource, Clone, Default)]
pub(crate) struct RenderTimings(Arc<Mutex<Snapshot>>, Arc<AtomicBool>);

impl RenderTimings {
    pub(crate) fn set_enabled(&self, enabled: bool) {
        self.1.store(enabled, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self) -> Snapshot {
        *self.0.lock().expect("render timings mutex")
    }
}

pub(crate) fn terrain_passes_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MECHANIC_PERF_TERRAIN_PASSES").as_deref() == Ok("1")
            && std::env::var_os("MECHANIC_PERF_CAPTURE_DIR").is_some_and(|v| !v.is_empty())
    })
}

pub(crate) struct RenderTimingsPlugin;

impl Plugin for RenderTimingsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<ProfiledCamera>::default());
        let timings = RenderTimings::default();
        let recording = Recording::default();
        let hook_recording = recording.clone();
        app.insert_resource(timings.clone());
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            if terrain_passes_enabled() {
                render_app.insert_resource(bevy::core_pipeline::core_3d::OpaquePassPartition(
                    Arc::new(|world, view, key, cache| {
                        world.get::<ProfiledCamera>(view)?;
                        cache.get_render_pipeline(key.pipeline)?;
                        let label = cache
                            .get_render_pipeline_descriptor(key.pipeline)
                            .label
                            .as_deref()?;
                        label
                            .starts_with("mechanic_terrain:")
                            .then_some("main_opaque_terrain_3d")
                    }),
                ));
            }
            render_app
                .insert_resource(timings)
                .insert_resource(recording)
                .insert_resource(RenderPassTimestampHook(Arc::new(move |view, label| {
                    hook_recording.allocate(view, label)
                })))
                .init_resource::<Probe>()
                .add_systems(
                    Render,
                    (
                        acquire_start
                            .in_set(RenderSystems::PrepareViews)
                            .before(prepare_windows),
                        acquire_end
                            .in_set(RenderSystems::PrepareViews)
                            .after(prepare_windows),
                        render_start
                            .in_set(RenderSystems::Render)
                            .before(render_system),
                        capture_extent
                            .in_set(RenderSystems::Render)
                            .before(render_start),
                        render_end
                            .in_set(RenderSystems::Render)
                            .after(render_system),
                    ),
                )
                .add_systems(
                    RenderGraph,
                    (
                        gpu_start
                            .after(RenderGraphSystems::Begin)
                            .before(RenderGraphSystems::Render),
                        gpu_end
                            .after(RenderGraphSystems::Render)
                            .before(bevy::render::diagnostic::resolve_encoder)
                            .before(RenderGraphSystems::Submit),
                    ),
                );
        }
    }
}

fn capture_extent(
    views: Query<(&ViewTarget, &ExtractedCamera, &Msaa), With<ProfiledCamera>>,
    timings: Res<RenderTimings>,
) {
    timings.0.lock().expect("render timings mutex").extent =
        views
            .single()
            .ok()
            .map(|(target, camera, msaa)| RenderExtent {
                target: UVec2::new(
                    target.main_texture().width(),
                    target.main_texture().height(),
                ),
                viewport: camera.physical_viewport_size.unwrap_or_default(),
                samples: msaa.samples(),
            });
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PassGroup {
    Prepass,
    Opaque,
    Terrain,
    OpaqueOther,
    Transparent,
    Xray,
    Other,
}

fn pass_group(label: &str, world: bool, xray: bool) -> PassGroup {
    // Shadows use light views, not necessarily the profiled camera's entity.
    if label.starts_with("shadow_") {
        PassGroup::Prepass
    } else if xray {
        PassGroup::Xray
    } else if world && label.contains("prepass") {
        PassGroup::Prepass
    } else if world && label == "main_opaque_terrain_3d" {
        PassGroup::Terrain
    } else if world && label == "main_opaque_other_3d" {
        PassGroup::OpaqueOther
    } else if world && label == "main_opaque_pass_3d" {
        PassGroup::Opaque
    } else if world && label == "main_transparent_pass_3d" {
        PassGroup::Transparent
    } else {
        PassGroup::Other
    }
}

#[derive(Resource, Clone, Default)]
struct Recording(Arc<Mutex<Option<ActiveRecording>>>);

struct ActiveRecording {
    queries: wgpu::QuerySet,
    groups: Vec<PassGroup>,
    views: Vec<(Entity, bool, bool)>,
    overflow: bool,
}

impl Recording {
    fn allocate(
        &self,
        view: Option<Entity>,
        label: Option<&str>,
    ) -> Option<RenderPassTimestampAllocation> {
        let mut guard = self.0.lock().expect("render recording mutex");
        let active = guard.as_mut()?;
        let beginning = u32::try_from(active.groups.len()).ok()? * 2;
        if beginning >= QUERY_COUNT {
            active.overflow = true;
            return None;
        }
        let (_, world, xray) = active
            .views
            .iter()
            .find(|(entity, _, _)| Some(*entity) == view)
            .copied()
            .unwrap_or((Entity::PLACEHOLDER, false, false));
        active
            .groups
            .push(pass_group(label.unwrap_or_default(), world, xray));
        Some(RenderPassTimestampAllocation {
            query_set: active.queries.clone(),
            beginning,
            end: beginning + 1,
        })
    }
}

#[derive(Resource, Default)]
struct Probe {
    acquire_started: Option<Instant>,
    render_started: Option<Instant>,
    initialized: bool,
    slots: Vec<Slot>,
    active: Option<usize>,
    next_frame: u64,
    published_frame: u64,
}

struct Slot {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    status: Arc<AtomicU8>,
    frame: u64,
    captured_at: Instant,
    groups: Vec<PassGroup>,
    overflow: bool,
}

impl Slot {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("render graph timing"),
                ty: wgpu::QueryType::Timestamp,
                count: QUERY_COUNT,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("render graph timestamp resolve"),
                size: QUERY_BYTES,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("render graph timestamp readback"),
                size: QUERY_BYTES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            status: Arc::new(AtomicU8::new(IDLE)),
            frame: 0,
            captured_at: Instant::now(),
            groups: Vec::new(),
            overflow: false,
        }
    }
}

fn acquire_start(mut probe: ResMut<Probe>) {
    probe.acquire_started = Some(Instant::now());
}
fn acquire_end(mut probe: ResMut<Probe>, timings: Res<RenderTimings>) {
    let ms = probe
        .acquire_started
        .take()
        .map(|start| start.elapsed().as_secs_f64() * 1000.0);
    timings
        .0
        .lock()
        .expect("render timings mutex")
        .acquire_cpu_ms = ms;
    crate::performance_capture::record("render_acquire", || serde_json::json!({"duration_ms":ms}));
}
fn render_start(mut probe: ResMut<Probe>) {
    probe.render_started = Some(Instant::now());
}
fn render_end(mut probe: ResMut<Probe>, timings: Res<RenderTimings>) {
    let ms = probe
        .render_started
        .take()
        .map(|start| start.elapsed().as_secs_f64() * 1000.0);
    timings
        .0
        .lock()
        .expect("render timings mutex")
        .render_cpu_ms = ms;
    crate::performance_capture::record("render_cpu", || serde_json::json!({"duration_ms":ms}));
}

fn capture_gpu_sample(slot: &Slot, value: GpuSample) {
    crate::performance_capture::record("render_gpu", || {
        serde_json::json!({
            "sample_id":slot.frame, "sample_age_ms":slot.captured_at.elapsed().as_secs_f64()*1000.0,
            "tracked_span_ms":value.0, "status":value.2.status.label(),
            "valid_pairs":value.2.valid_pairs, "total_pairs":value.2.total_pairs,
            "prepass_ms":value.1.prepass_ms, "opaque_ms":value.1.opaque_ms,
            "terrain_ms":value.1.terrain_ms, "opaque_other_ms":value.1.opaque_other_ms,
            "terrain_passes":slot.groups.iter().filter(|&&g| g == PassGroup::Terrain).count(),
            "transparent_ms":value.1.transparent_ms, "xray_ms":value.1.xray_ms,
            "other_ms":value.1.other_ms
        })
    });
}

type ProfiledViewFilter = Or<(With<ProfiledCamera>, With<MosaicCamera>)>;

fn gpu_start(
    mut probe: ResMut<Probe>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    timings: Res<RenderTimings>,
    recording: Res<Recording>,
    views: Query<(Entity, Has<ProfiledCamera>, Has<MosaicCamera>), ProfiledViewFilter>,
) {
    let enabled = timings.1.load(Ordering::Relaxed);
    if enabled && !probe.initialized {
        probe.initialized = true;
        if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            probe.slots = (0..SLOTS)
                .map(|_| Slot::new(device.wgpu_device()))
                .collect();
        } else {
            timings
                .0
                .lock()
                .expect("render timings mutex")
                .health
                .status = GpuSampleStatus::Unsupported;
        }
    }
    let mut newest = None;
    for slot in &probe.slots {
        match slot.status.load(Ordering::Acquire) {
            READY => {
                let value = {
                    let bytes = slot.readback.slice(..).get_mapped_range();
                    if slot.overflow {
                        unavailable_sample(GpuSampleStatus::CapacityExceeded, QUERY_COUNT / 2)
                    } else {
                        read_breakdown(&bytes, &slot.groups, queue.get_timestamp_period())
                    }
                };
                capture_gpu_sample(slot, value);
                slot.readback.unmap();
                slot.status.store(IDLE, Ordering::Release);
                if slot.frame > probe.published_frame
                    && newest.is_none_or(|(frame, _)| slot.frame > frame)
                {
                    newest = Some((slot.frame, value));
                }
            }
            FAILED => {
                capture_gpu_sample(
                    slot,
                    unavailable_sample(
                        GpuSampleStatus::ReadbackFailed,
                        u32::try_from(slot.groups.len()).unwrap(),
                    ),
                );
                slot.status.store(IDLE, Ordering::Release);
                if slot.frame > probe.published_frame
                    && newest.is_none_or(|(frame, _)| slot.frame > frame)
                {
                    newest = Some((
                        slot.frame,
                        unavailable_sample(
                            GpuSampleStatus::ReadbackFailed,
                            u32::try_from(slot.groups.len()).unwrap(),
                        ),
                    ));
                }
            }
            _ => {}
        }
    }
    if let Some((frame, value)) = newest {
        probe.published_frame = frame;
        let mut snapshot = timings.0.lock().expect("render timings mutex");
        snapshot.render_gpu_ms = value.0;
        snapshot.breakdown = value.1;
        snapshot.health = value.2;
    }
    if !enabled {
        return;
    }
    // Busy rings skip sampling; bounded storage and no waits on the render thread.
    probe.active = probe
        .slots
        .iter()
        .position(|slot| slot.status.load(Ordering::Acquire) == IDLE);
    if let Some(index) = probe.active {
        probe.next_frame += 1;
        probe.slots[index].frame = probe.next_frame;
        probe.slots[index].captured_at = Instant::now();
        probe.slots[index].groups.clear();
        probe.slots[index].status.store(PENDING, Ordering::Release);
        *recording.0.lock().expect("render recording mutex") = Some(ActiveRecording {
            queries: probe.slots[index].queries.clone(),
            groups: Vec::with_capacity(QUERY_COUNT as usize / 2),
            views: views
                .iter()
                .filter(|(_, world, xray)| *world || *xray)
                .collect(),
            overflow: false,
        });
    }
}

fn gpu_end(
    mut probe: ResMut<Probe>,
    device: Res<RenderDevice>,
    mut pending: ResMut<PendingCommandBuffers>,
    recording: Res<Recording>,
    timings: Res<RenderTimings>,
) {
    let Some(index) = probe.active.take() else {
        return;
    };
    let active = recording
        .0
        .lock()
        .expect("render recording mutex")
        .take()
        .unwrap();
    let frame = probe.slots[index].frame;
    if active.groups.is_empty() {
        probe.slots[index].status.store(IDLE, Ordering::Release);
        // Publish missing coverage too, without leaking samples from older views.
        probe.published_frame = frame;
        let mut snapshot = timings.0.lock().expect("render timings mutex");
        snapshot.render_gpu_ms = None;
        snapshot.breakdown = GpuBreakdown::default();
        snapshot.health = GpuSampleHealth {
            status: GpuSampleStatus::NoPasses,
            ..default()
        };
        return;
    }
    let slot = &mut probe.slots[index];
    slot.groups = active.groups;
    slot.overflow = active.overflow;
    let count = u32::try_from(slot.groups.len()).unwrap() * 2;
    // Resolve only queries written this frame, after all tracked pass encoders.
    let rendered = pending.take();
    pending.push(rendered);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("tracked render timestamp resolve"),
    });
    encoder.resolve_query_set(&slot.queries, 0..count, &slot.resolve, 0);
    encoder.copy_buffer_to_buffer(&slot.resolve, 0, &slot.readback, 0, u64::from(count) * 8);
    let status = Arc::clone(&slot.status);
    encoder.map_buffer_on_submit(&slot.readback, wgpu::MapMode::Read, .., move |result| {
        status.store(
            if result.is_ok() { READY } else { FAILED },
            Ordering::Release,
        );
    });
    pending.push([encoder.finish()]);
}

fn query_interval(bytes: &[u8], index: usize) -> Result<(u64, u64), TimestampError> {
    let pair = bytes
        .get(index * 16..index * 16 + 16)
        .ok_or(TimestampError::Truncated)?;
    let start = u64::from_le_bytes(pair[..8].try_into().unwrap());
    let end = u64::from_le_bytes(pair[8..].try_into().unwrap());
    if start == 0 {
        Err(TimestampError::MissingStart)
    } else if end == 0 {
        Err(TimestampError::MissingEnd)
    } else if end < start {
        Err(TimestampError::Reversed)
    } else {
        // Equal, nonzero timestamps are valid work below the clock's resolution.
        Ok((start, end))
    }
}

fn union_ticks(intervals: &mut [(u64, u64)]) -> u64 {
    intervals.sort_unstable_by_key(|interval| interval.0);
    let mut cursor = 0;
    let mut covered = 0;
    for &(start, end) in intervals.iter() {
        covered += end.saturating_sub(cursor.max(start));
        cursor = cursor.max(end);
    }
    covered
}

fn read_breakdown(bytes: &[u8], groups: &[PassGroup], period: f32) -> GpuSample {
    let total_pairs = u32::try_from(groups.len()).unwrap();
    if groups.is_empty() {
        return unavailable_sample(GpuSampleStatus::NoPasses, 0);
    }
    if !period.is_finite() || period <= 0.0 {
        return unavailable_sample(GpuSampleStatus::InvalidPeriod, total_pairs);
    }
    let intervals: Vec<_> = (0..groups.len())
        .map(|index| query_interval(bytes, index))
        .collect();
    let health = GpuSampleHealth {
        status: intervals
            .iter()
            .find_map(|interval| interval.as_ref().err())
            .map_or(GpuSampleStatus::Complete, |error| {
                GpuSampleStatus::Partial(*error)
            }),
        valid_pairs: u32::try_from(intervals.iter().filter(|interval| interval.is_ok()).count())
            .unwrap(),
        total_pairs,
    };
    #[allow(clippy::cast_precision_loss)] // Convert relative ticks, not absolute timestamps.
    let milliseconds = |ticks: u64| ticks as f64 * f64::from(period) / 1_000_000.0;
    let group_ms = |group| {
        // An incomplete shadow cascade must not discard valid opaque timings, but
        // neither may its remaining cascades masquerade as the whole shadow group.
        let mut selected = intervals
            .iter()
            .zip(groups)
            .filter_map(|(&interval, &kind)| (kind == group).then_some(interval))
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        (!selected.is_empty()).then(|| milliseconds(union_ticks(&mut selected)))
    };
    let span = (health.status == GpuSampleStatus::Complete).then(|| {
        let start = intervals
            .iter()
            .flatten()
            .map(|interval| interval.0)
            .min()
            .unwrap();
        let end = intervals
            .iter()
            .flatten()
            .map(|interval| interval.1)
            .max()
            .unwrap();
        milliseconds(end - start)
    });
    (
        span,
        GpuBreakdown {
            prepass_ms: group_ms(PassGroup::Prepass),
            opaque_ms: group_ms(PassGroup::Opaque),
            terrain_ms: group_ms(PassGroup::Terrain),
            opaque_other_ms: group_ms(PassGroup::OpaqueOther),
            transparent_ms: group_ms(PassGroup::Transparent),
            xray_ms: group_ms(PassGroup::Xray),
            other_ms: group_ms(PassGroup::Other),
            // Raw post-processing, output and external Mosaic passes bypass the hook.
            ..Default::default()
        },
        health,
    )
}

#[cfg(test)]
mod tests {
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
            timestamp_writes: raster.explicit.as_ref().map(|queries| {
                wgpu::RenderPassTimestampWrites {
                    query_set: queries,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }
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
        let target =
            app.world_mut()
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
}
