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
#[expect(
    clippy::struct_field_names,
    reason = "milliseconds throughout the diagnostic boundary"
)]
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
        crate::env::flag(crate::env::PERF_TERRAIN_PASSES)
            && crate::env::is_set(crate::env::PERF_CAPTURE_DIR)
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
    #[expect(
        clippy::cast_precision_loss,
        reason = "convert relative ticks, not absolute timestamps"
    )]
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
mod tests;
