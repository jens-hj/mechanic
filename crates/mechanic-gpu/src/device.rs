mod construct;
mod encode;
mod hold;
mod host_writes;
mod pipelines;
mod readback;
mod submit;
mod terrain;
mod wgpu_util;
pub use hold::GpuHoldError;
pub use terrain::{
    GpuTerrainError, PreparedTerrainUpdate, TerrainPreparationCache, TerrainPreparationRequest,
    TerrainResidency, TerrainUploadStats,
};

use std::collections::BTreeMap;
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicU64},
    mpsc,
};
use std::time::Instant;

use bevy_math::Vec3;
use bytemuck::{Pod, Zeroable};
use thiserror::Error;

use wgpu_util::{
    bind_group, compute_pipeline, create_sized_buffer, direct_compute_pass, entry,
    indirect_compute_pass, indirect_dispatch_in_pass, shader_module, timestamp_writes,
};

use crate::{
    GpuBearing, GpuMass, GpuMechanismBody, GpuMechanismCoordinate, GpuMechanismDrive, GpuTransform,
    GpuVelocity,
};

const FUSED_VELOCITY_BEARING_LIMIT: u32 = 64;
const FUSED_GROUND_CONTACT_BEARING_LIMIT: u32 = 64;
const FUSED_STREAMED_CONTACT_BEARING_LIMIT: u32 = 64;
const ASYNC_READBACK_RING_SIZE: usize = 12;

/// Number of external impulses staged and applied by one serial GPU pass.
pub const EXTERNAL_IMPULSE_BATCH_CAPACITY: usize = 64;

/// Per-scene pipeline switches that do not adapt during simulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuPhysicsConfig {
    /// Whether broadphase, SAT, and projected contact impulses are dispatched.
    pub collisions_enabled: bool,
    /// Whether the explicit flat ground plane participates in collision.
    /// Garage and benchmark scenes opt into this; streamed worlds disable it.
    pub ground_plane_enabled: bool,
    /// Whether colliders in the same articulated mechanism may contact.
    pub mechanism_self_collisions: bool,
    /// Base number of projected impulse iterations. Small mechanisms with angle
    /// drives use at least 32 pre-integration velocity iterations; contact
    /// sweeps retain this configured count.
    pub solver_iterations: u32,
}

/// Immutable contact-solver schedule selected when a scene is uploaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuSolverRoute {
    /// Parallel contact projection with separate mechanism correction passes.
    General,
    /// One ordered dispatch for a small floating articulated mechanism.
    FusedSmallMechanism,
}

impl GpuSolverRoute {
    /// Stable diagnostic label used by benchmark JSONL.
    pub const fn name(self) -> &'static str {
        match self {
            Self::General => "general",
            Self::FusedSmallMechanism => "fused_small_mechanism",
        }
    }
}

impl Default for GpuPhysicsConfig {
    fn default() -> Self {
        Self {
            collisions_enabled: true,
            ground_plane_enabled: true,
            mechanism_self_collisions: true,
            solver_iterations: 8,
        }
    }
}

/// Device-local shader and compute-pipeline cache shared by replaceable scenes.
///
/// Scene buffers and bind groups remain owned by [`GpuPhysics`]; rebuilding a
/// topology with this cache only uploads those scene-specific resources.
#[derive(Debug, Default)]
pub struct GpuPhysicsPipelines {
    shaders: Mutex<BTreeMap<&'static str, wgpu::ShaderModule>>,
    pipelines: Mutex<BTreeMap<(&'static str, &'static str), wgpu::ComputePipeline>>,
}

impl GpuPhysicsPipelines {
    /// Creates an empty cache that compiles each embedded kernel lazily once.
    pub fn new() -> Self {
        Self::default()
    }
}

const fn uses_fused_velocity_schedule(bearing_count: u32, body_count: u32) -> bool {
    bearing_count <= FUSED_VELOCITY_BEARING_LIMIT && body_count <= 256
}

const fn uses_fused_contact_schedule(bearing_count: u32, ground_plane_enabled: bool) -> bool {
    // Keep contact and bearing projection tightly coupled for small mechanisms.
    // This is necessary for large body-to-wheel mass ratios: a parallel pass
    // can otherwise let the chassis outrun the wheel contacts before their
    // impulses propagate through the bearings. Both choices are immutable
    // after the scene is loaded.
    let limit = if ground_plane_enabled {
        FUSED_GROUND_CONTACT_BEARING_LIMIT
    } else {
        FUSED_STREAMED_CONTACT_BEARING_LIMIT
    };
    bearing_count <= limit
}

/// GPU buffers containing one complete renderable physics snapshot.
#[derive(Debug)]
pub struct SnapshotBuffers {
    positions: wgpu::Buffer,
    rotations: wgpu::Buffer,
}

impl SnapshotBuffers {
    /// Compound positions as tightly packed `vec4<f32>` rows.
    pub const fn positions(&self) -> &wgpu::Buffer {
        &self.positions
    }

    /// Compound orientations as tightly packed quaternion `vec4<f32>` rows.
    pub const fn rotations(&self) -> &wgpu::Buffer {
        &self.rotations
    }
}

/// Submitted tick identity. Completion is asynchronous on the shared queue.
#[derive(Debug)]
pub struct GpuTickSubmission {
    /// Contiguous submission ordinal, independent of scheduler tick gaps.
    pub submission_sequence: u64,
    /// Tick encoded into the submission.
    pub tick_index: u64,
    /// Snapshot ring destination written by that tick.
    pub snapshot_slot: u8,
    /// Shared queue submission token.
    pub submission_index: wgpu::SubmissionIndex,
    /// CPU wall-clock stage costs; these do not wait for GPU completion.
    pub cpu_timings: GpuSubmissionTimings,
}

/// Non-overlapping CPU wall-clock costs for encoding and submitting one tick.
/// Includes driver calls and any contention inside them, not just CPU execution.
/// External-impulse dispatches and later readback polling are outside these stages.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuSubmissionTimings {
    /// Config upload, command encoding, and readback staging allocation/copies.
    pub encoding_ms: f64,
    /// Finishing the command encoder into a command buffer.
    pub finalization_ms: f64,
    /// Shared queue submission, including pending uploads and resource maintenance.
    pub submission_ms: f64,
    /// Registering asynchronous mapping callbacks, not waiting for their completion.
    pub readback_setup_ms: f64,
}

/// Validation values copied back after a tick without reading body state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuTickReadback {
    /// Evidence written by the executing shaders, independent of scenario labels.
    pub execution: GpuExecutionEvidence,
    /// Timestamp-query duration, or `None` if the shared device lacks support.
    pub gpu_tick_ms: Option<f64>,
    /// Per-stage timestamp durations, or `None` without timestamp-query support.
    pub kernel_timings: Option<GpuKernelTimings>,
    /// Kernel failure flags. A non-zero value blocks publication.
    pub error_flags: u32,
    /// Broadphase candidates requested during the tick.
    pub pair_count: u32,
    /// SAT contacts requested during the tick.
    pub contact_count: u32,
    /// Contacts dispatched through the projected impulse iterations.
    pub active_contact_count: u32,
    /// Contact/bearing sweeps budgeted for the selected solver route.
    pub planned_solver_sweeps: u32,
    /// Contact/bearing sweeps executed by the selected solver route.
    pub executed_solver_sweeps: u32,
    /// Largest derived bearing anchor residual in metres.
    pub anchor_residual_meters: f32,
    /// Largest derived bearing axis residual in degrees.
    pub axis_residual_degrees: f32,
}

/// Device-written execution evidence. Stage markers prove entry, not every kernel
/// or physical correctness. Body counters exclude early returns after a failure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuExecutionEvidence {
    /// Integration=1, articulated validation=2, broadphase=4, narrowphase=8,
    /// general/serial contact accumulator=16, bearing validation=32, publication=64,
    /// terrain=128. The fused contact kernel has no spare portable storage binding
    /// for diagnostics and is deliberately unmarked.
    pub stage_mask: u32,
    /// Body rows admitted past integration's initial bounds/failure checks.
    pub integrated_bodies: u32,
    /// Body rows copied to the validated snapshot.
    pub published_bodies: u32,
    /// Bearing rows visited by validation.
    pub validated_bearings: u32,
}

/// One asynchronously completed tick and its prototype-render snapshot.
///
/// The body rows are staged with the fixed-size diagnostics so application
/// frames can consume them without ever waiting for the GPU. Production
/// renderers should continue to bind [`SnapshotBuffers`] directly.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuCompletedTickReadback {
    /// Contiguous submission ordinal, independent of scheduler tick gaps.
    pub submission_sequence: u64,
    /// Monotonic tick encoded into the submission.
    pub tick_index: u64,
    /// Snapshot-ring slot written by the completed tick.
    pub snapshot_slot: u8,
    /// Wall-clock latency from queue submission until mapped readback consumption.
    pub submission_to_readback_ms: f64,
    /// Time when the last mapping callback ran, if diagnostic timing is enabled.
    /// This observes callback servicing, not the instant GPU execution completed.
    pub callbacks_completed_at: Option<Instant>,
    /// Queue-return to final mapping callback latency, when enabled.
    pub submission_to_callbacks_ms: Option<f64>,
    /// Whether the final callback ran during this call to `poll_tick_readback`.
    pub callbacks_during_poll: Option<bool>,
    /// Fixed-size validation and timestamp telemetry.
    pub diagnostics: GpuTickReadback,
    /// CPU prototype-render rows captured from the same tick.
    pub transforms: Vec<GpuTransform>,
    /// Authoritative body velocities captured in the same submission as the poses.
    pub velocities: Vec<GpuVelocity>,
    /// Permitted joint positions and velocities from the same tick.
    pub coordinates: Vec<GpuMechanismCoordinate>,
}

/// GPU timestamp durations for the fixed production pipeline stages.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuKernelTimings {
    /// Body gravity/damping and root-pose integration.
    pub integration_ms: f64,
    /// Exact terrain BVH traversal and contact generation (part of narrowphase).
    pub terrain_traversal_ms: f64,
    /// Conservative rotational CCD.
    pub rotational_sweep_ms: f64,
    /// Split positional recovery, including its mechanism projection.
    pub terrain_recovery_ms: f64,
    /// Mechanism projection inside the three recovery rounds.
    pub recovery_projection_ms: f64,
    /// Reduced-coordinate projection, closure factorization, and forward kinematics.
    pub mechanism_ms: f64,
    /// Spatial broadphase and candidate generation.
    pub broadphase_ms: f64,
    /// OBB SAT and manifold-cache update.
    pub narrowphase_ms: f64,
    /// Warm-started projected impulses, persistence, and articulated feedback.
    pub contact_solver_ms: f64,
    /// Bearing closure validation.
    pub bearings_ms: f64,
    /// GPU snapshot-ring publication.
    pub snapshot_ms: f64,
}

/// Fixed-size diagnostic readback failed.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum GpuReadbackError {
    /// The requested snapshot ring slot does not exist.
    #[error("snapshot slot {0} is outside the published ring")]
    InvalidSnapshotSlot(u8),
    /// Device polling failed before the mapping callback ran.
    #[error("device polling failed: {0}")]
    DevicePoll(String),
    /// wgpu rejected a diagnostic buffer map.
    #[error("diagnostic buffer mapping failed: {0}")]
    BufferMap(String),
    /// Mapping callback channel closed unexpectedly.
    #[error("diagnostic buffer mapping callback was lost")]
    CallbackLost,
}

/// GPU upload or dispatch could not be created safely.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum GpuPhysicsError {
    /// Scene exceeds the fixed compound capacity.
    #[error("scene requires {required} bodies but capacity is {capacity}")]
    BodyCapacity {
        /// Required rows.
        required: usize,
        /// Allocated rows.
        capacity: usize,
    },
    /// Scene exceeds the fixed bearing capacity.
    #[error("scene requires {required} bearings but capacity is {capacity}")]
    BearingCapacity {
        /// Required rows.
        required: usize,
        /// Allocated rows.
        capacity: usize,
    },
    /// Scene exceeds the fixed collider capacity.
    #[error("scene requires {required} colliders but capacity is {capacity}")]
    ColliderCapacity {
        /// Required rows.
        required: usize,
        /// Allocated rows.
        capacity: usize,
    },
    /// Scene exceeds the fixed convex-shape buffer.
    #[error("scene requires {required} convex-shape slots but capacity is {capacity}")]
    ConvexShapeCapacity {
        /// Required slots.
        required: usize,
        /// Allocated slots.
        capacity: usize,
    },
    /// Replacement drive rows do not match the compiled coordinate count.
    #[error("drive state has {provided} rows but scene requires {required}")]
    DriveStateCount {
        /// Rows supplied by the caller.
        provided: usize,
        /// Rows required by the compiled forest.
        required: usize,
    },
    /// Initial mechanism-coordinate state does not match the compiled forest.
    #[error("coordinate state has {provided} rows but scene requires {required}")]
    CoordinateStateCount {
        /// Supplied rows.
        provided: usize,
        /// Required rows.
        required: usize,
    },
}

/// A requested external impulse cannot be submitted safely.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum GpuImpulseError {
    /// The requested body row is outside the uploaded creation.
    #[error("body index {body_index} is outside the uploaded body count {body_count}")]
    BodyIndexOutOfRange {
        /// Requested body row.
        body_index: u32,
        /// Uploaded row count.
        body_count: u32,
    },
    /// The world point or impulse contains NaN or infinity.
    #[error("external impulse point and vector must be finite")]
    NonFinite,
    /// Direct batch submission accepts one fixed staging batch.
    #[error("external impulse batch contains {provided} rows; capacity is {capacity}")]
    BatchCapacity {
        /// Supplied rows.
        provided: usize,
        /// Fixed staging capacity.
        capacity: usize,
    },
}

/// A tick cannot be dispatched.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum GpuDispatchError {
    /// An external impulse row is invalid.
    #[error(transparent)]
    Impulse(#[from] GpuImpulseError),
    /// The GPU runtime has no kernel for meshes between gears yet, and
    /// silently dropping them would run a different machine.
    #[error("the GPU runtime does not simulate gear meshes yet; this scene has {count}")]
    UnsupportedGearLinks {
        /// Meshes in the uploaded creation.
        count: usize,
    },
}

/// A replacement scene state cannot be uploaded safely.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum GpuBodyStateError {
    /// State rows must exactly match the uploaded body count.
    #[error("body state count {provided} does not match uploaded body count {expected}")]
    BodyCount {
        /// Number of rows supplied by the caller.
        provided: usize,
        /// Number of rows allocated by the scene.
        expected: u32,
    },
    /// Every transform and velocity lane must be finite.
    #[error("body state contains NaN or infinity")]
    NonFinite,
}

/// A replacement set of collider-local terrain support planes is invalid.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum GpuGroundPlaneError {
    /// Plane rows must exactly match the uploaded collider count.
    #[error("ground plane count {provided} does not match collider count {expected}")]
    PlaneCount {
        /// Number of rows supplied by the caller.
        provided: usize,
        /// Number of collider rows allocated by the scene.
        expected: u32,
    },
    /// A normal or offset contains NaN or infinity.
    #[error("ground plane contains NaN or infinity")]
    NonFinite,
}

/// One local terrain plane assigned to a compiled collider.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuGroundPlane {
    /// Normal pointing out of terrain toward the construction.
    pub normal: Vec3,
    /// Signed plane offset, satisfying `dot(point, normal) == offset`.
    pub offset: f32,
}

impl GpuGroundPlane {
    /// No terrain surface is available beneath this collider.
    pub const DISABLED: Self = Self {
        normal: Vec3::ZERO,
        offset: 0.0,
    };

    /// Creates a plane passing through `point` with the supplied outward normal.
    pub fn through_point(normal: Vec3, point: Vec3) -> Self {
        let normal = normal.normalize_or_zero();
        Self {
            normal,
            offset: normal.dot(point),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
/// One validated construction-frame impulse consumed by the GPU serial pass.
pub struct GpuExternalImpulse {
    /// xyz construction-frame contact point; w is unused.
    pub world_point: [f32; 4],
    /// xyz impulse applied to the compound; w is unused.
    pub impulse: [f32; 4],
    /// Body index in x; remaining lanes are reserved.
    pub metadata: [u32; 4],
}

impl GpuExternalImpulse {
    /// Creates one world-space impulse row.
    pub const fn new(body_index: u32, world_point: Vec3, impulse: Vec3) -> Self {
        Self {
            world_point: [world_point.x, world_point.y, world_point.z, 0.0],
            impulse: [impulse.x, impulse.y, impulse.z, 0.0],
            metadata: [body_index, 0, 0, 0],
        }
    }

    /// Target compound row.
    pub const fn body_index(self) -> u32 {
        self.metadata[0]
    }

    fn is_finite(self) -> bool {
        self.world_point[..3].iter().all(|lane| lane.is_finite())
            && self.impulse[..3].iter().all(|lane| lane.is_finite())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuExternalImpulseBatch {
    metadata: [u32; 4],
    rows: [GpuExternalImpulse; EXTERNAL_IMPULSE_BATCH_CAPACITY],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuDriveConstraint {
    bearing: GpuBearing,
    drive: GpuMechanismDrive,
    /// Axis inertia, accumulated impulse, current angle, reserved.
    state: [f32; 4],
    /// Child body, parent body, tree direction, coordinate.
    metadata: [u32; 4],
}

/// Custom compute resources backed by Bevy's shared wgpu device and queue.
#[derive(Debug)]
pub struct GpuPhysics {
    body_count: u32,
    holds: Mutex<hold::HoldResources>,
    collider_count: u32,
    bearing_count: u32,
    suppression_count: u32,
    /// Meshes in the uploaded creation, which no kernel simulates yet.
    gear_link_count: usize,
    pair_capacity: u32,
    pipeline_config: GpuPhysicsConfig,
    config: wgpu::Buffer,
    positions: wgpu::Buffer,
    rotations: wgpu::Buffer,
    linear_velocities: wgpu::Buffer,
    angular_velocities: wgpu::Buffer,
    inverse_masses: wgpu::Buffer,
    diagnostics: wgpu::Buffer,
    diagnostics_readback: wgpu::Buffer,
    snapshot_positions_readback: wgpu::Buffer,
    snapshot_rotations_readback: wgpu::Buffer,
    masses: wgpu::Buffer,
    _spatial_inertias: wgpu::Buffer,
    colliders: wgpu::Buffer,
    convex_shapes: wgpu::Buffer,
    bearings: wgpu::Buffer,
    external_impulse: wgpu::Buffer,
    external_impulse_pipeline: wgpu::ComputePipeline,
    external_impulse_bind_group: wgpu::BindGroup,
    snapshots: Vec<SnapshotBuffers>,
    bind_groups: Vec<wgpu::BindGroup>,
    integration_pipeline: wgpu::ComputePipeline,
    mechanism: MechanismResources,
    collision: CollisionResources,
    terrain_recovery: Option<terrain::PositionRecovery>,
    terrain_scene: terrain::TerrainGpuScene,
    terrain_free_bodies: wgpu::Buffer,
    terrain_has_free_bodies: bool,
    bearing_pipeline: wgpu::ComputePipeline,
    bearing_bind_group: wgpu::BindGroup,
    snapshot_pipeline: wgpu::ComputePipeline,
    snapshot_bind_groups: Vec<wgpu::BindGroup>,
    timestamps: Option<TimestampResources>,
    submission_sequence: AtomicU64,
    async_readback_enabled: AtomicBool,
    readback_timing_enabled: AtomicBool,
    async_readbacks: Mutex<Vec<AsyncReadbackSlot>>,
}

#[derive(Debug)]
struct AsyncReadbackSlot {
    diagnostics: wgpu::Buffer,
    timestamps: Option<wgpu::Buffer>,
    positions: wgpu::Buffer,
    rotations: wgpu::Buffer,
    linear_velocities: wgpu::Buffer,
    angular_velocities: wgpu::Buffer,
    coordinates: wgpu::Buffer,
    pending: Option<PendingAsyncReadback>,
}

#[derive(Debug)]
struct PendingAsyncReadback {
    submission_sequence: u64,
    tick_index: u64,
    snapshot_slot: u8,
    receiver: mpsc::Receiver<(Result<(), String>, Option<Instant>)>,
    callbacks_completed_at: Option<Instant>,
    remaining_callbacks: u8,
    submitted_at: Instant,
}

#[derive(Debug)]
struct CollisionResources {
    lbvh: LbvhResources,
    body_components: wgpu::Buffer,
    _pairs: wgpu::Buffer,
    contacts: wgpu::Buffer,
    manifold_keys: wgpu::Buffer,
    persistent_manifolds: wgpu::Buffer,
    ground_surfaces: wgpu::Buffer,
    active_contacts: wgpu::Buffer,
    indirect_args: wgpu::Buffer,
    velocity_deltas: wgpu::Buffer,
    world_masses: wgpu::Buffer,
    update_world_masses_pipeline: wgpu::ComputePipeline,
    update_world_masses_bind_group: wgpu::BindGroup,
    narrowphase_pipeline: wgpu::ComputePipeline,
    narrowphase_bind_group: wgpu::BindGroup,
    ground_contacts_pipeline: wgpu::ComputePipeline,
    ground_contacts_bind_group: wgpu::BindGroup,
    terrain_pipeline: wgpu::ComputePipeline,
    terrain_bind_group: Option<wgpu::BindGroup>,
    terrain_geometry_bind_group: Option<wgpu::BindGroup>,
    terrain_triangle_count: usize,
    finalize_contacts_pipeline: wgpu::ComputePipeline,
    finalize_contacts_bind_group: wgpu::BindGroup,
    select_active_pipeline: wgpu::ComputePipeline,
    select_active_bind_group: wgpu::BindGroup,
    finalize_active_pipeline: wgpu::ComputePipeline,
    finalize_active_bind_group: wgpu::BindGroup,
    count_body_contacts_pipeline: wgpu::ComputePipeline,
    count_body_contacts_bind_group: wgpu::BindGroup,
    warm_start_pipeline: wgpu::ComputePipeline,
    warm_start_bind_group: wgpu::BindGroup,
    solve_accumulate_pipeline: wgpu::ComputePipeline,
    solve_accumulate_bind_group: wgpu::BindGroup,
    solve_small_mechanism_pipeline: wgpu::ComputePipeline,
    solve_small_mechanism_bind_group: wgpu::BindGroup,
    solve_apply_pipeline: wgpu::ComputePipeline,
    solve_apply_bind_group: wgpu::BindGroup,
    persist_contacts_pipeline: wgpu::ComputePipeline,
    persist_contacts_bind_group: wgpu::BindGroup,
}

#[derive(Debug)]
struct LbvhResources {
    sort_count: u32,
    _collider_aabbs: wgpu::Buffer,
    _morton_entries: wgpu::Buffer,
    _node_aabbs: wgpu::Buffer,
    _node_children: wgpu::Buffer,
    node_parents: wgpu::Buffer,
    node_visits: wgpu::Buffer,
    sort_params: wgpu::Buffer,
    sort_params_upload: wgpu::Buffer,
    sort_steps: Vec<(u64, bool)>,
    compute_morton_pipeline: wgpu::ComputePipeline,
    compute_morton_bind_group: wgpu::BindGroup,
    sort_local_initial_pipeline: wgpu::ComputePipeline,
    sort_local_initial_bind_group: wgpu::BindGroup,
    sort_global_pipeline: wgpu::ComputePipeline,
    sort_global_bind_group: wgpu::BindGroup,
    sort_local_merge_pipeline: wgpu::ComputePipeline,
    sort_local_merge_bind_group: wgpu::BindGroup,
    build_topology_pipeline: wgpu::ComputePipeline,
    build_topology_bind_group: wgpu::BindGroup,
    prepare_leaves_pipeline: wgpu::ComputePipeline,
    prepare_leaves_bind_group: wgpu::BindGroup,
    build_bounds_pipeline: wgpu::ComputePipeline,
    build_bounds_bind_group: wgpu::BindGroup,
    traverse_pipeline: wgpu::ComputePipeline,
    traverse_bind_group: wgpu::BindGroup,
    finalize_pairs_pipeline: wgpu::ComputePipeline,
    finalize_pairs_bind_group: wgpu::BindGroup,
}

#[derive(Debug)]
struct MechanismResources {
    root_flags: wgpu::Buffer,
    bodies: wgpu::Buffer,
    body_rows: Vec<GpuMechanismBody>,
    coordinates: wgpu::Buffer,
    drives: wgpu::Buffer,
    drive_constraints: wgpu::Buffer,
    drive_constraint_rows: Vec<u32>,
    _preorder: wgpu::Buffer,
    _contraction_schedule: wgpu::Buffer,
    velocity_deltas: wgpu::Buffer,
    _articulated_inertia: wgpu::Buffer,
    _bias_force: wgpu::Buffer,
    _generalized_force: wgpu::Buffer,
    _constraint_impulse: wgpu::Buffer,
    _reduction_scratch: wgpu::Buffer,
    links_a: wgpu::Buffer,
    links_b: wgpu::Buffer,
    closure_accumulators: wgpu::Buffer,
    closure_state: wgpu::Buffer,
    closure_indirect_args: wgpu::Buffer,
    prepare_pipeline: wgpu::ComputePipeline,
    prepare_bind_group: wgpu::BindGroup,
    jump_a_to_b_pipeline: wgpu::ComputePipeline,
    jump_a_to_b_bind_group: wgpu::BindGroup,
    jump_b_to_a_pipeline: wgpu::ComputePipeline,
    jump_b_to_a_bind_group: wgpu::BindGroup,
    publish_a_pipeline: wgpu::ComputePipeline,
    publish_a_bind_group: wgpu::BindGroup,
    publish_b_pipeline: wgpu::ComputePipeline,
    publish_b_bind_group: wgpu::BindGroup,
    evaluate_closures_pipeline: wgpu::ComputePipeline,
    evaluate_closures_bind_group: wgpu::BindGroup,
    finalize_closures_pipeline: wgpu::ComputePipeline,
    finalize_closures_bind_group: wgpu::BindGroup,
    apply_closure_step_pipeline: wgpu::ComputePipeline,
    apply_closure_step_bind_group: wgpu::BindGroup,
    project_velocity_pipeline: wgpu::ComputePipeline,
    project_velocity_bind_group: wgpu::BindGroup,
    project_small_velocity_pipeline: wgpu::ComputePipeline,
    project_small_velocity_bind_group: wgpu::BindGroup,
    project_velocity_serial_pipeline: wgpu::ComputePipeline,
    project_velocity_serial_bind_group: wgpu::BindGroup,
    apply_velocity_pipeline: wgpu::ComputePipeline,
    apply_velocity_bind_group: wgpu::BindGroup,
    prepare_drives_pipeline: wgpu::ComputePipeline,
    prepare_drives_bind_group: wgpu::BindGroup,
    advance_coordinates_pipeline: wgpu::ComputePipeline,
    advance_coordinates_bind_group: wgpu::BindGroup,
    capture_coordinates_pipeline: wgpu::ComputePipeline,
    capture_coordinates_bind_group: wgpu::BindGroup,
    reconstruct_velocities_pipeline: wgpu::ComputePipeline,
    reconstruct_velocities_bind_group: wgpu::BindGroup,
    validate_state_pipeline: wgpu::ComputePipeline,
    validate_state_bind_group: wgpu::BindGroup,
    pointer_jump_rounds: u32,
    coordinate_count: u32,
    closure_count: u32,
    final_is_a: bool,
    active: bool,
    has_dynamic_root: bool,
}

#[derive(Debug)]
struct TimestampResources {
    boundary: wgpu::ComputePipeline,
    boundary_bindings: wgpu::BindGroup,
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_nanoseconds: f64,
}

impl GpuPhysics {
    /// Contact-solver route fixed for this uploaded scene.
    pub const fn solver_route(&self) -> GpuSolverRoute {
        if self.mechanism.active
            && self.mechanism.has_dynamic_root
            && uses_fused_contact_schedule(
                self.bearing_count,
                self.pipeline_config.ground_plane_enabled,
            )
        {
            GpuSolverRoute::FusedSmallMechanism
        } else {
            GpuSolverRoute::General
        }
    }

    /// Number of asynchronous tick slots awaiting readback completion.
    pub fn in_flight_tick_count(&self) -> usize {
        self.async_readbacks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .filter(|slot| slot.pending.is_some())
            .count()
    }

    /// Whether the shared device supports real GPU timestamp queries.
    pub const fn has_gpu_timestamps(&self) -> bool {
        self.timestamps.is_some()
    }

    /// Snapshot buffers for direct GPU-driven render consumption.
    pub fn snapshot(&self, slot: u8) -> Option<&SnapshotBuffers> {
        self.snapshots.get(usize::from(slot))
    }

    /// Number of uploaded compound rows.
    pub const fn body_count(&self) -> u32 {
        self.body_count
    }

    /// Current authoritative position buffer.
    pub const fn positions(&self) -> &wgpu::Buffer {
        &self.positions
    }

    /// Current authoritative orientation buffer.
    pub const fn rotations(&self) -> &wgpu::Buffer {
        &self.rotations
    }

    /// Current authoritative linear-velocity buffer.
    pub const fn linear_velocities(&self) -> &wgpu::Buffer {
        &self.linear_velocities
    }

    /// Current authoritative angular-velocity buffer.
    pub const fn angular_velocities(&self) -> &wgpu::Buffer {
        &self.angular_velocities
    }

    /// Current inverse-mass struct-of-arrays buffer.
    pub const fn inverse_masses(&self) -> &wgpu::Buffer {
        &self.inverse_masses
    }
}

#[cfg(test)]
const FULL_CYLINDER_GROUND_FIRST: u32 = 1;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct CylinderGroundData {
    center_radius: f32,
    outer_radius: f32,
    role: u32,
}

fn full_cylinder_ground_data(
    colliders: &[mechanic_core::LocalCollider],
) -> Vec<CylinderGroundData> {
    let mut result = vec![CylinderGroundData::default(); colliders.len()];
    let mut start = 0;
    while start < colliders.len() {
        let source_part = colliders[start].source_part;
        let mut end = start + 1;
        while end < colliders.len() && colliders[end].source_part == source_part {
            end += 1;
        }
        let group = &colliders[start..end];
        // Gate on the compiled shape, never on the run length: a shaped cuboid
        // that happened to fuse into sixteen pieces would otherwise be mistaken
        // for a cylinder and given analytic ground contacts.
        let cuboid_extents = |collider: &mechanic_core::LocalCollider| match &collider.shape {
            mechanic_core::ColliderShape::Cuboid {
                local_rotation,
                half_extents,
            } => Some((*local_rotation, *half_extents)),
            mechanic_core::ColliderShape::Convex(_) => None,
        };
        if group.len() == mechanic_core::CYLINDER_COLLIDER_COUNT
            && group
                .iter()
                .all(|collider| cuboid_extents(collider).is_some())
        {
            let radial_sum = group
                .iter()
                .filter_map(cuboid_extents)
                .map(|(rotation, _)| rotation * Vec3::X)
                .sum::<Vec3>();
            if radial_sum.length_squared() < 1.0e-8 {
                let cylinder_center = group
                    .iter()
                    .map(|collider| collider.local_center)
                    .sum::<Vec3>()
                    * (1.0 / 16.0);
                let center_radius = (group[0].local_center - cylinder_center).length();
                let outer_radius = center_radius
                    + cuboid_extents(&group[0])
                        .expect("every row in a cylinder run is a box")
                        .1
                        .x;
                for (segment, row) in result[start..end].iter_mut().enumerate() {
                    *row = CylinderGroundData {
                        center_radius,
                        outer_radius,
                        role: u32::try_from(segment + 1).expect("cylinder segment fits u32"),
                    };
                }
            }
        }
        start = end;
    }
    result
}

#[cfg(test)]
mod tests;
