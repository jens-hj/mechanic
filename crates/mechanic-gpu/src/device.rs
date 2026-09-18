mod hold;
mod terrain;
pub use hold::GpuHoldError;
pub use terrain::{
    GpuTerrainError, PreparedTerrainUpdate, TerrainPreparationCache, TerrainPreparationRequest,
    TerrainResidency, TerrainUploadStats,
};

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc,
};
use std::time::Instant;

use bevy_math::Vec3;
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use mechanic_core::{
    ColliderShape, CompiledCreation, ConstructionMaterial, STANDARD_GRAVITY_M_S2_F32,
    TICK_SECONDS_F32,
};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::{
    BROADPHASE_HASH_CAPACITY, COLLIDER_SHAPE_CONVEX, COLLIDER_SHAPE_CUBOID, GpuBearing,
    GpuCollider, GpuContact, GpuContractionNode, GpuDiagnostics, GpuGroundSurface, GpuLinkState,
    GpuMass, GpuMechanismBody, GpuMechanismCoordinate, GpuMechanismDrive, GpuPair,
    GpuPersistentManifold, GpuSpatialInertia, GpuTickConfig, GpuTransform, GpuVelocity,
    MAX_BEARINGS, MAX_BODIES, MAX_COLLIDERS, MAX_CONTACT_PAIRS, MAX_CONVEX_SHAPE_SLOTS,
    SNAPSHOT_RING_SIZE, pack_convex_counts,
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

    /// Uploads a compiled creation. The supplied device/queue may be Bevy's
    /// `RenderDevice` and `RenderQueue` deref targets, avoiding a second device.
    ///
    /// # Errors
    ///
    /// Returns [`GpuPhysicsError`] when a fixed scene capacity is exceeded.
    ///
    /// # Panics
    ///
    /// wgpu may panic if `device` is invalid or its implementation rejects the
    /// statically embedded, startup-validated WGSL module.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        creation: &CompiledCreation,
    ) -> Result<Self, GpuPhysicsError> {
        Self::new_with_config(device, queue, creation, GpuPhysicsConfig::default())
    }

    /// Uploads a compiled creation with fixed scene-wide pipeline settings.
    ///
    /// # Errors
    ///
    /// Returns [`GpuPhysicsError`] when a fixed scene capacity is exceeded.
    ///
    /// # Panics
    ///
    /// wgpu may panic if `device` is invalid or rejects an embedded WGSL module.
    pub fn new_with_config(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        creation: &CompiledCreation,
        pipeline_config: GpuPhysicsConfig,
    ) -> Result<Self, GpuPhysicsError> {
        Self::new_with_pipelines(
            device,
            queue,
            creation,
            pipeline_config,
            &GpuPhysicsPipelines::new(),
        )
    }

    /// Uploads a compiled scene while reusing previously compiled GPU kernels.
    ///
    /// # Errors
    ///
    /// Returns [`GpuPhysicsError`] when a fixed scene capacity is exceeded.
    ///
    /// # Panics
    ///
    /// wgpu may panic if `device` is invalid or rejects an embedded WGSL module.
    #[expect(clippy::too_many_lines)]
    pub fn new_with_pipelines(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        creation: &CompiledCreation,
        pipeline_config: GpuPhysicsConfig,
        pipelines: &GpuPhysicsPipelines,
    ) -> Result<Self, GpuPhysicsError> {
        if creation.compounds.len() > MAX_BODIES {
            return Err(GpuPhysicsError::BodyCapacity {
                required: creation.compounds.len(),
                capacity: MAX_BODIES,
            });
        }
        if creation.bearings.len() > MAX_BEARINGS {
            return Err(GpuPhysicsError::BearingCapacity {
                required: creation.bearings.len(),
                capacity: MAX_BEARINGS,
            });
        }
        if creation.colliders.len() > MAX_COLLIDERS {
            return Err(GpuPhysicsError::ColliderCapacity {
                required: creation.colliders.len(),
                capacity: MAX_COLLIDERS,
            });
        }

        let body_count = u32::try_from(creation.compounds.len()).unwrap_or(u32::MAX);
        let collider_count = u32::try_from(creation.colliders.len()).unwrap_or(u32::MAX);
        let bearing_count = u32::try_from(creation.bearings.len()).unwrap_or(u32::MAX);
        let suppression_count =
            u32::try_from(creation.collision_suppression.len()).unwrap_or(u32::MAX);
        let pair_capacity = contact_pair_capacity(creation.colliders.len());
        let positions = creation
            .compounds
            .iter()
            .map(|compound| vec4(compound.root_translation, 0.0))
            .collect::<Vec<_>>();
        let rotations = creation
            .compounds
            .iter()
            .map(|compound| {
                let rotation = compound.root_rotation;
                [rotation.x, rotation.y, rotation.z, rotation.w]
            })
            .collect::<Vec<_>>();
        let zero_vectors = vec![[0.0_f32; 4]; creation.compounds.len()];
        let inverse_masses = creation
            .compounds
            .iter()
            .map(|compound| compound.mass_properties.inverse_mass)
            .collect::<Vec<_>>();
        let body_components = creation
            .loop_topology
            .body_parents
            .iter()
            .map(|body| body.component_index)
            .collect::<Vec<_>>();
        let masses = creation
            .compounds
            .iter()
            .map(|compound| {
                let properties = compound.mass_properties;
                GpuMass {
                    inverse_mass: [properties.inverse_mass, 0.0, 0.0, 0.0],
                    inverse_inertia_x: vec4(properties.inverse_inertia.x_axis, 0.0),
                    inverse_inertia_y: vec4(properties.inverse_inertia.y_axis, 0.0),
                    inverse_inertia_z: vec4(properties.inverse_inertia.z_axis, 0.0),
                }
            })
            .collect::<Vec<_>>();
        let spatial_inertias = creation
            .compounds
            .iter()
            .map(|compound| {
                let properties = compound.mass_properties;
                GpuSpatialInertia {
                    mass: [properties.mass, 0.0, 0.0, 0.0],
                    inertia_x: vec4(properties.inertia.x_axis, 0.0),
                    inertia_y: vec4(properties.inertia.y_axis, 0.0),
                    inertia_z: vec4(properties.inertia.z_axis, 0.0),
                }
            })
            .collect::<Vec<_>>();
        let cylinder_ground_data = full_cylinder_ground_data(&creation.colliders);
        let mut convex_shapes: Vec<[f32; 4]> = Vec::new();
        let colliders = creation
            .colliders
            .iter()
            .zip(cylinder_ground_data)
            .map(|(collider, ground)| {
                let (local_rotation, half_extents, mut shape) = match &collider.shape {
                    ColliderShape::Cuboid {
                        local_rotation,
                        half_extents,
                    } => (
                        [
                            local_rotation.x,
                            local_rotation.y,
                            local_rotation.z,
                            local_rotation.w,
                        ],
                        vec4(*half_extents, ground.outer_radius),
                        [COLLIDER_SHAPE_CUBOID, 0, 0, 0],
                    ),
                    ColliderShape::Convex(convex) => {
                        let offset =
                            u32::try_from(convex_shapes.len()).expect("convex slot fits u32");
                        convex_shapes
                            .extend(convex.vertices.iter().map(|vertex| vec4(*vertex, 0.0)));
                        convex_shapes.extend(
                            convex
                                .face_planes
                                .iter()
                                .map(|plane| [plane.x, plane.y, plane.z, plane.w]),
                        );
                        convex_shapes
                            .extend(convex.edge_directions.iter().map(|edge| vec4(*edge, 0.0)));
                        let counts = pack_convex_counts(
                            u32::try_from(convex.vertices.len()).expect("vertex count fits u32"),
                            u32::try_from(convex.face_planes.len()).expect("face count fits u32"),
                            u32::try_from(convex.edge_directions.len())
                                .expect("edge count fits u32"),
                        );
                        (
                            [0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [COLLIDER_SHAPE_CONVEX, offset, counts, 0],
                        )
                    }
                };
                // Terrain contacts name no second body, so the solver discards
                // every one belonging to an immovable body. Marking the row lets
                // the contact kernel skip a full BVH descent that can only
                // produce discarded contacts.
                shape[3] = u32::from(
                    creation.compounds[collider.compound_index as usize]
                        .mass_properties
                        .inverse_mass
                        <= 0.0,
                );
                GpuCollider {
                    local_center: vec4(collider.local_center, ground.center_radius),
                    local_rotation,
                    half_extents,
                    metadata: [
                        collider.compound_index,
                        collider.source_part.index(),
                        collider.source_part.generation(),
                        ground.role,
                    ],
                    surface_response: [
                        collider.material_properties.static_friction,
                        collider.material_properties.dynamic_friction,
                        collider.material_properties.restitution,
                        collider.material_properties.rolling_resistance,
                    ],
                    surface_elasticity: [
                        collider.material_properties.nominal_block_compliance(),
                        collider.material_properties.youngs_modulus_pa,
                        0.0,
                        0.0,
                    ],
                    shape,
                }
            })
            .collect::<Vec<_>>();
        if convex_shapes.len() > MAX_CONVEX_SHAPE_SLOTS {
            return Err(GpuPhysicsError::ConvexShapeCapacity {
                required: convex_shapes.len(),
                capacity: MAX_CONVEX_SHAPE_SLOTS,
            });
        }
        // The buffer is fixed size, so an empty scene still needs one slot.
        if convex_shapes.is_empty() {
            convex_shapes.push([0.0; 4]);
        }
        let bearings = creation
            .bearings
            .iter()
            .map(|bearing| GpuBearing {
                local_anchor_a: vec4(bearing.local_anchor_a, bearing.kind.bounds()[0]),
                local_anchor_b: vec4(bearing.local_anchor_b, bearing.kind.bounds()[1]),
                local_axis_a: vec4(
                    bearing.local_axis_a,
                    if bearing.kind.is_translational() {
                        1.0
                    } else {
                        0.0
                    },
                ),
                local_axis_b: vec4(bearing.local_axis_b, 0.0),
                suspension: match bearing.kind {
                    mechanic_core::BearingKind::Suspension(s) => s.passive_rows()[0],
                    _ => [0.0; 4],
                },
                bump_stop: match bearing.kind {
                    mechanic_core::BearingKind::Suspension(s) => s.passive_rows()[1],
                    _ => [0.0; 4],
                },
                metadata: [
                    bearing.compound_a,
                    bearing.compound_b,
                    bearing.coordinate_index.unwrap_or(u32::MAX),
                    u32::from(bearing.coordinate_index.is_none()),
                ],
            })
            .collect::<Vec<_>>();
        let holds = Mutex::new(hold::HoldResources::new(
            masses.clone(),
            bearings.clone(),
            body_components.clone(),
        ));
        let suppressed_pairs = creation
            .collision_suppression
            .iter()
            .map(|pair| GpuPair {
                collider_a: pair[0],
                collider_b: pair[1],
            })
            .collect::<Vec<_>>();

        let config =
            create_uniform_buffer(device, "mechanic tick config", &GpuTickConfig::zeroed());
        let positions_buffer = create_storage_buffer(device, "mechanic positions", &positions);
        let rotations_buffer = create_storage_buffer(device, "mechanic rotations", &rotations);
        let linear_velocities =
            create_state_buffer(device, "mechanic linear velocities", &zero_vectors);
        let angular_velocities =
            create_state_buffer(device, "mechanic angular velocities", &zero_vectors);
        let inverse_masses =
            create_storage_buffer(device, "mechanic inverse masses", &inverse_masses);
        let diagnostics = create_buffer(
            device,
            "mechanic diagnostics",
            // The fixed readback header is followed by per-body contact counts.
            // Sharing this atomic scratch buffer keeps collision passes within
            // the baseline limit of eight storage-buffer bindings.
            &vec![0_u32; size_of::<GpuDiagnostics>() / size_of::<u32>() + body_count as usize],
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let diagnostics_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mechanic diagnostics readback"),
            size: size_of::<GpuDiagnostics>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let snapshot_readback_size = u64::from(body_count) * 16;
        let snapshot_positions_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mechanic snapshot positions readback"),
            size: snapshot_readback_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let snapshot_rotations_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mechanic snapshot rotations readback"),
            size: snapshot_readback_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let masses = create_storage_buffer(device, "mechanic mass rows", &masses);
        let spatial_inertias = create_readonly_storage_buffer(
            device,
            "mechanic direct spatial inertia rows",
            &spatial_inertias,
        );
        let colliders = create_readonly_storage_buffer(device, "mechanic colliders", &colliders);
        let convex_shapes =
            create_readonly_storage_buffer(device, "mechanic convex shapes", &convex_shapes);
        let bearings = create_storage_buffer(device, "mechanic bearings", &bearings);
        let suppressed_pairs = create_readonly_storage_buffer(
            device,
            "mechanic collision suppression",
            &suppressed_pairs,
        );

        let mechanism = create_mechanism_resources(
            device,
            pipelines,
            creation,
            &config,
            &positions_buffer,
            &rotations_buffer,
            &diagnostics,
            &bearings,
            &masses,
            &spatial_inertias,
            &linear_velocities,
            &angular_velocities,
        );

        let shader = shader_module(
            pipelines,
            device,
            "mechanic physics kernels",
            include_str!("kernels/physics.wgsl"),
        );
        let integration_pipeline = compute_pipeline(
            pipelines,
            device,
            "mechanic integrate and snapshot",
            &shader,
            "integrate",
        );
        let external_impulse = create_uniform_buffer(
            device,
            "mechanic external impulse",
            &GpuExternalImpulseBatch::zeroed(),
        );
        let external_impulse_pipeline = compute_pipeline(
            pipelines,
            device,
            "mechanic apply external impulse",
            &shader,
            "apply_external_impulse",
        );
        let external_impulse_bind_group = bind_group(
            device,
            "mechanic external impulse bindings",
            &external_impulse_pipeline,
            &[
                entry(1, &positions_buffer),
                entry(2, &rotations_buffer),
                entry(3, &linear_velocities),
                entry(4, &angular_velocities),
                entry(7, &masses),
                entry(8, &external_impulse),
            ],
        );
        let layout = integration_pipeline.get_bind_group_layout(0);
        let snapshot_shader = shader_module(
            pipelines,
            device,
            "mechanic snapshot kernel",
            include_str!("kernels/snapshot.wgsl"),
        );
        let snapshot_pipeline = compute_pipeline(
            pipelines,
            device,
            "mechanic publish snapshot",
            &snapshot_shader,
            "publish_snapshot",
        );
        let timestamps = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| {
                let boundary = compute_pipeline(pipelines, device, "mechanic timestamp boundary", &shader_module(pipelines, device, "mechanic timestamp boundary",
                    "@group(0) @binding(0) var<storage, read_write> positions: array<atomic<u32>>; @compute @workgroup_size(1) fn boundary() { atomicOr(&positions[0], 0u); }"), "boundary");
                let boundary_bindings = bind_group(device, "mechanic ordered timestamp boundary", &boundary, &[entry(0, &positions_buffer)]);
                TimestampResources {
                    boundary,
                    boundary_bindings,
                query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("mechanic physics timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 28,
                }),
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mechanic timestamp resolve"),
                    size: 224,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mechanic timestamp readback"),
                    size: 224,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                period_nanoseconds: f64::from(queue.get_timestamp_period()),
                }
            });
        let mut snapshots = Vec::with_capacity(SNAPSHOT_RING_SIZE);
        let mut bind_groups = Vec::with_capacity(SNAPSHOT_RING_SIZE);
        let mut snapshot_bind_groups = Vec::with_capacity(SNAPSHOT_RING_SIZE);
        for slot in 0..SNAPSHOT_RING_SIZE {
            let snapshot = SnapshotBuffers {
                positions: create_buffer(
                    device,
                    &format!("mechanic snapshot {slot} positions"),
                    &positions,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                ),
                rotations: create_buffer(
                    device,
                    &format!("mechanic snapshot {slot} rotations"),
                    &rotations,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                ),
            };
            bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mechanic integration bindings"),
                layout: &layout,
                entries: &[
                    entry(0, &config),
                    entry(1, &positions_buffer),
                    entry(2, &rotations_buffer),
                    entry(3, &linear_velocities),
                    entry(4, &angular_velocities),
                    entry(5, &inverse_masses),
                    entry(6, &diagnostics),
                    entry(9, &mechanism.root_flags),
                ],
            }));
            snapshot_bind_groups.push(bind_group(
                device,
                "mechanic snapshot bindings",
                &snapshot_pipeline,
                &[
                    entry(0, &positions_buffer),
                    entry(1, &rotations_buffer),
                    entry(2, &snapshot.positions),
                    entry(3, &snapshot.rotations),
                    entry(4, &diagnostics),
                ],
            ));
            snapshots.push(snapshot);
        }

        let collision = create_collision_resources(
            device,
            pipelines,
            creation.colliders.len(),
            usize::try_from(pair_capacity).unwrap_or(MAX_CONTACT_PAIRS),
            &config,
            &positions_buffer,
            &rotations_buffer,
            &linear_velocities,
            &angular_velocities,
            &masses,
            &diagnostics,
            &colliders,
            &convex_shapes,
            &suppressed_pairs,
            &mechanism.drive_constraints,
            &body_components,
            pipeline_config.mechanism_self_collisions,
        );
        let bearing_shader = shader_module(
            pipelines,
            device,
            "mechanic bearing kernels",
            include_str!("kernels/bearings.wgsl"),
        );
        let bearing_entry_point = if mechanism.active {
            "validate_mechanism_bearings"
        } else {
            "validate_bearings"
        };
        let bearing_pipeline = compute_pipeline(
            pipelines,
            device,
            "mechanic validate bearings",
            &bearing_shader,
            bearing_entry_point,
        );
        let bearing_bind_group = if mechanism.active {
            let final_links = if mechanism.final_is_a {
                &mechanism.links_a
            } else {
                &mechanism.links_b
            };
            bind_group(
                device,
                "mechanic local bearing bindings",
                &bearing_pipeline,
                &[
                    entry(0, &config),
                    entry(3, &diagnostics),
                    entry(4, &bearings),
                    entry(5, final_links),
                ],
            )
        } else {
            bind_group(
                device,
                "mechanic bearing bindings",
                &bearing_pipeline,
                &[
                    entry(0, &config),
                    entry(1, &positions_buffer),
                    entry(2, &rotations_buffer),
                    entry(3, &diagnostics),
                    entry(4, &bearings),
                ],
            )
        };

        let mut free_bodies: Vec<u32> = creation
            .compounds
            .iter()
            .map(|body| u32::from(!body.is_static))
            .collect();
        for bearing in &creation.bearings {
            free_bodies[bearing.compound_a as usize] = 0;
            free_bodies[bearing.compound_b as usize] = 0;
        }
        free_bodies.resize(free_bodies.len().max(1), 0);
        let terrain_free_bodies = create_readonly_storage_buffer(
            device,
            "mechanic unjointed terrain sweep bodies",
            &free_bodies,
        );

        // Make the upload boundary explicit before the first fixed tick.
        queue.write_buffer(&diagnostics, 0, bytes_of(&GpuDiagnostics::zeroed()));
        Ok(Self {
            holds,
            body_count,
            collider_count,
            bearing_count,
            suppression_count,
            pair_capacity,
            pipeline_config,
            config,
            positions: positions_buffer,
            rotations: rotations_buffer,
            linear_velocities,
            angular_velocities,
            inverse_masses,
            diagnostics,
            diagnostics_readback,
            snapshot_positions_readback,
            snapshot_rotations_readback,
            masses,
            _spatial_inertias: spatial_inertias,
            colliders,
            convex_shapes,
            bearings,
            external_impulse,
            external_impulse_pipeline,
            external_impulse_bind_group,
            snapshots,
            bind_groups,
            integration_pipeline,
            mechanism,
            collision,
            terrain_recovery: None,
            terrain_scene: terrain::TerrainGpuScene::default(),
            terrain_has_free_bodies: free_bodies.contains(&1),
            terrain_free_bodies,
            bearing_pipeline,
            bearing_bind_group,
            snapshot_pipeline,
            snapshot_bind_groups,
            timestamps,
            submission_sequence: AtomicU64::new(0),
            async_readback_enabled: AtomicBool::new(false),
            readback_timing_enabled: AtomicBool::new(false),
            async_readbacks: Mutex::new(Vec::new()),
        })
    }

    /// Adds a world-space impulse at a world-space point on one compound body.
    ///
    /// Static bodies ignore the impulse. The submission is ordered before later
    /// fixed ticks submitted to the same queue.
    ///
    /// # Errors
    ///
    /// Returns [`GpuImpulseError`] for an invalid body row or non-finite input.
    pub fn apply_impulse(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        body_index: u32,
        world_point: Vec3,
        impulse: Vec3,
    ) -> Result<wgpu::SubmissionIndex, GpuImpulseError> {
        self.apply_impulses(
            device,
            queue,
            &[GpuExternalImpulse::new(body_index, world_point, impulse)],
        )
    }

    /// Validates and applies at most one fixed staging batch as a serial pass.
    ///
    /// Validation covers every row before the queue is modified, so invalid input
    /// can never produce a partial submission.
    ///
    /// # Errors
    ///
    /// Returns [`GpuImpulseError`] when the batch exceeds staging capacity or any
    /// row has an invalid body index or non-finite point/vector.
    pub fn apply_impulses(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        impulses: &[GpuExternalImpulse],
    ) -> Result<wgpu::SubmissionIndex, GpuImpulseError> {
        if impulses.len() > EXTERNAL_IMPULSE_BATCH_CAPACITY {
            return Err(GpuImpulseError::BatchCapacity {
                provided: impulses.len(),
                capacity: EXTERNAL_IMPULSE_BATCH_CAPACITY,
            });
        }
        self.validate_impulses(impulses)?;
        let mut batch = GpuExternalImpulseBatch::zeroed();
        batch.metadata[0] = u32::try_from(impulses.len()).unwrap_or(u32::MAX);
        batch.rows[..impulses.len()].copy_from_slice(impulses);
        queue.write_buffer(&self.external_impulse, 0, bytes_of(&batch));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mechanic external impulse"),
        });
        direct_compute_pass(
            &mut encoder,
            "mechanic apply external impulse",
            &self.external_impulse_pipeline,
            &self.external_impulse_bind_group,
            1,
            None,
        );
        Ok(queue.submit([encoder.finish()]))
    }

    fn validate_impulses(&self, impulses: &[GpuExternalImpulse]) -> Result<(), GpuImpulseError> {
        for row in impulses {
            let body_index = row.body_index();
            if body_index >= self.body_count {
                return Err(GpuImpulseError::BodyIndexOutOfRange {
                    body_index,
                    body_count: self.body_count,
                });
            }
            if !row.is_finite() {
                return Err(GpuImpulseError::NonFinite);
            }
        }
        Ok(())
    }

    /// Replaces all authoritative body transforms and velocities at a safe app-owned boundary.
    ///
    /// This is used when a live world rebuilds its static construction topology after an edit;
    /// unchanged moving compounds keep their latest pose and motion.
    ///
    /// # Errors
    ///
    /// Returns [`GpuBodyStateError`] when row counts differ or any lane is non-finite.
    pub fn write_body_states(
        &self,
        queue: &wgpu::Queue,
        transforms: &[GpuTransform],
        velocities: &[GpuVelocity],
    ) -> Result<(), GpuBodyStateError> {
        let expected = self.body_count;
        if transforms.len() != expected as usize || velocities.len() != expected as usize {
            return Err(GpuBodyStateError::BodyCount {
                provided: transforms.len().max(velocities.len()),
                expected,
            });
        }
        let finite = transforms.iter().all(|state| {
            state.position.into_iter().all(f32::is_finite)
                && state.rotation.into_iter().all(f32::is_finite)
        }) && velocities.iter().all(|state| {
            state.linear.into_iter().all(f32::is_finite)
                && state.angular.into_iter().all(f32::is_finite)
        });
        if !finite {
            return Err(GpuBodyStateError::NonFinite);
        }
        let positions = transforms
            .iter()
            .map(|state| state.position)
            .collect::<Vec<_>>();
        let rotations = transforms
            .iter()
            .map(|state| state.rotation)
            .collect::<Vec<_>>();
        let linear = velocities
            .iter()
            .map(|state| state.linear)
            .collect::<Vec<_>>();
        let angular = velocities
            .iter()
            .map(|state| state.angular)
            .collect::<Vec<_>>();
        queue.write_buffer(&self.positions, 0, cast_slice(&positions));
        queue.write_buffer(&self.rotations, 0, cast_slice(&rotations));
        queue.write_buffer(&self.linear_velocities, 0, cast_slice(&linear));
        queue.write_buffer(&self.angular_velocities, 0, cast_slice(&angular));
        Ok(())
    }

    /// Assigns the same explicit flat collision plane to every collider.
    pub fn write_ground_plane(&self, queue: &wgpu::Queue, normal: Vec3, offset: f32) {
        let length = normal.length();
        let plane = if length > 0.0 {
            GpuGroundPlane {
                normal: normal / length,
                offset: offset / length,
            }
        } else {
            GpuGroundPlane::DISABLED
        };
        let planes = vec![plane; self.collider_count as usize];
        // This method constructs exactly one finite row per uploaded collider.
        let _ = self.write_ground_planes(queue, &planes);
    }

    /// Replaces the terrain-support plane independently for every collider.
    ///
    /// Per-collider planes let a large mechanism rest on the streamed terrain
    /// beneath each of its parts without pretending the entire world is one
    /// moving infinite plane.
    ///
    /// # Errors
    ///
    /// Returns [`GpuGroundPlaneError`] when the row count differs from the
    /// uploaded collider count or a row is not finite.
    pub fn write_ground_planes(
        &self,
        queue: &wgpu::Queue,
        planes: &[GpuGroundPlane],
    ) -> Result<(), GpuGroundPlaneError> {
        if planes.len() != self.collider_count as usize {
            return Err(GpuGroundPlaneError::PlaneCount {
                provided: planes.len(),
                expected: self.collider_count,
            });
        }
        if planes
            .iter()
            .any(|plane| !plane.normal.is_finite() || !plane.offset.is_finite())
        {
            return Err(GpuGroundPlaneError::NonFinite);
        }
        let concrete = ConstructionMaterial::Concrete.properties();
        let rows = planes
            .iter()
            .map(|plane| {
                let length = plane.normal.length();
                let (normal, offset) = if length > 0.0 {
                    (plane.normal / length, plane.offset / length)
                } else {
                    (Vec3::ZERO, 0.0)
                };
                GpuGroundSurface {
                    response: [
                        concrete.static_friction,
                        concrete.dynamic_friction,
                        concrete.restitution,
                        concrete.rolling_resistance,
                    ],
                    elasticity: [
                        concrete.nominal_block_compliance(),
                        concrete.youngs_modulus_pa,
                        0.0,
                        0.0,
                    ],
                    plane: [normal.x, normal.y, normal.z, offset],
                }
            })
            .collect::<Vec<_>>();
        queue.write_buffer(&self.collision.ground_surfaces, 0, cast_slice(&rows));
        Ok(())
    }

    /// Enables non-blocking per-tick telemetry and prototype snapshot staging.
    ///
    /// This is opt-in because correctness tests and headless benchmarks use
    /// explicit synchronous sampling and should not allocate an application
    /// readback ring for unobserved warm-up ticks.
    pub fn enable_async_readback(&self) {
        self.async_readback_enabled.store(true, Ordering::Release);
    }

    /// Enables callback timestamps for subsequently submitted asynchronous ticks.
    /// Adds no polling, waits, queue work, or readback slots.
    pub fn enable_readback_timing(&self) {
        self.readback_timing_enabled.store(true, Ordering::Release);
    }

    /// Number of ticks that can be staged without waiting or allocating.
    ///
    /// The application uses this as its in-flight submission budget. Logical
    /// scheduler ticks remain in the CPU backlog until a fixed ring slot is
    /// available, preventing a slow GPU from turning into an unbounded queue.
    pub fn async_readback_slots_available(&self) -> usize {
        if !self.async_readback_enabled.load(Ordering::Acquire) {
            return usize::MAX;
        }
        let slots = self
            .async_readbacks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        slots.iter().filter(|slot| slot.pending.is_none()).count()
            + ASYNC_READBACK_RING_SIZE.saturating_sub(slots.len())
    }

    /// Encodes and submits one 60 Hz integration/publication pass.
    pub fn dispatch_tick(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tick_index: u64,
    ) -> GpuTickSubmission {
        self.encode_and_submit_tick(device, queue, tick_index)
    }

    /// Applies all pending impulses in ordered serial batches, then dispatches a tick.
    ///
    /// Every row is validated before the first queue write. Sets larger than the fixed
    /// staging buffer are split without dropping contacts.
    ///
    /// # Errors
    ///
    /// Returns [`GpuImpulseError`] when any pending row has an invalid body index or
    /// non-finite point/vector.
    pub fn dispatch_tick_with_impulses(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tick_index: u64,
        impulses: &[GpuExternalImpulse],
    ) -> Result<GpuTickSubmission, GpuImpulseError> {
        self.validate_impulses(impulses)?;
        for batch in impulses.chunks(EXTERNAL_IMPULSE_BATCH_CAPACITY) {
            self.apply_impulses(device, queue, batch)?;
        }
        Ok(self.encode_and_submit_tick(device, queue, tick_index))
    }

    #[expect(clippy::too_many_lines)]
    fn encode_and_submit_tick(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tick_index: u64,
    ) -> GpuTickSubmission {
        let submission_sequence = self.submission_sequence.fetch_add(1, Ordering::Relaxed) + 1;
        let encoding_started = Instant::now();
        let snapshot_slot = u8::try_from(tick_index % 3).unwrap_or(0);
        let config = GpuTickConfig {
            body_count: self.body_count,
            tick_index: wrapping_u32(tick_index),
            snapshot_slot: u32::from(snapshot_slot),
            collider_count: self.collider_count,
            delta_seconds: TICK_SECONDS_F32,
            gravity_y: -STANDARD_GRAVITY_M_S2_F32,
            linear_damping: 0.999,
            angular_damping: 0.98,
            bearing_count: self.bearing_count,
            suppression_count: self.suppression_count,
            pair_capacity: self.pair_capacity,
            flags: u32::from(self.pipeline_config.collisions_enabled)
                | (u32::from(self.solver_route() == GpuSolverRoute::FusedSmallMechanism) << 1),
            hash_capacity: u32::try_from(BROADPHASE_HASH_CAPACITY).unwrap_or(u32::MAX),
            solver_iterations: self.pipeline_config.solver_iterations.max(1),
            reserved_a: self.collision.lbvh.sort_count,
            reserved_b: self.mechanism.coordinate_count,
        };
        queue.write_buffer(&self.config, 0, bytes_of(&config));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mechanic physics tick"),
        });
        // Error flags are a persistent GPU failure latch. Per-tick counters are
        // cleared independently, so a delayed CPU readback can never erase a
        // terminal failure reported by an earlier submission.
        encoder.clear_buffer(&self.diagnostics, 4, None);
        let run_collisions = self.pipeline_config.collisions_enabled && self.collider_count > 0;
        let run_bearings = self.bearing_count > 0;
        if run_collisions {
            encoder.clear_buffer(&self.collision.lbvh.node_parents, 0, None);
            encoder.clear_buffer(&self.collision.lbvh.node_visits, 0, None);
            encoder.clear_buffer(&self.collision.velocity_deltas, 0, None);
        }
        if self.mechanism.active {
            encoder.clear_buffer(&self.mechanism.velocity_deltas, 0, None);
        }
        if run_collisions
            && self.collision.terrain_triangle_count > 0
            && let Some(recovery) = &self.terrain_recovery
        {
            recovery.capture(self, &mut encoder);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mechanic integrate pass"),
                timestamp_writes: timestamp_writes(self.timestamps.as_ref(), Some(0), Some(1)),
            });
            pass.set_pipeline(&self.integration_pipeline);
            pass.set_bind_group(0, &self.bind_groups[usize::from(snapshot_slot)], &[]);
            pass.dispatch_workgroups(self.body_count.div_ceil(256), 1, 1);
        }
        if self.mechanism.active {
            self.encode_mechanism_passes(&mut encoder);
        }
        if run_collisions
            && self.collision.terrain_triangle_count > 0
            && let Some(recovery) = &self.terrain_recovery
        {
            self.encode_terrain_timestamp(&mut encoder, 16);
            if self.terrain_has_free_bodies {
                recovery.sweep.encode(self, &mut encoder);
            }
            self.encode_terrain_timestamp(&mut encoder, 17);
        }
        if run_collisions {
            self.encode_collision_passes(&mut encoder);
        }
        if self.mechanism.active && run_collisions {
            self.encode_post_contact_mechanism(&mut encoder);
        }
        if run_collisions
            && self.collision.terrain_triangle_count > 0
            && let Some(recovery) = &self.terrain_recovery
        {
            self.encode_terrain_timestamp(&mut encoder, 18);
            recovery.encode(self, &mut encoder);
            self.encode_terrain_timestamp(&mut encoder, 19);
        }
        if self.mechanism.active {
            direct_compute_pass(
                &mut encoder,
                "mechanic validate articulated state",
                &self.mechanism.validate_state_pipeline,
                &self.mechanism.validate_state_bind_group,
                self.body_count.div_ceil(256),
                None,
            );
        }
        if run_bearings {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mechanic bearing validation"),
                timestamp_writes: timestamp_writes(self.timestamps.as_ref(), Some(10), Some(11)),
            });
            pass.set_pipeline(&self.bearing_pipeline);
            pass.set_bind_group(0, &self.bearing_bind_group, &[]);
            pass.dispatch_workgroups(self.bearing_count.div_ceil(256), 1, 1);
        }
        direct_compute_pass(
            &mut encoder,
            "mechanic snapshot publication",
            &self.snapshot_pipeline,
            &self.snapshot_bind_groups[usize::from(snapshot_slot)],
            self.body_count.div_ceil(256),
            timestamp_writes(self.timestamps.as_ref(), Some(12), Some(13)),
        );
        encoder.copy_buffer_to_buffer(
            &self.diagnostics,
            0,
            &self.diagnostics_readback,
            0,
            u64::try_from(size_of::<GpuDiagnostics>()).unwrap_or(32),
        );
        if let Some(timestamps) = &self.timestamps {
            encoder.resolve_query_set(&timestamps.query_set, 0..28, &timestamps.resolve, 0);
            encoder.copy_buffer_to_buffer(&timestamps.resolve, 0, &timestamps.readback, 0, 224);
        }
        let mut async_readbacks = self
            .async_readback_enabled
            .load(Ordering::Acquire)
            .then(|| {
                self.async_readbacks
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
            });
        let async_slot_index = async_readbacks.as_mut().and_then(|slots| {
            let index = if let Some(index) = slots.iter().position(|slot| slot.pending.is_none()) {
                index
            } else if slots.len() < ASYNC_READBACK_RING_SIZE {
                let index = slots.len();
                slots.push(create_async_readback_slot(
                    device,
                    self.body_count,
                    self.mechanism.coordinate_count,
                    self.timestamps.is_some(),
                    index,
                ));
                index
            } else {
                return None;
            };
            let slot = &slots[index];
            let diagnostics_size = u64::try_from(size_of::<GpuDiagnostics>()).unwrap_or(32);
            let snapshot_size = u64::from(self.body_count) * 16;
            encoder.copy_buffer_to_buffer(
                &self.diagnostics,
                0,
                &slot.diagnostics,
                0,
                diagnostics_size,
            );
            if let (Some(timestamps), Some(readback)) = (&self.timestamps, slot.timestamps.as_ref())
            {
                encoder.copy_buffer_to_buffer(&timestamps.resolve, 0, readback, 0, 224);
            }
            let snapshot = &self.snapshots[usize::from(snapshot_slot)];
            encoder.copy_buffer_to_buffer(
                snapshot.positions(),
                0,
                &slot.positions,
                0,
                snapshot_size,
            );
            encoder.copy_buffer_to_buffer(
                snapshot.rotations(),
                0,
                &slot.rotations,
                0,
                snapshot_size,
            );
            // Copies share the tick command encoder, before any subsequent tick can
            // mutate the source buffers. Zero-coordinate scenes retain a dummy row.
            for (source, destination, size) in [
                (
                    &self.linear_velocities,
                    &slot.linear_velocities,
                    snapshot_size,
                ),
                (
                    &self.angular_velocities,
                    &slot.angular_velocities,
                    snapshot_size,
                ),
                (
                    &self.mechanism.coordinates,
                    &slot.coordinates,
                    u64::from(self.mechanism.coordinate_count) * 8,
                ),
            ] {
                if size != 0 {
                    encoder.copy_buffer_to_buffer(source, 0, destination, 0, size);
                }
            }
            Some(index)
        });
        let encoding_ms = encoding_started.elapsed().as_secs_f64() * 1_000.0;
        let finalization_started = Instant::now();
        let command_buffer = encoder.finish();
        let finalization_ms = finalization_started.elapsed().as_secs_f64() * 1_000.0;
        let submission_started = Instant::now();
        let submission_index = queue.submit([command_buffer]);
        let submission_finished = Instant::now();
        let submission_ms = submission_finished
            .duration_since(submission_started)
            .as_secs_f64()
            * 1_000.0;
        let readback_setup_started = Instant::now();
        if let (Some(slots), Some(index)) = (&mut async_readbacks, async_slot_index) {
            begin_async_mapping(
                &mut slots[index],
                submission_sequence,
                tick_index,
                snapshot_slot,
                submission_finished,
                self.readback_timing_enabled.load(Ordering::Acquire),
            );
        }
        let readback_setup_ms = readback_setup_started.elapsed().as_secs_f64() * 1_000.0;
        GpuTickSubmission {
            submission_sequence,
            tick_index,
            snapshot_slot,
            submission_index,
            cpu_timings: GpuSubmissionTimings {
                encoding_ms,
                finalization_ms,
                submission_ms,
                readback_setup_ms,
            },
        }
    }

    fn encode_mechanism_passes(&self, encoder: &mut wgpu::CommandEncoder) {
        let mechanism = &self.mechanism;
        let workgroups = self.body_count.div_ceil(256);
        direct_compute_pass(
            encoder,
            "mechanic prepare drive constraints",
            &mechanism.prepare_drives_pipeline,
            &mechanism.prepare_drives_bind_group,
            self.bearing_count.div_ceil(256),
            None,
        );
        self.encode_bearing_velocity_projection(encoder, true);
        direct_compute_pass(
            encoder,
            "mechanic advance reduced coordinates",
            &mechanism.advance_coordinates_pipeline,
            &mechanism.advance_coordinates_bind_group,
            workgroups,
            None,
        );
        self.encode_mechanism_pose_projection(encoder);
        direct_compute_pass(
            encoder,
            "mechanic reconstruct body velocities",
            &mechanism.reconstruct_velocities_pipeline,
            &mechanism.reconstruct_velocities_bind_group,
            1,
            timestamp_writes(self.timestamps.as_ref(), None, Some(3)),
        );
    }

    // Metal drops timestamps on empty passes. A bit-preserving atomic access
    // orders markers against pose readers/writers, including zero-work dispatches.
    fn encode_terrain_timestamp(&self, encoder: &mut wgpu::CommandEncoder, index: u32) {
        if let Some(timestamps) = &self.timestamps {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mechanic terrain stage boundary"),
                timestamp_writes: timestamp_writes(self.timestamps.as_ref(), Some(index), None),
            });
            pass.set_pipeline(&timestamps.boundary);
            pass.set_bind_group(0, &timestamps.boundary_bindings, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    fn encode_recovery_pose_projection(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gate: &wgpu::Buffer,
    ) {
        self.encode_mechanism_pose_projection_gated(encoder, Some(gate));
    }

    fn encode_mechanism_pose_projection(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_mechanism_pose_projection_gated(encoder, None);
    }

    fn encode_mechanism_pose_projection_gated(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gate: Option<&wgpu::Buffer>,
    ) {
        let mechanism = &self.mechanism;
        let workgroups = self.body_count.div_ceil(256);
        self.encode_mechanism_forward_kinematics(encoder, gate, 0);
        if mechanism.closure_count > 0 {
            const CLOSURE_CORRECTION_STEPS: u32 = 12;
            for step in 0..CLOSURE_CORRECTION_STEPS {
                encoder.clear_buffer(&mechanism.closure_accumulators, 0, None);
                encoder.clear_buffer(&mechanism.closure_state, 0, None);
                if step == 0
                    && let Some(gate) = gate
                {
                    indirect_compute_pass(
                        encoder,
                        "mechanic evaluate recovery closures",
                        &mechanism.evaluate_closures_pipeline,
                        &mechanism.evaluate_closures_bind_group,
                        gate,
                        32,
                        None,
                    );
                } else if step == 0 {
                    direct_compute_pass(
                        encoder,
                        "mechanic evaluate closures",
                        &mechanism.evaluate_closures_pipeline,
                        &mechanism.evaluate_closures_bind_group,
                        self.bearing_count.div_ceil(256),
                        None,
                    );
                } else {
                    indirect_compute_pass(
                        encoder,
                        "mechanic evaluate closures",
                        &mechanism.evaluate_closures_pipeline,
                        &mechanism.evaluate_closures_bind_group,
                        &mechanism.closure_indirect_args,
                        0,
                        None,
                    );
                }
                direct_compute_pass(
                    encoder,
                    "mechanic finalize closures",
                    &mechanism.finalize_closures_pipeline,
                    &mechanism.finalize_closures_bind_group,
                    1,
                    None,
                );
                indirect_compute_pass(
                    encoder,
                    "mechanic closure Newton PCG step",
                    &mechanism.apply_closure_step_pipeline,
                    &mechanism.apply_closure_step_bind_group,
                    &mechanism.closure_indirect_args,
                    12,
                    None,
                );
                self.encode_mechanism_forward_kinematics(
                    encoder,
                    Some(&mechanism.closure_indirect_args),
                    12,
                );
            }
        }
        let (pipeline, bindings) = if mechanism.final_is_a {
            (
                &mechanism.publish_a_pipeline,
                &mechanism.publish_a_bind_group,
            )
        } else {
            (
                &mechanism.publish_b_pipeline,
                &mechanism.publish_b_bind_group,
            )
        };
        if let Some(gate) = gate {
            indirect_compute_pass(
                encoder,
                "mechanic publish recovered mechanism poses",
                pipeline,
                bindings,
                gate,
                0,
                None,
            );
            return;
        }
        direct_compute_pass(
            encoder,
            "mechanic publish mechanism poses",
            pipeline,
            bindings,
            workgroups,
            None,
        );
    }

    fn encode_bearing_velocity_projection(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_start: bool,
    ) {
        if uses_fused_velocity_schedule(self.bearing_count, self.body_count) {
            direct_compute_pass(
                encoder,
                "mechanic fused small bearing velocity projection",
                &self.mechanism.project_small_velocity_pipeline,
                &self.mechanism.project_small_velocity_bind_group,
                1,
                if timestamp_start {
                    timestamp_writes(self.timestamps.as_ref(), Some(2), None)
                } else {
                    None
                },
            );
            return;
        }
        for iteration in 0..self.pipeline_config.solver_iterations.max(1) {
            self.encode_bearing_velocity_projection_iteration(
                encoder,
                timestamp_start && iteration == 0,
                false,
            );
        }
    }

    fn encode_bearing_velocity_projection_iteration(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_start: bool,
        serial: bool,
    ) {
        let mechanism = &self.mechanism;
        if serial {
            direct_compute_pass(
                encoder,
                "mechanic project bearing velocities serially",
                &mechanism.project_velocity_serial_pipeline,
                &mechanism.project_velocity_serial_bind_group,
                1,
                if timestamp_start {
                    timestamp_writes(self.timestamps.as_ref(), Some(2), None)
                } else {
                    None
                },
            );
            return;
        }
        direct_compute_pass(
            encoder,
            "mechanic project bearing velocities",
            &mechanism.project_velocity_pipeline,
            &mechanism.project_velocity_bind_group,
            self.bearing_count.div_ceil(256),
            if timestamp_start {
                timestamp_writes(self.timestamps.as_ref(), Some(2), None)
            } else {
                None
            },
        );
        direct_compute_pass(
            encoder,
            "mechanic apply bearing velocity deltas",
            &mechanism.apply_velocity_pipeline,
            &mechanism.apply_velocity_bind_group,
            self.body_count.div_ceil(256),
            None,
        );
    }

    fn encode_contact_bearing_velocity_projection_iteration(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        serial: bool,
    ) {
        if serial {
            indirect_compute_pass(
                encoder,
                "mechanic project contact bearing velocities serially",
                &self.mechanism.project_velocity_serial_pipeline,
                &self.mechanism.project_velocity_serial_bind_group,
                &self.collision.indirect_args,
                48,
                None,
            );
        } else {
            self.encode_bearing_velocity_projection_iteration(encoder, false, false);
        }
    }

    fn encode_post_contact_mechanism(&self, encoder: &mut wgpu::CommandEncoder) {
        let mechanism = &self.mechanism;
        if !mechanism.has_dynamic_root {
            self.encode_bearing_velocity_projection(encoder, false);
        }
        direct_compute_pass(
            encoder,
            "mechanic capture reduced velocities",
            &mechanism.capture_coordinates_pipeline,
            &mechanism.capture_coordinates_bind_group,
            self.body_count.div_ceil(256),
            if mechanism.has_dynamic_root {
                timestamp_writes(self.timestamps.as_ref(), None, Some(9))
            } else {
                None
            },
        );
        if !mechanism.has_dynamic_root {
            direct_compute_pass(
                encoder,
                "mechanic reconstruct grounded post-contact velocities",
                &mechanism.reconstruct_velocities_pipeline,
                &mechanism.reconstruct_velocities_bind_group,
                1,
                timestamp_writes(self.timestamps.as_ref(), None, Some(9)),
            );
        }
    }

    fn encode_mechanism_forward_kinematics(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        indirect: Option<&wgpu::Buffer>,
        indirect_offset: u64,
    ) {
        let mechanism = &self.mechanism;
        let workgroups = self.body_count.div_ceil(256);
        let mut dispatch = |label: &str,
                            pipeline: &wgpu::ComputePipeline,
                            bindings: &wgpu::BindGroup,
                            timestamp_writes| {
            if let Some(indirect) = indirect {
                indirect_compute_pass(
                    encoder,
                    label,
                    pipeline,
                    bindings,
                    indirect,
                    indirect_offset,
                    timestamp_writes,
                );
            } else {
                direct_compute_pass(
                    encoder,
                    label,
                    pipeline,
                    bindings,
                    workgroups,
                    timestamp_writes,
                );
            }
        };
        dispatch(
            "mechanic prepare mechanism links",
            &mechanism.prepare_pipeline,
            &mechanism.prepare_bind_group,
            None,
        );
        let mut final_is_a = true;
        for _ in 0..mechanism.pointer_jump_rounds {
            if final_is_a {
                dispatch(
                    "mechanic mechanism jump A to B",
                    &mechanism.jump_a_to_b_pipeline,
                    &mechanism.jump_a_to_b_bind_group,
                    None,
                );
            } else {
                dispatch(
                    "mechanic mechanism jump B to A",
                    &mechanism.jump_b_to_a_pipeline,
                    &mechanism.jump_b_to_a_bind_group,
                    None,
                );
            }
            final_is_a = !final_is_a;
        }
        debug_assert_eq!(final_is_a, mechanism.final_is_a);
    }

    #[expect(clippy::too_many_lines)]
    fn encode_collision_passes(&self, encoder: &mut wgpu::CommandEncoder) {
        let collision = &self.collision;
        let lbvh = &collision.lbvh;
        let sort_workgroups = lbvh.sort_count.div_ceil(256);
        direct_compute_pass(
            encoder,
            "mechanic world inverse inertias",
            &collision.update_world_masses_pipeline,
            &collision.update_world_masses_bind_group,
            self.body_count.div_ceil(256),
            timestamp_writes(self.timestamps.as_ref(), Some(4), None),
        );
        direct_compute_pass(
            encoder,
            "mechanic LBVH Morton codes",
            &lbvh.compute_morton_pipeline,
            &lbvh.compute_morton_bind_group,
            sort_workgroups,
            None,
        );
        direct_compute_pass(
            encoder,
            "mechanic LBVH local sort",
            &lbvh.sort_local_initial_pipeline,
            &lbvh.sort_local_initial_bind_group,
            sort_workgroups,
            None,
        );
        for &(parameter_offset, local_merge) in &lbvh.sort_steps {
            encoder.copy_buffer_to_buffer(
                &lbvh.sort_params_upload,
                parameter_offset,
                &lbvh.sort_params,
                0,
                16,
            );
            let (pipeline, bindings, label) = if local_merge {
                (
                    &lbvh.sort_local_merge_pipeline,
                    &lbvh.sort_local_merge_bind_group,
                    "mechanic LBVH local merge",
                )
            } else {
                (
                    &lbvh.sort_global_pipeline,
                    &lbvh.sort_global_bind_group,
                    "mechanic LBVH global merge",
                )
            };
            direct_compute_pass(encoder, label, pipeline, bindings, sort_workgroups, None);
        }
        direct_compute_pass(
            encoder,
            "mechanic LBVH topology",
            &lbvh.build_topology_pipeline,
            &lbvh.build_topology_bind_group,
            self.collider_count.saturating_sub(1).max(1).div_ceil(256),
            None,
        );
        direct_compute_pass(
            encoder,
            "mechanic LBVH leaves",
            &lbvh.prepare_leaves_pipeline,
            &lbvh.prepare_leaves_bind_group,
            self.collider_count.div_ceil(256),
            None,
        );
        direct_compute_pass(
            encoder,
            "mechanic LBVH bounds",
            &lbvh.build_bounds_pipeline,
            &lbvh.build_bounds_bind_group,
            self.collider_count.div_ceil(256),
            None,
        );
        direct_compute_pass(
            encoder,
            "mechanic LBVH traversal",
            &lbvh.traverse_pipeline,
            &lbvh.traverse_bind_group,
            self.collider_count.div_ceil(256),
            None,
        );
        direct_compute_pass(
            encoder,
            "mechanic finalize pairs",
            &lbvh.finalize_pairs_pipeline,
            &lbvh.finalize_pairs_bind_group,
            1,
            timestamp_writes(self.timestamps.as_ref(), None, Some(5)),
        );
        indirect_compute_pass(
            encoder,
            "mechanic OBB SAT",
            &collision.narrowphase_pipeline,
            &collision.narrowphase_bind_group,
            &collision.indirect_args,
            0,
            timestamp_writes(self.timestamps.as_ref(), Some(6), None),
        );
        if self.pipeline_config.ground_plane_enabled {
            direct_compute_pass(
                encoder,
                "mechanic ground contacts",
                &collision.ground_contacts_pipeline,
                &collision.ground_contacts_bind_group,
                self.collider_count.div_ceil(256),
                None,
            );
        }
        if let Some(bindings) = &collision.terrain_bind_group
            && collision.terrain_triangle_count > 0
        {
            direct_compute_pass(
                encoder,
                "mechanic terrain BVH contacts",
                &collision.terrain_pipeline,
                bindings,
                self.collider_count.div_ceil(256),
                timestamp_writes(self.timestamps.as_ref(), Some(14), Some(15)),
            );
        }
        direct_compute_pass(
            encoder,
            "mechanic finalize contacts",
            &collision.finalize_contacts_pipeline,
            &collision.finalize_contacts_bind_group,
            1,
            timestamp_writes(self.timestamps.as_ref(), None, Some(7)),
        );
        indirect_compute_pass(
            encoder,
            "mechanic prepare persistent contacts",
            &collision.select_active_pipeline,
            &collision.select_active_bind_group,
            &collision.indirect_args,
            12,
            timestamp_writes(self.timestamps.as_ref(), Some(8), None),
        );
        direct_compute_pass(
            encoder,
            "mechanic finalize active contacts",
            &collision.finalize_active_pipeline,
            &collision.finalize_active_bind_group,
            1,
            None,
        );
        if self.mechanism.active
            && !self.mechanism.has_dynamic_root
            && uses_fused_velocity_schedule(self.bearing_count, self.body_count)
        {
            self.encode_grounded_contact_solver_pass(encoder);
            return;
        }
        if self.solver_route() == GpuSolverRoute::General || collision.terrain_triangle_count > 0 {
            indirect_compute_pass(
                encoder,
                "mechanic count body contacts",
                &collision.count_body_contacts_pipeline,
                &collision.count_body_contacts_bind_group,
                &collision.indirect_args,
                24,
                None,
            );
        }
        indirect_compute_pass(
            encoder,
            "mechanic contact warm start",
            &collision.warm_start_pipeline,
            &collision.warm_start_bind_group,
            &collision.indirect_args,
            24,
            None,
        );
        indirect_compute_pass(
            encoder,
            "mechanic contact warm start apply",
            &collision.solve_apply_pipeline,
            &collision.solve_apply_bind_group,
            &collision.indirect_args,
            36,
            None,
        );
        let serial_mechanism = self.mechanism.active
            && self.mechanism.has_dynamic_root
            && uses_fused_contact_schedule(
                self.bearing_count,
                self.pipeline_config.ground_plane_enabled,
            );
        if serial_mechanism {
            direct_compute_pass(
                encoder,
                "mechanic fused small mechanism contacts",
                &collision.solve_small_mechanism_pipeline,
                &collision.solve_small_mechanism_bind_group,
                1,
                None,
            );
        } else if self.mechanism.active && self.mechanism.has_dynamic_root {
            self.encode_contact_bearing_velocity_projection_iteration(encoder, serial_mechanism);
        }
        let iterations = self.pipeline_config.solver_iterations.max(1);
        if !serial_mechanism {
            for _ in 1..iterations {
                indirect_compute_pass(
                    encoder,
                    "mechanic contact projection",
                    &collision.solve_accumulate_pipeline,
                    &collision.solve_accumulate_bind_group,
                    &collision.indirect_args,
                    24,
                    None,
                );
                indirect_compute_pass(
                    encoder,
                    "mechanic contact apply",
                    &collision.solve_apply_pipeline,
                    &collision.solve_apply_bind_group,
                    &collision.indirect_args,
                    36,
                    None,
                );
                if self.mechanism.active && self.mechanism.has_dynamic_root {
                    self.encode_contact_bearing_velocity_projection_iteration(encoder, false);
                }
            }
        }
        indirect_compute_pass(
            encoder,
            "mechanic persist contact manifolds",
            &collision.persist_contacts_pipeline,
            &collision.persist_contacts_bind_group,
            &collision.indirect_args,
            24,
            if self.mechanism.active {
                None
            } else {
                timestamp_writes(self.timestamps.as_ref(), None, Some(9))
            },
        );
    }

    /// Records the small grounded mechanism's contact reconciliation in one
    /// Metal compute pass. Dispatch order and iteration count match the general
    /// schedule; eliminating empty pass boundaries matters when no contacts exist.
    fn encode_grounded_contact_solver_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let collision = &self.collision;
        let mechanism = &self.mechanism;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mechanic grounded contact solver"),
            timestamp_writes: timestamp_writes(self.timestamps.as_ref(), None, Some(9)),
        });
        indirect_dispatch_in_pass(
            &mut pass,
            &collision.count_body_contacts_pipeline,
            &collision.count_body_contacts_bind_group,
            &collision.indirect_args,
            24,
        );
        indirect_dispatch_in_pass(
            &mut pass,
            &collision.warm_start_pipeline,
            &collision.warm_start_bind_group,
            &collision.indirect_args,
            24,
        );
        indirect_dispatch_in_pass(
            &mut pass,
            &collision.solve_apply_pipeline,
            &collision.solve_apply_bind_group,
            &collision.indirect_args,
            36,
        );
        for _ in 1..self.pipeline_config.solver_iterations.max(1) {
            indirect_dispatch_in_pass(
                &mut pass,
                &collision.solve_accumulate_pipeline,
                &collision.solve_accumulate_bind_group,
                &collision.indirect_args,
                24,
            );
            indirect_dispatch_in_pass(
                &mut pass,
                &collision.solve_apply_pipeline,
                &collision.solve_apply_bind_group,
                &collision.indirect_args,
                36,
            );
        }
        indirect_dispatch_in_pass(
            &mut pass,
            &collision.persist_contacts_pipeline,
            &collision.persist_contacts_bind_group,
            &collision.indirect_args,
            24,
        );
        indirect_dispatch_in_pass(
            &mut pass,
            &mechanism.project_small_velocity_pipeline,
            &mechanism.project_small_velocity_bind_group,
            &collision.indirect_args,
            48,
        );
        indirect_dispatch_in_pass(
            &mut pass,
            &mechanism.capture_coordinates_pipeline,
            &mechanism.capture_coordinates_bind_group,
            &collision.indirect_args,
            36,
        );
        indirect_dispatch_in_pass(
            &mut pass,
            &mechanism.reconstruct_velocities_pipeline,
            &mechanism.reconstruct_velocities_bind_group,
            &collision.indirect_args,
            48,
        );
        if let Some(timestamps) = &self.timestamps {
            pass.set_pipeline(&timestamps.boundary);
            pass.set_bind_group(0, &timestamps.boundary_bindings, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    /// Polls asynchronous telemetry and prototype-render staging without
    /// waiting for queue completion.
    ///
    /// Completed ticks are returned in submission order. `None` means the GPU
    /// has not completed another staged tick yet.
    ///
    /// # Errors
    ///
    /// Returns [`GpuReadbackError`] if device polling or a completed mapping
    /// failed. The caller should latch that as a terminal simulation failure.
    pub fn poll_tick_readback(
        &self,
        device: &wgpu::Device,
    ) -> Result<Option<GpuCompletedTickReadback>, GpuReadbackError> {
        let poll_started = self
            .readback_timing_enabled
            .load(Ordering::Acquire)
            .then(Instant::now);
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| GpuReadbackError::DevicePoll(error.to_string()))?;
        let mut slots = self
            .async_readbacks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        for slot in &mut *slots {
            let Some(pending) = &mut slot.pending else {
                continue;
            };
            while pending.remaining_callbacks > 0 {
                match pending.receiver.try_recv() {
                    Ok((Ok(()), observed_at)) => {
                        pending.remaining_callbacks -= 1;
                        pending.callbacks_completed_at =
                            pending.callbacks_completed_at.max(observed_at);
                    }
                    Ok((Err(error), _)) => {
                        unmap_async_slot(slot);
                        slot.pending = None;
                        return Err(GpuReadbackError::BufferMap(error));
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        unmap_async_slot(slot);
                        slot.pending = None;
                        return Err(GpuReadbackError::CallbackLost);
                    }
                }
            }
        }

        let Some(slot_index) =
            oldest_completed_readback(slots.iter().enumerate().filter_map(|(index, slot)| {
                let pending = slot.pending.as_ref()?;
                Some((
                    index,
                    pending.submission_sequence,
                    pending.remaining_callbacks,
                ))
            }))
        else {
            return Ok(None);
        };
        let slot = &mut slots[slot_index];
        let Some(pending) = slot.pending.take() else {
            return Ok(None);
        };
        let diagnostics = {
            let bytes = slot
                .diagnostics
                .get_mapped_range(0..u64::try_from(size_of::<GpuDiagnostics>()).unwrap_or(32));
            bytemuck::pod_read_unaligned::<GpuDiagnostics>(&bytes)
        };
        let timestamp_values = slot.timestamps.as_ref().map(|buffer| {
            bytemuck::pod_read_unaligned::<[u64; 28]>(&buffer.get_mapped_range(0..224))
        });
        let positions = mapped_rows::<[f32; 4]>(&slot.positions, self.body_count);
        let rotations = mapped_rows::<[f32; 4]>(&slot.rotations, self.body_count);
        let linear = mapped_rows::<[f32; 4]>(&slot.linear_velocities, self.body_count);
        let angular = mapped_rows::<[f32; 4]>(&slot.angular_velocities, self.body_count);
        let coordinates = mapped_rows::<GpuMechanismCoordinate>(
            &slot.coordinates,
            self.mechanism.coordinate_count,
        );
        unmap_async_slot(slot);

        Ok(Some(GpuCompletedTickReadback {
            submission_sequence: pending.submission_sequence,
            tick_index: pending.tick_index,
            snapshot_slot: pending.snapshot_slot,
            submission_to_readback_ms: pending.submitted_at.elapsed().as_secs_f64() * 1_000.0,
            callbacks_completed_at: pending.callbacks_completed_at,
            submission_to_callbacks_ms: pending
                .callbacks_completed_at
                .map(|at| at.duration_since(pending.submitted_at).as_secs_f64() * 1_000.0),
            callbacks_during_poll: pending
                .callbacks_completed_at
                .zip(poll_started)
                .map(|(at, start)| at >= start),
            diagnostics: self.decode_tick_readback(diagnostics, timestamp_values),
            velocities: linear
                .into_iter()
                .zip(angular)
                .map(|(linear, angular)| GpuVelocity { linear, angular })
                .collect(),
            coordinates,
            transforms: positions
                .into_iter()
                .zip(rotations)
                .map(|(position, rotation)| GpuTransform { position, rotation })
                .collect(),
        }))
    }

    fn decode_tick_readback(
        &self,
        diagnostics: GpuDiagnostics,
        timestamp_values: Option<[u64; 28]>,
    ) -> GpuTickReadback {
        let timestamp_readback =
            timestamp_values
                .zip(self.timestamps.as_ref())
                .map(|(values, timestamps)| {
                    let elapsed = |start, end| {
                        timestamp_milliseconds(
                            values[start],
                            values[end],
                            timestamps.period_nanoseconds,
                        )
                    };
                    let timings = GpuKernelTimings {
                        integration_ms: elapsed(0, 1),
                        terrain_traversal_ms: if self.pipeline_config.collisions_enabled
                            && self.collision.terrain_triangle_count > 0
                        {
                            elapsed(14, 15)
                        } else {
                            0.0
                        },
                        rotational_sweep_ms: if self.pipeline_config.collisions_enabled
                            && self.collision.terrain_triangle_count > 0
                        {
                            elapsed(16, 17)
                        } else {
                            0.0
                        },
                        terrain_recovery_ms: if self.pipeline_config.collisions_enabled
                            && self.collision.terrain_triangle_count > 0
                        {
                            elapsed(18, 19)
                        } else {
                            0.0
                        },
                        recovery_projection_ms: if self.pipeline_config.collisions_enabled
                            && self.collision.terrain_triangle_count > 0
                            && self.mechanism.active
                        {
                            elapsed(20, 21) + elapsed(22, 23) + elapsed(24, 25)
                        } else {
                            0.0
                        },
                        mechanism_ms: if self.mechanism.active {
                            elapsed(2, 3)
                        } else {
                            0.0
                        },
                        broadphase_ms: if self.pipeline_config.collisions_enabled {
                            elapsed(4, 5)
                        } else {
                            0.0
                        },
                        narrowphase_ms: if self.pipeline_config.collisions_enabled {
                            elapsed(6, 7)
                        } else {
                            0.0
                        },
                        contact_solver_ms: if self.pipeline_config.collisions_enabled {
                            elapsed(8, 9)
                        } else {
                            0.0
                        },
                        bearings_ms: if self.bearing_count > 0 {
                            elapsed(10, 11)
                        } else {
                            0.0
                        },
                        snapshot_ms: elapsed(12, 13),
                    };
                    let total = timings.rotational_sweep_ms
                        + timings.terrain_recovery_ms
                        + timings.integration_ms
                        + timings.mechanism_ms
                        + timings.broadphase_ms
                        + timings.narrowphase_ms
                        + timings.contact_solver_ms
                        + timings.bearings_ms
                        + timings.snapshot_ms;
                    (total, timings)
                });
        GpuTickReadback {
            execution: GpuExecutionEvidence {
                stage_mask: diagnostics.executed_stage_mask,
                integrated_bodies: diagnostics.integrated_bodies,
                published_bodies: diagnostics.published_bodies,
                validated_bearings: diagnostics.validated_bearings,
            },
            gpu_tick_ms: timestamp_readback.map(|(total, _)| total),
            kernel_timings: timestamp_readback.map(|(_, timings)| timings),
            error_flags: diagnostics.error_flags,
            pair_count: diagnostics.pair_count,
            contact_count: diagnostics.contact_count,
            active_contact_count: diagnostics.active_contact_count,
            planned_solver_sweeps: diagnostics.planned_solver_sweeps,
            executed_solver_sweeps: diagnostics.executed_solver_sweeps,
            anchor_residual_meters: diagnostic_units(diagnostics.max_anchor_micrometers),
            axis_residual_degrees: diagnostic_units(diagnostics.max_axis_microdegrees),
        }
    }

    /// Reads only stage timestamps and fixed-size diagnostics after a completed
    /// submission. Authoritative body state remains GPU-resident.
    ///
    /// # Errors
    ///
    /// Returns [`GpuReadbackError`] if device polling or either fixed-size map fails.
    pub fn read_last_tick(
        &self,
        device: &wgpu::Device,
    ) -> Result<GpuTickReadback, GpuReadbackError> {
        map_for_read(device, &self.diagnostics_readback)?;
        let diagnostics = {
            let bytes = self
                .diagnostics_readback
                .get_mapped_range(0..u64::try_from(size_of::<GpuDiagnostics>()).unwrap_or(32));
            bytemuck::pod_read_unaligned::<GpuDiagnostics>(&bytes)
        };
        self.diagnostics_readback.unmap();

        let timestamp_values = self
            .timestamps
            .as_ref()
            .map(|timestamps| {
                map_for_read(device, &timestamps.readback)?;
                let values = bytemuck::pod_read_unaligned::<[u64; 28]>(
                    &timestamps.readback.get_mapped_range(0..224),
                );
                timestamps.readback.unmap();
                Ok::<_, GpuReadbackError>(values)
            })
            .transpose()?;
        Ok(self.decode_tick_readback(diagnostics, timestamp_values))
    }

    /// Copies one published snapshot to CPU memory for prototype renderers.
    ///
    /// Production rendering should consume [`SnapshotBuffers`] directly and avoid
    /// this synchronous readback.
    ///
    /// # Errors
    ///
    /// Returns [`GpuReadbackError`] when `slot` is invalid or GPU mapping fails.
    pub fn read_snapshot_transforms(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: u8,
    ) -> Result<Vec<GpuTransform>, GpuReadbackError> {
        let snapshot = self
            .snapshot(slot)
            .ok_or(GpuReadbackError::InvalidSnapshotSlot(slot))?;
        let byte_len = u64::from(self.body_count) * 16;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mechanic snapshot readback"),
        });
        encoder.copy_buffer_to_buffer(
            snapshot.positions(),
            0,
            &self.snapshot_positions_readback,
            0,
            byte_len,
        );
        encoder.copy_buffer_to_buffer(
            snapshot.rotations(),
            0,
            &self.snapshot_rotations_readback,
            0,
            byte_len,
        );
        queue.submit([encoder.finish()]);

        let positions = read_vec4_buffer(device, &self.snapshot_positions_readback, byte_len)?;
        let rotations = read_vec4_buffer(device, &self.snapshot_rotations_readback, byte_len)?;
        Ok(positions
            .into_iter()
            .zip(rotations)
            .map(|(position, rotation)| GpuTransform { position, rotation })
            .collect())
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

    /// Replaces the drive parameters of every mechanism coordinate.
    ///
    /// This is the one write permitted while the simulation is running: it
    /// changes no topology, mass, or buffer size, so compiled row indices stay
    /// valid and a control block can be retuned without recompiling.
    ///
    /// # Errors
    ///
    /// Returns [`GpuPhysicsError::DriveStateCount`] unless one row is supplied
    /// per compiled tree bearing.
    pub fn write_mechanism_drives(
        &self,
        queue: &wgpu::Queue,
        drives: &[GpuMechanismDrive],
    ) -> Result<(), GpuPhysicsError> {
        let required = usize::try_from(self.mechanism.coordinate_count).unwrap_or(usize::MAX);
        if drives.len() != required {
            return Err(GpuPhysicsError::DriveStateCount {
                provided: drives.len(),
                required,
            });
        }
        if !drives.is_empty() {
            queue.write_buffer(&self.mechanism.drives, 0, cast_slice(drives));
            for (coordinate, drive) in drives.iter().enumerate() {
                let row = self.mechanism.drive_constraint_rows[coordinate];
                let offset = u64::from(row)
                    * u64::try_from(size_of::<GpuDriveConstraint>()).unwrap_or(u64::MAX)
                    + u64::try_from(size_of::<GpuBearing>()).unwrap_or(u64::MAX);
                queue.write_buffer(&self.mechanism.drive_constraints, offset, bytes_of(drive));
            }
        }
        Ok(())
    }

    /// Replaces the permitted bearing-coordinate state at a paused/load boundary.
    ///
    /// # Errors
    ///
    /// Returns [`GpuPhysicsError::CoordinateStateCount`] unless one row is
    /// supplied for every tree bearing in the compiled mechanism forest.
    pub fn initialize_mechanism_coordinates(
        &self,
        queue: &wgpu::Queue,
        coordinates: &[GpuMechanismCoordinate],
    ) -> Result<(), GpuPhysicsError> {
        let required = usize::try_from(self.mechanism.coordinate_count).unwrap_or(usize::MAX);
        if coordinates.len() != required {
            return Err(GpuPhysicsError::CoordinateStateCount {
                provided: coordinates.len(),
                required,
            });
        }
        if !coordinates.is_empty() {
            queue.write_buffer(&self.mechanism.coordinates, 0, cast_slice(coordinates));
        }
        Ok(())
    }
}

#[expect(
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
fn create_mechanism_resources(
    device: &wgpu::Device,
    pipelines: &GpuPhysicsPipelines,
    creation: &CompiledCreation,
    config: &wgpu::Buffer,
    positions: &wgpu::Buffer,
    rotations: &wgpu::Buffer,
    diagnostics: &wgpu::Buffer,
    bearings: &wgpu::Buffer,
    masses: &wgpu::Buffer,
    _spatial_inertias: &wgpu::Buffer,
    linear_velocities: &wgpu::Buffer,
    angular_velocities: &wgpu::Buffer,
) -> MechanismResources {
    let body_count = creation.compounds.len();
    let has_dynamic_root =
        creation
            .loop_topology
            .body_parents
            .iter()
            .enumerate()
            .any(|(body, topology)| {
                topology.is_root && creation.compounds[body].mass_properties.inverse_mass > 0.0
            });
    let bearing_rows = creation
        .bearings
        .iter()
        .enumerate()
        .map(|(row, bearing)| (bearing.source_bearing, row))
        .collect::<BTreeMap<_, _>>();
    let coordinate_by_bearing = creation
        .loop_topology
        .tree_bearings
        .iter()
        .enumerate()
        .map(|(coordinate, &bearing)| (bearing, coordinate))
        .collect::<BTreeMap<_, _>>();
    let powered_components = creation
        .loop_topology
        .body_parents
        .iter()
        .filter_map(|body| {
            let coordinate = *coordinate_by_bearing.get(&body.tree_bearing?)?;
            (creation.coordinate_drives.get(coordinate)?.mode != mechanic_core::DriveMode::Passive)
                .then_some(body.component_index)
        })
        .collect::<BTreeSet<_>>();
    let root_flags = creation
        .loop_topology
        .body_parents
        .iter()
        .map(|body| {
            let powered_coordinate = body
                .tree_bearing
                .and_then(|bearing| coordinate_by_bearing.get(&bearing))
                .and_then(|&coordinate| creation.coordinate_drives.get(coordinate))
                .is_some_and(|drive| drive.mode != mechanic_core::DriveMode::Passive);
            u32::from(body.is_root)
                | (u32::from(
                    powered_coordinate
                        || (body.is_root && powered_components.contains(&body.component_index)),
                ) << 1)
        })
        .collect::<Vec<_>>();
    let bodies = creation
        .compounds
        .iter()
        .enumerate()
        .map(|(body, compound)| {
            let topology = creation.loop_topology.body_parents[body];
            if topology.is_root {
                return GpuMechanismBody {
                    metadata: [u32::try_from(body).unwrap_or(u32::MAX), u32::MAX, 0, 1],
                    traversal: [
                        topology.component_index,
                        topology.depth,
                        topology.preorder_index,
                        topology.postorder_index,
                    ],
                    bind_relative_position: [0.0; 4],
                    bind_relative_rotation: [0.0, 0.0, 0.0, 1.0],
                };
            }
            let parent = topology.parent_body as usize;
            let bearing = topology.tree_bearing.expect("non-root has a tree bearing");
            let bearing_index = bearing_rows[&bearing];
            let parent_compound = &creation.compounds[parent];
            let child_compound = compound;
            let inverse_parent = parent_compound.root_rotation.inverse();
            let relative_position = inverse_parent
                * (child_compound.root_translation - parent_compound.root_translation);
            let relative_rotation = (inverse_parent * child_compound.root_rotation).normalize();
            GpuMechanismBody {
                metadata: [
                    u32::try_from(parent).unwrap_or(u32::MAX),
                    u32::try_from(bearing_index).unwrap_or(u32::MAX),
                    topology.bearing_direction,
                    0,
                ],
                traversal: [
                    topology.component_index,
                    topology.depth,
                    topology.preorder_index,
                    topology.postorder_index,
                ],
                bind_relative_position: vec4(relative_position, 0.0),
                bind_relative_rotation: [
                    relative_rotation.x,
                    relative_rotation.y,
                    relative_rotation.z,
                    relative_rotation.w,
                ],
            }
        })
        .collect::<Vec<_>>();
    let maximum_depth = creation
        .loop_topology
        .body_parents
        .iter()
        .map(|body| body.depth)
        .max()
        .unwrap_or(0);
    let mut preorder = (0..body_count).collect::<Vec<_>>();
    preorder.sort_unstable_by_key(|&body| creation.loop_topology.body_parents[body].preorder_index);
    let preorder = preorder
        .into_iter()
        .map(|body| u32::try_from(body).unwrap_or(u32::MAX))
        .collect::<Vec<_>>();
    let contraction_schedule = creation
        .loop_topology
        .contraction_rounds
        .iter()
        .enumerate()
        .flat_map(|(round, bodies)| {
            bodies.iter().map(move |&body| {
                let topology = creation.loop_topology.body_parents[body as usize];
                GpuContractionNode {
                    metadata: [
                        body,
                        topology.parent_body,
                        u32::try_from(round).unwrap_or(u32::MAX),
                        topology.component_index,
                    ],
                }
            })
        })
        .collect::<Vec<_>>();

    let coordinates =
        vec![GpuMechanismCoordinate::zeroed(); creation.loop_topology.tree_bearings.len()];
    let empty_links = vec![GpuLinkState::zeroed(); body_count];
    let root_flags =
        create_readonly_storage_buffer(device, "mechanic mechanism root flags", &root_flags);
    let body_rows = bodies;
    let bodies = create_storage_buffer(device, "mechanic mechanism bodies", &body_rows);
    let coordinate_count = u32::try_from(coordinates.len()).unwrap_or(u32::MAX);
    let closure_count =
        u32::try_from(creation.loop_topology.closure_bearings.len()).unwrap_or(u32::MAX);
    let coordinates = create_state_buffer(device, "mechanic mechanism coordinates", &coordinates);
    let drive_rows = if creation.coordinate_drives.len() == coordinate_count as usize {
        creation
            .coordinate_drives
            .iter()
            .copied()
            .map(GpuMechanismDrive::from)
            .collect::<Vec<_>>()
    } else {
        vec![GpuMechanismDrive::PASSIVE; coordinate_count as usize]
    };
    let drives = create_storage_buffer(device, "mechanic mechanism drives", &drive_rows);
    let child_by_bearing = creation
        .loop_topology
        .body_parents
        .iter()
        .enumerate()
        .filter_map(|(body, topology)| {
            topology.tree_bearing.map(|bearing| {
                (
                    bearing,
                    (
                        u32::try_from(body).unwrap_or(u32::MAX),
                        topology.parent_body,
                        topology.bearing_direction,
                    ),
                )
            })
        })
        .collect::<BTreeMap<_, _>>();
    let mut drive_constraint_rows = vec![u32::MAX; coordinate_count as usize];
    let drive_constraint_rows_gpu = creation
        .bearings
        .iter()
        .enumerate()
        .map(|(row, bearing)| {
            let coordinate = bearing.coordinate_index.unwrap_or(u32::MAX);
            if coordinate != u32::MAX {
                drive_constraint_rows[coordinate as usize] = u32::try_from(row).unwrap_or(u32::MAX);
            }
            let (child, parent, direction) = child_by_bearing
                .get(&bearing.source_bearing)
                .copied()
                .unwrap_or((u32::MAX, u32::MAX, 0));
            let drive = usize::try_from(coordinate)
                .ok()
                .and_then(|coordinate| drive_rows.get(coordinate))
                .copied()
                .unwrap_or(GpuMechanismDrive::PASSIVE);
            let axis_inertia = usize::try_from(coordinate)
                .ok()
                .and_then(|coordinate| {
                    creation
                        .loop_topology
                        .coordinate_axis_inertia
                        .get(coordinate)
                })
                .copied()
                .unwrap_or(f32::INFINITY);
            GpuDriveConstraint {
                bearing: GpuBearing {
                    local_anchor_a: vec4(bearing.local_anchor_a, bearing.kind.bounds()[0]),
                    local_anchor_b: vec4(bearing.local_anchor_b, bearing.kind.bounds()[1]),
                    local_axis_a: vec4(
                        bearing.local_axis_a,
                        if bearing.kind.is_translational() {
                            1.0
                        } else {
                            0.0
                        },
                    ),
                    local_axis_b: vec4(bearing.local_axis_b, 0.0),
                    suspension: match bearing.kind {
                        mechanic_core::BearingKind::Suspension(s) => s.passive_rows()[0],
                        _ => [0.0; 4],
                    },
                    bump_stop: match bearing.kind {
                        mechanic_core::BearingKind::Suspension(s) => s.passive_rows()[1],
                        _ => [0.0; 4],
                    },
                    metadata: [
                        bearing.compound_a,
                        bearing.compound_b,
                        coordinate,
                        u32::from(bearing.coordinate_index.is_none()),
                    ],
                },
                drive,
                state: [axis_inertia, 0.0, 0.0, 0.0],
                metadata: [child, parent, direction, coordinate],
            }
        })
        .collect::<Vec<_>>();
    let drive_constraints = create_storage_buffer(
        device,
        "mechanic drive constraints",
        &drive_constraint_rows_gpu,
    );
    let preorder = create_readonly_storage_buffer(device, "mechanic mechanism preorder", &preorder);
    let contraction_schedule = create_readonly_storage_buffer(
        device,
        "mechanic articulated contraction schedule",
        &contraction_schedule,
    );
    let velocity_deltas = create_sized_buffer(
        device,
        "mechanic bearing velocity deltas",
        body_count.max(1) * 6 * size_of::<i32>(),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let articulated_inertia_rows = creation
        .compounds
        .iter()
        .map(|compound| GpuSpatialInertia {
            mass: [compound.mass_properties.mass, 0.0, 0.0, 0.0],
            inertia_x: vec4(compound.mass_properties.inertia.x_axis, 0.0),
            inertia_y: vec4(compound.mass_properties.inertia.y_axis, 0.0),
            inertia_z: vec4(compound.mass_properties.inertia.z_axis, 0.0),
        })
        .collect::<Vec<_>>();
    let articulated_inertia = create_storage_buffer(
        device,
        "mechanic articulated inertia",
        &articulated_inertia_rows,
    );
    let bias_force = create_sized_buffer(
        device,
        "mechanic articulated bias force",
        body_count.max(1) * 32,
        wgpu::BufferUsages::STORAGE,
    );
    let generalized_force = create_sized_buffer(
        device,
        "mechanic generalized force",
        body_count.max(1) * 32,
        wgpu::BufferUsages::STORAGE,
    );
    let constraint_impulse = create_sized_buffer(
        device,
        "mechanic generalized constraint impulse",
        body_count.max(1) * 32,
        wgpu::BufferUsages::STORAGE,
    );
    let reduction_scratch = create_sized_buffer(
        device,
        "mechanic contraction scratch",
        body_count.max(1) * 64,
        wgpu::BufferUsages::STORAGE,
    );
    let links_a = create_storage_buffer(device, "mechanic mechanism links A", &empty_links);
    let links_b = create_storage_buffer(device, "mechanic mechanism links B", &empty_links);
    let closure_accumulators = create_sized_buffer(
        device,
        "mechanic closure accumulators",
        usize::try_from(coordinate_count)
            .unwrap_or(usize::MAX)
            .max(1)
            * 8,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let closure_state = create_sized_buffer(
        device,
        "mechanic closure state",
        16,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let closure_indirect_args = create_sized_buffer(
        device,
        "mechanic closure indirect dispatch",
        24,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
    );

    let shader = shader_module(
        pipelines,
        device,
        "mechanic mechanism kernels",
        include_str!("kernels/mechanism.wgsl"),
    );
    let prepare_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic prepare mechanism links",
        &shader,
        "prepare_links",
    );
    let prepare_bind_group = bind_group(
        device,
        "mechanic prepare mechanism bindings",
        &prepare_pipeline,
        &[
            entry(0, config),
            entry(3, &bodies),
            entry(4, bearings),
            entry(5, &coordinates),
            entry(6, &links_a),
        ],
    );
    let jump_a_to_b_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic mechanism jump A to B",
        &shader,
        "jump_a_to_b",
    );
    let jump_a_to_b_bind_group = bind_group(
        device,
        "mechanic mechanism jump A to B bindings",
        &jump_a_to_b_pipeline,
        &[entry(0, config), entry(6, &links_a), entry(7, &links_b)],
    );
    let jump_b_to_a_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic mechanism jump B to A",
        &shader,
        "jump_b_to_a",
    );
    let jump_b_to_a_bind_group = bind_group(
        device,
        "mechanic mechanism jump B to A bindings",
        &jump_b_to_a_pipeline,
        &[entry(0, config), entry(6, &links_a), entry(7, &links_b)],
    );
    let publish_a_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic publish mechanism A",
        &shader,
        "publish_a",
    );
    let publish_a_bind_group = bind_group(
        device,
        "mechanic publish mechanism A bindings",
        &publish_a_pipeline,
        &[
            entry(0, config),
            entry(1, positions),
            entry(2, rotations),
            entry(6, &links_a),
        ],
    );
    let publish_b_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic publish mechanism B",
        &shader,
        "publish_b",
    );
    let publish_b_bind_group = bind_group(
        device,
        "mechanic publish mechanism B bindings",
        &publish_b_pipeline,
        &[
            entry(0, config),
            entry(1, positions),
            entry(2, rotations),
            entry(7, &links_b),
        ],
    );

    let articulated_shader = shader_module(
        pipelines,
        device,
        "mechanic articulated dynamics kernels",
        include_str!("kernels/articulated.wgsl"),
    );
    let prepare_drives_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic prepare drive constraints",
        &articulated_shader,
        "prepare_drive_constraints",
    );
    let prepare_drives_bind_group = bind_group(
        device,
        "mechanic prepare drive constraint bindings",
        &prepare_drives_pipeline,
        &[
            entry(0, config),
            entry(8, &coordinates),
            entry(13, &drive_constraints),
        ],
    );
    let project_velocity_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic project bearing velocities",
        &articulated_shader,
        "project_bearing_velocities",
    );
    let project_velocity_bind_group = bind_group(
        device,
        "mechanic bearing velocity projection bindings",
        &project_velocity_pipeline,
        &[
            entry(0, config),
            entry(1, positions),
            entry(2, rotations),
            entry(3, linear_velocities),
            entry(4, angular_velocities),
            entry(5, masses),
            entry(9, &velocity_deltas),
            entry(13, &drive_constraints),
        ],
    );
    let project_small_velocity_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic fused small bearing velocity projection",
        &articulated_shader,
        "project_small_mechanism_velocities",
    );
    let project_small_velocity_bind_group = bind_group(
        device,
        "mechanic fused small bearing velocity bindings",
        &project_small_velocity_pipeline,
        &[
            entry(0, config),
            entry(1, positions),
            entry(2, rotations),
            entry(3, linear_velocities),
            entry(4, angular_velocities),
            entry(5, masses),
            entry(9, &velocity_deltas),
            entry(13, &drive_constraints),
        ],
    );
    let project_velocity_serial_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic serial bearing velocity projection",
        &articulated_shader,
        "project_bearing_velocities_serial",
    );
    let project_velocity_serial_bind_group = bind_group(
        device,
        "mechanic serial bearing velocity bindings",
        &project_velocity_serial_pipeline,
        &[
            entry(0, config),
            entry(1, positions),
            entry(2, rotations),
            entry(3, linear_velocities),
            entry(4, angular_velocities),
            entry(5, masses),
            entry(9, &velocity_deltas),
            entry(13, &drive_constraints),
        ],
    );
    let apply_velocity_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic apply bearing velocity deltas",
        &articulated_shader,
        "apply_velocity_deltas",
    );
    let apply_velocity_bind_group = bind_group(
        device,
        "mechanic apply bearing velocity bindings",
        &apply_velocity_pipeline,
        &[
            entry(0, config),
            entry(3, linear_velocities),
            entry(4, angular_velocities),
            entry(9, &velocity_deltas),
        ],
    );
    let advance_coordinates_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic advance bearing coordinates",
        &articulated_shader,
        "advance_coordinates",
    );
    let advance_coordinates_bind_group = bind_group(
        device,
        "mechanic advance bearing coordinate bindings",
        &advance_coordinates_pipeline,
        &[
            entry(0, config),
            entry(1, positions),
            entry(2, rotations),
            entry(3, linear_velocities),
            entry(4, angular_velocities),
            entry(6, bearings),
            entry(7, &bodies),
            entry(8, &coordinates),
            entry(12, &drives),
        ],
    );
    let capture_coordinates_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic capture bearing velocities",
        &articulated_shader,
        "capture_coordinates",
    );
    let capture_coordinates_bind_group = bind_group(
        device,
        "mechanic capture bearing velocity bindings",
        &capture_coordinates_pipeline,
        &[
            entry(0, config),
            entry(1, positions),
            entry(2, rotations),
            entry(3, linear_velocities),
            entry(4, angular_velocities),
            entry(6, bearings),
            entry(7, &bodies),
            entry(8, &coordinates),
        ],
    );
    let reconstruct_velocities_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic reconstruct mechanism velocities",
        &articulated_shader,
        "reconstruct_body_velocities",
    );
    let reconstruct_velocities_bind_group = bind_group(
        device,
        "mechanic reconstruct mechanism velocity bindings",
        &reconstruct_velocities_pipeline,
        &[
            entry(0, config),
            entry(1, positions),
            entry(2, rotations),
            entry(3, linear_velocities),
            entry(4, angular_velocities),
            entry(6, bearings),
            entry(7, &bodies),
            entry(8, &coordinates),
            entry(10, &preorder),
        ],
    );
    let validate_state_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic validate articulated state",
        &articulated_shader,
        "validate_articulated_state",
    );
    let validate_state_bind_group = bind_group(
        device,
        "mechanic articulated validation bindings",
        &validate_state_pipeline,
        &[
            entry(0, config),
            entry(1, positions),
            entry(2, rotations),
            entry(3, linear_velocities),
            entry(4, angular_velocities),
            entry(6, bearings),
            entry(7, &bodies),
            entry(8, &coordinates),
            entry(11, diagnostics),
        ],
    );

    let mut covered_depth = 1_u32;
    let mut pointer_jump_rounds = 0_u32;
    while covered_depth < maximum_depth {
        covered_depth = covered_depth.saturating_mul(2);
        pointer_jump_rounds = pointer_jump_rounds.saturating_add(1);
    }
    if maximum_depth > 0 {
        pointer_jump_rounds = pointer_jump_rounds.max(1);
    }
    let final_links = if pointer_jump_rounds.is_multiple_of(2) {
        &links_a
    } else {
        &links_b
    };
    let closure_shader = shader_module(
        pipelines,
        device,
        "mechanic closure kernels",
        include_str!("kernels/closure.wgsl"),
    );
    let evaluate_closures_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic evaluate closures",
        &closure_shader,
        "evaluate_closures",
    );
    let evaluate_closures_bind_group = bind_group(
        device,
        "mechanic evaluate closure bindings",
        &evaluate_closures_pipeline,
        &[
            entry(0, config),
            entry(1, diagnostics),
            entry(2, bearings),
            entry(3, &bodies),
            entry(5, final_links),
            entry(6, &closure_accumulators),
            entry(7, &closure_state),
        ],
    );
    let finalize_closures_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic finalize closures",
        &closure_shader,
        "finalize_closures",
    );
    let finalize_closures_bind_group = bind_group(
        device,
        "mechanic finalize closure bindings",
        &finalize_closures_pipeline,
        &[
            entry(0, config),
            entry(7, &closure_state),
            entry(8, &closure_indirect_args),
        ],
    );
    let apply_closure_step_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic solve closure Newton PCG step",
        &closure_shader,
        "solve_closure_pcg",
    );
    let apply_closure_step_bind_group = bind_group(
        device,
        "mechanic apply closure step bindings",
        &apply_closure_step_pipeline,
        &[
            entry(0, config),
            entry(4, &coordinates),
            entry(2, bearings),
            entry(3, &bodies),
            entry(5, final_links),
            entry(6, &closure_accumulators),
            entry(1, diagnostics),
            entry(9, &reduction_scratch),
        ],
    );

    MechanismResources {
        root_flags,
        bodies,
        body_rows,
        coordinates,
        drives,
        drive_constraints,
        drive_constraint_rows,
        _preorder: preorder,
        _contraction_schedule: contraction_schedule,
        velocity_deltas,
        _articulated_inertia: articulated_inertia,
        _bias_force: bias_force,
        _generalized_force: generalized_force,
        _constraint_impulse: constraint_impulse,
        _reduction_scratch: reduction_scratch,
        links_a,
        links_b,
        closure_accumulators,
        closure_state,
        closure_indirect_args,
        prepare_pipeline,
        prepare_bind_group,
        jump_a_to_b_pipeline,
        jump_a_to_b_bind_group,
        jump_b_to_a_pipeline,
        jump_b_to_a_bind_group,
        publish_a_pipeline,
        publish_a_bind_group,
        publish_b_pipeline,
        publish_b_bind_group,
        evaluate_closures_pipeline,
        evaluate_closures_bind_group,
        finalize_closures_pipeline,
        finalize_closures_bind_group,
        apply_closure_step_pipeline,
        apply_closure_step_bind_group,
        project_velocity_pipeline,
        project_velocity_bind_group,
        project_small_velocity_pipeline,
        project_small_velocity_bind_group,
        project_velocity_serial_pipeline,
        project_velocity_serial_bind_group,
        apply_velocity_pipeline,
        apply_velocity_bind_group,
        prepare_drives_pipeline,
        prepare_drives_bind_group,
        advance_coordinates_pipeline,
        advance_coordinates_bind_group,
        capture_coordinates_pipeline,
        capture_coordinates_bind_group,
        reconstruct_velocities_pipeline,
        reconstruct_velocities_bind_group,
        validate_state_pipeline,
        validate_state_bind_group,
        pointer_jump_rounds,
        coordinate_count,
        closure_count,
        final_is_a: pointer_jump_rounds.is_multiple_of(2),
        active: maximum_depth > 0,
        has_dynamic_root,
    }
}

#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
fn create_lbvh_resources(
    device: &wgpu::Device,
    pipelines: &GpuPhysicsPipelines,
    collider_count: usize,
    config: &wgpu::Buffer,
    positions: &wgpu::Buffer,
    rotations: &wgpu::Buffer,
    diagnostics: &wgpu::Buffer,
    colliders: &wgpu::Buffer,
    convex_shapes: &wgpu::Buffer,
    pairs: &wgpu::Buffer,
    suppressed_pairs: &wgpu::Buffer,
    indirect_args: &wgpu::Buffer,
) -> LbvhResources {
    let sort_count = collider_count.next_power_of_two().max(256);
    let internal_count = collider_count.saturating_sub(1).max(1);
    let node_count = collider_count.saturating_mul(2).saturating_sub(1).max(1);
    let collider_aabbs = create_sized_buffer(
        device,
        "mechanic LBVH collider AABBs",
        collider_count.max(1) * 32,
        wgpu::BufferUsages::STORAGE,
    );
    let morton_entries = create_sized_buffer(
        device,
        "mechanic LBVH Morton entries",
        sort_count * size_of::<GpuPair>(),
        wgpu::BufferUsages::STORAGE,
    );
    let node_aabbs = create_sized_buffer(
        device,
        "mechanic LBVH node AABBs",
        node_count * 32,
        wgpu::BufferUsages::STORAGE,
    );
    let node_children = create_sized_buffer(
        device,
        "mechanic LBVH children",
        internal_count * size_of::<GpuPair>(),
        wgpu::BufferUsages::STORAGE,
    );
    let node_parents = create_sized_buffer(
        device,
        "mechanic LBVH parents",
        node_count * size_of::<u32>(),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let node_visits = create_sized_buffer(
        device,
        "mechanic LBVH bound visits",
        internal_count * size_of::<u32>(),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let sort_params = create_uniform_buffer(device, "mechanic LBVH sort parameters", &[0_u32; 4]);
    let mut sort_parameter_rows = Vec::<[u32; 4]>::new();
    let mut sort_steps = Vec::<(u64, bool)>::new();
    let mut k = 512_u32;
    let sort_count_u32 = u32::try_from(sort_count).unwrap_or(u32::MAX);
    while k <= sort_count_u32 {
        let mut j = k / 2;
        while j >= 256 {
            let offset = u64::try_from(sort_parameter_rows.len() * 16).unwrap_or(u64::MAX);
            sort_parameter_rows.push([k, j, 0, 0]);
            sort_steps.push((offset, false));
            j /= 2;
        }
        let offset = u64::try_from(sort_parameter_rows.len() * 16).unwrap_or(u64::MAX);
        sort_parameter_rows.push([k, 0, 0, 0]);
        sort_steps.push((offset, true));
        k = k.saturating_mul(2);
    }
    let sort_params_upload = create_buffer(
        device,
        "mechanic LBVH sort parameter upload",
        &sort_parameter_rows,
        wgpu::BufferUsages::COPY_SRC,
    );

    let shader = shader_module(
        pipelines,
        device,
        "mechanic LBVH kernels",
        include_str!("kernels/lbvh.wgsl"),
    );
    let compute_morton_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic LBVH Morton codes",
        &shader,
        "compute_morton",
    );
    let compute_morton_bind_group = bind_group(
        device,
        "mechanic LBVH Morton bindings",
        &compute_morton_pipeline,
        &[
            entry(0, config),
            entry(1, positions),
            entry(2, rotations),
            entry(6, colliders),
            entry(17, &collider_aabbs),
            entry(18, &morton_entries),
            entry(28, convex_shapes),
        ],
    );
    let sort_local_initial_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic LBVH local sort",
        &shader,
        "sort_local_initial",
    );
    let sort_local_initial_bind_group = bind_group(
        device,
        "mechanic LBVH local sort bindings",
        &sort_local_initial_pipeline,
        &[entry(18, &morton_entries)],
    );
    let sort_global_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic LBVH global merge",
        &shader,
        "sort_global_step",
    );
    let sort_global_bind_group = bind_group(
        device,
        "mechanic LBVH global merge bindings",
        &sort_global_pipeline,
        &[
            entry(0, config),
            entry(18, &morton_entries),
            entry(23, &sort_params),
        ],
    );
    let sort_local_merge_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic LBVH local merge",
        &shader,
        "sort_local_merge",
    );
    let sort_local_merge_bind_group = bind_group(
        device,
        "mechanic LBVH local merge bindings",
        &sort_local_merge_pipeline,
        &[entry(18, &morton_entries), entry(23, &sort_params)],
    );
    let build_topology_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic LBVH topology",
        &shader,
        "build_topology",
    );
    let build_topology_bind_group = bind_group(
        device,
        "mechanic LBVH topology bindings",
        &build_topology_pipeline,
        &[
            entry(0, config),
            entry(18, &morton_entries),
            entry(20, &node_children),
            entry(21, &node_parents),
        ],
    );
    let prepare_leaves_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic LBVH prepare leaves",
        &shader,
        "prepare_leaves",
    );
    let prepare_leaves_bind_group = bind_group(
        device,
        "mechanic LBVH leaf bindings",
        &prepare_leaves_pipeline,
        &[
            entry(0, config),
            entry(17, &collider_aabbs),
            entry(18, &morton_entries),
            entry(19, &node_aabbs),
        ],
    );
    let build_bounds_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic LBVH bounds",
        &shader,
        "build_bounds",
    );
    let build_bounds_bind_group = bind_group(
        device,
        "mechanic LBVH bound bindings",
        &build_bounds_pipeline,
        &[
            entry(0, config),
            entry(5, diagnostics),
            entry(19, &node_aabbs),
            entry(20, &node_children),
            entry(21, &node_parents),
            entry(22, &node_visits),
        ],
    );
    let traverse_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic LBVH traversal",
        &shader,
        "traverse",
    );
    let traverse_bind_group = bind_group(
        device,
        "mechanic LBVH traversal bindings",
        &traverse_pipeline,
        &[
            entry(0, config),
            entry(5, diagnostics),
            entry(6, colliders),
            entry(9, pairs),
            entry(11, suppressed_pairs),
            entry(17, &collider_aabbs),
            entry(18, &morton_entries),
            entry(19, &node_aabbs),
            entry(20, &node_children),
        ],
    );
    let finalize_pairs_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic finalize LBVH pairs",
        &shader,
        "finalize_pairs",
    );
    let finalize_pairs_bind_group = bind_group(
        device,
        "mechanic finalize LBVH pair bindings",
        &finalize_pairs_pipeline,
        &[
            entry(0, config),
            entry(5, diagnostics),
            entry(12, indirect_args),
        ],
    );
    LbvhResources {
        sort_count: sort_count_u32,
        _collider_aabbs: collider_aabbs,
        _morton_entries: morton_entries,
        _node_aabbs: node_aabbs,
        _node_children: node_children,
        node_parents,
        node_visits,
        sort_params,
        sort_params_upload,
        sort_steps,
        compute_morton_pipeline,
        compute_morton_bind_group,
        sort_local_initial_pipeline,
        sort_local_initial_bind_group,
        sort_global_pipeline,
        sort_global_bind_group,
        sort_local_merge_pipeline,
        sort_local_merge_bind_group,
        build_topology_pipeline,
        build_topology_bind_group,
        prepare_leaves_pipeline,
        prepare_leaves_bind_group,
        build_bounds_pipeline,
        build_bounds_bind_group,
        traverse_pipeline,
        traverse_bind_group,
        finalize_pairs_pipeline,
        finalize_pairs_bind_group,
    }
}

#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
fn create_collision_resources(
    device: &wgpu::Device,
    pipelines: &GpuPhysicsPipelines,
    collider_count: usize,
    pair_capacity: usize,
    config: &wgpu::Buffer,
    positions: &wgpu::Buffer,
    rotations: &wgpu::Buffer,
    linear_velocities: &wgpu::Buffer,
    angular_velocities: &wgpu::Buffer,
    masses: &wgpu::Buffer,
    diagnostics: &wgpu::Buffer,
    colliders: &wgpu::Buffer,
    convex_shapes: &wgpu::Buffer,
    suppressed_pairs: &wgpu::Buffer,
    drive_constraints: &wgpu::Buffer,
    body_components: &[u32],
    mechanism_self_collisions: bool,
) -> CollisionResources {
    let body_components = create_readonly_storage_buffer(
        device,
        "mechanic body mechanism components",
        body_components,
    );
    let pairs = create_sized_buffer(
        device,
        "mechanic candidate pairs",
        pair_capacity * size_of::<GpuPair>(),
        wgpu::BufferUsages::STORAGE,
    );
    let contacts = create_sized_buffer(
        device,
        "mechanic contact manifolds",
        pair_capacity * size_of::<GpuContact>(),
        wgpu::BufferUsages::STORAGE,
    );
    let manifold_keys = create_sized_buffer(
        device,
        "mechanic persistent manifold keys",
        pair_capacity * size_of::<u32>(),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let persistent_manifolds = create_sized_buffer(
        device,
        "mechanic persistent manifolds",
        pair_capacity * size_of::<GpuPersistentManifold>(),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let concrete = ConstructionMaterial::Concrete.properties();
    let ground_surface = GpuGroundSurface {
        response: [
            concrete.static_friction,
            concrete.dynamic_friction,
            concrete.restitution,
            concrete.rolling_resistance,
        ],
        elasticity: [
            concrete.nominal_block_compliance(),
            concrete.youngs_modulus_pa,
            0.0,
            0.0,
        ],
        plane: [0.0, 1.0, 0.0, 0.0],
    };
    let ground_surfaces = create_storage_buffer(
        device,
        "mechanic collider terrain surfaces",
        &vec![ground_surface; collider_count.max(1)],
    );
    let active_contacts = create_sized_buffer(
        device,
        "mechanic active contact indices",
        pair_capacity.saturating_add(1) * size_of::<u32>(),
        wgpu::BufferUsages::STORAGE,
    );
    let indirect_args = create_sized_buffer(
        device,
        "mechanic indirect dispatch arguments",
        15 * size_of::<u32>(),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
    );
    let velocity_deltas = create_sized_buffer(
        device,
        "mechanic projected velocity deltas",
        MAX_BODIES * 6 * size_of::<i32>(),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let world_masses = create_sized_buffer(
        device,
        "mechanic world inverse inertias",
        MAX_BODIES * 64,
        wgpu::BufferUsages::STORAGE,
    );
    let lbvh = create_lbvh_resources(
        device,
        pipelines,
        collider_count,
        config,
        positions,
        rotations,
        diagnostics,
        colliders,
        convex_shapes,
        &pairs,
        suppressed_pairs,
        &indirect_args,
    );

    let shader = shader_module(
        pipelines,
        device,
        "mechanic collision kernels",
        include_str!("kernels/collision.wgsl"),
    );
    let update_world_masses_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic update world inverse inertias",
        &shader,
        "update_world_masses",
    );
    let update_world_masses_bind_group = bind_group(
        device,
        "mechanic world inverse inertia bindings",
        &update_world_masses_pipeline,
        &[
            entry(0, config),
            entry(1, positions),
            entry(2, rotations),
            entry(24, masses),
            entry(26, &world_masses),
        ],
    );
    let narrowphase_entry = if mechanism_self_collisions {
        "narrowphase"
    } else {
        "narrowphase_without_mechanism_self_collisions"
    };
    let narrowphase_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic OBB SAT",
        &shader,
        narrowphase_entry,
    );
    let mut narrowphase_bindings = vec![
        entry(0, config),
        entry(1, positions),
        entry(2, rotations),
        entry(5, diagnostics),
        entry(6, colliders),
        entry(9, &pairs),
        entry(10, &contacts),
        entry(28, convex_shapes),
    ];
    if !mechanism_self_collisions {
        narrowphase_bindings.push(entry(27, &body_components));
    }
    let narrowphase_bind_group = bind_group(
        device,
        "mechanic narrowphase bindings",
        &narrowphase_pipeline,
        &narrowphase_bindings,
    );
    let ground_contacts_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic ground contacts",
        &shader,
        "generate_ground_contacts",
    );
    let terrain_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic terrain BVH contacts",
        &shader,
        "generate_terrain_contacts",
    );
    let ground_contacts_bind_group = bind_group(
        device,
        "mechanic ground contact bindings",
        &ground_contacts_pipeline,
        &[
            entry(0, config),
            entry(1, positions),
            entry(2, rotations),
            entry(5, diagnostics),
            entry(6, colliders),
            entry(10, &contacts),
            entry(28, convex_shapes),
            entry(29, &ground_surfaces),
        ],
    );
    let finalize_contacts_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic finalize contacts",
        &shader,
        "finalize_contacts",
    );
    let finalize_contacts_bind_group = bind_group(
        device,
        "mechanic finalize contact bindings",
        &finalize_contacts_pipeline,
        &[
            entry(0, config),
            entry(5, diagnostics),
            entry(12, &indirect_args),
        ],
    );
    let select_active_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic prepare persistent contacts",
        &shader,
        "prepare_contacts",
    );
    let select_active_bind_group = bind_group(
        device,
        "mechanic persistent contact preparation bindings",
        &select_active_pipeline,
        &[
            entry(0, config),
            entry(3, linear_velocities),
            entry(5, diagnostics),
            entry(10, &contacts),
            entry(14, &manifold_keys),
            entry(15, &persistent_manifolds),
            entry(16, &active_contacts),
            entry(26, &world_masses),
            entry(25, angular_velocities),
        ],
    );
    let finalize_active_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic finalize active contacts",
        &shader,
        "finalize_active_contacts",
    );
    let finalize_active_bind_group = bind_group(
        device,
        "mechanic finalize active contact bindings",
        &finalize_active_pipeline,
        &[
            entry(0, config),
            entry(5, diagnostics),
            entry(12, &indirect_args),
            entry(16, &active_contacts),
        ],
    );
    let count_body_contacts_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic count body contacts",
        &shader,
        "count_body_contacts",
    );
    let count_body_contacts_bind_group = bind_group(
        device,
        "mechanic body contact count bindings",
        &count_body_contacts_pipeline,
        &[
            entry(0, config),
            entry(5, diagnostics),
            entry(10, &contacts),
            entry(16, &active_contacts),
            entry(26, &world_masses),
        ],
    );
    let warm_start_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic warm start contacts",
        &shader,
        "warm_start",
    );
    let warm_start_bind_group = bind_group(
        device,
        "mechanic warm start bindings",
        &warm_start_pipeline,
        &[
            entry(0, config),
            entry(3, linear_velocities),
            entry(5, diagnostics),
            entry(10, &contacts),
            entry(13, &velocity_deltas),
            entry(16, &active_contacts),
            entry(15, &persistent_manifolds),
            entry(26, &world_masses),
            entry(25, angular_velocities),
        ],
    );
    let solve_accumulate_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic accumulate contact impulses",
        &shader,
        "solve_accumulate",
    );
    let solve_accumulate_bind_group = bind_group(
        device,
        "mechanic contact impulse bindings",
        &solve_accumulate_pipeline,
        &[
            entry(0, config),
            entry(3, linear_velocities),
            entry(5, diagnostics),
            entry(10, &contacts),
            entry(13, &velocity_deltas),
            entry(16, &active_contacts),
            entry(15, &persistent_manifolds),
            entry(26, &world_masses),
            entry(25, angular_velocities),
        ],
    );
    let solve_small_mechanism_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic fused small mechanism contact projection",
        &shader,
        "solve_small_mechanism_contacts",
    );
    let solve_small_mechanism_bind_group = bind_group(
        device,
        "mechanic fused small mechanism contact bindings",
        &solve_small_mechanism_pipeline,
        &[
            entry(0, config),
            entry(2, rotations),
            entry(3, linear_velocities),
            entry(10, &contacts),
            entry(15, &persistent_manifolds),
            entry(16, &active_contacts),
            entry(25, angular_velocities),
            entry(26, &world_masses),
            entry(30, drive_constraints),
        ],
    );
    let solve_apply_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic apply contact impulses",
        &shader,
        "solve_apply",
    );
    let solve_apply_bind_group = bind_group(
        device,
        "mechanic contact apply bindings",
        &solve_apply_pipeline,
        &[
            entry(0, config),
            entry(3, linear_velocities),
            entry(13, &velocity_deltas),
            entry(25, angular_velocities),
        ],
    );
    let persist_contacts_pipeline = compute_pipeline(
        pipelines,
        device,
        "mechanic persist contacts",
        &shader,
        "persist_contacts",
    );
    let persist_contacts_bind_group = bind_group(
        device,
        "mechanic persist contact bindings",
        &persist_contacts_pipeline,
        &[
            entry(0, config),
            entry(5, diagnostics),
            entry(10, &contacts),
            entry(15, &persistent_manifolds),
            entry(16, &active_contacts),
        ],
    );
    CollisionResources {
        lbvh,
        body_components,
        _pairs: pairs,
        contacts,
        manifold_keys,
        persistent_manifolds,
        ground_surfaces,
        active_contacts,
        indirect_args,
        velocity_deltas,
        world_masses,
        update_world_masses_pipeline,
        update_world_masses_bind_group,
        narrowphase_pipeline,
        narrowphase_bind_group,
        ground_contacts_pipeline,
        ground_contacts_bind_group,
        terrain_pipeline,
        terrain_bind_group: None,
        terrain_geometry_bind_group: None,
        terrain_triangle_count: 0,
        finalize_contacts_pipeline,
        finalize_contacts_bind_group,
        select_active_pipeline,
        select_active_bind_group,
        finalize_active_pipeline,
        finalize_active_bind_group,
        count_body_contacts_pipeline,
        count_body_contacts_bind_group,
        warm_start_pipeline,
        warm_start_bind_group,
        solve_accumulate_pipeline,
        solve_accumulate_bind_group,
        solve_small_mechanism_pipeline,
        solve_small_mechanism_bind_group,
        solve_apply_pipeline,
        solve_apply_bind_group,
        persist_contacts_pipeline,
        persist_contacts_bind_group,
    }
}

fn contact_pair_capacity(collider_count: usize) -> u32 {
    const MINIMUM_PAIR_CAPACITY: usize = 256;
    let collider_pairs = collider_count.saturating_mul(collider_count.saturating_sub(1)) / 2;
    let required = collider_pairs.saturating_add(collider_count);
    let capacity = if required >= MAX_CONTACT_PAIRS {
        MAX_CONTACT_PAIRS
    } else {
        required.max(MINIMUM_PAIR_CAPACITY).next_power_of_two()
    };
    u32::try_from(capacity).unwrap_or(u32::MAX)
}

fn shader_module(
    pipelines: &GpuPhysicsPipelines,
    device: &wgpu::Device,
    label: &'static str,
    source: &'static str,
) -> wgpu::ShaderModule {
    if let Some(shader) = pipelines.shaders.lock().unwrap().get(label).cloned() {
        return shader;
    }
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    pipelines
        .shaders
        .lock()
        .unwrap()
        .insert(label, shader.clone());
    shader
}

fn compute_pipeline(
    pipelines: &GpuPhysicsPipelines,
    device: &wgpu::Device,
    label: &'static str,
    shader: &wgpu::ShaderModule,
    entry_point: &'static str,
) -> wgpu::ComputePipeline {
    let key = (label, entry_point);
    if let Some(pipeline) = pipelines.pipelines.lock().unwrap().get(&key).cloned() {
        return pipeline;
    }
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    pipelines
        .pipelines
        .lock()
        .unwrap()
        .insert(key, pipeline.clone());
    pipeline
}

fn bind_group(
    device: &wgpu::Device,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    entries: &[wgpu::BindGroupEntry<'_>],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pipeline.get_bind_group_layout(0),
        entries,
    })
}

fn timestamp_writes(
    timestamps: Option<&TimestampResources>,
    beginning: Option<u32>,
    end: Option<u32>,
) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
    if beginning.is_none() && end.is_none() {
        return None;
    }
    timestamps.map(|timestamps| wgpu::ComputePassTimestampWrites {
        query_set: &timestamps.query_set,
        beginning_of_pass_write_index: beginning,
        end_of_pass_write_index: end,
    })
}

fn direct_compute_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    bindings: &wgpu::BindGroup,
    workgroups: u32,
    timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bindings, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

fn indirect_compute_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    bindings: &wgpu::BindGroup,
    indirect_args: &wgpu::Buffer,
    indirect_offset: u64,
    timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bindings, &[]);
    pass.dispatch_workgroups_indirect(indirect_args, indirect_offset);
}

fn indirect_dispatch_in_pass<'a>(
    pass: &mut wgpu::ComputePass<'a>,
    pipeline: &'a wgpu::ComputePipeline,
    bindings: &'a wgpu::BindGroup,
    indirect_args: &'a wgpu::Buffer,
    indirect_offset: u64,
) {
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bindings, &[]);
    pass.dispatch_workgroups_indirect(indirect_args, indirect_offset);
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn create_uniform_buffer<T: Pod>(device: &wgpu::Device, label: &str, value: &T) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes_of(value),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

fn create_storage_buffer<T: Pod>(device: &wgpu::Device, label: &str, values: &[T]) -> wgpu::Buffer {
    create_buffer(
        device,
        label,
        values,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    )
}

fn create_state_buffer<T: Pod>(device: &wgpu::Device, label: &str, values: &[T]) -> wgpu::Buffer {
    create_buffer(
        device,
        label,
        values,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    )
}

fn create_readonly_storage_buffer<T: Pod>(
    device: &wgpu::Device,
    label: &str,
    values: &[T],
) -> wgpu::Buffer {
    create_buffer(device, label, values, wgpu::BufferUsages::STORAGE)
}

fn create_buffer<T: Pod>(
    device: &wgpu::Device,
    label: &str,
    values: &[T],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    if values.is_empty() {
        return device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: u64::try_from(size_of::<T>().max(16)).unwrap_or(16),
            usage,
            mapped_at_creation: false,
        });
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: cast_slice(values),
        usage,
    })
}

fn create_sized_buffer(
    device: &wgpu::Device,
    label: &str,
    size: usize,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: u64::try_from(size).unwrap_or(u64::MAX),
        usage,
        mapped_at_creation: false,
    })
}

fn vec4(vector: bevy_math::Vec3, w: f32) -> [f32; 4] {
    [vector.x, vector.y, vector.z, w]
}

fn wrapping_u32(value: u64) -> u32 {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[expect(clippy::cast_precision_loss)]
fn diagnostic_units(value: u32) -> f32 {
    value as f32 / 1_000_000.0
}

fn timestamp_milliseconds(start: u64, end: u64, period_nanoseconds: f64) -> f64 {
    let ticks = end.wrapping_sub(start);
    let bounded_ticks = u32::try_from(ticks).unwrap_or(u32::MAX);
    f64::from(bounded_ticks) * period_nanoseconds / 1_000_000.0
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

fn map_for_read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<(), GpuReadbackError> {
    let (sender, receiver) = mpsc::sync_channel(1);
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| GpuReadbackError::DevicePoll(error.to_string()))?;
    receiver
        .recv()
        .map_err(|_| GpuReadbackError::CallbackLost)?
        .map_err(|error| GpuReadbackError::BufferMap(error.to_string()))
}

fn create_async_readback_slot(
    device: &wgpu::Device,
    body_count: u32,
    coordinate_count: u32,
    timestamps_enabled: bool,
    index: usize,
) -> AsyncReadbackSlot {
    let readback_buffer = |label: String, size: u64| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&label),
            size: size.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    };
    let snapshot_size = u64::from(body_count) * 16;
    AsyncReadbackSlot {
        diagnostics: readback_buffer(
            format!("mechanic async diagnostics {index}"),
            u64::try_from(size_of::<GpuDiagnostics>()).unwrap_or(32),
        ),
        timestamps: timestamps_enabled
            .then(|| readback_buffer(format!("mechanic async timestamps {index}"), 224)),
        positions: readback_buffer(
            format!("mechanic async snapshot positions {index}"),
            snapshot_size,
        ),
        rotations: readback_buffer(
            format!("mechanic async snapshot rotations {index}"),
            snapshot_size,
        ),
        linear_velocities: readback_buffer(
            format!("mechanic async linear velocities {index}"),
            snapshot_size,
        ),
        angular_velocities: readback_buffer(
            format!("mechanic async angular velocities {index}"),
            snapshot_size,
        ),
        coordinates: readback_buffer(
            format!("mechanic async coordinates {index}"),
            u64::from(coordinate_count) * 8,
        ),
        pending: None,
    }
}

fn mapped_rows<T: Pod>(buffer: &wgpu::Buffer, count: u32) -> Vec<T> {
    if count == 0 {
        return Vec::new();
    }
    let bytes = buffer.get_mapped_range(..u64::from(count) * size_of::<T>() as u64);
    bytes
        .chunks_exact(size_of::<T>())
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

// A later mapping callback must not overtake an earlier incomplete snapshot.
fn oldest_completed_readback(pending: impl Iterator<Item = (usize, u64, u8)>) -> Option<usize> {
    pending
        .min_by_key(|&(_, sequence, _)| sequence)
        .and_then(|(index, _, remaining)| (remaining == 0).then_some(index))
}

fn begin_async_mapping(
    slot: &mut AsyncReadbackSlot,
    submission_sequence: u64,
    tick_index: u64,
    snapshot_slot: u8,
    submitted_at: Instant,
    timing: bool,
) {
    let callback_count = 6 + u8::from(slot.timestamps.is_some());
    let (sender, receiver) = mpsc::sync_channel(usize::from(callback_count));
    let map = |buffer: &wgpu::Buffer,
               sender: mpsc::SyncSender<(Result<(), String>, Option<Instant>)>| {
        buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            let observed_at = timing.then(Instant::now);
            let _ = sender.send((result.map_err(|error| error.to_string()), observed_at));
        });
    };
    map(&slot.diagnostics, sender.clone());
    if let Some(timestamps) = &slot.timestamps {
        map(timestamps, sender.clone());
    }
    map(&slot.positions, sender.clone());
    map(&slot.rotations, sender.clone());
    map(&slot.linear_velocities, sender.clone());
    map(&slot.angular_velocities, sender.clone());
    map(&slot.coordinates, sender);
    slot.pending = Some(PendingAsyncReadback {
        submission_sequence,
        tick_index,
        snapshot_slot,
        receiver,
        remaining_callbacks: callback_count,
        submitted_at,
        callbacks_completed_at: None,
    });
}

fn unmap_async_slot(slot: &AsyncReadbackSlot) {
    slot.diagnostics.unmap();
    if let Some(timestamps) = &slot.timestamps {
        timestamps.unmap();
    }
    slot.positions.unmap();
    slot.rotations.unmap();
    slot.linear_velocities.unmap();
    slot.angular_velocities.unmap();
    slot.coordinates.unmap();
}

fn read_vec4_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    byte_len: u64,
) -> Result<Vec<[f32; 4]>, GpuReadbackError> {
    map_for_read(device, buffer)?;
    let values = {
        let bytes = buffer.get_mapped_range(0..byte_len);
        cast_slice::<u8, [f32; 4]>(&bytes).to_vec()
    };
    buffer.unmap();
    Ok(values)
}

#[cfg(test)]
mod tests;
