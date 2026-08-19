use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::mpsc;

use bevy_math::Vec3;
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use mechanic_core::CompiledCreation;
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::{
    BROADPHASE_HASH_CAPACITY, FIXED_DT_SECONDS, GpuBearing, GpuCollider, GpuContact,
    GpuContractionNode, GpuDiagnostics, GpuLinkState, GpuMass, GpuMechanismBody,
    GpuMechanismCoordinate, GpuPair, GpuPersistentManifold, GpuSpatialInertia, GpuTickConfig,
    GpuTransform, MAX_BEARINGS, MAX_BODIES, MAX_COLLIDERS, MAX_CONTACT_PAIRS, SNAPSHOT_RING_SIZE,
};

/// Per-scene pipeline switches that do not adapt during simulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuPhysicsConfig {
    /// Whether broadphase, SAT, and projected contact impulses are dispatched.
    pub collisions_enabled: bool,
    /// Whether colliders in the same articulated mechanism may contact.
    pub mechanism_self_collisions: bool,
    /// Fixed number of projected impulse iterations.
    pub solver_iterations: u32,
}

impl Default for GpuPhysicsConfig {
    fn default() -> Self {
        Self {
            collisions_enabled: true,
            mechanism_self_collisions: true,
            solver_iterations: 8,
        }
    }
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
    /// Tick encoded into the submission.
    pub tick_index: u64,
    /// Snapshot ring destination written by that tick.
    pub snapshot_slot: u8,
    /// Shared queue submission token.
    pub submission_index: wgpu::SubmissionIndex,
}

/// Validation values copied back after a tick without reading body state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuTickReadback {
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
    /// Largest derived bearing anchor residual in metres.
    pub anchor_residual_meters: f32,
    /// Largest derived bearing axis residual in degrees.
    pub axis_residual_degrees: f32,
}

/// GPU timestamp durations for the fixed production pipeline stages.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuKernelTimings {
    /// Body gravity/damping and root-pose integration.
    pub integration_ms: f64,
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
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuExternalImpulse {
    world_point: [f32; 4],
    impulse: [f32; 4],
    metadata: [u32; 4],
}

/// Custom compute resources backed by Bevy's shared wgpu device and queue.
#[derive(Debug)]
pub struct GpuPhysics {
    body_count: u32,
    collider_count: u32,
    bearing_count: u32,
    suppression_count: u32,
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
    _masses: wgpu::Buffer,
    _spatial_inertias: wgpu::Buffer,
    _colliders: wgpu::Buffer,
    _bearings: wgpu::Buffer,
    external_impulse: wgpu::Buffer,
    external_impulse_pipeline: wgpu::ComputePipeline,
    external_impulse_bind_group: wgpu::BindGroup,
    snapshots: Vec<SnapshotBuffers>,
    bind_groups: Vec<wgpu::BindGroup>,
    integration_pipeline: wgpu::ComputePipeline,
    mechanism: MechanismResources,
    collision: CollisionResources,
    bearing_pipeline: wgpu::ComputePipeline,
    bearing_bind_group: wgpu::BindGroup,
    snapshot_pipeline: wgpu::ComputePipeline,
    snapshot_bind_groups: Vec<wgpu::BindGroup>,
    timestamps: Option<TimestampResources>,
}

#[derive(Debug)]
struct CollisionResources {
    lbvh: LbvhResources,
    _body_components: wgpu::Buffer,
    _pairs: wgpu::Buffer,
    _contacts: wgpu::Buffer,
    _manifold_keys: wgpu::Buffer,
    _persistent_manifolds: wgpu::Buffer,
    _active_contacts: wgpu::Buffer,
    indirect_args: wgpu::Buffer,
    velocity_deltas: wgpu::Buffer,
    _world_masses: wgpu::Buffer,
    update_world_masses_pipeline: wgpu::ComputePipeline,
    update_world_masses_bind_group: wgpu::BindGroup,
    narrowphase_pipeline: wgpu::ComputePipeline,
    narrowphase_bind_group: wgpu::BindGroup,
    ground_contacts_pipeline: wgpu::ComputePipeline,
    ground_contacts_bind_group: wgpu::BindGroup,
    finalize_contacts_pipeline: wgpu::ComputePipeline,
    finalize_contacts_bind_group: wgpu::BindGroup,
    select_active_pipeline: wgpu::ComputePipeline,
    select_active_bind_group: wgpu::BindGroup,
    finalize_active_pipeline: wgpu::ComputePipeline,
    finalize_active_bind_group: wgpu::BindGroup,
    warm_start_pipeline: wgpu::ComputePipeline,
    warm_start_bind_group: wgpu::BindGroup,
    solve_accumulate_pipeline: wgpu::ComputePipeline,
    solve_accumulate_bind_group: wgpu::BindGroup,
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
    _bodies: wgpu::Buffer,
    coordinates: wgpu::Buffer,
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
    apply_velocity_pipeline: wgpu::ComputePipeline,
    apply_velocity_bind_group: wgpu::BindGroup,
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
}

#[derive(Debug)]
struct TimestampResources {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_nanoseconds: f64,
}

impl GpuPhysics {
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
    #[allow(clippy::too_many_lines)]
    pub fn new_with_config(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        creation: &CompiledCreation,
        pipeline_config: GpuPhysicsConfig,
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
        let colliders = creation
            .colliders
            .iter()
            .map(|collider| {
                let rotation = collider.local_rotation;
                GpuCollider {
                    local_center: vec4(collider.local_center, 0.0),
                    local_rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
                    half_extents: vec4(collider.half_extents, 0.0),
                    metadata: [
                        collider.compound_index,
                        collider.source_part.index(),
                        collider.source_part.generation(),
                        0,
                    ],
                }
            })
            .collect::<Vec<_>>();
        let bearings = creation
            .bearings
            .iter()
            .map(|bearing| GpuBearing {
                local_anchor_a: vec4(bearing.local_anchor_a, 0.0),
                local_anchor_b: vec4(bearing.local_anchor_b, 0.0),
                local_axis_a: vec4(bearing.local_axis_a, 0.0),
                local_axis_b: vec4(bearing.local_axis_b, 0.0),
                metadata: [
                    bearing.compound_a,
                    bearing.compound_b,
                    bearing.coordinate_index.unwrap_or(u32::MAX),
                    u32::from(bearing.coordinate_index.is_none()),
                ],
            })
            .collect::<Vec<_>>();
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
            create_storage_buffer(device, "mechanic linear velocities", &zero_vectors);
        let angular_velocities =
            create_storage_buffer(device, "mechanic angular velocities", &zero_vectors);
        let inverse_masses =
            create_readonly_storage_buffer(device, "mechanic inverse masses", &inverse_masses);
        let diagnostics = create_buffer(
            device,
            "mechanic diagnostics",
            &[GpuDiagnostics::zeroed()],
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
        let masses = create_readonly_storage_buffer(device, "mechanic mass rows", &masses);
        let spatial_inertias = create_readonly_storage_buffer(
            device,
            "mechanic direct spatial inertia rows",
            &spatial_inertias,
        );
        let colliders = create_readonly_storage_buffer(device, "mechanic colliders", &colliders);
        let bearings = create_readonly_storage_buffer(device, "mechanic bearings", &bearings);
        let suppressed_pairs = create_readonly_storage_buffer(
            device,
            "mechanic collision suppression",
            &suppressed_pairs,
        );

        let mechanism = create_mechanism_resources(
            device,
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mechanic physics kernels"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("kernels/physics.wgsl"))),
        });
        let integration_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mechanic integrate and snapshot"),
                layout: None,
                module: &shader,
                entry_point: Some("integrate"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let external_impulse = create_uniform_buffer(
            device,
            "mechanic external impulse",
            &GpuExternalImpulse::zeroed(),
        );
        let external_impulse_pipeline = compute_pipeline(
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
            device,
            "mechanic snapshot kernel",
            include_str!("kernels/snapshot.wgsl"),
        );
        let snapshot_pipeline = compute_pipeline(
            device,
            "mechanic publish snapshot",
            &snapshot_shader,
            "publish_snapshot",
        );
        let timestamps = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| TimestampResources {
                query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("mechanic physics timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 14,
                }),
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mechanic timestamp resolve"),
                    size: 112,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mechanic timestamp readback"),
                    size: 112,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                period_nanoseconds: f64::from(queue.get_timestamp_period()),
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
                ],
            ));
            snapshots.push(snapshot);
        }

        let collision = create_collision_resources(
            device,
            creation.colliders.len(),
            &config,
            &positions_buffer,
            &rotations_buffer,
            &linear_velocities,
            &angular_velocities,
            &masses,
            &diagnostics,
            &colliders,
            &suppressed_pairs,
            &body_components,
            pipeline_config.mechanism_self_collisions,
        );
        let bearing_shader = shader_module(
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

        // Make the upload boundary explicit before the first fixed tick.
        queue.write_buffer(&diagnostics, 0, bytes_of(&GpuDiagnostics::zeroed()));
        Ok(Self {
            body_count,
            collider_count,
            bearing_count,
            suppression_count,
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
            _masses: masses,
            _spatial_inertias: spatial_inertias,
            _colliders: colliders,
            _bearings: bearings,
            external_impulse,
            external_impulse_pipeline,
            external_impulse_bind_group,
            snapshots,
            bind_groups,
            integration_pipeline,
            mechanism,
            collision,
            bearing_pipeline,
            bearing_bind_group,
            snapshot_pipeline,
            snapshot_bind_groups,
            timestamps,
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
        if body_index >= self.body_count {
            return Err(GpuImpulseError::BodyIndexOutOfRange {
                body_index,
                body_count: self.body_count,
            });
        }
        if !world_point.is_finite() || !impulse.is_finite() {
            return Err(GpuImpulseError::NonFinite);
        }
        let row = GpuExternalImpulse {
            world_point: [world_point.x, world_point.y, world_point.z, 0.0],
            impulse: [impulse.x, impulse.y, impulse.z, 0.0],
            metadata: [body_index, 0, 0, 0],
        };
        queue.write_buffer(&self.external_impulse, 0, bytes_of(&row));
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

    /// Encodes and submits one 60 Hz integration/publication pass.
    pub fn dispatch_tick(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tick_index: u64,
    ) -> GpuTickSubmission {
        let snapshot_slot = u8::try_from(tick_index % 3).unwrap_or(0);
        let config = GpuTickConfig {
            body_count: self.body_count,
            tick_index: wrapping_u32(tick_index),
            snapshot_slot: u32::from(snapshot_slot),
            collider_count: self.collider_count,
            delta_seconds: FIXED_DT_SECONDS,
            gravity_y: -9.81,
            linear_damping: 0.999,
            angular_damping: 0.98,
            bearing_count: self.bearing_count,
            suppression_count: self.suppression_count,
            pair_capacity: u32::try_from(MAX_CONTACT_PAIRS).unwrap_or(u32::MAX),
            flags: u32::from(self.pipeline_config.collisions_enabled),
            hash_capacity: u32::try_from(BROADPHASE_HASH_CAPACITY).unwrap_or(u32::MAX),
            solver_iterations: self.pipeline_config.solver_iterations.max(1),
            reserved_a: self.collision.lbvh.sort_count,
            reserved_b: self.mechanism.coordinate_count,
        };
        queue.write_buffer(&self.config, 0, bytes_of(&config));
        queue.write_buffer(&self.diagnostics, 0, bytes_of(&GpuDiagnostics::zeroed()));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mechanic physics tick"),
        });
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
        if run_collisions {
            self.encode_collision_passes(&mut encoder);
        }
        if self.mechanism.active && run_collisions {
            self.encode_post_contact_mechanism(&mut encoder);
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
            encoder.resolve_query_set(&timestamps.query_set, 0..14, &timestamps.resolve, 0);
            encoder.copy_buffer_to_buffer(&timestamps.resolve, 0, &timestamps.readback, 0, 112);
        }
        let submission_index = queue.submit([encoder.finish()]);
        GpuTickSubmission {
            tick_index,
            snapshot_slot,
            submission_index,
        }
    }

    fn encode_mechanism_passes(&self, encoder: &mut wgpu::CommandEncoder) {
        let mechanism = &self.mechanism;
        let workgroups = self.body_count.div_ceil(256);
        self.encode_bearing_velocity_projection(encoder, true);
        direct_compute_pass(
            encoder,
            "mechanic advance reduced coordinates",
            &mechanism.advance_coordinates_pipeline,
            &mechanism.advance_coordinates_bind_group,
            workgroups,
            None,
        );
        self.encode_mechanism_forward_kinematics(encoder, false, 0);
        if mechanism.closure_count > 0 {
            const CLOSURE_CORRECTION_STEPS: u32 = 12;
            for step in 0..CLOSURE_CORRECTION_STEPS {
                encoder.clear_buffer(&mechanism.closure_accumulators, 0, None);
                encoder.clear_buffer(&mechanism.closure_state, 0, None);
                if step == 0 {
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
                self.encode_mechanism_forward_kinematics(encoder, true, 12);
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
        direct_compute_pass(
            encoder,
            "mechanic publish mechanism poses",
            pipeline,
            bindings,
            workgroups,
            None,
        );
        direct_compute_pass(
            encoder,
            "mechanic reconstruct body velocities",
            &mechanism.reconstruct_velocities_pipeline,
            &mechanism.reconstruct_velocities_bind_group,
            1,
            timestamp_writes(self.timestamps.as_ref(), None, Some(3)),
        );
    }

    fn encode_bearing_velocity_projection(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_start: bool,
    ) {
        let mechanism = &self.mechanism;
        for iteration in 0..self.pipeline_config.solver_iterations.max(1) {
            direct_compute_pass(
                encoder,
                "mechanic project bearing velocities",
                &mechanism.project_velocity_pipeline,
                &mechanism.project_velocity_bind_group,
                self.bearing_count.div_ceil(256),
                if timestamp_start && iteration == 0 {
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
    }

    fn encode_post_contact_mechanism(&self, encoder: &mut wgpu::CommandEncoder) {
        let mechanism = &self.mechanism;
        self.encode_bearing_velocity_projection(encoder, false);
        direct_compute_pass(
            encoder,
            "mechanic capture reduced velocities",
            &mechanism.capture_coordinates_pipeline,
            &mechanism.capture_coordinates_bind_group,
            self.body_count.div_ceil(256),
            None,
        );
        direct_compute_pass(
            encoder,
            "mechanic reconstruct post-contact velocities",
            &mechanism.reconstruct_velocities_pipeline,
            &mechanism.reconstruct_velocities_bind_group,
            1,
            timestamp_writes(self.timestamps.as_ref(), None, Some(9)),
        );
    }

    fn encode_mechanism_forward_kinematics(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        indirect: bool,
        indirect_offset: u64,
    ) {
        let mechanism = &self.mechanism;
        let workgroups = self.body_count.div_ceil(256);
        let mut dispatch = |label: &str,
                            pipeline: &wgpu::ComputePipeline,
                            bindings: &wgpu::BindGroup,
                            timestamp_writes| {
            if indirect {
                indirect_compute_pass(
                    encoder,
                    label,
                    pipeline,
                    bindings,
                    &mechanism.closure_indirect_args,
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

    #[allow(clippy::too_many_lines)]
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
        direct_compute_pass(
            encoder,
            "mechanic ground contacts",
            &collision.ground_contacts_pipeline,
            &collision.ground_contacts_bind_group,
            self.collider_count.div_ceil(256),
            None,
        );
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
        let iterations = self.pipeline_config.solver_iterations.max(1);
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

        let timestamp_readback = self
            .timestamps
            .as_ref()
            .map(|timestamps| {
                map_for_read(device, &timestamps.readback)?;
                let values = {
                    let bytes = timestamps.readback.get_mapped_range(0..112);
                    let mut values = [0_u64; 14];
                    for (index, chunk) in bytes.chunks_exact(8).enumerate() {
                        let mut raw = [0_u8; 8];
                        raw.copy_from_slice(chunk);
                        values[index] = u64::from_ne_bytes(raw);
                    }
                    values
                };
                timestamps.readback.unmap();
                let elapsed = |start, end| {
                    timestamp_milliseconds(
                        values[start],
                        values[end],
                        timestamps.period_nanoseconds,
                    )
                };
                let timings = GpuKernelTimings {
                    integration_ms: elapsed(0, 1),
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
                let total = timings.integration_ms
                    + timings.mechanism_ms
                    + timings.broadphase_ms
                    + timings.narrowphase_ms
                    + timings.contact_solver_ms
                    + timings.bearings_ms
                    + timings.snapshot_ms;
                Ok((total, timings))
            })
            .transpose()?;
        Ok(GpuTickReadback {
            gpu_tick_ms: timestamp_readback.map(|(total, _)| total),
            kernel_timings: timestamp_readback.map(|(_, timings)| timings),
            error_flags: diagnostics.error_flags,
            pair_count: diagnostics.pair_count,
            contact_count: diagnostics.contact_count,
            active_contact_count: diagnostics.active_contact_count,
            anchor_residual_meters: diagnostic_units(diagnostics.max_anchor_micrometers),
            axis_residual_degrees: diagnostic_units(diagnostics.max_axis_microdegrees),
        })
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

#[allow(
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
fn create_mechanism_resources(
    device: &wgpu::Device,
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
    let bearing_rows = creation
        .bearings
        .iter()
        .enumerate()
        .map(|(row, bearing)| (bearing.source_bearing, row))
        .collect::<BTreeMap<_, _>>();
    let root_flags = creation
        .loop_topology
        .body_parents
        .iter()
        .map(|body| u32::from(body.is_root))
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
    let bodies = create_readonly_storage_buffer(device, "mechanic mechanism bodies", &bodies);
    let coordinate_count = u32::try_from(coordinates.len()).unwrap_or(u32::MAX);
    let closure_count =
        u32::try_from(creation.loop_topology.closure_bearings.len()).unwrap_or(u32::MAX);
    let coordinates = create_storage_buffer(device, "mechanic mechanism coordinates", &coordinates);
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
        device,
        "mechanic mechanism kernels",
        include_str!("kernels/mechanism.wgsl"),
    );
    let prepare_pipeline = compute_pipeline(
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
    let publish_a_pipeline =
        compute_pipeline(device, "mechanic publish mechanism A", &shader, "publish_a");
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
    let publish_b_pipeline =
        compute_pipeline(device, "mechanic publish mechanism B", &shader, "publish_b");
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
        device,
        "mechanic articulated dynamics kernels",
        include_str!("kernels/articulated.wgsl"),
    );
    let project_velocity_pipeline = compute_pipeline(
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
            entry(2, rotations),
            entry(3, linear_velocities),
            entry(4, angular_velocities),
            entry(5, masses),
            entry(6, bearings),
            entry(9, &velocity_deltas),
        ],
    );
    let apply_velocity_pipeline = compute_pipeline(
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
            entry(2, rotations),
            entry(4, angular_velocities),
            entry(6, bearings),
            entry(7, &bodies),
            entry(8, &coordinates),
        ],
    );
    let capture_coordinates_pipeline = compute_pipeline(
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
            entry(2, rotations),
            entry(4, angular_velocities),
            entry(6, bearings),
            entry(7, &bodies),
            entry(8, &coordinates),
        ],
    );
    let reconstruct_velocities_pipeline = compute_pipeline(
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
        device,
        "mechanic closure kernels",
        include_str!("kernels/closure.wgsl"),
    );
    let evaluate_closures_pipeline = compute_pipeline(
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
        _bodies: bodies,
        coordinates,
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
        apply_velocity_pipeline,
        apply_velocity_bind_group,
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
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn create_lbvh_resources(
    device: &wgpu::Device,
    collider_count: usize,
    config: &wgpu::Buffer,
    positions: &wgpu::Buffer,
    rotations: &wgpu::Buffer,
    diagnostics: &wgpu::Buffer,
    colliders: &wgpu::Buffer,
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
        device,
        "mechanic LBVH kernels",
        include_str!("kernels/lbvh.wgsl"),
    );
    let compute_morton_pipeline = compute_pipeline(
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
        ],
    );
    let sort_local_initial_pipeline = compute_pipeline(
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
    let build_topology_pipeline =
        compute_pipeline(device, "mechanic LBVH topology", &shader, "build_topology");
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
    let build_bounds_pipeline =
        compute_pipeline(device, "mechanic LBVH bounds", &shader, "build_bounds");
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
    let traverse_pipeline =
        compute_pipeline(device, "mechanic LBVH traversal", &shader, "traverse");
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

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn create_collision_resources(
    device: &wgpu::Device,
    collider_count: usize,
    config: &wgpu::Buffer,
    positions: &wgpu::Buffer,
    rotations: &wgpu::Buffer,
    linear_velocities: &wgpu::Buffer,
    angular_velocities: &wgpu::Buffer,
    masses: &wgpu::Buffer,
    diagnostics: &wgpu::Buffer,
    colliders: &wgpu::Buffer,
    suppressed_pairs: &wgpu::Buffer,
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
        MAX_CONTACT_PAIRS * size_of::<GpuPair>(),
        wgpu::BufferUsages::STORAGE,
    );
    let contacts = create_sized_buffer(
        device,
        "mechanic contact manifolds",
        MAX_CONTACT_PAIRS * size_of::<GpuContact>(),
        wgpu::BufferUsages::STORAGE,
    );
    let manifold_keys = create_sized_buffer(
        device,
        "mechanic persistent manifold keys",
        MAX_CONTACT_PAIRS * size_of::<u32>(),
        wgpu::BufferUsages::STORAGE,
    );
    let persistent_manifolds = create_sized_buffer(
        device,
        "mechanic persistent manifolds",
        MAX_CONTACT_PAIRS * size_of::<GpuPersistentManifold>(),
        wgpu::BufferUsages::STORAGE,
    );
    let active_contacts = create_sized_buffer(
        device,
        "mechanic active contact indices",
        MAX_CONTACT_PAIRS * size_of::<u32>(),
        wgpu::BufferUsages::STORAGE,
    );
    let indirect_args = create_sized_buffer(
        device,
        "mechanic indirect dispatch arguments",
        12 * size_of::<u32>(),
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
        MAX_BODIES * 48,
        wgpu::BufferUsages::STORAGE,
    );
    let lbvh = create_lbvh_resources(
        device,
        collider_count,
        config,
        positions,
        rotations,
        diagnostics,
        colliders,
        &pairs,
        suppressed_pairs,
        &indirect_args,
    );

    let shader = shader_module(
        device,
        "mechanic collision kernels",
        include_str!("kernels/collision.wgsl"),
    );
    let update_world_masses_pipeline = compute_pipeline(
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
    let narrowphase_pipeline =
        compute_pipeline(device, "mechanic OBB SAT", &shader, narrowphase_entry);
    let mut narrowphase_bindings = vec![
        entry(0, config),
        entry(1, positions),
        entry(2, rotations),
        entry(5, diagnostics),
        entry(6, colliders),
        entry(9, &pairs),
        entry(10, &contacts),
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
        device,
        "mechanic ground contacts",
        &shader,
        "generate_ground_contacts",
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
        ],
    );
    let finalize_contacts_pipeline = compute_pipeline(
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
        ],
    );
    let warm_start_pipeline = compute_pipeline(
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
            entry(26, &world_masses),
            entry(25, angular_velocities),
        ],
    );
    let solve_accumulate_pipeline = compute_pipeline(
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
            entry(26, &world_masses),
            entry(25, angular_velocities),
        ],
    );
    let solve_apply_pipeline = compute_pipeline(
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
        _body_components: body_components,
        _pairs: pairs,
        _contacts: contacts,
        _manifold_keys: manifold_keys,
        _persistent_manifolds: persistent_manifolds,
        _active_contacts: active_contacts,
        indirect_args,
        velocity_deltas,
        _world_masses: world_masses,
        update_world_masses_pipeline,
        update_world_masses_bind_group,
        narrowphase_pipeline,
        narrowphase_bind_group,
        ground_contacts_pipeline,
        ground_contacts_bind_group,
        finalize_contacts_pipeline,
        finalize_contacts_bind_group,
        select_active_pipeline,
        select_active_bind_group,
        finalize_active_pipeline,
        finalize_active_bind_group,
        warm_start_pipeline,
        warm_start_bind_group,
        solve_accumulate_pipeline,
        solve_accumulate_bind_group,
        solve_apply_pipeline,
        solve_apply_bind_group,
        persist_contacts_pipeline,
        persist_contacts_bind_group,
    }
}

fn shader_module(device: &wgpu::Device, label: &str, source: &'static str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    })
}

fn compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    shader: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
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

#[allow(clippy::too_many_arguments)]
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

#[allow(clippy::cast_precision_loss)]
fn diagnostic_units(value: u32) -> f32 {
    value as f32 / 1_000_000.0
}

fn timestamp_milliseconds(start: u64, end: u64, period_nanoseconds: f64) -> f64 {
    let ticks = end.wrapping_sub(start);
    let bounded_ticks = u32::try_from(ticks).unwrap_or(u32::MAX);
    f64::from(bounded_ticks) * period_nanoseconds / 1_000_000.0
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
mod tests {
    use bevy_math::{IVec3, Vec3};
    use mechanic_core::{
        BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec,
        CylinderDimensions, CylinderSpec, FaceKind, FaceRef, GridRotation, RigidLinkSpec, WeldSpec,
    };

    use crate::GpuMechanismCoordinate;

    use super::{GpuPhysics, GpuPhysicsConfig};

    #[test]
    fn physics_wgsl_parses_and_validates_without_a_gpu() {
        for (name, source) in [
            ("physics", include_str!("kernels/physics.wgsl")),
            ("collision", include_str!("kernels/collision.wgsl")),
            ("lbvh", include_str!("kernels/lbvh.wgsl")),
            ("bearings", include_str!("kernels/bearings.wgsl")),
            ("mechanism", include_str!("kernels/mechanism.wgsl")),
            ("articulated", include_str!("kernels/articulated.wgsl")),
            ("closure", include_str!("kernels/closure.wgsl")),
            ("snapshot", include_str!("kernels/snapshot.wgsl")),
        ] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|error| panic!("{name} WGSL parses: {error}"));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{name} WGSL validates: {error:#?}"));
        }
    }

    fn pendulum_creation(grounded: bool) -> mechanic_core::CompiledCreation {
        let mut graph = ConstructionGraph::new();
        let mut spawn = |units| {
            let spec =
                CuboidSpec::new([4, 4, 4], BuildPose::new(units, GridRotation::default())).unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        };
        let root = spawn(IVec3::new(0, 2, 0));
        let arm_a = spawn(IVec3::new(4, 2, 0));
        let arm_b = spawn(IVec3::new(4, 2, 4));
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(arm_a, FaceKind::PositiveZ),
                second: FaceRef::part(arm_b, FaceKind::NegativeZ),
            }))
            .unwrap();
        if grounded {
            graph
                .apply(BuildCommand::Weld(WeldSpec {
                    first: FaceRef::part(root, FaceKind::NegativeY),
                    second: FaceRef::ground(),
                }))
                .unwrap();
        }
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(root, FaceKind::PositiveX),
                FaceRef::part(arm_a, FaceKind::NegativeX),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::X,
            )))
            .unwrap();
        graph.compile().unwrap()
    }

    fn tall_pendulum_creation(second_link: bool) -> mechanic_core::CompiledCreation {
        let mut graph = ConstructionGraph::new();
        let (support, arm, child) = {
            let mut spawn = |units| {
                let spec =
                    CuboidSpec::new([2, 2, 2], BuildPose::new(units, GridRotation::default()))
                        .unwrap();
                let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
                else {
                    unreachable!()
                };
                part
            };
            let support = (0..7)
                .map(|row| spawn(IVec3::new(0, row * 2 + 1, 0)))
                .collect::<Vec<_>>();
            let arm = (0..4)
                .map(|column| spawn(IVec3::new(2, 13, column * 2)))
                .collect::<Vec<_>>();
            let child = second_link.then(|| {
                (0..3)
                    .map(|row| spawn(IVec3::new(2, row * 2 + 13, 8)))
                    .collect::<Vec<_>>()
            });
            (support, arm, child)
        };
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(support[0], FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();
        for pair in support.windows(2) {
            graph
                .apply(BuildCommand::Weld(WeldSpec {
                    first: FaceRef::part(pair[0], FaceKind::PositiveY),
                    second: FaceRef::part(pair[1], FaceKind::NegativeY),
                }))
                .unwrap();
        }
        for pair in arm.windows(2) {
            graph
                .apply(BuildCommand::Weld(WeldSpec {
                    first: FaceRef::part(pair[0], FaceKind::PositiveZ),
                    second: FaceRef::part(pair[1], FaceKind::NegativeZ),
                }))
                .unwrap();
        }
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(support[6], FaceKind::PositiveX),
                FaceRef::part(arm[0], FaceKind::NegativeX),
                Vec3::new(0.25, 3.25, 0.0),
                Vec3::X,
            )))
            .unwrap();
        if let Some(child) = child {
            for pair in child.windows(2) {
                graph
                    .apply(BuildCommand::Weld(WeldSpec {
                        first: FaceRef::part(pair[0], FaceKind::PositiveY),
                        second: FaceRef::part(pair[1], FaceKind::NegativeY),
                    }))
                    .unwrap();
            }
            graph
                .apply(BuildCommand::AddBearing(BearingSpec::new(
                    FaceRef::part(arm[3], FaceKind::PositiveZ),
                    FaceRef::part(child[0], FaceKind::NegativeZ),
                    Vec3::new(0.5, 3.25, 1.75),
                    Vec3::Z,
                )))
                .unwrap();
        }
        graph.compile().unwrap()
    }

    fn branching_pendulum_creation(
        arm_count: usize,
        hanging: bool,
    ) -> mechanic_core::CompiledCreation {
        let mut graph = ConstructionGraph::new();
        let root = CuboidSpec::new(
            [8, 4, 8],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(root) = graph.apply(BuildCommand::Spawn(root)).unwrap() else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(root, FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();
        let bar = CuboidSpec::new(
            [24, 2, 2],
            BuildPose::from_half_grid(IVec3::new(0, 10, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(bar) = graph.apply(BuildCommand::Spawn(bar)).unwrap() else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(root, FaceKind::PositiveY),
                FaceRef::part(bar, FaceKind::NegativeY),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::Y,
            )))
            .unwrap();

        for (center, bar_face, arm_face, anchor, axis) in [
            (
                IVec3::new(26, if hanging { 7 } else { 16 }, 1),
                FaceKind::PositiveX,
                FaceKind::NegativeX,
                Vec3::new(3.0, 1.25, 0.0),
                Vec3::X,
            ),
            (
                IVec3::new(-26, if hanging { 7 } else { 16 }, -1),
                FaceKind::NegativeX,
                FaceKind::PositiveX,
                Vec3::new(-3.0, 1.25, 0.0),
                Vec3::NEG_X,
            ),
        ]
        .into_iter()
        .take(arm_count)
        {
            let arm = CuboidSpec::new(
                [2, 8, 2],
                BuildPose::from_half_grid(center, GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(arm) = graph.apply(BuildCommand::Spawn(arm)).unwrap() else {
                unreachable!()
            };
            graph
                .apply(BuildCommand::AddBearing(BearingSpec::new(
                    FaceRef::part(bar, bar_face),
                    FaceRef::part(arm, arm_face),
                    anchor,
                    axis,
                )))
                .unwrap();
        }
        graph.compile().unwrap()
    }

    fn swinging_arm_with_coaxial_rotor_creation() -> mechanic_core::CompiledCreation {
        let mut graph = ConstructionGraph::new();
        let root = CuboidSpec::new(
            [8, 8, 8],
            BuildPose::new(IVec3::new(0, 4, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(root) = graph.apply(BuildCommand::Spawn(root)).unwrap() else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(root, FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();

        let arm = CuboidSpec::new(
            [2, 2, 24],
            BuildPose::from_half_grid(IVec3::new(10, 10, 24), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(arm) = graph.apply(BuildCommand::Spawn(arm)).unwrap() else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(root, FaceKind::PositiveX),
                FaceRef::part(arm, FaceKind::NegativeX),
                Vec3::new(1.0, 1.25, 0.25),
                Vec3::X,
            )))
            .unwrap();

        let rotor = CylinderSpec::new(
            CylinderDimensions::new(0.25, 0.20, 0.75).unwrap(),
            BuildPose::from_half_grid(IVec3::new(10, 5, 46), GridRotation::default()),
        );
        let BuildOutcome::Spawned(rotor) = graph.apply(BuildCommand::SpawnCylinder(rotor)).unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(arm, FaceKind::NegativeY),
                FaceRef::part(rotor, FaceKind::PositiveY),
                Vec3::new(1.25, 1.0, 5.75),
                Vec3::NEG_Y,
            )))
            .unwrap();
        graph.compile().unwrap()
    }

    fn relative_bearing_rotation(
        snapshot: &[crate::GpuTransform],
        bearing: &mechanic_core::CompiledBearing,
    ) -> bevy_math::Quat {
        let a = bevy_math::Quat::from_array(snapshot[bearing.compound_a as usize].rotation);
        let b = bevy_math::Quat::from_array(snapshot[bearing.compound_b as usize].rotation);
        a.conjugate() * b
    }

    fn run_long_pendulum(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        creation: &mechanic_core::CompiledCreation,
        collisions_enabled: bool,
    ) -> (Vec<f32>, Vec<f32>, super::GpuTickReadback) {
        let gpu = GpuPhysics::new_with_config(
            device,
            queue,
            creation,
            GpuPhysicsConfig {
                collisions_enabled,
                ..Default::default()
            },
        )
        .unwrap();
        for tick in 1..=1_200 {
            gpu.dispatch_tick(device, queue, tick);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let current = gpu.read_snapshot_transforms(device, queue, 0).unwrap();
        let previous = gpu.read_snapshot_transforms(device, queue, 2).unwrap();
        let diagnostics = gpu.read_last_tick(device).unwrap();
        let mut angles = Vec::with_capacity(creation.bearings.len());
        let mut angular_speeds = Vec::with_capacity(creation.bearings.len());
        for bearing in &creation.bearings {
            let current_relative = relative_bearing_rotation(&current, bearing);
            let previous_relative = relative_bearing_rotation(&previous, bearing);
            angles.push(
                2.0 * current_relative
                    .xyz()
                    .dot(bearing.local_axis_a)
                    .atan2(current_relative.w),
            );
            let delta = previous_relative.conjugate() * current_relative;
            angular_speeds.push(2.0 * delta.xyz().length().atan2(delta.w.abs()) * 60.0);
        }
        (angles, angular_speeds, diagnostics)
    }

    #[test]
    fn tall_single_and_double_pendulums_dissipate_energy() {
        let Some((device, queue)) = test_device() else {
            return;
        };

        let single = tall_pendulum_creation(false);
        assert_eq!(single.colliders.len(), 11);
        let (single_angles, single_speeds, single_diagnostics) =
            run_long_pendulum(&device, &queue, &single, true);
        assert_eq!(single_diagnostics.error_flags, 0);
        assert_eq!(single_diagnostics.active_contact_count, 0);
        assert!(
            (single_angles[0] - std::f32::consts::FRAC_PI_2).abs() < 0.02,
            "single pendulum angle was {}",
            single_angles[0]
        );
        assert!(single_speeds[0] < 0.2);

        let double = tall_pendulum_creation(true);
        assert_eq!(double.colliders.len(), 14);
        let (free_angles, free_speeds, free_diagnostics) =
            run_long_pendulum(&device, &queue, &double, false);
        assert_eq!(free_diagnostics.error_flags, 0);
        assert!((free_angles[0] - std::f32::consts::FRAC_PI_2).abs() < 0.02);
        assert!((free_angles[1] + std::f32::consts::FRAC_PI_2).abs() < 0.02);
        assert!(
            free_speeds.iter().all(|speed| *speed < 1.0e-4),
            "double pendulum retained free speeds {free_speeds:?}"
        );

        let (_, contact_speeds, contact_diagnostics) =
            run_long_pendulum(&device, &queue, &double, true);
        assert_eq!(contact_diagnostics.error_flags, 0);
        assert!(
            contact_speeds.iter().all(|speed| *speed < 1.0e-4),
            "double pendulum retained contact speeds {contact_speeds:?}"
        );
    }

    #[test]
    fn branching_pendulum_reaches_rest_instead_of_retaining_spin() {
        let Some((device, queue)) = test_device() else {
            return;
        };
        for (arm_count, hanging) in [(1, false), (2, false), (1, true), (2, true)] {
            let creation = branching_pendulum_creation(arm_count, hanging);
            let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
            if arm_count == 1 && hanging {
                gpu.initialize_mechanism_coordinates(
                    &queue,
                    &[
                        GpuMechanismCoordinate {
                            angle: 0.0,
                            angular_velocity: 0.0,
                        },
                        GpuMechanismCoordinate {
                            angle: 0.35,
                            angular_velocity: 0.0,
                        },
                    ],
                )
                .unwrap();
            }
            let sample = |tick: u64| {
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let current = gpu
                    .read_snapshot_transforms(&device, &queue, u8::try_from(tick % 3).unwrap())
                    .unwrap();
                let previous = gpu
                    .read_snapshot_transforms(
                        &device,
                        &queue,
                        u8::try_from((tick - 1) % 3).unwrap(),
                    )
                    .unwrap();
                creation
                    .bearings
                    .iter()
                    .map(|bearing| {
                        let current_relative = relative_bearing_rotation(&current, bearing);
                        let previous_relative = relative_bearing_rotation(&previous, bearing);
                        let delta = previous_relative.conjugate() * current_relative;
                        2.0 * delta.xyz().length().atan2(delta.w.abs()) * 60.0
                    })
                    .collect::<Vec<_>>()
            };
            for tick in 1..=1_200 {
                gpu.dispatch_tick(&device, &queue, tick);
            }
            let early = sample(1_200);
            for tick in 1_201..=2_400 {
                gpu.dispatch_tick(&device, &queue, tick);
            }
            let late = sample(2_400);
            assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
            assert!(
                late.iter().all(|speed| *speed < 1.0e-4),
                "{arm_count}-arm hanging={hanging} bearing speeds plateaued from {early:?} to {late:?}"
            );
        }
    }

    #[test]
    fn moving_coaxial_rotor_does_not_gain_perpetual_spin() {
        let Some((device, queue)) = test_device() else {
            return;
        };
        let creation = swinging_arm_with_coaxial_rotor_creation();
        let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
        gpu.initialize_mechanism_coordinates(
            &queue,
            &[
                GpuMechanismCoordinate {
                    angle: 0.35,
                    angular_velocity: 0.0,
                },
                GpuMechanismCoordinate {
                    angle: 0.0,
                    angular_velocity: 0.0,
                },
            ],
        )
        .unwrap();
        let sample = |tick: u64| {
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let current = gpu
                .read_snapshot_transforms(&device, &queue, u8::try_from(tick % 3).unwrap())
                .unwrap();
            let previous = gpu
                .read_snapshot_transforms(&device, &queue, u8::try_from((tick - 1) % 3).unwrap())
                .unwrap();
            creation
                .bearings
                .iter()
                .map(|bearing| {
                    let current_relative = relative_bearing_rotation(&current, bearing);
                    let previous_relative = relative_bearing_rotation(&previous, bearing);
                    let delta = previous_relative.conjugate() * current_relative;
                    2.0 * delta.xyz().length().atan2(delta.w.abs()) * 60.0
                })
                .collect::<Vec<_>>()
        };
        for tick in 1..=1_200 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        let early_speeds = sample(1_200);
        for tick in 1_201..=2_400 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        let late_speeds = sample(2_400);
        let diagnostics = gpu.read_last_tick(&device).unwrap();
        assert_eq!(diagnostics.error_flags, 0);
        assert_eq!(diagnostics.active_contact_count, 0);
        assert!(
            late_speeds.iter().all(|speed| *speed < 1.0e-4),
            "bearing state failed to settle from {early_speeds:?} to {late_speeds:?}"
        );
    }

    fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mechanic articulated test device"),
            ..Default::default()
        }))
        .ok()
    }

    #[test]
    fn gpu_pipelines_construct_on_noop_backend() {
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
                    mechanism_self_collisions,
                    solver_iterations: 8,
                },
            )
            .unwrap();
        }
    }

    #[test]
    fn off_centre_external_impulse_changes_linear_and_angular_motion() {
        let Some((device, queue)) = test_device() else {
            return;
        };
        let mut graph = ConstructionGraph::new();
        let spec = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        let creation = graph.compile().unwrap();
        let body = creation.part_to_compound[0].1;
        let initial = creation.compounds[body as usize].root_translation;
        let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
        gpu.apply_impulse(
            &device,
            &queue,
            body,
            initial + Vec3::Y * 0.5,
            Vec3::X * 500.0,
        )
        .unwrap();
        gpu.dispatch_tick(&device, &queue, 1);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let snapshot = gpu.read_snapshot_transforms(&device, &queue, 1).unwrap();
        assert!(snapshot[body as usize].position[0] > initial.x + 0.01);
        assert!(snapshot[body as usize].rotation[2] < -0.01);
        assert_eq!(creation.part_to_compound[0].0, part);
    }

    #[test]
    fn external_impulse_drives_a_bearing_coordinate() {
        let Some((device, queue)) = test_device() else {
            return;
        };
        let creation = pendulum_creation(true);
        let child = creation.bearings[0].compound_b;
        let run = |impulse: Option<Vec3>| {
            let gpu = GpuPhysics::new_with_config(
                &device,
                &queue,
                &creation,
                GpuPhysicsConfig {
                    collisions_enabled: false,
                    ..Default::default()
                },
            )
            .unwrap();
            if let Some(impulse) = impulse {
                gpu.apply_impulse(
                    &device,
                    &queue,
                    child,
                    creation.compounds[child as usize].root_translation,
                    impulse,
                )
                .unwrap();
            }
            for tick in 1..=10 {
                gpu.dispatch_tick(&device, &queue, tick);
            }
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            gpu.read_snapshot_transforms(&device, &queue, 1).unwrap()[child as usize]
        };
        let baseline = run(None);
        let struck = run(Some(Vec3::NEG_Y * 500.0));
        let baseline_rotation = bevy_math::Quat::from_array(baseline.rotation);
        let struck_rotation = bevy_math::Quat::from_array(struck.rotation);
        assert!(baseline_rotation.angle_between(struck_rotation) > 0.01);
    }

    fn run_ticks(
        creation: &mechanic_core::CompiledCreation,
        ticks: u64,
        collisions_enabled: bool,
    ) -> Option<(Vec<crate::GpuTransform>, crate::GpuTickReadback)> {
        let (device, queue) = test_device()?;
        let gpu = GpuPhysics::new_with_config(
            &device,
            &queue,
            creation,
            GpuPhysicsConfig {
                collisions_enabled,
                mechanism_self_collisions: true,
                solver_iterations: 16,
            },
        )
        .ok()?;
        for tick in 1..=ticks {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let readback = gpu.read_last_tick(&device).ok()?;
        let snapshot = gpu
            .read_snapshot_transforms(&device, &queue, (ticks % 3) as u8)
            .ok()?;
        Some((snapshot, readback))
    }

    #[test]
    fn grounded_offset_pendulum_swings_without_detaching() {
        let creation = pendulum_creation(true);
        let Some((snapshot, readback)) = run_ticks(&creation, 30, false) else {
            return;
        };
        assert_eq!(readback.error_flags, 0);
        assert!(snapshot[1].rotation[0].abs() > 1.0e-3);
        let root = Vec3::from_array(snapshot[0].position[..3].try_into().unwrap());
        assert!(root.abs_diff_eq(creation.compounds[0].root_translation, 1.0e-5));
    }

    #[test]
    fn freely_falling_hinge_has_no_gravity_induced_relative_rotation() {
        let creation = pendulum_creation(false);
        let Some((snapshot, readback)) = run_ticks(&creation, 30, false) else {
            return;
        };
        assert_eq!(readback.error_flags, 0);
        assert!(snapshot[0].position[1] < creation.compounds[0].root_translation.y - 0.5);
        assert!(snapshot[1].rotation[0].abs() < 1.0e-4);
        assert!(snapshot[1].rotation[1].abs() < 1.0e-4);
        assert!(snapshot[1].rotation[2].abs() < 1.0e-4);
    }

    fn cylinder_drop_creation(
        inner_diameter: f32,
        drop_x_half_units: i32,
    ) -> mechanic_core::CompiledCreation {
        let mut graph = ConstructionGraph::new();
        let support_spec = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(8, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(support) =
            graph.apply(BuildCommand::Spawn(support_spec)).unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(support, FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();
        let cylinder_spec = CylinderSpec::new(
            CylinderDimensions::new(1.0, inner_diameter, 1.0).unwrap(),
            BuildPose::new(IVec3::new(0, 12, 0), GridRotation::default()),
        );
        let BuildOutcome::Spawned(cylinder) = graph
            .apply(BuildCommand::SpawnCylinder(cylinder_spec))
            .unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: support,
                second: cylinder,
            }))
            .unwrap();
        let drop_spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(
                IVec3::new(drop_x_half_units, 40, 0),
                GridRotation::default(),
            ),
        )
        .unwrap();
        graph.apply(BuildCommand::Spawn(drop_spec)).unwrap();
        graph.compile().unwrap()
    }

    #[test]
    fn gpu_cylinder_bore_is_passable_and_annular_material_blocks_motion() {
        let hollow = cylinder_drop_creation(0.60, 0);
        let solid = cylinder_drop_creation(0.0, 0);
        let annular = cylinder_drop_creation(0.60, 3);
        let Some((hollow_snapshot, hollow_readback)) = run_ticks(&hollow, 60, true) else {
            return;
        };
        let Some((solid_snapshot, solid_readback)) = run_ticks(&solid, 60, true) else {
            return;
        };
        let Some((annular_snapshot, annular_readback)) = run_ticks(&annular, 60, true) else {
            return;
        };
        assert_eq!(hollow_readback.error_flags, 0);
        assert_eq!(solid_readback.error_flags, 0);
        assert_eq!(annular_readback.error_flags, 0);
        let falling_body = 1;
        assert!(hollow_snapshot[falling_body].position[1] < 2.5);
        assert!(solid_snapshot[falling_body].position[1] > 3.4);
        assert!(
            annular_snapshot[falling_body].position[1] > 3.4,
            "annular drop reached y={} instead of resting on the ring",
            annular_snapshot[falling_body].position[1]
        );
    }

    #[test]
    fn contact_supported_unwelded_tower_drives_welded_arm() {
        let mut graph = ConstructionGraph::new();
        let mut parts = Vec::new();
        for y in [2, 6, 10, 14] {
            let spec = CuboidSpec::new(
                [4, 4, 4],
                BuildPose::new(IVec3::new(0, y, 0), GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            parts.push(part);
        }
        for z in [0, 4, 8] {
            let spec = CuboidSpec::new(
                [4, 4, 4],
                BuildPose::new(IVec3::new(4, 14, z), GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            parts.push(part);
        }
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(parts[4], FaceKind::PositiveZ),
                second: FaceRef::part(parts[5], FaceKind::NegativeZ),
            }))
            .unwrap();
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(parts[5], FaceKind::PositiveZ),
                second: FaceRef::part(parts[6], FaceKind::NegativeZ),
            }))
            .unwrap();
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(parts[3], FaceKind::PositiveX),
                FaceRef::part(parts[4], FaceKind::NegativeX),
                Vec3::new(0.5, 3.5, 0.0),
                Vec3::X,
            )))
            .unwrap();
        let creation = graph.compile().unwrap();
        let Some((snapshot, readback)) = run_ticks(&creation, 90, true) else {
            return;
        };
        assert_eq!(readback.error_flags, 0);
        let bearing = creation.bearings[0];
        let body_a = bearing.compound_a as usize;
        let body_b = bearing.compound_b as usize;
        let rotation_a = bevy_math::Quat::from_array(snapshot[body_a].rotation);
        let rotation_b = bevy_math::Quat::from_array(snapshot[body_b].rotation);
        let anchor_a = Vec3::from_array(snapshot[body_a].position[..3].try_into().unwrap())
            + rotation_a * bearing.local_anchor_a;
        let anchor_b = Vec3::from_array(snapshot[body_b].position[..3].try_into().unwrap())
            + rotation_b * bearing.local_anchor_b;
        assert!(anchor_a.abs_diff_eq(anchor_b, 1.0e-5));
        assert!(snapshot[body_b].rotation[0].abs() > 1.0e-3);
        assert!(
            snapshot[body_a].position[1] > 0.5,
            "contact-supported body fell to {} m",
            snapshot[body_a].position[1]
        );
    }

    #[test]
    fn double_pendulum_transfers_motion_through_both_bearings() {
        let mut graph = ConstructionGraph::new();
        let mut spawned = Vec::new();
        for units in [
            IVec3::new(0, 2, 0),
            IVec3::new(4, 2, 0),
            IVec3::new(4, 2, 4),
            IVec3::new(8, 2, 0),
            IVec3::new(8, 2, 4),
        ] {
            let spec =
                CuboidSpec::new([4, 4, 4], BuildPose::new(units, GridRotation::default())).unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            spawned.push(part);
        }
        for (a, b) in [(1, 2), (3, 4)] {
            graph
                .apply(BuildCommand::Weld(WeldSpec {
                    first: FaceRef::part(spawned[a], FaceKind::PositiveZ),
                    second: FaceRef::part(spawned[b], FaceKind::NegativeZ),
                }))
                .unwrap();
        }
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(spawned[0], FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();
        for (a, b, x) in [(0, 1, 0.5), (1, 3, 1.5)] {
            graph
                .apply(BuildCommand::AddBearing(BearingSpec::new(
                    FaceRef::part(spawned[a], FaceKind::PositiveX),
                    FaceRef::part(spawned[b], FaceKind::NegativeX),
                    Vec3::new(x, 0.5, 0.0),
                    Vec3::X,
                )))
                .unwrap();
        }
        let creation = graph.compile().unwrap();
        let Some((snapshot, readback)) = run_ticks(&creation, 30, false) else {
            return;
        };
        assert_eq!(readback.error_flags, 0);
        let first = bevy_math::Quat::from_array(snapshot[1].rotation);
        let second = bevy_math::Quat::from_array(snapshot[2].rotation);
        assert!(first.x.abs() > 1.0e-3);
        assert!((first.conjugate() * second).x.abs() > 1.0e-4);
    }

    #[test]
    fn child_contact_changes_floating_root_and_joint_motion() {
        let mut graph = ConstructionGraph::new();
        let mut spawned = Vec::new();
        for units in [
            IVec3::new(0, 6, 0),
            IVec3::new(4, 6, 0),
            IVec3::new(4, 2, 0),
            IVec3::new(4, 2, 4),
        ] {
            let spec =
                CuboidSpec::new([4, 4, 4], BuildPose::new(units, GridRotation::default())).unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            spawned.push(part);
        }
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(spawned[1], FaceKind::NegativeY),
                second: FaceRef::part(spawned[2], FaceKind::PositiveY),
            }))
            .unwrap();
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(spawned[2], FaceKind::PositiveZ),
                second: FaceRef::part(spawned[3], FaceKind::NegativeZ),
            }))
            .unwrap();
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(spawned[0], FaceKind::PositiveX),
                FaceRef::part(spawned[1], FaceKind::NegativeX),
                Vec3::new(0.5, 1.5, 0.0),
                Vec3::X,
            )))
            .unwrap();
        let creation = graph.compile().unwrap();
        let Some((free, _)) = run_ticks(&creation, 8, false) else {
            return;
        };
        let Some((contact, readback)) = run_ticks(&creation, 8, true) else {
            return;
        };
        assert_eq!(readback.error_flags, 0);
        let root = creation.bearings[0].compound_a as usize;
        let child = creation.bearings[0].compound_b as usize;
        assert!(contact[root].position[1] > free[root].position[1] + 1.0e-4);
        let root_rotation = bevy_math::Quat::from_array(contact[root].rotation);
        let child_rotation = bevy_math::Quat::from_array(contact[child].rotation);
        assert!((root_rotation.conjugate() * child_rotation).x.abs() > 1.0e-4);
    }
}
