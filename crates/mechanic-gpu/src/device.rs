use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::mpsc;

use bevy_math::Vec3;
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use mechanic_core::{ColliderShape, CompiledCreation, ConstructionMaterial};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::{
    BROADPHASE_HASH_CAPACITY, COLLIDER_SHAPE_CONVEX, COLLIDER_SHAPE_CUBOID, FIXED_DT_SECONDS,
    GpuBearing, GpuCollider, GpuContact, GpuContractionNode, GpuDiagnostics, GpuGroundSurface,
    GpuLinkState, GpuMass, GpuMechanismBody, GpuMechanismCoordinate, GpuMechanismDrive, GpuPair,
    GpuPersistentManifold, GpuSpatialInertia, GpuTickConfig, GpuTransform, GpuVelocity,
    MAX_BEARINGS, MAX_BODIES, MAX_COLLIDERS, MAX_CONTACT_PAIRS, MAX_CONVEX_SHAPE_SLOTS,
    SNAPSHOT_RING_SIZE, pack_convex_counts,
};

const SERIAL_MECHANISM_BEARING_LIMIT: u32 = 64;
const SERIAL_MECHANISM_SOLVER_MULTIPLIER: u32 = 12;

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
    /// Fixed number of projected impulse iterations.
    pub solver_iterations: u32,
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
    ground_surface: wgpu::Buffer,
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
    solve_accumulate_serial_pipeline: wgpu::ComputePipeline,
    solve_accumulate_serial_bind_group: wgpu::BindGroup,
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
    drives: wgpu::Buffer,
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
    project_velocity_serial_pipeline: wgpu::ComputePipeline,
    project_velocity_serial_bind_group: wgpu::BindGroup,
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
    has_dynamic_root: bool,
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
                let (local_rotation, half_extents, shape) = match &collider.shape {
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
        let convex_shapes =
            create_readonly_storage_buffer(device, "mechanic convex shapes", &convex_shapes);
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

    /// Changes the explicit flat collision plane used by garage and benchmark scenes.
    pub fn write_ground_plane(&self, queue: &wgpu::Queue, normal: Vec3, offset: f32) {
        let normal = normal.normalize_or_zero();
        let concrete = ConstructionMaterial::Concrete.properties();
        queue.write_buffer(
            &self.collision.ground_surface,
            0,
            bytes_of(&GpuGroundSurface {
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
            }),
        );
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
            pair_capacity: self.pair_capacity,
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
        if serial && self.bearing_count <= SERIAL_MECHANISM_BEARING_LIMIT {
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
        let serial_mechanism = self.mechanism.active
            && self.mechanism.has_dynamic_root
            && self.bearing_count <= SERIAL_MECHANISM_BEARING_LIMIT;
        if self.mechanism.active && self.mechanism.has_dynamic_root {
            self.encode_contact_bearing_velocity_projection_iteration(encoder, serial_mechanism);
        }
        let iterations = if serial_mechanism {
            self.pipeline_config.solver_iterations.max(1) * SERIAL_MECHANISM_SOLVER_MULTIPLIER
        } else {
            self.pipeline_config.solver_iterations.max(1)
        };
        for _ in 1..iterations {
            if serial_mechanism {
                direct_compute_pass(
                    encoder,
                    "mechanic contact projection serially",
                    &collision.solve_accumulate_serial_pipeline,
                    &collision.solve_accumulate_serial_bind_group,
                    1,
                    None,
                );
            } else {
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
            if self.mechanism.active && self.mechanism.has_dynamic_root {
                self.encode_contact_bearing_velocity_projection_iteration(
                    encoder,
                    serial_mechanism,
                );
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
    let project_velocity_serial_pipeline = compute_pipeline(
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
            entry(2, rotations),
            entry(3, linear_velocities),
            entry(4, angular_velocities),
            entry(5, masses),
            entry(6, bearings),
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
            entry(12, &drives),
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
        drives,
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
        project_velocity_serial_pipeline,
        project_velocity_serial_bind_group,
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
        has_dynamic_root,
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
            entry(28, convex_shapes),
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
        wgpu::BufferUsages::STORAGE,
    );
    let persistent_manifolds = create_sized_buffer(
        device,
        "mechanic persistent manifolds",
        pair_capacity * size_of::<GpuPersistentManifold>(),
        wgpu::BufferUsages::STORAGE,
    );
    let concrete = ConstructionMaterial::Concrete.properties();
    let ground_surface = create_uniform_buffer(
        device,
        "mechanic ground surface",
        &GpuGroundSurface {
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
        },
    );
    let active_contacts = create_sized_buffer(
        device,
        "mechanic active contact indices",
        pair_capacity * size_of::<u32>(),
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
        convex_shapes,
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
        entry(28, convex_shapes),
        entry(29, &ground_surface),
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
            entry(28, convex_shapes),
            entry(29, &ground_surface),
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
            entry(15, &persistent_manifolds),
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
            entry(15, &persistent_manifolds),
            entry(26, &world_masses),
            entry(25, angular_velocities),
        ],
    );
    let solve_accumulate_serial_pipeline = compute_pipeline(
        device,
        "mechanic serial contact projection",
        &shader,
        "solve_accumulate_serial",
    );
    let solve_accumulate_serial_bind_group = bind_group(
        device,
        "mechanic serial contact projection bindings",
        &solve_accumulate_serial_pipeline,
        &[
            entry(0, config),
            entry(3, linear_velocities),
            entry(5, diagnostics),
            entry(10, &contacts),
            entry(16, &active_contacts),
            entry(15, &persistent_manifolds),
            entry(25, angular_velocities),
            entry(26, &world_masses),
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
        ground_surface,
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
        solve_accumulate_serial_pipeline,
        solve_accumulate_serial_bind_group,
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
        BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
        ConstructionMaterial, CuboidSpec, CylinderDimensions, CylinderSpec, FaceKind, FaceRef,
        GridRotation, PartId, PipeBendDimensions, PipeBendSpec, RigidLinkSpec, WeldSpec,
    };

    use crate::GpuMechanismCoordinate;

    use super::{
        FULL_CYLINDER_GROUND_FIRST, GpuPhysics, GpuPhysicsConfig, contact_pair_capacity,
        full_cylinder_ground_data,
    };

    #[test]
    fn collision_buffers_scale_with_the_scene_up_to_the_hard_limit() {
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
                late.iter().all(|speed| *speed < 1.0e-4)
                    && late
                        .iter()
                        .zip(&early)
                        .all(|(late, early)| *late <= *early + 1.0e-5),
                "{arm_count}-arm hanging={hanging} bearing speeds grew from {early:?} to {late:?}"
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
        assert_eq!(diagnostics.error_flags, 0, "{diagnostics:?}");
        assert!(
            late_speeds.iter().all(|speed| *speed < 0.02)
                && late_speeds
                    .iter()
                    .zip(&early_speeds)
                    .all(|(late, early)| *late <= *early + 1.0e-5),
            "bearing state gained speed from {early_speeds:?} to {late_speeds:?}"
        );
    }

    /// Grounded base plus a hinged arm wired to one control block.
    ///
    /// `loaded` extends the arm sideways off the hinge axis so gravity applies a
    /// real torque; otherwise the arm's centre of mass sits on the axis.
    fn driven_arm(
        axis: Vec3,
        limits: mechanic_core::DriveLimits,
        program: mechanic_core::DriveProgram,
        loaded: bool,
    ) -> (ConstructionGraph, mechanic_core::CompiledCreation) {
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
        let (base_units, arm_units, source_face, target_face, anchor) = if axis == Vec3::X {
            (
                IVec3::new(0, 2, 0),
                IVec3::new(4, 2, 0),
                FaceKind::PositiveX,
                FaceKind::NegativeX,
                Vec3::new(0.5, 0.5, 0.0),
            )
        } else {
            (
                IVec3::new(0, 2, 0),
                IVec3::new(0, 6, 0),
                FaceKind::PositiveY,
                FaceKind::NegativeY,
                Vec3::new(0.0, 1.0, 0.0),
            )
        };
        let base = spawn(base_units);
        let arm = spawn(arm_units);
        let outrigger = loaded.then(|| spawn(arm_units + IVec3::new(0, 0, 4)));
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(base, FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();
        if let Some(outrigger) = outrigger {
            graph
                .apply(BuildCommand::Weld(WeldSpec {
                    first: FaceRef::part(arm, FaceKind::PositiveZ),
                    second: FaceRef::part(outrigger, FaceKind::NegativeZ),
                }))
                .unwrap();
        }
        let BuildOutcome::BearingAdded(bearing) = graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(base, source_face),
                FaceRef::part(arm, target_face),
                anchor,
                axis,
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(
                mechanic_core::ControllerSpec::new(BuildPose::new(
                    IVec3::new(0, 40, 0),
                    GridRotation::default(),
                )),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let mut link = mechanic_core::DriveLinkSpec::new(controller, bearing);
        link.limits = limits;
        link.program = program;
        graph.apply(BuildCommand::AddDriveLink(link)).unwrap();
        let creation = graph.compile().unwrap();
        (graph, creation)
    }

    /// Signed joint angle of the first bearing, read back from a snapshot.
    fn joint_angle(
        snapshot: &[crate::GpuTransform],
        creation: &mechanic_core::CompiledCreation,
    ) -> f32 {
        let bearing = &creation.bearings[0];
        let delta = relative_bearing_rotation(snapshot, bearing);
        let axis = bearing.local_axis_a.normalize();
        2.0 * delta.xyz().dot(axis).atan2(delta.w)
    }

    fn run_driven_arm(
        creation: &mechanic_core::CompiledCreation,
        ticks: u64,
    ) -> Option<(f32, super::GpuTickReadback)> {
        let (device, queue) = test_device()?;
        let gpu = GpuPhysics::new_with_config(
            &device,
            &queue,
            creation,
            GpuPhysicsConfig {
                collisions_enabled: false,
                ..Default::default()
            },
        )
        .unwrap();
        for tick in 1..=ticks {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let snapshot = gpu
            .read_snapshot_transforms(&device, &queue, u8::try_from(ticks % 3).unwrap())
            .unwrap();
        let diagnostics = gpu.read_last_tick(&device).unwrap();
        Some((joint_angle(&snapshot, creation), diagnostics))
    }

    /// Limits with the given maximum torque and no travel stops.
    fn limits(max_speed: f32, torque: f32) -> mechanic_core::DriveLimits {
        mechanic_core::DriveLimits::new(max_speed, torque, None).expect("test limits are in range")
    }

    /// Single-state program holding one target forever.
    fn holding(target: mechanic_core::DriveTarget) -> mechanic_core::DriveProgram {
        mechanic_core::DriveProgram::new(
            &[mechanic_core::DriveState::new(target).expect("test target is in range")],
            false,
        )
        .expect("a one-state program is valid")
    }

    #[test]
    fn speed_state_advances_a_bearing_coordinate_at_its_target_speed() {
        let (_, creation) = driven_arm(
            Vec3::X,
            limits(3.0, f32::INFINITY),
            holding(mechanic_core::DriveTarget::Speed(1.0)),
            false,
        );
        let Some((angle, diagnostics)) = run_driven_arm(&creation, 60) else {
            return;
        };

        assert_eq!(diagnostics.error_flags, 0);
        assert!(
            (angle - 1.0).abs() < 0.05,
            "one second at 1 rad/s should reach about 1 rad, got {angle}"
        );
    }

    #[test]
    fn negative_target_speed_drives_the_joint_the_other_way() {
        let (_, creation) = driven_arm(
            Vec3::X,
            limits(3.0, f32::INFINITY),
            holding(mechanic_core::DriveTarget::Speed(-1.0)),
            false,
        );
        let Some((angle, diagnostics)) = run_driven_arm(&creation, 60) else {
            return;
        };

        assert_eq!(diagnostics.error_flags, 0);
        assert!(
            (angle + 1.0).abs() < 0.05,
            "a negative target speed should reach about -1 rad, got {angle}"
        );
    }

    #[test]
    fn max_speed_caps_a_faster_state_target() {
        let (_, creation) = driven_arm(
            Vec3::X,
            limits(0.5, f32::INFINITY),
            holding(mechanic_core::DriveTarget::Speed(3.0)),
            false,
        );
        let Some((angle, diagnostics)) = run_driven_arm(&creation, 60) else {
            return;
        };

        assert_eq!(diagnostics.error_flags, 0);
        assert!(
            (angle - 0.5).abs() < 0.05,
            "the row's 0.5 rad/s ceiling should hold the joint to 0.5 rad, got {angle}"
        );
    }

    #[test]
    fn angle_state_reaches_its_target_and_holds_without_overshooting() {
        let (_, creation) = driven_arm(
            Vec3::X,
            limits(3.0, 400.0),
            holding(mechanic_core::DriveTarget::Angle(0.8)),
            false,
        );
        // Long enough that a servo which overshoots would be caught swinging
        // back through the target rather than sitting on it.
        let Some((angle, diagnostics)) = run_driven_arm(&creation, 240) else {
            return;
        };

        assert_eq!(diagnostics.error_flags, 0);
        assert!(
            (angle - 0.8).abs() < 0.02,
            "the joint should settle on its 0.8 rad target, got {angle}"
        );
    }

    #[test]
    fn weak_drive_stalls_lifting_a_gravity_loaded_arm() {
        // The outrigger hangs off the hinge axis, so gravity applies a positive
        // torque about it. Driving negative means the motor must lift that load,
        // which is the only direction in which stalling is observable.
        let program = holding(mechanic_core::DriveTarget::Speed(-1.0));
        let (_, strong_creation) = driven_arm(Vec3::X, limits(3.0, f32::INFINITY), program, true);
        let (_, weak_creation) = driven_arm(Vec3::X, limits(3.0, 0.5), program, true);

        let Some((strong_angle, strong_diagnostics)) = run_driven_arm(&strong_creation, 60) else {
            return;
        };
        let (weak_angle, weak_diagnostics) = run_driven_arm(&weak_creation, 60).unwrap();

        assert_eq!(strong_diagnostics.error_flags, 0);
        assert_eq!(weak_diagnostics.error_flags, 0);
        assert!(
            (strong_angle + 1.0).abs() < 0.05,
            "an unlimited motor holds its target under load, got {strong_angle}"
        );
        assert!(
            weak_angle > strong_angle + 0.5,
            "a 0.5 N·m motor should stall well short of {strong_angle}, got {weak_angle}"
        );
    }

    #[test]
    fn driven_coordinate_stops_and_holds_at_its_travel_limit() {
        let stopped = mechanic_core::DriveLimits::new(3.0, f32::INFINITY, Some((-0.2, 0.2)))
            .expect("test limits are in range");
        let (_, creation) = driven_arm(
            Vec3::X,
            stopped,
            holding(mechanic_core::DriveTarget::Speed(1.0)),
            false,
        );
        let Some((angle, diagnostics)) = run_driven_arm(&creation, 120) else {
            return;
        };

        assert_eq!(diagnostics.error_flags, 0);
        assert!(
            (angle - 0.2).abs() < 0.02,
            "the joint should hold at its 0.2 rad limit, got {angle}"
        );
    }

    #[test]
    fn vertical_axis_drive_is_not_zeroed_by_the_sleep_clamp() {
        // Below GRAVITY_ALIGNED_BEARING_SLEEP_SPEED, which stops passive joints.
        let (_, creation) = driven_arm(
            Vec3::Y,
            limits(3.0, f32::INFINITY),
            holding(mechanic_core::DriveTarget::Speed(0.002)),
            false,
        );
        let Some((angle, diagnostics)) = run_driven_arm(&creation, 600) else {
            return;
        };

        assert_eq!(diagnostics.error_flags, 0);
        assert!(
            angle > 0.015,
            "ten seconds at 0.002 rad/s should still turn the joint, got {angle}"
        );
    }

    #[test]
    fn reprogramming_a_wire_changes_the_drive_without_reloading_the_scene() {
        let (mut graph, creation) = driven_arm(
            Vec3::X,
            limits(3.0, f32::INFINITY),
            holding(mechanic_core::DriveTarget::Speed(1.0)),
            false,
        );
        let Some((device, queue)) = test_device() else {
            return;
        };
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

        assert_eq!(
            gpu.write_mechanism_drives(&queue, &[]),
            Err(super::GpuPhysicsError::DriveStateCount {
                provided: 0,
                required: 1,
            })
        );

        let (link, drive_spec) = graph
            .drive_links()
            .map(|(id, spec)| (id, *spec))
            .next()
            .unwrap();
        graph
            .apply(BuildCommand::SetDriveLink {
                link,
                limits: limits(3.0, f32::INFINITY),
                program: holding(mechanic_core::DriveTarget::Speed(-2.0)),
                name: mechanic_core::DriveName::EMPTY,
                actuator: drive_spec.actuator,
            })
            .unwrap();
        let rows = creation
            .resolve_coordinate_drives(&graph)
            .into_iter()
            .map(crate::GpuMechanismDrive::from)
            .collect::<Vec<_>>();
        gpu.write_mechanism_drives(&queue, &rows).unwrap();

        for tick in 1..=60 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let snapshot = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
        assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
        let angle = joint_angle(&snapshot, &creation);
        assert!(
            (angle + 2.0).abs() < 0.1,
            "the live reprogram should drive -2 rad/s, got {angle}"
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

    struct ArticulatedCarFixture {
        creation: mechanic_core::CompiledCreation,
        chassis: u32,
        dynamic_bodies: Vec<u32>,
        wheel_bodies: Vec<u32>,
    }

    fn spawned_part(outcome: BuildOutcome) -> PartId {
        let BuildOutcome::Spawned(part) = outcome else {
            unreachable!()
        };
        part
    }

    fn articulated_car_fixture() -> ArticulatedCarFixture {
        let mut graph = ConstructionGraph::new();
        let chassis_spec = CuboidSpec::new(
            [8, 2, 12],
            BuildPose::new(IVec3::new(0, 5, 0), GridRotation::default()),
        )
        .unwrap();
        let chassis_part = spawned_part(graph.apply(BuildCommand::Spawn(chassis_spec)).unwrap());

        let mut knuckles = Vec::new();
        let mut wheels = Vec::new();
        for z_units in [-4, 4] {
            let anchor_z = if z_units < 0 { -1.0 } else { 1.0 };
            for x_units in [-3, 3] {
                let steering_anchor_x = if x_units < 0 { -0.75 } else { 0.75 };
                let knuckle_spec = CuboidSpec::new(
                    [2, 2, 2],
                    BuildPose::new(IVec3::new(x_units, 3, z_units), GridRotation::default()),
                )
                .unwrap();
                let knuckle = spawned_part(graph.apply(BuildCommand::Spawn(knuckle_spec)).unwrap());
                knuckles.push(knuckle);

                let wheel_spec = CylinderSpec::new(
                    CylinderDimensions::new(1.0, 0.0, 0.5).unwrap(),
                    BuildPose::new(
                        IVec3::new(if x_units < 0 { -5 } else { 5 }, 3, z_units),
                        GridRotation::new(0, 0, 1),
                    ),
                );
                let wheel = spawned_part(
                    graph
                        .apply(BuildCommand::SpawnCylinder(wheel_spec))
                        .unwrap(),
                );
                wheels.push(wheel);

                graph
                    .apply(BuildCommand::AddBearing(BearingSpec::new(
                        FaceRef::part(chassis_part, FaceKind::NegativeY),
                        FaceRef::part(knuckle, FaceKind::PositiveY),
                        Vec3::new(steering_anchor_x, 1.0, anchor_z),
                        Vec3::NEG_Y,
                    )))
                    .unwrap();
                let (source_face, target_face, axis, anchor_x) = if x_units < 0 {
                    (FaceKind::NegativeX, FaceKind::NegativeY, Vec3::NEG_X, -1.0)
                } else {
                    (FaceKind::PositiveX, FaceKind::PositiveY, Vec3::X, 1.0)
                };
                graph
                    .apply(BuildCommand::AddBearing(BearingSpec::new(
                        FaceRef::part(knuckle, source_face),
                        FaceRef::part(wheel, target_face),
                        Vec3::new(anchor_x, 0.75, anchor_z),
                        axis,
                    )))
                    .unwrap();
            }
        }

        let wall_spec = CuboidSpec::new(
            [16, 8, 2],
            BuildPose::new(IVec3::new(0, 4, -10), GridRotation::default()),
        )
        .unwrap();
        let wall = spawned_part(graph.apply(BuildCommand::Spawn(wall_spec)).unwrap());
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(wall, FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();

        let creation = graph.compile().unwrap();
        let body_for = |part| {
            creation
                .part_to_compound
                .iter()
                .find_map(|(candidate, body)| (*candidate == part).then_some(*body))
                .unwrap()
        };
        let chassis = body_for(chassis_part);
        let mut dynamic_bodies = vec![chassis];
        dynamic_bodies.extend(knuckles.iter().copied().map(body_for));
        let wheel_bodies = wheels.iter().copied().map(body_for).collect::<Vec<_>>();
        dynamic_bodies.extend(wheel_bodies.iter().copied());
        dynamic_bodies.sort_unstable();
        dynamic_bodies.dedup();
        ArticulatedCarFixture {
            creation,
            chassis,
            dynamic_bodies,
            wheel_bodies,
        }
    }

    #[test]
    fn full_cylinder_ground_contacts_match_visual_wheel_radius() {
        let fixture = articulated_car_fixture();
        let ground_data = full_cylinder_ground_data(&fixture.creation.colliders);
        let analytic = ground_data
            .iter()
            .filter(|ground| ground.role != 0)
            .collect::<Vec<_>>();
        let primary = ground_data
            .iter()
            .filter(|ground| ground.role == FULL_CYLINDER_GROUND_FIRST)
            .collect::<Vec<_>>();
        let secondary_count = ground_data
            .iter()
            .filter(|ground| ground.role > FULL_CYLINDER_GROUND_FIRST)
            .count();
        assert_eq!(primary.len(), 4);
        assert_eq!(analytic.len(), 64);
        assert_eq!(secondary_count, 60);
        assert!(analytic.iter().all(|ground| {
            (ground.center_radius - 0.25).abs() < 1.0e-6
                && (ground.outer_radius - 0.5).abs() < 1.0e-6
        }));
        for role in 1..=16 {
            assert_eq!(
                analytic.iter().filter(|ground| ground.role == role).count(),
                4
            );
        }

        let mut sector_graph = ConstructionGraph::new();
        let sector_dimensions = CylinderDimensions::new(1.0, 0.0, 0.5)
            .unwrap()
            .with_sweep_angle_degrees(180)
            .unwrap();
        sector_graph
            .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
                sector_dimensions,
                BuildPose::default(),
            )))
            .unwrap();
        let sector = sector_graph.compile().unwrap();
        assert!(
            full_cylinder_ground_data(&sector.colliders)
                .iter()
                .all(|ground| ground.role == 0)
        );
    }

    fn transform_position(transform: crate::GpuTransform) -> Vec3 {
        Vec3::new(
            transform.position[0],
            transform.position[1],
            transform.position[2],
        )
    }

    fn snapshot_speed(current: crate::GpuTransform, previous: crate::GpuTransform) -> f32 {
        (transform_position(current) - transform_position(previous)).length() * 60.0
    }

    fn bearing_speed(
        current: &[crate::GpuTransform],
        previous: &[crate::GpuTransform],
        bearing: &mechanic_core::CompiledBearing,
    ) -> f32 {
        let current_relative = relative_bearing_rotation(current, bearing);
        let previous_relative = relative_bearing_rotation(previous, bearing);
        let delta = previous_relative.conjugate() * current_relative;
        2.0 * delta.xyz().length().atan2(delta.w.abs()) * 60.0
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
                    ground_plane_enabled: true,
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
    fn offset_ground_contact_applies_angular_impulse_about_compound_centre() {
        let mut graph = ConstructionGraph::new();
        let lower_spec = CuboidSpec::new(
            [2, 2, 2],
            BuildPose::new(IVec3::new(4, 1, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(lower) = graph.apply(BuildCommand::Spawn(lower_spec)).unwrap()
        else {
            unreachable!()
        };
        let upper_spec = CuboidSpec::new(
            [2, 2, 2],
            BuildPose::new(IVec3::new(-4, 9, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(upper) = graph.apply(BuildCommand::Spawn(upper_spec)).unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: lower,
                second: upper,
            }))
            .unwrap();
        let creation = graph.compile().unwrap();
        assert!(creation.colliders[0].local_center.x.abs() > 0.5);
        let Some((device, queue)) = test_device() else {
            return;
        };
        let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
        for tick in 1..=4 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let snapshot = gpu.read_snapshot_transforms(&device, &queue, 1).unwrap();
        let rotation = bevy_math::Quat::from_array(snapshot[0].rotation);
        assert!(
            rotation.z > 1.0e-4,
            "offset support acted through the compound centre: {rotation:?}"
        );
    }

    #[test]
    fn articulated_car_drop_settles_without_drift_or_ground_penetration() {
        let fixture = articulated_car_fixture();
        let Some((device, queue)) = test_device() else {
            return;
        };
        let gpu = GpuPhysics::new_with_config(
            &device,
            &queue,
            &fixture.creation,
            GpuPhysicsConfig {
                mechanism_self_collisions: false,
                solver_iterations: 8,
                ..Default::default()
            },
        )
        .unwrap();
        let initial_chassis = fixture.creation.compounds[fixture.chassis as usize].root_translation;
        let mut previous_sample_tick = 0;
        let sample_ticks = (10..=300).step_by(10).chain((330..=1_200).step_by(30));
        for sample_tick in sample_ticks {
            for tick in previous_sample_tick + 1..=sample_tick {
                gpu.dispatch_tick(&device, &queue, tick);
            }
            previous_sample_tick = sample_tick;
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let snapshot = gpu
                .read_snapshot_transforms(&device, &queue, (sample_tick % 3) as u8)
                .unwrap();
            let chassis_rotation =
                bevy_math::Quat::from_array(snapshot[fixture.chassis as usize].rotation);
            assert!(
                transform_position(snapshot[fixture.chassis as usize]).y >= 0.74,
                "chassis entered the ground at tick {sample_tick}"
            );
            assert!(
                (chassis_rotation * Vec3::Y).dot(Vec3::Y) > 0.9,
                "unpowered chassis tipped at tick {sample_tick}"
            );
            for &wheel in &fixture.wheel_bodies {
                let wheel_height = transform_position(snapshot[wheel as usize]).y;
                assert!(
                    wheel_height >= 0.495,
                    "wheel {wheel} entered the ground at tick {sample_tick}: y={wheel_height}"
                );
            }
        }

        let current = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
        let previous = gpu.read_snapshot_transforms(&device, &queue, 2).unwrap();
        let diagnostics = gpu.read_last_tick(&device).unwrap();
        assert_eq!(diagnostics.error_flags, 0);
        assert!(diagnostics.anchor_residual_meters <= mechanic_core::ANCHOR_TOLERANCE_METERS);
        assert!(diagnostics.axis_residual_degrees <= mechanic_core::AXIS_TOLERANCE_DEGREES);
        let chassis_position = transform_position(current[fixture.chassis as usize]);
        let horizontal_drift = Vec3::new(
            chassis_position.x - initial_chassis.x,
            0.0,
            chassis_position.z - initial_chassis.z,
        )
        .length();
        assert!(
            horizontal_drift < 0.05,
            "unpowered chassis drifted {horizontal_drift} m"
        );
        let max_linear_speed = fixture
            .dynamic_bodies
            .iter()
            .map(|&body| snapshot_speed(current[body as usize], previous[body as usize]))
            .fold(0.0_f32, f32::max);
        let max_bearing_speed = fixture
            .creation
            .bearings
            .iter()
            .map(|bearing| bearing_speed(&current, &previous, bearing))
            .fold(0.0_f32, f32::max);
        assert!(
            max_linear_speed < 0.02,
            "linear speed was {max_linear_speed}"
        );
        assert!(
            max_bearing_speed < 0.02,
            "bearing angular speed was {max_bearing_speed}"
        );
    }

    #[test]
    fn struck_articulated_wheel_recovers_without_crossing_ground() {
        let fixture = articulated_car_fixture();
        let Some((device, queue)) = test_device() else {
            return;
        };
        let gpu = GpuPhysics::new_with_config(
            &device,
            &queue,
            &fixture.creation,
            GpuPhysicsConfig {
                mechanism_self_collisions: false,
                ..Default::default()
            },
        )
        .unwrap();
        for tick in 1..=300 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let settled = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
        let wheel = fixture.wheel_bodies[0];
        let wheel_position = transform_position(settled[wheel as usize]);
        let wheel_mass = fixture.creation.compounds[wheel as usize]
            .mass_properties
            .mass;
        gpu.apply_impulse(
            &device,
            &queue,
            wheel,
            wheel_position,
            Vec3::NEG_Y * wheel_mass * 3.0,
        )
        .unwrap();

        let mut minimum_height = f32::INFINITY;
        let mut previous_sample_tick = 300;
        let sample_ticks = (301..=360).chain((370..=900).step_by(10));
        for sample_tick in sample_ticks {
            for tick in previous_sample_tick + 1..=sample_tick {
                gpu.dispatch_tick(&device, &queue, tick);
            }
            previous_sample_tick = sample_tick;
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let snapshot = gpu
                .read_snapshot_transforms(&device, &queue, (sample_tick % 3) as u8)
                .unwrap();
            minimum_height = minimum_height.min(transform_position(snapshot[wheel as usize]).y);
        }
        let final_snapshot = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
        let final_height = transform_position(final_snapshot[wheel as usize]).y;
        let chassis_rotation =
            bevy_math::Quat::from_array(final_snapshot[fixture.chassis as usize].rotation);
        assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
        assert!(
            minimum_height >= 0.44,
            "struck wheel crossed too far into the ground: {minimum_height} m"
        );
        assert!(
            final_height >= 0.495,
            "struck wheel remained in the ground at {final_height} m"
        );
        assert!((chassis_rotation * Vec3::Y).dot(Vec3::Y) > 0.9);
    }

    #[test]
    fn articulated_car_wall_impact_remains_bounded_and_decays() {
        let fixture = articulated_car_fixture();
        let Some((device, queue)) = test_device() else {
            return;
        };
        let gpu = GpuPhysics::new_with_config(
            &device,
            &queue,
            &fixture.creation,
            GpuPhysicsConfig {
                mechanism_self_collisions: false,
                ..Default::default()
            },
        )
        .unwrap();
        let chassis_position =
            fixture.creation.compounds[fixture.chassis as usize].root_translation;
        let chassis_mass = fixture.creation.compounds[fixture.chassis as usize]
            .mass_properties
            .mass;
        gpu.apply_impulse(
            &device,
            &queue,
            fixture.chassis,
            chassis_position,
            Vec3::NEG_Z * chassis_mass * 0.35,
        )
        .unwrap();

        let mut maximum_sampled_speed = 0.0_f32;
        let mut maximum_sampled_bearing_speed = 0.0_f32;
        for sample_tick in (30..=1_200).step_by(30) {
            for tick in sample_tick - 29..=sample_tick {
                gpu.dispatch_tick(&device, &queue, tick);
            }
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let current = gpu
                .read_snapshot_transforms(&device, &queue, (sample_tick % 3) as u8)
                .unwrap();
            let previous = gpu
                .read_snapshot_transforms(&device, &queue, ((sample_tick - 1) % 3) as u8)
                .unwrap();
            for &body in &fixture.dynamic_bodies {
                let position = transform_position(current[body as usize]);
                assert!(position.is_finite(), "body {body} became non-finite");
                maximum_sampled_speed = maximum_sampled_speed.max(snapshot_speed(
                    current[body as usize],
                    previous[body as usize],
                ));
            }
            for bearing in &fixture.creation.bearings {
                maximum_sampled_bearing_speed =
                    maximum_sampled_bearing_speed.max(bearing_speed(&current, &previous, bearing));
            }
            assert!(
                transform_position(current[fixture.chassis as usize]).y >= 0.74,
                "chassis entered the ground at tick {sample_tick}"
            );
            for &wheel in &fixture.wheel_bodies {
                assert!(
                    transform_position(current[wheel as usize]).y >= 0.49,
                    "wheel {wheel} entered the ground at tick {sample_tick}"
                );
            }
        }

        let current = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
        let previous = gpu.read_snapshot_transforms(&device, &queue, 2).unwrap();
        let diagnostics = gpu.read_last_tick(&device).unwrap();
        assert_eq!(diagnostics.error_flags, 0);
        assert!(diagnostics.anchor_residual_meters <= mechanic_core::ANCHOR_TOLERANCE_METERS);
        assert!(diagnostics.axis_residual_degrees <= mechanic_core::AXIS_TOLERANCE_DEGREES);
        assert!(
            maximum_sampled_speed < 2.0,
            "wall impact accelerated the car to {maximum_sampled_speed} m/s"
        );
        assert!(
            maximum_sampled_bearing_speed < 10.0,
            "wall impact accelerated a bearing to {maximum_sampled_bearing_speed} rad/s"
        );
        let final_speed = fixture
            .dynamic_bodies
            .iter()
            .map(|&body| snapshot_speed(current[body as usize], previous[body as usize]))
            .fold(0.0_f32, f32::max);
        let final_bearing_speed = fixture
            .creation
            .bearings
            .iter()
            .map(|bearing| bearing_speed(&current, &previous, bearing))
            .fold(0.0_f32, f32::max);
        assert!(
            final_speed < 0.02,
            "wall-impact motion did not decay: final speed {final_speed} m/s"
        );
        assert!(
            final_bearing_speed < 0.02,
            "wall-impact bearing motion did not decay: final speed {final_bearing_speed} rad/s"
        );
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
                ground_plane_enabled: true,
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

    fn material_cube(
        material: ConstructionMaterial,
        center_half_units_y: i32,
    ) -> mechanic_core::CompiledCreation {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4; 3],
                    BuildPose::from_half_grid(
                        IVec3::new(0, center_half_units_y, 0),
                        GridRotation::default(),
                    ),
                )
                .unwrap()
                .with_material(material),
            ))
            .unwrap();
        graph.compile().unwrap()
    }

    /// A block whose top +z edge is collapsed onto the bottom, dropped from
    /// `center_half_units_y`.
    fn shaped_wedge(center_half_units_y: i32) -> mechanic_core::CompiledCreation {
        let spec = CuboidSpec::new(
            [4; 3],
            BuildPose::from_half_grid(
                IVec3::new(0, center_half_units_y, 0),
                GridRotation::default(),
            ),
        )
        .unwrap();
        let mut graph = ConstructionGraph::new();
        graph.apply(BuildCommand::Spawn(spec)).unwrap();
        let cells = mechanic_core::part_cells(spec);
        let region = mechanic_core::ShapeRegion::new(
            cells.corner_half_units(IVec3::ZERO, 0),
            cells.counts(),
            ConstructionMaterial::Steel,
        )
        .expect("the block spans at least one cell");
        let mechanic_core::BuildOutcome::RegionAdded(id) =
            graph.apply(BuildCommand::AddRegion(region)).unwrap()
        else {
            panic!("adding a region reports the region it added")
        };
        // Drop the top pair of cage corners on +z a whole cell onto the corners
        // below them, which slopes the whole top face.
        let cell = i16::try_from(mechanic_core::STEPS_PER_CELL).expect("a cell is twenty steps");
        graph
            .apply(BuildCommand::SetRegionVertices {
                region: id,
                vertices: vec![([0, 1, 1], [0, -cell, 0]), ([1, 1, 1], [0, -cell, 0])],
            })
            .expect("collapsing an edge is a legal shape");
        graph.compile().unwrap()
    }

    #[test]
    fn a_shaped_part_compiles_to_convex_collider_rows() {
        let creation = shaped_wedge(8);
        assert!(
            creation
                .colliders
                .iter()
                .any(|collider| !collider.shape.is_cuboid()),
            "shaping must produce convex collider rows, not boxes"
        );
        assert!(
            creation.colliders.len() < 64,
            "fusing should keep the row count small; got {}",
            creation.colliders.len()
        );
    }

    #[test]
    fn a_shaped_wedge_settles_on_the_ground_without_failing() {
        // The whole point of shaping being truthful: the solver has to accept
        // convex rows and bring them to rest like any other body.
        let Some((device, queue)) = test_device() else {
            return;
        };
        let creation = shaped_wedge(8);
        let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
        for tick in 1..=180 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(
            gpu.read_last_tick(&device).unwrap().error_flags,
            0,
            "convex colliders must not raise a failure flag"
        );
        let y = gpu
            .read_snapshot_transforms(&device, &queue, (180 % 3) as u8)
            .unwrap()[0]
            .position[1];
        assert!(
            y.is_finite() && y > -0.1,
            "the wedge should rest on the ground rather than sink; y={y}"
        );
        assert!(
            y < 1.0,
            "the wedge should have fallen from its 1 m drop; y={y}"
        );
    }

    #[test]
    fn higher_friction_material_loses_more_sliding_speed() {
        let Some((device, queue)) = test_device() else {
            return;
        };
        let slide = |material| {
            let creation = material_cube(material, 4);
            let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
            let mass = creation.compounds[0].mass_properties.mass;
            gpu.apply_impulse(&device, &queue, 0, Vec3::new(0.0, 0.5, 0.0), Vec3::X * mass)
                .unwrap();
            for tick in 1..=90 {
                gpu.dispatch_tick(&device, &queue, tick);
            }
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            gpu.read_snapshot_transforms(&device, &queue, 0).unwrap()[0].position[0]
        };
        let plastic_distance = slide(ConstructionMaterial::Plastic);
        let concrete_distance = slide(ConstructionMaterial::Concrete);
        assert!(
            concrete_distance < plastic_distance - 0.02,
            "concrete slid {concrete_distance} m while plastic slid {plastic_distance} m",
        );
    }

    #[test]
    fn static_friction_holds_a_sub_threshold_load_while_kinetic_friction_slows_sliding() {
        let Some((device, queue)) = test_device() else {
            return;
        };
        let creation = material_cube(ConstructionMaterial::Aluminium, 4);
        let mass = creation.compounds[0].mass_properties.mass;
        let contact_center = Vec3::new(0.0, 0.5, 0.0);
        let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
        for tick in 1..=60 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        let static_load_impulse = mass * 9.81 * crate::FIXED_DT_SECONDS * 0.62;
        for tick in 61..=120 {
            gpu.apply_impulse(
                &device,
                &queue,
                0,
                contact_center,
                Vec3::X * static_load_impulse,
            )
            .unwrap();
            gpu.dispatch_tick(&device, &queue, tick);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let stuck = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap()[0];
        assert!(
            stuck.position[0].abs() < 0.01,
            "sub-threshold static load moved the cube {} m",
            stuck.position[0],
        );

        let sliding = GpuPhysics::new(&device, &queue, &creation).unwrap();
        sliding
            .apply_impulse(&device, &queue, 0, contact_center, Vec3::X * mass * 3.0)
            .unwrap();
        for tick in 1..=2 {
            sliding.dispatch_tick(&device, &queue, tick);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let early = sliding
            .read_snapshot_transforms(&device, &queue, 2)
            .unwrap()[0];
        let initial = sliding
            .read_snapshot_transforms(&device, &queue, 1)
            .unwrap()[0];
        let early_speed = snapshot_speed(early, initial);
        for tick in 3..=20 {
            sliding.dispatch_tick(&device, &queue, tick);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let late = sliding
            .read_snapshot_transforms(&device, &queue, 2)
            .unwrap()[0];
        let previous = sliding
            .read_snapshot_transforms(&device, &queue, 1)
            .unwrap()[0];
        let late_speed = snapshot_speed(late, previous);
        assert!(
            late_speed < early_speed - 0.5,
            "sliding speed did not decay: {early_speed} to {late_speed}"
        );
        assert!(
            late_speed > 0.5,
            "kinetic friction used the static limit: final speed {late_speed}"
        );
    }

    #[test]
    fn higher_rolling_resistance_stops_an_otherwise_identical_cylinder_sooner() {
        let Some((device, queue)) = test_device() else {
            return;
        };
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::SpawnCylinder(
                CylinderSpec::new(
                    CylinderDimensions::new(1.0, 0.0, 1.0).unwrap(),
                    BuildPose::new(IVec3::new(0, 2, 0), GridRotation::new(0, 0, 1)),
                )
                .with_material(ConstructionMaterial::Steel),
            ))
            .unwrap();
        let base = graph.compile().unwrap();
        let roll = |rolling_resistance: f32| {
            let mut creation = base.clone();
            for collider in &mut creation.colliders {
                collider.material_properties.rolling_resistance = rolling_resistance;
            }
            let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
            let mass = creation.compounds[0].mass_properties.mass;
            gpu.apply_impulse(
                &device,
                &queue,
                0,
                Vec3::new(0.0, 0.75, 0.0),
                Vec3::Z * mass * 2.0,
            )
            .unwrap();
            for tick in 1..=120 {
                gpu.dispatch_tick(&device, &queue, tick);
            }
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
            gpu.read_snapshot_transforms(&device, &queue, 0).unwrap()[0].position[2]
        };
        let low_distance = roll(0.002);
        let high_distance = roll(0.040);
        assert!(
            high_distance < low_distance - 0.05,
            "high rolling resistance travelled {high_distance} m; low travelled {low_distance} m",
        );
    }

    #[test]
    fn flat_disc_stops_sliding_without_long_term_contact_drift() {
        let Some((device, queue)) = test_device() else {
            return;
        };
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
                CylinderDimensions::new(1.0, 0.0, 0.5).unwrap(),
                BuildPose::new(IVec3::new(0, 4, 0), GridRotation::default()),
            )))
            .unwrap();
        let creation = graph.compile().unwrap();
        let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
        let mass = creation.compounds[0].mass_properties.mass;
        gpu.apply_impulse(
            &device,
            &queue,
            0,
            Vec3::new(0.35, 0.65, 0.0),
            Vec3::new(1.0, -0.35, 0.2) * mass,
        )
        .unwrap();

        for tick in 1..=600 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let settled =
            transform_position(gpu.read_snapshot_transforms(&device, &queue, 0).unwrap()[0]);

        for tick in 601..=1_200 {
            gpu.dispatch_tick(&device, &queue, tick);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let later =
            transform_position(gpu.read_snapshot_transforms(&device, &queue, 0).unwrap()[0]);
        let tail_drift = Vec3::new(later.x - settled.x, 0.0, later.z - settled.z).length();
        assert!(
            tail_drift < 0.005,
            "flat disc drifted {tail_drift} m after it should have stopped"
        );
        assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
    }

    #[test]
    fn flat_disc_landing_on_another_disc_does_not_gain_energy() {
        let Some((device, queue)) = test_device() else {
            return;
        };
        let mut graph = ConstructionGraph::new();
        let dimensions = CylinderDimensions::new(1.0, 0.0, 0.25).unwrap();
        let BuildOutcome::Spawned(lower) = graph
            .apply(BuildCommand::SpawnCylinder(
                CylinderSpec::new(
                    dimensions,
                    BuildPose::new(IVec3::new(0, 1, 0), GridRotation::default()),
                )
                .with_material(ConstructionMaterial::Concrete),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(lower, FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();
        graph
            .apply(BuildCommand::SpawnCylinder(
                CylinderSpec::new(
                    dimensions,
                    BuildPose::new(IVec3::new(0, 4, 0), GridRotation::default()),
                )
                .with_material(ConstructionMaterial::Concrete),
            ))
            .unwrap();
        let creation = graph.compile().unwrap();
        let dynamic_body = creation
            .compounds
            .iter()
            .position(|compound| !compound.is_static)
            .unwrap();
        let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
        let mut maximum_height_after_impact = 0.0_f32;
        for tick in 1..=120 {
            gpu.dispatch_tick(&device, &queue, tick);
            if tick >= 25 {
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let position = transform_position(
                    gpu.read_snapshot_transforms(&device, &queue, (tick % 3) as u8)
                        .unwrap()[dynamic_body],
                );
                maximum_height_after_impact = maximum_height_after_impact.max(position.y);
                assert!(position.is_finite(), "stacked disc became non-finite");
                assert!(
                    position.x.abs() < 0.25 && position.z.abs() < 0.25,
                    "stacked disc was launched sideways to {position:?}"
                );
            }
        }
        assert!(
            maximum_height_after_impact < 0.9,
            "stacked disc rebounded above {maximum_height_after_impact} m"
        );
        assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
    }

    #[test]
    fn lower_modulus_allows_more_transient_penetration_without_failure() {
        let Some((device, queue)) = test_device() else {
            return;
        };
        let base = material_cube(ConstructionMaterial::Steel, 16);
        let minimum_height = |youngs_modulus_pa: f32| {
            let mut creation = base.clone();
            for collider in &mut creation.colliders {
                collider.material_properties.youngs_modulus_pa = youngs_modulus_pa;
            }
            let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
            let mut minimum = f32::INFINITY;
            for tick in 1..=60 {
                gpu.dispatch_tick(&device, &queue, tick);
                if tick >= 25 {
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    minimum = minimum.min(
                        gpu.read_snapshot_transforms(&device, &queue, (tick % 3) as u8)
                            .unwrap()[0]
                            .position[1],
                    );
                }
            }
            assert_eq!(gpu.read_last_tick(&device).unwrap().error_flags, 0);
            minimum
        };
        let stiff_height = minimum_height(200.0e9);
        let soft_height = minimum_height(0.01e9);
        assert!(
            soft_height < stiff_height - 1.0e-4,
            "soft contact reached {soft_height} m; stiff contact reached {stiff_height} m",
        );
        assert!(
            soft_height > 0.4,
            "soft contact became unstable at {soft_height} m"
        );
    }

    #[test]
    fn plastic_rebounds_more_than_concrete() {
        let Some((device, queue)) = test_device() else {
            return;
        };
        let rebound_height = |material| {
            let creation = material_cube(material, 16);
            let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
            let mut maximum_after_impact = 0.5_f32;
            for tick in 1..=100 {
                gpu.dispatch_tick(&device, &queue, tick);
                if tick >= 40 {
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    let y = gpu
                        .read_snapshot_transforms(&device, &queue, (tick % 3) as u8)
                        .unwrap()[0]
                        .position[1];
                    maximum_after_impact = maximum_after_impact.max(y);
                }
            }
            maximum_after_impact
        };
        let plastic_height = rebound_height(ConstructionMaterial::Plastic);
        let concrete_height = rebound_height(ConstructionMaterial::Concrete);
        assert!(
            plastic_height > concrete_height + 0.05,
            "plastic rebounded to {plastic_height} m while concrete reached {concrete_height} m",
        );
    }

    #[test]
    fn sub_threshold_ground_contact_settles_without_repeated_bounce() {
        let Some((device, queue)) = test_device() else {
            return;
        };
        let creation = material_cube(ConstructionMaterial::Plastic, 4);
        let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
        let mass = creation.compounds[0].mass_properties.mass;
        gpu.apply_impulse(
            &device,
            &queue,
            0,
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::NEG_Y * mass * 0.5,
        )
        .unwrap();
        let mut tail = Vec::new();
        for tick in 1..=120 {
            gpu.dispatch_tick(&device, &queue, tick);
            if tick >= 100 {
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                tail.push(
                    gpu.read_snapshot_transforms(&device, &queue, (tick % 3) as u8)
                        .unwrap()[0]
                        .position[1],
                );
            }
        }
        let movement = tail
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            movement < 1.0e-3,
            "settled contact moved {movement} m per tick"
        );
    }

    #[test]
    fn flat_cylinder_face_stays_supported_above_ground() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
                CylinderDimensions::new(1.0, 0.0, 0.5).unwrap(),
                BuildPose::new(IVec3::new(0, 4, 0), GridRotation::default()),
            )))
            .unwrap();
        let creation = graph.compile().unwrap();
        let Some((snapshot, readback)) = run_ticks(&creation, 120, true) else {
            return;
        };
        assert_eq!(readback.error_flags, 0);
        let position = transform_position(snapshot[0]);
        let rotation = bevy_math::Quat::from_array(snapshot[0].rotation);
        assert!(
            position.y >= 0.245,
            "flat cylinder sank through its 0.25 m half-length to {} m",
            position.y
        );
        assert!((rotation * Vec3::Y).dot(Vec3::Y) > 0.98);
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
            annular_snapshot[falling_body].position[1] > 3.35,
            "annular drop reached y={} instead of resting on the ring",
            annular_snapshot[falling_body].position[1]
        );
    }

    fn pipe_bend_drop_creation(
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
        let bend_spec = PipeBendSpec::new(
            PipeBendDimensions::new(1.0, inner_diameter, 1.0).unwrap(),
            BuildPose::from_half_grid(IVec3::new(0, 16, 0), GridRotation::default()),
        );
        let BuildOutcome::Spawned(bend) =
            graph.apply(BuildCommand::SpawnPipeBend(bend_spec)).unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: support,
                second: bend,
            }))
            .unwrap();
        let drop_spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(
                IVec3::new(drop_x_half_units, 32, 0),
                GridRotation::default(),
            ),
        )
        .unwrap();
        graph.apply(BuildCommand::Spawn(drop_spec)).unwrap();
        graph.compile().unwrap()
    }

    #[test]
    fn gpu_pipe_bend_bore_is_passable_and_annular_material_blocks_motion() {
        let hollow = pipe_bend_drop_creation(0.60, 0);
        let solid = pipe_bend_drop_creation(0.0, 0);
        let annular = pipe_bend_drop_creation(0.60, 3);
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
        assert!(
            hollow_snapshot[falling_body].position[1] < 2.95,
            "centered drop did not enter the bend bore: y={}",
            hollow_snapshot[falling_body].position[1]
        );
        assert!(solid_snapshot[falling_body].position[1] > 3.0);
        assert!(annular_snapshot[falling_body].position[1] > 3.0);
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
    fn balanced_child_contacts_move_root_without_spurious_joint_motion() {
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
        assert!((root_rotation.conjugate() * child_rotation).x.abs() < 1.0e-4);
    }
}
