//! Buffer allocation and pipeline creation for a compiled creation.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64};

use bytemuck::{Zeroable, bytes_of};
use mechanic_core::{ColliderShape, CompiledCreation};

use super::pipelines::collision::{contact_pair_capacity, create_collision_resources};
use super::pipelines::mechanism::create_mechanism_resources;
use super::wgpu_util::{
    bind_group, compute_pipeline, create_buffer, create_readonly_storage_buffer,
    create_state_buffer, create_storage_buffer, create_uniform_buffer, entry, shader_module, vec4,
};
use super::{
    GpuExternalImpulseBatch, GpuPhysics, GpuPhysicsConfig, GpuPhysicsError, GpuPhysicsPipelines,
    SnapshotBuffers, TimestampResources, full_cylinder_ground_data, hold, terrain,
};
use crate::{
    COLLIDER_SHAPE_CONVEX, COLLIDER_SHAPE_CUBOID, GpuBearing, GpuCollider, GpuDiagnostics, GpuMass,
    GpuPair, GpuSpatialInertia, GpuTickConfig, MAX_BEARINGS, MAX_BODIES, MAX_COLLIDERS,
    MAX_CONTACT_PAIRS, MAX_CONVEX_SHAPE_SLOTS, SNAPSHOT_RING_SIZE, pack_convex_counts,
};

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
                    mechanic_core::JointKind::Suspension(s) => s.passive_rows()[0],
                    _ => [0.0; 4],
                },
                bump_stop: match bearing.kind {
                    mechanic_core::JointKind::Suspension(s) => s.passive_rows()[1],
                    _ => [0.0; 4],
                },
                metadata: [
                    bearing.compound_a,
                    bearing.compound_b,
                    bearing.coordinate_index.unwrap_or(u32::MAX),
                    if bearing.coordinate_index.is_none() {
                        crate::abi::BEARING_CLOSURE_FLAG
                    } else {
                        0
                    },
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
            &crate::shaders::PHYSICS.source(),
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
            &crate::shaders::SNAPSHOT.source(),
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
            &crate::shaders::BEARINGS.source(),
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
}
