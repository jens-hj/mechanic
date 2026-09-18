//! Narrowphase, manifold, and contact-solver pipelines.

use mechanic_core::ConstructionMaterial;

use super::super::pipelines::lbvh::create_lbvh_resources;
use super::super::wgpu_util::{
    bind_group, compute_pipeline, create_readonly_storage_buffer, create_sized_buffer,
    create_storage_buffer, entry, shader_module,
};
use super::super::{CollisionResources, GpuPhysicsPipelines};
use crate::{
    GpuContact, GpuGroundSurface, GpuPair, GpuPersistentManifold, MAX_BODIES, MAX_CONTACT_PAIRS,
};

#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
pub(in crate::device) fn create_collision_resources(
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
        &crate::shaders::COLLISION.source(),
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

pub(in crate::device) fn contact_pair_capacity(collider_count: usize) -> u32 {
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
