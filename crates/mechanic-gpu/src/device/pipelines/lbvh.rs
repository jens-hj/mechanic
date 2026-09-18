//! Broadphase sort and LBVH pipelines.

use super::super::wgpu_util::{
    bind_group, compute_pipeline, create_buffer, create_sized_buffer, create_uniform_buffer, entry,
    shader_module,
};
use super::super::{GpuPhysicsPipelines, LbvhResources};
use crate::GpuPair;

#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
pub(in crate::device) fn create_lbvh_resources(
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
        &crate::shaders::LBVH.source(),
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
