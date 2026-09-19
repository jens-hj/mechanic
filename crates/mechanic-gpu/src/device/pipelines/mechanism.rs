//! Mechanism, articulated-dynamics, and closure pipelines.

use std::collections::{BTreeMap, BTreeSet};

use bytemuck::Zeroable;
use mechanic_core::CompiledCreation;

use super::super::wgpu_util::{
    bind_group, compute_pipeline, create_readonly_storage_buffer, create_sized_buffer,
    create_state_buffer, create_storage_buffer, entry, shader_module, vec4,
};
use super::super::{GpuDriveConstraint, GpuPhysicsPipelines, MechanismResources};
use crate::{
    GpuBearing, GpuContractionNode, GpuLinkState, GpuMechanismBody, GpuMechanismCoordinate,
    GpuMechanismDrive, GpuSpatialInertia,
};

#[expect(
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
pub(in crate::device) fn create_mechanism_resources(
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
                        coordinate,
                        if bearing.coordinate_index.is_none() {
                            crate::abi::BEARING_CLOSURE_FLAG
                        } else {
                            0
                        },
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
        &crate::shaders::MECHANISM_KERNEL.source(),
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
        &crate::shaders::ARTICULATED.source(),
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
        &crate::shaders::CLOSURE.source(),
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
