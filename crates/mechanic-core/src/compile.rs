mod colliders;
mod compaction;
mod disjoint_set;
mod drives;
mod mass;
mod model;

pub use colliders::cylinder_collider_count;
use colliders::{
    append_evaluated_colliders, append_part_colliders, append_region_colliders,
    band_contact_properties, compose_raw_colliders, solid_full_cylinder,
};
use compaction::compact_grid_aligned_cuboids;
use disjoint_set::{DisjointSet, ordered_pair};
use drives::{
    compile_coordinate_axis_inertia, resolve_coordinate_actuation, resolve_coordinate_drives,
    validate_actuator_programs, validate_transmission_depths,
};
use mass::{calculate_mass_properties, region_pieces};
pub use model::{
    CYLINDER_COLLIDER_COUNT, ColliderShape, CompiledBearing, CompiledCompound, CompiledConvex,
    CompiledCreation, CompiledCylinder, CoordinateDrive, DriveMode, GearSelection, LocalCollider,
    LoopTopology, MAX_COMPILED_COLLIDERS, MassProperties, MechanismBodyTopology,
    PIPE_BEND_COLLIDER_COUNT, TopologyError,
};

use std::collections::{BTreeMap, BTreeSet};

use bevy_math::{Quat, Vec3};

use crate::{BearingId, ConstructionGraph, FaceOwner, PartId, PartSpec, RegionId};

impl ConstructionGraph {
    /// Compiles rigid groups, mass properties, bearings, and loop equations atomically.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError`] when the graph is empty, a bearing collapses
    /// into a weld group, or derived mass properties are invalid.
    pub fn compile(&self) -> Result<CompiledCreation, TopologyError> {
        self.compile_with_sockets([], &[])
    }

    /// Compiles the graph while treating compounds containing any supplied part as static.
    ///
    /// World terrain anchors are instance state rather than authored construction data, so
    /// they must not be represented by a weld to the Garage's flat ground plane.
    ///
    /// # Errors
    ///
    /// Returns the same topology and capacity errors as [`Self::compile`].
    pub fn compile_with_static_parts(
        &self,
        static_parts: impl IntoIterator<Item = PartId>,
    ) -> Result<CompiledCreation, TopologyError> {
        self.compile_with_sockets(static_parts, &[])
    }

    /// Compiles suspension and piston sockets as carried mass on their source compounds.
    ///
    /// An unattached socket contributes its entire assembly mass and inertia.
    /// Sockets already represented by an attached bearing are ignored.
    ///
    /// # Errors
    /// Returns the same topology and capacity errors as [`Self::compile`].
    pub fn compile_with_sockets(
        &self,
        static_parts: impl IntoIterator<Item = PartId>,
        sockets: &[crate::BearingSocket],
    ) -> Result<CompiledCreation, TopologyError> {
        let static_parts = static_parts.into_iter().collect::<BTreeSet<_>>();
        compile_graph(self, &static_parts, sockets)
    }
}

#[expect(clippy::too_many_lines)]
fn compile_graph(
    graph: &ConstructionGraph,
    externally_static_parts: &BTreeSet<PartId>,
    sockets: &[crate::BearingSocket],
) -> Result<CompiledCreation, TopologyError> {
    if graph.parts.is_empty() {
        return Err(TopologyError::EmptyConstruction);
    }

    let part_rows = graph.parts.iter().collect::<Vec<_>>();
    let dense_by_part = part_rows
        .iter()
        .enumerate()
        .map(|(dense, (id, _))| (*id, dense))
        .collect::<BTreeMap<_, _>>();
    // Hardware that carries its own head adds a node for it, so everything
    // attached to one head is one rigid body even when nothing is attached.
    let mut head_nodes = BTreeMap::<HeadKey, usize>::new();
    for (_, bearing) in graph.bearings.iter() {
        if bearing.kind.owns_head() {
            let next = part_rows.len() + head_nodes.len();
            head_nodes.entry(head_key(bearing)).or_insert(next);
        }
    }
    let mut weld_groups = DisjointSet::new(part_rows.len() + head_nodes.len());
    let mut directly_grounded = vec![false; part_rows.len() + head_nodes.len()];
    for (_, bearing) in graph.bearings.iter() {
        if let Some(&head) = head_nodes.get(&head_key(bearing))
            && let Some(FaceOwner::Part(target)) = bearing.target.map(|face| face.owner)
        {
            weld_groups.union(head, dense_by_part[&target]);
        }
    }

    for part in externally_static_parts {
        if let Some(&dense) = dense_by_part.get(part) {
            directly_grounded[dense] = true;
        }
    }

    for (_, weld) in graph.welds.iter() {
        match (weld.first.owner, weld.second.owner) {
            (FaceOwner::Part(a), FaceOwner::Part(b)) => {
                weld_groups.union(dense_by_part[&a], dense_by_part[&b]);
            }
            (FaceOwner::Part(part), FaceOwner::Ground)
            | (FaceOwner::Ground, FaceOwner::Part(part)) => {
                directly_grounded[dense_by_part[&part]] = true;
            }
            (FaceOwner::Ground, FaceOwner::Ground) => {}
        }
    }
    for (_, link) in graph.rigid_links.iter() {
        weld_groups.union(dense_by_part[&link.first], dense_by_part[&link.second]);
    }

    let mut grouped = BTreeMap::<usize, Vec<usize>>::new();
    for dense in 0..part_rows.len() {
        grouped
            .entry(weld_groups.find(dense))
            .or_default()
            .push(dense);
    }
    // A bare head is a body with no parts.
    let mut heads_by_group = BTreeMap::<usize, Vec<HeadKey>>::new();
    for (&key, &node) in &head_nodes {
        let group = weld_groups.find(node);
        grouped.entry(group).or_default();
        heads_by_group.entry(group).or_default().push(key);
    }
    let mut compound_by_head = BTreeMap::<HeadKey, u32>::new();

    let mut compounds = Vec::with_capacity(grouped.len());
    let collider_capacity = part_rows
        .iter()
        .map(|(part, spec)| {
            if graph.owner_has_shape_features(crate::SolidOwner::Part(*part)) {
                return graph
                    .evaluated_solid_shared(crate::SolidOwner::Part(*part))
                    .expect("committed feature geometry replays")
                    .cells
                    .len();
            }
            match spec {
            PartSpec::Controller(_)
            | PartSpec::Engine(_)
            | PartSpec::Transmission(_)
            | PartSpec::Servo(_)
            | PartSpec::Seat(_)
            | PartSpec::Input(_)
            | PartSpec::DimensionLink(_)
            | PartSpec::Cuboid(_) => 1,
            PartSpec::Cylinder(cylinder) => cylinder_collider_count(*cylinder),
            PartSpec::PipeBend(_) => PIPE_BEND_COLLIDER_COUNT,
            PartSpec::PipeJunction(junction) => junction.collider_count(),
            }
        })
        .sum::<usize>()
        // A region emits one row per fused convex piece, so the count is only
        // known by running its decomposition; its blocks emit nothing.
        + graph
            .regions()
            .map(|(id, region)| {
                if graph.owner_has_shape_features(crate::SolidOwner::Region(id)) {
                    graph
                        .evaluated_solid_shared(crate::SolidOwner::Region(id))
                        .expect("committed region feature geometry replays")
                        .cells
                        .len()
                } else {
                    region_pieces(region).len()
                }
            })
            .sum::<usize>();
    if collider_capacity > MAX_COMPILED_COLLIDERS {
        return Err(TopologyError::ColliderBudgetExceeded {
            required: collider_capacity,
            available: MAX_COMPILED_COLLIDERS,
        });
    }
    let mut colliders = Vec::with_capacity(collider_capacity);
    let mut cylinders = Vec::new();
    let mut compound_by_dense_part = vec![0_u32; part_rows.len()];

    // A part inside a region hands its geometry over to that region, so it must
    // not emit a box or a mass of its own as well.
    let mut covered: BTreeSet<PartId> = BTreeSet::new();
    let mut region_of_part: BTreeMap<PartId, RegionId> = BTreeMap::new();
    for (id, _) in graph.parts() {
        if let Some(region) = graph.region_of(id) {
            covered.insert(id);
            region_of_part.insert(id, region);
        }
    }

    for (group, member_rows) in &grouped {
        let compound_index = u32::try_from(compounds.len()).expect("compound count fits u32");
        let member_heads = heads_by_group.get(group).map_or(&[][..], Vec::as_slice);
        for &head in member_heads {
            compound_by_head.insert(head, compound_index);
        }
        let is_static = member_rows.iter().any(|&row| directly_grounded[row]);
        let source_parts = member_rows
            .iter()
            .map(|&row| part_rows[row].0)
            .collect::<Vec<_>>();
        // Regions whose blocks live in this compound, each counted once.
        let mut member_regions: Vec<RegionId> = Vec::new();
        for &row in member_rows {
            if let Some(&region) = region_of_part.get(&part_rows[row].0)
                && !member_regions.contains(&region)
            {
                member_regions.push(region);
            }
        }
        let region_shapes = member_regions
            .iter()
            .filter_map(|&id| graph.region(id).map(|region| (id, region)))
            .collect::<Vec<_>>();

        let mass_properties = calculate_mass_properties(
            member_rows
                .iter()
                .map(|&row| (part_rows[row].0, *part_rows[row].1)),
            is_static,
            &covered,
            &region_shapes,
            graph,
            sockets,
            member_heads,
        )?;
        let collider_start = u32::try_from(colliders.len()).expect("collider count fits u32");
        for &row in member_rows {
            let (part, spec) = part_rows[row];
            compound_by_dense_part[row] = compound_index;
            if covered.contains(&part) {
                continue;
            }
            // Layered cuboids collide per band. An unfeatured layered cylinder
            // keeps its envelope colliders and analytic rolling contact, which
            // only its outer wall's material ever touches.
            if graph.owner_has_shape_features(crate::SolidOwner::Part(part))
                || (spec.is_layered() && spec.as_cylinder().is_none())
            {
                let solid = graph
                    .evaluated_solid_shared(crate::SolidOwner::Part(part))
                    .expect("committed feature geometry replays");
                append_evaluated_colliders(
                    &mut colliders,
                    &solid,
                    part,
                    compound_index,
                    mass_properties.center_of_mass,
                    |band| band_contact_properties(*spec, band),
                );
            } else {
                let start = colliders.len();
                append_part_colliders(&mut colliders, part, compound_index, *spec, Vec3::ZERO);
                compose_raw_colliders(
                    &mut colliders[start..],
                    graph.part_frame(part).expect("compiled part has a frame"),
                    mass_properties.center_of_mass,
                );
                cylinders.extend(solid_full_cylinder(
                    *spec,
                    part,
                    compound_index,
                    start,
                    &colliders[start..],
                ));
            }
        }
        for &(id, region) in &region_shapes {
            let source_part = member_rows
                .iter()
                .map(|&row| part_rows[row].0)
                .find(|part| region_of_part.get(part) == Some(&id))
                .expect("a region in this compound has a member part");
            if graph.owner_has_shape_features(crate::SolidOwner::Region(id)) {
                let solid = graph
                    .evaluated_solid_shared(crate::SolidOwner::Region(id))
                    .expect("committed region feature geometry replays");
                append_evaluated_colliders(
                    &mut colliders,
                    &solid,
                    source_part,
                    compound_index,
                    mass_properties.center_of_mass,
                    |_| region.material().properties(),
                );
            } else {
                let start = colliders.len();
                append_region_colliders(
                    &mut colliders,
                    id,
                    region,
                    compound_index,
                    Vec3::ZERO,
                    source_part,
                );
                compose_raw_colliders(
                    &mut colliders[start..],
                    graph.owner_frame(crate::SolidOwner::Region(id)),
                    mass_properties.center_of_mass,
                );
            }
        }
        compact_grid_aligned_cuboids(
            &mut colliders,
            usize::try_from(collider_start).expect("collider start fits usize"),
        );
        let collider_end = u32::try_from(colliders.len()).expect("collider count fits u32");
        compounds.push(CompiledCompound {
            source_parts,
            root_translation: mass_properties.center_of_mass,
            root_rotation: Quat::IDENTITY,
            is_static,
            mass_properties,
            collider_range: collider_start..collider_end,
        });
    }

    let part_to_compound = part_rows
        .iter()
        .enumerate()
        .map(|(dense, (part, _))| (*part, compound_by_dense_part[dense]))
        .collect::<Vec<_>>();
    let compound_lookup = part_to_compound.iter().copied().collect::<BTreeMap<_, _>>();

    let mut bearing_components = DisjointSet::new(compounds.len());
    let mut topology = LoopTopology::default();
    let mut bearings = Vec::with_capacity(graph.bearings.len());
    let mut suppressed = BTreeSet::new();

    let driven_bearings = graph
        .drive_links
        .iter()
        .map(|(_, link)| link.bearing)
        .collect::<BTreeSet<_>>();

    // Collapse bearings that describe the same physical joint. When one of a
    // duplicate group carries a drive, that row represents the group so its
    // control block is not silently dropped.
    let mut physical_order = Vec::new();
    let mut physical_by_key = BTreeMap::new();
    let mut bearing_keys = Vec::new();
    for (bearing_id, bearing) in graph.bearings.iter() {
        let FaceOwner::Part(part_a) = bearing.source.owner else {
            unreachable!("graph validation rejects ground bearings")
        };
        let compound_a = compound_lookup[&part_a];
        let compound_b = match compound_by_head.get(&head_key(bearing)) {
            Some(&head) if bearing.kind.owns_head() => head,
            _ => {
                let Some(FaceOwner::Part(part_b)) = bearing.target.map(|face| face.owner) else {
                    unreachable!("graph validation rejects ground and bare bearings")
                };
                compound_lookup[&part_b]
            }
        };
        if compound_a == compound_b {
            // Welding a loop shut can leave a bearing with both sides in one
            // rigid body. The weld already fixes their relative pose, so the
            // joint constrains nothing and simply does not compile to one —
            // the player gets a locked bearing, not a build that will not run.
            // A drive is the exception: dropping the joint would kill the
            // motor with nothing to show for it.
            if driven_bearings.contains(&bearing_id) {
                return Err(TopologyError::SelfBearing {
                    bearing: bearing_id,
                    compound: compound_a,
                });
            }
            continue;
        }
        let physical_key = (
            compound_a,
            compound_b,
            bearing.shared_anchor.to_array().map(f32::to_bits),
            bearing.axis.to_array().map(f32::to_bits),
            bearing.kind.bounds().map(f32::to_bits),
            match bearing.kind {
                crate::JointKind::Rotational => (0, 0, [0; 3], 0),
                crate::JointKind::Suspension(_) => (2, 0, [0; 3], 0),
                crate::JointKind::Piston(piston) => (
                    3,
                    u32::from(piston.dimensions.blocks()),
                    match piston.mount {
                        crate::PistonMount::End => [0; 3],
                        crate::PistonMount::Side { mount_normal } => {
                            mount_normal.to_array().map(f32::to_bits)
                        }
                    },
                    piston.dimensions.stages(),
                ),
                crate::JointKind::Linear(rail) => (
                    1,
                    rail.dimensions.width().to_bits(),
                    rail.mount_normal.to_array().map(f32::to_bits),
                    rail.face as u8,
                ),
            },
        );
        bearing_components.union(compound_a as usize, compound_b as usize);
        suppressed.insert(ordered_pair(compound_a, compound_b));
        bearing_keys.push((bearing_id, physical_key));
        match physical_by_key.get(&physical_key).copied() {
            None => {
                physical_by_key.insert(physical_key, physical_order.len());
                physical_order.push((bearing_id, compound_a, compound_b));
            }
            Some(existing) => {
                let (kept, ..) = physical_order[existing];
                if driven_bearings.contains(&bearing_id) && !driven_bearings.contains(&kept) {
                    physical_order[existing].0 = bearing_id;
                }
            }
        }
    }

    // Choose the spanning forest with driven bearings considered first so a
    // drive is never stranded on a loop-closure edge that a passive bearing
    // could have taken instead.
    let mut mechanism_forest = DisjointSet::new(compounds.len());
    let mut forest_has_fixed_root = compounds
        .iter()
        .map(|compound| compound.is_static)
        .collect::<Vec<_>>();
    let mut tree_edges = BTreeSet::new();
    for driven_pass in [true, false] {
        for &(bearing_id, compound_a, compound_b) in &physical_order {
            if driven_bearings.contains(&bearing_id) != driven_pass {
                continue;
            }
            let root_a = mechanism_forest.find(compound_a as usize);
            let root_b = mechanism_forest.find(compound_b as usize);
            let joins_two_fixed_trees =
                root_a != root_b && forest_has_fixed_root[root_a] && forest_has_fixed_root[root_b];
            if root_a == root_b || joins_two_fixed_trees {
                continue;
            }
            let has_fixed_root = forest_has_fixed_root[root_a] || forest_has_fixed_root[root_b];
            mechanism_forest.union(root_a, root_b);
            let joined_root = mechanism_forest.find(root_a);
            forest_has_fixed_root[joined_root] = has_fixed_root;
            tree_edges.insert(bearing_id);
        }
    }

    let mut representative_coordinates = BTreeMap::new();
    for &(bearing_id, compound_a, compound_b) in &physical_order {
        let bearing = graph
            .bearings
            .get(bearing_id)
            .copied()
            .expect("physical bearing rows come from live graph handles");
        let coordinate_index = if tree_edges.contains(&bearing_id) {
            let coordinate = u32::try_from(topology.tree_bearings.len())
                .expect("bearing coordinate count fits u32");
            topology.tree_bearings.push(bearing_id);
            representative_coordinates.insert(bearing_id, coordinate);
            Some(coordinate)
        } else {
            if driven_bearings.contains(&bearing_id) {
                return Err(TopologyError::DrivenClosureBearing {
                    bearing: bearing_id,
                });
            }
            topology.closure_bearings.push(bearing_id);
            None
        };
        let root_a = compounds[compound_a as usize].root_translation;
        let root_b = compounds[compound_b as usize].root_translation;
        bearings.push(CompiledBearing {
            kind: bearing.kind,
            source_bearing: bearing_id,
            compound_a,
            compound_b,
            local_anchor_a: bearing.shared_anchor - root_a,
            local_anchor_b: bearing.shared_anchor - root_b,
            local_axis_a: bearing.axis,
            local_axis_b: bearing.axis,
            coordinate_index,
        });
    }

    // Duplicate rows describing one physical joint share that joint's
    // coordinate, so a drive wired to any of them addresses the same row.
    topology.bearing_coordinates = bearing_keys
        .into_iter()
        .filter_map(|(bearing_id, key)| {
            let representative = physical_order[physical_by_key[&key]].0;
            let coordinate = representative_coordinates.get(&representative)?;
            Some((bearing_id, *coordinate))
        })
        .collect();

    let mut components = BTreeMap::<usize, Vec<u32>>::new();
    for compound in 0..compounds.len() {
        components
            .entry(bearing_components.find(compound))
            .or_default()
            .push(u32::try_from(compound).expect("compound count fits u32"));
    }
    topology.mechanism_components = components.into_values().collect();
    compile_tree_metadata(&compounds, &bearings, &mut topology);
    topology.coordinate_axis_inertia =
        compile_coordinate_axis_inertia(&compounds, &bearings, &topology);

    validate_transmission_depths(graph)?;
    validate_actuator_programs(graph)?;
    let actuation = resolve_coordinate_actuation(&topology, graph, &[])?;
    let coordinate_drives = resolve_coordinate_drives(&topology, graph, &actuation);

    let dynamics = crate::CompiledDynamics::compile(&compounds, &bearings, &topology);
    Ok(CompiledCreation {
        dynamics,
        compounds,
        colliders,
        bearings,
        loop_topology: topology,
        collision_suppression: suppressed.into_iter().collect(),
        part_to_compound,
        coordinate_drives,
        cylinders,
    })
}

/// Identifies the moving head of one mounted piece of hardware by its
/// supporting part and anchor.
pub(super) type HeadKey = (PartId, [u32; 3]);

pub(super) fn head_key(bearing: &crate::BearingSpec) -> HeadKey {
    let FaceOwner::Part(support) = bearing.source.owner else {
        unreachable!("graph validation rejects ground bearings")
    };
    (support, bearing.shared_anchor.to_array().map(f32::to_bits))
}

fn floating_component_root(compounds: &[CompiledCompound], component: &[u32]) -> Option<u32> {
    component.iter().copied().reduce(|root, candidate| {
        if compounds[candidate as usize].mass_properties.mass
            > compounds[root as usize].mass_properties.mass
        {
            candidate
        } else {
            root
        }
    })
}

fn compile_tree_metadata(
    compounds: &[CompiledCompound],
    bearings: &[CompiledBearing],
    topology: &mut LoopTopology,
) {
    let body_count = compounds.len();
    let mut adjacency = vec![Vec::<(usize, BearingId, u32)>::new(); body_count];
    for bearing in bearings {
        if bearing.coordinate_index.is_none() {
            continue;
        }
        let a = bearing.compound_a as usize;
        let b = bearing.compound_b as usize;
        adjacency[a].push((b, bearing.source_bearing, 0));
        adjacency[b].push((a, bearing.source_bearing, 1));
    }
    for neighbours in &mut adjacency {
        neighbours.sort_unstable_by_key(|&(body, bearing, direction)| {
            (body, bearing.index(), bearing.generation(), direction)
        });
    }

    let mut metadata = (0..body_count)
        .map(|body| MechanismBodyTopology {
            parent_body: u32::try_from(body).expect("body count fits u32"),
            tree_bearing: None,
            bearing_direction: 0,
            component_index: 0,
            depth: 0,
            preorder_index: 0,
            postorder_index: 0,
            is_root: true,
        })
        .collect::<Vec<_>>();
    let mut visited = vec![false; body_count];
    let mut preorder = Vec::with_capacity(body_count);
    topology.component_roots.clear();

    for (component_index, component) in topology.mechanism_components.iter().enumerate() {
        let fixed_roots = component
            .iter()
            .copied()
            .filter(|&body| compounds[body as usize].is_static)
            .collect::<Vec<_>>();
        let roots = if fixed_roots.is_empty() {
            // Rooting a floating mechanism at a light appendage makes the
            // opposite side of that joint contain almost the whole machine,
            // producing asymmetric coordinate inertia for mirrored actuators.
            floating_component_root(compounds, component)
                .into_iter()
                .collect::<Vec<_>>()
        } else {
            fixed_roots
        };
        topology.component_roots.push(roots.clone());

        for root in roots {
            let root = root as usize;
            if visited[root] {
                continue;
            }
            visited[root] = true;
            metadata[root].component_index =
                u32::try_from(component_index).expect("component count fits u32");
            let mut queue = std::collections::VecDeque::from([root]);
            while let Some(parent) = queue.pop_front() {
                metadata[parent].preorder_index =
                    u32::try_from(preorder.len()).expect("body count fits u32");
                preorder.push(parent);
                let parent_depth = metadata[parent].depth;
                for &(child, bearing, direction) in &adjacency[parent] {
                    if visited[child] {
                        continue;
                    }
                    visited[child] = true;
                    metadata[child] = MechanismBodyTopology {
                        parent_body: u32::try_from(parent).expect("body count fits u32"),
                        tree_bearing: Some(bearing),
                        bearing_direction: direction,
                        component_index: u32::try_from(component_index)
                            .expect("component count fits u32"),
                        depth: parent_depth + 1,
                        preorder_index: 0,
                        postorder_index: 0,
                        is_root: false,
                    };
                    queue.push_back(child);
                }
            }
        }
    }

    let mut postorder = preorder.clone();
    postorder.sort_unstable_by_key(|&body| (core::cmp::Reverse(metadata[body].depth), body));
    for (index, &body) in postorder.iter().enumerate() {
        metadata[body].postorder_index = u32::try_from(index).expect("body count fits u32");
    }
    let maximum_depth = metadata.iter().map(|body| body.depth).max().unwrap_or(0);
    topology.contraction_rounds = (1..=maximum_depth)
        .rev()
        .map(|depth| {
            metadata
                .iter()
                .enumerate()
                .filter(|(_, row)| row.depth == depth)
                .map(|(body, _)| u32::try_from(body).expect("body count fits u32"))
                .collect()
        })
        .collect();
    topology.body_parents = metadata;
}

#[cfg(test)]
mod tests;
