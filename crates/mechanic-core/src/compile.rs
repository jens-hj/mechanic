use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

use bevy_math::{Mat3, Quat, Vec3};
use thiserror::Error;

use crate::{BearingId, CUBOID_DENSITY_KG_M3, ConstructionGraph, FaceOwner, PartId};

/// Aggregate mass properties expressed in the compiled root frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MassProperties {
    /// Total mass in kilograms, retained for diagnostics even for static bodies.
    pub mass: f32,
    /// Inverse mass used by the solver; zero for static compounds.
    pub inverse_mass: f32,
    /// Centre of mass in build-world coordinates.
    pub center_of_mass: Vec3,
    /// Inertia tensor around the centre of mass.
    pub inertia: Mat3,
    /// Inverse inertia used by the solver; zero for static compounds.
    pub inverse_inertia: Mat3,
}

/// One compound body produced by collapsing an explicit weld group.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledCompound {
    /// Canonically ordered source parts.
    pub source_parts: Vec<PartId>,
    /// Initial root position. Dynamic roots are located at their centre of mass.
    pub root_translation: Vec3,
    /// Initial root rotation.
    pub root_rotation: Quat,
    /// Whether a weld connects this group to the static ground.
    pub is_static: bool,
    /// Aggregate physical properties.
    pub mass_properties: MassProperties,
    /// Contiguous collider rows owned by this body.
    pub collider_range: Range<u32>,
}

/// Cuboid collider expressed relative to its compound root.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalCuboidCollider {
    /// Source editable part.
    pub source_part: PartId,
    /// Owning compound row.
    pub compound_index: u32,
    /// Collider centre in compound-local coordinates.
    pub local_center: Vec3,
    /// Collider orientation relative to the compound root.
    pub local_rotation: Quat,
    /// Cuboid half-extents in metres.
    pub half_extents: Vec3,
}

/// Bearing row connecting two distinct compiled compounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompiledBearing {
    /// Source editable bearing.
    pub source_bearing: BearingId,
    /// Source compound row.
    pub compound_a: u32,
    /// Target compound row.
    pub compound_b: u32,
    /// Anchor relative to source root.
    pub local_anchor_a: Vec3,
    /// Anchor relative to target root.
    pub local_anchor_b: Vec3,
    /// Axis in source root coordinates.
    pub local_axis_a: Vec3,
    /// Axis in target root coordinates.
    pub local_axis_b: Vec3,
    /// Independent mechanism coordinate for a tree edge; `None` for closure edges.
    pub coordinate_index: Option<u32>,
}

/// Canonical parent metadata for one body in the reduced-coordinate forest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MechanismBodyTopology {
    /// Parent body row, or this body for a root.
    pub parent_body: u32,
    /// Tree bearing connecting this body to its parent.
    pub tree_bearing: Option<BearingId>,
    /// Zero when the bearing is traversed from A to B, one for B to A.
    pub bearing_direction: u32,
    /// Canonical connected-component row.
    pub component_index: u32,
    /// Distance from the canonical tree root.
    pub depth: u32,
    /// Stable root-before-children traversal position.
    pub preorder_index: u32,
    /// Stable children-before-root traversal position.
    pub postorder_index: u32,
    /// Whether this body owns a fixed or floating root coordinate.
    pub is_root: bool,
}

/// Canonical forest and hard loop-closure partition.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoopTopology {
    /// Bearings that introduce independent one-dimensional coordinates.
    pub tree_bearings: Vec<BearingId>,
    /// Bearings represented by hard dependent closure equations.
    pub closure_bearings: Vec<BearingId>,
    /// Connected components, each containing sorted compound rows.
    pub mechanism_components: Vec<Vec<u32>>,
    /// Canonical roots for each connected component. Components with multiple
    /// ground anchors have one fixed root for every anchored tree.
    pub component_roots: Vec<Vec<u32>>,
    /// Canonical parent and traversal metadata indexed by compound row.
    pub body_parents: Vec<MechanismBodyTopology>,
    /// Deterministic leaf-to-root contraction rounds over non-root bodies.
    pub contraction_rounds: Vec<Vec<u32>>,
}

/// Complete, immutable upload image for the GPU runtime.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledCreation {
    /// Compound bodies.
    pub compounds: Vec<CompiledCompound>,
    /// Cuboid collider rows.
    pub colliders: Vec<LocalCuboidCollider>,
    /// Passive bearing rows.
    pub bearings: Vec<CompiledBearing>,
    /// Canonical mechanism topology.
    pub loop_topology: LoopTopology,
    /// Sorted compound pairs excluded from collision generation.
    pub collision_suppression: Vec<[u32; 2]>,
    /// Canonical source-part to compound-row lookup.
    pub part_to_compound: Vec<(PartId, u32)>,
}

/// Construction topology cannot be represented by the exact-coordinate model.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum TopologyError {
    /// Simulation requires at least one part.
    #[error("construction contains no cuboids")]
    EmptyConstruction,
    /// A bearing's endpoints were welded into the same compound.
    #[error("bearing {bearing:?} connects compound {compound} to itself after weld compilation")]
    SelfBearing {
        /// Offending bearing.
        bearing: BearingId,
        /// Collapsed compound row.
        compound: u32,
    },
    /// A computed mass or inertia was non-finite or singular.
    #[error("compound containing {part:?} has invalid mass properties")]
    InvalidMassProperties {
        /// One source part identifying the group.
        part: PartId,
    },
}

impl ConstructionGraph {
    /// Compiles rigid groups, mass properties, bearings, and loop equations atomically.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError`] when the graph is empty, a bearing collapses
    /// into a weld group, or derived mass properties are invalid.
    pub fn compile(&self) -> Result<CompiledCreation, TopologyError> {
        compile_graph(self)
    }
}

#[allow(clippy::too_many_lines)]
fn compile_graph(graph: &ConstructionGraph) -> Result<CompiledCreation, TopologyError> {
    if graph.parts.is_empty() {
        return Err(TopologyError::EmptyConstruction);
    }

    let part_rows = graph.parts.iter().collect::<Vec<_>>();
    let dense_by_part = part_rows
        .iter()
        .enumerate()
        .map(|(dense, (id, _))| (*id, dense))
        .collect::<BTreeMap<_, _>>();
    let mut weld_groups = DisjointSet::new(part_rows.len());
    let mut directly_grounded = vec![false; part_rows.len()];

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

    let mut compounds = Vec::with_capacity(grouped.len());
    let mut colliders = Vec::with_capacity(part_rows.len());
    let mut compound_by_dense_part = vec![0_u32; part_rows.len()];

    for member_rows in grouped.values() {
        let compound_index = u32::try_from(compounds.len()).expect("compound count fits u32");
        let is_static = member_rows.iter().any(|&row| directly_grounded[row]);
        let source_parts = member_rows
            .iter()
            .map(|&row| part_rows[row].0)
            .collect::<Vec<_>>();
        let mass_properties = calculate_mass_properties(
            member_rows
                .iter()
                .map(|&row| (part_rows[row].0, *part_rows[row].1)),
            is_static,
        )?;
        let collider_start = u32::try_from(colliders.len()).expect("collider count fits u32");
        for &row in member_rows {
            let (part, spec) = part_rows[row];
            compound_by_dense_part[row] = compound_index;
            colliders.push(LocalCuboidCollider {
                source_part: part,
                compound_index,
                local_center: spec.pose.translation() - mass_properties.center_of_mass,
                local_rotation: spec.pose.rotation.quaternion(),
                half_extents: spec.size_meters() * 0.5,
            });
        }
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

    let mut mechanism_forest = DisjointSet::new(compounds.len());
    let mut bearing_components = DisjointSet::new(compounds.len());
    let mut forest_has_fixed_root = compounds
        .iter()
        .map(|compound| compound.is_static)
        .collect::<Vec<_>>();
    let mut topology = LoopTopology::default();
    let mut bearings = Vec::with_capacity(graph.bearings.len());
    let mut physical_bearing_keys = BTreeSet::new();
    let mut suppressed = BTreeSet::new();

    for (bearing_id, bearing) in graph.bearings.iter() {
        let FaceOwner::Part(part_a) = bearing.source.owner else {
            unreachable!("graph validation rejects ground bearings")
        };
        let FaceOwner::Part(part_b) = bearing.target.owner else {
            unreachable!("graph validation rejects ground bearings")
        };
        let compound_a = compound_lookup[&part_a];
        let compound_b = compound_lookup[&part_b];
        if compound_a == compound_b {
            return Err(TopologyError::SelfBearing {
                bearing: bearing_id,
                compound: compound_a,
            });
        }
        let physical_key = (
            compound_a,
            compound_b,
            bearing.shared_anchor.to_array().map(f32::to_bits),
            bearing.axis.to_array().map(f32::to_bits),
        );
        if !physical_bearing_keys.insert(physical_key) {
            continue;
        }

        let a = compound_a as usize;
        let b = compound_b as usize;
        bearing_components.union(a, b);
        let root_a = mechanism_forest.find(a);
        let root_b = mechanism_forest.find(b);
        let joins_two_fixed_trees =
            root_a != root_b && forest_has_fixed_root[root_a] && forest_has_fixed_root[root_b];
        let coordinate_index = if root_a == root_b || joins_two_fixed_trees {
            topology.closure_bearings.push(bearing_id);
            None
        } else {
            let has_fixed_root = forest_has_fixed_root[root_a] || forest_has_fixed_root[root_b];
            mechanism_forest.union(root_a, root_b);
            let joined_root = mechanism_forest.find(root_a);
            forest_has_fixed_root[joined_root] = has_fixed_root;
            let coordinate = u32::try_from(topology.tree_bearings.len())
                .expect("bearing coordinate count fits u32");
            topology.tree_bearings.push(bearing_id);
            Some(coordinate)
        };
        let root_a = compounds[a].root_translation;
        let root_b = compounds[b].root_translation;
        bearings.push(CompiledBearing {
            source_bearing: bearing_id,
            compound_a,
            compound_b,
            local_anchor_a: bearing.shared_anchor - root_a,
            local_anchor_b: bearing.shared_anchor - root_b,
            local_axis_a: bearing.axis,
            local_axis_b: bearing.axis,
            coordinate_index,
        });
        suppressed.insert(ordered_pair(compound_a, compound_b));
    }

    let mut components = BTreeMap::<usize, Vec<u32>>::new();
    for compound in 0..compounds.len() {
        components
            .entry(bearing_components.find(compound))
            .or_default()
            .push(u32::try_from(compound).expect("compound count fits u32"));
    }
    topology.mechanism_components = components.into_values().collect();
    compile_tree_metadata(&compounds, &bearings, &mut topology);

    Ok(CompiledCreation {
        compounds,
        colliders,
        bearings,
        loop_topology: topology,
        collision_suppression: suppressed.into_iter().collect(),
        part_to_compound,
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
            component.first().copied().into_iter().collect::<Vec<_>>()
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

fn calculate_mass_properties<'a>(
    parts: impl Iterator<Item = (PartId, crate::CuboidSpec)> + Clone + 'a,
    is_static: bool,
) -> Result<MassProperties, TopologyError> {
    let mut total_mass = 0.0;
    let mut weighted_center = Vec3::ZERO;
    let identifying_part = parts.clone().next().expect("weld groups are non-empty").0;
    for (_, spec) in parts.clone() {
        let size = spec.size_meters();
        let mass = CUBOID_DENSITY_KG_M3 * size.x * size.y * size.z;
        total_mass += mass;
        weighted_center += spec.pose.translation() * mass;
    }
    let center_of_mass = weighted_center / total_mass;
    let mut inertia = Mat3::ZERO;
    for (_, spec) in parts {
        let size = spec.size_meters();
        let mass = CUBOID_DENSITY_KG_M3 * size.x * size.y * size.z;
        let diagonal = Vec3::new(
            mass * (size.y * size.y + size.z * size.z) / 12.0,
            mass * (size.x * size.x + size.z * size.z) / 12.0,
            mass * (size.x * size.x + size.y * size.y) / 12.0,
        );
        let rotation = Mat3::from_quat(spec.pose.rotation.quaternion());
        let own_inertia = rotation * Mat3::from_diagonal(diagonal) * rotation.transpose();
        let offset = spec.pose.translation() - center_of_mass;
        let outer = Mat3::from_cols(offset * offset.x, offset * offset.y, offset * offset.z);
        inertia += own_inertia + mass * (Mat3::IDENTITY * offset.length_squared() - outer);
    }

    let determinant = inertia.determinant();
    if !total_mass.is_finite()
        || total_mass <= 0.0
        || !center_of_mass.is_finite()
        || !inertia.is_finite()
        || !determinant.is_finite()
        || determinant <= f32::EPSILON
    {
        return Err(TopologyError::InvalidMassProperties {
            part: identifying_part,
        });
    }

    Ok(MassProperties {
        mass: total_mass,
        inverse_mass: if is_static { 0.0 } else { total_mass.recip() },
        center_of_mass,
        inertia,
        inverse_inertia: if is_static {
            Mat3::ZERO
        } else {
            inertia.inverse()
        },
    })
}

const fn ordered_pair(a: u32, b: u32) -> [u32; 2] {
    if a < b { [a, b] } else { [b, a] }
}

#[derive(Clone, Debug)]
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, mut item: usize) -> usize {
        let mut root = item;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        while self.parent[item] != item {
            let parent = self.parent[item];
            self.parent[item] = root;
            item = parent;
        }
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return;
        }
        if self.rank[root_a] < self.rank[root_b] {
            core::mem::swap(&mut root_a, &mut root_b);
        }
        self.parent[root_b] = root_a;
        if self.rank[root_a] == self.rank[root_b] {
            self.rank[root_a] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy_math::{IVec3, Vec3};

    use crate::{
        BearingDimensions, BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
        CuboidSpec, FaceKind, FaceRef, GridRotation, RigidLinkSpec, TopologyError, WeldSpec,
    };

    fn cube_at(units: IVec3) -> CuboidSpec {
        CuboidSpec::new([4, 4, 4], BuildPose::new(units, GridRotation::default())).unwrap()
    }

    fn spawn(graph: &mut ConstructionGraph, units: IVec3) -> crate::PartId {
        let BuildOutcome::Spawned(id) = graph.apply(BuildCommand::Spawn(cube_at(units))).unwrap()
        else {
            panic!("wrong spawn outcome")
        };
        id
    }

    fn ground(graph: &mut ConstructionGraph, part: crate::PartId) {
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(part, FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();
    }

    fn bearing(
        graph: &mut ConstructionGraph,
        a: crate::PartId,
        face_a: FaceKind,
        b: crate::PartId,
        face_b: FaceKind,
        anchor: Vec3,
        axis: Vec3,
    ) {
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(a, face_a),
                FaceRef::part(b, face_b),
                anchor,
                axis,
            )))
            .unwrap();
    }

    #[test]
    fn welds_compile_to_one_compound_with_parallel_axis_inertia() {
        let mut graph = ConstructionGraph::new();
        let a = spawn(&mut graph, IVec3::ZERO);
        let b = spawn(&mut graph, IVec3::new(4, 0, 0));
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(a, FaceKind::PositiveX),
                second: FaceRef::part(b, FaceKind::NegativeX),
            }))
            .unwrap();

        let compiled = graph.compile().unwrap();
        let properties = compiled.compounds[0].mass_properties;
        assert_eq!(compiled.compounds.len(), 1);
        assert_eq!(compiled.colliders.len(), 2);
        assert!((properties.mass - 1000.0).abs() < 1.0e-3);
        assert!(
            properties
                .center_of_mass
                .abs_diff_eq(Vec3::new(0.5, 0.0, 0.0), 1.0e-6)
        );
        assert!((properties.inertia.x_axis.x - 166.666_67).abs() < 1.0e-3);
        assert!((properties.inertia.y_axis.y - 416.666_7).abs() < 1.0e-3);
        assert!((properties.inertia.z_axis.z - 416.666_7).abs() < 1.0e-3);
    }

    #[test]
    fn shared_bearing_attachments_compile_as_one_rotor_and_one_joint() {
        let mut graph = ConstructionGraph::new();
        let support = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(support) = graph.apply(BuildCommand::Spawn(support)).unwrap()
        else {
            unreachable!()
        };
        let targets = [IVec3::new(0, 9, 0), IVec3::new(2, 9, 0)].map(|center| {
            let spec = CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_half_grid(center, GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        });
        let dimensions = BearingDimensions::new(0.80, 0.10).unwrap();
        for target in targets {
            graph
                .apply(BuildCommand::AddBearing(
                    BearingSpec::new(
                        FaceRef::part(support, FaceKind::PositiveY),
                        FaceRef::part(target, FaceKind::NegativeY),
                        Vec3::Y,
                        Vec3::Y,
                    )
                    .with_dimensions(dimensions),
                ))
                .unwrap();
        }
        graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: targets[0],
                second: targets[1],
            }))
            .unwrap();

        let compiled = graph.compile().unwrap();
        let compound_for = |part| {
            compiled
                .part_to_compound
                .iter()
                .find_map(|&(candidate, compound)| (candidate == part).then_some(compound))
                .unwrap()
        };
        assert_eq!(compiled.compounds.len(), 2);
        assert_eq!(compiled.bearings.len(), 1);
        assert_eq!(compound_for(targets[0]), compound_for(targets[1]));
        assert_ne!(compound_for(support), compound_for(targets[0]));
    }

    #[test]
    fn welding_to_ground_makes_only_that_group_static() {
        let mut graph = ConstructionGraph::new();
        let grounded = spawn(&mut graph, IVec3::new(0, 2, 0));
        spawn(&mut graph, IVec3::new(8, 2, 0));
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(grounded, FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.compounds.len(), 2);
        assert!(compiled.compounds[0].is_static);
        assert!(compiled.compounds[0].mass_properties.inverse_mass.abs() < f32::EPSILON);
        assert!(!compiled.compounds[1].is_static);
    }

    #[test]
    fn bearing_collapsed_by_later_weld_is_rejected() {
        let mut graph = ConstructionGraph::new();
        let a = spawn(&mut graph, IVec3::ZERO);
        let b = spawn(&mut graph, IVec3::new(4, 0, 0));
        let BuildOutcome::BearingAdded(bearing) = graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(a, FaceKind::PositiveX),
                FaceRef::part(b, FaceKind::NegativeX),
                Vec3::new(0.5, 0.0, 0.0),
                Vec3::X,
            )))
            .unwrap()
        else {
            panic!("wrong bearing outcome")
        };
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(a, FaceKind::PositiveX),
                second: FaceRef::part(b, FaceKind::NegativeX),
            }))
            .unwrap();

        assert_eq!(
            graph.compile(),
            Err(TopologyError::SelfBearing {
                bearing,
                compound: 0
            })
        );
    }

    #[test]
    fn bearing_dimensions_do_not_change_compiled_physics() {
        let compile_with = |dimensions: BearingDimensions| {
            let mut graph = ConstructionGraph::new();
            let a = spawn(&mut graph, IVec3::ZERO);
            let b = spawn(&mut graph, IVec3::new(4, 0, 0));
            graph
                .apply(BuildCommand::AddBearing(
                    BearingSpec::new(
                        FaceRef::part(a, FaceKind::PositiveX),
                        FaceRef::part(b, FaceKind::NegativeX),
                        Vec3::new(0.5, 0.0, 0.0),
                        Vec3::X,
                    )
                    .with_dimensions(dimensions),
                ))
                .unwrap();
            graph.compile().unwrap()
        };

        assert_eq!(
            compile_with(BearingDimensions::default()),
            compile_with(BearingDimensions::new(0.50, 0.20).unwrap())
        );
    }

    #[test]
    fn closed_square_has_one_hard_closure_edge() {
        let mut graph = ConstructionGraph::new();
        let a = spawn(&mut graph, IVec3::ZERO);
        let b = spawn(&mut graph, IVec3::new(4, 0, 0));
        let c = spawn(&mut graph, IVec3::new(4, 4, 0));
        let d = spawn(&mut graph, IVec3::new(0, 4, 0));
        let edges = [
            (
                a,
                FaceKind::PositiveX,
                b,
                FaceKind::NegativeX,
                Vec3::new(0.5, 0.0, 0.0),
                Vec3::X,
            ),
            (
                b,
                FaceKind::PositiveY,
                c,
                FaceKind::NegativeY,
                Vec3::new(1.0, 0.5, 0.0),
                Vec3::Y,
            ),
            (
                c,
                FaceKind::NegativeX,
                d,
                FaceKind::PositiveX,
                Vec3::new(0.5, 1.0, 0.0),
                Vec3::NEG_X,
            ),
            (
                d,
                FaceKind::NegativeY,
                a,
                FaceKind::PositiveY,
                Vec3::new(0.0, 0.5, 0.0),
                Vec3::NEG_Y,
            ),
        ];
        for (source, source_face, target, target_face, anchor, axis) in edges {
            graph
                .apply(BuildCommand::AddBearing(BearingSpec::new(
                    FaceRef::part(source, source_face),
                    FaceRef::part(target, target_face),
                    anchor,
                    axis,
                )))
                .unwrap();
        }

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.loop_topology.tree_bearings.len(), 3);
        assert_eq!(compiled.loop_topology.closure_bearings.len(), 1);
        assert_eq!(
            compiled.loop_topology.mechanism_components,
            vec![vec![0, 1, 2, 3]]
        );
        assert_eq!(compiled.collision_suppression.len(), 4);
    }

    #[test]
    fn floating_branch_has_one_canonical_root_and_stable_traversals() {
        let mut graph = ConstructionGraph::new();
        let root = spawn(&mut graph, IVec3::ZERO);
        let x_child = spawn(&mut graph, IVec3::new(4, 0, 0));
        let y_child = spawn(&mut graph, IVec3::new(0, 4, 0));
        bearing(
            &mut graph,
            root,
            FaceKind::PositiveX,
            x_child,
            FaceKind::NegativeX,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::X,
        );
        bearing(
            &mut graph,
            root,
            FaceKind::PositiveY,
            y_child,
            FaceKind::NegativeY,
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::Y,
        );

        let topology = graph.compile().unwrap().loop_topology;
        assert_eq!(topology.component_roots, vec![vec![0]]);
        assert!(topology.body_parents[0].is_root);
        assert_eq!(topology.body_parents[1].parent_body, 0);
        assert_eq!(topology.body_parents[2].parent_body, 0);
        assert_eq!(topology.body_parents[1].preorder_index, 1);
        assert_eq!(topology.body_parents[2].preorder_index, 2);
        assert_eq!(topology.contraction_rounds, vec![vec![1, 2]]);
    }

    #[test]
    fn grounded_body_is_root_even_when_not_the_first_compound() {
        let mut graph = ConstructionGraph::new();
        let child = spawn(&mut graph, IVec3::new(0, 6, 0));
        let root = spawn(&mut graph, IVec3::new(0, 2, 0));
        ground(&mut graph, root);
        bearing(
            &mut graph,
            child,
            FaceKind::NegativeY,
            root,
            FaceKind::PositiveY,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::NEG_Y,
        );

        let topology = graph.compile().unwrap().loop_topology;
        assert_eq!(topology.component_roots, vec![vec![1]]);
        assert_eq!(topology.body_parents[0].parent_body, 1);
        assert_eq!(topology.body_parents[0].bearing_direction, 1);
        assert!(topology.body_parents[1].is_root);
    }

    #[test]
    fn multiple_ground_anchors_remain_fixed_roots_with_a_closure_edge() {
        let mut graph = ConstructionGraph::new();
        let left = spawn(&mut graph, IVec3::new(0, 2, 0));
        let middle = spawn(&mut graph, IVec3::new(4, 2, 0));
        let right = spawn(&mut graph, IVec3::new(8, 2, 0));
        ground(&mut graph, left);
        ground(&mut graph, right);
        bearing(
            &mut graph,
            left,
            FaceKind::PositiveX,
            middle,
            FaceKind::NegativeX,
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::X,
        );
        bearing(
            &mut graph,
            middle,
            FaceKind::PositiveX,
            right,
            FaceKind::NegativeX,
            Vec3::new(1.5, 0.5, 0.0),
            Vec3::X,
        );

        let topology = graph.compile().unwrap().loop_topology;
        assert_eq!(topology.component_roots, vec![vec![0, 2]]);
        assert_eq!(topology.tree_bearings.len(), 1);
        assert_eq!(topology.closure_bearings.len(), 1);
        assert!(topology.body_parents[0].is_root);
        assert_eq!(topology.body_parents[1].parent_body, 0);
        assert!(topology.body_parents[2].is_root);
    }
}
