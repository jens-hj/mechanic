//! Immutable schedules shared by numerical dynamics backends.

use std::{collections::BTreeMap, ops::Range};

use bevy_math::{Mat3, Vec3};

use crate::{CompiledBearing, CompiledCompound, LoopTopology};

/// Spatial inertia about a body's local origin, before pose-dependent rotation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialInertia {
    /// Physical mass, including for anchored bodies.
    pub mass: f32,
    /// Centre of mass relative to the body origin.
    pub center: Vec3,
    /// Rotational inertia about the centre of mass in the body frame.
    pub rotational: Mat3,
}

/// Contiguous ranges into the compiled traversal and generalized velocity rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DynamicsComponent {
    /// Body indices in root-before-child order in `CompiledDynamics::preorder`.
    pub bodies: Range<usize>,
    /// Six velocities per floating root and one per permitted joint motion.
    pub velocities: Range<usize>,
}

/// Compressed loop Jacobian sparsity: two ancestor chains in the elimination tree.
/// Common columns are combined by subtraction, never duplicated as constraints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopConstraintPattern {
    /// Index into the compiled bearing array.
    pub bearing: usize,
    /// Last velocity row of each endpoint; follow `elimination_parent` to its root.
    pub branch_heads: [Option<usize>; 2],
}

/// Reusable symbolic dynamics data. Storage is linear in bodies and bearings;
/// numerical factors and contact matrices are owned by the runtime.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompiledDynamics {
    /// Direct parent-joint lookup indexed by body, absent for roots.
    pub body_bearings: Vec<Option<usize>>,
    /// Direct joint lookup indexed by the authored mechanism coordinate.
    pub coordinate_bearings: Vec<usize>,
    /// Direct generalized velocity row for each authored tree coordinate.
    pub coordinate_velocities: Vec<usize>,
    /// Parent-before-child body order, grouped into component ranges.
    pub preorder: Vec<usize>,
    /// Child-before-parent order, suitable for inertia accumulation.
    pub postorder: Vec<usize>,
    /// Generalized velocity rows owned by each body (zero, one, or six).
    pub body_velocities: Vec<Range<usize>>,
    /// Symbolic factor elimination tree. Eliminate rows in descending order;
    /// each parent is smaller than its child. No dense symbolic matrix is stored.
    pub elimination_parent: Vec<Option<usize>>,
    /// Component ranges into the shared schedules.
    pub components: Vec<DynamicsComponent>,
    /// Loop equation sparsity without expanding long ancestor chains.
    pub loops: Vec<LoopConstraintPattern>,
    /// Body-frame mass data, indexed by body.
    pub inertias: Vec<SpatialInertia>,
}

impl CompiledDynamics {
    pub(crate) fn compile(
        compounds: &[CompiledCompound],
        bearings: &[CompiledBearing],
        topology: &LoopTopology,
    ) -> Self {
        let lookup = bearings
            .iter()
            .enumerate()
            .map(|(row, bearing)| (bearing.source_bearing, row))
            .collect::<BTreeMap<_, _>>();
        let body_bearings = topology
            .body_parents
            .iter()
            .map(|body| body.tree_bearing.map(|id| lookup[&id]))
            .collect();
        let mut result = Self {
            body_bearings,
            coordinate_bearings: topology.tree_bearings.iter().map(|id| lookup[id]).collect(),
            body_velocities: vec![0..0; compounds.len()],
            inertias: compounds
                .iter()
                .map(|body| {
                    let rotation = Mat3::from_quat(body.root_rotation);
                    SpatialInertia {
                        mass: body.mass_properties.mass,
                        center: rotation.transpose()
                            * (body.mass_properties.center_of_mass - body.root_translation),
                        rotational: rotation.transpose() * body.mass_properties.inertia * rotation,
                    }
                })
                .collect(),
            ..Self::default()
        };
        for bodies in &topology.mechanism_components {
            let first_body = result.preorder.len();
            let first_velocity = result.elimination_parent.len();
            let mut order = bodies.iter().map(|&body| body as usize).collect::<Vec<_>>();
            order.sort_unstable_by_key(|&body| topology.body_parents[body].preorder_index);
            for &body in &order {
                let parent = topology.body_parents[body];
                let count = if parent.is_root {
                    if compounds[body].is_static { 0 } else { 6 }
                } else {
                    1
                };
                let start = result.elimination_parent.len();
                let mut previous = if parent.is_root {
                    None
                } else {
                    result.body_velocities[parent.parent_body as usize]
                        .clone()
                        .last()
                };
                for row in start..start + count {
                    result.elimination_parent.push(previous);
                    previous = Some(row);
                }
                result.body_velocities[body] = start..start + count;
            }
            result.preorder.extend(order);
            result.components.push(DynamicsComponent {
                bodies: first_body..result.preorder.len(),
                velocities: first_velocity..result.elimination_parent.len(),
            });
        }
        result.postorder = result.preorder.iter().rev().copied().collect();
        result.coordinate_velocities = vec![0; result.coordinate_bearings.len()];
        for (body, bearing) in result.body_bearings.iter().enumerate() {
            if let Some(coordinate) = bearing.and_then(|row| bearings[row].coordinate_index) {
                result.coordinate_velocities[coordinate as usize] =
                    result.body_velocities[body].start;
            }
        }
        result.loops = topology
            .closure_bearings
            .iter()
            .map(|id| {
                let row = lookup[id];
                let bearing = bearings[row];
                LoopConstraintPattern {
                    bearing: row,
                    branch_heads: [bearing.compound_a, bearing.compound_b]
                        .map(|body| result.body_velocities[body as usize].clone().last()),
                }
            })
            .collect();
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::{BuildCommand, BuildPose, ConstructionGraph, CuboidSpec, GridRotation};
    use bevy_math::IVec3;

    #[test]
    fn independent_bodies_have_disjoint_six_velocity_components() {
        let mut graph = ConstructionGraph::new();
        for x in [0, 10] {
            graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [1, 1, 1],
                        BuildPose::new(IVec3::new(x, 10, 0), GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap();
        }
        let creation = graph.compile().unwrap();
        let dynamics = &creation.dynamics;
        assert_eq!(dynamics.components.len(), 2);
        assert_eq!(dynamics.body_velocities, [0..6, 6..12]);
        assert_eq!(dynamics.elimination_parent[0], None);
        assert_eq!(dynamics.elimination_parent[6], None);
        assert_eq!(dynamics, &graph.compile().unwrap().dynamics);
    }
}
