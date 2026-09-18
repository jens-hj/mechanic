//! Append-only free-body compilation without touching authored graph identities.

use bevy_math::{Mat3, Quat, Vec3};

use crate::{
    ColliderShape, CompiledCompound, CompiledCreation, LocalCollider, MassProperties,
    MaterialProperties, MechanismBodyTopology, PartId, id::Handle,
};

/// Backend-independent loose convex box supplied by a runtime owner.
#[derive(Clone, Copy, Debug)]
pub struct RuntimeBox {
    /// Half extents in the body frame, in metres.
    pub half_extents: Vec3,
    /// Physical mass in kg.
    pub mass: f32,
    /// Contact response.
    pub material: MaterialProperties,
}

impl CompiledCreation {
    /// Appends independent dynamic bodies and rebuilds symbolic schedules.
    /// Authored body, collider, coordinate and part indices remain unchanged.
    /// Runtime colliders have no editable source; their sentinel source handles
    /// must never be used for graph lookup or included in `part_to_compound`.
    ///
    /// # Errors
    /// Rejects non-finite dimensions, mass or overflowing row indices, without
    /// changing the input creation.
    pub fn with_runtime_boxes(&self, bodies: &[RuntimeBox]) -> Result<Self, &'static str> {
        if bodies.iter().any(|body| {
            !body.half_extents.is_finite()
                || body.half_extents.min_element() <= 0.0
                || !body.mass.is_finite()
                || body.mass <= 0.0
        }) {
            return Err("invalid runtime body mass or geometry");
        }
        let mut result = self.clone();
        for body in bodies {
            let row = u32::try_from(result.compounds.len()).map_err(|_| "too many bodies")?;
            let collider =
                u32::try_from(result.colliders.len()).map_err(|_| "too many colliders")?;
            let end = collider.checked_add(1).ok_or("too many colliders")?;
            let component = u32::try_from(result.loop_topology.mechanism_components.len())
                .map_err(|_| "too many components")?;
            let squared = body.half_extents * body.half_extents;
            let inertia = Mat3::from_diagonal(
                Vec3::new(
                    squared.y + squared.z,
                    squared.x + squared.z,
                    squared.x + squared.y,
                ) * (body.mass / 3.0),
            );
            if !inertia.is_finite() || !inertia.inverse().is_finite() {
                return Err("invalid runtime body inertia");
            }
            result.compounds.push(CompiledCompound {
                source_parts: Vec::new(),
                root_translation: Vec3::ZERO,
                root_rotation: Quat::IDENTITY,
                is_static: false,
                mass_properties: MassProperties {
                    mass: body.mass,
                    inverse_mass: body.mass.recip(),
                    center_of_mass: Vec3::ZERO,
                    inertia,
                    inverse_inertia: inertia.inverse(),
                },
                collider_range: collider..end,
            });
            result.colliders.push(LocalCollider {
                source_part: PartId::from_parts(u32::MAX, row),
                compound_index: row,
                local_center: Vec3::ZERO,
                material_properties: body.material,
                shape: ColliderShape::Cuboid {
                    local_rotation: Quat::IDENTITY,
                    half_extents: body.half_extents,
                },
            });
            result.loop_topology.mechanism_components.push(vec![row]);
            result.loop_topology.component_roots.push(vec![row]);
            result
                .loop_topology
                .body_parents
                .push(MechanismBodyTopology {
                    parent_body: row,
                    tree_bearing: None,
                    bearing_direction: 0,
                    component_index: component,
                    depth: 0,
                    preorder_index: row,
                    postorder_index: row,
                    is_root: true,
                });
        }
        result.dynamics = crate::CompiledDynamics::compile(
            &result.compounds,
            &result.bearings,
            &result.loop_topology,
        );
        Ok(result)
    }
}
