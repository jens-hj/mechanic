//! Runtime clumps share the construction solver and its collision scene.

use bevy_math::DVec3;
use mechanic_core::{CompiledCreation, MaterialProperties, RuntimeBox};
use mechanic_world::{BreakageResponse, ClumpCollection};

use crate::{BodyPose, MachineCollisionGeometry, MachineState, PhysicsError};

/// Fully validated replacement topology, ready to install between ticks.
pub struct PreparedClumpBodies {
    /// Authored bodies followed by runtime bodies in stable identity order.
    pub creation: CompiledCreation,
    /// Collision geometry for the same rows.
    pub geometry: MachineCollisionGeometry,
    /// Preserved authored state and current clump state.
    pub state: MachineState,
    /// Stable identity of each appended body.
    pub ids: Vec<u64>,
}

impl PreparedClumpBodies {
    /// Prepares batched insertion/removal without modifying the running solver.
    ///
    /// # Errors
    /// Rejects malformed clumps, authored state, geometry or row capacity.
    #[allow(clippy::cast_possible_truncation)] // Validated small fragment geometry and material masses.
    pub fn new(
        base: &CompiledCreation,
        state: &MachineState,
        clumps: &ClumpCollection,
        origin: DVec3,
        generation: u64,
    ) -> Result<Self, PhysicsError> {
        if !clumps.is_valid()
            || !origin.is_finite()
            || state.poses.len() < base.compounds.len()
            || state.velocities.len() < base.dynamics.elimination_parent.len()
        {
            return Err(PhysicsError::InvalidDynamics);
        }
        let boxes = clumps
            .bodies
            .values()
            .map(|body| {
                let response = body.material.surface_response();
                RuntimeBox {
                    half_extents: body.half_extents.as_vec3(),
                    mass: body.mass_kg() as f32,
                    material: MaterialProperties {
                        density_kg_m3: BreakageResponse::for_material(body.material).density_kg_m3
                            as f32,
                        static_friction: response.static_friction,
                        dynamic_friction: response.dynamic_friction,
                        restitution: response.restitution,
                        rolling_resistance: response.rolling_resistance,
                        youngs_modulus_pa: 1e8,
                    },
                }
            })
            .collect::<Vec<_>>();
        let creation = base
            .with_runtime_boxes(&boxes)
            .map_err(|_| PhysicsError::InvalidDynamics)?;
        let geometry = MachineCollisionGeometry::new(&creation, generation)?;
        let mut state = MachineState {
            poses: state.poses[..base.compounds.len()].to_vec(),
            velocities: state.velocities[..base.dynamics.elimination_parent.len()].to_vec(),
            coordinates: state.coordinates.clone(),
        };
        for body in clumps.bodies.values() {
            state.poses.push(BodyPose {
                position: body.position.0 - origin,
                rotation: body.rotation,
            });
            state.velocities.extend(body.linear_velocity.to_array());
            state.velocities.extend(body.angular_velocity.to_array());
        }
        Ok(Self {
            creation,
            geometry,
            state,
            ids: clumps.bodies.keys().copied().collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuMachine, SoftStepConfig, SoftStepTerrain, TerrainContactScene};
    use bevy_math::DQuat;
    use mechanic_world::{MaterialClump, TerrainMaterial, WorldPosition};

    fn loose_box(id: u64, position: DVec3) -> MaterialClump {
        MaterialClump {
            id,
            material: TerrainMaterial::Rock,
            quanta: 510 * 8,
            half_extents: DVec3::splat(0.05),
            position: WorldPosition(position),
            rotation: DQuat::IDENTITY,
            linear_velocity: DVec3::ZERO,
            angular_velocity: DVec3::ZERO,
            settled_seconds: 0.0,
            sleeping: false,
        }
    }

    #[test]
    fn loose_bodies_collide_with_terrain_and_each_other() {
        let base = CompiledCreation::default();
        let clumps = ClumpCollection {
            next_id: 3,
            bodies: [
                (1, loose_box(1, DVec3::Y * 0.1)),
                (2, loose_box(2, DVec3::Y * 0.3)),
            ]
            .into(),
        };
        let prepared = PreparedClumpBodies::new(
            &base,
            &MachineState::at_rest(&base),
            &clumps,
            DVec3::ZERO,
            1,
        )
        .unwrap();
        let mut scene = TerrainContactScene::default();
        scene
            .publish(
                1,
                &[crate::terrain_contacts::tests::terrain(
                    [TerrainMaterial::Rock; 2],
                )],
                &[],
            )
            .unwrap();
        let mut machine = CpuMachine::new(prepared.creation, 1, prepared.state).unwrap();
        for _ in 0..180 {
            machine
                .step(
                    mechanic_core::GRAVITY,
                    &SoftStepConfig::default(),
                    &[],
                    &[],
                    Some(SoftStepTerrain {
                        scene: &scene,
                        geometry: &prepared.geometry,
                        topology_generation: 1,
                        origin: DVec3::ZERO,
                    }),
                )
                .unwrap();
            assert!(!machine.diagnostics().degraded);
        }
        let poses = &machine.snapshot().state.poses;
        assert!(poses.iter().all(|pose| pose.position.y > 0.045));
        assert!(poses[0].position.distance(poses[1].position) > 0.095);
    }

    #[test]
    fn adding_and_removing_clumps_preserves_authored_state_and_tick() {
        let (base, _, _) = crate::terrain_contacts::tests::cube();
        let mut initial = MachineState::at_rest(&base);
        initial.velocities[0] = 0.7;
        let mut machine = CpuMachine::new(base.clone(), 1, initial).unwrap();
        machine
            .step(DVec3::ZERO, &SoftStepConfig::default(), &[], &[], None)
            .unwrap();
        let before = machine.snapshot().clone();
        let clumps = ClumpCollection {
            next_id: 2,
            bodies: [(1, loose_box(1, DVec3::new(10.0, 2.0, 0.0)))].into(),
        };
        let prepared =
            PreparedClumpBodies::new(&base, &before.state, &clumps, DVec3::ZERO, 1).unwrap();
        machine
            .replace_bodies(prepared.creation, prepared.state, base.compounds.len())
            .unwrap();
        let prepared = PreparedClumpBodies::new(
            &base,
            &machine.snapshot().state,
            &ClumpCollection::default(),
            DVec3::ZERO,
            1,
        )
        .unwrap();
        machine
            .replace_bodies(prepared.creation, prepared.state, base.compounds.len())
            .unwrap();
        assert_eq!(machine.snapshot(), &before);
    }
}
