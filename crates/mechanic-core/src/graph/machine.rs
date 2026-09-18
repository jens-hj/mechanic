//! Transmissions, gearboxes, and the actuator inventory of a controller's machine module.

use super::ConstructionGraph;
use super::error::GraphError;
use super::specs::ActuatorInventory;
use crate::{
    BuildPose, EngineKind, FaceKind, FaceOwner, FaceRef, GearboxConfig, PartId, PartSpec,
    TransmissionSpec, WeldId,
};
use bevy_math::Vec3;
use std::collections::BTreeSet;

impl ConstructionGraph {
    /// Parent of a transmission, when `part` is one in a live chain.
    pub fn transmission_parent(&self, part: PartId) -> Option<PartId> {
        self.transmission_parents.get(&part).copied()
    }

    /// Required weld belonging to a transmission.
    pub fn transmission_weld(&self, part: PartId) -> Option<WeldId> {
        self.transmission_welds.get(&part).copied()
    }

    /// Root engine and physical depth of a transmission chain member.
    pub fn transmission_root(&self, part: PartId) -> Option<(PartId, EngineKind, u8)> {
        let mut current = part;
        let mut depth = 0_u8;
        loop {
            match self.parts.get(current)? {
                PartSpec::Engine(engine) => return Some((current, engine.kind, depth)),
                PartSpec::Transmission(_) => {
                    current = *self.transmission_parents.get(&current)?;
                    depth = depth.checked_add(1)?;
                }
                _ => return None,
            }
        }
    }

    /// Number of transmission blocks downstream of one engine.
    pub fn engine_transmission_depth(&self, engine: PartId) -> Option<u8> {
        matches!(self.parts.get(engine), Some(PartSpec::Engine(_))).then_some(())?;
        let mut current = engine;
        let mut depth = 0_u8;
        while let Some((&child, _)) = self
            .transmission_parents
            .iter()
            .find(|(_, parent)| **parent == current)
        {
            current = child;
            depth = depth.saturating_add(1);
        }
        Some(depth)
    }

    /// Exact candidate pose which extends `parent` along the root engine's local +Z axis.
    ///
    /// # Errors
    ///
    /// Returns an error if `parent` is not an engine-line tail, its output is occupied,
    /// or the chain has reached the seventeen-block limit.
    pub fn next_transmission_spec(&self, parent: PartId) -> Result<TransmissionSpec, GraphError> {
        let parent_spec = self
            .parts
            .get(parent)
            .copied()
            .ok_or(GraphError::MissingPart(parent))?;
        let (root, _, depth) = match parent_spec {
            PartSpec::Engine(engine) => (parent, engine.kind, 0),
            PartSpec::Transmission(_) => self
                .transmission_root(parent)
                .ok_or(GraphError::InvalidTransmissionParent(parent))?,
            _ => return Err(GraphError::InvalidTransmissionParent(parent)),
        };
        if self
            .transmission_parents
            .values()
            .any(|candidate| *candidate == parent)
        {
            return Err(GraphError::TransmissionOutputOccupied(parent));
        }
        let output = FaceRef::part(parent, FaceKind::PositiveZ);
        if self
            .welds
            .iter()
            .any(|(_, weld)| weld.first == output || weld.second == output)
        {
            return Err(GraphError::TransmissionOutputOccupied(parent));
        }
        if depth >= 17 {
            return Err(GraphError::TransmissionLimitReached);
        }
        let Some(root_pose) = self.parts.get(root).copied().and_then(|spec| match spec {
            PartSpec::Engine(engine) => Some(engine.pose),
            _ => None,
        }) else {
            return Err(GraphError::InvalidTransmissionParent(parent));
        };
        let parent_z_units = match parent_spec {
            PartSpec::Engine(engine) => engine.kind.grid_units()[2],
            PartSpec::Transmission(_) => TransmissionSpec::GRID_UNITS[2],
            _ => unreachable!(),
        };
        let local_z = root_pose.rotation.quaternion() * Vec3::Z;
        let direction = local_z.round().as_ivec3();
        let centre = parent_spec.pose().translation_position_ticks()
            + direction
                * i32::from(parent_z_units + TransmissionSpec::GRID_UNITS[2])
                * crate::POSITION_TICKS_PER_HALF_GRID_UNIT;
        Ok(TransmissionSpec::new(BuildPose::from_position_ticks(
            centre,
            root_pose.rotation,
        )))
    }

    /// Derived authored appearance for one transmission.
    pub fn transmission_kind(&self, transmission: PartId) -> Option<EngineKind> {
        self.transmission_root(transmission)
            .map(|(_, kind, _)| kind)
    }

    /// Physical same-type engine depths in a Controller's direct machine module.
    pub fn transmission_depths(&self, controller: PartId, kind: EngineKind) -> Option<Vec<u8>> {
        self.is_controller(controller).then(|| {
            let mut depths = self
                .machine_module(controller)
                .into_iter()
                .filter_map(|part| match self.parts.get(part) {
                    Some(PartSpec::Engine(engine)) if engine.kind == kind => {
                        self.engine_transmission_depth(part)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            depths.sort_unstable();
            depths
        })
    }

    /// Effective gearbox settings, including defaults when no override was authored.
    ///
    /// # Errors
    ///
    /// Returns an error if the controller has no engines of `kind` or their physical
    /// transmission depths do not match.
    pub fn gearbox_config(
        &self,
        controller: PartId,
        kind: EngineKind,
    ) -> Result<GearboxConfig, GraphError> {
        let depth = self.common_transmission_depth(controller, kind)?;
        let gear_count = usize::from(depth) + 1;
        Ok(self
            .gearbox_configs
            .get(&(controller, kind))
            .cloned()
            .unwrap_or_else(|| GearboxConfig::for_depth(depth, kind == EngineKind::Gas))
            .resized(gear_count))
    }

    /// Explicit gearbox records in stable controller/type order.
    pub fn gearbox_configs(&self) -> impl Iterator<Item = ((PartId, EngineKind), &GearboxConfig)> {
        self.gearbox_configs
            .iter()
            .map(|(&(controller, kind), config)| ((controller, kind), config))
    }

    pub(super) fn common_transmission_depth(
        &self,
        controller: PartId,
        kind: EngineKind,
    ) -> Result<u8, GraphError> {
        if !self.is_controller(controller) {
            return Err(if self.parts.get(controller).is_some() {
                GraphError::NotAController(controller)
            } else {
                GraphError::MissingPart(controller)
            });
        }
        let depths = self
            .transmission_depths(controller, kind)
            .expect("the controller was checked above");
        let Some(&depth) = depths.first() else {
            return Err(GraphError::GearboxUnavailable { controller, kind });
        };
        if depths.iter().any(|candidate| *candidate != depth) {
            return Err(GraphError::TransmissionDepthMismatch {
                controller,
                kind,
                depths,
            });
        }
        Ok(depth)
    }

    pub(super) fn editable_gearbox(
        &self,
        controller: PartId,
        kind: EngineKind,
    ) -> Result<GearboxConfig, GraphError> {
        let depth = self.common_transmission_depth(controller, kind)?;
        if depth == 0 {
            return Err(GraphError::GearboxUnavailable { controller, kind });
        }
        self.gearbox_config(controller, kind)
    }

    /// Actuator hardware and graph-level assignment demand in a Controller's
    /// direct machine-only weld module.
    pub fn actuator_inventory(&self, controller: PartId) -> Option<ActuatorInventory> {
        self.is_controller(controller).then(|| {
            let members = self.machine_module(controller);
            let mut inventory = ActuatorInventory::default();
            for part in &members {
                match self.parts.get(*part) {
                    Some(PartSpec::Engine(engine)) => match engine.kind {
                        EngineKind::Electric => inventory.electric_engines += 1,
                        EngineKind::Gas => inventory.gas_engines += 1,
                    },
                    Some(PartSpec::Servo(_)) => inventory.servos += 1,
                    _ => {}
                }
            }
            for kind in [EngineKind::Electric, EngineKind::Gas] {
                let mut depths = members
                    .iter()
                    .filter_map(|part| match self.parts.get(*part) {
                        Some(PartSpec::Engine(engine)) if engine.kind == kind => {
                            self.engine_transmission_depth(*part)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                depths.sort_unstable();
                let mismatch = depths
                    .first()
                    .is_some_and(|first| depths.iter().any(|depth| depth != first));
                let common = (!mismatch).then(|| depths.first().copied()).flatten();
                match kind {
                    EngineKind::Electric => {
                        inventory.electric_transmission_depth = common;
                        inventory.electric_transmission_mismatch = mismatch;
                    }
                    EngineKind::Gas => {
                        inventory.gas_transmission_depth = common;
                        inventory.gas_transmission_mismatch = mismatch;
                    }
                }
            }
            let mut electric_coordinates = Vec::<(Vec3, Vec3)>::new();
            let mut gas_coordinates = Vec::<(Vec3, Vec3)>::new();
            let mut servo_coordinates = Vec::<(Vec3, Vec3)>::new();
            for (_, link) in self
                .drive_links
                .iter()
                .filter(|(_, link)| members.contains(&link.controller))
            {
                let Some(bearing) = self.bearings.get(link.bearing) else {
                    continue;
                };
                let coordinate = (bearing.shared_anchor, bearing.axis);
                let contains = |coordinates: &[(Vec3, Vec3)]| {
                    coordinates.iter().any(|(anchor, axis)| {
                        anchor.abs_diff_eq(coordinate.0, 1.0e-5)
                            && axis.abs_diff_eq(coordinate.1, 1.0e-5)
                    })
                };
                if link.actuator.uses_electric() && !contains(&electric_coordinates) {
                    electric_coordinates.push(coordinate);
                }
                if link.actuator.uses_gas() && !contains(&gas_coordinates) {
                    gas_coordinates.push(coordinate);
                }
                if link.actuator.uses_servo() && !contains(&servo_coordinates) {
                    servo_coordinates.push(coordinate);
                }
            }
            inventory.electric_joints =
                u32::try_from(electric_coordinates.len()).unwrap_or(u32::MAX);
            inventory.gas_joints = u32::try_from(gas_coordinates.len()).unwrap_or(u32::MAX);
            inventory.servo_joints = u32::try_from(servo_coordinates.len()).unwrap_or(u32::MAX);
            inventory
        })
    }

    /// Machine parts directly connected through machine-to-machine welds.
    pub(crate) fn machine_module(&self, start: PartId) -> BTreeSet<PartId> {
        let mut found = BTreeSet::from([start]);
        let mut pending = vec![start];
        while let Some(part) = pending.pop() {
            for (_, weld) in self.welds.iter() {
                let (FaceOwner::Part(first), FaceOwner::Part(second)) =
                    (weld.first.owner, weld.second.owner)
                else {
                    continue;
                };
                let next = if first == part {
                    second
                } else if second == part {
                    first
                } else {
                    continue;
                };
                if self.is_machine_part(part) && self.is_machine_part(next) && found.insert(next) {
                    pending.push(next);
                }
            }
        }
        found
    }

    pub(super) fn is_machine_part(&self, part: PartId) -> bool {
        matches!(
            self.parts.get(part),
            Some(
                PartSpec::Controller(_)
                    | PartSpec::Engine(_)
                    | PartSpec::Transmission(_)
                    | PartSpec::Servo(_)
                    | PartSpec::DimensionLink(_)
            )
        )
    }
}
