//! Graph-owned physical input configuration and typed target validation.

use super::ConstructionGraph;
use crate::{
    AnalogRange, DriveParameter, DriveTarget, GearParameter, GraphError, InputBindingError,
    InputConfiguration, NumericParameter, PartId, PartSpec,
};
use std::collections::BTreeSet;

impl ConstructionGraph {
    /// Persistent configuration of a physical input, including its controller link.
    pub fn input_configuration(&self, input: PartId) -> Option<&InputConfiguration> {
        self.physical_inputs.get(&input)
    }

    /// Configured physical inputs in stable part order.
    pub fn physical_inputs(&self) -> impl Iterator<Item = (PartId, &InputConfiguration)> {
        self.physical_inputs
            .iter()
            .map(|(&id, config)| (id, config))
    }

    /// Reads a typed numeric target in native units, rejecting incompatible modes.
    pub fn numeric_value(&self, target: NumericParameter) -> Option<f32> {
        match target {
            NumericParameter::Drive { link, parameter } => {
                let link = self.drive_link(link)?;
                match parameter {
                    DriveParameter::AngularPosition(state) => {
                        match link.program.state(state)?.target() {
                            DriveTarget::Angle(value) => Some(value),
                            _ => None,
                        }
                    }
                    DriveParameter::AngularSpeed(state) => {
                        match link.program.state(state)?.target() {
                            DriveTarget::Speed(value) => Some(value),
                            _ => None,
                        }
                    }
                    DriveParameter::LinearPosition(state) => {
                        match link.program.state(state)?.target() {
                            DriveTarget::LinearPosition(value) => Some(value),
                            _ => None,
                        }
                    }
                    DriveParameter::LinearSpeed(state) => match link.program.state(state)?.target()
                    {
                        DriveTarget::LinearSpeed(value) => Some(value),
                        _ => None,
                    },
                    DriveParameter::Dwell(state) => {
                        Some(link.program.state(state)?.dwell()?.seconds())
                    }
                    DriveParameter::TravelMinimum => link
                        .linear_limits
                        .map(crate::LinearDriveLimits::minimum)
                        .or_else(|| link.limits.angle_limits().map(|limits| limits.0)),
                    DriveParameter::TravelMaximum => link
                        .linear_limits
                        .map(crate::LinearDriveLimits::maximum)
                        .or_else(|| link.limits.angle_limits().map(|limits| limits.1)),
                    DriveParameter::ElectricContribution => (!link.actuator.uses_servo())
                        .then_some(f32::from(link.actuator.electric_percent())),
                    DriveParameter::GasContribution => (!link.actuator.uses_servo())
                        .then_some(f32::from(link.actuator.gas_percent())),
                }
            }
            NumericParameter::Gear {
                controller,
                kind,
                parameter,
            } => {
                let config = self.gearbox_config(controller, kind).ok()?;
                match parameter {
                    GearParameter::Ratio(index) => config.ratios().get(usize::from(index)).copied(),
                    GearParameter::ReverseCount => (kind == crate::EngineKind::Gas)
                        .then_some(f32::from(config.reverse_gears())),
                }
            }
        }
    }

    pub(super) fn set_input_configuration(
        &mut self,
        input: PartId,
        mut config: InputConfiguration,
    ) -> Result<(), GraphError> {
        let part = self.part(input).ok_or(GraphError::MissingPart(input))?;
        let is_dial = matches!(part, PartSpec::Dial(_));
        if !is_dial && !matches!(part, PartSpec::Button(_)) {
            return Err(InputBindingError::InvalidTarget.into());
        }
        let previous = self
            .physical_inputs
            .get(&input)
            .and_then(|config| config.controller);
        if previous != config.controller {
            // Changing ownership never carries bindings into another controller.
            config.analog.clear();
        }
        if let Some(controller) = config.controller {
            if !self.is_controller(controller) {
                return Err(InputBindingError::InvalidTarget.into());
            }
        } else if !config.analog.is_empty() {
            return Err(InputBindingError::WrongController.into());
        }
        if (is_dial && config.key.is_some()) || (!is_dial && !config.analog.is_empty()) {
            return Err(InputBindingError::InvalidTarget.into());
        }
        let mut targets = BTreeSet::new();
        for mapping in &config.analog {
            AnalogRange::new(mapping.range.endpoints(), mapping.range.inverted())?;
            if self.numeric_value(mapping.target).is_none() {
                return Err(InputBindingError::InvalidTarget.into());
            }
            if self.numeric_controller(mapping.target) != config.controller {
                return Err(InputBindingError::WrongController.into());
            }
            if !targets.insert(mapping.target)
                || self.physical_inputs.iter().any(|(&other, config)| {
                    other != input
                        && config
                            .analog
                            .iter()
                            .any(|binding| binding.target == mapping.target)
                })
            {
                return Err(InputBindingError::DuplicateParameter.into());
            }
        }
        let controller = config.controller;
        self.physical_inputs.insert(input, config);
        if let Some(inventory) =
            controller.and_then(|controller| self.actuator_inventory(controller))
            && (inventory.electric_joints > inventory.electric_capacity()
                || inventory.gas_joints > inventory.gas_capacity())
        {
            return Err(InputBindingError::InsufficientCapacity.into());
        }
        Ok(())
    }

    /// Assigns a connected dial to a field or grouped controller row atomically.
    /// Existing sources for those targets are explicitly replaced. Values are unchanged.
    ///
    /// # Errors
    /// Rejects invalid ranges, integer endpoints, ownership or unavailable motor capacity.
    pub fn assign_dial(
        &mut self,
        input: PartId,
        mappings: &[crate::AnalogMapping],
    ) -> Result<(), GraphError> {
        if mappings.is_empty() || !matches!(self.part(input), Some(PartSpec::Dial(_))) {
            return Err(InputBindingError::InvalidTarget.into());
        }
        let mut staged = self.clone();
        let targets: BTreeSet<_> = mappings.iter().map(|mapping| mapping.target).collect();
        if targets.len() != mappings.len() {
            return Err(InputBindingError::DuplicateParameter.into());
        }
        for mapping in mappings {
            let metadata = self
                .numeric_metadata(mapping.target)
                .ok_or(InputBindingError::InvalidTarget)?;
            if mapping.range.endpoints().iter().any(|value| {
                *value < metadata.minimum
                    || *value > metadata.maximum
                    || metadata
                        .integer_step
                        .is_some_and(|step| (value / step).fract() != 0.0)
            }) {
                return Err(InputBindingError::InvalidRange.into());
            }
        }
        for config in staged.physical_inputs.values_mut() {
            config
                .analog
                .retain(|mapping| !targets.contains(&mapping.target));
        }
        let mut config = staged
            .input_configuration(input)
            .cloned()
            .ok_or(InputBindingError::InvalidTarget)?;
        config.analog.extend_from_slice(mappings);
        staged.set_input_configuration(input, config)?;
        *self = staged;
        Ok(())
    }

    /// Removes assignments for a field or grouped row, preserving current numeric values.
    pub fn unlink_dial_targets(&mut self, targets: &[NumericParameter]) {
        for config in self.physical_inputs.values_mut() {
            config
                .analog
                .retain(|mapping| !targets.contains(&mapping.target));
        }
    }

    /// Whether a drive needs a motor port now or anywhere in its dial range.
    pub fn reserves_motor_port(&self, link: crate::DriveLinkId, kind: crate::EngineKind) -> bool {
        let Some(spec) = self.drive_link(link) else {
            return false;
        };
        let parameter = match kind {
            crate::EngineKind::Electric => {
                if spec.actuator.uses_electric() {
                    return true;
                }
                DriveParameter::ElectricContribution
            }
            crate::EngineKind::Gas => {
                if spec.actuator.uses_gas() {
                    return true;
                }
                DriveParameter::GasContribution
            }
        };
        let target = NumericParameter::Drive { link, parameter };
        self.physical_inputs.values().any(|config| {
            config.analog.iter().any(|mapping| {
                mapping.target == target
                    && mapping
                        .range
                        .endpoints()
                        .into_iter()
                        .any(|value| value > 0.0)
            })
        })
    }

    fn numeric_controller(&self, target: NumericParameter) -> Option<PartId> {
        match target {
            NumericParameter::Drive { link, .. } => {
                self.drive_link(link).map(|link| link.controller)
            }
            NumericParameter::Gear { controller, .. } => {
                self.is_controller(controller).then_some(controller)
            }
        }
    }

    pub(super) fn remap_removed_input_state(
        &mut self,
        removed_link: crate::DriveLinkId,
        removed: u8,
    ) {
        let remap = |state: u8| {
            if state == removed {
                None
            } else {
                Some(state - u8::from(state > removed))
            }
        };
        for config in self.physical_inputs.values_mut() {
            config.analog.retain_mut(|mapping| {
                let NumericParameter::Drive { link, parameter } = &mut mapping.target else {
                    return true;
                };
                if *link != removed_link {
                    return true;
                }
                let (DriveParameter::AngularPosition(state)
                | DriveParameter::AngularSpeed(state)
                | DriveParameter::LinearPosition(state)
                | DriveParameter::LinearSpeed(state)
                | DriveParameter::Dwell(state)) = parameter
                else {
                    return true;
                };
                let Some(new) = remap(*state) else {
                    return false;
                };
                *state = new;
                true
            });
        }
    }

    pub(super) fn prune_input_bindings(&mut self) {
        let mut configs = self.physical_inputs.clone();
        configs.retain(|&input, config| {
            if !matches!(
                self.part(input),
                Some(PartSpec::Dial(_) | PartSpec::Button(_))
            ) {
                return false;
            }
            if config
                .controller
                .is_some_and(|controller| !self.is_controller(controller))
            {
                config.controller = None;
            }
            config.analog.retain(|mapping| {
                self.numeric_value(mapping.target).is_some()
                    && self.numeric_controller(mapping.target) == config.controller
            });
            true
        });
        if configs != self.physical_inputs {
            self.physical_inputs = configs;
        }
    }
}
