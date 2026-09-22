//! Atomic controller-number edits shared by physical inputs and controller tools.

use std::collections::{BTreeMap, BTreeSet};

use super::ConstructionGraph;
use crate::{
    ActuatorAssignment, DriveDwell, DriveLinkSpec, DriveParameter, DriveTarget, GearParameter,
    GearboxConfig, GraphError, InputBindingError, LinearDriveLimits, MAX_DRIVE_DWELL_SECONDS,
    MAX_DRIVE_LIMIT_RADIANS, MAX_DRIVE_SPEED_RAD_S, MAX_GEAR_RATIO, MIN_GEAR_RATIO,
    NumericParameter, PartId,
};

impl ConstructionGraph {
    /// Writes native-unit values together, without changing construction topology.
    /// Individual bounds clamp; invalid coupled settings reject the entire batch.
    /// Returns whether any accepted value was constrained.
    ///
    /// # Errors
    /// Rejects missing, incompatible, repeated, nonfinite or jointly invalid targets,
    /// and assignments requiring unavailable motor capacity. No values change on error.
    pub fn apply_numeric_values(
        &mut self,
        values: &[(NumericParameter, f32)],
    ) -> Result<bool, GraphError> {
        let mut unique = BTreeSet::new();
        for &(target, value) in values {
            if !value.is_finite() || self.numeric_value(target).is_none() {
                return Err(InputBindingError::InvalidTarget.into());
            }
            if !unique.insert(target) {
                return Err(InputBindingError::DuplicateParameter.into());
            }
        }
        let mut staged = self.clone();
        let mut drives = BTreeMap::new();
        let mut gears = BTreeMap::new();
        for &(target, value) in values {
            match target {
                NumericParameter::Drive { link, parameter } => {
                    drives
                        .entry(link)
                        .or_insert_with(Vec::new)
                        .push((parameter, value));
                }
                NumericParameter::Gear {
                    controller,
                    kind,
                    parameter,
                } => {
                    gears
                        .entry((controller, kind))
                        .or_insert_with(Vec::new)
                        .push((parameter, value));
                }
            }
        }
        let mut controllers = BTreeSet::new();
        for (link, edits) in drives {
            let mut spec = *staged
                .drive_link(link)
                .ok_or(InputBindingError::InvalidTarget)?;
            let physical = staged
                .bearing(spec.bearing)
                .ok_or(InputBindingError::InvalidTarget)?
                .kind
                .bounds();
            edit_drive(&mut spec, &edits, physical)?;
            staged.validate_drive_units(spec.bearing, spec.program, spec.linear_limits)?;
            controllers.insert(spec.controller);
            *staged
                .drive_links
                .get_mut(link)
                .ok_or(InputBindingError::InvalidTarget)? = spec;
        }
        for ((controller, kind), edits) in gears {
            let config = staged.editable_gearbox(controller, kind)?;
            let mut ratios = config.ratios().to_vec();
            let mut reverse = config.reverse_gears();
            for (parameter, value) in edits {
                match parameter {
                    GearParameter::Ratio(index) => {
                        *ratios
                            .get_mut(usize::from(index))
                            .ok_or(InputBindingError::InvalidTarget)? =
                            value.clamp(MIN_GEAR_RATIO, MAX_GEAR_RATIO);
                    }
                    GearParameter::ReverseCount => {
                        reverse = rounded_byte(
                            value,
                            u8::try_from(ratios.len())
                                .map_err(|_| InputBindingError::InvalidTarget)?,
                        );
                    }
                }
            }
            let config = GearboxConfig::new(
                config.mode(),
                ratios,
                reverse,
                config.gear_up(),
                config.gear_down(),
            )?;
            staged.gearbox_configs.insert((controller, kind), config);
        }
        for controller in controllers {
            let inventory = staged
                .actuator_inventory(controller)
                .ok_or(InputBindingError::InvalidTarget)?;
            if inventory.electric_joints > inventory.electric_capacity()
                || inventory.gas_joints > inventory.gas_capacity()
            {
                return Err(InputBindingError::InsufficientCapacity.into());
            }
        }
        let constrained = values.iter().any(|&(target, requested)| {
            staged
                .numeric_value(target)
                .is_none_or(|actual| (actual - requested).abs() > f32::EPSILON)
        });
        *self = staged;
        Ok(constrained)
    }

    /// Operates every mapping of a connected dial as one atomic batch.
    ///
    /// # Errors
    /// Returns the same validation errors as [`Self::apply_numeric_values`], or
    /// an invalid-target error for a disconnected input or nonfinite position.
    pub fn operate_dial(&mut self, input: PartId, position: f32) -> Result<bool, GraphError> {
        if !position.is_finite() {
            return Err(InputBindingError::InvalidTarget.into());
        }
        let config = self
            .input_configuration(input)
            .ok_or(InputBindingError::InvalidTarget)?;
        if config.controller.is_none()
            || !matches!(self.part(input), Some(crate::PartSpec::Dial(_)))
        {
            return Err(InputBindingError::InvalidTarget.into());
        }
        let values = config
            .analog
            .iter()
            .map(|mapping| {
                mapping
                    .range
                    .value(position)
                    .map(|value| (mapping.target, value))
                    .ok_or(InputBindingError::InvalidTarget)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.apply_numeric_values(&values)
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "rounded value is clamped to the u8 range"
)]
fn rounded_byte(value: f32, maximum: u8) -> u8 {
    value.round().clamp(0.0, f32::from(maximum)) as u8
}

fn edit_drive(
    spec: &mut DriveLinkSpec,
    edits: &[(DriveParameter, f32)],
    physical: [f32; 2],
) -> Result<(), GraphError> {
    // Resolve both travel endpoints before validating or clamping any position.
    let travel = spec
        .linear_limits
        .map(|limits| (limits.minimum(), limits.maximum()))
        .or_else(|| spec.limits.angle_limits());
    if let Some((mut minimum, mut maximum)) = travel
        && edits.iter().any(|(parameter, _)| {
            matches!(
                parameter,
                DriveParameter::TravelMinimum | DriveParameter::TravelMaximum
            )
        })
    {
        let bounds = if spec.linear_limits.is_some() {
            physical
        } else {
            [
                -crate::units::MAX_PROGRAMMED_TRAVEL_RADIANS,
                crate::units::MAX_PROGRAMMED_TRAVEL_RADIANS,
            ]
        };
        for &(parameter, value) in edits {
            match parameter {
                DriveParameter::TravelMinimum => minimum = value.clamp(bounds[0], bounds[1]),
                DriveParameter::TravelMaximum => maximum = value.clamp(bounds[0], bounds[1]),
                _ => {}
            }
        }
        let minimum_span = if spec.linear_limits.is_some() {
            crate::units::MIN_PROGRAMMED_TRAVEL_METERS
        } else {
            crate::units::MIN_PROGRAMMED_TRAVEL_RADIANS
        };
        if maximum - minimum < minimum_span {
            return Err(InputBindingError::InvalidRange.into());
        }
        if let Some(limits) = spec.linear_limits {
            spec.linear_limits = Some(
                LinearDriveLimits::new(limits.max_speed(), limits.max_force(), minimum, maximum)
                    .map_err(|_| InputBindingError::InvalidRange)?,
            );
        } else {
            spec.limits = spec
                .limits
                .with_angle_limits(Some((minimum, maximum)))
                .map_err(|_| InputBindingError::InvalidRange)?;
        }
    }
    for &(parameter, value) in edits {
        edit_drive_field(spec, parameter, value)?;
    }
    if edits.iter().any(|(parameter, _)| {
        matches!(
            parameter,
            DriveParameter::TravelMinimum | DriveParameter::TravelMaximum
        )
    }) {
        for index in 0..spec.program.len() {
            let index = u8::try_from(index).map_err(|_| InputBindingError::InvalidTarget)?;
            match spec
                .program
                .state(index)
                .ok_or(InputBindingError::InvalidTarget)?
                .target()
            {
                DriveTarget::Angle(value) => {
                    edit_drive_field(spec, DriveParameter::AngularPosition(index), value)?;
                }
                DriveTarget::LinearPosition(value) => {
                    edit_drive_field(spec, DriveParameter::LinearPosition(index), value)?;
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn edit_drive_field(
    spec: &mut DriveLinkSpec,
    parameter: DriveParameter,
    value: f32,
) -> Result<(), GraphError> {
    let (index, target) = match parameter {
        DriveParameter::AngularPosition(index) => {
            let (low, high) = spec
                .limits
                .angle_limits()
                .unwrap_or((-MAX_DRIVE_LIMIT_RADIANS, MAX_DRIVE_LIMIT_RADIANS));
            (index, DriveTarget::Angle(value.clamp(low, high)))
        }
        DriveParameter::AngularSpeed(index) => (
            index,
            DriveTarget::Speed(value.clamp(-MAX_DRIVE_SPEED_RAD_S, MAX_DRIVE_SPEED_RAD_S)),
        ),
        DriveParameter::LinearPosition(index) => {
            let limits = spec.linear_limits.ok_or(InputBindingError::InvalidTarget)?;
            (
                index,
                DriveTarget::LinearPosition(value.clamp(limits.minimum(), limits.maximum())),
            )
        }
        DriveParameter::LinearSpeed(index) => {
            let limits = spec.linear_limits.ok_or(InputBindingError::InvalidTarget)?;
            (
                index,
                DriveTarget::LinearSpeed(value.clamp(-limits.max_speed(), limits.max_speed())),
            )
        }
        DriveParameter::Dwell(index) => {
            let state = spec
                .program
                .state(index)
                .ok_or(InputBindingError::InvalidTarget)?;
            let next = state
                .dwell()
                .ok_or(InputBindingError::InvalidTarget)?
                .next();
            let dwell = DriveDwell::new(
                value.clamp(crate::units::MIN_DWELL_SECONDS, MAX_DRIVE_DWELL_SECONDS),
                next,
            )
            .map_err(|_| InputBindingError::InvalidRange)?;
            spec.program = spec
                .program
                .with_state(index, state.with_dwell(Some(dwell)))
                .map_err(|_| InputBindingError::InvalidTarget)?;
            return Ok(());
        }
        DriveParameter::ElectricContribution | DriveParameter::GasContribution => {
            let mut electric = spec.actuator.electric_percent();
            let mut gas = spec.actuator.gas_percent();
            if parameter == DriveParameter::ElectricContribution {
                electric = rounded_byte(value, 100);
            } else {
                gas = rounded_byte(value, 100);
            }
            spec.actuator = ActuatorAssignment::motor(electric, gas)
                .map_err(|_| InputBindingError::InvalidRange)?;
            return Ok(());
        }
        DriveParameter::TravelMinimum | DriveParameter::TravelMaximum => return Ok(()),
    };
    let state = spec
        .program
        .state(index)
        .ok_or(InputBindingError::InvalidTarget)?;
    spec.program = spec
        .program
        .with_state(
            index,
            state
                .with_target(target)
                .map_err(|_| InputBindingError::InvalidRange)?,
        )
        .map_err(|_| InputBindingError::InvalidTarget)?;
    Ok(())
}
