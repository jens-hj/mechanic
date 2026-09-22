//! Shared native-unit editing bounds for numeric controller settings.

use super::ConstructionGraph;
use crate::{DriveParameter, GearParameter, NumericMetadata, NumericParameter};

impl ConstructionGraph {
    /// Legal endpoints for one enabled setting, including adjacent gear/travel limits.
    pub fn numeric_metadata(&self, target: NumericParameter) -> Option<NumericMetadata> {
        self.numeric_value(target)?;
        let (minimum, maximum, step) = match target {
            NumericParameter::Drive { link, parameter } => {
                let spec = self.drive_link(link)?;
                let physical = self.bearing(spec.bearing)?.kind.bounds();
                let travel = spec
                    .linear_limits
                    .map(|limits| (limits.minimum(), limits.maximum()))
                    .or_else(|| spec.limits.angle_limits());
                match parameter {
                    DriveParameter::AngularPosition(_) => {
                        let (low, high) = travel.unwrap_or((
                            -crate::MAX_DRIVE_LIMIT_RADIANS,
                            crate::MAX_DRIVE_LIMIT_RADIANS,
                        ));
                        (low, high, None)
                    }
                    DriveParameter::AngularSpeed(_) => (
                        -crate::MAX_DRIVE_SPEED_RAD_S,
                        crate::MAX_DRIVE_SPEED_RAD_S,
                        None,
                    ),
                    DriveParameter::LinearPosition(_) => {
                        let (low, high) = travel?;
                        (low, high, None)
                    }
                    DriveParameter::LinearSpeed(_) => {
                        let speed = spec.linear_limits?.max_speed();
                        (-speed, speed, None)
                    }
                    DriveParameter::TravelMinimum | DriveParameter::TravelMaximum => {
                        let (low, high) = travel?;
                        let (bounds, span) = if spec.linear_limits.is_some() {
                            (physical, crate::units::MIN_PROGRAMMED_TRAVEL_METERS)
                        } else {
                            (
                                [
                                    -crate::units::MAX_PROGRAMMED_TRAVEL_RADIANS,
                                    crate::units::MAX_PROGRAMMED_TRAVEL_RADIANS,
                                ],
                                crate::units::MIN_PROGRAMMED_TRAVEL_RADIANS,
                            )
                        };
                        if parameter == DriveParameter::TravelMinimum {
                            (bounds[0], high - span, None)
                        } else {
                            (low + span, bounds[1], None)
                        }
                    }
                    DriveParameter::Dwell(_) => (
                        crate::units::MIN_DWELL_SECONDS,
                        crate::MAX_DRIVE_DWELL_SECONDS,
                        None,
                    ),
                    DriveParameter::ElectricContribution | DriveParameter::GasContribution => {
                        (0.0, 100.0, Some(1.0))
                    }
                }
            }
            NumericParameter::Gear {
                controller,
                kind,
                parameter,
            } => {
                let config = self.editable_gearbox(controller, kind).ok()?;
                match parameter {
                    GearParameter::ReverseCount => (
                        0.0,
                        f32::from(u8::try_from(config.ratios().len()).ok()?),
                        Some(1.0),
                    ),
                    GearParameter::Ratio(index) => {
                        let index = usize::from(index);
                        let maximum = index
                            .checked_sub(1)
                            .and_then(|index| config.ratios().get(index))
                            .map_or(crate::MAX_GEAR_RATIO, |ratio| ratio - 0.01);
                        let minimum = config
                            .ratios()
                            .get(index + 1)
                            .map_or(crate::MIN_GEAR_RATIO, |ratio| ratio + 0.01);
                        (minimum, maximum, None)
                    }
                }
            }
        };
        Some(NumericMetadata {
            minimum,
            maximum,
            integer_step: step,
        })
    }
}
