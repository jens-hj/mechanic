//! Actuator programs resolved into per-coordinate drive rows.

use super::model::{
    CompiledBearing, CompiledCompound, CoordinateDrive, DriveMode, GearSelection, LoopTopology,
    TopologyError,
};
use crate::{
    ActuatorAssignment, ConstructionGraph, DriveLimits, DriveTarget, EngineKind, PartId, PartSpec,
    ServoSpec,
};
use bevy_math::Vec3;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn validate_transmission_depths(graph: &ConstructionGraph) -> Result<(), TopologyError> {
    for (controller, spec) in graph.parts() {
        if !matches!(spec, PartSpec::Controller(_)) {
            continue;
        }
        for kind in [EngineKind::Electric, EngineKind::Gas] {
            let depths = graph
                .transmission_depths(controller, kind)
                .expect("the part was checked as a controller");
            if let Some(first) = depths.first()
                && depths.iter().any(|depth| depth != first)
            {
                return Err(TopologyError::TransmissionDepthMismatch {
                    controller,
                    kind,
                    depths,
                });
            }
        }
    }
    Ok(())
}

/// Rotational inertia of each tree bearing's child subtree about that bearing's
/// own axis, evaluated in the compile-time bind pose.
pub(super) fn compile_coordinate_axis_inertia(
    compounds: &[CompiledCompound],
    bearings: &[CompiledBearing],
    topology: &LoopTopology,
) -> Vec<f32> {
    let mut children = vec![Vec::<usize>::new(); compounds.len()];
    for (body, row) in topology.body_parents.iter().enumerate() {
        if !row.is_root {
            children[row.parent_body as usize].push(body);
        }
    }
    let child_body_by_bearing = topology
        .body_parents
        .iter()
        .enumerate()
        .filter_map(|(body, row)| row.tree_bearing.map(|bearing| (bearing, body)))
        .collect::<BTreeMap<_, _>>();

    topology
        .tree_bearings
        .iter()
        .map(|source_bearing| {
            let Some(&child_body) = child_body_by_bearing.get(source_bearing) else {
                return f32::INFINITY;
            };
            let Some(bearing) = bearings
                .iter()
                .find(|row| row.source_bearing == *source_bearing)
            else {
                return f32::INFINITY;
            };
            let axis = bearing.local_axis_a.normalize_or_zero();
            if axis == Vec3::ZERO {
                return f32::INFINITY;
            }
            let anchor =
                compounds[bearing.compound_a as usize].root_translation + bearing.local_anchor_a;

            let mut total = 0.0_f32;
            let mut stack = vec![child_body];
            while let Some(body) = stack.pop() {
                let compound = &compounds[body];
                if compound.is_static {
                    return f32::INFINITY;
                }
                let properties = compound.mass_properties;
                let offset = properties.center_of_mass - anchor;
                let radial = offset - axis * offset.dot(axis);
                total += if bearing.kind.is_translational() {
                    properties.mass
                } else {
                    axis.dot(properties.inertia * axis) + properties.mass * radial.length_squared()
                };
                stack.extend(children[body].iter().copied());
            }
            if total.is_finite() && total > 0.0 {
                total
            } else {
                f32::INFINITY
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CoordinateActuation {
    pub(super) source_a_torque: f32,
    pub(super) source_a_no_load_speed: f32,
    pub(super) source_b_torque: f32,
    pub(super) source_b_no_load_speed: f32,
    pub(super) max_speed: f32,
}

#[derive(Default)]
pub(super) struct ModuleBudget {
    pub(super) controller: Option<PartId>,
    pub(super) electric_engines: u32,
    pub(super) gas_engines: u32,
    pub(super) servos: u32,
    pub(super) electric_coordinates: BTreeSet<u32>,
    pub(super) gas_coordinates: BTreeSet<u32>,
    pub(super) servo_coordinates: BTreeSet<u32>,
}

pub(super) fn validate_actuator_programs(graph: &ConstructionGraph) -> Result<(), TopologyError> {
    for (_, link) in graph.drive_links() {
        let compatible = link.program.states().iter().all(|state| {
            matches!(
                (link.actuator, state.target()),
                (ActuatorAssignment::Unpowered, _)
                    | (
                        ActuatorAssignment::Motor { .. },
                        DriveTarget::Speed(_) | DriveTarget::LinearSpeed(_)
                    )
                    | (
                        ActuatorAssignment::Servo,
                        DriveTarget::Angle(_) | DriveTarget::LinearPosition(_)
                    )
            )
        });
        if !compatible {
            return Err(TopologyError::IncompatibleActuatorProgram {
                bearing: link.bearing,
            });
        }
    }
    Ok(())
}

#[expect(clippy::too_many_lines, clippy::cast_precision_loss)]
// Graph counts are far below f32's exact-integer range in any compilable
// creation; converting them keeps the torque-sharing arithmetic readable.
pub(super) fn resolve_coordinate_actuation(
    topology: &LoopTopology,
    graph: &ConstructionGraph,
    active_gears: &[GearSelection],
) -> Result<Vec<CoordinateActuation>, TopologyError> {
    let mut modules = BTreeMap::<PartId, ModuleBudget>::new();
    let mut assignment_by_coordinate = BTreeMap::<u32, (PartId, PartId, ActuatorAssignment)>::new();

    for (link_id, link) in graph.drive_links() {
        let Some(&coordinate) = topology.bearing_coordinates.get(&link.bearing) else {
            continue;
        };
        let members = graph.machine_module(link.controller);
        let module_key = members.iter().next().copied().unwrap_or(link.controller);
        let module = modules.entry(module_key).or_insert_with(|| {
            let mut budget = ModuleBudget {
                controller: Some(link.controller),
                ..ModuleBudget::default()
            };
            for part in &members {
                match graph.part(*part) {
                    Some(PartSpec::Engine(engine)) => match engine.kind {
                        EngineKind::Electric => budget.electric_engines += 1,
                        EngineKind::Gas => budget.gas_engines += 1,
                    },
                    Some(PartSpec::Servo(_)) => budget.servos += 1,
                    _ => {}
                }
            }
            budget
        });
        if graph.reserves_motor_port(link_id, EngineKind::Electric) {
            module.electric_coordinates.insert(coordinate);
        }
        if graph.reserves_motor_port(link_id, EngineKind::Gas) {
            module.gas_coordinates.insert(coordinate);
        }
        if link.actuator.uses_servo() {
            module.servo_coordinates.insert(coordinate);
        }
        assignment_by_coordinate.entry(coordinate).or_insert((
            module_key,
            link.controller,
            link.actuator,
        ));
    }

    for module in modules.values() {
        let controller = module
            .controller
            .expect("a module budget is created from a controller link");
        let electric_required =
            u32::try_from(module.electric_coordinates.len()).expect("coordinate count fits u32");
        let electric_available = module.electric_engines * EngineKind::Electric.bearing_capacity();
        if electric_required > electric_available {
            return Err(TopologyError::InsufficientElectricPorts {
                controller,
                required: electric_required,
                available: electric_available,
            });
        }
        let gas_required =
            u32::try_from(module.gas_coordinates.len()).expect("coordinate count fits u32");
        let gas_available = module.gas_engines * EngineKind::Gas.bearing_capacity();
        if gas_required > gas_available {
            return Err(TopologyError::InsufficientGasPorts {
                controller,
                required: gas_required,
                available: gas_available,
            });
        }
        let servo_required =
            u32::try_from(module.servo_coordinates.len()).expect("coordinate count fits u32");
        if servo_required > module.servos {
            return Err(TopologyError::InsufficientServos {
                controller,
                required: servo_required,
                available: module.servos,
            });
        }
    }

    let mut result = vec![CoordinateActuation::default(); topology.tree_bearings.len()];
    for (coordinate, (module_key, controller, assignment)) in assignment_by_coordinate {
        let module = &modules[&module_key];
        let row = &mut result[coordinate as usize];
        let active_ratio = |kind| {
            active_gears
                .iter()
                .find(|gear| gear.controller == controller && gear.kind == kind)
                .map_or(Some(1.0), |gear| gear.ratio)
        };
        match assignment {
            ActuatorAssignment::Unpowered => {}
            ActuatorAssignment::Motor {
                electric_percent,
                gas_percent,
            } => {
                if electric_percent != 0
                    && let Some(ratio) = active_ratio(EngineKind::Electric)
                {
                    let consumers = module.electric_coordinates.len() as f32;
                    row.source_a_torque = module.electric_engines as f32
                        * EngineKind::Electric.stall_torque_newton_meters()
                        / consumers
                        * (f32::from(electric_percent) / 100.0)
                        * ratio;
                    row.source_a_no_load_speed =
                        crate::rpm_to_rad_s(EngineKind::Electric.no_load_rpm()) / ratio;
                    row.max_speed = row.max_speed.max(row.source_a_no_load_speed);
                }
                if gas_percent != 0
                    && let Some(ratio) = active_ratio(EngineKind::Gas)
                {
                    let consumers = module.gas_coordinates.len() as f32;
                    row.source_b_torque = module.gas_engines as f32
                        * EngineKind::Gas.stall_torque_newton_meters()
                        / consumers
                        * (f32::from(gas_percent) / 100.0)
                        * ratio;
                    row.source_b_no_load_speed =
                        crate::rpm_to_rad_s(EngineKind::Gas.no_load_rpm()) / ratio;
                    row.max_speed = row.max_speed.max(row.source_b_no_load_speed);
                }
            }
            ActuatorAssignment::Servo => {
                row.source_a_torque = ServoSpec::STALL_TORQUE_NEWTON_METERS;
                row.source_a_no_load_speed = crate::rpm_to_rad_s(ServoSpec::NO_LOAD_RPM);
                row.max_speed = row.source_a_no_load_speed;
            }
        }
    }
    for (coordinate, bearing) in topology.tree_bearings.iter().enumerate() {
        if graph
            .bearing(*bearing)
            .is_some_and(|bearing| bearing.kind.is_translational())
        {
            let row = &mut result[coordinate];
            row.source_a_torque /= crate::LINEAR_METERS_PER_RADIAN;
            row.source_b_torque /= crate::LINEAR_METERS_PER_RADIAN;
            row.source_a_no_load_speed *= crate::LINEAR_METERS_PER_RADIAN;
            row.source_b_no_load_speed *= crate::LINEAR_METERS_PER_RADIAN;
            row.max_speed *= crate::LINEAR_METERS_PER_RADIAN;
        }
    }
    Ok(result)
}

pub(super) fn resolve_coordinate_drives(
    topology: &LoopTopology,
    graph: &ConstructionGraph,
    actuation: &[CoordinateActuation],
) -> Vec<CoordinateDrive> {
    topology
        .tree_bearings
        .iter()
        .enumerate()
        .map(|(coordinate, &bearing)| {
            let kind = graph.bearing(bearing).expect("compiled bearing").kind;
            let bounds = kind.bounds();
            let passive = CoordinateDrive {
                min_angle: bounds[0],
                max_angle: bounds[1],
                ..CoordinateDrive::PASSIVE
            };
            let Some((_, link)) = graph.bearing_drive_link(bearing) else {
                return passive;
            };
            let inertia = topology
                .coordinate_axis_inertia
                .get(coordinate)
                .copied()
                .unwrap_or(f32::INFINITY);
            if !inertia.is_finite() {
                return passive;
            }
            let Some(target) = link.resolved_target(0) else {
                return passive;
            };
            let mut drive = coordinate_drive(
                target,
                link.limits,
                inertia,
                actuation.get(coordinate).copied().unwrap_or_default(),
            );
            if let Some(limits) = link.linear_limits {
                drive.min_angle = limits.minimum().max(bounds[0]);
                drive.max_angle = limits.maximum().min(bounds[1]);
                drive.max_speed = drive.max_speed.min(limits.max_speed());
                let acceleration = limits.max_force() / inertia;
                if drive.max_acceleration > acceleration {
                    let scale = acceleration / drive.max_acceleration;
                    drive.source_a_max_acceleration *= scale;
                    drive.source_b_max_acceleration *= scale;
                    drive.max_acceleration = acceleration;
                }
                drive.target_speed = drive.target_speed.clamp(-drive.max_speed, drive.max_speed);
                drive.target_angle = drive.target_angle.clamp(drive.min_angle, drive.max_angle);
            }
            drive
        })
        .collect()
}

/// Builds one GPU-bound drive row from a resolved state target.
pub(super) fn coordinate_drive(
    target: DriveTarget,
    limits: DriveLimits,
    axis_inertia: f32,
    actuation: CoordinateActuation,
) -> CoordinateDrive {
    if actuation.max_speed <= 0.0 {
        return CoordinateDrive::PASSIVE;
    }
    let max_acceleration = (actuation.source_a_torque + actuation.source_b_torque) / axis_inertia;
    let (mode, target_speed, target_angle) = match target {
        DriveTarget::Speed(speed) | DriveTarget::LinearSpeed(speed) => (
            DriveMode::Speed,
            speed.clamp(-actuation.max_speed, actuation.max_speed),
            0.0,
        ),
        DriveTarget::Angle(angle) | DriveTarget::LinearPosition(angle) => {
            (DriveMode::Angle, 0.0, angle)
        }
    };
    CoordinateDrive {
        mode,
        target_speed,
        target_angle,
        max_speed: actuation.max_speed,
        max_acceleration,
        source_a_max_acceleration: actuation.source_a_torque / axis_inertia,
        source_a_no_load_speed: actuation.source_a_no_load_speed,
        source_b_max_acceleration: actuation.source_b_torque / axis_inertia,
        source_b_no_load_speed: actuation.source_b_no_load_speed,
        min_angle: limits.min_angle(),
        max_angle: limits.max_angle(),
    }
}
