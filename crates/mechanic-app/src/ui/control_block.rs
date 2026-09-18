//! The control block panel: one lane per driven joint.
//!
//! The construction graph stays the truth about the machine. This module is
//! the seam: it reads the graph into a [`PanelModel`] the tree renders, and
//! folds the [`Intent`]s the tree produces back into build commands. Nothing
//! in the view touches the graph, and nothing in the graph knows the panel
//! exists.
//!
//! The panel does not mount itself — [`crate::ui`] owns the tree and hangs this
//! view in it, along with every other panel.

mod geometry;
mod model;
mod view;

use bevy::prelude::*;
use mechanic_core::{
    ActuatorAssignment, BuildCommand, DriveLinkId, DriveProgram, DriveTarget, EngineKind, GearKey,
    GearKeyChord, PartId, ServoSpec,
};

use crate::control_panel::{ControlPanelState, panel_rows, set_row_commands};
use crate::sequencer::GearboxRuntime;
use crate::{AppSimulation, EditorGraph, EditorHistory, EditorSnapshot, EditorState};

pub(crate) use model::{GearboxEdit, GearboxIntent, Intent, PanelEdit, PanelModel};
pub(crate) use view::{ControlPanel, ControlPanelProps, Handles};

use model::{EngineLaneModel, HardwareModel, LaneModel, apply_edit};

/// The joint the panel is pointing out in the world, if any.
///
/// A plain resource rather than something read off the tree, so the systems
/// that draw the world do not have to be pinned to the main thread.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LocatedJoint(pub(crate) Option<DriveLinkId>);

/// The construction the panel edits, and everything one edit touches.
#[derive(bevy::ecs::system::SystemParam)]
pub(crate) struct EditTarget<'w> {
    pub(super) graph: ResMut<'w, EditorGraph>,
    pub(super) editor: ResMut<'w, EditorState>,
    pub(super) history: ResMut<'w, EditorHistory>,
    simulation: Res<'w, AppSimulation>,
}

/// Writes one intent to every wire behind the joint it names.
///
/// This is an undoable edit except part-way through a drag: a drag writes on
/// every pointer move, and only the move that ends it belongs in history. A
/// live World also marks the uploaded drive rows dirty without pausing physics.
pub(crate) fn write_joint(panel: &mut ControlPanelState, target: &mut EditTarget, intent: &Intent) {
    if matches!(intent.edit, PanelEdit::ToggleSpeedUnit) {
        panel.toggle_speed_unit();
        return;
    }
    let Some(controller) = panel.controller() else {
        return;
    };
    write_to(controller, target, intent);
}

/// Writes one persistent engine-lane setting transactionally.
pub(crate) fn write_gearbox(
    panel: &ControlPanelState,
    target: &mut EditTarget,
    intent: &GearboxIntent,
) {
    let Some(controller) = panel.controller() else {
        return;
    };
    let command = match &intent.edit {
        GearboxEdit::Mode(mode) => BuildCommand::SetGearboxMode {
            controller,
            kind: intent.kind,
            mode: *mode,
        },
        GearboxEdit::Ratios(ratios) => BuildCommand::SetGearboxRatios {
            controller,
            kind: intent.kind,
            ratios: ratios.clone(),
        },
        GearboxEdit::Bindings { up, down } => BuildCommand::SetGearboxBindings {
            controller,
            kind: intent.kind,
            up: *up,
            down: *down,
        },
        GearboxEdit::ReverseGears(reverse_gears) => {
            if intent.kind != EngineKind::Gas {
                return;
            }
            BuildCommand::SetGasDivider {
                controller,
                reverse_gears: *reverse_gears,
            }
        }
    };
    let previous = EditorSnapshot::capture(&target.graph.0, &target.editor);
    match target.graph.0.apply(command) {
        Ok(_) => {
            if target.simulation.is_running() {
                target.editor.drive_rows_dirty = true;
            }
            if !intent.transient {
                target.history.commit(previous);
            }
        }
        Err(error) => target.editor.feedback = Some(error.to_string()),
    }
}

/// The half of [`write_joint`] that already knows which control block is open.
#[expect(
    clippy::too_many_lines,
    reason = "keep one transactional edit and its validation together"
)]
fn write_to(controller: PartId, target: &mut EditTarget, intent: &Intent) {
    let rows = panel_rows(&target.graph.0, controller);
    let Some(row) = rows.iter().find(|row| row.links.contains(&intent.lane)) else {
        return;
    };
    let Some(spec) = target.graph.0.drive_link(row.primary).copied() else {
        return;
    };
    let inventory = target
        .graph
        .0
        .actuator_inventory(controller)
        .unwrap_or_default();
    let mut linear_limits = spec.linear_limits;
    let physical_bounds = target
        .graph
        .0
        .bearing(spec.bearing)
        .expect("live joint")
        .kind
        .bounds();
    let mut actuator = spec.actuator;
    let edited = match intent.edit {
        PanelEdit::CycleActuator => {
            actuator = match actuator {
                ActuatorAssignment::Unpowered => default_motor(inventory).unwrap_or({
                    if inventory.servos != 0 {
                        ActuatorAssignment::Servo
                    } else {
                        ActuatorAssignment::Unpowered
                    }
                }),
                ActuatorAssignment::Motor { .. } if inventory.servos != 0 => {
                    ActuatorAssignment::Servo
                }
                ActuatorAssignment::Motor { .. } | ActuatorAssignment::Servo => {
                    ActuatorAssignment::Unpowered
                }
            };
            Some((spec.limits, spec.program, spec.name))
        }
        PanelEdit::CycleElectric => {
            let next = stepped_percent(actuator.electric_percent());
            actuator = ActuatorAssignment::motor(next, actuator.gas_percent())
                .expect("stepped percentages are valid");
            Some((spec.limits, spec.program, spec.name))
        }
        PanelEdit::CycleGas => {
            let next = stepped_percent(actuator.gas_percent());
            actuator = ActuatorAssignment::motor(actuator.electric_percent(), next)
                .expect("stepped percentages are valid");
            Some((spec.limits, spec.program, spec.name))
        }
        PanelEdit::ToggleSpeedUnit => unreachable!("handled before locating the row"),
        _ => {
            if let Some(linear) = linear_limits {
                model::apply_linear_edit(
                    spec.limits,
                    linear,
                    spec.program,
                    spec.name,
                    physical_bounds,
                    &intent.edit,
                )
                .map(|(limits, linear, program, name)| {
                    linear_limits = Some(linear);
                    (limits, program, name)
                })
            } else {
                apply_edit(spec.limits, spec.program, spec.name, &intent.edit)
            }
        }
    };
    let Some((mut limits, mut program, name)) = edited else {
        return;
    };
    if let PanelEdit::ApplyPreset(preset) = intent.edit {
        actuator = match preset {
            model::Preset::Steer => ActuatorAssignment::Servo,
            model::Preset::Drive | model::Preset::Spin => match actuator {
                motor @ ActuatorAssignment::Motor { .. } => motor,
                ActuatorAssignment::Unpowered | ActuatorAssignment::Servo => {
                    default_motor(inventory).unwrap_or(ActuatorAssignment::Unpowered)
                }
            },
        };
    }
    let (mut hardware_speed, _) = actuator_capability(actuator, inventory);
    if matches!(
        intent.edit,
        PanelEdit::ApplyPreset(model::Preset::Drive | model::Preset::Spin)
    ) {
        // Author an output-speed request that still reaches the drivetrain's
        // ceiling after an upshift. Runtime rows apply the active gear's limit.
        hardware_speed =
            preset_motor_speed(&target.graph.0, controller, actuator).unwrap_or(hardware_speed);
    }
    if hardware_speed > 0.0
        && let Ok(hardware_limits) = limits.with_max_speed(hardware_speed)
    {
        limits = hardware_limits;
    }
    // Presets use the active hardware's actual ceiling, including the 70%
    // reverse speed in the Drive preset.
    if let PanelEdit::ApplyPreset(preset) = intent.edit
        && linear_limits.is_none()
        && let Some((next_limits, next_program, _)) =
            apply_edit(limits, program, name, &PanelEdit::ApplyPreset(preset))
    {
        limits = next_limits;
        program = next_program;
    }
    let program = compatible_program(program, actuator);
    let mut commands: Vec<BuildCommand> = row
        .links
        .iter()
        .filter_map(|&link| {
            linear_limits.map(|limits| BuildCommand::SetLinearDriveLimits { link, limits })
        })
        .collect();
    commands.extend(set_row_commands(row, limits, program, name, actuator));
    let previous = EditorSnapshot::capture(&target.graph.0, &target.editor);
    let mut staged = target.graph.0.clone();
    match staged.apply_batch(commands) {
        Ok(_) => {
            if let Some(error) = capacity_error(&staged, controller) {
                target.editor.feedback = Some(error);
                return;
            }
            target.graph.0 = staged;
            if target.simulation.is_running() {
                target.editor.drive_rows_dirty = true;
            }
            if !intent.transient {
                target.history.commit(previous);
            }
            target.editor.construction_mesh_dirty = true;
        }
        Err(error) => target.editor.feedback = Some(error.to_string()),
    }
}

fn default_motor(inventory: mechanic_core::ActuatorInventory) -> Option<ActuatorAssignment> {
    if inventory.electric_engines != 0 {
        ActuatorAssignment::motor(100, 0).ok()
    } else if inventory.gas_engines != 0 {
        ActuatorAssignment::motor(0, 100).ok()
    } else {
        None
    }
}

fn preset_motor_speed(
    graph: &mechanic_core::ConstructionGraph,
    controller: PartId,
    actuator: ActuatorAssignment,
) -> Option<f32> {
    [EngineKind::Gas, EngineKind::Electric]
        .into_iter()
        .filter(|kind| match kind {
            EngineKind::Gas => actuator.uses_gas(),
            EngineKind::Electric => actuator.uses_electric(),
        })
        .filter_map(|kind| {
            let config = graph.gearbox_config(controller, kind).ok()?;
            let ratio = config.ratios().iter().copied().reduce(f32::min)?;
            Some(mechanic_core::rpm_to_rad_s(kind.no_load_rpm()) / ratio)
        })
        .reduce(f32::max)
}

const fn stepped_percent(current: u8) -> u8 {
    if current >= 100 { 0 } else { current + 25 }
}

fn compatible_program(program: DriveProgram, actuator: ActuatorAssignment) -> DriveProgram {
    let mut result = program;
    for index in 0..program.len() {
        let Ok(index) = u8::try_from(index) else {
            break;
        };
        let Some(state) = result.state(index) else {
            break;
        };
        let replacement = match (actuator, state.target()) {
            (ActuatorAssignment::Motor { .. }, DriveTarget::LinearPosition(_)) => {
                Some(DriveTarget::LinearSpeed(0.0))
            }
            (ActuatorAssignment::Servo, DriveTarget::LinearSpeed(_)) => {
                Some(DriveTarget::LinearPosition(0.0))
            }
            (ActuatorAssignment::Motor { .. }, DriveTarget::Angle(_)) => {
                Some(DriveTarget::Speed(0.0))
            }
            (ActuatorAssignment::Servo, DriveTarget::Speed(_)) => Some(DriveTarget::Angle(0.0)),
            _ => None,
        };
        if let Some(target) = replacement
            && let Ok(state) = state.with_target(target)
            && let Ok(next) = result.with_state(index, state)
        {
            result = next;
        }
    }
    result
}

fn capacity_error(graph: &mechanic_core::ConstructionGraph, controller: PartId) -> Option<String> {
    let inventory = graph.actuator_inventory(controller)?;
    if inventory.electric_joints > inventory.electric_capacity() {
        return Some(format!(
            "Electric ports full: {} assigned, {} available",
            inventory.electric_joints,
            inventory.electric_capacity()
        ));
    }
    if inventory.gas_joints > inventory.gas_capacity() {
        return Some(format!(
            "Gas ports full: {} assigned, {} available",
            inventory.gas_joints,
            inventory.gas_capacity()
        ));
    }
    (inventory.servo_joints > inventory.servo_capacity()).then(|| {
        format!(
            "Servo ports full: {} assigned, {} available",
            inventory.servo_joints,
            inventory.servo_capacity()
        )
    })
}

/// Reads the open control block's wires into what the panel draws.
#[expect(
    clippy::cast_precision_loss,
    reason = "construction counts are far below f32's exact integer range"
)]
pub(crate) fn capture(
    panel: &ControlPanelState,
    graph: &EditorGraph,
    gearboxes: &GearboxRuntime,
    gameplay_binding_conflict: bool,
) -> PanelModel {
    let Some(controller) = panel.controller() else {
        return PanelModel::default();
    };
    let inventory = graph.0.actuator_inventory(controller).unwrap_or_default();
    let lanes: Vec<LaneModel> = panel_rows(&graph.0, controller)
        .iter()
        .enumerate()
        .filter_map(|(index, row)| {
            let spec = graph.0.drive_link(row.primary)?;
            let (max_speed, torque) = actuator_capability(spec.actuator, inventory);
            let lane = LaneModel::capture(
                row.primary,
                index + 1,
                spec.limits,
                &spec.program,
                &spec.name,
                spec.actuator,
                panel.speed_unit(),
                max_speed,
                torque,
            );
            Some(if let Some(limits) = spec.linear_limits {
                lane.with_linear_limits(limits, graph.0.bearing(spec.bearing)?.kind.bounds())
            } else {
                lane
            })
        })
        .collect();
    let mut engine_lanes: Vec<EngineLaneModel> = [EngineKind::Electric, EngineKind::Gas]
        .into_iter()
        .filter_map(|kind| {
            let (engine_count, slots, transmission_depth, mismatch) = match kind {
                EngineKind::Electric => (
                    inventory.electric_engines,
                    model::BearingSlots::new(
                        inventory.electric_joints,
                        inventory.electric_capacity(),
                    ),
                    inventory.electric_transmission_depth,
                    inventory.electric_transmission_mismatch,
                ),
                EngineKind::Gas => (
                    inventory.gas_engines,
                    model::BearingSlots::new(inventory.gas_joints, inventory.gas_capacity()),
                    inventory.gas_transmission_depth,
                    inventory.gas_transmission_mismatch,
                ),
            };
            (engine_count != 0).then(|| EngineLaneModel {
                kind,
                engine_count,
                combined_stall_torque: engine_count as f32 * kind.stall_torque_newton_meters(),
                base_rpm: kind.no_load_rpm(),
                slots,
                transmission_depth,
                physical_depths: graph
                    .0
                    .transmission_depths(controller, kind)
                    .unwrap_or_default(),
                mismatch,
                config: graph.0.gearbox_config(controller, kind).ok(),
                active_gear: gearboxes.active_gear(controller, kind),
                binding_conflict: false,
            })
        })
        .collect();
    let gearbox_chords = engine_lanes
        .iter()
        .filter_map(|lane| lane.config.as_ref())
        .flat_map(|config| [config.gear_up(), config.gear_down()])
        .collect::<Vec<_>>();
    let joint_keys = lanes
        .iter()
        .flat_map(|lane| lane.states.iter().filter_map(|state| state.key))
        .collect::<Vec<_>>();
    for lane in &mut engine_lanes {
        let Some(config) = lane.config.as_ref() else {
            continue;
        };
        lane.binding_conflict = [config.gear_up(), config.gear_down()]
            .into_iter()
            .any(|chord| {
                gearbox_chords
                    .iter()
                    .filter(|candidate| **candidate == chord)
                    .count()
                    > 1
                    || chord_symbol(chord).is_some_and(|symbol| joint_keys.contains(&symbol))
            });
    }
    PanelModel {
        open: true,
        lanes,
        engine_lanes,
        hardware: HardwareModel::from(inventory),
        gameplay_binding_conflict,
    }
}

fn chord_symbol(chord: GearKeyChord) -> Option<char> {
    if chord.shift || chord.control || chord.alt || chord.super_key {
        return None;
    }
    match chord.key {
        GearKey::Letter(symbol) => Some(symbol),
        GearKey::Digit(digit) => char::from_digit(u32::from(digit), 10),
        GearKey::Space
        | GearKey::ArrowUp
        | GearKey::ArrowDown
        | GearKey::ArrowLeft
        | GearKey::ArrowRight
        | GearKey::PageUp
        | GearKey::PageDown => None,
    }
}

#[expect(clippy::cast_precision_loss)]
// Editor-scale part and joint counts remain far below f32's exact integer
// range, and the result is display/physics scalar data.
fn actuator_capability(
    actuator: ActuatorAssignment,
    inventory: mechanic_core::ActuatorInventory,
) -> (f32, f32) {
    let rpm_to_rad_s = mechanic_core::rpm_to_rad_s;
    match actuator {
        ActuatorAssignment::Unpowered => (0.0, 0.0),
        ActuatorAssignment::Servo => (
            rpm_to_rad_s(ServoSpec::NO_LOAD_RPM),
            ServoSpec::STALL_TORQUE_NEWTON_METERS,
        ),
        ActuatorAssignment::Motor {
            electric_percent,
            gas_percent,
        } => {
            let electric = if electric_percent == 0 || inventory.electric_joints == 0 {
                0.0
            } else {
                inventory.electric_engines as f32
                    * EngineKind::Electric.stall_torque_newton_meters()
                    / inventory.electric_joints as f32
                    * (f32::from(electric_percent) / 100.0)
            };
            let gas = if gas_percent == 0 || inventory.gas_joints == 0 {
                0.0
            } else {
                inventory.gas_engines as f32 * EngineKind::Gas.stall_torque_newton_meters()
                    / inventory.gas_joints as f32
                    * (f32::from(gas_percent) / 100.0)
            };
            let rpm = if gas_percent != 0 {
                EngineKind::Gas.no_load_rpm()
            } else if electric_percent != 0 {
                EngineKind::Electric.no_load_rpm()
            } else {
                0.0
            };
            (rpm_to_rad_s(rpm), electric + gas)
        }
    }
}

/// Binds the next key pressed while a keycap is waiting for one.
///
/// The panel owns the whole keyboard while it is open, so this reads the raw
/// key rather than going through the tree: nothing else may act on the press
/// that binds a state.
pub(crate) fn capture_key(handles: &Handles, keyboard: &ButtonInput<KeyCode>) {
    if keyboard.just_pressed(KeyCode::Escape) {
        handles.capturing.set(None);
        handles.gearbox_capturing.set(None);
        return;
    }
    if let Some((kind, up)) = handles.gearbox_capturing.get_untracked() {
        for pressed in keyboard.get_just_pressed() {
            let Some(key) = crate::sequencer::gear_key(*pressed) else {
                continue;
            };
            let chord = mechanic_core::GearKeyChord {
                key,
                shift: keyboard.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]),
                control: keyboard.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]),
                alt: keyboard.any_pressed([KeyCode::AltLeft, KeyCode::AltRight]),
                super_key: keyboard.any_pressed([KeyCode::SuperLeft, KeyCode::SuperRight]),
            };
            let Some((old_up, old_down)) = handles.model.with(|panel| {
                let config = panel.engine_lane(kind)?.config.as_ref()?;
                Some((config.gear_up(), config.gear_down()))
            }) else {
                handles.gearbox_capturing.set(None);
                return;
            };
            let (gear_up, gear_down) = if up {
                (chord, old_down)
            } else {
                (old_up, chord)
            };
            handles.gearbox(
                kind,
                GearboxEdit::Bindings {
                    up: gear_up,
                    down: gear_down,
                },
            );
            handles.gearbox_capturing.set(None);
            return;
        }
        return;
    }
    let Some((link, slot)) = handles.capturing.get_untracked() else {
        return;
    };
    for pressed in keyboard.get_just_pressed() {
        let Some(key) = crate::sequencer::drive_key(*pressed) else {
            continue;
        };
        handles.edit(
            link,
            PanelEdit::BindKey {
                state: slot,
                key: key.symbol(),
            },
        );
        handles.capturing.set(None);
        return;
    }
}

#[cfg(test)]
mod tests;
