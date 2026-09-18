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
#[allow(clippy::too_many_lines)] // Keep one transactional edit and its validation together.
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
#[allow(clippy::cast_precision_loss)] // Construction counts are far below f32's exact integer range.
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

#[allow(clippy::cast_precision_loss)]
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
mod tests {
    use bevy::math::{IVec3, Vec3};
    use bevy_mosaic::ui::TextStyle;
    use mechanic_core::{
        ActuatorAssignment, BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
        ControllerSpec, CuboidSpec, DriveLinkId, DriveLinkSpec, EngineKind, FaceKind, FaceRef,
        GearboxConfig, GridRotation,
    };
    use mosaic_core::{Rect, Vector2};
    use mosaic_widgets::input::{PointerButton, PointerEventKind};

    use super::geometry;
    use super::model::{BearingSlots, EngineLaneModel, Mode, PanelEdit, StateModel};
    use crate::ui::UiIntent;
    use crate::ui::testing::{Overlay, away};
    use crate::ui::theme::typeface;

    /// A control block driving one bearing.
    fn wired() -> (ConstructionGraph, DriveLinkId) {
        let mut graph = ConstructionGraph::new();
        let cuboid = |dimensions: [u8; 3], units: IVec3| {
            CuboidSpec::new(dimensions, BuildPose::new(units, GridRotation::default()))
                .expect("test dimensions are in range")
        };
        let spawned = |outcome: BuildOutcome| match outcome {
            BuildOutcome::Spawned(part) => part,
            other => panic!("expected a spawn, got {other:?}"),
        };
        let base = spawned(
            graph
                .apply(BuildCommand::Spawn(cuboid([4, 2, 4], IVec3::new(0, 1, 0))))
                .expect("the base spawns"),
        );
        let rotor = spawned(
            graph
                .apply(BuildCommand::Spawn(cuboid([2, 2, 2], IVec3::new(0, 3, 0))))
                .expect("the rotor spawns"),
        );
        let controller = spawned(
            graph
                .apply(BuildCommand::SpawnController(ControllerSpec::new(
                    BuildPose::from_half_grid(IVec3::new(2, 5, 0), GridRotation::default()),
                )))
                .expect("the control block spawns"),
        );
        let BuildOutcome::BearingAdded(bearing) = graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(base, FaceKind::PositiveY),
                FaceRef::part(rotor, FaceKind::NegativeY),
                Vec3::new(0.0, 0.5, 0.0),
                Vec3::Y,
            )))
            .expect("the bearing is added")
        else {
            panic!("expected a bearing outcome");
        };
        graph
            .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
                controller, bearing,
            )))
            .expect("the wire is added");
        let link = graph
            .controller_links(controller)
            .next()
            .expect("the wire is there")
            .0;
        (graph, link)
    }

    /// Exercises the same graph writer as the controller's preset buttons.
    #[test]
    #[allow(clippy::too_many_lines)] // Hardware fixture plus assertions across both gear ranges.
    fn drive_preset_uses_two_gas_engines_and_the_full_gear_range() {
        use bevy::ecs::system::SystemState;
        use bevy::prelude::World;
        use mechanic_core::{DriveTarget, EngineSpec, WeldSpec};

        let (mut graph, link) = wired();
        let controller = graph.drive_link(link).unwrap().controller;
        for (position, first, second) in [
            (
                IVec3::new(2, 9, 0),
                FaceKind::PositiveY,
                FaceKind::NegativeY,
            ),
            (
                IVec3::new(6, 5, 0),
                FaceKind::PositiveX,
                FaceKind::NegativeX,
            ),
        ] {
            let BuildOutcome::Spawned(engine) = graph
                .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                    EngineKind::Gas,
                    BuildPose::from_half_grid(position, GridRotation::default()),
                )))
                .unwrap()
            else {
                panic!("engine");
            };
            graph
                .apply(BuildCommand::Weld(WeldSpec {
                    first: FaceRef::part(controller, first),
                    second: FaceRef::part(engine, second),
                }))
                .unwrap();
            let mut parent = engine;
            for _ in 0..2 {
                let spec = graph.next_transmission_spec(parent).unwrap();
                let BuildOutcome::Spawned(child) = graph
                    .apply(BuildCommand::AttachTransmission { parent, spec })
                    .unwrap()
                else {
                    panic!("transmission");
                };
                parent = child;
            }
        }
        assert_eq!(graph.actuator_inventory(controller).unwrap().gas_engines, 2);
        let mut world = World::new();
        world.insert_resource(crate::EditorGraph(graph));
        world.init_resource::<crate::EditorState>();
        world.init_resource::<crate::EditorHistory>();
        world.init_resource::<crate::AppSimulation>();
        let mut system = SystemState::<super::EditTarget>::new(&mut world);
        let intent = super::Intent {
            lane: link,
            edit: PanelEdit::ApplyPreset(super::model::Preset::Drive),
            transient: false,
        };
        for ratios in [vec![4.0, 3.0, 1.0], vec![4.0, 3.0, 0.25]] {
            world
                .resource_mut::<crate::EditorGraph>()
                .0
                .apply(BuildCommand::SetGearboxRatios {
                    controller,
                    kind: EngineKind::Gas,
                    ratios: ratios.clone(),
                })
                .unwrap();
            super::write_to(
                controller,
                &mut system.get_mut(&mut world).unwrap(),
                &intent,
            );
            let graph = &world.resource::<crate::EditorGraph>().0;
            let drive = graph.drive_link(link).unwrap();
            let top = mechanic_core::rpm_to_rad_s(EngineKind::Gas.no_load_rpm()) / ratios[2];
            assert!(
                matches!(drive.program.state(1).unwrap().target(), DriveTarget::Speed(speed) if (speed - top).abs() < 0.0001),
                "W must request {top} rad/s, got {:?}",
                drive.program
            );
            assert!(
                matches!(drive.program.state(2).unwrap().target(), DriveTarget::Speed(speed) if (speed + top * 0.7).abs() < 0.0001)
            );
            assert_eq!(
                drive
                    .program
                    .state(1)
                    .unwrap()
                    .trigger()
                    .unwrap()
                    .key()
                    .symbol(),
                'W'
            );
            assert_eq!(
                drive
                    .program
                    .state(2)
                    .unwrap()
                    .trigger()
                    .unwrap()
                    .key()
                    .symbol(),
                'S'
            );
            assert!(world.resource::<crate::EditorState>().feedback.is_none());
            graph.compile().unwrap();
        }
    }

    /// The overlay with a control block open on one wired joint.
    fn open() -> (Overlay, DriveLinkId) {
        let (graph, link) = wired();
        let controller = graph
            .drive_link(link)
            .expect("the wire is there")
            .controller;
        let overlay = Overlay::mount();
        let mut state = crate::control_panel::ControlPanelState::default();
        state.open(controller);
        overlay.handles.block.model.set(super::capture(
            &state,
            &crate::EditorGraph(graph),
            &crate::sequencer::GearboxRuntime::default(),
            false,
        ));
        overlay.settle();
        (overlay, link)
    }

    /// The overlay with a five-speed gas engine line and no joint rows.
    fn open_gas_gearbox() -> Overlay {
        let (overlay, _link) = open();
        overlay.handles.block.model.update(|model| {
            model.engine_lanes = vec![EngineLaneModel {
                kind: EngineKind::Gas,
                engine_count: 1,
                combined_stall_torque: EngineKind::Gas.stall_torque_newton_meters(),
                base_rpm: EngineKind::Gas.no_load_rpm(),
                slots: BearingSlots::new(0, EngineKind::Gas.bearing_capacity()),
                transmission_depth: Some(4),
                physical_depths: vec![4],
                mismatch: false,
                config: Some(GearboxConfig::for_depth(4, true)),
                active_gear: None,
                binding_conflict: false,
            }];
        });
        overlay.settle();
        overlay
    }

    /// Gives the joint travel limits and puts its first state on an angle,
    /// which is what brings the travel grips out.
    fn limit_travel(overlay: &Overlay, low: f32, high: f32) {
        overlay.handles.block.model.update(|model| {
            let Some(lane) = model.lanes.first_mut() else {
                return;
            };
            lane.travel = Some((low, high));
            if let Some(state) = lane.states.first_mut() {
                state.mode = Mode::Angle;
            }
        });
        overlay.settle();
    }

    /// The drive edits the panel asked for.
    fn edits(overlay: &Overlay) -> Vec<super::Intent> {
        overlay
            .intents()
            .into_iter()
            .filter_map(|intent| match intent {
                UiIntent::Drive(edit) => Some(edit),
                _ => None,
            })
            .collect()
    }

    /// The dial's box, which everything on the dial is placed from.
    fn dial_box(overlay: &Overlay) -> Rect {
        overlay
            .rects()
            .into_iter()
            .find(|(_, rect)| {
                (rect.size.width - 132.0).abs() < 0.5 && (rect.size.height - 132.0).abs() < 0.5
            })
            .expect("the dial is laid out")
            .1
    }

    #[test]
    fn a_mounted_panel_builds_a_tree_rather_than_an_empty_root() {
        let (overlay, link) = open();
        let states = overlay
            .handles
            .block
            .model
            .with(|model| model.lane(link).map(|lane| lane.states.len()))
            .expect("the joint is in the model");
        assert_eq!(states, 1, "a fresh wire holds one state");
        assert!(
            overlay.element_count() > 20,
            "the panel builds a tree; it had {} elements",
            overlay.element_count(),
        );
    }

    /// The bug this guards against: a full-bleed grouping wrapper marked
    /// `nohit` takes its whole subtree out of reach with it, and every control
    /// on the card stops responding while still looking perfectly right.
    #[test]
    fn the_controls_on_a_card_can_be_reached_by_the_pointer() {
        let (overlay, _link) = open();
        let boxes = overlay.reachable_boxes();
        for (what, width, height) in [
            ("the keycap", 46.0, 34.0),
            ("a mode switch", 26.0, 20.0),
            ("the delete button", 20.0, 20.0),
            ("the dial", 132.0, 132.0),
            ("a port", 22.0, 22.0),
            ("a preset", 34.0, 34.0),
        ] {
            assert!(
                overlay.reaches_box(&boxes, width, height),
                "{what} must take the pointer; \
                 nothing {width}×{height} was reachable anywhere in the panel",
            );
        }
    }

    #[test]
    fn the_header_close_button_requests_that_the_panel_close() {
        let (overlay, _link) = open();
        let close = overlay
            .reachable_boxes()
            .into_iter()
            .find(|rect| {
                (rect.size.width - 32.0).abs() < 0.5 && (rect.size.height - 32.0).abs() < 0.5
            })
            .expect("the close button is reachable in the header");

        overlay.click(close.center());

        assert_eq!(overlay.intents(), vec![UiIntent::CloseControlPanel]);
    }

    #[test]
    fn the_header_close_icon_is_geometrically_centred() {
        let (overlay, _link) = open();
        let tree = overlay.rects();
        let (button_index, (button_depth, button)) = tree
            .iter()
            .enumerate()
            .find(|(_, (_, rect))| {
                (rect.size.width - 32.0).abs() < 0.5 && (rect.size.height - 32.0).abs() < 0.5
            })
            .expect("the close button is in the header");
        let icon = tree[button_index + 1..]
            .iter()
            .take_while(|(depth, _)| depth > button_depth)
            .find(|(_, rect)| {
                (rect.size.width - 18.0).abs() < 0.5 && (rect.size.height - 18.0).abs() < 0.5
            })
            .expect("the close button contains its canvas")
            .1;

        assert!(away(icon.center(), button.center()) < 0.5);
    }

    #[test]
    fn the_header_keeps_all_three_capacity_stats_without_hardware() {
        let (overlay, _link) = open();
        let stats = overlay
            .rects()
            .into_iter()
            .filter(|(_, rect)| {
                (rect.size.width - 110.0).abs() < 0.5 && (rect.size.height - 38.0).abs() < 0.5
            })
            .count();

        assert_eq!(stats, 3, "electric, gas, and Servo stats stay visible");
    }

    #[test]
    fn compact_powertrain_text_fits_its_tiles_in_the_body_font() {
        let (overlay, link) = open();
        overlay.handles.block.model.update(|model| {
            let lane = model.lanes.first_mut().expect("the joint is present");
            lane.actuator = ActuatorAssignment::motor(100, 100).expect("valid percentages");
            lane.torque = 9_999.0;
        });
        let (heading, speed, torque, electric, gas) = overlay.handles.block.model.with(|model| {
            let lane = model.lane(link).expect("the joint is present");
            (
                lane.torque_label(),
                lane.speed_text(),
                lane.torque_text(),
                lane.electric_text(),
                lane.gas_text(),
            )
        });
        let heading_style = TextStyle::new(8.0)
            .family(mosaic_core::theme::typed(
                typeface.body,
                bevy_mosaic::ui::FontFamily::default,
            ))
            .weight(700)
            .letter_spacing(0.45);
        let value_style = TextStyle::new(11.0)
            .family(mosaic_core::theme::typed(
                typeface.body,
                bevy_mosaic::ui::FontFamily::default,
            ))
            .letter_spacing(-0.12);

        for text in ["ELECTRIC", "SPEED", heading] {
            assert!(
                overlay.text_width(text, &heading_style) <= super::view::CAPABILITY_TEXT_WIDTH,
                "capability heading {text:?} overflows its tile",
            );
        }
        for text in [
            speed.as_str(),
            torque.as_str(),
            electric.as_str(),
            gas.as_str(),
        ] {
            assert!(
                overlay.text_width(text, &value_style) <= super::view::CAPABILITY_TEXT_WIDTH,
                "capability value {text:?} overflows its tile",
            );
        }

        let capacity_style = TextStyle::new(9.0).family(mosaic_core::theme::typed(
            typeface.body,
            bevy_mosaic::ui::FontFamily::default,
        ));
        let capacity = BearingSlots::new(99, 99).text();
        assert!(
            overlay.text_width(&capacity, &capacity_style) <= super::view::CAPACITY_TEXT_WIDTH,
            "capacity value {capacity:?} overflows its header tile",
        );
    }

    #[test]
    fn gas_direction_split_uses_the_available_lane_width() {
        let overlay = open_gas_gearbox();
        let gear_card_rects = overlay
            .rects()
            .into_iter()
            .map(|(_, rect)| rect)
            .filter(|rect| {
                (rect.size.width - 128.0).abs() < 0.5 && (rect.size.height - 82.0).abs() < 0.5
            })
            .collect::<Vec<_>>();
        let widest_track = overlay
            .rects()
            .into_iter()
            .map(|(_, rect)| rect)
            .filter(|rect| (rect.size.height - 30.0).abs() < 0.5)
            .map(|rect| rect.size.width)
            .fold(0.0_f32, f32::max);

        assert!(
            widest_track > crate::ui::testing::VIEWPORT.width * 0.5,
            "the direction track should fill the gearbox lane; it was only {widest_track}px wide",
        );
        assert_eq!(
            gear_card_rects.len(),
            5,
            "all five ratio cards should be visible"
        );
    }

    /// The bug this guards against: a canvas with no size of its own shrinks
    /// onto the drawing inside it and pulls that drawing flush with its own
    /// corner. The dial then sits off the number at its centre by however far
    /// the sweep happened to reach — and every mark on it, the travel grips
    /// included, moves out from under the pointer with it.
    #[test]
    fn the_dial_is_drawn_from_the_corner_of_its_own_box() {
        let (overlay, _link) = open();
        let tree = overlay.rects();
        let dial = tree
            .iter()
            .position(|(_, rect)| {
                (rect.size.width - 132.0).abs() < 0.5 && (rect.size.height - 132.0).abs() < 0.5
            })
            .expect("the dial is laid out");
        let (depth, box_) = tree[dial];
        let (canvas_depth, canvas) = tree[dial + 1];
        assert_eq!(canvas_depth, depth + 1);
        assert_eq!(canvas, box_, "the drawing surface is the dial's own box");
        for (mark_depth, mark) in &tree[dial + 2..] {
            if *mark_depth <= canvas_depth {
                break;
            }
            assert_eq!(
                mark.origin, box_.origin,
                "every mark on the dial is placed from the dial's own corner",
            );
        }
    }

    /// A grip is drawn on the dial and grabbed by a box of its own, so the two
    /// have to agree about where it is.
    #[test]
    fn a_travel_grip_is_grabbed_where_it_is_drawn() {
        let (overlay, _link) = open();
        limit_travel(&overlay, -45.0, 60.0);
        let dial = dial_box(&overlay);
        // Only the boxes out on the dial's rim: an 18×18 box is a common
        // enough size that the header's own marks would otherwise count.
        let grips: Vec<Rect> = overlay
            .reachable_boxes()
            .into_iter()
            .filter(|rect| {
                (rect.size.width - 18.0).abs() < 0.5
                    && (rect.size.height - 18.0).abs() < 0.5
                    && away(rect.center(), dial.center()) < geometry::GRIP_RADIUS + 1.0
            })
            .collect();
        assert_eq!(grips.len(), 2, "both ends of the travel take the pointer");
        for (degrees, grip) in [-45.0_f32, 60.0].into_iter().zip(grips) {
            let (x, y) = geometry::polar(geometry::GRIP_RADIUS, degrees);
            let wanted = dial.origin + Vector2::new(x, y);
            let found = grip.center();
            assert!(
                (found.x - wanted.x).abs() < 0.5 && (found.y - wanted.y).abs() < 0.5,
                "the grip for {degrees}° is grabbed at {found:?}, but drawn at {wanted:?}",
            );
        }
    }

    /// Travel is a switch like any other, so its grips come and go — and an
    /// element built once outside the branch that shows it is freed the first
    /// time that branch closes.
    #[test]
    fn travel_grips_survive_being_switched_off_and_on() {
        let (overlay, _link) = open();
        for _ in 0..2 {
            limit_travel(&overlay, -45.0, 60.0);
            overlay.handles.block.model.update(|model| {
                if let Some(lane) = model.lanes.first_mut() {
                    lane.travel = None;
                }
            });
            overlay.settle();
        }
    }

    /// Dragging the dial is how a state's number is set without typing it.
    #[test]
    fn dragging_the_dial_moves_the_number_it_reads() {
        let (overlay, link) = open();
        let centre = dial_box(&overlay).center();
        overlay.drag(
            centre + Vector2::new(0.0, -40.0),
            centre + Vector2::new(40.0, 0.0),
        );
        let queued = edits(&overlay);
        assert!(
            queued.iter().any(|edit| edit.lane == link
                && matches!(edit.edit, PanelEdit::SetValue { state: 0, .. })),
            "a quarter turn round the dial sets the state's number",
        );
        assert!(
            queued.iter().all(|edit| edit.transient),
            "nothing part-way through a drag belongs in history",
        );
    }

    /// A grip sits inside the dial, which reads the same gesture as a change of
    /// value — so the grip has to keep the pointer to itself.
    #[test]
    fn dragging_a_travel_grip_moves_the_limit_and_not_the_reading() {
        let (overlay, link) = open();
        limit_travel(&overlay, -45.0, 60.0);
        let dial = dial_box(&overlay);
        let (x, y) = geometry::polar(geometry::GRIP_RADIUS, -45.0);
        let grip = dial.origin + Vector2::new(x, y);
        // Round to nine o'clock, which is a limit of -90°.
        overlay.drag(
            grip,
            dial.center() + Vector2::new(-geometry::GRIP_RADIUS, 0.0),
        );

        let queued = edits(&overlay);
        assert!(
            queued.iter().any(|edit| edit.lane == link
                && matches!(
                    edit.edit,
                    PanelEdit::SetTravel { min, max }
                        if (min + 90.0).abs() < 0.01 && (max - 60.0).abs() < 0.01
                )),
            "the grip moves the end of the travel it belongs to; got {queued:?}",
        );
        assert!(
            queued
                .iter()
                .all(|edit| !matches!(edit.edit, PanelEdit::SetValue { .. })),
            "and the dial underneath it must not read the same gesture as a value",
        );
    }

    /// Dragging out of a port draws the wire it is about to make, rather than
    /// leaving the pointer to be aimed at nothing.
    #[test]
    fn dragging_out_of_a_port_draws_the_wire_before_it_lands() {
        let (overlay, link) = open();
        overlay.handles.block.model.update(|model| {
            let Some(lane) = model.lanes.first_mut() else {
                return;
            };
            lane.states[0].key = Some('W');
            lane.states.push(StateModel {
                mode: Mode::Speed,
                value: 90.0,
                key: None,
                release: None,
                dwell: None,
            });
        });
        overlay.settle();

        let resting = overlay.element_count();
        let card = overlay
            .rects()
            .into_iter()
            .find(|(_, rect)| {
                (rect.size.width - 204.0).abs() < 0.5 && (rect.size.height - 214.0).abs() < 0.5
            })
            .expect("a card is laid out")
            .1;
        // The release port hangs off the card's top-right corner. Picked by
        // where it sits rather than by its size: a 22-pixel box is also what
        // the header's legend tiles measure.
        let corner = Vector2::new(card.origin.x + card.size.width, card.origin.y);
        let port = overlay
            .reachable_boxes()
            .into_iter()
            .find(|rect| {
                (rect.size.width - 22.0).abs() < 0.5
                    && (rect.size.height - 22.0).abs() < 0.5
                    && away(rect.center(), corner) < 2.0
            })
            .expect("the release port is reachable");
        let second = Vector2::new(
            card.center().x + geometry::NODE_W + geometry::GAP,
            card.center().y,
        );

        overlay.drag(port.center(), second);
        assert!(
            overlay.element_count() > resting,
            "the wire being dragged is drawn while the pointer holds it",
        );

        overlay.dispatch(PointerEventKind::Up(PointerButton::Primary), second);
        assert_eq!(
            overlay.element_count(),
            resting,
            "and is put away once it is let go",
        );
        let queued = edits(&overlay);
        assert!(
            queued.iter().any(|edit| edit.lane == link
                && matches!(
                    edit.edit,
                    PanelEdit::SetRelease {
                        state: 0,
                        target: Some(1)
                    }
                )),
            "letting go on a card is what wires the state to it; got {queued:?}",
        );
    }

    /// The speed capability is a stable control even while its displayed unit
    /// changes beneath it.
    #[test]
    fn the_speed_chip_survives_repeated_unit_toggles() {
        let (overlay, _link) = open();
        let chip = overlay
            .reachable_boxes()
            .into_iter()
            .find(|rect| {
                (rect.size.width - 114.0).abs() < 0.5 && (rect.size.height - 44.0).abs() < 0.5
            })
            .expect("the speed chip is reachable");

        for _ in 0..3 {
            overlay.click(chip.center());
        }

        let queued = edits(&overlay);
        assert!(
            queued
                .iter()
                .filter(|edit| matches!(edit.edit, PanelEdit::ToggleSpeedUnit))
                .count()
                >= 3,
            "clicking the hardware speed chip toggles its unit; got {queued:?}",
        );
    }

    #[test]
    fn clicking_an_unbound_keycap_arms_a_key_capture() {
        let (overlay, link) = open();
        assert_eq!(overlay.handles.block.capturing.get_untracked(), None);

        let keycap = overlay
            .reachable_boxes()
            .into_iter()
            .find(|rect| {
                (rect.size.width - 46.0).abs() < 0.5 && (rect.size.height - 34.0).abs() < 0.5
            })
            .expect("the keycap is reachable");
        overlay.click(keycap.center());

        assert_eq!(
            overlay.handles.block.capturing.get_untracked(),
            Some((link, 0)),
            "clicking an empty keycap is how a state waits for its key",
        );
    }
}
