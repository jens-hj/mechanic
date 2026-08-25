//! Runs each driven bearing's state program while the simulation ticks.
//!
//! A row advances on two things only: a key the player pressed or let go, and
//! elapsed simulated time. Time is counted in dispatched physics ticks rather
//! than frames, so a paused simulation freezes every dwell and a slow frame
//! never skips one.

use std::collections::BTreeMap;

use bevy::prelude::*;
use mechanic_core::{
    CompiledCreation, ConstructionGraph, DriveKey, DriveLinkId, DriveProgram, DriveRelease,
    DriveTarget, EngineKind, GearKey, GearKeyChord, GearSelection, PartSpec, ShiftMode,
};
use mechanic_gpu::{DRIVE_MODE_ANGLE, DRIVE_MODE_SPEED, FIXED_DT_SECONDS, GpuMechanismDrive};

/// Which of a program's keys are down, and which went down this frame.
#[derive(Clone, Debug, Default)]
pub(crate) struct DriveKeyState {
    held: Vec<DriveKey>,
    pressed: Vec<DriveKey>,
}

impl DriveKeyState {
    /// Reads the bound keys from the keyboard.
    ///
    /// Returns an empty state while another system owns the keyboard, so
    /// typing in the control panel never drives a machine.
    pub(crate) fn from_keyboard(keyboard: &ButtonInput<KeyCode>, blocked: bool) -> Self {
        if blocked {
            return Self::default();
        }
        let mut state = Self::default();
        for key in keyboard.get_pressed() {
            if let Some(bound) = drive_key(*key) {
                state.held.push(bound);
            }
        }
        for key in keyboard.get_just_pressed() {
            if let Some(bound) = drive_key(*key) {
                state.pressed.push(bound);
            }
        }
        state
    }

    /// Whether the key is down right now.
    fn is_held(&self, key: DriveKey) -> bool {
        self.held.contains(&key)
    }

    /// Whether the key went down this frame.
    fn is_pressed(&self, key: DriveKey) -> bool {
        self.pressed.contains(&key)
    }
}

/// Maps a physical key to its drive binding, when one exists.
///
/// `E` is deliberately absent: it opens the control panel, and one key must not
/// both drive a machine and open the window used to program it.
pub(crate) fn drive_key(key: KeyCode) -> Option<DriveKey> {
    let symbol = match key {
        KeyCode::KeyA => 'A',
        KeyCode::KeyB => 'B',
        KeyCode::KeyC => 'C',
        KeyCode::KeyD => 'D',
        KeyCode::KeyF => 'F',
        KeyCode::KeyG => 'G',
        KeyCode::KeyH => 'H',
        KeyCode::KeyI => 'I',
        KeyCode::KeyJ => 'J',
        KeyCode::KeyK => 'K',
        KeyCode::KeyL => 'L',
        KeyCode::KeyM => 'M',
        KeyCode::KeyN => 'N',
        KeyCode::KeyO => 'O',
        KeyCode::KeyP => 'P',
        KeyCode::KeyQ => 'Q',
        KeyCode::KeyR => 'R',
        KeyCode::KeyS => 'S',
        KeyCode::KeyT => 'T',
        KeyCode::KeyU => 'U',
        KeyCode::KeyV => 'V',
        KeyCode::KeyW => 'W',
        KeyCode::KeyX => 'X',
        KeyCode::KeyY => 'Y',
        KeyCode::KeyZ => 'Z',
        KeyCode::Digit0 => '0',
        KeyCode::Digit1 => '1',
        KeyCode::Digit2 => '2',
        KeyCode::Digit3 => '3',
        KeyCode::Digit4 => '4',
        KeyCode::Digit5 => '5',
        KeyCode::Digit6 => '6',
        KeyCode::Digit7 => '7',
        KeyCode::Digit8 => '8',
        KeyCode::Digit9 => '9',
        _ => return None,
    };
    DriveKey::new(symbol)
}

/// Maps every main key accepted by a gearbox binding.
pub(crate) fn gear_key(key: KeyCode) -> Option<GearKey> {
    let symbol = match key {
        KeyCode::KeyA => Some('A'),
        KeyCode::KeyB => Some('B'),
        KeyCode::KeyC => Some('C'),
        KeyCode::KeyD => Some('D'),
        KeyCode::KeyE => Some('E'),
        KeyCode::KeyF => Some('F'),
        KeyCode::KeyG => Some('G'),
        KeyCode::KeyH => Some('H'),
        KeyCode::KeyI => Some('I'),
        KeyCode::KeyJ => Some('J'),
        KeyCode::KeyK => Some('K'),
        KeyCode::KeyL => Some('L'),
        KeyCode::KeyM => Some('M'),
        KeyCode::KeyN => Some('N'),
        KeyCode::KeyO => Some('O'),
        KeyCode::KeyP => Some('P'),
        KeyCode::KeyQ => Some('Q'),
        KeyCode::KeyR => Some('R'),
        KeyCode::KeyS => Some('S'),
        KeyCode::KeyT => Some('T'),
        KeyCode::KeyU => Some('U'),
        KeyCode::KeyV => Some('V'),
        KeyCode::KeyW => Some('W'),
        KeyCode::KeyX => Some('X'),
        KeyCode::KeyY => Some('Y'),
        KeyCode::KeyZ => Some('Z'),
        KeyCode::Digit0 => Some('0'),
        KeyCode::Digit1 => Some('1'),
        KeyCode::Digit2 => Some('2'),
        KeyCode::Digit3 => Some('3'),
        KeyCode::Digit4 => Some('4'),
        KeyCode::Digit5 => Some('5'),
        KeyCode::Digit6 => Some('6'),
        KeyCode::Digit7 => Some('7'),
        KeyCode::Digit8 => Some('8'),
        KeyCode::Digit9 => Some('9'),
        _ => None,
    };
    symbol.and_then(GearKey::from_char).or(match key {
        KeyCode::Space => Some(GearKey::Space),
        KeyCode::ArrowUp => Some(GearKey::ArrowUp),
        KeyCode::ArrowDown => Some(GearKey::ArrowDown),
        KeyCode::ArrowLeft => Some(GearKey::ArrowLeft),
        KeyCode::ArrowRight => Some(GearKey::ArrowRight),
        KeyCode::PageUp => Some(GearKey::PageUp),
        KeyCode::PageDown => Some(GearKey::PageDown),
        _ => None,
    })
}

/// Where one bearing currently sits in its program.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RowCursor {
    /// State currently active.
    pub(crate) active: u8,
    /// Tick the active state was entered on.
    pub(crate) entered_tick: u64,
}

/// One driven bearing's live position in its program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SequencerRow {
    /// Wire this row runs.
    pub(crate) link: DriveLinkId,
    /// Mechanism coordinate the wire's bearing moves.
    pub(crate) coordinate: u32,
    /// Where the bearing sits in its program.
    pub(crate) cursor: RowCursor,
}

/// Live state of every driven bearing in the running simulation.
#[derive(Resource, Default)]
pub(crate) struct DriveSequencer {
    rows: Vec<SequencerRow>,
    started: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GearboxRow {
    controller: mechanic_core::PartId,
    kind: EngineKind,
    gear: Option<usize>,
    pending: Option<usize>,
    last_shift_tick: u64,
}

/// Transient selected gears. Persistent ratios and bindings remain graph-owned.
#[derive(Resource, Default)]
pub(crate) struct GearboxRuntime {
    rows: Vec<GearboxRow>,
    started: bool,
}

impl GearboxRuntime {
    pub(crate) fn start(&mut self, graph: &ConstructionGraph, sequencer: &DriveSequencer) {
        self.rows.clear();
        for (controller, spec) in graph.parts() {
            if !matches!(spec, PartSpec::Controller(_)) {
                continue;
            }
            for kind in [EngineKind::Electric, EngineKind::Gas] {
                let Ok(config) = graph.gearbox_config(controller, kind) else {
                    continue;
                };
                let requested_sign = dominant_request_sign(graph, sequencer, controller, kind);
                let gear = if config.mode() == ShiftMode::Manual {
                    if kind == EngineKind::Gas {
                        let first_forward = usize::from(config.reverse_gears());
                        (first_forward < config.ratios().len())
                            .then_some(first_forward)
                            .or(Some(0))
                    } else {
                        Some(0)
                    }
                } else {
                    initial_gear(&config, kind, requested_sign)
                };
                self.rows.push(GearboxRow {
                    controller,
                    kind,
                    gear,
                    pending: None,
                    last_shift_tick: 0,
                });
            }
        }
        self.started = true;
    }

    pub(crate) fn stop(&mut self) {
        self.rows.clear();
        self.started = false;
    }

    pub(crate) fn active_gear(
        &self,
        controller: mechanic_core::PartId,
        kind: EngineKind,
    ) -> Option<usize> {
        self.rows
            .iter()
            .find(|row| row.controller == controller && row.kind == kind)
            .and_then(|row| row.gear)
    }

    /// Applies every matching manual binding. Duplicate chords intentionally all fire.
    #[allow(clippy::too_many_arguments)] // Runtime inputs stay explicit at the simulation boundary.
    pub(crate) fn step(
        &mut self,
        graph: &ConstructionGraph,
        sequencer: &DriveSequencer,
        keyboard: &ButtonInput<KeyCode>,
        keyboard_controller: Option<mechanic_core::PartId>,
        tick: u64,
        measured_speeds: &[(mechanic_core::PartId, EngineKind, f32)],
        paused: bool,
    ) -> bool {
        let mut changed = false;
        for row in &mut self.rows {
            let Ok(config) = graph.gearbox_config(row.controller, row.kind) else {
                continue;
            };
            let measured_speed = measured_speeds
                .iter()
                .find(|(controller, kind, _)| *controller == row.controller && *kind == row.kind)
                .map_or(0.0, |(_, _, speed)| *speed);
            if let Some(destination) = row.pending {
                if paused {
                    continue;
                }
                if reversal_is_safe(row.kind, &config, destination, measured_speed) {
                    row.gear = Some(destination);
                    row.pending = None;
                    row.last_shift_tick = tick;
                    changed = true;
                }
                continue;
            }
            if config.mode() == ShiftMode::Auto {
                if paused {
                    continue;
                }
                let requested_sign =
                    dominant_request_sign(graph, sequencer, row.controller, row.kind);
                let requested = initial_gear(&config, row.kind, requested_sign);
                if requested.is_none() {
                    if row.gear.take().is_some() {
                        changed = true;
                    }
                    continue;
                }
                let requested = requested.expect("the missing direction was handled");
                if row.kind == EngineKind::Gas
                    && row.gear.is_some_and(|current| {
                        gear_bank(&config, current) != gear_bank(&config, requested)
                    })
                {
                    row.gear = None;
                    row.pending = Some(requested);
                    changed = true;
                    continue;
                }
                if row.gear.is_none() {
                    row.gear = Some(requested);
                    row.last_shift_tick = tick;
                    changed = true;
                    continue;
                }
                if tick.saturating_sub(row.last_shift_tick) < 21 {
                    continue;
                }
                let current = row.gear.expect("the missing gear was handled");
                let next = automatic_shift_destination(row.kind, &config, current, measured_speed);
                if next != current {
                    row.gear = Some(next);
                    row.last_shift_tick = tick;
                    changed = true;
                }
                continue;
            }
            if keyboard_controller != Some(row.controller) {
                continue;
            }
            let delta = i8::from(chord_just_pressed(keyboard, config.gear_up()))
                - i8::from(chord_just_pressed(keyboard, config.gear_down()));
            if delta == 0 {
                continue;
            }
            let current = row.gear.unwrap_or(0);
            let maximum = config.ratios().len().saturating_sub(1);
            let next = if delta > 0 {
                current.saturating_add(1).min(maximum)
            } else {
                current.saturating_sub(1)
            };
            if row.gear != Some(next) {
                if row.kind == EngineKind::Gas
                    && gear_bank(&config, current) != gear_bank(&config, next)
                {
                    row.gear = None;
                    row.pending = Some(next);
                    if reversal_is_safe(row.kind, &config, next, measured_speed) {
                        row.gear = Some(next);
                        row.pending = None;
                    }
                } else {
                    row.gear = Some(next);
                }
                row.last_shift_tick = tick;
                changed = true;
            }
        }
        changed
    }

    pub(crate) fn selections(&self, graph: &ConstructionGraph) -> Vec<GearSelection> {
        self.rows
            .iter()
            .map(|row| GearSelection {
                controller: row.controller,
                kind: row.kind,
                ratio: row.gear.and_then(|gear| {
                    graph
                        .gearbox_config(row.controller, row.kind)
                        .ok()?
                        .ratios()
                        .get(gear)
                        .copied()
                }),
            })
            .collect()
    }

    fn gas_direction(
        &self,
        graph: &ConstructionGraph,
        controller: mechanic_core::PartId,
    ) -> Option<i8> {
        let row = self
            .rows
            .iter()
            .find(|row| row.controller == controller && row.kind == EngineKind::Gas)?;
        let gear = row.gear?;
        let config = graph.gearbox_config(controller, EngineKind::Gas).ok()?;
        Some(gear_bank(&config, gear))
    }
}

fn gear_bank(config: &mechanic_core::GearboxConfig, gear: usize) -> i8 {
    if gear < usize::from(config.reverse_gears()) {
        -1
    } else {
        1
    }
}

fn gear_bank_range(
    kind: EngineKind,
    config: &mechanic_core::GearboxConfig,
    gear: usize,
) -> (usize, usize) {
    if kind == EngineKind::Electric {
        return (0, config.ratios().len().saturating_sub(1));
    }
    let divider = usize::from(config.reverse_gears());
    if gear < divider {
        (0, divider.saturating_sub(1))
    } else {
        (divider, config.ratios().len().saturating_sub(1))
    }
}

fn automatic_shift_destination(
    kind: EngineKind,
    config: &mechanic_core::GearboxConfig,
    current: usize,
    measured_speed: f32,
) -> usize {
    let ratio = config.ratios()[current];
    let engine_rpm = measured_speed.abs() * ratio * 60.0 / core::f32::consts::TAU;
    let (first, last) = gear_bank_range(kind, config, current);
    if engine_rpm >= 0.85 * kind.no_load_rpm() && current < last {
        current + 1
    } else if engine_rpm <= 0.40 * kind.no_load_rpm() && current > first {
        current - 1
    } else {
        current
    }
}

fn reversal_is_safe(
    kind: EngineKind,
    config: &mechanic_core::GearboxConfig,
    destination: usize,
    measured_speed: f32,
) -> bool {
    if kind != EngineKind::Gas {
        return true;
    }
    let output_speed =
        kind.no_load_rpm() * core::f32::consts::TAU / 60.0 / config.ratios()[destination];
    measured_speed.abs() < 0.05_f32.max(output_speed * 0.05)
}

fn initial_gear(
    config: &mechanic_core::GearboxConfig,
    kind: EngineKind,
    requested_sign: f32,
) -> Option<usize> {
    if kind == EngineKind::Electric {
        return Some(0);
    }
    let divider = usize::from(config.reverse_gears());
    if requested_sign < 0.0 {
        (divider != 0).then_some(0)
    } else {
        (divider < config.ratios().len()).then_some(divider)
    }
}

fn dominant_request_sign(
    graph: &ConstructionGraph,
    sequencer: &DriveSequencer,
    controller: mechanic_core::PartId,
    kind: EngineKind,
) -> f32 {
    sequencer
        .rows()
        .iter()
        .filter_map(|row| {
            let spec = graph.drive_link(row.link)?;
            if spec.controller != controller
                || match kind {
                    EngineKind::Electric => !spec.actuator.uses_electric(),
                    EngineKind::Gas => !spec.actuator.uses_gas(),
                }
            {
                return None;
            }
            spec.resolved_target(row.cursor.active)?.speed()
        })
        .max_by(|left, right| left.abs().total_cmp(&right.abs()))
        .unwrap_or(0.0)
        .signum()
}

fn chord_just_pressed(keyboard: &ButtonInput<KeyCode>, chord: GearKeyChord) -> bool {
    let key = match chord.key {
        GearKey::Letter(letter) => match letter {
            'A' => KeyCode::KeyA,
            'B' => KeyCode::KeyB,
            'C' => KeyCode::KeyC,
            'D' => KeyCode::KeyD,
            'E' => KeyCode::KeyE,
            'F' => KeyCode::KeyF,
            'G' => KeyCode::KeyG,
            'H' => KeyCode::KeyH,
            'I' => KeyCode::KeyI,
            'J' => KeyCode::KeyJ,
            'K' => KeyCode::KeyK,
            'L' => KeyCode::KeyL,
            'M' => KeyCode::KeyM,
            'N' => KeyCode::KeyN,
            'O' => KeyCode::KeyO,
            'P' => KeyCode::KeyP,
            'Q' => KeyCode::KeyQ,
            'R' => KeyCode::KeyR,
            'S' => KeyCode::KeyS,
            'T' => KeyCode::KeyT,
            'U' => KeyCode::KeyU,
            'V' => KeyCode::KeyV,
            'W' => KeyCode::KeyW,
            'X' => KeyCode::KeyX,
            'Y' => KeyCode::KeyY,
            'Z' => KeyCode::KeyZ,
            _ => return false,
        },
        GearKey::Digit(digit) => match digit {
            0 => KeyCode::Digit0,
            1 => KeyCode::Digit1,
            2 => KeyCode::Digit2,
            3 => KeyCode::Digit3,
            4 => KeyCode::Digit4,
            5 => KeyCode::Digit5,
            6 => KeyCode::Digit6,
            7 => KeyCode::Digit7,
            8 => KeyCode::Digit8,
            9 => KeyCode::Digit9,
            _ => return false,
        },
        GearKey::Space => KeyCode::Space,
        GearKey::ArrowUp => KeyCode::ArrowUp,
        GearKey::ArrowDown => KeyCode::ArrowDown,
        GearKey::ArrowLeft => KeyCode::ArrowLeft,
        GearKey::ArrowRight => KeyCode::ArrowRight,
        GearKey::PageUp => KeyCode::PageUp,
        GearKey::PageDown => KeyCode::PageDown,
    };
    let shift = keyboard.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    let control = keyboard.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]);
    let alt = keyboard.any_pressed([KeyCode::AltLeft, KeyCode::AltRight]);
    let super_key = keyboard.any_pressed([KeyCode::SuperLeft, KeyCode::SuperRight]);
    keyboard.just_pressed(key)
        && shift == chord.shift
        && control == chord.control
        && alt == chord.alt
        && super_key == chord.super_key
}

impl DriveSequencer {
    /// Whether rows have been built for a running simulation.
    pub(crate) const fn is_started(&self) -> bool {
        self.started
    }

    /// Live rows, in coordinate order.
    pub(crate) fn rows(&self) -> &[SequencerRow] {
        &self.rows
    }

    /// State one wire is currently in, when the sequencer is running it.
    pub(crate) fn active_state(&self, link: DriveLinkId) -> Option<u8> {
        self.rows
            .iter()
            .find(|row| row.link == link)
            .map(|row| row.cursor.active)
    }

    /// Builds one row per driven bearing that has a coordinate to move.
    ///
    /// A bearing that lost the physical-duplicate collapse still resolves,
    /// because compilation records every graph bearing's coordinate.
    pub(crate) fn start(&mut self, creation: &CompiledCreation, graph: &ConstructionGraph) {
        let coordinates = &creation.loop_topology.bearing_coordinates;
        let mut by_coordinate = BTreeMap::new();
        for (link, spec) in graph.drive_links() {
            let Some(coordinate) = coordinates.get(&spec.bearing).copied() else {
                continue;
            };
            by_coordinate.entry(coordinate).or_insert(SequencerRow {
                link,
                coordinate,
                cursor: RowCursor::default(),
            });
        }
        self.rows = by_coordinate.into_values().collect();
        self.started = true;
    }

    /// Clears every row when the simulation ends.
    pub(crate) fn stop(&mut self) {
        self.rows.clear();
        self.started = false;
    }

    /// Advances every row, reporting whether any of them changed state.
    pub(crate) fn step(
        &mut self,
        graph: &ConstructionGraph,
        keys: &DriveKeyState,
        keyboard_controller: Option<mechanic_core::PartId>,
        tick: u64,
    ) -> bool {
        let mut changed = false;
        let no_keys = DriveKeyState::default();
        for row in &mut self.rows {
            let Some(spec) = graph.drive_link(row.link) else {
                continue;
            };
            let routed_keys = if keyboard_controller == Some(spec.controller) {
                keys
            } else {
                &no_keys
            };
            let stepped = stepped_cursor(row.cursor, &spec.program, routed_keys, tick);
            if stepped != row.cursor {
                row.cursor = stepped;
                changed = true;
            }
        }
        changed
    }
}

/// Dwell length in dispatched physics ticks.
///
/// Always at least one tick, so a very short dwell still holds for a frame
/// rather than collapsing into an instant chain of handoffs.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
// A dwell is validated positive and at most MAX_DRIVE_DWELL_SECONDS, so the
// tick count is a small positive integer well inside u64.
fn dwell_ticks(seconds: f32) -> u64 {
    (seconds / FIXED_DT_SECONDS).round().max(1.0) as u64
}

/// Advances one bearing's cursor by one frame.
///
/// A key press wins over everything: it is the player acting now. Otherwise a
/// held state that has been let go returns to its named state, and only then
/// does an elapsed dwell hand off. A state with neither rule holds forever,
/// which is what makes a latched pose stay put.
fn stepped_cursor(
    cursor: RowCursor,
    program: &DriveProgram,
    keys: &DriveKeyState,
    tick: u64,
) -> RowCursor {
    let entered = |state: u8| RowCursor {
        active: state,
        entered_tick: tick,
    };

    for (index, state) in program.states().iter().enumerate() {
        let Some(trigger) = state.trigger() else {
            continue;
        };
        if keys.is_pressed(trigger.key()) {
            let target = u8::try_from(index).unwrap_or(cursor.active);
            return if target == cursor.active {
                cursor
            } else {
                entered(target)
            };
        }
    }

    let active = program.state(cursor.active).unwrap_or_default();
    if let Some(trigger) = active.trigger()
        && let DriveRelease::RevertTo(target) = trigger.release()
        && !keys.is_held(trigger.key())
        && target != cursor.active
    {
        return entered(target);
    }

    if let Some(dwell) = active.dwell()
        && tick.saturating_sub(cursor.entered_tick) >= dwell_ticks(dwell.seconds())
        && let Some(next) = program.advanced_state(cursor.active)
        && next != cursor.active
    {
        return entered(next);
    }

    cursor
}

/// Builds the GPU drive rows for the sequencer's current states.
///
/// Starts from the compiled state-zero rows so undriven coordinates stay
/// passive, then overwrites every coordinate the sequencer owns.
#[cfg(test)]
pub(crate) fn gpu_drive_rows(
    creation: &CompiledCreation,
    graph: &ConstructionGraph,
    sequencer: &DriveSequencer,
) -> Vec<GpuMechanismDrive> {
    let mut rows = creation
        .resolve_coordinate_drives(graph)
        .into_iter()
        .map(GpuMechanismDrive::from)
        .collect::<Vec<_>>();
    for row in sequencer.rows() {
        let Some(spec) = graph.drive_link(row.link) else {
            continue;
        };
        let Some(target) = spec.resolved_target(row.cursor.active) else {
            continue;
        };
        let Some(slot) = rows.get_mut(row.coordinate as usize) else {
            continue;
        };
        match target {
            DriveTarget::Speed(speed) => {
                slot.mode = DRIVE_MODE_SPEED;
                slot.target_speed = speed;
                slot.target_angle = 0.0;
            }
            DriveTarget::Angle(angle) => {
                slot.mode = DRIVE_MODE_ANGLE;
                slot.target_speed = 0.0;
                slot.target_angle = angle;
            }
        }
    }
    rows
}

/// Builds GPU rows with independently geared gas and electric contributions.
pub(crate) fn geared_gpu_drive_rows(
    creation: &CompiledCreation,
    graph: &ConstructionGraph,
    sequencer: &DriveSequencer,
    gearboxes: &GearboxRuntime,
) -> Vec<GpuMechanismDrive> {
    let selections = gearboxes.selections(graph);
    let mut rows = creation
        .resolve_coordinate_drives_with_gears(graph, &selections)
        .into_iter()
        .map(GpuMechanismDrive::from)
        .collect::<Vec<_>>();
    apply_live_targets(&mut rows, graph, sequencer);
    for row in sequencer.rows() {
        let Some(spec) = graph.drive_link(row.link) else {
            continue;
        };
        if !spec.actuator.uses_gas() {
            continue;
        }
        let requested_sign = spec
            .resolved_target(row.cursor.active)
            .and_then(DriveTarget::speed)
            .map_or(0, |speed| {
                if speed > 0.0 {
                    1
                } else if speed < 0.0 {
                    -1
                } else {
                    0
                }
            });
        if gearboxes.gas_direction(graph, spec.controller) != Some(requested_sign)
            && let Some(slot) = rows.get_mut(row.coordinate as usize)
        {
            slot.source_b_max_acceleration = 0.0;
            slot.source_b_no_load_speed = 0.0;
            slot.max_acceleration = slot.source_a_max_acceleration;
            slot.max_speed = slot.source_a_no_load_speed;
        }
    }
    rows
}

fn apply_live_targets(
    rows: &mut [GpuMechanismDrive],
    graph: &ConstructionGraph,
    sequencer: &DriveSequencer,
) {
    for row in sequencer.rows() {
        let Some(spec) = graph.drive_link(row.link) else {
            continue;
        };
        let Some(target) = spec.resolved_target(row.cursor.active) else {
            continue;
        };
        let Some(slot) = rows.get_mut(row.coordinate as usize) else {
            continue;
        };
        match target {
            DriveTarget::Speed(speed) => {
                slot.mode = DRIVE_MODE_SPEED;
                slot.target_speed = speed.clamp(-slot.max_speed, slot.max_speed);
                slot.target_angle = 0.0;
            }
            DriveTarget::Angle(angle) => {
                slot.mode = DRIVE_MODE_ANGLE;
                slot.target_speed = 0.0;
                slot.target_angle = angle;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DriveKeyState, RowCursor, automatic_shift_destination, chord_just_pressed, initial_gear,
        reversal_is_safe, stepped_cursor,
    };
    use mechanic_core::{
        DriveDwell, DriveKey, DriveProgram, DriveRelease, DriveState, DriveTarget, DriveTrigger,
        EngineKind, GearKey, GearKeyChord, GearboxConfig,
    };

    fn key(symbol: char) -> DriveKey {
        DriveKey::new(symbol).expect("test keys are letters")
    }

    fn keys(held: &[char], pressed: &[char]) -> DriveKeyState {
        DriveKeyState {
            held: held.iter().copied().map(key).collect(),
            pressed: pressed.iter().copied().map(key).collect(),
        }
    }

    fn angle_state(degrees: f32) -> DriveState {
        DriveState::new(DriveTarget::Angle(degrees.to_radians())).expect("test angle is in range")
    }

    fn row() -> RowCursor {
        RowCursor::default()
    }

    /// `S1 0°` · `S2 30° hold A` · `S3 -30° hold D`, both reverting to S1.
    fn steering() -> DriveProgram {
        let held = |degrees: f32, symbol: char| {
            angle_state(degrees).with_trigger(Some(DriveTrigger::new(
                key(symbol),
                DriveRelease::RevertTo(0),
            )))
        };
        DriveProgram::new(
            &[angle_state(0.0), held(30.0, 'A'), held(-30.0, 'D')],
            false,
        )
        .expect("steering program is valid")
    }

    #[test]
    fn held_key_reverts_to_its_default_state_on_release() {
        let program = steering();
        let start = row();

        let pressed = stepped_cursor(start, &program, &keys(&['A'], &['A']), 10);
        assert_eq!(pressed.active, 1);
        assert_eq!(pressed.entered_tick, 10);

        // Still held: the row stays put rather than re-entering and resetting
        // its dwell clock.
        let holding = stepped_cursor(pressed, &program, &keys(&['A'], &[]), 30);
        assert_eq!(holding, pressed);

        let released = stepped_cursor(holding, &program, &keys(&[], &[]), 45);
        assert_eq!(
            released.active, 0,
            "letting go returns to the default state"
        );
        assert_eq!(released.entered_tick, 45);
    }

    #[test]
    fn steering_the_other_way_switches_states_without_passing_through_neutral() {
        let program = steering();
        let left = stepped_cursor(row(), &program, &keys(&['A'], &['A']), 5);
        assert_eq!(left.active, 1);

        let right = stepped_cursor(left, &program, &keys(&['D'], &['D']), 6);
        assert_eq!(right.active, 2, "a fresh press wins over the held state");
    }

    #[test]
    fn latched_pose_keys_stay_until_another_key() {
        let latched = |degrees: f32, symbol: char| {
            angle_state(degrees)
                .with_trigger(Some(DriveTrigger::new(key(symbol), DriveRelease::Latch)))
        };
        let program = DriveProgram::new(
            &[latched(30.0, 'Q'), latched(40.0, 'W'), latched(80.0, 'E')],
            false,
        )
        .expect("arm program is valid");

        let pressed = stepped_cursor(row(), &program, &keys(&['W'], &['W']), 4);
        assert_eq!(pressed.active, 1);

        // Released, and many ticks later: a latched state has no dwell and no
        // revert, so it holds.
        let later = stepped_cursor(pressed, &program, &keys(&[], &[]), 4_000);
        assert_eq!(later, pressed);

        let moved = stepped_cursor(later, &program, &keys(&['E'], &['E']), 4_100);
        assert_eq!(moved.active, 2);
    }

    #[test]
    fn dwell_advances_to_its_named_state_and_cycles() {
        // S1 0° key R · S2 90° key S, 2 s -> S3 · S3 -90°, 4 s -> S2.
        let reset =
            angle_state(0.0).with_trigger(Some(DriveTrigger::new(key('R'), DriveRelease::Latch)));
        let forward = angle_state(90.0)
            .with_trigger(Some(DriveTrigger::new(key('S'), DriveRelease::Latch)))
            .with_dwell(Some(
                DriveDwell::new(2.0, Some(2)).expect("dwell is in range"),
            ));
        let back = angle_state(-90.0).with_dwell(Some(
            DriveDwell::new(4.0, Some(1)).expect("dwell is in range"),
        ));
        let program =
            DriveProgram::new(&[reset, forward, back], false).expect("procedure is valid");

        let started = stepped_cursor(row(), &program, &keys(&['S'], &['S']), 100);
        assert_eq!(started.active, 1);

        // 2 s is 120 ticks: one tick early it holds, on the boundary it hands off.
        let waiting = stepped_cursor(started, &program, &keys(&[], &[]), 219);
        assert_eq!(waiting.active, 1);
        let advanced = stepped_cursor(waiting, &program, &keys(&[], &[]), 220);
        assert_eq!(advanced.active, 2);
        assert_eq!(advanced.entered_tick, 220);

        // 4 s is 240 ticks, and S3 names S2, so the pair cycles forever.
        let cycled = stepped_cursor(advanced, &program, &keys(&[], &[]), 460);
        assert_eq!(cycled.active, 1);
    }

    #[test]
    fn reset_key_interrupts_a_running_procedure() {
        let reset =
            angle_state(0.0).with_trigger(Some(DriveTrigger::new(key('R'), DriveRelease::Latch)));
        let forward = angle_state(90.0)
            .with_trigger(Some(DriveTrigger::new(key('S'), DriveRelease::Latch)))
            .with_dwell(Some(
                DriveDwell::new(2.0, Some(2)).expect("dwell is in range"),
            ));
        let back = angle_state(-90.0).with_dwell(Some(
            DriveDwell::new(4.0, Some(1)).expect("dwell is in range"),
        ));
        let program =
            DriveProgram::new(&[reset, forward, back], false).expect("procedure is valid");

        let running = RowCursor {
            active: 2,
            entered_tick: 500,
        };
        let stopped = stepped_cursor(running, &program, &keys(&['R'], &['R']), 520);
        assert_eq!(stopped.active, 0);

        // State 0 has no dwell, so the procedure stays stopped.
        let still = stepped_cursor(stopped, &program, &keys(&[], &[]), 5_000);
        assert_eq!(still, stopped);
    }

    #[test]
    fn a_dwell_without_a_target_walks_the_program_and_wraps_only_when_it_loops() {
        let timed = |degrees: f32| {
            angle_state(degrees)
                .with_dwell(Some(DriveDwell::new(1.0, None).expect("dwell is in range")))
        };
        let states = [timed(0.0), timed(45.0)];

        let once = DriveProgram::new(&states, false).expect("program is valid");
        let advanced = stepped_cursor(row(), &once, &keys(&[], &[]), 60);
        assert_eq!(advanced.active, 1);
        let held = stepped_cursor(advanced, &once, &keys(&[], &[]), 600);
        assert_eq!(held, advanced, "the last state of a one-shot program holds");

        let looping = DriveProgram::new(&states, true).expect("program is valid");
        let wrapped = stepped_cursor(advanced, &looping, &keys(&[], &[]), 600);
        assert_eq!(wrapped.active, 0);
    }

    /// Grounded base, one hinged arm, and a control block wired to it.
    fn driven_arm(
        program: DriveProgram,
    ) -> (
        mechanic_core::ConstructionGraph,
        mechanic_core::CompiledCreation,
        mechanic_core::PartId,
    ) {
        use bevy::prelude::{IVec3, Vec3};
        use mechanic_core::{
            BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, ControllerSpec,
            CuboidSpec, DriveLinkSpec, FaceKind, FaceRef, GridRotation, WeldSpec,
        };

        let mut graph = ConstructionGraph::new();
        let mut spawn = |units| {
            let spec =
                CuboidSpec::new([4, 4, 4], BuildPose::new(units, GridRotation::default())).unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        };
        let base = spawn(IVec3::new(0, 2, 0));
        let arm = spawn(IVec3::new(4, 2, 0));
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(base, FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();
        let BuildOutcome::BearingAdded(bearing) = graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(base, FaceKind::PositiveX),
                FaceRef::part(arm, FaceKind::NegativeX),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::X,
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(IVec3::new(0, 40, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let mut link = DriveLinkSpec::new(controller, bearing);
        link.program = program;
        graph.apply(BuildCommand::AddDriveLink(link)).unwrap();
        let creation = graph.compile().unwrap();
        (graph, creation, controller)
    }

    #[test]
    fn a_key_press_reaches_the_gpu_row_for_that_bearings_coordinate() {
        use mechanic_core::{DriveTarget, DriveTrigger};

        let latched = |speed: f32, symbol: char| {
            DriveState::new(DriveTarget::Speed(speed))
                .expect("speed is in range")
                .with_trigger(Some(DriveTrigger::new(key(symbol), DriveRelease::Latch)))
        };
        let program = DriveProgram::new(&[latched(0.0, 'N'), latched(2.0, 'W')], false)
            .expect("driving program is valid");
        let (graph, creation, controller) = driven_arm(program);

        let mut sequencer = super::DriveSequencer::default();
        sequencer.start(&creation, &graph);
        assert_eq!(sequencer.rows().len(), 1, "one driven bearing, one row");

        // State 0 holds the arm still.
        let idle = super::gpu_drive_rows(&creation, &graph, &sequencer);
        assert_eq!(idle.len(), creation.loop_topology.tree_bearings.len());
        assert!(idle[0].target_speed.abs() < f32::EPSILON);

        // A key is ignored until an occupied Seat's Input route selects this
        // controller.
        assert!(!sequencer.step(&graph, &keys(&['W'], &['W']), None, 29));

        // Pressing W enters state 1, and that target reaches the same row.
        assert!(sequencer.step(&graph, &keys(&['W'], &['W']), Some(controller), 30));
        let driving = super::gpu_drive_rows(&creation, &graph, &sequencer);
        assert!((driving[0].target_speed - 2.0).abs() < 1.0e-5);
        assert_eq!(driving[0].mode, mechanic_gpu::DRIVE_MODE_SPEED);

        // Holding the same key changes nothing, so no needless GPU write.
        assert!(!sequencer.step(&graph, &keys(&['W'], &[]), Some(controller), 31));
    }

    #[test]
    fn a_reversed_wire_flips_the_row_the_sequencer_uploads() {
        use mechanic_core::{BuildCommand, DriveTarget};

        let program =
            DriveProgram::new(&[DriveState::new(DriveTarget::Speed(2.0)).unwrap()], false).unwrap();
        let (mut graph, creation, _) = driven_arm(program);
        let (link, spec) = graph
            .drive_links()
            .map(|(id, spec)| (id, *spec))
            .next()
            .unwrap();

        let mut sequencer = super::DriveSequencer::default();
        sequencer.start(&creation, &graph);
        let forward = super::gpu_drive_rows(&creation, &graph, &sequencer);
        assert!(forward[0].target_speed > 0.0);

        graph.apply(BuildCommand::RemoveDriveLink(link)).unwrap();
        let mut reversed = spec;
        reversed.reversed = true;
        graph.apply(BuildCommand::AddDriveLink(reversed)).unwrap();
        let mut sequencer = super::DriveSequencer::default();
        sequencer.start(&creation, &graph);
        let backward = super::gpu_drive_rows(&creation, &graph, &sequencer);
        assert!((backward[0].target_speed + forward[0].target_speed).abs() < 1.0e-5);
    }

    #[test]
    fn the_panel_key_is_not_bindable() {
        // E opens the control panel, so binding it to a state would make one
        // press both move the machine and open the window used to edit it.
        assert_eq!(super::drive_key(bevy::prelude::KeyCode::KeyE), None);
        assert!(super::drive_key(bevy::prelude::KeyCode::KeyD).is_some());
        assert!(super::drive_key(bevy::prelude::KeyCode::KeyF).is_some());
        assert_eq!(
            super::gear_key(bevy::prelude::KeyCode::KeyE),
            Some(GearKey::Letter('E'))
        );
        assert_eq!(
            super::gear_key(bevy::prelude::KeyCode::Space),
            Some(GearKey::Space)
        );
        assert_eq!(
            super::gear_key(bevy::prelude::KeyCode::PageDown),
            Some(GearKey::PageDown)
        );
    }

    #[test]
    fn a_blocked_keyboard_drives_nothing() {
        let program = steering();
        let mut keyboard = bevy::input::ButtonInput::<bevy::prelude::KeyCode>::default();
        keyboard.press(bevy::prelude::KeyCode::KeyA);

        let typing = DriveKeyState::from_keyboard(&keyboard, true);
        assert_eq!(stepped_cursor(row(), &program, &typing, 10).active, 0);

        let playing = DriveKeyState::from_keyboard(&keyboard, false);
        assert_eq!(stepped_cursor(row(), &program, &playing, 10).active, 1);
    }

    #[test]
    fn automatic_shift_thresholds_have_hysteresis() {
        let config = GearboxConfig::for_depth(3, false);
        let output_speed =
            |rpm: f32, gear: usize| rpm * core::f32::consts::TAU / 60.0 / config.ratios()[gear];
        assert_eq!(
            automatic_shift_destination(
                EngineKind::Electric,
                &config,
                0,
                output_speed(EngineKind::Electric.no_load_rpm() * 0.85, 0),
            ),
            1,
        );
        assert_eq!(
            automatic_shift_destination(
                EngineKind::Electric,
                &config,
                1,
                output_speed(EngineKind::Electric.no_load_rpm() * 0.40, 1),
            ),
            0,
        );
        assert_eq!(
            automatic_shift_destination(
                EngineKind::Electric,
                &config,
                1,
                output_speed(EngineKind::Electric.no_load_rpm() * 0.60, 1),
            ),
            1,
        );
    }

    #[test]
    fn gas_direction_banks_and_reversal_gate_cover_missing_and_safe_destinations() {
        let mut config = GearboxConfig::for_depth(1, true);
        assert_eq!(initial_gear(&config, EngineKind::Gas, -1.0), Some(0));
        assert_eq!(initial_gear(&config, EngineKind::Gas, 1.0), Some(1));
        assert!(!reversal_is_safe(EngineKind::Gas, &config, 1, 30.0));
        assert!(reversal_is_safe(EngineKind::Gas, &config, 1, 0.01));
        config = mechanic_core::GearboxConfig::new(
            mechanic_core::ShiftMode::Auto,
            config.ratios().to_vec(),
            0,
            config.gear_up(),
            config.gear_down(),
        )
        .unwrap();
        assert_eq!(initial_gear(&config, EngineKind::Gas, -1.0), None);
    }

    #[test]
    fn shift_chords_accept_space_page_keys_and_modifiers() {
        let mut keyboard = bevy::input::ButtonInput::<bevy::prelude::KeyCode>::default();
        keyboard.press(bevy::prelude::KeyCode::ShiftLeft);
        keyboard.press(bevy::prelude::KeyCode::Space);
        let chord = GearKeyChord {
            shift: true,
            ..GearKeyChord::new(GearKey::Space)
        };
        assert!(chord_just_pressed(&keyboard, chord));
        assert!(!chord_just_pressed(
            &keyboard,
            GearKeyChord::new(GearKey::Space)
        ));
    }
}
