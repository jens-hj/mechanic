//! Runs each driven bearing's state program while the simulation ticks.
//!
//! A row advances on two things only: a key the player pressed or let go, and
//! elapsed simulated time. Time is counted in dispatched physics ticks rather
//! than frames, so a paused simulation freezes every dwell and a slow frame
//! never skips one.

use crate::camera::PlayerState;
use crate::editor::state::EditorState;
use crate::simulation::state::AppSimulation;
use crate::{automation, freeze, ui};
use mechanic_core::TICK_SECONDS_F32;
use mechanic_gpu::GpuTransform;
use std::collections::{BTreeMap, BTreeSet};

use bevy::prelude::*;
use mechanic_core::{
    BearingId, CompiledCreation, ConstructionGraph, DriveKey, DriveLinkId, DriveProgram,
    DriveRelease, DriveTarget, EngineKind, GearKey, GearKeyChord, GearSelection, PartId, PartSpec,
    ShiftMode,
};
use mechanic_gpu::{DRIVE_MODE_ANGLE, DRIVE_MODE_SPEED, GpuMechanismDrive};

/// Which of a program's keys are down, and which went down this frame.
#[derive(Clone, Debug, Default)]
pub(crate) struct DriveKeyState {
    held: Vec<DriveKey>,
    pressed: Vec<DriveKey>,
    routed: Option<mechanic_core::ControllerKeys>,
}

impl DriveKeyState {
    /// Constructs application-level scripted input without synthesizing OS events.
    pub(crate) fn scripted(held: &[char], previous: &[char]) -> Self {
        Self {
            routed: None,
            held: held
                .iter()
                .filter_map(|symbol| DriveKey::new(*symbol))
                .collect(),
            pressed: held
                .iter()
                .filter(|symbol| !previous.contains(symbol))
                .filter_map(|symbol| DriveKey::new(*symbol))
                .collect(),
        }
    }

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
        KeyCode::KeyE => 'E',
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
    last_step_tick: u64,
    publication: Option<(u64, u64)>,
    programs: BTreeMap<DriveLinkId, mechanic_core::DriveLinkSpec>,
    routed_keys: Option<mechanic_core::ControllerKeys>,
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
    last_step_tick: u64,
    configs: Vec<(
        mechanic_core::PartId,
        EngineKind,
        mechanic_core::GearboxConfig,
    )>,
}

impl GearboxRuntime {
    pub(crate) fn start(&mut self, graph: &ConstructionGraph, sequencer: &DriveSequencer) {
        self.rows.clear();
        self.configs.clear();
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
                self.configs.push((controller, kind, config));
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
        self.last_step_tick = 0;
    }

    /// Rebinds unchanged controller lanes after a live publication.
    pub(crate) fn sync_publication(
        &mut self,
        graph: &ConstructionGraph,
        sequencer: &DriveSequencer,
    ) {
        let last_step_tick = self.last_step_tick;
        let previous_rows = std::mem::take(&mut self.rows);
        let previous_configs = std::mem::take(&mut self.configs);
        self.start(graph, sequencer);
        self.last_step_tick = last_step_tick;
        for row in &mut self.rows {
            let unchanged = previous_configs.iter().any(|previous| {
                previous.0 == row.controller
                    && previous.1 == row.kind
                    && self.configs.iter().any(|current| current == previous)
            });
            if unchanged
                && let Some(previous) = previous_rows.iter().find(|previous| {
                    previous.controller == row.controller && previous.kind == row.kind
                })
            {
                *row = *previous;
            }
        }
    }

    pub(crate) fn stop(&mut self) {
        self.rows.clear();
        self.configs.clear();
        self.started = false;
        self.last_step_tick = 0;
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
    #[expect(
        clippy::too_many_arguments,
        reason = "runtime inputs stay explicit at the simulation boundary"
    )]
    #[cfg_attr(not(test), allow(dead_code))]
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
        self.step_with_suspension(
            graph,
            sequencer,
            keyboard,
            keyboard_controller,
            tick,
            measured_speeds,
            paused,
            &BTreeSet::new(),
        )
    }

    /// Pauses selected controller lanes, including pending shifts and cooldowns.
    #[expect(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(crate) fn step_with_suspension(
        &mut self,
        graph: &ConstructionGraph,
        sequencer: &DriveSequencer,
        keyboard: &ButtonInput<KeyCode>,
        keyboard_controller: Option<mechanic_core::PartId>,
        tick: u64,
        measured_speeds: &[(mechanic_core::PartId, EngineKind, f32)],
        paused: bool,
        suspended_controllers: &BTreeSet<PartId>,
    ) -> bool {
        let elapsed = tick.saturating_sub(self.last_step_tick);
        self.last_step_tick = tick;
        let mut changed = false;
        for row in &mut self.rows {
            if suspended_controllers.contains(&row.controller) {
                row.last_shift_tick = row
                    .last_shift_tick
                    .saturating_add(elapsed.min(tick.saturating_sub(row.last_shift_tick)));
                continue;
            }
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
            let pressed = |chord: GearKeyChord| {
                if !chord.shift
                    && !chord.control
                    && !chord.alt
                    && !chord.super_key
                    && let Some(keys) = &sequencer.routed_keys
                    && let Some(key) = match chord.key {
                        GearKey::Letter(letter) => DriveKey::new(letter),
                        GearKey::Digit(digit) => {
                            char::from_digit(u32::from(digit), 10).and_then(DriveKey::new)
                        }
                        _ => None,
                    }
                {
                    return keys.pressed(row.controller, key)
                        && (keys.button_held(row.controller, key)
                            || (keyboard_controller == Some(row.controller)
                                && chord_just_pressed(keyboard, chord)));
                }
                keyboard_controller == Some(row.controller) && chord_just_pressed(keyboard, chord)
            };
            let delta = i8::from(pressed(config.gear_up())) - i8::from(pressed(config.gear_down()));
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
    let engine_rpm = mechanic_core::rad_s_to_rpm(measured_speed.abs() * ratio);
    let (first, last) = gear_bank_range(kind, config, current);
    if engine_rpm >= 0.75 * kind.no_load_rpm() && current < last {
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
        mechanic_core::rpm_to_rad_s(kind.no_load_rpm()) / config.ratios()[destination];
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
            target_speed(spec.program.state(row.cursor.active)?.target())
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

    /// Whether the rows belong to the currently published physics scene.
    pub(crate) fn is_started_for(&self, publication: Option<(u64, u64)>) -> bool {
        self.started && self.publication == publication
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
    pub(crate) fn start(
        &mut self,
        creation: &CompiledCreation,
        graph: &ConstructionGraph,
        publication: Option<(u64, u64)>,
    ) {
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
        self.programs = graph.drive_links().map(|(id, spec)| (id, *spec)).collect();
        self.started = true;
        self.last_step_tick = 0;
        self.publication = publication;
    }

    /// Preserves unchanged wire cursors while remapping compiled coordinates.
    /// The caller must retain the simulation tick counter across publication.
    pub(crate) fn sync_publication(
        &mut self,
        creation: &CompiledCreation,
        graph: &ConstructionGraph,
        publication: Option<(u64, u64)>,
        tick: u64,
    ) {
        let last_step_tick = self.last_step_tick;
        let previous_rows = std::mem::take(&mut self.rows);
        let previous_programs = std::mem::take(&mut self.programs);
        self.start(creation, graph, publication);
        self.last_step_tick = last_step_tick;
        for row in &mut self.rows {
            row.cursor.entered_tick = tick;
            let unchanged_program = previous_programs
                .get(&row.link)
                .zip(self.programs.get(&row.link))
                .is_some_and(|(previous, current)| {
                    previous.program == current.program
                        && previous.controller == current.controller
                        && previous.bearing == current.bearing
                });
            if unchanged_program
                && let Some(previous) = previous_rows.iter().find(|old| old.link == row.link)
            {
                row.cursor = previous.cursor;
            }
        }
    }

    /// Clears every row when the simulation ends.
    pub(crate) fn stop(&mut self) {
        self.rows.clear();
        self.started = false;
        self.last_step_tick = 0;
        self.publication = None;
        self.programs.clear();
    }

    /// Advances every row, reporting whether any of them changed state.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn step(
        &mut self,
        graph: &ConstructionGraph,
        keys: &DriveKeyState,
        keyboard_controller: Option<mechanic_core::PartId>,
        tick: u64,
    ) -> bool {
        self.step_with_suspension(graph, keys, keyboard_controller, tick, &BTreeSet::new())
    }

    /// Holds selected controller programs without consuming keys or dwell time.
    pub(crate) fn step_with_suspension(
        &mut self,
        graph: &ConstructionGraph,
        keys: &DriveKeyState,
        keyboard_controller: Option<mechanic_core::PartId>,
        tick: u64,
        suspended_controllers: &BTreeSet<PartId>,
    ) -> bool {
        self.step_with_held_bearings(
            graph,
            keys,
            keyboard_controller,
            tick,
            suspended_controllers,
            &BTreeSet::new(),
        )
    }

    /// Suspends rows whose controller or driven bearing belongs to a held creation.
    pub(crate) fn step_with_held_bearings(
        &mut self,
        graph: &ConstructionGraph,
        keys: &DriveKeyState,
        keyboard_controller: Option<PartId>,
        tick: u64,
        suspended_controllers: &BTreeSet<PartId>,
        held_bearings: &BTreeSet<BearingId>,
    ) -> bool {
        let elapsed = tick.saturating_sub(self.last_step_tick);
        self.last_step_tick = tick;
        let mut changed = false;
        self.routed_keys.clone_from(&keys.routed);
        let no_keys = DriveKeyState::default();
        for row in &mut self.rows {
            let Some(spec) = graph.drive_link(row.link) else {
                continue;
            };
            if suspended_controllers.contains(&spec.controller)
                || held_bearings.contains(&spec.bearing)
            {
                row.cursor.entered_tick = row
                    .cursor
                    .entered_tick
                    .saturating_add(elapsed.min(tick.saturating_sub(row.cursor.entered_tick)));
                continue;
            }
            let routed_keys = if keyboard_controller == Some(spec.controller) {
                keys
            } else {
                &no_keys
            };
            let combined = keys.routed.as_ref().map(|combined| {
                let held: Vec<_> = combined
                    .held_keys()
                    .filter(|(controller, _)| *controller == spec.controller)
                    .map(|(_, key)| key)
                    .collect();
                let pressed = held
                    .iter()
                    .copied()
                    .filter(|key| combined.pressed(spec.controller, *key))
                    .collect();
                DriveKeyState {
                    held,
                    pressed,
                    routed: None,
                }
            });
            let stepped = stepped_cursor(
                row.cursor,
                &spec.program,
                combined.as_ref().unwrap_or(routed_keys),
                tick,
            );
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
#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
// A dwell is validated positive and at most MAX_DRIVE_DWELL_SECONDS, so the
// tick count is a small positive integer well inside u64.
fn dwell_ticks(seconds: f32) -> u64 {
    (seconds / mechanic_core::TICK_SECONDS_F32).round().max(1.0) as u64
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
            DriveTarget::Speed(speed) | DriveTarget::LinearSpeed(speed) => {
                slot.mode = DRIVE_MODE_SPEED;
                slot.target_speed = speed;
                slot.target_angle = 0.0;
            }
            DriveTarget::Angle(angle) | DriveTarget::LinearPosition(angle) => {
                slot.mode = DRIVE_MODE_ANGLE;
                slot.target_speed = 0.0;
                slot.target_angle = if target.is_linear() {
                    angle.clamp(slot.min_angle, slot.max_angle)
                } else {
                    angle
                };
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
            .program
            .state(row.cursor.active)
            .map(mechanic_core::DriveState::target)
            .and_then(target_speed)
            .map_or(0, |speed| {
                if speed > 0.0 {
                    1
                } else if speed < 0.0 {
                    -1
                } else {
                    0
                }
            });
        let gas_engaged = requested_sign != 0
            && gearboxes.gas_direction(graph, spec.controller) == Some(requested_sign);
        if !gas_engaged && let Some(slot) = rows.get_mut(row.coordinate as usize) {
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
            DriveTarget::Speed(speed) | DriveTarget::LinearSpeed(speed) => {
                slot.mode = DRIVE_MODE_SPEED;
                slot.target_speed = speed.clamp(-slot.max_speed, slot.max_speed);
                slot.target_angle = 0.0;
            }
            DriveTarget::Angle(angle) | DriveTarget::LinearPosition(angle) => {
                slot.mode = DRIVE_MODE_ANGLE;
                slot.target_speed = 0.0;
                slot.target_angle = if target.is_linear() {
                    angle.clamp(slot.min_angle, slot.max_angle)
                } else {
                    angle
                };
            }
        }
    }
}

fn target_speed(target: DriveTarget) -> Option<f32> {
    match target {
        DriveTarget::Speed(speed) | DriveTarget::LinearSpeed(speed) => Some(speed),
        DriveTarget::Angle(_) | DriveTarget::LinearPosition(_) => None,
    }
}

/// Advances every driven bearing's program and pushes changed rows to the GPU.
///
/// Runs immediately before the tick is dispatched, so a state entered this
/// frame takes effect in the same tick rather than the next one.
#[expect(
    clippy::too_many_arguments,
    reason = "bevy systems receive each independent resource explicitly"
)]
pub(crate) fn run_drive_sequencer(
    keyboard: Res<ButtonInput<KeyCode>>,
    controls: Res<crate::physical_controls::PhysicalControls>,
    pause: Res<crate::pause_menu::PauseMenuState>,
    overlay: Res<ui::UiInput>,
    simulation: Res<AppSimulation>,
    frozen: Res<freeze::DimensionFreeze>,
    mut sequencer: ResMut<DriveSequencer>,
    mut gearboxes: ResMut<GearboxRuntime>,
    mut state: ResMut<EditorState>,
    mut player: ResMut<PlayerState>,
) {
    if !simulation.is_running() {
        if sequencer.is_started() {
            sequencer.stop();
            gearboxes.stop();
        }
        return;
    }
    if !sequencer.is_started_for(simulation.world_revision) {
        let Some(creation) = simulation.creation.as_ref() else {
            return;
        };
        if sequencer.is_started() && simulation.world_revision.is_some() {
            sequencer.sync_publication(
                creation,
                simulation.effective_graph(),
                simulation.world_revision,
                simulation.next_tick,
            );
            gearboxes.sync_publication(simulation.effective_graph(), &sequencer);
        } else {
            sequencer.start(
                creation,
                simulation.effective_graph(),
                simulation.world_revision,
            );
            gearboxes.start(simulation.effective_graph(), &sequencer);
        }
        state.drive_rows_dirty = true;
    }
    if automation::driving_enabled() {
        player.seat = automation::driving_seat(&simulation);
        return; // Scripted programs advance at each dispatched tick below.
    }
    if pause.blocks_world_input() {
        return;
    }
    let mut keys = DriveKeyState::from_keyboard(&keyboard, overlay.blocks_keyboard());
    keys.routed = Some(controls.keys.clone());
    let keyboard_controller = player
        .seat
        .filter(|seat| simulation.published_graph.seat_input(*seat).is_some())
        .and_then(|seat| simulation.published_graph.seat_controller(seat));
    step_drive_programs(
        &simulation,
        &frozen,
        &mut sequencer,
        &mut gearboxes,
        &mut state,
        &keyboard,
        &keys,
        keyboard_controller,
        (!overlay.blocks_keyboard())
            .then_some(keyboard_controller)
            .flatten(),
        simulation.next_tick,
    );
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn step_drive_programs(
    simulation: &AppSimulation,
    frozen: &freeze::DimensionFreeze,
    sequencer: &mut DriveSequencer,
    gearboxes: &mut GearboxRuntime,
    state: &mut EditorState,
    keyboard: &ButtonInput<KeyCode>,
    keys: &DriveKeyState,
    keyboard_controller: Option<PartId>,
    gearbox_keyboard_controller: Option<PartId>,
    tick: u64,
) {
    let suspended = frozen.suspended_controllers(simulation);
    let sequencer_changed = sequencer.step_with_held_bearings(
        simulation.effective_graph(),
        keys,
        keyboard_controller,
        tick,
        &suspended,
        &frozen.suspended_bearings(simulation),
    );
    let measured_speeds =
        measured_engine_speeds(simulation.effective_graph(), simulation, sequencer);
    let gearbox_changed = gearboxes.step_with_suspension(
        simulation.effective_graph(),
        sequencer,
        keyboard,
        gearbox_keyboard_controller,
        tick,
        &measured_speeds,
        false,
        &suspended,
    );
    if sequencer_changed || gearbox_changed {
        state.drive_rows_dirty = true;
    }
}

/// Signed joint speeds from the two transform snapshots already read for rendering.
#[expect(clippy::cast_precision_loss)]
pub(crate) fn measured_engine_speeds(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    sequencer: &DriveSequencer,
) -> Vec<(PartId, EngineKind, f32)> {
    let Some(creation) = simulation.creation.as_ref() else {
        return Vec::new();
    };
    let tick_delta = simulation
        .snapshot_tick
        .saturating_sub(simulation.previous_snapshot_tick);
    if tick_delta == 0 || simulation.previous_transforms.len() != simulation.transforms.len() {
        return Vec::new();
    }
    let delta_seconds = tick_delta as f32 * TICK_SECONDS_F32;
    let mut result = Vec::<(PartId, EngineKind, f32)>::new();
    for row in sequencer.rows() {
        let Some(link) = graph.drive_link(row.link) else {
            continue;
        };
        let Some(bearing) = creation
            .bearings
            .iter()
            .find(|bearing| bearing.coordinate_index == Some(row.coordinate))
        else {
            continue;
        };
        let a = bearing.compound_a as usize;
        let b = bearing.compound_b as usize;
        let (Some(previous_a), Some(previous_b), Some(current_a), Some(current_b)) = (
            simulation.previous_transforms.get(a),
            simulation.previous_transforms.get(b),
            simulation.transforms.get(a),
            simulation.transforms.get(b),
        ) else {
            continue;
        };
        let speed = if bearing.kind.is_translational() {
            let displacement = |a: &GpuTransform, b: &GpuTransform| {
                let rotation_a = Quat::from_array(a.rotation);
                let rotation_b = Quat::from_array(b.rotation);
                let separation = Vec3::from_slice(&b.position[..3])
                    + rotation_b * bearing.local_anchor_b
                    - Vec3::from_slice(&a.position[..3])
                    - rotation_a * bearing.local_anchor_a;
                separation.dot(rotation_a * bearing.local_axis_a)
            };
            (displacement(current_a, current_b) - displacement(previous_a, previous_b))
                / delta_seconds
                / mechanic_core::LINEAR_METERS_PER_RADIAN
        } else {
            signed_joint_speed(
                Quat::from_array(previous_a.rotation),
                Quat::from_array(previous_b.rotation),
                Quat::from_array(current_a.rotation),
                Quat::from_array(current_b.rotation),
                bearing.local_axis_a,
                delta_seconds,
            )
        };
        for kind in [EngineKind::Electric, EngineKind::Gas] {
            let powered = match kind {
                EngineKind::Electric => link.actuator.uses_electric(),
                EngineKind::Gas => link.actuator.uses_gas(),
            };
            if !powered {
                continue;
            }
            if let Some(entry) = result.iter_mut().find(|(controller, candidate, _)| {
                *controller == link.controller && *candidate == kind
            }) {
                if speed.abs() > entry.2.abs() {
                    entry.2 = speed;
                }
            } else {
                result.push((link.controller, kind, speed));
            }
        }
    }
    result
}

pub(crate) fn signed_joint_speed(
    previous_a: Quat,
    previous_b: Quat,
    current_a: Quat,
    current_b: Quat,
    local_axis_a: Vec3,
    delta_seconds: f32,
) -> f32 {
    let angular_a = (current_a * previous_a.inverse()).to_scaled_axis() / delta_seconds;
    let angular_b = (current_b * previous_b.inverse()).to_scaled_axis() / delta_seconds;
    (angular_b - angular_a).dot(current_a * local_axis_a)
}

#[cfg(test)]
mod tests;
