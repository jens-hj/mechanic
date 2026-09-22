//! What the panel shows, and what it asks the graph to change.
//!
//! The Mosaic tree is built once and driven by reactive state, while the
//! construction graph is the truth about the machine. Those two cannot be the
//! same object, so the panel renders a [`PanelModel`] snapshot and sends back
//! [`PanelEdit`] intents. Everything here is plain data: no Bevy, no Mosaic,
//! and no allocation the view has to own.
//!
//! Units cross over here and nowhere else. The graph stores radians and
//! radians per second; the panel reads degrees, because a joint's travel is
//! something a person describes in degrees.

use crate::control_panel::SpeedUnit;
use mechanic_core::{
    ActuatorAssignment, ActuatorInventory, DriveDwell, DriveKey, DriveLimits, DriveLinkId,
    DriveName, DriveProgram, DriveRelease, DriveState, DriveTarget, DriveTrigger, EngineKind,
    GearKeyChord, GearboxConfig, LinearDriveLimits, MAX_DRIVE_DWELL_SECONDS,
    MAX_DRIVE_LIMIT_RADIANS, MAX_DRIVE_SPEED_RAD_S, MAX_DRIVE_STATES,
    MAX_PROGRAMMED_TRAVEL_RADIANS, MIN_DWELL_SECONDS, MIN_PROGRAMMED_TRAVEL_METERS,
    MIN_PROGRAMMED_TRAVEL_RADIANS, ShiftMode,
};

/// Smallest travel range the grips may close to, in degrees. Two limits that
/// meet would leave the joint with nowhere to go.
const MIN_TRAVEL_SPAN_DEGREES: f32 = MIN_PROGRAMMED_TRAVEL_RADIANS.to_degrees();

/// Furthest a travel limit may sit from centre, in degrees.
const MAX_TRAVEL_DEGREES: f32 = MAX_PROGRAMMED_TRAVEL_RADIANS.to_degrees();

/// What a state asks of its joint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Hold a position.
    Angle,
    /// Spin at a rate.
    Speed,
}

/// One of the three ready-made programs the sidebar offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Preset {
    /// Hold left or right, spring back when released.
    Steer,
    /// Forward and reverse.
    Drive,
    /// Run and stop, both latching.
    Spin,
}

/// One thing the panel asks the graph to change.
///
/// Every variant names the joint's own view of the change — degrees, a
/// character, a state number — so the view never has to know what the graph
/// stores.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PanelEdit {
    /// Rename the joint.
    SetName(String),
    /// Toggle speed readouts between RPM and degrees per second.
    ToggleSpeedUnit,
    /// Step unpowered, motor, and Servo assignment modes.
    CycleActuator,
    /// Step the electric contribution through 0, 25, 50, 75, and 100 percent.
    CycleElectric,
    /// Step the gas contribution through 0, 25, 50, 75, and 100 percent.
    CycleGas,
    /// Turn travel limits on at a default range, or off.
    ToggleTravel,
    /// Move both travel limits, in degrees.
    SetTravel { min: f32, max: f32 },
    /// Repeat the sequence, or run it once.
    ToggleLoop,
    /// Replace the whole program with a ready-made one.
    ApplyPreset(Preset),
    /// Hold an angle, or spin at a speed.
    SetMode { state: u8, mode: Mode },
    /// The state's target, in degrees or degrees per second.
    SetValue { state: u8, value: f32 },
    /// Bind a key, taking it from whichever sibling held it.
    BindKey { state: u8, key: char },
    /// Unbind the key, which also drops the release behaviour it carried.
    ClearKey { state: u8 },
    /// Step what happens on release: stay, then each state in turn.
    CycleRelease { state: u8 },
    /// Hand off to a state on release, or latch when `None`.
    SetRelease { state: u8, target: Option<u8> },
    /// Give the state a dwell, or take it away.
    ToggleDwell { state: u8 },
    /// How long the state waits before handing off, in seconds.
    SetDwell { state: u8, seconds: f32 },
    /// Which state the dwell hands off to, giving it a dwell if it had none.
    SetDwellTarget { state: u8, target: u8 },
    /// Append a state, copying the last one's mode.
    AddState,
    /// Remove a state.
    RemoveState { state: u8 },
}

/// One change the panel is asking for, and which joint it lands on.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Intent {
    /// The wire whose row the change is written to.
    pub(crate) lane: DriveLinkId,
    /// What to change.
    pub(crate) edit: PanelEdit,
    /// Whether this is one step of a gesture still in progress. A drag writes
    /// on every pointer move, and only the last of them belongs in history.
    pub(crate) transient: bool,
}

/// One engine-lane edit requested by the panel.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GearboxIntent {
    pub(crate) kind: EngineKind,
    pub(crate) edit: GearboxEdit,
    pub(crate) transient: bool,
}

/// Editable persistent settings in one engine lane.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum GearboxEdit {
    Mode(ShiftMode),
    Ratio {
        index: usize,
        value: f32,
    },
    Bindings {
        up: GearKeyChord,
        down: GearKeyChord,
    },
    ReverseGears(u8),
}

/// The joint's whole configuration, as the graph stores it.
type Wire = (DriveLimits, DriveProgram, DriveName);

/// Applies SI edits while keeping programmable travel within the physical stops.
pub(crate) fn apply_linear_edit(
    limits: DriveLimits,
    mut linear: LinearDriveLimits,
    mut program: DriveProgram,
    name: DriveName,
    physical: [f32; 2],
    edit: &PanelEdit,
) -> Option<(DriveLimits, LinearDriveLimits, DriveProgram, DriveName)> {
    let bounds = (linear.minimum(), linear.maximum());
    let clamp_target = |target| match target {
        DriveTarget::LinearPosition(value) => {
            DriveTarget::LinearPosition(value.clamp(bounds.0, bounds.1))
        }
        DriveTarget::LinearSpeed(value) => {
            DriveTarget::LinearSpeed(value.clamp(-linear.max_speed(), linear.max_speed()))
        }
        other => other,
    };
    match edit {
        PanelEdit::SetTravel { min, max } => {
            if !min.is_finite() || !max.is_finite() {
                return None;
            }
            let min = min.clamp(physical[0], physical[1] - MIN_PROGRAMMED_TRAVEL_METERS);
            let max = max.clamp(min + MIN_PROGRAMMED_TRAVEL_METERS, physical[1]);
            linear =
                LinearDriveLimits::new(linear.max_speed(), linear.max_force(), min, max).ok()?;
        }
        PanelEdit::ToggleTravel => {
            // A physical stop is never disabled. This resets the programmed envelope.
            linear = LinearDriveLimits::new(
                linear.max_speed(),
                linear.max_force(),
                physical[0],
                physical[1],
            )
            .ok()?;
        }
        PanelEdit::SetMode { state, mode } => {
            let current = program.state(*state)?;
            let value = reading(current.target());
            let target = match mode {
                Mode::Angle => DriveTarget::LinearPosition(value),
                Mode::Speed => DriveTarget::LinearSpeed(value),
            };
            program = program
                .with_state(*state, current.with_target(clamp_target(target)).ok()?)
                .ok()?;
        }
        PanelEdit::SetValue { state, value } => {
            if !value.is_finite() {
                return None;
            }
            let current = program.state(*state)?;
            let target = match current.target() {
                DriveTarget::LinearPosition(_) => DriveTarget::LinearPosition(*value),
                DriveTarget::LinearSpeed(_) => DriveTarget::LinearSpeed(*value),
                _ => return None,
            };
            program = program
                .with_state(*state, current.with_target(clamp_target(target)).ok()?)
                .ok()?;
        }
        PanelEdit::ApplyPreset(preset) => {
            let position =
                |value| DriveState::new(clamp_target(DriveTarget::LinearPosition(value))).ok();
            let speed = |value| DriveState::new(clamp_target(DriveTarget::LinearSpeed(value))).ok();
            let keyed = |state: DriveState, key, release| {
                DriveKey::new(key)
                    .map(|key| state.with_trigger(Some(DriveTrigger::new(key, release))))
            };
            let states = match preset {
                Preset::Steer => vec![
                    position(0.0)?,
                    keyed(position(bounds.0 * 0.7)?, 'A', DriveRelease::RevertTo(0))?,
                    keyed(position(bounds.1 * 0.7)?, 'D', DriveRelease::RevertTo(0))?,
                ],
                Preset::Drive => vec![
                    speed(0.0)?,
                    keyed(speed(linear.max_speed())?, 'W', DriveRelease::RevertTo(0))?,
                    keyed(
                        speed(-linear.max_speed() * 0.7)?,
                        'S',
                        DriveRelease::RevertTo(0),
                    )?,
                ],
                Preset::Spin => vec![
                    keyed(speed(0.0)?, 'Z', DriveRelease::Latch)?,
                    keyed(speed(linear.max_speed())?, 'X', DriveRelease::Latch)?,
                ],
            };
            program = DriveProgram::new(&states, false).ok()?;
        }
        _ => {
            let (limits, program, name) = apply_edit(limits, program, name, edit)?;
            return Some((limits, linear, clamp_linear_program(program, linear), name));
        }
    }
    Some((limits, linear, clamp_linear_program(program, linear), name))
}

fn clamp_linear_program(program: DriveProgram, limits: LinearDriveLimits) -> DriveProgram {
    fold_states(&program, |state| {
        let target = match state.target() {
            DriveTarget::LinearPosition(value) => {
                DriveTarget::LinearPosition(value.clamp(limits.minimum(), limits.maximum()))
            }
            DriveTarget::LinearSpeed(value) => {
                DriveTarget::LinearSpeed(value.clamp(-limits.max_speed(), limits.max_speed()))
            }
            _ => return None,
        };
        state.with_target(target).ok()
    })
}

/// Folds one edit into a joint's configuration.
///
/// Returns `None` when the edit cannot apply — an unparseable number, a value
/// out of range, a state that is not there — leaving the joint as it was. That
/// is the same answer for "you typed nonsense" and "that click does nothing",
/// because in both cases the right behaviour is to change nothing.
#[expect(
    clippy::too_many_lines,
    reason = "one match arm per thing the panel can change"
)]
pub(crate) fn apply_edit(
    limits: DriveLimits,
    program: DriveProgram,
    name: DriveName,
    edit: &PanelEdit,
) -> Option<Wire> {
    match edit {
        PanelEdit::SetName(text) => Some((limits, program, DriveName::new(text))),

        PanelEdit::ToggleSpeedUnit
        | PanelEdit::CycleActuator
        | PanelEdit::CycleElectric
        | PanelEdit::CycleGas => None,

        PanelEdit::ToggleTravel => {
            let travel = if limits.angle_limits().is_some() {
                None
            } else {
                Some((-45f32.to_radians(), 45f32.to_radians()))
            };
            let limits = limits.with_angle_limits(travel).ok()?;
            Some((limits, clamped_angles(&program, limits), name))
        }

        PanelEdit::SetTravel { min, max } => {
            let min = min.clamp(
                -MAX_TRAVEL_DEGREES,
                MAX_TRAVEL_DEGREES - MIN_TRAVEL_SPAN_DEGREES,
            );
            let max = max.clamp(min + MIN_TRAVEL_SPAN_DEGREES, MAX_TRAVEL_DEGREES);
            let limits = limits
                .with_angle_limits(Some((min.to_radians(), max.to_radians())))
                .ok()?;
            Some((limits, clamped_angles(&program, limits), name))
        }

        PanelEdit::ToggleLoop => Some((limits, program.with_loops(!program.loops()), name)),

        PanelEdit::ApplyPreset(preset) => {
            let (limits, program) = preset_program(*preset, limits)?;
            Some((limits, program, name))
        }

        PanelEdit::SetMode { state, mode } => {
            let current = program.state(*state)?;
            // The number carries across and clamps into the new unit's range,
            // so switching mode never fails and never silently loses what was
            // typed.
            let target = match mode {
                Mode::Angle => DriveTarget::Angle(
                    reading(current.target())
                        .clamp(-MAX_DRIVE_LIMIT_RADIANS, MAX_DRIVE_LIMIT_RADIANS),
                ),
                Mode::Speed => DriveTarget::Speed(
                    reading(current.target()).clamp(-MAX_DRIVE_SPEED_RAD_S, MAX_DRIVE_SPEED_RAD_S),
                ),
            };
            let program = program
                .with_state(*state, current.with_target(target).ok()?)
                .ok()?;
            Some((limits, clamped_angles(&program, limits), name))
        }

        PanelEdit::SetValue { state, value } => {
            let current = program.state(*state)?;
            let target = match current.target() {
                DriveTarget::LinearPosition(_) => DriveTarget::LinearPosition(*value),
                DriveTarget::LinearSpeed(_) => DriveTarget::LinearSpeed(*value),
                DriveTarget::Angle(_) => {
                    let radians = value.to_radians();
                    let (low, high) = limits
                        .angle_limits()
                        .unwrap_or((-MAX_DRIVE_LIMIT_RADIANS, MAX_DRIVE_LIMIT_RADIANS));
                    DriveTarget::Angle(radians.clamp(low, high))
                }
                DriveTarget::Speed(_) => DriveTarget::Speed(
                    value
                        .to_radians()
                        .clamp(-MAX_DRIVE_SPEED_RAD_S, MAX_DRIVE_SPEED_RAD_S),
                ),
            };
            Some((
                limits,
                program
                    .with_state(*state, current.with_target(target).ok()?)
                    .ok()?,
                name,
            ))
        }

        PanelEdit::BindKey { state, key } => {
            let key = DriveKey::new(*key)?;
            let current = program.state(*state)?;
            let release = current
                .trigger()
                .map_or(DriveRelease::Latch, DriveTrigger::release);
            // One key can only mean one state on a joint, so binding it takes
            // it from whichever sibling had it rather than refusing.
            let program = released_key(&program, key, *state)?;
            let current = program.state(*state).unwrap_or(current);
            Some((
                limits,
                program
                    .with_state(
                        *state,
                        current.with_trigger(Some(DriveTrigger::new(key, release))),
                    )
                    .ok()?,
                name,
            ))
        }

        PanelEdit::ClearKey { state } => {
            let current = program.state(*state)?;
            current.trigger()?;
            Some((
                limits,
                program
                    .with_state(*state, current.with_trigger(None))
                    .ok()?,
                name,
            ))
        }

        PanelEdit::CycleRelease { state } => {
            let current = program.state(*state)?;
            let trigger = current.trigger()?;
            let target = stepped(trigger.release().target(), program.len());
            Some((limits, with_release(&program, *state, target)?, name))
        }

        PanelEdit::SetRelease { state, target } => {
            program.state(*state)?.trigger()?;
            Some((limits, with_release(&program, *state, *target)?, name))
        }

        PanelEdit::ToggleDwell { state } => {
            let current = program.state(*state)?;
            let dwell = if current.dwell().is_some() {
                None
            } else {
                // A fresh dwell hands off to the next state round, which is
                // the sequence a person almost always means.
                let next = (usize::from(*state) + 1) % program.len().max(1);
                Some(DriveDwell::new(1.0, Some(u8::try_from(next).ok()?)).ok()?)
            };
            Some((
                limits,
                program.with_state(*state, current.with_dwell(dwell)).ok()?,
                name,
            ))
        }

        PanelEdit::SetDwell { state, seconds } => {
            let current = program.state(*state)?;
            // A dwell of nothing is not a dwell; the port toggle is how a
            // state stops waiting.
            let seconds = seconds.clamp(MIN_DWELL_SECONDS, MAX_DRIVE_DWELL_SECONDS);
            let dwell =
                DriveDwell::new(seconds, current.dwell().and_then(DriveDwell::next)).ok()?;
            Some((
                limits,
                program
                    .with_state(*state, current.with_dwell(Some(dwell)))
                    .ok()?,
                name,
            ))
        }

        PanelEdit::SetDwellTarget { state, target } => {
            let current = program.state(*state)?;
            if usize::from(*target) >= program.len() {
                return None;
            }
            let seconds = current.dwell().map_or(1.0, DriveDwell::seconds);
            let dwell = DriveDwell::new(seconds, Some(*target)).ok()?;
            Some((
                limits,
                program
                    .with_state(*state, current.with_dwell(Some(dwell)))
                    .ok()?,
                name,
            ))
        }

        PanelEdit::AddState => {
            if program.len() >= MAX_DRIVE_STATES {
                return None;
            }
            let last = u8::try_from(program.len().checked_sub(1)?).ok()?;
            // The new state inherits the last one's mode and rests at zero,
            // which is what an empty dial reads as.
            let target = match program.state(last)?.target() {
                DriveTarget::Angle(_) => DriveTarget::Angle(0.0),
                DriveTarget::Speed(_) => DriveTarget::Speed(0.0),
                DriveTarget::LinearPosition(_) => DriveTarget::LinearPosition(0.0),
                DriveTarget::LinearSpeed(_) => DriveTarget::LinearSpeed(0.0),
            };
            Some((
                limits,
                program
                    .with_pushed_state(DriveState::new(target).ok()?)
                    .ok()?,
                name,
            ))
        }

        PanelEdit::RemoveState { state } => {
            Some((limits, program.with_removed_state(*state).ok()?, name))
        }
    }
}

/// The raw number behind a target, whichever unit it is in.
const fn reading(target: DriveTarget) -> f32 {
    match target {
        DriveTarget::Angle(angle) | DriveTarget::LinearPosition(angle) => angle,
        DriveTarget::Speed(speed) | DriveTarget::LinearSpeed(speed) => speed,
    }
}

/// Replaces one state's release behaviour, keeping its key.
fn with_release(program: &DriveProgram, state: u8, target: Option<u8>) -> Option<DriveProgram> {
    let current = program.state(state)?;
    let trigger = current.trigger()?;
    let release = target.map_or(DriveRelease::Latch, DriveRelease::RevertTo);
    program
        .with_state(
            state,
            current.with_trigger(Some(DriveTrigger::new(trigger.key(), release))),
        )
        .ok()
}

/// Steps a state reference through `stay -> S1 -> .. -> Sn -> stay`.
fn stepped(current: Option<u8>, len: usize) -> Option<u8> {
    let len = u8::try_from(len).ok()?;
    match current {
        None => Some(0),
        Some(index) if index + 1 < len => Some(index + 1),
        Some(_) => None,
    }
}

/// Frees `key` from every state but `keep`, so it can be bound there.
fn released_key(program: &DriveProgram, key: DriveKey, keep: u8) -> Option<DriveProgram> {
    let mut next = *program;
    for index in 0..u8::try_from(program.len()).ok()? {
        if index == keep {
            continue;
        }
        let state = next.state(index)?;
        if state.trigger().is_some_and(|trigger| trigger.key() == key) {
            next = next.with_state(index, state.with_trigger(None)).ok()?;
        }
    }
    Some(next)
}

/// Pulls every held angle back inside the joint's travel limits.
fn clamped_angles(program: &DriveProgram, limits: DriveLimits) -> DriveProgram {
    let Some((low, high)) = limits.angle_limits() else {
        return *program;
    };
    fold_states(program, |state| match state.target() {
        DriveTarget::Angle(angle) => state
            .with_target(DriveTarget::Angle(angle.clamp(low, high)))
            .ok(),
        DriveTarget::Speed(_) | DriveTarget::LinearPosition(_) | DriveTarget::LinearSpeed(_) => {
            None
        }
    })
}

/// Rewrites every state a mapping has an opinion about, keeping the rest.
///
/// A rewrite that fails validation is dropped rather than failing the whole
/// program: these are corrections applied on the player's behalf, and one that
/// cannot be made should not undo the edit that prompted it.
fn fold_states(
    program: &DriveProgram,
    mut rewrite: impl FnMut(DriveState) -> Option<DriveState>,
) -> DriveProgram {
    let mut next = *program;
    for index in 0..program.len() {
        let Ok(index) = u8::try_from(index) else {
            break;
        };
        let Some(state) = next.state(index) else {
            break;
        };
        let Some(rewritten) = rewrite(state) else {
            continue;
        };
        if let Ok(updated) = next.with_state(index, rewritten) {
            next = updated;
        }
    }
    next
}

/// The program and envelope one preset installs.
fn preset_program(preset: Preset, limits: DriveLimits) -> Option<(DriveLimits, DriveProgram)> {
    let held = |degrees: f32| DriveState::new(DriveTarget::Angle(degrees.to_radians())).ok();
    let spun = |rad_s: f32| DriveState::new(DriveTarget::Speed(rad_s)).ok();
    let keyed = |state: DriveState, key: char, release: DriveRelease| {
        DriveKey::new(key).map(|key| state.with_trigger(Some(DriveTrigger::new(key, release))))
    };
    let top = limits.max_speed_rad_s();

    match preset {
        Preset::Steer => {
            let limits = limits
                .with_angle_limits(Some((-45f32.to_radians(), 45f32.to_radians())))
                .ok()?;
            let states = [
                held(0.0)?,
                keyed(held(-30.0)?, 'A', DriveRelease::RevertTo(0))?,
                keyed(held(30.0)?, 'D', DriveRelease::RevertTo(0))?,
            ];
            Some((limits, DriveProgram::new(&states, false).ok()?))
        }
        Preset::Drive => {
            let limits = limits.with_angle_limits(None).ok()?;
            let states = [
                spun(0.0)?,
                keyed(spun(top)?, 'W', DriveRelease::RevertTo(0))?,
                keyed(spun(-top * 0.7)?, 'S', DriveRelease::RevertTo(0))?,
            ];
            Some((limits, DriveProgram::new(&states, false).ok()?))
        }
        Preset::Spin => {
            let limits = limits.with_angle_limits(None).ok()?;
            let states = [
                keyed(spun(0.0)?, 'Z', DriveRelease::Latch)?,
                keyed(spun(top)?, 'X', DriveRelease::Latch)?,
            ];
            Some((limits, DriveProgram::new(&states, false).ok()?))
        }
    }
}

/// Where a force stops reading in newtons and starts reading in kilonewtons.
const KILONEWTON: f32 = 1_000.0;

/// A rail's force, in the unit that keeps it to four significant figures.
///
/// A rail strong enough to carry a machine is tens of thousands of newtons,
/// and six digits in a tile this size is a number nobody reads — the
/// kilonewtons are what the rail is actually specified in.
pub(crate) fn force_text(newtons: f32) -> String {
    if newtons.abs() < KILONEWTON {
        return format!("{newtons:.0} N");
    }
    format!("{:.1} kN", newtons / KILONEWTON)
}

/// A dwell written the way it is typed and scrubbed: as many decimals as the
/// number actually has, up to the tenth of a second a scrub can reach.
///
/// Fixed to one decimal it could not show a quarter-second step at all, and
/// every whole second would read as a measurement it is not.
pub(crate) fn dwell_text(seconds: f32) -> String {
    let mut text = format!("{:.2}", (seconds * 100.0).round() / 100.0);
    while text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.pop();
    }
    text
}

/// One wire drawn between two state cards.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WireModel {
    /// Card the wire leaves.
    pub(crate) source: usize,
    /// Card the wire arrives at.
    pub(crate) target: usize,
    /// What the pill on the wire reads.
    pub(crate) label: String,
}

/// One state, as the panel draws it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StateModel {
    /// Whether the dial reads an angle or a speed.
    pub(crate) mode: Mode,
    /// The dial's reading, in degrees or degrees per second.
    pub(crate) value: f32,
    /// The bound key, if any.
    pub(crate) key: Option<char>,
    /// Which state the key going up hands off to. `None` latches.
    pub(crate) release: Option<u8>,
    /// How long the state waits, and where it hands off to.
    pub(crate) dwell: Option<(f32, u8)>,
}

impl StateModel {
    /// A state that holds still, for reading a card whose joint has gone.
    pub(crate) const fn resting() -> Self {
        Self {
            mode: Mode::Speed,
            value: 0.0,
            key: None,
            release: None,
            dwell: None,
        }
    }

    /// How far round the dial this state's reading sits, in degrees.
    ///
    /// An angle is its own reading. A speed is a fraction of the joint's
    /// ceiling, drawn as half a turn either way, so a dial reads the same
    /// whatever the joint's top speed happens to be.
    pub(crate) fn sweep(&self, ceiling: f32) -> f32 {
        match self.mode {
            Mode::Angle => self.value.clamp(-360.0, 360.0),
            Mode::Speed if ceiling > 0.0 => (self.value / ceiling).clamp(-1.0, 1.0) * 180.0,
            Mode::Speed => 0.0,
        }
    }

    /// Whether this state asks for more speed than the joint can give.
    pub(crate) fn overspeed(&self, ceiling: f32) -> bool {
        self.mode == Mode::Speed && self.value.abs() > ceiling
    }
}

/// One joint's lane.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LaneModel {
    /// Linear lanes read metres, metres per second, and newtons.
    pub(crate) is_linear: bool,
    /// Immutable physical rail endpoints in metres.
    pub(crate) physical_travel: Option<(f32, f32)>,
    /// The wire this lane speaks for, which is what keeps a lane's elements
    /// the same elements when the joints around it change.
    pub(crate) id: DriveLinkId,
    /// The joint's number, which the badge shows and an unnamed joint is
    /// called by.
    pub(crate) number: usize,
    /// What the joint is called. Empty falls back to the number.
    pub(crate) name: String,
    /// Fastest the joint may turn, in degrees per second.
    pub(crate) speed: f32,
    /// Strongest torque it may apply. Infinite means unlimited.
    pub(crate) torque: f32,
    /// Hardware family assigned to this joint.
    pub(crate) actuator: ActuatorAssignment,
    /// Unit used by continuous speed readouts.
    pub(crate) speed_unit: SpeedUnit,
    /// Travel limits in degrees, or `None` when the joint turns freely.
    pub(crate) travel: Option<(f32, f32)>,
    /// Whether the sequence repeats.
    pub(crate) loops: bool,
    /// The states, in order.
    pub(crate) states: Vec<StateModel>,
    /// Wires above the cards: what a key going up hands off to.
    pub(crate) release_wires: Vec<WireModel>,
    /// Wires below the cards: what a dwell hands off to.
    pub(crate) dwell_wires: Vec<WireModel>,
}

impl LaneModel {
    /// Reads one joint's configuration into what the panel draws.
    #[expect(
        clippy::too_many_arguments,
        reason = "A flat immutable snapshot at the graph/UI seam"
    )]
    pub(crate) fn capture(
        id: DriveLinkId,
        number: usize,
        limits: DriveLimits,
        program: &DriveProgram,
        name: &DriveName,
        actuator: ActuatorAssignment,
        speed_unit: SpeedUnit,
        max_speed_rad_s: f32,
        effective_torque: f32,
    ) -> Self {
        let states: Vec<StateModel> = program
            .states()
            .iter()
            .map(|state| StateModel {
                mode: match state.target() {
                    DriveTarget::Angle(_) | DriveTarget::LinearPosition(_) => Mode::Angle,
                    DriveTarget::Speed(_) | DriveTarget::LinearSpeed(_) => Mode::Speed,
                },
                value: match state.target() {
                    DriveTarget::LinearPosition(value) | DriveTarget::LinearSpeed(value) => value,
                    DriveTarget::Angle(angle) => angle.to_degrees(),
                    DriveTarget::Speed(speed) => match speed_unit {
                        SpeedUnit::Rpm => mechanic_core::rad_s_to_rpm(speed),
                        SpeedUnit::DegreesPerSecond => speed.to_degrees(),
                    },
                },
                key: state.trigger().map(|trigger| trigger.key().symbol()),
                release: state
                    .trigger()
                    .and_then(|trigger| trigger.release().target()),
                dwell: state.dwell().map(|dwell| {
                    // A dwell with no named target falls through to the next
                    // state, wrapping at the end.
                    let next = dwell.next().unwrap_or(0);
                    (dwell.seconds(), next)
                }),
            })
            .collect();

        Self {
            is_linear: program
                .states()
                .iter()
                .any(|state| state.target().is_linear()),
            physical_travel: None,
            id,
            number,
            name: name.as_str().to_owned(),
            speed: match speed_unit {
                SpeedUnit::Rpm => mechanic_core::rad_s_to_rpm(max_speed_rad_s),
                SpeedUnit::DegreesPerSecond => max_speed_rad_s.to_degrees(),
            },
            torque: effective_torque,
            actuator,
            speed_unit,
            travel: limits
                .angle_limits()
                .map(|(low, high)| (low.to_degrees(), high.to_degrees())),
            loops: program.loops(),
            release_wires: wires(&states, WireKind::Release),
            dwell_wires: wires(&states, WireKind::Dwell),
            states,
        }
    }

    /// Applies the rail's SI envelope after capturing the shared lane state.
    pub(crate) fn with_linear_limits(
        mut self,
        limits: LinearDriveLimits,
        physical: [f32; 2],
    ) -> Self {
        self.is_linear = true;
        let angular_speed = match self.speed_unit {
            SpeedUnit::Rpm => mechanic_core::rpm_to_rad_s(self.speed),
            SpeedUnit::DegreesPerSecond => self.speed.to_radians(),
        };
        self.speed = limits
            .max_speed()
            .min(angular_speed * mechanic_core::LINEAR_METERS_PER_RADIAN);
        self.torque = limits
            .max_force()
            .min(self.torque / mechanic_core::LINEAR_METERS_PER_RADIAN);
        self.travel = Some((limits.minimum(), limits.maximum()));
        self.physical_travel = Some((physical[0], physical[1]));
        self
    }

    /// What the speed chip reads.
    pub(crate) fn speed_text(&self) -> String {
        if self.is_linear {
            return format!("{:.3} m/s", self.speed);
        }
        if matches!(self.actuator, ActuatorAssignment::Motor { .. }) {
            return "GEARED".to_owned();
        }
        match self.speed_unit {
            SpeedUnit::Rpm => format!("{:.0} RPM", self.speed),
            SpeedUnit::DegreesPerSecond => format!("{:.0} °/s", self.speed),
        }
    }

    /// What the torque chip's heading reads.
    pub(crate) const fn torque_label(&self) -> &'static str {
        if self.is_linear {
            return "FORCE";
        }
        match self.actuator {
            ActuatorAssignment::Unpowered => "ACTUATOR",
            ActuatorAssignment::Servo => "SERVO TORQUE",
            ActuatorAssignment::Motor { .. } => "MOTOR TORQUE",
        }
    }

    /// What the torque chip reads.
    pub(crate) fn torque_text(&self) -> String {
        if self.actuator == ActuatorAssignment::Unpowered {
            return "NONE".to_owned();
        }
        if self.is_linear {
            return force_text(self.torque);
        }
        match self.actuator {
            ActuatorAssignment::Unpowered => "NONE".to_owned(),
            ActuatorAssignment::Servo => format!("{:.0} N·m", self.torque),
            ActuatorAssignment::Motor { .. } => "GEARED".to_owned(),
        }
    }

    pub(crate) const fn speed_unit_text(&self) -> &'static str {
        if self.is_linear {
            return "m/s";
        }
        match self.speed_unit {
            SpeedUnit::Rpm => "RPM",
            SpeedUnit::DegreesPerSecond => "°/s",
        }
    }

    pub(crate) fn electric_text(&self) -> String {
        format!("{}%", self.actuator.electric_percent())
    }

    pub(crate) fn gas_text(&self) -> String {
        format!("{}%", self.actuator.gas_percent())
    }

    /// What the travel chip reads.
    pub(crate) fn travel_text(&self) -> String {
        match self.travel {
            Some((low, high)) if self.is_linear => format!("{low:+.3} to {high:+.3} m"),
            Some((low, high)) => format!("{low:.0}° to {high:.0}°"),
            None => "free".to_owned(),
        }
    }

    /// What the repeat chip reads.
    pub(crate) const fn loop_text(&self) -> &'static str {
        if self.loops { "loop" } else { "once" }
    }

    /// What the joint is called in prose.
    pub(crate) fn title(&self) -> String {
        if self.name.is_empty() {
            format!("Joint {}", self.number)
        } else {
            self.name.clone()
        }
    }
}

/// Which family of wire is being collected.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WireKind {
    /// A key going up hands off.
    Release,
    /// A timer runs out and hands off.
    Dwell,
}

/// The wires of one family, ordered so a long hop sits nearest the cards.
///
/// Ranking longest-first is what keeps wires from crossing: a short hop drawn
/// further out can pass cleanly over a long one running beneath it.
fn wires(states: &[StateModel], kind: WireKind) -> Vec<WireModel> {
    let mut found: Vec<WireModel> = states
        .iter()
        .enumerate()
        .filter_map(|(source, state)| match kind {
            WireKind::Release => {
                let key = state.key?;
                let target = state.release?;
                Some(WireModel {
                    source,
                    target: usize::from(target).min(states.len().saturating_sub(1)),
                    label: key.to_string(),
                })
            }
            WireKind::Dwell => {
                let (seconds, target) = state.dwell?;
                Some(WireModel {
                    source,
                    target: usize::from(target).min(states.len().saturating_sub(1)),
                    label: format!("{} s", dwell_text(seconds)),
                })
            }
        })
        .collect();
    found.sort_by_key(|wire| std::cmp::Reverse(wire.source.abs_diff(wire.target)));
    found
}

impl WireModel {
    /// The four turning points of this wire, and the lane it runs along.
    ///
    /// `rank` is how many wires of the same family are drawn nearer the cards:
    /// ranking the longest hop nearest is what keeps wires from crossing.
    pub(crate) fn route(&self, rank: usize, top: f32, release: bool) -> [(f32, f32); 4] {
        let (from, to) = if release {
            crate::ui::control_block::geometry::release_wire_ends(self.source, self.target, top)
        } else {
            crate::ui::control_block::geometry::dwell_wire_ends(self.source, self.target, top)
        };
        let lane = if release {
            crate::ui::control_block::geometry::release_wire_lane(top, rank)
        } else {
            crate::ui::control_block::geometry::dwell_wire_lane(top, rank)
        };
        crate::ui::control_block::geometry::route_points(from, to, lane)
    }

    /// Where the wire's label sits: the midpoint of its run along the lane.
    pub(crate) fn label_at(&self, rank: usize, top: f32, release: bool) -> (f32, f32) {
        let points = self.route(rank, top, release);
        (f32::midpoint(points[1].0, points[2].0), points[1].1)
    }
}

/// Everything the panel draws.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PanelModel {
    /// Whether the panel is showing at all.
    pub(crate) open: bool,
    /// One lane per joint the control block drives.
    pub(crate) lanes: Vec<LaneModel>,
    /// Electric then gas engine lanes present in the Controller module.
    pub(crate) engine_lanes: Vec<EngineLaneModel>,
    /// Bearing-port usage supplied by the controller's attached actuators.
    pub(crate) hardware: HardwareModel,
    /// A vehicle key overlaps a rebindable gameplay action.
    pub(crate) gameplay_binding_conflict: bool,
}

/// Static build information and persistent gearing for one engine family.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EngineLaneModel {
    pub(crate) kind: EngineKind,
    pub(crate) engine_count: u32,
    pub(crate) combined_stall_torque: f32,
    pub(crate) base_rpm: f32,
    pub(crate) slots: BearingSlots,
    pub(crate) transmission_depth: Option<u8>,
    pub(crate) physical_depths: Vec<u8>,
    pub(crate) mismatch: bool,
    pub(crate) config: Option<GearboxConfig>,
    pub(crate) active_gear: Option<usize>,
    pub(crate) binding_conflict: bool,
}

impl EngineLaneModel {
    pub(crate) const fn label(&self) -> &'static str {
        match self.kind {
            EngineKind::Electric => "ELECTRIC ENGINE LINE",
            EngineKind::Gas => "GAS ENGINE LINE",
        }
    }
}

/// Bearing-port usage for every actuator family in a Controller module.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HardwareModel {
    pub(crate) electric: BearingSlots,
    pub(crate) gas: BearingSlots,
    pub(crate) servo: BearingSlots,
}

impl From<ActuatorInventory> for HardwareModel {
    fn from(inventory: ActuatorInventory) -> Self {
        Self {
            electric: BearingSlots::new(inventory.electric_joints, inventory.electric_capacity()),
            gas: BearingSlots::new(inventory.gas_joints, inventory.gas_capacity()),
            servo: BearingSlots::new(inventory.servo_joints, inventory.servo_capacity()),
        }
    }
}

/// Used and total bearing ports for one actuator family.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BearingSlots {
    pub(crate) used: u32,
    pub(crate) capacity: u32,
}

impl BearingSlots {
    pub(crate) const fn new(used: u32, capacity: u32) -> Self {
        Self { used, capacity }
    }

    pub(crate) fn text(self) -> String {
        format!("{}/{} ports", self.used, self.capacity)
    }
}

impl PanelModel {
    /// Whether there is anything to draw.
    pub(crate) const fn is_open(&self) -> bool {
        self.open
    }

    /// What the header's subtitle reads.
    pub(crate) fn subtitle(&self) -> String {
        match self.lanes.len() {
            1 => "1 joint wired".to_owned(),
            count => format!("{count} joints wired"),
        }
    }
}

impl PanelModel {
    /// The lane for one wire, or `None` once it has gone.
    pub(crate) fn lane(&self, id: DriveLinkId) -> Option<&LaneModel> {
        self.lanes.iter().find(|lane| lane.id == id)
    }

    /// The keys the view's lane list is built from. Structure only: a lane's
    /// contents are read through bindings, so a value changing never rebuilds
    /// one.
    pub(crate) fn keys(&self) -> Vec<(DriveLinkId, ())> {
        self.lanes.iter().map(|lane| (lane.id, ())).collect()
    }

    pub(crate) fn engine_keys(&self) -> Vec<(EngineKind, ())> {
        self.engine_lanes
            .iter()
            .map(|lane| (lane.kind, ()))
            .collect()
    }

    pub(crate) fn engine_lane(&self, kind: EngineKind) -> Option<&EngineLaneModel> {
        self.engine_lanes.iter().find(|lane| lane.kind == kind)
    }

    pub(crate) fn engine_gear_keys(&self, kind: EngineKind) -> Vec<(usize, ())> {
        self.engine_lane(kind)
            .and_then(|lane| lane.config.as_ref())
            .map_or_else(Vec::new, |config| {
                (0..config.ratios().len())
                    .map(|index| (index, ()))
                    .collect()
            })
    }
}

#[cfg(test)]
mod tests;
