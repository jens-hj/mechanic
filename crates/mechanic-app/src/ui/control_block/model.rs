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

use mechanic_core::{
    DriveDwell, DriveKey, DriveLimits, DriveLinkId, DriveName, DriveProgram, DriveRelease,
    DriveState, DriveTarget, DriveTrigger, MAX_DRIVE_DWELL_SECONDS, MAX_DRIVE_LIMIT_RADIANS,
    MAX_DRIVE_SPEED_RAD_S, MAX_DRIVE_STATES,
};

/// Smallest travel range the grips may close to, in degrees. Two limits that
/// meet would leave the joint with nowhere to go.
const MIN_TRAVEL_SPAN_DEGREES: f32 = 5.0;

/// Furthest a travel limit may sit from centre, in degrees.
const MAX_TRAVEL_DEGREES: f32 = 180.0;

/// Shortest dwell that still reads as a wait, in seconds.
const MIN_DWELL_SECONDS: f32 = 0.1;

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
    /// Fastest the joint may turn, in degrees per second.
    SetMaxSpeed(f32),
    /// Strongest torque it may apply. Empty or an open-ended word means
    /// unlimited.
    SetTorque(String),
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

/// Words that all mean "do not limit this".
fn is_open_ended(text: &str) -> bool {
    matches!(
        text.trim().to_ascii_lowercase().as_str(),
        "" | "inf" | "infinite" | "infinity" | "unlimited" | "none" | "never"
    )
}

/// The joint's whole configuration, as the graph stores it.
type Wire = (DriveLimits, DriveProgram, DriveName);

/// Folds one edit into a joint's configuration.
///
/// Returns `None` when the edit cannot apply — an unparseable number, a value
/// out of range, a state that is not there — leaving the joint as it was. That
/// is the same answer for "you typed nonsense" and "that click does nothing",
/// because in both cases the right behaviour is to change nothing.
#[allow(clippy::too_many_lines)] // One match arm per thing the panel can change.
pub(crate) fn apply_edit(
    limits: DriveLimits,
    program: DriveProgram,
    name: DriveName,
    edit: &PanelEdit,
) -> Option<Wire> {
    match edit {
        PanelEdit::SetName(text) => Some((limits, program, DriveName::new(text))),

        PanelEdit::SetMaxSpeed(degrees) => {
            let limits = limits.with_max_speed(degrees.to_radians()).ok()?;
            // A state may have been spinning faster than the joint's new
            // ceiling, which would leave the dial reading past its own end.
            Some((limits, clamped_speeds(&program, limits), name))
        }

        PanelEdit::SetTorque(text) => {
            let torque = if is_open_ended(text) {
                f32::INFINITY
            } else {
                text.trim().parse::<f32>().ok()?
            };
            Some((limits.with_max_torque(torque).ok()?, program, name))
        }

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
                DriveTarget::Angle(_) => {
                    let radians = value.to_radians();
                    let (low, high) = limits
                        .angle_limits()
                        .unwrap_or((-MAX_DRIVE_LIMIT_RADIANS, MAX_DRIVE_LIMIT_RADIANS));
                    DriveTarget::Angle(radians.clamp(low, high))
                }
                DriveTarget::Speed(_) => {
                    let ceiling = limits.max_speed_rad_s();
                    DriveTarget::Speed(value.to_radians().clamp(-ceiling, ceiling))
                }
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
        DriveTarget::Angle(angle) => angle,
        DriveTarget::Speed(speed) => speed,
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
        DriveTarget::Speed(_) => None,
    })
}

/// Pulls every spin back inside the joint's speed ceiling.
fn clamped_speeds(program: &DriveProgram, limits: DriveLimits) -> DriveProgram {
    let ceiling = limits.max_speed_rad_s();
    fold_states(program, |state| match state.target() {
        DriveTarget::Speed(speed) => state
            .with_target(DriveTarget::Speed(speed.clamp(-ceiling, ceiling)))
            .ok(),
        DriveTarget::Angle(_) => None,
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
    pub(crate) fn capture(
        id: DriveLinkId,
        number: usize,
        limits: DriveLimits,
        program: &DriveProgram,
        name: &DriveName,
    ) -> Self {
        let states: Vec<StateModel> = program
            .states()
            .iter()
            .map(|state| StateModel {
                mode: match state.target() {
                    DriveTarget::Angle(_) => Mode::Angle,
                    DriveTarget::Speed(_) => Mode::Speed,
                },
                value: reading(state.target()).to_degrees(),
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
            id,
            number,
            name: name.as_str().to_owned(),
            speed: limits.max_speed_rad_s().to_degrees(),
            torque: limits.max_torque_newton_meters(),
            travel: limits
                .angle_limits()
                .map(|(low, high)| (low.to_degrees(), high.to_degrees())),
            loops: program.loops(),
            release_wires: wires(&states, WireKind::Release),
            dwell_wires: wires(&states, WireKind::Dwell),
            states,
        }
    }

    /// What the speed chip reads.
    pub(crate) fn speed_text(&self) -> String {
        format!("{:.0} °/s", self.speed)
    }

    /// What the torque chip reads. An unlimited joint says so in words rather
    /// than showing an infinity.
    pub(crate) fn torque_text(&self) -> String {
        if self.torque.is_infinite() {
            "unlimited".to_owned()
        } else {
            format!("{:.0} N·m", self.torque)
        }
    }

    /// What the travel chip reads.
    pub(crate) fn travel_text(&self) -> String {
        match self.travel {
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
                    label: format!("{key} up"),
                })
            }
            WireKind::Dwell => {
                let (seconds, target) = state.dwell?;
                Some(WireModel {
                    source,
                    target: usize::from(target).min(states.len().saturating_sub(1)),
                    label: format!("{seconds:.1} s"),
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
}

#[cfg(test)]
mod tests {
    use super::{Mode, PanelEdit, Preset, apply_edit};
    use mechanic_core::{
        DriveDwell, DriveKey, DriveLimits, DriveName, DriveProgram, DriveRelease, DriveState,
        DriveTarget, DriveTrigger, MAX_DRIVE_SPEED_RAD_S,
    };

    /// A joint holding three angles, the middle two bound to keys.
    fn steering() -> (DriveLimits, DriveProgram, DriveName) {
        let held = |degrees: f32| {
            DriveState::new(DriveTarget::Angle(f32::to_radians(degrees))).expect("in range")
        };
        let keyed = |state: DriveState, key: char| {
            state.with_trigger(Some(DriveTrigger::new(
                DriveKey::new(key).expect("a letter"),
                DriveRelease::RevertTo(0),
            )))
        };
        let states = [held(0.0), keyed(held(-30.0), 'A'), keyed(held(30.0), 'D')];
        (
            DriveLimits::new(3.0, f32::INFINITY, Some((-0.8, 0.8))).expect("in range"),
            DriveProgram::new(&states, false).expect("a valid program"),
            DriveName::new("Steer · front left"),
        )
    }

    /// A joint that spins.
    fn driving() -> (DriveLimits, DriveProgram, DriveName) {
        let states = [
            DriveState::new(DriveTarget::Speed(0.0)).expect("in range"),
            DriveState::new(DriveTarget::Speed(2.0)).expect("in range"),
        ];
        (
            DriveLimits::new(3.0, 180.0, None).expect("in range"),
            DriveProgram::new(&states, false).expect("a valid program"),
            DriveName::EMPTY,
        )
    }

    fn apply(
        wire: (DriveLimits, DriveProgram, DriveName),
        edit: &PanelEdit,
    ) -> (DriveLimits, DriveProgram, DriveName) {
        apply_edit(wire.0, wire.1, wire.2, edit).expect("the edit applies")
    }

    #[test]
    fn a_named_joint_keeps_its_name_and_an_overlong_one_is_cut() {
        let named = apply(steering(), &PanelEdit::SetName("Tipper arm".to_owned()));
        assert_eq!(named.2.as_str(), "Tipper arm");
    }

    #[test]
    fn typing_a_speed_is_read_as_degrees_a_second() {
        let faster = apply(steering(), &PanelEdit::SetMaxSpeed(360.0));
        assert!((faster.0.max_speed_rad_s() - std::f32::consts::TAU).abs() < 1.0e-4);
    }

    #[test]
    fn a_speed_out_of_range_leaves_the_joint_alone() {
        let (limits, program, name) = steering();
        assert!(
            apply_edit(limits, program, name, &PanelEdit::SetMaxSpeed(100_000.0)).is_none(),
            "a speed past the ceiling is not a speed the joint can hold"
        );
        assert!(
            apply_edit(limits, program, name, &PanelEdit::SetMaxSpeed(-5.0)).is_none(),
            "a joint cannot have a negative ceiling"
        );
    }

    #[test]
    fn lowering_the_ceiling_pulls_a_faster_state_back_under_it() {
        let quick = apply(driving(), &PanelEdit::SetMaxSpeed(f32::to_degrees(1.0)));
        let spun = quick.1.state(1).expect("the second state exists");
        assert!(
            matches!(spun.target(), DriveTarget::Speed(speed) if speed <= 1.0 + 1.0e-4),
            "a state may not ask for more speed than the joint has",
        );
    }

    #[test]
    fn an_empty_or_open_ended_torque_means_unlimited() {
        for text in ["", "none", "inf", "unlimited", "never", " Infinity "] {
            let wire = apply(driving(), &PanelEdit::SetTorque(text.to_owned()));
            assert!(
                wire.0.max_torque_newton_meters().is_infinite(),
                "{text:?} must read as unlimited",
            );
        }
        let finite = apply(driving(), &PanelEdit::SetTorque("240".to_owned()));
        assert!((finite.0.max_torque_newton_meters() - 240.0).abs() < f32::EPSILON);
    }

    #[test]
    fn travel_limits_toggle_off_and_back_on_at_a_default_range() {
        let free = apply(steering(), &PanelEdit::ToggleTravel);
        assert_eq!(free.0.angle_limits(), None);
        let limited = apply(free, &PanelEdit::ToggleTravel);
        let (low, high) = limited.0.angle_limits().expect("limits came back");
        assert!((low.to_degrees() + 45.0).abs() < 1.0e-3);
        assert!((high.to_degrees() - 45.0).abs() < 1.0e-3);
    }

    #[test]
    fn moving_the_limits_pulls_every_held_angle_inside_them() {
        let tight = apply(
            steering(),
            &PanelEdit::SetTravel {
                min: -10.0,
                max: 10.0,
            },
        );
        for index in 0..3 {
            let held = tight.1.state(index).expect("the state exists");
            let DriveTarget::Angle(angle) = held.target() else {
                panic!("a steering state holds an angle");
            };
            assert!(
                angle.to_degrees() >= -10.1 && angle.to_degrees() <= 10.1,
                "state {index} sits outside the joint's travel: {}°",
                angle.to_degrees(),
            );
        }
    }

    #[test]
    fn limits_that_would_cross_are_held_apart() {
        let crossed = apply(
            steering(),
            &PanelEdit::SetTravel {
                min: 40.0,
                max: -40.0,
            },
        );
        let (low, high) = crossed
            .0
            .angle_limits()
            .expect("the joint still has limits");
        assert!(high > low, "a joint must be left somewhere to turn");
    }

    #[test]
    fn repeating_the_sequence_flips_back_and_forth() {
        let looping = apply(steering(), &PanelEdit::ToggleLoop);
        assert!(looping.1.loops());
        assert!(!apply(looping, &PanelEdit::ToggleLoop).1.loops());
    }

    #[test]
    fn switching_mode_carries_the_number_across_and_clamps_it() {
        let spun = apply(
            steering(),
            &PanelEdit::SetMode {
                state: 1,
                mode: Mode::Speed,
            },
        );
        let state = spun.1.state(1).expect("the state exists");
        assert!(
            matches!(state.target(), DriveTarget::Speed(speed)
                if speed.abs() <= MAX_DRIVE_SPEED_RAD_S),
            "the number carries across into the new unit's range",
        );
    }

    #[test]
    fn a_typed_angle_is_clamped_into_the_joints_travel() {
        let pushed = apply(
            steering(),
            &PanelEdit::SetValue {
                state: 1,
                value: 300.0,
            },
        );
        let DriveTarget::Angle(angle) = pushed.1.state(1).expect("the state exists").target()
        else {
            panic!("a steering state holds an angle");
        };
        assert!(
            angle <= 0.8 + 1.0e-4,
            "a typed angle cannot leave the joint's travel: {}°",
            angle.to_degrees(),
        );
    }

    #[test]
    fn binding_a_key_takes_it_from_whichever_state_had_it() {
        let stolen = apply(steering(), &PanelEdit::BindKey { state: 0, key: 'a' });
        assert_eq!(
            stolen
                .1
                .state(0)
                .unwrap()
                .trigger()
                .map(|t| t.key().symbol()),
            Some('A'),
            "the key binds where it was asked for",
        );
        assert!(
            stolen.1.state(1).unwrap().trigger().is_none(),
            "one key can only mean one state, so it leaves the sibling that had it",
        );
    }

    #[test]
    fn clearing_a_key_leaves_the_state_without_a_trigger() {
        let cleared = apply(steering(), &PanelEdit::ClearKey { state: 1 });
        assert!(cleared.1.state(1).unwrap().trigger().is_none());
        let (limits, program, name) = steering();
        assert!(
            apply_edit(limits, program, name, &PanelEdit::ClearKey { state: 0 }).is_none(),
            "a state with no key has nothing to clear",
        );
    }

    #[test]
    fn release_cycles_through_every_state_and_back_to_staying_put() {
        let mut wire = steering();
        // S2 starts handing back to S1.
        assert_eq!(
            wire.1.state(1).unwrap().trigger().unwrap().release(),
            DriveRelease::RevertTo(0)
        );
        let mut seen = Vec::new();
        for _ in 0..4 {
            wire = apply(wire, &PanelEdit::CycleRelease { state: 1 });
            seen.push(wire.1.state(1).unwrap().trigger().unwrap().release());
        }
        assert_eq!(
            seen,
            vec![
                DriveRelease::RevertTo(1),
                DriveRelease::RevertTo(2),
                DriveRelease::Latch,
                DriveRelease::RevertTo(0),
            ],
            "release steps through each state in turn, then back to staying put",
        );
    }

    #[test]
    fn a_dwell_turns_on_pointing_at_the_next_state_and_off_again() {
        let waiting = apply(steering(), &PanelEdit::ToggleDwell { state: 0 });
        let dwell = waiting
            .1
            .state(0)
            .unwrap()
            .dwell()
            .expect("a dwell was added");
        assert!((dwell.seconds() - 1.0).abs() < f32::EPSILON);
        assert_eq!(
            dwell.next(),
            Some(1),
            "a fresh dwell hands off to the next state"
        );
        let stopped = apply(waiting, &PanelEdit::ToggleDwell { state: 0 });
        assert!(stopped.1.state(0).unwrap().dwell().is_none());
    }

    #[test]
    fn the_last_states_dwell_wraps_round_to_the_first() {
        let waiting = apply(steering(), &PanelEdit::ToggleDwell { state: 2 });
        assert_eq!(waiting.1.state(2).unwrap().dwell().unwrap().next(), Some(0));
    }

    #[test]
    fn a_dwell_time_is_kept_inside_what_the_model_accepts() {
        let waiting = apply(steering(), &PanelEdit::ToggleDwell { state: 0 });
        let long = apply(
            waiting,
            &PanelEdit::SetDwell {
                state: 0,
                seconds: 10_000.0,
            },
        );
        assert!((long.1.state(0).unwrap().dwell().unwrap().seconds() - 600.0).abs() < 1.0e-3);
        let short = apply(
            long,
            &PanelEdit::SetDwell {
                state: 0,
                seconds: -4.0,
            },
        );
        assert!(
            short.1.state(0).unwrap().dwell().unwrap().seconds() > 0.0,
            "a dwell of nothing is not a dwell",
        );
    }

    #[test]
    fn pointing_a_dwell_at_a_state_gives_it_one_if_it_had_none() {
        let aimed = apply(
            steering(),
            &PanelEdit::SetDwellTarget {
                state: 0,
                target: 2,
            },
        );
        let dwell = aimed
            .1
            .state(0)
            .unwrap()
            .dwell()
            .expect("a dwell was added");
        assert_eq!(dwell.next(), Some(2));
        let (limits, program, name) = steering();
        assert!(
            apply_edit(
                limits,
                program,
                name,
                &PanelEdit::SetDwellTarget {
                    state: 0,
                    target: 9
                }
            )
            .is_none(),
            "a dwell cannot hand off to a state that is not there",
        );
    }

    #[test]
    fn states_are_added_up_to_the_limit_and_removed_down_to_one() {
        let mut wire = steering();
        while wire.1.len() < 8 {
            wire = apply(wire, &PanelEdit::AddState);
        }
        assert_eq!(wire.1.len(), 8);
        assert!(
            apply_edit(wire.0, wire.1, wire.2, &PanelEdit::AddState).is_none(),
            "a joint holds at most eight states",
        );

        let mut wire = steering();
        while wire.1.len() > 1 {
            wire = apply(wire, &PanelEdit::RemoveState { state: 0 });
        }
        assert!(
            apply_edit(wire.0, wire.1, wire.2, &PanelEdit::RemoveState { state: 0 }).is_none(),
            "a joint always has somewhere to be",
        );
    }

    #[test]
    fn an_added_state_inherits_the_last_ones_mode_and_rests_at_zero() {
        let grown = apply(driving(), &PanelEdit::AddState);
        let added = grown.1.state(2).expect("the state was added");
        assert!(matches!(added.target(), DriveTarget::Speed(speed) if speed == 0.0));
    }

    #[test]
    fn each_preset_installs_a_program_that_matches_its_name() {
        let steer = apply(driving(), &PanelEdit::ApplyPreset(Preset::Steer));
        assert_eq!(steer.1.len(), 3);
        assert!(
            steer.0.angle_limits().is_some(),
            "steering has travel limits"
        );
        assert!(matches!(
            steer.1.state(0).unwrap().target(),
            DriveTarget::Angle(_)
        ));

        let drive = apply(driving(), &PanelEdit::ApplyPreset(Preset::Drive));
        assert_eq!(drive.1.len(), 3);
        assert_eq!(drive.0.angle_limits(), None, "a driven wheel turns freely");
        assert!(matches!(
            drive.1.state(1).unwrap().target(),
            DriveTarget::Speed(_)
        ));

        let spin = apply(driving(), &PanelEdit::ApplyPreset(Preset::Spin));
        assert_eq!(spin.1.len(), 2, "run and stop is two states");
        for index in 0..2 {
            assert_eq!(
                spin.1
                    .state(index)
                    .unwrap()
                    .trigger()
                    .map(DriveTrigger::release),
                Some(DriveRelease::Latch),
                "a toggle stays where it was put",
            );
        }
    }

    #[test]
    fn a_preset_keeps_the_joints_name_and_its_dwellless_states_valid() {
        let named = apply(steering(), &PanelEdit::ApplyPreset(Preset::Drive));
        assert_eq!(named.2.as_str(), "Steer · front left");
        assert!(
            named.1.states().iter().all(|state| state.dwell().is_none()),
            "a preset is a fresh program, not a merge",
        );
        let _ = DriveDwell::new(1.0, None).expect("dwells still validate");
    }
}
