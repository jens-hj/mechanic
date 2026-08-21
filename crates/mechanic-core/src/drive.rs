use core::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Largest supported drive speed, in radians per second.
pub const MAX_DRIVE_SPEED_RAD_S: f32 = 25.0;

/// Largest supported drive angle magnitude, in radians.
pub const MAX_DRIVE_LIMIT_RADIANS: f32 = core::f32::consts::TAU;

/// Largest number of states one driven bearing can hold.
pub const MAX_DRIVE_STATES: usize = 8;

/// Longest supported dwell time, in seconds.
pub const MAX_DRIVE_DWELL_SECONDS: f32 = 600.0;

/// Invalid per-bearing drive limits.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum DriveLimitsError {
    /// The maximum speed was not positive, or exceeded the supported range.
    #[error("drive maximum speed must be positive and at most 25 rad/s")]
    SpeedOutOfRange,
    /// The maximum torque was zero, negative, or NaN.
    #[error("drive maximum torque must be positive, or infinite for an unlimited drive")]
    NonPositiveTorque,
    /// An angle limit was not finite or exceeded the supported range.
    #[error("drive angle limits must be finite and within one full turn")]
    LimitOutOfRange,
    /// The minimum angle limit was not below the maximum.
    #[error("drive minimum angle limit must be below its maximum")]
    InvertedLimits,
}

/// Invalid drive state program.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum DriveProgramError {
    /// A program needs at least one state.
    #[error("a drive program needs at least one state")]
    EmptyProgram,
    /// A program cannot hold more than [`MAX_DRIVE_STATES`] states.
    #[error("a drive program holds at most {MAX_DRIVE_STATES} states")]
    TooManyStates,
    /// A dwell handoff or release target named a state that does not exist.
    #[error("drive state index {0} is outside the program")]
    StateIndexOutOfRange(u8),
    /// A target angle or speed was not finite.
    #[error("drive state targets must be finite")]
    NonFiniteTarget,
    /// A target speed magnitude exceeded the supported range.
    #[error("drive state speed magnitude must be at most 25 rad/s")]
    SpeedOutOfRange,
    /// A target angle magnitude exceeded the supported range.
    #[error("drive state angle magnitude must be within one full turn")]
    AngleOutOfRange,
    /// A dwell time was not positive, or exceeded the supported range.
    #[error("drive dwell time must be positive and at most {MAX_DRIVE_DWELL_SECONDS} seconds")]
    DwellOutOfRange,
    /// Two states in one program were bound to the same key.
    #[error("key {0} is bound to two states of the same bearing")]
    DuplicateKey(DriveKey),
}

/// What one drive state asks of its bearing.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum DriveTarget {
    /// Hold a joint angle, in radians, measured from the built pose.
    Angle(f32),
    /// Spin at a signed speed, in radians per second.
    Speed(f32),
}

impl DriveTarget {
    /// Returns the target when it is finite and within range.
    ///
    /// # Errors
    ///
    /// Returns [`DriveProgramError`] when the value is non-finite, or when an
    /// angle exceeds one full turn or a speed exceeds 25 rad/s.
    pub fn validated(self) -> Result<Self, DriveProgramError> {
        let value = match self {
            Self::Angle(angle) => angle,
            Self::Speed(speed) => speed,
        };
        if !value.is_finite() {
            return Err(DriveProgramError::NonFiniteTarget);
        }
        match self {
            Self::Angle(angle) if angle.abs() > MAX_DRIVE_LIMIT_RADIANS => {
                Err(DriveProgramError::AngleOutOfRange)
            }
            Self::Speed(speed) if speed.abs() > MAX_DRIVE_SPEED_RAD_S => {
                Err(DriveProgramError::SpeedOutOfRange)
            }
            target => Ok(target),
        }
    }

    /// Returns the target with its direction flipped.
    #[must_use]
    pub const fn reversed(self) -> Self {
        match self {
            Self::Angle(angle) => Self::Angle(-angle),
            Self::Speed(speed) => Self::Speed(-speed),
        }
    }

    /// Target angle in radians, when this state holds a position.
    pub const fn angle(self) -> Option<f32> {
        match self {
            Self::Angle(angle) => Some(angle),
            Self::Speed(_) => None,
        }
    }

    /// Target speed in radians per second, when this state spins the bearing.
    pub const fn speed(self) -> Option<f32> {
        match self {
            Self::Speed(speed) => Some(speed),
            Self::Angle(_) => None,
        }
    }
}

impl Default for DriveTarget {
    fn default() -> Self {
        Self::Speed(0.0)
    }
}

/// Keyboard key a drive state can be bound to.
///
/// Only uppercase ASCII letters and digits are accepted, so the same binding
/// reads the same way in the panel, in a saved creation, and on the keyboard.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DriveKey(u8);

impl DriveKey {
    /// Creates a binding from `A`-`Z`, `a`-`z`, or `0`-`9`.
    ///
    /// Lowercase letters are folded to uppercase. Any other character returns
    /// `None`.
    pub const fn new(symbol: char) -> Option<Self> {
        let upper = symbol.to_ascii_uppercase();
        if upper.is_ascii_uppercase() || upper.is_ascii_digit() {
            Some(Self(upper as u8))
        } else {
            None
        }
    }

    /// The bound character, always uppercase.
    pub const fn symbol(self) -> char {
        self.0 as char
    }
}

impl fmt::Display for DriveKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.symbol())
    }
}

/// What a triggered state does when its key is released.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriveRelease {
    /// Stay in this state until something else changes it.
    #[default]
    Latch,
    /// Return to the named state as soon as the key is let go.
    RevertTo(u8),
}

impl DriveRelease {
    /// State this release rule returns to, when it reverts at all.
    pub const fn target(self) -> Option<u8> {
        match self {
            Self::Latch => None,
            Self::RevertTo(state) => Some(state),
        }
    }
}

/// Key binding that activates one drive state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DriveTrigger {
    key: DriveKey,
    release: DriveRelease,
}

impl DriveTrigger {
    /// Binds a key to a state with the given release behaviour.
    pub const fn new(key: DriveKey, release: DriveRelease) -> Self {
        Self { key, release }
    }

    /// The bound key.
    pub const fn key(self) -> DriveKey {
        self.key
    }

    /// What happens when the key is released.
    pub const fn release(self) -> DriveRelease {
        self.release
    }
}

/// Automatic handoff from one drive state to another after a delay.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriveDwell {
    seconds: f32,
    next: Option<u8>,
}

impl DriveDwell {
    /// Creates a validated dwell.
    ///
    /// `next` names the state to enter; `None` means the following state,
    /// wrapping to the first when the program loops.
    ///
    /// # Errors
    ///
    /// Returns [`DriveProgramError::DwellOutOfRange`] when the time is not
    /// positive, not finite, or longer than [`MAX_DRIVE_DWELL_SECONDS`].
    pub fn new(seconds: f32, next: Option<u8>) -> Result<Self, DriveProgramError> {
        if !seconds.is_finite() || seconds <= 0.0 || seconds > MAX_DRIVE_DWELL_SECONDS {
            return Err(DriveProgramError::DwellOutOfRange);
        }
        Ok(Self { seconds, next })
    }

    /// How long the state stays active, in seconds.
    pub const fn seconds(self) -> f32 {
        self.seconds
    }

    /// Explicit handoff target, or `None` for the following state.
    pub const fn next(self) -> Option<u8> {
        self.next
    }
}

/// One state of a driven bearing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DriveState {
    target: DriveTarget,
    dwell: Option<DriveDwell>,
    trigger: Option<DriveTrigger>,
}

impl DriveState {
    /// Creates a state that holds the given target indefinitely.
    ///
    /// # Errors
    ///
    /// Returns [`DriveProgramError`] when the target is out of range.
    pub fn new(target: DriveTarget) -> Result<Self, DriveProgramError> {
        Ok(Self {
            target: target.validated()?,
            dwell: None,
            trigger: None,
        })
    }

    /// Replaces the target.
    ///
    /// # Errors
    ///
    /// Returns [`DriveProgramError`] when the target is out of range.
    pub fn with_target(mut self, target: DriveTarget) -> Result<Self, DriveProgramError> {
        self.target = target.validated()?;
        Ok(self)
    }

    /// Sets or clears the automatic handoff.
    #[must_use]
    pub const fn with_dwell(mut self, dwell: Option<DriveDwell>) -> Self {
        self.dwell = dwell;
        self
    }

    /// Sets or clears the key binding.
    #[must_use]
    pub const fn with_trigger(mut self, trigger: Option<DriveTrigger>) -> Self {
        self.trigger = trigger;
        self
    }

    /// What this state asks of the bearing.
    pub const fn target(self) -> DriveTarget {
        self.target
    }

    /// Automatic handoff, when this state advances on its own.
    pub const fn dwell(self) -> Option<DriveDwell> {
        self.dwell
    }

    /// Key binding, when this state can be triggered by hand.
    pub const fn trigger(self) -> Option<DriveTrigger> {
        self.trigger
    }
}

/// Ordered states of one driven bearing.
///
/// State zero is the state a bearing enters when the simulation starts, and is
/// the conventional target for a release or a reset key.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriveProgram {
    states: [DriveState; MAX_DRIVE_STATES],
    len: u8,
    loops: bool,
}

impl DriveProgram {
    /// Creates a validated program.
    ///
    /// # Errors
    ///
    /// Returns [`DriveProgramError`] when there are no states or more than
    /// [`MAX_DRIVE_STATES`], when a target or dwell is out of range, when a
    /// dwell or release names a state that does not exist, or when two states
    /// share a key.
    ///
    /// # Panics
    ///
    /// Never in practice: the state count is bounded by [`MAX_DRIVE_STATES`]
    /// before it is narrowed.
    pub fn new(states: &[DriveState], loops: bool) -> Result<Self, DriveProgramError> {
        if states.is_empty() {
            return Err(DriveProgramError::EmptyProgram);
        }
        if states.len() > MAX_DRIVE_STATES {
            return Err(DriveProgramError::TooManyStates);
        }
        let len = u8::try_from(states.len()).expect("state count is at most MAX_DRIVE_STATES");
        for (index, state) in states.iter().enumerate() {
            state.target().validated()?;
            if let Some(dwell) = state.dwell() {
                DriveDwell::new(dwell.seconds(), dwell.next())?;
                if let Some(next) = dwell.next()
                    && next >= len
                {
                    return Err(DriveProgramError::StateIndexOutOfRange(next));
                }
            }
            let Some(trigger) = state.trigger() else {
                continue;
            };
            if let Some(target) = trigger.release().target()
                && target >= len
            {
                return Err(DriveProgramError::StateIndexOutOfRange(target));
            }
            // A key that selects two states of the same bearing is ambiguous.
            // The same key on two different bearings is not: that is how one
            // key steers a whole axle.
            if states[..index]
                .iter()
                .filter_map(|earlier| earlier.trigger())
                .any(|earlier| earlier.key() == trigger.key())
            {
                return Err(DriveProgramError::DuplicateKey(trigger.key()));
            }
        }
        let mut rows = [DriveState::default(); MAX_DRIVE_STATES];
        rows[..states.len()].copy_from_slice(states);
        Ok(Self {
            states: rows,
            len,
            loops,
        })
    }

    /// The states, in order.
    pub fn states(&self) -> &[DriveState] {
        &self.states[..self.len as usize]
    }

    /// Number of states.
    pub const fn len(&self) -> usize {
        self.len as usize
    }

    /// Always false: a program holds at least one state.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the last state hands back to the first.
    pub const fn loops(&self) -> bool {
        self.loops
    }

    /// The state at `index`, when it exists.
    pub fn state(&self, index: u8) -> Option<DriveState> {
        (index < self.len).then(|| self.states[index as usize])
    }

    /// Sets whether the program wraps.
    #[must_use]
    pub const fn with_loops(mut self, loops: bool) -> Self {
        self.loops = loops;
        self
    }

    /// Replaces one state.
    ///
    /// # Errors
    ///
    /// Returns [`DriveProgramError`] when the index is outside the program or
    /// the replacement makes the program invalid.
    pub fn with_state(&self, index: u8, state: DriveState) -> Result<Self, DriveProgramError> {
        if index >= self.len {
            return Err(DriveProgramError::StateIndexOutOfRange(index));
        }
        let mut states = self.states().to_vec();
        states[index as usize] = state;
        Self::new(&states, self.loops)
    }

    /// Appends a state.
    ///
    /// # Errors
    ///
    /// Returns [`DriveProgramError`] when the program is already full or the
    /// new state makes it invalid.
    pub fn with_pushed_state(&self, state: DriveState) -> Result<Self, DriveProgramError> {
        if self.len as usize >= MAX_DRIVE_STATES {
            return Err(DriveProgramError::TooManyStates);
        }
        let mut states = self.states().to_vec();
        states.push(state);
        Self::new(&states, self.loops)
    }

    /// Removes a state and renumbers every reference to the states after it.
    ///
    /// A dwell that handed off to the removed state falls back to the following
    /// state, and a release that returned to it returns to state zero instead.
    ///
    /// # Errors
    ///
    /// Returns [`DriveProgramError`] when the index is outside the program, or
    /// [`DriveProgramError::EmptyProgram`] when it is the only state left.
    pub fn with_removed_state(&self, index: u8) -> Result<Self, DriveProgramError> {
        if index >= self.len {
            return Err(DriveProgramError::StateIndexOutOfRange(index));
        }
        if self.len <= 1 {
            return Err(DriveProgramError::EmptyProgram);
        }
        let renumber = |reference: u8| -> Option<u8> {
            match reference.cmp(&index) {
                core::cmp::Ordering::Equal => None,
                core::cmp::Ordering::Greater => Some(reference - 1),
                core::cmp::Ordering::Less => Some(reference),
            }
        };
        let mut states = self.states().to_vec();
        states.remove(index as usize);
        for state in &mut states {
            if let Some(dwell) = state.dwell() {
                let next = dwell.next().and_then(renumber);
                *state = state.with_dwell(Some(DriveDwell::new(dwell.seconds(), next)?));
            }
            if let Some(trigger) = state.trigger()
                && let Some(target) = trigger.release().target()
            {
                let release = renumber(target).map_or(DriveRelease::RevertTo(0), |kept| {
                    DriveRelease::RevertTo(kept)
                });
                *state = state.with_trigger(Some(DriveTrigger::new(trigger.key(), release)));
            }
        }
        Self::new(&states, self.loops)
    }

    /// State a dwell hands off to, or `None` when the program should hold.
    ///
    /// Returns `None` for a state without a dwell, and for the last state of a
    /// program that does not loop.
    pub fn advanced_state(&self, from: u8) -> Option<u8> {
        let dwell = self.state(from)?.dwell()?;
        if let Some(next) = dwell.next() {
            return (next < self.len).then_some(next);
        }
        let following = from.saturating_add(1);
        if following < self.len {
            Some(following)
        } else if self.loops {
            Some(0)
        } else {
            None
        }
    }

    /// Lowest-indexed state bound to `key`.
    ///
    /// Programs reject duplicate keys, so this is unambiguous; taking the
    /// lowest index keeps the result stable regardless.
    pub fn state_for_key(&self, key: DriveKey) -> Option<u8> {
        self.states()
            .iter()
            .position(|state| state.trigger().is_some_and(|trigger| trigger.key() == key))?
            .try_into()
            .ok()
    }
}

impl Default for DriveProgram {
    fn default() -> Self {
        Self {
            states: [DriveState::default(); MAX_DRIVE_STATES],
            len: 1,
            loops: false,
        }
    }
}

/// Speed, torque, and travel envelope of one driven bearing.
///
/// No state may exceed these; they are the physical capability of the joint,
/// while the program says what to do with it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriveLimits {
    max_speed_rad_s: f32,
    max_torque_newton_meters: f32,
    angle_limits: Option<(f32, f32)>,
}

impl DriveLimits {
    /// Default maximum speed, in radians per second.
    pub const DEFAULT_MAX_SPEED_RAD_S: f32 = 3.0;

    /// Creates validated limits.
    ///
    /// # Errors
    ///
    /// Returns [`DriveLimitsError`] when the speed is not positive or above
    /// 25 rad/s, the torque is not positive, or the angle limits are
    /// non-finite, beyond one full turn, or not ordered.
    pub fn new(
        max_speed_rad_s: f32,
        max_torque_newton_meters: f32,
        angle_limits: Option<(f32, f32)>,
    ) -> Result<Self, DriveLimitsError> {
        if !max_speed_rad_s.is_finite()
            || max_speed_rad_s <= 0.0
            || max_speed_rad_s > MAX_DRIVE_SPEED_RAD_S
        {
            return Err(DriveLimitsError::SpeedOutOfRange);
        }
        if max_torque_newton_meters.is_nan() || max_torque_newton_meters <= 0.0 {
            return Err(DriveLimitsError::NonPositiveTorque);
        }
        if let Some((minimum, maximum)) = angle_limits {
            if !minimum.is_finite()
                || !maximum.is_finite()
                || minimum.abs() > MAX_DRIVE_LIMIT_RADIANS
                || maximum.abs() > MAX_DRIVE_LIMIT_RADIANS
            {
                return Err(DriveLimitsError::LimitOutOfRange);
            }
            if minimum >= maximum {
                return Err(DriveLimitsError::InvertedLimits);
            }
        }
        Ok(Self {
            max_speed_rad_s,
            max_torque_newton_meters,
            angle_limits,
        })
    }

    /// Fastest the bearing may turn, in radians per second.
    pub const fn max_speed_rad_s(self) -> f32 {
        self.max_speed_rad_s
    }

    /// Maximum applied torque in newton metres. Infinite means unlimited.
    pub const fn max_torque_newton_meters(self) -> f32 {
        self.max_torque_newton_meters
    }

    /// Travel limits in radians, when the bearing stops and holds at its ends.
    pub const fn angle_limits(self) -> Option<(f32, f32)> {
        self.angle_limits
    }

    /// Lower travel limit, or negative infinity when the bearing is free.
    pub fn min_angle(self) -> f32 {
        self.angle_limits
            .map_or(f32::NEG_INFINITY, |(minimum, _)| minimum)
    }

    /// Upper travel limit, or positive infinity when the bearing is free.
    pub fn max_angle(self) -> f32 {
        self.angle_limits
            .map_or(f32::INFINITY, |(_, maximum)| maximum)
    }

    /// Applies a validated maximum speed.
    ///
    /// # Errors
    ///
    /// Returns [`DriveLimitsError`] when the speed is out of range.
    pub fn with_max_speed(self, max_speed_rad_s: f32) -> Result<Self, DriveLimitsError> {
        Self::new(
            max_speed_rad_s,
            self.max_torque_newton_meters,
            self.angle_limits,
        )
    }

    /// Applies a validated maximum torque.
    ///
    /// # Errors
    ///
    /// Returns [`DriveLimitsError`] when the torque is not positive.
    pub fn with_max_torque(self, max_torque_newton_meters: f32) -> Result<Self, DriveLimitsError> {
        Self::new(
            self.max_speed_rad_s,
            max_torque_newton_meters,
            self.angle_limits,
        )
    }

    /// Applies validated travel limits, or removes them.
    ///
    /// # Errors
    ///
    /// Returns [`DriveLimitsError`] when the limits are non-finite, beyond one
    /// full turn, or not ordered.
    pub fn with_angle_limits(
        self,
        angle_limits: Option<(f32, f32)>,
    ) -> Result<Self, DriveLimitsError> {
        Self::new(
            self.max_speed_rad_s,
            self.max_torque_newton_meters,
            angle_limits,
        )
    }
}

impl Default for DriveLimits {
    fn default() -> Self {
        Self {
            max_speed_rad_s: Self::DEFAULT_MAX_SPEED_RAD_S,
            max_torque_newton_meters: f32::INFINITY,
            angle_limits: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DriveDwell, DriveKey, DriveLimits, DriveLimitsError, DriveProgram, DriveProgramError,
        DriveRelease, DriveState, DriveTarget, DriveTrigger, MAX_DRIVE_STATES,
    };

    fn key(symbol: char) -> DriveKey {
        DriveKey::new(symbol).expect("test keys are letters or digits")
    }

    fn angle_state(degrees: f32) -> DriveState {
        DriveState::new(DriveTarget::Angle(degrees.to_radians())).expect("test angle is in range")
    }

    #[test]
    fn drive_keys_accept_letters_and_digits_and_fold_case() {
        assert_eq!(DriveKey::new('a').map(DriveKey::symbol), Some('A'));
        assert_eq!(DriveKey::new('W').map(DriveKey::symbol), Some('W'));
        assert_eq!(DriveKey::new('7').map(DriveKey::symbol), Some('7'));
        assert_eq!(DriveKey::new('-'), None);
        assert_eq!(DriveKey::new(' '), None);
    }

    #[test]
    fn programs_reject_out_of_range_targets_dwells_and_state_references() {
        assert_eq!(
            DriveProgram::new(&[], false),
            Err(DriveProgramError::EmptyProgram)
        );
        assert_eq!(
            DriveProgram::new(&[DriveState::default(); MAX_DRIVE_STATES + 1], false),
            Err(DriveProgramError::TooManyStates)
        );
        assert_eq!(
            DriveState::new(DriveTarget::Speed(f32::NAN)),
            Err(DriveProgramError::NonFiniteTarget)
        );
        assert_eq!(
            DriveState::new(DriveTarget::Speed(25.5)),
            Err(DriveProgramError::SpeedOutOfRange)
        );
        assert_eq!(
            DriveState::new(DriveTarget::Angle(10.0)),
            Err(DriveProgramError::AngleOutOfRange)
        );
        assert_eq!(
            DriveDwell::new(0.0, None),
            Err(DriveProgramError::DwellOutOfRange)
        );

        let dangling = angle_state(0.0).with_dwell(Some(
            DriveDwell::new(1.0, Some(3)).expect("dwell time is in range"),
        ));
        assert_eq!(
            DriveProgram::new(&[dangling], false),
            Err(DriveProgramError::StateIndexOutOfRange(3))
        );

        let reverting = angle_state(0.0)
            .with_trigger(Some(DriveTrigger::new(key('A'), DriveRelease::RevertTo(2))));
        assert_eq!(
            DriveProgram::new(&[reverting], false),
            Err(DriveProgramError::StateIndexOutOfRange(2))
        );
    }

    #[test]
    fn one_key_may_drive_many_bearings_but_not_two_states_of_one_bearing() {
        let first =
            angle_state(0.0).with_trigger(Some(DriveTrigger::new(key('A'), DriveRelease::Latch)));
        let second =
            angle_state(30.0).with_trigger(Some(DriveTrigger::new(key('A'), DriveRelease::Latch)));

        assert_eq!(
            DriveProgram::new(&[first, second], false),
            Err(DriveProgramError::DuplicateKey(key('A')))
        );

        // The same key in two separate programs is exactly how one key steers a
        // whole axle, so both must build.
        assert!(DriveProgram::new(&[first], false).is_ok());
        assert!(DriveProgram::new(&[second], false).is_ok());
    }

    #[test]
    fn dwell_advances_to_its_named_state_then_to_the_following_one() {
        let dwell_to_third = angle_state(90.0).with_dwell(Some(
            DriveDwell::new(2.0, Some(2)).expect("dwell time is in range"),
        ));
        let dwell_to_second = angle_state(-90.0).with_dwell(Some(
            DriveDwell::new(4.0, Some(1)).expect("dwell time is in range"),
        ));
        let program =
            DriveProgram::new(&[angle_state(0.0), dwell_to_third, dwell_to_second], false)
                .expect("procedure program is valid");

        assert_eq!(program.advanced_state(0), None, "state 0 has no dwell");
        assert_eq!(program.advanced_state(1), Some(2));
        assert_eq!(program.advanced_state(2), Some(1));
        assert_eq!(program.advanced_state(7), None, "index outside the program");
    }

    #[test]
    fn a_dwell_without_a_target_follows_the_program_and_wraps_only_when_it_loops() {
        let timed = |seconds: f32| {
            angle_state(0.0).with_dwell(Some(
                DriveDwell::new(seconds, None).expect("dwell time is in range"),
            ))
        };
        let states = [timed(1.0), timed(1.0)];

        let once = DriveProgram::new(&states, false).expect("program is valid");
        assert_eq!(once.advanced_state(0), Some(1));
        assert_eq!(once.advanced_state(1), None, "the last state holds");

        let looping = DriveProgram::new(&states, true).expect("program is valid");
        assert_eq!(looping.advanced_state(1), Some(0));
    }

    #[test]
    fn removing_a_state_renumbers_every_reference_to_the_states_after_it() {
        let hold = angle_state(0.0);
        let middle = angle_state(30.0).with_dwell(Some(
            DriveDwell::new(1.0, Some(2)).expect("dwell time is in range"),
        ));
        let last = angle_state(60.0)
            .with_trigger(Some(DriveTrigger::new(key('D'), DriveRelease::RevertTo(2))))
            .with_dwell(Some(
                DriveDwell::new(1.0, Some(0)).expect("dwell time is in range"),
            ));
        let program = DriveProgram::new(&[hold, middle, last], false).expect("program is valid");

        // Drop the middle state: the dwell that pointed at state 2 now points
        // at the state that took its index, and the release that pointed at the
        // removed state falls back to state 0.
        let trimmed = program
            .with_removed_state(1)
            .expect("a two-state program survives a removal");
        assert_eq!(trimmed.len(), 2);
        let survivor = trimmed.state(1).expect("the last state survives");
        assert_eq!(
            survivor.trigger().map(DriveTrigger::release),
            Some(DriveRelease::RevertTo(1)),
        );
        assert_eq!(survivor.dwell().and_then(DriveDwell::next), Some(0));

        let single = DriveProgram::default();
        assert_eq!(
            single.with_removed_state(0),
            Err(DriveProgramError::EmptyProgram)
        );
    }

    #[test]
    fn state_lookup_by_key_finds_the_bound_state() {
        let latched = |degrees: f32, symbol: char| {
            angle_state(degrees)
                .with_trigger(Some(DriveTrigger::new(key(symbol), DriveRelease::Latch)))
        };
        let program = DriveProgram::new(
            &[latched(30.0, 'Q'), latched(40.0, 'W'), latched(80.0, 'E')],
            false,
        )
        .expect("arm program is valid");

        assert_eq!(program.state_for_key(key('Q')), Some(0));
        assert_eq!(program.state_for_key(key('E')), Some(2));
        assert_eq!(program.state_for_key(key('Z')), None);
    }

    #[test]
    fn drive_limits_validate_speed_torque_and_limit_ordering() {
        let default = DriveLimits::default();
        assert!((default.max_speed_rad_s() - 3.0).abs() < f32::EPSILON);
        assert!(default.max_torque_newton_meters().is_infinite());
        assert!(default.min_angle().is_infinite() && default.min_angle() < 0.0);
        assert!(default.max_angle().is_infinite() && default.max_angle() > 0.0);

        assert_eq!(
            DriveLimits::new(0.0, 1.0, None),
            Err(DriveLimitsError::SpeedOutOfRange)
        );
        assert_eq!(
            DriveLimits::new(25.5, 1.0, None),
            Err(DriveLimitsError::SpeedOutOfRange)
        );
        assert_eq!(
            DriveLimits::new(1.0, 0.0, None),
            Err(DriveLimitsError::NonPositiveTorque)
        );
        assert_eq!(
            DriveLimits::new(1.0, 1.0, Some((0.0, f32::INFINITY))),
            Err(DriveLimitsError::LimitOutOfRange)
        );
        assert_eq!(
            DriveLimits::new(1.0, 1.0, Some((1.0, 1.0))),
            Err(DriveLimitsError::InvertedLimits)
        );

        let limited = DriveLimits::new(2.5, 40.0, Some((-1.0, 2.0))).expect("limits are valid");
        assert!((limited.min_angle() + 1.0).abs() < f32::EPSILON);
        assert!((limited.max_angle() - 2.0).abs() < f32::EPSILON);
        assert!(limited.with_angle_limits(None).is_ok());
        assert_eq!(
            limited.with_max_torque(-1.0),
            Err(DriveLimitsError::NonPositiveTorque)
        );
    }
}
