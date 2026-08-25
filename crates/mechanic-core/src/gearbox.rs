use core::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Largest number of selectable gears supported by an engine line.
pub const MAX_GEARS: usize = 18;
/// Smallest supported input-to-output ratio.
pub const MIN_GEAR_RATIO: f32 = 0.25;
/// Largest supported input-to-output ratio.
pub const MAX_GEAR_RATIO: f32 = 20.0;

/// Whether shifts are selected by the simulation or by bound keys.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShiftMode {
    /// Select gears from measured joint speed and requested direction.
    #[default]
    Auto,
    /// Move through the ordered gear strip using the configured bindings.
    Manual,
}

/// A physical key accepted by a gearbox binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GearKey {
    /// An uppercase ASCII letter.
    Letter(char),
    /// An ASCII digit.
    Digit(u8),
    /// Space bar.
    Space,
    /// Up arrow.
    ArrowUp,
    /// Down arrow.
    ArrowDown,
    /// Left arrow.
    ArrowLeft,
    /// Right arrow.
    ArrowRight,
    /// Page Up.
    PageUp,
    /// Page Down.
    PageDown,
}

impl GearKey {
    /// Creates a letter or digit binding. Lowercase letters are folded to uppercase.
    pub const fn from_char(symbol: char) -> Option<Self> {
        let symbol = symbol.to_ascii_uppercase();
        if symbol.is_ascii_uppercase() {
            Some(Self::Letter(symbol))
        } else if symbol.is_ascii_digit() {
            Some(Self::Digit(symbol as u8 - b'0'))
        } else {
            None
        }
    }
}

impl fmt::Display for GearKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Letter(letter) => write!(formatter, "{letter}"),
            Self::Digit(digit) => write!(formatter, "{digit}"),
            Self::Space => formatter.write_str("Space"),
            Self::ArrowUp => formatter.write_str("Up"),
            Self::ArrowDown => formatter.write_str("Down"),
            Self::ArrowLeft => formatter.write_str("Left"),
            Self::ArrowRight => formatter.write_str("Right"),
            Self::PageUp => formatter.write_str("Page Up"),
            Self::PageDown => formatter.write_str("Page Down"),
        }
    }
}

/// One gearbox key and its optional modifier keys.
#[allow(clippy::struct_excessive_bools)] // A chord independently records the four platform modifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GearKeyChord {
    /// Main physical key.
    pub key: GearKey,
    /// Shift must be held.
    pub shift: bool,
    /// Control must be held.
    pub control: bool,
    /// Alt/Option must be held.
    pub alt: bool,
    /// Super/Command/Windows must be held.
    pub super_key: bool,
}

impl GearKeyChord {
    /// Creates a chord without modifiers.
    pub const fn new(key: GearKey) -> Self {
        Self {
            key,
            shift: false,
            control: false,
            alt: false,
            super_key: false,
        }
    }
}

impl fmt::Display for GearKeyChord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.control {
            formatter.write_str("Ctrl+")?;
        }
        if self.alt {
            formatter.write_str("Alt+")?;
        }
        if self.shift {
            formatter.write_str("Shift+")?;
        }
        if self.super_key {
            formatter.write_str("Super+")?;
        }
        self.key.fmt(formatter)
    }
}

/// Invalid gearbox configuration.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum GearboxError {
    /// A gearbox always has at least one ratio and never more than eighteen.
    #[error("a gearbox must contain between 1 and {MAX_GEARS} ratios")]
    InvalidGearCount,
    /// A ratio was non-finite or outside the supported range.
    #[error("gear ratios must be finite and between 0.25:1 and 20.0:1")]
    RatioOutOfRange,
    /// Shift order must move strictly from larger ratios to smaller ratios.
    #[error("gear ratios must be strictly descending in shift order")]
    RatiosNotDescending,
    /// A gas divider was outside the N+1 boundaries of the gear strip.
    #[error("the gas reverse-gear count cannot exceed the number of gears")]
    InvalidReverseGearCount,
}

/// Persistent settings for one Controller and one engine family.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GearboxConfig {
    mode: ShiftMode,
    ratios: Vec<f32>,
    reverse_gears: u8,
    gear_up: GearKeyChord,
    gear_down: GearKeyChord,
}

impl GearboxConfig {
    /// Validates and creates a complete gearbox configuration.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid gear count, ratio, ordering, or gas divider.
    pub fn new(
        mode: ShiftMode,
        ratios: Vec<f32>,
        reverse_gears: u8,
        gear_up: GearKeyChord,
        gear_down: GearKeyChord,
    ) -> Result<Self, GearboxError> {
        validate_ratios(&ratios)?;
        if usize::from(reverse_gears) > ratios.len() {
            return Err(GearboxError::InvalidReverseGearCount);
        }
        Ok(Self {
            mode,
            ratios,
            reverse_gears,
            gear_up,
            gear_down,
        })
    }

    /// Direct 1:1 automatic configuration used without a transmission and by old files.
    pub fn direct() -> Self {
        Self {
            mode: ShiftMode::Auto,
            ratios: vec![1.0],
            reverse_gears: 0,
            gear_up: GearKeyChord::new(GearKey::Letter('R')),
            gear_down: GearKeyChord::new(GearKey::Letter('F')),
        }
    }

    /// Default configuration for a chain depth. A depth of zero is direct drive.
    pub fn for_depth(depth: u8, gas: bool) -> Self {
        let gear_count = usize::from(depth).saturating_add(1).min(MAX_GEARS);
        let mut ratios = vec![1.0];
        if gear_count >= 2 {
            ratios = vec![3.0, 1.0];
        }
        if gear_count >= 3 {
            ratios.push(0.75);
        }
        while ratios.len() < gear_count {
            insert_largest_gap(&mut ratios);
        }
        Self {
            mode: ShiftMode::Auto,
            ratios,
            reverse_gears: u8::from(gas && depth != 0),
            gear_up: GearKeyChord::new(GearKey::Letter('R')),
            gear_down: GearKeyChord::new(GearKey::Letter('F')),
        }
    }

    /// Shift selection mode.
    pub const fn mode(&self) -> ShiftMode {
        self.mode
    }

    /// Ratios in shift order, expressed as input : output.
    pub fn ratios(&self) -> &[f32] {
        &self.ratios
    }

    /// Number of leading ratios belonging to the gas reverse bank.
    pub const fn reverse_gears(&self) -> u8 {
        self.reverse_gears
    }

    /// Manual upshift chord.
    pub const fn gear_up(&self) -> GearKeyChord {
        self.gear_up
    }

    /// Manual downshift chord.
    pub const fn gear_down(&self) -> GearKeyChord {
        self.gear_down
    }

    pub(crate) fn set_mode(&mut self, mode: ShiftMode) {
        self.mode = mode;
    }

    pub(crate) fn set_ratios(&mut self, ratios: Vec<f32>) -> Result<(), GearboxError> {
        validate_ratios(&ratios)?;
        if usize::from(self.reverse_gears) > ratios.len() {
            return Err(GearboxError::InvalidReverseGearCount);
        }
        self.ratios = ratios;
        Ok(())
    }

    pub(crate) fn set_bindings(&mut self, up: GearKeyChord, down: GearKeyChord) {
        self.gear_up = up;
        self.gear_down = down;
    }

    pub(crate) fn set_reverse_gears(&mut self, reverse_gears: u8) -> Result<(), GearboxError> {
        if usize::from(reverse_gears) > self.ratios.len() {
            return Err(GearboxError::InvalidReverseGearCount);
        }
        self.reverse_gears = reverse_gears;
        Ok(())
    }

    pub(crate) fn resized(mut self, gear_count: usize) -> Self {
        if gear_count == 1 {
            self.ratios = vec![1.0];
            self.reverse_gears = 0;
            return self;
        }
        self.ratios.truncate(gear_count);
        if self.ratios.len() == 1 {
            self.ratios = vec![3.0, 1.0];
        }
        if self.ratios.len() == 2 && gear_count >= 3 {
            if self.ratios == [3.0, 1.0] {
                self.ratios.push(0.75);
            } else {
                insert_largest_gap(&mut self.ratios);
            }
        }
        while self.ratios.len() < gear_count {
            insert_largest_gap(&mut self.ratios);
        }
        self.reverse_gears = self
            .reverse_gears
            .min(u8::try_from(gear_count).unwrap_or(u8::MAX));
        self
    }
}

fn validate_ratios(ratios: &[f32]) -> Result<(), GearboxError> {
    if ratios.is_empty() || ratios.len() > MAX_GEARS {
        return Err(GearboxError::InvalidGearCount);
    }
    if ratios
        .iter()
        .any(|ratio| !ratio.is_finite() || !(MIN_GEAR_RATIO..=MAX_GEAR_RATIO).contains(ratio))
    {
        return Err(GearboxError::RatioOutOfRange);
    }
    if ratios.windows(2).any(|pair| pair[0] <= pair[1]) {
        return Err(GearboxError::RatiosNotDescending);
    }
    Ok(())
}

fn insert_largest_gap(ratios: &mut Vec<f32>) {
    let (index, pair) = ratios
        .windows(2)
        .enumerate()
        .max_by(|(_, first), (_, second)| (first[0] / first[1]).total_cmp(&(second[0] / second[1])))
        .expect("gear generation has at least two ratios");
    ratios.insert(index + 1, (pair[0] * pair[1]).sqrt());
}

#[cfg(test)]
mod tests {
    use super::{
        GearKey, GearKeyChord, GearboxConfig, GearboxError, MAX_GEARS, MIN_GEAR_RATIO, ShiftMode,
    };

    #[test]
    fn defaults_grow_to_eighteen_strictly_descending_gears() {
        for depth in 0..=17 {
            let config = GearboxConfig::for_depth(depth, false);
            assert_eq!(config.ratios().len(), usize::from(depth) + 1);
            assert!(config.ratios().windows(2).all(|pair| pair[0] > pair[1]));
        }
        assert_eq!(GearboxConfig::for_depth(17, true).ratios().len(), MAX_GEARS);
    }

    #[test]
    fn gas_divider_accepts_both_extremes() {
        let mut config = GearboxConfig::for_depth(1, true);
        config.set_reverse_gears(0).unwrap();
        config.set_reverse_gears(2).unwrap();
        assert_eq!(
            config.set_reverse_gears(3),
            Err(GearboxError::InvalidReverseGearCount)
        );
    }

    #[test]
    fn growing_an_edited_two_gear_box_preserves_both_ratios_and_order() {
        let config = GearboxConfig::new(
            ShiftMode::Manual,
            vec![3.0, MIN_GEAR_RATIO],
            1,
            GearKeyChord::new(GearKey::Letter('R')),
            GearKeyChord::new(GearKey::Letter('F')),
        )
        .unwrap()
        .resized(3);
        assert_eq!(config.ratios().first(), Some(&3.0));
        assert_eq!(config.ratios().last(), Some(&MIN_GEAR_RATIO));
        assert!(config.ratios().windows(2).all(|pair| pair[0] > pair[1]));
    }
}
