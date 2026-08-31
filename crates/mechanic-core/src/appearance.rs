//! Validated per-part construction appearance.

use serde::{Deserialize, Deserializer, Serialize};
use thiserror::Error;

/// A construction surface's independent color and finish treatments.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MaterialAppearance {
    /// Color treatment applied to the baked base-color map.
    pub color: MaterialColor,
    /// Surface response applied after sampling the baked ORM map.
    pub finish: MaterialFinish,
}

impl MaterialAppearance {
    /// Untouched texture color and surface response.
    pub const BAKED: Self = Self {
        color: MaterialColor::Baked,
        finish: MaterialFinish::Baked,
    };

    /// Creates an appearance from independently selected treatments.
    pub const fn new(color: MaterialColor, finish: MaterialFinish) -> Self {
        Self { color, finish }
    }
}

/// Color treatment applied to a construction material.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum MaterialColor {
    /// Preserve the baked base-color map.
    #[default]
    Baked,
    /// Rotate and scale the baked material's `OKLab` color.
    Shift(MaterialShift),
    /// Re-anchor the baked material's structure around an absolute RGB color.
    Dye(MaterialDye),
}

/// Surface treatment applied after sampling a construction material's ORM map.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MaterialFinish {
    /// Preserve baked metalness and roughness.
    #[default]
    Baked,
    /// Force a metallic surface while retaining baked roughness.
    Anodised,
    /// Apply a low-metalness semi-gloss coating while retaining roughness detail.
    Painted,
}

/// Validated relative `OKLab` color adjustment.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct MaterialShift {
    hue_degrees: f32,
    chroma: f32,
    lightness: f32,
}

impl MaterialShift {
    /// Creates a relative color adjustment.
    ///
    /// # Errors
    ///
    /// Returns [`AppearanceError`] if any value is non-finite or outside its
    /// supported range.
    pub fn new(hue_degrees: f32, chroma: f32, lightness: f32) -> Result<Self, AppearanceError> {
        validate(hue_degrees, -180.0..=180.0, AppearanceError::InvalidHue)?;
        validate(chroma, 0.0..=1.8, AppearanceError::InvalidChroma)?;
        validate(lightness, 0.0..=2.0, AppearanceError::InvalidLightness)?;
        Ok(Self {
            hue_degrees,
            chroma,
            lightness,
        })
    }

    /// `OKLab` hue rotation in degrees.
    pub const fn hue_degrees(self) -> f32 {
        self.hue_degrees
    }

    /// `OKLab` chroma multiplier.
    pub const fn chroma(self) -> f32 {
        self.chroma
    }

    /// `OKLab` lightness multiplier.
    pub const fn lightness(self) -> f32 {
        self.lightness
    }
}

impl<'de> Deserialize<'de> for MaterialShift {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            hue_degrees: f32,
            chroma: f32,
            lightness: f32,
        }
        let raw = Raw::deserialize(deserializer)?;
        Self::new(raw.hue_degrees, raw.chroma, raw.lightness).map_err(serde::de::Error::custom)
    }
}

/// Validated absolute dye color and retained-structure strength.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct MaterialDye {
    target_rgb: [u8; 3],
    structure: f32,
}

impl MaterialDye {
    /// Creates an opaque RGB8 dye treatment.
    ///
    /// # Errors
    ///
    /// Returns [`AppearanceError::InvalidStructure`] when `structure` is
    /// non-finite or outside `0..=3`.
    pub fn new(target_rgb: [u8; 3], structure: f32) -> Result<Self, AppearanceError> {
        validate(structure, 0.0..=3.0, AppearanceError::InvalidStructure)?;
        Ok(Self {
            target_rgb,
            structure,
        })
    }

    /// Absolute opaque sRGB target.
    pub const fn target_rgb(self) -> [u8; 3] {
        self.target_rgb
    }

    /// Multiplier for baked lightness variation around the target.
    pub const fn structure(self) -> f32 {
        self.structure
    }
}

impl<'de> Deserialize<'de> for MaterialDye {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            target_rgb: [u8; 3],
            structure: f32,
        }
        let raw = Raw::deserialize(deserializer)?;
        Self::new(raw.target_rgb, raw.structure).map_err(serde::de::Error::custom)
    }
}

/// Invalid Chroma brush parameter.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum AppearanceError {
    /// Hue was non-finite or outside −180°..=180°.
    #[error("appearance hue must be finite and between -180 and 180 degrees")]
    InvalidHue,
    /// Chroma was non-finite or outside 0..=1.8.
    #[error("appearance chroma must be finite and between 0 and 1.8")]
    InvalidChroma,
    /// Lightness was non-finite or outside 0..=2.
    #[error("appearance lightness must be finite and between 0 and 2")]
    InvalidLightness,
    /// Dye structure was non-finite or outside 0..=3.
    #[error("appearance dye structure must be finite and between 0 and 3")]
    InvalidStructure,
}

fn validate(
    value: f32,
    range: core::ops::RangeInclusive<f32>,
    error: AppearanceError,
) -> Result<(), AppearanceError> {
    if value.is_finite() && range.contains(&value) {
        Ok(())
    } else {
        Err(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_and_dye_ranges_are_validated() {
        assert!(MaterialShift::new(-180.0, 0.0, 2.0).is_ok());
        assert!(MaterialShift::new(180.1, 1.0, 1.0).is_err());
        assert!(MaterialShift::new(0.0, 1.81, 1.0).is_err());
        assert!(MaterialShift::new(0.0, 1.0, f32::NAN).is_err());
        assert!(MaterialDye::new([1, 2, 3], 3.0).is_ok());
        assert!(MaterialDye::new([1, 2, 3], -0.1).is_err());
    }

    #[test]
    fn baked_is_the_default_appearance() {
        assert_eq!(MaterialAppearance::default(), MaterialAppearance::BAKED);
    }
}
