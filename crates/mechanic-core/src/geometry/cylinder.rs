//! Cylinders and retained cylinder sectors.

use super::face::FaceKind;
use super::grid::{
    BuildPose, GRID_UNIT_METERS, GRID_UNIT_TICKS, MAX_GRID_UNITS, POSITION_TICK_METERS,
    POSITION_TICKS_PER_GRID_UNIT,
};
use super::layers::{
    LayerEnvelope, LayerError, LayerFace, LayerRegion, MaterialLayer, MaterialLayers,
    layer_thickness_ticks, shifted_pose, unwind_layer_regions,
};
use super::material::ConstructionMaterial;
use super::spiral::{SpiralError, SpiralSpec};
use crate::MaterialAppearance;
use bevy_math::Vec3;
use thiserror::Error;

/// Smallest supported cylinder outer diameter, in metres.
pub const MIN_CYLINDER_OUTER_DIAMETER: f32 = 0.05;

/// Largest supported cylinder outer diameter, in metres.
pub const MAX_CYLINDER_OUTER_DIAMETER: f32 = 8.0;

/// Minimum difference between a cylinder's outer and inner diameters, in metres.
pub const MIN_CYLINDER_DIAMETER_GAP: f32 = 0.05;

/// Smallest supported retained cylinder sector, in degrees.
pub const MIN_CYLINDER_SWEEP_DEGREES: u16 = 15;

/// Full-cylinder sweep angle, in degrees.
pub const MAX_CYLINDER_SWEEP_DEGREES: u16 = 360;

/// Adjustment increment for retained cylinder sectors, in degrees.
pub const CYLINDER_SWEEP_STEP_DEGREES: u16 = 15;

/// Invalid load-bearing cylinder dimensions.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum CylinderDimensionError {
    /// The outer diameter was not finite.
    #[error("cylinder outer diameter must be finite")]
    NonFiniteOuterDiameter,
    /// The outer diameter was outside the supported range.
    #[error("cylinder outer diameter must be between 0.05 m and 8.00 m")]
    OuterDiameterOutOfRange,
    /// The inner diameter was not finite.
    #[error("cylinder inner diameter must be finite")]
    NonFiniteInnerDiameter,
    /// The inner diameter was negative or left less than the minimum wall thickness.
    #[error(
        "cylinder inner diameter must be non-negative and at least 0.05 m smaller than the outer diameter"
    )]
    InnerDiameterOutOfRange,
    /// The axial length was not finite.
    #[error("cylinder axial length must be finite")]
    NonFiniteAxialLength,
    /// The axial length was outside the supported range or not a quarter-metre increment.
    #[error("cylinder axial length must be between 0.25 m and 8.00 m in 0.25 m increments")]
    AxialLengthOutOfRange,
    /// The retained angular sector was outside the supported stepped range.
    #[error("cylinder sweep angle must be between 15 and 360 degrees in 15-degree increments")]
    SweepAngleOutOfRange,
}

/// Validated dimensions for a solid or hollow cylinder.
///
/// Authored cylinders have quarter-metre lengths; only end-cap material layers
/// lengthen an envelope off that grid, in whole position ticks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CylinderDimensions {
    pub(super) outer_diameter: f32,
    pub(super) inner_diameter: f32,
    pub(super) axial_length_ticks: u16,
    pub(super) sweep_angle_degrees: u16,
}

impl CylinderDimensions {
    /// Default cylinder outer diameter, in metres.
    pub const DEFAULT_OUTER_DIAMETER: f32 = 0.25;
    /// Default cylinder inner diameter, in metres.
    pub const DEFAULT_INNER_DIAMETER: f32 = 0.0;
    /// Default cylinder axial length, in metres.
    pub const DEFAULT_AXIAL_LENGTH: f32 = 0.25;

    /// Creates validated cylinder dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`CylinderDimensionError`] when a diameter is non-finite or out
    /// of range, or when the length is not a supported quarter-metre increment.
    pub fn new(
        outer_diameter: f32,
        inner_diameter: f32,
        axial_length: f32,
    ) -> Result<Self, CylinderDimensionError> {
        if !outer_diameter.is_finite() {
            return Err(CylinderDimensionError::NonFiniteOuterDiameter);
        }
        if !(MIN_CYLINDER_OUTER_DIAMETER..=MAX_CYLINDER_OUTER_DIAMETER).contains(&outer_diameter) {
            return Err(CylinderDimensionError::OuterDiameterOutOfRange);
        }
        if !inner_diameter.is_finite() {
            return Err(CylinderDimensionError::NonFiniteInnerDiameter);
        }
        if inner_diameter < 0.0 || inner_diameter > outer_diameter - MIN_CYLINDER_DIAMETER_GAP {
            return Err(CylinderDimensionError::InnerDiameterOutOfRange);
        }
        if !axial_length.is_finite() {
            return Err(CylinderDimensionError::NonFiniteAxialLength);
        }
        let length_units = axial_length / GRID_UNIT_METERS;
        let rounded_units = length_units.round();
        if (length_units - rounded_units).abs() > 1.0e-5
            || !(1.0..=f32::from(MAX_GRID_UNITS)).contains(&rounded_units)
        {
            return Err(CylinderDimensionError::AxialLengthOutOfRange);
        }
        let units = (1..=MAX_GRID_UNITS)
            .find(|&units| (f32::from(units) - rounded_units).abs() < 1.0e-5)
            .ok_or(CylinderDimensionError::AxialLengthOutOfRange)?;
        Self::validated(
            outer_diameter,
            inner_diameter,
            u16::from(units) * GRID_UNIT_TICKS,
            MAX_CYLINDER_SWEEP_DEGREES,
        )
    }

    /// Validates a whole envelope, including an off-grid tick length.
    pub(super) fn validated(
        outer_diameter: f32,
        inner_diameter: f32,
        axial_length_ticks: u16,
        sweep_angle_degrees: u16,
    ) -> Result<Self, CylinderDimensionError> {
        if !outer_diameter.is_finite() {
            return Err(CylinderDimensionError::NonFiniteOuterDiameter);
        }
        if !(MIN_CYLINDER_OUTER_DIAMETER..=MAX_CYLINDER_OUTER_DIAMETER + 1.0e-4)
            .contains(&outer_diameter)
        {
            return Err(CylinderDimensionError::OuterDiameterOutOfRange);
        }
        if !inner_diameter.is_finite() {
            return Err(CylinderDimensionError::NonFiniteInnerDiameter);
        }
        if inner_diameter < 0.0
            || inner_diameter > outer_diameter - MIN_CYLINDER_DIAMETER_GAP + 1.0e-4
        {
            return Err(CylinderDimensionError::InnerDiameterOutOfRange);
        }
        if !(GRID_UNIT_TICKS..=u16::from(MAX_GRID_UNITS) * GRID_UNIT_TICKS)
            .contains(&axial_length_ticks)
        {
            return Err(CylinderDimensionError::AxialLengthOutOfRange);
        }
        Self {
            outer_diameter,
            inner_diameter,
            axial_length_ticks,
            sweep_angle_degrees: MAX_CYLINDER_SWEEP_DEGREES,
        }
        .with_sweep_angle_degrees(sweep_angle_degrees)
    }

    /// Sets the retained angular sector centred on local positive X.
    ///
    /// # Errors
    ///
    /// Returns [`CylinderDimensionError::SweepAngleOutOfRange`] unless the
    /// angle is between 15 and 360 degrees in 15-degree increments.
    pub const fn with_sweep_angle_degrees(
        mut self,
        sweep_angle_degrees: u16,
    ) -> Result<Self, CylinderDimensionError> {
        if sweep_angle_degrees < MIN_CYLINDER_SWEEP_DEGREES
            || sweep_angle_degrees > MAX_CYLINDER_SWEEP_DEGREES
            || !sweep_angle_degrees.is_multiple_of(CYLINDER_SWEEP_STEP_DEGREES)
        {
            return Err(CylinderDimensionError::SweepAngleOutOfRange);
        }
        self.sweep_angle_degrees = sweep_angle_degrees;
        Ok(self)
    }

    /// Outer diameter in metres.
    pub const fn outer_diameter(self) -> f32 {
        self.outer_diameter
    }

    /// Inner diameter in metres. Zero represents a solid cylinder.
    pub const fn inner_diameter(self) -> f32 {
        self.inner_diameter
    }

    /// Axial length in metres.
    pub fn axial_length(self) -> f32 {
        f32::from(self.axial_length_ticks) * POSITION_TICK_METERS
    }

    /// Axial length in whole quarter-metre grid units, rounded down.
    #[expect(clippy::cast_possible_truncation, reason = "at most 32 units")]
    pub const fn axial_length_units(self) -> u8 {
        (self.axial_length_ticks / GRID_UNIT_TICKS) as u8
    }

    /// Axial length in position ticks.
    pub const fn axial_length_ticks(self) -> u16 {
        self.axial_length_ticks
    }

    /// Retained angular sector in degrees, centred on local positive X.
    pub const fn sweep_angle_degrees(self) -> u16 {
        self.sweep_angle_degrees
    }

    /// Retained angular sector in radians.
    pub fn sweep_angle_radians(self) -> f32 {
        f32::from(self.sweep_angle_degrees).to_radians()
    }
}

impl Default for CylinderDimensions {
    fn default() -> Self {
        Self {
            outer_diameter: Self::DEFAULT_OUTER_DIAMETER,
            inner_diameter: Self::DEFAULT_INNER_DIAMETER,
            axial_length_ticks: GRID_UNIT_TICKS,
            sweep_angle_degrees: MAX_CYLINDER_SWEEP_DEGREES,
        }
    }
}

/// Editable cylinder dimensions and build pose. Its axis is local Y.
///
/// A layered cylinder is still one solid: `dimensions` and `pose` describe its
/// whole envelope, and [`MaterialLayers`] only divide that envelope's material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CylinderSpec {
    /// Validated solid or hollow envelope dimensions.
    pub dimensions: CylinderDimensions,
    /// Envelope centre and cardinal orientation.
    pub pose: BuildPose,
    /// Core material used for appearance, mass, and contact response.
    pub material: ConstructionMaterial,
    /// Core color and finish treatment.
    pub appearance: MaterialAppearance,
    pub(super) layers: MaterialLayers,
    pub(super) spiral: Option<SpiralSpec>,
}

impl CylinderSpec {
    /// Creates a cylinder from validated dimensions and a build pose.
    pub const fn new(dimensions: CylinderDimensions, pose: BuildPose) -> Self {
        Self {
            dimensions,
            pose,
            material: ConstructionMaterial::Steel,
            appearance: MaterialAppearance::BAKED,
            layers: MaterialLayers::NONE,
            spiral: None,
        }
    }

    /// Material layers over the core, oldest first.
    pub const fn layers(self) -> MaterialLayers {
        self.layers
    }

    /// The spiral drawn into this cylinder's walls, if any.
    pub const fn spiral(self) -> Option<SpiralSpec> {
        self.spiral
    }

    /// Draws a spiral into this cylinder's walls, replacing any it had. The
    /// dimensions stay the envelope: the outer profile cuts in from the outer
    /// diameter and the bore profile out from the inner one.
    ///
    /// # Errors
    ///
    /// Returns [`SpiralError`] when the cylinder is a sector, layered, or solid
    /// under a bore profile, when the cuts leave too little wall, when the
    /// taper does not fit, or when the ridges need too many colliders.
    pub fn with_spiral(self, spiral: SpiralSpec) -> Result<Self, SpiralError> {
        let dimensions = self.dimensions;
        if dimensions.sweep_angle_degrees != MAX_CYLINDER_SWEEP_DEGREES {
            return Err(SpiralError::PartialSector);
        }
        if !self.layers.is_empty() {
            return Err(SpiralError::Layered);
        }
        if !spiral.inner().is_plain() && dimensions.inner_diameter <= 0.0 {
            return Err(SpiralError::BoreRequired);
        }
        let ticks = |ticks: u16| f32::from(ticks) * POSITION_TICK_METERS;
        let core_radius = dimensions.outer_diameter * 0.5 - ticks(spiral.outer().max_depth_ticks());
        let bore_radius = dimensions.inner_diameter * 0.5 + ticks(spiral.inner().max_depth_ticks());
        let wall = MIN_CYLINDER_DIAMETER_GAP * 0.5 - 1.0e-4;
        let mut tip_scale = 1.0;
        if let Some(taper) = spiral.taper() {
            let tip = ticks(taper.tip_diameter_ticks);
            if taper.length_ticks == 0
                || taper.length_ticks > dimensions.axial_length_ticks
                || taper.tip_diameter_ticks < super::spiral::MIN_SPIRAL_TIP_DIAMETER_TICKS
                || tip > dimensions.outer_diameter + 1.0e-4
            {
                return Err(SpiralError::TaperOutOfRange);
            }
            tip_scale = tip / dimensions.outer_diameter;
        }
        if core_radius - bore_radius < wall
            || (dimensions.inner_diameter > 0.0 && core_radius * tip_scale - bore_radius < wall)
        {
            return Err(SpiralError::TooDeep);
        }
        if spiral
            .collider_steps_per_turn(dimensions.axial_length())
            .is_none()
        {
            return Err(SpiralError::TooFine);
        }
        Ok(Self {
            spiral: Some(spiral),
            ..self
        })
    }

    /// The plain cylinder this spiral was drawn into.
    #[must_use]
    pub const fn without_spiral(mut self) -> Self {
        self.spiral = None;
        self
    }

    /// Material of the outermost curved wall, which meets the world.
    pub fn outer_contact_material(self) -> ConstructionMaterial {
        self.layers
            .iter()
            .rev()
            .find(|layer| layer.face == LayerFace::OuterWall)
            .map_or(self.material, |layer| layer.material)
    }

    /// Adds a material layer to the outer wall, bore, or one end cap.
    ///
    /// A bore layer thicker than the bore's radius fills it to a solid core. An
    /// end-cap layer lengthens the cylinder and moves its centre outward by half
    /// the thickness, so the opposite cap stays put.
    ///
    /// # Errors
    ///
    /// Returns [`LayerError`] when the thickness, face, layer count, or grown
    /// envelope is invalid.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        reason = "validated tick lengths within 8 m"
    )]
    pub fn with_layer(
        self,
        face: LayerFace,
        thickness: f32,
        material: ConstructionMaterial,
        appearance: MaterialAppearance,
    ) -> Result<Self, LayerError> {
        if self.spiral.is_some() {
            return Err(LayerError::SpiralPart);
        }
        let dimensions = self.dimensions;
        let layer = |thickness| MaterialLayer {
            face,
            thickness,
            material,
            appearance,
        };
        let validated = |outer, inner, ticks| {
            CylinderDimensions::validated(outer, inner, ticks, dimensions.sweep_angle_degrees)
        };
        let (dimensions, pose, layers) = match face {
            LayerFace::OuterWall => {
                layer_thickness_ticks(thickness, false)?;
                let layers = self.layers.pushed(layer(thickness))?;
                let outer = dimensions.outer_diameter + 2.0 * thickness;
                if outer > MAX_CYLINDER_OUTER_DIAMETER + 1.0e-4 {
                    return Err(LayerError::TooLarge);
                }
                (
                    validated(
                        outer,
                        dimensions.inner_diameter,
                        dimensions.axial_length_ticks,
                    )?,
                    self.pose,
                    layers,
                )
            }
            LayerFace::Bore => {
                layer_thickness_ticks(thickness, false)?;
                if dimensions.inner_diameter <= 0.0 {
                    return Err(LayerError::BoreRequired);
                }
                let inner = (dimensions.inner_diameter - 2.0 * thickness).max(0.0);
                let layers = self
                    .layers
                    .pushed(layer((dimensions.inner_diameter - inner) * 0.5))?;
                (
                    validated(
                        dimensions.outer_diameter,
                        inner,
                        dimensions.axial_length_ticks,
                    )?,
                    self.pose,
                    layers,
                )
            }
            LayerFace::Face(cap @ (FaceKind::PositiveY | FaceKind::NegativeY)) => {
                let ticks = layer_thickness_ticks(thickness, true)?;
                let layers = self
                    .layers
                    .pushed(layer(ticks as f32 * POSITION_TICK_METERS))?;
                let length = i32::from(dimensions.axial_length_ticks) + ticks;
                if length > i32::from(MAX_GRID_UNITS) * POSITION_TICKS_PER_GRID_UNIT {
                    return Err(LayerError::TooLarge);
                }
                (
                    validated(
                        dimensions.outer_diameter,
                        dimensions.inner_diameter,
                        length as u16,
                    )?,
                    shifted_pose(self.pose, Vec3::Y * cap.sign() * (ticks / 2) as f32),
                    layers,
                )
            }
            LayerFace::Face(_) => return Err(LayerError::UnsupportedFace),
        };
        Ok(Self {
            dimensions,
            pose,
            layers,
            ..self
        })
    }

    /// The core this cylinder's layers were laid on.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        reason = "unwinds validated ticks within 8 m"
    )]
    pub fn without_layers(self) -> Self {
        let mut outer = self.dimensions.outer_diameter;
        let mut inner = self.dimensions.inner_diameter;
        let mut length = i32::from(self.dimensions.axial_length_ticks);
        let mut shift = 0;
        for layer in self.layers.iter() {
            match layer.face {
                LayerFace::OuterWall => outer -= 2.0 * layer.thickness,
                LayerFace::Bore => inner += 2.0 * layer.thickness,
                LayerFace::Face(face) => {
                    let ticks = (layer.thickness / POSITION_TICK_METERS).round() as i32;
                    length -= ticks;
                    shift += ticks / 2 * face.sign() as i32;
                }
            }
        }
        Self {
            dimensions: CylinderDimensions {
                outer_diameter: outer,
                inner_diameter: if inner < 1.0e-5 { 0.0 } else { inner },
                axial_length_ticks: length.max(0) as u16,
                ..self.dimensions
            },
            pose: shifted_pose(self.pose, Vec3::Y * -(shift as f32)),
            layers: MaterialLayers::NONE,
            ..self
        }
    }

    /// Each layer's part-local region, oldest first.
    pub fn layer_regions(self) -> Vec<LayerRegion> {
        let half_length = self.dimensions.axial_length() * 0.5;
        let outer = self.dimensions.outer_diameter * 0.5;
        unwind_layer_regions(
            LayerEnvelope {
                minimum: Vec3::new(-outer, -half_length, -outer),
                maximum: Vec3::new(outer, half_length, outer),
                outer_radius: outer,
                inner_radius: self.dimensions.inner_diameter * 0.5,
            },
            self.layers,
        )
    }

    /// Uses an explicit construction material.
    #[must_use]
    pub const fn with_material(mut self, material: ConstructionMaterial) -> Self {
        self.material = material;
        self
    }

    /// Uses an explicit construction appearance.
    #[must_use]
    pub const fn with_appearance(mut self, appearance: MaterialAppearance) -> Self {
        self.appearance = appearance;
        self
    }
}
