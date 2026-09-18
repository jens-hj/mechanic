//! Construction materials and their physical properties.

use super::grid::GRID_UNIT_METERS;
use serde::{Deserialize, Serialize};

/// A selectable material for ordinary construction parts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstructionMaterial {
    /// Lightweight aluminium alloy.
    Aluminium,
    /// Graphite construction material.
    Graphite,
    /// Carbon-fibre composite construction material.
    CarbonFiber,
    /// Dense, high-friction concrete.
    Concrete,
    /// Dense conductive copper.
    Copper,
    /// Compactable earth fill.
    Dirt,
    /// General-purpose structural iron.
    Iron,
    /// Lightweight resilient plastic.
    Plastic,
    /// Compliant high-grip rubber.
    Rubber,
    /// Granular mineral fill.
    Sand,
    /// General-purpose structural steel.
    #[default]
    Steel,
    /// Dense natural stone.
    Stone,
    /// Lightweight timber.
    Wood,
}

impl ConstructionMaterial {
    /// Every selectable material in alphabetical display order.
    pub const ALL: [Self; 13] = [
        Self::Aluminium,
        Self::CarbonFiber,
        Self::Concrete,
        Self::Copper,
        Self::Dirt,
        Self::Graphite,
        Self::Iron,
        Self::Plastic,
        Self::Rubber,
        Self::Sand,
        Self::Steel,
        Self::Stone,
        Self::Wood,
    ];

    /// Human-readable material name.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Aluminium => "Aluminium",
            Self::CarbonFiber => "Carbon Fiber",
            Self::Concrete => "Concrete",
            Self::Copper => "Copper",
            Self::Dirt => "Dirt",
            Self::Graphite => "Graphite",
            Self::Iron => "Iron",
            Self::Plastic => "Plastic",
            Self::Rubber => "Rubber",
            Self::Sand => "Sand",
            Self::Steel => "Steel",
            Self::Stone => "Stone",
            Self::Wood => "Wood",
        }
    }

    /// Density and contact response used by construction physics.
    pub const fn properties(self) -> MaterialProperties {
        match self {
            Self::Aluminium => MaterialProperties::new(2_700.0, 0.61, 0.47, 0.25, 0.004, 69.0e9),
            Self::CarbonFiber => MaterialProperties::new(1_600.0, 0.40, 0.30, 0.20, 0.008, 70.0e9),
            Self::Concrete => MaterialProperties::new(2_400.0, 0.80, 0.65, 0.05, 0.020, 30.0e9),
            Self::Copper => MaterialProperties::new(8_960.0, 0.53, 0.36, 0.20, 0.003, 117.0e9),
            Self::Dirt => MaterialProperties::new(1_600.0, 0.72, 0.55, 0.05, 0.030, 0.05e9),
            Self::Graphite => MaterialProperties::new(1_900.0, 0.25, 0.15, 0.10, 0.010, 12.0e9),
            Self::Iron => MaterialProperties::new(7_870.0, 0.70, 0.55, 0.15, 0.003, 170.0e9),
            Self::Plastic => MaterialProperties::new(950.0, 0.40, 0.30, 0.40, 0.020, 1.0e9),
            Self::Rubber => MaterialProperties::new(1_100.0, 1.00, 0.80, 0.70, 0.010, 0.01e9),
            Self::Sand => MaterialProperties::new(1_700.0, 0.65, 0.50, 0.05, 0.035, 0.03e9),
            Self::Steel => MaterialProperties::new(7_850.0, 0.74, 0.57, 0.20, 0.002, 200.0e9),
            Self::Stone => MaterialProperties::new(2_700.0, 0.60, 0.48, 0.05, 0.015, 50.0e9),
            Self::Wood => MaterialProperties::new(700.0, 0.48, 0.30, 0.15, 0.025, 10.0e9),
        }
    }
}

/// How a surface answers contact: the four coefficients every solver mixes
/// pairwise. Construction materials and terrain materials both resolve to this.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceResponse {
    /// Static Coulomb friction coefficient.
    pub static_friction: f32,
    /// Kinetic Coulomb friction coefficient.
    pub dynamic_friction: f32,
    /// Coefficient of restitution.
    pub restitution: f32,
    /// Dimensionless rolling-resistance coefficient.
    pub rolling_resistance: f32,
}

impl SurfaceResponse {
    /// Static friction, dynamic friction, restitution, and rolling resistance.
    pub const fn to_array(self) -> [f32; 4] {
        [
            self.static_friction,
            self.dynamic_friction,
            self.restitution,
            self.rolling_resistance,
        ]
    }
}

/// Physical properties belonging to one construction material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialProperties {
    /// Density in kilograms per cubic metre.
    pub density_kg_m3: f32,
    /// Static Coulomb friction coefficient.
    pub static_friction: f32,
    /// Kinetic Coulomb friction coefficient.
    pub dynamic_friction: f32,
    /// Coefficient of restitution.
    pub restitution: f32,
    /// Dimensionless rolling-resistance coefficient.
    pub rolling_resistance: f32,
    /// Young's modulus in pascals.
    pub youngs_modulus_pa: f32,
}

impl MaterialProperties {
    pub(super) const fn new(
        density_kg_m3: f32,
        static_friction: f32,
        dynamic_friction: f32,
        restitution: f32,
        rolling_resistance: f32,
        youngs_modulus_pa: f32,
    ) -> Self {
        Self {
            density_kg_m3,
            static_friction,
            dynamic_friction,
            restitution,
            rolling_resistance,
            youngs_modulus_pa,
        }
    }

    /// The contact coefficients, without density or stiffness.
    pub const fn surface_response(self) -> SurfaceResponse {
        SurfaceResponse {
            static_friction: self.static_friction,
            dynamic_friction: self.dynamic_friction,
            restitution: self.restitution,
            rolling_resistance: self.rolling_resistance,
        }
    }

    /// Nominal normal compliance of one 25 cm construction block, in metres
    /// per newton.
    ///
    /// A one-dimensional block column has stiffness `E A / L`. At the engine's
    /// nominal one-block contact area (`A = L²`) this reduces to
    /// `C = 1 / (E × L)`, with `L = GRID_UNIT_METERS`.
    pub const fn nominal_block_compliance(self) -> f32 {
        1.0 / (self.youngs_modulus_pa * GRID_UNIT_METERS)
    }
}
