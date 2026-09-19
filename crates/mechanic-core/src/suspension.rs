//! Validated suspension dimensions and pure SI geometry/force calculations.
//!
//! The asset guide supplies spring steel (79.3 `GPa`), rubber (5.5 `MPa`), and
//! piston-area damping (22,000 N·s/m³). Single-stage shock packaging is a game
//! construction rule, not a physical law. All user dimensions use 2.5 mm ticks.

use crate::BearingMassElement;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const TICK: f32 = crate::POSITION_TICK_METERS;
const EPS: f32 = 1.0e-6;
const MIN_TRAVEL: f32 = 0.05;
const PI: f32 = core::f32::consts::PI;

/// Invalid suspension input, with an actionable editor message.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum SuspensionError {
    /// All attachment rows must describe the same shared assembly.
    #[error("shared mounts must use identical suspension settings")]
    SharedMounts,
    /// Invalid length or off-grid input.
    #[error("{0}: use finite dimensions in 2.5 mm increments within the supported range")]
    Dimension(&'static str),
    /// Unsupported wire section.
    #[error("wire diameter must be 2–90 mm; adjust the spring OD or ID")]
    Wire,
    /// Invalid active coil count.
    #[error("use 3–16 whole active coils")]
    Coils,
    /// Too little travel remains.
    #[error("at least 50 mm travel is required; increase length or reduce stacked hardware")]
    Travel,
    /// A spring is too slender to meet the guide's solid-height rule.
    #[error(
        "closed spring length must be at least 25% of natural length; add coils or thicken wire"
    )]
    SolidHeight,
    /// Collars must also clear the spring.
    #[error(
        "hardware needs 2 mm radial clearance per side; increase spring ID or reduce inserted OD"
    )]
    RadialFit,
    /// No component supplies the two mounting plates.
    #[error("install a spring or shock")]
    Empty,
    /// Stop requires a shaft.
    #[error("a bump stop requires a shock")]
    MissingShock,
    /// Stop cannot cover most of the stroke.
    #[error("stop length must be 10 mm or more and no more than 80% of shock travel")]
    StopLength,
    /// Invalid stop section.
    #[error(
        "stop OD must clear the shaft by 6 mm, cover the gland shoulder, and fit within 2.6 × shock OD"
    )]
    StopDiameter,
    /// Initial position outside the legal joint range.
    #[error("starting compression exceeds available assembly travel")]
    StartingCompression,
    /// Invalid damping multiplier.
    #[error("compression and rebound damping multipliers must be finite and between 0 and 100")]
    Damping,
    /// Existing attachment cannot be moved implicitly.
    #[error("release the opposite attachment before changing mount spacing")]
    AttachedSpacing,
}

fn dimension(value: f32, min: f32, max: f32, label: &'static str) -> Result<(), SuspensionError> {
    if !value.is_finite()
        || value < min - EPS
        || value > max + EPS
        || (value / TICK - (value / TICK).round()).abs() > 0.0005
    {
        return Err(SuspensionError::Dimension(label));
    }
    Ok(())
}

/// Independent spring inputs; constructors and deserialization validate them.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "SpringInputs", into = "SpringInputs")]
pub struct SpringSpec(SpringInputs);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct SpringInputs {
    length: f32,
    od: f32,
    id: f32,
    coils: u8,
    preload: f32,
}

impl SpringSpec {
    /// Constructs a spring. Length is fully extended mount spacing, in metres;
    /// preload adds natural length beyond it.
    ///
    /// # Errors
    /// Rejects off-grid dimensions, invalid coils, solid height, or travel.
    pub fn new(
        length: f32,
        od: f32,
        id: f32,
        coils: u8,
        preload: f32,
    ) -> Result<Self, SuspensionError> {
        dimension(length, 0.25, 8.0, "spring length")?;
        dimension(od, 0.05, 0.4, "spring OD")?;
        dimension(id, 0.02, 0.4, "spring ID")?;
        dimension(preload, 0.0, 8.0 - length, "spring preload")?;
        if !(3..=16).contains(&coils) {
            return Err(SuspensionError::Coils);
        }
        let spec = Self(SpringInputs {
            length,
            od,
            id,
            coils,
            preload,
        });
        if spec.wire() < 0.002 - EPS || spec.wire() > 0.09 + EPS {
            return Err(SuspensionError::Wire);
        }
        let plate = MountPlates::new(od);
        if length - 2.0 * plate.thickness - spec.solid_height() < MIN_TRAVEL - EPS {
            return Err(SuspensionError::Travel);
        }
        if spec.solid_height() + 2.0 * plate.thickness < 0.25 * spec.natural_length() - EPS {
            return Err(SuspensionError::SolidHeight);
        }
        Ok(spec)
    }
    /// Fully extended mount spacing in metres.
    pub const fn length(self) -> f32 {
        self.0.length
    }
    /// Outer coil diameter in metres.
    pub const fn od(self) -> f32 {
        self.0.od
    }
    /// Inner coil diameter in metres.
    pub const fn id(self) -> f32 {
        self.0.id
    }
    /// Active turns; two dead end turns are additional.
    pub const fn coils(self) -> u8 {
        self.0.coils
    }
    /// Extra natural length, in metres.
    pub const fn preload(self) -> f32 {
        self.0.preload
    }
    /// Natural mount-to-mount length, in metres.
    pub fn natural_length(self) -> f32 {
        self.length() + self.preload()
    }
    /// Round wire diameter in metres.
    pub fn wire(self) -> f32 {
        (self.od() - self.id()) / 2.0
    }
    /// Mean coil diameter in metres.
    pub fn mean_diameter(self) -> f32 {
        self.od().midpoint(self.id())
    }
    /// Wire stack at coil contact, excluding plates, in metres.
    pub fn solid_height(self) -> f32 {
        f32::from(self.coils() + 2) * self.wire()
    }
    /// Linear stiffness in N/m.
    pub fn rate(self) -> f32 {
        79.3e9 * self.wire().powi(4)
            / (8.0 * self.mean_diameter().powi(3) * f32::from(self.coils()))
    }
    /// Wire mass excluding shared plates, in kg.
    pub fn mass(self, plates: MountPlates) -> f32 {
        let turns = f32::from(self.coils() + 2);
        let pitch = (self.natural_length() - 2.0 * plates.thickness) / turns;
        turns * (PI * self.mean_diameter()).hypot(pitch) * PI * (self.wire() / 2.0).powi(2) * 7850.0
    }
}
impl Default for SpringSpec {
    fn default() -> Self {
        Self::new(0.5, 0.16, 0.12, 6, 0.0).expect("default spring")
    }
}
impl TryFrom<SpringInputs> for SpringSpec {
    type Error = SuspensionError;
    fn try_from(v: SpringInputs) -> Result<Self, Self::Error> {
        Self::new(v.length, v.od, v.id, v.coils, v.preload)
    }
}
impl From<SpringSpec> for SpringInputs {
    fn from(v: SpringSpec) -> Self {
        v.0
    }
}

/// Mount to which the rigid shock body belongs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShockBodyEnd {
    /// Source mount.
    #[default]
    Source,
    /// Opposite mount.
    Opposite,
}

/// Independent shock inputs.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ShockInputs", into = "ShockInputs")]
pub struct ShockSpec(ShockInputs);
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct ShockInputs {
    length: f32,
    od: f32,
    body_end: ShockBodyEnd,
    starting_compression: f32,
    compression: f32,
    rebound: f32,
}
impl ShockSpec {
    /// Constructs an independently packaged shock, with SI dimensions.
    ///
    /// # Errors
    /// Rejects invalid dimensions, damping, or insufficient contained stroke.
    pub fn new(
        length: f32,
        od: f32,
        body_end: ShockBodyEnd,
        starting_compression: f32,
        compression: f32,
        rebound: f32,
    ) -> Result<Self, SuspensionError> {
        dimension(length, 0.25, 8.0, "shock length")?;
        dimension(od, 0.02, 0.4, "shock OD")?;
        dimension(starting_compression, 0.0, length, "starting compression")?;
        if !compression.is_finite()
            || !rebound.is_finite()
            || !(0.0..=100.0).contains(&compression)
            || !(0.0..=100.0).contains(&rebound)
        {
            return Err(SuspensionError::Damping);
        }
        let spec = Self(ShockInputs {
            length,
            od,
            body_end,
            starting_compression,
            compression,
            rebound,
        });
        let geometry = spec.geometry(MountPlates::new(spec.hardware_od()))?;
        if starting_compression > geometry.stroke + EPS {
            return Err(SuspensionError::StartingCompression);
        }
        Ok(spec)
    }
    /// Extended mount spacing, in metres.
    pub const fn length(self) -> f32 {
        self.0.length
    }
    /// Body cylinder diameter, excluding collars, in metres.
    pub const fn od(self) -> f32 {
        self.0.od
    }
    /// Full visible radial envelope, including the adjuster collar.
    pub fn hardware_od(self) -> f32 {
        1.1 * self.od()
    }
    /// Rigid-body owner.
    pub const fn body_end(self) -> ShockBodyEnd {
        self.0.body_end
    }
    /// Initial compression from the assembly maximum extension, in metres.
    pub const fn starting_compression(self) -> f32 {
        self.0.starting_compression
    }
    /// Piston-area coefficient times the chosen compression/rebound multiplier.
    pub fn damping(self, compressing: bool) -> f32 {
        22_000.0 * PI / 4.0
            * (self.od().powi(2) - self.shaft_diameter().powi(2))
            * if compressing {
                self.0.compression
            } else {
                self.0.rebound
            }
    }
    /// Shaft diameter derived by the game packaging rule.
    pub fn shaft_diameter(self) -> f32 {
        0.008_f32.max(0.34 * self.od())
    }
    /// Packages the largest single-stage stroke inside a rigid body.
    ///
    /// # Errors
    /// Rejects less than 50 mm stroke with the supplied shared plates.
    pub fn geometry(self, plates: MountPlates) -> Result<ShockGeometry, SuspensionError> {
        let shaft = self.shaft_diameter();
        let allowance = 0.02_f32.max(0.5 * self.od());
        let minimum_exposed = 0.01_f32.max(0.6 * shaft);
        // The gland extends 0.2 OD above the main body cylinder in the guide.
        let gland = 0.2 * self.od();
        let gap = self.length() - 2.0 * plates.thickness;
        let stroke =
            (((gap - allowance - minimum_exposed - gland) / 2.0 + EPS) / TICK).floor() * TICK;
        if stroke < MIN_TRAVEL - EPS {
            return Err(SuspensionError::Travel);
        }
        let body_length = gap - minimum_exposed - stroke - gland;
        Ok(ShockGeometry {
            stroke,
            body_length,
            gland_height: gland,
            exposed_at_extension: minimum_exposed + stroke,
            // The internal allowance is split between piston engagement and base clearance.
            shaft_length: minimum_exposed + stroke + gland + allowance / 2.0,
            hardware_allowance: allowance,
        })
    }
}
impl Default for ShockSpec {
    fn default() -> Self {
        Self::new(0.5, 0.1, ShockBodyEnd::Source, 0.0, 1.0, 1.6).expect("default shock")
    }
}
impl TryFrom<ShockInputs> for ShockSpec {
    type Error = SuspensionError;
    fn try_from(v: ShockInputs) -> Result<Self, Self::Error> {
        Self::new(
            v.length,
            v.od,
            v.body_end,
            v.starting_compression,
            v.compression,
            v.rebound,
        )
    }
}
impl From<ShockSpec> for ShockInputs {
    fn from(v: ShockSpec) -> Self {
        v.0
    }
}

/// Derived single-stage geometry, in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShockGeometry {
    /// Maximum contained insertion.
    pub stroke: f32,
    /// Rigid body cylinder length, excluding gland.
    pub body_length: f32,
    /// Gland shoulder above the cylinder.
    pub gland_height: f32,
    /// Distance from gland shoulder to shaft-side plate at extension.
    pub exposed_at_extension: f32,
    /// Fixed shaft length from its plate to its internal tip.
    pub shaft_length: f32,
    /// Unavailable internal length for piston and base hardware.
    pub hardware_allowance: f32,
}

/// Rubber stop inputs. Bore and direction come from its host shock.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "[f32; 2]", into = "[f32; 2]")]
pub struct BumpStopSpec {
    length: f32,
    od: f32,
}
impl BumpStopSpec {
    /// Constructs grid-aligned stop dimensions; assembly validation checks fit.
    ///
    /// # Errors
    /// Rejects nonfinite, off-grid, or out-of-range dimensions.
    pub fn new(length: f32, od: f32) -> Result<Self, SuspensionError> {
        dimension(length, 0.01, 8.0, "stop length")?;
        dimension(od, 0.01, 1.04, "stop OD")?;
        Ok(Self { length, od })
    }
    /// Free axial length in metres.
    pub const fn length(self) -> f32 {
        self.length
    }
    /// Base outer diameter in metres.
    pub const fn od(self) -> f32 {
        self.od
    }
    /// Maximum axial deformation, in metres.
    pub fn max_crush(self) -> f32 {
        0.55 * self.length
    }
    /// Small-strain stiffness in N/m, using the actual shaft bore.
    pub fn rate(self, bore: f32) -> f32 {
        5.5e6 * self.area(bore) / self.length
    }
    fn area(self, bore: f32) -> f32 {
        PI / 4.0 * (self.od.powi(2) - bore.powi(2)).max(0.0)
    }
    /// Progressive tangent stiffness in N/m.
    pub fn tangent(self, bore: f32, crush: f32) -> f32 {
        self.rate(bore) * (1.0 + 12.8 * (crush.clamp(0.0, self.max_crush()) / self.length).powi(2))
    }
    /// Integral of tangent stiffness, in newtons, bounded at maximum crush.
    pub fn force(self, bore: f32, crush: f32) -> f32 {
        let x = crush.clamp(0.0, self.max_crush());
        self.rate(bore) * (x + 12.8 * x.powi(3) / (3.0 * self.length.powi(2)))
    }
    /// Integral of force, in joules, bounded at maximum crush.
    pub fn energy(self, bore: f32, crush: f32) -> f32 {
        let x = crush.clamp(0.0, self.max_crush());
        self.rate(bore) * (0.5 * x.powi(2) + 12.8 * x.powi(4) / (12.0 * self.length.powi(2)))
    }
    /// Profile fill fraction from the procedural guide, excluding steel clamp.
    pub fn mass(self, bore: f32) -> f32 {
        self.area(bore) * self.length * 0.72 * 1150.0
    }
}
impl Default for BumpStopSpec {
    fn default() -> Self {
        Self::new(0.05, 0.06).expect("default stop")
    }
}
impl TryFrom<[f32; 2]> for BumpStopSpec {
    type Error = SuspensionError;
    fn try_from(v: [f32; 2]) -> Result<Self, Self::Error> {
        Self::new(v[0], v[1])
    }
}
impl From<BumpStopSpec> for [f32; 2] {
    fn from(v: BumpStopSpec) -> Self {
        [v.length, v.od]
    }
}

/// Common mounting plates, sized once from all installed components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MountPlates {
    /// One plate's axial thickness, in metres.
    pub thickness: f32,
    /// Lattice-sized outside diameter, in metres.
    pub diameter: f32,
}
impl MountPlates {
    fn new(envelope: f32) -> Self {
        Self {
            thickness: (((0.01 + envelope * 0.055) / TICK).round() * TICK).clamp(0.0125, 0.03),
            diameter: (((envelope + 0.05 - EPS) / 0.025).ceil() * 0.025).min(0.6),
        }
    }
    /// Mass of one aluminium plate, in kg.
    pub fn mass(self) -> f32 {
        PI / 4.0 * self.diameter.powi(2) * self.thickness * 2700.0
    }
}

/// Component defining the first compression limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompressionLimit {
    /// Coils touch.
    Spring,
    /// Shaft reaches its internal stop.
    Shock,
    /// Rubber reaches 55% crush.
    BumpStop,
}

/// Independent components sharing one pair of rigid mounts.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "SuspensionInputs", into = "SuspensionInputs")]
pub struct SuspensionSpec(SuspensionInputs);
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct SuspensionInputs {
    spring: Option<SpringSpec>,
    shock: Option<ShockSpec>,
    bump_stop: Option<BumpStopSpec>,
    appearances: [crate::MaterialAppearance; 3],
    mount_envelope: f32,
    initial_length: f32,
}
impl SuspensionSpec {
    /// Validates the complete hardware envelope and travel without changing inputs.
    ///
    /// # Errors
    /// Rejects incompatible radial/axial packaging or invalid initial compression.
    pub fn new(
        spring: Option<SpringSpec>,
        shock: Option<ShockSpec>,
        bump_stop: Option<BumpStopSpec>,
    ) -> Result<Self, SuspensionError> {
        let mount_envelope = spring
            .map_or(0.0, SpringSpec::od)
            .max(shock.map_or(0.0, ShockSpec::hardware_od))
            .max(bump_stop.map_or(0.0, |s| s.od * 1.02));
        let extended = spring
            .map_or(f32::INFINITY, SpringSpec::length)
            .min(shock.map_or(f32::INFINITY, ShockSpec::length));
        Self::validate_inputs(SuspensionInputs {
            spring,
            shock,
            bump_stop,
            appearances: [crate::MaterialAppearance::BAKED; 3],
            mount_envelope,
            initial_length: extended - shock.map_or(0.0, ShockSpec::starting_compression),
        })
    }

    fn validate_inputs(inputs: SuspensionInputs) -> Result<Self, SuspensionError> {
        let SuspensionInputs {
            spring,
            shock,
            bump_stop,
            mount_envelope,
            initial_length,
            ..
        } = inputs;
        if spring.is_none() && shock.is_none() {
            return Err(SuspensionError::Empty);
        }
        if bump_stop.is_some() && shock.is_none() {
            return Err(SuspensionError::MissingShock);
        }
        if let (Some(spring), Some(shock)) = (spring, shock)
            && shock.hardware_od() + 0.004 > spring.id() + EPS
        {
            return Err(SuspensionError::RadialFit);
        }
        let required_envelope = spring
            .map_or(0.0, SpringSpec::od)
            .max(shock.map_or(0.0, ShockSpec::hardware_od))
            .max(bump_stop.map_or(0.0, |s| s.od * 1.02));
        if !mount_envelope.is_finite()
            || mount_envelope < required_envelope - EPS
            || mount_envelope > 1.0608 + EPS
        {
            return Err(SuspensionError::Dimension("shared mounting envelope"));
        }
        dimension(initial_length, 0.05, 8.0, "installed mount spacing")?;
        let spec = Self(inputs);
        if let Some(shock) = shock {
            let geometry = shock.geometry(spec.plates())?;
            if let Some(stop) = bump_stop {
                if stop.length > geometry.stroke * 0.8 + EPS {
                    return Err(SuspensionError::StopLength);
                }
                // Tip must overlap the gland shoulder around its 1.05 × shaft bore.
                let tip = (stop.od / 2.0 * 0.52).max(shock.shaft_diameter() / 2.0 * 1.25);
                if stop.od < shock.shaft_diameter() + 0.006 - EPS
                    || stop.od > shock.od() * 2.6 + EPS
                    || tip <= shock.shaft_diameter() / 2.0 * 1.05
                {
                    return Err(SuspensionError::StopDiameter);
                }
                // Both rib crowns and the steel clamp participate in radial fit.
                let envelope = (stop.od * 1.02).max(shock.shaft_diameter() * 1.55);
                if spring.is_some_and(|spring| envelope + 0.004 > spring.id() + EPS) {
                    return Err(SuspensionError::RadialFit);
                }
            }
        }
        let (travel, _) = spec.compression_limit();
        if travel < MIN_TRAVEL - EPS {
            return Err(SuspensionError::Travel);
        }
        if spec.starting_compression() < -EPS || spec.starting_compression() > travel + EPS {
            return Err(SuspensionError::StartingCompression);
        }
        Ok(spec)
    }
    /// Independent component appearances: spring, shock, and rubber stop.
    pub const fn appearances(self) -> [crate::MaterialAppearance; 3] {
        self.0.appearances
    }
    /// Applies independent appearances without changing construction dimensions.
    #[must_use]
    pub const fn with_appearances(mut self, appearances: [crate::MaterialAppearance; 3]) -> Self {
        self.0.appearances = appearances;
        self
    }

    /// Installed spring.
    pub const fn spring(self) -> Option<SpringSpec> {
        self.0.spring
    }
    /// Installed shock.
    pub const fn shock(self) -> Option<ShockSpec> {
        self.0.shock
    }
    /// Installed rubber stop.
    pub const fn bump_stop(self) -> Option<BumpStopSpec> {
        self.0.bump_stop
    }
    /// Shared plate dimensions.
    pub fn plates(self) -> MountPlates {
        MountPlates::new(self.0.mount_envelope)
    }
    /// Maximum spacing allowed by both independently specified components.
    pub fn extended_length(self) -> f32 {
        self.spring()
            .map_or(f32::INFINITY, SpringSpec::length)
            .min(self.shock().map_or(f32::INFINITY, ShockSpec::length))
    }
    /// Initial compression within the common range, in metres.
    pub fn starting_compression(self) -> f32 {
        self.extended_length() - self.initial_length()
    }
    /// Placement spacing, including starting compression.
    pub fn initial_length(self) -> f32 {
        self.0.initial_length
    }
    /// Compression when rubber first touches the actual gland shoulder.
    pub fn bump_contact(self) -> Option<f32> {
        let shock = self.shock()?;
        let stop = self.bump_stop()?;
        let g = shock.geometry(self.plates()).ok()?;
        Some(g.exposed_at_extension - (shock.length() - self.extended_length()) - stop.length)
    }
    /// Common travel and its limiting component.
    pub fn compression_limit(self) -> (f32, CompressionLimit) {
        let plates = self.plates();
        let mut limit = (f32::INFINITY, CompressionLimit::Spring);
        if let Some(spring) = self.spring() {
            limit.0 = self.extended_length() - 2.0 * plates.thickness - spring.solid_height();
        }
        if let Some(shock) = self.shock() {
            let travel = shock.geometry(plates).map_or(f32::NEG_INFINITY, |g| {
                g.stroke - (shock.length() - self.extended_length())
            });
            if travel < limit.0 {
                limit = (travel, CompressionLimit::Shock);
            }
        }
        if let (Some(contact), Some(stop)) = (self.bump_contact(), self.bump_stop())
            && contact + stop.max_crush() < limit.0
        {
            limit = (contact + stop.max_crush(), CompressionLimit::BumpStop);
        }
        limit
    }
    /// Physical translation bounds around the initial construction pose.
    pub fn bounds(self) -> [f32; 2] {
        [
            self.starting_compression() - self.compression_limit().0,
            self.starting_compression(),
        ]
    }
    /// Stationary axial extension force, in newtons; hard-stop reactions are separate.
    pub fn elastic_force(self, compression: f32) -> f32 {
        let spring = self.spring().map_or(0.0, |s| {
            s.rate() * (s.natural_length() - self.extended_length() + compression).max(0.0)
        });
        spring
            + match (self.bump_stop(), self.shock(), self.bump_contact()) {
                (Some(stop), Some(shock), Some(contact)) => {
                    stop.force(shock.shaft_diameter(), compression - contact)
                }
                _ => 0.0,
            }
    }
    /// Axisymmetric component masses in the construction pose. The axial centre
    /// is measured from the source plate; inertia is about each component centre.
    /// Shared plates occur exactly once each, and wire mass is split between mounts.
    #[expect(
        clippy::missing_panics_doc,
        reason = "constructor validation guarantees shock packaging"
    )]
    pub fn mass_elements(self) -> Vec<BearingMassElement> {
        let p = self.plates();
        let length = self.initial_length();
        let mut elements = Vec::new();
        let mut add = |opposite, mass, center, radius, height| {
            elements.push(BearingMassElement {
                opposite,
                mass,
                center,
                axial_inertia: mass * radius * radius / 2.0,
                transverse_inertia: mass * (3.0 * radius * radius + height * height) / 12.0,
            });
        };
        add(
            false,
            p.mass(),
            p.thickness / 2.0,
            p.diameter / 2.0,
            p.thickness,
        );
        add(
            true,
            p.mass(),
            length - p.thickness / 2.0,
            p.diameter / 2.0,
            p.thickness,
        );
        if let Some(spring) = self.spring() {
            let height = (length - 2.0 * p.thickness) / 2.0;
            for opposite in [false, true] {
                let center = p.thickness + height / 2.0 + if opposite { height } else { 0.0 };
                add(
                    opposite,
                    spring.mass(p) / 2.0,
                    center,
                    spring.mean_diameter() / 2.0 * 2.0_f32.sqrt(),
                    height,
                );
            }
        }
        if let Some(shock) = self.shock() {
            let g = shock.geometry(p).expect("validated shock packaging");
            let opposite = shock.body_end() == ShockBodyEnd::Opposite;
            let body_mass = PI / 4.0 * shock.od().powi(2) * g.body_length * 0.42 * 7850.0;
            let body_center = p.thickness + g.body_length / 2.0;
            add(
                opposite,
                body_mass,
                if opposite {
                    length - body_center
                } else {
                    body_center
                },
                shock.od() / 2.0,
                g.body_length,
            );
            let shaft_center = p.thickness + g.shaft_length / 2.0;
            add(
                !opposite,
                PI / 4.0 * shock.shaft_diameter().powi(2) * g.shaft_length * 7850.0,
                if opposite {
                    shaft_center
                } else {
                    length - shaft_center
                },
                shock.shaft_diameter() / 2.0,
                g.shaft_length,
            );
            if let Some(stop) = self.bump_stop() {
                let center = p.thickness + stop.length() / 2.0;
                add(
                    !opposite,
                    stop.mass(shock.shaft_diameter()),
                    if opposite { center } else { length - center },
                    stop.od() / 2.0,
                    stop.length(),
                );
            }
        }
        elements
    }

    /// Packed passive force rows shared by CPU and GPU compilation.
    /// First row: spring stiffness, build-pose spring compression, compression
    /// damping, rebound damping. Second row: rubber stiffness, free length,
    /// contact compression, initial assembly compression.
    #[expect(
        clippy::missing_panics_doc,
        reason = "constructor validation guarantees every stop has a host"
    )]
    pub fn passive_rows(self) -> [[f32; 4]; 2] {
        let spring = self.spring();
        let shock = self.shock();
        [
            [
                spring.map_or(0.0, SpringSpec::rate),
                spring.map_or(0.0, |s| s.natural_length() - self.initial_length()),
                shock.map_or(0.0, |s| s.damping(true)),
                shock.map_or(0.0, |s| s.damping(false)),
            ],
            [
                self.bump_stop().map_or(0.0, |s| {
                    s.rate(shock.expect("validated stop host").shaft_diameter())
                }),
                self.bump_stop().map_or(1.0, BumpStopSpec::length),
                self.bump_contact().unwrap_or(0.0),
                self.starting_compression(),
            ],
        ]
    }

    /// Validates an edit against rigid attachments.
    ///
    /// # Errors
    /// Rejects a spacing change while the opposite mount is attached.
    pub fn validate_edit(
        self,
        replacement: Self,
        opposite_attached: bool,
    ) -> Result<(), SuspensionError> {
        if opposite_attached && (self.initial_length() - replacement.initial_length()).abs() > EPS {
            return Err(SuspensionError::AttachedSpacing);
        }
        Ok(())
    }
    /// Replaces component inputs while retaining installed plates and mount spacing.
    /// Existing length or starting-compression edits request a new spacing; insertion
    /// and removal retain the current spacing. Plates may grow but never shrink.
    ///
    /// # Errors
    /// Rejects invalid fit, unavailable travel, or a spacing edit while attached.
    pub fn with_components(
        self,
        spring: Option<SpringSpec>,
        shock: Option<ShockSpec>,
        bump_stop: Option<BumpStopSpec>,
        opposite_attached: bool,
    ) -> Result<Self, SuspensionError> {
        let fresh = Self::new(spring, shock, bump_stop)?;
        let spacing_edit = self
            .spring()
            .zip(spring)
            .is_some_and(|(a, b)| (a.length() - b.length()).abs() > EPS)
            || self.shock().zip(shock).is_some_and(|(a, b)| {
                (a.length() - b.length()).abs() > EPS
                    || (a.starting_compression() - b.starting_compression()).abs() > EPS
            });
        let replacement = Self::validate_inputs(SuspensionInputs {
            appearances: self.0.appearances,
            mount_envelope: self.0.mount_envelope.max(fresh.0.mount_envelope),
            initial_length: if spacing_edit {
                fresh.initial_length()
            } else {
                self.initial_length()
            },
            ..fresh.0
        })?;
        self.validate_edit(replacement, opposite_attached)?;
        Ok(replacement)
    }

    /// Removes spring while retaining the independently configured shock.
    ///
    /// # Errors
    /// Reports if the retained hardware or spacing cannot fit the remaining shock.
    pub fn without_spring(self) -> Result<Option<Self>, SuspensionError> {
        self.shock()
            .map(|s| self.with_components(None, Some(s), self.bump_stop(), true))
            .transpose()
    }
    /// Removes shock and its stop together; retains any spring.
    ///
    /// # Errors
    /// Reports invalid remaining spring packaging.
    pub fn without_shock(self) -> Result<Option<Self>, SuspensionError> {
        self.spring()
            .map(|s| self.with_components(Some(s), None, None, true))
            .transpose()
    }
}
impl TryFrom<SuspensionInputs> for SuspensionSpec {
    type Error = SuspensionError;
    fn try_from(v: SuspensionInputs) -> Result<Self, Self::Error> {
        Self::validate_inputs(v)
    }
}
impl From<SuspensionSpec> for SuspensionInputs {
    fn from(v: SuspensionSpec) -> Self {
        v.0
    }
}

#[cfg(test)]
#[expect(
    clippy::float_cmp,
    reason = "exact retained input bits and stationary zero forces are the contract"
)]
mod tests {
    use super::*;
    #[test]
    fn guide_reference_springs_match_rate_solid_height_and_travel() {
        for (length, od, id, coils, rate, solid, travel) in [
            (0.25, 0.06, 0.04, 5, 158.6, 0.07, 0.155),
            (0.5, 0.16, 0.12, 6, 96.3, 0.16, 0.3),
            (1.0, 0.25, 0.19, 8, 94.3, 0.3, 0.65),
            (2.5, 0.4, 0.3, 10, 144.5, 0.6, 1.84),
        ] {
            let spring = SpringSpec::new(length, od, id, coils, 0.0).unwrap();
            assert!((spring.rate() / 1000.0 - rate).abs() < 0.06);
            assert!((spring.solid_height() - solid).abs() < EPS);
            assert!(
                (SuspensionSpec::new(Some(spring), None, None)
                    .unwrap()
                    .compression_limit()
                    .0
                    - travel)
                    .abs()
                    < EPS
            );
        }
        assert_eq!(
            SpringSpec::new(0.5, 0.16, 0.15, 6, 0.0),
            Err(SuspensionError::SolidHeight)
        );
    }
    #[test]
    fn invalid_inputs_cannot_enter_through_constructors_or_deserialization() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.5, 0.5001] {
            assert!(SpringSpec::new(bad, 0.16, 0.12, 6, 0.0).is_err());
            assert!(ShockSpec::new(bad, 0.1, ShockBodyEnd::Source, 0.0, 1.0, 1.6).is_err());
            assert!(BumpStopSpec::new(bad, 0.06).is_err());
        }
        assert!(
            ron::from_str::<SpringSpec>("(length:0.5,od:0.16,id:0.15,coils:6,preload:0.0)")
                .is_err()
        );
        assert!(ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.0, f32::NAN, 1.6).is_err());
    }
    #[test]
    fn independent_shock_contains_its_shaft_throughout_stroke() {
        for od in [0.02, 0.1, 0.4] {
            for length in [1.0, 2.0, 8.0] {
                let shock =
                    ShockSpec::new(length, od, ShockBodyEnd::Source, 0.0, 1.0, 1.6).unwrap();
                let g = shock
                    .geometry(MountPlates::new(shock.hardware_od()))
                    .unwrap();
                for fraction in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    let insertion = fraction * g.stroke;
                    assert!(insertion <= g.body_length - g.hardware_allowance + EPS);
                    assert!(g.exposed_at_extension - insertion >= 0.01 - EPS);
                    let tip_depth =
                        g.shaft_length - (g.exposed_at_extension - insertion) - g.gland_height;
                    assert!(tip_depth >= g.hardware_allowance / 2.0 - EPS);
                    assert!(tip_depth <= g.body_length - g.hardware_allowance / 2.0 + EPS);
                }
                assert!((g.stroke / TICK - (g.stroke / TICK).round()).abs() < 0.001);
            }
        }
    }
    #[test]
    fn complete_collars_constrain_insertion_in_both_orders() {
        let spring = SpringSpec::default();
        let shock = ShockSpec::default();
        let a = SuspensionSpec::new(Some(spring), None, None).unwrap();
        let b = SuspensionSpec::new(None, Some(shock), None).unwrap();
        assert_eq!(
            SuspensionSpec::new(a.spring(), Some(shock), None),
            SuspensionSpec::new(Some(spring), b.shock(), None)
        );
        let fat = ShockSpec::new(0.5, 0.11, ShockBodyEnd::Source, 0.0, 1.0, 1.6).unwrap();
        assert_eq!(
            SuspensionSpec::new(Some(spring), Some(fat), None),
            Err(SuspensionError::RadialFit)
        );
    }
    #[test]
    fn preload_adds_force_and_starting_compression_only_changes_pose() {
        let spring = SpringSpec::new(0.5, 0.16, 0.12, 6, 0.025).unwrap();
        let assembly = SuspensionSpec::new(Some(spring), None, None).unwrap();
        assert!((assembly.elastic_force(0.0) - spring.rate() * 0.025).abs() < 0.01);
        let shock = ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.05, 1.0, 1.6).unwrap();
        let assembly = SuspensionSpec::new(None, Some(shock), None).unwrap();
        assert!((assembly.initial_length() - 0.45).abs() < EPS);
        assert_eq!(assembly.elastic_force(0.05), 0.0);
        assert!((shock.damping(false) / shock.damping(true) - 1.6).abs() < EPS);
    }
    #[test]
    fn bump_contact_uses_gland_and_integrated_force_matches_energy() {
        let stop = BumpStopSpec::new(0.05, 0.06).unwrap();
        let shock = ShockSpec::default();
        let assembly = SuspensionSpec::new(None, Some(shock), Some(stop)).unwrap();
        let g = shock.geometry(assembly.plates()).unwrap();
        assert!(
            (assembly.bump_contact().unwrap() - (g.exposed_at_extension - stop.length())).abs()
                < EPS
        );
        assert_eq!(assembly.compression_limit().1, CompressionLimit::BumpStop);
        let x = 0.02;
        let dx = 0.00001;
        let bore = shock.shaft_diameter();
        let energy_slope = (stop.energy(bore, x + dx) - stop.energy(bore, x - dx)) / (2.0 * dx);
        let force_slope = (stop.force(bore, x + dx) - stop.force(bore, x - dx)) / (2.0 * dx);
        assert!((energy_slope / stop.force(bore, x) - 1.0).abs() < 0.001);
        assert!((force_slope / stop.tangent(bore, x) - 1.0).abs() < 0.001);
        assert!((stop.tangent(bore, stop.length() / 2.0) / stop.rate(bore) - 4.2).abs() < EPS);
    }
    #[test]
    fn removing_components_preserves_installed_hardware_and_compressed_spacing() {
        let shock = ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.05, 1.0, 1.6).unwrap();
        let assembly = SuspensionSpec::new(Some(SpringSpec::default()), Some(shock), None).unwrap();
        for remaining in [
            assembly.without_spring().unwrap().unwrap(),
            assembly.without_shock().unwrap().unwrap(),
        ] {
            assert_eq!(remaining.plates(), assembly.plates());
            assert_eq!(remaining.initial_length(), assembly.initial_length());
            let decoded: SuspensionSpec =
                ron::from_str(&ron::to_string(&remaining).unwrap()).unwrap();
            assert_eq!(decoded, remaining);
        }
        let spring_only = assembly.without_shock().unwrap().unwrap();
        let restored = spring_only
            .with_components(spring_only.spring(), Some(shock), None, true)
            .unwrap();
        assert_eq!(restored, assembly);
    }

    #[test]
    fn replacement_rejects_attached_spacing_edits_but_allows_preload() {
        let spring = SpringSpec::default();
        let shock = ShockSpec::default();
        let assembly = SuspensionSpec::new(Some(spring), Some(shock), None).unwrap();
        let compressed = ShockSpec::new(0.5, 0.1, ShockBodyEnd::Source, 0.05, 1.0, 1.6).unwrap();
        assert_eq!(
            assembly.with_components(Some(spring), Some(compressed), None, true),
            Err(SuspensionError::AttachedSpacing)
        );
        let edited = assembly
            .with_components(Some(spring), Some(compressed), None, false)
            .unwrap();
        assert!((edited.initial_length() - 0.45).abs() < EPS);
        let preloaded = SpringSpec::new(0.5, 0.16, 0.12, 6, 0.025).unwrap();
        assert_eq!(
            assembly
                .with_components(Some(preloaded), Some(shock), None, true)
                .unwrap()
                .initial_length(),
            assembly.initial_length()
        );
    }

    #[test]
    fn persisted_mount_geometry_and_spacing_are_validated() {
        let assembly = SuspensionSpec::new(Some(SpringSpec::default()), None, None).unwrap();
        for envelope in [f32::NAN, 0.01, 2.0] {
            assert!(
                SuspensionSpec::validate_inputs(SuspensionInputs {
                    mount_envelope: envelope,
                    ..assembly.0
                })
                .is_err()
            );
        }
        for initial_length in [f32::NAN, 0.1, 0.6] {
            assert!(
                SuspensionSpec::validate_inputs(SuspensionInputs {
                    initial_length,
                    ..assembly.0
                })
                .is_err()
            );
        }
    }

    #[test]
    fn removal_and_round_trip_preserve_independent_inputs() {
        let assembly = SuspensionSpec::new(
            Some(SpringSpec::default()),
            Some(ShockSpec::default()),
            Some(BumpStopSpec::new(0.05, 0.06).unwrap()),
        )
        .unwrap();
        let decoded: SuspensionSpec = ron::from_str(&ron::to_string(&assembly).unwrap()).unwrap();
        assert_eq!(assembly, decoded);
        let spring_only = assembly.without_shock().unwrap().unwrap();
        assert_eq!(spring_only.spring(), assembly.spring());
        assert_eq!(spring_only.bump_stop(), None);
        assert_eq!(
            assembly.without_spring().unwrap().unwrap().shock(),
            assembly.shock()
        );
        assert_eq!(spring_only.without_spring().unwrap(), None);
    }
}
