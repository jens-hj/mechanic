//! Gear teeth on cylinders and racks on cuboid faces.
//!
//! Teeth are a feature the envelope hides. For placement, welds, bearings,
//! picking, mass and collision a toothed cylinder is its cylinder and a rack is
//! its cuboid; only rendering and the mesh between two toothed parts see the
//! teeth. Meshing gears never touch: they follow each other through a magnetic
//! coupling, so the pitch geometry only has to be close.

use super::face::FaceKind;
use super::grid::{Axis, BuildPose, POSITION_TICK_METERS};
use super::spiral::SpiralEnd;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Share of a tooth pitch from a tooth's leading root to its centre, as the
/// teeth are drawn: a quarter pitch of root, then the flank up to the tip. It
/// is also half the width of a tooth's base.
pub const GEAR_TOOTH_CENTER_FRACTION: f32 = 0.375;

/// Smallest tooth module, in position ticks (5 mm).
pub const MIN_GEAR_MODULE_TICKS: u8 = 2;

/// Largest tooth module, in position ticks (5 cm).
pub const MAX_GEAR_MODULE_TICKS: u8 = 20;

/// Fewest teeth on one gear.
pub const MIN_GEAR_TEETH: u16 = 6;

/// Most teeth on one gear.
pub const MAX_GEAR_TEETH: u16 = 800;

/// Height of a tooth above its pitch circle, in modules.
pub const GEAR_ADDENDUM_MODULES: f32 = 1.0;

/// Depth of a tooth below its pitch circle, in modules.
pub const GEAR_DEDENDUM_MODULES: f32 = 1.25;

/// Invalid gear teeth or rack.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum GearError {
    /// The module was outside the supported range.
    #[error("a gear module is between 5 mm and 5 cm in 2.5 mm steps")]
    ModuleOutOfRange,
    /// The tooth count was outside the supported range.
    #[error("a gear has between {MIN_GEAR_TEETH} and {MAX_GEAR_TEETH} teeth")]
    TeethOutOfRange,
    /// A bevel gear's pitch cone was flat or a cylinder.
    #[error("a bevel gear's cone angle is between 1 and 89 degrees")]
    ConeAngleOutOfRange,
    /// The cylinder is a partial sector.
    #[error("only a full cylinder takes teeth")]
    PartialSector,
    /// The part carries material layers.
    #[error("a layered part does not take teeth")]
    Layered,
    /// The cylinder carries a spiral.
    #[error("a spiral cylinder does not take teeth; it meshes as a worm")]
    Spiralled,
    /// Internal teeth need a bore to sit in.
    #[error("internal teeth need a hollow cylinder")]
    BoreRequired,
    /// The cylinder's diameter is not the one these teeth reach.
    #[error("the cylinder's diameter must equal the teeth's tip diameter")]
    EnvelopeMismatch,
    /// The rack's tooth direction lies along its face normal.
    #[error("rack teeth run along the face, not into it")]
    RackAlongNormal,
}

/// Which surface the teeth are cut on and how it is shaped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GearKind {
    /// Straight teeth on the outer wall.
    Spur,
    /// Straight teeth on the bore, pointing inward: a ring gear.
    Internal,
    /// Teeth on a cone, for meshing across intersecting axes.
    Bevel {
        /// Half-angle of the pitch cone, in whole degrees.
        cone_angle_degrees: u8,
        /// The end the cone's large face sits at.
        large_end: SpiralEnd,
    },
}

/// Teeth cut into a cylinder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GearSpec {
    module_ticks: u8,
    teeth: u16,
    kind: GearKind,
}

impl GearSpec {
    /// Creates teeth, checking everything that does not depend on the
    /// cylinder they go on.
    ///
    /// # Errors
    ///
    /// Returns [`GearError`] when the module, tooth count, or cone angle is
    /// out of range.
    pub fn new(module_ticks: u8, teeth: u16, kind: GearKind) -> Result<Self, GearError> {
        if !(MIN_GEAR_MODULE_TICKS..=MAX_GEAR_MODULE_TICKS).contains(&module_ticks) {
            return Err(GearError::ModuleOutOfRange);
        }
        if !(MIN_GEAR_TEETH..=MAX_GEAR_TEETH).contains(&teeth) {
            return Err(GearError::TeethOutOfRange);
        }
        if let GearKind::Bevel {
            cone_angle_degrees, ..
        } = kind
            && !(1..=89).contains(&cone_angle_degrees)
        {
            return Err(GearError::ConeAngleOutOfRange);
        }
        Ok(Self {
            module_ticks,
            teeth,
            kind,
        })
    }

    /// Tooth size, in position ticks.
    pub const fn module_ticks(self) -> u8 {
        self.module_ticks
    }

    /// Tooth size, in metres: the pitch diameter per tooth.
    pub fn module_meters(self) -> f32 {
        f32::from(self.module_ticks) * POSITION_TICK_METERS
    }

    /// Number of teeth.
    pub const fn teeth(self) -> u16 {
        self.teeth
    }

    /// Where and how the teeth are cut.
    pub const fn kind(self) -> GearKind {
        self.kind
    }

    /// Whether the teeth point inward from a bore.
    pub const fn is_internal(self) -> bool {
        matches!(self.kind, GearKind::Internal)
    }

    /// The same teeth with another count.
    ///
    /// # Errors
    ///
    /// Returns [`GearError::TeethOutOfRange`] outside the supported range.
    pub fn with_teeth(self, teeth: u16) -> Result<Self, GearError> {
        Self::new(self.module_ticks, teeth, self.kind)
    }

    /// The same teeth at another size.
    ///
    /// # Errors
    ///
    /// Returns [`GearError::ModuleOutOfRange`] outside the supported range.
    pub fn with_module_ticks(self, module_ticks: u8) -> Result<Self, GearError> {
        Self::new(module_ticks, self.teeth, self.kind)
    }

    /// The same teeth on another surface.
    ///
    /// # Errors
    ///
    /// Returns [`GearError::ConeAngleOutOfRange`] for a flat or cylindrical cone.
    pub fn with_kind(self, kind: GearKind) -> Result<Self, GearError> {
        Self::new(self.module_ticks, self.teeth, kind)
    }

    /// Diameter of the pitch circle, at the large end of a bevel gear.
    pub fn pitch_diameter(self) -> f32 {
        self.module_meters() * f32::from(self.teeth)
    }

    /// Radius of the pitch circle, at the large end of a bevel gear.
    pub fn pitch_radius(self) -> f32 {
        self.pitch_diameter() * 0.5
    }

    /// Distance between neighbouring teeth along the pitch circle.
    pub fn circular_pitch(self) -> f32 {
        self.module_meters() * core::f32::consts::PI
    }

    /// Diameter the tooth tips reach: the cylinder's outer diameter for
    /// external teeth, its bore for internal ones.
    pub fn tip_diameter(self) -> f32 {
        let addendum = 2.0 * GEAR_ADDENDUM_MODULES * self.module_meters();
        if self.is_internal() {
            self.pitch_diameter() - addendum
        } else {
            self.pitch_diameter() + addendum
        }
    }

    /// Diameter at the bottom of the tooth gaps.
    pub fn root_diameter(self) -> f32 {
        let dedendum = 2.0 * GEAR_DEDENDUM_MODULES * self.module_meters();
        if self.is_internal() {
            self.pitch_diameter() + dedendum
        } else {
            self.pitch_diameter() - dedendum
        }
    }

    /// Tooth count that puts the tip diameter closest to `diameter` for this
    /// module and kind, within the supported range.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to the supported tooth range"
    )]
    pub fn teeth_for_tip_diameter(module_ticks: u8, kind: GearKind, diameter: f32) -> u16 {
        let module = f32::from(module_ticks) * POSITION_TICK_METERS;
        let addendum = 2.0 * GEAR_ADDENDUM_MODULES;
        let teeth = if matches!(kind, GearKind::Internal) {
            diameter / module + addendum
        } else {
            diameter / module - addendum
        };
        (teeth.round().max(0.0) as u16).clamp(MIN_GEAR_TEETH, MAX_GEAR_TEETH)
    }
}

/// Teeth cut into one face of a cuboid, running along one of its axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RackSpec {
    module_ticks: u8,
    face: FaceKind,
    along: Axis,
}

impl RackSpec {
    /// Creates a rack.
    ///
    /// # Errors
    ///
    /// Returns [`GearError`] when the module is out of range or the teeth
    /// would run into the face instead of along it.
    pub fn new(module_ticks: u8, face: FaceKind, along: Axis) -> Result<Self, GearError> {
        if !(MIN_GEAR_MODULE_TICKS..=MAX_GEAR_MODULE_TICKS).contains(&module_ticks) {
            return Err(GearError::ModuleOutOfRange);
        }
        if face.axis() == along {
            return Err(GearError::RackAlongNormal);
        }
        Ok(Self {
            module_ticks,
            face,
            along,
        })
    }

    /// Tooth size, in position ticks.
    pub const fn module_ticks(self) -> u8 {
        self.module_ticks
    }

    /// Tooth size, in metres.
    pub fn module_meters(self) -> f32 {
        f32::from(self.module_ticks) * POSITION_TICK_METERS
    }

    /// The face the teeth are cut into.
    pub const fn face(self) -> FaceKind {
        self.face
    }

    /// The local axis the teeth are spaced along.
    pub const fn along(self) -> Axis {
        self.along
    }

    /// Distance between neighbouring teeth.
    pub fn pitch(self) -> f32 {
        self.module_meters() * core::f32::consts::PI
    }

    /// Depth of the pitch line below the face.
    pub fn pitch_depth(self) -> f32 {
        GEAR_ADDENDUM_MODULES * self.module_meters()
    }

    /// Depth of the tooth gaps below the face.
    pub fn root_depth(self) -> f32 {
        (GEAR_ADDENDUM_MODULES + GEAR_DEDENDUM_MODULES) * self.module_meters()
    }

    /// Where a tooth centre lies along the face, from the cuboid's centre,
    /// for a cuboid at `pose`. Teeth are spaced from the construction frame's
    /// origin rather than the block's, so racks on neighbouring blocks of one
    /// body continue each other tooth for tooth.
    pub fn tooth_line(self, pose: BuildPose) -> f32 {
        let along = pose.rotation.quaternion() * self.along.unit();
        (-along.dot(pose.translation())).rem_euclid(self.pitch())
    }

    /// The centres of every tooth whose profile crosses a face `length` long
    /// on a cuboid at `pose`, from the cuboid's centre along the face, in
    /// order. Teeth at the ends may be cut by the face's edge.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "a small tooth index"
    )]
    pub fn tooth_centers(self, pose: BuildPose, length: f32) -> impl Iterator<Item = f32> {
        let pitch = self.pitch();
        let line = self.tooth_line(pose);
        let reach = length * 0.5 + pitch * GEAR_TOOTH_CENTER_FRACTION;
        let first = ((-reach - line) / pitch).ceil() as i32;
        let last = ((reach - line) / pitch).floor() as i32;
        (first..=last).map(move |tooth| line + tooth as f32 * pitch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tip_and_root_diameters_bracket_the_pitch_circle() {
        let gear = GearSpec::new(4, 24, GearKind::Spur).expect("valid");
        assert!((gear.module_meters() - 0.01).abs() < 1.0e-6);
        assert!((gear.pitch_diameter() - 0.24).abs() < 1.0e-6);
        assert!((gear.tip_diameter() - 0.26).abs() < 1.0e-6);
        assert!((gear.root_diameter() - 0.215).abs() < 1.0e-6);
        let ring = GearSpec::new(4, 72, GearKind::Internal).expect("valid");
        assert!((ring.tip_diameter() - 0.70).abs() < 1.0e-6);
        assert!((ring.root_diameter() - 0.745).abs() < 1.0e-6);
    }

    #[test]
    fn tooth_count_is_read_back_from_the_diameter_it_reaches() {
        let gear = GearSpec::new(4, 24, GearKind::Spur).expect("valid");
        assert_eq!(
            GearSpec::teeth_for_tip_diameter(4, GearKind::Spur, gear.tip_diameter()),
            24
        );
        let ring = GearSpec::new(4, 72, GearKind::Internal).expect("valid");
        assert_eq!(
            GearSpec::teeth_for_tip_diameter(4, GearKind::Internal, ring.tip_diameter()),
            72
        );
        assert_eq!(
            GearSpec::teeth_for_tip_diameter(2, GearKind::Spur, 0.0),
            MIN_GEAR_TEETH
        );
    }

    #[test]
    fn out_of_range_teeth_modules_and_cones_are_refused() {
        assert_eq!(
            GearSpec::new(1, 24, GearKind::Spur),
            Err(GearError::ModuleOutOfRange)
        );
        assert_eq!(
            GearSpec::new(4, 5, GearKind::Spur),
            Err(GearError::TeethOutOfRange)
        );
        let flat = GearKind::Bevel {
            cone_angle_degrees: 0,
            large_end: SpiralEnd::PositiveY,
        };
        assert_eq!(
            GearSpec::new(4, 24, flat),
            Err(GearError::ConeAngleOutOfRange)
        );
        assert_eq!(
            RackSpec::new(4, FaceKind::PositiveY, Axis::Y),
            Err(GearError::RackAlongNormal)
        );
    }

    #[test]
    fn rack_teeth_continue_across_neighbouring_blocks() {
        use super::super::grid::GridRotation;
        use bevy_math::IVec3;
        let rack = RackSpec::new(4, FaceKind::PositiveY, Axis::X).expect("valid");
        let pitch = rack.pitch();
        // Three one-block cubes in a row, the third turned a quarter turn
        // about the face normal so its teeth run along its own Z.
        let poses = [
            (IVec3::new(0, 800, 0), GridRotation::default(), Axis::X),
            (IVec3::new(100, 800, 0), GridRotation::default(), Axis::X),
            (IVec3::new(200, 800, 0), GridRotation::new(0, 1, 0), Axis::Z),
        ];
        let mut all = Vec::new();
        for (ticks, rotation, along) in poses {
            let rack = RackSpec::new(4, FaceKind::PositiveY, along).expect("valid");
            let pose = BuildPose::from_position_ticks(ticks, rotation);
            let centers = rack.tooth_centers(pose, 0.25).collect::<Vec<_>>();
            // 25 cm holds seven whole teeth of 3.14 cm and a cut one at
            // each end.
            assert_eq!(centers.len(), 9, "{centers:?}");
            // Back in the frame, every centre lies on the one tooth line.
            let along = rotation.quaternion() * along.unit();
            for center in centers {
                let frame = (pose.translation() + along * center).x;
                assert!(
                    (frame / pitch - (frame / pitch).round()).abs() < 1.0e-4,
                    "{frame}"
                );
                all.push(frame);
            }
        }
        all.sort_by(f32::total_cmp);
        all.dedup_by(|a, b| (*a - *b).abs() < 1.0e-4);
        // Along 75 cm the three blocks share a run of teeth one pitch apart.
        for pair in all.windows(2) {
            assert!((pair[1] - pair[0] - pitch).abs() < 1.0e-4, "{pair:?}");
        }
    }
}
