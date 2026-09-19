//! Dimensions, mounting frames, and mass of the telescopic piston.
//!
//! A piston is one block square, a whole number of blocks long closed, and every
//! stage draws its own whole length, so the extended length always lands on a
//! block line.

use crate::{GRID_UNIT_METERS, JointMassElement};
use bevy_math::{Mat3, Quat, Vec3};
use core::f32::consts::PI;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Density of the steel barrel and stages, in kg/m³.
const STEEL_DENSITY_KG_M3: f32 = 7850.0;
/// Density of the aluminium head crown, in kg/m³.
const ALUMINIUM_DENSITY_KG_M3: f32 = 2700.0;

/// Invalid piston dimensions or mounting frame.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum PistonError {
    /// Closed length must be a supported whole number of blocks.
    #[error("piston closed length must be between 2 and 8 blocks")]
    Blocks,
    /// Stage count must be supported.
    #[error("piston must have between 1 and 6 stages")]
    Stages,
    /// The mounting frame must be orthonormal.
    #[error("piston travel and mounting axes must be perpendicular unit vectors")]
    Frame,
}

/// Validated closed length and stage count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "[u8; 2]", into = "[u8; 2]")]
pub struct PistonDimensions {
    blocks: u8,
    stages: u8,
}

impl PistonDimensions {
    /// Side of the square envelope, in metres. Fixed at every length.
    pub const SECTION: f32 = GRID_UNIT_METERS;
    /// Shortest closed length, in blocks.
    pub const MIN_BLOCKS: u8 = 2;
    /// Longest closed length, in blocks.
    pub const MAX_BLOCKS: u8 = 8;
    /// Fewest nested stages.
    pub const MIN_STAGES: u8 = 1;
    /// Most nested stages.
    pub const MAX_STAGES: u8 = 6;
    /// Stage wall thickness, in metres.
    pub const WALL: f32 = 0.008;
    /// Radial clearance between one stage and the next, in metres.
    pub const CLEARANCE: f32 = 0.0015;
    /// Thickness of the head crown on the last stage, in metres.
    pub const CROWN: f32 = 0.014;
    /// Attachment lattice pitch on the head crown, in metres.
    pub const ATTACHMENT_PITCH: f32 = 0.025;
    /// Axial length of one saddle bracket, in metres.
    pub const SADDLE_LENGTH: f32 = 0.10;

    /// Creates dimensions from whole blocks and stages.
    ///
    /// # Errors
    /// Returns an error when either count is outside its supported range.
    pub const fn new(blocks: u8, stages: u8) -> Result<Self, PistonError> {
        if blocks < Self::MIN_BLOCKS || blocks > Self::MAX_BLOCKS {
            return Err(PistonError::Blocks);
        }
        if stages < Self::MIN_STAGES || stages > Self::MAX_STAGES {
            return Err(PistonError::Stages);
        }
        Ok(Self { blocks, stages })
    }

    /// Closed length in whole blocks.
    pub const fn blocks(self) -> u8 {
        self.blocks
    }

    /// Number of nested stages.
    pub const fn stages(self) -> u8 {
        self.stages
    }

    /// Closed length, which is also the draw of every stage, in metres.
    pub fn closed(self) -> f32 {
        f32::from(self.blocks) * GRID_UNIT_METERS
    }

    /// Total head travel, in metres.
    pub fn stroke(self) -> f32 {
        f32::from(self.stages) * self.closed()
    }

    /// Overall length at full draw, in metres.
    pub fn extended(self) -> f32 {
        self.closed() + self.stroke()
    }

    /// Overall length at full draw, in whole blocks.
    pub const fn extended_blocks(self) -> u8 {
        self.blocks * (self.stages + 1)
    }

    /// Outer diameter of stage `stage`, in metres; stage zero is the body.
    pub fn section(self, stage: u8) -> f32 {
        Self::SECTION - 2.0 * f32::from(stage) * (Self::WALL + Self::CLEARANCE)
    }

    /// Radius of the head crown, the only face that moves, in metres.
    pub fn head_radius(self) -> f32 {
        self.section(self.stages) / 2.0
    }

    /// Head displacement limits relative to the collapsed build pose.
    pub fn bounds(self) -> [f32; 2] {
        [0.0, self.stroke()]
    }

    /// Axial offset of every stage at a head extension, in metres.
    ///
    /// Stages draw largest first: a telescopic ram fills its biggest chamber
    /// before the next one moves. Entry `i` is stage `i + 1`; entries past the
    /// stage count repeat the head offset.
    pub fn stage_offsets(self, extension: f32) -> [f32; Self::MAX_STAGES as usize] {
        let extension = extension.clamp(0.0, self.stroke());
        let draw = self.closed();
        let mut offsets = [0.0; Self::MAX_STAGES as usize];
        let mut total = 0.0;
        for (index, offset) in (0_u8..).zip(&mut offsets) {
            if index < self.stages {
                total += (extension - f32::from(index) * draw).clamp(0.0, draw);
            }
            *offset = total;
        }
        offsets
    }

    /// Axial positions of the two saddle brackets of a side mount, from the base.
    pub fn saddle_positions(self) -> [f32; 2] {
        [Self::SADDLE_LENGTH, self.closed() - Self::SADDLE_LENGTH]
    }

    /// Mass contributions at the collapsed pose, measured along the axis from the base.
    ///
    /// The body and every intermediate stage ride with the source mount; the
    /// solid last stage and its crown ride with whatever the head carries.
    pub fn mass_elements(self) -> Vec<JointMassElement> {
        let closed = self.closed();
        let tube = |stage: u8| {
            let outer = self.section(stage) / 2.0;
            let inner = outer - Self::WALL;
            let mass = PI * (outer * outer - inner * inner) * closed * STEEL_DENSITY_KG_M3;
            let radii = outer * outer + inner * inner;
            JointMassElement {
                opposite: false,
                mass,
                center: closed / 2.0,
                axial_inertia: mass * radii / 2.0,
                transverse_inertia: mass * (3.0 * radii + closed * closed) / 12.0,
            }
        };
        let radius = self.head_radius();
        let solid = |mass: f32, center: f32, height: f32| JointMassElement {
            opposite: true,
            mass,
            center,
            axial_inertia: mass * radius * radius / 2.0,
            transverse_inertia: mass * (3.0 * radius * radius + height * height) / 12.0,
        };
        let area = PI * radius * radius;
        (0..self.stages)
            .map(tube)
            .chain([
                solid(area * closed * STEEL_DENSITY_KG_M3, closed / 2.0, closed),
                solid(
                    area * Self::CROWN * ALUMINIUM_DENSITY_KG_M3,
                    closed - Self::CROWN / 2.0,
                    Self::CROWN,
                ),
            ])
            .collect()
    }
}

impl Default for PistonDimensions {
    fn default() -> Self {
        Self {
            blocks: 2,
            stages: 4,
        }
    }
}

impl TryFrom<[u8; 2]> for PistonDimensions {
    type Error = PistonError;
    fn try_from(value: [u8; 2]) -> Result<Self, Self::Error> {
        Self::new(value[0], value[1])
    }
}

impl From<PistonDimensions> for [u8; 2] {
    fn from(value: PistonDimensions) -> Self {
        [value.blocks, value.stages]
    }
}

/// How the body is fixed to its supporting face.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum PistonMount {
    /// The base rings sit on the face and the piston extends along its normal.
    #[default]
    End,
    /// Saddle brackets hold the body flat on the face; it extends parallel to it.
    Side {
        /// World-space normal of the supporting face, perpendicular to travel.
        mount_normal: Vec3,
    },
}

/// A piston's dimensions and mount.
///
/// An end mount anchors at the base centre. A side mount anchors at the centre
/// of the collapsed body's footprint on the supporting face.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Piston {
    /// Closed length and stage count.
    pub dimensions: PistonDimensions,
    /// Supporting-face arrangement.
    pub mount: PistonMount,
}

impl Piston {
    /// Validates the frame and returns the local-to-world rotation.
    ///
    /// Travel is local Y. A side mount's supporting normal is local negative Z,
    /// so the body's skin pads face the support and away from it.
    ///
    /// # Errors
    /// Returns an error unless the axes form an orthonormal frame.
    pub fn rotation(self, travel_axis: Vec3) -> Result<Quat, PistonError> {
        let unit = |axis: Vec3| axis.is_finite() && (axis.length_squared() - 1.0).abs() <= 1.0e-5;
        if !unit(travel_axis) {
            return Err(PistonError::Frame);
        }
        match self.mount {
            PistonMount::End => Ok(Quat::from_rotation_arc(Vec3::Y, travel_axis)),
            PistonMount::Side { mount_normal } => {
                if !unit(mount_normal) || travel_axis.dot(mount_normal).abs() > 1.0e-5 {
                    return Err(PistonError::Frame);
                }
                Ok(Quat::from_mat3(&Mat3::from_cols(
                    travel_axis.cross(mount_normal),
                    travel_axis,
                    mount_normal,
                )))
            }
        }
    }

    /// Centre of the base face for an anchor and travel axis.
    pub fn base_center(self, anchor: Vec3, travel_axis: Vec3) -> Vec3 {
        match self.mount {
            PistonMount::End => anchor,
            PistonMount::Side { mount_normal } => {
                anchor + mount_normal * (PistonDimensions::SECTION / 2.0)
                    - travel_axis * (self.dimensions.closed() / 2.0)
            }
        }
    }

    /// Centre of the head crown at a head extension.
    pub fn head_center(self, anchor: Vec3, travel_axis: Vec3, extension: f32) -> Vec3 {
        self.base_center(anchor, travel_axis) + travel_axis * (self.dimensions.closed() + extension)
    }
}

#[cfg(test)]
mod tests;
