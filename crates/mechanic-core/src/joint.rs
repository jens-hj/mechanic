//! The family of one-degree-of-freedom joints a construction can carry.

use bevy_math::Vec3;
use serde::{Deserialize, Serialize};

/// Physical one-dimensional motion permitted by a joint, and the hardware providing it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum JointKind {
    /// Unbounded rotation about the source-face normal.
    #[default]
    Rotational,
    /// Bounded translation along the rail.
    Linear(crate::LinearBearing),
    /// Passive axial suspension with rigid mount orientations.
    Suspension(crate::SuspensionSpec),
    /// Telescopic extension from the collapsed build pose.
    Piston(crate::Piston),
}

/// One axisymmetric mass contribution of joint hardware, assigned to a rigid mount.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointMassElement {
    /// True for the opposite attachment; false for the source.
    pub opposite: bool,
    /// Mass in kg.
    pub mass: f32,
    /// Axial centre relative to the hardware's mass origin, in metres.
    pub center: f32,
    /// Inertia about the bearing axis, in kg·m².
    pub axial_inertia: f32,
    /// Inertia about either transverse axis, in kg·m².
    pub transverse_inertia: f32,
}

impl JointKind {
    /// Whether the physical coordinate is translation, in metres.
    pub const fn is_translational(self) -> bool {
        !matches!(self, Self::Rotational)
    }

    /// Physical coordinate bounds independent of drive programming.
    pub fn bounds(self) -> [f32; 2] {
        match self {
            Self::Rotational => [f32::NEG_INFINITY, f32::INFINITY],
            Self::Linear(rail) => rail.dimensions.bounds(),
            Self::Suspension(spec) => spec.bounds(),
            Self::Piston(piston) => piston.dimensions.bounds(),
        }
    }

    /// Whether a Controller wire can drive this joint. Suspension is passive.
    pub const fn accepts_drive(self) -> bool {
        !matches!(self, Self::Suspension(_))
    }

    /// Whether drive targets are offsets from a collapsed pose rather than
    /// signed displacements about a centre, so reversing a wire has no meaning.
    pub const fn is_one_sided(self) -> bool {
        matches!(self, Self::Piston(_))
    }

    /// Whether the hardware's moving side is a body of its own, so the joint
    /// exists before anything is attached to it.
    pub const fn owns_head(self) -> bool {
        matches!(self, Self::Piston(_))
    }

    /// Carries the world-space mounting frame through a rotation of construction space.
    pub fn rotate(&mut self, rotate: impl Fn(Vec3) -> Vec3) {
        match self {
            Self::Linear(rail) => rail.mount_normal = rotate(rail.mount_normal),
            Self::Piston(crate::Piston {
                mount: crate::PistonMount::Side { mount_normal },
                ..
            }) => *mount_normal = rotate(*mount_normal),
            Self::Rotational | Self::Suspension(_) | Self::Piston(_) => {}
        }
    }

    /// Mass the hardware itself adds to its two mounts; empty for weightless kinds.
    pub fn mass_elements(self) -> Vec<JointMassElement> {
        match self {
            Self::Rotational | Self::Linear(_) => Vec::new(),
            Self::Suspension(spec) => spec.mass_elements(),
            Self::Piston(piston) => piston.dimensions.mass_elements(),
        }
    }

    /// Point on the axis that [`JointMassElement::center`] is measured from.
    pub fn mass_origin(self, anchor: Vec3, axis: Vec3) -> Vec3 {
        match self {
            Self::Piston(piston) => piston.base_center(anchor, axis),
            Self::Rotational | Self::Linear(_) | Self::Suspension(_) => anchor,
        }
    }
}
